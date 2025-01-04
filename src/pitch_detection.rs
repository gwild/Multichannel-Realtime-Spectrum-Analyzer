use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use log::info;
use crate::audio_stream::CircularBuffer;
use crate::fft_analysis::{FFTConfig, filter_buffer};
use crate::plot::SpectrumApp;
use pitch_detector::{
    pitch::{HannedFftDetector, PitchDetector},
    note::{detect_note},
    core::NoteName,
};

pub struct PitchResults {
    pub frequencies: Vec<f32>,
    pub confidences: Vec<f32>,
    prev_frequencies: Vec<f32>,
}

impl PitchResults {
    pub fn new(num_channels: usize) -> Self {
        PitchResults {
            frequencies: vec![0.0; num_channels],
            confidences: vec![0.0; num_channels],
            prev_frequencies: vec![0.0; num_channels],
        }
    }
}

fn is_valid_harmonic_relationship(frequency: f32, partials: &[(f32, f32)]) -> (bool, f32) {
    if partials.is_empty() {
        return (true, 1.0);  // No partials to validate against
    }

    // Sort partials by amplitude to get strongest ones
    let mut strong_partials = partials.to_vec();
    strong_partials.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut best_match: f32 = 0.0;
    // Check if frequency matches harmonic series of strongest partials
    for &(partial_freq, _) in strong_partials.iter().take(3) {  // Check against top 3 partials
        // Check if frequency is a harmonic or sub-harmonic
        for n in 1..=8 {  // Check up to 8th harmonic
            let harmonic = partial_freq * n as f32;
            let sub_harmonic = partial_freq / n as f32;
            
            // Calculate how close we are to a harmonic
            let harmonic_error = ((frequency - harmonic).abs() / harmonic).min(
                (frequency - sub_harmonic).abs() / sub_harmonic
            );
            
            // Convert error to confidence (1.0 = perfect match, 0.0 = far off)
            let match_confidence = (1.0 - (harmonic_error / 0.03)).max(0.0);
            best_match = best_match.max(match_confidence);
        }
    }
    
    (best_match > 0.0, best_match)
}

pub fn start_pitch_detection(
    audio_buffer: Arc<RwLock<CircularBuffer>>,
    pitch_results: Arc<Mutex<PitchResults>>,
    sample_rate: u32,
    selected_channels: Vec<usize>,
    shutdown_flag: Arc<AtomicBool>,
    fft_config: Arc<Mutex<FFTConfig>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
) {
    // Get initial buffer size and frames per buffer
    let mut buffer_size = audio_buffer.read()
        .map(|buffer| buffer.clone_data().len() / selected_channels.len())
        .unwrap_or(4096);
    
    let mut frames_per_buffer = fft_config.lock()
        .map(|config| config.frames_per_buffer as usize)
        .unwrap_or(1024);

    info!("Starting pitch detection thread with buffer size: {}, hop size: {}", 
        buffer_size, frames_per_buffer);

    // Create a detector for each channel
    let mut detectors: Vec<HannedFftDetector> = selected_channels
        .iter()
        .map(|_| HannedFftDetector::default())
        .collect();

    while !shutdown_flag.load(Ordering::SeqCst) {
        // Check if buffer size has changed
        if let Ok(buffer) = audio_buffer.read() {
            let new_buffer_size = buffer.clone_data().len() / selected_channels.len();
            if new_buffer_size != buffer_size {
                buffer_size = new_buffer_size;
                if let Ok(config) = fft_config.lock() {
                    frames_per_buffer = config.frames_per_buffer as usize;
                }
                
                // Recreate detectors with new buffer size
                detectors = selected_channels
                    .iter()
                    .map(|_| HannedFftDetector::default())
                    .collect();
                
                info!("Pitch detectors recreated with buffer size: {}, hop size: {}", 
                    buffer_size, frames_per_buffer);
            }
        }

        // Get current threshold from FFT config
        let db_threshold = fft_config.lock()
            .map(|config| config.db_threshold)
            .unwrap_or(-24.0);
        
        // Get a clone of the current audio data
        let audio_data = match audio_buffer.read() {
            Ok(buffer) => buffer.clone_data(),
            Err(_) => {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };

        if audio_data.is_empty() || audio_data.len() < buffer_size * selected_channels.len() {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Process each channel
        let mut new_frequencies = vec![0.0; selected_channels.len()];
        let mut new_confidences = vec![0.0; selected_channels.len()];

        for (i, &channel) in selected_channels.iter().enumerate() {
            let channel_data: Vec<f32> = audio_data
                .chunks(selected_channels.len())
                .take(buffer_size)
                .map(|chunk| chunk.get(channel).copied().unwrap_or(0.0))
                .collect();

            // Apply the same filtering as FFT
            let filtered_data = filter_buffer(&channel_data, db_threshold);
            if filtered_data.is_empty() {
                continue;
            }

            // Ensure we use the correct buffer size for pitch detection
            let analysis_buffer = if filtered_data.len() >= frames_per_buffer {
                filtered_data[filtered_data.len() - frames_per_buffer..].to_vec()
            } else {
                let mut padded = vec![0.0; frames_per_buffer];
                padded[frames_per_buffer - filtered_data.len()..].copy_from_slice(&filtered_data);
                padded
            };

            // Convert analysis buffer to f64 for pitch detector
            let analysis_buffer_f64: Vec<f64> = analysis_buffer.iter()
                .map(|&x| x as f64)
                .collect();

            if let Some(frequency) = detectors[i].detect_pitch(&analysis_buffer_f64, sample_rate as f64) {
                // Since get_confidence isn't available, use a fixed confidence
                let raw_confidence = 0.8;  // Default confidence
                let confidence = raw_confidence;
                
                let frequency_f32 = frequency as f32;  // Convert to f32 for later use
                
                // Get current FFT partials for validation
                let fft_partials = spectrum_app.lock()
                    .map(|app| app.partials[i].clone())
                    .unwrap_or_default();

                let (min_freq, max_freq) = fft_config.lock()
                    .map(|config| (config.min_frequency as f32, config.max_frequency as f32))
                    .unwrap_or((20.0, 2000.0));

                // Check harmonic relationship with f32
                let (is_harmonic, harmonic_confidence) = 
                    is_valid_harmonic_relationship(frequency_f32, &fft_partials);
                let combined_confidence = confidence * harmonic_confidence;

                // Only update if confidence is good enough and frequency matches harmonics
                if combined_confidence > 0.5 && 
                   frequency_f32 >= min_freq && 
                   frequency_f32 <= max_freq &&
                   is_harmonic {
                    let avg_factor = fft_config.lock().unwrap().averaging_factor;
                    let smoothed_freq = if pitch_results.lock().unwrap().prev_frequencies[i] > 0.0 {
                        avg_factor * pitch_results.lock().unwrap().prev_frequencies[i] + 
                        (1.0 - avg_factor) * frequency_f32
                    } else {
                        frequency_f32
                    };
                    
                    new_frequencies[i] = smoothed_freq;
                    new_confidences[i] = combined_confidence;
                    pitch_results.lock().unwrap().prev_frequencies[i] = smoothed_freq;
                } else {
                    // Keep previous values if confidence is low
                    if let Ok(results) = pitch_results.lock() {
                        new_frequencies[i] = results.prev_frequencies[i];
                        new_confidences[i] = 0.0;  // Indicate low confidence
                    }
                }
            }
        }

        // Update results
        if let Ok(mut results) = pitch_results.lock() {
            results.frequencies = new_frequencies;
            results.confidences = new_confidences;
        }

        thread::sleep(Duration::from_millis(10));
    }

    info!("Pitch detection thread shutting down");
}

pub fn detect_pitch(
    signal: &[f32],
    sample_rate: u32
) -> Option<(f32, f32)> {  // (frequency, confidence)
    let signal_f64: Vec<f64> = signal.iter().map(|&x| x as f64).collect();
    let mut detector = HannedFftDetector::default();
    
    if let Some(freq) = detector.detect_pitch(&signal_f64, sample_rate as f64) {
        let confidence = if freq > 0.0 { 0.8 } else { 0.0 };
        Some((freq as f32, confidence))
    } else {
        None
    }
} 