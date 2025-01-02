use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use aubio_rs::{Pitch, PitchMode};
use log::info;
use crate::audio_stream::CircularBuffer;
use crate::fft_analysis::FFTConfig;

pub struct PitchResults {
    pub frequencies: Vec<f32>,
    pub confidences: Vec<f32>,
}

impl PitchResults {
    pub fn new(num_channels: usize) -> Self {
        PitchResults {
            frequencies: vec![0.0; num_channels],
            confidences: vec![0.0; num_channels],
        }
    }
}

pub fn start_pitch_detection(
    audio_buffer: Arc<RwLock<CircularBuffer>>,
    pitch_results: Arc<Mutex<PitchResults>>,
    sample_rate: u32,
    selected_channels: Vec<usize>,
    shutdown_flag: Arc<AtomicBool>,
    fft_config: Arc<Mutex<FFTConfig>>,
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

    // Create a pitch detector for each channel
    let mut detectors: Vec<Pitch> = selected_channels
        .iter()
        .map(|_| {
            Pitch::new(
                PitchMode::Yinfft,
                buffer_size,
                frames_per_buffer,
                sample_rate,
            ).expect("Failed to create pitch detector")
        })
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
                    .map(|_| {
                        Pitch::new(
                            PitchMode::Yinfft,
                            buffer_size,
                            frames_per_buffer,
                            sample_rate,
                        ).expect("Failed to create pitch detector")
                    })
                    .collect();
                
                info!("Pitch detectors recreated with new buffer size: {}, hop size: {}", 
                    buffer_size, frames_per_buffer);
            }
        }

        // Get current threshold from FFT config
        let db_threshold = fft_config.lock()
            .map(|config| config.db_threshold)
            .unwrap_or(-24.0);
        
        // Convert dB threshold to linear amplitude
        let amplitude_threshold = 10.0f32.powf((db_threshold as f32) / 20.0);

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

            // Check if signal is above threshold
            let max_amplitude = channel_data.iter()
                .map(|&x| x.abs())
                .fold(0.0f32, f32::max);

            if max_amplitude > amplitude_threshold {
                if let Ok(frequency) = detectors[i].do_result(&channel_data) {
                    let raw_confidence = detectors[i].get_confidence();
                    let amplitude_factor = (max_amplitude / amplitude_threshold).min(1.0);
                    let confidence = raw_confidence.abs().min(1.0) * amplitude_factor;
                    
                    // Only update values if we have meaningful confidence
                    if confidence > 0.1 {  // Adjust this threshold as needed
                        info!(
                            "Channel {}: Detected pitch {:.1} Hz with confidence {:.3} (raw: {:.3}, amp: {:.3}, threshold: {:.3})",
                            i + 1, frequency, confidence, raw_confidence, max_amplitude, amplitude_threshold
                        );
                        
                        new_frequencies[i] = frequency;
                        new_confidences[i] = confidence;
                    }
                    // If confidence is too low, keep previous values by not updating
                }
                // If pitch detection fails, keep previous values by not updating
            }
            // If amplitude is below threshold, keep previous values by not updating
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