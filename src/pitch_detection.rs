use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use aubio_rs::{Pitch, PitchMode};
use log::info;
use crate::audio_stream::CircularBuffer;
use crate::fft_analysis::{FFTConfig, filter_buffer};

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
            if cfg!(target_os = "linux") {
                Pitch::new(
                    PitchMode::Yin,
                    buffer_size,
                    frames_per_buffer,
                    sample_rate,
                ).expect("Failed to create pitch detector")
            } else {
                Pitch::new(
                    PitchMode::Yinfft,
                    buffer_size,
                    frames_per_buffer,
                    sample_rate,
                ).expect("Failed to create pitch detector")
            }
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
                        if cfg!(target_os = "linux") {
                            // Use YIN algorithm for Linux with dynamic sizes
                            Pitch::new(
                                PitchMode::Yin,  // YIN mode for Linux
                                buffer_size,
                                frames_per_buffer,
                                sample_rate,
                            ).expect("Failed to create pitch detector")
                        } else {
                            // Keep Yinfft for OSX where it's working
                            Pitch::new(
                                PitchMode::Yinfft,
                                buffer_size,
                                frames_per_buffer,
                                sample_rate,
                            ).expect("Failed to create pitch detector")
                        }
                    })
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

            // Check if signal is above threshold and perform pitch detection
            if let Ok(frequency) = detectors[i].do_result(&analysis_buffer) {
                let raw_confidence = detectors[i].get_confidence();
                let confidence = raw_confidence.abs().min(1.0);
                
                // Only update if confidence is good enough
                if confidence > 0.7 {
                    let avg_factor = fft_config.lock().unwrap().averaging_factor;
                    let smoothed_freq = if pitch_results.lock().unwrap().prev_frequencies[i] > 0.0 {
                        avg_factor * pitch_results.lock().unwrap().prev_frequencies[i] + 
                        (1.0 - avg_factor) * frequency
                    } else {
                        frequency
                    };
                    
                    new_frequencies[i] = smoothed_freq;
                    new_confidences[i] = confidence;
                    pitch_results.lock().unwrap().prev_frequencies[i] = smoothed_freq;
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