// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The following imports are protected. Any modification requires explicit permission.
use std::sync::{Arc, RwLock, Mutex};
use std::thread;
use std::time::Duration;
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::plot::SpectrumApp;
use crate::audio_stream::CircularBuffer;
use log::{info, warn, error};
use std::sync::atomic::{AtomicBool, Ordering};
use pitch_detector::{
    pitch::{HannedFftDetector, PitchDetector},
};

pub const NUM_PARTIALS: usize = 12;

/// Configuration struct for FFT settings.
pub struct FFTConfig {
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub db_threshold: f64,
    #[allow(dead_code)]
    pub num_channels: usize,
    pub averaging_factor: f32,
    pub frames_per_buffer: u32,
}

/// Spawns a thread to continuously process FFT data and update the plot.
pub fn start_fft_processing(
    audio_buffer: Arc<RwLock<CircularBuffer>>,
    fft_config: Arc<Mutex<FFTConfig>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
    sample_rate: u32,
    shutdown_flag: Arc<AtomicBool>,
) {
    let mut prev_results = vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

    thread::spawn(move || {
        while !shutdown_flag.load(Ordering::Relaxed) {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let buffer_clone = {
                    let buffer = audio_buffer.read().unwrap();
                    buffer.clone_data()
                };

                // First deinterleave into separate channel buffers
                let channel_buffers: Vec<Vec<f32>> = selected_channels.iter().enumerate()
                    .map(|(i, _)| extract_channel_data(&buffer_clone, i, selected_channels.len()))
                    .collect();

                let mut all_channels_results = 
                    vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

                let config = fft_config.lock().unwrap();
                for (channel_index, _) in selected_channels.iter().enumerate() {
                    let spectrum = compute_spectrum(
                        &channel_buffers,  // Pass all channel data
                        channel_index,     // Specify which channel to process
                        sample_rate, 
                        &*config,
                        Some(&prev_results[channel_index])
                    );

                    all_channels_results[channel_index] = spectrum.clone();
                    prev_results[channel_index] = spectrum;
                }

                if let Ok(mut spectrum) = spectrum_app.lock() {
                    spectrum.update_partials(all_channels_results);
                }

                thread::sleep(Duration::from_millis(100));
            }));

            if let Err(e) = result {
                error!("FFT computation failed: {:?}", e);
                warn!("Panic in FFT thread: {:?}", e);
                
                error!("Failed to allocate FFT buffer");
                warn!("Buffer size mismatch, adjusting...");
            }
        }
        info!("FFT processing thread shutting down.");
    });
}
/// Extracts data for a specific channel from the interleaved buffer.
fn extract_channel_data(buffer: &[f32], channel: usize, num_channels: usize) -> Vec<f32> {
    buffer
        .iter()
        .skip(channel)  // Start at the correct channel offset
        .step_by(num_channels)  // Pick every Nth sample (de-interleaving)
        .copied()
        .collect()
}

/// Computes the frequency spectrum from the audio buffer.
pub fn compute_spectrum(
    all_channel_data: &[Vec<f32>],
    channel_index: usize,
    sample_rate: u32, 
    _config: &FFTConfig,
    _prev_magnitudes: Option<&[(f32, f32)]>
) -> Vec<(f32, f32)> {
    let signal: Vec<f64> = all_channel_data[channel_index]
        .iter()
        .map(|&x| x as f64)
        .collect();

    let mut detector = HannedFftDetector::default();
    
    if let Some(freq) = detector.detect_pitch(&signal, sample_rate as f64) {
        generate_spectrum(freq, sample_rate)
    } else {
        vec![(0.0, 0.0); NUM_PARTIALS]
    }
}

fn generate_spectrum(
    fundamental: f64,
    sample_rate: u32
) -> Vec<(f32, f32)> {
    let mut partials = Vec::with_capacity(NUM_PARTIALS);
    
    // Generate harmonics based on fundamental frequency
    for harmonic in 1..=NUM_PARTIALS {
        let freq = fundamental * harmonic as f64;
        if freq < sample_rate as f64 / 2.0 {  // Below Nyquist
            let magnitude = 1.0 / harmonic as f64;  // Simple 1/n falloff
            // Convert directly to amplitude without going through dB
            let amplitude = (80.0 * magnitude) as f32;  // Scale to 0-80 range
            partials.push((freq as f32, amplitude));
        } else {
            partials.push((0.0, 0.0));
        }
    }

    // Pad with zeros if needed
    while partials.len() < NUM_PARTIALS {
        partials.push((0.0, 0.0));
    }

    partials
}

/// Filters audio buffer based on amplitude threshold
#[allow(dead_code)]
pub fn filter_buffer(buffer: &[f32], db_threshold: f64) -> Vec<f32> {
    let linear_threshold = 10.0_f32.powf((db_threshold as f32) / 20.0);
    buffer
        .iter()
        .cloned()
        .filter(|&sample| sample.abs() >= linear_threshold)
        .collect()
}
