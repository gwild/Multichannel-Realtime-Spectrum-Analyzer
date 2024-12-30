// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The following imports are protected. Any modification requires explicit permission.
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::{Arc, RwLock, Mutex};
use std::thread::{self, sleep};
use std::time::Duration;
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::plot::SpectrumApp;
use crate::audio_stream::CircularBuffer;
use log::{info, warn};
use std::sync::atomic::{AtomicBool, Ordering};

pub const NUM_PARTIALS: usize = 12;

/// Configuration struct for FFT settings.
pub struct FFTConfig {
    pub min_frequency: f32,
    pub max_frequency: f32,
    pub db_threshold: f32,
    pub num_channels: usize,  // Add channel count to config
}


/// Spawns a thread to continuously process FFT data and update the plot.
pub fn start_fft_processing(
    audio_buffer: Arc<RwLock<CircularBuffer>>,  // Single buffer for all channels
    fft_config: Arc<Mutex<FFTConfig>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
    sample_rate: u32,
    shutdown_flag: Arc<AtomicBool>,  // Accept shutdown flag
) {
    thread::spawn(move || {
        while !shutdown_flag.load(Ordering::Relaxed) {
            let result = catch_unwind(AssertUnwindSafe(|| {
                // Clone the circular buffer for processing
                let buffer_clone = {
                    let buffer = audio_buffer.read().unwrap();
                    buffer.clone_data()
                };

                let mut all_channels_results =
                    vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

                // Process each channel separately by de-interleaving the buffer
                for (channel_index, channel) in selected_channels.iter().enumerate() {
                    let buffer_data = extract_channel_data(&buffer_clone, channel_index, selected_channels.len());

                    let sum: f32 = buffer_data.iter().map(|&x| x.abs()).sum();
                    if sum == 0.0 {
                        warn!("Buffer for channel {} is empty, skipping FFT.", channel);
                        continue;
                    }

                    let config = fft_config.lock().unwrap();
                    let spectrum = compute_spectrum(&buffer_data, sample_rate, &*config);

                    all_channels_results[channel_index] = spectrum;
                }

                // Update the plot after FFT processing
                if let Ok(mut spectrum) = spectrum_app.lock() {
                    spectrum.update_partials(all_channels_results);
                }

                // Process FFT every 100ms
                sleep(Duration::from_millis(100));
            }));

            if let Err(e) = result {
                warn!("Panic in FFT thread: {:?}", e);
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
pub fn compute_spectrum(buffer: &[f32], sample_rate: u32, config: &FFTConfig) -> Vec<(f32, f32)> {
    if buffer.is_empty() {
        return vec![(0.0, 0.0); NUM_PARTIALS];
    }

    let linear_threshold = 10.0_f32.powf(config.db_threshold / 20.0);
    let filtered_buffer: Vec<f32> = buffer
        .iter()
        .cloned()
        .filter(|&sample| sample.abs() >= linear_threshold)
        .collect();

    if filtered_buffer.is_empty() {
        return vec![(0.0, 0.0); NUM_PARTIALS];
    }

    let windowed_buffer: Vec<f32> = apply_blackman_harris(&filtered_buffer);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(windowed_buffer.len());

    let mut complex_buffer: Vec<Complex<f32>> = windowed_buffer
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();
    fft.process(&mut complex_buffer);

    let mut magnitudes: Vec<_> = complex_buffer
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let frequency = (i as f32) * (sample_rate as f32) / (windowed_buffer.len() as f32);
            let amplitude = value.norm();
            let amplitude_db = if amplitude > 0.0 {
                20.0 * amplitude.log10()
            } else {
                f32::MIN
            };
            (frequency, amplitude_db.abs())
        })
        .collect();

    magnitudes.retain(|&(freq, _)| freq >= config.min_frequency && freq <= config.max_frequency);
    magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    while magnitudes.len() < NUM_PARTIALS {
        magnitudes.push((0.0, 0.0));
    }

    magnitudes.truncate(NUM_PARTIALS);
    magnitudes
}

/// Applies Blackman-Harris windowing to the audio buffer.
fn apply_blackman_harris(buffer: &[f32]) -> Vec<f32> {
    let n = buffer.len();
    buffer
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let alpha0 = 0.35875;
            let alpha1 = 0.48829;
            let alpha2 = 0.14128;
            let alpha3 = 0.01168;
            let factor = alpha0
                - alpha1 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos()
                + alpha2 * (4.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos()
                - alpha3 * (6.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();
            sample * factor
        })
        .collect()
}
