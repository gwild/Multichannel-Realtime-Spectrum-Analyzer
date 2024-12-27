use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::{Arc, Mutex};
use std::thread::{self, sleep};
use std::time::Duration;
use crate::plot::SpectrumApp;
use crate::audio_stream::CircularBuffer;
use log::{info, warn};

pub const NUM_PARTIALS: usize = 12;

/// Configuration struct for FFT settings.
pub struct FFTConfig {
    pub min_frequency: f32,
    pub max_frequency: f32,
    pub db_threshold: f32,
}

/// Spawns a thread to continuously process FFT data in the background.
pub fn start_fft_processing(
    buffer_clone_fft: Arc<Vec<Mutex<CircularBuffer>>>,
    fft_config_fft: Arc<Mutex<FFTConfig>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
    sample_rate: u32,
) {
    thread::spawn(move || loop {
        let mut all_channels_results = vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

        for (channel_index, channel) in selected_channels.iter().enumerate() {
            let buffer_data = {
                let buffer = buffer_clone_fft[channel_index].lock().unwrap();
                buffer.get().to_vec() // Clone buffer to avoid borrow conflicts
            };

            // Skip empty buffers to prevent unnecessary computation
            let sum: f32 = buffer_data.iter().map(|&x| x.abs()).sum();
            if sum == 0.0 {
                warn!("Buffer for channel {} is empty, skipping FFT.", channel);
                continue;
            }

            let config = fft_config_fft.lock().unwrap();
            let spectrum = compute_spectrum(&buffer_data, sample_rate, &*config);

            all_channels_results[channel_index] = spectrum;
        }

        if let Ok(mut spectrum) = spectrum_app.lock() {
            spectrum.update_partials(all_channels_results);
        }

        // Sleep to limit processing frequency to 10 Hz (100 ms)
        sleep(Duration::from_millis(100));
    });
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

    let mut filtered = Vec::new();
    let mut last_frequency = -1.0;

    for &(frequency, amplitude_db) in &magnitudes {
        if frequency - last_frequency >= config.min_frequency && amplitude_db >= config.db_threshold {
            let rounded_frequency = (frequency * 100.0).round() / 100.0;
            let rounded_amplitude = (amplitude_db * 100.0).round() / 100.0;
            filtered.push((rounded_frequency, rounded_amplitude));
            last_frequency = frequency;
        }
    }

    while filtered.len() < NUM_PARTIALS {
        filtered.push((0.0, 0.0));
    }

    filtered.truncate(NUM_PARTIALS);
    filtered
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
