// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The following imports are protected. Any modification requires explicit permission.
use std::sync::{Arc, RwLock, Mutex};
use std::thread;
use std::time::Duration;
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::plot::SpectrumApp;
use crate::audio_stream::CircularBuffer;
use log::{info, warn, debug, error};
use std::sync::atomic::{AtomicBool, Ordering};
use pitch_detector::{
    pitch::{HannedFftDetector, PitchDetector},
    note::{detect_note},
    core::NoteName,
};

pub const NUM_PARTIALS: usize = 12;

/// Configuration struct for FFT settings.
pub struct FFTConfig {
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub db_threshold: f64,
    #[allow(dead_code)]
    pub num_channels: usize,
    pub averaging_factor: f32,  // Add averaging factor (0.0 to 1.0)
    pub frames_per_buffer: u32,  // Add frames per buffer setting
    pub crosstalk_threshold: f32,  // Add crosstalk threshold (0.0 to 1.0)
    pub crosstalk_reduction: f32,  // Add reduction factor (0.0 to 1.0)
    pub crosstalk_enabled: bool,  // Add enable flag for crosstalk filtering
    pub harmonic_enabled: bool,   // Add enable flag for harmonic filtering
    pub smoothing_enabled: bool,    // For temporal smoothing
    pub hanning_enabled: bool,      // For Hanning window
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
    config: &FFTConfig,
    prev_magnitudes: Option<&[(f32, f32)]>
) -> Vec<(f32, f32)> {
    let signal: Vec<f64> = all_channel_data[channel_index]
        .iter()
        .map(|&x| x as f64)
        .collect();

    let mut detector = HannedFftDetector::default();
    
    // Since pitch-detector returns Option<f64>
    if let Some(freq) = detector.detect_pitch(&signal, sample_rate as f64) {
        let mut magnitudes = generate_spectrum(freq, sample_rate);
        
        if config.smoothing_enabled && prev_magnitudes.is_some() {
            magnitudes = apply_smoothing(magnitudes, prev_magnitudes.unwrap(), config.averaging_factor);
        }
        
        magnitudes
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
            let magnitude_db = 20.0 * magnitude.log10();
            partials.push((freq as f32, magnitude_db as f32));
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
pub fn filter_buffer(buffer: &[f32], db_threshold: f64) -> Vec<f32> {
    let linear_threshold = 10.0_f32.powf((db_threshold as f32) / 20.0);
    buffer
        .iter()
        .cloned()
        .filter(|&sample| sample.abs() >= linear_threshold)
        .collect()
}

/// Reduces crosstalk between channels
fn filter_crosstalk(
    channel_data: &[Vec<f32>],
    threshold: f32,
    reduction: f32
) -> Vec<Vec<f32>> {
    if channel_data.is_empty() {
        return Vec::new();
    }

    let num_channels = channel_data.len();
    let samples_per_channel = channel_data[0].len();
    let mut filtered = vec![vec![0.0; samples_per_channel]; num_channels];

    for sample_idx in 0..samples_per_channel {
        // Get all channel values for this sample
        let sample_values: Vec<f32> = channel_data.iter()
            .map(|channel| channel[sample_idx])
            .collect();

        // Process each channel
        for (ch_idx, channel) in channel_data.iter().enumerate() {
            let main_signal = channel[sample_idx];
            let mut crosstalk = 0.0;

            // Calculate crosstalk from other channels
            for (other_idx, &other_val) in sample_values.iter().enumerate() {
                if other_idx != ch_idx {
                    let ratio = (other_val / main_signal).abs();
                    if ratio > threshold {
                        crosstalk += other_val * reduction;
                    }
                }
            }

            // Subtract crosstalk from main signal
            filtered[ch_idx][sample_idx] = main_signal - crosstalk;
        }
    }

    filtered
}

// Add Hanning window function
fn apply_hanning_window(data: &[f32]) -> Vec<f32> {
    data.iter()
        .enumerate()
        .map(|(i, &sample)| {
            let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / data.len() as f32).cos());
            sample * window
        })
        .collect()
}

fn apply_smoothing(
    current: Vec<(f32, f32)>,
    previous: &[(f32, f32)],
    factor: f32
) -> Vec<(f32, f32)> {
    current.iter().enumerate()
        .map(|(i, &(freq, amp))| {
            let prev_amp = previous.get(i).map(|&(_, a)| a).unwrap_or(amp);
            let smoothed_amp = factor * prev_amp + (1.0 - factor) * amp;
            (freq, smoothed_amp)
        })
        .collect()
}
