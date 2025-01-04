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
use aubio_rs::FFT;

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
    // Only log once per cycle and only if any processing is happening
    if channel_index == 0 {
        let active_filters = [
            (config.hanning_enabled, "Hanning window"),
            (config.crosstalk_enabled, "Crosstalk filtering"),
            (config.smoothing_enabled, "Temporal smoothing")
        ];
        
        let enabled: Vec<&str> = active_filters.iter()
            .filter(|(enabled, _)| *enabled)
            .map(|(_, name)| *name)
            .collect();
            
        if !enabled.is_empty() {
            // Move to debug level since this is detailed processing info
            debug!("Processing cycle with: {}", enabled.join(", "));
        }
    }

    // Start with raw data
    let mut processed_data = all_channel_data[channel_index].clone();

    // Important state changes should be info level
    if processed_data.is_empty() {
        info!("Empty data received for channel {}", channel_index);
        return vec![(0.0, 0.0); NUM_PARTIALS];
    }

    // Processing details should be debug level
    if config.hanning_enabled {
        debug!("Applying Hanning window to channel {}", channel_index);
        processed_data = apply_hanning_window(&processed_data);
    }

    if config.crosstalk_enabled {
        debug!("Crosstalk filtering ch:{} (thresh:{}, red:{})", 
               channel_index, config.crosstalk_threshold, config.crosstalk_reduction);
        let filtered = filter_crosstalk(
            all_channel_data, 
            config.crosstalk_threshold,
            config.crosstalk_reduction
        );
        processed_data = filtered[channel_index].clone();
    }

    // Ensure FFT size is a power of 2 and at least 512
    let win_size = {
        let base_size = (config.frames_per_buffer as usize)
            .next_power_of_two();
        base_size.max(512)
    };

    let analysis_buffer = if processed_data.len() >= win_size {
        processed_data[processed_data.len() - win_size..].to_vec()
    } else {
        let mut padded = vec![0.0; win_size];
        padded[win_size - processed_data.len()..].copy_from_slice(&processed_data);
        padded
    };

    let mut fft = FFT::new(win_size).expect("Failed to create FFT");
    let mut spectrum = vec![0.0; win_size + 2];
    
    // aubio's FFT applies a Hanning window by default
    fft.do_(&analysis_buffer, &mut spectrum)
        .expect("FFT computation failed");

    // Get magnitudes from FFT output (only use first half due to Nyquist)
    let mut magnitudes: Vec<_> = (0..win_size/2)
        .map(|i| {
            let frequency = (i as f32) * (sample_rate as f32) / (win_size as f32);
            let amplitude = (spectrum[i*2].powi(2) + spectrum[i*2+1].powi(2)).sqrt();  // Complex magnitude
            let amplitude_db = if amplitude > 0.0 {
                20.0 * amplitude.log10()
            } else {
                f32::MIN
            };
            (frequency, amplitude_db.abs())
        })
        .collect();

    // Filter frequencies based on config
    magnitudes.retain(|&(freq, _)| {
        freq >= config.min_frequency as f32 && freq <= config.max_frequency as f32
    });

    // First sort by amplitude to get top NUM_PARTIALS
    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by amplitude descending
    
    // Only keep strong partials (above -60dB from max)
    if let Some(max_amp) = magnitudes.first().map(|&(_, amp)| amp) {
        magnitudes.retain(|&(_, amp)| amp > max_amp - 60.0);
    }
    magnitudes.truncate(NUM_PARTIALS);  // Keep top NUM_PARTIALS amplitudes

    // Then sort these by frequency
    magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // Sort by frequency ascending

    // Pad with zeros if needed
    while magnitudes.len() < NUM_PARTIALS {
        magnitudes.push((0.0, 0.0));
    }

    // Apply smoothing only if enabled
    let final_magnitudes = if config.smoothing_enabled {
        let mut smoothed = Vec::with_capacity(NUM_PARTIALS);
        for (i, &(freq, amp)) in magnitudes.iter().enumerate() {
            let prev_amp = prev_magnitudes
                .and_then(|prev| prev.get(i))
                .map(|&(_, a)| a)
                .unwrap_or(amp);
                
            let smoothed_amp = config.averaging_factor * prev_amp + 
                (1.0 - config.averaging_factor) * amp;
                
            smoothed.push((freq, smoothed_amp));
        }
        smoothed
    } else {
        magnitudes
    };

    final_magnitudes
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
