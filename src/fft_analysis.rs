// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The following imports are protected. Any modification requires explicit permission.
use std::sync::{Arc, RwLock, Mutex};
use std::thread;
use std::time::Duration;
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::plot::SpectrumApp;
use crate::audio_stream::CircularBuffer;
use log::{info, warn};
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

                let mut all_channels_results = 
                    vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

                for (channel_index, _channel) in selected_channels.iter().enumerate() {
                    let buffer_data = extract_channel_data(&buffer_clone, channel_index, selected_channels.len());

                    let config = fft_config.lock().unwrap();
                    let spectrum = compute_spectrum(
                        &buffer_data, 
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
pub fn compute_spectrum(
    buffer: &[f32], 
    sample_rate: u32, 
    config: &FFTConfig,
    prev_magnitudes: Option<&[(f32, f32)]>
) -> Vec<(f32, f32)> {
    if buffer.is_empty() {
        return vec![(0.0, 0.0); NUM_PARTIALS];
    }

    // Use the common filtering function
    let filtered_buffer = filter_buffer(buffer, config.db_threshold);
    if filtered_buffer.is_empty() {
        return vec![(0.0, 0.0); NUM_PARTIALS];
    }

    // Ensure FFT size is a power of 2 and at least 512
    let win_size = {
        let base_size = (config.frames_per_buffer as usize)
            .next_power_of_two();
        base_size.max(512)
    };

    let analysis_buffer = if filtered_buffer.len() >= win_size {
        filtered_buffer[filtered_buffer.len() - win_size..].to_vec()
    } else {
        let mut padded = vec![0.0; win_size];
        padded[win_size - filtered_buffer.len()..].copy_from_slice(&filtered_buffer);
        padded
    };

    let mut fft = FFT::new(win_size).expect("Failed to create FFT");
    let mut spectrum = vec![0.0; win_size + 2];
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
    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by amplitude descending

    // Ensure we have exactly NUM_PARTIALS entries
    while magnitudes.len() < NUM_PARTIALS {
        magnitudes.push((0.0, 0.0));
    }
    magnitudes.truncate(NUM_PARTIALS);

    // Apply smoothing
    let mut smoothed_magnitudes = Vec::with_capacity(NUM_PARTIALS);
    for (i, &(freq, amp)) in magnitudes.iter().enumerate() {
        let prev_amp = prev_magnitudes
            .and_then(|prev| prev.get(i))
            .map(|&(_, a)| a)
            .unwrap_or(amp);
            
        let smoothed_amp = config.averaging_factor * prev_amp + 
            (1.0 - config.averaging_factor) * amp;
            
        smoothed_magnitudes.push((freq, smoothed_amp));
    }

    smoothed_magnitudes
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
