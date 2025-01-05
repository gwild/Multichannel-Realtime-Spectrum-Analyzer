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
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

pub const NUM_PARTIALS: usize = 12;  // Keep original 12 partials

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
    let signal = &all_channel_data[channel_index];
    
    // Create FFT planner
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(signal.len());
    
    // Create buffers
    let mut indata = signal.clone();
    let mut spectrum = fft.make_output_vec();
    
    // Perform FFT
    if let Err(e) = fft.process(&mut indata, &mut spectrum) {
        error!("FFT computation error: {:?}", e);
        return vec![(0.0, 0.0); NUM_PARTIALS];
    }

    // Convert complex spectrum to magnitude pairs and convert to absolute dB
    let freq_step = sample_rate as f32 / signal.len() as f32;
    let mut all_magnitudes: Vec<(f32, f32)> = spectrum.iter().enumerate()
        .map(|(i, &complex_val)| {
            let frequency = i as f32 * freq_step;
            // Get power spectrum first
            let power = complex_val.re * complex_val.re + complex_val.im * complex_val.im;
            // Convert to dB with proper scaling and reference
            let db = if power > 0.0 {
                (10.0 * power.log10()).round()  // Convert to dB and round
            } else {
                -100.0
            };
            (frequency, db)  // No need for abs() since we're using power
        })
        .collect();

    // Sort by magnitude to find top 12
    all_magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut top_magnitudes = all_magnitudes.into_iter()
        .take(NUM_PARTIALS)
        .collect::<Vec<_>>();

    // Sort by frequency for consistent display
    top_magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Pad with zeros if needed (shouldn't be necessary but keeping for safety)
    while top_magnitudes.len() < NUM_PARTIALS {
        top_magnitudes.push((0.0, -100.0));  // Use -100 dB for silence
    }

    top_magnitudes
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
