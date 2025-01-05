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
use rayon::prelude::*;
use std::f32::consts::PI;

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
    pub window_type: WindowType,
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
        .map(|&sample| {
            // Handle potential i32 scaled values on some platforms
            if sample > 1.0 || sample < -1.0 {
                sample / 32768.0  // Convert from i16 range to f32 [-1,1]
            } else {
                sample  // Already in correct range
            }
        })
        .collect()
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

/// Computes the frequency spectrum from the audio buffer.
pub fn compute_spectrum(
    all_channel_data: &[Vec<f32>],
    channel_index: usize,
    sample_rate: u32, 
    config: &FFTConfig,
    prev_magnitudes: Option<&[(f32, f32)]>
) -> Vec<(f32, f32)> {
    let signal = &all_channel_data[channel_index];
    
    // Check if signal is effectively zero
    if signal.iter().all(|&sample| sample.abs() < 1e-6) {
        return vec![(0.0, -100.0); NUM_PARTIALS];
    }
    
    // Apply db_threshold filter first
    let linear_threshold = 10.0f32.powf((config.db_threshold as f32) / 20.0);
    let filtered_signal: Vec<f32> = signal.iter()
        .map(|&sample| if sample.abs() >= linear_threshold { sample } else { 0.0 })
        .collect();
    
    // Check if filtered signal is effectively zero
    if filtered_signal.iter().all(|&sample| sample.abs() < 1e-6) {
        return vec![(0.0, -100.0); NUM_PARTIALS];
    }

    // Apply window to filtered signal
    let windowed_signal = apply_window(&filtered_signal, config.window_type);

    // Create FFT planner
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(windowed_signal.len());
    
    // Create buffers
    let mut indata = windowed_signal;  // Use windowed signal
    let mut spectrum = fft.make_output_vec();
    
    // Perform FFT
    if let Err(e) = fft.process(&mut indata, &mut spectrum) {
        error!("FFT computation error: {:?}", e);
        return vec![(0.0, 0.0); NUM_PARTIALS];
    }

    // Convert complex spectrum to magnitude pairs and convert to absolute dB
    let freq_step = sample_rate as f32 / signal.len() as f32;
    let mut all_magnitudes: Vec<(f32, f32)> = spectrum
        .par_iter()  // Use parallel iterator
        .enumerate()
        .filter_map(|(i, &complex_val)| {
            let frequency = i as f32 * freq_step;
            if frequency < config.min_frequency as f32 || frequency > config.max_frequency as f32 {
                None
            } else {
                let power = complex_val.re * complex_val.re + complex_val.im * complex_val.im;
                let current_db = if power > 0.0 {
                    (10.0f32 * power.log10()).round()
                } else {
                    -100.0
                };
                Some((frequency, current_db))
            }
        })
        .collect();

    // Sort by magnitude to find top 12
    all_magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut top_magnitudes = all_magnitudes.into_iter()
        .filter(|&(freq, _)| {
            freq >= config.min_frequency as f32 && freq <= config.max_frequency as f32
        })
        .take(NUM_PARTIALS)
        .collect::<Vec<_>>();

    // Sort by frequency for consistent display
    top_magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Pad with zeros if needed
    while top_magnitudes.len() < NUM_PARTIALS {
        top_magnitudes.push((0.0, -100.0));
    }

    top_magnitudes
}

/// Window function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    Rectangular,  // No window (flat)
    Hanning,
    Hamming,     // Similar to Hanning but doesn't go to zero at edges
    BlackmanHarris,
    FlatTop,     // Best amplitude accuracy
    Kaiser(f32), // Adjustable side-lobe level, beta parameter
}

fn apply_window(signal: &[f32], window_type: WindowType) -> Vec<f32> {
    let len = signal.len();
    let window = match window_type {
        WindowType::Rectangular => vec![1.0; len],
        WindowType::Hanning => hanning_window(len),
        WindowType::Hamming => hamming_window(len),
        WindowType::BlackmanHarris => blackman_harris_window(len),
        WindowType::FlatTop => flattop_window(len),
        WindowType::Kaiser(beta) => kaiser_window(len, beta),
    };
    
    signal.iter()
        .zip(window.iter())
        .map(|(&s, &w)| s * w)
        .collect()
}

fn hamming_window(len: usize) -> Vec<f32> {
    (0..len).map(|i| {
        let x = 2.0 * PI * i as f32 / (len - 1) as f32;
        0.54 - 0.46 * x.cos()
    }).collect()
}

fn flattop_window(len: usize) -> Vec<f32> {
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;
    
    (0..len).map(|i| {
        let x = 2.0 * PI * i as f32 / (len - 1) as f32;
        a0 - a1 * x.cos() + a2 * (2.0 * x).cos() - a3 * (3.0 * x).cos() + a4 * (4.0 * x).cos()
    }).collect()
}

fn kaiser_window(len: usize, beta: f32) -> Vec<f32> {
    let i0_beta = bessel_i0(beta);
    (0..len).map(|i| {
        let x = beta * (1.0 - (2.0 * i as f32 / (len - 1) as f32 - 1.0).powi(2)).sqrt();
        bessel_i0(x) / i0_beta
    }).collect()
}

// Modified Bessel function of the first kind, order 0
fn bessel_i0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt()) * (0.39894228 + y * (0.01328592
            + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
            + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633
            + y * 0.00392377))))))))
    }
}

fn hanning_window(len: usize) -> Vec<f32> {
    (0..len).map(|i| {
        let x = 2.0 * PI * i as f32 / (len - 1) as f32;
        0.5 * (1.0 - x.cos())
    }).collect()
}

fn blackman_harris_window(len: usize) -> Vec<f32> {
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;
    
    (0..len).map(|i| {
        let x = 2.0 * PI * i as f32 / (len - 1) as f32;
        a0 - a1 * x.cos() + a2 * (2.0 * x).cos() - a3 * (3.0 * x).cos()
    }).collect()
}
