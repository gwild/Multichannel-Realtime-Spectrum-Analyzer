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
use rayon::prelude::*;
use std::f32::consts::PI;

pub const NUM_PARTIALS: usize = 12;  // Keep original 12 partials

/// Configuration struct for FFT settings.
pub struct FFTConfig {
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub magnitude_threshold: f64,  // Renamed from db_threshold
    pub min_freq_spacing: f64,  // Add new parameter
    #[allow(dead_code)]
    pub num_channels: usize,
    pub frames_per_buffer: u32,
    pub crosstalk_threshold: f32,  // Add crosstalk threshold (0.0 to 1.0)
    pub crosstalk_reduction: f32,  // Add reduction factor (0.0 to 1.0)
    pub crosstalk_enabled: bool,  // Add enable flag for crosstalk filtering
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
                        None
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
pub fn extract_channel_data(buffer: &[f32], channel: usize, num_channels: usize) -> Vec<f32> {
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

/// Reduces crosstalk between channels
pub fn filter_crosstalk(
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
    _prev_magnitudes: Option<&[(f32, f32)]>  // Not used anymore
) -> Vec<(f32, f32)> {
    let signal = &all_channel_data[channel_index];
    
    // 1. Apply window to signal
    let windowed_signal = apply_window(&signal, config.window_type);

    // 2. Perform FFT
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(windowed_signal.len());
    let mut indata = windowed_signal;
    let mut spectrum = fft.make_output_vec();
    
    if let Err(e) = fft.process(&mut indata, &mut spectrum) {
        error!("FFT computation error: {:?}", e);
        return vec![(0.0, 0.0); NUM_PARTIALS];
    }

    // 3. Convert to absolute magnitudes and scale
    let freq_step = sample_rate as f32 / signal.len() as f32;
    let mut all_magnitudes: Vec<(f32, f32)> = spectrum
        .par_iter()
        .enumerate()
        .map(|(i, &complex_val)| {
            let frequency = i as f32 * freq_step;
            let magnitude = (complex_val.re * complex_val.re + complex_val.im * complex_val.im).sqrt() / 4.0;  // Divide by 4 after sqrt
            (frequency, magnitude)
        })
        .collect();

    // 4. Apply frequency thresholds
    all_magnitudes.retain(|&(freq, _)| {
        freq >= config.min_frequency as f32 && freq <= config.max_frequency as f32
    });

    // 5. Apply magnitude threshold
    all_magnitudes.retain(|&(_, magnitude)| {
        magnitude >= config.magnitude_threshold as f32
    });

    // 6. Sort by magnitude to get top partials
    all_magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // After sorting by magnitude but before truncating:
    let mut filtered_magnitudes = Vec::new();
    for &mag in all_magnitudes.iter() {
        if filtered_magnitudes.iter().all(|&prev: &(f32, f32)| 
            (mag.0 - prev.0).abs() >= config.min_freq_spacing as f32) {
            filtered_magnitudes.push(mag);
        }
        if filtered_magnitudes.len() >= NUM_PARTIALS {
            break;
        }
    }
    all_magnitudes = filtered_magnitudes;

    // 7. Sort by frequency for display
    all_magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // 8. Pad with zeros if needed
    while all_magnitudes.len() < NUM_PARTIALS {
        all_magnitudes.push((0.0, 0.0));
    }

    all_magnitudes
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

pub fn apply_window(signal: &[f32], window_type: WindowType) -> Vec<f32> {
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
