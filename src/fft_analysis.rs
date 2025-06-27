// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The following imports are protected. Any modification requires explicit permission.
use std::sync::{Arc, RwLock, Mutex};
use std::thread;
use std::time::Duration;
use std::panic::{catch_unwind, AssertUnwindSafe};
use crate::plot::SpectrumApp;
use crate::audio_stream::CircularBuffer;
use log::{info, warn, error, debug};
use std::sync::atomic::{AtomicBool, Ordering};
use realfft::RealFftPlanner;
use rayon::prelude::*;
use std::f32::consts::PI;
use crate::{DEFAULT_BUFFER_SIZE, MAX_FREQ, MIN_FREQ}; // Update imports
use crate::DEFAULT_NUM_PARTIALS; // Import the new constant
use crate::plot::SpectrographSlice;
use std::collections::VecDeque;
use std::time::Instant;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use rustfft::num_complex::Complex;
use tokio::sync::broadcast; // Added import
use serde::{Serialize, Deserialize};

// Change the constant declaration to be public
pub const MAX_SPECTROGRAPH_HISTORY: usize = 500;

// Add a type alias for clarity
type PartialsData = Vec<Vec<(f32, f32)>>;

/// Configuration struct for FFT settings.
#[derive(Debug, Clone, PartialEq)]  // Add Clone derive
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
    pub harmonic_tolerance: f32,  // Add this field - controls how closely a frequency must match a harmonic
    pub window_type: WindowType,
    pub root_freq_min: f32,  // Add this (default: 20.0)
    pub root_freq_max: f32,  // Add this (default: DEFAULT_BUFFER_SIZE / 4)
    pub freq_match_distance: f32,  // Maximum Hz difference to consider frequencies as matching
    pub num_partials: usize,  // Add configurable number of partials
}

impl Default for FFTConfig {
    fn default() -> Self {
        Self {
            min_frequency: MIN_FREQ,
            max_frequency: *MAX_FREQ,
            magnitude_threshold: 6.0, 
            min_freq_spacing: 20.0,
            num_channels: 1,
            frames_per_buffer: 512,
            crosstalk_threshold: 0.3,
            crosstalk_reduction: 0.5,
            crosstalk_enabled: false,
            harmonic_tolerance: 0.03,
            root_freq_min: 20.0,
            root_freq_max: (DEFAULT_BUFFER_SIZE as f32 / 4.0),
            freq_match_distance: 5.0,
            window_type: WindowType::Hanning,
            num_partials: DEFAULT_NUM_PARTIALS, // Use default value from main.rs
        }
    }
}

// Define a macro to log specifically under target "crosstalk"
#[macro_export]
macro_rules! crosstalk_info {
    ($($arg:tt)*) => {
        log::debug!(target: "crosstalk", $($arg)*);
    };
}

// Add a new struct to hold both types of FFT data
pub struct FFTData {
    partials: Vec<Vec<(f32, f32)>>,
    line_data: Vec<Vec<(f32, f32)>>,
}

// Add near the top of the file with other structs
pub struct CurrentPartials {
    pub data: Vec<Vec<(f32, f32)>>,
}

impl CurrentPartials {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }
}

/// Computes both partial data and full FFT line data
fn compute_all_fft_data(
    all_channel_data: &[Vec<f32>],
    channel_index: usize,
    sample_rate: u32, 
    config: &FFTConfig,
) -> (Vec<(f32, f32)>, Vec<(f32, f32)>) {
    let signal = &all_channel_data[channel_index];
    let signal_len = signal.len(); // Store original signal length

    // Apply window to signal
    let windowed_signal = apply_window(&signal, config.window_type);

    // Perform FFT (once)
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(windowed_signal.len());
    let mut indata = windowed_signal;
    let mut complex_spectrum_output = fft.make_output_vec(); // Store the complex output
    
    if let Err(e) = fft.process(&mut indata, &mut complex_spectrum_output) {
        error!("FFT computation error: {:?}", e);
        return (vec![(0.0, 0.0); config.num_partials], Vec::new());
    }

    // Convert to dB scale for line_data
    let freq_step = sample_rate as f32 / signal_len as f32; // Use original signal_len
    let line_data: Vec<(f32, f32)> = complex_spectrum_output
        .par_iter()
        .enumerate()
        .map(|(i, &complex_val)| {
            let frequency = i as f32 * freq_step;
            let magnitude = (complex_val.re * complex_val.re + complex_val.im * complex_val.im).sqrt();
            let db = if magnitude > 1e-10 {
                20.0 * (magnitude + 1e-10).log10() // Add epsilon for stability
            } else {
                0.0 // Or a very small dB value like -120.0
            };
            (frequency, db.max(0.0)) // Ensure non-negative dB for line plot
        })
        .collect();

    // Compute partials (now linear magnitude) using the new function
    let partials = extract_partials_from_spectrum(
        &complex_spectrum_output, 
        sample_rate, 
        signal_len, // Pass original signal length
        config
    );

    (partials, line_data)
}

/// Processes audio data to extract spectral information.
/// Returns a tuple containing:
/// 1. Partials data (frequency, magnitude) for each channel
/// 2. FFT line data for visualization
/// 3. Spectrograph data for history tracking
pub fn process_audio_data(
    audio_data: &[f32],
    config: &FFTConfig,
    num_channels: usize,
    sample_rate: u32,
) -> Result<(PartialsData, Vec<Vec<(f32, f32)>>, Vec<(f64, f32)>), String> {
    if audio_data.is_empty() {
        return Err("Empty audio data".to_string());
    }

    // Extract channel data
    let channel_buffers: Vec<Vec<f32>> = (0..num_channels)
        .map(|i| extract_channel_data(audio_data, i, num_channels))
        .collect();
    
    if channel_buffers.is_empty() || channel_buffers[0].is_empty() {
        return Err("Failed to extract channel data".to_string());
    }

    // Process each channel to get both partial and line data
    let mut all_channels_partials = Vec::with_capacity(num_channels);
    let mut all_channels_line_data = Vec::with_capacity(num_channels);

    for channel_index in 0..num_channels {
        let (partials, line_data) = compute_all_fft_data(
            &channel_buffers,
            channel_index,
            sample_rate,
            config
        );
        
        all_channels_partials.push(partials);
        all_channels_line_data.push(line_data);
    }

    // Apply crosstalk filtering if enabled
    let filtered_partials: PartialsData = if config.crosstalk_enabled {
        filter_crosstalk_frequency_domain(
            &mut all_channels_partials.clone(),
            config.crosstalk_threshold,
            config.crosstalk_reduction,
            config.harmonic_tolerance,
            config.root_freq_min,
            config.root_freq_max,
            config.freq_match_distance,
            sample_rate
        )
    } else {
        all_channels_partials.clone()
    };

    // Generate spectrograph data
    let linear_threshold = 10.0_f32.powf(config.magnitude_threshold as f32 / 20.0);
    let spectrograph_data: Vec<(f64, f32)> = filtered_partials.iter()
        .flat_map(|channel_data| {
            channel_data.iter()
                .filter(move |&&(_freq, magnitude)| magnitude >= linear_threshold)
                .map(|&(freq, magnitude)| (freq as f64, magnitude))
        })
        .collect();

    Ok((filtered_partials, all_channels_line_data, spectrograph_data))
}

/// Spawns a thread to continuously process FFT data and update the plot.
pub fn start_fft_processing(
    audio_buffer: Arc<RwLock<CircularBuffer>>,
    fft_config: Arc<Mutex<FFTConfig>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
    sample_rate: u32,
    shutdown_flag: Arc<AtomicBool>,
    partials_tx: broadcast::Sender<PartialsData>,
    spectrograph_history: Option<Arc<Mutex<VecDeque<SpectrographSlice>>>>,
    start_time: Option<Arc<Instant>>,
) {
    // Add a counter to track FFT processing cycles
    let mut fft_cycle_count = 0;
    let mut last_log_time = Instant::now();
    let mut last_successful_process = Instant::now();

    info!("FFT processing thread started");
    debug!("FFT thread initialized with {} channels at {} Hz", selected_channels.len(), sample_rate);

    while !shutdown_flag.load(Ordering::SeqCst) {
        // Sleep to avoid excessive CPU usage
        thread::sleep(Duration::from_millis(10));
        
        fft_cycle_count += 1;
        
        // Log processing rate periodically
        if last_log_time.elapsed() >= Duration::from_secs(5) {
            debug!("FFT processing stats: {} cycles in last 5 seconds", fft_cycle_count);
            fft_cycle_count = 0;
            last_log_time = Instant::now();
        }

        // Check if buffer resize is in progress
        let buffer_resize_in_progress = {
            if let Ok(buffer) = audio_buffer.read() {
                let needs_restart = buffer.needs_restart();
                let needs_reinit = buffer.needs_reinit();
                if needs_restart || needs_reinit {
                    debug!("FFT thread detected buffer resize operation - needs_restart={}, needs_reinit={}", 
                           needs_restart, needs_reinit);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };

        if buffer_resize_in_progress {
            // If buffer resize is in progress, wait for it to complete
            debug!("FFT thread pausing processing during buffer resize operation");
            
            // Wait for the resize operation to complete
            let mut resize_completed = false;
            let start_wait = Instant::now();
            
            while !resize_completed && start_wait.elapsed() < Duration::from_secs(5) {
                if let Ok(buffer) = audio_buffer.read() {
                    resize_completed = !buffer.needs_restart() && !buffer.needs_reinit();
                    if resize_completed {
                        info!("FFT thread detected buffer resize completion");
                        debug!("FFT thread resuming processing after buffer resize");
                    }
                }
                
                if !resize_completed {
                    // Sleep briefly to avoid tight loop
                    thread::sleep(Duration::from_millis(50));
                }
            }
            
            if !resize_completed {
                warn!("FFT thread timed out waiting for buffer resize to complete");
            }
            
            // Skip processing this cycle
            continue;
        }

        // Get a copy of the audio data for FFT processing
        let audio_data = if let Ok(buffer) = audio_buffer.read() {
            buffer.clone_data()
        } else {
            continue;
        };

        if audio_data.is_empty() {
            continue;
        }

        // Get the current FFT configuration
        let fft_config_copy = if let Ok(config) = fft_config.lock() {
            config.clone()
        } else {
            continue;
        };

        // Process the audio data to extract spectral information
        match process_audio_data(
            &audio_data,
            &fft_config_copy,
            selected_channels.len(),
            sample_rate,
        ) {
            Ok((partials, fft_data, spectrograph_data)) => {
                last_successful_process = Instant::now();
                
                // Update the spectrum app with the FFT line data
                if let Ok(mut app) = spectrum_app.lock() {
                    app.update_fft_line_data(fft_data.clone());
                    debug!("Updated spectrum app with new FFT line data: {} channels", fft_data.len());
                } else {
                    debug!("Failed to lock spectrum_app to update FFT line data");
                }

                // Send the partials data to any subscribers (GUI and resynth)
                let receiver_count = partials_tx.receiver_count();
                match partials_tx.send(partials.clone()) {
                    Ok(_) => {
                        // Log every successful FFT completion with a cycle counter
                        static mut FFT_COMPLETION_COUNT: usize = 0;
                        unsafe {
                            FFT_COMPLETION_COUNT += 1;
                            if FFT_COMPLETION_COUNT % 10 == 0 {  // Log every 10th completion to avoid spam
                                debug!("FFT cycle #{} complete - sent data to {} receivers: {} channels, {} partials/channel", 
                                       FFT_COMPLETION_COUNT, 
                                       receiver_count,
                                       partials.len(),
                                       if !partials.is_empty() { partials[0].len() } else { 0 });
                            }
                        }
                        
                        // After buffer resize, log more details about data flow resumption
                        if last_successful_process.elapsed() > Duration::from_secs(1) {
                            info!("Data flow to GUI resumed after buffer resize");
                            debug!("Sent first batch of partials after resize: {} channels, {} partials per channel",
                                   partials.len(), 
                                   if !partials.is_empty() { partials[0].len() } else { 0 });
                        }
                    },
                    Err(e) => {
                        error!("Failed to send partials data: {} (receivers: {})", e, receiver_count);
                    }
                }

                // Update the spectrograph history if available
                if let Some(history) = &spectrograph_history {
                    if let Ok(mut history) = history.lock() {
                        let current_time = if let Some(start) = &start_time {
                            start.elapsed().as_secs_f64()
                        } else {
                            0.0
                        };

                        history.push_back(SpectrographSlice {
                            time: current_time,
                            data: spectrograph_data.clone(),
                        });
                        
                        debug!("Updated spectrograph history: {} entries, {} data points in latest slice", 
                               history.len(), spectrograph_data.len());

                        // Limit the history size
                        while history.len() > MAX_SPECTROGRAPH_HISTORY {
                            history.pop_front();
                        }
                    } else {
                        debug!("Failed to lock spectrograph history for update");
                    }
                }
            }
            Err(e) => {
                error!("Error processing audio data: {}", e);
                
                // If we haven't had a successful process in a while, log more details
                if last_successful_process.elapsed() > Duration::from_secs(5) {
                    debug!("No successful FFT processing for 5+ seconds. Last error: {}", e);
                    debug!("Audio data stats: {} samples, {} channels", 
                           audio_data.len(), selected_channels.len());
                    debug!("FFT config: window_type={:?}, frames_per_buffer={}, max_freq={}", 
                           fft_config_copy.window_type, fft_config_copy.frames_per_buffer, fft_config_copy.max_frequency);
                    
                    // Reset the timer to avoid spamming logs
                    last_successful_process = Instant::now();
                }
            }
        }
    }

    info!("FFT processing thread shutting down");
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

    // Calculate channel energy levels for normalization
    let channel_energy: Vec<f32> = channel_data.iter()
        .map(|channel| {
            channel.iter().map(|&s| s * s).sum::<f32>().sqrt() / channel.len() as f32
        })
        .collect();

    // Apply a stronger reduction factor (squared to make it more aggressive)
    let enhanced_reduction = reduction * 2.0;
    
    for sample_idx in 0..samples_per_channel {
        // Process each channel
        for (ch_idx, channel) in channel_data.iter().enumerate() {
            let main_signal = channel[sample_idx];
            let mut filtered_signal = main_signal;
            
            // Apply crosstalk reduction from other channels
            for (other_idx, other_channel) in channel_data.iter().enumerate() {
                if other_idx != ch_idx {
                    let other_signal = other_channel[sample_idx];
                    
                    // Calculate correlation between signals at this point
                    let correlation = (main_signal * other_signal).abs();
                    
                    // Normalize by channel energy to get relative correlation
                    let normalized_correlation = if channel_energy[ch_idx] > 1e-6 && channel_energy[other_idx] > 1e-6 {
                        correlation / (channel_energy[ch_idx] * channel_energy[other_idx])
                    } else {
                        0.0
                    };
                    
                    // Apply stronger reduction with a non-linear response curve
                    if normalized_correlation > threshold {
                        // Apply non-linear scaling to make reduction more aggressive
                        let scale_factor = (normalized_correlation - threshold) / (1.0 - threshold);
                        let dynamic_reduction = enhanced_reduction * (1.0 + scale_factor);
                        
                        // Limit maximum reduction to 1.0 to avoid over-subtraction
                        let clamped_reduction = dynamic_reduction.min(1.0);
                        filtered_signal -= other_signal * clamped_reduction;
                    }
                }
            }
            
            filtered[ch_idx][sample_idx] = filtered_signal;
        }
    }

    filtered
}

/// Check if all channels have nearly identical signals
fn is_identical_signals(channel_data: &[Vec<f32>]) -> bool {
    if channel_data.len() <= 1 {
        return false;
    }
    
    let first_channel = &channel_data[0];
    let sample_count = first_channel.len().min(100); // Check first 100 samples
    
    for ch_idx in 1..channel_data.len() {
        let other_channel = &channel_data[ch_idx];
        
        // Compare a subset of samples
        for i in 0..sample_count {
            let diff = (first_channel[i] - other_channel[i]).abs();
            if diff > 0.01 { // Allow small differences
                return false;
            }
        }
    }
    
    true
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
    _prev_magnitudes: Option<&[(f32, f32)]>
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
        return vec![(0.0, 0.0); config.num_partials];
    }

    // Keep threshold in dB for comparison
    // let linear_magnitude_threshold = 10.0_f32.powf(config.magnitude_threshold as f32 / 20.0);

    // 3. First collect all valid magnitudes above threshold
    let freq_step = sample_rate as f32 / signal.len() as f32;
    let mut all_magnitudes: Vec<(f32, f32)> = spectrum
        .par_iter()
        .enumerate()
        .filter_map(|(i, &complex_val)| {
            let frequency = i as f32 * freq_step;
            let magnitude = (complex_val.re * complex_val.re + complex_val.im * complex_val.im).sqrt();
            
            // Only compute dB if magnitude is significant
            if magnitude > 1e-10 { // Use a small epsilon to avoid log(0)
                let db = 20.0 * magnitude.log10();
                // Only include if above dB threshold and in frequency range
                if db > config.magnitude_threshold as f32 &&
                   frequency >= config.min_frequency as f32 && 
                   frequency <= config.max_frequency as f32 {
                    Some((frequency, db)) // Return dB magnitude
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // 4. If no peaks above threshold, return array of zeros
    if all_magnitudes.is_empty() {
        return vec![(0.0, 0.0); config.num_partials];
    }

    // 5. Sort by frequency (ascending)
    all_magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // 6. Apply minimum frequency spacing while maintaining frequency order
    let mut filtered_magnitudes: Vec<(f32, f32)> = Vec::with_capacity(config.num_partials);
    for &mag in all_magnitudes.iter() {
        if filtered_magnitudes.is_empty() {
            filtered_magnitudes.push(mag);
        } else {
            let last_freq = filtered_magnitudes.last().unwrap().0;
            if (mag.0 - last_freq).abs() >= config.min_freq_spacing as f32 {
                filtered_magnitudes.push(mag);
            }
        }
        if filtered_magnitudes.len() >= config.num_partials {
            break;
        }
    }

    // 7. Create final result vector with proper padding
    let mut result = Vec::with_capacity(config.num_partials);
    result.extend(filtered_magnitudes);
    while result.len() < config.num_partials {
        result.push((0.0, 0.0));
    }

    result
}

/// Window function types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

/// Applies crosstalk filtering in the frequency domain after FFT analysis
pub fn filter_crosstalk_frequency_domain(
    spectra: &mut Vec<Vec<(f32, f32)>>,
    threshold: f32,
    reduction: f32,
    harmonic_tolerance: f32,
    root_freq_min: f32,
    mut root_freq_max: f32,
    freq_match_distance: f32,
    sample_rate: u32
) -> Vec<Vec<(f32, f32)>> {
    // Instead of using sample_rate:
    let nyquist = (sample_rate as f32 / 2.0).min(8192.0);

    if root_freq_max > nyquist {
        root_freq_max = nyquist;
        crosstalk_info!("Clamping root_freq_max to {} (frames_per_buffer-based nyquist)", root_freq_max);
    }

    crosstalk_info!("Applying crosstalk filter to {} channels (threshold={}, reduction={})",
                    spectra.len(), threshold, reduction);

    if spectra.is_empty() {
        return Vec::new();
    }

    let num_channels = spectra.len();
    if num_channels == 1 {
        return spectra.clone(); // No crosstalk with single channel
    }

    // Continue crosstalk logic as before, using the newly clamped `root_freq_max`:
    // e.g. finding root in range [root_freq_min .. root_freq_max]
    let mut root_frequencies: Vec<f32> = Vec::with_capacity(num_channels);
    for (ch_idx, channel_spectra) in spectra.iter().enumerate() {
        let root = channel_spectra.iter()
            .filter(|&&(freq, _)| freq > root_freq_min && freq < root_freq_max)
            .max_by(|&&(_, mag_a), &&(_, mag_b)|
                mag_a.partial_cmp(&mag_b).unwrap_or(std::cmp::Ordering::Equal)
            )
            .map(|&(freq, _)| freq)
            .unwrap_or(0.0);

        crosstalk_info!(" Channel {} root freq = {:.2} Hz", ch_idx, root);
        root_frequencies.push(root);
    }
    
    // 2. Debug: Log partials before filtering
    for (ch_idx, channel_spectra) in spectra.iter().enumerate() {
        crosstalk_info!("Pre-filter partials for channel {}:", ch_idx);
        for &(freq, mag) in channel_spectra.iter() {
            crosstalk_info!("   freq={:.2}, mag={:.4}", freq, mag);
        }
    }

    let mut filtered_spectra = spectra.clone();
    let scaled_reduction = reduction;

    let mut count_filtered = 0;
    
    // For each frequency in each channel
    for ch_idx in 0..num_channels {
        if root_frequencies[ch_idx] < root_freq_min {
            continue;
        }
        
        for i in 0..filtered_spectra[ch_idx].len() {
            let (freq, magnitude) = filtered_spectra[ch_idx][i];
            if magnitude <= 0.0 {
                continue;
            }
            
            let is_harmonic = is_harmonic_of(freq, root_frequencies[ch_idx], harmonic_tolerance);
            
            for other_ch in 0..num_channels {
                if other_ch == ch_idx {
                    continue;
                }
                
                if let Some(other_idx) = find_closest_frequency(&filtered_spectra[other_ch], freq, freq_match_distance) {
                    let (other_freq, other_mag) = filtered_spectra[other_ch][other_idx];
                    if other_mag <= 0.0 {
                        continue;
                    }
                    
                    let other_is_harmonic = is_harmonic_of(other_freq, root_frequencies[other_ch], harmonic_tolerance);
                    
                    // Debug logging for the decision
                    crosstalk_info!("Comparing ch{} freq={:.1} (harm={}) to ch{} freq={:.1} (harm={} mag={:.3})",
                          ch_idx, freq, is_harmonic, other_ch, other_freq, other_is_harmonic, other_mag);
                    
                    // (same crosstalk logic as before)
                    // If both channels have harmonic claim
                    if is_harmonic && other_is_harmonic {
                        if other_mag > magnitude * 1.5 {
                            filtered_spectra[ch_idx][i].1 *= 0.1;
                            count_filtered += 1;
                            crosstalk_info!("  → ch{} weaker harmonic freq={:.1}, reducing 90%", ch_idx, freq);
                        } else if magnitude > other_mag * 1.5 {
                            filtered_spectra[other_ch][other_idx].1 *= 0.1;
                            count_filtered += 1;
                            crosstalk_info!("  → ch{} weaker harmonic freq={:.1}, reducing 90%", other_ch, other_freq);
                        } else {
                            // Similar strength - reduce both proportionally
                            let total = magnitude + other_mag;
                            let my_ratio = magnitude / total;
                            let other_ratio = other_mag / total;
                            filtered_spectra[ch_idx][i].1 *= my_ratio;
                            filtered_spectra[other_ch][other_idx].1 *= other_ratio;
                            count_filtered += 2;
                            crosstalk_info!("  → Both harmonics freq={:.1} ~ freq={:.1}, proportionally reduced", freq, other_freq);
                        }
                    }
                    else if is_harmonic && !other_is_harmonic {
                        filtered_spectra[other_ch][other_idx].1 *= 1.0 - scaled_reduction;
                        crosstalk_info!("  → Reduced ch{} freq={:.1} (NON-harm), mag now={:.3}", other_ch, other_freq, filtered_spectra[other_ch][other_idx].1);
                        count_filtered += 1;
                    }
                    else if !is_harmonic && other_is_harmonic {
                        filtered_spectra[ch_idx][i].1 *= 1.0 - scaled_reduction;
                        crosstalk_info!("  → Reduced ch{} freq={:.1} (NON-harm), mag now={:.3}", ch_idx, freq, filtered_spectra[ch_idx][i].1);
                        count_filtered += 1;
                    }
                    else {
                        // Neither is harmonic – compare magnitudes
                        if other_mag > magnitude * (1.0 + threshold) {
                            filtered_spectra[ch_idx][i].1 *= 1.0 - scaled_reduction;
                            count_filtered += 1;
                            crosstalk_info!("  → ch{} significantly weaker freq={:.1}, mag now={:.3}", ch_idx, freq, filtered_spectra[ch_idx][i].1);
                        } else if magnitude > other_mag * (1.0 + threshold) {
                            filtered_spectra[other_ch][other_idx].1 *= 1.0 - scaled_reduction;
                            count_filtered += 1;
                            crosstalk_info!("  → ch{} significantly weaker freq={:.1}, mag now={:.3}", other_ch, other_freq, filtered_spectra[other_ch][other_idx].1);
                        }
                    }
                }
            }
        }
    }
    
    // Sort partials, remove zeros, etc.
    //  (same cleanup code as before)
    
    // Debug: Log partials for each channel after filter
    for (ch_idx, channel_spectra) in filtered_spectra.iter().enumerate() {
        crosstalk_info!("Post-filter partials for channel {}:", ch_idx);
        for &(freq, mag) in channel_spectra.iter() {
            crosstalk_info!("   freq={:.2}, mag={:.4}", freq, mag);
        }
    }

    crosstalk_info!("Crosstalk filter applied - filtered {} frequencies", count_filtered);
    
    filtered_spectra
}

/// Helper function to check if a frequency is a harmonic of a root frequency
fn is_harmonic_of(freq: f32, root: f32, tolerance: f32) -> bool {
    if root <= 0.0 {
        return false;
    }
    
    // Calculate harmonic ratio
    let ratio = freq / root;
    
    // Check if it's close to an integer
    let nearest_harmonic = ratio.round();
    let distance = (ratio - nearest_harmonic).abs();
    
    // Allow more tolerance for higher harmonics
    let adjusted_tolerance = tolerance * (1.0 + 0.1 * nearest_harmonic);
    
    // Return true if it's within tolerance of an integer ratio
    distance < adjusted_tolerance && nearest_harmonic > 0.0 && nearest_harmonic < 20.0
}

/// Helper function to find the index of the closest frequency in a spectrum
fn find_closest_frequency(spectrum: &[(f32, f32)], target: f32, max_distance: f32) -> Option<usize> {
    let mut closest_idx = None;
    let mut min_distance = max_distance;
    
    for (idx, &(freq, _)) in spectrum.iter().enumerate() {
        let distance = (freq - target).abs();
        if distance < min_distance {
            min_distance = distance;
            closest_idx = Some(idx);
        }
    }
    
    closest_idx
}

/// Extracts partials (frequency, magnitude peaks) from a pre-computed complex FFT spectrum.
fn extract_partials_from_spectrum(
    spectrum: &[Complex<f32>],
    sample_rate: u32,
    signal_len: usize, // Need original signal length for freq_step
    config: &FFTConfig,
) -> Vec<(f32, f32)> {
    // 1. Calculate frequency step
    let freq_step = sample_rate as f32 / signal_len as f32;

    // Convert dB threshold to linear magnitude threshold once
    let linear_magnitude_threshold = 10.0_f32.powf(config.magnitude_threshold as f32 / 20.0);

    // 2. Collect all valid magnitudes above threshold from the complex spectrum
    let mut all_magnitudes: Vec<(f32, f32)> = spectrum
        .par_iter()
        .enumerate()
        .filter_map(|(i, &complex_val)| {
            let frequency = i as f32 * freq_step;
            let magnitude = (complex_val.re * complex_val.re + complex_val.im * complex_val.im).sqrt();

            // Filter based on linear magnitude threshold and frequency range
            if magnitude >= linear_magnitude_threshold &&
               frequency >= config.min_frequency as f32 &&
               frequency <= config.max_frequency as f32 {
                // MOVE THE GUI's CONVERSION HERE: Robustly convert to dB
                let db = if magnitude > 1e-10 { // Use an epsilon for stability
                    20.0 * magnitude.log10()
                } else {
                    -120.0 // Use a large negative number for silence, as is standard
                };
                Some((frequency, db)) // Return correct dB magnitude
            } else {
                None
            }
        })
        .collect();

    // 3. If no peaks above threshold, return array of zeros
    if all_magnitudes.is_empty() {
        return vec![(0.0, 0.0); config.num_partials];
    }

    // 4. Sort by frequency (ascending)
    all_magnitudes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // 5. Apply minimum frequency spacing while maintaining frequency order
    let mut filtered_magnitudes: Vec<(f32, f32)> = Vec::with_capacity(config.num_partials);
    for &mag in all_magnitudes.iter() {
        if filtered_magnitudes.is_empty() {
            filtered_magnitudes.push(mag);
        } else {
            let last_freq = filtered_magnitudes.last().unwrap().0;
            if (mag.0 - last_freq).abs() >= config.min_freq_spacing as f32 {
                filtered_magnitudes.push(mag);
            }
        }
        // Stop early if we have enough partials
        if filtered_magnitudes.len() >= config.num_partials {
            break;
        }
    }

    // 6. Create final result vector with proper padding
    let mut result = Vec::with_capacity(config.num_partials);
    result.extend(filtered_magnitudes);
    // Pad with zeros if fewer than num_partials were found
    while result.len() < config.num_partials {
        result.push((0.0, 0.0));
    }

    result
}
