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
    pub harmonic_tolerance: f32,  // Add this field - controls how closely a frequency must match a harmonic
    pub window_type: WindowType,
    pub root_freq_min: f32,  // Add this (default: 20.0)
    pub root_freq_max: f32,  // Add this (default: 500.0)
    pub freq_match_distance: f32,  // Maximum Hz difference to consider frequencies as matching
}

impl Default for FFTConfig {
    fn default() -> Self {
        Self {
            min_frequency: 20.0,
            max_frequency: 1400.0,
            magnitude_threshold: 6.0, 
            min_freq_spacing: 20.0,
            num_channels: 1,
            frames_per_buffer: 512,
            crosstalk_threshold: 0.3,
            crosstalk_reduction: 0.5,
            crosstalk_enabled: false,
            harmonic_tolerance: 0.03,
            root_freq_min: 20.0,
            root_freq_max: 500.0,
            freq_match_distance: 5.0,
            window_type: WindowType::Hanning,
        }
    }
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
    let _prev_results = vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

    thread::spawn(move || {
        while !shutdown_flag.load(Ordering::Relaxed) {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let buffer_clone = {
                    let buffer = audio_buffer.read().unwrap();
                    buffer.clone_data()
                };

                // Extract channel data
                let channel_buffers: Vec<Vec<f32>> = selected_channels.iter().enumerate()
                    .map(|(i, _)| extract_channel_data(&buffer_clone, i, selected_channels.len()))
                    .collect();

                // Get crosstalk settings from config
                let config = fft_config.lock().unwrap();
                let crosstalk_enabled = config.crosstalk_enabled;
                let crosstalk_threshold = config.crosstalk_threshold;
                let crosstalk_reduction = config.crosstalk_reduction;
                
                // Process each channel independently first to get spectra
                let mut all_channels_spectra: Vec<Vec<(f32, f32)>> = Vec::new();
                
                for (channel_index, _) in selected_channels.iter().enumerate() {
                    let spectrum = compute_spectrum(
                        &channel_buffers,
                        channel_index,     
                        sample_rate, 
                        &*config,
                        None
                    );
                    
                    all_channels_spectra.push(spectrum);
                }
                
                // Now apply frequency-domain crosstalk filtering if enabled
                let all_channels_results = if crosstalk_enabled {
                    filter_crosstalk_frequency_domain(
                        &mut all_channels_spectra,
                        crosstalk_threshold,
                        crosstalk_reduction,
                        config.harmonic_tolerance,
                        config.root_freq_min,
                        config.root_freq_max,
                        config.freq_match_distance
                    )
                } else {
                    all_channels_spectra
                };

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

/// Applies crosstalk filtering in the frequency domain after FFT analysis
pub fn filter_crosstalk_frequency_domain(
    spectra: &mut Vec<Vec<(f32, f32)>>,
    crosstalk_threshold: f32,
    crosstalk_reduction: f32,
    harmonic_tolerance: f32,
    root_freq_min: f32,
    root_freq_max: f32,
    freq_match_distance: f32
) -> Vec<Vec<(f32, f32)>> {
    if spectra.is_empty() {
        return Vec::new();
    }

    let num_channels = spectra.len();
    if num_channels == 1 {
        return spectra.clone(); // No crosstalk with single channel
    }

    // Find root frequency for each channel (strongest low frequency peak)
    let mut root_frequencies: Vec<f32> = Vec::with_capacity(num_channels);
    
    for channel_spectra in spectra.iter() {
        // Sort by magnitude to find strongest peaks
        let mut peaks = channel_spectra.clone();
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find the lowest significant frequency below 500Hz
        // This is a common range for fundamental frequencies
        let root = peaks.iter()
            .filter(|&&(freq, mag)| freq > root_freq_min && freq < root_freq_max && mag > 0.01)
            .next()
            .map(|&(freq, _)| freq)
            .unwrap_or(0.0);
            
        root_frequencies.push(root);
    }
    
    // Create a copy to store filtered results
    let mut filtered_spectra = spectra.clone();
    
    // Make reduction 2x more aggressive by doubling the effect
    let scaled_reduction = crosstalk_reduction * 2.0;
    
    // Apply a limit to ensure we don't exceed 1.0 which would invert the signal
    let clamped_reduction = scaled_reduction.min(1.0);
    
    // For each frequency in each channel
    for ch_idx in 0..num_channels {
        // Skip if no root frequency detected for this channel
        if root_frequencies[ch_idx] < root_freq_min {
            continue;
        }
        
        // For each partial in this channel
        for i in 0..filtered_spectra[ch_idx].len() {
            let (freq, magnitude) = filtered_spectra[ch_idx][i];
            
            // Skip if already filtered out
            if magnitude <= 0.0 {
                continue;
            }
            
            // Check if this is a harmonic of our root
            let is_harmonic = is_harmonic_of(freq, root_frequencies[ch_idx], harmonic_tolerance);
            
            // Check other channels for the same frequency
            for other_ch in 0..num_channels {
                if other_ch == ch_idx {
                    continue;
                }
                
                // Find closest matching frequency in other channel
                if let Some(other_idx) = find_closest_frequency(&filtered_spectra[other_ch], freq, freq_match_distance) {
                    let (other_freq, other_mag) = filtered_spectra[other_ch][other_idx];
                    
                    // Is this frequency a harmonic of the other channel's root?
                    let other_is_harmonic = is_harmonic_of(other_freq, root_frequencies[other_ch], harmonic_tolerance);
                    
                    // If both channels have harmonic claim to this frequency
                    if is_harmonic && other_is_harmonic {
                        // Keep it in both channels if they're both strong
                        if magnitude > 0.1 && other_mag > 0.1 {
                            continue;
                        }
                        
                        // Otherwise, assign to the stronger channel
                        if other_mag > magnitude * (1.0 + crosstalk_threshold) {
                            // Other channel is stronger, remove from this channel
                            filtered_spectra[ch_idx][i].1 = 0.0;
                        } else if magnitude > other_mag * (1.0 + crosstalk_threshold) {
                            // This channel is stronger, remove from other channel
                            filtered_spectra[other_ch][other_idx].1 = 0.0;
                        }
                        // If magnitudes are similar, keep in both channels
                    }
                    // If only one has harmonic claim
                    else if is_harmonic && !other_is_harmonic {
                        // This is our harmonic, reduce or remove it from other channel
                        filtered_spectra[other_ch][other_idx].1 *= 1.0 - (crosstalk_threshold * clamped_reduction);
                    }
                    else if !is_harmonic && other_is_harmonic {
                        // This is their harmonic, reduce it in our channel
                        filtered_spectra[ch_idx][i].1 *= 1.0 - (crosstalk_threshold * clamped_reduction);
                    }
                    // Neither has harmonic claim - decide based on magnitude
                    else if other_mag > magnitude * (1.0 + crosstalk_threshold) {
                        // Other channel is significantly stronger
                        filtered_spectra[ch_idx][i].1 *= 1.0 - (crosstalk_threshold * clamped_reduction);
                    } else if magnitude > other_mag * (1.0 + crosstalk_threshold) {
                        // This channel is significantly stronger
                        filtered_spectra[other_ch][other_idx].1 *= 1.0 - (crosstalk_threshold * clamped_reduction);
                    }
                }
            }
        }
    }
    
    // Remove zero-magnitude partials from each channel
    for ch_idx in 0..num_channels {
        filtered_spectra[ch_idx].retain(|&(_, mag)| mag > 0.001);
        
        // Sort by frequency
        filtered_spectra[ch_idx].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Ensure we have exactly NUM_PARTIALS partials
        while filtered_spectra[ch_idx].len() < NUM_PARTIALS {
            filtered_spectra[ch_idx].push((0.0, 0.0));
        }
        
        // Truncate if we have too many
        filtered_spectra[ch_idx].truncate(NUM_PARTIALS);
    }
    
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
