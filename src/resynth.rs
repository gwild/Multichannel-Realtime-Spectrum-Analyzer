use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;
use portaudio as pa;
use log::{info, error};
use crate::plot::SpectrumApp;
use crate::DEFAULT_NUM_PARTIALS;
use crossbeam_queue::ArrayQueue;
use anyhow::{Result, Error, anyhow};

// Constants for audio performance - with optimized values for JACK
#[cfg(target_os = "linux")]
const OUTPUT_BUFFER_SIZE: usize = 8192;  // Much larger for Linux/JACK compatibility

#[cfg(not(target_os = "linux"))]
const OUTPUT_BUFFER_SIZE: usize = 4096;  // Smaller on non-Linux platforms

const UPDATE_RING_SIZE: usize = 8;       // Increased for smoother updates
const WAVETABLE_SIZE: usize = 4096;      // Size of wavetable for sine synthesis
pub const DEFAULT_UPDATE_RATE: f32 = 3.0; // Even higher to reduce CPU load for JACK

/// Configuration for resynthesis
pub struct ResynthConfig {
    pub gain: f32,
    pub smoothing: f32,
    pub freq_scale: f32,  // Frequency scaling factor (1.0 = normal, 2.0 = one octave up, 0.5 = one octave down)
    pub update_rate: f32, // How often to update synthesis (in seconds)
    pub needs_restart: Arc<AtomicBool>,  // Flag to signal when stream needs to restart
}

impl Default for ResynthConfig {
    fn default() -> Self {
        Self {
            gain: 0.25,
            smoothing: 0.0,
            freq_scale: 1.0,  // Default to no scaling
            update_rate: DEFAULT_UPDATE_RATE,
            needs_restart: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Parameter update structure
#[derive(Clone)]
struct SynthUpdate {
    partials: Vec<Vec<(f32, f32)>>,
    gain: f32,
    freq_scale: f32,
    smoothing: f32,  // Add smoothing parameter for crossfade control
}

/// State for a single partial
#[derive(Clone)]
struct PartialState {
    freq: f32,               // Current frequency
    target_freq: f32,        // Target frequency 
    amp: f32,
    phase: f32,
    phase_delta: f32,
    target_amp: f32,
    current_amp: f32,
}

impl PartialState {
    fn new() -> Self {
        Self {
            freq: 0.0,
            target_freq: 0.0,
            amp: 0.0,
            phase: 0.0,
            phase_delta: 0.0,
            target_amp: 0.0,
            current_amp: 0.0,
        }
    }
}

/// Lock-free synthesis engine
#[derive(Clone)]
struct WavetableSynth {
    // Sine wavetable for efficient synthesis
    wavetable: Vec<f32>,
    
    // Current state for all partials
    partials: Vec<Vec<PartialState>>,
    
    // Sample rate for phase calculation
    sample_rate: f32,
    
    // Pre-allocated output buffer to avoid allocations in audio thread
    output_buffer: Vec<f32>,
    
    // Channels for parameter updates - lock-free queue
    update_queue: Arc<ArrayQueue<SynthUpdate>>,
    
    // Current parameter values
    current_gain: f32,
    current_freq_scale: f32,
    smoothing: f32,  // Add smoothing parameter for crossfade control
}

impl WavetableSynth {
    fn new(num_channels: usize, num_partials: usize, sample_rate: f32, update_queue: Arc<ArrayQueue<SynthUpdate>>) -> Self {
        // Create sine wavetable
        let wavetable = (0..WAVETABLE_SIZE)
            .map(|i| (i as f32 / WAVETABLE_SIZE as f32 * 2.0 * PI).sin())
            .collect();
        
        // Initialize partials
        let partials = (0..num_channels)
            .map(|_| (0..num_partials)
                .map(|_| PartialState::new())
                .collect())
            .collect();
        
        // Pre-allocate output buffer (stereo by default)
        let output_buffer = vec![0.0; 2];
        
        Self {
            wavetable,
            partials,
            sample_rate,
            output_buffer,
            update_queue,
            current_gain: 0.25,
            current_freq_scale: 1.0,
            smoothing: 0.0,  // Add smoothing parameter for crossfade control
        }
    }
    
    /// Apply any pending updates
    fn apply_updates(&mut self) {
        // Process all available updates, keeping only the most recent
        while let Some(update) = self.update_queue.pop() {
            // Update global parameters
            self.current_gain = update.gain;
            self.current_freq_scale = update.freq_scale;
            self.smoothing = update.smoothing;
            
            // Update partial parameters
            for (ch, channel_partials) in update.partials.iter().enumerate() {
                if ch < self.partials.len() {
                    for (i, &(freq, amp)) in channel_partials.iter().enumerate() {
                        if i < self.partials[ch].len() {
                            let partial = &mut self.partials[ch][i];
                            
                            // Store target frequency (with scaling applied)
                            let target_freq = freq * self.current_freq_scale;
                            partial.target_freq = target_freq;
                            
                            // If smoothing is disabled, immediately update freq
                            if self.smoothing < 0.001 {
                                partial.freq = target_freq;
                                // Calculate phase_delta for direct frequency changes
                                partial.phase_delta = target_freq * WAVETABLE_SIZE as f32 / self.sample_rate;
                            }
                            
                            // Update amplitude target directly (no smoothing for amplitude)
                            partial.target_amp = amp / 100.0;  // Convert from dB scale
                            partial.current_amp = partial.target_amp;
                        }
                    }
                }
            }
        }
    }
    
    /// Builds a buffer for resampling when needed
    fn resample_buffer(_buffer: &mut [f32], _channels: usize) {
        // Use a simpler processing approach on resource-constrained hardware
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        {
            // Process in bigger chunks for better efficiency
            for frame in (0.._buffer.len() / _channels).step_by(4) {
                for ch in 0.._channels {
                    let idx = frame * _channels + ch;
                    if idx + 3 * _channels < _buffer.len() {
                        // Simple lowpass filtering
                        _buffer[idx] = _buffer[idx] * 0.7 + _buffer[idx + _channels] * 0.3;
                    }
                }
            }
        }
    }
    
    /// Process a complete buffer of audio
    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        // Early return if buffer is empty or invalid
        if buffer.is_empty() || channels == 0 {
            return;
        }

        // Check if output buffer needs resizing
        if self.output_buffer.len() < channels {
            self.output_buffer.resize(channels, 0.0);
        }

        // Apply any pending parameter updates first
        self.apply_updates();
        
        // Calculate frames to process
        let frames = buffer.len() / channels;
        
        // Calculate frequency crossfade rate based on smoothing parameter
        let crossfade_rate = if self.smoothing < 0.001 {
            1.0  // No smoothing = immediate changes
        } else {
            self.smoothing
        };
        
        // Process each frame
        for frame in 0..frames {
            // Clear output buffer
            for ch in 0..channels {
                self.output_buffer[ch] = 0.0;
            }
            
            // Process all partials
            for (ch_idx, channel) in self.partials.iter_mut().enumerate() {
                for partial in channel.iter_mut() {
                    // Apply frequency smoothing/crossfade if needed
                    if (partial.freq - partial.target_freq).abs() > 0.01 {
                        partial.freq = partial.freq * crossfade_rate + partial.target_freq * (1.0 - crossfade_rate);
                        partial.phase_delta = partial.freq * WAVETABLE_SIZE as f32 / self.sample_rate;
                    } else {
                        partial.freq = partial.target_freq;
                    }
                    
                    // Only generate audio if we have a frequency and amplitude
                    if partial.freq > 0.0 && partial.target_amp > 0.0 {
                        let idx_f = partial.phase % WAVETABLE_SIZE as f32;
                        let idx1 = idx_f as usize;
                        let idx2 = (idx1 + 1) % WAVETABLE_SIZE;
                        let frac = idx_f - idx1 as f32;
                        
                        let sample = self.wavetable[idx1] * (1.0 - frac) + self.wavetable[idx2] * frac;
                        let output_sample = sample * partial.target_amp;
                        
                        let output_ch = ch_idx % channels;
                        self.output_buffer[output_ch] += output_sample;
                        
                        partial.phase = (partial.phase + partial.phase_delta) % WAVETABLE_SIZE as f32;
                    }
                }
            }
            
            // Apply gain and write to output buffer with bounds checking
            let frame_start = frame * channels;
            if frame_start + channels <= buffer.len() {
                for ch in 0..channels {
                    let idx = frame_start + ch;
                    buffer[idx] = (self.output_buffer[ch] * self.current_gain).clamp(-1.0, 1.0);
                }
            }
        }
    }
}

/// Starts a thread that performs real-time resynthesis of the analyzed spectrum.
pub fn start_resynth_thread(
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    config: Arc<Mutex<ResynthConfig>>,
    device_index: pa::DeviceIndex,
    sample_rate: f64,
    shutdown_flag: Arc<AtomicBool>,
) {
    // Initialize channels
    let (num_channels, num_partials) = {
        let spec_app = spectrum_app.lock().unwrap();
        (spec_app.clone_absolute_data().len(), 
         if !spec_app.clone_absolute_data().is_empty() {
             spec_app.clone_absolute_data()[0].len()
         } else {
             DEFAULT_NUM_PARTIALS
         })
    };
    
    // Create synthesis engine and shared queue between threads
    let update_queue = Arc::new(ArrayQueue::new(UPDATE_RING_SIZE * 2));
    
    // Create update thread to feed the synth with new analysis data
    let update_thread_flag = Arc::new(AtomicBool::new(true));
    let update_thread_flag_clone = Arc::clone(&update_thread_flag);
    let shutdown_flag_clone = Arc::clone(&shutdown_flag);
    let update_queue_clone = Arc::clone(&update_queue);
    let config_for_thread = Arc::clone(&config);
    
    let _update_thread = thread::spawn(move || {
        let mut last_update = Instant::now();
        
        while !shutdown_flag_clone.load(Ordering::Relaxed) && update_thread_flag_clone.load(Ordering::Relaxed) {
            // Get current config
            if let Ok(config_guard) = config_for_thread.lock() {
                let update_rate = config_guard.update_rate;
                let gain = config_guard.gain;
                let freq_scale = config_guard.freq_scale;
                let smoothing = config_guard.smoothing;
                drop(config_guard);
                
                if last_update.elapsed().as_secs_f32() >= update_rate {
                    // Get new partial data and config
                    if let Ok(spec_guard) = spectrum_app.lock() {
                        let partials = spec_guard.clone_absolute_data();
                        drop(spec_guard);
                        
                        // Create parameter update
                        let update = SynthUpdate {
                            partials,
                            gain,
                            freq_scale,
                            smoothing,
                        };
                        
                        // Submit update to the queue
                        let _ = update_queue_clone.push(update);
                        last_update = Instant::now();
                    }
                }
            }
            
            thread::sleep(Duration::from_millis(20));
        }
    });

    // Main thread that handles audio stream lifecycle
    std::thread::spawn(move || {
        let mut current_stream: Option<pa::Stream<pa::NonBlocking, pa::Output<f32>>> = None;
        let mut last_restart_time = Instant::now();
        let mut consecutive_errors = 0;
        
        // Store the restart flag for checking
        let needs_restart = Arc::clone(&config.lock().unwrap().needs_restart);
        
        // Main loop continues until shutdown
        while !shutdown_flag.load(Ordering::Relaxed) {
            // Check if we need to restart
            let needs_restart_now = needs_restart.load(Ordering::Relaxed) || current_stream.is_none();
            
            // Debounce restarts with exponential backoff
            let backoff_time = if consecutive_errors > 0 {
                Duration::from_millis((500 * consecutive_errors as u64).min(5000))
            } else {
                Duration::from_millis(500)
            };
            
            if needs_restart_now && last_restart_time.elapsed() < backoff_time {
                thread::sleep(Duration::from_millis(50));
                continue;
            }
            
            // Handle restart case
            if needs_restart_now {
                info!("Restarting audio stream...");
                
                // Clean up existing stream if any
                if let Some(mut stream) = current_stream.take() {
                    let _ = stream.stop();
                    // Add small delay after stopping the stream
                    thread::sleep(Duration::from_millis(100));
                }
                
                // Mark restart handled
                needs_restart.store(false, Ordering::Relaxed);
                last_restart_time = Instant::now();
                
                // Setup new stream with retry logic
                match setup_audio_stream(
                    device_index,
                    sample_rate,
                    Arc::clone(&update_queue),
                    num_channels,
                    num_partials
                ) {
                    Ok(stream) => {
                        current_stream = Some(stream);
                        consecutive_errors = 0;
                        info!("Audio stream restarted successfully");
                    },
                    Err(e) => {
                        consecutive_errors += 1;
                        error!("Failed to restart audio stream (attempt {}): {}", consecutive_errors, e);
                        // Let the next iteration handle the retry
                        continue;
                    }
                }
            }
            
            // Check stream health if we have one
            if let Some(ref stream) = current_stream {
                if let Err(e) = stream.is_active() {
                    error!("Stream health check failed: {}, will restart", e);
                    current_stream = None;
                    consecutive_errors += 1;
                }
            }
            
            // Sleep a bit before next check
            thread::sleep(Duration::from_millis(100));
        }
        
        // Shutdown - stop stream if it exists
        if let Some(mut stream) = current_stream {
            let _ = stream.stop();
        }
        
        // Stop the update thread
        update_thread_flag.store(false, Ordering::Relaxed);
        info!("Resynthesis thread shutting down");
    });
}

/// Sets up a PortAudio output stream with the given parameters
fn setup_audio_stream(
    device_index: pa::DeviceIndex,
    sample_rate: f64, 
    update_queue: Arc<ArrayQueue<SynthUpdate>>,
    num_channels: usize,
    num_partials: usize
) -> Result<pa::Stream<pa::NonBlocking, pa::Output<f32>>, anyhow::Error> {
    // Initialize PortAudio
    let pa = pa::PortAudio::new()?;
    
    // Get device info
    let device_info = pa.device_info(device_index)?;
    
    // Create output parameters
    let output_params = pa::StreamParameters::<f32>::new(
        device_index,
        2, // Stereo output
        true,
        device_info.default_low_output_latency
    );
    
    // Choose a supported sample rate
    let supported_sample_rate = match pa.is_output_format_supported(output_params, sample_rate) {
        Ok(_) => sample_rate,
        Err(_) => {
            // Try common sample rates
            let fallback_sample_rates = [44100.0, 48000.0, 96000.0, 192000.0];
            let supported_rate = fallback_sample_rates
                .iter()
                .find(|&&rate| pa.is_output_format_supported(output_params, rate).is_ok())
                .copied()
                .unwrap_or(44100.0);
            
            info!("Sample rate {} not supported, using {} instead", sample_rate, supported_rate);
            supported_rate
        }
    };
    
    // Create output settings
    let settings = pa::stream::OutputSettings::new(
        output_params,
        supported_sample_rate,
        OUTPUT_BUFFER_SIZE as u32
    );
    
    // Create synthesizer for the callback
    let synth_for_callback = std::cell::RefCell::new(WavetableSynth::new(
        num_channels, 
        num_partials,
        supported_sample_rate as f32,
        update_queue
    ));
    
    // Create callback function
    let callback = move |args: pa::OutputStreamCallbackArgs<f32>| {
        // Use borrow_mut() to get mutable access
        let mut synth = synth_for_callback.borrow_mut();
        synth.process_buffer(args.buffer, 2); // 2 = stereo
        pa::Continue
    };
    
    // Open and start the stream
    let mut stream = pa.open_non_blocking_stream(settings, callback)?;
    stream.start()?;
    
    Ok(stream)
} 