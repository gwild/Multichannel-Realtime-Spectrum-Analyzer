use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;
use portaudio as pa;
use log::{info, error, debug};
use crate::plot::SpectrumApp;
use crate::DEFAULT_NUM_PARTIALS;
use crossbeam_queue::ArrayQueue;
use anyhow::{Result, Error, anyhow};
use crate::SharedMemory;  // Import SharedMemory from main.rs

// Constants for audio performance - with optimized values for JACK
#[cfg(target_os = "linux")]
const OUTPUT_BUFFER_SIZE: usize = 16384;  // Much larger for Linux/JACK compatibility

#[cfg(not(target_os = "linux"))]
const OUTPUT_BUFFER_SIZE: usize = 4096;  // Smaller on non-Linux platforms

const UPDATE_RING_SIZE: usize = 16;       // Increased for smoother updates
pub const DEFAULT_UPDATE_RATE: f32 = 1.0; // Default update rate in seconds

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
            gain: 0.5,
            smoothing: 0.0,
            freq_scale: 1.0,  // Default to no scaling
            update_rate: DEFAULT_UPDATE_RATE,
            needs_restart: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ResynthConfig {
    // Create a snapshot of current settings without holding a lock
    pub fn snapshot(&self) -> ResynthConfigSnapshot {
        ResynthConfigSnapshot {
            gain: self.gain,
            smoothing: self.smoothing,
            freq_scale: self.freq_scale,
            update_rate: self.update_rate,
        }
    }
}

/// Snapshot of resynth config for lock-free access
#[derive(Clone, Copy)]
pub struct ResynthConfigSnapshot {
    pub gain: f32,
    pub smoothing: f32,
    pub freq_scale: f32,
    pub update_rate: f32,
}

/// Parameter update structure
#[derive(Clone)]
struct SynthUpdate {
    partials: Vec<Vec<(f32, f32)>>,
    gain: f32,
    freq_scale: f32,
    smoothing: f32,  // Add smoothing parameter for crossfade control
    update_rate: f32,  // Add update rate parameter
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
    current_wavetable: Vec<Vec<f32>>,
    next_wavetable: Vec<Vec<f32>>,
    sample_counter: usize,
    wavetable_size: usize,
    crossfade_start: usize,    // Point at which crossfade begins (2/3 through wavetable)
    crossfade_length: usize,   // Length of crossfade period (1/3 of wavetable)
    in_crossfade: bool,
    sample_rate: f32,
    update_rate: f32,
    output_buffer: Vec<f32>,
    update_queue: Arc<ArrayQueue<SynthUpdate>>,
    current_gain: f32,
}

impl WavetableSynth {
    fn build_wavetable(partials: &[(f32, f32)], sample_rate: f32, update_rate: f32) -> Vec<f32> {
        let wavetable_size = (sample_rate * update_rate) as usize;
        let mut table = vec![0.0; wavetable_size];
        
        debug!("Building wavetable with {} samples from partials:", wavetable_size);
        for &(freq, amp) in partials {
            if freq > 0.0 && amp > 0.0 {
                let amplitude = (amp / 20.0).exp(); // Convert dB to linear amplitude
                debug!("  Freq: {:.1} Hz, Amp: {:.1} dB -> Linear: {:.3}", freq, amp, amplitude);
            }
        }

        // Generate wavetable exactly matching the update period
        for i in 0..wavetable_size {
            let t = (i as f32 / wavetable_size as f32) * update_rate; // Exact time in seconds over update period
            for &(freq, amp) in partials {
                if freq > 0.0 && amp > 0.0 {
                    let amplitude = (amp / 20.0).exp(); // Convert dB to linear amplitude
                    table[i] += amplitude * (2.0 * PI * freq * t).sin();
                }
            }
        }

        table
    }

    fn new(num_channels: usize, sample_rate: f32, update_queue: Arc<ArrayQueue<SynthUpdate>>) -> Self {
        let update_rate = DEFAULT_UPDATE_RATE;
        let wavetable_size = (sample_rate * update_rate) as usize;
        let crossfade_length = wavetable_size / 3;  // Exactly 1/3 of wavetable
        let crossfade_start = wavetable_size - crossfade_length;  // Start at 2/3 point

        Self {
            current_wavetable: vec![vec![0.0; wavetable_size]; num_channels],
            next_wavetable: vec![vec![0.0; wavetable_size]; num_channels],
            sample_counter: 0,
            wavetable_size,
            crossfade_start,
            crossfade_length,
            in_crossfade: false,
            sample_rate,
            update_rate,
            output_buffer: vec![0.0; num_channels],
            update_queue,
            current_gain: 0.25,
        }
    }

    fn resize_state_vectors(&mut self, num_channels: usize) {
        self.current_wavetable = vec![vec![0.0; self.wavetable_size]; num_channels];
        self.next_wavetable = vec![vec![0.0; self.wavetable_size]; num_channels];
        self.output_buffer = vec![0.0; num_channels];
    }

    fn apply_updates(&mut self) {
        let mut new_update = None;
        while let Some(update) = self.update_queue.pop() {
            new_update = Some(update);
        }

        if let Some(update) = new_update {
            debug!("Received new partials update:");
            for (ch, ch_partials) in update.partials.iter().enumerate() {
                debug!("Channel {}: {}", ch, format_partials_debug(ch_partials));
            }

            self.current_gain = update.gain;

            // Check if update rate has changed significantly
            if (self.update_rate - update.update_rate).abs() > f32::EPSILON {
                let old_size = self.wavetable_size;
                self.update_rate = update.update_rate;
                let new_wavetable_size = (self.sample_rate * self.update_rate) as usize;
                let new_crossfade_length = new_wavetable_size / 3;
                let new_crossfade_start = new_wavetable_size - new_crossfade_length;

                // Generate new wavetables safely
                let mut new_current_wavetable = vec![vec![0.0; new_wavetable_size]; self.current_wavetable.len()];
                let mut new_next_wavetable = vec![vec![0.0; new_wavetable_size]; self.next_wavetable.len()];

                // If we're in a crossfade, we need to preserve both current and next wavetables
                if self.in_crossfade {
                    for ch in 0..self.current_wavetable.len() {
                        // Resample current wavetable to new size
                        for i in 0..new_wavetable_size {
                            let old_idx = (i as f32 * old_size as f32 / new_wavetable_size as f32) as usize;
                            new_current_wavetable[ch][i] = self.current_wavetable[ch][old_idx % old_size];
                        }
                        // Generate new next wavetable
                        new_next_wavetable[ch] = Self::build_wavetable(
                            &update.partials[ch],
                            self.sample_rate,
                            self.update_rate
                        );
                    }
                    // Adjust sample counter position
                    self.sample_counter = (self.sample_counter as f32 * new_wavetable_size as f32 / old_size as f32) as usize;
                } else {
                    // Not in crossfade, just generate new wavetable
                    for (ch, ch_partials) in update.partials.iter().enumerate() {
                        new_next_wavetable[ch] = Self::build_wavetable(
                            ch_partials,
                            self.sample_rate,
                            self.update_rate
                        );
                    }
                    self.sample_counter = 0;
                }

                // Atomically swap in new wavetables and parameters
                self.current_wavetable = new_current_wavetable;
                self.next_wavetable = new_next_wavetable;
                self.wavetable_size = new_wavetable_size;
                self.crossfade_length = new_crossfade_length;
                self.crossfade_start = new_crossfade_start;

                info!("Update rate changed safely: {} s, new wavetable size: {}", 
                    self.update_rate, self.wavetable_size);
            } else {
                // Regular update without resizing
                for (ch, ch_partials) in update.partials.iter().enumerate() {
                    self.next_wavetable[ch] = Self::build_wavetable(
                        ch_partials,
                        self.sample_rate,
                        self.update_rate
                    );
                    debug!("Generated new wavetable for channel {} with {} partials", 
                        ch, ch_partials.len());
                }

                // Only reset sample counter if not in crossfade
                if !self.in_crossfade {
                    self.sample_counter = 0;
                }
            }
        }
    }

    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        if buffer.is_empty() || channels == 0 {
            return;
        }

        self.apply_updates();
        let frames = buffer.len() / channels;

        for frame in 0..frames {
            // Ensure sample_counter stays within bounds
            if self.sample_counter >= self.wavetable_size {
                self.sample_counter = 0;
                if self.in_crossfade {
                    std::mem::swap(&mut self.current_wavetable, &mut self.next_wavetable);
                    self.in_crossfade = false;
                }
            }

            // Calculate precise linear crossfade for 1/3 overlap
            let crossfade = if self.sample_counter >= self.crossfade_start {
                self.in_crossfade = true;
                let fade_pos = (self.sample_counter - self.crossfade_start) as f32 
                    / self.crossfade_length as f32;
                fade_pos.clamp(0.0, 1.0) // Linear crossfade
            } else {
                0.0
            };

            // Process each channel with precise amplitude control
            for ch in 0..channels {
                if ch >= self.current_wavetable.len() || self.sample_counter >= self.current_wavetable[ch].len() {
                    continue;
                }

                // Ensure perfect complementary crossfade
                let fade_out = 1.0 - crossfade;
                let fade_in = crossfade;

                let curr_sample = self.current_wavetable[ch][self.sample_counter] * fade_out;
                let next_sample = self.next_wavetable[ch][self.sample_counter] * fade_in;

                // Sum the crossfaded samples
                let sample = curr_sample + next_sample;

                let idx = frame * channels + ch;
                if idx < buffer.len() {
                    let final_sample = sample * self.current_gain;
                    buffer[idx] = if final_sample.abs() > 1.0 {
                        final_sample.signum() * (1.0 - (-final_sample.abs()).exp())
                    } else {
                        final_sample
                    };
                }
            }

            self.sample_counter += 1;
        }
    }
}

/// Starts a thread that performs real-time resynthesis of the analyzed spectrum.
pub fn start_resynth_thread(
    spectrum_app: Arc<Mutex<SpectrumApp>>,  // Remove underscore to use this parameter
    config: Arc<Mutex<ResynthConfig>>,
    device_index: pa::DeviceIndex,
    sample_rate: f64,
    shutdown_flag: Arc<AtomicBool>,
    _shared_partials: Option<SharedMemory>, // Mark as unused
) {
    // Initialize channels
    let (num_channels, num_partials) = {
        let spec_app = spectrum_app.lock().unwrap();
        let data = spec_app.clone_absolute_data();
        (data.len(), 
         if !data.is_empty() {
             data[0].len()
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
    let spectrum_app_clone = Arc::clone(&spectrum_app);

    let _update_thread = thread::spawn(move || {
        let mut last_update = Instant::now();
        while !shutdown_flag_clone.load(Ordering::Relaxed) && update_thread_flag_clone.load(Ordering::Relaxed) {
            // Clone config and spectrum data with minimal lock duration
            let (partials, gain, freq_scale, smoothing, update_rate) = {
                let config_snapshot = config_for_thread.lock().unwrap().snapshot();

                // Lock spectrum_app briefly to clone partials data
                let partials_data = {
                    let spec_app = spectrum_app_clone.lock().unwrap();
                    spec_app.clone_absolute_data()
                };

                (
                    partials_data,
                    config_snapshot.gain,
                    config_snapshot.freq_scale,
                    config_snapshot.smoothing,
                    config_snapshot.update_rate,
                )
            };

            // Only do heavy computation (wavetable build, etc.) after releasing locks
            if last_update.elapsed().as_secs_f32() >= update_rate {
                debug!("Time for update: elapsed={}s, update_rate={}s", last_update.elapsed().as_secs_f32(), update_rate);
                // Create parameter update
                let update = SynthUpdate {
                    partials,
                    gain,
                    freq_scale,
                    smoothing,
                    update_rate,  // Add update_rate to the update
                };
                // Submit update to the queue
                match update_queue_clone.push(update) {
                    Ok(_) => debug!("Successfully pushed update to queue"),
                    Err(_) => debug!("Failed to push update to queue, queue might be full"),
                }
                last_update = Instant::now();
            } else {
                debug!("Skipping update: elapsed={}s, update_rate={}s", last_update.elapsed().as_secs_f32(), update_rate);
            }
            thread::sleep(Duration::from_millis(20));
        }
        debug!("Resynth update thread shutting down");
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
                debug!("Restarting audio stream...");
                
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
                        debug!("Audio stream restarted successfully");
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

fn format_partials_debug(partials: &[(f32, f32)]) -> String {
    partials.iter()
        .filter(|&&(freq, amp)| freq > 0.0 && amp > 0.0)
        .map(|&(freq, amp)| format!("({:.1} Hz, {:.1} dB)", freq, amp))
        .collect::<Vec<_>>()
        .join(", ")
} 