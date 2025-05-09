use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::cell::RefCell;
use portaudio as pa;
use log::{info, error, debug, warn};
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

pub const UPDATE_RING_SIZE: usize = 64;       // Increased for smoother updates and to prevent queue overflow
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
            gain: 0.8,  // Increased default gain for better audibility
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
#[derive(Clone, Debug)]
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
    wavetable_queue: Arc<ArrayQueue<Vec<Vec<f32>>>>,  // Queue for new wavetables
    sample_counter: usize,
    wavetable_size: usize,
    crossfade_start: usize,
    crossfade_length: usize,
    in_crossfade: bool,
    sample_rate: f32,
    update_rate: f32,
    current_gain: f32,
    update_queue: Arc<ArrayQueue<SynthUpdate>>,
    needs_resize: bool,  // Flag to indicate wavetable resize needed
    new_size: usize,    // New size for resize operation
}

impl WavetableSynth {
    fn new(num_channels: usize, sample_rate: f32, update_queue: Arc<ArrayQueue<SynthUpdate>>) -> Self {
        let update_rate = DEFAULT_UPDATE_RATE;
        let wavetable_size = (sample_rate * update_rate) as usize;
        let crossfade_length = wavetable_size / 3;
        let crossfade_start = wavetable_size - crossfade_length;
        Self {
            current_wavetable: vec![vec![0.0; wavetable_size]; num_channels],
            next_wavetable: vec![vec![0.0; wavetable_size]; num_channels],
            wavetable_queue: Arc::new(ArrayQueue::new(10)),
            sample_counter: 0,
            wavetable_size,
            crossfade_start,
            crossfade_length,
            in_crossfade: false,
            sample_rate,
            update_rate,
            current_gain: 0.25,
            update_queue,
            needs_resize: false,
            new_size: wavetable_size,
        }
    }

    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        // Check for updates
        self.apply_updates();

        // Check if resize is needed
        if self.needs_resize {
            self.resize_wavetables(self.new_size);
            self.needs_resize = false;
        }

        let frames = buffer.len() / channels;

        for frame in 0..frames {
            // Reset counter and swap buffers if needed
            if self.sample_counter >= self.wavetable_size {
                self.sample_counter = 0;
                if self.in_crossfade {
                    std::mem::swap(&mut self.current_wavetable, &mut self.next_wavetable);
                    self.in_crossfade = false;
                }
            }

            // Calculate crossfade coefficients using smoother curve
            let (fade_out, fade_in) = if self.in_crossfade && self.sample_counter >= self.crossfade_start {
                let progress = (self.sample_counter - self.crossfade_start) as f32 / self.crossfade_length as f32;
                let fade = 0.5 * (1.0 - (progress * std::f32::consts::PI).cos()); // Cosine interpolation
                (1.0 - fade, fade)
            } else {
                (1.0, 0.0)
            };

            // Process each channel
            for ch in 0..channels {
                if ch >= self.current_wavetable.len() {
                    continue;
                }

                let curr_sample = self.current_wavetable[ch][self.sample_counter] * fade_out;
                let next_sample = if self.in_crossfade {
                    self.next_wavetable[ch][self.sample_counter] * fade_in
                } else {
                    0.0
                };

                buffer[frame * channels + ch] = (curr_sample + next_sample) * self.current_gain;
            }

            self.sample_counter += 1;
        }
    }

    fn resize_wavetables(&mut self, new_size: usize) {
        let old_size = self.wavetable_size;
        self.wavetable_size = new_size;
        self.crossfade_length = new_size / 3;
        self.crossfade_start = new_size - self.crossfade_length;

        // Resize current wavetables with interpolation
        for channel in 0..self.current_wavetable.len() {
            let mut new_table = vec![0.0; new_size];
            for i in 0..new_size {
                let old_idx = (i as f32 * old_size as f32 / new_size as f32) as usize;
                let next_idx = (old_idx + 1).min(old_size - 1);
                let frac = (i as f32 * old_size as f32 / new_size as f32) - old_idx as f32;
                
                new_table[i] = self.current_wavetable[channel][old_idx] * (1.0 - frac) +
                              self.current_wavetable[channel][next_idx] * frac;
            }
            self.current_wavetable[channel] = new_table;
        }

        // Resize next wavetables
        for channel in 0..self.next_wavetable.len() {
            self.next_wavetable[channel] = vec![0.0; new_size];
        }

        // Adjust sample counter
        self.sample_counter = (self.sample_counter as f32 * new_size as f32 / old_size as f32) as usize;
    }

    fn build_wavetable(partials: &[(f32, f32)], sample_rate: f32, update_rate: f32) -> Vec<f32> {
        let wavetable_size = (sample_rate * update_rate) as usize;
        let mut table = vec![0.0; wavetable_size];
        let mut max_amplitude: f32 = 0.0;
        
        // Generate wavetable
        for i in 0..wavetable_size {
            let t = i as f32 / sample_rate;  // Time in seconds
            for &(freq, amp) in partials {
                if freq > 0.0 && amp > 0.0 {
                    // Convert dB to linear amplitude
                    let amplitude = (amp / 20.0).exp();
                    table[i] += amplitude * (2.0 * PI * freq * t).sin();
                    max_amplitude = max_amplitude.max(amplitude);
                }
            }
        }

        // Normalize to prevent clipping
        if max_amplitude > 1.0 {
            for sample in &mut table {
                *sample /= max_amplitude;
            }
        }

        table
    }

    fn apply_updates(&mut self) {
        while let Some(update) = self.update_queue.pop() {
            // Update parameters
            self.current_gain = update.gain;
            
            // Check if update rate has changed
            if update.update_rate != self.update_rate {
                self.update_rate = update.update_rate;
                let new_size = (self.sample_rate * self.update_rate) as usize;
                if new_size != self.wavetable_size {
                    self.new_size = new_size;
                    self.needs_resize = true;
                }
            }

            // Process partials to build new wavetables
            let mut new_wavetables = Vec::with_capacity(update.partials.len());
            for partials in &update.partials {
                new_wavetables.push(Self::build_wavetable(partials, self.sample_rate, update.update_rate));
            }

            // Try to push to queue, with retry on failure
            if let Ok(_) = self.wavetable_queue.push(new_wavetables.clone()) {
                debug!("Successfully pushed new wavetables to queue");
            } else {
                debug!("Queue full, retrying after clearing");
                // Clear queue and try again
                while self.wavetable_queue.pop().is_some() {}
                let _ = self.wavetable_queue.push(new_wavetables);
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
    shared_memory: Option<SharedMemory>,
    num_channels: usize,
    num_partials: usize,
) {
    debug!("Module path for resynth: {}", module_path!());
    debug!("Using fixed num_channels: {}, num_partials: {}", num_channels, num_partials);

    // Create synthesis engine and shared queue between threads
    let update_queue = Arc::new(ArrayQueue::new(UPDATE_RING_SIZE * 2));

    // Get the resynthesis queue from shared memory
    let resynth_queue = if let Some(shared) = shared_memory {
        match shared.resynth_queue {
            Some(queue) => {
                debug!("[RESYNTH SETUP] Using shared memory queue from fft_analysis");
                Some(queue)
            },
            None => {
                error!("[RESYNTH SETUP] Shared memory exists but resynth_queue is None");
                None
            }
        }
    } else {
        error!("[RESYNTH SETUP] Shared memory not available - fft_analysis data cannot be accessed");
        None
    };
    
    // Log whether a queue is available
    if resynth_queue.is_some() {
        debug!("[RESYNTH SETUP] Resynth queue initialized successfully");
    } else {
        error!("[RESYNTH SETUP] Failed to initialize resynth queue - fft_analysis may not be providing data");
    }

    // Create update thread to feed the synth with new analysis data
    let update_thread_flag = Arc::new(AtomicBool::new(true));
    let update_thread_flag_clone = Arc::clone(&update_thread_flag);
    let shutdown_flag_clone = Arc::clone(&shutdown_flag);
    let update_queue_clone = Arc::clone(&update_queue);
    let config_for_thread = Arc::clone(&config);

    let _update_thread = thread::spawn(move || {
        let mut last_update = Instant::now();
        while !shutdown_flag_clone.load(Ordering::Relaxed) && update_thread_flag_clone.load(Ordering::Relaxed) {
            // First check if it's time to update with minimal locking
            let update_rate = {
                let config = config_for_thread.lock().unwrap();
                config.update_rate
            };

            // Only get new data if it's time for an update
            if last_update.elapsed().as_secs_f32() >= update_rate {
                // Get config snapshot with minimal lock duration
                let config_snapshot = config_for_thread.lock().unwrap().snapshot();

                // Get partials data from dedicated queue (no GUI locking)
                if let Some(queue) = &resynth_queue {
                    match queue.pop() {
                        Some(partials_data) => {
                            debug!("[RESYNTH UPDATE] Successfully retrieved partials data from shared memory queue");
                            let update = SynthUpdate {
                                partials: partials_data,
                                gain: config_snapshot.gain,
                                freq_scale: config_snapshot.freq_scale,
                                smoothing: config_snapshot.smoothing,
                                update_rate: config_snapshot.update_rate,
                            };

                            // Push update to synth
                            debug!("[RESYNTH UPDATE] Attempting to push update to queue: gain={}, freq_scale={}, update_rate={}", update.gain, update.freq_scale, update.update_rate);
                            match update_queue_clone.push(update) {
                                Ok(_) => debug!("[RESYNTH UPDATE] Successfully pushed update to queue"),
                                Err(e) => debug!("[RESYNTH UPDATE] Failed to push update to queue: {:?}", e),
                            }
                            last_update = Instant::now();
                        },
                        None => {
                            debug!("[RESYNTH UPDATE] Shared memory queue is empty, no partials data available");
                        }
                    }
                } else {
                    debug!("[RESYNTH UPDATE] Shared memory queue not initialized, unable to retrieve partials data");
                }
            }

            // Sleep longer when not updating to reduce CPU usage
            let sleep_duration = if last_update.elapsed().as_secs_f32() >= update_rate {
                // Short sleep when actively updating
                Duration::from_millis(1)
            } else {
                // Longer sleep when waiting for next update
                // Sleep for 1/10th of remaining time until next update, or 10ms minimum
                let time_to_next = (update_rate - last_update.elapsed().as_secs_f32()).max(0.0);
                Duration::from_secs_f32((time_to_next / 10.0).max(0.010))
            };
            thread::sleep(sleep_duration);
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
            
            if needs_restart_now && last_restart_time.elapsed() >= backoff_time {
                debug!("Setting up new audio stream");
                match setup_audio_stream(
                    device_index,
                    sample_rate,
                    Arc::clone(&update_queue),
                    num_channels,
                    num_partials
                ) {
                    Ok(stream) => {
                        debug!("Successfully created new audio stream");
                        current_stream = Some(stream);
                        consecutive_errors = 0;
                        needs_restart.store(false, Ordering::Relaxed);
                    },
                    Err(e) => {
                        error!("Failed to create audio stream: {}", e);
                        consecutive_errors += 1;
                    }
                }
                last_restart_time = Instant::now();
            }

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
        num_channels as i32,
        true,
        device_info.default_low_output_latency
    );

    let settings = pa::OutputStreamSettings::new(output_params, sample_rate, OUTPUT_BUFFER_SIZE as u32);
    
    // Create wavetable synth
    let synth = WavetableSynth::new(num_channels, sample_rate as f32, update_queue);
    let synth = Arc::new(Mutex::new(synth));
    
    // Create stream with callback
    let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
        if let Ok(mut synth) = synth.lock() {
            synth.process_buffer(buffer, num_channels);
        }
        pa::Continue
    };

    // Open stream
    let mut stream = pa.open_non_blocking_stream(settings, callback)?;

    // Start the stream
    stream.start()?;
    debug!("Audio stream started successfully");

    Ok(stream)
}

fn format_partials_debug(partials: &[(f32, f32)]) -> String {
    partials.iter()
        .filter(|&&(freq, amp)| freq > 0.0 && amp > 0.0)
        .map(|&(freq, amp)| format!("({:.1} Hz, {:.1} dB)", freq, amp))
        .collect::<Vec<_>>()
        .join(", ")
}
