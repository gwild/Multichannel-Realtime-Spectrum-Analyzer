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
const WAVETABLE_SIZE: usize = 4096;      // Size of wavetable for sine synthesis
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
            gain: 0.25,
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
    current_wavetables: Vec<Vec<f32>>,
    next_wavetables: Vec<Vec<f32>>,
    // Dual phase accumulators per channel
    curr_phases: Vec<f32>,
    next_phases: Vec<f32>,
    root_freqs: Vec<f32>,
    next_root_freqs: Vec<f32>,
    sample_counter: usize,
    crossfade_samples: usize,
    update_samples: usize,
    in_crossfade: bool,
    sample_rate: f32,
    output_buffer: Vec<f32>,
    update_queue: Arc<ArrayQueue<SynthUpdate>>,
    current_gain: f32,
    update_rate: f32,
}

impl WavetableSynth {
    fn resize_state_vectors(&mut self, num_channels: usize) {
        self.current_wavetables.resize(num_channels, vec![0.0; WAVETABLE_SIZE]);
        self.next_wavetables.resize(num_channels, vec![0.0; WAVETABLE_SIZE]);
        self.curr_phases.resize(num_channels, 0.0);
        self.next_phases.resize(num_channels, 0.0);
        self.root_freqs.resize(num_channels, 1.0);
        self.next_root_freqs.resize(num_channels, 1.0);
        self.output_buffer.resize(num_channels, 0.0);
    }

    fn new(num_channels: usize, _num_partials: usize, sample_rate: f32, update_queue: Arc<ArrayQueue<SynthUpdate>>) -> Self {
        let update_rate = DEFAULT_UPDATE_RATE;
        let update_samples = (update_rate * sample_rate) as usize;
        let crossfade_samples = (update_samples as f32 * 0.5) as usize; // crossfade is half the update period

        let mut synth = Self {
            current_wavetables: vec![],
            next_wavetables: vec![],
            curr_phases: vec![],
            next_phases: vec![],
            root_freqs: vec![],
            next_root_freqs: vec![],
            sample_counter: 0,
            crossfade_samples,
            update_samples,
            in_crossfade: false,
            sample_rate,
            output_buffer: vec![],
            update_queue,
            current_gain: 0.25,
            update_rate,
        };
        synth.resize_state_vectors(num_channels);
        synth
    }

    // Helper: build a wavetable from a set of partials (freq, amp), length = WAVETABLE_SIZE
    fn build_wavetable(partials: &[(f32, f32)], sample_rate: f32) -> Vec<f32> {
        let mut table = vec![0.0; WAVETABLE_SIZE];
        
        // Find the root frequency (first nonzero partial)
        let f0 = partials.iter()
            .find(|&&(f, a)| f > 0.0 && a > 0.0)
            .map(|&(f, _)| f)
            .unwrap_or(1.0);

        // Adjust f0 to ensure integer cycles in wavetable
        let cycles = (f0 * WAVETABLE_SIZE as f32 / sample_rate).round();
        let adjusted_f0 = cycles * sample_rate / WAVETABLE_SIZE as f32;
        
        // First pass - generate sine components
        for i in 0..WAVETABLE_SIZE {
            let t = i as f32 / WAVETABLE_SIZE as f32; // 0..1
            for &(freq, amp) in partials {
                if freq > 0.0 && amp > 0.0 {
                    let norm_amp = amp / 100.0; // scale dB to linear
                    // Adjust each partial frequency proportionally to adjusted f0
                    let adjusted_freq = freq * (adjusted_f0 / f0);
                    table[i] += norm_amp * (2.0 * PI * adjusted_freq * t).sin();
                }
            }
        }

        // Ensure zero-crossing at boundaries
        let start_val = table[0];
        let end_val = table[WAVETABLE_SIZE - 1];
        let correction = -(start_val + end_val) / 2.0;
        
        // Apply correction and normalize
        let mut max = 0.0_f32;
        for v in table.iter_mut() {
            *v += correction;
            max = max.max(v.abs());
        }
        
        // Normalize if needed
        if max > 1.0 {
            for v in table.iter_mut() {
                *v /= max;
            }
        }
        
        // Ensure exact zero at boundaries
        table[0] = 0.0;
        table[WAVETABLE_SIZE - 1] = 0.0;
        
        table
    }

    fn apply_updates(&mut self) {
        let mut new_update = None;
        while let Some(update) = self.update_queue.pop() {
            new_update = Some(update);
        }
        if let Some(update) = new_update {
            self.current_gain = update.gain;
            if self.update_rate != update.update_rate {
                self.update_rate = update.update_rate;
                self.update_samples = (self.update_rate * self.sample_rate) as usize;
                self.crossfade_samples = (self.update_samples as f32 * 0.5) as usize; // half the update period
            }

            let update_channels = update.partials.len();
            if update_channels != self.current_wavetables.len() {
                self.resize_state_vectors(update_channels);
            }

            // Start new crossfade immediately upon receiving new data
            self.sample_counter = 0;
            self.in_crossfade = true;

            for (ch, ch_partials) in update.partials.iter().enumerate() {
                self.next_wavetables[ch] = Self::build_wavetable(ch_partials, self.sample_rate);
                self.next_root_freqs[ch] = ch_partials.iter()
                    .find(|&&(f, a)| f > 0.0 && a > 0.0)
                    .map(|&(f, _)| f * update.freq_scale)
                    .unwrap_or(1.0);
                self.next_phases[ch] = self.curr_phases[ch]; // maintain phase continuity
            }
        }
    }

    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        if buffer.is_empty() || channels == 0 {
            return;
        }
        if self.output_buffer.len() < channels {
            self.output_buffer.resize(channels, 0.0);
        }
        self.apply_updates();
        let frames = buffer.len() / channels;

        for frame in 0..frames {
            let crossfade = if self.in_crossfade && self.sample_counter < self.crossfade_samples {
                let fade_pos = self.sample_counter as f32 / self.crossfade_samples as f32;
                0.5 * (1.0 - (PI * fade_pos).cos())
            } else {
                if self.in_crossfade {
                    // Crossfade complete, switch to new wavetables
                    self.current_wavetables = self.next_wavetables.clone();
                    self.root_freqs = self.next_root_freqs.clone();
                    self.curr_phases = self.next_phases.clone();
                    self.in_crossfade = false;
                }
                0.0
            };

            for ch in 0..channels {
                if ch >= self.output_buffer.len() || ch >= self.current_wavetables.len() || 
                   ch >= self.next_wavetables.len() || ch >= self.curr_phases.len() || 
                   ch >= self.next_phases.len() || ch >= self.root_freqs.len() || 
                   ch >= self.next_root_freqs.len() {
                    continue;
                }

                // Calculate phase increments based on root frequency
                let curr_freq = self.root_freqs[ch];
                let next_freq = self.next_root_freqs[ch];
                
                // Phase increment for one sample
                let curr_phase_inc = (curr_freq * WAVETABLE_SIZE as f32) / self.sample_rate;
                let next_phase_inc = (next_freq * WAVETABLE_SIZE as f32) / self.sample_rate;

                // Get current phases
                let curr_phase = self.curr_phases[ch];
                let next_phase = self.next_phases[ch];

                // Interpolate current wavetable
                let curr_idx_float = curr_phase % WAVETABLE_SIZE as f32;
                let curr_idx1 = curr_idx_float as usize;
                let curr_idx2 = (curr_idx1 + 1) % WAVETABLE_SIZE;
                let curr_frac = curr_idx_float - curr_idx1 as f32;
                
                let curr_sample = self.current_wavetables[ch][curr_idx1] * (1.0 - curr_frac) + 
                                self.current_wavetables[ch][curr_idx2] * curr_frac;

                // Interpolate next wavetable
                let next_idx_float = next_phase % WAVETABLE_SIZE as f32;
                let next_idx1 = next_idx_float as usize;
                let next_idx2 = (next_idx1 + 1) % WAVETABLE_SIZE;
                let next_frac = next_idx_float - next_idx1 as f32;
                
                let next_sample = self.next_wavetables[ch][next_idx1] * (1.0 - next_frac) + 
                                self.next_wavetables[ch][next_idx2] * next_frac;

                // Crossfade between current and next
                let sample = if self.in_crossfade {
                    curr_sample * (1.0 - crossfade) + next_sample * crossfade
                } else {
                    curr_sample
                };

                self.output_buffer[ch] = sample;

                // Update phases
                self.curr_phases[ch] = (curr_phase + curr_phase_inc) % WAVETABLE_SIZE as f32;
                self.next_phases[ch] = (next_phase + next_phase_inc) % WAVETABLE_SIZE as f32;
            }

            let frame_start = frame * channels;
            if frame_start + channels <= buffer.len() {
                for ch in 0..channels {
                    if ch >= self.output_buffer.len() { continue; }
                    let idx = frame_start + ch;
                    let sample = self.output_buffer[ch] * self.current_gain;
                    // Soft clipping
                    buffer[idx] = if sample.abs() > 1.0 {
                        sample.signum() * (1.0 - (-sample.abs()).exp())
                    } else {
                        sample
                    };
                }
            }

            // Always increment sample counter
            self.sample_counter += 1;
            if self.sample_counter >= self.update_samples {
                self.sample_counter = 0;
            }
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