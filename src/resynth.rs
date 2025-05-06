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
    current_wavetables: Vec<Vec<f32>>,
    next_wavetables: Vec<Vec<f32>>,
    // Dual phase accumulators per channel
    curr_phases: Vec<f32>,
    next_phases: Vec<f32>,
    root_freqs: Vec<f32>,
    next_root_freqs: Vec<f32>,
    sample_counter: usize,
    crossfade_start_sample: usize,
    crossfade_end_sample: usize,
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
        let mut synth = Self {
            current_wavetables: vec![],
            next_wavetables: vec![],
            curr_phases: vec![],
            next_phases: vec![],
            root_freqs: vec![],
            next_root_freqs: vec![],
            sample_counter: 0,
            crossfade_start_sample: 0,
            crossfade_end_sample: 0,
            update_samples: 0,
            in_crossfade: false,
            sample_rate,
            output_buffer: vec![],
            update_queue,
            current_gain: 0.25,
            update_rate: DEFAULT_UPDATE_RATE,
        };
        synth.resize_state_vectors(num_channels);
        synth.update_samples = (synth.update_rate * sample_rate) as usize;
        synth.crossfade_start_sample = (synth.update_samples as f32 * 2.0 / 3.0) as usize;
        synth.crossfade_end_sample = synth.update_samples;
        synth
    }

    // Helper: build a wavetable from a set of partials (freq, amp), length = WAVETABLE_SIZE
    fn build_wavetable(partials: &[(f32, f32)], sample_rate: f32) -> Vec<f32> {
        let mut table = vec![0.0; WAVETABLE_SIZE];
        // Find the root frequency (first nonzero partial)
        let f0 = partials.iter().find(|&&(f, a)| f > 0.0 && a > 0.0).map(|&(f, _)| f).unwrap_or(1.0);
        for i in 0..WAVETABLE_SIZE {
            let t = i as f32 / WAVETABLE_SIZE as f32; // 0..1
            for &(freq, amp) in partials {
                if freq > 0.0 && amp > 0.0 {
                    let norm_amp = amp / 100.0; // scale dB to linear
                    // Compose as sum of sines, as if real-time synthesis
                    table[i] += norm_amp * (2.0 * PI * freq * t / f0).sin();
                }
            }
        }
        // Normalize to prevent clipping
        let max = table.iter().cloned().fold(0.0_f32, |a, b| a.abs().max(b.abs()));
        if max > 1.0 {
            for v in &mut table {
                *v /= max;
            }
        }
        table
    }

    fn apply_updates(&mut self) {
        let mut new_update = None;
        while let Some(update) = self.update_queue.pop() {
            new_update = Some(update);
        }
        if let Some(update) = new_update {
            self.current_gain = update.gain;
            let update_channels = update.partials.len();
            if update_channels != self.current_wavetables.len() {
                self.resize_state_vectors(update_channels);
            }
            self.update_samples = (self.update_rate * self.sample_rate) as usize;
            self.crossfade_start_sample = (self.update_samples as f32 * 2.0 / 3.0) as usize;
            self.crossfade_end_sample = self.update_samples;
            self.sample_counter = 0;
            self.in_crossfade = false;
            for (ch, ch_partials) in update.partials.iter().enumerate() {
                self.next_wavetables[ch] = Self::build_wavetable(ch_partials, self.sample_rate);
                self.next_root_freqs[ch] = ch_partials.iter().find(|&&(f, a)| f > 0.0 && a > 0.0).map(|&(f, _)| f).unwrap_or(1.0);
                self.next_phases[ch] = 0.0;
            }
        }
    }

    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        if buffer.is_empty() || channels == 0 {
            return;
        }
        // Always guard output_buffer size
        if self.output_buffer.len() < channels {
            self.output_buffer.resize(channels, 0.0);
        }
        self.apply_updates();
        let frames = buffer.len() / channels;
        for frame in 0..frames {
            let mut crossfade = 0.0;
            if self.sample_counter >= self.crossfade_start_sample && self.sample_counter < self.crossfade_end_sample {
                crossfade = (self.sample_counter - self.crossfade_start_sample) as f32 /
                    (self.crossfade_end_sample - self.crossfade_start_sample) as f32;
                self.in_crossfade = true;
            } else if self.sample_counter >= self.crossfade_end_sample {
                self.current_wavetables = self.next_wavetables.clone();
                self.root_freqs = self.next_root_freqs.clone();
                self.curr_phases = self.next_phases.clone();
                self.sample_counter = 0;
                self.in_crossfade = false;
                crossfade = 0.0;
            }
            for ch in 0..channels {
                if ch >= self.output_buffer.len() || ch >= self.current_wavetables.len() || ch >= self.next_wavetables.len() || ch >= self.curr_phases.len() || ch >= self.next_phases.len() || ch >= self.root_freqs.len() || ch >= self.next_root_freqs.len() {
                    continue;
                }
                self.output_buffer[ch] = 0.0;
            }
            for ch in 0..channels {
                if ch >= self.output_buffer.len() || ch >= self.current_wavetables.len() || ch >= self.next_wavetables.len() || ch >= self.curr_phases.len() || ch >= self.next_phases.len() || ch >= self.root_freqs.len() || ch >= self.next_root_freqs.len() {
                    continue;
                }
                let curr_phase = self.curr_phases[ch];
                let next_phase = self.next_phases[ch];
                let idx1 = curr_phase % WAVETABLE_SIZE as f32;
                let idx1a = idx1 as usize;
                let idx1b = (idx1a + 1) % WAVETABLE_SIZE;
                let frac1 = idx1 - idx1a as f32;
                let curr_sample = self.current_wavetables[ch][idx1a] * (1.0 - frac1) + self.current_wavetables[ch][idx1b] * frac1;
                let idx2 = next_phase % WAVETABLE_SIZE as f32;
                let idx2a = idx2 as usize;
                let idx2b = (idx2a + 1) % WAVETABLE_SIZE;
                let frac2 = idx2 - idx2a as f32;
                let next_sample = self.next_wavetables[ch][idx2a] * (1.0 - frac2) + self.next_wavetables[ch][idx2b] * frac2;
                let sample = if self.in_crossfade {
                    curr_sample * (1.0 - crossfade) + next_sample * crossfade
                } else {
                    curr_sample
                };
                self.output_buffer[ch] = sample;
                let curr_phase_inc = self.root_freqs[ch] * WAVETABLE_SIZE as f32 / self.sample_rate;
                let next_phase_inc = self.next_root_freqs[ch] * WAVETABLE_SIZE as f32 / self.sample_rate;
                self.curr_phases[ch] = (curr_phase + curr_phase_inc) % WAVETABLE_SIZE as f32;
                self.next_phases[ch] = (next_phase + next_phase_inc) % WAVETABLE_SIZE as f32;
            }
            let frame_start = frame * channels;
            if frame_start + channels <= buffer.len() {
                for ch in 0..channels {
                    if ch >= self.output_buffer.len() { continue; }
                    let idx = frame_start + ch;
                    buffer[idx] = (self.output_buffer[ch] * self.current_gain).clamp(-1.0, 1.0);
                }
            }
            self.sample_counter += 1;
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
            // Clone config and spectrum data with minimal lock duration
            let (partials, gain, freq_scale, smoothing, update_rate) = {
                let (partials, gain, freq_scale, smoothing, update_rate);
                if let Ok(config_guard) = config_for_thread.lock() {
                    gain = config_guard.gain;
                    freq_scale = config_guard.freq_scale;
                    smoothing = config_guard.smoothing;
                    update_rate = config_guard.update_rate;
                } else {
                    gain = 0.25;
                    freq_scale = 1.0;
                    smoothing = 0.0;
                    update_rate = DEFAULT_UPDATE_RATE;
                }
                if let Ok(spec_guard) = spectrum_app.lock() {
                    partials = spec_guard.clone_absolute_data();
                } else {
                    partials = Vec::new();
                }
                (partials, gain, freq_scale, smoothing, update_rate)
            };
            // Only do heavy computation (wavetable build, etc.) after releasing locks
            if last_update.elapsed().as_secs_f32() >= update_rate {
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