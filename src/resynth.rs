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
use crate::SharedMemory;
use crate::get_results::start_update_thread;
use crate::fft_analysis::CurrentPartials;
use crate::make_waves::{build_wavetable, format_partials_debug};

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
pub struct SynthUpdate {
    pub partials: Vec<Vec<(f32, f32)>>,
    pub gain: f32,
    pub freq_scale: f32,
    pub smoothing: f32,  // Add smoothing parameter for crossfade control
    pub update_rate: f32,  // Add update rate parameter
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
    wavetables: Vec<Vec<Vec<f32>>>,  // Vector of wavetables for each channel
    transition_tables: Vec<Vec<Vec<f32>>>,  // Pre-combined transition wavetables
    sample_counter: usize,
    current_table: usize,
    sample_rate: f32,
    update_rate: f32,
    current_gain: f32,
    update_queue: Arc<ArrayQueue<SynthUpdate>>,
}

impl WavetableSynth {
    fn new(num_channels: usize, sample_rate: f32, update_queue: Arc<ArrayQueue<SynthUpdate>>) -> Self {
        let update_rate = DEFAULT_UPDATE_RATE;
        Self {
            wavetables: vec![vec![vec![0.0; (sample_rate * update_rate / 3.0) as usize]; 1]; num_channels],
            transition_tables: vec![vec![vec![0.0; (sample_rate * update_rate / 3.0) as usize]; 1]; num_channels],
            sample_counter: 0,
            current_table: 0,
            sample_rate,
            update_rate,
            current_gain: 0.25,
            update_queue,
        }
    }

    fn create_transition_table(old_data: &[f32], new_data: &[f32]) -> Vec<f32> {
        let len = old_data.len();
        let mut transition = vec![0.0; len];
        
        for i in 0..len {
            let fade = i as f32 / len as f32;
            transition[i] = old_data[i] * (1.0 - fade) + new_data[i] * fade;
        }
        
        transition
    }

    fn process_update(&mut self, update: SynthUpdate) {
        self.current_gain = update.gain;
        let rate_changed = (update.update_rate - self.update_rate).abs() > 1e-6;
        
        // Calculate segment length (1/3 of total period)
        let new_segment_len = ((if rate_changed { update.update_rate } else { self.update_rate } 
            * self.sample_rate) / 3.0) as usize;
            
        // Build new wavetable segments
        let mut new_segments = update.partials.iter()
            .map(|partials| build_wavetable(partials, self.sample_rate, update.update_rate))
            .collect::<Vec<Vec<f32>>>();
            
        // Split into thirds and create transition tables
        for channel in 0..new_segments.len() {
            let mut channel_tables = Vec::new();
            
            // Create pure segment (middle third)
            let pure_segment = new_segments[channel][0..new_segment_len].to_vec();
            
            // Create transition from current to new (if we have current data)
            if !self.wavetables[channel].is_empty() {
                let current = &self.wavetables[channel][self.current_table];
                channel_tables.push(Self::create_transition_table(
                    &current[0..new_segment_len],
                    &new_segments[channel][0..new_segment_len]
                ));
            }
            
            // Add pure segment
            channel_tables.push(pure_segment);
            
            // Create transition to next update (will be replaced when next update arrives)
            channel_tables.push(new_segments[channel][0..new_segment_len].to_vec());
            
            // Update the wavetables for this channel
            self.wavetables[channel] = channel_tables;
        }
        
        if rate_changed {
            self.update_rate = update.update_rate;
        }
    }

    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        // Process any pending updates
        while let Some(update) = self.update_queue.pop() {
            self.process_update(update);
        }

        let frames = buffer.len() / channels;
        let segment_len = (self.sample_rate * self.update_rate / 3.0) as usize;

        for frame in 0..frames {
            for ch in 0..channels {
                if ch >= self.wavetables.len() || self.wavetables[ch].is_empty() { continue; }
                
                let table = &self.wavetables[ch][self.current_table];
                buffer[frame * channels + ch] = table[self.sample_counter] * self.current_gain;
            }

            self.sample_counter += 1;
            if self.sample_counter >= segment_len {
                self.sample_counter = 0;
                self.current_table = (self.current_table + 1) % self.wavetables[0].len();
            }
        }
    }
}

/// Starts a thread that performs real-time resynthesis of the analyzed spectrum.
pub fn start_resynth_thread(
    config: Arc<Mutex<ResynthConfig>>,
    device_index: pa::DeviceIndex,
    sample_rate: f64,
    shutdown_flag: Arc<AtomicBool>,
    current_partials: Arc<Mutex<CurrentPartials>>,
    num_channels: usize,
    num_partials: usize,
) {
    debug!("Module path for resynth: {}", module_path!());
    debug!("Using fixed num_channels: {}, num_partials: {}", num_channels, num_partials);

    // Create synthesis engine and shared queue between threads
    let update_queue = Arc::new(ArrayQueue::new(UPDATE_RING_SIZE * 2));

    // Start update thread to feed the synth with new analysis data
    start_update_thread(
        Arc::clone(&config),
        Arc::clone(&shutdown_flag),
        Arc::clone(&update_queue),
        Arc::clone(&current_partials),
    );

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
