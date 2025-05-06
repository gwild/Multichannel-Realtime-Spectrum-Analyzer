use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use portaudio as pa;
use log::{info, error};
use crate::plot::SpectrumApp;
use crate::DEFAULT_NUM_PARTIALS;
use crossbeam_queue::ArrayQueue;

// Constants for audio performance
const OUTPUT_BUFFER_SIZE: usize = 2048;  // Increased for better stability
const UPDATE_RING_SIZE: usize = 4;
const WAVETABLE_SIZE: usize = 4096;  // Size of wavetable for sine synthesis
pub const DEFAULT_UPDATE_RATE: f32 = 1.0;   // Default update rate in seconds

/// Configuration for resynthesis
pub struct ResynthConfig {
    pub gain: f32,
    pub smoothing: f32,
    pub freq_scale: f32,  // Frequency scaling factor (1.0 = normal, 2.0 = one octave up, 0.5 = one octave down)
    pub update_rate: f32, // How often to update synthesis (in seconds)
}

impl Default for ResynthConfig {
    fn default() -> Self {
        Self {
            gain: 0.25,
            smoothing: 0.0,
            freq_scale: 1.0,  // Default to no scaling
            update_rate: DEFAULT_UPDATE_RATE,
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
    
    /// Process a complete buffer of audio
    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        // Apply any pending parameter updates first
        self.apply_updates();
        
        // Calculate frames to process
        let frames = buffer.len() / channels;
        
        // Ensure output buffer is large enough
        if self.output_buffer.len() < channels {
            self.output_buffer.resize(channels, 0.0);
        }
        
        // Calculate frequency crossfade rate based on smoothing parameter
        // - When smoothing = 0, crossfade_rate = 1.0 (immediate change)
        // - When smoothing = 0.9999, crossfade_rate is very small (slow change)
        let crossfade_rate = if self.smoothing < 0.001 {
            1.0  // No smoothing = immediate changes
        } else {
            // Use direct smoothing value which matches the original behavior better
            self.smoothing  // Original code used this value directly
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
                        // Original crossfade formula that matches Python implementation
                        // Higher smoothing = slower transition
                        partial.freq = partial.freq * crossfade_rate + partial.target_freq * (1.0 - crossfade_rate);
                        
                        // Recalculate phase_delta based on current frequency
                        partial.phase_delta = partial.freq * WAVETABLE_SIZE as f32 / self.sample_rate;
                    } else {
                        // Close enough to target - snap to exact value
                        partial.freq = partial.target_freq;
                    }
                    
                    // Only generate audio if we have a frequency and amplitude
                    if partial.freq > 0.0 && partial.target_amp > 0.0 {
                        // Fast wavetable lookup with linear interpolation
                        let idx_f = partial.phase % WAVETABLE_SIZE as f32;
                        let idx1 = idx_f as usize;
                        let idx2 = (idx1 + 1) % WAVETABLE_SIZE;
                        let frac = idx_f - idx1 as f32;
                        
                        // Linear interpolation between adjacent wavetable samples
                        let sample = self.wavetable[idx1] * (1.0 - frac) + self.wavetable[idx2] * frac;
                        
                        // Apply amplitude and add to output
                        let output_sample = sample * partial.target_amp;
                        
                        // Distribute to stereo channels (simple panning)
                        let output_ch = ch_idx % channels;
                        self.output_buffer[output_ch] += output_sample;
                        
                        // Update phase accumulator for next sample
                        partial.phase = (partial.phase + partial.phase_delta) % WAVETABLE_SIZE as f32;
                    }
                }
            }
            
            // Apply gain and write to output buffer
            for ch in 0..channels {
                buffer[frame * channels + ch] = (self.output_buffer[ch] * self.current_gain).clamp(-1.0, 1.0);
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
    let update_queue = Arc::new(ArrayQueue::new(UPDATE_RING_SIZE * 2)); // Extra capacity to avoid blocking
    
    // Create and initialize synth directly; no mutex needed for audio thread
    let mut synth = WavetableSynth::new(
        num_channels, 
        num_partials,
        sample_rate as f32,
        Arc::clone(&update_queue)
    );
    
    // Clone for update thread
    let update_queue_clone = Arc::clone(&update_queue);
    let shutdown_flag_clone = Arc::clone(&shutdown_flag);
    
    // Update thread to feed the synth with new analysis data
    let _update_thread = thread::spawn(move || {
        let mut last_update = Instant::now();
        
        while !shutdown_flag_clone.load(Ordering::Relaxed) {
            // Get current config
            if let Ok(config_guard) = config.lock() {
                let update_rate = config_guard.update_rate;
                let gain = config_guard.gain;
                let freq_scale = config_guard.freq_scale;
                let smoothing = config_guard.smoothing;  // Add smoothing parameter for crossfade control
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
                            smoothing,  // Add smoothing parameter for crossfade control
                        };
                        
                        // Submit update to the queue
                        let _ = update_queue_clone.push(update);
                        last_update = Instant::now();
                    }
                }
            }
            
            thread::sleep(Duration::from_millis(10));
        }
    });
    
    // Set up PortAudio
    let pa = match pa::PortAudio::new() {
        Ok(pa) => pa,
        Err(e) => {
            error!("Failed to initialize PortAudio: {}", e);
            return;
        }
    };
    
    let output_params = pa::StreamParameters::<f32>::new(
        device_index,
        2, // Stereo output
        true,
        0.1 // Higher latency for stability
    );
    
    let settings = pa::OutputStreamSettings::new(
        output_params,
        sample_rate,
        OUTPUT_BUFFER_SIZE as u32
    );
    
    // Create audio callback - capturing synth by value
    let callback = move |args: pa::OutputStreamCallbackArgs<f32>| {
        // Process directly without any locks
        synth.process_buffer(args.buffer, 2); // 2 = stereo
        pa::Continue
    };
    
    // Open stream
    let mut stream = match pa.open_non_blocking_stream(settings, callback) {
        Ok(stream) => stream,
        Err(e) => {
            error!("Failed to open output stream: {}", e);
            return;
        }
    };
    
    // Start stream
    if let Err(e) = stream.start() {
        error!("Failed to start output stream: {}", e);
        return;
    }
    
    // Wait for shutdown
    while !shutdown_flag.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
    }
    
    // Cleanup
    if let Err(e) = stream.stop() {
        error!("Error stopping stream: {}", e);
    }
    
    info!("Resynthesis thread shutting down");
} 