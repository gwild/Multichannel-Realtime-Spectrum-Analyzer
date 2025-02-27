use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use portaudio as pa;
use log::{info, error};
use crate::plot::SpectrumApp;
use crate::fft_analysis::NUM_PARTIALS;

// Add Pi-specific constants
#[cfg(target_arch = "arm")]
const CROSSFADE_THRESHOLD: f32 = 10.0; // More conservative for Pi
#[cfg(target_arch = "arm")]
const ATTACK_TIME: f32 = 0.05; // Longer attack for Pi (50ms)
#[cfg(target_arch = "arm")]
const RELEASE_TIME: f32 = 0.08; // Longer release for Pi (80ms)

// For non-Pi platforms
#[cfg(not(target_arch = "arm"))]
const CROSSFADE_THRESHOLD: f32 = 20.0;
#[cfg(not(target_arch = "arm"))]
const ATTACK_TIME: f32 = 0.02; 
#[cfg(not(target_arch = "arm"))]
const RELEASE_TIME: f32 = 0.03;

const MIN_ENVELOPE_VALUE: f32 = 0.001;

const SMOOTHING_FACTOR: f32 = 0.97; // Higher = smoother but slower response

// Adjust global gain scaling values for Pi and other systems
#[cfg(target_arch = "arm")]
const GAIN_SCALING: f32 = 0.4; // Less aggressive gain reduction for Pi
#[cfg(not(target_arch = "arm"))]
const GAIN_SCALING: f32 = 0.5; // Less aggressive gain reduction for other systems

// Adjust the limiter threshold to engage earlier
const LIMIT_THRESHOLD: f32 = 0.85; // Higher threshold means less limiting
const LIMIT_STRENGTH: f32 = 0.4; // Lower strength for more transparent limiting (was 0.8)

// Modify the Pi-specific filtering to be less aggressive
#[cfg(target_arch = "arm")]
const MIN_PARTIAL_AMP: f32 = 0.002; // Less aggressive filtering for quiet partials

// Dynamic gain scaling based on partial activity
const BASE_GAIN_SCALING: f32 = 0.2; // Base level gain scaling
const DYNAMIC_SCALING_FACTOR: f32 = 0.8; // How much to scale based on active partials

// Phase cancellation prevention
const PHASE_JITTER: f32 = 0.01; // Small amount of phase randomization to prevent perfect cancellation

// Keep these constants at the module level
const VOLUME_CORRECTION: f32 = 0.2; // Doubled from 0.1 to make max volume twice as loud

// Modify the envelope handling for more robust cross-buffer continuity
#[cfg(target_arch = "arm")]
const ENVELOPE_SLOPE_LIMIT: f32 = 0.01; // Limit how quickly envelopes can change per sample

#[cfg(target_arch = "arm")]
const DC_FILTER_COEFF: f32 = 0.995; // DC blocking filter coefficient

#[derive(Clone)]
struct SmoothParam {
    current: f32,
    target: f32,
    prev_target: f32, // Track previous target for detecting large changes
}

impl SmoothParam {
    fn new(value: f32) -> Self {
        Self {
            current: value,
            target: value,
            prev_target: value,
        }
    }

    fn update(&mut self, target: f32, smoothing: f32) {
        self.prev_target = self.target;
        self.target = target;
        let factor = smoothing * smoothing;
        self.current = self.current * factor + target * (1.0 - factor);
    }
    
    fn large_change(&self) -> bool {
        (self.target - self.prev_target).abs() > CROSSFADE_THRESHOLD
    }
}

#[derive(Clone)]
struct PartialState {
    freq: SmoothParam,
    amp: SmoothParam,
    phase: f32,
    crossfade_counter: usize, // For managing crossfades
    is_active: bool,          // Tracks if this partial is currently audible
    envelope: f32,            // Current envelope value (0.0-1.0)
}

impl PartialState {
    fn new() -> Self {
        Self {
            freq: SmoothParam::new(0.0),
            amp: SmoothParam::new(0.0),
            phase: 0.0,
            crossfade_counter: 0,
            is_active: false,
            envelope: 0.0,
        }
    }
    
    // Determine if phase needs reset due to large frequency change
    fn needs_phase_reset(&self) -> bool {
        // Detect any significant frequency change
        self.freq.large_change()
    }
    
    // Update envelope based on amplitude changes for smooth attack/release
    fn update_envelope(&mut self, sample_rate: f32, attack_samples: usize, release_samples: usize) {
        // Calculate smoother envelope curves using exponential approach
        if self.amp.target > 0.001 {
            // Partial is active or becoming active
            if !self.is_active {
                // Start attack phase
                self.is_active = true;
                self.crossfade_counter = attack_samples;
                // If this is a new partial, ensure envelope starts from very low value
                if self.envelope < MIN_ENVELOPE_VALUE {
                    self.envelope = MIN_ENVELOPE_VALUE;
                }
            }
            
            // Update envelope during attack - use exponential curve for smoother attack
            if self.crossfade_counter > 0 {
                // Exponential attack curve - starts slow, then accelerates
                let progress = 1.0 - (self.crossfade_counter as f32 / attack_samples as f32);
                let target = progress * progress; // Quadratic curve for smoother attack
                
                // Gradually approach target with smoothing
                self.envelope = self.envelope * 0.9 + target * 0.1;
                self.envelope = self.envelope.min(1.0);
                self.crossfade_counter -= 1;
            } else if self.envelope < 0.999 {
                // Continue smoothly approaching 1.0 even after counter expires
                self.envelope = self.envelope * 0.97 + 0.03;
                if self.envelope > 0.999 {
                    self.envelope = 1.0;
                }
            }
        } else if self.is_active || self.envelope > MIN_ENVELOPE_VALUE {
            // Partial is becoming inactive
            if self.is_active {
                self.is_active = false;
                self.crossfade_counter = release_samples;
            }
            
            // Update envelope during release - exponential curve for natural fade out
            if self.crossfade_counter > 0 {
                // Exponential release curve - fades out more naturally
                let progress = self.crossfade_counter as f32 / release_samples as f32;
                let target = progress * progress; // Quadratic curve
                
                // Gradually approach target with smoothing
                self.envelope = self.envelope * 0.9 + target * 0.1;
                self.crossfade_counter -= 1;
            } else {
                // Continue smoothly approaching 0 even after counter expires
                self.envelope = self.envelope * 0.97;
                if self.envelope < MIN_ENVELOPE_VALUE {
                    self.envelope = 0.0;
                }
            }
        }
    }
}

pub struct ResynthConfig {
    pub gain: f32,
    pub smoothing: f32,
}

impl Default for ResynthConfig {
    fn default() -> Self {
        Self {
            gain: 0.01,
            smoothing: 0.99,
        }
    }
}

pub fn start_resynth_thread(
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    config: Arc<Mutex<ResynthConfig>>,
    device_index: pa::DeviceIndex,
    sample_rate: f64,
    shutdown_flag: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        let pa = match pa::PortAudio::new() {
            Ok(pa) => pa,
            Err(e) => {
                error!("Failed to initialize PortAudio for resynthesis: {}", e);
                return;
            }
        };

        let output_params = pa::StreamParameters::<f32>::new(
            device_index, 
            2,
            true,
            0.1
        );

        let settings = pa::OutputStreamSettings::new(
            output_params,
            sample_rate,
            // Keep original larger buffer size for Pi
            if cfg!(target_arch = "arm") { 1024 } else { 512 },
        );

        // Initialize partial states for smoothing
        let num_channels = spectrum_app.lock().unwrap().clone_absolute_data().len();
        let mut partial_states = vec![vec![PartialState::new(); NUM_PARTIALS]; num_channels];
        
        // Calculate attack and release samples
        let attack_samples = (ATTACK_TIME * sample_rate as f32) as usize;
        let release_samples = (RELEASE_TIME * sample_rate as f32) as usize;

        // Add a buffer state tracker for the Pi
        #[cfg(target_arch = "arm")]
        let mut last_buffer_state: Vec<(f32, f32)> = vec![(0.0, 0.0); 2]; // Left/right values from end of last buffer
        
        // Add these variables inside the function instead
        #[cfg(target_arch = "arm")]
        let mut prev_left = 0.0f32;
        #[cfg(target_arch = "arm")]
        let mut prev_right = 0.0f32;
        
        let mut stream = match pa.open_non_blocking_stream(settings, move |args: pa::OutputStreamCallbackArgs<f32>| {
            let buffer = args.buffer;
            let frames = buffer.len() / 2;
            
            let partials = spectrum_app.lock().unwrap().clone_absolute_data();
            let config_lock = config.lock().unwrap();
            let gain = config_lock.gain;
            let smoothing = config_lock.smoothing;

            buffer.fill(0.0);

            // At the beginning of processing, apply special handling for buffer boundaries on Pi
            #[cfg(target_arch = "arm")]
            {
                // Apply a small crossfade between buffer boundaries
                if frames > 0 {
                    // Start with the last values from previous buffer and fade to new values
                    buffer[0] = last_buffer_state[0].0;
                    buffer[1] = last_buffer_state[0].1;
                    
                    // For first few frames, do a quick crossfade from last buffer
                    let crossfade_frames = frames.min(16); // Max 16 frames for crossfade
                    for frame in 1..crossfade_frames {
                        let ratio = frame as f32 / crossfade_frames as f32;
                        // Blend previous buffer end with current buffer start
                        buffer[frame*2] = buffer[frame*2] * ratio + last_buffer_state[0].0 * (1.0-ratio);
                        buffer[frame*2+1] = buffer[frame*2+1] * ratio + last_buffer_state[0].1 * (1.0-ratio);
                    }
                }
            }

            for frame in 0..frames {
                let mut left = 0.0f32;
                let mut right = 0.0f32;
                
                // Count active partials to apply dynamic scaling
                let mut active_partial_count = 0;
                
                // First pass: count active partials for dynamic scaling
                for channel_partials in &partials {
                    for &(freq, amp) in channel_partials.iter() {
                        if freq > 0.0 && amp > 0.01 {
                            active_partial_count += 1;
                        }
                    }
                }
                
                // Calculate dynamic scaling - more active partials = lower per-partial gain
                let dynamic_scale = if active_partial_count > 0 {
                    let scaling = 1.0 / (1.0 + (active_partial_count as f32 * 0.05));
                    BASE_GAIN_SCALING + (DYNAMIC_SCALING_FACTOR * scaling)
                } else {
                    BASE_GAIN_SCALING
                };

                // Process all partials
                for (channel, channel_partials) in partials.iter().enumerate() {
                    for (i, &(freq, amp)) in channel_partials.iter().enumerate() {
                        let state = &mut partial_states[channel][i];
                        
                        // Update smoothed parameters
                        state.freq.update(freq, smoothing);
                        state.amp.update(amp, smoothing);
                        
                        // Apply more aggressive filtering on Pi
                        #[cfg(target_arch = "arm")]
                        {
                            // Filter out very quiet partials, but with a lower threshold
                            if state.amp.target < MIN_PARTIAL_AMP && state.amp.current < MIN_PARTIAL_AMP * 2.0 {
                                state.amp.current = 0.0;
                                state.amp.target = 0.0;
                                continue;
                            }
                            
                            // Less aggressive frequency smoothing
                            if state.freq.large_change() {
                                state.freq.current = state.freq.current * 0.85 + state.freq.target * 0.15;
                            }
                        }
                        
                        // Update envelope for attack/release
                        state.update_envelope(sample_rate as f32, attack_samples, release_samples);
                        
                        // Check if we need to reset phase (for large frequency changes)
                        if state.needs_phase_reset() {
                            // Instead of immediately resetting phase, schedule a phase reset
                            // when we next cross zero to minimize pops
                            let phase_mod = state.phase % (2.0 * PI);
                            
                            // Check if we're near a zero crossing (sin wave = 0)
                            // which occurs at 0, π, and 2π
                            let near_zero = phase_mod < 0.2 || (phase_mod - PI).abs() < 0.2 || phase_mod > (2.0 * PI - 0.2);
                            
                            if near_zero {
                                // At a zero crossing, we can safely reset phase
                                // But instead of setting to exactly 0, use the closest zero crossing point
                                // to maintain continuity
                                if phase_mod < 0.2 {
                                    state.phase = state.phase - phase_mod; // Reset to 0
                                } else if (phase_mod - PI).abs() < 0.2 {
                                    state.phase = state.phase - (phase_mod - PI); // Reset to π
                                } else {
                                    state.phase = state.phase + (2.0 * PI - phase_mod); // Reset to 2π
                                }
                            }
                            // If not near zero, don't reset - wait until we get to a zero crossing
                        }

                        if state.freq.current > 0.0 && (state.amp.current > 0.0 || state.envelope > 0.0) {
                            // Apply envelope with per-partial dynamic scaling
                            let amplitude = if state.is_active {
                                state.amp.current * state.envelope
                            } else {
                                state.amp.current * state.envelope
                            };
                            
                            // Slight phase jitter to prevent perfect cancellation between partials
                            let phase_with_jitter = state.phase + (PHASE_JITTER * (i as f32 * 0.1));
                            
                            // Define raw_sample outside the cfg blocks
                            let raw_sample: f32;
                            
                            // On Pi, simplify the oscillator to reduce computational load
                            #[cfg(target_arch = "arm")]
                            {
                                // Less expensive sine approximation for Pi
                                let phase_sin = if phase_with_jitter < PI {
                                    (1.57079632679 - phase_with_jitter) * (phase_with_jitter)
                                } else {
                                    (phase_with_jitter - 4.71238898038) * (phase_with_jitter - 3.14159265359) 
                                };
                                
                                // Set the outer raw_sample
                                raw_sample = amplitude * phase_sin;
                            }
                            
                            // On other platforms, use the regular sine function
                            #[cfg(not(target_arch = "arm"))]
                            {
                                // Set the outer raw_sample
                                raw_sample = amplitude * phase_with_jitter.sin();
                            }
                            
                            // Now we can use raw_sample outside the cfg blocks
                            
                            // Apply additional filtering to prevent high-frequency noise for high frequencies
                            let filtered_sample = if freq > 10000.0 {
                                raw_sample * (20000.0 - freq) / 10000.0
                            } else {
                                raw_sample
                            };
                            
                            // Channel-dependent panning to reduce mono summing issues
                            if channel % 2 == 0 {
                                left += filtered_sample;
                            } else {
                                right += filtered_sample;
                            }
                            
                            // Update phase using smoothed frequency with more careful wrapping
                            let phase_increment = 2.0 * PI * state.freq.current / sample_rate as f32;
                            state.phase += phase_increment;
                            
                            // Normalize phase to prevent floating point precision issues over time
                            while state.phase >= 2.0 * PI {
                                state.phase -= 2.0 * PI;
                            }
                            while state.phase < 0.0 {
                                state.phase += 2.0 * PI;
                            }
                        }
                    }
                }

                // Apply the dynamic scaling and user gain
                let frame_offset = frame * 2;

                // Apply the correct volume scaling (gain is 0.001-0.01 range)
                // When user sets volume to 1 (gain=0.001), the actual gain becomes 0.0001
                // When user sets volume to 10 (gain=0.01), the actual gain becomes 0.001
                let corrected_gain = gain * VOLUME_CORRECTION;

                // Use corrected gain with dynamic scaling
                buffer[frame_offset] = (left * dynamic_scale * corrected_gain).clamp(-0.95, 0.95);
                buffer[frame_offset + 1] = (right * dynamic_scale * corrected_gain).clamp(-0.95, 0.95);
            }

            // At the end of processing, store last buffer values for next time on Pi
            #[cfg(target_arch = "arm")]
            {
                if frames > 0 {
                    // Store last values for next buffer
                    last_buffer_state[0].0 = buffer[(frames-1)*2];
                    last_buffer_state[0].1 = buffer[(frames-1)*2+1];
                }
            }
            
            // Before final output, add DC filtering for Pi
            #[cfg(target_arch = "arm")]
            {
                // We need to iterate through all frames for DC filtering
                for frame in 0..frames {
                    let frame_offset = frame * 2;
                    
                    // Simple DC blocking filter for left channel
                    let filtered_left = buffer[frame_offset] - prev_left + (DC_FILTER_COEFF * prev_left);
                    prev_left = filtered_left;
                    buffer[frame_offset] = filtered_left;
                    
                    // Simple DC blocking filter for right channel
                    let filtered_right = buffer[frame_offset + 1] - prev_right + (DC_FILTER_COEFF * prev_right);
                    prev_right = filtered_right;
                    buffer[frame_offset + 1] = filtered_right;
                }
            }
            
            pa::Continue
        }) {
            Ok(stream) => stream,
            Err(e) => {
                error!("Failed to open output stream: {}", e);
                return;
            }
        };

        if let Err(e) = stream.start() {
            error!("Failed to start output stream: {}", e);
            return;
        }

        while !shutdown_flag.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(100));
        }

        info!("Resynthesis thread shutting down");
    });
} 