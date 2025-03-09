use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use portaudio as pa;
use log::{info, error};
use crate::plot::SpectrumApp;
use crate::fft_analysis::NUM_PARTIALS;
use rayon::prelude::*;  // Add at top with other imports

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

// Add these constants at the top with the other constants
const ZERO_CROSSING_THRESHOLD: f32 = 0.01; // How close to zero is considered a crossing
const CROSSFADE_FRAMES: usize = 32; // Number of frames to crossfade between significant changes
const PHASE_RESET_THRESHOLD: f32 = 5.0; // Hz difference that triggers phase reset consideration
const DEFAULT_UPDATE_RATE: f32 = 1.0; // Default update rate in seconds

// Constants for timing
const SAMPLE_RATE: f32 = 48000.0; // Example sample rate
const UPDATE_RATE: f32 = 4.0; // Update rate in seconds
const CROSSFADE_DURATION: f32 = 1.0; // Crossfade duration in seconds
const BUFFER_SIZE: usize = 256;       // Audio buffer size

// At module level, keep track of all state
static mut CURRENT_SYNTHESIS: [[f32; BUFFER_SIZE]; 2] = [[0.0; BUFFER_SIZE]; 2];  // Match buffer size
static mut NEXT_SYNTHESIS: [[f32; BUFFER_SIZE]; 2] = [[0.0; BUFFER_SIZE]; 2];     // Match buffer size
static mut PHASES: [[f32; 12]; 2] = [[0.0; 12]; 2];
static mut IS_CROSSFADING: bool = false;
static mut CROSSFADE_POS: usize = 0;
static mut LAST_UPDATE: Option<std::time::Instant> = None;

// At module level, add smooth state tracking
static mut SMOOTH_FREQS: [[f32; 12]; 2] = [[0.0; 12]; 2];
static mut SMOOTH_AMPS: [[f32; 12]; 2] = [[0.0; 12]; 2];

pub fn start_resynth_thread(
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    config: Arc<Mutex<ResynthConfig>>,
    device_index: pa::DeviceIndex,
    sample_rate: f64,               // Input is f64 from PortAudio
    shutdown_flag: Arc<AtomicBool>,
) {
    // Convert sample rate to f32 once for our internal calculations
    let sample_rate_f32 = sample_rate as f32;
    
    thread::spawn(move || {
        let pa = match pa::PortAudio::new() {
            Ok(pa) => pa,
            Err(e) => {
                error!("Failed to initialize PortAudio for resynthesis: {}", e);
                return;
            }
        };

        let settings = pa::OutputStreamSettings::new(
            pa::stream::Parameters::new(device_index, 2, true, 0.1),
            sample_rate,  // Use original f64 for PortAudio
            BUFFER_SIZE as u32,
        );

        let mut stream = match pa.open_non_blocking_stream(settings, move |args: pa::OutputStreamCallbackArgs<f32>| {
            let buffer = args.buffer;
            let frames = buffer.len() / 2;
            
            let config_lock = config.lock().unwrap();
            let gain = config_lock.gain;
            let freq_scale = config_lock.freq_scale;
            let update_rate = config_lock.update_rate;
            let smoothing = config_lock.smoothing;
            drop(config_lock);

            let crossfade_time = (sample_rate_f32 * CROSSFADE_DURATION) as usize;  // Use f32 version

            unsafe {
                // Check for update
                let now = std::time::Instant::now();
                let should_update = LAST_UPDATE.map_or(true, |last| {
                    now.duration_since(last).as_secs_f32() >= update_rate
                });

                if should_update {
                    // ONE read of partials
                    let partials = spectrum_app.lock().unwrap().clone_absolute_data();

                    // ONE synthesis
                    for channel in 0..2 {
                        for frame in 0..frames {
                            let mut sample = 0.0f32;
                            let mut active_partials: f32 = 0.0;

                            for (i, &(freq, amp)) in partials[channel].iter().enumerate() {
                                // Smooth frequency and amplitude changes
                                SMOOTH_FREQS[channel][i] = SMOOTH_FREQS[channel][i] * smoothing + 
                                                         freq * (1.0 - smoothing);
                                SMOOTH_AMPS[channel][i] = SMOOTH_AMPS[channel][i] * smoothing + 
                                                       amp * (1.0 - smoothing);

                                let smooth_freq = SMOOTH_FREQS[channel][i];
                                let smooth_amp = SMOOTH_AMPS[channel][i];

                                if smooth_freq > 0.0 && smooth_amp > 0.0 {
                                    let phase = &mut PHASES[channel][i];
                                    sample += smooth_amp * phase.sin();
                                    *phase = (*phase + 2.0 * PI * smooth_freq * freq_scale / sample_rate_f32) % (2.0 * PI);
                                    active_partials += 1.0;
                                }
                            }

                            // Normalize and apply limiting
                            if active_partials > 0.0 {
                                sample /= active_partials.sqrt();
                                if sample.abs() > LIMIT_THRESHOLD {
                                    let excess = sample.abs() - LIMIT_THRESHOLD;
                                    sample *= 1.0 - (excess * LIMIT_STRENGTH);
                                }
                            }

                            NEXT_SYNTHESIS[channel][frame] = sample;
                        }
                    }

                    IS_CROSSFADING = true;
                    CROSSFADE_POS = 0;
                    LAST_UPDATE = Some(now);
                }

                // Output with crossfade
                for frame in 0..frames {
                    let frame_offset = frame * 2;

                    if IS_CROSSFADING {
                        let fade_out = (crossfade_time - CROSSFADE_POS) as f32 / crossfade_time as f32;
                        let fade_in = CROSSFADE_POS as f32 / crossfade_time as f32;

                        buffer[frame_offset] = (CURRENT_SYNTHESIS[0][frame] * fade_out + 
                                              NEXT_SYNTHESIS[0][frame] * fade_in) * gain;
                        buffer[frame_offset + 1] = (CURRENT_SYNTHESIS[1][frame] * fade_out + 
                                                  NEXT_SYNTHESIS[1][frame] * fade_in) * gain;

                        CROSSFADE_POS += 1;
                        if CROSSFADE_POS >= crossfade_time {
                            IS_CROSSFADING = false;
                            CURRENT_SYNTHESIS = NEXT_SYNTHESIS;
                        }
                    } else {
                        buffer[frame_offset] = CURRENT_SYNTHESIS[0][frame] * gain;
                        buffer[frame_offset + 1] = CURRENT_SYNTHESIS[1][frame] * gain;
                    }
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
    crossfade_counter: usize,
    is_active: bool,
    envelope: f32,
    old_sample: f32, // Previous sample for zero-crossing detection
    needs_crossfade: bool, // Flag to indicate when crossfading is needed
    crossfade_buffer: Vec<f32>, // Buffer to store old samples for crossfading
    crossfade_length: usize,    // Total length of the crossfade
    last_freq_change: f32, // Track when the last significant frequency change occurred
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
            old_sample: 0.0,
            needs_crossfade: false,
            crossfade_buffer: Vec::with_capacity(1024), // Larger capacity for longer crossfades
            crossfade_length: 0,
            last_freq_change: 0.0,
        }
    }
    
    // Determine if phase needs reset due to large frequency change
    fn needs_phase_reset(&self) -> bool {
        // Only consider phase reset if the frequency change is significant
        (self.freq.target - self.freq.prev_target).abs() > PHASE_RESET_THRESHOLD
    }

    // Check if we're at or near a zero crossing
    fn is_at_zero_crossing(&self, current_sample: f32) -> bool {
        // Zero crossing occurs when samples change sign or are very close to zero
        (self.old_sample * current_sample <= 0.0) || 
        current_sample.abs() < ZERO_CROSSING_THRESHOLD
    }

    // Prepare for crossfade when significant changes occur
    fn prepare_crossfade(&mut self, frames: usize) {
        self.needs_crossfade = true;
        self.crossfade_counter = frames;
        self.crossfade_length = frames;
        // Ensure buffer is the right size
        if self.crossfade_buffer.len() < frames {
            self.crossfade_buffer.resize(frames, 0.0);
        }
    }

    // Store a sample in the crossfade buffer
    fn store_sample(&mut self, sample: f32, position: usize) {
        if position < self.crossfade_buffer.len() {
            self.crossfade_buffer[position] = sample;
        }
    }

    // Get a crossfaded sample
    fn get_crossfaded_sample(&self, new_sample: f32, position: usize) -> f32 {
        if !self.needs_crossfade || position >= self.crossfade_counter {
            return new_sample;
        }
        
        let old_sample = if position < self.crossfade_buffer.len() {
            self.crossfade_buffer[position]
        } else {
            0.0
        };
        
        // Apply smooth crossfade curve (cosine interpolation)
        let ratio = position as f32 / self.crossfade_length as f32;
        // Use a smoother S-curve for crossfading
        let crossfade_gain = if ratio < 0.5 {
            2.0 * ratio * ratio
        } else {
            1.0 - 2.0 * (1.0 - ratio) * (1.0 - ratio)
        };
        old_sample * (1.0 - crossfade_gain) + new_sample * crossfade_gain
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
    pub freq_scale: f32,  // Frequency scaling factor (1.0 = normal, 2.0 = one octave up, 0.5 = one octave down)
    pub update_rate: f32, // How often to update synthesis (in seconds)
}

impl Default for ResynthConfig {
    fn default() -> Self {
        Self {
            gain: 0.01,
            smoothing: 0.99,
            freq_scale: 1.0,  // Default to no scaling
            update_rate: DEFAULT_UPDATE_RATE,
        }
    }
}

// Add this near your imports
fn fetch_new_partials(spectrum_app: &Arc<Mutex<SpectrumApp>>) -> Vec<Vec<(f32, f32)>> {
    // Get the actual analyzed partials from SpectrumApp
    spectrum_app.lock().unwrap().clone_absolute_data()
} 