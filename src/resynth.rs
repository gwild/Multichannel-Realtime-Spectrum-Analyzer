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

// Constants used in the code
const CROSSFADE_THRESHOLD: f32 = 20.0;  // Used in SmoothParam::large_change()
const ATTACK_TIME: f32 = 0.02;          // Used in envelope calculations
const RELEASE_TIME: f32 = 0.03;         // Used in envelope calculations
const MIN_ENVELOPE_VALUE: f32 = 0.001;  // Used in envelope calculations
const ZERO_CROSSING_THRESHOLD: f32 = 0.01;  // Used in is_at_zero_crossing()
const CROSSFADE_FRAMES: usize = 64;     // Used in prepare_crossfade()
const PHASE_RESET_THRESHOLD: f32 = 5.0;  // Used in needs_phase_reset()
const DEFAULT_UPDATE_RATE: f32 = 1.0;   // Used in ResynthConfig default

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
        if smoothing == 0.0 {
            self.current = target;  // Direct update, no smoothing calculation
        } else {
            // High smoothing = more weight on current value
            let old_weight = smoothing;           // Keep more of old value when smoothing is high
            let new_weight = 1.0 - smoothing;     // Take less of new value when smoothing is high
            self.current = self.current * old_weight + target * new_weight;
        }
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
            crossfade_buffer: vec![0.0; CROSSFADE_FRAMES],
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
        self.crossfade_counter = frames.min(CROSSFADE_FRAMES).max(1);
        // Ensure buffer is the right size
        if self.crossfade_buffer.len() < self.crossfade_counter {
            self.crossfade_buffer.resize(self.crossfade_counter, 0.0);
        }
    }

    // Store a sample in the crossfade buffer
    fn store_sample(&mut self, sample: f32, position: usize) {
        if position < self.crossfade_buffer.len() && position < CROSSFADE_FRAMES {
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
        let ratio = position as f32 / self.crossfade_counter as f32;
        let crossfade_gain = 0.5 - 0.5 * (ratio * std::f32::consts::PI).cos();
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
            smoothing: 0.0,
            freq_scale: 1.0,  // Default to no scaling
            update_rate: DEFAULT_UPDATE_RATE,
        }
    }
}

#[derive(Clone)]
struct SynthInstance {
    partial_states: Vec<Vec<PartialState>>,
    fade_level: f32,  // 0.0 to 1.0 for crossfading
}

impl SynthInstance {
    fn new(num_channels: usize) -> Self {
        Self {
            partial_states: vec![vec![PartialState::new(); NUM_PARTIALS]; num_channels],
            fade_level: 0.0,
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
    // Create two synthesis instances
    let mut current_synth = SynthInstance::new(spectrum_app.lock().unwrap().clone_absolute_data().len());
    let mut next_synth = current_synth.clone();
    let mut is_crossfading = false;
    let mut crossfade_samples_remaining = 0;
    let mut last_update = std::time::Instant::now();

    // Helper function to generate a frame of audio
    fn generate_frame(synth: &mut SynthInstance, frame: usize, sample_rate: f64) -> (f32, f32) {
        let mut left = 0.0;
        let mut right = 0.0;
        
        for (channel, states) in synth.partial_states.iter_mut().enumerate() {
            for state in states {
                if state.freq.current > 0.0 && (state.amp.current > 0.0 || state.envelope > 0.0) {
                    let amplitude = (state.amp.current / 100.0) * state.envelope;
                    let sample = amplitude * state.phase.sin();
                    
                    // Update phase for next sample
                    state.phase += 2.0 * PI * state.freq.current / sample_rate as f32;
                    if state.phase >= 2.0 * PI {
                        state.phase -= 2.0 * PI;
                    }
                    
                    if channel % 2 == 0 {
                        left += sample;
                    } else {
                        right += sample;
                    }
                }
            }
        }
        (left, right)
    }

    let callback = move |args: pa::OutputStreamCallbackArgs<f32>| {
        let buffer = args.buffer;
        let frames = buffer.len() / 2;
        
        // Get config values
        let config_lock = config.lock().unwrap();
        let gain = config_lock.gain;
        let update_rate = config_lock.update_rate;
        let crossfade_duration = (update_rate * 0.25 * sample_rate as f32) as usize;
        
        // Check if it's time to update
        let should_update = last_update.elapsed().as_secs_f32() >= update_rate;
        
        if should_update {
            last_update = std::time::Instant::now();
            
            // Debug print FFT data
            let partials = spectrum_app.lock().unwrap().clone_absolute_data();
            info!("Got FFT data: {} channels", partials.len());
            for (ch, channel_data) in partials.iter().enumerate() {
                info!("Channel {}: {} partials", ch, channel_data.len());
                for &(freq, amp) in channel_data.iter().take(3) {  // Print first 3 partials
                    info!("  Freq: {:.1}, Amp: {:.1}", freq, amp);
                }
            }
            
            // Clone current synth to preserve smoothing state
            next_synth = current_synth.clone();
            
            // Update with new FFT data
            let partials = spectrum_app.lock().unwrap().clone_absolute_data();
            for (channel, channel_partials) in partials.iter().enumerate() {
                for (i, &(freq, amp)) in channel_partials.iter().enumerate() {
                    let state = &mut next_synth.partial_states[channel][i];
                    state.freq.update(freq * config_lock.freq_scale, config_lock.smoothing);
                    state.amp.update(amp, config_lock.smoothing);
                    
                    // Update envelope with smoothing instead of direct assignment
                    let target_envelope = if amp > 0.0 { 1.0 } else { 0.0 };
                    state.envelope = if config_lock.smoothing == 0.0 {
                        target_envelope
                    } else {
                        state.envelope * config_lock.smoothing + target_envelope * (1.0 - config_lock.smoothing)
                    };
                }
            }
            
            is_crossfading = true;
            crossfade_samples_remaining = crossfade_duration;
        }

        if is_crossfading {
            // Generate audio from both instances and crossfade
            let fade_out = (crossfade_samples_remaining as f32 / crossfade_duration as f32).max(0.0);
            let fade_in = 1.0 - fade_out;

            for frame in 0..frames {
                let (old_left, old_right) = generate_frame(&mut current_synth, frame, sample_rate);
                let (new_left, new_right) = generate_frame(&mut next_synth, frame, sample_rate);

                buffer[frame * 2] = (old_left * fade_out + new_left * fade_in) * gain;
                buffer[frame * 2 + 1] = (old_right * fade_out + new_right * fade_in) * gain;
            }

            if crossfade_samples_remaining > frames {
                crossfade_samples_remaining -= frames;
            } else {
                crossfade_samples_remaining = 0;
                is_crossfading = false;
                current_synth = next_synth.clone();
            }
        } else {
            // Normal synthesis from current instance
            for frame in 0..frames {
                let (left, right) = generate_frame(&mut current_synth, frame, sample_rate);
                buffer[frame * 2] = left * gain;
                buffer[frame * 2 + 1] = right * gain;
            }
        }

        pa::Continue
    };

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

        let mut stream = match pa.open_non_blocking_stream(settings, callback) {
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