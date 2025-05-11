use std::sync::{Arc, Mutex, mpsc};
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
use crate::get_results::{start_update_thread, start_update_thread_with_sender};
use crate::fft_analysis::CurrentPartials;
use crate::make_waves::{build_wavetable, format_partials_debug};
use std::collections::VecDeque;
use hound;

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
struct WaveSegment {
    samples: Vec<Vec<f32>>,  // Per-channel samples
    length: usize,           // Length of segment
}

struct WaveSynth {
    playback_queue: Arc<Mutex<VecDeque<WaveSegment>>>,  // Shared queue of segments ready for playback
    sample_counter: usize,
    sample_rate: f32,
    update_rate: f32,
    current_gain: f32,
    max_queue_size: usize,
}

impl WaveSynth {
    fn new(sample_rate: f32) -> Self {
        let update_rate = DEFAULT_UPDATE_RATE;
        let segment_length = (sample_rate * update_rate / 3.0) as usize;
        let silent_segment = WaveSegment {
            samples: vec![vec![0.0; segment_length]; 2],
            length: segment_length,
        };
        let mut queue = VecDeque::new();
        queue.push_back(silent_segment.clone());
        queue.push_back(silent_segment.clone());
        queue.push_back(silent_segment);
        Self {
            playback_queue: Arc::new(Mutex::new(queue)),
            sample_counter: 0,
            sample_rate,
            update_rate,
            current_gain: 1.0,
            max_queue_size: 6,
        }
    }

    fn combine_partials_to_stereo(partials: &[Vec<(f32, f32)>]) -> [Vec<(f32, f32)>; 2] {
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut left_count = 0;
        let mut right_count = 0;
        for (i, ch_partials) in partials.iter().enumerate() {
            if i % 2 == 0 {
                left.extend_from_slice(ch_partials);
                left_count += 1;
            } else {
                right.extend_from_slice(ch_partials);
                right_count += 1;
            }
        }
        // Attenuate amplitudes by number of contributing channels
        if left_count > 0 {
            for partial in &mut left {
                partial.1 /= left_count as f32;
            }
        }
        if right_count > 0 {
            for partial in &mut right {
                partial.1 /= right_count as f32;
            }
        }
        [left, right]
    }

    fn prepare_segment(&mut self, partials: &[Vec<(f32, f32)>], is_transition: bool, old_samples: Option<&[Vec<f32>]>) -> WaveSegment {
        let segment_length = (self.sample_rate * self.update_rate / 3.0) as usize;
        let mut segment = WaveSegment {
            samples: vec![vec![0.0; segment_length]; 2], // Always stereo
            length: segment_length,
        };
        let stereo_partials = Self::combine_partials_to_stereo(partials);
        let gain = self.current_gain * 0.01;
        for ch in 0..2 {
            for i in 0..segment_length {
                let time = i as f32 / self.sample_rate;
                let mut sample = 0.0;
                for &(freq, amp) in stereo_partials[ch].iter().filter(|&&(f, a)| f > 0.0 && a > 0.0) {
                    let phase = 2.0 * std::f32::consts::PI * freq * time;
                    sample += amp * phase.sin();
                }
                sample *= gain;
                if is_transition {
                    let fade_in = i as f32 / segment_length as f32;
                    let fade_out = 1.0 - fade_in;
                    if let Some(old) = old_samples {
                        if ch < old.len() && i < old[ch].len() {
                            let old_sample = old[ch][i];
                            segment.samples[ch][i] = old_sample * fade_out + sample * fade_in;
                            continue;
                        }
                    }
                }
                segment.samples[ch][i] = sample;
            }
        }
        segment
    }

    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        let channels = 2;
        let frames = buffer.len() / channels;
        let mut queue = self.playback_queue.lock().unwrap();
        for frame in 0..frames {
            if let Some(current_segment) = queue.front() {
                if self.sample_counter >= current_segment.length {
                    self.sample_counter = 0;
                    if queue.len() > 1 {
                        queue.pop_front();
                    }
                }
                if let Some(current_segment) = queue.front() {
                    for ch in 0..channels {
                        if ch < current_segment.samples.len() {
                            buffer[frame * channels + ch] = current_segment.samples[ch][self.sample_counter];
                        }
                    }
                }
            }
            self.sample_counter += 1;
        }
    }
}

// Utility function to dump a WaveSegment to a stereo .wav file
fn dump_wave_segment_to_wav(segment: &WaveSegment, sample_rate: u32, path: &str) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create wav file");
    let len = segment.length;
    for i in 0..len {
        let l = (segment.samples[0][i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r = (segment.samples[1][i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(l).unwrap();
        writer.write_sample(r).unwrap();
    }
    writer.finalize().unwrap();
}

// --- Wave Generation Thread ---
pub fn start_wavegen_thread(
    update_rx: mpsc::Receiver<SynthUpdate>,
    playback_queue: Arc<Mutex<VecDeque<WaveSegment>>>,
    sample_rate: f32,
    max_queue_size: usize,
) {
    std::thread::spawn(move || {
        let mut update_rate = DEFAULT_UPDATE_RATE;
        let mut last_pure_segment: Option<WaveSegment> = None;
        while let Ok(update) = update_rx.recv() {
            let current_gain = update.gain;
            let freq_scale = update.freq_scale;
            if (update.update_rate - update_rate).abs() > f32::EPSILON {
                let mut queue = playback_queue.lock().unwrap();
                queue.clear();
                update_rate = update.update_rate;
                // Do NOT reset last_pure_segment here!
            }
            let segment_length = (sample_rate * update_rate / 3.0) as usize;
            let stereo_partials = WaveSynth::combine_partials_to_stereo(&update.partials);
            // --- Pure segment: new only, gain applied ---
            let mut pure = WaveSegment {
                samples: vec![vec![0.0; segment_length]; 2],
                length: segment_length,
            };
            for ch in 0..2 {
                for i in 0..segment_length {
                    let time = i as f32 / sample_rate;
                    let mut sample = 0.0;
                    for &(freq, amp) in stereo_partials[ch].iter().filter(|&&(f, a)| f > 0.0 && a > 0.0) {
                        let phase = 2.0 * std::f32::consts::PI * (freq * freq_scale) * time;
                        sample += amp * phase.sin();
                    }
                    pure.samples[ch][i] = sample * current_gain * 0.01;
                }
            }
            // --- Dump pure segment to wav for inspection ---
            dump_wave_segment_to_wav(&pure, sample_rate as u32, "pure_segment_dump.wav");
            // --- Transition segment: crossfade from last pure to new pure ---
            let mut transition = WaveSegment {
                samples: vec![vec![0.0; segment_length]; 2],
                length: segment_length,
            };
            for ch in 0..2 {
                for i in 0..segment_length {
                    let fade_in = i as f32 / segment_length as f32;
                    let fade_out = 1.0 - fade_in;
                    let old_sample = last_pure_segment.as_ref().and_then(|old| old.samples.get(ch).and_then(|v| v.get(i).copied())).unwrap_or(0.0);
                    let new_sample = pure.samples[ch][i];
                    transition.samples[ch][i] = old_sample * fade_out + new_sample * fade_in;
                }
            }
            // Push transition and pure to queue
            let mut queue = playback_queue.lock().unwrap();
            if queue.len() < max_queue_size {
                queue.push_back(transition.clone());
                queue.push_back(pure.clone());
            }
            // Update last_pure_segment for next transition
            last_pure_segment = Some(pure);
        }
    });
}

/// Starts a thread that performs real-time resynthesis of the analyzed spectrum.
pub fn start_resynth_thread(
    config: Arc<Mutex<ResynthConfig>>,
    device_index: pa::DeviceIndex,
    sample_rate: f64,
    shutdown_flag: Arc<AtomicBool>,
    current_partials: Arc<Mutex<CurrentPartials>>,
    _num_channels: usize,
    _num_partials: usize,
) {
    debug!("Module path for resynth: {}", module_path!());
    let (update_tx, update_rx) = mpsc::channel::<SynthUpdate>();
    let synth = WaveSynth::new(sample_rate as f32);
    let playback_queue = synth.playback_queue.clone();
    let max_queue_size = synth.max_queue_size;
    // Start wavegen thread
    start_wavegen_thread(update_rx, playback_queue, sample_rate as f32, max_queue_size);
    // Start update thread to feed the wavegen with new analysis data
    start_update_thread_with_sender(
        Arc::clone(&config),
        Arc::clone(&shutdown_flag),
        update_tx,
        Arc::clone(&current_partials),
    );
    // Main thread that handles audio stream lifecycle
    std::thread::spawn(move || {
        let mut current_stream: Option<pa::Stream<pa::NonBlocking, pa::Output<f32>>> = None;
        let mut last_restart_time = Instant::now();
        let mut consecutive_errors = 0;
        let needs_restart = Arc::clone(&config.lock().unwrap().needs_restart);
        while !shutdown_flag.load(Ordering::Relaxed) {
            let needs_restart_now = needs_restart.load(Ordering::Relaxed) || current_stream.is_none();
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
                    synth.playback_queue.clone(),
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
        if let Some(mut stream) = current_stream {
            let _ = stream.stop();
        }
        info!("Resynthesis thread shutting down");
    });
}

fn setup_audio_stream(
    device_index: pa::DeviceIndex,
    sample_rate: f64,
    playback_queue: Arc<Mutex<VecDeque<WaveSegment>>>,
) -> Result<pa::Stream<pa::NonBlocking, pa::Output<f32>>, anyhow::Error> {
    let pa = pa::PortAudio::new()?;
    let device_info = pa.device_info(device_index)?;
    let output_params = pa::StreamParameters::<f32>::new(
        device_index,
        2,
        true,
        device_info.default_low_output_latency
    );
    let settings = pa::OutputStreamSettings::new(output_params, sample_rate, OUTPUT_BUFFER_SIZE as u32);
    let synth = WaveSynth {
        playback_queue: playback_queue.clone(),
        sample_counter: 0,
        sample_rate: sample_rate as f32,
        update_rate: DEFAULT_UPDATE_RATE,
        current_gain: 1.0,
        max_queue_size: 6,
    };
    let synth = Arc::new(Mutex::new(synth));
    let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
        if let Ok(mut synth) = synth.lock() {
            synth.process_buffer(buffer, 2);
        }
        pa::Continue
    };
    let mut stream = pa.open_non_blocking_stream(settings, callback)?;
    stream.start()?;
    debug!("Audio stream started successfully");
    Ok(stream)
}
