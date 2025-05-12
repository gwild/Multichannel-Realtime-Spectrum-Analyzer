use std::sync::{Arc, Mutex, mpsc, Condvar};
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
use crate::make_waves::{build_segment_buffer, format_partials_debug};
use std::collections::VecDeque;
use hound;
use std::sync::atomic::AtomicUsize;

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

#[derive(Debug)]
enum CrossfadeState { Idle, Active { old_pos: usize, fade_len: usize } }

struct WaveSynth {
    playback_queue: Arc<Mutex<VecDeque<WaveSegment>>>,  // Shared queue of segments ready for playback
    sample_counter: usize,
    sample_rate: f32,
    update_rate: f32,
    current_gain: f32,
    max_queue_size: usize,
    rate_change_flag: Arc<Mutex<bool>>,
    prev_segment: Option<WaveSegment>,
    crossfade_len: usize,
    crossfade_start_pos: Option<usize>, // Start position in old segment for crossfade
    crossfade_state: Option<CrossfadeState>,
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
            rate_change_flag: Arc::new(Mutex::new(false)),
            prev_segment: None,
            crossfade_len: 0,
            crossfade_start_pos: None,
            crossfade_state: Some(CrossfadeState::Idle),  // Initialize with Idle state
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
        let segment_length = (self.sample_rate * self.update_rate) as usize;
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

    /// Handle update rate change: reset state and queue
    fn handle_rate_change(&mut self, new_rate: f32) {
        self.update_rate = new_rate;
        self.sample_counter = 0;
        self.crossfade_state = Some(CrossfadeState::Idle);
        self.prev_segment = None;
        self.crossfade_start_pos = None;
        self.crossfade_len = 0;
        // Clear the queue and reinitialize with silent segments
        let segment_length = (self.sample_rate * self.update_rate / 3.0) as usize;
        let silent_segment = WaveSegment {
            samples: vec![vec![0.0; segment_length]; 2],
            length: segment_length,
        };
        if let Ok(mut queue) = self.playback_queue.lock() {
            queue.clear();
            queue.push_back(silent_segment.clone());
            queue.push_back(silent_segment.clone());
            queue.push_back(silent_segment);
        }
    }

    fn process_buffer(&mut self, buffer: &mut [f32], channels: usize) {
        let channels = 2;
        let frames = buffer.len() / channels;
        let mut queue = self.playback_queue.lock().unwrap();

        if queue.is_empty() {
            // Fill buffer with silence if queue is empty
            for sample in buffer.iter_mut() {
                *sample = 0.0;
            }
            return;
        }

        // --- Immediate crossfade on rate change ---
        if let Ok(mut flag) = self.rate_change_flag.lock() {
            if *flag {
                debug!(target: "resynth::playback", "[rate change] Immediate switch to new segment");
                if queue.len() > 1 {
                    self.prev_segment = queue.pop_front();
                    debug!(target: "resynth::playback", "Popped segment from queue for rate change, new queue_len={}", queue.len());
                }
                self.sample_counter = 0;
                self.crossfade_state = Some(CrossfadeState::Idle);
                *flag = false;
            }
        }

        // --- Main playback loop with state machine ---
        for frame in 0..frames {
            match self.crossfade_state.as_ref().unwrap() {
                CrossfadeState::Active { old_pos, fade_len } => {
                    let current_segment = queue.front().unwrap();
                    let prev = self.prev_segment.as_ref().unwrap();
                    let fade_len = *fade_len;
                    let crossfade_start = *old_pos;
                    
                    for ch in 0..channels {
                        if ch < current_segment.samples.len() && ch < prev.samples.len() {
                            let t = self.sample_counter as f32 / (fade_len - 1) as f32;
                            let fade_in = t;
                            let fade_out = 1.0 - t;
                            let prev_idx = crossfade_start + self.sample_counter;
                            let prev_val = if prev_idx < prev.length { prev.samples[ch][prev_idx] } else { 0.0 };
                            let curr_val = current_segment.samples[ch][self.sample_counter];
                            buffer[frame * channels + ch] = prev_val * fade_out + curr_val * fade_in;
                        }
                    }
                    
                    if self.sample_counter == fade_len - 1 {
                        debug!(target: "resynth::playback", "[crossfade] Crossfade complete at sample_counter={}", self.sample_counter);
                        self.crossfade_state = Some(CrossfadeState::Idle);
                        self.prev_segment = None;
                        self.crossfade_len = 0;
                        self.crossfade_start_pos = None;
                        self.sample_counter = 0;
                    } else {
                        self.sample_counter += 1;
                    }
                }
                CrossfadeState::Idle => {
                    // Normal playback
                    let current_segment = queue.front().unwrap();
                    if self.sample_counter >= current_segment.length {
                        // If sample_counter is out of bounds, reset and fill rest of buffer with silence
                        self.sample_counter = 0;
                        for i in frame..frames {
                            for ch in 0..channels {
                                buffer[i * channels + ch] = 0.0;
                            }
                        }
                        break;
                    }
                    for ch in 0..channels {
                        if ch < current_segment.samples.len() {
                            buffer[frame * channels + ch] = current_segment.samples[ch][self.sample_counter];
                        }
                    }
                    if self.sample_counter >= current_segment.length - 1 {
                        debug!(target: "resynth::playback", "Segment finished (len={}), sample_counter={}, queue_len={}", current_segment.length, self.sample_counter, queue.len());
                        self.sample_counter = 0;
                        if queue.len() > 1 {
                            self.prev_segment = queue.pop_front();
                            debug!(target: "resynth::playback", "Popped segment from queue, new queue_len={}", queue.len());
                            if let (Some(ref prev), Some(current_segment)) = (self.prev_segment.as_ref(), queue.front()) {
                                self.crossfade_len = (prev.length.min(current_segment.length) / 3).max(1);
                                debug!(target: "resynth::playback", "[segment] Crossfade window set: {} samples (old: {}, new: {})", self.crossfade_len, prev.length, current_segment.length);
                                self.crossfade_start_pos = Some(0);
                            }
                        }
                    } else {
                        self.sample_counter += 1;
                    }
                }
            }
        }
        debug!(target: "resynth::playback", "Buffer processed: frames={}, queue_len={}, sample_counter={}, crossfade_state={:?}", frames, queue.len(), self.sample_counter, self.crossfade_state);
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
// Proactive segment generation: always prepare the next segment in advance
// Never interrupt segment generation. On update/refresh, store the latest update as 'pending'.
// The segment generation thread always finishes the current segment, then uses the latest pending update for the next segment.
// Always keep the queue full.
pub fn start_wavegen_thread(
    update_rx: mpsc::Receiver<SynthUpdate>,
    playback_queue: Arc<Mutex<VecDeque<WaveSegment>>>,
    sample_rate: f32,
    max_queue_size: usize,
    rate_change_flag: Arc<Mutex<bool>>,
) {
    // Shared state for latest update
    let pending_update = Arc::new(Mutex::new(None::<SynthUpdate>));
    let update_cvar = Arc::new(Condvar::new());

    // Update/refresh thread: just stores the latest update
    {
        let pending_update = pending_update.clone();
        let update_cvar = update_cvar.clone();
        thread::spawn(move || {
            while let Ok(update) = update_rx.recv() {
                debug!(target: "resynth::wavegen", "[main] Received update: update_rate={:.3}", update.update_rate);
                let mut guard = pending_update.lock().unwrap();
                *guard = Some(update);
                update_cvar.notify_one();
            }
        });
    }

    // Segment generation thread: always finish current segment, then use latest update
    {
        let playback_queue = playback_queue.clone();
        let pending_update = pending_update.clone();
        let update_cvar = update_cvar.clone();
        let rate_change_flag = rate_change_flag.clone();
        // Add a local WaveSynth for rate change handling
        let mut local_synth = WaveSynth::new(sample_rate);
        thread::spawn(move || {
            let mut last_update: Option<SynthUpdate> = None;
            let mut update_rate = DEFAULT_UPDATE_RATE;
            let mut pending_rate_change = false;
            loop {
                // Wait for an update if we don't have one
                let update = {
                    let mut guard = pending_update.lock().unwrap();
                    while guard.is_none() {
                        guard = update_cvar.wait(guard).unwrap();
                    }
                    guard.clone().unwrap()
                };
                // If update rate changed, set pending_rate_change
                if last_update.as_ref().map(|u| u.update_rate).unwrap_or(DEFAULT_UPDATE_RATE) != update.update_rate {
                    debug!(target: "resynth::wavegen", "[segment] Update rate changed: {:.3} -> {:.3}, will clear queue after new segment is ready", update_rate, update.update_rate);
                    update_rate = update.update_rate;
                    pending_rate_change = true;
                    // Reset synth state and queue for new rate
                    local_synth.handle_rate_change(update_rate);
                }
                // Always keep the queue full
                loop {
                    let mut queue = playback_queue.lock().unwrap();
                    if queue.len() >= max_queue_size {
                        drop(queue);
                        // Wait for a new update or for space in the queue
                        let _ = update_cvar.wait_timeout(pending_update.lock().unwrap(), Duration::from_millis(10)).unwrap();
                        continue;
                    }
                    drop(queue);
                    // Use the latest update for the next segment
                    let update = {
                        let guard = pending_update.lock().unwrap();
                        guard.clone().unwrap()
                    };
                    let segment_length = (sample_rate * update.update_rate) as usize;
                    let stereo_partials = WaveSynth::combine_partials_to_stereo(&update.partials);
                    let mut pure = WaveSegment {
                        samples: vec![vec![0.0; segment_length]; 2],
                        length: segment_length,
                    };
                    debug!(target: "resynth::wavegen", "[segment] Starting segment generation: len={}, update_rate={:.3}", segment_length, update.update_rate);
                    for ch in 0..2 {
                        for i in 0..segment_length {
                            let time = i as f32 / sample_rate;
                            let mut sample = 0.0;
                            for &(freq, amp) in stereo_partials[ch].iter().filter(|&&(f, a)| f > 0.0 && a > 0.0) {
                                let phase = 2.0 * std::f32::consts::PI * (freq * update.freq_scale) * time;
                                sample += amp * phase.sin();
                            }
                            pure.samples[ch][i] = sample * update.gain * 0.01;
                        }
                    }
                    debug!(target: "resynth::wavegen", "[segment] Segment generation complete");
                    // If pending_rate_change, clear the queue and push only the new segment
                    if pending_rate_change {
                        let mut queue = playback_queue.lock().unwrap();
                        queue.clear();
                        queue.push_back(pure);
                        debug!(target: "resynth::wavegen", "[segment] Cleared queue and pushed new segment after rate change");
                        if let Ok(mut flag) = rate_change_flag.lock() {
                            *flag = true;
                        }
                        pending_rate_change = false;
                    } else {
                        let mut queue = playback_queue.lock().unwrap();
                        queue.push_back(pure);
                        debug!(target: "resynth::wavegen", "[segment] Queue length after push: {}", queue.len());
                    }
                    last_update = Some(update);
                    break;
                }
            }
        });
    }
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
    let rate_change_flag = Arc::new(Mutex::new(false));
    // Start wavegen thread
    start_wavegen_thread(update_rx, playback_queue, sample_rate as f32, max_queue_size, rate_change_flag.clone());
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
    let rate_change_flag = Arc::new(Mutex::new(false));
    let synth = WaveSynth {
        playback_queue: playback_queue.clone(),
        sample_counter: 0,
        sample_rate: sample_rate as f32,
        update_rate: DEFAULT_UPDATE_RATE,
        current_gain: 1.0,
        max_queue_size: 6,
        rate_change_flag: rate_change_flag.clone(),
        prev_segment: None,
        crossfade_len: 0,
        crossfade_start_pos: None,
        crossfade_state: Some(CrossfadeState::Idle),  // Initialize with Idle state
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
