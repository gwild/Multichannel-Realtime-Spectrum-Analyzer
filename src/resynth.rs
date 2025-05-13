use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};
use portaudio as pa;
use log::{info, error, debug, warn};
use crate::get_results::{start_update_thread_with_sender};
use tokio::sync::broadcast;

// Define type alias (same as other files)
type PartialsData = Vec<Vec<(f32, f32)>>;

// Constants for audio performance - with optimized values for JACK
#[cfg(target_os = "linux")]
const OUTPUT_BUFFER_SIZE: usize = 16384;  // Much larger for Linux/JACK compatibility

#[cfg(not(target_os = "linux"))]
const OUTPUT_BUFFER_SIZE: usize = 4096;  // Smaller on non-Linux platforms

pub const DEFAULT_UPDATE_RATE: f32 = 1.0; // Default update rate in seconds
// Maximum possible update rate from the GUI slider in plot.rs (currently 0.01 to 60.0 seconds)
const MAX_POSSIBLE_GUI_UPDATE_RATE_SECONDS: f32 = 60.0;
// This will be our fixed actual length for all generated audio segments.
const FIXED_AUDIO_SEGMENT_LEN_SECONDS: f32 = MAX_POSSIBLE_GUI_UPDATE_RATE_SECONDS;

/// Configuration for resynthesis
pub struct ResynthConfig {
    pub gain: f32,
    pub freq_scale: f32,  // Frequency scaling factor (1.0 = normal, 2.0 = one octave up, 0.5 = one octave down)
    pub update_rate: f32, // THIS IS THE GUI DRIVEN RATE for refresh/crossfade timing
    pub needs_restart: Arc<AtomicBool>,  // Flag to signal when stream needs to restart
}

impl Default for ResynthConfig {
    fn default() -> Self {
        Self {
            gain: 0.5,
            freq_scale: 1.0,
            update_rate: DEFAULT_UPDATE_RATE,
            needs_restart: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Parameter update structure from get_results -> wavegen_thread
#[derive(Clone, Debug)]
pub struct SynthUpdate {
    pub partials: Vec<Vec<(f32, f32)>>,
    pub gain: f32,
    pub freq_scale: f32,
    pub update_rate: f32,  // This is the rate at which this specific set of partials was generated/analyzed.
                           // It IS NOW USED by wavegen_thread to determine generated wave length.
}

/// Represents a segment of generated stereo audio.
#[derive(Clone)] // Clone needed for swapping and Option.take()
struct AudioSegment {
    left_samples: Vec<f32>,
    right_samples: Vec<f32>,
    len_frames: usize, // Length of this specific segment in frames
}

#[derive(Debug, Clone, Copy)]
enum SynthPlayState {
    Playing,
    Crossfading,
}

struct WaveSynth {
    current_segment: Option<AudioSegment>,
    next_segment: Option<AudioSegment>, // Segment to fade into

    // Cursors are frame indices within the respective segments
    current_cursor_frames: usize,
    next_cursor_frames: usize,

    fade_progress_frames: usize,
    total_fade_duration_frames: usize, // Based on GUI update_rate when fade started
    play_state: SynthPlayState,

    sample_rate: f32,
}

impl WaveSynth {
    fn new(sample_rate: f32) -> Self {
        // Initial silent segment is long to ensure safety during startup.
        let initial_segment_len_frames = (sample_rate * MAX_POSSIBLE_GUI_UPDATE_RATE_SECONDS).max(1.0) as usize;
        let initial_silent_segment = AudioSegment {
            left_samples: vec![0.0f32; initial_segment_len_frames],
            right_samples: vec![0.0f32; initial_segment_len_frames],
            len_frames: initial_segment_len_frames,
        };

        debug!(target: "resynth::synth", "WaveSynth initialized with a silent segment of {} frames.", initial_segment_len_frames);

        Self {
            current_segment: Some(initial_silent_segment),
            next_segment: None,
            current_cursor_frames: 0,
            next_cursor_frames: 0,
            fade_progress_frames: 0,
            total_fade_duration_frames: 0,
            play_state: SynthPlayState::Playing,
            sample_rate,
        }
    }

    /// Called by the outer timed loop in start_resynth_thread to initiate a switch.
    pub fn prepare_for_crossfade(&mut self, new_segment: AudioSegment, gui_update_rate_for_fade: f32) {
        // current_segment is guaranteed to be Some due to initialization in new().
        // The first call to this function will be with the first *actual* (non-silent) segment.
        debug!(target: "resynth::playback", "New segment received for crossfade. Current playing segment len: {}, New segment len: {}. Fade rate for crossfade: {:.3}s",
            self.current_segment.as_ref().map_or(0, |s| s.len_frames),
            new_segment.len_frames,
            gui_update_rate_for_fade
        );

        self.next_segment = Some(new_segment);
        self.next_cursor_frames = 0; // New segment starts from its beginning for the fade-in

        // The fade duration is 1/3 of the *current GUI update rate*.
        self.total_fade_duration_frames = ((gui_update_rate_for_fade / 3.0) * self.sample_rate).max(1.0) as usize;
        self.fade_progress_frames = 0;
        self.play_state = SynthPlayState::Crossfading;
        debug!(target: "resynth::playback", "Crossfade prepared: {} total fade frames.", self.total_fade_duration_frames);
    }

    /// Fills the output buffer with audio samples. Called by PortAudio callback.
    pub fn process_buffer(&mut self, out_buffer: &mut [f32]) {
        let frames_to_fill = out_buffer.len() / 2; // Assuming stereo

        for i in 0..frames_to_fill {
            let mut sample_l = 0.0;
            let mut sample_r = 0.0;

            match self.play_state {
                SynthPlayState::Playing => {
                    // current_segment should always be Some here because it's initialized in new()
                    if let Some(curr) = &self.current_segment {
                        if self.current_cursor_frames < curr.len_frames {
                            sample_l = curr.left_samples[self.current_cursor_frames];
                            sample_r = curr.right_samples[self.current_cursor_frames];
                            self.current_cursor_frames += 1;
                        } else {
                            // Current segment (could be initial silence or a played-out real segment) finished.
                            // Play silence until new one is prepared by outer loop and prepare_for_crossfade is called.
                            // This can happen if prepare_for_crossfade isn't called in time.
                            // warn!(target: "resynth::playback", "Playing: Current segment overrun. Cursor: {}, Len: {}. Outputting silence.", self.current_cursor_frames, curr.len_frames);
                        }
                    } else {
                        // This case should ideally not be reached if current_segment is always Some.
                        error!(target: "resynth::playback", "Playing: current_segment is None! This should not happen. Outputting silence.");
                    }
                }
                SynthPlayState::Crossfading => {
                    let fade_ratio = if self.total_fade_duration_frames > 0 {
                        (self.fade_progress_frames as f32 / self.total_fade_duration_frames as f32).min(1.0)
                    } else {
                        1.0 // Instant fade if duration is zero (should not happen with .max(1.0))
                    };
                    let fade_out_factor = 1.0 - fade_ratio;
                    let fade_in_factor = fade_ratio;

                    let mut s_l_curr = 0.0; let mut s_r_curr = 0.0;
                    let mut s_l_next = 0.0; let mut s_r_next = 0.0;

                    // current_segment is the outgoing segment
                    if let Some(curr) = &self.current_segment {
                        if self.current_cursor_frames < curr.len_frames {
                            s_l_curr = curr.left_samples[self.current_cursor_frames];
                            s_r_curr = curr.right_samples[self.current_cursor_frames];
                        }
                    }
                    // next_segment is the incoming segment
                    if let Some(nxt) = &self.next_segment {
                        if self.next_cursor_frames < nxt.len_frames {
                            s_l_next = nxt.left_samples[self.next_cursor_frames];
                            s_r_next = nxt.right_samples[self.next_cursor_frames];
                        }
                    }

                    sample_l = s_l_curr * fade_out_factor + s_l_next * fade_in_factor;
                    sample_r = s_r_curr * fade_out_factor + s_r_next * fade_in_factor;

                    // Advance cursor for the outgoing current_segment if it's still providing samples
                    if self.current_segment.is_some() && self.current_cursor_frames < self.current_segment.as_ref().unwrap().len_frames {
                        self.current_cursor_frames += 1;
                    }
                    // Advance cursor for the incoming next_segment if it's providing samples
                    if self.next_segment.is_some() && self.next_cursor_frames < self.next_segment.as_ref().unwrap().len_frames {
                         self.next_cursor_frames += 1;
                    }
                    self.fade_progress_frames += 1;

                    if self.fade_progress_frames >= self.total_fade_duration_frames {
                        debug!(target: "resynth::playback", "Crossfade complete. New segment takes over.");
                        self.current_segment = self.next_segment.take(); // The 'next' segment is now 'current'
                        self.current_cursor_frames = self.next_cursor_frames; // Continue from where the fade-in reached
                        self.play_state = SynthPlayState::Playing;
                        self.next_cursor_frames = 0; // Reset for the *next* potential 'next_segment'
                        self.fade_progress_frames = 0; // Reset for next fade
                    }
                }
            }
            out_buffer[i * 2] = sample_l;
            out_buffer[i * 2 + 1] = sample_r;
        }
    }

    // Helper to combine partials for stereo, can be used by wavegen_thread
    // This function was part of the old WaveSynth, kept for utility.
    pub fn combine_partials_to_stereo(partials_data: &[Vec<(f32, f32)>]) -> [Vec<(f32, f32)>; 2] {
        let mut left_ch_partials = Vec::new();
        let mut right_ch_partials = Vec::new();
        let mut left_sources_count = 0;
        let mut right_sources_count = 0;

        for (idx, channel_partials_vec) in partials_data.iter().enumerate() {
            if idx % 2 == 0 { // Even index for left channel
                left_ch_partials.extend_from_slice(channel_partials_vec);
                left_sources_count += 1;
            } else { // Odd index for right channel
                right_ch_partials.extend_from_slice(channel_partials_vec);
                right_sources_count += 1;
            }
        }

        // Attenuate if multiple sources contributed to L or R
        // User stated no normalization, this attenuation might be considered part of "mixing"
        // If only one original channel goes to L and one to R, counts will be 1, no change.
        if left_sources_count > 1 {
            for p in left_ch_partials.iter_mut() {
                p.1 /= left_sources_count as f32;
            }
        }
        if right_sources_count > 1 {
            for p in right_ch_partials.iter_mut() {
                p.1 /= right_sources_count as f32;
            }
        }
        [left_ch_partials, right_ch_partials]
    }
}

/// Generates audio segments based on SynthUpdate and places them into a shared slot.
fn start_wavegen_thread(
    mut update_rx: mpsc::Receiver<SynthUpdate>, // Receives updates from get_results
    incoming_segment_slot: Arc<Mutex<Option<AudioSegment>>>,
    sample_rate: f32,
    shutdown_flag: Arc<AtomicBool>,
) {
    info!(target: "resynth::wavegen", "Wavegen thread started. Segments will be fixed at {} seconds.", FIXED_AUDIO_SEGMENT_LEN_SECONDS);

    let fixed_segment_len_frames = (sample_rate * FIXED_AUDIO_SEGMENT_LEN_SECONDS).max(1.0) as usize;

    thread::spawn(move || {
        while !shutdown_flag.load(Ordering::Relaxed) {
            match update_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(update) => {
                    debug!(target: "resynth::wavegen", "Received SynthUpdate. Partials data: {} channels, first partial of ch0: {:?}", update.partials.len(), update.partials.get(0).and_then(|ch| ch.get(0)));
                    // The GUI update rate from `update.update_rate` is primarily used by get_results for send frequency
                    // and by the main resynth loop for polling and crossfade duration.
                    // Here, we always fill the entire fixed_segment_len_frames.
                    debug!(target: "resynth::wavegen",
                        "Received SynthUpdate (GUI rate: {:.3}s). Generating full segment of {} frames. Gain: {:.2}, FreqScale: {:.2}",
                        update.update_rate, fixed_segment_len_frames, update.gain, update.freq_scale);

                    // Initialize buffers to the fixed maximum length, filled with silence (will be overwritten).
                    let mut left_samples = vec![0.0f32; fixed_segment_len_frames];
                    let mut right_samples = vec![0.0f32; fixed_segment_len_frames];

                    let stereo_partials_arrays = WaveSynth::combine_partials_to_stereo(&update.partials);
                    let wavegen_start_time = Instant::now();

                    for ch_idx in 0..2 { // 0 for Left, 1 for Right
                        let target_buffer = if ch_idx == 0 { &mut left_samples } else { &mut right_samples };
                        let source_partials = &stereo_partials_arrays[ch_idx];

                        // Synthesize for the entire fixed_segment_len_frames.
                        for frame_idx in 0..fixed_segment_len_frames { 
                            let time = frame_idx as f32 / sample_rate;
                            let mut sample_val = 0.0f32;

                            for &(freq, amp) in source_partials.iter() {
                                if freq > 0.0 && amp > 0.0 {
                                    let phase = 2.0 * std::f32::consts::PI * (freq * update.freq_scale) * time;
                                    sample_val += amp * phase.sin();
                                }
                            }
                            target_buffer[frame_idx] = sample_val * update.gain;
                        }
                    }
                    
                    let wavegen_duration = wavegen_start_time.elapsed();
                    debug!(target: "resynth::wavegen", "Finished generating waveform data for {} frames. Duration: {:?}", fixed_segment_len_frames, wavegen_duration);

                    let new_segment = AudioSegment {
                        left_samples,
                        right_samples,
                        len_frames: fixed_segment_len_frames, // The actual length of the sample vectors
                    };

                    debug!(target: "resynth::wavegen", "Generated new audio segment. Actual Length: {} frames.", new_segment.len_frames);
                    
                    let mut slot_guard = incoming_segment_slot.lock().unwrap();
                    *slot_guard = Some(new_segment);
                    debug!(target: "resynth::wavegen", "New segment placed in incoming_segment_slot.");

                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    error!(target: "resynth::wavegen", "Update channel disconnected. Shutting down wavegen thread.");
                    break;
                }
            }
        }
        info!(target: "resynth::wavegen", "Wavegen thread shutting down.");
    });
}

/// Starts a thread that performs real-time resynthesis of the analyzed spectrum.
pub fn start_resynth_thread(
    config: Arc<Mutex<ResynthConfig>>, // For GUI-driven update_rate (fade timing) and gain/freq_scale (for wavegen)
    device_index: pa::DeviceIndex,
    sample_rate: f64, // sample_rate for PortAudio and wave generation
    shutdown_flag: Arc<AtomicBool>,
    mut partials_rx_broadcast: broadcast::Receiver<PartialsData>, // From FFT analysis
    _num_input_channels: usize, // Informational, used by get_results
    _num_partials_config: usize, // Informational, used by get_results
) {
    debug!(target: "resynth::main", "Initializing resynthesis thread. Sample rate: {}", sample_rate);
    let sample_rate_f32 = sample_rate as f32;

    // Create the shared WaveSynth instance for the PA callback
    let synth_callback_instance = Arc::new(Mutex::new(WaveSynth::new(sample_rate_f32)));

    // Slot for wavegen_thread to place newly generated segments
    let incoming_segment_slot = Arc::new(Mutex::new(None::<AudioSegment>));

    // Channel for get_results -> wavegen_thread
    let (update_tx_for_wavegen, update_rx_for_wavegen) = mpsc::channel::<SynthUpdate>();

    // Start the thread that gets partials from broadcast and sends SynthUpdate to wavegen
    start_update_thread_with_sender(
        Arc::clone(&config), // Used by get_results for gain, freq_scale, and its own timing
        Arc::clone(&shutdown_flag),
        update_tx_for_wavegen, // Sender for SynthUpdate
        partials_rx_broadcast, // Receiver of PartialsData from FFT
    );

    // Start the wave generation thread
    start_wavegen_thread(
        update_rx_for_wavegen,
        Arc::clone(&incoming_segment_slot),
        sample_rate_f32,
        Arc::clone(&shutdown_flag),
    );

    // Main loop for this thread: manages PortAudio stream and timed segment switching
    let resynth_thread_shutdown_flag = Arc::clone(&shutdown_flag);
    let resynth_config_accessor = Arc::clone(&config);
    let pa_synth_instance_accessor = Arc::clone(&synth_callback_instance);

    thread::spawn(move || {
        let mut pa_stream: Option<pa::Stream<pa::NonBlocking, pa::Output<f32>>> = None;
        let mut last_pa_restart_time = Instant::now();
        let mut consecutive_pa_errors = 0;
        
        let mut needs_pa_restart_due_to_config = false; // From ResynthConfig.needs_restart

        debug!(target: "resynth::main", "Entering main resynthesis loop (PA management and segment switching).");

        while !resynth_thread_shutdown_flag.load(Ordering::Relaxed) {
            // --- PortAudio Stream Management ---
            let current_resynth_config = resynth_config_accessor.lock().unwrap();
            if current_resynth_config.needs_restart.load(Ordering::Relaxed) {
                needs_pa_restart_due_to_config = true;
                current_resynth_config.needs_restart.store(false, Ordering::Relaxed); // Consume flag
                debug!(target: "resynth::main", "PA stream restart explicitly requested by config.");
            }
            drop(current_resynth_config); // Release lock

            let should_attempt_pa_restart = pa_stream.is_none() || needs_pa_restart_due_to_config;
            let backoff_duration = if consecutive_pa_errors > 0 {
                Duration::from_millis((500 * consecutive_pa_errors.min(10) as u64).min(5000))
            } else {
                Duration::from_millis(100) // Faster retry if no errors
            };

            if should_attempt_pa_restart && last_pa_restart_time.elapsed() >= backoff_duration {
                if let Some(mut stream) = pa_stream.take() {
                    let _ = stream.stop();
                    let _ = stream.close();
                    debug!(target: "resynth::main", "Stopped existing PA stream for restart.");
                }
                
                debug!(target: "resynth::main", "Attempting to setup/restart PA output stream.");
                match setup_audio_stream(
                    device_index,
                    sample_rate,
                    Arc::clone(&pa_synth_instance_accessor), // Pass WaveSynth for callback
                ) {
                    Ok(stream) => {
                        pa_stream = Some(stream);
                        consecutive_pa_errors = 0;
                        needs_pa_restart_due_to_config = false;
                        info!(target: "resynth::main", "PA output stream started/restarted successfully.");
                    }
                    Err(e) => {
                        error!(target: "resynth::main", "Failed to setup/restart PA output stream: {}. Retrying after backoff.", e);
                        consecutive_pa_errors += 1;
                        pa_stream = None; // Ensure it's None on failure
                    }
                }
                last_pa_restart_time = Instant::now();
            }

            // --- Timed Segment Switching Logic ---
            let gui_driven_update_rate = resynth_config_accessor.lock().unwrap().update_rate;
            
            // Check for a newly generated segment from wavegen_thread
            let new_segment_option = incoming_segment_slot.lock().unwrap().take();

            if let Some(new_segment) = new_segment_option {
                debug!(target: "resynth::main", "New segment taken from slot. Preparing for crossfade. GUI rate for fade: {:.3}s", gui_driven_update_rate);
                pa_synth_instance_accessor.lock().unwrap().prepare_for_crossfade(new_segment, gui_driven_update_rate);
            }
            
            // The sleep duration determines how often we check for new segments and potentially switch.
            // This should align with the GUI update rate to make switches timely.
            // A shorter sleep allows faster reaction to new segments in the slot if get_results is faster than GUI rate.
            // Let's sleep for a fraction of the GUI update rate to be responsive.
            let sleep_time_for_switch_check = (gui_driven_update_rate / 4.0).max(0.005); // Check fairly often, min 5ms
            thread::sleep(Duration::from_secs_f32(sleep_time_for_switch_check));
        }

        // Shutdown PA stream
        if let Some(mut stream) = pa_stream.take() {
            let _ = stream.stop();
            let _ = stream.close();
            debug!(target: "resynth::main", "Final PA stream stop and close.");
        }
        info!(target: "resynth::main", "Resynthesis thread (PA management and segment switching) shutting down.");
    });
}

/// Sets up and starts the PortAudio output stream.
fn setup_audio_stream(
    device_index: pa::DeviceIndex,
    sample_rate: f64,
    synth_instance: Arc<Mutex<WaveSynth>>, // WaveSynth instance for the audio callback
) -> Result<pa::Stream<pa::NonBlocking, pa::Output<f32>>, anyhow::Error> {
    let pa_ctx = pa::PortAudio::new()?;
    let device_info = pa_ctx.device_info(device_index)
        .map_err(|e| anyhow::anyhow!("Failed to get device info: {}", e))?;
    
    debug!(target: "resynth::pa_setup", "Setting up PA stream for device: {}, Output Channels: {}, Default SR: {}",
        device_info.name, device_info.max_output_channels, device_info.default_sample_rate);

    if device_info.max_output_channels < 2 {
        return Err(anyhow::anyhow!("Selected output device {} does not support stereo output (has {} channels).", device_info.name, device_info.max_output_channels));
    }

    let output_params = pa::StreamParameters::<f32>::new(
        device_index,
        2, // Force stereo output
        true, // Interleaved
        device_info.default_low_output_latency // Or default_high_output_latency for more stability
    );

    // Validate format
    pa_ctx.is_output_format_supported(output_params, sample_rate)
        .map_err(|e| anyhow::anyhow!("Output format not supported (SR: {}, Ch: 2): {}", sample_rate, e))?;
    
    let stream_settings = pa::OutputStreamSettings::new(
        output_params,
        sample_rate,
        OUTPUT_BUFFER_SIZE as u32 // PortAudio's internal buffer size, not our segment length
    );

    let callback = move |pa::OutputStreamCallbackArgs { buffer, frames, .. }| {
        // Ensure buffer has enough space for stereo: frames * 2
        if buffer.len() < frames * 2 {
            error!(target: "resynth::pa_callback", "PA callback buffer too small! Expected {}, got {}. Filling with silence.", frames * 2, buffer.len());
            for sample in buffer.iter_mut() { *sample = 0.0; }
            return pa::Continue;
        }
        
        // Assuming buffer is mutable slice for stereo interleaved data
        if let Ok(mut synth) = synth_instance.lock() {
            synth.process_buffer(buffer); // process_buffer now handles stereo internally
        } else {
            // Failed to lock synth, fill with silence to avoid PA issues
            warn!(target: "resynth::pa_callback", "Failed to lock WaveSynth in PA callback. Outputting silence.");
            for sample_pair in buffer.chunks_mut(2) {
                if sample_pair.len() == 2 {
                    sample_pair[0] = 0.0; // L
                    sample_pair[1] = 0.0; // R
                }
            }
        }
        pa::Continue
    };

    let mut stream = pa_ctx.open_non_blocking_stream(stream_settings, callback)
        .map_err(|e| anyhow::anyhow!("Failed to open PA non-blocking stream: {}", e))?;
    
    stream.start().map_err(|e| anyhow::anyhow!("Failed to start PA stream: {}", e))?;
    
    info!(target: "resynth::pa_setup", "PortAudio output stream started successfully on device '{}'.", device_info.name);
    Ok(stream)
}


