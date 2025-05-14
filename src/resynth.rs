use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};
use portaudio as pa;
use log::{info, error, debug, warn};
use crate::get_results::{start_update_thread_with_sender, GuiParameter};
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
const MAX_POSSIBLE_GUI_UPDATE_RATE_SECONDS: f32 = 30.0;
// This will be our fixed actual length for all generated audio segments.
const FIXED_AUDIO_SEGMENT_LEN_SECONDS: f32 = MAX_POSSIBLE_GUI_UPDATE_RATE_SECONDS;
const INSTANT_MUTE_FADE_DURATION_SECONDS: f32 = 0.020; // 20ms for a quick mute

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
    current_gain: f32, // GUI gain, applied at playback
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
            current_gain: 0.5, // Default gain
        }
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.current_gain = gain;
    }

    /// Called by the outer timed loop in start_resynth_thread to initiate a switch.
    pub fn prepare_for_crossfade(&mut self, new_segment: AudioSegment, gui_update_rate_for_fade: f32, new_segment_target_gain: f32) {
        // current_segment is guaranteed to be Some due to initialization in new().
        // The first call to this function will be with the first *actual* (non-silent) segment.
        debug!(target: "audio_streaming::resynth", "New segment received for crossfade. Current playing segment len: {}, New segment len: {}. Base fade rate: {:.3}s, Target gain for new segment: {:.3}",
            self.current_segment.as_ref().map_or(0, |s| s.len_frames),
            new_segment.len_frames,
            gui_update_rate_for_fade,
            new_segment_target_gain
        );

        // Log cursor positions at the start of crossfade
        debug!(target: "audio_streaming::resynth", "Crossfade START: current_cursor_frames={}, next_cursor_frames=0, fade_progress_frames=0", self.current_cursor_frames);

        self.next_segment = Some(new_segment);
        self.next_cursor_frames = 0; // New segment starts from its beginning for the fade-in

        let fade_duration_seconds = if new_segment_target_gain < 0.001 { // If gain is effectively zero
            debug!(target: "audio_streaming::resynth", "Target gain is near zero ({:.3}). Overriding fade to be very short ({:.3}s) for mute effect.", new_segment_target_gain, INSTANT_MUTE_FADE_DURATION_SECONDS);
            INSTANT_MUTE_FADE_DURATION_SECONDS
        } else {
            gui_update_rate_for_fade / 3.0 // Standard fade: 1/3 of the current GUI update rate
        };
        
        self.total_fade_duration_frames = (fade_duration_seconds * self.sample_rate).max(1.0) as usize;
        self.fade_progress_frames = 0;
        self.play_state = SynthPlayState::Crossfading;
        debug!(target: "audio_streaming::resynth", "Crossfade prepared: {} total fade frames (derived from {:.3}s duration).", self.total_fade_duration_frames, fade_duration_seconds);
    }

    /// Fills the output buffer with audio samples. Called by PortAudio callback.
    pub fn process_buffer(&mut self, out_buffer: &mut [f32]) {
        debug!(target: "audio_streaming::resynth", "PROCESS_BUFFER_ENTRY: WaveSynth::process_buffer entered. Play_state: {:?}", self.play_state);

        let frames_to_fill = out_buffer.len() / 2; // Assuming stereo

        for i in 0..frames_to_fill {
            let mut sample_l = 0.0;
            let mut sample_r = 0.0;

            match self.play_state {
                SynthPlayState::Playing => {
                    if let Some(curr) = &self.current_segment {
                        if self.current_cursor_frames < curr.len_frames {
                            sample_l = curr.left_samples[self.current_cursor_frames];
                            sample_r = curr.right_samples[self.current_cursor_frames];
                            if self.current_cursor_frames < 5 {
                                debug!(target: "audio_streaming::resynth", "Playing frame {}: L={:.4}, R={:.4} from current_segment (len {})", self.current_cursor_frames, sample_l, sample_r, curr.len_frames);
                            }
                            self.current_cursor_frames += 1;
                        } else {
                            // Current segment (could be initial silence or a played-out real segment) finished.
                            // Play silence until new one is prepared by outer loop and prepare_for_crossfade is called.
                            // This can happen if prepare_for_crossfade isn't called in time.
                            // warn!(target: "resynth::playback", "Playing: Current segment overrun. Cursor: {}, Len: {}. Outputting silence.", self.current_cursor_frames, curr.len_frames);
                        }
                    } else {
                        error!(target: "audio_streaming::resynth", "Playing: current_segment is None! This should not happen. Outputting silence.");
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
                        } else {
                            debug!(target: "audio_streaming::resynth", "Crossfade: current_segment underrun at frame {} (len {}). Outputting 0.", self.current_cursor_frames, curr.len_frames);
                        }
                    }
                    // next_segment is the incoming segment
                    if let Some(nxt) = &self.next_segment {
                        if self.next_cursor_frames < nxt.len_frames {
                            s_l_next = nxt.left_samples[self.next_cursor_frames];
                            s_r_next = nxt.right_samples[self.next_cursor_frames];
                        } else {
                            debug!(target: "audio_streaming::resynth", "Crossfade: next_segment underrun at frame {} (len {}). Outputting 0.", self.next_cursor_frames, nxt.len_frames);
                        }
                    }

                    sample_l = s_l_curr * fade_out_factor + s_l_next * fade_in_factor;
                    sample_r = s_r_curr * fade_out_factor + s_r_next * fade_in_factor;

                    // Log fade factors and cursor positions for first and last 5 frames of crossfade
                    if self.fade_progress_frames < 5 || self.fade_progress_frames + 5 >= self.total_fade_duration_frames {
                        debug!(target: "audio_streaming::resynth", 
                               "Crossfade frame {}: fade_out={:.4}, fade_in={:.4}, curr_cursor={}, next_cursor={}, total_fade_frames={}",
                               self.fade_progress_frames, fade_out_factor, fade_in_factor, self.current_cursor_frames, self.next_cursor_frames, self.total_fade_duration_frames);
                    }

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
                        debug!(target: "audio_streaming::resynth", "Crossfade END: current_cursor_frames={}, next_cursor_frames={}, total_fade_frames={}", self.current_cursor_frames, self.next_cursor_frames, self.total_fade_duration_frames);
                        debug!(target: "audio_streaming::resynth", "Crossfade complete. New segment takes over.");
                        self.current_segment = self.next_segment.take(); // The 'next' segment is now 'current'
                        self.current_cursor_frames = self.next_cursor_frames; // Continue from where the fade-in reached
                        self.play_state = SynthPlayState::Playing;
                        self.next_cursor_frames = 0; // Reset for the *next* potential 'next_segment'
                        self.fade_progress_frames = 0; // Reset for next fade
                    }
                }
            }
            // Apply gain at playback
            out_buffer[i * 2] = sample_l * self.current_gain;
            out_buffer[i * 2 + 1] = sample_r * self.current_gain;
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

        // --- Scaling to prevent sum of amplitudes > 1.0 ---
        let left_sum: f32 = left_ch_partials.iter().map(|p| p.1.abs()).sum();
        if left_sum > 1.0 {
            let scale = 1.0 / left_sum;
            for p in left_ch_partials.iter_mut() {
                p.1 *= scale;
            }
            debug!(target: "audio_streaming::resynth", "Scaling left partials by {:.4} to prevent clipping (sum was {:.4})", scale, left_sum);
        }
        let right_sum: f32 = right_ch_partials.iter().map(|p| p.1.abs()).sum();
        if right_sum > 1.0 {
            let scale = 1.0 / right_sum;
            for p in right_ch_partials.iter_mut() {
                p.1 *= scale;
            }
            debug!(target: "audio_streaming::resynth", "Scaling right partials by {:.4} to prevent clipping (sum was {:.4})", scale, right_sum);
        }
        [left_ch_partials, right_ch_partials]
    }
}

/// Generates audio segments based on SynthUpdate and places them into a shared slot.
fn start_wavegen_thread(
    update_rx: mpsc::Receiver<SynthUpdate>, // Receives updates from get_results
    incoming_segment_slot: Arc<Mutex<Option<AudioSegment>>>,
    sample_rate: f32,
    shutdown_flag: Arc<AtomicBool>,
) {
    info!(target: "resynth::wavegen", "Wavegen thread started. Segments will be fixed at {} seconds.", FIXED_AUDIO_SEGMENT_LEN_SECONDS);

    let fixed_segment_len_frames = (sample_rate * FIXED_AUDIO_SEGMENT_LEN_SECONDS).max(1.0) as usize;

    thread::spawn(move || {
        while !shutdown_flag.load(Ordering::Relaxed) {
            // Block for the definitive next update to process for a new segment
            let mut current_update = match update_rx.recv() { 
                Ok(upd) => upd,
                Err(_) => { // Changed from mpsc::RecvError for clarity with blocking recv
                    error!(target: "resynth::wavegen", "Update channel disconnected. Shutting down wavegen thread.");
                    break;
                }
            };

            debug!(target: "audio_streaming::resynth::wavegen", 
                   "Starting new segment synthesis with initial Gain: {:.2}, FScale: {:.2}, Partials: {} chans", 
                   current_update.gain, current_update.freq_scale, current_update.partials.len());

            let mut left_samples = vec![0.0f32; fixed_segment_len_frames];
            let mut right_samples = vec![0.0f32; fixed_segment_len_frames];
            // Initial combination of partials based on the starting update
            let mut stereo_partials_arrays = WaveSynth::combine_partials_to_stereo(&current_update.partials);

            const SUB_CHUNK_FRAMES: usize = 4096; // Approx 85ms at 48kHz. Tune as needed.
            let wavegen_segment_start_time = Instant::now();
            let mut max_abs_sample_val_pre_gain_this_segment = 0.0f32;

            for frame_chunk_start in (0..fixed_segment_len_frames).step_by(SUB_CHUNK_FRAMES) {
                if shutdown_flag.load(Ordering::Relaxed) { break; }

                // Before synthesizing this sub-chunk, check for newer updates from get_results
                match update_rx.try_recv() {
                    Ok(newly_arrived_update) => {
                        // Compare critical parameters to see if a meaningful change occurred
                        if newly_arrived_update.gain != current_update.gain || 
                           newly_arrived_update.freq_scale != current_update.freq_scale || 
                           newly_arrived_update.partials.len() != current_update.partials.len() || // Basic check for partials change
                           !newly_arrived_update.partials.iter().zip(current_update.partials.iter()).all(|(v1,v2)| v1.len() == v2.len()) // Deeper check if needed
                        {
                            debug!(target: "resynth::wavegen",
                                   "Mid-segment parameter change detected. Old Gain: {:.2} -> New Gain: {:.2}. Old FScale: {:.2} -> New FScale {:.2}. Switching params.",
                                   current_update.gain, newly_arrived_update.gain, current_update.freq_scale, newly_arrived_update.freq_scale);
                            current_update = newly_arrived_update; // Adopt new parameters
                            // Re-process partials if they have changed structure or content significantly
                            stereo_partials_arrays = WaveSynth::combine_partials_to_stereo(&current_update.partials); 
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => { /* No new update, continue with current_update */ }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        error!(target: "resynth::wavegen", "Update channel disconnected mid-segment. Shutting down.");
                        shutdown_flag.store(true, Ordering::Relaxed); // Signal global shutdown if not already
                        break; 
                    }
                }
                if shutdown_flag.load(Ordering::Relaxed) { break; } // Check again after try_recv

                // Synthesize one sub-chunk using current_update parameters
                for frame_idx_offset in 0..SUB_CHUNK_FRAMES {
                    let frame_idx = frame_chunk_start + frame_idx_offset;
                    if frame_idx >= fixed_segment_len_frames { break; }

                    let time = frame_idx as f32 / sample_rate;
                    
                    for ch_idx in 0..2 { // 0 for Left, 1 for Right
                        let target_buffer = if ch_idx == 0 { &mut left_samples } else { &mut right_samples };
                        let source_partials = &stereo_partials_arrays[ch_idx];
                        let mut sample_val = 0.0f32;

                        for &(freq, amp) in source_partials.iter() {
                            if freq > 0.0 && amp > 0.0 { // Ensure partials are valid
                                let phase = 2.0 * std::f32::consts::PI * (freq * current_update.freq_scale) * time;
                                sample_val += amp * phase.sin();
                            }
                        }
                        if sample_val.abs() > max_abs_sample_val_pre_gain_this_segment {
                            max_abs_sample_val_pre_gain_this_segment = sample_val.abs();
                        }
                        target_buffer[frame_idx] = sample_val;
                    }
                }
            } // End of sub-chunk synthesis loop

            if shutdown_flag.load(Ordering::Relaxed) { break; } // Check after main synthesis loop for the segment
            
            let segment_synthesis_duration = wavegen_segment_start_time.elapsed();
            // Log the gain that was active at the *end* of synthesis for this segment.
            debug!(target: "audio_streaming::resynth::wavegen", 
                   "Finished generating one full segment ({} frames, potentially with mid-params change). Duration: {:?}. Effective Gain at end: {:.2}. Max pre-gain sample val: {:.4}", 
                   fixed_segment_len_frames, segment_synthesis_duration, current_update.gain, max_abs_sample_val_pre_gain_this_segment);

            // Detailed sample logging (using the gain from the *final* state of current_update for this segment)
            let final_gain_for_segment_log = current_update.gain;
            if fixed_segment_len_frames > 0 {
                let num_samples_to_log = 5.min(fixed_segment_len_frames);
                let l_samples_head_str: Vec<String> = left_samples.iter().take(num_samples_to_log).map(|s| format!("{:.4}", s)).collect();
                let r_samples_head_str: Vec<String> = right_samples.iter().take(num_samples_to_log).map(|s| format!("{:.4}", s)).collect();
                
                let max_abs_left_post_gain = left_samples.iter().fold(0.0f32, |max, &val| max.max(val.abs()));
                let max_abs_right_post_gain = right_samples.iter().fold(0.0f32, |max, &val| max.max(val.abs()));

                debug!(target: "audio_streaming::resynth::wavegen_detail", 
                       "Generated segment samples (final gain for log: {:.4}). Left[0..{}]: [{}]. Right[0..{}]: [{}]. Max L Post-Gain: {:.4}, Max R Post-Gain: {:.4}", 
                       final_gain_for_segment_log, 
                       num_samples_to_log, l_samples_head_str.join(", "), 
                       num_samples_to_log, r_samples_head_str.join(", "),
                       max_abs_left_post_gain, max_abs_right_post_gain);
            }

            let new_segment = AudioSegment {
                left_samples,
                right_samples,
                len_frames: fixed_segment_len_frames, 
            };
            
            let mut slot_guard = incoming_segment_slot.lock().unwrap();
            *slot_guard = Some(new_segment);
            debug!(target: "audio_streaming::resynth::wavegen", "New segment placed in incoming_segment_slot.");

        } // End of main `while !shutdown_flag` loop
        info!(target: "resynth::wavegen", "Wavegen thread shutting down.");
    });
}

/// Starts a thread that performs real-time resynthesis of the analyzed spectrum.
pub fn start_resynth_thread(
    config: Arc<Mutex<ResynthConfig>>, // For GUI-driven update_rate (fade timing) and gain/freq_scale (for wavegen)
    device_index: pa::DeviceIndex,
    sample_rate: f64, // sample_rate for PortAudio and wave generation
    shutdown_flag: Arc<AtomicBool>,
    partials_rx_broadcast: broadcast::Receiver<PartialsData>, // From FFT analysis
    _num_input_channels: usize, // Informational, used by get_results
    _num_partials_config: usize, // Informational, used by get_results
    gui_param_rx: mpsc::Receiver<GuiParameter>, // NEW: Accept the receiver
    gain_update_rx: mpsc::Receiver<f32>, // NEW: Gain update receiver for instant volume
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
        gui_param_rx, // NEW: Pass it through
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
            // --- Instant Gain Update ---
            // Check for new gain values from the GUI and update WaveSynth immediately
            while let Ok(new_gain) = gain_update_rx.try_recv() {
                if let Ok(mut synth) = pa_synth_instance_accessor.lock() {
                    synth.set_gain(new_gain);
                    debug!(target: "audio_streaming::resynth", "Instant gain update: set to {:.4}", new_gain);
                }
            }

            // --- PortAudio Stream Management ---
            let current_resynth_config_locked = resynth_config_accessor.lock().unwrap();
            if current_resynth_config_locked.needs_restart.load(Ordering::Relaxed) {
                needs_pa_restart_due_to_config = true;
                current_resynth_config_locked.needs_restart.store(false, Ordering::Relaxed); // Consume flag
                debug!(target: "resynth::main", "PA stream restart explicitly requested by config.");
            }
            // Extract values needed later before dropping lock
            let gui_driven_update_rate_for_polling = current_resynth_config_locked.update_rate;
            let current_target_gain_for_fade_logic = current_resynth_config_locked.gain;
            drop(current_resynth_config_locked); // Release lock early

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
            // gui_driven_update_rate is now gui_driven_update_rate_for_polling
            // current_target_gain_for_fade_logic is available here
            
            // Check for a newly generated segment from wavegen_thread
            let new_segment_option = incoming_segment_slot.lock().unwrap().take();

            if let Some(new_segment) = new_segment_option {
                debug!(target: "resynth::main", "New segment taken from slot. Preparing for crossfade. Base GUI rate for fade: {:.3}s, Current target gain: {:.3}", gui_driven_update_rate_for_polling, current_target_gain_for_fade_logic);
                pa_synth_instance_accessor.lock().unwrap().prepare_for_crossfade(new_segment, gui_driven_update_rate_for_polling, current_target_gain_for_fade_logic);
            }
            
            // The sleep duration determines how often we check for new segments and potentially switch.
            // This should align with the GUI update rate to make switches timely.
            // A shorter sleep allows faster reaction to new segments in the slot if get_results is faster than GUI rate.
            // Let's sleep for a fraction of the GUI update rate to be responsive.
            let sleep_time_for_switch_check = Duration::from_millis(10); // Check much more frequently (100Hz)
            thread::sleep(sleep_time_for_switch_check);
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


