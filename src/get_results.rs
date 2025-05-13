use std::sync::{Arc, Mutex, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use log::{debug, warn, error};
use crate::ResynthConfig;
use crate::resynth::SynthUpdate;
use tokio::sync::broadcast;

// Define type alias
type PartialsData = Vec<Vec<(f32, f32)>>;

#[derive(Debug, Clone, Copy)]
pub enum GuiParameter {
    Gain(f32),
    FreqScale(f32),
    UpdateRate(f32),
}

// The old start_update_thread function that used ArrayQueue and ResynthConfig.snapshot()
// has been removed entirely as it was causing compilation errors and is no longer used.
// The active function is start_update_thread_with_sender.

pub fn start_update_thread_with_sender(
    config: Arc<Mutex<ResynthConfig>>,
    shutdown_flag: Arc<AtomicBool>,
    update_sender: mpsc::Sender<SynthUpdate>,
    mut partials_rx: broadcast::Receiver<PartialsData>,
    gui_param_rx: mpsc::Receiver<GuiParameter>,
) {
    debug!(target: "get_results", "Starting update thread (event-driven) for FFT data retrieval.");

    thread::spawn(move || {
        let mut update_count = 0;
        
        // Initialize local state from the initial config
        let initial_config_guard = config.lock().unwrap();
        let mut local_gain = initial_config_guard.gain;
        let mut local_freq_scale = initial_config_guard.freq_scale;
        let mut local_update_rate = initial_config_guard.update_rate;
        drop(initial_config_guard);
        debug!(target: "get_results", "Initial local state: Gain={:.2}, FreqScale={:.2}, UpdateRate={:.3}s",
               local_gain, local_freq_scale, local_update_rate);

        let mut last_actual_update_sent_time = Instant::now();
        let mut consecutive_send_failures = 0;
        let mut latest_known_partials: Option<PartialsData> = None;
        let mut last_sent_partials: Option<PartialsData> = None; // Cache for last successfully sent partials
        let mut force_update_next_cycle = false; // Flag to send update even if no new partials/GUI event this exact instant

        while !shutdown_flag.load(Ordering::Relaxed) {
            let mut immediate_send_triggered_this_cycle = false;
            let mut gui_event_occurred = false; // Specifically track if a GUI parameter change happened

            // 1. Check for GUI parameter changes (with timeout)
            //    Timeout is dynamically based on local_update_rate, ensuring responsiveness.
            //    Minimum timeout of 10ms, max of 500ms or half of update_rate.
            let timeout_duration = Duration::from_secs_f32((local_update_rate / 2.0).max(0.01).min(0.5));
            match gui_param_rx.recv_timeout(timeout_duration) {
                Ok(GuiParameter::Gain(g)) => {
                    if (g - local_gain).abs() > 1e-6 {
                        debug!(target: "get_results", "Event: GUI Gain changed: {:.2} -> {:.2}", local_gain, g);
                        local_gain = g;
                        immediate_send_triggered_this_cycle = true;
                        gui_event_occurred = true;
                    }
                }
                Ok(GuiParameter::FreqScale(fs)) => {
                    if (fs - local_freq_scale).abs() > 1e-6 {
                        debug!(target: "get_results", "Event: GUI FreqScale changed: {:.2} -> {:.2}", local_freq_scale, fs);
                        local_freq_scale = fs;
                        immediate_send_triggered_this_cycle = true;
                        gui_event_occurred = true;
                    }
                }
                Ok(GuiParameter::UpdateRate(ur)) => {
                    if (ur - local_update_rate).abs() > 1e-6 {
                        debug!(target: "get_results", "Event: GUI UpdateRate changed: {:.3}s -> {:.3}s", local_update_rate, ur);
                        local_update_rate = ur;
                        // Update rate change itself doesn't force an immediate send, the timer will adapt.
                        // However, if other changes were batched, this ensures they go out.
                        immediate_send_triggered_this_cycle = true; 
                        // gui_event_occurred = true; // An update rate change doesn't need to force using cached partials
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // This is expected. Proceed to check other conditions.
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    error!(target: "get_results", "GUI parameter channel disconnected. Shutting down update thread.");
                    shutdown_flag.store(true, Ordering::Relaxed);
                    break;
                }
            }

            // 2. Check for new partials (non-blocking, drain the channel)
            let mut new_partials_received_this_cycle = false;
            loop {
                match partials_rx.try_recv() {
                    Ok(partials_data) => {
                        latest_known_partials = Some(partials_data);
                        new_partials_received_this_cycle = true;
                    }
                    Err(broadcast::error::TryRecvError::Empty) => break, // No more messages
                    Err(broadcast::error::TryRecvError::Lagged(n)) => {
                        warn!(target: "get_results", "Partials receiver lagged by {} messages. Catching up.", n);
                        // Continue trying to get the newest, latest_known_partials will be updated if successful.
                        continue;
                    }
                    Err(broadcast::error::TryRecvError::Closed) => {
                        error!(target: "get_results", "Partials broadcast channel closed. Shutting down update thread.");
                        shutdown_flag.store(true, Ordering::Relaxed);
                        return; // Exit thread using return
                    }
                }
            }
            if new_partials_received_this_cycle {
                debug!(target: "get_results", "Event: New partials received and updated locally.");
                immediate_send_triggered_this_cycle = true;
            }

            // 3. Check if timer for the current local_update_rate has elapsed
            let timer_elapsed = last_actual_update_sent_time.elapsed().as_secs_f32() >= local_update_rate;
            if timer_elapsed {
                debug!(target: "get_results", "Event: Timer elapsed for update rate: {:.3}s", local_update_rate);
            }

            // Determine if we should send an update
            let should_attempt_send = immediate_send_triggered_this_cycle || timer_elapsed || force_update_next_cycle;

            if should_attempt_send {
                let mut selected_partials_for_send: Option<PartialsData> = None;

                if latest_known_partials.is_some() {
                    selected_partials_for_send = latest_known_partials.clone();
                } else if last_sent_partials.is_some() {
                    selected_partials_for_send = last_sent_partials.clone();
                    if gui_event_occurred { // Specifically log if cached was used for a GUI event
                        debug!(target: "get_results", "GUI event: No new partials, using cached partials for SynthUpdate.");
                    }
                }

                // If it was a critical GUI event (Gain/FreqScale, indicated by gui_event_occurred) 
                // and we *still* have no partials (i.e., selected_partials_for_send is None 
                // because both fresh and cached were None), then force send with empty partials.
                if gui_event_occurred && selected_partials_for_send.is_none() {
                    debug!(target: "get_results", "GUI event (Gain/FreqScale): No fresh or cached partials. Sending SynthUpdate with EMPTY partials to force parameter change.");
                    selected_partials_for_send = Some(Vec::new()); // Use empty Vec<Vec<(f32,f32)>>
                }

                if let Some(current_partials_to_send) = selected_partials_for_send {
                    update_count += 1;
                    let update_payload = SynthUpdate {
                        partials: current_partials_to_send.clone(), 
                        gain: local_gain,
                        freq_scale: local_freq_scale,
                        update_rate: local_update_rate, // Send the current *effective* update rate
                    };
                    debug!(target: "get_results", 
                           "Update #{}: Sending SynthUpdate. Gain={:.2}, FScale={:.2}, URate={:.3}s, Partials_Chans={}", 
                           update_count, local_gain, local_freq_scale, local_update_rate, current_partials_to_send.len());

                    match update_sender.send(update_payload) {
                        Ok(_) => {
                            last_actual_update_sent_time = Instant::now();
                            consecutive_send_failures = 0;
                            force_update_next_cycle = false; // Reset flag
                            last_sent_partials = Some(current_partials_to_send); // Cache successful sent partials
                        }
                        Err(e) => {
                            consecutive_send_failures += 1;
                            warn!(target: "get_results", "Failed to send SynthUpdate #{}, attempt {}: {}. Wavegen may have shut down.", 
                                  update_count, consecutive_send_failures, e);
                            force_update_next_cycle = true; // Retry next cycle
                            if consecutive_send_failures > 5 {
                                error!(target: "get_results", "Too many send failures. Shutting down update thread.");
                                shutdown_flag.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                } else {
                    // This 'else' block will now only be reached if:
                    // 1. It was NOT a gui_event_occurred (so it was timer or force_update_next_cycle)
                    // 2. AND latest_known_partials was None
                    // 3. AND last_sent_partials was None
                    // In this case, we defer for non-critical updates.
                    debug!(target: "get_results", "Update condition met (timer/forced), but no partials (fresh or cached) available. Will retry. Forced: {}", force_update_next_cycle);
                    force_update_next_cycle = true; // Ensure we try to send once partials (new or cached for GUI) become available.
                }
            }
            // If no conditions met, the loop will iterate based on gui_param_rx.recv_timeout implicitly sleeping.
        }
        debug!(target: "get_results", "Update thread (event-driven) shutting down after {} updates.", update_count);
    });
} 