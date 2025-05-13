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

// The old start_update_thread function that used ArrayQueue and ResynthConfig.snapshot()
// has been removed entirely as it was causing compilation errors and is no longer used.
// The active function is start_update_thread_with_sender.

pub fn start_update_thread_with_sender(
    config: Arc<Mutex<ResynthConfig>>,
    shutdown_flag: Arc<AtomicBool>,
    update_sender: mpsc::Sender<SynthUpdate>,
    mut partials_rx: broadcast::Receiver<PartialsData>,
) {
    debug!(target: "get_results", "Starting update thread for FFT data retrieval (mpsc sender)");

    thread::spawn(move || {
        let mut update_count = 0;
        let mut last_processed_gui_update_rate = config.lock().unwrap().update_rate; // Initialize with current GUI rate
        let mut last_actual_update_sent_time = Instant::now(); // Tracks time since last SynthUpdate was sent
        let mut consecutive_send_failures = 0;

        debug!(target: "get_results", "Update thread started, beginning main loop (mpsc sender). Initial GUI rate: {:.3}s", last_processed_gui_update_rate);
        
        while !shutdown_flag.load(Ordering::Relaxed) {
            let current_gui_config_guard = config.lock().unwrap();
            let current_gui_update_rate = current_gui_config_guard.update_rate;
            let current_gain = current_gui_config_guard.gain;
            let current_freq_scale = current_gui_config_guard.freq_scale;
            drop(current_gui_config_guard); // Release lock as soon as possible

            let mut should_send_update_now = false;

            // Condition 1: GUI update rate has changed significantly since last processing cycle.
            if (current_gui_update_rate - last_processed_gui_update_rate).abs() > 1e-6 {
                debug!(target: "get_results", 
                       "GUI Update rate changed: {:.3}s -> {:.3}s. Forcing immediate SynthUpdate generation.",
                       last_processed_gui_update_rate, current_gui_update_rate);
                last_processed_gui_update_rate = current_gui_update_rate; // Update the rate we've now processed
                should_send_update_now = true;
            }

            // Condition 2: Timer for the current (potentially new) GUI update rate has elapsed since the last send.
            if !should_send_update_now && last_actual_update_sent_time.elapsed().as_secs_f32() >= current_gui_update_rate {
                debug!(target: "get_results", 
                       "Scheduled update interval of {:.3}s elapsed. Generating SynthUpdate.", 
                       current_gui_update_rate);
                should_send_update_now = true;
            }

            if should_send_update_now {
                debug!(target: "get_results", "Condition met to send SynthUpdate.");
                let mut latest_partials_data_for_this_cycle: Option<PartialsData> = None;

                loop {
                    match partials_rx.try_recv() {
                        Ok(partials) => {
                            latest_partials_data_for_this_cycle = Some(partials);
                        }
                        Err(broadcast::error::TryRecvError::Empty) => {
                            break;
                        }
                        Err(broadcast::error::TryRecvError::Lagged(n)) => {
                            warn!(target: "get_results", "Partials receiver lagged by {} messages. Attempting to clear and get newest.", n);
                            continue;
                        }
                        Err(broadcast::error::TryRecvError::Closed) => {
                            warn!(target: "get_results", "Partials broadcast channel closed (mpsc). Shutting down update thread.");
                            shutdown_flag.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }

                debug!(target: "get_results", "Partials receiver processing complete. latest_partials_data_for_this_cycle is Some: {}", latest_partials_data_for_this_cycle.is_some());

                if let Some(linear_partials_data) = latest_partials_data_for_this_cycle {
                    update_count += 1;
                    debug!(target: "get_results", 
                           "Update #{}: Preparing SynthUpdate - Partials count: {}, Gain: {:.2}, FreqScale: {:.2}, GUIRate: {:.3}s",
                           update_count,
                           linear_partials_data.len(), 
                           current_gain, 
                           current_freq_scale, 
                           current_gui_update_rate);
                    debug!(target: "get_results", "Preparing SynthUpdate. Partials data: {} channels, first partial of ch0: {:?}", linear_partials_data.len(), linear_partials_data.get(0).and_then(|ch| ch.get(0)));

                    let update_payload = SynthUpdate {
                        partials: linear_partials_data,
                        gain: current_gain, 
                        freq_scale: current_freq_scale,
                        update_rate: current_gui_update_rate,
                    };

                    debug!(target: "get_results", "Attempting to send SynthUpdate to wavegen_thread.");
                    match update_sender.send(update_payload) {
                        Ok(_) => {
                            debug!(target: "get_results", 
                                   "Successfully sent SynthUpdate #{} to wavegen thread (mpsc)", 
                                   update_count);
                            last_actual_update_sent_time = Instant::now();
                            consecutive_send_failures = 0;
                        },
                        Err(e) => {
                            consecutive_send_failures += 1;
                            warn!(target: "get_results", 
                                  "Failed to send SynthUpdate #{} to wavegen thread (attempt {}): {}. Wavegen may have shut down.", 
                                  update_count,
                                  consecutive_send_failures,
                                  e);
                            if consecutive_send_failures > 5 {
                                error!(target: "get_results", "Too many consecutive send failures or channel closed. Shutting down update thread.");
                                shutdown_flag.store(true, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                } 
            }
            thread::sleep(Duration::from_millis(10));
        }
        debug!(target: "get_results", "Update thread shutting down after {} updates (mpsc sender)", update_count);
    });
} 