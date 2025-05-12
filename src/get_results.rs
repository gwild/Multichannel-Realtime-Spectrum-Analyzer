use std::sync::{Arc, Mutex, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use log::{debug, warn};
use crossbeam_queue::ArrayQueue;
use crate::ResynthConfig;
use crate::resynth::SynthUpdate;
use crate::fft_analysis::CurrentPartials;

pub fn start_update_thread(
    config: Arc<Mutex<ResynthConfig>>,
    shutdown_flag: Arc<AtomicBool>,
    update_queue: Arc<ArrayQueue<SynthUpdate>>,
    current_partials: Arc<Mutex<CurrentPartials>>,
) {
    debug!(target: "get_results", "Starting update thread for FFT data retrieval");

    thread::spawn(move || {
        let mut last_update = Instant::now();
        let mut update_count = 0;
        let mut last_update_rate = 1.0;  // Track last update rate
        let mut consecutive_failures = 0;  // Track consecutive queue push failures
        
        debug!(target: "get_results", "Update thread started, beginning main loop");
        
        while !shutdown_flag.load(Ordering::Relaxed) {
            let update_rate = {
                let config = config.lock().unwrap();
                config.update_rate
            };

            // If update rate changed, wait for queue to clear
            if (update_rate - last_update_rate).abs() > 1e-6 {
                debug!(target: "get_results", 
                    "Update rate changed: {:.3}s -> {:.3}s, waiting for queue to clear",
                    last_update_rate, update_rate
                );
                while update_queue.len() > 0 && !shutdown_flag.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(10));
                }
                last_update_rate = update_rate;
                last_update = Instant::now();  // Reset timing after rate change
            }

            if last_update.elapsed().as_secs_f32() >= update_rate {
                let config_snapshot = config.lock().unwrap().snapshot();
                debug!(target: "get_results", 
                    "Update #{} - Config: gain={:.3}, freq_scale={:.3}, smoothing={:.3}, update_rate={:.3}s",
                    update_count + 1,
                    config_snapshot.gain,
                    config_snapshot.freq_scale,
                    config_snapshot.smoothing,
                    config_snapshot.update_rate
                );

                // Get current FFT data
                if let Ok(current) = current_partials.lock() {
                    update_count += 1;
                    
                    // Create update with current data
                    let update = SynthUpdate {
                        partials: current.data.clone(),
                        gain: config_snapshot.gain,
                        freq_scale: config_snapshot.freq_scale,
                        smoothing: config_snapshot.smoothing,
                        update_rate: config_snapshot.update_rate,
                    };

                    // Try to push update to queue with backoff on failure
                    match update_queue.push(update) {
                        Ok(_) => {
                            debug!(target: "get_results", 
                                "Successfully pushed update #{} to synthesis queue", 
                                update_count
                            );
                            consecutive_failures = 0;  // Reset failure counter on success
                            last_update = Instant::now();
                        },
                        Err(_) => {
                            consecutive_failures += 1;
                            warn!(target: "get_results", 
                                "Failed to push update #{} to synthesis queue (attempt {})", 
                                update_count,
                                consecutive_failures
                            );
                            
                            // Exponential backoff on consecutive failures
                            let backoff = Duration::from_millis(
                                (50 * consecutive_failures as u64).min(1000)
                            );
                            thread::sleep(backoff);
                            
                            // If too many failures, force a rate change pause
                            if consecutive_failures > 5 {
                                warn!(target: "get_results", "Too many consecutive failures, forcing pause");
                                thread::sleep(Duration::from_millis(500));
                                consecutive_failures = 0;
                            }
                        }
                    }
                }
            }

            // Adaptive sleep based on update rate and queue state
            let queue_usage = update_queue.len() as f32 / update_queue.capacity() as f32;
            let base_sleep = if queue_usage > 0.8 {
                // Sleep longer if queue is nearly full
                Duration::from_millis(50)
            } else {
                let time_to_next = (update_rate - last_update.elapsed().as_secs_f32()).max(0.0);
                Duration::from_secs_f32((time_to_next / 10.0).max(0.001))
            };
            thread::sleep(base_sleep);
        }
        debug!(target: "get_results", "Update thread shutting down after {} updates", update_count);
    });
}

pub fn start_update_thread_with_sender(
    config: Arc<Mutex<ResynthConfig>>,
    shutdown_flag: Arc<AtomicBool>,
    update_sender: mpsc::Sender<SynthUpdate>,
    current_partials: Arc<Mutex<CurrentPartials>>,
) {
    debug!(target: "get_results", "Starting update thread for FFT data retrieval (mpsc sender)");

    thread::spawn(move || {
        let mut last_update = Instant::now();
        let mut update_count = 0;
        let mut last_update_rate = 1.0;
        let mut consecutive_failures = 0;
        debug!(target: "get_results", "Update thread started, beginning main loop (mpsc sender)");
        while !shutdown_flag.load(Ordering::Relaxed) {
            let update_rate = {
                let config = config.lock().unwrap();
                config.update_rate
            };
            if (update_rate - last_update_rate).abs() > 1e-6 {
                debug!(target: "get_results", 
                    "Update rate changed: {:.3}s -> {:.3}s, pausing for rate change",
                    last_update_rate, update_rate
                );
                last_update_rate = update_rate;
                last_update = Instant::now();
            }
            if last_update.elapsed().as_secs_f32() >= update_rate {
                let config_snapshot = config.lock().unwrap().snapshot();
                debug!(target: "get_results", 
                    "Update #{}: Preparing update - Config (used for freq/smooth/rate): gain={:.3}, freq_scale={:.3}, smoothing={:.3}, update_rate={:.3}s",
                    update_count + 1,
                    config_snapshot.gain, // Gain from config is NOT used for resynth now
                    config_snapshot.freq_scale,
                    config_snapshot.smoothing,
                    config_snapshot.update_rate
                );
                if let Ok(current) = current_partials.lock() {
                    update_count += 1;

                    // Clone the original dB partials from current_partials
                    let db_partials = current.data.clone();

                    // --- Conversion Step --- 
                    // Convert partials to linear scale for resynthesis
                    let linear_partials: Vec<Vec<(f32, f32)>> = db_partials.iter().map(|channel_partials| {
                        channel_partials.iter().map(|&(freq, db_amp)| {
                            // Convert dB to linear amplitude: amp = 10^(dB/20)
                            let linear_amp = 10.0_f32.powf(db_amp / 20.0);
                            (freq, linear_amp)
                        }).collect()
                    }).collect();
                    // --- End Conversion Step --- 

                    // Create the update with linear partials and use the gain from the config snapshot
                    let update = SynthUpdate {
                        partials: linear_partials, // Use converted linear partials
                        gain: config_snapshot.gain, // Use gain from config (controlled by GUI slider)
                        freq_scale: config_snapshot.freq_scale,
                        smoothing: config_snapshot.smoothing,
                        update_rate: config_snapshot.update_rate,
                    };

                    match update_sender.send(update) {
                        Ok(_) => {
                            debug!(target: "get_results", 
                                "Successfully sent update #{} to wavegen thread (mpsc)", 
                                update_count
                            );
                            consecutive_failures = 0;
                            last_update = Instant::now();
                        },
                        Err(e) => {
                            consecutive_failures += 1;
                            warn!(target: "get_results", 
                                "Failed to send update #{} to wavegen thread (attempt {}): {}", 
                                update_count,
                                consecutive_failures,
                                e
                            );
                            let backoff = Duration::from_millis((50 * consecutive_failures as u64).min(1000));
                            thread::sleep(backoff);
                            if consecutive_failures > 5 {
                                warn!(target: "get_results", "Too many consecutive failures, forcing pause");
                                thread::sleep(Duration::from_millis(500));
                                consecutive_failures = 0;
                            }
                        }
                    }
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
        debug!(target: "get_results", "Update thread shutting down after {} updates (mpsc sender)", update_count);
    });
} 