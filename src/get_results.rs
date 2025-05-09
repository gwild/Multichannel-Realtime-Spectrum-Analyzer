use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use log::debug;
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
        
        debug!(target: "get_results", "Update thread started, beginning main loop");
        
        while !shutdown_flag.load(Ordering::Relaxed) {
            let update_rate = {
                let config = config.lock().unwrap();
                config.update_rate
            };

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
                    
                    // Log a separator for better readability
                    debug!(target: "get_results", "----------------------------------------");
                    debug!(target: "get_results", "FFT Data Update #{}", update_count);
                    
                    // Log data for each channel
                    for (channel_idx, channel_data) in current.data.iter().enumerate() {
                        // Log channel header with total partials count
                        let active_partials = channel_data.iter()
                            .filter(|&&(f, a)| f > 0.0 && a > 0.0)
                            .count();
                            
                        debug!(target: "get_results", 
                            "Channel {} - Active partials: {}/{}", 
                            channel_idx, 
                            active_partials,
                            channel_data.len()
                        );
                        
                        // Log each active partial's data
                        for (i, &(freq, amp)) in channel_data.iter()
                            .filter(|&&(f, a)| f > 0.0 && a > 0.0)
                            .enumerate() 
                        {
                            debug!(target: "get_results", 
                                "  [{:2}] f={:8.1} Hz, amp={:6.1} dB", 
                                i,
                                freq, 
                                amp
                            );
                        }
                        
                        // Add a blank line between channels
                        if channel_idx < current.data.len() - 1 {
                            debug!(target: "get_results", "");
                        }
                    }
                    debug!(target: "get_results", "----------------------------------------");

                    let update = SynthUpdate {
                        partials: current.data.clone(),
                        gain: config_snapshot.gain,
                        freq_scale: config_snapshot.freq_scale,
                        smoothing: config_snapshot.smoothing,
                        update_rate: config_snapshot.update_rate,
                    };

                    match update_queue.push(update) {
                        Ok(_) => debug!(target: "get_results", 
                            "Successfully pushed update #{} to synthesis queue", 
                            update_count
                        ),
                        Err(e) => debug!(target: "get_results", 
                            "Failed to push update #{} to synthesis queue: {:?}", 
                            update_count, 
                            e
                        ),
                    }
                    last_update = Instant::now();
                }
            }

            let sleep_duration = if last_update.elapsed().as_secs_f32() >= update_rate {
                Duration::from_millis(1)
            } else {
                let time_to_next = (update_rate - last_update.elapsed().as_secs_f32()).max(0.0);
                Duration::from_secs_f32((time_to_next / 10.0).max(0.010))
            };
            thread::sleep(sleep_duration);
        }
        debug!(target: "get_results", "Update thread shutting down after {} updates", update_count);
    });
} 