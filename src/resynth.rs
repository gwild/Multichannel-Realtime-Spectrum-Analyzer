use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use portaudio as pa;
use log::{info, error};
use crate::plot::SpectrumApp;

pub struct ResynthConfig {
    pub gain: f32,  // 0.0 to 1.0
}

impl Default for ResynthConfig {
    fn default() -> Self {
        Self {
            gain: 0.5  // Default gain of 0.5
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
    thread::spawn(move || {
        let pa = match pa::PortAudio::new() {
            Ok(pa) => pa,
            Err(e) => {
                error!("Failed to initialize PortAudio for resynthesis: {}", e);
                return;
            }
        };

        // Setup output stream
        let output_params = pa::StreamParameters::<f32>::new(
            device_index, 
            2,  // Stereo output
            true,
            0.1  // latency
        );

        let settings = pa::OutputStreamSettings::new(
            output_params,
            sample_rate,
            512, // frames per buffer
        );

        // Phase accumulators for each partial
        let mut phases = vec![0.0f32; 24];  // 12 partials * 2 channels

        let mut stream = match pa.open_non_blocking_stream(settings, move |args: pa::OutputStreamCallbackArgs<f32>| {
            let buffer = args.buffer;
            let frames = buffer.len() / 2;  // Stereo output
            
            // Get current partials and gain
            let partials = spectrum_app.lock().unwrap().clone_absolute_data();
            let gain = config.lock().unwrap().gain;

            // Clear buffer
            buffer.fill(0.0);

            // Synthesize each frame
            for frame in 0..frames {
                let mut left = 0.0f32;
                let mut right = 0.0f32;

                // Process each channel's partials
                for (channel, channel_partials) in partials.iter().enumerate() {
                    for (i, &(freq, amp)) in channel_partials.iter().enumerate() {
                        if freq > 0.0 && amp > 0.0 {
                            let phase = &mut phases[channel * 12 + i];
                            let sample = amp * phase.sin();
                            
                            // Route odd channels left, even channels right
                            if channel % 2 == 0 {
                                left += sample;
                            } else {
                                right += sample;
                            }

                            // Update phase
                            *phase += 2.0 * PI * freq / sample_rate as f32;
                            if *phase >= 2.0 * PI {
                                *phase -= 2.0 * PI;
                            }
                        }
                    }
                }

                // Apply gain and write to buffer
                let frame_offset = frame * 2;
                buffer[frame_offset] = (left * gain).clamp(-1.0, 1.0);
                buffer[frame_offset + 1] = (right * gain).clamp(-1.0, 1.0);
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