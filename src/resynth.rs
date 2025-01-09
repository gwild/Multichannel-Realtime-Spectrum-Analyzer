use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use portaudio as pa;
use log::{info, error};
use crate::plot::SpectrumApp;
use crate::fft_analysis::NUM_PARTIALS;

const SMOOTHING_FACTOR: f32 = 0.97; // Higher = smoother but slower response

#[derive(Clone)]
struct SmoothParam {
    current: f32,
    target: f32,
}

impl SmoothParam {
    fn new(value: f32) -> Self {
        Self {
            current: value,
            target: value,
        }
    }

    fn update(&mut self, target: f32, smoothing: f32) {
        self.target = target;
        let factor = smoothing * smoothing;
        self.current = self.current * factor + target * (1.0 - factor);
    }
}

#[derive(Clone)]
struct PartialState {
    freq: SmoothParam,
    amp: SmoothParam,
    phase: f32,
}

impl PartialState {
    fn new() -> Self {
        Self {
            freq: SmoothParam::new(0.0),
            amp: SmoothParam::new(0.0),
            phase: 0.0,
        }
    }
}

pub struct ResynthConfig {
    pub gain: f32,
    pub smoothing: f32,
}

impl Default for ResynthConfig {
    fn default() -> Self {
        Self {
            gain: 0.01,
            smoothing: 0.99,
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

        let output_params = pa::StreamParameters::<f32>::new(
            device_index, 
            2,
            true,
            0.1
        );

        let settings = pa::OutputStreamSettings::new(
            output_params,
            sample_rate,
            512,
        );

        // Initialize partial states for smoothing
        let num_channels = spectrum_app.lock().unwrap().clone_absolute_data().len();
        let mut partial_states = vec![vec![PartialState::new(); NUM_PARTIALS]; num_channels];

        let mut stream = match pa.open_non_blocking_stream(settings, move |args: pa::OutputStreamCallbackArgs<f32>| {
            let buffer = args.buffer;
            let frames = buffer.len() / 2;
            
            let partials = spectrum_app.lock().unwrap().clone_absolute_data();
            let gain = config.lock().unwrap().gain;
            let smoothing = config.lock().unwrap().smoothing;

            buffer.fill(0.0);

            for frame in 0..frames {
                let mut left = 0.0f32;
                let mut right = 0.0f32;

                for (channel, channel_partials) in partials.iter().enumerate() {
                    for (i, &(freq, amp)) in channel_partials.iter().enumerate() {
                        let state = &mut partial_states[channel][i];
                        
                        // Update smoothed parameters
                        state.freq.update(freq, smoothing);
                        state.amp.update(amp, smoothing);

                        if state.freq.current > 0.0 && state.amp.current > 0.0 {
                            let sample = state.amp.current * state.phase.sin();
                            
                            if channel % 2 == 0 {
                                left += sample;
                            } else {
                                right += sample;
                            }

                            // Update phase using smoothed frequency
                            state.phase += 2.0 * PI * state.freq.current / sample_rate as f32;
                            if state.phase >= 2.0 * PI {
                                state.phase -= 2.0 * PI;
                            }
                        }
                    }
                }

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