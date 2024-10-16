use anyhow::Result;
use cpal::traits::DeviceTrait;
use cpal::{Stream, StreamConfig, Sample, SizedSample};
use std::sync::{Arc, Mutex};
use num_traits::ToPrimitive;
use crate::fft_analysis::compute_spectrum;
use crate::plot::SpectrumApp;
use std::fmt::Debug;

const MAX_BUFFER_SIZE: usize = 1024; // Define a max buffer size
const MAX_I32: f32 = i32::MAX as f32; // Define the max value for scaling

// Trait for processing audio samples
pub trait AudioSample {
    fn to_f32(&self) -> f32; // Convert to f32
}

impl AudioSample for f32 {
    fn to_f32(&self) -> f32 {
        *self // Already f32
    }
}

impl AudioSample for i32 {
    fn to_f32(&self) -> f32 {
        (*self as f32 / MAX_I32).clamp(-1.0, 1.0) // Scale i32 to f32
    }
}

// Implementing the trait for additional types
impl AudioSample for i16 {
    fn to_f32(&self) -> f32 {
        (*self as f32 / i16::MAX as f32).clamp(-1.0, 1.0) // Scale i16 to f32
    }
}

impl AudioSample for u16 {
    fn to_f32(&self) -> f32 {
        (*self as f32 / u16::MAX as f32).clamp(0.0, 1.0) // Scale u16 to f32
    }
}

impl AudioSample for f64 {
    fn to_f32(&self) -> f32 {
        *self as f32 // Convert f64 to f32 directly
    }
}

// Build input stream function
pub fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    audio_buffers: Arc<Vec<Mutex<Vec<f32>>>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>, // List of channels to use
) -> Result<Stream>
where
    T: Sample + SizedSample + ToPrimitive + Debug + AudioSample + 'static,
{
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0;

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // Fill the audio buffers for each selected channel
            let mut buffer_f32: Vec<f32> = Vec::with_capacity(data.len());

            // Convert all samples to f32
            for sample in data.iter() {
                buffer_f32.push(AudioSample::to_f32(sample));
            }

            // Distribute samples to appropriate buffers
            for (i, sample_as_f32) in buffer_f32.iter().enumerate() {
                let channel = i % channels;
                if selected_channels.contains(&channel) {
                    let buffer_index = selected_channels.iter().position(|&ch| ch == channel).unwrap();
                    let mut buffer = audio_buffers[buffer_index].lock().unwrap();
                    
                    buffer.push(*sample_as_f32);

                    // Manage buffer size by removing the oldest sample if it exceeds the max size
                    if buffer.len() > MAX_BUFFER_SIZE {
                        buffer.remove(0);
                    }
                }
            }

            // Compute spectrum for each selected channel and store the partials and FFT results
            let mut partials_results = Vec::with_capacity(selected_channels.len());
            let mut fft_results = Vec::with_capacity(selected_channels.len());

            for (i, &channel) in selected_channels.iter().enumerate() {
                let buffer = audio_buffers[i].lock().unwrap();

                // Compute the partials
                let partials = compute_spectrum(&buffer, sample_rate);
                let partials_converted: Vec<(f64, f64)> = partials
                    .iter()
                    .map(|&(freq, amp)| (f64::from(freq), f64::from(amp)))
                    .collect();
                partials_results.push(partials_converted);

                // Compute FFT results
                let fft = compute_spectrum(&buffer, sample_rate);
                let fft_converted: Vec<(f64, f64)> = fft
                    .iter()
                    .map(|&(freq, amp)| (f64::from(freq), f64::from(amp)))
                    .collect();
                fft_results.push(fft_converted);
            }

            // Update the spectrum_app with the new partials and FFT results for each channel
            let mut app = spectrum_app.lock().unwrap();
            app.partials.clone_from_slice(&partials_results);
            app.fft_results.clone_from_slice(&fft_results);
        },
        move |err| {
            eprintln!("Stream error: {:?}", err);
        },
        None,
    )?;

    Ok(stream)
}
