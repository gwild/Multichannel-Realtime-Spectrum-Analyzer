use anyhow::Result;
use cpal::traits::DeviceTrait;
use cpal::{Stream, StreamConfig, Sample, SizedSample};
use std::sync::{Arc, Mutex};
use num_traits::ToPrimitive;
use crate::fft_analysis::compute_spectrum;
use crate::plot::SpectrumApp;
use std::fmt::Debug;

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

// Circular buffer implementation
pub struct CircularBuffer {
    buffer: Vec<f32>,
    head: usize,
    size: usize,
}

impl CircularBuffer {
    pub fn new(size: usize) -> Self {
        CircularBuffer {
            buffer: vec![0.0; size],
            head: 0,
            size,
        }
    }

    fn push(&mut self, value: f32) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.size; // Wrap around
    }

    fn get(&self) -> &[f32] {
        &self.buffer
    }
}

// Build input stream function
pub fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
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
            for (i, sample) in data.iter().enumerate() {
                let channel = i % channels;
                if selected_channels.contains(&channel) {
                    let buffer_index = selected_channels.iter().position(|&ch| ch == channel).unwrap();
                    let mut buffer = audio_buffers[buffer_index].lock().unwrap();
                    
                    // Convert the sample and push it into the circular buffer
                    let sample_as_f32 = AudioSample::to_f32(sample);
                    buffer.push(sample_as_f32);
                }
            }

            // Compute spectrum for each selected channel and store the partials and FFT results
            let mut partials_results = Vec::with_capacity(selected_channels.len());
            let mut fft_results = Vec::with_capacity(selected_channels.len());

            for (i, &channel) in selected_channels.iter().enumerate() {
                let buffer = audio_buffers[i].lock().unwrap();
                let audio_data = buffer.get();

                if !audio_data.is_empty() { // Ensure there is data to process
                    // Compute the partials and FFT results
                    let partials = compute_spectrum(audio_data, sample_rate);

                    // Store the results as f32
                    partials_results.push(partials.clone());
                    fft_results.push(partials.clone());
                    
                    println!("Channel {}: Partial Results: {:?}", channel, partials);
                } else {
                    // Optionally log a warning if there is no data to process
                    eprintln!("No data in buffer for channel {} to process.", channel);
                }
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
