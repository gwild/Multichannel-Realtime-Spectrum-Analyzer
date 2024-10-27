use anyhow::Result;
use cpal::traits::{DeviceTrait}; // Removed HostTrait as it's unused
use cpal::{Stream, StreamConfig}; // Removed Sample and SizedSample as they're unused
use std::sync::{Arc, Mutex};
use crate::fft_analysis::{compute_spectrum, NUM_PARTIALS};
use crate::plot::SpectrumApp;
use crate::convert::{convert_i32_buffer_to_f32, f32_to_i16}; // Removed i16_to_f32 as it's unused

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
        if *self == i32::MIN {
            -1.0
        } else {
            *self as f32 / i32::MAX as f32
        }
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
pub fn build_input_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
) -> Result<Stream> {
    if selected_channels.is_empty() {
        return Err(anyhow::anyhow!("No channels selected"));
    }

    println!("Building input stream with config: {:?}", config);
    println!("Selected channels: {:?}", selected_channels);

    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0;

    // Determine the sample format and handle accordingly
    let sample_format = device.default_input_config()?.sample_format(); // Use method call
    let stream = match sample_format {
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let data_as_f32: Vec<f32> = data.iter().map(|s| AudioSample::to_f32(s)).collect();
                process_samples(data_as_f32, channels, &audio_buffers, &spectrum_app, &selected_channels, sample_rate);
            },
            move |err| {
                eprintln!("Stream error: {:?}", err);
            },
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let data_as_f32: Vec<f32> = data.iter().map(|s| AudioSample::to_f32(s)).collect();
                process_samples(data_as_f32, channels, &audio_buffers, &spectrum_app, &selected_channels, sample_rate);
            },
            move |err| {
                eprintln!("Stream error: {:?}", err);
            },
            None,
        )?,
        cpal::SampleFormat::I32 => device.build_input_stream(
            config,
            move |data: &[i32], _: &cpal::InputCallbackInfo| {
                let data_as_f32 = convert_i32_buffer_to_f32(data, channels);
                process_samples(data_as_f32, channels, &audio_buffers, &spectrum_app, &selected_channels, sample_rate);
            },
            move |err| {
                eprintln!("Stream error: {:?}", err);
            },
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                process_samples(data.to_vec(), channels, &audio_buffers, &spectrum_app, &selected_channels, sample_rate);
            },
            move |err| {
                eprintln!("Stream error: {:?}", err);
            },
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    println!("Stream built successfully");
    Ok(stream)
}

// Helper function to process samples
fn process_samples(
    data_as_f32: Vec<f32>,
    channels: usize,
    audio_buffers: &Arc<Vec<Mutex<CircularBuffer>>>,
    spectrum_app: &Arc<Mutex<SpectrumApp>>,
    selected_channels: &[usize],
    sample_rate: u32,
) {
    // Fill the audio buffers for each selected channel
    for (i, &sample) in data_as_f32.iter().enumerate() {
        let channel = i % channels;
        if selected_channels.contains(&channel) {
            let buffer_index = selected_channels.iter().position(|&ch| ch == channel).unwrap();
            let mut buffer = audio_buffers[buffer_index].lock().unwrap();
            buffer.push(sample);
        }
    }

    // Initialize partials_results with zeroes for all channels
    let mut partials_results = vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

    // Compute spectrum for each selected channel
    for (i, &channel) in selected_channels.iter().enumerate() {
        let buffer = audio_buffers[i].lock().unwrap();
        let audio_data = buffer.get();

        if !audio_data.is_empty() {
            let computed_partials = compute_spectrum(audio_data, sample_rate);
            // Only update partials that were actually computed
            for (j, &partial) in computed_partials.iter().enumerate().take(NUM_PARTIALS) {
                partials_results[i][j] = partial;
            }
        }

        println!("Channel {}: Partial Results: {:?}", channel, partials_results[i]);
    }

    // Update the spectrum_app with the new partials results
    let mut app = spectrum_app.lock().unwrap();
    app.partials = partials_results;
}

pub fn process_audio_stream(
    input_samples: &[f32],
    output_buffer: &mut [i16],
    selected_channels: &[usize],
) {
    // Convert input samples from f32 to i16
    let converted_samples = f32_to_i16(input_samples);

    // Process each channel separately
    for &channel in selected_channels {
        // Assuming `output_buffer` is organized by channels
        let channel_samples = &mut output_buffer[channel * input_samples.len()..(channel + 1) * input_samples.len()];
        channel_samples.copy_from_slice(&converted_samples);
    }
}
