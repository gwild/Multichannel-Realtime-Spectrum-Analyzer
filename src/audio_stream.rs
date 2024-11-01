use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Stream, StreamConfig};
use std::sync::{Arc, Mutex};
use std::io::{self, Write};
use crate::fft_analysis::{compute_spectrum, NUM_PARTIALS};
use crate::plot::SpectrumApp;
use crate::conversion::{AudioSample, convert_i32_buffer_to_f32, f32_to_i16}; // Import conversions

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

// Function to list supported input formats
fn list_supported_input_formats(device: &cpal::Device) -> Result<(), anyhow::Error> {
    let supported_configs = device.supported_input_configs()?;
    println!("Supported input configurations:");
    for config in supported_configs {
        println!(
            "Channels: {}, Sample Rate: {:?}, Sample Format: {:?}",
            config.channels(),
            config.min_sample_rate()..=config.max_sample_rate(),
            config.sample_format()
        );
    }
    Ok(())
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

    // List supported input formats for the selected device
    list_supported_input_formats(device)?;

    // Iterate over supported configurations to find a compatible one
    let supported_configs = device.supported_input_configs()?;
    let mut stream = None;

    for supported_config in supported_configs {
        let sample_format = supported_config.sample_format();
        println!("Trying sample format: {:?}", sample_format);

        let config = supported_config.with_max_sample_rate().into();

        // Clone the Arc and Vec for each iteration
        let audio_buffers_clone = Arc::clone(&audio_buffers);
        let spectrum_app_clone = Arc::clone(&spectrum_app);
        let selected_channels_clone = selected_channels.clone();

        stream = match sample_format {
            cpal::SampleFormat::I16 => Some(device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let data_as_f32: Vec<f32> = data.iter().map(|s| *s as f32).collect();
                    process_samples(data_as_f32, channels, &audio_buffers_clone, &spectrum_app_clone, &selected_channels_clone, sample_rate);
                },
                move |err| {
                    eprintln!("Stream error: {:?}", err);
                },
                None,
            )?),
            cpal::SampleFormat::U16 => Some(device.build_input_stream(
                &config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let data_as_f32: Vec<f32> = data.iter().map(|s| *s as f32).collect();
                    process_samples(data_as_f32, channels, &audio_buffers_clone, &spectrum_app_clone, &selected_channels_clone, sample_rate);
                },
                move |err| {
                    eprintln!("Stream error: {:?}", err);
                },
                None,
            )?),
            cpal::SampleFormat::I32 => Some(device.build_input_stream(
                &config,
                move |data: &[i32], _: &cpal::InputCallbackInfo| {
                    let data_as_f32 = convert_i32_buffer_to_f32(data, channels);
                    process_samples(data_as_f32, channels, &audio_buffers_clone, &spectrum_app_clone, &selected_channels_clone, sample_rate);
                },
                move |err| {
                    eprintln!("Stream error: {:?}", err);
                },
                None,
            )?),
            cpal::SampleFormat::F32 => Some(device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    process_samples(data.to_vec(), channels, &audio_buffers_clone, &spectrum_app_clone, &selected_channels_clone, sample_rate);
                },
                move |err| {
                    eprintln!("Stream error: {:?}", err);
                },
                None,
            )?),
            _ => {
                eprintln!("Unsupported sample format: {:?}", sample_format);
                None
            },
        };

        if stream.is_some() {
            println!("Stream built successfully with format: {:?}", sample_format);
            break;
        }
    }

    stream.ok_or_else(|| anyhow::anyhow!("Unsupported sample format in Rust"))
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
            // println!("Pushed sample to channel {}: {}", channel, sample); // Debugging line
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
            for (j, &partial) in computed_partials.iter().enumerate().take(NUM_PARTIALS) {
                partials_results[i][j] = partial;
            }
            println!("Channel {}: Computed Partials: {:?}", channel, computed_partials); // Debugging line
        }

        println!("Channel {}: Partial Results: {:?}", channel, partials_results[i]);
    }

    // Update the spectrum_app with the new partials results
    let mut app = spectrum_app.lock().unwrap();
    app.partials = partials_results;
    // println!("Updated spectrum app with new partials results."); // Debugging line
}

// Function for processing audio stream (optional, for output purposes)
pub fn process_audio_stream(
    input_samples: &[f32],
    output_buffer: &mut [i16],
    selected_channels: &[usize],
) {
    // Convert input samples from f32 to i16
    let converted_samples = f32_to_i16(input_samples);

    // Process each channel separately
    for &channel in selected_channels {
        let channel_samples = &mut output_buffer[channel * input_samples.len()..(channel + 1) * input_samples.len()];
        channel_samples.copy_from_slice(&converted_samples);
        println!("Processed channel {} for output.", channel); // Debugging line
    }
}
