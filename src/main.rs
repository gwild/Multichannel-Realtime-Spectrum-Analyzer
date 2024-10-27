mod audio_stream;
mod fft_analysis;
mod plot;
mod convert;

use anyhow::{Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use audio_stream::{build_input_stream, CircularBuffer};

const MAX_BUFFER_SIZE: usize = 512;

fn main() -> Result<()> {
    let host = cpal::default_host();

    // Get a list of available input devices
    let devices: Vec<_> = host.input_devices()?.collect();

    // Print device names and indexes
    println!("Available Input Devices:");
    for (i, device) in devices.iter().enumerate() {
        println!("  [{}] - {}", i, device.name()?);
    }

    // Prompt user for device selection
    print!("Enter the index of the desired device: ");
    io::stdout().flush()?;
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;

    // Parse user input
    let device_index = user_input.trim().parse::<usize>()
        .map_err(|_| anyhow!("Invalid device index"))?;

    // Check for valid input within available devices
    if device_index >= devices.len() {
        return Err(anyhow!("Invalid device index. Exiting."));
    }

    let selected_device = &devices[device_index];
    println!("Selected device: {}", selected_device.name()?);

    // Print supported input configurations
    println!("Supported input configs:");
    let mut supported_configs = Vec::new();
    for (i, config) in selected_device.supported_input_configs()?.enumerate() {
        println!("  [{}]", i);
        println!("    Channels: {}", config.channels());
        println!("    Sample Rate: {} - {} Hz", config.min_sample_rate().0, config.max_sample_rate().0);
        println!("    Buffer Size: {:?}", config.buffer_size());
        println!("    Sample Format: {:?}", config.sample_format());
        println!();  // Add a blank line between configs for better readability
        supported_configs.push(config);
    }

    println!("Enter the index of the desired configuration:");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let config_index: usize = input.trim().parse()?;

    let selected_config = supported_configs[config_index].with_max_sample_rate();
    println!("Selected configuration:");
    println!("  Channels: {}", selected_config.channels());
    println!("  Sample Rate: {} Hz", selected_config.sample_rate().0);
    println!("  Buffer Size: {:?}", selected_config.buffer_size());
    println!("  Sample Format: {:?}", selected_config.sample_format());

    let config = StreamConfig::from(selected_config.clone());
    let sample_format = selected_config.sample_format();

    println!("\nEnter the channel numbers to use (comma-separated, e.g., 0,1):");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let selected_channels: Vec<usize> = input
        .trim()
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    println!("Selected channels: {:?}", selected_channels);

    // After parsing selected_channels
    let max_channel = config.channels as usize - 1;
    for &channel in &selected_channels {
        if channel > max_channel {
            return Err(anyhow!("Invalid channel selected. Maximum channel is {}", max_channel));
        }
    }

    // Create audio buffers for selected channels
    let audio_buffers: Arc<Vec<Mutex<CircularBuffer>>> = Arc::new(
        selected_channels
            .iter()
            .map(|_| Mutex::new(CircularBuffer::new(MAX_BUFFER_SIZE)))
            .collect(),
    );

    // Create spectrum app
    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));

    // Build the input stream
    let stream = match sample_format {
        SampleFormat::F32 => build_input_stream(
            selected_device,
            &config,
            audio_buffers.clone(),
            spectrum_app.clone(),
            selected_channels.clone(),
        ),
        SampleFormat::I16 => build_input_stream(
            selected_device,
            &config,
            audio_buffers.clone(),
            spectrum_app.clone(),
            selected_channels.clone(),
        ),
        SampleFormat::U16 => build_input_stream(
            selected_device,
            &config,
            audio_buffers.clone(),
            spectrum_app.clone(),
            selected_channels.clone(),
        ),
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    }?;

    // Start the stream
    stream.play()?;

    // Launch the eframe application for plotting
    let native_options = plot::NativeOptions::default();
    if let Err(e) = plot::run_native(
        "Real-Time Spectrum Analyzer",
        native_options,
        Box::new(move |_cc| {
            Box::new(plot::MyApp::new(spectrum_app.clone()))
        }),
    ) {
        eprintln!("Error launching application: {:?}", e);
    }

    Ok(())
}
