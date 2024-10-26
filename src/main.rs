mod audio_stream;
mod fft_analysis;
mod plot;

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
    for config in selected_device.supported_input_configs()? {
        println!("  {:?}", config);
    }

    // Get default input config
    let default_config = selected_device.default_input_config()?;
    println!("Default input config: {:?}", default_config);

    // Create a config with the default settings
    let config: StreamConfig = default_config.clone().into();

    // Prompt user for channel selection
    print!("Enter the channel numbers to use (comma-separated, e.g., 0,1): ");
    io::stdout().flush()?;
    let mut channel_input = String::new();
    io::stdin().read_line(&mut channel_input)?;

    // Parse channel input
    let selected_channels: Vec<usize> = channel_input
        .trim()
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    println!("Selected channels: {:?}", selected_channels);

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
    let stream = match default_config.sample_format() {
        SampleFormat::F32 => build_input_stream::<f32>(
            selected_device,
            &config,
            audio_buffers.clone(),
            spectrum_app.clone(),
            selected_channels.clone(),
        ),
        SampleFormat::I16 => build_input_stream::<i16>(
            selected_device,
            &config,
            audio_buffers.clone(),
            spectrum_app.clone(),
            selected_channels.clone(),
        ),
        SampleFormat::U16 => build_input_stream::<u16>(
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
            Box::new(plot::MyApp {
                spectrum: spectrum_app.clone(),  // Pass spectrum app to MyApp
            })
        }),
    ) {
        eprintln!("Error launching application: {:?}", e);
    }

    Ok(())
}
