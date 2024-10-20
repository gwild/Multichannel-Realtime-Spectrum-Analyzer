mod audio_stream;
mod fft_analysis;
mod plot;
// mod convert;  // Add the new module

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::io::{self};
use std::sync::{Arc, Mutex};
use audio_stream::{build_input_stream, CircularBuffer}; // Import CircularBuffer
use plot::{MyApp, SpectrumApp};
use eframe::NativeOptions;

const MAX_BUFFER_SIZE: usize = 512; // Set your desired buffer size

fn main() -> Result<()> {
    let host = cpal::default_host();

    // Get a list of available input devices
    let devices: Vec<_> = host.devices()?.collect();

    // Print device names and indexes
    println!("Available Input Devices:");
    for (i, device) in devices.iter().enumerate() {
        println!("  [{}] - {}", i, device.name()?);
    }

    // Prompt user for device selection
    println!("Enter the index of the desired device: ");
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;

    // Parse user input
    let device_index = user_input.trim().parse::<usize>()?;

    // Check for valid input within available devices
    if device_index < devices.len() {
        let selected_device = devices[device_index].clone();

        // Retrieve supported input configurations
        let supported_configs: Vec<_> = selected_device.supported_input_configs()?.collect();

        // Print available sample rates
        println!("Available Sample Rates:");
        for (i, config) in supported_configs.iter().enumerate() {
            println!(
                "  [{}] - {} Hz ({} channels, format: {:?})",
                i,
                config.min_sample_rate().0,
                config.channels(),
                config.sample_format()
            );
        }

        // Prompt user for sample rate selection
        println!("Enter the index of the desired sample rate: ");
        user_input.clear();
        io::stdin().read_line(&mut user_input)?;

        // Parse user input
        let config_index = user_input.trim().parse::<usize>()?;

        // Check if the selected index is valid
        if config_index < supported_configs.len() {
            let selected_config = &supported_configs[config_index];

            // Set up the stream config with the selected sample rate and number of channels
            let stream_config = StreamConfig {
                channels: selected_config.channels(),
                sample_rate: selected_config.min_sample_rate(),
                buffer_size: cpal::BufferSize::Default,
            };

            // Get the number of available channels
            let num_channels = stream_config.channels as usize;

            // Print available channels
            println!("Available Channels (0-{}):", num_channels - 1);
            for i in 0..num_channels {
                println!("  Channel {}", i);
            }

            // Prompt user for channel selection
            println!("Enter the indices of the desired channels, separated by spaces (e.g., '0 1 2'): ");
            user_input.clear();
            io::stdin().read_line(&mut user_input)?;

            // Parse user input into a list of selected channels
            let selected_channels: Vec<usize> = user_input
                .trim()
                .split_whitespace()
                .filter_map(|s| s.parse::<usize>().ok())
                .filter(|&ch| ch < num_channels)  // Ensure selected channels are valid
                .collect();

            if selected_channels.is_empty() {
                println!("No valid channels selected. Exiting.");
                return Ok(());
            }

            // Initialize audio buffers for each selected channel using CircularBuffer
            let audio_buffers: Arc<Vec<Mutex<CircularBuffer>>> = Arc::new(
                selected_channels
                    .iter()
                    .map(|_| Mutex::new(CircularBuffer::new(MAX_BUFFER_SIZE))) // Create CircularBuffer for each channel
                    .collect(),
            );

            // Create the spectrum app state
            let spectrum_app = Arc::new(Mutex::new(SpectrumApp::new(selected_channels.len())));

            // Build input stream based on sample format, prioritizing F32
            let stream = match selected_config.sample_format() {
                SampleFormat::F32 => build_input_stream::<f32>(&selected_device, &stream_config, audio_buffers.clone(), spectrum_app.clone(), selected_channels.clone())?,
                SampleFormat::I16 => build_input_stream::<f32>(&selected_device, &stream_config, audio_buffers.clone(), spectrum_app.clone(), selected_channels.clone())?,
                SampleFormat::U16 => build_input_stream::<f32>(&selected_device, &stream_config, audio_buffers.clone(), spectrum_app.clone(), selected_channels.clone())?,
                SampleFormat::I32 => build_input_stream::<f32>(&selected_device, &stream_config, audio_buffers.clone(), spectrum_app.clone(), selected_channels.clone())?,
                SampleFormat::F64 => build_input_stream::<f32>(&selected_device, &stream_config, audio_buffers.clone(), spectrum_app.clone(), selected_channels.clone())?,
                _ => return Err(anyhow::anyhow!("Unsupported sample format")),
            };

            // Start the stream
            stream.play()?;

            // Launch the eframe application for plotting
            let native_options = NativeOptions::default();
            eframe::run_native(
                "Real-Time Spectrum Analyzer",
                native_options,
                Box::new(move |_cc| {
                    Box::new(MyApp {
                        spectrum: spectrum_app.clone(),  // Pass spectrum app to MyApp
                    })
                }),
            );
        } else {
            println!("Invalid configuration index. Exiting.");
        }
    } else {
        println!("Invalid device index. Exiting.");
    }

    Ok(())
}
