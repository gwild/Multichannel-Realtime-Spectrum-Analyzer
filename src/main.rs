mod audio_stream;
mod fft_analysis;
mod plot;
mod conversion;

use anyhow::{Result, anyhow};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use audio_stream::{CircularBuffer};
use std::thread;
use std::time::Duration;

const MAX_BUFFER_SIZE: usize = 4096;

fn main() -> Result<()> {
    let pa = pa::PortAudio::new()?;

    // Get a list of available devices
    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;

    println!("Available Input Devices:");
    let mut input_devices = Vec::new();
    for (i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_input_channels > 0 {
            println!("  [{}] - {}", i, info.name);
            input_devices.push(*index);
        }
    }

    // Prompt user for device selection
    print!("Enter the index of the desired device: ");
    io::stdout().flush()?;
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;

    let device_index = user_input.trim().parse::<usize>()
        .map_err(|_| anyhow!("Invalid device index"))?;
    
    if device_index >= input_devices.len() {
        return Err(anyhow!("Invalid device index. Exiting."));
    }

    let selected_device_index = input_devices[device_index];
    let selected_device_info = pa.device_info(selected_device_index)?;

    println!("Selected device: {}", selected_device_info.name);

    let default_sample_rate = selected_device_info.default_sample_rate;

    // Prompt user for channel configuration
    println!("\nEnter the number of channels to use (max {}):", selected_device_info.max_input_channels);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let num_channels: i32 = input.trim().parse()?;
    if num_channels > selected_device_info.max_input_channels {
        return Err(anyhow!("Invalid number of channels. Maximum is {}", selected_device_info.max_input_channels));
    }

    println!("\nEnter the channel numbers to use (comma-separated, e.g., 0,1):");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let selected_channels: Vec<usize> = input
        .trim()
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    println!("Selected channels: {:?}", selected_channels);

    let audio_buffers: Arc<Vec<Mutex<CircularBuffer>>> = Arc::new(
        selected_channels
            .iter()
            .map(|_| Mutex::new(CircularBuffer::new(MAX_BUFFER_SIZE)))
            .collect(),
    );

    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));

    // Stream settings
    let latency = selected_device_info.default_low_input_latency;
    let input_params = pa::StreamParameters::<f32>::new(
        selected_device_index,
        num_channels,
        true,
        latency,
    );

    let settings = pa::InputStreamSettings::new(
        input_params,
        default_sample_rate,
        MAX_BUFFER_SIZE as u32,
    );

    // Initialize and start the stream
    let mut stream = pa.open_blocking_stream(settings)?;

    stream.start()?;
    println!("Stream started with {} channels at {} Hz.", num_channels, default_sample_rate);

    // Read audio input in a loop
    loop {
        match stream.read(MAX_BUFFER_SIZE as u32) {
            Ok(buffer) => {
                for (channel, &sample) in buffer.iter().enumerate() {
                    if let Some(audio_buffer) = audio_buffers.get(channel) {
                        let mut audio_buffer = audio_buffer.lock().unwrap();
                        audio_buffer.push(sample);
                    }
                }
            }
            Err(err) => {
                eprintln!("Error reading stream: {:?}", err);
                break;
            }
        }

        // Simulate processing delay or perform actual processing here
        thread::sleep(Duration::from_millis(10));
    }

    Ok(())
}
