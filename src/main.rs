mod audio_stream;
mod fft_analysis;
mod plot;
mod conversion;

use anyhow::{Result, anyhow};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use audio_stream::CircularBuffer;
use std::thread;
use std::time::Duration;
use eframe::NativeOptions;

const MAX_BUFFER_SIZE: usize = 512;

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

    // Print supported configurations (assuming all configurations use default sample rate)
    println!("Supported input configs:");
    println!("  Channels: 1 - {}", selected_device_info.max_input_channels);
    println!("  Sample Rate: {} Hz", default_sample_rate);

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

    let max_channel = num_channels as usize - 1;
    for &channel in &selected_channels {
        if channel > max_channel {
            return Err(anyhow!("Invalid channel selected. Maximum channel is {}", max_channel));
        }
    }

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

    // Launch the eframe application for plotting
    let native_options = NativeOptions {
        initial_window_size: Some(eframe::epaint::Vec2::new(960.0, 420.0)), // Set window size to 960x420
        ..Default::default()
    };

    let spectrum_thread = thread::spawn({
        let native_options = native_options.clone(); // Clone the options to move into the thread
        let spectrum_app_clone = Arc::clone(&spectrum_app); // Clone the Arc for use in the thread
        move || {
            // Ensure that the closure does not capture non-Send types
            let result = plot::run_native(
                "Real-Time Spectrum Analyzer",
                native_options,
                Box::new(move |cc| {
                    // Your application logic here
                    Box::new(plot::MyApp::new(spectrum_app_clone)) // Pass the Arc<Mutex<SpectrumApp>> as a Box<dyn App>
                }),
            );
            result.expect("Error launching application");
        }
    });

    // Read audio input in a loop
    loop {
        match stream.read(MAX_BUFFER_SIZE as u32) {
            Ok(buffer) => {
                for (i, sample) in buffer.iter().enumerate() {
                    let channel = i % num_channels as usize;
                    if let Some(audio_buffer) = audio_buffers.get(channel) {
                        let mut audio_buffer = audio_buffer.lock().unwrap();
                        audio_buffer.push(*sample);
                    }
                }
            }
            Err(err) => {
                eprintln!("Error reading stream: {:?}", err);
                break;
            }
        }

        thread::sleep(Duration::from_millis(10));
    }

    spectrum_thread.join().unwrap();

    Ok(())
}
