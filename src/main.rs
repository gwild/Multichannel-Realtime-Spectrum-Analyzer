// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: Do not remove or rename any modules listed here, or their usage, without explicit permission.
mod audio_stream;
mod fft_analysis;
mod plot;

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: These imports must remain unless permission is explicitly granted to change them.
use anyhow::{anyhow, Result};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{
    Arc,
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering}
};
use std::sync::mpsc;
use audio_stream::{CircularBuffer, build_input_stream};
use eframe::NativeOptions;
use log::{info, error, warn};
use env_logger;
// Added to conditionally set environment variable for logs:
use std::env;

use fft_analysis::FFTConfig;

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: This constant must remain as is, unless permission is requested to modify or remove it.
const MAX_BUFFER_SIZE: usize = 4096;

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The `main` function logic, including `env_logger::init()`, is protected.
fn main() {
    // *** ADDED LINES: Conditionally disable logs unless "--enable-logs" is passed.
    // We do NOT remove or rename the existing env_logger::init() line.
    {
        let args: Vec<String> = env::args().collect();
        // If user did NOT pass --enable-logs, we set RUST_LOG=off to suppress logs
        if !args.iter().any(|arg| arg == "--enable-logs") {
            env::set_var("RUST_LOG", "off");
        }
    }

    // Reminder: The line below is protected. Must remain. It will respect RUST_LOG if set above.
    env_logger::init();

    // Reminder: The error handling for run() is protected.
    if let Err(e) = run() {
        error!("Application encountered an error: {:?}", e);
        std::process::exit(1);
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: The `run` function logic must remain unless permission is explicitly granted.
fn run() -> Result<()> {
    let pa = Arc::new(pa::PortAudio::new()?);
    info!("PortAudio initialized.");

    // Reset devices if no devices are detected
    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    if devices.is_empty() {
        warn!("No devices found. Attempting to reset devices.");
        reset_audio_devices(&pa)?;
    }

    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    if devices.is_empty() {
        return Err(anyhow!("No audio devices available after reset."));
    }

    info!("Retrieved list of audio devices.");
    println!("Available Input Devices:");
    let mut input_devices = Vec::new();

    for (i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_input_channels > 0 {
            println!("  [{}] - {} ({} channels)", i, info.name, info.max_input_channels);

            // Check if the device is ready by attempting to open and close it
            if ensure_audio_device_ready(&pa, *index) {
                input_devices.push(*index);
            } else {
                error!("Device {} is not ready for use.", info.name);
            }
        }
    }

    if input_devices.is_empty() {
        return Err(anyhow!("No input audio devices found."));
    }

    print!("Enter the index of the desired device: ");
    io::stdout().flush()?;
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;
    let device_index = user_input
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow!("Invalid device index"))?;

    if device_index >= input_devices.len() {
        return Err(anyhow!("Invalid device index."));
    }

    let selected_device_index = input_devices[device_index];
    let selected_device_info = pa.device_info(selected_device_index)?;
    info!("Selected device: {} ({} channels)", selected_device_info.name, selected_device_info.max_input_channels);

    let supported_sample_rates = get_supported_sample_rates(selected_device_index, selected_device_info.max_input_channels, &pa);
    if supported_sample_rates.is_empty() {
        return Err(anyhow!("No supported sample rates for the selected device."));
    }

    println!("Supported sample rates:");
    for (i, rate) in supported_sample_rates.iter().enumerate() {
        println!("  [{}] - {} Hz", i, rate);
    }

    print!("Enter the index of the desired sample rate: ");
    io::stdout().flush()?;
    user_input.clear();
    io::stdin().read_line(&mut user_input)?;
    let sample_rate_index = user_input.trim().parse::<usize>()?;

    if sample_rate_index >= supported_sample_rates.len() {
        return Err(anyhow!("Invalid sample rate index."));
    }
    let selected_sample_rate = supported_sample_rates[sample_rate_index];

    println!("Available channels: 0 to {}", selected_device_info.max_input_channels - 1);
    println!("Enter channels to use (comma-separated, e.g., 0,1): ");
    user_input.clear();
    io::stdin().read_line(&mut user_input)?;
    let selected_channels: Vec<usize> = user_input
        .trim()
        .split(',')
        .filter_map(|s| s.parse::<usize>().ok())
        .filter(|&ch| ch < selected_device_info.max_input_channels as usize)
        .collect();

    if selected_channels.is_empty() {
        return Err(anyhow!("No valid channels selected."));
    }
    info!("Selected channels: {:?}", selected_channels);

    let buffer_size = Arc::new(Mutex::new(MAX_BUFFER_SIZE));
    let audio_buffers: Arc<Vec<Mutex<CircularBuffer>>> = Arc::new(
        selected_channels
            .iter()
            .map(|_| Mutex::new(CircularBuffer::new(*buffer_size.lock().unwrap())))
            .collect(),
    );

    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));
    let fft_config = Arc::new(Mutex::new(FFTConfig {
        min_frequency: 20.0,
        max_frequency: 20000.0,
        db_threshold: -32.0,
    }));

    let running = Arc::new(AtomicBool::new(true));
    let sampling_done = Arc::new(AtomicBool::new(false));
    let last_sample_timestamp = Arc::new(AtomicUsize::new(0)); // Use timestamp to track last sample

    // Start the sampling thread
    let worker_thread = {
        let pa_clone = Arc::clone(&pa);
        let audio_buffers_clone = Arc::clone(&audio_buffers);
        let running_clone = Arc::clone(&running);
        let sampling_done_clone = Arc::clone(&sampling_done);
        let last_sample_timestamp_clone = Arc::<AtomicUsize>::clone(&last_sample_timestamp);
        let selected_channels_clone = selected_channels.clone();
        let spectrum_app_clone = Arc::clone(&spectrum_app);
        let fft_config_clone = Arc::clone(&fft_config);

        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(|| {
                if let Ok(mut stream) = build_input_stream(
                    &pa_clone,
                    selected_device_index,
                    selected_device_info.max_input_channels as i32,
                    selected_sample_rate,
                    audio_buffers_clone,
                    spectrum_app_clone,
                    selected_channels_clone,
                    fft_config_clone,
                ) {
                    stream.start().expect("Failed to start audio stream.");
                    sampling_done_clone.store(true, Ordering::SeqCst);

                    while running_clone.load(Ordering::SeqCst) {
                        // Update the timestamp on each iteration
                        last_sample_timestamp_clone.store(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs() as usize,
                            Ordering::SeqCst,
                        );
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }

                    stream.stop().expect("Failed to stop stream.");
                    stream.close().expect("Failed to close stream.");
                } else {
                    error!("Failed to build audio stream.");
                }
            });

            if result.is_err() {
                error!("Thread panicked while running the audio stream.");
            }
        })
    };

    // GUI and thread monitoring loop
    tokio::task::block_in_place(|| {
        let monitoring_interval = std::time::Duration::from_secs(1);
        let mut last_reported_time = 0;

        plot::run_native(
            "Real-Time Spectrum Analyzer",
            NativeOptions {
                initial_window_size: Some(eframe::epaint::Vec2::new(1024.0, 420.0)),
                ..Default::default()
            },
            Box::new(move |_cc| {
                Box::new(plot::MyApp::new(
                    spectrum_app.clone(),
                    fft_config.clone(),
                    buffer_size.clone(),
                    audio_buffers.clone(),
                ))
            }),
        );

        // Monitoring thread health
        while running.load(Ordering::SeqCst) {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let last_sample_time = last_sample_timestamp.load(Ordering::SeqCst);
            if current_time as usize - last_sample_time > 2 && last_sample_time > last_reported_time {
                error!("Sampling thread has not updated in over 2 seconds!");
                last_reported_time = last_sample_time;
            }

            std::thread::sleep(monitoring_interval);
        }
    });

    // Terminate sampling thread
    running.store(false, Ordering::SeqCst);
    if let Err(e) = worker_thread.join() {
        error!("Worker thread failed to join: {:?}", e);
    }

    // Terminate PortAudio
    match Arc::try_unwrap(Arc::clone(&pa)) {
        Ok(pa_inner) => {
            pa_inner.terminate()?;
        }
        Err(_) => {
            warn!("Unable to terminate PortAudio directly; multiple references exist.");
        }
    }

    Ok(())
}



// THE FUNCTION WAS NOT REMOVED
// ONLY A FUCKING DEVIANT WOULD REMOVE THIS FUNCTION
// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Reminder: get_supported_sample_rates logic is protected, including the array of common_rates.
fn get_supported_sample_rates(
    device_index: pa::DeviceIndex,
    num_channels: i32,
    pa: &pa::PortAudio,
) -> Vec<f64> {
    let common_rates = [8000.0, 16000.0, 22050.0, 44100.0, 48000.0, 96000.0, 192000.0];
    common_rates
        .iter()
        .cloned()
        .filter(|&rate| {
            let params = pa::StreamParameters::<f32>::new(device_index, num_channels, true, 0.0);
            pa.is_input_format_supported(params, rate).is_ok()
        })
        .collect()
}

fn reset_audio_devices(pa: &Arc<pa::PortAudio>) -> Result<()> {
    match Arc::try_unwrap(Arc::clone(pa)) {
        Ok(pa_inner) => {
            pa_inner.terminate()?;
        }
        Err(_) => {
            warn!("Unable to terminate PortAudio directly; multiple references exist.");
        }
    }
    let _ = pa::PortAudio::new()?;
    info!("PortAudio reset successful.");
    Ok(())
}

// This function ensures audio devices are ready by opening and immediately closing them.
fn ensure_audio_device_ready(pa: &pa::PortAudio, device_index: pa::DeviceIndex) -> bool {
    let params = pa::StreamParameters::<f32>::new(device_index, 1, true, 0.0);
    if let Ok(mut stream) = pa.open_non_blocking_stream(
        pa::InputStreamSettings::new(params, 44100.0, 256),
        |_args| pa::Continue,
    ) {
        stream.close().is_ok()
    } else {
        false
    }
}
