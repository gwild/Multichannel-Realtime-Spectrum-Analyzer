mod audio_stream;
mod fft_analysis;
mod plot;
mod utils;

use anyhow::{anyhow, Result};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{
    Arc,
    Mutex,
    RwLock,
    atomic::{AtomicBool, Ordering}
};
use audio_stream::{CircularBuffer, start_sampling_thread};
use eframe::NativeOptions;
use log::{info, error, warn};
use env_logger;
use std::env;
use fft_analysis::FFTConfig;
use utils::{MIN_FREQ, MAX_FREQ, calculate_optimal_buffer_size};

fn main() {
    // Only set up logging if --enable-logs flag is present
    if std::env::args().any(|arg| arg == "--enable-logs") {
        // Only set RUST_LOG if it's not already set
        if std::env::var("RUST_LOG").is_err() {
            env::set_var("RUST_LOG", "info");
        }
        env_logger::init();
        info!("Application starting...");
    }

    if let Err(e) = run() {
        if std::env::args().any(|arg| arg == "--enable-logs") {
            error!("Application encountered an error: {:?}", e);
        } else {
            eprintln!("Error: {:?}", e);
        }
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let pa = Arc::new(pa::PortAudio::new()?);
    info!("PortAudio initialized.");

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
            if ensure_audio_device_ready(&pa, *index) {
                input_devices.push(*index);
            } else {
                warn!("Device {} is not ready for use.", info.name);
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
    info!(
        "Selected device: {} ({} channels)",
        selected_device_info.name, selected_device_info.max_input_channels
    );

    let supported_sample_rates = get_supported_sample_rates(
        selected_device_index,
        selected_device_info.max_input_channels,
        &pa,
    );
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
    let sample_rate_index = user_input
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow!("Invalid sample rate index"))?;

    if sample_rate_index >= supported_sample_rates.len() {
        return Err(anyhow!("Invalid sample rate index."));
    }
    let selected_sample_rate = supported_sample_rates[sample_rate_index];
    info!("Selected sample rate: {} Hz", selected_sample_rate);

    println!(
        "Available channels: 0 to {}",
        selected_device_info.max_input_channels - 1
    );
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

    let buffer_size = Arc::new(Mutex::new(calculate_optimal_buffer_size(selected_sample_rate)));
    let audio_buffer = Arc::new(RwLock::new(CircularBuffer::new(
        *buffer_size.lock().unwrap(),
        selected_channels.len()
    )));
    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));
    let fft_config = Arc::new(Mutex::new(FFTConfig {
        min_frequency: MIN_FREQ,
        max_frequency: MAX_FREQ,
        db_threshold: -24.0,
        num_channels: selected_channels.len(),
        averaging_factor: 0.8,
        frames_per_buffer: 512,
    }));

    let running = Arc::new(AtomicBool::new(false));
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let stream_ready = Arc::new(AtomicBool::new(false));

    let audio_buffer_clone = Arc::clone(&audio_buffer);
    let selected_channels_clone = selected_channels.clone();
    let buffer_size_clone = Arc::clone(&buffer_size);
    let shutdown_flag_audio = Arc::clone(&shutdown_flag);
    let stream_ready_audio = Arc::clone(&stream_ready);

    // Start audio thread
    info!("Starting audio sampling thread...");
    let audio_thread = std::thread::spawn(move || {
        start_sampling_thread(
            running,
            audio_buffer_clone,
            selected_channels_clone,
            selected_sample_rate,
            buffer_size_clone,
            selected_device_index,
            shutdown_flag_audio,
            stream_ready_audio,
        );
    });

    // Start FFT processing thread only after stream is ready
    info!("Waiting for audio stream to initialize...");
    let stream_ready_fft = Arc::clone(&stream_ready);
    while !stream_ready_fft.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    info!("Starting FFT processing...");
    let fft_thread = std::thread::spawn({
        let audio_buffer = Arc::clone(&audio_buffer);
        let fft_config = Arc::clone(&fft_config);
        let spectrum_app = Arc::clone(&spectrum_app);
        let selected_channels = selected_channels.clone();
        let shutdown_flag_fft = Arc::clone(&shutdown_flag);
        move || {
            fft_analysis::start_fft_processing(
                audio_buffer,
                fft_config,
                spectrum_app,
                selected_channels,
                selected_sample_rate as u32,
                shutdown_flag_fft,
            );
        }
    });

    // Start GUI
    info!("Starting GUI...");
    let app = plot::MyApp::new(
        spectrum_app.clone(),
        fft_config.clone(),
        buffer_size.clone(),
        audio_buffer.clone(),
        shutdown_flag.clone(),
    );

    let native_options = NativeOptions {
        initial_window_size: Some(egui::vec2(1024.0, 420.0)),
        vsync: true,
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "Real-Time Spectrum Analyzer",
        native_options,
        Box::new(|_cc| Box::new(app)),
    ) {
        error!("GUI error: {}", e);
    }

    // Set shutdown flag to stop processing threads
    info!("Setting shutdown flag...");
    shutdown_flag.store(true, Ordering::SeqCst);

    // Wait for threads to finish
    if let Ok(_) = audio_thread.join() {
        info!("Audio thread terminated successfully");
    } else {
        warn!("Audio thread may not have terminated cleanly");
    }

    if let Ok(_) = fft_thread.join() {
        info!("FFT thread terminated successfully");
    } else {
        warn!("FFT thread may not have terminated cleanly");
    }

    Ok(())
}

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

fn ensure_audio_device_ready(pa: &pa::PortAudio, device_index: pa::DeviceIndex) -> bool {
    let params = pa::StreamParameters::<f32>::new(device_index, 1, true, 0.0);
    if let Ok(mut stream) = pa.open_non_blocking_stream(
        pa::InputStreamSettings::new(params, 48000.0f64, 512),
        |_args| pa::Continue,
    ) {
        stream.close().is_ok()
    } else {
        false
    }
}

fn test_audio_input(pa: &pa::PortAudio, device_index: pa::DeviceIndex, channels: i32) -> Result<bool> {
    info!("Testing audio input for device...");
    
    let latency = pa.device_info(device_index)?.default_low_input_latency;
    let input_params = pa::StreamParameters::new(device_index, channels, true, latency);
    
    // Create a test buffer
    let mut test_buffer = vec![0.0f32; 1024 * channels as usize];
    
    // Create a blocking stream for testing
    let mut stream = pa.open_blocking_stream(
        pa::InputStreamSettings::new(input_params, 44100.0f64, 1024)
    )?;
    
    stream.start()?;
    info!("Reading test audio data...");
    
    // Try to read some data
    match stream.read(1024) {
        Ok(data) => {
            test_buffer.copy_from_slice(data);
            let non_zero = test_buffer.iter().filter(|&&x| x != 0.0).count();
            info!("Test read - Buffer size: {}, Non-zero samples: {}", test_buffer.len(), non_zero);
            if non_zero > 0 {
                info!("First few non-zero samples: {:?}", 
                    test_buffer.iter()
                        .filter(|&&x| x != 0.0)
                        .take(5)
                        .collect::<Vec<_>>());
            }
            stream.stop()?;
            Ok(non_zero > 0)
        },
        Err(e) => {
            stream.stop()?;
            Err(anyhow!("Failed to read audio data: {}", e))
        }
    }
}
