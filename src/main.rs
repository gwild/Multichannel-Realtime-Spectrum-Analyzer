mod audio_stream;
mod fft_analysis;
mod plot;
mod display;
mod resynth;
mod get_results;
mod presets;

use clap::Parser;
use std::sync::LazyLock;
use std::sync::OnceLock;
use eframe::egui;
use egui::ViewportBuilder;
use signal_hook::{iterator::Signals, consts::*};
use std::os::unix::process::CommandExt;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input device index
    #[arg(short = 'i', long)]
    input_device: Option<usize>,

    /// Output device index
    #[arg(short = 'o', long)]
    output_device: Option<usize>,

    /// Input sample rate in Hz (e.g. 44100, 48000, 96000)
    #[arg(long = "input-rate")]
    input_sample_rate: Option<f64>,

    /// Output sample rate in Hz (e.g. 44100, 48000, 96000)
    #[arg(long = "output-rate")]
    output_sample_rate: Option<f64>,
    
    /// Legacy sample rate parameter (deprecated, use input-rate and output-rate instead)
    #[arg(short = 'r', long)]
    sample_rate: Option<f64>,

    /// Input channels to use, comma-separated list such as "0,1"
    #[arg(short = 'c', long)]
    channels: Option<String>,

    /// Number of partials to detect per channel
    #[arg(short = 'p', long)]
    num_partials: Option<usize>,
    
    /// Enable info logging
    #[arg(long)]
    info: bool,

    /// Enable debug logging
    #[arg(long)]
    debug: bool,

    /// Internal flag used when the app relaunches itself in a new terminal. Not meant for users.
    #[arg(long = "launched-by-python", hide = true, default_value_t = false)]
    launched_by_python: bool,

    /// Internal flag from Python launcher (ignored by Rust)
    #[arg(long = "gui-ipc", hide = true, default_value_t = false)]
    gui_ipc: bool,

    /// Internal flag to enable extra logs (ignored here)
    #[arg(long = "enable-logs", hide = true, default_value_t = false)]
    enable_logs: bool,
}

pub const MIN_FREQ: f64 = 20.0;
// Store the sample rate in a thread-safe OnceLock
pub static SAMPLE_RATE: OnceLock<f64> = OnceLock::new();
pub static MAX_FREQ: LazyLock<f64> = LazyLock::new(|| {
    // Calculate max frequency based on sample rate if available
    if let Some(sample_rate) = SAMPLE_RATE.get() {
        calculate_max_freq(*sample_rate)
    } else {
        // Default to 20kHz if sample rate not yet set
        20000.0
    }
});
pub const MIN_BUFFER_SIZE: usize = 512;
pub const MAX_BUFFER_SIZE: usize = 65536;
pub const DEFAULT_BUFFER_SIZE: usize = 8192;
pub const DEFAULT_FRAMES_PER_BUFFER: u32 = 2048;
pub const FRAME_SIZES: [u32; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

use anyhow::{anyhow, Result};
use portaudio as pa;
use std::io::{self, Write};
use std::sync::{
    Arc,
    Mutex,
    RwLock,
    atomic::{AtomicBool, Ordering}
};
use audio_stream::CircularBuffer;
use eframe::NativeOptions;
use log::{info, error, warn, debug, LevelFilter};
use fern::Dispatch;
use env_logger;
use fft_analysis::{FFTConfig, MAX_SPECTROGRAPH_HISTORY, start_fft_processing};
use crate::resynth::{ResynthConfig, start_resynth_thread};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use crate::plot::SpectrographSlice;
use tokio::sync::broadcast;
use memmap2::MmapMut;
use crate::get_results::GuiParameter;
use std::sync::mpsc;
use std::fs::OpenOptions;
use std::path::Path;

#[derive(Clone)]
pub struct SharedMemory {
    pub path: String,
}

pub const DEFAULT_NUM_PARTIALS: usize = 12;

type PartialsData = Vec<Vec<(f32, f32)>>;

async fn shared_memory_updater_loop(
    mut partials_rx: broadcast::Receiver<PartialsData>,
    shared_memory_path: String,
    shutdown_flag: Arc<AtomicBool>,
) {
    debug!(target: "shared_memory", "Starting shared memory update loop for path: {}", shared_memory_path);
    let mut last_update_time = Instant::now();
    let mut update_count = 0;

    while !shutdown_flag.load(Ordering::Relaxed) {
        match partials_rx.recv().await {
            Ok(db_partials) => {
                update_count += 1;
                let now = Instant::now();
                if now.duration_since(last_update_time).as_secs() >= 5 {
                    debug!(target: "shared_memory", "GUI update stats: {} updates in last 5 seconds", update_count);
                    last_update_time = now;
                    update_count = 0;
                }

                let channel_count = db_partials.len();
                let partials_count = if !db_partials.is_empty() { db_partials[0].len() } else { 0 };
                
                debug!(target: "shared_memory", 
                    "Received dB-scaled partials: {} channels, {} partials per channel", 
                    channel_count, partials_count);

                // Serialize dB-scaled partials to bytes for Python.
                let mut bytes_to_write = Vec::<u8>::new();
                for channel_data in db_partials { // Iterate over dB-scaled partials
                    for (freq, db_amp) in channel_data { // Variable renamed for clarity
                        bytes_to_write.extend_from_slice(&freq.to_ne_bytes());
                        bytes_to_write.extend_from_slice(&db_amp.to_ne_bytes()); // Write the dB amplitude
                    }
                }

                // Write to shared memory file
                match OpenOptions::new().read(true).write(true).open(&shared_memory_path) {
                    Ok(file) => {
                        match unsafe { MmapMut::map_mut(&file) } {
                            Ok(mut mmap) => {
                                let len = bytes_to_write.len().min(mmap.len());
                                mmap[..len].copy_from_slice(&bytes_to_write[..len]);
                                debug!(target: "shared_memory", "Updated shared memory with {} bytes", len);
                                // Optional: Write a sentinel/length if protocol requires
                                // mmap.flush(); // Ensure changes are written (usually optional)
                            }
                            Err(e) => {
                                error!(target: "shared_memory", "Failed to memory map file {}: {}", shared_memory_path, e);
                            }
                        }
                    }
                    Err(e) => {
                        error!(target: "shared_memory", "Failed to open shared memory file {}: {}", shared_memory_path, e);
                    }
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!(target: "shared_memory", "Shared memory partials receiver lagged by {} messages.", n);
            }
            Err(broadcast::error::RecvError::Closed) => {
                info!(target: "shared_memory", "Partials broadcast channel closed for shared memory.");
                break; // Exit loop if channel is closed
            }
        }
    }
    info!(target: "shared_memory", "Shared memory update loop shutting down.");
}

fn calculate_max_freq(sample_rate: f64) -> f64 {
    sample_rate / 2.0
}

#[cfg(target_os = "macos")]
fn filter_realistic_sample_rates(rates: Vec<f64>) -> Vec<f64> {
    use std::process::Command;
    
    // On macOS, query the system directly for current sample rate capabilities
    info!("Querying macOS for actual audio device capabilities...");
    
    // Use system_profiler to get the actual hardware sample rate
    let output = Command::new("system_profiler")
        .arg("SPAudioDataType")
        .output();
        
    let mut native_rates = vec![44100.0, 48000.0]; // Default fallback rates
    
    if let Ok(output) = output {
        let output_str = String::from_utf8_lossy(&output.stdout);
        
        // Try to extract supported sample rates from system_profiler output
        if let Some(pos) = output_str.find("Current SampleRate:") {
            if let Some(end_pos) = output_str[pos..].find('\n') {
                let rate_str = &output_str[pos + 19..pos + end_pos].trim();
                if let Ok(rate) = rate_str.parse::<f64>() {
                    info!("Detected current hardware sample rate: {}", rate);
                    
                    // Usually hardware supports multiples/fractions of the current rate
                    native_rates = vec![
                        rate,
                        rate / 2.0,
                        rate * 2.0
                    ];
                    
                    // Most pro audio devices support 44.1k and 48k variants
                    if rate % 44100.0 == 0.0 || rate % 48000.0 == 0.0 {
                        native_rates.extend_from_slice(&[
                            44100.0,
                            48000.0,
                            88200.0,
                            96000.0
                        ]);
                    }
                }
            }
        }
    }

    // Filter the reported rates based on what we know about the hardware
    // Allow rates only if they're actually supported by PortAudio AND likely supported by hardware
    let mut realistic_rates: Vec<f64> = rates.into_iter()
        .filter(|&rate| {
            // Consider it realistic if:
            // 1. It's one of the natively detected rates, or
            // 2. It's a common audio rate <= 96kHz (most Mac hardware max)
            native_rates.contains(&rate) || 
            (rate <= 96000.0 && (
                rate == 8000.0 || 
                rate == 11025.0 || 
                rate == 16000.0 || 
                rate == 22050.0 || 
                rate == 32000.0 || 
                rate == 44100.0 || 
                rate == 48000.0 || 
                rate == 88200.0 || 
                rate == 96000.0
            ))
        })
        .collect();
    
    realistic_rates.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    if realistic_rates.is_empty() {
        // If filtering removed everything (shouldn't happen), include standard rates
        info!("No realistic rates detected, falling back to standard rates");
        vec![48000.0, 44100.0]
    } else {
        info!("Filtered to realistic hardware-supported rates: {:?}", realistic_rates);
        realistic_rates
    }
}

#[cfg(not(target_os = "macos"))]
fn filter_realistic_sample_rates(rates: Vec<f64>) -> Vec<f64> {
    // On other platforms, return original rates
    rates
}

fn get_supported_sample_rates(
    device_index: pa::DeviceIndex,
    num_channels: i32,
    pa: &pa::PortAudio,
) -> Vec<f64> {
    if let Ok(device_info) = pa.device_info(device_index) {
        info!("Detecting supported sample rates for device: {}", device_info.name);
        info!("Device default sample rate: {}", device_info.default_sample_rate);
        
        // Standard rates to check
        let sample_rates = [
            192000.0, 176400.0, 96000.0, 88200.0, 48000.0, 44100.0,
            32000.0, 22050.0, 16000.0, 11025.0, 8000.0
        ];
        
        let mut supported_rates = Vec::new();
        
        for &rate in &sample_rates {
            let params = pa::StreamParameters::<f32>::new(
                device_index,
                num_channels,
                true,
                0.1
            );
            if pa.is_input_format_supported(params, rate as f64).is_ok() {
                info!("Device supports sample rate: {} Hz", rate);
                supported_rates.push(rate);
            }
        }
        
        // Add the device's default rate if not already included
        if !supported_rates.contains(&(device_info.default_sample_rate as f64)) {
            supported_rates.push(device_info.default_sample_rate);
        }
        
        // Sort in descending order
        supported_rates.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        // Filter out unrealistic rates on macOS
        let filtered_rates = filter_realistic_sample_rates(supported_rates);
        
        info!("Device '{}' supported sample rates: {:?}", device_info.name, filtered_rates);
        filtered_rates
    } else {
        vec![44100.0]  // If we can't query the device, return a safe default
    }
}

fn get_supported_output_sample_rates(
    device_index: pa::DeviceIndex,
    pa: &pa::PortAudio,
) -> Vec<f64> {
    if let Ok(device_info) = pa.device_info(device_index) {
        info!("Detecting supported sample rates for output device: {}", device_info.name);
        info!("Device default sample rate: {}", device_info.default_sample_rate);
        
        // Standard rates to check
        let sample_rates = [
            192000.0, 176400.0, 96000.0, 88200.0, 48000.0, 44100.0,
            32000.0, 22050.0, 16000.0, 11025.0, 8000.0
        ];
        
        let mut supported_rates = Vec::new();
        
        for &rate in &sample_rates {
            let params = pa::StreamParameters::<f32>::new(
                device_index,
                device_info.max_output_channels,
                true,
                0.1
            );
            if pa.is_output_format_supported(params, rate).is_ok() {
                info!("Output device supports sample rate: {} Hz", rate);
                supported_rates.push(rate);
            }
        }
        
        // Add the device's default rate if not already included
        if !supported_rates.contains(&(device_info.default_sample_rate as f64)) {
            supported_rates.push(device_info.default_sample_rate);
        }
        
        // Sort in descending order
        supported_rates.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        // Filter out unrealistic rates on macOS
        let filtered_rates = filter_realistic_sample_rates(supported_rates);
        
        info!("Output device '{}' supported sample rates: {:?}", device_info.name, filtered_rates);
        filtered_rates
    } else {
        vec![44100.0]  // If we can't query the device, return a safe default
    }
}

fn main() -> Result<(), anyhow::Error> {
    // Parse command line arguments
    let args = Args::parse();
    
    // Initialize logging - setup both console and file logging
    setup_logging(args.debug, args.info)?;
    info!("Starting audio streaming application");

    // Set up signal handlers
    let term = Arc::new(AtomicBool::new(false));
    let term_signal = Arc::clone(&term);
    
    #[cfg(target_os = "linux")]
    {
        use signal_hook::{iterator::Signals, consts::signal::*};
        let mut signals = Signals::new(&[SIGTERM, SIGINT, SIGQUIT])?;
        let term = Arc::clone(&term);
        
        std::thread::spawn(move || {
            for sig in signals.forever() {
                info!("Received signal {}", sig);
                term.store(true, Ordering::Relaxed);
                
                // Try to read and kill the process group
                if let Ok(pgid_str) = std::fs::read_to_string("/tmp/sendaq_pgid") {
                    if let Ok(pgid) = pgid_str.trim().parse::<i32>() {
                        unsafe {
                            libc::kill(-pgid, libc::SIGTERM);
                        }
                        let _ = std::fs::remove_file("/tmp/sendaq_pgid");
                    }
                }
                
                std::process::exit(0);
            }
        });
    }

    if !args.launched_by_python {
        // Relaunch in a new terminal
        println!("Relaunching in a new terminal for consistent environment...");
        let current_exe = std::env::current_exe().expect("Failed to get current executable path");
        let current_dir = std::env::current_dir().expect("Failed to get current directory");

        // Build the command, preserving all original arguments
        let orig_args: Vec<String> = std::env::args().skip(1).collect();
        let mut cmd_str = format!("cd '{}' && {} --launched-by-python", 
            current_dir.display(), 
            current_exe.display()
        );
        
        // Add all original arguments
        for arg in orig_args {
            cmd_str.push_str(&format!(" '{}'", arg));
        }
        
        // Use platform-specific terminal commands
        #[cfg(target_os = "linux")]
        {
            cmd_str.push_str(" 2>&1 | tee debug.txt");

            // Create a new session and process group for the terminal
            let mut child = std::process::Command::new("xterm")
                .arg("-hold")  // This is sufficient to keep the terminal open
                .arg("-e")
                .arg("bash")
                .arg("-c")
                .arg(cmd_str)
                .process_group(0) // Create new process group
                .spawn()
                .expect("Failed to launch new terminal");

            // Store the process group ID for later cleanup
            match unsafe { libc::getpgid(child.id() as libc::pid_t) } {
                pgid if pgid >= 0 => {
                    std::fs::write("/tmp/sendaq_pgid", pgid.to_string())
                        .expect("Failed to write process group ID");
                }
                _ => {
                    error!("Failed to get process group ID");
                }
            }

            child.wait().expect("Failed to wait for child process");
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, use osascript to open Terminal.app with our command
            let osascript_cmd = format!(
                "tell application \"Terminal\"\nactivate\ndo script \"{}\"\nend tell",
                cmd_str.replace("\"", "\\\"")
            );

            let mut child = std::process::Command::new("osascript")
                .arg("-e")
                .arg(osascript_cmd)
                .spawn()
                .expect("Failed to launch new terminal");
            child.wait().expect("Failed to wait for child process");
        }

        #[cfg(target_os = "windows")]
        {
            cmd_str.push_str(" & pause");
            
            let mut child = std::process::Command::new("cmd")
                .arg("/C")
                .arg("start")
                .arg("cmd")
                .arg("/K")
                .arg(cmd_str)
                .spawn()
                .expect("Failed to launch new terminal");
            child.wait().expect("Failed to wait for child process");
        }
        
        return Ok(());
    }

    // Run the async part of the application
    run(&args)
}

fn run(args: &Args) -> Result<()> {
    info!("run() function entered."); // New log
    let pa = Arc::new(pa::PortAudio::new()?);
    info!("PortAudio initialized successfully in run()."); // New log

    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    info!("Initial device list collected in run(). Count: {}", devices.len()); // New log
    if devices.is_empty() {
        warn!("No devices found. Attempting to reset devices.");
        reset_audio_devices(&pa)?;
        info!("reset_audio_devices() called."); // New log
    }

    // Add a log after potentially resetting devices, before trying to list them again.
    info!("Device reset attempted if necessary. Proceeding to list devices.");

    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    info!("Final device list collected in run(). Count: {}", devices.len()); // New log
    if devices.is_empty() {
        return Err(anyhow!("No audio devices available after reset."));
    }

    info!("Retrieved list of audio devices.");
    println!("Available Input Devices:");
    
    // Create a mapping of display index to actual device index
    let mut input_devices = Vec::new();
    for (_i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_input_channels > 0 {
            println!("  [{}] - {} ({} channels)", input_devices.len(), info.name, info.max_input_channels);
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

    // Device selection: use CLI arg if provided, otherwise prompt
    let selected_device_index = if let Some(idx) = args.input_device {
        if idx >= input_devices.len() {
            return Err(anyhow!(
                "Invalid device index {} provided via --input-device. Must be 0..{}",
                idx,
                input_devices.len() - 1
            ));
        }
        idx
    } else {
    print!("Enter the index of the desired input device: ");
    io::stdout().flush()?;
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;
    let device_index = user_input
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow!("Invalid device index"))?;

    if device_index >= input_devices.len() {
            return Err(anyhow!(
                "Invalid device index. Please choose a number between 0 and {}",
                input_devices.len() - 1
            ));
    }
        device_index
    };
    let selected_input_device = input_devices[selected_device_index];
    let selected_device_info = pa.device_info(selected_input_device)?;
    info!(
        "Selected input device: {} ({} channels)",
        selected_device_info.name, selected_device_info.max_input_channels
    );

    if let Ok(device_info) = pa.device_info(selected_input_device) {
        info!("Device: {}", device_info.name);
        info!("Default sample rate: {}", device_info.default_sample_rate);
        info!("Input channels: {}", device_info.max_input_channels);
        info!("Default low latency: {}", device_info.default_low_input_latency);
        info!("Default high latency: {}", device_info.default_high_input_latency);
        
        // Try to get supported formats
        let input_params = pa::StreamParameters::<f32>::new(
            selected_input_device,
            device_info.max_input_channels,
            true,
            device_info.default_low_input_latency
        );
        
        // Test different sample formats
        for &rate in &[44100.0, 48000.0, 96000.0] {
            match pa.is_input_format_supported(input_params, rate) {
                Ok(_) => info!("Sample rate {} Hz is supported", rate),
                Err(e) => info!("Sample rate {} Hz not supported: {}", rate, e)
            }
        }
    }

    // Get supported input sample rates
    let input_sample_rates = get_supported_sample_rates(
        selected_input_device,
        selected_device_info.max_input_channels,
        &pa,
    );
    if input_sample_rates.is_empty() {
        return Err(anyhow!("No supported sample rates for the selected input device."));
    }

    // Let user select input sample rate
    let selected_input_sample_rate = if let Some(rate_cli) = args.input_sample_rate.or(args.sample_rate) {
        if !input_sample_rates.contains(&rate_cli) {
            return Err(anyhow!("Sample rate {} is not supported by selected input device", rate_cli));
        }
        rate_cli
    } else {
        println!("Supported input sample rates:");
        for (i, rate) in input_sample_rates.iter().enumerate() {
            println!("  [{}] - {} Hz", i, rate);
        }

        print!("Enter the index of the desired input sample rate: ");
        io::stdout().flush()?;
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let sample_rate_index = user_input
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow!("Invalid sample rate index"))?;

        if sample_rate_index >= input_sample_rates.len() {
            return Err(anyhow!("Invalid sample rate index."));
        }
        input_sample_rates[sample_rate_index]
    };
    info!("Selected input sample rate: {} Hz", selected_input_sample_rate);
    
    // Set the sample rate in the OnceLock for MAX_FREQ calculation
    let _ = SAMPLE_RATE.get_or_init(|| selected_input_sample_rate);
    
    // Force initialization of MAX_FREQ based on input sample rate
    let max_freq = *MAX_FREQ;
    info!("Using MAX_FREQ: {} Hz (based on input sample rate)", max_freq);

    // Now select output device
    println!("\nAvailable Output Devices:");
    let mut output_devices = Vec::new();
    for (_i, device) in devices.iter().enumerate() {
        let (index, info) = device;
        if info.max_output_channels >= 2 {  // Need at least stereo output
            println!("  [{}] - {} ({} channels)", output_devices.len(), info.name, info.max_output_channels);
            output_devices.push(*index);
        }
    }

    if output_devices.is_empty() {
        return Err(anyhow!("No stereo output devices found."));
    }

    let output_device_index = if let Some(idx) = args.output_device {
        idx
    } else {
        print!("Enter the index of the desired output device: ");
        io::stdout().flush()?;
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line)?;
        input_line
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow!("Invalid device index"))?
    };

    if output_device_index >= output_devices.len() {
        return Err(anyhow!("Invalid output device index"));
    }

    let selected_output_device = output_devices[output_device_index];
    let output_device_info = pa.device_info(selected_output_device)?;
    info!("Selected output device: {} ({} channels)", output_device_info.name, output_device_info.max_output_channels);

    // Get supported output sample rates
    let output_sample_rates = get_supported_output_sample_rates(
        selected_output_device,
        &pa,
    );
    if output_sample_rates.is_empty() {
        return Err(anyhow!("No supported sample rates for the selected output device."));
    }

    // Let user select output sample rate
    let selected_output_sample_rate = if let Some(rate_cli) = args.output_sample_rate.or(args.sample_rate) {
        if !output_sample_rates.contains(&rate_cli) {
            warn!("Note: CLI specified output sample rate {} is not supported by output device, using default", rate_cli);
            output_device_info.default_sample_rate
        } else {
            rate_cli
        }
    } else {
        println!("Supported output sample rates:");
        for (i, rate) in output_sample_rates.iter().enumerate() {
            println!("  [{}] - {} Hz", i, rate);
        }

        print!("Enter the index of the desired output sample rate: ");
        io::stdout().flush()?;
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let sample_rate_index = user_input
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow!("Invalid sample rate index"))?;

        if sample_rate_index >= output_sample_rates.len() {
            return Err(anyhow!("Invalid sample rate index."));
        }
        output_sample_rates[sample_rate_index]
    };
    info!("Selected output sample rate: {} Hz", selected_output_sample_rate);
    
    // Log the difference between input and output sample rates if they differ
    if selected_input_sample_rate != selected_output_sample_rate {
        info!("Note: Input sample rate ({} Hz) differs from output sample rate ({} Hz)", 
              selected_input_sample_rate, selected_output_sample_rate);
        info!("Analysis will use full input sample rate range, but resynthesis will be limited to output capabilities");
    }

    let selected_channels: Vec<usize> = if let Some(ref ch_str) = args.channels {
        ch_str
            .split(',')
            .filter_map(|s| s.parse::<usize>().ok())
            .filter(|&ch| ch < selected_device_info.max_input_channels as usize)
            .collect()
    } else {
    println!(
        "Available channels: 0 to {}",
        selected_device_info.max_input_channels - 1
    );
    println!("Enter channels to use (comma-separated, e.g., 0,1): ");
        let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;
        user_input
        .trim()
        .split(',')
        .filter_map(|s| s.parse::<usize>().ok())
        .filter(|&ch| ch < selected_device_info.max_input_channels as usize)
            .collect()
    };

    if selected_channels.is_empty() {
        return Err(anyhow!("No valid channels selected."));
    }
    info!("Selected channels: {:?}", selected_channels);

    // Add prompt for number of partials here
    let num_partials = if let Some(p) = args.num_partials {
        p.max(1)
    } else {
    println!("Enter number of partials to detect per channel (default is {}): ", DEFAULT_NUM_PARTIALS);
        let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;
        if user_input.trim().is_empty() {
        DEFAULT_NUM_PARTIALS
    } else {
        user_input
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow!("Invalid number of partials"))?
                .max(1)
        }
    };
    info!("Using {} partials per channel", num_partials);

    // --- Core Application State Setup ---
    let buffer_size = Arc::new(Mutex::new(DEFAULT_BUFFER_SIZE));
    let audio_buffer = Arc::new(RwLock::new(CircularBuffer::new(
        DEFAULT_BUFFER_SIZE,
        selected_channels.len()
    )));
    let spectrum_app = Arc::new(Mutex::new(plot::SpectrumApp::new(selected_channels.len())));
    
    let mut config = FFTConfig::default();
    // Override only what needs to be different from defaults
    config.num_channels = selected_channels.len();
    config.frames_per_buffer = if cfg!(target_os = "linux") {
        2048u32  // Larger buffer for Linux stability
    } else {
        match selected_input_sample_rate as u32 {
            48000 => 1024u32,   // Increased for better frequency resolution
            44100 => 1024u32,   // Increased for better frequency resolution
            96000 => 2048u32,   // Increased for higher frequency analysis
            192000 => 4096u32,  // Added option for very high sample rates
            _ => {
                let mut base_size = 1024u32;  // Increased base size
                while base_size * 2 <= (selected_input_sample_rate / 50.0) as u32 {
                    base_size *= 2;
                }
                base_size
            }
        }
    };
    config.num_partials = num_partials;

    let fft_config = Arc::new(Mutex::new(config));

    // --- GUI setup ---
    let native_options = eframe::NativeOptions {
        viewport: ViewportBuilder::default()
            .with_inner_size([1024.0, 440.0]),
        ..Default::default()
    };

    // Create the MPSC channel for GUI parameter updates
    let (gui_param_tx, gui_param_rx) = mpsc::channel::<GuiParameter>();

    // Create the MPSC channel for instant gain updates
    let (gain_update_tx, gain_update_rx) = mpsc::channel::<f32>();
    
    // Create the ResynthConfig instance
    let resynth_config = Arc::new(Mutex::new(ResynthConfig {
        gain: 0.5,
        freq_scale: 1.0,
        update_rate: 1.0,
        needs_restart: Arc::new(AtomicBool::new(false)),
        needs_stop: Arc::new(AtomicBool::new(false)),
        output_sample_rate: Arc::new(Mutex::new(selected_output_sample_rate)),
    }));

    // Get appropriate shared memory directory based on platform
    #[cfg(target_os = "linux")]
    let shm_dir = "/dev/shm";
    
    #[cfg(target_os = "macos")]
    let shm_dir = "/tmp";
    
    #[cfg(target_os = "windows")]
    let shm_dir = std::env::temp_dir().to_str().unwrap_or("C:\\Temp");

    // Write control file for external processes
    let control_path = format!("{}/audio_control", shm_dir);
    let mut control_file = std::fs::File::create(&control_path)?;
    writeln!(control_file, "{}\n{}\n{}", std::process::id(), selected_channels.len(), num_partials)?;

    // Shared state for shutdown and timers
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_complete = Arc::new(AtomicBool::new(false));
    let stream_ready = Arc::new(AtomicBool::new(false));
    let running = Arc::new(AtomicBool::new(true));
    let start_time = Arc::new(Instant::now());

    // Before starting the FFT thread, initialize spectrograph history with fixed capacity
    let spectrograph_history = Arc::new(Mutex::new(VecDeque::<SpectrographSlice>::with_capacity(MAX_SPECTROGRAPH_HISTORY)));
    
    // Initialize shared memory without mutex BEFORE threads start
    let shared_partials = {
        let shmem_name = "audio_peaks";
        let shared_memory_path = format!("{}/{}", shm_dir, shmem_name);
        let file = std::fs::File::create(&shared_memory_path)?;
        file.set_len(4 * 1024 * 1024)?;
        info!("Shared memory initialized at {} for internal use", shared_memory_path);
        Some(SharedMemory {
            path: shared_memory_path,
        })
    };
    
    // --- Thread variable setup ---
    let shutdown_flag_audio = Arc::clone(&shutdown_flag);
    let shutdown_flag_fft = Arc::clone(&shutdown_flag);
    let shutdown_flag_resynth = Arc::clone(&shutdown_flag);
    
    let main_buffer_audio = Arc::clone(&audio_buffer);
    let main_buffer_fft = Arc::clone(&audio_buffer);

    let buffer_size_audio = Arc::clone(&buffer_size);

    let fft_config_audio = Arc::clone(&fft_config);
    let fft_config_gui = Arc::clone(&fft_config);
    let fft_config_fft = Arc::clone(&fft_config);

    let resynth_config_audio = Arc::clone(&resynth_config);
    let resynth_config_resynth = Arc::clone(&resynth_config);
    let resynth_config_gui = Arc::clone(&resynth_config);

    let stream_ready_audio = Arc::clone(&stream_ready);
    let stream_ready_fft = Arc::clone(&stream_ready);
    
    let selected_channels_audio = selected_channels.clone();
    let selected_channels_fft = selected_channels.clone();
    let num_input_channels_resynth = selected_channels.len();
    let num_partials_resynth = num_partials;

    let (partials_tx, _) = broadcast::channel::<PartialsData>(16); // Receiver is in resynth and GUI
    let partials_tx_fft = partials_tx.clone();
    let partials_rx_resynth = partials_tx.subscribe();
    let partials_rx_gui = partials_tx.subscribe();

    let (gui_param_tx_gui, gui_param_rx_resynth) = mpsc::channel::<GuiParameter>();
    let (gain_update_tx_gui, gain_update_rx_resynth) = mpsc::channel::<f32>();
    
    // --- Start Threads ---
    
    // Audio Input Thread
    let audio_thread_args = (
        Arc::clone(&running),
        Arc::clone(&main_buffer_audio),
        selected_channels_audio.clone(),
        selected_input_sample_rate,
        Arc::clone(&buffer_size_audio),
        selected_input_device,
        Arc::clone(&shutdown_flag_audio),
        Arc::clone(&stream_ready_audio),
        Arc::clone(&fft_config_audio),
        Arc::clone(&resynth_config_audio),
    );
    let _audio_thread = thread::spawn(move || {
        audio_stream::start_sampling_thread(
            audio_thread_args.0,
            audio_thread_args.1,
            audio_thread_args.2,
            audio_thread_args.3,
            audio_thread_args.4,
            audio_thread_args.5,
            audio_thread_args.6,
            audio_thread_args.7,
            audio_thread_args.8,
            audio_thread_args.9,
        );
    });

    // FFT Analysis Thread
    let fft_thread_args = (
        Arc::clone(&main_buffer_fft),
        Arc::clone(&fft_config_fft),
        Arc::clone(&spectrum_app),
        selected_channels_fft.clone(),
        selected_input_sample_rate as u32,
        Arc::clone(&shutdown_flag_fft),
        partials_tx_fft,
        Some(Arc::clone(&spectrograph_history)),
        Some(Arc::clone(&start_time)),
    );
    let _fft_thread = thread::spawn(move || {
        start_fft_processing(
            fft_thread_args.0,
            fft_thread_args.1,
            fft_thread_args.2,
            fft_thread_args.3,
            fft_thread_args.4,
            fft_thread_args.5,
            fft_thread_args.6,
            fft_thread_args.7,
            fft_thread_args.8,
        );
    });

    // Resynth Thread
    let resynth_thread_args = (
        Arc::clone(&resynth_config_resynth),
        selected_output_device,
        selected_output_sample_rate,  // Make sure we use the output sample rate here
        Arc::clone(&shutdown_flag_resynth),
        partials_rx_resynth,
        num_input_channels_resynth,
        num_partials_resynth,
        gui_param_rx_resynth,
        gain_update_rx_resynth,
    );
    let _resynth_thread = std::thread::spawn({
        move || {
            start_resynth_thread(
                resynth_thread_args.0,
                resynth_thread_args.1,
                resynth_thread_args.2,  // This is selected_output_sample_rate
                resynth_thread_args.3,
                resynth_thread_args.4,
                resynth_thread_args.5,
                resynth_thread_args.6,
                resynth_thread_args.7,
                resynth_thread_args.8,
            );
        }
    });

    // --- Start GUI ---
    info!("Starting GUI...");
    // Before creating the app_creator, create clones of all variables needed for the GUI
    let main_buffer_gui = Arc::clone(&audio_buffer);
    let shutdown_flag_gui = Arc::clone(&shutdown_flag);
    
    // Create the GUI app directly
    let app = plot::MyApp::new(
        spectrum_app,
        fft_config_gui,
        buffer_size,
        main_buffer_gui,
        resynth_config_gui,
        shutdown_flag_gui,
        spectrograph_history,
        start_time,
        selected_input_sample_rate,
        partials_rx_gui,
        gui_param_tx_gui,
        gain_update_tx_gui,
    );
    
    // Spawn SharedMemory update thread
    if let Some(shared_mem_info) = shared_partials {
        let shared_memory_partials_rx = partials_tx.subscribe();
        let sm_shutdown_flag = Arc::clone(&shutdown_flag);
        let shared_memory_path = shared_mem_info.path.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(shared_memory_updater_loop(shared_memory_partials_rx, shared_memory_path, sm_shutdown_flag));
        });
    } else {
        warn!("SharedMemory struct not initialized, skipping shared memory update thread.");
    }
    
    let native_options = NativeOptions {
        viewport: ViewportBuilder::default()
            .with_inner_size([1024.0, 440.0]),
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
    // shutdown_flag is moved to the GUI thread, so we can't use it here.
    // The GUI's on_close_event will handle setting the flag.

    // Wait for threads to finish with timeout
    let timeout = Duration::from_secs(5);
    let start_wait = std::time::Instant::now();

    while start_wait.elapsed() < timeout {
        if shutdown_complete.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }

    // Clean up PortAudio
    if let Ok(pa) = pa::PortAudio::new() {
        if let Err(e) = pa.terminate() {
            warn!("Error terminating PortAudio: {}", e);
        }
    }

    info!("Application shutdown complete.");
    Ok(())
}

fn find_compatible_sample_rates(
    input_device_index: pa::DeviceIndex,
    input_channels: i32,
    output_device_index: pa::DeviceIndex,
    pa: &pa::PortAudio,
) -> Vec<f64> {
    let input_rates = get_supported_sample_rates(input_device_index, input_channels, pa);
    let output_rates = get_supported_output_sample_rates(output_device_index, pa);
    
    // Find common sample rates
    let compatible = input_rates.into_iter()
        .filter(|rate| output_rates.contains(rate))
        .collect();
    
    info!("Compatible sample rates for both input and output: {:?}", compatible);
    
    compatible
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

#[allow(dead_code)]
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

// Add this new function to set up logging to both console and file
fn setup_logging(debug_mode: bool, info_mode: bool) -> Result<(), anyhow::Error> {
    // Set the log level based on debug flag
    let log_level = if debug_mode {
        LevelFilter::Debug
    } else if info_mode {
        LevelFilter::Info
    } else {
        LevelFilter::Warn
    };

    // Get the current executable's directory to place logs alongside the binary
    let exe_dir = std::env::current_exe()?
        .parent()
        .ok_or_else(|| anyhow!("Failed to get executable directory"))?
        .to_path_buf();
    
    // Create log directory if it doesn't exist
    let log_dir = exe_dir.join("logs");
    if !log_dir.exists() {
        std::fs::create_dir_all(&log_dir)?;
    }

    // Generate log filename with timestamp
    let now = chrono::Local::now();
    let log_filename = log_dir.join(format!("debug_{}.log", now.format("%Y%m%d_%H%M%S")));
    
    println!("Debug logging enabled - writing to {}", log_filename.display());

    // Make sure we remove any existing loggers first
    log::set_max_level(LevelFilter::Off);
    
    // Configure logging to output to both terminal and file with detailed formatting
    Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}][{}][{}] {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(log_level)
        // Terminal output
        .chain(std::io::stdout())
        // File output with explicit options to ensure writing works
        .chain(
            OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(&log_filename)?
        )
        .apply()?;

    // Log a test message to verify logging is working
    info!("Logging initialized: console and file output to {}", log_filename.display());
    debug!("Debug logging test message - if you see this, debug logging is working");
    
    Ok(())
}

