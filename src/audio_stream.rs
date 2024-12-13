use anyhow::{anyhow, Result};
use portaudio as pa;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, Sender};
use crate::fft_analysis::{compute_spectrum, NUM_PARTIALS};
use crate::plot::SpectrumApp;
use portaudio::stream::InputCallbackArgs;
use log::{info, error};

/// Circular buffer implementation for storing audio samples.
pub struct CircularBuffer {
    buffer: Vec<f32>,
    head: usize,
    size: usize,
}

impl CircularBuffer {
    /// Creates a new `CircularBuffer` with the specified size.
    pub fn new(size: usize) -> Self {
        CircularBuffer {
            buffer: vec![0.0; size],
            head: 0,
            size,
        }
    }

    /// Pushes a new sample into the buffer.
    pub fn push(&mut self, value: f32) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.size;
    }

    /// Retrieves the latest `count` samples.
    pub fn get_latest(&self, count: usize) -> Vec<f32> {
        let start = if count > self.size {
            0
        } else {
            (self.head + self.size - count) % self.size
        };
        let mut data = Vec::with_capacity(count);
        data.extend_from_slice(&self.buffer[start..]);
        if start > self.head {
            data.extend_from_slice(&self.buffer[..self.head]);
        }
        data
    }
}

/// Builds and configures the audio input stream using PortAudio.
pub fn build_input_stream(
    pa: &pa::PortAudio,
    device_index: pa::DeviceIndex,
    num_channels: i32,
    sample_rate: f64,
    audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
    sender: Sender<Vec<f32>>,
) -> Result<pa::Stream<pa::NonBlocking, pa::Input<f32>>> {
    if selected_channels.is_empty() {
        return Err(anyhow!("No channels selected"));
    }

    // Retrieve device information to get latency settings
    let device_info = pa.device_info(device_index)?;
    let latency = device_info.default_low_input_latency;

    // Configure stream parameters
    let input_params = pa::StreamParameters::<f32>::new(device_index, num_channels, true, latency);

    // Define stream settings with the specified sample rate and frames per buffer
    let settings = pa::InputStreamSettings::new(input_params, sample_rate, 256);

    info!("Opening non-blocking PortAudio stream.");

    // Create the non-blocking stream with a callback to process incoming audio data
    let stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            if let Err(e) = sender.send(args.buffer.to_vec()) {
                error!("Failed to send audio buffer: {}", e);
                return pa::Complete;
            }

            if args.flags.contains(pa::StreamCallbackFlags::INPUT_OVERFLOW) {
                error!("Input overflow detected.");
            }
            pa::Continue
        },
    )?;

    info!("PortAudio stream opened successfully.");

    Ok(stream)
}

/// Processes incoming audio samples and updates the spectrum analyzer.
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
            if let Some(buffer_index) = selected_channels.iter().position(|&ch| ch == channel) {
                if let Ok(mut buffer) = audio_buffers[buffer_index].lock() {
                    buffer.push(sample);
                } else {
                    error!("Failed to lock buffer for channel {}", buffer_index);
                }
            }
        }
    }

    // Initialize partials_results with zeroes for all channels
    let mut partials_results = vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

    // Compute spectrum for each selected channel
    for (i, &channel) in selected_channels.iter().enumerate() {
        if let Ok(buffer) = audio_buffers[i].lock() {
            let audio_data = buffer.get_latest(1024); // Adjust size as needed for FFT

            if audio_data.len() >= 1024 { // Ensure sufficient data for FFT
                let computed_partials = compute_spectrum(&audio_data, sample_rate);
                for (j, &partial) in computed_partials.iter().enumerate().take(NUM_PARTIALS) {
                    partials_results[i][j] = partial;
                }
            }
        }
        println!(
            "Channel {}: Partial Results: {:?}",
            channel, partials_results[i]
        );
    }

    // Update the spectrum_app with the new partials results
    if let Ok(mut app) = spectrum_app.lock() {
        app.partials = partials_results;
    } else {
        error!("Failed to lock spectrum app.");
    }
}

/// Starts the processing thread to handle audio data and spectrum updates.
pub fn start_processing_thread(
    num_channels: usize,
    audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
    sample_rate: u32,
    receiver: std::sync::mpsc::Receiver<Vec<f32>>,
) {
    std::thread::spawn(move || {
        while let Ok(buffer) = receiver.recv() {
            process_samples(
                buffer,
                num_channels,
                &audio_buffers,
                &spectrum_app,
                &selected_channels,
                sample_rate,
            );
        }
    });
}
