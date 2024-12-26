use anyhow::Result;
use portaudio as pa;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
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
    buffer: Arc<RwLock<CircularBuffer>>,
    sender: Sender<Vec<f32>>,
) -> Result<pa::Stream<pa::NonBlocking, pa::Input<f32>>> {
    let device_info = pa.device_info(device_index)?;
    let latency = device_info.default_low_input_latency;

    let input_params = pa::StreamParameters::<f32>::new(device_index, num_channels, true, latency);
    let settings = pa::InputStreamSettings::new(input_params, sample_rate, 256);

    info!("Opening non-blocking PortAudio stream.");

    let stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            let buffer_data = args.buffer.to_vec();

            // Lock the buffer and push new samples
            if let Ok(mut buf) = buffer.write() {
                for &sample in &buffer_data {
                    buf.push(sample);
                }
            } else {
                error!("Failed to lock audio buffer.");
            }

            // Send buffer data to the processing thread
            if let Err(e) = sender.send(buffer_data) {
                error!("Failed to send audio buffer: {}", e);
            }

            // Handle overflow errors
            if args.flags.contains(pa::StreamCallbackFlags::INPUT_OVERFLOW) {
                error!("Input overflow detected.");
            }
            pa::Continue
        },
    )?;

    info!("PortAudio stream opened successfully.");
    Ok(stream)
}

/// Starts the audio sampling thread.
pub fn start_sampling_thread(
    running: Arc<AtomicBool>,
    buffer: Arc<RwLock<CircularBuffer>>,
    sender: Sender<Vec<f32>>,
    num_channels: usize,
    sample_rate: f64,
    _buffer_size: Arc<RwLock<usize>>,
) {
    std::thread::spawn(move || {
        let pa = pa::PortAudio::new().expect("Failed to initialize PortAudio");

        let mut stream = build_input_stream(
            &pa,
            pa.default_input_device().expect("No default input device available"),
            num_channels as i32,
            sample_rate,
            buffer.clone(),
            sender.clone(),
        ).expect("Failed to build input stream");

        stream.start().expect("Failed to start stream");
        info!("Audio sampling thread started.");

        // Keep sampling while running flag is true
        while running.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Stop and clean up the stream when the program terminates
        stream.stop().expect("Failed to stop stream");
        stream.close().expect("Failed to close stream");
        pa.terminate().expect("Failed to terminate PortAudio");
        info!("Audio sampling thread stopped.");
    });
}
