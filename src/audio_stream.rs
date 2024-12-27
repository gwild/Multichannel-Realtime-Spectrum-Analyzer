use anyhow::Result;
use portaudio as pa;
use std::sync::{Arc, Mutex};
use portaudio::stream::InputCallbackArgs;
use log::{info, error};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;

/// Circular buffer for storing audio samples per channel.
pub struct CircularBuffer {
    buffer: Vec<f32>,
    head: usize,
    size: usize,
    cycle_count: usize,
}

impl CircularBuffer {
    /// Creates a new CircularBuffer.
    pub fn new(size: usize) -> Self {
        CircularBuffer {
            buffer: vec![0.0; size],
            head: 0,
            size,
            cycle_count: 0,
        }
    }

    /// Pushes a sample into the buffer.
    pub fn push(&mut self, value: f32) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.size;

        if self.head == 0 {
            self.cycle_count += 1;
            info!("Buffer for channel wrapped around {} times.", self.cycle_count);
        }
    }

    /// Retrieves the latest samples.
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

/// Builds the audio input stream with per-channel buffer handling.
pub fn build_input_stream(
    pa: &pa::PortAudio,
    device_index: pa::DeviceIndex,
    num_channels: i32,
    sample_rate: f64,
    audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
    selected_channels: Vec<usize>,
    sender: Sender<Vec<f32>>,  // New sender to pass audio data
) -> Result<pa::Stream<pa::NonBlocking, pa::Input<f32>>> {
    let device_info = pa.device_info(device_index)?;
    let latency = device_info.default_low_input_latency;

    let input_params = pa::StreamParameters::<f32>::new(device_index, num_channels, true, latency);
    let settings = pa::InputStreamSettings::new(input_params, sample_rate, 256);

    info!("Opening PortAudio stream for {} channels.", num_channels);

    let stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            let buffer_data = args.buffer.to_vec();
            
            // Process each channel separately
            for (i, &sample) in buffer_data.iter().enumerate() {
                let channel = i % num_channels as usize;
                if let Some(pos) = selected_channels.iter().position(|&c| c == channel) {
                    if let Ok(mut buffer) = audio_buffers[pos].lock() {
                        buffer.push(sample);
                    } else {
                        error!("Failed to lock buffer for channel {}", channel);
                    }
                }
            }

            // Send the full buffer data to the processing thread
            if let Err(e) = sender.send(buffer_data) {
                error!("Failed to send audio buffer: {}", e);
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

/// Starts audio sampling with per-channel buffers.
pub fn start_sampling_thread(
    running: Arc<AtomicBool>,
    audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
    num_channels: usize,
    sample_rate: f64,
    selected_channels: Vec<usize>,
    sender: Sender<Vec<f32>>,  // Add sender to thread spawning
) {
    std::thread::spawn(move || {
        let pa = pa::PortAudio::new().expect("Failed to initialize PortAudio");

        let mut stream = build_input_stream(
            &pa,
            pa.default_input_device().expect("No default input device available"),
            num_channels as i32,
            sample_rate,
            audio_buffers.clone(),
            selected_channels.clone(),
            sender.clone(),  // Pass the sender to the stream builder
        ).expect("Failed to build input stream");

        stream.start().expect("Failed to start stream");
        info!("Audio sampling thread started.");

        while running.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        stream.stop().expect("Failed to stop stream");
        stream.close().expect("Failed to close stream");
        pa.terminate().expect("Failed to terminate PortAudio");
        info!("Audio sampling thread stopped.");
    });
}
