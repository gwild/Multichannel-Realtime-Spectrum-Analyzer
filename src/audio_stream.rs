use anyhow::Result;
use portaudio as pa;
use std::sync::{Arc, Mutex};
use std::io::{self, Write};
use crate::fft_analysis::{compute_spectrum, NUM_PARTIALS};
use crate::plot::SpectrumApp;
use portaudio::stream::InputCallbackArgs;

// Circular buffer implementation
pub struct CircularBuffer {
    buffer: Vec<f32>,
    head: usize,
    size: usize,
}

impl CircularBuffer {
    pub fn new(size: usize) -> Self {
        CircularBuffer {
            buffer: vec![0.0; size],
            head: 0,
            size,
        }
    }

    pub fn push(&mut self, value: f32) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % self.size; // Wrap around
    }

    fn get(&self) -> &[f32] {
        &self.buffer
    }
}

// Build input stream function
pub fn build_input_stream(
    pa: &pa::PortAudio,
    device_index: pa::DeviceIndex,
    num_channels: i32,
    sample_rate: f64,
    audio_buffers: Arc<Vec<Mutex<CircularBuffer>>>,
    spectrum_app: Arc<Mutex<SpectrumApp>>,
    selected_channels: Vec<usize>,
) -> Result<pa::Stream<pa::NonBlocking, pa::Input<f32>>> {
    if selected_channels.is_empty() {
        return Err(anyhow::anyhow!("No channels selected"));
    }

    let latency = pa.device_info(device_index)?.default_low_input_latency;
    let input_params = pa::StreamParameters::<f32>::new(device_index, num_channels, true, latency);

    let settings = pa::InputStreamSettings::new(input_params, sample_rate, 256);

    // Create the stream
    let mut stream = pa.open_non_blocking_stream(
        settings,
        move |args: InputCallbackArgs<f32>| {
            process_samples(
                args.buffer.to_vec(),
                num_channels as usize,
                &audio_buffers,
                &spectrum_app,
                &selected_channels,
                sample_rate as u32,
            );
            pa::Continue
        },
    )?;

    println!("Stream created with device index: {:?}", device_index);

    Ok(stream)
}

// Helper function to process samples
fn process_samples(
    data_as_f32: Vec<f32>,
    _channels: usize,
    audio_buffers: &Arc<Vec<Mutex<CircularBuffer>>>,
    spectrum_app: &Arc<Mutex<SpectrumApp>>,
    selected_channels: &[usize],
    sample_rate: u32,
) {
    for (i, &sample) in data_as_f32.iter().enumerate() {
        let channel = i % _channels;
        if selected_channels.contains(&channel) {
            let buffer_index = selected_channels.iter().position(|&ch| ch == channel).unwrap();
            let mut buffer = audio_buffers[buffer_index].lock().unwrap();
            buffer.push(sample);
        }
    }

    let mut partials_results = vec![vec![(0.0, 0.0); NUM_PARTIALS]; selected_channels.len()];

    for (i, &_channel) in selected_channels.iter().enumerate() {
        let buffer = audio_buffers[i].lock().unwrap();
        let audio_data = buffer.get();

        if !audio_data.is_empty() {
            let computed_partials = compute_spectrum(audio_data, sample_rate);
            for (j, &partial) in computed_partials.iter().enumerate().take(NUM_PARTIALS) {
                partials_results[i][j] = partial;
            }
        }
    }

    let mut app = spectrum_app.lock().unwrap();
    app.partials = partials_results;
}

// Main function
pub fn main() -> Result<()> {
    let pa = pa::PortAudio::new()?;

    // List devices
    let devices = pa.devices()?.collect::<Result<Vec<_>, _>>()?;
    println!("Available Input Devices:");
    for (i, (_index, info)) in devices.iter().enumerate() {
        if info.max_input_channels > 0 {
            println!("  [{}] - {}", i, info.name);
        }
    }

    // Select a device
    print!("Enter the index of the desired device: ");
    io::stdout().flush()?;
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;
    let device_index = user_input.trim().parse::<usize>()?;
    let selected_device_index = devices[device_index].0;
    let selected_device_info = pa.device_info(selected_device_index)?;

    println!("Selected device: {}", selected_device_info.name);

    let num_channels = selected_device_info.max_input_channels;
    let sample_rate = selected_device_info.default_sample_rate;

    // Create audio buffers for selected channels
    let selected_channels = vec![0, 1]; // Example: First two channels
    let audio_buffers: Arc<Vec<Mutex<CircularBuffer>>> = Arc::new(
        selected_channels
            .iter()
            .map(|_| Mutex::new(CircularBuffer::new(512)))
            .collect(),
    );

    // Initialize spectrum app
    let spectrum_app = Arc::new(Mutex::new(SpectrumApp::new(selected_channels.len())));

    // Build and start the stream
    let mut stream = build_input_stream(
        &pa,
        selected_device_index,
        num_channels,
        sample_rate,
        audio_buffers.clone(),
        spectrum_app.clone(),
        selected_channels.clone(),
    )?;
    stream.start()?;

    println!("Stream started!");

    loop {
        // Prevent the main thread from exiting
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
