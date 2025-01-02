use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use aubio_rs::{Pitch, PitchMode};
use log::info;
use crate::audio_stream::CircularBuffer;

pub struct PitchResults {
    pub frequencies: Vec<f32>,
    pub confidences: Vec<f32>,
}

impl PitchResults {
    pub fn new(num_channels: usize) -> Self {
        PitchResults {
            frequencies: vec![0.0; num_channels],
            confidences: vec![0.0; num_channels],
        }
    }
}

pub fn start_pitch_detection(
    audio_buffer: Arc<RwLock<CircularBuffer>>,
    pitch_results: Arc<Mutex<PitchResults>>,
    sample_rate: u32,
    selected_channels: Vec<usize>,
    shutdown_flag: Arc<AtomicBool>,
) {
    info!("Starting pitch detection thread");

    // Create a pitch detector for each channel
    let mut detectors: Vec<Pitch> = selected_channels
        .iter()
        .map(|_| {
            Pitch::new(
                PitchMode::Yinfft,
                2048,
                512,
                sample_rate,
            ).expect("Failed to create pitch detector")
        })
        .collect();

    while !shutdown_flag.load(Ordering::SeqCst) {
        // Get a clone of the current audio data
        let audio_data = if let Ok(buffer) = audio_buffer.read() {
            buffer.clone_data()
        } else {
            thread::sleep(Duration::from_millis(10));
            continue;
        };

        if audio_data.is_empty() {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Process each channel
        let mut new_frequencies = vec![0.0; selected_channels.len()];
        let mut new_confidences = vec![0.0; selected_channels.len()];

        for (i, &channel) in selected_channels.iter().enumerate() {
            // Extract single channel data
            let channel_data: Vec<f32> = audio_data
                .chunks(selected_channels.len())
                .map(|chunk| chunk[channel])
                .collect();

            // Process the audio in chunks
            if let Ok(frequency) = detectors[i].do_result(&channel_data) {
                new_frequencies[i] = frequency;
                // Get confidence directly - it's not wrapped in a Result
                new_confidences[i] = detectors[i].get_confidence();
            }
        }

        // Update results
        if let Ok(mut results) = pitch_results.lock() {
            results.frequencies = new_frequencies;
            results.confidences = new_confidences;
        }

        // Don't hog the CPU
        thread::sleep(Duration::from_millis(10));
    }

    info!("Pitch detection thread shutting down");
} 