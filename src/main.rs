mod audio_stream;
mod fft_analysis;
mod plot;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat};
use std::sync::{Arc, Mutex};
use audio_stream::build_input_stream;
use plot::{MyApp, SpectrumApp};
use eframe::NativeOptions;

fn main() -> Result<()> {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");
    println!("Using input device: {:?}", device.name());
    let config = device.default_input_config()?;

    // Specify the channels to use explicitly
    let selected_channels = vec![2, 3, 4, 5]; // Example: using channels 0, 2, 3, and 6

    // Initialize audio buffers for each selected channel
    let audio_buffers: Arc<Vec<Mutex<Vec<f32>>>> = Arc::new(
        selected_channels
            .iter()
            .map(|_| Mutex::new(Vec::new()))
            .collect(),
    );

    // Create the spectrum app state
    let spectrum_app = Arc::new(Mutex::new(SpectrumApp::new(selected_channels.len())));

    // Build input stream based on sample format
    let stream = match config.sample_format() {
        SampleFormat::F32 => build_input_stream::<f32>(&device, &config.into(), audio_buffers.clone(), spectrum_app.clone(), selected_channels.clone())?,
        SampleFormat::I16 => build_input_stream::<i16>(&device, &config.into(), audio_buffers.clone(), spectrum_app.clone(), selected_channels.clone())?,
        SampleFormat::U16 => build_input_stream::<u16>(&device, &config.into(), audio_buffers.clone(), spectrum_app.clone(), selected_channels.clone())?,
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    // Start the stream
    stream.play()?;

    // Launch the eframe application for plotting
    let native_options = NativeOptions::default();
    eframe::run_native(
        "Real-Time Spectrum Analyzer",
        native_options,
        Box::new(move |_cc| {
            Box::new(MyApp {
                spectrum: spectrum_app.clone(),
            })
        }),
    );

    Ok(())
}
