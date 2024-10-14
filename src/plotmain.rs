use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{StreamConfig};
use eframe::{self, egui};
use std::sync::{Arc, Mutex};

struct SpectrumApp {
    partials: Vec<Vec<(f32, f32)>>, // Frequency and amplitude pairs for multiple channels
}

impl SpectrumApp {
    fn new() -> Self {
        SpectrumApp {
            partials: vec![vec![(0.0, 0.0); 12]; 6], // Example placeholder for 6 channels with 12 partials
        }
    }
}

struct MyApp {
    spectrum: Arc<Mutex<SpectrumApp>>,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let spectrum_app = self.spectrum.lock().unwrap();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Frequency vs Amplitude");

            // Create a plot
            egui::plot::Plot::new("spectrum_plot")
                .show(ui, |plot_ui| {
                    for (_channel, partials) in spectrum_app.partials.iter().enumerate() {
                        // Print the contents of partials for troubleshooting
                        println!("Channel {}: {:?}", _channel, partials);

                        let points: Vec<[f64; 2]> = partials
                            .iter()
                            .map(|&(freq, amp)| [freq as f64, amp as f64])
                            .collect();

                        plot_ui.line(egui::plot::Line::new(egui::plot::PlotPoints::new(points)));
                    }
                });
        });
    }
}

// Handle the Result from run_native
fn main() {
    // Create the application state
    let app_state = Arc::new(Mutex::new(SpectrumApp::new()));

    // Run the eframe app
    let native_options = eframe::NativeOptions::default();
    let _ = eframe::run_native( // Ignore the result
        "Real-Time Spectrum Analyzer",
        native_options,
        Box::new(move |_cc| {
            // Initialize the audio input stream here
            let audio_buffers = Arc::new(Mutex::new(Vec::new()));
            let app_state_clone = Arc::clone(&app_state);

            // Initialize the audio stream
            let host = cpal::default_host();
            let input_device = host.default_input_device().expect("No input device available");
            let input_config = input_device.default_input_config().expect("Error getting default config");
            let input_stream_config: StreamConfig = StreamConfig {
                channels: input_config.channels(),
                sample_rate: input_config.sample_rate(),
                buffer_size: cpal::BufferSize::Default,
            };

            // Start the input stream
            let stream = build_input_stream::<f32>(
                &input_device,
                &input_stream_config,
                audio_buffers.clone(),
                app_state_clone,
            ).expect("Failed to build input stream");

            // Start the stream
            stream.play().expect("Failed to play stream");

            // Return the app instance
            Box::new(MyApp {
                spectrum: app_state,
            })
        }),
    );
}

// Function to build the input stream
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    audio_buffers: Arc<Mutex<Vec<f32>>>,
    app: Arc<Mutex<SpectrumApp>>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::Sample + Into<f32> + std::fmt::Debug + cpal::SizedSample,
{
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            // Lock the audio buffer to safely access it
            let mut buffers = audio_buffers.lock().unwrap();

            // Process incoming audio data
            for sample in data {
                let sample_f32: f32 = (*sample).into();
                buffers.push(sample_f32);
            }

            // Update the SpectrumApp with dummy data
            let mut app_data = app.lock().unwrap();
            for channel in 0..6 {
                app_data.partials[channel] = (0..12) // Generate 12 partials
                    .map(|i| {
                        let frequency = (i + 1) as f32 * 100.0; // Example frequency
                        let amplitude = buffers.last().copied().unwrap_or(0.0) * (channel as f32 + 1.0); // Amplitude based on last sample
                        (frequency, amplitude)
                    })
                    .collect();
            }
        },
        move |err| {
            eprintln!("Stream error: {}", err);
        },
        None, // Provide default latency
    )
}
