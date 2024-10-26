use std::sync::{Arc, Mutex};
use eframe::egui;
use egui::plot::{Plot, BarChart};
use eframe::NativeOptions;

pub struct SpectrumApp {
    pub partials: Vec<Vec<(f32, f32)>>, // Frequency, amplitude pairs for partials
}

impl SpectrumApp {
    pub fn new(num_channels: usize) -> Self {
        SpectrumApp {
            partials: vec![Vec::new(); num_channels], // Initialize partial results
        }
    }
}

pub struct MyApp {
    pub spectrum: Arc<Mutex<SpectrumApp>>,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Lock the spectrum app
        let spectrum_app = self.spectrum.lock().unwrap();

        // Request continuous repaint to update the plot
        ctx.request_repaint();

        // Set the UI to dark mode visuals
        let dark_visuals = egui::Visuals::dark();
        ctx.set_visuals(dark_visuals);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Frequency vs Amplitude (Partials)");

            // Define colors for each channel
            let colors = vec![
                egui::Color32::from_rgb(0, 0, 255),    // Channel 1 - Blue
                egui::Color32::from_rgb(255, 165, 0),  // Channel 2 - Orange
                egui::Color32::from_rgb(0, 255, 0),    // Channel 3 - Green
                egui::Color32::from_rgb(255, 0, 0),    // Channel 4 - Red
                egui::Color32::from_rgb(238, 130, 238), // Channel 5 - Violet
                egui::Color32::from_rgb(165, 42, 42),   // Channel 6 - Brown
                egui::Color32::from_rgb(75, 0, 130),   // Channel 7 - Indigo
                egui::Color32::from_rgb(255, 255, 0),  // Channel 8 - Yellow
            ];

            // Customize plot style using Plot's settings
            egui::plot::Plot::new("spectrum_plot")
                .legend(egui::plot::Legend::default())
                .view_aspect(2.0)  // Adjust aspect ratio (optional)
                .include_x(0.0)    // Fixed x scale from 0 Hz
                .include_x(1000.0) // Maximum x scale set to 1000 Hz
                .include_y(0.0)    // Set y scale to 0
                .include_y(80.0)   // Maximum y scale set to 80
                .show(ui, |plot_ui| {
                    // Plot partials as bars for each channel
                    for channel in 0..spectrum_app.partials.len() {
                        let bars: Vec<egui::plot::Bar> = if spectrum_app.partials[channel].is_empty() {
                            vec![egui::plot::Bar::new(0.0, 0.0)] // Dummy value to ensure the channel appears in the legend
                        } else {
                            spectrum_app.partials[channel]
                                .iter()
                                .map(|&(freq, amp)| egui::plot::Bar::new(freq as f64, amp as f64).width(3.0)) // Convert f32 to f64
                                .collect()
                        };

                        let bar_chart = egui::plot::BarChart::new(bars)
                            .name(format!("Partials Channel {}", channel + 1))
                            .color(colors[channel % colors.len()]);

                        // Draw the bar chart in the plot UI
                        plot_ui.bar_chart(bar_chart);
                    }
                });
        });
    }
}

// Add this function to make it accessible from main.rs
pub fn run_native(
    app_name: &str,
    native_options: NativeOptions,
    app_creator: Box<dyn FnOnce(&eframe::CreationContext<'_>) -> Box<dyn eframe::App>>,
) -> Result<(), eframe::Error> {
    eframe::run_native(app_name, native_options, app_creator)
}
