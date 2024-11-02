use std::sync::{Arc, Mutex};
use eframe::egui;
use egui::plot::{Plot, BarChart};
pub use eframe::NativeOptions;

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
    colors: Vec<egui::Color32>,
    y_scale: f32, // Y scale for the plot
    x_max: f64,
    alpha: u8, // Alpha value for colors
    bar_width: f32, // New field for bar width
}

impl MyApp {
    pub fn new(spectrum: Arc<Mutex<SpectrumApp>>) -> Self {
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
        MyApp {
            spectrum,
            colors,
            y_scale: 80.0, // Default Y scale value
            x_max: 1000.0,
            alpha: 255, // Set default alpha to fully opaque
            bar_width: 25.0, // Default bar width
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request continuous repaint to update the plot
        ctx.request_repaint();

        // Set the UI to dark mode visuals
        let dark_visuals = egui::Visuals::dark();
        ctx.set_visuals(dark_visuals);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("First Twelve Partials per Channel");

            // Slider to adjust the X max, Y max, and alpha
            ui.horizontal(|ui| {
                ui.label("X Max:"); // Capitalize "X" in "X max"
                ui.add(egui::Slider::new(&mut self.x_max, 100.0..=5000.0).text("Hz"));
                ui.label("Y Max:"); // Change label to "Y max"
                ui.add(egui::Slider::new(&mut self.y_scale, 0.0..=100.0).text("dB")); // Change "Scale" to "dB"
                ui.label("Alpha:");
                ui.add(egui::Slider::new(&mut self.alpha, 0..=255).text(""));
                ui.label("Width:");
                ui.add(egui::Slider::new(&mut self.bar_width, 0.0..=50.0).text(""));
            });

            // Add buttons for resetting to defaults and resetting the origin
            ui.horizontal(|ui| {
                if ui.button("Reset to Defaults").clicked() {
                    self.y_scale = 80.0;
                    self.x_max = 1000.0;
                    self.alpha = 255; // Reset alpha to fully opaque
                    self.bar_width = 25.0; // Reset bar width to default
                }
                if ui.button("Reset Origin").clicked() {
                    self.y_scale = 80.0; // Reset y_scale to default
                    self.x_max = 1000.0; // Reset x_max to default
                }
            });

            // Lock the spectrum app only for the duration of data access
            let partials = {
                let spectrum_app = self.spectrum.lock().unwrap();
                spectrum_app.partials.clone() // Clone the data to release the lock quickly
            };

            // Customize plot style using Plot's settings
            Plot::new("spectrum_plot")
                .legend(egui::plot::Legend::default().position(egui::plot::Corner::LeftTop)) // Move legend to upper left
                .view_aspect(3.0)  // Adjust aspect ratio to make the plot wider
                .include_x(0.0)    // Fixed x scale from 0 Hz
                .include_x(self.x_max)  // Use configurable maximum
                .include_y(0.0)    // Set y scale to 0
                .include_y(self.y_scale) // Use adjustable Y scale
                .x_axis_formatter(|x, _| format!("{:.0} Hz", x)) // X-axis label
                .y_axis_formatter(|y, _| format!("{:.0} dB", y)) // Y-axis label
                .label_formatter(|name, value| {
                    if name == "x" {
                        format!("Frequency: {:?} Hz", value)
                    } else {
                        format!("Amplitude: {:?} dB", value)
                    }
                })
                .show_x(true) // Show x-axis labels outside the plot
                .show_y(true) // Show y-axis labels outside the plot
                .show(ui, |plot_ui| {
                    // Plot partials as bars for each channel
                    for (channel, channel_partials) in partials.iter().enumerate() {
                        let bars: Vec<egui::plot::Bar> = if channel_partials.is_empty() {
                            vec![egui::plot::Bar::new(0.0, 0.0).width(0.0)] // Dummy value with zero width
                        } else {
                            channel_partials
                                .iter()
                                .map(|&(freq, amp)| egui::plot::Bar::new(freq as f64, amp as f64).width(self.bar_width as f64))
                                .collect()
                        };

                        let color = egui::Color32::from_rgba_unmultiplied(
                            self.colors[channel % self.colors.len()].r(),
                            self.colors[channel % self.colors.len()].g(),
                            self.colors[channel % self.colors.len()].b(),
                            self.alpha,
                        );

                        let bar_chart = BarChart::new(bars)
                            .name(format!("Channel {}", channel)) // Updated legend label
                            .color(color);

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
