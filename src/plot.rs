use std::sync::{Arc, Mutex};
use eframe::egui;
use egui::plot::{Plot, BarChart};
pub use eframe::NativeOptions;
use crate::fft_analysis::FFTConfig;

pub struct SpectrumApp {
    pub partials: Vec<Vec<(f32, f32)>>, // Frequency, amplitude pairs for partials
}

impl SpectrumApp {
    pub fn new(num_channels: usize) -> Self {
        // Initialize each channel with twelve (frequency, amplitude) pairs, defaulting to zero
        SpectrumApp {
            partials: vec![vec![(0.0, 0.0); 12]; num_channels],
        }
    }

    pub fn update_partials(&mut self, new_partials: Vec<Vec<(f32, f32)>>) {
        // Update each channel with the new data
        for (channel, data) in new_partials.into_iter().enumerate() {
            if channel < self.partials.len() {
                self.partials[channel] = data;
            }
        }
    }
}

pub struct MyApp {
    pub spectrum: Arc<Mutex<SpectrumApp>>,
    pub fft_config: Arc<Mutex<FFTConfig>>, // Configuration sliders
    pub buffer_size: Arc<Mutex<usize>>,   // Shared buffer size
    colors: Vec<egui::Color32>,
    y_scale: f32,   // Y scale for the plot
    alpha: u8,      // Alpha value for bar colors
    bar_width: f32, // Width of the bars in the plot
}

impl MyApp {
    pub fn new(
        spectrum: Arc<Mutex<SpectrumApp>>,
        fft_config: Arc<Mutex<FFTConfig>>,
        buffer_size: Arc<Mutex<usize>>,
    ) -> Self {
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
            fft_config,
            buffer_size,
            colors,
            y_scale: 80.0, // Default Y scale
            alpha: 255,    // Fully opaque by default
            bar_width: 5.0, // Default bar width
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Always repaint to reflect new data
        ctx.request_repaint();

        // Set the UI to dark mode
        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("First Twelve Partials per Channel");

            // Configuration sliders for FFT
            let mut fft_config = self.fft_config.lock().unwrap();
            ui.horizontal(|ui| {
                ui.label("Min Frequency:");
                ui.add(egui::Slider::new(&mut fft_config.min_frequency, 10.0..=200.0).text("Hz"));
                ui.label("Max Frequency:");
                ui.add(egui::Slider::new(&mut fft_config.max_frequency, 500.0..=5000.0).text("Hz"));
                ui.label("DB Threshold:");
                ui.add(egui::Slider::new(&mut fft_config.db_threshold, -80.0..=-10.0).text("dB"));
            });

            // Slider for buffer size (powers of 2)
            let mut buffer_size = *self.buffer_size.lock().unwrap();
            let mut buffer_log_slider = (buffer_size as f32).log2().round() as u32;
            ui.horizontal(|ui| {
                ui.label("Buffer Size:");
                if ui
                    .add(egui::Slider::new(&mut buffer_log_slider, 6..=14).text("Power of 2"))
                    .changed()
                {
                    buffer_size = 1 << buffer_log_slider;
                    *self.buffer_size.lock().unwrap() = buffer_size;
                }
                ui.label(format!("{} samples", buffer_size)); // Display actual buffer size
            });

            // Sliders for Y scale and bar width
            ui.horizontal(|ui| {
                ui.label("Y Max:");
                ui.add(egui::Slider::new(&mut self.y_scale, 0.0..=100.0).text("dB"));
                ui.label("Alpha:");
                ui.add(egui::Slider::new(&mut self.alpha, 0..=255).text(""));
                ui.label("Bar Width:");
                ui.add(egui::Slider::new(&mut self.bar_width, 1.0..=10.0).text(""));
            });

            // Reset to defaults
            if ui.button("Reset to Defaults").clicked() {
                fft_config.min_frequency = 20.0;
                fft_config.max_frequency = 1000.0;
                fft_config.db_threshold = -32.0;
                self.y_scale = 80.0;
                self.alpha = 255;
                self.bar_width = 5.0;
                *self.buffer_size.lock().unwrap() = 8192;
            }

            // Lock spectrum data for access
            let partials = {
                let spectrum = self.spectrum.lock().unwrap();
                spectrum.partials.clone()
            };

            // Prepare all channels' bar charts before rendering
            let all_bar_charts: Vec<BarChart> = partials
                .iter()
                .enumerate()
                .map(|(channel, channel_partials)| {
                    let bars: Vec<egui::plot::Bar> = channel_partials
                        .iter()
                        .map(|&(freq, amp)| {
                            egui::plot::Bar::new(freq as f64, amp as f64)
                                .width(self.bar_width as f64)
                        })
                        .collect();

                    let color = egui::Color32::from_rgba_unmultiplied(
                        self.colors[channel % self.colors.len()].r(),
                        self.colors[channel % self.colors.len()].g(),
                        self.colors[channel % self.colors.len()].b(),
                        self.alpha,
                    );

                    BarChart::new(bars)
                        .name(format!("Channel {}", channel + 1))
                        .color(color)
                })
                .collect();

            // Display all channels together in the plot
            Plot::new("spectrum_plot")
                .legend(
                    egui::plot::Legend::default()
                        .position(egui::plot::Corner::RightTop)
                        .background_alpha(0.5),
                )
                .view_aspect(6.0)
                .include_x(0.0)
                .include_x(fft_config.max_frequency as f64)
                .include_y(0.0)
                .include_y(self.y_scale)
                .show(ui, |plot_ui| {
                    for bar_chart in all_bar_charts {
                        plot_ui.bar_chart(bar_chart);
                    }
                });

            // Add text display for partials below the plot
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.label("Channel Results:");

                    // Display each channel's results
                    for (channel, channel_partials) in partials.iter().enumerate() {
                        let formatted_partials: Vec<String> = channel_partials
                            .iter()
                            .map(|&(freq, amp)| format!("({:.2}, {})", freq, amp.round() as i32)) // Convert amplitude to integer
                            .collect();
                        let text = format!(
                            "Channel {}: [{}]",
                            channel + 1,
                            formatted_partials.join(", ")
                        );
                        ui.label(egui::RichText::new(text).size(10.0)); // Smaller font size
                    }
                });
        });
    }
}

pub fn run_native(
    app_name: &str,
    native_options: NativeOptions,
    app_creator: Box<dyn FnOnce(&eframe::CreationContext<'_>) -> Box<dyn eframe::App>>,
) -> Result<(), eframe::Error> {
    eframe::run_native(app_name, native_options, app_creator)
}
