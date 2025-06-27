// This section is protected. Do not alter unless permission is requested by you and granted by me.
use std::sync::{Arc, Mutex};
use eframe::egui;
use egui::plot::{Plot, BarChart, Legend};
pub use eframe::NativeOptions;
use crate::fft_analysis::FFTConfig;
use crate::audio_stream::CircularBuffer;
use log::{info, debug, error, warn};
use std::sync::atomic::{AtomicBool, Ordering};// Importing necessary types for GUI throttling.
// Reminder: Added to implement GUI throttling. Do not modify without permission.
use std::time::{Duration, Instant};
use std::sync::RwLock;
use crate::{MIN_FREQ, MAX_FREQ, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE, DEFAULT_BUFFER_SIZE};
use crate::fft_analysis::WindowType;  // Add at top with other imports
use crate::resynth::ResynthConfig;  // Add this import
use crate::resynth::DEFAULT_UPDATE_RATE;
use crate::DEFAULT_NUM_PARTIALS;  // Import the new constant
use egui::widgets::plot::uniform_grid_spacer;
use std::collections::VecDeque;
use chrono;
use egui::TextStyle;
use egui::FontId;
use egui::FontFamily;
use egui::Color32;
use tokio::sync::broadcast; // Added import
use crate::display::SpectralDisplay; // Added import
use std::sync::mpsc; // Add this for mpsc::Sender
use crate::get_results::GuiParameter; // Add this for the enum
use crate::presets::{PresetManager, Preset};

// Define type alias
type PartialsData = Vec<Vec<(f32, f32)>>; 

pub struct SpectrographSlice {
    pub time: f64,
    pub data: Vec<(f64, f32)>,
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub struct SpectrumApp {
    absolute_values: Vec<Vec<(f32, f32)>>,       // Frequency, absolute pairs
    partial_textures: Vec<Option<egui::TextureHandle>>,
    freq_axis_lines: Vec<Vec<[f32; 2]>>,
    num_channels: usize,
    num_partials: usize,  // Add num_partials field
    fft_line_data: Vec<Vec<(f32, f32)>>,  // Add this field
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
impl SpectrumApp {
    pub fn new(num_channels: usize) -> Self {
        SpectrumApp {
            absolute_values: vec![vec![(0.0, 0.0); DEFAULT_NUM_PARTIALS]; num_channels],  // Use DEFAULT_NUM_PARTIALS
            partial_textures: vec![None; num_channels],
            freq_axis_lines: vec![vec![]; 2],
            num_channels,
            num_partials: DEFAULT_NUM_PARTIALS,  // Initialize with default
            fft_line_data: Vec::new(),  // Initialize empty
        }
    }

    pub fn update_partials(&mut self, partials: Vec<Vec<(f32, f32)>>) {
        let num_channels = partials.len();
        self.absolute_values = partials;
        
        // Update num_partials if needed (assuming all channels have same number of partials)
        if !self.absolute_values.is_empty() {
            self.num_partials = self.absolute_values[0].len();
        }
        
        // Resize textures array to match channel count
        self.partial_textures.resize_with(num_channels, || None);
        
        // Always keep 2 frequency axis lines
        self.freq_axis_lines = vec![vec![]; 2]; 
        
        // Update channel count display
        self.num_channels = num_channels;
    }

    #[allow(dead_code)]
    /// Get a copy of the current spectral data in absolute values (matching GUI display)
    pub fn clone_absolute_data(&self) -> Vec<Vec<(f32, f32)>> {
        self.absolute_values.clone()
    }

    pub fn update_fft_line_data(&mut self, data: Vec<Vec<(f32, f32)>>) {
        self.fft_line_data = data;
    }

    pub fn get_fft_line_data(&self) -> &Vec<Vec<(f32, f32)>> {
        &self.fft_line_data
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
pub struct MyApp {
    pub spectrum: Arc<Mutex<SpectrumApp>>,
    pub fft_config: Arc<Mutex<FFTConfig>>,
    pub buffer_size: Arc<Mutex<usize>>,
    pub audio_buffer: Arc<RwLock<CircularBuffer>>,
    pub resynth_config: Arc<Mutex<ResynthConfig>>,
    colors: Vec<egui::Color32>,
    y_scale: f32,
    alpha: u8,
    bar_width: f32,
    show_line_plot: bool,
    show_spectrograph: bool,
    last_repaint: Instant,
    shutdown_flag: Arc<AtomicBool>,
    spectrograph_history: Arc<Mutex<VecDeque<SpectrographSlice>>>,
    start_time: Arc<Instant>,
    sample_rate: f64,
    show_results: bool,
    partials_rx: Option<broadcast::Receiver<PartialsData>>,
    gui_param_tx: mpsc::Sender<GuiParameter>, // Add this field
    // Fields for buffer size debouncing
    desired_buffer_size: Option<usize>,
    buffer_debounce_timer: Option<Instant>,
    gain_update_tx: std::sync::mpsc::Sender<f32>,
    // Preset management fields
    preset_manager: PresetManager,
    selected_preset_name: String,
    is_adding_preset: bool,
    new_preset_name: String,
    show_delete_confirmation: bool,
    // New fields for overwrite confirmation
    show_overwrite_confirmation: bool,
    preset_to_overwrite: String,
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
impl MyApp {
    pub fn new(
        spectrum: Arc<Mutex<SpectrumApp>>,
        fft_config: Arc<Mutex<FFTConfig>>,
        buffer_size: Arc<Mutex<usize>>,
        audio_buffer: Arc<RwLock<CircularBuffer>>,
        resynth_config: Arc<Mutex<ResynthConfig>>,
        shutdown_flag: Arc<AtomicBool>,
        spectrograph_history: Arc<Mutex<VecDeque<SpectrographSlice>>>,
        start_time: Arc<Instant>,
        sample_rate: f64,
        partials_rx: broadcast::Receiver<PartialsData>,
        gui_param_tx: mpsc::Sender<GuiParameter>, // Add this parameter
        gain_update_tx: mpsc::Sender<f32>, // Add this param
    ) -> Self {
        let colors = vec![
            egui::Color32::from_rgb(0, 0, 255),
            egui::Color32::from_rgb(255, 165, 0),
            egui::Color32::from_rgb(0, 255, 0),
            egui::Color32::from_rgb(255, 0, 0),
            egui::Color32::from_rgb(238, 130, 238),
            egui::Color32::from_rgb(165, 42, 42),
            egui::Color32::from_rgb(75, 0, 130),
            egui::Color32::from_rgb(255, 255, 0),
        ];

        let preset_manager = PresetManager::new("presets.yaml").expect("Failed to load or create presets");
        let selected_preset_name = "default".to_string();

        let mut instance = MyApp {
            spectrum,
            fft_config,
            buffer_size,
            audio_buffer,
            resynth_config,
            colors,
            y_scale: 80.0,
            alpha: 255,
            bar_width: 5.0,
            show_line_plot: false,
            show_spectrograph: false,
            last_repaint: Instant::now(),
            shutdown_flag,
            spectrograph_history,
            start_time,
            sample_rate,
            show_results: true,
            partials_rx: Some(partials_rx),
            gui_param_tx, // Store the sender
            // Initialize debounce fields
            desired_buffer_size: None,
            buffer_debounce_timer: None,
            gain_update_tx,
            // Initialize preset fields
            preset_manager,
            selected_preset_name,
            is_adding_preset: false,
            new_preset_name: String::new(),
            show_delete_confirmation: false,
            // Initialize new fields
            show_overwrite_confirmation: false,
            preset_to_overwrite: String::new(),
        };

        // Apply the default preset on startup
        instance.load_preset("default");

        // FIX IMPLEMENTATION:
        // First, ensure the initial max_frequency from default() respects the
        // current runtime Nyquist limit (sets the slider's *range* correctly).
        {
            let buffer_s = instance.buffer_size.lock().unwrap();
            let nyquist_limit = (*buffer_s as f64 / 2.0).min(*MAX_FREQ); // Use runtime buffer size
            let mut cfg = instance.fft_config.lock().unwrap();
            if cfg.max_frequency > nyquist_limit {
                cfg.max_frequency = nyquist_limit;
                debug!("Clamped initial max frequency range to {:.1} Hz", cfg.max_frequency);
            }
            // Now, explicitly set the initial *value* of the slider from the preset,
            // which we already loaded. This might seem redundant, but it ensures
            // the initial state is correct even if default preset changes.
            if let Some(preset) = instance.preset_manager.presets.get("default") {
                cfg.max_frequency = preset.max_frequency;
                debug!("Set initial max frequency value to {} Hz from preset", cfg.max_frequency);
            } else {
                // Fallback in case default preset is somehow missing
                cfg.max_frequency = 1400.0;
                warn!("Default preset not found during initialization, falling back to 1400.0 Hz for max_frequency");
            }
        }

        // Return the newly created instance with the fix
        instance
    }

    pub fn update_buffer_size(&mut self, new_size: usize) {
        let current_size = match self.buffer_size.lock() {
            Ok(guard) => *guard,
            Err(_) => 0
        };
        
        info!("BUFFER RESIZE: Starting buffer resize process from {} to {}", 
              current_size, new_size);
          
        let validated_size = new_size
            .next_power_of_two()
            .max(512)
            .clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE);

        // Only log size adjustments at info level
        if validated_size != new_size {
            info!("BUFFER RESIZE: Size adjusted: {} → {}", new_size, validated_size);
        }

        // Log detailed calculations at debug level
        debug!("BUFFER RESIZE: Validation - min={}, max={}, power_of_2={}",
               MIN_BUFFER_SIZE, MAX_BUFFER_SIZE, validated_size);

        // Update the target buffer size value
        if let Ok(mut size) = self.buffer_size.lock() {
            info!("BUFFER RESIZE: Updating buffer_size from {} to {}", *size, validated_size);
            *size = validated_size;
        } else {
            error!("BUFFER RESIZE: Failed to lock buffer_size for update.");
            return; // Don't proceed if we can't update the size
        }

        // First try to perform a direct buffer resize to ensure it completes properly
        info!("BUFFER RESIZE: Attempting direct buffer resize operation");
        let direct_resize_success = crate::audio_stream::perform_buffer_resize(
            &self.audio_buffer,
            &self.buffer_size,
            &self.resynth_config
        );
        
        if direct_resize_success {
            info!("BUFFER RESIZE: Direct buffer resize completed successfully");
        } else {
            // If direct resize failed, signal the audio thread to handle the resize
            info!("BUFFER RESIZE: Direct resize failed, signaling audio thread");
            if let Ok(buffer) = self.audio_buffer.read() {
                // Use read lock just to access the flag Arc
                info!("BUFFER RESIZE: Setting needs_restart flag");
                buffer.needs_restart.store(true, Ordering::SeqCst);
                #[cfg(target_os = "linux")]
                {
                    info!("BUFFER RESIZE: Linux detected, setting force_reinit flag");
                    buffer.force_reinit.store(true, Ordering::SeqCst);
                }
                info!("BUFFER RESIZE: Signaled audio thread to restart due to buffer size change request to {}", validated_size);
            } else {
                error!("BUFFER RESIZE: Failed to lock audio_buffer to signal restart.");
            }
        }
        
        // Update FFT config to adjust max_frequency if needed based on new buffer size
        if let Ok(mut fft_config) = self.fft_config.lock() {
            // Calculate new Nyquist limit based on input sample rate
            let nyquist_limit = (self.sample_rate / 2.0).min(validated_size as f64 / 2.0);
            
            info!("BUFFER RESIZE: Checking FFT config - current max_frequency: {}, new nyquist_limit: {}", 
                  fft_config.max_frequency, nyquist_limit);
            
            // If current max_frequency exceeds the new Nyquist limit, adjust it
            if fft_config.max_frequency > nyquist_limit {
                info!("BUFFER RESIZE: Adjusting max_frequency from {} to {} Hz due to buffer resize", 
                       fft_config.max_frequency, nyquist_limit);
                fft_config.max_frequency = nyquist_limit;
            }
            
            // Also adjust root_freq_max if needed
            if fft_config.root_freq_max as f64 > nyquist_limit {
                info!("BUFFER RESIZE: Adjusting root_freq_max from {} to {} Hz due to buffer resize", 
                       fft_config.root_freq_max, nyquist_limit);
                fft_config.root_freq_max = nyquist_limit as f32;
            }
        } else {
            error!("BUFFER RESIZE: Failed to lock fft_config to adjust frequencies.");
        }

        info!("BUFFER RESIZE: Process completed for size {}", validated_size);
        
        // Explicitly request repaint to update UI state if needed
        self.last_repaint = Instant::now();
    }

    // Helper method to get the current nyquist limit based on input sample rate
    fn get_nyquist_limit(&self) -> f32 {
        let sample_rate = self.sample_rate as f32;
        (sample_rate / 2.0) as f32
    }

    // Capture the current GUI state into a Preset object
    fn capture_current_preset(&self) -> Preset {
        let fft_config = self.fft_config.lock().unwrap();
        let resynth_config = self.resynth_config.lock().unwrap();
        let buffer_size = *self.buffer_size.lock().unwrap();

        Preset {
            // FFTConfig fields
            min_frequency: fft_config.min_frequency,
            max_frequency: fft_config.max_frequency,
            magnitude_threshold: fft_config.magnitude_threshold,
            min_freq_spacing: fft_config.min_freq_spacing,
            window_type: fft_config.window_type.clone(),
            crosstalk_enabled: fft_config.crosstalk_enabled,
            crosstalk_threshold: fft_config.crosstalk_threshold,
            crosstalk_reduction: fft_config.crosstalk_reduction,
            harmonic_tolerance: fft_config.harmonic_tolerance,
            root_freq_min: fft_config.root_freq_min,
            root_freq_max: fft_config.root_freq_max,
            freq_match_distance: fft_config.freq_match_distance,
            // ResynthConfig fields
            gain: resynth_config.gain,
            freq_scale: resynth_config.freq_scale,
            update_rate: resynth_config.update_rate,
            // MyApp display fields
            y_scale: self.y_scale,
            alpha: self.alpha,
            bar_width: self.bar_width,
            show_line_plot: self.show_line_plot,
            show_spectrograph: self.show_spectrograph,
            show_results: self.show_results,
            buffer_size,
        }
    }

    // Load a preset's values into the current GUI state
    fn load_preset(&mut self, name: &str) {
        if let Some(preset) = self.preset_manager.presets.get(name).cloned() {
            info!("Loading preset: {}", name);
            let mut fft_config = self.fft_config.lock().unwrap();
            let mut resynth_config = self.resynth_config.lock().unwrap();

            // Apply FFTConfig fields
            fft_config.min_frequency = preset.min_frequency;
            fft_config.max_frequency = preset.max_frequency;
            fft_config.magnitude_threshold = preset.magnitude_threshold;
            fft_config.min_freq_spacing = preset.min_freq_spacing;
            fft_config.window_type = preset.window_type;
            fft_config.crosstalk_enabled = preset.crosstalk_enabled;
            fft_config.crosstalk_threshold = preset.crosstalk_threshold;
            fft_config.crosstalk_reduction = preset.crosstalk_reduction;
            fft_config.harmonic_tolerance = preset.harmonic_tolerance;
            fft_config.root_freq_min = preset.root_freq_min;
            fft_config.root_freq_max = preset.root_freq_max;
            fft_config.freq_match_distance = preset.freq_match_distance;

            // Apply ResynthConfig fields
            resynth_config.gain = preset.gain;
            resynth_config.freq_scale = preset.freq_scale;
            resynth_config.update_rate = preset.update_rate;

            // Apply MyApp display fields
            self.y_scale = preset.y_scale;
            self.alpha = preset.alpha;
            self.bar_width = preset.bar_width;
            self.show_line_plot = preset.show_line_plot;
            self.show_spectrograph = preset.show_spectrograph;
            self.show_results = preset.show_results;
            
            // Apply Buffer Size if it has changed
            let current_buffer_size = *self.buffer_size.lock().unwrap();
            if current_buffer_size != preset.buffer_size {
                info!("Preset loading new buffer size: {} -> {}", current_buffer_size, preset.buffer_size);
                self.desired_buffer_size = Some(preset.buffer_size);
                self.buffer_debounce_timer = Some(Instant::now());
            }
            
            // Send updates for parameters that require it (like gain)
            self.gui_param_tx.send(GuiParameter::Gain(resynth_config.gain)).unwrap_or_else(|e| error!("Failed to send Gain update on preset load: {}", e));
            self.gain_update_tx.send(resynth_config.gain).unwrap_or_else(|e| error!("Failed to send instant gain update on preset load: {}", e));
            self.gui_param_tx.send(GuiParameter::FreqScale(resynth_config.freq_scale)).unwrap_or_else(|e| error!("Failed to send FreqScale update on preset load: {}", e));
            self.gui_param_tx.send(GuiParameter::UpdateRate(resynth_config.update_rate)).unwrap_or_else(|e| error!("Failed to send UpdateRate update on preset load: {}", e));

            // Clear spectrograph history to avoid displaying stale data
            if let Ok(mut history) = self.spectrograph_history.lock() {
                info!("Clearing spectrograph history due to preset change.");
                history.clear();
            } else {
                error!("Failed to lock spectrograph history for clearing.");
            }

        } else {
            warn!("Attempted to load non-existent preset: {}", name);
        }
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
// Implementing eframe::App for MyApp

impl eframe::App for MyApp {
    fn on_close_event(&mut self) -> bool {
        info!("GUI close event detected, setting shutdown flag");
        self.shutdown_flag.store(true, Ordering::SeqCst);
        true
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Track GUI update cycles for debugging
        static mut LAST_UPDATE_LOG: Option<Instant> = None;
        static mut UPDATE_COUNT: usize = 0;
        static mut PLOT_UPDATE_COUNT: usize = 0;
        
        unsafe {
            UPDATE_COUNT += 1;
            
            // Log every update cycle
            PLOT_UPDATE_COUNT += 1;
            // Log every GUI update to track rendering frequency
            debug!("GUI update #{} - frame time: {:?}", 
                   PLOT_UPDATE_COUNT, 
                   self.last_repaint.elapsed());
            
            if let Some(last_time) = LAST_UPDATE_LOG {
                if last_time.elapsed() >= Duration::from_secs(5) {
                    debug!("GUI update stats: {} frames in last 5 seconds", UPDATE_COUNT);
                    LAST_UPDATE_LOG = Some(Instant::now());
                    UPDATE_COUNT = 0;
                }
            } else {
                LAST_UPDATE_LOG = Some(Instant::now());
            }
        }
        
        // Check if buffer resize operation is in progress
        let buffer_resize_in_progress = {
            if let Ok(buffer) = self.audio_buffer.read() {
                let needs_restart = buffer.needs_restart();
                let needs_reinit = buffer.needs_reinit();
                if needs_restart || needs_reinit {
                    debug!("GUI update: Buffer resize in progress - needs_restart={}, needs_reinit={}", 
                           needs_restart, needs_reinit);
                }
                needs_restart || needs_reinit
            } else {
                false
            }
        };

        // Force more frequent updates during and immediately after buffer resize
        static mut RESIZE_RECOVERY_TIMER: Option<Instant> = None;
        static mut RESIZE_RECOVERY_COUNT: usize = 0;
        
        if buffer_resize_in_progress {
            unsafe {
                // Start recovery timer when resize begins
                RESIZE_RECOVERY_TIMER = Some(Instant::now());
                RESIZE_RECOVERY_COUNT = 0;
                debug!("GUI detected buffer resize in progress - starting recovery timer");
            }
        }
        
        // Check if we're in the recovery period after a resize
        let in_recovery_period = unsafe {
            if let Some(timer) = RESIZE_RECOVERY_TIMER {
                if !buffer_resize_in_progress && timer.elapsed() < Duration::from_secs(5) {
                    // We're in recovery period - resize flags cleared but still need frequent updates
                    RESIZE_RECOVERY_COUNT += 1;
                    debug!("GUI in post-resize recovery period: {}ms elapsed, count={}", 
                           timer.elapsed().as_millis(),
                           RESIZE_RECOVERY_COUNT);
                    true
                } else if !buffer_resize_in_progress && timer.elapsed() >= Duration::from_secs(5) {
                    // Recovery period ended
                    debug!("GUI post-resize recovery period complete after {} updates", RESIZE_RECOVERY_COUNT);
                    RESIZE_RECOVERY_TIMER = None;
                    RESIZE_RECOVERY_COUNT = 0;
                    false
                } else {
                    // Still in resize
                    buffer_resize_in_progress
                }
            } else {
                false
            }
        };

        // --- Buffer Size Debounce Check --- 
        let debounce_duration = Duration::from_millis(300);
        let mut size_to_apply: Option<usize> = None;

        if let Some(timer) = self.buffer_debounce_timer {
            if timer.elapsed() >= debounce_duration {
                // Timer has elapsed, clear it.
                self.buffer_debounce_timer = None; 
                
                if let Some(desired_val) = self.desired_buffer_size.take() { // Take ownership of the value
                    let current_size = *self.buffer_size.lock().unwrap();
                    if desired_val != current_size {
                        size_to_apply = Some(desired_val);
                        debug!("Buffer size debounce complete - applying new size: {}", desired_val);
                    }
                }
            }
        }

        // Apply the change if the debounce timer has elapsed
        if let Some(new_size) = size_to_apply {
            debug!("Calling update_buffer_size with size: {}", new_size);
            self.update_buffer_size(new_size);
        }
        // --- End Debounce Check ---

        // Try to receive the latest partials data
        let mut latest_partials: Option<PartialsData> = None;
        let mut received_count = 0;
        
        // Try to get the latest partials data
        if let Some(rx) = self.partials_rx.as_mut() {
            loop {
                match rx.try_recv() {
                    Ok(partials) => { 
                        received_count += 1;
                        debug!("GUI received partials update #{}: {} channels, {} partials per channel", 
                               received_count,
                               partials.len(), 
                               if !partials.is_empty() { partials[0].len() } else { 0 });
                        latest_partials = Some(partials); 
                    }
                    Err(broadcast::error::TryRecvError::Empty) => {
                        if received_count == 0 {
                            debug!("GUI received no new partials data this frame");
                        }
                        break;
                    },
                    Err(broadcast::error::TryRecvError::Lagged(n)) => {
                        warn!("GUI partials receiver lagged by {} messages", n);
                        break;
                    }
                    Err(broadcast::error::TryRecvError::Closed) => {
                        warn!("GUI partials channel closed");
                        self.partials_rx = None;
                        break;
                    }
                }
            }
        } else {
            debug!("GUI has no partials receiver - data flow broken");
        }

        // Process the latest partials data if we got any
        if let Some(linear_partials) = latest_partials {
            debug!("GUI processing partials data - update #{}", unsafe { PLOT_UPDATE_COUNT });
            // Convert linear magnitudes to dB for display
            let db_partials: PartialsData = linear_partials.into_iter().map(|channel_partials| {
                channel_partials.into_iter().map(|(freq, magnitude)| {
                    let db_val = if magnitude > 1e-10 { // Avoid log(0)
                        20.0 * magnitude.log10()
                    } else {
                        -100.0 // Or some other very low dB value
                    };
                    (freq, db_val.max(-100.0)) // Clamp dB to a minimum floor, e.g., -100dB
                }).collect()
            }).collect();

            // Update the shared SpectrumApp state with dB values
            if let Ok(mut spectrum) = self.spectrum.lock() {
                debug!("GUI updating spectrum display with {} channels of data", db_partials.len());
                spectrum.update_partials(db_partials);
            } else {
                error!("GUI failed to lock spectrum app for partials update");
            }
        }

        // Request continuous repaints to keep UI responsive
        // If buffer resize is in progress or in recovery period, request more frequent repaints
        if buffer_resize_in_progress || in_recovery_period {
            if buffer_resize_in_progress {
                debug!("GUI requesting immediate repaint due to buffer resize operation");
            } else if in_recovery_period {
                debug!("GUI requesting immediate repaint during recovery period");
            }
            
            // Force immediate repaint during resize/recovery
            ctx.request_repaint();
            self.last_repaint = Instant::now();
        } else {
            // Normal operation - throttle repaints to avoid excessive CPU usage
            let now = Instant::now();
            if now.duration_since(self.last_repaint) > Duration::from_millis(50) {
                ctx.request_repaint();
                self.last_repaint = now;
            }
        }

        // Throttling: Limit repaint to at most 10 times per second (every 100 ms)
        let now = Instant::now();
        if now.duration_since(self.last_repaint) >= Duration::from_millis(100) {
            ctx.request_repaint();
            self.last_repaint = now;
        }

        // Force continuous updates every 100 ms
        ctx.request_repaint_after(Duration::from_millis(16)); // ~60 FPS

        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            // --- Preset Management UI ---
            ui.horizontal(|ui| {
                ui.label("Presets:");
                // Add Preset Button
                if ui.button("+").clicked() {
                    self.is_adding_preset = true;
                    self.new_preset_name.clear();
                }

                // Preset Dropdown / Add Text Field
                if self.is_adding_preset {
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.new_preset_name)
                            .desired_width(120.0)
                            .hint_text("New preset name..."),
                    );
                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        let name_to_save = self.new_preset_name.trim();
                        if !name_to_save.is_empty() {
                            if self.preset_manager.presets.contains_key(name_to_save) {
                                // Preset exists, show confirmation
                                self.show_overwrite_confirmation = true;
                                self.preset_to_overwrite = name_to_save.to_string();
                            } else {
                                // New preset, save directly
                                let new_preset = self.capture_current_preset();
                                self.preset_manager.presets.insert(name_to_save.to_string(), new_preset);
                                if let Err(e) = self.preset_manager.save() {
                                    error!("Failed to save preset: {}", e);
                                }
                                self.selected_preset_name = name_to_save.to_string();
                            }
                        }
                        self.is_adding_preset = false;
                    }
                    if response.lost_focus() {
                        self.is_adding_preset = false;
                    }
                } else {
                    let mut selected_name = self.selected_preset_name.clone();
                    let mut reselected = false;
                    egui::ComboBox::from_id_source("preset_selector")
                        .selected_text(selected_name.clone())
                        .show_ui(ui, |ui| {
                            for name in self.preset_manager.presets.keys() {
                                // The selectable_value response's `clicked()` method detects any click.
                                if ui.selectable_value(&mut selected_name, name.clone(), name.clone()).clicked() {
                                    reselected = true; // Mark that a selection was made
                                }
                            }
                        });

                    // If a selection was clicked (even if it's the same one), reload it.
                    if reselected {
                        self.load_preset(&selected_name);
                        self.selected_preset_name = selected_name;
                    }
                }

                // Delete Preset Button
                if ui.add_enabled(self.selected_preset_name != "default", egui::Button::new("-")).clicked() {
                    self.show_delete_confirmation = true;
                }
            });

            // Delete Confirmation Dialog
            if self.show_delete_confirmation {
                egui::Window::new("Confirm Deletion")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.label(format!("Are you sure you want to delete the '{}' preset?", self.selected_preset_name));
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            if ui.button("Yes").clicked() {
                                self.preset_manager.presets.remove(&self.selected_preset_name);
                                if let Err(e) = self.preset_manager.save() {
                                    error!("Failed to save presets after deletion: {}", e);
                                }
                                self.selected_preset_name = "default".to_string();
                                self.load_preset("default");
                                self.show_delete_confirmation = false;
                            }
                            if ui.button("No").clicked() {
                                self.show_delete_confirmation = false;
                            }
                        });
                    });
            }

            // Overwrite Confirmation Dialog
            if self.show_overwrite_confirmation {
                egui::Window::new("Confirm Overwrite")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.label(format!("A preset named '{}' already exists. Overwrite it?", self.preset_to_overwrite));
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            if ui.button("Yes").clicked() {
                                let updated_preset = self.capture_current_preset();
                                self.preset_manager.presets.insert(self.preset_to_overwrite.clone(), updated_preset);
                                if let Err(e) = self.preset_manager.save() {
                                    error!("Failed to save overwritten preset: {}", e);
                                }
                                self.selected_preset_name = self.preset_to_overwrite.clone();
                                self.show_overwrite_confirmation = false;
                                self.preset_to_overwrite.clear();
                            }
                            if ui.button("No").clicked() {
                                self.show_overwrite_confirmation = false;
                                self.preset_to_overwrite.clear();
                            }
                        });
                    });
            }

            let mut size_changed = false;

            // 1) First row of sliders
            {
                let mut fft_config = self.fft_config.lock().unwrap();
                ui.horizontal(|ui| {
                    ui.label("Min Frequency:");
                    let buffer_size = *self.buffer_size.lock().unwrap();
                    let nyquist_limit = (buffer_size as f64 / 2.0).min(*MAX_FREQ);
                    ui.add(egui::Slider::new(&mut fft_config.min_frequency, MIN_FREQ..=nyquist_limit).text("Hz"));
                    
                    ui.label("Max Frequency:");
                    ui.add(egui::Slider::new(&mut fft_config.max_frequency, 0.0..=nyquist_limit).text("Hz"));

                    // Ensure min_frequency is always less than max_frequency
                    if fft_config.min_frequency >= fft_config.max_frequency {
                        fft_config.min_frequency = fft_config.max_frequency * 0.5;
                        debug!("Adjusted min_frequency to {} Hz (half of max_frequency)", fft_config.min_frequency);
                    }

                    ui.label("Magnitude Threshold:");
                    ui.add(egui::Slider::new(&mut fft_config.magnitude_threshold, 0.0..=60.0));
                });
            }

            // 2) Second row with frames, buffer size, and averaging
            ui.horizontal(|ui| {
                // Buffer size slider
                {
                    let current_actual_buffer_size = *self.buffer_size.lock().unwrap();
                    let display_size = self.desired_buffer_size.unwrap_or(current_actual_buffer_size);
                    let mut buffer_log_slider = (display_size as f32).log2().round() as u32;

                    ui.label("Buffer Size:");
                    let min_power = (MIN_BUFFER_SIZE as f32).log2() as u32;
                    let max_power = (MAX_BUFFER_SIZE as f32).log2() as u32;
                    if ui
                        .add(egui::Slider::new(&mut buffer_log_slider, min_power..=max_power)
                            .custom_formatter(|n, _| {
                                format!("{}", 1_usize << (n as usize))
                            }))
                        .changed()
                    {
                        let new_size = 1 << buffer_log_slider;
                        // self.update_buffer_size(new_size); // REMOVED direct call
                        // Instead, set desired size and timer for debouncing
                        self.desired_buffer_size = Some(new_size);
                        self.buffer_debounce_timer = Some(Instant::now());
                        size_changed = true; // Keep UI responsive even if change is debounced
                    }
                    ui.label("samples");
                }

                // Min freq spacing slider
                {
                    let mut fft_config = self.fft_config.lock().unwrap();
                    ui.label("Min Freq Spacing:");
                    ui.add(egui::Slider::new(&mut fft_config.min_freq_spacing, 0.0..=500.0).text("Hz"));
                }
                
                // Window Type section
                {
                    let mut fft_config = self.fft_config.lock().unwrap();
                    ui.label("Window Type:");
                    egui::ComboBox::from_id_source("window_type")
                        .selected_text(format!("{:?}", fft_config.window_type))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut fft_config.window_type, WindowType::Rectangular, "Rectangular");
                            ui.selectable_value(&mut fft_config.window_type, WindowType::Hanning, "Hanning");
                            ui.selectable_value(&mut fft_config.window_type, WindowType::Hamming, "Hamming");
                            ui.selectable_value(&mut fft_config.window_type, WindowType::BlackmanHarris, "Blackman-Harris");
                            ui.selectable_value(&mut fft_config.window_type, WindowType::FlatTop, "Flat Top");
                            if ui.selectable_value(&mut fft_config.window_type, WindowType::Kaiser(4.0), "Kaiser").clicked() {
                                fft_config.window_type = WindowType::Kaiser(4.0);
                            }
                        });
                        
                        // Kaiser beta slider appears if Kaiser window is selected
                        if let WindowType::Kaiser(_) = fft_config.window_type {
                            ui.label("Kaiser β:");
                            let mut beta = match fft_config.window_type {
                                WindowType::Kaiser(b) => b,
                                _ => 4.0,
                            };
                            if ui.add(egui::Slider::new(&mut beta, 0.0..=20.0)).changed() {
                                fft_config.window_type = WindowType::Kaiser(beta);
                            }
                        }
                }
            });

            // 3) Sliders for Y scale, alpha, bar width
            ui.horizontal(|ui| {
                ui.label("Y Max:");
                ui.add(egui::Slider::new(&mut self.y_scale, 0.0..=100.0).text(""));
                ui.label("Alpha:");
                ui.add(egui::Slider::new(&mut self.alpha, 0..=255).text(""));
                ui.label("Bar Width:");
                ui.add(egui::Slider::new(&mut self.bar_width, 1.0..=10.0).text(""));
                
                ui.separator();
                
                // Show FFT checkbox moved from row 2 to here
                ui.checkbox(&mut self.show_line_plot, "Show FFT");
                ui.checkbox(&mut self.show_spectrograph, "Show Spectrograph");
                ui.checkbox(&mut self.show_results, "Show Results");
                ui.separator();
            });

            // 4) Volume and Smoothing row + Crosstalk checkbox + Frequency Scale
            ui.horizontal(|ui| {
                // Volume slider (exactly matching update rate slider pattern)
                ui.label("Volume:");
                if let Ok(mut resynth_config) = self.resynth_config.lock() {
                    if ui.add(
                        egui::Slider::new(&mut resynth_config.gain, 0.0..=1.0)
                            .text("")
                    ).changed() {
                        self.gui_param_tx.send(GuiParameter::Gain(resynth_config.gain)).unwrap_or_else(|e| error!("Failed to send Gain update: {}", e));
                        self.gain_update_tx.send(resynth_config.gain).unwrap_or_else(|e| error!("Failed to send instant gain update: {}", e));
                    }
                }
             
            
                ui.separator();
                
                // Add Frequency Scale control right after Volume
                ui.label("Freq Scale:");
                if let Ok(mut resynth_config) = self.resynth_config.lock() {
                    let mut freq_scale_changed = false;
                    ui.horizontal(|ui| {
                        // Down button (divide by 2 = one octave down)
                        if ui.button("▼").clicked() {
                            resynth_config.freq_scale /= 2.0;
                            freq_scale_changed = true;
                        }
                        
                        // Display current value with 2 decimal places in a compact field
                        let mut freq_text = format!("{:.2}", resynth_config.freq_scale);
                        if ui.add(egui::TextEdit::singleline(&mut freq_text)
                            .desired_width(50.0)  // Make the text field narrower
                            .hint_text("1.00"))   // Add hint text
                            .changed() 
                        {
                            // Try to parse the new value
                            if let Ok(new_value) = freq_text.parse::<f32>() {
                                if new_value > 0.0 {  // Ensure positive value
                                    resynth_config.freq_scale = new_value;
                                    freq_scale_changed = true;
                                }
                            }
                        }
                        
                        // Up button (multiply by 2 = one octave up)
                        if ui.button("▲").clicked() {
                            resynth_config.freq_scale *= 2.0;
                            freq_scale_changed = true;
                        }
                    });
                    if freq_scale_changed {
                        self.gui_param_tx.send(GuiParameter::FreqScale(resynth_config.freq_scale)).unwrap_or_else(|e| error!("Failed to send FreqScale update: {}", e));
                    }
                }
                
                ui.separator();
                
                // Move Crosstalk checkbox to this row
                let mut fft_config = self.fft_config.lock().unwrap();
                ui.checkbox(&mut fft_config.crosstalk_enabled, "Crosstalk Filtering");
            });

            // The 'if' condition below is to prevent empty rows from being created when crosstalk is disabled.
            // This does not reorder any elements, it just hides the controls when they are not applicable.
            if self.fft_config.lock().unwrap().crosstalk_enabled {
                // 5) Crosstalk basic parameters row + Harmonic Tolerance
                ui.horizontal(|ui| {
                    let mut fft_config = self.fft_config.lock().unwrap();
                    
                    // Only show crosstalk parameters if enabled
                    if fft_config.crosstalk_enabled {
                        ui.label("Threshold:");
                        ui.add(egui::Slider::new(&mut fft_config.crosstalk_threshold, 0.0..=1.0));
                        ui.label("Reduction:");
                        ui.add(egui::Slider::new(&mut fft_config.crosstalk_reduction, 0.0..=1.0));
                        
                        // Move Harmonic Tolerance to this row
                        ui.separator();
                        ui.label("Harmonic Tolerance:");
                        ui.add(egui::Slider::new(&mut fft_config.harmonic_tolerance, 0.01..=0.10)
                            .logarithmic(true));
                    }
                });

                // 6) Advanced Crosstalk controls - only show if crosstalk is enabled
                ui.horizontal(|ui| {
                    let mut fft_config = self.fft_config.lock().unwrap();
                    
                    // Use the helper method
                    let nyquist_limit = self.get_nyquist_limit();
                    
                    // Only show advanced parameters if enabled
                    if fft_config.crosstalk_enabled {
                        ui.label("Root Freq Min:");
                        ui.add(egui::Slider::new(&mut fft_config.root_freq_min, 0.0..=nyquist_limit as f32));
                        
                        ui.label("Root Freq Max (Advanced):");
                        ui.add(
                            egui::Slider::new(&mut fft_config.root_freq_max, 0.0..=nyquist_limit as f32)
                                .logarithmic(true)
                        );
                        
                        ui.label("Frequency Match:");
                        ui.add(egui::Slider::new(&mut fft_config.freq_match_distance, 1.0..=20.0));
                    }
                });
            }

            // 7) Resynth update timer control
            ui.horizontal(|ui| {
                if let Ok(mut resynth_config) = self.resynth_config.lock() {
                    ui.label("Resynth Update Rate:");
                    if ui.add(
                        egui::Slider::new(&mut resynth_config.update_rate, 0.01..=30.0)
                            .text("seconds")
                            .logarithmic(true)
                    ).changed() {
                        self.gui_param_tx.send(GuiParameter::UpdateRate(resynth_config.update_rate)).unwrap_or_else(|e| error!("Failed to send UpdateRate update: {}", e));
                    }
                }
            }); 

            // Handle max frequency adjustment if buffer size changed
            if size_changed {
                let nyquist_limit = self.get_nyquist_limit() as f64;
                let mut fft_config = self.fft_config.lock().unwrap();
                
                // Update max_frequency if needed
                if fft_config.max_frequency > nyquist_limit {
                    fft_config.max_frequency = nyquist_limit;
                    debug!("Adjusted max_frequency to nyquist limit: {} Hz", nyquist_limit);
                    
                    // Also update root_freq_max to match the same nyquist limit
                    if fft_config.root_freq_max > nyquist_limit as f32 {
                        fft_config.root_freq_max = nyquist_limit as f32;
                        debug!("Adjusted root_freq_max to nyquist limit: {} Hz", nyquist_limit);
                    }
                }
            }

            // 8) Plot logic
            let absolute_values = {
                let spectrum = self.spectrum.lock().unwrap();
                spectrum.absolute_values.clone()  // Contains dB values used for both plotting and display
            };

            // Bar charts with static legend names - always show all channels
            let all_bar_charts: Vec<BarChart> = (0..absolute_values.len())
                .map(|channel| {
                    let channel_partials = &absolute_values[channel];
                    let mut bars: Vec<egui::plot::Bar> = channel_partials
                        .iter()
                        // Filter out non-positive frequencies and values (assuming dB)
                        .filter(|&&(freq, db_val)| freq > 0.0 && db_val > -f32::INFINITY) // Use -inf for dB check
                        .map(|&(freq, db_val)| {
                            // Use dB value directly for plotting
                            egui::plot::Bar::new(freq as f64, db_val as f64)
                                .width(self.bar_width as f64)
                        })
                        .collect();

                    // If no visible bars, add a single invisible bar
                    if bars.is_empty() {
                        bars.push(egui::plot::Bar::new(0.0, 0.0).width(0.0));
                    }

                    let color = self.colors[channel % self.colors.len()]
                        .linear_multiply(self.alpha as f32 / 255.0);

                    BarChart::new(bars)
                        .name(format!("Channel {}", channel + 1))
                        .color(color)
                })
                .collect();

            // Line plots without legend names - more memory efficient
            let all_line_plots: Vec<egui::plot::Line> = if self.show_line_plot {
                let spectrum = self.spectrum.lock().unwrap();
                let fft_data = spectrum.get_fft_line_data();
                
                fft_data.iter().enumerate().map(|(channel, points)| {
                    let color = self.colors[channel % self.colors.len()]
                        .linear_multiply(self.alpha as f32 / 255.0);

                    egui::plot::Line::new(
                        points.iter()
                            .map(|&(freq, mag)| [freq as f64, mag as f64])
                            .collect::<Vec<[f64; 2]>>()
                    )
                    .color(color)
                }).collect()
            } else {
                Vec::new()
            };

            let max_freq = {
                let fft = self.fft_config.lock().unwrap();
                fft.max_frequency
            };

            // Move the style modification outside the closure to avoid borrowing conflicts
            ui.style_mut().text_styles.insert(
                TextStyle::Monospace,
                FontId::new(14.0, FontFamily::Proportional)
            );
            ui.style_mut().text_styles.insert(
                TextStyle::Body,
                FontId::new(14.0, FontFamily::Proportional)
            );

            Plot::new("spectrum_plot")
                .legend(Legend::default())
                .view_aspect(6.0)
                .include_x(0.0)
                .include_x(max_freq as f64)
                .include_y(0.0)
                .include_y(self.y_scale as f64)
                .x_axis_formatter(|value, _range| format!("{} Hz", value as i32))
                .y_axis_formatter(|value, _range| format!("{} dB", value as i32))
                .y_grid_spacer(uniform_grid_spacer(|_input| [5.0, 10.0, 20.0]))  // More frequent grid lines
                .show_axes([true, true])
                .show_x(true)
                .show_y(true)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .allow_double_click_reset(false)
                .label_formatter(|name, value| {
                    if !name.is_empty() {
                        format!("{}: {:.1} Hz, {:.1} dB", name, value.x, value.y)
                    } else {
                        String::new()
                    }
                })
                .show(ui, |plot_ui| {
                    // 1. Clone the current style so we can mutate safely (compatible with older egui versions)
                    let mut style = (*plot_ui.ctx().style()).clone();

                    // 2. Adjust the desired text style (Body)
                    if let Some(body_style) = style.text_styles.get_mut(&TextStyle::Body) {
                        *body_style = FontId::new(14.0, FontFamily::Proportional);
                    }

                    // 3. Override the main text color (axis labels, etc.) with pure white for maximum contrast
                    style.visuals.override_text_color = Some(Color32::WHITE);
                    
                    // 4. Make sure widget foreground strokes are also bright for readability
                    style.visuals.widgets.noninteractive.fg_stroke.color = Color32::WHITE;
                    style.visuals.widgets.inactive.fg_stroke.color = Color32::WHITE;
                    style.visuals.widgets.hovered.fg_stroke.color = Color32::WHITE;
                    style.visuals.widgets.active.fg_stroke.color = Color32::WHITE;
                    style.visuals.widgets.open.fg_stroke.color = Color32::WHITE;
                    
                    // 5. Apply the modified style back to the context
                    plot_ui.ctx().set_style(style);
                    
                    // Explicitly set plot bounds to match min and max frequency settings
                    let fft_config = self.fft_config.lock().unwrap();
                    let min_freq = fft_config.min_frequency;
                    let max_freq = fft_config.max_frequency;
                    let bounds = egui::plot::PlotBounds::from_min_max(
                        [min_freq, 0.0], // Min X, Min Y
                        [max_freq, self.y_scale as f64] // Max X, Max Y
                    );
                    debug!("Setting plot bounds to X: [{} to {}] Hz, Y: [0 to {}]", 
                           min_freq, max_freq, self.y_scale);
                    
                    // Add detailed plot update debug log with buffer resize status
                    let buffer_resize_status = if let Ok(buffer) = self.audio_buffer.read() {
                        format!("needs_restart={}, needs_reinit={}", 
                               buffer.needs_restart(), buffer.needs_reinit())
                    } else {
                        "buffer_lock_failed".to_string()
                    };
                    
                    debug!(target: "audio_streaming::plot", 
                           "Plot Render: cycle={}, max_freq={}, buffer_resize={}, received_data={}, partials_count={}", 
                           unsafe { PLOT_UPDATE_COUNT }, 
                           max_freq, 
                           buffer_resize_status,
                           received_count > 0,
                           all_bar_charts.len());
                    
                    plot_ui.set_plot_bounds(bounds);

                    // Plot the data using the new style
                    for bar_chart in all_bar_charts {
                        plot_ui.bar_chart(bar_chart);
                    }
                    if self.show_line_plot {
                        for line in all_line_plots {
                            plot_ui.line(line);
                        }
                    }
                });

            // Optimized spectrograph update logic
            if self.show_spectrograph {
                let current_time = Instant::now();
                let current_timestamp = chrono::Local::now();
                let elapsed = current_time.duration_since(self.start_time.as_ref().clone());
                let start_timestamp = current_timestamp - chrono::Duration::from_std(elapsed).unwrap_or_default();
                
                // Get the oldest timestamp from the history
                let (earliest_time, latest_time) = {
                    let history = self.spectrograph_history.lock().unwrap();
                    if let Some(oldest) = history.front() {
                        let oldest_time = oldest.time;
                        (oldest_time, oldest_time + 5.0) // Show 5 seconds from oldest timestamp
                    } else {
                        let current = elapsed.as_secs_f64();
                        (current, current + 5.0)
                    }
                };
                
                let (min_freq, max_freq) = {
                    let fft = self.fft_config.lock().unwrap();
                    let buffer_size = *self.buffer_size.lock().unwrap();
                    let max = (fft.max_frequency as f32).min(buffer_size as f32 / 2.0);
                    (fft.min_frequency as f32, max)
                };

                Plot::new("spectrograph_plot")
                    .legend(Legend::default())
                    .view_aspect(6.0)
                    .include_y(min_freq as f64)
                    .include_y(max_freq as f64)
                    .x_axis_formatter(move |value, _range| {
                        let timestamp = start_timestamp + chrono::Duration::milliseconds((value * 1000.0) as i64);
                        format!("{}", timestamp.format("%H:%M:%S"))
                    })
                    .y_axis_formatter(|value, _range| format!("{} Hz", value as i32))
                    .show_axes([true, true])
                    .show_x(true)
                    .show_y(true)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .allow_double_click_reset(false)
                    .label_formatter(|name, value| {
                        if !name.is_empty() {
                            format!("{}: {:.1} Hz, {:.1} s", name, value.y, value.x)
                        } else {
                            String::new()
                        }
                    })
                    .show(ui, |plot_ui| {
                        let history = self.spectrograph_history.lock().unwrap();
                        if !history.is_empty() {
                            plot_ui.set_plot_bounds(egui::plot::PlotBounds::from_min_max(
                                [earliest_time * 1000.0, min_freq as f64],
                                [latest_time * 1000.0, max_freq as f64]
                            ));

                            for slice in history.iter() {
                                if slice.time >= earliest_time && slice.time <= latest_time {
                                    // slice.data contains (freq: f64, unnormalized_linear_magnitude: f32)
                                    for &(freq, unnormalized_magnitude_f32) in &slice.data { 
                                        let unnormalized_magnitude = unnormalized_magnitude_f32 as f64;

                                        // 1. Calculate the value to scale: 20 * log10(unnormalized magnitude)
                                        let value_to_scale = if unnormalized_magnitude > 1e-10 { // Avoid log(0)
                                            20.0 * unnormalized_magnitude.log10()
                                        } else {
                                            // Map silence/low values to ensure intensity is 0
                                            0.0 
                                        };

                                        // 2. Calculate intensity by normalizing value_to_scale against y_scale
                                        // Intensity = 0.0 if value_to_scale <= 0
                                        // Intensity = 1.0 if value_to_scale >= y_scale
                                        let intensity = (value_to_scale / self.y_scale as f64).clamp(0.0, 1.0);
                                        
                                        // 3. Apply color based on intensity (Blue -> Green -> Red)
                                        let color = egui::Color32::from_rgb(
                                            (255.0 * intensity) as u8, // Red increases with intensity
                                            (255.0 * (1.0 - (intensity - 0.5).abs() * 2.0).max(0.0)) as u8, // Green peaks at mid-intensity
                                            (255.0 * (1.0 - intensity)) as u8, // Blue decreases with intensity
                                        );

                                        plot_ui.points(
                                            egui::plot::Points::new(vec![[slice.time * 1000.0, freq]])
                                                .color(color)
                                                .radius(2.0)
                                        );
                                    }
                                }
                            }
                        }
                    });
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
                if self.show_results {
                    let display = SpectralDisplay::new(&absolute_values);
                    for line in display.format_all() {
                        ui.label(egui::RichText::new(line).size(12.0));
                    }
                }
            });
        });

        // Instead of dynamic resizing, let the user decide the window size
        // This avoids issues with different screen resolutions and DPI settings
        // The window will be scrollable if content doesn't fit
    }
}

// This section is protected. Do not alter unless permission is requested by you and granted by me.
#[allow(dead_code)]
pub fn run_native(
    app_name: &str,
    native_options: NativeOptions,
    app_creator: Box<dyn FnOnce(&eframe::CreationContext<'_>) -> Box<MyApp>>,
) -> Result<(), eframe::Error> {
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    eframe::run_native(
        app_name,
        native_options,
        Box::new(move |cc| {
            let mut app = app_creator(cc);
            app.shutdown_flag = shutdown_flag.clone();  // Directly assign to MyApp
            app as Box<dyn eframe::App>
        }),
    )
}

// Total line count: 259

// Update the format_all method in display.rs to use the configured number of partials
pub mod display_utils {
    // This helper function formats partials with any number of partials
    pub fn format_partials(values: &Vec<(f32, f32)>, num_partials: usize) -> String {
        // Format exactly num_partials values, creating a single horizontal string
        let magnitudes = (0..num_partials)
            .map(|i| {
                if i < values.len() {
                    let (freq, db_val) = values[i];
                    // Format dB value directly, as it is now pre-calculated
                    if db_val.is_finite() && freq > 0.0 {
                        format!("({:.2}, {:.0})", freq, db_val)
                    } else {
                        "(0.0, -)".to_string() // Display placeholder for invalid/silent values
                    }
                } else {
                    "(0.0, -)".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", "); // Join into a single comma-separated string
        
        magnitudes
    }
}
