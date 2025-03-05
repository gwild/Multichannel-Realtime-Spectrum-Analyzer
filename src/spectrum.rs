use std::sync::{Arc, Mutex};
use log::{info, error};

impl SpectrumApp {
    // Add this method to the SpectrumApp implementation
    pub fn update_shared_partials(&self, shared_partials: &Arc<Mutex<Vec<Vec<(f32, f32)>>>>) {
        // Get the current partials data
        let partials_data = self.clone_absolute_data();
        
        // Update the shared partials
        if let Ok(mut partials) = shared_partials.lock() {
            *partials = partials_data;
            info!("Updated shared partials with new data");
        } else {
            error!("Failed to lock shared partials for update");
        }
    }
} 