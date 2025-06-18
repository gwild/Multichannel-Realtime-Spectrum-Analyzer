fn main() {
    // Add system library paths - these may differ by platform
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
        
        // Link to Linux-specific libraries
        println!("cargo:rustc-link-lib=static=portaudio");
        println!("cargo:rustc-link-lib=jack");
        println!("cargo:rustc-link-lib=asound");  // ALSA - Linux only
        
        // Set rpath for Linux
        println!("cargo:rustc-link-arg=-Wl,-rpath=/usr/lib/aarch64-linux-gnu");
    }

    #[cfg(target_os = "macos")]
    {
        // Try to find Homebrew-installed PortAudio
        println!("cargo:rustc-link-search=native=/usr/local/Cellar/portaudio/19.7.0/lib");
        
        // Link to macOS-specific libraries
        println!("cargo:rustc-link-lib=portaudio");
        
        // macOS frameworks will be automatically linked by the portaudio crate
    }
    
    #[cfg(target_os = "windows")]
    {
        // Windows linking
        println!("cargo:rustc-link-lib=static=portaudio");
    }
}