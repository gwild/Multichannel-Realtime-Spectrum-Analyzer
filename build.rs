fn main() {
    // Disable JACK support at compile time
    println!("cargo:rustc-cfg=pa_disable_jack");
    println!("cargo:rustc-env=PA_DISABLE_JACK=1");
    println!("cargo:rustc-env=PA_USE_JACK=0");
    println!("cargo:rustc-env=PA_USE_ALSA=1");
    println!("cargo:rustc-env=PA_BUILD_SHARED=0");

    // System library paths
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/local/lib");  // For locally built libraries
    println!("cargo:rustc-link-search=native=/usr/lib/arm-linux-gnueabihf");  // Alternative ARM path
    
    // Link against ALSA
    println!("cargo:rustc-link-lib=asound");
    
    // Link against JACK with full path specification
    println!("cargo:rustc-link-search=native=/usr/lib/jack");  // JACK specific directory
    println!("cargo:rustc-link-lib=jack");
    
    // Link against PortAudio
    println!("cargo:rustc-link-lib=static=portaudio");

    // Set compile-time flags for PortAudio
    println!("cargo:rustc-cfg=feature=\"pa_disable_jack\"");
    println!("cargo:rustc-cfg=feature=\"pa_use_alsa\"");
} 