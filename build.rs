fn main() {
    // Configure PortAudio build without JACK support
    println!("cargo:rustc-env=PA_DISABLE_JACK=1");
    println!("cargo:rustc-cfg=pa_disable_jack");
    
    // Link against ALSA (Linux sound system)
    println!("cargo:rustc-link-lib=asound");
    
    // Configure PortAudio build options
    println!("cargo:rustc-env=PA_USE_JACK=0");
    println!("cargo:rustc-env=PA_USE_ALSA=1");
    println!("cargo:rustc-env=PA_BUILD_SHARED=0");
    
    // Link against PortAudio
    println!("cargo:rustc-link-lib=static=portaudio");
} 