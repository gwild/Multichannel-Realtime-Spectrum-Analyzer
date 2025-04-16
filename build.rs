fn main() {
    // Disable JACK support at compile time since we don't need it on Windows
    println!("cargo:rustc-env=PA_DISABLE_JACK=1");
    println!("cargo:rustc-env=PA_USE_JACK=0");
    println!("cargo:rustc-env=PA_USE_WASAPI=1"); // Use WASAPI on Windows
    println!("cargo:rustc-env=PA_BUILD_SHARED=0");

    // Add the path where JACK is installed
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    
    // Link against JACK
    println!("cargo:rustc-link-lib=jack");

    // Link against PortAudio
    println!("cargo:rustc-link-lib=static=portaudio");

    // Windows-specific settings
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=ole32");
        println!("cargo:rustc-link-lib=setupapi");
        println!("cargo:rustc-link-lib=winmm");
    }
} 