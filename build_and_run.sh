#!/bin/bash

echo "Script started"

# Save all files in the current directory
echo "Saving all files..."
if command -v osascript &> /dev/null; then
    osascript -e 'tell application "System Events" to keystroke "s" using command down'
else
    echo "Warning: Couldn't automatically save files. Please ensure all files are saved manually."
fi

# Clean the project
echo "Cleaning the project..."
cargo clean

# Build the project
echo "Building the project..."
cargo build

# If build is successful, run the project
if [ $? -eq 0 ]; then
    echo "Running the project..."
    cargo run
else
    echo "Build failed. Please check the errors above."
fi

echo "Script ended"
