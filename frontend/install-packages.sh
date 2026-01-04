#!/bin/bash
echo "Installing required packages for the frontend..."
echo ""

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Installing @craco/craco for webpack configuration overrides..."
npm install @craco/craco --save

echo ""
echo "Done! Now you can start the application with: npm start"