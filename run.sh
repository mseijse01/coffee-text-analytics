#!/bin/bash
# Helper script for running the Coffee Text Analytics project

# Function to display help
show_help() {
  echo "Coffee Text Analytics Helper Script"
  echo "=================================="
  echo "Usage: ./run.sh [command]"
  echo ""
  echo "Available commands:"
  echo "  all          - Run the complete pipeline"
  echo "  preprocess   - Run only the data preprocessing step"
  echo "  features     - Run only the feature extraction step"
  echo "  train        - Run only the model training step"
  echo "  visualize    - Run only the visualization step"
  echo "  setup        - Set up project directories and environment"
  echo "  help         - Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./run.sh all"
  echo "  ./run.sh preprocess"
}

# Function to ensure the Python environment is activated
ensure_environment() {
  # Check if we're in a virtual environment
  if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not detected."
    
    # Check if venv exists
    if [ -d "venv" ]; then
      echo "Activating existing virtual environment..."
      source venv/bin/activate || source venv/Scripts/activate
    else
      echo "Creating and activating new virtual environment..."
      python -m venv venv
      source venv/bin/activate || source venv/Scripts/activate
      
      # Install requirements
      echo "Installing dependencies..."
      pip install -r requirements.txt
    fi
  fi
}

# Function to set up project directories
setup_project() {
  echo "Setting up project directories..."
  mkdir -p data/raw
  mkdir -p data/processed
  mkdir -p models
  mkdir -p output/figures
  mkdir -p notebooks
  
  echo "Project structure set up successfully."
}

# Ensure the environment is set up
ensure_environment

# Process command line arguments
case "$1" in
  all)
    echo "Running complete pipeline..."
    python main.py --steps all
    ;;
  preprocess)
    echo "Running preprocessing step..."
    python main.py --steps preprocess
    ;;
  features)
    echo "Running feature extraction step..."
    python main.py --steps features
    ;;
  train)
    echo "Running model training step..."
    python main.py --steps train
    ;;
  visualize)
    echo "Running visualization step..."
    python main.py --steps visualize
    ;;
  setup)
    setup_project
    ;;
  help|*)
    show_help
    ;;
esac 