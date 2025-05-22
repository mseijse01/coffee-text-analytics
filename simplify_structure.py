#!/usr/bin/env python3
"""
Simplify the coffee-text-analytics project structure.
This script:
1. Flattens redundant directories (removes nested coffee-text-analytics folder)
2. Creates a simpler directory structure
3. Preserves essential files
"""

import os
import shutil
from pathlib import Path
import sys


def simplify_structure():
    """Simplify and reorganize the project structure."""
    # Base directory is the current working directory
    base_dir = Path(os.getcwd())

    print(f"Starting structure simplification in: {base_dir}")

    # New simpler directory structure
    new_dirs = [
        "data/raw",  # Raw data files
        "data/processed",  # Processed data
        "models",  # Model files (simplified from data/models)
        "src",  # Core implementation
        "notebooks",  # For Jupyter notebooks
        "output",  # For results/figures (simplified from results/figures)
    ]

    # Create temporary directory for moving important files
    temp_dir = base_dir / "temp_reorg"
    os.makedirs(temp_dir, exist_ok=True)

    # Files that should definitely be preserved
    files_to_preserve = [
        "requirements.txt",
        "README.md",
        ".gitignore",
        "Leveraging_Text_Analytics_and_Predictive_Modeling_to_Analyze_Consumer_Coffee_Reviews__A_Data_Driven_Approach - Final Version.pdf",
    ]

    # Step 1: Handle the redundant coffee-text-analytics folder if it exists
    nested_dir = base_dir / "coffee-text-analytics"
    if nested_dir.exists() and nested_dir.is_dir():
        print("Found redundant nested coffee-text-analytics folder. Fixing...")

        # Move all files from nested directory to temp directory
        for item in os.listdir(nested_dir):
            src_path = nested_dir / item
            dst_path = temp_dir / item

            if os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                print(f"Preserved directory from nested folder: {item}")
            else:
                shutil.copy2(src_path, dst_path)
                print(f"Preserved file from nested folder: {item}")

    # Step 2: Preserve important files from the root directory
    for filename in files_to_preserve:
        file_path = base_dir / filename
        if file_path.exists():
            shutil.copy2(file_path, temp_dir / filename)
            print(f"Preserved file from root: {filename}")

    # Step 3: Handle existing src directory
    src_dir = base_dir / "src"
    if src_dir.exists() and src_dir.is_dir():
        # Just copy the entire src directory
        if os.path.exists(temp_dir / "src"):
            shutil.rmtree(temp_dir / "src")
        shutil.copytree(src_dir, temp_dir / "src")
        print("Preserved src directory")

    # Step 4: Handle existing data
    data_dir = base_dir / "data"
    if data_dir.exists() and data_dir.is_dir():
        if not (temp_dir / "data").exists():
            os.makedirs(temp_dir / "data")

        # Copy raw and processed subdirectories if they exist
        for subdir in ["raw", "processed"]:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                if os.path.exists(temp_dir / "data" / subdir):
                    shutil.rmtree(temp_dir / "data" / subdir)
                shutil.copytree(subdir_path, temp_dir / "data" / subdir)
                print(f"Preserved data/{subdir} directory")

        # Move models to root level models directory
        models_dir = data_dir / "models"
        if models_dir.exists():
            if os.path.exists(temp_dir / "models"):
                shutil.rmtree(temp_dir / "models")
            shutil.copytree(models_dir, temp_dir / "models")
            print("Moved data/models to models directory")

    # Step 5: Move results to output
    results_dir = base_dir / "results"
    if results_dir.exists() and results_dir.is_dir():
        if not (temp_dir / "output").exists():
            os.makedirs(temp_dir / "output")

        # Copy figures if they exist
        figures_dir = results_dir / "figures"
        if figures_dir.exists():
            if os.path.exists(temp_dir / "output" / "figures"):
                shutil.rmtree(temp_dir / "output" / "figures")
            shutil.copytree(figures_dir, temp_dir / "output" / "figures")
            print("Moved results/figures to output/figures directory")

    # Step 6: Create notebooks directory (if it doesn't exist or has content)
    notebooks_dir = base_dir / "notebooks"
    if notebooks_dir.exists() and notebooks_dir.is_dir():
        if os.path.exists(temp_dir / "notebooks"):
            shutil.rmtree(temp_dir / "notebooks")
        shutil.copytree(notebooks_dir, temp_dir / "notebooks")
        print("Preserved notebooks directory")
    else:
        # Create empty notebooks directory
        os.makedirs(temp_dir / "notebooks", exist_ok=True)
        print("Created empty notebooks directory")

    # Step 7: Clean everything except temp directory and this script
    for item in os.listdir(base_dir):
        if item != "temp_reorg" and item != "simplify_structure.py":
            item_path = base_dir / item
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed directory: {item}")
            else:
                os.remove(item_path)
                print(f"Removed file: {item}")

    # Step 8: Copy everything back from temp directory
    for item in os.listdir(temp_dir):
        src_path = temp_dir / item
        dst_path = base_dir / item
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            print(f"Restored directory: {item}")
        else:
            shutil.copy2(src_path, dst_path)
            print(f"Restored file: {item}")

    # Step 9: Remove temp directory
    shutil.rmtree(temp_dir)
    print("Removed temporary directory")

    # Step 10: Create any missing directories from our new structure
    for dir_path in new_dirs:
        os.makedirs(base_dir / dir_path, exist_ok=True)

    print("Project structure simplified successfully.")

    # Display the final project structure
    print("\nFinal project structure:")
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(str(base_dir), "").count(os.sep)
        indent = " " * 4 * level
        rel_path = os.path.relpath(root, base_dir) if root != base_dir else "."
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for file in files:
            if file != "simplify_structure.py":  # Don't show this script in the output
                print(f"{sub_indent}{file}")


if __name__ == "__main__":
    simplify_structure()
