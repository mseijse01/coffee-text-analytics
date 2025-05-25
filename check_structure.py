#!/usr/bin/env python3
"""
Check if the project structure is ready for GitHub.
"""

import os
import sys
from pathlib import Path


def check_structure():
    """Check if the project structure is ready for GitHub."""
    print("Checking project structure for GitHub readiness...")

    # Define required directories
    required_dirs = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "output/figures",
        "src/data",
        "src/features",
        "src/models",
        "src/utils",
        "src/visualization",
        "src/config",
    ]

    # Define required files
    required_files = [
        "README.md",
        "requirements.txt",
        "main.py",
        ".gitignore",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "src/visualization/__init__.py",
        "src/config/__init__.py",
    ]

    # Define required .gitkeep files
    required_gitkeeps = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "models/.gitkeep",
        "notebooks/.gitkeep",
        "output/figures/.gitkeep",
    ]

    # Check if required directories exist
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)

    if missing_dirs:
        print("❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
    else:
        print("✅ All required directories exist")

    # Check if required files exist
    missing_files = []
    for file_path in required_files:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    else:
        print("✅ All required files exist")

    # Check if required .gitkeep files exist
    missing_gitkeeps = []
    for gitkeep_path in required_gitkeeps:
        if not os.path.isfile(gitkeep_path):
            missing_gitkeeps.append(gitkeep_path)

    if missing_gitkeeps:
        print("❌ Missing .gitkeep files:")
        for gitkeep_path in missing_gitkeeps:
            print(f"  - {gitkeep_path}")
    else:
        print("✅ All required .gitkeep files exist")

    # Check for .DS_Store files
    ds_store_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == ".DS_Store":
                ds_store_files.append(os.path.join(root, file))

    if ds_store_files:
        print("❌ Found .DS_Store files:")
        for ds_store_file in ds_store_files:
            print(f"  - {ds_store_file}")
    else:
        print("✅ No .DS_Store files found")

    # Check if __pycache__ directories are in .gitignore
    with open(".gitignore", "r") as f:
        gitignore_content = f.read()

    if "__pycache__/" not in gitignore_content:
        print("❌ __pycache__/ not found in .gitignore")
    else:
        print("✅ __pycache__/ is in .gitignore")

    # Overall check
    if not (
        missing_dirs
        or missing_files
        or missing_gitkeeps
        or ds_store_files
        or "__pycache__/" not in gitignore_content
    ):
        print("\n✅ Project structure is ready for GitHub!")
    else:
        print("\n❌ Project structure needs fixing before pushing to GitHub")


if __name__ == "__main__":
    check_structure()
