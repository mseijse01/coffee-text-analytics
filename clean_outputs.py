#!/usr/bin/env python3
"""
Clean Output Directories Utility

This script provides a safe way to clean output directories before running
the coffee text analytics pipeline to ensure fresh starts without artifacts
from previous runs.

Usage:
    python clean_outputs.py [--confirm] [--dry-run] [--selective]

Options:
    --confirm    Skip confirmation prompt and proceed with cleaning
    --dry-run    Show what would be deleted without actually deleting
    --selective  Choose specific directories to clean interactively
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from config import config
except ImportError:
    print("Warning: Could not import config. Using default paths.")
    config = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_directories_to_clean():
    """
    Get list of directories that can be safely cleaned.

    Returns:
        dict: Dictionary with directory descriptions and paths
    """
    if config:
        base_paths = {
            "Output Directory": config.paths.output,
            "Models Directory": config.paths.models,
            "Processed Data": config.paths.processed,
        }
    else:
        # Fallback to default paths
        base_paths = {
            "Output Directory": Path("output"),
            "Models Directory": Path("models"),
            "Processed Data": Path("data/processed"),
        }

    # Specific subdirectories and files to clean
    clean_targets = {
        "📊 All Output Results": base_paths["Output Directory"],
        "🤖 All Trained Models": base_paths["Models Directory"],
        "📈 Figures and Plots": base_paths["Output Directory"] / "figures",
        "🔍 SHAP Analysis Results": base_paths["Output Directory"] / "shap_analysis",
        "📋 Model Evaluation Results": base_paths["Output Directory"]
        / "comprehensive_model_evaluation",
        "🎯 Feature Selection Results": base_paths["Output Directory"]
        / "feature_selection_validation",
        "📊 Stratified Sampling Results": base_paths["Output Directory"]
        / "stratified_sampling_validation",
        "🔄 Box-Cox Pipeline Results": base_paths["Output Directory"]
        / "box_cox_dual_pipeline_results.json",
        "📝 Processed Features": base_paths["Processed Data"] / "coffee_features.csv",
        "🎯 Selected Features": base_paths["Processed Data"]
        / "coffee_features_selected.csv",
        "📊 Evaluation Summary": base_paths["Output Directory"]
        / "comprehensive_model_evaluation.txt",
        "🔧 Feature Extractors": base_paths["Models Directory"]
        / "tfidf_vectorizer.pkl",
        "🧠 BERT Models Cache": base_paths["Models Directory"] / "bert_model",
        "📈 Saved Model Files": [
            base_paths["Models Directory"] / "linear_model.pkl",
            base_paths["Models Directory"] / "ridge_model.pkl",
            base_paths["Models Directory"] / "lasso_model.pkl",
            base_paths["Models Directory"] / "random_forest_model.pkl",
            base_paths["Models Directory"] / "xgboost_model.pkl",
            base_paths["Models Directory"] / "svr_model.pkl",
            base_paths["Models Directory"] / "mnir_model.pkl",
        ],
    }

    return clean_targets


def get_directory_size(path):
    """
    Calculate the total size of a directory or file.

    Args:
        path: Path to directory or file

    Returns:
        int: Size in bytes
    """
    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    total_size = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
    except (PermissionError, OSError):
        pass

    return total_size


def format_size(size_bytes):
    """
    Format size in bytes to human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def clean_path(path, dry_run=False):
    """
    Clean a single path (file or directory).

    Args:
        path: Path to clean
        dry_run: If True, only show what would be deleted

    Returns:
        bool: True if successful (or would be successful in dry-run)
    """
    if isinstance(path, list):
        # Handle list of paths
        success = True
        for p in path:
            success &= clean_path(p, dry_run)
        return success

    path = Path(path)

    if not path.exists():
        logger.info(f"  ⏭️  {path} (doesn't exist)")
        return True

    size = get_directory_size(path)
    size_str = format_size(size)

    if dry_run:
        if path.is_dir():
            logger.info(f"  🗂️  Would delete directory: {path} ({size_str})")
        else:
            logger.info(f"  📄 Would delete file: {path} ({size_str})")
        return True

    try:
        if path.is_dir():
            shutil.rmtree(path)
            logger.info(f"  ✅ Deleted directory: {path} ({size_str})")
        else:
            path.unlink()
            logger.info(f"  ✅ Deleted file: {path} ({size_str})")
        return True
    except Exception as e:
        logger.error(f"  ❌ Failed to delete {path}: {e}")
        return False


def interactive_selection(clean_targets):
    """
    Allow user to interactively select which directories to clean.

    Args:
        clean_targets: Dictionary of clean targets

    Returns:
        dict: Selected targets to clean
    """
    print("\n🎯 Select directories to clean:")
    print("=" * 50)

    # Show options with sizes
    options = {}
    for i, (desc, path) in enumerate(clean_targets.items(), 1):
        if isinstance(path, list):
            total_size = sum(get_directory_size(Path(p)) for p in path)
        else:
            total_size = get_directory_size(Path(path))

        size_str = format_size(total_size)
        exists = "✅" if total_size > 0 else "⚪"

        print(f"{i:2d}. {exists} {desc} ({size_str})")
        options[str(i)] = (desc, path)

    print(f"{len(options) + 1:2d}. 🚀 Clean ALL directories")
    print(f"{len(options) + 2:2d}. ❌ Cancel")

    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(options) + 2}): ").strip()

            if choice == str(len(options) + 2):  # Cancel
                return {}
            elif choice == str(len(options) + 1):  # All
                return clean_targets
            elif choice in options:
                desc, path = options[choice]
                return {desc: path}
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(options) + 2}")
        except KeyboardInterrupt:
            print("\n\n❌ Cancelled by user")
            return {}


def main():
    """Main function to handle directory cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean output directories for fresh pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clean_outputs.py                    # Interactive mode with confirmation
    python clean_outputs.py --confirm          # Clean all without confirmation
    python clean_outputs.py --dry-run          # Show what would be deleted
    python clean_outputs.py --selective        # Choose specific directories
    python clean_outputs.py --selective --dry-run  # Preview selective cleaning
        """,
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt and proceed with cleaning",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    parser.add_argument(
        "--selective",
        action="store_true",
        help="Choose specific directories to clean interactively",
    )

    args = parser.parse_args()

    print("🧹 Coffee Text Analytics - Output Directory Cleaner")
    print("=" * 55)

    # Get directories to clean
    clean_targets = get_directories_to_clean()

    if args.selective:
        clean_targets = interactive_selection(clean_targets)
        if not clean_targets:
            print("❌ No directories selected. Exiting.")
            return

    # Calculate total size
    total_size = 0
    existing_targets = {}

    for desc, path in clean_targets.items():
        if isinstance(path, list):
            size = sum(get_directory_size(Path(p)) for p in path)
        else:
            size = get_directory_size(Path(path))

        if size > 0:
            existing_targets[desc] = path
            total_size += size

    if not existing_targets:
        print("✅ No files or directories to clean. Everything is already clean!")
        return

    # Show what will be cleaned
    print(f"\n📋 The following will be {'cleaned' if not args.dry_run else 'shown'}:")
    print("-" * 50)

    for desc, path in existing_targets.items():
        if isinstance(path, list):
            size = sum(get_directory_size(Path(p)) for p in path)
            print(f"📁 {desc}: {len(path)} files ({format_size(size)})")
        else:
            size = get_directory_size(Path(path))
            print(f"📁 {desc}: {format_size(size)}")

    print(
        f"\n💾 Total size to {'clean' if not args.dry_run else 'show'}: {format_size(total_size)}"
    )

    # Confirmation
    if not args.confirm and not args.dry_run:
        print(
            f"\n⚠️  This will permanently delete {len(existing_targets)} directories/files!"
        )
        response = (
            input("Are you sure you want to continue? (yes/no): ").strip().lower()
        )

        if response not in ["yes", "y"]:
            print("❌ Cleaning cancelled.")
            return

    # Perform cleaning
    print(f"\n🚀 {'Simulating' if args.dry_run else 'Starting'} cleanup...")
    print("-" * 50)

    success_count = 0
    total_count = len(existing_targets)

    for desc, path in existing_targets.items():
        print(f"\n🧹 {desc}:")
        if clean_path(path, args.dry_run):
            success_count += 1

    # Summary
    print("\n" + "=" * 50)
    if args.dry_run:
        print(
            f"📋 Dry run completed: {success_count}/{total_count} items would be cleaned"
        )
        print("💡 Run without --dry-run to actually perform the cleanup")
    else:
        if success_count == total_count:
            print(
                f"✅ Cleanup completed successfully: {success_count}/{total_count} items cleaned"
            )
            print(f"💾 Freed up {format_size(total_size)} of disk space")
        else:
            print(
                f"⚠️  Cleanup partially completed: {success_count}/{total_count} items cleaned"
            )
            print("❌ Some items could not be deleted (check permissions)")

    print("\n🚀 Ready for fresh pipeline run!")


if __name__ == "__main__":
    main()
