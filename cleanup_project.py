#!/usr/bin/env python3
"""
Project Cleanup Utility

This script cleans up the major bloat sources in the coffee text analytics project:
- Large log files (coffee_analytics.log: 5.1MB)
- Test coverage reports (htmlcov/: 4.4MB)
- Large processed data files (data/processed/coffee_features.csv: 30MB)
- Cache files and temporary data

Usage:
    python cleanup_project.py [--dry-run] [--confirm] [--aggressive]

Options:
    --dry-run     Show what would be cleaned without actually deleting
    --confirm     Skip confirmation and proceed with cleanup
    --aggressive  Also clean large processed data files (regeneratable)
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def format_size(size_bytes):
    """Format size in bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def get_path_size(path):
    """Get total size of a file or directory."""
    path = Path(path)
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


def clean_item(path, description, dry_run=False):
    """Clean a single file or directory safely."""
    path = Path(path)

    if not path.exists():
        logger.info(f"  ⏭️  {description}: {path} (doesn't exist)")
        return 0

    size = get_path_size(path)
    size_str = format_size(size)

    if dry_run:
        logger.info(f"  🗂️  Would clean {description}: {path} ({size_str})")
        return size

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

        logger.info(f"  ✅ Cleaned {description}: {path} ({size_str})")
        return size

    except Exception as e:
        logger.error(f"  ❌ Failed to clean {description} at {path}: {e}")
        return 0


def backup_log_file(log_path, max_lines=100):
    """Backup the last N lines of a log file before cleaning."""
    log_path = Path(log_path)

    if not log_path.exists():
        return

    backup_path = log_path.with_suffix(".log.backup")

    try:
        # Read last N lines
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Keep only last max_lines
        if len(lines) > max_lines:
            lines = lines[-max_lines:]

        # Write backup
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(f"# Backup of last {len(lines)} lines from {log_path.name}\n")
            f.write(f"# Backup created: {datetime.now().isoformat()}\n")
            f.write("# " + "=" * 50 + "\n\n")
            f.writelines(lines)

        logger.info(f"  💾 Created backup: {backup_path} ({len(lines)} lines)")

    except Exception as e:
        logger.warning(f"  ⚠️  Could not create log backup: {e}")


def get_cleanup_targets(aggressive=False):
    """Get list of items to clean based on mode."""

    # Safe cleanup targets (always safe to remove)
    safe_targets = [
        {
            "path": "coffee_analytics.log",
            "description": "Main log file",
            "backup": True,
            "category": "logs",
        },
        {
            "path": "htmlcov/",
            "description": "Test coverage reports",
            "backup": False,
            "category": "coverage",
        },
        {
            "path": "cache/",
            "description": "Cache directory",
            "backup": False,
            "category": "cache",
        },
        {
            "path": ".pytest_cache/",
            "description": "Pytest cache",
            "backup": False,
            "category": "cache",
        },
        {
            "path": "src/__pycache__/",
            "description": "Python cache (src)",
            "backup": False,
            "category": "cache",
        },
        {
            "path": "tests/__pycache__/",
            "description": "Python cache (tests)",
            "backup": False,
            "category": "cache",
        },
    ]

    # Aggressive cleanup targets (regeneratable but might require processing time)
    aggressive_targets = [
        {
            "path": "data/processed/coffee_features.csv",
            "description": "Large feature matrix (30MB)",
            "backup": False,
            "category": "processed_data",
        },
        {
            "path": "data/processed/coffee_features_selected.csv",
            "description": "Selected features",
            "backup": False,
            "category": "processed_data",
        },
        {
            "path": "models/",
            "description": "Trained models directory",
            "backup": False,
            "category": "models",
        },
    ]

    targets = safe_targets
    if aggressive:
        targets.extend(aggressive_targets)

    return targets


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean up coffee text analytics project bloat"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be cleaned"
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--aggressive", action="store_true", help="Also clean processed data files"
    )

    args = parser.parse_args()

    print("☕ Coffee Text Analytics - Project Cleanup")
    print("🧹 Targeting major bloat sources")
    print("=" * 60)

    # Get cleanup targets
    targets = get_cleanup_targets(args.aggressive)

    # Calculate current sizes and show preview
    total_size = 0
    existing_targets = []

    print("\n📊 Current Space Usage Analysis:")
    for target in targets:
        path = Path(target["path"])
        size = get_path_size(path)

        if size > 0:
            existing_targets.append(target)
            total_size += size
            size_str = format_size(size)
            print(f"  📁 {target['description']}: {size_str}")

    if total_size == 0:
        print("✨ Project is already clean! No cleanup needed.")
        return

    print(f"\n🎯 Total potential cleanup: {format_size(total_size)}")

    if args.aggressive:
        print("⚠️  AGGRESSIVE mode: Will also clean processed data files")
        print("   (These will need to be regenerated by running the pipeline)")

    # Confirmation prompt
    if not args.confirm and not args.dry_run:
        print("\n" + "=" * 60)
        response = input("Proceed with cleanup? [y/N]: ").lower().strip()
        if response not in ["y", "yes"]:
            print("❌ Cleanup cancelled.")
            return

    # Perform cleanup
    print(f"\n🚀 {'DRY RUN - ' if args.dry_run else ''}Starting cleanup...")

    cleaned_size = 0
    cleanup_summary = {
        "logs": 0,
        "coverage": 0,
        "cache": 0,
        "processed_data": 0,
        "models": 0,
    }

    for target in existing_targets:
        # Backup logs if needed
        if target.get("backup") and not args.dry_run:
            backup_log_file(target["path"])

        # Clean the item
        size_cleaned = clean_item(target["path"], target["description"], args.dry_run)
        cleaned_size += size_cleaned
        cleanup_summary[target["category"]] += size_cleaned

    # Summary
    print(f"\n{'=' * 60}")
    print(f"🎉 {'DRY RUN COMPLETE' if args.dry_run else 'CLEANUP COMPLETE'}")
    print(f"{'=' * 60}")
    print(
        f"📊 {'Would clean' if args.dry_run else 'Cleaned'}: {format_size(cleaned_size)}"
    )

    if not args.dry_run:
        print(f"💾 Space freed: {format_size(cleaned_size)}")

    print("\n📋 Breakdown by category:")
    for category, size in cleanup_summary.items():
        if size > 0:
            print(f"  {category.replace('_', ' ').title()}: {format_size(size)}")

    # Next steps
    print(f"\n{'=' * 60}")
    print("📋 Next Steps:")
    if args.dry_run:
        print("1. Run without --dry-run to perform actual cleanup")
        print("2. Add --aggressive to also clean processed data files")
    else:
        print("1. ✅ Project size reduced successfully")
        print("2. 🚀 Ready for MLflow integration")

        if args.aggressive:
            print("3. ⚠️  Processed data cleaned - run pipeline to regenerate if needed")

        # Show updated project size
        try:
            new_total = get_path_size(".")
            print(f"4. 📊 Updated project size: {format_size(new_total)}")
        except:
            pass

    print("3. 📝 Consider updating .gitignore for better exclusions")


if __name__ == "__main__":
    main()
