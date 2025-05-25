"""
Configuration Management CLI

This module provides a command-line interface for managing configuration settings,
validating configurations, and switching between environments.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from .settings import Config
from .validation import validate_config, print_config_summary, check_dependencies
from .environments import (
    list_available_environments,
    get_environment_config,
    apply_environment_config,
)


def create_config_parser():
    """Create argument parser for configuration CLI."""
    parser = argparse.ArgumentParser(
        description="Coffee Text Analytics Configuration Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m config.cli --validate                    # Validate current configuration
  python -m config.cli --summary                     # Show configuration summary
  python -m config.cli --environment production      # Show production config
  python -m config.cli --check-deps                  # Check dependencies
  python -m config.cli --export config.json          # Export configuration
        """,
    )

    parser.add_argument(
        "--validate", action="store_true", help="Validate current configuration"
    )

    parser.add_argument(
        "--summary", action="store_true", help="Show configuration summary"
    )

    parser.add_argument(
        "--environment",
        type=str,
        choices=list_available_environments(),
        help="Show configuration for specific environment",
    )

    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are available",
    )

    parser.add_argument(
        "--export", type=str, metavar="FILE", help="Export configuration to JSON file"
    )

    parser.add_argument(
        "--list-environments",
        action="store_true",
        help="List available environment configurations",
    )

    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("ENV1", "ENV2"),
        help="Compare two environment configurations",
    )

    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="Set configuration value (can be used multiple times)",
    )

    parser.add_argument(
        "--get", type=str, metavar="KEY", help="Get configuration value"
    )

    return parser


def validate_configuration():
    """Validate current configuration and print results."""
    config = Config()

    print("Validating configuration...")
    is_valid = validate_config(config, raise_on_error=False)

    if is_valid:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration validation failed!")
        return False

    return True


def show_configuration_summary():
    """Show detailed configuration summary."""
    config = Config()
    print_config_summary(config)


def show_environment_config(environment: str):
    """Show configuration for specific environment."""
    print(f"\n{environment.upper()} Environment Configuration:")
    print("=" * 60)

    env_config = get_environment_config(environment)

    if not env_config:
        print(f"No specific configuration found for environment: {environment}")
        return

    def print_nested_dict(d: Dict[str, Any], indent: int = 0):
        """Print nested dictionary with proper indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_nested_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    print_nested_dict(env_config)


def check_dependencies_cli():
    """Check and report on dependencies."""
    print("Checking dependencies...")

    deps_available, missing_deps = check_dependencies()

    if deps_available:
        print("✅ All required dependencies are available!")
    else:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall missing dependencies with:")
        print(f"pip install {' '.join(missing_deps)}")
        return False

    return True


def export_configuration(filename: str):
    """Export configuration to JSON file."""
    config = Config()

    try:
        config_dict = config.to_dict()

        with open(filename, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        print(f"✅ Configuration exported to {filename}")

    except Exception as e:
        print(f"❌ Failed to export configuration: {e}")
        return False

    return True


def list_environments():
    """List all available environment configurations."""
    environments = list_available_environments()

    print("Available Environment Configurations:")
    print("=" * 40)

    for env in environments:
        print(f"  - {env}")

    print(f"\nTotal: {len(environments)} environments")


def compare_environments(env1: str, env2: str):
    """Compare two environment configurations."""
    print(f"\nComparing {env1.upper()} vs {env2.upper()}:")
    print("=" * 60)

    config1 = get_environment_config(env1)
    config2 = get_environment_config(env2)

    if not config1:
        print(f"❌ Environment '{env1}' not found")
        return False

    if not config2:
        print(f"❌ Environment '{env2}' not found")
        return False

    def compare_dicts(d1: Dict[str, Any], d2: Dict[str, Any], path: str = ""):
        """Compare two dictionaries and show differences."""
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in sorted(all_keys):
            current_path = f"{path}.{key}" if path else key

            if key not in d1:
                print(f"  + {current_path}: {d2[key]} (only in {env2})")
            elif key not in d2:
                print(f"  - {current_path}: {d1[key]} (only in {env1})")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                compare_dicts(d1[key], d2[key], current_path)
            elif d1[key] != d2[key]:
                print(f"  ~ {current_path}: {d1[key]} → {d2[key]}")

    compare_dicts(config1, config2)


def get_config_value(key: str):
    """Get a specific configuration value."""
    config = Config()

    try:
        # Navigate through nested attributes
        parts = key.split(".")
        value = config

        for part in parts:
            value = getattr(value, part)

        print(f"{key}: {value}")

    except AttributeError:
        print(f"❌ Configuration key not found: {key}")
        return False

    return True


def set_config_value(key: str, value: str):
    """Set a specific configuration value."""
    print(f"Setting {key} = {value}")
    print("Note: This only affects the current session.")
    print("To make permanent changes, modify the configuration files.")

    # This is a placeholder - in practice, you might want to implement
    # temporary configuration overrides or generate configuration files
    return True


def main():
    """Main CLI function."""
    parser = create_config_parser()
    args = parser.parse_args()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    success = True

    # Handle different commands
    if args.validate:
        success = validate_configuration()

    if args.summary:
        show_configuration_summary()

    if args.environment:
        show_environment_config(args.environment)

    if args.check_deps:
        success = check_dependencies_cli() and success

    if args.export:
        success = export_configuration(args.export) and success

    if args.list_environments:
        list_environments()

    if args.compare:
        success = compare_environments(args.compare[0], args.compare[1]) and success

    if args.get:
        success = get_config_value(args.get) and success

    if args.set:
        for key, value in args.set:
            success = set_config_value(key, value) and success

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
