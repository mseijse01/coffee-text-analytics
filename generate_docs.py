#!/usr/bin/env python3
"""
Documentation Generation Script

Simple script to regenerate API documentation for the Coffee Text Analytics project.
Run this whenever you make changes to the codebase to keep documentation up-to-date.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")


def main():
    """Generate comprehensive API documentation."""
    print("🔄 Generating Coffee Text Analytics API Documentation...")
    print("=" * 60)

    try:
        from src.utils.doc_generator import generate_api_docs

        # Generate main API documentation
        generate_api_docs(src_path="src", output_file="API_DOCUMENTATION.md")

        print("\n✅ Documentation generation completed successfully!")
        print("\n📚 Generated files:")
        print("   📄 API_DOCUMENTATION.md - Complete API reference")
        print("   📄 API_QUICK_REFERENCE.md - Quick start guide")

        print("\n🎯 Usage:")
        print("   • Open API_QUICK_REFERENCE.md for a quick overview")
        print("   • Open API_DOCUMENTATION.md for detailed documentation")
        print("   • Both files are in Markdown format for easy viewing")

    except ImportError as e:
        print(f"❌ Error: Could not import documentation generator: {e}")
        print("   Make sure you're running this from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
