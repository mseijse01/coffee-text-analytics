#!/usr/bin/env python3
"""
API Documentation Generator for Coffee Text Analytics

This module automatically generates comprehensive API documentation
by inspecting modules, classes, and functions to extract docstrings,
signatures, and usage examples.
"""

import ast
import inspect
import importlib
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import textwrap
from datetime import datetime


class APIDocumentationGenerator:
    """
    Generates comprehensive API documentation for Python modules.
    """

    def __init__(self, src_path: str = "src"):
        """
        Initialize the documentation generator.

        Args:
            src_path: Path to the source code directory
        """
        self.src_path = Path(src_path)
        self.modules_info = {}

    def discover_modules(self) -> List[str]:
        """
        Discover all Python modules in the source directory.

        Returns:
            List of module names
        """
        modules = []

        for py_file in self.src_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            # Convert file path to module name
            relative_path = py_file.relative_to(self.src_path)
            module_name = str(relative_path.with_suffix("")).replace(os.sep, ".")
            modules.append(module_name)

        return sorted(modules)

    def extract_module_info(self, module_name: str) -> Dict[str, Any]:
        """
        Extract comprehensive information from a module.

        Args:
            module_name: Name of the module to analyze

        Returns:
            Dictionary with module information
        """
        try:
            # Add src to path if not already there
            if str(self.src_path) not in sys.path:
                sys.path.insert(0, str(self.src_path))

            module = importlib.import_module(module_name)

            module_info = {
                "name": module_name,
                "docstring": inspect.getdoc(module)
                or "No module documentation available.",
                "file_path": inspect.getfile(module),
                "classes": [],
                "functions": [],
                "constants": [],
                "imports": [],
            }

            # Extract classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    obj.__module__ == module.__name__
                ):  # Only classes defined in this module
                    class_info = self._extract_class_info(obj)
                    module_info["classes"].append(class_info)

            # Extract functions
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if (
                    obj.__module__ == module.__name__
                ):  # Only functions defined in this module
                    func_info = self._extract_function_info(obj)
                    module_info["functions"].append(func_info)

            # Extract constants (uppercase variables)
            for name, obj in inspect.getmembers(module):
                if (
                    not name.startswith("_")
                    and name.isupper()
                    and not inspect.isclass(obj)
                    and not inspect.isfunction(obj)
                    and not inspect.ismodule(obj)
                ):
                    module_info["constants"].append(
                        {"name": name, "value": repr(obj), "type": type(obj).__name__}
                    )

            return module_info

        except Exception as e:
            return {
                "name": module_name,
                "error": f"Failed to import module: {e}",
                "docstring": "",
                "classes": [],
                "functions": [],
                "constants": [],
                "imports": [],
            }

    def _extract_class_info(self, cls) -> Dict[str, Any]:
        """Extract information from a class."""
        class_info = {
            "name": cls.__name__,
            "docstring": inspect.getdoc(cls) or "No class documentation available.",
            "methods": [],
            "properties": [],
            "inheritance": [base.__name__ for base in cls.__bases__ if base != object],
        }

        # Extract methods
        for name, method in inspect.getmembers(cls, inspect.ismethod):
            if not name.startswith("_") or name in ["__init__", "__call__"]:
                method_info = self._extract_function_info(method, is_method=True)
                class_info["methods"].append(method_info)

        # Extract regular functions that are methods
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_") or name in ["__init__", "__call__"]:
                method_info = self._extract_function_info(func, is_method=True)
                class_info["methods"].append(method_info)

        # Extract properties
        for name, prop in inspect.getmembers(cls):
            if isinstance(prop, property):
                class_info["properties"].append(
                    {
                        "name": name,
                        "docstring": inspect.getdoc(prop)
                        or "No property documentation.",
                        "getter": prop.fget is not None,
                        "setter": prop.fset is not None,
                        "deleter": prop.fdel is not None,
                    }
                )

        return class_info

    def _extract_function_info(self, func, is_method: bool = False) -> Dict[str, Any]:
        """Extract information from a function."""
        try:
            signature = inspect.signature(func)

            func_info = {
                "name": func.__name__,
                "docstring": inspect.getdoc(func)
                or "No function documentation available.",
                "signature": str(signature),
                "parameters": [],
                "returns": None,
                "raises": [],
                "examples": [],
            }

            # Extract parameter information
            for param_name, param in signature.parameters.items():
                param_info = {
                    "name": param_name,
                    "annotation": str(param.annotation)
                    if param.annotation != param.empty
                    else None,
                    "default": str(param.default)
                    if param.default != param.empty
                    else None,
                    "kind": str(param.kind),
                }
                func_info["parameters"].append(param_info)

            # Extract return annotation
            if signature.return_annotation != signature.empty:
                func_info["returns"] = str(signature.return_annotation)

            # Parse docstring for additional info
            docstring = inspect.getdoc(func)
            if docstring:
                parsed_doc = self._parse_docstring(docstring)
                func_info.update(parsed_doc)

            return func_info

        except Exception as e:
            return {
                "name": func.__name__,
                "error": f"Failed to extract function info: {e}",
                "docstring": inspect.getdoc(func) or "",
                "signature": "Unable to extract signature",
                "parameters": [],
                "returns": None,
                "raises": [],
                "examples": [],
            }

    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """
        Parse docstring to extract structured information.

        Args:
            docstring: The docstring to parse

        Returns:
            Dictionary with parsed information
        """
        lines = docstring.split("\n")
        parsed = {
            "description": "",
            "args_doc": {},
            "returns_doc": "",
            "raises_doc": [],
            "examples_doc": [],
        }

        current_section = "description"
        current_content = []

        for line in lines:
            line = line.strip()

            if line.lower().startswith("args:") or line.lower().startswith(
                "arguments:"
            ):
                if current_content:
                    parsed[current_section] = "\n".join(current_content).strip()
                current_section = "args"
                current_content = []
            elif line.lower().startswith("returns:") or line.lower().startswith(
                "return:"
            ):
                if current_content:
                    if current_section == "args":
                        # Process args content
                        self._parse_args_section(current_content, parsed)
                    else:
                        parsed[current_section] = "\n".join(current_content).strip()
                current_section = "returns"
                current_content = []
            elif line.lower().startswith("raises:") or line.lower().startswith(
                "raise:"
            ):
                if current_content:
                    if current_section == "args":
                        self._parse_args_section(current_content, parsed)
                    else:
                        parsed[current_section] = "\n".join(current_content).strip()
                current_section = "raises"
                current_content = []
            elif line.lower().startswith("example:") or line.lower().startswith(
                "examples:"
            ):
                if current_content:
                    if current_section == "args":
                        self._parse_args_section(current_content, parsed)
                    else:
                        parsed[current_section] = "\n".join(current_content).strip()
                current_section = "examples"
                current_content = []
            else:
                current_content.append(line)

        # Process final section
        if current_content:
            if current_section == "args":
                self._parse_args_section(current_content, parsed)
            elif current_section == "returns":
                parsed["returns_doc"] = "\n".join(current_content).strip()
            elif current_section == "examples":
                parsed["examples_doc"] = current_content
            else:
                parsed[current_section] = "\n".join(current_content).strip()

        return parsed

    def _parse_args_section(self, content: List[str], parsed: Dict[str, Any]):
        """Parse the arguments section of a docstring."""
        current_arg = None
        current_desc = []

        for line in content:
            if ":" in line and not line.startswith(" "):
                # Save previous arg
                if current_arg:
                    parsed["args_doc"][current_arg] = " ".join(current_desc).strip()

                # Start new arg
                parts = line.split(":", 1)
                current_arg = parts[0].strip()
                current_desc = [parts[1].strip()] if len(parts) > 1 else []
            else:
                current_desc.append(line.strip())

        # Save final arg
        if current_arg:
            parsed["args_doc"][current_arg] = " ".join(current_desc).strip()

    def generate_markdown_docs(self, output_file: str = "API_DOCUMENTATION.md") -> str:
        """
        Generate comprehensive markdown documentation.

        Args:
            output_file: Output file name

        Returns:
            Generated markdown content
        """
        modules = self.discover_modules()

        # Generate documentation for each module
        for module_name in modules:
            self.modules_info[module_name] = self.extract_module_info(module_name)

        # Generate markdown
        markdown_content = self._generate_markdown_content()

        # Write to file
        output_path = Path(output_file)
        output_path.write_text(markdown_content, encoding="utf-8")

        print(f"📚 API documentation generated: {output_path.absolute()}")
        return markdown_content

    def _generate_markdown_content(self) -> str:
        """Generate the complete markdown documentation."""
        lines = [
            "# 📚 Coffee Text Analytics - API Documentation",
            "",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## 📋 Table of Contents",
            "",
        ]

        # Generate table of contents
        for module_name in sorted(self.modules_info.keys()):
            module_info = self.modules_info[module_name]
            if "error" not in module_info:
                lines.append(f"- [{module_name}](#{module_name.replace('.', '')})")

        lines.extend(["", "---", ""])

        # Generate documentation for each module
        for module_name in sorted(self.modules_info.keys()):
            module_info = self.modules_info[module_name]
            lines.extend(self._generate_module_markdown(module_info))
            lines.append("")

        return "\n".join(lines)

    def _generate_module_markdown(self, module_info: Dict[str, Any]) -> List[str]:
        """Generate markdown for a single module."""
        lines = []
        module_name = module_info["name"]

        # Module header
        lines.extend(
            [
                f"## 📦 {module_name}",
                "",
                f"**File:** `{module_info.get('file_path', 'Unknown')}`",
                "",
                module_info["docstring"],
                "",
            ]
        )

        # Handle import errors
        if "error" in module_info:
            lines.extend(
                ["⚠️ **Import Error:**", f"```", module_info["error"], "```", ""]
            )
            return lines

        # Constants
        if module_info["constants"]:
            lines.extend(["### 🔢 Constants", ""])
            for const in module_info["constants"]:
                lines.extend(
                    [
                        f"#### `{const['name']}`",
                        f"- **Type:** `{const['type']}`",
                        f"- **Value:** `{const['value']}`",
                        "",
                    ]
                )

        # Functions
        if module_info["functions"]:
            lines.extend(["### 🔧 Functions", ""])
            for func in module_info["functions"]:
                lines.extend(self._generate_function_markdown(func))

        # Classes
        if module_info["classes"]:
            lines.extend(["### 🏗️ Classes", ""])
            for cls in module_info["classes"]:
                lines.extend(self._generate_class_markdown(cls))

        return lines

    def _generate_function_markdown(self, func_info: Dict[str, Any]) -> List[str]:
        """Generate markdown for a function."""
        lines = []

        lines.extend([f"#### `{func_info['name']}{func_info['signature']}`", ""])

        if "error" in func_info:
            lines.extend(
                [
                    "⚠️ **Error extracting function information:**",
                    f"```",
                    func_info["error"],
                    "```",
                    "",
                ]
            )
            return lines

        # Description
        if func_info.get("description"):
            lines.extend([func_info["description"], ""])
        elif func_info["docstring"]:
            lines.extend([func_info["docstring"], ""])

        # Parameters
        if func_info["parameters"]:
            lines.extend(["**Parameters:**", ""])
            for param in func_info["parameters"]:
                param_line = f"- `{param['name']}`"
                if param["annotation"]:
                    param_line += f" ({param['annotation']})"
                if param["default"]:
                    param_line += f" = {param['default']}"
                lines.append(param_line)

                # Add parameter description if available
                if func_info.get("args_doc", {}).get(param["name"]):
                    lines.append(f"  - {func_info['args_doc'][param['name']]}")
            lines.append("")

        # Returns
        if func_info["returns"] or func_info.get("returns_doc"):
            lines.extend(["**Returns:**", ""])
            if func_info["returns"]:
                lines.append(f"- Type: `{func_info['returns']}`")
            if func_info.get("returns_doc"):
                lines.append(f"- {func_info['returns_doc']}")
            lines.append("")

        # Examples
        if func_info.get("examples_doc"):
            lines.extend(["**Examples:**", "", "```python"])
            lines.extend(func_info["examples_doc"])
            lines.extend(["```", ""])

        lines.append("---")
        lines.append("")

        return lines

    def _generate_class_markdown(self, class_info: Dict[str, Any]) -> List[str]:
        """Generate markdown for a class."""
        lines = []

        lines.extend([f"#### 🏗️ `{class_info['name']}`", ""])

        # Inheritance
        if class_info["inheritance"]:
            lines.extend(
                [
                    f"**Inherits from:** {', '.join(f'`{base}`' for base in class_info['inheritance'])}",
                    "",
                ]
            )

        # Description
        lines.extend([class_info["docstring"], ""])

        # Properties
        if class_info["properties"]:
            lines.extend(["**Properties:**", ""])
            for prop in class_info["properties"]:
                prop_line = f"- `{prop['name']}`"
                if prop["getter"]:
                    prop_line += " (readable)"
                if prop["setter"]:
                    prop_line += " (writable)"
                lines.append(prop_line)
                if prop["docstring"] != "No property documentation.":
                    lines.append(f"  - {prop['docstring']}")
            lines.append("")

        # Methods
        if class_info["methods"]:
            lines.extend(["**Methods:**", ""])
            for method in class_info["methods"]:
                lines.extend(self._generate_function_markdown(method))

        return lines


def generate_api_docs(src_path: str = "src", output_file: str = "API_DOCUMENTATION.md"):
    """
    Generate API documentation for the entire project.

    Args:
        src_path: Path to source code directory
        output_file: Output documentation file
    """
    generator = APIDocumentationGenerator(src_path)
    content = generator.generate_markdown_docs(output_file)

    # Generate summary
    total_modules = len(generator.modules_info)
    total_classes = sum(
        len(info.get("classes", [])) for info in generator.modules_info.values()
    )
    total_functions = sum(
        len(info.get("functions", [])) for info in generator.modules_info.values()
    )

    print(f"\n📊 Documentation Summary:")
    print(f"   📦 Modules documented: {total_modules}")
    print(f"   🏗️ Classes documented: {total_classes}")
    print(f"   🔧 Functions documented: {total_functions}")
    print(f"   📄 Output file: {output_file}")


if __name__ == "__main__":
    generate_api_docs()
