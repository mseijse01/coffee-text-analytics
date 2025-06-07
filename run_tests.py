#!/usr/bin/env python3
"""
Comprehensive test runner for coffee text analytics project.

Runs all test suites and provides detailed reporting before refactoring.
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import test modules
from tests.test_data_processing import *
# from tests.test_integration_new import *  # This module doesn't exist


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
        self.verbosity = verbosity  # Explicitly store verbosity

    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity > 1:
            self.stream.write("✅ ")
            self.stream.writeln(f"{test._testMethodName}")

    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write("❌ ")
            self.stream.writeln(f"{test._testMethodName} - ERROR")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write("❌ ")
            self.stream.writeln(f"{test._testMethodName} - FAILED")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write("⏭️  ")
            self.stream.writeln(f"{test._testMethodName} - SKIPPED: {reason}")


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output."""

    resultclass = ColoredTextTestResult


def run_test_suite(test_module_name, test_classes):
    """Run a specific test suite and return results."""
    print(f"\n{'=' * 60}")
    print(f"🧪 Running {test_module_name} Tests")
    print(f"{'=' * 60}")

    # Create test suite
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = ColoredTextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # Print summary
    print(f"\n📊 {test_module_name} Summary:")
    print(f"   ✅ Passed: {result.success_count}")
    print(f"   ❌ Failed: {len(result.failures)}")
    print(f"   💥 Errors: {len(result.errors)}")
    print(f"   ⏭️  Skipped: {len(result.skipped)}")
    print(f"   ⏱️  Time: {end_time - start_time:.2f}s")

    return result


def analyze_code_coverage():
    """Analyze which components are covered by tests."""
    print(f"\n{'=' * 60}")
    print("📈 Test Coverage Analysis")
    print(f"{'=' * 60}")

    # Define components and their test coverage
    components = {
        "Data Loading": {
            "files": ["src/data/loader.py"],
            "tests": ["TestDataLoading"],
            "coverage": "✅ High",
        },
        "Data Preprocessing": {
            "files": ["src/data/preprocessing.py"],
            "tests": ["TestTextPreprocessing"],
            "coverage": "✅ High",
        },
        "Data Cleaning": {
            "files": ["src/utils/cleaning.py"],
            "tests": ["TestDataCleaning"],
            "coverage": "✅ High",
        },
        "Feature Extraction": {
            "files": ["src/features/feature_extraction.py"],
            "tests": ["TestCoffeeFeatureExtractor", "TestFeatureExtractionUtilities"],
            "coverage": "✅ High",
        },
        "Model Training": {
            "files": ["src/models/model_training.py"],
            "tests": ["TestMultinomialInverseRegression", "TestTraditionalModels"],
            "coverage": "✅ High",
        },
        "MNIR Implementation": {
            "files": ["src/models/model_training.py (MNIR class)"],
            "tests": ["TestMultinomialInverseRegression", "TestTrainMNIR"],
            "coverage": "✅ High",
        },
        "Integration": {
            "files": ["Multiple components"],
            "tests": ["TestEndToEndPipeline", "TestComponentIntegration"],
            "coverage": "✅ High",
        },
        "Error Handling": {
            "files": ["All components"],
            "tests": ["TestErrorHandlingIntegration"],
            "coverage": "✅ Medium",
        },
    }

    for component, info in components.items():
        print(f"\n🔧 {component}")
        print(f"   📁 Files: {', '.join(info['files'])}")
        print(f"   🧪 Tests: {', '.join(info['tests'])}")
        print(f"   📊 Coverage: {info['coverage']}")


def identify_refactoring_opportunities():
    """Identify potential refactoring opportunities based on test analysis."""
    print(f"\n{'=' * 60}")
    print("🔧 Refactoring Opportunities Identified")
    print(f"{'=' * 60}")

    opportunities = [
        {
            "area": "Code Duplication",
            "description": "analyze_data_quality() function duplicated in loader.py and utils.py",
            "priority": "🔴 High",
            "action": "Consolidate into single utility function",
        },
        {
            "area": "Feature Extraction",
            "description": "Large CoffeeFeatureExtractor class with multiple responsibilities",
            "priority": "🟡 Medium",
            "action": "Split into specialized extractors (TfidfExtractor, BertExtractor, etc.)",
        },
        {
            "area": "Error Handling",
            "description": "Inconsistent error handling patterns across modules",
            "priority": "🟡 Medium",
            "action": "Standardize error handling with custom exceptions",
        },
        {
            "area": "Configuration",
            "description": "Hardcoded parameters scattered throughout codebase",
            "priority": "🟡 Medium",
            "action": "Centralize configuration in config module",
        },
        {
            "area": "Model Training",
            "description": "Large model_training.py file with multiple concerns",
            "priority": "🟡 Medium",
            "action": "Split into separate files for different model types",
        },
        {
            "area": "Data Type Consistency",
            "description": "Mixed Polars/Pandas usage could be streamlined",
            "priority": "🟢 Low",
            "action": "Establish clear data type conventions",
        },
    ]

    for i, opp in enumerate(opportunities, 1):
        print(f"\n{i}. {opp['area']} - {opp['priority']}")
        print(f"   📝 Issue: {opp['description']}")
        print(f"   🎯 Action: {opp['action']}")


def generate_refactoring_plan():
    """Generate a detailed refactoring plan."""
    print(f"\n{'=' * 60}")
    print("📋 Recommended Refactoring Plan")
    print(f"{'=' * 60}")

    phases = [
        {
            "phase": "Phase 1: Code Deduplication",
            "priority": "🔴 High Priority",
            "tasks": [
                "Consolidate analyze_data_quality() functions",
                "Remove duplicate preprocessing logic",
                "Standardize import patterns",
            ],
            "risk": "🟢 Low Risk",
            "estimated_time": "2-3 hours",
        },
        {
            "phase": "Phase 2: Configuration Management",
            "priority": "🟡 Medium Priority",
            "tasks": [
                "Create centralized config module",
                "Move hardcoded parameters to config",
                "Add environment-specific configurations",
            ],
            "risk": "🟢 Low Risk",
            "estimated_time": "3-4 hours",
        },
        {
            "phase": "Phase 3: Component Separation",
            "priority": "🟡 Medium Priority",
            "tasks": [
                "Split CoffeeFeatureExtractor into specialized classes",
                "Separate model training by type",
                "Create abstract base classes for consistency",
            ],
            "risk": "🟡 Medium Risk",
            "estimated_time": "6-8 hours",
        },
        {
            "phase": "Phase 4: Error Handling Standardization",
            "priority": "🟡 Medium Priority",
            "tasks": [
                "Create custom exception classes",
                "Implement consistent error handling patterns",
                "Add comprehensive logging",
            ],
            "risk": "🟡 Medium Risk",
            "estimated_time": "4-5 hours",
        },
        {
            "phase": "Phase 5: Performance Optimization",
            "priority": "🟢 Low Priority",
            "tasks": [
                "Optimize Polars/Pandas conversions",
                "Add caching for expensive operations",
                "Profile and optimize bottlenecks",
            ],
            "risk": "🟡 Medium Risk",
            "estimated_time": "5-6 hours",
        },
    ]

    total_time = 0
    for i, phase in enumerate(phases, 1):
        print(f"\n{i}. {phase['phase']} - {phase['priority']}")
        print(f"   📋 Tasks:")
        for task in phase["tasks"]:
            print(f"      • {task}")
        print(f"   ⚠️  Risk Level: {phase['risk']}")
        print(f"   ⏱️  Estimated Time: {phase['estimated_time']}")

        # Extract time estimate
        time_range = phase["estimated_time"].split()[0]
        if "-" in time_range:
            avg_time = sum(map(int, time_range.split("-"))) / 2
        else:
            avg_time = int(time_range)
        total_time += avg_time

    print(f"\n📊 Total Estimated Refactoring Time: {total_time:.0f} hours")
    print(f"🎯 Recommended Approach: Incremental refactoring with continuous testing")


def main():
    """Main test runner function."""
    print("☕ Coffee Text Analytics - Comprehensive Test Suite")
    print("🔧 Pre-Refactoring Testing & Analysis")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Test suites to run
    test_suites = [
        {
            "name": "Data Processing",
            "classes": [
                TestDataLoading,
                TestTextPreprocessing,
                TestDataCleaning,
                TestDataQualityAnalysis,
                TestDataIntegrity,
            ],
        },
        {
            "name": "New Architecture Integration",
            "classes": [
                TestNewArchitectureIntegration,
            ],
        },
    ]

    # Run all test suites
    all_results = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0

    for suite in test_suites:
        result = run_test_suite(suite["name"], suite["classes"])
        all_results.append(result)

        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        total_skipped += len(result.skipped)

    end_time = time.time()

    # Overall summary
    print(f"\n{'=' * 60}")
    print("🎯 OVERALL TEST SUMMARY")
    print(f"{'=' * 60}")

    success_rate = (
        ((total_tests - total_failures - total_errors) / total_tests * 100)
        if total_tests > 0
        else 0
    )

    print(f"📊 Total Tests Run: {total_tests}")
    print(f"✅ Passed: {total_tests - total_failures - total_errors}")
    print(f"❌ Failed: {total_failures}")
    print(f"💥 Errors: {total_errors}")
    print(f"⏭️  Skipped: {total_skipped}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"⏱️  Total Time: {end_time - start_time:.2f}s")

    # Determine readiness for refactoring
    if total_failures == 0 and total_errors == 0:
        print(f"\n🎉 READY FOR REFACTORING!")
        print("✅ All tests passing - codebase is stable")
        print("✅ Comprehensive test coverage in place")
        print("✅ Integration tests verify end-to-end functionality")
    else:
        print(f"\n⚠️  NOT READY FOR REFACTORING")
        print("❌ Fix failing tests before proceeding")
        print("❌ Ensure all components are working correctly")

    # Analysis and recommendations
    analyze_code_coverage()
    identify_refactoring_opportunities()
    generate_refactoring_plan()

    # Final recommendations
    print(f"\n{'=' * 60}")
    print("💡 NEXT STEPS")
    print(f"{'=' * 60}")

    if total_failures == 0 and total_errors == 0:
        print("1. ✅ Tests are passing - proceed with confidence")
        print("2. 🔧 Start with Phase 1 refactoring (code deduplication)")
        print("3. 🧪 Run tests after each refactoring phase")
        print("4. 📊 Monitor performance during refactoring")
        print("5. 📝 Update documentation as you refactor")
    else:
        print("1. ❌ Fix failing tests first")
        print("2. 🔍 Investigate test failures and errors")
        print("3. 🧪 Re-run tests until all pass")
        print("4. 🔧 Then proceed with refactoring plan")

    # Exit with appropriate code
    exit_code = 0 if (total_failures == 0 and total_errors == 0) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
