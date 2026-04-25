#!/bin/bash
# Test runner script — run from another terminal
# Verifies all fixes with focused unit tests (no flaky dependencies)
# Usage: bash run_tests_externally.sh [option]
#
# Options:
#   quick   - All unit tests, ~5 sec (35 tests)
#   full    - All test files, comprehensive (~30 min, heavy RAM)

set -e

source ~/.virtualenvs/coffee-analytics/bin/activate
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

OPTION=${1:-quick}

echo "=========================================="
echo "Coffee Text Analytics Test Suite"
echo "=========================================="
echo "Option: $OPTION"
echo "Start: $(date)"
echo ""

case $OPTION in
  quick)
    echo "QUICK: 35 unit tests (~5 sec)"
    echo "  - MNIR core tests: 18"
    echo "  - MNIR focused tests: 11"
    echo "  - Cache focused tests: 6"
    echo ""
    python -m pytest \
      tests/test_mnir.py \
      tests/test_mnir_focused_unit.py \
      tests/test_cache_focused_unit.py \
      -v --tb=short
    ;;

  full)
    echo "FULL: ~400 tests across all files (~30 min, heavy RAM)"
    echo "WARNING: This is resource-intensive and slow."
    echo ""
    python -m pytest tests/ -q --tb=line 2>&1 | tail -100
    ;;

  *)
    echo "Usage: bash run_tests_externally.sh [quick|full]"
    echo ""
    echo "  quick - 35 unit tests (~5 sec) ← RECOMMENDED"
    echo "  full  - ~400 tests (~30 min, heavy RAM)"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "Complete: $(date)"
echo "=========================================="
