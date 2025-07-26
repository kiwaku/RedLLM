#!/usr/bin/env python3
"""
Comprehensive test runner for RedLLM hybrid output implementation.
Runs all test suites and provides detailed reporting.
"""

import unittest
import sys
import os
from pathlib import Path
from io import StringIO

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_test_suite():
    """Run all test suites and return results."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create a test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        descriptions=True,
        failfast=False
    )
    
    print("🧪 Running RedLLM Hybrid Output Test Suite")
    print("=" * 50)
    
    # Run the tests
    result = runner.run(suite)
    
    # Print results
    output = stream.getvalue()
    print(output)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0] if 'AssertionError: ' in traceback else 'See details above'}")
    
    if result.errors:
        print("\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception: ')[-1].split('\\n')[0] if 'Exception: ' in traceback else 'See details above'}")
    
    # Overall result
    if result.wasSuccessful():
        print("\n✅ All tests passed! Hybrid output implementation is ready.")
        return True
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return False

def run_specific_test_module(module_name):
    """Run tests from a specific module."""
    print(f"🧪 Running tests for {module_name}")
    print("=" * 50)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def validate_test_environment():
    """Validate that the test environment is properly set up."""
    print("🔍 Validating test environment...")
    
    # Check that source modules can be imported
    try:
        from redllm.utils.run_tracking import generate_run_id
        print("   ✅ run_tracking module imports successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import run_tracking: {e}")
        return False
    
    try:
        from redllm.utils.reporter import generate_hybrid_report
        print("   ✅ reporter module imports successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import reporter: {e}")
        return False
    
    try:
        from redllm.core.attack_engine import MultiVariationAttack
        print("   ✅ attack_engine module imports successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import attack_engine: {e}")
        return False
    
    try:
        from redllm.main import create_parser
        print("   ✅ main module imports successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import main: {e}")
        return False
    
    # Test that run_tracking functions work
    try:
        from redllm.utils.run_tracking import generate_run_id, SUMMARY_HEADERS
        run_id = generate_run_id()
        assert len(run_id.split('_')) == 3
        assert len(SUMMARY_HEADERS) == 10
        print("   ✅ run_tracking functions work correctly")
    except Exception as e:
        print(f"   ❌ run_tracking functions failed: {e}")
        return False
    
    print("   ✅ Test environment validation successful!")
    return True

def main():
    """Main test runner function."""
    print("🚀 RedLLM Hybrid Output Test Suite")
    print("Testing implementation of dual-output CSV+JSON system")
    print()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--validate':
            success = validate_test_environment()
            sys.exit(0 if success else 1)
        
        elif sys.argv[1] == '--module':
            if len(sys.argv) < 3:
                print("Usage: python test_runner.py --module <module_name>")
                sys.exit(1)
            success = run_specific_test_module(sys.argv[2])
            sys.exit(0 if success else 1)
        
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python test_runner.py                 # Run all tests")
            print("  python test_runner.py --validate      # Validate test environment")
            print("  python test_runner.py --module <name> # Run specific test module")
            print("  python test_runner.py --help          # Show this help")
            print()
            print("Available test modules:")
            print("  test_hybrid_output")
            print("  test_attack_engine_hybrid")
            print("  test_cli_hybrid")
            sys.exit(0)
    
    # Validate environment first
    if not validate_test_environment():
        print("\n❌ Environment validation failed. Please fix issues before running tests.")
        sys.exit(1)
    
    print()
    
    # Run all tests
    success = run_test_suite()
    
    if success:
        print("\n🎉 Hybrid output implementation is ready for production!")
        print("\nNext steps:")
        print("1. Test with real LLM integration")
        print("2. Validate CSV and JSON output formats")
        print("3. Test with large datasets")
        print("4. Performance benchmarking")
    else:
        print("\n🔧 Please fix failing tests before proceeding.")
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
