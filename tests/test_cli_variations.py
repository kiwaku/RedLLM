#!/usr/bin/env python3
"""Simple test to validate CLI argument processing for dynamic variations."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path so imports work correctly
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

import argparse
from src.redllm.main import create_parser

def test_cli_variation_parsing():
    """Test that CLI properly parses and processes max-variations parameter."""
    print("🧪 Testing CLI Argument Processing for Dynamic Variations")
    print("=" * 60)
    
    parser = create_parser()
    
    # Test different variation values
    test_cases = [3, 6, 8, 12, 15]
    
    for max_vars in test_cases:
        print(f"\n📊 Testing --max-variations {max_vars}")
        
        # Create test args (minimal required args)
        test_args = [
            "--models", "gpt-3.5-turbo",
            "--api-keys", "dummy-key",
            "--judge-keys", "dummy-judge-key",
            "--max-variations", str(max_vars),
            "--verbose", "1"
        ]
        
        try:
            args = parser.parse_args(test_args)
            print(f"   ✅ CLI parsing successful")
            print(f"   📈 max_variations parsed as: {args.max_variations}")
            print(f"   🔧 Type: {type(args.max_variations)}")
            
            # Validate it's the expected value
            if args.max_variations == max_vars:
                print(f"   ✅ Value correctly parsed: {max_vars}")
            else:
                print(f"   ❌ Value mismatch: expected {max_vars}, got {args.max_variations}")
                
        except Exception as e:
            print(f"   ❌ CLI parsing failed: {e}")
    
    print(f"\n🎯 Testing Default Value")
    test_args_default = [
        "--models", "gpt-3.5-turbo",
        "--api-keys", "dummy-key", 
        "--judge-keys", "dummy-judge-key"
    ]
    
    try:
        args = parser.parse_args(test_args_default)
        print(f"   ✅ Default parsing successful")
        print(f"   📈 default max_variations: {args.max_variations}")
        if args.max_variations == 3:  # Expected default
            print(f"   ✅ Default value correct: 3")
        else:
            print(f"   ❌ Default value unexpected: {args.max_variations}")
    except Exception as e:
        print(f"   ❌ Default parsing failed: {e}")
    
    print(f"\n🔬 Testing Edge Cases")
    edge_cases = [1, 50, 100]
    
    for edge_val in edge_cases:
        test_args_edge = [
            "--models", "gpt-3.5-turbo",
            "--api-keys", "dummy-key",
            "--judge-keys", "dummy-judge-key", 
            "--max-variations", str(edge_val)
        ]
        
        try:
            args = parser.parse_args(test_args_edge)
            print(f"   ✅ Edge case {edge_val}: {args.max_variations}")
        except Exception as e:
            print(f"   ❌ Edge case {edge_val} failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("🎉 CLI argument processing test completed!")

if __name__ == "__main__":
    test_cli_variation_parsing()
