#!/usr/bin/env python3
"""End-to-end integration test for dynamic variation system."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from src.redllm.main import RedLLMConfig, create_parser
from src.redllm.core.attack_engine import ParallelMultiVariationAttack, create_attack_engine

class MockArgs:
    """Mock arguments for testing."""
    def __init__(self, max_variations):
        self.models = "gpt-3.5-turbo" 
        self.api_keys = "dummy-key"
        self.judge_models = "gpt-4"
        self.judge_keys = "dummy-judge-key"
        self.output = "test_output.csv"
        self.details_dir = None
        self.fresh = False
        self.deduplicate = False
        self.max_variations = max_variations
        self.max_per_category = 2
        self.max_jailbreaks_per_category = 5
        self.verbose = 1
        self.use_adaptive = False
        self.adaptive_threshold = 50
        self.use_semantic_dedup = False

def test_end_to_end_integration():
    """Test complete integration from CLI args to attack engine configuration."""
    print("🚀 End-to-End Dynamic Variation Integration Test")
    print("=" * 65)
    
    test_values = [3, 6, 8, 12, 20]
    
    for max_vars in test_values:
        print(f"\n🎯 Testing max_variations = {max_vars}")
        print("-" * 45)
        
        # 1. CLI Argument Processing
        mock_args = MockArgs(max_vars)
        config = RedLLMConfig(mock_args)
        
        print(f"   📋 Config max_variations: {config.max_variations}")
        
        # 2. Attack Engine Creation (using factory function)
        attack_engine = create_attack_engine(
            max_total_variations=config.max_variations,
            max_workers=3
        )
        
        print(f"   🏭 Factory max_variations: {attack_engine.max_variations}")
        
        # 3. Validate Generator Configuration (check if attack engine has the generator)
        if hasattr(attack_engine, 'generator'):
            generator = attack_engine.generator
            if hasattr(generator, 'max_variations'):
                print(f"   🎲 Generator max_variations: {generator.max_variations}")
            else:
                print(f"   ⚠️  Generator doesn't have max_variations attribute")
        else:
            print(f"   ℹ️  Attack engine doesn't use separate generator")
        
        # 4. Validate Component Configuration  
        if hasattr(attack_engine, 'generate_variations'):
            generator = attack_engine.generate_variations
            if hasattr(generator, 'max_variations'):
                print(f"   🎲 Variation module max_variations: {generator.max_variations}")
            else:
                print(f"   ⚠️  Variation module doesn't have max_variations")
        else:
            print(f"   ℹ️  No variation generator found")
        
        # 5. Integration Validation
        config_match = (config.max_variations == max_vars)
        engine_match = (attack_engine.max_variations == max_vars)
        
        all_match = config_match and engine_match
        
        if all_match:
            print(f"   ✅ INTEGRATION SUCCESS: All components configured for {max_vars}")
        else:
            print(f"   ❌ INTEGRATION FAILURE:")
            print(f"      Config: {config.max_variations} ({'✅' if config_match else '❌'})")
            print(f"      Engine: {attack_engine.max_variations} ({'✅' if engine_match else '❌'})")
    
    print(f"\n🎉 End-to-End Integration Test Complete!")
    print("=" * 65)

def test_parameter_propagation():
    """Test that parameters properly propagate through the entire system."""
    print("\\n🔗 Parameter Propagation Validation")
    print("=" * 45)
    
    # Test CLI → Config → Engine → Components chain
    parser = create_parser()
    
    # Simulate CLI parsing
    cli_args = [
        "--models", "gpt-3.5-turbo",
        "--api-keys", "dummy",
        "--judge-keys", "dummy",
        "--max-variations", "15",
        "--use-adaptive",
        "--use-semantic-dedup"
    ]
    
    args = parser.parse_args(cli_args)
    config = RedLLMConfig(args)
    
    print(f"CLI Input: --max-variations 15")
    print(f"Config Output: {config.max_variations}")
    print(f"Config adaptive: {getattr(config, 'use_adaptive', False)}")
    print(f"Config semantic_dedup: {getattr(config, 'use_semantic_dedup', False)}")
    
    # Create engine using config
    engine = create_attack_engine(
        max_total_variations=config.max_variations,
        max_workers=3
    )
    
    print(f"Engine max_variations: {engine.max_variations}")
    
    # Validate complete chain (simplified)
    if (args.max_variations == config.max_variations == engine.max_variations == 15):
        print("✅ COMPLETE PROPAGATION SUCCESS: 15 → 15 → 15")
    else:
        print("❌ PROPAGATION FAILURE: Values don't match through chain")

if __name__ == "__main__":
    test_end_to_end_integration()
    test_parameter_propagation()
