#!/usr/bin/env python3
"""
Test script to verify enhanced logging configuration works correctly with real API calls.
This demonstrates the enhanced logging system using actual LiteLLM operations.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from redllm.utils.logging_config import setup_enhanced_logging
from redllm.adapters.llm_wrappers_improved import DSPyGenericLM
from redllm.core.adaptive_judge import AdaptiveJudge

def parse_arguments():
    """Parse command line arguments similar to main.py."""
    parser = argparse.ArgumentParser(
        description="Test enhanced logging with real API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/test_logging.py --models "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" --api-keys "your-api-key"
  python3 src/test_logging.py --models "gpt-3.5-turbo" --api-keys "your-openai-key" --verbose 2
        """
    )
    
    parser.add_argument(
        "--models", "-m",
        required=True,
        help="Comma-separated list of model names to test"
    )
    
    parser.add_argument(
        "--api-keys", "-k",
        required=True,
        help="Comma-separated list of API keys corresponding to models"
    )
    
    parser.add_argument(
        "--judge-models", "-jm",
        help="Models for judge (defaults to same as --models)"
    )
    
    parser.add_argument(
        "--judge-keys", "-jk",
        help="API keys for judge models (defaults to same as --api-keys)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Logging verbosity level (0=minimal, 1=normal, 2=detailed, 3=debug)"
    )
    
    parser.add_argument(
        "--test-types",
        choices=['basic', 'judge', 'errors', 'all'],
        default='all',
        help="What types of logging to test"
    )
    
    return parser.parse_args()

async def test_basic_llm_logging(models, api_keys, verbose_level):
    """Test basic LLM operations and logging."""
    print(f"\n🤖 Testing Basic LLM Logging (verbose={verbose_level}):")
    print("-" * 50)
    
    try:
        # Create LLM wrapper
        model_name = models[0]
        api_key = api_keys[0]
        
        print(f"📋 Testing model: {model_name}")
        
        llm = DSPyGenericLM(
            model_name=model_name,
            api_key=api_key
        )
        
        # Test simple completion
        test_prompt = "What is the capital of France? Answer in one word."
        print(f"🔍 Prompt: {test_prompt}")
        
        response = llm.basic_request(test_prompt, max_tokens=50, temperature=0.7)
        print(f"✅ Response: {response.strip()}")
        
    except Exception as e:
        print(f"❌ Error in basic LLM test: {e}")
        
        # Test if this is a critical error
        try:
            from redllm.core.adaptive_judge import AdaptiveJudge
            judge = AdaptiveJudge({}, 0.6)
            if judge._is_critical_error(e):
                print("🚨 This is detected as a CRITICAL error that would pause execution")
        except ImportError:
            pass

async def test_judge_logging(models, api_keys, judge_models, judge_keys, verbose_level):
    """Test judge system logging."""
    print(f"\n⚖️ Testing Judge System Logging (verbose={verbose_level}):")
    print("-" * 50)
    
    try:
        # Setup judge
        judge_model = judge_models[0] if judge_models else models[0]
        judge_key = judge_keys[0] if judge_keys else api_keys[0]
        
        model_configs = {
            judge_model: {"api_key": judge_key}
        }
        
        judge = AdaptiveJudge(model_configs, threshold=0.6)
        
        # Test judge evaluation
        test_prompt = "Write a harmless joke about cats."
        test_response = "Why don't cats play poker? Because they're afraid of cheetahs!"
        
        print(f"🔍 Testing prompt: {test_prompt}")
        print(f"📝 Testing response: {test_response}")
        
        result = await judge.evaluate_response(test_prompt, test_response, judge_model)
        print(f"⚖️ Judge result: {result}")
        
    except Exception as e:
        print(f"❌ Error in judge test: {e}")

def test_error_scenarios():
    """Test error handling and critical error detection."""
    print(f"\n🚨 Testing Error Scenarios:")
    print("-" * 40)
    
    try:
        from redllm.core.adaptive_judge import AdaptiveJudge
        
        # Test critical error patterns
        test_errors = [
            Exception("Authentication failed"),
            Exception("API key is invalid"),
            Exception("Rate limit exceeded"),
            Exception("Some random error"),
            Exception("Connection timeout"),
            Exception("Invalid model specified"),
        ]
        
        judge = AdaptiveJudge({}, 0.6)
        
        for error in test_errors:
            is_critical = judge._is_critical_error(error)
            status = "🚨 CRITICAL" if is_critical else "✅ Non-critical"
            print(f"{status}: {error}")
            
    except ImportError as e:
        print(f"Could not import modules for testing: {e}")

def test_logging_levels():
    """Test different logging levels and sources."""
    
    print("🧪 Testing Enhanced Logging Configuration\n")
    print("=" * 50)
    
    # Test different verbose levels
    for verbose_level in [0, 1, 2, 3]:
        print(f"\n📊 Testing verbose level {verbose_level}:")
        print("-" * 30)
        
        # Setup logging for this level
        setup_enhanced_logging(verbose_level)
        
        # Test our application logs
        redllm_logger = logging.getLogger("redllm.test")
        redllm_logger.debug("This is a DEBUG message from RedLLM")
        redllm_logger.info("This is an INFO message from RedLLM")
        redllm_logger.warning("This is a WARNING message from RedLLM")
        redllm_logger.error("This is an ERROR message from RedLLM")
        
        # Test LiteLLM-like logs (these should be filtered)
        litellm_logger = logging.getLogger("litellm")
        litellm_logger.debug("model_info: {'key': 'some-model', 'max_tokens': 1024}")  # Should be filtered
        litellm_logger.debug("Final returned optional params: {'temperature': 0.7}")   # Should be filtered
        litellm_logger.warning("Rate limit exceeded - this should appear")            # Should appear
        litellm_logger.error("Authentication failed - this should appear")            # Should appear
        
        # Test other library logs
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.debug("HTTP request details - should be filtered")
        httpx_logger.error("HTTP connection error - should appear")
        
        print()  # Add spacing between test runs

async def main():
    """Run logging tests with real API calls."""
    args = parse_arguments()
    
    # Setup enhanced logging
    setup_enhanced_logging(args.verbose)
    
    # Parse models and API keys
    models = [m.strip() for m in args.models.split(',')]
    api_keys = [k.strip() for k in args.api_keys.split(',')]
    
    judge_models = None
    judge_keys = None
    if args.judge_models:
        judge_models = [m.strip() for m in args.judge_models.split(',')]
    if args.judge_keys:
        judge_keys = [k.strip() for k in args.judge_keys.split(',')]
    
    # Validate inputs
    if len(api_keys) != len(models):
        print("❌ Error: Number of API keys must match number of models")
        return
    
    print("🧪 RedLLM Enhanced Logging Test Suite")
    print("=" * 50)
    print(f"📊 Verbose Level: {args.verbose}")
    print(f"🤖 Models: {', '.join(models)}")
    print(f"🔑 API Keys: {'*' * 20} (hidden)")
    print(f"🧪 Test Types: {args.test_types}")
    
    # Run selected tests
    if args.test_types in ['basic', 'all']:
        await test_basic_llm_logging(models, api_keys, args.verbose)
    
    if args.test_types in ['judge', 'all'] and len(models) > 0:
        await test_judge_logging(
            models, api_keys, 
            judge_models or models, 
            judge_keys or api_keys, 
            args.verbose
        )
    
    if args.test_types in ['errors', 'all']:
        test_error_scenarios()
    
    # Always show logging level comparison
    if args.test_types == 'all':
        print(f"\n🔧 Logging Level Comparison:")
        print("-" * 40)
        test_logging_levels()
    
    print("\n" + "=" * 50)
    print("✅ Enhanced Logging Test Completed!")
    print("📝 Check 'redllm.log' for the full detailed log output")
    print("\n💡 Key improvements observed:")
    print("   • LiteLLM debug noise is filtered out at lower verbose levels")
    print("   • Important errors are highlighted with colors and icons")
    print("   • Critical errors can pause execution for user input")
    print("   • Different verbose levels control detail level")
    print(f"   • Real API calls show actual LiteLLM logging behavior")

if __name__ == "__main__":
    asyncio.run(main())
