# 🧠 Enhanced RedLLM: Adaptive Variation Generation

## Overview

The enhanced RedLLM system introduces sophisticated, user-configurable variation generation with adaptive exploration capabilities. This upgrade maintains full backward compatibility while adding powerful new features for advanced red-team testing.

## 🎯 Key Features

### 1. **Configurable Variation Count**
- `--max-variations N`: Generate up to N variations per jailbreak (default: 3)
- Replaces fixed 3-variation limit with user-specified counts
- Scales efficiently from 1 to 20+ variations

### 2. **6 Core Atomic Strategies**
- **Encoding**: Base64, unicode, ROT13, character substitution
- **Role-Playing**: Fictional characters, personas, creative scenarios  
- **Chain-of-Thought**: Multi-step reasoning, logical progression
- **Emotional**: Urgency, empathy, social pressure tactics
- **Technical**: Academic framing, research context, professional inquiry
- **Misdirection**: Hidden intent, nested instructions, innocent wrappers

### 3. **Adaptive Exploration**
- `--use-adaptive`: Enable performance-based strategy expansion
- `--adaptive-threshold N`: Set minimum leak score for expansion (default: 50)
- Automatically focuses on successful strategies
- Learns from evaluation results to improve future generations

### 4. **Semantic Deduplication**  
- `--use-semantic-dedup`: Use AI-powered similarity detection
- Removes semantically similar variations (not just text duplicates)
- Requires `sentence-transformers` package
- Falls back to basic deduplication if unavailable

## 🚀 Quick Start

### Basic Usage (Backward Compatible)
```bash
python src/redllm/main.py \
  --models "gpt-4" \
  --api-keys "your-api-key" \
  --judge-models "gpt-4" \
  --judge-keys "your-judge-key" \
  --output results.csv
```

### Enhanced Usage with New Features
```bash
python src/redllm/main.py \
  --models "gpt-4" \
  --api-keys "your-api-key" \
  --judge-models "gpt-4" \
  --judge-keys "your-judge-key" \
  --max-variations 8 \
  --use-adaptive \
  --use-semantic-dedup \
  --adaptive-threshold 60 \
  --output enhanced_results.csv
```

## 📦 Installation

### Standard Installation
The enhanced system works with existing dependencies. No additional setup required for basic functionality.

### Enhanced Features Setup
For semantic deduplication and optimal performance:
```bash
./setup_enhanced.sh
```

Or manually:
```bash
pip install sentence-transformers torch numpy
```

## 🔧 Command Line Options

### Core Options (Existing)
- `--models`: Comma-separated model identifiers
- `--api-keys`: Matching API keys
- `--judge-models`: Judge model identifiers
- `--judge-keys`: Judge API keys
- `--output`: Output CSV file path
- `--verbose`: Verbosity level (0-2)

### Enhanced Options (New)
- `--max-variations N`: Maximum variations per jailbreak (1-20)
- `--use-adaptive`: Enable adaptive strategy expansion
- `--adaptive-threshold N`: Leak score threshold for expansion (0-100)
- `--use-semantic-dedup`: Enable semantic similarity deduplication
- `--max-per-category N`: Max variations per strategy category
- `--max-jailbreaks-per-category N`: Max jailbreaks per dataset category

## 🏗️ Architecture

### Strategy Template System
```python
STRATEGY_TEMPLATES = {
    "encoding": {
        "name": "Encoding/Obfuscation",
        "prompt_template": "Transform using encoding techniques...",
        "examples": ["base64", "unicode", "rot13"]
    },
    # ... 5 more strategies
}
```

### Adaptive Generator Flow
1. **Initial Seeds**: Generate 1 variation per strategy (6 total)
2. **Evaluation**: Score variations using leak detection
3. **Performance Tracking**: Record strategy effectiveness
4. **Adaptive Expansion**: Generate more variations from successful strategies
5. **Semantic Deduplication**: Remove similar variations using AI embeddings
6. **Final Selection**: Return top N variations

### Integration Points
- **GenerateJailbreakVariationsModule**: Enhanced with adaptive capabilities
- **ParallelMultiVariationAttack**: Updated for configurable variation counts
- **ValidationUtils**: Enhanced strategy detection and tracking
- **CLI**: Extended argument parsing for new features

## 📊 Performance Impact

### Memory Usage
- Base system: ~100MB
- With sentence-transformers: ~500MB additional
- Semantic embeddings: ~10MB per 1000 variations

### Speed Comparison
- Standard (3 variations): ~30 seconds per jailbreak
- Enhanced (8 variations): ~60 seconds per jailbreak  
- Adaptive mode: +20% time for performance tracking
- Semantic dedup: +10% time for similarity computation

## 🧪 Example Workflows

### Research-Focused Testing
```bash
# Generate diverse variations for comprehensive analysis
python src/redllm/main.py \
  --models "gpt-4,claude-3" \
  --api-keys "key1,key2" \
  --judge-models "gpt-4" \
  --judge-keys "judge-key" \
  --max-variations 12 \
  --use-adaptive \
  --use-semantic-dedup \
  --verbose 2
```

### Production Security Testing
```bash
# Focus on high-scoring variations
python src/redllm/main.py \
  --models "production-model" \
  --api-keys "prod-key" \
  --judge-models "gpt-4" \
  --judge-keys "judge-key" \
  --max-variations 6 \
  --use-adaptive \
  --adaptive-threshold 70 \
  --output security_audit.csv
```

### Strategy Analysis
```bash
# Test specific strategy effectiveness
python src/redllm/main.py \
  --models "target-model" \
  --api-keys "key" \
  --judge-models "gpt-4" \
  --judge-keys "judge-key" \
  --max-variations 18 \
  --use-adaptive \
  --max-per-category 3 \
  --verbose 2
```

## 🔍 Output Analysis

The enhanced CSV includes strategy information:
- `Best_Variation_Strategy`: Strategy used for highest-scoring variation
- `Variation_N_Strategy`: Strategy for each individual variation
- `Total_Variations_Tested`: Actual variations generated and tested
- `Successful_Variations_Count`: Variations above success threshold

## 🐛 Troubleshooting

### Common Issues

**"sentence-transformers not available"**
```bash
pip install sentence-transformers
```

**"Adaptive generator failed"**
- Check DSPy model configuration
- Verify API key has sufficient quota
- Try `--verbose 2` for detailed logging

**"Variation generation timeout"**
- Reduce `--max-variations` count
- Disable `--use-adaptive` for faster processing
- Check network connectivity

### Debug Mode
```bash
python src/redllm/main.py --verbose 2 [other options]
```

## 🔄 Migration Guide

### From Standard to Enhanced

**No changes required** - Enhanced system is fully backward compatible.

**Optional enhancements:**
1. Add `--max-variations 6` to generate more variations
2. Add `--use-semantic-dedup` for better deduplication
3. Add `--use-adaptive` for smarter strategy selection

### Performance Tuning

**For Speed**: Use standard mode without adaptive features
**For Coverage**: Use `--max-variations 10+` with semantic deduplication  
**For Intelligence**: Enable adaptive mode with appropriate threshold

## 📈 Roadmap

- **v2.1**: Multi-model adaptive feedback
- **v2.2**: Custom strategy definitions
- **v2.3**: Real-time strategy optimization
- **v2.4**: Integration with external red-team databases
