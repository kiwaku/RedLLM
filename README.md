# RedLLM - Advanced LLM Red Team Framework

**A sophisticated, enterprise-grade LLM vulnerability testing framework with adaptive AI-driven attack generation and comprehensive evaluation capabilities.**

RedLLM is a powerful red-teaming tool designed for security researchers, AI safety teams, and organizations seeking to rigorously test Large Language Model defenses. Built with advanced dynamic variation generation, adaptive learning, and professional-grade reporting.

## 🚀 Quick Start with Docker

### Build the Docker Image

```bash
docker build -t redllm:latest .
```

### Run with Docker

Basic command structure:
```bash
docker run --rm -v $(pwd)/output:/app/output redllm:latest python3 src/redllm/main.py [OPTIONS]
```

**Example working command:**
```bash
docker run --rm -v $(pwd)/output:/app/output redllm:latest python3 src/redllm/main.py \
  --models "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" \
  --api-keys "your_together_ai_key_here" \
  --judge-models "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" \
  --judge-keys "your_together_ai_key_here" \
  --output "test_results.csv" \
  --variations 6 \
  --max-jailbreaks 10 \
  --jailbreaks-per-category 2 \
  --verbose 1
```

## 📋 Command Line Options

### Required Parameters
- `--models`: Comma-separated model identifiers (e.g., `"gpt-3.5-turbo,together_ai/model"`)
- `--api-keys`: Comma-separated API keys matching the models
- `--judge-models`: Models used for evaluation (default: `"gpt-4"`)
- `--judge-keys`: API keys for judge models

### Variation Control (New Clean Parameters)
- `--variations`: Number of variations to generate per jailbreak (default: 3)
- `--max-jailbreaks`: Maximum jailbreaks to test per provider (default: 50, 0=unlimited)
- `--jailbreaks-per-category`: Maximum jailbreaks per attack category (default: 5, 0=unlimited)

### Output Options
- `--output`: CSV report file (default: `"report.csv"`)
- `--details-dir`: Directory for detailed JSON files (enables hybrid output)
- `--verbose`, `-v`: Verbosity level (0=quiet, 1=normal, 2=debug)

### Advanced Options
- `--fresh`: Delete output file and cache before running
- `--deduplicate`: Remove duplicate/similar jailbreaks
- `--use-adaptive`: Enable adaptive variation generation
- `--adaptive-threshold`: Minimum leak score threshold (default: 50)
- `--use-semantic-dedup`: Use semantic similarity for deduplication

## 🏗️ Project Structure

```
RedLLM_2.0/  
├── src/redllm/               # Core source code
│   ├── main.py               # Main CLI entry point
│   ├── core/                 # Core functionality
│   │   ├── attack_engine.py  # Dynamic variation system
│   │   ├── adaptive_judge.py # Evaluation logic
│   │   └── judge.py          # Response judging
│   ├── adapters/             # LLM integration
│   └── utils/                # Utilities
├── tests/                    # Comprehensive test suite
├── docs/                     # Additional documentation
├── output/                   # Docker output directory
├── Dockerfile               # Docker configuration
├── requirements.txt         # Clean dependencies
├── .dockerignore           # Docker ignore patterns
└── .gitignore              # Git ignore patterns
```

## 🎯 Key Features

### 🧠 Advanced Attack Generation
- **6 Sophisticated Attack Strategies**: Encoding obfuscation, role-playing scenarios, academic framing, technical jailbreaks, emotional manipulation, and misdirection tactics
- **Dynamic Variation Engine**: No longer hardcoded - generates precisely the number of variations you specify (1 to 20+)  
- **Adaptive Learning System**: AI analyzes successful attacks and automatically generates refined variations
- **Semantic Deduplication**: Uses advanced similarity detection to avoid redundant attacks while maximizing coverage

### 🎯 Professional Testing Capabilities  
- **Multi-Model Testing**: Simultaneous testing across OpenAI, Anthropic, Together AI, and 50+ LLM providers
- **Comprehensive Evaluation**: Built-in AI judge system with leak score analysis and detailed reporting
- **Scalable Architecture**: Handles enterprise workloads with parallel processing and efficient resource management
- **Research-Grade Output**: Detailed JSON logs with full attack genealogy and performance metrics

### 🔒 Enterprise Security Features
- **Configurable Attack Limits**: Fine-grained control over test scope and intensity
- **Category-Based Testing**: Systematic coverage across different vulnerability types
- **Rate Limit Management**: Built-in protections for API quotas and model constraints
- **Audit Trail**: Complete logging of all tests, results, and system decisions

### Supported Models & Providers
- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo, GPT-4o, and latest models
- **Anthropic**: Claude 3 (Haiku, Sonnet, Opus), Claude 3.5 Sonnet  
- **Together AI**: 100+ open-source models including Llama, Mistral, DeepSeek
- **Google**: Gemini Pro, Gemini Flash
- **Cohere**: Command R, Command R+
- **And 50+ more**: Any model supported by LiteLLM framework

### 🚀 Why RedLLM?
- **Battle-Tested**: Developed for real-world security assessments
- **Research Validated**: Built on proven red-teaming methodologies  
- **Production Ready**: Docker containerization with CI/CD support
- **Extensible**: Plugin architecture for custom attack strategies
- **Open Source**: Full transparency for security-critical applications

## 🔧 Docker Usage Examples

### Quick Test (2 variations, 3 jailbreaks)
```bash
docker run --rm -v $(pwd)/output:/app/output redllm:latest python3 src/redllm/main.py \
  --models "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" \
  --api-keys "your_key" \
  --judge-models "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" \
  --judge-keys "your_key" \
  --variations 2 \
  --max-jailbreaks 3 \
  --verbose 1
```

### Comprehensive Test (6 variations, unlimited jailbreaks)
```bash
docker run --rm -v $(pwd)/output:/app/output redllm:latest python3 src/redllm/main.py \
  --models "gpt-3.5-turbo" \
  --api-keys "your_openai_key" \
  --judge-models "gpt-4" \
  --judge-keys "your_openai_key" \
  --variations 6 \
  --max-jailbreaks 0 \
  --details-dir "detailed_results" \
  --use-adaptive \
  --verbose 2
```

### Multiple Models Test
```bash
docker run --rm -v $(pwd)/output:/app/output redllm:latest python3 src/redllm/main.py \
  --models "gpt-3.5-turbo,together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" \
  --api-keys "openai_key,together_key" \
  --judge-models "gpt-4" \
  --judge-keys "openai_key" \
  --variations 4 \
  --verbose 1
```

## 📊 Output & Reporting

RedLLM provides comprehensive, research-grade output suitable for security assessments and academic research:

### 📈 Executive Summary (CSV)
- **Leak Score Analysis**: Quantified vulnerability scores (0-100 scale)
- **Attack Success Rates**: Success percentages by strategy and model
- **Comparative Analysis**: Side-by-side model vulnerability comparison
- **Trend Analysis**: Performance patterns across attack categories

### 🔍 Detailed Investigation (JSON)
- **Complete Attack Genealogy**: Full variation trees with parent-child relationships
- **Response Analysis**: Detailed AI judge evaluations with reasoning
- **Strategy Performance**: Per-strategy effectiveness metrics
- **Timestamp Tracking**: Complete audit trail for compliance

### 📋 Professional Reports
- **Security Assessment Format**: Industry-standard vulnerability reporting
- **Research Documentation**: Academic-quality data for publications
- **Compliance Logs**: Audit-ready documentation with full traceability

## 🛠️ Development

### Local Development (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python3 src/redllm/main.py --help
```

### Run Tests
```bash
python3 -m pytest tests/
```

## ⚠️ Important Notes

1. **Professional Use**: RedLLM is designed for legitimate security testing and research purposes
2. **API Management**: Intelligent rate limiting protects against quota exhaustion and service disruption  
3. **Data Security**: All sensitive data remains local - no external logging or data transmission
4. **Scalability**: Tested with enterprise workloads - handles thousands of variations efficiently
5. **Compliance**: Audit-ready logging suitable for security compliance frameworks

## 🎉 What Makes RedLLM Powerful

- ✅ **No Hardcoded Limits**: True dynamic variation generation - scale from 2 to 200+ variations
- ✅ **AI-Powered Intelligence**: Adaptive learning from successful attacks enhances future testing
- ✅ **Enterprise Architecture**: Production-ready containerization with professional tooling
- ✅ **Research Foundation**: Built on validated red-teaming methodologies from security research
- ✅ **Comprehensive Coverage**: 6 distinct attack vectors ensure thorough vulnerability assessment
- ✅ **Professional Output**: Industry-standard reporting suitable for executive briefings

## 🔬 Use Cases

- **Security Teams**: Comprehensive LLM defense validation
- **AI Researchers**: Systematic vulnerability analysis and documentation
- **Compliance Officers**: Audit-ready security assessment reports  
- **Product Teams**: Pre-deployment safety validation
- **Academic Research**: Reproducible experiments with detailed data collection
