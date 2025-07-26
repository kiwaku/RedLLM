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

<details>
<summary><strong>Required Parameters</strong></summary>

- `--models`: Comma-separated model identifiers (e.g., `"gpt-3.5-turbo,together_ai/model"`)
- `--api-keys`: Comma-separated API keys matching the models
- `--judge-models`: Models used for evaluation (default: `"gpt-4"`)
- `--judge-keys`: API keys for judge models

</details>

<details>
<summary><strong>Variation Control (Clean Parameters)</strong></summary>

- `--variations`: Number of variations to generate per jailbreak (default: 3)
- `--max-jailbreaks`: Maximum jailbreaks to test per provider (default: 50, 0=unlimited)
- `--jailbreaks-per-category`: Maximum jailbreaks per attack category (default: 5, 0=unlimited)

</details>

<details>
<summary><strong>Output Options</strong></summary>

- `--output`: CSV report file (default: `"report.csv"`)
- `--details-dir`: Directory for detailed JSON files (enables hybrid output)
- `--verbose`, `-v`: Verbosity level (0=quiet, 1=normal, 2=debug)

</details>

<details>
<summary><strong>Advanced Options</strong></summary>

- `--fresh`: Delete output file and cache before running
- `--deduplicate`: Remove duplicate/similar jailbreaks
- `--use-adaptive`: Enable adaptive variation generation
- `--adaptive-threshold`: Minimum leak score threshold (default: 50)
- `--use-semantic-dedup`: Use semantic similarity for deduplication

</details>

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

<details>
<summary><strong>🧠 Advanced Attack Generation</strong></summary>

- **6 Sophisticated Attack Strategies**: Encoding obfuscation, role-playing scenarios, academic framing, technical jailbreaks, emotional manipulation, and misdirection tactics
- **Dynamic Variation Engine**: No longer hardcoded - generates precisely the number of variations you specify (1 to 20+)  
- **Adaptive Learning System**: AI analyzes successful attacks and automatically generates refined variations
- **Semantic Deduplication**: Uses advanced similarity detection to avoid redundant attacks while maximizing coverage

</details>

<details>
<summary><strong>🎯 Professional Testing Capabilities</strong></summary>

- **Multi-Model Testing**: Simultaneous testing across OpenAI, Anthropic, Together AI, and 50+ LLM providers
- **Comprehensive Evaluation**: Built-in AI judge system with leak score analysis and detailed reporting
- **Scalable Architecture**: Handles enterprise workloads with parallel processing and efficient resource management
- **Research-Grade Output**: Detailed JSON logs with full attack genealogy and performance metrics

</details>

<details>
<summary><strong>🔒 Enterprise Security Features</strong></summary>

- **Configurable Attack Limits**: Fine-grained control over test scope and intensity
- **Category-Based Testing**: Systematic coverage across different vulnerability types
- **Rate Limit Management**: Built-in protections for API quotas and model constraints
- **Audit Trail**: Complete logging of all tests, results, and system decisions

</details>

<details>
<summary><strong>📚 Academic Foundations</strong></summary>

### Prompt Design Philosophy: Research-Grounded Variation Generation

Our variation generation strategy is built on recent AI red-teaming research, incorporating Quality-Diversity optimization, compositional jailbreaks, and behavioral attack taxonomies.

Rather than naive prompt mutation, our system uses structured prompt engineering that:

- **Promotes semantic diversity** across multiple behavioral axes (misdirection, encoding, roleplay, emotional appeals)
- **Incorporates QDRT insights** (Quality-Diversity Red-Teaming) emphasizing diverse, strategically distinct adversarial inputs spanning model decision boundaries [1]
- **Uses compositional guidance** inspired by CoP (Compositions of Principles) and APRT (Automated Principle-driven Red Teaming) frameworks [2][3]
- **Treats response categories distinctly** - refusals, misdirections, and prompt leakage require different scoring heuristics and variation strategies

This principled approach ensures variations are harder to detect while covering a wider attack surface. Our system explicitly optimizes along multiple behavioral strategies while preserving adversarial goals—proven more effective than superficial lexical modifications.

**References:**
- [1] Liu et al. (2024). Quality-Diversity Red-Teaming: Framework for Evaluating Robust Jailbreaks. arXiv:2506.07121
- [2] Scherer et al. (2024). Compositions of Principles for Jailbreaking LLMs. arXiv:2506.00781  
- [3] Shinn et al. (2023). Automated Principle-Driven Red Teaming for LLMs (APRT). ACL TrustNLP

</details>

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

<details>
<summary><strong>Quick Test (2 variations, 3 jailbreaks)</strong></summary>

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

</details>

<details>
<summary><strong>Comprehensive Test (6 variations, unlimited jailbreaks)</strong></summary>

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

</details>

<details>
<summary><strong>Multiple Models Test</strong></summary>

```bash
docker run --rm -v $(pwd)/output:/app/output redllm:latest python3 src/redllm/main.py \
  --models "gpt-3.5-turbo,together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" \
  --api-keys "openai_key,together_key" \
  --judge-models "gpt-4" \
  --judge-keys "openai_key" \
  --variations 4 \
  --verbose 1
```

</details>

## 📊 Output & Reporting

RedLLM provides comprehensive, research-grade output suitable for security assessments and academic research:

<details>
<summary><strong>Executive Summary (CSV)</strong></summary>

- **Leak Score Analysis**: Quantified vulnerability scores (0-100 scale)
- **Attack Success Rates**: Success percentages by strategy and model
- **Comparative Analysis**: Side-by-side model vulnerability comparison
- **Trend Analysis**: Performance patterns across attack categories

</details>

<details>
<summary><strong>Detailed Investigation (JSON)</strong></summary>

- **Complete Attack Genealogy**: Full variation trees with parent-child relationships
- **Response Analysis**: Detailed AI judge evaluations with reasoning
- **Strategy Performance**: Per-strategy effectiveness metrics
- **Timestamp Tracking**: Complete audit trail for compliance

</details>

<details>
<summary><strong>Professional Reports</strong></summary>

- **Security Assessment Format**: Industry-standard vulnerability reporting
- **Research Documentation**: Academic-quality data for publications
- **Compliance Logs**: Audit-ready documentation with full traceability

</details>

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

## 🙏 Acknowledgments & Data Sources

RedLLM builds upon excellent research and datasets from the security community:

### Core Jailbreak Datasets
- **[plinytheelder/red-llm](https://github.com/plinytheelder/red-llm)** - Comprehensive jailbreak prompt collection
- **[elder-plinius/l1b3rt4s](https://github.com/elder-plinius/l1b3rt4s)** - Model-specific jailbreak repository  
- **[elder-plinius/cl4r1t4s](https://github.com/elder-plinius/cl4r1t4s)** - Base prompt and template collection

### Research Foundation
This framework incorporates methodologies and insights from:
- **Pliny the Elder's Red Team Research** - Foundational jailbreak taxonomy and evaluation methods
- **Academic Red-Teaming Literature** - Systematic vulnerability assessment approaches
- **Open Source Security Community** - Collaborative improvement of LLM safety testing

### Special Thanks
- **@plinytheelder** and **@elder-plinius** for pioneering open-source LLM red-teaming research
- **DSPy Framework** - For structured prompting and reliable LLM interactions
- **LiteLLM** - For unified multi-provider LLM access
- **The broader AI Safety community** - For continued focus on responsible AI development

**Note**: RedLLM is designed for legitimate security research and responsible disclosure. All included datasets are used in accordance with their respective licenses for security research purposes.
