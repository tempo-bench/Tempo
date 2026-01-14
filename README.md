# TEMPO: A Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval

<p align="center">
    <!-- <a href="https://tempo-bench.github.io" target="_blank">
        <img src="https://img.shields.io/badge/🌐_Website-TEMPO_Benchmark-blue?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Website">
    </a>
    <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
        <img src="https://img.shields.io/badge/📄_Paper-ArXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="ArXiv">
    </a> -->
    <a href="https://huggingface.co/datasets/tempo26/Tempo" target="_blank">
        <img src="https://img.shields.io/badge/🤗_Dataset-Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Datasets">
    </a>
    <a href="https://github.com/tempo-bench/Tempo/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/⚖️_License-MIT-green?style=for-the-badge" alt="License">
    </a>
</p>

<p align="center">
    <img src="figures/intro.jpg" width="90%" alt="Overview of TEMPO Benchmark" style="border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
</p>

### 🚨 News
- **[2026-01]** 🛠️ **Code**: Full evaluation code and temporal metrics are released.

---

Existing temporal QA benchmarks focus on simple fact-seeking queries, while reasoning-intensive retrieval benchmarks lack temporal grounding. **TEMPO** bridges this gap as the first benchmark combining temporal reasoning with reasoning-intensive retrieval across 13 domains.

**TEMPO Features:**
1.  **1,730 Complex Queries**: Requiring deep temporal reasoning (tracking changes, trends, cross-period evidence).
2.  **Step-wise Retrieval Planning**: 3,976 decomposed steps with gold documents mapped for multi-hop evaluation.
3.  **Novel Temporal Metrics**: Introducing **Temporal Coverage@k** and **Temporal Precision@k** to measure temporal completeness.

---

## 🏆 Leaderboard

> **Note:** For the most up-to-date and interactive leaderboard, visit our [Official Website](https://tempo-bench.github.io).

| Domain      | BM25 | BGE  | Contriever | DiVeR          | E5             | GritLM | Inst-L | Qwen | Rader | ReasonIR | SBERT | SFR            |
|-------------|:----:|:----:|:----------:|:--------------:|:--------------:|:------:|:------:|:----:|:-----:|:--------:|:-----:|:--------------:|
| **Cardano**   | 13.4 | 13.1 | 12.1       | <u>29.3</u>     | **35.7**       | 21.7   | 14.6   | 20.6 | 18.6  | 22.9     | 21.4  | 28.1           |
| **Iota**      | 9.7  | 36.1 | <u>38.3</u>     | 38.2           | **41.7**       | 36.6   | 34.3   | 28.6 | 19.2  | **41.7** | 33.2  | 37.1           |
| **Monero**    | 2.8  | 14.5 | 9.9        | 20.3           | 20.0           | 14.7   | 16.9   | 11.0 | <u>21.0</u>  | 19.6     | 15.1  | **23.7**       |
| **Bitcoin**   | 6.2  | 14.4 | 13.3       | 17.4           | 16.3           | **19.1** | 15.7   | 11.4 | 14.9  | 16.3     | 14.3  | <u>17.6</u>     |
| **Economics** | 5.8  | 12.6 | 16.3       | **27.8**       | <u>25.0</u>     | 17.2   | 17.5   | 17.1 | 22.7  | 20.0     | 15.3  | 21.9           |
| **Law**       | 12.7 | 31.9 | 28.1       | <u>40.4</u>     | 34.0           | 38.3   | 37.3   | 32.0 | 33.5  | 37.9     | 33.8  | **40.8**       |
| **Political** | 32.7 | 28.2 | 31.6       | <u>45.5</u>     | **47.9**       | 41.4   | 32.6   | 38.1 | 32.4  | 35.4     | 34.6  | 44.9           |
| **History**   | 9.2  | 27.4 | 26.5       | **34.5**       | 28.7           | 27.3   | 28.5   | 25.6 | 25.8  | 34.3     | 28.7  | <u>32.4</u>     |
| **Quant**     | 2.5  | 11.7 | 11.1       | <u>27.2</u>     | 13.8           | 21.6   | 14.6   | 12.7 | **27.8** | 19.5     | 15.7  | 16.8           |
| **Travel**    | 4.6  | 23.8 | 23.7       | 26.8           | <u>28.3</u>     | 25.0   | 25.0   | 22.0 | 26.1  | 21.4     | 27.3  | **29.7**       |
| **Workplace** | 6.2  | 27.2 | 23.9       | **42.6**       | 32.9           | 30.8   | 36.2   | 30.3 | <u>36.6</u>  | 30.0     | 34.6  | 31.6           |
| **Genealogy** | 13.3 | 22.0 | 24.9       | **35.6**       | <u>33.5</u>     | 26.9   | 24.6   | 25.3 | 18.7  | 30.3     | 23.5  | 31.7           |
| **HSM**       | 21.2 | 23.2 | 18.9       | 31.0           | **37.7**       | 33.4   | 24.4   | 21.3 | 16.9  | 24.7     | 26.1  | <u>33.5</u>     |
| **Avg.**      | 10.8 | 22.0 | 21.4       | **32.0**       | <u>30.4</u>     | 27.2   | 24.8   | 22.8 | 24.2  | 27.2     | 24.9  | 30.0           |

---

## ⚙️ Setup & Usage

### 1. Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/tempo-bench/Tempo.git
cd Tempo
pip install -r requirements.txt
```

### 2. Retrieval Evaluation

Run standard retrieval evaluation using `run.py`. This supports running over all domains or specific ones.

```bash
# Evaluate retrieval on the 'history' domain with BM25
python run.py --task history --model bm25 --output_dir outputs

# Evaluate on ALL domains
python run.py --task all --model bm25 --output_dir outputs
```

### 3. Step-wise Evaluation

For deeper analysis, evaluating the intermediate retrieval steps:

```bash
python run_step.py --task economics --model e5 --output_dir outputs_steps
```

### 4. 📈 Calculating Temporal Metrics (New!)

We provide a dedicated script `temporal_metrics.py` to calculate advanced temporal metrics like **Temporal Coverage** and **Temporal Precision**. This script uses an LLM to judge the temporal relevance of retrieved documents.

**Note**: You must perform step 1 first to install the required provider libraries (e.g., `openai`, `anthropic`, `google-generativeai`).

The script automatically downloads the necessary queries and corpus data from the **Hugging Face Hub** (dataset `tempo26/Tempo`), so you don't need to manually download data files.

#### Supported Providers & Configuration
Set the environment variables for your chosen provider:

**🔷 Azure OpenAI (Default)**
```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_DEPLOYMENT_NAME="gpt-4o"
```

**🟢 OpenAI**
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o" # Optional
```

**🟠 Anthropic (Claude)**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-3-5-sonnet-20240620" # Optional
```

**🔵 Google (Gemini)**
```bash
export GOOGLE_API_KEY="AIza..."
export GEMINI_MODEL="gemini-1.5-pro" # Optional
```

#### Running the Script
Run the script specifying the provider using `--provider`.

```bash
# Using Azure (default)
python temporal_metrics.py \
    --model_results_dir ./outputs \
    --k_values 10 20 \
    --provider azure

# Using OpenAI
python temporal_metrics.py --model_results_dir ./outputs --k_values 10 --provider openai

# Using Claude
python temporal_metrics.py --model_results_dir ./outputs --k_values 10 --provider anthropic
```

---

## 📊 Benchmark Comparison

| Benchmark         | #Q     | #D | Src.          | Temp.    | Reason.  | Expert   | Step     | Cross    |
|-------------------|--------|----|---------------|:--------:|:--------:|:--------:|:--------:|:--------:|
| **BRIGHT**        | 1,384  | 12 | Mixed         | ❌        | ✅        | ✅        | ❌        | ❌        |
| **RAR-b**         | 45,745 | 17 | Mixed         | ❌        | ✅        | ✅        | ❌        | ❌        |
| NTCIR Temporalia  | 100    | Open| News/Blogs    | ✅        | ❌        | ❌        | ❌        | ❌        |
| **TEMPO (Ours)**  | **1,730**| **13**| **Stack Exch.**| ✅      | ✅      | ✅      | ✅      | ✅      |

## 📝 Citation

If you use TEMPO in your work, please cite our paper:

```bibtex
soon
```
