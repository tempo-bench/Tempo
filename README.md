# TEMPO: A Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval

<p align="center">
    <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
        <img src="https://img.shields.io/badge/ArXiv-xxxx.xxxxx-b31b1b.svg?style=for-the-badge" alt="ArXiv">
    </a>
    <a href="https://huggingface.co/datasets/tempo26/Tempo" target="_blank">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-datasets-yellow.svg?style=for-the-badge" alt="Hugging Face Datasets">
    </a>
</p>

<p align="center">
    <img src="figures/intro.jpg" width="85%" alt="Overview of BRIGHT benchmark">
</p>


Existing temporal QA benchmarks focus on simple fact-seeking queries from news corpora, while reasoning-intensive retrieval benchmarks lack temporal grounding. However, real-world information needs often require reasoning about temporal evolution and synthesizing evidence across time periods. We introduce **TEMPO**, the first benchmark combining temporal reasoning with reasoning-intensive retrieval across 13 domains. TEMPO features: (1) 1,730 complex queries requiring deep temporal reasoning such as tracking changes, identifying trends, or comparing cross-period evidence; (2) step-wise retrieval planning with 3,976 decomposed steps and gold documents mapped to each step for multi-hop evaluation; and (3) novel temporal metrics including Temporal Coverage@k and Temporal Precision@k measuring whether results span required time periods. Evaluation of 12 retrieval systems reveals substantial challenges: the best model (DiVeR) achieves only 32.0 NDCG@10 and 71.4% Temporal Coverage@10, demonstrating difficulty in retrieving temporally complete evidence. 

We believe TEMPO provides a challenging benchmark for improving temporal reasoning in retrieval and RAG systems.

---

## 🚀 Leaderboard (NDCG@10)

This table shows the performance of various retrieval models on the TEMPO benchmark.

| Domain      | BM25 | BGE  | Contriever | DiVeR          | E5             | GritLM | Inst-L | Qwen | Rader | ReasonIR | SBERT | SFR            |
|-------------|:----:|:----:|:----------:|:--------------:|:--------------:|:------:|:------:|:----:|:-----:|:--------:|:-----:|:--------------:|
| **Cardano**   | 13.4 | 13.1 | 12.1       | <u>29.3</u>     | **35.7**       | 21.7   | 14.6   | 20.6 | 18.6  | 22.9     | 21.4  | 28.1           |
| **Iota**      | 9.7  | 36.1 | <u>38.3</u>     | 38.2           | **41.7**       | 36.6   | 34.3   | 28.6 | 19.2  | **41.7** | 33.2  | 37.1           |
| **Monero**    | 2.8  | 14.5 | 9.9        | 20.3           | 20.0           | 14.7   | 16.9   | 11.0 | <u>21.0</u>  | 19.6     | 15.1  | **23.7**       |
| **Bitcoin**   | 6.2  | 14.4 | 13.3       | 17.4           | 16.3           | **19.1** | 15.7   | 11.4 | 14.9  | 16.3     | 14.3  | <u>17.6</u>     |
| **Economics** | 5.8  | 12.6 | 16.3       | **27.8**       | <u>25.0</u>     | 17.2   | 17.5   | 17.1 | 22.7  | 20.0     | 15.3  | 21.9           |
| **Law**       | 12.7 | 31.9 | 28.1       | <u>40.4</u>     | 34.0           | 38.3   | 37.3   | 32.0 | 33.5  | 37.9     | 33.8  | **40.8**       |
| **Politics**  | 32.7 | 28.2 | 31.6       | <u>45.5</u>     | **47.9**       | 41.4   | 32.6   | 38.1 | 32.4  | 35.4     | 34.6  | 44.9           |
| **History**   | 9.2  | 27.4 | 26.5       | **34.5**       | 28.7           | 27.3   | 28.5   | 25.6 | 25.8  | 34.3     | 28.7  | <u>32.4</u>     |
| **Quant**     | 2.5  | 11.7 | 11.1       | <u>27.2</u>     | 13.8           | 21.6   | 14.6   | 12.7 | **27.8** | 19.5     | 15.7  | 16.8           |
| **Travel**    | 4.6  | 23.8 | 23.7       | 26.8           | <u>28.3</u>     | 25.0   | 25.0   | 22.0 | 26.1  | 21.4     | 27.3  | **29.7**       |
| **Workplace** | 6.2  | 27.2 | 23.9       | **42.6**       | 32.9           | 30.8   | 36.2   | 30.3 | <u>36.6</u>  | 30.0     | 34.6  | 31.6           |
| **Genealogy** | 13.3 | 22.0 | 24.9       | **35.6**       | <u>33.5</u>     | 26.9   | 24.6   | 25.3 | 18.7  | 30.3     | 23.5  | 31.7           |
| **HSM**       | 21.2 | 23.2 | 18.9       | 31.0           | **37.7**       | 33.4   | 24.4   | 21.3 | 16.9  | 24.7     | 26.1  | <u>33.5</u>     |
| **Avg.**      | 10.8 | 22.0 | 21.4       | **32.0**       | <u>30.4</u>     | 27.2   | 24.8   | 22.8 | 24.2  | 27.2     | 24.9  | 30.0           |

*Best score per domain is in **bold**, second best is <u>underlined</u>.*

## 📊 Benchmark Comparison

TEMPO is designed to bridge the gap between reasoning-intensive and temporal retrieval benchmarks.

| Benchmark         | #Q     | #D | Src.          | Temp.    | Reason.  | Expert   | Step     | Cross    |
|-------------------|--------|----|---------------|:--------:|:--------:|:--------:|:--------:|:--------:|
| **BRIGHT**        | 1,384  | 12 | Mixed         | ❌        | ✅        | ✅        | ❌        | ❌        |
| **RAR-b**         | 45,745 | 17 | Mixed         | ❌        | ✅        | ✅        | ❌        | ❌        |
| NTCIR Temporalia  | 100    | Open| News/Blogs    | ✅        | ❌        | ❌        | ❌        | ❌        |
| TempQuestions     | 1,271  | Open| Freebase      | ✅        | ❌        | ❌        | ✅        | ❌        |
| ChronoQA          | 5,176  | Open| News (CN)     | ✅        | ❌        | ❌        | ❌        | ❌        |
| TIME              | 38,522 | 3  | Wiki/News/D   | ✅        | ❌        | ❌        | ❌        | ❌        |
| HistoryBankQA     | 535K   | 10 | Wikipedia     | ✅        | ❌        | ❌        | ❌        | ❌        |
| ComplexTempQA     | 100M+  | Open| Wikipedia     | ✅        | ❌        | ❌        | ✅        | ❌        |
| **TEMPO (Ours)**  | **1,730**| **13**| **Stack Exch.**| ✅      | ✅      | ✅      | ✅      | ✅      |

## ⚙️ Setup & Usage

This repository provides the code to reproduce the results on the TEMPO benchmark.

**1. Installation**

Clone the repository and install the required dependencies. We recommend using a virtual environment.

```bash
git clone https://github.com/tempo-bench/Tempo.git
cd Tempo
pip install -r requirements.txt
```

**2. Running Evaluation**

Use `run.py` for standard retrieval and `run_step.py` for step-wise evaluation.

```bash
# Example for standard evaluation on the 'history' domain with BM25
python run.py \
    --task history \
    --model bm25 \
    --output_dir outputs

# Example for step-wise evaluation on the 'economics' domain with E5
python run_step.py \
    --task economics \
    --model e5 \
    --output_dir outputs_steps \
    --encode_batch_size 4
```

See `python run.py --help` for a full list of models and arguments.

## 📝 Citation

If you use TEMPO in your work, please cite our paper:

```bibtex
soon
```
