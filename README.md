<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-26-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3819+-orange?style=for-the-badge&logo=academia)](data/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/AtharvaDomale/Daily-HuggingFace-AI-Papers?style=social)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/stargazers)

**Automatically updated every day at 00:00 UTC** ⏰

[📊 View Data](data/) | [🔍 Latest Papers](data/latest.json) | [📅 Archives](#-historical-archives) | [⭐ Star This Repo](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers)

</div>

---

## 🎯 Why This Repo?

- ✅ **Saves 30+ minutes** of daily paper hunting
- ✅ **Organized archives** - daily, weekly, and monthly snapshots
- ✅ **Direct links** to arXiv, PDFs, and GitHub repositories
- ✅ **Machine-readable JSON** format for easy integration
- ✅ **Zero maintenance** - fully automated via GitHub Actions
- ✅ **Historical data** - track AI research trends over time

---

## 🚀 Who Is This For?

<table>
<tr>
<td align="center">🔬<br/><b>Researchers</b><br/>Stay current with latest developments</td>
<td align="center">💼<br/><b>ML Engineers</b><br/>Discover SOTA techniques</td>
<td align="center">📚<br/><b>Students</b><br/>Learn from cutting-edge research</td>
</tr>
<tr>
<td align="center">🏢<br/><b>Companies</b><br/>Track AI trends & competition</td>
<td align="center">📰<br/><b>Content Creators</b><br/>Find topics for blogs & videos</td>
<td align="center">🤖<br/><b>AI Enthusiasts</b><br/>Explore the latest in AI</td>
</tr>
</table>

---

## ⚡ Quick Start

### 1️⃣ Get Today's Papers (cURL)

```bash
curl https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json
```

### 2️⃣ Python Integration

```python
import requests
import pandas as pd

# Load latest papers
url = "https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json"
papers = requests.get(url).json()

# Convert to DataFrame for analysis
df = pd.DataFrame(papers)
print(f"📚 Today's papers: {len(df)}")

# Filter by stars
trending = df[df['stars'].astype(int) > 10]
print(f"🔥 Trending papers: {len(trending)}")
```

### 3️⃣ JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

async function getTodaysPapers() {
  const response = await fetch(
    'https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json'
  );
  const papers = await response.json();
  
  console.log(`📚 Found ${papers.length} papers today!`);
  papers.forEach(paper => {
    console.log(`\n📄 ${paper.title}`);
    console.log(`⭐ ${paper.stars} stars`);
    console.log(`🔗 ${paper.details.arxiv_page_url}`);
  });
}

getTodaysPapers();
```

---

## 📈 Statistics

<table>
<tr>
<td align="center"><b>📄 Today</b><br/><font size="5">26</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">26</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">198</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3819+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 11, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08083) • [📄 arXiv](https://arxiv.org/abs/2605.08083) • [📥 PDF](https://arxiv.org/pdf/2605.08083)

**💻 Code:** [⭐ Code](https://github.com/zhengkid/AutoTTS)

> Test-time scaling (TTS) has become an effective approach for improving large language model performance by allocating additional computation during inference. However, existing TTS strategies are largely hand-crafted: researchers manually design r...

</details>

<details>
<summary><b>2. Listwise Policy Optimization: Group-based RLVR as Target-Projection on the LLM Response Simplex</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06139) • [📄 arXiv](https://arxiv.org/abs/2605.06139) • [📥 PDF](https://arxiv.org/pdf/2605.06139)

> This work reveals that group-based methods in RLVR share a common geometric structure: each implicitly defines a target distribution on the response simplex and projects toward it via first-order approximation. Building on this insight, we propose...

</details>

<details>
<summary><b>3. Beyond Retrieval: A Multitask Benchmark and Model for Code Search</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04615) • [📄 arXiv](https://arxiv.org/abs/2605.04615) • [📥 PDF](https://arxiv.org/pdf/2605.04615)

**💻 Code:** [⭐ Code](https://github.com/hq-bench/coreb)

> Code search has usually been evaluated as first-stage retrieval, even though production systems rely on broader pipelines with reranking and developer-style queries. Existing benchmarks also suffer from data contamination, label noise, and degener...

</details>

<details>
<summary><b>4. HumanNet: Scaling Human-centric Video Learning to One Million Hours</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Daquan Zhou, Yufan Deng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06747) • [📄 arXiv](https://arxiv.org/abs/2605.06747) • [📥 PDF](https://arxiv.org/pdf/2605.06747)

**💻 Code:** [⭐ Code](https://github.com/DAGroup-PKU/HumanNet)

> No abstract available.

</details>

<details>
<summary><b>5. 4DThinker: Thinking with 4D Imagery for Dynamic Spatial Understanding</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05997) • [📄 arXiv](https://arxiv.org/abs/2605.05997) • [📥 PDF](https://arxiv.org/pdf/2605.05997)

**💻 Code:** [⭐ Code](https://github.com/zhangquanchen/4DThinker)

> Dynamic spatial reasoning from monocular video is essential for bridging visual intelligence and the physical world, yet remains challenging for vision-language models (VLMs). Prior approaches either verbalize spatial-temporal reasoning entirely a...

</details>

<details>
<summary><b>6. Flow-OPD: On-Policy Distillation for Flow Matching Models</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Shuang Chen, Yiming Zhao, Yu Zeng, Wenxuan Huang, Zhen Fang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08063) • [📄 arXiv](https://arxiv.org/abs/2605.08063) • [📥 PDF](https://arxiv.org/pdf/2605.08063)

**💻 Code:** [⭐ Code](https://github.com/CostaliyA/Flow-OPD)

> Introduce on-policy distillation to flow matching models.

</details>

<details>
<summary><b>7. Anisotropic Modality Align</b> ⭐ 62</summary>

<br/>

**👥 Authors:** Yue Yang, Hanzhen Zhao, Yuhui Zhang, Yijiang Li, Xiaomin Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.07825) • [📄 arXiv](https://arxiv.org/abs/2605.07825) • [📥 PDF](https://arxiv.org/pdf/2605.07825)

**💻 Code:** [⭐ Code](https://github.com/Yu-xm/Modality_Gap_Theory)

> Anisotropic Modality Align

</details>

<details>
<summary><b>8. MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.07363) • [📄 arXiv](https://arxiv.org/abs/2605.07363) • [📥 PDF](https://arxiv.org/pdf/2605.07363)

> Mixture of Indexer Sparse Attention

</details>

<details>
<summary><b>9. UniSD: Towards a Unified Self-Distillation Framework for Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06597) • [📄 arXiv](https://arxiv.org/abs/2605.06597) • [📥 PDF](https://arxiv.org/pdf/2605.06597)

> A unified framework for on-policy self-distillation

</details>

<details>
<summary><b>10. A^2RD: Agentic Autoregressive Diffusion for Long Video Consistency</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06924) • [📄 arXiv](https://arxiv.org/abs/2605.06924) • [📥 PDF](https://arxiv.org/pdf/2605.06924)

**💻 Code:** [⭐ Code](https://github.com/dxlong2000/AARD)

> We introduce A^2RD, an agentic autoregressive diffusion architecture for long video synthesis that allows  diffusion models to synthesize and self-improve long videos, achieving state-of-the-art consistency and narrative coherence over long horizons.

</details>

<details>
<summary><b>11. HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.07177) • [📄 arXiv](https://arxiv.org/abs/2605.07177) • [📥 PDF](https://arxiv.org/pdf/2605.07177)

**💻 Code:** [⭐ Code](https://github.com/DeepExperience/HyperEyes)

> Code Repo: https://github.com/DeepExperience/HyperEyes

</details>

<details>
<summary><b>12. What Matters for Diffusion-Friendly Latent Manifold? Prior-Aligned Autoencoders for Latent Diffusion</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.07915) • [📄 arXiv](https://arxiv.org/abs/2605.07915) • [📥 PDF](https://arxiv.org/pdf/2605.07915)

**💻 Code:** [⭐ Code](https://github.com/ZhengrongYue/PAE)

> 🚀 PAE: State-of-the-Art Latent Diffusion with 13x Faster Training! We are excited to present Prior-Aligned AutoEncoder (PAE), a new paradigm for constructing diffusion-friendly latent manifolds! By explicitly shaping the latent space geometry, PAE...

</details>

<details>
<summary><b>13. STARFlow2: Bridging Language Models and Normalizing Flows for Unified Multimodal Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08029) • [📄 arXiv](https://arxiv.org/abs/2605.08029) • [📥 PDF](https://arxiv.org/pdf/2605.08029)

**💻 Code:** [⭐ Code](https://github.com/apple/ml-starflow)

> Github: https://github.com/apple/ml-starflow

</details>

<details>
<summary><b>14. Normalizing Trajectory Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08078) • [📄 arXiv](https://arxiv.org/abs/2605.08078) • [📥 PDF](https://arxiv.org/pdf/2605.08078)

**💻 Code:** [⭐ Code](https://github.com/apple/ml-starflow)

> Github: https://github.com/apple/ml-starflow

</details>

<details>
<summary><b>15. From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Hongzhan Lin, Ziyang Luo, Chuxue Cao, Yuchen Tian, Jinghao Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06716) • [📄 arXiv](https://arxiv.org/abs/2605.06716) • [📥 PDF](https://arxiv.org/pdf/2605.06716)

**💻 Code:** [⭐ Code](https://github.com/FeishuLuo/Evolving-LLM-Agent-Memory-Survey)

> An evolutionary framework is proposed for LLM agent memory that unifies fragmented research into three stages—Storage (save trajectories), Reflection (refine them), and Experience (abstract them)—driven by the need for long-range consistency, dyna...

</details>

<details>
<summary><b>16. IntentGrasp: A Comprehensive Benchmark for Intent Understanding</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06832) • [📄 arXiv](https://arxiv.org/abs/2605.06832) • [📥 PDF](https://arxiv.org/pdf/2605.06832)

**💻 Code:** [⭐ Code](https://github.com/YuweiYin/IntentGrasp)

> Paper: https://arxiv.org/abs/2605.06832 Authors: Yuwei Yin , Chuyuan Li , Giuseppe Carenini Institute: UBC NLP Group , Department of Computer Science, University of British Columbia Keywords: Intent Understanding, Dataset, Benchmark, LLM, Evaluati...

</details>

<details>
<summary><b>17. LiVeAction: a Lightweight, Versatile, and Asymmetric Neural Codec Design for Real-time Operation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Neeraja J. Yadwadkar, danjacobellis

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06628) • [📄 arXiv](https://arxiv.org/abs/2605.06628) • [📥 PDF](https://arxiv.org/pdf/2605.06628)

**💻 Code:** [⭐ Code](https://github.com/ut-sysml/liveaction)

> https://ut-sysml.github.io/liveaction pip install livecodec

</details>

<details>
<summary><b>18. Fast Byte Latent Transformer</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Luke Zettlemoyer, Gargi Ghosh, Tomasz Limisiewicz, Artidoro Pagnoni, Julie Kallini

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08044) • [📄 arXiv](https://arxiv.org/abs/2605.08044) • [📥 PDF](https://arxiv.org/pdf/2605.08044)

> No abstract available.

</details>

<details>
<summary><b>19. Who Prices Cognitive Labor in the Age of Agents? Compute-Anchored Wages</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05558) • [📄 arXiv](https://arxiv.org/abs/2605.05558) • [📥 PDF](https://arxiv.org/pdf/2605.05558)

> This paper discusses how cognitive labor is priced in the age of agents

</details>

<details>
<summary><b>20. Steering Visual Generation in Unified Multimodal Models with Understanding Supervision</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Huan Yang, Cheng Da, Yang Yue, Zanlin Ni, Zeyu Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05781) • [📄 arXiv](https://arxiv.org/abs/2605.05781) • [📥 PDF](https://arxiv.org/pdf/2605.05781)

> Unified multimodal models are envisioned to bridge the gap between understanding and generation. Yet, to achieve competitive performance, state-of-the-art models adopt largely decoupled understanding and generation components. This design, while e...

</details>

<details>
<summary><b>21. UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06221) • [📄 arXiv](https://arxiv.org/abs/2605.06221) • [📥 PDF](https://arxiv.org/pdf/2605.06221)

**💻 Code:** [⭐ Code](https://github.com/qhfan/UniPrefill)

> As large language models (LLMs) continue to advance rapidly, they are becoming increasingly capable while simultaneously demanding ever-longer context lengths. To improve the inference efficiency of long-context processing, several novel low-compl...

</details>

<details>
<summary><b>22. MDN: Parallelizing Stepwise Momentum for Delta Linear Attention</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Zunchang Liu, Xiaopeng Lin, Hongxiang Huang, Xiang Liu, huuuuyulong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05838) • [📄 arXiv](https://arxiv.org/abs/2605.05838) • [📥 PDF](https://arxiv.org/pdf/2605.05838)

**💻 Code:** [⭐ Code](https://github.com/HuuYuLong/MomentumDeltaNet)

> We introduce Momentum DeltaNet (MDN), a new Delta Linear Attention model that combines stepwise momentum, chunkwise-parallel training, and stability-aware gating to improve both efficiency and performance in long-sequence language modeling.

</details>

<details>
<summary><b>23. CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jun Wang, Yi Chang, Hechang Chen, Yali Du, Siyuan Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06702) • [📄 arXiv](https://arxiv.org/abs/2605.06702) • [📥 PDF](https://arxiv.org/pdf/2605.06702)

**💻 Code:** [⭐ Code](https://github.com/guosyjlu/CASCADE)

> Large language models (LLMs) have become a central foundation of modern artificial intelligence, yet their lifecycle remains constrained by a rigid separation between training and deployment, after which learning effectively ceases. This limitatio...

</details>

<details>
<summary><b>24. Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03473) • [📄 arXiv](https://arxiv.org/abs/2602.03473) • [📥 PDF](https://arxiv.org/pdf/2602.03473)

**💻 Code:** [⭐ Code](https://github.com/LMMMEng/CaRE)

> A strong continual learner capable of scaling to over 300 tasks.

</details>

<details>
<summary><b>25. InterLV-Search: Benchmarking Interleaved Multimodal Agentic Search</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Sicong Leng, Ronghao Dang, Jiayan Guo, Jiuning Gu, Bohan Hou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.07510) • [📄 arXiv](https://arxiv.org/abs/2605.07510) • [📥 PDF](https://arxiv.org/pdf/2605.07510)

**💻 Code:** [⭐ Code](https://github.com/hbhalpha/InterLV-Search-Bench)

> No abstract available.

</details>

<details>
<summary><b>26. CPCANet: Deep Unfolding Common Principal Component Analysis for Domain Generalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05136) • [📄 arXiv](https://arxiv.org/abs/2605.05136) • [📥 PDF](https://arxiv.org/pdf/2605.05136)

**💻 Code:** [⭐ Code](https://github.com/wish44165/CPCANet)

> We propose CPCANet, which unrolls CPCA into a differentiable network to learn a structured domain-invariant subspace for domain generalization. It is architecture-agnostic and tuning-free, and achieves SOTA performance on standard benchmarks.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 26 |
| 📅 Today | [`2026-05-11.json`](data/daily/2026-05-11.json) | 26 |
| 📆 This Week | [`2026-W19.json`](data/weekly/2026-W19.json) | 26 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 198 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-11 | 26 | [View JSON](data/daily/2026-05-11.json) |
| 📄 2026-05-10 | 38 | [View JSON](data/daily/2026-05-10.json) |
| 📄 2026-05-09 | 38 | [View JSON](data/daily/2026-05-09.json) |
| 📄 2026-05-08 | 18 | [View JSON](data/daily/2026-05-08.json) |
| 📄 2026-05-07 | 3 | [View JSON](data/daily/2026-05-07.json) |
| 📄 2026-05-06 | 8 | [View JSON](data/daily/2026-05-06.json) |
| 📄 2026-05-05 | 3 | [View JSON](data/daily/2026-05-05.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W19 | 26 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 198 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |
| 🗓️ 2025-12 | 787 | [View JSON](data/monthly/2025-12.json) |

---

## ✨ Features

- 🔄 **Automated Daily Updates** - Runs every day at midnight UTC
- 📊 **Comprehensive Data** - Abstracts, authors, links, and metadata
- 🗄️ **Historical Archives** - Daily, weekly, and monthly snapshots
- 🔗 **Direct Links** - arXiv, PDF, GitHub repos, and HuggingFace pages
- 📈 **Trending Papers** - Star counts and popularity metrics
- 💾 **JSON Format** - Easy to parse and integrate into your projects
- 🎨 **Clean Interface** - Beautiful, organized README

---

## 🚀 Usage

### View Papers

- **Latest Papers**: Check this README (updated daily)
- **JSON Data**: Download from [`data/latest.json`](data/latest.json)
- **Historical Data**: Browse the [`data/`](data/) directory

### Integrate Into Your Project

```python
import requests

# Get latest papers
response = requests.get('https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json')
papers = response.json()

for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"arXiv: {paper['details']['arxiv_page_url']}")
    print(f"PDF: {paper['details']['pdf_url']}")
```

### Use as RSS Alternative

Monitor this repo for daily AI paper updates:
- ⭐ Star this repository
- 👀 Watch for notifications
- 🔔 Enable "All Activity" for daily updates

---

## 📊 Data Structure

```
data/
├── daily/              # Individual day snapshots
│   ├── 2024-12-04.json
│   ├── 2024-12-05.json
│   └── ...
├── weekly/             # Cumulative weekly papers
│   ├── 2024-W48.json
│   └── ...
├── monthly/            # Cumulative monthly papers
│   ├── 2024-12.json
│   └── ...
└── latest.json         # Most recent scrape
```

### JSON Schema

```json
{
  "title": "Paper Title",
  "paper_url": "https://huggingface.co/papers/...",
  "authors": ["Author 1", "Author 2"],
  "stars": "42",
  "scraped_date": "2024-12-04",
  "details": {
    "abstract": "Paper abstract...",
    "arxiv_page_url": "https://arxiv.org/abs/...",
    "pdf_url": "https://arxiv.org/pdf/...",
    "github_links": ["https://github.com/..."],
    "metadata": {}
  }
}
```

---

## 🛠️ How It Works

This repository uses:

- **[Crawl4AI](https://github.com/unclecode/crawl4ai)** - Modern web scraping framework
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** - HTML parsing
- **[GitHub Actions](https://github.com/features/actions)** - Automated daily runs
- **Python 3.11+** - Data processing and generation

### Workflow

1. 🕐 GitHub Actions triggers at 00:00 UTC daily
2. 🔍 Scrapes HuggingFace Papers page
3. 📥 Downloads detailed info for each paper
4. 💾 Saves to daily/weekly/monthly archives
5. 📝 Generates this beautiful README
6. ✅ Commits and pushes updates

---

## 🤝 Contributing

Found a bug or have a feature request? 

- 🐛 [Report Issues](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/issues)
- 💡 [Submit Ideas](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/discussions)
- 🔧 [Pull Requests Welcome](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/pulls)

---

## 📜 License

MIT License - feel free to use this data for your own projects!

See [LICENSE](LICENSE) for more details.

---

## 🌟 Star History

If you find this useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=AtharvaDomale/Daily-HuggingFace-AI-Papers&type=Date)](https://star-history.com/#AtharvaDomale/Daily-HuggingFace-AI-Papers&Date)

---

## 📬 Contact & Support

- 💬 [GitHub Discussions](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/discussions)
- 🐛 [Issue Tracker](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/issues)
- ⭐ Don't forget to star this repo!

---

<div align="center">

**Made with ❤️ for the AI Community**

[⬆ Back to Top](#-daily-huggingface-ai-papers)

</div>
