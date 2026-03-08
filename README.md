<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-24-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2796+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">24</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">201</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">229</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2796+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 08, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03756) • [📄 arXiv](https://arxiv.org/abs/2603.03756) • [📥 PDF](https://arxiv.org/pdf/2603.03756)

**💻 Code:** [⭐ Code](https://github.com/ZonglinY/MOOSE-Star)

> Most current LLMs for scientific discovery rely on inference-time prompting or external feedback for training. But how can we directly train an LLM to generate scientific hypotheses from a research background , i.e., P(h|b)? In this work, we theor...

</details>

<details>
<summary><b>2. SkillNet: Create, Evaluate, and Connect AI Skills</b> ⭐ 147</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04448) • [📄 arXiv](https://arxiv.org/abs/2603.04448) • [📥 PDF](https://arxiv.org/pdf/2603.04448)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/SkillNet)

> From reinventing solutions to accumulating skills—SkillNet builds the infrastructure for lifelong learning agents.

</details>

<details>
<summary><b>3. DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04743) • [📄 arXiv](https://arxiv.org/abs/2603.04743) • [📥 PDF](https://arxiv.org/pdf/2603.04743)

**💻 Code:** [⭐ Code](https://github.com/AMA-CMFAI/DARE)

> We introduce DARE, an embedding model for improving LLM Agents on R package retrieval and downstream statistical analysis tasks. DARE outperforms open-sourced embedding models on R retrieval with higher efficiency and accuracy. Paper: https://arxi...

</details>

<details>
<summary><b>4. AgentVista: Evaluating Multimodal Agents in Ultra-Challenging Realistic Visual Scenarios</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23166) • [📄 arXiv](https://arxiv.org/abs/2602.23166) • [📥 PDF](https://arxiv.org/pdf/2602.23166)

**💻 Code:** [⭐ Code](https://github.com/hkust-nlp/AgentVista)

> Benchmarking multimodal agents on realistic, ultra-challenging visual scenarios requiring long-horizon hybrid tool use.

</details>

<details>
<summary><b>5. RoboPocket: Improve Robot Policies Instantly with Your Phone</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.05504) • [📄 arXiv](https://arxiv.org/abs/2603.05504) • [📥 PDF](https://arxiv.org/pdf/2603.05504)

> RoboPocket enables interactive online policy finetuning through smartphone AR: users see what the robot intends to do, provide corrections on-the-fly, and watch the policy improve in minutes, all without deploying to real hardware.

</details>

<details>
<summary><b>6. HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02210) • [📄 arXiv](https://arxiv.org/abs/2603.02210) • [📥 PDF](https://arxiv.org/pdf/2603.02210)

**💻 Code:** [⭐ Code](https://github.com/Correr-Zhou/HiFi-Inpaint)

> [🔥CVPR 2026] HiFi-Inpaint enables high-fidelity reference-based inpainting. HiFi-Inpaint can seamlessly integrate product reference images into masked human images, generating high-quality human-product images with high-fidelity detail preservation.

</details>

<details>
<summary><b>7. Large Multimodal Models as General In-Context Classifiers</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23229) • [📄 arXiv](https://arxiv.org/abs/2602.23229) • [📥 PDF](https://arxiv.org/pdf/2602.23229)

**💻 Code:** [⭐ Code](https://github.com/marco-garosi/CIRCLE)

> Everyone says CLIP-like models are the gold standard for classification, while Large Multimodal Models (LMMs) should be saved strictly for complex reasoning. But what if we are drastically underestimating LMMs? 🤔 The current consensus overlooks on...

</details>

<details>
<summary><b>8. Interactive Benchmarks</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04737) • [📄 arXiv](https://arxiv.org/abs/2603.04737) • [📥 PDF](https://arxiv.org/pdf/2603.04737)

**💻 Code:** [⭐ Code](https://github.com/interactivebench/interactivebench)

> Standard benchmarks have become increasingly unreliable due to saturation, subjectivity, and poor generalization. We argue that evaluating model's ability to acquire information actively is important to assess model's intelligence. We propose Inte...

</details>

<details>
<summary><b>9. DreamWorld: Unified World Modeling in Video Generation</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Shaofeng Zhang, Yuqing Zhang, Ning Liao, Xiangdong Zhang, Boming Tan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.00466) • [📄 arXiv](https://arxiv.org/abs/2603.00466) • [📥 PDF](https://arxiv.org/pdf/2603.00466)

**💻 Code:** [⭐ Code](https://github.com/ABU121111/DreamWorld)

> Despite impressive progress in video generation, existing models remain limited to surface-level plausibility, lacking a coherent and unified understanding of the world. Prior approaches typically incorporate only a single form of world-related kn...

</details>

<details>
<summary><b>10. SageBwd: A Trainable Low-bit Attention</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02170) • [📄 arXiv](https://arxiv.org/abs/2603.02170) • [📥 PDF](https://arxiv.org/pdf/2603.02170)

**💻 Code:** [⭐ Code](https://github.com/thu-ml/SageAttention)

> SageBwd: A Trainable Low-bit Attention

</details>

<details>
<summary><b>11. Timer-S1: A Billion-Scale Time Series Foundation Model with Serial Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04791) • [📄 arXiv](https://arxiv.org/abs/2603.04791) • [📥 PDF](https://arxiv.org/pdf/2603.04791)

> Timer-S1 introduces a billion-parameter MoE time-series foundation model with serial scaling, long-context capabilities, TimeMoE/TimeSTP blocks, TimeBench data, and post-training for enhanced forecasting.

</details>

<details>
<summary><b>12. MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04800) • [📄 arXiv](https://arxiv.org/abs/2603.04800) • [📥 PDF](https://arxiv.org/pdf/2603.04800)

**💻 Code:** [⭐ Code](https://github.com/alibaba/EfficientAI/tree/main/masquant)

> Up

</details>

<details>
<summary><b>13. RealWonder: Real-Time Physical Action-Conditioned Video Generation</b> ⭐ 76</summary>

<br/>

**👥 Authors:** Hong-Xing Yu, Yue Wang, Zizhang Li, Ziyu Chen, Wei Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.05449) • [📄 arXiv](https://arxiv.org/abs/2603.05449) • [📥 PDF](https://arxiv.org/pdf/2603.05449)

**💻 Code:** [⭐ Code](https://github.com/liuwei283/RealWonder)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API PerpetualWonder: Long-Horizon Action-Conditioned 4D Scene Generation (2026)...

</details>

<details>
<summary><b>14. Locality-Attending Vision Transformer</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04892) • [📄 arXiv](https://arxiv.org/abs/2603.04892) • [📥 PDF](https://arxiv.org/pdf/2603.04892)

**💻 Code:** [⭐ Code](https://github.com/sinahmr/LocAtViT)

> LocAtViT is a method to pretrain vision transformers so that their patch representations transfer better to dense prediction (e.g., segmentation), without changing the pretraining objective.

</details>

<details>
<summary><b>15. On-Policy Self-Distillation for Reasoning Compression</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zhipeng Wang, Ran He, Zhengze Zhou, Yuanda Xu, Hejian Sang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.05433) • [📄 arXiv](https://arxiv.org/abs/2603.05433) • [📥 PDF](https://arxiv.org/pdf/2603.05433)

**💻 Code:** [⭐ Code](https://github.com/HJSang/OPSD_Reasoning_Compression)

> Reasoning models think out loud, but much of what they say is noise. We introduce OPSDC (On-Policy Self-Distillation for Reasoning Compression), a method that teaches models to reason more concisely by distilling their own concise behavior back in...

</details>

<details>
<summary><b>16. UltraDexGrasp: Learning Universal Dexterous Grasping for Bimanual Robots with Synthetic Data</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jia Zeng, Yang Tian, Zhixuan Liang, Yiman Xie, Sizhe Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.05312) • [📄 arXiv](https://arxiv.org/abs/2603.05312) • [📥 PDF](https://arxiv.org/pdf/2603.05312)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API DexImit: Learning Bimanual Dexterous Manipulation from Monocular Human Vide...

</details>

<details>
<summary><b>17. KARL: Knowledge Agents via Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.05218) • [📄 arXiv](https://arxiv.org/abs/2603.05218) • [📥 PDF](https://arxiv.org/pdf/2603.05218)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinfor...

</details>

<details>
<summary><b>18. STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.00695) • [📄 arXiv](https://arxiv.org/abs/2603.00695) • [📥 PDF](https://arxiv.org/pdf/2603.00695)

> STMI tackles RGB/NIR/TIR ReID by injecting SAM masks into attention (SFM), replacing hard token filtering with learnable query-based redistribution (STR), and modeling higher-order cross-modal relations via a unified hypergraph (CHI), achieving st...

</details>

<details>
<summary><b>19. Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23440) • [📄 arXiv](https://arxiv.org/abs/2602.23440) • [📥 PDF](https://arxiv.org/pdf/2602.23440)

**💻 Code:** [⭐ Code](https://github.com/algoprog/SLATE)

> SLATE trains LLMs to reason with search engines via RL by (1) sampling multiple continuations from a shared trajectory prefix at each step instead of full trajectories, provably reducing gradient variance up to T-fold, and (2) using an LLM judge t...

</details>

<details>
<summary><b>20. Towards Multimodal Lifelong Understanding: A Dataset and Agentic Baseline</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.05484) • [📄 arXiv](https://arxiv.org/abs/2603.05484) • [📥 PDF](https://arxiv.org/pdf/2603.05484)

**💻 Code:** [⭐ Code](https://github.com/cg1177/Recursive-Multimodal-Agent)

> While datasets for video understanding have scaled to hour-long durations, they typically consist of densely concatenated clips that differ from natural, unscripted daily life. To bridge this gap, we introduce MM-Lifelong, a dataset designed for M...

</details>

<details>
<summary><b>21. Mozi: Governed Autonomy for Drug Discovery LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03655) • [📄 arXiv](https://arxiv.org/abs/2603.03655) • [📥 PDF](https://arxiv.org/pdf/2603.03655)

> This paper made an interesting exploration on how to implement controllable automated intelligent agent systems for the entire drug discovery process research.

</details>

<details>
<summary><b>22. Distribution-Conditioned Transport</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Omar Abudayyeh, Marinka Zitnik, Paolo L. B. Fischer, Gokul Gowri, Nic Fishman

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04736) • [📄 arXiv](https://arxiv.org/abs/2603.04736) • [📥 PDF](https://arxiv.org/pdf/2603.04736)

> Introducing distribution-conditioned transport (DCT): generalizes transport maps to unseen distribution pairs via distribution embeddings, enabling semi-supervised forecasting and compatibility with diverse transport models such as flow matching a...

</details>

<details>
<summary><b>23. Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Chuan Li, Amir Zadeh, Dan Haramati, Carl Qi, taldatech

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04553) • [📄 arXiv](https://arxiv.org/abs/2603.04553) • [📥 PDF](https://arxiv.org/pdf/2603.04553)

**💻 Code:** [⭐ Code](https://github.com/taldatech/lpwm)

> Latent Particle World Model (LPWM) - a self-supervised object-centric world model scaled to real-world multi-object datasets and applicable in decision-making. LPWM autonomously discovers keypoints, bounding boxes, and object masks directly from v...

</details>

<details>
<summary><b>24. Lightweight Visual Reasoning for Socially-Aware Robots</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03942) • [📄 arXiv](https://arxiv.org/abs/2603.03942) • [📥 PDF](https://arxiv.org/pdf/2603.03942)

**💻 Code:** [⭐ Code](https://github.com/alessioGalatolo/VLM-Reasoning-for-Robotics)

> 🤖 What if your VLM could look twice before answering? Most VLMs encode an image once, then reason purely in text. We think that's leaving performance on the table. We built a tiny feedback module (< 3% extra parameters) that lets the language mode...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 24 |
| 📅 Today | [`2026-03-08.json`](data/daily/2026-03-08.json) | 24 |
| 📆 This Week | [`2026-W09.json`](data/weekly/2026-W09.json) | 201 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 229 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-08 | 24 | [View JSON](data/daily/2026-03-08.json) |
| 📄 2026-03-07 | 24 | [View JSON](data/daily/2026-03-07.json) |
| 📄 2026-03-06 | 21 | [View JSON](data/daily/2026-03-06.json) |
| 📄 2026-03-05 | 41 | [View JSON](data/daily/2026-03-05.json) |
| 📄 2026-03-04 | 41 | [View JSON](data/daily/2026-03-04.json) |
| 📄 2026-03-03 | 22 | [View JSON](data/daily/2026-03-03.json) |
| 📄 2026-03-02 | 28 | [View JSON](data/daily/2026-03-02.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W09 | 201 | [View JSON](data/weekly/2026-W09.json) |
| 📅 2026-W08 | 184 | [View JSON](data/weekly/2026-W08.json) |
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 229 | [View JSON](data/monthly/2026-03.json) |
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
