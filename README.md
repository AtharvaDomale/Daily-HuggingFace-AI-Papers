<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5991+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">22</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">48</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">610</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5991+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 25, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Graph Engineering in the Era of LLM Agents: From Individual Intelligence to System Intelligence</b> ⭐ 50</summary>

<br/>

**👥 Authors:** Zerui Chen, Qichao Ma, Chaobin Yang, Zhishang Xiang, Yuyuan Feng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.21156) • [📄 arXiv](https://arxiv.org/abs/2608.21156) • [📥 PDF](https://arxiv.org/pdf/2608.21156)

**💻 Code:** [⭐ Code](https://github.com/DEEP-JLU/Awesome-Graph-Engineering) • [⭐ Code](https://github.com/huggingface)

> Large language models (LLMs) have rapidly evolved from language generation models into autonomous agents capable of solving increasingly complex and long-horizon tasks. This evolution has been accom- panied by a series of emerging engineering para...

</details>

<details>
<summary><b>2. Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20061) • [📄 arXiv](https://arxiv.org/abs/2608.20061) • [📥 PDF](https://arxiv.org/pdf/2608.20061)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Training large Mixture-of-Experts (MoE) models on trillions of tokens makes conventional hyperparameter sweeps prohibitively expensive. In this work, we introduce a compute-efficient two-step framework that combines μP-based learning-rate transfer...

</details>

<details>
<summary><b>3. InfinityEdit: Infinite Video Editing with a Lightweight Edit-Ignition Adapter</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Didi Zhu, Shiyi Zhang, Canyu Zhao, Mushui Liu, Yunze Tong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20910) • [📄 arXiv](https://arxiv.org/abs/2608.20910) • [📥 PDF](https://arxiv.org/pdf/2608.20910)

**💻 Code:** [⭐ Code](https://github.com/YunzeTong/InfinityEdit) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>4. ParaTempo: Efficient Parallel Reasoning via Temporal Confidence</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16425) • [📄 arXiv](https://arxiv.org/abs/2608.16425) • [📥 PDF](https://arxiv.org/pdf/2608.16425)

**💻 Code:** [⭐ Code](https://github.com/ScottZhang812/ParaTempo) • [⭐ Code](https://github.com/huggingface)

> Efficient Parallel Reasoning 🔥

</details>

<details>
<summary><b>5. OmniAssistBench: Assistant-style Interaction Benchmark for Omni-LLMs</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.21360) • [📄 arXiv](https://arxiv.org/abs/2608.21360) • [📥 PDF](https://arxiv.org/pdf/2608.21360)

**💻 Code:** [⭐ Code](https://github.com/XianyunSun/OmniAssistBench) • [⭐ Code](https://github.com/huggingface)

> Recent advances in Omni-LLMs are paving the way for real-time video assistant applications, where models constantly perceive the environment and guide users to achieve certain goals through multi-turn conversations. However, evaluations under thes...

</details>

<details>
<summary><b>6. Beyond Correctness: Benchmarking and Aligning Response Behaviors in Hybrid-Thinking MLLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.12781) • [📄 arXiv](https://arxiv.org/abs/2608.12781) • [📥 PDF](https://arxiv.org/pdf/2608.12781)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> PatternEval reveals widespread response-pattern misalignment between thinking and non-thinking modes in hybrid-thinking MLLMs, with non-thinking inference exhibiting substantially more failures such as chain-of-thought leakage, repetition, contrad...

</details>

<details>
<summary><b>7. Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16647) • [📄 arXiv](https://arxiv.org/abs/2608.16647) • [📥 PDF](https://arxiv.org/pdf/2608.16647)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> On-policy distillation (OPD) transfers teacher capabilities by supervising trajectories sampled from the student's own policy, yet its generalization behavior remains poorly understood, as most studies evaluate OPD on a single domain and on benchm...

</details>

<details>
<summary><b>8. EviRank: Structured Relevance Evidence for Multimodal Image Re-ranking</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jinwen Luo, Xinyu Zuo, Zirong Chen, Siyi Liu, EnjunDu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20886) • [📄 arXiv](https://arxiv.org/abs/2608.20886) • [📥 PDF](https://arxiv.org/pdf/2608.20886)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> EMNLP 2026 main

</details>

<details>
<summary><b>9. UniSpace: Unified Visual Representation and Scalable Multimodal Modeling</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.08676) • [📄 arXiv](https://arxiv.org/abs/2608.08676) • [📥 PDF](https://arxiv.org/pdf/2608.08676)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yjb6/UniSpace)

> 🚀 Can one vision encoder understand, reconstruct, and generate? UniSpace explores this question by reparameterizing the patch embedding of pretrained ViTs. Instead of adding separate semantic and reconstruction encoders, it keeps the original sema...

</details>

<details>
<summary><b>10. AgentMercury: Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chanwoong Yoon, Minbyul Jeong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20634) • [📄 arXiv](https://arxiv.org/abs/2608.20634) • [📥 PDF](https://arxiv.org/pdf/2608.20634)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://huggingface.co/collections/Minbyul/agentmercury

</details>

<details>
<summary><b>11. Daedalus-150M: A Convolution-Attention Hybrid Designed for CPU Inference</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Unseen1980

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20210) • [📄 arXiv](https://arxiv.org/abs/2608.20210) • [📥 PDF](https://arxiv.org/pdf/2608.20210)

**💻 Code:** [⭐ Code](https://github.com/unseen1980/daedalus) • [⭐ Code](https://github.com/huggingface)

> Daedalus-150M is a 160M-parameter language model designed backwards from its deployment target: one user, one token at a time, 4-bit weights, an ordinary CPU. Only 6 of its 18 blocks use full attention; the other 12 use short depthwise convolution...

</details>

<details>
<summary><b>12. Partition the Support, Reconstruct the Residual: Training-Free Sparse Attention for Video Generation and World Models</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18484) • [📄 arXiv](https://arxiv.org/abs/2608.18484) • [📥 PDF](https://arxiv.org/pdf/2608.18484)

**💻 Code:** [⭐ Code](https://github.com/PardisTaghavi/SparsePR) • [⭐ Code](https://github.com/huggingface)

> SparsePR is a training-free sparse attention method for accelerating video generation. It combines response-coupled partitioning with probe-fitted residual reconstruction to reduce attention computation while preserving generation quality. We eval...

</details>

<details>
<summary><b>13. Towards Faithful Simulation of Human Shopping Behavior</b> ⭐ 0</summary>

<br/>

**👥 Authors:** See-Kiong Ng, Yang Zhang, Jing Yu, Yan Mi, TangJiakai5704

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20707) • [📄 arXiv](https://arxiv.org/abs/2608.20707) • [📥 PDF](https://arxiv.org/pdf/2608.20707)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> ✨ RecVerse: Towards Faithful Simulation of Human Shopping Behavior 🛒 A GUI-grounded shopping agent that interacts with interfaces through screenshots 🖥️ like a real user, retains relevant information using a cognitively inspired hierarchical memor...

</details>

<details>
<summary><b>14. Llama-Mobile: Efficient 2.7-Bit Quantization of VLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Douglas Orr, Jeevan Bhoot, Luka Ribar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.21134) • [📄 arXiv](https://arxiv.org/abs/2608.21134) • [📥 PDF](https://arxiv.org/pdf/2608.21134)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. Hydra-0: Action Flow for Generalist World Modeling and Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18077) • [📄 arXiv](https://arxiv.org/abs/2608.18077) • [📥 PDF](https://arxiv.org/pdf/2608.18077)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A generalist world model conditioned on action flow: robot actions represented as pixel motion. At deployment it runs as a hybrid simulator: a physics engine moves the robot, a learned video model predicts what the world does in response. One mode...

</details>

<details>
<summary><b>16. Human-Centric Intelligence in the Era of Foundation Models: A Survey</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Yihua Shao, Yilei Man, Xiaorui Jiang, Tianqi Wang, cseeychen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18184) • [📄 arXiv](https://arxiv.org/abs/2608.18184) • [📥 PDF](https://arxiv.org/pdf/2608.18184)

**💻 Code:** [⭐ Code](https://github.com/cseeyangchen/Human-Centric-AI) • [⭐ Code](https://github.com/huggingface)

> We are pleased to share Human-Centric Intelligence in the Era of Foundation Models: A Survey . This survey presents a full-spectrum view of human-centric intelligence through six interconnected context levels: visual appearance, spatial geometry, ...

</details>

<details>
<summary><b>17. PhysCaP: Grounding Code-as-Policy Agent with Physics-Informed Exploration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hung-An Chen, Hsueh-En Chang, Jing-Wen Chen, Chen-Yu Lin, cmhungsteve

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.21031) • [📄 arXiv](https://arxiv.org/abs/2608.21031) • [📥 PDF](https://arxiv.org/pdf/2608.21031)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present PhysCaP, a Physics-Informed Code-as-Policy agent for active perception in robotic manipulation. While vision-language-action policies excel at imitating demonstrations, they rely on passive observation and fail to infer latent physical ...

</details>

<details>
<summary><b>18. CLEAR: Continuous Latent Adapter Routing for Utility-Preserving LLM Safety Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.21278) • [📄 arXiv](https://arxiv.org/abs/2608.21278) • [📥 PDF](https://arxiv.org/pdf/2608.21278)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Improving the safety of large language models (LLMs) often comes at the expense of utility, as globally applied safety tuning may affect model responses to both harmful and benign inputs. We propose \textbf{C}ontinuous \textbf{L}at\textbf{E}nt \te...

</details>

<details>
<summary><b>19. Peer-Voted LLM-Agent Stress Tests Find Feed-Induced Lexical Convergence but No Reliable Matched-Exposure Advantage for Distributed Sources</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dominic Williamson, Rana Muhammad Usman

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20438) • [📄 arXiv](https://arxiv.org/abs/2608.20438) • [📥 PDF](https://arxiv.org/pdf/2608.20438)

**💻 Code:** [⭐ Code](https://github.com/ranausmanai/synthetic-social-networks) • [⭐ Code](https://github.com/huggingface)

> What happens when AI agents start shaping each other’s feeds? Across 448 preregistered trials, we find that peer-ranked feeds make agents increasingly similar in language, while four coordinated sources are not reliably more influential than one w...

</details>

<details>
<summary><b>20. FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Erim Hayretci, josefchen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20574) • [📄 arXiv](https://arxiv.org/abs/2608.20574) • [📥 PDF](https://arxiv.org/pdf/2608.20574)

**💻 Code:** [⭐ Code](https://github.com/josefchen/flavourbench) • [⭐ Code](https://github.com/huggingface)

> BREAKING: world's first culinary intelligence benchmark for LLMs now on arXiv. We tested the strongest model from 16 labs across 534 executable culinary tasks powered by Epicure. Grok 4.6 leads the pack.

</details>

<details>
<summary><b>21. WorldMind: Decoupled Game World Model for State-Aware NPC Behavior</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yeying Jin, Danze Chen, Boran Zhang, Zhiyang Deng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.21439) • [📄 arXiv](https://arxiv.org/abs/2608.21439) • [📥 PDF](https://arxiv.org/pdf/2608.21439)

**💻 Code:** [⭐ Code](https://github.com/TeaWhiteBro/WorldMind) • [⭐ Code](https://github.com/huggingface)

> Game world models have recently demonstrated promising capabilities in generating visually coherent and action-controllable gameplay videos. However, non-player character (NPC) behavior in existing models is either implicitly entangled with video ...

</details>

<details>
<summary><b>22. Hadith computational science in the age of large language models: a critical narrative review</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Riasat Islam, Md. Ashraful Haque

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20364) • [📄 arXiv](https://arxiv.org/abs/2608.20364) • [📥 PDF](https://arxiv.org/pdf/2608.20364)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> new

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-08-25.json`](data/daily/2026-08-25.json) | 22 |
| 📆 This Week | [`2026-W34.json`](data/weekly/2026-W34.json) | 48 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 610 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-25 | 22 | [View JSON](data/daily/2026-08-25.json) |
| 📄 2026-08-24 | 26 | [View JSON](data/daily/2026-08-24.json) |
| 📄 2026-08-23 | 26 | [View JSON](data/daily/2026-08-23.json) |
| 📄 2026-08-22 | 26 | [View JSON](data/daily/2026-08-22.json) |
| 📄 2026-08-21 | 22 | [View JSON](data/daily/2026-08-21.json) |
| 📄 2026-08-20 | 33 | [View JSON](data/daily/2026-08-20.json) |
| 📄 2026-08-19 | 42 | [View JSON](data/daily/2026-08-19.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W34 | 48 | [View JSON](data/weekly/2026-W34.json) |
| 📅 2026-W33 | 213 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 610 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
