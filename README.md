<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7456+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">21</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">42</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">848</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7456+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 20, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. MT Forcing: Towards Stable and Consistent Multi-Turn Image Editing with Autoregressive Diffusion</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25404) • [📄 arXiv](https://arxiv.org/abs/2610.25404) • [📥 PDF](https://arxiv.org/pdf/2610.25404)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/houyuanchen111/MT-Forcing)

> Introducing MT Forcing : towards stable and consistent multi-turn image editing with autoregressive diffusion. Image editing models often produce good results in a single turn, yet accumulate artifacts and lose previously established details over ...

</details>

<details>
<summary><b>2. Self-Evolving Coding Agents via Harness-Policy Optimization</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22791) • [📄 arXiv](https://arxiv.org/abs/2610.22791) • [📥 PDF](https://arxiv.org/pdf/2610.22791)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MSR-Orchard/Self-HPO)

> 🤩 Introducing Self-HPO: let the coding agent improve its own harness, then train a stronger policy model with the optimized harness, and repeat the process for recursive improvement! 💪 Self-evolving: no external model required 🎯 Harness search tar...

</details>

<details>
<summary><b>3. RoAR-VAE: Improving Latent Diffusability through Robust Prediction</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26216) • [📄 arXiv](https://arxiv.org/abs/2610.26216) • [📥 PDF](https://arxiv.org/pdf/2610.26216)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LeapLabTHU/RoAR-VAE)

> This project aims to develop an open VAE training recipe with performance comparable to FLUX.2-VAE. Revisiting REPA-E, we identify robust prediction as an effective regularization principle: requiring useful information to remain recoverable from ...

</details>

<details>
<summary><b>4. ScaleNav: Scalable Internet Video Pretraining for Language-Conditioned Local Navigation</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Jia Shi, Zhongrui Yu, Jiawei Wang, Yuanhong Yu, Yihui Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25690) • [📄 arXiv](https://arxiv.org/abs/2610.25690) • [📥 PDF](https://arxiv.org/pdf/2610.25690)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MachEmbodied/ScaleNav)

> We are excited to share ScaleNav: Scalable Internet Video Pretraining for Language-Conditioned Local Navigation . ScaleNav uses scalable Internet video pretraining to learn navigation priors from action-free egocentric videos. A unified world-acti...

</details>

<details>
<summary><b>5. OR-License: Evaluating LLM Agents as Real-World Operations Research Practitioners</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.24087) • [📄 arXiv](https://arxiv.org/abs/2610.24087) • [📥 PDF](https://arxiv.org/pdf/2610.24087)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NineAbyss/OR-License)

> 🚀OR-License benchmarks LLM agents as real-world OR practitioners using industry-derived optimization problems from $3.68M in commissioned projects, evaluating end-to-end OR problem solving. We evaluated 11 frontier LLMs and analyzed their performa...

</details>

<details>
<summary><b>6. Frechet Forcing: Training Long-Horizon Streaming World Models from Short Clips</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Jian Zhang, Ruochong Zheng, Hongcan Guo, Yu Yan, Yian Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25666) • [📄 arXiv](https://arxiv.org/abs/2610.25666) • [📥 PDF](https://arxiv.org/pdf/2610.25666)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zhao-yian/Frechet_Forcing)

> Frechet Forcing: Training Long-Horizon Streaming World Models from Short Clips

</details>

<details>
<summary><b>7. AnyGoal: On World Model Generalization Beyond Expert Rollouts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26844) • [📄 arXiv](https://arxiv.org/abs/2610.26844) • [📥 PDF](https://arxiv.org/pdf/2610.26844)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/galilai-group/AnyGoal)

> 💡A compositional generalization suite for world model research. pip install anygoal and see how much your world model knows about the world! Project Page: https://hazel-heejeong-nam.github.io/anygoal/ github: https://github.com/galilai-group/AnyGo...

</details>

<details>
<summary><b>8. Self-Evaluating Recursive Agents</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04902) • [📄 arXiv](https://arxiv.org/abs/2610.04902) • [📥 PDF](https://arxiv.org/pdf/2610.04902)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OliverLeeXZ/SERA)

> 🚀 Excited to share our latest work: Self-Evaluating Recursive Agents (SERA) ! As coding agents such as Claude Code and OpenAI Codex increasingly adopt task decomposition and subagent delegation, recursive agent architectures are emerging as a prom...

</details>

<details>
<summary><b>9. HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-offs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junru Lu, Honglin Lin, Yukuo Cen, ZhiYuan Feng, Zhuoshi Pan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26770) • [📄 arXiv](https://arxiv.org/abs/2610.26770) • [📥 PDF](https://arxiv.org/pdf/2610.26770)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/pzs19/HarnessPro)

> HarnessPro recursively improves agent harnesses to achieve better task performance at lower execution cost, offering multiple performance–cost trade-offs for different deployment budgets.

</details>

<details>
<summary><b>10. Learning to Discriminate Scene Structures Makes Self-Supervised Depth Learning Scalable</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23600) • [📄 arXiv](https://arxiv.org/abs/2610.23600) • [📥 PDF](https://arxiv.org/pdf/2610.23600)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/mengtanZ/MaSS)

> Beyond achieving SoTA performance across diverse datasets with a unified model, MaSS is the first SSMDE method to effectively benefit from increased data diversity and exhibit remarkable generalizability to entirely out-of-distribution scenes. Cod...

</details>

<details>
<summary><b>11. DeOverlay: A Benchmark and Source-Anchored Flow for Video Overlay Removal</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yue Zhang, Wenzhao Gao, Shaoyong Jia, Yuhang Ma, Deshui Miao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26081) • [📄 arXiv](https://arxiv.org/abs/2610.26081) • [📥 PDF](https://arxiv.org/pdf/2610.26081)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A Benchmark and Source-Anchored Flow for Video Overlay Removal

</details>

<details>
<summary><b>12. Co-Evolving Acting and Forecasting for Language Agents</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Yuxiang Liu, Xuyan Ye, Yining Zhao, Yixing Chen, Haofei Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20597) • [📄 arXiv](https://arxiv.org/abs/2610.20597) • [📥 PDF](https://arxiv.org/pdf/2610.20597)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ulab-uiuc/CoPE)

> Agentic reinforcement learning learns only from a terminal scalar reward, so most of a long rollout goes unused. Per-step value targets all derive from that one scalar, and next-observation targets are long, noisy, and reach only one step ahead. C...

</details>

<details>
<summary><b>13. RSIFlow: Adaptive Self-Improvement Workflows with an Evolving Meta-Agent</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21791) • [📄 arXiv](https://arxiv.org/abs/2610.21791) • [📥 PDF](https://arxiv.org/pdf/2610.21791)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/1HUAN1/RSI-Flow)

> Large language model (LLM) agents have demonstrated strong capabilities in complex tasks, including code generation, tool use, and multi-hop question answering. Existing self-improvement methods enhance agent performance through model optimization...

</details>

<details>
<summary><b>14. World Model Interpretability: From Decoding Representations to Understanding Latent Worlds</b> ⭐ 95</summary>

<br/>

**👥 Authors:** Xinping Guan, Cailian Chen, Xinyi Le, Yongcheng Jing, Jize Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25296) • [📄 arXiv](https://arxiv.org/abs/2610.25296) • [📥 PDF](https://arxiv.org/pdf/2610.25296)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Jize-W/Awesome-World-Model-Interpretability)

> This Perspective aims to establish latent-world understanding as a central goal for interpreting predictive world models. We note that decodability is not understanding: accessible information does not reveal how it is organized, evolves, or causa...

</details>

<details>
<summary><b>15. CoworkEnv: Synthesizing Training Environments for Long-Horizon Cowork Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chunyang Li, Baixuan Xu, Jiaxin Bai, Yiming Du, Zhongwei Xie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.24960) • [📄 arXiv](https://arxiv.org/abs/2610.24960) • [📥 PDF](https://arxiv.org/pdf/2610.24960)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HKUST-KnowComp/CoworkEnv)

> How can we build training environments for AI coworkers that need to use tools, handle multi-turn interactions, and operate in changing workspaces? We introduce CoworkEnv, a framework that synthesizes executable environments for training these age...

</details>

<details>
<summary><b>16. EgoXperience: Learning World-Grounded Egocentric Representations from Human Experience</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26625) • [📄 arXiv](https://arxiv.org/abs/2610.26625) • [📥 PDF](https://arxiv.org/pdf/2610.26625)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Ropedia/EgoXperience)

> The SOTA mocap pipeline and model.

</details>

<details>
<summary><b>17. Remembering Has a Price: Learning Memory Worth under Resource Constraints</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26453) • [📄 arXiv](https://arxiv.org/abs/2610.26453) • [📥 PDF](https://arxiv.org/pdf/2610.26453)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIGNLAI/TokenWorth)

> We introduce TokenWorth , a lightweight online memory framework for lifelong agents operating under tight resource budgets. By learning query-dependent memory utility from execution feedback, TokenWorth improves memory retrieval and management, ac...

</details>

<details>
<summary><b>18. Think Visually Across Space and Time: Grounded Reasoning for Video Segmentation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xin Li, Ming-Hsuan Yang, Guangming Lu, Deshui Miao, Junwei Wen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25090) • [📄 arXiv](https://arxiv.org/abs/2610.25090) • [📥 PDF](https://arxiv.org/pdf/2610.25090)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A spatiotemporally grounded form of Video CoT that organizes query-relevant observations into a chronological reasoning chain.

</details>

<details>
<summary><b>19. Empowering Edge Agents: A Systematic Analysis of Edge-Cloud Collaboration in LLM Agent Execution</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17816) • [📄 arXiv](https://arxiv.org/abs/2610.17816) • [📥 PDF](https://arxiv.org/pdf/2610.17816)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenBMB/EC2-Bench)

> We systematically analyze edge-cloud collaboration strategies for LLM agents across 128 executable long-horizon tasks, measuring task performance, resource cost, and privacy exposure. Key finding: a 9B edge model with collaboration achieves ~73% o...

</details>

<details>
<summary><b>20. QD-Rec: Open-Ended Evolution of Recommendations via Quality-Diversity</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23470) • [📄 arXiv](https://arxiv.org/abs/2610.23470) • [📥 PDF](https://arxiv.org/pdf/2610.23470)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sakanaai/qd-rec)

> QD-Rec repeatedly selects an item, generates a hypothesis, and updates the archive through local novelty and quality comparisons: QD-Rec's archives during search in three domains (claude-sonnet-5 proposer). Each map cell shows the highest score am...

</details>

<details>
<summary><b>21. SPARC: Spoken Profile Authorship Recognition Corpus</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25265) • [📄 arXiv](https://arxiv.org/abs/2610.25265) • [📥 PDF](https://arxiv.org/pdf/2610.25265)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/bowenyi-pierre/SPARC)

> SPARC asks a simple question, once you strip the voice from speech, how much of the speaker is still in the words? Dataset and code: https://huggingface.co/datasets/timtsapras23/SPARC https://github.com/bowenyi-pierre/SPARC

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2026-10-20.json`](data/daily/2026-10-20.json) | 21 |
| 📆 This Week | [`2026-W42.json`](data/weekly/2026-W42.json) | 42 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 848 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-20 | 21 | [View JSON](data/daily/2026-10-20.json) |
| 📄 2026-10-19 | 21 | [View JSON](data/daily/2026-10-19.json) |
| 📄 2026-10-18 | 84 | [View JSON](data/daily/2026-10-18.json) |
| 📄 2026-10-17 | 84 | [View JSON](data/daily/2026-10-17.json) |
| 📄 2026-10-16 | 34 | [View JSON](data/daily/2026-10-16.json) |
| 📄 2026-10-15 | 28 | [View JSON](data/daily/2026-10-15.json) |
| 📄 2026-10-14 | 21 | [View JSON](data/daily/2026-10-14.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W42 | 42 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 848 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |

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
