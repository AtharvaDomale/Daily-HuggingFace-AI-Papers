<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7184+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">57</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">576</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7184+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 14, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. DoSA: Dots Sparse Attention</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.13990) • [📄 arXiv](https://arxiv.org/abs/2610.13990) • [📥 PDF](https://arxiv.org/pdf/2610.13990)

**💻 Code:** [⭐ Code](https://github.com/llm-sys/flash-indexer) • [⭐ Code](https://github.com/huggingface)

> DoSA skips unnecessary indexer scoring while preserving exact Top-K token selection. It uses score bounds to reject candidates early and fully scores only those still eligible. It achieves up to 2.74× Indexer speedup and 1.94× time-to-first-token ...

</details>

<details>
<summary><b>2. LeWAM: End-to-End World and Action Modeling with JEPAs</b> ⭐ 228</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12818) • [📄 arXiv](https://arxiv.org/abs/2610.12818) • [📥 PDF](https://arxiv.org/pdf/2610.12818)

**💻 Code:** [⭐ Code](https://github.com/MinghaoFu/lewam) • [⭐ Code](https://github.com/huggingface)

> JEPAs can 𝐧𝐨𝐰 jointly imagine what happens and what to do next! Meet 𝐋𝐞𝐖𝐀𝐌: A JEPA for the real world with 32.3× faster planning, 19 ms per plan Improves success 28.6% → 89.7% on contact-rich manipulations But also ... 🫶 a WAM for everyone with 17...

</details>

<details>
<summary><b>3. Harness^2: Recursive Agent Harnessing for an Open World</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14224) • [📄 arXiv](https://arxiv.org/abs/2610.14224) • [📥 PDF](https://arxiv.org/pdf/2610.14224)

**💻 Code:** [⭐ Code](https://github.com/google-research/harness2) • [⭐ Code](https://github.com/huggingface)

> TLDR: Harness² brings recursive self-improvement to open-world agents, improving the current task’s execution harness and learning from those edits to improve future harnessing, without ground-truth feedback.

</details>

<details>
<summary><b>4. UltraBERT: Beyond the Limit of BERT Pre-Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jaewoo Kang, Taewhoo Lee, Jungwoo Park

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15028) • [📄 arXiv](https://arxiv.org/abs/2610.15028) • [📥 PDF](https://arxiv.org/pdf/2610.15028)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Also please check out our blog ! Collection: dmis-lab/ultrabert UltraBERT-150M : pre-trained encoder, 32K context UltraBERT-380M : pre-trained encoder, 32K context UltraBERT-Embed-150M : ready-to-use embedding model

</details>

<details>
<summary><b>5. ClawLoop: Self-Improving Agents via Harness-Native Training Loop</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14118) • [📄 arXiv](https://arxiv.org/abs/2610.14118) • [📥 PDF](https://arxiv.org/pdf/2610.14118)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A self-improving agent running inside Claw-like harness for its own performance on general-purpose tasks

</details>

<details>
<summary><b>6. PANINI-DR: Controlled Deep Research through Generative Semantic Workspaces</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kartik Sharma, Chenda Duan, M. Efe Lorasdagi, Shreyas Rajesh, M. Yigit Turali

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15208) • [📄 arXiv](https://arxiv.org/abs/2610.15208) • [📥 PDF](https://arxiv.org/pdf/2610.15208)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Deep research requires connecting multiple steps of search, reasoning, and computation, with intermediate findings guiding what to investigate next. As investigations grow, agents must maintain research state across model calls and check findings ...

</details>

<details>
<summary><b>7. PhyProbe: Rethinking Physical Consistency Evaluation in Generated Videos</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38377) • [📄 arXiv](https://arxiv.org/abs/2609.38377) • [📥 PDF](https://arxiv.org/pdf/2609.38377)

**💻 Code:** [⭐ Code](https://github.com/vinesmsuic/PhyProbe) • [⭐ Code](https://github.com/huggingface)

> pip install phyprobe PhyProbe scores how severely a generated video violates physical consistency. It extracts features with a frozen spatiotemporal encoder (PE-Core-G14-448) and maps them to a single score with a lightweight (~1M-parameter) head....

</details>

<details>
<summary><b>8. TPPO: Tracking-Value PPO For Language Model Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhi Wang, Yun Luo, Jianhao Yan, Chenxi Li, Futing Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16794) • [📄 arXiv](https://arxiv.org/abs/2610.16794) • [📥 PDF](https://arxiv.org/pdf/2610.16794)

**💻 Code:** [⭐ Code](https://github.com/LINs-lab/TPPO) • [⭐ Code](https://github.com/huggingface)

> Relative token-level credit is determined by how value evolves along the trajectory, yet PPO's critic never models it — so we analyze successive prefix values, find a Bayesian decomposition with additive updates in log-odds space, and build TPPO o...

</details>

<details>
<summary><b>9. Scoring Higher, Answering Worse: Mitigating Reward Hacking in Rubric-Based RL via Protocol-Level Rubrics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38847) • [📄 arXiv](https://arxiv.org/abs/2609.38847) • [📥 PDF](https://arxiv.org/pdf/2609.38847)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Can an AI get a better score by giving a worse answer? Surprisingly, yes. Imagine asking whether to call 911—and getting 2,000+ characters of explanation before the model tells you to call. The answer checks plenty of boxes, but misses what matter...

</details>

<details>
<summary><b>10. UniForge3D: Joint 3D Reconstruction and Generation in a Shared Anchor-Guided Space</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15357) • [📄 arXiv](https://arxiv.org/abs/2610.15357) • [📥 PDF](https://arxiv.org/pdf/2610.15357)

**💻 Code:** [⭐ Code](https://github.com/xibushu/UniForge3D) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>11. RSIPO: Recursive Self-Improving Policy Optimization for Agentic Reinforcement Learning via Skill-Policy Co-Evolution</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15772) • [📄 arXiv](https://arxiv.org/abs/2610.15772) • [📥 PDF](https://arxiv.org/pdf/2610.15772)

**💻 Code:** [⭐ Code](https://github.com/zfkarl/RSIPO) • [⭐ Code](https://github.com/huggingface)

> RSIPO is a self-evolving agentic RL framework that enables skill-policy co-evolution and trust-calibrated self-distillation.

</details>

<details>
<summary><b>12. Spectral-Driven Adaptive-Rank Optimizer for Large Language Model Reinforcement Learning with Verifiable Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16533) • [📄 arXiv](https://arxiv.org/abs/2610.16533) • [📥 PDF](https://arxiv.org/pdf/2610.16533)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We observe a fundamental difference between RLVR and SFT: RLVR gradients progressively concentrate in a low-dimensional subspace, revealing a unique optimization geometry. Building on this insight, we propose Ares, an adaptive-rank optimizer that ...

</details>

<details>
<summary><b>13. Controlled Acquisition and Abstention in Three-Channel Score Conflicts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.10808) • [📄 arXiv](https://arxiv.org/abs/2610.10808) • [📥 PDF](https://arxiv.org/pdf/2610.10808)

**💻 Code:** [⭐ Code](https://github.com/MENGZHEGENG/audio-video-text-conflicts) • [⭐ Code](https://github.com/huggingface)

> First paper deal with multimodal information conflict.

</details>

<details>
<summary><b>14. Agent Sociodrama: Social-Dynamics-Guided Persona Recasting for LLM Agent Teams</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15788) • [📄 arXiv](https://arxiv.org/abs/2610.15788) • [📥 PDF](https://arxiv.org/pdf/2610.15788)

**💻 Code:** [⭐ Code](https://github.com/lululu896/Agent-Sociodrama) • [⭐ Code](https://github.com/huggingface)

> .

</details>

<details>
<summary><b>15. Vela: User Behavior as Supervision Sparks LLM Intelligence</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14709) • [📄 arXiv](https://arxiv.org/abs/2610.14709) • [📥 PDF](https://arxiv.org/pdf/2610.14709)

**💻 Code:** [⭐ Code](https://github.com/AllSpark-Research/Vela) • [⭐ Code](https://github.com/huggingface)

> Hi HF community! I'm Yang Xiao , the first author of Vela. As LLMs are deployed across more real-world scenarios, they must be able to learn from long-term context, understand users' evolving preferences and intent, and make decisions grounded in ...

</details>

<details>
<summary><b>16. Diffusion Editing with Soft Mask: Pixel Level Redo of Image and Video with Adjustable Strength</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00359) • [📄 arXiv](https://arxiv.org/abs/2610.00359) • [📥 PDF](https://arxiv.org/pdf/2610.00359)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🎨🎨🎨 Introduces a soft-mask guided diffusion editing framework that enables pixel-level, fine-grained control over the balance between content preservation and regeneration for image and video editing, without requiring retraining 🎨🎨🎨

</details>

<details>
<summary><b>17. AutoEnvScaling: Automating the Data Flywheel with Terminal Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14163) • [📄 arXiv](https://arxiv.org/abs/2610.14163) • [📥 PDF](https://arxiv.org/pdf/2610.14163)

**💻 Code:** [⭐ Code](https://github.com/MSR-Orchard/AutoEnvScaling) • [⭐ Code](https://github.com/huggingface)

> Recursive self-improvement needs agents that can build their own training environments. But researchers still sit in the loop, limiting it to scale. We introduce AutoEnvScaling, which turns env design into a terminal task: a proposer agent builds ...

</details>

<details>
<summary><b>18. CIViC: Evidence Gates for Audio-Conditioned Visual Prediction on Proxy Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15327) • [📄 arXiv](https://arxiv.org/abs/2610.15327) • [📥 PDF](https://arxiv.org/pdf/2610.15327)

**💻 Code:** [⭐ Code](https://github.com/MENGZHEGENG/Audio-to-Video) • [⭐ Code](https://github.com/huggingface)

> Excellent work on audio-to-video generation

</details>

<details>
<summary><b>19. CURIO: Curiosity-Driven Test-Time Learning for Open-Ended Discovery</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04851) • [📄 arXiv](https://arxiv.org/abs/2610.04851) • [📥 PDF](https://arxiv.org/pdf/2610.04851)

**💻 Code:** [⭐ Code](https://github.com/ft2023/CURIO) • [⭐ Code](https://github.com/huggingface)

> CURIO: Curiosity-Driven Test-Time Learning for Open-Ended Discovery Test-time RL lets an LLM actually learn from its own attempts on a discovery problem, but reward-only updates tend to collapse onto early high-reward trajectories and kill off dir...

</details>

<details>
<summary><b>20. ShellElastic: Scalable Elastic Pretraining with Simultaneous Maximal Updates and Hyperparameter Transfer</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Mingze Xu, Chongjian Ge, Yifan Gong, Yan Kang, Can Jin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15124) • [📄 arXiv](https://arxiv.org/abs/2610.15124) • [📥 PDF](https://arxiv.org/pdf/2610.15124)

**💻 Code:** [⭐ Code](https://github.com/jincan333/ShellElastic) • [⭐ Code](https://github.com/huggingface)

> Foundation models need to support diverse inference budgets, yet training a separate model for each budget duplicates pretraining effort. Elastic models train several nested sub-networks inside one weight-shared supernet, but existing recipes face...

</details>

<details>
<summary><b>21. SALD: Self-Referenced Advantage Learning for Diffusion Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01496) • [📄 arXiv](https://arxiv.org/abs/2610.01496) • [📥 PDF](https://arxiv.org/pdf/2610.01496)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The proposed Advantage-Guided Diffusion (AGD), Temporal Advantage Memory (TAM), and Spectral Advantage Decomposition (SAD) show strong potential to improve Diffusion Image generation. Code will be released soon....

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2026-10-14.json`](data/daily/2026-10-14.json) | 21 |
| 📆 This Week | [`2026-W41.json`](data/weekly/2026-W41.json) | 57 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 576 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-14 | 21 | [View JSON](data/daily/2026-10-14.json) |
| 📄 2026-10-13 | 12 | [View JSON](data/daily/2026-10-13.json) |
| 📄 2026-10-12 | 24 | [View JSON](data/daily/2026-10-12.json) |
| 📄 2026-10-11 | 72 | [View JSON](data/daily/2026-10-11.json) |
| 📄 2026-10-10 | 72 | [View JSON](data/daily/2026-10-10.json) |
| 📄 2026-10-09 | 35 | [View JSON](data/daily/2026-10-09.json) |
| 📄 2026-10-08 | 40 | [View JSON](data/daily/2026-10-08.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W41 | 57 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 576 | [View JSON](data/monthly/2026-10.json) |
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
