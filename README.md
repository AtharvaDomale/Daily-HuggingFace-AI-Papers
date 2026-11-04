<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-20-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7815+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">20</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">48</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">87</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7815+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 04, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Behind Video Generation: Observations and Insights on Multimodal Understanding from Kling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04916) • [📄 arXiv](https://arxiv.org/abs/2611.04916) • [📥 PDF](https://arxiv.org/pdf/2611.04916)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This technical report shares practical lessons from the evolution of Kling’s video-generation systems. It argues that the next bottleneck is no longer visual rendering, but semantic control—the ability to understand what to generate, preserve, org...

</details>

<details>
<summary><b>2. RoboColosseum: Evaluating Physical Intelligence at Scale In-The-Wild</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02497) • [📄 arXiv](https://arxiv.org/abs/2611.02497) • [📥 PDF](https://arxiv.org/pdf/2611.02497)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RoboColosseum/robocolosseum-client)

> General-purpose robot policies must work across robots, tasks, and environments, yet the variation that makes physical trials informative also makes them difficult to reproduce. We introduce RoboColosseum, a distributed evaluation framework that c...

</details>

<details>
<summary><b>3. MoviEdit: Long-form Multi-Shot Video Editing Made Easy</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02737) • [📄 arXiv](https://arxiv.org/abs/2611.02737) • [📥 PDF](https://arxiv.org/pdf/2611.02737)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/adobe-research/MoviEdit-Bench)

> Long-form multi-shot video editing requires an intended edit to remain coherent across abrupt changes in scene, viewpoint, and subject appearance, while processing substantially longer visual context than conventional single-shot editors. We intro...

</details>

<details>
<summary><b>4. Axion 1.0: Open Agent Foundation Models, Simplified</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jingchen Sun, Yijun Liang, Jiayu Zheng, Yunjie Tian, Yuqi Jia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.03233) • [📄 arXiv](https://arxiv.org/abs/2611.03233) • [📥 PDF](https://arxiv.org/pdf/2611.03233)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/bytedance/axion1_0)

> Introducing Axion 1.0: Open Agent Foundation Models, Simplified. Strong scientific reasoning and tool use with a simple post-training recipe: two lightweight supervised fine-tuning stages followed by reinforcement learning. Learn to reason: model-...

</details>

<details>
<summary><b>5. SpecPilot: Autonomous Algorithm Discovery and Systems Optimization for Speculative Decoding</b> ⭐ 29</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02501) • [📄 arXiv](https://arxiv.org/abs/2611.02501) • [📥 PDF](https://arxiv.org/pdf/2611.02501)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/StargazerX0/SpecPilot)

> SpecPilot is an agentic loop that discovers lossless speculative decoding optimizations for each deployment, raising the speedup over AR from 4.46× to 10.87×. Paper: https://arxiv.org/pdf/2611.02501 Code: https://github.com/StargazerX0/SpecPilot P...

</details>

<details>
<summary><b>6. ParetoSaddler: Pareto-Guided Optimization of Agent Harnesses for Quality and Cost</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02079) • [📄 arXiv](https://arxiv.org/abs/2611.02079) • [📥 PDF](https://arxiv.org/pdf/2611.02079)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/AutoSaddler/tree/feat/paretosaddler)

> The code is available at: https://github.com/microsoft/AutoSaddler/tree/feat/paretosaddler . We welcome your comments and feedback!

</details>

<details>
<summary><b>7. Recursa: Toward Recursively Self-Improving Foundation Model Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04035) • [📄 arXiv](https://arxiv.org/abs/2611.04035) • [📥 PDF](https://arxiv.org/pdf/2611.04035)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> When a foundation model moves into a new domain, there's usually little expert-written data or reliable supervision to train on. Recursa adapts to the domain by learning from its own experience, without human experts or a stronger teacher model. A...

</details>

<details>
<summary><b>8. PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaowen Chang, Tengfei Ma, Lihao Guo, Wentao Ma, Travis Davies

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04870) • [📄 arXiv](https://arxiv.org/abs/2611.04870) • [📥 PDF](https://arxiv.org/pdf/2611.04870)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> PrimeSim is an action-conditioned, multiview world model that runs as a robot simulator. It starts from a pretrained video generator, midtrains on 30,110 hours across 22 platforms, then posttrains into a causal few-step generator. Posttraining com...

</details>

<details>
<summary><b>9. Near-Zero Token Embeddings Silently Stall Training in Gated Sequence Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mingjian Zhao, Haoran Song, Yuxin Wang, Boyuan Sun, Xin Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04449) • [📄 arXiv](https://arxiv.org/abs/2611.04449) • [📥 PDF](https://arxiv.org/pdf/2611.04449)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Near-zero input tokens trigger gradient blowup in gated models, not mid-sequence Transformers.

</details>

<details>
<summary><b>10. Can AI Reviewers See Your Efforts to Improve Your Paper? A Large-Scale Study of Paper Revision Trajectories</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02244) • [📄 arXiv](https://arxiv.org/abs/2611.02244) • [📥 PDF](https://arxiv.org/pdf/2611.02244)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/c-steve-wang/PaperTrace)

> Do AI reviewers recognize the improvements authors make through revision? We introduce PaperTrajectory-2026, covering 6,959 paper trajectories across five AI conferences, with automatic annotations along 14 revision dimensions. Evaluating eight AI...

</details>

<details>
<summary><b>11. Mage-VL-Cowork: A Full Training Recipe to Learn Hybrid Computer Use From Evolved Harnesses</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40747) • [📄 arXiv](https://arxiv.org/abs/2610.40747) • [📥 PDF](https://arxiv.org/pdf/2610.40747)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Mage-VL-Cowork presents a complete recipe for training a hybrid computer-use model—from multimodal alignment for enhanced GUI perception and office-oriented coding to agentic capabilities and hybrid policy execution for completing everyday office ...

</details>

<details>
<summary><b>12. LOLPlay: Benchmarking Real-Time Visual Agents in League of Legends</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Zhaojian Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.03685) • [📄 arXiv](https://arxiv.org/abs/2611.03685) • [📥 PDF](https://arxiv.org/pdf/2611.03685)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/answers111/LOLPlay)

> Benchmarking visual agents in real-world LOL game.

</details>

<details>
<summary><b>13. Route Anything: A Study of Model and Policy Routing for General-Purpose AI Systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhenglin Wan, Weidong Tang, Hexin Wang, Yixing Ma, Pengfei Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.05275) • [📄 arXiv](https://arxiv.org/abs/2611.05275) • [📥 PDF](https://arxiv.org/pdf/2611.05275)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/InfRec-Cardinal/route-anything)

> A potential path to AGI could be a router: Route Anything brings routing to text, images, video, speech, code, and robots, delivering 8.82× average relative efficiency across four settings and exceeding the performance of the best fixed model in d...

</details>

<details>
<summary><b>14. "Shall We Approve?" Consequence-Aware Approval for Autonomous Coding Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40753) • [📄 arXiv](https://arxiv.org/abs/2610.40753) • [📥 PDF](https://arxiv.org/pdf/2610.40753)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/wudaoz/Caper)

> Does a coding agent really know what its next command will do? Approving every command by hand does not scale: Claude Code users approve 93% of permission prompts anyway. So agents now rely on automated approvers, such as Claude Code's Auto Mode, ...

</details>

<details>
<summary><b>15. Is Forgetting Unified? Exploring Concept Unlearning in Unified Multimodal Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04467) • [📄 arXiv](https://arxiv.org/abs/2611.04467) • [📥 PDF](https://arxiv.org/pdf/2611.04467)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Long-horizon embodied tasks require agents to learn, reason, and generalize robustly across complex real-world environments. However, existing approaches often struggle with partial observability, fragmented spatial reasoning, and inefficient inte...

</details>

<details>
<summary><b>16. Where to Learn and How Far to Roll Out: Efficient Adaptive Training for GUI Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yue Wang, Steven Hoi, Panrong Tong, Quyu Kong, Yuxiang Chai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39251) • [📄 arXiv](https://arxiv.org/abs/2610.39251) • [📥 PDF](https://arxiv.org/pdf/2610.39251)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Trajectory-level GRPO is a common approach for training multi-step GUI agents, but repeatedly rolling out complete trajectories from the task root can be inefficient: task-level rewards provide coarse supervision over which intermediate decisions ...

</details>

<details>
<summary><b>17. SeMo3D: Unifying 3D Generation, Editing, and Completion via Selective Modeling</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02568) • [📄 arXiv](https://arxiv.org/abs/2611.02568) • [📥 PDF](https://arxiv.org/pdf/2611.02568)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/XiaokunSun/SeMo3D)

> 3D generation, editing, and completion are typically treated as regenerating the complete 3D asset under task-specific constraints, even when only a local region needs modification. This leads to task isolation, inconsistency in content that shoul...

</details>

<details>
<summary><b>18. What We Keep of What We Forget: Replacement Operators and Routing Semantics in Training-Free MoE Expert Pruning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Faisal Shafait, Momina Moetesum, Sehrish Bibi, drezzaali

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.41366) • [📄 arXiv](https://arxiv.org/abs/2610.41366) • [📥 PDF](https://arxiv.org/pdf/2610.41366)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The paper provides a controlled fixed-pruning-mask comparison of replacement/rerouting strategies and shows that post-pruning gate scale is part of the pruning algorithm, proposing the trained-scale rule: preserving the router’s training-time norm...

</details>

<details>
<summary><b>19. Are Frontier VLM Agents Ready to Be Robot Generalists? An Empirical Study with the Embodied Agent Arena</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Yangkai Wei, Yehang Zhang, Zexi Li, Pukun Zhao, Haojian Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00854) • [📄 arXiv](https://arxiv.org/abs/2610.00854) • [📥 PDF](https://arxiv.org/pdf/2610.00854)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/embodied-agent-arena/embodied-agent-arena)

> Embodied Agent Arena evaluates ten frontier VLM agents on 1,000 cases across Geometry, Spatial Reasoning, Affordance, Task Planning, and Manipulation. It combines 32 established sources with GeoProbe, a new 168-case benchmark for camera, motion, d...

</details>

<details>
<summary><b>20. Enhancing Stock Market Information Retrieval with Retrieval-Augmented Generation</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Lenh Cong Phan Pham, Nhan Huu Tran, Nhu Huynh Thi Le, Thu Anh Ngoc Le, Huan Phong Thai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.03747) • [📄 arXiv](https://arxiv.org/abs/2611.03747) • [📥 PDF](https://arxiv.org/pdf/2611.03747)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/nhunguet/SearchEngine)

> Demo

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 20 |
| 📅 Today | [`2026-11-04.json`](data/daily/2026-11-04.json) | 20 |
| 📆 This Week | [`2026-W44.json`](data/weekly/2026-W44.json) | 48 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 87 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-04 | 20 | [View JSON](data/daily/2026-11-04.json) |
| 📄 2026-11-03 | 14 | [View JSON](data/daily/2026-11-03.json) |
| 📄 2026-11-02 | 14 | [View JSON](data/daily/2026-11-02.json) |
| 📄 2026-11-01 | 39 | [View JSON](data/daily/2026-11-01.json) |
| 📄 2026-10-31 | 39 | [View JSON](data/daily/2026-10-31.json) |
| 📄 2026-10-30 | 10 | [View JSON](data/daily/2026-10-30.json) |
| 📄 2026-10-29 | 5 | [View JSON](data/daily/2026-10-29.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W44 | 48 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 87 | [View JSON](data/monthly/2026-11.json) |
| 🗓️ 2026-10 | 1120 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |

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
