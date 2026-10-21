<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7477+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">63</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">869</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7477+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 21, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Focused Omni-Modal GUI Agents</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26203) • [📄 arXiv](https://arxiv.org/abs/2610.26203) • [📥 PDF](https://arxiv.org/pdf/2610.26203)

**💻 Code:** [⭐ Code](https://github.com/fyzhang1/FocusGUI) • [⭐ Code](https://github.com/huggingface)

> Focused Omni-Modal GUI Agents

</details>

<details>
<summary><b>2. Sage-A1.5: From Long-Horizon Execution to Scientific Exploration with Capability Composition</b> ⭐ 571</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28578) • [📄 arXiv](https://arxiv.org/abs/2610.28578) • [📥 PDF](https://arxiv.org/pdf/2610.28578)

**💻 Code:** [⭐ Code](https://github.com/InternScience/Agents-A1) • [⭐ Code](https://github.com/huggingface)

> Sage-A1.5: From Long-Horizon Execution to Scientific Exploration with Capability Composition We’re open-sourcing the model weights today. Stay tuned!

</details>

<details>
<summary><b>3. SolarRenderer: Few-Step Generative Reference Rendering</b> ⭐ 153</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28630) • [📄 arXiv](https://arxiv.org/abs/2610.28630) • [📥 PDF](https://arxiv.org/pdf/2610.28630)

**💻 Code:** [⭐ Code](https://github.com/Enderfga/SolarRenderer) • [⭐ Code](https://github.com/huggingface)

> SolarRenderer turns any engine's untextured structure pass into a finished, styled video, guided by a reference first frame. The model is built on MiniMax-H3, with a control LoRA and a few-step distilled LoRA. 🎯 Precise structure. The output follo...

</details>

<details>
<summary><b>4. SWIFT: Self-guided Weak-to-strong dIFfusion models via Target extrapolation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25844) • [📄 arXiv](https://arxiv.org/abs/2610.25844) • [📥 PDF](https://arxiv.org/pdf/2610.25844)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce SWIFT , a simple and effective weak-to-strong self-guided training method. It uses a weaker model to improve a stronger diffusion model through target extrapolation , without adding inference overhead. On ImageNet-256, SWIFT surpasses...

</details>

<details>
<summary><b>5. PhysSkin++: Real-Time and Generalizable Physics-Based Animation via Self-Supervised Neural Skinning</b> ⭐ 34</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26067) • [📄 arXiv](https://arxiv.org/abs/2610.26067) • [📥 PDF](https://arxiv.org/pdf/2610.26067)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zju3dv/PhysSkin)

> No abstract available.

</details>

<details>
<summary><b>6. MMEnterprise-Bench: Evaluating Enterprise Agents over Evolving Multimodal Records</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Seoung Choi, Ji Soo Lee, Dohwan Ko, Jongha Kim, Seohyun Lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27911) • [📄 arXiv](https://arxiv.org/abs/2610.27911) • [📥 PDF](https://arxiv.org/pdf/2610.27911)

**💻 Code:** [⭐ Code](https://github.com/mlvlab/MMEnterprise-Bench) • [⭐ Code](https://github.com/huggingface)

> How reliable are enterprise agents when workplace records disagree or change over time? MMEnterprise-Bench evaluates this challenge with 1,613 questions across 30 simulated collaboration projects, grounded in 7,140 textual and visual records. It t...

</details>

<details>
<summary><b>7. Ropedia W0: Scaling Loco-Manipulation World Models across Embodiments with Human Experience</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26855) • [📄 arXiv](https://arxiv.org/abs/2610.26855) • [📥 PDF](https://arxiv.org/pdf/2610.26855)

**💻 Code:** [⭐ Code](https://github.com/Ropedia/ropedia-w0) • [⭐ Code](https://github.com/huggingface)

> Ropedia W0 is an egocentric world foundation model that learns loco-manipulation from 10K hours of human experience and transfers it across embodiments — from robot arms to humanoids, and from grippers to dexterous hands — through one shared kinem...

</details>

<details>
<summary><b>8. Correction Training for Diffusion Models</b> ⭐ 65</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14136) • [📄 arXiv](https://arxiv.org/abs/2610.14136) • [📥 PDF](https://arxiv.org/pdf/2610.14136)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sunovivid/correction-training)

> ✨Correction Training for Diffusion Models✨ Donghoon Ahn¹, Runqian Wang¹, Alexei A. Efros¹, Taesung Park², Yash Belhe² ¹ UC Berkeley  ² Reve Can diffusion models generate high-quality images without classifier-free guidance? Yes! We introduce corre...

</details>

<details>
<summary><b>9. From Short Clips to Real-Time Streams: Identity-Aware Supervision and Context-Efficient Generation for Video Face Swapping</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Xuehao Wang, Haoyu Huang, Ruiqi Ma, Yuwei Liang, CrazyK828

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26681) • [📄 arXiv](https://arxiv.org/abs/2610.26681) • [📥 PDF](https://arxiv.org/pdf/2610.26681)

**💻 Code:** [⭐ Code](https://github.com/aim-uofa/StreamingSwap) • [⭐ Code](https://github.com/huggingface)

> Hello~ We introduce StreamingSwap, a reference-guided face swapping method for real-time video streams, preserving target attributes and motion. project page: https://aim-uofa.github.io/StreamingSwap/ code: https://github.com/aim-uofa/StreamingSwa...

</details>

<details>
<summary><b>10. EdgeKernelBench: Unified LLM-Driven Kernel Generation for Edge Backends</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.24648) • [📄 arXiv](https://arxiv.org/abs/2610.24648) • [📥 PDF](https://arxiv.org/pdf/2610.24648)

**💻 Code:** [⭐ Code](https://github.com/pillor9/EdgeKernelBench) • [⭐ Code](https://github.com/huggingface)

> This paper introduces EdgeKernelBench, which asks: Can LLMs generate correct and efficient kernels across heterogeneous edge devices through a unified intermediate representation? We also introduce the Layered Edge Kernel Agent (LEKA), which organ...

</details>

<details>
<summary><b>11. VLA-Rolling: Fewer Denoising Steps, Better Control via Adaptive Action Reuse</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25043) • [📄 arXiv](https://arxiv.org/abs/2610.25043) • [📥 PDF](https://arxiv.org/pdf/2610.25043)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-OmniAI/VLA-Rolling)

> Why restart from noise when useful actions are already there? Current VLA paradigm typically execute only a short prefix before replanning with updated observations, then discard the unexecuted action tail. VLA-Rolling carries forward the unexecut...

</details>

<details>
<summary><b>12. S-Agent 2.0: Spatial Reasoning with a General Coding Agent</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28354) • [📄 arXiv](https://arxiv.org/abs/2610.28354) • [📥 PDF](https://arxiv.org/pdf/2610.28354)

**💻 Code:** [⭐ Code](https://github.com/Ropedia/S-Agent-2.0) • [⭐ Code](https://github.com/huggingface)

> S-Agent 2.0 is a spatial reasoning agent built on an off-the-shelf general coding agent and a command-line perception toolkit.

</details>

<details>
<summary><b>13. PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27628) • [📄 arXiv](https://arxiv.org/abs/2610.27628) • [📥 PDF](https://arxiv.org/pdf/2610.27628)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🚀 Introducing PAR, our new work on length extrapolation in looped language models. 📦 The task is simple: five boxes, one object in each. Follow a sequence of swaps, then answer what’s inside. We train the model on short sequences and test whether ...

</details>

<details>
<summary><b>14. HybridResearcher: Bridging the Structured-Unstructured Boundary for Deep Research Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.24899) • [📄 arXiv](https://arxiv.org/abs/2610.24899) • [📥 PDF](https://arxiv.org/pdf/2610.24899)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This work introduces HybridResearcher, a 4B agent trained to bridge evidence across the structured and unstructured boundary, and synthesizes cross-source bridge trajectories by starting with validated single-source tasks and fabricating the missi...

</details>

<details>
<summary><b>15. EMBER: Predicting TCR--pMHC Binding Without a Docked Complex</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27334) • [📄 arXiv](https://arxiv.org/abs/2610.27334) • [📥 PDF](https://arxiv.org/pdf/2610.27334)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A new PPI classification model proposed submitted in ICLR2027

</details>

<details>
<summary><b>16. Matrix-Game 4.0: Harnessing Video Models for Real-Time Playable Game Worlds</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Puyi Wang, Biao Jiang, Kaiqin Kong, Runjia Qian, Jihai Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27999) • [📄 arXiv](https://arxiv.org/abs/2610.27999) • [📥 PDF](https://arxiv.org/pdf/2610.27999)

**💻 Code:** [⭐ Code](https://github.com/SkyworkAI/Matrix-Game) • [⭐ Code](https://github.com/huggingface)

> Github: https://github.com/SkyworkAI/Matrix-Game

</details>

<details>
<summary><b>17. EvaOmni: An Evidence-Grounded Benchmark for Audio-Visual Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28444) • [📄 arXiv](https://arxiv.org/abs/2610.28444) • [📥 PDF](https://arxiv.org/pdf/2610.28444)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VisionXLab/EvaOmni)

> Hi everyone! 👋 I’m an author of EvaOmni: An Evidence-Grounded Benchmark for Audio-Visual Reasoning . We’re excited to share our work with the Hugging Face community! In this paper, we address the challenge of evaluating whether models’ answers are...

</details>

<details>
<summary><b>18. Large Flow Language Models</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Seunghoon Hong, Jaehoon Yoo, Udbhav Bamba, Chanhyuk Lee, Arnav Chavan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28559) • [📄 arXiv](https://arxiv.org/abs/2610.28559) • [📥 PDF](https://arxiv.org/pdf/2610.28559)

**💻 Code:** [⭐ Code](https://github.com/lafa-gh/large-flow-lms) • [⭐ Code](https://github.com/huggingface)

> Continuous flow language models offer a path to parallel generation with tools for control and acceleration unavailable to discrete diffusion, yet effective techniques for building them at the billion-parameter scale remain less established. We pr...

</details>

<details>
<summary><b>19. HumanSkill-4D: Training-Free Language Querying of Animatable Gaussian Avatars in Space and Time</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.24452) • [📄 arXiv](https://arxiv.org/abs/2610.24452) • [📥 PDF](https://arxiv.org/pdf/2610.24452)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce HumanSkill-4D , a training-free framework for selecting body parts, clothing, and matching motion intervals of animatable Gaussian avatars through natural language. For example: “Highlight the pants while both knees are deeply bent.” ...

</details>

<details>
<summary><b>20. TRACE: Tracking Multimodal Clinical Trajectories for Inpatient Risk Prediction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jong Hak Moon, Minjun Kim

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25765) • [📄 arXiv](https://arxiv.org/abs/2610.25765) • [📥 PDF](https://arxiv.org/pdf/2610.25765)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>21. WinARDet: Compressed Window Auto-Regression for Long-Horizon Vision-Language Grounding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19419) • [📄 arXiv](https://arxiv.org/abs/2610.19419) • [📥 PDF](https://arxiv.org/pdf/2610.19419)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/facias914/WinARDet)

> MLLM-based detectors formulate vision-language grounding as autoregressive coordinate generation, yet in dense scenes, they often collapse into a box repetition loop, repeatedly emitting identical boxes until the decoding budget is exhausted. We t...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2026-10-21.json`](data/daily/2026-10-21.json) | 21 |
| 📆 This Week | [`2026-W42.json`](data/weekly/2026-W42.json) | 63 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 869 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-21 | 21 | [View JSON](data/daily/2026-10-21.json) |
| 📄 2026-10-20 | 21 | [View JSON](data/daily/2026-10-20.json) |
| 📄 2026-10-19 | 21 | [View JSON](data/daily/2026-10-19.json) |
| 📄 2026-10-18 | 84 | [View JSON](data/daily/2026-10-18.json) |
| 📄 2026-10-17 | 84 | [View JSON](data/daily/2026-10-17.json) |
| 📄 2026-10-16 | 34 | [View JSON](data/daily/2026-10-16.json) |
| 📄 2026-10-15 | 28 | [View JSON](data/daily/2026-10-15.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W42 | 63 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 869 | [View JSON](data/monthly/2026-10.json) |
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
