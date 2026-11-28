<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8342+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">151</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">614</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8342+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 28, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34051) • [📄 arXiv](https://arxiv.org/abs/2611.34051) • [📥 PDF](https://arxiv.org/pdf/2611.34051)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Linear attention offers an efficient solution for long-sequence modeling via linear complexity and fixed-size recurrent states. Gated DeltaNet-2 (GDN-2) introduces independent channel-wise control over erasure and writing, enhancing state-update e...

</details>

<details>
<summary><b>2. OS-Omni: Benchmarking Generalist Computer-Using Agents on Long-Horizon Workflows Across Platforms</b> ⭐ 106</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34232) • [📄 arXiv](https://arxiv.org/abs/2611.34232) • [📥 PDF](https://arxiv.org/pdf/2611.34232)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OS-Copilot/OS-Omni)

> NeurIPS 2026

</details>

<details>
<summary><b>3. Solve Vision with Code: Coding Agents Are Stronger Visual Reasoners Than Video Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34813) • [📄 arXiv](https://arxiv.org/abs/2611.34813) • [📥 PDF](https://arxiv.org/pdf/2611.34813)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Same first frame, same prompt, same evaluator: instead of generating pixels, a coding agent writes a program that renders the answer video. On VBVR-Pro-Bench, Codex, Claude Code and Gemini CLI score 0.92 / 0.89 / 0.89, more than 0.22 above the str...

</details>

<details>
<summary><b>4. What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models</b> ⭐ 10</summary>

<br/>

**👥 Authors:** therealram18, Skider

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31619) • [📄 arXiv](https://arxiv.org/abs/2611.31619) • [📥 PDF](https://arxiv.org/pdf/2611.31619)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/therealram18/vernier)

> A foundation model for time series is pre-trained once and then asked to forecast series it has never seen, so the evidence for it is its score on held-out datasets. Such datasets are scarce, and a few public leaderboards carry most comparisons, y...

</details>

<details>
<summary><b>5. ProxyGenBench: Benchmarking Proxy-Guided Video Generation</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34457) • [📄 arXiv](https://arxiv.org/abs/2611.34457) • [📥 PDF](https://arxiv.org/pdf/2611.34457)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/2kxx/ProxyGenBench)

> Recent video generators produce realistic videos, yet controlling how scenes unfold remains challenging. Structural proxies, such as animated 3D blockouts, guide generation by specifying scene layout, motion, camera behavior, and event order. Howe...

</details>

<details>
<summary><b>6. StateWeaver: Multi-Agent Coordination of Story-State Transitions for Long-Horizon Video Generation</b> ⭐ 49</summary>

<br/>

**👥 Authors:** Yanwen Ma, Haoran Li, Yijing Guo, Xuying Zhang, Modi Jin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34290) • [📄 arXiv](https://arxiv.org/abs/2611.34290) • [📥 PDF](https://arxiv.org/pdf/2611.34290)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ghost233lism/StateWeaver)

> Modern video foundation models generate high-quality short clips, but preserving long-video consistency and correctly depicting event-driven state changes remain challenging. Existing agentic workflows face these challenges because continual plann...

</details>

<details>
<summary><b>7. ShowNow: Fine-Grained Speculative Streaming for Responsive World Models</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34693) • [📄 arXiv](https://arxiv.org/abs/2611.34693) • [📥 PDF](https://arxiv.org/pdf/2611.34693)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/showlab/ShowNow)

> Real-Time on a Single H100: Distilled to 3 denoising steps (AnyFlow + Self-Forcing) and paired with FlashAttention-3, CUDA graphs, and fused commit kernels, the optimized path runs all 8 candidates in ~228 ms/step—fitting inside the 250 ms budget ...

</details>

<details>
<summary><b>8. DRBench Pro: A Knowledge-Centric Benchmark for Professional Deep Research Agents</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34852) • [📄 arXiv](https://arxiv.org/abs/2611.34852) • [📥 PDF](https://arxiv.org/pdf/2611.34852)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/DRBenchPro) • [⭐ Code](https://github.com/huggingface)

> How much of the relevant knowledge does a deep research agent actually cover? We built DRBench Pro to answer this question. For each case, we derive a research query and evidence-grounded rubrics from the same Wikipedia-derived knowledge tree. Thi...

</details>

<details>
<summary><b>9. Boundary-Aware Policy Learning for Agent-Orchestrated Robot Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32238) • [📄 arXiv](https://arxiv.org/abs/2611.32238) • [📥 PDF](https://arxiv.org/pdf/2611.32238)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: When a multimodal LLM agent calls a learned visuomotor policy (e.g., a VLA), the system often fails at the handoff : the agent invokes the policy from states where the policy is not reliable. We call the region where the policy is reliable ...

</details>

<details>
<summary><b>10. Revisiting LLMs as Context Engineers</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30494) • [📄 arXiv](https://arxiv.org/abs/2611.30494) • [📥 PDF](https://arxiv.org/pdf/2611.30494)

**💻 Code:** [⭐ Code](https://github.com/costa-nus/LCEBench) • [⭐ Code](https://github.com/huggingface)

> LLMs can engineer contexts via task-specific training, but no single method can achieve the best performance in all datasets at test time. Scaling helps but cannot reverse existing failures of LCEs. Even scaling to newer generations and more param...

</details>

<details>
<summary><b>11. When an AI agent makes a mistake in a database, what decides the damage?</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30483) • [📄 arXiv](https://arxiv.org/abs/2611.30483) • [📥 PDF](https://arxiv.org/pdf/2611.30483)

**💻 Code:** [⭐ Code](https://github.com/libredb/agent-blast-radius) • [⭐ Code](https://github.com/huggingface)

> We gave a minimal tool-calling agent database tasks with hazards in them (free up disk space, clean up tables, an instruction injected into the data, a superuser password left in an application table) and ran six small open-weight models against n...

</details>

<details>
<summary><b>12. LayoutRecall: Preserving Appearance in Multi-Round 3D Layout-Controlled Image Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33885) • [📄 arXiv](https://arxiv.org/abs/2611.33885) • [📥 PDF](https://arxiv.org/pdf/2611.33885)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project Page: https://ykchizs.github.io/layoutrecall.github.io/

</details>

<details>
<summary><b>13. Guarding with Judgment: Learning to Intervene for Multimodal Safety</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28849) • [📄 arXiv](https://arxiv.org/abs/2611.28849) • [📥 PDF](https://arxiv.org/pdf/2611.28849)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> New Guard！

</details>

<details>
<summary><b>14. Generation as Reverse Encoding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pietro Lio, Yipeng Qin, Hanzhong Guo, Sibei Yang, Cheng Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34782) • [📄 arXiv](https://arxiv.org/abs/2611.34782) • [📥 PDF](https://arxiv.org/pdf/2611.34782)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. Reading, Not Adapting: Unlocking Frozen Visual Knowledge for AI-Generated Image Detection</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34719) • [📄 arXiv](https://arxiv.org/abs/2611.34719) • [📥 PDF](https://arxiv.org/pdf/2611.34719)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yunncheng/ReMoT)

> arXiv: https://arxiv.org/abs/2611.34719 code: https://github.com/yunncheng/ReMoT

</details>

<details>
<summary><b>16. Damped Atomic Wave Integration Neurons (DAWIN): A Universal Primitive for Continuous-Time Spatiotemporal Learning</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Justin Kim, yumakemore

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30025) • [📄 arXiv](https://arxiv.org/abs/2611.30025) • [📥 PDF](https://arxiv.org/pdf/2611.30025)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yumakemore/DAWIN)

> 👋 Hi everyone! Author here sharing Damped Atomic Wave Integration Neurons (DAWIN) — a solver-free, continuous-time neural primitive designed to replace standard perceptrons across MLPs, Transformers, and GNNs when modeling irregularly sampled spat...

</details>

<details>
<summary><b>17. BiSAT: Bidirectionally Aligned Semantic-Acoustic Tokenization for Speech Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30723) • [📄 arXiv](https://arxiv.org/abs/2611.30723) • [📥 PDF](https://arxiv.org/pdf/2611.30723)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi everyone! We are excited to share BiSAT: Bidirectionally Aligned Semantic-Acoustic Tokenization for Speech Language Models ( arXiv:2611.30723 ). 🎧 Listen to Live Audio Demos: https://seekmind-lab.github.io/bisat-demo/ (zero-shot TTS & speech re...

</details>

<details>
<summary><b>18. Pretrained Tabular Features and the Measurement of Financial Proximity</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31491) • [📄 arXiv](https://arxiv.org/abs/2611.31491) • [📥 PDF](https://arxiv.org/pdf/2611.31491)

**💻 Code:** [⭐ Code](https://github.com/marcusinthesky/free-foundations) • [⭐ Code](https://github.com/huggingface)

> Much of the excitement around foundation models in finance concerns their predictive capabilities. But what can their internal representations tell us about the economy itself? In our new paper, Pretrained Tabular Features and the Measurement of F...

</details>

<details>
<summary><b>19. Decision-Native Multimodal Clinical Modeling via Anatomical Grounding and Confidence Calibration</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32993) • [📄 arXiv](https://arxiv.org/abs/2611.32993) • [📥 PDF](https://arxiv.org/pdf/2611.32993)

**💻 Code:** [⭐ Code](https://github.com/juntaoJianggavin/Radev) • [⭐ Code](https://github.com/huggingface)

> Radev addresses uncalibrated autoregressive decoding in 3D medical VLMs through a structured, decision-native approach. A few notable aspects: 1️⃣ Decision-native design: Estimating candidate probabilities in a single forward pass improves both in...

</details>

<details>
<summary><b>20. Markdown Is Not a Firewall: four coding agents, one rules file, four boundaries</b> ⭐ 0</summary>

<br/>

**👥 Authors:** sandeepdhuriauthor

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27112) • [📄 arXiv](https://arxiv.org/abs/2611.27112) • [📥 PDF](https://arxiv.org/pdf/2611.27112)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Four coding agents, one 278-word rules file. One agent's hidden-test pass rate fell from 99.0 to 67.7 percent. A network gate blocked all 5 attempts to cross a boundary. Pre-registered, open data and transcripts.

</details>

<details>
<summary><b>21. InitFree-SfM: When Initialization-Free Bundle Adjustment meets Learned Structure-from-Motion</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31516) • [📄 arXiv](https://arxiv.org/abs/2611.31516) • [📥 PDF](https://arxiv.org/pdf/2611.31516)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/simonwebertum/initfree-sfm)

> Initialization-free bundle adjustment (InitFree BA) aims to jointly recover camera poses and scene structure without requiring an initial global estimate. It is a very challenging problem, especially compared to traditional BA where a good estimat...

</details>

<details>
<summary><b>22. So Many Ways to Multiply 3times3 Matrices: A Large Dataset of Rank-23 Schemes</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Andrew Igorevich Perminov

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31155) • [📄 arXiv](https://arxiv.org/abs/2611.31155) • [📥 PDF](https://arxiv.org/pdf/2611.31155)

**💻 Code:** [⭐ Code](https://github.com/dronperminov/rank23_3x3_dataset) • [⭐ Code](https://github.com/huggingface)

> Dataset of rank-23 schemes for 3x3 matrix multiplication with pairwise distinct rank patterns. The dataset currently contains 62029 schemes, one for every rank pattern found, so no two schemes of the dataset are equivalent. The coefficients are te...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-11-28.json`](data/daily/2026-11-28.json) | 22 |
| 📆 This Week | [`2026-W47.json`](data/weekly/2026-W47.json) | 151 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 614 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-28 | 22 | [View JSON](data/daily/2026-11-28.json) |
| 📄 2026-11-27 | 13 | [View JSON](data/daily/2026-11-27.json) |
| 📄 2026-11-26 | 30 | [View JSON](data/daily/2026-11-26.json) |
| 📄 2026-11-25 | 34 | [View JSON](data/daily/2026-11-25.json) |
| 📄 2026-11-24 | 40 | [View JSON](data/daily/2026-11-24.json) |
| 📄 2026-11-23 | 12 | [View JSON](data/daily/2026-11-23.json) |
| 📄 2026-11-22 | 50 | [View JSON](data/daily/2026-11-22.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W47 | 151 | [View JSON](data/weekly/2026-W47.json) |
| 📅 2026-W46 | 184 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 614 | [View JSON](data/monthly/2026-11.json) |
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
