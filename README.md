<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7435+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">21</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">827</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7435+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 19, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Pack Light, CarryOn: Building Object-Oriented Context for Long-Horizon Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18784) • [📄 arXiv](https://arxiv.org/abs/2610.18784) • [📥 PDF](https://arxiv.org/pdf/2610.18784)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Long-horizon agents often keep revisiting how work was done, even when the results are already preserved in files and task state. We introduce CarryOn, an object-oriented context runtime that enables agents to continue from reusable task progress ...

</details>

<details>
<summary><b>2. Has GPT-6-Astra Achieved Embodied Intelligence?</b> ⭐ 76</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22471) • [📄 arXiv](https://arxiv.org/abs/2610.22471) • [📥 PDF](https://arxiv.org/pdf/2610.22471)

**💻 Code:** [⭐ Code](https://github.com/OpenSenseNova/SenseNova-RoboRSI) • [⭐ Code](https://github.com/huggingface)

> Has GPT-6-Astra Achieved Embodied Intelligence? Embodied competence — largely yes. Practical embodied intelligence — not yet.

</details>

<details>
<summary><b>3. BizBench: Unveiling Hidden Challenges in Deploying Long-Horizon Agents for Real-World Business</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20424) • [📄 arXiv](https://arxiv.org/abs/2610.20424) • [📥 PDF](https://arxiv.org/pdf/2610.20424)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ant-intl/BizAgentBench)

> AI agents are moving from answering questions to executing business workflows, where actions change business states, commit resources, and shape later decisions. We introduce BizAgentBench, a long-horizon benchmark grounded in workflows from a glo...

</details>

<details>
<summary><b>4. Into the Unknown: Benchmarking AI agents on open scientific research problems</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19159) • [📄 arXiv](https://arxiv.org/abs/2610.19159) • [📥 PDF](https://arxiv.org/pdf/2610.19159)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/prometheus-eval/into-the-unknown)

> Can AI agents carry out research on open scientific problems? We introduce Into the Unknown, a benchmark that provides an agent with a science paper along with a research direction to pursue and tasks it to write a follow-up paper. We work with sc...

</details>

<details>
<summary><b>5. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23479) • [📄 arXiv](https://arxiv.org/abs/2610.23479) • [📥 PDF](https://arxiv.org/pdf/2610.23479)

**💻 Code:** [⭐ Code](https://github.com/HexaFuture/HexaShell) • [⭐ Code](https://github.com/huggingface)

> A thin harness in which a coding agent drives a robot from RGB images and writes its own tools; frozen tools for the 42 RoboDojo tasks

</details>

<details>
<summary><b>6. Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23181) • [📄 arXiv](https://arxiv.org/abs/2610.23181) • [📥 PDF](https://arxiv.org/pdf/2610.23181)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We are excited to share TreeOPSD (Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning). Training long-horizon LLM agents with verifiable rewards faces two practical bottlenecks: terminal rewards alone make turn-level ...

</details>

<details>
<summary><b>7. OpenMobile-2: Building Versatile Mobile Agents with Scalable Environments and App-Native Tools</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14157) • [📄 arXiv](https://arxiv.org/abs/2610.14157) • [📥 PDF](https://arxiv.org/pdf/2610.14157)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenMobile2/OpenMobile2-Code) • [⭐ Code](https://github.com/OpenMobile2/MobileGym-plusplus)

> 🚀 Introducing OpenMobile-2: Building Versatile Mobile Agents with Scalable Environments and App-Native Tools We're excited to share OpenMobile-2 , our latest work on mobile agents, with environments, data, benchmarks, and models fully open-sourced...

</details>

<details>
<summary><b>8. DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23839) • [📄 arXiv](https://arxiv.org/abs/2610.23839) • [📥 PDF](https://arxiv.org/pdf/2610.23839)

**💻 Code:** [⭐ Code](https://github.com/Gen-Verse/DiscoveryRSI) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/Gen-Verse/DiscoveryRSI

</details>

<details>
<summary><b>9. DLite: History-free Speculative Decoding for Agentic AI</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23787) • [📄 arXiv](https://arxiv.org/abs/2610.23787) • [📥 PDF](https://arxiv.org/pdf/2610.23787)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SJTU-DAI-Sys/DLite)

> Speculative decoding has become a mainstream technique for accelerating large language model (LLM) inference by using a lightweight draft model to generate candidate tokens that are subsequently verified by the target model. However, agentic workl...

</details>

<details>
<summary><b>10. From Research Goals to Scientific Closure: Evaluating Agents Through Evidence-Supported Physics Research</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21767) • [📄 arXiv](https://arxiv.org/abs/2610.21767) • [📥 PDF](https://arxiv.org/pdf/2610.21767)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/uncertainty2019/PhysClosureBench)

> Can AI agents do more than produce plausible calculations—can they establish a well-supported scientific answer? PhysClosureBench leaves the investigation to the agent. Across 50 physics tasks, we evaluate both the scientific work accomplished and...

</details>

<details>
<summary><b>11. OS-Helix: Unifying GUI Interaction and Code Execution in Native Computer-Using Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22234) • [📄 arXiv](https://arxiv.org/abs/2610.22234) • [📥 PDF](https://arxiv.org/pdf/2610.22234)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OS-Copilot/OS-Helix)

> Hi everyone! We introduce OS-Helix , a computer-using agent trained to coordinate GUI interaction and code execution within a single policy. Desktop tasks often need both: the GUI exposes what users see, while code can work precisely with files an...

</details>

<details>
<summary><b>12. DuoDiscover: Scaling Autonomous Discovery Beyond the Evaluation Bottleneck</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17184) • [📄 arXiv](https://arxiv.org/abs/2610.17184) • [📥 PDF](https://arxiv.org/pdf/2610.17184)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Agents can write a code change in seconds, but real evals take hours (85 min for vLLM, up to 28 h for CPU design). DuoDiscover has each agent find a target to optimize and a fast eval for it, keeps those evals honest against the real one, and scal...

</details>

<details>
<summary><b>13. Game2Real-Bench: Do Toy Games Reveal and Transfer to Real-World Visual Reasoning?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xinzhe Han, Yujie Cheng, Yue Zhu, Juntong Feng, Qingan Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18834) • [📄 arXiv](https://arxiv.org/abs/2610.18834) • [📥 PDF](https://arxiv.org/pdf/2610.18834)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/QingAnLiu/Game2Real-Bench)

> Visual reasoning is an important capability for vision-language models (VLMs) to understand visual evidence, spatial structure, constraints, and state changes. Existing evaluations mainly rely on either real-world data or procedurally generated ta...

</details>

<details>
<summary><b>14. ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ngai Wong, Jianjin Li, Lie24, zunhai, Bohansss

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20503) • [📄 arXiv](https://arxiv.org/abs/2610.20503) • [📥 PDF](https://arxiv.org/pdf/2610.20503)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/StartLuxLabs/ReaDy)

> Speculative decoding accelerates autoregressive generation by drafting multiple tokens per step, yet feature-based drafters rely on target hidden features whose contextual emphasis may not align with the needs of subsequent draft positions. Hybrid...

</details>

<details>
<summary><b>15. DuET: Note-Frame Dual Encoding for Piano Performance-to-Score Transcription</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21467) • [📄 arXiv](https://arxiv.org/abs/2610.21467) • [📥 PDF](https://arxiv.org/pdf/2610.21467)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/wei-zeng98/DuET)

> 🎹 Excited to share DuET, our piano transcription model! DuET converts expressive piano performances into readable sheet music by jointly modeling note-level transcription and beat tracking. Try DuET and explore our resources: 🎹 Online Demo: https:...

</details>

<details>
<summary><b>16. Speculative Sampling in Interactive Video Generation</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20556) • [📄 arXiv](https://arxiv.org/abs/2610.20556) • [📥 PDF](https://arxiv.org/pdf/2610.20556)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Yubo-Shankui/SpeculativeSampling)

> Speculative Sampling is a serving method for autoregressive video generation. It preserves the strong modeling capability of large bidirectional attention windows while retaining the low-latency interaction of small bidirectional attention windows...

</details>

<details>
<summary><b>17. GeoState: Learning View-Decoupled Geometry States for 3D World Modeling</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22343) • [📄 arXiv](https://arxiv.org/abs/2610.22343) • [📥 PDF](https://arxiv.org/pdf/2610.22343)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Hongchang-Liu/GeoState)

> Existing geometry-based world models often rely on view-dependent features, entangling scene dynamics with camera-induced variations. GeoState addresses this by learning a view-decoupled scene state, allowing the world model to focus on scene dyna...

</details>

<details>
<summary><b>18. StopWatch: Goal Grounding for Object-goal Navigation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23841) • [📄 arXiv](https://arxiv.org/abs/2610.23841) • [📥 PDF](https://arxiv.org/pdf/2610.23841)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Why do navigation policies walk past a goal that is already in view? We introduce StopWatch, which explicitly grounds the navigation goal by predicting whether it is visible and how far away it is. We find that goal grounding is the dominant failu...

</details>

<details>
<summary><b>19. Persistent Planar Memory for Video World Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23795) • [📄 arXiv](https://arxiv.org/abs/2610.23795) • [📥 PDF](https://arxiv.org/pdf/2610.23795)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/robbyant-research/PPMem)

> No abstract available.

</details>

<details>
<summary><b>20. PRISM: PRIvileged Sibling-guided Multi-Agent Topology Design</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20628) • [📄 arXiv](https://arxiv.org/abs/2610.20628) • [📥 PDF](https://arxiv.org/pdf/2610.20628)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The novel topology design method for multi-agent system

</details>

<details>
<summary><b>21. Sparse Autoencoders Are a Shared Coordinate System for Symbolic Matching</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19636) • [📄 arXiv](https://arxiv.org/abs/2610.19636) • [📥 PDF](https://arxiv.org/pdf/2610.19636)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AI-Interpretability-Illinois/Symbolic-Merge)

> TL;DR: An LLM's sparse autoencoder works as a shared coordinate system for matching symbols across heterogeneous systems (formal provers, code, table schemas, knowledge graphs, ontologies), with no training and no aligned pairs. How SAE-IDF works ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2026-10-19.json`](data/daily/2026-10-19.json) | 21 |
| 📆 This Week | [`2026-W42.json`](data/weekly/2026-W42.json) | 21 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 827 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-19 | 21 | [View JSON](data/daily/2026-10-19.json) |
| 📄 2026-10-18 | 84 | [View JSON](data/daily/2026-10-18.json) |
| 📄 2026-10-17 | 84 | [View JSON](data/daily/2026-10-17.json) |
| 📄 2026-10-16 | 34 | [View JSON](data/daily/2026-10-16.json) |
| 📄 2026-10-15 | 28 | [View JSON](data/daily/2026-10-15.json) |
| 📄 2026-10-14 | 21 | [View JSON](data/daily/2026-10-14.json) |
| 📄 2026-10-13 | 12 | [View JSON](data/daily/2026-10-13.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W42 | 21 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 827 | [View JSON](data/monthly/2026-10.json) |
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
