<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5323+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📆 This Month</b><br/><font size="5">308</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5323+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** July 25, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. AREX: Towards a Recursively Self-Improving Agent for Deep Research</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21461) • [📄 arXiv](https://arxiv.org/abs/2607.21461) • [📥 PDF](https://arxiv.org/pdf/2607.21461)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VectorSpaceLab/arex-model)

> The model weights have been publicly released at: https://huggingface.co/collections/BAAI/arex The models have also been integrated into our online application: https://arex-research.com/

</details>

<details>
<summary><b>2. ReferTrack: Referring Then Tracking for Embodied Visual Tracking</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.20061) • [📄 arXiv](https://arxiv.org/abs/2607.20061) • [📥 PDF](https://arxiv.org/pdf/2607.20061)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MedlarTea/referTrack)

> No abstract available.

</details>

<details>
<summary><b>3. Visual Contrastive Self-Distillation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21556) • [📄 arXiv](https://arxiv.org/abs/2607.21556) • [📥 PDF](https://arxiv.org/pdf/2607.21556)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/joliang17/VCSD)

> 🚀 Visual Contrastive Self-Distillation (VCSD) Can on-policy self-distillation improve Vision-Language Models without privileged answers or auxiliary visual evidence? Yes. The Core Idea: VCSD compares the same EMA teacher under two matched visual c...

</details>

<details>
<summary><b>4. K12-KGraph: A Curriculum-Aligned Knowledge Graph for Benchmarking and Training Educational LLMs</b> ⭐ 49</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09635) • [📄 arXiv](https://arxiv.org/abs/2605.09635) • [📥 PDF](https://arxiv.org/pdf/2605.09635)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/haolpku/K12-Dataset)

> Hi everyone! We’re excited to share K12-KGraph, a curriculum-aligned knowledge graph for benchmarking and training educational LLMs. Instead of only testing whether a model can answer isolated questions, we study whether it understands how knowled...

</details>

<details>
<summary><b>5. Show, Don't Tell: Evaluating Spatial Cognition in Generative Pixels Rather Than LLM Text</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21072) • [📄 arXiv](https://arxiv.org/abs/2607.21072) • [📥 PDF](https://arxiv.org/pdf/2607.21072)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-OmniAI/ProVisE)

> Spatial intelligence is often easier to show than tell. We introduce ProVisE, a benchmark-agnostic evaluation framework that enables image-generation models to answer spatial tasks directly in pixels. ProVisE converts these visual responses into t...

</details>

<details>
<summary><b>6. NVIDIA-labs OO Agents: Native Python Object-Oriented Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.20709) • [📄 arXiv](https://arxiv.org/abs/2607.20709) • [📥 PDF](https://arxiv.org/pdf/2607.20709)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. Tencent WorkBuddy Bench: A Multi-Domain Coding-Agent Benchmark with Contamination-Resistant Task Construction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.20911) • [📄 arXiv](https://arxiv.org/abs/2607.20911) • [📥 PDF](https://arxiv.org/pdf/2607.20911)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>8. Color Pass-Through via Camera-Display Coupling</b> ⭐ 38</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.12746) • [📄 arXiv](https://arxiv.org/abs/2607.12746) • [📥 PDF](https://arxiv.org/pdf/2607.12746)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/lyricccco/color-pass-through)

> Color Pass-Through aims to make an image captured by a camera and reproduced on a display perceptually match the real scene as viewed through the device. Instead of calibrating the camera and display independently through a predefined intermediate...

</details>

<details>
<summary><b>9. SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21553) • [📄 arXiv](https://arxiv.org/abs/2607.21553) • [📥 PDF](https://arxiv.org/pdf/2607.21553)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NVlabs/Sana/)

> We introduce SANA-Video 2.0, a hybrid video diffusion transformer instantiated at 5B and 14B scales under a unified architecture. Designed to generate high-quality video up to 720p on a single GPU, SANA-Video 2.0 matches full-softmax video DiTs in...

</details>

<details>
<summary><b>10. LLMs Get Lost in Evolving User Intent</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.20734) • [📄 arXiv](https://arxiv.org/abs/2607.20734) • [📥 PDF](https://arxiv.org/pdf/2607.20734)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/evolving-intent)

> As LLMs become more capable, they are increasingly deployed as collaborative agents, taking on user-delegated tasks through iterative interaction. Yet genuine interaction is inherently dynamic: users rarely specify their intent upfront, instead di...

</details>

<details>
<summary><b>11. Self-Supervised Learning of Structured Dynamics from Videos</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21576) • [📄 arXiv](https://arxiv.org/abs/2607.21576) • [📥 PDF](https://arxiv.org/pdf/2607.21576)

**💻 Code:** [⭐ Code](https://github.com/lukasknobel/StructuredDynamics) • [⭐ Code](https://github.com/huggingface)

> Check out our project page: https://lukasknobel.github.io/projects/StructuredDynamics

</details>

<details>
<summary><b>12. Streaming Multi-Agent Autoregressive Diffusion Model with World State Registers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bolei Zhou, Krishna Kumar Singh, Ziyang Leng, Yuheng Li, Sicheng Mo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21594) • [📄 arXiv](https://arxiv.org/abs/2607.21594) • [📥 PDF](https://arxiv.org/pdf/2607.21594)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>13. Sample-Efficient Learning from Agent Experience</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hamid Rezatofighi, Jianfei Cai, Yunhao Fang, Haoqin Tu, gouc

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21051) • [📄 arXiv](https://arxiv.org/abs/2607.21051) • [📥 PDF](https://arxiv.org/pdf/2607.21051)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Real-world agent learning is often constrained by costly environment interactions, such as running time-consuming experiments or obtaining human feedback. In-context learning offers a highly sample-efficient way for agents to learn from their own ...

</details>

<details>
<summary><b>14. Robostral Navigate</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.20785) • [📄 arXiv](https://arxiv.org/abs/2607.20785) • [📥 PDF](https://arxiv.org/pdf/2607.20785)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. Multi-Turn On-Policy Distillation with Prefix Replay</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.04763) • [📄 arXiv](https://arxiv.org/abs/2607.04763) • [📥 PDF](https://arxiv.org/pdf/2607.04763)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/BaohaoLiao/ReOPD)

> On-policy distillation for agents without the environment: replay teacher prefixes instead of live rollouts.

</details>

<details>
<summary><b>16. Predictive Divergence Masks for LLM RL</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiaqi Tang, Bowen Ping, Penghui Qi, Jiarui Yao, Xiangxin Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.10848) • [📄 arXiv](https://arxiv.org/abs/2607.10848) • [📥 PDF](https://arxiv.org/pdf/2607.10848)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reinforcement learning for large language models (LLMs) typically relies on trust-region masks to stabilize off-policy updates. The dominant PPO-style approach uses the sampled-token importance ratio for two criteria: a proximity criterion, which ...

</details>

<details>
<summary><b>17. Recurrent Sinusoidal INRs for Efficient High-Fidelity Representation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kyong Hwan Jin, Jaejun Yoo, hyeoncho01

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21485) • [📄 arXiv](https://arxiv.org/abs/2607.21485) • [📥 PDF](https://arxiv.org/pdf/2607.21485)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hyeon-cho/Harmonic-Siren)

> Accepted to ECCV 2026 (Poster) We study sinusoidal recurrence as an iterative mechanism for harmonic spectral enrichment in implicit neural representations (INRs). Our analysis reveals that sinusoidal activations induce a harmonic line spectrum, p...

</details>

<details>
<summary><b>18. TableVerse: A Large-scale Tabletop Dataset with Real-world Grounded Layouts for Generalizable Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21017) • [📄 arXiv](https://arxiv.org/abs/2607.21017) • [📥 PDF](https://arxiv.org/pdf/2607.21017)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>19. OpenForgeRL: Train Harness-native Agents in Any Environment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21557) • [📄 arXiv](https://arxiv.org/abs/2607.21557) • [📥 PDF](https://arxiv.org/pdf/2607.21557)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Modern AI agents rely on elaborate inference harnesses such as Claude Code, Codex, and OpenClaw to drive multi-turn reasoning, tool use, and access to external systems. While powerful, these complex harnesses also make agents hard to train end-to-...

</details>

<details>
<summary><b>20. FinanceComplexQA: Benchmarking Agentic Reasoning on Industrial-grade Financial Documents</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.19238) • [📄 arXiv](https://arxiv.org/abs/2607.19238) • [📥 PDF](https://arxiv.org/pdf/2607.19238)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/buaacxf/FinanceComplexQA)

> Agentic Reasoning has become a transformative force in financial analysis due to its ability to integrate large-scale information and generate reliable and accurate content. However, when handling complex real-world problems, different agents stil...

</details>

<details>
<summary><b>21. GraphVid: Interactive Graph-Controllable Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21580) • [📄 arXiv](https://arxiv.org/abs/2607.21580) • [📥 PDF](https://arxiv.org/pdf/2607.21580)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> GraphVid: Interactive Graph-Controllable Video Generation (ECCV'26)

</details>

<details>
<summary><b>22. Dataset Distillation by Influence Matching</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.16859) • [📄 arXiv](https://arxiv.org/abs/2607.16859) • [📥 PDF](https://arxiv.org/pdf/2607.16859)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hrtan/infmatch)

> We revisit dataset distillation from an outcome-centric perspective. Rather than aligning process surrogates (per-step gradients or training trajectories), Influence Matching (Inf-Match) aligns the final outcome of training: it learns a compact sy...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-07-25.json`](data/daily/2026-07-25.json) | 22 |
| 📆 This Week | [`2026-W29.json`](data/weekly/2026-W29.json) | 48 |
| 🗓️ This Month | [`2026-07.json`](data/monthly/2026-07.json) | 308 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-07-25 | 22 | [View JSON](data/daily/2026-07-25.json) |
| 📄 2026-07-24 | 6 | [View JSON](data/daily/2026-07-24.json) |
| 📄 2026-07-23 | 5 | [View JSON](data/daily/2026-07-23.json) |
| 📄 2026-07-22 | 4 | [View JSON](data/daily/2026-07-22.json) |
| 📄 2026-07-21 | 2 | [View JSON](data/daily/2026-07-21.json) |
| 📄 2026-07-20 | 9 | [View JSON](data/daily/2026-07-20.json) |
| 📄 2026-07-19 | 29 | [View JSON](data/daily/2026-07-19.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W29 | 48 | [View JSON](data/weekly/2026-W29.json) |
| 📅 2026-W28 | 94 | [View JSON](data/weekly/2026-W28.json) |
| 📅 2026-W27 | 79 | [View JSON](data/weekly/2026-W27.json) |
| 📅 2026-W26 | 117 | [View JSON](data/weekly/2026-W26.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-07 | 308 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |

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
