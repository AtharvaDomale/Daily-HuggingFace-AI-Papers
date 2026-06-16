<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-17-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4764+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">17</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">17</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">361</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4764+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 16, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. DreamX-World 1.0: A General-Purpose Interactive World Model</b> ⭐ 264</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16993) • [📄 arXiv](https://arxiv.org/abs/2606.16993) • [📥 PDF](https://arxiv.org/pdf/2606.16993)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AMAP-ML/DreamX-World)

> Project Page: https://amap-ml.github.io/DreamX_World Code: https://github.com/AMAP-ML/DreamX-World

</details>

<details>
<summary><b>2. Data Journalist Agent: Transforming Data into Verifiable Multimodal Stories</b> ⭐ 34</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11176) • [📄 arXiv](https://arxiv.org/abs/2606.11176) • [📥 PDF](https://arxiv.org/pdf/2606.11176)

**💻 Code:** [⭐ Code](https://github.com/QinghongLin/data2story-skill) • [⭐ Code](https://github.com/huggingface)

> Can an AI agent act like an editor, turning raw data into readable, verifiable, multimodal stories? Data2Story brings data analysis, information retrieval, narrative writing, visual design, and fact-checking into one “virtual newsroom.” Every key ...

</details>

<details>
<summary><b>3. FastContext: Training Efficient Repository Explorer for Coding Agents</b> ⭐ 152</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.14066) • [📄 arXiv](https://arxiv.org/abs/2606.14066) • [📥 PDF](https://arxiv.org/pdf/2606.14066)

**💻 Code:** [⭐ Code](https://github.com/YerbaPage/Awesome-Agent-Context-Compression) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/fastcontext)

> Can an explore subagent rise the performance of main agent?

</details>

<details>
<summary><b>4. Who Should Lead Decoding Now? Tracking Reliable Trajectories for Ensembling Masked Diffusion Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16281) • [📄 arXiv](https://arxiv.org/abs/2606.16281) • [📥 PDF](https://arxiv.org/pdf/2606.16281)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We observe that successful MDLM generations exhibit stable confidence dynamics over answer-relevant positions, and that unreliable trajectories can often be corrected using promising intermediate states from other models. Building on this observat...

</details>

<details>
<summary><b>5. BRDFusion: Physics Meets Generation for Urban Scene Inverse Rendering</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Chih-Hao Lin, Yu-Lun Liu, Zheng-Hui Huang, Jie-Ying Lee, Yi-Ruei Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17049) • [📄 arXiv](https://arxiv.org/abs/2606.17049) • [📥 PDF](https://arxiv.org/pdf/2606.17049)

**💻 Code:** [⭐ Code](https://github.com/shigon255/BRDFusion) • [⭐ Code](https://github.com/huggingface)

> Inverse rendering of urban scenes from captured videos enables numerous applications, including content creation and autonomous driving simulation. Physically-based rendering methods follow and control lighting physics, but suffer from reconstruct...

</details>

<details>
<summary><b>6. BadWorld: Adversarial Attacks on World Models</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16519) • [📄 arXiv](https://arxiv.org/abs/2606.16519) • [📥 PDF](https://arxiv.org/pdf/2606.16519)

**💻 Code:** [⭐ Code](https://github.com/LinghuiiShen/BadWorld) • [⭐ Code](https://github.com/huggingface)

> 🚨 We introduce BadWorld , a label-free adversarial attack for visual world models. Starting from a single perturbed context image 🖼️, BadWorld can break down model's future predictions, even under user controls it has never seen before 🎮. This exp...

</details>

<details>
<summary><b>7. CODA-BENCH: Can Code Agents Handle Data-Intensive Tasks?</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.15300) • [📄 arXiv](https://arxiv.org/abs/2606.15300) • [📥 PDF](https://arxiv.org/pdf/2606.15300)

**💻 Code:** [⭐ Code](https://github.com/ruc-datalab/CoDA-Bench) • [⭐ Code](https://github.com/huggingface)

> How do LLM agents perform when placed in a data-intensive environment with 1,000+ files? CoDA-Bench (Code and Data-intensive Benchmark) is a benchmark for evaluating LLM agents on data-intensive analytical tasks. Highlights of CoDA-Bench: 🔍 1000 t...

</details>

<details>
<summary><b>8. OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu Zhao, Zhiluohan Guo, Kun Wang, Sunhao Dai, Jiakai Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16838) • [📄 arXiv](https://arxiv.org/abs/2606.16838) • [📥 PDF](https://arxiv.org/pdf/2606.16838)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🎉 Excited to share that our paper "OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation" has been accepted at KDD 2026! See you in Jeju Island 🇰🇷 🔍 The problem: Traditional multi-task recommenders follow an encode...

</details>

<details>
<summary><b>9. VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models</b> ⭐ 577</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16140) • [📄 arXiv](https://arxiv.org/abs/2606.16140) • [📥 PDF](https://arxiv.org/pdf/2606.16140)

**💻 Code:** [⭐ Code](https://github.com/WeiboAI/VibeThinker) • [⭐ Code](https://github.com/huggingface)

> We introduce VibeThinker-3B , a 3B dense reasoning model that explores how far verifiable reasoning can be pushed within a strictly small-model regime. The core idea behind this work is that capabilities such as mathematics, coding, and STEM reaso...

</details>

<details>
<summary><b>10. JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence</b> ⭐ 100</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.14777) • [📄 arXiv](https://arxiv.org/abs/2606.14777) • [📥 PDF](https://arxiv.org/pdf/2606.14777)

**💻 Code:** [⭐ Code](https://github.com/jd-opensource/JoyAI-VL-Interaction) • [⭐ Code](https://github.com/huggingface)

> Figure 1: Overview of JoyAI-VL-Interaction. Figure 3: Overview of the JoyAI-VL-Interaction System.

</details>

<details>
<summary><b>11. TokenPilot: Cache-Efficient Context Management for LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17016) • [📄 arXiv](https://arxiv.org/abs/2606.17016) • [📥 PDF](https://arxiv.org/pdf/2606.17016)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TokenPilot cuts the cost of long-horizon LLM agents by making context shorter without breaking the prompt cache.

</details>

<details>
<summary><b>12. VisualClaw: A Real-Time, Personalized Agent for the Physical World</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Juncheng Wu, Siwei Han, Zijun Wang, Jianwen Chen, Haoqin Tu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16295) • [📄 arXiv](https://arxiv.org/abs/2606.16295) • [📥 PDF](https://arxiv.org/pdf/2606.16295)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>13. UniDDT: Unifying Multimodal Understanding and Generation with Decoupled Diffusion Transformer</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16255) • [📄 arXiv](https://arxiv.org/abs/2606.16255) • [📥 PDF](https://arxiv.org/pdf/2606.16255)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Unified Multimodal Models (UMMs) have emerged as a critical direction for general-purpose multimodal intelligence, integrating understanding and generation into a single framework. However, existing UMMs face prominent challenges: (1) the inherent...

</details>

<details>
<summary><b>14. Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.15007) • [📄 arXiv](https://arxiv.org/abs/2606.15007) • [📥 PDF](https://arxiv.org/pdf/2606.15007)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NVIDIA-NeMo/Nemotron)

> Github: https://github.com/NVIDIA-NeMo/Nemotron

</details>

<details>
<summary><b>15. Qwen-RobotWorld Technical Report: Unifying Embodied World Modeling through Language-Conditioned Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17030) • [📄 arXiv](https://arxiv.org/abs/2606.17030) • [📥 PDF](https://arxiv.org/pdf/2606.17030)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>16. TuneJury: An Open Metric for Improving Music Generation Preference Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17006) • [📄 arXiv](https://arxiv.org/abs/2606.17006) • [📥 PDF](https://arxiv.org/pdf/2606.17006)

**💻 Code:** [⭐ Code](https://github.com/yonghyunk1m/TuneJury) • [⭐ Code](https://github.com/huggingface)

> TuneJury is an open reward model for music generation preference alignment. A lightweight head sits on top of frozen music encoders and maps an audio clip and an optional text prompt to a single preference score. We train it on human pairwise judg...

</details>

<details>
<summary><b>17. Retrieve, Don't Retrain: Extending Vision Language Action Models to New Tasks at Test Time</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.15631) • [📄 arXiv](https://arxiv.org/abs/2606.15631) • [📥 PDF](https://arxiv.org/pdf/2606.15631)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: Adding a new task to a VLA policy usually means collecting teleop demos and fine-tuning per task. We replace that target-side cost with retrieval: train the policy once, freeze it, and add new tasks at deployment just by appending cheap hum...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 17 |
| 📅 Today | [`2026-06-16.json`](data/daily/2026-06-16.json) | 17 |
| 📆 This Week | [`2026-W24.json`](data/weekly/2026-W24.json) | 17 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 361 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-16 | 17 | [View JSON](data/daily/2026-06-16.json) |
| 📄 2026-06-15 | 0 | [View JSON](data/daily/2026-06-15.json) |
| 📄 2026-06-14 | 44 | [View JSON](data/daily/2026-06-14.json) |
| 📄 2026-06-13 | 44 | [View JSON](data/daily/2026-06-13.json) |
| 📄 2026-06-12 | 22 | [View JSON](data/daily/2026-06-12.json) |
| 📄 2026-06-11 | 15 | [View JSON](data/daily/2026-06-11.json) |
| 📄 2026-06-10 | 18 | [View JSON](data/daily/2026-06-10.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W24 | 17 | [View JSON](data/weekly/2026-W24.json) |
| 📅 2026-W23 | 166 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 361 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |

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
