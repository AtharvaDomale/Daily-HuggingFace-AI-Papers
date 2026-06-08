<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-15-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4596+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">15</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">15</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">193</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4596+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 08, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Weinan Zhang, Mingyang Song, Fukuan Hou, Haoyu Sun, Yummytanmo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05761) • [📄 arXiv](https://arxiv.org/abs/2606.05761) • [📥 PDF](https://arxiv.org/pdf/2606.05761)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Yummytanmo/SubtleMemory)

> Persistent AI assistants, such as OpenClaw, accumulate large collections of related memories over long-term interactions. As these memories grow, they may reinforce one another, diverge across contexts, or directly conflict, making correct assista...

</details>

<details>
<summary><b>2. When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05806) • [📄 arXiv](https://arxiv.org/abs/2606.05806) • [📥 PDF](https://arxiv.org/pdf/2606.05806)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Zhudongsheng75/ToolMaze)

> When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents

</details>

<details>
<summary><b>3. UniSHARP: Universal Sharp Monocular View Synthesis</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Bo Du, Ruiyang Zhang, Hao Ren, Dizhe Zhang, Meixi Song

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.07514) • [📄 arXiv](https://arxiv.org/abs/2606.07514) • [📥 PDF](https://arxiv.org/pdf/2606.07514)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Insta360-Research-Team/UniSHARP)

> Given a single image from a perspective, wide-FoV, fisheye, or panoramic camera, UniSHARP predicts a 3D Gaussian point cloud and renders high-quality novel views.

</details>

<details>
<summary><b>4. LIMMT: Less is More for Motion Tracking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06953) • [📄 arXiv](https://arxiv.org/abs/2606.06953) • [📥 PDF](https://arxiv.org/pdf/2606.06953)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Accepted at ICML 2026.

</details>

<details>
<summary><b>5. Watch, Remember, Reason: Human-View Video Understanding with MLLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.07433) • [📄 arXiv](https://arxiv.org/abs/2606.07433) • [📥 PDF](https://arxiv.org/pdf/2606.07433)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/marinero4972/Awesome-HumanView-VideoUnderstanding)

> Watch, Remember, Reason: Human-View Video Understanding with MLLMs

</details>

<details>
<summary><b>6. Thinking with Imagination: Agentic Visual Spatial Reasoning with World Simulators</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06476) • [📄 arXiv](https://arxiv.org/abs/2606.06476) • [📥 PDF](https://arxiv.org/pdf/2606.06476)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZCMax/Thinking-With-Imagination)

> Spatial reasoning from multi-view images often requires evidence that is not visible in the given observations. Astra studies this problem as thinking with imagination: a VLM actively queries a world simulator for missing viewpoints and uses the r...

</details>

<details>
<summary><b>7. OpenSkill: Open-World Self-Evolution for LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuxuan Zhang, Wei Liang, Hanrong Zhang, Dingjie Song, Zhiling Yan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06741) • [📄 arXiv](https://arxiv.org/abs/2606.06741) • [📥 PDF](https://arxiv.org/pdf/2606.06741)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenLAIR/OpenSkill)

> OpenSkill is a framework enabling LLM agents to self-evolve by synthesizing grounded skills and verification signals from open-world resources without external target-task supervision.

</details>

<details>
<summary><b>8. When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26046) • [📄 arXiv](https://arxiv.org/abs/2603.00451) • [📥 PDF](https://arxiv.org/pdf/2605.26046)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/adivekar-utexas/when-gradients-collide)

> Title: When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges Authors: Parth Darshan (IIT Jodhpur), Abhishek Divekar (Amazon) Blogpost: https://textgrad-failure-modes.github.io Codebase: https://github.com/adiv...

</details>

<details>
<summary><b>9. Stream3D-VLM: Online 3D Spatial Understanding with Incremental Geometry Priors</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06891) • [📄 arXiv](https://arxiv.org/abs/2606.06891) • [📥 PDF](https://arxiv.org/pdf/2606.06891)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hanxunyu/Stream3D-VLM)

> Stream3D-VLM is an online 3D vision-language model that supports real-time spatial understanding and interaction directly from streaming video. By incrementally integrating geometry priors and employing geometry-adaptive voxel compression, our app...

</details>

<details>
<summary><b>10. Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Cong Li, Heng Cui, Yuxuan Liu, Zhongxin Chen, Songhao Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.07502) • [📄 arXiv](https://arxiv.org/abs/2606.07502) • [📥 PDF](https://arxiv.org/pdf/2606.07502)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/CentreChen/EmbFilter)

> We show that the unemebdding matrix within LLMs serve as an overlooked feature extractor for free.  It encodes a latent semantic space; filtering out its effects from the primary text embeddings markedly improves zero-shot representation performan...

</details>

<details>
<summary><b>11. PaperFlow: Profiling, Recommending, and Adapting Across Daily Paper Streams</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.07454) • [📄 arXiv](https://arxiv.org/abs/2606.07454) • [📥 PDF](https://arxiv.org/pdf/2606.07454)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenRaiser/PaperFlow)

> Scientific paper recommendation is typically evaluated as static ranking over a fixed candidate set, yet real scientific reading unfolds as a daily, longitudinal process in which interests shift and feedback accumulates. We introduce PaperFlow, a ...

</details>

<details>
<summary><b>12. Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bing Zhao, Wei Wang, Shaobo Wang, Zhengbo Jiao, Chuan Xiao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.07412) • [📄 arXiv](https://arxiv.org/abs/2606.07412) • [📥 PDF](https://arxiv.org/pdf/2606.07412)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Socratic-SWE is a self-evolution framework for coding agents that distills historical solving traces into structured skills, enabling the generation of targeted tasks to improve software engineering performance.

</details>

<details>
<summary><b>13. dots.tts Technical Report</b> ⭐ 210</summary>

<br/>

**👥 Authors:** Da Zheng, Hankun Wang, Bohan Li, Changtao Li, Shi Lian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.07080) • [📄 arXiv](https://arxiv.org/abs/2606.07080) • [📥 PDF](https://arxiv.org/pdf/2606.07080)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/rednote-hilab/dots.tts)

> dots.tts is a 2B-parameter continuous autoregressive text-to-speech foundation model utilizing Audio-VAE, flow-matching with full-history conditioning, and reward-free self-corrective post-training to achieve state-of-the-art speech generation per...

</details>

<details>
<summary><b>14. WorldBench: A Challenging and Visually Diverse Multimodal Reasoning Benchmark</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenhao Chai, Boya Zeng, Chung Peng Lee, Harish Krishnakumar, Yida Yin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06538) • [📄 arXiv](https://arxiv.org/abs/2606.06538) • [📥 PDF](https://arxiv.org/pdf/2606.06538)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> WorldBench is a challenging and visually diverse benchmark designed to evaluate Multimodal Large Language Models, addressing gaps in visual concept representation found in existing multimodal evaluation frameworks.

</details>

<details>
<summary><b>15. Parametric Social Identity Injection and Diversification in Public Opinion Simulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16142) • [📄 arXiv](https://arxiv.org/abs/2603.16142) • [📥 PDF](https://arxiv.org/pdf/2603.16142)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/halsayxi/PSII)

> Excited to share our work on Parametric Social Identity Injection (PSII) for socially diverse public opinion simulation with LLM agents. We identify Diversity Collapse , a phenomenon where prompt-conditioned LLM agents produce overly homogeneous r...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 15 |
| 📅 Today | [`2026-06-08.json`](data/daily/2026-06-08.json) | 15 |
| 📆 This Week | [`2026-W23.json`](data/weekly/2026-W23.json) | 15 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 193 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-08 | 15 | [View JSON](data/daily/2026-06-08.json) |
| 📄 2026-06-07 | 50 | [View JSON](data/daily/2026-06-07.json) |
| 📄 2026-06-06 | 50 | [View JSON](data/daily/2026-06-06.json) |
| 📄 2026-06-05 | 21 | [View JSON](data/daily/2026-06-05.json) |
| 📄 2026-06-04 | 0 | [View JSON](data/daily/2026-06-04.json) |
| 📄 2026-06-03 | 10 | [View JSON](data/daily/2026-06-03.json) |
| 📄 2026-06-02 | 22 | [View JSON](data/daily/2026-06-02.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W23 | 15 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 193 | [View JSON](data/monthly/2026-06.json) |
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
