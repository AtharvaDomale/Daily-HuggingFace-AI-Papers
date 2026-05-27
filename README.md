<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-17-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4242+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">48</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">621</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4242+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 27, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Geometry-Aware Representation Denoising for Robust Multi-view 3D Reconstruction</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26230) • [📄 arXiv](https://arxiv.org/abs/2605.26230) • [📥 PDF](https://arxiv.org/pdf/2605.26230)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cvlab-kaist/GARD)

> Project page: https://cvlab-kaist.github.io/GARD/

</details>

<details>
<summary><b>2. MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research</b> ⭐ 24</summary>

<br/>

**👥 Authors:** Han Xiao, Shuzhe Wu, Haiyang Wang, Rui Hao, Dingbang Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26114) • [📄 arXiv](https://arxiv.org/abs/2605.26114) • [📥 PDF](https://arxiv.org/pdf/2605.26114)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Purewhiter/mobilegym)

> MobileGym is a browser-hosted, lightweight, fully controllable, and highly parallel environment for everyday mobile use.

</details>

<details>
<summary><b>3. EvalVerse: Pipeline-Aware and Expert-Calibrated Benchmarking for Professional Cinematic Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23271) • [📄 arXiv](https://arxiv.org/abs/2605.23271) • [📥 PDF](https://arxiv.org/pdf/2605.23271)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The rapid evolution of generative video foundation models has propelled the field toward professional-grade cinematic synthesis. To achieve such demanding quality, the community transitions towards Reinforcement Learning (RL) and agentic workflows...

</details>

<details>
<summary><b>4. LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27365) • [📄 arXiv](https://arxiv.org/abs/2605.27365) • [📥 PDF](https://arxiv.org/pdf/2605.27365)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NVlabs/Eagle)

> Github: https://github.com/NVlabs/Eagle

</details>

<details>
<summary><b>5. SpatialBench: Is Your Spatial Foundation Model an All-Round Player?</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27367) • [📄 arXiv](https://arxiv.org/abs/2605.27367) • [📥 PDF](https://arxiv.org/pdf/2605.27367)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Ropedia/SpatialBench)

> The first benchmark for the spatial foundation model.

</details>

<details>
<summary><b>6. LongAV-Compass: Towards Unified Evaluation of Minute-Scale Audio-Visual Generation Across T2AV, I2AV, and V2AV</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26244) • [📄 arXiv](https://arxiv.org/abs/2605.26244) • [📥 PDF](https://arxiv.org/pdf/2605.26244)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/pkucs-Ltf/LongAV-Compass)

> Audio-visual generation is rapidly advancing from short clips to minute-long content, while existing evaluation protocols remain largely confined to short-form settings. Existing benchmarks primarily focus on 5--10 second text-conditioned generati...

</details>

<details>
<summary><b>7. Does Seeing More Mean Knowing More? Mono-Anchored Advantage Normalization for Multi-Source Visual Reasoning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25437) • [📄 arXiv](https://arxiv.org/abs/2605.25437) • [📥 PDF](https://arxiv.org/pdf/2605.25437)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AI9Stars/MARS)

> An effective way for resolving conflict in multi-source visual reasoning

</details>

<details>
<summary><b>8. The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26494) • [📄 arXiv](https://arxiv.org/abs/2605.26494) • [📥 PDF](https://arxiv.org/pdf/2605.26494)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. Efficient Agentic Reinforcement Learning with On-Policy Intrinsic Knowledge Boundary Enhancement</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26952) • [📄 arXiv](https://arxiv.org/abs/2605.26952) • [📥 PDF](https://arxiv.org/pdf/2605.26952)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/CuSO4-Chen/AKBE)

> We propose AKBE (Agentic Knowledge Boundary Enhancement), an on-policy method that dynamically probes the model's intrinsic knowledge boundary through dual-path (with-tool and no-tool) rollouts during training. We define the knowledge boundary as ...

</details>

<details>
<summary><b>10. Soap2Soap: Long Cinematic Video Remaking via Multi-Agent Collaboration</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.17423) • [📄 arXiv](https://arxiv.org/abs/2605.17423) • [📥 PDF](https://arxiv.org/pdf/2605.17423)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/showlab/Soap2Soap)

> Meet Soap2Soap — Video-to-Video generation via multi-agent collaboration. Transform any video into a fully stylized animated version — Pixar, Disney, LEGO, anime, clay, and more — with consistent characters, environments, and cinematic composition...

</details>

<details>
<summary><b>11. MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27366) • [📄 arXiv](https://arxiv.org/abs/2605.27366) • [📥 PDF](https://arxiv.org/pdf/2605.27366)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. Beyond Final Answers: Auditing Trajectory-Level Hallucinations in Multi-Agent Industrial Workflows</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.24219) • [📄 arXiv](https://arxiv.org/abs/2605.24219) • [📥 PDF](https://arxiv.org/pdf/2605.24219)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large Language Models (LLMs) are increasingly deployed as autonomous agents that reason, use tools, and act over multiple steps. Yet most hallucination benchmarks still evaluate only the final output, missing failures that originate in intermediat...

</details>

<details>
<summary><b>13. Squeezing Capacity from Multimodal Large Language Models for Subject-driven Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26111) • [📄 arXiv](https://arxiv.org/abs/2605.26111) • [📥 PDF](https://arxiv.org/pdf/2605.26111)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A comprehensive investigation of how to squeeze capacity from multimodal large language models for subject-driven generation, which requires both text instruction understanding and identity preservation for the subjects.

</details>

<details>
<summary><b>14. MobileMoE: Scaling On-Device Mixture of Experts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27358) • [📄 arXiv](https://arxiv.org/abs/2605.27358) • [📥 PDF](https://arxiv.org/pdf/2605.27358)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Seems really interesting and promising on mobile devices.

</details>

<details>
<summary><b>15. Negligible in Size, Significant in Effect: On Scale Vectors in Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26895) • [📄 arXiv](https://arxiv.org/abs/2605.26895) • [📥 PDF](https://arxiv.org/pdf/2605.26895)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This work provides a systematic study of scale vectors in LLMs, proposing a unified strategy of lightweight architectural improvements that enhances optimization and training efficiency across diverse model scales.

</details>

<details>
<summary><b>16. MRT: Masked Region Transformer for Layered Image Generation and Editing at Scale</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yifan Pu, Mohan Zhou, Jingye Chen, Zhao Zhang, Zhicong Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27235) • [📄 arXiv](https://arxiv.org/abs/2605.27235) • [📥 PDF](https://arxiv.org/pdf/2605.27235)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>17. RT-Lynx: Putting the GEMM Sparsity In a Right Way for Diffusion Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26632) • [📄 arXiv](https://arxiv.org/abs/2605.26632) • [📥 PDF](https://arxiv.org/pdf/2605.26632)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 👋 Hi everyone!  We’re excited to share our ICML 2026 work RT-Lynx: Putting GEMM Sparsity in the Right Place for Diffusion Models . Semi-structured sparsity has the potential to nearly halve GEMM FLOPs, but applying it to diffusion models remains c...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 17 |
| 📅 Today | [`2026-05-27.json`](data/daily/2026-05-27.json) | 17 |
| 📆 This Week | [`2026-W21.json`](data/weekly/2026-W21.json) | 48 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 621 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-27 | 17 | [View JSON](data/daily/2026-05-27.json) |
| 📄 2026-05-26 | 15 | [View JSON](data/daily/2026-05-26.json) |
| 📄 2026-05-25 | 16 | [View JSON](data/daily/2026-05-25.json) |
| 📄 2026-05-24 | 49 | [View JSON](data/daily/2026-05-24.json) |
| 📄 2026-05-23 | 49 | [View JSON](data/daily/2026-05-23.json) |
| 📄 2026-05-22 | 18 | [View JSON](data/daily/2026-05-22.json) |
| 📄 2026-05-21 | 18 | [View JSON](data/daily/2026-05-21.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W21 | 48 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 621 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |
| 🗓️ 2025-12 | 787 | [View JSON](data/monthly/2025-12.json) |

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
