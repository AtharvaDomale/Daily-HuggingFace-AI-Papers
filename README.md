<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-16-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4210+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">16</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">16</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">589</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4210+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 25, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Rethinking Cross-Layer Information Routing in Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20708) • [📄 arXiv](https://arxiv.org/abs/2605.20708) • [📥 PDF](https://arxiv.org/pdf/2605.20708)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi everyone! We’re excited to share our work “Rethinking Cross-Layer Information Routing in Diffusion Transformers.” While DiTs have been extensively improved in tokenization, attention, conditioning, objectives, and autoencoders, their residual s...

</details>

<details>
<summary><b>2. Lens: Rethinking Training Efficiency for Foundational Text-to-Image Models</b> ⭐ 81</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21573) • [📄 arXiv](https://arxiv.org/abs/2605.21573) • [📥 PDF](https://arxiv.org/pdf/2605.21573)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/Lens)

> Lens is a 3.8B-parameter foundational text-to-image model designed for efficient training and fast high-resolution generation . It combines dense-caption pre-training, mixed-resolution learning, GPT-OSS multi-layer text features, and the FLUX.2 se...

</details>

<details>
<summary><b>3. See What I Mean: Aligning Vision and Language Representations for Video Fine-grained Object Understanding</b> ⭐ 78</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.18018) • [📄 arXiv](https://arxiv.org/abs/2605.18018) • [📥 PDF](https://arxiv.org/pdf/2605.18018)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HumanMLLM/SWIM)

> A new paradigm for fine-grained video understanding.

</details>

<details>
<summary><b>4. StepAudio 2.5 Technical Report</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chen Wu, Chao Yan, Boyong Wu, Bo Zhao, Bin Lin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23463) • [📄 arXiv](https://arxiv.org/abs/2605.23463) • [📥 PDF](https://arxiv.org/pdf/2605.23463)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This report presents StepAudio 2.5, a unified audiolanguage foundation model that matches or exceeds specialized systems across all three capabilities.

</details>

<details>
<summary><b>5. RankE: End-to-End Post-Training for Discrete Text-to-Image Generation with Decoder Co-Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21195) • [📄 arXiv](https://arxiv.org/abs/2605.21195) • [📥 PDF](https://arxiv.org/pdf/2605.21195)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/syjmelody/RankE)

> Project Page: https://syjmelody.github.io/RankE/ GitHub: https://github.com/syjmelody/RankE ⚡ TL;DR RankE is the first end-to-end post-training framework for discrete text-to-image generation that jointly optimizes the Generator and the Decoder . ...

</details>

<details>
<summary><b>6. SkillOpt: Executive Strategy for Self-Evolving Agent Skills</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ziwei Zhou, Qihao Yang, Weiquan Huang, Ziyang Gong, Yifan Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23904) • [📄 arXiv](https://arxiv.org/abs/2605.23904) • [📥 PDF](https://arxiv.org/pdf/2605.23904)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. SCOPE: Simulating Cross-game Operations in Playable Environments for FPS World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kexu Cheng, Zhaohu Xing, Zeqing Wang, Hongfeng Lai, Zizhao Tong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23345) • [📄 arXiv](http://arxiv.org/abs/2605.23345) • [📥 PDF](https://arxiv.org/pdf/2605.23345)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/z2tong/SCOPE)

> 🚀 Introducing SCOPE — an interactive world model for FPS games. SCOPE handles dense FPS controls by learning per-pixel temporal action responses, separating localized weapon/scope effects from stable scene generation without segmentation labels. W...

</details>

<details>
<summary><b>8. PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23902) • [📄 arXiv](https://arxiv.org/abs/2605.23902) • [📥 PDF](https://arxiv.org/pdf/2605.23902)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. Geo-Align: Video Generation Alignment via Metric Geometry Reward</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tong He, Chunhua Shen, Runzhe Teng, Haoyu Guo, Zizun Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23903) • [📄 arXiv](https://arxiv.org/abs/2605.23903) • [📥 PDF](https://arxiv.org/pdf/2605.23903)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LiZizun/GeoAlign)

> Camera-controlled video generation has achieved remarkable progress in recent years. However, existing video-to-video re-rendering methods primarily rely on Supervised Fine-Tuning using synthetic datasets. At present, there is an extreme scarcity ...

</details>

<details>
<summary><b>10. ETCHR: Editing To Clarify and Harness Reasoning</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23897) • [📄 arXiv](https://arxiv.org/abs/2605.23897) • [📥 PDF](https://arxiv.org/pdf/2605.23897)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/InternLM/ETCHR)

> We introduce ETCHR (Editing To Clarify and Harness Reasoning), a question-conditioned, reasoning-aware image editor decoupled from the downstream understanding model and trained with a two-stage recipe targeted at the two gaps: Reasoning Imitation...

</details>

<details>
<summary><b>11. VGenST-Bench: A Benchmark for Spatio-Temporal Reasoning via Active Video Synthesis</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Eunbyung Park, Hogun Park, Youbin Kim, zino1

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22570) • [📄 arXiv](https://arxiv.org/abs/2605.22570) • [📥 PDF](https://arxiv.org/pdf/2605.22570)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zinosii/VGenST-Bench)

> Awesome work!

</details>

<details>
<summary><b>12. Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19282) • [📄 arXiv](https://arxiv.org/abs/2605.19282) • [📥 PDF](https://arxiv.org/pdf/2605.19282)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OPTML-Group/Pion)

> This paper proposes a new optimizer called Pion, which replaces Muon's uniform spectral whitening with a spectral high-pass filtering mechanism, successfully addressing the performance failures that occur in non-LLM-pretraining scenarios such as V...

</details>

<details>
<summary><b>13. GenRecon: Bridging Generative Priors for Multi-View 3D Scene Reconstruction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Matthias Nießner, Angela Dai, Jozef Hladký, Nicolas von Lützow, Katharina Schmid

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23888) • [📄 arXiv](https://arxiv.org/abs/2605.23888) • [📥 PDF](https://arxiv.org/pdf/2605.23888)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuan Yang, Jing Liu, Yuhang Cai, Deyi Liu, Xu Ouyang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23901) • [📄 arXiv](https://arxiv.org/abs/2605.23901) • [📥 PDF](https://arxiv.org/pdf/2605.23901)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. From Raw Experience to Skill Consumption: A Systematic Study of Model-Generated Agent Skills</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qihao Yang, Ziyang Gong, Yifan Yang, Jingwen Xu, Zisu Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23899) • [📄 arXiv](https://arxiv.org/abs/2605.23899) • [📥 PDF](https://arxiv.org/pdf/2605.23899)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>16. HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sung Ju Hwang, Taekyung Ki, Yumin Choi, Woongyeng Yeo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.17873) • [📄 arXiv](https://arxiv.org/abs/2605.17873) • [📥 PDF](https://arxiv.org/pdf/2605.17873)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We propose HINT-SD, a targeted hindsight self-distillation framework for long-horizon agents that improves learning by identifying and correcting only the actions responsible for task failure. Instead of distilling entire trajectories, HINT-SD per...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 16 |
| 📅 Today | [`2026-05-25.json`](data/daily/2026-05-25.json) | 16 |
| 📆 This Week | [`2026-W21.json`](data/weekly/2026-W21.json) | 16 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 589 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-25 | 16 | [View JSON](data/daily/2026-05-25.json) |
| 📄 2026-05-24 | 49 | [View JSON](data/daily/2026-05-24.json) |
| 📄 2026-05-23 | 49 | [View JSON](data/daily/2026-05-23.json) |
| 📄 2026-05-22 | 18 | [View JSON](data/daily/2026-05-22.json) |
| 📄 2026-05-21 | 18 | [View JSON](data/daily/2026-05-21.json) |
| 📄 2026-05-20 | 16 | [View JSON](data/daily/2026-05-20.json) |
| 📄 2026-05-19 | 15 | [View JSON](data/daily/2026-05-19.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W21 | 16 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 589 | [View JSON](data/monthly/2026-05.json) |
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
