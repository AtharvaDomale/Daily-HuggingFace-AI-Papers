<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-9-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5114+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">9</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">12</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">99</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5114+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** July 07, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation</b> ⭐ 118</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02642) • [📄 arXiv](https://arxiv.org/abs/2607.02642) • [📥 PDF](https://arxiv.org/pdf/2607.02642)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/open-gigaai/giga-world-1)

> Code： https://github.com/open-gigaai/giga-world-1 Model： https://huggingface.co/open-gigaai/Giga-World-1 Data： https://huggingface.co/datasets/open-gigaai/CVPR-2026-WorldModel-Track-Dataset Benchmark： https://huggingface.co/spaces/open-gigaai/CVPR...

</details>

<details>
<summary><b>2. ResearchStudio-Reel: Automate the Last Mile of Research from Paper to Poster, Video, and Blog</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.04438) • [📄 arXiv](https://arxiv.org/abs/2607.04438) • [📥 PDF](https://arxiv.org/pdf/2607.04438)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> ResearchStudio - Reel From paper to poster, video, blog, and reel — Automating the Last Mile of Research Dissemination. ResearchStudio streamlines the final steps of a research project — the materials a paper needs after the writing is done. Drop ...

</details>

<details>
<summary><b>3. ResearchStudio-Idea: An Evidence-Grounded Research-Ideation Skill Suite from ML Conference Outcomes</b> ⭐ 84</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.04439) • [📄 arXiv](https://arxiv.org/abs/2607.04439) • [📥 PDF](https://arxiv.org/pdf/2607.04439)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/ResearchStudio)

> From a research problem to a reviewer-defensible idea card — Automating the First Mile of Research Ideation. ResearchStudio-Idea is a reusable research ideation skill suite that assists researchers in developing well-grounded research proposals. I...

</details>

<details>
<summary><b>4. Wan-Streamer v0.2: Higher Resolution, Same Latency</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.04443) • [📄 arXiv](https://arxiv.org/abs/2607.04443) • [📥 PDF](https://arxiv.org/pdf/2607.04443)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A latency-preserving Wan Streamer upgrade: from 192p close-up calls to 640×368 clearer calls and scene-grounded mid-shot agents, still at 25 fps with ~200 ms model-side latency.

</details>

<details>
<summary><b>5. Vision Pretraining for Dense Spatial Perception</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kecheng Zheng, Shaohui Liu, Changjiang Sun, Bin Tan, Zelin Fu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.05247) • [📄 arXiv](https://arxiv.org/abs/2607.05247) • [📥 PDF](https://arxiv.org/pdf/2607.05247)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Robbyant/lingbot-vision)

> A novel SSL pretraining for spatial perception with a STRONG depth model, LingBot-Depth 2.0

</details>

<details>
<summary><b>6. Safety Testing LLM Agents at Scale: From Risk Discovery to Evidence-Grounded Verification</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01793) • [📄 arXiv](https://arxiv.org/abs/2607.01793) • [📥 PDF](https://arxiv.org/pdf/2607.01793)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Yunhao-Feng/Vera)

> LLM agents increasingly perform autonomous actions through external tools, leading to complex and evolving safety risks. However, existing safety testing targets expert-designed safety violations, and the corresponding outcomes are evaluated by ha...

</details>

<details>
<summary><b>7. PraMem: Practice-derived Experiential Memory for Long-horizon Behavior Prediction</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Ruoxi Xu, Hanshu Zhou, Jiawei Chen, Boxi Cao, Zhuoqun Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02881) • [📄 arXiv](https://arxiv.org/abs/2607.02881) • [📥 PDF](https://arxiv.org/pdf/2607.02881)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/icip-cas/PraMem)

> Long-horizon behavior prediction aims to infer a user's next action based on a lengthy historical sequence, playing a crucial role in artificial intelligence field. The rise of large language models (LLMs) offers a promising direction for sequenti...

</details>

<details>
<summary><b>8. Mastermind: Strategy-grounded Learning for Repository-Scale Vulnerability Reproduction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhijiang Guo, Renyang Liu, Tianyi Wu, Luu Anh Tuan, Mingzhe Du

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01764) • [📄 arXiv](https://arxiv.org/abs/2607.01764) • [📥 PDF](https://arxiv.org/pdf/2607.01764)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Repository-level vulnerability reproduction is a demanding software engineering (SE) task: an agent must inspect a codebase, infer the input grammar that reaches a vulnerable path, construct a proof-of-conceptv(PoC), and verify that the crash disa...

</details>

<details>
<summary><b>9. Speaker-Disentangled Chunk-Wise Regression for Syllabic Tokenization</b> ⭐ 46</summary>

<br/>

**👥 Authors:** Takahiro Shinozaki, Takuma Okamoto, Kota Kawakita, Ryota Komatsu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.04064) • [📄 arXiv](https://arxiv.org/abs/2607.04064) • [📥 PDF](https://arxiv.org/pdf/2607.04064)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ryota-komatsu/speaker_disentangled_hubert)

> In this IEEE OJSP paper, we introduce SylReg-LM 7B, an efficiently scalable interleaved syllable-text language model!

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 9 |
| 📅 Today | [`2026-07-07.json`](data/daily/2026-07-07.json) | 9 |
| 📆 This Week | [`2026-W27.json`](data/weekly/2026-W27.json) | 12 |
| 🗓️ This Month | [`2026-07.json`](data/monthly/2026-07.json) | 99 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-07-07 | 9 | [View JSON](data/daily/2026-07-07.json) |
| 📄 2026-07-06 | 3 | [View JSON](data/daily/2026-07-06.json) |
| 📄 2026-07-05 | 27 | [View JSON](data/daily/2026-07-05.json) |
| 📄 2026-07-04 | 27 | [View JSON](data/daily/2026-07-04.json) |
| 📄 2026-07-03 | 12 | [View JSON](data/daily/2026-07-03.json) |
| 📄 2026-07-02 | 10 | [View JSON](data/daily/2026-07-02.json) |
| 📄 2026-07-01 | 11 | [View JSON](data/daily/2026-07-01.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W27 | 12 | [View JSON](data/weekly/2026-W27.json) |
| 📅 2026-W26 | 117 | [View JSON](data/weekly/2026-W26.json) |
| 📅 2026-W25 | 108 | [View JSON](data/weekly/2026-W25.json) |
| 📅 2026-W24 | 130 | [View JSON](data/weekly/2026-W24.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-07 | 99 | [View JSON](data/monthly/2026-07.json) |
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
