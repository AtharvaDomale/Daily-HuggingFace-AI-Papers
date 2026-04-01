<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-10-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3181+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">10</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">13</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">10</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3181+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 01, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. LongCat-Next: Lexicalizing Modalities as Discrete Tokens</b> ⭐ 278</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.27538) • [📄 arXiv](https://arxiv.org/abs/2603.27538) • [📥 PDF](https://arxiv.org/pdf/2603.27538)

**💻 Code:** [⭐ Code](https://github.com/meituan-longcat/LongCat-Next)

> An open-source native multimodal model built with a pure discrete autoregressive architecture, supporting unified modeling for visual understanding, generation, audio, and language tasks.

</details>

<details>
<summary><b>2. Lingshu-Cell: A generative cellular world model for transcriptome modeling toward virtual cells</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25240) • [📄 arXiv](https://arxiv.org/abs/2603.25240) • [📥 PDF](https://arxiv.org/pdf/2603.25240)

> ✨ Highlights Lingshu-Cell introduces a generative cellular world model for single-cell transcriptomics based on a masked discrete diffusion framework. Lingshu-Cell performs transcriptome-wide modeling over ~18,000 genes directly in a discrete toke...

</details>

<details>
<summary><b>3. CARLA-Air: Fly Drones Inside a CARLA World -- A Unified Infrastructure for Air-Ground Embodied Intelligence</b> ⭐ 214</summary>

<br/>

**👥 Authors:** Hong Zhang, Yanci Wen, Hanxuan Chen, tianlezeng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.28032) • [📄 arXiv](https://arxiv.org/abs/2603.28032) • [📥 PDF](https://arxiv.org/pdf/2603.28032)

**💻 Code:** [⭐ Code](https://github.com/louiszengCN/CarlaAir)

> CARLA-Air is an open-source infrastructure that unifies high-fidelity urban driving and physics-accurate multirotor flight within a single Unreal Engine process, providing a practical simulation foundation for air-ground embodied intelligence rese...

</details>

<details>
<summary><b>4. OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.28858) • [📄 arXiv](https://arxiv.org/abs/2603.28858) • [📥 PDF](https://arxiv.org/pdf/2603.28858)

**💻 Code:** [⭐ Code](https://github.com/shyyhs/optimer)

> 🤯 Struggling with dataset mixing ratios in LLM continual training? 🧩 We propose OptiMer: train one model per dataset, then merge them optimally. No more costly ratio tuning! 📄OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing ...

</details>

<details>
<summary><b>5. VGGRPO: Towards World-Consistent Video Generation with 4D Latent Reward</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.26599) • [📄 arXiv](https://arxiv.org/abs/2603.26599) • [📥 PDF](https://arxiv.org/pdf/2603.26599)

> No abstract available.

</details>

<details>
<summary><b>6. AutoWeather4D: Autonomous Driving Video Weather Conversion via G-Buffer Dual-Pass Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.26546) • [📄 arXiv](https://arxiv.org/abs/2603.26546) • [📥 PDF](https://arxiv.org/pdf/2603.26546)

**💻 Code:** [⭐ Code](https://github.com/lty2226262/AutoWeather4D)

> No abstract available.

</details>

<details>
<summary><b>7. Think Anywhere in Code Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Taozhi Chen, Mengyang Liu, Ge Li, Tianyu Zhang, Xue Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.29957) • [📄 arXiv](https://arxiv.org/abs/2603.29957) • [📥 PDF](https://arxiv.org/pdf/2603.29957)

**💻 Code:** [⭐ Code](https://github.com/jiangxxxue/Think-Anywhere)

> No abstract available.

</details>

<details>
<summary><b>8. SeGPruner: Semantic-Geometric Visual Token Pruner for 3D Question Answering</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.29437) • [📄 arXiv](https://arxiv.org/abs/2603.29437) • [📥 PDF](https://arxiv.org/pdf/2603.29437)

**💻 Code:** [⭐ Code](https://github.com/intcomp/SegPruner)

> SeGPruner is a token pruning framework for 3D visual question answering that combines attention-based saliency selection with geometry-guided spatial diversification to reduce visual token count by 91% and inference latency by 86%, while preservin...

</details>

<details>
<summary><b>9. VectorGym: A Multitask Benchmark for SVG Code Generation, Sketching, and Editing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rishav Pramanik, Tianyang Zhang, Abhay Puri, Haotian Zhang, Juan Rodriguez

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.29852) • [📄 arXiv](https://arxiv.org/abs/2603.29852) • [📥 PDF](https://arxiv.org/pdf/2603.29852)

> No abstract available.

</details>

<details>
<summary><b>10. Learn2Fold: Structured Origami Generation with World Model Planning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhengzhong Tu, Jinru Han, Ying Jiang, Yunuo Chen, Yanjia Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.29585) • [📄 arXiv](https://arxiv.org/abs/2603.29585) • [📥 PDF](https://arxiv.org/pdf/2603.29585)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 10 |
| 📅 Today | [`2026-04-01.json`](data/daily/2026-04-01.json) | 10 |
| 📆 This Week | [`2026-W13.json`](data/weekly/2026-W13.json) | 13 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 10 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-01 | 10 | [View JSON](data/daily/2026-04-01.json) |
| 📄 2026-03-31 | 1 | [View JSON](data/daily/2026-03-31.json) |
| 📄 2026-03-30 | 2 | [View JSON](data/daily/2026-03-30.json) |
| 📄 2026-03-29 | 29 | [View JSON](data/daily/2026-03-29.json) |
| 📄 2026-03-28 | 29 | [View JSON](data/daily/2026-03-28.json) |
| 📄 2026-03-27 | 6 | [View JSON](data/daily/2026-03-27.json) |
| 📄 2026-03-26 | 4 | [View JSON](data/daily/2026-03-26.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W13 | 13 | [View JSON](data/weekly/2026-W13.json) |
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |
| 📅 2026-W10 | 119 | [View JSON](data/weekly/2026-W10.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 10 | [View JSON](data/monthly/2026-04.json) |
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
