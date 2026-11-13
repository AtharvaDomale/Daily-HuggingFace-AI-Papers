<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-6-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7939+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">6</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">39</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">211</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7939+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 13, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. BirdDreams: Joint Perspective and BEV Video Generation for a Plausible, Reactive Driving World Model</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.14815) • [📄 arXiv](https://arxiv.org/abs/2611.14815) • [📥 PDF](https://arxiv.org/pdf/2611.14815)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> BirdDreams: A driving world model with accurate ego control and traffic that can react. BirdDreams is a driving world model that generates driving video and a bird’s-eye view of the scene from a single camera image.  It follows input ego trajector...

</details>

<details>
<summary><b>2. On the Design of Hyper-Connections: Simplification, Stability, and Scalability</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.05984) • [📄 arXiv](https://arxiv.org/abs/2611.05984) • [📥 PDF](https://arxiv.org/pdf/2611.05984)

**💻 Code:** [⭐ Code](https://github.com/aHapBean/SimpleHC) • [⭐ Code](https://github.com/huggingface)

> We introduce SimpleHC, one of the fastest Hyper-Connections variants to date, reducing repeated memory access through a simpler design. It adds just 2.4% training overhead at the largest tested width while maintaining competitive performance with ...

</details>

<details>
<summary><b>3. Duo: A Reciprocal World Model for Full-Duplex Interaction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.15131) • [📄 arXiv](https://arxiv.org/abs/2611.15131) • [📥 PDF](https://arxiv.org/pdf/2611.15131)

**💻 Code:** [⭐ Code](https://github.com/inclusionAI/Duo) • [⭐ Code](https://github.com/huggingface)

> arXiv: https://arxiv.org/abs/2611.15131 Homepage: https://duo-v1.github.io/ Github: https://github.com/inclusionAI/Duo Hugging Face Model: https://huggingface.co/collections/inclusionAI/duo-v1

</details>

<details>
<summary><b>4. G-STAR: Group-Scaled Token Capacity for Deeply Compressed Autoregressive Generation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.13004) • [📄 arXiv](https://arxiv.org/abs/2611.13004) • [📥 PDF](https://arxiv.org/pdf/2611.13004)

**💻 Code:** [⭐ Code](https://github.com/DAGroup-PKU/G-STAR) • [⭐ Code](https://github.com/huggingface)

> Discrete autoregressive models offer a scalable framework for image generation. However, they face a fundamental trade-off: long token sequences preserve representation quality but slow generation, while aggressive spatial compression reduces repr...

</details>

<details>
<summary><b>5. From Large Language Models to Efficient System One Models via Adaptive Computation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Minh-Anh Nguyen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16321) • [📄 arXiv](https://arxiv.org/abs/2611.16321) • [📥 PDF](https://arxiv.org/pdf/2611.16321)

**💻 Code:** [⭐ Code](https://github.com/minhkks/NestOne) • [⭐ Code](https://github.com/huggingface)

> Transferring LLMs into System One decision models typically replaces the language modeling head with a decision head while retaining the full backbone, making every decision pay for all layers. Yet it remains unclear how much depth such decisions ...

</details>

<details>
<summary><b>6. PhysLoop: Harnessing LLMs' Physics Awareness for Self-Improving Video Generation Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kewei Zhang, Juncheng Ma, Hao Chen, Xiaojie Li, Yuxuan Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16748) • [📄 arXiv](https://arxiv.org/abs/2611.16748) • [📥 PDF](https://arxiv.org/pdf/2611.16748)

**💻 Code:** [⭐ Code](https://github.com/DAGroup-PKU/PhysLoop) • [⭐ Code](https://github.com/huggingface)

> PhysLoop is a physics-aware harness for building controllable video training data and repairing physically inconsistent video at inference time. The project video shows how the harness works and presents examples of video generation and repair. Co...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 6 |
| 📅 Today | [`2026-11-13.json`](data/daily/2026-11-13.json) | 6 |
| 📆 This Week | [`2026-W45.json`](data/weekly/2026-W45.json) | 39 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 211 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-13 | 6 | [View JSON](data/daily/2026-11-13.json) |
| 📄 2026-11-12 | 3 | [View JSON](data/daily/2026-11-12.json) |
| 📄 2026-11-11 | 11 | [View JSON](data/daily/2026-11-11.json) |
| 📄 2026-11-10 | 9 | [View JSON](data/daily/2026-11-10.json) |
| 📄 2026-11-09 | 10 | [View JSON](data/daily/2026-11-09.json) |
| 📄 2026-11-08 | 32 | [View JSON](data/daily/2026-11-08.json) |
| 📄 2026-11-07 | 32 | [View JSON](data/daily/2026-11-07.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W45 | 39 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 211 | [View JSON](data/monthly/2026-11.json) |
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
