<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-9-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3539+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">17</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">368</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3539+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 22, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. AgentSPEX: An Agent SPecification and EXecution Language</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.13346) • [📄 arXiv](https://arxiv.org/abs/2604.13346) • [📥 PDF](https://arxiv.org/pdf/2604.13346)

**💻 Code:** [⭐ Code](https://github.com/ScaleML/AgentSPEX)

> Right now, many agent workflows fall into two categories, either 1) they are primarily built with Python code—flexible, but increasingly hard to read, modify, and reproduce, or 2) they rely heavily on natural language (e.g., Markdown-based “skills...

</details>

<details>
<summary><b>2. PlayCoder: Making LLM-Generated GUI Code Playable</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.19742) • [📄 arXiv](https://arxiv.org/abs/2604.19742) • [📥 PDF](https://arxiv.org/pdf/2604.19742)

**💻 Code:** [⭐ Code](https://github.com/Tencent/PlayCoder)

> 🤖 Current code LLMs can generate GUI code that compiles, but rarely playable and interactively functional . This work builds a complete pipeline from evaluation to refinement for LLM-generated GUI programs. 🎮 PlayEval: a new multi-language benchma...

</details>

<details>
<summary><b>3. CoInteract: Physically-Consistent Human-Object Interaction Video Synthesis via Spatially-Structured Co-Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.19636) • [📄 arXiv](https://arxiv.org/abs/2604.19636) • [📥 PDF](https://arxiv.org/pdf/2604.19636)

**💻 Code:** [⭐ Code](https://github.com/luoxyhappy/CoInteract)

> We introduce CoInteract, a spatially-structured co-generation framework for speech-driven human-object interaction video synthesis. Project Page: https://xinxiaozhe12345.github.io/CoInteract_Project/ Code: https://github.com/luoxyhappy/CoInteract

</details>

<details>
<summary><b>4. LoopCTR: Unlocking the Loop Scaling Power for Click-Through Rate Prediction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chuan Wang, Yifei Liu, Weiqiu Wang, Runfeng Zhang, Jiakai Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.19550) • [📄 arXiv](https://arxiv.org/abs/2604.19550) • [📥 PDF](https://arxiv.org/pdf/2604.19550)

> 🔥 Recently, OpenMythos has been making waves in the AI community with its Recurrent-Depth Transformer , showing that scaling does not have to rely solely on stacking more layers or adding more parameters. Instead, recursive computation with shared...

</details>

<details>
<summary><b>5. UniMesh: Unifying 3D Mesh Understanding and Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.17472) • [📄 arXiv](https://arxiv.org/abs/2604.17472) • [📥 PDF](https://arxiv.org/pdf/2604.17472)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/UniMesh)

> Code coming soon. (Should be available these days.)

</details>

<details>
<summary><b>6. ClawNet: Human-Symbiotic Agent Network for Cross-User Autonomous Cooperation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wei Xue, Jun Song, Xianzhang Jia, Zhenyuan Zhang, Zhiqin Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.19211) • [📄 arXiv](https://arxiv.org/abs/2604.19211) • [📥 PDF](https://arxiv.org/pdf/2604.19211)

> No abstract available.

</details>

<details>
<summary><b>7. Tstars-Tryon 1.0: Robust and Realistic Virtual Try-On for Diverse Fashion Items</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Taihang Hu, Zuan Gao, Yongchao Du, Zhengrui Chen, Mengting Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.19748) • [📄 arXiv](https://arxiv.org/abs/2604.19748) • [📥 PDF](https://arxiv.org/pdf/2604.19748)

> GPT-Image 2.0 attempt to make a poster for this paper:

</details>

<details>
<summary><b>8. Evaluation-driven Scaling for Scientific Discovery</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Caiyin Yang, Yizhen Luo, Jingyi Tang, Haowei Lin, Haotian Ye

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.19341) • [📄 arXiv](https://arxiv.org/abs/2604.19341) • [📥 PDF](https://arxiv.org/pdf/2604.19341)

> GPT-Image 2.0 attempt to make a poster for this paper:

</details>

<details>
<summary><b>9. SPRITE: From Static Mockups to Engine-Ready Game UI</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ming Yan, Chien Her Lim, Hao Zhang, RuiHao Li, Yunshu Bai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.18591) • [📄 arXiv](https://arxiv.org/abs/2604.18591) • [📥 PDF](https://arxiv.org/pdf/2604.18591)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 9 |
| 📅 Today | [`2026-04-22.json`](data/daily/2026-04-22.json) | 9 |
| 📆 This Week | [`2026-W16.json`](data/weekly/2026-W16.json) | 17 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 368 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-22 | 9 | [View JSON](data/daily/2026-04-22.json) |
| 📄 2026-04-21 | 3 | [View JSON](data/daily/2026-04-21.json) |
| 📄 2026-04-20 | 5 | [View JSON](data/daily/2026-04-20.json) |
| 📄 2026-04-19 | 29 | [View JSON](data/daily/2026-04-19.json) |
| 📄 2026-04-18 | 29 | [View JSON](data/daily/2026-04-18.json) |
| 📄 2026-04-17 | 12 | [View JSON](data/daily/2026-04-17.json) |
| 📄 2026-04-16 | 10 | [View JSON](data/daily/2026-04-16.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W16 | 17 | [View JSON](data/weekly/2026-W16.json) |
| 📅 2026-W15 | 99 | [View JSON](data/weekly/2026-W15.json) |
| 📅 2026-W14 | 140 | [View JSON](data/weekly/2026-W14.json) |
| 📅 2026-W13 | 115 | [View JSON](data/weekly/2026-W13.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 368 | [View JSON](data/monthly/2026-04.json) |
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
