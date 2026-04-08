<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-8-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3313+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">8</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">30</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">142</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3313+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 08, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding</b> ⭐ 145</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.05015) • [📄 arXiv](https://arxiv.org/abs/2604.05015) • [📥 PDF](https://arxiv.org/pdf/2604.05015)

**💻 Code:** [⭐ Code](https://github.com/MME-Benchmarks/Video-MME-v2)

> Video-MME-v2: Towards the Next Stage in Video Understanding Evaluation Technical Report : https://arxiv.org/pdf/2604.05015 Project Page : https://video-mme-v2.netlify.app/ Leaderboard : https://video-mme-v2.netlify.app/#leaderboard GitHub: https :...

</details>

<details>
<summary><b>2. Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Zehui Chen, Ziyan Chen, Zhen Fang, Shiting Huang, Qisheng Su

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.05404) • [📄 arXiv](https://arxiv.org/abs/2604.05404) • [📥 PDF](https://arxiv.org/pdf/2604.05404)

**💻 Code:** [⭐ Code](https://github.com/sqs-ustc/tool-reasoning-framework-PTE)

> In real-world Tool-Integrated Reasoning (TIR) scenarios, a major source of inefficiency is that the toolcalls create pauses between LLM requests and cause KV-cache eviction. Also, the long, unfiltered response returned by external tools inflates t...

</details>

<details>
<summary><b>3. Learning to Retrieve from Agent Trajectories</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.04949) • [📄 arXiv](https://arxiv.org/abs/2604.04949) • [📥 PDF](https://arxiv.org/pdf/2604.04949)

**💻 Code:** [⭐ Code](https://github.com/Yuqi-Zhou/LRAT)

> Key insights: We identify a fundamental misalignment between human-centric retrieval training and agentic search, and formulate learning to retrieve from agent trajectories as a new retrieval paradigm. In this setting, supervision is derived from ...

</details>

<details>
<summary><b>4. Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Linli Yao, Yuanxin Liu, Qibin Yang, Rang Li, Bowen Ye

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.06132) • [📄 arXiv](https://arxiv.org/abs/2604.06132) • [📥 PDF](https://arxiv.org/pdf/2604.06132)

**💻 Code:** [⭐ Code](https://github.com/claw-eval/claw-eval)

> Claw-Eval

</details>

<details>
<summary><b>5. FactReview: Evidence-Grounded Reviews with Literature Positioning and Execution-Based Claim Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.04074) • [📄 arXiv](https://arxiv.org/abs/2604.04074) • [📥 PDF](https://arxiv.org/pdf/2604.04074)

> Peer review in ML is under real pressure — submission volumes keep growing, but reviewer bandwidth doesn't scale with it. This paper tackles a critical piece of the problem: grounding automated reviews in actual evidence rather than just reading t...

</details>

<details>
<summary><b>6. MMEmb-R1: Reasoning-Enhanced Multimodal Embedding with Pair-Aware Selection and Adaptive Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.06156) • [📄 arXiv](https://arxiv.org/abs/2604.06156) • [📥 PDF](https://arxiv.org/pdf/2604.06156)

> MLLMs have been successfully applied to multimodal embedding tasks, yet their generative reasoning capabilities remain underutilized. Directly incorporating chain-of-thought reasoning into embedding learning introduces two fundamental challenges. ...

</details>

<details>
<summary><b>7. Demystifying When Pruning Works via Representation Hierarchies</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24652) • [📄 arXiv](https://arxiv.org/abs/2603.24652) • [📥 PDF](https://arxiv.org/pdf/2603.24652)

**💻 Code:** [⭐ Code](https://github.com/CASE-Lab-UMD/Pruning-on-Representations)

> Network pruning, which removes less important parameters or architectures, is often expected to improve efficiency while preserving performance. However, this expectation does not consistently hold across language tasks: pruned models can perform ...

</details>

<details>
<summary><b>8. ACES: Who Tests the Tests? Leave-One-Out AUC Consistency for Code Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yali Du, Ren-Biao Liu, Zheng Xie, Yun-Ji Zhang, sun0o0

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.03922) • [📄 arXiv](https://arxiv.org/abs/2604.03922) • [📥 PDF](https://arxiv.org/pdf/2604.03922)

> Selecting LLM-generated code candidates using LLM-generated tests is challenging because the tests themselves may be incorrect. Existing methods either treat all tests equally or rely on ad-hoc heuristics to filter unreliable tests. Yet determinin...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 8 |
| 📅 Today | [`2026-04-08.json`](data/daily/2026-04-08.json) | 8 |
| 📆 This Week | [`2026-W14.json`](data/weekly/2026-W14.json) | 30 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 142 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-08 | 8 | [View JSON](data/daily/2026-04-08.json) |
| 📄 2026-04-07 | 17 | [View JSON](data/daily/2026-04-07.json) |
| 📄 2026-04-06 | 5 | [View JSON](data/daily/2026-04-06.json) |
| 📄 2026-04-05 | 45 | [View JSON](data/daily/2026-04-05.json) |
| 📄 2026-04-04 | 45 | [View JSON](data/daily/2026-04-04.json) |
| 📄 2026-04-03 | 7 | [View JSON](data/daily/2026-04-03.json) |
| 📄 2026-04-02 | 5 | [View JSON](data/daily/2026-04-02.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W14 | 30 | [View JSON](data/weekly/2026-W14.json) |
| 📅 2026-W13 | 115 | [View JSON](data/weekly/2026-W13.json) |
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 142 | [View JSON](data/monthly/2026-04.json) |
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
