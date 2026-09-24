<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-10-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6462+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">41</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">334</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6462+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** September 24, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. The Past Frames the Future: Memory for Autoregressive Video Generation</b> ⭐ 30</summary>

<br/>

**👥 Authors:** Hongfei Zhang, Wen-Jie Shu, Disen Lan, Rongjin Guo, Harold Haodong Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.28466) • [📄 arXiv](https://arxiv.org/abs/2609.28466) • [📥 PDF](https://arxiv.org/pdf/2609.28466)

**💻 Code:** [⭐ Code](https://github.com/HaroldChen19/Awesome-AR-Video-Memory) • [⭐ Code](https://github.com/huggingface)

> The first survey on memory mechanisms for long video generation.

</details>

<details>
<summary><b>2. SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue</b> ⭐ 70</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.26780) • [📄 arXiv](https://arxiv.org/abs/2609.26780) • [📥 PDF](https://arxiv.org/pdf/2609.26780)

**💻 Code:** [⭐ Code](https://github.com/2022hpsk/SpeakerMemR1) • [⭐ Code](https://github.com/huggingface)

> Excited to share our work on SpeakerMem-R1 ! Recent benchmarks reveal a surprising gap: general-purpose memory systems struggle with multi-party conversations and can even underperform simple BM25 retrieval in some settings. These limitations high...

</details>

<details>
<summary><b>3. RewardVerse: Rubric-Guided Policy Optimization for Video Reward Modeling</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.22947) • [📄 arXiv](https://arxiv.org/abs/2609.22947) • [📥 PDF](https://arxiv.org/pdf/2609.22947)

**💻 Code:** [⭐ Code](https://github.com/2kxx/RewardVerse) • [⭐ Code](https://github.com/huggingface)

> Reinforcement learning (RL) is vital for optimizing video generation models, with a robust reward model (RM) serving as the cornerstone. However, existing video reward models often produce unstable scalar scores because they directly map complex, ...

</details>

<details>
<summary><b>4. Schrödinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.27891) • [📄 arXiv](https://arxiv.org/abs/2609.27891) • [📥 PDF](https://arxiv.org/pdf/2609.27891)

**💻 Code:** [⭐ Code](https://github.com/cslsolow/Schrodinger-Repo) • [⭐ Code](https://github.com/huggingface)

> What if a code repository were like Schrödinger’s cat—its final form only revealed when the agent opens the box? We introduce Schrödinger’s Repository : at evaluation time, the same SWE task “collapses” into a behavior-equivalent but unfamiliar re...

</details>

<details>
<summary><b>5. PACT: From Credit Assignment to Critic Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.26355) • [📄 arXiv](https://arxiv.org/abs/2609.26355) • [📥 PDF](https://arxiv.org/pdf/2609.26355)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We characterize token-level credit through three conditions that uniquely determine its form, and use this perspective to understand RLOO, GAE, and on-policy distillation. These insights lead to PACT, which improves critic learning and alignment w...

</details>

<details>
<summary><b>6. WhatWorkedBench: Benchmarking Experimental Understanding in AI Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.27490) • [📄 arXiv](https://arxiv.org/abs/2609.27490) • [📥 PDF](https://arxiv.org/pdf/2609.27490)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/EthanNing/WhatWorkedBench)

> Benchmarking Experimental Understanding in AI Agents

</details>

<details>
<summary><b>7. Hunyuan-A13B Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.27284) • [📄 arXiv](https://arxiv.org/abs/2609.27284) • [📥 PDF](https://arxiv.org/pdf/2609.27284)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>8. Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yang Su, Jianhong Tu, Xudong Guo, Wei Fan, Xinjie Shen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.27321) • [📄 arXiv](https://arxiv.org/abs/2609.27321) • [📥 PDF](https://arxiv.org/pdf/2609.27321)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. InternW0: A Foundational Physical World Model for Efficient Real-World Interactions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhangzheng Tu, Zhe Cao, Ganlin Yang, Yao Mu, Jisong Cai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.27656) • [📄 arXiv](https://arxiv.org/abs/2609.27656) • [📥 PDF](https://arxiv.org/pdf/2609.27656)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. MemBodied: Recurrent Associative Memory for Vision-Language-Action Models</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jianfei Yang, Raphael Yee, Bryce Goh, Navonil Majumder, Tej Deep Pala

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.28256) • [📄 arXiv](https://arxiv.org/abs/2609.28256) • [📥 PDF](https://arxiv.org/pdf/2609.28256)

**💻 Code:** [⭐ Code](https://github.com/declare-lab/MemBodied) • [⭐ Code](https://github.com/huggingface)

> Recurrent Associative Memory for VLAs

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 10 |
| 📅 Today | [`2026-09-24.json`](data/daily/2026-09-24.json) | 10 |
| 📆 This Week | [`2026-W38.json`](data/weekly/2026-W38.json) | 41 |
| 🗓️ This Month | [`2026-09.json`](data/monthly/2026-09.json) | 334 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-09-24 | 10 | [View JSON](data/daily/2026-09-24.json) |
| 📄 2026-09-23 | 10 | [View JSON](data/daily/2026-09-23.json) |
| 📄 2026-09-22 | 10 | [View JSON](data/daily/2026-09-22.json) |
| 📄 2026-09-21 | 11 | [View JSON](data/daily/2026-09-21.json) |
| 📄 2026-09-20 | 24 | [View JSON](data/daily/2026-09-20.json) |
| 📄 2026-09-19 | 24 | [View JSON](data/daily/2026-09-19.json) |
| 📄 2026-09-18 | 14 | [View JSON](data/daily/2026-09-18.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W38 | 41 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |
| 📅 2026-W36 | 88 | [View JSON](data/weekly/2026-W36.json) |
| 📅 2026-W35 | 121 | [View JSON](data/weekly/2026-W35.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-09 | 334 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |

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
