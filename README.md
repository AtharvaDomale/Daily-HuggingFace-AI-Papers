<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-10-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3452+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">29</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">281</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3452+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 16, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. GameWorld: Towards Standardized and Verifiable Evaluation of Multimodal Game Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07429) • [📄 arXiv](https://arxiv.org/abs/2604.07429) • [📥 PDF](https://arxiv.org/pdf/2604.07429)

**💻 Code:** [⭐ Code](https://github.com/gameworld-project/gameworld)

> Proposes GameWorld, a standardized, verifiable benchmark for evaluating multimodal game agents across 34 browser games and 170 tasks using semantic actions and keyboard/mouse interfaces.

</details>

<details>
<summary><b>2. OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language World Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.10866) • [📄 arXiv](https://arxiv.org/abs/2604.10866) • [📥 PDF](https://arxiv.org/pdf/2604.10866)

**💻 Code:** [⭐ Code](https://github.com/GregxmHu/OccuBench)

> AI agents excel at coding and web tasks, but most real-world professional scenarios lack executable environments. OccuBench uses LLM-based Language World Models to simulate 100 professional task environments (emergency triage, customs processing, ...

</details>

<details>
<summary><b>3. Seedance 2.0: Advancing Video Generation for World Complexity</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14148) • [📄 arXiv](https://arxiv.org/abs/2604.14148) • [📥 PDF](https://arxiv.org/pdf/2604.14148)

> Seedance 2.0 Model Card, Official Page: https://seed.bytedance.com/seedance2_0

</details>

<details>
<summary><b>4. TREX: Automating LLM Fine-tuning via Agent-Driven Tree-based Exploration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14116) • [📄 arXiv](https://arxiv.org/abs/2604.14116) • [📥 PDF](https://arxiv.org/pdf/2604.14116)

**💻 Code:** [⭐ Code](https://github.com/trex-project)

> While Large Language Models (LLMs) have empowered AI research agents to perform isolated scientific tasks, automating complex, real-world workflows, such as LLM training, remains a significant challenge. In this paper, we introduce TREX, a multi-a...

</details>

<details>
<summary><b>5. TIP: Token Importance in On-Policy Distillation</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14084) • [📄 arXiv](https://arxiv.org/abs/2604.14084) • [📥 PDF](https://arxiv.org/pdf/2604.14084)

**💻 Code:** [⭐ Code](https://github.com/HJSang/OPSD_OnPolicyDistillation)

> Useful learning signal in OPD concentrates in uncertain tokens and overconfident mistakes, and selecting tokens based on entropy + divergence enables more efficient training with far fewer tokens.

</details>

<details>
<summary><b>6. SpatialEvo: Self-Evolving Spatial Intelligence via Deterministic Geometric Environments</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Hongbo Peng, Kangheng Lin, Xinrui Cheng, Yingxiu Zhao, Dinging Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14144) • [📄 arXiv](https://arxiv.org/abs/2604.14144) • [📥 PDF](https://arxiv.org/pdf/2604.14144)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/SpatialEvo)

> No abstract available.

</details>

<details>
<summary><b>7. UI-Copilot: Advancing Long-Horizon GUI Automation via Tool-Integrated Policy Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xu Tan, Kaitao Song, Guangyi Liu, Fei Tang, Zhengxi Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.13822) • [📄 arXiv](https://arxiv.org/abs/2604.13822) • [📥 PDF](https://arxiv.org/pdf/2604.13822)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/UI-Copilot)

> We propose UI-Copilot, a collaborative framework where the GUI agent selectively invokes a lightweight copilot for memory retrieval and numerical computation, enabling efficient long-horizon GUI navigation.

</details>

<details>
<summary><b>8. UI-Zoomer: Uncertainty-Driven Adaptive Zoom-In for GUI Grounding</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Songqin Nong, Tongbo Chen, Zhengxi Lu, Bofan Chen, Fei Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14113) • [📄 arXiv](https://arxiv.org/abs/2604.14113) • [📥 PDF](https://arxiv.org/pdf/2604.14113)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/UI-Zoomer)

> No abstract available.

</details>

<details>
<summary><b>9. ROSE: Retrieval-Oriented Segmentation Enhancement</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu-Gang Jiang, Henghui Ding, Guangquan Jie, Song Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14147) • [📄 arXiv](https://arxiv.org/abs/2604.14147) • [📥 PDF](https://arxiv.org/pdf/2604.14147)

> No abstract available.

</details>

<details>
<summary><b>10. InfiniteScienceGym: An Unbounded, Procedurally-Generated Benchmark for Scientific Analysis</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vivek Srikumar, Oliver Bentham

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.13201) • [📄 arXiv](https://arxiv.org/abs/2604.13201) • [📥 PDF](https://arxiv.org/pdf/2604.13201)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 10 |
| 📅 Today | [`2026-04-16.json`](data/daily/2026-04-16.json) | 10 |
| 📆 This Week | [`2026-W15.json`](data/weekly/2026-W15.json) | 29 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 281 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-16 | 10 | [View JSON](data/daily/2026-04-16.json) |
| 📄 2026-04-15 | 5 | [View JSON](data/daily/2026-04-15.json) |
| 📄 2026-04-14 | 5 | [View JSON](data/daily/2026-04-14.json) |
| 📄 2026-04-13 | 9 | [View JSON](data/daily/2026-04-13.json) |
| 📄 2026-04-12 | 42 | [View JSON](data/daily/2026-04-12.json) |
| 📄 2026-04-11 | 42 | [View JSON](data/daily/2026-04-11.json) |
| 📄 2026-04-10 | 18 | [View JSON](data/daily/2026-04-10.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W15 | 29 | [View JSON](data/weekly/2026-W15.json) |
| 📅 2026-W14 | 140 | [View JSON](data/weekly/2026-W14.json) |
| 📅 2026-W13 | 115 | [View JSON](data/weekly/2026-W13.json) |
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 281 | [View JSON](data/monthly/2026-04.json) |
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
