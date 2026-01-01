<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-7-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-745+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">7</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">7</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">7</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">745+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 01, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement</b> ⭐ 110</summary>

<br/>

**👥 Authors:** Yang Li, Dehao Hao, LutaoJiang, StarYDY, infinith

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21185) • [📄 arXiv](https://arxiv.org/abs/2512.21185) • [📥 PDF](https://arxiv.org/pdf/2512.21185)

**💻 Code:** [⭐ Code](https://github.com/PKU-YuanGroup/UltraShape-1.0)

> Github: https://github.com/PKU-YuanGroup/UltraShape-1.0 Project Page: https://pku-yuangroup.github.io/UltraShape-1.0/

</details>

<details>
<summary><b>2. DreamOmni3: Scribble-based Editing and Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22525) • [📄 arXiv](https://arxiv.org/abs/2512.22525) • [📥 PDF](https://arxiv.org/pdf/2512.22525)

**💻 Code:** [⭐ Code](https://github.com/dvlab-research/DreamOmni3)

> Github: https://github.com/dvlab-research/DreamOmni3

</details>

<details>
<summary><b>3. End-to-End Test-Time Training for Long Context</b> ⭐ 96</summary>

<br/>

**👥 Authors:** Marcel Rød, Daniel Koceja, Xinhao Li, Karan Dalal, Arnuv Tandon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23675) • [📄 arXiv](https://arxiv.org/abs/2512.23675) • [📥 PDF](https://arxiv.org/pdf/2512.23675)

**💻 Code:** [⭐ Code](https://github.com/test-time-training/e2e)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Sliding Window Attention Adaptation (2025) Let's (not) just put things in C...

</details>

<details>
<summary><b>4. Evaluating Parameter Efficient Methods for RLVR</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23165) • [📄 arXiv](https://arxiv.org/abs/2512.23165) • [📥 PDF](https://arxiv.org/pdf/2512.23165)

> https://www.alphaxiv.org/abs/2512.23165

</details>

<details>
<summary><b>5. GraphLocator: Graph-guided Causal Reasoning for Issue Localization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wei Zhang, Aofan Liu, Pengfei Gao, Wei Liu, pengchao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22469) • [📄 arXiv](https://arxiv.org/abs/2512.22469) • [📥 PDF](https://arxiv.org/pdf/2512.22469)

> Issues describe symptoms—the real buggy code is often hidden behind multi-hop dependencies. GraphLocator moves beyond relevance retrieval by explicitly modelling causal structure: decomposing sub-problems, capturing causal dependencies, and perfor...

</details>

<details>
<summary><b>6. CosineGate: Semantic Dynamic Routing via Cosine Incompatibility in Residual Networks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yogeswar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22206) • [📄 arXiv](https://arxiv.org/abs/2512.22206) • [📥 PDF](https://arxiv.org/pdf/2512.22206)

**💻 Code:** [⭐ Code](https://github.com/thotayogeswarreddy/CosineGate)

> "I introduce CosineGate, a SOTA dynamic routing mechanism for ResNets that uses the Cosine Incompatibility Ratio (CIR) as a self-supervised signal. 🚀 Why it matters: It matches ResNet-20 accuracy on CIFAR-10 while slashing computation by 28.5%—wit...

</details>

<details>
<summary><b>7. GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21008) • [📄 arXiv](https://arxiv.org/abs/2512.21008) • [📥 PDF](https://arxiv.org/pdf/2512.21008)

> Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 7 |
| 📅 Today | [`2026-01-01.json`](data/daily/2026-01-01.json) | 7 |
| 📆 This Week | [`2026-W00.json`](data/weekly/2026-W00.json) | 7 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 7 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-01 | 7 | [View JSON](data/daily/2026-01-01.json) |
| 📄 2025-12-31 | 31 | [View JSON](data/daily/2025-12-31.json) |
| 📄 2025-12-30 | 14 | [View JSON](data/daily/2025-12-30.json) |
| 📄 2025-12-29 | 7 | [View JSON](data/daily/2025-12-29.json) |
| 📄 2025-12-28 | 7 | [View JSON](data/daily/2025-12-28.json) |
| 📄 2025-12-27 | 7 | [View JSON](data/daily/2025-12-27.json) |
| 📄 2025-12-26 | 17 | [View JSON](data/daily/2025-12-26.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W00 | 7 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 7 | [View JSON](data/monthly/2026-01.json) |
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
