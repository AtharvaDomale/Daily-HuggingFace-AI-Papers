<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-8-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7836+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">69</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">108</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7836+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 06, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.00883) • [📄 arXiv](https://arxiv.org/abs/2611.00883) • [📥 PDF](https://arxiv.org/pdf/2611.00883)

**💻 Code:** [⭐ Code](https://github.com/George-Ling3/BestRand) • [⭐ Code](https://github.com/huggingface)

> Evaluating models on full benchmarks is often expensive. Benchmark compression reduces this cost by using small subsets to approximate full-benchmark results. This paper theoretically explains why random sampling works surprisingly well for benchm...

</details>

<details>
<summary><b>2. Towards Foundation Decision Models: Drex</b> ⭐ 88</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07854) • [📄 arXiv](https://arxiv.org/abs/2611.07854) • [📥 PDF](https://arxiv.org/pdf/2611.07854)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/nace-ai/drex-decision-models)

> We present Drex, a family of four open decision models that return a calibrated probability distribution over user-defined options in a single forward pass, without autoregressive decoding. Each request pairs a state (a document, record, or agent ...

</details>

<details>
<summary><b>3. EDGAR: Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07909) • [📄 arXiv](https://arxiv.org/abs/2611.07909) • [📥 PDF](https://arxiv.org/pdf/2611.07909)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kwanyun/edgar-code)

> Accepted to ACM Transactions on Graphics

</details>

<details>
<summary><b>4. VisSearch-Bench: Evaluating Long Horizon Visual Search in Multimodal Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04894) • [📄 arXiv](https://arxiv.org/abs/2611.04894) • [📥 PDF](https://arxiv.org/pdf/2611.04894)

**💻 Code:** [⭐ Code](https://github.com/antins-labs/VisSearch-Bench-v1) • [⭐ Code](https://github.com/huggingface)

> VisSearch-Bench evaluates how multimodal agents organize and use visual evidence during long-horizon web search. It contains 204 human-written tasks across five search categories, spanning 7 domains and 25 subdomains. The tasks require agents to c...

</details>

<details>
<summary><b>5. Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06817) • [📄 arXiv](https://arxiv.org/abs/2611.06817) • [📥 PDF](https://arxiv.org/pdf/2611.06817)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale)

> Hi everyone! We’re the authors of Open-HumanWAM , an open recipe for scaling human and robot experience for embodied foundation models. We study how human experience can improve robot learning, using Open-HumanWAM-50K , a 50,000-hour egocentric co...

</details>

<details>
<summary><b>6. Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04277) • [📄 arXiv](https://arxiv.org/abs/2611.04277) • [📥 PDF](https://arxiv.org/pdf/2611.04277)

**💻 Code:** [⭐ Code](https://github.com/DeepExperience/Harness-S) • [⭐ Code](https://github.com/huggingface)

> Harness evolution from a common base exhibits Locality, Complementarity, and Nonstationarity: variants repair different failures, and their utility changes as the policy learns. Rather than seeking a universally superior harness, we introduce Harn...

</details>

<details>
<summary><b>7. WeLM-Image: A Unified Image Generation Model towards Visual Intelligence</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hongyang Wei, Hao Lu, Guohui Zhang, Chen Li, Canmiao Fu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07392) • [📄 arXiv](https://arxiv.org/abs/2611.07392) • [📥 PDF](https://arxiv.org/pdf/2611.07392)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>8. NPCBench: A Clinical Apprenticeship Benchmark for Multimodal Care-Pathway Reasoning in Nasopharyngeal Carcinoma</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08369) • [📄 arXiv](https://arxiv.org/abs/2611.08369) • [📥 PDF](https://arxiv.org/pdf/2611.08369)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Current medical benchmarks typically break down clinical competence into isolated factual questions, single-image tasks, or static case vignettes. These tests score individual answers, but reveal less about whether a model can integrate subspecial...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 8 |
| 📅 Today | [`2026-11-06.json`](data/daily/2026-11-06.json) | 8 |
| 📆 This Week | [`2026-W44.json`](data/weekly/2026-W44.json) | 69 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 108 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-06 | 8 | [View JSON](data/daily/2026-11-06.json) |
| 📄 2026-11-05 | 13 | [View JSON](data/daily/2026-11-05.json) |
| 📄 2026-11-04 | 20 | [View JSON](data/daily/2026-11-04.json) |
| 📄 2026-11-03 | 14 | [View JSON](data/daily/2026-11-03.json) |
| 📄 2026-11-02 | 14 | [View JSON](data/daily/2026-11-02.json) |
| 📄 2026-11-01 | 39 | [View JSON](data/daily/2026-11-01.json) |
| 📄 2026-10-31 | 39 | [View JSON](data/daily/2026-10-31.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W44 | 69 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 108 | [View JSON](data/monthly/2026-11.json) |
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
