<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-7-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-772+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">34</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">34</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">772+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 03, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Improving Multi-step RAG with Hypergraph-based Memory for Long-Context Complex Relational Modeling</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23959) • [📄 arXiv](https://arxiv.org/abs/2512.23959) • [📥 PDF](https://arxiv.org/pdf/2512.23959)

**💻 Code:** [⭐ Code](https://github.com/Encyclomen/HGMem)

> Code released at: https://github.com/Encyclomen/HGMem

</details>

<details>
<summary><b>2. Dynamic Large Concept Models: Latent Reasoning in an Adaptive Semantic Space</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24617) • [📄 arXiv](https://arxiv.org/abs/2512.24617) • [📥 PDF](https://arxiv.org/pdf/2512.24617)

> Dynamic Large Concept Models (DLCM) introduce an end-to-end trained concept-level language modeling architecture that breaks the token-uniform computation paradigm in modern LLMs. Inspired by hierarchical models such as H-Net, DLCM learns semantic...

</details>

<details>
<summary><b>3. DiffThinker: Towards Generative Multimodal Reasoning with Diffusion Models</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Siyuan Huang, Yafu Li, Xiaoye Qu, Spico, yhx12

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24165) • [📄 arXiv](https://arxiv.org/abs/2512.24165) • [📥 PDF](https://arxiv.org/pdf/2512.24165)

**💻 Code:** [⭐ Code](https://github.com/lcqysl/DiffThinker)

> TLDR: A new paradigm for multi-modal reasoning with image-to-image generation. Diffusion could think too!

</details>

<details>
<summary><b>4. On the Role of Discreteness in Diffusion LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22630) • [📄 arXiv](https://arxiv.org/abs/2512.22630) • [📥 PDF](https://arxiv.org/pdf/2512.22630)

> TL;DR: We identify two core failure modes in current large diffusion LLMs: uniform corruption ignores where information lives in a sentence, and token-wise marginal training struggles with multi-token dependencies during parallel decoding.

</details>

<details>
<summary><b>5. FlowBlending: Stage-Aware Multi-Model Sampling for Fast and High-Fidelity Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Youngjung Uh, Jaeseok Jeong, Mingi Kwon, Jibin Song

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24724) • [📄 arXiv](https://arxiv.org/abs/2512.24724) • [📥 PDF](https://arxiv.org/pdf/2512.24724)

> No abstract available.

</details>

<details>
<summary><b>6. Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruohan Zhang, Li Fei-Fei, Jiajun Wu, Wenlong Huang, Karthik Dharmarajan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24766) • [📄 arXiv](https://arxiv.org/abs/2512.24766) • [📥 PDF](https://arxiv.org/pdf/2512.24766)

> No abstract available.

</details>

<details>
<summary><b>7. TESO Tabu Enhanced Simulation Optimization for Noisy Black Box Problems</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Ghaith Rabadi, Sean Mondesire, Bulent Soykan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24007) • [📄 arXiv](https://arxiv.org/abs/2512.24007) • [📥 PDF](https://arxiv.org/pdf/2512.24007)

**💻 Code:** [⭐ Code](https://github.com/bulentsoykan/TESO)

> Simulation optimization (SO) is frequently challenged by noisy evaluations, high computational costs, and complex, multimodal search landscapes. This paper introduces Tabu-Enhanced Simulation Optimization (TESO), a novel metaheuristic framework in...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 7 |
| 📅 Today | [`2026-01-03.json`](data/daily/2026-01-03.json) | 7 |
| 📆 This Week | [`2026-W00.json`](data/weekly/2026-W00.json) | 34 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 34 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-03 | 7 | [View JSON](data/daily/2026-01-03.json) |
| 📄 2026-01-02 | 20 | [View JSON](data/daily/2026-01-02.json) |
| 📄 2026-01-01 | 7 | [View JSON](data/daily/2026-01-01.json) |
| 📄 2025-12-31 | 31 | [View JSON](data/daily/2025-12-31.json) |
| 📄 2025-12-30 | 14 | [View JSON](data/daily/2025-12-30.json) |
| 📄 2025-12-29 | 7 | [View JSON](data/daily/2025-12-29.json) |
| 📄 2025-12-28 | 7 | [View JSON](data/daily/2025-12-28.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W00 | 34 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 34 | [View JSON](data/monthly/2026-01.json) |
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
