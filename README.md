<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-6-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3110+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">62</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">543</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3110+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 27, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. MACRO: Advancing Multi-Reference Image Generation with Structured Long-Context Data</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25319) • [📄 arXiv](https://arxiv.org/abs/2603.25319) • [📥 PDF](https://arxiv.org/pdf/2603.25319)

**💻 Code:** [⭐ Code](https://github.com/HKU-MMLab/Macro)

> We present MACRO, a large-scale multi-reference image generation dataset MacroData with 400K samples and the corresponding multi-image generation metric MacroBench. Our dataset supports the input of up to 10 reference maps, covering the four long-...

</details>

<details>
<summary><b>2. SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24755) • [📄 arXiv](https://arxiv.org/abs/2603.24755) • [📥 PDF](https://arxiv.org/pdf/2603.24755)

**💻 Code:** [⭐ Code](https://github.com/SprocketLab/slop-code-bench)

> Coding benchmarks tell you if an agent passed the tests. They don't tell you if the code is any good, or if it's going to be workable three features from now. SlopCodeBench makes agents extend their own solutions across multiple steps without pres...

</details>

<details>
<summary><b>3. MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens</b> ⭐ 2.21k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.23516) • [📄 arXiv](https://arxiv.org/abs/2603.23516) • [📥 PDF](https://arxiv.org/pdf/2603.23516)

**💻 Code:** [⭐ Code](https://github.com/EverMind-AI/MSA)

> 📝 Abstract Long-term memory is essential for general intelligence, yet the full attention bottleneck constrains most LLMs’ effective context length to 128K–1M . Existing attempts，hybrid linear attention, fixed-size state memory (e.g., RNNs), and e...

</details>

<details>
<summary><b>4. Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25745) • [📄 arXiv](https://arxiv.org/abs/2603.25745) • [📥 PDF](https://arxiv.org/pdf/2603.25745)

> No abstract available.

</details>

<details>
<summary><b>5. Vega: Learning to Drive with Natural Language Instructions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jie Zhou, Zheng Zhu, Wenzhao Zheng, Yuxuan Li, Sicheng Zuo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25741) • [📄 arXiv](https://arxiv.org/abs/2603.25741) • [📥 PDF](https://arxiv.org/pdf/2603.25741)

> No abstract available.

</details>

<details>
<summary><b>6. Voxtral TTS</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25551) • [📄 arXiv](https://arxiv.org/abs/2603.25551) • [📥 PDF](https://arxiv.org/pdf/2603.25551)

> Voxtral TTS is a multilingual expressive TTS with a hybrid autoregressive semantic token generator and flow-matching acoustic tokens, using Voxtral Codec for high-quality voice cloning from 3 seconds of audio.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 6 |
| 📅 Today | [`2026-03-27.json`](data/daily/2026-03-27.json) | 6 |
| 📆 This Week | [`2026-W12.json`](data/weekly/2026-W12.json) | 62 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 543 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-27 | 6 | [View JSON](data/daily/2026-03-27.json) |
| 📄 2026-03-26 | 4 | [View JSON](data/daily/2026-03-26.json) |
| 📄 2026-03-25 | 11 | [View JSON](data/daily/2026-03-25.json) |
| 📄 2026-03-24 | 37 | [View JSON](data/daily/2026-03-24.json) |
| 📄 2026-03-23 | 4 | [View JSON](data/daily/2026-03-23.json) |
| 📄 2026-03-22 | 32 | [View JSON](data/daily/2026-03-22.json) |
| 📄 2026-03-21 | 32 | [View JSON](data/daily/2026-03-21.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W12 | 62 | [View JSON](data/weekly/2026-W12.json) |
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |
| 📅 2026-W10 | 119 | [View JSON](data/weekly/2026-W10.json) |
| 📅 2026-W09 | 201 | [View JSON](data/weekly/2026-W09.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 543 | [View JSON](data/monthly/2026-03.json) |
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
