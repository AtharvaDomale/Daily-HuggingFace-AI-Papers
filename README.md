<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-7-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-686+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">132</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">735</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">686+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 28, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Latent Implicit Visual Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21218) • [📄 arXiv](https://arxiv.org/abs/2512.21218) • [📥 PDF](https://arxiv.org/pdf/2512.21218)

> TL;DR: We introduce a new method that improves visual reasoning by allowing models to implicitly learn latent visual representations, without requiring explicit supervision or additional data for these latents.

</details>

<details>
<summary><b>2. Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20605) • [📄 arXiv](https://arxiv.org/abs/2512.20605) • [📥 PDF](https://arxiv.org/pdf/2512.20605)

> TLDR: This work reveals that autoregressive models inherently learn linearly controllable, temporally abstract action representations within their residual streams, which can be activated and composed to execute long-horizon behaviors. We leverage...

</details>

<details>
<summary><b>3. Spatia: Video Generation with Updatable Spatial Memory</b> ⭐ 61</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15716) • [📄 arXiv](https://arxiv.org/abs/2512.15716) • [📥 PDF](https://arxiv.org/pdf/2512.15716)

**💻 Code:** [⭐ Code](https://github.com/ZhaoJingjing713/Spatia)

> Existing video generation models struggle to maintain long-term spatial and temporal consistency due to the dense, high-dimensional nature of video signals. To overcome this limitation, we propose Spatia, a spatial memory-aware video generation fr...

</details>

<details>
<summary><b>4. Schoenfeld's Anatomy of Mathematical Reasoning by Language Models</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Tianyi Zhou, Soheil Feizi, Yize Cheng, Chenrui Fan, Ming Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19995) • [📄 arXiv](https://arxiv.org/abs/2512.19995) • [📥 PDF](https://arxiv.org/pdf/2512.19995)

**💻 Code:** [⭐ Code](https://github.com/MingLiiii/ThinkARM)

> We extend a cognitive science-inspired episode annotation framework to an automatic, scalable, sentence-level representation that supports large-scale analysis of reasoning traces and conduct a systematic study of reasoning dynamics across a diver...

</details>

<details>
<summary><b>5. How Much 3D Do Video Foundation Models Encode?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19949) • [📄 arXiv](https://arxiv.org/abs/2512.19949) • [📥 PDF](https://arxiv.org/pdf/2512.19949)

> After training on large 2D videos, will video foundation models naturally encode 3D structure and ego-motion? Our study reveals that state-of-the-art video generators develop strong, generalizable 3D understanding even compared to 3D experts, desp...

</details>

<details>
<summary><b>6. VA-π: Variational Policy Alignment for Pixel-Aware Autoregressive Generation</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Yicong Li, Xiaoye Qu, Kai Xu, Qiyuan He, Xinyao Liao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19680) • [📄 arXiv](https://arxiv.org/abs/2512.19680) • [📥 PDF](https://arxiv.org/pdf/2512.19680)

**💻 Code:** [⭐ Code](https://github.com/Lil-Shake/VA-Pi)

> Autoregressive (AR) visual generation relies on tokenizers to map images to and from discrete sequences. However, tokenizers are trained to reconstruct clean images from ground-truth tokens, while AR generators are optimized only for token likelih...

</details>

<details>
<summary><b>7. GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuanchun Shi, Junliang Xing, Changhao Zhang, Yijun Yang, Tong Wei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13043) • [📄 arXiv](https://arxiv.org/abs/2512.13043) • [📥 PDF](https://arxiv.org/pdf/2512.13043)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 7 |
| 📅 Today | [`2025-12-28.json`](data/daily/2025-12-28.json) | 7 |
| 📆 This Week | [`2025-W51.json`](data/weekly/2025-W51.json) | 132 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 735 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-28 | 7 | [View JSON](data/daily/2025-12-28.json) |
| 📄 2025-12-27 | 7 | [View JSON](data/daily/2025-12-27.json) |
| 📄 2025-12-26 | 17 | [View JSON](data/daily/2025-12-26.json) |
| 📄 2025-12-25 | 18 | [View JSON](data/daily/2025-12-25.json) |
| 📄 2025-12-24 | 23 | [View JSON](data/daily/2025-12-24.json) |
| 📄 2025-12-23 | 22 | [View JSON](data/daily/2025-12-23.json) |
| 📄 2025-12-22 | 38 | [View JSON](data/daily/2025-12-22.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 735 | [View JSON](data/monthly/2025-12.json) |

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
