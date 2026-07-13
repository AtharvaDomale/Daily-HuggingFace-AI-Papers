<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-9-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5190+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">9</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">175</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5190+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** July 13, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. PanoWorld: Real-World Panoramic Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.09661) • [📄 arXiv](https://arxiv.org/abs/2607.09661) • [📥 PDF](https://arxiv.org/pdf/2607.09661)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Insta360-Research-Team/PanoWorld)

> Project page： https://lihaoy-ux.github.io/panoworld-page/ GitHub repo： https://github.com/Insta360-Research-Team/PanoWorld

</details>

<details>
<summary><b>2. Video Generation Models are General-Purpose Vision Learners</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.09024) • [📄 arXiv](https://arxiv.org/abs/2607.09024) • [📥 PDF](https://arxiv.org/pdf/2607.09024)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>3. From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yijiang Li, Dengyang Jiang, Haodong Li, Xin Lin, xmz111

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.06553) • [📄 arXiv](https://arxiv.org/abs/2607.06553) • [📥 PDF](https://arxiv.org/pdf/2607.06553)

**💻 Code:** [⭐ Code](https://github.com/xmz111/ReChannel) • [⭐ Code](https://github.com/huggingface)

> Pixel-Space Dense Prediction with Pretrained Text-to-Image Diffusion Models

</details>

<details>
<summary><b>4. Trust Region Policy Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.04751) • [📄 arXiv](https://arxiv.org/abs/2607.04751) • [📥 PDF](https://arxiv.org/pdf/2607.04751)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Improving the stability and performance of On-Policy Distillation (OPD) at no additional cost.

</details>

<details>
<summary><b>5. Phone Segmentation and Recognition through Phonological Activation Mapping</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Eunjung Yeo, Chin-Jou Li, Stephen McIntosh, Kwanghee Choi, Shikhar Bharadwaj

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.09020) • [📄 arXiv](https://arxiv.org/abs/2607.09020) • [📥 PDF](https://arxiv.org/pdf/2607.09020)

**💻 Code:** [⭐ Code](https://github.com/stephenmac7/phone-metrics) • [⭐ Code](https://github.com/huggingface)

> What it does : Solves phone segmentation and phone recognition jointly, instead of modeling them as separate tasks. Core idea : Phonetic structure is already latent inside self-supervised speech model (S3M) representations. We develop a method to ...

</details>

<details>
<summary><b>6. Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08393) • [📄 arXiv](https://arxiv.org/abs/2607.08393) • [📥 PDF](https://arxiv.org/pdf/2607.08393)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. Self-Guided Test-Time Training for Long-Context LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fei Tian, Yunchen Pu, Xiaohan Wei, Zhe Xu, Xinyu Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.09415) • [📄 arXiv](https://arxiv.org/abs/2607.09415) • [📥 PDF](https://arxiv.org/pdf/2607.09415)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>8. A Sovereign, Open-Source Foundation Model for German and English</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lukas Helff, Ruben Härle, David Fitzek, Benedikt Droste, The Soofi-Team

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.09424) • [📄 arXiv](https://arxiv.org/abs/2607.09424) • [📥 PDF](https://arxiv.org/pdf/2607.09424)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. Scalable Visual Pretraining for Language Intelligence</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tianyang Lin, Haiteng Zhao, Wenwei Zhang, Zhonghan Zhao, Yiming Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.09657) • [📄 arXiv](https://arxiv.org/abs/2607.09657) • [📥 PDF](https://arxiv.org/pdf/2607.09657)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 9 |
| 📅 Today | [`2026-07-13.json`](data/daily/2026-07-13.json) | 9 |
| 📆 This Week | [`2026-W28.json`](data/weekly/2026-W28.json) | 9 |
| 🗓️ This Month | [`2026-07.json`](data/monthly/2026-07.json) | 175 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-07-13 | 9 | [View JSON](data/daily/2026-07-13.json) |
| 📄 2026-07-12 | 23 | [View JSON](data/daily/2026-07-12.json) |
| 📄 2026-07-11 | 23 | [View JSON](data/daily/2026-07-11.json) |
| 📄 2026-07-10 | 8 | [View JSON](data/daily/2026-07-10.json) |
| 📄 2026-07-09 | 5 | [View JSON](data/daily/2026-07-09.json) |
| 📄 2026-07-08 | 8 | [View JSON](data/daily/2026-07-08.json) |
| 📄 2026-07-07 | 9 | [View JSON](data/daily/2026-07-07.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W28 | 9 | [View JSON](data/weekly/2026-W28.json) |
| 📅 2026-W27 | 79 | [View JSON](data/weekly/2026-W27.json) |
| 📅 2026-W26 | 117 | [View JSON](data/weekly/2026-W26.json) |
| 📅 2026-W25 | 108 | [View JSON](data/weekly/2026-W25.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-07 | 175 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |

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
