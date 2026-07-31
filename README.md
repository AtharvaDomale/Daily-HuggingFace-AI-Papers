<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-10-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5381+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">36</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">366</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5381+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** July 31, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28618) • [📄 arXiv](https://arxiv.org/abs/2607.28618) • [📥 PDF](https://arxiv.org/pdf/2607.28618)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/bingyan4science/askchem)

> Working on chemistry? What if you could search chemistry findings instead of papers? We turned 147,000 papers into 2.4 million searchable claims , making it possible to find results, compare evidence, and surface contradictions. Try AskChem: 🔎 htt...

</details>

<details>
<summary><b>2. SpatialCLI: Learning to Reason With Spatial Tools, Then Without Them</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Chen Zhang, Zhuo Yang, Sunzhu Li, Zixuan Huang, Yang Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27703) • [📄 arXiv](https://arxiv.org/abs/2607.27703) • [📥 PDF](https://arxiv.org/pdf/2607.27703)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/IANNXANG/SpatialCLI)

> We introduce SpatialCLI, a framework that teaches vision-language models to reason with spatial tools and internalize their capabilities for tool-free inference.

</details>

<details>
<summary><b>3. PhiZero: A World Model Built Around Physical Language</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Tieniu Tan, Xu Chen, Ruopeng Gao, Yuqi Wang, Shuyao Shang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28624) • [📄 arXiv](https://arxiv.org/abs/2607.28624) • [📥 PDF](https://arxiv.org/pdf/2607.28624)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yaoyao-jpg/PhiZero)

> No abstract available.

</details>

<details>
<summary><b>4. MemHarness: Memory Is Reconstructed, Not Replayed</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28272) • [📄 arXiv](https://arxiv.org/abs/2607.28272) • [📥 PDF](https://arxiv.org/pdf/2607.28272)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/KnowledgeXLab/MemHarness)

> Most memory-augmented LLM agents follow a retrieve-and-replay paradigm, directly inserting retrieved trajectories into the context. This conflates retrieval relevance with action-level applicability: a memory may be semantically related yet inappr...

</details>

<details>
<summary><b>5. Metis: Memory Foundation Model</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26760) • [📄 arXiv](https://arxiv.org/abs/2607.26760) • [📥 PDF](https://arxiv.org/pdf/2607.26760)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MemTensor/Metis)

> Metis : The first prototype of a memory foundation model, equipping foundation models with a persistent and dynamically evolving native memory state. What if memory were a native capability of foundation models, rather than an external module? Rec...

</details>

<details>
<summary><b>6. Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28568) • [📄 arXiv](https://arxiv.org/abs/2607.28568) • [📥 PDF](https://arxiv.org/pdf/2607.28568)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/FrontisAI/OpenRSI)

> Recursive self-improvement (RSI) requires AI systems that improve the process of building AI (i.e., AI4AI); machine learning engineering (MLE) offers a concrete, executable testbed for studying this capability. We introduce OpenMLE, an open full-s...

</details>

<details>
<summary><b>7. VideoCoCo: Code-as-CoT for Physically-Consistent Video Generation via an Agentic Dual-Engine System</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhen Fang, Chunmei Qing, Xiaoxiao Ma, Tianfei Ren, Haodong Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27380) • [📄 arXiv](https://arxiv.org/abs/2607.27380) • [📥 PDF](https://arxiv.org/pdf/2607.27380)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🧠 VideoCoCo uses executable Blender code as process-level CoT✨, generating deterministic spatiotemporal drafts that guide a video editor toward photorealistic and physically consistent results. 🎬

</details>

<details>
<summary><b>8. ReToken: One Token to Improve Vision-Language Models for Visual Retrieval</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jianfeng Gao, Yuqun Wu, Zhen Zhu, Reuben Tan, Yao Xiao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28627) • [📄 arXiv](https://arxiv.org/abs/2607.28627) • [📥 PDF](https://arxiv.org/pdf/2607.28627)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/avaxiao/ReToken)

> No abstract available.

</details>

<details>
<summary><b>9. Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28611) • [📄 arXiv](https://arxiv.org/abs/2607.28611) • [📥 PDF](https://arxiv.org/pdf/2607.28611)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yinghao Liu, Runmao Yao, Beichen Wen, Haozhe Xie, Yukang Cao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28625) • [📄 arXiv](https://arxiv.org/abs/2607.28625) • [📥 PDF](https://arxiv.org/pdf/2607.28625)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 10 |
| 📅 Today | [`2026-07-31.json`](data/daily/2026-07-31.json) | 10 |
| 📆 This Week | [`2026-W30.json`](data/weekly/2026-W30.json) | 36 |
| 🗓️ This Month | [`2026-07.json`](data/monthly/2026-07.json) | 366 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-07-31 | 10 | [View JSON](data/daily/2026-07-31.json) |
| 📄 2026-07-30 | 6 | [View JSON](data/daily/2026-07-30.json) |
| 📄 2026-07-29 | 8 | [View JSON](data/daily/2026-07-29.json) |
| 📄 2026-07-28 | 4 | [View JSON](data/daily/2026-07-28.json) |
| 📄 2026-07-27 | 8 | [View JSON](data/daily/2026-07-27.json) |
| 📄 2026-07-26 | 22 | [View JSON](data/daily/2026-07-26.json) |
| 📄 2026-07-25 | 22 | [View JSON](data/daily/2026-07-25.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W30 | 36 | [View JSON](data/weekly/2026-W30.json) |
| 📅 2026-W29 | 70 | [View JSON](data/weekly/2026-W29.json) |
| 📅 2026-W28 | 94 | [View JSON](data/weekly/2026-W28.json) |
| 📅 2026-W27 | 79 | [View JSON](data/weekly/2026-W27.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
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
