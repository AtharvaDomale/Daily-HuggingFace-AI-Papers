<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-14-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-707+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">14</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">21</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">756</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">707+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 30, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. InsertAnywhere: Bridging 4D Scene Geometry and Diffusion Models for Realistic Video Object Insertion</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17504) • [📄 arXiv](https://arxiv.org/abs/2512.17504) • [📥 PDF](https://arxiv.org/pdf/2512.17504)

**💻 Code:** [⭐ Code](https://github.com/myyzzzoooo/InsertAnywhere)

> No abstract available.

</details>

<details>
<summary><b>2. Mindscape-Aware Retrieval Augmented Generation for Improved Long Context Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17220) • [📄 arXiv](https://arxiv.org/abs/2512.17220) • [📥 PDF](https://arxiv.org/pdf/2512.17220)

> Our trained models can be downloaded from: https://huggingface.co/MindscapeRAG

</details>

<details>
<summary><b>3. MAI-UI Technical Report: Real-World Centric Foundation GUI Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22047) • [📄 arXiv](https://arxiv.org/abs/2512.22047) • [📥 PDF](https://arxiv.org/pdf/2512.22047)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Step-GUI Technical Report (2025) GUI-360\°: A Comprehensive Dataset and Ben...

</details>

<details>
<summary><b>4. UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture</b> ⭐ 27</summary>

<br/>

**👥 Authors:** Kaiwen Zhu, Xiaohui Li, Jiayang Li, Shuo Cao, Andrew613

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21675) • [📄 arXiv](https://arxiv.org/abs/2512.21675) • [📥 PDF](https://arxiv.org/pdf/2512.21675)

**💻 Code:** [⭐ Code](https://github.com/thunderbolt215/UniPercept)

> Unipercept

</details>

<details>
<summary><b>5. ProEdit: Inversion-based Editing From Prompts Done Right</b> ⭐ 24</summary>

<br/>

**👥 Authors:** Kun-Yu Lin, Jian-Jian Jiang, Xiao-Ming Wu, Zhi Ouyang, zhengli1013

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22118) • [📄 arXiv](https://arxiv.org/abs/2512.22118) • [📥 PDF](https://arxiv.org/pdf/2512.22118)

**💻 Code:** [⭐ Code](https://github.com/iSEE-Laboratory/ProEdit)

> Project page: https://isee-laboratory.github.io/ProEdit

</details>

<details>
<summary><b>6. TimeBill: Time-Budgeted Inference for Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yehan Ma, An Zou, fanqiNO1

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21859) • [📄 arXiv](https://arxiv.org/abs/2512.21859) • [📥 PDF](https://arxiv.org/pdf/2512.21859)

> 🚀 Large language models can infer within strict time budgets! 📉 Fixed KV cache eviction or naive speed-up strategies hurt performance under real-time constraints. 🎯 TimeBill enables adaptive, time-aware LLM inference by predicting response length ...

</details>

<details>
<summary><b>7. See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22120) • [📄 arXiv](https://arxiv.org/abs/2512.22120) • [📥 PDF](https://arxiv.org/pdf/2512.22120)

**💻 Code:** [⭐ Code](https://github.com/zss02/BiPS)

> A framework that leverages programmatically generated paired views to train VLMs to focus on critical visual evidence while rejecting text-only shortcuts.

</details>

<details>
<summary><b>8. Omni-Weather: Unified Multimodal Foundation Model for Weather Generation and Understanding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yixin Chen, Yidi Liu, Xuming He, Zhiwang Zhou, Andrew613

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21643) • [📄 arXiv](https://arxiv.org/abs/2512.21643) • [📥 PDF](https://arxiv.org/pdf/2512.21643)

> Submit Omni-Weather

</details>

<details>
<summary><b>9. InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Jierun Chen, Tiezheng Yu, Jiannan Wu, Lewei Yao, m-Just

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18745) • [📄 arXiv](https://arxiv.org/abs/2512.18745) • [📥 PDF](https://arxiv.org/pdf/2512.18745)

**💻 Code:** [⭐ Code](https://github.com/m-Just/InSight-o3)

> Check out O3-Bench at https://huggingface.co/datasets/m-Just/O3-Bench !

</details>

<details>
<summary><b>10. SWE-RM: Execution-free Feedback For Software Engineering Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** X. W., Lei Zhang, Jiawei Chen, Binyuan Hui, KaShun Shum

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21919) • [📄 arXiv](https://arxiv.org/abs/2512.21919) • [📥 PDF](https://arxiv.org/pdf/2512.21919)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Training Versatile Coding Agents in Synthetic Environments (2025) Klear-Age...

</details>

<details>
<summary><b>11. SVBench: Evaluation of Video Generation Models on Social Reasoning</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Xiaojie Xu, Chuanhao Li, Tianmeng Yang, Gongxuan Wang, Wenshuo Peng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21507) • [📄 arXiv](https://arxiv.org/abs/2512.21507) • [📥 PDF](https://arxiv.org/pdf/2512.21507)

**💻 Code:** [⭐ Code](https://github.com/Gloria2tt/SVBench-Evaluation)

> Currently, most of the work focuses on discussing the physical plausibility of the videos; we need more research to examine whether the actions themselves are inherently reasonable. Our project page is available https://github.com/Gloria2tt/SVBenc...

</details>

<details>
<summary><b>12. SlideTailor: Personalized Presentation Slide Generation for Scientific Papers</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20292) • [📄 arXiv](https://arxiv.org/abs/2512.20292) • [📥 PDF](https://arxiv.org/pdf/2512.20292)

**💻 Code:** [⭐ Code](https://github.com/nusnlp/SlideTailor)

> 🔆 Overview We argue that presentation design is inherently subjective. Users have different preferences in terms of narrative structure, emphasis, conciseness, aesthetic choices, etc. So in this work, we ask: Can we better model such diverse user ...

</details>

<details>
<summary><b>13. A 58-Addition, Rank-23 Scheme for General 3x3 Matrix Multiplication</b> ⭐ 0</summary>

<br/>

**👥 Authors:** dronperminov

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21980) • [📄 arXiv](https://arxiv.org/abs/2512.21980) • [📥 PDF](https://arxiv.org/pdf/2512.21980)

**💻 Code:** [⭐ Code](https://github.com/dronperminov/ternary_flip_graph)

> A 58-addition, rank-23 scheme for exact 3×3 matrix multiplication sets a new SOTA. This improves the previous best of 60 additions without basis change. The scheme uses only ternary coefficients {-1,0,1} and was discovered via combinatorial flip-g...

</details>

<details>
<summary><b>14. Rethinking Sample Polarity in Reinforcement Learning with Verifiable Rewards</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhenduo Zhang, Wayne Xin Zhao, Zhixun Li, Yuliang Zhan, Xinyu Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21625) • [📄 arXiv](https://arxiv.org/abs/2512.21625) • [📥 PDF](https://arxiv.org/pdf/2512.21625)

> Large reasoning models (LRMs) are typically trained using reinforcement learning with verifiable reward (RLVR) to enhance their reasoning abilities. In this paradigm, policies are updated using both positive and negative self-generated rollouts, w...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 14 |
| 📅 Today | [`2025-12-30.json`](data/daily/2025-12-30.json) | 14 |
| 📆 This Week | [`2025-W52.json`](data/weekly/2025-W52.json) | 21 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 756 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-30 | 14 | [View JSON](data/daily/2025-12-30.json) |
| 📄 2025-12-29 | 7 | [View JSON](data/daily/2025-12-29.json) |
| 📄 2025-12-28 | 7 | [View JSON](data/daily/2025-12-28.json) |
| 📄 2025-12-27 | 7 | [View JSON](data/daily/2025-12-27.json) |
| 📄 2025-12-26 | 17 | [View JSON](data/daily/2025-12-26.json) |
| 📄 2025-12-25 | 18 | [View JSON](data/daily/2025-12-25.json) |
| 📄 2025-12-24 | 23 | [View JSON](data/daily/2025-12-24.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W52 | 21 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 756 | [View JSON](data/monthly/2025-12.json) |

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
