<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-12-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3617+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">12</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">21</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">446</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3617+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 29, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Recursive Multi-Agent Systems</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.25917) • [📄 arXiv](https://arxiv.org/abs/2604.25917) • [📥 PDF](https://arxiv.org/pdf/2604.25917)

**💻 Code:** [⭐ Code](https://github.com/RecursiveMAS/RecursiveMAS)

> Code and Data are Available at https://recursivemas.github.io

</details>

<details>
<summary><b>2. Meta-CoT: Enhancing Granularity and Generalization in Image Editing</b> ⭐ 47</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24625) • [📄 arXiv](https://arxiv.org/abs/2604.24625) • [📥 PDF](https://arxiv.org/pdf/2604.24625)

**💻 Code:** [⭐ Code](https://github.com/shiyi-zh0408/Meta-CoT)

> Paper: https://arxiv.org/abs/2604.24625 Project Page: https://shiyi-zh0408.github.io/projectpages/Meta-CoT/ Github: https://github.com/shiyi-zh0408/Meta-CoT Model: https://huggingface.co/shiyi0408/Meta-CoT Benchmark: https://huggingface.co/dataset...

</details>

<details>
<summary><b>3. Programming with Data: Test-Driven Data Engineering for Self-Improving LLMs from Raw Corpora</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24819) • [📄 arXiv](https://arxiv.org/abs/2604.24819) • [📥 PDF](https://arxiv.org/pdf/2604.24819)

**💻 Code:** [⭐ Code](https://github.com/OpenRaiser/ProDa)

> Reliably transferring specialized human knowledge from text into large language models remains a fundamental challenge in artificial intelligence. Fine-tuning on domain corpora has enabled substantial capability gains, but the process operates wit...

</details>

<details>
<summary><b>4. Refinement via Regeneration: Enlarging Modification Space Boosts Image Refinement in Unified Multimodal Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.25636) • [📄 arXiv](https://arxiv.org/abs/2604.25636) • [📥 PDF](https://arxiv.org/pdf/2604.25636)

**💻 Code:** [⭐ Code](https://github.com/LeapLabTHU/RvR)

> 🔥 Regeneration over editing: unlocking more effective image refinement! We present Refinement via Regeneration (RvR), a novel framework that reformulates image refinement in unified multimodal models from an editing-based paradigm to a regeneratio...

</details>

<details>
<summary><b>5. Mutual Forcing: Dual-Mode Self-Evolution for Fast Autoregressive Audio-Video Character Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yupeng Shi, Jiabao Wang, Zhifan Wu, Lianghua Huang, Yupeng Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.25819) • [📄 arXiv](https://arxiv.org/abs/2604.25819) • [📥 PDF](https://arxiv.org/pdf/2604.25819)

**💻 Code:** [⭐ Code](https://github.com/HVision-NKU/MutualForcing)

> Recent fast streaming generation methods often rely on a complicated pipeline, typically starting from a non-causal bidirectional diffusion model and requiring additional steps such as ODE initialization and DMD distillation. Mutual Forcing explor...

</details>

<details>
<summary><b>6. DV-World: Benchmarking Data Visualization Agents in Real-World Scenarios</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haoxiang Liu, Jingyu Guo, Fangyu Lei, Shaoping Huang, Jinxiang Meng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.25914) • [📄 arXiv](https://arxiv.org/abs/2604.25914) • [📥 PDF](https://arxiv.org/pdf/2604.25914)

> No abstract available.

</details>

<details>
<summary><b>7. Co-Director: Agentic Generative Video Storytelling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24842) • [📄 arXiv](https://arxiv.org/abs/2604.24842) • [📥 PDF](https://arxiv.org/pdf/2604.24842)

> No abstract available.

</details>

<details>
<summary><b>8. IAM: Identity-Aware Human Motion and Shape Joint Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chuan Guo, Chengcheng Tang, Abhay Mittal, Zekun Li, Wenqi Jia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.25164) • [📄 arXiv](https://arxiv.org/abs/2604.25164) • [📥 PDF](https://arxiv.org/pdf/2604.25164)

> No abstract available.

</details>

<details>
<summary><b>9. Toward Scalable Terminal Task Synthesis via Skill Graphs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.25727) • [📄 arXiv](https://arxiv.org/abs/2604.25727) • [📥 PDF](https://arxiv.org/pdf/2604.25727)

> No abstract available.

</details>

<details>
<summary><b>10. BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.25203) • [📄 arXiv](https://arxiv.org/abs/2604.25203) • [📥 PDF](https://arxiv.org/pdf/2604.25203)

**💻 Code:** [⭐ Code](https://github.com/plurai-ai/BARRED)

> The BARRED dataset can be found here: https://huggingface.co/datasets/Plurai/BARRED

</details>

<details>
<summary><b>11. A Systematic Post-Train Framework for Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haoran Li, Shuai Lu, Jie Huang, Siming Fu, Zeyue Xue

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.25427) • [📄 arXiv](https://arxiv.org/abs/2604.25427) • [📥 PDF](https://arxiv.org/pdf/2604.25427)

> No abstract available.

</details>

<details>
<summary><b>12. AutoGUI-v2: A Comprehensive Multi-Modal GUI Functionality Understanding Benchmark</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuntao Chen, Zheng Ju, Jingran Su, Xiping Wang, Hongxin Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24441) • [📄 arXiv](https://arxiv.org/abs/2604.24441) • [📥 PDF](https://arxiv.org/pdf/2604.24441)

**💻 Code:** [⭐ Code](https://github.com/ZJULiHongxin/AutoGUI-v2)

> AutoGUI-v2 introduces comprehensive functional GUI understanding : rather than simply detecting what elements look like, it asks what they do — and whether models can tell apart elements that look the same but behave differently. Background: GUIs ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 12 |
| 📅 Today | [`2026-04-29.json`](data/daily/2026-04-29.json) | 12 |
| 📆 This Week | [`2026-W17.json`](data/weekly/2026-W17.json) | 21 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 446 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-29 | 12 | [View JSON](data/daily/2026-04-29.json) |
| 📄 2026-04-28 | 3 | [View JSON](data/daily/2026-04-28.json) |
| 📄 2026-04-27 | 6 | [View JSON](data/daily/2026-04-27.json) |
| 📄 2026-04-26 | 22 | [View JSON](data/daily/2026-04-26.json) |
| 📄 2026-04-25 | 22 | [View JSON](data/daily/2026-04-25.json) |
| 📄 2026-04-24 | 5 | [View JSON](data/daily/2026-04-24.json) |
| 📄 2026-04-23 | 8 | [View JSON](data/daily/2026-04-23.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W17 | 21 | [View JSON](data/weekly/2026-W17.json) |
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |
| 📅 2026-W15 | 99 | [View JSON](data/weekly/2026-W15.json) |
| 📅 2026-W14 | 140 | [View JSON](data/weekly/2026-W14.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 446 | [View JSON](data/monthly/2026-04.json) |
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
