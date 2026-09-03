<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-12-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6163+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">47</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">35</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6163+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** September 03, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02886) • [📄 arXiv](https://arxiv.org/abs/2609.02886) • [📥 PDF](https://arxiv.org/pdf/2609.02886)

**💻 Code:** [⭐ Code](https://github.com/Junchao-cs/SolarWM) • [⭐ Code](https://github.com/huggingface)

> We present SolarWM , a fully open foundation for building interactive video world models from data preparation through scalable training and long-horizon inference. Open, reconfigurable data infrastructure. SolarWM converts 1.43 million canonical ...

</details>

<details>
<summary><b>2. It Takes Two to Match: Co-Evolving Generative Retriever with Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.00638) • [📄 arXiv](https://arxiv.org/abs/2609.00638) • [📥 PDF](https://arxiv.org/pdf/2609.00638)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Proud to share our work, CoGR (Co-evolving Generative Retrieval), where we train LLMs to generate retrieval keywords for both user queries and apps. CoGR follows a co-evolving training framework: we alternately optimize one side while keeping the ...

</details>

<details>
<summary><b>3. Cliff: Learning Process Rewards from the First Mistake</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Gerald Friedland, Jie Hao, Ketan Ramaneti, Runhui Wang, Peixuan Han

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02817) • [📄 arXiv](https://arxiv.org/abs/2609.02817) • [📥 PDF](https://arxiv.org/pdf/2609.02817)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A paper focusing on LLM RL and reward shaping

</details>

<details>
<summary><b>4. Language Models Can Control Their Own Attention</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02737) • [📄 arXiv](https://arxiv.org/abs/2609.02737) • [📥 PDF](https://arxiv.org/pdf/2609.02737)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Language models can control their own attention. Zero-shot evaluation of Gemma 4 31B shows a 52% reduction is global attention cost during decoding across 15 long-context benchmarks with 1.52pp accuracy drop.

</details>

<details>
<summary><b>5. On the Design Fundamentals of Pixel Text Representation Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01147) • [📄 arXiv](https://arxiv.org/abs/2609.01147) • [📥 PDF](https://arxiv.org/pdf/2609.01147)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> How do vision models learn to understand text directly from pixels? How can models pretrained on synthetic text rendered at ≤224×224 resolution generalize all the way to real-world 4K documents at test time? Pixel Linguist II answers these questio...

</details>

<details>
<summary><b>6. Influence-Directed Distillation: Solving the Diversity Bottleneck in Sampled-Token On-Policy Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.29846) • [📄 arXiv](https://arxiv.org/abs/2608.29846) • [📥 PDF](https://arxiv.org/pdf/2608.29846)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Sampled-token on-policy distillation (OPD) efficiently transfers capabilities from teacher to student using student-generated tokens, requiring teacher probabilities only for sampled tokens. Yet it frequently suffers from diversity distillation fa...

</details>

<details>
<summary><b>7. EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02783) • [📄 arXiv](https://arxiv.org/abs/2609.02783) • [📥 PDF](https://arxiv.org/pdf/2609.02783)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Your agent evaluation can be cheaper!

</details>

<details>
<summary><b>8. Post-Training Language Models for Gold-Medal Performance in Coding Competitions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02849) • [📄 arXiv](https://arxiv.org/abs/2609.02849) • [📥 PDF](https://arxiv.org/pdf/2609.02849)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. PaperCompiler: Faithful Paper-to-Code Generation via Repository-Level Specification Compilation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02272) • [📄 arXiv](https://arxiv.org/abs/2609.02272) • [📥 PDF](https://arxiv.org/pdf/2609.02272)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> .

</details>

<details>
<summary><b>10. ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01865) • [📄 arXiv](https://arxiv.org/abs/2609.01865) • [📥 PDF](https://arxiv.org/pdf/2609.01865)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Excited to share ExecRetrieval ! We ask a simple question: can code embeddings actually distinguish correct code from near-identical buggy code? Across 939 tasks and 24 retrieval systems, the best system reaches 100% exec@10 but only 33.1% exec@1 ...

</details>

<details>
<summary><b>11. Kirin: Animal Motion Generation from In-the-Wild Video</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shangzhe Wu, Jiajun Wu, James M. Rehg, Zhuoyang Pan, Brian Nlong Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01823) • [📄 arXiv](https://arxiv.org/abs/2609.01823) • [📥 PDF](https://arxiv.org/pdf/2609.01823)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. MULTI3IR: A Benchmark for Multi-perspective Multi-domain Multi-modal Information Retrieval</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.30949) • [📄 arXiv](https://arxiv.org/abs/2608.30949) • [📥 PDF](https://arxiv.org/pdf/2608.30949)

**💻 Code:** [⭐ Code](https://github.com/seokwon99/Multi3IR) • [⭐ Code](https://github.com/huggingface)

> EMNLP 2026; code is available at https://github.com/seokwon99/Multi3IR .

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 12 |
| 📅 Today | [`2026-09-03.json`](data/daily/2026-09-03.json) | 12 |
| 📆 This Week | [`2026-W35.json`](data/weekly/2026-W35.json) | 47 |
| 🗓️ This Month | [`2026-09.json`](data/monthly/2026-09.json) | 35 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-09-03 | 12 | [View JSON](data/daily/2026-09-03.json) |
| 📄 2026-09-02 | 16 | [View JSON](data/daily/2026-09-02.json) |
| 📄 2026-09-01 | 7 | [View JSON](data/daily/2026-09-01.json) |
| 📄 2026-08-31 | 12 | [View JSON](data/daily/2026-08-31.json) |
| 📄 2026-08-30 | 23 | [View JSON](data/daily/2026-08-30.json) |
| 📄 2026-08-29 | 23 | [View JSON](data/daily/2026-08-29.json) |
| 📄 2026-08-28 | 18 | [View JSON](data/daily/2026-08-28.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W35 | 47 | [View JSON](data/weekly/2026-W35.json) |
| 📅 2026-W34 | 173 | [View JSON](data/weekly/2026-W34.json) |
| 📅 2026-W33 | 213 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-09 | 35 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |

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
