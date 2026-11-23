<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-12-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8203+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">12</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">475</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8203+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 23, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27193) • [📄 arXiv](https://arxiv.org/abs/2611.27193) • [📥 PDF](https://arxiv.org/pdf/2611.27193)

**💻 Code:** [⭐ Code](https://github.com/Newiz430/cmHC) • [⭐ Code](https://github.com/huggingface)

> Manifold-constrained Hyper-Connections (mHC) extends residual connections to multiple streams and stabilizes residual propagation by constraining stream-mixing residual matrices to the Birkhoff polytope, the set of doubly stochastic matrices. In t...

</details>

<details>
<summary><b>2. Non-penetrating World Models</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27963) • [📄 arXiv](https://arxiv.org/abs/2611.27963) • [📥 PDF](https://arxiv.org/pdf/2611.27963)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yuhangdai427/NoPeWorld)

> 🚧 World models can generate amazing worlds, but they still walk through walls! 👻 🎮 Today's video world models can generate impressive interactive environments, but they often fail to respect basic physical boundaries. For example, when you press W...

</details>

<details>
<summary><b>3. Clover-35B-A3B-Preview Technical Report</b> ⭐ 161</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.24636) • [📄 arXiv](https://arxiv.org/abs/2611.24636) • [📥 PDF](https://arxiv.org/pdf/2611.24636)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DiffuSpace/Clover-35B-A3B-Preview)

> Clover-35B-A3B-Preview Technical Report

</details>

<details>
<summary><b>4. Reading the Unreadable: Text-Aware Image Super-Resolution Needs Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.15449) • [📄 arXiv](https://arxiv.org/abs/2611.15449) • [📥 PDF](https://arxiv.org/pdf/2611.15449)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jasonleex1995/TAISR_needs_Reasoning)

> [TL;DR] Text-aware image super-resolution (TAISR) needs reasoning! Humans can often read badly degraded text by reasoning over context: a word repeated elsewhere in the image, neighboring numbers on a ruler, or a familiar logo. Super-resolution (S...

</details>

<details>
<summary><b>5. MemOnce: A Scalable Memory Architecture for Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27973) • [📄 arXiv](https://arxiv.org/abs/2611.27973) • [📥 PDF](https://arxiv.org/pdf/2611.27973)

**💻 Code:** [⭐ Code](https://github.com/Clare-Nie/MemOnce) • [⭐ Code](https://github.com/huggingface)

> MemOnce is a scalable foundation memory architecture for large language models that caches compact memory states once, retrieves relevant memory once per query, and decodes them into layer-specific KV caches on demand — eliminating multi-layer KV ...

</details>

<details>
<summary><b>6. Lost in the Hallway: Benchmarking Persistent Spatial Memory Across Revisits</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21485) • [📄 arXiv](https://arxiv.org/abs/2611.21485) • [📥 PDF](https://arxiv.org/pdf/2611.21485)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. AeroRSI: Recursive Self-Improvement for Aerial Vision-Language Navigation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.24077) • [📄 arXiv](https://arxiv.org/abs/2611.24077) • [📥 PDF](https://arxiv.org/pdf/2611.24077)

**💻 Code:** [⭐ Code](https://github.com/starsholic/AeroRSI) • [⭐ Code](https://github.com/huggingface)

> Hi everyone, we're the authors! AeroRSI lets a drone navigation policy fix its own failures: before each failure, it flies on from every candidate action, learns the ones it actually completes, and repeats on the failures that remain, since each r...

</details>

<details>
<summary><b>8. PURE: A Diagnostic Benchmark for Multi-Granularity Multimodal Identity Matching</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.22343) • [📄 arXiv](https://arxiv.org/abs/2611.22343) • [📥 PDF](https://arxiv.org/pdf/2611.22343)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/alibaba-multimodal-industrial-ai/PURE)

> Hi HF community! 👋 I’m one of the authors of PURE. Same design, different color: should two product listings count as a match? The answer depends on whether we care about the underlying product design or the specific variant. We introduce PURE , a...

</details>

<details>
<summary><b>9. AgentRoboCritic: Understanding the Capability of Frontier Agents on Direct Robot Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26613) • [📄 arXiv](https://arxiv.org/abs/2611.26613) • [📥 PDF](https://arxiv.org/pdf/2611.26613)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cxcscmu/AgentRoboCritic)

> 🤖 Can frontier agents control robots directly with low-level actions? And how do we actually tell how good they are? 📉 Existing evaluations focus on episode-level scores such as success rates. They hide the action-level feedback that is key to imp...

</details>

<details>
<summary><b>10. Prune to Expand: Training Prompt Enhancers with Less to Imitate</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27229) • [📄 arXiv](https://arxiv.org/abs/2611.27229) • [📥 PDF](https://arxiv.org/pdf/2611.27229)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We show that dense captions yielding larger gains in image generation do not necessarily provide better supervision for prompt enhancement, and introduce Prune to Expand (P2E), which removes sentences weakly related to user intent from training ca...

</details>

<details>
<summary><b>11. Explicit Style Extraction for Personalized Long-Form Text Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Abhinav Sharma, Anisa Dye, Amulya Rayasam, Sameeksha Kini, Jenitha Reddy Patel

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27211) • [📄 arXiv](https://arxiv.org/abs/2611.27211) • [📥 PDF](https://arxiv.org/pdf/2611.27211)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. Duopra: From Full-Duplex Conversation to Full-Duplex Agency</b> ⭐ 35</summary>

<br/>

**👥 Authors:** Can Liu, Xiaocheng Li, Yuezhao Ma, Pengzhou Cheng, Chechen123

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25773) • [📄 arXiv](https://arxiv.org/abs/2611.25773) • [📥 PDF](https://arxiv.org/pdf/2611.25773)

**💻 Code:** [⭐ Code](https://github.com/cyChen2003/duopra) • [⭐ Code](https://github.com/huggingface)

> We introduce Duopra , a full-duplex speech model that brings task agency into real-time conversational interaction. Duopra learns to invoke tools and delegate tasks while preserving natural turn-taking and interruption handling. On Full-Duplex-Ben...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 12 |
| 📅 Today | [`2026-11-23.json`](data/daily/2026-11-23.json) | 12 |
| 📆 This Week | [`2026-W47.json`](data/weekly/2026-W47.json) | 12 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 475 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-23 | 12 | [View JSON](data/daily/2026-11-23.json) |
| 📄 2026-11-22 | 50 | [View JSON](data/daily/2026-11-22.json) |
| 📄 2026-11-21 | 50 | [View JSON](data/daily/2026-11-21.json) |
| 📄 2026-11-20 | 18 | [View JSON](data/daily/2026-11-20.json) |
| 📄 2026-11-19 | 27 | [View JSON](data/daily/2026-11-19.json) |
| 📄 2026-11-18 | 17 | [View JSON](data/daily/2026-11-18.json) |
| 📄 2026-11-17 | 10 | [View JSON](data/daily/2026-11-17.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W47 | 12 | [View JSON](data/weekly/2026-W47.json) |
| 📅 2026-W46 | 184 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 475 | [View JSON](data/monthly/2026-11.json) |
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
