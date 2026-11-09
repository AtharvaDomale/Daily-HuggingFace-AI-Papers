<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-10-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7910+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">10</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">182</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7910+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 09, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Beneath the Surface: What We Overlook When We Talk About Agent Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07624) • [📄 arXiv](https://arxiv.org/abs/2611.07624) • [📥 PDF](https://arxiv.org/pdf/2611.07624)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/simple-agent-lab/RSIHub)

> Does a higher benchmark score really mean an agent is getting better? 🤔 In our paper, we go beyond surface-level benchmark evaluations to investigate what actually happens during agent evolution and why improvements do not always translate into be...

</details>

<details>
<summary><b>2. EvoWorld: Towards Out-of-Sight World Evolution in Generative Video Models</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Haoxuan Che, Xiaozhuan Liang, Zhixuan Chen, Dengyang Jiang, Tommy-Bie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06748) • [📄 arXiv](https://arxiv.org/abs/2611.06748) • [📥 PDF](https://arxiv.org/pdf/2611.06748)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tommy-Bie/EvoWorld)

> We are excited to introduce EvoWorld, a video generation framework for out-of-sight world evolution. EvoWorld combines a VLM brain with a visual renderer, enabling off-screen entities to remain active participants in an evolving world. Within a bi...

</details>

<details>
<summary><b>3. Video-Coder: Reinforcing Agentic Coding for Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.09894) • [📄 arXiv](https://arxiv.org/abs/2611.09894) • [📥 PDF](https://arxiv.org/pdf/2611.09894)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/CYWang735/Video-Coder)

> Recent years have witnessed remarkable progress in video generation in terms of visual quality and temporal consistency. However, generating videos with precise spatial structure, coherent motion dynamics, and controllable scene composition remain...

</details>

<details>
<summary><b>4. Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Chun-Mei Feng, Hongze Zhu, Wei Li, Cheng Tang, Junru Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.09382) • [📄 arXiv](https://arxiv.org/abs/2611.09382) • [📥 PDF](https://arxiv.org/pdf/2611.09382)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/WSP-Scope/WSP-Scope-Project)

> We systematically evaluate Weight-Space Programming through 16 editing trajectories across seven benchmarks and two model scales. Our results show uneven cross-task transfer, substantial variation across independent searches, and more limited gain...

</details>

<details>
<summary><b>5. Confucius4-R2T2: A Stable-Prefix Streaming ASR Model for Real Real-Time Transcription</b> ⭐ 868</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06043) • [📄 arXiv](https://arxiv.org/abs/2611.06043) • [📥 PDF](https://arxiv.org/pdf/2611.06043)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/netease-youdao/Confucius4-R2T2)

> Excited to share Confucius4-R2T2 , our stable-prefix streaming ASR model designed for real-time applications! Unlike conventional streaming ASR systems that repeatedly revise partial transcripts, R2T2 produces append-only, stable text that downstr...

</details>

<details>
<summary><b>6. AgentBaton: Efficient Model Collaboration in Agentic Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.35959) • [📄 arXiv](https://arxiv.org/abs/2610.35959) • [📥 PDF](https://arxiv.org/pdf/2610.35959)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://research.nvidia.com/labs/lpr/agentbaton/

</details>

<details>
<summary><b>7. Frontier3DBench: Benchmarking Coding Agents for 3D Creation across Objects, Scenes, and Interactive Open Worlds</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaoqiu Wang, John Lam, Wenyi Li, Qingle Liu, Xinjie Lin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.09017) • [📄 arXiv](https://arxiv.org/abs/2611.09017) • [📥 PDF](https://arxiv.org/pdf/2611.09017)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>8. OmniLatent: Unified Latent for 3D Part Modeling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Minxiu Wang, Jiachen Xu, Zeqiang Lai, Yunfei Zhao, Yunhan Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.09055) • [📄 arXiv](https://arxiv.org/abs/2611.09055) • [📥 PDF](https://arxiv.org/pdf/2611.09055)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaopeng Yuan, Peng Kuang, Haibo Jin, haohanw, YaoningYu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07461) • [📄 arXiv](https://arxiv.org/abs/2611.07461) • [📥 PDF](https://arxiv.org/pdf/2611.07461)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> How can AI agents reliably solve real-world tasks end to end? We introduce Agent-Native Primitives, reusable building blocks that equip tools, code, knowledge, and other resources with their own LLM-powered experts. Building on this idea, HEELO dy...

</details>

<details>
<summary><b>10. Sketchpad: certified exact arithmetic with a small vision-language model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** mohsenvand

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04434) • [📄 arXiv](https://arxiv.org/abs/2611.04434) • [📥 PDF](https://arxiv.org/pdf/2611.04434)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NeoVand/sketchpad)

> Author here. Can a small model be proven exact at arithmetic on inputs of every length? 🧮 The theory If each step of a model reads only a bounded view of an external memory, exactness at all lengths reduces to exactness on a finite set of views . ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 10 |
| 📅 Today | [`2026-11-09.json`](data/daily/2026-11-09.json) | 10 |
| 📆 This Week | [`2026-W45.json`](data/weekly/2026-W45.json) | 10 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 182 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-09 | 10 | [View JSON](data/daily/2026-11-09.json) |
| 📄 2026-11-08 | 32 | [View JSON](data/daily/2026-11-08.json) |
| 📄 2026-11-07 | 32 | [View JSON](data/daily/2026-11-07.json) |
| 📄 2026-11-06 | 8 | [View JSON](data/daily/2026-11-06.json) |
| 📄 2026-11-05 | 13 | [View JSON](data/daily/2026-11-05.json) |
| 📄 2026-11-04 | 20 | [View JSON](data/daily/2026-11-04.json) |
| 📄 2026-11-03 | 14 | [View JSON](data/daily/2026-11-03.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W45 | 10 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 182 | [View JSON](data/monthly/2026-11.json) |
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
