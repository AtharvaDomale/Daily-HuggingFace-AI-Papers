<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-16-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4809+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">16</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">62</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">406</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4809+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 19, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Beyond Static Leaderboards: Predictive Validity for the Evaluation of LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19704) • [📄 arXiv](https://arxiv.org/abs/2606.19704) • [📥 PDF](https://arxiv.org/pdf/2606.19704)

**💻 Code:** [⭐ Code](https://github.com/IBM/AssetOpsBench) • [⭐ Code](https://github.com/huggingface)

> Agent benchmarks are growing fast, but no single benchmark touches more than four or five of the dimensions that deployment exposes. This paper aggregates the largest coordinated deep-dive of one MCP-based industrial-agent benchmark to date: fourt...

</details>

<details>
<summary><b>2. S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.20515) • [📄 arXiv](https://arxiv.org/abs/2606.20515) • [📥 PDF](https://arxiv.org/pdf/2606.20515)

**💻 Code:** [⭐ Code](https://github.com/Ropedia/S-Agent) • [⭐ Code](https://github.com/huggingface)

> The first agentic model for Spatial Intelligence. S-Agent turns perception into action: grounding, reconstructing, and reasoning with tools to solve complex spatial tasks step by step.

</details>

<details>
<summary><b>3. Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance</b> ⭐ 31</summary>

<br/>

**👥 Authors:** Xiaoxin Chen, Xiaohu Ruan, Wenyu Liu, Ziyang Xu, Kangsheng Duan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19195) • [📄 arXiv](https://arxiv.org/abs/2606.19195) • [📥 PDF](https://arxiv.org/pdf/2606.19195)

**💻 Code:** [⭐ Code](https://github.com/hustvl/PixelHacker) • [⭐ Code](https://github.com/hustvl/Moebius) • [⭐ Code](https://github.com/huggingface)

> Moebius is our latest AI Image Inpainting endeavor, serving as a direct continuation of our previous work, PixelHacker. Named after the concepts of "infinity" and "master painter," Moebius embodies our vision: maintaining exceptional generation qu...

</details>

<details>
<summary><b>4. FreeStyle: Free Control of Style-Content Dual-Reference Generation from Community LoRA Mining</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.20506) • [📄 arXiv](https://arxiv.org/abs/2606.20506) • [📥 PDF](https://arxiv.org/pdf/2606.20506)

**💻 Code:** [⭐ Code](https://github.com/Blue2Giant/FreeStyle) • [⭐ Code](https://github.com/huggingface)

> 📚 Resources 📄 Paper: https://arxiv.org/pdf/2606.20506 🌐 Project Page: https://blue2giant.github.io/FreeStyle/ 💻 GitHub: https://github.com/Blue2Giant/FreeStyle 📦 Dataset: https://huggingface.co/datasets/Blue2Giant/FreeStyle_Dataset ⚖️ Model Weight...

</details>

<details>
<summary><b>5. JanusMesh: Fast and Zero-Shot 3D Visual Illusion Generation via Cross-Space Denoising</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.20563) • [📄 arXiv](https://arxiv.org/abs/2606.20563) • [📥 PDF](https://arxiv.org/pdf/2606.20563)

**💻 Code:** [⭐ Code](https://github.com/siang1105/JanusMesh) • [⭐ Code](https://github.com/huggingface)

> Project Page: https://siang1105.github.io/JanusMesh.github.io/ Code: https://github.com/siang1105/JanusMesh

</details>

<details>
<summary><b>6. ENPIRE: Agentic Robot Policy Self-Improvement in the Real World</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19980) • [📄 arXiv](https://arxiv.org/abs/2606.19980) • [📥 PDF](https://arxiv.org/pdf/2606.19980)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. Thinking with Visual Grounding</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16122) • [📄 arXiv](https://arxiv.org/abs/2606.16122) • [📥 PDF](https://arxiv.org/pdf/2606.16122)

**💻 Code:** [⭐ Code](https://github.com/Jun-Kai-Zhang/visually_grounded_thinking) • [⭐ Code](https://github.com/Jun-Kai-Zhang/visually_grounded_thinking/tree/main) • [⭐ Code](https://github.com/huggingface)

> Check our dataset: https://huggingface.co/datasets/JunkaiZ/TVG and code: https://github.com/Jun-Kai-Zhang/visually_grounded_thinking/tree/main

</details>

<details>
<summary><b>8. Adaptive Volumetric Mechanical Property Fields Invariant to Resolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.18231) • [📄 arXiv](https://arxiv.org/abs/2606.18231) • [📥 PDF](https://arxiv.org/pdf/2606.18231)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://research.nvidia.com/labs/sil/projects/adavomp/

</details>

<details>
<summary><b>9. Holo-World: Unified Camera, Object and Weather Control for Video World Model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yinda Chen, Zijie Liu, Jiahui Yuan, Wenzhang Sun, Xiangchen Yin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.20083) • [📄 arXiv](https://arxiv.org/abs/2606.20083) • [📥 PDF](https://arxiv.org/pdf/2606.20083)

**💻 Code:** [⭐ Code](https://github.com/XiangchenYin/Holo-World) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. FAPO: Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19605) • [📄 arXiv](https://arxiv.org/abs/2606.19605) • [📥 PDF](https://arxiv.org/pdf/2606.19605)

**💻 Code:** [⭐ Code](https://github.com/cisco-foundation-ai/fully-automated-prompt-optimization) • [⭐ Code](https://github.com/huggingface)

> Multi-step LLM pipelines fail through interactions among retrieval, reasoning, and formatting steps, so prompt-only optimization can miss bottlenecks in the chain. We present FAPO (Fully Autonomous Prompt Optimization), a framework that lets Claud...

</details>

<details>
<summary><b>11. ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Haitao Lin, He Zhang, Zekun Qi, Wenyao Zhang, Yuyang Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19531) • [📄 arXiv](https://arxiv.org/abs/2606.19531) • [📥 PDF](https://arxiv.org/pdf/2606.19531)

**💻 Code:** [⭐ Code](https://github.com/yuyangalin/ImageWAM) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. Understanding the Behaviors of Environment-aware Information Retrieval</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16817) • [📄 arXiv](https://arxiv.org/abs/2606.16817) • [📥 PDF](https://arxiv.org/pdf/2606.16817)

**💻 Code:** [⭐ Code](https://github.com/LCO-Embedding/Envs-aware-Information-Retrieval) • [⭐ Code](https://github.com/huggingface)

> Search agents are usually optimized around one or a few “search environments”, whether web search APIs or local search built with a single retriever. In practice, search environments are diverse, shaped by the retriever’s behavior, the indexing pi...

</details>

<details>
<summary><b>13. Current World Models Lack a Persistent State Core</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guo Tang, Linghan Cai, Haoyuan Shi, Dexu Zhu, Jinpeng Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.20545) • [📄 arXiv](https://arxiv.org/abs/2606.20545) • [📥 PDF](https://arxiv.org/pdf/2606.20545)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. JAMER: Project-Level Code Framework Dataset and Benchmark on Professional Game Engines</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fanrui Zhang, Yukang Feng, Zizhen Li, Chuanhao Li, Jianwen Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19830) • [📄 arXiv](https://arxiv.org/abs/2606.19830) • [📥 PDF](https://arxiv.org/pdf/2606.19830)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. LooseControlVideo: Directorial Video Control using Spatial Blocking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19495) • [📄 arXiv](https://arxiv.org/abs/2606.19495) • [📥 PDF](https://arxiv.org/pdf/2606.19495)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project page: https://shariqfarooq123.github.io/LooseControlVideo/

</details>

<details>
<summary><b>16. Selective Synergistic Learning for Video Object-Centric Learning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.15527) • [📄 arXiv](https://arxiv.org/abs/2606.15527) • [📥 PDF](https://arxiv.org/pdf/2606.15527)

**💻 Code:** [⭐ Code](https://github.com/wjun0830/SSync) • [⭐ Code](https://github.com/huggingface)

> Typical video object-centric learning (VOCL) approaches employ slot-based frameworks that rely on reconstruction-driven encoder–decoder architectures, where learning is mediated by two spatial maps: attention maps from the encoder and object maps ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 16 |
| 📅 Today | [`2026-06-19.json`](data/daily/2026-06-19.json) | 16 |
| 📆 This Week | [`2026-W24.json`](data/weekly/2026-W24.json) | 62 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 406 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-19 | 16 | [View JSON](data/daily/2026-06-19.json) |
| 📄 2026-06-18 | 11 | [View JSON](data/daily/2026-06-18.json) |
| 📄 2026-06-17 | 18 | [View JSON](data/daily/2026-06-17.json) |
| 📄 2026-06-16 | 17 | [View JSON](data/daily/2026-06-16.json) |
| 📄 2026-06-15 | 0 | [View JSON](data/daily/2026-06-15.json) |
| 📄 2026-06-14 | 44 | [View JSON](data/daily/2026-06-14.json) |
| 📄 2026-06-13 | 44 | [View JSON](data/daily/2026-06-13.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W24 | 62 | [View JSON](data/weekly/2026-W24.json) |
| 📅 2026-W23 | 166 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 406 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |

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
