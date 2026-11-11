<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-11-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7930+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">11</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">30</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">202</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7930+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 11, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. dNaViT: Discrete Codes are General-Purpose Visual Representations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.12426) • [📄 arXiv](https://arxiv.org/abs/2611.12426) • [📥 PDF](https://arxiv.org/pdf/2611.12426)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce dNaViT: Discrete Codes are General-Purpose Visual Representations. dNaViT is a native-resolution visual tokenizer that combines residual modulation, grouped product quantization, and progressive three-stage training to preserve both s...

</details>

<details>
<summary><b>2. World In-Context Learning for Agentic Robot Manipulation</b> ⭐ 36</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.12885) • [📄 arXiv](https://arxiv.org/abs/2611.12885) • [📥 PDF](https://arxiv.org/pdf/2611.12885)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ArcheOne-AI/RoboScript)

> Trained action policies write manipulation skill into network weights, tying what is learned to the distribution it is learned from. In-context learning (ICL) avoids this coupling but has not matched trained policies, because a demonstration recor...

</details>

<details>
<summary><b>3. Few-Step Streaming Video Generation via Condition Shifting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.14033) • [📄 arXiv](https://arxiv.org/abs/2611.14033) • [📥 PDF](https://arxiv.org/pdf/2611.14033)

**💻 Code:** [⭐ Code](https://github.com/shengshu-ai/Vidu-S) • [⭐ Code](https://github.com/huggingface)

> Hi everyone! We’re sharing our latest work extending APEX from image generation to few-step streaming video generation . Condition Shifting lets the generator and fake estimator share one network, removing the separate fake-score network and reduc...

</details>

<details>
<summary><b>4. DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.03142) • [📄 arXiv](https://arxiv.org/abs/2611.03142) • [📥 PDF](https://arxiv.org/pdf/2611.03142)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jianke0604/DepthGraft)

> Scaling depth is a key route to stronger large language models, but Pre-LN architectures often suffer from depth-wise information dilution, limiting the effective use of deep layers. While cross-layer connectivity has proven effective at mitigatin...

</details>

<details>
<summary><b>5. PhysOPSD: Code as Privilege for Physically Plausible Video Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.12411) • [📄 arXiv](https://arxiv.org/abs/2611.12411) • [📥 PDF](https://arxiv.org/pdf/2611.12411)

**💻 Code:** [⭐ Code](https://github.com/aigc3d/PhysOPSD) • [⭐ Code](https://github.com/huggingface)

> Empowering generation with coding agents: Recent approaches use coding agents to replace parts of the video generation process. We explore how coding agents can empower the generator itself. Executable code supplies structured physical supervision...

</details>

<details>
<summary><b>6. ExRL: Sample-Efficient Online Adaptation of Long-Chunk Policies via Execution Length Control</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zijin Hong, Junlin Xie, Hao Wu, Jiaying Yang, Xu Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.13482) • [📄 arXiv](https://arxiv.org/abs/2611.13482) • [📥 PDF](https://arxiv.org/pdf/2611.13482)

**💻 Code:** [⭐ Code](https://github.com/Frankluox/ExRL) • [⭐ Code](https://github.com/huggingface)

> ExRL enables sample-efficient online improvement of long-chunk robot policies by RL. It learns how much of each predicted chunk to execute using a lightweight RL policy, improving task performance across simulation and the real world.

</details>

<details>
<summary><b>7. ProTEUS: A mutation-resolved world model for closed-loop protein evolution</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wojciech Matusik, Peter Yichen Chen, Guangxue Xu, Minghao Guo, Yikun Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.13348) • [📄 arXiv](https://arxiv.org/abs/2611.13348) • [📥 PDF](https://arxiv.org/pdf/2611.13348)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yikunpku/ProTEUS)

> No abstract available.

</details>

<details>
<summary><b>8. DexForesight: Grounding Dexterous Policies with Implicit Spatial Foresight</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.11166) • [📄 arXiv](https://arxiv.org/abs/2611.11166) • [📥 PDF](https://arxiv.org/pdf/2611.11166)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Excited to share DexForesight , our latest work from the Sharpa Foundation Model Team ! Project Page: https://dexforesight.github.io/ Most existing dexterous manipulation models rely primarily on RGB observations. However, even in simple pick-and-...

</details>

<details>
<summary><b>9. Predictive Surprise as Self-Grounding Concept Bottleneck for Interpretable Time Series</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.12955) • [📄 arXiv](https://arxiv.org/abs/2611.12955) • [📥 PDF](https://arxiv.org/pdf/2611.12955)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Sachithx/ConceptTime)

> CONCEPTTIME uses predictive surprise from a frozen probabilistic forecaster to jointly identify segment boundaries and learn an interpretable vocabulary of temporal concepts. A lightweight classifier predicts from the concept sequence, while dista...

</details>

<details>
<summary><b>10. Learning to Act with World Models by Valuing What Comes Next</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.12200) • [📄 arXiv](https://arxiv.org/abs/2611.12200) • [📥 PDF](https://arxiv.org/pdf/2611.12200)

**💻 Code:** [⭐ Code](https://github.com/nuskonghou/wm-ppo) • [⭐ Code](https://github.com/huggingface)

> TL;DR. A sparse terminal reward says little about individual steps, yet each state transition can change the chances of success long before the reward arrives. WM-PPO learns from this information: a world model trained on real transitions predicts...

</details>

<details>
<summary><b>11. Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.14488) • [📄 arXiv](https://arxiv.org/abs/2611.14488) • [📥 PDF](https://arxiv.org/pdf/2611.14488)

**💻 Code:** [⭐ Code](https://github.com/bennidict23/memory-clock) • [⭐ Code](https://github.com/huggingface)

> We asked a robot to repeat a task three times. Two 5-second pauses made it stop after two. On RoboMME, those pauses cut a released memory VLA’s success from 47/50 to 16/50. Hide the same pause frames from memory, and success returns to 45/50. The ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 11 |
| 📅 Today | [`2026-11-11.json`](data/daily/2026-11-11.json) | 11 |
| 📆 This Week | [`2026-W45.json`](data/weekly/2026-W45.json) | 30 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 202 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-11 | 11 | [View JSON](data/daily/2026-11-11.json) |
| 📄 2026-11-10 | 9 | [View JSON](data/daily/2026-11-10.json) |
| 📄 2026-11-09 | 10 | [View JSON](data/daily/2026-11-09.json) |
| 📄 2026-11-08 | 32 | [View JSON](data/daily/2026-11-08.json) |
| 📄 2026-11-07 | 32 | [View JSON](data/daily/2026-11-07.json) |
| 📄 2026-11-06 | 8 | [View JSON](data/daily/2026-11-06.json) |
| 📄 2026-11-05 | 13 | [View JSON](data/daily/2026-11-05.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W45 | 30 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 202 | [View JSON](data/monthly/2026-11.json) |
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
