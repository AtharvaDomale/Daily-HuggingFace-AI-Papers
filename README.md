<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-10-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8029+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">22</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">301</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8029+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 17, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. K2-Horizon: A Frontier Fleet of Open Source Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21003) • [📄 arXiv](https://arxiv.org/abs/2611.21003) • [📥 PDF](https://arxiv.org/pdf/2611.21003)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present K2-Horizon, a fleet of six fully open language models, from a 0.9B dense model to a 375B-A23B MoE, built for general-purpose reasoning and agentic work across math and science, software engineering, deep research, office automation, and...

</details>

<details>
<summary><b>2. PaperCritic: Training AI Reviewers to Assess the Soundness of Papers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17834) • [📄 arXiv](https://arxiv.org/abs/2611.17834) • [📥 PDF](https://arxiv.org/pdf/2611.17834)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> As AI conference submissions grow rapidly, AI reviewers are being considered as an alternative. However, simply imitating human reviewing is problematic because reviewing serves multiple purposes, including evaluating papers' soundness, interestin...

</details>

<details>
<summary><b>3. AllocFlow: End-to-End Portfolio Trading Action Generation from Raw Market Data</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Qianyu Julie Zhu, Leon Liangyu Chen, Yu Cen, Yuchen Xie, Annan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.19937) • [📄 arXiv](https://arxiv.org/abs/2611.19937) • [📥 PDF](https://arxiv.org/pdf/2611.19937)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/allocflow/AllocFlow)

> Project Page: https://allocflow.github.io/

</details>

<details>
<summary><b>4. SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07222) • [📄 arXiv](https://arxiv.org/abs/2611.07222) • [📥 PDF](https://arxiv.org/pdf/2611.07222)

**💻 Code:** [⭐ Code](https://github.com/saintmachines/spillway) • [⭐ Code](https://github.com/huggingface)

> Keep your LLM and AI agents. Add SPILLWAY. SPILLWAY 1.0.0, a real-time safety guard for AI agents, is the only one of seven prompt-injection defenses on AgentDojo with zero in-scope attack success in all 12 backbone-by-suite cells, and it has the ...

</details>

<details>
<summary><b>5. Calpha: Toward Native Biomolecular Intelligence in a General-Purpose Language Model</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.20986) • [📄 arXiv](https://arxiv.org/abs/2611.20986) • [📥 PDF](https://arxiv.org/pdf/2611.20986)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yliust/Calpha)

> Excited to share Calpha , our open-source effort to bring native protein structure prediction into general-purpose LLMs! Instead of calling an external folding tool, Calpha equips Qwen3.5-9B with a flow-matching structure head that generates all-a...

</details>

<details>
<summary><b>6. VoxMel TTS Technical Report</b> ⭐ 75</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.20254) • [📄 arXiv](https://arxiv.org/abs/2611.20254) • [📥 PDF](https://arxiv.org/pdf/2611.20254)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VoxDance-Lab/VoxMel)

> VoxMel explores a different path for autoregressive TTS: generating directly in the continuous Mel spectrogram space, without relying on learned acoustic tokenizers. The key idea is to use wavelet decomposition to model acoustic information at dif...

</details>

<details>
<summary><b>7. MovieRoll: A Systematic Framework for Rolling Out Long-Form Videos</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qi Tian, Yue Wu, Ryan Xu, Weijie Kong, Yunyang Ge

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21339) • [📄 arXiv](https://arxiv.org/abs/2611.21339) • [📥 PDF](https://arxiv.org/pdf/2611.21339)

**💻 Code:** [⭐ Code](https://github.com/Tencent-Hunyuan/MovieRoll) • [⭐ Code](https://github.com/huggingface)

> Streaming video generation must sustain quality beyond the training horizon while handling self-generated history and growing cache costs. To address these challenges, we present MovieRoll, a base-to-distilled framework that connects streaming fin...

</details>

<details>
<summary><b>8. LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.19649) • [📄 arXiv](https://arxiv.org/abs/2611.19649) • [📥 PDF](https://arxiv.org/pdf/2611.19649)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Looped Transformers improve parameter effi- ciency by repeatedly applying the same Trans- former layers. During autoregressive decoding, however, generating each token requires several sequential passes through these layers, leading to high decodi...

</details>

<details>
<summary><b>9. Latent Feedback Is a Compute and Data Amplifier</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.20404) • [📄 arXiv](https://arxiv.org/abs/2611.20404) • [📥 PDF](https://arxiv.org/pdf/2611.20404)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ifm-ai/LoRM)

> Code, data, checkpoints: https://github.com/ifm-ai/LoRM

</details>

<details>
<summary><b>10. Ego Guidance: Enhancing Robot Policy Learning with Action-Free Human Egocentric Videos</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.13685) • [📄 arXiv](https://arxiv.org/abs/2611.13685) • [📥 PDF](https://arxiv.org/pdf/2611.13685)

**💻 Code:** [⭐ Code](https://github.com/llong-cs/EgoGuidance) • [⭐ Code](https://github.com/huggingface)

> Can inexpensive, readily available human egocentric videos benefit embodied tasks without explicit action annotations? We design an architecture to uncover the implicit mutual information between actions and video, and explore its potential for ge...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 10 |
| 📅 Today | [`2026-11-17.json`](data/daily/2026-11-17.json) | 10 |
| 📆 This Week | [`2026-W46.json`](data/weekly/2026-W46.json) | 22 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 301 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-17 | 10 | [View JSON](data/daily/2026-11-17.json) |
| 📄 2026-11-16 | 12 | [View JSON](data/daily/2026-11-16.json) |
| 📄 2026-11-15 | 34 | [View JSON](data/daily/2026-11-15.json) |
| 📄 2026-11-14 | 34 | [View JSON](data/daily/2026-11-14.json) |
| 📄 2026-11-13 | 6 | [View JSON](data/daily/2026-11-13.json) |
| 📄 2026-11-12 | 3 | [View JSON](data/daily/2026-11-12.json) |
| 📄 2026-11-11 | 11 | [View JSON](data/daily/2026-11-11.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W46 | 22 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 301 | [View JSON](data/monthly/2026-11.json) |
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
