<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-9-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5476+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">19</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">95</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5476+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 05, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03974) • [📄 arXiv](https://arxiv.org/abs/2608.03974) • [📥 PDF](https://arxiv.org/pdf/2608.03974)

**💻 Code:** [⭐ Code](https://github.com/jd-opensource/JoyAI-Video-Edit) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>2. MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28956) • [📄 arXiv](https://arxiv.org/abs/2607.28956) • [📥 PDF](https://arxiv.org/pdf/2607.28956)

**💻 Code:** [⭐ Code](https://github.com/KhanCold/merchantbench) • [⭐ Code](https://github.com/huggingface)

> homepage: https://air.1688.com/kapp/next1688/merchantbench/?spm=defwork.home.0.0.7f8c530dSdRT9v code: https://github.com/KhanCold/merchantbench paper: https://arxiv.org/abs/2607.28956

</details>

<details>
<summary><b>3. UniWorld-Design: From Pixel Generation to Layer-Native Design</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03971) • [📄 arXiv](https://arxiv.org/abs/2608.03971) • [📥 PDF](https://arxiv.org/pdf/2608.03971)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Paper: https://arxiv.org/abs/2608.03971

</details>

<details>
<summary><b>4. Any-OPD: Heterogeneous On-Policy Distillation for Flow-Matching Models via Representation-Space Bridging</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03316) • [📄 arXiv](https://arxiv.org/abs/2608.03316) • [📥 PDF](https://arxiv.org/pdf/2608.03316)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> On-policy distillation, in which a teacher corrects samples that the student itself generates, presupposes that the two models speak the same language: identical VAE latents, matching architectures, and a common timestep grid. We ask what happens ...

</details>

<details>
<summary><b>5. SkillJack: Persistent Skill Backdoors in Self-Evolving Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03509) • [📄 arXiv](https://arxiv.org/abs/2608.03509) • [📥 PDF](https://arxiv.org/pdf/2608.03509)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper introduces SkillJack, the first attack targeting the skill evolution process of self-evolving agents. Unlike prior memory poisoning attacks that rely on poisoned context being retrieved, SkillJack compromises the agent's own experience-...

</details>

<details>
<summary><b>6. Hunyuan3D-Buffalo 1.0: A Unified Multimodal Model for Scalable 3D Generation, Understanding, and Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.02711) • [📄 arXiv](https://arxiv.org/abs/2608.02711) • [📥 PDF](https://arxiv.org/pdf/2608.02711)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. GROVE: Growing and Reasoning over Temporally Stratified Memory from Streaming Video Experience</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bo Zheng, Guo Chen, Tianyu Yan, Caixin Kang, Sitong Gong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.02392) • [📄 arXiv](https://arxiv.org/abs/2608.02392) • [📥 PDF](https://arxiv.org/pdf/2608.02392)

**💻 Code:** [⭐ Code](https://github.com/SitongGong/GROVE) • [⭐ Code](https://github.com/huggingface)

> A wearable assistant should both answer questions about its visual history and recognize when that history is useful to the present situation. Existing video-memory systems primarily support question-conditioned recall, whereas proactive assistant...

</details>

<details>
<summary><b>8. CAPEval: A Decoupled Caption Evaluation across Understanding and Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhaoxiang Zhang, Haochen Wang, LiuzhipengUCAS

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.02589) • [📄 arXiv](https://arxiv.org/abs/2608.02589) • [📥 PDF](https://arxiv.org/pdf/2608.02589)

**💻 Code:** [⭐ Code](https://github.com/liuzhipenggg/CAPEval) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/liuzhipenggg/CAPEval Suppose an image contains 10 factual claims: Caption A references 9 of them but misstates 2 , while Caption B only mentions 5, all of which are factually correct . Which caption is higher‑quality? You ...

</details>

<details>
<summary><b>9. TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04007) • [📄 arXiv](https://arxiv.org/abs/2608.04007) • [📥 PDF](https://arxiv.org/pdf/2608.04007)

**💻 Code:** [⭐ Code](https://github.com/quchangle1/TurnSight) • [⭐ Code](https://github.com/huggingface)

> TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning 💡 Overview We propose TurnSight , a turn-level hindsight self-distillation framework designed for Tool-Integrated Reasoning (TIR). The key idea is to derive fine-grain...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 9 |
| 📅 Today | [`2026-08-05.json`](data/daily/2026-08-05.json) | 9 |
| 📆 This Week | [`2026-W31.json`](data/weekly/2026-W31.json) | 19 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 95 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-05 | 9 | [View JSON](data/daily/2026-08-05.json) |
| 📄 2026-08-04 | 2 | [View JSON](data/daily/2026-08-04.json) |
| 📄 2026-08-03 | 8 | [View JSON](data/daily/2026-08-03.json) |
| 📄 2026-08-02 | 38 | [View JSON](data/daily/2026-08-02.json) |
| 📄 2026-08-01 | 38 | [View JSON](data/daily/2026-08-01.json) |
| 📄 2026-07-31 | 10 | [View JSON](data/daily/2026-07-31.json) |
| 📄 2026-07-30 | 6 | [View JSON](data/daily/2026-07-30.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W31 | 19 | [View JSON](data/weekly/2026-W31.json) |
| 📅 2026-W30 | 112 | [View JSON](data/weekly/2026-W30.json) |
| 📅 2026-W29 | 70 | [View JSON](data/weekly/2026-W29.json) |
| 📅 2026-W28 | 94 | [View JSON](data/weekly/2026-W28.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 95 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
