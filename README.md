<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-10-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7689+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">71</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">1081</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7689+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 30, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. VibeGame: Prompt-to-Game Development with AI-Native Engine and Self-Evolving Adversarial Agent Team</b> ⭐ 284</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33106) • [📄 arXiv](https://arxiv.org/abs/2610.33106) • [📥 PDF](https://arxiv.org/pdf/2610.33106)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/tettethu/VibeGame)

> Excited to share VibeGame, a framework that turns prompts (text and optional reference images) into complete, playable 2D game projects! To let agents build a complete, vivid game with no given assets, we build an AI-native game engine that agents...

</details>

<details>
<summary><b>2. WeVisCut: The Whole Editing Room in One Agent From Narration to Long-Form Video</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38352) • [📄 arXiv](https://arxiv.org/abs/2610.38352) • [📥 PDF](https://arxiv.org/pdf/2610.38352)

**💻 Code:** [⭐ Code](https://github.com/Tencent/WeVisCut) • [⭐ Code](https://github.com/huggingface)

> WeVisCut connects tool execution with policy learning for long-form script-to-video production. Editing inversion and verified replay recover executable supervision from 3,750 finished projects, yielding 134,771 training examples for a single 9B p...

</details>

<details>
<summary><b>3. IntReal: A Real-World Dataset and Benchmark for Multimodal Interleaved Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38425) • [📄 arXiv](https://arxiv.org/abs/2610.38425) • [📥 PDF](https://arxiv.org/pdf/2610.38425)

**💻 Code:** [⭐ Code](https://github.com/yushaohan/IntReal) • [⭐ Code](https://github.com/huggingface)

> Paper: https://arxiv.org/abs/2610.38425 Project Page: http://yushaohan.github.io/IntReal/ Dataset: https://huggingface.co/datasets/yushaohan/IntReal-100K Fine-tuned Models: https://huggingface.co/collections/yushaohan/intreal Leaderboard: https://...

</details>

<details>
<summary><b>4. WebAnything: From the Wild Web to Scalable Agent Environments</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Bingxuan Li, Pinxue Guo, Chaoyun Zhang, Zikun Cai, Tianci Xie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.37786) • [📄 arXiv](https://arxiv.org/abs/2610.37786) • [📥 PDF](https://arxiv.org/pdf/2610.37786)

**💻 Code:** [⭐ Code](https://github.com/china-qijizhifeng/Web-Anything) • [⭐ Code](https://github.com/huggingface)

> Making any url as an web agent environment!

</details>

<details>
<summary><b>5. WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.37161) • [📄 arXiv](https://arxiv.org/abs/2610.37161) • [📥 PDF](https://arxiv.org/pdf/2610.37161)

**💻 Code:** [⭐ Code](https://github.com/IQuestLab/WorkFactory) • [⭐ Code](https://github.com/huggingface)

> Excited to share WorkFactory, a scalable framework for synthesizing coherent and executable professional tasks to train Working Agents! WorkFactory combines professional intent synthesis, information support graphs, and execution-guided repair to ...

</details>

<details>
<summary><b>6. EditAgent: Rethinking Agentic Image Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39664) • [📄 arXiv](https://arxiv.org/abs/2610.39664) • [📥 PDF](https://arxiv.org/pdf/2610.39664)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. A Broader Look into MLLM's Retrieval Heads</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.32411) • [📄 arXiv](https://arxiv.org/abs/2610.32411) • [📥 PDF](https://arxiv.org/pdf/2610.32411)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We study how omni-modal LLMs retrieve relevant evidence, focusing on how retrieval mechanisms extend from text and vision to audio-visual inputs. Across 18 model checkpoints, including eight omni-modal models, we find that retrieval relies on a sp...

</details>

<details>
<summary><b>8. Co-Design Bench: Multimodal Collaborative Coding Evaluation for Agentic Models</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.35138) • [📄 arXiv](https://arxiv.org/abs/2610.35138) • [📥 PDF](https://arxiv.org/pdf/2610.35138)

**💻 Code:** [⭐ Code](https://github.com/Wiselnn570/Co-Design-Bench) • [⭐ Code](https://github.com/huggingface)

> Can a coding agent turn a real user's multimodal request into a complete, working application revision? We're sharing Co-Design Bench: Multimodal Collaborative Coding Evaluation for Agentic Models , a benchmark built around real users' requests to...

</details>

<details>
<summary><b>9. AskStream: When and What to Ask Proactively under Uncertainty in Streaming Video Understanding</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38678) • [📄 arXiv](https://arxiv.org/abs/2610.38678) • [📥 PDF](https://arxiv.org/pdf/2610.38678)

**💻 Code:** [⭐ Code](https://github.com/wanglu-cs/AskStream) • [⭐ Code](https://github.com/huggingface)

> Recent advances in multimodal large language models have increased interest in interaction paradigms for video understanding. The evolution from offline to streaming and proactive interaction has gradually shifted models from passive responders to...

</details>

<details>
<summary><b>10. After One-Click Video: The Inflow of Service-Type Content, the Failure of the Recommendation Pool Mechanism, and the BoLe System</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.37695) • [📄 arXiv](https://arxiv.org/abs/2610.37695) • [📥 PDF](https://arxiv.org/pdf/2610.37695)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> One-click video tools now let anyone make good-looking content fast. Most of it is service-type content: people watch it to meet a need, not to enjoy the content itself. Before competing, creators, tool builders and model researchers should first ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 10 |
| 📅 Today | [`2026-10-30.json`](data/daily/2026-10-30.json) | 10 |
| 📆 This Week | [`2026-W43.json`](data/weekly/2026-W43.json) | 71 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 1081 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-30 | 10 | [View JSON](data/daily/2026-10-30.json) |
| 📄 2026-10-29 | 5 | [View JSON](data/daily/2026-10-29.json) |
| 📄 2026-10-28 | 27 | [View JSON](data/daily/2026-10-28.json) |
| 📄 2026-10-27 | 19 | [View JSON](data/daily/2026-10-27.json) |
| 📄 2026-10-26 | 10 | [View JSON](data/daily/2026-10-26.json) |
| 📄 2026-10-25 | 49 | [View JSON](data/daily/2026-10-25.json) |
| 📄 2026-10-24 | 50 | [View JSON](data/daily/2026-10-24.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W43 | 71 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 1081 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |

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
