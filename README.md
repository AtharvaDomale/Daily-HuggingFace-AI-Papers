<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-12-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7163+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">36</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">555</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7163+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 13, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39601) • [📄 arXiv](https://arxiv.org/abs/2609.39601) • [📥 PDF](https://arxiv.org/pdf/2609.39601)

**💻 Code:** [⭐ Code](https://github.com/groundingpi/GroundingPI) • [⭐ Code](https://github.com/huggingface)

> 🤖 GroundingPI: What foundation model does embodied AI need? 4B parameters. SOTA overall across 34 grounding benchmarks, surpassing GPT-6 Astra! By training precise perception as a native capability, GroundingPI achieves the best robot generalizati...

</details>

<details>
<summary><b>2. Scaling Agentic Data for Long-Horizon Terminal Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.13166) • [📄 arXiv](https://arxiv.org/abs/2610.13166) • [📥 PDF](https://arxiv.org/pdf/2610.13166)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Scaling Agentic Data for Long-Horizon Terminal Intelligence Website: https://zs1314.github.io/TerminalHorizon

</details>

<details>
<summary><b>3. Multi-Stream Attention Residuals</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.13630) • [📄 arXiv](https://arxiv.org/abs/2610.13630) • [📥 PDF](https://arxiv.org/pdf/2610.13630)

**💻 Code:** [⭐ Code](https://github.com/jianke0604/MSAR) • [⭐ Code](https://github.com/huggingface)

> Attention Residuals (AttnRes) enables content-dependent retrieval over network depth, which can be viewed as single-stream softmax attention along the depth axis. Inspired by the success of multi-head attention along the sequence axis, we introduc...

</details>

<details>
<summary><b>4. AgentSWE: Can Coding Agents Build the Agent You Actually Want?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zheng Liu, Jiajun Zhang, Yuyang Hu, Hongjin Qian, Jiahao Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.13361) • [📄 arXiv](https://arxiv.org/abs/2610.13361) • [📥 PDF](https://arxiv.org/pdf/2610.13361)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VectorSpaceLab/AgentSWE)

> pdf: https://arxiv.org/pdf/2610.13361 Code & Dataset: https://github.com/VectorSpaceLab/AgentSWE

</details>

<details>
<summary><b>5. SpatialASM: Internalizing Active Mentalization for Spatial Reasoning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.13832) • [📄 arXiv](https://arxiv.org/abs/2610.13832) • [📥 PDF](https://arxiv.org/pdf/2610.13832)

**💻 Code:** [⭐ Code](https://github.com/InternLM/SpatialASM) • [⭐ Code](https://github.com/huggingface)

> With SpatialASM, we explore whether LVLMs can go beyond passively observing the given images and instead actively reason by planning task-relevant viewpoints and mentally imagining what should be visible from them. We introduce Active Spatial Ment...

</details>

<details>
<summary><b>6. GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39600) • [📄 arXiv](https://arxiv.org/abs/2609.39600) • [📥 PDF](https://arxiv.org/pdf/2609.39600)

**💻 Code:** [⭐ Code](https://github.com/groundingpi/GroundAnything) • [⭐ Code](https://github.com/huggingface)

> ⚡ Why locate objects one token at a time? SOTA grounding with a 4× speedup is here! Meet GroundAnything: an open-source 4B diffusion model that generates grounding predictions in parallel, delivering a 4× speedup over the autoregressive baseline r...

</details>

<details>
<summary><b>7. Systematic Multi-Agent Vision-and-Language Navigation: Formulation, Benchmark, and Method</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35965) • [📄 arXiv](https://arxiv.org/abs/2609.35965) • [📥 PDF](https://arxiv.org/pdf/2609.35965)

**💻 Code:** [⭐ Code](https://github.com/xyz9911/MA-VLN) • [⭐ Code](https://github.com/huggingface)

> Vision-and-Language Navigation (VLN) has largely focused on a single agent following a single instruction, yet many real-world applications require teams of robots to tackle tasks beyond the capabilities of any individual agent. We present Systema...

</details>

<details>
<summary><b>8. Thinking Inertia: LLMs Keep Thinking When Told Not To</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11765) • [📄 arXiv](https://arxiv.org/abs/2610.11765) • [📥 PDF](https://arxiv.org/pdf/2610.11765)

**💻 Code:** [⭐ Code](https://github.com/thinking-inertia/code) • [⭐ Code](https://github.com/huggingface)

> Today’s LLMs are used to thinking longer. But can they stop when explicitly told NOT to? Excited to share our #NeurIPS2026 work: LLMs Keep Thinking When Told Not To We observe persistent “Thinking Inertia” : LLMs keep reasoning even when explicitl...

</details>

<details>
<summary><b>9. T3lescope: Arbitrary-Resolution High-Fidelity Generative Surface Reconstruction from Images</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03308) • [📄 arXiv](https://arxiv.org/abs/2610.03308) • [📥 PDF](https://arxiv.org/pdf/2610.03308)

**💻 Code:** [⭐ Code](https://github.com/pfnet-research/t3lescope) • [⭐ Code](https://github.com/huggingface)

> T3lescope 🔭 is a high-fidelity 3D mesh reconstruction method from images. It is a generalizable reconstruction model: a single model reconstructs surface geometry of unseen scenes without optimizing a 3D representation per scene.

</details>

<details>
<summary><b>10. ARCS: Towards Precise Text-to-SQL via Structured Disambiguation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09396) • [📄 arXiv](https://arxiv.org/abs/2610.09396) • [📥 PDF](https://arxiv.org/pdf/2610.09396)

**💻 Code:** [⭐ Code](https://github.com/megagonlabs/tabulaflow) • [⭐ Code](https://github.com/huggingface)

> ARCS ( A mbiguity R esolution C orpus for S QL) is a text-to-SQL benchmark featuring naturally occurring, unconstrained ambiguities over real-world databases, with complete annotations of valid ambiguity points, interpretations, and SQL queries. W...

</details>

<details>
<summary><b>11. Learning More by Training Less: Uncertainty-Guided Progressive Contraction for Efficient Reasoning Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12933) • [📄 arXiv](https://arxiv.org/abs/2610.12933) • [📥 PDF](https://arxiv.org/pdf/2610.12933)

**💻 Code:** [⭐ Code](https://github.com/Jackbrocp/Learning-More-by-Training-Less) • [⭐ Code](https://github.com/huggingface)

> Large language models have achieved remarkable advances in complex reasoning. However, scaling reasoning post-training with ever-larger corpora can substantially increase training cost without proportional performance gains. In this study, we ther...

</details>

<details>
<summary><b>12. Relevance Does Not Imply Applicability: Experience Activation for Personal GUI Agents</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33304) • [📄 arXiv](https://arxiv.org/abs/2609.33304) • [📥 PDF](https://arxiv.org/pdf/2609.33304)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/fyzhang1/ExpActivator)

> ExpActivator activates the experience that applies to the current screen, time, and scenario. A training-free framework for personalized GUI execution and proactive suggestions.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 12 |
| 📅 Today | [`2026-10-13.json`](data/daily/2026-10-13.json) | 12 |
| 📆 This Week | [`2026-W41.json`](data/weekly/2026-W41.json) | 36 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 555 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-13 | 12 | [View JSON](data/daily/2026-10-13.json) |
| 📄 2026-10-12 | 24 | [View JSON](data/daily/2026-10-12.json) |
| 📄 2026-10-11 | 72 | [View JSON](data/daily/2026-10-11.json) |
| 📄 2026-10-10 | 72 | [View JSON](data/daily/2026-10-10.json) |
| 📄 2026-10-09 | 35 | [View JSON](data/daily/2026-10-09.json) |
| 📄 2026-10-08 | 40 | [View JSON](data/daily/2026-10-08.json) |
| 📄 2026-10-07 | 23 | [View JSON](data/daily/2026-10-07.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W41 | 36 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 555 | [View JSON](data/monthly/2026-10.json) |
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
