<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-15-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4637+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">15</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">56</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">234</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4637+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 11, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Agentic Environment Engineering for Large Language Models: A Survey of Environment Modeling, Synthesis, Evaluation, and Application</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12191) • [📄 arXiv](https://arxiv.org/abs/2606.12191) • [📥 PDF](https://arxiv.org/pdf/2606.12191)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper systematically reviews agentic environments for LLM-based agents from the perspective of environment engineering, covering environment modeling, synthesis, evaluation, application, and future opportunities.

</details>

<details>
<summary><b>2. Toward Generalist Autonomous Research via Hypothesis-Tree Refinement</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11926) • [📄 arXiv](https://arxiv.org/abs/2606.11926) • [📥 PDF](https://arxiv.org/pdf/2606.11926)

**💻 Code:** [⭐ Code](https://github.com/RUC-NLPIR/Arbor) • [⭐ Code](https://github.com/huggingface)

> From single-turn chatbots, to multi-turn dialogue systems, and then to tool-using agents, we believe the next important stage is the rise of Autonomous Agents. However, many existing efforts are either tightly bound to specific scenarios and singl...

</details>

<details>
<summary><b>3. DeNovoSWE: Scaling Long-Horizon Environments for Generating Entire Repositories from Scratch</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10728) • [📄 arXiv](https://arxiv.org/abs/2606.10728) • [📥 PDF](https://arxiv.org/pdf/2606.10728)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AweAI-Team/DeNovoSWE)

> As the capabilities of LLM-based code agents continue to advance, their expected role is expanding beyond localized bug fixing in existing codebases toward architecting and implementing complete software repositories from high-level specifications...

</details>

<details>
<summary><b>4. Beyond Scalar Rewards by Internalizing Reasoning into Score Distributions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09076) • [📄 arXiv](https://arxiv.org/abs/2606.09076) • [📥 PDF](https://arxiv.org/pdf/2606.09076)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Tech report. Z-Reward is a reasoning-internalized teacher-student reward modeling framework for visual generation, developed by the Z-Image Team. Z-Reward decouples reasoning-heavy judgment from efficient reward deployment: 🧠 The Teacher (27B): A ...

</details>

<details>
<summary><b>5. World Pilot: Steering Vision-Language-Action Models with World-Action Priors</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Wenling Li, Xiaojuan Jin, Junjia Xu, Rongxu Cui, Zefu Lin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12403) • [📄 arXiv](https://arxiv.org/abs/2606.12403) • [📥 PDF](https://arxiv.org/pdf/2606.12403)

**💻 Code:** [⭐ Code](https://github.com/ZefuLin/WorldPilot) • [⭐ Code](https://github.com/huggingface)

> Cool Embodied Manipulation framework

</details>

<details>
<summary><b>6. TRL-Bench: Standardizing Cross-Paradigm Representation-Level Evaluation of Tabular Encoders</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09323) • [📄 arXiv](https://arxiv.org/abs/2606.09323) • [📥 PDF](https://arxiv.org/pdf/2606.09323)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LOGO-CUHKSZ/TRL-Bench)

> 📊 Releasing TRL-Bench — a unified framework + library for tabular representation learning, one stop for tabular representation learning. 🧩 20 encoders · 16 tasks · 87 datasets across 3 suites 🔍 Built to make heterogeneous tabular models directly c...

</details>

<details>
<summary><b>7. ComBench: A Benchmark for Rigorous Proof Reasoning and Constructive Realization in Olympiad-Level Combinatorics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10479) • [📄 arXiv](https://arxiv.org/abs/2606.10479) • [📥 PDF](https://arxiv.org/pdf/2606.10479)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Simplified-Reasoning/ComBench)

> Combinatorics is central to Olympiad-level mathematical problem solving, requiring deep discrete reasoning, creative constructions, and rigorous structural insight. Recent evidence suggests that even today's strongest frontier models remain uneven...

</details>

<details>
<summary><b>8. InternVideo3: Agentify Foundation Models with Multimodal Contextual Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tianxiang Jiang, Yue Wu, Jiashuo Yu, Sheng Xia, Ziang Yan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12195) • [📄 arXiv](https://arxiv.org/abs/2606.12195) • [📥 PDF](https://arxiv.org/pdf/2606.12195)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo3)

> A framework for long-horizon video understanding via closed-loop contextual reasoning and efficient latent attention. Github: https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo3

</details>

<details>
<summary><b>9. ICA Lens: Interpreting Language Models Without Training Another Dictionary</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11722) • [📄 arXiv](https://arxiv.org/abs/2606.11722) • [📥 PDF](https://arxiv.org/pdf/2606.11722)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/liusida/ica-lens-paper)

> Many meaningful activation directions are selective: they fire for recurring words, constructions, topics, contexts, or discourse patterns rather than for typical random projections. That selectivity leaves a non-Gaussian footprint. ICA Lens turns...

</details>

<details>
<summary><b>10. Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12373) • [📄 arXiv](https://arxiv.org/abs/2606.12373) • [📥 PDF](https://arxiv.org/pdf/2606.12373)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reinforcement Learning (RL) with verifiable environments has emerged as a powerful approach for enhancing the reasoning capabilities of Large Language Models (LLMs). While prior research demonstrates that scaling environment quantity improves RL p...

</details>

<details>
<summary><b>11. Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12370) • [📄 arXiv](https://arxiv.org/abs/2606.12370) • [📥 PDF](https://arxiv.org/pdf/2606.12370)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Accelerates RL training in LLMs using MTP with rejection sampling and a novel end-to-end TV loss.

</details>

<details>
<summary><b>12. Lius: Translation Model Based Instructional Lingustic Using Continual Instruction Tuning In Kupang Malay</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11786) • [📄 arXiv](https://arxiv.org/abs/2606.11786) • [📥 PDF](https://arxiv.org/pdf/2606.11786)

**💻 Code:** [⭐ Code](https://github.com/joanitolopo/instructional-linguistic-llm) • [⭐ Code](https://github.com/huggingface)

> We introduce Lius , an Indonesian → Kupang Malay translation model designed for low-resource machine translation. Kupang Malay is a Malay-based creole spoken in East Nusa Tenggara, Indonesia, but it remains underrepresented in current NLP resource...

</details>

<details>
<summary><b>13. World Model Self-Distillation: Training World Models to Solve General Tasks</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Paolo Favaro, Aram Davtyan, Pablo Acuaviva Huertos, Sebastian Stapf

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12072) • [📄 arXiv](https://arxiv.org/abs/2606.12072) • [📥 PDF](https://arxiv.org/pdf/2606.12072)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sebastian-stapf/world-model-self-distillation)

> A scalable framework that trains world models to solve tasks via self-distillation and RL from VLM feedback.

</details>

<details>
<summary><b>14. Embodied-R1.5: Evolving Physical Intelligence via Embodied Foundation Models</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Shuoheng Zhang, Yutong Li, Xianze Yao, Yaoting Huang, Yifu Yuan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11324) • [📄 arXiv](https://arxiv.org/abs/2606.11324) • [📥 PDF](https://arxiv.org/pdf/2606.11324)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/pickxiguapi/Embodied-R1.5)

> A unified embodied foundation model achieving SOTA on embodied VLM and manipulation benchmarks with self-correction capabilities.

</details>

<details>
<summary><b>15. i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Taiming Lu, Jucheng Shen, Shu Pu, Tianze Luo, Boya Zeng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11289) • [📄 arXiv](https://arxiv.org/abs/2606.11289) • [📥 PDF](https://arxiv.org/pdf/2606.11289)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zlab-princeton/i1)

> A high-performing, fully open-source 3B-parameter text-to-image diffusion model with a systematic training recipe.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 15 |
| 📅 Today | [`2026-06-11.json`](data/daily/2026-06-11.json) | 15 |
| 📆 This Week | [`2026-W23.json`](data/weekly/2026-W23.json) | 56 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 234 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-11 | 15 | [View JSON](data/daily/2026-06-11.json) |
| 📄 2026-06-10 | 18 | [View JSON](data/daily/2026-06-10.json) |
| 📄 2026-06-09 | 8 | [View JSON](data/daily/2026-06-09.json) |
| 📄 2026-06-08 | 15 | [View JSON](data/daily/2026-06-08.json) |
| 📄 2026-06-07 | 50 | [View JSON](data/daily/2026-06-07.json) |
| 📄 2026-06-06 | 50 | [View JSON](data/daily/2026-06-06.json) |
| 📄 2026-06-05 | 21 | [View JSON](data/daily/2026-06-05.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W23 | 56 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 234 | [View JSON](data/monthly/2026-06.json) |
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
