<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-14-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7781+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">14</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">14</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">53</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7781+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 02, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Does Scaling Toward AGI Depend on What System-1 Lets System-2 Execute?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39497) • [📄 arXiv](https://arxiv.org/abs/2610.39497) • [📥 PDF](https://arxiv.org/pdf/2610.39497)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Progress toward artificial general intelligence (AGI) has largely relied on scaling a resource-intensive executive, System-2, through larger models, more data and compute, longer reasoning, and more agents. We ask whether the next stage may come i...

</details>

<details>
<summary><b>2. R^2Orch: Recursive Self-Improvement for Agent Orchestration via Self-Play within a Skill-Internalization Loop</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22740) • [📄 arXiv](https://arxiv.org/abs/2610.22740) • [📥 PDF](https://arxiv.org/pdf/2610.22740)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/beita6969/R2Orch)

> R²Orch (Recursive Self-Play Orchestration) trains an agent orchestrator without a hand-designed curriculum: an inner self-play loop improves the orchestrator on tasks a Proposer picks, and an outer loop improves how that inner loop learns. The Pro...

</details>

<details>
<summary><b>3. CEPO: A General Framework for Improving Prompt Optimization in Multi-Agent Systems</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Hyunwoo J. Kim, Jinsung Yoon, Jonathan Minho Chung, Seohyun Lee, jonghakim

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.34722) • [📄 arXiv](https://arxiv.org/abs/2610.34722) • [📥 PDF](https://arxiv.org/pdf/2610.34722)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/mlvlab/CEPO)

> Multi-agent systems (MAS) require both effective coordination and specialized task execution. Existing prompt optimization approaches typically couple these functions within agent-specific prompts. We introduce Coordinator-Executor Prompt Optimiza...

</details>

<details>
<summary><b>4. HiFOR: Hierarchical Foresight with Action-Guided World Grounding for Vision-Language-Action Model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fan Zhang, Xudong Wang, Kehong Wu, Hongyu Qu, Rongxing Ding

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40373) • [📄 arXiv](https://arxiv.org/abs/2610.40373) • [📥 PDF](https://arxiv.org/pdf/2610.40373)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Vision-Language-Action (VLA) models have demonstrated promising capabilities in robotic manipulation by integrating visual perception, language understanding, and action generation. Recent advances have further incorporated future prediction to en...

</details>

<details>
<summary><b>5. FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.36545) • [📄 arXiv](https://arxiv.org/abs/2610.36545) • [📥 PDF](https://arxiv.org/pdf/2610.36545)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Lyne1/FrameForge)

> Unified conditional image generation requires a single model to preserve different aspects of a visual input while producing a target image that follows textual guidance. Pretrained video models provide a promising foundation by encoding visual tr...

</details>

<details>
<summary><b>6. Wizard: Manipulate Any World in Real Time via Agentic Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40339) • [📄 arXiv](https://arxiv.org/abs/2610.40339) • [📥 PDF](https://arxiv.org/pdf/2610.40339)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🪄 Wizard: Manipulate Any World in Real Time via Agentic Control! Existing game world models produce realistic visuals but remain constrained by familiar environments and predefined motions. Control largely focuses on locomotion, with limited suppo...

</details>

<details>
<summary><b>7. WorldBridge: Bridging Video and Action for Task Generalization in World Action Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yaoxu Lyu, Liangkai Li, Hao Li, Yubo Guo, Jun Cen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40748) • [📄 arXiv](https://arxiv.org/abs/2610.40748) • [📥 PDF](https://arxiv.org/pdf/2610.40748)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project Page: https://worldbridge-jd.github.io/

</details>

<details>
<summary><b>8. Probing Spatial State Representations in Visual Foundation Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40636) • [📄 arXiv](https://arxiv.org/abs/2610.40636) • [📥 PDF](https://arxiv.org/pdf/2610.40636)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. RoboArchitect: Automating the Design of Robotic Agents with Persistent Memory</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hao Zhu, Ria Doshi, Cherie Ho, Suning Huang, Shengguang Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.42141) • [📄 arXiv](https://arxiv.org/abs/2610.42141) • [📥 PDF](https://arxiv.org/pdf/2610.42141)

**💻 Code:** [⭐ Code](https://github.com/roboarchitect-ai/RoboArchitect) • [⭐ Code](https://github.com/huggingface)

> RoboArchitect automatically designs and continually improves a robotic agent around a frozen robot policy, without updating any weights. These agents develop their own control routines. Most converge on a dynamic split between the VLA policy and c...

</details>

<details>
<summary><b>10. CaPtain: A Pipeline for Post-Training Code-as-Policy Robot Agents</b> ⭐ 32</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40799) • [📄 arXiv](https://arxiv.org/abs/2610.40799) • [📥 PDF](https://arxiv.org/pdf/2610.40799)

**💻 Code:** [⭐ Code](https://github.com/waltstephen/CaPtain) • [⭐ Code](https://github.com/huggingface)

> Across coding, web, and tool-use tasks, language-model agents are now routinely trained with the same recipe: supervised fine-tuning (SFT) to learn the interface, then multi-turn reinforcement learning (RL) to master it. Code-aspolicy (CaP) puts r...

</details>

<details>
<summary><b>11. On the Origin and Evolution of Training-Inference Mismatch in LLM Reinforcement Learning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40945) • [📄 arXiv](https://arxiv.org/abs/2610.40945) • [📥 PDF](https://arxiv.org/pdf/2610.40945)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Skylanding/Training-Inference-Mismatch)

> We split the training-inference mismatch in LLM RL into a numerical discrepancy, which stays nearly constant, and a policy lag, which compounds through the trainer's own updates. From this split we derive the discrepancy-lag correction (DLC), whic...

</details>

<details>
<summary><b>12. CausalWAM: Learning the Causes of Actions in World Action Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.42096) • [📄 arXiv](https://arxiv.org/abs/2610.42096) • [📥 PDF](https://arxiv.org/pdf/2610.42096)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Introducing CWAM (Causal WAM), a world action model that provably learns the true causes of actions.

</details>

<details>
<summary><b>13. WeFair: Fidelity-Aware Alignment from Human Feedback for Personalized Query Rewriting in Short-Video Search</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16927) • [📄 arXiv](https://arxiv.org/abs/2610.16927v2) • [📥 PDF](https://arxiv.org/pdf/2610.16927)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Personalization should clarify user intent, not overwrite it. How can we build a compact query rewriter that understands users' personalized intent while avoiding intent drift? We introduce WeFair , a fidelity-aware alignment framework for persona...

</details>

<details>
<summary><b>14. Apriel Voice 27B Preview: An Open Model for Speech, Reasoning, and Tool Use</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40738) • [📄 arXiv](https://arxiv.org/abs/2610.40738) • [📥 PDF](https://arxiv.org/pdf/2610.40738)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Apriel Voice 27B Preview is an open speech-language model that integrates speech understanding, reasoning, and tool use within a shared backbone, achieving strong results on audio benchmarks through a two-stage training procedure with online disti...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 14 |
| 📅 Today | [`2026-11-02.json`](data/daily/2026-11-02.json) | 14 |
| 📆 This Week | [`2026-W44.json`](data/weekly/2026-W44.json) | 14 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 53 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-02 | 14 | [View JSON](data/daily/2026-11-02.json) |
| 📄 2026-11-01 | 39 | [View JSON](data/daily/2026-11-01.json) |
| 📄 2026-10-31 | 39 | [View JSON](data/daily/2026-10-31.json) |
| 📄 2026-10-30 | 10 | [View JSON](data/daily/2026-10-30.json) |
| 📄 2026-10-29 | 5 | [View JSON](data/daily/2026-10-29.json) |
| 📄 2026-10-28 | 27 | [View JSON](data/daily/2026-10-28.json) |
| 📄 2026-10-27 | 19 | [View JSON](data/daily/2026-10-27.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W44 | 14 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 53 | [View JSON](data/monthly/2026-11.json) |
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
