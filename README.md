<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-16-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6151+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">35</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">23</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6151+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** September 02, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. StudentSim: Training LLM-based Student Simulators</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01591) • [📄 arXiv](https://arxiv.org/abs/2609.01591) • [📥 PDF](https://arxiv.org/pdf/2609.01591)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/StudentSim)

> AI tutors are most useful when they adaptively respond to each student's strengths, weaknesses, and preferred kinds of guidance, but which guidance works for which student is a sparse signal, slow and costly to collect from real students. Student ...

</details>

<details>
<summary><b>2. Hi-Q: Hierarchical Evidence-guided Query Refinement for Multi-Hop Question Answering</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.30468) • [📄 arXiv](https://arxiv.org/abs/2608.30468) • [📥 PDF](https://arxiv.org/pdf/2608.30468)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Multi-hop QA has a granularity problem: you ask at one level, the corpus stores evidence at another. Graph RAG, iterative retrieval, and program-executing agents all change how you retrieve — none of them checks whether the query they just issued ...

</details>

<details>
<summary><b>3. Uncovering Understanding-Generation Synergy in Native Unified Multimodal Models: From Representation, Task to System</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dahua Lin, Lewei Lu, Weichen Fan, Haiwen Diao, Penghao Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01607) • [📄 arXiv](https://arxiv.org/abs/2609.01607) • [📥 PDF](https://arxiv.org/pdf/2609.01607)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We study when visual understanding and generation truly reinforce each other in native unified multimodal models, from representation to task to system.

</details>

<details>
<summary><b>4. SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01343) • [📄 arXiv](https://arxiv.org/abs/2609.01343) • [📥 PDF](https://arxiv.org/pdf/2609.01343)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Looped Transformers increase effective depth by iterating a shared block of layers, but most evaluations compare at fixed model size, conflating architectural advantage with extra FLOPs. We study looping on Mixture-of-Experts Transformers while cl...

</details>

<details>
<summary><b>5. Evaluating Multimodal LLMs as Generalist Vision-Language-Action Agents for Drone Control: Commanding, Approaching, Tracking and Searching</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01404) • [📄 arXiv](https://arxiv.org/abs/2609.01404) • [📥 PDF](https://arxiv.org/pdf/2609.01404)

**💻 Code:** [⭐ Code](https://github.com/naver-ai/DroneCATS) • [⭐ Code](https://github.com/huggingface)

> Can an off-the-shelf multimodal LLM work as a VLA agent for drones? We drop MLLMs directly into a drone’s control loop with the entire action space declared in the prompt: point and go, rotate to search, think, and declare arrival. No fine-tuning,...

</details>

<details>
<summary><b>6. ZimaBlue: Evolving Generalizable World Action Models through Scalable Video Pre-training</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.00188) • [📄 arXiv](https://arxiv.org/abs/2609.00188) • [📥 PDF](https://arxiv.org/pdf/2609.00188)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZimaBlue-WAM/ZimaBlue)

> We present ZimaBlue, framing video scaling as a practical route toward generalizable World Action Models, and provide empirical evidence that scaling video pre-training significantly improves zero-shot generalization and strengthens performance on...

</details>

<details>
<summary><b>7. H3-World: Turning Language Understanding into World Control</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Yeying Jin, Xingyi Yang, Ziyue Lin, Zeqing Wang, Danze Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01560) • [📄 arXiv](https://arxiv.org/abs/2609.01560) • [📥 PDF](https://arxiv.org/pdf/2609.01560)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Danzer1xxxxChan/H3-World)

> 🚀 H3-World: Turning Language Understanding into World Control! Instead of learning a new action interface from scratch, H3-World turns keyboard controls into language that MiniMax-H3 already understands, then grounds each instruction to the right ...

</details>

<details>
<summary><b>8. UI-Venus-2 Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.00028) • [📄 arXiv](https://arxiv.org/abs/2609.00028) • [📥 PDF](https://arxiv.org/pdf/2609.00028)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. EM^2Mem: Event-Centric Multimodal Memory for Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.00551) • [📄 arXiv](https://arxiv.org/abs/2609.00551) • [📥 PDF](https://arxiv.org/pdf/2609.00551)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zjunlp/LightMem%7D)

> Multimodal memory offers a scalable interface for long-video question answering, but existing methods often retrieve captions, frames, transcripts, summaries, or graph facts as isolated fragments. Although searchable, such fragments are not genera...

</details>

<details>
<summary><b>10. Control-Data Flow Separation: Stable Prompt Optimization in Multi-Agent LLMs</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.00621) • [📄 arXiv](https://arxiv.org/abs/2609.00621) • [📥 PDF](https://arxiv.org/pdf/2609.00621)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yuntian-group/cdsep)

> A simple problem we ran into: prompt optimization can improve a multi-agent system while quietly breaking its routing, formatting, or termination logic. We separate control flow into typed program objects and only optimize the natural-language dat...

</details>

<details>
<summary><b>11. Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.00111) • [📄 arXiv](https://arxiv.org/abs/2609.00111) • [📥 PDF](https://arxiv.org/pdf/2609.00111)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. The Mechanics of Democratic Dominance: A System Dynamics Paradigm for Dynamic Consent Engineering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Muhammad Sukri Bin Ramli

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27509) • [📄 arXiv](https://arxiv.org/abs/2608.27509) • [📥 PDF](https://arxiv.org/pdf/2608.27509)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Dynamic Democratic Consent: A System Dynamics Framework for Public Trust, Political Communication, and Policy Feedback

</details>

<details>
<summary><b>13. Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chen Zhang, Zhanhao Li, Hangfan Zhang, Min-le Su, Haoyang Yan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01481) • [📄 arXiv](https://arxiv.org/abs/2609.01481) • [📥 PDF](https://arxiv.org/pdf/2609.01481)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. E-Commerce Bench: Evaluating LLM Agents on Long-Horizon Autonomous Business Operation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.30730) • [📄 arXiv](https://arxiv.org/abs/2608.30730) • [📥 PDF](https://arxiv.org/pdf/2608.30730)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/QwenLM/E-CommerceBench)

> E-Commerce Bench: Evaluating LLM Agents on Long-Horizon Autonomous Business Operation Presented as one of the demos in the Qwen3.8-Max Blog: https://qwen.ai/blog?id=qwen3.8

</details>

<details>
<summary><b>15. InternReviewer & InternAdvocate: Objective Reward and Evaluation for Agentic Reinforcement Learning in Peer Review and Rebuttal</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.28612) • [📄 arXiv](https://arxiv.org/abs/2608.28612) • [📥 PDF](https://arxiv.org/pdf/2608.28612)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/openreview/openreview-py)

> Generating professional scholarly content, such as peer reviews and rebuttals, requires an intricate synergy between domain reasoning and factual grounding. This work presents a comprehensive framework for the development and evaluation of special...

</details>

<details>
<summary><b>16. ReFlowSET: Representation-Aligned Latent Flow Matching for SAR-to-EO Image Translation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.00968) • [📄 arXiv](https://arxiv.org/abs/2609.00968) • [📥 PDF](https://arxiv.org/pdf/2609.00968)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/KAIST-VICLab/ReFlowSET)

> ReFlowSET, a representation-aligned latent flow-matching framework for SAR-to-EO image translation with code & checkpoints for all benchmarked methods fully released!

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 16 |
| 📅 Today | [`2026-09-02.json`](data/daily/2026-09-02.json) | 16 |
| 📆 This Week | [`2026-W35.json`](data/weekly/2026-W35.json) | 35 |
| 🗓️ This Month | [`2026-09.json`](data/monthly/2026-09.json) | 23 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-09-02 | 16 | [View JSON](data/daily/2026-09-02.json) |
| 📄 2026-09-01 | 7 | [View JSON](data/daily/2026-09-01.json) |
| 📄 2026-08-31 | 12 | [View JSON](data/daily/2026-08-31.json) |
| 📄 2026-08-30 | 23 | [View JSON](data/daily/2026-08-30.json) |
| 📄 2026-08-29 | 23 | [View JSON](data/daily/2026-08-29.json) |
| 📄 2026-08-28 | 18 | [View JSON](data/daily/2026-08-28.json) |
| 📄 2026-08-27 | 26 | [View JSON](data/daily/2026-08-27.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W35 | 35 | [View JSON](data/weekly/2026-W35.json) |
| 📅 2026-W34 | 173 | [View JSON](data/weekly/2026-W34.json) |
| 📅 2026-W33 | 213 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-09 | 23 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |

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
