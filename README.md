<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-14-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7795+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">28</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">67</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7795+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 03, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Wave Forcing: Towards Speed-of-Light Streaming Video Generation</b> ⭐ 68</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.00859) • [📄 arXiv](https://arxiv.org/abs/2611.00859) • [📥 PDF](https://arxiv.org/pdf/2611.00859)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SJTU-DENG-Lab/WaveForcing)

> We introduce Wave Forcing, a training and execution framework for streaming video generation. The key idea is wavefront scheduling: later video chunks begin denoising before earlier chunks finish, using their cleaner intermediate states as causal ...

</details>

<details>
<summary><b>2. Video Generation Meets Agent Harness: Agentic Video Generation, Agentic World Modeling and Recursive Self-Improvement</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40191) • [📄 arXiv](https://arxiv.org/abs/2610.40191) • [📥 PDF](https://arxiv.org/pdf/2610.40191)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/agentic-videogen/awesome-agentic-videogen)

> Video generation is evolving from end-to-end synthesis toward agentic systems that integrate planning, tool use, and feedback. Meanwhile, video world models are enabling interactive, action-conditioned environments for agents. In this survey, we e...

</details>

<details>
<summary><b>3. Democratizing Robot-Use Agents: A Unified Harness with Open Models</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Kwanyoung Park, Wenbo Zhang, Xiaoyu Huang, Yuxin Zuo, Jiarui Yuan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.01560) • [📄 arXiv](https://arxiv.org/abs/2611.01560) • [📥 PDF](https://arxiv.org/pdf/2611.01560)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Open-RUA/uni-robot-agent)

> We introduce uni-robot-agent, a unified harness for robot-use agents with open-weight models, designed to support further training. Without privileged inputs, out-of-the-box Qwen3.8-Flash-Next achieves a RoboDojo Score of 31.39, exceeding the publ...

</details>

<details>
<summary><b>4. From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.00172) • [📄 arXiv](https://arxiv.org/abs/2611.00172) • [📥 PDF](https://arxiv.org/pdf/2611.00172)

**💻 Code:** [⭐ Code](https://github.com/codefuse-ai/Awesome-Omnimodal-Embeddings) • [⭐ Code](https://github.com/huggingface)

> Despite shared mathematical foundations, embedding research across text, vision, audio, graphs, tables, and time series remains largely fragmented into modality-specific silos. This survey and position paper argues that embedding research is quiet...

</details>

<details>
<summary><b>5. NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02102) • [📄 arXiv](https://arxiv.org/abs/2611.02102) • [📥 PDF](https://arxiv.org/pdf/2611.02102)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/beipo-ai/NorthCUA)

> Worlds in mind. Wonders at your fingertips. Turn intent into work — across Desktop and Web

</details>

<details>
<summary><b>6. Diagnosing the Reference Gap: What Visual Scaffolds Can and Cannot Fix in Multimodal Reasoning</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Hou Pong Chan, Anh Tuan Luu, Wotao Yin, Weiwen Xu, Guizhen Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39419) • [📄 arXiv](https://arxiv.org/abs/2610.39419) • [📥 PDF](https://arxiv.org/pdf/2610.39419)

**💻 Code:** [⭐ Code](https://github.com/regap-bench/ReGap-Bench) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. DeFT-RAG: Decomposing Multi-Hop Questions into Self-Contained Sub-Questions with Reusable Answers</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Bingzhe Li, Dingy-Kang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.37826) • [📄 arXiv](https://arxiv.org/abs/2610.37826) • [📥 PDF](https://arxiv.org/pdf/2610.37826)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Dingyi-Kang/DeFT-RAG)

> DeFT-RAG asks a different question about multi-hop RAG: instead of planning all sub-questions and their dependencies in advance, what if the LLM writes just one self-contained sub-question at a time? Each sub-question names every entity it asks ab...

</details>

<details>
<summary><b>8. DecisionRank: On-Policy Distillation for Decision-Model Rerankers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.42128) • [📄 arXiv](https://arxiv.org/abs/2610.42128) • [📥 PDF](https://arxiv.org/pdf/2610.42128)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Decision models target structured tasks by directly predicting choices or scores, avoiding autoregressive decoding latency. However, transferring a generative LLM's ranking knowledge to a decision model remains challenging: the teacher expresses p...

</details>

<details>
<summary><b>9. Lost in Recovery: Demystifying Recovery Behavior of Computer-Use Agents via Goal-Oriented Action Replay</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27954) • [📄 arXiv](https://arxiv.org/abs/2610.27954) • [📥 PDF](https://arxiv.org/pdf/2610.27954)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/BangProx/ReTRACE)

> We investigate the recovery problem in computer-use agents and present RETRACE, a scalable pipeline for conducting the first systematic study of autonomous, model-based recovery behavior.

</details>

<details>
<summary><b>10. TinyAstral: Transferring Self-Evolving Harnesses to Tiny Generalist Manipulation Agents</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Mengqi Guo, Seunghun Lee, Guanli Hou, Xiaoxu Zheng, Kehong Gong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.41216) • [📄 arXiv](https://arxiv.org/abs/2610.41216) • [📥 PDF](https://arxiv.org/pdf/2610.41216)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Garfield-kh/TinyAstral)

> TinyAstral evolves the harness (tools, feedback, memory, recovery) of a frozen ~180B robot teacher through recursive self-improvement (33.4% → 63.6% on RoboTwin), then distills it into 0.8B–4B students that run in the same harness. The 4B student ...

</details>

<details>
<summary><b>11. MAServeBench: Benchmarking Multi-Agent LLM Serving Systems with Hierarchical Collaboration Workloads</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40293) • [📄 arXiv](https://arxiv.org/abs/2610.40293) • [📥 PDF](https://arxiv.org/pdf/2610.40293)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ShengxuanQiu/MAServeBench)

> Multi-agent collaboration fundamentally reshapes LLM serving workloads through dynamic dependencies, synchronization, and context sharing—beyond what conventional request-level benchmarks can capture. We introduce MAServeBench , a benchmark featur...

</details>

<details>
<summary><b>12. Self-Improvement with Auxiliary Rewards: Learning from Verified Traces and Evaluating Transfer</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.41296) • [📄 arXiv](https://arxiv.org/abs/2610.41296) • [📥 PDF](https://arxiv.org/pdf/2610.41296)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Pushing self-improvement in small language models with auxiliary rewards

</details>

<details>
<summary><b>13. On-Policy Self-Distillation with Teacher Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38637) • [📄 arXiv](https://arxiv.org/abs/2610.38637) • [📥 PDF](https://arxiv.org/pdf/2610.38637)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We wanted to solve a frustrating problem in On-Policy Self-Distillation (OPSD): the "self-teacher" model often gets blinded by the privileged context and gives pretty unreliable, overconfident token-level guidance, which ends up misleading the stu...

</details>

<details>
<summary><b>14. ProVe: Proposer-Verifier Decoding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qi She, Yujie Zhong, Jiaxi Li, Weiwen Xu, Ziqi Jin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27815) • [📄 arXiv](https://arxiv.org/abs/2610.27815) • [📥 PDF](https://arxiv.org/pdf/2610.27815)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Speculative decoding lets a small model draft and a large model verify. ProVe reverses the roles: the large model proposes a block of 16 tokens in one forward pass, and a verifier reads it in one more.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 14 |
| 📅 Today | [`2026-11-03.json`](data/daily/2026-11-03.json) | 14 |
| 📆 This Week | [`2026-W44.json`](data/weekly/2026-W44.json) | 28 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 67 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-03 | 14 | [View JSON](data/daily/2026-11-03.json) |
| 📄 2026-11-02 | 14 | [View JSON](data/daily/2026-11-02.json) |
| 📄 2026-11-01 | 39 | [View JSON](data/daily/2026-11-01.json) |
| 📄 2026-10-31 | 39 | [View JSON](data/daily/2026-10-31.json) |
| 📄 2026-10-30 | 10 | [View JSON](data/daily/2026-10-30.json) |
| 📄 2026-10-29 | 5 | [View JSON](data/daily/2026-10-29.json) |
| 📄 2026-10-28 | 27 | [View JSON](data/daily/2026-10-28.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W44 | 28 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 67 | [View JSON](data/monthly/2026-11.json) |
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
