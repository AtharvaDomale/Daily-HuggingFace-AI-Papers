<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4096+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">18</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">85</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">475</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4096+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 22, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. π-Bench: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14678) • [📄 arXiv](https://arxiv.org/abs/2605.14678) • [📥 PDF](https://arxiv.org/pdf/2605.14678)

**💻 Code:** [⭐ Code](https://github.com/Simplified-Reasoning/Pi-Bench) • [⭐ Code](https://github.com/huggingface)

> The rise of personal assistant agents, e.g., OpenClaw, highlights the growing potential of large language models to support users across everyday life and work. A core challenge in these settings is proactive assistance, since users often begin wi...

</details>

<details>
<summary><b>2. ACC: Compiling Agent Trajectories for Long-Context Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21850) • [📄 arXiv](https://arxiv.org/abs/2605.21850) • [📥 PDF](https://arxiv.org/pdf/2605.21850)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Recent development of agents has renewed demand for long-context reasoning capacity of LLMs. However, training LLMs for this capacity requires costly long-document curation or heuristic context synthesis. We observe that agents produce massive tra...

</details>

<details>
<summary><b>3. WorldKV: Efficient World Memory with World Retrieval and Compression</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22718) • [📄 arXiv](https://arxiv.org/abs/2605.22718) • [📥 PDF](https://arxiv.org/pdf/2605.22718)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project Page: https://cvlab-kaist.github.io/WorldKV/

</details>

<details>
<summary><b>4. Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22642) • [📄 arXiv](https://arxiv.org/abs/2605.22642) • [📥 PDF](https://arxiv.org/pdf/2605.22642)

**💻 Code:** [⭐ Code](https://github.com/Spreadsheet-RL/Spreadsheet-RL) • [⭐ Code](https://github.com/huggingface)

> Spreadsheet-RL is an RL fine-tuning framework and benchmarking environment designed to improve LLM agent performance on complex, multi-step spreadsheet tasks within Microsoft Excel.

</details>

<details>
<summary><b>5. FlowLong: Inference-time Long Video Generation via Manifold-constrained Tweedie Matching</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20910) • [📄 arXiv](https://arxiv.org/abs/2605.20910) • [📥 PDF](https://arxiv.org/pdf/2605.20910)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jhq1234/flowlong)

> FlowLong is a training-free, model-agnostic inference-time method that extends pretrained flow-based video diffusion models beyond their native generation horizon — works uniformly for text-to-video, audio-video joint, and text-to-3D scene generat...

</details>

<details>
<summary><b>6. Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Zhengxi Lu, Yuhao Shen, Ruihan Jin, Guocheng Zhai, Jinyang Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22177) • [📄 arXiv](https://arxiv.org/abs/2605.22177) • [📥 PDF](https://arxiv.org/pdf/2605.22177)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jinyangwu/Maestro)

> No abstract available.

</details>

<details>
<summary><b>7. Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Songyou Peng, Vincent Casser, Yijing Bai, Bo Sun, Jiahao Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22809) • [📄 arXiv](https://arxiv.org/abs/2605.22809) • [📥 PDF](https://arxiv.org/pdf/2605.22809)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Accepted by CVPR 2026

</details>

<details>
<summary><b>8. Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22109) • [📄 arXiv](https://arxiv.org/abs/2605.22109) • [📥 PDF](https://arxiv.org/pdf/2605.22109)

**💻 Code:** [⭐ Code](https://github.com/kkkcx/MM-OCEAN) • [⭐ Code](https://github.com/huggingface)

> 👋 Hello AI Community! We are the authors of Perception or Prejudice . Do Multimodal LLMs truly understand human personality, or do they just get the right score for the wrong reason? 🤔 Our paper introduces MM-OCEAN , a multi-granularity benchmark ...

</details>

<details>
<summary><b>9. Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22791) • [📄 arXiv](https://arxiv.org/abs/2605.22791) • [📥 PDF](https://arxiv.org/pdf/2605.22791)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NVlabs/GatedDeltaNet-2)

> Gated DeltaNet-2 introduces a linear attention architecture that improves memory management by decoupling erase and write operations, achieving superior performance on long-context benchmarks compared to existing recurrent models.

</details>

<details>
<summary><b>10. KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13734) • [📄 arXiv](https://arxiv.org/abs/2605.13734) • [📥 PDF](https://arxiv.org/pdf/2605.13734)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hpdps-group/KVServe)

> We’re excited to share that KVServe has been accepted by SIGCOMM 2026 🎉 KVServe is a service-aware KV cache compression framework for communication-efficient disaggregated LLM serving. It treats KV compression as a runtime strategy selection probl...

</details>

<details>
<summary><b>11. Swift Sampling: Selecting Temporal Surprises via Taylor Series</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22678) • [📄 arXiv](https://arxiv.org/abs/2605.22678) • [📥 PDF](https://arxiv.org/pdf/2605.22678)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> While most frames in long-form video are redundant, the critical information resides in temporal surprises: moments where the actual visual features deviate from their predicted evolution. Inspired by the human brain’s predictive coding, we introd...

</details>

<details>
<summary><b>12. Diversed Model Discovery via Structured Table Discovery</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22766) • [📄 arXiv](https://arxiv.org/abs/2605.22766) • [📥 PDF](https://arxiv.org/pdf/2605.22766)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RJMillerLab/ModelSearch)

> Diversed Model Discovery via Structured Table Discovery

</details>

<details>
<summary><b>13. SceneAligner: 3D-Grounded Floorplan Localization in the Wild</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22581) • [📄 arXiv](https://arxiv.org/abs/2605.22581) • [📥 PDF](https://arxiv.org/pdf/2605.22581)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Cornell-VAILab/SceneAligner)

> Project Page: https://Cornell-VAILab.github.io/SceneAligner

</details>

<details>
<summary><b>14. Bernini: Latent Semantic Planning for Video Diffusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22344) • [📄 arXiv](https://arxiv.org/abs/2605.22344) • [📥 PDF](https://arxiv.org/pdf/2605.22344)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Bernini is a unified framework for video generation and editing that uses an MLLM-based planner to predict latent semantic representations for a DiT-based renderer.

</details>

<details>
<summary><b>15. Training Large Language Models to Predict Clinical Events</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.12817) • [📄 arXiv](https://arxiv.org/abs/2605.12817) • [📥 PDF](https://arxiv.org/pdf/2605.12817)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Longitudinal clinical notes contain rich evidence of how patients evolve over time, but converting this signal into training supervision for clinical prediction remains challenging. We extend Foresight Learning to clinical prediction by converting...

</details>

<details>
<summary><b>16. From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shenzhi Wang, Yang Yue, Wenze Lin, Zihan Tang, Xitai Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22074) • [📄 arXiv](https://arxiv.org/abs/2605.22074) • [📥 PDF](https://arxiv.org/pdf/2605.22074)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Introduces SCRL, a curriculum reinforcement learning framework for LLM reasoning that derives verifiable subproblems from reasoning chains to enable fine-grained, subproblem-level credit assignment without external reward models.

</details>

<details>
<summary><b>17. TerminalWorld: Benchmarking Agents on Real-World Terminal Tasks</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Han Li, Pengyu Zou, Xingyu Jiang, Jiarui Hu, Zhaoyang Chu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22535) • [📄 arXiv](https://arxiv.org/abs/2605.22535) • [📥 PDF](https://arxiv.org/pdf/2605.22535)

**💻 Code:** [⭐ Code](https://github.com/EuniAI/TerminalWorld) • [⭐ Code](https://github.com/huggingface)

> TerminalWorld is a scalable data engine that reverse-engineers real-world terminal recordings into a benchmark of 1,530 validated tasks to evaluate agent performance on authentic software engineering terminal workflows.

</details>

<details>
<summary><b>18. Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21803) • [📄 arXiv](https://arxiv.org/abs/2605.21803) • [📥 PDF](https://arxiv.org/pdf/2605.21803)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Same Architecture, Different Optimizer, Different Capacity: In this work, we have shown that realized representation capacity is not architecture-only, it emerges from the architecture-optimizer interaction. Optimizer geometry changes the scaling ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-05-22.json`](data/daily/2026-05-22.json) | 18 |
| 📆 This Week | [`2026-W20.json`](data/weekly/2026-W20.json) | 85 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 475 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-22 | 18 | [View JSON](data/daily/2026-05-22.json) |
| 📄 2026-05-21 | 18 | [View JSON](data/daily/2026-05-21.json) |
| 📄 2026-05-20 | 16 | [View JSON](data/daily/2026-05-20.json) |
| 📄 2026-05-19 | 15 | [View JSON](data/daily/2026-05-19.json) |
| 📄 2026-05-18 | 18 | [View JSON](data/daily/2026-05-18.json) |
| 📄 2026-05-17 | 53 | [View JSON](data/daily/2026-05-17.json) |
| 📄 2026-05-16 | 53 | [View JSON](data/daily/2026-05-16.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W20 | 85 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 475 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |
| 🗓️ 2025-12 | 787 | [View JSON](data/monthly/2025-12.json) |

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
