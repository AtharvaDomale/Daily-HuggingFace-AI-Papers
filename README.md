<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-14-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6373+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">48</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">245</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6373+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** September 18, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. An Empirical Study of Harness Design for Coding Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.20804) • [📄 arXiv](https://arxiv.org/abs/2609.20804) • [📥 PDF](https://arxiv.org/pdf/2609.20804)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Coding agents rely on a harness to plan, use tools, and manage context. But which of these components actually help, and when? Most evaluations compare complete agent systems, making it difficult to separate the contributions of individual harness...

</details>

<details>
<summary><b>2. JEPA-Anything: Learning Predictive Models across Different Worlds</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Zhaochen Yu, Weiyang Liu, Xinyue Xu, Zhongyao Wang, Taoyong Cui

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.20800) • [📄 arXiv](https://arxiv.org/abs/2609.20800) • [📥 PDF](https://arxiv.org/pdf/2609.20800)

**💻 Code:** [⭐ Code](https://github.com/Gen-Verse/JEPA-Anything) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>3. RiskChainBench: A Benchmark for Obfuscated Platform Message Restoration and Evidence-Grounded Web Investigation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yifan Wang, Mengzheng Yang, Ying Zhang, Zhiming Ma, leonliuzx

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.16900) • [📄 arXiv](https://arxiv.org/abs/2609.16900) • [📥 PDF](https://arxiv.org/pdf/2609.16900)

**💻 Code:** [⭐ Code](https://github.com/mattheliu/riskchainbench-task1) • [⭐ Code](https://github.com/mattheliu/riskchainbench-task2) • [⭐ Code](https://github.com/huggingface)

> We introduce RiskChainBench, a benchmark connecting obfuscated-message restoration with evidence-grounded web investigation. The paper evaluates 3,600 synthetic restoration inputs paired with 600 controlled web environments across ten models. Rest...

</details>

<details>
<summary><b>4. SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness</b> ⭐ 2.19k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.20519) • [📄 arXiv](https://arxiv.org/abs/2609.20519) • [📥 PDF](https://arxiv.org/pdf/2609.20519)

**💻 Code:** [⭐ Code](https://github.com/NVlabs/SoL-Pi) • [⭐ Code](https://github.com/huggingface)

> As coding agents move from supervised code completion to unattended, around-the-clock exploration, their work expands from isolated predictions into long trajectories of reasoning, tool use, and feedback. Token efficiency therefore becomes importa...

</details>

<details>
<summary><b>5. DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.19969) • [📄 arXiv](https://arxiv.org/abs/2609.19969) • [📥 PDF](https://arxiv.org/pdf/2609.19969)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>6. VABench: Measuring Embodied Spatial Intelligence through Visual Demonstrations, Active Perception, and Metric Control</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.19554) • [📄 arXiv](https://arxiv.org/abs/2609.19554) • [📥 PDF](https://arxiv.org/pdf/2609.19554)

**💻 Code:** [⭐ Code](https://github.com/zhangzhongbo2213/VABench) • [⭐ Code](https://github.com/huggingface)

> Spatial intelligence requires more than describing object locations. Under incomplete observation, models must identify and acquire missing evidence, interpret it in a common spatial frame, and act on it. We introduce VA-BENCH to evaluate the co...

</details>

<details>
<summary><b>7. WeVisDoc: From Coverage to Capability for Robust End-to-End Document Parsing</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.20423) • [📄 arXiv](https://arxiv.org/abs/2609.20423) • [📥 PDF](https://arxiv.org/pdf/2609.20423)

**💻 Code:** [⭐ Code](https://github.com/Tencent/WeVisDoc) • [⭐ Code](https://github.com/huggingface)

> Document parsing converts document images into structured content and requires reliable performance across diverse layouts and acquisition conditions. Yet training corpora are biased toward common document types and clean digital pages, while expa...

</details>

<details>
<summary><b>8. RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aozhe Wang, Yichen Pan, Yizhou Liu, Zhengxi Lu, Yan Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.20784) • [📄 arXiv](https://arxiv.org/abs/2609.20784) • [📥 PDF](https://arxiv.org/pdf/2609.20784)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/SDAR) • [⭐ Code](https://github.com/huggingface)

> Rather than following a predefined distillation schedule, RetireOPD adopts Adaptive Retirement: the student drops the teacher on its own once their discrepancy stops shrinking and it reaches a target fraction of the teacher's success rate, after w...

</details>

<details>
<summary><b>9. When2Think: Learning Difficulty-Aware Length Control for Efficient Hybrid Reasoning Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.19671) • [📄 arXiv](https://arxiv.org/abs/2609.19671) • [📥 PDF](https://arxiv.org/pdf/2609.19671)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Efficient Reasoning models by RFT

</details>

<details>
<summary><b>10. Reflect, Revise, Reuse: Training-Free Skill Evolution for GUI Agents</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Yong Du, Zhengxi Lu, Fei Tang, Boxuan Zhang, Bofan Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.17653) • [📄 arXiv](https://arxiv.org/abs/2609.17653) • [📥 PDF](https://arxiv.org/pdf/2609.17653)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/EvoSkill-GUI) • [⭐ Code](https://github.com/huggingface)

> We propose EvoSkill-GUI, a training-free framework in which each skill is a structured multi-file package containing retrieval metadata, executable plans, backup localization, failure-recovery rules, accessibility utilities, and failure cases.

</details>

<details>
<summary><b>11. What Does Privileged Information Add to On-Policy Self-Distillation?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.20612) • [📄 arXiv](https://arxiv.org/abs/2609.20612) • [📥 PDF](https://arxiv.org/pdf/2609.20612)

**💻 Code:** [⭐ Code](https://github.com/xiuyuz/opsd-reference-study) • [⭐ Code](https://github.com/huggingface)

> Giving the teacher more of the solution does not necessarily give the student more to learn. We introduce AMPLE-Math ( https://huggingface.co/datasets/xiuyuz/ample-math ) to isolate what privileged references add beyond the existing asymmetry in t...

</details>

<details>
<summary><b>12. UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.12397) • [📄 arXiv](https://arxiv.org/abs/2609.12397) • [📥 PDF](https://arxiv.org/pdf/2609.12397)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Multi‑modal image generation, particularly subject‑driven customization, has garnered growing attention in recent years. Despite the rapid advancement of generative models, their evaluation remains largely lagging. Existing methods, whether embedd...

</details>

<details>
<summary><b>13. Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Michael Liu, Yiwen Zhang, Hexu Zhao, Yiming Xie, Haocheng Xi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.20744) • [📄 arXiv](https://arxiv.org/abs/2609.20744) • [📥 PDF](https://arxiv.org/pdf/2609.20744)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. Region-Level Policy Optimization for Fine-grained MLLM Perception</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chang Xu, Minjing Dong, Xiaohuan Pei, Yuheng Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.19745) • [📄 arXiv](https://arxiv.org/abs/2609.19745) • [📥 PDF](https://arxiv.org/pdf/2609.19745)

**💻 Code:** [⭐ Code](https://github.com/YuHengsss/VisionRL2) • [⭐ Code](https://github.com/huggingface)

> Vision-RL²: Region-Level Policy Optimization for Fine-grained MLLM Perception Fine-grained perception in MLLMs usually means raising the resolution, which inflates visual-token and prefill cost. We show that localizing the region of interest toler...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 14 |
| 📅 Today | [`2026-09-18.json`](data/daily/2026-09-18.json) | 14 |
| 📆 This Week | [`2026-W37.json`](data/weekly/2026-W37.json) | 48 |
| 🗓️ This Month | [`2026-09.json`](data/monthly/2026-09.json) | 245 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-09-18 | 14 | [View JSON](data/daily/2026-09-18.json) |
| 📄 2026-09-17 | 6 | [View JSON](data/daily/2026-09-17.json) |
| 📄 2026-09-16 | 11 | [View JSON](data/daily/2026-09-16.json) |
| 📄 2026-09-15 | 11 | [View JSON](data/daily/2026-09-15.json) |
| 📄 2026-09-14 | 6 | [View JSON](data/daily/2026-09-14.json) |
| 📄 2026-09-13 | 26 | [View JSON](data/daily/2026-09-13.json) |
| 📄 2026-09-12 | 26 | [View JSON](data/daily/2026-09-12.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W37 | 48 | [View JSON](data/weekly/2026-W37.json) |
| 📅 2026-W36 | 88 | [View JSON](data/weekly/2026-W36.json) |
| 📅 2026-W35 | 121 | [View JSON](data/weekly/2026-W35.json) |
| 📅 2026-W34 | 173 | [View JSON](data/weekly/2026-W34.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-09 | 245 | [View JSON](data/monthly/2026-09.json) |
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
