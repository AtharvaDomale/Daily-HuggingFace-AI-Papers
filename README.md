<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-655+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">101</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">704</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">655+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 25, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. SemanticGen: Video Generation in Semantic Space</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20619) • [📄 arXiv](https://arxiv.org/abs/2512.20619) • [📥 PDF](https://arxiv.org/pdf/2512.20619)

> Project Page: https://jianhongbai.github.io/SemanticGen/

</details>

<details>
<summary><b>2. Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19673) • [📄 arXiv](https://arxiv.org/abs/2512.19673) • [📥 PDF](https://arxiv.org/pdf/2512.19673)

**💻 Code:** [⭐ Code](https://github.com/Trae1ounG/BuPO)

> Bottom-up Policy Optimization (BuPO) provides a novel framework to decompose LLM policies into internal layer and modular policies, reveals distinct reasoning patterns across different model architectures, and introduces a bottom-up optimization a...

</details>

<details>
<summary><b>3. LongVideoAgent: Multi-Agent Reasoning with Long Videos</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Renjie Pi, Yue Ma, Jiaqi Tang, Ziyi Liu, Runtao Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20618) • [📄 arXiv](https://arxiv.org/abs/2512.20618) • [📥 PDF](https://arxiv.org/pdf/2512.20618)

> Recent advances in multimodal LLMs and systems that use tools for long-video QA point to the promise of reasoning over hour-long episodes. However, many methods still compress content into lossy summaries or rely on limited toolsets, weakening tem...

</details>

<details>
<summary><b>4. SpatialTree: How Spatial Abilities Branch Out in MLLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20617) • [📄 arXiv](https://arxiv.org/abs/2512.20617) • [📥 PDF](https://arxiv.org/pdf/2512.20617)

> Introducing SpatialTree, a four-level hierarchy for spatial abilities in multimodal LLMs, benchmark 27 sub-abilities, reveal transfer patterns, and propose auto-think to improve reinforcement-learning performance.

</details>

<details>
<summary><b>5. MemEvolve: Meta-Evolution of Agent Memory Systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junhao Wang, Zhenhong Zhou, Chong Zhan, Haotian Ren, Guibin Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18746) • [📄 arXiv](https://arxiv.org/abs/2512.18746) • [📥 PDF](https://arxiv.org/pdf/2512.18746)

> Self-evolving memory systems are unprecedentedly reshaping the evolutionary paradigm of large language model (LLM)-based agents. Prior work has predominantly relied on manually engineered memory architectures to store trajectories, distill experie...

</details>

<details>
<summary><b>6. Step-DeepResearch Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20491) • [📄 arXiv](https://arxiv.org/abs/2512.20491) • [📥 PDF](https://arxiv.org/pdf/2512.20491)

> As LLMs shift toward autonomous agents, Deep Research has emerged as a pivotal metric. However, existing academic benchmarks like BrowseComp often fail to meet real-world demands for open-ended research, which requires robust skills in intent reco...

</details>

<details>
<summary><b>7. Reinforcement Learning for Self-Improving Agent with Skill Library</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Soumya Smruti Mishra, Yijun Tian, Yawei Wang, Qiaojing Yan, Jiongxiao Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17102) • [📄 arXiv](https://arxiv.org/abs/2512.17102) • [📥 PDF](https://arxiv.org/pdf/2512.17102)

> Apply RL to Skill Library Agent.

</details>

<details>
<summary><b>8. SAM Audio: Segment Anything in Audio</b> ⭐ 2.49k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18099) • [📄 arXiv](https://arxiv.org/abs/2512.18099) • [📥 PDF](https://arxiv.org/pdf/2512.18099)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/sam-audio)

> No abstract available.

</details>

<details>
<summary><b>9. INTELLECT-3: Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16144) • [📄 arXiv](https://arxiv.org/abs/2512.16144) • [📥 PDF](https://arxiv.org/pdf/2512.16144)

> INTELLECT-3: Technical Report

</details>

<details>
<summary><b>10. FaithLens: Detecting and Explaining Faithfulness Hallucination</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20182) • [📄 arXiv](https://arxiv.org/abs/2512.20182) • [📥 PDF](https://arxiv.org/pdf/2512.20182)

**💻 Code:** [⭐ Code](https://github.com/S1s-Z/FaithLens)

> In this paper, we introduce FaithLens, a cost-efficient and effective faithfulness hallucination detection model that can jointly provide binary predictions and corresponding explanations to improve trustworthiness.

</details>

<details>
<summary><b>11. Scaling Laws for Code: Every Programming Language Matters</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aishan Liu, Wei Zhang, Lin Jing, Shawn Guo, Jian Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13472) • [📄 arXiv](https://arxiv.org/abs/2512.13472) • [📥 PDF](https://arxiv.org/pdf/2512.13472)

> Code large language models (Code LLMs) are powerful but costly to train, with scaling laws predicting performance from model size, data, and compute. However, different programming languages (PLs) have varying impacts during pre-training that sign...

</details>

<details>
<summary><b>12. QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19526) • [📄 arXiv](https://arxiv.org/abs/2512.19526) • [📥 PDF](https://arxiv.org/pdf/2512.19526)

> QuantiPhy is the first benchmark that asks vision–language models to do physics with numerical accuracy. Across 3,300+ video–text instances, we show that today’s VLMs often sound plausible but fail quantitatively on physical reasoning tasks—they r...

</details>

<details>
<summary><b>13. Simulstream: Open-Source Toolkit for Evaluation and Demonstration of Streaming Speech-to-Text Translation Systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Luisa Bentivogli, Matteo Negri, Mauro Cettolo, Marco Gaido, spapi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17648) • [📄 arXiv](https://arxiv.org/abs/2512.17648) • [📥 PDF](https://arxiv.org/pdf/2512.17648)

> Already available on PyPi at https://pypi.org/project/simulstream/

</details>

<details>
<summary><b>14. Active Intelligence in Video Avatars via Closed-loop World Modeling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Cheng Meng, Ruiqi Wu, Ke Cao, Tianyu Yang, Xuanhua He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20615) • [📄 arXiv](https://arxiv.org/abs/2512.20615) • [📥 PDF](https://arxiv.org/pdf/2512.20615)

> project page: https://xuanhuahe.github.io/ORCA/

</details>

<details>
<summary><b>15. Multi-LLM Thematic Analysis with Dual Reliability Metrics: Combining Cohen's Kappa and Semantic Similarity for Qualitative Research Validation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20352) • [📄 arXiv](https://arxiv.org/abs/2512.20352) • [📥 PDF](https://arxiv.org/pdf/2512.20352)

**💻 Code:** [⭐ Code](https://github.com/NileshArnaiya/LLM-Thematic-Analysis-Tool)

> Qualitative research faces a critical reliability challenge: traditional inter-rater agreement methods require multiple human coders, are time-intensive, and often yield moderate consistency. We present a multi-perspective validation framework for...

</details>

<details>
<summary><b>16. Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-session Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenyu Huang, Zhaowei Wang, Yifan Xiang, Baojun Wang, Yiming Du

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20092) • [📄 arXiv](https://arxiv.org/abs/2512.20092) • [📥 PDF](https://arxiv.org/pdf/2512.20092)

**💻 Code:** [⭐ Code](https://github.com/Elvin-Yiming-Du/Memory-T1/)

> Temporal reasoning over long, multi-session dialogues is a critical capability for conversational agents. However, existing works and our pilot study have shown that as dialogue histories grow in length and accumulate noise, current long-context m...

</details>

<details>
<summary><b>17. Toxicity Ahead: Forecasting Conversational Derailment on GitHub</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kostadin Damevski, Preetha Chatterjee, Rahat Rizvi Rahman, Robert Zita, imranraad

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15031) • [📄 arXiv](https://arxiv.org/abs/2512.15031) • [📥 PDF](https://arxiv.org/pdf/2512.15031)

> Toxic interactions in Open Source Software (OSS) communities reduce contributor engagement and threaten project sustainability. Preventing such toxicity before it emerges requires a clear understanding of how harmful conversations unfold. However,...

</details>

<details>
<summary><b>18. Learning to Refocus with Video Diffusion Models</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Shumian Xin, Xuaner Zhang, Zhoutong Zhang, SaiKiran Tedla

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19823) • [📄 arXiv](https://arxiv.org/abs/2512.19823) • [📥 PDF](https://arxiv.org/pdf/2512.19823)

**💻 Code:** [⭐ Code](https://github.com/tedlasai/learn2refocus)

> Learning to Refocus with Video Diffusion Models, SIGGRAPH Asia 2025

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2025-12-25.json`](data/daily/2025-12-25.json) | 18 |
| 📆 This Week | [`2025-W51.json`](data/weekly/2025-W51.json) | 101 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 704 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-25 | 18 | [View JSON](data/daily/2025-12-25.json) |
| 📄 2025-12-24 | 23 | [View JSON](data/daily/2025-12-24.json) |
| 📄 2025-12-23 | 22 | [View JSON](data/daily/2025-12-23.json) |
| 📄 2025-12-22 | 38 | [View JSON](data/daily/2025-12-22.json) |
| 📄 2025-12-21 | 38 | [View JSON](data/daily/2025-12-21.json) |
| 📄 2025-12-20 | 37 | [View JSON](data/daily/2025-12-20.json) |
| 📄 2025-12-19 | 30 | [View JSON](data/daily/2025-12-19.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W51 | 101 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 704 | [View JSON](data/monthly/2025-12.json) |

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
