<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4782+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">35</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">379</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4782+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 17, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. GameCraft-Bench: Can Agents Build Playable Games End-to-End in a Real Game Engine?</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17861) • [📄 arXiv](https://arxiv.org/abs/2606.17861) • [📥 PDF](https://arxiv.org/pdf/2606.17861)

**💻 Code:** [⭐ Code](https://github.com/tongxuluo/gamecraft-bench) • [⭐ Code](https://github.com/huggingface)

> Can coding agents build an actual game in a real game engine? We introduce GameCraft-Bench , a benchmark of 140 Godot tasks across 15 game families for evaluating end-to-end game generation through interactive gameplay verification. The strongest ...

</details>

<details>
<summary><b>2. ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17200) • [📄 arXiv](https://arxiv.org/abs/2606.17200) • [📥 PDF](https://arxiv.org/pdf/2606.17200)

**💻 Code:** [⭐ Code](https://github.com/ACERobotics-VLA/ACE-Ego-0) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>3. OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17628) • [📄 arXiv](https://arxiv.org/abs/2606.17628) • [📥 PDF](https://arxiv.org/pdf/2606.17628)

**💻 Code:** [⭐ Code](https://github.com/bingreeky/opd-evolver) • [⭐ Code](https://github.com/huggingface)

> We introduce OPD-Evolver, a slow-fast co-evolution framework that helps agents not only store experience, but learn how to select, use, write, and maintain it. Across multi-domain benchmarks, OPD-Evolver outperforms existing memory systems, skill-...

</details>

<details>
<summary><b>4. LectūraAgents: A Multi-Agent Framework for Adaptive Personalized AI-Assisted Learning and Embodied Teaching</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guangyao Chen, Yemin Shi, Siwei Dong, Yue Yu, Jaward Sesay

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16428) • [📄 arXiv](https://arxiv.org/abs/2606.16428) • [📥 PDF](https://arxiv.org/pdf/2606.16428)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Effective personalized AI-assisted learning demands learning systems that can not only generate accurate learner-specific educational materials, but also dynamically adapt their instruction to diverse learners. LectūraAgents is here to address suc...

</details>

<details>
<summary><b>5. TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09030) • [📄 arXiv](https://arxiv.org/abs/2606.09030) • [📥 PDF](https://arxiv.org/pdf/2606.09030)

**💻 Code:** [⭐ Code](https://github.com/HyeongWon-Jang/TRIAGE) • [⭐ Code](https://github.com/huggingface)

> Predicting clinical risk from irregularly sampled medical time series demands more than an accurate label. It calls for calibrated risk scores to prioritize patients for triage, along with interpretable rationales that clinicians can verify. Large...

</details>

<details>
<summary><b>6. Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.18216) • [📄 arXiv](https://arxiv.org/abs/2606.18216) • [📥 PDF](https://arxiv.org/pdf/2606.18216)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Our research question is "For hard questions, how can we transfer the teacher's knowledge to the student without imitating the teacher's logits or injecting the teacher's response directly into the student's gradient? In other words, how to make t...

</details>

<details>
<summary><b>7. Show the Signal, Hide the Noise: Spectral Forcing for Pixel-Space Diffusion</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.15236) • [📄 arXiv](https://arxiv.org/abs/2606.15236) • [📥 PDF](https://arxiv.org/pdf/2606.15236)

**💻 Code:** [⭐ Code](https://github.com/WeichenFan/Spectral_Forcing) • [⭐ Code](https://github.com/huggingface)

> Spectral Forcing makes pixel-space diffusion more efficient by filtering noisy high frequencies over time, improving ImageNet (JIT) and T2I benchmark (SenseNova-U1) without adding parameters.

</details>

<details>
<summary><b>8. A Gradient Perspective on RLVR Stability and Winner Advantage Policy Optimization</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16154) • [📄 arXiv](https://arxiv.org/abs/2606.16154) • [📥 PDF](https://arxiv.org/pdf/2606.16154)

**💻 Code:** [⭐ Code](https://github.com/layer6ai-labs/wapo) • [⭐ Code](https://github.com/huggingface)

> Rather than treating training collapse as a black box, the paper analyzes token-level gradient dynamics and derives a simple taxonomy showing that the effect of an update depends jointly on the advantage sign and the token's probability under the ...

</details>

<details>
<summary><b>9. ChLogic: Evaluating Robustness of Logical Reasoning in Chinese Expressions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bo Bai, Wei Han, Chaorui Zhang, Yuxu Chen, Peixian Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17905) • [📄 arXiv](https://arxiv.org/abs/2606.17905) • [📥 PDF](https://arxiv.org/pdf/2606.17905)

**💻 Code:** [⭐ Code](https://github.com/0328zpx/ChLogic) • [⭐ Code](https://github.com/huggingface)

> Large language models perform increasingly well on standardized logical reasoning benchmarks, but whether this ability remains robust beyond English is unclear. We introduce ChLogic, an English--Chinese aligned benchmark that tests whether models ...

</details>

<details>
<summary><b>10. ActWorld: From Explorable to Interactive World Model via Action-Aware Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17730) • [📄 arXiv](https://arxiv.org/abs/2606.17730) • [📥 PDF](https://arxiv.org/pdf/2606.17730)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>11. Variable-Width Transformers</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yury Polyanskiy, Rameswar Panda, Shawn Tan, Oliver Sieberling, Zhaofeng Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.18246) • [📄 arXiv](https://arxiv.org/abs/2606.18246) • [📥 PDF](https://arxiv.org/pdf/2606.18246)

**💻 Code:** [⭐ Code](https://github.com/ZhaofengWu/variable-width-transformers) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. Unified Multimodal Autoregressive Modeling with Shared Context-Visual Tokenizer is Key to Unification</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.18249) • [📄 arXiv](https://arxiv.org/abs/2606.18249) • [📥 PDF](https://arxiv.org/pdf/2606.18249)

**💻 Code:** [⭐ Code](https://github.com/ShareLab-SII/UniAR) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>13. ProCUA-SFT Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.17321) • [📄 arXiv](https://arxiv.org/abs/2606.17321) • [📥 PDF](https://arxiv.org/pdf/2606.17321)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A 3.1M sample synthetic dataset for training computer-use agents, significantly improving performance on desktop interaction tasks.

</details>

<details>
<summary><b>14. Beyond Monolingual Deep Research: Evaluating Agents and Retrievers with Cross-Lingual BrowseComp-Plus</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.15345) • [📄 arXiv](https://arxiv.org/abs/2606.15345) • [📥 PDF](https://arxiv.org/pdf/2606.15345)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce XBCP, a controlled benchmark for cross-lingual deep research, showing that language-mismatched evidence causes both retrieval failures and independent agent-side difficulties in evidence integration, calibration, and citation fidelity.

</details>

<details>
<summary><b>15. Aligning Quantum Operators with Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** David Kremer, Hang Hua, Pengyuan Li, Yunchao Liu, Rogerio Feris

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13811) • [📄 arXiv](https://arxiv.org/abs/2606.13811) • [📥 PDF](https://arxiv.org/pdf/2606.13811)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Can Large Language Models (LLMs) understand and reason about quantum operators? Despite their remarkable capabilities in mathematics and symbolic reasoning, LLMs remain inherently blind to quantum representations such as unitary matrices. In this ...

</details>

<details>
<summary><b>16. LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yaxin Du, Tianyu Zheng, Wei Zhang, Shawn Guo, Jian Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.18023) • [📄 arXiv](https://arxiv.org/abs/2606.18023) • [📥 PDF](https://arxiv.org/pdf/2606.18023)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>17. Looped World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.18208) • [📄 arXiv](https://arxiv.org/abs/2606.18208) • [📥 PDF](https://arxiv.org/pdf/2606.18208)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Current world models face a fundamental tension: faithful long-horizon simulation demands deep computation, but deeper models are expensive to deploy and prone to compounding errors. We resolve this by introducing Looped World Models (LoopWM), whi...

</details>

<details>
<summary><b>18. MotionVLA: Vision-Language-Action Model for Humanoid Motion</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.15142) • [📄 arXiv](https://arxiv.org/abs/2606.15142) • [📥 PDF](https://arxiv.org/pdf/2606.15142)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/MotionVLA) • [⭐ Code](https://github.com/huggingface)

> Open-sourced.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-06-17.json`](data/daily/2026-06-17.json) | 18 |
| 📆 This Week | [`2026-W24.json`](data/weekly/2026-W24.json) | 35 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 379 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-17 | 18 | [View JSON](data/daily/2026-06-17.json) |
| 📄 2026-06-16 | 17 | [View JSON](data/daily/2026-06-16.json) |
| 📄 2026-06-15 | 0 | [View JSON](data/daily/2026-06-15.json) |
| 📄 2026-06-14 | 44 | [View JSON](data/daily/2026-06-14.json) |
| 📄 2026-06-13 | 44 | [View JSON](data/daily/2026-06-13.json) |
| 📄 2026-06-12 | 22 | [View JSON](data/daily/2026-06-12.json) |
| 📄 2026-06-11 | 15 | [View JSON](data/daily/2026-06-11.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W24 | 35 | [View JSON](data/weekly/2026-W24.json) |
| 📅 2026-W23 | 166 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 379 | [View JSON](data/monthly/2026-06.json) |
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
