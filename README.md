<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4029+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">18</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">408</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4029+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 18, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. MMSkills: Towards Multimodal Skills for General Visual Agents</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13527) • [📄 arXiv](https://arxiv.org/abs/2605.13527) • [📥 PDF](https://arxiv.org/pdf/2605.13527)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DeepExperience/MMSkills)

> Reusable skills have become a core substrate for improving agent capabilities, yet most existing skill packages encode reusable behavior primarily as textual prompts, executable code, or learned routines. For visual agents, however, procedural kno...

</details>

<details>
<summary><b>2. DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo</b> ⭐ 7</summary>

<br/>

**👥 Authors:** He Lin, Siyuan Huang, Xiangyu Wang, Weizhi Zhao, Hanwen Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.16257) • [📄 arXiv](https://arxiv.org/abs/2605.16257) • [📥 PDF](https://arxiv.org/pdf/2605.16257)

**💻 Code:** [⭐ Code](https://github.com/brave-eai/dexjoco) • [⭐ Code](https://github.com/huggingface)

> DexJoCo is a toolkit and benchmark for task-oriented dexterous manipulation based on MuJoCo with full-stack infrastructure, including open-sourced hardware design, data collection pipeline, human-collected trajectories, and long-horizon tool-use t...

</details>

<details>
<summary><b>3. Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15726) • [📄 arXiv](https://arxiv.org/abs/2605.15726) • [📥 PDF](https://arxiv.org/pdf/2605.15726)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/tally0818/NudgeRL)

> We introduce NudgeRL, a method that enables the model to generate diverse reasoning paths during rollout through strategy nudging, thereby improving exploration. To effectively learn from the exploration induced by nudging, we further introduce th...

</details>

<details>
<summary><b>4. InsightTok: Improving Text and Face Fidelity in Discrete Tokenization for Autoregressive Image Generation</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14333) • [📄 arXiv](https://arxiv.org/abs/2605.14333) • [📥 PDF](https://arxiv.org/pdf/2605.14333)

**💻 Code:** [⭐ Code](https://github.com/LeapLabTHU/InsightTok) • [⭐ Code](https://github.com/huggingface)

> InsightTok is a discrete visual tokenizer designed to improve the fidelity of text and faces, two of the most challenging yet perceptually important structures in autoregressive image generation. Existing visual tokenizers are typically trained wi...

</details>

<details>
<summary><b>5. FashionChameleon: Towards Real-Time and Interactive Human-Garment Video Customization</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15824) • [📄 arXiv](https://arxiv.org/abs/2605.15824) • [📥 PDF](https://arxiv.org/pdf/2605.15824)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/quanjiansong/FashionChameleon)

> No abstract available.

</details>

<details>
<summary><b>6. Solvita: Enhancing Large Language Models for Competitive Programming via Agentic Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15301) • [📄 arXiv](https://arxiv.org/abs/2605.15301) • [📥 PDF](https://arxiv.org/pdf/2605.15301)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language models (LLMs) still struggle with the rigorous reasoning demands of hard competitive programming. While recent multi-agent frameworks attempt to bridge this reliability gap, they remain fundamentally stateless: they rely on static r...

</details>

<details>
<summary><b>7. ReactiveGWM: Steering NPC in Reactive Game World Models</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Yinhan Zhang, Zizhao Tong, Zhaohu Xing, Danze Chen, Zeqing Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15256) • [📄 arXiv](http://arxiv.org/abs/2605.15256) • [📥 PDF](https://arxiv.org/pdf/2605.15256)

**💻 Code:** [⭐ Code](https://github.com/INV-WZQ/ReactiveGWM) • [⭐ Code](https://github.com/huggingface)

> 🚀 ReactiveGWM: Bringing true autonomy to NPCs in Game World Models! Instead of treating NPCs as background pixels , we explicitly decouple player controls from NPC behaviors. ✨ Highlights: Strategy-Driven: NPCs autonomously execute high-level inte...

</details>

<details>
<summary><b>8. Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Ruizhe He, Weijie Wang, Zeyue Xue, Siming Fu, Xiaoxuan He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15980) • [📄 arXiv](https://arxiv.org/abs/2605.15980) • [📥 PDF](https://arxiv.org/pdf/2605.15980)

**💻 Code:** [⭐ Code](https://github.com/Shredded-Pork/Flash-GRPO) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. Unlocking Dense Metric Depth Estimation in VLMs</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15876) • [📄 arXiv](https://arxiv.org/abs/2605.15876) • [📥 PDF](https://arxiv.org/pdf/2605.15876)

**💻 Code:** [⭐ Code](https://github.com/hanxunyu/DepthVLM) • [⭐ Code](https://github.com/huggingface)

> DepthVLM serves as a unified foundation model for both low-level dense geometry prediction and high-level multimodal understanding, while achieving substantially faster inference compared with existing VLM-based approaches such as DepthLM and Yout...

</details>

<details>
<summary><b>10. PAGER: Bridging the Semantic-Execution Gap in Point-Precise Geometric GUI Control</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15963) • [📄 arXiv](https://arxiv.org/abs/2605.15963) • [📥 PDF](https://arxiv.org/pdf/2605.15963)

**💻 Code:** [⭐ Code](https://github.com/OpenRaiser/Pager) • [⭐ Code](https://github.com/huggingface)

> Large vision-language models have significantly advanced GUI agents, enabling executable interaction across web, mobile, and desktop interfaces. Yet these gains largely rely on a forgiving region-tolerant paradigm, where many nearby pixels inside ...

</details>

<details>
<summary><b>11. FFAvatar: Few-Shot, Feed-Forward, and Generalizable Avatar Reconstruction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Gordon Guocheng Qian, Hao Li, Yinyu Nie, Jiahao Luo, Thuan Hoang Nguyen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15320) • [📄 arXiv](https://arxiv.org/abs/2605.15320) • [📥 PDF](https://arxiv.org/pdf/2605.15320)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. DiagnosticIQ: A Benchmark for LLM-Based Industrial Maintenance Action Recommendation from Symbolic Rules</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08614) • [📄 arXiv](https://arxiv.org/abs/2605.08614) • [📥 PDF](https://arxiv.org/pdf/2605.08614)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Monitoring complex industrial assets relies on engineer-authored symbolic rules that trigger based on sensor conditions and prompt technicians to perform corrective actions. The bottleneck is not detection but response: translating rules into main...

</details>

<details>
<summary><b>13. WorldAct: Activating Monolithic 3D Worlds into Interactive-Ready Object-Centric Scenes</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Sikuang Li, Chen Yang, Jiazhong Cen, Jiawei Guo, Jichen Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15843) • [📄 arXiv](https://arxiv.org/abs/2605.15843) • [📥 PDF](https://arxiv.org/pdf/2605.15843)

**💻 Code:** [⭐ Code](https://github.com/SJTU-DeepVisionLab/WorldAct) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. Learning from Failures: Correction-Oriented Policy Optimization with Verifiable Rewards</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hongyu Lin, Xueru Wen, Boxi Cao, Jie Lou, Mengjie Ren

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14539) • [📄 arXiv](https://arxiv.org/abs/2605.14539) • [📥 PDF](https://arxiv.org/pdf/2605.14539)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an effective paradigm for improving the reasoning capabilities of large language models. However, RLVR training is often hindered by sparse binary rewards and weak credit assignm...

</details>

<details>
<summary><b>15. HodgeCover: Higher-Order Topological Coverage Drives Compression of Sparse Mixture-of-Experts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13997) • [📄 arXiv](https://arxiv.org/abs/2605.13997) • [📥 PDF](https://arxiv.org/pdf/2605.13997)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Sparse Mixture-of-Experts (MoE) layers route tokens through a handful of experts, and learning-free compression of these layers reduces inference cost without retraining. A subtle obstruction blocks every existing compressor in this family: three ...

</details>

<details>
<summary><b>16. OmniHumanoid: Streaming Cross-Embodiment Video Generation with Paired-Free Adaptation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mike Zheng Shou, Yihan Wang, Pei Yang, Xiyao Deng, Yiren Song

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.12038) • [📄 arXiv](https://arxiv.org/abs/2605.12038) • [📥 PDF](https://arxiv.org/pdf/2605.12038)

**💻 Code:** [⭐ Code](https://github.com/showlab/OmniHumanoid) • [⭐ Code](https://github.com/huggingface)

> Cross-embodiment video generation aims to transfer motions across different humanoid embodiments, such as human-to-robot and robot-to-robot, enabling scalable data generation for embodied intelligence. A major challenge in this setting is that mot...

</details>

<details>
<summary><b>17. Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yannan Nellie Wu, Bilge Acun, Despoina Magka, Chien-Yu Lin, Alberto Pepe

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15871) • [📄 arXiv](https://arxiv.org/abs/2605.15871) • [📥 PDF](https://arxiv.org/pdf/2605.15871)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>18. Look Before You Leap: Autonomous Exploration for LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhengzhou Cai, Yu Wang, Yuxin Liu, Wentao Shi, Ziang Ye

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.16143) • [📄 arXiv](https://arxiv.org/abs/2605.16143) • [📥 PDF](https://arxiv.org/pdf/2605.16143)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-05-18.json`](data/daily/2026-05-18.json) | 18 |
| 📆 This Week | [`2026-W20.json`](data/weekly/2026-W20.json) | 18 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 408 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-18 | 18 | [View JSON](data/daily/2026-05-18.json) |
| 📄 2026-05-17 | 53 | [View JSON](data/daily/2026-05-17.json) |
| 📄 2026-05-16 | 53 | [View JSON](data/daily/2026-05-16.json) |
| 📄 2026-05-15 | 26 | [View JSON](data/daily/2026-05-15.json) |
| 📄 2026-05-14 | 22 | [View JSON](data/daily/2026-05-14.json) |
| 📄 2026-05-13 | 22 | [View JSON](data/daily/2026-05-13.json) |
| 📄 2026-05-12 | 16 | [View JSON](data/daily/2026-05-12.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W20 | 18 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 408 | [View JSON](data/monthly/2026-05.json) |
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
