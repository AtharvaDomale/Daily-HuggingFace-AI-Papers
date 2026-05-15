<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-26-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3905+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">26</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">112</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">284</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3905+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 15, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling</b> ⭐ 20</summary>

<br/>

**👥 Authors:** Yizhuo Li, Shunkai Zhang, Haoran Zhang, Runzhe Zhan, Yafu Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13301) • [📄 arXiv](https://arxiv.org/abs/2605.13301) • [📥 PDF](https://arxiv.org/pdf/2605.13301)

**💻 Code:** [⭐ Code](https://github.com/Simplified-Reasoning/SU-01)

> We have open-sourced our code and model. Please check out our project page and GitHub repository: https://simplified-reasoning.github.io/SU-01/ https://github.com/Simplified-Reasoning/SU-01

</details>

<details>
<summary><b>2. MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Boxuan Zhang, Yihao Quan, Zeru Shi, Qingyue Jiao, Minghao Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15128) • [📄 arXiv](https://arxiv.org/abs/2605.15128) • [📥 PDF](https://arxiv.org/pdf/2605.15128)

**💻 Code:** [⭐ Code](https://github.com/MinghoKwok/MemEye)

> MemEye is a vision-centric long-term memory benchmark designed to evaluate how agents remember and reason over long-running image-grounded interactions. The benchmark focuses on assessing agents’ abilities to retain and utilize visual information ...

</details>

<details>
<summary><b>3. Self-Distilled Agentic Reinforcement Learning</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Jinyang Wu, Zi-Han Wang, Zhuowen Han, Zhiyuan Yao, Zhengxi Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15155) • [📄 arXiv](https://arxiv.org/abs/2605.15155) • [📥 PDF](https://arxiv.org/pdf/2605.15155)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/SDAR)

> No abstract available.

</details>

<details>
<summary><b>4. Warp-as-History: Generalizable Camera-Controlled Video Generation from One Training Video</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Tong He, Yifan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15182) • [📄 arXiv](https://arxiv.org/abs/2605.15182) • [📥 PDF](https://arxiv.org/pdf/2605.15182)

**💻 Code:** [⭐ Code](https://github.com/yyfz/Warp-as-History)

> Our method enables interactive camera trajectory following and viewpoint manipulation, similar to HappyOyster and Genie 3, using only a single camera-annotated training example.

</details>

<details>
<summary><b>5. SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15178) • [📄 arXiv](https://arxiv.org/abs/2605.15178) • [📥 PDF](https://arxiv.org/pdf/2605.15178)

**💻 Code:** [⭐ Code](https://github.com/NVlabs/Sana/)

> A 2.6B open-source world model that turns one image and a camera trajectory into 720p, minute-long, controllable video on a single GPU. Project Page: https://nvlabs.github.io/Sana/WM/ Code: https://github.com/NVlabs/Sana/

</details>

<details>
<summary><b>6. Beyond Individual Intelligence: Surveying Collaboration, Failure Attribution, and Self-Evolution in LLM-based Multi-Agent Systems</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14892) • [📄 arXiv](https://arxiv.org/abs/2605.14892) • [📥 PDF](https://arxiv.org/pdf/2605.14892)

**💻 Code:** [⭐ Code](https://github.com/mira-ai-lab/awesome-mas-life)

> LLM-based autonomous agents have demonstrated strong capabilities in reasoning, planning, and tool use, yet remain limited when tasks require sustained coordination across roles, tools, and environments. Multi-agent systems address this through st...

</details>

<details>
<summary><b>7. Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of Language-Model Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14386) • [📄 arXiv](https://arxiv.org/abs/2605.14386) • [📥 PDF](https://arxiv.org/pdf/2605.14386)

> FINAL Bench introduces a new evaluation paradigm for LLMs: functional metacognitive reasoning — not just "can the model solve it," but "does the model know when, why, and how it solves it." 100 tasks across 15 domains, built on the TICOS framework...

</details>

<details>
<summary><b>8. ATLAS: Agentic or Latent Visual Reasoning? One Word is Enough for Both</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pheng-Ann Heng, Xinyan Chen, Rain Liu, Ziyu Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15198) • [📄 arXiv](https://arxiv.org/abs/2605.15198) • [📥 PDF](https://arxiv.org/pdf/2605.15198)

> No abstract available.

</details>

<details>
<summary><b>9. Orchard: An Open-Source Agentic Modeling Framework</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15040) • [📄 arXiv](https://arxiv.org/abs/2605.15040) • [📥 PDF](https://arxiv.org/pdf/2605.15040)

**💻 Code:** [⭐ Code](https://github.com/microsoft/Orchard)

> No abstract available.

</details>

<details>
<summary><b>10. VGGT-Edit: Feed-forward Native 3D Scene Editing with Residual Field Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15186) • [📄 arXiv](https://arxiv.org/abs/2605.15186) • [📥 PDF](https://arxiv.org/pdf/2605.15186)

> https://arxiv.org/pdf/2605.15186

</details>

<details>
<summary><b>11. DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhen Xing, Yujie Wei, Kaixun Jiang, Junqiu Yu, Quanhao Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15055) • [📄 arXiv](https://arxiv.org/abs/2605.15055) • [📥 PDF](https://arxiv.org/pdf/2605.15055)

> Project page: https://quanhaol.github.io/DiffusionOPD-site/

</details>

<details>
<summary><b>12. BOOKMARKS: Efficient Active Storyline Memory for Role-playing</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14169) • [📄 arXiv](https://arxiv.org/abs/2605.14169) • [📥 PDF](https://arxiv.org/pdf/2605.14169)

**💻 Code:** [⭐ Code](https://github.com/KomeijiForce/BOOKMARKS_Koishiday_2026)

> BOOKMARKS: Efficient Active Storyline Memory for Role-playing

</details>

<details>
<summary><b>13. RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15190) • [📄 arXiv](https://arxiv.org/abs/2605.15190) • [📥 PDF](https://arxiv.org/pdf/2605.15190)

**💻 Code:** [⭐ Code](https://github.com/mvp-ai-lab/RAVEN)

> Causal autoregressive video diffusion models support real-time streaming generation by extrapolating future chunks from previously generated content. Distilling such generators from high-fidelity bidirectional teachers yields competitive few-step ...

</details>

<details>
<summary><b>14. Does Synthetic Layered Design Data Benefit Layered Design Decomposition?</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15167) • [📄 arXiv](https://arxiv.org/abs/2605.15167) • [📥 PDF](https://arxiv.org/pdf/2605.15167)

**💻 Code:** [⭐ Code](https://github.com/YangHaolin0526/SynLayers)

> Pure synthetic layered design dataset can indeed benefit the layer decomposition task.

</details>

<details>
<summary><b>15. EvolveMem:Self-Evolving Memory Architecture via AutoResearch for LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Cihang Xie, Zeyu Zheng, Peng Xia, Xinyu Ye, Jiaqi Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13941) • [📄 arXiv](https://arxiv.org/abs/2605.13941) • [📥 PDF](https://arxiv.org/pdf/2605.13941)

> We present EvolveMem, a self-evolving memory architecture that exposes its full retrieval configuration as a structured action space optimized by an LLM-powered diagnosis module.

</details>

<details>
<summary><b>16. MemLens: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14906) • [📄 arXiv](https://arxiv.org/abs/2605.14906) • [📥 PDF](https://arxiv.org/pdf/2605.14906)

**💻 Code:** [⭐ Code](https://github.com/xrenaf/MEMLENS)

> We have open-sourced our code and dataset. Please check out our GitHub repository: https://github.com/xrenaf/MEMLENS

</details>

<details>
<summary><b>17. Topology-Preserving Neural Operator Learning via Hodge Decomposition</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13834) • [📄 arXiv](https://arxiv.org/abs/2605.13834) • [📥 PDF](https://arxiv.org/pdf/2605.13834)

**💻 Code:** [⭐ Code](https://github.com/ContinuumCoder/Hodge-Spectral-Duality)

> In this paper, we study solution operators of physical field equations on geometric meshes from a function-space perspective. We reveal that Hodge orthogonality fundamentally resolves spectral interference by isolating unlearnable topological degr...

</details>

<details>
<summary><b>18. PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14269) • [📄 arXiv](https://arxiv.org/abs/2605.14269) • [📥 PDF](https://arxiv.org/pdf/2605.14269)

**💻 Code:** [⭐ Code](https://github.com/h6kplus/PhyMotion)

> Generating realistic human motion is a central yet unsolved challenge in video generation. While reinforcement learning (RL)-based post-training has driven recent gains in general video quality, extending it to human motion remains bottlenecked by...

</details>

<details>
<summary><b>19. SPIN: Structural LLM Planning via Iterative Navigation for Industrial Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14051) • [📄 arXiv](https://arxiv.org/abs/2605.14051) • [📥 PDF](https://arxiv.org/pdf/2605.14051)

> Industrial LLM agent systems often separate planning from execution, yet LLM planners frequently produce structurally invalid or unnecessarily long workflows, leading to brittle failures and avoidable tool and API cost. We propose \texttt{SPIN}, a...

</details>

<details>
<summary><b>20. Nexus : An Agentic Framework for Time Series Forecasting</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vishy Tirumalashetty, Nanyun Peng, Mihir Parmar, Palash Goyal, Sarkar Snigdha Sarathi Das

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14389) • [📄 arXiv](https://arxiv.org/abs/2605.14389) • [📥 PDF](https://arxiv.org/pdf/2605.14389)

> No abstract available.

</details>

<details>
<summary><b>21. Learning to Build the Environment: Self-Evolving Reasoning RL via Verifiable Environment Synthesis</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenhao Yu, Dian Yu, Kishan Panaganti, Zhenwen Liang, Yucheng Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14392) • [📄 arXiv](https://arxiv.org/abs/2605.14392) • [📥 PDF](https://arxiv.org/pdf/2605.14392)

> No abstract available.

</details>

<details>
<summary><b>22. Quantitative Video World Model Evaluation for Geometric-Consistency</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Xueyan Zou, Yuheng Li, Yinling Zhang, Yihao Pi, Jiaxin Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15185) • [📄 arXiv](https://arxiv.org/abs/2605.15185) • [📥 PDF](https://arxiv.org/pdf/2605.15185)

**💻 Code:** [⭐ Code](https://github.com/AnteaWu/PDI-Bench)

> No abstract available.

</details>

<details>
<summary><b>23. LLM-based Detection of Manipulative Political Narratives</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14354) • [📄 arXiv](https://arxiv.org/abs/2605.14354) • [📥 PDF](https://arxiv.org/pdf/2605.14354)

**💻 Code:** [⭐ Code](https://github.com/SinclairSchneider/manipulative_narrative_detection)

> We present a new computational framework for detecting and structuring manipulative political narratives. A task that became more important due to the shift of political discussions to social media. One of the primary challenges thereby is differe...

</details>

<details>
<summary><b>24. FutureSim: Replaying World Events to Evaluate Adaptive Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15188) • [📄 arXiv](https://arxiv.org/abs/2605.15188) • [📥 PDF](https://arxiv.org/pdf/2605.15188)

**💻 Code:** [⭐ Code](https://github.com/OpenForecaster/futuresim)

> No abstract available.

</details>

<details>
<summary><b>25. Ideology Prediction of German Political Texts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14352) • [📄 arXiv](https://arxiv.org/abs/2605.14352) • [📥 PDF](https://arxiv.org/pdf/2605.14352)

> Elections represent a crucial milestone in a nation's ongoing development. To better understand the political rhetoric from various movements, ranging from left to right, we propose a transformer-based model capable of projecting the political ori...

</details>

<details>
<summary><b>26. PREPING: Building Agent Memory without Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13880) • [📄 arXiv](https://arxiv.org/abs/2605.13880) • [📥 PDF](https://arxiv.org/pdf/2605.13880)

**💻 Code:** [⭐ Code](https://github.com/Dozi01/PREPING)

> LLM agents often need memory to solve tasks in new tool environments, but memory is usually built only after seeing human tasks or deployment-time user interactions. This creates a cold-start problem: the agent needs memory most when it has the le...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 26 |
| 📅 Today | [`2026-05-15.json`](data/daily/2026-05-15.json) | 26 |
| 📆 This Week | [`2026-W19.json`](data/weekly/2026-W19.json) | 112 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 284 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-15 | 26 | [View JSON](data/daily/2026-05-15.json) |
| 📄 2026-05-14 | 22 | [View JSON](data/daily/2026-05-14.json) |
| 📄 2026-05-13 | 22 | [View JSON](data/daily/2026-05-13.json) |
| 📄 2026-05-12 | 16 | [View JSON](data/daily/2026-05-12.json) |
| 📄 2026-05-11 | 26 | [View JSON](data/daily/2026-05-11.json) |
| 📄 2026-05-10 | 38 | [View JSON](data/daily/2026-05-10.json) |
| 📄 2026-05-09 | 38 | [View JSON](data/daily/2026-05-09.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W19 | 112 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 284 | [View JSON](data/monthly/2026-05.json) |
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
