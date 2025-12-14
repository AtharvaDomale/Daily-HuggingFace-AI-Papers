<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-25-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-324+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">25</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">186</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">373</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">324+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 14, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. T-pro 2.0: An Efficient Russian Hybrid-Reasoning Model and Playground</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10430) • [📄 arXiv](https://arxiv.org/abs/2512.10430) • [📥 PDF](https://arxiv.org/pdf/2512.10430)

> T-pro 2.0 is an open-weight Russian LLM with hybrid reasoning and fast inference, released with datasets, benchmarks, and an optimized decoding pipeline to support reproducible research and practical applications.

</details>

<details>
<summary><b>2. Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10739) • [📄 arXiv](https://arxiv.org/abs/2512.10739) • [📥 PDF](https://arxiv.org/pdf/2512.10739)

> Due to a user error, the abstract displayed in this paper contains some errors 😭 (the abstract in the PDF is correct). The correct and complete abstract is as follows: Large Reasoning Models (LRMs) have expanded the mathematical reasoning frontier...

</details>

<details>
<summary><b>3. Are We Ready for RL in Text-to-3D Generation? A Progressive Investigation</b> ⭐ 45</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10949) • [📄 arXiv](https://arxiv.org/abs/2512.10949) • [📥 PDF](https://arxiv.org/pdf/2512.10949)

**💻 Code:** [⭐ Code](https://github.com/Ivan-Tang-3D/3DGen-R1)

> Code is released at https://github.com/Ivan-Tang-3D/3DGen-R1 . Model is released at https://huggingface.co/IvanTang/3DGen-R1 .

</details>

<details>
<summary><b>4. OPV: Outcome-based Process Verifier for Efficient Long Chain-of-Thought Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10756) • [📄 arXiv](https://arxiv.org/abs/2512.10756) • [📥 PDF](https://arxiv.org/pdf/2512.10756)

> We propose the Outcome-based Process Verifier (OPV), which verifies the rationale process of summarized outcomes from long CoTs to achieve both accurate and efficient verification and enable large-scale annotation.

</details>

<details>
<summary><b>5. Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10534) • [📄 arXiv](https://arxiv.org/abs/2512.10534) • [📥 PDF](https://arxiv.org/pdf/2512.10534)

> InternGeometry overcomes the heuristic limitations in geometry by iteratively proposing propositions and auxiliary constructions, verifying them with a symbolic engine, and reflecting on the engine's feedback to guide subsequent proposals. Built o...

</details>

<details>
<summary><b>6. MoCapAnything: Unified 3D Motion Capture for Arbitrary Skeletons from Monocular Videos</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mingxi Xu, DonaldLian, weixia111111, wzy27, kehong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10881) • [📄 arXiv](https://arxiv.org/abs/2512.10881) • [📥 PDF](https://arxiv.org/pdf/2512.10881)

> Motion capture now underpins content creation far beyond digital humans, yet most existing pipelines remain species- or template-specific. We formalize this gap as Category-Agnostic Motion Capture (CAMoCap): given a monocular video and an arbitrar...

</details>

<details>
<summary><b>7. BEAVER: An Efficient Deterministic LLM Verifier</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05439) • [📄 arXiv](https://arxiv.org/abs/2512.05439) • [📥 PDF](https://arxiv.org/pdf/2512.05439)

> BEAVER is the first practical framework to formally verify an LLM’s output distribution. It enables rigorous assessment and comparison beyond traditional sampling-based evaluation. BEAVER computes deterministic, sound bounds on the total probabili...

</details>

<details>
<summary><b>8. Thinking with Images via Self-Calling Agent</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Qixiang Ye, Fang Wan, callsys, ywenxi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08511) • [📄 arXiv](https://arxiv.org/abs/2512.08511) • [📥 PDF](https://arxiv.org/pdf/2512.08511)

**💻 Code:** [⭐ Code](https://github.com/YWenxi/think-with-images-through-self-calling)

> 🧠🖼️ Vision-language models are getting smarter—but also harder to train. Many recent systems “think with images,” weaving visual information directly into their reasoning. While powerful, this approach can be hard to incentivize, as it usually req...

</details>

<details>
<summary><b>9. From Macro to Micro: Benchmarking Microscopic Spatial Intelligence on Molecules via Vision-Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10867) • [📄 arXiv](https://arxiv.org/abs/2512.10867) • [📥 PDF](https://arxiv.org/pdf/2512.10867)

> This paper introduces the concept of Microscopic Spatial Intelligence (MiSI), the capability to perceive and reason about the spatial relationships of invisible microscopic entities, which is fundamental to scientific discovery. To assess the pote...

</details>

<details>
<summary><b>10. Stronger Normalization-Free Transformers</b> ⭐ 31</summary>

<br/>

**👥 Authors:** Zhuang Liu, Mingjie Sun, Jiachen Zhu, TaiMingLu, Fishloong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10938) • [📄 arXiv](https://arxiv.org/abs/2512.10938) • [📥 PDF](https://arxiv.org/pdf/2512.10938)

**💻 Code:** [⭐ Code](https://github.com/zlab-princeton/Derf)

> Although normalization layers have long been viewed as indispensable components of deep learning architectures, the recent introduction of Dynamic Tanh (DyT) has demonstrated that alternatives are possible. The point-wise function DyT constrains e...

</details>

<details>
<summary><b>11. VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2511.23386) • [📄 arXiv](https://arxiv.org/abs/2511.23386) • [📥 PDF](https://arxiv.org/pdf/2511.23386)

> arXiv: https://arxiv.org/pdf/2511.23386 Overall Architecture

</details>

<details>
<summary><b>12. StereoSpace: Depth-Free Synthesis of Stereo Geometry via End-to-End Diffusion in a Canonical Space</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10959) • [📄 arXiv](https://arxiv.org/abs/2512.10959) • [📥 PDF](https://arxiv.org/pdf/2512.10959)

**💻 Code:** [⭐ Code](https://github.com/prs-eth/stereospace)

> Project page: https://huggingface.co/spaces/prs-eth/stereospace_web

</details>

<details>
<summary><b>13. Evaluating Gemini Robotics Policies in a Veo World Simulator</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10675) • [📄 arXiv](https://arxiv.org/abs/2512.10675) • [📥 PDF](https://arxiv.org/pdf/2512.10675)

> Generative world models hold significant potential for simulating interactions with visuomotor policies in varied environments. Frontier video models can enable generation of realistic observations and environment interactions in a scalable and ge...

</details>

<details>
<summary><b>14. MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Won-Sik Cheong, Geonho Kim, shurek20, klavna, sangwoonkwak

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09270) • [📄 arXiv](https://arxiv.org/abs/2512.09270) • [📥 PDF](https://arxiv.org/pdf/2512.09270)

**💻 Code:** [⭐ Code](https://github.com/CMLab-Korea/MoRel-arXiv)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Dynamic-eDiTor: Training-Free Text-Driven 4D Scene Editing with Multimodal ...

</details>

<details>
<summary><b>15. The FACTS Leaderboard: A Comprehensive Benchmark for Large Language Model Factuality</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10791) • [📄 arXiv](https://arxiv.org/abs/2512.10791) • [📥 PDF](https://arxiv.org/pdf/2512.10791)

> We introduce The FACTS Leaderboard, an online leaderboard suite and associated set of benchmarks that comprehensively evaluates the ability of language models to generate factually accurate text across diverse scenarios. The suite provides a holis...

</details>

<details>
<summary><b>16. Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10359) • [📄 arXiv](https://arxiv.org/abs/2512.10359) • [📥 PDF](https://arxiv.org/pdf/2512.10359)

**💻 Code:** [⭐ Code](https://github.com/fansunqi/VideoTool)

> Tool-augmented VideoQA system, accepted by NeurIPS'25 main track.

</details>

<details>
<summary><b>17. ReViSE: Towards Reason-Informed Video Editing in Unified Models with Self-Reflective Learning</b> ⭐ 4</summary>

<br/>

**👥 Authors:** SuaLily, whluo, Yanbiao, LewisPan, JacobYuan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09924) • [📄 arXiv](https://arxiv.org/abs/2512.09924) • [📥 PDF](https://arxiv.org/pdf/2512.09924)

**💻 Code:** [⭐ Code](https://github.com/Liuxinyv/ReViSE)

> Code: https://github.com/Liuxinyv/ReViSE

</details>

<details>
<summary><b>18. H2R-Grounder: A Paired-Data-Free Paradigm for Translating Human Interaction Videos into Physically Grounded Robot Videos</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Pei Yang, Xiaokang Liu, AnalMom, yiren98, HaiCi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09406) • [📄 arXiv](https://arxiv.org/abs/2512.09406) • [📥 PDF](https://arxiv.org/pdf/2512.09406)

**💻 Code:** [⭐ Code](https://github.com/showlab/H2R-Grounder)

> A framework to translate human object interaction (HOI) videos into grounded robot object interaction (ROI) videos.

</details>

<details>
<summary><b>19. Fed-SE: Federated Self-Evolution for Privacy-Constrained Multi-Environment LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaodong Gu, Yuchao Qiu, Xiang Chen, lanqz7766, YerbaPage

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08870) • [📄 arXiv](https://arxiv.org/abs/2512.08870) • [📥 PDF](https://arxiv.org/pdf/2512.08870)

> Check this out!

</details>

<details>
<summary><b>20. Omni-Attribute: Open-vocabulary Attribute Encoder for Visual Concept Personalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10955) • [📄 arXiv](https://arxiv.org/abs/2512.10955) • [📥 PDF](https://arxiv.org/pdf/2512.10955)

> This work can isolate a specific attribute from any image and merge those selected attributes from multiple images into a coherent generation.

</details>

<details>
<summary><b>21. DuetSVG: Unified Multimodal SVG Generation with Internal Visual Guidance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jing Liao, Yiran Xu, Matthew Fisher, Nanxuan Zhao, Peiying Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10894) • [📄 arXiv](https://arxiv.org/abs/2512.10894) • [📥 PDF](https://arxiv.org/pdf/2512.10894)

> We introduce DuetSVG, a unified multimodal model that jointly generates image tokens and corresponding SVG tokens in an end-to-end manner. DuetSVG is trained on both image and SVG datasets. At inference, we apply a novel test-time scaling strategy...

</details>

<details>
<summary><b>22. Confucius Code Agent: An Open-sourced AI Software Engineer at Industrial Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10398) • [📄 arXiv](https://arxiv.org/abs/2512.10398) • [📥 PDF](https://arxiv.org/pdf/2512.10398)

> Real-world AI software engineering demands coding agents that can reason over massive repositories, maintain durable memory across and within long sessions, and robustly coordinate complex toolchains at test time. Existing open-source coding agent...

</details>

<details>
<summary><b>23. X-Humanoid: Robotize Human Videos to Generate Humanoid Videos at Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04537) • [📄 arXiv](https://arxiv.org/abs/2512.04537) • [📥 PDF](https://arxiv.org/pdf/2512.04537)

> The advancement of embodied AI has unlocked significant potential for intelligent humanoid robots. However, progress in both Vision-Language-Action (VLA) models and world models is severely hampered by the scarcity of large-scale, diverse training...

</details>

<details>
<summary><b>24. MOA: Multi-Objective Alignment for Role-Playing Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fei Huang, Ke Wang, Yongbin-Li, yuchuan123, ChonghuaLiao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09756) • [📄 arXiv](https://arxiv.org/abs/2512.09756) • [📥 PDF](https://arxiv.org/pdf/2512.09756)

> Role-playing agents (RPAs) must simultaneously master many conflicting skills -- following multi-turn instructions, exhibiting domain knowledge, and adopting a consistent linguistic style. Existing work either relies on supervised fine-tuning (SFT...

</details>

<details>
<summary><b>25. DragMesh: Interactive 3D Generation Made Easy</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06424) • [📄 arXiv](https://arxiv.org/abs/2512.06424) • [📥 PDF](https://arxiv.org/pdf/2512.06424)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/DragMesh)

> DragMesh enables real time, physically valid 3D object articulation by decoupling kinematic reasoning from motion generation and producing plausible motions via a dual quaternion based generative model.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 25 |
| 📅 Today | [`2025-12-14.json`](data/daily/2025-12-14.json) | 25 |
| 📆 This Week | [`2025-W49.json`](data/weekly/2025-W49.json) | 186 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 373 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-14 | 25 | [View JSON](data/daily/2025-12-14.json) |
| 📄 2025-12-13 | 24 | [View JSON](data/daily/2025-12-13.json) |
| 📄 2025-12-12 | 21 | [View JSON](data/daily/2025-12-12.json) |
| 📄 2025-12-11 | 25 | [View JSON](data/daily/2025-12-11.json) |
| 📄 2025-12-10 | 29 | [View JSON](data/daily/2025-12-10.json) |
| 📄 2025-12-09 | 24 | [View JSON](data/daily/2025-12-09.json) |
| 📄 2025-12-08 | 38 | [View JSON](data/daily/2025-12-08.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 373 | [View JSON](data/monthly/2025-12.json) |

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
