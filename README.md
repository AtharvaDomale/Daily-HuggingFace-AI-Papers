<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4659+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">22</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">78</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">256</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4659+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 12, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13673) • [📄 arXiv](https://arxiv.org/abs/2606.13673) • [📥 PDF](https://arxiv.org/pdf/2606.13673)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NVlabs/SpatialClaw)

> "Code is the right action interface for spatial reasoning!!" SpatialClaw lets a VLM-backed agent write Python in a persistent kernel, composing perception modules, inspecting intermediate results, and revising its strategy across steps. It is trai...

</details>

<details>
<summary><b>2. EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13681) • [📄 arXiv](https://arxiv.org/abs/2606.13681) • [📥 PDF](https://arxiv.org/pdf/2606.13681)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language model (LLM) agents have achieved strong performance on a wide range of benchmarks, yet most evaluations assume static environments. In contrast, real-world deployment is inherently dynamic, requiring agents to continually align thei...

</details>

<details>
<summary><b>3. FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Shuo Tang, Ziyang Zeng, Xiaoqing Xiang, Yimeng Chen, Jia Deng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12087) • [📄 arXiv](https://arxiv.org/abs/2606.12087) • [📥 PDF](https://arxiv.org/pdf/2606.12087)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RUCAIBox/FORT-Searcher)

> Training deep search agents requires verifiable questions whose answers remain unavailable until sufficient evidence has been acquired through search. Existing synthesis methods often increase apparent difficulty by enriching graph structures, but...

</details>

<details>
<summary><b>4. InterleaveThinker: Reinforcing Agentic Interleaved Generation</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Zoey Guo, Kaituo Feng, Manyuan Zhang, Harry Lee, Dian Zheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13679) • [📄 arXiv](https://arxiv.org/abs/2606.13679) • [📥 PDF](https://arxiv.org/pdf/2606.13679)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zhengdian1/InterleaveThinker)

> hf: https://huggingface.co/InterleaveThinker

</details>

<details>
<summary><b>5. WeaveBench: A Long-Horizon, Real-World Benchmark for Computer-Use Agents with Hybrid Interfaces</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09426) • [📄 arXiv](https://arxiv.org/abs/2606.09426) • [📥 PDF](https://arxiv.org/pdf/2606.09426)

**💻 Code:** [⭐ Code](https://github.com/weavebench/WeaveBench) • [⭐ Code](https://github.com/huggingface)

> We introduce WeaveBench, a long-horizon benchmark with 114 tasks across 8 real-world domains, where agents must interleave GUI and CLI/code operations in a single trajectory on a real Ubuntu desktop. The best frontier model reaches only 41.2% Pass...

</details>

<details>
<summary><b>6. N-GRPO: Embedding-Level Neighbor Mixing for Enhanced Policy Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Linchao Zhu, Peng Di, Hang Yu, Xukun Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10768) • [📄 arXiv](https://arxiv.org/abs/2606.10768) • [📥 PDF](https://arxiv.org/pdf/2606.10768)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJUSCL/N-GRPO)

> N-GRPO: Embedding-Level Neighbor Mixing for Enhanced Policy Optimization Github: https://github.com/ZJUSCL/N-GRPO

</details>

<details>
<summary><b>7. MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13376) • [📄 arXiv](https://arxiv.org/abs/2606.13376) • [📥 PDF](https://arxiv.org/pdf/2606.13376)

**💻 Code:** [⭐ Code](https://github.com/Orange-3DV-Team/MoVerse) • [⭐ Code](https://github.com/huggingface)

> MoVerse is a real-time video world model that transforms a single narrow-field-of-view image into an interactively navigable environment by lifting a topology-aware 360° panorama into a persistent 3D Gaussian scaffold, achieving high-fidelity scen...

</details>

<details>
<summary><b>8. Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Runtao Liu, Wei Wei, Youyang Zhai, Jianmin Chen, Jiaqi Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.08063) • [📄 arXiv](https://arxiv.org/abs/2606.08063) • [📥 PDF](https://arxiv.org/pdf/2606.08063)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jqtangust/Robust-U1)

> Multimodal Large Language Models (MLLMs) have demonstrated remarkable success in visual understanding, yet their performance degrades significantly under real-world visual corruptions. While existing robustness enhancement approaches exist, they a...

</details>

<details>
<summary><b>9. Visual Para-Thinker++: A Single-Policy Multi-Agent Framework for Visual Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiaze Li, Yifei Gao, Hongyu Wang, Haoran Xu, zizhaotong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09290) • [📄 arXiv](https://arxiv.org/abs/2606.09290) • [📥 PDF](https://arxiv.org/pdf/2606.09290)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Visual ParaThinker++

</details>

<details>
<summary><b>10. MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13473) • [📄 arXiv](https://arxiv.org/abs/2606.13473) • [📥 PDF](https://arxiv.org/pdf/2606.13473)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>11. SG-OPD: Sign-Gated On-Policy Distillation via Sign-Consistency Gating and Phased Teacher Sampling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaofeng Zhang, Yifei Gao, Hongyu Wang, Haoran Xu, williamljz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09304) • [📄 arXiv](https://arxiv.org/abs/2606.09304) • [📥 PDF](https://arxiv.org/pdf/2606.09304)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> SG-OPD

</details>

<details>
<summary><b>12. HarnessBridge: Learnable Bidirectional Controller for LLM Agent Harness</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12882) • [📄 arXiv](https://arxiv.org/abs/2606.12882) • [📥 PDF](https://arxiv.org/pdf/2606.12882)

**💻 Code:** [⭐ Code](https://github.com/mandyyyyii/HarnessBridge) • [⭐ Code](https://github.com/huggingface)

> HarnessBridge

</details>

<details>
<summary><b>13. VideoMDM: Towards 3D Human Motion Generation From 2D Supervision</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Or Litany, Merav Keidar, Gal Michael Harari, Amir Mann

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13364) • [📄 arXiv](https://arxiv.org/abs/2606.13364) • [📥 PDF](https://arxiv.org/pdf/2606.13364)

**💻 Code:** [⭐ Code](https://github.com/Amir-Mann/VideoMDM_release) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. EvoBrowseComp: Benchmarking Search Agents on Evolving Knowledge</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fandong Meng, Xianfeng Zeng, Lianzhe Huang, Jiaan Wang, Yunhan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13120) • [📄 arXiv](https://arxiv.org/abs/2606.13120) • [📥 PDF](https://arxiv.org/pdf/2606.13120)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> EvoBrowseComp is a search agent benchmark of 400 English and 400 Chinese contamination-free complex questions synthesized via live-web traversal.

</details>

<details>
<summary><b>15. Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuxuan Fan, Yinhong Liu, Shengen Wu, Chao Chen, Jiayu Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13106) • [📄 arXiv](https://arxiv.org/abs/2606.13106) • [📥 PDF](https://arxiv.org/pdf/2606.13106)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper introduces a pair of learned boundary tokens (/) that mark where latent reasoning begins and ends, making hidden-state-recurrence latent CoT both trainable with standard on-policy RL (GRPO) and open to direct mechanistic probing and cau...

</details>

<details>
<summary><b>16. Surflo: Consistent 3D Surface Flow Model with Global State</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ko Nishino, Jiahui Lei, Nicolas Dufour, Shu Nakamura, Antoine Guédon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13644) • [📄 arXiv](https://arxiv.org/abs/2606.13644) • [📥 PDF](https://arxiv.org/pdf/2606.13644)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>17. EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13662) • [📄 arXiv](https://arxiv.org/abs/2606.13662) • [📥 PDF](https://arxiv.org/pdf/2606.13662)

**💻 Code:** [⭐ Code](https://github.com/THU-Team-Eureka/EurekAgent) • [⭐ Code](https://github.com/huggingface)

> As model capabilities continue to improve, we argue that the bottleneck for autonomous scientific discovery is shifting from prescribing agent workflows to designing agent environments: the resources, constraints, and interfaces that shape agent b...

</details>

<details>
<summary><b>18. Evoflux: Inference-Time Evolution of Executable Tool Workflows for Compact Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12674) • [📄 arXiv](https://arxiv.org/abs/2606.12674) • [📥 PDF](https://arxiv.org/pdf/2606.12674)

**💻 Code:** [⭐ Code](https://github.com/IBM/Evoflux) • [⭐ Code](https://github.com/huggingface)

> Evoflux tackles a practical bottleneck for small (1.5B–4B) tool-using agents: with only a few hundred teacher traces available, should that scarce supervision go into fine-tuning the model's weights, or into search at inference time? The paper ref...

</details>

<details>
<summary><b>19. MaskAlign: Token-Subset Representation Alignment for Efficient Diffusion Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.08788) • [📄 arXiv](https://arxiv.org/abs/2606.08788) • [📥 PDF](https://arxiv.org/pdf/2606.08788)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>20. WEAVER, Better, Faster, Longer: An Effective World Model for Robotic Manipulation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Andrea Bajcsy, Gokul Swamy, Jesse Farebrother, Yilin Wu, Arnav Kumar Jain

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13672) • [📄 arXiv](https://arxiv.org/abs/2606.13672) • [📥 PDF](https://arxiv.org/pdf/2606.13672)

**💻 Code:** [⭐ Code](https://github.com/arnavkj1995/WEAVER) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>21. LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories</b> ⭐ 39</summary>

<br/>

**👥 Authors:** Chenxi Li, Yanshuo Liu, Xi Chen, Xinjie Liu, Baochang Ren

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.13578) • [📄 arXiv](https://arxiv.org/abs/2606.13578) • [📥 PDF](https://arxiv.org/pdf/2606.13578)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/LabVLA) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. MuJoCo-Drones-Gym: A GPU-Accelerated Multi-Drone Simulator for Control and Reinforcement Learning</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.08039) • [📄 arXiv](https://arxiv.org/abs/2606.08039) • [📥 PDF](https://arxiv.org/pdf/2606.08039)

**💻 Code:** [⭐ Code](https://github.com/tau-intelligence/MuJoCo-drones-gym) • [⭐ Code](https://github.com/huggingface)

> We present MuJoCo-Drones-Gym, an open-source Gymnasium-compatible multi-drone environment built on top of the MuJoCo physics engine. Multi-drone environment for RL with MuJoCo, with GPU vectorization, wind models, domain randomization, and curricu...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-06-12.json`](data/daily/2026-06-12.json) | 22 |
| 📆 This Week | [`2026-W23.json`](data/weekly/2026-W23.json) | 78 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 256 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-12 | 22 | [View JSON](data/daily/2026-06-12.json) |
| 📄 2026-06-11 | 15 | [View JSON](data/daily/2026-06-11.json) |
| 📄 2026-06-10 | 18 | [View JSON](data/daily/2026-06-10.json) |
| 📄 2026-06-09 | 8 | [View JSON](data/daily/2026-06-09.json) |
| 📄 2026-06-08 | 15 | [View JSON](data/daily/2026-06-08.json) |
| 📄 2026-06-07 | 50 | [View JSON](data/daily/2026-06-07.json) |
| 📄 2026-06-06 | 50 | [View JSON](data/daily/2026-06-06.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W23 | 78 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 256 | [View JSON](data/monthly/2026-06.json) |
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
