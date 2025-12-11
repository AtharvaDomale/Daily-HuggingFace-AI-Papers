<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-25-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-254+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">116</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">303</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">254+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 11, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance</b> ⭐ 197</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08765) • [📄 arXiv](https://arxiv.org/abs/2512.08765) • [📥 PDF](https://arxiv.org/pdf/2512.08765)

**💻 Code:** [⭐ Code](https://github.com/ali-vilab/Wan-Move)

> NeurIPS 2025: Wan-Move: Motion-controllable Video Generation viaLatent Trajectory Guidance

</details>

<details>
<summary><b>2. Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform</b> ⭐ 162</summary>

<br/>

**👥 Authors:** Muyao Niu, Yifan Zhan, Yifei Liu, Yuning Gong, Zuica96

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08478) • [📄 arXiv](https://arxiv.org/abs/2512.08478) • [📥 PDF](https://arxiv.org/pdf/2512.08478)

**💻 Code:** [⭐ Code](https://github.com/Visionary-Laboratory/visionary)

> TL;DR: Visionary is an open, web-native platform built on WebGPU and ONNX Runtime. Enabling real-time rendering of diverse Gaussian Splatting variants (3DGS, MLP-based 3DGS, 4DGS, Neural Avatars and ✨any future algorithms✨), and traditional 3d Mes...

</details>

<details>
<summary><b>3. Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07951) • [📄 arXiv](https://arxiv.org/abs/2512.07951) • [📥 PDF](https://arxiv.org/pdf/2512.07951)

> Project webpage: this https URL

</details>

<details>
<summary><b>4. OneStory: Coherent Multi-Shot Video Generation with Adaptive Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07802) • [📄 arXiv](https://arxiv.org/abs/2512.07802) • [📥 PDF](https://arxiv.org/pdf/2512.07802)

> No abstract available.

</details>

<details>
<summary><b>5. ThreadWeaver: Adaptive Threading for Efficient Parallel Reasoning in Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiuyu Li, Tsu-Jui Fu, Sida Wang, katanaxu, longlian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07843) • [📄 arXiv](https://arxiv.org/abs/2512.07843) • [📥 PDF](https://arxiv.org/pdf/2512.07843)

> No abstract available.

</details>

<details>
<summary><b>6. Boosting Unsupervised Video Instance Segmentation with Automatic Quality-Guided Self-Training</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Dim P. Papadopoulos, Kaixuan Lu, monurcan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06864) • [📄 arXiv](https://arxiv.org/abs/2512.06864) • [📥 PDF](https://arxiv.org/pdf/2512.06864)

**💻 Code:** [⭐ Code](https://github.com/wcbup/AutoQ-VIS/)

> Accepted at WACV'26! Keywords: Video Instance Segmentation; Unsupervised Learning; Segmentation Quality Assessment

</details>

<details>
<summary><b>7. Arbitrage: Efficient Reasoning via Advantage-Aware Speculation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05033) • [📄 arXiv](https://arxiv.org/abs/2512.05033) • [📥 PDF](https://arxiv.org/pdf/2512.05033)

**💻 Code:** [⭐ Code](https://github.com/SqueezeAILab/Arbitrage)

> Modern large language models achieve impressive reasoning capabilities with long chains of thought, but they incur substantial computational cost at inference time. Speculative decoding improves efficiency by using a fast, less accurate draft mode...

</details>

<details>
<summary><b>8. MIND-V: Hierarchical Video Generation for Long-Horizon Robotic Manipulation with RL-based Physical Alignment</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06628) • [📄 arXiv](https://arxiv.org/abs/2512.06628) • [📥 PDF](https://arxiv.org/pdf/2512.06628)

**💻 Code:** [⭐ Code](https://github.com/Richard-Zhang-AI/MIND-V)

> We propose MIND-V, a hierarchical framework designed to synthesize physically plausible and logically coherent videos of long-horizon robotic manipulation.

</details>

<details>
<summary><b>9. See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.02231) • [📄 arXiv](https://arxiv.org/abs/2512.02231) • [📥 PDF](https://arxiv.org/pdf/2512.02231)

**💻 Code:** [⭐ Code](https://github.com/plnguyen2908/AV-SpeakerBench)

> Multimodal large language models (MLLMs) are expected to jointly interpret vision, audio, and language, yet existing video benchmarks rarely assess fine-grained reasoning about human speech. Many tasks remain visually solvable or only coarsely eva...

</details>

<details>
<summary><b>10. DeepCode: Open Agentic Coding</b> ⭐ 11.8k</summary>

<br/>

**👥 Authors:** Chao Huang, Xubin Ren, Zirui Guo, Zhonghang Li, Zongwei Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07921) • [📄 arXiv](https://arxiv.org/abs/2512.07921) • [📥 PDF](https://arxiv.org/pdf/2512.07921)

**💻 Code:** [⭐ Code](https://github.com/HKUDS/DeepCode)

> Recent advances in large language models (LLMs) have given rise to powerful coding agents, making it possible for code assistants to evolve into code engineers. However, existing methods still face significant challenges in achieving high-fidelity...

</details>

<details>
<summary><b>11. TreeGRPO: Tree-Advantage GRPO for Online RL Post-Training of Diffusion Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Weirui Ye, Zheng Ding

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08153) • [📄 arXiv](https://arxiv.org/abs/2512.08153) • [📥 PDF](https://arxiv.org/pdf/2512.08153)

> Reinforcement learning (RL) post-training is crucial for aligning generative models with human preferences, but its prohibitive computational cost remains a major barrier to widespread adoption. We introduce \textbf{TreeGRPO}, a novel RL framework...

</details>

<details>
<summary><b>12. From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06776) • [📄 arXiv](https://arxiv.org/abs/2512.06776) • [📥 PDF](https://arxiv.org/pdf/2512.06776)

> NBDiff: A principled path from AR to Diffusion LLMs

</details>

<details>
<summary><b>13. Efficiently Reconstructing Dynamic Scenes One D4RT at a Time</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08924) • [📄 arXiv](https://arxiv.org/abs/2512.08924) • [📥 PDF](https://arxiv.org/pdf/2512.08924)

> 📍 A simple, unified interface for 3D tracking, depth, and pose 🌟 SOTA results on 4D reconstruction & tracking 🚀 Up to 100x faster pose estimation than prior works

</details>

<details>
<summary><b>14. Modular Neural Image Signal Processing</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Michael S. Brown, Ran Zhang, Zhongling Wang, mafifi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08564) • [📄 arXiv](https://arxiv.org/abs/2512.08564) • [📥 PDF](https://arxiv.org/pdf/2512.08564)

**💻 Code:** [⭐ Code](https://github.com/mahmoudnafifi/modular_neural_isp)

> Modular Neural Image Signal Processing 🎬 Click to watch the video We present a modular neural image signal processing (ISP) framework that produces high-quality display-referred images while providing a high degree of modularity with explicit cont...

</details>

<details>
<summary><b>15. Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation</b> ⭐ 455</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08186) • [📄 arXiv](https://arxiv.org/abs/2512.08186) • [📥 PDF](https://arxiv.org/pdf/2512.08186)

**💻 Code:** [⭐ Code](https://github.com/InternRobotics/InternNav)

> Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-Language Navigation

</details>

<details>
<summary><b>16. EcomBench: Towards Holistic Evaluation of Foundation Agents in E-commerce</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08868) • [📄 arXiv](https://arxiv.org/abs/2512.08868) • [📥 PDF](https://arxiv.org/pdf/2512.08868)

> EcomBench introduces a holistic e-commerce benchmark to evaluate foundation agents on real-world tasks, emphasizing deep retrieval, multi-step reasoning, and cross-source knowledge integration.

</details>

<details>
<summary><b>17. TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tianyu Huang, Peng Li, Jiacheng Deng, Jiahao Lu, xwt123

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08358) • [📄 arXiv](https://arxiv.org/abs/2512.08358) • [📥 PDF](https://arxiv.org/pdf/2512.08358)

> Monocular 3D tracking aims to capture the long-term motion of pixels in 3D space from a single monocular video and has witnessed rapid progress in recent years. However, we argue that the existing monocular 3D tracking methods still fall short in ...

</details>

<details>
<summary><b>18. SUCCESS-GS: Survey of Compactness and Compression for Efficient Static and Dynamic Gaussian Splatting</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Soohyun Lee, Seokhyun Youn, ozbro, shbae84, klavna

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07197) • [📄 arXiv](https://arxiv.org/abs/2512.07197) • [📥 PDF](https://arxiv.org/pdf/2512.07197)

**💻 Code:** [⭐ Code](https://github.com/CMLab-Korea/Awesome-Efficient-GS)

> project page: https://cmlab-korea.github.io/Awesome-Efficient-GS/

</details>

<details>
<summary><b>19. Novel Deep Learning Architectures for Classification and Segmentation of Brain Tumors from MRI Images</b> ⭐ 0</summary>

<br/>

**👥 Authors:** arghadip2002, Necromancer0912

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06531) • [📄 arXiv](https://arxiv.org/abs/2512.06531) • [📥 PDF](https://arxiv.org/pdf/2512.06531)

**💻 Code:** [⭐ Code](https://github.com/arghadip2002/SAETCN-and-SASNET-Architectures)

> We are excited to share our new work tackling the critical challenge of brain tumor detection from MRI scans. Due to high data volume and generalization issues in existing systems, we developed two novel deep learning architectures: SAETCN (Self-A...

</details>

<details>
<summary><b>20. LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05325) • [📄 arXiv](https://arxiv.org/abs/2512.05325) • [📥 PDF](https://arxiv.org/pdf/2512.05325)

**💻 Code:** [⭐ Code](https://github.com/farukakgul/LYNX)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference ...

</details>

<details>
<summary><b>21. SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Jungong Han, Yunqi Miao, gaomingqi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08406) • [📄 arXiv](https://arxiv.org/abs/2512.08406) • [📥 PDF](https://arxiv.org/pdf/2512.08406)

**💻 Code:** [⭐ Code](https://github.com/gaomingqi/sam-body4d)

> Code & Gradio Demo : https://github.com/gaomingqi/sam-body4d See our FULL demo and Gradio Demo video below:

</details>

<details>
<summary><b>22. MemLoRA: Distilling Expert Adapters for On-Device Memory Systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mete Ozay, Zeynep Akata, Umberto Michieli, Ondrej Bohdal, mwbini

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04763) • [📄 arXiv](https://arxiv.org/abs/2512.04763) • [📥 PDF](https://arxiv.org/pdf/2512.04763)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API LightMem: Lightweight and Efficient Memory-Augmented Generation (2025) MemV...

</details>

<details>
<summary><b>23. Predicting Time-Dependent Flow Over Complex Geometries Using Operator Networks</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04434) • [📄 arXiv](https://arxiv.org/abs/2512.04434) • [📥 PDF](https://arxiv.org/pdf/2512.04434)

**💻 Code:** [⭐ Code](https://github.com/baskargroup/TimeDependent-DeepONet)

> This paper introduces a new deep learning algorithem to model transient flow around varied complex geometries using the deep operator network (DeepONet)

</details>

<details>
<summary><b>24. Same Content, Different Answers: Cross-Modal Inconsistency in MLLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08923) • [📄 arXiv](https://arxiv.org/abs/2512.08923) • [📥 PDF](https://arxiv.org/pdf/2512.08923)

> Paper that evaluates and analyses consistency of MLLMs when providing questions in text vs as rendered-text.

</details>

<details>
<summary><b>25. Terrain Diffusion: A Diffusion-Based Successor to Perlin Noise in Infinite, Real-Time Terrain Generation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** xandergos

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08309) • [📄 arXiv](https://arxiv.org/abs/2512.08309) • [📥 PDF](https://arxiv.org/pdf/2512.08309)

**💻 Code:** [⭐ Code](https://github.com/xandergos/terrain-diffusion)

> Terrain Diffusion introduces a procedural generation primitive built around InfiniteDiffusion, a sampling method that delivers seamless, seed-consistent, infinite-domain generation with constant-time random access. A multi-scale diffusion hierarch...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 25 |
| 📅 Today | [`2025-12-11.json`](data/daily/2025-12-11.json) | 25 |
| 📆 This Week | [`2025-W49.json`](data/weekly/2025-W49.json) | 116 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 303 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-11 | 25 | [View JSON](data/daily/2025-12-11.json) |
| 📄 2025-12-10 | 29 | [View JSON](data/daily/2025-12-10.json) |
| 📄 2025-12-09 | 24 | [View JSON](data/daily/2025-12-09.json) |
| 📄 2025-12-08 | 38 | [View JSON](data/daily/2025-12-08.json) |
| 📄 2025-12-07 | 38 | [View JSON](data/daily/2025-12-07.json) |
| 📄 2025-12-06 | 38 | [View JSON](data/daily/2025-12-06.json) |
| 📄 2025-12-05 | 38 | [View JSON](data/daily/2025-12-05.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W49 | 116 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 303 | [View JSON](data/monthly/2025-12.json) |

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
