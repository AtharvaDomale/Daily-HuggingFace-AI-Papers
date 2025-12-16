<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-370+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">21</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">46</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">419</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">370+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 16, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. EgoX: Egocentric Video Generation from a Single Exocentric Video</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08269) • [📄 arXiv](https://arxiv.org/abs/2512.08269) • [📥 PDF](https://arxiv.org/pdf/2512.08269)

**💻 Code:** [⭐ Code](https://github.com/KEH0T0/EgoX)

> No abstract available.

</details>

<details>
<summary><b>2. DentalGPT: Incentivizing Multimodal Complex Reasoning in Dentistry</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yanchao Li, Junjie Zhao, Jiaming Zhang, Zhenyang Cai, CocoNutZENG

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11558) • [📄 arXiv](https://arxiv.org/abs/2512.11558) • [📥 PDF](https://arxiv.org/pdf/2512.11558)

> Reliable interpretation of multimodal data in dentistry is essential for automated oral healthcare, yet current multimodal large language models (MLLMs) struggle to capture fine-grained dental visual details and lack sufficient reasoning ability f...

</details>

<details>
<summary><b>3. SVG-T2I: Scaling Up Text-to-Image Latent Diffusion Model Without Variational Autoencoder</b> ⭐ 40</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11749) • [📄 arXiv](https://arxiv.org/abs/2512.11749) • [📥 PDF](https://arxiv.org/pdf/2512.11749)

**💻 Code:** [⭐ Code](https://github.com/KlingTeam/SVG-T2I)

> Visual generation grounded in Visual Foundation Model (VFM) representations offers a highly promising unified pathway for integrating visual understanding, perception, and generation. Despite this potential, training large-scale text-to-image diff...

</details>

<details>
<summary><b>4. V-RGBX: Video Editing with Accurate Controls over Intrinsic Properties</b> ⭐ 40</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11799) • [📄 arXiv](https://arxiv.org/abs/2512.11799) • [📥 PDF](https://arxiv.org/pdf/2512.11799)

**💻 Code:** [⭐ Code](https://github.com/Aleafy/V-RGBX)

> Large-scale video generation models have shown remarkable potential in modeling photorealistic appearance and lighting interactions in real-world scenes. However, a closed-loop framework that jointly understands intrinsic scene properties (e.g., a...

</details>

<details>
<summary><b>5. Sliding Window Attention Adaptation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10411) • [📄 arXiv](https://arxiv.org/abs/2512.10411) • [📥 PDF](https://arxiv.org/pdf/2512.10411)

**💻 Code:** [⭐ Code](https://github.com/yuyijiong/sliding-window-attention-adaptation)

> We propose a set of practical recipes that can let a full-attention LLM use sliding window attention to improve efficiency. For example, some can achieve nearly 100% acceleration of LLM long-context inference speed with 90% accuracy retainment; so...

</details>

<details>
<summary><b>6. PersonaLive! Expressive Portrait Image Animation for Live Streaming</b> ⭐ 206</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11253) • [📄 arXiv](https://arxiv.org/abs/2512.11253) • [📥 PDF](https://arxiv.org/pdf/2512.11253)

**💻 Code:** [⭐ Code](https://github.com/GVCLab/PersonaLive)

> Current diffusion-based portrait animation models predominantly focus on enhancing visual quality and expression realism, while overlooking generation latency and real-time performance, which restricts their application range in the live streaming...

</details>

<details>
<summary><b>7. Exploring MLLM-Diffusion Information Transfer with MetaCanvas</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11464) • [📄 arXiv](https://arxiv.org/abs/2512.11464) • [📥 PDF](https://arxiv.org/pdf/2512.11464)

> Multimodal learning has rapidly advanced visual understanding, largely via multimodal large language models (MLLMs) that use powerful LLMs as cognitive cores. In visual generation, however, these powerful core models are typically reduced to globa...

</details>

<details>
<summary><b>8. Structure From Tracking: Distilling Structure-Preserving Motion for Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qifeng Chen, Jingyuan Liu, George Stoica, Tim666, sunfly

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11792) • [📄 arXiv](https://arxiv.org/abs/2512.11792) • [📥 PDF](https://arxiv.org/pdf/2512.11792)

> We introduce an algorithm to distill structure-preserving motion priors from an autoregressive video tracking model (SAM2) into a bidirectional video diffusion model (CogVideoX).

</details>

<details>
<summary><b>9. MeshSplatting: Differentiable Rendering with Opaque Meshes</b> ⭐ 273</summary>

<br/>

**👥 Authors:** Matheus Gadelha, Daniel Rebain, Renaud Vandeghen, Sanghyun Son, Jan Held

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06818) • [📄 arXiv](https://arxiv.org/abs/2512.06818) • [📥 PDF](https://arxiv.org/pdf/2512.06818)

**💻 Code:** [⭐ Code](https://github.com/meshsplatting/mesh-splatting)

> MeshSplatting introduces a differentiable rendering approach that reconstructs connected, fully opaque triangle meshes for fast, memory efficient, high quality novel view synthesis.

</details>

<details>
<summary><b>10. LEO-RobotAgent: A General-purpose Robotic Agent for Language-driven Embodied Operator</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10605) • [📄 arXiv](https://arxiv.org/abs/2512.10605) • [📥 PDF](https://arxiv.org/pdf/2512.10605)

**💻 Code:** [⭐ Code](https://github.com/LegendLeoChen/LEO-RobotAgent)

> A general-purpose robotic agent framework based on LLMs. The LLM can independently reason, plan, and execute actions to operate diverse robot types across various scenarios to complete unpredictable, complex tasks.

</details>

<details>
<summary><b>11. Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems</b> ⭐ 7</summary>

<br/>

**👥 Authors:** elandy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11150) • [📄 arXiv](https://arxiv.org/abs/2512.11150) • [📥 PDF](https://arxiv.org/pdf/2512.11150)

**💻 Code:** [⭐ Code](https://github.com/cimo-labs/cje)

> LLM-as-judge evals are convenient, but meaningful (fixable) failure modes lurk beneath the surface. CJE treats LLM-judge evaluation as a statistics problem: • calibrate a cheap judge to a small oracle slice of high-quality labels • quantify uncert...

</details>

<details>
<summary><b>12. Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in {pm 1, pm i}</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.02901) • [📄 arXiv](https://arxiv.org/abs/2512.02901) • [📥 PDF](https://arxiv.org/pdf/2512.02901)

**💻 Code:** [⭐ Code](https://github.com/PKULab1806/Fairy2i-W2)

> Is it possible to run LLMs at 2-bit with virtually NO loss in accuracy? 🤔 No with Real numbers, but Yes with Complex ones! 🚀 Meet Fairy2i-W2(2bit): QAT from LLaMA-2 7B with Complex Phase quant PPL: 7.85 (vs FP16's 6.63) Accuracy: 62.00% (vs FP16's...

</details>

<details>
<summary><b>13. CLINIC: Evaluating Multilingual Trustworthiness in Language Models for Healthcare</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11437) • [📄 arXiv](https://arxiv.org/abs/2512.11437) • [📥 PDF](https://arxiv.org/pdf/2512.11437)

> First and largest multilingual trustworthiness benchmark for healthcare

</details>

<details>
<summary><b>14. Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge</b> ⭐ 111</summary>

<br/>

**👥 Authors:** Akash Karnatak, Gleb Zarin, IliaLarchenko

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06951) • [📄 arXiv](https://arxiv.org/abs/2512.06951) • [📥 PDF](https://arxiv.org/pdf/2512.06951)

**💻 Code:** [⭐ Code](https://github.com/IliaLarchenko/behavior-1k-solution)

> We present our 1st place solution to the 2025 NeurIPS BEHAVIOR Challenge, where a single Vision-Language-Action robotics policy is trained to perform 50 household manipulation tasks in a photorealistic simulator. The approach builds on Pi0.5 with ...

</details>

<details>
<summary><b>15. Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching</b> ⭐ 45</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11130) • [📄 arXiv](https://arxiv.org/abs/2512.11130) • [📥 PDF](https://arxiv.org/pdf/2512.11130)

**💻 Code:** [⭐ Code](https://github.com/NVlabs/Fast-FoundationStereo)

> A real-time foundation model for stereo depth estimation, which is crucial for robotics/humanoid 3D spatial perception.

</details>

<details>
<summary><b>16. Scaling Behavior of Discrete Diffusion Language Models</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10858) • [📄 arXiv](https://arxiv.org/abs/2512.10858) • [📥 PDF](https://arxiv.org/pdf/2512.10858)

**💻 Code:** [⭐ Code](https://github.com/dvruette/gidd-easydel)

> We scale diffusion language models up to 3B (masked and uniform diffusion) and 10B (uniform diffusion) parameters,  pre-trained on a pure diffusion objective (mixture of unconditional and conditional) via Nemotron-CC. 🤖 GitHub: https://github.com/...

</details>

<details>
<summary><b>17. CheXmask-U: Quantifying uncertainty in landmark-based anatomical segmentation for X-ray images</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Enzo Ferrante, Rodrigo Echeveste, Nicolas Gaggion, Matias Cosarinsky

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10715) • [📄 arXiv](https://arxiv.org/abs/2512.10715) • [📥 PDF](https://arxiv.org/pdf/2512.10715)

**💻 Code:** [⭐ Code](https://github.com/mcosarinsky/CheXmask-U)

> We present CheXmask-U , a framework for quantifying uncertainty in landmark-based anatomical segmentation models on chest X-rays and release the CheXmask-U dataset providing per-node uncertainty estimates to support research in robust and safe med...

</details>

<details>
<summary><b>18. The N-Body Problem: Parallel Execution from Single-Person Egocentric Video</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dima Damen, Yoichi Sato, Yifei Huang, Zhifan Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11393) • [📄 arXiv](https://arxiv.org/abs/2512.11393) • [📥 PDF](https://arxiv.org/pdf/2512.11393)

> Humans can intuitively parallelise complex activities, but can a model learn this from observing a single person? Given one egocentric video, we introduce the N-Body Problem: how N individuals, can hypothetically perform the same set of tasks obse...

</details>

<details>
<summary><b>19. Sharp Monocular View Synthesis in Less Than a Second</b> ⭐ 139</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10685) • [📄 arXiv](https://arxiv.org/abs/2512.10685) • [📥 PDF](https://arxiv.org/pdf/2512.10685)

**💻 Code:** [⭐ Code](https://github.com/apple/ml-sharp)

> Sharp Monocular View Synthesis in Less Than a Second https://huggingface.co/papers/2512.10685 Real-time photorealistic view synthesis from a single image. Given a single photograph, regresses the parameters of a 3D Gaussian representation of the d...

</details>

<details>
<summary><b>20. Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Neel Nanda, Lewis Smith, Lisa Dunlap, Xiaoqing Sun, Nick Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10092) • [📄 arXiv](https://arxiv.org/abs/2512.10092) • [📥 PDF](https://arxiv.org/pdf/2512.10092)

**💻 Code:** [⭐ Code](https://github.com/nickjiang2378/interp_embed)

> Analyzing large-scale text corpora is a core challenge in machine learning, crucial for tasks like identifying undesirable model behaviors or biases in training data. Current methods often rely on costly LLM-based techniques (e.g. annotating datas...

</details>

<details>
<summary><b>21. Particulate: Feed-Forward 3D Object Articulation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Joan Lasenby, Christian Rupprecht, Chuanxia Zheng, Yuxin Yao, Ruining Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11798) • [📄 arXiv](https://arxiv.org/abs/2512.11798) • [📥 PDF](https://arxiv.org/pdf/2512.11798)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2025-12-16.json`](data/daily/2025-12-16.json) | 21 |
| 📆 This Week | [`2025-W50.json`](data/weekly/2025-W50.json) | 46 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 419 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-16 | 21 | [View JSON](data/daily/2025-12-16.json) |
| 📄 2025-12-15 | 25 | [View JSON](data/daily/2025-12-15.json) |
| 📄 2025-12-14 | 25 | [View JSON](data/daily/2025-12-14.json) |
| 📄 2025-12-13 | 24 | [View JSON](data/daily/2025-12-13.json) |
| 📄 2025-12-12 | 21 | [View JSON](data/daily/2025-12-12.json) |
| 📄 2025-12-11 | 25 | [View JSON](data/daily/2025-12-11.json) |
| 📄 2025-12-10 | 29 | [View JSON](data/daily/2025-12-10.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W50 | 46 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 419 | [View JSON](data/monthly/2025-12.json) |

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
