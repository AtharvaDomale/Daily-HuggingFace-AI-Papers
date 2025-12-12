<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-275+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">137</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">324</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">275+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 12, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. StereoWorld: Geometry-Aware Monocular-to-Stereo Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guixun Luo, Hanwen Liang, Longfei Li, yuyangyin, KXingLab

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09363) • [📄 arXiv](https://arxiv.org/abs/2512.09363) • [📥 PDF](https://arxiv.org/pdf/2512.09363)

> StereoWorld presents geometry-aware monocular-to-stereo video generation using a pretrained video generator with geometry regularization and tiling for high-resolution, consistent stereo videos.

</details>

<details>
<summary><b>2. BrainExplore: Large-Scale Discovery of Interpretable Visual Representations in the Human Brain</b> ⭐ 0</summary>

<br/>

**👥 Authors:** tamarott, Antoniotorralbaborruel, yuvalgolbari, mcosarinsky, navvew

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08560) • [📄 arXiv](https://arxiv.org/abs/2512.08560) • [📥 PDF](https://arxiv.org/pdf/2512.08560)

> We present a large-scale, automated framework for discovering and explaining visual representations across the human cortex.

</details>

<details>
<summary><b>3. OmniPSD: Layered PSD Generation with Diffusion Transformer</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Cheng Liu, AnalMom, wanghaofan, yiren98

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09247) • [📄 arXiv](https://arxiv.org/abs/2512.09247) • [📥 PDF](https://arxiv.org/pdf/2512.09247)

> OmniPSD presents a diffusion-transformer framework for text-to-PSD generation and image-to-PSD decomposition, enabling layered, transparent PSDs with hierarchical, editable channels via in-context learning.

</details>

<details>
<summary><b>4. Composing Concepts from Images and Videos via Concept-prompt Binding</b> ⭐ 45</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09824) • [📄 arXiv](https://arxiv.org/abs/2512.09824) • [📥 PDF](https://arxiv.org/pdf/2512.09824)

**💻 Code:** [⭐ Code](https://github.com/refkxh/bico)

> We introduce Bind & Compose (BiCo), a one-shot method that enables flexible visual concept composition by binding visual concepts with the corresponding prompt tokens and composing the target prompt with bound tokens from various sources. 🌍 Projec...

</details>

<details>
<summary><b>5. InfiniteVL: Synergizing Linear and Sparse Attention for Highly-Efficient, Unlimited-Input Vision-Language Models</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08829) • [📄 arXiv](https://arxiv.org/abs/2512.08829) • [📥 PDF](https://arxiv.org/pdf/2512.08829)

**💻 Code:** [⭐ Code](https://github.com/hustvl/InfiniteVL)

> Window attention and linear attention represent two principal strategies for mitigating the quadratic complexity and ever-growing KV cache in Vision-Language Models (VLMs). However, we observe that window-based VLMs suffer performance degradation ...

</details>

<details>
<summary><b>6. HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09928) • [📄 arXiv](https://arxiv.org/abs/2512.09928) • [📥 PDF](https://arxiv.org/pdf/2512.09928)

**💻 Code:** [⭐ Code](https://github.com/OpenHelix-Team/HiF-VLA)

> Code and checkpoints are available! Github: https://github.com/OpenHelix-Team/HiF-VLA Project page: https://hifvla.github.io/

</details>

<details>
<summary><b>7. Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yang Zhang, guokan-shang, mvazirg, amr-mohamed

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.02892) • [📄 arXiv](https://arxiv.org/abs/2512.02892) • [📥 PDF](https://arxiv.org/pdf/2512.02892)

> SchED introduces a training-free, early-exit decoding criterion for diffusion LLMs , halting sampling once a smooth, progress-adaptive confidence threshold is satisfied. SchED achieves up to ~4× decoding speedups on average with ≥99–100% performan...

</details>

<details>
<summary><b>8. Rethinking Chain-of-Thought Reasoning for Videos</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Liwei Wang, Yin Li, Zi-Yuan Hu, Yiwu Zhong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09616) • [📄 arXiv](https://arxiv.org/abs/2512.09616) • [📥 PDF](https://arxiv.org/pdf/2512.09616)

**💻 Code:** [⭐ Code](https://github.com/LaVi-Lab/Rethink_CoT_Video)

> Rethinking Chain-of-Thought Reasoning for Videos

</details>

<details>
<summary><b>9. EtCon: Edit-then-Consolidate for Reliable Knowledge Editing</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Chenglin Li, Wenhong Zhu, Ruilin Li, Rethinker, CodeGoat24

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04753) • [📄 arXiv](https://arxiv.org/abs/2512.04753) • [📥 PDF](https://arxiv.org/pdf/2512.04753)

**💻 Code:** [⭐ Code](https://github.com/RlinL/EtCon)

> EtCon: Edit-then-Consolidate for Reliable Knowledge Editing

</details>

<details>
<summary><b>10. UniUGP: Unifying Understanding, Generation, and Planing For End-to-end Autonomous Driving</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09864) • [📄 arXiv](https://arxiv.org/abs/2512.09864) • [📥 PDF](https://arxiv.org/pdf/2512.09864)

> Proposes UniUGP, a unified framework integrating scene understanding, video generation, and trajectory planning for autonomous driving with visual reasoning.

</details>

<details>
<summary><b>11. WonderZoom: Multi-Scale 3D World Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiajun Wu, Hong-Xing Yu, Jin Cao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09164) • [📄 arXiv](https://arxiv.org/abs/2512.09164) • [📥 PDF](https://arxiv.org/pdf/2512.09164)

> WonderZoom enables multi-scale 3D world generation from a single image via scale-adaptive Gaussian surfels and progressive detail synthesis for zoomed-in realism.

</details>

<details>
<summary><b>12. Learning Unmasking Policies for Diffusion Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09106) • [📄 arXiv](https://arxiv.org/abs/2512.09106) • [📥 PDF](https://arxiv.org/pdf/2512.09106)

> Trains a lightweight RL-based policy to unmask tokens in masked diffusion LMs, achieving competitive performance with heuristics and generalizing to new models and longer sequences.

</details>

<details>
<summary><b>13. Towards a Science of Scaling Agent Systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Samuel Schmidgall, Chunjong Park, Chanwoo Park, Ken Gu, Yubin Kim

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08296) • [📄 arXiv](https://arxiv.org/abs/2512.08296) • [📥 PDF](https://arxiv.org/pdf/2512.08296)

> No abstract available.

</details>

<details>
<summary><b>14. IF-Bench: Benchmarking and Enhancing MLLMs for Infrared Images with Generative Visual Prompting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09663) • [📄 arXiv](https://arxiv.org/abs/2512.09663) • [📥 PDF](https://arxiv.org/pdf/2512.09663)

> Recent advances in multimodal large language models (MLLMs) have led to impressive progress across various benchmarks. However, their capability in understanding infrared images remains unexplored. To address this gap, we introduce IF-Bench, the f...

</details>

<details>
<summary><b>15. TED-4DGS: Temporally Activated and Embedding-based Deformation for 4DGS Compression</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05446) • [📄 arXiv](https://arxiv.org/abs/2512.05446) • [📥 PDF](https://arxiv.org/pdf/2512.05446)

> Building on the success of 3D Gaussian Splatting (3DGS) in static 3D scene representation, its extension to dynamic scenes, commonly referred to as 4DGS or dynamic 3DGS, has attracted increasing attention. However, designing more compact and effic...

</details>

<details>
<summary><b>16. Beyond Unified Models: A Service-Oriented Approach to Low Latency, Context Aware Phonemization for Real Time TTS</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Morteza Abolghasemi, hrrabiee, ZahraDehghanian97, dninvb, MahtaFetrat

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08006) • [📄 arXiv](https://arxiv.org/abs/2512.08006) • [📥 PDF](https://arxiv.org/pdf/2512.08006)

**💻 Code:** [⭐ Code](https://github.com/MahtaFetrat/Piper-with-LCA-Phonemizer)

> Lightweight, real-time text-to-speech systems are crucial for accessibility. However, the most efficient TTS models often rely on lightweight phonemizers that struggle with context-dependent challenges. In contrast, more advanced phonemizers with ...

</details>

<details>
<summary><b>17. VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04519) • [📄 arXiv](https://arxiv.org/abs/2512.04519) • [📥 PDF](https://arxiv.org/pdf/2512.04519)

> We introduce VideoSSM, an AR video diffusion model equipped with a novel hybrid memory architecture that combines a causal sliding-window local lossless cache with an SSM-based global compressed memory for long video generation.

</details>

<details>
<summary><b>18. GimbalDiffusion: Gravity-Aware Camera Control for Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09112) • [📄 arXiv](https://arxiv.org/abs/2512.09112) • [📥 PDF](https://arxiv.org/pdf/2512.09112)

> No abstract available.

</details>

<details>
<summary><b>19. Pay Less Attention to Function Words for Free Robustness of Vision-Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07222) • [📄 arXiv](https://arxiv.org/abs/2512.07222) • [📥 PDF](https://arxiv.org/pdf/2512.07222)

**💻 Code:** [⭐ Code](https://github.com/michaeltian108/FDA)

> We had an interesting yet explorable observation that lowering the attention on function words of VLMs increaes robustness and zero-shot performance on several datasets/models/tasks, casuing little or no performance drops , surpasing SOTA adversar...

</details>

<details>
<summary><b>20. Smart Timing for Mining: A Deep Learning Framework for Bitcoin Hardware ROI Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05402) • [📄 arXiv](https://arxiv.org/abs/2512.05402) • [📥 PDF](https://arxiv.org/pdf/2512.05402)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API A FEDformer-Based Hybrid Framework for Anomaly Detection and Risk Forecasti...

</details>

<details>
<summary><b>21. Reinventing Clinical Dialogue: Agentic Paradigms for LLM Enabled Healthcare Communication</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hengshu Zhu, Hongke Zhao, ChuangZhao, likang03, zxq1942461723

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.01453) • [📄 arXiv](https://arxiv.org/abs/2512.01453) • [📥 PDF](https://arxiv.org/pdf/2512.01453)

**💻 Code:** [⭐ Code](https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue)

> Fresh medical LLM survey

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2025-12-12.json`](data/daily/2025-12-12.json) | 21 |
| 📆 This Week | [`2025-W49.json`](data/weekly/2025-W49.json) | 137 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 324 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-12 | 21 | [View JSON](data/daily/2025-12-12.json) |
| 📄 2025-12-11 | 25 | [View JSON](data/daily/2025-12-11.json) |
| 📄 2025-12-10 | 29 | [View JSON](data/daily/2025-12-10.json) |
| 📄 2025-12-09 | 24 | [View JSON](data/daily/2025-12-09.json) |
| 📄 2025-12-08 | 38 | [View JSON](data/daily/2025-12-08.json) |
| 📄 2025-12-07 | 38 | [View JSON](data/daily/2025-12-07.json) |
| 📄 2025-12-06 | 38 | [View JSON](data/daily/2025-12-06.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W49 | 137 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 324 | [View JSON](data/monthly/2025-12.json) |

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
