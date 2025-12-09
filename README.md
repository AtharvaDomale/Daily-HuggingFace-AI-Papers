<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-24-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-200+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">24</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">62</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">249</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">200+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 09, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05150) • [📄 arXiv](https://arxiv.org/abs/2512.05150) • [📥 PDF](https://arxiv.org/pdf/2512.05150)

> Taming 20B full-parameter few-step training with self-adversarial flows! 👏🏻 One-model Simplicity: We eliminate the need for auxiliary networks (discriminators, teachers, fake score estimators...), everything in one model! Scalability on Large Mode...

</details>

<details>
<summary><b>2. EditThinker: Unlocking Iterative Reasoning for Any Image Editor</b> ⭐ 25</summary>

<br/>

**👥 Authors:** Ziyu Guo, Manyuan Zhang, Longin-Yu, zhengli1013, appletea2333

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05965) • [📄 arXiv](https://arxiv.org/abs/2512.05965) • [📥 PDF](https://arxiv.org/pdf/2512.05965)

**💻 Code:** [⭐ Code](https://github.com/appletea233/EditThinker)

> Instruction-based image editing has emerged as a prominent research area. Benefiting from image generation foundation models, it has achieved high aesthetic quality, making instruction-following capability the primary challenge. Existing approache...

</details>

<details>
<summary><b>3. From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yang Li, Shuai Zhang, Yuchen Liu, Jinyang Wu, Changpeng Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.02580) • [📄 arXiv](https://arxiv.org/abs/2512.02580) • [📥 PDF](https://arxiv.org/pdf/2512.02580)

> 🚀 [New Paper] CAPO: From Imitation to Discrimination – Rethinking Advantage in RL Early RL training often suffers from instability due to "mixed signals" (simultaneous positive & negative feedback). Inspired by child cognitive development, we prop...

</details>

<details>
<summary><b>4. EMMA: Efficient Multimodal Understanding, Generation, and Editing with a Unified Architecture</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Qi Tian, Lingxi Xie, Jianbo Ouyang, Longhui Wei, Xin He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04810) • [📄 arXiv](https://arxiv.org/abs/2512.04810) • [📥 PDF](https://arxiv.org/pdf/2512.04810)

**💻 Code:** [⭐ Code](https://github.com/umm-emma/emma)

> The project page https://emma-umm.github.io/emma/

</details>

<details>
<summary><b>5. PaCo-RL: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04784) • [📄 arXiv](https://arxiv.org/abs/2512.04784) • [📥 PDF](https://arxiv.org/pdf/2512.04784)

**💻 Code:** [⭐ Code](https://github.com/X-GenGroup/PaCo-RL)

> Consistent image generation requires faithfully preserving identities, styles, and logical coherence across multiple images, which is essential for applications such as storytelling and character design. Supervised training approaches struggle wit...

</details>

<details>
<summary><b>6. SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05905) • [📄 arXiv](https://arxiv.org/abs/2512.05905) • [📥 PDF](https://arxiv.org/pdf/2512.05905)

> SCAIL is a new framework for studio-grade character animation that uses a novel 3D pose representation and full-sequence context injection to deliver more stable, realistic motion transfer under complex and cross-identity scenarios.

</details>

<details>
<summary><b>7. Entropy Ratio Clipping as a Soft Global Constraint for Stable Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zijia Lin, Tiehua Mei, Minxuan Lv, Leiyu Pan, Suu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05591) • [📄 arXiv](https://arxiv.org/abs/2512.05591) • [📥 PDF](https://arxiv.org/pdf/2512.05591)

> Large language model post-training relies on reinforcement learning to improve model capability and alignment quality. However, the off-policy training paradigm introduces distribution shift, which often pushes the policy beyond the trust region, ...

</details>

<details>
<summary><b>8. Joint 3D Geometry Reconstruction and Motion Generation for 4D Synthesis from a Single Image</b> ⭐ 37</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05044) • [📄 arXiv](https://arxiv.org/abs/2512.05044) • [📥 PDF](https://arxiv.org/pdf/2512.05044)

**💻 Code:** [⭐ Code](https://github.com/Zhangyr2022/MoRe4D)

> Project Page: https://ivg-yanranzhang.github.io/MoRe4D/ Github Repo: https://github.com/Zhangyr2022/MoRe4D The dataset is coming soon. Stay tuned!

</details>

<details>
<summary><b>9. COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Jiawei Sheng, Zhenyu Zhang, Hengzhu Tang, CUDAOUTOFMEMORY, Starrrrrry

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04563) • [📄 arXiv](https://arxiv.org/abs/2512.04563) • [📥 PDF](https://arxiv.org/pdf/2512.04563)

**💻 Code:** [⭐ Code](https://github.com/zhangzef/COOPER)

> Visual Spatial Reasoning is crucial for enabling Multimodal Large Language Models (MLLMs) to understand object properties and spatial relationships, yet current models still struggle with 3D-aware reasoning. Existing approaches typically enhance e...

</details>

<details>
<summary><b>10. RealGen: Photorealistic Text-to-Image Generation via Detector-Guided Rewards</b> ⭐ 65</summary>

<br/>

**👥 Authors:** Zilong Huang, Dongzhi Jiang, Yuncheng Guo, Leiqi Zhu, Junyan Ye

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.00473) • [📄 arXiv](https://arxiv.org/abs/2512.00473) • [📥 PDF](https://arxiv.org/pdf/2512.00473)

**💻 Code:** [⭐ Code](https://github.com/yejy53/RealGen)

> arXiv: https://arxiv.org/abs/2512.00473 code: https://github.com/yejy53/RealGen project page: https://yejy53.github.io/RealGen/

</details>

<details>
<summary><b>11. Self-Improving VLM Judges Without Human Annotations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05145) • [📄 arXiv](https://arxiv.org/abs/2512.05145) • [📥 PDF](https://arxiv.org/pdf/2512.05145)

> Effective judges of Vision-Language Models (VLMs) are crucial for model development. Current methods for training VLM judges mainly rely on large-scale human preference annotations. However, such an approach is costly, and the annotations easily b...

</details>

<details>
<summary><b>12. World Models That Know When They Don't Know: Controllable Video Generation with Calibrated Uncertainty</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Anirudha Majumdar, Ola Shorinwa, Micah Baker, Tenny Yin, Zhiting Mei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05927) • [📄 arXiv](https://arxiv.org/abs/2512.05927) • [📥 PDF](https://arxiv.org/pdf/2512.05927)

> Recent advances in generative video models have led to significant breakthroughs in high-fidelity video synthesis, specifically in controllable video generation where the generated video is conditioned on text and action inputs, e.g., in instructi...

</details>

<details>
<summary><b>13. SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Marc Pollefeys, Or Litany, Ian Huang, Francis Engelmann, efedele

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05343) • [📄 arXiv](https://arxiv.org/abs/2512.05343) • [📥 PDF](https://arxiv.org/pdf/2512.05343)

> Generative methods for 3D assets have recently achieved remarkable progress, yet providing intuitive and precise control over the object geometry remains a key challenge. Existing approaches predominantly rely on text or image prompts, which often...

</details>

<details>
<summary><b>14. ReVSeg: Incentivizing the Reasoning Chain for Video Segmentation with Reinforcement Learning</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Shengju Qian, Weikai Chen, Lingting Zhu, Yingda Yin, Tangerine24

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.02835) • [📄 arXiv](https://arxiv.org/abs/2512.02835) • [📥 PDF](https://arxiv.org/pdf/2512.02835)

**💻 Code:** [⭐ Code](https://github.com/Clementine24/ReVSeg)

> Reasoning-centric video object segmentation is an inherently complex task: the query often refers to dynamics, causality, and temporal interactions, rather than static appearances. Yet existing solutions generally collapse these factors into simpl...

</details>

<details>
<summary><b>15. AI & Human Co-Improvement for Safer Co-Superintelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05356) • [📄 arXiv](https://arxiv.org/abs/2512.05356) • [📥 PDF](https://arxiv.org/pdf/2512.05356)

> Self-improvement is a goal currently exciting the field of AI, but is fraught with danger, and may take time to fully achieve. We advocate that a more achievable and better goal for humanity is to maximize co-improvement: collaboration between hum...

</details>

<details>
<summary><b>16. M3DR: Towards Universal Multilingual Multimodal Document Retrieval</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03514) • [📄 arXiv](https://arxiv.org/abs/2512.03514) • [📥 PDF](https://arxiv.org/pdf/2512.03514)

**💻 Code:** [⭐ Code](https://github.com/adithya-s-k/colpali)

> Can we build universal document retrievers that maintain strong results across typologically diverse languages without losing English performance. This question led us to design synthetic training data and multilingual benchmarks to teach a model ...

</details>

<details>
<summary><b>17. From Segments to Scenes: Temporal Understanding in Autonomous Driving via Vision-Language Model</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05277) • [📄 arXiv](https://arxiv.org/abs/2512.05277) • [📥 PDF](https://arxiv.org/pdf/2512.05277)

**💻 Code:** [⭐ Code](https://github.com/vbdi/tad_bench)

> This work introduces TAD, the first benchmark targeting temporal understanding in ego-centric autonomous-driving videos, evaluates SoTA VLMs, and boosts their performance with two training-free motion-reasoning methods (Scene-CoT and TCogMap).

</details>

<details>
<summary><b>18. ProPhy: Progressive Physical Alignment for Dynamic World Simulation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuhao Cheng, Terry Jingchen Zhang, Jing Wang, Panwen Hu, Zijun Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05564) • [📄 arXiv](https://arxiv.org/abs/2512.05564) • [📥 PDF](https://arxiv.org/pdf/2512.05564)

> Recent advances in video generation have shown remarkable potential for constructing world simulators. However, current models still struggle to produce physically consistent results, particularly when handling large-scale or complex dynamics. Thi...

</details>

<details>
<summary><b>19. SQ-format: A Unified Sparse-Quantized Hardware-friendly Data Format for LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Minghui Yu, Jinyuan Shi, Hantao Huang, Hao Zeng, Ruixuan Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05409) • [📄 arXiv](https://arxiv.org/abs/2512.05409) • [📥 PDF](https://arxiv.org/pdf/2512.05409)

> Post-training quantization (PTQ) plays a crucial role in the democratization of large language models (LLMs). However, existing low-bit quantization and sparsification techniques are difficult to balance accuracy and efficiency due to the limited ...

</details>

<details>
<summary><b>20. TimesNet-Gen: Deep Learning-based Site Specific Strong Motion Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Salih Tileylioglu, Erdem Akagündüz, Bevan Deniz Cilgin, Barisylmz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04694) • [📄 arXiv](https://arxiv.org/abs/2512.04694) • [📥 PDF](https://arxiv.org/pdf/2512.04694)

**💻 Code:** [⭐ Code](https://github.com/brsylmz23/TimesNet-Gen/tree/main)

> This work presents a transformer-based generative model for complex time-series signals, with experiments on seismic accelerometer data. Key idea: treat seismic waveforms as structured high-dimensional sequences and learn a latent trajectory that ...

</details>

<details>
<summary><b>21. Colon-X: Advancing Intelligent Colonoscopy from Multimodal Understanding to Clinical Reasoning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03667) • [📄 arXiv](https://arxiv.org/abs/2512.03667) • [📥 PDF](https://arxiv.org/pdf/2512.03667)

**💻 Code:** [⭐ Code](https://github.com/ai4colonoscopy/Colon-X)

> Colonoscopy saves lives — but AI for colonoscopy is still far from intelligent. We are excited to launch the Colon-X project, an open initiative aimed at advancing multimodal intelligence in colonoscopy and beyond. Beyond serving as a community-wi...

</details>

<details>
<summary><b>22. Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Caiming Xiong, Junnan Li, Shijie Wang, Honglu Zhou, Ziyang Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05774) • [📄 arXiv](https://arxiv.org/abs/2512.05774) • [📥 PDF](https://arxiv.org/pdf/2512.05774)

> No abstract available.

</details>

<details>
<summary><b>23. From FLOPs to Footprints: The Resource Cost of Artificial Intelligence</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aimee van Wynsberghe, Lisa Biber-Freudenberger, Sasha Luccioni, nicholasKluge, sophia-falk

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04142) • [📄 arXiv](https://arxiv.org/abs/2512.04142) • [📥 PDF](https://arxiv.org/pdf/2512.04142)

> The study quantifies the material footprint of AI training, focusing on the Nvidia A100 GPU, and examines the environmental impact of training models like GPT-4 🌱🍃

</details>

<details>
<summary><b>24. Taxonomy-Adaptive Moderation Model with Robust Guardrails for Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05339) • [📄 arXiv](https://arxiv.org/abs/2512.05339) • [📥 PDF](https://arxiv.org/pdf/2512.05339)

> Large Language Models (LLMs) are typically aligned for safety during the post-training phase; however, they may still generate inappropriate outputs that could potentially pose risks to users. This challenge underscores the need for robust safegua...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 24 |
| 📅 Today | [`2025-12-09.json`](data/daily/2025-12-09.json) | 24 |
| 📆 This Week | [`2025-W49.json`](data/weekly/2025-W49.json) | 62 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 249 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-09 | 24 | [View JSON](data/daily/2025-12-09.json) |
| 📄 2025-12-08 | 38 | [View JSON](data/daily/2025-12-08.json) |
| 📄 2025-12-07 | 38 | [View JSON](data/daily/2025-12-07.json) |
| 📄 2025-12-06 | 38 | [View JSON](data/daily/2025-12-06.json) |
| 📄 2025-12-05 | 38 | [View JSON](data/daily/2025-12-05.json) |
| 📄 2025-12-04 | 24 | [View JSON](data/daily/2025-12-04.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W49 | 62 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 249 | [View JSON](data/monthly/2025-12.json) |

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
