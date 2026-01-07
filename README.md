<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-24-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-823+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">44</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">85</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">823+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 07, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20578) • [📄 arXiv](https://arxiv.org/abs/2512.20578) • [📥 PDF](https://arxiv.org/pdf/2512.20578)

**💻 Code:** [⭐ Code](https://github.com/Amirhosein-gh98/Gnosis)

> Can Large Language Models predict their own failures? 🧠⚡ We all know the critical bottleneck in GenAI: LLMs are incredible, but they can confidently hallucinate and make mistakes. Until now, most fixes have been computationally massive — relying o...

</details>

<details>
<summary><b>2. NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation</b> ⭐ 60</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02204) • [📄 arXiv](https://arxiv.org/abs/2601.02204) • [📥 PDF](https://arxiv.org/pdf/2601.02204)

**💻 Code:** [⭐ Code](https://github.com/ByteVisionLab/NextFlow)

> No abstract available.

</details>

<details>
<summary><b>3. K-EXAONE Technical Report</b> ⭐ 39</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01739) • [📄 arXiv](https://arxiv.org/abs/2601.01739) • [📥 PDF](https://arxiv.org/pdf/2601.01739)

**💻 Code:** [⭐ Code](https://github.com/LG-AI-EXAONE/K-EXAONE)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Nanbeige4-3B Technical Report: Exploring the Frontier of Small Language Mod...

</details>

<details>
<summary><b>4. DreamID-V:Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer</b> ⭐ 86</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01425) • [📄 arXiv](https://arxiv.org/abs/2601.01425) • [📥 PDF](https://arxiv.org/pdf/2601.01425)

**💻 Code:** [⭐ Code](https://github.com/bytedance/DreamID-V)

> We introduce DreamID-V, the first Diffusion Transformer-based framework for high-fidelity video face swapping. DreamID-V bridges the gap between image and video domains, achieving exceptional identity similarity and temporal coherence even in chal...

</details>

<details>
<summary><b>5. VAR RL Done Right: Tackling Asynchronous Policy Conflicts in Visual Autoregressive Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02256) • [📄 arXiv](https://arxiv.org/abs/2601.02256) • [📥 PDF](https://arxiv.org/pdf/2601.02256)

> No abstract available.

</details>

<details>
<summary><b>6. GARDO: Reinforcing Diffusion Models without Reward Hacking</b> ⭐ 18</summary>

<br/>

**👥 Authors:** Zhiyong Wang, Jiajun Liang, Jie Liu, Yuxiao Ye, Haoran He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24138) • [📄 arXiv](https://arxiv.org/abs/2512.24138) • [📥 PDF](https://arxiv.org/pdf/2512.24138)

**💻 Code:** [⭐ Code](https://github.com/tinnerhrhe/GARDO) • [⭐ Code](https://github.com/tinnerhrhe/gardo)

> Introducing GARDO: Reinforcing Diffusion Models without Reward Hacking paper: https://arxiv.org/abs/2512.24138 code: https://github.com/tinnerhrhe/gardo project: https://tinnerhrhe.github.io/gardo_project/

</details>

<details>
<summary><b>7. VINO: A Unified Visual Generator with Interleaved OmniModal Context</b> ⭐ 42</summary>

<br/>

**👥 Authors:** Kun Gai, Pengfei Wan, Zhoujie Fu, Tong He, Junyi Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02358) • [📄 arXiv](https://arxiv.org/abs/2601.02358) • [📥 PDF](https://arxiv.org/pdf/2601.02358)

**💻 Code:** [⭐ Code](https://github.com/SOTAMak1r/VINO-code)

> No abstract available.

</details>

<details>
<summary><b>8. InfiniteVGGT: Visual Geometry Grounded Transformer for Endless Streams</b> ⭐ 76</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02281) • [📄 arXiv](https://arxiv.org/abs/2601.02281) • [📥 PDF](https://arxiv.org/pdf/2601.02281)

**💻 Code:** [⭐ Code](https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT)

> The grand vision of enabling persistent, large-scale 3D visual geometry understanding is shackled by the irreconcilable demands of scalability and long-term stability. While offline models like VGGT achieve inspiring geometry capability, their bat...

</details>

<details>
<summary><b>9. Recursive Language Models</b> ⭐ 675</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24601) • [📄 arXiv](https://arxiv.org/abs/2512.24601) • [📥 PDF](https://arxiv.org/pdf/2512.24601)

**💻 Code:** [⭐ Code](https://github.com/alexzhang13/rlm/tree/main)

> Study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. They propose Recursive Language Models (RLMs), a general inference strategy that treats long prompts as part of an external...

</details>

<details>
<summary><b>10. Falcon-H1R: Pushing the Reasoning Frontiers with a Hybrid Model for Efficient Test-Time Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02346) • [📄 arXiv](https://arxiv.org/abs/2601.02346) • [📥 PDF](https://arxiv.org/pdf/2601.02346)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Motif-2-12.7B-Reasoning: A Practitioner's Guide to RL Training Recipes (202...

</details>

<details>
<summary><b>11. Talk2Move: Reinforcement Learning for Text-Instructed Object-Level Geometric Transformation in Scenes</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Shuo Yang, Jiarui Cai, Yantao Shen, ZyZcuhk, jingtan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02356) • [📄 arXiv](https://arxiv.org/abs/2601.02356) • [📥 PDF](https://arxiv.org/pdf/2601.02356)

**💻 Code:** [⭐ Code](https://github.com/sparkstj/Talk2Move)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VIVA: VLM-Guided Instruction-Based Video Editing with Reward Optimization (...

</details>

<details>
<summary><b>12. Confidence Estimation for LLMs in Multi-turn Interactions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02179) • [📄 arXiv](https://arxiv.org/abs/2601.02179) • [📥 PDF](https://arxiv.org/pdf/2601.02179)

> In this paper, we explore the confidence estimation in a new paradigm: multi-turn interactions! Check it out!

</details>

<details>
<summary><b>13. KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yi Yang, Yixuan Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01046) • [📄 arXiv](https://arxiv.org/abs/2601.01046) • [📥 PDF](https://arxiv.org/pdf/2601.01046)

> ✨ Turn any decoder-only LLM into a powerful embedding model—zero training needed! ✨ The Trick : Re-route the final token's key-value states as an internal prefix, giving all tokens access to global context in one forward pass. No input modificatio...

</details>

<details>
<summary><b>14. CPPO: Contrastive Perception for Vision Language Policy Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mohammad Asiful Hossain, Kevin Cannons, Saeed Ranjbar Alvar, Mohsen Gholami, Ahmad Rezaei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00501) • [📄 arXiv](https://arxiv.org/abs/2601.00501) • [📥 PDF](https://arxiv.org/pdf/2601.00501)

> CPPO: Contrastive Perception for Vision Language Policy Optimization introduces a new method (CPPO) for fine-tuning vision-language models (VLMs) using reinforcement learning. Instead of relying on explicit perception rewards or auxiliary models, ...

</details>

<details>
<summary><b>15. DiffProxy: Multi-View Human Mesh Recovery via Diffusion-Generated Dense Proxies</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jian Yang, Ying Tai, Zhenyu Zhang, wrk226

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02267) • [📄 arXiv](https://arxiv.org/abs/2601.02267) • [📥 PDF](https://arxiv.org/pdf/2601.02267)

**💻 Code:** [⭐ Code](https://github.com/wrk226/DiffProxy)

> Project page: https://wrk226.github.io/DiffProxy.html Code: https://github.com/wrk226/DiffProxy

</details>

<details>
<summary><b>16. COMPASS: A Framework for Evaluating Organization-Specific Policy Alignment in LLMs</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01836) • [📄 arXiv](https://arxiv.org/abs/2601.01836) • [📥 PDF](https://arxiv.org/pdf/2601.01836)

**💻 Code:** [⭐ Code](https://github.com/AIM-Intelligence/COMPASS)

> COMPASS is the first framework for evaluating LLM alignment with organization-specific policies rather than universal harms. While models handle legitimate requests well (>95% accuracy), they catastrophically fail at enforcing prohibitions, refusi...

</details>

<details>
<summary><b>17. Toward Stable Semi-Supervised Remote Sensing Segmentation via Co-Guidance and Co-Fusion</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Shiying Wang, Kai Li, Shun Zhang, Xuechao Zou, Yi Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23035) • [📄 arXiv](https://arxiv.org/abs/2512.23035) • [📥 PDF](https://arxiv.org/pdf/2512.23035)

**💻 Code:** [⭐ Code](https://github.com/XavierJiezou/Co2S)

> We are excited to introduce our latest work on semi-supervised semantic segmentation : 📄 Toward Stable Semi-Supervised Remote Sensing Segmentation via Co-Guidance and Co-Fusion This paper tackles one of the most challenging issues in semi-supervis...

</details>

<details>
<summary><b>18. SWE-Lego: Pushing the Limits of Supervised Fine-tuning for Software Issue Resolving</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01426) • [📄 arXiv](https://arxiv.org/abs/2601.01426) • [📥 PDF](https://arxiv.org/pdf/2601.01426)

**💻 Code:** [⭐ Code](https://github.com/SWE-Lego/SWE-Lego)

> No abstract available.

</details>

<details>
<summary><b>19. OpenNovelty: An LLM-powered Agentic System for Verifiable Scholarly Novelty Assessment</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Chunchun Ma, Yujiong Shen, Yueyuan Huang, Kexin Tan, Ming Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01576) • [📄 arXiv](https://arxiv.org/abs/2601.01576) • [📥 PDF](https://arxiv.org/pdf/2601.01576)

**💻 Code:** [⭐ Code](https://github.com/january-blue/OpenNovelty)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API ARISE: Agentic Rubric-Guided Iterative Survey Engine for Automated Scholarl...

</details>

<details>
<summary><b>20. Selective Imperfection as a Generative Framework for Analysis, Creativity and Discovery</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00863) • [📄 arXiv](https://arxiv.org/abs/2601.00863) • [📥 PDF](https://arxiv.org/pdf/2601.00863)

**💻 Code:** [⭐ Code](https://github.com/lamm-mit/MusicAnalysis)

> Selective Imperfection as a Generative Framework for Analysis, Creativity and Discovery We introduce materiomusic as a generative framework linking the hierarchical structures of matter with the compositional logic of music. Across proteins, spide...

</details>

<details>
<summary><b>21. IMA++: ISIC Archive Multi-Annotator Dermoscopic Skin Lesion Segmentation Dataset</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21472) • [📄 arXiv](https://arxiv.org/abs/2512.21472) • [📥 PDF](https://arxiv.org/pdf/2512.21472)

**💻 Code:** [⭐ Code](https://github.com/sfu-mial/IMAplusplus)

> ✨ The largest publicly available dermoscopic skin lesion segmentation dataset with 17,684 segmentation masks spanning 14,967 dermoscopic images, where 2,394 dermoscopic images have 2-5 segmentations per image. ✨ 16 unique annotators , 3 different ...

</details>

<details>
<summary><b>22. Prithvi-Complimentary Adaptive Fusion Encoder (CAFE): unlocking full-potential for flood inundation mapping</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Beth Tellman, Lalit Maurya, Saurabh Kaushik

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02315) • [📄 arXiv](https://arxiv.org/abs/2601.02315) • [📥 PDF](https://arxiv.org/pdf/2601.02315)

**💻 Code:** [⭐ Code](https://github.com/Sk-2103/Prithvi-CAFE)

> Despite the recent success of large pretrained encoders (Geo‑Foundation Models), we consistently observe that U‑Net‑based models remain highly competitive—and in some cases outperform transformers, particularly due to their strength in capturing l...

</details>

<details>
<summary><b>23. Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02314) • [📄 arXiv](https://arxiv.org/abs/2601.02314) • [📥 PDF](https://arxiv.org/pdf/2601.02314)

**💻 Code:** [⭐ Code](https://github.com/skhanzad/AridadneXAI)

> Does COT in llms stay faithful to their thoughts?

</details>

<details>
<summary><b>24. M-ErasureBench: A Comprehensive Multimodal Evaluation Benchmark for Concept Erasure in Diffusion Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jun-Cheng Chen, Cheng-Fu Chou, Ju-Hsuan Weng, jwliao1209

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22877) • [📄 arXiv](https://arxiv.org/abs/2512.22877) • [📥 PDF](https://arxiv.org/pdf/2512.22877)

> Concept Erasure Benchmark

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 24 |
| 📅 Today | [`2026-01-07.json`](data/daily/2026-01-07.json) | 24 |
| 📆 This Week | [`2026-W01.json`](data/weekly/2026-W01.json) | 44 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 85 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-07 | 24 | [View JSON](data/daily/2026-01-07.json) |
| 📄 2026-01-06 | 13 | [View JSON](data/daily/2026-01-06.json) |
| 📄 2026-01-05 | 7 | [View JSON](data/daily/2026-01-05.json) |
| 📄 2026-01-04 | 7 | [View JSON](data/daily/2026-01-04.json) |
| 📄 2026-01-03 | 7 | [View JSON](data/daily/2026-01-03.json) |
| 📄 2026-01-02 | 20 | [View JSON](data/daily/2026-01-02.json) |
| 📄 2026-01-01 | 7 | [View JSON](data/daily/2026-01-01.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W01 | 44 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 85 | [View JSON](data/monthly/2026-01.json) |
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
