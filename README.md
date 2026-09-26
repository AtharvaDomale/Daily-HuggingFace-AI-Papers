<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6497+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">76</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">369</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6497+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** September 26, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Training Object Permanence in World Models</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.28654) • [📄 arXiv](https://arxiv.org/abs/2609.28654) • [📥 PDF](https://arxiv.org/pdf/2609.28654)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hokindeng/object-permanence)

> Object permanence is the foundation of human cognition. Here we present a very complete data infrastructure that's composed of a very diverse set of object permanence cognitive tasks, and with each task we have a Blender-based data generator that ...

</details>

<details>
<summary><b>2. Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nikita Dragunov, Matvey Mikhalchuk, Anton Korznikov, razzant, pashocles

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29845) • [📄 arXiv](https://arxiv.org/abs/2609.29845) • [📥 PDF](https://arxiv.org/pdf/2609.29845)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Your LLM can hold two thoughts at once: when you average the embeddings of two unrelated texts, the model predicts the next token for both streams at the same time. This superposition comes from the architecture itself, since pretraining erodes it...

</details>

<details>
<summary><b>3. WanPE: Towards Cinematic Prompt Enhancement for Modern Text-to-Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kai Zhu, Zixun Fang, Ziyun Dai, Yawen Shao, Yubo Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.30221) • [📄 arXiv](https://arxiv.org/abs/2609.30221) • [📥 PDF](https://arxiv.org/pdf/2609.30221)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API CAPE-T2V: Captioner-Anchored Prompt Enhancement toward Two-Sided Conditioni...

</details>

<details>
<summary><b>4. OmniEcho: Spatial Audio Understanding for Embodied Agents</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.23407) • [📄 arXiv](https://arxiv.org/abs/2609.23407) • [📥 PDF](https://arxiv.org/pdf/2609.23407)

**💻 Code:** [⭐ Code](https://github.com/PKU-VaLuE-Lab/OmniEcho) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>5. Agent-Editing World Model: Rethinking World Modeling for LLM Agents</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.28416) • [📄 arXiv](https://arxiv.org/abs/2609.28416) • [📥 PDF](https://arxiv.org/pdf/2609.28416)

**💻 Code:** [⭐ Code](https://github.com/RUCAIBox/Agent-Editing-World-Model) • [⭐ Code](https://github.com/huggingface)

> Agent-Editing World Model (AEWM) rethinks world modeling for long-horizon LLM agents by shifting the objective from predicting environment observations to modeling how an agent’s reasoning and actions affect future task progress. AEWM combines Act...

</details>

<details>
<summary><b>6. Parts-of-Speech as Emergent Categories in SAE Latent Space</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29362) • [📄 arXiv](https://arxiv.org/abs/2609.29362) • [📥 PDF](https://arxiv.org/pdf/2609.29362)

**💻 Code:** [⭐ Code](https://github.com/colinglab/pos-sae-latents) • [⭐ Code](https://github.com/huggingface)

> Hi all! This is our new paper on how Sparse Autoencoders encode parts of speech, which was accepted at EMNLP 2026 🎉 SAE latents are often described as monosemantic features, so we tested whether linguistic categories like POS map onto individual l...

</details>

<details>
<summary><b>7. Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29892) • [📄 arXiv](https://arxiv.org/abs/2609.29892) • [📥 PDF](https://arxiv.org/pdf/2609.29892)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Qwen-Planner-Agent

</details>

<details>
<summary><b>8. IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29444) • [📄 arXiv](https://arxiv.org/abs/2609.29444) • [📥 PDF](https://arxiv.org/pdf/2609.29444)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/IterSynth)

> IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis Deep-search agents that follow the ReAct recipe inherit two coupled bottlenecks: one policy has to plan, use evidence, and synthesize at the same time, while its conte...

</details>

<details>
<summary><b>9. RGBD20K: A Large-Scale Benchmark for RGB-D Semantic Segmentation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29028) • [📄 arXiv](https://arxiv.org/abs/2609.29028) • [📥 PDF](https://arxiv.org/pdf/2609.29028)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ShaohuaDong2021/RGBD20K)

> We introduce RGBD20K, a large-scale benchmark for RGB-D semantic segmentation with 20,000 RGB-D image pairs and 160 fine-grained semantic categories. We also provide high-quality re-annotated ground truth to address annotation noise in existing be...

</details>

<details>
<summary><b>10. Rufus-Air: An Open LLM Post-Training Recipe</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29421) • [📄 arXiv](https://arxiv.org/abs/2609.29421) • [📥 PDF](https://arxiv.org/pdf/2609.29421)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Rufus-Air is an open and reproducible post-training recipe on GLM-4.5-Air-Base (106B-A12B), organized as a serial pipeline of eight stages: SFT, Reasoning RL, Coding RL, Instruction-Following RL, General Agent, Coding Agent, Search Agent, and RLHF...

</details>

<details>
<summary><b>11. Coding Agents for Generalized Task and Motion Planning Problems</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.30233) • [📄 arXiv](https://arxiv.org/abs/2609.30233) • [📥 PDF](https://arxiv.org/pdf/2609.30233)

**💻 Code:** [⭐ Code](https://github.com/tomsilver/robocode) • [⭐ Code](https://github.com/huggingface)

> How far can you get by asking a coding agent to write a robot policy? 🤖 We've seen some amazing one-off demos of Astra in robotics, but we need more systematic evaluations of these capabilities. We've spent months doing just that: 98,000 evaluatio...

</details>

<details>
<summary><b>12. Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.23087) • [📄 arXiv](https://arxiv.org/abs/2609.23087) • [📥 PDF](https://arxiv.org/pdf/2609.23087)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Optima-CityU/neural-spectral-capacity)

> TL;DR: NSC scores a Transformer architecture from its specification alone — no model instantiation, no data, no gradients — and because the score is additive across layers, maximizing it under a budget is an exact dynamic program rather than a bla...

</details>

<details>
<summary><b>13. AgentKernel: The Trust-Native Agentic Operating System</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shuo Li, Lepeng Zhao, Qiuyang Zhan, Sheng Guo, Zhenhua Zou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29647) • [📄 arXiv](https://arxiv.org/abs/2609.29647) • [📥 PDF](https://arxiv.org/pdf/2609.29647)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API InterSAGE: The Secure and Verifiable Interoperability Protocol for An Inter...

</details>

<details>
<summary><b>14. Learning to Discover Interesting Mathematics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.28603) • [📄 arXiv](https://arxiv.org/abs/2609.28603) • [📥 PDF](https://arxiv.org/pdf/2609.28603)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> LLMs are now proving results that have resisted mathematicians for decades. Finding interesting theorems without human guidance is a new bottleneck. We show that we can teach an LLM to do it! TL;DR: → a quantitative notion of interestingness → 4.3...

</details>

<details>
<summary><b>15. PUBG Ally: A Conversational Embodied Agent as an AI Teammate</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Eunchong Kim, Dongwon Kim, Dohyun Kim, Byeongju Kim, Beomsoo Kim

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29837) • [📄 arXiv](https://arxiv.org/abs/2609.29837) • [📥 PDF](https://arxiv.org/pdf/2609.29837)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API AI for Games in the Foundation Model Era (2026) "What Can I Do for You'': H...

</details>

<details>
<summary><b>16. ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.30199) • [📄 arXiv](https://arxiv.org/abs/2609.30199) • [📥 PDF](https://arxiv.org/pdf/2609.30199)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Can LLMs Discover Scientific Laws in Real and Parallel Worlds? (2026) Scien...

</details>

<details>
<summary><b>17. AV-GRPO: Modality-Anchored Decoupling Diffusion Reinforcement Learning for Joint Audio-Video Generation</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29816) • [📄 arXiv](https://arxiv.org/abs/2609.29816) • [📥 PDF](https://arxiv.org/pdf/2609.29816)

**💻 Code:** [⭐ Code](https://github.com/zhiyuxu03/AV-GRPO) • [⭐ Code](https://github.com/huggingface)

> AV-GRPO, to our knowledge, is the first GRPO framework designed for joint audio-video generation models. It enables full-parameter training or LoRA training of the 22B LTX-2.3 model on just 8 A800 GPUs.

</details>

<details>
<summary><b>18. World Action Agent: Harnessing VLMs for Robot Manipulation via World Action Rehearsal</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bohan Zhou, Jianchong Su, Yifan Chang, Haojian Huang, Yehang Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29964) • [📄 arXiv](https://arxiv.org/abs/2609.29964) • [📥 PDF](https://arxiv.org/pdf/2609.29964)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Show-Harness: Just a VLM Agent Can Play Robots (2026) In-Context Robot Lear...

</details>

<details>
<summary><b>19. Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Lida Zhao, Yuekang Li, Gelei Deng, Yi Liu, Ruoqi Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29429) • [📄 arXiv](https://arxiv.org/abs/2609.29429) • [📥 PDF](https://arxiv.org/pdf/2609.29429)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sumleo/RLCDAlignBench)

> The first empirical study of RLCD models like Jev on detecting AI alignment failures.

</details>

<details>
<summary><b>20. DeltaWAM: Delta World Action Models for Bimanual Manipulation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qilin Wang, Zeyu Zhang, Haokai Jiang, Zishang Xiang, coldyan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.28811) • [📄 arXiv](https://arxiv.org/abs/2609.28811) • [📥 PDF](https://arxiv.org/pdf/2609.28811)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIGeeksGroup/DeltaWAM)

> Work in progress, authors are preparing the code 😉

</details>

<details>
<summary><b>21. ViRDM: Taming Representation Distribution Matching for Few-Step Causal Video Generation</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Huaizu Jiang, Yang Zhou, Chun-Hao P. Huang, Chongjian Ge, cr8br0ze

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.28923) • [📄 arXiv](https://arxiv.org/abs/2609.28923) • [📥 PDF](https://arxiv.org/pdf/2609.28923)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/neu-vi/ViRDM)

> Hi everyone! 👋 We’re sharing ViRDM : A recipe for RDM-based few-step causal video post-training, along with the pitfalls and lessons learned (single-GPU training supported 💥) Inspired by the impressive results of FD Loss and RDM in image generatio...

</details>

<details>
<summary><b>22. Rate-distortion optimization for full-reference image quality metrics via stochastic Hessian estimates</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.30077) • [📄 arXiv](https://arxiv.org/abs/2609.30077) • [📥 PDF](https://arxiv.org/pdf/2609.30077)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sf219/RDO_IQA_FR)

> RDO with full-reference image quality assessment metrics.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-09-26.json`](data/daily/2026-09-26.json) | 22 |
| 📆 This Week | [`2026-W38.json`](data/weekly/2026-W38.json) | 76 |
| 🗓️ This Month | [`2026-09.json`](data/monthly/2026-09.json) | 369 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-09-26 | 22 | [View JSON](data/daily/2026-09-26.json) |
| 📄 2026-09-25 | 13 | [View JSON](data/daily/2026-09-25.json) |
| 📄 2026-09-24 | 10 | [View JSON](data/daily/2026-09-24.json) |
| 📄 2026-09-23 | 10 | [View JSON](data/daily/2026-09-23.json) |
| 📄 2026-09-22 | 10 | [View JSON](data/daily/2026-09-22.json) |
| 📄 2026-09-21 | 11 | [View JSON](data/daily/2026-09-21.json) |
| 📄 2026-09-20 | 24 | [View JSON](data/daily/2026-09-20.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W38 | 76 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |
| 📅 2026-W36 | 88 | [View JSON](data/weekly/2026-W36.json) |
| 📅 2026-W35 | 121 | [View JSON](data/weekly/2026-W35.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-09 | 369 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |

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
