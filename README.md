<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5891+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">161</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">510</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5891+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 21, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. SemComp-Bench: Benchmarking Semantic Task Completion in Video Generation</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17426) • [📄 arXiv](https://arxiv.org/abs/2608.17426) • [📥 PDF](https://arxiv.org/pdf/2608.17426)

**💻 Code:** [⭐ Code](https://github.com/Kelly372/SemComp-Bench) • [⭐ Code](https://github.com/huggingface)

> Can a video generator actually finish the task—not merely make a convincing video? SemComp-Bench evaluates outcome achievement together with task-relevant semantic grounding.

</details>

<details>
<summary><b>2. Zetta ζ: An Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence</b> ⭐ 186</summary>

<br/>

**👥 Authors:** Zixuan Wang, Mingzhe Huang, Liang Mi, XXXXyu, Xin64

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16590) • [📄 arXiv](https://arxiv.org/abs/2608.16590) • [📥 PDF](https://arxiv.org/pdf/2608.16590)

**💻 Code:** [⭐ Code](https://github.com/air-embodied-brain/Zetta-Embodiment) • [⭐ Code](https://github.com/huggingface)

> We present Zetta, a closed-loop embodied harness that evolves code-based runtime critics and recovery skills online while keeping the base policy frozen. Through three timescale-separated loops, Zetta provides action-frequency governance, rollout-...

</details>

<details>
<summary><b>3. SemaPLC: A Project-Grounded, Verification-Gated Agent Harness for PLC Code Generation</b> ⭐ 37</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18565) • [📄 arXiv](https://arxiv.org/abs/2608.18565) • [📥 PDF](https://arxiv.org/pdf/2608.18565)

**💻 Code:** [⭐ Code](https://github.com/midea-ai/SemaPLC) • [⭐ Code](https://github.com/huggingface)

> Project link： https://github.com/midea-ai/SemaPLC Document： https://midea-ai.github.io/SemaPLC/#/en/ SemaPLC turns AI-assisted PLC programming from "generating code" into "delivering verified control logic." It provides a browser-based IDE for gen...

</details>

<details>
<summary><b>4. Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17253) • [📄 arXiv](https://arxiv.org/abs/2608.17253) • [📥 PDF](https://arxiv.org/pdf/2608.17253)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DrStranded/Co-RL)

> Reinforcement learning (RL) has emerged as a powerful approach for improving reasoning in language and vision-language models, yet its strongest successes still depend heavily on ground-truth supervision (e.g., verifiable reward). Such annotations...

</details>

<details>
<summary><b>5. OmniScientist: An Omni-Modal Omni-Discipline AI Scientist</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13558) • [📄 arXiv](https://arxiv.org/abs/2608.13558) • [📥 PDF](https://arxiv.org/pdf/2608.13558)

**💻 Code:** [⭐ Code](https://github.com/Omni-Scientist/OmniScientist) • [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API PaperClaw: Harnessing Agents for Autonomous Research and Human-in-the-Loop ...

</details>

<details>
<summary><b>6. SPADE: Self-Play in Adaptive Synthetic Executable Environments</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19197) • [📄 arXiv](https://arxiv.org/abs/2608.19197) • [📥 PDF](https://arxiv.org/pdf/2608.19197)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/spade-rl/spade)

> We let the model do environment scaling on its own: one LLM writes its own executable, agentic training environments and learns to solve them, with hint-based regret keeping each one at the agent's frontier. +5.3 over the strongest fixed-environme...

</details>

<details>
<summary><b>7. Training Chemical Plausibility-Aware Large Language Models for Single-Step Retrosynthesis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18940) • [📄 arXiv](https://arxiv.org/abs/2602.03554) • [📥 PDF](https://arxiv.org/pdf/2608.18940)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This work extends the previously introduced ChemCensor framework for single-step retrosynthesis ( https://arxiv.org/abs/2602.03554 ). It addresses the one-to-many nature of retrosynthetic prediction, where a target molecule may have several chemic...

</details>

<details>
<summary><b>8. Training Leaves Traces: Centered Residual Signatures for Language Model Lineage Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14929) • [📄 arXiv](https://arxiv.org/abs/2608.14929) • [📥 PDF](https://arxiv.org/pdf/2608.14929)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Open-weight language models are fine-tuned, quantized, pruned, and merged, yet their provenance is often undocumented. We study data-free white-box lineage verification: can weights alone reveal whether two compatible model checkpoints share ances...

</details>

<details>
<summary><b>9. Decision-Metric Alignment in Latent World Models: Diagnostics and Action-Conditioned Objectives for MPC Planning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18746) • [📄 arXiv](https://arxiv.org/abs/2608.18746) • [📥 PDF](https://arxiv.org/pdf/2608.18746)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: Strong latent representations are not necessarily good planning metrics. This paper introduces diagnostics for measuring whether latent distances reflect real task progress and shows that action-conditioned objectives substantially improve ...

</details>

<details>
<summary><b>10. Looped Language Models Improve Compositional Tool Calling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pietro Liò, Haitz Sáez de Ocáriz Borde, Andrei Cristian Popescu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18171) • [📄 arXiv](https://arxiv.org/abs/2608.18171) • [📥 PDF](https://arxiv.org/pdf/2608.18171)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>11. FM-Bench: A Benchmark for Long-Horizon Management with Competing Agents</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Yinghao He, Chen Dong, Kezhen Chen, Chongyang Gao, Tianyou Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18423) • [📄 arXiv](https://arxiv.org/abs/2608.18423) • [📥 PDF](https://arxiv.org/pdf/2608.18423)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Analogy-AI/fm-bench)

> No abstract available.

</details>

<details>
<summary><b>12. The More Popular, The Harder to Forget: Adaptive Popularity for LLM Unlearning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alexander Panchenko, Andrey Savchenko, tlenusik, SwetieePawsss

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14229) • [📄 arXiv](https://arxiv.org/abs/2608.14229) • [📥 PDF](https://arxiv.org/pdf/2608.14229)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Popular facts are memorised more deeply during pretraining and resist removal longer than rare ones, yet existing LLM unlearning methods apply uniform gradient pressure regardless of training-data frequency. We propose the AdaPop (Adaptive Popular...

</details>

<details>
<summary><b>13. Scaling Creative Writing Beyond Story-Centric Data with Attribute-Guided Genre Expansion</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jinsik Lee, Yireun Kim, Heuiyeen Yeen, Yongil Kim, HwanChang0106

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13947) • [📄 arXiv](https://arxiv.org/abs/2608.13947) • [📥 PDF](https://arxiv.org/pdf/2608.13947)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> High-quality creative writing data for language models remains dominated by story-centric data, limiting models’ ability to follow the structural and functional conventions of diverse creative formats. We propose an attribute-guided genre expansio...

</details>

<details>
<summary><b>14. SoftVTBench: A Deformation-Aware Visuo-Tactile Dataset and Benchmark for Deformable-Object Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18701) • [📄 arXiv](https://arxiv.org/abs/2608.18701) • [📥 PDF](https://arxiv.org/pdf/2608.18701)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> SoftVTBench: A Deformation-Aware Visuo-Tactile Dataset and Benchmark for Deformable-Object Manipulation

</details>

<details>
<summary><b>15. LLMs Get Smarter from Targeted Synthetic Multilingual Data</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Andreas Stolcke, Neha Gupta, Tanner Sorensen, Arkajyoti Charaborty, Ishika Agarwal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15964) • [📄 arXiv](https://arxiv.org/abs/2608.15964) • [📥 PDF](https://arxiv.org/pdf/2608.15964)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Language-specific competency (LSC) is the phenomenon of a language model performing better or worse depending on the language of the prompt. In other words, a language model outputs different (and potentially incorrect) responses to the same seman...

</details>

<details>
<summary><b>16. SkillGate: Training In-Policy Skill Selection in Long-Horizon Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18852) • [📄 arXiv](https://arxiv.org/abs/2608.18852) • [📥 PDF](https://arxiv.org/pdf/2608.18852)

**💻 Code:** [⭐ Code](https://github.com/DeepExperience/SkillGate) • [⭐ Code](https://github.com/huggingface)

> We introduce SkillGate, a training method for in-policy skill selection in long-horizon agents. It addresses selector credit starvation by separating outcome credit for execution tokens from an action-local advantage applied only to skill-naming t...

</details>

<details>
<summary><b>17. Evaluating Music Context Preservation: A Multi-facet Framework for Music Editing Systems</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14629) • [📄 arXiv](https://arxiv.org/abs/2512.14629) • [📥 PDF](https://arxiv.org/pdf/2512.14629)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Yashvishe13/MuseCPEval)

> If you want to evaluate how well the unedited parts of a music clip are preserved after music editing, feel free to try MuseCPEval !!! MuseCPEval introduces 12 metrics across 5 music facets: harmony, rhythm&meter, structure, melody&motif, and timb...

</details>

<details>
<summary><b>18. Towards Real-Time and Adaptable LiDAR Scene Completion</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16490) • [📄 arXiv](https://arxiv.org/abs/2608.16490) • [📥 PDF](https://arxiv.org/pdf/2608.16490)

**💻 Code:** [⭐ Code](https://github.com/AzharSindhi/RapidLiDAR) • [⭐ Code](https://github.com/huggingface)

> If you're tired of slow 3D scene completion pipelines, we built RapidLiDAR to hit 10 Hz in real-time. Drop your questions below!

</details>

<details>
<summary><b>19. Bounded Agents: Delegation Security for Multi-Agent AI Systems</b> ⭐ 2</summary>

<br/>

**👥 Authors:** xmuruaga

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15888) • [📄 arXiv](https://arxiv.org/abs/2608.15888) • [📥 PDF](https://arxiv.org/pdf/2608.15888)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/xmuruaga/bounded-agents)

> Bounded Agents introduces the Agentic Principal Chain (APC), an authorization model for multi-agent AI systems. APC carries authorization state across actions, agents and tools, narrows scope and budgets through delegation, and uses prior executio...

</details>

<details>
<summary><b>20. Temporal Multi-Signal Fusion for Token-Level Hallucination Detection</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Igor Itkin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18115) • [📄 arXiv](https://arxiv.org/abs/2608.18115) • [📥 PDF](https://arxiv.org/pdf/2608.18115)

**💻 Code:** [⭐ Code](https://github.com/YehudaItkin/temporal-hallucination-detection) • [⭐ Code](https://github.com/huggingface)

> code

</details>

<details>
<summary><b>21. VA-Judger: Reward Modeling from Human Preference Feedback for Joint Video-Audio Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18607) • [📄 arXiv](https://arxiv.org/abs/2608.18607) • [📥 PDF](https://arxiv.org/pdf/2608.18607)

**💻 Code:** [⭐ Code](https://github.com/ShareLab-SII/VA-Judger) • [⭐ Code](https://github.com/huggingface)

> 🚀 We are excited to introduce VA-Judger: Reward Modeling from Human Preference Feedback for Joint Video-Audio Generation , the first reward model specifically designed for joint video-audio generation . Existing metrics typically evaluate video an...

</details>

<details>
<summary><b>22. SPK: Eliciting Structured Prior Knowledge for Interpretable Out-of-Distribution Detection in Real-Time Object Detection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19080) • [📄 arXiv](https://arxiv.org/abs/2608.19080) • [📥 PDF](https://arxiv.org/pdf/2608.19080)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Modern object detectors (e.g., YOLO, RT-DETR, and Faster R-CNN) can produce overconfident predictions for objects outside their training classes, which we refer to as out-of-distribution (OoD) objects, while the training classes are referred to as...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-08-21.json`](data/daily/2026-08-21.json) | 22 |
| 📆 This Week | [`2026-W33.json`](data/weekly/2026-W33.json) | 161 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 510 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-21 | 22 | [View JSON](data/daily/2026-08-21.json) |
| 📄 2026-08-20 | 33 | [View JSON](data/daily/2026-08-20.json) |
| 📄 2026-08-19 | 42 | [View JSON](data/daily/2026-08-19.json) |
| 📄 2026-08-18 | 32 | [View JSON](data/daily/2026-08-18.json) |
| 📄 2026-08-17 | 32 | [View JSON](data/daily/2026-08-17.json) |
| 📄 2026-08-16 | 32 | [View JSON](data/daily/2026-08-16.json) |
| 📄 2026-08-15 | 32 | [View JSON](data/daily/2026-08-15.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W33 | 161 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |
| 📅 2026-W30 | 112 | [View JSON](data/weekly/2026-W30.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 510 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
