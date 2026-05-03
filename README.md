<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-24-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3680+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">84</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">59</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3680+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 03, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Heterogeneous Scientific Foundation Model Collaboration</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27351) • [📄 arXiv](https://arxiv.org/abs/2604.27351) • [📥 PDF](https://arxiv.org/pdf/2604.27351)

**💻 Code:** [⭐ Code](https://github.com/Violet24K/Eywa)

> We are bringing Eywa from fiction to digital reality: a heterogeneous agentic framework that enables language models and domain-specific foundation models to collaborate together. If you find the idea interesting, we would really appreciate your s...

</details>

<details>
<summary><b>2. Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haowei Zhu, Shizun Wang, Kaichen Zhang, Zuhao Yang, Keming Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.28185) • [📄 arXiv](https://arxiv.org/abs/2604.28185) • [📥 PDF](https://arxiv.org/pdf/2604.28185)

**💻 Code:** [⭐ Code](https://github.com/EvolvingLMMs-Lab/Evolving-Visual-Generation)

> Thanks @ taesiri for promoting our paper! We are excited to share our new roadmap, "Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling". The thesis is a pragmatic one: visual generation has entered its sec...

</details>

<details>
<summary><b>3. Co-Evolving Policy Distillation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dingyu Yao, Chuanyu Qin, Qingyi Si, Chenxu Yang, Naibin Gu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27083) • [📄 arXiv](https://arxiv.org/abs/2604.27083) • [📥 PDF](https://arxiv.org/pdf/2604.27083)

> work in progress

</details>

<details>
<summary><b>4. ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu Bai, Zhenguo Sun, Yibo Peng, Jingyu Ma, Yanghao Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27711) • [📄 arXiv](https://arxiv.org/abs/2604.27711) • [📥 PDF](https://arxiv.org/pdf/2604.27711)

> ExoActor provides a scalable approach to modeling interaction-rich humanoid behaviors, potentially opening a new avenue for generative models to advance general-purpose humanoid intelligence. Feedback is very welcome!

</details>

<details>
<summary><b>5. Efficient Training on Multiple Consumer GPUs with RoundPipe</b> ⭐ 51</summary>

<br/>

**👥 Authors:** Jiwu Shu, Youyou Lu, Huichuan Zheng, Shiwei Gao, Yibin Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27085) • [📄 arXiv](https://arxiv.org/abs/2604.27085) • [📥 PDF](https://arxiv.org/pdf/2604.27085)

**💻 Code:** [⭐ Code](https://github.com/ITcarrot/RoundPipe)

> Fine-tuning Large Language Models (LLMs) on consumer-grade GPUs is highly cost-effective, yet constrained by limited GPU memory and slow PCIe interconnects. Pipeline parallelism combined with CPU offloading mitigates these hardware bottlenecks by ...

</details>

<details>
<summary><b>6. Leveraging Verifier-Based Reinforcement Learning in Image Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27505) • [📄 arXiv](https://arxiv.org/abs/2604.27505) • [📥 PDF](https://arxiv.org/pdf/2604.27505)

> Leveraging Verifier-Based Reinforcement Learning in Image Editing

</details>

<details>
<summary><b>7. Intern-Atlas: A Methodological Evolution Graph as Research Infrastructure for AI Scientists</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yiling Duan, Jinhang Xu, Xinchen Li, Dongxu Zhang, Yujun Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.28158) • [📄 arXiv](https://arxiv.org/abs/2604.28158) • [📥 PDF](https://arxiv.org/pdf/2604.28158)

> i think this is a fascinating work, very timely in this field. thank you for your contribution to the community! the only thing i'd suggest is that the online graph seems pretty slow when clicking some large concepts, like Skip Connection.

</details>

<details>
<summary><b>8. Claw-Eval-Live: A Live Agent Benchmark for Evolving Real-World Workflows</b> ⭐ 21</summary>

<br/>

**👥 Authors:** Shijue Huang, Yunlong Lin, Huangxin Lin, Zhengyang Tang, Chenxin Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.28139) • [📄 arXiv](https://arxiv.org/abs/2604.28139) • [📥 PDF](https://arxiv.org/pdf/2604.28139)

**💻 Code:** [⭐ Code](https://github.com/Claw-Eval-Live/Claw-Eval-Live)

> Claw-Eval-Live is a live benchmark for LLM workflow agents. Each release is constructed from public workflow-demand signals (ClawHub Top-500 skills) rather than frozen at release time, and materialized as 105 executable tasks with fixed fixtures, ...

</details>

<details>
<summary><b>9. Length Value Model: Scalable Value Pretraining for Token-Level Length Modeling</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27039) • [📄 arXiv](https://arxiv.org/abs/2604.27039) • [📥 PDF](https://arxiv.org/pdf/2604.27039)

**💻 Code:** [⭐ Code](https://github.com/eric-ai-lab/Length-Value-Model)

> Try our demo to see the impact of each token on the expected length. https://length-value-model.github.io/demo/index.html

</details>

<details>
<summary><b>10. Representation Fréchet Loss for Visual Generation</b> ⭐ 133</summary>

<br/>

**👥 Authors:** Yue Wang, Yonglong Tian, Xuan Ju, Zhengyang Geng, Jiawei Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.28190) • [📄 arXiv](https://arxiv.org/abs/2604.28190) • [📥 PDF](https://arxiv.org/pdf/2604.28190)

**💻 Code:** [⭐ Code](https://github.com/Jiawei-Yang/FD-Loss)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Evaluating Generative Models via One-Dimensional Code Distributions (2026) ...

</details>

<details>
<summary><b>11. Synthetic Computers at Scale for Long-Horizon Productivity Simulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.28181) • [📄 arXiv](https://arxiv.org/abs/2604.28181) • [📥 PDF](https://arxiv.org/pdf/2604.28181)

> We create user-specific synthetic computers from personas and use them as grounding environments for long-horizon productivity simulations, producing both professional deliverables and process signals for improving agents in productivity scenarios.

</details>

<details>
<summary><b>12. Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24954) • [📄 arXiv](https://arxiv.org/abs/2604.24954) • [📥 PDF](https://arxiv.org/pdf/2604.24954)

> We introduce Nemotron 3 Nano Omni, the latest model in the Nemotron multimodal series and the first to natively support audio inputs alongside text, images, and video. Nemotron 3 Nano Omni delivers consistent accuracy improvements over its predece...

</details>

<details>
<summary><b>13. The Last Human-Written Paper: Agent-Native Research Artifacts</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24658) • [📄 arXiv](https://arxiv.org/abs/2604.24658) • [📥 PDF](https://arxiv.org/pdf/2604.24658)

**💻 Code:** [⭐ Code](https://github.com/Orchestra-Research/Agent-Native-Research-Artifact)

> Scientific publication compresses a branching, iterative research process into a linear narrative, discarding the majority of what was discovered along the way. This compilation imposes two structural costs: a Storytelling Tax, where failed experi...

</details>

<details>
<summary><b>14. InteractWeb-Bench: Can Multimodal Agent Escape Blind Execution in Interactive Website Generation?</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27419) • [📄 arXiv](https://arxiv.org/abs/2604.27419) • [📥 PDF](https://arxiv.org/pdf/2604.27419)

**💻 Code:** [⭐ Code](https://github.com/AIforIP/InteractWeb-Bench)

> InteractWeb-Bench is a multimodal interactive benchmark for evaluating website generation agents under real-world, non-expert user conditions. It simulates ambiguous, noisy, and conflicting user instructions through persona-driven user agents, and...

</details>

<details>
<summary><b>15. Step-level Optimization for Efficient Computer-use Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27151) • [📄 arXiv](https://arxiv.org/abs/2604.27151) • [📥 PDF](https://arxiv.org/pdf/2604.27151)

**💻 Code:** [⭐ Code](https://github.com/yale-nlp/StepWise)

> We introduce an event-driven, step-level cascade for efficient computer-use agents. Instead of invoking a frontier VLM at every GUI step, our framework runs a smaller policy by default and escalates only when lightweight monitors detect progress s...

</details>

<details>
<summary><b>16. PhyCo: Learning Controllable Physical Priors for Generative Motion</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Manmohan Chandraker, Srinivasa Narasimhan, Ziyu Jiang, Sriram Narayanan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.28169) • [📄 arXiv](https://arxiv.org/abs/2604.28169) • [📥 PDF](https://arxiv.org/pdf/2604.28169)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Physical Simulator In-the-Loop Video Generation (2026) PhysAlign: Physics-C...

</details>

<details>
<summary><b>17. MoCapAnything V2: End-to-End Motion Capture for Arbitrary Skeletons</b> ⭐ 79</summary>

<br/>

**👥 Authors:** Weixia He, Mingxi Xu, Dao Thien Phong, Zhengyu Wen, Kehong Gong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.28130) • [📄 arXiv](https://arxiv.org/abs/2604.28130) • [📥 PDF](https://arxiv.org/pdf/2604.28130)

**💻 Code:** [⭐ Code](https://github.com/animotionlab26/MocapAnything)

> 🔗 Fully end-to-end motion capture: Video → Pose → Rotation jointly learned, removing analytical IK entirely. ⚓ Reference-anchored rotation: a single pose–rotation pair defines the rotation coordinate system, resolving the intrinsic ambiguity of po...

</details>

<details>
<summary><b>18. Compliance versus Sensibility: On the Reasoning Controllability in Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Maria Liakata, Yuxiang Zhou, Mahmud Elahi Akhter, Marco Valentino, Xingwei Tan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27251) • [📄 arXiv](https://arxiv.org/abs/2604.27251) • [📥 PDF](https://arxiv.org/pdf/2604.27251)

**💻 Code:** [⭐ Code](https://github.com/Xingwei-Tan/compliance_sensibility)

> We explore whether LLMs can decouple reasoning types from specific tasks by analyzing reasoning conflicts, finding that while models prioritize task logic, these patterns are internally detectable and can be steered to improve controllability.

</details>

<details>
<summary><b>19. World2Minecraft: Occupancy-Driven Simulated Scenes Construction</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Yuan Xie, Xuhong Wang, Jingyu Gong, Haoran Xu, Lechao Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27578) • [📄 arXiv](https://arxiv.org/abs/2604.27578) • [📥 PDF](https://arxiv.org/pdf/2604.27578)

**💻 Code:** [⭐ Code](https://github.com/Nepenthes-zlc/World2Minecraft)

> No abstract available.

</details>

<details>
<summary><b>20. Learning from Noisy Preferences: A Semi-Supervised Learning Approach to Direct Preference Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chen Chen, Yuzhang Shang, Zonglin Lyu, Ming Li, Xinxin Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24952) • [📄 arXiv](https://arxiv.org/abs/2604.24952) • [📥 PDF](https://arxiv.org/pdf/2604.24952)

> Project Page: https://liming-ai.github.io/SemiDPO

</details>

<details>
<summary><b>21. Instruction-Guided Poetry Generation in Arabic and Its Dialects</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.27766) • [📄 arXiv](https://arxiv.org/abs/2604.27766) • [📥 PDF](https://arxiv.org/pdf/2604.27766)

**💻 Code:** [⭐ Code](https://github.com/mbzuai-nlp/instructpoet-ar)

> Arabic poetry finally gets instruction tuning: 1.35M examples, 5 language varieties, and controllable generation for writing, revising, continuing, and analyzing verse.

</details>

<details>
<summary><b>22. ViPO: Visual Preference Optimization at Scale</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rui Wang, Xiaojie Li, Justin Cui, Jie Wu, Ming Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24953) • [📄 arXiv](https://arxiv.org/abs/2604.24953) • [📥 PDF](https://arxiv.org/pdf/2604.24953)

> Project Page: https://liming-ai.github.io/ViPO

</details>

<details>
<summary><b>23. FlashRT: Towards Computationally and Memory Efficient Red-Teaming for Prompt Injection and Knowledge Corruption</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.28157) • [📄 arXiv](https://arxiv.org/abs/2604.28157) • [📥 PDF](https://arxiv.org/pdf/2604.28157)

**💻 Code:** [⭐ Code](https://github.com/wang-yanting/FlashRT)

> The code is available at https://github.com/wang-yanting/FlashRT

</details>

<details>
<summary><b>24. Safety Drift After Fine-Tuning: Evidence from High-Stakes Domains</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dylan Hadfield-Menell, Miranda Bogen, Amy Winecoff, Emaan Bilal Khan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.24902) • [📄 arXiv](https://arxiv.org/abs/2604.24902) • [📥 PDF](https://arxiv.org/pdf/2604.24902)

> This paper analyzes the safety behavior of 100 models to understand the safety impact of benign fine-tuning, including widely deployed fine-tunes in the medical and legal domains as well as controlled adaptations of open foundation models alongsid...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 24 |
| 📅 Today | [`2026-05-03.json`](data/daily/2026-05-03.json) | 24 |
| 📆 This Week | [`2026-W17.json`](data/weekly/2026-W17.json) | 84 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 59 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-03 | 24 | [View JSON](data/daily/2026-05-03.json) |
| 📄 2026-05-02 | 24 | [View JSON](data/daily/2026-05-02.json) |
| 📄 2026-05-01 | 11 | [View JSON](data/daily/2026-05-01.json) |
| 📄 2026-04-30 | 4 | [View JSON](data/daily/2026-04-30.json) |
| 📄 2026-04-29 | 12 | [View JSON](data/daily/2026-04-29.json) |
| 📄 2026-04-28 | 3 | [View JSON](data/daily/2026-04-28.json) |
| 📄 2026-04-27 | 6 | [View JSON](data/daily/2026-04-27.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |
| 📅 2026-W15 | 99 | [View JSON](data/weekly/2026-W15.json) |
| 📅 2026-W14 | 140 | [View JSON](data/weekly/2026-W14.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 59 | [View JSON](data/monthly/2026-05.json) |
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
