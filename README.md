<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-35-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6677+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">35</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">158</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">69</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6677+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 02, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Adaptive Reward Routing: Dynamic Multi-Reward Optimization for Joint Audio-Video Diffusion via Forward-Process RL</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37200) • [📄 arXiv](https://arxiv.org/abs/2609.37200) • [📥 PDF](https://arxiv.org/pdf/2609.37200)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Multi-reward guided reinforcement learning (i.e., RL) offers a promising way to improve joint audio-video diffusion models along several complementary objectives, including modality-specific quality, cross-modal semantic alignment, and temporal sy...

</details>

<details>
<summary><b>2. Hierarchical Continuous Diffusion Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02193) • [📄 arXiv](https://arxiv.org/abs/2610.02193) • [📥 PDF](https://arxiv.org/pdf/2610.02193)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/rhfeiyang/HC-DLM)

> HC-DLM couples discrete tokens with a persistent continuous latent state, preserving token dependencies during parallel denoising and improving reasoning and language modeling over discrete and continuous diffusion baselines.

</details>

<details>
<summary><b>3. AutoGUIWorld: Image Generators as Visual World Models for GUI Agent</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01215) • [📄 arXiv](https://arxiv.org/abs/2610.01215) • [📥 PDF](https://arxiv.org/pdf/2610.01215)

**💻 Code:** [⭐ Code](https://github.com/ImYangC7/AutoGUIWorld) • [⭐ Code](https://github.com/huggingface)

> Image Generators as Visual World Models for GUI Agent github: https://github.com/ImYangC7/AutoGUIWorld full paper: https://huggingface.co/YangC777/AGW-35B/blob/main/AutoGUIWorld_Report.pdf

</details>

<details>
<summary><b>4. ActiveSaddler: Automated Curriculum Learning for Agent Harness Optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00906) • [📄 arXiv](https://arxiv.org/abs/2610.00906) • [📥 PDF](https://arxiv.org/pdf/2610.00906)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/AutoSaddler/tree/feat/activesaddler)

> https://github.com/microsoft/AutoSaddler/tree/feat/activesaddler

</details>

<details>
<summary><b>5. Retrieval-Augmented Skill Optimization via Cross-Harness Adaptation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jeehye Na, Dohwan Ko, Jihwan Park, Ji Soo Lee, Jaewon Chu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38024) • [📄 arXiv](https://arxiv.org/abs/2609.38024) • [📥 PDF](https://arxiv.org/pdf/2609.38024)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Agent skills are becoming valuable repositories of reusable procedural knowledge, yet most skill optimization methods still learn each skill largely from scratch through costly agent rollouts. We introduce RASO, a framework that retrieves relevant...

</details>

<details>
<summary><b>6. Agent Priors-guided Policy Learning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35690) • [📄 arXiv](https://arxiv.org/abs/2609.35690) • [📥 PDF](https://arxiv.org/pdf/2609.35690)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Agentics-robotics/Agent-Priors-guided-Policy-Learning)

> No abstract available.

</details>

<details>
<summary><b>7. Beyond the Current Scene: Event-Referential Grasping with Active View Selection</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39375) • [📄 arXiv](https://arxiv.org/abs/2609.39375) • [📥 PDF](https://arxiv.org/pdf/2609.39375)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SNU-VGILab/BeyondCSe)

> BeyondCSe enables event-referential robotic grasping, allowing robots to identify and grasp objects referenced by past events even when they are no longer visible, through video reasoning and event-conditioned active view selection.

</details>

<details>
<summary><b>8. World Observer: Joint Actor-Observer Generation for Persistent World Modeling</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02162) • [📄 arXiv](https://arxiv.org/abs/2610.02162) • [📥 PDF](https://arxiv.org/pdf/2610.02162)

**💻 Code:** [⭐ Code](https://github.com/cvlab-kaist/world-observer) • [⭐ Code](https://github.com/huggingface)

> How can a world model continuously observe regions beyond the actor's current view? Video world models simulate how an environment evolves from an agent's actions, yet remain actor-centric. Once an object leaves the actor's view, they lose direct ...

</details>

<details>
<summary><b>9. ROWBench: Do Video Models Render What the Program Specifies?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02205) • [📄 arXiv](https://arxiv.org/abs/2610.02205) • [📥 PDF](https://arxiv.org/pdf/2610.02205)

**💻 Code:** [⭐ Code](https://github.com/AlayaLab/PROWBench) • [⭐ Code](https://github.com/huggingface)

> A program runs the world; a video model renders it. PROWBench asks whether video models render what the program specifies — the right action, the right outcome, at the right time. 170 test cases, 200 camera views and 600 proxy inputs (Coarse 3D, C...

</details>

<details>
<summary><b>10. GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38923) • [📄 arXiv](https://arxiv.org/abs/2609.38923) • [📥 PDF](https://arxiv.org/pdf/2609.38923)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The data and models are available at https://huggingface.co/collections/groundhogLLM/graphforge

</details>

<details>
<summary><b>11. Scaling and Distilling Text Embeddings for Better Diffusibility</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01016) • [📄 arXiv](https://arxiv.org/abs/2605.10938) • [📥 PDF](https://arxiv.org/pdf/2610.01016)

**💻 Code:** [⭐ Code](https://github.com/la0ka1/diffusing-scaled-text-embeddings) • [⭐ Code](https://github.com/huggingface)

> Glad to share our recent paper on the latent space for continuous diffusion language models (DLMs)! In this paper, we find that scaling text embeddings can greatly boost the performance of continuous DLMs; for instance, by replacing the T5-small e...

</details>

<details>
<summary><b>12. 4Director: Controlling Video World Models with Rigid 3D Geometry</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02160) • [📄 arXiv](https://arxiv.org/abs/2610.02160) • [📥 PDF](https://arxiv.org/pdf/2610.02160)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VVeiCao/4Director)

> 4Director turns an image into an editable 3D scene. Every object becomes a complete mesh, placed in the same 3D space as the background and the camera. You place the camera, draw a rigid trajectory for each object, and can bring in new objects fro...

</details>

<details>
<summary><b>13. Sharpening Tax in Post-Training</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01509) • [📄 arXiv](https://arxiv.org/abs/2610.01509) • [📥 PDF](https://arxiv.org/pdf/2610.01509)

**💻 Code:** [⭐ Code](https://github.com/changdaeoh/sharpening-tax) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. Beyond Memory: Harnessing Long-Horizon Agents with Explicit Belief States</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01415) • [📄 arXiv](https://arxiv.org/abs/2610.01415) • [📥 PDF](https://arxiv.org/pdf/2610.01415)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/luoyu100/PoS)

> Long-horizon agents need to keep track of the current world state to guide their decisions. Yet maintaining a coherent belief alone does not ensure progress: agents can keep acting without meaningfully advancing their goals, a failure mode this pa...

</details>

<details>
<summary><b>15. RPTune: Learned Context Curation for LLM Catalog Search</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00964) • [📄 arXiv](https://arxiv.org/abs/2610.00964) • [📥 PDF](https://arxiv.org/pdf/2610.00964)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> For small merchant businesses (SMBs) whose catalogs fit within a long-context LLM, full-catalog prompting offers a compelling alternative to multi-stage retrieval designed primarily for large marketplaces with millions of items. However, fitting t...

</details>

<details>
<summary><b>16. AutoDataBench: A Data-centric Testbed for Accelerating Auto Research</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40097) • [📄 arXiv](https://arxiv.org/abs/2609.40097) • [📥 PDF](https://arxiv.org/pdf/2609.40097)

**💻 Code:** [⭐ Code](https://github.com/AutoDataBench/AutoDataBench) • [⭐ Code](https://github.com/huggingface)

> Existing auto-research benchmarks often entangle multiple sources of improvement, including training frameworks, hyperparameters, compute budgets, and data, making it difficult to attribute why one frontier agent outperforms another to specific re...

</details>

<details>
<summary><b>17. Do Audio LLMs Listen Before They Act? Diagnosing Acoustic-Context Gating in Voice Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yushi Sun, Nanchen Hu, doudouwer

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32536) • [📄 arXiv](https://arxiv.org/abs/2609.32536) • [📥 PDF](https://arxiv.org/pdf/2609.32536)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce VGBench, a 1,018-item diagnostic benchmark for action-level addressedness across side-talk, self-talk, and speaker-switch scenarios.

</details>

<details>
<summary><b>18. When Users Change Their Minds: Measuring and Repairing Intent Drift in LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yushi Sun, Zixin Chen, Bowen Cao, doudouwer

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32520) • [📄 arXiv](https://arxiv.org/abs/2609.32520) • [📥 PDF](https://arxiv.org/pdf/2609.32520)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce IntentFlux, an executable benchmark that converts verifiable tasks into dialogues with controlled intent changes while preserving their original graders. In a 627-case calibration, mean task score falls from 0.476 to 0.384 as dialogue...

</details>

<details>
<summary><b>19. PixelDense: Dense Prediction as Representation Alignment for Pixel Diffusion</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Yiqing Yang, Avery Li, Wenhao Zhang, Daiqing Qi, Lehan Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00483) • [📄 arXiv](https://arxiv.org/abs/2610.00483) • [📥 PDF](https://arxiv.org/pdf/2610.00483)

**💻 Code:** [⭐ Code](https://github.com/Hansxsourse/PixelDense) • [⭐ Code](https://github.com/huggingface)

> NeurIPS 2026

</details>

<details>
<summary><b>20. A Missing Piece for Trustworthy AI Reviewers: From Benchmarking Rhetorical Robustness to SciCore Review</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Han Chen, Jianpeng Chen, Chengrui Fan, Ming Li, Chenguang Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39027) • [📄 arXiv](https://arxiv.org/abs/2609.39027) • [📥 PDF](https://arxiv.org/pdf/2609.39027)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> AI reviewers can assign different judgments to manuscripts that report the same science in different wording, potentially rewarding rhetorical optimization over scientific improvement. We formulate Rhetorical Robustness as the joint requirement of...

</details>

<details>
<summary><b>21. FlexRouter: Learning Complementary Model Sets for Flexible LLM Routing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hongjie Chen, Samyadeep Basu, Tiankai Yang, Harry Yang, Wang Wei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38585) • [📄 arXiv](https://arxiv.org/abs/2609.38585) • [📥 PDF](https://arxiv.org/pdf/2609.38585)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. Controlled Decoding Attacks on Black-Box LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ryan A. Rossi, Franck Dernoncourt, Wei Yang, Shawn Li, Jesson Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36956) • [📄 arXiv](https://arxiv.org/abs/2609.36956) • [📥 PDF](https://arxiv.org/pdf/2609.36956)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>23. Joint and Cross-Modal Video-Audio Generation and Editing: A Unified Formulation and Design Taxonomy</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Daksh Dangi, Wang Wei, Sai Karthik Navuluru, Abhinav Sharma, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34381) • [📄 arXiv](https://arxiv.org/abs/2609.34381) • [📥 PDF](https://arxiv.org/pdf/2609.34381)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>24. OmniSeek: Native Tool Integration for Multi-turn Audio-Visual Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuanjun Xiong, Jingru Yi, Jialu Li, Jiteng Mu, Haibo Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02181) • [📄 arXiv](https://arxiv.org/abs/2610.02181) • [📥 PDF](https://arxiv.org/pdf/2610.02181)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present OmniSeek, an agentic framework that transforms an Omni Large Language Model (Omni-LLM) into an active, multi-turn reasoning agent with native tool use. Rather than passively processing an entire audio-visual sequence in a single forward...

</details>

<details>
<summary><b>25. Align Then Reason: A Multimodal Lip-Sync Judge for Dubbing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00825) • [📄 arXiv](https://arxiv.org/abs/2610.00825) • [📥 PDF](https://arxiv.org/pdf/2610.00825)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Dubbing quality control requires a reference-free judge that can determine whether a candidate text line matches a speaker's visible articulation in both content and timing, using only silent video and text because dubbed audio may not yet exist. ...

</details>

<details>
<summary><b>26. Predictive Credit: Measuring What Scientific Explanations Add to Experimental Forecasts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00314) • [📄 arXiv](https://arxiv.org/abs/2610.00314) • [📥 PDF](https://arxiv.org/pdf/2610.00314)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Measuring What Scientific Explanations Add to Experimental Forecasts

</details>

<details>
<summary><b>27. Rules to Tools: Executable Checks for LLM Agents in Scientific Computing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00313) • [📄 arXiv](https://arxiv.org/abs/2610.00313) • [📥 PDF](https://arxiv.org/pdf/2610.00313)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Executable Checks for LLM Agents in Scientific Computing

</details>

<details>
<summary><b>28. Argo-Bench: Evaluating Data Agents on Enterprise-Scale Workflows</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02122) • [📄 arXiv](https://arxiv.org/abs/2610.02122) • [📥 PDF](https://arxiv.org/pdf/2610.02122)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/TextQLLabs/Argo-Bench)

> Hi everyone, first author here! We realized most established text-to-SQL benchmarks don't yet cover the most important care most companies care about: the agents' ability to make good decisions. While some benchmarks have been pretty good for this...

</details>

<details>
<summary><b>29. Pay for the Fault, Not the Flow: Label-Free In-Flow Multi-Agent Workflow Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wei Cheng, Zach Chen, Shengyu Chen, Haoyu Wang, Xuehang Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01017) • [📄 arXiv](https://arxiv.org/abs/2610.01017) • [📥 PDF](https://arxiv.org/pdf/2610.01017)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://arxiv.org/abs/2610.01017

</details>

<details>
<summary><b>30. Personalized Image Generation with Reasoning and Reflection</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Seunghyun Yoon, Franck Dernoncourt, Qinwen Ge, Ngoc N. Tran, Bo Ni

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00737) • [📄 arXiv](https://arxiv.org/abs/2610.00737) • [📥 PDF](https://arxiv.org/pdf/2610.00737)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>31. Prompt2Skill: Unsupervised Skill Optimization From Natural Language Instructions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tyler Derr, Franck Dernoncourt, Ryan A. Rossi, Li Li, Bo Ni

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38593) • [📄 arXiv](https://arxiv.org/abs/2609.38593) • [📥 PDF](https://arxiv.org/pdf/2609.38593)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>32. Pretrain Once, Route Anywhere: Towards a Foundation Model for LLM Routing</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37362) • [📄 arXiv](https://arxiv.org/abs/2609.37362) • [📥 PDF](https://arxiv.org/pdf/2609.37362)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LAMDA-Model-Reuse/RouteFM)

> Most LLM routers are trained for a fixed workload and candidate pool, requiring retraining as the routing environment changes. RouteFM explores a different paradigm: pretrain the routing capability once and adapt to new environments through contex...

</details>

<details>
<summary><b>33. DexPolicy: Scheduled Exploration for Trajectory-Guided Dexterous Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00360) • [📄 arXiv](https://arxiv.org/abs/2610.00360) • [📥 PDF](https://arxiv.org/pdf/2610.00360)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/DexPolicy) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/AIGeeksGroup/DexPolicy

</details>

<details>
<summary><b>34. Learning What to Recall: Adaptive Multi-Cue Episodic Memory for World Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34677) • [📄 arXiv](https://arxiv.org/abs/2609.34677) • [📥 PDF](https://arxiv.org/pdf/2609.34677)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sony/far)

> More memory isn’t enough—world models need the right memory at the right time. Most retrieval methods rely on fixed heuristics such as recency, pose proximity, or visual similarity, but the most similar past observation is not always the one that ...

</details>

<details>
<summary><b>35. It Takes Workflows to Evolve Better Workflows</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhenhailong Wang, Yangyi Chen, Haifeng Chen, Haoyu Wang, Xuehang Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01026) • [📄 arXiv](https://arxiv.org/abs/2610.01026) • [📥 PDF](https://arxiv.org/pdf/2610.01026)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://arxiv.org/abs/2610.01026

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 35 |
| 📅 Today | [`2026-10-02.json`](data/daily/2026-10-02.json) | 35 |
| 📆 This Week | [`2026-W39.json`](data/weekly/2026-W39.json) | 158 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 69 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-02 | 35 | [View JSON](data/daily/2026-10-02.json) |
| 📄 2026-10-01 | 34 | [View JSON](data/daily/2026-10-01.json) |
| 📄 2026-09-30 | 45 | [View JSON](data/daily/2026-09-30.json) |
| 📄 2026-09-29 | 37 | [View JSON](data/daily/2026-09-29.json) |
| 📄 2026-09-28 | 7 | [View JSON](data/daily/2026-09-28.json) |
| 📄 2026-09-27 | 22 | [View JSON](data/daily/2026-09-27.json) |
| 📄 2026-09-26 | 22 | [View JSON](data/daily/2026-09-26.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W39 | 158 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |
| 📅 2026-W36 | 88 | [View JSON](data/weekly/2026-W36.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 69 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |

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
