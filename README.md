<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-84-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7414+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">84</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">287</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">806</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7414+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 18, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Weiqing Li, Guofeng Quan, wenfengfwf, Nothing2Say, Houbingxi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21601) • [📄 arXiv](https://arxiv.org/abs/2610.21601) • [📥 PDF](https://arxiv.org/pdf/2610.21601)

**💻 Code:** [⭐ Code](https://github.com/hhhbxloveai/WaypointOPD) • [⭐ Code](https://github.com/huggingface)

> We introduce Waypoint-guided On-Policy Distillation (Waypoint-OPD), which uses an intermediate teacher checkpoint from the same RL run to compare early and late shifts in relative action preferences at student-visited prefixes.

</details>

<details>
<summary><b>2. OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding</b> ⭐ 76</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18829) • [📄 arXiv](https://arxiv.org/abs/2610.18829) • [📥 PDF](https://arxiv.org/pdf/2610.18829)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/PolyU-VCLab/OmniCompass)

> OmniCompass is a holistic benchmark for omni-modal video grounding and understanding, comprising 820 videos and built with approximately 1,800 human annotation hours. It covers 2 target specifications (Text and Text+Image), 2 complementary evaluat...

</details>

<details>
<summary><b>3. MJev-Spatial: System One Spatial Intelligence via Native Attention Matching</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22276) • [📄 arXiv](https://arxiv.org/abs/2610.22276) • [📥 PDF](https://arxiv.org/pdf/2610.22276)

**💻 Code:** [⭐ Code](https://github.com/Fr0zenCrane/MJev-Spatial) • [⭐ Code](https://github.com/huggingface)

> 🚀 We devloped MJev-Spatial , a System 1 multimodal decision model tailored for spatial intelligence ! 🧠 MJev-Spatial unify spatial question answering, numerical estimation, and image pointing through direct candidate classification. The model reus...

</details>

<details>
<summary><b>4. OpenGCUA: Toward a Fully Open Stack for Generalist Computer Use</b> ⭐ 75</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22200) • [📄 arXiv](https://arxiv.org/abs/2610.22200) • [📥 PDF](https://arxiv.org/pdf/2610.22200)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tongyi-MAI/OpenGCUA)

> 🚀 OpenGCUA First Release | Opening the Full Stack for Generalist Computer Use GPT-6 made computer use the frontier. We're making it open. The MAI-UI team at Alibaba introduces OpenGCUA (Open Generalist CUA) , an ongoing project to develop and open...

</details>

<details>
<summary><b>5. FDE-Gym: The Last Mile of AI Delivery</b> ⭐ 102</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22350) • [📄 arXiv](https://arxiv.org/abs/2610.22350) • [📥 PDF](https://arxiv.org/pdf/2610.22350)

**💻 Code:** [⭐ Code](https://github.com/deeplethe/fde-gym) • [⭐ Code](https://github.com/huggingface)

> Existing agent benchmarks hand the agent a well-specified task. Real customer delivery doesn't: the sponsor's request may target the wrong problem, key constraints are unstated, and success is the customer's own business metric. FDE-Gym puts agent...

</details>

<details>
<summary><b>6. RoTA-Bench: Evaluating 3D Spatial Reasoning through Interactive Projection Alignment</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17665) • [📄 arXiv](https://arxiv.org/abs/2610.17665) • [📥 PDF](https://arxiv.org/pdf/2610.17665)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MAC-AutoML/RoTA-Bench)

> No abstract available.

</details>

<details>
<summary><b>7. Self-Evolving Agent Harness</b> ⭐ 173</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18813) • [📄 arXiv](https://arxiv.org/abs/2610.18813) • [📥 PDF](https://arxiv.org/pdf/2610.18813)

**💻 Code:** [⭐ Code](https://github.com/zgcagi/Harness-RSI) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zgcagi/Harness-RSI%7D)

> Building agents that learn from experience is a central goal of artificial intelligence. Harness self-evolution offers a path toward this goal by adapting the system around a fixed base model. This article develops a unified framework for examinin...

</details>

<details>
<summary><b>8. MAI-Gym: An Open All-in-One Platform for GUI Agent Evaluation and RL Training across Mobile and Computer Environments</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20939) • [📄 arXiv](https://arxiv.org/abs/2610.20939) • [📥 PDF](https://arxiv.org/pdf/2610.20939)

**💻 Code:** [⭐ Code](https://github.com/Tongyi-MAI/MAI-Gym) • [⭐ Code](https://github.com/huggingface)

> Excited to share MAI-Gym, our open platform for evaluating and training GUI agents across mobile, desktop, and web! 🚀 ⚡ One-command evaluation and training Run evaluations across 10 GUI benchmarks or launch large-scale online RL with a single comm...

</details>

<details>
<summary><b>9. Learning Where to Split: Dynamic Band Splitting for Audio Source Separation</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14993) • [📄 arXiv](https://arxiv.org/abs/2610.14993) • [📥 PDF](https://arxiv.org/pdf/2610.14993)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/JusperLee/Dynamic-Band-Splitting)

> We introduce Dynamic Band Splitting (DBS), an end-to-end framework that learns where to split the frequency axis for audio source separation, while keeping a fixed band-token budget. DBS adapts band boundaries to each mixture and works with BSRNN,...

</details>

<details>
<summary><b>10. InfiniSplat v2: Analyzing and Improving Feed-Forward 3DGS from Any Views</b> ⭐ 418</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20341) • [📄 arXiv](https://arxiv.org/abs/2610.20341) • [📥 PDF](https://arxiv.org/pdf/2610.20341)

**💻 Code:** [⭐ Code](https://github.com/zju3dv/InfiniSplat) • [⭐ Code](https://github.com/huggingface)

> We present InfiniSplat v2 , a feed-forward Gaussian reconstruction method for novel-view rendering from arbitrary input views. Through geometry–attribute decoupling, ground-truth camera pose alignment, and large-angle target-view supervision, it a...

</details>

<details>
<summary><b>11. Behavioral Compositionality of Sparse Autoencoder Representations</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21261) • [📄 arXiv](https://arxiv.org/abs/2610.21261) • [📥 PDF](https://arxiv.org/pdf/2610.21261)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenSAE/SAE-Behavioral-Compositionality)

> Can sparse autoencoder features explain model behavior one feature at a time? Sparse autoencoders (SAEs) decode activations linearly, but their downstream behavioral effects need not be additive. We introduce SAE Additivity Residuals (SAR) to meas...

</details>

<details>
<summary><b>12. 3DCodeVerse: Exploring 3D Worlds as Code</b> ⭐ 105</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22341) • [📄 arXiv](https://arxiv.org/abs/2610.22341) • [📥 PDF](https://arxiv.org/pdf/2610.22341)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/gaoypeng/3dcodeverse)

> Everything 3D, generated as code only (no image/video/3D diffusion models needed): static and articulated objects, scenes, and entire worlds in space, plus animation and motion over time.

</details>

<details>
<summary><b>13. SCOPE: Semantic Bounded Credit Optimization for Prompt Engineering in Text-to-image Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22264) • [📄 arXiv](https://arxiv.org/abs/2610.22264) • [📥 PDF](https://arxiv.org/pdf/2610.22264)

**💻 Code:** [⭐ Code](https://github.com/MAC-AutoML/SCOPE) • [⭐ Code](https://github.com/huggingface)

> We introduce SCOPE, a GRPO-based RL framework for Text-to-Image prompt rewriting that pools policy log-probability shifts within coherent semantic spans, outperforming vanilla GRPO by 9.48% on GenEval and 7.62% on T2I-CompBench across Qwen-Image, ...

</details>

<details>
<summary><b>14. Speculative Forcing for Efficient Autoregressive Video Generation</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22347) • [📄 arXiv](https://arxiv.org/abs/2610.22347) • [📥 PDF](https://arxiv.org/pdf/2610.22347)

**💻 Code:** [⭐ Code](https://github.com/naver-ai/speculative-forcing) • [⭐ Code](https://github.com/huggingface)

> Few-step autoregressive video generators enable streaming generation, yet they run every token through the final denoising step, regardless of how much refinement it needs. We find this uniform computation is often unnecessary: many tokens already...

</details>

<details>
<summary><b>15. Each Reward to Its Relevant Tokens: Reward-Specific Token-Level Credit Assignment for Multi-Reward RL</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21439) • [📄 arXiv](https://arxiv.org/abs/2610.21439) • [📥 PDF](https://arxiv.org/pdf/2610.21439)

**💻 Code:** [⭐ Code](https://github.com/zhaihaotian/MRTPO) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>16. VLA-Δ: Learning Physical State Transition for Vision-Language-Action Models</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21306) • [📄 arXiv](https://arxiv.org/abs/2610.21306) • [📥 PDF](https://arxiv.org/pdf/2610.21306)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-OmniAI/VLA-Delta)

> Reliable robot manipulation requires understanding how objects, grippers, and contacts change as a task unfolds. We introduce VLA-Δ, a vision-language-action framework that explicitly learns physical state transitions through structured transition...

</details>

<details>
<summary><b>17. Mosaic: Multi-Session Agentic Reinforcement Learning</b> ⭐ 33</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22174) • [📄 arXiv](https://arxiv.org/abs/2610.22174) • [📥 PDF](https://arxiv.org/pdf/2610.22174)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LARK-AI-Lab/Mosaic)

> In this paper, we propose a more general setting for agent training: Multi-Session Agentic Reinforcement Learning . In this setting, each rollout may contain an arbitrary number of sessions. Each session consists of agent–environment interactions ...

</details>

<details>
<summary><b>18. CommerceGen-Studio: An End-to-End Framework for Commercial-Grade Main-Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20603) • [📄 arXiv](https://arxiv.org/abs/2610.20603) • [📥 PDF](https://arxiv.org/pdf/2610.20603)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present CommerceGen-Studio, an end-to-end framework for generating commercial-grade main images for e-commerce Stock Keeping Units (SKUs).

</details>

<details>
<summary><b>19. Intern Lumina U2: A Multi-Codebook Diffusion Large Language Model for Omni-Visual Understanding and Image Generation</b> ⭐ 78</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20818) • [📄 arXiv](https://arxiv.org/abs/2610.20818) • [📥 PDF](https://arxiv.org/pdf/2610.20818)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/InternLM/InternLumina-U2)

> 🌟 Intern Lumina U2: A Multi-Codebook Diffusion Large Language Model for Omni-Visual Understanding and Image Generation Intern Lumina U2 is a single unified model for image / video / 3D understanding, text-to-image generation and image editing. It ...

</details>

<details>
<summary><b>20. AutoSFT: An Agentic System for Automated Data Research in Supervised Fine-Tuning</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22194) • [📄 arXiv](https://arxiv.org/abs/2610.22194) • [📥 PDF](https://arxiv.org/pdf/2610.22194)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Thinking-Space/AutoSFT)

> A first step toward RSI-Service for post-training. AutoSFT is an AI system that post-trains AI: a multi-agent research policy that iteratively designs the SFT data recipe, with a fixed base model, trainer, and budget. Code: https://github.com/Thin...

</details>

<details>
<summary><b>21. DeltaStreamer: Towards Human-Like Agentic Streaming Video Understanding</b> ⭐ 74</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19849) • [📄 arXiv](https://arxiv.org/abs/2610.19849) • [📥 PDF](https://arxiv.org/pdf/2610.19849)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/BBBBchan/DeltaStreamer) • [⭐ Code](https://github.com/BBBBchan/AI-Tongdan)

> Meet DeltaStreamer , an open-source agent that continuously watches video, remembers what happened, and revisits visual evidence when memory falls short. 🎬 Video keeps arriving, while questions can come at any time. Before a question appears, the ...

</details>

<details>
<summary><b>22. Can World and Action Modeling Benefit Each Other under Unified Causal Training?</b> ⭐ 50</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21095) • [📄 arXiv](https://arxiv.org/abs/2610.21095) • [📥 PDF](https://arxiv.org/pdf/2610.21095)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Interactive-Active-AI/OneWM)

> OneWM — official resources OneWM in 27 seconds A 1080p overview: a complete real-robot stacking task (6×); a selected Cream cheese world-prediction clip with ground-truth comparison, recorded end-effector trajectory, and input actions (0.5×, repla...

</details>

<details>
<summary><b>23. Orbia: Do Generated Videos Form a Consistent 3D World?</b> ⭐ 41</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20693) • [📄 arXiv](https://arxiv.org/abs/2610.20693) • [📥 PDF](https://arxiv.org/pdf/2610.20693)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/worldbench/orbia)

> Video generators serve as interactive worlds for exploration and simulation, yet realistic and visually appealing frames do not necessarily depict a consistent 3D world. Existing benchmarks assess selected geometric properties or use vision-langua...

</details>

<details>
<summary><b>24. WorldCast: Exploring Intelligent Visual Generation in World State Prediction</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20731) • [📄 arXiv](https://arxiv.org/abs/2610.20731) • [📥 PDF](https://arxiv.org/pdf/2610.20731)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/teal024/WorldCast)

> WorldCast is a benchmark for multi-step world state prediction: given multimodal observations and references, can a model generate a coherent sequence showing how the world evolves? WorldCast contains 320 manually designed cases across eight task ...

</details>

<details>
<summary><b>25. InfoVis: Unified Slide Image Generation and Editing</b> ⭐ 34</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16784) • [📄 arXiv](https://arxiv.org/abs/2610.16784) • [📥 PDF](https://arxiv.org/pdf/2610.16784)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VectorSpaceLab/InfoVis)

> github: https://github.com/VectorSpaceLab/InfoVis arxiv: https://arxiv.org/pdf/2610.16784 huggingface: https://huggingface.co/papers/2610.16784 Demo： https://huggingface.co/spaces/sabulin/InfoVis

</details>

<details>
<summary><b>26. Data World: Scaling Data Environments from Scratch</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21656) • [📄 arXiv](https://arxiv.org/abs/2610.21656) • [📥 PDF](https://arxiv.org/pdf/2610.21656)

**💻 Code:** [⭐ Code](https://github.com/Zhanli-Li/Data-World) • [⭐ Code](https://github.com/huggingface)

> Since early this year, we've been following the data agent space, where a common pain point keeps surfacing: agentic data-analysis tasks are notoriously hard to verify — take "why did food-delivery sales drop?"; even human experts can't give one d...

</details>

<details>
<summary><b>27. EdgeWise: Bringing Long-Horizon Agents to the Edge via Selective Cloud Delegation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17560) • [📄 arXiv](https://arxiv.org/abs/2610.17560) • [📥 PDF](https://arxiv.org/pdf/2610.17560)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Complex and long-horizon agent tasks often rely on frontier models deployed in the cloud to achieve strong performance. However, using these models for every step of a long-horizon trajectory incurs substantial inference cost, even though not all ...

</details>

<details>
<summary><b>28. MLExpert: Towards a Generalist Agentic Model for Autonomous Machine Learning Engineering</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22344) • [📄 arXiv](https://arxiv.org/abs/2610.22344) • [📥 PDF](https://arxiv.org/pdf/2610.22344)

**💻 Code:** [⭐ Code](https://github.com/InternScience/MLExpert) • [⭐ Code](https://github.com/huggingface)

> MLExpert is a 35B Mixture-of-Experts agentic model trained to autonomously plan, execute, and refine machine-learning experiments through sustained tool interaction. Rather than relying on hand-crafted workflows, it learns to guide optimization di...

</details>

<details>
<summary><b>29. Intern-Decision: Efficient Multimodal Decision Making with Autoregressive Masked Language Modeling</b> ⭐ 826</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22131) • [📄 arXiv](https://arxiv.org/abs/2610.22131) • [📥 PDF](https://arxiv.org/pdf/2610.22131)

**💻 Code:** [⭐ Code](https://github.com/InternLM/Intern-Decision) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/InternLM/Intern-Decision/issues/4)

> We present Intern-Decision-v1's open model family tech report, which contains more details about pretraining data distribution. Future versions of models are under development. Comments and community contributions are welcome! We also find there a...

</details>

<details>
<summary><b>30. Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22192) • [📄 arXiv](https://arxiv.org/abs/2610.22192) • [📥 PDF](https://arxiv.org/pdf/2610.22192)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Converting verifier outcomes into useful policy updates and partial progress into complete task resolution is crucial for terminal agents. We present Terminal-RSI, which alternates Recursive Self-Rewrite (RSR) of verified trajectories with critic-...

</details>

<details>
<summary><b>31. OmniVerifier-C1: Decomposed Graded Verifier for Rule-Governed Generative World Modeling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kaian Jiang, Tianlin Zhang, Xuhuan Li, comin, liubowei2003

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22355) • [📄 arXiv](https://arxiv.org/abs/2610.22355) • [📥 PDF](https://arxiv.org/pdf/2610.22355)

**💻 Code:** [⭐ Code](https://github.com/Cominclip/OmniVerifier) • [⭐ Code](https://github.com/huggingface)

> code: https://github.com/Cominclip/OmniVerifier

</details>

<details>
<summary><b>32. Routing Policy Optimization for Online Diffusion Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22282) • [📄 arXiv](https://arxiv.org/abs/2610.22282) • [📥 PDF](https://arxiv.org/pdf/2610.22282)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reinforcement learning can align visual generators with reward signals beyond their pretraining objectives, but existing score-function methods typically introduce stochasticity by perturbing the reverse denoising trajectory, leading to costly rol...

</details>

<details>
<summary><b>33. Trajectory-Level Speculation with Semantic Verification for Tool-Using Agents</b> ⭐ 22</summary>

<br/>

**👥 Authors:** Zhijiang Guo, Qun Liu, Chao Chen, yinhongliu, terr1ble

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21463) • [📄 arXiv](https://arxiv.org/abs/2610.21463) • [📥 PDF](https://arxiv.org/pdf/2610.21463)

**💻 Code:** [⭐ Code](https://github.com/LARK-AI-Lab/SpecAgent) • [⭐ Code](https://github.com/huggingface)

> Tool-using agents spend most of their serving cost on one thing: a large model decoding every step, token by token. But many steps are routine (look up an order, read a record, search again), and checking a step is much cheaper than writing it. Sp...

</details>

<details>
<summary><b>34. Velox: Spatial Memory Makes World Models Efficient</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22311) • [📄 arXiv](https://arxiv.org/abs/2610.22311) • [📥 PDF](https://arxiv.org/pdf/2610.22311)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Video world models turn actions and camera motion into long visual rollouts, making video a natural interface for interactive simulation. Such interaction requires responsive generation, yet coherent rollouts must retain visual memory of what has ...

</details>

<details>
<summary><b>35. JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21412) • [📄 arXiv](https://arxiv.org/abs/2610.21412) • [📥 PDF](https://arxiv.org/pdf/2610.21412)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/JevAgentHarness/JevAgentHarness)

> JevAgentHarness

</details>

<details>
<summary><b>36. SWE-at-Scale: Scalable Issue Synthesis from Historical Bug Fixes</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21359) • [📄 arXiv](https://arxiv.org/abs/2610.21359) • [📥 PDF](https://arxiv.org/pdf/2610.21359)

**💻 Code:** [⭐ Code](https://github.com/cslsolow/SWE-at-scale) • [⭐ Code](https://github.com/huggingface)

> What if we could turn bug fixes back into bugs? 🐛 That's the idea behind SWE-at-Scale ! We train a model to learn from real-world bug fixes and use it to create new bugs at scale. After all, if we want coding agents to get better at fixing bugs, w...

</details>

<details>
<summary><b>37. Do Better Visual Representations Always Lead to Better End-to-End Autonomous Driving?</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09695) • [📄 arXiv](https://arxiv.org/abs/2610.09695) • [📥 PDF](https://arxiv.org/pdf/2610.09695)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenDriveLab/ViRA)

> Visual foundation models (VFMs) are increasingly integrated into end-to-end autonomous driving for their powerful representations, yet it remains unclear when these representations improve driving performance. To investigate this question, we intr...

</details>

<details>
<summary><b>38. SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.13781) • [📄 arXiv](https://arxiv.org/abs/2610.13781) • [📥 PDF](https://arxiv.org/pdf/2610.13781)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenDriveLab/SimVTA)

> Driving world action models (WAMs) learn from paired trajectories and future observations. Post-training with reinforcement learning allows the policy to explore alternative trajectories and improve planning using driving rewards. However, driving...

</details>

<details>
<summary><b>39. MFVSR: A Multi-Style and Fine-Grained Benchmark for Video Subtitle Removal</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Chaoyi Wang, Bo Yin, Peng Tang, Qingdong He, siiian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18461) • [📄 arXiv](https://arxiv.org/abs/2610.18461) • [📥 PDF](https://arxiv.org/pdf/2610.18461)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hxyiiii/MFVSR)

> We introduce MFVSR , a multi-style benchmark for fine-grained video subtitle removal evaluation. 3,000 paired videos from Real-HQ, Real-LQ, and AI-generated sources, plus 953 auxiliary in-the-wild videos . 16 languages and 24 subtitle configuratio...

</details>

<details>
<summary><b>40. Athanor: An Agent-Native System for Autonomous Post-Training Research</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21055) • [📄 arXiv](https://arxiv.org/abs/2610.21055) • [📥 PDF](https://arxiv.org/pdf/2610.21055)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 4 Claude Opus 5.5 agents post-trained Qwen3-4B-Base on their own, and beat the official Qwen3-4B-Instruct.

</details>

<details>
<summary><b>41. Virtual Head Attention</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17959) • [📄 arXiv](https://arxiv.org/abs/2610.17959) • [📥 PDF](https://arxiv.org/pdf/2610.17959)

**💻 Code:** [⭐ Code](https://github.com/PaddlePaddle/VHA) • [⭐ Code](https://github.com/huggingface)

> Can we reduce KV cache by 4× while improving model performance? Meet VHA.

</details>

<details>
<summary><b>42. Learning Self-Orchestration in Decentralized Agent Swarms</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22322) • [📄 arXiv](https://arxiv.org/abs/2610.22322) • [📥 PDF](https://arxiv.org/pdf/2610.22322)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Embodied-Minds-Lab/MPT)

> Market-based post-training (MPT), a framework for jointly learning action policies and self-orchestration in decentralized agent swarms.

</details>

<details>
<summary><b>43. CarryOn: A Dual-Loop Harness for Personalized and Reliable Robot Assistance</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jinhui Ye, Sengqiao Yang, Yuxin Chen, Ricky06662, Vincent2311

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21453) • [📄 arXiv](https://arxiv.org/abs/2610.21453) • [📥 PDF](https://arxiv.org/pdf/2610.21453)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/JIA-Lab-research/CarryOn)

> CarryOn combines personalized planning , runtime protection , and offline recovery refinement around a frozen VLA. The planning loop maintains persistent, revisable user and scene context to guide personalized task decisions. The execution loop co...

</details>

<details>
<summary><b>44. CoDA: Co-Evolution of Data Engines and Action Policies</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22362) • [📄 arXiv](https://arxiv.org/abs/2610.22362) • [📥 PDF](https://arxiv.org/pdf/2610.22362)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>45. Beyond Policy Rollouts: Embedding-Space Directional Exploration with Anchored Noise for LLM RLVR</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dong Gong, Lina Yao, Huiyi Wang, Daijiao Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16105) • [📄 arXiv](https://arxiv.org/abs/2610.16105) • [📥 PDF](https://arxiv.org/pdf/2610.16105)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce EDEN, which helps LLM RL post-training explore by adding anchored noise to the rollout’s input in embedding space, so the model can reach solutions beyond its usual sampling. In RLVR, a policy sampling the same prompt several times of...

</details>

<details>
<summary><b>46. FlashMask-3: Efficient and Expressive Mask-Aware Distributed Attention</b> ⭐ 35</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17901) • [📄 arXiv](https://arxiv.org/abs/2610.17901) • [📥 PDF](https://arxiv.org/pdf/2610.17901)

**💻 Code:** [⭐ Code](https://github.com/PaddlePaddle/FlashMask) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>47. SUPRA-TEX: 3D Generation with Super-Aligned Texture</b> ⭐ 39</summary>

<br/>

**👥 Authors:** Jichen Pang, Fanpeng Meng, Inso, JieminFang, supra3d

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22352) • [📄 arXiv](https://arxiv.org/abs/2610.22352) • [📥 PDF](https://arxiv.org/pdf/2610.22352)

**💻 Code:** [⭐ Code](https://github.com/Supra-3D/SUPRA-Tex) • [⭐ Code](https://github.com/huggingface)

> We present SUPRA-Tex: 3D Generation with Super-Aligned Texture. Our focus is WYSIWYG texture fidelity: preserving a reference image's lettering, patterns, and material details on a 3D asset, with relightable PBR materials. SUPRA-Tex separates visi...

</details>

<details>
<summary><b>48. LEASH: Uncertainty-Aware Weak-to-Strong Escalation and Oversight for Computer-Use Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiachen Yang, Sharon Li, Yuzhe Yang, Changdae Oh, cece1031

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22280) • [📄 arXiv](https://arxiv.org/abs/2610.22280) • [📥 PDF](https://arxiv.org/pdf/2610.22280)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>49. Cowork-Gym: Synthesizing Interactive Work Environments at Scale</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xinji Mai, Weikang Zhou, Jiaji Deng, Cheng Chen, Anni Zou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17670) • [📄 arXiv](https://arxiv.org/abs/2610.17670) • [📥 PDF](https://arxiv.org/pdf/2610.17670)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Introducing Cowork-Gym , a framework for synthesizing executable work environments at scale to train agents for dynamic human–agent collaboration. Real work involves changing requirements, user feedback, and edits to shared files. Cowork-Gym gener...

</details>

<details>
<summary><b>50. Memory as Weights: Internalizing Long-Term History for Streaming Videos</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22070) • [📄 arXiv](https://arxiv.org/abs/2610.22070) • [📥 PDF](https://arxiv.org/pdf/2610.22070)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/dragonlzm/MaLoW)

> 💾 What if a streaming AI remembered the past in its weights—not its context window? Excited to introduce MaLoW (Memory as LoRA Weights) in our new paper, Memory as Weights: Internalizing Long-Term History for Streaming Videos. MaLoW incrementally ...

</details>

<details>
<summary><b>51. Calibrated Typed Decisions for Entity, Relation, and Event Extraction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chenghua Lin, Chenghao Xiao, Yanan Ma, Hanlin Xiao, Yang Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17113) • [📄 arXiv](https://arxiv.org/abs/2610.17113) • [📥 PDF](https://arxiv.org/pdf/2610.17113)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/extraordinarylab/calon)

> Today we’re releasing CalOn, our 0.6B model for fast decisions in information extraction. ⚡ In our NER benchmark, CalOn processes 120 sentences per second. That’s 5.5× the throughput of GLiNER2 and 6.7× that of GLiFormer. We built on TypeSafe’s RL...

</details>

<details>
<summary><b>52. LeanPRM: Lean-Grounded Process Supervision for Natural Language Proof Verification</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Saadia Gabriel, James Shiffer, Jingyan Shen, Salman Rahman, Rodney Lafuente-Mercado

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17697) • [📄 arXiv](https://arxiv.org/abs/2610.17697) • [📥 PDF](https://arxiv.org/pdf/2610.17697)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce LeanPRM, a framework that uses the Lean proof assistant to automatically label each step of natural-language proofs, for training verifiers that check AI-generated mathematical reasoning without human annotation.

</details>

<details>
<summary><b>53. Gen4P: Video Generators are 4D Perceivers</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Chuanxia Zheng, Zeren Jiang, yitongx

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22371) • [📄 arXiv](https://arxiv.org/abs/2610.22371) • [📥 PDF](https://arxiv.org/pdf/2610.22371)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yitongx/Gen4P)

> TL;DR: Gen4P repurposes a video generator for feed-forward 4D perception, with single-pass encoding, on-demand decoding for spatial flexibility, continuous-time embeddings for temporal flexibility, and a frozen depth LoRA as a plug-in geometry pri...

</details>

<details>
<summary><b>54. SERO: Learning from Executable Oracle Continuations for Long-Horizon Agentic Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19397) • [📄 arXiv](https://arxiv.org/abs/2610.19397) • [📥 PDF](https://arxiv.org/pdf/2610.19397)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Long-horizon agentic RL is constrained by the futures visited by the current policy. When rollouts fail to reach useful behaviors, improved credit assignment cannot provide supervision for states and actions that were never encountered. On-policy ...

</details>

<details>
<summary><b>55. MMSearch-OPD: Easy Made Hard, Hard Made Learnable for Long-Horizon Multimodal Search Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22240) • [📄 arXiv](https://arxiv.org/abs/2610.22240) • [📥 PDF](https://arxiv.org/pdf/2610.22240)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> MMSearch-OPD refines existing easy VQA into controllably harder search questions and combines process-quality rewards with on-policy distillation to improve long-horizon multimodal search.

</details>

<details>
<summary><b>56. WorldMemBench: A Comprehensive Benchmark for Memory of Video World Models</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19063) • [📄 arXiv](https://arxiv.org/abs/2610.19063) • [📥 PDF](https://arxiv.org/pdf/2610.19063)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/WorldMemBench)

> Hi everyone! We are the authors of WorldMemBench. Here are the official resources: 📄 Paper: https://arxiv.org/abs/2610.19063 🐙 GitHub: https://github.com/ZJU-REAL/WorldMemBench 🏠 Website and leaderboard: https://zju-real.github.io/WorldMemBench/ 🤗...

</details>

<details>
<summary><b>57. CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06829) • [📄 arXiv](https://arxiv.org/abs/2610.06829) • [📥 PDF](https://arxiv.org/pdf/2610.06829)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We’re sharing CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling. The core idea is to turn expensive judge feedback into a reusable bank of verification questions, each with a URL scope, polarity, and calibrated trust ...

</details>

<details>
<summary><b>58. ANTE: Anticipate, Then Execute -- Learning Internal World Models for Coding Agents from Verified Beliefs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15368) • [📄 arXiv](https://arxiv.org/abs/2610.15368) • [📥 PDF](https://arxiv.org/pdf/2610.15368)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce ANTE (Anticipate, Then Execute), an agentic RL framework that trains coding agents to predict action outcomes before execution and learn from verified feedback. Belief-Anchored Policy Optimization (BAPO) turns execution-verified forec...

</details>

<details>
<summary><b>59. Mine, Sample, Render: Test-Time Diversity Alignment Of Generative Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nebojsa Jojic, Morris E. Sharp, kostino

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21185) • [📄 arXiv](https://arxiv.org/abs/2610.21185) • [📥 PDF](https://arxiv.org/pdf/2610.21185)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Modern generative models produce high-quality individual outputs but struggle to sample diversely, even when explicitly instructed to do so. MINE-SAMPLE-RENDER (MSR) is a test-time approach that externalizes diversity into an explicit, corpus-deri...

</details>

<details>
<summary><b>60. Scalable Weak-to-Strong Verifier Hardening for Agent Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15924) • [📄 arXiv](https://arxiv.org/abs/2610.15924) • [📥 PDF](https://arxiv.org/pdf/2610.15924)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> As agents get stronger, reward hacking is becoming more common. We show that weaker models, given more compute and white-box verifier access, can harden verifiers against stronger agents. https://verifier-hardening.github.io

</details>

<details>
<summary><b>61. Copy-Paste Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19247) • [📄 arXiv](https://arxiv.org/abs/2610.19247) • [📥 PDF](https://arxiv.org/pdf/2610.19247)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Transformers are intuitively considered to be efficient at copying from their inputs. We actually show that the ability of LLMs to copy in context can be improved through explicit modeling. We propose Copy-Paste Language Models (CPLMs), which add ...

</details>

<details>
<summary><b>62. Recursive Environment Scaling for General Long-Horizon Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lingbei Meng, Bofan Zhu, Yichen Xiang, Zhengbo Jiao, JLingxiao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16397) • [📄 arXiv](https://arxiv.org/abs/2610.16397) • [📥 PDF](https://arxiv.org/pdf/2610.16397)

**💻 Code:** [⭐ Code](https://github.com/Glimmerxx/recursive-environment-scaling) • [⭐ Code](https://github.com/huggingface)

> Hi everyone! We introduce Recursive Environment Scaling (RES), a framework that makes existing agent environments more challenging while preserving their core task objectives. RES recursively transforms information access, resource relations, and ...

</details>

<details>
<summary><b>63. A team of Gemini 4 Argon agents beats Minecraft</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Roberta Raileanu, Satinder Singh, Maxence Faldor

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22319) • [📄 arXiv](https://arxiv.org/abs/2610.22319) • [📥 PDF](https://arxiv.org/pdf/2610.22319)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API LongPuzzleBench: Evaluating GUI Agents on Long-Horizon Visual Puzzles (2026...

</details>

<details>
<summary><b>64. PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19605) • [📄 arXiv](https://arxiv.org/abs/2610.19605) • [📥 PDF](https://arxiv.org/pdf/2610.19605)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: PN-Mix enables accurate low-bit key-cache quantization through input-adaptive mixed-precision allocation across RoPE channel pairs, guided by query–key norms computed once during prefill.

</details>

<details>
<summary><b>65. PathFM: Fast Generative Forecasting of Multivariate Trajectories</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22358) • [📄 arXiv](https://arxiv.org/abs/2610.22358) • [📥 PDF](https://arxiv.org/pdf/2610.22358)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> PathFM instead generates complete multivariate trajectories, enabling more agile probabilistic forecasts. Most existing time-series foundation models predict marginal quantiles, which do not capture multi-mode and temporal/cross-var dependence wel...

</details>

<details>
<summary><b>66. Learning to Read Out: Unembedding Dynamics in Language Model Pretraining</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Xinchi Qiu, Andrej Jovanovic, Alex Iacob, William F. Shen, hematteo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21025) • [📄 arXiv](https://arxiv.org/abs/2610.21025) • [📥 PDF](https://arxiv.org/pdf/2610.21025)

**💻 Code:** [⭐ Code](https://github.com/hematteo/learning-to-read-out) • [⭐ Code](https://github.com/huggingface)

> We introduce parameter-trajectory crosscoding, which fits one sparse dictionary to every token's readout (unembedding) row at every checkpoint, using only checkpoint weights. On Pythia, the readout reorganizes in a narrow early window, 0.4–1.1% of...

</details>

<details>
<summary><b>67. Explaining the Roles of Experts in Mixture-of-Experts LLMs for Personalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22127) • [📄 arXiv](https://arxiv.org/abs/2610.22127) • [📥 PDF](https://arxiv.org/pdf/2610.22127)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Do Mixture-of-Experts LLMs find unique routing paths for different user preferences? Our new preprint, “Explaining the Roles of Experts in Mixture-of-Experts LLMs for Personalization,” shows they do.

</details>

<details>
<summary><b>68. H-DDCM: Hadamard Codebooks for Fast Generative Image and Video Compression</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18125) • [📄 arXiv](https://arxiv.org/abs/2610.18125) • [📥 PDF](https://arxiv.org/pdf/2610.18125)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Training-free diffusion-based compression methods repurpose the generative priors of a diffusion or a flow model to compress data without retraining them as task-specific codecs. Recent codebook-based approaches such as Turbo-DDCM achieve strong p...

</details>

<details>
<summary><b>69. Skill-Decoupled On-Policy Expert Distillation for Unified Vision-Language Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Zhenghao Liu, Qing Yang, Zhipeng Xu, Yuqi Xiong, hmhm1229

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22217) • [📄 arXiv](https://arxiv.org/abs/2610.22217) • [📥 PDF](https://arxiv.org/pdf/2610.22217)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NEUIR/SCOPED)

> Reinforcement learning (RL) can elicit specialized visual capabilities in visionlanguage models (VLMs), yet jointly optimizing heterogeneous objectives often introduces cross-capability gradient conflicts. Training experts independently mitigate...

</details>

<details>
<summary><b>70. RepForcing: Representation-Space Initialization for Autoregressive Video Distillation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20111) • [📄 arXiv](https://arxiv.org/abs/2610.20111) • [📥 PDF](https://arxiv.org/pdf/2610.20111)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LMIUD/RepForcing)

> No abstract available.

</details>

<details>
<summary><b>71. TimeRSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19725) • [📄 arXiv](https://arxiv.org/abs/2610.19725) • [📥 PDF](https://arxiv.org/pdf/2610.19725)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/haoxin1998/TimeRSI)

> TimeRSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models. It boosts TimesFM 3.0 to #1 on both Fev Bench and TIME leaderboard as of Oct 15, 2026.

</details>

<details>
<summary><b>72. Equal CER, Unequal Damage: Severity-Weighted Character Error Rate for OCR</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20461) • [📄 arXiv](https://arxiv.org/abs/2610.20461) • [📥 PDF](https://arxiv.org/pdf/2610.20461)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/nguyenx2t/semcer)

> Hi all, author here. Thanks for taking a look. CER charges every character error 1, so a typo costs the same as a wrong digit in a price or a wrong name. semCER keeps CER's alignment and denominator, and gives each edit a severity from 0 (harmless...

</details>

<details>
<summary><b>73. One Bias Score, Two Factors: How Mitigations Affect Option Bias in Multiple-Choice Evaluation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jaeho Lee, Minjae Park, Hyunjong Ok

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20993) • [📄 arXiv](https://arxiv.org/abs/2610.20993) • [📥 PDF](https://arxiv.org/pdf/2610.20993)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Paper about option bias.

</details>

<details>
<summary><b>74. When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for Context-Parallel Prefill</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21500) • [📄 arXiv](https://arxiv.org/abs/2610.21500) • [📥 PDF](https://arxiv.org/pdf/2610.21500)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A remote KV tier is only useful if it beats recomputation, and on current GPUs that is a high bar: for GLM-5.3 on a B300 node, a cache hit must arrive at more than 2.53GB/s, and a fixed engine cost puts short prompts out of reach altogether. A sin...

</details>

<details>
<summary><b>75. Speech event detector: Timestamped detection of non-verbal vocalizations and background sounds for rich speech transcription</b> ⭐ 6</summary>

<br/>

**👥 Authors:** pr0mila-gh0sh, theboringai-work

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14232) • [📄 arXiv](https://arxiv.org/abs/2610.14232) • [📥 PDF](https://arxiv.org/pdf/2610.14232)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/theboringai-work/surround-sound)

> We built this cnn-transformer model for speech-event-detection, I want you to try this and put feedback also if you can come up with a better solution that would be a meaningful step. Thank You.

</details>

<details>
<summary><b>76. ForcingBooster: Adaptive Patch-wise Denoising for Fast Autoregressive Video Diffusion</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12625) • [📄 arXiv](https://arxiv.org/abs/2610.12625) • [📥 PDF](https://arxiv.org/pdf/2610.12625)

**💻 Code:** [⭐ Code](https://github.com/IGL-HKUST/ForcingBooster) • [⭐ Code](https://github.com/huggingface)

> We introduce ForcingBooster, a general framework that dynamically allocates denoising steps to individual patches: easy patches stop early to improve efficiency, while difficult patches receive more refinement steps. This design accelerates genera...

</details>

<details>
<summary><b>77. CoWorld Reasoner: Agentic Multimodal Reasoning with the Evolving Video World Model-Driven Harness</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21910) • [📄 arXiv](https://arxiv.org/abs/2610.21910) • [📥 PDF](https://arxiv.org/pdf/2610.21910)

**💻 Code:** [⭐ Code](https://github.com/ModalityDance/CoWorld-Reasoner) • [⭐ Code](https://github.com/huggingface)

> CoWorld Reasoner is an agentic multimodal reasoning framework that uses a VWM as a unified visual reasoning engine. Its Video World Model-Driven Harness exposes multiple visual operations through tools and uses skills to guide the VLM in deciding ...

</details>

<details>
<summary><b>78. PINT: A Unified Model with Shared Predictive Intent for Robot Manipulation</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Zhengxing Wu, Xingyu Chen, Li Yi, Chao Jing, yxgh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16355) • [📄 arXiv](https://arxiv.org/abs/2610.16355) • [📥 PDF](https://arxiv.org/pdf/2610.16355)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZGCA-HMI-Lab/PINT)

> Multiple experts, one intent. PINT explores how to unify semantic reasoning, world modeling, and action generation while preserving expert specialization. It introduces a shared, chunk-level Predictive Intent that coordinates a pretrained vision-l...

</details>

<details>
<summary><b>79. VascuSR: Iterative Anatomy-Aware Diffusion for CT Super-Resolution</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rose Yu, Wojciech Matusik, Minghao Guo, Jiahe Huang, Hairong Ni

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20509) • [📄 arXiv](https://arxiv.org/abs/2610.20509) • [📥 PDF](https://arxiv.org/pdf/2610.20509)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Topology-aware super resolution for medical imaging.

</details>

<details>
<summary><b>80. EVA-Cap: Optimizing Audiovisual Video Captioning via Event-Centric Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22186) • [📄 arXiv](https://arxiv.org/abs/2610.22186) • [📥 PDF](https://arxiv.org/pdf/2610.22186)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Event-Centric Optimization for Audiovisual Video Captioning: Novel Perspective, Dataset, and Post-Training Framework.

</details>

<details>
<summary><b>81. Diminishing Returns of Data in Sparse Autoencoders: Functional Behavior does not Guarantee Feature Reproducibility</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20680) • [📄 arXiv](https://arxiv.org/abs/2610.20680) • [📥 PDF](https://arxiv.org/pdf/2610.20680)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://arxiv.org/abs/2610.20680

</details>

<details>
<summary><b>82. SimForcing: Distilling Simulation Motion Priors into Real-Domain Robot World Models</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06598) • [📄 arXiv](https://arxiv.org/abs/2610.06598) • [📥 PDF](https://arxiv.org/pdf/2610.06598)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Wang-Xiaodong1899/SimForcing)

> Code: https://github.com/Wang-Xiaodong1899/SimForcing

</details>

<details>
<summary><b>83. AI-Driven, Timing-Aware Power Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** sharmaam85

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18890) • [📄 arXiv](https://arxiv.org/abs/2610.18890) • [📥 PDF](https://arxiv.org/pdf/2610.18890)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Proposes a framework for post-route power recovery where a learned model ranks power-saving cell edits and their repairs, while deterministic EDA analysis gates every acceptance — plus a pre-registered public evaluation protocol on open designs.

</details>

<details>
<summary><b>84. PAIR: Bridging Perception and Action in Vision-Language-Action Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ang Li, Guoheng Sun, AAyano

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09016) • [📄 arXiv](https://arxiv.org/abs/2610.09016) • [📥 PDF](https://arxiv.org/pdf/2610.09016)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Excited to share PAIR: Bridging Perception and Action in Vision-Language-Action Models ! 🚀 A key challenge in continuous-action VLAs is bridging the gap between visual-language understanding and executable robot actions. Instead of leaving this tr...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 84 |
| 📅 Today | [`2026-10-18.json`](data/daily/2026-10-18.json) | 84 |
| 📆 This Week | [`2026-W41.json`](data/weekly/2026-W41.json) | 287 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 806 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-18 | 84 | [View JSON](data/daily/2026-10-18.json) |
| 📄 2026-10-17 | 84 | [View JSON](data/daily/2026-10-17.json) |
| 📄 2026-10-16 | 34 | [View JSON](data/daily/2026-10-16.json) |
| 📄 2026-10-15 | 28 | [View JSON](data/daily/2026-10-15.json) |
| 📄 2026-10-14 | 21 | [View JSON](data/daily/2026-10-14.json) |
| 📄 2026-10-13 | 12 | [View JSON](data/daily/2026-10-13.json) |
| 📄 2026-10-12 | 24 | [View JSON](data/daily/2026-10-12.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 806 | [View JSON](data/monthly/2026-10.json) |
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
