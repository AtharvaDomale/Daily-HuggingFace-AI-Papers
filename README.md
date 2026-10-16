<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-34-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7246+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">34</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">119</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">638</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7246+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 16, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding</b> ⭐ 54</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18829) • [📄 arXiv](https://arxiv.org/abs/2610.18829) • [📥 PDF](https://arxiv.org/pdf/2610.18829)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/PolyU-VCLab/OmniCompass)

> OmniCompass is a holistic benchmark for omni-modal video grounding and understanding, comprising 820 videos and built with approximately 1,800 human annotation hours. It covers 2 target specifications (Text and Text+Image), 2 complementary evaluat...

</details>

<details>
<summary><b>2. Speculative Forcing for Efficient Autoregressive Video Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22347) • [📄 arXiv](https://arxiv.org/abs/2610.22347) • [📥 PDF](https://arxiv.org/pdf/2610.22347)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/naver-ai/speculative-forcing)

> Few-step autoregressive video generators enable streaming generation, yet they run every token through the final denoising step, regardless of how much refinement it needs. We find this uniform computation is often unnecessary: many tokens already...

</details>

<details>
<summary><b>3. Orbia: Do Generated Videos Form a Consistent 3D World?</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20693) • [📄 arXiv](https://arxiv.org/abs/2610.20693) • [📥 PDF](https://arxiv.org/pdf/2610.20693)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/worldbench/orbia)

> Video generators serve as interactive worlds for exploration and simulation, yet realistic and visually appealing frames do not necessarily depict a consistent 3D world. Existing benchmarks assess selected geometric properties or use vision-langua...

</details>

<details>
<summary><b>4. Intern Lumina U2: A Multi-Codebook Diffusion Large Language Model for Omni-Visual Understanding and Image Generation</b> ⭐ 68</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20818) • [📄 arXiv](https://arxiv.org/abs/2610.20818) • [📥 PDF](https://arxiv.org/pdf/2610.20818)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/InternLM/InternLumina-U2)

> 🌟 Intern Lumina U2: A Multi-Codebook Diffusion Large Language Model for Omni-Visual Understanding and Image Generation Intern Lumina U2 is a single unified model for image / video / 3D understanding, text-to-image generation and image editing. It ...

</details>

<details>
<summary><b>5. EdgeWise: Bringing Long-Horizon Agents to the Edge via Selective Cloud Delegation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17560) • [📄 arXiv](https://arxiv.org/abs/2610.17560) • [📥 PDF](https://arxiv.org/pdf/2610.17560)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Complex and long-horizon agent tasks often rely on frontier models deployed in the cloud to achieve strong performance. However, using these models for every step of a long-horizon trajectory incurs substantial inference cost, even though not all ...

</details>

<details>
<summary><b>6. 3DCodeVerse: Exploring 3D Worlds as Code</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22341) • [📄 arXiv](https://arxiv.org/abs/2610.22341) • [📥 PDF](https://arxiv.org/pdf/2610.22341)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/gaoypeng/3dcodeverse)

> Everything 3D, generated as code only (no image/video/3D diffusion models needed): static and articulated objects, scenes, and entire worlds in space, plus animation and motion over time.

</details>

<details>
<summary><b>7. Terminal-RSI: Recursive Self-Improving Terminal Agents via Recursive Self-Rewrite and Policy-Refine</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22192) • [📄 arXiv](https://arxiv.org/abs/2610.22192) • [📥 PDF](https://arxiv.org/pdf/2610.22192)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Converting verifier outcomes into useful policy updates and partial progress into complete task resolution is crucial for terminal agents. We present Terminal-RSI, which alternates Recursive Self-Rewrite (RSR) of verified trajectories with critic-...

</details>

<details>
<summary><b>8. JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21412) • [📄 arXiv](https://arxiv.org/abs/2610.21412) • [📥 PDF](https://arxiv.org/pdf/2610.21412)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/JevAgentHarness/JevAgentHarness)

> JevAgentHarness

</details>

<details>
<summary><b>9. Routing Policy Optimization for Online Diffusion Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22282) • [📄 arXiv](https://arxiv.org/abs/2610.22282) • [📥 PDF](https://arxiv.org/pdf/2610.22282)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reinforcement learning can align visual generators with reward signals beyond their pretraining objectives, but existing score-function methods typically introduce stochasticity by perturbing the reverse denoising trajectory, leading to costly rol...

</details>

<details>
<summary><b>10. Velox: Spatial Memory Makes World Models Efficient</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22311) • [📄 arXiv](https://arxiv.org/abs/2610.22311) • [📥 PDF](https://arxiv.org/pdf/2610.22311)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Video world models turn actions and camera motion into long visual rollouts, making video a natural interface for interactive simulation. Such interaction requires responsive generation, yet coherent rollouts must retain visual memory of what has ...

</details>

<details>
<summary><b>11. Do Better Visual Representations Always Lead to Better End-to-End Autonomous Driving?</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09695) • [📄 arXiv](https://arxiv.org/abs/2610.09695) • [📥 PDF](https://arxiv.org/pdf/2610.09695)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenDriveLab/ViRA)

> Visual foundation models (VFMs) are increasingly integrated into end-to-end autonomous driving for their powerful representations, yet it remains unclear when these representations improve driving performance. To investigate this question, we intr...

</details>

<details>
<summary><b>12. SimVTA: Simulation-Grounded Video-Trajectory Alignment for Driving World Action Models</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.13781) • [📄 arXiv](https://arxiv.org/abs/2610.13781) • [📥 PDF](https://arxiv.org/pdf/2610.13781)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenDriveLab/SimVTA)

> Driving world action models (WAMs) learn from paired trajectories and future observations. Post-training with reinforcement learning allows the policy to explore alternative trajectories and improve planning using driving rewards. However, driving...

</details>

<details>
<summary><b>13. SWE-at-Scale: Scalable Issue Synthesis from Historical Bug Fixes</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21359) • [📄 arXiv](https://arxiv.org/abs/2610.21359) • [📥 PDF](https://arxiv.org/pdf/2610.21359)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cslsolow/SWE-at-scale)

> What if we could turn bug fixes back into bugs? 🐛 That's the idea behind SWE-at-Scale ! We train a model to learn from real-world bug fixes and use it to create new bugs at scale. After all, if we want coding agents to get better at fixing bugs, w...

</details>

<details>
<summary><b>14. Intern-Decision: Efficient Multimodal Decision Making with Autoregressive Masked Language Modeling</b> ⭐ 786</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22131) • [📄 arXiv](https://arxiv.org/abs/2610.22131) • [📥 PDF](https://arxiv.org/pdf/2610.22131)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/InternLM/Intern-Decision) • [⭐ Code](https://github.com/InternLM/Intern-Decision/issues/4)

> We present Intern-Decision-v1's open model family tech report, which contains more details about pretraining data distribution. Future versions of models are under development. Comments and community contributions are welcome! We also find there a...

</details>

<details>
<summary><b>15. InfiniSplat v2: Analyzing and Improving Feed-Forward 3DGS from Any Views</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20341) • [📄 arXiv](https://arxiv.org/abs/2610.20341) • [📥 PDF](https://arxiv.org/pdf/2610.20341)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zju3dv/InfiniSplat)

> We present InfiniSplat v2 , a feed-forward Gaussian reconstruction method for novel-view rendering from arbitrary input views. Through geometry–attribute decoupling, ground-truth camera pose alignment, and large-angle target-view supervision, it a...

</details>

<details>
<summary><b>16. DeltaStreamer: Towards Human-Like Agentic Streaming Video Understanding</b> ⭐ 61</summary>

<br/>

**👥 Authors:** Shaohui Jiao, Shaoyong Jia, Wenzhao Gao, Deshui Miao, BBBBCHAN

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19849) • [📄 arXiv](https://arxiv.org/abs/2610.19849) • [📥 PDF](https://arxiv.org/pdf/2610.19849)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/BBBBchan/DeltaStreamer)

> Meet DeltaStreamer , an open-source agent that continuously watches video, remembers what happened, and revisits visual evidence when memory falls short. 🎬 Video keeps arriving, while questions can come at any time. Before a question appears, the ...

</details>

<details>
<summary><b>17. Learning Where to Split: Dynamic Band Splitting for Audio Source Separation</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.14993) • [📄 arXiv](https://arxiv.org/abs/2610.14993) • [📥 PDF](https://arxiv.org/pdf/2610.14993)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/JusperLee/Dynamic-Band-Splitting)

> We introduce Dynamic Band Splitting (DBS), an end-to-end framework that learns where to split the frequency axis for audio source separation, while keeping a fixed band-token budget. DBS adapts band boundaries to each mixture and works with BSRNN,...

</details>

<details>
<summary><b>18. Memory as Weights: Internalizing Long-Term History for Streaming Videos</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22070) • [📄 arXiv](https://arxiv.org/abs/2610.22070) • [📥 PDF](https://arxiv.org/pdf/2610.22070)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/dragonlzm/MaLoW)

> 💾 What if a streaming AI remembered the past in its weights—not its context window? Excited to introduce MaLoW (Memory as LoRA Weights) in our new paper, Memory as Weights: Internalizing Long-Term History for Streaming Videos. MaLoW incrementally ...

</details>

<details>
<summary><b>19. SCOPE: Semantic Bounded Credit Optimization for Prompt Engineering in Text-to-image Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22264) • [📄 arXiv](https://arxiv.org/abs/2610.22264) • [📥 PDF](https://arxiv.org/pdf/2610.22264)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MAC-AutoML/SCOPE)

> We introduce SCOPE, a GRPO-based RL framework for Text-to-Image prompt rewriting that pools policy log-probability shifts within coherent semantic spans, outperforming vanilla GRPO by 9.48% on GenEval and 7.62% on T2I-CompBench across Qwen-Image, ...

</details>

<details>
<summary><b>20. MFVSR: A Multi-Style and Fine-Grained Benchmark for Video Subtitle Removal</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Chaoyi Wang, Bo Yin, Peng Tang, Qingdong He, siiian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18461) • [📄 arXiv](https://arxiv.org/abs/2610.18461) • [📥 PDF](https://arxiv.org/pdf/2610.18461)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hxyiiii/MFVSR)

> We introduce MFVSR , a multi-style benchmark for fine-grained video subtitle removal evaluation. 3,000 paired videos from Real-HQ, Real-LQ, and AI-generated sources, plus 953 auxiliary in-the-wild videos . 16 languages and 24 subtitle configuratio...

</details>

<details>
<summary><b>21. CommerceGen-Studio: An End-to-End Framework for Commercial-Grade Main-Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20603) • [📄 arXiv](https://arxiv.org/abs/2610.20603) • [📥 PDF](https://arxiv.org/pdf/2610.20603)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present CommerceGen-Studio, an end-to-end framework for generating commercial-grade main images for e-commerce Stock Keeping Units (SKUs).

</details>

<details>
<summary><b>22. RepForcing: Representation-Space Initialization for Autoregressive Video Distillation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20111) • [📄 arXiv](https://arxiv.org/abs/2610.20111) • [📥 PDF](https://arxiv.org/pdf/2610.20111)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LMIUD/RepForcing)

> No abstract available.

</details>

<details>
<summary><b>23. AutoSFT: An Agentic System for Automated Data Research in Supervised Fine-Tuning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22194) • [📄 arXiv](https://arxiv.org/abs/2610.22194) • [📥 PDF](https://arxiv.org/pdf/2610.22194)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Thinking-Space/AutoSFT)

> A first step toward RSI-Service for post-training. AutoSFT is an AI system that post-trains AI: a multi-agent research policy that iteratively designs the SFT data recipe, with a fixed base model, trainer, and budget. Code: https://github.com/Thin...

</details>

<details>
<summary><b>24. Calibrated Typed Decisions for Entity, Relation, and Event Extraction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chenghua Lin, Chenghao Xiao, Yanan Ma, Hanlin Xiao, Yang Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17113) • [📄 arXiv](https://arxiv.org/abs/2610.17113) • [📥 PDF](https://arxiv.org/pdf/2610.17113)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/extraordinarylab/calon)

> Today we’re releasing CalOn, our 0.6B model for fast decisions in information extraction. ⚡ In our NER benchmark, CalOn processes 120 sentences per second. That’s 5.5× the throughput of GLiNER2 and 6.7× that of GLiFormer. We built on TypeSafe’s RL...

</details>

<details>
<summary><b>25. Skill-Decoupled On-Policy Expert Distillation for Unified Vision-Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhenghao Liu, Qing Yang, Zhipeng Xu, Chunyi Peng, Yuqi Xiong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22217) • [📄 arXiv](https://arxiv.org/abs/2610.22217) • [📥 PDF](https://arxiv.org/pdf/2610.22217)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NEUIR/SCOPED)

> Reinforcement learning (RL) can elicit specialized visual capabilities in visionlanguage models (VLMs), yet jointly optimizing heterogeneous objectives often introduces cross-capability gradient conflicts. Training experts independently mitigate...

</details>

<details>
<summary><b>26. CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06829) • [📄 arXiv](https://arxiv.org/abs/2610.06829) • [📥 PDF](https://arxiv.org/pdf/2610.06829)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We’re sharing CLIFT: Conformal Self-Verification for Web Agent Training and Test-Time Scaling. The core idea is to turn expensive judge feedback into a reusable bank of verification questions, each with a URL scope, polarity, and calibrated trust ...

</details>

<details>
<summary><b>27. InfoVis: Unified Slide Image Generation and Editing</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Xin Luo, Haoqiang Zhang, Chenyuan Wu, Xiyan Jiang, sabulin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16784) • [📄 arXiv](https://arxiv.org/abs/2610.16784) • [📥 PDF](https://arxiv.org/pdf/2610.16784)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VectorSpaceLab/InfoVis)

> github: https://github.com/VectorSpaceLab/InfoVis arxiv: https://arxiv.org/pdf/2610.16784 huggingface: https://huggingface.co/papers/2610.16784 Demo： https://huggingface.co/spaces/sabulin/InfoVis

</details>

<details>
<summary><b>28. A team of Gemini 4 Argon agents beats Minecraft</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Roberta Raileanu, Satinder Singh, Maxence Faldor

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22319) • [📄 arXiv](https://arxiv.org/abs/2610.22319) • [📥 PDF](https://arxiv.org/pdf/2610.22319)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>29. Athanor: An Agent-Native System for Autonomous Post-Training Research</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhichao Liu, Ziqi Shu, Jiakun Fan, Yichen Wang, Yufa Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21055) • [📄 arXiv](https://arxiv.org/abs/2610.21055) • [📥 PDF](https://arxiv.org/pdf/2610.21055)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 4 Claude Opus 5.5 agents post-trained Qwen3-4B-Base on their own, and beat the official Qwen3-4B-Instruct.

</details>

<details>
<summary><b>30. TimeRSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19725) • [📄 arXiv](https://arxiv.org/abs/2610.19725) • [📥 PDF](https://arxiv.org/pdf/2610.19725)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/haoxin1998/TimeRSI)

> TimeRSI: Towards Self-Improved Agentic Deployment for Time-Series Foundation Models. It boosts TimesFM 3.0 to #1 on both Fev Bench and TIME leaderboard as of Oct 15, 2026.

</details>

<details>
<summary><b>31. PN-Mix: Pair-Norm Guided Mixed-Precision Key Cache Quantization for Efficient Long-Output LLM Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19605) • [📄 arXiv](https://arxiv.org/abs/2610.19605) • [📥 PDF](https://arxiv.org/pdf/2610.19605)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: PN-Mix enables accurate low-bit key-cache quantization through input-adaptive mixed-precision allocation across RoPE channel pairs, guided by query–key norms computed once during prefill.

</details>

<details>
<summary><b>32. WorldMemBench: A Comprehensive Benchmark for Memory of Video World Models</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19063) • [📄 arXiv](https://arxiv.org/abs/2610.19063) • [📥 PDF](https://arxiv.org/pdf/2610.19063)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/WorldMemBench)

> Hi everyone! We are the authors of WorldMemBench. Here are the official resources: 📄 Paper: https://arxiv.org/abs/2610.19063 🐙 GitHub: https://github.com/ZJU-REAL/WorldMemBench 🏠 Website and leaderboard: https://zju-real.github.io/WorldMemBench/ 🤗...

</details>

<details>
<summary><b>33. EVA-Cap: Optimizing Audiovisual Video Captioning via Event-Centric Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.22186) • [📄 arXiv](https://arxiv.org/abs/2610.22186) • [📥 PDF](https://arxiv.org/pdf/2610.22186)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Event-Centric Optimization for Audiovisual Video Captioning: Novel Perspective, Dataset, and Post-Training Framework.

</details>

<details>
<summary><b>34. ANTE: Anticipate, Then Execute -- Learning Internal World Models for Coding Agents from Verified Beliefs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15368) • [📄 arXiv](https://arxiv.org/abs/2610.15368) • [📥 PDF](https://arxiv.org/pdf/2610.15368)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce ANTE (Anticipate, Then Execute), an agentic RL framework that trains coding agents to predict action outcomes before execution and learn from verified feedback. Belief-Anchored Policy Optimization (BAPO) turns execution-verified forec...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 34 |
| 📅 Today | [`2026-10-16.json`](data/daily/2026-10-16.json) | 34 |
| 📆 This Week | [`2026-W41.json`](data/weekly/2026-W41.json) | 119 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 638 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-16 | 34 | [View JSON](data/daily/2026-10-16.json) |
| 📄 2026-10-15 | 28 | [View JSON](data/daily/2026-10-15.json) |
| 📄 2026-10-14 | 21 | [View JSON](data/daily/2026-10-14.json) |
| 📄 2026-10-13 | 12 | [View JSON](data/daily/2026-10-13.json) |
| 📄 2026-10-12 | 24 | [View JSON](data/daily/2026-10-12.json) |
| 📄 2026-10-11 | 72 | [View JSON](data/daily/2026-10-11.json) |
| 📄 2026-10-10 | 72 | [View JSON](data/daily/2026-10-10.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W41 | 119 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 638 | [View JSON](data/monthly/2026-10.json) |
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
