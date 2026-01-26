<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-27-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1377+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">27</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">27</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">639</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1377+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 26, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience</b> ⭐ 154</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15876) • [📄 arXiv](https://arxiv.org/abs/2601.15876) • [📥 PDF](https://arxiv.org/pdf/2601.15876)

**💻 Code:** [⭐ Code](https://github.com/meituan/EvoCUA)

> EvoCUA: Evolving Computer Use Agent 🥇 #1 Open-Source Model on OSWorld | A General-Purpose Multimodal Model Excelling at Computer Use 🔗 Paper: https://arxiv.org/abs/2601.14724 💻 Code: https://github.com/meituan/EvoCUA 🌟 Highlights 🥇 #1 Open-Source ...

</details>

<details>
<summary><b>2. HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding</b> ⭐ 38</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14724) • [📄 arXiv](https://arxiv.org/abs/2601.14724) • [📥 PDF](https://arxiv.org/pdf/2601.14724)

**💻 Code:** [⭐ Code](https://github.com/haowei-freesky/HERMES)

> 🚀 Introducing HERMES: The Future of Real-Time Streaming Video Understanding! While today's Multimodal Large Language Models (MLLMs) perform impressively at offline video comprehension, they often face a "painful trade-off" when it comes to real-ti...

</details>

<details>
<summary><b>3. LLM-in-Sandbox Elicits General Agentic Intelligence</b> ⭐ 81</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16206) • [📄 arXiv](https://arxiv.org/abs/2601.16206) • [📥 PDF](https://arxiv.org/pdf/2601.16206)

**💻 Code:** [⭐ Code](https://github.com/llm-in-sandbox/llm-in-sandbox)

> Introducing LLM-in-Sandbox — put your LLM in a virtual computer to unlock general agentic intelligence for non-code tasks! Significant gains for chemistry, long-context QA, instruction following, and more. No extra training needed. 🌐 Demo: https:/...

</details>

<details>
<summary><b>4. The Flexibility Trap: Why Arbitrary Order Limits Reasoning Potential in Diffusion Language Models</b> ⭐ 71</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15165) • [📄 arXiv](https://arxiv.org/abs/2601.15165) • [📥 PDF](https://arxiv.org/pdf/2601.15165)

**💻 Code:** [⭐ Code](https://github.com/LeapLabTHU/JustGRPO)

> Links 📄 paper: https://arxiv.org/abs/2601.15165 🏠 project page: https://nzl-thu.github.io/the-flexibility-trap 💻 code: https://github.com/LeapLabTHU/JustGRPO 🤗 model: https://huggingface.co/nzl-thu/LLaDA-Instruct-JustGRPO

</details>

<details>
<summary><b>5. BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15197) • [📄 arXiv](https://arxiv.org/abs/2601.15197) • [📥 PDF](https://arxiv.org/pdf/2601.15197)

**💻 Code:** [⭐ Code](https://github.com/ZGC-EmbodyAI/BayesianVLA)

> 🏗️ Architecture BayesianVLA is a novel framework designed to solve the Vision Shortcut problem in Vision-Language-Action (VLA) models. In current VLA training, goal-driven datasets often make language instructions highly predictable from visual ob...

</details>

<details>
<summary><b>6. Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders</b> ⭐ 129</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16208) • [📄 arXiv](https://arxiv.org/abs/2601.16208) • [📥 PDF](https://arxiv.org/pdf/2601.16208)

**💻 Code:** [⭐ Code](https://github.com/ZitengWangNYU/Scale-RAE)

> We scale RAE to text-to-image, and its advantage over VAEs still holds!

</details>

<details>
<summary><b>7. Stable-DiffCoder: Pushing the Frontier of Code Diffusion Large Language Model</b> ⭐ 28</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15892) • [📄 arXiv](https://arxiv.org/abs/2601.15892) • [📥 PDF](https://arxiv.org/pdf/2601.15892)

**💻 Code:** [⭐ Code](https://github.com/ByteDance-Seed/Stable-DiffCoder)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion L...

</details>

<details>
<summary><b>8. SAMTok: Representing Any Mask with Two Words</b> ⭐ 1.51k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16093) • [📄 arXiv](https://arxiv.org/abs/2601.16093) • [📥 PDF](https://arxiv.org/pdf/2601.16093)

**💻 Code:** [⭐ Code](https://github.com/bytedance/Sa2VA/tree/main/projects/samtok)

> Project page: https://zhouyiks.github.io/projects/SAMTok/ Training Code: https://github.com/bytedance/Sa2VA/tree/main/projects/samtok Short Bio:   We present SAMTok, a discrete mask tokenizer that converts any region mask into two special tokens a...

</details>

<details>
<summary><b>9. Learning to Discover at Test Time</b> ⭐ 163</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16175) • [📄 arXiv](https://arxiv.org/abs/2601.16175) • [📥 PDF](https://arxiv.org/pdf/2601.16175)

**💻 Code:** [⭐ Code](https://github.com/test-time-training/discover)

> New paper on scientific discovery with test time training. New discoveries on several open scientific problems.

</details>

<details>
<summary><b>10. Qwen3-TTS Technical Report</b> ⭐ 4.27k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15621) • [📄 arXiv](https://arxiv.org/abs/2601.15621) • [📥 PDF](https://arxiv.org/pdf/2601.15621)

**💻 Code:** [⭐ Code](https://github.com/QwenLM/Qwen3-TTS)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API IndexTTS 2.5 Technical Report (2026) FlashLabs Chroma 1.0: A Real-Time End-...

</details>

<details>
<summary><b>11. Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces</b> ⭐ 1.41k</summary>

<br/>

**👥 Authors:** Boxuan Li, Nicholas Carlini, Alexander G. Shaw, Mike A. Merrill, menorf

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11868) • [📄 arXiv](https://arxiv.org/abs/2601.11868) • [📥 PDF](https://arxiv.org/pdf/2601.11868)

**💻 Code:** [⭐ Code](https://github.com/laude-institute/terminal-bench)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Re...

</details>

<details>
<summary><b>12. Towards Automated Kernel Generation in the Era of LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yixin Shen, Haiming Wu, Chi Hsu Tsai, Peiyu Zang, Yang Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15727) • [📄 arXiv](https://arxiv.org/abs/2601.15727) • [📥 PDF](https://arxiv.org/pdf/2601.15727)

**💻 Code:** [⭐ Code](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation)

> Summary of Key Points Kernel quality is a fundamental bottleneck for modern AI system performance, yet high-quality kernel engineering is expert-intensive, time-consuming, and difficult to scale. Recent advances in large language models (LLMs) and...

</details>

<details>
<summary><b>13. OpenVision 3: A Family of Unified Visual Encoder for Both Understanding and Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15369) • [📄 arXiv](https://arxiv.org/abs/2601.15369) • [📥 PDF](https://arxiv.org/pdf/2601.15369)

> Project Page: https://ucsc-vlaa.github.io/OpenVision3/

</details>

<details>
<summary><b>14. Rethinking Composed Image Retrieval Evaluation: A Fine-Grained Benchmark from Image Editing</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Dingkun Long, Zhuoning Guo, Mingxin Li, Yanzhao Zhang, songtingyu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16125) • [📄 arXiv](https://arxiv.org/abs/2601.16125) • [📥 PDF](https://arxiv.org/pdf/2601.16125)

**💻 Code:** [⭐ Code](https://github.com/SighingSnow/edir)

> A new benchmark for Composed Image Retrieval.

</details>

<details>
<summary><b>15. Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16163) • [📄 arXiv](https://arxiv.org/abs/2601.16163) • [📥 PDF](https://arxiv.org/pdf/2601.16163)

> Cosmos Policy fine-tunes a pretrained video model in one stage for visuomotor control, enabling action latent frames, future state prediction, and planning, achieving state-of-the-art robotic benchmarks.

</details>

<details>
<summary><b>16. ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion</b> ⭐ 91</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16148) • [📄 arXiv](https://arxiv.org/abs/2601.16148) • [📥 PDF](https://arxiv.org/pdf/2601.16148)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/actionmesh)

> 🤗Try it out: https://huggingface.co/spaces/facebook/ActionMesh 🌐Project Page: https://remysabathier.github.io/actionmesh/ 📄Paper: https://remysabathier.github.io/actionmesh/actionmesh_2026.pdf 💻Code: https://github.com/facebookresearch/actionmesh

</details>

<details>
<summary><b>17. VideoMaMa: Mask-Guided Video Matting via Generative Prior</b> ⭐ 108</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14255) • [📄 arXiv](https://arxiv.org/abs/2601.14255) • [📥 PDF](https://arxiv.org/pdf/2601.14255)

**💻 Code:** [⭐ Code](https://github.com/cvlab-kaist/VideoMaMa)

> Demo: https://huggingface.co/spaces/SammyLim/VideoMaMa Git: https://github.com/cvlab-kaist/VideoMaMa Project Page: https://cvlab-kaist.github.io/VideoMaMa/

</details>

<details>
<summary><b>18. PROGRESSLM: Towards Progress Reasoning in Vision-Language Models</b> ⭐ 76</summary>

<br/>

**👥 Authors:** Dingcheng Wang, Haoran Lu, Haosen Sun, Jianshu Zhang, Raymond-Qiancx

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15224) • [📄 arXiv](https://arxiv.org/abs/2601.15224) • [📥 PDF](https://arxiv.org/pdf/2601.15224)

**💻 Code:** [⭐ Code](https://github.com/ProgressLM/ProgressLM)

> Towards General Progress Understanding for Embodied Agents

</details>

<details>
<summary><b>19. Agentic Uncertainty Quantification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15703) • [📄 arXiv](https://arxiv.org/abs/2601.15703) • [📥 PDF](https://arxiv.org/pdf/2601.15703)

> 🛑 Stop the "Spiral of Hallucination" in Autonomous Agents! Long-horizon agents often fail because minor early errors snowball into irreversible failures. We introduce Agentic Uncertainty Quantification (AUQ) , a training-free Dual-Process framewor...

</details>

<details>
<summary><b>20. 360Anything: Geometry-Free Lifting of Images and Videos to 360°</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16192) • [📄 arXiv](https://arxiv.org/abs/2601.16192) • [📥 PDF](https://arxiv.org/pdf/2601.16192)

> 360Anything lifts arbitrary perspective images and videos to seamless, gravity-aligned 360° panoramas, without using any camera or 3D information. Project page: https://360anything.github.io/

</details>

<details>
<summary><b>21. Agentic Confidence Calibration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15778) • [📄 arXiv](https://arxiv.org/abs/2601.15778) • [📥 PDF](https://arxiv.org/pdf/2601.15778)

> 🎯 Don't let your Agents be "Confidently Wrong"! Traditional calibration works for static text, but Autonomous Agents fail differently—errors compound over long trajectories. We introduce Holistic Trajectory Calibration (HTC) , a new paradigm to di...

</details>

<details>
<summary><b>22. From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15690) • [📄 arXiv](https://arxiv.org/abs/2601.15690) • [📥 PDF](https://arxiv.org/pdf/2601.15690)

> 🗺️ The 2026 Roadmap for Reliable AI: Making Uncertainty Actionable We are witnessing a paradigm shift in LLMs: Uncertainty is no longer just a passive score for diagnosis—it is evolving into an Active Control Signal for real-time decision-making. ...

</details>

<details>
<summary><b>23. VIOLA: Towards Video In-Context Learning with Minimal Annotations</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ryo Hachiuma, Hideo Saito, Ryo Fujii

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15549) • [📄 arXiv](https://arxiv.org/abs/2601.15549) • [📥 PDF](https://arxiv.org/pdf/2601.15549)

> Abstract: Generalizing Multimodal Large Language Models (MLLMs) to novel video domains is essential for real-world deployment but remains challenging due to the scarcity of labeled data. While In-Context Learning (ICL) offers a training-free adapt...

</details>

<details>
<summary><b>24. LLM Prompt Evaluation for Educational Applications</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16134) • [📄 arXiv](https://arxiv.org/abs/2601.16134) • [📥 PDF](https://arxiv.org/pdf/2601.16134)

> Some of the observations founded are :- -- Prompt design matters as much as the model : The study shows that different prompt templates using the same LLM produce significantly different educational outcomes, proving prompt engineering is a critic...

</details>

<details>
<summary><b>25. Wigner's Friend as a Circuit: Inter-Branch Communication Witness Benchmarks on Superconducting Quantum Hardware</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Cohaerence

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16004) • [📄 arXiv](https://arxiv.org/abs/2601.16004) • [📥 PDF](https://arxiv.org/pdf/2601.16004)

**💻 Code:** [⭐ Code](https://github.com/christopher-altman/ibm-qml-kernel)

> We implement and benchmark on IBM Quantum hardware the circuit family proposed by Violaris for estimating operational inter-branch communication witnesses, defined as correlations in classical measurement records produced by compiled Wigner's-frie...

</details>

<details>
<summary><b>26. Numba-Accelerated 2D Diffusion-Limited Aggregation: Implementation and Fractal Characterization</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15440) • [📄 arXiv](https://arxiv.org/abs/2601.15440) • [📥 PDF](https://arxiv.org/pdf/2601.15440)

**💻 Code:** [⭐ Code](https://github.com/sandyherho/dla-ideal-solver)

> In this work, we address the performance limitations often encountered in Python-based DLA simulations. By utilizing Numba for just-in-time compilation, we developed an implementation that achieves computational speeds comparable to legacy Fortran...

</details>

<details>
<summary><b>27. MirrorBench: An Extensible Framework to Evaluate User-Proxy Agents for Human-Likeness</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08118) • [📄 arXiv](https://arxiv.org/abs/2601.08118) • [📥 PDF](https://arxiv.org/pdf/2601.08118)

**💻 Code:** [⭐ Code](https://github.com/SAP/mirrorbench)

> The framework is open-sourced at https://github.com/SAP/mirrorbench

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 27 |
| 📅 Today | [`2026-01-26.json`](data/daily/2026-01-26.json) | 27 |
| 📆 This Week | [`2026-W04.json`](data/weekly/2026-W04.json) | 27 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 639 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-26 | 27 | [View JSON](data/daily/2026-01-26.json) |
| 📄 2026-01-25 | 27 | [View JSON](data/daily/2026-01-25.json) |
| 📄 2026-01-24 | 27 | [View JSON](data/daily/2026-01-24.json) |
| 📄 2026-01-23 | 26 | [View JSON](data/daily/2026-01-23.json) |
| 📄 2026-01-22 | 32 | [View JSON](data/daily/2026-01-22.json) |
| 📄 2026-01-21 | 11 | [View JSON](data/daily/2026-01-21.json) |
| 📄 2026-01-20 | 22 | [View JSON](data/daily/2026-01-20.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W04 | 27 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 639 | [View JSON](data/monthly/2026-01.json) |
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
