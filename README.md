<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-37-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6563+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">37</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">44</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">435</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6563+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** September 29, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. TraceDance: An Automated System for Building Agent Behavior Benchmarks from Real-World Agent Deployment Traces</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33295) • [📄 arXiv](https://arxiv.org/abs/2609.33295) • [📥 PDF](https://arxiv.org/pdf/2609.33295)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZhishanQ/TraceDance)

> An agent can complete a task while exhibiting undesirable behavior during execution. Developers need tests for the specific behaviors encountered in deployment, beyond fixed benchmark suites. We present TraceDance, an agent system that constructs ...

</details>

<details>
<summary><b>2. YuE2: Unifying Symbolic and Audio Music Generation at Frontier Quality</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33757) • [📄 arXiv](https://arxiv.org/abs/2609.33757) • [📥 PDF](https://arxiv.org/pdf/2609.33757)

**💻 Code:** [⭐ Code](https://github.com/multimodal-art-projection/YuE/blob/main/docs/technical_report.pdf) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/multimodal-art-projection/YuE)

> We're sharing the YuE2 technical report. YuE2 unifies symbolic and audio music generation: it first writes an editable melody-and-chord score, then renders a full song with vocals and accompaniment. The same model supports zero-shot covers and sco...

</details>

<details>
<summary><b>3. CompoWorld: Compositional Environment Scaling for General Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33665) • [📄 arXiv](https://arxiv.org/abs/2609.33665) • [📥 PDF](https://arxiv.org/pdf/2609.33665)

**💻 Code:** [⭐ Code](https://github.com/AllSpark-Research/CompoWorld) • [⭐ Code](https://github.com/huggingface)

> Automatically generated environments provide a scalable source of interaction data for training general agents. However, existing approaches mainly generate tasks within a single environment, while real-world workflows require agents to connect in...

</details>

<details>
<summary><b>4. Learning to Learn from Context: Synthetic Training from Perturbed Public Documents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33642) • [📄 arXiv](https://arxiv.org/abs/2609.33642) • [📥 PDF](https://arxiv.org/pdf/2609.33642)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Real-world tasks often require large language models (LLMs) to learn from complex task-specific context rather than pretrained parametric knowledge. This capability remains a weakness of LLMs, while human annotation for such task contexts is expen...

</details>

<details>
<summary><b>5. Skill2Env: Capability-Oriented Environment Synthesis from Skills for General Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33772) • [📄 arXiv](https://arxiv.org/abs/2609.33772) • [📥 PDF](https://arxiv.org/pdf/2609.33772)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Executable environments are critical for post-training agents on tasks that require tool use and multi-step interaction, but constructing executable tasks together with their environments remains difficult to scale. Skills provide reusable domain ...

</details>

<details>
<summary><b>6. EmbodiedMemory-Bench: Benchmarking Embodied Memory for Long-Horizon Embodied Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.28236) • [📄 arXiv](https://arxiv.org/abs/2609.28236) • [📥 PDF](https://arxiv.org/pdf/2609.28236)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Excited to share EmbodiedMemory-Bench! 🚀 We study embodied memory for long-horizon interactive tasks, where agents must not only remember past observations, but also continuously update world states, learn from interaction outcomes, and reuse expe...

</details>

<details>
<summary><b>7. AdaTutoRank: Learning to Rerank Document Sets via Adaptive Tutoring Optimization for RAG and Deep Research</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32472) • [📄 arXiv](https://arxiv.org/abs/2609.32472) • [📥 PDF](https://arxiv.org/pdf/2609.32472)

**💻 Code:** [⭐ Code](https://github.com/AdaTutoRank/AdaTutoRank) • [⭐ Code](https://github.com/huggingface)

> Document rerankers determine what evidence reaches the downstream model in RAG and deep research, yet mainstream rerankers select by relevance matching, and individually relevant documents rarely constitute the complete, complementary, non-redunda...

</details>

<details>
<summary><b>8. Diffusion Reward Models</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33803) • [📄 arXiv](https://arxiv.org/abs/2609.33803) • [📥 PDF](https://arxiv.org/pdf/2609.33803)

**💻 Code:** [⭐ Code](https://github.com/thunlp/DRM) • [⭐ Code](https://github.com/huggingface)

> We study the multimodal structure of human preference and recast reward modeling as conditional density estimation over p(r | x, y), and propose Diffusion Reward Model that represents this distribution without committing to any parametric family. ...

</details>

<details>
<summary><b>9. SpatialSpeak: QA-Native Reconstruction with Local and Global Context for Spatial Chain-of-Thought Reasoning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33616) • [📄 arXiv](https://arxiv.org/abs/2609.33616) • [📥 PDF](https://arxiv.org/pdf/2609.33616)

**💻 Code:** [⭐ Code](https://github.com/yangcaoai/SpatialSpeak-VLM) • [⭐ Code](https://github.com/huggingface)

> We introduce SpatialSpeak , a vision-language framework for multi-view spatial reasoning. It first learns local geometry and global scene context through QA-native reconstruction, then combines spatial chain-of-thought reasoning with visual compen...

</details>

<details>
<summary><b>10. RoboFoundry: System-as-Policy Evolution for Self-Learning Embodied Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuxin Cai, Diyuan Hou, Shizhe Zhang, Shuhao Liao, Jingsong Liang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32862) • [📄 arXiv](https://arxiv.org/abs/2609.32862) • [📥 PDF](https://arxiv.org/pdf/2609.32862)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Arxiv: https://arxiv.org/abs/2609.32862 Website: https://jingsongliang.com/robofoundry/

</details>

<details>
<summary><b>11. Self-Evolving Coding Agents: From Digital Programs to Physical-World Intelligence</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Jay Zhu, Shijia Ge, Zelin Zheng, Jingjing Zhou, Hongcheng Gao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35432) • [📄 arXiv](https://arxiv.org/abs/2609.35432) • [📥 PDF](https://arxiv.org/pdf/2609.35432)

**💻 Code:** [⭐ Code](https://github.com/HexaFuture/PhysicalCoding) • [⭐ Code](https://github.com/huggingface)

> Physical Coding represents task state and execution as code. Code as World records objects, relations, constraints, and progress; Code as Policy organizes actions, verification, and recovery. HexaAnything implements this interface through a Harnes...

</details>

<details>
<summary><b>12. Just MLPs: Efficient Visual State Reconstruction for Multimodal Language Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34972) • [📄 arXiv](https://arxiv.org/abs/2609.34972) • [📥 PDF](https://arxiv.org/pdf/2609.34972)

**💻 Code:** [⭐ Code](https://github.com/declare-lab/delta-Vision) • [⭐ Code](https://github.com/huggingface)

> Do we really fully use the a attention mechanism?

</details>

<details>
<summary><b>13. REALM: A Coarse-to-Fine Generative Framework for Embodied Reactive Listening</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33095) • [📄 arXiv](https://arxiv.org/abs/2609.33095) • [📥 PDF](https://arxiv.org/pdf/2609.33095)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/lipzh5/REALM)

> REALM: A Coarse-to-Fine Generative Framework for Embodied Reactive Listening Ever wonder how to make humanoid robots look like they are actually listening to you? 🤖👂 Standard talking-head models struggle with listener motions, resulting in frozen ...

</details>

<details>
<summary><b>14. Rethinking Training-Inference Mismatch in LLM Reinforcement Learning: Where It Arises and How to Correct It</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Porter Jenkins, Yuxiao Yang, Shangzhe Li, Kaixiang Zhao, Tianrun Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32444) • [📄 arXiv](https://arxiv.org/abs/2609.32444) • [📥 PDF](https://arxiv.org/pdf/2609.32444)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kzhao5/CIS-RL)

> 👋 Hi everyone! We introduce Calibrated Importance Sampling (CIS) to address training–inference mismatch in LLM reinforcement learning. 🔍 Why it matters: Even with identical model weights, inference and training engines can assign different token p...

</details>

<details>
<summary><b>15. Program-Verified Self-Evolution for Vision-Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33855) • [📄 arXiv](https://arxiv.org/abs/2609.33855) • [📥 PDF](https://arxiv.org/pdf/2609.33855)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ahmedheakl/VQS)

> Self-evolving vision-language models learn from questions they make from unlabeled images, but the labels they use are often wrong, with human checks showing 24% of majority-vote labels and 18% of model-judge labels are incorrect. VQS fixes this b...

</details>

<details>
<summary><b>16. Improving Test-Time Scaling with Adaptive Looped Transformers</b> ⭐ 85</summary>

<br/>

**👥 Authors:** Xuefei Ning, Xingtai Lv, Aosong Feng, Tianyu Fu, Yichen You

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35748) • [📄 arXiv](https://arxiv.org/abs/2609.35748) • [📥 PDF](https://arxiv.org/pdf/2609.35748)

**💻 Code:** [⭐ Code](https://github.com/thu-nics/TaH) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>17. Fewer Tokens, More Self-Teaching: On-Policy Self-Distillation for Extreme Visual Token Reduction</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32353) • [📄 arXiv](https://arxiv.org/abs/2609.32353) • [📥 PDF](https://arxiv.org/pdf/2609.32353)

**💻 Code:** [⭐ Code](https://github.com/Yrxxxxxxxx1007/LT-OPD) • [⭐ Code](https://github.com/huggingface)

> This paper discusses the scenario of extreme visual token reduction and leverages on-policy self-distillation to solve it.

</details>

<details>
<summary><b>18. SciGen-Verifier: A Multimodal Reasoner for Explainable Verification in Scientific Image Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xi Yu, Shirong Lin, Zuqi Wang, Zhengteng Lin, Jiali Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33399) • [📄 arXiv](https://arxiv.org/abs/2609.33399) • [📥 PDF](https://arxiv.org/pdf/2609.33399)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper introduces SciGen-Verify, the first benchmark for explainable verification of scientific image generation, covering instruction following, multidisciplinary reasoning, and world knowledge with a three-tier protocol over binary judgement...

</details>

<details>
<summary><b>19. Rethinking Automated Voice Similarity by Shifting from EER to Embedding Geometry</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33999) • [📄 arXiv](https://arxiv.org/abs/2609.33999) • [📥 PDF](https://arxiv.org/pdf/2609.33999)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Speaker verification (SV) models are commonly assumed to better capture nuances among speaker characteristics as verification accuracy improves, leading to their widespread use as automated proxies for human voice similarity in speech generation t...

</details>

<details>
<summary><b>20. CoWindow Attention: Full Causal Coverage Is a Collective Property</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32704) • [📄 arXiv](https://arxiv.org/abs/2609.32712) • [📥 PDF](https://arxiv.org/pdf/2609.32704)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Sharing two recent explorations in attention design from our team. We started with two straightforward questions: Does every attention head need to repeatedly attend to the entire causal history? Once attention scores have been computed, do region...

</details>

<details>
<summary><b>21. MassAlloc Attention: Let Attention Allocate Its Own Compute</b> ⭐ 761</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32712) • [📄 arXiv](https://arxiv.org/abs/2609.32712) • [📥 PDF](https://arxiv.org/pdf/2609.32712)

**💻 Code:** [⭐ Code](https://github.com/HKUSTDial/flash-sparse-attention) • [⭐ Code](https://github.com/huggingface)

> Sharing two recent explorations in attention design from our team. We started with two straightforward questions: Does every attention head need to repeatedly attend to the entire causal history? Once attention scores have been computed, do region...

</details>

<details>
<summary><b>22. BaRe-Mem: Bayesian Reliability Memory for Robust and Adaptive Agent Consultation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35551) • [📄 arXiv](https://arxiv.org/abs/2609.35551) • [📥 PDF](https://arxiv.org/pdf/2609.35551)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/declare-lab/BaRe-Mem)

> BaRe-Mem is an online Bayesian reliability memory that learns context-dependent advisor reliability from verified interactions, modulates external advice accordingly, and adaptively decides whether to consult or reason autonomously.

</details>

<details>
<summary><b>23. Beyond Timestamps: Decision-Aligned On-Policy Distillation for Long-Horizon Agents</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Heng Chang, Huan Zhang, Jinrong Liu, Can Lv, Mingju Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33391) • [📄 arXiv](https://arxiv.org/abs/2609.33391) • [📥 PDF](https://arxiv.org/pdf/2609.33391)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/mingju-c/Align-OPSD)

> We introduce AlignOPSD, a framework for improving on-policy distillation in long-horizon agents. The key insight is that temporal alignment does not necessarily imply decision alignment. Existing approaches typically assign supervision based on ti...

</details>

<details>
<summary><b>24. Adaptive Consistency Graph for Long-Horizon Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32754) • [📄 arXiv](https://arxiv.org/abs/2609.32754) • [📥 PDF](https://arxiv.org/pdf/2609.32754)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> How can AI agents go further on long tasks—and stay connected to their original goals? As tasks grow longer, agents need to connect the original requirements, accumulated evidence, and current execution state. Even a locally reasonable decision ca...

</details>

<details>
<summary><b>25. SolveEdit: Benchmarking Visual Problem Solving in Generative Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zehan Wang, Xuerui Qiu, Harold Haodong Chen, Yexin Liu, Wenjie Shu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35504) • [📄 arXiv](https://arxiv.org/abs/2609.35504) • [📥 PDF](https://arxiv.org/pdf/2609.35504)

**💻 Code:** [⭐ Code](https://github.com/WenjieShu/SolveEdit) • [⭐ Code](https://github.com/huggingface)

> Repo: https://github.com/WenjieShu/SolveEdit

</details>

<details>
<summary><b>26. Precise Editing and Flexible Referencing for Interactable Worlds</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34470) • [📄 arXiv](https://arxiv.org/abs/2609.34470) • [📥 PDF](https://arxiv.org/pdf/2609.34470)

**💻 Code:** [⭐ Code](https://github.com/leoisufa/EditWorld) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>27. WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.23033) • [📄 arXiv](https://arxiv.org/abs/2609.23033) • [📥 PDF](https://arxiv.org/pdf/2609.23033)

**💻 Code:** [⭐ Code](https://github.com/summerbro-hhj/wavefront-decoding) • [⭐ Code](https://github.com/huggingface)

> We introduce WaveFront Decoding for Looped Language Models . Check it out!

</details>

<details>
<summary><b>28. Surprising Success, Repeated Failure: Entropy-Guided Credit Assignment for Exploration in LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33781) • [📄 arXiv](https://arxiv.org/abs/2609.33781) • [📥 PDF](https://arxiv.org/pdf/2609.33781)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/wgcyeo/EAPO)

> We introduce EAPO (Entropic Advantage Policy Optimization), an entropy-guided credit-assignment method for exploration in LLM reasoning. It redistributes each response's advantage across tokens, assigning stronger penalties to low-entropy tokens i...

</details>

<details>
<summary><b>29. VGGT-Diff: Visual Geometry Meets Diffusion for Sparse-View Novel View Synthesis</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33253) • [📄 arXiv](https://arxiv.org/abs/2609.33253) • [📥 PDF](https://arxiv.org/pdf/2609.33253)

**💻 Code:** [⭐ Code](https://github.com/chenkangjie1123/VGGT-Diff) • [⭐ Code](https://github.com/huggingface)

> VGGT-Diff is a geometry-routed multi-view diffusion model for sparse-view novel view synthesis from six input images. Its first key innovation, the confidence-aware Visual Geometry Router (VGR), transforms VGGT-Ω features into query-aligned geomet...

</details>

<details>
<summary><b>30. ControlScope: Workflow Revision and Reliability in LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34313) • [📄 arXiv](https://arxiv.org/abs/2609.34313) • [📥 PDF](https://arxiv.org/pdf/2609.34313)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Workflow Revision and Reliability in LLM Agents

</details>

<details>
<summary><b>31. Knowing When Thinking Is Not Enough: Teaching Small Reasoning Models to Reason Beyond Their Parametric Knowledge</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34327) • [📄 arXiv](https://arxiv.org/abs/2609.34327) • [📥 PDF](https://arxiv.org/pdf/2609.34327)

**💻 Code:** [⭐ Code](https://github.com/tally0818/FlyBy) • [⭐ Code](https://github.com/huggingface)

> We study why small reasoning models fail despite extended reasoning, distinguishing execution bottlenecks, where the correct continuation remains internally reachable, from knowledge bottlenecks, where missing parametric knowledge hinders further ...

</details>

<details>
<summary><b>32. KVCMAS: Efficient KV cache Correction for Shared Context in Multi-Agent Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34060) • [📄 arXiv](https://arxiv.org/abs/2609.34060) • [📥 PDF](https://arxiv.org/pdf/2609.34060)

**💻 Code:** [⭐ Code](https://github.com/hjeon2k/KVCMAS) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>33. PReCache: Efficient KV Cache Sharing for Multi-LoRA Agents via Low-Rank Precomputation and Neutral Reconstruction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34054) • [📄 arXiv](https://arxiv.org/abs/2609.34054) • [📥 PDF](https://arxiv.org/pdf/2609.34054)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hjeon2k/PReCache)

> No abstract available.

</details>

<details>
<summary><b>34. AdaGuard: An Adaptive Guard Model with User-defined Policies</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mingrui Lao, Zheng Li, Yuxiang Xie, Yifan Ding, Yunhao Feng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34241) • [📄 arXiv](https://arxiv.org/abs/2609.34241) • [📥 PDF](https://arxiv.org/pdf/2609.34241)

**💻 Code:** [⭐ Code](https://github.com/Yunhao-Feng/AdaGuard) • [⭐ Code](https://github.com/huggingface)

> A adapative guard model of agents.

</details>

<details>
<summary><b>35. Routing Drift Alone Does Not Diagnose Failure in Merged MoE LLMs</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32821) • [📄 arXiv](https://arxiv.org/abs/2609.32821) • [📥 PDF](https://arxiv.org/pdf/2609.32821)

**💻 Code:** [⭐ Code](https://github.com/wyy-code/SRR) • [⭐ Code](https://github.com/huggingface)

> This work conducts a comprehensive analysis across s DeepSeekMoE, OLMoE, and Qwen3-MoE, by proposing a routing analysis toolkit for controlled counterfactual interventions and token-level analysis. These findings show that routing drift alone is i...

</details>

<details>
<summary><b>36. PyroAdapt: Adapting Wildfire Prediction under Spatial Heterogeneity and Temporal Shift</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.12435) • [📄 arXiv](https://arxiv.org/abs/2605.12435) • [📥 PDF](https://arxiv.org/pdf/2605.12435)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Prediction of wildfire occurrence is a rare-event problem compounded by spatial heterogeneity and temporal distribution shift, as fire occurrences are vastly outnumbered by non-occurrences, and predictor--fire relationship varies across space and ...

</details>

<details>
<summary><b>37. Rolling-WAM: World Action Models with Rolling Imagination</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.30247) • [📄 arXiv](https://arxiv.org/abs/2609.30247) • [📥 PDF](https://arxiv.org/pdf/2609.30247)

**💻 Code:** [⭐ Code](https://github.com/zyinghua/Rolling-WAM) • [⭐ Code](https://github.com/huggingface)

> https://rolling-wam.github.io/

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 37 |
| 📅 Today | [`2026-09-29.json`](data/daily/2026-09-29.json) | 37 |
| 📆 This Week | [`2026-W39.json`](data/weekly/2026-W39.json) | 44 |
| 🗓️ This Month | [`2026-09.json`](data/monthly/2026-09.json) | 435 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-09-29 | 37 | [View JSON](data/daily/2026-09-29.json) |
| 📄 2026-09-28 | 7 | [View JSON](data/daily/2026-09-28.json) |
| 📄 2026-09-27 | 22 | [View JSON](data/daily/2026-09-27.json) |
| 📄 2026-09-26 | 22 | [View JSON](data/daily/2026-09-26.json) |
| 📄 2026-09-25 | 13 | [View JSON](data/daily/2026-09-25.json) |
| 📄 2026-09-24 | 10 | [View JSON](data/daily/2026-09-24.json) |
| 📄 2026-09-23 | 10 | [View JSON](data/daily/2026-09-23.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W39 | 44 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |
| 📅 2026-W36 | 88 | [View JSON](data/weekly/2026-W36.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-09 | 435 | [View JSON](data/monthly/2026-09.json) |
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
