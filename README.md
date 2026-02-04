<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-73-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1722+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">73</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">158</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">203</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1722+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 04, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Green-VLA: Staged Vision-Language-Action Model for Generalist Robots</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00919) • [📄 arXiv](https://arxiv.org/abs/2602.00919) • [📥 PDF](https://arxiv.org/pdf/2602.00919)

**💻 Code:** [⭐ Code](https://github.com/greenvla/GreenVLA)

> TL;DR: Scaling VLA isn’t enough—you need quality-aligned trajectories + a unified action interface + staged RL refinement to get reliable cross-robot generalization. This work (1) introduces a unified R64 action space with a fixed semantic layout ...

</details>

<details>
<summary><b>2. Kimi K2.5: Visual Agentic Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02276) • [📄 arXiv](https://arxiv.org/abs/2602.02276) • [📥 PDF](https://arxiv.org/pdf/2602.02276)

> amazing <3

</details>

<details>
<summary><b>3. Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models</b> ⭐ 96</summary>

<br/>

**👥 Authors:** Zhen Fang, Qiuchen Wang, Lin-Chen, YuZeng260, Osilly

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22060) • [📄 arXiv](https://arxiv.org/abs/2601.22060) • [📥 PDF](https://arxiv.org/pdf/2601.22060)

**💻 Code:** [⭐ Code](https://github.com/Osilly/Vision-DeepResearch)

> We present the first long-horizon multimodal deep-research MLLM, introducing multi-turn, multi-entity, and multi-scale visual/textual search, and scaling up to dozens of reasoning steps and hundreds of search-engine interactions. By combining VQA ...

</details>

<details>
<summary><b>4. Vision-DeepResearch Benchmark: Rethinking Visual and Textual Search for Multimodal Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** gaotiexinqu, chocckaka, Lin-Chen, Osilly, YuZeng260

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02185) • [📄 arXiv](https://arxiv.org/abs/2602.02185) • [📥 PDF](https://arxiv.org/pdf/2602.02185)

> We introduce the Vision-DeepResearch Benchmark (VDR-Bench) to address two key limitations of existing multimodal deep-research benchmarks: (1) they are not visual-search-centric, allowing many instances to be solved without genuine visual retrieva...

</details>

<details>
<summary><b>5. Closing the Loop: Universal Repository Representation with RPG-Encoder</b> ⭐ 59</summary>

<br/>

**👥 Authors:** Steven Liu, Qingtao Li, Xin Zhang, Cipherxzc, Luo2003

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02084) • [📄 arXiv](https://arxiv.org/abs/2602.02084) • [📥 PDF](https://arxiv.org/pdf/2602.02084)

**💻 Code:** [⭐ Code](https://github.com/microsoft/RPG-ZeroRepo)

> Current repository agents encounter a reasoning disconnect due to fragmented representations, as existing methods rely on isolated API documentation or dependency graphs that lack semantic depth. We consider repository comprehension and generation...

</details>

<details>
<summary><b>6. UniReason 1.0: A Unified Reasoning Framework for World Knowledge Aligned Image Generation and Editing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Size Wu, Feng Han, Chaofan Ma, Dianyi Wang, CodeGoat24

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02437) • [📄 arXiv](https://arxiv.org/abs/2602.02437) • [📥 PDF](https://arxiv.org/pdf/2602.02437)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Unified Thinker: A General Reasoning Modular Core for Image Generation (202...

</details>

<details>
<summary><b>7. WildGraphBench: Benchmarking GraphRAG with Wild-Source Corpora</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02053) • [📄 arXiv](https://arxiv.org/abs/2602.02053) • [📥 PDF](https://arxiv.org/pdf/2602.02053)

**💻 Code:** [⭐ Code](https://github.com/BstWPY/WildGraphBench)

> hi

</details>

<details>
<summary><b>8. FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01566) • [📄 arXiv](https://arxiv.org/abs/2602.01566) • [📥 PDF](https://arxiv.org/pdf/2602.01566)

> A deep research agent with file system as the scaling substrate, allowing external and persistent context. Although achieving maximal performance on downstream tasks still requires a lot of task-specific design at this stage, we believe that a fil...

</details>

<details>
<summary><b>9. SWE-Universe: Scale Real-World Verifiable Environments to Millions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02361) • [📄 arXiv](https://arxiv.org/abs/2602.02361) • [📥 PDF](https://arxiv.org/pdf/2602.02361)

> We propose SWE-Universe, a scalable and efficient framework for automatically constructing real-world software engineering (SWE) verifiable environments from GitHub pull requests (PRs). Using this method, we scale the number of real-world multilin...

</details>

<details>
<summary><b>10. Wiki Live Challenge: Challenging Deep Research Agents with Expert-Level Wikipedia Articles</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01590) • [📄 arXiv](https://arxiv.org/abs/2602.01590) • [📥 PDF](https://arxiv.org/pdf/2602.01590)

**💻 Code:** [⭐ Code](https://github.com/WangShao2000/Wiki_Live_Challenge)

> Hi everyone, we have released the Wiki Live Challenge, a benchmark that uses Wikipedia Good Articles as a high-level human baseline. It is designed to evaluate the writing quality and information-gathering capabilities of Deep Research Agents in a...

</details>

<details>
<summary><b>11. PixelGen: Pixel Diffusion Beats Latent Diffusion with Perceptual Loss</b> ⭐ 51</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02493) • [📄 arXiv](https://arxiv.org/abs/2602.02493) • [📥 PDF](https://arxiv.org/pdf/2602.02493)

**💻 Code:** [⭐ Code](https://github.com/Zehong-Ma/PixelGen)

> Pixel diffusion generates images directly in pixel space in an end-to-end manner, avoiding the artifacts and bottlenecks introduced by VAEs in two-stage latent diffusion. However, it is challenging to optimize high-dimensional pixel manifolds that...

</details>

<details>
<summary><b>12. Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01058) • [📄 arXiv](https://arxiv.org/abs/2602.01058) • [📥 PDF](https://arxiv.org/pdf/2602.01058)

> A good objective for supervised post-training is commonly taken as one that optimizes for performance after supervised stage. But when this supervised stage is followed by an online RL stage,  SFT stage gains may not be preserved after online RL ....

</details>

<details>
<summary><b>13. RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System</b> ⭐ 193</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02488) • [📄 arXiv](https://arxiv.org/abs/2602.02488) • [📥 PDF](https://arxiv.org/pdf/2602.02488)

**💻 Code:** [⭐ Code](https://github.com/Gen-Verse/Open-AgentRL)

> Code and model: https://github.com/Gen-Verse/Open-AgentRL

</details>

<details>
<summary><b>14. SLIME: Stabilized Likelihood Implicit Margin Enforcement for Preference Optimization</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02383) • [📄 arXiv](https://arxiv.org/abs/2602.02383) • [📥 PDF](https://arxiv.org/pdf/2602.02383)

**💻 Code:** [⭐ Code](https://github.com/fpsigma/trl-slime)

> We introduce SLIME, a reference-free preference optimization objective designed to decouple preference learning from generation quality. Our approach uses a three-pronged objective: Likelihood Anchoring: An explicit term to maximize the likelihood...

</details>

<details>
<summary><b>15. PISCES: Annotation-free Text-to-Video Post-Training via Optimal Transport-Aligned Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01624) • [📄 arXiv](https://arxiv.org/abs/2602.01624) • [📥 PDF](https://arxiv.org/pdf/2602.01624)

> PISCES is an annotation-free post-training framework for text-to-video models. We tackle a key bottleneck in VLM-based rewards, text/video embedding misalignment, by using Optimal Transport (OT) to align text embeddings to the video space. This yi...

</details>

<details>
<summary><b>16. Mind-Brush: Integrating Agentic Cognitive Search and Reasoning into Image Generation</b> ⭐ 36</summary>

<br/>

**👥 Authors:** Chenjue Zhang, Dongzhi Jiang, Junyan Ye, Jun He, SereinH

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01756) • [📄 arXiv](https://arxiv.org/abs/2602.01756) • [📥 PDF](https://arxiv.org/pdf/2602.01756)

**💻 Code:** [⭐ Code](https://github.com/PicoTrex/Mind-Brush)

> 🧠 Mind-Brush Framework: A novel agentic paradigm that unifies Intent Analysis, Multi-modal Search, and Knowledge Reasoning into a seamless "Think-Research-Create" workflow for image generation. 📊 Mind-Bench: A specialized benchmark designed to eva...

</details>

<details>
<summary><b>17. Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation</b> ⭐ 164</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02214) • [📄 arXiv](https://arxiv.org/abs/2602.02214) • [📥 PDF](https://arxiv.org/pdf/2602.02214)

**💻 Code:** [⭐ Code](https://github.com/thu-ml/Causal-Forcing)

> Causal Forcing exposes a mathematical fallacy in Self Forcing and significantly outperforms it in both visual quality and motion dynamics , while maintaining the same training budget and inference efficiency , enabling real time streaming video ge...

</details>

<details>
<summary><b>18. Fast Autoregressive Video Diffusion and World Models with Temporal Cache Compression and Sparse Attention</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Gal Chechik, Micahel Green, Matan Levy, Issar Tzachor, Dvir Samuel

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01801) • [📄 arXiv](https://arxiv.org/abs/2602.01801) • [📥 PDF](https://arxiv.org/pdf/2602.01801)

> Project Page: https://dvirsamuel.github.io/fast-auto-regressive-video/

</details>

<details>
<summary><b>19. Rethinking Selective Knowledge Distillation</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01395) • [📄 arXiv](https://arxiv.org/abs/2602.01395) • [📥 PDF](https://arxiv.org/pdf/2602.01395)

**💻 Code:** [⭐ Code](https://github.com/almogtavor/SE-KD3x)

> For the GitHub repo: https://github.com/almogtavor/SE-KD3x

</details>

<details>
<summary><b>20. Generative Visual Code Mobile World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01576) • [📄 arXiv](https://arxiv.org/abs/2602.01576) • [📥 PDF](https://arxiv.org/pdf/2602.01576)

> Coming soon: Website, demo video, github, and eval dataset

</details>

<details>
<summary><b>21. FSVideo: Fast Speed Video Diffusion Model in a Highly-Compressed Latent Space</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02092) • [📄 arXiv](https://arxiv.org/abs/2602.02092) • [📥 PDF](https://arxiv.org/pdf/2602.02092)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API S2DiT: Sandwich Diffusion Transformer for Mobile Streaming Video Generation...

</details>

<details>
<summary><b>22. Toward Cognitive Supersensing in Multimodal Large Language Model</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01541) • [📄 arXiv](https://arxiv.org/abs/2602.01541) • [📥 PDF](https://arxiv.org/pdf/2602.01541)

**💻 Code:** [⭐ Code](https://github.com/PediaMedAI/Cognition-MLLM)

> Multimodal Large Language Models (MLLMs) have achieved remarkable success in open-vocabulary perceptual tasks, yet their ability to solve complex cognitive problems remains limited, especially when visual details are abstract and require visual me...

</details>

<details>
<summary><b>23. Ebisu: Benchmarking Large Language Models in Japanese Finance</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01479) • [📄 arXiv](https://arxiv.org/abs/2602.01479) • [📥 PDF](https://arxiv.org/pdf/2602.01479)

> Japanese finance combines agglutinative, head-final linguistic structure, mixed writing systems, and high-context communication norms that rely on indirect expression and implicit commitment, posing a substantial challenge for LLMs. We introduce E...

</details>

<details>
<summary><b>24. Beyond Pixels: Visual Metaphor Transfer via Schema-Driven Agentic Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01335) • [📄 arXiv](https://arxiv.org/abs/2602.01335) • [📥 PDF](https://arxiv.org/pdf/2602.01335)

> This paper introduces Visual Metaphor Transfer (VMT), a new task that goes beyond pixel-level editing to model abstract, cross-domain creative logic in visual generation. Inspired by Conceptual Blending Theory, the authors propose a schema-based, ...

</details>

<details>
<summary><b>25. How Well Do Models Follow Visual Instructions? VIBE: A Systematic Benchmark for Visual Instruction-Driven Image Editing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haochen Tian, Chen Liang, Chengzu Li, Xuehai Bai, Huanyu Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01851) • [📄 arXiv](https://arxiv.org/abs/2602.01851) • [📥 PDF](https://arxiv.org/pdf/2602.01851)

**💻 Code:** [⭐ Code](https://github.com/hwanyu112/VIBE-Benchmark)

> 🚀 Introducing VIBE: The Visual Instruction Benchmark for Image Editing! Why limit image editing to text? Human intent is multimodal. We’re filling the gap with VIBE , a new benchmark designed to evaluate how models follow visual instructions. Chec...

</details>

<details>
<summary><b>26. Making Avatars Interact: Towards Text-Driven Human-Object Interaction for Controllable Talking Avatars</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Teng Hu, Ziyao Huang, Zhentao Yu, Zhengguang Zhou, youliang1233214

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01538) • [📄 arXiv](https://arxiv.org/abs/2602.01538) • [📥 PDF](https://arxiv.org/pdf/2602.01538)

**💻 Code:** [⭐ Code](https://github.com/angzong/InteractAvatar)

> link: https://arxiv.org/abs/2602.01538

</details>

<details>
<summary><b>27. RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02486) • [📄 arXiv](https://arxiv.org/abs/2602.02486) • [📥 PDF](https://arxiv.org/pdf/2602.02486)

**💻 Code:** [⭐ Code](https://github.com/microsoft/InfoAgent)

> We proposed RE-TRAC, a recursive framework addresses the inefficiency of disjointed traditional agent search by compressing historical trajectories to guide subsequent steps. Experiments demonstrate that this  explicit guidance mechanism not only ...

</details>

<details>
<summary><b>28. SPARKLING: Balancing Signal Preservation and Symmetry Breaking for Width-Progressive Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02472) • [📄 arXiv](https://arxiv.org/abs/2602.02472) • [📥 PDF](https://arxiv.org/pdf/2602.02472)

> Progressive Learning (PL) reduces pre-training computational overhead by gradually increasing model scale. While prior work has extensively explored depth expansion, width expansion remains significantly understudied, with the few existing methods...

</details>

<details>
<summary><b>29. Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics</b> ⭐ 2.71k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02343) • [📄 arXiv](https://arxiv.org/abs/2508.11290) • [📥 PDF](https://arxiv.org/pdf/2602.02343)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/EasyEdit/blob/main/examples/SPLIT.md)

> We unify LLM control methods as dynamic weight updates, analyze their trade-offs between preference (targeted behavior) and utility (task-valid generation) via a shared log-odds framework, explain these effects through activation manifolds, and in...

</details>

<details>
<summary><b>30. Show, Don't Tell: Morphing Latent Reasoning into Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02227) • [📄 arXiv](https://arxiv.org/abs/2602.02227) • [📥 PDF](https://arxiv.org/pdf/2602.02227)

**💻 Code:** [⭐ Code](https://github.com/EnVision-Research/LatentMorph)

> Code: https://github.com/EnVision-Research/LatentMorph

</details>

<details>
<summary><b>31. CUA-Skill: Develop Skills for Computer Using Agent</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21123) • [📄 arXiv](https://arxiv.org/abs/2601.21123) • [📥 PDF](https://arxiv.org/pdf/2601.21123)

**💻 Code:** [⭐ Code](https://github.com/microsoft/cua_skill)

> Computer-Using Agents (CUAs) aim to autonomously operate computer systems to complete real-world tasks. However, existing agentic systems remain difficult to scale and lag behind human performance. A key limitation is the absence of reusable and s...

</details>

<details>
<summary><b>32. LoopViT: Scaling Visual ARC with Looped Transformers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yexin Liu, Rui-Jie Zhu, Wen-Jie Shu, Harold328, Xuerui123

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02156) • [📄 arXiv](https://arxiv.org/abs/2602.02156) • [📥 PDF](https://arxiv.org/pdf/2602.02156)

**💻 Code:** [⭐ Code](https://github.com/WenjieShu/LoopViT)

> Code: https://github.com/WenjieShu/LoopViT

</details>

<details>
<summary><b>33. Thinking with Comics: Enhancing Multimodal Reasoning through Structured Visual Storytelling</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Muyun Yang, Yuchen Song, Qiuyu Ding, Wenxin Zhu, AndongChen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02453) • [📄 arXiv](https://arxiv.org/abs/2602.02453) • [📥 PDF](https://arxiv.org/pdf/2602.02453)

**💻 Code:** [⭐ Code](https://github.com/andongBlue/Think-with-Comics)

> No abstract available.

</details>

<details>
<summary><b>34. PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mihalis Nicolaou, Yannis Panagakis, James Oldfield, Andreas D. Demou, Panagiotis Koromilas

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01322) • [📄 arXiv](https://arxiv.org/abs/2602.01322) • [📥 PDF](https://arxiv.org/pdf/2602.01322)

> PolySAE: Modeling Feature Interactions via Polynomial Decoding What: We generalize Sparse Autoencoders to capture feature interactions via polynomial decoding while preserving linear, interpretable encodings. Why: Standard SAEs assume features com...

</details>

<details>
<summary><b>35. Sparse Reward Subsystem in Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00986) • [📄 arXiv](https://arxiv.org/abs/2602.00986) • [📥 PDF](https://arxiv.org/pdf/2602.00986)

> In this paper, we identify a sparse reward subsystem within the hidden states of Large Language Models (LLMs), drawing an analogy to the biological reward subsystem in the human brain.

</details>

<details>
<summary><b>36. TRIP-Bench: A Benchmark for Long-Horizon Interactive Agents in Real-World Scenarios</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01675) • [📄 arXiv](https://arxiv.org/abs/2602.01675) • [📥 PDF](https://arxiv.org/pdf/2602.01675)

> We are motivated by the gap between existing LLM-agent benchmarks and real deployment needs, where agents must handle long, multi-turn interactions, satisfy global constraints, and coordinate tools under frequent user revisions. We introduce TRIP-...

</details>

<details>
<summary><b>37. Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01511) • [📄 arXiv](https://arxiv.org/abs/2602.01511) • [📥 PDF](https://arxiv.org/pdf/2602.01511)

> Standard reward models typically predict scalar scores that fail to capture the multifaceted nature of response quality in non-verifiable domains, such as creative writing or open-ended instruction following. To address this limitation, we propose...

</details>

<details>
<summary><b>38. AgentIF-OneDay: A Task-level Instruction-Following Benchmark for General AI Agents in Daily Scenarios</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tianhao Tang, Taiyu Hou, Qimin Wu, Kaiyuan Chen, YuanshuoZhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20613) • [📄 arXiv](https://arxiv.org/abs/2601.20613) • [📥 PDF](https://arxiv.org/pdf/2601.20613)

> The capacity of AI agents to effectively handle tasks of increasing duration and complexity continues to grow, demonstrating exceptional performance in coding, deep research, and complex problem-solving evaluations. However, in daily scenarios, th...

</details>

<details>
<summary><b>39. CoDiQ: Test-Time Scaling for Controllable Difficult Question Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01660) • [📄 arXiv](https://arxiv.org/abs/2602.01660) • [📥 PDF](https://arxiv.org/pdf/2602.01660)

**💻 Code:** [⭐ Code](https://github.com/ALEX-nlp/CoDiQ)

> Large Reasoning Models (LRMs) benefit substantially from training on challenging competition-level questions. However, existing automated question synthesis methods lack precise difficulty control, incur high computational costs, and struggle to g...

</details>

<details>
<summary><b>40. VoxServe: Streaming-Centric Serving System for Speech Language Models</b> ⭐ 29</summary>

<br/>

**👥 Authors:** Stephanie Wang, Rohan Kadekodi, Atindra Jha, Wei-Tzu Lee, Keisuke Kamahori

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00269) • [📄 arXiv](https://arxiv.org/abs/2602.00269) • [📥 PDF](https://arxiv.org/pdf/2602.00269)

**💻 Code:** [⭐ Code](https://github.com/vox-serve/vox-serve)

> A serving system for SpeechLMs https://github.com/vox-serve/vox-serve

</details>

<details>
<summary><b>41. Training LLMs for Divide-and-Conquer Reasoning Elevates Test-Time Scalability</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02477) • [📄 arXiv](https://arxiv.org/abs/2602.02477) • [📥 PDF](https://arxiv.org/pdf/2602.02477)

**💻 Code:** [⭐ Code](https://github.com/MasterVito/DAC-RL)

> We introduce an end-to-end RL framework to endow LLMs with divide-and-conquer reasoning capabilities, enabling a higher performance ceiling and stronger test-time scalability.

</details>

<details>
<summary><b>42. Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02338) • [📄 arXiv](https://arxiv.org/abs/2602.02338) • [📥 PDF](https://arxiv.org/pdf/2602.02338)

> No abstract available.

</details>

<details>
<summary><b>43. PromptRL: Prompt Matters in RL for Flow-Based Image Generation</b> ⭐ 59</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01382) • [📄 arXiv](https://arxiv.org/abs/2602.01382) • [📥 PDF](https://arxiv.org/pdf/2602.01382)

**💻 Code:** [⭐ Code](https://github.com/G-U-N/UniRL)

> Image Editing Text-to-Image Generation

</details>

<details>
<summary><b>44. Adaptive Ability Decomposing for Unlocking Large Reasoning Model Effective Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00759) • [📄 arXiv](https://arxiv.org/abs/2602.00759) • [📥 PDF](https://arxiv.org/pdf/2602.00759)

> This is a new work by Renmin University of China and ByteDance Seed. We introduce a novel RLVR algorithm that allows a single base model to evolve into two complementary models i.e., Decomposer and Reasoner, which can mutually reinforce each other...

</details>

<details>
<summary><b>45. An Empirical Study of World Model Quantization</b> ⭐ 930</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02110) • [📄 arXiv](https://arxiv.org/abs/2602.02110) • [📥 PDF](https://arxiv.org/pdf/2602.02110)

**💻 Code:** [⭐ Code](https://github.com/huawei-noah/noah-research/tree/master/QuantWM)

> World models learn an internal representation of environment dynamics, enabling agents to simulate and reason about future states within a compact latent space for tasks such as planning, prediction, and inference. However, running world models re...

</details>

<details>
<summary><b>46. Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02039) • [📄 arXiv](https://arxiv.org/abs/2602.02039) • [📥 PDF](https://arxiv.org/pdf/2602.02039)

**💻 Code:** [⭐ Code](https://github.com/thinkwee/DDR_Bench)

> This paper introduces Deep Data Research , shifting from executional intelligence , which focuses on completing assigned tasks, to investigatory intelligence , where agents autonomously set goals and explore. Under this paradigm, Agentic LLMs are ...

</details>

<details>
<summary><b>47. Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Yiming Zeng, Yuelyu Ji, Ming Li, Zhuochun Li, ReRaWo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22588) • [📄 arXiv](https://arxiv.org/abs/2601.22588) • [📥 PDF](https://arxiv.org/pdf/2601.22588)

**💻 Code:** [⭐ Code](https://github.com/zhuochunli/Representation-as-a-judge)

> The paper motivates a paradigm shift from LLM-as-a-Judge to Representation-as-a-Judge, a decoding-free evaluation strategy that probes internal model structure rather than relying on prompted output.

</details>

<details>
<summary><b>48. Enhancing Multi-Image Understanding through Delimiter Token Scaling</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Seong Joon Oh, Yejin Kim, Dongjun Hwang, Yeji Park, MYMY-young

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01984) • [📄 arXiv](https://arxiv.org/abs/2602.01984) • [📥 PDF](https://arxiv.org/pdf/2602.01984)

**💻 Code:** [⭐ Code](https://github.com/MYMY-young/DelimScaling)

> Large Vision-Language Models (LVLMs) achieve strong performance on single-image tasks, but their performance declines when multiple images are provided as input. One major reason is the cross-image information leakage, where the model struggles to...

</details>

<details>
<summary><b>49. Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Qingyu Shi, Yi Xin, Yuchen Zhu, Yixuan Li, BryanW

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01842) • [📄 arXiv](https://arxiv.org/abs/2602.01842) • [📥 PDF](https://arxiv.org/pdf/2602.01842)

**💻 Code:** [⭐ Code](https://github.com/viiika/Prism)

> Inference-time compute has re-emerged as a practical way to improve LLM reasoning. Most test-time scaling (TTS) algorithms rely on autoregressive decoding, which is ill-suited to discrete diffusion language models (dLLMs) due to their parallel dec...

</details>

<details>
<summary><b>50. Interacted Planes Reveal 3D Line Mapping</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01296) • [📄 arXiv](https://arxiv.org/abs/2602.01296) • [📥 PDF](https://arxiv.org/pdf/2602.01296)

**💻 Code:** [⭐ Code](https://github.com/calmke/LiPMAP)

> Structured 3D reconstruction

</details>

<details>
<summary><b>51. PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01077) • [📄 arXiv](https://arxiv.org/abs/2602.01077) • [📥 PDF](https://arxiv.org/pdf/2602.01077)

**💻 Code:** [⭐ Code](https://github.com/xie-lab-ml/piecewise-sparse-attention)

> Code: https://github.com/xie-lab-ml/piecewise-sparse-attention

</details>

<details>
<summary><b>52. VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22674) • [📄 arXiv](https://arxiv.org/abs/2601.22674) • [📥 PDF](https://arxiv.org/pdf/2601.22674)

> An efficient vision token compression framework with two modules, Dominant Vision Token Selection (DVTS) and Text-Guided Vision Complement (TGVC).

</details>

<details>
<summary><b>53. On the Relationship Between Representation Geometry and Generalization in Deep Neural Networks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** rockerritesh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00130) • [📄 arXiv](https://arxiv.org/abs/2505.08727) • [📥 PDF](https://arxiv.org/pdf/2602.00130)

> On the Relationship Between Representation Geometry and Generalization in Deep Neural Networks.

</details>

<details>
<summary><b>54. On the Limits of Layer Pruning for Generative Reasoning in LLMs</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01997) • [📄 arXiv](https://arxiv.org/abs/2602.01997) • [📥 PDF](https://arxiv.org/pdf/2602.01997)

**💻 Code:** [⭐ Code](https://github.com/safal312/on-the-limits-of-layer-pruning)

> TLDR; Layer pruning compresses LLMs with little impact on classification, but severely degrades generative reasoning by disrupting core algorithmic abilities like arithmetic and syntax. Self-generated supervision improves recovery, yet explains cl...

</details>

<details>
<summary><b>55. Evolving from Tool User to Creator via Training-Free Experience Reuse in Multimodal Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01983) • [📄 arXiv](https://arxiv.org/abs/2602.01983) • [📥 PDF](https://arxiv.org/pdf/2602.01983)

> This paper introduces UCT, a training-free framework that enables LLM agents to evolve during inference by transforming reasoning experience into reusable tools. Unlike prior tool-augmented methods that rely on fixed or single-use tools, UCT allow...

</details>

<details>
<summary><b>56. Mano: Restriking Manifold Optimization for LLM Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23000) • [📄 arXiv](https://arxiv.org/abs/2601.23000) • [📥 PDF](https://arxiv.org/pdf/2601.23000)

**💻 Code:** [⭐ Code](https://github.com/xie-lab-ml/Mano-Restriking-Manifold-Optimization-for-LLM-Training)

> Code: https://github.com/xie-lab-ml/Mano-Restriking-Manifold-Optimization-for-LLM-Training

</details>

<details>
<summary><b>57. Implicit neural representation of textures</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Dounia Hammou, Albert Kwok, Peter2023HuggingFace

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02354) • [📄 arXiv](https://arxiv.org/abs/2602.02354) • [📥 PDF](https://arxiv.org/pdf/2602.02354)

**💻 Code:** [⭐ Code](https://github.com/PeterHUistyping/INR-Tex)

> Implicit neural representation of textures @ misc {KH2026INR-Tex,
      title={Implicit neural representation of textures}, 
      author={Albert Kwok and Zheyuan Hu and Dounia Hammou},
      year={2026},
      eprint={2602.02354},
      archivePr...

</details>

<details>
<summary><b>58. Cross-Lingual Stability of LLM Judges Under Controlled Generation: Evidence from Finno-Ugric Languages</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Linda Freienthal, Isaac Chung

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02287) • [📄 arXiv](https://arxiv.org/abs/2602.02287) • [📥 PDF](https://arxiv.org/pdf/2602.02287)

**💻 Code:** [⭐ Code](https://github.com/isaac-chung/cross-lingual-stability-judges) • [⭐ Code](https://github.com/isaac-chung/)

> Cross-lingual evaluation of large language models (LLMs) typically conflates two sources of variance: genuine model performance differences and measurement instability. We investigate evaluation reliability by holding generation conditions constan...

</details>

<details>
<summary><b>59. Small Generalizable Prompt Predictive Models Can Steer Efficient RL Post-Training of Large Reasoning Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuhang Jiang, Heming Zou, Yixiu Mao, Qi Wang, yunqu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01970) • [📄 arXiv](https://arxiv.org/abs/2602.01970) • [📥 PDF](https://arxiv.org/pdf/2602.01970)

> This study introduces Generalizable Predictive Prompt Selection (GPS), which performs Bayesian inference towards prompt difficulty using a lightweight generative model trained on the shared optimization history.

</details>

<details>
<summary><b>60. Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bugeun Kim, Hyeonchu Park, Chanhee Cho, Sohhyung Park, Junhyuk Choi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00521) • [📄 arXiv](https://arxiv.org/abs/2602.00521) • [📥 PDF](https://arxiv.org/pdf/2602.00521)

> ,

</details>

<details>
<summary><b>61. Clipping-Free Policy Optimization for Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xuandong Zhao, Gözde Gül Şahin, Barış Akgün, Ömer Veysel Çağatan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22801) • [📄 arXiv](https://arxiv.org/abs/2601.22801) • [📥 PDF](https://arxiv.org/pdf/2601.22801)

> Clipping-Free Policy Optimization for Large Language Models

</details>

<details>
<summary><b>62. AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Adrian Popescu, Elif Nebioglu, emirhanbilgic

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00192) • [📄 arXiv](https://arxiv.org/abs/2602.00192) • [📥 PDF](https://arxiv.org/pdf/2602.00192)

> Key takeaway: 𝐂𝐮𝐫𝐫𝐞𝐧𝐭 𝐛𝐞𝐧𝐜𝐡𝐦𝐚𝐫𝐤𝐬 for AI-generated Image Detection, 𝐜𝐚𝐧 𝐝𝐫𝐚𝐦𝐚𝐭𝐢𝐜𝐚𝐥𝐥𝐲 𝐨𝐯𝐞𝐫𝐞𝐬𝐭𝐢𝐦𝐚𝐭𝐞 𝐫𝐨𝐛𝐮𝐬𝐭𝐧𝐞𝐬𝐬. 𝐓𝐫𝐚𝐢𝐧𝐢𝐧𝐠 𝐰𝐢𝐭𝐡 𝐈𝐍𝐏-𝐗 𝐟𝐨𝐫𝐜𝐞𝐬 𝐝𝐞𝐭𝐞𝐜𝐭𝐨𝐫𝐬 𝐭𝐨 𝐟𝐨𝐜𝐮𝐬 𝐨𝐧 𝐠𝐞𝐧𝐞𝐫𝐚𝐭𝐞𝐝 𝐜𝐨𝐧𝐭𝐞𝐧𝐭, 𝐢𝐦𝐩𝐫𝐨𝐯𝐢𝐧𝐠 𝐠𝐞𝐧𝐞𝐫𝐚𝐥𝐢𝐳𝐚𝐭𝐢𝐨𝐧 𝐚𝐧𝐝 𝐥𝐨𝐜𝐚𝐥𝐢𝐳𝐚𝐭𝐢𝐨𝐧.

</details>

<details>
<summary><b>63. A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22599) • [📄 arXiv](https://arxiv.org/abs/2601.22599) • [📥 PDF](https://arxiv.org/pdf/2601.22599)

**💻 Code:** [⭐ Code](https://github.com/ShandaAI/Hive)

> Query-based universal sound separation is a cornerstone capability for intelligent auditory systems, yet progress is often hindered by a data bottleneck: in-the-wild datasets typically come with weak labels and heavy event co-occurrence, encouragi...

</details>

<details>
<summary><b>64. ParalESN: Enabling parallel information processing in Reservoir Computing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Claudio Gallicchio, Andrea Ceni, Giacomo Lagomarsini, nennomp

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22296) • [📄 arXiv](https://arxiv.org/abs/2601.22296) • [📥 PDF](https://arxiv.org/pdf/2601.22296)

> TL;DR: We revisit the Reservoir Computing paradigm through the lens of structured operators and state space modelling, introducing Parallel Echo State Networks (ParalESN). ParalESN enables the construction of high-dimensional, efficient, and paral...

</details>

<details>
<summary><b>65. OVD: On-policy Verbal Distillation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jianghan Shen, Yuxin Cheng, Shansan Gong, Hui Shen, Jing Xiong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21968) • [📄 arXiv](https://arxiv.org/abs/2601.21968) • [📥 PDF](https://arxiv.org/pdf/2601.21968)

> Knowledge distillation offers a promising path to transfer reasoning capabilities from large teacher models to efficient student models; however, existing token-level on-policy distillation methods require token-level alignment between the student...

</details>

<details>
<summary><b>66. Influence Guided Sampling for Domain Adaptation of Text Retrievers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jaydeep Sen, Yulong Li, vishwajeetkumar, meetdoshi90

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21759) • [📄 arXiv](https://arxiv.org/abs/2601.21759) • [📥 PDF](https://arxiv.org/pdf/2601.21759)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API GradientSpace: Unsupervised Data Clustering for Improved Instruction Tuning...

</details>

<details>
<summary><b>67. Competing Visions of Ethical AI: A Case Study of OpenAI</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Madelyn Rose Sanfilippo, Mengting Ai, Melissa Wilfley

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16513) • [📄 arXiv](https://arxiv.org/abs/2601.16513) • [📥 PDF](https://arxiv.org/pdf/2601.16513)

> AI Ethics is framed distinctly across actors and stakeholder groups. We report results from a case study of OpenAI analysing ethical AI discourse. Research addressed: How has OpenAI's public discourse leveraged 'ethics', 'safety', 'alignment', and...

</details>

<details>
<summary><b>68. Internal Flow Signatures for Self-Checking and Refinement in LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01897) • [📄 arXiv](https://arxiv.org/abs/2602.01897) • [📥 PDF](https://arxiv.org/pdf/2602.01897)

**💻 Code:** [⭐ Code](https://github.com/EavnJeong/Internal-Flow-Signatures-for-Self-Checking-and-Refinement-in-LLMs)

> This repository implements Internal Flow Signatures, a training-free method for auditing and refining LLM decisions by analyzing depthwise hidden-state dynamics. The approach enables lightweight self-checking and targeted refinement without modify...

</details>

<details>
<summary><b>69. INDIBATOR: Diverse and Fact-Grounded Individuality for Multi-Agent Debate in Molecular Discovery</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sungsoo Ahn, Jaehyung Kim, Seonghyun Park, Yunhui Jang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01815) • [📄 arXiv](https://arxiv.org/abs/2602.01815) • [📥 PDF](https://arxiv.org/pdf/2602.01815)

> This paper suggests constructing agent persona based on the research trajectory instead of static role-based prompting or keywords. This enhances the individuality of each agent, which guarantees high diversity and fact-grounding agents.

</details>

<details>
<summary><b>70. SEA-Guard: Culturally Grounded Multilingual Safeguard for Southeast Asia</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01618) • [📄 arXiv](https://arxiv.org/abs/2602.01618) • [📥 PDF](https://arxiv.org/pdf/2602.01618)

> Model: https://huggingface.co/collections/aisingapore/sea-guard

</details>

<details>
<summary><b>71. Where to Attend: A Principled Vision-Centric Position Encoding with Parabolas</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01418) • [📄 arXiv](https://arxiv.org/abs/2602.01418) • [📥 PDF](https://arxiv.org/pdf/2602.01418)

**💻 Code:** [⭐ Code](https://github.com/DTU-PAS/parabolic-position-encoding)

> Parabolic Position Encoding (PaPE) We propose a position encoding that is designed from the ground up for vision modalities. It works by treating relative positions as the dependent variable in a sum of parabolas. PaPE is the highest scoring posit...

</details>

<details>
<summary><b>72. YOLOE-26: Integrating YOLO26 with YOLOE for Real-Time Open-Vocabulary Instance Segmentation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00168) • [📄 arXiv](https://arxiv.org/abs/2602.00168) • [📥 PDF](https://arxiv.org/pdf/2602.00168)

> This paper presents YOLOE-26, a unified framework that integrates the deployment-optimized YOLO26(or YOLOv26) architecture with the open-vocabulary learning paradigm of YOLOE for real-time open-vocabulary instance segmentation. Building on the NMS...

</details>

<details>
<summary><b>73. Gaming the Judge: Unfaithful Chain-of-Thought Can Undermine Agent Evaluation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14691) • [📄 arXiv](https://arxiv.org/abs/2601.14691) • [📥 PDF](https://arxiv.org/pdf/2601.14691)

> TL;DR: The assumption that an agent's Chain-of-Thought (CoT) faithfully reflects its internal reasoning and environment state is brittle, which can reflect badly on the reliability of LLM judges that use the agent reasoning for evaluation. Our key...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 73 |
| 📅 Today | [`2026-02-04.json`](data/daily/2026-02-04.json) | 73 |
| 📆 This Week | [`2026-W05.json`](data/weekly/2026-W05.json) | 158 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 203 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-04 | 73 | [View JSON](data/daily/2026-02-04.json) |
| 📄 2026-02-03 | 40 | [View JSON](data/daily/2026-02-03.json) |
| 📄 2026-02-02 | 45 | [View JSON](data/daily/2026-02-02.json) |
| 📄 2026-02-01 | 45 | [View JSON](data/daily/2026-02-01.json) |
| 📄 2026-01-31 | 45 | [View JSON](data/daily/2026-01-31.json) |
| 📄 2026-01-30 | 21 | [View JSON](data/daily/2026-01-30.json) |
| 📄 2026-01-29 | 21 | [View JSON](data/daily/2026-01-29.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W05 | 158 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 203 | [View JSON](data/monthly/2026-02.json) |
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
