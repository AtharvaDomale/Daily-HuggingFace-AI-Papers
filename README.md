<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-41-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2727+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">41</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">132</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">160</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2727+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 05, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Utonia: Toward One Encoder for All Point Clouds</b> ⭐ 180</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03283) • [📄 arXiv](https://arxiv.org/abs/2603.03283) • [📥 PDF](https://arxiv.org/pdf/2603.03283)

**💻 Code:** [⭐ Code](https://github.com/Pointcept/Utonia)

> Page: https://pointcept.github.io/Utonia Code: https://github.com/Pointcept/Utonia Demo: https://huggingface.co/spaces/pointcept-bot/Utonia Weight: https://huggingface.co/Pointcept/Utonia

</details>

<details>
<summary><b>2. UniG2U-Bench: Do Unified Models Advance Multimodal Understanding?</b> ⭐ 19</summary>

<br/>

**👥 Authors:** Xiaoyu Chen, Junxiang Lei, Wanbo Zhang, Boxiu Li, Zimo Wen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03241) • [📄 arXiv](https://arxiv.org/abs/2603.03241) • [📥 PDF](https://arxiv.org/pdf/2603.03241)

**💻 Code:** [⭐ Code](https://github.com/nssmd/UniG2U)

> Unified multimodal models have recently demonstrated strong generative capabilities, yet whether and when generation improves understanding remains unclear. Existing benchmarks lack a systematic exploration of the specific tasks where generation f...

</details>

<details>
<summary><b>3. Beyond Language Modeling: An Exploration of Multimodal Pretraining</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03276) • [📄 arXiv](https://arxiv.org/abs/2603.03276) • [📥 PDF](https://arxiv.org/pdf/2603.03276)

> Lets train beyond language

</details>

<details>
<summary><b>4. BeyondSWE: Can Current Code Agent Survive Beyond Single-Repo Bug Fixing?</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03194) • [📄 arXiv](https://arxiv.org/abs/2603.03194) • [📥 PDF](https://arxiv.org/pdf/2603.03194)

**💻 Code:** [⭐ Code](https://github.com/AweAI-Team/AweAgent) • [⭐ Code](https://github.com/AweAI-Team/BeyondSWE)

> BeyondSWE evaluates code agents beyond single-repo bug fixing with 500 real-world tasks across 4 challenging dimensions: cross-repo reasoning, domain-specific scientific coding, dependency migration, and full repo generation from specs — where eve...

</details>

<details>
<summary><b>5. Beyond Length Scaling: Synergizing Breadth and Depth for Generative Reward Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01571) • [📄 arXiv](https://arxiv.org/abs/2603.01571) • [📥 PDF](https://arxiv.org/pdf/2603.01571)

> 🚀 Is making CoT longer really the silver bullet for Reward Models? As long-cot dominates the LLM landscape, the standard approach to improving Generative Reward Models (LLM-as-a-Judge) has been straightforward: just force the model to generate lon...

</details>

<details>
<summary><b>6. Kling-MotionControl Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03160) • [📄 arXiv](https://arxiv.org/abs/2603.03160) • [📥 PDF](https://arxiv.org/pdf/2603.03160)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Co...

</details>

<details>
<summary><b>7. How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02578) • [📄 arXiv](https://arxiv.org/abs/2603.02578) • [📥 PDF](https://arxiv.org/pdf/2603.02578)

> We propose SteerEval, a hierarchical benchmark that systematically evaluates LLM controllability from high-level behavioral intent to fine-grained textual realization, revealing degradation in control at deeper specification levels and providing a...

</details>

<details>
<summary><b>8. Qwen3-Coder-Next Technical Report</b> ⭐ 15.9k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.00729) • [📄 arXiv](https://arxiv.org/abs/2603.00729) • [📥 PDF](https://arxiv.org/pdf/2603.00729)

**💻 Code:** [⭐ Code](https://github.com/QwenLM/Qwen3-Coder)

> We present Qwen3-Coder-Next, an open-weight language model specialized for coding agents. Qwen3-Coder-Next is an 80-billion-parameter model that activates only 3 billion parameters during inference, enabling strong coding capability with efficient...

</details>

<details>
<summary><b>9. PRISM: Pushing the Frontier of Deep Think via Process Reward Model-Guided Inference</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Noah Provenzano, Weiyuan Chen, Rituraj Sharma, tuvllms

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02479) • [📄 arXiv](https://arxiv.org/abs/2603.02479) • [📥 PDF](https://arxiv.org/pdf/2603.02479)

**💻 Code:** [⭐ Code](https://github.com/Rituraj003/PRISM)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Recycling Failures: Salvaging Exploration in RLVR via Fine-Grained Off-Poli...

</details>

<details>
<summary><b>10. Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance</b> ⭐ 45</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02175) • [📄 arXiv](https://arxiv.org/abs/2603.02175) • [📥 PDF](https://arxiv.org/pdf/2603.02175)

**💻 Code:** [⭐ Code](https://github.com/showlab/Kiwi-Edit)

> We present Kiwi-Edit, a unified and fully open-source framework for instruction-guided and reference-guided video editing using natural language. Kiwi-Edit supports high-quality, temporally consistent edits across global and local tasks, and deliv...

</details>

<details>
<summary><b>11. Next Embedding Prediction Makes World Models Stronger</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02765) • [📄 arXiv](https://arxiv.org/abs/2603.02765) • [📥 PDF](https://arxiv.org/pdf/2603.02765)

**💻 Code:** [⭐ Code](https://github.com/corl-team/nedreamer)

> Most world models learn representations by reconstructing pixels. But reconstruction isn’t necessarily aligned with control. In this paper we explore a different idea: ➡️predict the next encoder embedding instead of reconstructing the observation....

</details>

<details>
<summary><b>12. Humans and LLMs Diverge on Probabilistic Inferences</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23546) • [📄 arXiv](https://arxiv.org/abs/2602.23546) • [📥 PDF](https://arxiv.org/pdf/2602.23546)

**💻 Code:** [⭐ Code](https://github.com/McGill-NLP/probabilistic-reasoning)

> Abstract Human reasoning often involves working over limited information to arrive at probabilistic conclusions. In its simplest form, this involves making an inference that is not strictly entailed by a premise, but rather only likely given the p...

</details>

<details>
<summary><b>13. Surgical Post-Training: Cutting Errors, Keeping Knowledge</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01683) • [📄 arXiv](https://arxiv.org/abs/2603.01683) • [📥 PDF](https://arxiv.org/pdf/2603.01683)

**💻 Code:** [⭐ Code](https://github.com/Visual-AI/SPoT)

> Injecting new knowledge into LLMs via SFT often triggers catastrophic forgetting due to a "pull-up" effect , where boosting a target response unintentionally raises the probability of incorrect ones. While RL methods like GRPO are more robust, the...

</details>

<details>
<summary><b>14. InfoPO: Information-Driven Policy Optimization for User-Centric Agents</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Yuyu Luo, Chenglin Wu, Mingyi Deng, Jiayi Zhang, Fanqi Kong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.00656) • [📄 arXiv](https://arxiv.org/abs/2603.00656) • [📥 PDF](https://arxiv.org/pdf/2603.00656)

**💻 Code:** [⭐ Code](https://github.com/kfq20/InfoPO)

> 🌟 We introduce InfoPO (Information-Driven Policy Optimization) — a practical way to train multi-turn LLM agents with turn-level credit assignment. 🧠 Key idea: treat interaction as active uncertainty reduction . We compute a counterfactual informat...

</details>

<details>
<summary><b>15. BBQ-to-Image: Numeric Bounding Box and Qolor Control in Large-Scale Text-to-Image Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20672) • [📄 arXiv](https://arxiv.org/abs/2602.20672) • [📥 PDF](https://arxiv.org/pdf/2602.20672)

> Fibo BBQ: Bounding Box & Qolor Control in Large-Scale Text-to-Image Models Text prompts are a terrible UI for precision It’s much more intuitive to drag objects into place or use a color picker than to write “put it 20% left, make it teal, slightl...

</details>

<details>
<summary><b>16. Spilled Energy in Large Language Models</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Iacopo Masi, Hazem Dewidar, Adrian Robert Minut

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18671) • [📄 arXiv](https://arxiv.org/abs/2602.18671) • [📥 PDF](https://arxiv.org/pdf/2602.18671)

**💻 Code:** [⭐ Code](https://github.com/OmnAI-Lab/spilled-energy)

> We introduce a zero-shot, training-free method to detect LLM hallucinations by quantifying violations of the probability chain rule as an "energy spill" derived directly from output logits. We reinterpret the standard LLM softmax classifier as an ...

</details>

<details>
<summary><b>17. NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Binxin Yang, Zhengyao Lv, Chenpu Yuan, Jiayi Dai, ldiex

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02802) • [📄 arXiv](https://arxiv.org/abs/2603.02802) • [📥 PDF](https://arxiv.org/pdf/2603.02802)

**💻 Code:** [⭐ Code](https://github.com/WeChatCV/NovaEdit)

> Recent video editing models have achieved impressive results, but most still require large-scale paired datasets. Collecting such naturally aligned pairs at scale remains highly challenging and constitutes a critical bottleneck, especially for loc...

</details>

<details>
<summary><b>18. CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03281) • [📄 arXiv](https://arxiv.org/abs/2603.03281) • [📥 PDF](https://arxiv.org/pdf/2603.03281)

**💻 Code:** [⭐ Code](https://github.com/hanyang-21/CFG-Ctrl)

> Project page: https://hanyang-21.github.io/CFG-Ctrl GitHub repo: https://github.com/hanyang-21/CFG-Ctrl

</details>

<details>
<summary><b>19. Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels</b> ⭐ 39</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02573) • [📄 arXiv](https://arxiv.org/abs/2603.02573) • [📥 PDF](https://arxiv.org/pdf/2603.02573)

**💻 Code:** [⭐ Code](https://github.com/TencentARC/Track4World)

> Track4World estimates dense 3D scene flow of every pixel between arbitrary frame pairs from a monocular video in a global feedforward manner, enabling efficient and dense 3D tracking of every pixel in the world-centric coordinate system.

</details>

<details>
<summary><b>20. Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03205) • [📄 arXiv](https://arxiv.org/abs/2603.03205) • [📥 PDF](https://arxiv.org/pdf/2603.03205)

> Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use We introduce MOSAIC, a post-training framework that aligns agents for safe multi-step tool use by making safety decisions explicit and learnable. MOSAIC...

</details>

<details>
<summary><b>21. Chain of World: World Model Thinking in Latent Motion</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Lei Fan, Xuancheng Zhang, Lulu Tang, Donglin Di, Fuxiang Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03195) • [📄 arXiv](https://arxiv.org/abs/2603.03195) • [📥 PDF](https://arxiv.org/pdf/2603.03195)

**💻 Code:** [⭐ Code](https://github.com/fx-hit/CoWVLA)

> Vision-Language-Action (VLA) models are a promising path toward embodied intelligence, yet they often overlook the predictive and temporal-causal structure underlying visual dynamics. World-model VLAs address this by predicting future frames, but ...

</details>

<details>
<summary><b>22. SciDER: Scientific Data-centric End-to-end Researcher</b> ⭐ 70</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01421) • [📄 arXiv](https://arxiv.org/abs/2603.01421) • [📥 PDF](https://arxiv.org/pdf/2603.01421)

**💻 Code:** [⭐ Code](https://github.com/leonardodalinky/SciDER)

> SciDER is designed as a data-centric end-to-end system that flexibly automates the scientific research lifecycle. The system integrates a research framework comprising ideation, data analysis, experimentation, and iterative improvement. It support...

</details>

<details>
<summary><b>23. DREAM: Where Visual Understanding Meets Text-to-Image Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Satya Narayan Shukla, Hong-You Chen, Sai Vidyaranya Nuthalapati, Tianhong Li, Chao Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02667) • [📄 arXiv](https://arxiv.org/abs/2603.02667) • [📥 PDF](https://arxiv.org/pdf/2603.02667)

**💻 Code:** [⭐ Code](https://github.com/chaoli-charlie/dream)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API OpenVision 3: A Family of Unified Visual Encoder for Both Understanding and...

</details>

<details>
<summary><b>24. ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02510) • [📄 arXiv](https://arxiv.org/abs/2603.02510) • [📥 PDF](https://arxiv.org/pdf/2603.02510)

**💻 Code:** [⭐ Code](https://github.com/WildAlg/ParEVO)

> ParEVO: Synthesizing Code for Irregular Data: High-Performance Parallelism through Agentic Evolution TL;DR: While current LLMs excel at writing standard sequential code, they often fail catastrophically when attempting to write concurrent algorith...

</details>

<details>
<summary><b>25. Transformers converge to invariant algorithmic cores</b> ⭐ 0</summary>

<br/>

**👥 Authors:** joshseth

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22600) • [📄 arXiv](https://arxiv.org/abs/2602.22600) • [📥 PDF](https://arxiv.org/pdf/2602.22600)

> Training selects for behavior, not circuitry – so which internal structures reflect the computation, and which are accidents of a particular training run? Independently trained transformers – despite having very different weights – converge to sha...

</details>

<details>
<summary><b>26. QEDBENCH: Quantifying the Alignment Gap in Automated Evaluation of University-Level Mathematical Proofs</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20629) • [📄 arXiv](https://arxiv.org/abs/2602.20629) • [📥 PDF](https://arxiv.org/pdf/2602.20629)

**💻 Code:** [⭐ Code](https://github.com/qqliu/Yale-QEDBench)

> QEDBENCH: Quantifying the Alignment Gap in Automated Evaluation of University-Level Mathematical Proofs TL;DR: As LLMs max out elementary math benchmarks, the research frontier is shifting from solving math to reliably evaluating it. This paper in...

</details>

<details>
<summary><b>27. Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhiyuan Fan, Jiayu Liu, Qingyu Liu, Yuejin Xie, Dadi Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03202) • [📄 arXiv](https://arxiv.org/abs/2603.03202) • [📥 PDF](https://arxiv.org/pdf/2603.03202)

**💻 Code:** [⭐ Code](https://github.com/TarferSoul/Code2Math)

> Github: https://github.com/TarferSoul/Code2Math

</details>

<details>
<summary><b>28. APRES: An Agentic Paper Revision and Evaluation System</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03142) • [📄 arXiv](https://arxiv.org/abs/2603.03142) • [📥 PDF](https://arxiv.org/pdf/2603.03142)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Paper2Rebuttal: A Multi-Agent Framework for Transparent Author Response Ass...

</details>

<details>
<summary><b>29. AgentConductor: Topology Evolution for Multi-Agent Competition-Level Code Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17100) • [📄 arXiv](https://arxiv.org/abs/2602.17100) • [📥 PDF](https://arxiv.org/pdf/2602.17100)

> A new framework dynamically adjusts multi-agent connections to solve complex programming challenges while using fewer tokens. The big deal here is the shift from rigid workflows to fluid teamwork. Normal multi-agent systems use a fixed, hardcoded ...

</details>

<details>
<summary><b>30. Conditioned Activation Transport for T2I Safety Steering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Franziska Boenisch, Tomasz Trzciński, Jan Dubiński, Aleksander Szymczyk, Maciej Chrabąszcz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03163) • [📄 arXiv](https://arxiv.org/abs/2603.03163) • [📥 PDF](https://arxiv.org/pdf/2603.03163)

**💻 Code:** [⭐ Code](https://github.com/NASK-AISafety/conditional-activation-transport)

> We introduce non-linear conditioned transport for steering text to image models together with safety contrastive SafeSteerDataset.

</details>

<details>
<summary><b>31. HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Md. Shad Akhtar, Sunil Saumya, Shankar Biradar, UVSKKR

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02684) • [📄 arXiv](https://arxiv.org/abs/2603.02684) • [📥 PDF](https://arxiv.org/pdf/2603.02684)

**💻 Code:** [⭐ Code](https://github.com/Sai-Kartheek-Reddy/HateMirage)

> Subtle and indirect hate speech remains an underexplored challenge in online safety research, particularly when harmful intent is embedded within misleading or manipulative narratives. Existing hate speech datasets primarily capture overt toxicity...

</details>

<details>
<summary><b>32. DynaMoE: Dynamic Token-Level Expert Activation with Layer-Wise Adaptive Capacity for Mixture-of-Experts Neural Networks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Goekdeniz-Guelmez

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01697) • [📄 arXiv](https://arxiv.org/abs/2603.01697) • [📥 PDF](https://arxiv.org/pdf/2603.01697)

> As the writer and publisher, I think this is a interesting analysis of MoEs.

</details>

<details>
<summary><b>33. Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nicu Sebe, Haonan Zhang, Liyuan Jiang, TyroneDragon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01400) • [📄 arXiv](https://arxiv.org/abs/2603.01400) • [📥 PDF](https://arxiv.org/pdf/2603.01400)

> Video Large Language Models (VLLMs) demonstrate strong video understanding but suffer from inefficiency due to redundant visual tokens. Existing pruning primary targets intra-frame spatial redundancy or prunes inside the LLM with shallow-layer ove...

</details>

<details>
<summary><b>34. GroupGPT: A Token-efficient and Privacy-preserving Agentic Framework for Multi-User Chat Assistant</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Shaohui Lin, Wenxuan Huang, Hanyu Chen, Yifan Wang, Zhuokang Shen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01059) • [📄 arXiv](https://arxiv.org/abs/2603.01059) • [📥 PDF](https://arxiv.org/pdf/2603.01059)

**💻 Code:** [⭐ Code](https://github.com/Eliot-Shen/GroupGPT)

> No abstract available.

</details>

<details>
<summary><b>35. Whisper-RIR-Mega: A Paired Clean-Reverberant Speech Benchmark for ASR Robustness to Room Acoustics</b> ⭐ 1</summary>

<br/>

**👥 Authors:** mandipgoswami

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02252) • [📄 arXiv](https://arxiv.org/abs/2603.02252) • [📥 PDF](https://arxiv.org/pdf/2603.02252)

**💻 Code:** [⭐ Code](https://github.com/mandip42/whisper-rirmega-bench)

> We introduce Whisper-RIR-Mega, a benchmark dataset of paired clean and reverberant speech for evaluating automatic speech recognition (ASR) robustness to room acoustics. Each sample pairs a clean LibriSpeech utterance with the same utterance convo...

</details>

<details>
<summary><b>36. SGDC: Structurally-Guided Dynamic Convolution for Medical Image Segmentation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** M. N. S. Swamy, Wei-ping Zhu, solstice0621

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23496) • [📄 arXiv](https://arxiv.org/abs/2602.23496) • [📥 PDF](https://arxiv.org/pdf/2602.23496)

**💻 Code:** [⭐ Code](https://github.com/solstice0621/SGDC)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Phi-SegNet: Phase-Integrated Supervision for Medical Image Segmentation (20...

</details>

<details>
<summary><b>37. Towards Simulating Social Media Users with LLMs: Evaluating the Operational Validity of Conditioned Comment Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22752) • [📄 arXiv](https://arxiv.org/abs/2602.22752) • [📥 PDF](https://arxiv.org/pdf/2602.22752)

**💻 Code:** [⭐ Code](https://github.com/nsschw/Conditioned-Comment-Prediction)

> Towards Simulating Social Media Users with LLMs: Evaluating the Operational Validity of Conditioned Comment Prediction :)

</details>

<details>
<summary><b>38. Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Seunghoon Lee, Yoonji Lee, Eunju Lee, MiHyeon Kim, JuneHyoung Kwon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21773) • [📄 arXiv](https://arxiv.org/abs/2602.21773) • [📥 PDF](https://arxiv.org/pdf/2602.21773)

> Excited to share our new paper, "Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias". We identify a critical failure mode in current methods called "shortcut unlearning," where models paradoxically forget bias attributes inste...

</details>

<details>
<summary><b>39. Fast Matrix Multiplication in Small Formats: Discovering New Schemes with an Open-Source Flip Graph Framework</b> ⭐ 1</summary>

<br/>

**👥 Authors:** dronperminov

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02398) • [📄 arXiv](https://arxiv.org/abs/2603.02398) • [📥 PDF](https://arxiv.org/pdf/2603.02398)

**💻 Code:** [⭐ Code](https://github.com/dronperminov/ternary_flip_graph) • [⭐ Code](https://github.com/dronperminov/FastMatrixMultiplication)

> 71 improved ranks for matrix multiplication, a new 4x4x10 scheme with ω < 2.807 and 178 schemes rediscovered in ternary/integer coefficients. Results obtained with an open-source flip graph framework.

</details>

<details>
<summary><b>40. Transform-Invariant Generative Ray Path Sampling for Efficient Radio Propagation Modeling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01655) • [📄 arXiv](https://arxiv.org/abs/2603.01655) • [📥 PDF](https://arxiv.org/pdf/2603.01655)

**💻 Code:** [⭐ Code](https://github.com/jeertmans/sampling-paths)

> Hi everyone! I have just submitted my new journal paper on using Generative Flow Networks (GFlowNets) to speed up radio propagation modeling. Don't hesitate to checkout the paper or the tutorial notebook ! The problem and our solution Traditional ...

</details>

<details>
<summary><b>41. Multi-Domain Riemannian Graph Gluing for Building Graph Foundation Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Junda Ye, Lanxu Yang, Silei Chen, Li Sun, ZhenhHuang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.00618) • [📄 arXiv](https://arxiv.org/abs/2603.00618) • [📥 PDF](https://arxiv.org/pdf/2603.00618)

**💻 Code:** [⭐ Code](https://github.com/RiemannGraph/GraphGlue)

> Multi-domain graph pre-training integrates knowledge from diverse domains to enhance performance in the target domains, which is crucial for building graph foundation models. Despite initial success, existing solutions often fall short of answerin...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 41 |
| 📅 Today | [`2026-03-05.json`](data/daily/2026-03-05.json) | 41 |
| 📆 This Week | [`2026-W09.json`](data/weekly/2026-W09.json) | 132 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 160 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-05 | 41 | [View JSON](data/daily/2026-03-05.json) |
| 📄 2026-03-04 | 41 | [View JSON](data/daily/2026-03-04.json) |
| 📄 2026-03-03 | 22 | [View JSON](data/daily/2026-03-03.json) |
| 📄 2026-03-02 | 28 | [View JSON](data/daily/2026-03-02.json) |
| 📄 2026-03-01 | 28 | [View JSON](data/daily/2026-03-01.json) |
| 📄 2026-02-28 | 28 | [View JSON](data/daily/2026-02-28.json) |
| 📄 2026-02-27 | 30 | [View JSON](data/daily/2026-02-27.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W09 | 132 | [View JSON](data/weekly/2026-W09.json) |
| 📅 2026-W08 | 184 | [View JSON](data/weekly/2026-W08.json) |
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 160 | [View JSON](data/monthly/2026-03.json) |
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
