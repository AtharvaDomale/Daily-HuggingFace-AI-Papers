<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-32-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1270+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">32</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">103</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">532</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1270+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 22, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization</b> ⭐ 265</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.12993) • [📄 arXiv](https://arxiv.org/abs/2601.12993) • [📥 PDF](https://arxiv.org/pdf/2601.12993)

**💻 Code:** [⭐ Code](https://github.com/BeingBeyond/Being-H)

> We scale human-centric robot learning with Being-H0.5 toward cross-embodiment generalization. Building on over 35,000 hours data, we unify human hand motion and diverse robot embodiments with a Unified Action Space, and train all heterogeneous sup...

</details>

<details>
<summary><b>2. Advances and Frontiers of LLM-based Issue Resolution in Software Engineering: A Comprehensive Survey</b> ⭐ 40</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11655) • [📄 arXiv](https://arxiv.org/abs/2601.11655) • [📥 PDF](https://arxiv.org/pdf/2601.11655)

**💻 Code:** [⭐ Code](https://github.com/DeepSoftwareAnalytics/Awesome-Issue-Resolution)

> 🚀 Awesome issue resolution: a comprehensive survey! This paper surveyed 175+ works to construct the first unified taxonomy serving as the comprehensive roadmap for issue resolution.

</details>

<details>
<summary><b>3. Think3D: Thinking with Space for Spatial Reasoning</b> ⭐ 32</summary>

<br/>

**👥 Authors:** Yuhan Wu, JeremyYin, sunz525, luciasnowblack, MrBean2024

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13029) • [📄 arXiv](https://arxiv.org/abs/2601.13029) • [📥 PDF](https://arxiv.org/pdf/2601.13029)

**💻 Code:** [⭐ Code](https://github.com/zhangzaibin/spagent)

> We introduce Think3D, a framework that enables VLM agents to think in 3D space. By leveraging 3D reconstruction models that recover point clouds and camera poses from images or videos, Think3D allows the agent to actively manipulate space through ...

</details>

<details>
<summary><b>4. OmniTransfer: All-in-one Framework for Spatio-temporal Video Transfer</b> ⭐ 54</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14250) • [📄 arXiv](https://arxiv.org/abs/2601.14250) • [📥 PDF](https://arxiv.org/pdf/2601.14250)

**💻 Code:** [⭐ Code](https://github.com/PangzeCheung/OmniTransfer)

> Videos convey richer information than images or text, capturing both spatial and temporal dynamics. However, most existing video customization methods rely on reference images or task-specific temporal priors, failing to fully exploit the rich spa...

</details>

<details>
<summary><b>5. Toward Efficient Agents: Memory, Tool learning, and Planning</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14192) • [📄 arXiv](https://arxiv.org/abs/2601.14192) • [📥 PDF](https://arxiv.org/pdf/2601.14192)

**💻 Code:** [⭐ Code](https://github.com/yxf203/Awesome-Efficient-Agents)

> This paper surveys efficiency-oriented methods for agentic systems across memory, tool learning, and planning, distills shared design principles, and summarizes how recent methods and benchmarks measure efficiency, which hopes to guide the develop...

</details>

<details>
<summary><b>6. FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13836) • [📄 arXiv](https://arxiv.org/abs/2601.13836) • [📥 PDF](https://arxiv.org/pdf/2601.13836)

**💻 Code:** [⭐ Code](https://github.com/OpenMOSS/FutureOmni)

> FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs 🔗 Paper: https://arxiv.org/pdf/2601.13836 💻 Code: https://github.com/OpenMOSS/FutureOmni 🌐 Project: https://openmoss.github.io/FutureOmni 🎬 Datasets: https://hug...

</details>

<details>
<summary><b>7. MemoryRewardBench: Benchmarking Reward Models for Long-Term Memory Management in Large Language Models</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11969) • [📄 arXiv](https://arxiv.org/abs/2601.11969) • [📥 PDF](https://arxiv.org/pdf/2601.11969)

**💻 Code:** [⭐ Code](https://github.com/LCM-Lab/MemRewardBench)

> Check our code: https://github.com/LCM-Lab/MemRewardBench and Benchmark: https://huggingface.co/datasets/LCM-Lab/MemRewardBench

</details>

<details>
<summary><b>8. Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** qiaw99, WANGYIWEI, zunhai, mingyang26, hengyuanya

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14004) • [📄 arXiv](https://arxiv.org/abs/2601.14004) • [📥 PDF](https://arxiv.org/pdf/2601.14004)

> Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models

</details>

<details>
<summary><b>9. UniX: Unifying Autoregression and Diffusion for Chest X-Ray Understanding and Generation</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11522) • [📄 arXiv](https://arxiv.org/abs/2601.11522) • [📥 PDF](https://arxiv.org/pdf/2601.11522)

**💻 Code:** [⭐ Code](https://github.com/ZrH42/UniX)

> We introduce UniX, a unified foundation model for Chest X-Ray that combines Autoregression (for understanding) and Diffusion (for generation) within a decoupled dual-branch architecture! 🏥✨ Why UniX? Current unified models often face a conflict be...

</details>

<details>
<summary><b>10. ToolPRMBench: Evaluating and Advancing Process Reward Models for Tool-using Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.12294) • [📄 arXiv](https://arxiv.org/abs/2601.12294) • [📥 PDF](https://arxiv.org/pdf/2601.12294)

> ToolPRMBench: Evaluating and Advancing Process Reward Models for Tool-using Agents

</details>

<details>
<summary><b>11. Aligning Agentic World Models via Knowledgeable Experience Learning</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13247) • [📄 arXiv](https://arxiv.org/abs/2601.13247) • [📥 PDF](https://arxiv.org/pdf/2601.13247)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/WorldMind)

> WorldMind helps language models stop making physically impossible plans by learning real-world rules from feedback and successful experiences, rather than retraining the model itself.

</details>

<details>
<summary><b>12. Agentic-R: Learning to Retrieve for Agentic Search</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Daiting Shi, Yuchen Li, Yutao Zhu, Xinyu Ma, Wenhan Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11888) • [📄 arXiv](https://arxiv.org/abs/2601.11888) • [📥 PDF](https://arxiv.org/pdf/2601.11888)

> Agentic-R: Learning to Retrieve for Agentic Search

</details>

<details>
<summary><b>13. A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13288) • [📄 arXiv](https://arxiv.org/abs/2601.13288) • [📥 PDF](https://arxiv.org/pdf/2601.13288)

> Rather than adding another model to the stack, this work reuses computation already paid for in the serving LLM’s forward pass by training compact probes on hidden states. It frames the problem as principled selection across tokens and layers (not...

</details>

<details>
<summary><b>14. KAGE-Bench: Fast Known-Axis Visual Generalization Evaluation for Reinforcement Learning</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Aleksandr I. Panov, Alexey K. Kovalev, Daniil Zelezetsky, Egor Cherepanov

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14232) • [📄 arXiv](https://arxiv.org/abs/2601.14232) • [📥 PDF](https://arxiv.org/pdf/2601.14232)

**💻 Code:** [⭐ Code](https://github.com/CognitiveAISystems/kage-bench)

> Pixel-based reinforcement learning agents often fail under purely visual distribution shift even when latent dynamics and rewards are unchanged, but existing benchmarks entangle multiple sources of shift and hinder systematic analysis. We introduc...

</details>

<details>
<summary><b>15. LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14251) • [📄 arXiv](https://arxiv.org/abs/2601.14251) • [📥 PDF](https://arxiv.org/pdf/2601.14251)

> We present LightOnOCR-2-1B , a 1B-parameter end-to-end multilingual vision-language model that converts document images (e.g., PDFs) into clean, naturally ordered text without brittle OCR pipelines. Trained on a large-scale, high-quality distillat...

</details>

<details>
<summary><b>16. PRiSM: Benchmarking Phone Realization in Speech Models</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14046) • [📄 arXiv](https://arxiv.org/abs/2601.14046) • [📥 PDF](https://arxiv.org/pdf/2601.14046)

**💻 Code:** [⭐ Code](https://github.com/changelinglab/prism)

> Main take-aways PRiSM is the first fully-open benchmark that evaluates Phone-Recognition systems on both intrinsic (phone-transcription) and extrinsic (down-stream) tasks across 12 datasets covering clinical, L2-learning and multilingual settings....

</details>

<details>
<summary><b>17. FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13976) • [📄 arXiv](https://arxiv.org/abs/2601.13976) • [📥 PDF](https://arxiv.org/pdf/2601.13976)

**💻 Code:** [⭐ Code](https://github.com/Fantasy-AMAP/fantasy-vln) • [⭐ Code](https://github.com/FoundationVision/VAR)

> FantasyVLN is a unified multimodal Chain-of-Thought (CoT) reasoning framework that enables efficient and precise navigation based on natural language instructions and visual observations. FantasyVLN combines the benefits of textual, visual, and mu...

</details>

<details>
<summary><b>18. DARC: Decoupled Asymmetric Reasoning Curriculum for LLM Evolution</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13761) • [📄 arXiv](https://arxiv.org/abs/2601.13761) • [📥 PDF](https://arxiv.org/pdf/2601.13761)

**💻 Code:** [⭐ Code](https://github.com/RUCBM/DARC)

> In this work, we introduce the DARC framework, which adopts decoupled training and asymmetric self-distillation to stabilize self-evolving. We hope this work provides useful insights for LLM self-evolution. avXiv: https://arxiv.org/abs/2601.13761 ...

</details>

<details>
<summary><b>19. Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14249) • [📄 arXiv](https://arxiv.org/abs/2601.14249) • [📥 PDF](https://arxiv.org/pdf/2601.14249)

**💻 Code:** [⭐ Code](https://github.com/UmeanNever/RankSurprisalRatio)

> Code: https://github.com/UmeanNever/RankSurprisalRatio

</details>

<details>
<summary><b>20. InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14209) • [📄 arXiv](https://arxiv.org/abs/2601.14209) • [📥 PDF](https://arxiv.org/pdf/2601.14209)

> Outcome-reward reinforcement learning (RL) has proven effective at improving the reasoning capabilities of large language models (LLMs). However, standard RL assigns credit only at the level of the final answer, penalizing entire reasoning traces ...

</details>

<details>
<summary><b>21. Uncertainty-Aware Gradient Signal-to-Noise Data Selection for Instruction Tuning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13697) • [📄 arXiv](https://arxiv.org/abs/2601.13697) • [📥 PDF](https://arxiv.org/pdf/2601.13697)

> Instruction tuning is a standard paradigm for adapting large language models (LLMs), but modern instruction datasets are large, noisy, and redundant, making full-data fine-tuning costly and often unnecessary. Existing data selection methods either...

</details>

<details>
<summary><b>22. On the Evidentiary Limits of Membership Inference for Copyright Auditing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Marten van Dijk, Kaleel Mahmood, Min Chen, emirhanboge, bilgehanertan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.12937) • [📄 arXiv](https://arxiv.org/abs/2601.12937) • [📥 PDF](https://arxiv.org/pdf/2601.12937)

> 🧑‍⚖️📄 This paper shows that membership inference attacks are not reliable technical evidence for copyright infringement in court. Even with strong MIAs, semantics-preserving paraphrasing breaks the signal while keeping utility, making them brittle...

</details>

<details>
<summary><b>23. Fundamental Limitations of Favorable Privacy-Utility Guarantees for DP-SGD</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10237) • [📄 arXiv](https://arxiv.org/abs/2601.10237) • [📥 PDF](https://arxiv.org/pdf/2601.10237)

> This paper quantifies a fundamental lower bound on the noise required for differentially private stochastic gradient descent (DP-SGD) to maintain strong privacy, revealing that even with massive datasets and both shuffled and Poisson subsampling, ...

</details>

<details>
<summary><b>24. DSAEval: Evaluating Data Science Agents on a Wide Range of Real-World Data Science Problems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13591) • [📄 arXiv](https://arxiv.org/abs/2601.13591) • [📥 PDF](https://arxiv.org/pdf/2601.13591)

> This paper introduce the DSAEval, evaluating LLM based Data Agent in a wide-range of real world problems.

</details>

<details>
<summary><b>25. A Hybrid Protocol for Large-Scale Semantic Dataset Generation in Low-Resource Languages: The Turkish Semantic Relations Corpus</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Özay Ezerceli, Mehmet Emin Buldur, MElHuseyni, etosun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13253) • [📄 arXiv](https://arxiv.org/abs/2601.13253) • [📥 PDF](https://arxiv.org/pdf/2601.13253)

> Addressing data scarcity in low-resource languages, this paper introduces a cost-effective ($65) pipeline for generating large-scale semantic datasets. By integrating FastText clustering, Gemini 2.5-Flash labeling, and dictionary curation, the aut...

</details>

<details>
<summary><b>26. Beyond Cosine Similarity: Taming Semantic Drift and Antonym Intrusion in a 15-Million Node Turkish Synonym Graph</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Özay Ezerceli, Mehmet Emin Buldur, MElHuseyni, etosun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13251) • [📄 arXiv](https://arxiv.org/abs/2601.13251) • [📥 PDF](https://arxiv.org/pdf/2601.13251)

> This paper addresses the inability of neural embeddings to distinguish synonyms from antonyms. The authors introduce a soft-to-hard clustering algorithm that prevents semantic drift and a 3-way relation discriminator (90% F1). Validated against a ...

</details>

<details>
<summary><b>27. METIS: Mentoring Engine for Thoughtful Inquiry & Solutions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13075) • [📄 arXiv](https://arxiv.org/abs/2601.13075) • [📥 PDF](https://arxiv.org/pdf/2601.13075)

> Students have immense research potential, but enough mentors for them. What if we could design an AI system to mentor them? We introduce METIS (Mentoring Engine for Thoughtful Inquiry & Solutions), a stage-aware research mentor.

</details>

<details>
<summary><b>28. SciCoQA: Quality Assurance for Scientific Paper--Code Alignment</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.12910) • [📄 arXiv](https://arxiv.org/abs/2601.12910) • [📥 PDF](https://arxiv.org/pdf/2601.12910)

**💻 Code:** [⭐ Code](https://github.com/UKPLab/scicoqa) • [⭐ Code](https://github.com/ukplab/scicoqa)

> We introduce the SciCoQA dataset for evaluating models on detecting discrepancies between paper and code. Find all resources here: Paper: arXiv Data: Hugging Face Dataset Code: GitHub Demo: Hugging Face Space Project Page : UKPLab/scicoqa

</details>

<details>
<summary><b>29. LIBERTy: A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Counterfactuals</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10700) • [📄 arXiv](https://arxiv.org/abs/2601.10700) • [📥 PDF](https://arxiv.org/pdf/2601.10700)

**💻 Code:** [⭐ Code](https://github.com/GilatToker/Liberty-benchmark)

> The paper addresses the lack of reliable ground-truth benchmarks for evaluating concept-based explainability in Large Language Models. The authors introduce LIBERTy, a framework that generates "structural counterfactuals" by explicitly defining St...

</details>

<details>
<summary><b>30. Finally Outshining the Random Baseline: A Simple and Effective Solution for Active Learning in 3D Biomedical Imaging</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13677) • [📄 arXiv](https://arxiv.org/abs/2511.19183) • [📥 PDF](https://arxiv.org/pdf/2601.13677)

**💻 Code:** [⭐ Code](https://github.com/MIC-DKFZ/nnActive/tree/nnActive_v2)

> 🚀 Building on nnActive , an evaluation framework for active learning in 3D biomedical imaging, this paper proposes a simple and effective method that consistently outperforms strong random baselines.

</details>

<details>
<summary><b>31. Towards Efficient and Robust Linguistic Emotion Diagnosis for Mental Health via Multi-Agent Instruction Refinement</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu He, Weiping Fu, Zhiyuan Wang, Zhangqi Wang, Jian Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13481) • [📄 arXiv](https://arxiv.org/abs/2601.13481) • [📥 PDF](https://arxiv.org/pdf/2601.13481)

> We propose APOLO (Automated Prompt Optimization for Linguistic emOtion diagnosis), a framework that systematically explores a broader and finer-grained prompt space to enhance diagnostic efficiency and robustness.

</details>

<details>
<summary><b>32. RemoteVAR: Autoregressive Visual Modeling for Remote Sensing Change Detection</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11898) • [📄 arXiv](https://arxiv.org/abs/2601.11898) • [📥 PDF](https://arxiv.org/pdf/2601.11898)

**💻 Code:** [⭐ Code](https://github.com/yilmazkorkmaz1/RemoteVAR)

> https://github.com/yilmazkorkmaz1/RemoteVAR

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 32 |
| 📅 Today | [`2026-01-22.json`](data/daily/2026-01-22.json) | 32 |
| 📆 This Week | [`2026-W03.json`](data/weekly/2026-W03.json) | 103 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 532 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-22 | 32 | [View JSON](data/daily/2026-01-22.json) |
| 📄 2026-01-21 | 11 | [View JSON](data/daily/2026-01-21.json) |
| 📄 2026-01-20 | 22 | [View JSON](data/daily/2026-01-20.json) |
| 📄 2026-01-19 | 38 | [View JSON](data/daily/2026-01-19.json) |
| 📄 2026-01-18 | 38 | [View JSON](data/daily/2026-01-18.json) |
| 📄 2026-01-17 | 38 | [View JSON](data/daily/2026-01-17.json) |
| 📄 2026-01-16 | 27 | [View JSON](data/daily/2026-01-16.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W03 | 103 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 532 | [View JSON](data/monthly/2026-01.json) |
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
