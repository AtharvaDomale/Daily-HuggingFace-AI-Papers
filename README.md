<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-52-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1827+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">52</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">263</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">308</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1827+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 06, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. ERNIE 5.0 Technical Report</b> ⭐ 0</summary>

<br/>

**👥 Authors:** HasuerYu, LLLL, guanwcn, max-zhenyu-zhang, sjy1203

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04705) • [📄 arXiv](https://arxiv.org/abs/2602.04705) • [📥 PDF](https://arxiv.org/pdf/2602.04705)

> good work

</details>

<details>
<summary><b>2. FASA: Frequency-aware Sparse Attention</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03152) • [📄 arXiv](https://arxiv.org/abs/2602.03152) • [📥 PDF](https://arxiv.org/pdf/2602.03152)

> [ICLR26] A very interesting and effective work to speed up the inference of large models!

</details>

<details>
<summary><b>3. WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning</b> ⭐ 2.38k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04634) • [📄 arXiv](https://arxiv.org/abs/2602.04634) • [📥 PDF](https://arxiv.org/pdf/2602.04634)

**💻 Code:** [⭐ Code](https://github.com/RLinf/RLinf/tree/main/examples/wideseek_r1)

> We introduce WideSeek-R1, a lead-agent-subagent system trained via multi-agent RL to explore width scaling for broad information seeking. 🌐 Project Page | 📄 Paper | 💻 Code | 📦 Dataset | 🤗 Models

</details>

<details>
<summary><b>4. Training Data Efficiency in Multimodal Process Reward Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haolin Liu, Shaoyang Xu, Langlin Huang, Chengsong Huang, jinyuan222

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04145) • [📄 arXiv](https://arxiv.org/abs/2602.04145) • [📥 PDF](https://arxiv.org/pdf/2602.04145)

**💻 Code:** [⭐ Code](https://github.com/JinYuanLi0012/Balanced-Info-MPRM)

> Multimodal Process Reward Models (MPRMs) are central to step-level supervision for visual reasoning in MLLMs. Training MPRMs typically requires large-scale Monte Carlo (MC)-annotated corpora, incurring substantial training cost. This paper studies...

</details>

<details>
<summary><b>5. OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yiyan Ji, UnnamedWatcher, xuyang-liu16, Jungang, dingyue1011

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04804) • [📄 arXiv](https://arxiv.org/abs/2602.04804) • [📥 PDF](https://arxiv.org/pdf/2602.04804)

> We present OmniSIFT , which is a modality-asymmetric token compression framework tailored for Omni-LLMs.

</details>

<details>
<summary><b>6. HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03560) • [📄 arXiv](https://arxiv.org/abs/2602.03560) • [📥 PDF](https://arxiv.org/pdf/2602.03560)

> Efficient LLM Architecture, Sparse Attention, Hybrid Architecture

</details>

<details>
<summary><b>7. EgoActor: Grounding Task Planning into Spatial-aware Egocentric Actions for Humanoid Robots via Visual-Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ziyi Bai, Chaojie Li, MingMing Yu, Yu Bai, tellarin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04515) • [📄 arXiv](https://arxiv.org/abs/2602.04515) • [📥 PDF](https://arxiv.org/pdf/2602.04515)

> EgoActor is one of the key components of project RoboNoid. Project page: https://baai-agents.github.io/EgoActor/

</details>

<details>
<summary><b>8. Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02958) • [📄 arXiv](https://arxiv.org/abs/2602.02958) • [📥 PDF](https://arxiv.org/pdf/2602.02958)

> Efficient Long Video Generation, designed for world models and autoregressive video gen applications

</details>

<details>
<summary><b>9. SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02402) • [📄 arXiv](https://arxiv.org/abs/2602.02402) • [📥 PDF](https://arxiv.org/pdf/2602.02402)

> Project Page: https://city-super.github.io/SoMA/

</details>

<details>
<summary><b>10. TIDE: Trajectory-based Diagnostic Evaluation of Test-Time Improvement in LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qiushi Sun, Fangzhi Xu, Xinyu Che, Hang Yan, VentureZJ

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02196) • [📄 arXiv](https://arxiv.org/abs/2602.02196) • [📥 PDF](https://arxiv.org/pdf/2602.02196)

> First Paper For Diagnostic Evaluation of Test-Time Improvement in LLM Agents

</details>

<details>
<summary><b>11. Residual Context Diffusion Language Models</b> ⭐ 44</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22954) • [📄 arXiv](https://arxiv.org/abs/2601.22954) • [📥 PDF](https://arxiv.org/pdf/2601.22954)

**💻 Code:** [⭐ Code](https://github.com/yuezhouhu/residual-context-diffusion)

> We introduce Residual Context Diffusion (RCD): a simple idea to boost diffusion LLMs—stop wasting “remasked” tokens. Diffusion LLMs decode in parallel but often lag AR models because low-confidence tokens are discarded each step. RCD turns those d...

</details>

<details>
<summary><b>12. Rethinking the Trust Region in LLM Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04879) • [📄 arXiv](https://arxiv.org/abs/2602.04879) • [📥 PDF](https://arxiv.org/pdf/2602.04879)

> No abstract available.

</details>

<details>
<summary><b>13. Learning to Repair Lean Proofs from Compiler Feedback</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02990) • [📄 arXiv](https://arxiv.org/abs/2602.02990) • [📥 PDF](https://arxiv.org/pdf/2602.02990)

> Existing Lean datasets contain correct proofs. Models learn error correction with RL, that's expensive. We release a dataset of 260k erroneous Lean proofs, the compiler feedback, error explanation, proof repair reasoning trace, and the corrected p...

</details>

<details>
<summary><b>14. Semantic Routing: Exploring Multi-Layer LLM Feature Weighting for Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03510) • [📄 arXiv](https://arxiv.org/abs/2602.03510) • [📥 PDF](https://arxiv.org/pdf/2602.03510)

> Recent DiT-based text-to-image models increasingly adopt LLMs as text encoders, yet text conditioning remains largely static and often utilizes only a single LLM layer, despite pronounced semantic hierarchy across LLM layers and non-stationary den...

</details>

<details>
<summary><b>15. HY3D-Bench: Generation of 3D Assets</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03907) • [📄 arXiv](https://arxiv.org/abs/2602.03907) • [📥 PDF](https://arxiv.org/pdf/2602.03907)

> HY3D-Bench provides a unified 3D generation data ecosystem with 250k real assets, 125k synthetic assets, structured part-level decomposition, and a pipeline enabling scalable 3D model training.

</details>

<details>
<summary><b>16. AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03828) • [📄 arXiv](https://arxiv.org/abs/2602.03828) • [📥 PDF](https://arxiv.org/pdf/2602.03828)

**💻 Code:** [⭐ Code](https://github.com/ResearAI/AutoFigure-Edit)

> AutoFigure [Accepted to ICLR 2026] An automated scientific figure-drawing system for controllable generation of paper method diagrams. It is now fully open-sourced. The sketch generation process is user-intervenable and editable, avoiding “black-b...

</details>

<details>
<summary><b>17. Self-Hinting Language Models Enhance Reinforcement Learning</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03143) • [📄 arXiv](https://arxiv.org/abs/2602.03143) • [📥 PDF](https://arxiv.org/pdf/2602.03143)

**💻 Code:** [⭐ Code](https://github.com/BaohaoLiao/SAGE)

> RL for LLMs often stalls under sparse rewards — especially with GRPO, where whole rollout groups get identical 0 rewards and learning just… dies. 💡 SAGE fixes this with a simple but powerful idea: 👉 Let the model give itself hints during training....

</details>

<details>
<summary><b>18. CL-bench: A Benchmark for Context Learning</b> ⭐ 312</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03587) • [📄 arXiv](https://arxiv.org/abs/2602.03587) • [📥 PDF](https://arxiv.org/pdf/2602.03587)

**💻 Code:** [⭐ Code](https://github.com/Tencent-Hunyuan/CL-bench)

> A benchmark for context learning

</details>

<details>
<summary><b>19. Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04575) • [📄 arXiv](https://arxiv.org/abs/2602.04575) • [📥 PDF](https://arxiv.org/pdf/2602.04575)

> For the past decade, the trajectory of generative artificial intelligence (AI) has been dominated by a model-centric paradigm driven by scaling laws. Despite significant leaps in visual fidelity, this approach has encountered a “usability ceiling”...

</details>

<details>
<summary><b>20. VLS: Steering Pretrained Robot Policies via Vision-Language Models</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03973) • [📄 arXiv](https://arxiv.org/abs/2602.03973) • [📥 PDF](https://arxiv.org/pdf/2602.03973)

**💻 Code:** [⭐ Code](https://github.com/Vision-Language-Steering/code)

> Why do pretrained diffusion or flow-matching policies fail when the same task is performed near an obstacle, on a shifted support surface, or amid mild clutter? Such failures rarely reflect missing motor skills; instead, they expose a limitation o...

</details>

<details>
<summary><b>21. A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces</b> ⭐ 39</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03442) • [📄 arXiv](https://arxiv.org/abs/2602.03442) • [📥 PDF](https://arxiv.org/pdf/2602.03442)

**💻 Code:** [⭐ Code](https://github.com/Ayanami0730/arag)

> Existing RAG systems rely on Graph or Workflow paradigms that fail to scale with advances in model reasoning and tool-use capabilities. We introduce A-RAG, an Agentic RAG framework that exposes hierarchical retrieval interfaces directly to the mod...

</details>

<details>
<summary><b>22. PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR</b> ⭐ 21</summary>

<br/>

**👥 Authors:** Alejandro Lozano, Jan N. Hansen, yuhuizhang, pengxunduo, jmhb

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18207) • [📄 arXiv](https://arxiv.org/abs/2601.18207) • [📥 PDF](https://arxiv.org/pdf/2601.18207)

**💻 Code:** [⭐ Code](https://github.com/jmhb0/PaperSearchQA)

> Project page: https://jmhb0.github.io/PaperSearchQA/ Data: https://huggingface.co/collections/jmhb/papersearchqa Code for data-gen pipelines: https://github.com/jmhb0/PaperSearchQA

</details>

<details>
<summary><b>23. Horizon-LM: A RAM-Centric Architecture for LLM Training</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04816) • [📄 arXiv](https://arxiv.org/abs/2602.04816) • [📥 PDF](https://arxiv.org/pdf/2602.04816)

**💻 Code:** [⭐ Code](https://github.com/DLYuanGod/Horizon-LM)

> Horizon-LM: Train hundred-billion–parameter language models without buying more GPUs. We propose a RAM-centric, CPU-master training architecture that treats GPUs as transient compute engines rather than persistent parameter stores, enabling large-...

</details>

<details>
<summary><b>24. From Data to Behavior: Predicting Unintended Model Behaviors Before Training</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04735) • [📄 arXiv](https://arxiv.org/abs/2602.04735) • [📥 PDF](https://arxiv.org/pdf/2602.04735)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/Data2Behavior)

> Can we foresee unintended model behaviors before fine-tuning? We demonstrate that unintended biases and safety risks can be traced back to interpretable latent data statistics that mechanistically influence model activations, without any parameter...

</details>

<details>
<summary><b>25. MEnvAgent: Scalable Polyglot Environment Construction for Verifiable Software Engineering</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22859) • [📄 arXiv](https://arxiv.org/abs/2601.22859) • [📥 PDF](https://arxiv.org/pdf/2601.22859)

**💻 Code:** [⭐ Code](https://github.com/ernie-research/MEnvAgent)

> Check out this verifiable environment for SWE! Open-sourced dataset, Docker images, and evals!

</details>

<details>
<summary><b>26. Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04284) • [📄 arXiv](https://arxiv.org/abs/2602.04284) • [📥 PDF](https://arxiv.org/pdf/2602.04284)

**💻 Code:** [⭐ Code](https://github.com/usail-hkust/Agent-Omit)

> Efficient LLM Agents.

</details>

<details>
<summary><b>27. D-CORE: Incentivizing Task Decomposition in Large Reasoning Models for Complex Tool Use</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02160) • [📄 arXiv](https://arxiv.org/abs/2602.02160) • [📥 PDF](https://arxiv.org/pdf/2602.02160)

**💻 Code:** [⭐ Code](https://github.com/alibaba/EfficientAI)

> good job , awesome boys !

</details>

<details>
<summary><b>28. SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03916) • [📄 arXiv](https://arxiv.org/abs/2602.03916) • [📥 PDF](https://arxiv.org/pdf/2602.03916)

**💻 Code:** [⭐ Code](https://github.com/SpatiaLab-Reasoning/SpatiaLab)

> We are excited to share that our paper “𝐒𝐩𝐚𝐭𝐢𝐚𝐋𝐚𝐛: 𝐂𝐚𝐧 𝐕𝐢𝐬𝐢𝐨𝐧–𝐋𝐚𝐧𝐠𝐮𝐚𝐠𝐞 𝐌𝐨𝐝𝐞𝐥𝐬 𝐏𝐞𝐫𝐟𝐨𝐫𝐦 𝐒𝐩𝐚𝐭𝐢𝐚𝐥 𝐑𝐞𝐚𝐬𝐨𝐧𝐢𝐧𝐠 𝐢𝐧 𝐭𝐡𝐞 𝐖𝐢𝐥𝐝?” is accepted to ICLR 2026 (The Fourteenth International Conference on Learning Representations). SpatiaLab investigates how vision...

</details>

<details>
<summary><b>29. Quantifying the Gap between Understanding and Generation within Unified Multimodal Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02140) • [📄 arXiv](https://arxiv.org/abs/2602.02140) • [📥 PDF](https://arxiv.org/pdf/2602.02140)

> A benchmark focuses on quantifying the gap between understanding and generation in unified multimodal model.

</details>

<details>
<summary><b>30. BatCoder: Self-Supervised Bidirectional Code-Documentation Learning via Back-Translation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaohua Wang, Zisu Huang, Yiyang Lu, Jingwen Xu, fdu-lcz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02554) • [📄 arXiv](https://arxiv.org/abs/2602.02554) • [📥 PDF](https://arxiv.org/pdf/2602.02554)

> Training LLMs for code-related tasks typically depends on high-quality code-documentation pairs, which are costly to curate and often scarce for niche programming languages. We introduce BatCoder, a self-supervised reinforcement learning framework...

</details>

<details>
<summary><b>31. Likelihood-Based Reward Designs for General LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03979) • [📄 arXiv](https://arxiv.org/abs/2602.03979) • [📥 PDF](https://arxiv.org/pdf/2602.03979)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API P2S: Probabilistic Process Supervision for General-Domain Reasoning Questio...

</details>

<details>
<summary><b>32. A2Eval: Agentic and Automated Evaluation for Embodied Brain</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01640) • [📄 arXiv](https://arxiv.org/abs/2602.01640) • [📥 PDF](https://arxiv.org/pdf/2602.01640)

> A2Eval introduces an agentic framework that automates embodied VLM evaluation through two collaborative agents: one that curates balanced benchmarks by identifying capability dimensions, and another that synthesizes executable evaluation pipelines...

</details>

<details>
<summary><b>33. Beyond Unimodal Shortcuts: MLLMs as Cross-Modal Reasoners for Grounded Named Entity Recognition</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuwei Wang, Kehai Chen, Xuefeng Bai, Yu Zhang, Jinlong Ma

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04486) • [📄 arXiv](https://arxiv.org/abs/2602.04486) • [📥 PDF](https://arxiv.org/pdf/2602.04486)

> GMNER

</details>

<details>
<summary><b>34. MeKi: Memory-based Expert Knowledge Injection for Efficient LLM Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03359) • [📄 arXiv](https://arxiv.org/abs/2602.03359) • [📥 PDF](https://arxiv.org/pdf/2602.03359)

> We introduce MeKi, a memory-based architecture to scale LLM efficiently. MeKi is able to offload pre-trained token-level expert knowledge to ROM space before deployment. Tested on a Snapdragon mobile platform,  our method achieves superior perform...

</details>

<details>
<summary><b>35. Efficient Autoregressive Video Diffusion with Dummy Head</b> ⭐ 32</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20499) • [📄 arXiv](https://arxiv.org/abs/2601.20499) • [📥 PDF](https://arxiv.org/pdf/2601.20499)

**💻 Code:** [⭐ Code](https://github.com/csguoh/DummyForcing)

> Dummy Forcing is built on the observation that about 25% attention heads in existing autoregressive video diffusion models are "dummy", attending almost exclusively to the current frame despite access to historical context. Based on this observati...

</details>

<details>
<summary><b>36. No One-Size-Fits-All: Building Systems For Translation to Bashkir, Kazakh, Kyrgyz, Tatar and Chuvash Using Synthetic And Original Data</b> ⭐ 0</summary>

<br/>

**👥 Authors:** dimakarp1996

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04442) • [📄 arXiv](https://arxiv.org/abs/2602.04442) • [📥 PDF](https://arxiv.org/pdf/2602.04442)

> We show that effective machine translation for low-resource Turkic languages requires a tailored approach: fine-tuning works best for languages with some data, while retrieval-augmented LLM prompting is essential for extremely resource-scarce ones.

</details>

<details>
<summary><b>37. Context Learning for Multi-Agent Discussion</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02350) • [📄 arXiv](https://arxiv.org/abs/2602.02350) • [📥 PDF](https://arxiv.org/pdf/2602.02350)

**💻 Code:** [⭐ Code](https://github.com/HansenHua/M2CL-ICLR26)

> Try building your own multi-agent system to solve problems!

</details>

<details>
<summary><b>38. Protein Autoregressive Modeling via Multiscale Structure Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04883) • [📄 arXiv](https://arxiv.org/abs/2602.04883) • [📥 PDF](https://arxiv.org/pdf/2602.04883)

> Protein Autoregressive Modeling via Multiscale Structure Generation (PAR) introduces a coarse-to-fine transformer–flow framework for backbone generation with noisy context learning to mitigate exposure bias.

</details>

<details>
<summary><b>39. Skin Tokens: A Learned Compact Representation for Unified Autoregressive Rigging</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shi-Min Hu, Yan-Pei Cao, Meng-Hao Guo, Cheng-Feng Pu, Jia-peng Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04805) • [📄 arXiv](https://arxiv.org/abs/2602.04805) • [📥 PDF](https://arxiv.org/pdf/2602.04805)

> Proposes SkinTokens, a discrete, learnable skinning representation enabling a unified TokenRig autoregressive framework with reinforcement learning fine-tuning to improve rigging accuracy and generalization in 3D animation.

</details>

<details>
<summary><b>40. Self-Rewarding Sequential Monte Carlo for Masked Diffusion Language Models</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Thomas B. Schön, Lidong Bing, Lei Wang, Ziqi Jin, weblzw

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01849) • [📄 arXiv](https://arxiv.org/abs/2602.01849) • [📥 PDF](https://arxiv.org/pdf/2602.01849)

**💻 Code:** [⭐ Code](https://github.com/Algolzw/self-rewarding-smc)

> Self-Rewarding SMC improves sampling for diffusion language models without additional training or external reward guidance.

</details>

<details>
<summary><b>41. SAFE: Stable Alignment Finetuning with Entropy-Aware Predictive Control for RLHF</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dipan Maity

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04651) • [📄 arXiv](https://arxiv.org/abs/2602.04651) • [📥 PDF](https://arxiv.org/pdf/2602.04651)

> An alternative to ppo for RLHF.

</details>

<details>
<summary><b>42. RexBERT: Context Specialized Bidirectional Encoders for E-commerce</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04605) • [📄 arXiv](https://arxiv.org/abs/2602.04605) • [📥 PDF](https://arxiv.org/pdf/2602.04605)

> RexBERT Paper is finally out!

</details>

<details>
<summary><b>43. Trust The Typical</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kanan Gupta, Vikash Singh, Biyao Zhang, Sreehari Sankar, Debargha Ganguly

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04581) • [📄 arXiv](https://arxiv.org/abs/2602.04581) • [📥 PDF](https://arxiv.org/pdf/2602.04581)

> Current approaches to LLM safety fundamentally rely on a brittle cat-and-mouse game of identifying and blocking known threats via guardrails. We argue for a fresh approach: robust safety comes not from enumerating what is harmful, but from deeply ...

</details>

<details>
<summary><b>44. OmniRad: A Radiological Foundation Model for Multi-Task Medical Image Analysis</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Cecilia Di Ruberto, Andrea Loddo, Luca Zedda

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04547) • [📄 arXiv](https://arxiv.org/abs/2602.04547) • [📥 PDF](https://arxiv.org/pdf/2602.04547)

**💻 Code:** [⭐ Code](https://github.com/unica-visual-intelligence-lab/OmniRad)

> OmniRad introduces a self-supervised radiological foundation model pretrained on 1.2M medical images that’s designed for representation reuse across classification, segmentation, and vision–language tasks. The paper shows consistent gains over pri...

</details>

<details>
<summary><b>45. Proxy Compression for Language Modeling</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Lingpeng Kong, Xiachong Feng, Qian Liu, Xinyu Li, Lin Zheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04289) • [📄 arXiv](https://arxiv.org/abs/2602.04289) • [📥 PDF](https://arxiv.org/pdf/2602.04289)

**💻 Code:** [⭐ Code](https://github.com/LZhengisme/proxy-compression)

> This work introduces proxy compression, an alternative training scheme for language models that preserves the efficiency benefits of compression (e.g. tokenization) while providing an end-to-end, byte-level interface at inference time.

</details>

<details>
<summary><b>46. SkeletonGaussian: Editable 4D Generation through Gaussian Skeletonization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04271) • [📄 arXiv](https://arxiv.org/abs/2602.04271) • [📥 PDF](https://arxiv.org/pdf/2602.04271)

> 🚀 Introducing SkeletonGaussian — Editable 4D Generation through Gaussian Skeletonization! (Accepted by CVM 2026) ✨ Generate dynamic 3D Gaussians from text, images, or videos 🦴 Explicit skeleton-driven motion enables intuitive pose editing 🎯 Higher...

</details>

<details>
<summary><b>47. AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03955) • [📄 arXiv](https://arxiv.org/abs/2602.03955) • [📥 PDF](https://arxiv.org/pdf/2602.03955)

**💻 Code:** [⭐ Code](https://github.com/AIFrontierLab/AgentArk)

> Distilling multi-agent intelligence into a single agent. A comprehensive study.

</details>

<details>
<summary><b>48. "I May Not Have Articulated Myself Clearly": Diagnosing Dynamic Instability in LLM Reasoning at Inference Time</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vlado Keselj, Sijia Han, Fengxiang Cheng, Jinkun Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02863) • [📄 arXiv](https://arxiv.org/abs/2602.02863) • [📥 PDF](https://arxiv.org/pdf/2602.02863)

> Large language models often fail during multi-step reasoning, but the failure is usually only observable at the final answer. This paper introduces an inference-time, training-free diagnostic signal for identifying dynamic instability during reaso...

</details>

<details>
<summary><b>49. Reward-free Alignment for Conflicting Objectives</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tianyi Lin, Xi Chen, Xiaopeng Li, Peter Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02495) • [📄 arXiv](https://arxiv.org/abs/2602.02495) • [📥 PDF](https://arxiv.org/pdf/2602.02495)

> Direct alignment methods are increasingly used to align large language models (LLMs) with human preferences. However, many real-world alignment problems involve multiple conflicting objectives, where naive aggregation of preferences can lead to un...

</details>

<details>
<summary><b>50. LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Desen Meng, Xinhao Li, Zihan Jia, Jiaqi Li, hzp

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02341) • [📄 arXiv](https://arxiv.org/abs/2602.02341) • [📥 PDF](https://arxiv.org/pdf/2602.02341)

**💻 Code:** [⭐ Code](https://github.com/MCG-NJU/LongVPO)

> Code: https://github.com/MCG-NJU/LongVPO

</details>

<details>
<summary><b>51. FOTBCD: A Large-Scale Building Change Detection Benchmark from French Orthophotos and Topographic Data</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22596) • [📄 arXiv](https://arxiv.org/abs/2601.22596) • [📥 PDF](https://arxiv.org/pdf/2601.22596)

**💻 Code:** [⭐ Code](https://github.com/abdelpy/FOTBCD-datasets)

> We release FOTBCD, a large-scale French aerial building change detection benchmark (0.2 m), including ~28k binary-labeled pairs and 4k instance-level COCO pairs, plus pretrained weights and code for reproducible training and evaluation.

</details>

<details>
<summary><b>52. HalluHard: A Hard Multi-Turn Hallucination Benchmark</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Maksym Andriushchenko, Nicolas Flammarion, Sebastien Delsad, Dongyang Fan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01031) • [📄 arXiv](https://arxiv.org/abs/2602.01031) • [📥 PDF](https://arxiv.org/pdf/2602.01031)

> LLM hallucinations are far from solved!

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 52 |
| 📅 Today | [`2026-02-06.json`](data/daily/2026-02-06.json) | 52 |
| 📆 This Week | [`2026-W05.json`](data/weekly/2026-W05.json) | 263 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 308 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-06 | 52 | [View JSON](data/daily/2026-02-06.json) |
| 📄 2026-02-05 | 53 | [View JSON](data/daily/2026-02-05.json) |
| 📄 2026-02-04 | 73 | [View JSON](data/daily/2026-02-04.json) |
| 📄 2026-02-03 | 40 | [View JSON](data/daily/2026-02-03.json) |
| 📄 2026-02-02 | 45 | [View JSON](data/daily/2026-02-02.json) |
| 📄 2026-02-01 | 45 | [View JSON](data/daily/2026-02-01.json) |
| 📄 2026-01-31 | 45 | [View JSON](data/daily/2026-01-31.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W05 | 263 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 308 | [View JSON](data/monthly/2026-02.json) |
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
