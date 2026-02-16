<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-41-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2255+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">41</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">736</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2255+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 16, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. The Devil Behind Moltbook: Anthropic Safety is Always Vanishing in Self-Evolving AI Societies</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jinyu Hou, Zejian Chen, Songyang Liu, Chaozhuo Li, xunyoyo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09877) • [📄 arXiv](https://arxiv.org/abs/2602.09877) • [📥 PDF](https://arxiv.org/pdf/2602.09877)

> The emergence of multi-agent systems built from large language models (LLMs) offers a promising paradigm for scalable collective intelligence and self-evolution. Ideally, such systems would achieve continuous self-improvement in a fully closed loo...

</details>

<details>
<summary><b>2. Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12036) • [📄 arXiv](https://arxiv.org/abs/2602.12036) • [📥 PDF](https://arxiv.org/pdf/2602.12036)

**💻 Code:** [⭐ Code](https://github.com/XinXU-USTC/Composition-RL)

> Models and datasets are available at https://huggingface.co/collections/xx18/composition-rl

</details>

<details>
<summary><b>3. DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing</b> ⭐ 86</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12205) • [📄 arXiv](https://arxiv.org/abs/2602.12205) • [📥 PDF](https://arxiv.org/pdf/2602.12205)

**💻 Code:** [⭐ Code](https://github.com/DeepGenTeam/DeepGen)

> Models: https://huggingface.co/deepgenteam/DeepGen-1.0 Datasets: https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0

</details>

<details>
<summary><b>4. Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12125) • [📄 arXiv](https://arxiv.org/abs/2602.12125) • [📥 PDF](https://arxiv.org/pdf/2602.12125)

**💻 Code:** [⭐ Code](https://github.com/RUCBM/G-OPD)

> We propose G-OPD, a generalized on-policy distillation framework. Building on G-OPD, we propose ExOPD (Generalized On-Policy Distillation with Reward Extrapolation), which enables a unified student to surpass all domain teachers in the multi-teach...

</details>

<details>
<summary><b>5. GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning</b> ⭐ 2.31k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12099) • [📄 arXiv](https://arxiv.org/abs/2602.12099) • [📥 PDF](https://arxiv.org/pdf/2602.12099)

**💻 Code:** [⭐ Code](https://github.com/open-gigaai/giga-brain-0)

> GigaBrain-0.5M* is a VLA That Learns From World Model-Based Reinforcement Learning. GigaBrain-0.5M* exhibits reliable long-horizon execution, consistently accomplishing complex manipulation tasks without failure.

</details>

<details>
<summary><b>6. MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models</b> ⭐ 99</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10934) • [📄 arXiv](https://arxiv.org/abs/2602.10934) • [📥 PDF](https://arxiv.org/pdf/2602.10934)

**💻 Code:** [⭐ Code](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer)

> Start discussion in this paper

</details>

<details>
<summary><b>7. NarraScore: Bridging Visual Narrative and Musical Dynamics via Hierarchical Affective Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09070) • [📄 arXiv](https://arxiv.org/abs/2602.09070) • [📥 PDF](https://arxiv.org/pdf/2602.09070)

> No abstract available.

</details>

<details>
<summary><b>8. LawThinker: A Deep Research Legal Agent in Dynamic Environments</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12056) • [📄 arXiv](https://arxiv.org/abs/2602.12056) • [📥 PDF](https://arxiv.org/pdf/2602.12056)

**💻 Code:** [⭐ Code](https://github.com/yxy-919/LawThinker-agent)

> Legal reasoning requires not only correct outcomes but also procedurally compliant reasoning processes. However, existing methods lack mechanisms to verify intermediate reasoning steps, allowing errors such as inapplicable statute citations to pro...

</details>

<details>
<summary><b>9. Thinking with Drafting: Optical Decompression via Logical Reconstruction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11731) • [📄 arXiv](https://arxiv.org/abs/2602.11731) • [📥 PDF](https://arxiv.org/pdf/2602.11731)

> The core idea of Thinking with Drafting (TwD) is super refreshing: instead of letting a multimodal model “guess the answer” with fluent CoT or pretty-looking diagrams, it forces the model to draft its reasoning into executable structure. Not vibes...

</details>

<details>
<summary><b>10. Stroke of Surprise: Progressive Semantic Illusions in Vector Sketching</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12280) • [📄 arXiv](https://arxiv.org/abs/2602.12280) • [📥 PDF](https://arxiv.org/pdf/2602.12280)

**💻 Code:** [⭐ Code](https://github.com/stroke-of-surprise/Stroke-Of-Surprise)

> Visual illusions traditionally rely on spatial manipulations such as multi-view consistency. In this work, we introduce Progressive Semantic Illusions, a novel vector sketching task where a single sketch undergoes a dramatic semantic transformatio...

</details>

<details>
<summary><b>11. Think Longer to Explore Deeper: Learn to Explore In-Context via Length-Incentivized Reinforcement Learning</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11748) • [📄 arXiv](https://arxiv.org/abs/2602.11748) • [📥 PDF](https://arxiv.org/pdf/2602.11748)

**💻 Code:** [⭐ Code](https://github.com/LINs-lab/LIE)

> 🔗 Code: https://github.com/LINs-lab/LIE 🔗 Paper: https://arxiv.org/abs/2602.11748

</details>

<details>
<summary><b>12. RISE: Self-Improving Robot Policy with Compositional World Model</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11075) • [📄 arXiv](https://arxiv.org/abs/2602.11075) • [📥 PDF](https://arxiv.org/pdf/2602.11075)

> The first study on leveraging world models as an effective learning environment for challenging real-world manipulation, bootstrapping performance on tasks requiring high dynamics, dexterity, and precision.

</details>

<details>
<summary><b>13. χ_{0}: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies</b> ⭐ 187</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09021) • [📄 arXiv](https://arxiv.org/abs/2602.09021) • [📥 PDF](https://arxiv.org/pdf/2602.09021)

**💻 Code:** [⭐ Code](https://github.com/OpenDriveLab/KAI0)

> 🧥 Live-stream robotic teamwork that folds clothes. 6 clothes in 3 minutes straight. χ₀ = 20hrs data + 8 A100s + 3 key insights: Mode Consistency: align your distributions Model Arithmetic: merge, don't retrain Stage Advantage: pivot wisely 🔗 http:...

</details>

<details>
<summary><b>14. EgoHumanoid: Unlocking In-the-Wild Loco-Manipulation with Robot-Free Egocentric Demonstration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yinghui Li, Haoran Jiang, Jin Chen, Shijia Peng, Modi Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10106) • [📄 arXiv](https://arxiv.org/abs/2602.10106) • [📥 PDF](https://arxiv.org/pdf/2602.10106)

> Project page: https://opendrivelab.com/EgoHumanoid

</details>

<details>
<summary><b>15. dVoting: Fast Voting for dLLMs</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12153) • [📄 arXiv](https://arxiv.org/abs/2602.12153) • [📥 PDF](https://arxiv.org/pdf/2602.12153)

**💻 Code:** [⭐ Code](https://github.com/fscdc/dVoting)

> The first efficient test-time scaling strategy for dLLMs. Welcome any discussion!

</details>

<details>
<summary><b>16. Sparse Video Generation Propels Real-World Beyond-the-View Vision-Language Navigation</b> ⭐ 35</summary>

<br/>

**👥 Authors:** Yukuan Xu, Yuxian Li, Li Chen, Siqi Liang, Hai Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05827) • [📄 arXiv](https://arxiv.org/abs/2602.05827) • [📥 PDF](https://arxiv.org/pdf/2602.05827)

**💻 Code:** [⭐ Code](https://github.com/opendrivelab/sparsevideonav)

> SparseVideoNav introduces video generation models to real-world beyond-the-view vision-language navigation for the first time. It achieves sub-second trajectory inference with a sparse future spanning a 20-second horizon, yielding a remarkable 27×...

</details>

<details>
<summary><b>17. Voxtral Realtime</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11298) • [📄 arXiv](https://arxiv.org/abs/2602.11298) • [📥 PDF](https://arxiv.org/pdf/2602.11298)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Streaming Speech Recognition with Decoder-Only Large Language Models and La...

</details>

<details>
<summary><b>18. DeepSight: An All-in-One LM Safety Toolkit</b> ⭐ 35</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12092) • [📄 arXiv](https://arxiv.org/abs/2602.12092) • [📥 PDF](https://arxiv.org/pdf/2602.12092)

**💻 Code:** [⭐ Code](https://github.com/AI45Lab/DeepSafe) • [⭐ Code](https://github.com/AI45Lab/DeepScan/)

> We propose an open-source project, namely DeepSight, to practice a new safety evaluation-diagnosis integrated paradigm.

</details>

<details>
<summary><b>19. Unveiling Implicit Advantage Symmetry: Why GRPO Struggles with Exploration and Difficulty Adaptation</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05548) • [📄 arXiv](https://arxiv.org/abs/2602.05548) • [📥 PDF](https://arxiv.org/pdf/2602.05548)

**💻 Code:** [⭐ Code](https://github.com/HKU-HealthAI/A-GRAE)

> See https://github.com/HKU-HealthAI/A-GRAE for the code base, and https://yu7-code.github.io/A-GRAE-web/ for the project page

</details>

<details>
<summary><b>20. Adapting Vision-Language Models for E-commerce Understanding at Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11733) • [📄 arXiv](https://arxiv.org/abs/2602.11733) • [📥 PDF](https://arxiv.org/pdf/2602.11733)

> Figure 1: Output of our E-commerce Adapted VLMs compared against same size LLaVA-OneVision . We show our models ability to more faithfully extract attributes from e-commerce items. In red, we highlight wrong model predictions that are neither tied...

</details>

<details>
<summary><b>21. PISCO: Precise Video Instance Insertion with Sparse Control</b> ⭐ 32</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08277) • [📄 arXiv](https://arxiv.org/abs/2602.08277) • [📥 PDF](https://arxiv.org/pdf/2602.08277)

**💻 Code:** [⭐ Code](https://github.com/taco-group/PISCO)

> PISCO: Precise Video Instance Insertion with Sparse Control

</details>

<details>
<summary><b>22. Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11964) • [📄 arXiv](https://arxiv.org/abs/2602.11964) • [📥 PDF](https://arxiv.org/pdf/2602.11964)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API IDRBench: Interactive Deep Research Benchmark (2026) Agent World Model: Inf...

</details>

<details>
<summary><b>23. T3D: Few-Step Diffusion Language Models via Trajectory Self-Distillation with Direct Discriminative Optimization</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Xiaoxiao He, Haizhou Shi, Ligong Han, Xinxi Zhang, Tyrion279

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12262) • [📄 arXiv](https://arxiv.org/abs/2602.12262) • [📥 PDF](https://arxiv.org/pdf/2602.12262)

**💻 Code:** [⭐ Code](https://github.com/Tyrion58/T3D)

> Diffusion large language models (DLLMs) have the potential to enable fast text generation by decoding multiple tokens in parallel. However, in practice, their inference efficiency is constrained by the need for many refinement steps, while aggress...

</details>

<details>
<summary><b>24. MemFly: On-the-Fly Memory Optimization via Information Bottleneck</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wei Xue, Zhenbo Song, Zhiqin Yang, Xianzhang Jia, Zhenyuan Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07885) • [📄 arXiv](https://arxiv.org/abs/2602.07885) • [📥 PDF](https://arxiv.org/pdf/2602.07885)

> No abstract available.

</details>

<details>
<summary><b>25. Single-minus gluon tree amplitudes are nonzero</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12176) • [📄 arXiv](https://arxiv.org/abs/2602.12176) • [📥 PDF](https://arxiv.org/pdf/2602.12176)

> A group of theoretical physicists derive a new result in quantum field theory using GPT-5.2 Pro.

</details>

<details>
<summary><b>26. ThinkRouter: Efficient Reasoning via Routing Thinking between Latent and Discrete Spaces</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Julian McAuley, Haoliang Wang, Xiang Chen, Tong Yu, XinXuNLPer

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11683) • [📄 arXiv](https://arxiv.org/abs/2602.11683) • [📥 PDF](https://arxiv.org/pdf/2602.11683)

> This paper proposes ThinkRouter, a confidence-aware routing mechanism to improve reasoning performance for large reasoning models (LRMs), which routes LRMs thinking between latent and discrete token spaces based on model confidence at inference time.

</details>

<details>
<summary><b>27. Dreaming in Code for Curriculum Learning in Open-Ended Worlds</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08194) • [📄 arXiv](https://arxiv.org/abs/2602.08194) • [📥 PDF](https://arxiv.org/pdf/2602.08194)

**💻 Code:** [⭐ Code](https://github.com/konstantinosmitsides/dreaming-in-code)

> Large Language Models that "dream" and materialize executable environment code to scaffold learning in open-ended worlds.

</details>

<details>
<summary><b>28. MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11761) • [📄 arXiv](https://arxiv.org/abs/2602.11761) • [📥 PDF](https://arxiv.org/pdf/2602.11761)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Hybrid Linear Attention Done Right: Efficient Distillation and Effective Ar...

</details>

<details>
<summary><b>29. MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation</b> ⭐ 102</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11337) • [📄 arXiv](https://arxiv.org/abs/2602.11337) • [📥 PDF](https://arxiv.org/pdf/2602.11337)

**💻 Code:** [⭐ Code](https://github.com/allenai/molmospaces)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes (2026) Gen...

</details>

<details>
<summary><b>30. Pretraining A Large Language Model using Distributed GPUs: A Memory-Efficient Decentralized Paradigm</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11543) • [📄 arXiv](https://arxiv.org/abs/2602.11543) • [📥 PDF](https://arxiv.org/pdf/2602.11543)

**💻 Code:** [⭐ Code](https://github.com/zjr2000/SPES)

> We propose SPES, a decentralized framework for pretraining MoE LLMs. SPES supports sparse training on weakly connected nodes, reducing memory and communication costs and enabling efficient pretraining on resource-constrained devices.

</details>

<details>
<summary><b>31. MetaphorStar: Image Metaphor Understanding and Reasoning with End-to-End Visual Reinforcement Learning</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Hongsheng Li, Yazhe Niu, Chenhao Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10575) • [📄 arXiv](https://arxiv.org/abs/2602.10575) • [📥 PDF](https://arxiv.org/pdf/2602.10575)

**💻 Code:** [⭐ Code](https://github.com/MING-ZCH/MetaphorStar)

> Metaphorical comprehension in images remains a critical challenge for Nowadays AI systems. While Multimodal Large Language Models (MLLMs) excel at basic Visual Question Answering (VQA), they consistently struggle to grasp the nuanced cultural, emo...

</details>

<details>
<summary><b>32. ExStrucTiny: A Benchmark for Schema-Variable Structured Information Extraction from Document Images</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12203) • [📄 arXiv](https://arxiv.org/abs/2602.12203) • [📥 PDF](https://arxiv.org/pdf/2602.12203)

> We introduce ExStrucTiny, a new benchmark for structured information extraction from document images that unifies in one task (1) key entity extraction, (2) relation extraction, and (3) visual question answering across diverse input schemas and do...

</details>

<details>
<summary><b>33. Sci-CoE: Co-evolving Scientific Reasoning LLMs via Geometric Consensus with Sparse Supervision</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12164) • [📄 arXiv](https://arxiv.org/abs/2602.12164) • [📥 PDF](https://arxiv.org/pdf/2602.12164)

**💻 Code:** [⭐ Code](https://github.com/InternScience/Sci-CoE)

> We’re excited to share our new work, Sci-CoE ! 🎉 In this project, we tackle a fundamental challenge: 🧩 How can we train LLMs with RL when there is no explicit final answer and no way to compute outcome rewards via unit tests or exact matching? Thi...

</details>

<details>
<summary><b>34. P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12116) • [📄 arXiv](https://arxiv.org/abs/2602.12116) • [📥 PDF](https://arxiv.org/pdf/2602.12116)

**💻 Code:** [⭐ Code](https://github.com/Tongyi-ConvAI/Qwen-Character)

> Personalized alignment of large language models seeks to adapt responses to individual user preferences, typically via reinforcement learning. A key challenge is obtaining accurate, user-specific reward signals in open-ended scenarios. Existing pe...

</details>

<details>
<summary><b>35. Budget-Constrained Agentic Large Language Models: Intention-Based Planning for Costly Tool Use</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11541) • [📄 arXiv](https://arxiv.org/abs/2602.11541) • [📥 PDF](https://arxiv.org/pdf/2602.11541)

> Don't Let Your Agent Max Out Your Credit Card! 💳 Most current research pushes the boundaries of agent performance but often overlooks the actual economic cost. Can agents still make rational decisions when every tool call comes with a price tag? W...

</details>

<details>
<summary><b>36. Multimodal Fact-Level Attribution for Verifiable Reasoning</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Hyunji Lee, Elias Stengel-Eskin, Ziyang Wang, David Wan, HanNight

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11509) • [📄 arXiv](https://arxiv.org/abs/2602.11509) • [📥 PDF](https://arxiv.org/pdf/2602.11509)

**💻 Code:** [⭐ Code](https://github.com/meetdavidwan/murgat)

> No abstract available.

</details>

<details>
<summary><b>37. ScalSelect: Scalable Training-Free Multimodal Data Selection for Efficient Visual Instruction Tuning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11636) • [📄 arXiv](https://arxiv.org/abs/2602.11636) • [📥 PDF](https://arxiv.org/pdf/2602.11636)

**💻 Code:** [⭐ Code](https://github.com/ChangtiWu/ScalSelect)

> ScalSelect: Training-Free and Scalable Data Selection for Visual Instruction Tuning Large-scale visual instruction tuning datasets are highly redundant, yet full-data training remains the default — leading to substantial computational waste. This ...

</details>

<details>
<summary><b>38. ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation</b> ⭐ 40</summary>

<br/>

**👥 Authors:** Minghua Luo, Yanfen Shen, Xiaolong Wu, Shichao Xie, Zedong Chu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11598) • [📄 arXiv](https://arxiv.org/abs/2602.11598) • [📥 PDF](https://arxiv.org/pdf/2602.11598)

**💻 Code:** [⭐ Code](https://github.com/amap-cvlab/ABot-Navigation)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aeria...

</details>

<details>
<summary><b>39. Neural Additive Experts: Context-Gated Experts for Controllable Model Additivity</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Aidong Zhang, Sanchit Sinha, Guangzhi Xiong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10585) • [📄 arXiv](https://arxiv.org/abs/2602.10585) • [📥 PDF](https://arxiv.org/pdf/2602.10585)

**💻 Code:** [⭐ Code](https://github.com/Teddy-XiongGZ/NAE)

> TL;DR: We introduce Neural Additive Experts (NAEs), a context-gated mixture-of-experts extension of generalized additive models that preserves per-feature explanations while capturing interactions when needed, using a single tunable regularizer to...

</details>

<details>
<summary><b>40. Stemphonic: All-at-once Flexible Multi-stem Music Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09891) • [📄 arXiv](https://arxiv.org/abs/2602.09891) • [📥 PDF](https://arxiv.org/pdf/2602.09891)

> Check out intro & demo video here! https://youtu.be/IrGD3CHaPYU?si=nutPyU5sz5iHfES5 More sound examples: https://stemphonic-demo.vercel.app

</details>

<details>
<summary><b>41. Detecting RLVR Training Data via Structural Convergence of Reasoning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11792) • [📄 arXiv](https://arxiv.org/abs/2602.11792) • [📥 PDF](https://arxiv.org/pdf/2602.11792)

**💻 Code:** [⭐ Code](https://github.com/StevenZHB/Detect_RLVR_Data)

> Reinforcement learning with verifiable rewards (RLVR) is central to training modern reasoning models, but the undisclosed training data raises concerns about benchmark contamination. Unlike pretraining methods, which optimize models using token-le...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 41 |
| 📅 Today | [`2026-02-16.json`](data/daily/2026-02-16.json) | 41 |
| 📆 This Week | [`2026-W07.json`](data/weekly/2026-W07.json) | 41 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 736 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-16 | 41 | [View JSON](data/daily/2026-02-16.json) |
| 📄 2026-02-15 | 41 | [View JSON](data/daily/2026-02-15.json) |
| 📄 2026-02-14 | 41 | [View JSON](data/daily/2026-02-14.json) |
| 📄 2026-02-13 | 47 | [View JSON](data/daily/2026-02-13.json) |
| 📄 2026-02-12 | 57 | [View JSON](data/daily/2026-02-12.json) |
| 📄 2026-02-11 | 58 | [View JSON](data/daily/2026-02-11.json) |
| 📄 2026-02-10 | 2 | [View JSON](data/daily/2026-02-10.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W07 | 41 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 736 | [View JSON](data/monthly/2026-02.json) |
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
