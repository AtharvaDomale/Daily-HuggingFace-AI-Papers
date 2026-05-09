<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-38-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3755+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">38</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">75</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">134</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3755+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 09, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shugui Liu, Yuchun Miao, Zhengxi Lu, Yuxin Chen, Yaorui Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06130) • [📄 arXiv](https://arxiv.org/abs/2605.06130) • [📥 PDF](https://arxiv.org/pdf/2605.06130)

> Interesting breakdown of this paper on arXivLens: https://arxivlens.com/PaperView/Details/skill1-unified-evolution-of-skill-augmented-agents-via-reinforcement-learning-1299-5558df3d Covers the executive summary, detailed methodology, and practical...

</details>

<details>
<summary><b>2. Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05242) • [📄 arXiv](https://arxiv.org/abs/2605.05242) • [📥 PDF](https://arxiv.org/pdf/2605.05242)

**💻 Code:** [⭐ Code](https://github.com/DCI-Agent/DCI-Agent-Lite)

> 🔥 The best retriever for agentic search … is no retriever. Introducing Direct Corpus Interaction (DCI) . 🚀 We replaced the entire agentic search pipeline — embedding model, vector index, top-k retrieval — with only grep and bash . 🔧 💡The Magic: Th...

</details>

<details>
<summary><b>3. Continuous Latent Diffusion Language Model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rui Zhu, Shen Nie, Yian Zhao, Qinyu Zhao, Hongcan Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06548) • [📄 arXiv](https://arxiv.org/abs/2605.06548) • [📥 PDF](https://arxiv.org/pdf/2605.06548)

> the most interesting bit to me is the two-stage setup: a Text VAE to fix a stable text-to-latent mapping, then a block-causal diffusion transformer to model the latent prior. but i worry a bit about posterior collapse in the vAE and how the kl ter...

</details>

<details>
<summary><b>4. MiA-Signature: Approximating Global Activation for Long-Context Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06416) • [📄 arXiv](https://arxiv.org/abs/2605.06416) • [📥 PDF](https://arxiv.org/pdf/2605.06416)

> We believe this work provides a step toward bridging cognitive insights and practical system design, highlighting the importance of global activation in memory-driven reasoning.

</details>

<details>
<summary><b>5. RaguTeam at SemEval-2026 Task 8: Meno and Friends in a Judge-Orchestrated LLM Ensemble for Faithful Multi-Turn Response Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04523) • [📄 arXiv](https://arxiv.org/abs/2605.04523) • [📥 PDF](https://arxiv.org/pdf/2605.04523)

**💻 Code:** [⭐ Code](https://github.com/RaguTeam/ragu_mtrag_semeval)

> Why this paper is worth your time It’s a practical masterclass in squeezing top-tier RAG performance from a diverse ensemble without betting everything on a single giant. An ensemble of 7 LLMs — from a 7B specialist to frontier systems like Gemini...

</details>

<details>
<summary><b>6. MARBLE: Multi-Aspect Reward Balance for Diffusion RL</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06507) • [📄 arXiv](https://arxiv.org/abs/2605.06507) • [📥 PDF](https://arxiv.org/pdf/2605.06507)

**💻 Code:** [⭐ Code](https://github.com/aim-uofa/MARBLE)

> MARBLE harmonizes reward-specific policy gradients into a single update direction, simultaneously improving all rewards in one training run. No manual reward weighting, no multi-stage curriculum, and at near single-reward training cost. To the bes...

</details>

<details>
<summary><b>7. When to Trust Imagination: Adaptive Action Execution for World Action Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jianan Wang, Kuncheng Luo, Jiehong Lin, Yue Zhang, Rui Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06222) • [📄 arXiv](https://arxiv.org/abs/2605.06222) • [📥 PDF](https://arxiv.org/pdf/2605.06222)

> Can a robot tell when its imagined future is no longer trustworthy? We introduce FFDC, a lightweight verifier for World Action Models that adaptively decides whether to continue executing predicted actions or replan early based on future-reality c...

</details>

<details>
<summary><b>8. Continuous-Time Distribution Matching for Few-Step Diffusion Distillation</b> ⭐ 39</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06376) • [📄 arXiv](https://arxiv.org/abs/2605.06376) • [📥 PDF](https://arxiv.org/pdf/2605.06376)

**💻 Code:** [⭐ Code](https://github.com/byliutao/cdm)

> No abstract available.

</details>

<details>
<summary><b>9. Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuyi Yang, Donghong Cai, Jinyuan Li, Chengsong Huang, Langlin Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05566) • [📄 arXiv](https://arxiv.org/abs/2605.05566) • [📥 PDF](https://arxiv.org/pdf/2605.05566)

> prompt space perturbation broadens reasoning exploration

</details>

<details>
<summary><b>10. SkillOS: Learning Skill Curation for Self-Evolving Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zifeng Wang, Rujun Han, Yanfei Chen, Jun Yan, Siru Ouyang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06614) • [📄 arXiv](https://arxiv.org/abs/2605.06614) • [📥 PDF](https://arxiv.org/pdf/2605.06614)

> Interesting breakdown of this paper on arXivLens: https://arxivlens.com/PaperView/Details/skillos-learning-skill-curation-for-self-evolving-agents-2996-f8487b82 Covers the executive summary, detailed methodology, and practical applications.

</details>

<details>
<summary><b>11. Audio-Visual Intelligence in Large Foundation Models</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04045) • [📄 arXiv](https://arxiv.org/abs/2605.04045) • [📥 PDF](https://arxiv.org/pdf/2605.04045)

**💻 Code:** [⭐ Code](https://github.com/JavisVerse/Awesome-AVI?utm_source=chatgpt.com) • [⭐ Code](https://github.com/JavisVerse/Awesome-AVI)

> 🎧👀 Audio-Visual Intelligence in Large Foundation Models: A Comprehensive Survey 📄 arXiv: 2605.04045 We are excited to release what we believe is the first comprehensive survey on Audio-Visual Intelligence (AVI) in the era of large foundation model...

</details>

<details>
<summary><b>12. StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Philip Torr, Shengji Tang, Zidong Wang, Yifan Zhou, Xiangyuan Xue

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06642) • [📄 arXiv](https://arxiv.org/abs/2605.06642) • [📥 PDF](https://arxiv.org/pdf/2605.06642)

**💻 Code:** [⭐ Code](https://github.com/xxyQwQ/StraTA)

> Hi everyone! 👋 Sharing our latest work: StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction. Should long-horizon LLM agents stay purely reactive? We argue they shouldn't. When the agent has to decide both the...

</details>

<details>
<summary><b>13. Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05724) • [📄 arXiv](https://arxiv.org/abs/2605.05724) • [📥 PDF](https://arxiv.org/pdf/2605.05724)

**💻 Code:** [⭐ Code](https://github.com/cxcscmu/Auto-Research-Recipes)

> Closed-loop auto research turns agent-written code, real experiments, and evaluator feedback into an autonomous feedback loop that develops non-trivial training recipes.

</details>

<details>
<summary><b>14. A^2TGPO: Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06200) • [📄 arXiv](https://arxiv.org/abs/2605.06200) • [📥 PDF](https://arxiv.org/pdf/2605.06200)

**💻 Code:** [⭐ Code](https://github.com/CuSO4-Chen/A-TGPO)

> Overview we propose A²TGPO ( A gentic T urn- G roup P olicy O ptimization with A daptive Turn-level Clipping), which retains IG as the intrinsic signal but re-designs how it is normalized, accumulated, and consumed: (i) turn-group normalization: n...

</details>

<details>
<summary><b>15. Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sipeng Zhang, Xinpeng Wei, Guangchen Lan, Zhaoyang Wang, Tianle Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06638) • [📄 arXiv](https://arxiv.org/abs/2605.06638) • [📥 PDF](https://arxiv.org/pdf/2605.06638)

> 📄 Paper: https://arxiv.org/abs/2605.06638

</details>

<details>
<summary><b>16. UniPool: A Globally Shared Expert Pool for Mixture-of-Experts</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06665) • [📄 arXiv](https://arxiv.org/abs/2605.06665) • [📥 PDF](https://arxiv.org/pdf/2605.06665)

**💻 Code:** [⭐ Code](https://github.com/Centaurus-Alpha/UniPool)

> Modern Mixture-of-Experts (MoE) architectures allocate expert capacity through a rigid per-layer rule: each transformer layer owns a separate expert set. This convention couples depth scaling with linear expert-parameter growth and assumes that ev...

</details>

<details>
<summary><b>17. ReflectDrive-2: Reinforcement-Learning-Aligned Self-Editing for Discrete Diffusion Driving</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ben Lu, Pengxiang Li, Bihao Cui, Yue Wang, Huimin Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04647) • [📄 arXiv](https://arxiv.org/abs/2605.04647) • [📥 PDF](https://arxiv.org/pdf/2605.04647)

> We started ReflectDrive-2 from a concrete observation: when imitation-learned driving policies fail, they fail along two predictable axes — longitudinal (overshoot, late braking, under-progress) and lateral (lane drift, clipped turns). That kind o...

</details>

<details>
<summary><b>18. TabEmbed: Benchmarking and Learning Generalist Embeddings for Tabular Understanding</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yu Cheng, Xing Fu, Xiaoyi Bao, Mingming Zhang, Minjie Qiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04962) • [📄 arXiv](https://arxiv.org/abs/2605.04962) • [📥 PDF](https://arxiv.org/pdf/2605.04962)

**💻 Code:** [⭐ Code](https://github.com/qiangminjie27/TabEmbed)

> the generalist embedding model that unifies tabular classification and retrieval in a single shared space.

</details>

<details>
<summary><b>19. AI Co-Mathematician: Accelerating Mathematicians with Agentic AI</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lars Buesing, Iuliya Beloshapka, Yori Zwols, Ingrid von Glehn, Daniel Zheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06651) • [📄 arXiv](https://arxiv.org/abs/2605.06651) • [📥 PDF](https://arxiv.org/pdf/2605.06651)

> the part that stood out to me is how the workspace keeps a persistent, auditable narrative by logging uncertainty, failed hypotheses, and provenance while outputting native artifacts like living papers and proofs. it's a nice antidote to the usual...

</details>

<details>
<summary><b>20. RemoteZero: Geospatial Reasoning with Zero Human Annotations</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rui Min, Chuanyi Zhang, Shengxiang Xu, Fan Liu, Liang Yao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04451) • [📄 arXiv](https://arxiv.org/abs/2605.04451) • [📥 PDF](https://arxiv.org/pdf/2605.04451)

> We introduce RemoteZero, a box-supervision-free framework for geospatial reasoning. RemoteZero is motivated by a simple asymmetry: an MLLM is typically better at verifying whether a region satisfies a query than at directly generating precise coor...

</details>

<details>
<summary><b>21. SwiftI2V: Efficient High-Resolution Image-to-Video Generation via Conditional Segment-wise Generation</b> ⭐ 18</summary>

<br/>

**👥 Authors:** Rui Liu, Yufei Zhao, Wenbo Li, Yuechen Zhang, YaoYang Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06356) • [📄 arXiv](https://arxiv.org/abs/2605.06356) • [📥 PDF](https://arxiv.org/pdf/2605.06356)

**💻 Code:** [⭐ Code](https://github.com/HKUST-LongGroup/SwiftI2V)

> We propose SwiftI2V, an efficient framework for high-resolution (2K) I2V generation that decouples motion modeling from detail synthesis via progressive segment-wise generation and bidirectional contextual interaction. SwiftI2V achieves performanc...

</details>

<details>
<summary><b>22. The Granularity Axis: A Micro-to-Macro Latent Direction for Social Roles in Language Models</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06196) • [📄 arXiv](https://arxiv.org/abs/2605.06196) • [📥 PDF](https://arxiv.org/pdf/2605.06196)

**💻 Code:** [⭐ Code](https://github.com/qinchonghanzuibang/Granularity-Axis)

> We identify the Granularity Axis, a micro-to-macro latent direction that organizes social-role representations in language models and partially steers the scale of model reasoning.

</details>

<details>
<summary><b>23. EMO: Pretraining Mixture of Experts for Emergent Modularity</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06663) • [📄 arXiv](https://arxiv.org/abs/2605.06663) • [📥 PDF](https://arxiv.org/pdf/2605.06663)

**💻 Code:** [⭐ Code](https://github.com/allenai/EMO)

> Excited to share EMO, a new MoE model pretrained so modular expert structure emerges directly from data, without human-defined domain labels. The key idea is simple: during pretraining, tokens from the same document are constrained to route throug...

</details>

<details>
<summary><b>24. Prescriptive Scaling Laws for Data Constrained Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kilian Q. Weinberger, Shriya Sudhakar, Srivatsa Kundurthy, Christian Belardi, Justin Lovelace

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.01640) • [📄 arXiv](https://arxiv.org/abs/2605.01640) • [📥 PDF](https://arxiv.org/pdf/2605.01640)

> A scaling law that describes language model behavior under data repetition.

</details>

<details>
<summary><b>25. KernelBench-X: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04956) • [📄 arXiv](https://arxiv.org/abs/2605.04956) • [📥 PDF](https://arxiv.org/pdf/2605.04956)

**💻 Code:** [⭐ Code](https://github.com/BonnieW05/KernelBenchX)

> KernelBench-X: A Comprehensive Benchmark for Evaluating LLM-Generated GPU Kernels

</details>

<details>
<summary><b>26. The Scaling Properties of Implicit Deductive Reasoning in Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04330) • [📄 arXiv](https://arxiv.org/abs/2605.04330) • [📥 PDF](https://arxiv.org/pdf/2605.04330)

> code, datasets and models, although reproducible from paper, will be made public upon publication. For joint research, contact {enrico.vompa}@gmail.com as I'm open for collaboration

</details>

<details>
<summary><b>27. Balanced Aggregation: Understanding and Fixing Aggregation Bias in GRPO</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04077) • [📄 arXiv](https://arxiv.org/abs/2605.04077) • [📥 PDF](https://arxiv.org/pdf/2605.04077)

> This work reveals distinct optimization biases in sequence and token aggregation for GRPO-style RLVR, proposes a simple Balanced Aggregation plug-in method to mitigate their flaws

</details>

<details>
<summary><b>28. Are We Making Progress in Multimodal Domain Generalization? A Comprehensive Benchmark Study</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06643) • [📄 arXiv](https://arxiv.org/abs/2605.06643) • [📥 PDF](https://arxiv.org/pdf/2605.06643)

**💻 Code:** [⭐ Code](https://github.com/lihongzhao99/MMDG_Benchmark)

> MMDG-Bench is the first comprehensive and standardized benchmark for Multimodal Domain Generalization (MMDG). Unlike prior work that focuses on limited datasets or settings, MMDG-Bench unifies evaluation across multiple tasks, modalities, and real...

</details>

<details>
<summary><b>29. Think, then Score: Decoupled Reasoning and Scoring for Video Reward Modeling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiajun Liang, Borui Liao, Yulong Xu, Ouxiang Li, Yuan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05922) • [📄 arXiv](https://arxiv.org/abs/2605.05922) • [📥 PDF](https://arxiv.org/pdf/2605.05922)

> Interesting breakdown of this paper on arXivLens: https://arxivlens.com/PaperView/Details/think-then-score-decoupled-reasoning-and-scoring-for-video-reward-modeling-9789-4f677445 Covers the executive summary, detailed methodology, and practical ap...

</details>

<details>
<summary><b>30. PianoCoRe: Combined and Refined Piano MIDI Dataset</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06627) • [📄 arXiv](https://arxiv.org/abs/2605.06627) • [📥 PDF](https://arxiv.org/pdf/2605.06627)

**💻 Code:** [⭐ Code](https://github.com/ilya16/PianoCoRe) • [⭐ Code](https://github.com/ilya16/SyMuPe)

> PianoCoRe is a large-scale piano MIDI dataset of annotated and note-aligned scores and performances. It unifies multiple open-source corpora into a single, ready-to-use resource for piano performance modeling. 🎼 250,046 performances (21k+ hours) c...

</details>

<details>
<summary><b>31. GeoStack: A Framework for Quasi-Abelian Knowledge Composition in VLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06477) • [📄 arXiv](https://arxiv.org/abs/2605.06477) • [📥 PDF](https://arxiv.org/pdf/2605.06477)

**💻 Code:** [⭐ Code](https://github.com/QuantitativeImagingLaboratory/GeoStack)

> How many domain experts can you stack before a VLM collapses? 🧱 GeoStack introduces a geometric framework to compose independently trained experts into a single model with zero added inference cost. By using a perturbation prior and orthogonality ...

</details>

<details>
<summary><b>32. When No Benchmark Exists: Validating Comparative LLM Safety Scoring Without Ground-Truth Labels</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06652) • [📄 arXiv](https://arxiv.org/abs/2605.06652) • [📥 PDF](https://arxiv.org/pdf/2605.06652)

**💻 Code:** [⭐ Code](https://github.com/kelkalot/simpleaudit)

> Many deployments must compare candidate language models for safety before a labeled benchmark exists for the relevant language, sector, or regulatory regime. We formalize this setting as benchmarkless comparative safety scoring and specify the con...

</details>

<details>
<summary><b>33. Generative Quantum-inspired Kolmogorov-Arnold Eigensolver</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kuo-Chung Peng, Chun-Hua Lin, I-Shan Tsai, Yu-Chao Hsu, Yu-Cheng Lin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04604) • [📄 arXiv](https://arxiv.org/abs/2605.04604) • [📥 PDF](https://arxiv.org/pdf/2605.04604)

> GQKAE introduces a generative quantum-inspired Kolmogorov–Arnold eigensolver for quantum chemistry, replacing the parameter-heavy FFN layers in GPT-style GQE with compact hybrid quantum-inspired Kolmogorov-Arnold network (HQKAN) modules while pres...

</details>

<details>
<summary><b>34. Recovering Hidden Reward in Diffusion-Based Policies</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenyuan Xie, Shaokai Wu, Yuting Hu, Qiuchang Li, sotaagi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.00623) • [📄 arXiv](https://arxiv.org/abs/2605.00623) • [📥 PDF](https://arxiv.org/pdf/2605.00623)

> ENERGYFLOW unifies diffusion-based imitation learning and inverse reinforcement learning by learning a conservative energy field whose gradient drives action generation while exposing a recoverable reward signal, improving manipulation performance...

</details>

<details>
<summary><b>35. BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05758) • [📄 arXiv](https://arxiv.org/abs/2605.05758) • [📥 PDF](https://arxiv.org/pdf/2605.05758)

**💻 Code:** [⭐ Code](https://github.com/gxx27/BioTool)

> Published at ACL 2026; Code and data available at https://github.com/gxx27/BioTool

</details>

<details>
<summary><b>36. TIDE: Every Layer Knows the Token Beneath the Context</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06216) • [📄 arXiv](https://arxiv.org/abs/2605.06216) • [📥 PDF](https://arxiv.org/pdf/2605.06216)

> We investigate rare token and contextual collapse problem in LLM design and propose to inject token identity information to each transformer layer.

</details>

<details>
<summary><b>37. Sparkle: Realizing Lively Instruction-Guided Video Background Replacement via Decoupled Guidance</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06535) • [📄 arXiv](https://arxiv.org/abs/2605.06535) • [📥 PDF](https://arxiv.org/pdf/2605.06535)

**💻 Code:** [⭐ Code](https://github.com/showlab/Sparkle)

> In recent years, open-source efforts like Señorita-2M have propelled video editing toward natural language instruction. However, current publicly available datasets predominantly focus on local editing or style transfer, which largely preserve the...

</details>

<details>
<summary><b>38. EDU-CIRCUIT-HW: Evaluating Multimodal Large Language Models on Real-World University-Level STEM Student Handwritten Solutions</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Yi Zeng, Huiru Xie, Yongnuo Cai, Liangliang Chen, SWY666

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00095) • [📄 arXiv](https://arxiv.org/abs/2602.00095) • [📥 PDF](https://arxiv.org/pdf/2602.00095)

**💻 Code:** [⭐ Code](https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL)

> Hello all! Thanks for taking a look at our work - "EDU-CIRCUIT-HW: Evaluating Multimodal Large Language Models on Real-World University-Level STEM Student Handwritten Solutions" This paper investigates how MLLMs handle authentic university-level S...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 38 |
| 📅 Today | [`2026-05-09.json`](data/daily/2026-05-09.json) | 38 |
| 📆 This Week | [`2026-W18.json`](data/weekly/2026-W18.json) | 75 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 134 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-09 | 38 | [View JSON](data/daily/2026-05-09.json) |
| 📄 2026-05-08 | 18 | [View JSON](data/daily/2026-05-08.json) |
| 📄 2026-05-07 | 3 | [View JSON](data/daily/2026-05-07.json) |
| 📄 2026-05-06 | 8 | [View JSON](data/daily/2026-05-06.json) |
| 📄 2026-05-05 | 3 | [View JSON](data/daily/2026-05-05.json) |
| 📄 2026-05-04 | 5 | [View JSON](data/daily/2026-05-04.json) |
| 📄 2026-05-03 | 24 | [View JSON](data/daily/2026-05-03.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W18 | 75 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |
| 📅 2026-W15 | 99 | [View JSON](data/weekly/2026-W15.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 134 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
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
