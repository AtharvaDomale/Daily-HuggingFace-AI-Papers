<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-38-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5457+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">112</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">76</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5457+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 02, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. AskChem: Claim-Centered Infrastructure for Chemistry Literature Synthesis</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28618) • [📄 arXiv](https://arxiv.org/abs/2607.28618) • [📥 PDF](https://arxiv.org/pdf/2607.28618)

**💻 Code:** [⭐ Code](https://github.com/bingyan4science/askchem) • [⭐ Code](https://github.com/huggingface)

> Working on chemistry? What if you could search chemistry findings instead of papers? We turned 147,000 papers into 2.4 million searchable claims , making it possible to find results, compare evidence, and surface contradictions. Try AskChem: 🔎 htt...

</details>

<details>
<summary><b>2. Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28227) • [📄 arXiv](https://arxiv.org/abs/2607.28227) • [📥 PDF](https://arxiv.org/pdf/2607.28227)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tongyi-MAI/MAI-UI)

> We present Qwen-UI-Agent, a real-world centric foundation GUI agent unifying mobile, computer, browser, and DeepSearch scenarios in a single model. It sets a new SOTA on mobile use and stays competitive with Opus 4.8, Gemini 3.1 Pro, and GPT-5.6 S...

</details>

<details>
<summary><b>3. Metis: Memory Foundation Model</b> ⭐ 51</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26760) • [📄 arXiv](https://arxiv.org/abs/2607.26760) • [📥 PDF](https://arxiv.org/pdf/2607.26760)

**💻 Code:** [⭐ Code](https://github.com/MemTensor/Metis) • [⭐ Code](https://github.com/huggingface)

> Metis : The first prototype of a memory foundation model, equipping foundation models with a persistent and dynamically evolving native memory state. What if memory were a native capability of foundation models, rather than an external module? Rec...

</details>

<details>
<summary><b>4. Frontis-MA1: Training an AI4AI Model towards Recursive Self-Improvement in Machine Learning Engineering</b> ⭐ 103</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28568) • [📄 arXiv](https://arxiv.org/abs/2607.28568) • [📥 PDF](https://arxiv.org/pdf/2607.28568)

**💻 Code:** [⭐ Code](https://github.com/FrontisAI/OpenRSI) • [⭐ Code](https://github.com/huggingface)

> Recursive self-improvement (RSI) requires AI systems that improve the process of building AI (i.e., AI4AI); machine learning engineering (MLE) offers a concrete, executable testbed for studying this capability. We introduce OpenMLE, an open full-s...

</details>

<details>
<summary><b>5. PhiZero: A World Model Built Around Physical Language</b> ⭐ 37</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28624) • [📄 arXiv](https://arxiv.org/abs/2607.28624) • [📥 PDF](https://arxiv.org/pdf/2607.28624)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yaoyao-jpg/PhiZero)

> No abstract available.

</details>

<details>
<summary><b>6. VideoCoCo: Code-as-CoT for Physically-Consistent Video Generation via an Agentic Dual-Engine System</b> ⭐ 65</summary>

<br/>

**👥 Authors:** Xiaoxiao Ma, Haodong Li, Lin-Chen, Juanxi, rentianfei122

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27380) • [📄 arXiv](https://arxiv.org/abs/2607.27380) • [📥 PDF](https://arxiv.org/pdf/2607.27380)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/micky-li-hd/VideoCoCo)

> 🧠 VideoCoCo uses executable Blender code as process-level CoT✨, generating deterministic spatiotemporal drafts that guide a video editor toward photorealistic and physically consistent results. 🎬

</details>

<details>
<summary><b>7. Memory Decoder at Scale: A Pretrained, Parametric Long-Term Memory</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Qipeng Guo, Junming Zhang, Jiarui Wang, Jiaqi Cao, Rubin-Wei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27919) • [📄 arXiv](https://arxiv.org/abs/2607.27919) • [📥 PDF](https://arxiv.org/pdf/2607.27919)

**💻 Code:** [⭐ Code](https://github.com/LUMIA-Group/MemoryDecoder-at-Scale) • [⭐ Code](https://github.com/huggingface)

> We introduce Memory Decoder at Scale, which disentangles long-term memory from reasoning and scales parametric memory to 6.9B parameters and 300B training tokens. Pythia-410M + Mem-6.9B surpasses Pythia-12B across 17 tasks with 39% fewer total par...

</details>

<details>
<summary><b>8. Beacon: Knowing When and How to Perform Agentic Visual Reasoning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28595) • [📄 arXiv](https://arxiv.org/abs/2607.28595) • [📥 PDF](https://arxiv.org/pdf/2607.28595)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NOVAglow646/Beacon)

> The fundamental goal of agentic visual reasoning is to improve the success rate of multimodal large language models (MLLMs) on complex tasks, rather than merely equipping them with a sophisticated yet inefficient reasoning paradigm. In this work, ...

</details>

<details>
<summary><b>9. BM25 Wins at Scale: A Scaling Study of Retrieval-Augmented Generation Paradigms</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26497) • [📄 arXiv](https://arxiv.org/abs/2607.26497) • [📥 PDF](https://arxiv.org/pdf/2607.26497)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> hi! Thanks for supporting!

</details>

<details>
<summary><b>10. Flux-OPD: On-Policy Distillation with Evolving Contexts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28022) • [📄 arXiv](https://arxiv.org/abs/2607.28022) • [📥 PDF](https://arxiv.org/pdf/2607.28022)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language model training in open-ended domains lacks verifiable rewards, making task preferences difficult to formalize as effective supervision. Contexts can convey such preferences, yet provide little additional supervision once distilled i...

</details>

<details>
<summary><b>11. MPIE-Bench: Benchmarking Anatomically Plausible Multi-Person Interaction Editing</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27616) • [📄 arXiv](https://arxiv.org/abs/2607.27616) • [📥 PDF](https://arxiv.org/pdf/2607.27616)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AnnLin0628/mpie-bench)

> Text-to-image and personalized editing models now synthesize high-fidelity single-subject images with ease. Yet placing multiple named people into shared contact actions such as embrace, carry, or grapple still exposes major failures: fused limbs,...

</details>

<details>
<summary><b>12. ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yukang Cao, ChaosLiao, yaorunmao, wenbc21, hzxie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28625) • [📄 arXiv](https://arxiv.org/abs/2607.28625) • [📥 PDF](https://arxiv.org/pdf/2607.28625)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Human-as-Humanoid: Enabling Zero-Shot Humanoid Learning from Ego-Exo Human ...

</details>

<details>
<summary><b>13. Beyond Borrowed Histories: Person-Aligned User Simulation for Interactive Role-Playing Evaluation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27816) • [📄 arXiv](https://arxiv.org/abs/2607.27816) • [📥 PDF](https://arxiv.org/pdf/2607.27816)

**💻 Code:** [⭐ Code](https://github.com/Zhuyh1139/PALATE) • [⭐ Code](https://github.com/huggingface)

> Role-playing agents (RPAs) have become one of the most important consumer applications of large language models. Users engage in multi-turn conversations with RPAs for experiences such as emotional comfort, making reliable evaluation essential for...

</details>

<details>
<summary><b>14. RefCaptioner: Multi-Reference Image-Grounded Video Captioning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28509) • [📄 arXiv](https://arxiv.org/abs/2607.28509) • [📥 PDF](https://arxiv.org/pdf/2607.28509)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/pkucs-Ltf/RefCaptioner)

> Existing video captioning models generate natural descriptions of video content but cannot explicitly ground local visual elements to multiple reference images. We introduce multi-reference image-grounded video captioning, a new task requiring fac...

</details>

<details>
<summary><b>15. See2Think: Do Multimodal Models Really Use Intermediate Visual States?</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Jingyu Chen, Panhao Zhou, Haiying Xu, Zhuoran Yan, SiyuYanYan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26769) • [📄 arXiv](https://arxiv.org/abs/2607.26769) • [📥 PDF](https://arxiv.org/pdf/2607.26769)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/CSU-JPG/See2Think)

> We introduce See2Think, a benchmark for understanding whether multimodal large language models truly rely on intermediate visual states during reasoning. Unlike existing evaluations that only measure final answers, See2Think analyzes the full visu...

</details>

<details>
<summary><b>16. SpatialCLI: Learning to Reason With Spatial Tools, Then Without Them</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Zixuan Huang, liushunyu, Shunian, sojuL, Yang-Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27703) • [📄 arXiv](https://arxiv.org/abs/2607.27703) • [📥 PDF](https://arxiv.org/pdf/2607.27703)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/IANNXANG/SpatialCLI)

> We introduce SpatialCLI, a framework that teaches vision-language models to reason with spatial tools and internalize their capabilities for tool-free inference.

</details>

<details>
<summary><b>17. β-OPSD: Deriving with Policy Optimization, Training with Self-Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28582) • [📄 arXiv](https://arxiv.org/abs/2607.28582) • [📥 PDF](https://arxiv.org/pdf/2607.28582)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce β-OPSD, a principled generalization of on-policy self-distillation for reasoning language models. 🔍 Our key observation is that vanilla OPSD is exactly the β = 1 case of a broader KL-regularized policy-optimization objective. Its opti...

</details>

<details>
<summary><b>18. Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28611) • [📄 arXiv](https://arxiv.org/abs/2607.28611) • [📥 PDF](https://arxiv.org/pdf/2607.28611)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Effici...

</details>

<details>
<summary><b>19. ShadowDancer: Teaching Video World Models Any Action by Learning Unified Dynamics Representations from a Video and Its Shadow</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28362) • [📄 arXiv](https://arxiv.org/abs/2607.28362) • [📥 PDF](https://arxiv.org/pdf/2607.28362)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AlayaLab/ShadowDancer)

> ShadowDancer learns to control video world models by watching each dynamics twice — a video and its “shadow” (same motion, resampled appearance). This cross-shadow paradigm yields one unified action interface for any demonstrable behavior: first/t...

</details>

<details>
<summary><b>20. Can Large Language Models Execute Parent Orders?</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Guangyi Zhang, Xinli Xu, Zane Shen, YZCS, StarYDY

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28410) • [📄 arXiv](https://arxiv.org/abs/2607.28410) • [📥 PDF](https://arxiv.org/pdf/2607.28410)

**💻 Code:** [⭐ Code](https://github.com/zaneopen/PACE) • [⭐ Code](https://github.com/huggingface)

> Can LLMs execute large financial orders? PACE extends financial LLMs from deciding what to trade to determining how to execute, achieving competitive execution performance without task-specific training.

</details>

<details>
<summary><b>21. MemHarness: Memory Is Reconstructed, Not Replayed</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28272) • [📄 arXiv](https://arxiv.org/abs/2607.28272) • [📥 PDF](https://arxiv.org/pdf/2607.28272)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/KnowledgeXLab/MemHarness)

> Most memory-augmented LLM agents follow a retrieve-and-replay paradigm, directly inserting retrieved trajectories into the context. This conflates retrieval relevance with action-level applicability: a memory may be semantically related yet inappr...

</details>

<details>
<summary><b>22. Σ-Mem: An Online Reliability Memory for LLM-based Multi-Agent Systems</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27958) • [📄 arXiv](https://arxiv.org/abs/2607.27958) • [📥 PDF](https://arxiv.org/pdf/2607.27958)

**💻 Code:** [⭐ Code](https://github.com/declare-lab/sigma-mem) • [⭐ Code](https://github.com/huggingface)

> We introduce Σ-Mem, a persistent online reliability memory for LLM-based multi-agent systems. Instead of memorizing interaction content, Σ-Mem continuously learns which peers are reliable under different conditions from correctness feedback, enabl...

</details>

<details>
<summary><b>23. INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models</b> ⭐ 71</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26056) • [📄 arXiv](https://arxiv.org/abs/2607.26056) • [📥 PDF](https://arxiv.org/pdf/2607.26056)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zju3dv/INTACT-JEPA)

> [INTACT is an end-to-end unified JEPA that completes the intent-to-action half of LeWM. Instead of using an action-effect world model only for CEM planning, it jointly learns a deployable controller that maps latent motion intent directly to actio...

</details>

<details>
<summary><b>24. Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation</b> ⭐ 101</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27372) • [📄 arXiv](https://arxiv.org/abs/2607.27372) • [📥 PDF](https://arxiv.org/pdf/2607.27372)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/alexiglad/XM)

> TLDR: We introduce Explorative Modeling, a new paradigm for generative modeling that acts as a third pretraining axis when added to existing generative models, and also enables end-to-end generation. Increasing exploration monotonically improves e...

</details>

<details>
<summary><b>25. LEDGERMIND: Provenance-Constrained Multimodal Agentic Reasoning with a Structured Evidence Ledger</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zirong Chen, Siyi Liu, Chenxu Du, Hange Zhou, Enjun Du

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28374) • [📄 arXiv](https://arxiv.org/abs/2607.28374) • [📥 PDF](https://arxiv.org/pdf/2607.28374)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Multimodal agents for visual question answering increasingly operate as multi-step trajectories that interleave perception, retrieval, and reasoning, yet evaluation still largely reduces to final-answer accuracy. This aggregate signal cannot tell ...

</details>

<details>
<summary><b>26. Echoverse: Deep, Evolving Environments for Training Computer-Use Agents at Scale</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28074) • [📄 arXiv](https://arxiv.org/abs/2607.28074) • [📥 PDF](https://arxiv.org/pdf/2607.28074)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/Echoverse)

> Blog: https://www.microsoft.com/en-us/research/blog/echoverse-deep-evolving-environments-for-computer-use-agents/ Github: https://github.com/microsoft/Echoverse

</details>

<details>
<summary><b>27. Multi-Head Attention Residuals</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Junjie Hu, Zefan Cai, Cheng Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27230) • [📄 arXiv](https://arxiv.org/abs/2607.27230) • [📥 PDF](https://arxiv.org/pdf/2607.27230)

**💻 Code:** [⭐ Code](https://github.com/wdlctc/multi-head-attention-residuals) • [⭐ Code](https://github.com/huggingface)

> Multi-Head Attention Residuals

</details>

<details>
<summary><b>28. Filesystem-Based Memory for LLM Agents: Organization, Evolution, and Sustainability</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junda Wu, Hui Wei, Sheldon Yu, Sizhe Zhou, yuz9yuz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26637) • [📄 arXiv](https://arxiv.org/abs/2607.26637) • [📥 PDF](https://arxiv.org/pdf/2607.26637)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Filesystem-Based Memory for LLM Agents: Organization, Evolution, and Sustainability Deployed agents increasingly keep long-term memory as a folder of markdown files that the agent itself reads, writes, and reorganizes with generic file tools. The ...

</details>

<details>
<summary><b>29. Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Zikai Xiao, Heng Li, Wenbin Wang, Tianyu Wang, zhouyuxuanyx

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26627) • [📄 arXiv](https://arxiv.org/abs/2607.26627) • [📥 PDF](https://arxiv.org/pdf/2607.26627)

**💻 Code:** [⭐ Code](https://github.com/ZhouYuxuanYX/Fast-HSD) • [⭐ Code](https://github.com/huggingface)

> We provide Principled Characterization on lossy verification, revealing its underlying paradigms and distributional failure modes.

</details>

<details>
<summary><b>30. Harness-G: A Graph-Structured Harness for Search Agents</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Xirui Liu, Xiaoshu Chen, Sihang Zhou, Haoyuan Chen, Yanning Hou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27652) • [📄 arXiv](https://arxiv.org/abs/2607.27652) • [📥 PDF](https://arxiv.org/pdf/2607.27652)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/7HHHHH/Harness-G)

> Reinforcement-learning search agents typically generate free-form retrieval queries. Although the strings may look diverse, they often retrieve nearly identical evidence, leaving group-relative optimization with little meaningful retrieval contras...

</details>

<details>
<summary><b>31. ReToken: One Token to Improve Vision-Language Models for Visual Retrieval</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Jianfeng Gao, Yuqun Wu, Zhen Zhu, Reuben Tan, Yao Xiao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28627) • [📄 arXiv](https://arxiv.org/abs/2607.28627) • [📥 PDF](https://arxiv.org/pdf/2607.28627)

**💻 Code:** [⭐ Code](https://github.com/avaxiao/ReToken) • [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API QSVideo: Query-Conditioned Semantic Temporal Retrieval for Video Understand...

</details>

<details>
<summary><b>32. OmniScope: Modality-Decoupled Token Compression for Omnimodal Large Language Models</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Meiguang Jin, Yibo Hu, Yuexiao Ma, Yongdong Luo, Consonnm

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.23193) • [📄 arXiv](https://arxiv.org/abs/2607.23193) • [📥 PDF](https://arxiv.org/pdf/2607.23193)

**💻 Code:** [⭐ Code](https://github.com/MAC-AutoML/OmniScope) • [⭐ Code](https://github.com/huggingface)

> Existing token compression methods for omnimodal large language models typically rely on one modality to determine what to retain in the other. We show that this assumption often breaks down: for the same query, audio and video relevance often pea...

</details>

<details>
<summary><b>33. Is Deep Research Reliable? Misleading Knowledge Induces False Conclusions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sen Su, Longju Yang, Lijun Li, Pengyu Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.20891) • [📄 arXiv](https://arxiv.org/abs/2607.20891) • [📥 PDF](https://arxiv.org/pdf/2607.20891)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Deep Research agents conduct long-horizon investigations by iteratively planning, retrieving evidence, and generating reports. However, it remains unclear whether they can resist apparently credible but factually false information introduced into ...

</details>

<details>
<summary><b>34. AI Tour Meeting: Group Travel Planning by LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** oookiku

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.18806) • [📄 arXiv](https://arxiv.org/abs/2607.18806) • [📥 PDF](https://arxiv.org/pdf/2607.18806)

**💻 Code:** [⭐ Code](https://github.com/ntt-dkiku/ai-tour-meeting) • [⭐ Code](https://github.com/huggingface)

> AI Tour Meeting is a group travel planning framework powered by multiple LLM-based agents, where the agents collaborate with each other to find an itinerary that satisfies their constraints and preferences. Its two primary use cases include, but n...

</details>

<details>
<summary><b>35. Beyond Geometric Complementarity: Coherent Overlap in Sparse Mixture-of-Experts Routing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shijian Li, Bonan Xu, Huiyuancs

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28308) • [📄 arXiv](https://arxiv.org/abs/2607.28308) • [📥 PDF](https://arxiv.org/pdf/2607.28308)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce coherent overlap as a way to understand sparse MoE routing: co-selected experts can occupy substantially overlapping representation subspaces while still providing useful multi-expert computation. Across six MoE architectures, we find...

</details>

<details>
<summary><b>36. Fairness Pruning: Locating Demographic Bias in GLU-MLP Layers via Differential Activations</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alfonso Ureña López, Eugenio Martínez Cámara, oopere

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28319) • [📄 arXiv](https://arxiv.org/abs/2607.28319) • [📥 PDF](https://arxiv.org/pdf/2607.28319)

**💻 Code:** [⭐ Code](https://github.com/peremartra/fairness-pruning) • [⭐ Code](https://github.com/huggingface)

> Changing 5 neurons is enough to modify an LLM's response. 🧠 We locate neurons that respond differentially to demographic attributes using contrastive prompt pairs. No additional training required, runs on consumer hardware — the same used for mode...

</details>

<details>
<summary><b>37. AMRD: Adaptive Multi-Teacher Relational Distillation for Lightweight Speech Emotion Recognition</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaoqin Feng, Kuo Yang, Xianglong Wang, Yuqi Li, dlion168

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.25289) • [📄 arXiv](https://arxiv.org/abs/2607.25289) • [📥 PDF](https://arxiv.org/pdf/2607.25289)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> On-device speech emotion recognition (SER) is critical for real-time applications, yet large self-supervised models that excel at SER are too costly for edge devices. Multi-teacher knowledge distillation can compress them into a lightweight studen...

</details>

<details>
<summary><b>38. Pedestrian Archetypes Extension -- More Pedestrian Models for Autonomous Vehicle Safety Testing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Regina Lim, Ritvik Bansal, Namita Gaidhani, Taorui Huang, jub-aer

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.16922) • [📄 arXiv](https://arxiv.org/abs/2607.16922) • [📥 PDF](https://arxiv.org/pdf/2607.16922)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> this is the extended work of the work Pedestrian Archetypes- https://ieeexplore.ieee.org/document/11097414/

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 38 |
| 📅 Today | [`2026-08-02.json`](data/daily/2026-08-02.json) | 38 |
| 📆 This Week | [`2026-W30.json`](data/weekly/2026-W30.json) | 112 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 76 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-02 | 38 | [View JSON](data/daily/2026-08-02.json) |
| 📄 2026-08-01 | 38 | [View JSON](data/daily/2026-08-01.json) |
| 📄 2026-07-31 | 10 | [View JSON](data/daily/2026-07-31.json) |
| 📄 2026-07-30 | 6 | [View JSON](data/daily/2026-07-30.json) |
| 📄 2026-07-29 | 8 | [View JSON](data/daily/2026-07-29.json) |
| 📄 2026-07-28 | 4 | [View JSON](data/daily/2026-07-28.json) |
| 📄 2026-07-27 | 8 | [View JSON](data/daily/2026-07-27.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W30 | 112 | [View JSON](data/weekly/2026-W30.json) |
| 📅 2026-W29 | 70 | [View JSON](data/weekly/2026-W29.json) |
| 📅 2026-W28 | 94 | [View JSON](data/weekly/2026-W28.json) |
| 📅 2026-W27 | 79 | [View JSON](data/weekly/2026-W27.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 76 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
