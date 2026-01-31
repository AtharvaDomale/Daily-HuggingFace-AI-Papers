<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-45-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1519+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">45</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">169</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">781</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1519+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 31, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Idea2Story: An Automated Pipeline for Transforming Research Concepts into Complete Scientific Narratives</b> ⭐ 54</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20833) • [📄 arXiv](https://arxiv.org/abs/2601.20833) • [📥 PDF](https://arxiv.org/pdf/2601.20833)

**💻 Code:** [⭐ Code](https://github.com/AgentAlphaAGI/Idea2Paper)

> arXivLens breakdown of this paper 👉 https://arxivlens.com/PaperView/Details/idea2story-an-automated-pipeline-for-transforming-research-concepts-into-complete-scientific-narratives-2345-6407a884 Executive Summary Detailed Breakdown Practical Applic...

</details>

<details>
<summary><b>2. Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models</b> ⭐ 93</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20354) • [📄 arXiv](https://arxiv.org/abs/2601.20354) • [📥 PDF](https://arxiv.org/pdf/2601.20354)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/SpatialGenEval)

> A very interesting benchmark (ICLR2026) for T2I models!

</details>

<details>
<summary><b>3. Scaling Embeddings Outperforms Scaling Experts in Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21204) • [📄 arXiv](https://arxiv.org/abs/2601.21204) • [📥 PDF](https://arxiv.org/pdf/2601.21204)

> Embedding scaling can outperform mixture of experts for sparse language models, aided by system optimizations and speculative decoding, with LongCat-Flash-Lite achieving strong competitiveness.

</details>

<details>
<summary><b>4. DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation</b> ⭐ 48</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22153) • [📄 arXiv](https://arxiv.org/abs/2601.22153) • [📥 PDF](https://arxiv.org/pdf/2601.22153)

**💻 Code:** [⭐ Code](https://github.com/hzxie/DynamicVLA)

> TL; DR: DynamicVLA enables open-ended dynamic object manipulation by pairing a compact 0.4B VLM with low-latency Continuous Inference and Latent-aware Action Streaming, evaluated at scale through the new DOM benchmark in both simulation and the re...

</details>

<details>
<summary><b>5. OCRVerse: Towards Holistic OCR in End-to-End Vision-Language Models</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Liming Zheng, Wenkang Han, Xuanle Zhao, Lei Chen, Albert-Zhong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21639) • [📄 arXiv](https://arxiv.org/abs/2601.21639) • [📥 PDF](https://arxiv.org/pdf/2601.21639)

**💻 Code:** [⭐ Code](https://github.com/DocTron-hub/OCRVerse)

> OCRVerse: Towards Holistic OCR in End-to-End Vision-Language Models

</details>

<details>
<summary><b>6. MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21821) • [📄 arXiv](https://arxiv.org/abs/2601.21821) • [📥 PDF](https://arxiv.org/pdf/2601.21821)

> Recent advances in Vision Language Models (VLMs) have driven significant progress in visual reasoning. However, open-source VLMs still lag behind proprietary systems, largely due to the lack of high-quality reasoning data. Existing datasets offer ...

</details>

<details>
<summary><b>7. ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21420) • [📄 arXiv](https://arxiv.org/abs/2601.21420) • [📥 PDF](https://arxiv.org/pdf/2601.21420)

**💻 Code:** [⭐ Code](https://github.com/ZihaoHuang-notabot/ConceptMoE)

> ConceptMoE shifts language model processing from uniform token-level to adaptive concept-level computation. By learning to merge semantically similar tokens into unified concepts while preserving fine-grained granularity for complex tokens, it per...

</details>

<details>
<summary><b>8. PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22046) • [📄 arXiv](https://arxiv.org/abs/2601.22046) • [📥 PDF](https://arxiv.org/pdf/2601.22046)

> PLANING introduces a loosely coupled triangle-Gaussian representation and a monocular streaming framework that jointly achieves accurate geometry, high-fidelity rendering, and efficient planar abstraction for embodied AI applications.

</details>

<details>
<summary><b>9. Qwen3-ASR Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21337) • [📄 arXiv](https://arxiv.org/abs/2601.21337) • [📥 PDF](https://arxiv.org/pdf/2601.21337)

> Qwen3-ASR delivers two all-in-one ASR models with 52-language support and a non-autoregressive forced-aligner; achieves competitive SOTA accuracy, fast TTFT, and open-source Apache 2.0 release.

</details>

<details>
<summary><b>10. AgentLongBench: A Controllable Long Benchmark For Long-Contexts Agents via Environment Rollouts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20730) • [📄 arXiv](https://arxiv.org/abs/2601.20730) • [📥 PDF](https://arxiv.org/pdf/2601.20730)

**💻 Code:** [⭐ Code](https://github.com/euReKa025/AgentLongBench)

> The evolution of Large Language Models (LLMs) into autonomous agents necessitates the management of extensive, dynamic contexts. Current benchmarks, however, remain largely static, relying on passive retrieval tasks that fail to simulate the compl...

</details>

<details>
<summary><b>11. Exploring Reasoning Reward Model for Agents</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Zhixun Li, Tianshuo Peng, Manyuan Zhang, Kaituo Feng, bunny127

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22154) • [📄 arXiv](https://arxiv.org/abs/2601.22154) • [📥 PDF](https://arxiv.org/pdf/2601.22154)

**💻 Code:** [⭐ Code](https://github.com/kxfan2002/Reagent)

> Github: https://github.com/kxfan2002/Reagent Paper: https://arxiv.org/pdf/2601.22154

</details>

<details>
<summary><b>12. LoL: Longer than Longer, Scaling Video Generation to Hour</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaojie Li, Tao Yang, Ming Li, Jie Wu, Justin Cui

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16914) • [📄 arXiv](https://arxiv.org/abs/2601.16914) • [📥 PDF](https://arxiv.org/pdf/2601.16914)

**💻 Code:** [⭐ Code](https://github.com/justincui03/LoL)

> Scaling up video generation to hour long, please checkout our paper at: https://arxiv.org/abs/2601.16914 Project Page and code will released at: https://github.com/justincui03/LoL

</details>

<details>
<summary><b>13. Language-based Trial and Error Falls Behind in the Era of Experience</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21754) • [📄 arXiv](https://arxiv.org/abs/2601.21754) • [📥 PDF](https://arxiv.org/pdf/2601.21754)

**💻 Code:** [⭐ Code](https://github.com/Harry-mic/SCOUT)

> While Large Language Models (LLMs) excel in language-based agentic tasks, their applicability to unseen, nonlinguistic environments (e.g., symbolic or spatial tasks) remains limited. Previous work attributes this performance gap to the mismatch be...

</details>

<details>
<summary><b>14. Discovering Hidden Gems in Model Repositories</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yedid Hoshen, Eliahu Horwitz, Jonathan Kahana

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22157) • [📄 arXiv](https://arxiv.org/abs/2601.22157) • [📥 PDF](https://arxiv.org/pdf/2601.22157)

> An investigation of the available fine-tunes of popular foundation models. While over 90% of downloads are directed to the official base versions the paper shows the existence of other, rarely downloaded fine-tunes that significantly outperform them.

</details>

<details>
<summary><b>15. Latent Adversarial Regularization for Offline Preference Optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22083) • [📄 arXiv](https://arxiv.org/abs/2601.22083) • [📥 PDF](https://arxiv.org/pdf/2601.22083)

**💻 Code:** [⭐ Code](https://github.com/enyijiang/GANPO)

> Most offline preference optimization methods (e.g., DPO) constrain policy updates using token-level divergences. However, token-space similarity is often a weak proxy for semantic or structural behavior. We propose GANPO, a plug-and-play regulariz...

</details>

<details>
<summary><b>16. Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haitham Bou Ammar, Matthieu Zimmer, Rasul Tutunov, xtongji

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21590) • [📄 arXiv](https://arxiv.org/abs/2601.21590) • [📥 PDF](https://arxiv.org/pdf/2601.21590)

> What if RL isn’t teaching LLMs how to reason, but just sharpening what’s already there? Most recent progress in LLM reasoning comes from RL post-training (GRPO, verifiers, rewards). But there’s growing evidence that these gains may come less from ...

</details>

<details>
<summary><b>17. Shaping capabilities with token-level data filtering</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21571) • [📄 arXiv](https://arxiv.org/abs/2601.21571) • [📥 PDF](https://arxiv.org/pdf/2601.21571)

**💻 Code:** [⭐ Code](https://github.com/neilrathi/token-filtering)

> Key Findings: 1. Token-level Filtering vs Document-level Filtering (Figure 3) Token filtering Pareto-dominates document filtering : Can achieve equal reduction in undesired capabilities (equal medical loss) at lower cost to desired capabilities (l...

</details>

<details>
<summary><b>18. Llama-3.1-FoundationAI-SecurityLLM-Reasoning-8B Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21051) • [📄 arXiv](https://arxiv.org/abs/2601.21051) • [📥 PDF](https://arxiv.org/pdf/2601.21051)

> Model card: https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Reasoning

</details>

<details>
<summary><b>19. Typhoon-S: Minimal Open Post-Training for Sovereign Large Language Models</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18129) • [📄 arXiv](https://arxiv.org/abs/2601.18129) • [📥 PDF](https://arxiv.org/pdf/2601.18129)

**💻 Code:** [⭐ Code](https://github.com/scb-10x/typhoon-s)

> Code: https://github.com/scb-10x/typhoon-s Artifact: https://huggingface.co/collections/typhoon-ai/typhoon-s

</details>

<details>
<summary><b>20. VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22069) • [📄 arXiv](https://arxiv.org/abs/2601.22069) • [📥 PDF](https://arxiv.org/pdf/2601.22069)

**💻 Code:** [⭐ Code](https://github.com/w-yibo/VTC-R1)

> We propose VTC-R1, an efficient long-context reasoning paradigm that integrates vision-text compression into iterative reasoning. By rendering previous reasoning segments into compact visual representations, VTC-R1 replaces long textual contexts w...

</details>

<details>
<summary><b>21. MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yong Man Ro, Youngchae Chee, Se Yeon Kim, topyun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21181) • [📄 arXiv](https://arxiv.org/abs/2601.21181) • [📥 PDF](https://arxiv.org/pdf/2601.21181)

> Multimodal Large Language Models (MLLMs) suffer from cross-modal hallucinations, where one modality inappropriately influences generation about another, leading to fabricated output. This exposes a more fundamental deficiency in modality-interacti...

</details>

<details>
<summary><b>22. DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20975) • [📄 arXiv](https://arxiv.org/abs/2601.20975) • [📥 PDF](https://arxiv.org/pdf/2601.20975)

> Proposes DeepSearchQA, a 900-prompt benchmark across 17 fields to test long-horizon search, info synthesis, deduplication, and stopping criteria for open-web research agents.

</details>

<details>
<summary><b>23. EEG Foundation Models: Progresses, Benchmarking, and Open Problems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17883) • [📄 arXiv](https://arxiv.org/abs/2601.17883) • [📥 PDF](https://arxiv.org/pdf/2601.17883)

**💻 Code:** [⭐ Code](https://github.com/Dingkun0817/EEG-FM-Benchmark)

> We propose fair and comprehensive benchmarking for open source EEG foundation models.

</details>

<details>
<summary><b>24. Beyond Imitation: Reinforcement Learning for Active Latent Planning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wee Sun Lee, zz1358m

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21598) • [📄 arXiv](https://arxiv.org/abs/2601.21598) • [📥 PDF](https://arxiv.org/pdf/2601.21598)

> Our recent work on Latent Reasoning

</details>

<details>
<summary><b>25. One-step Latent-free Image Generation with Pixel Mean Flows</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhicheng Jiang, Hanhong Zhao, Qiao Sun, Susie Lu, Yiyang Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22158) • [📄 arXiv](https://arxiv.org/abs/2601.22158) • [📥 PDF](https://arxiv.org/pdf/2601.22158)

> One-step Latent-free Image Generation with Pixel Mean Flows

</details>

<details>
<summary><b>26. Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22156) • [📄 arXiv](https://arxiv.org/abs/2601.22156) • [📥 PDF](https://arxiv.org/pdf/2601.22156)

**💻 Code:** [⭐ Code](https://github.com/thunlp/hybrid-linear-attention) • [⭐ Code](https://www.github.com/THUNLP/hybrid-linear-attention)

> Code: https://www.github.com/THUNLP/hybrid-linear-attention

</details>

<details>
<summary><b>27. FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22146) • [📄 arXiv](https://arxiv.org/abs/2601.22146) • [📥 PDF](https://arxiv.org/pdf/2601.22146)

> @ AjayP13 and @ craffel really interesting work and approach, do you plan to add support for multilingual instructions 🤔

</details>

<details>
<summary><b>28. KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Danilo Mandic, Giorgos Iacovides, Yuxuan Gu, WuyangZzzz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21579) • [📄 arXiv](https://arxiv.org/abs/2601.21579) • [📥 PDF](https://arxiv.org/pdf/2601.21579)

**💻 Code:** [⭐ Code](https://github.com/wz1119/KromHC)

> KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices

</details>

<details>
<summary><b>29. Self-Improving Pretraining: using post-trained models to pretrain better models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21343) • [📄 arXiv](https://arxiv.org/abs/2601.21343) • [📥 PDF](https://arxiv.org/pdf/2601.21343)

> Streaming pretraining uses a strong post-trained model to judge next-token generations with RL, improving quality, safety, and factuality earlier in training.

</details>

<details>
<summary><b>30. ECO: Quantized Training without Full-Precision Master Weights</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22101) • [📄 arXiv](https://arxiv.org/abs/2601.22101) • [📥 PDF](https://arxiv.org/pdf/2601.22101)

> We present Error-Compensating Optimizer (ECO), which integrates with standard optimizers and, for the first time, enables quantized training of large-scale LLMs without requiring high-precision master weights.

</details>

<details>
<summary><b>31. MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources</b> ⭐ 41</summary>

<br/>

**👥 Authors:** Jianxun Cui, Xuancheng Zhang, Donglin Di, Baorui Ma, yjh001

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22054) • [📄 arXiv](https://arxiv.org/abs/2601.22054) • [📥 PDF](https://arxiv.org/pdf/2601.22054)

**💻 Code:** [⭐ Code](https://github.com/metric-anything/metric-anything)

> Project Page: https://metric-anything.github.io/metric-anything-io/ Code: https: https://github.com/metric-anything/metric-anything

</details>

<details>
<summary><b>32. Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21996) • [📄 arXiv](https://arxiv.org/abs/2601.21996) • [📥 PDF](https://arxiv.org/pdf/2601.21996)

**💻 Code:** [⭐ Code](https://github.com/chenjianhuii/Mechanistic-Data-Attribution)

> We introduce Mechanistic Data Attribution (MDA), a new paradigm that shifts the focus of mechanistic interpretability from post-hoc circuit analysis to the causal formation of these mechanisms during training.

</details>

<details>
<summary><b>33. Generation Enhances Understanding in Unified Multimodal Models via Multi-Representation Generation</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Guanhua Chen, Yong Wang, Kangrui Cen, Hongyang Wei, Zihan Su

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21406) • [📄 arXiv](https://arxiv.org/abs/2601.21406) • [📥 PDF](https://arxiv.org/pdf/2601.21406)

**💻 Code:** [⭐ Code](https://github.com/Sugewud/UniMRG)

> Paper: https://arxiv.org/abs/2601.21406 Github: https://github.com/Sugewud/UniMRG Project: https://sugewud.github.io/UniMRG-Project/

</details>

<details>
<summary><b>34. BMAM: Brain-inspired Multi-Agent Memory Framework</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mingkun Xu, Yujie Wu, Yusong Wang, Jiaxiang Liu, innovation64

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20465) • [📄 arXiv](https://arxiv.org/abs/2601.20465) • [📥 PDF](https://arxiv.org/pdf/2601.20465)

**💻 Code:** [⭐ Code](https://github.com/innovation64/BMAM)

> We introduce BMAM (Brain-inspired Multi-Agent Memory), a general-purpose memory architecture designed to solve "soul erosion"—the loss of temporal grounding and consistency in long-term agent interactions. 🧠 Key Innovations: Cognitive-inspired Arc...

</details>

<details>
<summary><b>35. JUST-DUB-IT: Video Dubbing via Joint Audio-Visual Diffusion</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Urska Jelercic, Matan Ben Yosef, Tavi Halperin, Naomi Ken Korem, Anthony Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22143) • [📄 arXiv](https://arxiv.org/abs/2601.22143) • [📥 PDF](https://arxiv.org/pdf/2601.22143)

> No abstract available.

</details>

<details>
<summary><b>36. FROST: Filtering Reasoning Outliers with Attention for Efficient Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19001) • [📄 arXiv](https://arxiv.org/abs/2601.19001) • [📥 PDF](https://arxiv.org/pdf/2601.19001)

> ICLR2026

</details>

<details>
<summary><b>37. Reinforcement Learning from Meta-Evaluation: Aligning Language Models Without Ground-Truth Labels</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jesse Roberts, Micah Rentschler

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21268) • [📄 arXiv](https://arxiv.org/abs/2601.21268) • [📥 PDF](https://arxiv.org/pdf/2601.21268)

> We present Reinforcement Learning from Meta-Evaluation (RLME), a label-free RL framework that trains LLMs using evaluator judgments to natural-language meta-questions, achieving performance comparable to supervised rewards while scaling to ambiguo...

</details>

<details>
<summary><b>38. Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20103) • [📄 arXiv](https://arxiv.org/abs/2601.20103) • [📥 PDF](https://arxiv.org/pdf/2601.20103)

> We show that contrasting reward hacks in an outlier detection setting helps LLMs detect code hacking behaviors. We further show that a cluster's benign-to-hacked trajectory ratio influences this detection rate. Finally we perform thorough QA and s...

</details>

<details>
<summary><b>39. Segment Length Matters: A Study of Segment Lengths on Audio Fingerprinting Performance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Melody Ma, Iram Kamdar, Yunyan Ouyang, Ziling Gong, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17690) • [📄 arXiv](https://arxiv.org/abs/2601.17690) • [📥 PDF](https://arxiv.org/pdf/2601.17690)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Lightweight Resolution-Aware Audio Deepfake Detection via Cross-Scale Atten...

</details>

<details>
<summary><b>40. PRISM: Learning Design Knowledge from Data for Stylistic Design Improvement</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Stefano Petrangeli, Yu Shen, Sunav Choudhary, Huaxiaoyue Wang, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11747) • [📄 arXiv](https://arxiv.org/abs/2601.11747) • [📥 PDF](https://arxiv.org/pdf/2601.11747)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API MiLDEdit: Reasoning-Based Multi-Layer Design Document Editing (2026) Styles...

</details>

<details>
<summary><b>41. WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21872) • [📄 arXiv](https://arxiv.org/abs/2601.21872) • [📥 PDF](https://arxiv.org/pdf/2601.21872)

> Accepted at ICLR 2026

</details>

<details>
<summary><b>42. Spotlighting Task-Relevant Features: Object-Centric Representations for Better Generalization in Robotic Manipulation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Liming Chen, Emmanuel Dellandréa, Bruno Machado, Beegbrain

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21416) • [📄 arXiv](https://arxiv.org/abs/2601.21416) • [📥 PDF](https://arxiv.org/pdf/2601.21416)

> The ability of visuomotor policies to generalize across tasks and environments critically depends on the structure of the underlying visual representations. While most state-of-the-art robot policies rely on either global or dense features from pr...

</details>

<details>
<summary><b>43. WorldBench: Disambiguating Physics for Diagnostic Evaluation of World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pranay Boreddy, Ayush Agrawal, Jim Solomon, Howard Zhang, Rishi Upadhyay

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21282) • [📄 arXiv](https://arxiv.org/abs/2601.21282) • [📥 PDF](https://arxiv.org/pdf/2601.21282)

> WorldBench provides a disentangled, concept-specific video benchmark to rigorously evaluate physical reasoning in world models and their video generation.

</details>

<details>
<summary><b>44. STORM: Slot-based Task-aware Object-centric Representation for robotic Manipulation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Liming Chen, Emmanuel Dellandréa, Beegbrain

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20381) • [📄 arXiv](https://arxiv.org/abs/2601.20381) • [📥 PDF](https://arxiv.org/pdf/2601.20381)

> We introduce a slot-based object-centric method with a "task-awareness" alignment in order to learn robotic manipulation. Our method obtains strong generalization improvements over existing VFM by simply adding a few layers of structure and keepin...

</details>

<details>
<summary><b>45. Flow-based Extremal Mathematical Structure Discovery</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18005) • [📄 arXiv](https://arxiv.org/abs/2601.18005) • [📥 PDF](https://arxiv.org/pdf/2601.18005)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 45 |
| 📅 Today | [`2026-01-31.json`](data/daily/2026-01-31.json) | 45 |
| 📆 This Week | [`2026-W04.json`](data/weekly/2026-W04.json) | 169 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 781 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-31 | 45 | [View JSON](data/daily/2026-01-31.json) |
| 📄 2026-01-30 | 21 | [View JSON](data/daily/2026-01-30.json) |
| 📄 2026-01-29 | 21 | [View JSON](data/daily/2026-01-29.json) |
| 📄 2026-01-28 | 37 | [View JSON](data/daily/2026-01-28.json) |
| 📄 2026-01-27 | 18 | [View JSON](data/daily/2026-01-27.json) |
| 📄 2026-01-26 | 27 | [View JSON](data/daily/2026-01-26.json) |
| 📄 2026-01-25 | 27 | [View JSON](data/daily/2026-01-25.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W04 | 169 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
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
