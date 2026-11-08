<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-32-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7900+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">133</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">172</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7900+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 08, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08449) • [📄 arXiv](https://arxiv.org/abs/2611.08449) • [📥 PDF](https://arxiv.org/pdf/2611.08449)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Jag: Direct Box Prediction for Efficient Visual Grounding (2026) PDRVLA: A ...

</details>

<details>
<summary><b>2. From Autonomous Agents to Cognitive Symbionts: Toward AI-Augmented Human Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.05376) • [📄 arXiv](https://arxiv.org/abs/2611.05376) • [📥 PDF](https://arxiv.org/pdf/2611.05376)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> From Autonomous Agents to Cognitive Symbionts: Toward AI-Augmented Human Intelligence For a decade, AI progress has been measured by how much a machine can do on its own. Agents now write code and run multistep tasks for us, yet people still forge...

</details>

<details>
<summary><b>3. SLA: Scaling Spatial-Language-Action Model via Continual Pre-training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08373) • [📄 arXiv](https://arxiv.org/abs/2611.08373) • [📥 PDF](https://arxiv.org/pdf/2611.08373)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenHelix-Team/Spatial-Forcing)

> Welcome to the era of Spatial-Language-Action (SLA) models. Forget VLAs and WAMs. Take control of your robot with SLA. Arxiv: https://arxiv.org/pdf/2611.08373 Homepage: https://embodiedsla.github.io/ Github: https://github.com/OpenHelix-Team/Spati...

</details>

<details>
<summary><b>4. Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training</b> ⭐ 57</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04277) • [📄 arXiv](https://arxiv.org/abs/2611.04277) • [📥 PDF](https://arxiv.org/pdf/2611.04277)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DeepExperience/Harness-S)

> Harness evolution from a common base exhibits Locality, Complementarity, and Nonstationarity: variants repair different failures, and their utility changes as the policy learns. Rather than seeking a universally superior harness, we introduce Harn...

</details>

<details>
<summary><b>5. Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.00883) • [📄 arXiv](https://arxiv.org/abs/2611.00883) • [📥 PDF](https://arxiv.org/pdf/2611.00883)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/George-Ling3/BestRand)

> Evaluating models on full benchmarks is often expensive. Benchmark compression reduces this cost by using small subsets to approximate full-benchmark results. This paper theoretically explains why random sampling works surprisingly well for benchm...

</details>

<details>
<summary><b>6. WeLM-Image: A Unified Image Generation Model towards Visual Intelligence</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hao Lu, Guohui Zhang, Chen Li, Canmiao Fu, AlexZhou2

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07392) • [📄 arXiv](https://arxiv.org/abs/2611.07392) • [📥 PDF](https://arxiv.org/pdf/2611.07392)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API SenseNova-U1.5: Towards Native Unified Visual Intelligence (2026) FrameForg...

</details>

<details>
<summary><b>7. TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing</b> ⭐ 44</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06741) • [📄 arXiv](https://arxiv.org/abs/2611.06741) • [📥 PDF](https://arxiv.org/pdf/2611.06741)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Lyricccco/tailsteer)

> TailSteer introduces a final refinement stage after SFT, RL, or OPD to unlock persistent multi-turn editing. It substantially reduces long-horizon drift while preserving the base model's original generative and editing capabilities, without changi...

</details>

<details>
<summary><b>8. EvoMemState: Evolving Memory State for Personal Agents</b> ⭐ 25</summary>

<br/>

**👥 Authors:** Junkai Chen, Zhongqiao Liu, Yunbo Tang, Zhishang Xiang, sihuichuibuqi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.03743) • [📄 arXiv](https://arxiv.org/abs/2611.03743) • [📥 PDF](https://arxiv.org/pdf/2611.03743)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/XMUDeepLIT/EvoMemState.git) • [⭐ Code](https://github.com/XMUDeepLIT/EvoMemState)

> Long-term memory enables large language model (LLM) agents to provide continuous personalized assistance by retaining user information across interactions. Existing systems support this process through fact extraction, organization, updating, and ...

</details>

<details>
<summary><b>9. VisSearch-Bench: Evaluating Long Horizon Visual Search in Multimodal Agents</b> ⭐ 59</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04894) • [📄 arXiv](https://arxiv.org/abs/2611.04894) • [📥 PDF](https://arxiv.org/pdf/2611.04894)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/antins-labs/VisSearch-Bench-v1)

> VisSearch-Bench evaluates how multimodal agents organize and use visual evidence during long-horizon web search. It contains 204 human-written tasks across five search categories, spanning 7 domains and 25 subdomains. The tasks require agents to c...

</details>

<details>
<summary><b>10. Towards Foundation Decision Models: Drex</b> ⭐ 115</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07854) • [📄 arXiv](https://arxiv.org/abs/2611.07854) • [📥 PDF](https://arxiv.org/pdf/2611.07854)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/nace-ai/drex-decision-models)

> We present Drex, a family of four open decision models that return a calibrated probability distribution over user-defined options in a single forward pass, without autoregressive decoding. Each request pairs a state (a document, record, or agent ...

</details>

<details>
<summary><b>11. Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06817) • [📄 arXiv](https://arxiv.org/abs/2611.06817) • [📥 PDF](https://arxiv.org/pdf/2611.06817)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale)

> Hi everyone! We’re the authors of Open-HumanWAM , an open recipe for scaling human and robot experience for embodied foundation models. We study how human experience can improve robot learning, using Open-HumanWAM-50K , a 50,000-hour egocentric co...

</details>

<details>
<summary><b>12. On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06976) • [📄 arXiv](https://arxiv.org/abs/2611.06976) • [📥 PDF](https://arxiv.org/pdf/2611.06976)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zhenyi4/platour)

> How much can a small model benefit from a larger model that only processes the prompt—and what actually drives the improvement? We explore these questions in this new paper. In prefill-to-autoregressive cross-model transfer (PACT), a larger sender...

</details>

<details>
<summary><b>13. EDGAR: Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07909) • [📄 arXiv](https://arxiv.org/abs/2611.07909) • [📥 PDF](https://arxiv.org/pdf/2611.07909)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kwanyun/edgar-code)

> Accepted to ACM Transactions on Graphics

</details>

<details>
<summary><b>14. Terminal-style Mathematical Modeling Bench: The First Agentic and Verifiable Benchmark for Mathematical Modeling</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08237) • [📄 arXiv](https://arxiv.org/abs/2611.08237) • [📥 PDF](https://arxiv.org/pdf/2611.08237)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/wxj630/terminal-bench-math-modeling)

> GPT-6 Astra and Claude Fable 5.1 barely pass!? Today's LLM agents are strong when the problem is already well posed — write code, solve a math problem, call an API. The real world never hands you a problem that way. Nobody defines your variables, ...

</details>

<details>
<summary><b>15. Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08376) • [📄 arXiv](https://arxiv.org/abs/2611.08376) • [📥 PDF](https://arxiv.org/pdf/2611.08376)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Muir co-designs gated delta-rule linear attention with sensitivity-allocated low-rank gating (+30.4% wider FFN at iso-params), matched-filter readout to fix 100x slow-channel "glare" (+19.6 pts 5-hop Variable Tracking root retention at 1.34B), and...

</details>

<details>
<summary><b>16. Future Representation Distillation for Imagination-free World Action Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04527) • [📄 arXiv](https://arxiv.org/abs/2611.04527) • [📥 PDF](https://arxiv.org/pdf/2611.04527)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Can we scale up imagination-free WAMs (e.g., Fast-WAM)? Excited to share DreamFree — an imagination-free WAM that improves with stronger video models and more video data, running at 11 Hz with a 14B video backbone. Video models learn how the world...

</details>

<details>
<summary><b>17. SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08483) • [📄 arXiv](https://arxiv.org/abs/2611.08483) • [📥 PDF](https://arxiv.org/pdf/2611.08483)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Qile-J/SwarmEvolve)

> Author here! SwarmEvolve is an LLM-guided evolutionary search framework for mathematical discovery and optimization . Unlike many existing frameworks, SwarmEvolve runs swarms of hundreds of small agents that use latent reasoning and communicate th...

</details>

<details>
<summary><b>18. SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08089) • [📄 arXiv](https://arxiv.org/abs/2611.08089) • [📥 PDF](https://arxiv.org/pdf/2611.08089)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ByteDance-Seed/SAE-interprets-MoE)

> TL;DR : We train a TopK SAE with 2^20 latents and k = 100 on the MoE block output of Qwen3-30B-A3B at layers 4, 30 and 46. Because the SAE encoder is affine and the MoE output is a weighted sum of expert outputs, the activation of each latent deco...

</details>

<details>
<summary><b>19. HeLM: Hierarchical Explicit Language Memory for Non-Markovian Manipulation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02660) • [📄 arXiv](https://arxiv.org/abs/2611.02660) • [📥 PDF](https://arxiv.org/pdf/2611.02660)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kimspacehuni/HeLM)

> 🤖 HeLM: Hierarchical Explicit Language Memory for Non-Markovian Manipulation How can robots track task progress when the current observation alone is insufficient? We introduce HeLM, a hierarchical VLM–VLA framework that maintains task-relevant hi...

</details>

<details>
<summary><b>20. Reducing Representational Interference in Retrieval Embeddings with Trainable Prefix Tokens</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07298) • [📄 arXiv](https://arxiv.org/abs/2611.07298) • [📥 PDF](https://arxiv.org/pdf/2611.07298)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Selecting What Matters: Semantic Compression-Guided Selective Pooling for L...

</details>

<details>
<summary><b>21. Towards Real-Time Full-Waveform LiDAR Transformers via Intensity-Guided Token Reduction and Physics-Aware Augmentation</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Mariko Isogawa, Ryo Yoshida, Kazuma Ikeda, Kotaro Oishi, ryhara

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07613) • [📄 arXiv](https://arxiv.org/abs/2611.07613) • [📥 PDF](https://arxiv.org/pdf/2611.07613)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Keio-CSG/FWL-ToPM)

> Accepted to NeurIPS 2026 (Main)

</details>

<details>
<summary><b>22. Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06982) • [📄 arXiv](https://arxiv.org/abs/2611.06982) • [📥 PDF](https://arxiv.org/pdf/2611.06982)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Decode Before You Hear: Speculative Listening for Full-Duplex Speech Language Models Most existing full-duplex speech models alternate between listening and speaking in fixed time chunks, which leaves the load across chunks highly uneven: compute ...

</details>

<details>
<summary><b>23. Tools Need Not Speak in Text: Latent Context Compression for Agents</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06500) • [📄 arXiv](https://arxiv.org/abs/2611.06500) • [📥 PDF](https://arxiv.org/pdf/2611.06500)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/THU-Agent/Coca-cola)

> We introduce Coca-Cola (Context Compression via Continuous Latent Tokens) , a method that lets LLM agents consume tool observations as compact latent representations instead of lengthy text. Rather than reconstructing the original tool output, Coc...

</details>

<details>
<summary><b>24. Signed but Steerable: Measuring and Closing the Text-Vector Integrity Gap in Offline Retrieval-Augmented Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06498) • [📄 arXiv](https://arxiv.org/abs/2611.06498) • [📥 PDF](https://arxiv.org/pdf/2611.06498)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/securagenai/signed-but-steerable)

> We study the text-vector integrity gap in offline RAG: signed knowledge packs carry text and precomputed embeddings, but a signature proves who published a pack, not that its vectors encode its text (evaluated on five corpora and two embedding mod...

</details>

<details>
<summary><b>25. StepX-OPD: Rethinking Multi-Teacher On-Policy Distillation in Diffusion Models</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.04386) • [📄 arXiv](https://arxiv.org/abs/2611.04386) • [📥 PDF](https://arxiv.org/pdf/2611.04386)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/WinterCCC/StepX-OPD)

> A deployable text-to-image diffusion model must hold several capabilities at once, such as instruction following, text rendering and human preference, and must sample in a few steps. Capabilities and few-step sampling are built in separate stages:...

</details>

<details>
<summary><b>26. OpenFrontiers: A Collection of Open Research Problems and a Benchmark for Scientific Rediscovery</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tong Zhang, Ruida Wang, Jiarui Yao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.37739) • [📄 arXiv](https://arxiv.org/abs/2610.37739) • [📥 PDF](https://arxiv.org/pdf/2610.37739)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language models (LLMs) have shown increasing promise in research-level problem solving, yet a comprehensive collection of such problems is still missing, and evaluating them on research-level problems remains challenging. We introduce OpenFr...

</details>

<details>
<summary><b>27. Decoupled Predictive Decision Making: A Shared Conditional Predictor for Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.02075) • [📄 arXiv](https://arxiv.org/abs/2611.02075) • [📥 PDF](https://arxiv.org/pdf/2611.02075)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce Decoupled Predictive Decision Making (DPDM) , an architecture that turns historical experience into predictions for agent decisions through a shared pretrained learner. The key idea is to separate semantic representation from empirica...

</details>

<details>
<summary><b>28. Information Games: Strategic Crowding and Firm Repositioning in Language-Model Space</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37820) • [📄 arXiv](https://arxiv.org/abs/2609.37820) • [📥 PDF](https://arxiv.org/pdf/2609.37820)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The recent work by Jan LeCunn and Randall Balestriero "Proving and Improving the Platonic Representation Hypothesis with JEPAs", begs some deep question about the role of representation learning in the study of body natural and social phenomenon -...

</details>

<details>
<summary><b>29. Identically Trained Language Models Break on Different Jailbreaks</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.34074) • [📄 arXiv](https://arxiv.org/abs/2610.34074) • [📥 PDF](https://arxiv.org/pdf/2610.34074)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cerulean-works/castor)

> I'll tell you what, only 27% of model pairs kept the same best attack template after a new pretraining seed, against 67% after a new post-training seed. For a bank that buys a vendor model and gets silent base-model refreshes, that reads like a re...

</details>

<details>
<summary><b>30. Hard Sample-Conditioned Decorrelated Attention for Robust Face Recognition</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.41789) • [📄 arXiv](https://arxiv.org/abs/2501.13420) • [📥 PDF](https://arxiv.org/pdf/2610.41789)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>31. NPCBench: A Clinical Apprenticeship Benchmark for Multimodal Care-Pathway Reasoning in Nasopharyngeal Carcinoma</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08369) • [📄 arXiv](https://arxiv.org/abs/2611.08369) • [📥 PDF](https://arxiv.org/pdf/2611.08369)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Current medical benchmarks typically break down clinical competence into isolated factual questions, single-image tasks, or static case vignettes. These tests score individual answers, but reveal less about whether a model can integrate subspecial...

</details>

<details>
<summary><b>32. SynthFaces: Balanced Large Scale Human Dataset</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Aibek Alanov, Kirill Morozov, Denis Bobkov, shredder67

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40119) • [📄 arXiv](https://arxiv.org/abs/2610.40119) • [📥 PDF](https://arxiv.org/pdf/2610.40119)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ControlGenAI/SynthFaces)

> Face datasets are heavily skewed across age, race and gender, and large real-world collections include people who never agreed to be used for training. We propose SynthFaces, a synthetic alternative that is balanced by design. A label-conditional ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 32 |
| 📅 Today | [`2026-11-08.json`](data/daily/2026-11-08.json) | 32 |
| 📆 This Week | [`2026-W44.json`](data/weekly/2026-W44.json) | 133 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 172 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-08 | 32 | [View JSON](data/daily/2026-11-08.json) |
| 📄 2026-11-07 | 32 | [View JSON](data/daily/2026-11-07.json) |
| 📄 2026-11-06 | 8 | [View JSON](data/daily/2026-11-06.json) |
| 📄 2026-11-05 | 13 | [View JSON](data/daily/2026-11-05.json) |
| 📄 2026-11-04 | 20 | [View JSON](data/daily/2026-11-04.json) |
| 📄 2026-11-03 | 14 | [View JSON](data/daily/2026-11-03.json) |
| 📄 2026-11-02 | 14 | [View JSON](data/daily/2026-11-02.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 172 | [View JSON](data/monthly/2026-11.json) |
| 🗓️ 2026-10 | 1120 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |

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
