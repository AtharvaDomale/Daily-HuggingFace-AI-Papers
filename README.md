<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-37-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1432+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">82</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">694</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1432+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 28, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Can LLMs Clean Up Your Mess? A Survey of Application-Ready Data Preparation with LLMs</b> ⭐ 644</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17058) • [📄 arXiv](https://arxiv.org/abs/2601.17058) • [📥 PDF](https://arxiv.org/pdf/2601.17058)

**💻 Code:** [⭐ Code](https://github.com/weAIDB/awesome-data-llm)

> Please refer to our repository for more details: https://github.com/weAIDB/awesome-data-llm .

</details>

<details>
<summary><b>2. daVinci-Dev: Agent-native Mid-training for Software Engineering</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18418) • [📄 arXiv](https://arxiv.org/abs/2601.18418) • [📥 PDF](https://arxiv.org/pdf/2601.18418)

**💻 Code:** [⭐ Code](https://github.com/GAIR-NLP/daVinci-Dev)

> Recently, the frontier of Large Language Model (LLM) capabilities has shifted from single-turn code generation to agentic software engineering-a paradigm where models autonomously navigate, edit, and test complex repositories.  While post-training...

</details>

<details>
<summary><b>3. The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation</b> ⭐ 228</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17737) • [📄 arXiv](https://arxiv.org/abs/2601.17737) • [📥 PDF](https://arxiv.org/pdf/2601.17737)

**💻 Code:** [⭐ Code](https://github.com/Tencent/digitalhuman/tree/main/ScriptAgent)

> Convert dialogue to script for video generation.

</details>

<details>
<summary><b>4. Scientific Image Synthesis: Benchmarking, Methodologies, and Downstream Utility</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17027) • [📄 arXiv](https://arxiv.org/abs/2601.17027) • [📥 PDF](https://arxiv.org/pdf/2601.17027)

**💻 Code:** [⭐ Code](https://github.com/SciGenBench/SciGenBench)

> While synthetic data has proven effective for improving scientific reasoning in the text domain, multimodal reasoning remains constrained by the difficulty of synthesizing scientifically rigorous images. Existing Text-to-Image (T2I) models often p...

</details>

<details>
<summary><b>5. Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17367) • [📄 arXiv](https://arxiv.org/abs/2601.17367) • [📥 PDF](https://arxiv.org/pdf/2601.17367)

**💻 Code:** [⭐ Code](https://github.com/LCM-Lab/Elastic-Attention)

> Elastic Attention enables models to achieve both strong performance and efficient inference by dynamically allocating computation modes (Full Attention or Sparse Attention) to each attention head through our designed Attention Router, adapting spa...

</details>

<details>
<summary><b>6. iFSQ: Improving FSQ for Image Generation with 1 Line of Code</b> ⭐ 59</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17124) • [📄 arXiv](https://arxiv.org/abs/2601.17124) • [📥 PDF](https://arxiv.org/pdf/2601.17124)

**💻 Code:** [⭐ Code](https://github.com/Tencent-Hunyuan/iFSQ)

> AR or Diffusion? It’s been hard to judge because different tokenizers (VQ vs. VAE) Enter iFSQ with just 1 line of code! We found: (1) AR wins on efficiency, but Diffusion hits a higher quality ceiling. (2) The sweet spot for representations is ~4 ...

</details>

<details>
<summary><b>7. Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18778) • [📄 arXiv](https://arxiv.org/abs/2601.18778) • [📥 PDF](https://arxiv.org/pdf/2601.18778)

> Check out our blog post: https://ssundaram21.github.io/soar/ !

</details>

<details>
<summary><b>8. Self-Refining Video Sampling</b> ⭐ 43</summary>

<br/>

**👥 Authors:** Sangwon Jang, jaehong31, sainx, harry9704, taekyungki

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18577) • [📄 arXiv](https://arxiv.org/abs/2601.18577) • [📥 PDF](https://arxiv.org/pdf/2601.18577)

**💻 Code:** [⭐ Code](https://github.com/agwmon/self-refine-video)

> [TL;DR] We present self-refining video sampling method that reuses a pre-trained video generator as a denoising autoencoder to iteratively refine latents. With ~50% additional NFEs, it improves physical realism (e.g., motion coherence and physics ...

</details>

<details>
<summary><b>9. VIBEVOICE-ASR Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18184) • [📄 arXiv](https://arxiv.org/abs/2601.18184) • [📥 PDF](https://arxiv.org/pdf/2601.18184)

> VibeVoice-ASR is a unified speech-to-text model designed to handle 60-minute long-form audio in a single pass, generating structured transcriptions containing Who (Speaker), When (Timestamps), and What (Content), with support for User-Customized C...

</details>

<details>
<summary><b>10. DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18137) • [📄 arXiv](https://arxiv.org/abs/2601.18137) • [📥 PDF](https://arxiv.org/pdf/2601.18137)

> DeepPlanning — a new benchmark for long-horizon agent planning in real-world scenarios!

</details>

<details>
<summary><b>11. CGPT: Cluster-Guided Partial Tables with LLM-Generated Supervision for Table Retrieval</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15849) • [📄 arXiv](https://arxiv.org/abs/2601.15849) • [📥 PDF](https://arxiv.org/pdf/2601.15849)

**💻 Code:** [⭐ Code](https://github.com/yumeow0122/CGPT)

> General-purpose embedding models have demonstrated strong performance in text retrieval but remain suboptimal for table retrieval, where highly structured content leads to semantic compression and query–table mismatch. Recent LLM-based retrieval a...

</details>

<details>
<summary><b>12. STAR: Semantic Table Representation with Header-Aware Clustering and Adaptive Weighted Fusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15860) • [📄 arXiv](https://arxiv.org/abs/2601.15860) • [📥 PDF](https://arxiv.org/pdf/2601.15860)

**💻 Code:** [⭐ Code](https://github.com/adsl135789/STAR)

> Table retrieval is the task of retrieving the most relevant tables from large-scale corpora given natural language queries. However, structural and semantic discrepancies between unstructured text and structured tables make embedding alignment par...

</details>

<details>
<summary><b>13. Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18217) • [📄 arXiv](https://arxiv.org/abs/2601.18217) • [📥 PDF](https://arxiv.org/pdf/2601.18217)

> Generalist LLM agents are often post-trained on a narrow set of environments but deployed across far broader, unseen domains. In this work, we investigate the challenge of agentic post-training when the eventual test domains are unknown. Specifica...

</details>

<details>
<summary><b>14. AR-Omni: A Unified Autoregressive Model for Any-to-Any Generation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17761) • [📄 arXiv](https://arxiv.org/abs/2601.17761) • [📥 PDF](https://arxiv.org/pdf/2601.17761)

**💻 Code:** [⭐ Code](https://github.com/ModalityDance/AR-Omni)

> AR-Omni is a single-decoder, single-token-stream autoregressive any-to-any model. It unifies multimodal generation (text, images, and speech) as standard next-token prediction over interleaved sequences. It improves training and inference with tas...

</details>

<details>
<summary><b>15. TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18744) • [📄 arXiv](https://arxiv.org/abs/2601.18744) • [📥 PDF](https://arxiv.org/pdf/2601.18744)

> No abstract available.

</details>

<details>
<summary><b>16. Agentic Very Long Video Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18157) • [📄 arXiv](https://arxiv.org/abs/2601.18157) • [📥 PDF](https://arxiv.org/pdf/2601.18157)

> EGAgent uses entity scene graphs and structured search over long, multimodal video streams to enable cross-modal, temporally coherent reasoning for egocentric video understanding.

</details>

<details>
<summary><b>17. DRPG (Decompose, Retrieve, Plan, Generate): An Agentic Framework for Academic Rebuttal</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Jingjun Xu, Yingjie Yu, jiaxuanYou, HakHan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18081) • [📄 arXiv](https://arxiv.org/abs/2601.18081) • [📥 PDF](https://arxiv.org/pdf/2601.18081)

**💻 Code:** [⭐ Code](https://github.com/ulab-uiuc/DRPG-RebuttalAgent/tree/master)

> DRPG - An Agentic Framework for Academic Rebuttal

</details>

<details>
<summary><b>18. IVRA: Improving Visual-Token Relations for Robot Action Policy with Training-Free Hint-Based Guidance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** yjang43, cfmata, jjh6297, kahnchana, jongwoopark7978

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16207) • [📄 arXiv](https://arxiv.org/abs/2601.16207) • [📥 PDF](https://arxiv.org/pdf/2601.16207)

> IVRA is a training-free, inference-time drop-in that restores spatial structure in VLA models by injecting encoder affinity signals into selected LLM layers (no retraining, no extra parameters, ~3% latency). It generalizes across LLaRA, OpenVLA, a...

</details>

<details>
<summary><b>19. SAGE: Steerable Agentic Data Generation for Deep Search with Execution Feedback</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18202) • [📄 arXiv](https://arxiv.org/abs/2601.18202) • [📥 PDF](https://arxiv.org/pdf/2601.18202)

> SAGE automatically generates difficulty-controlled deep-search QA pairs via an iterative agent-feedback loop, yielding higher-quality training data that improves deep search agent performance and adaptability.

</details>

<details>
<summary><b>20. SkyReels-V3 Technique Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17323) • [📄 arXiv](https://arxiv.org/abs/2601.17323) • [📥 PDF](https://arxiv.org/pdf/2601.17323)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API KlingAvatar 2.0 Technical Report (2025) YingVideo-MV: Music-Driven Multi-St...

</details>

<details>
<summary><b>21. Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17111) • [📄 arXiv](https://arxiv.org/abs/2601.17111) • [📥 PDF](https://arxiv.org/pdf/2601.17111)

> Mixture-of-Experts (MoE) models are typically pre-trained with explicit load-balancing constraints to ensure statistically balanced expert routing. Despite this, we observe that even well-trained MoE models exhibit significantly imbalanced routing...

</details>

<details>
<summary><b>22. One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18731) • [📄 arXiv](https://arxiv.org/abs/2601.18731) • [📥 PDF](https://arxiv.org/pdf/2601.18731)

**💻 Code:** [⭐ Code](https://github.com/ModalityDance/MRM)

> Alignment of Large Language Models (LLMs) aims to align outputs with human preferences, and personalized alignment further adapts models to individual users. This relies on personalized reward models that capture user-specific preferences and auto...

</details>

<details>
<summary><b>23. End-to-End Joint ASR and Speaker Role Diarization with Child-Adult Interactions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shrikanth Narayanan, Catherine Lord, Somer Bishop, Anfeng Xu, tiantiaf

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17640) • [📄 arXiv](https://arxiv.org/abs/2601.17640) • [📥 PDF](https://arxiv.org/pdf/2601.17640)

**💻 Code:** [⭐ Code](https://github.com/usc-sail/joint-asr-diarization-child-adult)

> Accurate transcription and speaker diarization of child–adult spoken interactions are crucial for developmental and clinical research. However, manual annotation is time-consuming and challenging to scale. Existing automated systems typically rely...

</details>

<details>
<summary><b>24. A Mechanistic View on Video Generation as World Models: State and Dynamics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17067) • [📄 arXiv](https://arxiv.org/abs/2601.17067) • [📥 PDF](https://arxiv.org/pdf/2601.17067)

> While large-scale video generation models show signs of emergent physical coherence, they remain distinct from true world models. A critical gap persists between modern "stateless" video architectures and the "state-centric" requirements of classi...

</details>

<details>
<summary><b>25. Diffusion In Diffusion: Reclaiming Global Coherence in Semi-Autoregressive Diffusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13599) • [📄 arXiv](https://arxiv.org/abs/2601.13599) • [📥 PDF](https://arxiv.org/pdf/2601.13599)

> One of the most compelling features of global discrete diffusion language models is their global bidirectional contextual capability. However, existing block-based diffusion studies tend to introduce autoregressive priors, which, while offering be...

</details>

<details>
<summary><b>26. UI Remix: Supporting UI Design Through Interactive Example Retrieval and Remixing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** April Yi Wang, Mustafa Doga Dogan, Xiaotian Su, Junling Wang, HenryLhy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18759) • [📄 arXiv](https://arxiv.org/abs/2601.18759) • [📥 PDF](https://arxiv.org/pdf/2601.18759)

> UI Remix enables interactive, example-driven design for mobile interfaces using multimodal retrieval-augmented generation to search, adapt, and remix interface components with source transparency.

</details>

<details>
<summary><b>27. Masked Depth Modeling for Spatial Perception</b> ⭐ 252</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17895) • [📄 arXiv](https://arxiv.org/abs/2601.17895) • [📥 PDF](https://arxiv.org/pdf/2601.17895)

**💻 Code:** [⭐ Code](https://github.com/Robbyant/lingbot-depth)

> Website: technology.robbyant.com/lingbot-depth Code: https://github.com/Robbyant/lingbot-depth

</details>

<details>
<summary><b>28. PingPong: A Natural Benchmark for Multi-Turn Code-Switching Dialogues</b> ⭐ 0</summary>

<br/>

**👥 Authors:** afaji, gentaiscool, faridlazuarda, hanifmz0711, rifqifarhansyah

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17277) • [📄 arXiv](https://arxiv.org/abs/2601.17277) • [📥 PDF](https://arxiv.org/pdf/2601.17277)

> PingPong: A Natural Benchmark for Multi-Turn Code-Switching Dialogues

</details>

<details>
<summary><b>29. Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15015) • [📄 arXiv](https://arxiv.org/abs/2601.15015v1) • [📥 PDF](https://arxiv.org/pdf/2601.15015)

**💻 Code:** [⭐ Code](https://github.com/safe-autonomous-systems/fluidgym)

> FluidGym: Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control There is enormous potential for reinforcement learning and other data-driven control paradigms for controlling large-scale fluid flows. But RL r...

</details>

<details>
<summary><b>30. The Side Effects of Being Smart: Safety Risks in MLLMs' Multi-Image Reasoning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14127) • [📄 arXiv](https://arxiv.org/abs/2601.14127) • [📥 PDF](https://arxiv.org/pdf/2601.14127)

**💻 Code:** [⭐ Code](https://github.com/thu-coai/MIR-SafetyBench)

> As Multimodal Large Language Models (MLLMs) acquire stronger reasoning capabilities to handle complex, multi-image instructions, this advancement may pose new safety risks. We study this problem by introducing MIR-SafetyBench, the first benchmark ...

</details>

<details>
<summary><b>31. Less Is More -- Until It Breaks: Security Pitfalls of Vision Token Compression in Large Vision-Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guanhong Tao, Yanjun Zhang, Leo Yu Zhang, Xiaomei Zhang, plll123

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.12042) • [📄 arXiv](https://arxiv.org/abs/2601.12042) • [📥 PDF](https://arxiv.org/pdf/2601.12042)

> Visual token compression is widely used to accelerate inference in Large Vision–Language Models (LVLMs), enabling deployment in latency- and resource-constrained settings. This paper reveals that such compression introduces a previously overlooked...

</details>

<details>
<summary><b>32. MortalMATH: Evaluating the Conflict Between Reasoning Objectives and Emergency Contexts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18790) • [📄 arXiv](https://arxiv.org/abs/2601.18790) • [📥 PDF](https://arxiv.org/pdf/2601.18790)

> Large Language Models are increasingly optimized for deep reasoning, prioritizing the correct execution of complex tasks over general conversation. We investigate whether this focus on calculation creates a "tunnel vision" that ignores safety in c...

</details>

<details>
<summary><b>33. HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junhong Lin, zhoudw, liangshi, yanyujun, xyzeng2000

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18753) • [📄 arXiv](https://arxiv.org/abs/2601.18753) • [📥 PDF](https://arxiv.org/pdf/2601.18753)

> 🚀 HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs Accepted at ICLR 2026 In this work, we introduce HalluGuard , a unified, theory-driven framework for hallucination detection in large language models , accepted at ...

</details>

<details>
<summary><b>34. RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yiming Song, Han Wu, larryle, zhiyuanyou, Jize1

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18130) • [📄 arXiv](https://arxiv.org/abs/2601.18130) • [📥 PDF](https://arxiv.org/pdf/2601.18130)

> An efficient mixture-of-agents framework with dynamic routing.

</details>

<details>
<summary><b>35. TensorLens: End-to-End Transformer Analysis via High-Order Attention Tensors</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17958) • [📄 arXiv](https://arxiv.org/abs/2601.17958) • [📥 PDF](https://arxiv.org/pdf/2601.17958)

> Attention matrices are fundamental to transformer research, supporting a broad range of applications including interpretability, visualization, manipulation, and distillation. Yet, most existing analyses focus on individual attention heads or laye...

</details>

<details>
<summary><b>36. Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17617) • [📄 arXiv](https://arxiv.org/abs/2601.17617) • [📥 PDF](https://arxiv.org/pdf/2601.17617)

> This paper presents a large-scale behavioral analysis of 14.44M agentic search interactions, characterizing how autonomous agents organize sessions by intent, execute query reformulations, and reuse retrieved evidence across multi-step trajectories.

</details>

<details>
<summary><b>37. Interp3D: Correspondence-aware Interpolation for Generative Textured 3D Morphing</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Wei Ji, Jiayin Zhu, Qiyuan He, Yicong Li, xiaolul2

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14103) • [📄 arXiv](https://arxiv.org/abs/2601.14103) • [📥 PDF](https://arxiv.org/pdf/2601.14103)

**💻 Code:** [⭐ Code](https://github.com/xiaolul2/Interp3D)

> In this work, we propose Interp3D, a training-free approach that instantiates the progressive alignment principle based on generative priors for textured 3D morphing.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 37 |
| 📅 Today | [`2026-01-28.json`](data/daily/2026-01-28.json) | 37 |
| 📆 This Week | [`2026-W04.json`](data/weekly/2026-W04.json) | 82 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 694 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-28 | 37 | [View JSON](data/daily/2026-01-28.json) |
| 📄 2026-01-27 | 18 | [View JSON](data/daily/2026-01-27.json) |
| 📄 2026-01-26 | 27 | [View JSON](data/daily/2026-01-26.json) |
| 📄 2026-01-25 | 27 | [View JSON](data/daily/2026-01-25.json) |
| 📄 2026-01-24 | 27 | [View JSON](data/daily/2026-01-24.json) |
| 📄 2026-01-23 | 26 | [View JSON](data/daily/2026-01-23.json) |
| 📄 2026-01-22 | 32 | [View JSON](data/daily/2026-01-22.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W04 | 82 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 694 | [View JSON](data/monthly/2026-01.json) |
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
