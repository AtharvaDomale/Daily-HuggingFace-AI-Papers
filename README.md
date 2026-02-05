<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-53-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1775+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">53</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">211</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">256</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1775+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 05, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. CodeOCR: On the Effectiveness of Vision Language Models in Code Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01785) • [📄 arXiv](https://arxiv.org/abs/2602.01785) • [📥 PDF](https://arxiv.org/pdf/2602.01785)

> Try compressing your code input to LLMs with CodeOCR with up to 8x compression ratio!

</details>

<details>
<summary><b>2. AOrchestra: Automating Sub-Agent Creation for Agentic Orchestration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fashen Ren, Yiran Peng, Zhihao Xu, didiforhugface, Aurorra1123

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03786) • [📄 arXiv](https://arxiv.org/abs/2602.03786) • [📥 PDF](https://arxiv.org/pdf/2602.03786)

> AORCHESTRA: Automating Sub-Agent Creation for Agentic Orchestration We introduce AORCHESTRA, a framework-agnostic orchestration paradigm for agentic systems that models any agent as a compositional four-tuple ⟨Instruction, Context, Tools, Model⟩. ...

</details>

<details>
<summary><b>3. No Global Plan in Chain-of-Thought: Uncover the Latent Planning Horizon of LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02103) • [📄 arXiv](https://arxiv.org/abs/2602.02103) • [📥 PDF](https://arxiv.org/pdf/2602.02103)

**💻 Code:** [⭐ Code](https://github.com/lxucs/tele-lens)

> Our data and models are available at: https://github.com/lxucs/tele-lens

</details>

<details>
<summary><b>4. MARS: Modular Agent with Reflective Search for Automated AI Research</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02660) • [📄 arXiv](https://arxiv.org/abs/2602.02660) • [📥 PDF](https://arxiv.org/pdf/2602.02660)

> MARS uses budget-aware planning, modular design, and reflective memory to automate AI research, achieving strong performance and cross-branch knowledge transfer.

</details>

<details>
<summary><b>5. 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03796) • [📄 arXiv](https://arxiv.org/abs/2602.03796) • [📥 PDF](https://arxiv.org/pdf/2602.03796)

> TL;DR : 3DiMo can faithfully transfer genuine 3D motion from a given driving video to a reference character, while enabling flexible free-view camera control.

</details>

<details>
<summary><b>6. daVinci-Agency: Unlocking Long-Horizon Agency Data-Efficiently</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02619) • [📄 arXiv](https://arxiv.org/abs/2602.02619) • [📥 PDF](https://arxiv.org/pdf/2602.02619)

**💻 Code:** [⭐ Code](https://github.com/GAIR-NLP/daVinci-Agency)

> While Large Language Models (LLMs) excel at short-term tasks, scaling them to long-horizon agentic workflows remains challenging. The core bottleneck lies in the scarcity of training data that captures authentic long-dependency structures and cros...

</details>

<details>
<summary><b>7. Research on World Models Is Not Merely Injecting World Knowledge into Specific Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01630) • [📄 arXiv](https://arxiv.org/abs/2602.01630) • [📥 PDF](https://arxiv.org/pdf/2602.01630)

**💻 Code:** [⭐ Code](https://github.com/OpenDCAI/DataFlow-MM) • [⭐ Code](https://github.com/OpenDCAI/DataFlow)

> In this paper, we discuss what the canonical format of world models should be. We welcome everyone to join the discussion.

</details>

<details>
<summary><b>8. CoBA-RL: Capability-Oriented Budget Allocation for Reinforcement Learning in LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03048) • [📄 arXiv](https://arxiv.org/abs/2602.03048) • [📥 PDF](https://arxiv.org/pdf/2602.03048)

> CoBA-RL adaptively allocates RL rollout budgets for LLMs using a capability-valued metric and a heap-based greedy strategy to focus training on high-value samples, improving generalization efficiently.

</details>

<details>
<summary><b>9. Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis</b> ⭐ 43</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03139) • [📄 arXiv](https://arxiv.org/abs/2602.03139) • [📥 PDF](https://arxiv.org/pdf/2602.03139)

**💻 Code:** [⭐ Code](https://github.com/Multimedia-Analytics-Laboratory/dpdmd)

> A simple yet effective approach to preserving sample diversity under DMD, with no perceptual backbone, no discriminator, no auxiliary networks, and no additional ground-truth images.

</details>

<details>
<summary><b>10. SWE-World: Building Software Engineering Agents in Docker-Free Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03419) • [📄 arXiv](https://arxiv.org/abs/2602.03419) • [📥 PDF](https://arxiv.org/pdf/2602.03419)

> Propose a Docker-free SWE RL, alleviating the strong dependence on Docker infrastructure during the SWE training process.

</details>

<details>
<summary><b>11. SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03411) • [📄 arXiv](https://arxiv.org/abs/2602.03411) • [📥 PDF](https://arxiv.org/pdf/2602.03411)

> Unleash the SWE capabilities of the 32B model and provide available infrastructure for academic research on RL

</details>

<details>
<summary><b>12. Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03845) • [📄 arXiv](https://arxiv.org/abs/2602.03845) • [📥 PDF](https://arxiv.org/pdf/2602.03845)

**💻 Code:** [⭐ Code](https://github.com/zhengkid/Parallel-Probe)

> Parallel thinking has emerged as a promising paradigm for reasoning, yet it imposes significant computational burdens. Existing efficiency methods primarily rely on local, per-trajectory signals and lack principled mechanisms to exploit global dyn...

</details>

<details>
<summary><b>13. Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03619) • [📄 arXiv](https://arxiv.org/abs/2602.03619) • [📥 PDF](https://arxiv.org/pdf/2602.03619)

> Nowadays, training and evaluating DeepResearch-generated reports remain challenging due to the lack of verifiable reward signals. Accordingly, rubric-based evaluation has become a common practice. However, existing approaches either rely on coarse...

</details>

<details>
<summary><b>14. RANKVIDEO: Reasoning Reranking for Text-to-Video Retrieval</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02444) • [📄 arXiv](https://arxiv.org/abs/2602.02444) • [📥 PDF](https://arxiv.org/pdf/2602.02444)

**💻 Code:** [⭐ Code](https://github.com/tskow99/RANKVIDEO-Reasoning-Reranker)

> Reasoning Reranking for text-to-video retrieval

</details>

<details>
<summary><b>15. Unified Personalized Reward Model for Vision Generation</b> ⭐ 691</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02380) • [📄 arXiv](https://arxiv.org/abs/2602.02380) • [📥 PDF](https://arxiv.org/pdf/2602.02380)

**💻 Code:** [⭐ Code](https://github.com/CodeGoat24/UnifiedReward)

> 🪐 Project Page: https://codegoat24.github.io/UnifiedReward/flex 🤗 Model Collections: https://huggingface.co/collections/CodeGoat24/unifiedreward-flex 🤗 Dataset: https://huggingface.co/datasets/CodeGoat24/UnifiedReward-Flex-SFT-90K 👋 Point of Conta...

</details>

<details>
<summary><b>16. Neural Predictor-Corrector: Solving Homotopy Problems with Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haoang Li, Yingping Zeng, Zhenjun Zhao, Bangyan Liao, Jiayao Mai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03086) • [📄 arXiv](https://arxiv.org/abs/2602.03086) • [📥 PDF](https://arxiv.org/pdf/2602.03086)

> The Homotopy paradigm, a general principle for solving challenging problems, appears across diverse domains such as robust optimization, global optimization, polynomial root-finding, and sampling. Practical solvers for these problems typically fol...

</details>

<details>
<summary><b>17. WideSeek: Advancing Wide Research via Multi-Agent Scaling</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Zhongtao Jiang, Xiaowei Yuan, Haolin Ren, Jarvis1111, hzy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02636) • [📄 arXiv](https://arxiv.org/abs/2602.02636) • [📥 PDF](https://arxiv.org/pdf/2602.02636)

**💻 Code:** [⭐ Code](https://github.com/hzy312/WideSeek)

> Search intelligence is evolving from Deep Research to Wide Research, a paradigm essential for retrieving and synthesizing comprehensive information sets under complex constraints in parallel. However, advancements in this field are impeded by the ...

</details>

<details>
<summary><b>18. Balancing Understanding and Generation in Discrete Diffusion Models</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Jianbin Jiao, Qixiang Ye, Zheyong Xie, callsys, Mzero17

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01362) • [📄 arXiv](https://arxiv.org/abs/2602.01362) • [📥 PDF](https://arxiv.org/pdf/2602.01362)

**💻 Code:** [⭐ Code](https://github.com/MzeroMiko/XDLM)

> We introduce XDLM, a discrete diffusion model that unifies MDLM and UDLM via a stationary noise kernel. XDLM theoretically bridges the two paradigms, recovers each as a special case, and reduces memory costs through an algebraic simplification of ...

</details>

<details>
<summary><b>19. Less Noise, More Voice: Reinforcement Learning for Reasoning via Instruction Purification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21244) • [📄 arXiv](https://arxiv.org/abs/2601.21244) • [📥 PDF](https://arxiv.org/pdf/2601.21244)

> Reinforcement Learning with Verifiable Rewards (RLVR) has shown strong promise for improving LLM reasoning, but in practice it often fails silently: for many hard prompts, all rollouts receive zero reward, causing training to stall or collapse. 🔍 ...

</details>

<details>
<summary><b>20. Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jae-Joon Kim, Jiwon Song, Beomseok Kang, dongwonjo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03216) • [📄 arXiv](https://arxiv.org/abs/2602.03216) • [📥 PDF](https://arxiv.org/pdf/2602.03216)

> Token Sparse Attention is a complementary approach to efficient sparse attention that dynamically performs token-level compression during attention and reversibly decompresses the representations afterward. Code release is in progress; a cleaned a...

</details>

<details>
<summary><b>21. FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zhuofan Zong, Yunqiao Yang, Houxing Ren, Zimu Lu, scikkk

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03798) • [📄 arXiv](https://arxiv.org/abs/2602.03798) • [📥 PDF](https://arxiv.org/pdf/2602.03798)

**💻 Code:** [⭐ Code](https://github.com/mnluzimu/FullStack-Agent)

> In this paper, we introduce FullStack-Agent, a unified system that combines a multi-agent full-stack development framework equipped with efficient coding and debugging tools (FullStack-Dev), an iterative self-improvement method that improves the a...

</details>

<details>
<summary><b>22. LIVE: Long-horizon Interactive Video World Modeling</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03747) • [📄 arXiv](https://arxiv.org/abs/2602.03747) • [📥 PDF](https://arxiv.org/pdf/2602.03747)

**💻 Code:** [⭐ Code](https://github.com/Junchao-cs/LIVE)

> Project Page: https://junchao-cs.github.io/LIVE-demo/ Technical Paper: https://arxiv.org/pdf/2602.03747

</details>

<details>
<summary><b>23. No Shortcuts to Culture: Indonesian Multi-hop Question Answering for Complex Cultural Understanding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nikos Aletras, Nafise Sadat Moosavi, Vynska Amalia Permadi, XingweiT

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03709) • [📄 arXiv](https://arxiv.org/abs/2602.03709) • [📥 PDF](https://arxiv.org/pdf/2602.03709)

> To move beyond simple fact-recalling, researchers have introduced ID-MoCQA, the first large-scale multi-hop reasoning dataset focused on Indonesian culture. The Problem: Most AI benchmarks use "single-hop" questions that models can answer using su...

</details>

<details>
<summary><b>24. AdaptMMBench: Benchmarking Adaptive Multimodal Reasoning for Mode Selection and Reasoning Process</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Shilin Yan, Zhi Gao, Jongrong Wu, Xiaowen Zhang, xintongzhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02676) • [📄 arXiv](https://arxiv.org/abs/2602.02676) • [📥 PDF](https://arxiv.org/pdf/2602.02676)

**💻 Code:** [⭐ Code](https://github.com/xtong-zhang/AdaptMMBench)

> AdaptMMBench is designed to evaluate adaptive multimodal reasoning beyond final accuracy. It focuses on whether vision-language models make correct reasoning mode selections and execute high-quality, efficient reasoning processes.

</details>

<details>
<summary><b>25. Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haifeng Liu, Jieying Ye, Kaiyan Zhao, Shengrui Li, Hiiamein

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00747) • [📄 arXiv](https://arxiv.org/abs/2602.00747) • [📥 PDF](https://arxiv.org/pdf/2602.00747)

> Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training

</details>

<details>
<summary><b>26. Instruction Anchors: Dissecting the Causal Dynamics of Modality Arbitration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pengfei Zhang, Kehai chen, Xuefeng Bai, Mufan Xu, Yu Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03677) • [📄 arXiv](https://arxiv.org/abs/2602.03677) • [📥 PDF](https://arxiv.org/pdf/2602.03677)

> In this paper, we investigate the working mechanism of modality following through an information flow lens and find that instruction tokens function as structural anchors for modality arbitration.

</details>

<details>
<summary><b>27. Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03647) • [📄 arXiv](https://arxiv.org/abs/2602.03647) • [📥 PDF](https://arxiv.org/pdf/2602.03647)

> Search-R2 trains an Actor and a Meta-Refiner to intervene and repair reasoning with a dense process reward, improving search-based reasoning over RAG/RL baselines.

</details>

<details>
<summary><b>28. LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01053) • [📄 arXiv](https://arxiv.org/abs/2602.01053) • [📥 PDF](https://arxiv.org/pdf/2602.01053)

**💻 Code:** [⭐ Code](https://github.com/hjeon2k/LRAgent)

> We propose LRAgent, an efficient KV-cache sharing framework for multi-LoRA LLM agents that shares highly similar base caches induced by the pretrained weights, while keeping lightweight low-rank caches induced by the LoRA adapters.

</details>

<details>
<summary><b>29. Position: Agentic Evolution is the Path to Evolving LLMs</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Rui Mao, Bing He, Zhan Shi, Hanqing Lu, ventr1c

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00359) • [📄 arXiv](https://arxiv.org/abs/2602.00359) • [📥 PDF](https://arxiv.org/pdf/2602.00359)

**💻 Code:** [⭐ Code](https://github.com/ventr1c/agentic-evoluiton)

> Code repository: https://github.com/ventr1c/agentic-evoluiton

</details>

<details>
<summary><b>30. WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02537) • [📄 arXiv](https://arxiv.org/abs/2602.02537) • [📥 PDF](https://arxiv.org/pdf/2602.02537)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Vision-DeepResearch Benchmark: Rethinking Visual and Textual Search for Mul...

</details>

<details>
<summary><b>31. Bridging Online and Offline RL: Contextual Bandit Learning for Multi-Turn Code Generation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03806) • [📄 arXiv](https://arxiv.org/abs/2602.03806) • [📥 PDF](https://arxiv.org/pdf/2602.03806)

**💻 Code:** [⭐ Code](https://github.com/OSU-NLP-Group/cobalt)

> Recently, there have been significant research interests in training large language models (LLMs) with reinforcement learning (RL) on real-world tasks, such as multi-turn code generation. While online RL tends to perform better than offline RL, it...

</details>

<details>
<summary><b>32. FIRE-Bench: Evaluating Agents on the Rediscovery of Scientific Insights</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02905) • [📄 arXiv](https://arxiv.org/abs/2602.02905) • [📥 PDF](https://arxiv.org/pdf/2602.02905)

> FIRE-Bench is a human-grounded benchmark designed to test whether AI can actually do science end-to-end, from ideation, planning, to implementation, execution, and conclusions. It converts recent, expert-validated scientific insights from top ML c...

</details>

<details>
<summary><b>33. ObjEmbed: Towards Universal Multimodal Object Embeddings</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Xiaohua Xie, Jing Lyu, Fengyun Rao, Yukun Su, fushh7

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01753) • [📄 arXiv](https://arxiv.org/abs/2602.01753) • [📥 PDF](https://arxiv.org/pdf/2602.01753)

**💻 Code:** [⭐ Code](https://github.com/WeChatCV/ObjEmbed)

> Code is available at https://github.com/WeChatCV/ObjEmbed

</details>

<details>
<summary><b>34. Glance and Focus Reinforcement for Pan-cancer Screening</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19103) • [📄 arXiv](https://arxiv.org/abs/2601.19103) • [📥 PDF](https://arxiv.org/pdf/2601.19103)

**💻 Code:** [⭐ Code](https://github.com/Luffy03/GF-Screen)

> Code is available at https://github.com/Luffy03/GF-Screen

</details>

<details>
<summary><b>35. Contextualized Visual Personalization in Vision-Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jisoo Mok, Han Cheol Moon, Junsung Park, Sangwon Yu, Yeongtak

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03454) • [📄 arXiv](https://arxiv.org/abs/2602.03454) • [📥 PDF](https://arxiv.org/pdf/2602.03454)

> We introduce CoViP, a unified framework for contextualized visual personalization in VLMs, featuring a novel personalized image captioning benchmark, an RL-based post-training scheme, and diagnostic downstream personalization tasks.

</details>

<details>
<summary><b>36. POP: Prefill-Only Pruning for Efficient Large Model Inference</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qingan Li, Jun Wang, Zhihui Fu, Junhuihe

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03295) • [📄 arXiv](https://arxiv.org/abs/2602.03295) • [📥 PDF](https://arxiv.org/pdf/2602.03295)

> This paper proposes Prefill-Only Pruning (POP), a stage-aware strategy that accelerates inference by pruning redundant deep layers exclusively during the prefill stage while retaining the full model capacity for decoding to preserve high generativ...

</details>

<details>
<summary><b>37. SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Xin Eric Wang, Yue Fan, Qingni Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02419) • [📄 arXiv](https://arxiv.org/abs/2602.02419) • [📥 PDF](https://arxiv.org/pdf/2602.02419)

**💻 Code:** [⭐ Code](https://github.com/Cece1031/SAFEGROUND)

> Code: https://github.com/Cece1031/SAFEGROUND

</details>

<details>
<summary><b>38. LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04541) • [📄 arXiv](https://arxiv.org/abs/2602.04541) • [📥 PDF](https://arxiv.org/pdf/2602.04541)

> ICLR 2026

</details>

<details>
<summary><b>39. Accelerating Scientific Research with Gemini: Case Studies and Common Techniques</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Song Zuo, Jieming Mao, Lalit Jain, Vincent Cohen-Addad, David P. Woodruff

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03837) • [📄 arXiv](https://arxiv.org/abs/2602.03837) • [📥 PDF](https://arxiv.org/pdf/2602.03837)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Evaluating Frontier LLMs on PhD-Level Mathematical Reasoning: A Benchmark o...

</details>

<details>
<summary><b>40. Privasis: Synthesizing the Largest "Public" Private Dataset from Scratch</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03183) • [📄 arXiv](https://arxiv.org/abs/2602.03183) • [📥 PDF](https://arxiv.org/pdf/2602.03183)

**💻 Code:** [⭐ Code](https://github.com/skywalker023/privasis)

> Project page: https://privasis.github.io Code: https://github.com/skywalker023/privasis

</details>

<details>
<summary><b>41. FaceLinkGen: Rethinking Identity Leakage in Privacy-Preserving Face Recognition with Identity Extraction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02914) • [📄 arXiv](https://arxiv.org/abs/2602.02914) • [📥 PDF](https://arxiv.org/pdf/2602.02914)

> A new red-teaming paper on PPFR systems

</details>

<details>
<summary><b>42. Scaling Small Agents Through Strategy Auctions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02751) • [📄 arXiv](https://arxiv.org/abs/2602.02751) • [📥 PDF](https://arxiv.org/pdf/2602.02751)

> Small language models are cheap but don’t scale to long-horizon tasks. Strategy auctions help them punch above their weight. 🚀

</details>

<details>
<summary><b>43. SimpleGPT: Improving GPT via A Simple Normalization Strategy</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rong Xiao, Jiaquan Ye, Yelin He, Xianbiao Qi, Marco Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01212) • [📄 arXiv](https://arxiv.org/abs/2602.01212) • [📥 PDF](https://arxiv.org/pdf/2602.01212)

> This paper revisits Transformer optimization through the lens of second-order geometry and establish a direct connection between architectural design, activation scale, the Hessian matrix, and the maximum tolerable learning rate.

</details>

<details>
<summary><b>44. MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Boyun Zheng, Wanting Geng, Qi Yang, Liuxin Bao, Saint-lsy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03320) • [📄 arXiv](https://arxiv.org/abs/2602.03320) • [📥 PDF](https://arxiv.org/pdf/2602.03320)

**💻 Code:** [⭐ Code](https://github.com/CUHK-AIM-Group/MedSAM-Agent)

> No abstract available.

</details>

<details>
<summary><b>45. The Necessity of a Unified Framework for LLM-Based Agent Evaluation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sen Su, Philip S. Yu, Li Sun, Pengyu Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03238) • [📄 arXiv](https://arxiv.org/abs/2602.03238) • [📥 PDF](https://arxiv.org/pdf/2602.03238)

> With the advent of Large Language Models (LLMs), general-purpose agents have seen fundamental advancements. However, evaluating these agents presents unique challenges that distinguish them from static QA benchmarks. We observe that current agent ...

</details>

<details>
<summary><b>46. MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02494) • [📄 arXiv](https://arxiv.org/abs/2602.02494) • [📥 PDF](https://arxiv.org/pdf/2602.02494)

**💻 Code:** [⭐ Code](https://github.com/neural-processing-lab/MEG-XL)

> MEG-XL is a brain-to-text foundation model pre-trained with 2.5 minutes of MEG context per sample. It is designed to capture extended neural context, enabling high data efficiency for decoding words from brain activity.

</details>

<details>
<summary><b>47. Didactic to Constructive: Turning Expert Solutions into Learnable Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alan Ritter, Jungsoo Park, Ethan Mendes

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02405) • [📄 arXiv](https://arxiv.org/abs/2602.02405) • [📥 PDF](https://arxiv.org/pdf/2602.02405)

**💻 Code:** [⭐ Code](https://github.com/ethanm88/DAIL)

> Improving the reasoning capabilities of large language models (LLMs) typically relies either on the model's ability to sample a correct solution to be reinforced or on the existence of a stronger model able to solve the problem. However, many diff...

</details>

<details>
<summary><b>48. LangMap: A Hierarchical Benchmark for Open-Vocabulary Goal Navigation</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02220) • [📄 arXiv](https://arxiv.org/abs/2602.02220) • [📥 PDF](https://arxiv.org/pdf/2602.02220)

**💻 Code:** [⭐ Code](https://github.com/bo-miao/LangMap)

> The paper introduces HieraNav, a hierarchical object-oriented navigation task spanning scene, room, region, and instance levels, and presents the first large-scale benchmark LangMap to advance language-driven goal navigation research.

</details>

<details>
<summary><b>49. Feedback by Design: Understanding and Overcoming User Feedback Barriers in Conversational Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01405) • [📄 arXiv](https://arxiv.org/abs/2602.01405) • [📥 PDF](https://arxiv.org/pdf/2602.01405)

> AI is designed to learn and adapt from human feedback. But humans give systematically worse feedback to AI than to other humans. Why? What feedback barriers exist and how can we fix them? Find out in our paper, accepted at CHI 2026.

</details>

<details>
<summary><b>50. RecGOAT: Graph Optimal Adaptive Transport for LLM-Enhanced Multimodal Recommendation with Dual Semantic Alignment</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Chi Lu, Wei Yang, Zeyu Song, Hengwei Ju, Yuecheng Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00682) • [📄 arXiv](https://arxiv.org/abs/2602.00682) • [📥 PDF](https://arxiv.org/pdf/2602.00682)

**💻 Code:** [⭐ Code](https://github.com/6lyc/RecGOAT-LLM4Rec)

> RecGOAT presents a novel yet simple dual-granularity semantic alignment framework for LLM-enhanced multimodal recommendation, which offers theoretically guaranteed alignment capability.

</details>

<details>
<summary><b>51. MemoryLLM: Plug-n-Play Interpretable Feed-Forward Memory for Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00398) • [📄 arXiv](https://arxiv.org/abs/2602.00398) • [📥 PDF](https://arxiv.org/pdf/2602.00398)

> Key Question: What if FFNs were actually human-interpretable, token-indexed memory?

</details>

<details>
<summary><b>52. Adaptive Evidence Weighting for Audio-Spatiotemporal Fusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03817) • [📄 arXiv](https://arxiv.org/abs/2602.03817) • [📥 PDF](https://arxiv.org/pdf/2602.03817)

**💻 Code:** [⭐ Code](https://github.com/leharris3/birdnoise)

> Authors introduce a family of adaptive evidence weighting models for audio spatial-temporal fusion; SoTA results on CBI audio-acoustic classification.

</details>

<details>
<summary><b>53. You Need an Encoder for Native Position-Independent Caching</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01519) • [📄 arXiv](https://arxiv.org/abs/2602.01519) • [📥 PDF](https://arxiv.org/pdf/2602.01519)

> Welcome back, Encoder.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 53 |
| 📅 Today | [`2026-02-05.json`](data/daily/2026-02-05.json) | 53 |
| 📆 This Week | [`2026-W05.json`](data/weekly/2026-W05.json) | 211 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 256 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-05 | 53 | [View JSON](data/daily/2026-02-05.json) |
| 📄 2026-02-04 | 73 | [View JSON](data/daily/2026-02-04.json) |
| 📄 2026-02-03 | 40 | [View JSON](data/daily/2026-02-03.json) |
| 📄 2026-02-02 | 45 | [View JSON](data/daily/2026-02-02.json) |
| 📄 2026-02-01 | 45 | [View JSON](data/daily/2026-02-01.json) |
| 📄 2026-01-31 | 45 | [View JSON](data/daily/2026-01-31.json) |
| 📄 2026-01-30 | 21 | [View JSON](data/daily/2026-01-30.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W05 | 211 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 256 | [View JSON](data/monthly/2026-02.json) |
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
