<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-42-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5836+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">42</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">106</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">455</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5836+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 19, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. StateM: Reaching 95.3% Raw Accuracy, or a \$15 Frontier Run, on Terminal-Bench 2.1 via Harness Scaling</b> ⭐ 49</summary>

<br/>

**👥 Authors:** Kai Wang, Zhangyang Atlas Wang, Yaxin Lu, Ziheng Qin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15089) • [📄 arXiv](https://arxiv.org/abs/2608.15089) • [📥 PDF](https://arxiv.org/pdf/2608.15089)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/henryqin1997/statem)

> Harness scaling, a different way to scale agent performance. On Terminal-Bench 2.1, StateM raises GPT-5.5 xhigh to 92.1%, versus 83.1% reference and GPT-5.6 Sol Ultra at 91.9%. The runbook transfers unchanged to GPT-5.6. With GPT-5.6 Sol xhigh, St...

</details>

<details>
<summary><b>2. HarnessEval-W: Agentifying the Evaluation of Visual Worlds</b> ⭐ 132</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16859) • [📄 arXiv](https://arxiv.org/abs/2608.16859) • [📥 PDF](https://arxiv.org/pdf/2608.16859)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MirroS-Lab/HarnessEval-W)

> HarnessEval is an agentified evaluation pipeline that brings the harness paradigm from the LLM ecosystem to world model benchmarking. Rather than applying a fixed rubric, HarnessEval interprets the context of each evaluation case, decomposes the e...

</details>

<details>
<summary><b>3. VibeWorlding: Can Multimodal Agents Construct 3D Open Worlds End-to-End?</b> ⭐ 52</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15265) • [📄 arXiv](https://arxiv.org/abs/2608.15265) • [📥 PDF](https://arxiv.org/pdf/2608.15265)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/usail-hkust/VibeWorlding-Gym)

> The first agentic RL framework for end-to-end 3D world construction tasks}, providing a unified stack: 2,616 annotated 3D assets, 323 seed 3D worlds, 6,828 user queries, an asset retrieval embedding model, an interactive sandbox environment, a dua...

</details>

<details>
<summary><b>4. Large Discovery Models: Empirically-grounded Model-Based Open-Ended Search</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15669) • [📄 arXiv](https://arxiv.org/abs/2608.15669) • [📥 PDF](https://arxiv.org/pdf/2608.15669)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yzailab/Large-Discovery-Models)

> Discovery happens through repeated rounds of proposing, testing and learning. LDM uses every result to focus the next round, finding promising programs, proteins and molecules with fewer trials.

</details>

<details>
<summary><b>5. Learn What's Left, Not What's Mastered: Saturation Aware Advantage Reweighting for Multi-Reward Policy Optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16072) • [📄 arXiv](https://arxiv.org/abs/2608.16072) • [📥 PDF](https://arxiv.org/pdf/2608.16072)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reinforcement learning (RL) with group-relative advantages has become the de facto standard for post-training language model reasoners. However, when optimizing multiple reward objectives, existing methods typically scalarize the reward vector wit...

</details>

<details>
<summary><b>6. MOSS-VL Technical Report</b> ⭐ 436</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15045) • [📄 arXiv](https://arxiv.org/abs/2608.15045) • [📥 PDF](https://arxiv.org/pdf/2608.15045)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenMOSS/MOSS-VL)

> MOSS-VL treats real-time interaction — perceiving while it speaks — as a first-class capability: vision enters the decoder only through gated cross-attention, so the model keeps watching incoming frames while it generates. Among open-source stream...

</details>

<details>
<summary><b>7. ClawGym II: Exploring Black-Box RL on Agent Harness</b> ⭐ 36</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16798) • [📄 arXiv](https://arxiv.org/abs/2608.16798) • [📥 PDF](https://arxiv.org/pdf/2608.16798)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ClawGym) • [⭐ Code](https://github.com/ClawGym/ClawGym-Agents)

> Agent harnesses have substantially improved performance on long-horizon tasks by coordinating agent interactions with the environment. However, reinforcement learning through complex harnesses remains largely unexplored, as scaling such training t...

</details>

<details>
<summary><b>8. UI-Mate: Advancing Open-Weight Foundation GUI Agents with In-Context Demonstrations</b> ⭐ 39</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15930) • [📄 arXiv](https://arxiv.org/abs/2608.15930) • [📥 PDF](https://arxiv.org/pdf/2608.15930)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/UI-Mate)

> Strong general computer use — plus one demonstration when the instruction alone is not enough.

</details>

<details>
<summary><b>9. An Empirical Study of Training Pixel-Space Text-to-Image Diffusion Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16887) • [📄 arXiv](https://arxiv.org/abs/2608.16887) • [📥 PDF](https://arxiv.org/pdf/2608.16887)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Z-Image-Pixel & Empirical Insight of Training Pixel-Space Diffusion Models

</details>

<details>
<summary><b>10. Agentic Transaction: Towards ACID-Compliant Agent Systems</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13900) • [📄 arXiv](https://arxiv.org/abs/2608.13900) • [📥 PDF](https://arxiv.org/pdf/2608.13900)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/TsinghuaDatabaseGroup/ACID-Agent)

> Can LLM agents have transactions like databases? • Agentic Transaction — a transactional abstraction for reliable long-horizon LLM agent execution. • ACID for Agents — Semantic Atomicity, Consistency, Isolation, and Durability. • ACID-Agent — a co...

</details>

<details>
<summary><b>11. Advancing Open and Reproducible Relational Learning: RelArena-α, TabPFN-Rel and RPI</b> ⭐ 29</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16319) • [📄 arXiv](https://arxiv.org/abs/2608.16319) • [📥 PDF](https://arxiv.org/pdf/2608.16319)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/PriorLabs/relarena)

> We’re happy to announce our first release in relational learning at Prior Labs, continuing our commitment to open science. We open-source three pieces of software that we expect to accelerate research in the field towards meaningful, real-world im...

</details>

<details>
<summary><b>12. How Do Agents Fail on AutoResearch: End-to-End Diagnostic Evaluation on 100 Real-World Frontier Research Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14905) • [📄 arXiv](https://arxiv.org/abs/2608.14905) • [📥 PDF](https://arxiv.org/pdf/2608.14905)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> How do agents fail at autonomous research? We stress-tested them on 100 real frontier-research tasks (800 trajectories, 8 harness–model combos). Every failure traced back to one thing — agents can't self-doubt or self-correct. The bottleneck is me...

</details>

<details>
<summary><b>13. GenRouter: Unified Workflow Routing for Agentic Image Generation</b> ⭐ 22</summary>

<br/>

**👥 Authors:** Yingjie Xu, Weilin Ruan, Wen-Jie Shu, Zhiyu Hou, Harold328

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16721) • [📄 arXiv](https://arxiv.org/abs/2608.16721) • [📥 PDF](https://arxiv.org/pdf/2608.16721)

**💻 Code:** [⭐ Code](https://github.com/EnVision-Research/GenRouter) • [⭐ Code](https://github.com/huggingface)

> GenCanvas: The first unified workflow space that standardizes the execution paradigm of agentic image generation. It systematically deconstructs the generative process into universal foundational primitives ( e.g. , search, reason, verify, and ske...

</details>

<details>
<summary><b>14. Understanding Cognition-Induced Risks in Agentic AI Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15304) • [📄 arXiv](https://arxiv.org/abs/2608.15304) • [📥 PDF](https://arxiv.org/pdf/2608.15304)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Frontier agentic systems powered by large language models (LLMs) exhibit human-like patterns of cognition. As these systems become deeply integrated across different domains, their cognitive engagement raises critical concerns for human society th...

</details>

<details>
<summary><b>15. MegaParts: Scaling Part-Aware 3D Object Generation to 300 Parts via Token-Efficient Autoregressive Modeling</b> ⭐ 515</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14783) • [📄 arXiv](https://arxiv.org/abs/2608.14783) • [📥 PDF](https://arxiv.org/pdf/2608.14783)

**💻 Code:** [⭐ Code](https://github.com/InternRobotics/MeshCoder) • [⭐ Code](https://github.com/huggingface)

> Paper: https://arxiv.org/abs/2608.14783 Github: https://github.com/InternRobotics/MeshCoder Project Page: https://expmaster.github.io/megaparts_webpage/

</details>

<details>
<summary><b>16. GRNEdit: Efficient General Video Editing from a New Binary-Evidence Perspective in Generative Refinement Networks</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Yuxuan Chen, Zepeng Wang, Fuhao Li, Jiagao Hu, Feng Xie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16328) • [📄 arXiv](https://arxiv.org/abs/2608.16328) • [📥 PDF](https://arxiv.org/pdf/2608.16328)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Foxerity/GRNEdit)

> No abstract available.

</details>

<details>
<summary><b>17. Ventor-QTest: Threat-Model-Driven Verification of Vendor-Hosted LLM APIs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16391) • [📄 arXiv](https://arxiv.org/abs/2608.16391) • [📥 PDF](https://arxiv.org/pdf/2608.16391)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/AI-Infra-Guard/tree/main/services/api_checker/ventor_qtest%7D)

> As large language models become increasingly widespread, third-party providers that deploy open-weight models have become an important part of the ecosystem.  Auditing the quality of their inference APIs is therefore an open problem.  We formalize...

</details>

<details>
<summary><b>18. R^3-Bench: LLMs Struggle with Resource-Rational Reasoning under Shared Budgets</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16033) • [📄 arXiv](https://arxiv.org/abs/2608.16033) • [📥 PDF](https://arxiv.org/pdf/2608.16033)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NineAbyss/R-3-Bench)

> Can LLMs spend a shared reasoning budget wisely? R³-Bench : A benchmark for resource-rational reasoning under shared computational budgets. Six-problem contests : Math, competitive programming, and abstract reasoning tasks compete for one shared t...

</details>

<details>
<summary><b>19. Improving the matrix multiplication exponent with modern optimization and AlphaEvolve</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16884) • [📄 arXiv](https://arxiv.org/abs/2608.16884) • [📥 PDF](https://arxiv.org/pdf/2608.16884)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>20. PACE-Bench: Benchmarking Physics Adaptation via Code Evolution in Dynamic Environments</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14441) • [📄 arXiv](https://arxiv.org/abs/2608.14441) • [📥 PDF](https://arxiv.org/pdf/2608.14441)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/thunlp/PACE-Bench)

> Self-evolving agents promise to improve through interaction. Yet most benchmarks keep execution conditions fixed. Adaptation matters when past success stops working. If the environment changes, can an agent recover? Introducing PACE-Bench. Code & ...

</details>

<details>
<summary><b>21. TRACE-Bench: Decomposing and Diagnosing Multi-Reference Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16765) • [📄 arXiv](https://arxiv.org/abs/2608.16765) • [📥 PDF](https://arxiv.org/pdf/2608.16765)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Amuseum-WHR/TraceBench)

> Despite recent advances in unified multimodal models for multi-reference image generation, existing benchmarks remain organized around predefined task types (e.g., "subject composition"), which are ill-suited to this combinatorial setting and lead...

</details>

<details>
<summary><b>22. VideoGAIA: A Benchmark for General AI Assistants on Agentic Video Understanding</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14718) • [📄 arXiv](https://arxiv.org/abs/2608.14718) • [📥 PDF](https://arxiv.org/pdf/2608.14718)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zfkarl/VideoGAIA)

> Code: https://github.com/zfkarl/VideoGAIA Data: https://huggingface.co/datasets/Karl28/VideoGAIA

</details>

<details>
<summary><b>23. AnyTalk: Speech Animation for Arbitrary Characters Leveraging a Video Generation Model</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16143) • [📄 arXiv](https://arxiv.org/abs/2608.16143) • [📥 PDF](https://arxiv.org/pdf/2608.16143)

**💻 Code:** [⭐ Code](https://github.com/kwanyun/AnyTalk_CsF%7D) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kwanyun/AnyTalk_CsF)

> Speech Animation using Video Diffusion Model Project page : { https://serin-yoon.github.io/projects/anytalk/} Code : { https://github.com/kwanyun/AnyTalk_CsF}

</details>

<details>
<summary><b>24. NaviDC-OCR: Navigating Document Parsing Across Digital and Camera-Captured Documents</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.12898) • [📄 arXiv](https://arxiv.org/abs/2608.12898) • [📥 PDF](https://arxiv.org/pdf/2608.12898)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/caipeng328/NaviDC-OCR)

> code： https://github.com/caipeng328/NaviDC-OCR paper： https://arxiv.org/pdf/2608.12898 huggingface： https://huggingface.co/StarDoc-AI/NaviDC-OCR

</details>

<details>
<summary><b>25. ConceptFormer: Learning Adaptive Latent Concepts for Query-Document Alignment in Visual Document Retrieval</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Yu Shi, Liu Zhenghao, Yan Yukun, Xu Zhipeng, hmhm1229

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15698) • [📄 arXiv](https://arxiv.org/abs/2608.15698) • [📥 PDF](https://arxiv.org/pdf/2608.15698)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NEUIR/ConceptFormer)

> Visual document retrieval is a critical component of multimodal retrieval-augmented generation, aiming to identify query-relevant pages from document collections where evidence is distributed across text, layout, charts, and visual structures. Rec...

</details>

<details>
<summary><b>26. Beyond Visual CoT: Internalized Visual Thinking for Proactive Video Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15869) • [📄 arXiv](https://arxiv.org/abs/2608.15869) • [📥 PDF](https://arxiv.org/pdf/2608.15869)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper introduces Internalized Visual Thinking (IVT), a post-training framework for proactive video reasoning that learns to predict future visual representations during training while retaining direct-answer inference. IVT consistently improv...

</details>

<details>
<summary><b>27. ENTLORE: A Graph-Grounded Benchmark for Latent Organizational Reasoning in Enterprise Question Answering</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.10679) • [📄 arXiv](https://arxiv.org/abs/2608.10679) • [📥 PDF](https://arxiv.org/pdf/2608.10679)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/scitix/entlore)

> a benchmark framework that evaluates enterprise question answering by requiring recovery of implicit organizational relations across routine documents, revealing that even with gold sources many latent reasoning questions remain unanswered.

</details>

<details>
<summary><b>28. Prior Audit-Repair Context Shifts LLM Verifier Thresholds Toward Leniency</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kasra Mazaheri, parsa-mz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16003) • [📄 arXiv](https://arxiv.org/abs/2608.16003) • [📥 PDF](https://arxiv.org/pdf/2608.16003)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/parsa-mz/crtitxer)

> A verifier that flags fewer errors has not necessarily become more accurate. It may only have moved its threshold. We looked for the accuracy gain on three open-weight models, at the quantity the improvement account predicts - discrimination - and...

</details>

<details>
<summary><b>29. Drive, Pack, Fly: The Travelling Thief Problem with Drone</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Abhay Sobhanan, Kabir Murjani

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16435) • [📄 arXiv](https://arxiv.org/abs/2608.16435) • [📥 PDF](https://arxiv.org/pdf/2608.16435)

**💻 Code:** [⭐ Code](https://github.com/corbit-lab/ttpd) • [⭐ Code](https://github.com/huggingface)

> In collection operations, accumulating payload progressively slows the vehicle, imposing a cumulative penalty on routing efficiency. An onboard drone can offset this penalty by retrieving outlying items, thereby shortening the makespan and increas...

</details>

<details>
<summary><b>30. Gathered, Not Admitted: How Attention Brings a Latent Variable into Verbalizable Form</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15022) • [📄 arXiv](https://arxiv.org/abs/2608.15022) • [📥 PDF](https://arxiv.org/pdf/2608.15022)

**💻 Code:** [⭐ Code](https://github.com/parsa-mz/innerj) • [⭐ Code](https://github.com/huggingface)

> A latent variable does not reach a language model's self-report because a gate opens. It gets there because attention carries it. We looked for that gate on an open-weight model, at the position the account predicts, and it is not there. What we f...

</details>

<details>
<summary><b>31. DumpsterCluster: From Dumpster Diving to Serving LLaMA-70B on $60 GPUs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14614) • [📄 arXiv](https://arxiv.org/abs/2608.14614) • [📥 PDF](https://arxiv.org/pdf/2608.14614)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Turns out you can use very very old hardware to serve relatively large models with a magnitude reduction in throughput

</details>

<details>
<summary><b>32. Plausible but Not Valid: A Psychometric Audit of LLMs as Synthetic Survey Respondents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Viktorija Šarkauskaitė, MLukauskas

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14606) • [📄 arXiv](https://arxiv.org/abs/2608.14606) • [📥 PDF](https://arxiv.org/pdf/2608.14606)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> LLMs can generate very plausible survey responses - but are they psychometrically valid? We find that plausibility often hides unstable constructs, weak reliability, and poor agreement with real human response patterns. The paper was originally su...

</details>

<details>
<summary><b>33. HarmProfile: Characterizing Harmful Distributions in Frontier LLMs</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Xiao Liu, Xiang Zheng, Yutao Wu, Zhouyuan Ma, hanxunh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14577) • [📄 arXiv](https://arxiv.org/abs/2608.14577) • [📥 PDF](https://arxiv.org/pdf/2608.14577)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/fresh-ma/HarmProfile)

> HarmProfile is a benchmark dataset that characterizes frontier LLM safety failures through content analysis, revealing that harmfulness and diversity increase with model capability.

</details>

<details>
<summary><b>34. A Plug-and-Play 2D Motion Interface for Real-World Motion Language Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Norimichi Ukita, KanameYOkoYAMA

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15984) • [📄 arXiv](https://arxiv.org/abs/2608.15984) • [📥 PDF](https://arxiv.org/pdf/2608.15984)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/irajisamurai/2D-Motion-Interface)

> Plug-and-play 2D motion interface for real-world Motion Language Models

</details>

<details>
<summary><b>35. WorldRover: A Scalable Synthetic Video Data Engine for World Exploration with Rich Annotations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15659) • [📄 arXiv](https://arxiv.org/abs/2608.15659) • [📥 PDF](https://arxiv.org/pdf/2608.15659)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A Scalable Synthetic Video Data Engine for World Exploration with Rich Annotations

</details>

<details>
<summary><b>36. Accuracy and Order Sensitivity Diverge Under Label-Free Strategies</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.11947) • [📄 arXiv](https://arxiv.org/abs/2608.11947) • [📥 PDF](https://arxiv.org/pdf/2608.11947)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cotenthusiast/choicebench)

> We study how MCQ accuracy and sensitivity to option order can diverge under label-free evaluation strategies. Alongside the paper, we release ChoiceBench, a modular open-source framework for evaluating and comparing MCQ evaluation methods across m...

</details>

<details>
<summary><b>37. When Context Bites: Detecting RAG Poisoning via Document-Level Attention Collapse</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu-Cheng Chang, Xiao Luo, Yiwei Fu, Ziyi Zhao, An998

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06947) • [📄 arXiv](https://arxiv.org/abs/2608.06947) • [📥 PDF](https://arxiv.org/pdf/2608.06947)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> RAG poisoning attacks can manipulate LLM outputs by injecting malicious documents into the retrieval context. Existing detectors often rely on output uncertainty, which can fail when poisoned generations appear deceptively confident. In this work,...

</details>

<details>
<summary><b>38. Position: AI Agents in Scientific Teams Should Be Studied as Human-Agent Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14667) • [📄 arXiv](https://arxiv.org/abs/2608.14667) • [📥 PDF](https://arxiv.org/pdf/2608.14667)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This position paper argues that AI agents used in scientific discovery should be studied as human-agent systems, not just evaluated on their standalone capabilities. The authors review current "AI Scientist" systems and find most only let humans s...

</details>

<details>
<summary><b>39. Valid Per-Field Selective Risk Control for Document Extraction: Three Failure Modes, a Validity Ladder, and When Conditioning Pays</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bhaskar Gurram

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14639) • [📄 arXiv](https://arxiv.org/abs/2608.14639) • [📥 PDF](https://arxiv.org/pdf/2608.14639)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/bhaskargurram-ai/verifydoc)

> Author here (Bhaskar Gurram, Zasti AI). The abstract is dense, so here's what the paper actually argues. The claim "Accept a field only if the error rate among accepted fields stays under α" is the trust contract every document-extraction system i...

</details>

<details>
<summary><b>40. StreamOPD: A Post-Training Recipe with Spatio-Temporal Cue Gating for Streaming Video Understanding</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Zuhao Yang, Kaichen Zhang, Baoyi Wang, Keming Wu, xiangan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16320) • [📄 arXiv](https://arxiv.org/abs/2608.16320) • [📥 PDF](https://arxiv.org/pdf/2608.16320)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/UniX-AI-Lab/StreamOPD)

> Streaming video understanding demands direct responses from the causally observed prefix of an unfolding video. Existing systems add inference-time memory, retrieval, and compression, yet a training-free sliding-window baseline already matches the...

</details>

<details>
<summary><b>41. HiFi-BRep: High-Fidelity Latent Representation for Robust B-Rep Generation</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Yusheng Liu, Jiaying Lu, Pufan Wang, Chenqi Luo, Junhao Hou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16485) • [📄 arXiv](https://arxiv.org/abs/2608.16485) • [📥 PDF](https://arxiv.org/pdf/2608.16485)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/1nnoh/HiFi-BRep)

> Code and dataset are available at https://github.com/1nnoh/HiFi-BRep

</details>

<details>
<summary><b>42. Prototype-Rectified Iterative Self-supervised Manifold Denoising under Severe Acoustic Shift</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Vinod K. Kurmi, Aryan Das, Rini Smita Thakur, ashish1108

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15037) • [📄 arXiv](https://arxiv.org/abs/2608.15037) • [📥 PDF](https://arxiv.org/pdf/2608.15037)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Ashish-1108/PRISM)

> Audio-Text Foundation Models (ATMs) fail catastrophically under severe acoustic noise, yet existing adaptation strategies either rely on gradient-based Test-Time Adaptation (TTA), which reinforces noise rather than signal, or on prompt tuning that...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 42 |
| 📅 Today | [`2026-08-19.json`](data/daily/2026-08-19.json) | 42 |
| 📆 This Week | [`2026-W33.json`](data/weekly/2026-W33.json) | 106 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 455 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-19 | 42 | [View JSON](data/daily/2026-08-19.json) |
| 📄 2026-08-18 | 32 | [View JSON](data/daily/2026-08-18.json) |
| 📄 2026-08-17 | 32 | [View JSON](data/daily/2026-08-17.json) |
| 📄 2026-08-16 | 32 | [View JSON](data/daily/2026-08-16.json) |
| 📄 2026-08-15 | 32 | [View JSON](data/daily/2026-08-15.json) |
| 📄 2026-08-14 | 1 | [View JSON](data/daily/2026-08-14.json) |
| 📄 2026-08-13 | 3 | [View JSON](data/daily/2026-08-13.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W33 | 106 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |
| 📅 2026-W30 | 112 | [View JSON](data/weekly/2026-W30.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 455 | [View JSON](data/monthly/2026-08.json) |
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
