<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-26-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5969+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">26</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">26</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">588</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5969+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 24, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. EnvHarness: Awakening Static Worlds for Agent Learning</b> ⭐ 279</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19880) • [📄 arXiv](https://arxiv.org/abs/2608.19880) • [📥 PDF](https://arxiv.org/pdf/2608.19880)

**💻 Code:** [⭐ Code](https://github.com/google-research/envharness) • [⭐ Code](https://github.com/huggingface)

> webpage: https://envharness.com/ code are all released.

</details>

<details>
<summary><b>2. FACET: Preserving Source Intent and Executable State in Terminal Task Synthesis</b> ⭐ 47</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18580) • [📄 arXiv](https://arxiv.org/abs/2608.18580) • [📥 PDF](https://arxiv.org/pdf/2608.18580)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/StoKou/FACET-Terminal)

> Training terminal agents requires scalable executable supervision, yet synthesizing high-quality terminal tasks remains challenging. Each task couples an instruction, an initialized environment, a reference solution, and an executable verifier; if...

</details>

<details>
<summary><b>3. 4DAnyone: Create Anyone in 4D from a Casual Monocular Video</b> ⭐ 492</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20335) • [📄 arXiv](https://arxiv.org/abs/2608.20335) • [📥 PDF](https://arxiv.org/pdf/2608.20335)

**💻 Code:** [⭐ Code](https://github.com/ant-research/4DAnyone) • [⭐ Code](https://github.com/huggingface)

> 4DAnyone turns a casual monocular video into multi-view videos, enabling downstream 4DGS reconstruction.

</details>

<details>
<summary><b>4. SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?</b> ⭐ 50</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19799) • [📄 arXiv](https://arxiv.org/abs/2608.19799) • [📥 PDF](https://arxiv.org/pdf/2608.19799)

**💻 Code:** [⭐ Code](https://github.com/OpenMOSS/SWE-bench-Science) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/OpenMOSS/SWE-bench-Science Data: https://huggingface.co/datasets/OpenMOSS-Team/SWE-bench-Science Leaderboard: https://swescience.github.io

</details>

<details>
<summary><b>5. WithEveryone: Unified Planning and Identity Grounding for Group Image Generation</b> ⭐ 43</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20336) • [📄 arXiv](https://arxiv.org/abs/2608.20336) • [📥 PDF](https://arxiv.org/pdf/2608.20336)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/doby-xu/WithEveryone)

> WithEveryone generates coherent group images from five to ten reference identities

</details>

<details>
<summary><b>6. MemTrapBench: Benchmarking Cognitive Traps in LLM Memory Use</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20202) • [📄 arXiv](https://arxiv.org/abs/2608.20202) • [📥 PDF](https://arxiv.org/pdf/2608.20202)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zjunlp/MemTrapBench)

> Memory is NOT always what you need, as it may impair rather than enhance model capabilities.

</details>

<details>
<summary><b>7. SkillEvo: Self-Renewing Evolution Gradients from Multi-Turn Interaction Feedback</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13120) • [📄 arXiv](https://arxiv.org/abs/2608.13120) • [📥 PDF](https://arxiv.org/pdf/2608.13120)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Up～

</details>

<details>
<summary><b>8. ForgeWM: Progressive Causal Training for Few-Step Action-Conditioned Video World Models</b> ⭐ 95</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14022) • [📄 arXiv](https://arxiv.org/abs/2608.14022) • [📥 PDF](https://arxiv.org/pdf/2608.14022)

**💻 Code:** [⭐ Code](https://github.com/asdfo123/ForgeWM) • [⭐ Code](https://github.com/huggingface)

> Train real-time, playable video world models. ForgeWM is fully open and reproducible, with keyboard, mouse, and gamepad control. 🔨

</details>

<details>
<summary><b>9. Repo0: Design-Driven Zero-to-All Code Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19854) • [📄 arXiv](https://arxiv.org/abs/2608.19854) • [📥 PDF](https://arxiv.org/pdf/2608.19854)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Repo0 doesn’t just generate code—it builds the architecture as it goes. It continuously reshapes the repository with cohesion and coupling until the structure truly makes sense.

</details>

<details>
<summary><b>10. FlashPrefill V2: Block-Sparse Prefill Attention for Long-Context LLM Serving</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19758) • [📄 arXiv](https://arxiv.org/abs/2608.19758) • [📥 PDF](https://arxiv.org/pdf/2608.19758)

**💻 Code:** [⭐ Code](https://github.com/qhfan/FlashPrefillv2) • [⭐ Code](https://github.com/huggingface)

> Long-context modeling is a pivotal capability for Large Language Models, yet the quadratic complexity of attention remains a critical bottleneck, particularly during the compute-intensive prefilling phase. Our previous work, FlashPrefill, mitigate...

</details>

<details>
<summary><b>11. Thinking in a Low-Resource Language: What SFT Builds, What RL Fixes, What Accuracy Cannot See</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17744) • [📄 arXiv](https://arxiv.org/abs/2608.17744) • [📥 PDF](https://arxiv.org/pdf/2608.17744)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Author here 👋 We fine-tuned four MoE models (Qwen / gpt-oss / Nemotron families, 3.6–4.0B active params) to reason in Greek, and the headline is a null: accuracy barely moves. Worse, the benchmark can't see it anyway — changing only the random see...

</details>

<details>
<summary><b>12. τ_0-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation</b> ⭐ 528</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16885) • [📄 arXiv](https://arxiv.org/abs/2608.16885) • [📥 PDF](https://arxiv.org/pdf/2608.16885)

**💻 Code:** [⭐ Code](https://github.com/sii-research/tau-0-vla) • [⭐ Code](https://github.com/huggingface)

> 🤖 What if a robot could compare possible futures before deciding what to do next? We introduce τ₀-VLA , a hierarchical robot foundation model for long-horizon manipulation. Its high-level policy maintains execution memory and, when a decision is u...

</details>

<details>
<summary><b>13. EXIMO: VLM Guided Exploration of VLA Policies</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19891) • [📄 arXiv](https://arxiv.org/abs/2608.19891) • [📥 PDF](https://arxiv.org/pdf/2608.19891)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Beyond Flat Policies: Hierarchical Post-Training for Embodied Agents in Rob...

</details>

<details>
<summary><b>14. The Embedder's Dilemma: LLMs Are Better, but at What Cost?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jinhyuk Lee, Niklas Muennighoff, AdnanElAssadi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.12875) • [📄 arXiv](https://arxiv.org/abs/2608.12875) • [📥 PDF](https://arxiv.org/pdf/2608.12875)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Embeddings vs LLMs ...

</details>

<details>
<summary><b>15. Hierarchical Self-Improvement: A Framework for Task-Specific Evolvable Agent Harnesses</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.08466) • [📄 arXiv](https://arxiv.org/abs/2608.08466) • [📥 PDF](https://arxiv.org/pdf/2608.08466)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/TailinZhou/hsi)

> Researchers at HKUST developed Hierarchical Self-Improvement (HSI), a framework enabling a single, frozen LLM to autonomously evolve its task-specific operational harness and the strategy for its own evolution. The system achieved substantial perf...

</details>

<details>
<summary><b>16. Towards Quantifying Benchmark Optimization in ASR Models</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19936) • [📄 arXiv](https://arxiv.org/abs/2608.19936) • [📥 PDF](https://arxiv.org/pdf/2608.19936)

**💻 Code:** [⭐ Code](https://github.com/HumeAI/asr-benchmark-optimization) • [⭐ Code](https://github.com/huggingface)

> New work from Hume AI on quantifying how an ASR model reproduces a benchmark's reference text rather than transcribing the audio

</details>

<details>
<summary><b>17. Inject, Align, Recover: Staged Post-Training for Retrieval-Free Document Knowledge Internalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20281) • [📄 arXiv](https://arxiv.org/abs/2608.20281) • [📥 PDF](https://arxiv.org/pdf/2608.20281)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> What happens when a language model must answer questions about a fixed document collection without receiving retrieved passages at inference time? We study this problem as document knowledge internalization and introduce IAR (Inject, Align, Recove...

</details>

<details>
<summary><b>18. TinyCast: Probabilistic Zero-Shot Forecasting with Computed Periodicity</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15767) • [📄 arXiv](https://arxiv.org/abs/2608.15767) • [📥 PDF](https://arxiv.org/pdf/2608.15767)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/raws-labs/tinycast)

> TinyCast is a 146,505-parameter zero-shot time series foundation model. It is the smallest model on the GIFT-Eval board with a public per-configuration result and no declared test-data leakage, and below 1.4M parameters it is the only zero-shot en...

</details>

<details>
<summary><b>19. PolicyGuide: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19861) • [📄 arXiv](https://arxiv.org/abs/2608.19861) • [📥 PDF](https://arxiv.org/pdf/2608.19861)

**💻 Code:** [⭐ Code](https://github.com/erjui/PolicyGuide) • [⭐ Code](https://github.com/huggingface)

> PolicyGuide: From Guarding One Action to Guiding the Whole Workflow for Policy-Compliant LLM Agents

</details>

<details>
<summary><b>20. Chain-of-Experience for Continual LLM Improvement</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18027) • [📄 arXiv](https://arxiv.org/abs/2608.18027) • [📥 PDF](https://arxiv.org/pdf/2608.18027)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Comprehensive investigation into LLM's learning from experience capabilities across several domains and tasks.

</details>

<details>
<summary><b>21. NARU: A Benchmark for NARrative Evolution and Cultural Nuance Understanding in Japanese Extreme Long Video</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13210) • [📄 arXiv](https://arxiv.org/abs/2608.13210) • [📥 PDF](https://arxiv.org/pdf/2608.13210)

**💻 Code:** [⭐ Code](https://github.com/infinimind-inc/naru_benchmark) • [⭐ Code](https://github.com/huggingface)

> Long-form video understanding encompasses tasks that go beyond retrieving isolated events, including tracking an evolving narrative and interpreting social meaning that may remain implicit. However, existing benchmarks rarely evaluate these capabi...

</details>

<details>
<summary><b>22. QuoteBench: How Matched Scores Can Hide Command-Path Failures</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Yuanyuan Yang, Volker Tresp, Yao Zhang, lsamc

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13547) • [📄 arXiv](https://arxiv.org/abs/2608.13547) • [📥 PDF](https://arxiv.org/pdf/2608.13547)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LeonardNJU/quoteBench)

> QuoteBench asks a deployment question that matched scores can hide: did the model generate a bad Bash command, or did the execution interface break it afterward? Across 56 execution-verified tasks, replaying the same reply through one added parser...

</details>

<details>
<summary><b>23. GOAG: Generative and Object-Agnostic Grasp Planner for Dexterous Robotic Manipulation</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Liming Chen, Mathieu Grossard, Boris Meden, JulienMERAND

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19759) • [📄 arXiv](https://arxiv.org/abs/2608.19759) • [📥 PDF](https://arxiv.org/pdf/2608.19759)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/CEA-LIST/GOAG)

> GOAG Paradigm: A successful grasp on an object induces dual contact zones on both object and gripper, at the intersection of the two geometries. Our method is built on this key observation: these contact zones are closely the same from either pers...

</details>

<details>
<summary><b>24. CoToGrasp: Contact-Topology-Conditioned Dexterous Grasp Synthesis via Canonical Workspace Learning</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Mathieu Grossard, Liming Chen, Boris Meden, JulienMERAND

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19776) • [📄 arXiv](https://arxiv.org/abs/2608.19776) • [📥 PDF](https://arxiv.org/pdf/2608.19776)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/CEA-LIST/CoToGrasp)

> Our framework synthesizes stable, functionally diverse grasps for unseen objects by decoupling functional intent from geometry. By learning a latent manifold in a canonical workspace, we achieve zero-shot generalization across multiple contact top...

</details>

<details>
<summary><b>25. Listening Forward: Next Patch Embedding Prediction Enables Scalable Audio Learners</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19863) • [📄 arXiv](https://arxiv.org/abs/2608.19863) • [📥 PDF](https://arxiv.org/pdf/2608.19863)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/umbertocappellazzo/nape)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API AV-JEPA: Extending LeJEPA to Audio-Visual Self-Supervised Learning (2026) D...

</details>

<details>
<summary><b>26. FlowEvo: Self-Evolving Agents through the Co-Evolution of Workflows and Executable Skills</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.21596) • [📄 arXiv](https://arxiv.org/abs/2607.21596) • [📥 PDF](https://arxiv.org/pdf/2607.21596)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DEFENSE-SEU/FlowEvo)

> This paper addresses a practical bottleneck in agent systems: useful workflows discovered during inference are usually discarded, while existing skill libraries are often static or built offline. FlowEvo closes this loop by turning verified succes...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 26 |
| 📅 Today | [`2026-08-24.json`](data/daily/2026-08-24.json) | 26 |
| 📆 This Week | [`2026-W34.json`](data/weekly/2026-W34.json) | 26 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 588 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-24 | 26 | [View JSON](data/daily/2026-08-24.json) |
| 📄 2026-08-23 | 26 | [View JSON](data/daily/2026-08-23.json) |
| 📄 2026-08-22 | 26 | [View JSON](data/daily/2026-08-22.json) |
| 📄 2026-08-21 | 22 | [View JSON](data/daily/2026-08-21.json) |
| 📄 2026-08-20 | 33 | [View JSON](data/daily/2026-08-20.json) |
| 📄 2026-08-19 | 42 | [View JSON](data/daily/2026-08-19.json) |
| 📄 2026-08-18 | 32 | [View JSON](data/daily/2026-08-18.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W34 | 26 | [View JSON](data/weekly/2026-W34.json) |
| 📅 2026-W33 | 213 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 588 | [View JSON](data/monthly/2026-08.json) |
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
