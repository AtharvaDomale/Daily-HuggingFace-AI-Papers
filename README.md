<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-28-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2567+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">28</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">156</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">1048</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2567+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 28, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. The Trinity of Consistency as a Defining Principle for General World Models</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23152) • [📄 arXiv](https://arxiv.org/abs/2602.23152) • [📥 PDF](https://arxiv.org/pdf/2602.23152)

**💻 Code:** [⭐ Code](https://github.com/openraiser/awesome-world-model-evolution)

> The construction of World Models capable of learning, simulating, and reasoning about objective physical laws constitutes a foundational challenge in the pursuit of Artificial General Intelligence. Recent advancements represented by video generati...

</details>

<details>
<summary><b>2. From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models</b> ⭐ 27</summary>

<br/>

**👥 Authors:** Wei Ye, Shikun Zhang, Chaoya Jiang, hongruijia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22859) • [📄 arXiv](https://arxiv.org/abs/2602.22859) • [📥 PDF](https://arxiv.org/pdf/2602.22859)

**💻 Code:** [⭐ Code](https://github.com/hongruijia/DPE)

> DPE (Diagnostic-driven Progressive Evolution) is a self-evolving training framework for Large Multimodal Models (LMMs). Inspired by the "diagnose-and-correct" mechanism in educational psychology, DPE moves beyond indiscriminate data expansion. It ...

</details>

<details>
<summary><b>3. MobilityBench: A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios</b> ⭐ 94</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22638) • [📄 arXiv](https://arxiv.org/abs/2602.22638) • [📥 PDF](https://arxiv.org/pdf/2602.22638)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/MobilityBench)

> A Benchmark for Evaluating Route-Planning Agents in Real-World Mobility Scenarios MobilityBench is a scalable benchmark for evaluating LLM-based route-planning agents in real-world mobility scenarios. It is built from large-scale, anonymized user ...

</details>

<details>
<summary><b>4. OmniGAIA: Towards Native Omni-Modal AI Agents</b> ⭐ 34</summary>

<br/>

**👥 Authors:** Shijian Wang, Jiarui Jin, Xiaoxi Li, dongguanting, wxjiao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22897) • [📄 arXiv](https://arxiv.org/abs/2602.22897) • [📥 PDF](https://arxiv.org/pdf/2602.22897)

**💻 Code:** [⭐ Code](https://github.com/RUC-NLPIR/OmniGAIA)

> 💡 Overview OmniGAIA is a comprehensive benchmark designed to evaluate the capabilities of omni-modal general AI assistants. Unlike existing benchmarks that focus on a single modality, OmniGAIA requires agents to jointly reason over video , audio ,...

</details>

<details>
<summary><b>5. Imagination Helps Visual Reasoning, But Not Yet in Latent Space</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22766) • [📄 arXiv](https://arxiv.org/abs/2602.22766) • [📥 PDF](https://arxiv.org/pdf/2602.22766)

**💻 Code:** [⭐ Code](https://github.com/AI9Stars/CapImagine)

> Rethinking the role of "latent" in latent visual reasoning

</details>

<details>
<summary><b>6. Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23008) • [📄 arXiv](https://arxiv.org/abs/2602.23008) • [📥 PDF](https://arxiv.org/pdf/2602.23008)

> No abstract available.

</details>

<details>
<summary><b>7. AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23258) • [📄 arXiv](https://arxiv.org/abs/2602.23258) • [📥 PDF](https://arxiv.org/pdf/2602.23258)

**💻 Code:** [⭐ Code](https://github.com/TonySY2/AgentDropoutV2)

> We introduce AgentDropoutV2, a test-time pruning framework for multi-agent systems based on a rectify-or-reject strategy to actively intercept and iteratively correct erroneous outputs, achieving performance gain on the math and code domains.

</details>

<details>
<summary><b>8. MediX-R1: Open Ended Medical Reinforcement Learning</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23363) • [📄 arXiv](https://arxiv.org/abs/2602.23363) • [📥 PDF](https://arxiv.org/pdf/2602.23363)

**💻 Code:** [⭐ Code](https://github.com/mbzuai-oryx/MediX-R1)

> MediX-R1 is an open-ended Reinforcement Learning (RL) framework for medical multimodal large language models (MLLMs) that enables clinically grounded, free-form answers beyond multiple-choice formats. MediX-R1 fine-tunes vision-language backbones ...

</details>

<details>
<summary><b>9. Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22675) • [📄 arXiv](https://arxiv.org/abs/2602.22675) • [📥 PDF](https://arxiv.org/pdf/2602.22675)

**💻 Code:** [⭐ Code](https://github.com/OPPO-PersonalAI/SMTL)

> Recent deep research agents often improve performance by scaling reasoning depth. While effective, this approach significantly increases inference cost and latency in search-intensive settings, and often struggles to generalize across heterogeneou...

</details>

<details>
<summary><b>10. VGG-T^3: Offline Feed-Forward 3D Reconstruction at Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23361) • [📄 arXiv](https://arxiv.org/abs/2602.23361) • [📥 PDF](https://arxiv.org/pdf/2602.23361)

> Traditional offline 3D reconstruction methods scale quadratically, making large scenes a massive computational burden. VGG-T³ (Visual Geometry Grounded Test-Time Training) achieves linear scaling by replacing standard global attention with a TTT-b...

</details>

<details>
<summary><b>11. Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Jae-Gil Lee, Seonghye Cho, Hyunjin Kim, Byunghyun Kim, jyssys

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21760) • [📄 arXiv](https://arxiv.org/abs/2602.21760) • [📥 PDF](https://arxiv.org/pdf/2602.21760)

**💻 Code:** [⭐ Code](https://github.com/kaist-dmlab/Hybridiff)

> Accepted to CVPR'26

</details>

<details>
<summary><b>12. EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents</b> ⭐ 45</summary>

<br/>

**👥 Authors:** Xuqian Ren, Yuke Lou, Huaijin Pi, Liang Pan, WenjiaWang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23205) • [📄 arXiv](https://arxiv.org/abs/2602.23205) • [📥 PDF](https://arxiv.org/pdf/2602.23205)

**💻 Code:** [⭐ Code](https://github.com/WenjiaWang0312/EmbodMocap)

> No abstract available.

</details>

<details>
<summary><b>13. AI Gamestore: Scalable, Open-Ended Evaluation of Machine General Intelligence with Human Games</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17594) • [📄 arXiv](https://arxiv.org/abs/2602.17594) • [📥 PDF](https://arxiv.org/pdf/2602.17594)

> Rigorously evaluating machine intelligence against the broad spectrum of human general intelligence has become increasingly important and challenging in this era of rapid technological advance. Conventional AI benchmarks typically assess only narr...

</details>

<details>
<summary><b>14. Causal Motion Diffusion Models for Autoregressive Motion Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kent Fujiwara, Akihisa Watanabe, Qing Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22594) • [📄 arXiv](https://arxiv.org/abs/2602.22594) • [📥 PDF](https://arxiv.org/pdf/2602.22594)

> Proposes Causal Motion Diffusion Models (CMDM) using a causal diffusion transformer and MAC-VAE latent space for autoregressive, streaming, long-horizon text-to-motion with improved fidelity and reduced latency.

</details>

<details>
<summary><b>15. Retrieve and Segment: Are a Few Examples Enough to Bridge the Supervision Gap in Open-Vocabulary Segmentation?</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23339) • [📄 arXiv](https://arxiv.org/abs/2602.23339) • [📥 PDF](https://arxiv.org/pdf/2602.23339)

**💻 Code:** [⭐ Code](https://github.com/TilemahosAravanis/Retrieve-and-Segment)

> Open-vocabulary segmentation (OVS) extends the zero-shot recognition capabilities of vision–language models (VLMs) to pixel-level prediction, enabling segmentation of arbitrary categories specified by text prompts. Despite recent progress, OVS lag...

</details>

<details>
<summary><b>16. GeoWorld: Geometric World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Richard Hartley, Ian Reid, Danning Li, SteveZeyuZhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23058) • [📄 arXiv](https://arxiv.org/abs/2602.23058) • [📥 PDF](https://arxiv.org/pdf/2602.23058)

> Accepted to CVPR 2026.

</details>

<details>
<summary><b>17. General Agent Evaluation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22953) • [📄 arXiv](https://arxiv.org/abs/2602.22953) • [📥 PDF](https://arxiv.org/pdf/2602.22953)

**💻 Code:** [⭐ Code](https://github.com/Exgentic/exgentic)

> 🚀 Welcome to the Open General Agent Era! 🤖🌍 What if we could finally measure how general our “general-purpose agents” really are? This paper takes on that challenge head-first 💥 Instead of evaluating agents in carefully engineered, domain-specific...

</details>

<details>
<summary><b>18. veScale-FSDP: Flexible and High-Performance FSDP at Scale</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Cong Xie, Jiacheng Yang, Zhiqi Lin, Youjie Li, Zezhou Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22437) • [📄 arXiv](https://arxiv.org/abs/2602.22437) • [📥 PDF](https://arxiv.org/pdf/2602.22437)

**💻 Code:** [⭐ Code](https://github.com/volcengine/veScale)

> Github: https://github.com/volcengine/veScale

</details>

<details>
<summary><b>19. DLT-Corpus: A Large-Scale Text Collection for the Distributed Ledger Technology Domain</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22045) • [📄 arXiv](https://arxiv.org/abs/2602.22045) • [📥 PDF](https://arxiv.org/pdf/2602.22045)

**💻 Code:** [⭐ Code](https://github.com/dlt-science/DLT-Corpus)

> We introduce DLT-Corpus , the largest domain-specific text collection for Distributed Ledger Technology (DLT) research to date: 2.98 billion tokens from 22.12 million documents spanning scientific literature (37,440 publications), United States Pa...

</details>

<details>
<summary><b>20. MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Yiming Xiao, Berardino Barile, Omid Nejati Manzari, Hojat Asgariandehkordi, TahaKoleilat

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20423) • [📄 arXiv](https://arxiv.org/abs/2602.20423) • [📥 PDF](https://arxiv.org/pdf/2602.20423)

**💻 Code:** [⭐ Code](https://github.com/HealthX-Lab/MedCLIPSeg)

> MedCLIPSeg introduces a probabilistic adaptation of CLIP for medical image segmentation, addressing key challenges such as limited annotations, ambiguous anatomical boundaries, and domain shift across imaging devices and institutions. The method p...

</details>

<details>
<summary><b>21. No One Size Fits All: QueryBandits for Hallucination Mitigation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Manuela Veloso, Sumitra Ganesh, Alec Koppel, William Watson, Nicole Cho

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20332) • [📄 arXiv](https://arxiv.org/abs/2602.20332) • [📥 PDF](https://arxiv.org/pdf/2602.20332)

> Advanced reasoning capabilities in Large Language Models (LLMs) have led to more frequent hallucinations; yet most mitigation work focuses on open-source models for post-hoc detection and parameter editing. The dearth of studies focusing on halluc...

</details>

<details>
<summary><b>22. What Makes a Good Query? Measuring the Impact of Human-Confusing Linguistic Features on LLM Performance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Manuela Veloso, Sumitra Ganesh, Nicole Cho, William Watson

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20300) • [📄 arXiv](https://arxiv.org/abs/2602.20300) • [📥 PDF](https://arxiv.org/pdf/2602.20300)

> Large Language Model (LLM) hallucinations are usually treated as defects of the model or its decoding strategy. Drawing on classical linguistics, we argue that a query's form can also shape a listener's (and model's) response. We operationalize th...

</details>

<details>
<summary><b>23. Risk-Aware World Model Predictive Control for Generalizable End-to-End Autonomous Driving</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jian-Fang Hu, Chang Liu, Teng Long, Feng Xue, Jiangxin Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23259) • [📄 arXiv](https://arxiv.org/abs/2602.23259) • [📥 PDF](https://arxiv.org/pdf/2602.23259)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Generative Scenario Rollouts for End-to-End Autonomous Driving (2026) Drive...

</details>

<details>
<summary><b>24. DyaDiT: A Multi-Modal Diffusion Transformer for Socially Favorable Dyadic Gesture Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haiyang Liu, Ruofan Liu, Siyeol Jung, Jyun-Ting Song, Yichen Peng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23165) • [📄 arXiv](https://arxiv.org/abs/2602.23165) • [📥 PDF](https://arxiv.org/pdf/2602.23165)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API RSATalker: Realistic Socially-Aware Talking Head Generation for Multi-Turn ...

</details>

<details>
<summary><b>25. Overconfident Errors Need Stronger Correction: Asymmetric Confidence Penalties for Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21420) • [📄 arXiv](https://arxiv.org/abs/2602.21420) • [📥 PDF](https://arxiv.org/pdf/2602.21420)

> The paper identifies overconfident wrong reasoning paths as a hidden failure mode in RL training of LLMs and proposes ACE, a simple penalty that suppresses these errors while preserving exploration—improving reasoning diversity and Pass@k performa...

</details>

<details>
<summary><b>26. MEG-to-MEG Transfer Learning and Cross-Task Speech/Silence Detection with Limited Data</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18253) • [📄 arXiv](https://arxiv.org/abs/2602.18253) • [📥 PDF](https://arxiv.org/pdf/2602.18253)

**💻 Code:** [⭐ Code](https://github.com/hitz-zentroa/meg-phone-decoding)

> We show that large-scale pretraining on a single-subject MEG dataset (50h) can improve speech decoding performance when fine-tuned on only ~5 minutes of data per new subject. Beyond in-task gains, transfer learning enables robust cross-task decodi...

</details>

<details>
<summary><b>27. Efficient Continual Learning in Language Models via Thalamically Routed Cortical Columns</b> ⭐ 0</summary>

<br/>

**👥 Authors:** akhadangi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22479) • [📄 arXiv](https://arxiv.org/abs/2602.22479) • [📥 PDF](https://arxiv.org/pdf/2602.22479)

> Continual learning is a core requirement for deployed language models, yet standard training and fine tuning pipelines remain brittle under non-stationary data. Online updates often induce catastrophic forgetting, while methods that improve stabil...

</details>

<details>
<summary><b>28. Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20981) • [📄 arXiv](https://arxiv.org/abs/2602.20981) • [📥 PDF](https://arxiv.org/pdf/2602.20981)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 28 |
| 📅 Today | [`2026-02-28.json`](data/daily/2026-02-28.json) | 28 |
| 📆 This Week | [`2026-W08.json`](data/weekly/2026-W08.json) | 156 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 1048 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-28 | 28 | [View JSON](data/daily/2026-02-28.json) |
| 📄 2026-02-27 | 30 | [View JSON](data/daily/2026-02-27.json) |
| 📄 2026-02-26 | 32 | [View JSON](data/daily/2026-02-26.json) |
| 📄 2026-02-25 | 25 | [View JSON](data/daily/2026-02-25.json) |
| 📄 2026-02-24 | 18 | [View JSON](data/daily/2026-02-24.json) |
| 📄 2026-02-23 | 23 | [View JSON](data/daily/2026-02-23.json) |
| 📄 2026-02-22 | 23 | [View JSON](data/daily/2026-02-22.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W08 | 156 | [View JSON](data/weekly/2026-W08.json) |
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
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
