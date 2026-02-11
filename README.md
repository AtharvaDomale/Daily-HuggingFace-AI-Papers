<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-58-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2028+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">58</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">107</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">509</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2028+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 11, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining</b> ⭐ 93</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07085) • [📄 arXiv](https://arxiv.org/abs/2602.07085) • [📥 PDF](https://arxiv.org/pdf/2602.07085)

**💻 Code:** [⭐ Code](https://github.com/QuantaAlpha/QuantaAlpha)

> QuantaAlpha tackles noisy, non-stationary markets by evolving alpha-mining trajectories via mutation and crossover, enabling controllable multi-round search and reliable reuse of successful patterns. It enforces hypothesis–factor–code semantic con...

</details>

<details>
<summary><b>2. Weak-Driven Learning: How Weak Agents make Strong Agents Stronger</b> ⭐ 39</summary>

<br/>

**👥 Authors:** Yifei Li, Tianxiang Ai, Gongxun Li, Yikunb, chhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08222) • [📄 arXiv](https://arxiv.org/abs/2602.08222) • [📥 PDF](https://arxiv.org/pdf/2602.08222)

**💻 Code:** [⭐ Code](https://github.com/chenzehao82/Weak-Driven-Learning)

> Weak-Driven Learning refers to a class of post-training paradigms in which the improvement of a strong model is driven by systematic discrepancies between its predictions and those of a weaker reference model (e.g., a historical checkpoint), rathe...

</details>

<details>
<summary><b>3. MOVA: Towards Scalable and Synchronized Video-Audio Generation</b> ⭐ 588</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08794) • [📄 arXiv](https://arxiv.org/abs/2602.08794) • [📥 PDF](https://arxiv.org/pdf/2602.08794)

**💻 Code:** [⭐ Code](https://github.com/OpenMOSS/MOVA)

> Blog： https://mosi.cn/models/mova Model： https://huggingface.co/collections/OpenMOSS-Team/mova Code： https://github.com/OpenMOSS/MOVA

</details>

<details>
<summary><b>4. Modality Gap-Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models</b> ⭐ 41</summary>

<br/>

**👥 Authors:** Hanzhen Zhao, Chonghan Liu, Wenjie Zhang, Yi Xin, Yu2020

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07026) • [📄 arXiv](https://arxiv.org/abs/2602.07026) • [📥 PDF](https://arxiv.org/pdf/2602.07026)

**💻 Code:** [⭐ Code](https://github.com/Yu-xm/ReVision.git)

> Modality Gap-Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models

</details>

<details>
<summary><b>5. Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07845) • [📄 arXiv](https://arxiv.org/abs/2602.07845) • [📥 PDF](https://arxiv.org/pdf/2602.07845)

**💻 Code:** [⭐ Code](https://github.com/rd-vla/rd-vla)

> Current Vision–Language–Action (VLA) models rely on fixed computational depth, expending the same amount of compute on simple adjustments and complex multi-step manipulation. While Chain-of-Thought (CoT) prompting enables variable computation, it ...

</details>

<details>
<summary><b>6. AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06855) • [📄 arXiv](https://arxiv.org/abs/2602.06855) • [📥 PDF](https://arxiv.org/pdf/2602.06855)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/airs-bench)

> We are introducing AIRS-bench, asking agents to beat human SOTA on 20 research tasks from recent ML papers (from NLP, math and coding to biochemical modeling and time series prediction). We provide no baseline code, and assess end-to-end research ...

</details>

<details>
<summary><b>7. LLaDA2.1: Speeding Up Text Diffusion via Token Editing</b> ⭐ 256</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08676) • [📄 arXiv](https://arxiv.org/abs/2602.08676) • [📥 PDF](https://arxiv.org/pdf/2602.08676)

**💻 Code:** [⭐ Code](https://github.com/inclusionAI/LLaDA2.X)

> LLaDA2.1-mini: https://huggingface.co/inclusionAI/LLaDA2.1-mini LLaDA2.1-flash: https://huggingface.co/inclusionAI/LLaDA2.1-flash

</details>

<details>
<summary><b>8. Alleviating Sparse Rewards by Modeling Step-Wise and Long-Term Sampling Effects in Flow-Based GRPO</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06422) • [📄 arXiv](https://arxiv.org/abs/2602.06422) • [📥 PDF](https://arxiv.org/pdf/2602.06422)

**💻 Code:** [⭐ Code](https://github.com/YunzeTong/TurningPoint-GRPO)

> Deploying GRPO on Flow Matching models has proven effective for text-to-image generation. However, existing paradigms typically propagate an outcome-based reward to all preceding denoising steps without distinguishing the local effect of each step...

</details>

<details>
<summary><b>9. GEBench: Benchmarking Image Generation Models as GUI Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09007) • [📄 arXiv](https://arxiv.org/abs/2602.09007) • [📥 PDF](https://arxiv.org/pdf/2602.09007)

**💻 Code:** [⭐ Code](https://github.com/stepfun-ai/GEBench)

> Recent advancements in image generation models have enabled the prediction of future Graphical User Interface (GUI) states based on user instructions. However, existing benchmarks primarily focus on general domain visual fidelity, leaving the eval...

</details>

<details>
<summary><b>10. Towards Agentic Intelligence for Materials Science</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu Song, Ziyu Hou, Wenhao Huang, Yizhan Li, Huan Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00169) • [📄 arXiv](https://arxiv.org/abs/2602.00169) • [📥 PDF](https://arxiv.org/pdf/2602.00169)

> AI4MatSci

</details>

<details>
<summary><b>11. Demo-ICL: In-Context Learning for Procedural Video Knowledge Acquisition</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08439) • [📄 arXiv](https://arxiv.org/abs/2602.08439) • [📥 PDF](https://arxiv.org/pdf/2602.08439)

**💻 Code:** [⭐ Code](https://github.com/dongyh20/Demo-ICL)

> Demo-ICL: In-Context Learning for Procedural Video Knowledge Acquisition

</details>

<details>
<summary><b>12. Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06025) • [📄 arXiv](https://arxiv.org/abs/2602.06025) • [📥 PDF](https://arxiv.org/pdf/2602.06025)

**💻 Code:** [⭐ Code](https://github.com/ViktorAxelsen/BudgetMem)

> Memory is increasingly central to Large Language Model (LLM) agents operating beyond a single context window, yet most existing systems rely on offline, query-agnostic memory construction that can be inefficient and may discard query-critical info...

</details>

<details>
<summary><b>13. LOCA-bench: Benchmarking Language Agents Under Controllable and Extreme Context Growth</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07962) • [📄 arXiv](https://arxiv.org/abs/2602.07962) • [📥 PDF](https://arxiv.org/pdf/2602.07962)

**💻 Code:** [⭐ Code](https://github.com/hkust-nlp/LOCA-bench)

> Long-running agents quietly fail as context grows. Even with 100K–1M token windows, reliability degrades — plans drift, constraints are forgotten, exploration collapses. We introduce LOCA-bench, a benchmark designed specifically for long-context, ...

</details>

<details>
<summary><b>14. InternAgent-1.5: A Unified Agentic Framework for Long-Horizon Autonomous Scientific Discovery</b> ⭐ 864</summary>

<br/>

**👥 Authors:** Xiangchao Yan, Runmin Ma, JiakangYuan, huangst, sY713

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08990) • [📄 arXiv](https://arxiv.org/abs/2602.08990) • [📥 PDF](https://arxiv.org/pdf/2602.08990)

**💻 Code:** [⭐ Code](https://github.com/InternScience/InternAgent)

> Proposes InternAgent-1.5, a unified, three-subsystem agent for end-to-end long-horizon scientific discovery with memory, verification, and evolution across computation and experiments.

</details>

<details>
<summary><b>15. RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI</b> ⭐ 2.44k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07837) • [📄 arXiv](https://arxiv.org/abs/2602.07837) • [📥 PDF](https://arxiv.org/pdf/2602.07837)

**💻 Code:** [⭐ Code](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/run_realworld_async.sh)

> We present USER, a Unified and extensible SystEm for Real-world online policy learning. USER treats physical robots as first-class hardware resources alongside GPUs through a unified hardware abstraction layer, enabling automatic discovery, manage...

</details>

<details>
<summary><b>16. GISA: A Benchmark for General Information-Seeking Assistant</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08543) • [📄 arXiv](https://arxiv.org/abs/2602.08543) • [📥 PDF](https://arxiv.org/pdf/2602.08543)

> A New Benchmark for General Information Seeking Assistant

</details>

<details>
<summary><b>17. WorldCompass: Reinforcement Learning for Long-Horizon World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09022) • [📄 arXiv](https://arxiv.org/abs/2602.09022) • [📥 PDF](https://arxiv.org/pdf/2602.09022)

> WorldCompass: Reinforcement Learning for Long-Horizon World Models

</details>

<details>
<summary><b>18. Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Letian Xue, Jieyu Zhang, Yue Wang, Zihan Huang, williamzhangNU

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07055) • [📄 arXiv](https://arxiv.org/abs/2602.07055) • [📥 PDF](https://arxiv.org/pdf/2602.07055)

**💻 Code:** [⭐ Code](https://github.com/mll-lab-nu/Theory-of-Space)

> Theory of Space studies whether foundation models can construct a globally consistent spatial belief from partial observations via active exploration, revise the belief in dynamic environments when new evidence contradicts prior assumptions, and e...

</details>

<details>
<summary><b>19. LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Jia Zhang, Yicheng Mao, JeremyYin, yoyoliuuu, XinwuYe

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07075) • [📄 arXiv](https://arxiv.org/abs/2602.07075) • [📥 PDF](https://arxiv.org/pdf/2602.07075)

**💻 Code:** [⭐ Code](https://github.com/xinwuye/LatentChem)

> great paper

</details>

<details>
<summary><b>20. AgentCPM-Report: Interleaving Drafting and Deepening for Open-Ended Deep Research</b> ⭐ 728</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06540) • [📄 arXiv](https://arxiv.org/abs/2602.06540) • [📥 PDF](https://arxiv.org/pdf/2602.06540)

**💻 Code:** [⭐ Code](https://github.com/OpenBMB/AgentCPM) • [⭐ Code](https://github.com/OpenBMB/MiniCPM) • [⭐ Code](https://github.com/OpenBMB/AgentCPM/tree/main/AgentCPM-Report)

> AgentCPM-Report是由 THUNLP 、中国人民大学 RUCBM 和 ModelBest 联合开发的开源大语言模型智能体。它基于 MiniCPM4.1 80亿参数基座模型，接受用户指令作为输入，自主生成长篇报告。其有以下亮点： 极致效能，以小博大：通过平均40轮的深度检索与近100轮的思维链推演，实现对信息的全方位挖掘与重组，让端侧模型也能产出逻辑严密、洞察深刻的万字长文，在深度调研任务上以8B参数规模达成与顶级闭源系统的性能对标。 物理隔绝，本地安全：专为高隐私场景设计，支持...

</details>

<details>
<summary><b>21. Context Compression via Explicit Information Transmission</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03784) • [📄 arXiv](https://arxiv.org/abs/2602.03784) • [📥 PDF](https://arxiv.org/pdf/2602.03784)

> A new paradigm for LLM context compression, which is very effective! We hope this work will inspire further exploration of this paradigm for context compression. Code will be open-source soon.

</details>

<details>
<summary><b>22. Fundamental Reasoning Paradigms Induce Out-of-Domain Generalization in Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Maria Liakata, Marco Valentino, Mahmud Akhter, Xingwei Tan, Mingzi Cao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08658) • [📄 arXiv](https://arxiv.org/abs/2602.08658) • [📥 PDF](https://arxiv.org/pdf/2602.08658)

**💻 Code:** [⭐ Code](https://github.com/voalmciaf/FR-OOD)

> Goal: We investigated how the three core reasoning types—deduction, induction, and abduction—help Large Language Models (LLMs) generalize their thinking skills. Data: We collected a new dataset of reasoning trajectories from symbolic tasks to focu...

</details>

<details>
<summary><b>23. TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Terminal Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07274) • [📄 arXiv](https://arxiv.org/abs/2602.07274) • [📥 PDF](https://arxiv.org/pdf/2602.07274)

**💻 Code:** [⭐ Code](https://github.com/ucsb-mlsec/terminal-bench-env)

> This paper introduces TermiGen, an end-to-end pipeline designed to enhance the performance of open-weight Large Language Models (LLMs) in executing complex terminal tasks. To address the scarcity of high-fidelity training data and the distribution...

</details>

<details>
<summary><b>24. RelayGen: Intra-Generation Model Switching for Efficient Reasoning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06454) • [📄 arXiv](https://arxiv.org/abs/2602.06454) • [📥 PDF](https://arxiv.org/pdf/2602.06454)

**💻 Code:** [⭐ Code](https://github.com/jiwonsong-dev/RelayGen)

> RelayGen is a training-free, segment-level runtime model switching framework that exploits intra-generation difficulty variation to reduce inference latency while preserving most of the accuracy of large reasoning models.

</details>

<details>
<summary><b>25. How2Everything: Mining the Web for How-To Procedures to Evaluate and Improve LLMs</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08808) • [📄 arXiv](https://arxiv.org/abs/2602.08808) • [📥 PDF](https://arxiv.org/pdf/2602.08808)

**💻 Code:** [⭐ Code](https://github.com/lilakk/how2everything)

> How2Everything builds scalable evaluation and improvement loops for LLMs using mined procedures, scoring with an LLM judge, distilling a frontier model, and RL rewards.

</details>

<details>
<summary><b>26. When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08236) • [📄 arXiv](https://arxiv.org/abs/2602.08236) • [📥 PDF](https://arxiv.org/pdf/2602.08236)

**💻 Code:** [⭐ Code](https://github.com/Yui010206/Adaptive-Visual-Imagination-Control/)

> website: https://adaptive-visual-tts.github.io/

</details>

<details>
<summary><b>27. Rolling Sink: Bridging Limited-Horizon Training and Open-Ended Testing in Autoregressive Video Diffusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07775) • [📄 arXiv](https://arxiv.org/abs/2602.07775) • [📥 PDF](https://arxiv.org/pdf/2602.07775)

> Thanks for sharing, @ taesiri !

</details>

<details>
<summary><b>28. NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06694) • [📄 arXiv](https://arxiv.org/abs/2602.06694) • [📥 PDF](https://arxiv.org/pdf/2602.06694)

> Blog-style summary: https://www.alphaxiv.org/overview/2602.06694v1

</details>

<details>
<summary><b>29. Reliable and Responsible Foundation Models: A Comprehensive Survey</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08145) • [📄 arXiv](https://arxiv.org/abs/2602.08145) • [📥 PDF](https://arxiv.org/pdf/2602.08145)

> The survey addresses the reliable and responsible development of foundation models.

</details>

<details>
<summary><b>30. Data Science and Technology Towards AGI Part I: Tiered Data Management</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09003) • [📄 arXiv](https://arxiv.org/abs/2602.09003) • [📥 PDF](https://arxiv.org/pdf/2602.09003)

**💻 Code:** [⭐ Code](https://github.com/UltraData-OpenBMB/UltraData-Math)

> AI evolution is shifting from "Data-Driven Learning" to "Data-Model Co-Evolution"—a cycle where models and data enhance each other. 🔄 Today, we launch #UltraData: An all-in-one Data Science platform featuring a systematic L0–L4 Tiered Data Managem...

</details>

<details>
<summary><b>31. Thinking Makes LLM Agents Introverted: How Mandatory Thinking Can Backfire in User-Engaged Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07796) • [📄 arXiv](https://arxiv.org/abs/2602.07796) • [📥 PDF](https://arxiv.org/pdf/2602.07796)

> A comprehensive analysis of the effect of thinking in user-engaged agentic LLM inference scenarios.

</details>

<details>
<summary><b>32. Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control</b> ⭐ 52</summary>

<br/>

**👥 Authors:** Yao Su, Biao Hou, Hangxin Liu, Zhehan Li, Weidong-Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21363) • [📄 arXiv](https://arxiv.org/abs/2601.21363) • [📥 PDF](https://arxiv.org/pdf/2601.21363)

**💻 Code:** [⭐ Code](https://github.com/bigai-ai/LIFT-humanoid)

> Real-world Reinforcement Learning on Humanoid Robot Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control 🔗 Project: https://lift-humanoid.github.io/ 💻 Code: https://github.com/bigai-ai/LIFT-humanoid

</details>

<details>
<summary><b>33. MotionCrafter: Dense Geometry and Motion Reconstruction with a 4D VAE</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08961) • [📄 arXiv](http://arxiv.org/abs/2602.08961) • [📥 PDF](https://arxiv.org/pdf/2602.08961)

**💻 Code:** [⭐ Code](https://github.com/TencentARC/MotionCrafter)

> 🚀 Excited to share our latest work MotionCrafter! 🌟 The first Video Diffusion-based framework for joint geometry and motion estimation. 📄 Paper: http://arxiv.org/abs/2602.08961 🌐 Project page: https://ruijiezhu94.github.io/MotionCrafter_Page 💻 Cod...

</details>

<details>
<summary><b>34. WildReward: Learning Reward Models from In-the-Wild Human Interactions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08829) • [📄 arXiv](https://arxiv.org/abs/2602.08829) • [📥 PDF](https://arxiv.org/pdf/2602.08829)

> This paper explores training reward models from in-the-wild human interactions.

</details>

<details>
<summary><b>35. Improving Data and Reward Design for Scientific Reasoning in Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08321) • [📄 arXiv](https://arxiv.org/abs/2602.08321) • [📥 PDF](https://arxiv.org/pdf/2602.08321)

> Solving open-ended science questions remains challenging for large language models, particularly due to inherently unreliable supervision and evaluation. The bottleneck lies in the data construction and reward design for scientific post-training. ...

</details>

<details>
<summary><b>36. ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiayang Wu, Shibowen Zhang, Jiongye Li, Jingwen Zhang, Weidong-Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06445) • [📄 arXiv](https://arxiv.org/abs/2602.06445) • [📥 PDF](https://arxiv.org/pdf/2602.06445)

**💻 Code:** [⭐ Code](https://github.com/bigai-ai/ECO-humanoid)

> ECO Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking

</details>

<details>
<summary><b>37. SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07803) • [📄 arXiv](https://arxiv.org/abs/2602.07803) • [📥 PDF](https://arxiv.org/pdf/2602.07803)

> While recent years have witnessed rapid progress in speech synthesis, open-source singing voice synthesis (SVS) systems still face significant barriers to industrial deployment, particularly in terms of robustness and zero-shot generalization. In ...

</details>

<details>
<summary><b>38. On Randomness in Agentic Evals</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07150) • [📄 arXiv](https://arxiv.org/abs/2602.07150) • [📥 PDF](https://arxiv.org/pdf/2602.07150)

> We just published a paper quantifying a problem the AI community has been quietly ignoring: single-run benchmark evaluations are far noisier than most people realize. And the decisions they inform — which model to deploy, which research direction ...

</details>

<details>
<summary><b>39. Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data, Vocabulary, Morphology Interplay</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06942) • [📄 arXiv](https://arxiv.org/abs/2602.06942) • [📥 PDF](https://arxiv.org/pdf/2602.06942)

> Tokenization is a pivotal design choice for morphologically rich languages like Turkish, where productive agglutination strains both vocabulary efficiency and morphological fidelity. Despite growing interest, prior work often varies vocabulary siz...

</details>

<details>
<summary><b>40. Echoes as Anchors: Probabilistic Costs and Attention Refocusing in LLM Reasoning</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Min Zhang, Fangming Liu, Wu Li, Zhuo Li, larry2210

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06600) • [📄 arXiv](https://arxiv.org/abs/2602.06600) • [📥 PDF](https://arxiv.org/pdf/2602.06600)

**💻 Code:** [⭐ Code](https://github.com/hhh2210/echoes-as-anchors)

> Tracing the spontaneous “Echo” phenomenon—where models repeat the user query—we link its emergence to the evolution of CoT and RLVR. Through probabilistic and Attention analyses, we show that Echo functions as an effective attention anchor, and de...

</details>

<details>
<summary><b>41. Flexible Entropy Control in RLVR with Gradient-Preserving Perspective</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhixiong Zeng, Haibo Qiu, Fanfan Liu, Peng Shi, Kun Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09782) • [📄 arXiv](https://arxiv.org/abs/2602.09782) • [📥 PDF](https://arxiv.org/pdf/2602.09782)

**💻 Code:** [⭐ Code](https://github.com/Kwen-Chen/Flexible-Entropy-Control)

> This paper proposes reshaping entropy control in RL from the perspective of Gradient-Preserving Clipping.

</details>

<details>
<summary><b>42. FlexMoRE: A Flexible Mixture of Rank-heterogeneous Experts for Efficient Federatedly-trained Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08818) • [📄 arXiv](https://arxiv.org/abs/2602.08818) • [📥 PDF](https://arxiv.org/pdf/2602.08818)

> FlexMoRE: A Flexible Mixture of Rank-heterogeneous Experts for Efficient Federatedly-trained Large Language Models

</details>

<details>
<summary><b>43. Learning-guided Kansa collocation for forward and inverse PDEs beyond linearity</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fangcheng Zhong, Chenliang Zhou, Cengiz Öztireli, Weitao Chen, Peter2023HuggingFace

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07970) • [📄 arXiv](https://arxiv.org/abs/2602.07970) • [📥 PDF](https://arxiv.org/pdf/2602.07970)

> Kansa solver extension beyond linearity.

</details>

<details>
<summary><b>44. GraphAgents: Knowledge Graph-Guided Agentic AI for Cross-Domain Materials Design</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07491) • [📄 arXiv](https://arxiv.org/abs/2602.07491) • [📥 PDF](https://arxiv.org/pdf/2602.07491)

> GraphAgents: Knowledge Graph-Guided Agentic AI for Cross-Domain Materials Design Large Language Models (LLMs) promise to accelerate discovery by reasoning across the expanding scientific landscape. Yet, the challenge is no longer access to informa...

</details>

<details>
<summary><b>45. Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07120) • [📄 arXiv](https://arxiv.org/abs/2602.07120) • [📥 PDF](https://arxiv.org/pdf/2602.07120)

**💻 Code:** [⭐ Code](https://github.com/jacqueline-he/anchored-decoding)

> The memorization and reproduction of copyrighted text in LLMs is an issue that has potentially harmful repercussions for both data creators and AI developers. To this end, Anchored Decoding is a decoding technique for language models (LMs) that pr...

</details>

<details>
<summary><b>46. Concept-Aware Privacy Mechanisms for Defending Embedding Inversion Attacks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shou-De Lin, Kuan-Yu Chen, Hsiang Hsiao, Yu-Che Tsai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07090) • [📄 arXiv](https://arxiv.org/abs/2602.07090) • [📥 PDF](https://arxiv.org/pdf/2602.07090)

> This work proposes a concept-aware privacy mechanism (SPARSE) to defend against embedding inversion attacks, selectively perturbing concept-sensitive dimensions while preserving downstream utility. Relevant to: embedding privacy, inversion attacks...

</details>

<details>
<summary><b>47. CodeCircuit: Toward Inferring LLM-Generated Code Correctness via Attribution Graphs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07080) • [📄 arXiv](https://arxiv.org/abs/2602.07080) • [📥 PDF](https://arxiv.org/pdf/2602.07080)

**💻 Code:** [⭐ Code](https://github.com/bruno686/CodeCircuit)

> Our code is available at: https://github.com/bruno686/CodeCircuit

</details>

<details>
<summary><b>48. KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05929) • [📄 arXiv](https://arxiv.org/abs/2602.05929) • [📥 PDF](https://arxiv.org/pdf/2602.05929)

> KV-CoRE introduces a clean, data-dependent framework for measuring (not just applying) KV-cache compression in LLMs. By performing incremental SVD directly on cached key/value activations, the paper provides a principled, layer-wise view of low-ra...

</details>

<details>
<summary><b>49. Cost-Efficient RAG for Entity Matching with LLMs: A Blocking-based Exploration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Paul Groth, Sebastian Schelter, Arijit Khan, Zeyu Zhang, Chuangtao Ma

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05708) • [📄 arXiv](https://arxiv.org/abs/2602.05708) • [📥 PDF](https://arxiv.org/pdf/2602.05708)

> Can blocking help in LLM and RAG-based entity matching? Check out CE-RAG4EM, a Cost-Efficient RAG for Entity Matching that aims to reduce the cost of RAG4EM via blocking-based batch retrieval and inference.

</details>

<details>
<summary><b>50. AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07054) • [📄 arXiv](https://arxiv.org/abs/2602.07054) • [📥 PDF](https://arxiv.org/pdf/2602.07054)

> Check out this latest release from our lab at the Institute for Creative Technologies at the University of Southern California. The proposed method improves emotion reasoning in audiovisual multimodal ("omni") LLMs, surpassing the state-of-the-art...

</details>

<details>
<summary><b>51. Aster: Autonomous Scientific Discovery over 20x Faster Than Existing Methods</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Emmett Bicker

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07040) • [📄 arXiv](https://arxiv.org/abs/2602.07040) • [📥 PDF](https://arxiv.org/pdf/2602.07040)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Learning to Discover at Test Time (2026) AIRS-Bench: a Suite of Tasks for F...

</details>

<details>
<summary><b>52. Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02827) • [📄 arXiv](https://arxiv.org/abs/2602.02827) • [📥 PDF](https://arxiv.org/pdf/2602.02827)

> No abstract available.

</details>

<details>
<summary><b>53. CauScale: Neural Causal Discovery at Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08629) • [📄 arXiv](https://arxiv.org/abs/2602.08629) • [📥 PDF](https://arxiv.org/pdf/2602.08629)

> We introduce CauScale, a neural architecture designed for efficient causal discovery that scales inference to graphs with up to 1000 nodes.

</details>

<details>
<summary><b>54. Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Richard Huang, Shanshan Zhong, George Ling

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08004) • [📄 arXiv](https://arxiv.org/abs/2602.08004) • [📥 PDF](https://arxiv.org/pdf/2602.08004)

> Understand Agent Skills at a Glance: The Ecosystem, Opportunities, and Risks Behind 40,000+ Claude Skills From patterns of explosive growth and a comprehensive, multi-dimensional functional taxonomy to multi-tier security audits, this data-driven ...

</details>

<details>
<summary><b>55. dewi-kadita: A Python Library for Idealized Fish Schooling Simulation with Entropy-Based Diagnostics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07948) • [📄 arXiv](https://arxiv.org/abs/2602.07948) • [📥 PDF](https://arxiv.org/pdf/2602.07948)

**💻 Code:** [⭐ Code](https://github.com/sandyherho/dewi-kadita)

> Simulating collective animal behavior requires robust tools to capture emergent complexity. This paper introduces dewi-kadita, an open-source Python library that implements the three-dimensional Couzin model for fish schooling. Unlike traditional ...

</details>

<details>
<summary><b>56. Reasoning-Augmented Representations for Multimodal Retrieval</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sukanta Ganguly, Soochahn Lee, Brandon Han, Anirudh Sundara Rajan, Jianrui Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07125) • [📄 arXiv](https://arxiv.org/abs/2602.07125) • [📥 PDF](https://arxiv.org/pdf/2602.07125)

**💻 Code:** [⭐ Code](https://github.com/AugmentedRetrieval/ReasoningAugmentedRetrieval)

> Universal Multimodal Retrieval (UMR) seeks any-to-any search across text and vision, yet modern embedding models remain brittle when queries require latent reasoning (e.g., resolving underspecified references or matching compositional constraints)...

</details>

<details>
<summary><b>57. f-GRPO and Beyond: Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qifan Song, Yue Xing, Guang Lin, Lantao Mei, Rajdeep Haldar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05946) • [📄 arXiv](https://arxiv.org/abs/2602.05946) • [📥 PDF](https://arxiv.org/pdf/2602.05946)

> Recent research shows that Preference Alignment (PA) objectives act as divergence estimators be- tween aligned (chosen) and unaligned (rejected) response distributions. In this work, we extend this divergence-based perspective to general align- me...

</details>

<details>
<summary><b>58. Statistical Learning Theory in Lean 4: Empirical Processes from Scratch</b> ⭐ 29</summary>

<br/>

**👥 Authors:** Fanghui Liu, Jason D. Lee, liminho123

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02285) • [📄 arXiv](https://arxiv.org/abs/2602.02285) • [📥 PDF](https://arxiv.org/pdf/2602.02285)

**💻 Code:** [⭐ Code](https://github.com/YuanheZ/lean-stat-learning-theory)

> We present the first comprehensive Lean 4 formalization of statistical learning theory (SLT) grounded in empirical process theory. Our end-to-end formal infrastructure implement the missing contents in latest Lean library, including a complete dev...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 58 |
| 📅 Today | [`2026-02-11.json`](data/daily/2026-02-11.json) | 58 |
| 📆 This Week | [`2026-W06.json`](data/weekly/2026-W06.json) | 107 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 509 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-11 | 58 | [View JSON](data/daily/2026-02-11.json) |
| 📄 2026-02-10 | 2 | [View JSON](data/daily/2026-02-10.json) |
| 📄 2026-02-09 | 47 | [View JSON](data/daily/2026-02-09.json) |
| 📄 2026-02-08 | 47 | [View JSON](data/daily/2026-02-08.json) |
| 📄 2026-02-07 | 47 | [View JSON](data/daily/2026-02-07.json) |
| 📄 2026-02-06 | 52 | [View JSON](data/daily/2026-02-06.json) |
| 📄 2026-02-05 | 53 | [View JSON](data/daily/2026-02-05.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W06 | 107 | [View JSON](data/weekly/2026-W06.json) |
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 509 | [View JSON](data/monthly/2026-02.json) |
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
