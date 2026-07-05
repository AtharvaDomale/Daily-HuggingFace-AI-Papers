<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-27-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5102+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">27</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">117</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">87</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5102+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** July 05, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Program-as-Weights: A Programming Paradigm for Fuzzy Functions</b> ⭐ 113</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02512) • [📄 arXiv](https://arxiv.org/abs/2607.02512) • [📥 PDF](https://arxiv.org/pdf/2607.02512)

**💻 Code:** [⭐ Code](https://github.com/programasweights/programasweights-python) • [⭐ Code](https://github.com/huggingface)

> Define functions in English → ProgramAsWeights (PAW) compiles them into neural programs → call them like ordinary Python functions. PAW enables: 🎮 Alien Taboo : give an alien free-form clues and see whether it can guess your secret word 🤖 Avatar D...

</details>

<details>
<summary><b>2. AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents</b> ⭐ 51</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02255) • [📄 arXiv](https://arxiv.org/abs/2607.02255) • [📥 PDF](https://arxiv.org/pdf/2607.02255)

**💻 Code:** [⭐ Code](https://github.com/AlayaLab/AgenticSTS) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>3. EvoPolicyGym: Evaluating Autonomous Policy Evolution in Interactive Environments</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiacheng Chen, Jusen Du, Runzhe Zhan, Han Song, Zhilin Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02440) • [📄 arXiv](https://arxiv.org/abs/2607.02440) • [📥 PDF](https://arxiv.org/pdf/2607.02440)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Autonomous agents are increasingly expected to improve executable policies through feedback, yet existing evaluations often collapse this process into a final score or confound it with open-ended software-engineering progress. We introduce Autonom...

</details>

<details>
<summary><b>4. Morphing into Hybrid Attention Models</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.30562) • [📄 arXiv](https://arxiv.org/abs/2606.30562) • [📥 PDF](https://arxiv.org/pdf/2606.30562)

**💻 Code:** [⭐ Code](https://github.com/LanDisen/FlashMorph) • [⭐ Code](https://github.com/huggingface)

> Morphing into Hybrid Attention Models

</details>

<details>
<summary><b>5. Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling</b> ⭐ 87</summary>

<br/>

**👥 Authors:** Junqing Lin, Weilun Feng, Yifu Ding, Xianglong Liu, Xingyu Zheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01642) • [📄 arXiv](https://arxiv.org/abs/2607.01642) • [📥 PDF](https://arxiv.org/pdf/2607.01642)

**💻 Code:** [⭐ Code](https://github.com/Xingyu-Zheng/MrFlow) • [⭐ Code](https://github.com/huggingface)

> MrFlow proposes a training-free multi-resolution strategy for accelerating image generation, following a clear coarse-to-fine pipeline: multi-step low-resolution structure sampling, pixel-space super-resolution, and one-step high-resolution detail...

</details>

<details>
<summary><b>6. AgenticDataBench: A Comprehensive Benchmark for Data Agents</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01647) • [📄 arXiv](https://arxiv.org/abs/2607.01647) • [📥 PDF](https://arxiv.org/pdf/2607.01647)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AgenticDataBench/AgenticDataBench)

> AgenticDataBench is a comprehensive benchmark for evaluating LLM-based data agents that automate real-world data science workflows. It addresses the lack of rigorous evaluation by providing diverse and realistic tasks with fine-grained ground-trut...

</details>

<details>
<summary><b>7. WorldDirector: Building Controllable World Simulators with Persistent Dynamic Memory</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qingyan Bai, Wen Wang, Qiuyu Wang, Hao Ouyang, Hanlin Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02517) • [📄 arXiv](https://arxiv.org/abs/2607.02517) • [📥 PDF](https://arxiv.org/pdf/2607.02517)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/pPetrichor/WorldDirector)

> Project Page: https://worlddirector.github.io/ Github: https://github.com/pPetrichor/WorldDirector

</details>

<details>
<summary><b>8. Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Jaehoon Yun, Sungwook Jung, Suhyeon Lim, Minbyul Jeong, Junha Jung

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.31825) • [📄 arXiv](https://arxiv.org/abs/2606.31825) • [📥 PDF](https://arxiv.org/pdf/2606.31825)

**💻 Code:** [⭐ Code](https://github.com/dmis-lab/MRPO) • [⭐ Code](https://github.com/huggingface)

> Recent multimodal large language models have shown great promise in clinical image reasoning, but existing post-training pipelines remain predominantly outcome-centric, relying on final answer correctness or sequence-level preferences. This suffer...

</details>

<details>
<summary><b>9. SkillCoach: Self-Evolving Rubrics for Evaluating and Enhancing Agentic Skill-Use</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sulong Xu, Dengbo He, Yudong Guo, Kelong Mao, Jiayin Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01874) • [📄 arXiv](https://arxiv.org/abs/2607.01874) • [📥 PDF](https://arxiv.org/pdf/2607.01874)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API SkillRevise: Improving LLM-Authored Agent Skills via Trace-Conditioned Skil...

</details>

<details>
<summary><b>10. Optimizing Visual Generative Models via Distribution-wise Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02291) • [📄 arXiv](https://arxiv.org/abs/2607.02291) • [📥 PDF](https://arxiv.org/pdf/2607.02291)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Proposes a distribution-wise RL framework for visual generation to mitigate reward hacking and mode collapse. By employing an efficient subset-replace strategy, this approach significantly improves the visual quality and diversity on the SiT model.

</details>

<details>
<summary><b>11. Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Pasquale Minervini, Beatrice Alex, Aryo Pradipta Gema

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01002) • [📄 arXiv](https://arxiv.org/abs/2607.01002) • [📥 PDF](https://arxiv.org/pdf/2607.01002)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/aryopg/locos)

> A fresh perspective on retrieval heads!

</details>

<details>
<summary><b>12. AGVBench: A Reliability-Oriented Benchmark of Data Augmentation for Vein Recognition</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02271) • [📄 arXiv](https://arxiv.org/abs/2607.02271) • [📥 PDF](https://arxiv.org/pdf/2607.02271)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Advance-VeinTech-Innovators/AGVBench)

> TL;DR: We open-source AGVBench, the first reliability-oriented benchmark for data augmentation in vein recognition. By systematically evaluating 30 representative augmentation strategies across 5 public palm- and finger-vein datasets using 7 backb...

</details>

<details>
<summary><b>13. AutoMem: Automated Learning of Memory as a Cognitive Skill</b> ⭐ 47</summary>

<br/>

**👥 Authors:** Serena Yeung-Levy, Xiaohan Wang, Yuhui Zhang, Hao Zhu, Shengguang Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01224) • [📄 arXiv](https://arxiv.org/abs/2607.01224) • [📥 PDF](https://arxiv.org/pdf/2607.01224)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/autoLearnMem/AutoMem)

> On long-horizon tasks, an LLM agent has to manage a lot of memory — what to write down, what to look up. We turn managing memory into a skill the model learns, and improve it automatically. An open 32B model approaches frontier-level performance o...

</details>

<details>
<summary><b>14. From SRA to Self-Flow: Data Augmentation or Self-Supervision?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jingdong Wang, Harry Yang, Mengmeng Wang, Dengyang Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02508) • [📄 arXiv](https://arxiv.org/abs/2607.02508) • [📥 PDF](https://arxiv.org/pdf/2607.02508)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The underlying mechanism behind the evolution from SRA to Self-Flow

</details>

<details>
<summary><b>15. InstanceControl: Controllable Complex Image Generation without Instance Labeling</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Jiaqi Xu, Zhixin Wang, Fan Li, Huan Wang, xiaoyu1104

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.31924) • [📄 arXiv](https://arxiv.org/abs/2606.31924) • [📥 PDF](https://arxiv.org/pdf/2606.31924)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/liuxiaoyu1104/InstanceControl)

> Accepted by ECCV 2026, project page: https://instancecontrol.github.io/InstanceControl/

</details>

<details>
<summary><b>16. When Search Agents Should Ask: DiscoBench for Clarification-Aware Deep Search</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27669) • [📄 arXiv](https://arxiv.org/abs/2606.27669) • [📥 PDF](https://arxiv.org/pdf/2606.27669)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce DiscoBench: a benchmark for evaluating when an LLM search agent should stop searching and ask for clarification. Many search-agent benchmarks assume that user queries are complete and well-specified. But in real-world deep search, req...

</details>

<details>
<summary><b>17. AnyGroundBench: A Specialized-Domain Benchmark for Video Grounding in Vision-Language Models</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Kanta Sawafuji, Taiki Kanaya, Reina Ishikawa, Ryo Fujii, Rintaro Otsubo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02269) • [📄 arXiv](https://arxiv.org/abs/2607.02269) • [📥 PDF](https://arxiv.org/pdf/2607.02269)

**💻 Code:** [⭐ Code](https://github.com/rinost081/AnyGroundBench) • [⭐ Code](https://github.com/huggingface)

> Vision-Language Models (VLMs) have demonstrated immense promise in Spatio-Temporal Video Grounding (STVG). However, current evaluation protocols are largely confined to zero-shot assessments on general, daily-life benchmarks. This creates a critic...

</details>

<details>
<summary><b>18. PACE: A Proxy for Agentic Capability Evaluation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiayi Geng, Lindia Tjuatja, Jiarui Liu, Lintang Sutawika, Yueqi Song

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02032) • [📄 arXiv](https://arxiv.org/abs/2607.02032) • [📥 PDF](https://arxiv.org/pdf/2607.02032)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation (2...

</details>

<details>
<summary><b>19. WARP: Weight-Space Analysis for Recovering Training Data Portfolios</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01686) • [📄 arXiv](https://arxiv.org/abs/2607.01686) • [📥 PDF](https://arxiv.org/pdf/2607.01686)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SprocketLab/WARP)

> Weight-space geometry encodes traces of training data. Can we use it to reverse-engineer data recipes? Introducing WARP: a new strategy to estimate domain mixtures from model weights alone!

</details>

<details>
<summary><b>20. Learning to Move Before Learning to Do: Task-Agnostic pretraining for VLAs</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Jingjing Gong, Li Ji, Xiaopeng Yu, Siyin Wang, Junhao Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02466) • [📄 arXiv](https://arxiv.org/abs/2607.02466) • [📥 PDF](https://arxiv.org/pdf/2607.02466)

**💻 Code:** [⭐ Code](https://github.com/sjh0354/Task-Agnostic-Pretrain) • [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API APT: Action Expert Pretraining Improves Instruction Generalization of Visio...

</details>

<details>
<summary><b>21. Denser neq Better: Limits of On-Policy Self-Distillation for Continual Post-Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Geng Liu, Lu Yang, Wenzhuo Liu, Haohan Zhao, Meng Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01763) • [📄 arXiv](https://arxiv.org/abs/2607.01763) • [📥 PDF](https://arxiv.org/pdf/2607.01763)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Moenupa/SDPO-CL)

> Denser Supervision ≠ Better Performance, as we found SDPO suffers forgetting much more than GRPO.

</details>

<details>
<summary><b>22. DuoMem: Towards Capable On-Device Memory Agents via Dual-Space Distillation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ignacio Castro, Andrea Maracani, Ahmed Alajrami, Ondrej Bohdal, Peyman Hosseini

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.29961) • [📄 arXiv](https://arxiv.org/abs/2606.29961) • [📥 PDF](https://arxiv.org/pdf/2606.29961)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We are excited to release DuoMem, a dual-space distillation framework for on-device AI agents! DuoMem combines context-space and parameter-space distillation to transfer knowledge from a large teacher model into a small on-device model. It boosts ...

</details>

<details>
<summary><b>23. Representation Distribution Matching for One-Step Visual Generation</b> ⭐ 34</summary>

<br/>

**👥 Authors:** Alexandre Alahi, Matthieu Cord, Eloi Zablocki, Wuyang Li, Lan Feng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.02375) • [📄 arXiv](https://arxiv.org/abs/2607.02375) • [📥 PDF](https://arxiv.org/pdf/2607.02375)

**💻 Code:** [⭐ Code](https://github.com/vita-epfl/RDM) • [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Drifting Preference Optimization for One-Step Generative Models (2026) Teac...

</details>

<details>
<summary><b>24. Discrete Diffusion Language Models for Interactive Radiology Report Drafting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01436) • [📄 arXiv](https://arxiv.org/abs/2607.01436) • [📥 PDF](https://arxiv.org/pdf/2607.01436)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/mxvp/discrete_diffusion_RRG)

> Discrete diffusion LMs can draft radiology reports interactively - and match autoregression while doing it. We finetune an MoE diffusion VLM (DiffusionGemma-26B, 3.8B active) head-to-head against its autoregressive sibling (Gemma-4-26B) under an i...

</details>

<details>
<summary><b>25. Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.25178) • [📄 arXiv](https://arxiv.org/abs/2606.25178) • [📥 PDF](https://arxiv.org/pdf/2606.25178)

**💻 Code:** [⭐ Code](https://github.com/YangYongJin/transfer-aware-curriculum) • [⭐ Code](https://github.com/huggingface)

> TAC studies general reasoning through the lens of transferability: instead of asking whether post-training improves performance on its source domain, we ask how well the learned behavior transfers across held-out domains. Across 14 benchmarks in 6...

</details>

<details>
<summary><b>26. Parameter-Efficient Quantum-Inspired Fast Weight Programmers for Traffic-Matrix Forecasting</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nan-Yow Chen, Tai-Yue Li, Chun-Hua Lin, Jiun-Cheng Jiang, Kuo-Chung Peng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27821) • [📄 arXiv](https://arxiv.org/abs/2606.27821) • [📥 PDF](https://arxiv.org/pdf/2606.27821)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper explores whether compact quantum-inspired recurrent models can efficiently forecast full network traffic matrices under realistic resource constraints. The authors adapt gated quantum-inspired Kolmogorov–Arnold Network Fast-Weight Progr...

</details>

<details>
<summary><b>27. Scaling Laws for Grid-Based Approximate Nearest Neighbor Search in High Dimensions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.01283) • [📄 arXiv](https://arxiv.org/abs/2607.01283) • [📥 PDF](https://arxiv.org/pdf/2607.01283)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/weiz345/MultiProbeANN)

> Grid-based approaches to approximate nearest neighbor (ANN) search have been absent from modern scaling analyses. We present a systematic characterization of a multiprobe grid algorithm with respect to dataset size $N$ and dimensionality $d$. Our ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 27 |
| 📅 Today | [`2026-07-05.json`](data/daily/2026-07-05.json) | 27 |
| 📆 This Week | [`2026-W26.json`](data/weekly/2026-W26.json) | 117 |
| 🗓️ This Month | [`2026-07.json`](data/monthly/2026-07.json) | 87 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-07-05 | 27 | [View JSON](data/daily/2026-07-05.json) |
| 📄 2026-07-04 | 27 | [View JSON](data/daily/2026-07-04.json) |
| 📄 2026-07-03 | 12 | [View JSON](data/daily/2026-07-03.json) |
| 📄 2026-07-02 | 10 | [View JSON](data/daily/2026-07-02.json) |
| 📄 2026-07-01 | 11 | [View JSON](data/daily/2026-07-01.json) |
| 📄 2026-06-30 | 20 | [View JSON](data/daily/2026-06-30.json) |
| 📄 2026-06-29 | 10 | [View JSON](data/daily/2026-06-29.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W26 | 117 | [View JSON](data/weekly/2026-W26.json) |
| 📅 2026-W25 | 108 | [View JSON](data/weekly/2026-W25.json) |
| 📅 2026-W24 | 130 | [View JSON](data/weekly/2026-W24.json) |
| 📅 2026-W23 | 166 | [View JSON](data/weekly/2026-W23.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-07 | 87 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |

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
