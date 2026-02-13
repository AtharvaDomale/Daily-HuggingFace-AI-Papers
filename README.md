<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-47-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2132+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">47</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">211</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">613</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2132+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 13, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters</b> ⭐ 1.25k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10604) • [📄 arXiv](https://arxiv.org/abs/2602.10604) • [📥 PDF](https://arxiv.org/pdf/2602.10604)

**💻 Code:** [⭐ Code](https://github.com/stepfun-ai/Step-3.5-Flash)

> Step-3.5-Flash is #1 on MathArena , an uncheatable math competition benchmark

</details>

<details>
<summary><b>2. PhyCritic: Multimodal Critic Models for Physical AI</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11124) • [📄 arXiv](https://arxiv.org/abs/2602.11124) • [📥 PDF](https://arxiv.org/pdf/2602.11124)

> A multimodal critic model that unifies physical judging and reasoning.

</details>

<details>
<summary><b>3. GENIUS: Generative Fluid Intelligence Evaluation Suite</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zijun Shen, Wei Dai, Ziyu Guo, Sihan Yang, Ruichuan An

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11144) • [📄 arXiv](https://arxiv.org/abs/2602.11144) • [📥 PDF](https://arxiv.org/pdf/2602.11144)

> No abstract available.

</details>

<details>
<summary><b>4. ASA: Training-Free Representation Engineering for Tool-Calling Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hongwei Zeng, Shuaishuai Cao, Rong Fu, Run Zhou, wangyoujin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04935) • [📄 arXiv](https://arxiv.org/abs/2602.04935) • [📥 PDF](https://arxiv.org/pdf/2602.04935)

> Adapting LLM agents to domain-specific tool calling remains notably brittle under evolving interfaces. Prompt and schema engineering is easy to deploy but often fragile under distribution shift and strict parsers, while continual parameter-efficie...

</details>

<details>
<summary><b>5. Towards Autonomous Mathematics Research</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10177) • [📄 arXiv](https://arxiv.org/abs/2602.10177) • [📥 PDF](https://arxiv.org/pdf/2602.10177)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Semi-Autonomous Mathematics Discovery with Gemini: A Case Study on the Erd\...

</details>

<details>
<summary><b>6. G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Liang Zeng, iphysresearch, ZBoyn

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08253) • [📄 arXiv](https://arxiv.org/abs/2602.08253) • [📥 PDF](https://arxiv.org/pdf/2602.08253)

**💻 Code:** [⭐ Code](https://github.com/ZBoyn/G-LNS)

> We’re moving from constructive rules to recursive destruction & repair 🔄. G-LNS introduces Synergy-Aware Co-evolution, allowing LLMs to generate coupled Destroy/Repair operators that break local optima. Reshaping > Constructing. 💡 It beats OR-Tool...

</details>

<details>
<summary><b>7. How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10622) • [📄 arXiv](https://arxiv.org/abs/2602.10622) • [📥 PDF](https://arxiv.org/pdf/2602.10622)

> 🎉 How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning Decoder-only LLMs have demonstrated remarkable generative capabilities, but how well do they understand users when repurposed for representati...

</details>

<details>
<summary><b>8. When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10560) • [📄 arXiv](https://arxiv.org/abs/2602.10560) • [📥 PDF](https://arxiv.org/pdf/2602.10560)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API InfMem: Learning System-2 Memory Control for Long-Context Agent (2026) Dyna...

</details>

<details>
<summary><b>9. TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08711) • [📄 arXiv](https://arxiv.org/abs/2602.08711) • [📥 PDF](https://arxiv.org/pdf/2602.08711)

**💻 Code:** [⭐ Code](https://github.com/yaolinli/TimeChat-Captioner)

> TimeChat-Captioner is a multimodal model designed to generate detailed, time-aware, and structurally coherent captions for multi-scene videos. It effectively coordinates visual and audio information to provide comprehensive video descriptions.

</details>

<details>
<summary><b>10. FeatureBench: Benchmarking Agentic Coding for Complex Feature Development</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Jiahe Wang, Rui Hao, Qixing Zhou, Haiyang-W, jiachengzhg

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10975) • [📄 arXiv](https://arxiv.org/abs/2602.10975) • [📥 PDF](https://arxiv.org/pdf/2602.10975)

**💻 Code:** [⭐ Code](https://github.com/LiberCoders/FeatureBench)

> FeatureBench focuses on evaluating the end-to-end development capability of coding agents for complex features. On our benchmark, even the strongest commercial models can solve only about 12% of the tasks. The full Docker environment and the scala...

</details>

<details>
<summary><b>11. ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11008) • [📄 arXiv](https://arxiv.org/abs/2602.11008) • [📥 PDF](https://arxiv.org/pdf/2602.11008)

**💻 Code:** [⭐ Code](https://github.com/mts-ai/ROCKET)

> ROCKET isn’t just another compression method. It is one of the first methods to shrink massive AI models down to compact sizes without sacrificing performance, often matching or even outperforming vanilla models of the same size trained from scrat...

</details>

<details>
<summary><b>12. Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhen Fang, Qingnan Ren, Zecheng Li, YuZeng260, chocckaka

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10224) • [📄 arXiv](https://arxiv.org/abs/2602.10224) • [📥 PDF](https://arxiv.org/pdf/2602.10224)

> We propose Meta-Experience Learning (MEL), which breaks the meta-learning and credit-assignment bottleneck of standard RLVR by explicitly modeling and internalizing reusable error-based knowledge. MEL exploits an LLM's self-verification ability to...

</details>

<details>
<summary><b>13. DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Kai Chen, Yining Li, Xinchen Xie, Zerun Ma, Yicheng Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11089) • [📄 arXiv](https://arxiv.org/abs/2602.11089) • [📥 PDF](https://arxiv.org/pdf/2602.11089)

**💻 Code:** [⭐ Code](https://github.com/yichengchen24/DataChef)

> demo: https://huggingface.co/spaces/yichengchen24/DataChef

</details>

<details>
<summary><b>14. GameDevBench: Evaluating Agentic Capabilities Through Game Development</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11103) • [📄 arXiv](https://arxiv.org/abs/2602.11103) • [📥 PDF](https://arxiv.org/pdf/2602.11103)

> Can agents develop video games? GameDevBench is the first benchmark to evaluate an agent's ability to solve game development tasks.

</details>

<details>
<summary><b>15. Online Causal Kalman Filtering for Stable and Effective Policy Optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10609) • [📄 arXiv](https://arxiv.org/abs/2602.10609) • [📥 PDF](https://arxiv.org/pdf/2602.10609)

> (Work in progress) We are adding more comparison methods and models for KPO and will soon open-source KPO.

</details>

<details>
<summary><b>16. Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yuki M. Asano, Tijmen Blankevoort, Sagar Vaze, dakopi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11149) • [📄 arXiv](https://arxiv.org/abs/2602.11149) • [📥 PDF](https://arxiv.org/pdf/2602.11149)

**💻 Code:** [⭐ Code](https://github.com/dkopi/data-repetition)

> Pretty interesting findings!

</details>

<details>
<summary><b>17. Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tianshu Yu, Yiwen Guo, Zhipeng Li, lemonade666

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07106) • [📄 arXiv](https://arxiv.org/abs/2602.07106) • [📥 PDF](https://arxiv.org/pdf/2602.07106)

> Omni-modal large language models (OLLMs) aim to unify multimodal understanding and generation, yet incorporating speech with 3D facial animation remains largely unexplored despite its importance for natural interaction. A key challenge arises from...

</details>

<details>
<summary><b>18. CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Feiyang Pan, Lue Fan, Shuzhe Wu, Yusong Lin, Haiyang-W

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10999) • [📄 arXiv](https://arxiv.org/abs/2602.10999) • [📥 PDF](https://arxiv.org/pdf/2602.10999)

**💻 Code:** [⭐ Code](https://github.com/LiberCoders/CLI-Gym)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Ter...

</details>

<details>
<summary><b>19. LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Xiang Li, Yisheng Ji, Zhe Fang, Dingjie Song, Zhiling Yan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10367) • [📄 arXiv](https://arxiv.org/abs/2602.10367) • [📥 PDF](https://arxiv.org/pdf/2602.10367)

**💻 Code:** [⭐ Code](https://github.com/ZhilingYan/LiveMedBench)

> LiveMedBench is a continuously updated, contamination-free, and rubric-based benchmark for evaluating LLMs on real-world medical cases. It is designed to measure not only overall medical quality, but also robustness over time and alignment with ph...

</details>

<details>
<summary><b>20. Blockwise Advantage Estimation for Multi-Objective RL with Verifiable Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10231) • [📄 arXiv](https://arxiv.org/abs/2602.10231) • [📥 PDF](https://arxiv.org/pdf/2602.10231)

> Blockwise Advantage Estimation makes GRPO work for segmented, multi-objective generations by routing each objective’s learning signal to the tokens that control it, using an outcome-conditioned baseline for later segments.

</details>

<details>
<summary><b>21. EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Yishuo Yuan, Kangqi Song, Shengze Xu, Jinxiang Xia, Xavier Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09514) • [📄 arXiv](https://arxiv.org/abs/2602.09514) • [📥 PDF](https://arxiv.org/pdf/2602.09514)

**💻 Code:** [⭐ Code](https://github.com/OPPO-PersonalAI/EcoGym)

> Long-horizon planning is widely recognized as a core capability of autonomous LLM-based agents; however, current evaluation frameworks suffer from being largely episodic, domain-specific, or insufficiently grounded in persistent economic dynamics....

</details>

<details>
<summary><b>22. VidVec: Unlocking Video MLLM Embeddings for Video-Text Retrieval</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rami Ben-Ari, Dvir Samuel, issart12345

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08099) • [📄 arXiv](https://www.arxiv.org/abs/2602.08099) • [📥 PDF](https://arxiv.org/pdf/2602.08099)

> What if your multimodal LLM already contains strong video representations—strong enough to beat Video Foundation Models? 🤔 VidVec 🎥 : Unlocking Video MLLM Embeddings for Video-Text Retrieval Key contributions (short): ✅ Layer-wise insight: interme...

</details>

<details>
<summary><b>23. Stroke3D: Lifting 2D strokes into rigged 3D model via latent diffusion models</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09713) • [📄 arXiv](https://arxiv.org/abs/2602.09713) • [📥 PDF](https://arxiv.org/pdf/2602.09713)

**💻 Code:** [⭐ Code](https://github.com/Whalesong-zrs/Stroke3D) • [⭐ Code](https://github.com/Whalesong-zrs/Stroke3D_project_page)

> Project Page: https://whalesong-zrs.github.io/Stroke3D_project_page/ Github Repo: https://github.com/Whalesong-zrs/Stroke3D

</details>

<details>
<summary><b>24. ECHO-2: A Large-Scale Distributed Rollout Framework for Cost-Efficient Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02192) • [📄 arXiv](https://arxiv.org/abs/2602.02192) • [📥 PDF](https://arxiv.org/pdf/2602.02192)

> Current RLHF/RLAIF is bottlenecked by rollouts and wasteful GPU idling. ECHO-2 changes the cost structure: we decouple RL into three planes—rollout (global inference swarm), learning (staleness-aware multi-step updates), and data/reward (fully mod...

</details>

<details>
<summary><b>25. When the Prompt Becomes Visual: Vision-Centric Jailbreak Attacks for Large Image Editing Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10179) • [📄 arXiv](https://arxiv.org/abs/2602.10179) • [📥 PDF](https://arxiv.org/pdf/2602.10179)

> Website: https://csu-jpg.github.io/vja.github.io/

</details>

<details>
<summary><b>26. QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hui Zhang, Yunpeng Liu, Xiaorui Huang, Jianzhao Huang, Hiiamein

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09901) • [📄 arXiv](https://arxiv.org/abs/2602.09901) • [📥 PDF](https://arxiv.org/pdf/2602.09901)

> QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search

</details>

<details>
<summary><b>27. Latent Thoughts Tuning: Bridging Context and Reasoning with Fused Information in Latent Tokens</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10229) • [📄 arXiv](https://arxiv.org/abs/2602.10229) • [📥 PDF](https://arxiv.org/pdf/2602.10229)

**💻 Code:** [⭐ Code](https://github.com/NeosKnight233/Latent-Thoughts-Tuning)

> a new framework for LLM reasoning in continuous latent space

</details>

<details>
<summary><b>28. Beyond Correctness: Learning Robust Reasoning via Transfer</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jinwoo Shin, Jihoon Tack, Soheil Abbasloo, hyunseoki

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08489) • [📄 arXiv](https://arxiv.org/abs/2602.08489) • [📥 PDF](https://arxiv.org/pdf/2602.08489)

> Reinforcement Learning with Verifiable Rewards (RLVR) has recently strengthened LLM reasoning, but its focus on final answer correctness leaves a critical gap: it does not ensure the robustness of the reasoning process itself. We adopt a simple ph...

</details>

<details>
<summary><b>29. Free(): Learning to Forget in Malloc-Only Reasoning Models</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08030) • [📄 arXiv](https://arxiv.org/abs/2602.08030) • [📥 PDF](https://arxiv.org/pdf/2602.08030)

**💻 Code:** [⭐ Code](https://github.com/TemporaryLoRA/FreeLM)

> The Magic of Forgetting! Reasoning models enhance problem-solving by scaling test-time compute, yet they face a critical paradox: excessive thinking tokens often degrade performance rather than improve it. We attribute this to a fundamental archit...

</details>

<details>
<summary><b>30. Benchmarking Large Language Models for Knowledge Graph Validation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10748) • [📄 arXiv](https://arxiv.org/abs/2602.10748) • [📥 PDF](https://arxiv.org/pdf/2602.10748)

**💻 Code:** [⭐ Code](https://github.com/FactCheck-AI) • [⭐ Code](https://github.com/FactCheck-AI/FactCheck)

> In this work, we introduce FactCheck, a benchmark to systematically evaluate LLMs for fact validation over Knowledge Graphs, covering internal model knowledge, Retrieval-Augmented Generation (RAG), and multi-model consensus strategies across three...

</details>

<details>
<summary><b>31. Bielik Guard: Efficient Polish Language Safety Classifiers for LLM Content Moderation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07954) • [📄 arXiv](https://arxiv.org/abs/2602.07954) • [📥 PDF](https://arxiv.org/pdf/2602.07954)

> Bielik Guard is a family of compact Polish-language safety classifiers (0.1B and 0.5B parameters) that accurately detect harmful content across five categories, achieving strong benchmark performance—with the 0.5B model offering the best overall F...

</details>

<details>
<summary><b>32. AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06008) • [📄 arXiv](https://arxiv.org/abs/2602.06008) • [📥 PDF](https://arxiv.org/pdf/2602.06008)

**💻 Code:** [⭐ Code](https://github.com/SafeRL-Lab/AgenticPay)

> Paper: https://arxiv.org/abs/2602.06008 Code: https://github.com/SafeRL-Lab/AgenticPay Tutorial: https://agenticpay-tutorial.readthedocs.io/en/latest/

</details>

<details>
<summary><b>33. Reasoning Cache: Continual Improvement Over Long Horizons via Short-Horizon RL</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aviral Kumar, Amrith Setlur, Yuxiao Qu, Ian Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03773) • [📄 arXiv](https://arxiv.org/abs/2602.03773) • [📥 PDF](https://arxiv.org/pdf/2602.03773)

> Reasoning Cache: Continual Improvement Over Long Horizons via Short-Horizon RL Large Language Models (LLMs) that can continually improve beyond their training budgets are able to solve increasingly difficult problems by adapting at test time, a pr...

</details>

<details>
<summary><b>34. ArcFlow: Unleashing 2-Step Text-to-Image Generation via High-Precision Non-Linear Flow Distillation</b> ⭐ 50</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09014) • [📄 arXiv](https://arxiv.org/abs/2602.09014) • [📥 PDF](https://arxiv.org/pdf/2602.09014)

**💻 Code:** [⭐ Code](https://github.com/pnotp/ArcFlow)

> In this work, we revisit few-step distillation from a geometric perspective. Based on the observation that teacher trajectories exhibit inherently non-linear dynamics, ArcFlow introduces a momentum-based velocity parameterization with an analytic ...

</details>

<details>
<summary><b>35. Rethinking the Value of Agent-Generated Tests for LLM-Based Software Engineering Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07900) • [📄 arXiv](https://arxiv.org/abs/2602.07900) • [📥 PDF](https://arxiv.org/pdf/2602.07900)

> In autonomous issue resolution, agent-written tests often increase interaction cost without meaningfully increasing task success.

</details>

<details>
<summary><b>36. LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11451) • [📄 arXiv](https://arxiv.org/abs/2602.11451) • [📥 PDF](https://arxiv.org/pdf/2602.11451)

**💻 Code:** [⭐ Code](https://github.com/armenjeddi/loopformer)

> The LoopFormer Paper accepted to ICLR 2026

</details>

<details>
<summary><b>37. Weight Decay Improves Language Model Plasticity</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sham Kakade, Hanlin Zhang, Sebastian Bordt, Tessa Han

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.11137) • [📄 arXiv](https://arxiv.org/abs/2602.11137) • [📥 PDF](https://arxiv.org/pdf/2602.11137)

> Increasing weight decay during language model pretraining enhances model plasticity, enabling greater performance gains after fine-tuning even when base validation loss is worse, and highlights the need to optimize hyperparameters with downstream ...

</details>

<details>
<summary><b>38. UMEM: Unified Memory Extraction and Management Framework for Generalizable Memory</b> ⭐ 247</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10652) • [📄 arXiv](https://arxiv.org/abs/2602.10652) • [📥 PDF](https://arxiv.org/pdf/2602.10652)

**💻 Code:** [⭐ Code](https://github.com/AIDC-AI/Marco-DeepResearch)

> UMEM: Unified Memory Extraction and Management Framework for Generalizable Memory This paper presents a systematic solution to a core bottleneck in self-evolving agents, offering the following notable contributions: Core Problem Insight The author...

</details>

<details>
<summary><b>39. When Actions Go Off-Task: Detecting and Correcting Misaligned Actions in Computer-Use Agents</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08995) • [📄 arXiv](https://arxiv.org/abs/2602.08995) • [📥 PDF](https://arxiv.org/pdf/2602.08995)

**💻 Code:** [⭐ Code](https://github.com/OSU-NLP-Group/Misaligned-Action-Detection)

> Project Homepage: https://osu-nlp-group.github.io/Misaligned-Action-Detection/ Github Repo: https://github.com/OSU-NLP-Group/Misaligned-Action-Detection Benchmark: https://huggingface.co/datasets/osunlp/MisActBench

</details>

<details>
<summary><b>40. TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02459) • [📄 arXiv](https://arxiv.org/abs/2602.02459) • [📥 PDF](https://arxiv.org/pdf/2602.02459)

**💻 Code:** [⭐ Code](https://github.com/ucla-mobility/TIC-VLA)

> Robots in dynamic, human-centric environments must follow language instructions while maintaining real-time reactive control. Vision-language-action (VLA) models offer a promising framework, but they assume temporally aligned reasoning and control...

</details>

<details>
<summary><b>41. FedPS: Federated data Preprocessing via aggregated Statistics</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Graham Cormode, xuefeng-xu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10870) • [📄 arXiv](https://arxiv.org/abs/2602.10870) • [📥 PDF](https://arxiv.org/pdf/2602.10870)

**💻 Code:** [⭐ Code](https://github.com/xuefeng-xu/fedps)

> TL;DR: A unified framework for tabular data preprocessing in federated learning.

</details>

<details>
<summary><b>42. GoodVibe: Security-by-Vibe for LLM-Based Code Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10778) • [📄 arXiv](https://arxiv.org/abs/2602.10778) • [📥 PDF](https://arxiv.org/pdf/2602.10778)

> Large language models (LLMs) are increasingly used for code generation in fast, informal development workflows, often referred to as vibe coding, where speed and convenience are prioritized, and security requirements are rarely made explicit. In t...

</details>

<details>
<summary><b>43. Spend Search Where It Pays: Value-Guided Structured Sampling and Optimization for Generative Recommendation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuling Xiong, Changping Wang, Zeyu Wang, Yangru Huang, Jie Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10699) • [📄 arXiv](https://arxiv.org/abs/2602.10699) • [📥 PDF](https://arxiv.org/pdf/2602.10699)

> V-STAR introduces value-guided decoding and tree-structured advantage reinforcement learning for generative recommendations, boosting exploration, diversity, and latency-constrained accuracy.

</details>

<details>
<summary><b>44. Large Language Lobotomy: Jailbreaking Mixture-of-Experts via Expert Silencing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08741) • [📄 arXiv](https://arxiv.org/abs/2602.08741) • [📥 PDF](https://arxiv.org/pdf/2602.08741)

**💻 Code:** [⭐ Code](https://github.com/jonatelintelo/LargeLanguageLobotomy)

> The rapid adoption of Mixture-of-Experts (MoE) architectures marks a major shift in the deployment of Large Language Models (LLMs). MoE LLMs improve scaling efficiency by activating only a small subset of parameters per token, but their routing st...

</details>

<details>
<summary><b>45. Graph-Enhanced Deep Reinforcement Learning for Multi-Objective Unrelated Parallel Machine Scheduling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Grace Bochenek, Ghaith Rabadi, Sean Mondesire, Bulent Soykan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08052) • [📄 arXiv](https://arxiv.org/abs/2602.08052) • [📥 PDF](https://arxiv.org/pdf/2602.08052)

**💻 Code:** [⭐ Code](https://github.com/bulentsoykan/GNN-DRL4UPMSP)

> The Unrelated Parallel Machine Scheduling Problem (UPMSP) with release dates, setups, and eligibility constraints presents a significant multi-objective challenge. Traditional methods struggle to balance minimizing Total Weighted Tardiness (TWT) a...

</details>

<details>
<summary><b>46. StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08934) • [📄 arXiv](https://arxiv.org/abs/2602.08934) • [📥 PDF](https://arxiv.org/pdf/2602.08934)

**💻 Code:** [⭐ Code](https://github.com/suraj-ranganath/StealthRL)

> StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors. Happy to discuss and get feedback!

</details>

<details>
<summary><b>47. From Features to Actions: Explainability in Traditional and Agentic AI Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06841) • [📄 arXiv](https://arxiv.org/abs/2602.06841) • [📥 PDF](https://arxiv.org/pdf/2602.06841)

**💻 Code:** [⭐ Code](https://github.com/VectorInstitute/unified-xai-evaluation-framework)

> As AI systems move from single predictions to autonomous, multi-step agents, our notion of explainability must evolve. In this paper, we show why traditional feature-attribution methods (e.g., SHAP, LIME) are insufficient for diagnosing failures i...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 47 |
| 📅 Today | [`2026-02-13.json`](data/daily/2026-02-13.json) | 47 |
| 📆 This Week | [`2026-W06.json`](data/weekly/2026-W06.json) | 211 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 613 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-13 | 47 | [View JSON](data/daily/2026-02-13.json) |
| 📄 2026-02-12 | 57 | [View JSON](data/daily/2026-02-12.json) |
| 📄 2026-02-11 | 58 | [View JSON](data/daily/2026-02-11.json) |
| 📄 2026-02-10 | 2 | [View JSON](data/daily/2026-02-10.json) |
| 📄 2026-02-09 | 47 | [View JSON](data/daily/2026-02-09.json) |
| 📄 2026-02-08 | 47 | [View JSON](data/daily/2026-02-08.json) |
| 📄 2026-02-07 | 47 | [View JSON](data/daily/2026-02-07.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W06 | 211 | [View JSON](data/weekly/2026-W06.json) |
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 613 | [View JSON](data/monthly/2026-02.json) |
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
