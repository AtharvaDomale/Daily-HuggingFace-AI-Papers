<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-42-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3381+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">98</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">210</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3381+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 11, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability</b> ⭐ 65</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.06628) • [📄 arXiv](https://arxiv.org/abs/2604.06628) • [📥 PDF](https://arxiv.org/pdf/2604.06628)

**💻 Code:** [⭐ Code](https://github.com/Nebularaid2000/rethink_sft_generalization)

> Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability We have open-sourced all our models and datasets: https://huggingface.co/collections/jasonrqh/rethink-sft-generalization Also find them ...

</details>

<details>
<summary><b>2. SkillClaw: Let Skills Evolve Collectively with Agentic Evolver</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yong Wang, Xucong Wang, Yuxiang Ji, Shidong Yang, Ziyu Ma

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08377) • [📄 arXiv](https://arxiv.org/abs/2604.08377) • [📥 PDF](https://arxiv.org/pdf/2604.08377)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/SkillClaw)

> Thanks for sharing our work. The code is released on https://github.com/AMAP-ML/SkillClaw

</details>

<details>
<summary><b>3. HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents</b> ⭐ 223</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07430) • [📄 arXiv](https://arxiv.org/abs/2604.07430) • [📥 PDF](https://arxiv.org/pdf/2604.07430)

**💻 Code:** [⭐ Code](https://github.com/Tencent-Hunyuan/HY-Embodied)

> GitHub: https://github.com/Tencent-Hunyuan/HY-Embodied HuggingFace: https://huggingface.co/tencent/HY-Embodied-0.5

</details>

<details>
<summary><b>4. When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08546) • [📄 arXiv](https://arxiv.org/abs/2604.08546) • [📥 PDF](https://arxiv.org/pdf/2604.08546)

**💻 Code:** [⭐ Code](https://github.com/H-EmbodVis/NUMINA)

> NUMINA is a training-free framework that tackles numerical misalignment in text-to-video diffusion models — the persistent failure of T2V models to generate the correct count of objects specified in prompts (e.g., producing 2 or 4 cats when "three...

</details>

<details>
<summary><b>5. ClawBench: Can AI Agents Complete Everyday Online Tasks?</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08523) • [📄 arXiv](https://arxiv.org/abs/2604.08523) • [📥 PDF](https://arxiv.org/pdf/2604.08523)

**💻 Code:** [⭐ Code](https://github.com/reacher-z/ClawBench)

> TL;DR: ClawBench evaluates AI agents on 153 everyday tasks (such as booking flights, ordering groceries, submitting job applications) across 144 live websites. We capture 5 layers of behavioral data (session replay, screenshots, HTTP traffic, agen...

</details>

<details>
<summary><b>6. MegaStyle: Constructing Diverse and Scalable Style Dataset via Consistent Text-to-Image Style Mapping</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08364) • [📄 arXiv](https://arxiv.org/abs/2604.08364) • [📥 PDF](https://arxiv.org/pdf/2604.08364)

> Visualizations of our style dataset (a)MegaStyle-1.4M and the stylized results produced by our style transfer model (b)MegaStyle-FLUX. MegaStyle-1.4M contains style pairs that share the style but have different content (intra-style consistency), a...

</details>

<details>
<summary><b>7. LPM 1.0: Video-based Character Performance Model</b> ⭐ 121</summary>

<br/>

**👥 Authors:** Chauncey Ge, Casper Yang, Ailing Zeng, YuchenMao, Fan-s

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07823) • [📄 arXiv](https://arxiv.org/abs/2604.07823) • [📥 PDF](https://arxiv.org/pdf/2604.07823)

**💻 Code:** [⭐ Code](https://github.com/large-performance-model/large-performance-model.github.io)

> Long, real-time, multimodal video generation.

</details>

<details>
<summary><b>8. KnowU-Bench: Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation</b> ⭐ 46</summary>

<br/>

**👥 Authors:** Guocheng Shao, Zhan Xu, Zhengxi Lu, Tongbo Chen, yanyc

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08455) • [📄 arXiv](https://arxiv.org/abs/2604.08455) • [📥 PDF](https://arxiv.org/pdf/2604.08455)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/KnowU-Bench)

> We introduce KnowU-Bench, an online, interactive personalization benchmark for mobile agents built on a reproducible Android emulation environment. We find that even frontier models like Claude Sonnet 4.6 struggle with our personalized and proacti...

</details>

<details>
<summary><b>9. Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rong Shan, Wenteng Chen, Huacan Chai, Chenyu Zhou, SII-ZihanGuo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08224) • [📄 arXiv](https://arxiv.org/abs/2604.08224) • [📥 PDF](https://arxiv.org/pdf/2604.08224)

> Large language model (LLM) agents are increasingly built less by changing model weights than by reorganizing the runtime around them. Capabilities that earlier systems expected the model to recover internally are now externalized into memory store...

</details>

<details>
<summary><b>10. OpenVLThinkerV2: A Generalist Multimodal Reasoning Model for Multi-domain Visual Tasks</b> ⭐ 138</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08539) • [📄 arXiv](https://arxiv.org/abs/2604.08539) • [📥 PDF](https://arxiv.org/pdf/2604.08539)

**💻 Code:** [⭐ Code](https://github.com/uclanlp/openvlthinker) • [⭐ Code](https://github.com/uclanlp/OpenVLThinker)

> We introduce OpenVLThinkerV2! Our code are available at: https://github.com/uclanlp/openvlthinker

</details>

<details>
<summary><b>11. Act Wisely: Cultivating Meta-Cognitive Tool Use in Agentic Multimodal Models</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08545) • [📄 arXiv](https://arxiv.org/abs/2604.08545) • [📥 PDF](https://arxiv.org/pdf/2604.08545)

**💻 Code:** [⭐ Code](https://github.com/Accio-Lab/Metis)

> 🔗 Project Page： https://Accio-Lab.github.io/Metis 💻 GitHub： https://github.com/Accio-Lab/Metis 🤗 HuggingFace： https://huggingface.co/Accio-Lab/Metis-8B-RL

</details>

<details>
<summary><b>12. DMax: Aggressive Parallel Decoding for dLLMs</b> ⭐ 53</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08302) • [📄 arXiv](https://arxiv.org/abs/2604.08302) • [📥 PDF](https://arxiv.org/pdf/2604.08302)

**💻 Code:** [⭐ Code](https://github.com/czg1225/DMax)

> DMax is a new dLLM paradigm achieving aggressive parallel decoding while preserving generation quality. Paper: https://arxiv.org/pdf/2604.08302 Code: https://github.com/czg1225/DMax Models: https://huggingface.co/collections/Zigeng/dmax-models Dat...

</details>

<details>
<summary><b>13. MolmoWeb: Open Visual Web Agent and Open Data for the Open Web</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rock Yuren Pang, Peter Sushko, Zixian Ma, Piper Wolters, Tanmay Gupta

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08516) • [📄 arXiv](https://arxiv.org/abs/2604.08516) • [📥 PDF](https://arxiv.org/pdf/2604.08516)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API OpAgent: Operator Agent for Web Navigation (2026) WebFactory: Automated Com...

</details>

<details>
<summary><b>14. OpenSpatial: A Principled Data Engine for Empowering Spatial Intelligence</b> ⭐ 51</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07296) • [📄 arXiv](https://arxiv.org/abs/2604.07296) • [📥 PDF](https://arxiv.org/pdf/2604.07296)

**💻 Code:** [⭐ Code](https://github.com/VINHYU/OpenSpatial)

> Hi HF Community! 👋 We are excited to share OpenSpatial , a principled data engine designed to empower the spatial intelligence of Large Multimodal Models. Key Highlights: 📊 OpenSpatial-3M Dataset: We are open-sourcing a large-scale, high-fidelity ...

</details>

<details>
<summary><b>15. Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.05333) • [📄 arXiv](https://arxiv.org/abs/2604.05333) • [📥 PDF](https://arxiv.org/pdf/2604.05333)

**💻 Code:** [⭐ Code](https://github.com/davidliuk/graph-of-skills)

> Skill usage has become a core component of modern agent systems and can substantially improve agents' ability to complete complex tasks. In real-world settings, where agents must monitor and interact with numerous personal applications, web browse...

</details>

<details>
<summary><b>16. OmniJigsaw: Enhancing Omni-Modal Reasoning via Modality-Orchestrated Reordering</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08209) • [📄 arXiv](https://arxiv.org/abs/2604.08209) • [📥 PDF](https://arxiv.org/pdf/2604.08209)

**💻 Code:** [⭐ Code](https://github.com/aim-uofa/OmniJigsaw)

> We introduce OmniJigsaw, a self-supervised RL post-training framework for omni-modal models. The core idea is a temporal jigsaw proxy task: reconstruct chronology from shuffled audio–visual clips, with three modality-orchestration strategies (JMI ...

</details>

<details>
<summary><b>17. Structured Distillation of Web Agent Capabilities Enables Generalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07776) • [📄 arXiv](https://arxiv.org/abs/2604.07776) • [📥 PDF](https://arxiv.org/pdf/2604.07776)

**💻 Code:** [⭐ Code](https://github.com/McGill-NLP/agent-as-annotators)

> Agent-as-Annotators replaces human annotation roles with LLM modules to synthesize web agent training data. A 9B model trained on 2,322 trajectories matches Qwen3.5-27B and nearly doubles the previous best open-weight result.

</details>

<details>
<summary><b>18. FIT: A Large-Scale Dataset for Fit-Aware Virtual Try-On</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08526) • [📄 arXiv](https://arxiv.org/abs/2604.08526) • [📥 PDF](https://arxiv.org/pdf/2604.08526)

> We introduce FIT (Fit-Inclusive Try-on), a virtual try-on dataset of 1.13M samples covering diverse garment fits, each annotated with precise body/garment measurements. Our novel data generation pipeline leverages synthetic garment simulation and ...

</details>

<details>
<summary><b>19. ViVa: A Video-Generative Value Model for Robot Reinforcement Learning</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08168) • [📄 arXiv](https://arxiv.org/abs/2604.08168) • [📥 PDF](https://arxiv.org/pdf/2604.08168)

**💻 Code:** [⭐ Code](https://github.com/GigaAI-research/ViVa)

> Vision-language-action (VLA) models have advanced robot manipulation through large-scale pretraining, but real-world deployment remains challenging due to partial observability and delayed feedback. Reinforcement learning addresses this via value ...

</details>

<details>
<summary><b>20. SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds</b> ⭐ 55</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08544) • [📄 arXiv](https://arxiv.org/abs/2604.08544) • [📥 PDF](https://arxiv.org/pdf/2604.08544)

**💻 Code:** [⭐ Code](https://github.com/InternRobotics/SIM1)

> SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds SIM1: a world where simulation is the same one as reality, making simulated experience directly executable in the physical world, at scale, without loss. A new scaling l...

</details>

<details>
<summary><b>21. Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07394) • [📄 arXiv](https://arxiv.org/abs/2604.07394) • [📥 PDF](https://arxiv.org/pdf/2604.07394)

**💻 Code:** [⭐ Code](https://github.com/qqtang-code/FluxAttention)

> Paper Title: Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference Link: arXiv:2604.07394 (Preprint) 【TL;DR / One-Sentence Summary】 ⭐⭐⭐⭐⭐ (Highly Recommended). A brilliant hardware-aware co-design that optimizes Long-Context ...

</details>

<details>
<summary><b>22. Automating Database-Native Function Code Synthesis with LLMs</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.06231) • [📄 arXiv](https://arxiv.org/abs/2604.06231) • [📥 PDF](https://arxiv.org/pdf/2604.06231)

**💻 Code:** [⭐ Code](https://github.com/weAIDB/OpenCook)

> 🌟 Start with a generic project. End with a perfectly tailored solution. OpenCook autonomously implements precise, production-grade features to personalize any standard codebase. Driven by deep function characterization and orchestrated subagents, ...

</details>

<details>
<summary><b>23. Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces</b> ⭐ 26</summary>

<br/>

**👥 Authors:** Yunfei Zhang, Ruotong Pan, Boxi Cao, Ruoxi Xu, jiawei-ucas

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08362) • [📄 arXiv](https://arxiv.org/abs/2604.08362) • [📥 PDF](https://arxiv.org/pdf/2604.08362)

**💻 Code:** [⭐ Code](https://github.com/icip-cas/OmniBehavior)

> We introduce OmniBehavior, to our knowledge, the first user simulation benchmark constructed entirely from authentic user interaction logs, integrating long-horizon, cross-scenario and heterogeneous behavior traces into a unified framework. We pro...

</details>

<details>
<summary><b>24. Lighting-grounded Video Generation with Renderer-based Agent Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Han Jiang, Si Li, Zheng Chang, Taoyu Yang, Ziqi Cai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07966) • [📄 arXiv](https://arxiv.org/abs/2604.07966) • [📥 PDF](https://arxiv.org/pdf/2604.07966)

> Accepted to CVPR 2026 TL;DR: LiVER is a controllable video diffusion framework that generates videos from explicit 3D scene conditions such as layout, lighting, and camera motion. It combines a densely annotated dataset, a lightweight conditioning...

</details>

<details>
<summary><b>25. GameWorld: Towards Standardized and Verifiable Evaluation of Multimodal Game Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mike Zheng Shou, Hwee Tou Ng, Kevin Qinghong Lin, Siyuan Hu, Mingyu Ouyang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07429) • [📄 arXiv](https://arxiv.org/abs/2604.07429) • [📥 PDF](https://arxiv.org/pdf/2604.07429)

> Proposes GameWorld, a standardized, verifiable benchmark for evaluating multimodal game agents across 34 browser games and 170 tasks using semantic actions and keyboard/mouse interfaces.

</details>

<details>
<summary><b>26. Small Vision-Language Models are Smart Compressors for Long Video Understanding</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Chong Zhou, Yunyang Xiong, Zechun Liu, Jun Chen, FeiElysia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08120) • [📄 arXiv](https://arxiv.org/abs/2604.08120) • [📥 PDF](https://arxiv.org/pdf/2604.08120)

**💻 Code:** [⭐ Code](https://github.com/FeiElysia/Tempo)

> 🔥 Tempo: Small Vision-Language Models are Smart Compressors for Long Video Understanding How do we make MLLMs understand hour-long videos without saturating context windows? Tempo uses an SVLM to actively filter and compress videos via query-aware...

</details>

<details>
<summary><b>27. Faithful GRPO: Improving Visual Spatial Reasoning in Multimodal Language Models via Constrained Policy Optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08476) • [📄 arXiv](https://arxiv.org/abs/2604.08476) • [📥 PDF](https://arxiv.org/pdf/2604.08476)

> Multimodal reasoning models (MRMs) trained with reinforcement learning with verifiable rewards (RLVR) show improved accuracy on visual reasoning benchmarks. However, we observe that accuracy gains often come at the cost of reasoning quality: gener...

</details>

<details>
<summary><b>28. The Master Key Hypothesis: Unlocking Cross-Model Capability Transfer via Linear Subspace Alignment</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fardin Abdi, Anjie Fang, Rituraj Sharma, Pin-Jie Lin, Rishab Balasubramanian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.06377) • [📄 arXiv](https://arxiv.org/abs/2604.06377) • [📥 PDF](https://arxiv.org/pdf/2604.06377)

> Sharing our new work on The Master Key Hypothesis: Unlocking Cross-Model Capability Transfer via Linear Subspace Alignment

</details>

<details>
<summary><b>29. AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20524) • [📄 arXiv](https://arxiv.org/abs/2601.20524) • [📥 PDF](https://arxiv.org/pdf/2601.20524)

**💻 Code:** [⭐ Code](https://github.com/MaticFuc/AnomalyVFM)

> Added to Daily papers

</details>

<details>
<summary><b>30. On the Global Photometric Alignment for Low-Level Vision</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaojie Guo, Qiming Hu, Hainuo Wang, Tianle Du, Mingjia Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08172) • [📄 arXiv](https://arxiv.org/abs/2604.08172) • [📥 PDF](https://arxiv.org/pdf/2604.08172)

> Supervised low-level vision models rely on pixel-wise losses against paired references, yet paired training sets exhibit per-pair photometric inconsistency, say, different image pairs demand different global brightness, color, or white-balance map...

</details>

<details>
<summary><b>31. ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08064) • [📄 arXiv](https://arxiv.org/abs/2604.08064) • [📥 PDF](https://arxiv.org/pdf/2604.08064)

> Beyond explicit recall: we benchmark whether LLMs can learn from experience and adapt behavior without conscious retrieval.

</details>

<details>
<summary><b>32. Beyond Stochastic Exploration: What Makes Training Data Valuable for Agentic Search</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guohua Liu, Guofeng Quan, Wenfeng Feng, Nothing2Say, Chuzhan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08124) • [📄 arXiv](https://arxiv.org/abs/2604.08124) • [📥 PDF](https://arxiv.org/pdf/2604.08124)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API SIGHT: Reinforcement Learning with Self-Evidence and Information-Gain Diver...

</details>

<details>
<summary><b>33. Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08503) • [📄 arXiv](https://arxiv.org/abs/2604.08503) • [📥 PDF](https://arxiv.org/pdf/2604.08503)

> Phantom is a Physics-Infused Video Generation model that jointly models visual content and latent physical dynamics. Conditioned on observed video frames and inferred physical states, Phantom jointly predicts latent physical dynamics and generates...

</details>

<details>
<summary><b>34. PokeGym: A Visually-Driven Long-Horizon Benchmark for Vision-Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhilin Liu, Chuanfu Shen, Yuangang Pan, Ye Huang, Ruizhi Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08340) • [📄 arXiv](https://arxiv.org/abs/2604.08340) • [📥 PDF](https://arxiv.org/pdf/2604.08340)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API OmniEarth: A Benchmark for Evaluating Vision-Language Models in Geospatial ...

</details>

<details>
<summary><b>35. POS-ISP: Pipeline Optimization at the Sequence Level for Task-aware ISP</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Sunghyun Cho, Jungseul Ok, Woohyeok Kim, Heemin Yang, jiyunwon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.06938) • [📄 arXiv](https://arxiv.org/abs/2604.06938) • [📥 PDF](https://arxiv.org/pdf/2604.06938)

**💻 Code:** [⭐ Code](https://github.com/w1jyun/POS-ISP)

> Recent work has explored optimizing image signal processing (ISP) pipelines for various tasks by composing predefined modules and adapting them to task-specific objectives. However, jointly optimizing module sequences and parameters remains challe...

</details>

<details>
<summary><b>36. QEIL v2: Heterogeneous Computing for Edge Intelligence via Roofline-Derived Pareto-Optimal Energy Modeling and Multi-Objective Orchestration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Saurabh Jha, Satyamk098

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06057) • [📄 arXiv](https://arxiv.org/abs/2602.06057) • [📥 PDF](https://arxiv.org/pdf/2602.06057)

> Excited to share QEIL v2: Roofline-Derived Pareto-Optimal Edge Intelligence via First-Principles Energy Modeling and Multi-Objective Orchestration. This work replaces static heuristics with physics-grounded runtime modeling using DASI, CPQ, and Φ,...

</details>

<details>
<summary><b>37. RewardFlow: Generate Images by Optimizing What You Reward</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08536) • [📄 arXiv](https://arxiv.org/abs/2604.08536) • [📥 PDF](https://arxiv.org/pdf/2604.08536)

> We introduce RewardFlow, an inversion-free framework that steers pretrained diffusion and flow-matching models at inference time through multi-reward Langevin dynamics.

</details>

<details>
<summary><b>38. Appear2Meaning: A Cross-Cultural Benchmark for Structured Cultural Metadata Inference from Images</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07338) • [📄 arXiv](https://arxiv.org/abs/2604.07338) • [📥 PDF](https://arxiv.org/pdf/2604.07338)

> We introduce Appear2Meaning , a cross-cultural benchmark for structured cultural metadata inference from images. Unlike standard image captioning, this task requires models to predict non-observable attributes such as culture, period, origin, and ...

</details>

<details>
<summary><b>39. Training a Student Expert via Semi-Supervised Foundation Model Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.03841) • [📄 arXiv](https://arxiv.org/abs/2604.03841) • [📥 PDF](https://arxiv.org/pdf/2604.03841)

> We propose a semi-supervised distillation framework for compressing vision foundation models into compact instance segmentation experts. Using self-training, multi-objective distillation, and an instance-aware pixel-wise contrastive loss, our meth...

</details>

<details>
<summary><b>40. Structural Graph Probing of Vision-Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.27070) • [📄 arXiv](https://arxiv.org/abs/2603.27070) • [📥 PDF](https://arxiv.org/pdf/2603.27070)

**💻 Code:** [⭐ Code](https://github.com/he-h/vlm-graphprobing) • [⭐ Code](https://github.com/he-h/vlm-graph-probing)

> Vision-language models (VLMs) achieve strong multimodal performance, yet how computation is organized across populations of neurons remains poorly understood. In this work, we study VLMs through the lens of neural topology, representing each layer...

</details>

<details>
<summary><b>41. CylinderDepth: Cylindrical Spatial Attention for Multi-View Consistent Self-Supervised Surround Depth Estimation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2511.16428) • [📄 arXiv](https://arxiv.org/abs/2511.16428) • [📥 PDF](https://arxiv.org/pdf/2511.16428)

**💻 Code:** [⭐ Code](https://github.com/abualhanud/CylinderDepth)

> Self-supervised surround-view depth estimation enables dense, low-cost 3D perception with a 360◦ field of view from multiple minimally overlapping images. Yet, most existing methods suffer from depth estimates that are inconsistent across overlapp...

</details>

<details>
<summary><b>42. Personalizing Text-to-Image Generation to Individual Taste</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07427) • [📄 arXiv](https://arxiv.org/abs/2604.07427) • [📥 PDF](https://arxiv.org/pdf/2604.07427)

> Modern text-to-image (T2I) models generate high-fidelity visuals but remain indifferent to individual user preferences. While existing reward models optimize for "average" human appeal, they fail to capture the inherent subjectivity of aesthetic j...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 42 |
| 📅 Today | [`2026-04-11.json`](data/daily/2026-04-11.json) | 42 |
| 📆 This Week | [`2026-W14.json`](data/weekly/2026-W14.json) | 98 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 210 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-11 | 42 | [View JSON](data/daily/2026-04-11.json) |
| 📄 2026-04-10 | 18 | [View JSON](data/daily/2026-04-10.json) |
| 📄 2026-04-09 | 8 | [View JSON](data/daily/2026-04-09.json) |
| 📄 2026-04-08 | 8 | [View JSON](data/daily/2026-04-08.json) |
| 📄 2026-04-07 | 17 | [View JSON](data/daily/2026-04-07.json) |
| 📄 2026-04-06 | 5 | [View JSON](data/daily/2026-04-06.json) |
| 📄 2026-04-05 | 45 | [View JSON](data/daily/2026-04-05.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W14 | 98 | [View JSON](data/weekly/2026-W14.json) |
| 📅 2026-W13 | 115 | [View JSON](data/weekly/2026-W13.json) |
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 210 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
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
