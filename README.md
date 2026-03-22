<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-32-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3048+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📆 This Month</b><br/><font size="5">481</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3048+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 22, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding</b> ⭐ 109</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19235) • [📄 arXiv](https://arxiv.org/abs/2603.19235) • [📥 PDF](https://arxiv.org/pdf/2603.19235)

**💻 Code:** [⭐ Code](https://github.com/H-EmbodVis/VEGA-3D)

> While Multimodal Large Language Models demonstrate impressive semantic capabilities, they often suffer from spatial blindness, struggling with fine-grained geometric reasoning and physical dynamics. Existing solutions typically rely on explicit 3D...

</details>

<details>
<summary><b>2. SAMA: Factorized Semantic Anchoring and Motion Alignment for Instruction-Guided Video Editing</b> ⭐ 33</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19228) • [📄 arXiv](https://arxiv.org/abs/2603.19228) • [📥 PDF](https://arxiv.org/pdf/2603.19228)

**💻 Code:** [⭐ Code](https://github.com/Cynthiazxy123/SAMA)

> SAMA factorizes instruction-guided video editing into semantic anchoring and motion alignment, improving edit precision while preserving temporal dynamics from the source video. SAMA achieves state-of-the-art performance among open-source models a...

</details>

<details>
<summary><b>3. 3DreamBooth: High-Fidelity 3D Subject-Driven Video Generation Model</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18524) • [📄 arXiv](https://arxiv.org/abs/2603.18524) • [📥 PDF](https://arxiv.org/pdf/2603.18524)

**💻 Code:** [⭐ Code](https://github.com/Ko-Lani/3DreamBooth)

> Key idea: 3DreamBooth treats subjects as 3D entities, not 2D — enabling multiview-consistent video generation by baking a spatial prior through 1-frame optimization. Highlights: 3DreamBooth decouples 3D geometry from temporal motion via 1-frame sp...

</details>

<details>
<summary><b>4. FASTER: Rethinking Real-Time Flow VLAs</b> ⭐ 47</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19199) • [📄 arXiv](https://arxiv.org/abs/2603.19199) • [📥 PDF](https://arxiv.org/pdf/2603.19199)

**💻 Code:** [⭐ Code](https://github.com/innovator-zero/FASTER)

> Real-time reaction in VLAs is constrained not only by inference latency, but also by how action chunks are generated and executed. FASTER introduces a new paradigm for fast action sampling under asynchronous execution. By compressing the sampling ...

</details>

<details>
<summary><b>5. Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19220) • [📄 arXiv](https://arxiv.org/abs/2603.19220) • [📥 PDF](https://arxiv.org/pdf/2603.19220)

> We release the Nemotron-Cascade-2-30B-A3B model and training data at: https://huggingface.co/collections/nvidia/nemotron-cascade-2

</details>

<details>
<summary><b>6. Bridging Semantic and Kinematic Conditions with Diffusion-based Discrete Motion Tokenizer</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19227) • [📄 arXiv](https://arxiv.org/abs/2603.19227) • [📥 PDF](https://arxiv.org/pdf/2603.19227)

**💻 Code:** [⭐ Code](https://github.com/rheallyc/MoTok)

> Project Page: https://rheallyc.github.io/projects/motok/ GitHub: https://github.com/rheallyc/MoTok

</details>

<details>
<summary><b>7. Memento-Skills: Let Agents Design Agents</b> ⭐ 61</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18743) • [📄 arXiv](https://arxiv.org/abs/2603.18743) • [📥 PDF](https://arxiv.org/pdf/2603.18743)

**💻 Code:** [⭐ Code](https://github.com/Memento-Teams/Memento-Skills)

> We introduceMemento-Skills, a generalist, continually-learnable LLM agent system that functions as an agent-designing agent: it autonomously constructs, adapts, and improves task-specific agents through experience. The system is built on a memory-...

</details>

<details>
<summary><b>8. Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens</b> ⭐ 39</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19232) • [📄 arXiv](https://arxiv.org/abs/2603.19232) • [📥 PDF](https://arxiv.org/pdf/2603.19232)

**💻 Code:** [⭐ Code](https://github.com/YuqingWang1029/CubiD)

> Can we generate high-dimensional semantic representations discretely, just like language models generate text? Generating high-dimensional semantic representations has long been a pursuit for visual generation, yet discrete methods, the paradigm s...

</details>

<details>
<summary><b>9. MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19231) • [📄 arXiv](https://arxiv.org/abs/2603.19231) • [📥 PDF](https://arxiv.org/pdf/2603.19231)

**💻 Code:** [⭐ Code](https://github.com/Quest4Science/MonoArt)

> Project Page: https://lihaitian.com/MonoArt/ GitHub: https://github.com/Quest4Science/MonoArt

</details>

<details>
<summary><b>10. LVOmniBench: Pioneering Long Audio-Video Understanding Evaluation for Omnimodal LLMs</b> ⭐ 31</summary>

<br/>

**👥 Authors:** Kele Shao, Wenjie Du, Jia Xu, Yuhua Zheng, Keda Tao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19217) • [📄 arXiv](https://arxiv.org/abs/2603.19217) • [📥 PDF](https://arxiv.org/pdf/2603.19217)

**💻 Code:** [⭐ Code](https://github.com/KD-TAO/LVOmniBench)

> LVOmniBench is a new audio-visual understanding evaluation benchmark in long-form audio-video inputs.

</details>

<details>
<summary><b>11. F2LLM-v2: Inclusive, Performant, and Efficient Embeddings for a Multilingual World</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19223) • [📄 arXiv](https://arxiv.org/abs/2603.19223) • [📥 PDF](https://arxiv.org/pdf/2603.19223)

> F2LLM-v2 is a family of general-purpose, multilingual embedding models in 8 distinct sizes ranging from 80M to 14B. Trained on a curated composite of 60 million publicly available high-quality data, F2LLM-v2 supports more than 200 languages, with ...

</details>

<details>
<summary><b>12. ReactMotion: Generating Reactive Listener Motions from Speaker Utterance</b> ⭐ 107</summary>

<br/>

**👥 Authors:** Ruibin Bai, Jianfeng Ren, Bing Li, Bizhu Wu, Cheng Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15083) • [📄 arXiv](https://arxiv.org/abs/2603.15083) • [📥 PDF](https://arxiv.org/pdf/2603.15083)

**💻 Code:** [⭐ Code](https://github.com/awakening-ai/ReactMotion)

> We present ReactMotionNet, a large-scale dataset that pairs speaker utterances with multiple candidate listener motions annotated with varying degrees of appropriateness. This dataset design explicitly captures the one-to-many nature of listener b...

</details>

<details>
<summary><b>13. AndroTMem: From Interaction Trajectories to Anchored Memory in Long-Horizon GUI Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Biao Wu, Zihao Dongfang, Linghao Zhang, Jungang Li, Yibo Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18429) • [📄 arXiv](https://arxiv.org/abs/2603.18429) • [📥 PDF](https://arxiv.org/pdf/2603.18429)

> We present AndroTMem , a diagnostic framework for anchored memory in long-horizon Android GUI agents, together with AndroTMem-Bench , a benchmark that evaluates memory via TCR on dependency-critical long-horizon tasks. Also, We propose Anchored St...

</details>

<details>
<summary><b>14. Cognitive Mismatch in Multimodal Large Language Models for Discrete Symbol Understanding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junnan Dong, Daixian Liu, Peng Xing, Jiayi Kuang, Yinghui Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18472) • [📄 arXiv](https://arxiv.org/abs/2603.18472) • [📥 PDF](https://arxiv.org/pdf/2603.18472)

> Interesting benchmark on symbolic understanding in multimodal LLMs. One striking result is that models can do better on reasoning than on basic symbol recognition, suggesting they often rely on language priors rather than true visual-symbol ground...

</details>

<details>
<summary><b>15. EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing</b> ⭐ 38</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19224) • [📄 arXiv](https://arxiv.org/abs/2603.19224) • [📥 PDF](https://arxiv.org/pdf/2603.19224)

**💻 Code:** [⭐ Code](https://github.com/FudanCVL/EffectErase)

> No abstract available.

</details>

<details>
<summary><b>16. VTC-Bench: Evaluating Agentic Multimodal Models via Compositional Visual Tool Chaining</b> ⭐ 29</summary>

<br/>

**👥 Authors:** Zhipeng Wu, Yang Shi, Rundong Wang, Yuhao Dong, Xuanyu Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15030) • [📄 arXiv](https://arxiv.org/abs/2603.15030) • [📥 PDF](https://arxiv.org/pdf/2603.15030)

**💻 Code:** [⭐ Code](https://github.com/zhuzil/VTC-Bench)

> Recent advancements extend Multimodal Large Language Models (MLLMs) beyond standard visual question answering to utilizing external tools for advanced visual tasks. Despite this progress, precisely executing and effectively composing diverse tools...

</details>

<details>
<summary><b>17. Tinted Frames: Question Framing Blinds Vision-Language Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19203) • [📄 arXiv](https://arxiv.org/abs/2603.19203) • [📥 PDF](https://arxiv.org/pdf/2603.19203)

**💻 Code:** [⭐ Code](https://github.com/davidhalladay/Tinted-Frames)

> Turns out VLMs aren’t inherently bad at vision, they’re selectively blind. Ask the wrong way, and they won't look at the image!

</details>

<details>
<summary><b>18. SimulU: Training-free Policy for Long-form Simultaneous Speech-to-Speech Translation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16924) • [📄 arXiv](https://arxiv.org/abs/2603.16924) • [📥 PDF](https://arxiv.org/pdf/2603.16924)

> SimulU is a training-free method for long-form simultaneous speech-to-speech translation that uses history management and cross-attention mechanisms to regulate input and output generation effectively.

</details>

<details>
<summary><b>19. Matryoshka Gaussian Splatting</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Jeffrey Hu, Kyle Fogarty, Hakan Aktas, Boqiao Zhang, Zhilin Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19234) • [📄 arXiv](https://arxiv.org/abs/2603.19234) • [📥 PDF](https://arxiv.org/pdf/2603.19234)

**💻 Code:** [⭐ Code](https://github.com/ZhilinGuo/matryoshka-gaussian-splatting)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API NanoGS: Training-Free Gaussian Splat Simplification (2026) Prune Wisely, Re...

</details>

<details>
<summary><b>20. ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18815) • [📄 arXiv](https://arxiv.org/abs/2603.18815) • [📥 PDF](https://arxiv.org/pdf/2603.18815)

**💻 Code:** [⭐ Code](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API MOSAIC: A Unified Platform for Cross-Paradigm Comparison and Evaluation of ...

</details>

<details>
<summary><b>21. Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18002) • [📄 arXiv](https://arxiv.org/abs/2603.18002) • [📥 PDF](https://arxiv.org/pdf/2603.18002)

> Loc3R-VLM enables 3D understanding from video input.  Inspired by human cognition, it builds an internal cognitive map of the global environment while explicitly modeling an agent's position and orientation. By jointly capturing global layout and ...

</details>

<details>
<summary><b>22. MOSS-TTS Technical Report</b> ⭐ 966</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18090) • [📄 arXiv](https://arxiv.org/abs/2603.18090) • [📥 PDF](https://arxiv.org/pdf/2603.18090)

**💻 Code:** [⭐ Code](https://github.com/OpenMOSS/MOSS-TTS)

> This technical report presents MOSS-TTS , a speech generation foundation model built on a scalable recipe: discrete audio tokens + autoregressive modeling + large-scale pretraining. Built on MOSS-Audio-Tokenizer , a causal Transformer tokenizer th...

</details>

<details>
<summary><b>23. MHPO: Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16929) • [📄 arXiv](https://arxiv.org/abs/2603.16929) • [📥 PDF](https://arxiv.org/pdf/2603.16929)

> Regulating the importance ratio is critical for the training stability of Group Relative Policy Optimization (GRPO) based frameworks. However, prevailing ratio control methods, such as hard clipping, suffer from non-differentiable boundaries and v...

</details>

<details>
<summary><b>24. OSM-based Domain Adaptation for Remote Sensing VLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11804) • [📄 arXiv](https://arxiv.org/abs/2603.11804) • [📥 PDF](https://arxiv.org/pdf/2603.11804)

> OSMDA-VLM offers a clever way to scale remote sensing models without the massive headache of manual labeling or the high cost of GPT-4V distillation. By using OpenStreetMap data, the model basically acts as its own annotation engine, leveraging it...

</details>

<details>
<summary><b>25. Reasoning over mathematical objects: on-policy reward modeling and test time aggregation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jack Lanchantin, Ilia Kulikov, Seungone Kim, Marjan Ghazvininejad, Pranjal Aggarwal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18886) • [📄 arXiv](https://arxiv.org/abs/2603.18886) • [📥 PDF](https://arxiv.org/pdf/2603.18886)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Characterizing, Evaluating, and Optimizing Complex Reasoning (2026) KARL: K...

</details>

<details>
<summary><b>26. Prompt-Free Universal Region Proposal Network</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Qi Fan, Wenbin Li, Shaofeng Zhang, Changhan Liu, Qihong Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.17554) • [📄 arXiv](https://arxiv.org/abs/2603.17554) • [📥 PDF](https://arxiv.org/pdf/2603.17554)

**💻 Code:** [⭐ Code](https://github.com/tangqh03/PF-RPN)

> Accepted to CVPR 2026.

</details>

<details>
<summary><b>27. What Really Controls Temporal Reasoning in Large Language Models: Tokenisation or Representation of Time?</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Wei Zhao, Maxime Peyrard, Ahmad Muhammad Isa, Gagan Bhatia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19017) • [📄 arXiv](https://arxiv.org/abs/2603.19017) • [📥 PDF](https://arxiv.org/pdf/2603.19017)

**💻 Code:** [⭐ Code](https://github.com/gagan3012/mtb)

> We present MULTITEMPBENCH, a multilingual temporal reasoning benchmark spanning three tasks, date arithmetic, time zone conversion, and temporal relation extraction across five languages (English, German, Chinese, Arabic, and Hausa) and multiple c...

</details>

<details>
<summary><b>28. PARSA-Bench: A Comprehensive Persian Audio-Language Model Benchmark</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Azadeh Shakery, Heshaam Faili, Parmis Bathayan, Mohammad Amini, MohammadJRanjbar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.14456) • [📄 arXiv](https://arxiv.org/abs/2603.14456) • [📥 PDF](https://arxiv.org/pdf/2603.14456)

> We introduce PARSA-Bench, the first benchmark for evaluating Large Audio-Language Models (LALMs) on Persian language and culture — 16 tasks, 8,000+ samples, across speech understanding, paralinguistic analysis, and Persian cultural audio understan...

</details>

<details>
<summary><b>29. VID-AD: A Dataset for Image-Level Logical Anomaly Detection under Vision-Induced Distraction</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Chunzhi Gu, Shun Maeda, Shunsuke Sakai, Yawen Zou, nkthiroto

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13964) • [📄 arXiv](https://arxiv.org/abs/2603.13964) • [📥 PDF](https://arxiv.org/pdf/2603.13964)

**💻 Code:** [⭐ Code](https://github.com/nkthiroto/VID-AD)

> This paper introduces VID-AD (Vision-Induced Distraction Anomaly Dataset), a new benchmark for image-level logical anomaly detection. Unlike existing datasets, VID-AD specifically evaluates robustness against visual distractions (background clutte...

</details>

<details>
<summary><b>30. Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Lei Li, Yatish Hosmane Revanasiddappa, Siqi Ouyang, lyf07

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13045) • [📄 arXiv](https://arxiv.org/abs/2603.13045) • [📥 PDF](https://arxiv.org/pdf/2603.13045)

**💻 Code:** [⭐ Code](https://github.com/LeiLiLab/WALAR)

> Large Language Models (LLMs) have demonstrated remarkable capability in machine translation on high-resource language pairs, yet their performance on low-resource translation still lags behind. Existing post-training methods rely heavily on high-q...

</details>

<details>
<summary><b>31. COT-FM: Cluster-wise Optimal Transport Flow Matching</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tsung-Wei Ke, Cheng-Fu Chou, Jia-Wei Liao, Kuan-Hsun Tu, Chiensheng Chiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13395) • [📄 arXiv](https://arxiv.org/abs/2603.13395) • [📥 PDF](https://arxiv.org/pdf/2603.13395)

> COT-FM

</details>

<details>
<summary><b>32. DreamPartGen: Semantically Grounded Part-Level 3D Generation via Collaborative Latent Denoising</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.19216) • [📄 arXiv](https://arxiv.org/abs/2603.19216) • [📥 PDF](https://arxiv.org/pdf/2603.19216)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 32 |
| 📅 Today | [`2026-03-22.json`](data/daily/2026-03-22.json) | 32 |
| 📆 This Week | [`2026-W11.json`](data/weekly/2026-W11.json) | 133 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 481 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-22 | 32 | [View JSON](data/daily/2026-03-22.json) |
| 📄 2026-03-21 | 32 | [View JSON](data/daily/2026-03-21.json) |
| 📄 2026-03-20 | 2 | [View JSON](data/daily/2026-03-20.json) |
| 📄 2026-03-19 | 52 | [View JSON](data/daily/2026-03-19.json) |
| 📄 2026-03-18 | 8 | [View JSON](data/daily/2026-03-18.json) |
| 📄 2026-03-17 | 1 | [View JSON](data/daily/2026-03-17.json) |
| 📄 2026-03-16 | 6 | [View JSON](data/daily/2026-03-16.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |
| 📅 2026-W10 | 119 | [View JSON](data/weekly/2026-W10.json) |
| 📅 2026-W09 | 201 | [View JSON](data/weekly/2026-W09.json) |
| 📅 2026-W08 | 184 | [View JSON](data/weekly/2026-W08.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 481 | [View JSON](data/monthly/2026-03.json) |
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
