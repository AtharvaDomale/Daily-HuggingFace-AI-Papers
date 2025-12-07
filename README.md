<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-38-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-138+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">38</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">187</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">187</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">138+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 07, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shifeng Zhang, Fangtai Wu, Hailong Guo, Yubo Huang, jamesliu1217

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04677) • [📄 arXiv](https://arxiv.org/abs/2512.04677) • [📥 PDF](https://arxiv.org/pdf/2512.04677)

**💻 Code:** [⭐ Code](https://github.com/Alibaba-Quark/LiveAvatar)

> No abstract available.

</details>

<details>
<summary><b>2. DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04324) • [📄 arXiv](https://arxiv.org/abs/2512.04324) • [📥 PDF](https://arxiv.org/pdf/2512.04324)

> Real-world enterprise data intelligence workflows encompass data engineering that turns raw sources into analytical-ready tables and data analysis that convert those tables into decision-oriented insights. We introduce DAComp, a benchmark of 210 t...

</details>

<details>
<summary><b>3. Nex-N1: Agentic Models Trained via a Unified Ecosystem for Large-Scale Environment Construction</b> ⭐ 69</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04987) • [📄 arXiv](https://arxiv.org/abs/2512.04987) • [📥 PDF](https://arxiv.org/pdf/2512.04987)

**💻 Code:** [⭐ Code](https://github.com/nex-agi/Nex-N1)

> The evolution of Large Language Models (LLMs) from passive responders to autonomous agents necessitates a fundamental shift in learning paradigms -- from static imitation to incentive-driven decision making. However, this transition is significant...

</details>

<details>
<summary><b>4. ARM-Thinker: Reinforcing Multimodal Generative Reward Models with Agentic Tool Use and Visual Reasoning</b> ⭐ 40</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05111) • [📄 arXiv](https://arxiv.org/abs/2512.05111) • [📥 PDF](https://arxiv.org/pdf/2512.05111)

**💻 Code:** [⭐ Code](https://github.com/InternLM/ARM-Thinker) • [⭐ Code](https://github.com/open-compass/VLMEvalKit/pull/1334)

> 🏠 Github Repo: https://github.com/InternLM/ARM-Thinker ⭐️ For Agent Evaluation: https://github.com/open-compass/VLMEvalKit/pull/1334 (We added a new feature to VLMEvalKit that supports evaluating diverse models within the ARM-Thinker agent flow, i...

</details>

<details>
<summary><b>5. Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation</b> ⭐ 101</summary>

<br/>

**👥 Authors:** Hao Ouyang, Haobo Li, Yanhong Zeng, Yunhong Lu, qiuyuu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04678) • [📄 arXiv](https://arxiv.org/abs/2512.04678) • [📥 PDF](https://arxiv.org/pdf/2512.04678)

**💻 Code:** [⭐ Code](https://github.com/JaydenLyh/Reward-Forcing)

> homepage: https://reward-forcing.github.io/

</details>

<details>
<summary><b>6. Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion</b> ⭐ 116</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04926) • [📄 arXiv](https://arxiv.org/abs/2512.04926) • [📥 PDF](https://arxiv.org/pdf/2512.04926)

**💻 Code:** [⭐ Code](https://github.com/yuemingPAN/SFD)

> Why denoise synchronously? SFD introduces an asynchronous paradigm where Semantics lead Texture. Time to rethink the generation ordering in LDMs! Paper: https://arxiv.org/abs/2512.04926 Github: https://github.com/yuemingPAN/SFD

</details>

<details>
<summary><b>7. PaperDebugger: A Plugin-Based Multi-Agent System for In-Editor Academic Writing, Review, and Editing</b> ⭐ 350</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.02589) • [📄 arXiv](https://arxiv.org/abs/2512.02589) • [📥 PDF](https://arxiv.org/pdf/2512.02589)

**💻 Code:** [⭐ Code](https://github.com/PaperDebugger/PaperDebugger)

> PaperDebugger lives inside Overleaf and rewrites your paper with you in real time! Paper: https://arxiv.org/abs/2512.02589 Github: https://github.com/PaperDebugger/PaperDebugger Enhancer Model XtraGPT: https://huggingface.co/Xtra-Computing/XtraGPT-7B

</details>

<details>
<summary><b>8. 4DLangVGGT: 4D Language-Visual Geometry Grounded Transformer</b> ⭐ 33</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05060) • [📄 arXiv](https://arxiv.org/abs/2512.05060) • [📥 PDF](https://arxiv.org/pdf/2512.05060)

**💻 Code:** [⭐ Code](https://github.com/hustvl/4DLangVGGT)

> code: https://github.com/hustvl/4DLangVGGT webpage: https://hustvl.github.io/4DLangVGGT/

</details>

<details>
<summary><b>9. DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling</b> ⭐ 40</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03000) • [📄 arXiv](https://arxiv.org/abs/2512.03000) • [📥 PDF](https://arxiv.org/pdf/2512.03000)

**💻 Code:** [⭐ Code](https://github.com/Dynamics-X/DynamicVerse)

> Understanding the dynamic physical world, characterized by its evolving 3D structure, real-world motion, and semantic content with textual descriptions, is crucial for human-agent interaction and enables embodied agents to perceive and act within ...

</details>

<details>
<summary><b>10. Splannequin: Freezing Monocular Mannequin-Challenge Footage with Dual-Detection Splatting</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu-Lun Liu, Wei-Lun Chao, Chung-Ho Wu, Yi-Chuan Huang, chien90190

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05113) • [📄 arXiv](https://arxiv.org/abs/2512.05113) • [📥 PDF](https://arxiv.org/pdf/2512.05113)

> Splannequin transforms imperfect Mannequin-Challenge videos into completely frozen videos.

</details>

<details>
<summary><b>11. UltraImage: Rethinking Resolution Extrapolation in Image Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04504) • [📄 arXiv](https://arxiv.org/abs/2512.04504) • [📥 PDF](https://arxiv.org/pdf/2512.04504)

> Project page is available at https://thu-ml.github.io/ultraimage.github.io/ .

</details>

<details>
<summary><b>12. NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vitor Guizilini, Vishal M. Patel, Mingyuan Zhou, Charles Ochoa, Yu Zeng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05106) • [📄 arXiv](https://arxiv.org/abs/2512.05106) • [📥 PDF](https://arxiv.org/pdf/2512.05106)

> No abstract available.

</details>

<details>
<summary><b>13. SIMA 2: A Generalist Embodied Agent for Virtual Worlds</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04797) • [📄 arXiv](https://arxiv.org/abs/2512.04797) • [📥 PDF](https://arxiv.org/pdf/2512.04797)

> We introduce SIMA 2, a generalist embodied agent that understands and acts in a wide variety of 3D virtual worlds. Built upon a Gemini foundation model, SIMA 2 represents a significant step toward active, goal-directed interaction within an embodi...

</details>

<details>
<summary><b>14. TV2TV: A Unified Framework for Interleaved Language and Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05103) • [📄 arXiv](https://arxiv.org/abs/2512.05103) • [📥 PDF](https://arxiv.org/pdf/2512.05103)

> Video generation models are rapidly advancing, but can still struggle with complex video outputs that require significant semantic branching or repeated high-level reasoning about what should happen next. In this paper, we introduce a new class of...

</details>

<details>
<summary><b>15. Model-Based and Sample-Efficient AI-Assisted Math Discovery in Sphere Packing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jun Wang, Xihan Li, Antoine Grosnit, Rasul Tutunov, alexmaraval

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04829) • [📄 arXiv](https://arxiv.org/abs/2512.04829) • [📥 PDF](https://arxiv.org/pdf/2512.04829)

> AlphaEvolve is amazing and tackles hard-to-solve and easy-to-evaluate problems! But, many math problems are actually hard-to-solve and hard-to-evaluate! Here, we can't do much trial and error; we need something more efficient - because trying one ...

</details>

<details>
<summary><b>16. Reflection Removal through Efficient Adaptation of Diffusion Transformers</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05000) • [📄 arXiv](https://arxiv.org/abs/2512.05000) • [📥 PDF](https://arxiv.org/pdf/2512.05000)

**💻 Code:** [⭐ Code](https://github.com/huawei-bayerlab/windowseat-reflection-removal)

> Project page: https://huggingface.co/spaces/huawei-bayerlab/windowseat-reflection-removal-web

</details>

<details>
<summary><b>17. On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement Death Spiral</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Christos Thrampoulidis, Yi Ren, Boying Gong, Yushu Li, Wenlong Deng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04220) • [📄 arXiv](https://arxiv.org/abs/2512.04220) • [📥 PDF](https://arxiv.org/pdf/2512.04220)

> Tool-integrated (TI) reinforcement learning (RL) enables large language models (LLMs) to perform multi-step reasoning by interacting with external tools such as search engines and retrievers. Group Relative Policy Optimization (GRPO), exemplified ...

</details>

<details>
<summary><b>18. DraCo: Draft as CoT for Text-to-Image Preview and Rare Concept Generation</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Ziyu Guo, Zhuofan Zong, Renrui Zhang, mickyhimself, CaraJ

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05112) • [📄 arXiv](https://arxiv.org/abs/2512.05112) • [📥 PDF](https://arxiv.org/pdf/2512.05112)

**💻 Code:** [⭐ Code](https://github.com/CaraJ7/DraCo)

> 🔥 Project Page: https://github.com/CaraJ7/DraCo

</details>

<details>
<summary><b>19. SignRoundV2: Closing the Performance Gap in Extremely Low-Bit Post-Training Quantization for LLMs</b> ⭐ 747</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04746) • [📄 arXiv](https://arxiv.org/abs/2512.04746) • [📥 PDF](https://arxiv.org/pdf/2512.04746)

**💻 Code:** [⭐ Code](https://github.com/intel/auto-round)

> Extremely low-bit quantization for LLMs. Check out https://github.com/intel/auto-round

</details>

<details>
<summary><b>20. Generative Neural Video Compression via Video Diffusion Prior</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05016) • [📄 arXiv](https://arxiv.org/abs/2512.05016) • [📥 PDF](https://arxiv.org/pdf/2512.05016)

> No abstract available.

</details>

<details>
<summary><b>21. Mitigating Object and Action Hallucinations in Multimodal LLMs via Self-Augmented Contrastive Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04356) • [📄 arXiv](https://arxiv.org/abs/2512.04356) • [📥 PDF](https://arxiv.org/pdf/2512.04356)

> Project page: https://kpc0810.github.io/santa/

</details>

<details>
<summary><b>22. LATTICE: Democratize High-Fidelity 3D Generation at Scale</b> ⭐ 153</summary>

<br/>

**👥 Authors:** Qingxiang Lin, Haolin Liu, Zibo Zhao, Yunfei Zhao, Zeqiang Lai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03052) • [📄 arXiv](https://arxiv.org/abs/2512.03052) • [📥 PDF](https://arxiv.org/pdf/2512.03052)

**💻 Code:** [⭐ Code](https://github.com/Zeqiang-Lai/LATTICE)

> 3D Shape foundation model

</details>

<details>
<summary><b>23. Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Heeji Yoon, Jisu Nam, Paul Hyunbin Cho, Wooseok Jang, YJ-142150

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05081) • [📄 arXiv](https://arxiv.org/abs/2512.05081) • [📥 PDF](https://arxiv.org/pdf/2512.05081)

> Project Page: https://cvlab-kaist.github.io/DeepForcing/

</details>

<details>
<summary><b>24. Aligned but Stereotypical? The Hidden Influence of System Prompts on Social Bias in LVLM-Based Text-to-Image Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04981) • [📄 arXiv](https://arxiv.org/abs/2512.04981) • [📥 PDF](https://arxiv.org/pdf/2512.04981)

**💻 Code:** [⭐ Code](https://github.com/nahyeonkaty/fairpro)

> We introduce: 1️⃣ A 1,024-prompt benchmark across 4 linguistic complexity levels 2️⃣ Fine-grained, systematic demographic (gender, age, ethnicity, physical appearance) bias diagnostics 3️⃣ FairPRO, a training-free meta-prompting framework that ena...

</details>

<details>
<summary><b>25. SeeNav-Agent: Enhancing Vision-Language Navigation with Visual Prompt and Step-Level Policy Optimization</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.02631) • [📄 arXiv](https://arxiv.org/abs/2512.02631) • [📥 PDF](https://arxiv.org/pdf/2512.02631)

**💻 Code:** [⭐ Code](https://github.com/WzcTHU/SeeNav-Agent)

> 🎯 Code: https://github.com/WzcTHU/SeeNav-Agent 🤗 Model: https://huggingface.co/wangzc9865/SeeNav-Agent

</details>

<details>
<summary><b>26. Some Modalities are More Equal Than Others: Decoding and Architecting Multimodal Integration in MLLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Deepti Ghadiyaram, Xavier Thomas, Arjun Reddy Akula, Chaitanya Chakka, Tianle Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2511.22826) • [📄 arXiv](https://arxiv.org/abs/2511.22826) • [📥 PDF](https://arxiv.org/pdf/2511.22826)

> Some Modalities are More Equal Than Others: Decoding and Architecting Multimodal Integration As the famous George Orwell quote goes - "all animals are equal but some animals are more equal than others", we indeed find that though present-day MLLMs...

</details>

<details>
<summary><b>27. BulletTime: Decoupled Control of Time and Camera Pose for Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jan Ackermann, Tong Wu, Shengqu Cai, Qihang Zhang, Yiming Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05076) • [📄 arXiv](https://arxiv.org/abs/2512.05076) • [📥 PDF](https://arxiv.org/pdf/2512.05076)

> No abstract available.

</details>

<details>
<summary><b>28. FMA-Net++: Motion- and Exposure-Aware Real-World Joint Video Super-Resolution and Deblurring</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Munchurl Kim, Jihyong Oh, Geunhyuk Youk

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04390) • [📄 arXiv](https://arxiv.org/abs/2512.04390) • [📥 PDF](https://arxiv.org/pdf/2512.04390)

**💻 Code:** [⭐ Code](https://github.com/KAIST-VICLab/FMA-Net-PlusPlus)

> Project page: https://kaist-viclab.github.io/fmanetpp_site/

</details>

<details>
<summary><b>29. EgoLCD: Egocentric Video Generation with Long Context Diffusion</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04515) • [📄 arXiv](https://arxiv.org/abs/2512.04515) • [📥 PDF](https://arxiv.org/pdf/2512.04515)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/EgoLCD)

> Generating long, coherent egocentric videos is difficult, as hand-object interactions and procedural tasks require reliable long-term memory. Existing autoregressive models suffer from content drift, where object identity and scene semantics degra...

</details>

<details>
<summary><b>30. Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nikolaos Aletras, Aline Villavicencio, Terufumi Morishita, atsuki-yamaguchi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04844) • [📄 arXiv](https://arxiv.org/abs/2512.04844) • [📥 PDF](https://arxiv.org/pdf/2512.04844)

**💻 Code:** [⭐ Code](https://github.com/gucci-j/ssu)

> Our code and a step-by-step guide for preprocessing, training, evaluation, and analysis for both our proposed method (SSU) and all baselines are available on GitHub: https://github.com/gucci-j/ssu .

</details>

<details>
<summary><b>31. Generative Action Tell-Tales: Assessing Human Motion in Synthesized Videos</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.01803) • [📄 arXiv](https://arxiv.org/abs/2512.01803) • [📥 PDF](https://arxiv.org/pdf/2512.01803)

> Project webpage: https://xthomasbu.github.io/video-gen-evals/ Dataset: https://huggingface.co/datasets/dghadiya/TAG-Bench-Video

</details>

<details>
<summary><b>32. ShadowDraw: From Any Object to Shadow-Drawing Compositional Art</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05110) • [📄 arXiv](https://arxiv.org/abs/2512.05110) • [📥 PDF](https://arxiv.org/pdf/2512.05110)

> No abstract available.

</details>

<details>
<summary><b>33. QKAN-LSTM: Quantum-inspired Kolmogorov-Arnold Long Short-term Memory</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nan-Yow Chen, Kuo-Chung Peng, Chun-Hua Lin, Yu-Chao Hsu, Jim137

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05049) • [📄 arXiv](https://arxiv.org/abs/2512.05049) • [📥 PDF](https://arxiv.org/pdf/2512.05049)

> A follow-up to our earlier QKAN research, this work explores how quantum-inspired activations can enhance classical LSTM models. With single-qubit DARUAN modules and QKAN-based gating, QKAN-LSTM cuts parameters by up to 79% while improving perform...

</details>

<details>
<summary><b>34. GaussianBlender: Instant Stylization of 3D Gaussians with Disentangled Latent Spaces</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sezer Karaoglu, Ngo Anh Vien, Yue Li, Xiaoyan Xing, melisocal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03683) • [📄 arXiv](https://arxiv.org/abs/2512.03683) • [📥 PDF](https://arxiv.org/pdf/2512.03683)

> Project page: https://gaussianblender.github.io/

</details>

<details>
<summary><b>35. Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Radu Marculescu, Mustafa Munir, Xiwen Wei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03125) • [📄 arXiv](https://arxiv.org/abs/2512.03125) • [📥 PDF](https://arxiv.org/pdf/2512.03125)

**💻 Code:** [⭐ Code](https://github.com/Christina200/MoDE-official)

> Unified Multimodal Generative Models (UMGMs) unify visual understanding and image generation within a single autoregressive framework. However, their ability to continually learn new tasks is severely hindered by catastrophic forgetting, both with...

</details>

<details>
<summary><b>36. When AI Takes the Couch: Psychometric Jailbreaks Reveal Internal Conflict in Frontier Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Gilbert Fridgen, Igor Tchappi, Amir Sartipi, Hanna Marxen, akhadangi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.04124) • [📄 arXiv](https://arxiv.org/abs/2512.04124) • [📥 PDF](https://arxiv.org/pdf/2512.04124)

> Frontier large language models (LLMs) such as ChatGPT, Grok and Gemini are increasingly used for mental health support with anxiety, trauma and self-worth. Most work treats them as tools or as targets of personality tests, assuming they merely sim...

</details>

<details>
<summary><b>37. REFLEX: Self-Refining Explainable Fact-Checking via Disentangling Truth into Style and Substance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yaxin Fan, Hongzhan Lin, Jing Ma, Gao Wei, Chuyi Kong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2511.20233) • [📄 arXiv](https://arxiv.org/abs/2511.20233) • [📥 PDF](https://arxiv.org/pdf/2511.20233)

> The prevalence of misinformation on social media threatens public trust, demanding automated fact-checking systems that provide accurate verdicts with interpretable explanations. However, existing large language model-based (LLM-based) approaches ...

</details>

<details>
<summary><b>38. A Theoretical Framework for Auxiliary-Loss-Free Load Balancing of Sparse Mixture-of-Experts in Large-Scale AI Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03915) • [📄 arXiv](https://arxiv.org/abs/2512.03915) • [📥 PDF](https://arxiv.org/pdf/2512.03915)

> In large-scale AI training, Sparse Mixture-of-Experts (s-MoE) layers enable scaling by activating only a small subset of experts per token. An operational challenge in this design is load balancing: routing tokens to minimize the number of idle ex...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 38 |
| 📅 Today | [`2025-12-07.json`](data/daily/2025-12-07.json) | 38 |
| 📆 This Week | [`2025-W48.json`](data/weekly/2025-W48.json) | 187 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 187 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-07 | 38 | [View JSON](data/daily/2025-12-07.json) |
| 📄 2025-12-06 | 38 | [View JSON](data/daily/2025-12-06.json) |
| 📄 2025-12-05 | 38 | [View JSON](data/daily/2025-12-05.json) |
| 📄 2025-12-04 | 24 | [View JSON](data/daily/2025-12-04.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 187 | [View JSON](data/monthly/2025-12.json) |

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
