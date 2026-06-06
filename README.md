<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-50-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4531+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">50</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">128</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">128</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4531+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 06, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06492) • [📄 arXiv](https://arxiv.org/abs/2606.06492) • [📥 PDF](https://arxiv.org/pdf/2606.06492)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Repository context is the bottleneck for code LLMs: every completion needs to know the project's imports, APIs, and conventions, and today we pay for that context at every single query -  through RAG, dependency analysis, or ever-longer prompts - ...

</details>

<details>
<summary><b>2. ArcANE: Do Role-Playing Language Agents Stay in Character at the Right Time?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05553) • [📄 arXiv](https://arxiv.org/abs/2606.05553) • [📥 PDF](https://arxiv.org/pdf/2606.05553)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Role-playing language agents (RPLAs) should play characters whose values and behavior evolve as the story progresses, not maintain a fixed persona. Existing benchmarks measure factual recall at a given chapter, not whether responses align with the...

</details>

<details>
<summary><b>3. TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.04743) • [📄 arXiv](https://arxiv.org/abs/2606.04743) • [📥 PDF](https://arxiv.org/pdf/2606.04743)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Most AI assistants only solve the problems users explicitly ask about, leaving many important issues hidden in the surrounding context. TIDE introduces a template-guided iterative discovery framework that proactively uncovers multiple latent probl...

</details>

<details>
<summary><b>4. VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Tingyu Song, Yang Wang, yilunzhao, Raywithyou, minuzero

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05259) • [📄 arXiv](https://arxiv.org/abs/2606.05259) • [📥 PDF](https://arxiv.org/pdf/2606.05259)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Fu-Fu-Fu-Fu/VideoKR)

> VideoKR presents a large-scale video reasoning dataset and benchmark designed to enhance knowledge-intensive video understanding through expert-domain content and human-in-the-loop example generation.

</details>

<details>
<summary><b>5. AdaPlanBench: Evaluating Adaptive Planning in Large Language Model Agents under World and User Constraints</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05622) • [📄 arXiv](https://arxiv.org/abs/2606.05622) • [📥 PDF](https://arxiv.org/pdf/2606.05622)

**💻 Code:** [⭐ Code](https://github.com/JiayuJeff/AdaPlanBench) • [⭐ Code](https://github.com/huggingface)

> Excited to share AdaPlanBench, a benchmark for studying how LLM agents adaptively re-plan as hidden world constraints and user preferences emerge.

</details>

<details>
<summary><b>6. Reinforcement Learning Elicits Contextual Learning of Unseen Language Translation</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06428) • [📄 arXiv](https://arxiv.org/abs/2606.06428) • [📥 PDF](https://arxiv.org/pdf/2606.06428)

**💻 Code:** [⭐ Code](https://github.com/hanxuhu/rl-new-language) • [⭐ Code](https://github.com/huggingface)

> In this paper, we propose a reinforcement learning approach to unseen language translation given rich linguistic context, we argue that LLMs can acquire the meta-skill of utilizing context knowledge rather than memorizing specific languages thus h...

</details>

<details>
<summary><b>7. RobotValues: Evaluating Household Robots When Human Values Conflict</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.03312) • [📄 arXiv](https://arxiv.org/abs/2606.03312) • [📥 PDF](https://arxiv.org/pdf/2606.03312)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Seems like a very timely work, nice job!

</details>

<details>
<summary><b>8. LoomVideo: Unifying Multimodal Inputs into Video Generation and Editing</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06042) • [📄 arXiv](https://arxiv.org/abs/2606.06042) • [📥 PDF](https://arxiv.org/pdf/2606.06042)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MSALab-PKU/LoomVideo)

> Developing unified video generation and editing models capable of interpreting interleaved multimodal inputs is a promising yet challenging frontier field. Existing unified frameworks predominantly rely on massive models (typically 13B parameters ...

</details>

<details>
<summary><b>9. Personal AI Agent for Camera Roll VQA</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Yuheng Li, Yong Jae Lee, Donghyun Kim, Krishna Kumar Singh, Thao Nguyen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05275) • [📄 arXiv](https://arxiv.org/abs/2606.05275) • [📥 PDF](https://arxiv.org/pdf/2606.05275)

**💻 Code:** [⭐ Code](https://github.com/thaoshibe/camroll) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/thaoshibe/camroll/)

> "if an AI could see your whole camera roll, what would you ask?" . project page: https://thaoshibe.github.io/camroll/ . github: https://github.com/thaoshibe/camroll/ We study the personal camera roll visual question answering setting. In this sett...

</details>

<details>
<summary><b>10. Complexity-Balanced Diffusion Splitting</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06477) • [📄 arXiv](https://arxiv.org/abs/2606.06477) • [📥 PDF](https://arxiv.org/pdf/2606.06477)

**💻 Code:** [⭐ Code](https://github.com/NoamIssachar/Complexity-Balanced-Splitting) • [⭐ Code](https://github.com/huggingface)

> Complexity-Balanced Splitting (CBS) is a principled framework for temporal capacity allocation in diffusion models. By mathematically partitioning the diffusion timeline based on local approximation burden (e.g., path acceleration), CBS significan...

</details>

<details>
<summary><b>11. Rethinking Continual Experience Internalization for Self-Evolving LLM Agents</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Chenxing Sun, Wenbo Nie, Shengda Fan, Keven16, cjw259wen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.04703) • [📄 arXiv](https://arxiv.org/abs/2606.04703) • [📥 PDF](https://arxiv.org/pdf/2606.04703)

**💻 Code:** [⭐ Code](https://github.com/RUCBM/ExpInternalization) • [⭐ Code](https://github.com/huggingface)

> Rethinking Continual Experience Internalization for Self-Evolving LLM Agents

</details>

<details>
<summary><b>12. The Road Ahead in Autonomous Driving: The KITScenes Multimodal Dataset</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.02956) • [📄 arXiv](https://arxiv.org/abs/2606.02956) • [📥 PDF](https://arxiv.org/pdf/2606.02956)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present KITScenes Multimodal, a European dataset built around a high-fidelity sensor suite together with the most complete HD maps for autonomous driving ever released. Highlights European urban focus — recordings from Karlsruhe, Frankfurt, and...

</details>

<details>
<summary><b>13. Dream.exe: Can Video Generation Models Dream Executable Robot Manipulation?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ziqi Wang, Siyang Chen, Jifeng Zhu, Kaiming Yang, Rui Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.04811) • [📄 arXiv](https://arxiv.org/abs/2606.04811) • [📥 PDF](https://arxiv.org/pdf/2606.04811)

**💻 Code:** [⭐ Code](https://github.com/showlab/Dream.exe) • [⭐ Code](https://github.com/huggingface)

> Repo: https://github.com/showlab/Dream.exe

</details>

<details>
<summary><b>14. Unsupervised Skill Discovery for Agentic Data Analysis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06416) • [📄 arXiv](https://arxiv.org/abs/2606.06416) • [📥 PDF](https://arxiv.org/pdf/2606.06416)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Inference-time skill augmentation provides a lightweight way to improve data-analytic agents by injecting reusable procedural knowledge without updating model parameters. However, discovering effective skills for data analysis remains challenging,...

</details>

<details>
<summary><b>15. LLMs Can Leak Training Data But Do They Want To? A Propensity-Aware Evaluation of Memorization in LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06286) • [📄 arXiv](https://arxiv.org/abs/2606.06286) • [📥 PDF](https://arxiv.org/pdf/2606.06286)

**💻 Code:** [⭐ Code](https://github.com/N-essuno/PropMe) • [⭐ Code](https://github.com/huggingface)

> We introduced PropMe and SimpleTrace to prove that while AI can be forced to leak training data under attack, its natural propensity to do so during everyday use is remarkably low. We also found that continuous training helps models naturally dilu...

</details>

<details>
<summary><b>16. OPRD: On-Policy Representation Distillation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mingxuan Xia, Haobo Wang, Bowen Song, Guangcheng Zhu, Shenzhi Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06021) • [📄 arXiv](https://arxiv.org/abs/2606.06021) • [📥 PDF](https://arxiv.org/pdf/2606.06021)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ShenzhiYang2000/OPRD)

> Due to the company's open-source process, the code will be released within the next week~

</details>

<details>
<summary><b>17. MAOAM: Unified Object and Material Selection with Vision-Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.04880) • [📄 arXiv](https://arxiv.org/abs/2606.04880) • [📥 PDF](https://arxiv.org/pdf/2606.04880)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/adobe-research/obj-and-mat-selection)

> In this work, we present MAOAM, a unified selection framework that enables precise object and material-level selection across both text- and click-based interactions. A key challenge is the lack of material selection datasets with text annotations...

</details>

<details>
<summary><b>18. The Shadow Price of Reasoning: Economic Perspective on Optimal Budget Allocation for LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.03092) • [📄 arXiv](https://arxiv.org/abs/2606.03092) • [📥 PDF](https://arxiv.org/pdf/2606.03092)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/waunx/CLEAR)

> Inference-time scaling has emerged as a critical avenue for enhancing Large Language Models' performance, yet real-world deployment is constrained by strict computational budgets. In this work, we formulate inference budget allocation as a global ...

</details>

<details>
<summary><b>19. World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Yanzhe Hu, Yiyang Chen, Siqi Kou, Zhihong Liu, Yi Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05979) • [📄 arXiv](https://arxiv.org/abs/2606.05979) • [📥 PDF](https://arxiv.org/pdf/2606.05979)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SJTU-DENG-Lab/WLA)

> No abstract available.

</details>

<details>
<summary><b>20. Imagine Before You Predict: Interleaved Latent Visual Reasoning for Video Event Prediction</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05769) • [📄 arXiv](https://arxiv.org/abs/2606.05769) • [📥 PDF](https://arxiv.org/pdf/2606.05769)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenGVLab/Future-L1)

> Future-L1 nails video event prediction by letting MLLMs interleave text reasoning with latent visual "imagination" of future frames.

</details>

<details>
<summary><b>21. Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30159) • [📄 arXiv](https://arxiv.org/abs/2605.30159) • [📥 PDF](https://arxiv.org/pdf/2605.30159)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents

</details>

<details>
<summary><b>22. Towards One-to-Many Temporal Grounding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06294) • [📄 arXiv](https://arxiv.org/abs/2606.06294) • [📥 PDF](https://arxiv.org/pdf/2606.06294)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Temporal Grounding (TG) aims to localize video segments corresponding to a textual query. Prior research predominantly focuses on single-segment retrieval. Real-world scenarios, however, often require localizing multiple disjoint segments for a si...

</details>

<details>
<summary><b>23. MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery</b> ⭐ 301</summary>

<br/>

**👥 Authors:** Shiyang Feng, Zongsheng Cao, Jinxin Shi, Xiangchao Yan, Shangheng Du

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06473) • [📄 arXiv](https://arxiv.org/abs/2606.06473) • [📥 PDF](https://arxiv.org/pdf/2606.06473)

**💻 Code:** [⭐ Code](https://github.com/InternScience/MLEvolve) • [⭐ Code](https://github.com/huggingface)

> MLEvolve is an LLM-based multi-agent framework for automated machine learning algorithm discovery, featuring Progressive Monte Carlo Graph Search and retrospective memory to enhance long-horizon optimization performance.

</details>

<details>
<summary><b>24. SEAOTTER: Sensor Embedded Autoencoding with One-Time Transcode for Efficient Reconstruction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Neeraja J. Yadwadkar, danjacobellis

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.03940) • [📄 arXiv](https://arxiv.org/abs/2606.03940) • [📥 PDF](https://arxiv.org/pdf/2606.03940)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/UT-SysML/seaotter)

> https://ut-sysml.github.io/seaotter/

</details>

<details>
<summary><b>25. AdaCodec: A Predictive Visual Code for Video MLLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chenglin Li, Qingyi Si, Zheming Liang, Zhen Huang, Haowen Hou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.02569) • [📄 arXiv](https://arxiv.org/abs/2606.02569) • [📥 PDF](https://arxiv.org/pdf/2606.02569)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Homepage： https://haowenhou.github.io/AdaCodec-Page/

</details>

<details>
<summary><b>26. Learning Geometric Representations from Videos for Spatial Intelligent Multimodal Large Language Models</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Lifu Huang, Haibo Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05833) • [📄 arXiv](https://arxiv.org/abs/2606.05833) • [📥 PDF](https://arxiv.org/pdf/2606.05833)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/WHB139426/GeoVR-MLLM)

> We propose GeoVR, a paradigm to restructure MLLM’s intrinsic representations with geometric awareness using purely 2D videos for Spatial Intelligence.

</details>

<details>
<summary><b>27. AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06155) • [📄 arXiv](https://arxiv.org/abs/2606.06155) • [📥 PDF](https://arxiv.org/pdf/2606.06155)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Skywalker-yqz/AffordanceVLA)

> AffordanceVLA introduces a structured affordance-forecasting bridge for VLA models, enabling robots to reason about what to manipulate, where to interact, and how to act for more robust instruction-following manipulation.

</details>

<details>
<summary><b>28. Towards Truly Multilingual ASR: Generalizing Code-Switching ASR to Unseen Language Pairs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05846) • [📄 arXiv](https://arxiv.org/abs/2606.05846) • [📥 PDF](https://arxiv.org/pdf/2606.05846)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Automatic Speech Recognition (ASR) has become a key technology for human--AI interaction. However, code-switching ASR (CS-ASR) remains particularly challenging due to the severe scarcity of multilingual CS speech resources across diverse language ...

</details>

<details>
<summary><b>29. SePO: Self-Evolving Prompt Agent for System Prompt Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Weng-Fai Wong, Han Wu, taowangcheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.04465) • [📄 arXiv](https://arxiv.org/abs/2606.04465) • [📥 PDF](https://arxiv.org/pdf/2606.04465)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> SePO is a self-evolving system prompt optimization framework that improves a prompt agent by applying the same prompt optimization procedure to the prompt agent itself.

</details>

<details>
<summary><b>30. Flash-WAM: Modality-Aware Distillation for World Action Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yixiao Chen, Lin Zhao, Arash Akbari, Ci Zhang, armanakbari4

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05254) • [📄 arXiv](https://arxiv.org/abs/2606.05254) • [📥 PDF](https://arxiv.org/pdf/2606.05254)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>31. MechVQA: Benchmarking and Enhancing Multimodal LLMs on Comprehensive Mechanical Drawing Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30794) • [📄 arXiv](https://arxiv.org/abs/2605.30794) • [📥 PDF](https://arxiv.org/pdf/2605.30794)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Accepted by ICML2026

</details>

<details>
<summary><b>32. Revising Context, Shifting Simulated Stance: Auditing LLM-Based Stance Simulation in Online Discussions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiebo Luo, Zhongyu Wei, Hanjia Lyu, Wanting Shan, Xinnong Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06443) • [📄 arXiv](https://arxiv.org/abs/2606.06443) • [📥 PDF](https://arxiv.org/pdf/2606.06443)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language models are increasingly used to simulate social media users and infer how individuals may respond to online discussions. In this work, we study counterfactual context revision as a framework for auditing LLM-based stance simulation.

</details>

<details>
<summary><b>33. Latent Reasoning with Normalizing Flows</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haoqiang Kang, Yao Tang, Suhao Yu, Xiangjun Fu, Guancheng Tu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06447) • [📄 arXiv](https://arxiv.org/abs/2606.06447) • [📥 PDF](https://arxiv.org/pdf/2606.06447)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> NF-CoT introduces a latent reasoning framework for LLMs that models intermediate thoughts as continuous states using normalizing flows, preserving autoregressive decoding while improving performance and efficiency.

</details>

<details>
<summary><b>34. Benchmark Everything Everywhere All at Once</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Bokang Yang, Yuang Ai, Peiwen Sun, Dongming Wu, Christinexx

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06462) • [📄 arXiv](https://arxiv.org/abs/2606.06462) • [📥 PDF](https://arxiv.org/pdf/2606.06462)

**💻 Code:** [⭐ Code](https://github.com/Shiyun-x/Benchmark-Agent) • [⭐ Code](https://github.com/huggingface)

> Paper: https://arxiv.org/abs/2606.06462 Code: https://github.com/Shiyun-x/Benchmark-Agent

</details>

<details>
<summary><b>35. EvoDS: Self-Evolving Autonomous Data Science Agent with Skill Learning and Context Management</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.03841) • [📄 arXiv](https://arxiv.org/abs/2606.03841) • [📥 PDF](https://arxiv.org/pdf/2606.03841)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/usail-hkust/EvoDS)

> Accepted by KDD2026

</details>

<details>
<summary><b>36. The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.03645) • [📄 arXiv](https://arxiv.org/abs/2606.03645) • [📥 PDF](https://arxiv.org/pdf/2606.03645)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RL-MIND/Shape-of-Addition)

> We introduce The Shape of Addition , a mechanistic interpretability study of why LLMs can still fail at basic multi-operand addition. By probing residual-stream activations at each generated digit, we find that arithmetic states are organized into...

</details>

<details>
<summary><b>37. Regret Minimization with Adaptive Opponents in Repeated Games</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kaiqing Zhang, Tiancheng Yu, Asuman Ozdaglar, Mingyang Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.06486) • [📄 arXiv](https://arxiv.org/abs/2606.06486) • [📥 PDF](https://arxiv.org/pdf/2606.06486)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> In this paper, we study regret minimization in repeated games with \emph{adaptive} opponents who can respond based on histories of play. The standard metric of \emph{external regret} in online learning is known to fail to capture such adaptivity. ...

</details>

<details>
<summary><b>38. AURA: Intent-Directed Probing for Implicit-Need Surfacing in Situated LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05557) • [📄 arXiv](https://arxiv.org/abs/2606.05557) • [📥 PDF](https://arxiv.org/pdf/2606.05557)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/innovation64/AURA)

> AURA: Intent-Directed Probing for Implicit-Need Surfacing in Situated LLM Agents Situated queries often carry implicit needs beyond their literal wording — asking "where is Lin Wei?" may really be asking "are they free to interrupt?" Standard ReAc...

</details>

<details>
<summary><b>39. Discrete-WAM: Unified Discrete Vision-Action Token Editing for World-Policy Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05645) • [📄 arXiv](https://arxiv.org/abs/2606.05645) • [📥 PDF](https://arxiv.org/pdf/2606.05645)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>40. Video2LoRA: Parametric Video Internalization for Vision-Language Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Dinesh Manocha, Sarvesh Baskar, MananSuri27

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.04351) • [📄 arXiv](https://arxiv.org/abs/2606.04351) • [📥 PDF](https://arxiv.org/pdf/2606.04351)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MananSuri27/video2lora)

> Zero token video understanding for VLMs! We train hypernetworks to metalearn video understanding for VLMs; during inference you can convert a video to a LoRA adapter in a single forward pass, and have efficient video understanding without any visu...

</details>

<details>
<summary><b>41. Absorbing Complexity: An Interaction-Native Knowledge Harness for Financial LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.01886) • [📄 arXiv](https://arxiv.org/abs/2606.01886) • [📥 PDF](https://arxiv.org/pdf/2606.01886)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Financial AI adoption is constrained not by model quality but by cognition friction: users repeatedly restate fragmented context, historical judgments, and risk preferences. Existing financial agents remain turn-based and workflow-disposable. We p...

</details>

<details>
<summary><b>42. Quality-Guided Semi-Supervised Learning for Medical Image Segmentation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.01753) • [📄 arXiv](https://arxiv.org/abs/2606.01753) • [📥 PDF](https://arxiv.org/pdf/2606.01753)

**💻 Code:** [⭐ Code](https://github.com/sfu-mial/QG-SSL) • [⭐ Code](https://github.com/huggingface)

> Introduces the first framework-agnostic method to leverage a learned quality prediction model to guide semi-supervised learning (SSL). Two complementary mechanisms for integrating as drop-in enhancements to existing SSL frameworks: differentiable ...

</details>

<details>
<summary><b>43. LLM Anonymization Against Agentic Re-Identification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30848) • [📄 arXiv](https://arxiv.org/abs/2605.30848) • [📥 PDF](https://arxiv.org/pdf/2605.30848)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/PEACH-Research-Lab/AURA)

> Agentic LLMs with web search change the threat model for text anonymization: weak contextual cues can become cross-referenceable evidence for re-identification, yet those same details also carry downstream analytic value of the text. Existing defe...

</details>

<details>
<summary><b>44. Combinatorial Synthesis: Scaling Code RLVR via Atomic Decomposition and Recombination</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31058) • [📄 arXiv](https://arxiv.org/abs/2605.31058) • [📥 PDF](https://arxiv.org/pdf/2605.31058)

**💻 Code:** [⭐ Code](https://github.com/icip-cas/ADR) • [⭐ Code](https://github.com/huggingface)

> We propose Atomic Decomposition and Recombination (ADR), a novel framework that generates verifiable code tasks via decomposition into atomic elements and controlled recombination, thereby enabling the generation of genuinely novel and challenging...

</details>

<details>
<summary><b>45. Multimodal Music Recommendation System using LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Swetha Mohan, Chandana Magapu, Sreehitha R. Narayana, Srikar Prabhas Kandagatla, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.00125) • [📄 arXiv](https://arxiv.org/abs/2606.00125) • [📥 PDF](https://arxiv.org/pdf/2606.00125)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Dataset: https://zenodo.org/records/20431748

</details>

<details>
<summary><b>46. Trust Region Q Adjoint Matching</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27079) • [📄 arXiv](https://arxiv.org/abs/2605.27079) • [📥 PDF](https://arxiv.org/pdf/2605.27079)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yonghdong/trqam)

> TRQAM internalizes the trust region as a scalar λ inside the flow-policy sampling SDE — an exact Girsanov path-space KL identity (Thm 1) makes the KL budget structurally enforceable via dual descent. 68% vs 46% on 50 OGBench tasks. 👇 blog & code

</details>

<details>
<summary><b>47. Is This Edit Correct? A Multi-Dimensional Benchmark for Reasoning-Aware Image Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05172) • [📄 arXiv](https://arxiv.org/abs/2606.05172) • [📥 PDF](https://arxiv.org/pdf/2606.05172)

**💻 Code:** [⭐ Code](https://github.com/Yixuan-Ding-ZJU/RE-Edit) • [⭐ Code](https://github.com/huggingface)

> Diffusion-based image editing has achieved strong visual fidelity under natural language instructions, yet most existing systems still operate at the level of surface instruction following, without reasoning about the implicit contextual constrain...

</details>

<details>
<summary><b>48. ForeSci: Evaluating LLM Agents for Forward-Looking AI Research Judgment</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zequn Liu, Youyong Kong, Yingce Xia, Haojie Yin, Qiuyu Tian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.00644) • [📄 arXiv](https://arxiv.org/abs/2606.00644) • [📥 PDF](https://arxiv.org/pdf/2606.00644)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> AI research often requires decisions before future evidence exists: which bottleneck to attack, which direction to pursue, or where a project should be positioned. We introduce ForeSci, a temporally controlled benchmark for evaluating whether LLM ...

</details>

<details>
<summary><b>49. BRepCLIP: Contrastive Multimodal Pretraining on BRep Primitives for CAD Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05515) • [📄 arXiv](https://arxiv.org/abs/2606.05515) • [📥 PDF](https://arxiv.org/pdf/2606.05515)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Learning representations of CAD models is a largely open problem. While 3D representation learning has flourished around point clouds and meshes, the native format of CAD - boundary representations BReps, which encodes exact parametric surfaces, c...

</details>

<details>
<summary><b>50. SABER: Benchmarking Operational Safety of LLM Coding Agents in Stateful Project Workspaces</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.01317) • [📄 arXiv](https://arxiv.org/abs/2606.01317) • [📥 PDF](https://arxiv.org/pdf/2606.01317)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sssr-lab/saber)

> SABER shifts coding-agent safety evaluation from single-turn refusal behavior to the final state of a realistic, stateful workspace after multi-step agent actions. This is an important benchmark direction because many safety failures in coding age...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 50 |
| 📅 Today | [`2026-06-06.json`](data/daily/2026-06-06.json) | 50 |
| 📆 This Week | [`2026-W22.json`](data/weekly/2026-W22.json) | 128 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 128 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-06 | 50 | [View JSON](data/daily/2026-06-06.json) |
| 📄 2026-06-05 | 21 | [View JSON](data/daily/2026-06-05.json) |
| 📄 2026-06-04 | 0 | [View JSON](data/daily/2026-06-04.json) |
| 📄 2026-06-03 | 10 | [View JSON](data/daily/2026-06-03.json) |
| 📄 2026-06-02 | 22 | [View JSON](data/daily/2026-06-02.json) |
| 📄 2026-06-01 | 25 | [View JSON](data/daily/2026-06-01.json) |
| 📄 2026-05-31 | 59 | [View JSON](data/daily/2026-05-31.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W22 | 128 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 128 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |

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
