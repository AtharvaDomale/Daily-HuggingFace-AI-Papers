<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-40-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8243+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">40</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">52</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">515</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8243+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 24, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29897) • [📄 arXiv](https://arxiv.org/abs/2611.29897) • [📥 PDF](https://arxiv.org/pdf/2611.29897)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/MotionMamba2.0) • [⭐ Code](https://github.com/huggingface)

> 🚀 We’re excited to release Motion Mamba 2.0 , bringing autoregressive diffusion and Mamba together for long-horizon humanoid control! 🤖 Our code, models, and training data are all fully open source! 🔓 Explore the demos, try it out, and build on ou...

</details>

<details>
<summary><b>2. 3AGameFactory-Motion: Agentic Rigging and Reliable Complex Motion Generation for Diverse 3D Assets</b> ⭐ 1.35k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30013) • [📄 arXiv](https://arxiv.org/abs/2611.30013) • [📥 PDF](https://arxiv.org/pdf/2611.30013)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenDCAI/GameFactory-3A)

> Our 3AGameFactory-motion leverages the capabilities of large models to support motion generation for a wide variety of 3D assets.

</details>

<details>
<summary><b>3. AutoPILOT: A Visual Agent Harness for Real-Time Gameplay</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28202) • [📄 arXiv](https://arxiv.org/abs/2611.28202) • [📥 PDF](https://arxiv.org/pdf/2611.28202)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper identifies a largely overlooked bottleneck in real-time visual agents: conventional harnesses execute only one action per model call, forcing agents to act slowly on stale observations. AutoPILOT allows models to delegate execution to o...

</details>

<details>
<summary><b>4. Plan4D: Generative Plannable 4D Worlds</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Xiangxiang Chu, Jiahong Wu, Ruimin Lin, Rujing Dang, Hangwei Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26516) • [📄 arXiv](https://arxiv.org/abs/2611.26516) • [📥 PDF](https://arxiv.org/pdf/2611.26516)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/plan4d) • [⭐ Code](https://github.com/huggingface)

> Plan4D: Generative Plannable 4D Worlds

</details>

<details>
<summary><b>5. ViDecide: Teaching Video Diffusion Models to Reason and Plan Step by Step</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26389) • [📄 arXiv](https://arxiv.org/abs/2611.26389) • [📥 PDF](https://arxiv.org/pdf/2611.26389)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Kevin-thu/ViDecide)

> We release ViDecide: Teaching video diffusion models to reason and plan step by step , which explores whether video diffusion models can reason and plan step by step, like the way language models do. Current video models often break physical or lo...

</details>

<details>
<summary><b>6. WorldLeague: Benchmarking Physical Agency</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30989) • [📄 arXiv](https://arxiv.org/abs/2611.30989) • [📥 PDF](https://arxiv.org/pdf/2611.30989)

**💻 Code:** [⭐ Code](https://github.com/xyangzhou/WorldLeague) • [⭐ Code](https://github.com/huggingface)

> WorldLeague benchmarks AI systems’ ability to autonomously achieve long-horizen physical goals across 60 simulated robot tasks spanning exploration, object rearrangement, and adaptation to changing environments. It evaluates VLMs, VLAs, and WAMs u...

</details>

<details>
<summary><b>7. A Glimpse of Training VLMs as Foundation Models for Robotic Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29338) • [📄 arXiv](https://arxiv.org/abs/2611.29338) • [📥 PDF](https://arxiv.org/pdf/2611.29338)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The paper introduces nanoAstra-9B, a framework that directly trains vision-language models (VLMs) as foundation models for robotic manipulation, which achieves a 17.8% zero-shot success rate, compared with 1.4% for π0.5 and 0.2% for DreamZero. Str...

</details>

<details>
<summary><b>8. Embo-Harness: Towards Understanding Frontier-Model-as-Policy Harness</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31201) • [📄 arXiv](https://arxiv.org/abs/2611.31201) • [📥 PDF](https://arxiv.org/pdf/2611.31201)

**💻 Code:** [⭐ Code](https://github.com/DravenALG/Embo-Harness) • [⭐ Code](https://github.com/huggingface)

> Hi everyone, author here 👋 In Embo-Harness, we study how to use frontier models to control robots well and why frontier models can work well in robotics.

</details>

<details>
<summary><b>9. RUIBench: What Makes a Good Robot-Use Interface?</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31115) • [📄 arXiv](https://arxiv.org/abs/2611.31115) • [📥 PDF](https://arxiv.org/pdf/2611.31115)

**💻 Code:** [⭐ Code](https://github.com/showlab/RUIBench) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. GameAnything: From Any Inspirations to Playable Game Worlds</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29238) • [📄 arXiv](https://arxiv.org/abs/2611.29238) • [📥 PDF](https://arxiv.org/pdf/2611.24154)

**💻 Code:** [⭐ Code](https://github.com/AetherLabsAI/GameAnything) • [⭐ Code](https://github.com/huggingface)

> GameAnything: From Any Inspirations to Playable Game Worlds AI-driven game development holds the potential to transform diverse human inspirations into interactive experiences. However, existing systems and benchmarks primarily focus on explicit d...

</details>

<details>
<summary><b>11. WeMM-Ecom: Advancing Multimodal Content Understanding for WeChat E-Commerce</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30662) • [📄 arXiv](https://arxiv.org/abs/2611.30662) • [📥 PDF](https://arxiv.org/pdf/2611.30662)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/WeMM-Ecom)

> We present WeMM-Ecom, a family of multimodal large language models that deliver frontier e-commerce capabilities while achieving state-of-the-art general-purpose performance at their respective scales. The family spans five sizes, from a 0.8B dens...

</details>

<details>
<summary><b>12. Self-WSP: Can Language Models Improve Themselves Through Weight-Space Programming?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28833) • [📄 arXiv](https://arxiv.org/abs/2611.28833) • [📥 PDF](https://arxiv.org/pdf/2611.28833)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Can a language model improve itself by editing its own weights? We explore this with a 0.8B model, achieving substantial gains on GSM8K (11.8% → 62.2%) and MATH500 (7.0% → 32.2%), while finding that the model often fails to understand what its edi...

</details>

<details>
<summary><b>13. WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming</b> ⭐ 28</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25476) • [📄 arXiv](https://arxiv.org/abs/2611.25476) • [📥 PDF](https://arxiv.org/pdf/2611.25476)

**💻 Code:** [⭐ Code](https://github.com/Gen-Verse/WeightCode) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/Gen-Verse/WeightCode

</details>

<details>
<summary><b>14. Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27376) • [📄 arXiv](https://arxiv.org/abs/2611.27376) • [📥 PDF](https://arxiv.org/pdf/2611.27376)

**💻 Code:** [⭐ Code](https://github.com/UCSB-AI/CSPD) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/UCSB-AI/CSPD

</details>

<details>
<summary><b>15. HIVE-Bench: Evaluating Patch-Level Visual Representations for Egocentric Robot Manipulation</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Yiming Wang, Yikeng Chen, Minghao Lai, Zhengtu Liang, Qiwei Liang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30813) • [📄 arXiv](https://arxiv.org/abs/2611.30813) • [📥 PDF](https://arxiv.org/pdf/2611.30813)

**💻 Code:** [⭐ Code](https://github.com/KolaKivy/HIVE-Bench) • [⭐ Code](https://github.com/huggingface)

> Pretrained vision is a common start for robot learning, but most comparisons still score one pooled vector from a third-person image. Bimanual policies instead read dense patch tokens from cameras on the robot. We introduce HIVE-Bench: one shared ...

</details>

<details>
<summary><b>16. Distilling Visual Agents from Limited Interaction Experience</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guodong Long, Chengqi Zhang, Jing Jiang, Siyu Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28618) • [📄 arXiv](https://arxiv.org/abs/2611.28618) • [📥 PDF](https://arxiv.org/pdf/2611.28618)

**💻 Code:** [⭐ Code](https://github.com/elated-sawyer/visual-agent-distillation) • [⭐ Code](https://github.com/huggingface)

> We study how much interaction experience is needed to distill capable visual agents. With just four selected trajectories per environment , offline OPD improves a Qwen3.5-4B student from 49.17% to 82.08% on Navigation and 60.16% to 85.55% on Sokob...

</details>

<details>
<summary><b>17. UUBench: Understanding the Unspoken toward Full-Duplex Multimodal Interaction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haocong He, Bin Ren, Chenfei Liao, Ye Pan, Lihanqian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30157) • [📄 arXiv](https://arxiv.org/abs/2611.30157) • [📥 PDF](https://arxiv.org/pdf/2611.30157)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We’re excited to share a preview of UUBench! 🤗 A hesitation, a raised eyebrow, or a change in tone can shift what someone means. We built UUBench to evaluate how well avatars pick up on those cues and respond appropriately. This is still a work in...

</details>

<details>
<summary><b>18. Embodied-Harness: Towards Recursive Self-Improvement for Robotic Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27689) • [📄 arXiv](https://arxiv.org/abs/2611.27689) • [📥 PDF](https://arxiv.org/pdf/2611.27689)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Can a robot improve across attempts while its model parameters stay frozen? In Embodied-Harness , we explore recursive self-improvement (RSI) for robotic manipulation. The Actor follows operational guidance to carry out a task. After a failed atte...

</details>

<details>
<summary><b>19. What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30207) • [📄 arXiv](https://arxiv.org/abs/2611.30207) • [📥 PDF](https://arxiv.org/pdf/2611.30207)

**💻 Code:** [⭐ Code](https://github.com/Arist12/SWE-Gym-Verified) • [⭐ Code](https://github.com/huggingface)

> 74% of the SWE-Gym tasks a 9B model never solves have tests that reject correct fixes. Fixing these reward labels gives +2.4 / +2.7 on SWE-bench Verified / Pro at equal data size. We release SWE-Gym-Verified: 1,087 audited tasks + the audit pipeli...

</details>

<details>
<summary><b>20. ViDiHand2: Unifying Video Diffusion Priors and Hand Geometry for 3D Hand Motion Reconstruction in the Wild</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30597) • [📄 arXiv](https://arxiv.org/abs/2611.30597) • [📥 PDF](https://arxiv.org/pdf/2611.30597)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We propose ViDiHand2, a geometry-driven framework that reformulates VDM-based hand depth estimation from direct regression into fine-grained geometric constraint solving.

</details>

<details>
<summary><b>21. GameDojo: Benchmarking Real-Time Interactive Agents in Video Games</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30771) • [📄 arXiv](https://arxiv.org/abs/2611.30771) • [📥 PDF](https://arxiv.org/pdf/2611.30771)

**💻 Code:** [⭐ Code](https://github.com/THUSI-Lab/GameDojo) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. A Pipeline-Parallel System for Depth-Recurrent Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30581) • [📄 arXiv](https://arxiv.org/abs/2611.30581) • [📥 PDF](https://arxiv.org/pdf/2611.30581)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Depth-recurrent (Looped) models raise effective depth by applying a shared stack of blocks several times. Scaling them usually requires pipeline parallelism, but commonly used interleaved 1F1B is drawn for an acyclic stage graph in which each micr...

</details>

<details>
<summary><b>23. Polytropos: What Do Vision-Language Model Descriptions Make of a Cultural Object?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chenying Lin, Mohsinul Kabir, Yuyan Wang, Enze Zhang, Yuechen Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29355) • [📄 arXiv](https://arxiv.org/abs/2611.29355) • [📥 PDF](https://arxiv.org/pdf/2611.29355)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Polytropos evaluates what vision-language model descriptions preserve about individual cultural objects by reconstructing images from those descriptions and comparing them with the original museum photographs. The benchmark covers 374 objects from...

</details>

<details>
<summary><b>24. Write, Evolve, Read: Streaming Cameras into a Persistent World State for 4D Occupancy Forecasting</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junkai Chen, Manni Cui, Zhihao Chen, Tianyi Wang, ZiAnwang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30573) • [📄 arXiv](https://arxiv.org/abs/2611.30573) • [📥 PDF](https://arxiv.org/pdf/2611.30573)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>25. Distill the Correction, Not the Rewrite: Hindsight Twins for Self-Improving Text-to-Image Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31048) • [📄 arXiv](https://arxiv.org/abs/2611.31048) • [📥 PDF](https://arxiv.org/pdf/2611.31048)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Distill the Correction, Not the Rewrite: Hindsight Twins for Self-Improving Text-to-Image Models

</details>

<details>
<summary><b>26. SAVIOR: Sample-efficient Adaptation of Vision-Language Models for OCR Representation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27319) • [📄 arXiv](https://arxiv.org/abs/2611.27319) • [📥 PDF](https://arxiv.org/pdf/2611.27319)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Author here (Hyperbots). OCR and vision-language models do well on standard benchmarks but break on patterns common in finance documents: rotated and vertical text, vendor names inside logos, fine print, degraded scans and dense multi-column pages...

</details>

<details>
<summary><b>27. NoHairCache: Calibrate-Once KV Cache Eviction with One Power-Law Exponent</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shuyang Xiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29006) • [📄 arXiv](https://arxiv.org/abs/2611.29006) • [📥 PDF](https://arxiv.org/pdf/2611.29006)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The decay exponent α in KV-cache eviction is an intrinsic property of the model: calibrated once on a single corpus, it transfers directly across domains. Attention-based accumulation (H2O) and recency-based sliding windows (StreamingLLM) are two ...

</details>

<details>
<summary><b>28. Feedback Shortcut: Speculative Decoding for Latent Feedback Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28227) • [📄 arXiv](https://arxiv.org/abs/2611.28227) • [📥 PDF](https://arxiv.org/pdf/2611.28227)

**💻 Code:** [⭐ Code](https://github.com/ifm-ai/LoRM) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/ifm-ai/LoRM

</details>

<details>
<summary><b>29. Prism: Hierarchical Text Detection with Minimalist Design</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26254) • [📄 arXiv](https://arxiv.org/abs/2611.26254) • [📥 PDF](https://arxiv.org/pdf/2611.26254)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Prism, a minimalist framework for hierarchical text detection. Prism performs multi-granularity detection by attaching parallel lightweight heads to a single-scale backbone feature map, covering words, text lines, and paragraphs, and reconstructs ...

</details>

<details>
<summary><b>30. Typhoon OCR 2: From Thai Document Transcription to Multilingual, Multi-Task OCR</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.27494) • [📄 arXiv](https://arxiv.org/abs/2611.27494) • [📥 PDF](https://arxiv.org/pdf/2611.27494)

**💻 Code:** [⭐ Code](https://github.com/srn-source/typhoon-ocr2) • [⭐ Code](https://github.com/huggingface)

> Open 2B/0.8B Thai–English-Chinese document parser and extractor. Open model on Thai MDPBench (80.7) and ThaiOCRBench (59.9), near its base on English benchmarks; GRPO cuts repetition loops from 3.2% to 0.7%. Weights, pip package and eval code rele...

</details>

<details>
<summary><b>31. Fast-JEPA: Advancing the Success-Latency Frontier for Latent World Models</b> ⭐ 20</summary>

<br/>

**👥 Authors:** Changti Wu, Haishan Liu, Shijie Lian, Bin Yu, Sofilyzia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25927) • [📄 arXiv](https://arxiv.org/abs/2611.25927) • [📥 PDF](https://arxiv.org/pdf/2611.25927)

**💻 Code:** [⭐ Code](https://github.com/DeepCybo-PhysAI/Fast-JEPA) • [⭐ Code](https://github.com/huggingface)

> 🎉 Introducing Fast-JEPA: fast visual-goal planning with latent world models! Fast-JEPA combines goal-directed action proposals with direct predictions of complete candidate outcomes, avoiding recursive state rollout during search refinement. 🚀 Acr...

</details>

<details>
<summary><b>32. Beyond Attention Maps: Understanding and Improving Medical MLLMs via Visual Values</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29361) • [📄 arXiv](https://arxiv.org/abs/2611.29361) • [📥 PDF](https://arxiv.org/pdf/2611.29361)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Are We Overlooking the V in MLLMs Attention? Most existing studies on MLLMs focus on improving spatial attention — teaching models where to look. But does looking at the right region necessarily lead to a correct answer? In our recent work, Beyond...

</details>

<details>
<summary><b>33. From Rewarding What Models Say to How They Think: Attention Process Reward Model for LLM Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zijing Hu, Xinpeng Dong, Ziyu Zhao, Nuanqiao Shan, Kairong Han

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30627) • [📄 arXiv](https://arxiv.org/abs/2611.30627) • [📥 PDF](https://arxiv.org/pdf/2611.30627)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> From Rewarding What Models Say to How They Think: Attention Process Reward Model for LLM Reinforcement Learning

</details>

<details>
<summary><b>34. MMMem: Self-Improving Agents with Task-Aware Multimodal Experience Consolidation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kehui Tan, Zhongjing Du, Yunfei Li, Changwei Lv, Yuxuan Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30287) • [📄 arXiv](https://arxiv.org/abs/2611.30287) • [📥 PDF](https://arxiv.org/pdf/2611.30287)

**💻 Code:** [⭐ Code](https://github.com/ChenYX24/MMMem) • [⭐ Code](https://github.com/huggingface)

> We’re sharing MMMem, a multimodal memory framework that helps agents learn from completed tasks. MMMem turns past answers into reusable experiences, links them back to the original evidence, and retrieves them when they can help with a new questio...

</details>

<details>
<summary><b>35. The norm axiom and factor hyperfields</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alessandro Linzi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29473) • [📄 arXiv](https://arxiv.org/abs/2611.29473) • [📥 PDF](https://arxiv.org/pdf/2611.29473)

**💻 Code:** [⭐ Code](https://github.com/linzialessandro/norm-axiom-factor-hyperfields) • [⭐ Code](https://github.com/huggingface)

> We show that the norm axiom forces every factor valued hyperfield to come from a valuation, settle the case of value group Z, and give counterexamples without the norm axiom.

</details>

<details>
<summary><b>36. Deterministic, Reproducible Inference in Exact Integer Arithmetic, and Verifiable Distributed Inference as Its Consequence</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.13730) • [📄 arXiv](https://arxiv.org/abs/2611.13730) • [📥 PDF](https://arxiv.org/pdf/2611.13730)

**💻 Code:** [⭐ Code](https://github.com/pelloran-inc/tritify) • [⭐ Code](https://github.com/huggingface)

> We present tritify, an open-source runtime that runs the forward pass of real models — decoder-only LLMs, MoE decoders, and the SAM 2.1 image/video stack — entirely in exact integer arithmetic, with floating point confined to offline quantization....

</details>

<details>
<summary><b>37. UniVEP: Unifying Velocity and Endpoint Prediction for Robot Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26103) • [📄 arXiv](https://arxiv.org/abs/2611.26103) • [📥 PDF](https://arxiv.org/pdf/2611.26103)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Welcome

</details>

<details>
<summary><b>38. Seeing Right, Writing Wrong: Where Short Chain-of-Thought Breaks Spatial VQA</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28665) • [📄 arXiv](https://arxiv.org/abs/2611.28665) • [📥 PDF](https://arxiv.org/pdf/2611.28665)

**💻 Code:** [⭐ Code](https://github.com/MING-ZCH/seeing-right-writing-wrong) • [⭐ Code](https://github.com/huggingface)

> Shows that short chain-of-thought turns spatial VQA from a perception problem into a transcription problem: once a VLM writes a spatial relation, its answer follows that sentence rather than the image, across 15 open VLMs and 6 frontier closed mod...

</details>

<details>
<summary><b>39. How Complete Must an Environment Be to Train a Capable GUI Agent?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.23946) • [📄 arXiv](https://arxiv.org/abs/2611.23946) • [📥 PDF](https://arxiv.org/pdf/2611.23946)

**💻 Code:** [⭐ Code](https://github.com/Hydr0pon1c/Mobilegym-lite) • [⭐ Code](https://github.com/huggingface)

> Graphical user interface (GUI) agents are widely trained in executable GUI environments. To lower cost, build environments in batches, and reduce manual effort, each work trades away part of the environment through its construction choices. We des...

</details>

<details>
<summary><b>40. SWEEP-Bench: Do Embodied Policies Act on the Speed They See?</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30637) • [📄 arXiv](https://arxiv.org/abs/2611.30637) • [📥 PDF](https://arxiv.org/pdf/2611.30637)

**💻 Code:** [⭐ Code](https://github.com/Damin3927/SWEEP-Bench) • [⭐ Code](https://github.com/huggingface)

> We introduced SWEEP-Bench, a benchmark of 80 conveyor tasks at 13 belt speeds. Many fine-tuned policies tend to produce actions suited to the average demonstration speed, and our results suggest that past frames widen the speed range only for thos...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 40 |
| 📅 Today | [`2026-11-24.json`](data/daily/2026-11-24.json) | 40 |
| 📆 This Week | [`2026-W47.json`](data/weekly/2026-W47.json) | 52 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 515 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-24 | 40 | [View JSON](data/daily/2026-11-24.json) |
| 📄 2026-11-23 | 12 | [View JSON](data/daily/2026-11-23.json) |
| 📄 2026-11-22 | 50 | [View JSON](data/daily/2026-11-22.json) |
| 📄 2026-11-21 | 50 | [View JSON](data/daily/2026-11-21.json) |
| 📄 2026-11-20 | 18 | [View JSON](data/daily/2026-11-20.json) |
| 📄 2026-11-19 | 27 | [View JSON](data/daily/2026-11-19.json) |
| 📄 2026-11-18 | 17 | [View JSON](data/daily/2026-11-18.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W47 | 52 | [View JSON](data/weekly/2026-W47.json) |
| 📅 2026-W46 | 184 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 515 | [View JSON](data/monthly/2026-11.json) |
| 🗓️ 2026-10 | 1120 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |

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
