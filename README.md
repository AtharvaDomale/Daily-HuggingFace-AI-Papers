<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-29-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-229+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">29</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">91</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">278</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">229+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 10, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07461) • [📄 arXiv](https://arxiv.org/abs/2512.07461) • [📥 PDF](https://arxiv.org/pdf/2512.07461)

**💻 Code:** [⭐ Code](https://github.com/bigai-nlco/Native-Parallel-Reasoner)

> Paper: https://arxiv.org/abs/2512.07461 Code: https://github.com/bigai-nlco/Native-Parallel-Reasoner Model & Data: https://huggingface.co/bigai-NPR Website: https://bigai-nlco.github.io/Native-Parallel-Reasoner

</details>

<details>
<summary><b>2. Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07525) • [📄 arXiv](https://arxiv.org/abs/2512.07525) • [📥 PDF](https://arxiv.org/pdf/2512.07525)

**💻 Code:** [⭐ Code](https://github.com/OpenMOSS/rope_pp)

> Rotary Position Embeddings (RoPE) have become a standard for encoding sequence order in Large Language Models (LLMs) by applying rotations to query and key vectors in the complex plane. Standard implementations, however, utilize only the real comp...

</details>

<details>
<summary><b>3. Unified Video Editing with Temporal Reasoner</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07469) • [📄 arXiv](https://arxiv.org/abs/2512.07469) • [📥 PDF](https://arxiv.org/pdf/2512.07469)

**💻 Code:** [⭐ Code](https://github.com/knightyxp/VideoCoF)

> A Chain of Frames video editing method enbale temporal reasoning and 4x video length extrapolation with just 50k training pairs! 🏠 Page: videocof.github.io/ 📄 Paper: arxiv.org/abs/2512.07469 💻 Code: github.com/knightyxp/VideoCoF

</details>

<details>
<summary><b>4. Voxify3D: Pixel Art Meets Volumetric Rendering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu-Lun Liu, chien90190, JiewenChan, YiChuanH

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07834) • [📄 arXiv](https://arxiv.org/abs/2512.07834) • [📥 PDF](https://arxiv.org/pdf/2512.07834)

> Stylized voxel art is widely used in games and digital media, but turning 3D meshes into visually appealing voxel forms remains challenging and often requires manual effort. Existing methods struggle to preserve semantic structure and offer limite...

</details>

<details>
<summary><b>5. Scaling Zero-Shot Reference-to-Video Generation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06905) • [📄 arXiv](https://arxiv.org/abs/2512.06905) • [📥 PDF](https://arxiv.org/pdf/2512.06905)

**💻 Code:** [⭐ Code](https://github.com/franciszzj/Saber)

> Reference-to-video (R2V) generation aims to synthesize videos that align with a text prompt while preserving the subject identity from reference images. However, current R2V methods are hindered by the reliance on explicit reference image-video-te...

</details>

<details>
<summary><b>6. DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06749) • [📄 arXiv](https://arxiv.org/abs/2512.06749) • [📥 PDF](https://arxiv.org/pdf/2512.06749)

> Project website with an intro video is available at: https://aka.ms/DoVer .

</details>

<details>
<summary><b>7. Distribution Matching Variational AutoEncoder</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07778) • [📄 arXiv](https://arxiv.org/abs/2512.07778) • [📥 PDF](https://arxiv.org/pdf/2512.07778)

**💻 Code:** [⭐ Code](https://github.com/sen-ye/dmvae%7D)

> Most visual generative models compress images into a latent space before applying diffusion or autoregressive modelling. Yet, existing approaches such as VAEs and foundation model aligned encoders implicitly constrain the latent space without expl...

</details>

<details>
<summary><b>8. EgoEdit: Dataset, Real-Time Streaming Model, and Benchmark for Egocentric Video Editing</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06065) • [📄 arXiv](https://arxiv.org/abs/2512.06065) • [📥 PDF](https://arxiv.org/pdf/2512.06065)

**💻 Code:** [⭐ Code](https://github.com/snap-research/EgoEdit)

> We propose a framework for real-time egocentric video editing. Our system is composed of: EgoEditData, a manually curated dataset of 100k video editing pairs focusing on the egocentric case and featuring object substitution and removal under chall...

</details>

<details>
<summary><b>9. Relational Visual Similarity</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Jing Shi, Yilin Wang, Krishna Kumar Singh, Sicheng Mo, thaoshibe

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07833) • [📄 arXiv](https://arxiv.org/abs/2512.07833) • [📥 PDF](https://arxiv.org/pdf/2512.07833)

**💻 Code:** [⭐ Code](https://github.com/thaoshibe/relsim)

> Humans do not just see attribute similarity -- we also see relational similarity. An apple is like a peach because both are reddish fruit, but the Earth is also like a peach: its crust, mantle, and core correspond to the peach's skin, flesh, and p...

</details>

<details>
<summary><b>10. Multi-view Pyramid Transformer: Look Coarser to See Broader</b> ⭐ 56</summary>

<br/>

**👥 Authors:** Jungwoo Kim, Younggeun Lee, Seungtae Nam, Seungkwon Yang, Gynjn

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07806) • [📄 arXiv](https://arxiv.org/abs/2512.07806) • [📥 PDF](https://arxiv.org/pdf/2512.07806)

**💻 Code:** [⭐ Code](https://github.com/Gynjn/MVP)

> We are excited to share our recent work "Multi-view Pyramid Transformer: Look Coarser to See Broader" Paper: https://arxiv.org/abs/2512.07806 Project page: https://gynjn.github.io/MVP/ Code: https://github.com/Gynjn/MVP

</details>

<details>
<summary><b>11. LongCat-Image Technical Report</b> ⭐ 307</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07584) • [📄 arXiv](https://arxiv.org/abs/2512.07584) • [📥 PDF](https://arxiv.org/pdf/2512.07584)

**💻 Code:** [⭐ Code](https://github.com/meituan-longcat/LongCat-Image)

> We introduce LongCat-Image, a pioneering open-source and bilingual (Chinese-English) foundation model for image generation, designed to address core challenges in multilingual text rendering, photorealism, deployment efficiency, and developer acce...

</details>

<details>
<summary><b>12. UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07831) • [📄 arXiv](https://arxiv.org/abs/2512.07831) • [📥 PDF](https://arxiv.org/pdf/2512.07831)

**💻 Code:** [⭐ Code](https://github.com/dvlab-research/UnityVideo)

> Project Website https://jackailab.github.io/Projects/UnityVideo/

</details>

<details>
<summary><b>13. On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07783) • [📄 arXiv](https://arxiv.org/abs/2512.07783) • [📥 PDF](https://arxiv.org/pdf/2512.07783)

**💻 Code:** [⭐ Code](https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning)

> We develop a fully controlled experimental framework that isolates the causal contributions of pre-training, mid-training, and RL-based post-training. We show that: 1) RL produces true capability gains (pass@128) only when pre-training leaves suff...

</details>

<details>
<summary><b>14. SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nanyun Peng, Swastik Roy, Arpit Gupta, Sruthi Gorantla, Salman Rahman

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03244) • [📄 arXiv](https://arxiv.org/abs/2512.03244) • [📥 PDF](https://arxiv.org/pdf/2512.03244)

> Please find our paper on training process reward models without ground truth by leveraging inference-time scaling methods, enabling reinforcement learning in domains where verifiable answers are unavailable.

</details>

<details>
<summary><b>15. ReCamDriving: LiDAR-Free Camera-Controlled Novel Trajectory Video Generation</b> ⭐ 23</summary>

<br/>

**👥 Authors:** Taojun Ding, Jiehui Huang, Mantang Guo, wangshx, Iron-lyk

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03621) • [📄 arXiv](https://arxiv.org/abs/2512.03621) • [📥 PDF](https://arxiv.org/pdf/2512.03621)

**💻 Code:** [⭐ Code](https://github.com/Iron-LYK/ReCamDriving)

> Project page: https://recamdriving.github.io/

</details>

<details>
<summary><b>16. Beyond Token-level Supervision: Unlocking the Potential of Decoding-based Regression via Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiacheng Chen, Ziniu Li, Sheng Tang, Ming Chen, trxcc2002

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06533) • [📄 arXiv](https://arxiv.org/abs/2512.06533) • [📥 PDF](https://arxiv.org/pdf/2512.06533)

> No abstract available.

</details>

<details>
<summary><b>17. VG-Refiner: Towards Tool-Refined Referring Grounded Reasoning via Agentic Reinforcement Learning</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Yansong Tang, Haoji Zhang, Jingxuan Niu, Wenlong Liu, VoyageWang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06373) • [📄 arXiv](https://arxiv.org/abs/2512.06373) • [📥 PDF](https://arxiv.org/pdf/2512.06373)

**💻 Code:** [⭐ Code](https://github.com/VoyageWang/VG-Refiner)

> The project page is https://github.com/VoyageWang/VG-Refiner

</details>

<details>
<summary><b>18. OmniSafeBench-MM: A Unified Benchmark and Toolbox for Multimodal Jailbreak Attack-Defense Evaluation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Simeng Qin, Teng Ma, Qi Guo, Jie Liao, jiaxiaojunQAQ

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06589) • [📄 arXiv](https://arxiv.org/abs/2512.06589) • [📥 PDF](https://arxiv.org/pdf/2512.06589)

> This work presents OmniSafeBench-MM, a unified, open-source benchmark and toolbox designed for comprehensive evaluation of multimodal jailbreak attack and defense methods. It integrates 13 representative attack techniques, 15 defense strategies, a...

</details>

<details>
<summary><b>19. One Layer Is Enough: Adapting Pretrained Visual Encoders for Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07829) • [📄 arXiv](https://arxiv.org/abs/2512.07829) • [📥 PDF](https://arxiv.org/pdf/2512.07829)

> We proposed FAE which adapts pretrained ViT as the latent space for visual generative models

</details>

<details>
<summary><b>20. Group Representational Position Encoding</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07805) • [📄 arXiv](https://arxiv.org/abs/2512.07805) • [📥 PDF](https://arxiv.org/pdf/2512.07805)

**💻 Code:** [⭐ Code](https://github.com/model-architectures/GRAPE)

> Introducing GRAPE: Group Representational Position Encoding. Embracing General Relative Law of Position Encoding, unifying and improving Multiplicative and Additive Position Encoding, such as RoPE and Alibi! Better performance with a clear theoret...

</details>

<details>
<summary><b>21. Decouple to Generalize: Context-First Self-Evolving Learning for Data-Scarce Vision-Language Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06835) • [📄 arXiv](https://arxiv.org/abs/2512.06835) • [📥 PDF](https://arxiv.org/pdf/2512.06835)

> Experiment Results 📊 We evaluate DoGe on 7 benchmarks covering: General visual reasoning & hallucination (MMMU, MMStar, HallBench) Specialized domain reasoning (MathVision, MathVista, ChemBench, MSEarthMCQ) 3B-level Models Performance Method MMMU ...

</details>

<details>
<summary><b>22. VideoVLA: Video Generators Can Be Generalizable Robot Manipulators</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yaobo Liang, Zhiying Du, Fangyun Wei, godjiaolongge, ys3197

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06963) • [📄 arXiv](https://arxiv.org/abs/2512.06963) • [📥 PDF](https://arxiv.org/pdf/2512.06963)

> Generalization in robot manipulation is essential for deploying robots in open-world environments and advancing toward artificial general intelligence. While recent Vision-Language-Action (VLA) models leverage large pre-trained understanding model...

</details>

<details>
<summary><b>23. Rethinking Training Dynamics in Scale-wise Autoregressive Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06421) • [📄 arXiv](https://arxiv.org/abs/2512.06421) • [📥 PDF](https://arxiv.org/pdf/2512.06421)

> Recent advances in autoregressive (AR) generative models have produced increasingly powerful systems for media synthesis. Among them, next-scale prediction has emerged as a popular paradigm, where models generate images in a coarse-to-fine manner....

</details>

<details>
<summary><b>24. Small-Gain Nash: Certified Contraction to Nash Equilibria in Differentiable Games</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06791) • [📄 arXiv](https://arxiv.org/abs/2512.06791) • [📥 PDF](https://arxiv.org/pdf/2512.06791)

**💻 Code:** [⭐ Code](https://github.com/AashVed/SmallGainNash)

> Gradient methods in games are usually proven to converge only under strong monotonicity in the Euclidean geometry (Rosen-style assumptions). That fails even for simple coupled quadratic games, yet in practice we still often see convergence. This p...

</details>

<details>
<summary><b>25. Vector Quantization using Gaussian Variational Autoencoder</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Wendi Zheng, jerytang, Ya-Qin, jmhernandezlobato, xutongda

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06609) • [📄 arXiv](https://arxiv.org/abs/2512.06609) • [📥 PDF](https://arxiv.org/pdf/2512.06609)

**💻 Code:** [⭐ Code](https://github.com/Stability-AI/generative-models) • [⭐ Code](https://github.com/tongdaxu/VQ-VAE-from-Gaussian-VAE)

> State-of-the-Art VQ-VAE from Gaussian VAE without Training! We train a Gaussian VAE, convert it into VQ-VAE with almost 100% codebook usage, and keeps reconstruction performance! As flexible to setup as VQ-VAE, supporting: codebook size, codebook ...

</details>

<details>
<summary><b>26. DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue</b> ⭐ 2</summary>

<br/>

**👥 Authors:** YijunLiao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.03704) • [📄 arXiv](https://arxiv.org/abs/2512.03704) • [📥 PDF](https://arxiv.org/pdf/2512.03704)

**💻 Code:** [⭐ Code](https://github.com/lyj20071013/DZ-TDPO)

> 🔥 Solving "State Inertia" in Long-Context LLMs! We introduce DZ-TDPO, a non-destructive alignment framework. Problem: Standard DPO causes "Alignment Tax" (PPL explosion >100) when updating user states in long context. Solution: Dynamic KL Constrai...

</details>

<details>
<summary><b>27. JEPA as a Neural Tokenizer: Learning Robust Speech Representations with Density Adaptive Attention</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Linsey Pang, Aaron Elkins, Aman Chadha, Christos Constantinou, Georgios Ioannides

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07168) • [📄 arXiv](https://arxiv.org/abs/2512.07168) • [📥 PDF](https://arxiv.org/pdf/2512.07168)

> This paper introduces JEPA+DAAM , a two-stage self-supervised framework that combines the Joint-Embedding Predictive Architecture (JEPA) with a Gaussian mixture–based Density Adaptive Attention Mechanism (DAAM) to learn semantically rich and highl...

</details>

<details>
<summary><b>28. Embodied Referring Expression Comprehension in Human-Robot Interaction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ganesh Nanduru, amanchadha, Anubis91, alexiglad, mmiakashs

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06558) • [📄 arXiv](https://arxiv.org/abs/2512.06558) • [📥 PDF](https://arxiv.org/pdf/2512.06558)

> The paper introduces Refer360 , a comprehensive multimodal dataset for embodied referring expression comprehension in human-robot interaction (HRI), and proposes MuRes , a lightweight guided residual module that selectively reinforces modality-spe...

</details>

<details>
<summary><b>29. The SAM2-to-SAM3 Gap in the Segment Anything Model Family: Why Prompt-Based Expertise Fails in Concept-Driven Image Segmentation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.06032) • [📄 arXiv](https://arxiv.org/abs/2512.06032) • [📥 PDF](https://arxiv.org/pdf/2512.06032)

**💻 Code:** [⭐ Code](https://github.com/Applied-AI-Research-Lab/The-SAM2-to-SAM3-Gap-in-the-Segment-Anything-Model-Family)

> This paper investigates the fundamental discontinuity between the latest two Segment Anything Models: SAM2 and SAM3 (also called SAMv2 and SAMv3). We explain why the expertise in prompt-based segmentation of SAM2 does not transfer to the multimoda...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 29 |
| 📅 Today | [`2025-12-10.json`](data/daily/2025-12-10.json) | 29 |
| 📆 This Week | [`2025-W49.json`](data/weekly/2025-W49.json) | 91 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 278 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-10 | 29 | [View JSON](data/daily/2025-12-10.json) |
| 📄 2025-12-09 | 24 | [View JSON](data/daily/2025-12-09.json) |
| 📄 2025-12-08 | 38 | [View JSON](data/daily/2025-12-08.json) |
| 📄 2025-12-07 | 38 | [View JSON](data/daily/2025-12-07.json) |
| 📄 2025-12-06 | 38 | [View JSON](data/daily/2025-12-06.json) |
| 📄 2025-12-05 | 38 | [View JSON](data/daily/2025-12-05.json) |
| 📄 2025-12-04 | 24 | [View JSON](data/daily/2025-12-04.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W49 | 91 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 278 | [View JSON](data/monthly/2025-12.json) |

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
