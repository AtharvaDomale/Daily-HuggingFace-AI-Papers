<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-30-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-479+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">30</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">155</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">528</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">479+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 19, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Step-GUI Technical Report</b> ⭐ 1.44k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15431) • [📄 arXiv](https://arxiv.org/abs/2512.15431) • [📥 PDF](https://arxiv.org/pdf/2512.15431)

**💻 Code:** [⭐ Code](https://github.com/stepfun-ai/gelab-zero)

> Recent advances in multimodal large language models unlock unprecedented opportunities for GUI automation. However, a fundamental challenge remains: how to efficiently acquire high-quality training data while maintaining annotation reliability? We...

</details>

<details>
<summary><b>2. DEER: Draft with Diffusion, Verify with Autoregressive Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhijie Deng, Jia Li, Guo-Wei Yang, Zicong Cheng, menghao22

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15176) • [📄 arXiv](https://arxiv.org/abs/2512.15176) • [📥 PDF](https://arxiv.org/pdf/2512.15176)

> Simultaneously leveraging the efficiency of dLLM and the performance of AR models.

</details>

<details>
<summary><b>3. Fast and Accurate Causal Parallel Decoding using Jacobi Forcing</b> ⭐ 52</summary>

<br/>

**👥 Authors:** Tajana Rosing, Samyam Rajbhandari, Yichao Fu, Siqi Kou, Lanxiang Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14681) • [📄 arXiv](https://arxiv.org/abs/2512.14681) • [📥 PDF](https://arxiv.org/pdf/2512.14681)

**💻 Code:** [⭐ Code](https://github.com/hao-ai-lab/JacobiForcing)

> Multi-token generation has emerged as a promising paradigm for accelerating transformer-based large model inference. Recent efforts primarily explore diffusion Large Language Models (dLLMs) for parallel decoding to reduce inference latency. To ach...

</details>

<details>
<summary><b>4. Puzzle Curriculum GRPO for Vision-Centric Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14944) • [📄 arXiv](https://arxiv.org/abs/2512.14944) • [📥 PDF](https://arxiv.org/pdf/2512.14944)

> Recent reinforcement learning (RL) approaches like outcome-supervised GRPO have advanced chain-of-thought reasoning in Vision Language Models (VLMs), yet key issues linger: (i) reliance on costly and noisy hand-curated annotations or external veri...

</details>

<details>
<summary><b>5. HyperVL: An Efficient and Dynamic Multimodal Large Language Model for Edge Devices</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuhang Dong, Zhiqiang Xia, Kaiyang Han, Yuchen Liu, HyperAI Team

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14052) • [📄 arXiv](https://arxiv.org/abs/2512.14052) • [📥 PDF](https://arxiv.org/pdf/2512.14052)

> 🚀 [New Paper] HyperVL: An Efficient and Dynamic Multimodal Large Language Model for Edge Devices Current multimodal large language models (MLLMs) possess strong perceptual and reasoning capabilities, but their high computational and memory require...

</details>

<details>
<summary><b>6. Universal Reasoning Model</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14693) • [📄 arXiv](https://arxiv.org/abs/2512.14693) • [📥 PDF](https://arxiv.org/pdf/2512.14693)

**💻 Code:** [⭐ Code](https://github.com/zitian-gao/URM)

> Universal transformers (UTs) have been widely used for complex reasoning tasks such as ARC-AGI and Sudoku, yet the specific sources of their performance gains remain underexplored. In this work, we systematically analyze UTs variants and show that...

</details>

<details>
<summary><b>7. IC-Effect: Precise and Efficient Video Effects Editing via In-Context Learning</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15635) • [📄 arXiv](https://arxiv.org/abs/2512.15635) • [📥 PDF](https://arxiv.org/pdf/2512.15635)

**💻 Code:** [⭐ Code](https://github.com/CUC-MIPG/IC-Effect)

> No abstract available.

</details>

<details>
<summary><b>8. Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15693) • [📄 arXiv](https://arxiv.org/abs/2512.15693) • [📥 PDF](https://arxiv.org/pdf/2512.15693)

**💻 Code:** [⭐ Code](https://github.com/JoeLeelyf/Skyra)

> Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning https://huggingface.co/papers/2512.15693 Explainable AI-generated video detection with a specialized multimodal LLM. Given an input video, Skyra explicitly identifies human-percei...

</details>

<details>
<summary><b>9. Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiao Xu, Kaiyuan Gao, Zecheng Tang, Zekai Zhang, Shengming Yin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15603) • [📄 arXiv](https://arxiv.org/abs/2512.15603) • [📥 PDF](https://arxiv.org/pdf/2512.15603)

> Recent visual generative models often struggle with consistency during image editing due to the entangled nature of raster images, where all visual content is fused into a single canvas. In contrast, professional design tools employ layered repres...

</details>

<details>
<summary><b>10. Robust and Calibrated Detection of Authentic Multimedia Content</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15182) • [📄 arXiv](https://arxiv.org/abs/2512.15182) • [📥 PDF](https://arxiv.org/pdf/2512.15182)

> The paper “Robust and Calibrated Detection of Authentic Multimedia Content” presents a new framework for identifying whether multimedia particularly deepfakes produced by generative models is genuinely authentic or can be plausibly denied as fake,...

</details>

<details>
<summary><b>11. SAGE: Training Smart Any-Horizon Agents for Long Video Reasoning with Reinforcement Learning</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13874) • [📄 arXiv](https://arxiv.org/abs/2512.13874) • [📥 PDF](https://arxiv.org/pdf/2512.13874)

**💻 Code:** [⭐ Code](https://github.com/allenai/SAGE)

> 📜 explainer thread: https://x.com/allen_ai/status/2001351082916630586 🔗 Project page: https://lnkd.in/eff-DjHx 💻 Code: github.com/allenai/SAGE 📦 Models & data: https://lnkd.in/eT9iVVRk 📝 Paper: arxiv.org/abs/2512.13874

</details>

<details>
<summary><b>12. Can LLMs Guide Their Own Exploration? Gradient-Guided Reinforcement Learning for LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15687) • [📄 arXiv](https://arxiv.org/abs/2512.15687) • [📥 PDF](https://arxiv.org/pdf/2512.15687)

> Can LLMs Guide Their Own Exploration? Gradient-Guided Reinforcement Learning for LLM Reasoning

</details>

<details>
<summary><b>13. FiNERweb: Datasets and Artifacts for Scalable Multilingual Named Entity Recognition</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13884) • [📄 arXiv](https://arxiv.org/abs/2512.13884) • [📥 PDF](https://arxiv.org/pdf/2512.13884)

**💻 Code:** [⭐ Code](https://github.com/whoisjones/FiNERweb)

> GitHub Repo: https://github.com/whoisjones/FiNERweb HF Collection: https://huggingface.co/collections/whoisjones/finerweb

</details>

<details>
<summary><b>14. MMSI-Video-Bench: A Holistic Benchmark for Video-Based Spatial Intelligence</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Peizhou Cao, Sihan Yang, Shaohao Zhu, Runsen Xu, rbler

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10863) • [📄 arXiv](https://arxiv.org/abs/2512.10863) • [📥 PDF](https://arxiv.org/pdf/2512.10863)

**💻 Code:** [⭐ Code](https://github.com/InternRobotics/MMSI-Video-Bench)

> Our homepage: https://rbler1234.github.io/MMSI-VIdeo-Bench.github.io GitHub Page: https://github.com/InternRobotics/MMSI-Video-Bench HuggingFace: https://huggingface.co/datasets/rbler/MMSI-Video-Bench Arxiv: https://arxiv.org/abs/2512.10863

</details>

<details>
<summary><b>15. DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15713) • [📄 arXiv](https://arxiv.org/abs/2512.15713) • [📥 PDF](https://arxiv.org/pdf/2512.15713)

**💻 Code:** [⭐ Code](https://github.com/hustvl/DiffusionVL)

> In recent multimodal research, the diffusion paradigm has emerged as a promising alternative to the autoregressive paradigm (AR), owing to its unique decoding advantages. However, due to the capability limitations of the base diffusion language mo...

</details>

<details>
<summary><b>16. VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12072) • [📄 arXiv](https://arxiv.org/abs/2512.12072) • [📥 PDF](https://arxiv.org/pdf/2512.12072)

> Diverse data is ALL you NEED

</details>

<details>
<summary><b>17. End-to-End Training for Autoregressive Video Diffusion via Self-Resampling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15702) • [📄 arXiv](https://arxiv.org/abs/2512.15702) • [📥 PDF](https://arxiv.org/pdf/2512.15702)

> End-to-End Training for Autoregressive Video Diffusion via Self-Resampling

</details>

<details>
<summary><b>18. VABench: A Comprehensive Benchmark for Audio-Video Generation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09299) • [📄 arXiv](https://arxiv.org/abs/2512.09299) • [📥 PDF](https://arxiv.org/pdf/2512.09299)

**💻 Code:** [⭐ Code](https://github.com/tanABCC/VABench)

> code link: https://github.com/tanABCC/VABench

</details>

<details>
<summary><b>19. In Pursuit of Pixel Supervision for Visual Pre-training</b> ⭐ 50</summary>

<br/>

**👥 Authors:** Dong Wang, Xinjie Lei, Yang Li, Shang-Wen Li, Lihe Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15715) • [📄 arXiv](https://arxiv.org/abs/2512.15715) • [📥 PDF](https://arxiv.org/pdf/2512.15715)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/pixio)

> arXiv lens breakdown of this paper 👉 https://arxivlens.com/PaperView/Details/in-pursuit-of-pixel-supervision-for-visual-pre-training-8810-5e30657e Executive Summary Detailed Breakdown Practical Applications

</details>

<details>
<summary><b>20. VTCBench: Can Vision-Language Models Understand Long Context with Vision-Text Compression?</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15649) • [📄 arXiv](https://arxiv.org/abs/2512.15649) • [📥 PDF](https://arxiv.org/pdf/2512.15649)

**💻 Code:** [⭐ Code](https://github.com/Moenupa/VTCBench)

> A comprehensive benchmark to study VLM's visual text compression ability. Code: https://github.com/Moenupa/VTCBench Huggingface: https://huggingface.co/datasets/MLLM-CL/VTCBench

</details>

<details>
<summary><b>21. Is Nano Banana Pro a Low-Level Vision All-Rounder? A Comprehensive Evaluation on 14 Tasks and 40 Datasets</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yicheng Zhang, Jiaxin Zhu, Hanyu Zhou, Haoyou Deng, Jialong Zuo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15110) • [📄 arXiv](https://arxiv.org/abs/2512.15110) • [📥 PDF](https://arxiv.org/pdf/2512.15110)

> The rapid evolution of text-to-image generation models has revolutionized visual content creation. While commercial products like Nano Banana Pro have garnered significant attention, their potential as generalist solvers for traditional low-level ...

</details>

<details>
<summary><b>22. WAY: Estimation of Vessel Destination in Worldwide AIS Trajectory</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Sung Won Han, Dongil Park, Wooseok Shin, Hyun Joon Park, sadPororo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13190) • [📄 arXiv](https://arxiv.org/abs/2512.13190) • [📥 PDF](https://arxiv.org/pdf/2512.13190)

**💻 Code:** [⭐ Code](https://github.com/sadPororo/WAY)

> A novel deep learning architecture, WAY, uses nested sequence structures and spatial grids for accurate long-term vessel destination estimation from AIS data.

</details>

<details>
<summary><b>23. FrontierCS: Evolving Challenges for Evolving Intelligence</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shang Zhou, Huanzhi Mao, Zhifei Li, Wenhao Chai, Qiuyang Mang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15699) • [📄 arXiv](https://arxiv.org/abs/2512.15699) • [📥 PDF](https://arxiv.org/pdf/2512.15699)

**💻 Code:** [⭐ Code](https://github.com/FrontierCS/Frontier-CS)

> https://github.com/FrontierCS/Frontier-CS Introducing FrontierCS. LiveCodeBench Pro is already a challenging competitive programming benchmark, so why do we still need to push one step further? The motivation behind FrontierCS is actually pretty s...

</details>

<details>
<summary><b>24. SCOPE: Prompt Evolution for Enhancing Agent Effectiveness</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Yunhe Wang, Sinno Jialin Pan, Shixiong Kai, Hui-Ling Zhen, Zehua Pei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15374) • [📄 arXiv](https://arxiv.org/abs/2512.15374) • [📥 PDF](https://arxiv.org/pdf/2512.15374)

**💻 Code:** [⭐ Code](https://github.com/JarvisPei/SCOPE)

> We introduce SCOPE (Self-evolving Context Optimization via Prompt Evolution), a framework that automatically evolves agent prompts by learning from execution traces. Try it now: pip install scope-optimizer 📄 Paper: https://arxiv.org/abs/2512.15374...

</details>

<details>
<summary><b>25. Understanding and Improving Hyperbolic Deep Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14202) • [📄 arXiv](https://arxiv.org/abs/2512.14202) • [📥 PDF](https://arxiv.org/pdf/2512.14202)

> tl;dr : We analytically show that large-norm embeddings destabilize hyperbolic representations in deep RL. In PPO, this coincides with trust-region violations. Existing methods based on SpectralNorm mitigate these issues only partially. We propose...

</details>

<details>
<summary><b>26. Hybrid Attribution Priors for Explainable and Robust Model Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuanxing Zhang, Shangyuan Li, Feng Zhang, Zhuoran Zhang, DogNeverSleep

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14719) • [📄 arXiv](https://arxiv.org/abs/2512.14719) • [📥 PDF](https://arxiv.org/pdf/2512.14719)

> Small language models (SLMs) are widely used in tasks that require low latency and lightweight deployment, particularly classification. As interpretability and robustness gain increasing importance, explanation-guided learning has emerged as an ef...

</details>

<details>
<summary><b>27. LikeBench: Evaluating Subjective Likability in LLMs for Personalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13077) • [📄 arXiv](https://arxiv.org/abs/2512.13077) • [📥 PDF](https://arxiv.org/pdf/2512.13077)

> Memory ≠ likability. LikeBench shows that models can remember more but still feel worse to talk to, and even SOTA models struggle to become likable over time despite having more information about a user.

</details>

<details>
<summary><b>28. Simultaneous Tactile-Visual Perception for Learning Multimodal Robot Manipulation</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09851) • [📄 arXiv](https://arxiv.org/abs/2512.09851) • [📥 PDF](https://arxiv.org/pdf/2512.09851)

**💻 Code:** [⭐ Code](https://github.com/YuyangLee/TacThru)

> Simultaneous Tactile-Visual Perception for Learning Multimodal Robot Manipulation

</details>

<details>
<summary><b>29. Towards Seamless Interaction: Causal Turn-Level Modeling of Interactive 3D Conversational Head Dynamics</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Kun Li, Qing Zhou, Zhihao Huang, Fei Wang, Junjie Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15340) • [📄 arXiv](https://arxiv.org/abs/2512.15340) • [📥 PDF](https://arxiv.org/pdf/2512.15340)

**💻 Code:** [⭐ Code](https://github.com/CoderChen01/towards-seamleass-interaction/blob/main/README.md) • [⭐ Code](https://github.com/CoderChen01/towards-seamleass-interaction)

> Human conversation is a continuous exchange of speech and nonverbal cues—including head nods, gaze shifts, and subtle expressions. Most existing approaches, however, treat talking-head and listening-head generation as separate problems, or rely on...

</details>

<details>
<summary><b>30. SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tri Dao, Ion Stoica, Xinle Cheng, Mayank Mishra, Wentao Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14080) • [📄 arXiv](https://arxiv.org/abs/2512.14080) • [📥 PDF](https://arxiv.org/pdf/2512.14080)

> We propose to co-design the MoE architecture with a GPU kernel tailored to NVIDIA Blackwell and Hopper generation GPUs and a novel routing method. (1) We derive an algorithm to compute the MoE backward pass more efficiently leading to a much small...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 30 |
| 📅 Today | [`2025-12-19.json`](data/daily/2025-12-19.json) | 30 |
| 📆 This Week | [`2025-W50.json`](data/weekly/2025-W50.json) | 155 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 528 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-19 | 30 | [View JSON](data/daily/2025-12-19.json) |
| 📄 2025-12-18 | 38 | [View JSON](data/daily/2025-12-18.json) |
| 📄 2025-12-17 | 41 | [View JSON](data/daily/2025-12-17.json) |
| 📄 2025-12-16 | 21 | [View JSON](data/daily/2025-12-16.json) |
| 📄 2025-12-15 | 25 | [View JSON](data/daily/2025-12-15.json) |
| 📄 2025-12-14 | 25 | [View JSON](data/daily/2025-12-14.json) |
| 📄 2025-12-13 | 24 | [View JSON](data/daily/2025-12-13.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W50 | 155 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 528 | [View JSON](data/monthly/2025-12.json) |

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
