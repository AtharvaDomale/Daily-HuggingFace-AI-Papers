<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-38-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-449+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">125</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">498</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">449+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 18, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. MMGR: Multi-Modal Generative Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haozhe Zhao, Haoyi Qiu, Zefan Cai, ZGZzz, SueMintony

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14691) • [📄 arXiv](https://arxiv.org/abs/2512.14691) • [📥 PDF](https://arxiv.org/pdf/2512.14691)

> MMGR proposes a principled, multi-domain benchmark for evaluating generative models' physical, logical, and spatial reasoning in video and image generation, diagnosing global consistency and causal correctness.

</details>

<details>
<summary><b>2. Video Reality Test: Can AI-Generated ASMR Videos fool VLMs and Humans?</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Rui Zhao, Yi Zhan, Weijia Wu, Jiaqi Wang, KevinQHLin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13281) • [📄 arXiv](https://arxiv.org/abs/2512.13281) • [📥 PDF](https://arxiv.org/pdf/2512.13281)

**💻 Code:** [⭐ Code](https://github.com/video-reality-test/video-reality-test)

> Recent advances in video generation have produced vivid content that are often indistinguishable from real videos, making AI-generated video detection an emerging societal challenge. Prior AIGC detection benchmarks mostly evaluate video without au...

</details>

<details>
<summary><b>3. WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling</b> ⭐ 302</summary>

<br/>

**👥 Authors:** Zehan Wang, Junta Wu, Haoyuan Wang, Haiyu Zhang, wenqsun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14614) • [📄 arXiv](https://arxiv.org/abs/2512.14614) • [📥 PDF](https://arxiv.org/pdf/2512.14614)

**💻 Code:** [⭐ Code](https://github.com/Tencent-Hunyuan/HY-WorldPlay)

> This paper presents WorldPlay, a streaming video diffusion model that enables real-time, interactive world modeling with long-term geometric consistency, resolving the trade-off between speed and memory that limits current methods. WorldPlay draws...

</details>

<details>
<summary><b>4. Scone: Bridging Composition and Distinction in Subject-Driven Image Generation via Unified Understanding-Generation Modeling</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12675) • [📄 arXiv](https://arxiv.org/abs/2512.12675) • [📥 PDF](https://arxiv.org/pdf/2512.12675)

**💻 Code:** [⭐ Code](https://github.com/Ryann-Ran/Scone)

> Code: https://github.com/Ryann-Ran/Scone

</details>

<details>
<summary><b>5. RoboTracer: Mastering Spatial Trace with Reasoning in Vision-Language Models for Robotics</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13660) • [📄 arXiv](https://arxiv.org/abs/2512.13660) • [📥 PDF](https://arxiv.org/pdf/2512.13660)

**💻 Code:** [⭐ Code](https://github.com/Zhoues/RoboTracer)

> Project Page: https://zhoues.github.io/RoboTracer/ We present RoboTracer, the first 3D-aware VLM for multi-step metric-grounded spatial tracing with explicit reasoning. Highlights: RoboTracer first acquires both 3D spatial referring and measuring ...

</details>

<details>
<summary><b>6. OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value</b> ⭐ 80</summary>

<br/>

**👥 Authors:** Xin Gao, Mengzhang Cai, ChampionZhong, Xiaoyang318, Word2Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14051) • [📄 arXiv](https://arxiv.org/abs/2512.14051) • [📥 PDF](https://arxiv.org/pdf/2512.14051)

**💻 Code:** [⭐ Code](https://github.com/OpenDataArena/OpenDataArena-Tool)

> https://opendataarena.github.io/index.html

</details>

<details>
<summary><b>7. Reveal Hidden Pitfalls and Navigate Next Generation of Vector Similarity Search from Task-Centric Views</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Hua Fan, Haotian Wu, Jiahua Wu, Cong Fu, Tingyang-Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12980) • [📄 arXiv](https://arxiv.org/abs/2512.12980) • [📥 PDF](https://arxiv.org/pdf/2512.12980)

**💻 Code:** [⭐ Code](https://github.com/ZJU-DAILY/Iceberg)

> Vector Similarity Search (VSS) in high-dimensional spaces is rapidly emerging as core functionality in next-generation database systems for numerous data-intensive services -- from embedding lookups in large language models (LLMs), to semantic inf...

</details>

<details>
<summary><b>8. Vector Prism: Animating Vector Graphics by Stratifying Semantic Structure</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14336) • [📄 arXiv](https://arxiv.org/abs/2512.14336) • [📥 PDF](https://arxiv.org/pdf/2512.14336)

**💻 Code:** [⭐ Code](https://github.com/YeolJ00/vector-prism)

> Project page: https://yeolj00.github.io/personal-projects/vector-prism/

</details>

<details>
<summary><b>9. MemFlow: Flowing Adaptive Memory for Consistent and Efficient Long Video Narratives</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xin Tao, Shuai Yang, Xi Chen, Sihui Ji, Hengshuang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14699) • [📄 arXiv](https://arxiv.org/abs/2512.14699) • [📥 PDF](https://arxiv.org/pdf/2512.14699)

> MemFlow uses a retrieval-driven adaptive memory and selective attention to maintain narrative coherence in long-streaming video generation with minimal overhead.

</details>

<details>
<summary><b>10. RecGPT-V2 Technical Report</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dian Chen, Chao Yi, zhjgao, TangJiakai5704, hairlatic

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14503) • [📄 arXiv](https://arxiv.org/abs/2512.14503) • [📥 PDF](https://arxiv.org/pdf/2512.14503)

> 🌟 RecGPT-V2: A Major Leap in LLM-Powered Recommendation (RecGPT-V1’s Power Upgrade!) 🌟 Thrilled to unveil RecGPT-V2—the highly anticipated successor to RecGPT-V1! This agentic framework addresses V1’s core limitations, fusing cognitive reasoning w...

</details>

<details>
<summary><b>11. ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhaohe Liao, Junjie Zhou, Pandeng Li, Xiaoyi Bao, lntzm

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13303) • [📄 arXiv](https://arxiv.org/abs/2512.13303) • [📥 PDF](https://arxiv.org/pdf/2512.13303)

> Nano Banana Pro excels at this. We hope our methods and bench can draw more community's attention to this type of genenration ability.

</details>

<details>
<summary><b>12. Feedforward 3D Editing via Text-Steerable Image-to-3D</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13678) • [📄 arXiv](https://arxiv.org/abs/2512.13678) • [📥 PDF](https://arxiv.org/pdf/2512.13678)

> Very cool model that lets you edit 3D digital objects into whatever way you like, using natural language instructions! Project Home: https://glab-caltech.github.io/steer3d/ Demo: https://glab-caltech.github.io/steer3d/#demo

</details>

<details>
<summary><b>13. Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13607) • [📄 arXiv](https://arxiv.org/abs/2512.13607) • [📥 PDF](https://arxiv.org/pdf/2512.13607)

> The Nemotron-Cascade models and the full collection of training data are released at: https://huggingface.co/collections/nvidia/nemotron-cascade

</details>

<details>
<summary><b>14. Olmo 3</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13961) • [📄 arXiv](https://arxiv.org/abs/2512.13961) • [📥 PDF](https://arxiv.org/pdf/2512.13961)

> We introduce Olmo 3, a family of state-of-the-art, fully-open language models at the 7B and 32B parameter scales. Olmo 3 model construction targets long-context reasoning, function calling, coding, instruction following, general chat, and knowledg...

</details>

<details>
<summary><b>15. Differentiable Evolutionary Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Difan Zou, Xunjian Yin, Xuhan Huang, Tianle Li, sitao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13399) • [📄 arXiv](https://arxiv.org/abs/2512.13399) • [📥 PDF](https://arxiv.org/pdf/2512.13399)

**💻 Code:** [⭐ Code](https://github.com/sitaocheng/DERL)

> Code: https://github.com/sitaocheng/DERL Models: https://huggingface.co/DifferentiableEvolutionaryRL

</details>

<details>
<summary><b>16. VersatileFFN: Achieving Parameter Efficiency in LLMs via Adaptive Wide-and-Deep Reuse</b> ⭐ 924</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14531) • [📄 arXiv](https://arxiv.org/abs/2512.14531) • [📥 PDF](https://arxiv.org/pdf/2512.14531)

**💻 Code:** [⭐ Code](https://github.com/huawei-noah/noah-research/tree/master/VersatileFFN)

> The rapid scaling of Large Language Models (LLMs) has achieved remarkable performance, but it also leads to prohibitive memory costs. Existing parameter-efficient approaches such as pruning and quantization mainly compress pretrained models withou...

</details>

<details>
<summary><b>17. A4-Agent: An Agentic Framework for Zero-Shot Affordance Reasoning</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Hanqing Wang, Kanghao Chen, Chenfei-Liao, Harold328, zhangzixin02

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14442) • [📄 arXiv](https://arxiv.org/abs/2512.14442) • [📥 PDF](https://arxiv.org/pdf/2512.14442)

**💻 Code:** [⭐ Code](https://github.com/EnVision-Research/A4-Agent)

> Project Page: https://zixinzhang02.github.io/A4-Agent-page/

</details>

<details>
<summary><b>18. SS4D: Native 4D Generative Model via Structured Spacetime Latents</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14284) • [📄 arXiv](https://arxiv.org/abs/2512.14284) • [📥 PDF](https://arxiv.org/pdf/2512.14284)

**💻 Code:** [⭐ Code](https://github.com/Lizb6626/SS4D/)

> project page: https://lizb6626.github.io/SS4D/ code: https://github.com/Lizb6626/SS4D/

</details>

<details>
<summary><b>19. Sparse-LaViDa: Sparse Multimodal Discrete Diffusion Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14008) • [📄 arXiv](https://arxiv.org/abs/2512.14008) • [📥 PDF](https://arxiv.org/pdf/2512.14008)

> Efficient Training and Inference for unified multi-modal diffusion language models

</details>

<details>
<summary><b>20. Spherical Leech Quantization for Visual Tokenization and Generation</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Chutong Yang, Zhenlin Xu, Hanwen Jiang, eadeli42, zhaoyue-zephyrus

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14697) • [📄 arXiv](https://arxiv.org/abs/2512.14697) • [📥 PDF](https://arxiv.org/pdf/2512.14697)

**💻 Code:** [⭐ Code](https://github.com/zhaoyue-zephyrus/InfinityCC) • [⭐ Code](https://github.com/zhaoyue-zephyrus/bsq-vit)

> Blog: https://ai.stanford.edu/~yzz/blog/articles/npq.html Code for reconstruction and compression: https://github.com/zhaoyue-zephyrus/bsq-vit Code for generation with InfinityCC: https://github.com/zhaoyue-zephyrus/InfinityCC

</details>

<details>
<summary><b>21. CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14696) • [📄 arXiv](https://arxiv.org/abs/2512.14696) • [📥 PDF](https://arxiv.org/pdf/2512.14696)

**💻 Code:** [⭐ Code](https://github.com/Z1hanW/CRISP-Real2Sim)

> We introduce CRISP, a method that recovers simulatable human motion and scene geometry from monocular video. Prior work on joint human-scene reconstruction relies on data-driven priors and joint optimization with no physics in the loop, or recover...

</details>

<details>
<summary><b>22. EVOLVE-VLA: Test-Time Training from Environment Feedback for Vision-Language-Action Models</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14666) • [📄 arXiv](https://arxiv.org/abs/2512.14666) • [📥 PDF](https://arxiv.org/pdf/2512.14666)

**💻 Code:** [⭐ Code](https://github.com/showlab/EVOLVE-VLA)

> EVOLVE-VLA is a test-time training framework that enables Vision-Language-Action models to continuously adapt through environment interaction with minimal or no task-specific demonstrations, overcoming the limitations of static supervised finetuni...

</details>

<details>
<summary><b>23. TAT: Task-Adaptive Transformer for All-in-One Medical Image Restoration</b> ⭐ 29</summary>

<br/>

**👥 Authors:** Bingzheng Wei, Jian Liang, Yang Yi, Jiaju, upyzwup

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14550) • [📄 arXiv](https://arxiv.org/abs/2512.14550) • [📥 PDF](https://arxiv.org/pdf/2512.14550)

**💻 Code:** [⭐ Code](https://github.com/Yaziwel/TAT)

> No abstract available.

</details>

<details>
<summary><b>24. Zoom-Zero: Reinforced Coarse-to-Fine Video Understanding via Temporal Zoom-in</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14273) • [📄 arXiv](https://arxiv.org/abs/2512.14273) • [📥 PDF](https://arxiv.org/pdf/2512.14273)

> Project page: https://xiaoqian-shen.github.io/Zoom-Zero/

</details>

<details>
<summary><b>25. Efficient-DLM: From Autoregressive to Diffusion Language Models, and Beyond in Speed</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14067) • [📄 arXiv](https://arxiv.org/abs/2512.14067) • [📥 PDF](https://arxiv.org/pdf/2512.14067)

> Proposes Efficient-DLM: converting autoregressive LMs to fast diffusion LMs via block-wise continuous pretraining and token masking, achieving higher accuracy and throughput than AR and existing dLMs.

</details>

<details>
<summary><b>26. Janus: Disaggregating Attention and Experts for Scalable MoE Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13525) • [📄 arXiv](https://arxiv.org/abs/2512.13525) • [📥 PDF](https://arxiv.org/pdf/2512.13525)

> Paper page: https://arxiv.org/pdf/2512.13525

</details>

<details>
<summary><b>27. RePo: Language Models with Context Re-Positioning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14391) • [📄 arXiv](https://arxiv.org/abs/2512.14391) • [📥 PDF](https://arxiv.org/pdf/2512.14391)

**💻 Code:** [⭐ Code](https://github.com/SakanaAI/repo)

> TL;DR: We want to give LLMs the architectural ability to reorganize input context just like humans do. Our solution is to incorporate a lightweight RePo module to dynamically assign positions before position encoding functions.

</details>

<details>
<summary><b>28. JMMMU-Pro: Image-based Japanese Multi-discipline Multimodal Understanding Benchmark via Vibe Benchmark Construction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14620) • [📄 arXiv](https://arxiv.org/abs/2512.14620) • [📥 PDF](https://arxiv.org/pdf/2512.14620)

> “Bro, Benchmarks like MMMU-Pro are too expensive to build, right?” One month ago: Yes. Now: No 🚀 Proposing Vibe Benchmark Construction! NanoBanana Pro generates VQA itself, and humans only check or lightly edit prompts for regeneration. 🚀Building ...

</details>

<details>
<summary><b>29. MobileWorldBench: Towards Semantic World Modeling For Mobile Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14014) • [📄 arXiv](https://arxiv.org/abs/2512.14014) • [📥 PDF](https://arxiv.org/pdf/2512.14014)

> A benchmark for world modeling of mobile GUI agents.

</details>

<details>
<summary><b>30. Comparative Analysis of LLM Abliteration Methods: A Cross-Architecture Evaluation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** richardyoung

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13655) • [📄 arXiv](https://arxiv.org/abs/2512.13655) • [📥 PDF](https://arxiv.org/pdf/2512.13655)

> TL;DR We benchmark 4 open-source LLM abliteration implementations across 16 instruction-tuned models. Key results: • Coverage differs a lot (Heretic 16/16; DECCP 11/16; ErisForge 9/16; FailSpy 5/16).  ￼ • Single-pass methods preserved capabilities...

</details>

<details>
<summary><b>31. TraPO: A Semi-Supervised Reinforcement Learning Framework for Boosting LLM Reasoning</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zhongqi Chen, Yingfan MA, Xing Zheng, Guangcheng Zhu, Shenzhi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13106) • [📄 arXiv](https://arxiv.org/abs/2512.13106) • [📥 PDF](https://arxiv.org/pdf/2512.13106)

**💻 Code:** [⭐ Code](https://github.com/ShenzhiYang2000/TRAPO)

> We’ve come up with a semi-supervised RLVR training method that uses just a few labeled examples to help pick out trustworthy samples from the unlabeled ones. Feel free to jump in with thoughts or suggestions— all feedback is welcome! 🤗

</details>

<details>
<summary><b>32. UAGLNet: Uncertainty-Aggregated Global-Local Fusion Network with Cooperative CNN-Transformer for Building Extraction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenqi Ren, Shengjie Li, Taotao Li, Dongxiu Liu, Siyuan Yao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12941) • [📄 arXiv](https://arxiv.org/abs/2512.12941) • [📥 PDF](https://arxiv.org/pdf/2512.12941)

**💻 Code:** [⭐ Code](https://github.com/Dstate/UAGLNet)

> UAGLNet Repository: https://github.com/Dstate/UAGLNet Paper: “UAGLNet: Uncertainty-Aggregated Global-Local Fusion Network with Cooperative CNN-Transformer for Building Extraction” ( arXiv:2512.12941 )

</details>

<details>
<summary><b>33. S2D: Sparse-To-Dense Keymask Distillation for Unsupervised Video Instance Segmentation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Timo Ropinski, phermosilla, xeTaiz, lhoyer, leonsick

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14440) • [📄 arXiv](https://arxiv.org/abs/2512.14440) • [📥 PDF](https://arxiv.org/pdf/2512.14440)

**💻 Code:** [⭐ Code](https://github.com/leonsick/s2d)

> Project page: https://leonsick.github.io/s2d Code: https://github.com/leonsick/s2d

</details>

<details>
<summary><b>34. Unveiling User Perceptions in the Generative AI Era: A Sentiment-Driven Evaluation of AI Educational Apps' Role in Digital Transformation of e-Teaching</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Erfan Nourbakhsh, Adeleh Mazaherian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11934) • [📄 arXiv](https://arxiv.org/abs/2512.11934) • [📥 PDF](https://arxiv.org/pdf/2512.11934)

**💻 Code:** [⭐ Code](https://github.com/erfan-nourbakhsh/GenAI-EdSent)

> Hello everyone, I hope you enjoy reading our paper! These are the helpful links: https://arxiv.org/abs/2512.11934 https://github.com/erfan-nourbakhsh/GenAI-EdSent

</details>

<details>
<summary><b>35. Hierarchical Dataset Selection for High-Quality Data Sharing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10952) • [📄 arXiv](https://arxiv.org/abs/2512.10952) • [📥 PDF](https://arxiv.org/pdf/2512.10952)

> How do you decide which datasets to train on when data comes from many noisy, heterogeneous sources? In this work, we formalize dataset selection as its own problem and introduce DaSH (Dataset Selection via Hierarchies), a method that models datas...

</details>

<details>
<summary><b>36. MeViS: A Multi-Modal Dataset for Referring Motion Expression Video Segmentation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10945) • [📄 arXiv](https://arxiv.org/abs/2512.10945) • [📥 PDF](https://arxiv.org/pdf/2512.10945)

> MeViSv2 Dataset, Project Page: https://henghuiding.com/MeViS/

</details>

<details>
<summary><b>37. CoSPlan: Corrective Sequential Planning via Scene Graph Incremental Updates</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yogesh S Rawat, Vibhav Vineet, Akash Kumar, Shresth Grover, ppriyank

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10342) • [📄 arXiv](https://arxiv.org/abs/2512.10342) • [📥 PDF](https://arxiv.org/pdf/2512.10342)

> LLM benchmark on sequence completion (Spoiler: They can't)

</details>

<details>
<summary><b>38. ContextAnyone: Context-Aware Diffusion for Character-Consistent Text-to-Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07328) • [📄 arXiv](https://arxiv.org/abs/2512.07328) • [📥 PDF](https://arxiv.org/pdf/2512.07328)

> ContextAnyone: Context-Aware Diffusion for Character-Consistent Text-to-Video Generation

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 38 |
| 📅 Today | [`2025-12-18.json`](data/daily/2025-12-18.json) | 38 |
| 📆 This Week | [`2025-W50.json`](data/weekly/2025-W50.json) | 125 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 498 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-18 | 38 | [View JSON](data/daily/2025-12-18.json) |
| 📄 2025-12-17 | 41 | [View JSON](data/daily/2025-12-17.json) |
| 📄 2025-12-16 | 21 | [View JSON](data/daily/2025-12-16.json) |
| 📄 2025-12-15 | 25 | [View JSON](data/daily/2025-12-15.json) |
| 📄 2025-12-14 | 25 | [View JSON](data/daily/2025-12-14.json) |
| 📄 2025-12-13 | 24 | [View JSON](data/daily/2025-12-13.json) |
| 📄 2025-12-12 | 21 | [View JSON](data/daily/2025-12-12.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W50 | 125 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 498 | [View JSON](data/monthly/2025-12.json) |

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
