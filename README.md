<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-29-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5246+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">65</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">231</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5246+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** July 18, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. VideoChat3: Fully Open Video MLLM for Efficient and Generalist Video Understanding</b> ⭐ 52</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14935) • [📄 arXiv](https://arxiv.org/abs/2607.14935) • [📥 PDF](https://arxiv.org/pdf/2607.14935)

**💻 Code:** [⭐ Code](https://github.com/MCG-NJU/VideoChat3) • [⭐ Code](https://github.com/huggingface)

> We introduce VideoChat3, a fully open, efficient, and generalist video-centric MLLM. VideoChat3 advances video understanding through two complementary designs. For efficiency, we introduce Inflated 3D Vision Transformer (I3D-ViT) and Adaptive Fram...

</details>

<details>
<summary><b>2. SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning</b> ⭐ 86</summary>

<br/>

**👥 Authors:** Yuhao Shen, Fan Zhang, Zhengxi Lu, Shuo Yang, Jinyang Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14777) • [📄 arXiv](https://arxiv.org/abs/2607.14777) • [📥 PDF](https://arxiv.org/pdf/2607.14777)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jinyangwu/SEED)

> No abstract available.

</details>

<details>
<summary><b>3. LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14952) • [📄 arXiv](https://arxiv.org/abs/2607.14952) • [📥 PDF](https://arxiv.org/pdf/2607.14952)

**💻 Code:** [⭐ Code](https://github.com/MindLab-Research/longstraw) • [⭐ Code](https://github.com/huggingface)

> We present LongStraw, an architecture-aware execution stack for million-token RL post-training under a fixed GPU budget, instantiated with Group Relative Policy Optimization (GRPO). LongStraw evaluates the shared prompt once without automatic diff...

</details>

<details>
<summary><b>4. SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration</b> ⭐ 57</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.15257) • [📄 arXiv](https://arxiv.org/abs/2607.15257) • [📥 PDF](https://arxiv.org/pdf/2607.15257)

**💻 Code:** [⭐ Code](https://github.com/antins-labs/SearchOS) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>5. BadWAM: When World-Action Models Dream Right but Act Wrong</b> ⭐ 32</summary>

<br/>

**👥 Authors:** Xinchao Wang, Xingyi Yang, Qi Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.15207) • [📄 arXiv](https://arxiv.org/abs/2607.15207) • [📥 PDF](https://arxiv.org/pdf/2607.15207)

**💻 Code:** [⭐ Code](https://github.com/LiQiiiii/BadWAM) • [⭐ Code](https://github.com/huggingface)

> BadWAM models World-Action Drift Attacks: a new class of WAM-specific adversarial attacks that use small visual perturbations to break the alignment between what a WAM imagines and what it executes. It instantiates two complementary objectives. Th...

</details>

<details>
<summary><b>6. KeyFrame-Compass: Towards Comprehensive Evaluation of Keyframe-Conditioned Video Generation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14202) • [📄 arXiv](https://arxiv.org/abs/2607.14202) • [📥 PDF](https://arxiv.org/pdf/2607.14202)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cactusqq/KeyFrame-Compass)

> Video generation increasingly relies on keyframe-based workflows, where creators specify a sequence of reference images to guide generation. Although recent models support multi-keyframe conditioning, it remains unclear whether they can faithfully...

</details>

<details>
<summary><b>7. MultiRef-Compass: Towards Comprehensive Evaluation of Multi-Reference-to-Audio-Video Generation</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14189) • [📄 arXiv](https://arxiv.org/abs/2607.14189) • [📥 PDF](https://arxiv.org/pdf/2607.14189)

**💻 Code:** [⭐ Code](https://github.com/zxhhh0201/MultiRef-Compass) • [⭐ Code](https://github.com/huggingface)

> Multi-reference-to-audio-video (MR2AV) generation aims to generate coherent audio-video content conditioned on multiple references and textual instructions. Existing benchmarks mainly focus on text-driven generation, single-reference subject prese...

</details>

<details>
<summary><b>8. From Pixels to States: Rethinking Interactive World Models as Game Engines</b> ⭐ 406</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14076) • [📄 arXiv](https://arxiv.org/abs/2607.14076) • [📥 PDF](https://arxiv.org/pdf/2607.14076)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AlayaLab/WildWorld)

> Building interactive worlds that respond coherently to player actions has long been a shared goal of computer graphics, games, and artificial intelligence. Recent video generative models provide a data-driven route toward this goal by predicting f...

</details>

<details>
<summary><b>9. Concurrent Image Understanding and Generation: Self-Correcting Coupled Markov Jump Processes</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.13188) • [📄 arXiv](https://arxiv.org/abs/2607.13188) • [📥 PDF](https://arxiv.org/pdf/2607.13188)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. UniVR: Thinking in Visual Space for Unified Visual Reasoning</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.12800) • [📄 arXiv](https://arxiv.org/abs/2607.12800) • [📥 PDF](https://arxiv.org/pdf/2607.12800)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/bytedance/UniVR)

> Learning broad world knowledge directly from raw visual data is a fundamental capability of intelligence. We introduce UniVR, the first investigation into simultaneously learning complex reasoning, fine-grained physical dynamics, and long-term pla...

</details>

<details>
<summary><b>11. Spectral Rewiring for Exploration, Purification, and Model Merging</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuxuan Song, Hanlin Wu, Huan-ang Gao, Hongli Yu, Zhilong Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.03065) • [📄 arXiv](https://arxiv.org/abs/2607.03065) • [📥 PDF](https://arxiv.org/pdf/2607.03065)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reinforcement learning has become a standard post-training recipe for large language models, but dense full-parameter updates create two deployment-relevant bottlenecks: suppressed reasoning performance, often reflected by premature saturation of ...

</details>

<details>
<summary><b>12. Video = World + Event Stream</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.15038) • [📄 arXiv](https://arxiv.org/abs/2607.15038) • [📥 PDF](https://arxiv.org/pdf/2607.15038)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Wan Streamer v0.3 learns a persistent world and the events unfolding within it. This general video pretraining brings free-form behavior to real-time full-duplex audio-visual interaction — still 640×368 at 25 fps, with the same ~200 ms model-side ...

</details>

<details>
<summary><b>13. Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tianqing Fang, Boyang Xue, Yi Chen, Hongru Wang, Rui Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.13399) • [📄 arXiv](https://arxiv.org/abs/2607.13399) • [📥 PDF](https://arxiv.org/pdf/2607.13399)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Role: OPD acts as an exploration catalyst, guiding student reasoning via dense token-level signals without raising capability ceilings. Its success hinges on signal quality. Pathologies: This reliance on signal quality exposes two flaws: Student-T...

</details>

<details>
<summary><b>14. RoboTTT: Context Scaling for Robot Policies</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.15275) • [📄 arXiv](https://arxiv.org/abs/2607.15275) • [📥 PDF](https://arxiv.org/pdf/2607.15275)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.15273) • [📄 arXiv](https://arxiv.org/abs/2607.15273) • [📥 PDF](https://arxiv.org/pdf/2607.15273)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Harahan/MeanFlowNFT)

> No abstract available.

</details>

<details>
<summary><b>16. WanSong v1.0 Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14749) • [📄 arXiv](https://arxiv.org/abs/2607.14749) • [📥 PDF](https://arxiv.org/pdf/2607.14749)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> this is open source or close?

</details>

<details>
<summary><b>17. DeepLoop: Depth Scaling for Looped Transformers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mengdi Wang, Quanquan Gu, Jiacheng Guo, Yifan Zhang, Shuzhen Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.13491) • [📄 arXiv](https://arxiv.org/abs/2607.13491) • [📥 PDF](https://arxiv.org/pdf/2607.13491)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> DeepLoop: Depth Scaling for Looped Transformers

</details>

<details>
<summary><b>18. VIABench: A Comprehensive Video Benchmark Collected from Blind Individuals for Visual Impairment Assistance</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14660) • [📄 arXiv](https://arxiv.org/abs/2607.14660) • [📥 PDF](https://arxiv.org/pdf/2607.14660)

**💻 Code:** [⭐ Code](https://github.com/MCG-NJU/VIABench) • [⭐ Code](https://github.com/huggingface)

> We introduce VIABench , a comprehensive, time-aligned video benchmark for evaluating multimodal large language models (MLLMs) in real-world visual assistance scenarios for blind and visually impaired people. VIABench contains 761 videos , 14,526 m...

</details>

<details>
<summary><b>19. Smarter and Cheaper at Once: Byte-Exact KV-Cache Grafting Turns a Frozen Small Model into a Verified-Knowledge Flywheel</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14431) • [📄 arXiv](https://arxiv.org/abs/2607.14431) • [📥 PDF](https://arxiv.org/pdf/2607.14431)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This Sunday on stage for agisummit

</details>

<details>
<summary><b>20. Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Celestine Mendler-Dünner, Andreas Krause, Thomas Kleine Buening, Patrik Wolf

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.15277) • [📄 arXiv](https://arxiv.org/abs/2607.15277) • [📥 PDF](https://arxiv.org/pdf/2607.15277)

**💻 Code:** [⭐ Code](https://github.com/patrikwolf/statistical_self_consistency) • [⭐ Code](https://github.com/huggingface)

> We test whether LLM estimates obey the law of total probability, revealing widespread self-consistency failures and a macro fallacy in which aggregating fine-grained subpopulation estimates outperforms direct population-level prompting.

</details>

<details>
<summary><b>21. RxBrain: Embodied Cognition Foundation Model with Joint Language-Visual Reasoning and Imagination</b> ⭐ 74</summary>

<br/>

**👥 Authors:** Xiaomeng Zhu, Yuchun Guo, Yufei Huang, Mingkang Chen, Haotian Liang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14187) • [📄 arXiv](https://arxiv.org/abs/2607.14187) • [📥 PDF](https://arxiv.org/pdf/2607.14187)

**💻 Code:** [⭐ Code](https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0) • [⭐ Code](https://github.com/huggingface)

> Excited to share Hy-Embodied-RxBrain, a foundation model for embodied cognition that jointly represents plans through language reasoning and visual imagination. RxBrain couples task decomposition, constraints, temporal logic, world-state predictio...

</details>

<details>
<summary><b>22. AsySplat: Efficient Asymmetric 3D Gaussian Splatting for Long-Sequence Scene Modeling</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.10995) • [📄 arXiv](https://arxiv.org/abs/2607.10995) • [📥 PDF](https://arxiv.org/pdf/2607.10995)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zhongyingji/asysplat)

> No abstract available.

</details>

<details>
<summary><b>23. GRASP: GRanularity-Aware Search Policy for Agentic RAG</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ryan Rossi, Franck Dernoncourt, Shantanu Todmal, Jaewook Lee, Varun Gandhi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.10463) • [📄 arXiv](https://arxiv.org/abs/2607.10463) • [📥 PDF](https://arxiv.org/pdf/2607.10463)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>24. SUFLECA: Scaling Up Feature Learning for CAD-to-image Alignment</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jose Luis Sanchez-Lopez, Holger Voos, Javier Civera, Miguel Fernandez-Cortizas, Saad Ejaz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.15058) • [📄 arXiv](https://arxiv.org/abs/2607.15058) • [📥 PDF](https://arxiv.org/pdf/2607.15058)

**💻 Code:** [⭐ Code](https://github.com/snt-arg/SUFLECA) • [⭐ Code](https://github.com/huggingface)

> An adapter over DUNE for generating visual features grounded in 3D object geometry for accurate and fast zero-shot CAD-to-image alignment

</details>

<details>
<summary><b>25. Hierarchical Denoising For Multi-Step Visual Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruibin Yuan, Tianze Zhou, Chak-Wing Mak, Xiaowei Chi, Zezhong Qian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.15278) • [📄 arXiv](https://arxiv.org/abs/2607.15278) • [📥 PDF](https://arxiv.org/pdf/2607.15278)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>26. Token Time Continuous Diffusion for Language Modeling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14106) • [📄 arXiv](https://arxiv.org/abs/2607.14106) • [📥 PDF](https://arxiv.org/pdf/2607.14106)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present a continuous diffusion framework for language modeling, wherein during the forward process noise gets added to tokens at different rates, instead of adding equal amount of noise to all tokens. This enables better conditional learning an...

</details>

<details>
<summary><b>27. Chat2Scenic: An Iterative RAG-Based Framework for Scenario Generation in Autonomous Driving</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.14387) • [📄 arXiv](https://arxiv.org/abs/2607.14387) • [📥 PDF](https://arxiv.org/pdf/2607.14387)

**💻 Code:** [⭐ Code](https://github.com/TUM-AVS/Chat2scenic) • [⭐ Code](https://github.com/huggingface)

> Accepted at 2026 IEEE International Conference on Intelligent Robots and Systems (IROS)

</details>

<details>
<summary><b>28. Rethinking the Evaluation of Harness Evolution for Agents</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.12227) • [📄 arXiv](https://arxiv.org/abs/2607.12227) • [📥 PDF](https://arxiv.org/pdf/2607.12227)

**💻 Code:** [⭐ Code](https://github.com/rethinking-harness-evolution/code) • [⭐ Code](https://github.com/rethinking-harness-evolution) • [⭐ Code](https://github.com/huggingface)

> In this work, we revisit how automatic harness evolution should be evaluated. Existing automatic harness evolution methods  often search over harnesses using feedback from benchmark tasks and then report final performance on the same benchmark. Th...

</details>

<details>
<summary><b>29. On Locality and Length Generalization in Visual Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.09061) • [📄 arXiv](https://arxiv.org/abs/2607.09061) • [📥 PDF](https://arxiv.org/pdf/2607.09061)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Your eyes don't see a whole scene at once — they dart around in a sequence of foveated glimpses. Modern vision models take in the whole image in one shot. That difference decides if a model can generalize to scenarios that are out-of-distribution....

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 29 |
| 📅 Today | [`2026-07-18.json`](data/daily/2026-07-18.json) | 29 |
| 📆 This Week | [`2026-W28.json`](data/weekly/2026-W28.json) | 65 |
| 🗓️ This Month | [`2026-07.json`](data/monthly/2026-07.json) | 231 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-07-18 | 29 | [View JSON](data/daily/2026-07-18.json) |
| 📄 2026-07-17 | 8 | [View JSON](data/daily/2026-07-17.json) |
| 📄 2026-07-16 | 4 | [View JSON](data/daily/2026-07-16.json) |
| 📄 2026-07-15 | 1 | [View JSON](data/daily/2026-07-15.json) |
| 📄 2026-07-14 | 14 | [View JSON](data/daily/2026-07-14.json) |
| 📄 2026-07-13 | 9 | [View JSON](data/daily/2026-07-13.json) |
| 📄 2026-07-12 | 23 | [View JSON](data/daily/2026-07-12.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W28 | 65 | [View JSON](data/weekly/2026-W28.json) |
| 📅 2026-W27 | 79 | [View JSON](data/weekly/2026-W27.json) |
| 📅 2026-W26 | 117 | [View JSON](data/weekly/2026-W26.json) |
| 📅 2026-W25 | 108 | [View JSON](data/weekly/2026-W25.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-07 | 231 | [View JSON](data/monthly/2026-07.json) |
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
