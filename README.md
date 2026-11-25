<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-34-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8277+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">34</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">86</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">549</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8277+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 25, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. ACAP: Asynchronous Coding Agents for Efficient Robot Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31205) • [📄 arXiv](https://arxiv.org/abs/2611.31205) • [📥 PDF](https://arxiv.org/pdf/2611.31205)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We are introducing ACAP: Asynchronous Coding Agents as Policy . A robot policy built from persistent coding agents that reason and act asynchronously. A single coding agent can issue parallel motion commands, but every decision still passes throug...

</details>

<details>
<summary><b>2. Rewirable Circuits in Transformers with Loom Attention</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Xingyuan Yuan, Shengping Li, Qingye Meng, Da Xiao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30634) • [📄 arXiv](https://arxiv.org/abs/2611.30634) • [📥 PDF](https://arxiv.org/pdf/2611.30634)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Caiyun-AI/LoomFormer)

> Hi everyone! Author here — happy to share LoomFormer , along with our JAX/Pallas training code, PyTorch/Triton inference kernels, and 300B-token 2.8B checkpoints (including a matched mHC-Pythia-2.8B baseline). TL;DR: Why layer-wise residual routin...

</details>

<details>
<summary><b>3. PartAgent: Agentic 3D Part Segmentation via Expert Integration and Closed-Loop Refinement</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32554) • [📄 arXiv](https://arxiv.org/abs/2611.32554) • [📥 PDF](https://arxiv.org/pdf/2611.32554)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yangcaoai/PartAgent_Agent_for_3D_Part_Seg)

> We present PartAgent, a training-free agentic framework for 3D part segmentation built upon two complementary modules: (i) Structural–Semantic Expert Integration combines regions predicted by the 3D expert with multiview semantic evidence from the...

</details>

<details>
<summary><b>4. S-Mind: Internalizing Multi-View Geometry for Spatial Intelligence</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28649) • [📄 arXiv](https://arxiv.org/abs/2611.28649) • [📥 PDF](https://arxiv.org/pdf/2611.28649)

**💻 Code:** [⭐ Code](https://github.com/KANABOON1/S-Mind) • [⭐ Code](https://github.com/huggingface)

> Spatial intelligence, the ability to reason about 3D space from visual observations, becomes essential as multimodal large language models (MLLMs) move into the physical world. Beyond perceiving each view, it rests on two cornerstones of human cog...

</details>

<details>
<summary><b>5. LingSante MedGuard Technical Report</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25509) • [📄 arXiv](https://arxiv.org/abs/2611.25509) • [📥 PDF](https://arxiv.org/pdf/2611.25509)

**💻 Code:** [⭐ Code](https://github.com/inclusionAI/LingSante-MedGuard) • [⭐ Code](https://github.com/huggingface)

> We introduce LingSante MedGuard, a medical guardrail that reviews AI responses for safety, compliance, and ethics. It distills supervision from a medical guardrail teacher and six capability experts into a unified model. The 27B model achieves 87....

</details>

<details>
<summary><b>6. Embodied AI Fails the Mirror Test: How a Simple Mirror Breaks Robot Policies</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33010) • [📄 arXiv](https://arxiv.org/abs/2611.33010) • [📥 PDF](https://arxiv.org/pdf/2611.33010)

**💻 Code:** [⭐ Code](https://github.com/iPPLab/robodojo-mirror) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. Antom 3-in-1 Transformer: A Payment Foundation Model</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25403) • [📄 arXiv](https://arxiv.org/abs/2611.25403) • [📥 PDF](https://arxiv.org/pdf/2611.25403)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce Antom 3-in-1 Transformer , a payment foundation model from Ant International. It brings sequence (transaction histories), tabular (features), and graph (relational) representations into one backbone with task-conditioned fusion, built...

</details>

<details>
<summary><b>8. KMed-31B Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28719) • [📄 arXiv](https://arxiv.org/abs/2611.28719) • [📥 PDF](https://arxiv.org/pdf/2611.28719)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> KMed-31B is a medical LLM developed by the NAVER KMed team for on-premises deployment with limited hardware. The report introduces: • MOPD² (Multi-teacher On-policy Delta Distillation), extending OPD² to integrate task-specific specialists into a ...

</details>

<details>
<summary><b>9. Self-Evolving Evaluation Environments for Computer-Use Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30962) • [📄 arXiv](https://arxiv.org/abs/2611.30962) • [📥 PDF](https://arxiv.org/pdf/2611.30962)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> CUA-SEE reconceptualizes evaluation as a dynamic, closed-loop process where tasks, executable verifiers, and agent rollout evidence continuously inform and refine one another.

</details>

<details>
<summary><b>10. PhyCoVE: Benchmarking Physical Counterfactual Video Editing across Diverse Material Dynamics</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29184) • [📄 arXiv](https://arxiv.org/abs/2611.29184) • [📥 PDF](https://arxiv.org/pdf/2611.29184)

**💻 Code:** [⭐ Code](https://github.com/Hammour-steak/PhyCoVE) • [⭐ Code](https://github.com/huggingface)

> 🔥 Can video editing models understand physics? Remove a stone before it drops into a full glass of water, and the water should never overflow. Make it heavier, and the splash should grow. A physical edit changes not only how a scene looks, but how...

</details>

<details>
<summary><b>11. Don't Sleep, Keep Watch: Proactive Perception for Computer-Use Agents</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Haiyan Yin, Long-Kai Huang, Yichen Liu, Jie-Jing Shao, Jiaqi Su

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28947) • [📄 arXiv](https://arxiv.org/abs/2611.28947) • [📥 PDF](https://arxiv.org/pdf/2611.28947)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SuJQ0722/NightWatch)

> No abstract available.

</details>

<details>
<summary><b>12. ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Huichen Wang, Yunzhuo Hao, Yiming Liu, Linjie Li, Jiawei Gu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32183) • [📄 arXiv](https://arxiv.org/abs/2611.32183) • [📥 PDF](https://arxiv.org/pdf/2611.32183)

**💻 Code:** [⭐ Code](https://github.com/ThinkMorph/ThinkMorph-SSL) • [⭐ Code](https://github.com/huggingface)

> Models that reason with intermediate images are usually trained on traces built for one task family at a time. Self-supervised pretext tasks need no such traces, because the original image is the answer. ThinkMorph-SSL turns these tasks into inter...

</details>

<details>
<summary><b>13. LeadVLA: Learning Follower-Aware Route Execution for Robot Leading</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zibo Zhang, Yuchen Zhou, Bingyi Xia, Hanjing Ye, Tianle Zeng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30149) • [📄 arXiv](https://arxiv.org/abs/2611.30149) • [📥 PDF](https://arxiv.org/pdf/2611.30149)

**💻 Code:** [⭐ Code](https://github.com/louiszengCN/LeadVLA_CODE) • [⭐ Code](https://github.com/huggingface)

> LeadVLA will be fully open-sourced! We'll gradually release our model checkpoints, code, and simulator. Stay tuned! 🚀 Follow our GitHub repo for updates: https://github.com/louiszengCN/LeadVLA_CODE For more DEMO, check our project page: https://lo...

</details>

<details>
<summary><b>14. MAGPIE: Agentic Multi-Reference Image Generation via Executable Scene Construction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32462) • [📄 arXiv](https://arxiv.org/abs/2611.32462) • [📥 PDF](https://arxiv.org/pdf/2611.32462)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Paper link: https://arxiv.org/pdf/2611.32462

</details>

<details>
<summary><b>15. Qwen-Image-HD: Autoregressive Diffusion for High Definition Image Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Donghao Zhou, Zikai Zhou, Shengming Yin, Zekai Zhang, Zihao Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32736) • [📄 arXiv](https://arxiv.org/abs/2611.32736) • [📥 PDF](https://arxiv.org/pdf/2611.32736)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>16. RoomTokens: Masked Diffusion over Structured Tokens for Indoor Scene Synthesis</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30219) • [📄 arXiv](https://arxiv.org/abs/2611.30219) • [📥 PDF](https://arxiv.org/pdf/2611.30219)

**💻 Code:** [⭐ Code](https://github.com/3dlg-hcvc/RoomToken) • [⭐ Code](https://github.com/huggingface)

> Our goal is to generate complete, physically plausible 3D indoor scenes from an architecture plan. MDiRT, our masked diffusion model, generates RoomTokens across height slices, with each token encoding semantic class, instance color, and facing. T...

</details>

<details>
<summary><b>17. Reinforcing Diffusion Models by Instantaneous Trajectory Distribution Matching</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30986) • [📄 arXiv](https://arxiv.org/abs/2611.30986) • [📥 PDF](https://arxiv.org/pdf/2611.30986)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Luo-Yihong/iTDM)

> Hi everyone! 👋 Excited to share our new work iTDM (Instantaneous Trajectory Distribution Matching) for reinforcing diffusion and flow-matching models! TL;DR: Existing forward diffusion RL methods (enabling fast deterministic ODE rollouts without t...

</details>

<details>
<summary><b>18. On-Policy Codec Distillation for Efficient All-in-One Video Generation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zezhou Cheng, Matheus Gadelha, Oindrila Saha, Kevin Blackburn-Matzen, Boyang Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31184) • [📄 arXiv](https://arxiv.org/abs/2611.31184) • [📥 PDF](https://arxiv.org/pdf/2611.31184)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Kiteretsu77/CodecOPSD)

> Most condition tokens in video generation are redundant, and a video codec can tell where they are. 🚀 We are thrilled to introduce CodecOPSD, which drops 66% of them with almost no quality loss. ✅ Works on MiniMax-H3 Ref2VA, MiniMax-H3-Image, Cosm...

</details>

<details>
<summary><b>19. VenusOCR2.0: Towards Efficient and Effective Document Parsing</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32617) • [📄 arXiv](https://arxiv.org/abs/2611.32617) • [📥 PDF](https://arxiv.org/pdf/2611.32617)

**💻 Code:** [⭐ Code](https://github.com/flow3rdown/VenusOCR) • [⭐ Code](https://github.com/huggingface)

> https://github.com/flow3rdown/VenusOCR

</details>

<details>
<summary><b>20. LoopDVR: Looped Transformer for Efficient Dense Visual Recognition</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32067) • [📄 arXiv](https://arxiv.org/abs/2611.32067) • [📥 PDF](https://arxiv.org/pdf/2611.32067)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Intellindust-AI-Lab/LoopDVR)

> Achieving exceptional parameter efficiency via looped Transformer.

</details>

<details>
<summary><b>21. Residual Guidance for Real-Time Interactive Video Generation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32820) • [📄 arXiv](https://arxiv.org/abs/2611.32820) • [📥 PDF](https://arxiv.org/pdf/2611.32820)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/nahyeonkaty/residual-guidance)

> Autoregressive video models often ignore a new prompt given mid-generation, as the cached context pulls the video back. Residual Guidance is a training-free, plug-and-play fix that makes the video follow the new prompt while staying continuous.

</details>

<details>
<summary><b>22. PINE: Point, Instance and Normal Estimation in One-Step Diffusion Model</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29389) • [📄 arXiv](https://arxiv.org/abs/2611.29389) • [📥 PDF](https://arxiv.org/pdf/2611.29389)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIGeeksGroup/PINE)

> 🚀 We release PINE, a one-step Omni Diffusion Transformer that jointly predicts the point map, surface normals, and instance map of an image in a single forward pass! 🧩 PINE integrates 3D geometry and semantics via token streams with stream masks, ...

</details>

<details>
<summary><b>23. AdProducer: Towards Agentic Advertising Video Generation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31466) • [📄 arXiv](https://arxiv.org/abs/2611.31466) • [📥 PDF](https://arxiv.org/pdf/2611.31466)

**💻 Code:** [⭐ Code](https://github.com/lky-ang/AdProducer) • [⭐ Code](https://github.com/huggingface)

> Recent advances in video generation have made automated advertising production increasingly feasible. However, an effective advertisement must faithfully represent the product and communicate its selling points through compelling visual storytelli...

</details>

<details>
<summary><b>24. LoopedWAM: Looped World Action Models for Elastic Inference</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25595) • [📄 arXiv](https://arxiv.org/abs/2611.25595) • [📥 PDF](https://arxiv.org/pdf/2611.25595)

**💻 Code:** [⭐ Code](https://github.com/anhdao69/LoopedWAM) • [⭐ Code](https://github.com/huggingface)

> TL;DR: LoopedWAM is a 0.58B world action model that loops a shared 6-block core, running 30 layers from 12 stored blocks. No robot pretraining: 96.1% LIBERO, 91.9% RoboTwin 2.0, 7.4× less GPU memory than Fast-WAM. One checkpoint (LoopedWAM-E) lets...

</details>

<details>
<summary><b>25. Coding4Games: Can Modern Coding Agents Play Modern Video Games?</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31974) • [📄 arXiv](https://arxiv.org/abs/2611.31974) • [📥 PDF](https://arxiv.org/pdf/2611.31974)

**💻 Code:** [⭐ Code](https://github.com/GVCLab/Coding4Games) • [⭐ Code](https://github.com/huggingface)

> Games, especially modern AAA games, challenge language models to perceive complex scenes, control actions, and make decisions over long horizons. Existing game benchmarks primarily test language models as computer-use agents that query a language ...

</details>

<details>
<summary><b>26. Interpretable Prototype-Conditioned Adaptation for Weather Forecasting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25848) • [📄 arXiv](https://arxiv.org/abs/2611.25848) • [📥 PDF](https://arxiv.org/pdf/2611.25848)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project page: https://zhixuanli.github.io/projects/li2026protocast

</details>

<details>
<summary><b>27. Can an Agent Trust Its Own Zoom? Cross-Magnification Provenance Verification for Whole-Slide Images</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhongxing Xu, Jianxiang Qi, Zizhuo Gong, Yiran Song, Yan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29150) • [📄 arXiv](https://arxiv.org/abs/2611.29150) • [📥 PDF](https://arxiv.org/pdf/2611.29150)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Yan2266336/PyraFault)

> Visual agents navigate whole-slide images through magnified observations associated with claimed locations. However, a visually plausible observation does not necessarily originate from its claimed location, particularly across magnifications and ...

</details>

<details>
<summary><b>28. Align the Pair, Not the Sample: Representation Alignment for Faster and Better One-Step Generato</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Han Zhang, Aoran Wang, Encheng Su, Xuantuo Huang, Mingxiao Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29547) • [📄 arXiv](https://arxiv.org/abs/2611.29547) • [📥 PDF](https://arxiv.org/pdf/2611.29547)

**💻 Code:** [⭐ Code](https://github.com/Mingxiao-Li/RePrime) • [⭐ Code](https://github.com/huggingface)

> One-step generation is fast at inference — but painfully slow to train. We found a placement for REPA that cuts Priming's training by 2× and its OT post-training by 4×, while improving FID. Meet RePrime.

</details>

<details>
<summary><b>29. Clinical3D-Bench: Evaluating and Improving Spatial Intelligence of Multimodal LLMs in Hospital Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32094) • [📄 arXiv](https://arxiv.org/abs/2611.32094) • [📥 PDF](https://arxiv.org/pdf/2611.32094)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Clinical3D-Bench tests whether MLLMs can reason about 3D space in hospital rooms: 3,368 multiple-choice questions on 569 photorealistic walkthrough clips across 8 question types, with exact answers computed from the 3D scenes and per-frame instanc...

</details>

<details>
<summary><b>30. Factorized Context Attention: From Emulated to Native Grid Probing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32423) • [📄 arXiv](https://arxiv.org/abs/2611.32423) • [📥 PDF](https://arxiv.org/pdf/2611.32423)

**💻 Code:** [⭐ Code](https://github.com/mohammad2012191/FCA) • [⭐ Code](https://github.com/huggingface)

> Grid probing is attention in disguise, so we run it inside the model. Encode a book or an hour of video once, then answer each new question in as little as a tenth of the time. Training-free, frozen models.

</details>

<details>
<summary><b>31. RoMo Hands: A Large Scale Richly Organized Text to Hand Motion Dataset</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31766) • [📄 arXiv](https://arxiv.org/abs/2611.31766) • [📥 PDF](https://arxiv.org/pdf/2611.31766)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RoMoDataset/motion-toolbox)

> Hi everyone! We're excited to share RoMo Hands (NeurIPS 2026) — a large-scale, text-paired 3D bimanual hand motion dataset and benchmark. Highlights: Scale: 541,929 in-the-wild bimanual MANO sequences ( 884 hours , 95.5M frames at 30 fps) — >10× l...

</details>

<details>
<summary><b>32. EEGJev: A System One Decision Model and Benchmark for Multi-Task EEG Question Answering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kazunori D. Yamada, Dengzhe Hou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32100) • [📄 arXiv](https://arxiv.org/abs/2611.32100) • [📥 PDF](https://arxiv.org/pdf/2611.32100)

**💻 Code:** [⭐ Code](https://github.com/dengzhe-hou/EEGJev) • [⭐ Code](https://github.com/huggingface)

> We introduce EEGJev, a decision model that uses EEG and a question to score candidate answers in one pass, and EEGJev-Bench, which covers six question types across 19 public EEG sources. The benchmark includes paired questions on the same recordin...

</details>

<details>
<summary><b>33. Do learned image captioning metrics generalize? Auditing duplicated samples in human-judgment benchmarks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Longjiang Guo, Chunlong Xia, Jin Li, hutmorning

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30488) • [📄 arXiv](https://arxiv.org/abs/2611.30488) • [📥 PDF](https://arxiv.org/pdf/2611.30488)

**💻 Code:** [⭐ Code](https://github.com/morninghut/ic-dataset-audit) • [⭐ Code](https://github.com/huggingface)

> Hi everyone! We're excited to share our recent work: Do Learned Image Captioning Metrics Generalize? Auditing Duplicated Samples in Human-Judgment Benchmarks How reliable are the reported gains of learned image captioning metrics? To find out, we ...

</details>

<details>
<summary><b>34. Escriba-OCR: A 1B Vision-Language Model for Error-Preserving Transcription of Handwritten Portuguese Essays</b> ⭐ 0</summary>

<br/>

**👥 Authors:** tardellirs

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.29874) • [📄 arXiv](https://arxiv.org/abs/2611.29874) • [📥 PDF](https://arxiv.org/pdf/2611.29874)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/tardellirs/escriba-ocr)

> Escriba-OCR-1B is a 1B-parameter OCR model that transcribes handwritten Brazilian Portuguese essays exactly as the student wrote them: misspellings, line breaks, struck-through words and insertions included. Feedback tools and research on learner ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 34 |
| 📅 Today | [`2026-11-25.json`](data/daily/2026-11-25.json) | 34 |
| 📆 This Week | [`2026-W47.json`](data/weekly/2026-W47.json) | 86 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 549 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-25 | 34 | [View JSON](data/daily/2026-11-25.json) |
| 📄 2026-11-24 | 40 | [View JSON](data/daily/2026-11-24.json) |
| 📄 2026-11-23 | 12 | [View JSON](data/daily/2026-11-23.json) |
| 📄 2026-11-22 | 50 | [View JSON](data/daily/2026-11-22.json) |
| 📄 2026-11-21 | 50 | [View JSON](data/daily/2026-11-21.json) |
| 📄 2026-11-20 | 18 | [View JSON](data/daily/2026-11-20.json) |
| 📄 2026-11-19 | 27 | [View JSON](data/daily/2026-11-19.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W47 | 86 | [View JSON](data/weekly/2026-W47.json) |
| 📅 2026-W46 | 184 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 549 | [View JSON](data/monthly/2026-11.json) |
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
