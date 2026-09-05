<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-31-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6206+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">31</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">90</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">78</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6206+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** September 05, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Compile by Training: Turning Natural-Language Specifications into Local Neural Functions</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04199) • [📄 arXiv](https://arxiv.org/abs/2609.04199) • [📥 PDF](https://arxiv.org/pdf/2609.04199)

**💻 Code:** [⭐ Code](https://github.com/programasweights/compile-by-training) • [⭐ Code](https://github.com/huggingface)

> What if LLMs built reusable tools instead of solving the same task over and over? Program-as-Weights (PAW) compiles a natural-language function description into a small neural program that runs on a shared 0.6B local interpreter. Our original comp...

</details>

<details>
<summary><b>2. Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04148) • [📄 arXiv](https://arxiv.org/abs/2609.04148) • [📥 PDF](https://arxiv.org/pdf/2609.04148)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API NexForge: Scaling Agent Capabilities through Requirement-Driven Task Synthe...

</details>

<details>
<summary><b>3. LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.03796) • [📄 arXiv](https://arxiv.org/abs/2609.03796) • [📥 PDF](https://arxiv.org/pdf/2609.03796)

**💻 Code:** [⭐ Code](https://github.com/inclusionAI/LLaDA-Image) • [⭐ Code](https://github.com/huggingface)

> LLaDA-Image is a competitive 6B-parameter open-source unified image generation and editing model family. It includes LLaDA-Image, a 50-step Base model for high-quality text-to-image generation and instruction-guided editing, and LLaDA-Image-Turbo,...

</details>

<details>
<summary><b>4. Knowing When Not to Reuse: Conditional Experience Transfer in Autonomous LLM Post-Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Abudukelimu Wuerkaixi, Weiqing Li, zhangywlfh, wenfengfwf, spongebob0715

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26730) • [📄 arXiv](https://arxiv.org/abs/2608.26730) • [📥 PDF](https://arxiv.org/pdf/2608.26730)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We are excited to share Knowing When Not to Reuse: Conditional Experience Transfer in Autonomous LLM Post-Training . Past post-training successes do not necessarily remain valid after the parent model, data, or training stage changes. We formulate...

</details>

<details>
<summary><b>5. Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.03430) • [📄 arXiv](https://arxiv.org/abs/2609.03430) • [📥 PDF](https://arxiv.org/pdf/2609.03430)

**💻 Code:** [⭐ Code](https://github.com/SalesforceAIResearch/Random-Attention) • [⭐ Code](https://github.com/huggingface)

> Large language models achieve superior performance on tasks that require extended reasoning, but long chains of thought make the KV cache a severe memory bottleneck. Existing KV cache compression methods share one paradigm: score each cached token...

</details>

<details>
<summary><b>6. LatentPress: Context Compression Beyond Text and Vision</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hejian Sang, Zhengze Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01507) • [📄 arXiv](https://arxiv.org/abs/2609.01507) • [📥 PDF](https://arxiv.org/pdf/2609.01507)

**💻 Code:** [⭐ Code](https://github.com/HJSang/LatentPress) • [⭐ Code](https://github.com/huggingface)

> https://github.com/HJSang/LatentPress

</details>

<details>
<summary><b>7. Rethinking On-Policy Distillation of Large Language Models II: One Training Example</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04172) • [📄 arXiv](https://arxiv.org/abs/2609.04172) • [📥 PDF](https://arxiv.org/pdf/2609.04172)

**💻 Code:** [⭐ Code](https://github.com/Thinking-Space/One-Shot-OPD) • [⭐ Code](https://github.com/huggingface)

> We investigate the data side and the training dynamics of on-policy distillation (OPD), and try to answer how far the training set of OPD can be reduced, and find that a single training example already induces most of the states a full dataset vis...

</details>

<details>
<summary><b>8. Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04098) • [📄 arXiv](https://arxiv.org/abs/2609.04098) • [📥 PDF](https://arxiv.org/pdf/2609.04098)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi everyone, TL;DR: the community assumption for hybrid LLMs has been that the recurrent half (Gated DeltaNet) is too fragile for 4-bit — early quants of Qwen3.8-27B all kept it at 8/16-bit. We quantized all 496 linear layers to NVFP4 W4A4 — GDN a...

</details>

<details>
<summary><b>9. Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04196) • [📄 arXiv](https://arxiv.org/abs/2609.04196) • [📥 PDF](https://arxiv.org/pdf/2609.04196)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04201) • [📄 arXiv](https://arxiv.org/abs/2609.04201) • [📥 PDF](https://arxiv.org/pdf/2609.04201)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Scalable online 3D reconstruction on kilometer-scale sequences, with only ~1% extra parameters on a frozen backbone trained in 8 hours on a single GPU.

</details>

<details>
<summary><b>11. Editable Visual Design</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04034) • [📄 arXiv](https://arxiv.org/abs/2609.04034) • [📥 PDF](https://arxiv.org/pdf/2609.04034)

**💻 Code:** [⭐ Code](https://github.com/yejy53/Editable-Design) • [⭐ Code](https://github.com/huggingface)

> Github: https://github.com/yejy53/Editable-Design

</details>

<details>
<summary><b>12. The Missing Temporal Link: Temporal Context Routing for Script-Driven Audio-Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaojie Li, Donghao Zhou, Haozhe Wang, zqw, starry0929

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02367) • [📄 arXiv](https://arxiv.org/abs/2609.02367) • [📥 PDF](https://arxiv.org/pdf/2609.02367)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This work introduces Temporal Context Routing (TCR), a new module for script‑driven audio‑video generation. It aligns script timestamps with audio‑video timeline, precisely controlling shot transitions and dialogue timestamps. Project page will be...

</details>

<details>
<summary><b>13. Beyond Retrieval: Progressive Latent Memory Evolution for Streaming Video Understanding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rongxing Ding, Xiaobin Hu, Ling Xing, Guangming Yao, Hongyu Qu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04131) • [📄 arXiv](https://arxiv.org/abs/2609.04131) • [📥 PDF](https://arxiv.org/pdf/2609.04131)

**💻 Code:** [⭐ Code](https://github.com/quhongyu/LatentStream) • [⭐ Code](https://github.com/huggingface)

> LatentStream advances streaming video understanding from external “store-and-retrieve” memory toward “retrieve-and-internalize,” progressively consolidating historical evidence into a compact, evolving latent working memory. By combining hierarchi...

</details>

<details>
<summary><b>14. CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04083) • [📄 arXiv](https://arxiv.org/abs/2609.04083) • [📥 PDF](https://arxiv.org/pdf/2609.04083)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi, we publish new embedding models which perform well on fine-grained multimodal retrieval and compositional reasoning tasks.

</details>

<details>
<summary><b>15. DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04094) • [📄 arXiv](https://arxiv.org/abs/2609.04094) • [📥 PDF](https://arxiv.org/pdf/2609.04094)

**💻 Code:** [⭐ Code](https://github.com/IBM/draco) • [⭐ Code](https://github.com/huggingface)

> Reinforcement Learning from Verifiable Rewards works well when a task has a programmatic checker, but most long-horizon agent domains have none. We work in the outcome-blind setting, where ground-truth success signals are not available. Multi-crit...

</details>

<details>
<summary><b>16. PACE: Towards Surfacing Hidden Conflicts in User Requests</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.03293) • [📄 arXiv](https://arxiv.org/abs/2609.03293) • [📥 PDF](https://arxiv.org/pdf/2609.03293)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> PACE is a novel dataset for evaluating whether personalized assistants can recognize hidden conflicts between seemingly reasonable user requests and contextual information stored in a user-specific knowledge base. These conflicts arise when releva...

</details>

<details>
<summary><b>17. Last Translation Benchmark</b> ⭐ 48</summary>

<br/>

**👥 Authors:** tarudesu, zetrozky, hannayukhymenko, pinzhenchen, zouhar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04173) • [📄 arXiv](https://arxiv.org/abs/2609.04173) • [📥 PDF](https://arxiv.org/pdf/2609.04173)

**💻 Code:** [⭐ Code](https://github.com/zouharvi/last-translation-benchmark) • [⭐ Code](https://github.com/huggingface)

> We just released the Last Translation Benchmark paper. In a massive crowdsourcing effort we collected 3456 unique hard-to-translate examples that break state-of-the-art translation models, and which can be used for more reliable evaluation. Machin...

</details>

<details>
<summary><b>18. RealSWE: A Compositional Evaluation of Coding Agents under Realistic User Requests</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27831) • [📄 arXiv](https://arxiv.org/abs/2608.27831) • [📥 PDF](https://arxiv.org/pdf/2608.27831)

**💻 Code:** [⭐ Code](https://github.com/gyuhyeong-x/RealSWE) • [⭐ Code](https://github.com/huggingface)

> RealSWE is a benchmark and framework built around 381 multi-variant task families, each preserving the same task and gold patch while varying information composition and linguistic style.

</details>

<details>
<summary><b>19. WorldReward: Reward Modeling for Camera-Conditioned World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junshu Tang, Zehan Wang, Yibin Wang, yuhangzang, yujieouo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.03952) • [📄 arXiv](https://arxiv.org/abs/2609.03952) • [📥 PDF](https://arxiv.org/pdf/2609.03952)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> WorldReward: Reward Modeling for Camera-Conditioned World Models

</details>

<details>
<summary><b>20. FlashRender: Few-Step Generative Rendering via Camera-Controlled Video MeanFlow</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.03563) • [📄 arXiv](https://arxiv.org/abs/2609.03563) • [📥 PDF](https://arxiv.org/pdf/2609.03563)

**💻 Code:** [⭐ Code](https://github.com/byeongjun-park/FlashRender) • [⭐ Code](https://github.com/huggingface)

> FlashRender retakes an input video along a target camera trajectory in seconds, using only 4 NFE.

</details>

<details>
<summary><b>21. Using Grounded Theory for Agent Behavior Analysis at Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.30391) • [📄 arXiv](https://arxiv.org/abs/2608.30391) • [📥 PDF](https://arxiv.org/pdf/2608.30391)

**💻 Code:** [⭐ Code](https://github.com/ZhuoranLu/Qual-Agent-Behavior-Analysis) • [⭐ Code](https://github.com/huggingface)

> EMNLP Findings 2026

</details>

<details>
<summary><b>22. Environment Evolution for Terminal Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04128) • [📄 arXiv](https://arxiv.org/abs/2609.04128) • [📥 PDF](https://arxiv.org/pdf/2609.04128)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Envs-FORGE: Frontier-Optimized Reward-Grounded Environment Synthesis for Ag...

</details>

<details>
<summary><b>23. Principia: Relational Physics Tests for Video Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Anand Bhattad, Venkatesh Babu Radhakrishnan, Shivam Tripathi, Varun Varma Thozhiyoor

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.04200) • [📄 arXiv](https://arxiv.org/abs/2609.04200) • [📥 PDF](https://arxiv.org/pdf/2609.04200)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Learning Explicit Physical Parameter Control and Benchmarking for Video Gen...

</details>

<details>
<summary><b>24. Select, Compress, Reinvest: A Controlled Study of Visual-Token Allocation in Long-Video MLLMs</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Prakh25

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.03820) • [📄 arXiv](https://arxiv.org/abs/2609.03820) • [📥 PDF](https://arxiv.org/pdf/2609.03820)

**💻 Code:** [⭐ Code](https://github.com/codeprakhar25/omp-keyframe-sampling) • [⭐ Code](https://github.com/huggingface)

> Long-video MLLMs can't process every frame; an hour at 1fps is 3,600 images, and most systems keep a small fixed slice. This paper holds selection, spatial compression, and reinvestment separate and varies one at a time, across 6 selection rules, ...

</details>

<details>
<summary><b>25. Let Confidence Change, Not the Prediction: Prediction-Preserving Repair for Post-hoc Calibration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ikbeom Jang, Haejun Chung, HYUDHKIM

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.01072) • [📄 arXiv](https://arxiv.org/abs/2609.01072) • [📥 PDF](https://arxiv.org/pdf/2609.01072)

**💻 Code:** [⭐ Code](https://github.com/labhai/CORD) • [⭐ Code](https://github.com/huggingface)

> Accuracy can hide how often post-hoc calibration changes a model’s top-1 prediction. CORD is a post-fit adapter that repairs the calibrated probability vector to recover the original top-1 exactly while preserving the calibrated conditional distri...

</details>

<details>
<summary><b>26. Percolation Dynamics in Optimization : Variance Cascades and Discrete Scale Invariance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Suvrit Sra, Sai Niranjan Ramachandran

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02373) • [📄 arXiv](https://arxiv.org/abs/2609.02373) • [📥 PDF](https://arxiv.org/pdf/2609.02373)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> SGD collapses deep neural networks toward sparse, low-rank representations generated by architectural symmetry. We show how this collapse progresses over time, and that the mechanism extends to Adam and AdamW.

</details>

<details>
<summary><b>27. RoboTok: An Internet-Scale Data Engine for Human Demonstration Retrieval and Dexterous Manipulation Learning</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.03199) • [📄 arXiv](https://arxiv.org/abs/2609.03199) • [📥 PDF](https://arxiv.org/pdf/2609.03199)

**💻 Code:** [⭐ Code](https://github.com/Rice-RobotPI-Lab/RoboTok-Code) • [⭐ Code](https://github.com/huggingface)

> 🤖 Robot manipulation data just got way cheaper (and it’s open source!) 🚀 Introducing RoboTok… an internet-scale data engine for human demonstration video retrieval and dexterous manipulation learning. RoboTok uses a single human demonstration vide...

</details>

<details>
<summary><b>28. QCell: Recombining and Aligning Cell Queries for Overlapping Instance Segmentation</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Dmytro Fishman, Anton Popov, YaroslavPrytula

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.29253) • [📄 arXiv](https://arxiv.org/abs/2608.29253) • [📥 PDF](https://arxiv.org/pdf/2608.29253)

**💻 Code:** [⭐ Code](https://github.com/SlavkoPrytula/QCell) • [⭐ Code](https://github.com/huggingface)

> QCell: Query-Based Cell Instance Segmentation ( BMVC 2026 ) In this work, we present: QCell , a novel query-based model for overlapping cell instance segmentation ⭐️ A new overlapping object segmentation dataset: Organoids 🔥 🔗 GitHub : https://git...

</details>

<details>
<summary><b>29. VeriPhy: Agentic Physical Reasoning for World Model Evaluation and Refinement</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jing Shi, Xuan Shen, Chongjian Ge, Yuchen Zhu, Wenzhuo Xu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.03153) • [📄 arXiv](https://arxiv.org/abs/2609.03153) • [📥 PDF](https://arxiv.org/pdf/2609.03153)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Plans You Can Check: Verifier-Grounded Learning of an Open-Weight Planner f...

</details>

<details>
<summary><b>30. A Common Measure of Communication for Speech Brain-Computer Interfaces</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.02887) • [📄 arXiv](https://arxiv.org/abs/2609.02887) • [📥 PDF](https://arxiv.org/pdf/2609.02887)

**💻 Code:** [⭐ Code](https://github.com/neural-processing-lab/OVMI) • [⭐ Code](https://github.com/huggingface)

> Speech BCI results are becoming increasingly hard to compare: one system reports WER over 125k words, another accuracy over 50, often under very different experimental settings. OVMI tries to put these results on a common communication scale by ac...

</details>

<details>
<summary><b>31. Locked at the Entrance, Open Inside: Where RLVR Narrows the Solution Space</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.29188) • [📄 arXiv](https://arxiv.org/abs/2608.29188) • [📥 PDF](https://arxiv.org/pdf/2608.29188)

**💻 Code:** [⭐ Code](https://github.com/ershiyidian/early-branch-locking) • [⭐ Code](https://github.com/huggingface)

> RLVR-induced solution collapse is an access failure, not an execution failure: models lose reasoning diversity at initial computational entrance while retaining latent downstream capability. Reasoning breadth is lost at the door, not inside the room.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 31 |
| 📅 Today | [`2026-09-05.json`](data/daily/2026-09-05.json) | 31 |
| 📆 This Week | [`2026-W35.json`](data/weekly/2026-W35.json) | 90 |
| 🗓️ This Month | [`2026-09.json`](data/monthly/2026-09.json) | 78 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-09-05 | 31 | [View JSON](data/daily/2026-09-05.json) |
| 📄 2026-09-04 | 12 | [View JSON](data/daily/2026-09-04.json) |
| 📄 2026-09-03 | 12 | [View JSON](data/daily/2026-09-03.json) |
| 📄 2026-09-02 | 16 | [View JSON](data/daily/2026-09-02.json) |
| 📄 2026-09-01 | 7 | [View JSON](data/daily/2026-09-01.json) |
| 📄 2026-08-31 | 12 | [View JSON](data/daily/2026-08-31.json) |
| 📄 2026-08-30 | 23 | [View JSON](data/daily/2026-08-30.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W35 | 90 | [View JSON](data/weekly/2026-W35.json) |
| 📅 2026-W34 | 173 | [View JSON](data/weekly/2026-W34.json) |
| 📅 2026-W33 | 213 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-09 | 78 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |

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
