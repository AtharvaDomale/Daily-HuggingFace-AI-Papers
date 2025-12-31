<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-31-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-738+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">52</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">787</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">738+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 31, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23447) • [📄 arXiv](https://arxiv.org/abs/2512.23447) • [📥 PDF](https://arxiv.org/pdf/2512.23447)

> We propose the Expert-Router Coupling (ERC) loss, a lightweight auxiliary loss that tightly couples the router’s decisions with expert capabilities. Unlike prior coupling methods that scale with the number of tokens (often millions per batch), the...

</details>

<details>
<summary><b>2. LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation</b> ⭐ 81</summary>

<br/>

**👥 Authors:** Steffi Chern, Jiadi Su, Bohao Tang, Zhulin Hu, Ethan Chern

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23576) • [📄 arXiv](https://arxiv.org/abs/2512.23576) • [📥 PDF](https://arxiv.org/pdf/2512.23576)

**💻 Code:** [⭐ Code](https://github.com/GAIR-NLP/LiveTalk)

> Real-time video generation via diffusion is essential for building general-purpose multimodal interactive AI systems. However, the simultaneous denoising of all video frames with bidirectional attention via an iterative process in diffusion models...

</details>

<details>
<summary><b>3. Yume-1.5: A Text-Controlled Interactive World Generation Model</b> ⭐ 426</summary>

<br/>

**👥 Authors:** Kaining Ying, Xiaojie Xu, Chuanhao Li, Zhen Li, Xiaofeng Mao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22096) • [📄 arXiv](https://arxiv.org/abs/2512.22096) • [📥 PDF](https://arxiv.org/pdf/2512.22096)

**💻 Code:** [⭐ Code](https://github.com/stdstu12/YUME)

> Recent approaches have demonstrated the promise of using diffusion models to generate interactive and explorable worlds. However, most of these methods face critical challenges such as excessively large parameter sizes, reliance on lengthy inferen...

</details>

<details>
<summary><b>4. SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22322) • [📄 arXiv](https://arxiv.org/abs/2512.22322) • [📥 PDF](https://arxiv.org/pdf/2512.22322)

> We introduce SmartSnap , a paradigm shift that transforms GUI agents📱💻🤖 from passive task executors into proactive self-verifiers. By empowering agents to curate their own evidence of success through the 3C Principles (Completeness, Conciseness, C...

</details>

<details>
<summary><b>5. Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation</b> ⭐ 94</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23705) • [📄 arXiv](https://arxiv.org/abs/2512.23705) • [📥 PDF](https://arxiv.org/pdf/2512.23705)

**💻 Code:** [⭐ Code](https://github.com/Daniellli/DKT)

> Abstract Transparent objects remain notoriously hard for perception systems: refraction, reflection and transmission break the assumptions behind stereo, ToF and purely discriminative monocular depth, causing holes and temporally unstable estimate...

</details>

<details>
<summary><b>6. Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Po-Fan Yu, Chi-Wei Hsiao, Zhixiang Wang, Chin-Yang Lin, Hau-Shiang Shiu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23709) • [📄 arXiv](https://arxiv.org/abs/2512.23709) • [📥 PDF](https://arxiv.org/pdf/2512.23709)

> Diffusion-based video super-resolution (VSR) methods achieve strong perceptual quality but remain impractical for latency-sensitive settings due to reliance on future frames and expensive multi-step denoising. We propose Stream-DiffVSR, a causally...

</details>

<details>
<summary><b>7. Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone</b> ⭐ 41</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22615) • [📄 arXiv](https://arxiv.org/abs/2512.22615) • [📥 PDF](https://arxiv.org/pdf/2512.22615)

**💻 Code:** [⭐ Code](https://github.com/DreamLM/Dream-VLX)

> Building on the success of Dream 7B, we introduce Dream-VL and Dream-VLA, open VL and VLA models that fully unlock discrete diffusion’s advantages in long-horizon planning, bidirectional reasoning, and parallel action generation for multimodal tasks.

</details>

<details>
<summary><b>8. SpotEdit: Selective Region Editing in Diffusion Transformers</b> ⭐ 48</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22323) • [📄 arXiv](https://arxiv.org/abs/2512.22323) • [📥 PDF](https://arxiv.org/pdf/2512.22323)

**💻 Code:** [⭐ Code](https://github.com/Biangbiang0321/SpotEdit)

> 🎯 SpotEdit: Edit Only What Needs to Be Edited Why regenerate the entire background just to add a scarf to the dog in your photo? This is a frustrating limitation facing many current AI image editing models. Existing methods typically perform a ful...

</details>

<details>
<summary><b>9. GRAN-TED: Generating Robust, Aligned, and Nuanced Text Embedding for Diffusion Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15560) • [📄 arXiv](https://arxiv.org/abs/2512.15560) • [📥 PDF](https://arxiv.org/pdf/2512.15560)

> The text encoder is a critical component of text-to-image and text-to-video diffusion models, fundamentally determining the semantic fidelity of the generated content. However, its development has been hindered by two major challenges: the lack of...

</details>

<details>
<summary><b>10. Act2Goal: From World Model To General Goal-conditioned Policy</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23541) • [📄 arXiv](https://arxiv.org/abs/2512.23541) • [📥 PDF](https://arxiv.org/pdf/2512.23541)

> Project page: https://act2goal.github.io/ Abs: Specifying robotic manipulation tasks in a manner that is both expressive and precise remains a central challenge. While visual goals provide a compact and unambiguous task specification, existing goa...

</details>

<details>
<summary><b>11. Web World Models</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23676) • [📄 arXiv](https://arxiv.org/abs/2512.23676) • [📥 PDF](https://arxiv.org/pdf/2512.23676)

**💻 Code:** [⭐ Code](https://github.com/Princeton-AI2-Lab/Web-World-Models)

> In this work, we introduce the Web World Model (WWM), a middle ground where world state and physics are implemented in ordinary web code to ensure logical consistency, while large language models generate context, narratives, and high-level decisi...

</details>

<details>
<summary><b>12. DiRL: An Efficient Post-Training Framework for Diffusion Language Models</b> ⭐ 113</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22234) • [📄 arXiv](https://arxiv.org/abs/2512.22234) • [📥 PDF](https://arxiv.org/pdf/2512.22234)

**💻 Code:** [⭐ Code](https://github.com/OpenMOSS/DiRL)

> Diffusion Language Models (dLLMs) have emerged as promising alternatives to Auto-Regressive (AR) models. While recent efforts have validated their pre-training potential and accelerated inference speeds, the post-training landscape for dLLMs remai...

</details>

<details>
<summary><b>13. Training AI Co-Scientists Using Rubric Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23707) • [📄 arXiv](https://arxiv.org/abs/2512.23707) • [📥 PDF](https://arxiv.org/pdf/2512.23707)

> How to train language models at generating research plans given diverse open-ended research goals?

</details>

<details>
<summary><b>14. Video-BrowseComp: Benchmarking Agentic Video Research on Open Web</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kaixin Liang, Minghao Qin, Xiangrui Liu, Yan Shu, Zhengyang Liang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23044) • [📄 arXiv](https://arxiv.org/abs/2512.23044) • [📥 PDF](https://arxiv.org/pdf/2512.23044)

> Introduces Video-BrowseComp, a benchmark of 210 open-web agentic video questions requiring temporal visual evidence to test proactive video reasoning in grounded retrieval.

</details>

<details>
<summary><b>15. OmniAgent: Audio-Guided Active Perception Agent for Omnimodal Audio-Video Understanding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jian Liu, Weiqiang Wang, Bohan Yu, Wenjie Du, Keda Tao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23646) • [📄 arXiv](https://arxiv.org/abs/2512.23646) • [📥 PDF](https://arxiv.org/pdf/2512.23646)

> Website: https://kd-tao.github.io/OmniAgent/

</details>

<details>
<summary><b>16. YOLO-Master: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23273) • [📄 arXiv](https://arxiv.org/abs/2512.23273) • [📥 PDF](https://arxiv.org/pdf/2512.23273)

> Existing Real-Time Object Detection (RTOD) methods commonly adopt YOLO-like architectures for their favorable trade-off between accuracy and speed. However, these models rely on static dense computation that applies uniform processing to all input...

</details>

<details>
<summary><b>17. VL-LN Bench: Towards Long-horizon Goal-oriented Navigation with Active Dialogs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xihui Liu, Jinming Xu, Meng Wei, Shaohao Zhu, Wensi Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22342) • [📄 arXiv](https://arxiv.org/abs/2512.22342) • [📥 PDF](https://arxiv.org/pdf/2512.22342)

> VL-LN Bench: Towards Long-horizon Goal-oriented Navigation with Active Dialogs

</details>

<details>
<summary><b>18. Nested Browser-Use Learning for Agentic Information Seeking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23647) • [📄 arXiv](https://arxiv.org/abs/2512.23647) • [📥 PDF](https://arxiv.org/pdf/2512.23647)

> Information-seeking (IS) agents have achieved strong performance across a range of wide and deep search tasks, yet their tool use remains largely restricted to API-level snippet retrieval and URL-based page fetching, limiting access to the richer ...

</details>

<details>
<summary><b>19. SurgWorld: Learning Surgical Robot Policies from Videos via World Modeling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23162) • [📄 arXiv](https://arxiv.org/abs/2512.23162) • [📥 PDF](https://arxiv.org/pdf/2512.23162)

> Proposes SurgWorld world model to learn surgical robot policies from unlabeled videos via synthetic pseudokinematics, enabling data-efficient VLA policies from SATA data.

</details>

<details>
<summary><b>20. Monadic Context Engineering</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22431) • [📄 arXiv](https://arxiv.org/abs/2512.22431) • [📥 PDF](https://arxiv.org/pdf/2512.22431)

**💻 Code:** [⭐ Code](https://github.com/yifanzhang-pro/monadic-context-engineering)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API A Declarative Language for Building And Orchestrating LLM-Powered Agent Wor...

</details>

<details>
<summary><b>21. An Information Theoretic Perspective on Agentic System Design</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21720) • [📄 arXiv](https://arxiv.org/abs/2512.21720) • [📥 PDF](https://arxiv.org/pdf/2512.21720)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Design and Evaluation of Cost-Aware PoQ for Decentralized LLM Inference (20...

</details>

<details>
<summary><b>22. Quantile Rendering: Efficiently Embedding High-dimensional Feature on 3D Gaussian Splatting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20927) • [📄 arXiv](https://arxiv.org/abs/2512.20927) • [📥 PDF](https://arxiv.org/pdf/2512.20927)

> project page: https://jaesung-choe.github.io/qrender/index.html

</details>

<details>
<summary><b>23. Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation</b> ⭐ 21</summary>

<br/>

**👥 Authors:** Yuheng Ji, Zixiao Wang, Yijie Xu, Sixiang Chen, Huajie Tan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23703) • [📄 arXiv](https://arxiv.org/abs/2512.23703) • [📥 PDF](https://arxiv.org/pdf/2512.23703)

**💻 Code:** [⭐ Code](https://github.com/FlagOpen/Robo-Dopamine)

> Upload Robo-Dopamine

</details>

<details>
<summary><b>24. ProGuard: Towards Proactive Multimodal Safeguard</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jing Shao, Lu Sheng, Chenyang Si, Lijun Li, Shaohan Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23573) • [📄 arXiv](https://arxiv.org/abs/2512.23573) • [📥 PDF](https://arxiv.org/pdf/2512.23573)

**💻 Code:** [⭐ Code](https://github.com/yushaohan/ProGuard)

> The rapid evolution of generative models has led to a continuous emergence of multimodal safety risks, exposing the limitations of existing defense methods. To address these challenges, we propose ProGuard, a vision-language proactive guard that i...

</details>

<details>
<summary><b>25. Bridging Your Imagination with Audio-Video Generation via a Unified Director</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23222) • [📄 arXiv](https://arxiv.org/abs/2512.23222) • [📥 PDF](https://arxiv.org/pdf/2512.23222)

> UniMAGE unifies script writing and keyframe generation for long-context video creation using Mixture-of-Transformers and a two-stage interleaving/disentangling training paradigm.

</details>

<details>
<summary><b>26. Knot Forcing: Taming Autoregressive Video Diffusion Models for Real-time Infinite Interactive Portrait Animation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21734) • [📄 arXiv](https://arxiv.org/abs/2512.21734) • [📥 PDF](https://arxiv.org/pdf/2512.21734)

> We propose Knot Forcing , a streaming framework for real-time portrait animation that enables high-fidelity, temporally consistent, and interactive video generation from dynamic inputs such as reference images and driving signals. Unlike diffusion...

</details>

<details>
<summary><b>27. KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23236) • [📄 arXiv](https://arxiv.org/abs/2512.23236) • [📥 PDF](https://arxiv.org/pdf/2512.23236)

> Excited to share our recent work on KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta . We designed, implemented, and deployed KernelEvolve to optimize a wide variety of production recommendation models across g...

</details>

<details>
<summary><b>28. Introducing TrGLUE and SentiTurca: A Comprehensive Benchmark for Turkish General Language Understanding and Sentiment Analysis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22100) • [📄 arXiv](https://arxiv.org/abs/2512.22100) • [📥 PDF](https://arxiv.org/pdf/2512.22100)

> We proudly present our brand new Turkish NLP benchmarking sets, TrGLUE. Unlike previous work, TrGLUE is not based on translation of original GLUE tasks but tailored for Turkish vocabulary, syntax, semantics and cultural heritage.

</details>

<details>
<summary><b>29. Self-Evaluation Unlocks Any-Step Text-to-Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22374) • [📄 arXiv](https://arxiv.org/abs/2512.22374) • [📥 PDF](https://arxiv.org/pdf/2512.22374)

> No abstract available.

</details>

<details>
<summary><b>30. Shape of Thought: When Distribution Matters More than Correctness in Reasoning Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22255) • [📄 arXiv](https://arxiv.org/abs/2512.22255) • [📥 PDF](https://arxiv.org/pdf/2512.22255)

> Training on synthetic CoT traces, even with wrong final answers, improves reasoning due to aligning with the model's distribution and leveraging partial reasoning steps, outperforming human-annotated data. In our paper we explore this interesting ...

</details>

<details>
<summary><b>31. Reverse Personalization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nicu Sebe, Tuomas Varanka, Han-Wei Kung

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22984) • [📄 arXiv](https://arxiv.org/abs/2512.22984) • [📥 PDF](https://arxiv.org/pdf/2512.22984)

**💻 Code:** [⭐ Code](https://github.com/hanweikung/reverse-personalization)

> https://github.com/hanweikung/reverse-personalization

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 31 |
| 📅 Today | [`2025-12-31.json`](data/daily/2025-12-31.json) | 31 |
| 📆 This Week | [`2025-W52.json`](data/weekly/2025-W52.json) | 52 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 787 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-31 | 31 | [View JSON](data/daily/2025-12-31.json) |
| 📄 2025-12-30 | 14 | [View JSON](data/daily/2025-12-30.json) |
| 📄 2025-12-29 | 7 | [View JSON](data/daily/2025-12-29.json) |
| 📄 2025-12-28 | 7 | [View JSON](data/daily/2025-12-28.json) |
| 📄 2025-12-27 | 7 | [View JSON](data/daily/2025-12-27.json) |
| 📄 2025-12-26 | 17 | [View JSON](data/daily/2025-12-26.json) |
| 📄 2025-12-25 | 18 | [View JSON](data/daily/2025-12-25.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
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
