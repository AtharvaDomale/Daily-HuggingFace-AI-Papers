<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-26-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-849+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">26</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">70</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">111</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">849+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 08, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03252) • [📄 arXiv](https://arxiv.org/abs/2601.03252) • [📥 PDF](https://arxiv.org/pdf/2601.03252)

> Depth Beyond Pixels 🚀 We Introduce InfiniDepth — casting monocular depth estimation as a neural implicit field. 🔍 Arbitrary-Resolution 📐 Accurate Metric Depth 📷 Single-View NVS under large viewpoints shifts Arxiv: https://arxiv.org/abs/2601.03252 ...

</details>

<details>
<summary><b>2. MOSS Transcribe Diarize: Accurate Transcription with Speaker Diarization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01554) • [📄 arXiv](https://arxiv.org/abs/2601.01554) • [📥 PDF](https://arxiv.org/pdf/2601.01554)

> MOSS Transcribe Diarize 🎙️ We introduce MOSS Transcribe Diarize — a unified multimodal model for Speaker-Attributed, Time-Stamped Transcription (SATS) . 🔍 End-to-end SATS in a single pass (transcription + speaker attribution + timestamps) 🧠 128k c...

</details>

<details>
<summary><b>3. LTX-2: Efficient Joint Audio-Visual Foundation Model</b> ⭐ 922</summary>

<br/>

**👥 Authors:** kvochko, jacobitterman, nisan, benibraz, yoavhacohen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03233) • [📄 arXiv](https://arxiv.org/abs/2601.03233) • [📥 PDF](https://arxiv.org/pdf/2601.03233)

**💻 Code:** [⭐ Code](https://github.com/Lightricks/LTX-2)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API 3MDiT: Unified Tri-Modal Diffusion Transformer for Text-Driven Synchronized...

</details>

<details>
<summary><b>4. SciEvalKit: An Open-source Evaluation Toolkit for Scientific General Intelligence</b> ⭐ 56</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22334) • [📄 arXiv](https://arxiv.org/abs/2512.22334) • [📥 PDF](https://arxiv.org/pdf/2512.22334)

**💻 Code:** [⭐ Code](https://github.com/InternScience/SciEvalKit)

> SciEvalKit is a unified benchmarking toolkit for evaluating AI models across scientific disciplines, focusing on core scientific intelligence competencies and supporting diverse domains from physics to materials science.

</details>

<details>
<summary><b>5. UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision</b> ⭐ 25</summary>

<br/>

**👥 Authors:** Lin-Chen, lovesnowbest, YuZeng260, CostaliyA, Hungryyan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03193) • [📄 arXiv](https://arxiv.org/abs/2601.03193) • [📥 PDF](https://arxiv.org/pdf/2601.03193)

**💻 Code:** [⭐ Code](https://github.com/Hungryyan1/UniCorn)

> UniCorn, a simple yet elegant self-improvement framework that eliminates the need for external data or teacher supervision.

</details>

<details>
<summary><b>6. NitroGen: An Open Foundation Model for Generalist Gaming Agents</b> ⭐ 1.44k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02427) • [📄 arXiv](https://arxiv.org/abs/2601.02427) • [📥 PDF](https://arxiv.org/pdf/2601.02427)

**💻 Code:** [⭐ Code](https://github.com/MineDojo/NitroGen)

> NitroGen is a vision-action foundation model trained on 40k hours of gameplay across 1,000+ games, enabling cross-game generalization with behavior cloning and benchmarking, achieving strong unseen-game transfer.

</details>

<details>
<summary><b>7. SOP: A Scalable Online Post-Training System for Vision-Language-Action Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03044) • [📄 arXiv](https://arxiv.org/abs/2601.03044) • [📥 PDF](https://arxiv.org/pdf/2601.03044)

> 🚀 Website: https://www.agibot.com/research/sop We introduce SOP for online post-training of generalist VLAs in the real world — unlocking persistent, reliable deployment of generalist robots in physical environments. 🔁 36 hours of continuous cloth...

</details>

<details>
<summary><b>8. DreamStyle: A Unified Framework for Video Stylization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02785) • [📄 arXiv](https://arxiv.org/abs/2601.02785) • [📥 PDF](https://arxiv.org/pdf/2601.02785)

> DreamStyle unifies text-, style-image-, and first-frame-guided video stylization on an I2V backbone, using LoRA with token-specific up matrices to improve style consistency and video quality.

</details>

<details>
<summary><b>9. MiMo-V2-Flash Technical Report</b> ⭐ 957</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02780) • [📄 arXiv](https://arxiv.org/abs/2601.02780) • [📥 PDF](https://arxiv.org/pdf/2601.02780)

**💻 Code:** [⭐ Code](https://github.com/XiaomiMiMo/MiMo-V2-Flash)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Xiaomi MiMo-VL-Miloco Technical Report (2025) Seer: Online Context Learning...

</details>

<details>
<summary><b>10. CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aojun Lu, Junjie Xie, Shuhang Chen, JacobYuan, Yunqiu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01874) • [📄 arXiv](https://arxiv.org/abs/2601.01874) • [📥 PDF](https://arxiv.org/pdf/2601.01874)

> Project page: https://shchen233.github.io/cogflow/

</details>

<details>
<summary><b>11. Digital Twin AI: Opportunities and Challenges from Large Language Models to World Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Yao Su, vztu, ZihanJia, fjchendp, roz322

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01321) • [📄 arXiv](https://arxiv.org/abs/2601.01321) • [📥 PDF](https://arxiv.org/pdf/2601.01321)

**💻 Code:** [⭐ Code](https://github.com/rongzhou7/Awesome-Digital-Twin-AI/tree/main)

> This paper systematically analyzes AI integration in Digital Twins through a four-stage framework (modeling → mirroring → intervention → autonomous management), covering LLMs, foundation models, world models, and intelligent agents across 11 appli...

</details>

<details>
<summary><b>12. WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02439) • [📄 arXiv](https://arxiv.org/abs/2601.02439) • [📥 PDF](https://arxiv.org/pdf/2601.02439)

> WebGym creates a large, non-stationary visual web task suite and scalable RL pipeline, enabling fast trajectory rollout and improved vision-language agent performance on unseen websites.

</details>

<details>
<summary><b>13. Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fatemeh Askari, Sadegh Mohammadian, Mohammadali Banayeeanzade, Hosein Hasani, safinal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02989) • [📄 arXiv](https://arxiv.org/abs/2601.02989) • [📥 PDF](https://arxiv.org/pdf/2601.02989)

> 🔢 Overcoming Transformer Depth Limits in Counting Tasks LLMs often fail at counting not because they aren't smart, but because of architectural depth constraints 🚧. We propose a simple, effective System-2 strategy 🧩 that decomposes counting tasks ...

</details>

<details>
<summary><b>14. Muses: Designing, Composing, Generating Nonexistent Fantasy 3D Creatures without Training</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03256) • [📄 arXiv](https://arxiv.org/abs/2601.03256) • [📥 PDF](https://arxiv.org/pdf/2601.03256)

**💻 Code:** [⭐ Code](https://github.com/luhexiao/Muses)

> Project page: https://luhexiao.github.io/Muses.github.io/ Code: https://github.com/luhexiao/Muses

</details>

<details>
<summary><b>15. FFP-300K: Scaling First-Frame Propagation for Generalizable Video Editing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Donghao Luo, yanweifuture, chengjie-wang, ChengmingX, ScarletAce

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01720) • [📄 arXiv](https://arxiv.org/abs/2601.01720) • [📥 PDF](https://arxiv.org/pdf/2601.01720)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API LoVoRA: Text-guided and Mask-free Video Object Removal and Addition with Le...

</details>

<details>
<summary><b>16. OpenRT: An Open-Source Red Teaming Framework for Multimodal LLMs</b> ⭐ 112</summary>

<br/>

**👥 Authors:** Yang Yao, Yixu Wang, Juncheng Li, Yunhao Chen, xinwang22

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01592) • [📄 arXiv](https://arxiv.org/abs/2601.01592) • [📥 PDF](https://arxiv.org/pdf/2601.01592)

**💻 Code:** [⭐ Code](https://github.com/AI45Lab/OpenRT)

> Even State-of-the-Art Models Fail to Hold Ground Against Sophisticated Adversaries. Our comprehensive evaluation highlights two key findings. (1) A clear stratification in defense capability: Top-tier models such as Claude Haiku 4.5, GPT-5.2, and ...

</details>

<details>
<summary><b>17. MindWatcher: Toward Smarter Multimodal Tool-Integrated Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23412) • [📄 arXiv](https://arxiv.org/abs/2512.23412) • [📥 PDF](https://arxiv.org/pdf/2512.23412)

**💻 Code:** [⭐ Code](https://github.com/TIMMY-CHAN/MindWatcher)

> In this work, we introduce MindWatcher, a TIR agent integrating interleaved thinking and multimodal chain-of-thought (CoT) reasoning. MindWatcher can autonomously decide whether and how to invoke diverse tools and coordinate their use, without rel...

</details>

<details>
<summary><b>18. The Sonar Moment: Benchmarking Audio-Language Models in Audio Geo-Localization</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03227) • [📄 arXiv](https://arxiv.org/abs/2601.03227) • [📥 PDF](https://arxiv.org/pdf/2601.03227)

**💻 Code:** [⭐ Code](https://github.com/Rising0321/AGL1K)

> We found the sonar moment in audio language models. We propose the task of audio geo-localization. And amazingly, Gemini 3 Pro can reach the distance error of less than 55km for 25%  samples.

</details>

<details>
<summary><b>19. X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sai Rithwik Reddy Chirra, Shashivardhan Reddy Koppula, Mohammad Zia Ur Rehman, shwetankssingh, UVSKKR

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03194) • [📄 arXiv](https://arxiv.org/abs/2601.03194) • [📥 PDF](https://arxiv.org/pdf/2601.03194)

**💻 Code:** [⭐ Code](https://github.com/ziarehman30/X-MuTeST)

> Hate speech detection on social media faces challenges in both accuracy and explainability, especially for underexplored Indic languages. We propose a novel explainability-guided training framework, X-MuTeST (explainable Multilingual haTe Speech d...

</details>

<details>
<summary><b>20. Parallel Latent Reasoning for Sequential Recommendation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuning Jiang, Jian Wu, Wen Chen, Xu Chen, TangJiakai5704

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03153) • [📄 arXiv](https://arxiv.org/abs/2601.03153) • [📥 PDF](https://arxiv.org/pdf/2601.03153)

> Parallel Latent Reasoning (PLR): Sequential Recommendation with Parallel Reasoning 🔥 📉 Depth-only reasoning often hits performance plateaus—PLR mitigates this with parallel latent reasoning. Core Innovation ✨ 🎯 Learnable trigger tokens: Build para...

</details>

<details>
<summary><b>21. Unified Thinker: A General Reasoning Modular Core for Image Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yue Cao, Hanqing Yang, Jijin Hu, Qiang Zhou, Sashuai Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03127) • [📄 arXiv](https://arxiv.org/abs/2601.03127) • [📥 PDF](https://arxiv.org/pdf/2601.03127)

> reasoning-based image generation and editing

</details>

<details>
<summary><b>22. Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02996) • [📄 arXiv](https://arxiv.org/abs/2601.02996) • [📥 PDF](https://arxiv.org/pdf/2601.02996)

**💻 Code:** [⭐ Code](https://github.com/cisnlp/multilingual-latent-reasoner)

> https://github.com/cisnlp/multilingual-latent-reasoner

</details>

<details>
<summary><b>23. ExposeAnyone: Personalized Audio-to-Expression Diffusion Models Are Robust Zero-Shot Face Forgery Detectors</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vladislav Golyanik, Toshihiko Yamasaki, mapooon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02359) • [📄 arXiv](https://arxiv.org/abs/2601.02359) • [📥 PDF](https://arxiv.org/pdf/2601.02359)

> Detecting deepfakes with generative AI. We introduce ExposeAnyone — a paradigm shift in face forgery detection! 🔍️ Fully self-supervised approach 🥇 Best average AUC on traditional deepfake benchmarks 💪 Best AUC even on Sora2 by OpenAI 💢 Strong Rob...

</details>

<details>
<summary><b>24. AceFF: A State-of-the-Art Machine Learning Potential for Small Molecules</b> ⭐ 458</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00581) • [📄 arXiv](https://arxiv.org/abs/2601.00581) • [📥 PDF](https://arxiv.org/pdf/2601.00581)

**💻 Code:** [⭐ Code](https://github.com/torchmd/torchmd-net)

> AceFF: A State-of-the-Art Machine Learning Potential for Small Molecules We introduce AceFF, a pre-trained machine learning interatomic potential (MLIP) optimized for small molecule drug discovery. While MLIPs have emerged as efficient alternative...

</details>

<details>
<summary><b>25. U-Net-Like Spiking Neural Networks for Single Image Dehazing</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Peng Li, Yulong Xiao, Mingzhe Liu, Huibin Li, FengShaner

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23950) • [📄 arXiv](https://arxiv.org/abs/2512.23950) • [📥 PDF](https://arxiv.org/pdf/2512.23950)

**💻 Code:** [⭐ Code](https://github.com/HaoranLiu507/DehazeSNN)

> Title: DehazeSNN — U-Net-like Spiking Neural Networks for Single Image Dehazing Short summary: DehazeSNN integrates a U-Net architecture with Spiking Neural Networks to reduce compute while achieving competitive dehazing results. Code: github.com/...

</details>

<details>
<summary><b>26. Steerability of Instrumental-Convergence Tendencies in LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** j-hoscilowic

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01584) • [📄 arXiv](https://arxiv.org/abs/2601.01584) • [📥 PDF](https://arxiv.org/pdf/2601.01584)

**💻 Code:** [⭐ Code](https://github.com/j-hoscilowicz/instrumental_steering/)

> This paper measures how easily “instrumental-convergence” behaviors (e.g., shutdown avoidance, self-replication) in LLMs can be amplified or suppressed by simple steering, and argues that the common claim “as AI capability (often glossed as ‘intel...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 26 |
| 📅 Today | [`2026-01-08.json`](data/daily/2026-01-08.json) | 26 |
| 📆 This Week | [`2026-W01.json`](data/weekly/2026-W01.json) | 70 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 111 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-08 | 26 | [View JSON](data/daily/2026-01-08.json) |
| 📄 2026-01-07 | 24 | [View JSON](data/daily/2026-01-07.json) |
| 📄 2026-01-06 | 13 | [View JSON](data/daily/2026-01-06.json) |
| 📄 2026-01-05 | 7 | [View JSON](data/daily/2026-01-05.json) |
| 📄 2026-01-04 | 7 | [View JSON](data/daily/2026-01-04.json) |
| 📄 2026-01-03 | 7 | [View JSON](data/daily/2026-01-03.json) |
| 📄 2026-01-02 | 20 | [View JSON](data/daily/2026-01-02.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W01 | 70 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 111 | [View JSON](data/monthly/2026-01.json) |
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
