<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-24-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4285+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">24</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">91</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">664</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4285+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 29, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29250) • [📄 arXiv](https://arxiv.org/abs/2605.29250) • [📥 PDF](https://arxiv.org/pdf/2605.29250)

**💻 Code:** [⭐ Code](https://github.com/JinheonBaek/OmniRetrieval) • [⭐ Code](https://github.com/huggingface)

> Real-world answers live in text, tables, and graphs. And, OmniRetrieval reaches them all through one natural-language interface, meeting each source on its own terms.

</details>

<details>
<summary><b>2. AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29801) • [📄 arXiv](https://arxiv.org/abs/2605.29801) • [📥 PDF](https://arxiv.org/pdf/2605.29801)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security

</details>

<details>
<summary><b>3. How LoRA Remembers? A Parametric Memory Law for LLM Finetuning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30260) • [📄 arXiv](https://arxiv.org/abs/2605.30260) • [📥 PDF](https://arxiv.org/pdf/2605.30260)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/ParametricMemoryLaw) • [⭐ Code](https://github.com/huggingface)

> We uncover a quantitative law of parametric memory in LLMs, showing that exact recall emerges through a sharp probability threshold and can be significantly improved with threshold-guided fine-tuning.

</details>

<details>
<summary><b>4. Native Audio-Visual Alignment for Generation</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30073) • [📄 arXiv](https://arxiv.org/abs/2605.30073) • [📥 PDF](https://arxiv.org/pdf/2605.30073)

**💻 Code:** [⭐ Code](https://github.com/ernie-research/NAVA) • [⭐ Code](https://github.com/huggingface)

> NAVA is a 6.3B-parameter Native Audio-Visual Alignment framework for joint audio-video generation. To overcome the limitations of existing dual-tower and unified paradigms, NAVA employs an Align-then-Fuse MMDiT architecture that first establishes ...

</details>

<details>
<summary><b>5. When Should Models Change Their Minds? Contextual Belief Management in Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30219) • [📄 arXiv](https://arxiv.org/abs/2605.30219) • [📥 PDF](https://arxiv.org/pdf/2605.30219)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We show that long-horizon reasoning in LLMs fundamentally depends on contextual belief management — knowing when to update, preserve, or ignore information — and that explicit belief-state optimization dramatically improves this ability.

</details>

<details>
<summary><b>6. LaRA: Layer-wise Representation Analysis for Detecting Data Contamination in RL Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29888) • [📄 arXiv](https://arxiv.org/abs/2605.29888) • [📥 PDF](https://arxiv.org/pdf/2605.29888)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper introduces LaRA (Layer-wise Representation Analysis), a framework for detecting data contamination in RL post-trained LLMs by examining how internal representations change across layers rather than relying on output-level signals such a...

</details>

<details>
<summary><b>7. GenClaw: Code-Driven Agentic Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30248) • [📄 arXiv](https://arxiv.org/abs/2605.30248) • [📥 PDF](https://arxiv.org/pdf/2605.30248)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Image generation models have evolved from text-conditioned pixel synthesis toward multimodal agents endowed with visual comprehension and tool invocation capabilities. Yet, existing agents remain at the mercy of underlying black-box image models. ...

</details>

<details>
<summary><b>8. UI-KOBE: Knowledge-Oriented Behavior Exploration for Lightweight Graph-Guided GUI Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rui Liu, Jinpeng Chen, Xinyu Fu, Han Xiao, Yuxiang Chai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29534) • [📄 arXiv](https://arxiv.org/abs/2605.29534) • [📥 PDF](https://arxiv.org/pdf/2605.29534)

**💻 Code:** [⭐ Code](https://github.com/YuxiangChai/UI-KOBE) • [⭐ Code](https://github.com/huggingface)

> Code at: https://github.com/YuxiangChai/UI-KOBE

</details>

<details>
<summary><b>9. AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zhen Fang, Avery Nie, Shiting Huang, Ziao Zhang, KouShi2

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27995) • [📄 arXiv](https://arxiv.org/abs/2605.27995) • [📥 PDF](https://arxiv.org/pdf/2605.27995)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/StoKou/repo-asynctool)

> Large language model (LLM)-based agents have shown strong capabilities in using external tools to solve complex tasks. However, existing evaluations often overlook the temporal dimension of tool use, especially the impact of tool response latency,...

</details>

<details>
<summary><b>10. LiteCoder-Terminal: Scaling Long-Horizon Terminal Environments for Learning Language Agents</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Yaojie Lu, Boxi Cao, Xinyu Lu, Kaiqi Zhang, Xiaoxuan Peng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29559) • [📄 arXiv](https://arxiv.org/abs/2605.29559) • [📥 PDF](https://arxiv.org/pdf/2605.29559)

**💻 Code:** [⭐ Code](https://github.com/icip-cas/LiteCoder) • [⭐ Code](https://github.com/huggingface)

> Add paper.

</details>

<details>
<summary><b>11. Learning A Unified Risk Map for Autonomous Driving in Partially Observable Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22189) • [📄 arXiv](https://arxiv.org/abs/2605.22189) • [📥 PDF](https://arxiv.org/pdf/2605.22189)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Learning A Unified Risk Map for Autonomous Driving in Partially Observable Environments  (IEEE RA-L) — How should a self-driving car reason about danger it can't see? This work (IEEE RA-L) introduces a unified spatiotemporal risk map that fuses tr...

</details>

<details>
<summary><b>12. LoMo: Local Modality Substitution for Deeper Vision-Language Fusion</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30265) • [📄 arXiv](https://arxiv.org/abs/2605.30265) • [📥 PDF](https://arxiv.org/pdf/2605.30265)

**💻 Code:** [⭐ Code](https://github.com/Maplebb/LoMo) • [⭐ Code](https://github.com/huggingface)

> LoMo is a lightweight, architecture-agnostic data curation paradigm that mitigates the cross-carrier modality gap in VLMs by recasting selected text spans into rendered images, turning standard SFT into an implicit cross-modal alignment objective ...

</details>

<details>
<summary><b>13. minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models</b> ⭐ 185</summary>

<br/>

**👥 Authors:** Yimin Chen, Zihan Zhou, Bokai Yan, Hongzhou Zhu, Min Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30263) • [📄 arXiv](https://arxiv.org/abs/2605.30263) • [📥 PDF](https://arxiv.org/pdf/2605.30263)

**💻 Code:** [⭐ Code](https://github.com/shengshu-ai/minWM) • [⭐ Code](https://github.com/huggingface)

> Github: https://github.com/shengshu-ai/minWM

</details>

<details>
<summary><b>14. Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30280) • [📄 arXiv](https://arxiv.org/abs/2605.30280) • [📥 PDF](https://arxiv.org/pdf/2605.30280)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Qwen-VLA is a unified embodied foundation model that extends Qwen's vision-language stack to support continuous action and trajectory generation across diverse robot platforms, tasks, and environments.

</details>

<details>
<summary><b>15. SmartDirector: Keyframe-Conditioned Cinematic Video Generation with Narrative Pacing Control</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27891) • [📄 arXiv](https://arxiv.org/abs/2605.27891) • [📥 PDF](https://arxiv.org/pdf/2605.27891)

**💻 Code:** [⭐ Code](https://github.com/Orange-3DV-Team/SmartDirector) • [⭐ Code](https://github.com/huggingface)

> SmartDirector introduces a two-stage framework that empowers cinematic video generation with precise narrative pacing and high-fidelity detail recovery by leveraging a novel Multi-Chunk VAE strategy to circumvent temporal causal constraints.

</details>

<details>
<summary><b>16. NeuROK: Generative 4D Neural Object Kinematics</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shangzhe Wu, Yunzhi Zhang, Yue Gao, Guangzhao He, Chen Geng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30347) • [📄 arXiv](https://arxiv.org/abs/2605.30347) • [📥 PDF](https://arxiv.org/pdf/2605.30347)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> NeuROK provides a data-driven kinematic state parameterization for object-centric systems, enabling the efficient generation of simulative 4D object dynamics without explicit physical models or prior inductive bias.

</details>

<details>
<summary><b>17. Beyond 3D VQAs: Injecting 3D Spatial Priors into Vision-Language Models for Enhanced Geometric Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Joseph Tighe, Yi Ma, Manchen Wang, Shengyi Qian, Chun-Hsiao Yeh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30231) • [📄 arXiv](https://arxiv.org/abs/2605.30231) • [📥 PDF](https://arxiv.org/pdf/2605.30231)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>18. AdaState: Self-Evolving Anchors for Streaming Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30349) • [📄 arXiv](https://arxiv.org/abs/2605.30349) • [📥 PDF](https://arxiv.org/pdf/2605.30349)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Streaming video diffusion models have a structural blind spot: they anchor on the first frame. Because that frame sits in the cleanest, most error-free slot of the KV cache, attention collapses onto this reference; suppressing dynamics and locking...

</details>

<details>
<summary><b>19. Parallax: Parameterized Local Linear Attention for Language Modeling</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29157) • [📄 arXiv](https://arxiv.org/abs/2510.01450) • [📥 PDF](https://arxiv.org/pdf/2605.29157)

**💻 Code:** [⭐ Code](https://github.com/Yifei-Zuo/Parallax) • [⭐ Code](https://github.com/Yifei-Zuo/FlashLLA) • [⭐ Code](https://github.com/huggingface)

> LLA paper: https://arxiv.org/abs/2510.01450 LLA kernel was published in the github: https://github.com/Yifei-Zuo/FlashLLA

</details>

<details>
<summary><b>20. OmniInteract: Benchmarking Real-World Streaming Interaction for Real-Time Omnimodal Assistants</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jinpeng Chen, Yang Bo, Annan Wang, Xueying Li, Xudong Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26485) • [📄 arXiv](https://arxiv.org/abs/2605.26485) • [📥 PDF](https://arxiv.org/pdf/2605.26485)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Lucky-Lance/OmniInteract)

> OmniInteract is a streaming benchmark for real-time omnimodal LLMs, evaluated through their native online inference over continuous audio-visual streams. User queries and ambient sounds live in the audio track, visual events live in the video, and...

</details>

<details>
<summary><b>21. MoZoo:Unleashing Video Diffusion power in animal fur and muscle simulation</b> ⭐ 85</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13857) • [📄 arXiv](https://arxiv.org/abs/2605.13857) • [📥 PDF](https://arxiv.org/pdf/2605.13857)

**💻 Code:** [⭐ Code](https://github.com/Orange-3DV-Team/MoZoo) • [⭐ Code](https://github.com/huggingface)

> MoZoo is a pioneering generative dynamics solver that revolutionizes cinematic animal production by directly synthesizing high-fidelity fur and muscle simulations from coarse meshes, effectively bypassing the labor-intensive refinement stages of t...

</details>

<details>
<summary><b>22. PhoneWorld: Scaling Phone-Use Agent Environments</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pengyuan Lyu, Junyi Li, Xin Lai, Yuxuan Liu, Zhengyang Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29486) • [📄 arXiv](https://arxiv.org/abs/2605.29486) • [📥 PDF](https://arxiv.org/pdf/2605.29486)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> check out our new paper for phone-use/mobile agents gym!

</details>

<details>
<summary><b>23. Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Naomi Saphra, Laura Ruis, Rachit Bansal, Daniel Wurgaft, Jing Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29548) • [📄 arXiv](https://arxiv.org/abs/2605.29548) • [📥 PDF](https://arxiv.org/pdf/2605.29548)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper provides a data-centric account of why larger models learn tasks smaller models fail, attributing this to reduced gradient interference and more efficient resource allocation for rare tasks.

</details>

<details>
<summary><b>24. WorldMemArena: Evaluating Multimodal Agent Memory Through Action-World Interaction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lin Long, Yepeng Liu, Sophia Xiao Pu, Yuzhe Yang, Chengzhi Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29341) • [📄 arXiv](https://arxiv.org/abs/2605.29341) • [📥 PDF](https://arxiv.org/pdf/2605.29341)

**💻 Code:** [⭐ Code](https://github.com/UCSB-AI/WorldMemArena) • [⭐ Code](https://github.com/huggingface)

> WorldMemArena is a new benchmark evaluating the multimodal memory of long-horizon agents using a four-stage Action-World Interaction Loop and multi-session tasks for detailed performance diagnostics.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 24 |
| 📅 Today | [`2026-05-29.json`](data/daily/2026-05-29.json) | 24 |
| 📆 This Week | [`2026-W21.json`](data/weekly/2026-W21.json) | 91 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 664 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-29 | 24 | [View JSON](data/daily/2026-05-29.json) |
| 📄 2026-05-28 | 19 | [View JSON](data/daily/2026-05-28.json) |
| 📄 2026-05-27 | 17 | [View JSON](data/daily/2026-05-27.json) |
| 📄 2026-05-26 | 15 | [View JSON](data/daily/2026-05-26.json) |
| 📄 2026-05-25 | 16 | [View JSON](data/daily/2026-05-25.json) |
| 📄 2026-05-24 | 49 | [View JSON](data/daily/2026-05-24.json) |
| 📄 2026-05-23 | 49 | [View JSON](data/daily/2026-05-23.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W21 | 91 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 664 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |
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
