<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-23-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6116+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">23</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">173</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">735</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6116+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 30, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25518) • [📄 arXiv](https://arxiv.org/abs/2608.25518) • [📥 PDF](https://arxiv.org/pdf/2608.25518)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Thanks for all contributors.

</details>

<details>
<summary><b>2. PAWBench: How Far Are We from Probabilistically Aligned World Modeling?</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Gabriel Jorge Menezes, Le Zhuo, Yuandong Pu, Avr, sayakpaul

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27345) • [📄 arXiv](https://arxiv.org/abs/2608.27345) • [📥 PDF](https://arxiv.org/pdf/2608.27345)

**💻 Code:** [⭐ Code](https://github.com/Andrew0613/PAWBench) • [⭐ Code](https://github.com/huggingface)

> We investigate the probabilistic failures in video generation models and present a reliable benchmark to measure it.

</details>

<details>
<summary><b>3. UrbanGround: From Local Perception to Spatial Agency in a Real-Scale City</b> ⭐ 73</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27456) • [📄 arXiv](https://arxiv.org/abs/2608.27456) • [📥 PDF](https://arxiv.org/pdf/2608.27456)

**💻 Code:** [⭐ Code](https://github.com/UrbanGround/UrbanGround) • [⭐ Code](https://github.com/huggingface)

> 🏙️ UrbanGround turns a real-scale city into an interactive sandbox for multimodal agents. What can you do with UrbanGround? Explore a georegistered 3D replica of Hong Kong from a first-person view Connect MLLM agents for closed-loop perception and...

</details>

<details>
<summary><b>4. TTPO: Test-Time Policy Optimization</b> ⭐ 24</summary>

<br/>

**👥 Authors:** Ying Liu, Shangke Lv, Jianze Wang, LZXzju, Aoshining

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27448) • [📄 arXiv](https://arxiv.org/abs/2608.27448) • [📥 PDF](https://arxiv.org/pdf/2608.27448)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/TTPO) • [⭐ Code](https://github.com/huggingface)

> We introduce TTPO, a label-free test-time training framework that remains robust even when majority-vote pseudo-labels are incorrect. TTPO uses an asymmetric objective: it distills agreeing rollouts while penalizing confident errors in disagreeing...

</details>

<details>
<summary><b>5. Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Siyu Zou, Wanggui He, Yunze Tong, Mushui Liu, Shiyi Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26872) • [📄 arXiv](https://arxiv.org/abs/2608.26872) • [📥 PDF](https://arxiv.org/pdf/2608.26872)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API DreOPD: Degraded-Reference Extrapolative On-Policy Distillation for Flow-ma...

</details>

<details>
<summary><b>6. What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuzhou Wu, Zishan Xu, Xingshan Zeng, vivienlau, clearwind0817

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27260) • [📄 arXiv](https://arxiv.org/abs/2608.27260) • [📥 PDF](https://arxiv.org/pdf/2608.27260)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> What makes generated data actually useful for training LLM agents? We organize agentic data generation around a common $(E,q,\tau,v)$ formulation and propose the ACE lens—Accuracy, Complexity, and divErsity—to connect generation, verification, dif...

</details>

<details>
<summary><b>7. Training Agents to Evolve with Their Harness: TaoLive Digital Avatar Agent Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15763) • [📄 arXiv](https://arxiv.org/abs/2608.15763) • [📥 PDF](https://arxiv.org/pdf/2608.15763)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> ✨ Highlights An end-to-end production Harness Agent system — A comprehensive technical report of a Harness Agent deployed in Taobao Live, covering modular Harness design, agent evaluation, training-free Harness Evolution, model training, offline e...

</details>

<details>
<summary><b>8. GameWAM: A World Action Model for Video Games</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26200) • [📄 arXiv](https://arxiv.org/abs/2608.26200) • [📥 PDF](https://arxiv.org/pdf/2608.26200)

**💻 Code:** [⭐ Code](https://github.com/yunncheng/GameWAM) • [⭐ Code](https://github.com/huggingface)

> Highlights A World Action Model for games: to our knowledge, GameWAM is the first WAM for native closed-loop gameplay and GUI control, jointly generating future visual observations and executable keyboard–mouse trajectories with parallel Video and...

</details>

<details>
<summary><b>9. PILOT in the Loop: Live Self-Improvement for Long-Horizon Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26530) • [📄 arXiv](https://arxiv.org/abs/2608.26530) • [📥 PDF](https://arxiv.org/pdf/2608.26530)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We propose PILOT, a supervisor–worker harness that makes agent self-improvement live rather than post-hoc. Two coupled mechanisms: a separate supervisor redirects or aborts the active worker mid-run (live steering), while runtime-discovered proced...

</details>

<details>
<summary><b>10. Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization</b> ⭐ 158</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26103) • [📄 arXiv](https://arxiv.org/abs/2608.26103) • [📥 PDF](https://arxiv.org/pdf/2608.26103)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/robbyant-research/Zero-WAM)

> Webiste: https://robbyant-research.github.io/Zero-WAM Code: https://github.com/robbyant-research/Zero-WAM Paper: https://arxiv.org/abs/2608.26103

</details>

<details>
<summary><b>11. Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xialiang Tong, Jiaqing Li, Yue Xie, Yunpeng Ba, zz1358m

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27351) • [📄 arXiv](https://arxiv.org/abs/2608.27351) • [📥 PDF](https://arxiv.org/pdf/2608.27351)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Evolution Strategies (ES) have recently emerged as a memory-efficient post-training paradigm for LLM reasoning. However, the optimization behavior of ES remains understudied, making it hard to define its advantage scope compared to mainstream post...

</details>

<details>
<summary><b>12. WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27454) • [📄 arXiv](https://arxiv.org/abs/2608.27454) • [📥 PDF](https://arxiv.org/pdf/2608.27454)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VCE-Skill: Enhancing Skill Self-Evolution with Version-Change Experience (2...

</details>

<details>
<summary><b>13. Procedura: Agentic 3D Modeling with Procedural Control</b> ⭐ 96</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26238) • [📄 arXiv](https://arxiv.org/abs/2608.26238) • [📥 PDF](https://arxiv.org/pdf/2608.26238)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SpatiaOS/Procedura)

> Procedura turns a text prompt into an editable parametric 3D program, with optional per-part materials and articulation.

</details>

<details>
<summary><b>14. CritICL: Inference-Time Weak-to-Strong Generalization from Small Language Model Failure Modes</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruichen Li, Lang Wei, Zhengyi Hu, Yinghui He, Yufan Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27455) • [📄 arXiv](https://arxiv.org/abs/2608.27455) • [📥 PDF](https://arxiv.org/pdf/2608.27455)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Recent advances in inference-time scaling have significantly improved the reasoning performance of large language models (LLMs). However, these methods typically rely on repeated generation or external verification. To address this limitation, we ...

</details>

<details>
<summary><b>15. Magpie: Real-Time World Renderer for Interactive Games</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tengjiao Sun, Huanjie Zhu, Xiaohong Zhang, Xinyu Wang, Xiaoyu Zhan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27168) • [📄 arXiv](https://arxiv.org/abs/2608.27168) • [📥 PDF](https://arxiv.org/pdf/2608.27168)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API AlayaWorld: Long-Horizon and Playable Video World Generation (2026) From Pi...

</details>

<details>
<summary><b>16. Luce: Relightable Gaussians for 3D Asset Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.23943) • [📄 arXiv](https://arxiv.org/abs/2608.23943) • [📥 PDF](https://arxiv.org/pdf/2608.23943)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> High-fidelity image-to-3D generation requires a 3D representation that captures both geometry and appearance. To support relighting and integration into standard rendering pipelines, the representation should include physically based rendering (PB...

</details>

<details>
<summary><b>17. CaRGo-T: Causal Reasoning Graph-of-Thought improves Multimodal Humor Comprehension</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Manav Nitin Kapadnis, Rounak Saha, Aman Bansal, Rahul Seetharaman, Abhilash Nandy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.23172) • [📄 arXiv](https://arxiv.org/abs/2608.23172) • [📥 PDF](https://arxiv.org/pdf/2608.23172)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/abhi1nandy2/CaRGo-T)

> https://arxiv.org/pdf/2608.23172 Large-scale vision-language models (VLMs) have demonstrated remarkable versatility across a wide range of multimodal tasks. However, understanding humor remains challenging because humorous content often depends on...

</details>

<details>
<summary><b>18. CaSKG: Counterfactual-Causal Skill Graphs for Scalable Agent Skill Retrieval</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25500) • [📄 arXiv](https://arxiv.org/abs/2608.25500) • [📥 PDF](https://arxiv.org/pdf/2608.25500)

**💻 Code:** [⭐ Code](https://github.com/ZhiyuanLi218/Caskg) • [⭐ Code](https://github.com/huggingface)

> Reusable skill libraries allow large language model (LLM) agents to reuse procedural knowledge across tasks, but they also turn memory access into a challenging retrieval problem. Full-library prompting preserves coverage at high context cost, vec...

</details>

<details>
<summary><b>19. TacForcing: Streaming Action Generation with Execution-Time Tactile Feedback</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yiyang Chen, Yuzheng Zhang, Boyuan Zhao, Jianbo Zhou, cwx-umich

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25798) • [📄 arXiv](https://arxiv.org/abs/2608.25798) • [📥 PDF](https://arxiv.org/pdf/2608.25798)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🚀 TacForcing project page and real-world demos are now available! TacForcing enables VLA policies to incorporate execution-time tactile feedback without a separate high-frequency reactive controller. It progressively generates and executes action ...

</details>

<details>
<summary><b>20. Thinking on Shots: Consistent Multi-Shot Video Editing with Agentic Reasoning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26809) • [📄 arXiv](https://arxiv.org/abs/2608.26809) • [📥 PDF](https://arxiv.org/pdf/2608.26809)

**💻 Code:** [⭐ Code](https://github.com/Wucy0519/MMLVE) • [⭐ Code](https://github.com/huggingface)

> Project Page: https://wucy0519.github.io/MMLVE/ Source Codes: https://github.com/Wucy0519/MMLVE Benchmark: https://huggingface.co/datasets/wcy1234567/MMLVE-Bench

</details>

<details>
<summary><b>21. EditaLive! Unified Character Video Editing for Live Streaming</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27123) • [📄 arXiv](https://arxiv.org/abs/2608.27123) • [📥 PDF](https://arxiv.org/pdf/2608.27123)

**💻 Code:** [⭐ Code](https://github.com/GVCLab/EditaLive) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. Aphanta: Diagnosing Task-Aligned Image-Edited Intermediates for Multimodal Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26993) • [📄 arXiv](https://arxiv.org/abs/2608.26993) • [📥 PDF](https://arxiv.org/pdf/2608.26993)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Explicit visual intermediates can help multimodal large language models (MLLMs) externalize spatial evidence and updated visual states, but their utility depends on whether an image editor can faithfully realize the required transformation. We int...

</details>

<details>
<summary><b>23. What Does an Evaluation License? A Commit-Bound Census of Claim-Relative Inference in Inspect Evals</b> ⭐ 0</summary>

<br/>

**👥 Authors:** qxxxxxxxxxxx

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19269) • [📄 arXiv](https://arxiv.org/abs/2608.19269) • [📥 PDF](https://arxiv.org/pdf/2608.19269)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> What does a benchmark result actually let us conclude? In a commit-bound census of 124 Inspect Evals units, 110 historical claims stop at explicit evidence or semantic gates. Among the executable cases, exact values, winners, complete rankings, an...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 23 |
| 📅 Today | [`2026-08-30.json`](data/daily/2026-08-30.json) | 23 |
| 📆 This Week | [`2026-W34.json`](data/weekly/2026-W34.json) | 173 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 735 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-30 | 23 | [View JSON](data/daily/2026-08-30.json) |
| 📄 2026-08-29 | 23 | [View JSON](data/daily/2026-08-29.json) |
| 📄 2026-08-28 | 18 | [View JSON](data/daily/2026-08-28.json) |
| 📄 2026-08-27 | 26 | [View JSON](data/daily/2026-08-27.json) |
| 📄 2026-08-26 | 35 | [View JSON](data/daily/2026-08-26.json) |
| 📄 2026-08-25 | 22 | [View JSON](data/daily/2026-08-25.json) |
| 📄 2026-08-24 | 26 | [View JSON](data/daily/2026-08-24.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W34 | 173 | [View JSON](data/weekly/2026-W34.json) |
| 📅 2026-W33 | 213 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 735 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
