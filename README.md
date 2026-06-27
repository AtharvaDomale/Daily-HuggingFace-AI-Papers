<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-25-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4960+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">25</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">83</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">557</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4960+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 27, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. DanceOPD: On-Policy Generative Field Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27377) • [📄 arXiv](https://arxiv.org/abs/2606.27377) • [📥 PDF](https://arxiv.org/pdf/2606.27377)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> DanceOPD uses hard routing to select one frozen capability field, queries it on a low-noise on-policy student state, and matches the selected velocity with a local velocity MSE loss.

</details>

<details>
<summary><b>2. ViQ: Text-Aligned Visual Quantized Representations at Any Resolution</b> ⭐ 36</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27313) • [📄 arXiv](https://arxiv.org/abs/2606.27313) • [📥 PDF](https://arxiv.org/pdf/2606.27313)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yuxumin/ViQ)

> GitHub: https://github.com/yuxumin/ViQ HuggingFace: https://huggingface.co/XuminYu/ViQ_weights

</details>

<details>
<summary><b>3. OPID: On-Policy Skill Distillation for Agentic Reinforcement Learning</b> ⭐ 29</summary>

<br/>

**👥 Authors:** Fan Zhang, Yuhao Shen, Zhengxi Lu, Jinyang Wu, Shuo Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26790) • [📄 arXiv](https://arxiv.org/abs/2606.26790) • [📥 PDF](https://arxiv.org/pdf/2606.26790)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jinyangwu/OPID)

> No abstract available.

</details>

<details>
<summary><b>4. Qwen-Image-Agent: Bridging the Context Gap in Real-World Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26907) • [📄 arXiv](https://arxiv.org/abs/2606.26907) • [📥 PDF](https://arxiv.org/pdf/2606.26907)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Qwen3.7-Max/Plus is already live as a closed API — any plans for open-weight releases of the 3.7 family? (like 3.6-35B-A3B / 3.6-27B alongside 3.6-Max) Would love to run it locally via llama.cpp / GGUF.

</details>

<details>
<summary><b>5. The Verification Horizon: No Silver Bullet for Coding Agent Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26300) • [📄 arXiv](https://arxiv.org/abs/2606.26300) • [📥 PDF](https://arxiv.org/pdf/2606.26300)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A classical intuition holds that verifying a solution is easier than producing one. For today's coding agents, this intuition is being inverted: as foundation models develop stronger reasoning capabilities and engineering harnesses grow more sophi...

</details>

<details>
<summary><b>6. JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting</b> ⭐ 35</summary>

<br/>

**👥 Authors:** Yujie Zhao, Haoran Yuan, Yulun Wu, Zhaoxiang Feng, Lanxiang Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.18394) • [📄 arXiv](https://arxiv.org/abs/2606.18394) • [📥 PDF](https://arxiv.org/pdf/2606.18394)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hao-ai-lab/JetSpec)

> Speculative decoding (SD) accelerates autoregressive Large Language Models (LLMs) by drafting multiple tokens and verifying them in parallel, but it faces a scaling limitation: increasing the draft budget improves speed only when acceptance remain...

</details>

<details>
<summary><b>7. GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Tingyu Song, Jinbiao Wei, Yilun Zhao, Siyue Zhang, rebeccazzzz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24551) • [📄 arXiv](https://arxiv.org/abs/2606.24551) • [📥 PDF](https://arxiv.org/pdf/2606.24551)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/rebeccaz4/gui-vs-cli)

> Computer-use agents can execute software tasks through either graphical interfaces or programmatic command interfaces, but existing evaluations confound interaction modality with differences in tasks, initial states, verifiers, and permitted actio...

</details>

<details>
<summary><b>8. Fast LeWorldModel</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Xiangyu Xu, Yuntian Gao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26217) • [📄 arXiv](https://arxiv.org/abs/2606.26217) • [📥 PDF](https://arxiv.org/pdf/2606.26217)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Yuntian-Gao/Fast-LeWorldModel)

> No abstract available.

</details>

<details>
<summary><b>9. Why Multi-Step Tool-Use Reinforcement Learning Collapses and How Supervisory Signals Fix It</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26027) • [📄 arXiv](https://arxiv.org/abs/2606.26027) • [📥 PDF](https://arxiv.org/pdf/2606.26027)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hypasd-art/Tool-RL-Box)

> Our Code is available at https://github.com/hypasd-art/Tool-RL-Box .

</details>

<details>
<summary><b>10. Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.14397) • [📄 arXiv](https://arxiv.org/abs/2606.14397) • [📥 PDF](https://arxiv.org/pdf/2606.14397)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/gauntlet-benchmark/evaluation-harness)

> https://gauntlet-landing-page.vercel.app/

</details>

<details>
<summary><b>11. LISA: Likelihood Score Alignment for Visual-condition Controllable Generation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27192) • [📄 arXiv](https://arxiv.org/abs/2606.27192) • [📥 PDF](https://arxiv.org/pdf/2606.27192)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HKUST-LongGroup/LISA)

> LISA can accelerate the training for controllable visual generation and bootstrap better synthetic results.

</details>

<details>
<summary><b>12. In-Context World Modeling for Robotic Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26025) • [📄 arXiv](https://arxiv.org/abs/2606.26025) • [📥 PDF](https://arxiv.org/pdf/2606.26025)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> In-Context World Modeling for Robotic Control

</details>

<details>
<summary><b>13. Confidence-Aware Tool Orchestration for Robust Video Understanding</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26904) • [📄 arXiv](https://arxiv.org/abs/2606.26904) • [📥 PDF](https://arxiv.org/pdf/2606.26904)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ROVA-V2/Robust-TO)

> Project Page: https://rova-v2.github.io/

</details>

<details>
<summary><b>14. PhysiFormer: Learning to Simulate Mechanics in World Space</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Andrea Vedaldi, Yushi Lan, Yiming Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27364) • [📄 arXiv](https://arxiv.org/abs/2606.27364) • [📥 PDF](https://arxiv.org/pdf/2606.27364)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. Hallucination in World Models is Predictable and Preventable</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27326) • [📄 arXiv](https://arxiv.org/abs/2606.27326) • [📥 PDF](https://arxiv.org/pdf/2606.27326)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/nicklashansen/mmbench2)

> 🌐 Interactive paper: https://www.nicklashansen.com/mmbench2 🕹️ Live demo: https://www.nicklashansen.com/mmbench2/#live-demo 📄 Paper: https://arxiv.org/abs/2606.27326 💻 Code: https://github.com/nicklashansen/mmbench2 📦 Dataset: https://huggingface....

</details>

<details>
<summary><b>16. CoffeeBench: Benchmarking Long-Horizon LLM Agents in Heterogeneous Multi-Agent Economies</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Shota Onose, Keita Ogawa, Kazuo Araragi, Daichi Hattori, speed

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16613) • [📄 arXiv](https://arxiv.org/abs/2606.16613) • [📥 PDF](https://arxiv.org/pdf/2606.16613)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SakanaAI/CoffeeBench)

> We introduce CoffeeBench, a benchmark for evaluating LLM agents in a long-horizon multi-agent economy composed of heterogeneous firms. In CoffeeBench, two farmers, two roasters, and two retailers autonomously operate their businesses over a 90-day...

</details>

<details>
<summary><b>17. Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26080) • [📄 arXiv](https://arxiv.org/abs/2606.26080) • [📥 PDF](https://arxiv.org/pdf/2606.26080)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/deeplearning-wisc/progress-advantage)

> We introduce Progress Advantage, an implicit process reward signal derived as a byproduct of post-training, enabling step-level guidance and monitoring for LLM agents in stochastic environments.

</details>

<details>
<summary><b>18. Discretizing Reward Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.21795) • [📄 arXiv](https://arxiv.org/abs/2606.21795) • [📥 PDF](https://arxiv.org/pdf/2606.21795)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reward models are an essential tool in language model alignment and post-training. But we don't really understand what makes a given reward model  suitable as a teacher for reinforcement learning. Some reward models that are near-perfect at imitat...

</details>

<details>
<summary><b>19. Information-Aware KV Cache Compression for Long Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhouhan Lin, Alexandra Birch, Zhuiri Xiao, Jushi Kai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26875) • [📄 arXiv](https://arxiv.org/abs/2606.26875) • [📥 PDF](https://arxiv.org/pdf/2606.26875)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>20. When Does Combining Language Models Help? A Co-Failure Ceiling on Routing, Voting, and Mixture-of-Agents Across 67 Frontier Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27288) • [📄 arXiv](https://arxiv.org/abs/2606.27288) • [📥 PDF](https://arxiv.org/pdf/2606.27288)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Combining LLMs Rarely Beats the Single Best Model: A Provable Co-Failure Ceiling Across 67 Frontier Models

</details>

<details>
<summary><b>21. COrigami: An AI Pipeline for Co-Designing Flat-Foldable Visually Recognisable Origami</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26299) • [📄 arXiv](https://arxiv.org/abs/2606.26299) • [📥 PDF](https://arxiv.org/pdf/2606.26299)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. OpenBioRQ: Unsolved Biomedical Research Questions for Agents</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Minbyul Jeong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.21959) • [📄 arXiv](https://arxiv.org/abs/2606.21959) • [📥 PDF](https://arxiv.org/pdf/2606.21959)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/minstar/healthcare-research)

> project page: https://minstar.github.io/OpenBioRQ/ dataset: https://huggingface.co/datasets/Minbyul/OpenBioRQ

</details>

<details>
<summary><b>23. How Post-Training Shapes Biological Reasoning Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bryan Perozzi, Eric Wang, Michelle M. Li, Hanlin Zhang, Lukas Fesser

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.16517) • [📄 arXiv](https://arxiv.org/abs/2606.16517) • [📥 PDF](https://arxiv.org/pdf/2606.16517)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This study of 100+ models reveals that biological reasoning follows non-monotonic dynamics, where more post-training compute does not always yield better generalization. We identify a "generalization collapse" where SFT drives in-domain accuracy b...

</details>

<details>
<summary><b>24. EO-WM: A Physically Informed World Model for Probabilistic Earth Observation Forecasting</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Zhe Liu, Yansheng Li, Zhenya Yang, Shuai Yuan, Junwei Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27277) • [📄 arXiv](https://arxiv.org/abs/2606.27277) • [📥 PDF](https://arxiv.org/pdf/2606.27277)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Luo-Z13/EO-WM)

> [EO-WM](EO-WM: A Physically Informed World Model for Probabilistic Earth Observation Forecasting)

</details>

<details>
<summary><b>25. ABACUS: Adapting Unified Foundation Model for Bridging Image Count Understanding and Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Anjan Dutta, Sauradip Nag, Anindya Mondal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.23835) • [📄 arXiv](https://arxiv.org/abs/2606.23835) • [📥 PDF](https://arxiv.org/pdf/2606.23835)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> ABACUS is a unified vision-language model that handles object counting, crowd counting, referring-expression counting, and count-faithful image generation without any benchmark-specific training required. Our model is built on existing 3B-paramete...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 25 |
| 📅 Today | [`2026-06-27.json`](data/daily/2026-06-27.json) | 25 |
| 📆 This Week | [`2026-W25.json`](data/weekly/2026-W25.json) | 83 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 557 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-27 | 25 | [View JSON](data/daily/2026-06-27.json) |
| 📄 2026-06-26 | 12 | [View JSON](data/daily/2026-06-26.json) |
| 📄 2026-06-25 | 13 | [View JSON](data/daily/2026-06-25.json) |
| 📄 2026-06-24 | 15 | [View JSON](data/daily/2026-06-24.json) |
| 📄 2026-06-23 | 14 | [View JSON](data/daily/2026-06-23.json) |
| 📄 2026-06-22 | 4 | [View JSON](data/daily/2026-06-22.json) |
| 📄 2026-06-21 | 34 | [View JSON](data/daily/2026-06-21.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W25 | 83 | [View JSON](data/weekly/2026-W25.json) |
| 📅 2026-W24 | 130 | [View JSON](data/weekly/2026-W24.json) |
| 📅 2026-W23 | 166 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 557 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |

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
