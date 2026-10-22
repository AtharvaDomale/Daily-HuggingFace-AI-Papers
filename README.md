<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-28-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7505+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">28</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">91</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">897</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7505+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 22, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Agent World Model in the Wild: Computer-Use Agent Environment Synthesis</b> ⭐ 50</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.26828) • [📄 arXiv](https://arxiv.org/abs/2610.26828) • [📥 PDF](https://arxiv.org/pdf/2610.26828)

**💻 Code:** [⭐ Code](https://github.com/MSR-Orchard/WildAWM) • [⭐ Code](https://github.com/huggingface)

> We introduce WildAWM (Agent World Model in the Wild), a scalable pipeline that synthesizes executable computer-use agent (CUA) environments and tasks for training general CUA, powered entirely by open-source models. Core contributions: Worlds: per...

</details>

<details>
<summary><b>2. Cache Engineering for Agents</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28577) • [📄 arXiv](https://arxiv.org/abs/2610.28577) • [📥 PDF](https://arxiv.org/pdf/2610.28577)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/thu-nics/AKV)

> We let agents manage their KV cache directly, proactively dropping what they no longer need without rewriting their conversation history or recomputing the remaining cache. We built AKV serving system to support this efficiently, reducing cache fo...

</details>

<details>
<summary><b>3. Socratic Skill Learning: Interleaving Teaching and Practice for Transferable Agent Skills</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17860) • [📄 arXiv](https://arxiv.org/abs/2610.17860) • [📥 PDF](https://arxiv.org/pdf/2610.17860)

**💻 Code:** [⭐ Code](https://github.com/NJU-LINK/Socratic-Skill-Learning) • [⭐ Code](https://github.com/huggingface)

> Agent Skills improve LLM agents by making task knowledge reusable across sessions. Learning transferable Skills requires evidence about a procedure and its conditions of use, which a successful task attempt can leave untested. We introduce Socrati...

</details>

<details>
<summary><b>4. Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29989) • [📄 arXiv](https://arxiv.org/abs/2610.29989) • [📥 PDF](https://arxiv.org/pdf/2610.29989)

**💻 Code:** [⭐ Code](https://github.com/TencentARC/Looking-Ahead) • [⭐ Code](https://github.com/huggingface)

> 🔭 Introducing Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation — teaching autoregressive video models to learn how the world evolves and better anticipate future dynamics through generative future prediction. Our method ...

</details>

<details>
<summary><b>5. AgenticRoboRL: Bringing Coding-Agent Post-Training to Agentic Robotics</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28589) • [📄 arXiv](https://arxiv.org/abs/2610.28589) • [📥 PDF](https://arxiv.org/pdf/2610.28589)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/xinxinliu-ai/AgenticRoboRL)

> AgenticRoboRL trains the planner model of a robot harness inside the harness itself. A harness pairs a language model planner with robot tools: the planner may call a VLA policy as a skill, or write Python against the harness's robot APIs. The har...

</details>

<details>
<summary><b>6. Memory-Centric Continual Learning for AI Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29357) • [📄 arXiv](https://arxiv.org/abs/2610.29357) • [📥 PDF](https://arxiv.org/pdf/2610.29357)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Memory-Centric Continual Learning for AI Agents

</details>

<details>
<summary><b>7. Building AI Research Teams for Autonomous Discovery of Linear Attention Architectures</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28887) • [📄 arXiv](https://arxiv.org/abs/2610.28887) • [📥 PDF](https://arxiv.org/pdf/2610.28887)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We build a team of agents with diverse scientific backgrounds that collaborate to discover better linear attention architectures.

</details>

<details>
<summary><b>8. TraceForge: Distilling with Your Own Thinking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29680) • [📄 arXiv](https://arxiv.org/abs/2610.29680) • [📥 PDF](https://arxiv.org/pdf/2610.29680)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🔧 TraceForge: feed it a context and an answer from Claude or GPT, and it writes the missing reasoning trace in the Qwen family's own voice. 📈 It beats the real thing. With questions and teacher answers held fixed, students trained on TraceForge Co...

</details>

<details>
<summary><b>9. Real In, Real Out: What If We Only Use Real Data for Scene Text Editing?</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27592) • [📄 arXiv](https://arxiv.org/abs/2610.27592) • [📥 PDF](https://arxiv.org/pdf/2610.27592)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/YesianRohn/TextRIRO)

> TextRIRO lifts STE to realistic, reliable, and widely applicable new levels.

</details>

<details>
<summary><b>10. Harness Scaling for Generalizable Agentic Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29458) • [📄 arXiv](https://arxiv.org/abs/2610.29458) • [📥 PDF](https://arxiv.org/pdf/2610.29458)

**💻 Code:** [⭐ Code](https://github.com/RUCAIBox/Harness-Scaling) • [⭐ Code](https://github.com/huggingface)

> Excited to share our work Harness Scaling for Generalizable Agentic Reinforcement Learning ! Most existing agentic RL methods train through a single fixed harness, potentially limiting generalization across different execution configurations. We i...

</details>

<details>
<summary><b>11. Spectral-Spatial Interpretation</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Philip Townsend, Ruqi Yang, Hunter26666666

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21114) • [📄 arXiv](https://arxiv.org/abs/2610.21114) • [📥 PDF](https://arxiv.org/pdf/2610.21114)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AtheavenMa/Spectral-Spatial-Interpretation)

> 🔥🔥🔥Which spectral channels matter, and where? SSI explains a model score with signed attributions for each spatial cell and channel. It keeps the surrounding image fixed while allocating a cell's contribution among its channels using Shapley value...

</details>

<details>
<summary><b>12. StreamEval: Evaluating Interactive Streaming Video Generation Models</b> ⭐ 129</summary>

<br/>

**👥 Authors:** StreamEval Team

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28402) • [📄 arXiv](https://arxiv.org/abs/2610.28402) • [📥 PDF](https://arxiv.org/pdf/2610.28402)

**💻 Code:** [⭐ Code](https://github.com/StreamEval/StreamEval) • [⭐ Code](https://github.com/huggingface)

> TL;DR: We introduce StreamEval, the first benchmark for interactive streaming video generation . It evaluates whether models can handle incrementally arriving multimodal requests while executing specified state changes and preserving content throu...

</details>

<details>
<summary><b>13. Which Agent's Skill Should Change? Group-Relative Credit Assignment for Evolving Multi-Agent Teams</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27227) • [📄 arXiv](https://arxiv.org/abs/2610.27227) • [📥 PDF](https://arxiv.org/pdf/2610.27227)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yananma1029/GRSE-Skills)

> 🚀 We present GRSE (Group-Relative Skill Evolution), a framework that tackles a fundamental question in multi-agent learning: When a team succeeds or fails, which agent's skill should actually change? Instead of rewarding all skills involved in suc...

</details>

<details>
<summary><b>14. SpatialBrain: Spatial Intelligence Growing from Experience</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29317) • [📄 arXiv](https://arxiv.org/abs/2610.29317) • [📥 PDF](https://arxiv.org/pdf/2610.29317)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Spatial reasoning requires more than access to geometric tools: a reasoner must decide when direct observation is sufficient and when to reconstruct, measure, or change reference frames. We present SpatialBrain, a framework that retains experience...

</details>

<details>
<summary><b>15. ACE-Brain-0.6: Scaling Spatial Grounding for Physical Foundation Models</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27040) • [📄 arXiv](https://arxiv.org/abs/2610.27040) • [📥 PDF](https://arxiv.org/pdf/2610.27040)

**💻 Code:** [⭐ Code](https://github.com/DAXIAORobotics/ACE-Brain-0.6) • [⭐ Code](https://github.com/huggingface)

> General-purpose multimodal models, including GPT-6 Astra and Claude Opus 5.5, demonstrate broad capabilities in embodied manipulation. Our analysis indicates that this generality is accompanied by strong spatial perception, as Astra can determine ...

</details>

<details>
<summary><b>16. Sparsity Should Be ALLOTted by Output: Training-Free Allocation of Block-Sparse Attention for Video Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29441) • [📄 arXiv](https://arxiv.org/abs/2610.29441) • [📥 PDF](https://arxiv.org/pdf/2610.29441)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce ALLOT, a training-free sparsity allocation method for block-sparse attention in video Diffusion Transformers. Our approach estimates each layer–step cell’s final-video sensitivity and jointly allocates densities under a measured-laten...

</details>

<details>
<summary><b>17. ALPS: Visual State Encoders for World Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.30120) • [📄 arXiv](https://arxiv.org/abs/2610.30120) • [📥 PDF](https://arxiv.org/pdf/2610.30120)

**💻 Code:** [⭐ Code](https://github.com/UCSC-VLAA/ALPS) • [⭐ Code](https://github.com/huggingface)

> World models often predict and plan over the tokens of a frozen visual encoder, so the encoder determines what they can see. We introduce ALPS (A Language-aligned, Pixel-renderable State encoder), a ViT-B/16 pretrained from scratch using only stil...

</details>

<details>
<summary><b>18. Answering in the User's Variety: Calibrated Accommodation Pairs for Post-Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jingshu Peng, Junyu Luo, Jian Yang, Heming Li, Zhizhuo Kou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28474) • [📄 arXiv](https://arxiv.org/abs/2610.28474) • [📥 PDF](https://arxiv.org/pdf/2610.28474)

**💻 Code:** [⭐ Code](https://github.com/kouzhizhuo/cap-accommodation) • [⭐ Code](https://github.com/huggingface)

> People adapt how they talk to the person in front of them. However, chat models tend to respond in Standard English regardless of the dialect used by the user. A uniform answer can hide whether training treats these users alike, and output audits ...

</details>

<details>
<summary><b>19. Scaling in Games: Continued Pre-Training For Embodied Agents In Diverse Virtual Worlds</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28698) • [📄 arXiv](https://arxiv.org/abs/2610.28698) • [📥 PDF](https://arxiv.org/pdf/2610.28698)

**💻 Code:** [⭐ Code](https://github.com/THUSI-Lab/GameScaling) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>20. DeLoop: Repeated Actions as Self-Supervision for Long-Horizon Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19376) • [📄 arXiv](https://arxiv.org/abs/2610.19376) • [📥 PDF](https://arxiv.org/pdf/2610.19376)

**💻 Code:** [⭐ Code](https://github.com/cskyl/deloop) • [⭐ Code](https://github.com/huggingface)

> TL DR: LLM agents often re-issue a tool call they already made and get the same observation back. We call this a loop . DeLoop turns each loop into a free training pair from the agent's own rollout: the repeated call is the rejected action, and th...

</details>

<details>
<summary><b>21. Qwen-Audio-3.1-ASR Technical Report</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chong Deng, Zhigao Chen, Peixin Chen, Daren Chen, Chuanmeng Bian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28711) • [📄 arXiv](https://arxiv.org/abs/2610.28711) • [📥 PDF](https://arxiv.org/pdf/2610.28711)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. Evolving with Open Worlds: Towards Recursive Self-Improvement for Future Forecasting</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Lingyun Feng, Hongkang Zhao, Yanxi Chen, Yunsen Liang, Zhipeng Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29784) • [📄 arXiv](https://arxiv.org/abs/2610.29784) • [📥 PDF](https://arxiv.org/pdf/2610.29784)

**💻 Code:** [⭐ Code](https://github.com/RUCAIBox/DeepEvolve) • [⭐ Code](https://github.com/huggingface)

> What does it take for AI agents to continuously improve themselves? We argue that the environment is not merely a place for training or evaluation, but a fundamental driver of recursive self-improvement (RSI). We introduce DeepEvolve, a system in ...

</details>

<details>
<summary><b>23. Agentic World Action Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jing Wang, Boning Shao, Botong Zhang, Dantong Niu, Zhuoyang Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.30136) • [📄 arXiv](https://arxiv.org/abs/2610.30136) • [📥 PDF](https://arxiv.org/pdf/2610.30136)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>24. CulturalRiddles: A Multicultural Benchmark of Riddles</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29968) • [📄 arXiv](https://arxiv.org/abs/2610.29968) • [📥 PDF](https://arxiv.org/pdf/2610.29968)

**💻 Code:** [⭐ Code](https://github.com/Cohere-Labs-Community/CulturalRiddles) • [⭐ Code](https://github.com/huggingface)

> A new benchmark testing cultural awareness and cultural reasoning of LLMs with riddles from 62 cultures (45 countries, 51 languages). The paper describes the participatory data creation process, human evaluation, automatic evaluation metric tuning...

</details>

<details>
<summary><b>25. RubatoCodec: Budgeted Adaptive Speech Tokenization with Streaming Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29434) • [📄 arXiv](https://arxiv.org/abs/2610.29434) • [📥 PDF](https://arxiv.org/pdf/2610.29434)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/third-intelligence/RubatoCodec)

> Code: https://github.com/third-intelligence/RubatoCodec Model: https://huggingface.co/third-intelligence/RubatoCodec

</details>

<details>
<summary><b>26. Toward Part-Aware Choral Transcription with singing voice assignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09861) • [📄 arXiv](https://arxiv.org/abs/2610.09861) • [📥 PDF](https://arxiv.org/pdf/2610.09861)

**💻 Code:** [⭐ Code](https://github.com/Multimodal-Music-Research-Lab/Paw_Choral_AMT) • [⭐ Code](https://github.com/huggingface)

> Hi everyone! I’m one of the authors of this paper. We introduce PawCT, an end-to-end framework that transcribes mixed choral audio into separate soprano, alto, tenor, and bass (SATB) note tracks. We also present PagCT, a part-agnostic variant that...

</details>

<details>
<summary><b>27. VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Juan S. Santillana

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.30090) • [📄 arXiv](https://arxiv.org/abs/2610.30090) • [📥 PDF](https://arxiv.org/pdf/2610.30090)

**💻 Code:** [⭐ Code](https://github.com/vectrayx/vx-one) • [⭐ Code](https://github.com/huggingface)

> Perimeter triage faces a latency/semantics trade-off: signature engines are fast and format-safe but blind to unseen attacks, while generative models are semantic but slow and produce unparseable output. In late 2026 a class of commercial System O...

</details>

<details>
<summary><b>28. Entertain, then Cancel: How Thinking Makes Reasoning Models Read Some and Or Literally</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jingshu Peng, Junyu Luo, Pengfei Hao, Jian Yang, Zhizhuo Kou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21204) • [📄 arXiv](https://arxiv.org/abs/2610.21204) • [📥 PDF](https://arxiv.org/pdf/2610.21204)

**💻 Code:** [⭐ Code](https://github.com/kouzhizhuo/entertain-then-cancel) • [⭐ Code](https://github.com/huggingface)

> When a speaker says ``some students passed'', listeners infer that not all of them did. Reasoning models now let users decide how long a model thinks before it answers, but we do not know what extra thinking does to such everyday inferences. We st...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 28 |
| 📅 Today | [`2026-10-22.json`](data/daily/2026-10-22.json) | 28 |
| 📆 This Week | [`2026-W42.json`](data/weekly/2026-W42.json) | 91 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 897 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-22 | 28 | [View JSON](data/daily/2026-10-22.json) |
| 📄 2026-10-21 | 21 | [View JSON](data/daily/2026-10-21.json) |
| 📄 2026-10-20 | 21 | [View JSON](data/daily/2026-10-20.json) |
| 📄 2026-10-19 | 21 | [View JSON](data/daily/2026-10-19.json) |
| 📄 2026-10-18 | 84 | [View JSON](data/daily/2026-10-18.json) |
| 📄 2026-10-17 | 84 | [View JSON](data/daily/2026-10-17.json) |
| 📄 2026-10-16 | 34 | [View JSON](data/daily/2026-10-16.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W42 | 91 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 897 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |

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
