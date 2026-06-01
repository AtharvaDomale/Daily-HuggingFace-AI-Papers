<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-25-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4428+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">25</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">25</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4428+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 01, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31584) • [📄 arXiv](https://arxiv.org/abs/2605.31584) • [📥 PDF](https://arxiv.org/pdf/2605.31584)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/THU-KEG/LongTraceRL)

> Paper: https://arxiv.org/abs/2605.31584 Code: https://github.com/THU-KEG/LongTraceRL

</details>

<details>
<summary><b>2. Function2Scene: 3D Indoor Scene Layout from Functional Specifications</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Manolis Savva, Angel X. Chang, Daniel Ritchie, Qimin Chen, Ruiqi Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30819) • [📄 arXiv](https://arxiv.org/abs/2605.30819) • [📥 PDF](https://arxiv.org/pdf/2605.30819)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Most text-driven 3D indoor scene synthesis methods generate rooms from object-centric prompts, asking what furniture should be placed rather than how the space is used. Yet in real interior design, a layout is judged by how well it supports its oc...

</details>

<details>
<summary><b>3. Representation Forcing for Bottleneck-Free Unified Multimodal Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31604) • [📄 arXiv](https://arxiv.org/abs/2605.31604) • [📥 PDF](https://arxiv.org/pdf/2605.31604)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The paper introduces Representation Forcing, a method enabling bottleneck-free unified multimodal models by autoregressively predicting visual representation tokens before pixels, removing reliance on frozen external VAEs for image generation.

</details>

<details>
<summary><b>4. COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31264) • [📄 arXiv](https://arxiv.org/abs/2605.31264) • [📥 PDF](https://arxiv.org/pdf/2605.31264)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/titanwings/colleague-skill)

> The tech report of COLLEAGUE.SKILL is now available!

</details>

<details>
<summary><b>5. Task-Focused Memorization for Multimodal Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31075) • [📄 arXiv](https://arxiv.org/abs/2605.31075) • [📥 PDF](https://arxiv.org/pdf/2605.31075)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://taskmem.github.io/

</details>

<details>
<summary><b>6. SANA-Streaming: Real-time Streaming Video Editing with Hybrid Diffusion Transformer</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30409) • [📄 arXiv](https://arxiv.org/abs/2605.30409) • [📥 PDF](https://arxiv.org/pdf/2605.30409)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Real-time streaming video-to-video editing (V2V) is critical for interactive applications such as live broadcasting and gaming, yet it remains a formidable challenge due to the stringent requirements for temporal consistency and inference throughp...

</details>

<details>
<summary><b>7. SwanVoice: Expressive Long-Form Zero-Shot Speech Synthesis for Both Monologue and Dialogue</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30993) • [📄 arXiv](https://arxiv.org/abs/2605.30993) • [📥 PDF](https://arxiv.org/pdf/2605.30993)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Zero-shot text-to-speech (TTS) has improved substantially for single-speaker synthesis, yet expressive long-form multi-speaker dialogue remains difficult. A common workaround is to synthesize each turn with a monologue TTS model and stitch the out...

</details>

<details>
<summary><b>8. GGT-100K: Generative Ground Truth for Generalizable Real-World Image Restoration</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31039) • [📄 arXiv](https://arxiv.org/abs/2605.31039) • [📥 PDF](https://arxiv.org/pdf/2605.31039)

**💻 Code:** [⭐ Code](https://github.com/PolyU-VCLab/GGT-100K) • [⭐ Code](https://github.com/huggingface)

> Real-world image restoration (IR) is bottlenecked by the scarcity of high-quality paired training data. Synthetic datasets are abundant but often fail to model real-world degradations, while real-world paired datasets are expensive and difficult t...

</details>

<details>
<summary><b>9. Exploring Autonomous Agentic Data Engineering for Model Specialization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30407) • [📄 arXiv](https://arxiv.org/abs/2605.30407) • [📥 PDF](https://arxiv.org/pdf/2605.30407)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/DataAgent) • [⭐ Code](https://github.com/huggingface)

> Large Language Models (LLMs) have demonstrated strong performance on general tasks, while often struggling to adapt to specialized domains without high-quality domain-specific data. Existing LLM-based data curation methods primarily rely on human-...

</details>

<details>
<summary><b>10. dMoE: dLLMs with Learnable Block Experts</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Xinchao Wang, Xinyin Ma, Gongfan Fang, Zigeng Chen, Sicheng Feng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30876) • [📄 arXiv](https://arxiv.org/abs/2605.30876) • [📥 PDF](https://arxiv.org/pdf/2605.30876)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/fscdc/dMoE)

> Welcome discussion!

</details>

<details>
<summary><b>11. Not All Disagreement Is Learnable: Token Teachability in On-Policy Distillation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26844) • [📄 arXiv](https://arxiv.org/abs/2605.26844) • [📥 PDF](https://arxiv.org/pdf/2605.26844)

**💻 Code:** [⭐ Code](https://github.com/wyy-code/TA-OPD) • [⭐ Code](https://github.com/huggingface)

> This work answers the question: "which token-level teacher signals in OPD are actually learnable?" Our fixed-context KL-reduction diagnostic shows that high disagreement token conflates learnable disagreement, where the teacher assigns corrective ...

</details>

<details>
<summary><b>12. Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30940) • [📄 arXiv](https://arxiv.org/abs/2605.30940) • [📥 PDF](https://arxiv.org/pdf/2605.30940)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Real-time and accurate spatial audio generation is pivotal for delivering an immersive experience. However, existing spatial audio synthesis technologies are often encumbered by a tradeoff between generation quality and high inference latency, as ...

</details>

<details>
<summary><b>13. LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30434) • [📄 arXiv](https://arxiv.org/abs/2605.30434) • [📥 PDF](https://arxiv.org/pdf/2605.30434)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> LongDS is the  benchmark to test whether data analysis agents can reliably track evolving analytical states over hundreds of interactions.

</details>

<details>
<summary><b>14. Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.28618) • [📄 arXiv](https://arxiv.org/abs/2605.28618) • [📥 PDF](https://arxiv.org/pdf/2605.28618)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Recent advances in speech generation have enabled high-fidelity synthesis, yet systematic evaluation of models under long-context conditions remains largely underexplored. A comprehensive evaluation benchmark for long-form speech is indispensable ...

</details>

<details>
<summary><b>15. Hide-and-Seek in Trajectories: Discovering Failure Signals for VLA Runtime Monitoring</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30834) • [📄 arXiv](https://arxiv.org/abs/2605.30834) • [📥 PDF](https://arxiv.org/pdf/2605.30834)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hide-and-Seek reformulates VLA failure detection as a coarsely supervised problem, using inter- and intra-trajectory contrastive objectives to induce temporally structured failure signals from trajectory-level labels only.

</details>

<details>
<summary><b>16. VLM3: Vision Language Models Are Native 3D Learners</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30561) • [📄 arXiv](https://arxiv.org/abs/2605.30561) • [📥 PDF](https://arxiv.org/pdf/2605.30561)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/VLM3) • [⭐ Code](https://github.com/huggingface)

> Vision Language Models (VLMs) enable a unified model to solve various vision tasks through prompting. They have shown promising performance in semantic understanding. However, 3D understanding still largely relies on expert vision models with comp...

</details>

<details>
<summary><b>17. Linear Scaling Video VLMs for Long Video Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31598) • [📄 arXiv](https://arxiv.org/abs/2605.31598) • [📥 PDF](https://arxiv.org/pdf/2605.31598)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> StateKV is an inference-time method that enables linear-time video prefill for video vision-language models by using a fixed-capacity recurrent state, improving efficiency for long video understanding.

</details>

<details>
<summary><b>18. OpenSkillEval: Automatically Auditing the Open Skill Ecosystem for LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yixin Cao, Siyuan Liu, Wei Tang, Boxian Ai, Jiahao Ying

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23657) • [📄 arXiv](https://arxiv.org/abs/2605.23657) • [📥 PDF](https://arxiv.org/pdf/2605.23657)

**💻 Code:** [⭐ Code](https://github.com/ALEX-nlp/OpenSkillEval) • [⭐ Code](https://github.com/huggingface)

> OpenSkillEval is an automatic evaluation framework for both skill-augmented agent systems and the skills themselves. It first measures how well different models and agent frameworks handle real downstream tasks — with and without skill augmentatio...

</details>

<details>
<summary><b>19. Light Interaction: Training-Free Inference Acceleration for Interactive Video World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31158) • [📄 arXiv](https://arxiv.org/abs/2605.31158) • [📥 PDF](https://arxiv.org/pdf/2605.31158)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Light Interaction is a training-free inference acceleration framework for interactive video world models that reduces computational costs via adaptive context management and hardware-aware sparse attention.

</details>

<details>
<summary><b>20. Lumos-Nexus: Efficient Frequency Bridging with Homogeneous Latent Space for Video Unified Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yujie Wei, Xinyu Liu, Lingling Cai, Hangjie Yuan, Jiazheng Xing

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31603) • [📄 arXiv](https://arxiv.org/abs/2605.31603) • [📥 PDF](https://arxiv.org/pdf/2605.31603)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/alibaba-damo-academy/Lumos-Custom)

> Lumos-Nexus is a training-efficient unified video generation framework that uses Unified Progressive Frequency Bridging to enhance visual fidelity while maintaining reasoning-driven semantic control, evaluated via the new VR-Bench.

</details>

<details>
<summary><b>21. From Prompt Injection to Persistent Control: Defending Agentic Harness Against Trojan Backdoors</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yiruo Cheng, Yuyang Hu, Xinyu Yang, Zhicheng Dou, Jiejun Tan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.31042) • [📄 arXiv](https://arxiv.org/abs/2605.31042) • [📥 PDF](https://arxiv.org/pdf/2605.31042)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RUC-NLPIR/ClawTrojan)

> LLM agents are evolving from conversational chatbots to operational tools in real-world workspaces. In local agentic harnesses, an LLM can read and write files, call tools, and reuse workspace state across sessions. While such capabilities enhance...

</details>

<details>
<summary><b>22. MAAT: Multi-phase Adapter-Aware Targeted Unlearning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aman Chadha, Vinija Jain, Saksham Thakur, Shubham Gaur, Suryash Yagnik

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30514) • [📄 arXiv](https://arxiv.org/abs/2605.30514) • [📥 PDF](https://arxiv.org/pdf/2605.30514)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> MAAT introduces a structured LoRA-adapter unlearning pipeline plus 5WBENCH, a balanced 5W benchmark, showing that causal “Why” knowledge is uniquely difficult to forget due to long multi-hop answer chains and gradient dilution—not because it has a...

</details>

<details>
<summary><b>23. The Good, the Bad, and the Ugly of Markov Boundary for Tabular Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29411) • [📄 arXiv](https://arxiv.org/abs/2605.29411) • [📥 PDF](https://arxiv.org/pdf/2605.29411)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> What Markov boundary offers to tabular prediction.

</details>

<details>
<summary><b>24. FRAPPE: Full Input, Residual Output Autoencoding with Projection Pursuit Encoder</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Neeraja J. Yadwadkar, Dan Jacobellis

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.28992) • [📄 arXiv](https://arxiv.org/abs/2605.28992) • [📥 PDF](https://arxiv.org/pdf/2605.28992)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://ut-sysml.github.io/FRAPPE

</details>

<details>
<summary><b>25. Frequency-Guided Action Diffusion via Sub-Frequency Manifold Traversal</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27919) • [📄 arXiv](https://arxiv.org/abs/2605.27919) • [📥 PDF](https://arxiv.org/pdf/2605.27919)

**💻 Code:** [⭐ Code](https://github.com/HenryWJL/fgo) • [⭐ Code](https://github.com/huggingface)

> Submit a paper to Daily papers

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 25 |
| 📅 Today | [`2026-06-01.json`](data/daily/2026-06-01.json) | 25 |
| 📆 This Week | [`2026-W22.json`](data/weekly/2026-W22.json) | 25 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 25 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-01 | 25 | [View JSON](data/daily/2026-06-01.json) |
| 📄 2026-05-31 | 59 | [View JSON](data/daily/2026-05-31.json) |
| 📄 2026-05-30 | 59 | [View JSON](data/daily/2026-05-30.json) |
| 📄 2026-05-29 | 24 | [View JSON](data/daily/2026-05-29.json) |
| 📄 2026-05-28 | 19 | [View JSON](data/daily/2026-05-28.json) |
| 📄 2026-05-27 | 17 | [View JSON](data/daily/2026-05-27.json) |
| 📄 2026-05-26 | 15 | [View JSON](data/daily/2026-05-26.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W22 | 25 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 25 | [View JSON](data/monthly/2026-06.json) |
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
