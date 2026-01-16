<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-27-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1091+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">27</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">156</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">353</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1091+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 16, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Controlled Self-Evolution for Algorithmic Code Optimization</b> ⭐ 79</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07348) • [📄 arXiv](https://arxiv.org/abs/2601.07348) • [📥 PDF](https://arxiv.org/pdf/2601.07348)

**💻 Code:** [⭐ Code](https://github.com/QuantaAlpha/EvoControl)

> arXiv explained breakdown of this paper 👉 https://arxivexplained.com/papers/controlled-self-evolution-for-algorithmic-code-optimization

</details>

<details>
<summary><b>2. DeepResearchEval: An Automated Framework for Deep Research Task Construction and Agentic Evaluation</b> ⭐ 67</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09688) • [📄 arXiv](https://arxiv.org/abs/2601.09688) • [📥 PDF](https://arxiv.org/pdf/2601.09688)

**💻 Code:** [⭐ Code](https://github.com/Infinity-AILab/DeepResearchEval)

> Deep research systems are widely used for multi-step web research, analysis, and cross-source synthesis, yet their evaluation remains challenging. Existing benchmarks often require annotation-intensive task construction, rely on static evaluation ...

</details>

<details>
<summary><b>3. MAXS: Meta-Adaptive Exploration with LLM Agents</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09259) • [📄 arXiv](https://arxiv.org/abs/2601.09259) • [📥 PDF](https://arxiv.org/pdf/2601.09259)

**💻 Code:** [⭐ Code](https://github.com/exoskeletonzj/MAXS)

> Large Language Model (LLM) Agents exhibit inherent reasoning abilities through the collaboration of multiple tools. However, during agent inference, existing methods often suffer from (i) locally myopic generation, due to the absence of lookahead,...

</details>

<details>
<summary><b>4. A^3-Bench: Benchmarking Memory-Driven Scientific Reasoning via Anchor and Attractor Activation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09274) • [📄 arXiv](https://arxiv.org/abs/2601.09274) • [📥 PDF](https://arxiv.org/pdf/2601.09274)

**💻 Code:** [⭐ Code](https://github.com/exoskeletonzj/A3-Bench)

> Scientific reasoning relies not only on logical inference but also on activating prior knowledge and experiential structures. Memory can efficiently reuse knowledge and enhance reasoning consistency and stability. However, existing benchmarks main...

</details>

<details>
<summary><b>5. Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09088) • [📄 arXiv](https://arxiv.org/abs/2601.09088) • [📥 PDF](https://arxiv.org/pdf/2601.09088)

**💻 Code:** [⭐ Code](https://github.com/D2I-ai/dasd-thinking)

> In this report, we introduce DASD-4B-Thinking, a lightweight yet highly capable, fully open-source reasoning model. It achieves SOTA performance among open-source models of comparable scale across challenging benchmarks in mathematics, scientific ...

</details>

<details>
<summary><b>6. Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09708) • [📄 arXiv](https://arxiv.org/abs/2601.09708) • [📥 PDF](https://arxiv.org/pdf/2601.09708)

> Project page: https://jasper0314-huang.github.io/fast-thinkact/

</details>

<details>
<summary><b>7. SkinFlow: Efficient Information Transmission for Open Dermatological Diagnosis via Dynamic Visual Encoding and Staged RL</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09136) • [📄 arXiv](https://arxiv.org/abs/2601.09136) • [📥 PDF](https://arxiv.org/pdf/2601.09136)

> General-purpose Large Vision-Language Models (LVLMs), despite their massive scale, often falter in dermatology due to "diffuse attention" - the inability to disentangle subtle pathological lesions from background noise. In this paper, we challenge...

</details>

<details>
<summary><b>8. OpenVoxel: Training-Free Grouping and Captioning Voxels for Open-Vocabulary 3D Scene Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09575) • [📄 arXiv](https://arxiv.org/abs/2601.09575) • [📥 PDF](https://arxiv.org/pdf/2601.09575)

> OpenVoxel provides training-free grouping and captioning of sparse voxels for open-vocabulary 3D scene understanding using VLMs/MLLMs and text search, enabling RES and OVS without CLIP embeddings.

</details>

<details>
<summary><b>9. OpenDecoder: Open Large Language Model Decoding to Incorporate Document Quality in RAG</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09028) • [📄 arXiv](https://arxiv.org/abs/2601.09028) • [📥 PDF](https://arxiv.org/pdf/2601.09028)

> OpenDecoder is a novel framework that directly 'opens' the LLM to modify its decoding process within RAG scenarios by leveraging relevance signals from retrieved documents. Through a robustness-oriented training algorithm, the model learns to perf...

</details>

<details>
<summary><b>10. ExpSeek: Self-Triggered Experience Seeking for Web Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08605) • [📄 arXiv](https://arxiv.org/abs/2601.08605) • [📥 PDF](https://arxiv.org/pdf/2601.08605)

> Experience intervention in web agents emerges as a promising technical paradigm, enhancing agent interaction capabilities by providing valuable insights from accumulated experiences. However, existing methods predominantly inject experience passiv...

</details>

<details>
<summary><b>11. EvoFSM: Controllable Self-Evolution for Deep Research with Finite State Machines</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09465) • [📄 arXiv](https://arxiv.org/abs/2601.09465) • [📥 PDF](https://arxiv.org/pdf/2601.09465)

> EvoFSM presents a controllable self-evolution framework using a finite state machine to guide adaptive problem-solving, separating macroscopic flow and microscopic skills with critic-guided updates and reusable priors.

</details>

<details>
<summary><b>12. FocusUI: Efficient UI Grounding via Position-Preserving Visual Token Selection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03928) • [📄 arXiv](https://arxiv.org/abs/2601.03928) • [📥 PDF](https://arxiv.org/pdf/2601.03928)

**💻 Code:** [⭐ Code](https://github.com/showlab/FocusUI)

> TL;DR: High-res UI screenshots (2K/4K) force VLMs to process thousands of visual tokens. Inspired by human vision, which selects only instruction-relevant image patches, FocusUI teaches VLMs where to look in UI screenshots smartly 🔍 📄 Paper: arXiv...

</details>

<details>
<summary><b>13. Are LLMs Vulnerable to Preference-Undermining Attacks (PUA)? A Factorial Analysis Methodology for Diagnosing the Trade-off between Preference Alignment and Real-World Validity</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chi Zhang, Jiawei Shao, Jiangan Chen, Yiliang Song, Hongjun An

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06596) • [📄 arXiv](https://arxiv.org/abs/2601.06596) • [📥 PDF](https://arxiv.org/pdf/2601.06596)

> This paper treats preference undermining as an experimental object, not a vibe. A clean factorial design isolates manipulation factors and quantifies when truth yields to compliance. Conclusion, stated politely: yes, a large model can be PUA-ed, a...

</details>

<details>
<summary><b>14. TranslateGemma Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09012) • [📄 arXiv](https://arxiv.org/abs/2601.09012) • [📥 PDF](https://arxiv.org/pdf/2601.09012)

> TranslateGemma extends Gemma 3 with two-stage fine-tuning (supervised then RL) for multilingual translation, achieving strong WMT performance and multimodal capabilities.

</details>

<details>
<summary><b>15. Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenjie Li, Beichen Guo, Hanlin Wang, Youwei Liu, jwanglvy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08955) • [📄 arXiv](https://arxiv.org/abs/2601.08955) • [📥 PDF](https://arxiv.org/pdf/2601.08955)

> TL;DR: An agent learning framework via lookahead imagination, where an agent's policy model interacts with the learned world model, yielding multi-step "imagined" trajectories. This imagination is conducted via a novel adaptive lookahead mechanism...

</details>

<details>
<summary><b>16. Efficient Camera-Controlled Video Generation of Static Scenes via Sparse Diffusion and 3D Rendering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ayush Tewari, Joan Lasenby, Jeffrey Hu, Jieying Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09697) • [📄 arXiv](https://arxiv.org/abs/2601.09697) • [📥 PDF](https://arxiv.org/pdf/2601.09697)

> Proposes SRENDER: generate sparse diffusion keyframes for static scenes and render 3D views to produce long videos fast and consistently.

</details>

<details>
<summary><b>17. Geometric Stability: The Missing Axis of Representations</b> ⭐ 0</summary>

<br/>

**👥 Authors:** pcr2120

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09173) • [📄 arXiv](https://arxiv.org/abs/2601.09173) • [📥 PDF](https://arxiv.org/pdf/2601.09173)

**💻 Code:** [⭐ Code](https://github.com/prashantcraju/shesha?tab=readme-ov-file#tutorials) • [⭐ Code](https://github.com/prashantcraju/geometric-stability)

> DeepSeek got it half right with their mHC paper: stability matters for scaling. But they only measure stability DURING training. What about the stability of what models LEARN? I built Shesha to measure this - a geometric stability metric with SOTA...

</details>

<details>
<summary><b>18. The AI Hippocampus: How Far are We From Human Memory?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tong Wu, Yuxuan Wang, Yipeng Kang, Jiaqi Li, Zixia Jia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09113) • [📄 arXiv](https://arxiv.org/abs/2601.09113) • [📥 PDF](https://arxiv.org/pdf/2601.09113)

> Survey of memory in LLMs and multimodal models, detailing implicit, explicit, and agentic memory, architectures, benchmarks, and challenges in persistence, alignment, and cross-modal retrieval.

</details>

<details>
<summary><b>19. Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Thomas Anderson Keller, Yilun Du, Fangneng Zhan, Benhao Huang, Hansen Jin Lillemark

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01075) • [📄 arXiv](https://arxiv.org/abs/2601.01075) • [📥 PDF](https://arxiv.org/pdf/2601.01075)

**💻 Code:** [⭐ Code](https://github.com/hlillemark/flowm)

> No abstract available.

</details>

<details>
<summary><b>20. DPWriter: Reinforcement Learning with Diverse Planning Branching for Creative Writing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruihua Song, Yi Zhao, Wei Bi, Yahui Liu, Qian Cao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09609) • [📄 arXiv](https://arxiv.org/abs/2601.09609) • [📥 PDF](https://arxiv.org/pdf/2601.09609)

> Reinforcement learning (RL)-based enhancement of large language models (LLMs) often leads to reduced output diversity, undermining their utility in open-ended tasks like creative writing. Current methods lack explicit mechanisms for guiding divers...

</details>

<details>
<summary><b>21. Omni-R1: Towards the Unified Generative Paradigm for Multimodal Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09536) • [📄 arXiv](https://arxiv.org/abs/2601.09536) • [📥 PDF](https://arxiv.org/pdf/2601.09536)

> This paper proposes a unified generative multimodal reasoning paradigm, using a two-stage SFT+RL framework with perception alignment loss and perception reward, and explores bootstrapping step-wise visualizations from text-only reasoning data when...

</details>

<details>
<summary><b>22. Focal Guidance: Unlocking Controllability from Semantic-Weak Layers in Video Diffusion Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiao Yang, Kaipeng Zhang, Shenghai Yuan, Yuanyang Yin, yfdeng10

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07287) • [📄 arXiv](https://arxiv.org/abs/2601.07287) • [📥 PDF](https://arxiv.org/pdf/2601.07287)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API V-Warper: Appearance-Consistent Video Diffusion Personalization via Value W...

</details>

<details>
<summary><b>23. No More Stale Feedback: Co-Evolving Critics for Open-World Agent Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yixia Li, Xingchen Zeng, Yulan Hu, Lingjie Jiang, Zhicong Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06794) • [📄 arXiv](https://arxiv.org/abs/2601.06794) • [📥 PDF](https://arxiv.org/pdf/2601.06794)

> Critique-guided reinforcement learning (RL) has emerged as a powerful paradigm for training LLM agents by augmenting sparse outcome rewards with natural-language feedback. However, current methods often rely on static or offline critic models, whi...

</details>

<details>
<summary><b>24. SCALER:Synthetic Scalable Adaptive Learning Environment for Reasoning</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Yixin Cao, Xinrun Wang, Zhongyuan Peng, Changyi Xiao, SII-Molu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04809) • [📄 arXiv](https://arxiv.org/abs/2601.04809) • [📥 PDF](https://arxiv.org/pdf/2601.04809)

**💻 Code:** [⭐ Code](https://github.com/molumolua/SCALER)

> Scalable Environment Synthesis Given a programming problem (statement + reference solution), SCALER synthesizes a reasoning environment with: Verifiability: deterministic oracle / unit tests provide correctness signals. Difficulty control: explici...

</details>

<details>
<summary><b>25. Cluster Workload Allocation: Semantic Soft Affinity Using Natural Language Processing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jolanta Mizeria-Pietraszko, lsliwko

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09282) • [📄 arXiv](https://arxiv.org/abs/2601.09282) • [📥 PDF](https://arxiv.org/pdf/2601.09282)

> Cluster workload allocation often requires complex configurations, creating a usability gap. This paper introduces a semantic, intent-driven scheduling paradigm for cluster systems using Natural Language Processing. The system employs a Large Lang...

</details>

<details>
<summary><b>26. sui-1: Grounded and Verifiable Long-Form Summarization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08472) • [📄 arXiv](https://arxiv.org/abs/2601.08472) • [📥 PDF](https://arxiv.org/pdf/2601.08472)

> No abstract available.

</details>

<details>
<summary><b>27. SampoNLP: A Self-Referential Toolkit for Morphological Analysis of Subword Tokenizers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aleksey Komissarov, Ekaterina Chelombitko, Iaroslav Chelombitko

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04469) • [📄 arXiv](https://arxiv.org/abs/2601.04469) • [📥 PDF](https://arxiv.org/pdf/2601.04469)

**💻 Code:** [⭐ Code](https://github.com/AragonerUA/SampoNLP)

> The quality of subword tokenization is critical for Large Language Models, yet evaluating tokenizers for morphologically rich Uralic languages is hampered by the lack of clean morpheme lexicons. We introduce SampoNLP, a corpus-free toolkit for mor...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 27 |
| 📅 Today | [`2026-01-16.json`](data/daily/2026-01-16.json) | 27 |
| 📆 This Week | [`2026-W02.json`](data/weekly/2026-W02.json) | 156 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 353 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-16 | 27 | [View JSON](data/daily/2026-01-16.json) |
| 📄 2026-01-15 | 24 | [View JSON](data/daily/2026-01-15.json) |
| 📄 2026-01-14 | 42 | [View JSON](data/daily/2026-01-14.json) |
| 📄 2026-01-13 | 30 | [View JSON](data/daily/2026-01-13.json) |
| 📄 2026-01-12 | 33 | [View JSON](data/daily/2026-01-12.json) |
| 📄 2026-01-11 | 33 | [View JSON](data/daily/2026-01-11.json) |
| 📄 2026-01-10 | 33 | [View JSON](data/daily/2026-01-10.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W02 | 156 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 353 | [View JSON](data/monthly/2026-01.json) |
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
