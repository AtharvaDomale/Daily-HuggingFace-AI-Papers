<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-23-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5181+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">79</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">166</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5181+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** July 12, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Vidu S1: A Real-Time Interactive Video Generation Model</b> ⭐ 158</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.03118) • [📄 arXiv](https://arxiv.org/abs/2607.03118) • [📥 PDF](https://arxiv.org/pdf/2607.03118)

**💻 Code:** [⭐ Code](https://github.com/shengshu-ai/Vidu-S1) • [⭐ Code](https://github.com/huggingface)

> Highly recommend giving Vidu S1 a try~ User Guide : https://auspicious-passive-36a.notion.site/Vidu-S1-Introduction-34324005a3e48009b3f3d9c07c79d83b Try it now : https://www.vidu.com/vidu-stream API Platform : https://platform.vidu.com/live/landin...

</details>

<details>
<summary><b>2. Video-Oasis: Rethinking Evaluation of Video Understanding</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.29616) • [📄 arXiv](https://arxiv.org/abs/2603.29616) • [📥 PDF](https://arxiv.org/pdf/2603.29616)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sejong-rcv/Video-Oasis)

> TL;DR. Video-Oasis rethinks the current benchmark landscape by examining whether proliferating video benchmarks truly satisfy shared criteria for genuine video understanding.

</details>

<details>
<summary><b>3. Why Can't I Open My Drawer? Mitigating Object-Driven Shortcuts in Zero-Shot Compositional Action Recognition</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16211) • [📄 arXiv](https://arxiv.org/abs/2601.16211) • [📥 PDF](https://arxiv.org/pdf/2601.16211)

**💻 Code:** [⭐ Code](https://github.com/KHU-VLL/RCORE) • [⭐ Code](https://github.com/huggingface)

> TL;DR: We quantify how object-driven shortcuts sabotage compositional generalization in video understanding. A model that has seen "Open Window" and "Close Drawer" should be able to recognize "Open Drawer"—an unseen yet plausible verb–object combi...

</details>

<details>
<summary><b>4. Ideas Have Genomes: Benchmarking Scientific Lineage Reasoning and Lineage-Grounded Idea Generation</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08758) • [📄 arXiv](https://arxiv.org/abs/2607.08758) • [📥 PDF](https://arxiv.org/pdf/2607.08758)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VisionXLab/IdeasHaveGenomes)

> Step length seems underappreciated from the image, can be measured as bits of fuzzing needed, maybe as a "removal of idea from trained model direction", and then probably from a positive direction as well, though that one seems more initialized an...

</details>

<details>
<summary><b>5. LongE2V: Long-Horizon Event-based Video Reconstruction, Prediction, and Frame Interpolation with Video Diffusion Models</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08770) • [📄 arXiv](https://arxiv.org/abs/2607.08770) • [📥 PDF](https://arxiv.org/pdf/2607.08770)

**💻 Code:** [⭐ Code](https://github.com/cdfan0627/LongE2V) • [⭐ Code](https://github.com/huggingface)

> Recovering high-quality video from sparse event streams is a challenging task. Regression methods often blur textures, while existing generative models struggle with long-term stability. We propose LongE2V, a novel approach that leverages pre-trai...

</details>

<details>
<summary><b>6. UniClawBench: A Universal Benchmark for Proactive Agents on Real-World Tasks</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08768) • [📄 arXiv](https://arxiv.org/abs/2607.08768) • [📥 PDF](https://arxiv.org/pdf/2607.08768)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HKU-MMLab/UniClawBench)

> The rapid development of large language models and multimodal large language models has accelerated the emergence of proactive agents capable of operating everyday tools and assisting users in real-world environments. However, existing benchmarks ...

</details>

<details>
<summary><b>7. Enhancing In-context Panoramic Generation via Geometric-aware Pretraining</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08765) • [📄 arXiv](https://arxiv.org/abs/2607.08765) • [📥 PDF](https://arxiv.org/pdf/2607.08765)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Insta360-Research-Team/Canvas360)

> Project page： https://zry000.github.io/Canvas360/ GitHub repo： https://github.com/Insta360-Research-Team/Canvas360

</details>

<details>
<summary><b>8. DrugGen 2: A disease-aware language model for enhancing drug discovery</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Matin Irajpour, Navid Mazrouei, Mahsa Sheikholeslami, Mohammadreza Ghaffarzadeh-Esfahani, alimotahharynia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08404) • [📄 arXiv](https://arxiv.org/abs/2607.08404) • [📥 PDF](https://arxiv.org/pdf/2607.08404)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/alimotahharynia/DrugGen-2)

> Current computational approaches for drug design typically focus on generating molecules conditioned on specific targets or general molecular properties, often neglecting the influence of disease context on target behavior and therapeutic outcomes...

</details>

<details>
<summary><b>9. Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.07740) • [📄 arXiv](https://arxiv.org/abs/2607.07740) • [📥 PDF](https://arxiv.org/pdf/2607.07740)

**💻 Code:** [⭐ Code](https://github.com/jet-ai-projects/jet-long) • [⭐ Code](https://github.com/huggingface)

> Why stop at 2?

</details>

<details>
<summary><b>10. CineMobile: On-Device Image-to-Video Diffusion for Cinematic Camera Motion Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.03803) • [📄 arXiv](https://arxiv.org/abs/2607.03803) • [📥 PDF](https://arxiv.org/pdf/2607.03803)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The growing demand for image-to-video creation on mobile devices has increasingly focused on cinematic motion effects like bullet time, dolly zoom, slow motion, etc. While Diffusion Transformers (DiTs) exhibit strong performance in video generatio...

</details>

<details>
<summary><b>11. OpenCoF: Learning to Reason Through Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08763) • [📄 arXiv](https://arxiv.org/abs/2607.08763) • [📥 PDF](https://arxiv.org/pdf/2607.08763)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VGenST-Bench: A Benchmark for Spatio-Temporal Reasoning via Active Video Sy...

</details>

<details>
<summary><b>12. Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08716) • [📄 arXiv](https://arxiv.org/abs/2607.08716) • [📥 PDF](https://arxiv.org/pdf/2607.08716)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yifannnwu/proactive-memory-agent)

> No abstract available.

</details>

<details>
<summary><b>13. Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.07953) • [📄 arXiv](https://arxiv.org/abs/2607.07953) • [📥 PDF](https://arxiv.org/pdf/2607.07953)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/tommasocerruti/linear-attention-architectures)

> We compare several recent recurrent linear-attention architectures —DeltaNet, Gated DeltaNet, Kimi Delta Attention, and Gated DeltaNet-2 — in a shared recurrent-memory notation and empirical setup. The report focuses on practical trade-offs among ...

</details>

<details>
<summary><b>14. ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08741) • [📄 arXiv](https://arxiv.org/abs/2607.08741) • [📥 PDF](https://arxiv.org/pdf/2607.08741)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Cont...

</details>

<details>
<summary><b>15. A Sparse and Truncated State Vector Simulator for Peaked Circuits</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.07816) • [📄 arXiv](https://arxiv.org/abs/2607.07816) • [📥 PDF](https://arxiv.org/pdf/2607.07816)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/diogoff/qstvec)

> Introduces a sparse and truncated state‑vector simulator for peaked quantum circuits, where the goal is to find the most probable output bit string. By storing only nonzero amplitudes and by vectorizing all operations (with optional GPU accelerati...

</details>

<details>
<summary><b>16. UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.06987) • [📄 arXiv](https://arxiv.org/abs/2607.06987) • [📥 PDF](https://arxiv.org/pdf/2607.06987)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper introduces Unbounded Positive Asymmetric Optimization (UP), a plug-and-play reinforcement learning objective that overcomes the exploration-stability dilemma and effectively mitigates entropy collapse by allowing unclipped gradients for...

</details>

<details>
<summary><b>17. SAM-MT: Real-Time Interactive Multi-Target Video Segmentation</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08688) • [📄 arXiv](https://arxiv.org/abs/2607.08688) • [📥 PDF](https://arxiv.org/pdf/2607.08688)

**💻 Code:** [⭐ Code](https://github.com/FudanCVL/SAM-MT) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>18. PhyMRI-SR: Toward Physics-Aware MRI Image Super-Resolution</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.06238) • [📄 arXiv](https://arxiv.org/abs/2607.06238) • [📥 PDF](https://arxiv.org/pdf/2607.06238)

**💻 Code:** [⭐ Code](https://github.com/weilihua0205/PhyMRI-SR) • [⭐ Code](https://github.com/huggingface)

> This paper reframes MRI super-resolution as a physics-aware problem, recognizing that resolution and SNR are inherently coupled in MRI acquisition, making resolution dynamic rather than fixed. The authors adapt 2D Gaussian Splatting as a resolutio...

</details>

<details>
<summary><b>19. Flash-BoN: Instant Drafts for Inference-Time Scaling in Diffusion Models</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.04461) • [📄 arXiv](https://arxiv.org/abs/2607.04461) • [📥 PDF](https://arxiv.org/pdf/2607.04461)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/flash-bon/flash-bon)

> Budget-constrained, verifier-guided best-of-N generation for diffusion models. Given a prompt and a wall-clock budget, Flash-BoN drafts cheap candidates (TaylorSeer-style caching + layer/timestep skipping), selects the best with a pairwise vision-...

</details>

<details>
<summary><b>20. A Quantized Native Runtime for On-Device Semantic Audio Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08526) • [📄 arXiv](https://arxiv.org/abs/2607.08526) • [📥 PDF](https://arxiv.org/pdf/2607.08526)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/matteospanio/aria)

> We release aria , a native inference engine for audio diffusion models, written from scratch in C with no third-party dependencies. It runs Stable Audio 3 (small-music and medium) end to end, from a text prompt to a stereo WAV, with no Python or d...

</details>

<details>
<summary><b>21. CausalDS: Benchmarking Causal Reasoning in Data-Science Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.08093) • [📄 arXiv](https://arxiv.org/abs/2607.08093) • [📥 PDF](https://arxiv.org/pdf/2607.08093)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/andleb/causalds)

> We introduce CausalDS , a benchmark generator for agentic causal data science. Rather than relying on a fixed collection of examples, CausalDS generates fresh hidden causal graphs and SCMs, synthetic tabular data, graph-audited natural-language st...

</details>

<details>
<summary><b>22. Can Dialects Be Steered Like Languages? Sparse Neurons and Distributed Directions in Arabic LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.03936) • [📄 arXiv](https://arxiv.org/abs/2607.03936) • [📥 PDF](https://arxiv.org/pdf/2607.03936)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/mbzuai-nlp/arabic-dialect-steering)

> A key challenge in Arabic NLP is the scarcity of dialectal data relative to Modern Standard Arabic (MSA), causing LLMs to overproduce MSA and struggle with dialectally accurate generation. From an interpretability perspective, this raises a fundam...

</details>

<details>
<summary><b>23. PAST-TIDE: Prototype-Anchored Statement Tuning with Topic-Invariant Normalization for Stance Detection</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Fakhri Karray, Md. Milon Islam, MD Jahid Hasan Jim, Md. Shakhoyat Rahman Shujon, mr3haque

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.04690) • [📄 arXiv](https://arxiv.org/abs/2607.04690) • [📥 PDF](https://arxiv.org/pdf/2607.04690)

**💻 Code:** [⭐ Code](https://github.com/Shakhoyat/PAST-TIDE) • [⭐ Code](https://github.com/huggingface)

> Published in The Fifteenth Language Resources and Evaluation Conference (LREC 2026)

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 23 |
| 📅 Today | [`2026-07-12.json`](data/daily/2026-07-12.json) | 23 |
| 📆 This Week | [`2026-W27.json`](data/weekly/2026-W27.json) | 79 |
| 🗓️ This Month | [`2026-07.json`](data/monthly/2026-07.json) | 166 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-07-12 | 23 | [View JSON](data/daily/2026-07-12.json) |
| 📄 2026-07-11 | 23 | [View JSON](data/daily/2026-07-11.json) |
| 📄 2026-07-10 | 8 | [View JSON](data/daily/2026-07-10.json) |
| 📄 2026-07-09 | 5 | [View JSON](data/daily/2026-07-09.json) |
| 📄 2026-07-08 | 8 | [View JSON](data/daily/2026-07-08.json) |
| 📄 2026-07-07 | 9 | [View JSON](data/daily/2026-07-07.json) |
| 📄 2026-07-06 | 3 | [View JSON](data/daily/2026-07-06.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W27 | 79 | [View JSON](data/weekly/2026-W27.json) |
| 📅 2026-W26 | 117 | [View JSON](data/weekly/2026-W26.json) |
| 📅 2026-W25 | 108 | [View JSON](data/weekly/2026-W25.json) |
| 📅 2026-W24 | 130 | [View JSON](data/weekly/2026-W24.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-07 | 166 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |

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
