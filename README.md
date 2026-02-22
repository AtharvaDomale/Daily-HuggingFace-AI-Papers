<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-23-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2411+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">197</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">892</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2411+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 22, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. SpargeAttention2: Trainable Sparse Attention via Hybrid Top-k+Top-p Masking and Distillation Fine-Tuning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.13515) • [📄 arXiv](https://arxiv.org/abs/2602.13515) • [📥 PDF](https://arxiv.org/pdf/2602.13515)

**💻 Code:** [⭐ Code](https://github.com/thu-ml/SpargeAttn)

> SpargeAttention2 reaches 95% attention sparsity and a 16.2× attention speedup while maintaining generation quality, consistently outperforming prior sparse attention methods.

</details>

<details>
<summary><b>2. Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16855) • [📄 arXiv](https://arxiv.org/abs/2602.16855) • [📥 PDF](https://arxiv.org/pdf/2602.16855)

**💻 Code:** [⭐ Code](https://github.com/X-PLUG/MobileAgent) • [⭐ Code](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v3.5)

> The paper introduces GUI-Owl-1.5, the latest native GUI agent model that features instruct/thinking variants in multiple sizes (2B/4B/8B/32B/235B) and supports a range of platforms (desktop, mobile, browser, and more) to enable cloud-edge collabor...

</details>

<details>
<summary><b>3. Unified Latents (UL): How to train your latents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17270) • [📄 arXiv](https://arxiv.org/abs/2602.17270) • [📥 PDF](https://arxiv.org/pdf/2602.17270)

> Unified Latents (UL) jointly regularizes encoders with a diffusion prior and decodes with a diffusion model, giving a tight latent bitrate bound and strong ImageNet/Kinetics performance.

</details>

<details>
<summary><b>4. Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report v1.5</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.14457) • [📄 arXiv](https://arxiv.org/abs/2602.14457) • [📥 PDF](https://arxiv.org/pdf/2602.14457)

> Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report v1.5

</details>

<details>
<summary><b>5. Arcee Trinity Large Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17004) • [📄 arXiv](https://arxiv.org/abs/2602.17004) • [📥 PDF](https://arxiv.org/pdf/2602.17004)

> arXivLens breakdown of this paper 👉 https://arxivlens.com/PaperView/Details/arcee-trinity-large-technical-report-2819-97270046 Executive Summary Detailed Breakdown Practical Applications

</details>

<details>
<summary><b>6. Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Greg Durrett, Nicholas Tomlin, Wenxuan Ding

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16699) • [📄 arXiv](https://arxiv.org/abs/2602.16699) • [📥 PDF](https://arxiv.org/pdf/2602.16699)

**💻 Code:** [⭐ Code](https://github.com/Wenwen-D/env-explorer)

> We introduce Calibrate-Then-Act, a framework that induces an LLM to reason about the cost-uncertainty tradeoff when exploring the environments.

</details>

<details>
<summary><b>7. "What Are You Doing?": Effects of Intermediate Feedback from Agentic LLM In-Car Assistants During Multi-Step Processing</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.15569) • [📄 arXiv](https://arxiv.org/abs/2602.15569) • [📥 PDF](https://arxiv.org/pdf/2602.15569)

**💻 Code:** [⭐ Code](https://github.com/johanneskirmayr/agentic_llm_feedback)

> As LLM agents move into real products, feedback timing and verbosity become critical UX decisions, yet there's little empirical work on this, especially in attention-critical contexts. How are you handling progress communication in your own agenti...

</details>

<details>
<summary><b>8. DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16968) • [📄 arXiv](https://arxiv.org/abs/2602.16968) • [📥 PDF](https://arxiv.org/pdf/2602.16968)

> Dynamic tokenization adapts patch sizes across diffusion denoising steps to accelerate DiTs for image and video generation while preserving quality.

</details>

<details>
<summary><b>9. TactAlign: Human-to-Robot Policy Transfer via Tactile Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.13579) • [📄 arXiv](https://arxiv.org/abs/2602.13579) • [📥 PDF](https://arxiv.org/pdf/2602.13579)

> Human demonstrations collected by wearable devices (e.g., tactile gloves) provide fast and dexterous supervision for policy learning, and are guided by rich, natural tactile feedback. However, a key challenge is how to transfer human-collected tac...

</details>

<details>
<summary><b>10. Computer-Using World Model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** John Zhang, Yiming Guan, wulu, vyokky, ruiyu0

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17365) • [📄 arXiv](https://arxiv.org/abs/2602.17365) • [📥 PDF](https://arxiv.org/pdf/2602.17365)

> CUWM learns a two-stage desktop UI world model that predicts textual state changes and renders next screenshots, enabling action search to improve decision quality in office software.

</details>

<details>
<summary><b>11. On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16849) • [📄 arXiv](https://arxiv.org/abs/2602.16849) • [📥 PDF](https://arxiv.org/pdf/2602.16849)

**💻 Code:** [⭐ Code](https://github.com/Y-Agent/modular-addition-feature-learning)

> A mechanistic interpretation and dynamical analysis of modular addition in two-layer networks, supported by empirical and theoretical evidence.

</details>

<details>
<summary><b>12. 2Mamba2Furious: Linear in Complexity, Competitive in Accuracy</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17363) • [📄 arXiv](https://arxiv.org/abs/2602.17363) • [📥 PDF](https://arxiv.org/pdf/2602.17363)

**💻 Code:** [⭐ Code](https://github.com/gmongaras/2Mamba2Furious)

> Linear attention transformers have become a strong alternative to softmax attention due to their efficiency. However, linear attention tends to be less expressive and results in reduced accuracy compared to softmax attention. To bridge the accurac...

</details>

<details>
<summary><b>13. ArXiv-to-Model: A Practical Study of Scientific LM Training</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17288) • [📄 arXiv](https://arxiv.org/abs/2602.17288) • [📥 PDF](https://arxiv.org/pdf/2602.17288)

**💻 Code:** [⭐ Code](https://github.com/kitefishai/KiteFish-A1-1.5B-Math)

> 💻 Github: https://github.com/kitefishai/KiteFish-A1-1.5B-Math

</details>

<details>
<summary><b>14. Discovering Multiagent Learning Algorithms with Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16928) • [📄 arXiv](https://arxiv.org/abs/2602.16928) • [📥 PDF](https://arxiv.org/pdf/2602.16928)

> We present AlphaEvolve, an LLM-powered evolutionary agent that discovers novel MARL algorithms, including VAD-CFR and SHOR-PSRO, improving regret-based methods and PSRO meta-solver performance.

</details>

<details>
<summary><b>15. FRAPPE: Infusing World Modeling into Generalist Policies via Multiple Future Representation Alignment</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Shuai Chen, Wenxuan Song, wangdonglin130, hhhJB, han1997

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17259) • [📄 arXiv](https://arxiv.org/abs/2602.17259) • [📥 PDF](https://arxiv.org/pdf/2602.17259)

**💻 Code:** [⭐ Code](https://github.com/OpenHelix-Team/frappe)

> models: https://huggingface.co/collections/hhhJB/frappe

</details>

<details>
<summary><b>16. NESSiE: The Necessary Safety Benchmark -- Identifying Errors that should not Exist</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Jonas Geiping, Johannes Bertram

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16756) • [📄 arXiv](https://arxiv.org/abs/2602.16756) • [📥 PDF](https://arxiv.org/pdf/2602.16756)

**💻 Code:** [⭐ Code](https://github.com/JohannesBertram/NESSiE)

> No abstract available.

</details>

<details>
<summary><b>17. CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.15823) • [📄 arXiv](https://arxiv.org/abs/2602.15823) • [📥 PDF](https://arxiv.org/pdf/2602.15823)

**💻 Code:** [⭐ Code](https://github.com/zarifikram/CrispEdit)

> CrispEdit enables continual LLM updates by projecting edits into low-curvature directions of capability loss, so you can apply thousands of knowledge edits without catastrophic forgetting. It achieves strong performance on targeted edits in autore...

</details>

<details>
<summary><b>18. World Models for Policy Refinement in StarCraft II</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.14857) • [📄 arXiv](https://arxiv.org/abs/2602.14857) • [📥 PDF](https://arxiv.org/pdf/2602.14857)

**💻 Code:** [⭐ Code](https://github.com/yxzzhang/StarWM)

> TLDR: This work proposes StarWM, the first world model for SC2 that predicts future observations under partial observability, and proposes StarWM-Agent, a world-model-augmented decision system that integrates StarWM into a Generate--Simulate--Refi...

</details>

<details>
<summary><b>19. Modeling Distinct Human Interaction in Web Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17588) • [📄 arXiv](https://arxiv.org/abs/2602.17588) • [📥 PDF](https://arxiv.org/pdf/2602.17588)

**💻 Code:** [⭐ Code](https://github.com/oaishi/PlowPilot)

> We introduce the task of modeling human intervention to support collaborative web task execution. We collect CowCorpus, a dataset of 400 real-user web navigation trajectories containing over 4,200 interleaved human and agent actions. We identify f...

</details>

<details>
<summary><b>20. References Improve LLM Alignment in Non-Verifiable Domains</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16802) • [📄 arXiv](https://arxiv.org/abs/2602.16802) • [📥 PDF](https://arxiv.org/pdf/2602.16802)

**💻 Code:** [⭐ Code](https://github.com/yale-nlp/RLRR)

> While Reinforcement Learning with Verifiable Rewards (RLVR) has shown strong effectiveness in reasoning tasks, it cannot be directly applied to non-verifiable domains lacking ground-truth verifiers, such as LLM alignment. In this work, we investig...

</details>

<details>
<summary><b>21. Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yan Song, Fengfa Li, Yifeng Ding, Jiwen Jiang, Luoyang Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10377) • [📄 arXiv](https://arxiv.org/abs/2602.10377) • [📥 PDF](https://arxiv.org/pdf/2602.10377)

> 🚩 Hardware is like a box — different sizes of boxes can accommodate different sizes of objects. The same holds for large language models: different hardware platforms impose different architectural constraints and opportunities. This paper introdu...

</details>

<details>
<summary><b>22. StereoAdapter-2: Globally Structure-Consistent Underwater Stereo Depth Estimation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16915) • [📄 arXiv](https://arxiv.org/abs/2602.16915) • [📥 PDF](https://arxiv.org/pdf/2602.16915)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/StereoAdapter-2)

> https://aigeeksgroup.github.io/StereoAdapter-2

</details>

<details>
<summary><b>23. NeST: Neuron Selective Tuning for LLM Safety</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16835) • [📄 arXiv](https://arxiv.org/abs/2602.16835) • [📥 PDF](https://arxiv.org/pdf/2602.16835)

> We introduce NeST, a lightweight and structure-aware safety alignment framework that selectively adapts clustered safety-relevant neurons to deliver stable, efficient, and model-agnostic refusal strengthening—achieving a 90.2% reduction in unsafe ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 23 |
| 📅 Today | [`2026-02-22.json`](data/daily/2026-02-22.json) | 23 |
| 📆 This Week | [`2026-W07.json`](data/weekly/2026-W07.json) | 197 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 892 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-22 | 23 | [View JSON](data/daily/2026-02-22.json) |
| 📄 2026-02-21 | 23 | [View JSON](data/daily/2026-02-21.json) |
| 📄 2026-02-20 | 18 | [View JSON](data/daily/2026-02-20.json) |
| 📄 2026-02-19 | 25 | [View JSON](data/daily/2026-02-19.json) |
| 📄 2026-02-18 | 35 | [View JSON](data/daily/2026-02-18.json) |
| 📄 2026-02-17 | 32 | [View JSON](data/daily/2026-02-17.json) |
| 📄 2026-02-16 | 41 | [View JSON](data/daily/2026-02-16.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 892 | [View JSON](data/monthly/2026-02.json) |
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
