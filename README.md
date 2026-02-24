<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2452+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">18</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">41</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">933</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2452+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 24, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10693) • [📄 arXiv](https://arxiv.org/abs/2602.10693) • [📥 PDF](https://arxiv.org/pdf/2602.10693)

**💻 Code:** [⭐ Code](https://github.com/FloyedShen/VESPO)

> Training stability under off-policy conditions is a critical bottleneck for scaling RL-based LLM training — policy staleness from mini-batch splitting, asynchronous pipelines, and training-inference mismatches all cause importance weights to explo...

</details>

<details>
<summary><b>2. Does Your Reasoning Model Implicitly Know When to Stop Thinking?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08354) • [📄 arXiv](https://arxiv.org/abs/2602.08354) • [📥 PDF](https://arxiv.org/pdf/2602.08354)

> Large reasoning models already implicitly know when they have reached the correct answer. We just don’t let them stop. Project Page: https://hzx122.github.io/sage-rl/

</details>

<details>
<summary><b>3. Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shengqu Cai, Tong Wu, Ashley Neall, Lisong C. Sun, Linxi Xie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18422) • [📄 arXiv](https://arxiv.org/abs/2602.18422) • [📥 PDF](https://arxiv.org/pdf/2602.18422)

> We introduce a human-centric video world model conditioned on head and hand poses, enabling interactive egocentric environments through bidirectional diffusion training and improved user control.

</details>

<details>
<summary><b>4. Spanning the Visual Analogy Space with a Weight Basis of LoRAs</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.15727) • [📄 arXiv](https://arxiv.org/abs/2602.15727) • [📥 PDF](https://arxiv.org/pdf/2602.15727)

**💻 Code:** [⭐ Code](https://github.com/NVlabs/LoRWeB)

> TLDR: We propose a novel modular framework that learns to dynamically mix low-rank adapters (LoRAs) to improve visual analogy learning, enabling flexible and generalizable image edits based on example transformations. We present LoRWeB ( LoR A We ...

</details>

<details>
<summary><b>5. Decoding as Optimisation on the Probability Simplex: From Top-K to Top-P (Nucleus) to Best-of-K Samplers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haitham Bou-Ammar, Matthieu Zimmer, Rasul Tutunov, Xiaotong Ji

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18292) • [📄 arXiv](https://arxiv.org/abs/2602.18292) • [📥 PDF](https://arxiv.org/pdf/2602.18292)

> We show that many sampling strategies for LLMs are a special case of a more general formulation. We then use this to design a new sampler.

</details>

<details>
<summary><b>6. EgoPush: Learning End-to-End Egocentric Multi-Object Rearrangement for Mobile Robots</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sihang Li, Jiaqi Li, Yipeng Wang, Zhexiong Wang, Boyuan An

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18071) • [📄 arXiv](https://arxiv.org/abs/2602.18071) • [📥 PDF](https://arxiv.org/pdf/2602.18071)

> EgoPush trains an object-centric, egocentric RL framework for long-horizon multi-object rearrangement with latent relational states, achieving effective sim-to-real transfer without explicit global state estimation.

</details>

<details>
<summary><b>7. SARAH: Spatially Aware Real-time Agentic Humans</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alexander Richard, Michael Zollhoefer, Zhang Chen, Siwei Zhang, Evonne Ng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18432) • [📄 arXiv](https://arxiv.org/abs/2602.18432) • [📥 PDF](https://arxiv.org/pdf/2602.18432)

> A real-time, fully causal system producing spatially aware, gaze-controlled full-body motion for embodied agents from user position and audio, achieving 300 FPS on streaming VR hardware.

</details>

<details>
<summary><b>8. VidEoMT: Your ViT is Secretly Also a Video Segmentation Model</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17807) • [📄 arXiv](https://arxiv.org/abs/2602.17807) • [📥 PDF](https://arxiv.org/pdf/2602.17807)

**💻 Code:** [⭐ Code](https://github.com/tue-mps/videomt)

> Summary: Video Encoder-only Mask Transformer (VidEoMT), a lightweight encoder-only model for online video segmentation built on a plain Vision Transformer (ViT). It performs both spatial and temporal reasoning within the ViT encoder, without relyi...

</details>

<details>
<summary><b>9. DeepVision-103K: A Visually Diverse, Broad-Coverage, and Verifiable Mathematical Dataset for Multimodal Reasoning</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Wei Wang, Wotao Yin, Bing Zhao, Lizhen Xu, Haoxiang Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16742) • [📄 arXiv](https://arxiv.org/abs/2602.16742) • [📥 PDF](https://arxiv.org/pdf/2602.16742)

**💻 Code:** [⭐ Code](https://github.com/SKYLENAGE-AI/DeepVision-103K)

> Reinforcement Learning with Verifiable Rewards (RLVR) has been shown effective in enhancing the visual reflection and reasoning capabilities of Large Multimodal Models (LMMs). However, existing datasets are predominantly derived from either small-...

</details>

<details>
<summary><b>10. Avey-B</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.15814) • [📄 arXiv](https://arxiv.org/abs/2602.15814) • [📥 PDF](https://arxiv.org/pdf/2602.15814)

**💻 Code:** [⭐ Code](https://github.com/rimads/avey-b)

> Hi @dacharya-avey is there any open implementation available for pretraining these models 🤔

</details>

<details>
<summary><b>11. Learning Smooth Time-Varying Linear Policies with an Action Jacobian Penalty</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jessica Hodgins, Kevin Karol, Zhaoming Xie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18312) • [📄 arXiv](https://arxiv.org/abs/2602.18312) • [📥 PDF](https://arxiv.org/pdf/2602.18312)

> Introduces an action Jacobian penalty with a Linear Policy Net to train smooth, motion-imitation policies and quadruped-arm control, reducing tuning and computational overhead.

</details>

<details>
<summary><b>12. Sink-Aware Pruning for Diffusion Language Models</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17664) • [📄 arXiv](https://arxiv.org/abs/2602.17664) • [📥 PDF](https://arxiv.org/pdf/2602.17664)

**💻 Code:** [⭐ Code](https://github.com/VILA-Lab/Sink-Aware-Pruning)

> Sink-Aware Pruning for Diffusion Language Models identifies and addresses a fundamental blind spot in current pruning recipes for large language models. Most pruning methods are inherited from autoregressive LLMs and assume that attention sink tok...

</details>

<details>
<summary><b>13. Selective Training for Large Vision Language Models via Visual Information Gain</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17186) • [📄 arXiv](https://arxiv.org/abs/2602.17186) • [📥 PDF](https://arxiv.org/pdf/2602.17186)

> This paper introduces Visual Information Gain (VIG), a perplexity-based metric that quantifies how much visual input reduces prediction uncertainty at both sample and token levels, enabling fine-grained measurement of visual grounding and language...

</details>

<details>
<summary><b>14. Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17080) • [📄 arXiv](https://arxiv.org/abs/2602.17080) • [📥 PDF](https://arxiv.org/pdf/2602.17080)

> We propose a new optimizer and its diagonal extension, NAMO and NAMO-D, providing the first theoretically principled integration of orthogonalized momentum with norm-based Adam-type noise adaptation, accompanied by rigorous convergence guarantees....

</details>

<details>
<summary><b>15. ReIn: Conversational Error Recovery with Reasoning Inception</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17022) • [📄 arXiv](https://arxiv.org/abs/2602.17022) • [📥 PDF](https://arxiv.org/pdf/2602.17022)

> Here is a brief summary: Constraint: "You can't touch the agent." We explicitly assume deployment settings where changing model parameters or the system prompt is not allowed (policy, safety validation, product constraints, etc.). So the question ...

</details>

<details>
<summary><b>16. Whom to Query for What: Adaptive Group Elicitation via Multi-Turn LLM Interactions</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.14279) • [📄 arXiv](https://arxiv.org/abs/2602.14279) • [📥 PDF](https://arxiv.org/pdf/2602.14279)

**💻 Code:** [⭐ Code](https://github.com/ZDCSlab/Group-Adaptive-Elicitation)

> 🧠 When we try to understand a group’s true preferences, the challenge is not only what to ask, but whom to query for what. We study a new problem setting: Adaptive Group Elicitation Under real costs and missing data, the system must dynamically de...

</details>

<details>
<summary><b>17. Rubrics as an Attack Surface: Stealthy Preference Drift in LLM Judges</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.13576) • [📄 arXiv](https://arxiv.org/abs/2602.13576) • [📥 PDF](https://arxiv.org/pdf/2602.13576)

**💻 Code:** [⭐ Code](https://github.com/ZDCSlab/Rubrics-as-an-Attack-Surface)

> 🔥 Evaluation rubrics can silently alter alignment. We identify a new failure mode in LLM-as-a-judge pipelines: Rubric-Induced Preference Drift (RIPD) Under RIPD: 👉 Benchmark scores remain stable 👉 Judge preferences shift on unseen domains 👉 Misali...

</details>

<details>
<summary><b>18. 4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chen Change Loy, Xingang Pan, Yushi Lan, Shangchen Zhou, Yihang Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10094) • [📄 arXiv](https://arxiv.org/abs/2602.10094) • [📥 PDF](https://arxiv.org/pdf/2602.10094)

> Project Page: https://yihangluo.com/projects/4RC/

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-02-24.json`](data/daily/2026-02-24.json) | 18 |
| 📆 This Week | [`2026-W08.json`](data/weekly/2026-W08.json) | 41 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 933 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-24 | 18 | [View JSON](data/daily/2026-02-24.json) |
| 📄 2026-02-23 | 23 | [View JSON](data/daily/2026-02-23.json) |
| 📄 2026-02-22 | 23 | [View JSON](data/daily/2026-02-22.json) |
| 📄 2026-02-21 | 23 | [View JSON](data/daily/2026-02-21.json) |
| 📄 2026-02-20 | 18 | [View JSON](data/daily/2026-02-20.json) |
| 📄 2026-02-19 | 25 | [View JSON](data/daily/2026-02-19.json) |
| 📄 2026-02-18 | 35 | [View JSON](data/daily/2026-02-18.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W08 | 41 | [View JSON](data/weekly/2026-W08.json) |
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 933 | [View JSON](data/monthly/2026-02.json) |
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
