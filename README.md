<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2748+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">21</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">153</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">181</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2748+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 06, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Helios: Real Real-Time Long Video Generation Model</b> ⭐ 462</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04379) • [📄 arXiv](https://arxiv.org/abs/2603.04379) • [📥 PDF](https://arxiv.org/pdf/2603.04379)

**💻 Code:** [⭐ Code](https://github.com/sgl-project/sglang/pull/19782) • [⭐ Code](https://github.com/PKU-YuanGroup/Helios) • [⭐ Code](https://github.com/vllm-project/vllm-omni/pull/1604)

> Code: https://github.com/PKU-YuanGroup/Helios Page: https://pku-yuangroup.github.io/Helios-Page/ Inference Speed:

</details>

<details>
<summary><b>2. T2S-Bench & Structure-of-Thought: Benchmarking and Prompting Comprehensive Text-to-Structure Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** linyueqian, Zishan-Shao, MartinKu, JinghanKe, Qinsi1

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03790) • [📄 arXiv](https://arxiv.org/abs/2603.03790) • [📥 PDF](https://arxiv.org/pdf/2603.03790)

> No abstract available.

</details>

<details>
<summary><b>3. Heterogeneous Agent Collaborative Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02604) • [📄 arXiv](https://arxiv.org/abs/2603.02604) • [📥 PDF](https://arxiv.org/pdf/2603.02604)

**💻 Code:** [⭐ Code](https://github.com/Fred990807/HACRL)

> We formalize a new reinforcement learning paradigm HACRL to allow rollout share among heterogeneous agents, further we propose a new algorithm HACPO based on GSPO. Project Page: https://zzx-peter.github.io/hacrl/ We are glad to share our code. Ple...

</details>

<details>
<summary><b>4. Proact-VL: A Proactive VideoLLM for Real-Time AI Companions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03447) • [📄 arXiv](https://arxiv.org/abs/2603.03447) • [📥 PDF](https://arxiv.org/pdf/2603.03447)

> Proact-VL presents a proactive, real-time VideoLLM framework enabling low-latency multimodal agents with autonomous response control, evaluated on Live Gaming Benchmark across commentary and user-guidance scenarios.

</details>

<details>
<summary><b>5. MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning</b> ⭐ 21</summary>

<br/>

**👥 Authors:** Liancheng Zhang, jrwen, ariya2357, douzc, zstanjj

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03379) • [📄 arXiv](https://arxiv.org/abs/2603.03379) • [📥 PDF](https://arxiv.org/pdf/2603.03379)

**💻 Code:** [⭐ Code](https://github.com/plageon/MemSifter)

> As Large Language Models (LLMs) are increasingly used for long-duration tasks, maintaining effective long-term memory has become a critical challenge. Current methods often face a trade-off between cost and accuracy. Simple storage methods often f...

</details>

<details>
<summary><b>6. ArtHOI: Articulated Human-Object Interaction Synthesis by 4D Reconstruction from Video Priors</b> ⭐ 14</summary>

<br/>

**👥 Authors:** liuziwei7, lxxiao, simon123905, FrozenBurning, tqliu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04338) • [📄 arXiv](https://arxiv.org/abs/2603.04338) • [📥 PDF](https://arxiv.org/pdf/2603.04338)

**💻 Code:** [⭐ Code](https://github.com/Inso-13/ArtHOI)

> Synthesizing physically plausible articulated human-object interactions (HOI) without 3D/4D supervision remains a fundamental challenge. While recent zero-shot approaches leverage video diffusion models to synthesize human-object interactions, the...

</details>

<details>
<summary><b>7. Phi-4-reasoning-vision-15B Technical Report</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03975) • [📄 arXiv](https://arxiv.org/abs/2603.03975) • [📥 PDF](https://arxiv.org/pdf/2603.03975)

**💻 Code:** [⭐ Code](https://github.com/microsoft/Phi-4-reasoning-vision-15B)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Me...

</details>

<details>
<summary><b>8. CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video</b> ⭐ 39</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04291) • [📄 arXiv](https://arxiv.org/abs/2603.04291) • [📥 PDF](https://arxiv.org/pdf/2603.04291)

**💻 Code:** [⭐ Code](https://github.com/TencentARC/CubeComposer)

> TL;DR: CubeComposer generates 360° video from perspective videos in a cubemap face‑wise spatio‑temporal autoregressive manner. Each step generates one face over a temporal window, which greatly reduces peak memory and enables native 2K/3K/4K 360° ...

</details>

<details>
<summary><b>9. Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wei Wei, Jiayun Wang, Huancheng Chen, ztwang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04257) • [📄 arXiv](https://arxiv.org/abs/2603.04257) • [📥 PDF](https://arxiv.org/pdf/2603.04257)

> Large language model (LLM) agents are fundamentally bottlenecked by finite context windows on longhorizon tasks. As trajectories grow, retaining tool outputs and intermediate reasoning in-context quickly becomes infeasible: the working context bec...

</details>

<details>
<summary><b>10. V_1: Unifying Generation and Self-Verification for Parallel Reasoners</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04304) • [📄 arXiv](https://arxiv.org/abs/2603.04304) • [📥 PDF](https://arxiv.org/pdf/2603.04304)

**💻 Code:** [⭐ Code](https://github.com/HarmanDotpy/pairwise-self-verification)

> Can LLMs self-verify their own solutions? Modern LLMs are increasingly used as parallel reasoners, where multiple candidate solutions are sampled and then filtered or aggregated. In this setting, the key bottleneck is often not generation but veri...

</details>

<details>
<summary><b>11. AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in Large Vision-Language Models</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Kyeongbo Kong, Sohyeon Kim, Jouwon Song, higokri

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01236) • [📄 arXiv](https://arxiv.org/abs/2603.01236) • [📥 PDF](https://arxiv.org/pdf/2603.01236)

**💻 Code:** [⭐ Code](https://github.com/cvsp-lab/AgilePruner)

> Accept to ICLR 2026

</details>

<details>
<summary><b>12. RIVER: A Real-Time Interaction Benchmark for Video LLMs</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03985) • [📄 arXiv](https://arxiv.org/abs/2603.03985) • [📥 PDF](https://arxiv.org/pdf/2603.03985)

**💻 Code:** [⭐ Code](https://github.com/OpenGVLab/RIVER)

> The rapid advancement of multimodal large language models has demonstrated impressive capabilities, yet nearly all operate in an offline paradigm, hindering real-time interactivity. Addressing this gap, we introduce the Real-tIme Video intERaction...

</details>

<details>
<summary><b>13. InfinityStory: Unlimited Video Generation with World Consistency and Character-Aware Shot Transitions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Subhojyoti Mukherjee, Xiaoqian Shen, Liangbing Zhao, Mohamed Elmoghany, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03646) • [📄 arXiv](https://arxiv.org/abs/2603.03646) • [📥 PDF](https://arxiv.org/pdf/2603.03646)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VideoMemory: Toward Consistent Video Generation via Memory Integration (202...

</details>

<details>
<summary><b>14. SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bing Zhao, Chuan Chen, Hu Wei, Xander Xu, Jialong Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03823) • [📄 arXiv](https://arxiv.org/abs/2603.03823) • [📥 PDF](https://arxiv.org/pdf/2603.03823)

> Proposes SWE-CI, a repository-level benchmark using CI loops to evaluate LLM-powered agents on long-term maintainability across evolving codebases.

</details>

<details>
<summary><b>15. MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yiran Chen, Hai Helen Li, Jingyang Zhang, Zhongxi Wang, linyueqian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02482) • [📄 arXiv](https://arxiv.org/abs/2603.02482) • [📥 PDF](https://arxiv.org/pdf/2603.02482)

> MUSE: Multimodal Unified Safety Evaluation of LLMs

</details>

<details>
<summary><b>16. EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Gim Hee Lee, Yunsong Wang, Zihan Wang, onandon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04254) • [📄 arXiv](https://arxiv.org/abs/2603.04254) • [📥 PDF](https://arxiv.org/pdf/2603.04254)

**💻 Code:** [⭐ Code](https://github.com/0nandon/EmbodiedSplat)

> Accepted to CVPR 2026!

</details>

<details>
<summary><b>17. BeamPERL: Parameter-Efficient RL with Verifiable Rewards Specializes Compact LLMs for Structured Beam Mechanics Reasoning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.04124) • [📄 arXiv](https://arxiv.org/abs/2603.04124) • [📥 PDF](https://arxiv.org/pdf/2603.04124)

**💻 Code:** [⭐ Code](https://github.com/lamm-mit/BeamPERL)

> Does pure RL with verifiable rewards teach models physics, or just clever pattern-matching? We explored this question in our new paper by training a compact 1.5B reasoning model on beam statics (a classic structural engineering problem). We used p...

</details>

<details>
<summary><b>18. MIBURI: Towards Expressive Interactive Gesture Synthesis</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Christian Theobalt, Vera Demberg, Rishabh Dabral, m-hamza-mughal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03282) • [📄 arXiv](https://arxiv.org/abs/2603.03282) • [📥 PDF](https://arxiv.org/pdf/2603.03282)

**💻 Code:** [⭐ Code](https://github.com/m-hamza-mughal/miburi)

> CVPR 2026 Project Page: https://vcai.mpi-inf.mpg.de/projects/MIBURI/

</details>

<details>
<summary><b>19. Specificity-aware reinforcement learning for fine-grained open-world classification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.03197) • [📄 arXiv](https://arxiv.org/abs/2603.03197) • [📥 PDF](https://arxiv.org/pdf/2603.03197)

> Classifying fine-grained visual concepts under open-world settings, i.e., without a predefined label set, demands models to be both accurate and specific. Recent reasoning Large Multimodal Models (LMMs) exhibit strong visual understanding capabili...

</details>

<details>
<summary><b>20. GroupEnsemble: Efficient Uncertainty Estimation for DETR-based Object Detection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.01847) • [📄 arXiv](https://arxiv.org/abs/2603.01847) • [📥 PDF](https://arxiv.org/pdf/2603.01847)

> GroupEnsemble enables efficient uncertainty estimation for DETR-like models by using independent query groups in a single forward pass, addressing limitations of Deep Ensembles and MC Dropout.

</details>

<details>
<summary><b>21. HDINO: A Concise and Efficient Open-Vocabulary Detector</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yong Li, Runze Fan, Qinran Lin, Yiqun Wang, HaoZ416

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.02924) • [📄 arXiv](https://arxiv.org/abs/2603.02924) • [📥 PDF](https://arxiv.org/pdf/2603.02924)

**💻 Code:** [⭐ Code](https://github.com/HaoZ416/HDINO)

> A Concise and Efficient Open-Vocabulary Detector

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2026-03-06.json`](data/daily/2026-03-06.json) | 21 |
| 📆 This Week | [`2026-W09.json`](data/weekly/2026-W09.json) | 153 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 181 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-06 | 21 | [View JSON](data/daily/2026-03-06.json) |
| 📄 2026-03-05 | 41 | [View JSON](data/daily/2026-03-05.json) |
| 📄 2026-03-04 | 41 | [View JSON](data/daily/2026-03-04.json) |
| 📄 2026-03-03 | 22 | [View JSON](data/daily/2026-03-03.json) |
| 📄 2026-03-02 | 28 | [View JSON](data/daily/2026-03-02.json) |
| 📄 2026-03-01 | 28 | [View JSON](data/daily/2026-03-01.json) |
| 📄 2026-02-28 | 28 | [View JSON](data/daily/2026-02-28.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W09 | 153 | [View JSON](data/weekly/2026-W09.json) |
| 📅 2026-W08 | 184 | [View JSON](data/weekly/2026-W08.json) |
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 181 | [View JSON](data/monthly/2026-03.json) |
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
