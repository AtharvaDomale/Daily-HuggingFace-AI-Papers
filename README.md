<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-15-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4225+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">15</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">31</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">604</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4225+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 26, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. DVAO: Dynamic Variance-adaptive Advantage Optimization for Multi-reward Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guohua Liu, Chuzhan Hao, Guofeng Quan, Jingyi Song, Guochao Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25604) • [📄 arXiv](https://arxiv.org/abs/2605.25604) • [📥 PDF](https://arxiv.org/pdf/2605.25604)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We propose Dynamic Variance-adaptive Advantage Optimization (DVAO), which dynamically adjusts combination weights based on the empirical reward variance of each objective within a rollout group, effectively up-weighting objectives with a stronger ...

</details>

<details>
<summary><b>2. WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25874) • [📄 arXiv](https://arxiv.org/abs/2605.25874) • [📥 PDF](https://arxiv.org/pdf/2605.25874)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/meituan-longcat/WBench)

> upload

</details>

<details>
<summary><b>3. Macaron-A2UI: A Model for Generative UI in Personal Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.24830) • [📄 arXiv](https://arxiv.org/abs/2605.24830) • [📥 PDF](https://arxiv.org/pdf/2605.24830)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Macaron-A2UI: A Model for Generative UI in Personal Agents

</details>

<details>
<summary><b>4. AutoResearch AI: Towards AI-Powered Research Automation for Scientific Discovery</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ziji Sheng, Yixiao Huang, Dingjie Song, Jiawen Shi, Guiyao Tie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23204) • [📄 arXiv](https://arxiv.org/abs/2605.23204) • [📥 PDF](https://arxiv.org/pdf/2605.23204)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Mr-Tieguigui/Autoresearch)

> No abstract available.

</details>

<details>
<summary><b>5. Your Embedding Model is SMARTer Than You Think</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Donghyun Kim, Tae-Eui Kam, Sukanta Ganguly, Hyun Jung Lee, Jianrui Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.24938) • [📄 arXiv](https://arxiv.org/abs/2605.24938) • [📥 PDF](https://arxiv.org/pdf/2605.24938)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HanSolo9682/SMART)

> Our work introduces SMART, a framework that unlocks the latent multi-vector capabilities of standard single-vector models for multimodal retrieval. While single-vector retrievers are highly efficient, they often discard the fine-grained, local evi...

</details>

<details>
<summary><b>6. ParaVT: Taming the Tool Prior Paradox for Parallel Tool Use in Agentic Video Reinforcement Learning</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20342) • [📄 arXiv](https://arxiv.org/abs/2605.20342) • [📥 PDF](https://arxiv.org/pdf/2605.20342)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/EvolvingLMMs-Lab/ParaVT)

> Long-video understanding is becoming agentic where LMMs are post-trained with RL to natively invoke video tools (e.g., temporal cropping). But every existing native-RL recipe (including our own LongVT @ CVPR 2026) dispatches tool calls sequentiall...

</details>

<details>
<summary><b>7. Pantheon360: Taming Digital Twin Generation via 3D-Aware 360° Video Diffusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25449) • [📄 arXiv](https://arxiv.org/abs/2605.25449) • [📥 PDF](https://arxiv.org/pdf/2605.25449)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Generating complete digital twins from videos requires precise camera control, global scene coverage, and strict spatial–temporal consistency—constraints that remain challenging for perspective video generators due to their limited field of view (...

</details>

<details>
<summary><b>8. TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26115) • [📄 arXiv](https://arxiv.org/abs/2605.26115) • [📥 PDF](https://arxiv.org/pdf/2605.26115)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ziplab/TriSplat)

> No abstract available.

</details>

<details>
<summary><b>9. Claw-Anything: Benchmarking Always-On Personal Assistants with Broader Access to User's Digital World</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Siqi Cheng, Qipeng Gu, Haiyang Wang, Xinyuan Liang, Yusong Lin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26086) • [📄 arXiv](https://arxiv.org/abs/2605.26086) • [📥 PDF](https://arxiv.org/pdf/2605.26086)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LiberCoders/CLaw-Anything)

> Claw-Anything: See anything, and do anything.  Scaling Agent Context. We believe the next leap for always-on LLM agents lies in scaling agent context — expanding the slice of the user's digital world an assistant can continuously perceive, reason ...

</details>

<details>
<summary><b>10. Geometry-Aware Image Flow Matching</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Joonseok Lee, Kwanseok Kim, Junho Lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25294) • [📄 arXiv](https://arxiv.org/abs/2605.25294) • [📥 PDF](https://arxiv.org/pdf/2605.25294)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: Natural images live on a hypersphere — and treating them that way improves flow matching. Geometry-aware generative modeling has worked well on known manifolds (molecules, crystals, proteins), but natural images have stayed stuck in Euclide...

</details>

<details>
<summary><b>11. MetaphorVU: Towards Metaphorical Video Understanding</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Ruotong Pan, Fangrui Lv, Guiping Jiang, Boxi Cao, Zhuoqun Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25461) • [📄 arXiv](https://arxiv.org/abs/2605.25461) • [📥 PDF](https://arxiv.org/pdf/2605.25461)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/icip-cas/MetaphorVU)

> • We propose MetaphorVU-Bench, which is the first benchmark dedicated to systematic and comprehensive evaluation for metaphorical video understanding. • We conduct extensive experiments and analysis, revealing the deficiencies of current MLLMs an...

</details>

<details>
<summary><b>12. Toward Native Multimodal Modeling: A Roadmap</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25343) • [📄 arXiv](https://arxiv.org/abs/2605.25343) • [📥 PDF](https://arxiv.org/pdf/2605.25343)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NMM-Roadmap/Awesome-NMM-List)

> The community is undergoing a macro-level paradigm shift from early modular assembly, i.e., late-fusion and grafted pipelines blind to raw sensory signals, toward born-native multimodal convergence, where multimodal understanding and generation fl...

</details>

<details>
<summary><b>13. Channel-wise Vector Quantization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zuxuan Wu, Tong Zhang, Yitong Chen, Tianhang Wang, Wei Song

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26089) • [📄 arXiv](https://arxiv.org/abs/2605.26089) • [📥 PDF](https://arxiv.org/pdf/2605.26089)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/songweii/CVQ)

> We present Channel-wise Vector Quantization (CVQ), a novel image tokenization paradigm that replaces patch-wise tokens with channel-wise tokens. Unlike conventional vector quantization, which assigns a discrete token to each patch feature vector, ...

</details>

<details>
<summary><b>14. SemBridge: Language Transfer in Sparse Encoders via Multilingual Semantic Bridges</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26002) • [📄 arXiv](https://arxiv.org/abs/2605.26002) • [📥 PDF](https://arxiv.org/pdf/2605.26002)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. ClaimDiff-RL: Fine-Grained Caption Reinforcement Learning through Visual Claim Comparison</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Shaoxiang Chen, Rongxin Guo, Yan Ma, Xuyang Shen, Tianle Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20278) • [📄 arXiv](https://arxiv.org/abs/2605.20278) • [📥 PDF](https://arxiv.org/pdf/2605.20278)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ltl3A87/ClaimDiff-RL)

> Can we make caption RL more diagnosable than a single scalar reward? In ClaimDiff-RL, we argue that long-form image captions should not be judged only as whole sequences. A dense caption is made of many local visual claims—objects, counts, colors,...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 15 |
| 📅 Today | [`2026-05-26.json`](data/daily/2026-05-26.json) | 15 |
| 📆 This Week | [`2026-W21.json`](data/weekly/2026-W21.json) | 31 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 604 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-26 | 15 | [View JSON](data/daily/2026-05-26.json) |
| 📄 2026-05-25 | 16 | [View JSON](data/daily/2026-05-25.json) |
| 📄 2026-05-24 | 49 | [View JSON](data/daily/2026-05-24.json) |
| 📄 2026-05-23 | 49 | [View JSON](data/daily/2026-05-23.json) |
| 📄 2026-05-22 | 18 | [View JSON](data/daily/2026-05-22.json) |
| 📄 2026-05-21 | 18 | [View JSON](data/daily/2026-05-21.json) |
| 📄 2026-05-20 | 16 | [View JSON](data/daily/2026-05-20.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W21 | 31 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 604 | [View JSON](data/monthly/2026-05.json) |
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
