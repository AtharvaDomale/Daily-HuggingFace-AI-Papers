<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-15-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4910+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">33</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">507</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4910+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 24, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24530) • [📄 arXiv](https://arxiv.org/abs/2606.24530) • [📥 PDF](https://arxiv.org/pdf/2606.24530)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/FrontisAI/NatureBench)

> NatureBench is a cross-discipline benchmark of 90 tasks distilled from peer-reviewed Nature-family publications, spanning 6 scientific domains, designed to evaluate whether AI coding agents can move beyond reproduction toward discovery. Each task ...

</details>

<details>
<summary><b>2. Qwen-AgentWorld: Language World Models for General Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24597) • [📄 arXiv](https://arxiv.org/abs/2606.24597) • [📥 PDF](https://arxiv.org/pdf/2606.24597)

**💻 Code:** [⭐ Code](https://github.com/QwenLM/Qwen-AgentWorld) • [⭐ Code](https://github.com/huggingface)

> https://github.com/QwenLM/Qwen-AgentWorld

</details>

<details>
<summary><b>3. MobileForge: Annotation-Free Adaptation for Mobile GUI Agents with Hierarchical Feedback-Guided Policy Optimization</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19930) • [📄 arXiv](https://arxiv.org/abs/2606.19930) • [📥 PDF](https://arxiv.org/pdf/2606.19930)

**💻 Code:** [⭐ Code](https://github.com/kwai/MobileForge) • [⭐ Code](https://github.com/huggingface)

> Code, data, and trained models will be released at https://mobile-forge.github.io .

</details>

<details>
<summary><b>4. MemGUI-Agent: An End-to-End Long-Horizon Mobile GUI Agent with Proactive Context Management</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.19926) • [📄 arXiv](https://arxiv.org/abs/2606.19926) • [📥 PDF](https://arxiv.org/pdf/2606.19926)

**💻 Code:** [⭐ Code](https://github.com/kwai/MemGUI-Agent) • [⭐ Code](https://github.com/huggingface)

> Code, data, and trained models will be released at https://memgui-agent.github.io .

</details>

<details>
<summary><b>5. FedOT: Ownership Verification and Leakage Tracing via Watermarks for Federated LDMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiaxu Miao, Yuan Gan, Wenlong Cheng, Yunqiu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.22875) • [📄 arXiv](https://arxiv.org/abs/2606.22875) • [📥 PDF](https://arxiv.org/pdf/2606.22875)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project page: https://spyzixuan.github.io/FedOT/

</details>

<details>
<summary><b>6. OpenThoughts-Agent: Data Recipes for Agentic Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Atula Tejaswi, Etash Guha, Marianna Nezhurina, Richard Zhuang, Negin Raoof

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24855) • [📄 arXiv](https://arxiv.org/abs/2606.24855) • [📥 PDF](https://arxiv.org/pdf/2606.24855)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>7. Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jing Xu, Rongpeng Zhu, Xingyu Han, Yuandong Pu, Jiayi Lei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24548) • [📄 arXiv](https://arxiv.org/abs/2606.24548) • [📥 PDF](https://arxiv.org/pdf/2606.24548)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://arxiv.org/pdf/2606.24548

</details>

<details>
<summary><b>8. DiffusionBench: On Holistic Evaluation of Diffusion Transformers</b> ⭐ 36</summary>

<br/>

**👥 Authors:** Martin Bell, Ethan Smith, Zhanhao Liang, Jaskirat Singh, Xingjian Leng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24888) • [📄 arXiv](https://arxiv.org/abs/2606.24888) • [📥 PDF](https://arxiv.org/pdf/2606.24888)

**💻 Code:** [⭐ Code](https://github.com/End2End-Diffusion/diffusion-bench) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yifeng Pan, Wenhua Han, Xiaoqing Ye, Zhe Liu, Xirui Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24231) • [📄 arXiv](https://arxiv.org/abs/2606.24231) • [📥 PDF](https://arxiv.org/pdf/2606.24231)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/lixirui142/FlowR2A)

> FlowR2A is a multimodal driving planner that learns the reward-conditioned action distribution p(a | r) with flow matching. Instead of treating simulation rewards as discriminative targets, FlowR2A treats them as a condition, unifying the dense su...

</details>

<details>
<summary><b>10. DREAM: Dense Retrieval Embeddings via Autoregressive Modeling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yi Yang, Yixuan Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24667) • [📄 arXiv](https://arxiv.org/abs/2606.24667) • [📥 PDF](https://arxiv.org/pdf/2606.24667)

**💻 Code:** [⭐ Code](https://github.com/yixuantt/DREAM) • [⭐ Code](https://github.com/huggingface)

> train dense retrievers using the LLM’s next-token prediction loss instead of relying only on contrastive labels. DREAM treats relevance as whether a document helps a frozen LLM predict the target output, then backpropagates that signal to the retr...

</details>

<details>
<summary><b>11. ReMMD: Realistic Multilingual Multi-Image Agentic Verification for Multimodal Misinformation Detection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24112) • [📄 arXiv](https://arxiv.org/abs/2606.24112) • [📥 PDF](https://arxiv.org/pdf/2606.24112)

**💻 Code:** [⭐ Code](https://github.com/DANG-ai/ReMMDBench-Agent) • [⭐ Code](https://github.com/huggingface)

> We introduce ReMMD, a realistic multilingual multi-image agentic verification framework for multimodal misinformation detection. ReMMDBench contains 500 real-world samples with 2,756 images, five monolingual languages, two cross-lingual settings, ...

</details>

<details>
<summary><b>12. Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning</b> ⭐ 199</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24133) • [📄 arXiv](https://arxiv.org/abs/2606.24133) • [📥 PDF](https://arxiv.org/pdf/2606.24133)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DANG-ai/LLM-Training-Holistic-Data-Schedule)

> KDD 2026 cycle 1 paper. This paper introduces Holistic Data Scheduler (HDS), an online data mixing framework for LLM pre-training that formulates dynamic data scheduling as a multi-objective reinforcement learning problem. Instead of relying on a ...

</details>

<details>
<summary><b>13. World Value Models for Robotic Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24742) • [📄 arXiv](https://arxiv.org/abs/2606.24742) • [📥 PDF](https://arxiv.org/pdf/2606.24742)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. FLUX3D: High-Fidelity 3D Gaussian Generation with Diffusion-Aligned Sparse Representation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24874) • [📄 arXiv](https://arxiv.org/abs/2606.24874) • [📥 PDF](https://arxiv.org/pdf/2606.24874)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. An Efficient Method for the Optimal Control of Microgrids Under Uncertainties using Local Reduction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.12345) • [📄 arXiv](https://arxiv.org/abs/2606.12345) • [📥 PDF](https://arxiv.org/pdf/2606.12345)

**💻 Code:** [⭐ Code](https://github.com/shidingz/EDV) • [⭐ Code](https://github.com/huggingface)

> Experience-driven self-evolution is essential for large language model (LLM) agents to improve through interaction with open-world environments. However, existing experience learning methods largely rely on single-agent loops, in which the same ag...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 15 |
| 📅 Today | [`2026-06-24.json`](data/daily/2026-06-24.json) | 15 |
| 📆 This Week | [`2026-W25.json`](data/weekly/2026-W25.json) | 33 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 507 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-24 | 15 | [View JSON](data/daily/2026-06-24.json) |
| 📄 2026-06-23 | 14 | [View JSON](data/daily/2026-06-23.json) |
| 📄 2026-06-22 | 4 | [View JSON](data/daily/2026-06-22.json) |
| 📄 2026-06-21 | 34 | [View JSON](data/daily/2026-06-21.json) |
| 📄 2026-06-20 | 34 | [View JSON](data/daily/2026-06-20.json) |
| 📄 2026-06-19 | 16 | [View JSON](data/daily/2026-06-19.json) |
| 📄 2026-06-18 | 11 | [View JSON](data/daily/2026-06-18.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W25 | 33 | [View JSON](data/weekly/2026-W25.json) |
| 📅 2026-W24 | 130 | [View JSON](data/weekly/2026-W24.json) |
| 📅 2026-W23 | 166 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 507 | [View JSON](data/monthly/2026-06.json) |
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
