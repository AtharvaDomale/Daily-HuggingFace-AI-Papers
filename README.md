<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-20-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-869+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">20</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">90</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">131</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">869+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 09, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02151) • [📄 arXiv](https://arxiv.org/abs/2601.02151) • [📥 PDF](https://arxiv.org/pdf/2601.02151)

**💻 Code:** [⭐ Code](https://github.com/hiyouga/LLaMA-Factory) • [⭐ Code](https://github.com/PRIS-CV/EAFT)

> 💻 Code: https://github.com/PRIS-CV/EAFT ✨ Project Page: https://ymxyll.github.io/EAFT/

</details>

<details>
<summary><b>2. Evolving Programmatic Skill Networks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03509) • [📄 arXiv](https://arxiv.org/abs/2601.03509) • [📥 PDF](https://arxiv.org/pdf/2601.03509)

> We study continual skill acquisition in openended embodied environments where an agent must construct, refine, and reuse an expanding library of executable skills. We introduce the Programmatic Skill Network (PSN), a framework in which skills are ...

</details>

<details>
<summary><b>3. Atlas: Orchestrating Heterogeneous Models and Tools for Multi-Domain Complex Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuhao Shen, Jiahao Yuan, Ruihan Jin, Guocheng Zhai, Jinyang23

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03872) • [📄 arXiv](https://arxiv.org/abs/2601.03872) • [📥 PDF](https://arxiv.org/pdf/2601.03872)

> 🚀 [New Paper] Atlas: Orchestrating Heterogeneous Models and Tools for Multi-Domain Complex Reasoning The growing diversity of LLMs and external tools presents a significant challenge: how to select the optimal model-tool combination for complex re...

</details>

<details>
<summary><b>4. Benchmark^2: Systematic Evaluation of LLM Benchmarks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Muling Wu, Changze Lv, Jingwen Xu, Qi Qian, ChengsongHuang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03986) • [📄 arXiv](https://arxiv.org/abs/2601.03986) • [📥 PDF](https://arxiv.org/pdf/2601.03986)

> The rapid proliferation of benchmarks for evaluating large language models (LLMs) has created an urgent need for systematic methods to assess benchmark quality itself. We propose Benchmark^2, a comprehensive framework comprising three complementar...

</details>

<details>
<summary><b>5. ROI-Reasoning: Rational Optimization for Inference via Pre-Computation Meta-Cognition</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03822) • [📄 arXiv](https://arxiv.org/abs/2601.03822) • [📥 PDF](https://arxiv.org/pdf/2601.03822)

> ROI-Reasoning introduces a principled framework for budget-aware inference-time reasoning in large language models. Instead of blindly scaling computation, the authors formulate multi-task reasoning under a global token constraint as an Ordered St...

</details>

<details>
<summary><b>6. Klear: Unified Multi-Task Audio-Video Joint Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04151) • [📄 arXiv](https://arxiv.org/abs/2601.04151) • [📥 PDF](https://arxiv.org/pdf/2601.04151)

> Klear: 26B model for joint audio-video generation Single-tower DiT with "Omni-Full Attention" across video, audio, and text Progressive multi-task training (T2V, T2A, T2AV, I2V all in one model) 81M sample dataset with dense captions Claims Veo 3-...

</details>

<details>
<summary><b>7. Choreographing a World of Dynamic Objects</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hadi Alzayer, Yunzhi Zhang, Karthik Dharmarajan, Chen Geng, Yanzhe Lyu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04194) • [📄 arXiv](https://arxiv.org/abs/2601.04194) • [📥 PDF](https://arxiv.org/pdf/2601.04194)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Animus3D: Text-driven 3D Animation via Motion Score Distillation (2025) Ani...

</details>

<details>
<summary><b>8. Agentic Rubrics as Contextual Verifiers for SWE Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04171) • [📄 arXiv](https://arxiv.org/abs/2601.04171) • [📥 PDF](https://arxiv.org/pdf/2601.04171)

> Agentic Rubrics for verifying SWE agent patches WITHOUT running tests! An agent explores the codebase to generate context-grounded checklists, then scores patches execution-free. Rubrics provide dense, interpretable reward signals that could scale...

</details>

<details>
<summary><b>9. MDAgent2: Large Language Model for Code Generation and Knowledge Q&A in Molecular Dynamics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02075) • [📄 arXiv](https://arxiv.org/abs/2601.02075) • [📥 PDF](https://arxiv.org/pdf/2601.02075)

**💻 Code:** [⭐ Code](https://github.com/FredericVAN/PKU_MDAgent2)

> project: https://github.com/FredericVAN/PKU_MDAgent2

</details>

<details>
<summary><b>10. E-GRPO: High Entropy Steps Drive Effective Reinforcement Learning for Flow Models</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00423) • [📄 arXiv](https://arxiv.org/abs/2601.00423) • [📥 PDF](https://arxiv.org/pdf/2601.00423)

**💻 Code:** [⭐ Code](https://github.com/shengjun-zhang/VisualGRPO)

> We propose an entropy aware Group Relative Policy Optimization (E-GRPO) to increase the entropy of SDE sampling steps. We have integrated a variety of current GRPO-based reinforcement learning methods as well as different image reward models. Code...

</details>

<details>
<summary><b>11. EpiQAL: Benchmarking Large Language Models in Epidemiological Question Answering for Enhanced Alignment and Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guanchen Wu, Yuzhang Xie, Zewen Liu, Dehai Min, Mingyang Wei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03471) • [📄 arXiv](https://arxiv.org/abs/2601.03471) • [📥 PDF](https://arxiv.org/pdf/2601.03471)

> EpiQAL, the first diagnostic benchmark for epidemiological question answering across diverse diseases, comprising three subsets built from open-access literature.

</details>

<details>
<summary><b>12. RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03699) • [📄 arXiv](https://arxiv.org/abs/2601.03699) • [📥 PDF](https://arxiv.org/pdf/2601.03699)

**💻 Code:** [⭐ Code](https://github.com/knoveleng/redeval)

> RedBench presents a unified dataset with standardized risk categorization for evaluating LLM vulnerabilities across multiple domains and attack types.

</details>

<details>
<summary><b>13. Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03315) • [📄 arXiv](https://arxiv.org/abs/2601.03315) • [📥 PDF](https://arxiv.org/pdf/2601.03315)

> We find that LLMs aren't scientists yet.

</details>

<details>
<summary><b>14. ThinkRL-Edit: Thinking in Reinforcement Learning for Reasoning-Centric Image Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03467) • [📄 arXiv](https://arxiv.org/abs/2601.03467) • [📥 PDF](https://arxiv.org/pdf/2601.03467)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API PaCo-RL: Advancing Reinforcement Learning for Consistent Image Generation w...

</details>

<details>
<summary><b>15. Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03448) • [📄 arXiv](https://arxiv.org/abs/2601.03448) • [📥 PDF](https://arxiv.org/pdf/2601.03448)

**💻 Code:** [⭐ Code](https://github.com/gucci-j/l2t)

> We propose L2T, a pre-training framework integrating Language Learning Tasks alongside standard next-token prediction. L2T establishes the structural scaffolding required for linguistic competence, complementing world knowledge acquired through st...

</details>

<details>
<summary><b>16. Pearmut: Human Evaluation of Translation Made Trivial</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Tom Kocmi, Vilém Zouhar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02933) • [📄 arXiv](https://arxiv.org/abs/2601.02933) • [📥 PDF](https://arxiv.org/pdf/2601.02933)

**💻 Code:** [⭐ Code](https://github.com/zouharvi/pearmut)

> Happy to discuss how people human-evaluate multilingual tasks! 🙂

</details>

<details>
<summary><b>17. ResTok: Learning Hierarchical Residuals in 1D Visual Tokenizers for Autoregressive Image Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ming Lu, Kun Gai, Huan Yang, Cheng Da, Xu Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03955) • [📄 arXiv](https://arxiv.org/abs/2601.03955) • [📥 PDF](https://arxiv.org/pdf/2601.03955)

> No abstract available.

</details>

<details>
<summary><b>18. MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bingzhe Li, Guanpeng Li, Yi Li, Dongming Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03236) • [📄 arXiv](https://arxiv.org/abs/2601.03236) • [📥 PDF](https://arxiv.org/pdf/2601.03236)

> This ia giid paper

</details>

<details>
<summary><b>19. Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction</b> ⭐ 34</summary>

<br/>

**👥 Authors:** Yuewen Ma, Lin Ma, Bangbang Yang, Yuanbo Yang, Jiaxin Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04090) • [📄 arXiv](https://arxiv.org/abs/2601.04090) • [📥 PDF](https://arxiv.org/pdf/2601.04090)

**💻 Code:** [⭐ Code](https://github.com/JaceyHuang/Gen3R)

> We Introduce Gen3R — create multi-quantity geometry with RGB from images. 📷 Photorealistic Video 🚀 Accurate 3D Scene Geometry Arxiv: https://arxiv.org/abs/2601.04090 Project page: https://xdimlab.github.io/Gen3R/

</details>

<details>
<summary><b>20. RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00705) • [📄 arXiv](https://arxiv.org/abs/2601.00705) • [📥 PDF](https://arxiv.org/pdf/2601.00705)

> We introduce RGS-SLAM, a robust Gaussian-splatting SLAM framework that replaces the residual-driven densification stage of GS-SLAM with a training-free correspondence-to-Gaussian initialization. Instead of progressively adding Gaussians as residua...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 20 |
| 📅 Today | [`2026-01-09.json`](data/daily/2026-01-09.json) | 20 |
| 📆 This Week | [`2026-W01.json`](data/weekly/2026-W01.json) | 90 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 131 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-09 | 20 | [View JSON](data/daily/2026-01-09.json) |
| 📄 2026-01-08 | 26 | [View JSON](data/daily/2026-01-08.json) |
| 📄 2026-01-07 | 24 | [View JSON](data/daily/2026-01-07.json) |
| 📄 2026-01-06 | 13 | [View JSON](data/daily/2026-01-06.json) |
| 📄 2026-01-05 | 7 | [View JSON](data/daily/2026-01-05.json) |
| 📄 2026-01-04 | 7 | [View JSON](data/daily/2026-01-04.json) |
| 📄 2026-01-03 | 7 | [View JSON](data/daily/2026-01-03.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W01 | 90 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 131 | [View JSON](data/monthly/2026-01.json) |
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
