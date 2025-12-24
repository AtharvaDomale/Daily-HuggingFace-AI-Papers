<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-23-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-637+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">83</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">686</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">637+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 24, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16676) • [📄 arXiv](https://arxiv.org/abs/2512.16676) • [📥 PDF](https://arxiv.org/pdf/2512.16676)

**💻 Code:** [⭐ Code](https://github.com/OpenDCAI/DataFlow)

> code link: https://github.com/OpenDCAI/DataFlow

</details>

<details>
<summary><b>2. The Prism Hypothesis: Harmonizing Semantic and Pixel Representations via Unified Autoencoding</b> ⭐ 56</summary>

<br/>

**👥 Authors:** Ziwei Liu, Dahua Lin, Quan Wang, Haiwen Diao, Weichen Fan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19693) • [📄 arXiv](https://arxiv.org/abs/2512.19693) • [📥 PDF](https://arxiv.org/pdf/2512.19693)

**💻 Code:** [⭐ Code](https://github.com/WeichenFan/UAE)

> Deep representations across modalities are inherently intertwined. In this paper, we systematically analyze the spectral characteristics of various semantic and pixel encoders. Interestingly, our study uncovers a highly inspiring and rarely explor...

</details>

<details>
<summary><b>3. Region-Constraint In-Context Generation for Instructional Video Editing</b> ⭐ 32</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17650) • [📄 arXiv](https://arxiv.org/abs/2512.17650) • [📥 PDF](https://arxiv.org/pdf/2512.17650)

**💻 Code:** [⭐ Code](https://github.com/HiDream-ai/ReCo)

> Region-Constraint In-Context Generation for Instructional Video Editing Paper: https://arxiv.org/abs/2512.17650 Project Page: https://zhw-zhang.github.io/ReCo-page/ Github: https://github.com/HiDream-ai/ReCo ReCo-Data: https://huggingface.co/datas...

</details>

<details>
<summary><b>4. Infinite-Homography as Robust Conditioning for Camera-Controlled Video Generation</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17040) • [📄 arXiv](https://arxiv.org/abs/2512.17040) • [📥 PDF](https://arxiv.org/pdf/2512.17040)

**💻 Code:** [⭐ Code](https://github.com/emjay73/InfCam)

> No abstract available.

</details>

<details>
<summary><b>5. QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Lu Cheng, Tongtong Wu, Kailin Zhang, Dehai Min

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19134) • [📄 arXiv](https://arxiv.org/abs/2512.19134) • [📥 PDF](https://arxiv.org/pdf/2512.19134)

**💻 Code:** [⭐ Code](https://github.com/ZhishanQ/QuCo-RAG)

> A new framework for dynamic retrieval-augmented generation.

</details>

<details>
<summary><b>6. Can LLMs Estimate Student Struggles? Human-AI Difficulty Alignment with Proficiency Simulation for Item Difficulty Prediction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hong Jiao, Jian Chen, Yunze Xiao, Han Chen, Ming Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18880) • [📄 arXiv](https://arxiv.org/abs/2512.18880) • [📥 PDF](https://arxiv.org/pdf/2512.18880)

**💻 Code:** [⭐ Code](https://github.com/MingLiiii/Difficulty_Alignment)

> Key Findings of our Human-LLM difficulty alignment study: Systematic Misalignment : Contrary to standard capability metrics, scaling does not reliably translate into alignment. Increasing model scale does not improve difficulty predictions; instea...

</details>

<details>
<summary><b>7. WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19678) • [📄 arXiv](https://arxiv.org/abs/2512.19678) • [📥 PDF](https://arxiv.org/pdf/2512.19678)

**💻 Code:** [⭐ Code](https://github.com/HyoKong/WorldWarp)

> Long-range camera-conditioned scene generation from a single image. Project page and code: https://hyokong.github.io/worldwarp-page/ .

</details>

<details>
<summary><b>8. LoGoPlanner: Localization Grounded Navigation Policy with Metric-aware Visual Geometry</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuan Shen, Tai Wang, Yuqiang Yang, Wenzhe Cai, Jiaqi Peng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19629) • [📄 arXiv](https://arxiv.org/abs/2512.19629) • [📥 PDF](https://arxiv.org/pdf/2512.19629)

> LoGoPlanner: Localization Grounded Navigation Policy with Metric-aware Visual Geometry

</details>

<details>
<summary><b>9. UCoder: Unsupervised Code Generation by Internal Probing of Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuqing Ma, Lin Jing, Wei Zhang, Jian Yang, Jiajun Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17385) • [📄 arXiv](https://arxiv.org/abs/2512.17385) • [📥 PDF](https://arxiv.org/pdf/2512.17385)

> This paper introduces UCoder, an unsupervised framework for training code-generating large language models without requiring any external datasets, including unlabeled code snippets. The approach, called IPC (Internal Probing of LLMs for Code gene...

</details>

<details>
<summary><b>10. GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19682) • [📄 arXiv](https://arxiv.org/abs/2512.19682) • [📥 PDF](https://arxiv.org/pdf/2512.19682)

**💻 Code:** [⭐ Code](https://github.com/Gen-Verse/GenEnv)

> Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutiona...

</details>

<details>
<summary><b>11. StoryMem: Multi-shot Long Video Storytelling with Memory</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19539) • [📄 arXiv](https://arxiv.org/abs/2512.19539) • [📥 PDF](https://arxiv.org/pdf/2512.19539)

**💻 Code:** [⭐ Code](https://github.com/Kevin-thu/StoryMem)

> Visual storytelling requires generating multi-shot videos with cinematic quality and long-range consistency. Inspired by human memory, we propose StoryMem, a paradigm that reformulates long-form video storytelling as iterative shot synthesis condi...

</details>

<details>
<summary><b>12. LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16229) • [📄 arXiv](https://arxiv.org/abs/2512.16229) • [📥 PDF](https://arxiv.org/pdf/2512.16229)

**💻 Code:** [⭐ Code](https://github.com/zhijie-group/LoPA)

> 🔗Paper： https://arxiv.org/abs/2512.16229 🔗GitHub： https://github.com/zhijie-group/LoPA 🔗blog: https://zhijie-group.github.io/blogs/lopa

</details>

<details>
<summary><b>13. Reasoning Palette: Modulating Reasoning via Latent Contextualization for Controllable Exploration for (V)LMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17206) • [📄 arXiv](https://arxiv.org/abs/2512.17206) • [📥 PDF](https://arxiv.org/pdf/2512.17206)

> Reasoning Palette addresses the challenge of controlling LLM generation style and enabling effective exploration in RL by introducing a stochastic latent variable that encodes diverse reasoning strategies. This latent, inferred via a VAE from ques...

</details>

<details>
<summary><b>14. MobileWorld: Benchmarking Autonomous Mobile Agents in Agent-User Interactive, and MCP-Augmented Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19432) • [📄 arXiv](https://arxiv.org/abs/2512.19432) • [📥 PDF](https://arxiv.org/pdf/2512.19432)

> Among existing online mobile-use benchmarks, AndroidWorld has emerged as the dominant benchmark due to its reproducible environment and deterministic evaluation; however, recent agents achieving over 90% success rates indicate its saturation and m...

</details>

<details>
<summary><b>15. Does It Tie Out? Towards Autonomous Legal Agents in Venture Capital</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18658) • [📄 arXiv](https://arxiv.org/abs/2512.18658) • [📥 PDF](https://arxiv.org/pdf/2512.18658)

> Most LLMs today are powerful at language but weak at worlds: they generate fluent outputs without maintaining a consistent, verifiable model of reality. As a result, many AI applications plateau at demos or copilots and fail in complex, high-stake...

</details>

<details>
<summary><b>16. Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Liliang Chen, Shengcong Chen, Di Chen, Hongwei Fan, Yujie Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19402) • [📄 arXiv](https://arxiv.org/abs/2512.19402) • [📥 PDF](https://arxiv.org/pdf/2512.19402)

> Paper: https://arxiv.org/abs/2512.19402 Project Page: https://real2edit2real.github.io/

</details>

<details>
<summary><b>17. CASA: Cross-Attention via Self-Attention for Efficient Vision-Language Fusion</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19535) • [📄 arXiv](https://arxiv.org/abs/2512.19535) • [📥 PDF](https://arxiv.org/pdf/2512.19535)

**💻 Code:** [⭐ Code](https://github.com/kyutai-labs/casa)

> Code: https://github.com/kyutai-labs/casa

</details>

<details>
<summary><b>18. Name That Part: 3D Part Segmentation and Naming</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alan Yuille, Anand Bhattad, Ankit Vaidya, Prakhar Kaushik, Soumava Paul

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18003) • [📄 arXiv](https://arxiv.org/abs/2512.18003) • [📥 PDF](https://arxiv.org/pdf/2512.18003)

> We address semantic 3D part segmentation: decomposing objects into parts with meaningful names. While datasets exist with part annotations, their definitions are inconsistent across datasets, limiting robust training. Previous methods produce unla...

</details>

<details>
<summary><b>19. MatSpray: Fusing 2D Material World Knowledge on 3D Geometry</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18314) • [📄 arXiv](https://arxiv.org/abs/2512.18314) • [📥 PDF](https://arxiv.org/pdf/2512.18314)

**💻 Code:** [⭐ Code](https://github.com/cgtuebingen/MatSpray)

> 🌐 https://matspray.jdihlmann.com/ 📃 https://arxiv.org/abs/2512.18314 💾 https://github.com/cgtuebingen/MatSpray

</details>

<details>
<summary><b>20. Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sujata Ghosh, Saptarshi Sahoo, Aheli Poddar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12620) • [📄 arXiv](https://arxiv.org/abs/2512.12620) • [📥 PDF](https://arxiv.org/pdf/2512.12620)

**💻 Code:** [⭐ Code](https://github.com/XAheli/Logic-in-LLMs)

> arXiv lens breakdown of this paper 👉 https://arxivlens.com/PaperView/Details/understanding-syllogistic-reasoning-in-llms-from-formal-and-natural-language-perspectives-822-84433a31 Executive Summary Detailed Breakdown Practical Applications

</details>

<details>
<summary><b>21. Over++: Generative Video Compositing for Layer Interaction Effects</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Roni Sengupta, Cary Phillips, Jun Myeong Choi, Jiaye Wu, Luchao Qi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19661) • [📄 arXiv](https://arxiv.org/abs/2512.19661) • [📥 PDF](https://arxiv.org/pdf/2512.19661)

> No abstract available.

</details>

<details>
<summary><b>22. Brain-Grounded Axes for Reading and Steering LLM States</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sandro Andric

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19399) • [📄 arXiv](https://arxiv.org/abs/2512.19399) • [📥 PDF](https://arxiv.org/pdf/2512.19399)

**💻 Code:** [⭐ Code](https://github.com/sandroandric/Brain-Grounded-Axes-for-Reading-and-Steering-LLM-States)

> These research supports a new interface: neurophysiology-grounded axes provide interpretable and controllable handles for LLM behavior.

</details>

<details>
<summary><b>23. SecureCode v2.0: A Production-Grade Dataset for Training Security-Aware Code Generation Models</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Scott Thornton

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18542) • [📄 arXiv](https://arxiv.org/abs/2512.18542) • [📥 PDF](https://arxiv.org/pdf/2512.18542)

**💻 Code:** [⭐ Code](https://github.com/scthornton/securecode-v2)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 23 |
| 📅 Today | [`2025-12-24.json`](data/daily/2025-12-24.json) | 23 |
| 📆 This Week | [`2025-W51.json`](data/weekly/2025-W51.json) | 83 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 686 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-24 | 23 | [View JSON](data/daily/2025-12-24.json) |
| 📄 2025-12-23 | 22 | [View JSON](data/daily/2025-12-23.json) |
| 📄 2025-12-22 | 38 | [View JSON](data/daily/2025-12-22.json) |
| 📄 2025-12-21 | 38 | [View JSON](data/daily/2025-12-21.json) |
| 📄 2025-12-20 | 37 | [View JSON](data/daily/2025-12-20.json) |
| 📄 2025-12-19 | 30 | [View JSON](data/daily/2025-12-19.json) |
| 📄 2025-12-18 | 38 | [View JSON](data/daily/2025-12-18.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W51 | 83 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 686 | [View JSON](data/monthly/2025-12.json) |

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
