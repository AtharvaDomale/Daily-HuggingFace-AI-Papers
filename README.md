<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-20-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-765+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">27</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">27</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">765+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 02, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. mHC: Manifold-Constrained Hyper-Connections</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24880) • [📄 arXiv](https://arxiv.org/abs/2512.24880) • [📥 PDF](https://arxiv.org/pdf/2512.24880)

> DeepSeek released a new paper proposing a novel architecture called mHC (Manifold-Constrained Hyper-Connections).

</details>

<details>
<summary><b>2. Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xinyi Dai, Yinghui Li, Lingfeng Qiao, Jiarui Qin, Junru Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24618) • [📄 arXiv](https://arxiv.org/abs/2512.24618) • [📥 PDF](https://arxiv.org/pdf/2512.24618)

> No abstract available.

</details>

<details>
<summary><b>3. Let It Flow: Agentic Crafting on Rock and Roll, Building the ROME Model within an Open Agentic Learning Ecosystem</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wei Gao, Fangwen Dai, Wanhe An, XiaoXiao Xu, Weixun Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24873) • [📄 arXiv](https://arxiv.org/abs/2512.24873) • [📥 PDF](https://arxiv.org/pdf/2512.24873)

> No abstract available.

</details>

<details>
<summary><b>4. GaMO: Geometry-aware Multi-view Diffusion Outpainting for Sparse-View 3D Reconstruction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu-Lun Liu, Ying-Huan Chen, Chin-Yang Lin, Hao-Jen Chien, Yi-Chuan Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.25073) • [📄 arXiv](https://arxiv.org/abs/2512.25073) • [📥 PDF](https://arxiv.org/pdf/2512.25073)

> Recent advances in 3D reconstruction have achieved remarkable progress in high-quality scene capture from dense multi-view imagery, yet struggle when input views are limited. Various approaches, including regularization techniques, semantic priors...

</details>

<details>
<summary><b>5. A unified framework for detecting point and collective anomalies in operating system logs via collaborative transformers</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23380) • [📄 arXiv](https://arxiv.org/abs/2512.23380) • [📥 PDF](https://arxiv.org/pdf/2512.23380)

**💻 Code:** [⭐ Code](https://github.com/NasirzadehMoh/CoLog)

> No abstract available.

</details>

<details>
<summary><b>6. Scaling Open-Ended Reasoning to Predict the Future</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.25070) • [📄 arXiv](https://arxiv.org/abs/2512.25070) • [📥 PDF](https://arxiv.org/pdf/2512.25070)

**💻 Code:** [⭐ Code](https://github.com/OpenForecaster/scaling-forecasting-training)

> No abstract available.

</details>

<details>
<summary><b>7. PhyGDPO: Physics-Aware Groupwise Direct Preference Optimization for Physically Consistent Text-to-Video Generation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24551) • [📄 arXiv](https://arxiv.org/abs/2512.24551) • [📥 PDF](https://arxiv.org/pdf/2512.24551)

**💻 Code:** [⭐ Code](https://github.com/caiyuanhao1998/Open-PhyGDPO)

> A data construction pipeline and a new DPO framework for physically consistent Text-to-video generation

</details>

<details>
<summary><b>8. AI Meets Brain: Memory Systems from Cognitive Neuroscience to Autonomous Agents</b> ⭐ 24</summary>

<br/>

**👥 Authors:** Shixin Jiang, Jiaqi Zhou, Chang Li, Hao Li, Jiafeng Liang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23343) • [📄 arXiv](https://arxiv.org/abs/2512.23343) • [📥 PDF](https://arxiv.org/pdf/2512.23343)

**💻 Code:** [⭐ Code](https://github.com/AgentMemory/Huaman-Agent-Memory)

> https://github.com/AgentMemory/Huaman-Agent-Memory

</details>

<details>
<summary><b>9. GR-Dexter Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24210) • [📄 arXiv](https://arxiv.org/abs/2512.24210) • [📥 PDF](https://arxiv.org/pdf/2512.24210)

> VLAs go from grippers to 21 DoF dexterous ByteDexter V2 :)

</details>

<details>
<summary><b>10. Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of the Reasoning Process</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23988) • [📄 arXiv](https://arxiv.org/abs/2512.23988) • [📥 PDF](https://arxiv.org/pdf/2512.23988)

> This is an impressive piece of work. Not only for the elegance of the sparse autoencoder pipeline, but because the empirical results reveal something far deeper than what is stated in the paper. Your SAE-derived “reasoning vectors” behave exactly ...

</details>

<details>
<summary><b>11. Pretraining Frame Preservation in Autoregressive Video Memory Compression</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Beijia Lu, Chong Zeng, Muyang Li, Shengqu Cai, Lvmin Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23851) • [📄 arXiv](https://arxiv.org/abs/2512.23851) • [📥 PDF](https://arxiv.org/pdf/2512.23851)

**💻 Code:** [⭐ Code](https://github.com/lllyasviel/PFP)

> Arxiv: https://arxiv.org/abs/2512.23851 Repo: https://github.com/lllyasviel/PFP

</details>

<details>
<summary><b>12. SpaceTimePilot: Generative Rendering of Dynamic Scenes Across Space and Time</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tuanfeng Y. Wang, Yulia Gryaditskaya, Xuelin Chen, Hyeonho Jeong, Zhening Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.25075) • [📄 arXiv](https://arxiv.org/abs/2512.25075) • [📥 PDF](https://arxiv.org/pdf/2512.25075)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API BulletTime: Decoupled Control of Time and Camera Pose for Video Generation ...

</details>

<details>
<summary><b>13. JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22905) • [📄 arXiv](https://arxiv.org/abs/2503.23377) • [📥 PDF](https://arxiv.org/pdf/2512.22905)

**💻 Code:** [⭐ Code](https://github.com/JavisVerse/JavisGPT)

> 🔥🔥🔥 JavisGPT 🌟 We introduce JavisGPT, a multimodal LLM that can understand audiovisual inputs and simultaneously generate synchronized sounding videos in a unified model. 🤠 We contribute JavisInst-Omni, a dataset to facilitate diverse and complex ...

</details>

<details>
<summary><b>14. Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mahşuk Taylan, Ahmet Feridun Işık, Selin Vulga Işık, Atakanisik

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22564) • [📄 arXiv](https://arxiv.org/abs/2512.22564) • [📥 PDF](https://arxiv.org/pdf/2512.22564)

**💻 Code:** [⭐ Code](https://github.com/Atakanisik/ICBHI-AST-SAM)

> Hi all, We present a robust framework for Lung Sound Classification using AST backbones enhanced with SAM optimizer . Traditional transformers often struggle with limited medical data, but our experiments show that geometry-aware optimization (SAM...

</details>

<details>
<summary><b>15. BEDA: Belief Estimation as Probabilistic Constraints for Performing Strategic Dialogue Acts</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mengmeng Wang, Chenxi Li, Qi Shen, Zhaoxin Yu, Hengli Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24885) • [📄 arXiv](https://arxiv.org/abs/2512.24885) • [📥 PDF](https://arxiv.org/pdf/2512.24885)

> Arxiv: https://arxiv.org/abs/2512.24885 X thread: https://x.com/Hengli_Li_pku/status/2006606887652045158

</details>

<details>
<summary><b>16. Forging Spatial Intelligence: A Roadmap of Multi-Modal Data Pre-Training for Autonomous Systems</b> ⭐ 105</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24385) • [📄 arXiv](https://arxiv.org/abs/2512.24385) • [📥 PDF](https://arxiv.org/pdf/2512.24385)

**💻 Code:** [⭐ Code](https://github.com/worldbench/awesome-spatial-intelligence)

> GitHub at https://github.com/worldbench/awesome-spatial-intelligence

</details>

<details>
<summary><b>17. Figure It Out: Improving the Frontier of Reasoning with Active Visual Thinking</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Jie Zhou, Fandong Meng, Meiqi Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24297) • [📄 arXiv](https://arxiv.org/abs/2512.24297) • [📥 PDF](https://arxiv.org/pdf/2512.24297)

**💻 Code:** [⭐ Code](https://github.com/chenmeiqii/FIGR)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API V-Thinker: Interactive Thinking with Images (2025) Interleaved Latent Visua...

</details>

<details>
<summary><b>18. Guiding a Diffusion Transformer with the Internal Dynamics of Itself</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24176) • [📄 arXiv](https://arxiv.org/abs/2512.24176) • [📥 PDF](https://arxiv.org/pdf/2512.24176)

**💻 Code:** [⭐ Code](https://github.com/CVL-UESTC/Internal-Guidance)

> 🔥 New SOTA on 256 × 256 ImageNet generation. We present Internal Guidance (IG), a simple yet powerful guidance mechanism for Diffusion Transformers. LightningDiT-XL/1 + IG sets a new state of the art with FID = 1.07 on ImageNet (balanced sampling)...

</details>

<details>
<summary><b>19. Factorized Learning for Temporally Grounded Video-Language Models</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24097) • [📄 arXiv](https://arxiv.org/abs/2512.24097) • [📥 PDF](https://arxiv.org/pdf/2512.24097)

**💻 Code:** [⭐ Code](https://github.com/nusnlp/d2vlm)

> We tackle temporally grounded video-language understanding from a factorized perspective. Some key takeaways: [1] We emphasize the distinct yet causally dependent nature of temporal grounding and textual response. [2] Our study highlights the impo...

</details>

<details>
<summary><b>20. Valori: A Deterministic Memory Substrate for AI Systems</b> ⭐ 2</summary>

<br/>

**👥 Authors:** varam17

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22280) • [📄 arXiv](https://arxiv.org/abs/2512.22280) • [📥 PDF](https://arxiv.org/pdf/2512.22280)

**💻 Code:** [⭐ Code](https://github.com/varshith-Git/Valori-Kernel)

> Valori: A Deterministic Fixed-Point Vector Kernel Tags: vector-database , rust , determinism , finance , audit , hnsw , fixed-point , systems-engineering TL;DR Floating-point math causes vector search results to drift between ARM (Mac) and x86 (Li...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 20 |
| 📅 Today | [`2026-01-02.json`](data/daily/2026-01-02.json) | 20 |
| 📆 This Week | [`2026-W00.json`](data/weekly/2026-W00.json) | 27 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 27 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-02 | 20 | [View JSON](data/daily/2026-01-02.json) |
| 📄 2026-01-01 | 7 | [View JSON](data/daily/2026-01-01.json) |
| 📄 2025-12-31 | 31 | [View JSON](data/daily/2025-12-31.json) |
| 📄 2025-12-30 | 14 | [View JSON](data/daily/2025-12-30.json) |
| 📄 2025-12-29 | 7 | [View JSON](data/daily/2025-12-29.json) |
| 📄 2025-12-28 | 7 | [View JSON](data/daily/2025-12-28.json) |
| 📄 2025-12-27 | 7 | [View JSON](data/daily/2025-12-27.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W00 | 27 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 27 | [View JSON](data/monthly/2026-01.json) |
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
