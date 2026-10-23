<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-14-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7519+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">14</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">105</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">911</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7519+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 23, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Making MLLM Latents Matter: Factor-Aware Predictive Learning for Cinematic Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.30926) • [📄 arXiv](https://arxiv.org/abs/2610.30926) • [📥 PDF](https://arxiv.org/pdf/2610.30926)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Cinematic video generation requires both temporal inference from limited inputs (i.e., sparse rather than frame-by-frame guidance) and factor-aware control of character, scene, action, and camera. Multimodal large language models (MLLMs) offer use...

</details>

<details>
<summary><b>2. Meta Expert: Cross-Layer Expert Sharing for Storage- and Retrieval-Efficient MoE</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27598) • [📄 arXiv](https://arxiv.org/abs/2610.27598) • [📥 PDF](https://arxiv.org/pdf/2610.27598)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/InternLM/Intern-S2-Mobius)

> A comparison between the Vanilla and Meta-Expert models shows that, when trained under the optimal settings of the Vanilla model at varying depths and scales, models with Meta Experts achieve loss comparable to the Vanilla baselines while using on...

</details>

<details>
<summary><b>3. Deep Search in Real-World Web Environments</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shao-Rong Su, Shilin Yan, Zhe Huang, Hongliang Chen, Xiaomin Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.30181) • [📄 arXiv](https://arxiv.org/abs/2610.30181) • [📥 PDF](https://arxiv.org/pdf/2610.30181)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Deep Search in the Real-World Web Environments

</details>

<details>
<summary><b>4. Beyond Watching and Searching: Benchmarking Agents for Video-centered Deep Research Reports</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.31158) • [📄 arXiv](https://arxiv.org/abs/2610.31158) • [📥 PDF](https://arxiv.org/pdf/2610.31158)

**💻 Code:** [⭐ Code](https://github.com/marinero4972/OmniDR-Bench) • [⭐ Code](https://github.com/huggingface)

> OmniDR-Bench

</details>

<details>
<summary><b>5. LaLa-World: Scaling Latent World Models with Natural Language Actions</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.27851) • [📄 arXiv](https://arxiv.org/abs/2610.27851) • [📥 PDF](https://arxiv.org/pdf/2610.27851)

**💻 Code:** [⭐ Code](https://github.com/junha1125/LaLa-World) • [⭐ Code](https://github.com/huggingface)

> TL;DR A La nguage-conditioned La tent World model ( LaLa-World ) uses natural language as a shared action interface to learn from simulation, robotics, images, and videos. It predicts future visual representations for efficient planning across dom...

</details>

<details>
<summary><b>6. Code2Video-Compass: A Comprehensive Benchmark for General-Purpose Code-Based Video Generation across T2V, I2V, and V2V</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28416) • [📄 arXiv](https://arxiv.org/abs/2610.28416) • [📥 PDF](https://arxiv.org/pdf/2610.28416)

**💻 Code:** [⭐ Code](https://github.com/code2video-compass/Code2Video-Compass) • [⭐ Code](https://github.com/huggingface)

> We introduce Code2Video-Compass, a comprehensive benchmark for general-purpose code-based video generation across text-to-video, image-to-video, and video-to-video settings. The benchmark comprises 2,104 tasks spanning 10 categories and 24 subtype...

</details>

<details>
<summary><b>7. Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.30755) • [📄 arXiv](https://arxiv.org/abs/2610.30755) • [📥 PDF](https://arxiv.org/pdf/2610.30755)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> RAW-RAG challenges the necessity of complex agentic memory for long-term conversation understanding. By combining dense retrieval and reranking over raw history, it ranks first among compared memory systems in six of seven settings across four ben...

</details>

<details>
<summary><b>8. Policy-Mask: Budget-Aware Adaptation of Diffusion Features to Discriminative Tasks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aibek Alanov, maksim64, Ivan1008

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.28889) • [📄 arXiv](https://arxiv.org/abs/2610.28889) • [📥 PDF](https://arxiv.org/pdf/2610.28889)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TLDR: Policy-Mask trains a REINFORCE policy that jointly picks the timestep, layer, and exactly b channels of a frozen diffusion model, then distills the selected features into a student that runs without the teacher. At 128 of 1280 channels, it b...

</details>

<details>
<summary><b>9. Rethinking Sequence Compression with Test-time Regression</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.31574) • [📄 arXiv](https://arxiv.org/abs/2610.31574) • [📥 PDF](https://arxiv.org/pdf/2610.31574)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Sphere-AI-Lab/ttr-compressor)

> Hi everyone! We introduce TTR-Compressor, a superior and fully practical alternative to standard pooling and projection-based compression. It shows strong empirical gains both as a compressor for block-wise sparse attention in efficient LLMs and a...

</details>

<details>
<summary><b>10. Reading, Writing, Sharing, and Interacting with the World through Video</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shuicheng Yan, Ziwei Liu, Bobo Li, Meng Luo, Shengqiong Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.31668) • [📄 arXiv](https://arxiv.org/abs/2610.31668) • [📥 PDF](https://arxiv.org/pdf/2610.31668)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>11. Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.30882) • [📄 arXiv](https://arxiv.org/abs/2610.30882) • [📥 PDF](https://arxiv.org/pdf/2610.30882)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kouzhizhuo/wait-provenance)

> Reinforcement learning (RL) is credited with teaching reasoning models to write {wait} before reconsidering a step, yet large RL accuracy gains do not require the word. This view can confuse what RL discovers with what a model inherits from its tr...

</details>

<details>
<summary><b>12. Skeleton-Grounded Latent Action Model for Robot World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiang Deng, Jianzong Wang, Yuxin Jiang, Jiayi Lin, Qi Lv

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.31474) • [📄 arXiv](https://arxiv.org/abs/2610.31474) • [📥 PDF](https://arxiv.org/pdf/2610.31474)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> SkeLAM learns structured latent actions from visual transitions with skeleton supervision, separating robot motion from complementary visual changes. This structure supports action-conditioned world modeling and robot action adaptation through a s...

</details>

<details>
<summary><b>13. SCOPE: Compiling Natural-Language Policies into Neurosymbolic Predicate Programs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.31062) • [📄 arXiv](https://arxiv.org/abs/2610.31062) • [📥 PDF](https://arxiv.org/pdf/2610.31062)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Dataset: https://huggingface.co/datasets/Plurai/SCOPE

</details>

<details>
<summary><b>14. Foundation4D: Towards a unified perception and reconstruction model for autonomous vehicles</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.30340) • [📄 arXiv](https://arxiv.org/abs/2610.30340) • [📥 PDF](https://arxiv.org/pdf/2610.30340)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 14 |
| 📅 Today | [`2026-10-23.json`](data/daily/2026-10-23.json) | 14 |
| 📆 This Week | [`2026-W42.json`](data/weekly/2026-W42.json) | 105 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 911 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-23 | 14 | [View JSON](data/daily/2026-10-23.json) |
| 📄 2026-10-22 | 28 | [View JSON](data/daily/2026-10-22.json) |
| 📄 2026-10-21 | 21 | [View JSON](data/daily/2026-10-21.json) |
| 📄 2026-10-20 | 21 | [View JSON](data/daily/2026-10-20.json) |
| 📄 2026-10-19 | 21 | [View JSON](data/daily/2026-10-19.json) |
| 📄 2026-10-18 | 84 | [View JSON](data/daily/2026-10-18.json) |
| 📄 2026-10-17 | 84 | [View JSON](data/daily/2026-10-17.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W42 | 105 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 911 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |

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
