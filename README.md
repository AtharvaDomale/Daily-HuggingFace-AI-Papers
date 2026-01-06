<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-13-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-799+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">13</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">20</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">61</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">799+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 06, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Youtu-Agent: Scaling Agent Productivity with Automated Generation and Hybrid Policy Optimization</b> ⭐ 4.1k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24615) • [📄 arXiv](https://arxiv.org/abs/2512.24615) • [📥 PDF](https://arxiv.org/pdf/2512.24615)

**💻 Code:** [⭐ Code](https://github.com/TencentCloudADP/youtu-tip) • [⭐ Code](https://github.com/TencentCloudADP/youtu-agent) • [⭐ Code](https://github.com/TencentCloudADP/Youtu-agent)

> LONG wait. Youtu-Agent ( https://github.com/TencentCloudADP/Youtu-agent ) now releases its technical report with two major updates, i.e., Automated Generation and Hybrid Policy Optimization. Additionally, we've launched Youtu-Tip ( https://github....

</details>

<details>
<summary><b>2. NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos</b> ⭐ 124</summary>

<br/>

**👥 Authors:** Feng Wang, Junran Peng, renshengjihe, Abyssaledge, Yuppie1204

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00393) • [📄 arXiv](https://arxiv.org/abs/2601.00393) • [📥 PDF](https://arxiv.org/pdf/2601.00393)

**💻 Code:** [⭐ Code](https://github.com/IamCreateAI/NeoVerse)

> NeoVerse is a versatile 4D world model that is capable of 4D reconstruction, novel-trajectory video generation, and rich downstream applications. Project page: https://neoverse-4d.github.io

</details>

<details>
<summary><b>3. Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation</b> ⭐ 65</summary>

<br/>

**👥 Authors:** Sung Ju Hwang, Jaehyeong Jo, Sangwon Jang, jaehong31, taekyungki

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00664) • [📄 arXiv](https://arxiv.org/abs/2601.00664) • [📥 PDF](https://arxiv.org/pdf/2601.00664)

**💻 Code:** [⭐ Code](https://github.com/TaekyungKi/AvatarForcing)

> arXiv explained breakdown of this paper 👉 https://arxivexplained.com/papers/avatar-forcing-real-time-interactive-head-avatar-generation-for-natural-conversation

</details>

<details>
<summary><b>4. SenseNova-MARS: Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24330) • [📄 arXiv](https://arxiv.org/abs/2512.24330) • [📥 PDF](https://arxiv.org/pdf/2512.24330)

**💻 Code:** [⭐ Code](https://github.com/OpenSenseNova/SenseNova-MARS)

> While Vision-Language Models (VLMs) can solve complex tasks through agentic reasoning, their capabilities remain largely constrained to text-oriented chain-of-thought or isolated tool invocation. They fail to exhibit the human-like proficiency req...

</details>

<details>
<summary><b>5. Taming Hallucinations: Boosting MLLMs' Video Understanding via Counterfactual Video Generation</b> ⭐ 29</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24271) • [📄 arXiv](https://arxiv.org/abs/2512.24271) • [📥 PDF](https://arxiv.org/pdf/2512.24271)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/Taming-Hallucinations)

> An interesting work! github: https://github.com/AMAP-ML/Taming-Hallucinations

</details>

<details>
<summary><b>6. AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu-Lun Liu, Zhenjun Zhao, Jiewen Chan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00796) • [📄 arXiv](https://arxiv.org/abs/2601.00796) • [📥 PDF](https://arxiv.org/pdf/2601.00796)

> Reconstructing dynamic 3D scenes from monocular videos requires simultaneously capturing high-frequency appearance details and temporally continuous motion. Existing methods using single Gaussian primitives are limited by their low-pass filtering ...

</details>

<details>
<summary><b>7. Deep Delta Learning</b> ⭐ 234</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00417) • [📄 arXiv](https://arxiv.org/abs/2601.00417) • [📥 PDF](https://arxiv.org/pdf/2601.00417)

**💻 Code:** [⭐ Code](https://github.com/yifanzhang-pro/deep-delta-learning)

> The efficacy of deep residual networks is fundamentally predicated on the identity shortcut connection. While this mechanism effectively mitigates the vanishing gradient problem, it imposes a strictly additive inductive bias on feature transformat...

</details>

<details>
<summary><b>8. Nested Learning: The Illusion of Deep Learning Architectures</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vahab Mirrokni, Peilin Zhong, Meisam Razaviyayn, AliBehrouz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24695) • [📄 arXiv](https://arxiv.org/abs/2512.24695) • [📥 PDF](https://arxiv.org/pdf/2512.24695)

> Nested Learning (NL) is a new learning paradigm for continual learning and machine learning in general.

</details>

<details>
<summary><b>9. The Reasoning-Creativity Trade-off: Toward Creativity-Driven Problem Solving</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00747) • [📄 arXiv](https://arxiv.org/abs/2601.00747) • [📥 PDF](https://arxiv.org/pdf/2601.00747)

> For those of you interested in RLVR, here is a paper that formally characterizes the mechanism behind "diversity collapse" in reasoning models trained with scalar rewards (such as STaR, GRPO, and DPO). The paper introduces a variational framework ...

</details>

<details>
<summary><b>10. Diversity or Precision? A Deep Dive into Next Token Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.22955) • [📄 arXiv](https://arxiv.org/abs/2512.22955) • [📥 PDF](https://arxiv.org/pdf/2512.22955)

> Recent advancements have shown that reinforcement learning (RL) can substantially improve the reasoning abilities of large language models (LLMs).  The effectiveness of such RL training, however, depends critically on the exploration space defined...

</details>

<details>
<summary><b>11. Fast-weight Product Key Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00671) • [📄 arXiv](https://arxiv.org/abs/2601.00671) • [📥 PDF](https://arxiv.org/pdf/2601.00671)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Trellis: Learning to Compress Key-Value Memory in Attention Models (2025) T...

</details>

<details>
<summary><b>12. InfoSynth: Information-Guided Benchmark Synthesis for LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00575) • [📄 arXiv](https://arxiv.org/abs/2601.00575) • [📥 PDF](https://arxiv.org/pdf/2601.00575)

**💻 Code:** [⭐ Code](https://github.com/ishirgarg/infosynth)

> Project Page: https://ishirgarg.github.io/infosynth_web/ Code: https://github.com/ishirgarg/infosynth Dataset: https://huggingface.co/datasets/ishirgarg/InfoSynth

</details>

<details>
<summary><b>13. MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing</b> ⭐ 19</summary>

<br/>

**👥 Authors:** Jian Yang, Ying Tai, Hao Tang, Zeyu Cai, XiaokunSun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00204) • [📄 arXiv](https://arxiv.org/abs/2601.00204) • [📥 PDF](https://arxiv.org/pdf/2601.00204)

**💻 Code:** [⭐ Code](https://github.com/XiaokunSun/MorphAny3D)

> 3D morphing remains challenging due to the difficulty of generating semantically consistent and temporally smooth deformations, especially across categories. We present MorphAny3D, a training-free framework that leverages Structured Latent (SLAT) ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 13 |
| 📅 Today | [`2026-01-06.json`](data/daily/2026-01-06.json) | 13 |
| 📆 This Week | [`2026-W01.json`](data/weekly/2026-W01.json) | 20 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 61 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-06 | 13 | [View JSON](data/daily/2026-01-06.json) |
| 📄 2026-01-05 | 7 | [View JSON](data/daily/2026-01-05.json) |
| 📄 2026-01-04 | 7 | [View JSON](data/daily/2026-01-04.json) |
| 📄 2026-01-03 | 7 | [View JSON](data/daily/2026-01-03.json) |
| 📄 2026-01-02 | 20 | [View JSON](data/daily/2026-01-02.json) |
| 📄 2026-01-01 | 7 | [View JSON](data/daily/2026-01-01.json) |
| 📄 2025-12-31 | 31 | [View JSON](data/daily/2025-12-31.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W01 | 20 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 61 | [View JSON](data/monthly/2026-01.json) |
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
