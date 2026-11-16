<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-12-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8019+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">12</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">12</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">291</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8019+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 16, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Envisioning Agentic Robotics: A New Era of Robotic Systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yiheng Lyu, Siyao Li, Duo Li, Qingfeng Lan, Grid Robotics

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.18359) • [📄 arXiv](https://arxiv.org/abs/2611.18359) • [📥 PDF](https://arxiv.org/pdf/2611.18359)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>2. Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhihao Yang, Yukun Chen, Qi Deng, Kuan Zhang, Run Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.18397) • [📄 arXiv](https://arxiv.org/abs/2611.18397) • [📥 PDF](https://arxiv.org/pdf/2611.18397)

**💻 Code:** [⭐ Code](https://github.com/RainBowLuoCS/Gamer) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>3. Loopback Transformer: Towards a Unified View of Residual Looped Transformers</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.07226) • [📄 arXiv](https://arxiv.org/abs/2611.07226) • [📥 PDF](https://arxiv.org/pdf/2611.07226)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yancyou/Loopback)

> In this paper, we study how residual connections should retain and reuse information when the same Transformer blocks run repeatedly. Inspired by recurrent memory, we develop a unified view of residual connections as memory operating along network...

</details>

<details>
<summary><b>4. Emergent Biological Capabilities in a Foundation Model for Molecular Interactions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.14504) • [📄 arXiv](https://arxiv.org/abs/2611.14504) • [📥 PDF](https://arxiv.org/pdf/2611.14504)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Bonbon is a vertical foundation model for biology that independently learned the hidden relationships governing molecular interactions, just as LLMs learned math without being explicitly trained on it. Accepted to the NeurIPS 2026 main track as an...

</details>

<details>
<summary><b>5. MEM-SQL: Memory-Empowered Text-to-SQL</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.18338) • [📄 arXiv](https://arxiv.org/abs/2611.18338) • [📥 PDF](https://arxiv.org/pdf/2611.18338)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Most Text-to-SQL benchmarks overlook a common scenario where a new request can reuse parts of an earlier query. For example, this month’s revenue report by region can use the same joins and revenue calculation as last month’s, with only the date r...

</details>

<details>
<summary><b>6. HALO: Human-Aligned Data Collection for WhoLe-body MObile Manipulation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16439) • [📄 arXiv](https://arxiv.org/abs/2611.16439) • [📥 PDF](https://arxiv.org/pdf/2611.16439)

**💻 Code:** [⭐ Code](https://github.com/robbyant-research/Lingbot-ALOHA) • [⭐ Code](https://github.com/huggingface)

> What if you could teach a mobile robot by walking with it and guiding its grippers by hand? HALO collects demonstrations up to 2.9× faster. The learned policies grasp and steer a chair precisely while staying in motion, pull apart tightly fitted L...

</details>

<details>
<summary><b>7. AI Reviews Are Biased Towards the Appendix</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17811) • [📄 arXiv](https://arxiv.org/abs/2611.17811) • [📥 PDF](https://arxiv.org/pdf/2611.17811)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cincynlp/review_appendix)

> Large language models (LLMs) are increasingly used by reviewers to highlight weaknesses during the reviewing process, and their outputs often shape the reviews that humans submit. Prior work studies how LLM reviews align with human ones, yet we kn...

</details>

<details>
<summary><b>8. JVT: Just Video Transformer makes pixel-space video diffusion simple</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haoyu Guo, Hua Geng, Jianjun Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.18549) • [📄 arXiv](https://arxiv.org/abs/2611.18549) • [📥 PDF](https://arxiv.org/pdf/2611.18549)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZhouTimeMachine/JVT)

> Direct diffusion modeling of high-dimensional raw video data is challenging, and existing pixel-space methods often rely on specialized architectures. We introduce JVT (Just Video Transformer), a simple large-patch DiT-style Transformer for pixel-...

</details>

<details>
<summary><b>9. Rethinking in Spikes: Mitigating Hallucinations in MDLMs with Step-Aware Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.12901) • [📄 arXiv](https://arxiv.org/abs/2611.12901) • [📥 PDF](https://arxiv.org/pdf/2611.12901)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> • We analyze the relationship between entropy spike steps and hallucinations in multimodal diffusion models from the perspective of uncertainty dynamics across denoising steps. • We propose StepRefine, a plug-and-play decoding approach that mitiga...

</details>

<details>
<summary><b>10. Aligning AI Teams</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Erik Jones, Jascha Sohl-Dickstein, Morgan Jane Matthews, Siddarth Srinivasan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17820) • [📄 arXiv](https://arxiv.org/abs/2611.17820) • [📥 PDF](https://arxiv.org/pdf/2611.17820)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>11. GH-AWTrace: A Large-Scale Dataset of GitHub Agentic Workflow Execution Logs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.06866) • [📄 arXiv](https://arxiv.org/abs/2611.06866) • [📥 PDF](https://arxiv.org/pdf/2611.06866)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A dataset ( https://huggingface.co/datasets/jaykay9999/GHAWTrace ) containing agent trajectories from real-world agentic workflows in software projects. You can use the dataset to study: single-agent vs. multi-agent memory collaboration & communic...

</details>

<details>
<summary><b>12. CANTraj: Recovering Vehicle Trajectories from CAN Signals in Naturalistic Driving</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.18442) • [📄 arXiv](https://arxiv.org/abs/2611.18442) • [📥 PDF](https://arxiv.org/pdf/2611.18442)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: Removing GPS from vehicle logs does not protect location privacy: we introduce CANTraj (3,940 naturalistic drives paired with GNSS trajectories) and CANTraj-Bench (1,380 hours, 85,919 km) to benchmark six re-implemented route-inference atta...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 12 |
| 📅 Today | [`2026-11-16.json`](data/daily/2026-11-16.json) | 12 |
| 📆 This Week | [`2026-W46.json`](data/weekly/2026-W46.json) | 12 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 291 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-16 | 12 | [View JSON](data/daily/2026-11-16.json) |
| 📄 2026-11-15 | 34 | [View JSON](data/daily/2026-11-15.json) |
| 📄 2026-11-14 | 34 | [View JSON](data/daily/2026-11-14.json) |
| 📄 2026-11-13 | 6 | [View JSON](data/daily/2026-11-13.json) |
| 📄 2026-11-12 | 3 | [View JSON](data/daily/2026-11-12.json) |
| 📄 2026-11-11 | 11 | [View JSON](data/daily/2026-11-11.json) |
| 📄 2026-11-10 | 9 | [View JSON](data/daily/2026-11-10.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W46 | 12 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 291 | [View JSON](data/monthly/2026-11.json) |
| 🗓️ 2026-10 | 1120 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |

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
