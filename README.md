<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-19-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6864+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">19</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">19</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">256</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6864+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 05, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. FrameMorrow: Future-guided Frame Selection with Prospective Tokens for Long-Horizon Video Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38839) • [📄 arXiv](https://arxiv.org/abs/2609.38839) • [📥 PDF](https://arxiv.org/pdf/2609.38839)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/YinBo0927/FrameMorrow)

> We introduce FrameMorrow, a prospective frame selector for long-horizon video generation. Instead of selecting historical frames based only on their relevance to the present, FrameMorrow anticipates future information needs and uses them to identi...

</details>

<details>
<summary><b>2. World Embedding Benchmark</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03632) • [📄 arXiv](https://arxiv.org/abs/2610.03632) • [📥 PDF](https://arxiv.org/pdf/2610.03632)

**💻 Code:** [⭐ Code](https://github.com/World-Representation-Lab/World-Embedding-Benchmark) • [⭐ Code](https://github.com/huggingface)

> We introduce World Embedding Benchmark , a benchmark for studying how video representations capture and expose physical information. Building the benchmark requires substantial simulation engineering: we develop physics simulation pipelines spanni...

</details>

<details>
<summary><b>3. Source Preference in the Wild: How LLM Agents Favor Items by Source, and How to Reduce It</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03195) • [📄 arXiv](https://arxiv.org/abs/2610.03195) • [📥 PDF](https://arxiv.org/pdf/2610.03195)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> If two options meet your requirements equally well, will your AI agent favor the one from a source it prefers? Across 12 models and three domains---shopping, hotels, and scholarly search---we found that source preferences can sway your agents' cho...

</details>

<details>
<summary><b>4. Scaling Trajectories for Complex Tasks through Recursive Self-Rewrite</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02826) • [📄 arXiv](https://arxiv.org/abs/2610.02826) • [📥 PDF](https://arxiv.org/pdf/2610.02826)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>5. Science Utopia? Closed-Loop LLM Simulation of Academic Research Ecosystems</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01257) • [📄 arXiv](https://arxiv.org/abs/2610.01257) • [📥 PDF](https://arxiv.org/pdf/2610.01257)

**💻 Code:** [⭐ Code](https://github.com/Ahren09/ScienceUtopia) • [⭐ Code](https://github.com/huggingface)

> Top conferences now have over 30,000 submissions. What's next? AI is accelerating research. How can our institutions keep pace? 🔬 As the research community grows, peer review, funding, and career incentives help shape which ideas advance—and who c...

</details>

<details>
<summary><b>6. PDE-JEPA: Predictive Representation Learning of Latent Dynamics Modeling for Parametric PDEs</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34715) • [📄 arXiv](https://arxiv.org/abs/2609.34715) • [📥 PDF](https://arxiv.org/pdf/2609.34715)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tanpig-X/PDE-JEPA)

> To our knowledge, we are the first to systematically investigate JEPA for parametric PDEs and to make it work effectively for PDE forecasting. We find that informative JEPA representations alone do not guarantee accurate rollout, and introduce PDE...

</details>

<details>
<summary><b>7. DEFINE: Exemplar-Guided Accent Control for Zero-Shot TTS</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32777) • [📄 arXiv](https://arxiv.org/abs/2609.32777) • [📥 PDF](https://arxiv.org/pdf/2609.32777)

**💻 Code:** [⭐ Code](https://github.com/AMAAI-Lab/define) • [⭐ Code](https://github.com/huggingface)

> Zero-shot text-to-speech (TTS) can reproduce an unseen speaker from a short reference recording, but typically entangles speaker identity and accent within the same reference. We introduce DEFINE, an end-to-end framework that decouples these facto...

</details>

<details>
<summary><b>8. Diptych: Scoped, AI-Interpreted Comparison for Reference Listening in Music Production</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39963) • [📄 arXiv](https://arxiv.org/abs/2609.39963) • [📥 PDF](https://arxiv.org/pdf/2609.39963)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reference listening is a common strategy in music production, but current comparison tools often obscure a key human judgment: deciding what should be compared. We present DipTych, an AI-assisted system that lets users define comparison scope acro...

</details>

<details>
<summary><b>9. Science or Slop?: Benchmarking and Mitigating Scientific Slop in AI-Generated Papers</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00531) • [📄 arXiv](https://arxiv.org/abs/2610.00531) • [📥 PDF](https://arxiv.org/pdf/2610.00531)

**💻 Code:** [⭐ Code](https://github.com/yerimoh/ScientificSlop) • [⭐ Code](https://github.com/huggingface)

> Author here! New preprint: Science or Slop? ❓ Quiz: Every sentence passes the AI detector. Every citation is real. Human or AI? -> If you can’t tell, neither can the detectors. 😅 ✨ Findings: (define scientific slop) Slop at the level of scientific...

</details>

<details>
<summary><b>10. MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02824) • [📄 arXiv](https://arxiv.org/abs/2610.02824) • [📥 PDF](https://arxiv.org/pdf/2610.02824)

**💻 Code:** [⭐ Code](https://github.com/metarubric/metarubric) • [⭐ Code](https://github.com/huggingface)

> .

</details>

<details>
<summary><b>11. ProAR: Learning Prospective Reasoning with Autoregressive Video Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03664) • [📄 arXiv](https://arxiv.org/abs/2610.03664) • [📥 PDF](https://arxiv.org/pdf/2610.03664)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/luka-group/ProAR)

> ProAR: Learning prospective guidance to transform autoregressive video generation into a visual reasoning process.

</details>

<details>
<summary><b>12. Covert Assistance: Helpful LLM Agents Evade Oversight in Multi-Agent Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39050) • [📄 arXiv](https://arxiv.org/abs/2609.39050) • [📥 PDF](https://arxiv.org/pdf/2609.39050)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Would love your feedback and thoughts!

</details>

<details>
<summary><b>13. Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36529) • [📄 arXiv](https://arxiv.org/abs/2609.36529) • [📥 PDF](https://arxiv.org/pdf/2609.36529)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OliverSieberling/TriadicLinearAttention)

> Recurrent neural networks (RNNs) compress the historical context into a memory state of fixed size, thus allowing for constant-time inference. The memory state size is a crucial factor in their performance, as exemplified by the strong performance...

</details>

<details>
<summary><b>14. On-Policy Parameter Update Direction Underlies Generalization in LLM Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36659) • [📄 arXiv](https://arxiv.org/abs/2609.36659) • [📥 PDF](https://arxiv.org/pdf/2609.36659)

**💻 Code:** [⭐ Code](https://github.com/ssfgunner/OPSFT) • [⭐ Code](https://github.com/huggingface)

> Why does on-policy training generalize better than SFT? We show that the direction of parameter updates plays a crucial role. By extracting update directions from a few on-policy steps and constraining subsequent SFT updates to these directions, o...

</details>

<details>
<summary><b>15. GTR: Gated Token Recurrence for Efficient Dense Prediction</b> ⭐ 38</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.26590) • [📄 arXiv](https://arxiv.org/abs/2609.26590) • [📥 PDF](https://arxiv.org/pdf/2609.26590)

**💻 Code:** [⭐ Code](https://github.com/Intellindust-AI-Lab/GTR) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>16. EditHero: A Benchmark for Long-Horizon Part-Level 3D Editing and Vibe Modeling</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Lian Fu, Runyi Li, Muyao Niu, Yu-Ju Tsai, Ruihan Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02298) • [📄 arXiv](https://arxiv.org/abs/2610.02298) • [📥 PDF](https://arxiv.org/pdf/2610.02298)

**💻 Code:** [⭐ Code](https://github.com/AlayaLab/EditHero) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>17. Octrees as an Explicit 3D Language</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Yadong Mu, Wei Zhang, Pengfei Xiong, Si-Tong Wei, Ran Dan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02388) • [📄 arXiv](https://arxiv.org/abs/2610.02388) • [📥 PDF](https://arxiv.org/pdf/2610.02388)

**💻 Code:** [⭐ Code](https://github.com/octree-nn/octllm) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>18. Skill2Real: Agentic Skill Learning for Zero-Shot Sim-to-Real Robot Manipulation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yanjia Huang, Yunuo Chen, Chang Yu, Siyu Ma, Xincheng He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02788) • [📄 arXiv](https://arxiv.org/abs/2610.02788) • [📥 PDF](https://arxiv.org/pdf/2610.02788)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>19. Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01153) • [📄 arXiv](https://arxiv.org/abs/2610.01153) • [📥 PDF](https://arxiv.org/pdf/2610.01153)

**💻 Code:** [⭐ Code](https://github.com/hed-ucas/LOOM) • [⭐ Code](https://github.com/huggingface)

> An open-source recipe to scale loops beyond twice for Looped MoE at the ISO-FLOP setting.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 19 |
| 📅 Today | [`2026-10-05.json`](data/daily/2026-10-05.json) | 19 |
| 📆 This Week | [`2026-W40.json`](data/weekly/2026-W40.json) | 19 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 256 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-05 | 19 | [View JSON](data/daily/2026-10-05.json) |
| 📄 2026-10-04 | 84 | [View JSON](data/daily/2026-10-04.json) |
| 📄 2026-10-03 | 84 | [View JSON](data/daily/2026-10-03.json) |
| 📄 2026-10-02 | 35 | [View JSON](data/daily/2026-10-02.json) |
| 📄 2026-10-01 | 34 | [View JSON](data/daily/2026-10-01.json) |
| 📄 2026-09-30 | 45 | [View JSON](data/daily/2026-09-30.json) |
| 📄 2026-09-29 | 37 | [View JSON](data/daily/2026-09-29.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W40 | 19 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 256 | [View JSON](data/monthly/2026-10.json) |
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
