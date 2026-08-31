<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-12-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6128+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📆 This Month</b><br/><font size="5">747</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6128+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 31, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning</b> ⭐ 202</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27549) • [📄 arXiv](https://arxiv.org/abs/2608.27549) • [📥 PDF](https://arxiv.org/pdf/2608.27549)

**💻 Code:** [⭐ Code](https://github.com/mirros-lab/code-as-world) • [⭐ Code](https://github.com/huggingface)

> Pixels are evidence of the physical world, not its ontology. A pixel-level observation records how the world appears at a particular moment and from a particular viewpoint, but does not directly specify what exists within it, how it is structured,...

</details>

<details>
<summary><b>2. ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.28476) • [📄 arXiv](https://arxiv.org/abs/2608.28476) • [📥 PDF](https://arxiv.org/pdf/2608.28476)

**💻 Code:** [⭐ Code](https://github.com/Tencent/ContextPilot) • [⭐ Code](https://github.com/huggingface)

> ContextPilot extends context management with planning, structured memory, and soft context offloading. Its context-aware partial rollout focuses exploration on sensitive context-editing decisions, while fine-grained credit assignment trains interm...

</details>

<details>
<summary><b>3. J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26582) • [📄 arXiv](https://arxiv.org/abs/2608.26582) • [📥 PDF](https://arxiv.org/pdf/2608.26582)

**💻 Code:** [⭐ Code](https://github.com/GyoukChu/J-Zero) • [⭐ Code](https://github.com/huggingface)

> Frozen Judge set the upper bound of self-evolving LLMs. By co-evolving Judge with the current frontier of self-evolution, a highly performant and sustained self-evolution can be realized.

</details>

<details>
<summary><b>4. Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27550) • [📄 arXiv](https://arxiv.org/abs/2608.27550) • [📥 PDF](https://arxiv.org/pdf/2608.27550)

**💻 Code:** [⭐ Code](https://github.com/starVLA/VLAct) • [⭐ Code](https://github.com/huggingface)

> Strong, open, and research-friendly. VLAct releases the data, models, and complete training/fine-tuning pipeline, with full continued pre-training requiring only 16 GPUs . It achieves 92.5% on RoboTwin 2.0 and ranks #6 on RoboDojo by Success Rate,...

</details>

<details>
<summary><b>5. StepGuard: Learning Step-Level Guardrails with Scalable Supervision and Safety-Utility Balancing</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24777) • [📄 arXiv](https://arxiv.org/abs/2608.24777) • [📥 PDF](https://arxiv.org/pdf/2608.24777)

**💻 Code:** [⭐ Code](https://github.com/zheng977/StepGuard) • [⭐ Code](https://github.com/huggingface)

> LLM-based agents can interact with external environments through tool invocation, but this capability also introduces security risks such as file modification, information leakage, and unauthorized actions. Existing guardrails often evaluate compl...

</details>

<details>
<summary><b>6. Blind Men and the Elephant: Probing the Epistemic Myopia of LLMs under Long-Tail Divergent Knowledge</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.28478) • [📄 arXiv](https://arxiv.org/abs/2608.28478) • [📥 PDF](https://arxiv.org/pdf/2608.28478)

**💻 Code:** [⭐ Code](https://github.com/Tencent/ElephantBench) • [⭐ Code](https://github.com/huggingface)

> ElephantBench is a closed-book knowledge probe for evaluating whether a language model remembers long-tail facts and whether it recalls the different verified accounts associated with those facts. The benchmark contains 1,094 questions using two f...

</details>

<details>
<summary><b>7. Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction</b> ⭐ 229</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27529) • [📄 arXiv](https://arxiv.org/abs/2608.27529) • [📥 PDF](https://arxiv.org/pdf/2608.27529)

**💻 Code:** [⭐ Code](https://github.com/amap-cvlab/ABot-Recon) • [⭐ Code](https://github.com/huggingface)

> ABot-Recon turns a single continuous video into a globally consistent 3D reconstruction in real time. Whether walking around a building with a phone, driving through city streets with a dashcam, or flying a drone over a campus, it reconstructs lon...

</details>

<details>
<summary><b>8. Rubric-to-Code Credit Assignment for Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.27906) • [📄 arXiv](https://arxiv.org/abs/2608.27906) • [📥 PDF](https://arxiv.org/pdf/2608.27906)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. LMSM: LLM Security Framework Inspired by Linux Security Modules</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25697) • [📄 arXiv](https://arxiv.org/abs/2608.25697) • [📥 PDF](https://arxiv.org/pdf/2608.25697)

**💻 Code:** [⭐ Code](https://github.com/xiuyuz/LMSM) • [⭐ Code](https://github.com/huggingface)

> LMSM converts pluggable model-internal evidence into per-request decisions and selective enforcement under continuous batching. By separating evidence backends from policy and enforcement, it provides a stable path for adopting stronger interpreta...

</details>

<details>
<summary><b>10. Video Generative Models as Geometry Learner</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiankang Deng, Xiatian Zhu, Zhensong Zhang, Jifei Song, Haosen Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.28549) • [📄 arXiv](https://arxiv.org/abs/2608.28549) • [📥 PDF](https://arxiv.org/pdf/2608.28549)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>11. PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24115) • [📄 arXiv](https://arxiv.org/abs/2608.24115) • [📥 PDF](https://arxiv.org/pdf/2608.24115)

**💻 Code:** [⭐ Code](https://github.com/worv-ai/PonderPounce) • [⭐ Code](https://github.com/huggingface)

> We’re excited to share PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control . Many robot tasks require remembering information that is no longer visible, such as a briefly shown target, an earlier instruction, or a demons...

</details>

<details>
<summary><b>12. Training, learning and inference: unified dynamics of neural systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** wind342

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.20965) • [📄 arXiv](https://arxiv.org/abs/2608.20965) • [📥 PDF](https://arxiv.org/pdf/2608.20965)

**💻 Code:** [⭐ Code](https://github.com/wind342/gfg-training-learning-inference-experiments) • [⭐ Code](https://github.com/huggingface)

> This work presents an experiment-first causal study of training, learning, inference, and feedback in realized neural networks. Across Transformer/Adam, ResNet/SGD momentum, and diffusion U-Net/AdamW systems, the same core relation structure is pr...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 12 |
| 📅 Today | [`2026-08-31.json`](data/daily/2026-08-31.json) | 12 |
| 📆 This Week | [`2026-W35.json`](data/weekly/2026-W35.json) | 12 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 747 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-31 | 12 | [View JSON](data/daily/2026-08-31.json) |
| 📄 2026-08-30 | 23 | [View JSON](data/daily/2026-08-30.json) |
| 📄 2026-08-29 | 23 | [View JSON](data/daily/2026-08-29.json) |
| 📄 2026-08-28 | 18 | [View JSON](data/daily/2026-08-28.json) |
| 📄 2026-08-27 | 26 | [View JSON](data/daily/2026-08-27.json) |
| 📄 2026-08-26 | 35 | [View JSON](data/daily/2026-08-26.json) |
| 📄 2026-08-25 | 22 | [View JSON](data/daily/2026-08-25.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W35 | 12 | [View JSON](data/weekly/2026-W35.json) |
| 📅 2026-W34 | 173 | [View JSON](data/weekly/2026-W34.json) |
| 📅 2026-W33 | 213 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
