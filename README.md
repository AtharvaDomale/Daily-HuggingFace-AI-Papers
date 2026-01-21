<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-11-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1238+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">11</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">71</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">500</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1238+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 21, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. ABC-Bench: Benchmarking Agentic Backend Coding in Real-World Development</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11077) • [📄 arXiv](https://arxiv.org/abs/2601.11077) • [📥 PDF](https://arxiv.org/pdf/2601.11077)

**💻 Code:** [⭐ Code](https://github.com/OpenMOSS/ABC-Bench)

> Hi everyone,  I'm one of the authors of ABC-Bench . (arXiv:2601.11077). While building Code Agents, we realized that current benchmarks often stop at "generating correct code snippets." But as developers, we know that real-world backend engineerin...

</details>

<details>
<summary><b>2. Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge</b> ⭐ 48</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08808) • [📄 arXiv](https://arxiv.org/abs/2601.08808) • [📥 PDF](https://arxiv.org/pdf/2601.08808)

**💻 Code:** [⭐ Code](https://github.com/GMLR-Penn/Multiplex-Thinking)

> Large language models often solve complex reasoning tasks more effectively with Chain-of-Thought (CoT), but at the cost of long, low-bandwidth token sequences. Humans, by contrast, often reason softly by maintaining a distribution over plausible n...

</details>

<details>
<summary><b>3. NAACL: Noise-AwAre Verbal Confidence Calibration for LLMs in RAG Systems</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Tianshi Zheng, Qingcheng Zeng, Qing Zong, Rui Wang, Jiayu Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11004) • [📄 arXiv](https://arxiv.org/abs/2601.11004) • [📥 PDF](https://arxiv.org/pdf/2601.11004)

**💻 Code:** [⭐ Code](https://github.com/HKUST-KnowComp/NAACL)

> This paper addresses the often-overlooked problem of confidence calibration for large language models (LLMs) in retrieval-augmented generation (RAG) settings, where noisy retrieved contexts can severely inflate model overconfidence. The authors sy...

</details>

<details>
<summary><b>4. Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation</b> ⭐ 18</summary>

<br/>

**👥 Authors:** Ziyang Yan, Jiachen Tu, Chuhan Song, Tianxingjian Ding, ChongCong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10880) • [📄 arXiv](https://arxiv.org/abs/2601.10880) • [📥 PDF](https://arxiv.org/pdf/2601.10880)

**💻 Code:** [⭐ Code](https://github.com/AIM-Research-Lab/Medical-SAM3.git)

> 🏥 Medical SAM3: Bridging the Gap in Text-Guided Medical Image Segmentation Existing foundation models often face challenges when applying "segment anything" paradigms to medical imaging, particularly in the absence of spatial prompts (bounding box...

</details>

<details>
<summary><b>5. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jack Lindsey, Kyle Fish, Jonathan Michala, Jack Gallagher, Christina Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10387) • [📄 arXiv](https://arxiv.org/abs/2601.10387) • [📥 PDF](https://arxiv.org/pdf/2601.10387)

> arXivlens breakdown of this paper 👉 https://arxivlens.com/PaperView/Details/the-assistant-axis-situating-and-stabilizing-the-default-persona-of-language-models-6264-f01123de Executive Summary Detailed Breakdown Practical Applications

</details>

<details>
<summary><b>6. CoDance: An Unbind-Rebind Paradigm for Robust Multi-Subject Animation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hengshuang, shen12313, DonJoey, fengyutong, kema

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11096) • [📄 arXiv](https://arxiv.org/abs/2601.11096) • [📥 PDF](https://arxiv.org/pdf/2601.11096)

> CoDance: An Unbind-Rebind Paradigm for Robust Multi-Subject Animation

</details>

<details>
<summary><b>7. Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Lecheng Yan, ChrisLee, kksinn, JiahuiGengNLP, rzdiversity

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11061) • [📄 arXiv](https://arxiv.org/abs/2601.11061) • [📥 PDF](https://arxiv.org/pdf/2601.11061)

**💻 Code:** [⭐ Code](https://github.com/idwts/How-RLVR-Activates-Memorization-Shortcuts)

> RLVR is the secret sauce for reasoning models, but it has a dark side. The Spurious Rewards Paradox reveals how models exploit latent contamination to achieve SOTA benchmark results without genuine reasoning. By identifying the specific Anchor-Ada...

</details>

<details>
<summary><b>8. YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08441) • [📄 arXiv](https://arxiv.org/abs/2601.08441) • [📥 PDF](https://arxiv.org/pdf/2601.08441)

**💻 Code:** [⭐ Code](https://github.com/MBZUAI-Paris/YaPO)

> Dense steering vectors often fail due to feature entanglement. YaPO solves this by learning sparse steering vectors directly in a Sparse Autoencoder's latent space using preference data in a DPO-fashion optimization loss. Highlights: Precision & S...

</details>

<details>
<summary><b>9. PubMed-OCR: PMC Open Access OCR Annotations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11425) • [📄 arXiv](https://arxiv.org/abs/2601.11425) • [📥 PDF](https://arxiv.org/pdf/2601.11425)

> PubMed-OCR is an OCR-centric corpus of scientific articles derived from PubMed Central Open Access PDFs. Each page image is annotated with Google Cloud Vision and released in a compact JSON schema with word-, line-, and paragraph-level bounding bo...

</details>

<details>
<summary><b>10. SIN-Bench: Tracing Native Evidence Chains in Long-Context Multimodal Scientific Interleaved Literature</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10108) • [📄 arXiv](https://arxiv.org/abs/2601.10108) • [📥 PDF](https://arxiv.org/pdf/2601.10108)

> Evaluating whether multimodal large language models truly understand long-form scientific papers remains challenging: answer-only metrics and synthetic "Needle-In-A-Haystack" tests often reward answer matching without requiring a causal, evidence-...

</details>

<details>
<summary><b>11. CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09512) • [📄 arXiv](https://arxiv.org/abs/2601.09512) • [📥 PDF](https://arxiv.org/pdf/2601.09512)

**💻 Code:** [⭐ Code](https://github.com/huggingface/lerobot) • [⭐ Code](https://github.com/utiasDSL/clare) • [⭐ Code](https://github.com/huggingface/peft)

> TL;DR 🤖 CLARE enables Vision-Language-Action models to learn new robot tasks without forgetting previous ones — no replay buffers, no task IDs at inference. 🔌 Plug-and-play adapters : Extends PEFT with a new CLARE adapter type 🧠 Smart expansion : ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 11 |
| 📅 Today | [`2026-01-21.json`](data/daily/2026-01-21.json) | 11 |
| 📆 This Week | [`2026-W03.json`](data/weekly/2026-W03.json) | 71 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 500 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-21 | 11 | [View JSON](data/daily/2026-01-21.json) |
| 📄 2026-01-20 | 22 | [View JSON](data/daily/2026-01-20.json) |
| 📄 2026-01-19 | 38 | [View JSON](data/daily/2026-01-19.json) |
| 📄 2026-01-18 | 38 | [View JSON](data/daily/2026-01-18.json) |
| 📄 2026-01-17 | 38 | [View JSON](data/daily/2026-01-17.json) |
| 📄 2026-01-16 | 27 | [View JSON](data/daily/2026-01-16.json) |
| 📄 2026-01-15 | 24 | [View JSON](data/daily/2026-01-15.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W03 | 71 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 500 | [View JSON](data/monthly/2026-01.json) |
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
