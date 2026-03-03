<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2645+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">22</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">50</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">78</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2645+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 03, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. dLLM: Simple Diffusion Language Modeling</b> ⭐ 1.92k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22661) • [📄 arXiv](https://arxiv.org/abs/2602.22661) • [📥 PDF](https://arxiv.org/pdf/2602.22661)

**💻 Code:** [⭐ Code](https://github.com/ZHZisZZ/dllm)

> https://github.com/ZHZisZZ/dllm

</details>

<details>
<summary><b>2. Enhancing Spatial Understanding in Image Generation via Reward Modeling</b> ⭐ 41</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.24233) • [📄 arXiv](https://arxiv.org/abs/2602.24233) • [📥 PDF](https://arxiv.org/pdf/2602.24233)

**💻 Code:** [⭐ Code](https://github.com/DAGroup-PKU/SpatialT2I)

> Accepted at CVPR 2026.

</details>

<details>
<summary><b>3. CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.24286) • [📄 arXiv](https://arxiv.org/abs/2602.24286) • [📥 PDF](https://arxiv.org/pdf/2602.24286)

> arXivLens breakdown of this paper 👉 https://arxivlens.com/PaperView/Details/cuda-agent-large-scale-agentic-rl-for-high-performance-cuda-kernel-generation-5816-50c4adfe Executive Summary Detailed Breakdown Practical Applications

</details>

<details>
<summary><b>4. Recovered in Translation: Efficient Pipeline for Automated Translation of Benchmarks and Datasets</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22207) • [📄 arXiv](https://arxiv.org/abs/2602.22207) • [📥 PDF](https://arxiv.org/pdf/2602.22207)

**💻 Code:** [⭐ Code](https://github.com/insait-institute/ritranslation)

> Existing multilingual benchmarks suffer from systematic translation quality issues that compromise reliable evaluation, including grammatical inconsistencies, context loss, and structural problems. We present an automated framework that applies te...

</details>

<details>
<summary><b>5. Mode Seeking meets Mean Seeking for Fast Long Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.24289) • [📄 arXiv](https://arxiv.org/abs/2602.24289) • [📥 PDF](https://arxiv.org/pdf/2602.24289)

> https://primecai.github.io/mmm/

</details>

<details>
<summary><b>6. LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23881) • [📄 arXiv](https://arxiv.org/abs/2602.23881) • [📥 PDF](https://arxiv.org/pdf/2602.23881)

> Datasets: https://huggingface.co/collections/nebius/infinity-instruct-completions Weights: https://huggingface.co/collections/nebius/lk-speculators

</details>

<details>
<summary><b>7. CiteAudit: You Cited It, But Did You Read It? A Benchmark for Verifying Scientific References in the LLM Era</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23452) • [📄 arXiv](https://arxiv.org/abs/2602.23452) • [📥 PDF](https://arxiv.org/pdf/2602.23452)

> Scientific research relies on accurate citation for attribution and integrity, yet large language models (LLMs) introduce a new risk: fabricated references that appear plausible but correspond to no real publications. Such hallucinated citations h...

</details>

<details>
<summary><b>8. Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Seong Joon Oh, Andrea Dittadi, Gigglingface

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.24264) • [📄 arXiv](https://arxiv.org/abs/2602.24264) • [📥 PDF](https://arxiv.org/pdf/2602.24264)

**💻 Code:** [⭐ Code](https://github.com/oshapio/necessary-compositionality)

> Code: https://github.com/oshapio/necessary-compositionality

</details>

<details>
<summary><b>9. InfoNCE Induces Gaussian Distribution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.24012) • [📄 arXiv](https://arxiv.org/abs/2602.24012) • [📥 PDF](https://arxiv.org/pdf/2602.24012)

**💻 Code:** [⭐ Code](https://github.com/rbetser/InfoNCE-induces-Gaussian-distribution)

> As scaling large models begins to saturate, it becomes increasingly important to revisit and deeply understand the fundamental tools we rely on. In this work, we return to a basic question in contrastive learning: what distribution does InfoNCE ac...

</details>

<details>
<summary><b>10. Accelerating Masked Image Generation by Learning Latent Controlled Dynamics</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Xiaohui Li, Yuandong Pu, Thunderbolt215215, ashun989, Kaiwen-Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23996) • [📄 arXiv](https://arxiv.org/abs/2602.23996) • [📥 PDF](https://arxiv.org/pdf/2602.23996)

**💻 Code:** [⭐ Code](https://github.com/Kaiwen-Zhu/MIGM-Shortcut)

> Upload arXiv:2602.23996

</details>

<details>
<summary><b>11. Memory Caching: RNNs with Growing Memory</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Meisam Razaviyayn, Peilin Zhong, Yuan Deng, Zeman Li, Ali Behrouz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.24281) • [📄 arXiv](https://arxiv.org/abs/2602.24281) • [📥 PDF](https://arxiv.org/pdf/2602.24281)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API RAM-Net: Expressive Linear Attention with Selectively Addressable Memory (2...

</details>

<details>
<summary><b>12. Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23898) • [📄 arXiv](https://arxiv.org/abs/2602.23898) • [📥 PDF](https://arxiv.org/pdf/2602.23898)

**💻 Code:** [⭐ Code](https://github.com/dddraxxx/Ref-Adv)

> Ref-Adv: a modern referring expression comprehension benchmark that suppresses shortcuts in standard benchmarks by pairing complex expressions with hard visual distractors. We release Ref-Adv-s (1,142 cases) with evaluation code and prediction fil...

</details>

<details>
<summary><b>13. LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20913) • [📄 arXiv](https://arxiv.org/abs/2602.20913) • [📥 PDF](https://arxiv.org/pdf/2602.20913)

**💻 Code:** [⭐ Code](https://github.com/qiujihao19/LongVideo-R1)

> Github: https://github.com/qiujihao19/LongVideo-R1 Model: https://huggingface.co/ChurchillQAQ/LongVideo-R1-Qwen2.5 Dataset: https://huggingface.co/datasets/ChurchillQAQ/LongVideo-R1-Data

</details>

<details>
<summary><b>14. SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Alexandre Alahi, Yasaman Haghighi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.24208) • [📄 arXiv](https://arxiv.org/abs/2602.24208) • [📥 PDF](https://arxiv.org/pdf/2602.24208)

**💻 Code:** [⭐ Code](https://github.com/vita-epfl/SenCache)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API AdaCorrection: Adaptive Offset Cache Correction for Accurate Diffusion Tran...

</details>

<details>
<summary><b>15. Shared Nature, Unique Nurture: PRISM for Pluralistic Reasoning via In-context Structure Modeling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21317) • [📄 arXiv](https://arxiv.org/abs/2602.21317) • [📥 PDF](https://arxiv.org/pdf/2602.21317)

> Moving beyond shared pre-training 'Nature', PRISM injects unique cognitive 'Nurture' into LLMs via dynamic inference-time epistemic graphs, instantly unlocking pluralistic reasoning and state-of-the-art novelty and discovery on multiple benchmarks...

</details>

<details>
<summary><b>16. CL4SE: A Context Learning Benchmark For Software Engineering Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23047) • [📄 arXiv](https://arxiv.org/abs/2602.23047) • [📥 PDF](https://arxiv.org/pdf/2602.23047)

**💻 Code:** [⭐ Code](https://github.com/Tomsawyerhu/CodeCL)

> Context Learning Benchmark for Software Engineering Tasks

</details>

<details>
<summary><b>17. Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lukasz Heldt, Ruining He, Yueqi Wang, Isay Katsman, Zhengyang Su

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22647) • [📄 arXiv](https://arxiv.org/abs/2602.22647) • [📥 PDF](https://arxiv.org/pdf/2602.22647)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Beyond GEMM-Centric NPUs: Enabling Efficient Diffusion LLM Sampling (2026) ...

</details>

<details>
<summary><b>18. How to Take a Memorable Picture? Empowering Users with Actionable Feedback</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Elisa Ricci, Jacopo Staiano, Davide Talon, laitifranz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21877) • [📄 arXiv](https://arxiv.org/abs/2602.21877) • [📥 PDF](https://arxiv.org/pdf/2602.21877)

**💻 Code:** [⭐ Code](https://github.com/laitifranz/MemCoach)

> 📸 🧠 [CVPR 2026] MemCoach: Actionable Memorability Feedback via MLLM Steering MemCoach is a framework designed to bridge the gap between image memorability scoring and practical image improvement. Rather than providing a simple numerical score, it ...

</details>

<details>
<summary><b>19. DUET-VLM: Dual stage Unified Efficient Token reduction for VLM Training and Inference</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18846) • [📄 arXiv](https://arxiv.org/abs/2602.18846) • [📥 PDF](https://arxiv.org/pdf/2602.18846)

**💻 Code:** [⭐ Code](https://github.com/AMD-AGI/DUET-VLM)

> No abstract available.

</details>

<details>
<summary><b>20. DLEBench: Evaluating Small-scale Object Editing Ability for Instruction-based Image Editing Model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** FengJiao Chen, Wei Wang, Jun Kuang, Boxian Ai, Shibo Hong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.23622) • [📄 arXiv](https://arxiv.org/abs/2602.23622) • [📥 PDF](https://arxiv.org/pdf/2602.23622)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API How Well Do Models Follow Visual Instructions? VIBE: A Systematic Benchmark...

</details>

<details>
<summary><b>21. Reinforcement-aware Knowledge Distillation for LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22495) • [📄 arXiv](https://arxiv.org/abs/2602.22495) • [📥 PDF](https://arxiv.org/pdf/2602.22495)

> Reinforcement learning (RL) post-training has recently driven major gains in long chain-of-thought reasoning large language models (LLMs), but the high inference cost of such models motivates distillation into smaller students. Most existing knowl...

</details>

<details>
<summary><b>22. Cognitive Models and AI Algorithms Provide Templates for Designing Language Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22523) • [📄 arXiv](https://arxiv.org/abs/2602.22523) • [📥 PDF](https://arxiv.org/pdf/2602.22523)

> Contemporary LLMs are increasingly capable in isolation, but there are still many difficult problems that lie beyond the abilities of a single LLM. For such tasks, there is still uncertainty about how best to take many LLMs as parts and combine th...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-03-03.json`](data/daily/2026-03-03.json) | 22 |
| 📆 This Week | [`2026-W09.json`](data/weekly/2026-W09.json) | 50 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 78 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-03 | 22 | [View JSON](data/daily/2026-03-03.json) |
| 📄 2026-03-02 | 28 | [View JSON](data/daily/2026-03-02.json) |
| 📄 2026-03-01 | 28 | [View JSON](data/daily/2026-03-01.json) |
| 📄 2026-02-28 | 28 | [View JSON](data/daily/2026-02-28.json) |
| 📄 2026-02-27 | 30 | [View JSON](data/daily/2026-02-27.json) |
| 📄 2026-02-26 | 32 | [View JSON](data/daily/2026-02-26.json) |
| 📄 2026-02-25 | 25 | [View JSON](data/daily/2026-02-25.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W09 | 50 | [View JSON](data/weekly/2026-W09.json) |
| 📅 2026-W08 | 184 | [View JSON](data/weekly/2026-W08.json) |
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 78 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |
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
