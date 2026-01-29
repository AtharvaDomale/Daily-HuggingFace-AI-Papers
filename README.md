<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1453+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">21</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">103</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">715</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1453+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 29, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security</b> ⭐ 178</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18491) • [📄 arXiv](https://arxiv.org/abs/2601.18491) • [📥 PDF](https://arxiv.org/pdf/2601.18491)

**💻 Code:** [⭐ Code](https://github.com/AI45Lab/AgentDoG)

> AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security

</details>

<details>
<summary><b>2. AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning</b> ⭐ 44</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18631) • [📄 arXiv](https://arxiv.org/abs/2601.18631) • [📥 PDF](https://arxiv.org/pdf/2601.18631)

**💻 Code:** [⭐ Code](https://github.com/ssmisya/AdaReasoner)

> For more information, please visit our homepage: https://adareasoner.github.io

</details>

<details>
<summary><b>3. A Pragmatic VLA Foundation Model</b> ⭐ 245</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18692) • [📄 arXiv](https://arxiv.org/abs/2601.18692) • [📥 PDF](https://arxiv.org/pdf/2601.18692)

**💻 Code:** [⭐ Code](https://github.com/robbyant/lingbot-vla)

> A Pragmatic VLA Foundation Model

</details>

<details>
<summary><b>4. Visual Generation Unlocks Human-Like Reasoning through Multimodal World Models</b> ⭐ 36</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19834) • [📄 arXiv](https://arxiv.org/abs/2601.19834) • [📥 PDF](https://arxiv.org/pdf/2601.19834)

**💻 Code:** [⭐ Code](https://github.com/thuml/Reasoning-Visual-World)

> TL;DR : From a world-model perspective, we study when and how visual generation enabled by unified multimodal models (UMMs) benefits reasoning. Humans construct mental models of the world, representing information and knowledge through two complem...

</details>

<details>
<summary><b>5. Youtu-VL: Unleashing Visual Potential via Unified Vision-Language Supervision</b> ⭐ 35</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19798) • [📄 arXiv](https://arxiv.org/abs/2601.19798) • [📥 PDF](https://arxiv.org/pdf/2601.19798)

**💻 Code:** [⭐ Code](https://github.com/TencentCloudADP/youtu-vl)

> Performs on par with Qwen3-VL-8B-Instruct on visual based tasks despite being half the size.

</details>

<details>
<summary><b>6. AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17645) • [📄 arXiv](https://arxiv.org/abs/2601.17645) • [📥 PDF](https://arxiv.org/pdf/2601.17645)

> Demo: https://avmemeexam.github.io/public Dataset: https://huggingface.co/datasets/naplab/AVMeme-Exam

</details>

<details>
<summary><b>7. World Craft: Agentic Framework to Create Visualizable Worlds via Text</b> ⭐ 40</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09150) • [📄 arXiv](https://arxiv.org/abs/2601.09150) • [📥 PDF](https://arxiv.org/pdf/2601.09150)

**💻 Code:** [⭐ Code](https://github.com/HerzogFL/World-Craft)

> https://github.com/HerzogFL/World-Craft Large Language Models (LLMs) motivate generative agent simulation (e.g., AI Town) to create a "dynamic world'', holding immense value across entertainment and research. However, for non-experts, especially t...

</details>

<details>
<summary><b>8. Post-LayerNorm Is Back: Stable, ExpressivE, and Deep</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19895) • [📄 arXiv](https://arxiv.org/abs/2601.19895) • [📥 PDF](https://arxiv.org/pdf/2601.19895)

> 🚀 Only a few lines of code changed, and we pushed deep LLMs to the next level. 📈 With Keel, we scaled LLM to 1000 layers. And the deeper we go, the more Keel pulls ahead of standard Pre-LN Transformers.

</details>

<details>
<summary><b>9. TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18292) • [📄 arXiv](https://arxiv.org/abs/2601.18292) • [📥 PDF](https://arxiv.org/pdf/2601.18292)

> TriPlay-RL: Solving the Safety vs. Reasoning Trade-off with 3-Way Self-Play Really interesting take on automated safety alignment. We've seen plenty of "Red Team vs. Blue Team" setups, but they often suffer from two issues: the Red Team eventually...

</details>

<details>
<summary><b>10. FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval for Multi-Document Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18116) • [📄 arXiv](https://arxiv.org/abs/2601.18116) • [📥 PDF](https://arxiv.org/pdf/2601.18116)

> With the rise of 1M+ context windows in Gemini and Claude, the biggest debate in AI right now is: "Do we still need RAG, or should we just dump everything into the prompt?" Today's pick, FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval ...

</details>

<details>
<summary><b>11. Towards Pixel-Level VLM Perception via Simple Points Prediction</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19228) • [📄 arXiv](https://arxiv.org/abs/2601.19228) • [📥 PDF](https://arxiv.org/pdf/2601.19228)

**💻 Code:** [⭐ Code](https://github.com/songtianhui/SimpleSeg)

> Project Page: https://simpleseg.github.io/ Github: https://github.com/songtianhui/SimpleSeg HuggingFace: https://huggingface.co/collections/sthui/simpleseg

</details>

<details>
<summary><b>12. Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19375) • [📄 arXiv](https://arxiv.org/abs/2601.19375) • [📥 PDF](https://arxiv.org/pdf/2601.19375)

**💻 Code:** [⭐ Code](https://github.com/knoveleng/steering)

> We introduce Selective Steering, a principled norm-preserving activation steering method that enables stable, continuous control of LLM behavior while significantly improving adversarial attack effectiveness without sacrificing model capabilities....

</details>

<details>
<summary><b>13. Revisiting Parameter Server in LLM Post-Training</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19362) • [📄 arXiv](https://arxiv.org/abs/2601.19362) • [📥 PDF](https://arxiv.org/pdf/2601.19362)

**💻 Code:** [⭐ Code](https://github.com/sail-sg/odc)

> Modern data parallel (DP) training favors collective communication over parameter servers (PS) for its simplicity and efficiency under balanced workloads. However, the balanced workload assumption no longer holds in large language model (LLM) post...

</details>

<details>
<summary><b>14. HalluCitation Matters: Revealing the Impact of Hallucinated References with 300 Hallucinated Papers in ACL Conferences</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Taro Watanabe, Hidetaka Kamigaito, Yusuke Sakai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18724) • [📄 arXiv](https://arxiv.org/abs/2601.18724) • [📥 PDF](https://arxiv.org/pdf/2601.18724)

> arXivlens breakdown of this paper 👉 https://arxivlens.com/PaperView/Details/hallucitation-matters-revealing-the-impact-of-hallucinated-references-with-300-hallucinated-papers-in-acl-conferences-2856-e431efdf Executive Summary Detailed Breakdown Pr...

</details>

<details>
<summary><b>15. HyperAlign: Hypernetwork for Efficient Test-Time Alignment of Diffusion Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiaxian Guo, Xin Xie, dginf

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15968) • [📄 arXiv](https://arxiv.org/abs/2601.15968) • [📥 PDF](https://arxiv.org/pdf/2601.15968)

> HyperAlign: Hypernetwork for Efficient Test-Time Alignment of Diffusion Models

</details>

<details>
<summary><b>16. Self-Distillation Enables Continual Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19897) • [📄 arXiv](https://arxiv.org/abs/2601.19897) • [📥 PDF](https://arxiv.org/pdf/2601.19897)

> Check out our website and code: www.idanshenfeld.com/SDFT

</details>

<details>
<summary><b>17. Benchmarks Saturate When The Model Gets Smarter Than The Judge</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vincent Ginis, Brecht Verbeken, Andres Algaba, martheballon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19532) • [📄 arXiv](https://arxiv.org/abs/2601.19532) • [📥 PDF](https://arxiv.org/pdf/2601.19532)

> No abstract available.

</details>

<details>
<summary><b>18. GPCR-Filter: a deep learning framework for efficient and precise GPCR modulator discovery</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19149) • [📄 arXiv](https://arxiv.org/abs/2601.19149) • [📥 PDF](https://arxiv.org/pdf/2601.19149)

> GPCR-Filter is a compound–protein interaction model that couples ESM-3 GPCR sequence embeddings with ligand graph representations through attention-based feature interaction, trained on 90k+ curated GPCR–ligand pairs. It shows stronger OOD general...

</details>

<details>
<summary><b>19. DeFM: Learning Foundation Representations from Depth for Robotics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18923) • [📄 arXiv](https://arxiv.org/abs/2601.18923) • [📥 PDF](https://arxiv.org/pdf/2601.18923)

> DeFM (Depth Foundation Model) is a vision backbone trained on 60M depth images via self-distillation. It is engineered for robotic perception, providing metric-aware representations that excel in sim-to-real transfer and cross-sensor generalizatio...

</details>

<details>
<summary><b>20. EvolVE: Evolutionary Search for LLM-based Verilog Generation and Optimization</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18067) • [📄 arXiv](https://arxiv.org/abs/2601.18067) • [📥 PDF](https://arxiv.org/pdf/2601.18067)

**💻 Code:** [⭐ Code](https://github.com/weiber2002/ICRTL)

> Verilog’s design cycle is inherently labor-intensive and necessitates extensive domain expertise. Although Large Language Models (LLMs) offer a promising pathway toward automation, their limited training data and intrinsic sequential reasoning fai...

</details>

<details>
<summary><b>21. CooperBench: Why Coding Agents Cannot be Your Teammates Yet</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13295) • [📄 arXiv](https://arxiv.org/abs/2601.13295) • [📥 PDF](https://arxiv.org/pdf/2601.13295)

**💻 Code:** [⭐ Code](https://github.com/cooperbench/CooperBench)

> Resolving team conflicts requires not only task-specific competence, but also social intelligence to find common ground and build consensus. As AI agents increasingly collaborate on complex work, they must develop coordination capabilities to func...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2026-01-29.json`](data/daily/2026-01-29.json) | 21 |
| 📆 This Week | [`2026-W04.json`](data/weekly/2026-W04.json) | 103 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 715 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-29 | 21 | [View JSON](data/daily/2026-01-29.json) |
| 📄 2026-01-28 | 37 | [View JSON](data/daily/2026-01-28.json) |
| 📄 2026-01-27 | 18 | [View JSON](data/daily/2026-01-27.json) |
| 📄 2026-01-26 | 27 | [View JSON](data/daily/2026-01-26.json) |
| 📄 2026-01-25 | 27 | [View JSON](data/daily/2026-01-25.json) |
| 📄 2026-01-24 | 27 | [View JSON](data/daily/2026-01-24.json) |
| 📄 2026-01-23 | 26 | [View JSON](data/daily/2026-01-23.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W04 | 103 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 715 | [View JSON](data/monthly/2026-01.json) |
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
