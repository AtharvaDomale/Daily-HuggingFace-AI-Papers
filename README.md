<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1395+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">18</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">45</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">657</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1395+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 27, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. LongCat-Flash-Thinking-2601 Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16725) • [📄 arXiv](https://arxiv.org/abs/2601.16725) • [📥 PDF](https://arxiv.org/pdf/2601.16725)

> this is informative.

</details>

<details>
<summary><b>2. SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents</b> ⭐ 35</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16746) • [📄 arXiv](https://arxiv.org/abs/2601.16746) • [📥 PDF](https://arxiv.org/pdf/2601.16746)

**💻 Code:** [⭐ Code](https://github.com/Ayanami1314/swe-pruner)

> wc，nb

</details>

<details>
<summary><b>3. TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14133) • [📄 arXiv](https://arxiv.org/abs/2601.14133) • [📥 PDF](https://arxiv.org/pdf/2601.14133)

**💻 Code:** [⭐ Code](https://github.com/ZGC-EmbodyAI/TwinBrainVLA)

> TwinBrainVLA , a novel architecture that coordinates a generalist VLM retaining universal semantic understanding and a specialist VLM dedicated to embodied proprioception for joint robotic control.

</details>

<details>
<summary><b>4. VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16973) • [📄 arXiv](https://arxiv.org/abs/2601.16973) • [📥 PDF](https://arxiv.org/pdf/2601.16973)

**💻 Code:** [⭐ Code](https://github.com/visgym/VIsGym) • [⭐ Code](https://github.com/visgym/VisGym)

> We released VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents. We systematically study the brittleness of vision-language models in multi-step visual interaction, analyze how training choices shape behavior, and open-sourc...

</details>

<details>
<summary><b>5. Memory-V2V: Augmenting Video-to-Video Diffusion Models with Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16296) • [📄 arXiv](https://arxiv.org/abs/2601.16296) • [📥 PDF](https://arxiv.org/pdf/2601.16296)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Plenoptic Video Generation (2026) Spatia: Video Generation with Updatable S...

</details>

<details>
<summary><b>6. Inference-Time Scaling of Verification: Self-Evolving Deep Research Agents via Test-Time Rubric-Guided Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15808) • [📄 arXiv](https://arxiv.org/abs/2601.15808) • [📥 PDF](https://arxiv.org/pdf/2601.15808)

> The scaling law of verification in deep research agent

</details>

<details>
<summary><b>7. Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14243) • [📄 arXiv](https://arxiv.org/abs/2601.14243) • [📥 PDF](https://arxiv.org/pdf/2601.14243)

> This paper analyzes why existing FP8 reinforcement learning methods fail. It proposes Jet-RL, an FP8 RL training framework that enables robust and stable RL optimization by eliminating training-inference mismatch.

</details>

<details>
<summary><b>8. SALAD: Achieve High-Sparsity Attention via Efficient Linear Attention Tuning for Video Diffusion Transformer</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16515) • [📄 arXiv](https://arxiv.org/abs/2601.16515) • [📥 PDF](https://arxiv.org/pdf/2601.16515)

> In this work, we propose SALAD, introducing a lightweight linear attention branch in parallel with the sparse attention. By incorporating an input-dependent gating mechanism to finely balance the two branches, our method attains 90% sparsity and 1...

</details>

<details>
<summary><b>9. GameTalk: Training LLMs for Strategic Conversation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16276) • [📄 arXiv](https://arxiv.org/abs/2601.16276) • [📥 PDF](https://arxiv.org/pdf/2601.16276)

> Strategic decision-making in multi-agent settings is a key challenge for large language models (LLMs), particularly when coordination and negotiation must unfold over extended conversations. While recent work has explored the use of LLMs in isolat...

</details>

<details>
<summary><b>10. MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences</b> ⭐ 18</summary>

<br/>

**👥 Authors:** Jianwen Sun, Yukang Feng, Yibin Wang, Chuanhao Li, Zizhen Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07251) • [📄 arXiv](https://arxiv.org/abs/2601.07251) • [📥 PDF](https://arxiv.org/pdf/2601.07251)

**💻 Code:** [⭐ Code](https://github.com/leroy9472/MeepleLM)

> https://github.com/leroy9472/MeepleLM

</details>

<details>
<summary><b>11. DSGym: A Holistic Framework for Evaluating and Training Data Science Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yongchan Kwon, Federico Bianchi, Harper Hua, Junlin Wang, Fan Nie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16344) • [📄 arXiv](https://arxiv.org/abs/2601.16344) • [📥 PDF](https://arxiv.org/pdf/2601.16344)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API DSAEval: Evaluating Data Science Agents on a Wide Range of Real-World Data ...

</details>

<details>
<summary><b>12. Mecellem Models: Turkish Models Trained from Scratch and Continually Pre-trained for the Legal Domain</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16018) • [📄 arXiv](https://arxiv.org/abs/2601.16018) • [📥 PDF](https://arxiv.org/pdf/2601.16018)

**💻 Code:** [⭐ Code](https://github.com/newmindai/mecellem-models)

> Mecellem Models propose Turkish legal-domain encoders and decoders trained from scratch and via continual pre-training. ModernBERT-based encoders (112.7B tokens) achieve top-3 Turkish retrieval results with high production efficiency, while Qwen3-...

</details>

<details>
<summary><b>13. Endless Terminals: Scaling RL Environments for Terminal Agents</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16443) • [📄 arXiv](https://arxiv.org/abs/2601.16443) • [📥 PDF](https://arxiv.org/pdf/2601.16443)

**💻 Code:** [⭐ Code](https://github.com/kanishkg/endless-terminals)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API From Failure to Mastery: Generating Hard Samples for Tool-use Agents (2026)...

</details>

<details>
<summary><b>14. ChartVerse: Scaling Chart Reasoning via Reliable Programmatic Synthesis from Scratch</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13606) • [📄 arXiv](https://arxiv.org/abs/2601.13606) • [📥 PDF](https://arxiv.org/pdf/2601.13606)

**💻 Code:** [⭐ Code](https://github.com/starriver030515/ChartVerse)

> High-quality synthetic Chart data and strong Chart reasoning model.

</details>

<details>
<summary><b>15. Knowledge is Not Enough: Injecting RL Skills for Continual Adaptation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11258) • [📄 arXiv](https://arxiv.org/abs/2601.11258) • [📥 PDF](https://arxiv.org/pdf/2601.11258)

> Let Your LLMs Use New Knowledge with “PaST” Skills Paper: https://arxiv.org/abs/2601.11258 Blog: https://past-blog.notion.site

</details>

<details>
<summary><b>16. Dancing in Chains: Strategic Persuasion in Academic Rebuttal via Theory of Mind</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yi R Fung, Zongwei Lyu, Zhitao He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15715) • [📄 arXiv](https://arxiv.org/abs/2601.15715) • [📥 PDF](https://arxiv.org/pdf/2601.15715)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Paper2Rebuttal: A Multi-Agent Framework for Transparent Author Response Ass...

</details>

<details>
<summary><b>17. Guidelines to Prompt Large Language Models for Code Generation: An Empirical Characterization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Gabriele Bavota, Rosalia Tufano, Fiorella Zampetti, Alessandro Midolo, Devy1

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13118) • [📄 arXiv](https://arxiv.org/abs/2601.13118) • [📥 PDF](https://arxiv.org/pdf/2601.13118)

> .

</details>

<details>
<summary><b>18. VISTA-PATH: An interactive foundation model for pathology image segmentation and quantitative analysis in computational pathology</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.16451) • [📄 arXiv](https://arxiv.org/abs/2601.16451) • [📥 PDF](https://arxiv.org/pdf/2601.16451)

> 🚀 VISTA-PATH is introduced as the first interactive segmentation foundation model for pathology. It advances computational pathology workflows by enabling more accurate, interpretable, and human-guided quantitative measurements. Key highlights inc...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-01-27.json`](data/daily/2026-01-27.json) | 18 |
| 📆 This Week | [`2026-W04.json`](data/weekly/2026-W04.json) | 45 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 657 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-27 | 18 | [View JSON](data/daily/2026-01-27.json) |
| 📄 2026-01-26 | 27 | [View JSON](data/daily/2026-01-26.json) |
| 📄 2026-01-25 | 27 | [View JSON](data/daily/2026-01-25.json) |
| 📄 2026-01-24 | 27 | [View JSON](data/daily/2026-01-24.json) |
| 📄 2026-01-23 | 26 | [View JSON](data/daily/2026-01-23.json) |
| 📄 2026-01-22 | 32 | [View JSON](data/daily/2026-01-22.json) |
| 📄 2026-01-21 | 11 | [View JSON](data/daily/2026-01-21.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W04 | 45 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 657 | [View JSON](data/monthly/2026-01.json) |
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
