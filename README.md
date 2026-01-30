<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1474+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">124</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">736</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1474+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 30, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation</b> ⭐ 84</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20614) • [📄 arXiv](https://arxiv.org/abs/2601.20614) • [📥 PDF](https://arxiv.org/pdf/2601.20614)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/MathForge)

> Accepted for ICLR 2026

</details>

<details>
<summary><b>2. Advancing Open-source World Models</b> ⭐ 756</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20540) • [📄 arXiv](https://arxiv.org/abs/2601.20540) • [📥 PDF](https://arxiv.org/pdf/2601.20540)

**💻 Code:** [⭐ Code](https://github.com/Robbyant/lingbot-world/)

> LingBot-World: Advancing Open-source World Models

</details>

<details>
<summary><b>3. Innovator-VL: A Multimodal Large Language Model for Scientific Discovery</b> ⭐ 70</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19325) • [📄 arXiv](https://arxiv.org/abs/2601.19325) • [📥 PDF](https://arxiv.org/pdf/2601.19325)

**💻 Code:** [⭐ Code](https://github.com/InnovatorLM/Innovator-VL)

> Homepage: https://innovatorlm.github.io/Innovator-VL Github: https://github.com/InnovatorLM/Innovator-VL

</details>

<details>
<summary><b>4. DeepSeek-OCR 2: Visual Causal Flow</b> ⭐ 1.56k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20552) • [📄 arXiv](https://arxiv.org/abs/2601.20552) • [📥 PDF](https://arxiv.org/pdf/2601.20552)

**💻 Code:** [⭐ Code](https://github.com/deepseek-ai/DeepSeek-OCR-2)

> Proposes DeepSeek-OCR 2 with a causal, reordering encoder (DeepEncoder V2) to dynamically rearrange visual tokens for LLMs, enabling 2D reasoning via two cascaded 1D causal structures.

</details>

<details>
<summary><b>5. Spark: Strategic Policy-Aware Exploration via Dynamic Branching for Long-Horizon Agentic Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shuai Zhang, Yuhao Shen, Changpeng Yang, Shuo Yang, Jinyang23

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20209) • [📄 arXiv](https://arxiv.org/abs/2601.20209) • [📥 PDF](https://arxiv.org/pdf/2601.20209)

> No abstract available.

</details>

<details>
<summary><b>6. Linear representations in language models can change dramatically over a conversation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20834) • [📄 arXiv](https://arxiv.org/abs/2601.20834) • [📥 PDF](https://arxiv.org/pdf/2601.20834)

> LM representations along linear directions evolve during conversations, with factual content shifting over time; changes depend on context, robust across models, and challenge static interpretability and steering.

</details>

<details>
<summary><b>7. Reinforcement Learning via Self-Distillation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20802) • [📄 arXiv](https://arxiv.org/abs/2601.20802) • [📥 PDF](https://arxiv.org/pdf/2601.20802)

**💻 Code:** [⭐ Code](https://github.com/lasgroup/SDPO)

> We introduce Self-Distillation Policy Optimization (SDPO), a method for online RL that leverages the model's own ability to interpret rich feedback to drastically speed up training and boost reasoning capabilities on hard tasks.

</details>

<details>
<summary><b>8. SERA: Soft-Verified Efficient Repository Agents</b> ⭐ 75</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20789) • [📄 arXiv](https://arxiv.org/abs/2601.20789) • [📥 PDF](https://arxiv.org/pdf/2601.20789)

**💻 Code:** [⭐ Code](https://github.com/allenai/SERA)

> Introduces SERA for efficient open-weight coding agents trained via supervised finetuning; SVG generates synthetic trajectories to cheaply specialize agents to private codebases with strong open-model performance.

</details>

<details>
<summary><b>9. VERGE: Formal Refinement and Guidance Engine for Verifiable LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20055) • [📄 arXiv](https://arxiv.org/abs/2601.20055) • [📥 PDF](https://arxiv.org/pdf/2601.20055)

> In this paper, we introduce VERGE, a neuro-symbolic framework designed to bridge the semantic gap between LLM fluency and formal correctness by addressing the "all-or-nothing" limitation of traditional solvers. A key innovation is Semantic Routing...

</details>

<details>
<summary><b>10. OmegaUse: Building a General-Purpose GUI Agent for Autonomous Task Execution</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yusai Zhao, Jingjia Cao, Xinjiang Lu, Yixiong Xiao, Le Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20380) • [📄 arXiv](https://arxiv.org/abs/2601.20380) • [📥 PDF](https://arxiv.org/pdf/2601.20380)

> OmegaUse is a general GUI agent enabling autonomous desktop and mobile task execution, trained with curated data, SFT and GRPO, MoE backbone, and evaluated on OS-Nav benchmarks.

</details>

<details>
<summary><b>11. Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19280) • [📄 arXiv](https://arxiv.org/abs/2601.19280) • [📥 PDF](https://arxiv.org/pdf/2601.19280)

> Beyond Uniform-data LLM Reasoning: We propose an optimization-first framework, based on Group Distributionally Robust Optimization (GDRO), that moves beyond uniform reasoning models by dynamically adapting the training distribution.

</details>

<details>
<summary><b>12. RIR-Mega-Speech: A Reverberant Speech Corpus with Comprehensive Acoustic Metadata and Reproducible Evaluation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** mandipgoswami

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19949) • [📄 arXiv](https://arxiv.org/abs/2601.19949) • [📥 PDF](https://arxiv.org/pdf/2601.19949)

**💻 Code:** [⭐ Code](https://github.com/mandip42/rir-mega-speech)

> Despite decades of research on reverberant speech, comparing methods remains difficult because most corpora lack per-file acoustic annotations or provide limited documentation for reproduction. We present RIR-Mega-Speech, a corpus of approximately...

</details>

<details>
<summary><b>13. UPLiFT: Efficient Pixel-Dense Feature Upsampling with Local Attenders</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.17950) • [📄 arXiv](https://arxiv.org/abs/2601.17950) • [📥 PDF](https://arxiv.org/pdf/2601.17950)

**💻 Code:** [⭐ Code](https://github.com/mwalmer-umd/UPLiFT/)

> Code: https://github.com/mwalmer-umd/UPLiFT/

</details>

<details>
<summary><b>14. Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20829) • [📄 arXiv](https://arxiv.org/abs/2601.20829) • [📥 PDF](https://arxiv.org/pdf/2601.20829)

**💻 Code:** [⭐ Code](https://github.com/minwukim/training-on-saturated-problems)

> TL;DR: We enable continued RL training on saturated reasoning tasks by conditioning on rare failure prefixes.

</details>

<details>
<summary><b>15. GDCNet: Generative Discrepancy Comparison Network for Multimodal Sarcasm Detection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20618) • [📄 arXiv](https://arxiv.org/abs/2601.20618) • [📥 PDF](https://arxiv.org/pdf/2601.20618)

> Existing multimodal sarcasm detection methods struggle with loosely related image-text pairs and noisy LLM-generated cues. GDCNet addresses this by using MLLM-generated factual captions as semantic anchors to compute semantic and sentiment discrep...

</details>

<details>
<summary><b>16. How AI Impacts Skill Formation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20245) • [📄 arXiv](https://arxiv.org/abs/2601.20245) • [📥 PDF](https://arxiv.org/pdf/2601.20245)

> https://www.anthropic.com/research/AI-assistance-coding-skills

</details>

<details>
<summary><b>17. SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.19194) • [📄 arXiv](https://arxiv.org/abs/2601.19194) • [📥 PDF](https://arxiv.org/pdf/2601.19194)

> Accepted at ICASSP 2026

</details>

<details>
<summary><b>18. FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18150) • [📄 arXiv](https://arxiv.org/abs/2601.18150) • [📥 PDF](https://arxiv.org/pdf/2601.18150)

> We have developed FP8 rollout features for the two frameworks, verl and NeMo-RL. In this report, we introduce the implementation solutions and a series of validation experiments conducted (covering both Dense and MoE models), with analyses perform...

</details>

<details>
<summary><b>19. Persona Prompting as a Lens on LLM Social Reasoning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20757) • [📄 arXiv](https://arxiv.org/abs/2601.20757) • [📥 PDF](https://arxiv.org/pdf/2601.20757)

**💻 Code:** [⭐ Code](https://github.com/jingyng/PP-social-reasoning)

> Persona prompting can improve classification in socially-sensitive tasks, but it often comes at the cost of rationale quality and fails to mitigate underlying biases, urging caution in its application.

</details>

<details>
<summary><b>20. SketchDynamics: Exploring Free-Form Sketches for Dynamic Intent Expression in Animation Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hongbo Fu, Zeyu Wang, Lin-Ping Yuan, Boyu Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20622) • [📄 arXiv](https://arxiv.org/abs/2601.20622) • [📥 PDF](https://arxiv.org/pdf/2601.20622)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API SketchPlay: Intuitive Creation of Physically Realistic VR Content with Gest...

</details>

<details>
<summary><b>21. Shallow-π: Knowledge Distillation for Flow-based VLAs</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20262) • [📄 arXiv](https://arxiv.org/abs/2601.20262) • [📥 PDF](https://arxiv.org/pdf/2601.20262)

**💻 Code:** [⭐ Code](https://github.com/icsl-Jeon/openpi)

> https://icsl-jeon.github.io/shallow-pi/

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2026-01-30.json`](data/daily/2026-01-30.json) | 21 |
| 📆 This Week | [`2026-W04.json`](data/weekly/2026-W04.json) | 124 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 736 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-30 | 21 | [View JSON](data/daily/2026-01-30.json) |
| 📄 2026-01-29 | 21 | [View JSON](data/daily/2026-01-29.json) |
| 📄 2026-01-28 | 37 | [View JSON](data/daily/2026-01-28.json) |
| 📄 2026-01-27 | 18 | [View JSON](data/daily/2026-01-27.json) |
| 📄 2026-01-26 | 27 | [View JSON](data/daily/2026-01-26.json) |
| 📄 2026-01-25 | 27 | [View JSON](data/daily/2026-01-25.json) |
| 📄 2026-01-24 | 27 | [View JSON](data/daily/2026-01-24.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W04 | 124 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 736 | [View JSON](data/monthly/2026-01.json) |
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
