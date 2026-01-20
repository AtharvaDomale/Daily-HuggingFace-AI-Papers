<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1227+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">60</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">489</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1227+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 20, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Your Group-Relative Advantage Is Biased</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaohan Wang, Yikunb, PandaChai, chenzherui007, ShortCatisLong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08521) • [📄 arXiv](https://arxiv.org/abs/2601.08521) • [📥 PDF](https://arxiv.org/pdf/2601.08521)

> This paper fundamentally shows that: "The commonly used group-relative advantage estimator is inherently biased except at p_t = 0.5: it systematically underestimates true advantage on hard prompts and overestimates true advantag on easy prompts". ...

</details>

<details>
<summary><b>2. The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11496) • [📄 arXiv](https://arxiv.org/abs/2601.11496) • [📥 PDF](https://arxiv.org/pdf/2601.11496)

> Imagine a company introducing a shiny new technology 🍎. Not to use it, but to force a regulator to rewrite the rules. Once the rules change? The apple is discarded. The technology is never used. But the strategic shift is complete: the manipulator...

</details>

<details>
<summary><b>3. Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10355) • [📄 arXiv](https://arxiv.org/abs/2601.10355) • [📥 PDF](https://arxiv.org/pdf/2601.10355)

> We propose a novel "Text to Trajectory" paradigm to address the scarcity of multi-turn tool usage trajectory data needed to train agents. Traditional methods rely on predefined API sets to synthesize data, but this approach is limited by the scope...

</details>

<details>
<summary><b>4. RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation</b> ⭐ 25</summary>

<br/>

**👥 Authors:** Jiale Zhao, Sunzhu Li, kaikezhang, liushunyu, renhuimin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08430) • [📄 arXiv](https://arxiv.org/abs/2601.08430) • [📥 PDF](https://arxiv.org/pdf/2601.08430)

**💻 Code:** [⭐ Code](https://github.com/teqkilla/RubricHub)

> We introduce RubricHub, a large-scale (~110k) and multi-domain rubric dataset constructed via an automated Coarse-to-Fine Rubric Generation framework. By synergizing principle-guided synthesis, multi-model aggregation, and difficulty evolution, ou...

</details>

<details>
<summary><b>5. When Personalization Misleads: Understanding and Mitigating Hallucinations in Personalized LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11000) • [📄 arXiv](https://arxiv.org/abs/2601.11000) • [📥 PDF](https://arxiv.org/pdf/2601.11000)

> 💡 Overview Personalization is increasingly adopted in modern LLM systems, but we find it can systematically distort factual reasoning. We identify personalization-induced hallucinations, where models generate answers aligned with user history rath...

</details>

<details>
<summary><b>6. ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11404) • [📄 arXiv](https://arxiv.org/abs/2601.11404) • [📥 PDF](https://arxiv.org/pdf/2601.11404)

> abs: Vision-Language-Action (VLA) models have emerged as essential generalist robot policies for diverse manipulation tasks, conventionally relying on directly translating multimodal inputs into actions via Vision-Language Model (VLM) embeddings. ...

</details>

<details>
<summary><b>7. BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Yunbo Tang, bitwjg, Elliott, yongjing, ShiyuLiu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11037) • [📄 arXiv](https://arxiv.org/abs/2601.11037) • [📥 PDF](https://arxiv.org/pdf/2601.11037)

**💻 Code:** [⭐ Code](https://github.com/Liushiyu-0709/BAPO-Reliable-Search)

> Boundary-Aware Policy Optimization（BAPO） is a novel reinforcement learning-based framework for training reliable agentic search models. Beyond correctness rewards, BAPO incorporates boundary-aware rewards to encourage appropriate "I Don't Know" (I...

</details>

<details>
<summary><b>8. FrankenMotion: Part-level Human Motion Generation and Composition</b> ⭐ 37</summary>

<br/>

**👥 Authors:** Gerard Pons-Moll, andreas-geiger, Yongcao, xianghuix, coralli

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10909) • [📄 arXiv](https://arxiv.org/abs/2601.10909) • [📥 PDF](https://arxiv.org/pdf/2601.10909)

**💻 Code:** [⭐ Code](https://github.com/Coral79/FrankenMotion-Code)

> TL;DR: We introduce the first framework for atomic, part-level motion control, powered by our new hierarchical Frankenstein dataset (39h) constructed via LLMs.

</details>

<details>
<summary><b>9. Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09001) • [📄 arXiv](https://arxiv.org/abs/2601.09001) • [📥 PDF](https://arxiv.org/pdf/2601.09001)

> A first exploration of a lightweight, inference-time method for monitoring LLM accuracy under domain drift using output-entropy traces derived from next-token probabilities. This approach demonstrates promising results for slice-level accuracy est...

</details>

<details>
<summary><b>10. ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09195) • [📄 arXiv](https://arxiv.org/abs/2601.09195) • [📥 PDF](https://arxiv.org/pdf/2601.09195)

**💻 Code:** [⭐ Code](https://github.com/Utaotao/ProFit)

> Supervised fine-tuning (SFT) is a fundamental post-training strategy to align Large Language Models (LLMs) with human intent. However, traditional SFT often ignores the one-to-many nature of language by forcing alignment with a single reference an...

</details>

<details>
<summary><b>11. Future Optical Flow Prediction Improves Robot Control & Video Generation</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10781) • [📄 arXiv](https://arxiv.org/abs/2601.10781) • [📥 PDF](https://arxiv.org/pdf/2601.10781)

**💻 Code:** [⭐ Code](https://github.com/SalesforceAIResearch/FOFPred)

> We introduce FOFPred, a language-driven future optical flow prediction framework that enables improved robot control and video generation. Instead of reacting to motion, FOFPred predicts how motion will evolve, conditioned on natural language. 🌐 P...

</details>

<details>
<summary><b>12. ShapeR: Robust Conditional 3D Shape Generation from Casual Captures</b> ⭐ 175</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11514) • [📄 arXiv](https://arxiv.org/abs/2601.11514) • [📥 PDF](https://arxiv.org/pdf/2601.11514)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/ShapeR)

> Project Page | Paper | Video | HF-Model | HF Evaluation Dataset

</details>

<details>
<summary><b>13. Reasoning Models Generate Societies of Thought</b> ⭐ 0</summary>

<br/>

**👥 Authors:** James Evans, Blaise Agüera y Arcas, ninoscherrer, ShiYangLAI, junsol

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10825) • [📄 arXiv](https://arxiv.org/abs/2601.10825) • [📥 PDF](https://arxiv.org/pdf/2601.10825)

> Reasoning models gain accuracy via internal multi-agent-like debates among diverse perspectives, enabling broader exploration of solutions and improved reasoning than single-agent baselines.

</details>

<details>
<summary><b>14. PhysRVG: Physics-Aware Unified Reinforcement Learning for Video Generative Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zheng Zhang, Qiyuan Zhang, shen12313, Shuaishuai0219, BiaoGong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11087) • [📄 arXiv](https://arxiv.org/abs/2601.11087) • [📥 PDF](https://arxiv.org/pdf/2601.11087)

> PhysRVG: Physics-Aware Unified Reinforcement Learning for Video Generative Models

</details>

<details>
<summary><b>15. PersonalAlign: Hierarchical Implicit Intent Alignment for Personalized GUI Agent with Long-Term User-Centric Records</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Liqiang Nie, Weili Guan, Rui Shao, cgwfeel, user0102

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09636) • [📄 arXiv](https://arxiv.org/abs/2601.09636) • [📥 PDF](https://arxiv.org/pdf/2601.09636)

> good

</details>

<details>
<summary><b>16. Building Production-Ready Probes For Gemini</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rohin Shah, Zheng Wang, Joshua Engels, bilalchughtai, jkramar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11516) • [📄 arXiv](https://arxiv.org/abs/2601.11516) • [📥 PDF](https://arxiv.org/pdf/2601.11516)

> Proposes long-context robust probes for Gemini misuse mitigation, showing architecture and diverse-training distribution requirements for generalization, and demonstrates efficient pairing with prompted classifiers and automated probe architecture...

</details>

<details>
<summary><b>17. AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11044) • [📄 arXiv](https://arxiv.org/abs/2601.11044) • [📥 PDF](https://arxiv.org/pdf/2601.11044)

**💻 Code:** [⭐ Code](https://github.com/GAIR-NLP/AgencyBench)

> Some of the observations founded are :- i. Long-horizon tasks remain challenging : Even frontier models struggle with sustained reasoning over real world tasks that require 1M tokens and 90 tool calls, indicating limits in long context autonomy. i...

</details>

<details>
<summary><b>18. More Images, More Problems? A Controlled Analysis of VLM Failure Modes</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07812) • [📄 arXiv](https://arxiv.org/abs/2601.07812) • [📥 PDF](https://arxiv.org/pdf/2601.07812)

**💻 Code:** [⭐ Code](https://github.com/anurag-198/MIMIC)

> Large Vision Language Models (LVLMs) have demonstrated remarkable capabilities, yet their proficiency in understanding and reasoning over multiple images remains largely unexplored. While existing benchmarks have initiated the evaluation of multi-...

</details>

<details>
<summary><b>19. AstroReason-Bench: Evaluating Unified Agentic Planning across Heterogeneous Space Planning Problems</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Jingjing Gong, Weiyi Wang, xpqiu, xjhuang, dalstonchen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11354) • [📄 arXiv](https://arxiv.org/abs/2601.11354) • [📥 PDF](https://arxiv.org/pdf/2601.11354)

**💻 Code:** [⭐ Code](https://github.com/Mtrya/astro-reason)

> Introduces AstroReason-Bench, a benchmark for evaluating unified agentic planning in space planning problems with physics constraints, heterogeneous objectives, and long-horizon decisions.

</details>

<details>
<summary><b>20. Language of Thought Shapes Output Diversity in Large Language Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11227) • [📄 arXiv](https://arxiv.org/abs/2601.11227) • [📥 PDF](https://arxiv.org/pdf/2601.11227)

**💻 Code:** [⭐ Code](https://github.com/iNLP-Lab/Multilingual-LoT-Diversity)

> This paper reveals that controlling the language used during model thinking—the language of thought—provides a novel and structural source of output diversity.

</details>

<details>
<summary><b>21. What Matters in Data Curation for Multimodal Reasoning? Insights from the DCVLR Challenge</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vikas Kumar, Pavel Bushuyeu, Boris Sobolev, Michael Buriek, Yosub Shin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10922) • [📄 arXiv](https://arxiv.org/abs/2601.10922) • [📥 PDF](https://arxiv.org/pdf/2601.10922)

> Some of the observations founded are : i. Difficulty based example selection is the dominant driver of performance: Selecting challenging but learnable examples yields the largest gains in multimodal reasoning accuracy, outperforming other curatio...

</details>

<details>
<summary><b>22. PhyRPR: Training-Free Physics-Constrained Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Boxi Wu, Xiaofei He, Hengjia Li, zjuyb

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09255) • [📄 arXiv](https://arxiv.org/abs/2601.09255) • [📥 PDF](https://arxiv.org/pdf/2601.09255)

> Recent diffusion-based video generation models can synthesize visually plausible videos, yet they often struggle to satisfy physical constraints. A key reason is that most existing approaches remain single-stage: they entangle high-level physical ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-01-20.json`](data/daily/2026-01-20.json) | 22 |
| 📆 This Week | [`2026-W03.json`](data/weekly/2026-W03.json) | 60 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 489 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-20 | 22 | [View JSON](data/daily/2026-01-20.json) |
| 📄 2026-01-19 | 38 | [View JSON](data/daily/2026-01-19.json) |
| 📄 2026-01-18 | 38 | [View JSON](data/daily/2026-01-18.json) |
| 📄 2026-01-17 | 38 | [View JSON](data/daily/2026-01-17.json) |
| 📄 2026-01-16 | 27 | [View JSON](data/daily/2026-01-16.json) |
| 📄 2026-01-15 | 24 | [View JSON](data/daily/2026-01-15.json) |
| 📄 2026-01-14 | 42 | [View JSON](data/daily/2026-01-14.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W03 | 60 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 489 | [View JSON](data/monthly/2026-01.json) |
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
