<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-19-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7647+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">29</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">1039</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7647+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 27, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Trajectory2Gym: Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33568) • [📄 arXiv](https://arxiv.org/abs/2610.33568) • [📥 PDF](https://arxiv.org/pdf/2610.33568)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yihaohu0118/Trajectory2Gym)

> Trajectory2Gym is a framework for scaling tool-use agents through compositional coverage and decision-grounded supervision. It constructs executable interactions that capture multi-step dependencies and trains agents on decisions supported by thei...

</details>

<details>
<summary><b>2. StreamVibe: Proactive Video Agents for Evolving User Intent</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.34662) • [📄 arXiv](https://arxiv.org/abs/2610.34662) • [📥 PDF](https://arxiv.org/pdf/2610.34662)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Vision-language models are shifting from offline passive understanding toward online proactive interaction. However, prevailing methods largely rely on explicit requests while overlooking implicit user intents (e.g., “where should I click next?” w...

</details>

<details>
<summary><b>3. WorldCast: World-Consistent Character Video Generation with Large-Angle Camera Control</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Fei Wang, Peng Zhang, Wenhua Liao, Yu Liu, Lei Cao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33135) • [📄 arXiv](https://arxiv.org/abs/2610.33135) • [📥 PDF](https://arxiv.org/pdf/2610.33135)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/PixCtrol/WorldCast)

> No abstract available.

</details>

<details>
<summary><b>4. Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.32285) • [📄 arXiv](https://arxiv.org/abs/2610.32285) • [📥 PDF](https://arxiv.org/pdf/2610.32285)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/TraceLite-AI/TraceTrial-Bench)

> Task success does not always mean the agent harness worked reliably. We introduce TraceTrial-Bench, a benchmark that evaluates agent task performance and harness reliability separately, using 95 mechanism-oriented tasks across seven components: co...

</details>

<details>
<summary><b>5. World-in-Context: Interactive Video World Modeling with In-Context Conditioning</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33169) • [📄 arXiv](https://arxiv.org/abs/2610.33169) • [📥 PDF](https://arxiv.org/pdf/2610.33169)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/aigc3d/World-in-Context)

> World-in-Context: Interactive Video World Modeling with In-Context Conditioning

</details>

<details>
<summary><b>6. Qwen-Image-PE: Learning Visual Design for Text-to-Image Prompt Enhancement</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33472) • [📄 arXiv](https://arxiv.org/abs/2610.33472) • [📥 PDF](https://arxiv.org/pdf/2610.33472)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We train a T2I prompt enhancer for visual design.

</details>

<details>
<summary><b>7. InteractiveWorld: A Verifiable Human-Centric Environment for Building Better Interactive MLLM Assistants</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33966) • [📄 arXiv](https://arxiv.org/abs/2610.33966) • [📥 PDF](https://arxiv.org/pdf/2610.33966)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/facebookresearch/InteractiveWorld)

> InteractiveWorld is the first environment that realistically simulates a human and a wearable AI assistant interacting in the physical world: the assistant sees through the human’s eyes and talks, the human decides and acts, and the world responds.

</details>

<details>
<summary><b>8. Δτ-Bench: Pushing Agents Toward a Moving Capability</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.34238) • [📄 arXiv](https://arxiv.org/abs/2610.34238) • [📥 PDF](https://arxiv.org/pdf/2610.34238)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zjunlp/delta-tau-bench)

> In this paper, we introduce △τ-Bench, an executable agent benchmark that models the structural evolution of large tool ecosystems.

</details>

<details>
<summary><b>9. Continual Learning Exam: Benchmarking Long-Horizon RSI and Experiential Learning in LLM Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33956) • [📄 arXiv](https://arxiv.org/abs/2610.33956) • [📥 PDF](https://arxiv.org/pdf/2610.33956)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/haoyu-zhao/Continual-Learning-Exam)

> 🚀 Introducing Continual Learning Exam (CLE): Benchmarking Long-Horizon RSI and Experiential Learning in LLM Agents! 🧠 Can LLM agents truly learn from experience and continuously improve over time? Most existing agent benchmarks focus on whether ag...

</details>

<details>
<summary><b>10. Work Agents: Systems, Capabilities, and Optimization</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Junhao Shen, Cheng Yang, Mingqi Gao, Enjun Du, Zishan Xu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.35845) • [📄 arXiv](https://arxiv.org/abs/2610.35845) • [📥 PDF](https://arxiv.org/pdf/2610.35845)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/EnjunDu/awesome-working-agent)

> As foundation models evolve from information generation to digital work execution, agents are increasingly capable of completing complex tasks across tools, applications, and workflows. However, existing research lacks a unified perspective on how...

</details>

<details>
<summary><b>11. DynaFlow: Learning Free-Form Multi-Agent Orchestration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.34467) • [📄 arXiv](https://arxiv.org/abs/2610.34467) • [📥 PDF](https://arxiv.org/pdf/2610.34467)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/amiya-special/Dynaflow)

> GitHub: https://github.com/amiya-special/Dynaflow Arxiv: https://arxiv.org/abs/2610.34467 Project Page: https://amiya-special.github.io/DynaFlow-website

</details>

<details>
<summary><b>12. Novel or Not? PatEx-Gym: Agentic Prior-Art Search for Novelty Assessment in Patent Examination</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33503) • [📄 arXiv](https://arxiv.org/abs/2610.33503) • [📥 PDF](https://arxiv.org/pdf/2610.33503)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIforIP/PatEx-Gym)

> PatEx-Gym introduces the first long-horizon agentic task in the patent domain, requiring agents to autonomously search for prior art, inspect technical evidence, iteratively refine their search strategies, and produce evidence-grounded Office Acti...

</details>

<details>
<summary><b>13. Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.32195) • [📄 arXiv](https://arxiv.org/abs/2610.32195) • [📥 PDF](https://arxiv.org/pdf/2610.32195)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce PRINCIPIA, an automatic framework for scientific knowledge construction at test time. By observing and interpreting experimental outcomes, PRINCIPIA forms hypotheses, generalizes them into  empirical principles, and revises hypotheses...

</details>

<details>
<summary><b>14. AgentProofBench: Benchmarking Reusable Verifiers for Web Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.34634) • [📄 arXiv](https://arxiv.org/abs/2610.34634) • [📥 PDF](https://arxiv.org/pdf/2610.34634)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/EthanNing/AgentProofBench)

> Benchmarking Reusable Verifiers for Web Agents

</details>

<details>
<summary><b>15. ProAS: A Progress-Aware Advantage Shaping Framework for Multi-Turn Agentic Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18683) • [📄 arXiv](https://arxiv.org/abs/2610.18683) • [📥 PDF](https://arxiv.org/pdf/2610.18683)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LINs-lab/ProAS)

> We argue against the limitations of the traditional GRPO-type credit assignment, such as (1) the unreasonable reinforcement of unreasonable actions within the positively advantaged trajectory,  and (2) the fact that actions within the negatively a...

</details>

<details>
<summary><b>16. Extreme Compression of Per-Layer Embeddings via Cross-Layer Latent Vector Quantization</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Xinyi Shang, Peng Sun, Xiaohan Zhao, Yaxin Luo, Jiacheng Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.31309) • [📄 arXiv](https://arxiv.org/abs/2610.31309) • [📥 PDF](https://arxiv.org/pdf/2610.31309)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VILA-Lab/optic)

> Recent work shows that increasing embedding capacity can improve language models, making it a promising direction for scaling. As more capacity goes into embedding tables, extreme compression becomes increasingly important: we want to keep the ben...

</details>

<details>
<summary><b>17. Steering Black-Box Large Language Model Agents with Pitfall Anticipation World Model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yun Zhou, Yawei Wang, Youzhi Luo, Han Ding, Jiongxiao Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29139) • [📄 arXiv](https://arxiv.org/abs/2610.29139) • [📥 PDF](https://arxiv.org/pdf/2610.29139)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Exploring selective world-model guidance in the agent loop for anticipating and avoiding future failures.

</details>

<details>
<summary><b>18. Trustworthy Recursive Self-Improvement in the Era of Large Models: Foundation, Risks, Evaluation and Open Challenges</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tongyu Wen, Qipeng Wang, Jinyi Bai, Ang Gao, Jiayue Pu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.35193) • [📄 arXiv](https://arxiv.org/abs/2610.35193) • [📥 PDF](https://arxiv.org/pdf/2610.35193)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> AI systems built around large models increasingly use experience and feedback to make persistent changes to models, prompts, memories, tools, and workflows. Recursive self-improvement (RSI) extends this process to inductive biases and to the mecha...

</details>

<details>
<summary><b>19. miniTE: Diagnosing and Bridging the Execution Gap in Native NVFP4 Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23928) • [📄 arXiv](https://arxiv.org/abs/2610.23928) • [📥 PDF](https://arxiv.org/pdf/2610.23928)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🚀 Can we really train LLMs with native FP4 reinforcement learning — without sacrificing stability? Excited to share miniTE , our open-source framework for stable and efficient native NVFP4 (W4A4) RL! One surprising finding: Same weights + same pre...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 19 |
| 📅 Today | [`2026-10-27.json`](data/daily/2026-10-27.json) | 19 |
| 📆 This Week | [`2026-W43.json`](data/weekly/2026-W43.json) | 29 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 1039 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-27 | 19 | [View JSON](data/daily/2026-10-27.json) |
| 📄 2026-10-26 | 10 | [View JSON](data/daily/2026-10-26.json) |
| 📄 2026-10-25 | 49 | [View JSON](data/daily/2026-10-25.json) |
| 📄 2026-10-24 | 50 | [View JSON](data/daily/2026-10-24.json) |
| 📄 2026-10-23 | 14 | [View JSON](data/daily/2026-10-23.json) |
| 📄 2026-10-22 | 28 | [View JSON](data/daily/2026-10-22.json) |
| 📄 2026-10-21 | 21 | [View JSON](data/daily/2026-10-21.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W43 | 29 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 1039 | [View JSON](data/monthly/2026-10.json) |
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
