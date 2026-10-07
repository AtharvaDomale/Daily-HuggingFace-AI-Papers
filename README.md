<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-23-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6908+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">23</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">63</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">300</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6908+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 07, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. DuoMatching: Joint-Marginal Distribution Matching for Few-Step Video Generation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03543) • [📄 arXiv](https://arxiv.org/abs/2610.03543) • [📥 PDF](https://arxiv.org/pdf/2610.03543)

**💻 Code:** [⭐ Code](https://github.com/JohnZhan2023/DuoMatching) • [⭐ Code](https://github.com/huggingface)

> Hi HF community! I’m one of the authors of DuoMatching, our work on high-quality, real-time video generation with image priors. DuoMatching combines joint distribution matching from a video teacher with direct frame-level supervision from an image...

</details>

<details>
<summary><b>2. AutoSciBench: Autonomous Benchmark Generation for Evaluating Scientific Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05140) • [📄 arXiv](https://arxiv.org/abs/2610.05140) • [📥 PDF](https://arxiv.org/pdf/2610.05140)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Scientific agents usually take the tests. With AutoSciBench, they also help build them. The framework constructs questions, scientific data, and ground-truth answers, then uses solver feedback to revise task designs. Lessons from earlier runs info...

</details>

<details>
<summary><b>3. HuatuoGPT-3: RL-Only Domain Adaptation from Base Models</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05966) • [📄 arXiv](https://arxiv.org/abs/2610.05966) • [📥 PDF](https://arxiv.org/pdf/2610.05966)

**💻 Code:** [⭐ Code](https://github.com/FreedomIntelligence/HuatuoGPT-3) • [⭐ Code](https://github.com/huggingface)

> HuatuoGPT-3 advances the HuatuoGPT line from medical data adaptation to training-paradigm innovation. Instead of following the conventional SFT-then-RL pipeline, it explores RL-only domain adaptation from base models through OnePO, using teacher o...

</details>

<details>
<summary><b>4. World Action Learning via Interaction-Centric Spectral Latent Guidance</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03607) • [📄 arXiv](https://arxiv.org/abs/2610.03607) • [📥 PDF](https://arxiv.org/pdf/2610.03607)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> How can egocentric human videos effectively benefit robot learning despite camera motion and human–robot execution differences? WING learns interaction-centric latent actions from ego videos and transfers them to robot policies through low-frequen...

</details>

<details>
<summary><b>5. EVISKILL: Grounding Skill Evolution in Replayable Evidence</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Xin Wang, Qinggang Zhang, Yiwei Dai, Yili Wang, Yan Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05030) • [📄 arXiv](https://arxiv.org/abs/2610.05030) • [📥 PDF](https://arxiv.org/pdf/2610.05030)

**💻 Code:** [⭐ Code](https://github.com/Zhouyaner/Eviskill) • [⭐ Code](https://github.com/huggingface)

> Continual skill evolution enables LLM agents to accumulate and refine reusable procedural knowledge from interaction experience without updating model parameters. Its effectiveness depends on determining not only what to change, but also why a cha...

</details>

<details>
<summary><b>6. Selection-Based Structured Reasoning: Toward Efficient Multimodal Search Agents</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01892) • [📄 arXiv](https://arxiv.org/abs/2610.01892) • [📥 PDF](https://arxiv.org/pdf/2610.01892)

**💻 Code:** [⭐ Code](https://github.com/zfy0314/ssr-unofficial) • [⭐ Code](https://github.com/huggingface)

> Multimodal agents commonly generate free-form reasoning before each action. For small models, limited model capacity can result in lengthy reasoning that provides little useful guidance for action generation while incurring substantial inference c...

</details>

<details>
<summary><b>7. DiVeR: Decision-Critical Verifier Learning for VLA Test-Time Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04933) • [📄 arXiv](https://arxiv.org/abs/2610.04933) • [📥 PDF](https://arxiv.org/pdf/2610.04933)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> DiVeR improves VLA test-time scaling by identifying sparse decision-critical states from action-representation dispersion and focusing verifier learning where action selection matters most.

</details>

<details>
<summary><b>8. Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38972) • [📄 arXiv](https://arxiv.org/abs/2609.38972) • [📥 PDF](https://arxiv.org/pdf/2609.38972)

**💻 Code:** [⭐ Code](https://github.com/yihuaihong/CIA-minimal-repro) • [⭐ Code](https://github.com/huggingface)

> Do LLMs actually reason the way their chain-of-thought says they do? We introduce CoT-Interpretability Alignment (CIA), a metric that checks whether the reasoning written in a model's CoT matches the internal strategy detected by interpretability ...

</details>

<details>
<summary><b>9. GUI-HARVEST: Self-Improving GUI Agents through Evidence-Driven Harness Evolution</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00948) • [📄 arXiv](https://arxiv.org/abs/2610.00948) • [📥 PDF](https://arxiv.org/pdf/2610.00948)

**💻 Code:** [⭐ Code](https://github.com/GaryYang12345/GUI-HARVEST) • [⭐ Code](https://github.com/huggingface)

> 👋 We’re excited to share GUI-HARVEST , which enables GUI agents to improve automatically by evolving their execution harness while keeping model weights frozen. The idea: learn from what actually happens on screen. GUI-HARVEST compares screenshots...

</details>

<details>
<summary><b>10. Understanding and Enhancing Backdoor Persistency in LLM Agent Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07510) • [📄 arXiv](https://arxiv.org/abs/2610.07510) • [📥 PDF](https://arxiv.org/pdf/2610.07510)

**💻 Code:** [⭐ Code](https://github.com/uiuc-kang-lab/PersistBD) • [⭐ Code](https://github.com/huggingface)

> Can benign post-training remove inherited backdoors in LLM agents? We find that supervised fine-tuning weakens backdoors, but subsequent RL often preserves—and sometimes amplifies—the remaining malicious behavior. Our method, PersistBD, increases ...

</details>

<details>
<summary><b>11. TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07767) • [📄 arXiv](https://arxiv.org/abs/2610.07767) • [📥 PDF](https://arxiv.org/pdf/2610.07767)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Reinforcement learning (RL) for post-training large language models (LLMs) incurs substantial computation and memory overhead during rollout generation, which motivates low-precision rollout for efficient RL training. However, existing FP4 RL meth...

</details>

<details>
<summary><b>12. ALIVE: Interaction-Aligned Object Insertion for First-Frame-Guided Video Editing</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08779) • [📄 arXiv](https://arxiv.org/abs/2610.08779) • [📥 PDF](https://arxiv.org/pdf/2610.08779)

**💻 Code:** [⭐ Code](https://github.com/zhouzhenghong-gt/ALIVE-code) • [⭐ Code](https://github.com/huggingface)

> Make inserted objects “alive”: not merely visible, but part of the video’s world, responding to surrounding actions.

</details>

<details>
<summary><b>13. Personal-Agent Mediated Recommendation with Cross-Platform User History</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07588) • [📄 arXiv](https://arxiv.org/abs/2610.07588) • [📥 PDF](https://arxiv.org/pdf/2610.07588)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We study how a personal agent can selectively revise a strong platform ranking using cross-platform user history. Personal-Agent Mediated Recommendation: We formalize this new recommendation setting. MediateRec: We introduce a benchmark spanning c...

</details>

<details>
<summary><b>14. EmbodiedSmith: Scaling Embodied Data through Recursive Self-Improvement Flywheel in Simulation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zepeng Lin, Wenxuan Song, Mingjian Liang, Yifei Deng, Yikai Qin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07969) • [📄 arXiv](https://arxiv.org/abs/2610.07969) • [📥 PDF](https://arxiv.org/pdf/2610.07969)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. Hiding Tool Latency in On-Device Cascaded Voice Agent through Speculative Execution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07641) • [📄 arXiv](https://arxiv.org/abs/2610.07641) • [📥 PDF](https://arxiv.org/pdf/2610.07641)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Tool-augmented speech assistants typically serialize automatic speech recognition, large language model inference, and external tool execution. As a result, tool latency is incurred only after the user has finished speaking and the LLM has identif...

</details>

<details>
<summary><b>16. Attacca: Goal-Directed Control under State Continuity for Long-Horizon Embodied Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07785) • [📄 arXiv](https://arxiv.org/abs/2610.07785) • [📥 PDF](https://arxiv.org/pdf/2610.07785)

**💻 Code:** [⭐ Code](https://github.com/attacca-project/attacca) • [⭐ Code](https://github.com/huggingface)

> .

</details>

<details>
<summary><b>17. Judged Useless, Queried Anyway: Tool-Using Agents Rarely Turn Their Own Evidence Judgments into Stopping Decisions</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06191) • [📄 arXiv](https://arxiv.org/abs/2610.06191) • [📥 PDF](https://arxiv.org/pdf/2610.06191)

**💻 Code:** [⭐ Code](https://github.com/bennidict23/judged-useless-queried-anyway) • [⭐ Code](https://github.com/huggingface)

> We study whether tool-using agents actually act on their own judgments that retrieved evidence is useless. Across seven agents, they recognize failing-source results as useless 97–100% of the time, yet rarely stop querying. An enforced integration...

</details>

<details>
<summary><b>18. DistScene: Object-to-Scene Distillation for 3D Scene Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tianyu Liu, Chengcheng Zhou, Ken Deng, Hongyu Yan, Kunming Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06960) • [📄 arXiv](https://arxiv.org/abs/2610.06960) • [📥 PDF](https://arxiv.org/pdf/2610.06960)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>19. VeriFine: Scaling Verification for Self-Improvement in Embodied Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08761) • [📄 arXiv](https://arxiv.org/abs/2610.08761) • [📥 PDF](https://arxiv.org/pdf/2610.08761)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>20. World Models' Last Exam in Physics</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ziming Qin, Xinjie Lin, Yuzhao Peng, Qingle Liu, Mingju Gao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08791) • [📄 arXiv](https://arxiv.org/abs/2610.08791) • [📥 PDF](https://arxiv.org/pdf/2610.08791)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>21. Building Rome from a Single Image</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Quentin Herau, Depu Meng, Tianshuo Xu, Fang Li, Jiraphon Yenphraphai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08790) • [📄 arXiv](https://arxiv.org/abs/2610.08790) • [📥 PDF](https://arxiv.org/pdf/2610.08790)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. Harness Engineering for Software Engineering via Modular Executable Dev-Primitives</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haohan Wang, Peng Kuang, Xinjie Li, Haibo Jin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07832) • [📄 arXiv](https://arxiv.org/abs/2610.07832) • [📥 PDF](https://arxiv.org/pdf/2610.07832)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> HERMES turns repository components into agent-native Dev-Primitives, enabling localized reasoning, natural-language inter-component communication, and diagnosis-driven revision for long-horizon software engineering.

</details>

<details>
<summary><b>23. HiPLEX: Hierarchical Policy Factorization for Full Duplex Speech Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07727) • [📄 arXiv](https://arxiv.org/abs/2610.07727) • [📥 PDF](https://arxiv.org/pdf/2610.07727)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Our method, HiPLEX (Hierarchical policy factorization for full-duPLEX SLMs), factorizes the policy into when to talk and what to talk. Plus, our credit assignment method causally attributes rewards to the correct or incorrect timing and semantic d...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 23 |
| 📅 Today | [`2026-10-07.json`](data/daily/2026-10-07.json) | 23 |
| 📆 This Week | [`2026-W40.json`](data/weekly/2026-W40.json) | 63 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 300 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-07 | 23 | [View JSON](data/daily/2026-10-07.json) |
| 📄 2026-10-06 | 21 | [View JSON](data/daily/2026-10-06.json) |
| 📄 2026-10-05 | 19 | [View JSON](data/daily/2026-10-05.json) |
| 📄 2026-10-04 | 84 | [View JSON](data/daily/2026-10-04.json) |
| 📄 2026-10-03 | 84 | [View JSON](data/daily/2026-10-03.json) |
| 📄 2026-10-02 | 35 | [View JSON](data/daily/2026-10-02.json) |
| 📄 2026-10-01 | 34 | [View JSON](data/daily/2026-10-01.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W40 | 63 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 300 | [View JSON](data/monthly/2026-10.json) |
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
