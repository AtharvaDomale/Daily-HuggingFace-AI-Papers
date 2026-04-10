<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3339+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">56</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">168</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3339+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 10, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rong Shan, Zihan Guo, Wenteng Chen, Huacan Chai, Chenyu Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08224) • [📄 arXiv](https://arxiv.org/abs/2604.08224) • [📥 PDF](https://arxiv.org/pdf/2604.08224)

> Large language model (LLM) agents are increasingly built less by changing model weights than by reorganizing the runtime around them. Capabilities that earlier systems expected the model to recover internally are now externalized into memory store...

</details>

<details>
<summary><b>2. KnowU-Bench: Towards Interactive, Proactive, and Personalized Mobile Agent Evaluation</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Shaohan Zhao, Guocheng Shao, Zhan Xu, Zhengxi Lu, Tongbo Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08455) • [📄 arXiv](https://arxiv.org/abs/2604.08455) • [📥 PDF](https://arxiv.org/pdf/2604.08455)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/KnowU-Bench)

> We introduce KnowU-Bench, an online, interactive personalization benchmark for mobile agents built on a reproducible Android emulation environment. We find that even frontier models like Claude Sonnet 4.6 struggle with our personalized and proacti...

</details>

<details>
<summary><b>3. Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.05333) • [📄 arXiv](https://arxiv.org/abs/2604.05333) • [📥 PDF](https://arxiv.org/pdf/2604.05333)

**💻 Code:** [⭐ Code](https://github.com/davidliuk/graph-of-skills)

> Skill usage has become a core component of modern agent systems and can substantially improve agents' ability to complete complex tasks. In real-world settings, where agents must monitor and interact with numerous personal applications, web browse...

</details>

<details>
<summary><b>4. DMax: Aggressive Parallel Decoding for dLLMs</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08302) • [📄 arXiv](https://arxiv.org/abs/2604.08302) • [📥 PDF](https://arxiv.org/pdf/2604.08302)

**💻 Code:** [⭐ Code](https://github.com/czg1225/DMax)

> DMax is a new dLLM paradigm achieving aggressive parallel decoding while preserving generation quality. Paper: https://arxiv.org/pdf/2604.08302 Code: https://github.com/czg1225/DMax Models: https://huggingface.co/collections/Zigeng/dmax-models Dat...

</details>

<details>
<summary><b>5. HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents</b> ⭐ 81</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07430) • [📄 arXiv](https://arxiv.org/abs/2604.07430) • [📥 PDF](https://arxiv.org/pdf/2604.07430)

**💻 Code:** [⭐ Code](https://github.com/Tencent-Hunyuan/HY-Embodied)

> GitHub: https://github.com/Tencent-Hunyuan/HY-Embodied HuggingFace: https://huggingface.co/tencent/HY-Embodied-0.5

</details>

<details>
<summary><b>6. When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08546) • [📄 arXiv](https://arxiv.org/abs/2604.08546) • [📥 PDF](https://arxiv.org/pdf/2604.08546)

**💻 Code:** [⭐ Code](https://github.com/H-EmbodVis/NUMINA)

> NUMINA is a training-free framework that tackles numerical misalignment in text-to-video diffusion models — the persistent failure of T2V models to generate the correct count of objects specified in prompts (e.g., producing 2 or 4 cats when "three...

</details>

<details>
<summary><b>7. FIT: A Large-Scale Dataset for Fit-Aware Virtual Try-On</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08526) • [📄 arXiv](https://arxiv.org/abs/2604.08526) • [📥 PDF](https://arxiv.org/pdf/2604.08526)

> We introduce FIT (Fit-Inclusive Try-on), a virtual try-on dataset of 1.13M samples covering diverse garment fits, each annotated with precise body/garment measurements. Our novel data generation pipeline leverages synthetic garment simulation and ...

</details>

<details>
<summary><b>8. Beyond Stochastic Exploration: What Makes Training Data Valuable for Agentic Search</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Guohua Liu, Guofeng Quan, Guochao Jiang, Wenfeng Feng, Chuzhan Hao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08124) • [📄 arXiv](https://arxiv.org/abs/2604.08124) • [📥 PDF](https://arxiv.org/pdf/2604.08124)

> No abstract available.

</details>

<details>
<summary><b>9. MolmoWeb: Open Visual Web Agent and Open Data for the Open Web</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rock Yuren Pang, Peter Sushko, Zixian Ma, Piper Wolters, Tanmay Gupta

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08516) • [📄 arXiv](https://arxiv.org/abs/2604.08516) • [📥 PDF](https://arxiv.org/pdf/2604.08516)

> No abstract available.

</details>

<details>
<summary><b>10. Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Yunfei Zhang, Ruotong Pan, Boxi Cao, Ruoxi Xu, Jiawei Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08362) • [📄 arXiv](https://arxiv.org/abs/2604.08362) • [📥 PDF](https://arxiv.org/pdf/2604.08362)

**💻 Code:** [⭐ Code](https://github.com/icip-cas/OmniBehavior)

> We introduce OmniBehavior, to our knowledge, the first user simulation benchmark constructed entirely from authentic user interaction logs, integrating long-horizon, cross-scenario and heterogeneous behavior traces into a unified framework. We pro...

</details>

<details>
<summary><b>11. OmniJigsaw: Enhancing Omni-Modal Reasoning via Modality-Orchestrated Reordering</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08209) • [📄 arXiv](https://arxiv.org/abs/2604.08209) • [📥 PDF](https://arxiv.org/pdf/2604.08209)

**💻 Code:** [⭐ Code](https://github.com/aim-uofa/OmniJigsaw)

> We introduce OmniJigsaw, a self-supervised RL post-training framework for omni-modal models. The core idea is a temporal jigsaw proxy task: reconstruct chronology from shuffled audio–visual clips, with three modality-orchestration strategies (JMI ...

</details>

<details>
<summary><b>12. SkillClaw: Let Skills Evolve Collectively with Agentic Evolver</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yong Wang, Xucong Wang, Yuxiang Ji, Shidong Yang, Ziyu Ma

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08377) • [📄 arXiv](https://arxiv.org/abs/2604.08377) • [📥 PDF](https://arxiv.org/pdf/2604.08377)

> No abstract available.

</details>

<details>
<summary><b>13. Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07394) • [📄 arXiv](https://arxiv.org/abs/2604.07394) • [📥 PDF](https://arxiv.org/pdf/2604.07394)

**💻 Code:** [⭐ Code](https://github.com/qqtang-code/FluxAttention)

> Paper Title: Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference Link: arXiv:2604.07394 (Preprint) 【TL;DR / One-Sentence Summary】 ⭐⭐⭐⭐⭐ (Highly Recommended). A brilliant hardware-aware co-design that optimizes Long-Context ...

</details>

<details>
<summary><b>14. RewardFlow: Generate Images by Optimizing What You Reward</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08536) • [📄 arXiv](https://arxiv.org/abs/2604.08536) • [📥 PDF](https://arxiv.org/pdf/2604.08536)

> We introduce RewardFlow, an inversion-free framework that steers pretrained diffusion and flow-matching models at inference time through multi-reward Langevin dynamics.

</details>

<details>
<summary><b>15. Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08503) • [📄 arXiv](https://arxiv.org/abs/2604.08503) • [📥 PDF](https://arxiv.org/pdf/2604.08503)

> Phantom is a Physics-Infused Video Generation model that jointly models visual content and latent physical dynamics. Conditioned on observed video frames and inferred physical states, Phantom jointly predicts latent physical dynamics and generates...

</details>

<details>
<summary><b>16. PokeGym: A Visually-Driven Long-Horizon Benchmark for Vision-Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhilin Liu, Chuanfu Shen, Yuangang Pan, Ye Huang, Ruizhi Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08340) • [📄 arXiv](https://arxiv.org/abs/2604.08340) • [📥 PDF](https://arxiv.org/pdf/2604.08340)

> No abstract available.

</details>

<details>
<summary><b>17. OpenVLThinkerV2: A Generalist Multimodal Reasoning Model for Multi-domain Visual Tasks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nanyun Peng, Yihe Deng, Yan Gao-Tian, Xin Chen, Wenbo Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08539) • [📄 arXiv](https://arxiv.org/abs/2604.08539) • [📥 PDF](https://arxiv.org/pdf/2604.08539)

> No abstract available.

</details>

<details>
<summary><b>18. GameWorld: Towards Standardized and Verifiable Evaluation of Multimodal Game Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mike Zheng Shou, Hwee Tou Ng, Kevin Qinghong Lin, Siyuan Hu, Mingyu Ouyang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07429) • [📄 arXiv](https://arxiv.org/abs/2604.07429) • [📥 PDF](https://arxiv.org/pdf/2604.07429)

> Proposes GameWorld, a standardized, verifiable benchmark for evaluating multimodal game agents across 34 browser games and 170 tasks using semantic actions and keyboard/mouse interfaces.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-04-10.json`](data/daily/2026-04-10.json) | 18 |
| 📆 This Week | [`2026-W14.json`](data/weekly/2026-W14.json) | 56 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 168 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-10 | 18 | [View JSON](data/daily/2026-04-10.json) |
| 📄 2026-04-09 | 8 | [View JSON](data/daily/2026-04-09.json) |
| 📄 2026-04-08 | 8 | [View JSON](data/daily/2026-04-08.json) |
| 📄 2026-04-07 | 17 | [View JSON](data/daily/2026-04-07.json) |
| 📄 2026-04-06 | 5 | [View JSON](data/daily/2026-04-06.json) |
| 📄 2026-04-05 | 45 | [View JSON](data/daily/2026-04-05.json) |
| 📄 2026-04-04 | 45 | [View JSON](data/daily/2026-04-04.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W14 | 56 | [View JSON](data/weekly/2026-W14.json) |
| 📅 2026-W13 | 115 | [View JSON](data/weekly/2026-W13.json) |
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 168 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
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
