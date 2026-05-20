<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-16-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4060+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">16</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">49</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">439</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4060+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 20, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Process Rewards with Learned Reliability</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15529) • [📄 arXiv](https://arxiv.org/abs/2605.15529) • [📥 PDF](https://arxiv.org/pdf/2605.15529)

**💻 Code:** [⭐ Code](https://github.com/JinYuanLi0012/Beta-Binomial-PRM) • [⭐ Code](https://github.com/huggingface)

> Process Reward Models (PRMs) provide step-level feedback for reasoning, but current PRMs usually output only a single reward score for each step. Downstream methods must therefore treat imperfect step-level reward predictions as reliable decision ...

</details>

<details>
<summary><b>2. Artifact-Bench: Evaluating MLLMs on Detecting and Assessing the Artifacts of AI-Generated Videos</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Xuehai Bai, Qixun Wang, Zhuoran Zhang, Yang Shi, Yuqi Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.18984) • [📄 arXiv](https://arxiv.org/abs/2605.18984) • [📥 PDF](https://arxiv.org/pdf/2605.18984)

**💻 Code:** [⭐ Code](https://github.com/FrankYang-17/Artifact-Bench) • [⭐ Code](https://github.com/huggingface)

> Recent video generative models have greatly improved the realism of AI-generated videos, yet their outputs still exhibit artifacts such as temporal inconsistencies, structural distortions, and semantic incoherence. While Multimodal Large Language ...

</details>

<details>
<summary><b>3. CogOmniControl: Reasoning-Driven Controllable Video Generation via Creative Intent Cognition</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alan Zhao, Xiaotong Zhao, Yucheng Zhou, Songlian Li, Hongji Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19995) • [📄 arXiv](https://arxiv.org/abs/2605.19995) • [📥 PDF](https://arxiv.org/pdf/2605.19995)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Recent diffusion models achieve strong photorealism and fluency in video generation, yet remain fragile under abstract, sparse or complex conditions, leading to poor performance in professional production workflows such as storyboard sketches and ...

</details>

<details>
<summary><b>4. Aurora: Unified Video Editing with a Tool-Using Agent</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Hang Hua, Zhenghong Zhou, Zhiyuan Xiao, Ziyun Zeng, Yongsheng Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.18748) • [📄 arXiv](https://arxiv.org/abs/2605.18748) • [📥 PDF](https://arxiv.org/pdf/2605.18748)

**💻 Code:** [⭐ Code](https://github.com/yeates/Aurora) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>5. OmniGUI: Benchmarking GUI Agents in Omni-Modal Smartphone Environments</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.18758) • [📄 arXiv](https://arxiv.org/abs/2605.18758) • [📥 PDF](https://arxiv.org/pdf/2605.18758)

**💻 Code:** [⭐ Code](https://github.com/omni-gui/OmniGUI) • [⭐ Code](https://github.com/huggingface)

> Current benchmarks for graphical user interface (GUI) agents predominantly rely on static screenshots. However, real-world smartphone interaction routinely requires agents to process transient audio cues and temporal video dynamics that are tightl...

</details>

<details>
<summary><b>6. MSAVBench: Towards Comprehensive and Reliable Evaluation of Multi-Shot Audio-Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20183) • [📄 arXiv](https://arxiv.org/abs/2605.20183) • [📥 PDF](https://arxiv.org/pdf/2605.20183)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> MSAVBench is the first comprehensive benchmark for multi-shot audio-video generation.

</details>

<details>
<summary><b>7. CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19436) • [📄 arXiv](https://arxiv.org/abs/2605.19436) • [📥 PDF](https://arxiv.org/pdf/2605.19436)

**💻 Code:** [⭐ Code](https://github.com/ahmedheakl/CEPO) • [⭐ Code](https://github.com/huggingface)

> CEPO is a token-level credit assignment method for RLVR that replaces uniform reward signals with a contrastive ratio between correct and wrong answer teachers (drawn from rejected rollouts at no extra cost), focusing gradient updates on the token...

</details>

<details>
<summary><b>8. AutoResearchClaw: Self-Reinforcing Autonomous Research with Human-AI Collaboration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haonian Ji, Bingzhou Li, Mairui Li, Shi Qiu, Jiaqi Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20025) • [📄 arXiv](https://arxiv.org/abs/2605.20025) • [📥 PDF](https://arxiv.org/pdf/2605.20025)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/aiming-lab/AutoResearchClaw)

> AutoResearchClaw is a multi-agent, human-in-the-loop research framework that leverages self-reinforcing mechanisms, structured debate, and adaptive execution to autonomously improve scientific discovery and outperform existing autonomous research ...

</details>

<details>
<summary><b>9. GoLongRL: Capability-Oriented Long Context Reinforcement Learning with Multitask Alignment</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhenpeng Su, Junmin Chen, Tanlong Du, Tiehua Mei, Minxuan Lv

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19577) • [📄 arXiv](https://arxiv.org/abs/2605.19577) • [📥 PDF](https://arxiv.org/pdf/2605.19577)

**💻 Code:** [⭐ Code](https://github.com/xiaoxuanNLP/GoLongRL) • [⭐ Code](https://github.com/huggingface)

> We present GoLongRL , a fully open-source, capability-oriented post-training recipe for long-context reinforcement learning with verifiable rewards (RLVR) 🚀 GoLongRL-30B-A3B achieves long-context performance comparable to DeepSeek-R1-0528 and Qwen...

</details>

<details>
<summary><b>10. Echo-Forcing: A Scene Memory Framework for Interactive Long Video Generation</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Yuqi Li, Haotong Qin, Zhefeng Zhang, Weilun Feng, Mingqiang Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.16003) • [📄 arXiv](https://arxiv.org/abs/2605.16003) • [📥 PDF](https://arxiv.org/pdf/2605.16003)

**💻 Code:** [⭐ Code](https://github.com/mingqiangWu/Echo-Forcing) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>11. DocAtlas: Multilingual Document Understanding Across 80+ Languages</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.12623) • [📄 arXiv](https://arxiv.org/abs/2605.12623) • [📥 PDF](https://arxiv.org/pdf/2605.12623)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> DocAtlas is a framework for constructing high-fidelity multilingual OCR datasets and benchmarks covering 82 languages and 9 evaluation tasks, using differential rendering to produce model-free structural annotations from native documents. Evaluati...

</details>

<details>
<summary><b>12. Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.11609) • [📄 arXiv](https://arxiv.org/abs/2605.11609) • [📥 PDF](https://arxiv.org/pdf/2605.11609)

**💻 Code:** [⭐ Code](https://github.com/FloyedShen/AntiSD) • [⭐ Code](https://github.com/huggingface)

> AntiSD reaches GRPO's accuracy in 2–10× fewer training steps and improves final accuracy by up to +11.5 points on AIME 2024/2025, HMMT 2025, and BeyondAIME — consistent across 4B–30B dense and MoE models. Standard self-distillation in reasoning RL...

</details>

<details>
<summary><b>13. SceneCode: Executable World Programs for Editable Indoor Scenes with Articulated Objects</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kevin Qinghong Lin, Zhengyuan Yang, Linjie Li, Yuhao Wang, Puyi Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19587) • [📄 arXiv](https://arxiv.org/abs/2605.19587) • [📥 PDF](https://arxiv.org/pdf/2605.19587)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. OpenComputer: Verifiable Software Worlds for Computer-Use Agents</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Kangqi Ni, Xiao Zhou, Yilun Zhao, Qianran Ma, Jinbiao Wei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19769) • [📄 arXiv](https://arxiv.org/abs/2605.19769) • [📥 PDF](https://arxiv.org/pdf/2605.19769)

**💻 Code:** [⭐ Code](https://github.com/echo0715/OpenComputer) • [⭐ Code](https://github.com/huggingface)

> OpenComputer provides a framework for creating verifiable software environments and benchmarks to evaluate computer-use agents through precise, state-aware task verification across diverse desktop applications.

</details>

<details>
<summary><b>15. Draft Less, Retrieve More: Hybrid Tree Construction for Speculative Decoding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Baolin Zhang, Quan Kong, Xinyi Hu, Tianyu Liu, Yuhao Shen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20104) • [📄 arXiv](https://arxiv.org/abs/2605.20104) • [📥 PDF](https://arxiv.org/pdf/2605.20104)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Graft is a training-free speculative decoding framework that combines pruning and token retrieval to improve acceptance rates and accelerate large language model inference without incurring additional computational overhead.

</details>

<details>
<summary><b>16. Delta Attention Residuals</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junjie Hu, Zefan Cai, Cheng Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.18855) • [📄 arXiv](https://arxiv.org/abs/2605.18855) • [📥 PDF](https://arxiv.org/pdf/2605.18855)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Delta Attention Residuals improve transformer performance by attending to sublayer deltas instead of cumulative hidden states, enabling more selective and effective routing across layers.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 16 |
| 📅 Today | [`2026-05-20.json`](data/daily/2026-05-20.json) | 16 |
| 📆 This Week | [`2026-W20.json`](data/weekly/2026-W20.json) | 49 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 439 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-20 | 16 | [View JSON](data/daily/2026-05-20.json) |
| 📄 2026-05-19 | 15 | [View JSON](data/daily/2026-05-19.json) |
| 📄 2026-05-18 | 18 | [View JSON](data/daily/2026-05-18.json) |
| 📄 2026-05-17 | 53 | [View JSON](data/daily/2026-05-17.json) |
| 📄 2026-05-16 | 53 | [View JSON](data/daily/2026-05-16.json) |
| 📄 2026-05-15 | 26 | [View JSON](data/daily/2026-05-15.json) |
| 📄 2026-05-14 | 22 | [View JSON](data/daily/2026-05-14.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W20 | 49 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 439 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
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
