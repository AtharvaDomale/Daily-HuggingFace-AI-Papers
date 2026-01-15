<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-24-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1064+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">24</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">129</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">326</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1064+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 15, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. MemGovern: Enhancing Code Agents through Learning from Governed Human Experiences</b> ⭐ 19</summary>

<br/>

**👥 Authors:** Yu2020, KunyiWang, shuozhang, cadche, jimson991

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06789) • [📄 arXiv](https://arxiv.org/abs/2601.06789) • [📥 PDF](https://arxiv.org/pdf/2601.06789)

**💻 Code:** [⭐ Code](https://github.com/QuantaAlpha/MemGovern)

> code agent

</details>

<details>
<summary><b>2. Solar Open Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07022) • [📄 arXiv](https://arxiv.org/abs/2601.07022) • [📥 PDF](https://arxiv.org/pdf/2601.07022)

> huggingface model: https://huggingface.co/upstage/Solar-Open-100B

</details>

<details>
<summary><b>3. KnowMe-Bench: Benchmarking Person Understanding for Lifelong Digital Companions</b> ⭐ 83</summary>

<br/>

**👥 Authors:** lanqz7766, ChenglongLi, Super-shuhe-v2, Zhisheng888, realty2333

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04745) • [📄 arXiv](https://arxiv.org/abs/2601.04745) • [📥 PDF](https://arxiv.org/pdf/2601.04745)

**💻 Code:** [⭐ Code](https://github.com/QuantaAlpha/KnowMeBench)

> know me

</details>

<details>
<summary><b>4. User-Oriented Multi-Turn Dialogue Generation with Tool Use at scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08225) • [📄 arXiv](https://arxiv.org/abs/2601.08225) • [📥 PDF](https://arxiv.org/pdf/2601.08225)

> While large language models have shown remarkable progress in tool use, maintaining high-quality, user-centric multi-turn conversations at scale remains a significant challenge. Our work focuses on: (1) Generating high-fidelity multi-turn dialogue...

</details>

<details>
<summary><b>5. ShowUI-π: Flow-based Generative Models as GUI Dexterous Hands</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24965) • [📄 arXiv](https://arxiv.org/abs/2512.24965) • [📥 PDF](https://arxiv.org/pdf/2512.24965)

**💻 Code:** [⭐ Code](https://github.com/showlab/showui-pi)

> TL;DR: ShowUI-π is a 450M flow-based vision-language-action model that treats GUI actions as continuous trajectories, generating smooth clicks and drags directly from screen observations. It unifies discrete and continuous actions, enabling precis...

</details>

<details>
<summary><b>6. MemoBrain: Executive Memory as an Agentic Brain for Reasoning</b> ⭐ 39</summary>

<br/>

**👥 Authors:** Zhao Cao, lz1001, TommyChien

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08079) • [📄 arXiv](https://arxiv.org/abs/2601.08079) • [📥 PDF](https://arxiv.org/pdf/2601.08079)

**💻 Code:** [⭐ Code](https://github.com/qhjqhj00/MemoBrain)

> Project Repo: https://github.com/qhjqhj00/MemoBrain

</details>

<details>
<summary><b>7. ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking</b> ⭐ 49</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06487) • [📄 arXiv](https://arxiv.org/abs/2601.06487) • [📥 PDF](https://arxiv.org/pdf/2601.06487)

**💻 Code:** [⭐ Code](https://github.com/Alibaba-NLP/qqr)

> As a key exploration of open-domain agents, our method has been validated within Amap's (Gaode Map) real-world business scenarios. Demonstrating significant practical value, we believe this paradigm represents one of the most important direction o...

</details>

<details>
<summary><b>8. Ministral 3</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08584) • [📄 arXiv](https://arxiv.org/abs/2601.08584) • [📥 PDF](https://arxiv.org/pdf/2601.08584)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Nanbeige4-3B Technical Report: Exploring the Frontier of Small Language Mod...

</details>

<details>
<summary><b>9. 3AM: Segment Anything with Geometric Consistency in Videos</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Min-Hung Chen, Fu-En Yang, Chin-Yang Lin, Cheng Sun, Yang-Che Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08831) • [📄 arXiv](https://arxiv.org/abs/2601.08831) • [📥 PDF](https://arxiv.org/pdf/2601.08831)

> Video object segmentation methods like SAM2 achieve strong performance through memory-based architectures but struggle under large viewpoint changes due to reliance on appearance features. Traditional 3D instance segmentation methods address viewp...

</details>

<details>
<summary><b>10. The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qingcheng Zeng, Naotoyokoyama, junjuewang, lrzneedresearch, weihao1115

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07264) • [📄 arXiv](https://arxiv.org/abs/2601.07264) • [📥 PDF](https://arxiv.org/pdf/2601.07264)

> We reveal a "confidence dichotomy" in tool-use LLM agents, finding that evidence tools like web search systematically induce overconfidence due to noisy retrieval, while verification tools like code interpreters help ground reasoning and reduce mi...

</details>

<details>
<summary><b>11. Parallel Context-of-Experts Decoding for Retrieval Augmented Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08670) • [📄 arXiv](https://arxiv.org/abs/2601.08670) • [📥 PDF](https://arxiv.org/pdf/2601.08670)

> Parallel Context-of-Experts Decoding (Pced) speeds up RAG by decoding in parallel from per-document KV-cache “experts” and selecting retrieval-supported tokens to recover cross-document reasoning.

</details>

<details>
<summary><b>12. Motion Attribution for Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08828) • [📄 arXiv](https://arxiv.org/abs/2601.08828) • [📥 PDF](https://arxiv.org/pdf/2601.08828)

> TL;DR: We propose MOTIVE, a scalable, motion-centric data attribution framework for video generation to identify which training clips improve or degrade motion dynamics, enabling curation and more.

</details>

<details>
<summary><b>13. ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08620) • [📄 arXiv](https://arxiv.org/abs/2601.08620) • [📥 PDF](https://arxiv.org/pdf/2601.08620)

> Retrieval-Augmented Generation (RAG) pipelines must address challenges beyond simple single-document retrieval, such as interpreting visual elements (tables, charts, images), synthesizing information across documents, and providing accurate source...

</details>

<details>
<summary><b>14. SnapGen++: Unleashing Diffusion Transformers for Efficient High-Fidelity Image Generation on Edge Devices</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08303) • [📄 arXiv](https://arxiv.org/abs/2601.08303) • [📥 PDF](https://arxiv.org/pdf/2601.08303)

> Proposes efficient diffusion transformers for edge devices via sparse attention, elastic training, and knowledge-guided distillation to achieve high-fidelity, fast on-device image generation.

</details>

<details>
<summary><b>15. VLingNav: Embodied Navigation with Adaptive Reasoning and Visual-Assisted Linguistic Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08665) • [📄 arXiv](https://arxiv.org/abs/2601.08665) • [📥 PDF](https://arxiv.org/pdf/2601.08665)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Unifying Perception and Action: A Hybrid-Modality Pipeline with Implicit Vi...

</details>

<details>
<summary><b>16. End-to-End Video Character Replacement without Structural Guidance</b> ⭐ 550</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08587) • [📄 arXiv](https://arxiv.org/abs/2601.08587) • [📥 PDF](https://arxiv.org/pdf/2601.08587)

**💻 Code:** [⭐ Code](https://github.com/Orange-3DV-Team/MoCha)

> End-to-End Video Character Replacement without Structural Guidance

</details>

<details>
<summary><b>17. JudgeRLVR: Judge First, Generate Second for Efficient Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sujian Li, Yudong Wang, Hailin Zhang, Hanyu Li, Jiangshan Duo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08468) • [📄 arXiv](https://arxiv.org/abs/2601.08468) • [📥 PDF](https://arxiv.org/pdf/2601.08468)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Structured Reasoning for Large Language Models (2026) Correct, Concise and ...

</details>

<details>
<summary><b>18. VideoLoom: A Video Large Language Model for Joint Spatial-Temporal Understanding</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07290) • [📄 arXiv](https://arxiv.org/abs/2601.07290) • [📥 PDF](https://arxiv.org/pdf/2601.07290)

**💻 Code:** [⭐ Code](https://github.com/JPShi12/VideoLoom)

> Joint temporal understanding and spatial perception within a single framework.

</details>

<details>
<summary><b>19. EpiCaR: Knowing What You Don't Know Matters for Better Reasoning in LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06786) • [📄 arXiv](https://arxiv.org/abs/2601.06786) • [📥 PDF](https://arxiv.org/pdf/2601.06786)

> Improving the reasoning abilities of large language models (LLMs) has largely relied on iterative self-training with model-generated data. While effective at boosting accuracy, existing approaches primarily reinforce successful reasoning paths, in...

</details>

<details>
<summary><b>20. UM-Text: A Unified Multimodal Model for Image Understanding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ting Zhu, Zipeng Guo, Gaojing Zhou, Xiaolong Fu, Lichen Ma

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08321) • [📄 arXiv](https://arxiv.org/abs/2601.08321) • [📥 PDF](https://arxiv.org/pdf/2601.08321)

> No abstract available.

</details>

<details>
<summary><b>21. Aligning Text, Code, and Vision: A Multi-Objective Reinforcement Learning Framework for Text-to-Visualization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04582) • [📄 arXiv](https://arxiv.org/abs/2601.04582) • [📥 PDF](https://arxiv.org/pdf/2601.04582)

> Generating working visualization code is not enough. Charts must be semantically correct and visually meaningful. We introduce RL-Text2Vis, the first reinforcement-learning framework for Text-to-Visualization, using post-execution feedback to join...

</details>

<details>
<summary><b>22. Towards Comprehensive Stage-wise Benchmarking of Large Language Models in Fact-Checking</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhen Ye, Ziyang Luo, Zhiqi Shen, Zixin Chen, Hongzhan Lin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02669) • [📄 arXiv](https://arxiv.org/abs/2601.02669) • [📥 PDF](https://arxiv.org/pdf/2601.02669)

> Current automatic LLM fact-checking tests are too narrow, they only check if a model can verify a claim, ignoring the hard parts like finding evidence and decomposing check-worthy claims. FactArena is built to evaluate the full fact-checking pipel...

</details>

<details>
<summary><b>23. The Agent's First Day: Benchmarking Learning, Exploration, and Scheduling in the Workplace Scenarios</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08173) • [📄 arXiv](https://arxiv.org/abs/2601.08173) • [📥 PDF](https://arxiv.org/pdf/2601.08173)

**💻 Code:** [⭐ Code](https://github.com/KnowledgeXLab/EvoEnv)

> The rapid evolution of Multi-modal Large Language Models (MLLMs) has advanced workflow automation; however, existing research mainly targets performance upper bounds in static environments, overlooking robustness for stochastic real-world deployme...

</details>

<details>
<summary><b>24. GeoMotionGPT: Geometry-Aligned Motion Understanding with Large Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07632) • [📄 arXiv](https://arxiv.org/abs/2601.07632) • [📥 PDF](https://arxiv.org/pdf/2601.07632)

**💻 Code:** [⭐ Code](https://github.com/JYe16/GeoMotionGPT)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 24 |
| 📅 Today | [`2026-01-15.json`](data/daily/2026-01-15.json) | 24 |
| 📆 This Week | [`2026-W02.json`](data/weekly/2026-W02.json) | 129 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 326 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-15 | 24 | [View JSON](data/daily/2026-01-15.json) |
| 📄 2026-01-14 | 42 | [View JSON](data/daily/2026-01-14.json) |
| 📄 2026-01-13 | 30 | [View JSON](data/daily/2026-01-13.json) |
| 📄 2026-01-12 | 33 | [View JSON](data/daily/2026-01-12.json) |
| 📄 2026-01-11 | 33 | [View JSON](data/daily/2026-01-11.json) |
| 📄 2026-01-10 | 33 | [View JSON](data/daily/2026-01-10.json) |
| 📄 2026-01-09 | 20 | [View JSON](data/daily/2026-01-09.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W02 | 129 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 326 | [View JSON](data/monthly/2026-01.json) |
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
