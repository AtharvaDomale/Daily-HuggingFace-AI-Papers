<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-17-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3305+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">17</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">22</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">134</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3305+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 07, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Self-Distilled RLVR</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Naibin Gu, Minghui Chen, Qingyi Si, Chuanyu Qin, Chenxu Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.03128) • [📄 arXiv](https://arxiv.org/abs/2604.03128) • [📥 PDF](https://arxiv.org/pdf/2604.03128)

> nice breakdown of this one here if anyone wants the tldr https://arxivexplained.com/papers/self-distilled-rlvr the part about rlvr is what got me

</details>

<details>
<summary><b>2. A Simple Baseline for Streaming Video Understanding</b> ⭐ 49</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02317) • [📄 arXiv](https://arxiv.org/abs/2511.16655) • [📥 PDF](https://arxiv.org/pdf/2604.02317)

**💻 Code:** [⭐ Code](https://github.com/EvolvingLMMs-Lab/SimpleStream)

> 🚀 SimpleStream: Rethinking Memory in Streaming Video Understanding Recent streaming video understanding methods increasingly rely on complex memory mechanisms. We revisit a simple question: do we really need them? 🔑 Key finding A simple sliding-wi...

</details>

<details>
<summary><b>3. Token Warping Helps MLLMs Look from Nearby Viewpoints</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Juil Koo, Seungwoo Yoo, Mingue Park, Chanho Park, Phillip Y. Lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02870) • [📄 arXiv](https://arxiv.org/abs/2604.02870) • [📥 PDF](https://arxiv.org/pdf/2604.02870)

**💻 Code:** [⭐ Code](https://github.com/KAIST-Visual-AI-Group/Token-Warping-MLLM)

> CVPR 2026 Paper: https://arxiv.org/abs/2604.02870 Project Page: https://token-warping-mllm.github.io/ Code: https://github.com/KAIST-Visual-AI-Group/Token-Warping-MLLM

</details>

<details>
<summary><b>4. Agentic-MME: What Agentic Capability Really Brings to Multimodal Intelligence?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Binyu Wang, Jinglin Chen, Siyi Wang, Yishan Yang, Qianshan Wei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.03016) • [📄 arXiv](https://arxiv.org/abs/2604.03016) • [📥 PDF](https://arxiv.org/pdf/2604.03016)

> Insightful Work!

</details>

<details>
<summary><b>5. Test-Time Scaling Makes Overtraining Compute-Optimal</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01411) • [📄 arXiv](https://arxiv.org/abs/2604.01411) • [📥 PDF](https://arxiv.org/pdf/2604.01411)

> That new LFM2.5-350M is super overtrained — and everyone was shocked by how far they pushed it. As it turns out, we have a scaling law for that. T² (Train-to-Test) scaling combines Chinchilla pretraining scaling with test-time scaling via repeated...

</details>

<details>
<summary><b>6. Communicating about Space: Language-Mediated Spatial Integration Across Partial Views</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.27183) • [📄 arXiv](https://arxiv.org/abs/2603.27183) • [📥 PDF](https://arxiv.org/pdf/2603.27183)

**💻 Code:** [⭐ Code](https://github.com/ankursikarwar/Cosmic)

> No abstract available.

</details>

<details>
<summary><b>7. GrandCode: Achieving Grandmaster Level in Competitive Programming via Agentic Reinforcement Learning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02721) • [📄 arXiv](https://arxiv.org/abs/2604.02721) • [📥 PDF](https://arxiv.org/pdf/2604.02721)

**💻 Code:** [⭐ Code](https://github.com/deepreinforce-ai/codeforces)

> In the most recent three Codeforces live competitions, i.e., Round 1087, Round 1088, and Round 1089, GrandCode, GrandCode, ranked first in all of them, beating all human participants, including legendary grandmasters. GrandCode is a multi-agent re...

</details>

<details>
<summary><b>8. InCoder-32B-Thinking: Industrial Code World Model for Thinking</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tuney Zheng, Junhang Cheng, Jiajun Wu, Wei Zhang, Jian Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.03144) • [📄 arXiv](https://arxiv.org/abs/2604.03144) • [📥 PDF](https://arxiv.org/pdf/2604.03144)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API InCoder-32B: Code Foundation Model for Industrial Scenarios (2026) SiliconM...

</details>

<details>
<summary><b>9. AgentSocialBench: Evaluating Privacy Risks in Human-Centered Agentic Social Networks</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01487) • [📄 arXiv](https://arxiv.org/abs/2604.01487) • [📥 PDF](https://arxiv.org/pdf/2604.01487)

**💻 Code:** [⭐ Code](https://github.com/kingofspace0wzz/agentsocialbench)

> With the rise of personalized, persistent LLM agent frameworks such as OpenClaw, human-centered agentic social networks in which teams of collaborative AI agents serve individual users in a social network across multiple domains are becoming a rea...

</details>

<details>
<summary><b>10. Swift-SVD: Theoretical Optimality Meets Practical Efficiency in Low-Rank LLM Compression</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01609) • [📄 arXiv](https://arxiv.org/abs/2604.01609) • [📥 PDF](https://arxiv.org/pdf/2604.01609)

> Instead of repeated SVD or indirect constructions, Swift-SVD computes the optimal activation-aware solution via a single eigendecomposition, making it practical at LLM scale.

</details>

<details>
<summary><b>11. AgentHazard: A Benchmark for Evaluating Harmful Behavior in Computer-Use Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yige Li, Xingjun Ma, Yingshui Tan, Yifan Ding, Yunhao Feng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02947) • [📄 arXiv](https://arxiv.org/abs/2604.02947) • [📥 PDF](https://arxiv.org/pdf/2604.02947)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API From Assistant to Double Agent: Formalizing and Benchmarking Attacks on Ope...

</details>

<details>
<summary><b>12. VLMs Need Words: Vision Language Models Ignore Visual Detail In Favor of Semantic Anchors</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nael Abu-Ghazaleh, Erfan Shayegani, Yu Fu, Xiaofu Chen, Haz Sameen Shahgir

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02486) • [📄 arXiv](https://arxiv.org/abs/2604.02486) • [📥 PDF](https://arxiv.org/pdf/2604.02486)

> 🤔 The VLM community has long had the intuition that vision-focused tasks resisting transcription into text are harder for VLMs. What makes this especially puzzling: prior work has shown the visual information IS there inside the LM's representatio...

</details>

<details>
<summary><b>13. Xpertbench: Expert Level Tasks with Rubrics-Based Evaluation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Duo Wang, Yongchang Peng, Yuxin Ma, Xin Ma, Xue Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02368) • [📄 arXiv](https://arxiv.org/abs/2604.02368) • [📥 PDF](https://arxiv.org/pdf/2604.02368)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API $OneMillion-Bench: How Far are Language Agents from Human Experts? (2026) R...

</details>

<details>
<summary><b>14. Salt: Self-Consistent Distribution Matching with Cache-Aware Training for Fast Video Generation</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Xiahong Wang, Dailan He, Yushi Huang, Yi Zhang, Xingtong Ge

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.03118) • [📄 arXiv](https://arxiv.org/abs/2604.03118) • [📥 PDF](https://arxiv.org/pdf/2604.03118)

**💻 Code:** [⭐ Code](https://github.com/XingtongGe/Salt)

> Distilling video generation models to extremely low inference budgets (e.g., 2--4 NFEs) is crucial for real-time deployment, yet remains challenging. Trajectory-style consistency distillation often becomes conservative under complex video dynamics...

</details>

<details>
<summary><b>15. CoME-VL: Scaling Complementary Multi-Encoder Vision-Language Learning</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Hisham Cholakkal, Imran Razzak, Xilin He, Komal Kumar, Ankan Deria

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.03231) • [📄 arXiv](https://arxiv.org/abs/2604.03231) • [📥 PDF](https://arxiv.org/pdf/2604.03231)

**💻 Code:** [⭐ Code](https://github.com/mbzuai-oryx/CoME-VL)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Stateful Cross-layer Vision Modulation (2026) Hierarchical Pre-Training of ...

</details>

<details>
<summary><b>16. DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01765) • [📄 arXiv](https://arxiv.org/abs/2604.01765) • [📥 PDF](https://arxiv.org/pdf/2604.01765)

**💻 Code:** [⭐ Code](https://github.com/youngzhou1999/DriveDreamer-Policy)

> Recently, world-action models (WAM) have emerged to bridge vision-language-action (VLA) models and world models, unifying their reasoning and instruction-following capabilities and spatio-temporal world modeling. However, existing WAM approaches o...

</details>

<details>
<summary><b>17. Do World Action Models Generalize Better than VLAs? A Robustness Study</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yintao Ma, Rui Heng Yang, Behnam Rahmati, Zhiyuan Li, Zhanguang Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.22078) • [📄 arXiv](https://arxiv.org/abs/2603.22078) • [📥 PDF](https://arxiv.org/pdf/2603.22078)

> Robot action planning in the real world is challenging as it requires not only understanding the current state of the environment but also predicting how it will evolve in response to actions. Vision-language-action (VLA), which repurpose large-sc...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 17 |
| 📅 Today | [`2026-04-07.json`](data/daily/2026-04-07.json) | 17 |
| 📆 This Week | [`2026-W14.json`](data/weekly/2026-W14.json) | 22 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 134 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-07 | 17 | [View JSON](data/daily/2026-04-07.json) |
| 📄 2026-04-06 | 5 | [View JSON](data/daily/2026-04-06.json) |
| 📄 2026-04-05 | 45 | [View JSON](data/daily/2026-04-05.json) |
| 📄 2026-04-04 | 45 | [View JSON](data/daily/2026-04-04.json) |
| 📄 2026-04-03 | 7 | [View JSON](data/daily/2026-04-03.json) |
| 📄 2026-04-02 | 5 | [View JSON](data/daily/2026-04-02.json) |
| 📄 2026-04-01 | 10 | [View JSON](data/daily/2026-04-01.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W14 | 22 | [View JSON](data/weekly/2026-W14.json) |
| 📅 2026-W13 | 115 | [View JSON](data/weekly/2026-W13.json) |
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 134 | [View JSON](data/monthly/2026-04.json) |
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
