<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-21-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6885+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">40</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">277</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6885+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 06, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. ALoDLM: Adaptively Looped Diffusion Language Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04198) • [📄 arXiv](https://arxiv.org/abs/2610.04198) • [📥 PDF](https://arxiv.org/pdf/2610.04198)

**💻 Code:** [⭐ Code](https://github.com/amazon-science/ALoDLM) • [⭐ Code](https://github.com/huggingface)

> Diffusion language models (DLMs) enable fast generation by predicting multiple tokens in parallel, but their practical adoption remains limited by a persistent quality gap relative to comparably sized autoregressive (AR) models. We attribute this ...

</details>

<details>
<summary><b>2. Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05608) • [📄 arXiv](https://arxiv.org/abs/2610.05608) • [📥 PDF](https://arxiv.org/pdf/2610.05608)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kandinskylab/kandinsky-6)

> 🎬 Kandinsky 6.0 Video — 3B Lite / 29B Pro for synchronized text/image-to-audio-video , with 44 kHz audio, lip-sync and Full-HD super-resolution. 🧠 Uses a dual-stream CrossDiT with bidirectional audio↔video attention, followed by SFT, RL and 10-ste...

</details>

<details>
<summary><b>3. ASCENT: Online Test-Time Training of Long-Horizon Agents via Self-Distillation of Verified Experience</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dong Gong, Haodong Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05303) • [📄 arXiv](https://arxiv.org/abs/2610.05303) • [📥 PDF](https://arxiv.org/pdf/2610.05303)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> ASCENT lets an LLM agent keep learning while it is deployed. In Online Agentic Test-Time Training (OaTTT), the agent executes each task once in one pass over its task stream, and that single attempt with its verification result is the only learnin...

</details>

<details>
<summary><b>4. CANOPY: Adaptive-Granularity Evidence Compression for Multimodal RAG</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00923) • [📄 arXiv](https://arxiv.org/abs/2610.00923) • [📥 PDF](https://arxiv.org/pdf/2610.00923)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Multimodal RAG decides which items to retrieve, but not how much of each item the reader actually needs. We introduce CANOPY, which represents each retrieved text, table, or video as a hierarchy of original regions and uses a fine-tuned node encod...

</details>

<details>
<summary><b>5. OSWorld-Pro: Process-based Evaluation for Computer Use Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.24890) • [📄 arXiv](https://arxiv.org/abs/2609.24890) • [📥 PDF](https://arxiv.org/pdf/2609.24890)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> OSWorld-Pro: Process-based Evaluation for Computer Use Agents

</details>

<details>
<summary><b>6. Rethinking Long-Video Efficiency: A Joint Allocation Perspective on Frames, Pixels, and Front-End Latency</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04318) • [📄 arXiv](https://arxiv.org/abs/2610.04318) • [📥 PDF](https://arxiv.org/pdf/2610.04318)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🎬 LoHi (NeurIPS 2026) : Rethinking long-video efficiency. 💡 Three lessons 🎞️ More frames, not more pixels : at the same token budget, dense low-resolution frames beat sparse native-resolution frames. 🔍 Resolution is task-dependent : most questions...

</details>

<details>
<summary><b>7. Towards Looped Models Done Right, Part II: Rethinking at Fixed Points</b> ⭐ 29</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06833) • [📄 arXiv](https://arxiv.org/abs/2610.06833) • [📥 PDF](https://arxiv.org/pdf/2610.06833)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](github.com/ifm-ai/xllm-loop) • [⭐ Code](https://github.com/ifm-ai/xllm-loop)

> Scaling up a model has meant paying twice, in compute and in memory.We show that looped models can pay in compute alone, using the loop's fixed point as a shortcut. A 1.6B looped model runs twelve blocks deep on four blocks of memory. On the same ...

</details>

<details>
<summary><b>8. RealtimeWAM: One-Step Asynchronous World Action Models</b> ⭐ 2.88k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06617) • [📄 arXiv](https://arxiv.org/abs/2610.06617) • [📥 PDF](https://arxiv.org/pdf/2610.06617)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ModelTC/LightX2V)

> An extremely efficient one-step asynchronous WAM for real-time world modeling. Achieves up to 25× speedup over existing WAMs with less than 1% accuracy degradation .

</details>

<details>
<summary><b>9. Base Models Can Reason By Taking a Cue From Training Data</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06851) • [📄 arXiv](https://arxiv.org/abs/2610.06851) • [📥 PDF](https://arxiv.org/pdf/2610.06851)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sophicle/cues)

> Two opening tokens can bring a base model’s reasoning performance close to that of its RL-trained counterpart. These token cues come from associations learned during training, and RL makes effective cues more likely. Changing those associations ca...

</details>

<details>
<summary><b>10. PaLoRA: Paced Low-Rank Adaptation for Continual Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hao Tang, Fanhu Zeng, Yuxuan Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04226) • [📄 arXiv](https://arxiv.org/abs/2610.04226) • [📥 PDF](https://arxiv.org/pdf/2610.04226)

**💻 Code:** [⭐ Code](https://github.com/liyuxuan-github/PaLoRA) • [⭐ Code](https://github.com/huggingface)

> PaLoRA introduces rank-aware pacing for parameter-efficient continual learning, adaptively controlling low-rank updates according to the effective rank of accumulated knowledge. Combined with adaptive SVD truncation and null-space gradient project...

</details>

<details>
<summary><b>11. Prism: Dynamic Sparse Attention for Native 2K Joint Video-Audio Generation Model Training</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05416) • [📄 arXiv](https://arxiv.org/abs/2610.05416) • [📥 PDF](https://arxiv.org/pdf/2610.05416)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent-Hunyuan/Prism)

> Natively training joint video-audio generation models at higher resolutions empowers them to learn richer visual details and sharper motion dynamics. However, full attention incurs quadratic cost and, as resolution increases, spreads attention ove...

</details>

<details>
<summary><b>12. Training Numerical Intelligence via Auto-Diagnosis and Skill Discovery</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wotao Yin, Peter Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03872) • [📄 arXiv](https://arxiv.org/abs/2610.03872) • [📥 PDF](https://arxiv.org/pdf/2610.03872)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> AI agents are becoming increasingly capable of generating scientific code, but generating code is not the same as improving the algorithms behind it. For numerical solvers, execution feedback can expose poor performance, but rarely reveals its und...

</details>

<details>
<summary><b>13. Dynamic Harness Search: Building Multi-Agent Systems Per-Query via Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04137) • [📄 arXiv](https://arxiv.org/abs/2610.04137) • [📥 PDF](https://arxiv.org/pdf/2610.04137)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>14. When to Switch: Reliable Action-Chunk Extension for Vision-Language-Action Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05719) • [📄 arXiv](https://arxiv.org/abs/2610.05719) • [📥 PDF](https://arxiv.org/pdf/2610.05719)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Seonghoon-Yu/RACE-VLA)

> Real-robot demo Why do longer action chunks become unreliable in VLAs? We find that action errors are not evenly distributed—they spike around transitions between manipulation subskills, and these spikes grow as the chunk gets longer. RACE explici...

</details>

<details>
<summary><b>15. PluginRSI: Recursive Improvement of Agent Harnesses with Reusable Plugins</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yueqing Sun, Jiayuan Zhang, Yuxin Chen, Yuchun Miao, Yaorui Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32423)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>16. Arm-wise Compositional Generalization in Dual-Arm Vision-Language-Action Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yifan Wang, Zhongbo Zhang, Yuhan Wu, Binghao Ran, Zaibin Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06184)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>17. The Missing Primitive: Diagnosing and Repairing Mathematical Reasoning in Large Language Models</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02191)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>18. Noise Out, Bias In: Targeted Bias Injection in Diffusion Language Models via Closed-Loop Activation Steering</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05894)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>19. Certification of Real Images through Calibrated Content Authentication</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.05870)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>20. OmniConfess: Eliciting Token Confessions to Mitigate Omni-Modal Hallucination</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Kaiwen Xue, Zhonghong Ou, Hui Feng, Haoran Luo, Huiqiang Rong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02999)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>21. InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Anatulya Nandi, Liuyu Bian, Jinhong Li, Sirui Xu, Yucheng Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06850)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 21 |
| 📅 Today | [`2026-10-06.json`](data/daily/2026-10-06.json) | 21 |
| 📆 This Week | [`2026-W40.json`](data/weekly/2026-W40.json) | 40 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 277 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-06 | 21 | [View JSON](data/daily/2026-10-06.json) |
| 📄 2026-10-05 | 19 | [View JSON](data/daily/2026-10-05.json) |
| 📄 2026-10-04 | 84 | [View JSON](data/daily/2026-10-04.json) |
| 📄 2026-10-03 | 84 | [View JSON](data/daily/2026-10-03.json) |
| 📄 2026-10-02 | 35 | [View JSON](data/daily/2026-10-02.json) |
| 📄 2026-10-01 | 34 | [View JSON](data/daily/2026-10-01.json) |
| 📄 2026-09-30 | 45 | [View JSON](data/daily/2026-09-30.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W40 | 40 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 277 | [View JSON](data/monthly/2026-10.json) |
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
