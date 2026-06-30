<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-20-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5015+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">20</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">30</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">612</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5015+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 30, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent</b> ⭐ 34</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.30616) • [📄 arXiv](https://arxiv.org/abs/2606.30616) • [📥 PDF](https://arxiv.org/pdf/2606.30616)

**💻 Code:** [⭐ Code](https://github.com/InternScience/Agents-A1) • [⭐ Code](https://github.com/huggingface)

> 🚀 We are excited to share Agents-A1 from the Shanghai AI Lab. Agents-A1 is a 35B MoE agentic model designed to scale long-horizon scientific and engineering capabilities, rather than simply scaling model parameters. It learns from knowledge-action...

</details>

<details>
<summary><b>2. ReFreeKV: Towards Threshold-Free KV Cache Compression</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2502.16886) • [📄 arXiv](https://arxiv.org/abs/2502.16886) • [📥 PDF](https://arxiv.org/pdf/2502.16886)

**💻 Code:** [⭐ Code](https://github.com/Patrick-Ni/ReFreeKV) • [⭐ Code](https://github.com/huggingface)

> Code is publicly released at: https://github.com/Patrick-Ni/ReFreeKV

</details>

<details>
<summary><b>3. AsyncOPD: How Stale Can On-Policy Distillation Be?</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24143) • [📄 arXiv](https://arxiv.org/abs/2606.24143) • [📥 PDF](https://arxiv.org/pdf/2606.24143)

**💻 Code:** [⭐ Code](https://github.com/furiosa-ai/async-opd) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/furiosa-ai/async-opd

</details>

<details>
<summary><b>4. Bridging VideoQA and Video-Guided Agentic Tasks via Generalized Keyframe Extraction</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Shuojin Yang, Meng-Hao Guo, Runqi Yin, Qingle Liu, Sunqi Fan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.29445) • [📄 arXiv](https://arxiv.org/abs/2606.29445) • [📥 PDF](https://arxiv.org/pdf/2606.29445)

**💻 Code:** [⭐ Code](https://github.com/VG-GUI-TASKER/VG-GUI-TASKER) • [⭐ Code](https://github.com/huggingface)

> Accepted by ECCV 2026.

</details>

<details>
<summary><b>5. TACO: Tool-Augmented Credit Optimization for Agentic Tool Use</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruihan Jin, Fangrui Lv, Hao Gu, Jinyang Wu, Mingkuan Feng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.30251) • [📄 arXiv](https://arxiv.org/abs/2606.30251) • [📥 PDF](https://arxiv.org/pdf/2606.30251)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>6. Orca: The World is in Your Mind</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Runze Xiao, Yanqing Shen, Mingyu Cao, Yuheng Ji, Yihao Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.30534) • [📄 arXiv](https://arxiv.org/abs/2606.30534) • [📥 PDF](https://arxiv.org/pdf/2606.30534)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Orca: An initial instantiation of a general world foundation model

</details>

<details>
<summary><b>7. Monte Carlo Energy Aggregation for Mobile 3D Gaussian Splatting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.30017) • [📄 arXiv](https://arxiv.org/abs/2606.30017) • [📥 PDF](https://arxiv.org/pdf/2606.30017)

**💻 Code:** [⭐ Code](https://github.com/xiaobiaodu/Flux-GS) • [⭐ Code](https://github.com/huggingface)

> Code is available: https://github.com/xiaobiaodu/Flux-GS Project page: https://xiaobiaodu.github.io/flux-gs-project/ You can use the idea of our Flux-GS for commercial use for free. We hope it can foster Gaussian Splatting industry development.

</details>

<details>
<summary><b>8. MIMFlow: Integrating Masked Image Modeling with Normalizing Flows for End-to-End Image Generation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Qiushi Guo, Xinwen Zhang, Shuai Wang, Xiaowei Xu, Y-Sisyphus

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26016) • [📄 arXiv](https://arxiv.org/abs/2606.26016) • [📥 PDF](https://arxiv.org/pdf/2606.26016)

**💻 Code:** [⭐ Code](https://github.com/MCG-NJU/MIMFlow) • [⭐ Code](https://github.com/huggingface)

> Accepted by ECCV2026

</details>

<details>
<summary><b>9. GUICrafter: Weakly-Supervised GUI Agent Leveraging Massive Unannotated Screenshots</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.29705) • [📄 arXiv](https://arxiv.org/abs/2606.29705) • [📥 PDF](https://arxiv.org/pdf/2606.29705)

**💻 Code:** [⭐ Code](https://github.com/fansunqi/GUICrafter) • [⭐ Code](https://github.com/huggingface)

> GUICrafter: Weakly-Supervised GUI Agent Leveraging Massive Unannotated Screenshots and environment interactive signals. Authors are from Tsinghua University and Tencent Hunyuan.

</details>

<details>
<summary><b>10. PolicyGuard: A Dialogue-Grounded Sub-Agent Verifier for Policy Adherence in LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.29225) • [📄 arXiv](https://arxiv.org/abs/2606.29225) • [📥 PDF](https://arxiv.org/pdf/2606.29225)

**💻 Code:** [⭐ Code](https://github.com/erjui/PolicyGuard) • [⭐ Code](https://github.com/huggingface)

> PolicyGuard: A Dialogue-Grounded Sub-Agent Verifier for Policy Adherence in LLM Agents

</details>

<details>
<summary><b>11. Cognitive Episodes in LLM Reasoning Traces Enable Interpretable Human Item Difficulty Prediction</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Hong Jiao, Zhuochun Li, Xinyue Zeng, Ming Li, Chenguang Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.28186) • [📄 arXiv](https://arxiv.org/abs/2606.28186) • [📥 PDF](https://arxiv.org/pdf/2606.28186)

**💻 Code:** [⭐ Code](https://github.com/c-steve-wang/Epi2Diff) • [⭐ Code](https://github.com/huggingface)

> Predicting human item difficulty is central to educational assessment, where reliable estimates support fairness and effective test construction. Existing methods often depend on costly human calibration or item-level textual representations, prov...

</details>

<details>
<summary><b>12. Focusing on What Matters: Saliency-Harnessing Accurate Routing for Diffusion MoE</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.26938) • [📄 arXiv](https://arxiv.org/abs/2606.26938) • [📥 PDF](https://arxiv.org/pdf/2606.26938)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A Diffusion MoE Framework with Saliency-Harnessing Accurate Routing

</details>

<details>
<summary><b>13. Nemotron-Labs-Diffusion-Image: Advancing Masked Discrete Diffusion for High-Resolution Image Synthesis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.29814) • [📄 arXiv](https://arxiv.org/abs/2606.29814) • [📥 PDF](https://arxiv.org/pdf/2606.29814)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We propose Nemotron-Labs-Diffusion-Image, a state-of-the-art masked discrete diffusion model (MDM) for high-resolution text-to-image synthesis. Compared with prior work on masked image generation, Nemotron-Labs-Diffusion-Image addresses two key ch...

</details>

<details>
<summary><b>14. How Good Can Linear Models Be for Time-Series Forecasting?</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.27282) • [📄 arXiv](https://arxiv.org/abs/2606.27282) • [📥 PDF](https://arxiv.org/pdf/2606.27282)

**💻 Code:** [⭐ Code](https://github.com/SakanaAI/SearchCast) • [⭐ Code](https://github.com/huggingface)

> Linear models aren't weak; their preprocessing is under-tuned . Closed-form Ridge, with only preprocessing tuned (L, r, α, augmentation), matches/beats Transformer, MLP & CNN baselines on 6/8 long-horizon benchmarks at orders-of-magnitude lower co...

</details>

<details>
<summary><b>15. ReasoningLens: Hierarchical Visualization and Diagnostic Auditing for Large Reasoning Models</b> ⭐ 24</summary>

<br/>

**👥 Authors:** Hongyu Lin, Yaojie Lu, Boxi Cao, Jiasheng Zheng, Jun Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.23404) • [📄 arXiv](https://arxiv.org/abs/2606.23404) • [📥 PDF](https://arxiv.org/pdf/2606.23404)

**💻 Code:** [⭐ Code](https://github.com/icip-cas/ReasoningLens) • [⭐ Code](https://github.com/huggingface)

> Long-form reasoning (CoT) is a double-edged sword. While reasoning models  are smarter than ever, debugging a 10,000-token reasoning trace is a nightmare. ReasoningLens turns that "Wall of Text" into an interactive, hierarchical map.

</details>

<details>
<summary><b>16. Walking in the Implicit: Interactive World Exploration via Neural Scene Representation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Cong Qiu, Hangning Zhou, Zhenhua Du, Chengrui Dong, Zhiqi Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.30045) • [📄 arXiv](https://arxiv.org/abs/2606.30045) • [📥 PDF](https://arxiv.org/pdf/2606.30045)

**💻 Code:** [⭐ Code](https://github.com/WU-CVGL/NeuWorld) • [⭐ Code](https://github.com/huggingface)

> Our system rolls out a fixed-length, renderable Neural Implicit Scene state and renders queried observations under camera control.

</details>

<details>
<summary><b>17. PoseShield: Neural Collision Fields for Human Self-Collision Resolution</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Miaolan Xie, Liangyan Gui, Yifan Shen, Zeyun Deng, Zhengyuan Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.29686) • [📄 arXiv](https://arxiv.org/abs/2606.29686) • [📥 PDF](https://arxiv.org/pdf/2606.29686)

**💻 Code:** [⭐ Code](https://github.com/lzhyu/PoseShield) • [⭐ Code](https://github.com/huggingface)

> Self-collision remains a persistent challenge in SMPL-based human pose estimation and motion generation. Under extreme articulations or stochastic motion synthesis, generated meshes frequently exhibit self-penetrations, leading to physically impla...

</details>

<details>
<summary><b>18. OSWorld2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiayang Sun, Weiming Wu, Xinzhuang Xiong, Zilong Zhou, Mengqi Yuan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.29537) • [📄 arXiv](https://arxiv.org/abs/2606.29537) • [📥 PDF](https://arxiv.org/pdf/2606.29537)

**💻 Code:** [⭐ Code](https://github.com/xlang-ai/OSWorld-V2) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>19. SafePyramid: A Hierarchical Benchmark for In-context Policy Guardrailing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.29887) • [📄 arXiv](https://arxiv.org/abs/2606.29887) • [📥 PDF](https://arxiv.org/pdf/2606.29887)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/bytedance/safepyramid)

> No abstract available.

</details>

<details>
<summary><b>20. One Forward Beats Two: InnerZoom for Accurate and Efficient GUI Grounding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.30084) • [📄 arXiv](https://arxiv.org/abs/2606.30084) • [📥 PDF](https://arxiv.org/pdf/2606.30084)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 20 |
| 📅 Today | [`2026-06-30.json`](data/daily/2026-06-30.json) | 20 |
| 📆 This Week | [`2026-W26.json`](data/weekly/2026-W26.json) | 30 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 612 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-30 | 20 | [View JSON](data/daily/2026-06-30.json) |
| 📄 2026-06-29 | 10 | [View JSON](data/daily/2026-06-29.json) |
| 📄 2026-06-28 | 25 | [View JSON](data/daily/2026-06-28.json) |
| 📄 2026-06-27 | 25 | [View JSON](data/daily/2026-06-27.json) |
| 📄 2026-06-26 | 12 | [View JSON](data/daily/2026-06-26.json) |
| 📄 2026-06-25 | 13 | [View JSON](data/daily/2026-06-25.json) |
| 📄 2026-06-24 | 15 | [View JSON](data/daily/2026-06-24.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W26 | 30 | [View JSON](data/weekly/2026-W26.json) |
| 📅 2026-W25 | 108 | [View JSON](data/weekly/2026-W25.json) |
| 📅 2026-W24 | 130 | [View JSON](data/weekly/2026-W24.json) |
| 📅 2026-W23 | 166 | [View JSON](data/weekly/2026-W23.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
| 🗓️ 2026-02 | 1048 | [View JSON](data/monthly/2026-02.json) |
| 🗓️ 2026-01 | 781 | [View JSON](data/monthly/2026-01.json) |

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
