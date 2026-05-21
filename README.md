<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4078+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">67</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">457</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4078+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 21, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19660) • [📄 arXiv](https://arxiv.org/abs/2605.19660) • [📥 PDF](https://arxiv.org/pdf/2605.19660)

**💻 Code:** [⭐ Code](https://github.com/ZunhaiSu/OScaR-KV-Quant) • [⭐ Code](https://github.com/huggingface)

> OScaR is an accurate, lightweight INT2 KV quantization method that establishes a new accuracy-efficiency Pareto frontier. It operates without data, training, or calibration, offering plug-and-play compatibility for X-LLMs (i.e., text-only, multi-m...

</details>

<details>
<summary><b>2. Evaluating Temporal Semantic Caching and Workflow Optimization in Agentic Plan-Execute Pipelines</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20630) • [📄 arXiv](https://arxiv.org/abs/2605.20630) • [📥 PDF](https://arxiv.org/pdf/2605.20630)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Industrial asset operations workflows are latency-sensitive because a single user query may require coordination over sensor data, work orders, failure modes, fore- casting tools, and domain-specific agents. We evaluate this problem on Asse- tOpsB...

</details>

<details>
<summary><b>3. Generative Recursive Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yoshua Bengio, Mengye Ren, Minsu Kim, Mingyu Jo, Junyeob Baek

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19376) • [📄 arXiv](https://arxiv.org/abs/2605.19376) • [📥 PDF](https://arxiv.org/pdf/2605.19376)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce GRAM, a generative framework for recursive reasoning. Unlike deterministic recursive models such as HRM, TRM, and Looped Transformers, GRAM treats reasoning as a stochastic latent trajectory, allowing the model to explore multiple pos...

</details>

<details>
<summary><b>4. Uni-Edit: Intelligent Editing Is A General Task For Unified Model Tuning</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Kai Zou, Hongbo Liu, Hongyu Li, Manyuan Zhang, Dian Zheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21487) • [📄 arXiv](https://arxiv.org/abs/2605.21487) • [📥 PDF](https://arxiv.org/pdf/2605.21487)

**💻 Code:** [⭐ Code](https://github.com/zhengdian1/Uni-Edit) • [⭐ Code](https://github.com/huggingface)

> Uni-Edit is a novel training task for Unified Multimodal Models that integrates image understanding and generation into a single pipeline to achieve simultaneous performance improvements.

</details>

<details>
<summary><b>5. On the limits and opportunities of AI reviewers: Reviewing the reviews of Nature-family papers with 45 expert scientists</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20668) • [📄 arXiv](https://arxiv.org/abs/2605.20668) • [📥 PDF](https://arxiv.org/pdf/2605.20668)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/prometheus-eval/cmu-paper-reviewer)

> With the advancement of AI capabilities, AI reviewers are beginning to be deployed in scientific peer review, yet their capability and credibility remain in question: many scientists simply view them as probabilistic systems without the expertise ...

</details>

<details>
<summary><b>6. UniT: Unified Geometry Learning with Group Autoregressive Transformer</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Xinhu Zheng, Hongliang Lu, Zhaonian Kuang, Yusong Huang, Haotian Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21131) • [📄 arXiv](https://arxiv.org/abs/2605.21131) • [📥 PDF](https://arxiv.org/pdf/2605.21131)

**💻 Code:** [⭐ Code](https://github.com/Wang-xjtu/UniT) • [⭐ Code](https://github.com/huggingface)

> 🚀 UniT: Unified Geometry Learning with Group Autoregressive Transformer UniT is a unified feed-forward model for geometry perception. It reformulates a wide range of geometry perception capabilities into a single framework, covering: Diverse view ...

</details>

<details>
<summary><b>7. OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21343) • [📄 arXiv](https://arxiv.org/abs/2605.21343) • [📥 PDF](https://arxiv.org/pdf/2605.21343)

**💻 Code:** [⭐ Code](https://github.com/FudanCVL/OcclusionFormer) • [⭐ Code](https://github.com/huggingface)

> ICML 2026. We first construct SA-Z, a large-scale dataset enriched with explicit occlusion ordering and pixel-level annotations. Building upon our proposed dataset, we introduce OcclusionFormer, a novel occlusion-aware Diffusion Transformer framew...

</details>

<details>
<summary><b>8. Conditional Equivalence of DPO and RLHF: Implicit Assumption, Failure Modes, and Provable Alignment</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bo Han, Dong Fang, Wei Xue, Yonggang Zhang, Zhiqin Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20834) • [📄 arXiv](https://arxiv.org/abs/2605.20834) • [📥 PDF](https://arxiv.org/pdf/2605.20834)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. HRM-Text: Efficient Pretraining Beyond Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20613) • [📄 arXiv](https://arxiv.org/abs/2605.20613) • [📥 PDF](https://arxiv.org/pdf/2605.20613)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sapientinc/HRM-Text)

> HRM-Text explores a different approach to language model pretraining: hierarchical recurrent computation, task-completion training, and latent-space reasoning. At just 1B parameters, HRM-Text achieves competitive performance with dramatically lowe...

</details>

<details>
<summary><b>10. Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Aidong Zhang, Zhiyong Lu, Sanchit Sinha, Qiao Jin, Guangzhi Xiong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20158) • [📄 arXiv](https://arxiv.org/abs/2605.20158) • [📥 PDF](https://arxiv.org/pdf/2605.20158)

**💻 Code:** [⭐ Code](https://github.com/gzxiong/medfocus) • [⭐ Code](https://github.com/huggingface)

> We propose MedFocus and MedGround-Bench.

</details>

<details>
<summary><b>11. PanoWorld: A Generative Spatial World Model for Consistent Whole-House Panorama Synthesis</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yifeng Shi, Yijiang Hu, Zhenjia Li, JiaJinrang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.17916) • [📄 arXiv](https://arxiv.org/abs/2605.17916) • [📥 PDF](https://arxiv.org/pdf/2605.17916)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jinsong Lan, Jing Wang, Mengting Chen, Zhengze Xu, Jun Zheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21431) • [📄 arXiv](https://arxiv.org/abs/2605.21431) • [📥 PDF](https://arxiv.org/pdf/2605.21431)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Video Virtual Try-On (VVT) aims to seamlessly replace a garment on a person in a video with a new one. While existing methods have made significant strides in maintaining temporal consistency, they are predominantly confined to non-interactive sce...

</details>

<details>
<summary><b>13. DrawMotion: Generating 3D Human Motions by Freehand Drawing</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Jiaming Chu, Qiaozhi He, Zhihua Wu, Lei Jin, Tao Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20955) • [📄 arXiv](https://arxiv.org/abs/2605.20955) • [📥 PDF](https://arxiv.org/pdf/2605.20955)

**💻 Code:** [⭐ Code](https://github.com/InvertedForest/DrawMotion) • [⭐ Code](https://github.com/huggingface)

> DrawMotion introduces a diffusion-based framework for generating 3D human motions from freehand drawings and text, using a Multi-Condition Module for efficient multi-condition fusion and guidance.

</details>

<details>
<summary><b>14. PlanningBench: Generating Scalable and Verifiable Planning Data for Evaluating and Training Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20873) • [📄 arXiv](https://arxiv.org/abs/2605.20873) • [📥 PDF](https://arxiv.org/pdf/2605.20873)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhengyao Jiang, Yuxiang Wu, Dhruv Srikanth, Bingchen Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21384) • [📄 arXiv](https://arxiv.org/abs/2605.21384) • [📥 PDF](https://arxiv.org/pdf/2605.21384)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> SpecBench is a novel benchmark designed to evaluate reward hacking in long-horizon coding agents by identifying performance gaps between visible validation suites and held-out real-world test sets.

</details>

<details>
<summary><b>16. Mem-π: Adaptive Memory through Learning When and What to Generate</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alexandre Lacoste, Christopher Pal, Hadi Nekoei, Chao Wang, Xiaoqiang Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21463) • [📄 arXiv](https://arxiv.org/abs/2605.21463) • [📥 PDF](https://arxiv.org/pdf/2605.21463)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Mem-π is an adaptive memory framework for LLM agents that replaces traditional retrieval with a model-generated, RL-optimized guidance mechanism to improve performance on complex, context-dependent agentic tasks.

</details>

<details>
<summary><b>17. It Takes Two: Complementary Self-Distillation for Contextual Integrity in LLMs</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20258) • [📄 arXiv](https://arxiv.org/abs/2605.20258) • [📥 PDF](https://arxiv.org/pdf/2605.20258)

**💻 Code:** [⭐ Code](https://github.com/sw-programmer/SelfCI) • [⭐ Code](https://github.com/huggingface)

> We introduce SelfCI, a complementary self-distillation framework that aligns LLMs with Contextual Integrity (CI) by jointly optimizing utility-preserving and privacy-preserving objectives, achieving stronger privacy–utility trade-offs than existin...

</details>

<details>
<summary><b>18. LongMINT: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Elias Stengel-Eskin, Zaid Khan, Joykirat Singh, Justin Chih-Yao Chen, Hyunji Lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.18565) • [📄 arXiv](https://arxiv.org/abs/2605.18565) • [📥 PDF](https://arxiv.org/pdf/2605.18565)

**💻 Code:** [⭐ Code](https://github.com/amy-hyunji/MINTEval) • [⭐ Code](https://github.com/huggingface)

> Code and dataset: https://github.com/amy-hyunji/MINTEval

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-05-21.json`](data/daily/2026-05-21.json) | 18 |
| 📆 This Week | [`2026-W20.json`](data/weekly/2026-W20.json) | 67 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 457 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-21 | 18 | [View JSON](data/daily/2026-05-21.json) |
| 📄 2026-05-20 | 16 | [View JSON](data/daily/2026-05-20.json) |
| 📄 2026-05-19 | 15 | [View JSON](data/daily/2026-05-19.json) |
| 📄 2026-05-18 | 18 | [View JSON](data/daily/2026-05-18.json) |
| 📄 2026-05-17 | 53 | [View JSON](data/daily/2026-05-17.json) |
| 📄 2026-05-16 | 53 | [View JSON](data/daily/2026-05-16.json) |
| 📄 2026-05-15 | 26 | [View JSON](data/daily/2026-05-15.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W20 | 67 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 457 | [View JSON](data/monthly/2026-05.json) |
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
