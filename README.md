<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-17-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-672+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">118</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">721</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">672+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 26, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times</b> ⭐ 1.96k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16093) • [📄 arXiv](https://arxiv.org/abs/2512.16093) • [📥 PDF](https://arxiv.org/pdf/2512.16093)

**💻 Code:** [⭐ Code](https://github.com/thu-ml/SageAttention) • [⭐ Code](https://github.com/thu-ml/SLA) • [⭐ Code](https://github.com/thu-ml/TurboDiffusion)

> TurboDiffusion : 100–200× acceleration in video generation on a single RTX 5090. A high-quality 5-second video can be generated in just 1.9 seconds . Efficient inference code, as well as model parameters (checkpoints) for TurboWan2.2/2.1 for Text-...

</details>

<details>
<summary><b>2. Learning to Reason in 4D: Dynamic Spatial Understanding for Vision Language Models</b> ⭐ 28</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20557) • [📄 arXiv](https://arxiv.org/abs/2512.20557) • [📥 PDF](https://arxiv.org/pdf/2512.20557)

**💻 Code:** [⭐ Code](https://github.com/TencentARC/DSR_Suite)

> DSR Suite delivers scalable 4D training/evaluation from real-world videos and a lightweight GSM module that injects targeted geometric priors into VLMs, markedly boosting dynamic spatial reasoning while preserving general video understanding. Key ...

</details>

<details>
<summary><b>3. DreaMontage: Arbitrary Frame-Guided One-Shot Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21252) • [📄 arXiv](https://arxiv.org/abs/2512.21252) • [📥 PDF](https://arxiv.org/pdf/2512.21252)

> In this paper, we introduce DreaMontage, a comprehensive framework designed for arbitrary frame-guided generation, capable of synthesizing seamless, expressive, and long-duration one-shot videos from diverse user-provided inputs. To achieve this, ...

</details>

<details>
<summary><b>4. T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21094) • [📄 arXiv](https://arxiv.org/abs/2512.21094) • [📥 PDF](https://arxiv.org/pdf/2512.21094)

**💻 Code:** [⭐ Code](https://github.com/NJU-LINK/T2AV-Compass/)

> Text-to-Audio-Video (T2AV) generation aims to synthesize temporally coherent video and semantically synchronized audio from natural language, yet its evaluation remains fragmented, often relying on unimodal metrics or narrowly scoped benchmarks th...

</details>

<details>
<summary><b>5. Beyond Memorization: A Multi-Modal Ordinal Regression Benchmark to Expose Popularity Bias in Vision-Language Models</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Yu-Lun Liu, He Syu, Chia-Jui Chang, Ting-Lin Wu, Li-Zhong Szu-Tu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21337) • [📄 arXiv](https://arxiv.org/abs/2512.21337) • [📥 PDF](https://arxiv.org/pdf/2512.21337)

**💻 Code:** [⭐ Code](https://github.com/Sytwu/BeyondMemo)

> We expose a significant popularity bias in state-of-the-art vision-language models (VLMs), which achieve up to 34% higher accuracy on famous buildings compared to ordinary ones, indicating a reliance on memorization over generalizable understandin...

</details>

<details>
<summary><b>6. HiStream: Efficient High-Resolution Video Generation via Redundancy-Eliminated Streaming</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21338) • [📄 arXiv](https://arxiv.org/abs/2512.21338) • [📥 PDF](https://arxiv.org/pdf/2512.21338)

**💻 Code:** [⭐ Code](https://github.com/arthur-qiu/HiStream)

> HiStream is an efficient autoregressive framework for high-resolution video generation that removes the quadratic inference bottleneck of diffusion models by reducing spatial, temporal, and timestep redundancy. It achieves state-of-the-art quality...

</details>

<details>
<summary><b>7. Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20848) • [📄 arXiv](https://arxiv.org/abs/2512.20848) • [📥 PDF](https://arxiv.org/pdf/2512.20848)

> Nemotron 3 Nano is a 30B mixture-of-experts hybrid Mamba-Transformer enabling agentic reasoning with 1M context, outperforming models in throughput and accuracy while using only a fraction of parameters per pass.

</details>

<details>
<summary><b>8. NVIDIA Nemotron 3: Efficient and Open Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20856) • [📄 arXiv](https://arxiv.org/abs/2512.20856) • [📥 PDF](https://arxiv.org/pdf/2512.20856)

> Nemotron 3 introduces Mixture-of-Experts Mamba-Transformer with 1M context, LatentMoE, MTP layers, and multi-environment RL for agentic reasoning and tool use, with open weights.

</details>

<details>
<summary><b>9. TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20757) • [📄 arXiv](https://arxiv.org/abs/2512.20757) • [📥 PDF](https://arxiv.org/pdf/2512.20757)

**💻 Code:** [⭐ Code](https://github.com/r-three/Tokenizers)

> No abstract available.

</details>

<details>
<summary><b>10. Learning from Next-Frame Prediction: Autoregressive Video Modeling Encodes Effective Representations</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21004) • [📄 arXiv](https://arxiv.org/abs/2512.21004) • [📥 PDF](https://arxiv.org/pdf/2512.21004)

**💻 Code:** [⭐ Code](https://github.com/Singularity0104/NExT-Vid)

> Code: https://github.com/Singularity0104/NExT-Vid

</details>

<details>
<summary><b>11. From Word to World: Can Large Language Models be Implicit Text-based World Models?</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18832) • [📄 arXiv](https://arxiv.org/abs/2512.18832) • [📥 PDF](https://arxiv.org/pdf/2512.18832)

**💻 Code:** [⭐ Code](https://github.com/X1AOX1A/Word2World)

> Explore the foundation of text-based world model

</details>

<details>
<summary><b>12. DramaBench: A Six-Dimensional Evaluation Framework for Drama Script Continuation</b> ⭐ 44</summary>

<br/>

**👥 Authors:** Yunqi Huang, jackylin2012, FutureMa

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.19012) • [📄 arXiv](https://arxiv.org/abs/2512.19012) • [📥 PDF](https://arxiv.org/pdf/2512.19012)

**💻 Code:** [⭐ Code](https://github.com/IIIIQIIII/DramaBench)

> Hi everyone! I'm Shijian, the first author of DramaBench. We're excited to share our work on evaluating creative writing, specifically drama script continuation. 🎭 Why DramaBench? Traditional "LLM-as-a-Judge" metrics often suffer from subjectivity...

</details>

<details>
<summary><b>13. Streaming Video Instruction Tuning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kaiyang Zhou, Xing Sun, Mengdan Zhang, Peixian Chen, Jiaer Xia

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21334) • [📄 arXiv](https://arxiv.org/abs/2512.21334) • [📥 PDF](https://arxiv.org/pdf/2512.21334)

> We present Streamo, a real-time streaming video LLM that serves as a general-purpose interactive assistant. Unlike existing online video models that focus narrowly on question answering or captioning, Streamo performs a broad spectrum of streaming...

</details>

<details>
<summary><b>14. Multi-hop Reasoning via Early Knowledge Alignment</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xuanjing Huang, Qi Luo, Bo Wang, Shicheng Fang, Yuxin Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.20144) • [📄 arXiv](https://arxiv.org/abs/2512.20144) • [📥 PDF](https://arxiv.org/pdf/2512.20144)

> Multi-hop Reasoning via Early Knowledge Alignment

</details>

<details>
<summary><b>15. SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nghi D. Q. Bui, Huy Phan Nhat, Dung Nguyen Manh, Tue Le, Minh V. T. Thai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.18470) • [📄 arXiv](https://arxiv.org/abs/2512.18470) • [📥 PDF](https://arxiv.org/pdf/2512.18470)

> Existing benchmarks for AI coding agents focus on isolated, single-issue tasks such as fixing a bug or implementing a small feature. However, real-world software engineering is fundamentally a long-horizon endeavor: developers must interpret high-...

</details>

<details>
<summary><b>16. PhononBench:A Large-Scale Phonon-Based Benchmark for Dynamical Stability in Crystal Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21227) • [📄 arXiv](https://arxiv.org/abs/2512.21227) • [📥 PDF](https://arxiv.org/pdf/2512.21227)

> No abstract available.

</details>

<details>
<summary><b>17. LLM Swiss Round: Aggregating Multi-Benchmark Performance via Competitive Swiss-System Dynamics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21010) • [📄 arXiv](https://arxiv.org/abs/2512.21010) • [📥 PDF](https://arxiv.org/pdf/2512.21010)

> Proposes Competitive Swiss-System Dynamics to rank LLMs across multiple benchmarks using dynamic pairings, Monte Carlo Estimated Win Score, and failure sensitivity analysis for risk-aware evaluation.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 17 |
| 📅 Today | [`2025-12-26.json`](data/daily/2025-12-26.json) | 17 |
| 📆 This Week | [`2025-W51.json`](data/weekly/2025-W51.json) | 118 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 721 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-26 | 17 | [View JSON](data/daily/2025-12-26.json) |
| 📄 2025-12-25 | 18 | [View JSON](data/daily/2025-12-25.json) |
| 📄 2025-12-24 | 23 | [View JSON](data/daily/2025-12-24.json) |
| 📄 2025-12-23 | 22 | [View JSON](data/daily/2025-12-23.json) |
| 📄 2025-12-22 | 38 | [View JSON](data/daily/2025-12-22.json) |
| 📄 2025-12-21 | 38 | [View JSON](data/daily/2025-12-21.json) |
| 📄 2025-12-20 | 37 | [View JSON](data/daily/2025-12-20.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W51 | 118 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 721 | [View JSON](data/monthly/2025-12.json) |

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
