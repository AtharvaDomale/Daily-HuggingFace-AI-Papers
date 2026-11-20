<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8091+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">84</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">363</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8091+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 20, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. WeVisForensics: Self-Evolving Agent for Omni AI-Generated Video Detection</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21373) • [📄 arXiv](https://arxiv.org/abs/2611.21373) • [📥 PDF](https://arxiv.org/pdf/2611.21373)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/WeVisForensics)

> We present WeVisForensics, an agentic framework for AI-generated video detection in which a main investigator agent coordinates a self-evolving library of forensic inspector sub-agents spanning visual and audio cues. Github: https://github.com/Ten...

</details>

<details>
<summary><b>2. TuringSpace: Grounding Spatial Intelligence in Action with Interactive CAPTCHAs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25387) • [📄 arXiv](https://arxiv.org/abs/2611.25387) • [📥 PDF](https://arxiv.org/pdf/2611.25387)

**💻 Code:** [⭐ Code](https://github.com/os-cua-framework/TuringSpace) • [⭐ Code](https://github.com/huggingface)

> MLLMs perform well on 2D perception but remain far behind humans in 3D spatial understanding, which makes space a natural basis for CAPTCHAs now that computer-use agents solve conventional ones. Yet existing spatial CAPTCHAs and benchmarks are mos...

</details>

<details>
<summary><b>3. The Mask Is Not the Object: Volumetric Supervision for 3D Gaussian Segmentation</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21383) • [📄 arXiv](https://arxiv.org/abs/2611.21383) • [📥 PDF](https://arxiv.org/pdf/2611.21383)

**💻 Code:** [⭐ Code](https://github.com/jhxcho/voda) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>4. WAM-Policy: World Action Models with On-Policy Distillation for Quadrupedal Loco-Manipulation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.24011) • [📄 arXiv](https://arxiv.org/abs/2611.24011) • [📥 PDF](https://arxiv.org/pdf/2611.24011)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIGeeksGroup/WAM-Policy)

> open-sourced.

</details>

<details>
<summary><b>5. Coda-DaVinci: Reimagining Video Generation with Code2Video</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26595) • [📄 arXiv](https://arxiv.org/abs/2611.26595) • [📥 PDF](https://arxiv.org/pdf/2611.26595)

**💻 Code:** [⭐ Code](https://github.com/FrankYang-17/Coda) • [⭐ Code](https://github.com/huggingface)

> Code2Video uses agent-authored 3D proxies to guide video generation, enabling explicit control over structure and dynamics. However, how to construct proxies and prompts for optimal downstream quality remains underexplored, as do the capabilities ...

</details>

<details>
<summary><b>6. WeMM-Video: Pushing the Frontiers of Open-Source Omni-Modal Video Understanding</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26409) • [📄 arXiv](https://arxiv.org/abs/2611.26409) • [📥 PDF](https://arxiv.org/pdf/2611.26409)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/WeMM-Video)

> Technical report of the WeMM-Video family.

</details>

<details>
<summary><b>7. On-Policy Morphing into Hybrid Attention Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.24809) • [📄 arXiv](https://arxiv.org/abs/2611.24809) • [📥 PDF](https://arxiv.org/pdf/2611.24809)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hybrid attention models combine full and linear attention to achieve a better quality--efficiency trade-off. Converting pretrained Transformers into such models requires identifying which layers should retain full attention under a prescribed budg...

</details>

<details>
<summary><b>8. Posterior Matching Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26629) • [📄 arXiv](https://arxiv.org/abs/2611.26629) • [📥 PDF](https://arxiv.org/pdf/2611.26629)

**💻 Code:** [⭐ Code](https://github.com/yesiltepe-hidir/PMD) • [⭐ Code](https://github.com/huggingface)

> Project Page: https://pmd-distillation.github.io/

</details>

<details>
<summary><b>9. EgoCaptioner: Towards Comprehensive and Fine-Grained Egocentric Video Captioning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26380) • [📄 arXiv](https://arxiv.org/abs/2611.26380) • [📥 PDF](https://arxiv.org/pdf/2611.26380)

**💻 Code:** [⭐ Code](https://github.com/cactusqq/EgoCaptioner) • [⭐ Code](https://github.com/huggingface)

> Egocentric video captioning requires fine-grained detail and accurate timing, yet current multimodal large language models (MLLMs) often miss brief actions, confuse their order, or misinterpret the wearer's motion. Existing caption formats also of...

</details>

<details>
<summary><b>10. AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26510) • [📄 arXiv](https://arxiv.org/abs/2611.26510) • [📥 PDF](https://arxiv.org/pdf/2611.26510)

**💻 Code:** [⭐ Code](https://github.com/RUCAIBox/AutoMaster) • [⭐ Code](https://github.com/huggingface)

> Task success does not guarantee good teaching . Even successful agent trajectories may contain undesirable behaviors that SFT can inadvertently reinforce. We introduce AutoMaster , an autonomous post-training framework that makes the same model a ...

</details>

<details>
<summary><b>11. WM^MLLM: Turning MLLMs into World Models</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.24193) • [📄 arXiv](https://arxiv.org/abs/2611.24193) • [📥 PDF](https://arxiv.org/pdf/2611.24193)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ztMotaLee/WM_mllm)

> WM^MLLM rethinks world modeling by moving the world state out of the generator. The scene is stored as an explicit panorama with version history, and MLLMs plan, place and verify each edit. This makes consistency a built-in guarantee: revisits are...

</details>

<details>
<summary><b>12. VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.22666) • [📄 arXiv](https://arxiv.org/abs/2611.22666) • [📥 PDF](https://arxiv.org/pdf/2611.22666)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/suho-yoo/VSpark)

> Our aim is to accelerate autoregressive vision-language-action models (VLAs), which generate robot actions one token at a time. Speculative decoding makes this possible: a lightweight draft model proposes several tokens at once, and the VLA verifi...

</details>

<details>
<summary><b>13. CompoundingKernel: Self-Improving from Zero to First</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.19230) • [📄 arXiv](https://arxiv.org/abs/2611.19230) • [📥 PDF](https://arxiv.org/pdf/2611.19230)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A self-improving agent system that writes GPU kernels by search, starting from only the PyTorch spec. As of Nov 11, 2026 it was #1 on all three public SOL-ExecBench leaderboards (L1, L2, Quantization) on NVIDIA B200: all 235 kernels correct on the...

</details>

<details>
<summary><b>14. DailySafe: Benchmarking Dynamic Safety Awareness in Everyday Multi-Turn Multimodal Interactions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Siyuan Liang, Ziheng Peng, Wenzhang Jiang, Xinhao Deng, Xuankun Rong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26298) • [📄 arXiv](https://arxiv.org/abs/2611.26298) • [📥 PDF](https://arxiv.org/pdf/2611.26298)

**💻 Code:** [⭐ Code](https://github.com/XuankunRong/DailySafe) • [⭐ Code](https://github.com/huggingface)

> As multimodal large language models (MLLMs) become increasingly integrated into everyday life, their ability to provide safe and reliable assistance in realistic daily activities remains insufficiently explored. Existing safety benchmarks primaril...

</details>

<details>
<summary><b>15. NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.23835) • [📄 arXiv](https://arxiv.org/abs/2611.23835) • [📥 PDF](https://arxiv.org/pdf/2611.23835)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Speculative decoding is widely used to speed up generation in large language models (LLMs), with a small draft model proposing tokens that one forward pass of the target model verifies. Draft models that read the target's hidden states, such as DF...

</details>

<details>
<summary><b>16. Suppressing the Shortcut Hidden Behind the Gains of VideoLLM Fine-Tuning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junmo Kim, Jongsuk Kim, Youngwoo Shin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25776) • [📄 arXiv](https://arxiv.org/abs/2611.25776) • [📥 PDF](https://arxiv.org/pdf/2611.25776)

**💻 Code:** [⭐ Code](https://github.com/Youngwoo-git/VETO) • [⭐ Code](https://github.com/huggingface)

> SFT raises the accuracy of VideoLLMs, yet behind this rise the model increasingly relies on textual regularities of the training data, a shortcut that bypasses the video. VETO suppresses this shortcut by projecting the fine-tuning update onto the ...

</details>

<details>
<summary><b>17. PlayCraft: Designing Engaging Games from Agent Play Experience</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Weihao Cheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26624) • [📄 arXiv](https://arxiv.org/abs/2611.26624) • [📥 PDF](https://arxiv.org/pdf/2611.26624)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/xt4d/PlayCraft)

> PlayCraft is an automated framework for designing engaging games that give players a path from beginner play to mastery. PlayCraft alternates between playing and crafting: A player agent watches the game screen, uses the controls, and pursues the ...

</details>

<details>
<summary><b>18. Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26603) • [📄 arXiv](https://arxiv.org/abs/2611.26603) • [📥 PDF](https://arxiv.org/pdf/2611.26603)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/apple-aiml-research/ml-depth-pro-2)

> Depth Pro 2 estimates the depth of every pixel, in meters, from images taken with an ordinary camera. One network handles a single photo, a whole video, or a live camera feed, at the input's full resolution. Its depth is sharp at object edges and ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-11-20.json`](data/daily/2026-11-20.json) | 18 |
| 📆 This Week | [`2026-W46.json`](data/weekly/2026-W46.json) | 84 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 363 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-20 | 18 | [View JSON](data/daily/2026-11-20.json) |
| 📄 2026-11-19 | 27 | [View JSON](data/daily/2026-11-19.json) |
| 📄 2026-11-18 | 17 | [View JSON](data/daily/2026-11-18.json) |
| 📄 2026-11-17 | 10 | [View JSON](data/daily/2026-11-17.json) |
| 📄 2026-11-16 | 12 | [View JSON](data/daily/2026-11-16.json) |
| 📄 2026-11-15 | 34 | [View JSON](data/daily/2026-11-15.json) |
| 📄 2026-11-14 | 34 | [View JSON](data/daily/2026-11-14.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W46 | 84 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 363 | [View JSON](data/monthly/2026-11.json) |
| 🗓️ 2026-10 | 1120 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |

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
