<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3717+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">37</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">96</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3717+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 08, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. MiA-Signature: Approximating Global Activation for Long-Context Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06416) • [📄 arXiv](https://arxiv.org/abs/2605.06416) • [📥 PDF](https://arxiv.org/pdf/2605.06416)

> We believe this work provides a step toward bridging cognitive insights and practical system design, highlighting the importance of global activation in memory-driven reasoning.

</details>

<details>
<summary><b>2. MARBLE: Multi-Aspect Reward Balance for Diffusion RL</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06507) • [📄 arXiv](https://arxiv.org/abs/2605.06507) • [📥 PDF](https://arxiv.org/pdf/2605.06507)

**💻 Code:** [⭐ Code](https://github.com/aim-uofa/MARBLE)

> MARBLE harmonizes reward-specific policy gradients into a single update direction, simultaneously improving all rewards in one training run. No manual reward weighting, no multi-stage curriculum, and at near single-reward training cost. To the bes...

</details>

<details>
<summary><b>3. Continuous Latent Diffusion Language Model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rui Zhu, Shen Nie, Yian Zhao, Qinyu Zhao, Hongcan Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06548) • [📄 arXiv](https://arxiv.org/abs/2605.06548) • [📥 PDF](https://arxiv.org/pdf/2605.06548)

> No abstract available.

</details>

<details>
<summary><b>4. Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05724) • [📄 arXiv](https://arxiv.org/abs/2605.05724) • [📥 PDF](https://arxiv.org/pdf/2605.05724)

**💻 Code:** [⭐ Code](https://github.com/cxcscmu/Auto-Research-Recipes)

> Closed-loop auto research turns agent-written code, real experiments, and evaluator feedback into an autonomous feedback loop that develops non-trivial training recipes.

</details>

<details>
<summary><b>5. ReflectDrive-2: Reinforcement-Learning-Aligned Self-Editing for Discrete Diffusion Driving</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ben Lu, Pengxiang Li, Bihao Cui, Yue Wang, Huimin Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04647) • [📄 arXiv](https://arxiv.org/abs/2605.04647) • [📥 PDF](https://arxiv.org/pdf/2605.04647)

> We started ReflectDrive-2 from a concrete observation: when imitation-learned driving policies fail, they fail along two predictable axes — longitudinal (overshoot, late braking, under-progress) and lateral (lane drift, clipped turns). That kind o...

</details>

<details>
<summary><b>6. Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuyi Yang, Donghong Cai, Jinyuan Li, Chengsong Huang, Langlin Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05566) • [📄 arXiv](https://arxiv.org/abs/2605.05566) • [📥 PDF](https://arxiv.org/pdf/2605.05566)

> prompt space perturbation broadens reasoning exploration

</details>

<details>
<summary><b>7. UniPool: A Globally Shared Expert Pool for Mixture-of-Experts</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06665) • [📄 arXiv](https://arxiv.org/abs/2605.06665) • [📥 PDF](https://arxiv.org/pdf/2605.06665)

**💻 Code:** [⭐ Code](https://github.com/Centaurus-Alpha/UniPool)

> Modern Mixture-of-Experts (MoE) architectures allocate expert capacity through a rigid per-layer rule: each transformer layer owns a separate expert set. This convention couples depth scaling with linear expert-parameter growth and assumes that ev...

</details>

<details>
<summary><b>8. Continuous-Time Distribution Matching for Few-Step Diffusion Distillation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06376) • [📄 arXiv](https://arxiv.org/abs/2605.06376) • [📥 PDF](https://arxiv.org/pdf/2605.06376)

**💻 Code:** [⭐ Code](https://github.com/byliutao/cdm)

> No abstract available.

</details>

<details>
<summary><b>9. A^2TGPO: Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06200) • [📄 arXiv](https://arxiv.org/abs/2605.06200) • [📥 PDF](https://arxiv.org/pdf/2605.06200)

**💻 Code:** [⭐ Code](https://github.com/CuSO4-Chen/A-TGPO)

> Overview we propose A²TGPO ( A gentic T urn- G roup P olicy O ptimization with A daptive Turn-level Clipping), which retains IG as the intrinsic signal but re-designs how it is normalized, accumulated, and consumed: (i) turn-group normalization: n...

</details>

<details>
<summary><b>10. SkillOS: Learning Skill Curation for Self-Evolving Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zifeng Wang, Rujun Han, Yanfei Chen, Jun Yan, Siru Ouyang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06614) • [📄 arXiv](https://arxiv.org/abs/2605.06614) • [📥 PDF](https://arxiv.org/pdf/2605.06614)

> No abstract available.

</details>

<details>
<summary><b>11. Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sipeng Zhang, Xinpeng Wei, Guangchen Lan, Zhaoyang Wang, Tianle Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06638) • [📄 arXiv](https://arxiv.org/abs/2605.06638) • [📥 PDF](https://arxiv.org/pdf/2605.06638)

> 📄 Paper: https://arxiv.org/abs/2605.06638

</details>

<details>
<summary><b>12. The Granularity Axis: A Micro-to-Macro Latent Direction for Social Roles in Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06196) • [📄 arXiv](https://arxiv.org/abs/2605.06196) • [📥 PDF](https://arxiv.org/pdf/2605.06196)

**💻 Code:** [⭐ Code](https://github.com/qinchonghanzuibang/Granularity-Axis)

> We identify the Granularity Axis, a micro-to-macro latent direction that organizes social-role representations in language models and partially steers the scale of model reasoning.

</details>

<details>
<summary><b>13. SwiftI2V: Efficient High-Resolution Image-to-Video Generation via Conditional Segment-wise Generation</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Rui Liu, Yufei Zhao, Wenbo Li, Yuechen Zhang, YaoYang Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06356) • [📄 arXiv](https://arxiv.org/abs/2605.06356) • [📥 PDF](https://arxiv.org/pdf/2605.06356)

**💻 Code:** [⭐ Code](https://github.com/HKUST-LongGroup/SwiftI2V)

> We propose SwiftI2V, an efficient framework for high-resolution (2K) I2V generation that decouples motion modeling from detail synthesis via progressive segment-wise generation and bidirectional contextual interaction. SwiftI2V achieves performanc...

</details>

<details>
<summary><b>14. Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shugui Liu, Yuchun Miao, Zhengxi Lu, Yuxin Chen, Yaorui Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06130) • [📄 arXiv](https://arxiv.org/abs/2605.06130) • [📥 PDF](https://arxiv.org/pdf/2605.06130)

> No abstract available.

</details>

<details>
<summary><b>15. AI Co-Mathematician: Accelerating Mathematicians with Agentic AI</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lars Buesing, Iuliya Beloshapka, Yori Zwols, Ingrid von Glehn, Daniel Zheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06651) • [📄 arXiv](https://arxiv.org/abs/2605.06651) • [📥 PDF](https://arxiv.org/pdf/2605.06651)

> No abstract available.

</details>

<details>
<summary><b>16. TabEmbed: Benchmarking and Learning Generalist Embeddings for Tabular Understanding</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yu Cheng, Xing Fu, Xiaoyi Bao, Mingming Zhang, Minjie Qiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.04962) • [📄 arXiv](https://arxiv.org/abs/2605.04962) • [📥 PDF](https://arxiv.org/pdf/2605.04962)

**💻 Code:** [⭐ Code](https://github.com/qiangminjie27/TabEmbed)

> the generalist embedding model that unifies tabular classification and retrieval in a single shared space.

</details>

<details>
<summary><b>17. Recovering Hidden Reward in Diffusion-Based Policies</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenyuan Xie, Shaokai Wu, Yuting Hu, Qiuchang Li, sotaagi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.00623) • [📄 arXiv](https://arxiv.org/abs/2605.00623) • [📥 PDF](https://arxiv.org/pdf/2605.00623)

> ENERGYFLOW unifies diffusion-based imitation learning and inverse reinforcement learning by learning a conservative energy field whose gradient drives action generation while exposing a recoverable reward signal, improving manipulation performance...

</details>

<details>
<summary><b>18. Think, then Score: Decoupled Reasoning and Scoring for Video Reward Modeling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiajun Liang, Borui Liao, Yulong Xu, Ouxiang Li, Yuan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05922) • [📄 arXiv](https://arxiv.org/abs/2605.05922) • [📥 PDF](https://arxiv.org/pdf/2605.05922)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-05-08.json`](data/daily/2026-05-08.json) | 18 |
| 📆 This Week | [`2026-W18.json`](data/weekly/2026-W18.json) | 37 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 96 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-08 | 18 | [View JSON](data/daily/2026-05-08.json) |
| 📄 2026-05-07 | 3 | [View JSON](data/daily/2026-05-07.json) |
| 📄 2026-05-06 | 8 | [View JSON](data/daily/2026-05-06.json) |
| 📄 2026-05-05 | 3 | [View JSON](data/daily/2026-05-05.json) |
| 📄 2026-05-04 | 5 | [View JSON](data/daily/2026-05-04.json) |
| 📄 2026-05-03 | 24 | [View JSON](data/daily/2026-05-03.json) |
| 📄 2026-05-02 | 24 | [View JSON](data/daily/2026-05-02.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W18 | 37 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |
| 📅 2026-W15 | 99 | [View JSON](data/weekly/2026-W15.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 96 | [View JSON](data/monthly/2026-05.json) |
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
