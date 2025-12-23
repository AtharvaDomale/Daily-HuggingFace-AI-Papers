<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-614+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">22</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">60</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">663</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">614+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 23, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows</b> ⭐ 56</summary>

<br/>

**👥 Authors:** Yuhao Zhou, SciYu, VitaCoco, BoKelvin, CoCoOne

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16969) • [📄 arXiv](https://arxiv.org/abs/2512.16969) • [📥 PDF](https://arxiv.org/pdf/2512.16969)

**💻 Code:** [⭐ Code](https://github.com/InternScience/SGI-Bench)

> Despite advances in scientific AI, a coherent framework for Scientific General Intelligence (SGI)-the ability to autonomously conceive, investigate, and reason across scientific domains-remains lacking. We present an operational SGI definition gro...

</details>

<details>
<summary><b>2. PhysBrain: Human Egocentric Data as a Bridge from Vision Language Models to Physical Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16793) • [📄 arXiv](https://arxiv.org/abs/2512.16793) • [📥 PDF](https://arxiv.org/pdf/2512.16793)

> No abstract available.

</details>

<details>
<summary><b>3. When Reasoning Meets Its Laws</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Liu Ziyin, Jingyan Shen, Tianang Leng, Yifan Sun, jyzhang1208

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17901) • [📄 arXiv](https://arxiv.org/abs/2512.17901) • [📥 PDF](https://arxiv.org/pdf/2512.17901)

**💻 Code:** [⭐ Code](https://github.com/ASTRAL-Group/LoRe)

> Despite the superior performance of Large Reasoning Models (LRMs), their reasoning behaviors are often counterintuitive, leading to suboptimal reasoning capabilities. To theoretically formalize the desired reasoning behaviors, this paper presents ...

</details>

<details>
<summary><b>4. Seed-Prover 1.5: Mastering Undergraduate-Level Theorem Proving via Learning from Experience</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17260) • [📄 arXiv](https://arxiv.org/abs/2512.17260) • [📥 PDF](https://arxiv.org/pdf/2512.17260)

**💻 Code:** [⭐ Code](https://github.com/ByteDance-Seed/Seed-Prover)

> Github: https://github.com/ByteDance-Seed/Seed-Prover

</details>

<details>
<summary><b>5. Both Semantics and Reconstruction Matter: Making Representation Encoders Ready for Text-to-Image Generation and Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17909) • [📄 arXiv](https://arxiv.org/abs/2512.17909) • [📥 PDF](https://arxiv.org/pdf/2512.17909)

> Modern Latent Diffusion Models (LDMs) typically operate in low-level Variational Autoencoder (VAE) latent spaces that are primarily optimized for pixel-level reconstruction. To unify vision generation and understanding, a burgeoning trend is to ad...

</details>

<details>
<summary><b>6. 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17012) • [📄 arXiv](https://arxiv.org/abs/2512.17012) • [📥 PDF](https://arxiv.org/pdf/2512.17012)

> Project page: https://ca-joe-yang.github.io/resource/projects/4D_RGPT We propose 4D-RGPT , a specialized MLLM that perceives 4D information for enhanced video understanding. We propose the P erceptual 4 D D istillation ( P4D ) training framework t...

</details>

<details>
<summary><b>7. Are We on the Right Way to Assessing LLM-as-a-Judge?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16041) • [📄 arXiv](https://arxiv.org/abs/2512.16041) • [📥 PDF](https://arxiv.org/pdf/2512.16041)

> We argue that evaluating LLM-as-a-Judge is biased by human-annotated ground truth, rethink the evaluation of LLM-as-a-Judge, and design metrics that do not need human annotations.

</details>

<details>
<summary><b>8. An Anatomy of Vision-Language-Action Models: From Modules to Milestones and Challenges</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11362) • [📄 arXiv](https://arxiv.org/abs/2512.11362) • [📥 PDF](https://arxiv.org/pdf/2512.11362)

> Vision-Language-Action (VLA) models are driving a revolution in robotics, enabling machines to understand instructions and interact with the physical world. This field is exploding with new models and datasets, making it both exciting and challeng...

</details>

<details>
<summary><b>9. RadarGen: Automotive Radar Point Cloud Generation from Cameras</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Or Litany, Shengyu Huang, Sanja Fidler, Fangqiang Ding, TomerBo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17897) • [📄 arXiv](https://arxiv.org/abs/2512.17897) • [📥 PDF](https://arxiv.org/pdf/2512.17897)

**💻 Code:** [⭐ Code](https://github.com/tomerborreda/RadarGen)

> Check out radargen.github.io

</details>

<details>
<summary><b>10. GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17495) • [📄 arXiv](https://arxiv.org/abs/2512.17495) • [📥 PDF](https://arxiv.org/pdf/2512.17495)

> Our new benchmark for evaluating the grounding capabilities of frontier MLLMs.

</details>

<details>
<summary><b>11. Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers</b> ⭐ 278</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17351) • [📄 arXiv](https://arxiv.org/abs/2512.17351) • [📥 PDF](https://arxiv.org/pdf/2512.17351)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/PhysicsLM4)

> https://x.com/ZeyuanAllenZhu/status/2000892470306152701 https://physics.allen-zhu.com/part-4-architecture-design/part-4-1

</details>

<details>
<summary><b>12. Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lihong Li, Meet P. Vadera, Rui Meng, Peng Zhou, ljb121002

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17008) • [📄 arXiv](https://arxiv.org/abs/2512.17008) • [📥 PDF](https://arxiv.org/pdf/2512.17008)

> Reinforcement learning (RL) has re-emerged as a natural approach for training interactive LLM agents in real-world environments. However, directly applying the widely used Group Relative Policy Optimization (GRPO) algorithm to multi-turn tasks exp...

</details>

<details>
<summary><b>13. HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.14870) • [📄 arXiv](https://arxiv.org/abs/2512.14870) • [📥 PDF](https://arxiv.org/pdf/2512.14870)

**💻 Code:** [⭐ Code](https://github.com/DanBenAmi/HERBench)

> 🔗 Project page: https://herbench.github.io/ 📄  arXiv: https://arxiv.org/abs/2512.14870 🤗  HF dataset card: https://huggingface.co/datasets/DanBenAmi/HERBench 🖥  Code (GitHub): https://github.com/DanBenAmi/HERBench

</details>

<details>
<summary><b>14. Animate Any Character in Any World</b> ⭐ 28</summary>

<br/>

**👥 Authors:** Yan Lu, Bo Dai, Hongyang Zhang, Fangyun Wei, Yitong Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17796) • [📄 arXiv](https://arxiv.org/abs/2512.17796) • [📥 PDF](https://arxiv.org/pdf/2512.17796)

**💻 Code:** [⭐ Code](https://github.com/snowflakewang/AniX)

> Introducing AniX, a system enables users to provide 3DGS scene along with a 3D or multi-view character, enabling interactive control of the character's behaviors and active exploration of the environment through natural language commands. The syst...

</details>

<details>
<summary><b>15. SWE-Bench++: A Framework for the Scalable Generation of Software Engineering Benchmarks from Open-Source Repositories</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17419) • [📄 arXiv](https://arxiv.org/abs/2512.17419) • [📥 PDF](https://arxiv.org/pdf/2512.17419)

> Benchmarks like SWE-bench have standardized the evaluation of Large Language Models (LLMs) on repository-level software engineering tasks. However, these efforts remain limited by manual curation, static datasets, and a focus on Python-based bug f...

</details>

<details>
<summary><b>16. StageVAR: Stage-Aware Acceleration for Visual Autoregressive Models</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16483) • [📄 arXiv](https://arxiv.org/abs/2512.16483) • [📥 PDF](https://arxiv.org/pdf/2512.16483)

**💻 Code:** [⭐ Code](https://github.com/sen-mao/StageVAR)

> github: https://github.com/sen-mao/StageVAR

</details>

<details>
<summary><b>17. Bolmo: Byteifying the Next Generation of Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.15586) • [📄 arXiv](https://arxiv.org/abs/2512.15586) • [📥 PDF](https://arxiv.org/pdf/2512.15586)

> So cool idea to make use of mLSTM and developing this byteifying approach 😍

</details>

<details>
<summary><b>18. Meta-RL Induces Exploration in Language Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Maria Brbic, Michael Moor, Damien Teney, Liangze Jiang, Yulun Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16848) • [📄 arXiv](https://arxiv.org/abs/2512.16848) • [📥 PDF](https://arxiv.org/pdf/2512.16848)

> 🌊LaMer, a general Meta-RL framework that enables LLM agents to explore and learn from the environment feedback at test time.

</details>

<details>
<summary><b>19. Robust-R1: Degradation-Aware Reasoning for Robust Visual Understanding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Runtao Liu, Xiaogang Xu, Wei Wei, Jianmin Chen, Jiaqi-hkust

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17532) • [📄 arXiv](https://arxiv.org/abs/2512.17532) • [📥 PDF](https://arxiv.org/pdf/2512.17532)

> Multimodal Large Language Models struggle to maintain reliable performance under extreme real-world visual degradations, which impede their practical robustness. Existing robust MLLMs predominantly rely on implicit training/adaptation that focuses...

</details>

<details>
<summary><b>20. 3D-RE-GEN: 3D Reconstruction of Indoor Scenes with a Generative Framework</b> ⭐ 33</summary>

<br/>

**👥 Authors:** Hendrik P. A. Lensch, Tobias Sautter, JDihlmann

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.17459) • [📄 arXiv](https://arxiv.org/abs/2512.17459) • [📥 PDF](https://arxiv.org/pdf/2512.17459)

**💻 Code:** [⭐ Code](https://github.com/cgtuebingen/3D-RE-GEN)

> 🌐 https://3dregen.jdihlmann.com/ 📃 https://arxiv.org/abs/2512.17459 💾 https://github.com/cgtuebingen/3D-RE-GEN

</details>

<details>
<summary><b>21. A Benchmark and Agentic Framework for Omni-Modal Reasoning and Tool Use in Long Videos</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.16978) • [📄 arXiv](https://arxiv.org/abs/2512.16978) • [📥 PDF](https://arxiv.org/pdf/2512.16978)

**💻 Code:** [⭐ Code](https://github.com/mbzuai-oryx/longshot)

> 🌐 Website: https://mbzuai-oryx.github.io/LongShOT/ 💻 Github: https://github.com/mbzuai-oryx/longshot 🤗 HuggingFace: https://huggingface.co/datasets/MBZUAI/longshot-bench

</details>

<details>
<summary><b>22. MineTheGap: Automatic Mining of Biases in Text-to-Image Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tomer Michaeli, Inbar Huberman-Spiegelglas, Nurit Spingarn-Eliezer, Noa Cohen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13427) • [📄 arXiv](https://arxiv.org/abs/2512.13427) • [📥 PDF](https://arxiv.org/pdf/2512.13427)

> Text-to-Image (TTI) models generate images based on text prompts, which often leave certain aspects of the desired image ambiguous. When faced with these ambiguities, TTI models have been shown to exhibit biases in their interpretations. These bia...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2025-12-23.json`](data/daily/2025-12-23.json) | 22 |
| 📆 This Week | [`2025-W51.json`](data/weekly/2025-W51.json) | 60 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 663 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-23 | 22 | [View JSON](data/daily/2025-12-23.json) |
| 📄 2025-12-22 | 38 | [View JSON](data/daily/2025-12-22.json) |
| 📄 2025-12-21 | 38 | [View JSON](data/daily/2025-12-21.json) |
| 📄 2025-12-20 | 37 | [View JSON](data/daily/2025-12-20.json) |
| 📄 2025-12-19 | 30 | [View JSON](data/daily/2025-12-19.json) |
| 📄 2025-12-18 | 38 | [View JSON](data/daily/2025-12-18.json) |
| 📄 2025-12-17 | 41 | [View JSON](data/daily/2025-12-17.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W51 | 60 | [View JSON](data/weekly/2025-W51.json) |
| 📅 2025-W50 | 230 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 663 | [View JSON](data/monthly/2025-12.json) |

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
