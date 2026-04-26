<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-22-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3596+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">74</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">425</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3596+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 26, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. LLaTiSA: Towards Difficulty-Stratified Time Series Reasoning from Visual Perception to Semantics</b> ⭐ 72</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.17295) • [📄 arXiv](https://arxiv.org/abs/2604.17295) • [📥 PDF](https://arxiv.org/pdf/2604.17295)

**💻 Code:** [⭐ Code](https://github.com/RainingNovember/LLaTiSA)

> HITSR - a hierarchical time series reasoning dataset comprising 83k samples with diverse task combinations and verified Chain-of-Thought (CoT) trajectories. LLATISA - a strong Time Series Reasoning Model (TSRM) that integrates visualized patterns ...

</details>

<details>
<summary><b>2. WorldMark: A Unified Benchmark Suite for Interactive Video World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yukang Feng, Kang He, Zhengyuan Lin, Xiaojie Xu, kpzhang996

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21686) • [📄 arXiv](https://arxiv.org/abs/2604.21686) • [📥 PDF](https://arxiv.org/pdf/2604.21686)

> one burning question for me: how well does the unified WASD+L/R action layer generalize to models whose native controls are continuous, nonuniform, or otherwise non-discretizable? an ablation i'd love to see is removing the adapter and feeding eac...

</details>

<details>
<summary><b>3. UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling</b> ⭐ 37</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.19734) • [📄 arXiv](https://arxiv.org/abs/2604.19734) • [📥 PDF](https://arxiv.org/pdf/2604.19734)

**💻 Code:** [⭐ Code](https://github.com/xpeng-robotics/UniT)

> the tri-branch cross-reconstruction and the shared discrete token space are slick, but i keep wondering how they scale when human and humanoid morphologies diverge a lot. the core assumption of universal visual consequences across embodiments is a...

</details>

<details>
<summary><b>4. StyleID: A Perception-Aware Dataset and Metric for Stylization-Agnostic Facial Identity Recognition</b> ⭐ 18</summary>

<br/>

**👥 Authors:** Seungmi Lee, Youngseo Kim, Ayeong Jeong, Changmin Lee, Kwan Yun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21689) • [📄 arXiv](https://arxiv.org/abs/2604.21689) • [📥 PDF](https://arxiv.org/pdf/2604.21689)

**💻 Code:** [⭐ Code](https://github.com/kwanyun/StyleID)

> Stylization-Agnostic Facial Identity Recognition

</details>

<details>
<summary><b>5. Co-Evolving LLM Decision and Skill Bank Agents for Long-Horizon Tasks</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.20987) • [📄 arXiv](https://arxiv.org/abs/2604.20987) • [📥 PDF](https://arxiv.org/pdf/2604.20987)

**💻 Code:** [⭐ Code](https://github.com/wuxiyang1996/cos-play)

> Github: https://github.com/wuxiyang1996/cos-play . Paper: https://arxiv.org/abs/2604.20987 Website: https://wuxiyang1996.github.io/COSPLAY_page/ Models: https://huggingface.co/IntelligenceLab/COS-PLAY Cold-Start Data: https://huggingface.co/datase...

</details>

<details>
<summary><b>6. Seeing Fast and Slow: Learning the Flow of Time in Videos</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ali Farhadi, Tao Tu, Jingsen Zhu, Rundong Luo, Yen-Siang Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21931) • [📄 arXiv](https://arxiv.org/abs/2604.21931) • [📥 PDF](https://arxiv.org/pdf/2604.21931)

> that audio-visual self-supervision for speed-change detection is neat, it exploits a real cross-modal cue that doesn't need labels. but i keep circling back to edge cases where audio is absent or misleading, like mute clips or mismatched soundtrac...

</details>

<details>
<summary><b>7. VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21375) • [📄 arXiv](https://arxiv.org/abs/2604.21375) • [📥 PDF](https://arxiv.org/pdf/2604.21375)

**💻 Code:** [⭐ Code](https://github.com/UCSC-VLAA/VLAA-GUI)

> Autonomous GUI agents suffer from two chronic failure modes: early stopping (declaring success before the task is actually done) and repetitive loops (cycling through the same failing action without recovering). VLAA-GUI is a modular framework wit...

</details>

<details>
<summary><b>8. TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21889) • [📄 arXiv](https://arxiv.org/abs/2604.21889) • [📥 PDF](https://arxiv.org/pdf/2604.21889)

> We present TingIS, an enterprise-grade end-to-end risk discovery system that processes 300,000 customer incidents every day and up to 2,000 incidents per minute. By synergizing efficient indexing techniques with Large Language Models, TingIS achie...

</details>

<details>
<summary><b>9. Hybrid Policy Distillation for LLMs</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Pengfei Liu, Rui Wang, Ruobing Xie, Wenhong Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.20244) • [📄 arXiv](https://arxiv.org/abs/2604.20244) • [📥 PDF](https://arxiv.org/pdf/2604.20244)

**💻 Code:** [⭐ Code](https://github.com/zwhong714/Hybrid-Policy-Distillation)

> 🧭 A unified view of policy distillation methods ⚡ Efficient one-hot-style distillation 🧩 A hybrid KL objective with a masking mechanism 🪶 Lightweight sampling under an offline-prefix setting

</details>

<details>
<summary><b>10. Context Unrolling in Omni Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hao He, Fei Xiao, Yang Zhao, Zhijie Lin, Ceyuan Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21921) • [📄 arXiv](https://arxiv.org/abs/2604.21921) • [📥 PDF](https://arxiv.org/pdf/2604.21921)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API MMCORE: MultiModal COnnection with Representation Aligned Latent Embeddings...

</details>

<details>
<summary><b>11. EditCrafter: Tuning-free High-Resolution Image Editing via Pretrained Diffusion Model</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Hyungjin Chung, Yongjun Cho, Sumin Seo, Kunho Kim

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.10268) • [📄 arXiv](https://arxiv.org/abs/2604.10268) • [📥 PDF](https://arxiv.org/pdf/2604.10268)

**💻 Code:** [⭐ Code](https://github.com/EditCrafter/EditCrafter)

> No abstract available.

</details>

<details>
<summary><b>12. Vista4D: Video Reshooting with 4D Point Clouds</b> ⭐ 88</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21915) • [📄 arXiv](https://arxiv.org/abs/2604.21915) • [📥 PDF](https://arxiv.org/pdf/2604.21915)

**💻 Code:** [⭐ Code](https://github.com/Eyeline-Labs/Vista4D)

> I'm one of the authors on the paper! Vista4D is a model which grounds video reshooting/novel view synthesis in a 4D point cloud. Notably, we trained Vista4D to be robust to 4D reconstruction artifacts (because 4D reconstruction models, though gett...

</details>

<details>
<summary><b>13. UniGenDet: A Unified Generative-Discriminative Framework for Co-Evolutionary Image Generation and Generated Image Detection</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21904) • [📄 arXiv](https://arxiv.org/abs/2604.21904) • [📥 PDF](https://arxiv.org/pdf/2604.21904)

**💻 Code:** [⭐ Code](https://github.com/Zhangyr2022/UniGenDet)

> Accepted to CVPR 2026. Image generation and generated-image detection have both advanced rapidly, but mostly along separate technical paths: generation is dominated by generative architectures, while detection is dominated by discriminative ones. ...

</details>

<details>
<summary><b>14. Temporally Extended Mixture-of-Experts Models</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.20156) • [📄 arXiv](https://arxiv.org/abs/2604.20156) • [📥 PDF](https://arxiv.org/pdf/2604.20156)

**💻 Code:** [⭐ Code](https://github.com/princeton-polaris-lab/rl_moe)

> Post-trains MoE LLMs into temporally extended MoEs, where active expert sets switch less frequently.

</details>

<details>
<summary><b>15. WebGen-R1: Incentivizing Large Language Models to Generate Functional and Aesthetic Websites with Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sunghun Kim, Jiasi Shen, Chansung Park, Chenglin Cai, Juyong Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.20398) • [📄 arXiv](https://arxiv.org/abs/2604.20398) • [📥 PDF](https://arxiv.org/pdf/2604.20398)

> WebGen-R1 substantially transforms a 7B base model from generating nearly nonfunctional websites into producing deployable, aesthetically aligned multi-page websites.

</details>

<details>
<summary><b>16. Coevolving Representations in Joint Image-Feature Diffusion</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Nikos Komodakis, Spyros Gidaris, Theodoros Kouzelis

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.17492) • [📄 arXiv](https://arxiv.org/abs/2604.17492) • [📥 PDF](https://arxiv.org/pdf/2604.17492)

**💻 Code:** [⭐ Code](https://github.com/zelaki/CoReDi)

> No abstract available.

</details>

<details>
<summary><b>17. Trust but Verify: Introducing DAVinCI -- A Framework for Dual Attribution and Verification in Claim Inference for Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nedim Lipka, Ryan Rossi, Vipula Rawte, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21193) • [📄 arXiv](https://arxiv.org/abs/2604.21193) • [📥 PDF](https://arxiv.org/pdf/2604.21193)

**💻 Code:** [⭐ Code](https://github.com/vr25/davinci)

> Code: https://github.com/vr25/davinci

</details>

<details>
<summary><b>18. Explainable Disentangled Representation Learning for Generalizable Authorship Attribution in the Era of Generative AI</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Thien Huu Nguyen, Nghia Trung Ngo, Van-Cuong Pham, Hieu Man, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21300) • [📄 arXiv](https://arxiv.org/abs/2604.21300) • [📥 PDF](https://arxiv.org/pdf/2604.21300)

**💻 Code:** [⭐ Code](https://github.com/hieum98/avae)

> Dataset: https://huggingface.co/collections/Hieuman/document-level-authorship-datasets

</details>

<details>
<summary><b>19. Test-Time Adaptation for EEG Foundation Models: A Systematic Study under Real-World Distribution Shifts</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jimeng Sun, Jathurshan Pradeepkumar, Gabriel Jason Lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.16926) • [📄 arXiv](https://arxiv.org/abs/2604.16926) • [📥 PDF](https://arxiv.org/pdf/2604.16926)

> We present NeuroAdapt-Bench, a systematic benchmark for evaluating test-time adaptation methods on EEG foundation models under realistic distribution shifts.

</details>

<details>
<summary><b>20. Encoder-Free Human Motion Understanding via Structured Motion Descriptions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu Xiao, Thomas Ploetz, Yao Zhang, Ryenhails

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.21668) • [📄 arXiv](https://arxiv.org/abs/2604.21668) • [📥 PDF](https://arxiv.org/pdf/2604.21668)

**💻 Code:** [⭐ Code](https://github.com/yaozhang182/motion-smd)

> Code, data, and pretrained LoRA adapters are available at https://yaozhang182.github.io/motion-smd/

</details>

<details>
<summary><b>21. PersonalAI: A Systematic Comparison of Knowledge Graph Storage and Retrieval Approaches for Personalized LLM agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2506.17001) • [📄 arXiv](https://arxiv.org/abs/2506.17001) • [📥 PDF](https://arxiv.org/pdf/2506.17001)

> Personalizing language models by effectively incorporating user interaction history remains a central challenge in the development of adaptive AI systems. While large language models (LLMs), combined with Retrieval-Augmented Generation (RAG), have...

</details>

<details>
<summary><b>22. 3D-VCD: Hallucination Mitigation in 3D-LLM Embodied Agents through Visual Contrastive Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.08645) • [📄 arXiv](https://arxiv.org/abs/2604.08645) • [📥 PDF](https://arxiv.org/pdf/2604.08645)

> We present 3D-VCD, a training-free inference-time framework that mitigates hallucinations in 3D embodied vision-language models through visual contrastive decoding. Our method constructs structured 3D scene graphs and introduces semantic and geome...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 22 |
| 📅 Today | [`2026-04-26.json`](data/daily/2026-04-26.json) | 22 |
| 📆 This Week | [`2026-W16.json`](data/weekly/2026-W16.json) | 74 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 425 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-26 | 22 | [View JSON](data/daily/2026-04-26.json) |
| 📄 2026-04-25 | 22 | [View JSON](data/daily/2026-04-25.json) |
| 📄 2026-04-24 | 5 | [View JSON](data/daily/2026-04-24.json) |
| 📄 2026-04-23 | 8 | [View JSON](data/daily/2026-04-23.json) |
| 📄 2026-04-22 | 9 | [View JSON](data/daily/2026-04-22.json) |
| 📄 2026-04-21 | 3 | [View JSON](data/daily/2026-04-21.json) |
| 📄 2026-04-20 | 5 | [View JSON](data/daily/2026-04-20.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |
| 📅 2026-W15 | 99 | [View JSON](data/weekly/2026-W15.json) |
| 📅 2026-W14 | 140 | [View JSON](data/weekly/2026-W14.json) |
| 📅 2026-W13 | 115 | [View JSON](data/weekly/2026-W13.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 425 | [View JSON](data/monthly/2026-04.json) |
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
