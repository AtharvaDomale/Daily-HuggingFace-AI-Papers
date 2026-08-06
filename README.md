<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-14-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5490+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">14</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">33</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">109</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5490+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 06, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05139) • [📄 arXiv](https://arxiv.org/abs/2608.05139) • [📥 PDF](https://arxiv.org/pdf/2608.05139)

**💻 Code:** [⭐ Code](https://github.com/Gen-Verse/Skill-Entropy-RL) • [⭐ Code](https://github.com/huggingface)

> code and data: https://github.com/Gen-Verse/Skill-Entropy-RL

</details>

<details>
<summary><b>2. Towards Physics of Multimodal Pretraining: Knowledge Flow, Modality Synergy, Early Unification, and Recipes</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05000) • [📄 arXiv](https://arxiv.org/abs/2608.05000) • [📥 PDF](https://arxiv.org/pdf/2608.05000)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Explorations into multimodal pretraining.

</details>

<details>
<summary><b>3. OneDayAgent: Towards a Long-Horizon Harness for Autonomous Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05013) • [📄 arXiv](https://arxiv.org/abs/2608.05013) • [📥 PDF](https://arxiv.org/pdf/2608.05013)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/OneDayAgent) • [⭐ Code](https://github.com/huggingface)

> One Prompt. One Day. One Agent. Turning open-ended requests into  long-horizon execution.

</details>

<details>
<summary><b>4. The Personalization Mirage: How LLMs Fabricate User Profiles, and Why Self-Monitoring Misleads</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rui Sheng, Yanjie Zhang, Yushi Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04570) • [📄 arXiv](https://arxiv.org/abs/2608.04570) • [📥 PDF](https://arxiv.org/pdf/2608.04570)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Personalized LLMs with persistent memory are increasingly deployed, yet the faithfulness of their user models remains unexamined. We study over-inference (OI): the phenomenon where LLMs fabricate user attributes beyond what evidence supports. We i...

</details>

<details>
<summary><b>5. When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03632) • [📄 arXiv](https://arxiv.org/abs/2608.03632) • [📥 PDF](https://arxiv.org/pdf/2608.03632)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jjjyinuo/SA-OPD)

> SA-OPD addresses a previously overlooked failure mode in on-policy distillation: teacher signals can appear confident, informative, and learnable while being driven by input-agnostic language priors, formatting conventions, or stereotyped reasonin...

</details>

<details>
<summary><b>6. When Memory Lies: An Empirical Study of Spatial Memory Staleness in VLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yanjie Zhang, Yushi Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04574) • [📄 arXiv](https://arxiv.org/abs/2608.04574) • [📥 PDF](https://arxiv.org/pdf/2608.04574)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Memory-augmented VLM agents act on persistent spatial knowledge, yet that knowledge silently goes stale as the environment changes. We ask what happens when an agent must reconcile a confident memory claim with a contradicting observation, and whe...

</details>

<details>
<summary><b>7. NOLLI: A Difficulty-Calibrated Puzzle Benchmark for Diagnosing the English-Korean Performance Gap</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04397) • [📄 arXiv](https://arxiv.org/abs/2608.04397) • [📥 PDF](https://arxiv.org/pdf/2608.04397)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HAE-RAE/NOLLI)

> NOLLI is a procedurally generated, behaviorally calibrated English–Korean puzzle benchmark that finds small matched presentation-language gaps and localizes candidate bottlenecks in multi-step Hangul-jamo execution and Korean kinship terminology.

</details>

<details>
<summary><b>8. UniWorld-View: Large-Baseline View Synthesis via Video Diffusion Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yonghong Tian, Xunyu Zhou, Chaoran Feng, Wangbo Yu, Haiyang Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04701) • [📄 arXiv](https://arxiv.org/abs/2608.04701) • [📥 PDF](https://arxiv.org/pdf/2608.04701)

**💻 Code:** [⭐ Code](https://github.com/PKU-YuanGroup/UniWorld-View) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. ToolArtist: Tool-Using Unified Multimodal Models for Agentic Image Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chengwei Qin, Fengwei Teng, Zhongxiang Sun, Xiaomin Yu, Jiahao Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04436) • [📄 arXiv](https://arxiv.org/abs/2608.04436) • [📥 PDF](https://arxiv.org/pdf/2608.04436)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. Poly-OPD: Heterogeneous Multi-Teacher On-Policy Distillation for Capability-Selectable Flow Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04349) • [📄 arXiv](https://arxiv.org/abs/2608.04349) • [📥 PDF](https://arxiv.org/pdf/2608.04349)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Leading open text-to-image models often carry complementary strengths: one may lead on preference-aligned aesthetics while another follows compositional instructions more faithfully. However, differences in their autoencoders and noise schedules m...

</details>

<details>
<summary><b>11. HelloWorld: Enabling Socially Interactive Characters in Video World Models</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yoichi Sato, Kaipeng Zhang, Xuangeng Chu, Ruicong Liu, Liangyang Ouyang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05070) • [📄 arXiv](https://arxiv.org/abs/2608.05070) • [📥 PDF](https://arxiv.org/pdf/2608.05070)

**💻 Code:** [⭐ Code](https://github.com/AlayaLab/HelloWorld) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>12. OPD-V: Visual On-Policy Self-Distillation with Modality Balance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Volker Tresp, Zengjie Jin, Peng Liao, Jinhe Bi, Aniri

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05131) • [📄 arXiv](https://arxiv.org/abs/2608.05131) • [📥 PDF](https://arxiv.org/pdf/2608.05131)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>13. TriGlue: a Biology-Inspired Generative Model for Generating Molecular Glue-Induced Ternary Complex</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Enyan Dai, Yiqin Sun, Haochun Tang, Shuo Yan, Yuliang Yan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.22143) • [📄 arXiv](https://arxiv.org/abs/2607.22143) • [📥 PDF](https://arxiv.org/pdf/2607.22143)

**💻 Code:** [⭐ Code](https://github.com/yuliangyan0807/molecular-glue-design) • [⭐ Code](https://github.com/huggingface)

> Excited to share TriGlue , a biology-inspired generative framework for the de novo design of molecular glue-induced ternary complexes. Unlike conventional structure-based drug design, molecular glue discovery involves an unknown protein–protein in...

</details>

<details>
<summary><b>14. SKILL-KD: Contrastive Skill Distillation for LLM Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.28048) • [📄 arXiv](https://arxiv.org/abs/2607.28048) • [📥 PDF](https://arxiv.org/pdf/2607.28048)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Excited to share our new work, SKILL-KD: Contrastive Skill Distillation for LLM Agents! SKILL-KD explores a simple question: can a stronger teacher agent improve a frozen student agent without fine-tuning its weights? Instead of directly summarizi...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 14 |
| 📅 Today | [`2026-08-06.json`](data/daily/2026-08-06.json) | 14 |
| 📆 This Week | [`2026-W31.json`](data/weekly/2026-W31.json) | 33 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 109 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-06 | 14 | [View JSON](data/daily/2026-08-06.json) |
| 📄 2026-08-05 | 9 | [View JSON](data/daily/2026-08-05.json) |
| 📄 2026-08-04 | 2 | [View JSON](data/daily/2026-08-04.json) |
| 📄 2026-08-03 | 8 | [View JSON](data/daily/2026-08-03.json) |
| 📄 2026-08-02 | 38 | [View JSON](data/daily/2026-08-02.json) |
| 📄 2026-08-01 | 38 | [View JSON](data/daily/2026-08-01.json) |
| 📄 2026-07-31 | 10 | [View JSON](data/daily/2026-07-31.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W31 | 33 | [View JSON](data/weekly/2026-W31.json) |
| 📅 2026-W30 | 112 | [View JSON](data/weekly/2026-W30.json) |
| 📅 2026-W29 | 70 | [View JSON](data/weekly/2026-W29.json) |
| 📅 2026-W28 | 94 | [View JSON](data/weekly/2026-W28.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 109 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
