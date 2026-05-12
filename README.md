<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-16-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3835+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">16</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">42</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">214</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3835+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 12, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Soohak: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09063) • [📄 arXiv](https://arxiv.org/abs/2605.09063) • [📥 PDF](https://arxiv.org/pdf/2605.09063)

> For questions or model-evaluation requests, contact guijin.son@snu.ac.kr .

</details>

<details>
<summary><b>2. Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08354) • [📄 arXiv](https://arxiv.org/abs/2605.08354) • [📥 PDF](https://arxiv.org/pdf/2605.08354)

**💻 Code:** [⭐ Code](https://github.com/OpenEnvision/AutoRubric-as-Reward)

> Auto-Rubric as Reward converts a small set of labeled visual supervision into readable rubric text, supports both pointwise and pairwise VLM grading, and lets practitioners freely scale up the rubric dimensions they care about. On top of that, we ...

</details>

<details>
<summary><b>3. X-OmniClaw Technical Report: A Unified Mobile Agent for Multimodal Understanding and Interaction</b> ⭐ 65</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.05765) • [📄 arXiv](https://arxiv.org/abs/2605.05765) • [📥 PDF](https://arxiv.org/pdf/2605.05765)

**💻 Code:** [⭐ Code](https://github.com/OPPO-Mente-Lab/X-OmniClaw)

> Hi HF community! If you are interested in our work or have any questions, feel free to reach out or leave a comment. I'd love to hear your thoughts!

</details>

<details>
<summary><b>4. Geometry Conflict: Explaining and Controlling Forgetting in LLM Continual Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09608) • [📄 arXiv](https://arxiv.org/abs/2605.09608) • [📥 PDF](https://arxiv.org/pdf/2605.09608)

**💻 Code:** [⭐ Code](https://github.com/wyy-code/GCWM)

> How should we continually post-train LLMs without causing catastrophic forgetting? We find that forgetting is not simply caused by large parameter updates. Instead, it is better understood as a state-relative update-integration failure: harmful st...

</details>

<details>
<summary><b>5. Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09649) • [📄 arXiv](https://arxiv.org/abs/2605.09649) • [📥 PDF](https://arxiv.org/pdf/2605.09649)

**💻 Code:** [⭐ Code](https://github.com/ngocbh/trimkv)

> Can we improve long-context performance with KV cache eviction?

</details>

<details>
<summary><b>6. Omni-Persona: Systematic Benchmarking and Improving Omnimodal Personalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09996) • [📄 arXiv](https://arxiv.org/abs/2605.09996) • [📥 PDF](https://arxiv.org/pdf/2605.09996)

**💻 Code:** [⭐ Code](https://github.com/oyt9306/Omni-Persona)

> We introduce Omni-Persona, the first comprehensive benchmark for omnimodal personalization spanning text, image, and audio. Built on the Persona Modality Graph (PMG), it formalizes personalization as cross-modal routing and jointly evaluates groun...

</details>

<details>
<summary><b>7. Reinforcing Multimodal Reasoning Against Visual Degradation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09262) • [📄 arXiv](https://arxiv.org/abs/2605.09262) • [📥 PDF](https://arxiv.org/pdf/2605.09262)

> Reinforcement Learning has significantly advanced the reasoning capabilities of Multimodal Large Language Models (MLLMs), yet the resulting policies remain brittle against real-world visual degradations such as blur, compression artifacts, and low...

</details>

<details>
<summary><b>8. DeltaRubric: Generative Multimodal Reward Modeling via Joint Planning and Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09269) • [📄 arXiv](https://arxiv.org/abs/2605.09269) • [📥 PDF](https://arxiv.org/pdf/2605.09269)

> Aligning Multimodal Large Language Models (MLLMs) requires reliable reward models, yet existing single-step evaluators can suffer from lazy judging, exploiting language priors over fine-grained visual verification. While rubric-based evaluation mi...

</details>

<details>
<summary><b>9. RigidFormer: Learning Rigid Dynamics using Transformers</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09196) • [📄 arXiv](https://arxiv.org/abs/2605.09196) • [📥 PDF](https://arxiv.org/pdf/2605.09196)

**💻 Code:** [⭐ Code](https://github.com/Frank-ZY-Dou/Dynamics-Modeling)

> RigidFormer: Learning Rigid Dynamics with Transformers - our attempt to scale learning-based physical dynamics with Transformers. RigidFormer learns rigid dynamics with Transformers. It is a mesh-free, object-centric Transformer for multi-object r...

</details>

<details>
<summary><b>10. G-Zero: Self-Play for Open-Ended Generation from Zero Data</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Langlin Huang, Runpeng Dai, Tong Zheng, Haolin Liu, Chengsong Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09959) • [📄 arXiv](https://arxiv.org/abs/2605.09959) • [📥 PDF](https://arxiv.org/pdf/2605.09959)

**💻 Code:** [⭐ Code](https://github.com/Chengsong-Huang/G-Zero)

> Self-evolving LLMs excel in verifiable domains but struggle in open-ended tasks, where reliance on proxy LLM judges introduces capability bottlenecks and reward hacking. To overcome this, we introduce G-Zero, a verifier-free, co-evolutionary frame...

</details>

<details>
<summary><b>11. Model Merging Scaling Laws in Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2509.24244) • [📄 arXiv](https://arxiv.org/abs/2509.24244) • [📥 PDF](https://arxiv.org/pdf/2509.24244)

**💻 Code:** [⭐ Code](https://github.com/InfiXAI/Merging-Scaling-Law)

> Can we predict the returns of language model merging before trying every expert combination? We study scaling laws for LLM merging and find a compact floor-plus-tail law that predicts merged-model cross-entropy from base model size and the number ...

</details>

<details>
<summary><b>12. Pushing Biomolecular Utility-Diversity Frontiers with Supergroup Relative Policy Optimization</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08659) • [📄 arXiv](https://arxiv.org/abs/2605.08659) • [📥 PDF](https://arxiv.org/pdf/2605.08659)

**💻 Code:** [⭐ Code](https://github.com/IDEA-XL/SGRPO)

> No abstract available.

</details>

<details>
<summary><b>13. Uncovering Entity Identity Confusion in Multimodal Knowledge Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06096) • [📄 arXiv](https://arxiv.org/abs/2605.06096) • [📥 PDF](https://arxiv.org/pdf/2605.06096)

> Multimodal knowledge editing (MKE) aims to correct the internal knowledge of large vision-language models after deployment, yet the behavioral patterns of post-edit models remain underexplored. In this paper, we identify a systemic failure mode in...

</details>

<details>
<summary><b>14. Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09241) • [📄 arXiv](https://arxiv.org/abs/2605.09241) • [📥 PDF](https://arxiv.org/pdf/2605.09241)

**💻 Code:** [⭐ Code](https://github.com/intcomp/Sub-JEPA)

> We're releasing Sub-JEPA 🌐 LeWM (from LeCun's group) is the first end-to-end trainable JEPA world model — it uses isotropic Gaussian regularization to prevent representation collapse. Clean and effective. Our take: latent representations sit on lo...

</details>

<details>
<summary><b>15. TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09810) • [📄 arXiv](https://arxiv.org/abs/2605.09810) • [📥 PDF](https://arxiv.org/pdf/2605.09810)

> No abstract available.

</details>

<details>
<summary><b>16. 100,000+ Movie Reviews from Kazakhstan: Russian, Kazakh, and Code-Switched Texts</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rustem Yeshpanov

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08600) • [📄 arXiv](https://arxiv.org/abs/2605.08600) • [📥 PDF](https://arxiv.org/pdf/2605.08600)

> Accepted to NLP4DH 2026

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 16 |
| 📅 Today | [`2026-05-12.json`](data/daily/2026-05-12.json) | 16 |
| 📆 This Week | [`2026-W19.json`](data/weekly/2026-W19.json) | 42 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 214 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-12 | 16 | [View JSON](data/daily/2026-05-12.json) |
| 📄 2026-05-11 | 26 | [View JSON](data/daily/2026-05-11.json) |
| 📄 2026-05-10 | 38 | [View JSON](data/daily/2026-05-10.json) |
| 📄 2026-05-09 | 38 | [View JSON](data/daily/2026-05-09.json) |
| 📄 2026-05-08 | 18 | [View JSON](data/daily/2026-05-08.json) |
| 📄 2026-05-07 | 3 | [View JSON](data/daily/2026-05-07.json) |
| 📄 2026-05-06 | 8 | [View JSON](data/daily/2026-05-06.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W19 | 42 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 214 | [View JSON](data/monthly/2026-05.json) |
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
