<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-18-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4622+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">41</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">219</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4622+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** June 10, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. ABot-Earth 0.5: Generative 3D Earth Model</b> ⭐ 83</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09967) • [📄 arXiv](https://arxiv.org/abs/2606.09967) • [📥 PDF](https://arxiv.org/pdf/2606.09967)

**💻 Code:** [⭐ Code](https://github.com/amap-cvlab/ABot-Earth-0.5) • [⭐ Code](https://github.com/huggingface)

> Tech report. ABot-Earth-0.5 is a generative 3D Earth model developed by the Amap CV Lab. The github repo is dedicated to our technical report and academic discourse. It does not contain implementation code. Media: https://www.youtube.com/watch?v=Q...

</details>

<details>
<summary><b>2. SearchSwarm: Towards Delegation Intelligence in Agentic LLMs for Long-Horizon Deep Research</b> ⭐ 28</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09730) • [📄 arXiv](https://arxiv.org/abs/2606.09730) • [📥 PDF](https://arxiv.org/pdf/2606.09730)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Search-Swarm/SearchSwarm)

> Real tasks can grow almost unbounded, yet a model's context is finite. We teach agentic LLMs delegation intelligence: to decompose a long-horizon task, delegate bounded subtasks to its own subagents, and integrate their condensed, evidence-grounde...

</details>

<details>
<summary><b>3. One Token per Multimodal Evidence: Latent Memory for Resource-Constrained QA</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10572) • [📄 arXiv](https://arxiv.org/abs/2606.10572) • [📥 PDF](https://arxiv.org/pdf/2606.10572)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zz1358m/Latent-Memory-Master)

> Latent Memory is a novel method for efficient representation and QA generation. It shows that one token per multimodal evidence can lead to a good performance-efficiency trade-off.

</details>

<details>
<summary><b>4. EEVEE: Towards Test-time Prompt Learning in the Real World for Self-Improving Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11182) • [📄 arXiv](https://arxiv.org/abs/2606.11182) • [📥 PDF](https://arxiv.org/pdf/2606.11182)

**💻 Code:** [⭐ Code](https://github.com/Princeton-AI2-Lab/EEVEE) • [⭐ Code](https://github.com/huggingface)

> EEVEE studies test-time prompt learning for LLM agents in more realistic settings, where tasks arrive as heterogeneous streams from multiple datasets and domains. Instead of optimizing a single prompt for a fixed benchmark, EEVEE introduces a rout...

</details>

<details>
<summary><b>5. How Does Reasoning Flow? Tracing Attention-Induced Information Flow for Targeted RL in LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yijia Luo, Weixun Wang, Yuhan Sun, Yang Li, Zhichen Dong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10646) • [📄 arXiv](https://arxiv.org/abs/2606.10646) • [📥 PDF](https://arxiv.org/pdf/2606.10646)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> What if we stopped rewarding every token equally and instead followed the actual "reasoning bloodstream" inside an LLM? FlowTracer tackles one of RL’s messiest blind spots—token-level credit assignment—by turning attention patterns into a directed...

</details>

<details>
<summary><b>6. Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.04391) • [📄 arXiv](https://arxiv.org/abs/2606.04391) • [📥 PDF](https://arxiv.org/pdf/2606.04391)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/plusnli/skill-dynamic-retrieval)

> This paper studies online skill learning for web agents, where agents continually induce reusable skills from previous task trajectories and reuse them for future web tasks. A key motivation is that most prior skill-based web agent methods retriev...

</details>

<details>
<summary><b>7. Bridging the Agent-World Gap: Text World Models for LLM-based Agents</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09032) • [📄 arXiv](https://arxiv.org/abs/2606.09032) • [📥 PDF](https://arxiv.org/pdf/2606.09032)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sustech-nlp/awesome-text-world-models)

> We present a systematic overview of text world models for agent applications, providing insights into narrowing the agent–world gap.

</details>

<details>
<summary><b>8. SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning</b> ⭐ 95</summary>

<br/>

**👥 Authors:** Jie Tang, Zhuoyi Yang, Fengjia Guo, Wenhao Yan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10804) • [📄 arXiv](https://arxiv.org/abs/2606.10804) • [📥 PDF](https://arxiv.org/pdf/2606.10804)

**💻 Code:** [⭐ Code](https://github.com/zai-org/SCAIL-2) • [⭐ Code](https://github.com/huggingface)

> SCAIL-2 provides an end-to-end framework for controlled character animation, eliminating reliance on intermediate pose skeletons and introducing the MotionPair-60K dataset for improved motion transfer performance.

</details>

<details>
<summary><b>9. Emergent Misalignment Can Be Induced by Sycophancy and Reversed via Alignment Gating</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09068) • [📄 arXiv](https://arxiv.org/abs/2606.09068) • [📥 PDF](https://arxiv.org/pdf/2606.09068)

**💻 Code:** [⭐ Code](https://github.com/stay1to0/Sycophancy_Emergent_Misalignment_and_Gated_attention_FT) • [⭐ Code](https://github.com/huggingface)

> Prior work has shown that fine-tuning large language models on malicious or incorrect outputs in narrow domains can induce broad misalignment and harmful behavior, a phenomenon known as emergent misalignment. However, efficient methods for reversi...

</details>

<details>
<summary><b>10. What Should Agents Say? Action-state Communication for Efficient Multi-Agent Systems</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.05304) • [📄 arXiv](https://arxiv.org/abs/2606.05304) • [📥 PDF](https://arxiv.org/pdf/2606.05304)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/iNLP-Lab/PACT)

> Multi-agent systems (MAS) built on large language models are typically organized around roles, pipelines, and turn schedules, while the content that agents pass to one another is often left as unconstrained natural language. However, this free-for...

</details>

<details>
<summary><b>11. ARM: An AutoRegressive Large Multimodal Model with Unified Discrete Representations</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Feng Li, Xuefeng Hu, Jiacheng Pan, Xiao Wang, Junke Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11188) • [📄 arXiv](https://arxiv.org/abs/2606.11188) • [📥 PDF](https://arxiv.org/pdf/2606.11188)

**💻 Code:** [⭐ Code](https://github.com/wdrink/ARM) • [⭐ Code](https://github.com/huggingface)

> Technical report: a discrete autoregressive model that unifies image generation, editing, and understanding.

</details>

<details>
<summary><b>12. When the Chain of Thought Knows Better: Failure Modes in Multi-Turn Reasoning Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Samuele Poppi, Nils Lukas, Sai Kartheek Reddy Kasu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10740) • [📄 arXiv](https://arxiv.org/abs/2606.10740) • [📥 PDF](https://arxiv.org/pdf/2606.10740)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: Standard safety evaluations are missing a massive chunk of how reasoning models actually fail. In this paper, we moved beyond static, single-turn prompts to analyze multi-turn adversarial dialogues across distilled models like DeepSeek-R1-7...

</details>

<details>
<summary><b>13. Dynamic Linear Attention</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10650) • [📄 arXiv](https://arxiv.org/abs/2606.10650) • [📥 PDF](https://arxiv.org/pdf/2606.10650)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> DLA (Dynamic Linear Attention) is a framework for multi-state linear attention that uses information-aware dynamic state merging and capacity-bounded memory modeling to improve long-context LLM scalability.

</details>

<details>
<summary><b>14. Workflow-GYM: Towards Long-Horizon Evaluation of Computer-use Agentic tasks in Real-World Professional Fields</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11042) • [📄 arXiv](https://arxiv.org/abs/2606.11042) • [📥 PDF](https://arxiv.org/pdf/2606.11042)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>15. Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Tobias Springenberg, Qiyang Li, Charles Xu, Andy Peng, Zhiyuan Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.11087) • [📄 arXiv](https://arxiv.org/abs/2606.11087) • [📥 PDF](https://arxiv.org/pdf/2606.11087)

**💻 Code:** [⭐ Code](https://github.com/zhouzypaul/qgf) • [⭐ Code](https://github.com/huggingface)

> QGF (Q-Guided Flow) is a test-time reinforcement learning method that utilizes critic gradients to guide pre-trained flow-based policies toward higher-value actions, enhancing stability and scalability in continuous control.

</details>

<details>
<summary><b>16. MilliVid: Hierarchical Latents for Long-Range Consistency in Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vitor Guizilini, Sergey Zakharov, Basile Van Hoorick, David Charatan, Ishaan Preetam Chandratreya

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.09056) • [📄 arXiv](https://arxiv.org/abs/2606.09056) • [📥 PDF](https://arxiv.org/pdf/2606.09056)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>17. Struct-Searcher: Agentic Structural Thinking Advances Multimodal Deep Information Seeking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.07689) • [📄 arXiv](https://arxiv.org/abs/2606.07689) • [📥 PDF](https://arxiv.org/pdf/2606.07689)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Struct-Searcher is a training-free agentic workflow that advances multimodal deep research with structure-aware thinking mechanisms.

</details>

<details>
<summary><b>18. BenSyc: Benchmarking Conversational Sycophancy and Human Alignment in LLMs for Bengali Contexts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10061) • [📄 arXiv](https://arxiv.org/abs/2606.10061) • [📥 PDF](https://arxiv.org/pdf/2606.10061)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language models (LLMs) increasingly participate in emotionally sensitive social conversations, where responses may shift from balanced support toward excessive validation or escalatory alignment. Existing sycophancy research primarily focuse...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 18 |
| 📅 Today | [`2026-06-10.json`](data/daily/2026-06-10.json) | 18 |
| 📆 This Week | [`2026-W23.json`](data/weekly/2026-W23.json) | 41 |
| 🗓️ This Month | [`2026-06.json`](data/monthly/2026-06.json) | 219 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-06-10 | 18 | [View JSON](data/daily/2026-06-10.json) |
| 📄 2026-06-09 | 8 | [View JSON](data/daily/2026-06-09.json) |
| 📄 2026-06-08 | 15 | [View JSON](data/daily/2026-06-08.json) |
| 📄 2026-06-07 | 50 | [View JSON](data/daily/2026-06-07.json) |
| 📄 2026-06-06 | 50 | [View JSON](data/daily/2026-06-06.json) |
| 📄 2026-06-05 | 21 | [View JSON](data/daily/2026-06-05.json) |
| 📄 2026-06-04 | 0 | [View JSON](data/daily/2026-06-04.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W23 | 41 | [View JSON](data/weekly/2026-W23.json) |
| 📅 2026-W22 | 178 | [View JSON](data/weekly/2026-W22.json) |
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-06 | 219 | [View JSON](data/monthly/2026-06.json) |
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
