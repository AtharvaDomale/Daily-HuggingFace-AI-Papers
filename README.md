<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-26-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1296+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">26</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">129</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">558</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1296+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 23, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Agentic Reasoning for Large Language Models</b> ⭐ 105</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.12538) • [📄 arXiv](https://arxiv.org/abs/2601.12538) • [📥 PDF](https://arxiv.org/pdf/2601.12538)

**💻 Code:** [⭐ Code](https://github.com/weitianxin/Awesome-Agentic-Reasoning)

> 🌐 Awesome-Agentic-Reasoning GitHub Link: https://github.com/weitianxin/Awesome-Agentic-Reasoning

</details>

<details>
<summary><b>2. MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Samiul Alam, Zhongwei Wan, Zixuan Zhong, Peizhou Huang, donghao-zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.12346) • [📄 arXiv](https://arxiv.org/abs/2601.12346) • [📥 PDF](https://arxiv.org/pdf/2601.12346)

**💻 Code:** [⭐ Code](https://github.com/AIoT-MLSys-Lab/MMDeepResearch-Bench)

> Introducing MMDeepResearch-Bench, a benchmark for multimodal deep research agents. Page: https://mmdeepresearch-bench.github.io/ Paper: https://arxiv.org/abs/2601.12346 Code: https://github.com/AIoT-MLSys-Lab/MMDeepResearch-Bench Dataset: https://...

</details>

<details>
<summary><b>3. Rethinking Video Generation Model for the Embodied World</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15282) • [📄 arXiv](https://arxiv.org/abs/2601.15282) • [📥 PDF](https://arxiv.org/pdf/2601.15282)

**💻 Code:** [⭐ Code](https://github.com/DAGroup-PKU/ReVidgen/)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Wow, wo, val! A Comprehensive Embodied World Model Evaluation Turing Test (...

</details>

<details>
<summary><b>4. Paper2Rebuttal: A Multi-Agent Framework for Transparent Author Response Assistance</b> ⭐ 146</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14171) • [📄 arXiv](https://arxiv.org/abs/2601.14171) • [📥 PDF](https://arxiv.org/pdf/2601.14171)

**💻 Code:** [⭐ Code](https://github.com/AutoLab-SAI-SJTU/Paper2Rebuttal)

> RebuttalAgent is an AI-powered multi-agent system that helps researchers craft high-quality rebuttals for academic paper reviews. The system analyzes reviewer comments, searches relevant literature, generates rebuttal strategies, and produces form...

</details>

<details>
<summary><b>5. Behavior Knowledge Merge in Reinforced Agentic Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13572) • [📄 arXiv](https://arxiv.org/abs/2601.13572) • [📥 PDF](https://arxiv.org/pdf/2601.13572)

**💻 Code:** [⭐ Code](https://github.com/xiangchi-yuan/mrl)

> 🚀 TL;DR We introduce RAM (Reinforced Agent Merging) , a method designed to merge RL-trained agents into a single generalist model without retraining, outperforming the original specialized agents in their domains. 💡 Key Insights The Problem: Stand...

</details>

<details>
<summary><b>6. Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14750) • [📄 arXiv](https://arxiv.org/abs/2601.14750) • [📥 PDF](https://arxiv.org/pdf/2601.14750)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Forest Before Trees: Latent Superposition for Efficient Visual Reasoning (2...

</details>

<details>
<summary><b>7. GutenOCR: A Grounded Vision-Language Front-End for Documents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14490) • [📄 arXiv](https://arxiv.org/abs/2601.14490) • [📥 PDF](https://arxiv.org/pdf/2601.14490)

**💻 Code:** [⭐ Code](https://github.com/Roots-Automation/GutenOCR)

> We're excited to share our first open model release, a grounded VLM for OCR applications!

</details>

<details>
<summary><b>8. Typhoon OCR: Open Vision-Language Model For Thai Document Extraction</b> ⭐ 85</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14722) • [📄 arXiv](https://arxiv.org/abs/2601.14722) • [📥 PDF](https://arxiv.org/pdf/2601.14722)

**💻 Code:** [⭐ Code](https://github.com/scb-10x/typhoon-ocr)

> Document extraction is a core component of digital workflows, yet existing vision-language models (VLMs) predominantly favor high-resource languages. Thai presents additional challenges due to script complexity from non-latin letters, the absence ...

</details>

<details>
<summary><b>9. Typhoon ASR Real-time: FastConformer-Transducer for Thai Automatic Speech Recognition</b> ⭐ 38</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13044) • [📄 arXiv](https://arxiv.org/abs/2601.13044) • [📥 PDF](https://arxiv.org/pdf/2601.13044)

**💻 Code:** [⭐ Code](https://github.com/scb-10x/typhoon-asr)

> Large encoder-decoder models like Whisper achieve strong offline transcription but remain impractical for streaming applications due to high latency. However, due to the accessibility of pre-trained checkpoints, the open Thai ASR landscape remains...

</details>

<details>
<summary><b>10. Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14027) • [📄 arXiv](https://arxiv.org/abs/2601.14027) • [📥 PDF](https://arxiv.org/pdf/2601.14027)

> Recommend to try our demo at: https://demo.projectnumina.ai/

</details>

<details>
<summary><b>11. FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning</b> ⭐ 141</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11141) • [📄 arXiv](https://arxiv.org/abs/2601.11141) • [📥 PDF](https://arxiv.org/pdf/2601.11141)

**💻 Code:** [⭐ Code](https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma)

> Some of the observations founded are :- -- End to end S2S advantage : Chroma 1.0 avoids cascaded ASR LLM TTS pipelines, reducing latency and preserving paralinguistic cues like timbre and prosody. -- High fidelity voice cloning : With only a few s...

</details>

<details>
<summary><b>12. FinVault: Benchmarking Financial Agent Safety in Execution-Grounded Environments</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07853) • [📄 arXiv](https://arxiv.org/abs/2601.07853) • [📥 PDF](https://arxiv.org/pdf/2601.07853)

**💻 Code:** [⭐ Code](https://github.com/aifinlab/FinVault)

> the first execution-grounded security benchmark for financial agents

</details>

<details>
<summary><b>13. Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15220) • [📄 arXiv](https://arxiv.org/abs/2601.15220) • [📥 PDF](https://arxiv.org/pdf/2601.15220)

**💻 Code:** [⭐ Code](https://github.com/parameterlab/privacy-collapse)

> Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models Overview This paper identifies a critical new failure mode in language models called "privacy collapse" . The researchers demonstrate that benign, high-quality fi...

</details>

<details>
<summary><b>14. XR: Cross-Modal Agents for Composed Image Retrieval</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14245) • [📄 arXiv](https://arxiv.org/abs/2601.14245) • [📥 PDF](https://arxiv.org/pdf/2601.14245)

> project website: https://01yzzyu.github.io/xr.github.io/

</details>

<details>
<summary><b>15. RoboBrain 2.5: Depth in Sight, Time in Mind</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuheng Ji, Yijie Xu, Zhiyu Li, Huajie Tan, Zhoues

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14352) • [📄 arXiv](https://arxiv.org/abs/2601.14352) • [📥 PDF](https://arxiv.org/pdf/2601.14352)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Towards Cross-View Point Correspondence in Vision-Language Models (2025) Ac...

</details>

<details>
<summary><b>16. Quantifying Speaker Embedding Phonological Rule Interactions in Accented Speech Synthesis</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jihwan Lee, Thanapat Trachu, Yoonjeong Lee, Thanathai Lertpetchpun, tiantiaf

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14417) • [📄 arXiv](https://arxiv.org/abs/2601.14417) • [📥 PDF](https://arxiv.org/pdf/2601.14417)

> Many spoken languages, including English, exhibit wide variation in dialects and accents, making accent control an important capability for flexible text-to-speech (TTS) models. Current TTS systems typically generate accented speech by conditionin...

</details>

<details>
<summary><b>17. Implicit Neural Representation Facilitates Unified Universal Vision Encoding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhenheng Yang, Xuefeng Hu, Xiao Wang, Matthew Gwilliam

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14256) • [📄 arXiv](https://arxiv.org/abs/2601.14256) • [📥 PDF](https://arxiv.org/pdf/2601.14256)

**💻 Code:** [⭐ Code](https://github.com/tiktok/huvr)

> Code: https://github.com/tiktok/huvr

</details>

<details>
<summary><b>18. AgentEHR: Advancing Autonomous Clinical Decision-Making via Retrospective Summarization</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13918) • [📄 arXiv](https://arxiv.org/abs/2601.13918) • [📥 PDF](https://arxiv.org/pdf/2601.13918)

**💻 Code:** [⭐ Code](https://github.com/BlueZeros/AgentEHR)

> This paper presents AGENTEHR, a novel benchmark designed to bridge the gap between idealized experimental settings and realistic clinical environments. Unlike previous tasks that focus on factual retrieval (e.g., searching for a specific medicatio...

</details>

<details>
<summary><b>19. FARE: Fast-Slow Agentic Robotic Exploration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jingsong Liang, Shizhe Zhang, Jeric Lew, Xuxin Lv, Shuhao Liao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14681) • [📄 arXiv](https://arxiv.org/abs/2601.14681) • [📥 PDF](https://arxiv.org/pdf/2601.14681)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API ORION: Option-Regularized Deep Reinforcement Learning for Cooperative Multi...

</details>

<details>
<summary><b>20. Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14152) • [📄 arXiv](https://arxiv.org/abs/2601.14152) • [📥 PDF](https://arxiv.org/pdf/2601.14152)

> Prompt order can break LMs performance — even with the same content.

</details>

<details>
<summary><b>21. The Responsibility Vacuum: Organizational Failure in Scaled Agent Systems</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Roman Bondar, Oleg Romanchuk

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15059) • [📄 arXiv](https://arxiv.org/abs/2601.15059) • [📥 PDF](https://arxiv.org/pdf/2601.15059)

> Some of the observations founded are :- -- Authority capacity mismatch is structural : Decisions are formally approved by humans, but the epistemic capacity to understand those decisions does not scale with agent generated throughput, creating a s...

</details>

<details>
<summary><b>22. Facilitating Proactive and Reactive Guidance for Decision Making on the Web: A Design Probe with WebSeek</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Arpit Narechania, Yanwei Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15100) • [📄 arXiv](https://arxiv.org/abs/2601.15100) • [📥 PDF](https://arxiv.org/pdf/2601.15100)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Developer Interaction Patterns with Proactive AI: A Five-Day Field Study (2...

</details>

<details>
<summary><b>23. Motion 3-to-4: 3D Motion Reconstruction for 4D Synthesis</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Anpei Chen, Zexiang Xu, Youjia Zhang, Xingyu Chen, Hongyuan Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.14253) • [📄 arXiv](https://arxiv.org/abs/2601.14253) • [📥 PDF](https://arxiv.org/pdf/2601.14253)

> No abstract available.

</details>

<details>
<summary><b>24. sangkuriang: A pseudo-spectral Python library for Korteweg-de Vries soliton simulation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.12029) • [📄 arXiv](https://arxiv.org/abs/2601.12029) • [📥 PDF](https://arxiv.org/pdf/2601.12029)

**💻 Code:** [⭐ Code](https://github.com/sandyherho/sangkuriang-ideal-solver)

> Korteweg-de Vries (KdV) equation serves as a foundational model in nonlinear wave physics, describing the balance between dispersive spreading and nonlinear steepening that gives rise to solitons. This article introduces sangkuriang, an open-sourc...

</details>

<details>
<summary><b>25. Show me the evidence: Evaluating the role of evidence and natural language explanations in AI-supported fact-checking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.11387) • [📄 arXiv](https://arxiv.org/abs/2601.11387) • [📥 PDF](https://arxiv.org/pdf/2601.11387)

> TL;DR: In an AI-supported fact-checking task, people consistently relied on underlying evidence to judge AI reliability, using explanations as a supplement rather than a substitute, showing that evidence is central to how people evaluate AI-aided ...

</details>

<details>
<summary><b>26. CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13262) • [📄 arXiv](https://arxiv.org/abs/2601.13262) • [📥 PDF](https://arxiv.org/pdf/2601.13262)

**💻 Code:** [⭐ Code](https://github.com/AikyamLab/cure-med)

> We introduce CURE-MED, a curriculum-informed reinforcement learning framework for multilingual medical reasoning across 13 languages, including low-resource settings. The work studies how code-switching-aware supervision and curriculum-guided RL j...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 26 |
| 📅 Today | [`2026-01-23.json`](data/daily/2026-01-23.json) | 26 |
| 📆 This Week | [`2026-W03.json`](data/weekly/2026-W03.json) | 129 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 558 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-23 | 26 | [View JSON](data/daily/2026-01-23.json) |
| 📄 2026-01-22 | 32 | [View JSON](data/daily/2026-01-22.json) |
| 📄 2026-01-21 | 11 | [View JSON](data/daily/2026-01-21.json) |
| 📄 2026-01-20 | 22 | [View JSON](data/daily/2026-01-20.json) |
| 📄 2026-01-19 | 38 | [View JSON](data/daily/2026-01-19.json) |
| 📄 2026-01-18 | 38 | [View JSON](data/daily/2026-01-18.json) |
| 📄 2026-01-17 | 38 | [View JSON](data/daily/2026-01-17.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W03 | 129 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 558 | [View JSON](data/monthly/2026-01.json) |
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
