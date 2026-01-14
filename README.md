<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-42-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1040+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">42</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">105</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">302</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1040+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 14, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Watching, Reasoning, and Searching: A Video Deep Research Benchmark on Open Web for Agentic Video Reasoning</b> ⭐ 51</summary>

<br/>

**👥 Authors:** Zhe Huang, Zhuoyue Chang, HJH2CMD, Yu2020, POTATO66

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06943) • [📄 arXiv](https://arxiv.org/abs/2601.06943) • [📥 PDF](https://arxiv.org/pdf/2601.06943)

**💻 Code:** [⭐ Code](https://github.com/QuantaAlpha/VideoDR-Benchmark)

> First video deep research benchmark.

</details>

<details>
<summary><b>2. BabyVision: Visual Reasoning Beyond Language</b> ⭐ 81</summary>

<br/>

**👥 Authors:** Liang Chen, Liuff23, Ziqi, ssz1111, chenxz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06521) • [📄 arXiv](https://arxiv.org/abs/2601.06521) • [📥 PDF](https://arxiv.org/pdf/2601.06521)

**💻 Code:** [⭐ Code](https://github.com/UniPat-AI/BabyVision)

> Feel free to follow our GitHub repo: https://github.com/UniPat-AI/BabyVision

</details>

<details>
<summary><b>3. PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning</b> ⭐ 261</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05593) • [📄 arXiv](https://arxiv.org/abs/2601.05593) • [📥 PDF](https://arxiv.org/pdf/2601.05593)

**💻 Code:** [⭐ Code](https://github.com/stepfun-ai/PaCoRe)

> 🎉 Introducing Parallel Coordinated Reasoning (PaCoRe) 📈 An 8B model beats GPT-5 on HMMT25 by unlocking parallel thinking for test-time scaling! 📂 Open-source deep think: data + model + inference code! 🆓 MIT-licensed — use it however you want 🔍Key ...

</details>

<details>
<summary><b>4. MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head</b> ⭐ 47</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07832) • [📄 arXiv](https://arxiv.org/abs/2601.07832) • [📥 PDF](https://arxiv.org/pdf/2601.07832)

**💻 Code:** [⭐ Code](https://github.com/DAGroup-PKU/MHLA)

> No abstract available.

</details>

<details>
<summary><b>5. X-Coder: Advancing Competitive Programming with Fully Synthetic Tasks, Solutions, and Tests</b> ⭐ 52</summary>

<br/>

**👥 Authors:** Jane Luo, Jiani Guo, Xin Zhang, Jie Wu, Ringo1110

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06953) • [📄 arXiv](https://arxiv.org/abs/2601.06953) • [📥 PDF](https://arxiv.org/pdf/2601.06953)

**💻 Code:** [⭐ Code](https://github.com/JieWu02/X-Coder)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Tailored Primitive Initialization is the Secret Key to Reinforcement Learni...

</details>

<details>
<summary><b>6. GlimpRouter: Efficient Collaborative Inference by Glimpsing One Token of Thoughts</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05110) • [📄 arXiv](https://arxiv.org/abs/2601.05110) • [📥 PDF](https://arxiv.org/pdf/2601.05110)

**💻 Code:** [⭐ Code](https://github.com/Zengwh02/GlimpRouter)

> LLM + SLM > LLM

</details>

<details>
<summary><b>7. Lost in the Noise: How Reasoning Models Fail with Contextual Distractors</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07226) • [📄 arXiv](https://arxiv.org/abs/2601.07226) • [📥 PDF](https://arxiv.org/pdf/2601.07226)

> The code and dataset will be released publicly.

</details>

<details>
<summary><b>8. OS-Symphony: A Holistic Framework for Robust and Generalist Computer-Using Agent</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07779) • [📄 arXiv](https://arxiv.org/abs/2601.07779) • [📥 PDF](https://arxiv.org/pdf/2601.07779)

**💻 Code:** [⭐ Code](https://github.com/OS-Copilot/OS-Symphony)

> Despite VLM advances, current CUA frameworks remain brittle in long-horizon workflows and weak in novel domains due to coarse historical visual context management and missing visual-aware tutorial retrieval, so we propose OS-SYMPHONY, an orchestra...

</details>

<details>
<summary><b>9. Beyond Hard Masks: Progressive Token Evolution for Diffusion Language Models</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Chenchen Jing, Tianjian Feng, Bozhen Fang, Linyu Wu, zhongzero

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07351) • [📄 arXiv](https://arxiv.org/abs/2601.07351) • [📥 PDF](https://arxiv.org/pdf/2601.07351)

**💻 Code:** [⭐ Code](https://github.com/aim-uofa/EvoTokenDLM)

> GitHub repo: https://github.com/aim-uofa/EvoTokenDLM

</details>

<details>
<summary><b>10. Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhengkang Guo, Jingwen Xu, Xiaohua Wang, Muzhao Tian, zisuh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05107) • [📄 arXiv](https://arxiv.org/abs/2601.05107) • [📥 PDF](https://arxiv.org/pdf/2601.05107)

> As LLM-based agents are increasingly used in long-term interactions, cumulative memory is critical for enabling personalization and maintaining stylistic consistency. However, most existing systems adopt an ``all-or-nothing'' approach to memory us...

</details>

<details>
<summary><b>11. DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01528) • [📄 arXiv](https://arxiv.org/abs/2601.01528) • [📥 PDF](https://arxiv.org/pdf/2601.01528)

**💻 Code:** [⭐ Code](https://github.com/youngzhou1999/DrivingGen)

> DrivingGen is a comprehensive benchmark for generative world models in the driving domain with a diverse data distribution and novel evaluation metrics.

</details>

<details>
<summary><b>12. MegaFlow: Large-Scale Distributed Orchestration System for the Agentic Era</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiawei Chen, Ruisheng Cao, Mouxiang Chen, zjj1233, Lemoncoke

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07526) • [📄 arXiv](https://arxiv.org/abs/2601.07526) • [📥 PDF](https://arxiv.org/pdf/2601.07526)

> The rapid development of interactive and autonomous AI systems signals our entry into the agentic era. Training and evaluating agents on complex agentic tasks such as software engineering and computer use requires not only efficient model computat...

</details>

<details>
<summary><b>13. Boosting Latent Diffusion Models via Disentangled Representation Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05823) • [📄 arXiv](https://arxiv.org/abs/2601.05823) • [📥 PDF](https://arxiv.org/pdf/2601.05823)

**💻 Code:** [⭐ Code](https://github.com/Kwai-Kolors/Send-VAE)

> arXiv link: Boosting Latent Diffusion Models via Disentangled Representation Alignment Code (Coming Soon): https://github.com/Kwai-Kolors/Send-VAE

</details>

<details>
<summary><b>14. What Users Leave Unsaid: Under-Specified Queries Limit Vision-Language Models</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06165) • [📄 arXiv](https://arxiv.org/abs/2601.06165) • [📥 PDF](https://arxiv.org/pdf/2601.06165)

**💻 Code:** [⭐ Code](https://github.com/HAE-RAE/HAERAE-VISION)

> Users often ask VLMs under-specified, informal visual questions, which current clean-prompt benchmarks fail to capture. We introduce HAERAE-Vision (653 real Korean community queries + explicit rewrites) and show that making queries explicit boosts...

</details>

<details>
<summary><b>15. ET-Agent: Incentivizing Effective Tool-Integrated Reasoning Agent via Behavior Calibration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06860) • [📄 arXiv](https://arxiv.org/abs/2601.06860) • [📥 PDF](https://arxiv.org/pdf/2601.06860)

> Most current TIR work only focuses on the accuracy of agents in downstream tasks, while lacking calibration of the agents' behavioral patterns in TIR tasks. To address this issue, we first quantitatively analyze several possible erroneous behavior...

</details>

<details>
<summary><b>16. Dr. Zero: Self-Evolving Search Agents without Training Data</b> ⭐ 74</summary>

<br/>

**👥 Authors:** Shaoliang Nie, Suyu Ge, Xianjun Yang, Kartikeya Upasani, Zhenrui Yue

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07055) • [📄 arXiv](https://arxiv.org/abs/2601.07055) • [📥 PDF](https://arxiv.org/pdf/2601.07055)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/drzero)

> Dr. Zero enables data-free self-evolving search agents through a self-evolution loop with HRPO, achieving strong multi-step reasoning while reducing compute.

</details>

<details>
<summary><b>17. Forest Before Trees: Latent Superposition for Efficient Visual Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yankai Lin, Yichen Wu, Yubo Wang, Yuhan, ZION121

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06803) • [📄 arXiv](https://arxiv.org/abs/2601.06803) • [📥 PDF](https://arxiv.org/pdf/2601.06803)

> We hope this work encourages a paradigm shift from explicit next-token prediction to latent visual reasoning.

</details>

<details>
<summary><b>18. TourPlanner: A Competitive Consensus Framework with Constraint-Gated Reinforcement Learning for Travel Planning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hao Wang, Xiaoxi Li, Wenxiang Jiao, Mining Tan, Yinuo Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04698) • [📄 arXiv](https://arxiv.org/abs/2601.04698) • [📥 PDF](https://arxiv.org/pdf/2601.04698)

> We propose TourPlanner , a comprehensive framework featuring multi-path reasoning and constraint-gated reinforcement learning. Specifically, we first introduce a Personalized Recall and Spatial Optimization (PReSO) workflow to construct spatially-...

</details>

<details>
<summary><b>19. OpenTinker: Separating Concerns in Agentic Reinforcement Learning</b> ⭐ 568</summary>

<br/>

**👥 Authors:** Jiaxuan You, zsqzz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07376) • [📄 arXiv](https://arxiv.org/abs/2601.07376) • [📥 PDF](https://arxiv.org/pdf/2601.07376)

**💻 Code:** [⭐ Code](https://github.com/open-tinker/OpenTinker?tab=readme-ov-file) • [⭐ Code](https://github.com/open-tinker/OpenTinker)

> 🎉 Introducing OpenTinker 🚀 A scalable RL infrastructure for LLM agents that separates what you build (agents + environments) from how it runs (training + inference)! 🧩 Composable RL-as-a-Service No more monolithic RL pipelines. OpenTinker decompos...

</details>

<details>
<summary><b>20. Are LLM Decisions Faithful to Verbal Confidence?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07767) • [📄 arXiv](https://arxiv.org/abs/2601.07767) • [📥 PDF](https://arxiv.org/pdf/2601.07767)

> While LLMs can express their confidence levels, their actual decisions do not demonstrate risk sensitivity. Even with high error penalties, they rarely abstain from making choices, often leading to utility collapse.

</details>

<details>
<summary><b>21. Structured Episodic Event Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06411) • [📄 arXiv](https://arxiv.org/abs/2601.06411) • [📥 PDF](https://arxiv.org/pdf/2601.06411)

> Current approaches to memory in Large Language Models (LLMs) predominantly rely on static Retrieval-Augmented Generation (RAG), which often results in scattered retrieval and fails to capture the structural dependencies required for complex reason...

</details>

<details>
<summary><b>22. e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhicheng Dou, Tetsuya Sakai, Radu Timofte, Sicheng Gao, Haon-Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03666) • [📄 arXiv](https://arxiv.org/abs/2601.03666) • [📥 PDF](https://arxiv.org/pdf/2601.03666)

> A lightweight explicit alignment recipe that adapts off-the-shelf VLMs into robust omni-modal embedding models. Checkpoints: https://huggingface.co/Haon-Chen/e5-omni-3B https://huggingface.co/Haon-Chen/e5-omni-7B

</details>

<details>
<summary><b>23. "TODO: Fix the Mess Gemini Created": Towards Understanding GenAI-Induced Self-Admitted Technical Debt</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mia Mohammad Imran, Abdullah Al Mujahid

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07786) • [📄 arXiv](https://arxiv.org/abs/2601.07786) • [📥 PDF](https://arxiv.org/pdf/2601.07786)

> As large language models (LLMs) such as ChatGPT, Copilot, Claude, and Gemini become integrated into software development workflows, developers increasingly leave traces of AI involvement in their code comments. Among these, some comments explicitl...

</details>

<details>
<summary><b>24. ShowUI-Aloha: Human-Taught GUI Agent</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhiheng Chen, Jessica Hu, Yauhong Goh, Xiangwu Guo, Yichun Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07181) • [📄 arXiv](https://arxiv.org/abs/2601.07181) • [📥 PDF](https://arxiv.org/pdf/2601.07181)

> No abstract available.

</details>

<details>
<summary><b>25. Codified Foreshadowing-Payoff Text Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jingbo Shang, Letian Peng, Kun Zhou, Longfei Yun, hyp1231

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07033) • [📄 arXiv](https://arxiv.org/abs/2601.07033) • [📥 PDF](https://arxiv.org/pdf/2601.07033)

> Codified Foreshadowing-Payoff Text Generation

</details>

<details>
<summary><b>26. Sci-Reasoning: A Dataset Decoding AI Innovation Patterns</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04577) • [📄 arXiv](https://arxiv.org/abs/2601.04577) • [📥 PDF](https://arxiv.org/pdf/2601.04577)

**💻 Code:** [⭐ Code](https://github.com/AmberLJC/Sci-Reasoning)

> While AI innovation accelerates rapidly, the intellectual process behind breakthroughs -- how researchers identify gaps, synthesize prior work, and generate insights -- remains poorly understood. The lack of structured data on scientific reasoning...

</details>

<details>
<summary><b>27. How Do Large Language Models Learn Concepts During Continual Pre-Training?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zaishuo Xia, Minqian Liu, Yunzhi Yao, Sha Li, Barry Menglong Yao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03570) • [📄 arXiv](https://arxiv.org/abs/2601.03570) • [📥 PDF](https://arxiv.org/pdf/2601.03570)

> Human beings primarily understand the world through concepts (e.g., dog), abstract mental representations that structure perception, reasoning, and learning. However, how large language models (LLMs) acquire, retain, and forget such concepts durin...

</details>

<details>
<summary><b>28. On the Non-decoupling of Supervised Fine-tuning and Reinforcement Learning in Post-training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Weixi Zhang, Wei Han, Bo Bai, Xueyan Niu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07389) • [📄 arXiv](https://arxiv.org/abs/2601.07389) • [📥 PDF](https://arxiv.org/pdf/2601.07389)

> Post-training of large language models routinely interleaves supervised fine-tuning (SFT) with reinforcement learning (RL). These two methods have different objectives: SFT minimizes the cross-entropy loss between model outputs and expert response...

</details>

<details>
<summary><b>29. Can Textual Reasoning Improve the Performance of MLLMs on Fine-grained Visual Classification?</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Xiaoming Liu, Yiyang Su, Paipile

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06993) • [📄 arXiv](https://arxiv.org/abs/2601.06993) • [📥 PDF](https://arxiv.org/pdf/2601.06993)

**💻 Code:** [⭐ Code](https://github.com/jiezhu23/ReFine-RFT)

> In this work, we investigate the impact of CoT on Fine-Grained Visual Classification (FGVC), revealing a paradox: the degradation in FGVC performance due to CoT is primarily driven by reasoning length, with longer textual reasoning consistently re...

</details>

<details>
<summary><b>30. RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shaolei Zhang, Zishan Xu, Sen Hu, Zhiyuan Yao, Haonan-Bian

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06966) • [📄 arXiv](https://arxiv.org/abs/2601.06966) • [📥 PDF](https://arxiv.org/pdf/2601.06966)

**💻 Code:** [⭐ Code](https://github.com/AvatarMemory/RealMemBench)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API EvolMem: A Cognitive-Driven Benchmark for Multi-Session Dialogue Memory (20...

</details>

<details>
<summary><b>31. SketchJudge: A Diagnostic Benchmark for Grading Hand-drawn Diagrams with Multimodal Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shixing Li, Guozhang Li, Yaoyao Zhong, Mei Wang, Yuhang Su

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06944) • [📄 arXiv](https://arxiv.org/abs/2601.06944) • [📥 PDF](https://arxiv.org/pdf/2601.06944)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API ViRectify: A Challenging Benchmark for Video Reasoning Correction with Mult...

</details>

<details>
<summary><b>32. Artificial Entanglement in the Fine-Tuning of Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Manling Li, Zeguan Wu, Canyu Chen, Zihan Wang, Min Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06788) • [📄 arXiv](https://arxiv.org/abs/2601.06788) • [📥 PDF](https://arxiv.org/pdf/2601.06788)

> No abstract available.

</details>

<details>
<summary><b>33. FinForge: Semi-Synthetic Financial Benchmark Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06747) • [📄 arXiv](https://arxiv.org/abs/2601.06747) • [📥 PDF](https://arxiv.org/pdf/2601.06747)

> This paper introduces FinForge, a novel framework designed to address the scarcity of high-quality, domain-specific datasets for evaluating Large Language Models (LLMs) in finance. The authors propose a scalable, semi-synthetic pipeline that combi...

</details>

<details>
<summary><b>34. Gecko: An Efficient Neural Architecture Inherently Processing Sequences with Arbitrary Lengths</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06463) • [📄 arXiv](https://arxiv.org/abs/2601.06463) • [📥 PDF](https://arxiv.org/pdf/2601.06463)

**💻 Code:** [⭐ Code](https://github.com/XuezheMax/gecko-llm)

> No abstract available.

</details>

<details>
<summary><b>35. Does Inference Scaling Improve Reasoning Faithfulness? A Multi-Model Analysis of Self-Consistency Tradeoffs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Deep Mehta

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06423) • [📄 arXiv](https://arxiv.org/abs/2601.06423) • [📥 PDF](https://arxiv.org/pdf/2601.06423)

> We ask a question that hasn't been studied before: does inference scaling improve reasoning faithfulness or just accuracy? Self-consistency (majority voting over multiple reasoning paths) reliably boosts LLM accuracy on reasoning tasks. But does g...

</details>

<details>
<summary><b>36. FlyPose: Towards Robust Human Pose Estimation From Aerial Views</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Peter St\ütz, Marvin Brenner, farooqhassaan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05747) • [📄 arXiv](https://arxiv.org/abs/2601.05747) • [📥 PDF](https://arxiv.org/pdf/2601.05747)

**💻 Code:** [⭐ Code](https://github.com/farooqhassaan/FlyPose)

> Unmanned Aerial Vehicles (UAVs) are increasingly deployed in close proximity to humans for applications such as parcel delivery, traffic monitoring, disaster response and infrastructure inspections. Ensuring safe and reliable operation in these hu...

</details>

<details>
<summary><b>37. Benchmarking Small Language Models and Small Reasoning Language Models on System Log Severity Classification</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Chaowei Yang, Joseph Rogers, Zifu Wang, Emily Ma, ymasri

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07790) • [📄 arXiv](https://arxiv.org/abs/2601.07790) • [📥 PDF](https://arxiv.org/pdf/2601.07790)

**💻 Code:** [⭐ Code](https://github.com/stccenter/Benchmarking-SLMs-and-SRLMs-on-System-Log-Severity-Classification)

> We evaluate 9 open-source models under zero-shot, few-shot, and RAG (FAISS) and measure both accuracy + per-log latency. Main takeaway: RAG can massively help small models (Qwen3-4B: 95.64%, Gemma3-1B: 85.28%), but some reasoning-focused models de...

</details>

<details>
<summary><b>38. Stochastic CHAOS: Why Deterministic Inference Kills, and Distributional Variability Is the Heartbeat of Artifical Cognition</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shreyash Dhoot, Aadi Pandey, Anusa Saha, Shourya Aggarwal, Tanmay Joshi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07239) • [📄 arXiv](https://arxiv.org/abs/2601.07239) • [📥 PDF](https://arxiv.org/pdf/2601.07239)

> Stochastic CHAOS: Why Deterministic Inference Kills, and Distributional Variability Is the Heartbeat of Artifical Cognition

</details>

<details>
<summary><b>39. 3D CoCa v2: Contrastive Learners with Test-Time Search for Generalizable Spatial Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06496) • [📄 arXiv](https://arxiv.org/abs/2601.06496) • [📥 PDF](https://arxiv.org/pdf/2601.06496)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/3DCoCav2)

> https://github.com/AIGeeksGroup/3DCoCav2

</details>

<details>
<summary><b>40. On the Fallacy of Global Token Perplexity in Spoken Language Model Evaluation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ju-Chieh Chou, Yen-Chun Kuo, Yi-Cheng Lin, Liang-Hsuan Tseng, Jeff Chan-Jan Sju

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06329) • [📄 arXiv](https://arxiv.org/abs/2601.06329) • [📥 PDF](https://arxiv.org/pdf/2601.06329)

> Generative spoken language models pretrained on large-scale raw audio can continue a speech prompt with appropriate content while preserving attributes like speaker and emotion, serving as foundation models for spoken dialogue. In prior literature...

</details>

<details>
<summary><b>41. A Rising Tide Lifts All Boats: MTQE Rewards for Idioms Improve General Translation Quality</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dilek Hakkani-Tür, Dhruva Patil, Zhenlin He, Ishika Agarwal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06307) • [📄 arXiv](https://arxiv.org/abs/2601.06307) • [📥 PDF](https://arxiv.org/pdf/2601.06307)

> https://huggingface.co/collections/ishikaa/a-rising-tide-lifts-all-boats-mtqe-rewards-for-idioms

</details>

<details>
<summary><b>42. SPINAL -- Scaling-law and Preference Integration in Neural Alignment Layers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aman Chadha, Vinija Jain, Amit Dhanda, Partha Pratim Saha, Arion Das

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06238) • [📄 arXiv](https://arxiv.org/abs/2601.06238) • [📥 PDF](https://arxiv.org/pdf/2601.06238)

> SPINAL -- Scaling-law and Preference Integration in Neural Alignment Layers

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 42 |
| 📅 Today | [`2026-01-14.json`](data/daily/2026-01-14.json) | 42 |
| 📆 This Week | [`2026-W02.json`](data/weekly/2026-W02.json) | 105 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 302 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-14 | 42 | [View JSON](data/daily/2026-01-14.json) |
| 📄 2026-01-13 | 30 | [View JSON](data/daily/2026-01-13.json) |
| 📄 2026-01-12 | 33 | [View JSON](data/daily/2026-01-12.json) |
| 📄 2026-01-11 | 33 | [View JSON](data/daily/2026-01-11.json) |
| 📄 2026-01-10 | 33 | [View JSON](data/daily/2026-01-10.json) |
| 📄 2026-01-09 | 20 | [View JSON](data/daily/2026-01-09.json) |
| 📄 2026-01-08 | 26 | [View JSON](data/daily/2026-01-08.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W02 | 105 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 302 | [View JSON](data/monthly/2026-01.json) |
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
