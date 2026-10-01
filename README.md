<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-34-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6642+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">34</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">123</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">34</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6642+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 01, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Learning Meta-Skills for Agent Harness Design in Test-Time AI4AI</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38143) • [📄 arXiv](https://arxiv.org/abs/2609.38143) • [📥 PDF](https://arxiv.org/pdf/2609.38143)

**💻 Code:** [⭐ Code](https://github.com/qiancheng-apodex/MetaSkill-AI4AI) • [⭐ Code](https://github.com/huggingface)

> ArXiv: https://arxiv.org/pdf/2609.38143 Code: https://github.com/qiancheng-apodex/MetaSkill-AI4AI

</details>

<details>
<summary><b>2. The Teacher Is a Direction, Not a Destination: Extrapolating RL-Induced Representation Residuals in On-Policy Distillation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** vaynetian, Donghanark, renweijie, chenmeijia30, LH2101

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36484) • [📄 arXiv](https://arxiv.org/abs/2609.36484) • [📥 PDF](https://arxiv.org/pdf/2609.36484)

**💻 Code:** [⭐ Code](https://github.com/xixixixixxxx/RIDE) • [⭐ Code](https://github.com/huggingface)

> We introduce RIDE (RL-Induced Direction Extrapolation), an on-policy distillation method that treats an RL-trained teacher as a direction for learning. On student-generated trajectories, RIDE computes the layerwise hidden-state residual between th...

</details>

<details>
<summary><b>3. WorldAuditBench: Interactive 3D World Auditing with Multimodal Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40325) • [📄 arXiv](https://arxiv.org/abs/2609.40325) • [📥 PDF](https://arxiv.org/pdf/2609.40325)

**💻 Code:** [⭐ Code](https://github.com/UCSB-NLP-Chang/WorldAuditBench) • [⭐ Code](https://github.com/huggingface)

> We introduce WorldAuditBench, a benchmark with 213 tasks across 13 interactive 3D environments that tests whether multimodal agents can explore, investigate, and identify world anomalies.

</details>

<details>
<summary><b>4. More Choices, Fewer Decisions: Ordinal-Scale Bias in JEV-like Direct-Decision Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38827) • [📄 arXiv](https://arxiv.org/abs/2609.38827) • [📥 PDF](https://arxiv.org/pdf/2609.38827)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Direct-decision models turn text into low-latency structured labels and scores, making them attractive for classification and automatic evaluation. Yet reliability requires more than accuracy: a model must also use the ordinal decision scale suppl...

</details>

<details>
<summary><b>5. EvoDuet: Bilevel Co-Evolution of Web Searching and Task Solving for Scientific Discovery</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40340) • [📄 arXiv](https://arxiv.org/abs/2609.40340) • [📥 PDF](https://arxiv.org/pdf/2609.40340)

**💻 Code:** [⭐ Code](https://github.com/Open-Galapagos/EvoDuet) • [⭐ Code](https://github.com/huggingface)

> Excited to introduce  🎶 EvoDuet: co-evolving web search and task solutions to push the state of the art across 8 scientific optimization tasks.

</details>

<details>
<summary><b>6. Agent Error Dataset: Scaling 50,000 Error--Diagnosis Pairs for Failure Analysis and Error-Aware Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40111) • [📄 arXiv](https://arxiv.org/abs/2609.40111) • [📥 PDF](https://arxiv.org/pdf/2609.40111)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> An unsuccessful LLM agent rollout contains more information than its final reward: the observations available to the agent, the actions it chose, and the environment’s responses. Reusing this experience for learning requires identifying a decision...

</details>

<details>
<summary><b>7. Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies</b> ⭐ 560</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38537) • [📄 arXiv](https://arxiv.org/abs/2609.38537) • [📥 PDF](https://arxiv.org/pdf/2609.38537)

**💻 Code:** [⭐ Code](https://github.com/anonymous-report-421/GPT-as-Policy) • [⭐ Code](https://github.com/huggingface)

> Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies

</details>

<details>
<summary><b>8. DC-SAE: Deep Compression Semantic Autoencoder for Faster Diffusion Convergence</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39222) • [📄 arXiv](https://arxiv.org/abs/2609.39222) • [📥 PDF](https://arxiv.org/pdf/2609.39222)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DAGroup-PKU/DCSAE)

> High-compression tokenizers are essential for scaling latent image generative models. However, aggressive compression creates a fundamental tradeoff between reconstruction fidelity and generation efficiency: high compression image encoder always i...

</details>

<details>
<summary><b>9. Imagine3D-LLM: Teaching MLLMs to Imagine 3D Scenes Before Answering</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38177) • [📄 arXiv](https://arxiv.org/abs/2609.38177) • [📥 PDF](https://arxiv.org/pdf/2609.38177)

**💻 Code:** [⭐ Code](https://github.com/cvlab-kaist/Imagine3D-LLM) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. UniEvo-VL: An On-policy Self-Distillation Training Recipe for Multimodal Model Self-improvement</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38721) • [📄 arXiv](https://arxiv.org/abs/2609.38721) • [📥 PDF](https://arxiv.org/pdf/2609.38721)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi everyone, first author here! 👋 Can multimodal models improve image generation by learning from their own feedback? We introduce UniEvo-VL , an on-policy self-distillation training recipe for multimodal model self-improvement. Building on Qwen-i...

</details>

<details>
<summary><b>11. AIM: Agentic Idea Management for Automated Research</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38445) • [📄 arXiv](https://arxiv.org/abs/2609.38445) • [📥 PDF](https://arxiv.org/pdf/2609.38445)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Implementing and evaluating research ideas is expensive. As agents generate more candidates, choosing what to explore and learning from previous experiments becomes increasingly important. Numerous approaches have been proposed for automated resea...

</details>

<details>
<summary><b>12. RSIGame: Autonomous Agentic Game Development with Recursive Self-improvement</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39045) • [📄 arXiv](https://arxiv.org/abs/2609.39045) • [📥 PDF](https://arxiv.org/pdf/2609.39045)

**💻 Code:** [⭐ Code](https://github.com/WenyiWU0111/RSIGame) • [⭐ Code](https://github.com/huggingface)

> 🎮 From game generation to autonomous game improvement. Recent advances in large language models have made automatic game generation increasingly feasible, yet reliably improving generated games beyond a playable version remains challenging. We int...

</details>

<details>
<summary><b>13. RoboCoach: World Models as Active Coaches for Compositional Robot Skills</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39685) • [📄 arXiv](https://arxiv.org/abs/2609.39685) • [📥 PDF](https://arxiv.org/pdf/2609.39685)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RoboCoach-AI/CoachWorld)

> Long-horizon robot manipulation reuses skills across many task compositions, but improving these compositions with additional end-to-end demonstrations is costly. A practical self-improving system must decide both what to teach next and where to a...

</details>

<details>
<summary><b>14. ThinkV2V: Unleashing the Reasoning Capability of MLLMs for Instruction-Guided Video Editing</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Guisheng Liu, Hao Yang, Fan Zhang, Haoyang He, Donghao Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38541) • [📄 arXiv](https://arxiv.org/abs/2609.38541) • [📥 PDF](https://arxiv.org/pdf/2609.38541)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Correr-Zhou/ThinkV2V)

> Paper: https://arxiv.org/pdf/2609.38541 Code: https://github.com/Correr-Zhou/ThinkV2V Model (ThinkV2V-5B): https://huggingface.co/donghao-zhou/ThinkV2V-5B Dataset #1 (OpenVE-HQ-1M): https://huggingface.co/datasets/donghao-zhou/OpenVE-HQ-1M Dataset...

</details>

<details>
<summary><b>15. AREX-2: Advancing Self-Improving Agents through Long-Horizon Reflective Tasks</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38288) • [📄 arXiv](https://arxiv.org/abs/2609.38288) • [📥 PDF](https://arxiv.org/pdf/2609.38288)

**💻 Code:** [⭐ Code](https://github.com/VectorSpaceLab/AREX-2) • [⭐ Code](https://github.com/huggingface)

> Homepage: https://github.com/VectorSpaceLab/AREX-2 Models: https://huggingface.co/collections/BAAI/arex-2

</details>

<details>
<summary><b>16. DuoOPD: Learning from Joint Teacher-Student Outcomes for Multi-Task On-Policy Distillation</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Rui Li, Linan Yue, Heng Zhou, Weibo Gao, Ao Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33711) • [📄 arXiv](https://arxiv.org/abs/2609.33711) • [📥 PDF](https://arxiv.org/pdf/2609.33711)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/YongYuanDeAo/DuoOPD)

> Hi HF community! 👋 Co-author here. A stronger teacher can still get questions wrong that its student gets right. Standard on-policy distillation can push down correct student responses, and these teacher–student disagreements vary across tasks. We...

</details>

<details>
<summary><b>17. A2Z GameSpec-Bench: How Faithfully Can Coding Agents Generate Games from Game Design Specifications?</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39564) • [📄 arXiv](https://arxiv.org/abs/2609.39564) • [📥 PDF](https://arxiv.org/pdf/2609.39564)

**💻 Code:** [⭐ Code](https://github.com/krafton-ai/a2z-gamespec-bench) • [⭐ Code](https://github.com/huggingface)

> A2Z GameSpec-Bench evaluates how faithfully coding agents build complete games from 100 long-form Game Design Documents. Games may compile and execute successfully but fail to adhere to the intended design, especially when multiple requirements in...

</details>

<details>
<summary><b>18. Decompose Radicals, Then Reward: Fine-Grained Inspection for Accurate Chinese Text Rendering</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37569) • [📄 arXiv](https://arxiv.org/abs/2609.37569) • [📥 PDF](https://arxiv.org/pdf/2609.37569)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> IDSpect: fine-grained structural credit from  Ideographic Description Sequences (IDS). Big gains on LongText / GenTextEval.

</details>

<details>
<summary><b>19. Thinking Outside the Box: Can Language Models Rely on External Guidance Selectively?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fang kong, Yuxin Tao, Jinhang Zuo, Boyuan Wang, Minghan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39578) • [📄 arXiv](https://arxiv.org/abs/2609.39578) • [📥 PDF](https://arxiv.org/pdf/2609.39578)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Thinking Outside the Box: Can Language Models Rely on External Guidance Selectively? Language model agents increasingly rely on human designed workflows to solve complex tasks. Good workflows can substantially improve performance by providing usef...

</details>

<details>
<summary><b>20. Game-Guided Skill Discovery through Self-Play for Playable Agent Control</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sehoon Ha, Xue Bin Peng, Jeonghwan Kim, Seungeun Rho

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40137) • [📄 arXiv](https://arxiv.org/abs/2609.40137) • [📥 PDF](https://arxiv.org/pdf/2609.40137)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>21. Physis-Lang: Self-Evolving Language as a Physical Representation for Video World Model</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40358) • [📄 arXiv](https://arxiv.org/abs/2609.40358) • [📥 PDF](https://arxiv.org/pdf/2609.40358)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. WUSH-KV: KV Cache Quantization with Data-Adaptive Transforms</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38121) • [📄 arXiv](https://arxiv.org/abs/2609.38121) • [📥 PDF](https://arxiv.org/pdf/2609.38121)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> KV cache memory and bandwidth costs grow with context length and batch size, which limits efficient long-context inference. To address this bottleneck, we introduce WUSH-KV for low-bit KV-cache quantization. It adapts WUSH, which constructs a data...

</details>

<details>
<summary><b>23. PivotOPD: Learning to Recover from Pivotal Mistakes in Multi-Turn Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40285) • [📄 arXiv](https://arxiv.org/abs/2609.40285) • [📥 PDF](https://arxiv.org/pdf/2609.40285)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: Multi-turn agents don't fail everywhere. Most failed rollouts hinge on one early pivotal mistake , the mistake is usually recoverable, and standard on-policy distillation can't repair it because the student never samples the recovery action...

</details>

<details>
<summary><b>24. Working Around the Compute Ceiling: Byte-Exact Memory in Galahad Makes LLM Reading a One-Time Cost LLM Reading a One-Time Cost</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sietse Schelpe

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39358) • [📄 arXiv](https://arxiv.org/abs/2609.39358) • [📥 PDF](https://arxiv.org/pdf/2609.39358)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We built Galahad because we believe AI should have lasting memory that isn’t limited to what fits in a GPU’s VRAM. When models and agents repeatedly process the same context, we can end up paying for work they’ve already done. We wanted to keep th...

</details>

<details>
<summary><b>25. Decision-Oriented Recommendation Reranking: An Empirical Study of Jev</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40241) • [📄 arXiv](https://arxiv.org/abs/2609.40241) • [📥 PDF](https://arxiv.org/pdf/2609.40241)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Can decision-oriented models be a better fit for recommendation reranking than general-purpose LLMs? In this work, we study Jev, a decision-oriented “System One Model,” for personalized recommendation reranking. Across multiple domains and candida...

</details>

<details>
<summary><b>26. SkillSeek: Revisiting Agent Skill Retrieval at Marketplace Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38822) • [📄 arXiv](https://arxiv.org/abs/2609.38822) • [📥 PDF](https://arxiv.org/pdf/2609.38822)

**💻 Code:** [⭐ Code](https://github.com/guanqun-yang/SkillSeek) • [⭐ Code](https://github.com/huggingface)

> Accepted at AACL-IJCNLP 2026, Findings. An Agent Skill is a folder with instructions that teach an AI agent one specific job. The agent reads a short summary of each skill at startup and opens the full text only when the job looks relevant. Public...

</details>

<details>
<summary><b>27. CoEvoWhen: Policy-Tool Coevolution for Ultra-Long Video Temporal Grounding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40048) • [📄 arXiv](https://arxiv.org/abs/2609.40048) • [📥 PDF](https://arxiv.org/pdf/2609.40048)

**💻 Code:** [⭐ Code](https://github.com/aim-uofa/CoEvoWhen) • [⭐ Code](https://github.com/huggingface)

> We introduce CoEvoWhen, a policy-tool coevolution framework for ultra-long video temporal grounding. Our key idea is to jointly evolve high-level policies and executable media tools from agentic reasoning trajectories, forming a reusable skill wit...

</details>

<details>
<summary><b>28. EviRover: Reinforcing Agentic Perception Beyond a Glance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Manyuan Zhang, Yilei Jiang, Tianshuo Peng, Kaituo Feng, Kaixuan Fan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40230) • [📄 arXiv](https://arxiv.org/abs/2609.40230) • [📥 PDF](https://arxiv.org/pdf/2609.40230)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Visual perception is conventionally formulated as a one-shot prediction from a single glance at the image, under the assumption that the image content and the model's parametric knowledge suffice to resolve the query. This assumption often fails i...

</details>

<details>
<summary><b>29. Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.29142) • [📄 arXiv](https://arxiv.org/abs/2609.29142) • [📥 PDF](https://arxiv.org/pdf/2609.29142)

**💻 Code:** [⭐ Code](https://github.com/Luli3220/S2D-OPD) • [⭐ Code](https://github.com/huggingface)

> S²D-OPD identifies that dense Direct-OPD supervision can include low-value states and proposes divergence-based selective supervision, achieving better reasoning transfer with only 10% retained states.

</details>

<details>
<summary><b>30. See it, Say it, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of Emergent Misalignment in LLMs</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34970) • [📄 arXiv](https://arxiv.org/abs/2609.34970) • [📥 PDF](https://arxiv.org/pdf/2609.34970)

**💻 Code:** [⭐ Code](https://github.com/WeiqiaoQUE/mechanistic-emergent-misalignment) • [⭐ Code](https://github.com/huggingface)

> Safety-aligned LLMs can exhibit emergent misalignment (EM): narrow domain adaptation unexpectedly triggers catastrophic safety failures across unrelated domains. Prior static analyses leave training dynamics unmapped, while existing defenses rely ...

</details>

<details>
<summary><b>31. AdviSD: Learning to Advise Frontier LLMs via Targeted Multi-Turn Self-Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38142) • [📄 arXiv](https://arxiv.org/abs/2609.38142) • [📥 PDF](https://arxiv.org/pdf/2609.38142)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Training an advisor to the Frontier LLMs via targeted multi-turn self-distillation.

</details>

<details>
<summary><b>32. PatchHolmes: Agentic Patch Retrieval via Listwise Selection</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38807) • [📄 arXiv](https://arxiv.org/abs/2609.38807) • [📥 PDF](https://arxiv.org/pdf/2609.38807)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Aizhouym/PatchHolmes)

> Accepted at AACL-IJCNLP 2026, Main Conference. Every software vulnerability needs to be paired with the commit that fixed it. Security advisories need that pairing. So do severity scores, affected-version trackers, and supply-chain scanners. ❌ In ...

</details>

<details>
<summary><b>33. Understanding Multimodality in Generative Behavioral Cloning</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22493) • [📄 arXiv](https://arxiv.org/abs/2605.22493) • [📥 PDF](https://arxiv.org/pdf/2605.22493)

**💻 Code:** [⭐ Code](https://github.com/Lorenzo-Mazza/VersatIL) • [⭐ Code](https://github.com/huggingface)

> A little story about the paper: the idea goes back a while. Last December, I happened to be looking into ACT[Zhao]’s CVAE structure and its surprisingly high KL-regularization coefficient (β = 10 or even 100). Reading InfoVAE(an old gem by Stefano...

</details>

<details>
<summary><b>34. Almost Human, Except When It Matters: VoxParity and the Decisions a Voice Should Change</b> ⭐ 0</summary>

<br/>

**👥 Authors:** bhavikmangla

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35922) • [📄 arXiv](https://arxiv.org/abs/2609.35922) • [📥 PDF](https://arxiv.org/pdf/2609.35922)

**💻 Code:** [⭐ Code](https://github.com/bhavik-mangla/voxparity-bench) • [⭐ Code](https://github.com/huggingface)

> A caller agrees to a power disconnection while a medical monitor beeps in the background. All four leading systems we tested identify the monitor in their own probe; three schedule the disconnection, and none of the 28 systems applies the vulnerab...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 34 |
| 📅 Today | [`2026-10-01.json`](data/daily/2026-10-01.json) | 34 |
| 📆 This Week | [`2026-W39.json`](data/weekly/2026-W39.json) | 123 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 34 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-01 | 34 | [View JSON](data/daily/2026-10-01.json) |
| 📄 2026-09-30 | 45 | [View JSON](data/daily/2026-09-30.json) |
| 📄 2026-09-29 | 37 | [View JSON](data/daily/2026-09-29.json) |
| 📄 2026-09-28 | 7 | [View JSON](data/daily/2026-09-28.json) |
| 📄 2026-09-27 | 22 | [View JSON](data/daily/2026-09-27.json) |
| 📄 2026-09-26 | 22 | [View JSON](data/daily/2026-09-26.json) |
| 📄 2026-09-25 | 13 | [View JSON](data/daily/2026-09-25.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W39 | 123 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |
| 📅 2026-W36 | 88 | [View JSON](data/weekly/2026-W36.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 34 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |

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
