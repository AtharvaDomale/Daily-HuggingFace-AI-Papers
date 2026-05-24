<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-49-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4194+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">49</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">183</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">573</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4194+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 24, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. DelTA: Discriminative Token Credit Assignment for Reinforcement Learning from Verifiable Rewards</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Yankai Lin, Wei Wu, Cardlnal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21467) • [📄 arXiv](https://arxiv.org/abs/2605.21467) • [📥 PDF](https://arxiv.org/pdf/2605.21467)

**💻 Code:** [⭐ Code](https://github.com/RUCBM/DelTA) • [⭐ Code](https://github.com/huggingface)

> We show that RLVR updates implicitly define a discriminator over token-gradient directions. This view suggests a reverse-design principle: better RLVR updates can be obtained by improving the discriminator induced by the update itself. Inspired by...

</details>

<details>
<summary><b>2. TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation</b> ⭐ 112</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22355) • [📄 arXiv](https://arxiv.org/abs/2605.22355) • [📥 PDF](https://arxiv.org/pdf/2605.22355)

**💻 Code:** [⭐ Code](https://github.com/HotTricker/TransitLM) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HotTricker/TransitLM.git)

> https://github.com/HotTricker/TransitLM.git

</details>

<details>
<summary><b>3. Perception or Prejudice: Can MLLMs Go Beyond First Impressions of Personality?</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22109) • [📄 arXiv](https://arxiv.org/abs/2605.22109) • [📥 PDF](https://arxiv.org/pdf/2605.22109)

**💻 Code:** [⭐ Code](https://github.com/kkkcx/MM-OCEAN) • [⭐ Code](https://github.com/huggingface)

> 👋 Hello AI Community! We are the authors of Perception or Prejudice . Do Multimodal LLMs truly understand human personality, or do they just get the right score for the wrong reason? 🤔 Our paper introduces MM-OCEAN , a multi-granularity benchmark ...

</details>

<details>
<summary><b>4. π-Bench: Evaluating Proactive Personal Assistant Agents in Long-Horizon Workflows</b> ⭐ 35</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14678) • [📄 arXiv](https://arxiv.org/abs/2605.14678) • [📥 PDF](https://arxiv.org/pdf/2605.14678)

**💻 Code:** [⭐ Code](https://github.com/Simplified-Reasoning/Pi-Bench) • [⭐ Code](https://github.com/huggingface)

> The rise of personal assistant agents, e.g., OpenClaw, highlights the growing potential of large language models to support users across everyday life and work. A core challenge in these settings is proactive assistance, since users often begin wi...

</details>

<details>
<summary><b>5. Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.16928) • [📄 arXiv](https://arxiv.org/abs/2605.16928) • [📥 PDF](https://arxiv.org/pdf/2605.16928)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Long-context inference in large language models is bottlenecked by the quadratic cost of full attention. Existing efficient alternatives often rely either on native sparse training or on heuristic token eviction, creating an undesirable trade-off ...

</details>

<details>
<summary><b>6. ACC: Compiling Agent Trajectories for Long-Context Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21850) • [📄 arXiv](https://arxiv.org/abs/2605.21850) • [📥 PDF](https://arxiv.org/pdf/2605.21850)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Recent development of agents has renewed demand for long-context reasoning capacity of LLMs. However, training LLMs for this capacity requires costly long-document curation or heuristic context synthesis. We observe that agents produce massive tra...

</details>

<details>
<summary><b>7. PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects</b> ⭐ 84</summary>

<br/>

**👥 Authors:** Fangzhou Hong, Runmao Yao, Haitian Li, Yinghao Liu, Ziang Cao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21572) • [📄 arXiv](https://arxiv.org/abs/2605.21572) • [📥 PDF](https://arxiv.org/pdf/2605.21572)

**💻 Code:** [⭐ Code](https://github.com/physx-omni/PhysX-Omni) • [⭐ Code](https://github.com/huggingface)

> Simulation-ready physical 3D assets have emerged as a promising direction owing to their broad applicability in downstream tasks. However, most existing 3D generation methods either neglect physical properties or are limited to a single asset cate...

</details>

<details>
<summary><b>8. LatentOmni: Rethinking Omni-Modal Understanding via Unified Audio-Visual Latent Reasoning</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22012) • [📄 arXiv](https://arxiv.org/abs/2605.22012) • [📥 PDF](https://arxiv.org/pdf/2605.22012)

**💻 Code:** [⭐ Code](https://github.com/yfanDai/LatentOmni) • [⭐ Code](https://github.com/huggingface)

> Paper link: https://arxiv.org/abs/2605.22012

</details>

<details>
<summary><b>9. Forecasting Scientific Progress with Artificial Intelligence</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22681) • [📄 arXiv](https://arxiv.org/abs/2605.22681) • [📥 PDF](https://arxiv.org/pdf/2605.22681)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SeanWu25/cusp-scientific-foresight)

> Can AI forecast scientific discovery, or just recognize past science? We built CUSP to test this rigorously. 4,760 real breakthroughs across biology, chemistry, physics, medicine, and AI, with verified knowledge cutoffs. Four forecasting tasks per...

</details>

<details>
<summary><b>10. SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22668) • [📄 arXiv](https://arxiv.org/abs/2605.22668) • [📥 PDF](https://arxiv.org/pdf/2605.22668)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TLDR: SEGA is a training-free method that uses spectral guidance to modify attention behavior through RoPE components scaling, improving high-resolution generation in diffusion transformers.

</details>

<details>
<summary><b>11. WorldKV: Efficient World Memory with World Retrieval and Compression</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22718) • [📄 arXiv](https://arxiv.org/abs/2605.22718) • [📥 PDF](https://arxiv.org/pdf/2605.22718)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project Page: https://cvlab-kaist.github.io/WorldKV/

</details>

<details>
<summary><b>12. Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22642) • [📄 arXiv](https://arxiv.org/abs/2605.22642) • [📥 PDF](https://arxiv.org/pdf/2605.22642)

**💻 Code:** [⭐ Code](https://github.com/Spreadsheet-RL/Spreadsheet-RL) • [⭐ Code](https://github.com/huggingface)

> Spreadsheet-RL is an RL fine-tuning framework and benchmarking environment designed to improve LLM agent performance on complex, multi-step spreadsheet tasks within Microsoft Excel.

</details>

<details>
<summary><b>13. SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22536) • [📄 arXiv](https://arxiv.org/abs/2605.22536) • [📥 PDF](https://arxiv.org/pdf/2605.22536)

**💻 Code:** [⭐ Code](https://github.com/Visionary-Laboratory/SpaceDG) • [⭐ Code](https://github.com/huggingface)

> We introduce SpaceDG, the first large-scale dataset for degradation-aware spatial intelligence, and SpaceDG-Bench, a human-verified benchmark for evaluating MLLMs under visual degradations 🔥

</details>

<details>
<summary><b>14. FlowLong: Inference-time Long Video Generation via Manifold-constrained Tweedie Matching</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20910) • [📄 arXiv](https://arxiv.org/abs/2605.20910) • [📥 PDF](https://arxiv.org/pdf/2605.20910)

**💻 Code:** [⭐ Code](https://github.com/jhq1234/flowlong) • [⭐ Code](https://github.com/huggingface)

> FlowLong is a training-free, model-agnostic inference-time method that extends pretrained flow-based video diffusion models beyond their native generation horizon — works uniformly for text-to-video, audio-video joint, and text-to-3D scene generat...

</details>

<details>
<summary><b>15. Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22809) • [📄 arXiv](https://arxiv.org/abs/2605.22809) • [📥 PDF](https://arxiv.org/pdf/2605.22809)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Accepted by CVPR 2026

</details>

<details>
<summary><b>16. Unsupervised Process Reward Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Maria Brbic, Hang Guo, Maxim Kodryan, Artyom Gadetsky, sibasmarakp

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.10158) • [📄 arXiv](https://arxiv.org/abs/2605.10158) • [📥 PDF](https://arxiv.org/pdf/2605.10158)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Process Reward Models (PRMs) are a powerful mechanism for steering large language model reasoning by providing fine-grained, step-level supervision. However, this effectiveness comes at a significant cost: PRMs require expert annotations for every...

</details>

<details>
<summary><b>17. Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention</b> ⭐ 110</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22791) • [📄 arXiv](https://arxiv.org/abs/2605.22791) • [📥 PDF](https://arxiv.org/pdf/2605.22791)

**💻 Code:** [⭐ Code](https://github.com/NVlabs/GatedDeltaNet-2) • [⭐ Code](https://github.com/huggingface)

> Gated DeltaNet-2 introduces a linear attention architecture that improves memory management by decoupling erase and write operations, achieving superior performance on long-context benchmarks compared to existing recurrent models.

</details>

<details>
<summary><b>18. Q-ARVD: Quantizing Autoregressive Video Diffusion Models</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21072) • [📄 arXiv](https://arxiv.org/abs/2605.21072) • [📥 PDF](https://arxiv.org/pdf/2605.21072)

**💻 Code:** [⭐ Code](https://github.com/tsa18/Q-ARVD) • [⭐ Code](https://github.com/huggingface)

> Q-ARVD proposes the first quantization framework tailored for autoregressive video diffusion models. It introduces a final-quality guided frame-weighting mechanism to handle the unbalanced frame-wise quantization sensitivity, and an outlier-aware ...

</details>

<details>
<summary><b>19. Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Zhengxi Lu, Yuhao Shen, Ruihan Jin, Guocheng Zhai, Jinyang23

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22177) • [📄 arXiv](https://arxiv.org/abs/2605.22177) • [📥 PDF](https://arxiv.org/pdf/2605.22177)

**💻 Code:** [⭐ Code](https://github.com/jinyangwu/Maestro) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>20. Training Large Language Models to Predict Clinical Events</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.12817) • [📄 arXiv](https://arxiv.org/abs/2605.12817) • [📥 PDF](https://arxiv.org/pdf/2605.12817)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Longitudinal clinical notes contain rich evidence of how patients evolve over time, but converting this signal into training supervision for clinical prediction remains challenging. We extend Foresight Learning to clinical prediction by converting...

</details>

<details>
<summary><b>21. AutoRubric-T2I: Robust Rule-Based Reward Model for Text-to-Image Alignment</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.17602) • [📄 arXiv](https://arxiv.org/abs/2605.17602) • [📥 PDF](https://arxiv.org/pdf/2605.17602)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/johnsonkao0213/AutoRubric-T2I)

> AutoRubric-T2I learns a compact set of weighted natural-language rubrics from image preference data, enabling interpretable VLM-based reward modeling without fine-tuning—using less than 0.01% of the annotated preference data that standard reward m...

</details>

<details>
<summary><b>22. GenEvolve: Self-Evolving Image Generation Agents via Tool-Orchestrated Visual Experience Distillation</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21605) • [📄 arXiv](https://arxiv.org/abs/2605.21605) • [📥 PDF](https://arxiv.org/pdf/2605.21605)

**💻 Code:** [⭐ Code](https://github.com/MeiGen-AI/GenEvolve) • [⭐ Code](https://github.com/huggingface)

> Open-ended image generation is no longer a simple prompt-to-image problem. High-quality generation often requires an agent to combine a model’s internal generative ability with external resources. As requests become more diverse and demanding, we ...

</details>

<details>
<summary><b>23. Forecasting Downstream Performance of LLMs With Proxy Metrics</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.18607) • [📄 arXiv](https://arxiv.org/abs/2605.18607) • [📥 PDF](https://arxiv.org/pdf/2605.18607)

**💻 Code:** [⭐ Code](https://github.com/McGill-NLP/proxy-metrics) • [⭐ Code](https://github.com/huggingface)

> Cross-entropy loss is a poor predictor of how models perform on downstream tasks (esp. reasoning). We propose something better: proxy metrics computed over expert reasoning traces.

</details>

<details>
<summary><b>24. KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13734) • [📄 arXiv](https://arxiv.org/abs/2605.13734) • [📥 PDF](https://arxiv.org/pdf/2605.13734)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hpdps-group/KVServe)

> We’re excited to share that KVServe has been accepted by SIGCOMM 2026 🎉 KVServe is a service-aware KV cache compression framework for communication-efficient disaggregated LLM serving. It treats KV compression as a runtime strategy selection probl...

</details>

<details>
<summary><b>25. One Sentence, One Drama: Personalized Short-Form Drama Generation via Multi-Agent Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22144) • [📄 arXiv](https://arxiv.org/abs/2605.22144) • [📥 PDF](https://arxiv.org/pdf/2605.22144)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🎬 One Sentence, One Drama: turn a single sentence into a fully produced short drama! A hierarchical multi-agent framework with three key ingredients: 1️⃣ Multi-agent debate for story generation — enforces short-drama pacing & narrative coherence (...

</details>

<details>
<summary><b>26. ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20176) • [📄 arXiv](https://arxiv.org/abs/2605.20176) • [📥 PDF](https://arxiv.org/pdf/2605.20176)

**💻 Code:** [⭐ Code](https://github.com/UCSC-VLAA/ClinSeekAgent) • [⭐ Code](https://github.com/huggingface)

> Patient evidence, found — not given. Real clinical workflows require an agent to actively seek evidence across raw EHRs, medical imaging, and external knowledge — not just reason over a pre-selected context. We introduce ClinSeekAgent, an automate...

</details>

<details>
<summary><b>27. Efficient Agentic Reasoning Through Self-Regulated Simulative Planning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22138) • [📄 arXiv](https://arxiv.org/abs/2507.23773) • [📥 PDF](https://arxiv.org/pdf/2605.22138)

**💻 Code:** [⭐ Code](https://github.com/sailing-lab/sr2am-self-regulated-planning) • [⭐ Code](https://github.com/sailing-lab/sr2am) • [⭐ Code](https://github.com/huggingface)

> Efficient reasoning is not about shorter chain-of-thought, but about better allocation of simulation (i.e., knowing when to imagine possible futures and when to act directly). Current adaptive-reasoning approaches (effort knobs, token budgets in O...

</details>

<details>
<summary><b>28. SceneAligner: 3D-Grounded Floorplan Localization in the Wild</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22581) • [📄 arXiv](https://arxiv.org/abs/2605.22581) • [📥 PDF](https://arxiv.org/pdf/2605.22581)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Cornell-VAILab/SceneAligner)

> Project Page: https://Cornell-VAILab.github.io/SceneAligner

</details>

<details>
<summary><b>29. Swift Sampling: Selecting Temporal Surprises via Taylor Series</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22678) • [📄 arXiv](https://arxiv.org/abs/2605.22678) • [📥 PDF](https://arxiv.org/pdf/2605.22678)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> While most frames in long-form video are redundant, the critical information resides in temporal surprises: moments where the actual visual features deviate from their predicted evolution. Inspired by the human brain’s predictive coding, we introd...

</details>

<details>
<summary><b>30. LoREnc: Low-Rank Encryption for Securing Foundation Models and LoRA Adapters</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13163) • [📄 arXiv](https://arxiv.org/abs/2605.13163) • [📥 PDF](https://arxiv.org/pdf/2605.13163)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> LoREnc protects foundation model weights from extraction by injecting noise into the FM that cancels out in the LoRA adapter — the fine-tuned output stays intact while FM weights remain obfuscated. Accepted at ICIP 2026.

</details>

<details>
<summary><b>31. Segment Anything with Motion, Geometry, and Semantic Adaptation for Complex Nonlinear Visual Object Tracking</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22538) • [📄 arXiv](https://arxiv.org/abs/2605.22538) • [📥 PDF](https://arxiv.org/pdf/2605.22538)

**💻 Code:** [⭐ Code](https://github.com/DurYi/SAMOSA) • [⭐ Code](https://github.com/huggingface)

> The project page is https://github.com/DurYi/SAMOSA

</details>

<details>
<summary><b>32. Bernini: Latent Semantic Planning for Video Diffusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22344) • [📄 arXiv](https://arxiv.org/abs/2605.22344) • [📥 PDF](https://arxiv.org/pdf/2605.22344)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Bernini is a unified framework for video generation and editing that uses an MLLM-based planner to predict latent semantic representations for a DiT-based renderer.

</details>

<details>
<summary><b>33. TerminalWorld: Benchmarking Agents on Real-World Terminal Tasks</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22535) • [📄 arXiv](https://arxiv.org/abs/2605.22535) • [📥 PDF](https://arxiv.org/pdf/2605.22535)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/EuniAI/TerminalWorld)

> TerminalWorld is a scalable data engine that reverse-engineers real-world terminal recordings into a benchmark of 1,530 validated tasks to evaluate agent performance on authentic software engineering terminal workflows.

</details>

<details>
<summary><b>34. Diversed Model Discovery via Structured Table Discovery</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22766) • [📄 arXiv](https://arxiv.org/abs/2605.22766) • [📥 PDF](https://arxiv.org/pdf/2605.22766)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RJMillerLab/ModelSearch)

> Diversed Model Discovery via Structured Table Discovery

</details>

<details>
<summary><b>35. Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15669) • [📄 arXiv](https://arxiv.org/abs/2605.15669) • [📥 PDF](https://arxiv.org/pdf/2605.15669)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/snu-mllab/Rule2DRC)

> ICML 2026

</details>

<details>
<summary><b>36. More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Paolo Rosso, VictorYeste

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22641) • [📄 arXiv](https://arxiv.org/abs/2605.22641) • [📥 PDF](https://arxiv.org/pdf/2605.22641)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VictorMYeste/human-value-detection-context-rag)

> Detecting Schwartz values in political text is difficult because implicit cues often depend on surrounding arguments and fine-grained distinctions between neighboring values. We study when context and explicit moral knowledge help sentence-level v...

</details>

<details>
<summary><b>37. "I didn't Make the Micro Decisions": Measuring, Inducing, and Exposing Goal-Level AI Contributions in Collaboration</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Sherry Tongshuang Wu, Kyungjin Kim, Jessica R. Mindel, EunsuKim

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21363) • [📄 arXiv](https://arxiv.org/abs/2605.21363) • [📥 PDF](https://arxiv.org/pdf/2605.21363)

**💻 Code:** [⭐ Code](https://github.com/rladmstn1714/CoTrace) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>38. OmniPro: A Comprehensive Benchmark for Omni-Proactive Streaming Video Understanding</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Fengyun Rao, Tianyi Wang, Jie Yang, Ruixiang Zhao, xxayt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.18577) • [📄 arXiv](https://arxiv.org/abs/2605.18577) • [📥 PDF](https://arxiv.org/pdf/2605.18577)

**💻 Code:** [⭐ Code](https://github.com/RuixiangZhao/OmniPro) • [⭐ Code](https://github.com/huggingface)

> We present OmniPro, the first benchmark to jointly evaluate omni-modal perception, proactive responding, and diverse video understanding tasks. It comprises 2,700 human-verified samples spanning 9 sub-tasks and 3 cognitive levels, covering 6 basic...

</details>

<details>
<summary><b>39. Live Music Diffusion Models: Efficient Fine-Tuning and Post-Training of Interactive Diffusion Music Generators</b> ⭐ 22</summary>

<br/>

**👥 Authors:** Nithya Shikarpur, Hugo Flores García, Haven Kim, Stephen Brade, Zachary Novack

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22717) • [📄 arXiv](https://arxiv.org/abs/2605.22717) • [📥 PDF](https://arxiv.org/pdf/2605.22717)

**💻 Code:** [⭐ Code](https://github.com/ZacharyNovack/live-music-diffusion-models) • [⭐ Code](https://github.com/huggingface)

> Demo: https://stephenbrade.github.io/lmdm-public/ Code: https://github.com/ZacharyNovack/live-music-diffusion-models

</details>

<details>
<summary><b>40. AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22715) • [📄 arXiv](https://arxiv.org/abs/2605.22715) • [📥 PDF](https://arxiv.org/pdf/2605.22715)

**💻 Code:** [⭐ Code](https://github.com/Breezelled/AnyMo) • [⭐ Code](https://github.com/huggingface)

> AnyMo is a geometry-aware, setup-agnostic framework for wearable motion understanding. It learns transferable IMU motion representations across sensing setups and datasets, connects sparse wearable signals to open-vocabulary recognition, cross-mod...

</details>

<details>
<summary><b>41. From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shenzhi Wang, Yang Yue, Wenze Lin, Zihan Tang, Xitai Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22074) • [📄 arXiv](https://arxiv.org/abs/2605.22074) • [📥 PDF](https://arxiv.org/pdf/2605.22074)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Introduces SCRL, a curriculum reinforcement learning framework for LLM reasoning that derives verifiable subproblems from reasoning chains to enable fine-grained, subproblem-level credit assignment without external reward models.

</details>

<details>
<summary><b>42. DecQ: Detail-Condensing Queries for Enhanced Reconstruction and Generation in Representation Autoencoders</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Min Li, Zuxuan Wu, Songweii, Row11n, tianhang-wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22777) • [📄 arXiv](https://arxiv.org/abs/2605.22777) • [📥 PDF](https://arxiv.org/pdf/2605.22777)

**💻 Code:** [⭐ Code](https://github.com/Tianhang-Wang/DecQ) • [⭐ Code](https://github.com/huggingface)

> DecQ introduces lightweight detail-condensing queries that extract fine-grained information from intermediate VFM features through condenser modules. These queries are incorporated into the decoder to support reconstruction and are jointly generat...

</details>

<details>
<summary><b>43. Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21803) • [📄 arXiv](https://arxiv.org/abs/2605.21803) • [📥 PDF](https://arxiv.org/pdf/2605.21803)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Same Architecture, Different Optimizer, Different Capacity: In this work, we have shown that realized representation capacity is not architecture-only, it emerges from the architecture-optimizer interaction. Optimizer geometry changes the scaling ...

</details>

<details>
<summary><b>44. Platonic Representations in the Human Brain: Unsupervised Recovery of Universal Geometry</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20496) • [📄 arXiv](https://arxiv.org/abs/2605.20496) • [📥 PDF](https://arxiv.org/pdf/2605.20496)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/memory-formation/platonic-representations-fmri)

> Can the Strong Platonic Representation Hypothesis extend beyond artificial models to human brains? In this paper, we ask whether independently learned fMRI representations from the visual cortex of different subjects can be translated using geomet...

</details>

<details>
<summary><b>45. Lean Refactor: Multi-Objective Controllable Proof Optimization via Agentic Strategy Search</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kaiyu Yang, Rodrigo Stehling, Soonho Kong, Jialin Lu, wuyangchen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20244) • [📄 arXiv](https://arxiv.org/abs/2605.20244) • [📥 PDF](https://arxiv.org/pdf/2605.20244)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present Lean Refactor, a plug-and-play retrieval-augmented agentic framework for multi-objective, controllable, and version-robust refactoring of Lean proofs. LLM-generated proofs are notoriously correct-but-verbose and brittle across library v...

</details>

<details>
<summary><b>46. SAM 3D Animal: Promptable Animal 3D Reconstruction from Images in the Wild</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Silvia Zuffi, Yebin Liu, Jiuming Liu, Jin Lyu, Xuyi Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.07604) • [📄 arXiv](https://arxiv.org/abs/2605.07604) • [📥 PDF](https://arxiv.org/pdf/2605.07604)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Under Review

</details>

<details>
<summary><b>47. Disentangling Sampling from Training Budget in Class-Imbalanced CT Body Composition Segmentation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ivana Išgum, wdika, iasonsky

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.20405) • [📄 arXiv](https://arxiv.org/abs/2605.20405) • [📥 PDF](https://arxiv.org/pdf/2605.20405)

**💻 Code:** [⭐ Code](https://github.com/iasonsky/episodic-sampling) • [⭐ Code](https://github.com/huggingface)

> Class imbalance is a fundamental challenge in medical image segmentation, where frequent classes typically dominate training at the expense of rare classes. Loss-based approaches mitigate imbalance by reweighting the per-pixel loss within the batc...

</details>

<details>
<summary><b>48. FashionLens: Toward Versatile Fashion Image Retrieval via Task-Adaptive Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiangyu Zhao, Xiaolin Chen, Xinghao Xie, Xuemeng Song, HaokunWen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22552) • [📄 arXiv](https://arxiv.org/abs/2605.22552) • [📥 PDF](https://arxiv.org/pdf/2605.22552)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> FashionLens — a unified MLLM framework for versatile fashion image retrieval.

</details>

<details>
<summary><b>49. Minimalist Visual Inertial Odometry</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.19990) • [📄 arXiv](https://arxiv.org/abs/2605.19990) • [📥 PDF](https://arxiv.org/pdf/2605.19990)

**💻 Code:** [⭐ Code](https://github.com/pastifra/four-pixel-vio) • [⭐ Code](https://github.com/huggingface)

> Robot Odometry with a 4-Pixel sensor and an IMU.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 49 |
| 📅 Today | [`2026-05-24.json`](data/daily/2026-05-24.json) | 49 |
| 📆 This Week | [`2026-W20.json`](data/weekly/2026-W20.json) | 183 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 573 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-24 | 49 | [View JSON](data/daily/2026-05-24.json) |
| 📄 2026-05-23 | 49 | [View JSON](data/daily/2026-05-23.json) |
| 📄 2026-05-22 | 18 | [View JSON](data/daily/2026-05-22.json) |
| 📄 2026-05-21 | 18 | [View JSON](data/daily/2026-05-21.json) |
| 📄 2026-05-20 | 16 | [View JSON](data/daily/2026-05-20.json) |
| 📄 2026-05-19 | 15 | [View JSON](data/daily/2026-05-19.json) |
| 📄 2026-05-18 | 18 | [View JSON](data/daily/2026-05-18.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 573 | [View JSON](data/monthly/2026-05.json) |
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
