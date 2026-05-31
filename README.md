<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-59-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4403+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">59</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">209</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">782</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4403+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 31, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29801) • [📄 arXiv](https://arxiv.org/abs/2605.29801) • [📥 PDF](https://arxiv.org/pdf/2605.29801)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security

</details>

<details>
<summary><b>2. Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30280) • [📄 arXiv](https://arxiv.org/abs/2605.30280) • [📥 PDF](https://arxiv.org/pdf/2605.30280)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Qwen-VLA is a unified embodied foundation model that extends Qwen's vision-language stack to support continuous action and trajectory generation across diverse robot platforms, tasks, and environments.

</details>

<details>
<summary><b>3. OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29250) • [📄 arXiv](https://arxiv.org/abs/2605.29250) • [📥 PDF](https://arxiv.org/pdf/2605.29250)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/JinheonBaek/OmniRetrieval)

> Real-world answers live in text, tables, and graphs. And, OmniRetrieval reaches them all through one natural-language interface, meeting each source on its own terms.

</details>

<details>
<summary><b>4. CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher On-Policy Distillation</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Shijie Huang, Hailong Guo, Fangtai Wu, jamesliu1217, sjy92

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25378) • [📄 arXiv](https://arxiv.org/abs/2605.25378) • [📥 PDF](https://arxiv.org/pdf/2605.25378)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Qwen-Applications/CollectionLoRA)

> This paper can be understood from two perspectives: A paradigm for multi-teacher distillation on image editing. A ready-to-use recipe for slashing LoRA serving costs. TL;DR: 50 effects, 1 LoRA, zero interference.

</details>

<details>
<summary><b>5. minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models</b> ⭐ 382</summary>

<br/>

**👥 Authors:** Zihan Zhou, Bokai Yan, Min Zhao, wenqsun, zhuhz22

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30263) • [📄 arXiv](https://arxiv.org/abs/2605.30263) • [📥 PDF](https://arxiv.org/pdf/2605.30263)

**💻 Code:** [⭐ Code](https://github.com/shengshu-ai/minWM) • [⭐ Code](https://github.com/huggingface)

> Github: https://github.com/shengshu-ai/minWM

</details>

<details>
<summary><b>6. YoCausal: How Far is Video Generation from World Model? A Causality Perspective</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30346) • [📄 arXiv](https://arxiv.org/abs/2605.30346) • [📥 PDF](https://arxiv.org/pdf/2605.30346)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/youzhe0305/YoCausal)

> As video diffusion models (VDMs) advance toward world models, a key question arises: do they truly understand causality, or merely overfit to statistical temporal patterns? Existing benchmarks mostly rely on synthetic data, limiting real-world gen...

</details>

<details>
<summary><b>7. Why Far Looks Up: Probing Spatial Representation in Vision-Language Models</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30161) • [📄 arXiv](https://arxiv.org/abs/2605.30161) • [📥 PDF](https://arxiv.org/pdf/2605.30161)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cheolhong0916/contrastive-probing)

> Shows that representation structure, not benchmark accuracy alone, is a strong predictor of robust spatial reasoning in vision-language models.

</details>

<details>
<summary><b>8. GenClaw: Code-Driven Agentic Image Generation</b> ⭐ 78</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30248) • [📄 arXiv](https://arxiv.org/abs/2605.30248) • [📥 PDF](https://arxiv.org/pdf/2605.30248)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yejy53/GenClaw)

> Image generation models have evolved from text-conditioned pixel synthesis toward multimodal agents endowed with visual comprehension and tool invocation capabilities. Yet, existing agents remain at the mercy of underlying black-box image models. ...

</details>

<details>
<summary><b>9. How LoRA Remembers? A Parametric Memory Law for LLM Finetuning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30260) • [📄 arXiv](https://arxiv.org/abs/2605.30260) • [📥 PDF](https://arxiv.org/pdf/2605.30260)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zjunlp/ParametricMemoryLaw)

> We uncover a quantitative law of parametric memory in LLMs, showing that exact recall emerges through a sharp probability threshold and can be significantly improved with threshold-guided fine-tuning.

</details>

<details>
<summary><b>10. EarlyTom: Early Token Compression Completes Fast Video Understanding</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Jian Chen, Chenhaowen Li, Lu Lu, Xin Jin, Hesong Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30010) • [📄 arXiv](https://arxiv.org/abs/2605.30010) • [📥 PDF](https://arxiv.org/pdf/2605.30010)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/viridisGreen/EarlyTom)

> Video large language models (Video-LLMs) have demonstrated strong capabilities in video understanding tasks. However, their practical deployment is still hindered by the inefficiency introduced by processing massive amounts of visual tokens. Altho...

</details>

<details>
<summary><b>11. Native Audio-Visual Alignment for Generation</b> ⭐ 58</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30073) • [📄 arXiv](https://arxiv.org/abs/2605.30073) • [📥 PDF](https://arxiv.org/pdf/2605.30073)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ernie-research/NAVA)

> NAVA is a 6.3B-parameter Native Audio-Visual Alignment framework for joint audio-video generation. To overcome the limitations of existing dual-tower and unified paradigms, NAVA employs an Align-then-Fuse MMDiT architecture that first establishes ...

</details>

<details>
<summary><b>12. LaRA: Layer-wise Representation Analysis for Detecting Data Contamination in RL Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29888) • [📄 arXiv](https://arxiv.org/abs/2605.29888) • [📥 PDF](https://arxiv.org/pdf/2605.29888)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper introduces LaRA (Layer-wise Representation Analysis), a framework for detecting data contamination in RL post-trained LLMs by examining how internal representations change across layers rather than relying on output-level signals such a...

</details>

<details>
<summary><b>13. UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30076) • [📄 arXiv](https://arxiv.org/abs/2605.30076) • [📥 PDF](https://arxiv.org/pdf/2605.30076)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper presents UniSteer, a text-guided flow matching framework for steering LLM activations. The goal is to provide a unified interface where different steering targets can be specified through natural-language conditions. We study flow inver...

</details>

<details>
<summary><b>14. LoMo: Local Modality Substitution for Deeper Vision-Language Fusion</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30265) • [📄 arXiv](https://arxiv.org/abs/2605.30265) • [📥 PDF](https://arxiv.org/pdf/2605.30265)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Maplebb/LoMo)

> LoMo is a lightweight, architecture-agnostic data curation paradigm that mitigates the cross-carrier modality gap in VLMs by recasting selected text spans into rendered images, turning standard SFT into an implicit cross-modal alignment objective ...

</details>

<details>
<summary><b>15. Skill0.5: Joint Skill Internalization and Utilization for Out-of-Distribution Generalization in Agentic Reinforcement Learning</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.28424) • [📄 arXiv](https://arxiv.org/abs/2605.28424) • [📥 PDF](https://arxiv.org/pdf/2605.28424)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/JasonZhujp/Skill0_5)

> Existing skill-based RL methods force a rigid choice between full externalization (prohibitive context overhead) and full internalization (overfitting and knowledge conflicts). Skill0.5 resolves this by jointly combining general skill internalizat...

</details>

<details>
<summary><b>16. Colored Noise Diffusion Sampling</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30332) • [📄 arXiv](https://arxiv.org/abs/2605.30332) • [📥 PDF](https://arxiv.org/pdf/2605.30332)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hadardavidson/colored-noise-sampling)

> Standard diffusion SDEs waste energy by injecting uniform white noise, ignoring the fact that models resolve low-frequency structures before high-frequency details. We introduce Colored Noise Sampling (CNS), plug-and-play solver that dynamically t...

</details>

<details>
<summary><b>17. When Should Models Change Their Minds? Contextual Belief Management in Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30219) • [📄 arXiv](https://arxiv.org/abs/2605.30219) • [📥 PDF](https://arxiv.org/pdf/2605.30219)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We show that long-horizon reasoning in LLMs fundamentally depends on contextual belief management — knowing when to update, preserve, or ignore information — and that explicit belief-state optimization dramatically improves this ability.

</details>

<details>
<summary><b>18. Xetrieval: Mechanistically Explaining Dense Retrieval</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29507) • [📄 arXiv](https://arxiv.org/abs/2605.29507) • [📥 PDF](https://arxiv.org/pdf/2605.29507)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Hihiczx/Xetrieval)

> Xetrieval: Mechanistically Explaining Dense Retrieval

</details>

<details>
<summary><b>19. Is Position Bias in Dense Retrievers Built In-or Learned from Data?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26578) • [📄 arXiv](https://arxiv.org/abs/2605.26578) • [📥 PDF](https://arxiv.org/pdf/2605.26578)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API MeVer at CheckThat! 2026: Cluster-Aware Hard-Negative Mining for Multilingu...

</details>

<details>
<summary><b>20. LiteCoder-Terminal: Scaling Long-Horizon Terminal Environments for Learning Language Agents</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Yaojie Lu, Boxi Cao, Xinyu Lu, Kaiqi Zhang, paraline

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29559) • [📄 arXiv](https://arxiv.org/abs/2605.29559) • [📥 PDF](https://arxiv.org/pdf/2605.29559)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/icip-cas/LiteCoder)

> Add paper.

</details>

<details>
<summary><b>21. AsyncTool: Evaluating the Asynchronous Function Calling Capability under Multi-Task Scenarios</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27995) • [📄 arXiv](https://arxiv.org/abs/2605.27995) • [📥 PDF](https://arxiv.org/pdf/2605.27995)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/StoKou/repo-asynctool)

> Large language model (LLM)-based agents have shown strong capabilities in using external tools to solve complex tasks. However, existing evaluations often overlook the temporal dimension of tool use, especially the impact of tool response latency,...

</details>

<details>
<summary><b>22. CausaLab: A Scalable Environment for Interactive Causal Discovery Toward AI Scientists</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26029) • [📄 arXiv](https://arxiv.org/abs/2605.26029) • [📥 PDF](https://arxiv.org/pdf/2605.26029)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> CausaLab turns “AI scientist” evaluation from a static quiz into a live laboratory: an LLM agent must study noisy measurement records, choose interventions on a controllable crystal, infer the hidden structural causal model, and transfer that mech...

</details>

<details>
<summary><b>23. PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26730) • [📄 arXiv](https://arxiv.org/abs/2605.26730) • [📥 PDF](https://arxiv.org/pdf/2605.26730)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> PRISM is a benchmarking framework that evaluates LLM peer reviewers against human experts across four scientifically grounded dimensions—depth of analysis, novelty assessment, flaw identification, and constructiveness. Applied to five leading auto...

</details>

<details>
<summary><b>24. When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30102) • [📄 arXiv](https://arxiv.org/abs/2605.30102) • [📥 PDF](https://arxiv.org/pdf/2605.30102)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> If you use an edge device-sized, or self hosted, LM to power your agentic system, you will usually observe subpar performance; on the other hand, while cloud-based frontier models can deliver satisfactory performance, they also come with potential...

</details>

<details>
<summary><b>25. PhyGenHOI: Physically-Aware 4D Generation of Dynamic Human-Object Interactions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30268) • [📄 arXiv](https://arxiv.org/abs/2605.30268) • [📥 PDF](https://arxiv.org/pdf/2605.30268)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> PhyGenHOI tackles the lack of physical realism in 4D generation. We model humans via MDM and objects as physical agents via MPM simulations, using 3DGS as a unified representation. With contact-driven re-simulation and Masked Video-SDS, we heavily...

</details>

<details>
<summary><b>26. Towards Verifiable Multimodal Deep Research: A Multi-Agent Harness for Interleaved Report Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29861) • [📄 arXiv](https://arxiv.org/abs/2605.29861) • [📥 PDF](https://arxiv.org/pdf/2605.29861)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Interleaved image-text reports are an important format for presenting complex multimodal information, yet generating them in a trustworthy and well-grounded way remains challenging. In this work, we introduce Ptah , an agentic harness for producin...

</details>

<details>
<summary><b>27. UI-KOBE: Knowledge-Oriented Behavior Exploration for Lightweight Graph-Guided GUI Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rui Liu, Jinpeng Chen, Xinyu Fu, Yuxiang Chai, HanXiao1999

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29534) • [📄 arXiv](https://arxiv.org/abs/2605.29534) • [📥 PDF](https://arxiv.org/pdf/2605.29534)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/YuxiangChai/UI-KOBE)

> Code at: https://github.com/YuxiangChai/UI-KOBE

</details>

<details>
<summary><b>28. DynaFLIP: Rethinking Robotics Perception via Tri-Modal-Dynamics Guided Representation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sungha Kim, Hoseong Jung, Jonghun Shin, Seungjae Lee, Jusuk Lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30350) • [📄 arXiv](https://arxiv.org/abs/2605.30350) • [📥 PDF](https://arxiv.org/pdf/2605.30350)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>29. RUBRIC-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29156) • [📄 arXiv](https://arxiv.org/abs/2605.29156) • [📥 PDF](https://arxiv.org/pdf/2605.29156)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Pointwise reward modeling offers critical signals for LLM post-training, yet struggles with absolute scoring in subjective, non-verifiable settings. Rubric-based methods address this by decomposing evaluation into explicit criteria, but existing a...

</details>

<details>
<summary><b>30. REPOT: Recoverable Program-of-Thought via Checkpoint Repair</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Parsa Mazaheri

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30052) • [📄 arXiv](https://arxiv.org/abs/2605.30052) • [📥 PDF](https://arxiv.org/pdf/2605.30052)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/parsa-mz/RePot)

> TL;DR — Large reasoning models that write a Python program to solve a multi-step puzzle silently invalidate the entire plan when one primitive step is wrong — even if 29 of 30 steps were right. RePoT treats the program as a checkpoint , not a fina...

</details>

<details>
<summary><b>31. WorldMemArena: Evaluating Multimodal Agent Memory Through Action-World Interaction</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Lin Long, Yepeng Liu, Sophia Xiao Pu, Yuzhe Yang, Chengzhi Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29341) • [📄 arXiv](https://arxiv.org/abs/2605.29341) • [📥 PDF](https://arxiv.org/pdf/2605.29341)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/UCSB-AI/WorldMemArena)

> WorldMemArena is a new benchmark evaluating the multimodal memory of long-horizon agents using a four-stage Action-World Interaction Loop and multi-session tasks for detailed performance diagnostics.

</details>

<details>
<summary><b>32. Verifiable Rewards Beyond Math and Code: Lightweight Corpus-Grounded Process Supervision for Factual Question Answering</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29648) • [📄 arXiv](https://arxiv.org/abs/2605.29648) • [📥 PDF](https://arxiv.org/pdf/2605.29648)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/shichengf/CorVer)

> Applying reinforcement learning to improve factual accuracy in knowledge-intensive question answering faces a reward design dilemma. Response-level rewards provide only coarse supervision and cannot distinguish correct from incorrect statements wi...

</details>

<details>
<summary><b>33. Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Naomi Saphra, Laura Ruis, Rachit Bansal, Daniel Wurgaft, Jing Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29548) • [📄 arXiv](https://arxiv.org/abs/2605.29548) • [📥 PDF](https://arxiv.org/pdf/2605.29548)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper provides a data-centric account of why larger models learn tasks smaller models fail, attributing this to reduced gradient interference and more efficient resource allocation for rare tasks.

</details>

<details>
<summary><b>34. AdaState: Self-Evolving Anchors for Streaming Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30349) • [📄 arXiv](https://arxiv.org/abs/2605.30349) • [📥 PDF](https://arxiv.org/pdf/2605.30349)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Streaming video diffusion models have a structural blind spot: they anchor on the first frame. Because that frame sits in the cleanest, most error-free slot of the KV cache, attention collapses onto this reference; suppressing dynamics and locking...

</details>

<details>
<summary><b>35. Thinking Before Constraining: A Unified Decoding Framework for Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07525) • [📄 arXiv](https://arxiv.org/abs/2601.07525) • [📥 PDF](https://arxiv.org/pdf/2601.07525)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Nokia-Bell-Labs/InWriting)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API The Format Tax (2026) Post Reasoning: Improving the Performance of Non-Thin...

</details>

<details>
<summary><b>36. Learning A Unified Risk Map for Autonomous Driving in Partially Observable Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22189) • [📄 arXiv](https://arxiv.org/abs/2605.22189) • [📥 PDF](https://arxiv.org/pdf/2605.22189)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Learning A Unified Risk Map for Autonomous Driving in Partially Observable Environments  (IEEE RA-L) — How should a self-driving car reason about danger it can't see? This work (IEEE RA-L) introduces a unified spatiotemporal risk map that fuses tr...

</details>

<details>
<summary><b>37. CoHyDE: Iterative Co-Training of LLM Rewriter & Dense Encoder for Tool Retrieval</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29271) • [📄 arXiv](https://arxiv.org/abs/2605.29271) • [📥 PDF](https://arxiv.org/pdf/2605.29271)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> CoHyDE introduces an iterative co-training procedure that jointly optimizes a dense encoder and an LLM rewriter for tool retrieval. By alternating between retraining the encoder on the rewriter’s catalog-aligned hypothetical descriptions and align...

</details>

<details>
<summary><b>38. NeuROK: Generative 4D Neural Object Kinematics</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shangzhe Wu, Yunzhi Zhang, Yue Gao, Guangzhao He, Chen Geng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30347) • [📄 arXiv](https://arxiv.org/abs/2605.30347) • [📥 PDF](https://arxiv.org/pdf/2605.30347)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> NeuROK provides a data-driven kinematic state parameterization for object-centric systems, enabling the efficient generation of simulative 4D object dynamics without explicit physical models or prior inductive bias.

</details>

<details>
<summary><b>39. ChildVox: A Speech, Audio, and Large Audio-Language Model Benchmark in Understanding and Characterizing Sound across Childhood</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29257) • [📄 arXiv](https://arxiv.org/abs/2605.29257) • [📥 PDF](https://arxiv.org/pdf/2605.29257)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present ChildVox, a novel benchmark for characterizing the diverse acoustic signals through which children communicate. Specifically, ChildVox follows the full developmental trajectory from birth through school age, covering physiological sound...

</details>

<details>
<summary><b>40. Parallax: Parameterized Local Linear Attention for Language Modeling</b> ⭐ 34</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29157) • [📄 arXiv](https://arxiv.org/abs/2510.01450) • [📥 PDF](https://arxiv.org/pdf/2605.29157)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Yifei-Zuo/FlashLLA) • [⭐ Code](https://github.com/Yifei-Zuo/Parallax)

> LLA paper: https://arxiv.org/abs/2510.01450 LLA kernel was published in the github: https://github.com/Yifei-Zuo/FlashLLA

</details>

<details>
<summary><b>41. PANDO: Efficient Multimodal AI Agents via Online Skill Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.24785) • [📄 arXiv](https://arxiv.org/abs/2605.24785) • [📥 PDF](https://arxiv.org/pdf/2605.24785)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> V1

</details>

<details>
<summary><b>42. CONF-KV: Confidence-Aware KV Cache Eviction with Mixed-Precision Storage for Long-Horizon LLM</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.24786) • [📄 arXiv](https://arxiv.org/abs/2605.24786) • [📥 PDF](https://arxiv.org/pdf/2605.24786)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> V1

</details>

<details>
<summary><b>43. Token-Level Generalization in LoRA Adapter Backdoors: Attack Characterization and Behavioral Detection</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Travis Lelle

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30189) • [📄 arXiv](https://arxiv.org/abs/2605.30189) • [📥 PDF](https://arxiv.org/pdf/2605.30189)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Travis-ML/lora-backdoors)

> A few things from this paper I'd love to hear other people's takes on: The chosen trigger anchor is family-dependent. Train Qwen 2.5 (1.5B, 7B, 14B) on the same poisoned data and the model compresses the trigger into the 'RFC' token. Train Llama 3...

</details>

<details>
<summary><b>44. SmartDirector: Keyframe-Conditioned Cinematic Video Generation with Narrative Pacing Control</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27891) • [📄 arXiv](https://arxiv.org/abs/2605.27891) • [📥 PDF](https://arxiv.org/pdf/2605.27891)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Orange-3DV-Team/SmartDirector)

> SmartDirector introduces a two-stage framework that empowers cinematic video generation with precise narrative pacing and high-fidelity detail recovery by leveraging a novel Multi-Chunk VAE strategy to circumvent temporal causal constraints.

</details>

<details>
<summary><b>45. Multi-view Consistent 3D Gaussian Head Avatars 'without' Multi-view Generation</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.25220) • [📄 arXiv](https://arxiv.org/abs/2605.25220) • [📥 PDF](https://arxiv.org/pdf/2605.25220)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/humansensinglab/MVCHead)

> Generating 3D Head Avatars from just 70K Random Internet Images! No 3D, no multi-view, no studio, no view synthesis at any stage of training or inference.

</details>

<details>
<summary><b>46. Reflective Prompt Tuning through Language Model Function-Calling</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.21781) • [📄 arXiv](https://arxiv.org/abs/2605.21781) • [📥 PDF](https://arxiv.org/pdf/2605.21781)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/megagonlabs/RPT)

> A diagnosis-driven framework for iterative prompt tuning with function-calling LLMs.

</details>

<details>
<summary><b>47. Geometry Matters: 3D Foundation Priors for Learning Semantic Correspondence</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30093) • [📄 arXiv](https://arxiv.org/abs/2605.30093) • [📥 PDF](https://arxiv.org/pdf/2605.30093)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/GenIntel/3D-SC)

> 🧵 New work: 3D-Aware Semantic Correspondence 2D foundation features (DINO, Stable Diffusion) are powerful for semantic correspondence — but they have a blind spot: they can't tell left from right, or distinguish repeated parts that are clearly sep...

</details>

<details>
<summary><b>48. Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.27355) • [📄 arXiv](https://arxiv.org/abs/2605.27355) • [📥 PDF](https://arxiv.org/pdf/2605.27355)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/alignment-tampering/alignment-tampering)

> We introduce Alignment Tampering , a vulnerability where the LLM undergoing alignment influences the preference dataset itself, causing RLHF to amplify undesired behaviors.

</details>

<details>
<summary><b>49. Uniform Diffusion Models Revisited: Leave-One-Out Denoiser and Absorbing State Reformulation</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Eric Moulines, Umut Simsekli, Dario Shariatian, Yazid Janati, Samson Gourevitch

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22765) • [📄 arXiv](https://arxiv.org/abs/2605.22765) • [📥 PDF](https://arxiv.org/pdf/2605.22765)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/samsongourevitch/rev_udm)

> We show that standard implementations of Uniform Diffusion Models (Multinomial Diffusion, USDM, Duo, GIDD...) do not learn a denoiser but instead a "leave-one-out" denoiser. We use this characterization to improve training and design new inference...

</details>

<details>
<summary><b>50. Beyond 3D VQAs: Injecting 3D Spatial Priors into Vision-Language Models for Enhanced Geometric Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Joseph Tighe, Yi Ma, Manchen Wang, Shengyi Qian, danielchyeh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30231) • [📄 arXiv](https://arxiv.org/abs/2605.30231) • [📥 PDF](https://arxiv.org/pdf/2605.30231)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Unlocking Dense Metric Depth Estimation in VLMs (2026) Let Geometry GUIDE: ...

</details>

<details>
<summary><b>51. Discovering Cooperative Pipelines: Autoresearch for Sequential Social Dilemmas</b> ⭐ 1</summary>

<br/>

**👥 Authors:** vicgalle

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30003) • [📄 arXiv](https://arxiv.org/abs/2605.30003) • [📥 PDF](https://arxiv.org/pdf/2605.30003)

**💻 Code:** [⭐ Code](https://github.com/vicgalle/autoresearch-social-dilemmas) • [⭐ Code](https://github.com/huggingface)

> @ librarian-bot recommend

</details>

<details>
<summary><b>52. OmniInteract: Benchmarking Real-World Streaming Interaction for Real-Time Omnimodal Assistants</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Jinpeng Chen, Yang Bo, Annan Wang, Xueying Li, Xudong Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.26485) • [📄 arXiv](https://arxiv.org/abs/2605.26485) • [📥 PDF](https://arxiv.org/pdf/2605.26485)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Lucky-Lance/OmniInteract)

> OmniInteract is a streaming benchmark for real-time omnimodal LLMs, evaluated through their native online inference over continuous audio-visual streams. User queries and ambient sounds live in the audio track, visual events live in the video, and...

</details>

<details>
<summary><b>53. Convex Low-resource Accent-Robust Language Detection in Speech Recognition</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.23235) • [📄 arXiv](https://arxiv.org/abs/2605.23235) • [📥 PDF](https://arxiv.org/pdf/2605.23235)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/pilancilab/CLD)

> 🎵 Meet Convex Language Detection (CLD)! Automatic Speech Recognition (ASR) frequently exhibits failures on accents and dialects. But collecting more data to retrain a larger model is slow and expensive. CLD solves this—not by grid-searching hyperp...

</details>

<details>
<summary><b>54. MoZoo:Unleashing Video Diffusion power in animal fur and muscle simulation</b> ⭐ 103</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13857) • [📄 arXiv](https://arxiv.org/abs/2605.13857) • [📥 PDF](https://arxiv.org/pdf/2605.13857)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Orange-3DV-Team/MoZoo)

> MoZoo is a pioneering generative dynamics solver that revolutionizes cinematic animal production by directly synthesizing high-fidelity fur and muscle simulations from coarse meshes, effectively bypassing the labor-intensive refinement stages of t...

</details>

<details>
<summary><b>55. Tiny but Trusted: Efficient Vision-Language Reasoning for Time-Series Anomaly Detection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30344) • [📄 arXiv](https://arxiv.org/abs/2605.30344) • [📥 PDF](https://arxiv.org/pdf/2605.30344)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce Tiny but Trusted, a parameter-efficient vision-language framework for time-series anomaly detection with grounded reasoning, enabling compact VLMs to provide accurate and explainable reasoning over sequential data. Instead of treating...

</details>

<details>
<summary><b>56. Reducing Political Manipulation with Consistency Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22771) • [📄 arXiv](https://arxiv.org/abs/2605.22771) • [📥 PDF](https://arxiv.org/pdf/2605.22771)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/centerforaisafety/political-manipulation)

> X: https://x.com/longphan3110/status/2058933197535170787

</details>

<details>
<summary><b>57. Towards Consistent Video Geometry Estimation</b> ⭐ 49</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.30060) • [📄 arXiv](https://arxiv.org/abs/2605.30060) • [📥 PDF](https://arxiv.org/pdf/2605.30060)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/aigc3d/ViGeo)

> This work presents ViGeo, a feed-forward foundation model for recovering spatially dense and temporally consistent geometry from video sequences. Built upon a plain transformer architecture without task-specific architectural modifications, ViGeo ...

</details>

<details>
<summary><b>58. PhoneWorld: Scaling Phone-Use Agent Environments</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pengyuan Lyu, Junyi Li, Xin Lai, Yuxuan Liu, tangzhy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.29486) • [📄 arXiv](https://arxiv.org/abs/2605.29486) • [📥 PDF](https://arxiv.org/pdf/2605.29486)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> check out our new paper for phone-use/mobile agents gym!

</details>

<details>
<summary><b>59. ORACLE: Anticipating Scams from Partial Trajectories in Streaming App Usage</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Gang Xu, Fei Shen, Zhongan Wang, Songbai Tan, snowleo135

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.16363) • [📄 arXiv](https://arxiv.org/abs/2605.16363) • [📥 PDF](https://arxiv.org/pdf/2605.16363)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> An online reasoning framework anticipates smartphone scams by analyzing streaming app-usage trajectories through self-evolving context management and on-policy self-distillation techniques.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 59 |
| 📅 Today | [`2026-05-31.json`](data/daily/2026-05-31.json) | 59 |
| 📆 This Week | [`2026-W21.json`](data/weekly/2026-W21.json) | 209 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 782 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-31 | 59 | [View JSON](data/daily/2026-05-31.json) |
| 📄 2026-05-30 | 59 | [View JSON](data/daily/2026-05-30.json) |
| 📄 2026-05-29 | 24 | [View JSON](data/daily/2026-05-29.json) |
| 📄 2026-05-28 | 19 | [View JSON](data/daily/2026-05-28.json) |
| 📄 2026-05-27 | 17 | [View JSON](data/daily/2026-05-27.json) |
| 📄 2026-05-26 | 15 | [View JSON](data/daily/2026-05-26.json) |
| 📄 2026-05-25 | 16 | [View JSON](data/daily/2026-05-25.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W21 | 209 | [View JSON](data/weekly/2026-W21.json) |
| 📅 2026-W20 | 183 | [View JSON](data/weekly/2026-W20.json) |
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
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
