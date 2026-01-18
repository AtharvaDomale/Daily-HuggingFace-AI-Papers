<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-38-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1167+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">38</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">232</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">429</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1167+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 18, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. STEP3-VL-10B Technical Report</b> ⭐ 168</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09668) • [📄 arXiv](https://arxiv.org/abs/2601.09668) • [📥 PDF](https://arxiv.org/pdf/2601.09668)

**💻 Code:** [⭐ Code](https://github.com/stepfun-ai/PaCoRe) • [⭐ Code](https://github.com/stepfun-ai/Step3-VL-10B)

> 🎉 Introducing Step3-VL-10B, Compact Yet Frontier Multimodal Intelligence, with best performance at 10B model scale, even matching 10x-20x size of open-source frontier models!

</details>

<details>
<summary><b>2. Urban Socio-Semantic Segmentation with Vision-Language Reasoning</b> ⭐ 135</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10477) • [📄 arXiv](https://arxiv.org/abs/2601.10477) • [📥 PDF](https://arxiv.org/pdf/2601.10477)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/SocioReasoner)

> It is a very interesting idea of practical value.  Applying VLM  + RL on (real world) Socio-Semantic Segmentation task!

</details>

<details>
<summary><b>3. Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08763) • [📄 arXiv](https://arxiv.org/abs/2601.08763) • [📥 PDF](https://arxiv.org/pdf/2601.08763)

> Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs

</details>

<details>
<summary><b>4. Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09667) • [📄 arXiv](https://arxiv.org/abs/2601.09667) • [📥 PDF](https://arxiv.org/pdf/2601.09667)

> Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning

</details>

<details>
<summary><b>5. VIBE: Visual Instruction Based Editor</b> ⭐ 29</summary>

<br/>

**👥 Authors:** Bulat Suleimanov, WildChlamydia, iitolstykh, grac20101, Riko0

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02242) • [📄 arXiv](https://arxiv.org/abs/2601.02242) • [📥 PDF](https://arxiv.org/pdf/2601.02242)

**💻 Code:** [⭐ Code](https://github.com/ai-forever/vibe)

> 🎉 Introducing VIBE: Visual Instruction Based Editor, a compact and powerful system that fits comfortably within 24 GB of GPU memory and generates edited images at up to 2K resolution in approximately 4 seconds on an NVIDIA H100, without any extra ...

</details>

<details>
<summary><b>6. Beyond Static Tools: Test-Time Tool Evolution for Scientific Reasoning</b> ⭐ 37</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.07641) • [📄 arXiv](https://arxiv.org/abs/2601.07641) • [📥 PDF](https://arxiv.org/pdf/2601.07641)

**💻 Code:** [⭐ Code](https://github.com/lujiaxuan0520/Test-Time-Tool-Evol)

> 🎉 Introducing Test-Time Tool Evolution (TTE) — Beyond Static Tools: Test-Time Tool Evolution for Scientific Reasoning (arXiv:2601.07641)! Why it matters: Scientific problems don’t come with a complete tool library. TTE lets agents synthesize → val...

</details>

<details>
<summary><b>7. DanQing: An Up-to-Date Large-Scale Chinese Vision-Language Pre-training Dataset</b> ⭐ 14</summary>

<br/>

**👥 Authors:** fengziyong, SeriousBro, dfgdgh, TianchengGu, dewecho

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10305) • [📄 arXiv](https://arxiv.org/abs/2601.10305) • [📥 PDF](https://arxiv.org/pdf/2601.10305)

**💻 Code:** [⭐ Code](https://github.com/deepglint/DanQing)

> Vision-Language Pre-training (VLP) models demonstrate strong performance across various downstream tasks by learning from large-scale image-text pairs through contrastive pretraining. The release of extensive English image-text datasets (e.g., COY...

</details>

<details>
<summary><b>8. Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering</b> ⭐ 336</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10402) • [📄 arXiv](https://arxiv.org/abs/2601.10402) • [📥 PDF](https://arxiv.org/pdf/2601.10402)

**💻 Code:** [⭐ Code](https://github.com/sjtu-sai-agents/ML-Master)

> Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering

</details>

<details>
<summary><b>9. CoF-T2I: Video Models as Pure Visual Reasoners for Text-to-Image Generation</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10061) • [📄 arXiv](https://arxiv.org/abs/2601.10061) • [📥 PDF](https://arxiv.org/pdf/2601.10061)

**💻 Code:** [⭐ Code](https://github.com/VisionChengzhuo/CoF-T2I)

> paper link: https://arxiv.org/pdf/2601.10061

</details>

<details>
<summary><b>10. Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders</b> ⭐ 86</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10332) • [📄 arXiv](https://arxiv.org/abs/2601.10332) • [📥 PDF](https://arxiv.org/pdf/2601.10332)

**💻 Code:** [⭐ Code](https://github.com/zhijie-group/Think-Then-Generate)

> Github: https://github.com/zhijie-group/Think-Then-Generate

</details>

<details>
<summary><b>11. Alterbute: Editing Intrinsic Attributes of Objects in Images</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10714) • [📄 arXiv](https://arxiv.org/abs/2601.10714) • [📥 PDF](https://arxiv.org/pdf/2601.10714)

> [TL;DR] We present Alterbute, a diffusion-based method for editing an object’s intrinsic attributes — color, texture, material, and shape — while preserving its perceived identity and scene context. Paper 📄: https://arxiv.org/pdf/2601.10714 Projec...

</details>

<details>
<summary><b>12. MatchTIR: Fine-Grained Supervision for Tool-Integrated Reasoning via Bipartite Matching</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10712) • [📄 arXiv](https://arxiv.org/abs/2601.10712) • [📥 PDF](https://arxiv.org/pdf/2601.10712)

**💻 Code:** [⭐ Code](https://github.com/quchangle1/MatchTIR)

> 💡 Overview We propose MatchTIR, a fine-grained reinforcement learning framework specifically designed for Tool-Integrated Reasoning (TIR). The core principle of MatchTIR is to introduce precise supervision via bipartite matching-based turn-level r...

</details>

<details>
<summary><b>13. ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shikun Zhang, Peiyang Liu, Lijun Li, Zhangchi Xue, MurrayTom

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10156) • [📄 arXiv](https://arxiv.org/abs/2601.10156) • [📥 PDF](https://arxiv.org/pdf/2601.10156)

**💻 Code:** [⭐ Code](https://github.com/MurrayTom/ToolSafe)

> GitHub: https://github.com/MurrayTom/ToolSafe

</details>

<details>
<summary><b>14. Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** gstoica3, praeclarumjj3, tairaa, kimdon20, zhongzhengrenzhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10611) • [📄 arXiv](https://arxiv.org/abs/2601.10611) • [📥 PDF](https://arxiv.org/pdf/2601.10611)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VideoWeave: A Data-Centric Approach for Efficient Video Understanding (2026...

</details>

<details>
<summary><b>15. A Safety Report on GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5</b> ⭐ 21</summary>

<br/>

**👥 Authors:** Yutao Wu, Yixu Wang, Xingjun Ma, xinwang22, DobyXu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10527) • [📄 arXiv](https://arxiv.org/abs/2601.10527) • [📥 PDF](https://arxiv.org/pdf/2601.10527)

**💻 Code:** [⭐ Code](https://github.com/XSafeAI/AI-safety-report)

> The rapid evolution of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has produced substantial gains in reasoning, perception, and generative capability across language and vision. However, whether these advances yield c...

</details>

<details>
<summary><b>16. Transition Matching Distillation for Fast Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09881) • [📄 arXiv](https://arxiv.org/abs/2601.09881) • [📥 PDF](https://arxiv.org/pdf/2601.09881)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VDOT: Efficient Unified Video Creation via Optimal Transport Distillation (...

</details>

<details>
<summary><b>17. PACEvolve: Enabling Long-Horizon Progress-Aware Consistent Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10657) • [📄 arXiv](https://arxiv.org/abs/2601.10657) • [📥 PDF](https://arxiv.org/pdf/2601.10657)

> No abstract available.

</details>

<details>
<summary><b>18. FlowAct-R1: Towards Interactive Humanoid Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10103) • [📄 arXiv](https://arxiv.org/abs/2601.10103) • [📥 PDF](https://arxiv.org/pdf/2601.10103)

> Project page link from the paper: https://grisoon.github.io/FlowAct-R1/

</details>

<details>
<summary><b>19. M^4olGen: Multi-Agent, Multi-Stage Molecular Generation under Precise Multi-Property Constraints</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10131) • [📄 arXiv](https://arxiv.org/abs/2601.10131) • [📥 PDF](https://arxiv.org/pdf/2601.10131)

> Generating molecules that satisfy precise numeric constraints over multiple physicochemical properties is critical and challenging. Although large language models (LLMs) are expressive, they struggle with precise multiobjective control and numeric...

</details>

<details>
<summary><b>20. HeartMuLa: A Family of Open Sourced Music Foundation Models</b> ⭐ 164</summary>

<br/>

**👥 Authors:** hhguo, YuanyuanWang, Gongxizhu, bverxie, Dongchao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10547) • [📄 arXiv](https://arxiv.org/abs/2601.10547) • [📥 PDF](https://arxiv.org/pdf/2601.10547)

**💻 Code:** [⭐ Code](https://github.com/HeartMuLa/heartlib)

> We present a family of open-source Music Foundation Models designed to advance large-scale music understanding and generation across diverse tasks and modalities. Our framework consists of four major components: (1) HeartCLAP, an audiotext alignm...

</details>

<details>
<summary><b>21. Action100M: A Large-scale Video Action Dataset</b> ⭐ 159</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10592) • [📄 arXiv](https://arxiv.org/abs/2601.10592) • [📥 PDF](https://arxiv.org/pdf/2601.10592)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/Action100M)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Vid...

</details>

<details>
<summary><b>22. Inference-time Physics Alignment of Video Generative Models with Latent World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10553) • [📄 arXiv](https://arxiv.org/abs/2601.10553) • [📥 PDF](https://arxiv.org/pdf/2601.10553)

> I assume I can also use any arbitrary off-the-shelf metric as a substitute for 'surprise' then?🤣 Coming soon: Inference-time Overall Alignment of Video Generative Models with Random Seed.😎

</details>

<details>
<summary><b>23. EvasionBench: Detecting Evasive Answers in Financial Q&A via Multi-Model Consensus and LLM-as-Judge</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yi Yang, Yan Lin, FutureMa

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09142) • [📄 arXiv](https://arxiv.org/abs/2601.09142) • [📥 PDF](https://arxiv.org/pdf/2601.09142)

> Thanks for featuring our work! 🚀 EvasionBench aims to bridge the gap in financial transparency. We've released the Eva-4B model and the 1k human-annotated test set. 📝 Paper: https://arxiv.org/abs/2601.09142 🤗 Model: https://huggingface.co/FutureMa...

</details>

<details>
<summary><b>24. TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tiankai Hang, Yiji Cheng, Juan Cao, Yu Xu, Yana-Hangabina

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08881) • [📄 arXiv](https://arxiv.org/abs/2601.08881) • [📥 PDF](https://arxiv.org/pdf/2601.08881)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API InstructMoLE: Instruction-Guided Mixture of Low-rank Experts for Multi-Cond...

</details>

<details>
<summary><b>25. LSRIF: Logic-Structured Reinforcement Learning for Instruction Following</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06431) • [📄 arXiv](https://arxiv.org/abs/2601.06431) • [📥 PDF](https://arxiv.org/pdf/2601.06431)

> In this work, we propose LSRIF, a logic-structured training framework. We construct LSRINSTRUCT, a multi-constraint instruction dataset covering parallel, sequential, and conditional constraint logic structures, and design LSRM, structure-aware re...

</details>

<details>
<summary><b>26. PRL: Process Reward Learning Improves LLMs' Reasoning Ability and Broadens the Reasoning Boundary</b> ⭐ 4</summary>

<br/>

**👥 Authors:** TongZhang, RickyDeSkywalker, FlippyDora

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10201) • [📄 arXiv](https://arxiv.org/abs/2601.10201) • [📥 PDF](https://arxiv.org/pdf/2601.10201)

**💻 Code:** [⭐ Code](https://github.com/MaxwellJryao/Process-Reward-Learning)

> Abstract Improving the reasoning abilities of Large Language Models (LLMs) has been a continuous topic recently. But most relevant works are based on outcome rewards at the trajectory level, missing fine-grained supervision during the reasoning pr...

</details>

<details>
<summary><b>27. LaViT: Aligning Latent Visual Thoughts for Multi-modal Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Linquan Wu, jackykeung, afunnyhy, fly1113, txjiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10129) • [📄 arXiv](https://arxiv.org/abs/2601.10129) • [📥 PDF](https://arxiv.org/pdf/2601.10129)

**💻 Code:** [⭐ Code](https://github.com/Svardfox/LaViT)

> https://github.com/Svardfox/LaViT

</details>

<details>
<summary><b>28. Deriving Character Logic from Storyline as Codified Decision Trees</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kun Zhou, shangjingbo, hyp1231, ylf1017, KomeijiForce

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10080) • [📄 arXiv](https://arxiv.org/abs/2601.10080) • [📥 PDF](https://arxiv.org/pdf/2601.10080)

> Deriving Character Logic from Storyline as Codified Decision Trees

</details>

<details>
<summary><b>29. Patient-Similarity Cohort Reasoning in Clinical Text-to-SQL</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09876) • [📄 arXiv](https://arxiv.org/abs/2601.09876) • [📥 PDF](https://arxiv.org/pdf/2601.09876)

**💻 Code:** [⭐ Code](https://github.com/Barryshen1/ClinSQL)

> Introducing ClinSQL, a 633-task expert-annotated benchmark on MIMIC-IV v3.1 for real-world clinical text-to-SQL.

</details>

<details>
<summary><b>30. RigMo: Unifying Rig and Motion Learning for Generative Animation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06378) • [📄 arXiv](https://arxiv.org/abs/2601.06378) • [📥 PDF](https://arxiv.org/pdf/2601.06378)

> 🚀 New work: RigMo — Unifying Rig & Motion Learning for Generative Animation Rigging and motion are two hard problems—usually solved separately. RigMo unifies them. A feed-forward framework that jointly learns rig structure + motion directly from r...

</details>

<details>
<summary><b>31. Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruitao Feng, Weizhe Wang, Gelei, yaozhang, sumleo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10338) • [📄 arXiv](https://arxiv.org/abs/2601.10338) • [📥 PDF](https://arxiv.org/pdf/2601.10338)

> Really interesting and timely work—agent “skills” seem like a rapidly growing supply-chain attack surface, and it’s refreshing to see a large-scale empirical analysis rather than a purely conceptual treatment. The scale and methodology (tens of th...

</details>

<details>
<summary><b>32. V-DPM: 4D Video Reconstruction with Dynamic Point Maps</b> ⭐ 55</summary>

<br/>

**👥 Authors:** vedaldi, zlai, einsafutdinov, edgarsucar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09499) • [📄 arXiv](https://arxiv.org/abs/2601.09499) • [📥 PDF](https://arxiv.org/pdf/2601.09499)

**💻 Code:** [⭐ Code](https://github.com/eldar/vdpm)

> No abstract available.

</details>

<details>
<summary><b>33. CaMeLs Can Use Computers Too: System-level Security for Computer Use Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** ftramer, nkristina, tom-bl, rcmullins, aprilflower

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.09923) • [📄 arXiv](https://arxiv.org/abs/2601.09923) • [📥 PDF](https://arxiv.org/pdf/2601.09923)

> When AI agents control your mouse, a single malicious email can drain your accounts. We built the first system-level defense to sandbox these agents, and the results challenged our assumptions about AI. It turns out, digital environments are more ...

</details>

<details>
<summary><b>34. WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10716) • [📄 arXiv](https://arxiv.org/abs/2601.10716) • [📥 PDF](https://arxiv.org/pdf/2601.10716)

> We present WildRayZer, a self-supervised framework for novel view synthesis (NVS) in dynamic environments where both the camera and objects move. Dynamic content breaks the multi-view consistency that static NVS models rely on, leading to ghosting...

</details>

<details>
<summary><b>35. VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Lei Zhu, xingzhaohu, yscript

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.10124) • [📄 arXiv](https://arxiv.org/abs/2601.10124) • [📥 PDF](https://arxiv.org/pdf/2601.10124)

**💻 Code:** [⭐ Code](https://github.com/script-Yang/VQ-Seg)

> Code available at: https://github.com/script-Yang/VQ-Seg

</details>

<details>
<summary><b>36. Enhancing Sentiment Classification and Irony Detection in Large Language Models through Advanced Prompt Engineering Techniques</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08302) • [📄 arXiv](https://arxiv.org/abs/2601.08302) • [📥 PDF](https://arxiv.org/pdf/2601.08302)

**💻 Code:** [⭐ Code](https://github.com/Marvin2108/ESCID-LLM-APET)

> Enhancing Sentiment Classification and Irony Detection in Large Language Models through Advanced Prompt Engineering Techniques

</details>

<details>
<summary><b>37. Demystifying the Slash Pattern in Attention: The Role of RoPE</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chao Du, Cunxiao Du, Yunlong Hou, Fengzhuo Zhang, Yuan Cheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.08297) • [📄 arXiv](https://arxiv.org/abs/2601.08297) • [📥 PDF](https://arxiv.org/pdf/2601.08297)

> Large Language Models (LLMs) often exhibit slash attention patterns, where attention scores concentrate along the Δ-th sub-diagonal for some offset Δ. These patterns play a key role in passing information across tokens. But why do they emerge? In ...

</details>

<details>
<summary><b>38. Memory Bank Compression for Continual Adaptation of Large Language Models</b> ⭐ 20</summary>

<br/>

**👥 Authors:** Dimitrios Rafailidis, Tomk187

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.00756) • [📄 arXiv](https://arxiv.org/abs/2601.00756) • [📥 PDF](https://arxiv.org/pdf/2601.00756)

**💻 Code:** [⭐ Code](https://github.com/Thomkat/MBC)

> Large Language Models (LLMs) have become a mainstay for many everyday applications. However, as data evolve their knowledge quickly becomes outdated. Continual learning aims to update LLMs with new information without erasing previously acquired k...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 38 |
| 📅 Today | [`2026-01-18.json`](data/daily/2026-01-18.json) | 38 |
| 📆 This Week | [`2026-W02.json`](data/weekly/2026-W02.json) | 232 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 429 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-18 | 38 | [View JSON](data/daily/2026-01-18.json) |
| 📄 2026-01-17 | 38 | [View JSON](data/daily/2026-01-17.json) |
| 📄 2026-01-16 | 27 | [View JSON](data/daily/2026-01-16.json) |
| 📄 2026-01-15 | 24 | [View JSON](data/daily/2026-01-15.json) |
| 📄 2026-01-14 | 42 | [View JSON](data/daily/2026-01-14.json) |
| 📄 2026-01-13 | 30 | [View JSON](data/daily/2026-01-13.json) |
| 📄 2026-01-12 | 33 | [View JSON](data/daily/2026-01-12.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 429 | [View JSON](data/monthly/2026-01.json) |
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
