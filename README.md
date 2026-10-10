<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-72-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7055+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">72</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">210</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">447</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7055+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 10, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. AgentGarten: Code Worlds for Evolving Agents</b> ⭐ 105</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12374) • [📄 arXiv](https://arxiv.org/abs/2610.12374) • [📥 PDF](https://arxiv.org/pdf/2610.12374)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MirroS-Lab/AgentGarten)

> Agents learn through interaction, and what they can learn is bounded by the environments they practice in. Those environments must be faithful, with consistent state, rules, and dynamics, and realistic, with observations that look like the real wo...

</details>

<details>
<summary><b>2. TokenRouter: Efficient Serving System for Token-Level LLM Routing</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12242) • [📄 arXiv](https://arxiv.org/abs/2610.12242) • [📥 PDF](https://arxiv.org/pdf/2610.12242)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/thu-nics/TokenRouter)

> We present TokenRouter, an efficient LLM serving engine for token-level routing.

</details>

<details>
<summary><b>3. Learn2Play Bench: How Well Do LLM Agents Learn from Experience in Unfamiliar Environments?</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08215) • [📄 arXiv](https://arxiv.org/abs/2610.08215) • [📥 PDF](https://arxiv.org/pdf/2610.08215)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/liushiliushi/Learn2Play-Bench)

> We introduce Learn2Play Bench, a benchmark of newly designed text-based games, whose rules are novel or counterintuitive, requiring agents to acquire knowledge through interaction rather than rely solely on pretrained knowledge.

</details>

<details>
<summary><b>4. From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation</b> ⭐ 29</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06100) • [📄 arXiv](https://arxiv.org/abs/2610.06100) • [📥 PDF](https://arxiv.org/pdf/2610.06100)

**💻 Code:** [⭐ Code](https://github.com/ruyue0001/Trace2Env) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ruyue0001/trace2env)

> What if the environment itself were an agent ? Trace2Env is a training-free framework for agentic language world modeling . It builds world model agents that serve as environments for other task agents, aiming to support faithful, stateful, and lo...

</details>

<details>
<summary><b>5. SuperNav: An Agentic Navigation System for Any Task in Any Scene</b> ⭐ 63</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12126) • [📄 arXiv](https://arxiv.org/abs/2610.12126) • [📥 PDF](https://arxiv.org/pdf/2610.12126)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zju3dv/SuperNav)

> SuperNav is an agentic navigation system for diverse tasks and scenes, from finding objects and visiting multiple targets to fulfilling high-level requests. A pretrained multimodal model directs navigation through tool use, task-progress tracking,...

</details>

<details>
<summary><b>6. MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11959) • [📄 arXiv](https://arxiv.org/abs/2610.11959) • [📥 PDF](https://arxiv.org/pdf/2610.11959)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Technical report of MiMo V2.6 series. Blog: https://mimo.xiaomi.com/mimo-v2-6 HF weights: https://huggingface.co/collections/XiaomiMiMo/mimo-v26

</details>

<details>
<summary><b>7. Multi-Agent Egocentric World Model with Fine-Grained Embodied Interaction</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12299) • [📄 arXiv](https://arxiv.org/abs/2610.12299) • [📥 PDF](https://arxiv.org/pdf/2610.12299)

**💻 Code:** [⭐ Code](https://github.com/cvlab-kaist/ME-World) • [⭐ Code](https://github.com/huggingface)

> Egocentric world models predict first-person observations conditioned on an agent's actions, but most focus on a single agent. Real embodied settings often involve multiple agents that act and interact within a shared environment. Existing multi-a...

</details>

<details>
<summary><b>8. In-context Robot Learning Made Simple: A Democratized Recipe for Manipulation Tasks</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Xiangshuo Liu, Weizhi Zhao, Minghao Han, Minxing Li, gothicwhw

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38173) • [📄 arXiv](https://arxiv.org/abs/2609.36012) • [📥 PDF](https://arxiv.org/pdf/2609.38173)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/simpleicl/simple-icl)

> We tackles a key ambiguity in robot in-context learning: what should a robot actually follow from a video demonstration? We defines the learning target across action, object, composition, and affordance semantics, then introduces a low-cost data r...

</details>

<details>
<summary><b>9. DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Mingchao Sun, Xiangshuo Liu, Yu Liu, Ruizhi Li, huoxingdawang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12468) • [📄 arXiv](https://arxiv.org/abs/2610.12468) • [📥 PDF](https://arxiv.org/pdf/2610.12468)

**💻 Code:** [⭐ Code](https://github.com/brave-eai/DreamTrue) • [⭐ Code](https://github.com/huggingface)

> We present DreamTrue, a multi-view, cross-embodiment robot world model for action-faithful and physically plausible video prediction. On AgiBot, DreamTrue attains state-of-the-art action following, while reducing the human-assessed interaction def...

</details>

<details>
<summary><b>10. Beyond Spatio-Temporal Priors: A Generalizable Approach for Dense Correspondence Matching</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12421) • [📄 arXiv](https://arxiv.org/abs/2610.12421) • [📥 PDF](https://arxiv.org/pdf/2610.12421)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/luping-liu/FreeMatching)

> FreeMatching extends dense correspondence to image editing and reference-guided generation through generative and semantic foundation representations, heterogeneous correspondence supervision, and teacher-guided iterative refinement.

</details>

<details>
<summary><b>11. OuroWorld: Bringing Any 3D World Alive as Diverse, Endlessly Looping 3D Cinemagraphs</b> ⭐ 34</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12461) • [📄 arXiv](https://arxiv.org/abs/2610.12461) • [📥 PDF](https://arxiv.org/pdf/2610.12461)

**💻 Code:** [⭐ Code](https://github.com/youzhe0305/OuroWorld) • [⭐ Code](https://github.com/huggingface)

> Recent 3D world models generate photorealistic, explorable scenes that remain frozen in time. OuroWorld is a mask-free framework that turns any static 3D Gaussian Splatting scene into a 3D cinemagraph: a dynamic scene with vivid, diverse motion lo...

</details>

<details>
<summary><b>12. U-Space: Uncovering When and Why Uncertainty Arises in Language Models</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09087) • [📄 arXiv](https://arxiv.org/abs/2610.09087) • [📥 PDF](https://arxiv.org/pdf/2610.09087)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/s2labres/U-Space)

> Can Mechanistic Interpretability be used for Uncertainty Quantification ? We present the U-Space , an interpretable subspace that tracks four sources of uncertainty through a model’s generation: Ambiguity , Incomplete information , Conflicting evi...

</details>

<details>
<summary><b>13. MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap in Diffusion Transformers</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06801) • [📄 arXiv](https://arxiv.org/abs/2610.06801) • [📥 PDF](https://arxiv.org/pdf/2610.06801)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/dodododddo/mcsparse)

> Sparse attention is a primary approach to reducing the latency of diffusion transformers in long-sequence generation tasks, such as video and high-resolution 3D asset generation. However, existing methods can degrade generation quality and fidelit...

</details>

<details>
<summary><b>14. TestPrism: Rethinking Test Evaluation Beyond a Single Reference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12289) • [📄 arXiv](https://arxiv.org/abs/2610.12289) • [📥 PDF](https://arxiv.org/pdf/2610.12289)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language model (LLM) coding agents have advanced test generation across diverse programming tasks. However, the common practice of evaluating tests against a single reference solution overlooks alternative valid implementations and can overs...

</details>

<details>
<summary><b>15. Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11794) • [📄 arXiv](https://arxiv.org/abs/2610.11794) • [📥 PDF](https://arxiv.org/pdf/2610.11794)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Agents in unfamiliar environments must infer the underlying rules and refine their understanding through experience. We introduce Memento 3 , a model-based approach to recursive self-improvement (RSI) with a frozen LLM. The agent builds a natural-...

</details>

<details>
<summary><b>16. Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02967) • [📄 arXiv](https://arxiv.org/abs/2610.02967) • [📥 PDF](https://arxiv.org/pdf/2610.02967)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Human preference alone isn’t enough for text-to-image post-training: an image can look appealing while missing prompt details or violating the requested style. This work combines a preference reward model trained on ~5 million human votes with rub...

</details>

<details>
<summary><b>17. SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12327) • [📄 arXiv](https://arxiv.org/abs/2610.12327) • [📥 PDF](https://arxiv.org/pdf/2610.12327)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce SparseDecoding, a principled decoding-aware LLM pruning framework, achieving 1.48x end-to-end wall-clock decoding speedup on A100 GPUs.

</details>

<details>
<summary><b>18. OneSearch-VL: Unified Multimodal Deep Research Agent for Image and Video</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Dian Zheng, Shu Chen, Kaituo Feng, Manyuan Zhang, appletea2333

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12419) • [📄 arXiv](https://arxiv.org/abs/2610.12419) • [📥 PDF](https://arxiv.org/pdf/2610.12419)

**💻 Code:** [⭐ Code](https://github.com/appletea233/OneSearch-VL) • [⭐ Code](https://github.com/huggingface)

> code: https://github.com/appletea233/OneSearch-VL data: https://huggingface.co/OneSearch-VL

</details>

<details>
<summary><b>19. LEGO: A Lifting-Free Approach for Exocentric-to-Egocentric Video Generation</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12442) • [📄 arXiv](https://arxiv.org/abs/2610.12442) • [📥 PDF](https://arxiv.org/pdf/2610.12442)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/suhwan-cho/LEGO)

> Hi @ akhaliq and the HF Papers team, thank you for featuring our paper! Our teaser video is very wide (2.75:1), so the Daily Papers card crops out half of the result. If possible, could you replace the media with the attached version, which is re-...

</details>

<details>
<summary><b>20. Foundations of Large Language Models</b> ⭐ 872</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2501.09223) • [📄 arXiv](https://arxiv.org/abs/2501.09223) • [📥 PDF](https://arxiv.org/pdf/2501.09223)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NiuTrans/NLPBook)

> Foundations of Large Language Models is an introductory book covering pre-training, generative models, prompting, alignment, inference, and reasoning. It presents the foundational concepts and methods behind LLMs for students, researchers, and pra...

</details>

<details>
<summary><b>21. Reasoning-Informed Visual Editing</b> ⭐ 158</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12343) • [📄 arXiv](https://arxiv.org/abs/2610.12343) • [📥 PDF](https://arxiv.org/pdf/2610.12343)

**💻 Code:** [⭐ Code](https://github.com/VisionXLab/RISEBench) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06406) • [📄 arXiv](https://arxiv.org/abs/2610.06406) • [📥 PDF](https://arxiv.org/pdf/2610.06406)

**💻 Code:** [⭐ Code](https://github.com/zhk-lab/EBG) • [⭐ Code](https://github.com/huggingface)

> As AI Agents Become More Autonomous, How Can We Know What They Actually Did? As AI agents become increasingly capable, the tasks they undertake are evolving from simple question answering and tool use to autonomous workflows that may last for hour...

</details>

<details>
<summary><b>23. SparseEngine: Sparse-First Inference Engine</b> ⭐ 81</summary>

<br/>

**👥 Authors:** Jun Yu, Qiang Huang, Quansheng Gu, Jitai Hao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39068) • [📄 arXiv](https://arxiv.org/abs/2609.39068) • [📥 PDF](https://arxiv.org/pdf/2609.39068)

**💻 Code:** [⭐ Code](https://github.com/CURRENTF/SparseEngine) • [⭐ Code](https://github.com/huggingface)

> Long-context LLM agents accumulate interaction histories that strain KV-cache memory and attention computation. Although sparse attention reduces these costs, heterogeneous cache representations and workflows hinder integration with existing infer...

</details>

<details>
<summary><b>24. VibeEdit: Image Editing with Canvas Instructions</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12229) • [📄 arXiv](https://arxiv.org/abs/2610.12229) • [📥 PDF](https://arxiv.org/pdf/2610.12229)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZhaoJingjing713/VibeEdit)

> In text-guided image editing, describing the desired change is often straightforward, but identifying the intended object or region can be cumbersome, especially when several objects look alike. We introduce a new image editing interface that lets...

</details>

<details>
<summary><b>25. SanSi: A Looped Typed Decision Model for System 1.5 Thinking</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07730) • [📄 arXiv](https://arxiv.org/abs/2610.07730) • [📥 PDF](https://arxiv.org/pdf/2610.07730)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/minnesotanlp/Sansi)

> TL;DR: Our work is a looped typed decision model that returns a probability for each declared option and can be read after any of its eight loops, so one model serves every compute budget without generating reasoning tokens. With the same data and...

</details>

<details>
<summary><b>26. USDCraft: Geometrically Grounded Programmatic Modeling of Articulated 3D Assets for Simulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11322) • [📄 arXiv](https://arxiv.org/abs/2610.11322) • [📥 PDF](https://arxiv.org/pdf/2610.11322)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> USDCraft reconstructs real-world articulated objects for simulation with an agentic harness: give it text, an image, or a mesh, and it builds a simulation-ready articulated asset with working doors, drawers, and hinges. The assets load straight in...

</details>

<details>
<summary><b>27. OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12458) • [📄 arXiv](https://arxiv.org/abs/2610.12458) • [📥 PDF](https://arxiv.org/pdf/2610.12458)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We propose OmniCapBench, a benchmark that reformulates audio-video captioning as structured prediction, enabling deterministic constraint checks and local semantic judging for reliable, fine-grained diagnosis of model failures.

</details>

<details>
<summary><b>28. Embodied Turing Machines: Stateful Code for Robot Recursive Self-Improvement</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ziwei Liu, Zhaoxi Chen, Fangzhou Hong, Siyuan Hu, Kairui Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12369) • [📄 arXiv](https://arxiv.org/abs/2610.12369) • [📥 PDF](https://arxiv.org/pdf/2610.12369)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Code was developed on development episodes; benchmark test episodes were not used for development. Oracle values are not used in code. This is not an attempt to achieve SOTA on RoboDojo, i.e., it is not meant to hack a benchmark. We hope to offer ...

</details>

<details>
<summary><b>29. ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12403) • [📄 arXiv](https://arxiv.org/abs/2610.12403) • [📥 PDF](https://arxiv.org/pdf/2610.12403)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/ViSkill)

> This paper proposes ViSkill, a visual-native skill learning framework that represents successful interactions as reusable visual skill cards and couples geometry-aware retrieval, skill-guided reinforcement learning, and online skill distillation i...

</details>

<details>
<summary><b>30. Do LLMs Understand Sequential Structure? A Controlled Study of Inference and Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04977) • [📄 arXiv](https://arxiv.org/abs/2610.04977) • [📥 PDF](https://arxiv.org/pdf/2610.04977)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Do LLMs truly understand sequential behavior, or do they simply reproduce surface-level patterns? We investigate this question through controlled experiments on strategy identification, Markov rule following, and higher-order sequential dependenci...

</details>

<details>
<summary><b>31. V-CoLA: Vision Token Compression with Linear Attention</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11251) • [📄 arXiv](https://arxiv.org/abs/2610.11251) • [📥 PDF](https://arxiv.org/pdf/2610.11251)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Vision token compression is one of the main ways to make VLMs faster, but most existing methods depend on softmax attention. That makes them a poor fit for emerging hybrid VLMs with linear attention, such as Qwen3.5. We introduce V-CoLA, a trainin...

</details>

<details>
<summary><b>32. Opera: A Verbal Critic Framework for Long-horizon Coding Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33987) • [📄 arXiv](https://arxiv.org/abs/2609.33987) • [📥 PDF](https://arxiv.org/pdf/2609.33987)

**💻 Code:** [⭐ Code](https://github.com/dongyuanjushi/Opera) • [⭐ Code](https://github.com/huggingface)

> Opera is a verbal critic for long-horizon coding agents that treats each correction as a persistent note and follows it until the diagnosed problem is actually resolved, rather than stopping once feedback is delivered. It decides when to review th...

</details>

<details>
<summary><b>33. ReSPO: Reshaped Sequence Policy Optimization for Gradient Starvation in Off-Policy Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35433) • [📄 arXiv](https://arxiv.org/abs/2609.35433) • [📥 PDF](https://arxiv.org/pdf/2609.35433)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yhangchen/ReSPO-code)

> Excited to share ReSPO, a new policy-optimization method designed for off-policy RL under high rollout staleness: a particularly difficult setting for large MoE models. ReSPO reshapes sequence-level importance weights to prevent gradient starvatio...

</details>

<details>
<summary><b>34. SpatialOPSD: Self-Distilling Spatial Intelligence from Verified Coding Agent Traces</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11366) • [📄 arXiv](https://arxiv.org/abs/2610.11366) • [📥 PDF](https://arxiv.org/pdf/2610.11366)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> With the widespread adoption of coding agents, we explore an approach to distilling the spatial reasoning capabilities of coding agents into standalone MLLMs through on-policy self-distillation (OPSD).

</details>

<details>
<summary><b>35. From Prompting to Composing: A Spatial Canvas Interface for Poster Generation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12230) • [📄 arXiv](https://arxiv.org/abs/2610.12230) • [📥 PDF](https://arxiv.org/pdf/2610.12230)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/snowflakewang/Compo)

> Introducing Compo , a poster generation model adapted from a pretrained image editing model without task-specific architecture design to interpret Spatial Canvas Interface and their associated Text Specifications . Our interface supports four comp...

</details>

<details>
<summary><b>36. Retrieval-Centric Deep Learning in Growing Nonparametric Neural Networks</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03858) • [📄 arXiv](https://arxiv.org/abs/2610.03858) • [📥 PDF](https://arxiv.org/pdf/2610.03858)

**💻 Code:** [⭐ Code](https://github.com/paradigms-of-intelligence/rcdl) • [⭐ Code](https://github.com/huggingface)

> TLDR: A well-known duality connects training of linear layers by gradient descent with linear attention over training examples. This paper builds on this insight and proposes retrieval-centric deep learning which replaces the standard linear atten...

</details>

<details>
<summary><b>37. SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12402) • [📄 arXiv](https://arxiv.org/abs/2610.12402) • [📥 PDF](https://arxiv.org/pdf/2610.12402)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/SpaceCast-Bench)

> This paper introduces SpaceCast-Bench, a benchmark for predictive spatial reasoning that evaluates whether vision-language models can infer unseen outcomes of spatial transformations, with 3,862 questions across 16 task types covering static perce...

</details>

<details>
<summary><b>38. A GPU-Parallel Framework for Heterogeneous Multi-Task Reinforcement Learning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.03335) • [📄 arXiv](https://arxiv.org/abs/2606.03335) • [📥 PDF](https://arxiv.org/pdf/2606.03335)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hebero-rl/Hebero)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Task-Space Imitation Guidance for Efficient Reinforcement Learning (2026) F...

</details>

<details>
<summary><b>39. Can AI Agents Make Open-Ended Scientific Discovery? Evidence from Station</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08927) • [📄 arXiv](https://arxiv.org/abs/2610.08927) • [📥 PDF](https://arxiv.org/pdf/2610.08927)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/dualverse-ai/station-open-reseach)

> Can AI agents make open-ended scientific discovery? We built open-ended research tasks from three ICLR oral papers, giving agents the research questions and experimental setups while withholding the original findings and disabling web access. Stat...

</details>

<details>
<summary><b>40. REMORY: Learning Residual Memory for Context Compaction</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Senqiao Yang, Naihao Deng, Yutang Ge, Baoyou Chen, mocoV3

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11287) • [📄 arXiv](https://arxiv.org/abs/2610.11287) • [📥 PDF](https://arxiv.org/pdf/2610.11287)

**💻 Code:** [⭐ Code](https://github.com/1ring2rta/Remory) • [⭐ Code](https://github.com/huggingface)

> Remory appends learned soft memory tokens to a compacted summary. On Qwen3.8-27B and GLM-5.3-Flash, it improves long-horizon benchmark performance while reducing repeated tool outputs and tool errors.

</details>

<details>
<summary><b>41. Chaos in the Text: Revealing the Modality Preference in Mixed-Modality Retrievers</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zhipeng Xu, Zhenghao Liu, Yukun Yan, Yubo Sun, hmhm1229

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11816) • [📄 arXiv](https://arxiv.org/abs/2610.11816) • [📥 PDF](https://arxiv.org/pdf/2610.11816)

**💻 Code:** [⭐ Code](https://github.com/OpenBMB/Trident) • [⭐ Code](https://github.com/huggingface)

> Dense retrievers have made significant progress on text and image corpora, but whether these capabilities extend reliably to mixed corpora containing text, image, and fused text-image documents remains unclear. In this paper, we systematically exa...

</details>

<details>
<summary><b>42. A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Taiye Lu, Yu-Jie Zhou, Ke Xue, Rong-Xi Tan, Ming Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12183) • [📄 arXiv](https://arxiv.org/abs/2610.12183) • [📥 PDF](https://arxiv.org/pdf/2610.12183)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/lamda-bbo/agentic-bbo)

> We introduce AgenticBBO-Bench, a unified benchmark across five black-box optimization domains. In our experiments, LLM agents outperform the best tested numerical optimizers in four of five domains. Key takeaways: more tools do not always help, ta...

</details>

<details>
<summary><b>43. WorldGuide: Goal-Directed Video World Model for Procedural Task Execution</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12459) • [📄 arXiv](https://arxiv.org/abs/2610.12459) • [📥 PDF](https://arxiv.org/pdf/2610.12459)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/mbzuai-oryx/WorldGuide)

> Excited to share WorldGuide, a goal-directed video world model for procedural task execution. Instead of generating a fixed open-loop rollout, WorldGuide treats procedural video generation as closed-loop task execution: a ContextPlanner predicts t...

</details>

<details>
<summary><b>44. Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Lu Yin, Guinan Su, Di He, Dilxat Muhtar, Pengxiang Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11570) • [📄 arXiv](https://arxiv.org/abs/2610.11570) • [📥 PDF](https://arxiv.org/pdf/2610.11570)

**💻 Code:** [⭐ Code](https://github.com/pixeli99/InfiLoop) • [⭐ Code](https://github.com/huggingface)

> 1/ More loops ≠ better reasoning. On Sudoku-Extreme, TRM peaks at 62% and then drops as you keep looping. Puzzles it already solved get broken again. 2/ The culprit is “carry-last”: every iteration just overwrites the state with the latest output....

</details>

<details>
<summary><b>45. Mara Chain: Rethinking Failure as a Stepping Stone for AI System Auto-Evolution</b> ⭐ 1.12k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35855) • [📄 arXiv](https://arxiv.org/abs/2609.35855) • [📥 PDF](https://arxiv.org/pdf/2609.35855)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ant-research/AntOmniEvo)

> Excited to share Mara Chain — rethinking failure as a stepping stone for AI system auto-evolution! 🧬 Standard propose–evaluate–select optimization discards every candidate that fails the acceptance bar — so later proposals keep revisiting the same...

</details>

<details>
<summary><b>46. Synthesis Through Simulation: Generating Coherent Enterprise Data via Scalable Agent-System Interaction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.10549) • [📄 arXiv](https://arxiv.org/abs/2610.10549) • [📥 PDF](https://arxiv.org/pdf/2610.10549)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Generating Enterprise Data by only interacting with APIs without any access to database or environment schema.

</details>

<details>
<summary><b>47. Incremental Open-Ended Deep Research with Structured Harness</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hongyuan Bao, merlinarer

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11566) • [📄 arXiv](https://arxiv.org/abs/2610.11566) • [📥 PDF](https://arxiv.org/pdf/2610.11566)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Existing Open-Ended Deep Research (OEDR) systems primarily generate reports from scratch, making them inefficient for scenarios where research reports need to be continuously maintained as new information emerges. We introduce Incremental Open-End...

</details>

<details>
<summary><b>48. One Block, Multiple Depths: Recurrent Vision Transformers with Depth-Programmed Experts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12448) • [📄 arXiv](https://arxiv.org/abs/2610.12448) • [📥 PDF](https://arxiv.org/pdf/2610.12448)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> In this work, we show that a single Transformer block, applied recurrently, can match the accuracy of a full-depth vision encoder at comparable inference FLOPs without intermediate feature distillation. reViT restores depth-specific transformation...

</details>

<details>
<summary><b>49. SpaceFlow: Locally Controllable 3D Generation</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12399) • [📄 arXiv](https://arxiv.org/abs/2610.12399) • [📥 PDF](https://arxiv.org/pdf/2610.12399)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SpaceFlow3D/spaceflow)

> SpaceFlow: Locally Controllable 3D Generation

</details>

<details>
<summary><b>50. BrickBench: Evaluating Agentic Brick Design</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Jiajun Wu, Cordelia Schmid, R. Kenny Jones, Yiqing Xu, Peter Kulits

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12452) • [📄 arXiv](https://arxiv.org/abs/2610.12452) • [📥 PDF](https://arxiv.org/pdf/2610.12452)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kulits/BrickAgent)

> We evaluate coding agents on their ability to design LEGO assemblies from a text prompt across three settings with different build constraints.

</details>

<details>
<summary><b>51. Accurate but Not Humble: Evaluating Epistemic Humility in LLM Agents under Knowledge Conflict</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12360) • [📄 arXiv](https://arxiv.org/abs/2610.12360) • [📥 PDF](https://arxiv.org/pdf/2610.12360)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/KaiserWhoLearns/EpistemicHumilityLLMAgents)

> When retrieved evidence contradicts an agent's prior beliefs, does it revise its answer, acknowledge uncertainty, or persist with an incorrect conclusion? Existing evaluations of agentic systems focus primarily on task success, offering limited in...

</details>

<details>
<summary><b>52. Frozen Models, Evolving Expertise: Model-Agnostic Learning from Deployment Experience for Multimodal Medical AI</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09146) • [📄 arXiv](https://arxiv.org/abs/2610.09146) • [📥 PDF](https://arxiv.org/pdf/2610.09146)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present FMEE (Frozen Models, Evolving Expertise) , a model-agnostic framework that lets frozen LLMs and VLMs learn from deployment experience through three forms of external expertise: Skill : a natural-language procedure for reasoning and tool...

</details>

<details>
<summary><b>53. SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04875) • [📄 arXiv](https://arxiv.org/abs/2610.04875) • [📥 PDF](https://arxiv.org/pdf/2610.04875)

**💻 Code:** [⭐ Code](https://github.com/chungen04/specfold) • [⭐ Code](https://github.com/huggingface)

> Specfold targets the forward and verification step cost in multi-branch speculative Diffusion Language Models (DLLM) decoding with an algorithm-system co-design. Across two DLLM families, five models, and five standard benchmarks, SpecFold achieve...

</details>

<details>
<summary><b>54. Pumpire: Unified Benchmark for Metric Distance Estimation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jialei Wang, Yihan Wu, Jiayang Xu, Zehan Wang, Rain729

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12423) • [📄 arXiv](https://arxiv.org/abs/2610.12423) • [📥 PDF](https://arxiv.org/pdf/2610.12423)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SpatialVision/Pumpire)

> We present Pumpire, a unified benchmark for evaluating metric point-pair distance estimation in image- and video-level 3D foundation models, with or without depth priors. We first manually collect a large-scale dataset comprising 6,400 frames from...

</details>

<details>
<summary><b>55. Investigating the Role of Reasoning-Language Alignment in Monolingual Retrieval-Augmented Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Katharina von der Wense, Oliver Hauck, mario-sanz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03136) • [📄 arXiv](https://arxiv.org/abs/2610.03136) • [📥 PDF](https://arxiv.org/pdf/2610.03136)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We investigate the role of reasoning-language alignment in monolingual RAG.

</details>

<details>
<summary><b>56. You Changed Your Mind, The Model Didn't: Demystifying Intent in Multi-Turn Dialogue</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Yuxuan Liu, Zhoujin Tian, Zhengjun Huang, Wei Chen, Junle-cs

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06496) • [📄 arXiv](https://arxiv.org/abs/2610.06496) • [📥 PDF](https://arxiv.org/pdf/2610.06496)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/junle-chen/intent)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API When Users Change Their Minds: Measuring and Repairing Intent Drift in LLM ...

</details>

<details>
<summary><b>57. EDiS: Edge Disjoint Subgraph Sparsification Framework for Graph Neural Networks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ryan A. Rossi, S M Ferdous, Siddhartha Shankar Das, Sai Karthik Navuluru, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09059) • [📄 arXiv](https://arxiv.org/abs/2610.09059) • [📥 PDF](https://arxiv.org/pdf/2610.09059)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Scaffold: Support Graph Theory Based Sparsification for Graph Neural Networ...

</details>

<details>
<summary><b>58. CARE: Certifying Acceleration for Vision-Language-Action Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08917) • [📄 arXiv](https://arxiv.org/abs/2610.08917) • [📥 PDF](https://arxiv.org/pdf/2610.08917)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> While vision-language-action (VLA) models have advanced rapidly, running them at every control step remains expensive. Prior work accelerates VLA inference using techniques like action chunking and visual-token pruning, typically evaluating based ...

</details>

<details>
<summary><b>59. On-Policy Distillation Teaches New Skills but Not New Knowledge</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yi Yang, Yixuan Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09639) • [📄 arXiv](https://arxiv.org/abs/2610.09639) • [📥 PDF](https://arxiv.org/pdf/2610.09639)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: On-policy distillation (OPD) does not expand parametric memory—it teaches models how to organize what they already know. Key Findings: Skill vs. Knowledge: Reverse-KL OPD reliably transfers compositional reasoning across unseen structures, ...

</details>

<details>
<summary><b>60. Distilling Routed 3D Privilege for Spatial Reasoning in Vision-Language Models</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12355) • [📄 arXiv](https://arxiv.org/abs/2610.12355) • [📥 PDF](https://arxiv.org/pdf/2610.12355)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/GPD) • [⭐ Code](https://github.com/huggingface)

> This paper proposes GPD, a geometry-privileged distillation framework that routes question-relevant 3D cues and reference answers to a teacher during on-policy self-distillation, augmenting GRPO with distillation on incorrect trajectories while ke...

</details>

<details>
<summary><b>61. Skill Constellations: Tracing the Supply Chain of Agent Skills on GitHub</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11169) • [📄 arXiv](https://arxiv.org/abs/2610.11169) • [📥 PDF](https://arxiv.org/pdf/2610.11169)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/FahdSeddik/Skill-Constellations)

> Agent skills are SKILL.md instructions and scripts that AI coding agents such as Claude Code and Codex run with the permissions of their user. Developers share skills by copying them between repositories, which makes them a software supply chain w...

</details>

<details>
<summary><b>62. Predicting Cable Dynamics with Physical Attention Bias</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Maya Bechler-Speicher, Avishai Sintov, RotemAtari, avihaig

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11975) • [📄 arXiv](https://arxiv.org/abs/2610.11975) • [📥 PDF](https://arxiv.org/pdf/2610.11975)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/avihaig/dlogps)

> We study which distance should bias global attention in a learned cable simulator: arc-length along the cable, which governs elastic forces, or Euclidean distance in space, which governs contact. A physical attention bias improves prediction on un...

</details>

<details>
<summary><b>63. Incidental information contaminates patient notes and disrupts clinical reasoning in large language models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08585) • [📄 arXiv](https://arxiv.org/abs/2610.08585) • [📥 PDF](https://arxiv.org/pdf/2610.08585)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language models (LLMs) are increasingly relied upon to support ambient documentation and clinical reasoning. Here we examine the impact of a failure mode shared between these two applications by assessing their sensitivity to information inc...

</details>

<details>
<summary><b>64. Learning to Steer, Steering to See: Unveiling the Geometry of RLVR in Large Language Models via Trainable Vectors</b> ⭐ 43</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34344) • [📄 arXiv](https://arxiv.org/abs/2609.34344) • [📥 PDF](https://arxiv.org/pdf/2609.34344)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/caiyuchen-ustc/On_Policy_Vector_Training)

> Reinforcement learning (RL) has become a key paradigm for enhancing the reasoning of large language models, yet the high dimensionality of parameter updates makes its training dynamics hard to analyze. We study reinforcement learning with verifiab...

</details>

<details>
<summary><b>65. MARGIN: Runtime Confidence Calibration for Multi-Agent Foundation Model Coordination</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.22949) • [📄 arXiv](https://arxiv.org/abs/2605.22949) • [📥 PDF](https://arxiv.org/pdf/2605.22949)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Self-reported confidence is the wrong signal for combining answers from multiple LLMs. On BigCodeBench, across 18 models, confidence runs backwards. Models with higher average confidence pass fewer tests. When one of two answers is right and the o...

</details>

<details>
<summary><b>66. The Lattice of Transition Laws</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11216) • [📄 arXiv](https://arxiv.org/abs/2610.11216) • [📥 PDF](https://arxiv.org/pdf/2610.11216)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/TSUITUENYUE/The-Lattice-of-Transition-Laws)

> Can we decide which decoding schedule works best BEFORE decoding, for diffusion, AR or anything in between? The answer is YES. In this paper, we unify diffusion, AR and their hybrids on one corruption lattice, where every decoder is a path. A sche...

</details>

<details>
<summary><b>67. Behavioral Persistence and Incomplete Functional Transfer of Co-evolved Communication in Evolutionary Robotics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38527) • [📄 arXiv](https://arxiv.org/abs/2609.38527) • [📥 PDF](https://arxiv.org/pdf/2609.38527)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ferdiex/essim)

> This paper evaluates the direct transfer of a co-evolved communication protocol from a 2D simulation to a 3D physical robotic environment without retraining the controller. Experiments with two GRU-controlled e-puck-type robots revealed partial an...

</details>

<details>
<summary><b>68. Evaluating the Transfer of Co-Evolved Communication from 2D to 3D Simulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.09280) • [📄 arXiv](https://arxiv.org/abs/2610.09280) • [📥 PDF](https://arxiv.org/pdf/2610.09280)

**💻 Code:** [⭐ Code](https://github.com/ferdiex/essim2d3d) • [⭐ Code](https://github.com/huggingface)

> This paper investigates whether a communication mechanism co-evolved between two robotic agents in a discrete 2D environment preserves its functional role after transfer to a 3D physics-based simulator. We analyze the effects of episode duration, ...

</details>

<details>
<summary><b>69. MIRA: A Musical Intent Refinement Agent for Aligning Text-to-Music Generation with User Intent</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yang Yang, Yu Cheng, Xuzheng He, Zhilin Wang, KKKKAAAAIIII

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.10355) • [📄 arXiv](https://arxiv.org/abs/2610.10355) • [📥 PDF](https://arxiv.org/pdf/2610.10355)

**💻 Code:** [⭐ Code](https://github.com/mirareview/MIRA) • [⭐ Code](https://github.com/huggingface)

> Hi everyone! I'm Zekai Liu, the first author of MIRA. We study a practical question: how can text-to-music systems better capture what users actually intend, including requirements that are only implied by a prompt? Our work introduces: MuRA-Bench...

</details>

<details>
<summary><b>70. SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Zeyuan Yang, Yanxing Wu, Zichun Chen, Chunliang Hua, Eku127

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.31507) • [📄 arXiv](https://arxiv.org/abs/2609.31507) • [📥 PDF](https://arxiv.org/pdf/2609.31507)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Eku127/SatNav)

> Hi everyone! We’re excited to share SatNav, our benchmark for long-horizon UAV vision-language navigation built from high-resolution satellite imagery. Highlights: 118K navigation episodes across 59 scenes in 18 cities, with an average trajectory ...

</details>

<details>
<summary><b>71. SPW-Nav: A Streaming Panoramic World Model for Language-Guided Navigation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08941) • [📄 arXiv](https://arxiv.org/abs/2610.08941) • [📥 PDF](https://arxiv.org/pdf/2610.08941)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: SPW-Nav is a real-time, language-guided panoramic world model that generates minute-long 2K 360° videos from a single panorama, enabling interactive navigation through spherical rotation decoupling, pose-aligned conditioning, and streaming ...

</details>

<details>
<summary><b>72. TerraVis: Towards Evaluation of World-Grounded Visual Consistency in Text-to-Image Generation via MLLM Workflows</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02959) • [📄 arXiv](https://arxiv.org/abs/2610.02959) • [📥 PDF](https://arxiv.org/pdf/2610.02959)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ShyFoo/TerraVis)

> Excited to share TerraVis , accepted to NeurIPS 2026 (Evaluations & Datasets Track)! 🎉🎉🎉 While existing T2I metrics primarily focus on prompt alignment and visual quality, TerraVis addresses a long-standing gap in text-to-image evaluation: world c...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 72 |
| 📅 Today | [`2026-10-10.json`](data/daily/2026-10-10.json) | 72 |
| 📆 This Week | [`2026-W40.json`](data/weekly/2026-W40.json) | 210 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 447 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-10 | 72 | [View JSON](data/daily/2026-10-10.json) |
| 📄 2026-10-09 | 35 | [View JSON](data/daily/2026-10-09.json) |
| 📄 2026-10-08 | 40 | [View JSON](data/daily/2026-10-08.json) |
| 📄 2026-10-07 | 23 | [View JSON](data/daily/2026-10-07.json) |
| 📄 2026-10-06 | 21 | [View JSON](data/daily/2026-10-06.json) |
| 📄 2026-10-05 | 19 | [View JSON](data/daily/2026-10-05.json) |
| 📄 2026-10-04 | 84 | [View JSON](data/daily/2026-10-04.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W40 | 210 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 447 | [View JSON](data/monthly/2026-10.json) |
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
