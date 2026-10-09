<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-35-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6983+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">35</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">138</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">375</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6983+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 09, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06100) • [📄 arXiv](https://arxiv.org/abs/2610.06100) • [📥 PDF](https://arxiv.org/pdf/2610.06100)

**💻 Code:** [⭐ Code](https://github.com/ruyue0001/Trace2Env) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ruyue0001/trace2env)

> What if the environment itself were an agent ? Trace2Env is a training-free framework for agentic language world modeling . It builds world model agents that serve as environments for other task agents, aiming to support faithful, stateful, and lo...

</details>

<details>
<summary><b>2. SuperNav: An Agentic Navigation System for Any Task in Any Scene</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12126) • [📄 arXiv](https://arxiv.org/abs/2610.12126) • [📥 PDF](https://arxiv.org/pdf/2610.12126)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zju3dv/SuperNav)

> SuperNav is an agentic navigation system for diverse tasks and scenes, from finding objects and visiting multiple targets to fulfilling high-level requests. A pretrained multimodal model directs navigation through tool use, task-progress tracking,...

</details>

<details>
<summary><b>3. TokenRouter: Efficient Serving System for Token-Level LLM Routing</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12242) • [📄 arXiv](https://arxiv.org/abs/2610.12242) • [📥 PDF](https://arxiv.org/pdf/2610.12242)

**💻 Code:** [⭐ Code](https://github.com/thu-nics/TokenRouter) • [⭐ Code](https://github.com/huggingface)

> We present TokenRouter, an efficient LLM serving engine for token-level routing.

</details>

<details>
<summary><b>4. Learn2Play Bench: How Well Do LLM Agents Learn from Experience in Unfamiliar Environments?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.08215) • [📄 arXiv](https://arxiv.org/abs/2610.08215) • [📥 PDF](https://arxiv.org/pdf/2610.08215)

**💻 Code:** [⭐ Code](https://github.com/liushiliushi/Learn2Play-Bench) • [⭐ Code](https://github.com/huggingface)

> We introduce Learn2Play Bench, a benchmark of newly designed text-based games, whose rules are novel or counterintuitive, requiring agents to acquire knowledge through interaction rather than rely solely on pretrained knowledge.

</details>

<details>
<summary><b>5. Multi-Agent Egocentric World Model with Fine-Grained Embodied Interaction</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12299) • [📄 arXiv](https://arxiv.org/abs/2610.12299) • [📥 PDF](https://arxiv.org/pdf/2610.12299)

**💻 Code:** [⭐ Code](https://github.com/cvlab-kaist/ME-World) • [⭐ Code](https://github.com/huggingface)

> Egocentric world models predict first-person observations conditioned on an agent's actions, but most focus on a single agent. Real embodied settings often involve multiple agents that act and interact within a shared environment. Existing multi-a...

</details>

<details>
<summary><b>6. AgentGarten: Code Worlds for Evolving Agents</b> ⭐ 54</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12374) • [📄 arXiv](https://arxiv.org/abs/2610.12374) • [📥 PDF](https://arxiv.org/pdf/2610.12374)

**💻 Code:** [⭐ Code](https://github.com/MirroS-Lab/AgentGarten) • [⭐ Code](https://github.com/huggingface)

> Agents learn through interaction, and what they can learn is bounded by the environments they practice in. Those environments must be faithful, with consistent state, rules, and dynamics, and realistic, with observations that look like the real wo...

</details>

<details>
<summary><b>7. DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mingchao Sun, Xiangshuo Liu, Yu Liu, Ruizhi Li, Junyan Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12468) • [📄 arXiv](https://arxiv.org/abs/2610.12468) • [📥 PDF](https://arxiv.org/pdf/2610.12468)

**💻 Code:** [⭐ Code](https://github.com/brave-eai/DreamTrue) • [⭐ Code](https://github.com/huggingface)

> We present DreamTrue, a multi-view, cross-embodiment robot world model for action-faithful and physically plausible video prediction. On AgiBot, DreamTrue attains state-of-the-art action following, while reducing the human-assessed interaction def...

</details>

<details>
<summary><b>8. In-context Robot Learning Made Simple: A Democratized Recipe for Manipulation Tasks</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Xiangshuo Liu, Hanwen Wang, Weizhi Zhao, Minghao Han, Minxing Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38173) • [📄 arXiv](https://arxiv.org/abs/2609.38173) • [📥 PDF](https://arxiv.org/pdf/2609.38173)

**💻 Code:** [⭐ Code](https://github.com/simpleicl/simple-icl) • [⭐ Code](https://github.com/huggingface)

> We tackles a key ambiguity in robot in-context learning: what should a robot actually follow from a video demonstration? We defines the learning target across action, object, composition, and affordance semantics, then introduces a low-cost data r...

</details>

<details>
<summary><b>9. SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12327) • [📄 arXiv](https://arxiv.org/abs/2610.12327) • [📥 PDF](https://arxiv.org/pdf/2610.12327)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce SparseDecoding, a principled decoding-aware LLM pruning framework, achieving 1.48x end-to-end wall-clock decoding speedup on A100 GPUs.

</details>

<details>
<summary><b>10. What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06406) • [📄 arXiv](https://arxiv.org/abs/2610.06406) • [📥 PDF](https://arxiv.org/pdf/2610.06406)

**💻 Code:** [⭐ Code](https://github.com/zhk-lab/EBG) • [⭐ Code](https://github.com/huggingface)

> As AI Agents Become More Autonomous, How Can We Know What They Actually Did? As AI agents become increasingly capable, the tasks they undertake are evolving from simple question answering and tool use to autonomous workflows that may last for hour...

</details>

<details>
<summary><b>11. MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap in Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.06801) • [📄 arXiv](https://arxiv.org/abs/2610.06801) • [📥 PDF](https://arxiv.org/pdf/2610.06801)

**💻 Code:** [⭐ Code](https://github.com/dodododddo/mcsparse) • [⭐ Code](https://github.com/huggingface)

> Sparse attention is a primary approach to reducing the latency of diffusion transformers in long-sequence generation tasks, such as video and high-resolution 3D asset generation. However, existing methods can degrade generation quality and fidelit...

</details>

<details>
<summary><b>12. Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02967) • [📄 arXiv](https://arxiv.org/abs/2610.02967) • [📥 PDF](https://arxiv.org/pdf/2610.02967)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Human preference alone isn’t enough for text-to-image post-training: an image can look appealing while missing prompt details or violating the requested style. This work combines a preference reward model trained on ~5 million human votes with rub...

</details>

<details>
<summary><b>13. LEGO: A Lifting-Free Approach for Exocentric-to-Egocentric Video Generation</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12442) • [📄 arXiv](https://arxiv.org/abs/2610.12442) • [📥 PDF](https://arxiv.org/pdf/2610.12442)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/suhwan-cho/LEGO)

> Hi @ akhaliq and the HF Papers team, thank you for featuring our paper! Our teaser video is very wide (2.75:1), so the Daily Papers card crops out half of the result. Could you replace the media with the attached version, which is re-framed for th...

</details>

<details>
<summary><b>14. VibeEdit: Image Editing with Canvas Instructions</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12229) • [📄 arXiv](https://arxiv.org/abs/2610.12229) • [📥 PDF](https://arxiv.org/pdf/2610.12229)

**💻 Code:** [⭐ Code](https://github.com/ZhaoJingjing713/VibeEdit) • [⭐ Code](https://github.com/huggingface)

> In text-guided image editing, describing the desired change is often straightforward, but identifying the intended object or region can be cumbersome, especially when several objects look alike. We introduce a new image editing interface that lets...

</details>

<details>
<summary><b>15. TestPrism: Rethinking Test Evaluation Beyond a Single Reference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12289) • [📄 arXiv](https://arxiv.org/abs/2610.12289) • [📥 PDF](https://arxiv.org/pdf/2610.12289)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language model (LLM) coding agents have advanced test generation across diverse programming tasks. However, the common practice of evaluating tests against a single reference solution overlooks alternative valid implementations and can overs...

</details>

<details>
<summary><b>16. USDCraft: Geometrically Grounded Programmatic Modeling of Articulated 3D Assets for Simulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11322) • [📄 arXiv](https://arxiv.org/abs/2610.11322) • [📥 PDF](https://arxiv.org/pdf/2610.11322)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> USDCraft reconstructs real-world articulated objects for simulation with an agentic harness: give it text, an image, or a mesh, and it builds a simulation-ready articulated asset with working doors, drawers, and hinges. The assets load straight in...

</details>

<details>
<summary><b>17. ReSPO: Reshaped Sequence Policy Optimization for Gradient Starvation in Off-Policy Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35433) • [📄 arXiv](https://arxiv.org/abs/2609.35433) • [📥 PDF](https://arxiv.org/pdf/2609.35433)

**💻 Code:** [⭐ Code](https://github.com/yhangchen/ReSPO-code) • [⭐ Code](https://github.com/huggingface)

> Excited to share ReSPO, a new policy-optimization method designed for off-policy RL under high rollout staleness: a particularly difficult setting for large MoE models. ReSPO reshapes sequence-level importance weights to prevent gradient starvatio...

</details>

<details>
<summary><b>18. Do LLMs Understand Sequential Structure? A Controlled Study of Inference and Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.04977) • [📄 arXiv](https://arxiv.org/abs/2610.04977) • [📥 PDF](https://arxiv.org/pdf/2610.04977)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Do LLMs truly understand sequential behavior, or do they simply reproduce surface-level patterns? We investigate this question through controlled experiments on strategy identification, Markov rule following, and higher-order sequential dependenci...

</details>

<details>
<summary><b>19. From Prompting to Composing: A Spatial Canvas Interface for Poster Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12230) • [📄 arXiv](https://arxiv.org/abs/2610.12230) • [📥 PDF](https://arxiv.org/pdf/2610.12230)

**💻 Code:** [⭐ Code](https://github.com/snowflakewang/Compo) • [⭐ Code](https://github.com/huggingface)

> Introducing Compo , a poster generation model adapted from a pretrained image editing model without task-specific architecture design to interpret Spatial Canvas Interface and their associated Text Specifications . Our interface supports four comp...

</details>

<details>
<summary><b>20. OneSearch-VL: Unified Multimodal Deep Research Agent for Image and Video</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Dian Zheng, Shu Chen, Kaituo Feng, Manyuan Zhang, Hongyu Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12419) • [📄 arXiv](https://arxiv.org/abs/2610.12419) • [📥 PDF](https://arxiv.org/pdf/2610.12419)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/appletea233/OneSearch-VL)

> code: https://github.com/appletea233/OneSearch-VL data: https://huggingface.co/OneSearch-VL

</details>

<details>
<summary><b>21. SparseEngine: Sparse-First Inference Engine</b> ⭐ 79</summary>

<br/>

**👥 Authors:** Jun Yu, Qiang Huang, Quansheng Gu, Jitai Hao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39068) • [📄 arXiv](https://arxiv.org/abs/2609.39068) • [📥 PDF](https://arxiv.org/pdf/2609.39068)

**💻 Code:** [⭐ Code](https://github.com/CURRENTF/SparseEngine) • [⭐ Code](https://github.com/huggingface)

> Long-context LLM agents accumulate interaction histories that strain KV-cache memory and attention computation. Although sparse attention reduces these costs, heterogeneous cache representations and workflows hinder integration with existing infer...

</details>

<details>
<summary><b>22. SanSi: A Looped Typed Decision Model for System 1.5 Thinking</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.07730) • [📄 arXiv](https://arxiv.org/abs/2610.07730) • [📥 PDF](https://arxiv.org/pdf/2610.07730)

**💻 Code:** [⭐ Code](https://github.com/minnesotanlp/Sansi) • [⭐ Code](https://github.com/huggingface)

> TL;DR: Our work is a looped typed decision model that returns a probability for each declared option and can be read after any of its eight loops, so one model serves every compute budget without generating reasoning tokens. With the same data and...

</details>

<details>
<summary><b>23. Chaos in the Text: Revealing the Modality Preference in Mixed-Modality Retrievers</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zhipeng Xu, Zhenghao Liu, Yukun Yan, Chunyi Peng, Yubo Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11816) • [📄 arXiv](https://arxiv.org/abs/2610.11816) • [📥 PDF](https://arxiv.org/pdf/2610.11816)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenBMB/Trident)

> Dense retrievers have made significant progress on text and image corpora, but whether these capabilities extend reliably to mixed corpora containing text, image, and fused text-image documents remains unclear. In this paper, we systematically exa...

</details>

<details>
<summary><b>24. V-CoLA: Vision Token Compression with Linear Attention</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11251) • [📄 arXiv](https://arxiv.org/abs/2610.11251) • [📥 PDF](https://arxiv.org/pdf/2610.11251)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Vision-language models (VLMs) have demonstrated impressive capabilities but suffer from substantial computational overhead, as vision tokens dominate the input sequence. This motivates vision token compression as a key direction to alleviate the b...

</details>

<details>
<summary><b>25. SpatialOPSD: Self-Distilling Spatial Intelligence from Verified Coding Agent Traces</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11366) • [📄 arXiv](https://arxiv.org/abs/2610.11366) • [📥 PDF](https://arxiv.org/pdf/2610.11366)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> With the widespread adoption of coding agents, we explore an approach to distilling the spatial reasoning capabilities of coding agents into standalone MLLMs through on-policy self-distillation (OPSD).

</details>

<details>
<summary><b>26. Accurate but Not Humble: Evaluating Epistemic Humility in LLM Agents under Knowledge Conflict</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12360) • [📄 arXiv](https://arxiv.org/abs/2610.12360) • [📥 PDF](https://arxiv.org/pdf/2610.12360)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/KaiserWhoLearns/EpistemicHumilityLLMAgents)

> When retrieved evidence contradicts an agent's prior beliefs, does it revise its answer, acknowledge uncertainty, or persist with an incorrect conclusion? Existing evaluations of agentic systems focus primarily on task success, offering limited in...

</details>

<details>
<summary><b>27. SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12402) • [📄 arXiv](https://arxiv.org/abs/2610.12402) • [📥 PDF](https://arxiv.org/pdf/2610.12402)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/SpaceCast-Bench)

> This paper introduces SpaceCast-Bench, a benchmark for predictive spatial reasoning that evaluates whether vision-language models can infer unseen outcomes of spatial transformations, with 3,862 questions across 16 task types covering static perce...

</details>

<details>
<summary><b>28. ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12403) • [📄 arXiv](https://arxiv.org/abs/2610.12403) • [📥 PDF](https://arxiv.org/pdf/2610.12403)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/ViSkill) • [⭐ Code](https://github.com/huggingface)

> This paper proposes ViSkill, a visual-native skill learning framework that represents successful interactions as reusable visual skill cards and couples geometry-aware retrieval, skill-guided reinforcement learning, and online skill distillation i...

</details>

<details>
<summary><b>29. Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Lu Yin, Guinan Su, Di He, Dilxat Muhtar, Pengxiang Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.11570) • [📄 arXiv](https://arxiv.org/abs/2610.11570) • [📥 PDF](https://arxiv.org/pdf/2610.11570)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/pixeli99/InfiLoop)

> 1/ More loops ≠ better reasoning. On Sudoku-Extreme, TRM peaks at 62% and then drops as you keep looping. Puzzles it already solved get broken again. 2/ The culprit is “carry-last”: every iteration just overwrites the state with the latest output....

</details>

<details>
<summary><b>30. Reasoning-Informed Visual Editing</b> ⭐ 155</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12343) • [📄 arXiv](https://arxiv.org/abs/2610.12343) • [📥 PDF](https://arxiv.org/pdf/2610.12343)

**💻 Code:** [⭐ Code](https://github.com/VisionXLab/RISEBench) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>31. Retrieval-Centric Deep Learning in Growing Nonparametric Neural Networks</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.03858) • [📄 arXiv](https://arxiv.org/abs/2610.03858) • [📥 PDF](https://arxiv.org/pdf/2610.03858)

**💻 Code:** [⭐ Code](https://github.com/paradigms-of-intelligence/rcdl) • [⭐ Code](https://github.com/huggingface)

> TLDR: A well-known duality connects training of linear layers by gradient descent with linear attention over training examples. This paper builds on this insight and proposes retrieval-centric deep learning which replaces the standard linear atten...

</details>

<details>
<summary><b>32. OmniCapBench: A Deep-Structured Evaluation Framework for Fine-Grained Audio-Visual Captioning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12458) • [📄 arXiv](https://arxiv.org/abs/2610.12458) • [📥 PDF](https://arxiv.org/pdf/2610.12458)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We propose OmniCapBench, a benchmark that reformulates audio-video captioning as structured prediction, enabling deterministic constraint checks and local semantic judging for reliable, fine-grained diagnosis of model failures.

</details>

<details>
<summary><b>33. Embodied Turing Machines: Stateful Code for Robot Recursive Self-Improvement</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ziwei Liu, Zhaoxi Chen, Fangzhou Hong, Siyuan Hu, Kairui Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12369) • [📄 arXiv](https://arxiv.org/abs/2610.12369) • [📥 PDF](https://arxiv.org/pdf/2610.12369)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Most robot policies keep a model in the control loop: a VLA maps observations to actions, and an Agent Harness, such as Agent-as-Policy or Harness VLA queries a VLM for decision making at run time. We propose a different view: the embodied world i...

</details>

<details>
<summary><b>34. Distilling Routed 3D Privilege for Spatial Reasoning in Vision-Language Models</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.12355) • [📄 arXiv](https://arxiv.org/abs/2610.12355) • [📥 PDF](https://arxiv.org/pdf/2610.12355)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/GPD) • [⭐ Code](https://github.com/huggingface)

> This paper proposes GPD, a geometry-privileged distillation framework that routes question-relevant 3D cues and reference answers to a teacher during on-policy self-distillation, augmenting GRPO with distillation on incorrect trajectories while ke...

</details>

<details>
<summary><b>35. TerraVis: Towards Evaluation of World-Grounded Visual Consistency in Text-to-Image Generation via MLLM Workflows</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02959) • [📄 arXiv](https://arxiv.org/abs/2610.02959) • [📥 PDF](https://arxiv.org/pdf/2610.02959)

**💻 Code:** [⭐ Code](https://github.com/ShyFoo/TerraVis) • [⭐ Code](https://github.com/huggingface)

> Excited to share TerraVis , accepted to NeurIPS 2026 (Evaluations & Datasets Track)! 🎉🎉🎉 While existing T2I metrics primarily focus on prompt alignment and visual quality, TerraVis addresses a long-standing gap in text-to-image evaluation: world c...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 35 |
| 📅 Today | [`2026-10-09.json`](data/daily/2026-10-09.json) | 35 |
| 📆 This Week | [`2026-W40.json`](data/weekly/2026-W40.json) | 138 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 375 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-09 | 35 | [View JSON](data/daily/2026-10-09.json) |
| 📄 2026-10-08 | 40 | [View JSON](data/daily/2026-10-08.json) |
| 📄 2026-10-07 | 23 | [View JSON](data/daily/2026-10-07.json) |
| 📄 2026-10-06 | 21 | [View JSON](data/daily/2026-10-06.json) |
| 📄 2026-10-05 | 19 | [View JSON](data/daily/2026-10-05.json) |
| 📄 2026-10-04 | 84 | [View JSON](data/daily/2026-10-04.json) |
| 📄 2026-10-03 | 84 | [View JSON](data/daily/2026-10-03.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W40 | 138 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 375 | [View JSON](data/monthly/2026-10.json) |
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
