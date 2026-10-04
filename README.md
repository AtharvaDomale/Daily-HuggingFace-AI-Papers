<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-84-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6845+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">84</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">326</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">237</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6845+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 04, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. OneStreamer: Unifying Perception, Memory, and Proactive Response in Streaming Video Interaction</b> ⭐ 86</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01762) • [📄 arXiv](https://arxiv.org/abs/2610.01762) • [📥 PDF](https://arxiv.org/pdf/2610.01762)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MCG-NJU/OneStreamer)

> Hi everyone! We’re sharing OneStreamer , a 4B model that unifies perception, memory, and proactive responses for streaming video interaction. The core idea is to learn both what to remember and when to respond . OneStreamer records evidence as vid...

</details>

<details>
<summary><b>2. On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35259) • [📄 arXiv](https://arxiv.org/abs/2609.35259) • [📥 PDF](https://arxiv.org/pdf/2609.35259)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Curious to hear your thoughts and comments about the differences between on- and off-policy learning in the context of strong-to-weak distillation!

</details>

<details>
<summary><b>3. GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38923) • [📄 arXiv](https://arxiv.org/abs/2609.38923) • [📥 PDF](https://arxiv.org/pdf/2609.38923)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The data and models are available at https://huggingface.co/collections/groundhogLLM/graphforge

</details>

<details>
<summary><b>4. Adaptive Reward Routing: Dynamic Multi-Reward Optimization for Joint Audio-Video Diffusion via Forward-Process RL</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37200) • [📄 arXiv](https://arxiv.org/abs/2609.37200) • [📥 PDF](https://arxiv.org/pdf/2609.37200)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Multi-reward guided reinforcement learning (i.e., RL) offers a promising way to improve joint audio-video diffusion models along several complementary objectives, including modality-specific quality, cross-modal semantic alignment, and temporal sy...

</details>

<details>
<summary><b>5. Hierarchical Continuous Diffusion Language Models</b> ⭐ 54</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02193) • [📄 arXiv](https://arxiv.org/abs/2610.02193) • [📥 PDF](https://arxiv.org/pdf/2610.02193)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/rhfeiyang/HC-DLM)

> HC-DLM couples discrete tokens with a persistent continuous latent state, preserving token dependencies during parallel denoising and improving reasoning and language modeling over discrete and continuous diffusion baselines.

</details>

<details>
<summary><b>6. Sharpening Tax in Post-Training</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01509) • [📄 arXiv](https://arxiv.org/abs/2610.01509) • [📥 PDF](https://arxiv.org/pdf/2610.01509)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/changdaeoh/sharpening-tax)

> No abstract available.

</details>

<details>
<summary><b>7. Beyond Memory: Harnessing Long-Horizon Agents with Explicit Belief States</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01415) • [📄 arXiv](https://arxiv.org/abs/2610.01415) • [📥 PDF](https://arxiv.org/pdf/2610.01415)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/luoyu100/PoS)

> Long-horizon agents need to keep track of the current world state to guide their decisions. Yet maintaining a coherent belief alone does not ensure progress: agents can keep acting without meaningfully advancing their goals, a failure mode this pa...

</details>

<details>
<summary><b>8. Agent Priors-guided Policy Learning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35690) • [📄 arXiv](https://arxiv.org/abs/2609.35690) • [📥 PDF](https://arxiv.org/pdf/2609.35690)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Agentics-robotics/Agent-Priors-guided-Policy-Learning)

> No abstract available.

</details>

<details>
<summary><b>9. World Observer: Joint Actor-Observer Generation for Persistent World Modeling</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02162) • [📄 arXiv](https://arxiv.org/abs/2610.02162) • [📥 PDF](https://arxiv.org/pdf/2610.02162)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cvlab-kaist/world-observer)

> How can a world model continuously observe regions beyond the actor's current view? Video world models simulate how an environment evolves from an agent's actions, yet remain actor-centric. Once an object leaves the actor's view, they lose direct ...

</details>

<details>
<summary><b>10. A Missing Piece for Trustworthy AI Reviewers: From Benchmarking Rhetorical Robustness to SciCore Review</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jianpeng Chen, Chengrui Fan, Ming Li, Chenguang Wang, zhoutianyi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39027) • [📄 arXiv](https://arxiv.org/abs/2609.39027) • [📥 PDF](https://arxiv.org/pdf/2609.39027)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> AI reviewers can assign different judgments to manuscripts that report the same science in different wording, potentially rewarding rhetorical optimization over scientific improvement. We formulate Rhetorical Robustness as the joint requirement of...

</details>

<details>
<summary><b>11. Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Junran Wang, Ruixuan Deng, Zehao Jin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36585) • [📄 arXiv](https://arxiv.org/abs/2609.36585) • [📥 PDF](https://arxiv.org/pdf/2609.36585)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Lunamos/stop-thinking-too-early)

> TL;DR: Asked to answer directly (no CoT), LLMs follow reference chains like K = apple; B = K; D = B; print(D) for only a few lines. A tiny rank-8 LoRA at one early layer lets the frozen middle layers carry the chain much further. The computation i...

</details>

<details>
<summary><b>12. ROWBench: Do Video Models Render What the Program Specifies?</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02205) • [📄 arXiv](https://arxiv.org/abs/2610.02205) • [📥 PDF](https://arxiv.org/pdf/2610.02205)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AlayaLab/PROWBench)

> A program runs the world; a video model renders it. PROWBench asks whether video models render what the program specifies — the right action, the right outcome, at the right time. 170 test cases, 200 camera views and 600 proxy inputs (Coarse 3D, C...

</details>

<details>
<summary><b>13. E-MoE: Enhanced Mixture-of-Experts for Non-Factorized Diffusion Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mikhail Goncharov, Ivan Oseledets, Alexander Korotin, Alexander Kolesov, ArsenyIvanov

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37533) • [📄 arXiv](https://arxiv.org/abs/2609.37533) • [📥 PDF](https://arxiv.org/pdf/2609.37533)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Masked diffusion language models generate multiple tokens in parallel, but their reverse process is typically factorized across positions, which limits generation quality in the few-step regime. We introduce E-MoE, which turns a Mixture-of-Experts...

</details>

<details>
<summary><b>14. ActiveSaddler: Automated Curriculum Learning for Agent Harness Optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00906) • [📄 arXiv](https://arxiv.org/abs/2610.00906) • [📥 PDF](https://arxiv.org/pdf/2610.00906)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/AutoSaddler/tree/feat/activesaddler)

> https://github.com/microsoft/AutoSaddler/tree/feat/activesaddler

</details>

<details>
<summary><b>15. Make Sparse Rewards Count: Density-Aware Reward Aggregation for Multi-Reward RL</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00574) • [📄 arXiv](https://arxiv.org/abs/2610.00574) • [📥 PDF](https://arxiv.org/pdf/2610.00574)

**💻 Code:** [⭐ Code](https://github.com/zhaihaotian/DARA) • [⭐ Code](https://github.com/huggingface)

> Why do some rewards get learned much more slowly in multi-reward RL, even with GDPO? We show the root cause is advantage energy: under GDPO, a reward's energy scales with how often it is active in a batch, so sparse rewards are drowned out. DARA f...

</details>

<details>
<summary><b>16. AutoGUIWorld: Image Generators as Visual World Models for GUI Agent</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01215) • [📄 arXiv](https://arxiv.org/abs/2610.01215) • [📥 PDF](https://arxiv.org/pdf/2610.01215)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ImYangC7/AutoGUIWorld)

> Image Generators as Visual World Models for GUI Agent github: https://github.com/ImYangC7/AutoGUIWorld full paper: https://huggingface.co/YangC777/AGW-35B/blob/main/AutoGUIWorld_Report.pdf

</details>

<details>
<summary><b>17. Retrieval-Augmented Skill Optimization via Cross-Harness Adaptation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jeehye Na, Dohwan Ko, Jihwan Park, simplecloud, allonsy07

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38024) • [📄 arXiv](https://arxiv.org/abs/2609.38024) • [📥 PDF](https://arxiv.org/pdf/2609.38024)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Agent skills are becoming valuable repositories of reusable procedural knowledge, yet most skill optimization methods still learn each skill largely from scratch through costly agent rollouts. We introduce RASO, a framework that retrieves relevant...

</details>

<details>
<summary><b>18. EgoTools: Towards Tool-Centric Reasoning in Real-World Egocentric Videos</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39378) • [📄 arXiv](https://arxiv.org/abs/2609.39378) • [📥 PDF](https://arxiv.org/pdf/2609.39378)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Ropedia/EgoTools)

> Proj page: https://ropedia.github.io/egotools/ Code: https://github.com/Ropedia/EgoTools Data: https://huggingface.co/datasets/ropedia-ai/egotools-data

</details>

<details>
<summary><b>19. Video Generation Models: A Survey of Post-Training and Alignment</b> ⭐ 196</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00812) • [📄 arXiv](https://arxiv.org/abs/2610.00812) • [📥 PDF](https://arxiv.org/pdf/2610.00812)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/people-robots/Awesome-Video-Generation-Post-Training)

> How can we make video generation models more controllable, temporally consistent, and aligned with human intent? Our survey, published in TMLR , provides a unified guide to post-training and alignment for video generation , covering supervised fin...

</details>

<details>
<summary><b>20. InterEvolve: Test-Time Evolution of Reward Programs for Humanoid Loco-Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02196) • [📄 arXiv](https://arxiv.org/abs/2610.02196) • [📥 PDF](https://arxiv.org/pdf/2610.02196)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>21. X-Tree: Tokenizing Reusable Experience for Efficient Agent Generalization</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32993) • [📄 arXiv](https://arxiv.org/abs/2609.32993) • [📥 PDF](https://arxiv.org/pdf/2609.32993)

**💻 Code:** [⭐ Code](https://github.com/sitaocheng/X-Tree) • [⭐ Code](https://github.com/huggingface)

> Instead of training agents on flat action sequences, we mine the reusable hierarchy latent in existing trajectories, with zero LLM calls, and use it to guide training as data, as reward and as context. Takeaways Agent trajectories hide a reusable ...

</details>

<details>
<summary><b>22. Decentralized Master-Mind: Joint Action Refinement through Iterative Intent Denoising in Multi-Agent Pathfinding</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Konstantin Yakovlev, Taisia Zlotnikova, Anton Andreychuk, Valeriy Vyaltsev, tviskaron

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32019) • [📄 arXiv](https://arxiv.org/abs/2609.32019) • [📥 PDF](https://arxiv.org/pdf/2609.32019)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/CognitiveAISystems/DMM)

> We found a serious problem in learned multi-agent pathfinding (MAPF). Even when agents learn to communicate, each one still picks its final action independently, and we call this the decentralized factorization gap. When several joint solutions ar...

</details>

<details>
<summary><b>23. Scaling and Distilling Text Embeddings for Better Diffusibility</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01016) • [📄 arXiv](https://arxiv.org/abs/2605.10938) • [📥 PDF](https://arxiv.org/pdf/2610.01016)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/la0ka1/diffusing-scaled-text-embeddings)

> Glad to share our recent paper on the latent space for continuous diffusion language models (DLMs)! In this paper, we find that scaling text embeddings can greatly boost the performance of continuous DLMs; for instance, by replacing the T5-small e...

</details>

<details>
<summary><b>24. Persona Dosing: Calibrated Activation Steering for Graded Trait Control</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jingyuan Zhang, Jiahao Chen, Ruixuan Deng, Junran Wang, Zehao Jin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36388) • [📄 arXiv](https://arxiv.org/abs/2609.36388) • [📥 PDF](https://arxiv.org/pdf/2609.36388)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: A steering coefficient sets how hard you push on the activations, not how much of a trait you actually get. PersonaDose lets you ask for a persona by description and a target intensity, by calibrating a FLAS controller's flow time against m...

</details>

<details>
<summary><b>25. Beyond the Current Scene: Event-Referential Grasping with Active View Selection</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39375) • [📄 arXiv](https://arxiv.org/abs/2609.39375) • [📥 PDF](https://arxiv.org/pdf/2609.39375)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SNU-VGILab/BeyondCSe)

> BeyondCSe enables event-referential robotic grasping, allowing robots to identify and grasp objects referenced by past events even when they are no longer visible, through video reasoning and event-conditioned active view selection.

</details>

<details>
<summary><b>26. Fewer Tokens, Better Action: GPT-6 Astra Robot Agents with 14% Higher Success Rate but 65% Fewer Tokens</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01939) • [📄 arXiv](https://arxiv.org/abs/2610.01939) • [📥 PDF](https://arxiv.org/pdf/2610.01939)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DAGroup-PKU/PyRUA-Lean)

> Vision language model (VLM) agents can control robots through visual feedback and action primitives, but repeated model invocations and redundant observations incur substantial token overhead. We introduce PyRUA-Lean, an interactive code-execution...

</details>

<details>
<summary><b>27. SILSA: Sliding-Window Slice Latents for Topology-Preserving High-Resolution 3D Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02201) • [📄 arXiv](https://arxiv.org/abs/2610.02201) • [📥 PDF](https://arxiv.org/pdf/2610.02201)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> SILSA generates high-resolution 3D shapes from a single image using 384 sliding-window slice tokens, preserving thin structures, holes, repeated parts, and long-range connectivity.

</details>

<details>
<summary><b>28. 4Director: Controlling Video World Models with Rigid 3D Geometry</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02160) • [📄 arXiv](https://arxiv.org/abs/2610.02160) • [📥 PDF](https://arxiv.org/pdf/2610.02160)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/VVeiCao/4Director)

> 4Director turns an image into an editable 3D scene. Every object becomes a complete mesh, placed in the same 3D space as the background and the camera. You place the camera, draw a rigid trajectory for each object, and can bring in new objects fro...

</details>

<details>
<summary><b>29. Decoding Looped Transformers Better for (Almost) Free</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02185) • [📄 arXiv](https://arxiv.org/abs/2610.02185) • [📥 PDF](https://arxiv.org/pdf/2610.02185)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> LoopCD turns a looped Transformer's earlier recurrent state into a built-in weak reference for contrastive decoding, improving accuracy across four looped model families without training and matching full-depth accuracy at half the recurrent itera...

</details>

<details>
<summary><b>30. Architect-Ant: Editable Automatic Furnishing of Architectural Floor Plans</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.10953) • [📄 arXiv](https://arxiv.org/abs/2606.10953) • [📥 PDF](https://arxiv.org/pdf/2606.10953)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce a new approach to automatic furniture layout generation for real architectural floor plans that combines a curated real-world dataset with rule-guided reinforcement learning. We release AntPlan, 505 professional floor plans with dense...

</details>

<details>
<summary><b>31. AutoDataBench: A Data-centric Testbed for Accelerating Auto Research</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40097) • [📄 arXiv](https://arxiv.org/abs/2609.40097) • [📥 PDF](https://arxiv.org/pdf/2609.40097)

**💻 Code:** [⭐ Code](https://github.com/AutoDataBench/AutoDataBench) • [⭐ Code](https://github.com/huggingface)

> Existing auto-research benchmarks often entangle multiple sources of improvement, including training frameworks, hyperparameters, compute budgets, and data, making it difficult to attribute why one frontier agent outperforms another to specific re...

</details>

<details>
<summary><b>32. Multimodal Flow: Unified Flow Modeling of Language and Vision in Embedding Spaces</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40362) • [📄 arXiv](https://arxiv.org/abs/2609.40362) • [📥 PDF](https://arxiv.org/pdf/2609.40362)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hustvl/Multimodal-Flow)

> Multimodal Flow introduces a fully continuous paradigm for unified multimodal modeling, representing both language and vision as continuous hyperchunks and modeling them with a shared chunk-causal Flow Matching backbone.

</details>

<details>
<summary><b>33. CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36820) • [📄 arXiv](https://arxiv.org/abs/2609.36820) • [📥 PDF](https://arxiv.org/pdf/2609.36820)

**💻 Code:** [⭐ Code](https://github.com/HKUST-KnowComp/CorrGRPO) • [⭐ Code](https://github.com/huggingface)

> CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning.

</details>

<details>
<summary><b>34. Ego2Act: Evaluating Goal-Directed Manipulation in Egocentric Video Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01092) • [📄 arXiv](https://arxiv.org/abs/2610.01092) • [📥 PDF](https://arxiv.org/pdf/2610.01092)

**💻 Code:** [⭐ Code](https://github.com/ego2act/ego2act) • [⭐ Code](https://github.com/huggingface)

> Video models can make anything look real. Can they actually do the task? Ego2Act gives a video model one real egocentric start frame and one everyday goal (e.g. "the round yellow paper is stapled to the white A4 paper" ) and asks it to generate th...

</details>

<details>
<summary><b>35. Argo-Bench: Evaluating Data Agents on Enterprise-Scale Workflows</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02122) • [📄 arXiv](https://arxiv.org/abs/2610.02122) • [📥 PDF](https://arxiv.org/pdf/2610.02122)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/TextQLLabs/Argo-Bench)

> Hi everyone, first author here! We realized most established text-to-SQL benchmarks don't yet cover the most important care most companies care about: the agents' ability to make good decisions. While some benchmarks have been pretty good for this...

</details>

<details>
<summary><b>36. RPTune: Learned Context Curation for LLM Catalog Search</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00964) • [📄 arXiv](https://arxiv.org/abs/2610.00964) • [📥 PDF](https://arxiv.org/pdf/2610.00964)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> For small merchant businesses (SMBs) whose catalogs fit within a long-context LLM, full-catalog prompting offers a compelling alternative to multi-stage retrieval designed primarily for large marketplaces with millions of items. However, fitting t...

</details>

<details>
<summary><b>37. Omni-Embed-Mini: Binding Modalities Without Forgetting via Dense Distillation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02148) • [📄 arXiv](https://arxiv.org/abs/2610.02148) • [📥 PDF](https://arxiv.org/pdf/2610.02148)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/k-m-irfan/Omni-Embed-Mini)

> Omni-Embed-Mini is a compact omni-modal embedding model that puts text, speech, general audio, images, video and visually-rich documents into one shared embedding space. We pair each media sample with a dense caption and train the media path to ma...

</details>

<details>
<summary><b>38. PhysVista: Benchmarking Physical Intelligence in VLMs via a Perception-Reasoning-Assessment Loop</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00559) • [📄 arXiv](https://arxiv.org/abs/2610.00559) • [📥 PDF](https://arxiv.org/pdf/2610.00559)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Helen1p/PhysVista)

> A comprehensive benchmark for physical intelligence evaluation in VLMs. Check our data here: 🤗Data

</details>

<details>
<summary><b>39. PixelDense: Dense Prediction as Representation Alignment for Pixel Diffusion</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Yiqing Yang, Avery Li, Wenhao Zhang, Daiqing Qi, xsourse

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00483) • [📄 arXiv](https://arxiv.org/abs/2610.00483) • [📥 PDF](https://arxiv.org/pdf/2610.00483)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Hansxsourse/PixelDense)

> NeurIPS 2026

</details>

<details>
<summary><b>40. Smaller Models, Better Rejects: Preference Distillation Scaling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38987) • [📄 arXiv](https://arxiv.org/abs/2609.38987) • [📥 PDF](https://arxiv.org/pdf/2609.38987)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We study how the source of rejected responses affects preference distillation when the chosen responses and training setup are held fixed. Across 7B to 72B students, smaller frozen models provide rejects that are cheaper to generate and lead to st...

</details>

<details>
<summary><b>41. Where-OPD: Spatially Guided On-Policy Self-Distillation of MLLMs with Synthetic Scenes</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02117) • [📄 arXiv](https://arxiv.org/abs/2610.02117) • [📥 PDF](https://arxiv.org/pdf/2610.02117)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sirkosophia/Where-OPD)

> On-policy self-distillation has recently emerged as an effective approach for improving language-model reasoning by supervising students with a frozen or EMA version of themselves that receives privileged information. Its application to multimodal...

</details>

<details>
<summary><b>42. Better Supervision Is Nearby: Neighborhood On-Policy Self-Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.39687) • [📄 arXiv](https://arxiv.org/abs/2609.39687) • [📥 PDF](https://arxiv.org/pdf/2609.39687)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi everyone! We’re excited to introduce our new work, N-OPSD. If the same model sees the same reference solution, is the help it can offer a student always the same? In on-policy self-distillation (OPSD), the student solves problems on its own, wh...

</details>

<details>
<summary><b>43. When Users Change Their Minds: Measuring and Repairing Intent Drift in LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yushi Sun, Zixin Chen, Bowen Cao, doudouwer

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32520) • [📄 arXiv](https://arxiv.org/abs/2609.32520) • [📥 PDF](https://arxiv.org/pdf/2609.32520)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce IntentFlux, an executable benchmark that converts verifiable tasks into dialogues with controlled intent changes while preserving their original graders. In a 627-case calibration, mean task score falls from 0.476 to 0.384 as dialogue...

</details>

<details>
<summary><b>44. OmniSeek: Native Tool Integration for Multi-turn Audio-Visual Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuanjun Xiong, Jingru Yi, Jialu Li, Jiteng Mu, WHB139426

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02181) • [📄 arXiv](https://arxiv.org/abs/2610.02181) • [📥 PDF](https://arxiv.org/pdf/2610.02181)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> code, weights and data are under Adobe’s internal review

</details>

<details>
<summary><b>45. Pretrain Once, Route Anywhere: Towards a Foundation Model for LLM Routing</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37362) • [📄 arXiv](https://arxiv.org/abs/2609.37362) • [📥 PDF](https://arxiv.org/pdf/2609.37362)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LAMDA-Model-Reuse/RouteFM)

> Most LLM routers are trained for a fixed workload and candidate pool, requiring retraining as the routing environment changes. RouteFM explores a different paradigm: pretrain the routing capability once and adapt to new environments through contex...

</details>

<details>
<summary><b>46. OpenTumorBoard: A Real-World Benchmark of Multidisciplinary Tumor Board Discussion Trajectories</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Chi-Yu Chen, Jiarong Qian, Yixuan Duan, Zhixuan Ge, al1219

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32810) • [📄 arXiv](https://arxiv.org/abs/2609.32810) • [📥 PDF](https://arxiv.org/pdf/2609.32810)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AnqiLi24/OpenTumorBoard)

> OpenTumorBoard is a benchmark built from 219 publicly recorded multidisciplinary tumor board meetings: 611 patient cases, 19,157 discussion turns across ten specialist roles, and 16,215 questions put to specialists during the meetings. Two setting...

</details>

<details>
<summary><b>47. AgSpec: Pushing the Limits of Retrieval-Based Speculative Decoding in Coding Agent Pipelines</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Youngjin Kwon, Suengjae Lim, Sukmin Cho, Sumin Lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01108) • [📄 arXiv](https://arxiv.org/abs/2610.01108) • [📥 PDF](https://arxiv.org/pdf/2610.01108)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We proposed a retrieval-based speculative decoding framework, AgSpec, for coding agent pipelines. AgSpec provides the proper policy for a retrieval engine, such as corpus design and draft-length decisions. We achieve about 4 times higher throughpu...

</details>

<details>
<summary><b>48. Benchmarking and Enhancing Skill-Level Memory for Partially Observable Robotic Manipulation</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Yuhan Zhu, Shaowei Zhang, Xijie Yang, Jiange Yang, nanamma

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38886) • [📄 arXiv](https://arxiv.org/abs/2609.38886) • [📥 PDF](https://arxiv.org/pdf/2609.38886)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/nanamma/HIDE-SEEK)

> No abstract available.

</details>

<details>
<summary><b>49. Do Audio LLMs Listen Before They Act? Diagnosing Acoustic-Context Gating in Voice Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yushi Sun, Nanchen Hu, doudouwer

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32536) • [📄 arXiv](https://arxiv.org/abs/2609.32536) • [📥 PDF](https://arxiv.org/pdf/2609.32536)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce VGBench, a 1,018-item diagnostic benchmark for action-level addressedness across side-talk, self-talk, and speaker-switch scenarios.

</details>

<details>
<summary><b>50. Align Then Reason: A Multimodal Lip-Sync Judge for Dubbing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00825) • [📄 arXiv](https://arxiv.org/abs/2610.00825) • [📥 PDF](https://arxiv.org/pdf/2610.00825)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Dubbing quality control requires a reference-free judge that can determine whether a candidate text line matches a speaker's visible articulation in both content and timing, using only silent video and text because dubbed audio may not yet exist. ...

</details>

<details>
<summary><b>51. RLE-Bench: A Qualifying Exam for Coding Agents as Robot Learning Engineers</b> ⭐ 67</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34210) • [📄 arXiv](https://arxiv.org/abs/2609.34210) • [📥 PDF](https://arxiv.org/pdf/2609.34210)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RLE-Bench/RLE-Bench)

> RLE-Bench is a benchmark evaluating coding agents on general robot learning tasks, inlcuding interactive control, policy learning, perception & estimation, and mechanical design.

</details>

<details>
<summary><b>52. SemanTok: Predictable Semantic Tokens for Efficient Autoregressive Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00686) • [📄 arXiv](https://arxiv.org/abs/2610.00686) • [📥 PDF](https://arxiv.org/pdf/2610.00686)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Stability-AI/SemanTok)

> Flexible video tokenizers (e.g., VideoFlexTok) let an autoregressive (AR) model stop after any number of tokens, which condition a diffusion decoder, so the first tokens should already capture what the clip shows. SemanTok supervises this explicit...

</details>

<details>
<summary><b>53. LOCI: Spatial Linear Memory for Streaming World Models</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.40222) • [📄 arXiv](https://arxiv.org/abs/2609.40222) • [📥 PDF](https://arxiv.org/pdf/2609.40222)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/xiaji2021/LOCI)

> When a camera turns away and comes back, most video world models forget what was there. LOCI keeps both kinds of memory: half of the blocks retain a KV cache of past observations, the other half add a recurrent linear-attention memory whose reads ...

</details>

<details>
<summary><b>54. Removing the NEEDLE in the Haystack: Backdoor Removal in LLMs via Weight Orthogonalisation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00348) • [📄 arXiv](https://arxiv.org/abs/2610.00348) • [📥 PDF](https://arxiv.org/pdf/2610.00348)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LocaiLabs/NEEDLE) • [⭐ Code](https://github.com/LocaiLabs/NEEDLE/)

> NEEDLE is a training-free backdoor defence designed to remove a backdoor with minimal changes to model behaviour and safety. It estimates a backdoor direction (how the trigger shifts the model's activations) and a refusal subspace (directions that...

</details>

<details>
<summary><b>55. VTR-Bench: A Systematic Benchmark for Evaluating Visual Text Rendering in Video Generation</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Song Dai, Yonghua Hei, Zhiyuan Wang, Yu Huang, Jungang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01499) • [📄 arXiv](https://arxiv.org/abs/2610.01499) • [📥 PDF](https://arxiv.org/pdf/2610.01499)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hardenyu21/VTR-Bench)

> No abstract available.

</details>

<details>
<summary><b>56. DataMagic: Authoring Data Videos through Declarative Multi-Agent Orchestration</b> ⭐ 276</summary>

<br/>

**👥 Authors:** Zhouan Shen, Jiayi Zhu, Liangwei Wang, Zhenyang Wang, Yupeng Xie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33403) • [📄 arXiv](https://arxiv.org/abs/2609.33403) • [📥 PDF](https://arxiv.org/pdf/2609.33403)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HKUSTDial/DataMagic)

> Data videos communicate data insights through dynamic charts, voice narration, and synchronized animations, and have become a widely adopted form of data storytelling. However, their production requires multidisciplinary expertise spanning data an...

</details>

<details>
<summary><b>57. Memorizon: Training World Models Beyond Their Context Window</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00544) • [📄 arXiv](https://arxiv.org/abs/2610.00544) • [📥 PDF](https://arxiv.org/pdf/2610.00544)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/TingtingLiao/memorizon)

> Memorizon trains a camera-controlled video world model on spans far longer than its context window. Each training sample packs up to 100-400s of history into a fixed-size sequence: every chunk retrieves the past frames whose camera frustums overla...

</details>

<details>
<summary><b>58. ScholarCatalyst: A Benchmark for Retrieving Papers That Inspire New Research</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rulin Shao, Dayoon Ko, Yoonho Lee, Benjamin-eecs, ohmyksh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02202) • [📄 arXiv](https://arxiv.org/abs/2610.02202) • [📥 PDF](https://arxiv.org/pdf/2610.02202)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/stanford-iris-lab/ScholarCatalyst)

> ScholarCatalyst is a benchmark for retrieving "catalyst papers," the earlier work that did or could have helped a research project. Labels come from 184 lead authors looking back at 207 of their own recent projects (about 1000 queries, with author...

</details>

<details>
<summary><b>59. Generalization Is Stability, Not Accuracy: Multi-Axis Evaluation of LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01428) • [📄 arXiv](https://arxiv.org/abs/2610.01428) • [📥 PDF](https://arxiv.org/pdf/2610.01428)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Accepted at the TAE (Trust-AI-Eval) Workshop: Can We Trust AI Evaluation?, NeurIPS 2026

</details>

<details>
<summary><b>60. Devils in Question Relay: Source-Conditioned Relay Steering to Mitigate Hallucinations in Audio-visual Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yang Xiang, Pengfei Zhang, Xuefeng Bai, Pingrui Zhang, Yu Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37568) • [📄 arXiv](https://arxiv.org/abs/2609.37568) • [📥 PDF](https://arxiv.org/pdf/2609.37568)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Audio-visual large language models (AVLLMs) have made remarkable progress in multimodal understanding and reasoning through interactions among visual, auditory, and linguistic information. However, recent studies show that AVLLMs face a critical c...

</details>

<details>
<summary><b>61. KaliBench: A Fine-Grained Benchmark for Cybersecurity Tool Use on Kali Linux with Runtime-Free Verifiable Rewards</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02206) • [📄 arXiv](https://arxiv.org/abs/2610.02206) • [📥 PDF](https://arxiv.org/pdf/2610.02206)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RISys-Lab/KaliBench)

> KaliBench: A Fine-Grained Benchmark for Cybersecurity Tool Use on Kali Linux with Runtime-Free Verifiable Rewards (NeurIPS 2026 Evaluations and Datasets Track)

</details>

<details>
<summary><b>62. Prompt2Skill: Unsupervised Skill Optimization From Natural Language Instructions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tyler Derr, Ryan A. Rossi, Li Li, Bo Ni, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38593) • [📄 arXiv](https://arxiv.org/abs/2609.38593) • [📥 PDF](https://arxiv.org/pdf/2609.38593)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Retrieval-Augmented Skill Optimization via Cross-Harness Adaptation (2026) ...

</details>

<details>
<summary><b>63. Rules to Tools: Executable Checks for LLM Agents in Scientific Computing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00313) • [📄 arXiv](https://arxiv.org/abs/2610.00313) • [📥 PDF](https://arxiv.org/pdf/2610.00313)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Executable Checks for LLM Agents in Scientific Computing

</details>

<details>
<summary><b>64. Explore Broadly, Reason Sharply: Push Small Models toward the Frontier via Sampling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38104) • [📄 arXiv](https://arxiv.org/abs/2609.38104) • [📥 PDF](https://arxiv.org/pdf/2609.38104)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A 9B model could reach GPT-5 / Opus-4.5-level reasoning 🧠, nearly free ⚡🆓, on your own desktop 💻. No additional post-training. Much less jagged 🧩 generalization. This was my intern Panagiotis’ summer project. Still a long way to go on engineering,...

</details>

<details>
<summary><b>65. Prefill-Free Cross-Family KV Cache Transfer for Heterogeneous Multi-Agent LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32259) • [📄 arXiv](https://arxiv.org/abs/2609.32259) • [📥 PDF](https://arxiv.org/pdf/2609.32259)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Can LLMs from different model families directly share KV caches, without receiver-side prefill? Our answer is HeteroFold. I am so excited to share our new paper: Prefill-Free Cross-Family KV Cache Transfer for Heterogeneous Multi-Agent LLMs! Recen...

</details>

<details>
<summary><b>66. Does Native 3D Texture Generation Necessarily Require 3D Assets for Training?</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34621) • [📄 arXiv](https://arxiv.org/abs/2609.34621) • [📥 PDF](https://arxiv.org/pdf/2609.34621)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/wangjiangshan0725/Tex-Zero)

> 🚀 We introduce Tex-Zero , a native 3D texture generation framework trained entirely without real textured 3D assets. Our key finding is that high-quality, fine-grained color information matters more than real 3D geometry for texture learning. Tex-...

</details>

<details>
<summary><b>67. Personalized Image Generation with Reasoning and Reflection</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Seunghyun Yoon, Qinwen Ge, Ngoc N. Tran, Bo Ni, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00737) • [📄 arXiv](https://arxiv.org/abs/2610.00737) • [📥 PDF](https://arxiv.org/pdf/2610.00737)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Grounding Free-Form Instructions for Fashion Complementary Image Generation...

</details>

<details>
<summary><b>68. FlexRouter: Learning Complementary Model Sets for Flexible LLM Routing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Samyadeep Basu, Tiankai Yang, Harry Yang, Wang Wei, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38585) • [📄 arXiv](https://arxiv.org/abs/2609.38585) • [📥 PDF](https://arxiv.org/pdf/2609.38585)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Beyond Top-$k$ Skill Retrieval: Diversity-Aware Skill Routing for LLM Agent...

</details>

<details>
<summary><b>69. Joint and Cross-Modal Video-Audio Generation and Editing: A Unified Formulation and Design Taxonomy</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Daksh Dangi, Wang Wei, Sai Karthik Navuluru, Abhinav Sharma, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34381) • [📄 arXiv](https://arxiv.org/abs/2609.34381) • [📥 PDF](https://arxiv.org/pdf/2609.34381)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Vorch-Omni: Multi-Task Orchestration of Sight and Sound (2026) DreamX-Creat...

</details>

<details>
<summary><b>70. MemFold: Learning Compact Soft Memory for Long-Context Personalization via On-Policy Optimization</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Qingni Wang, Chengzhi Liu, Yiqiao Huang, Yuzhe Yang, Jingxuan Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36435) • [📄 arXiv](https://arxiv.org/abs/2609.36435) • [📥 PDF](https://arxiv.org/pdf/2609.36435)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Johnny221B/memfold)

> No abstract available.

</details>

<details>
<summary><b>71. JevSpawn: Adaptive Agentic Inference through Compositional Action Spaces</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Weiran Huang, Hoyant-Su

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00437) • [📄 arXiv](https://arxiv.org/abs/2610.00437) • [📥 PDF](https://arxiv.org/pdf/2610.00437)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Hoyant-Su/JevSpawn)

> We introduce JevSpawn, a training-free approach to Jev-style agentic inference. The agent derives compositional action spaces from natural-language tasks, explores actions through finite probabilities, and adapts through execution feedback. Shared...

</details>

<details>
<summary><b>72. When Does Correction Become Repair? Mechanistic Auditing of Internal Interventions in Tool-Using LLMs</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36138) • [📄 arXiv](https://arxiv.org/abs/2609.36138) • [📥 PDF](https://arxiv.org/pdf/2609.36138)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ruizheliUOA/mechanistic-tool-use-llm)

> Before invoking external tools, an agentic LLM must select among a K-way action space: executing a call, seeking clarification, answering directly, or declining. While internal activation steering can alter these pre-execution decisions, conventio...

</details>

<details>
<summary><b>73. Controlled Decoding Attacks on Black-Box LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ryan A. Rossi, Wei Yang, Shawn Li, Jesson Wang, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36956) • [📄 arXiv](https://arxiv.org/abs/2609.36956) • [📥 PDF](https://arxiv.org/pdf/2609.36956)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Approximate Speculative Decoding (2026) Carryover Drafting: Recycling Rejec...

</details>

<details>
<summary><b>74. Learning What to Recall: Adaptive Multi-Cue Episodic Memory for World Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34677) • [📄 arXiv](https://arxiv.org/abs/2609.34677) • [📥 PDF](https://arxiv.org/pdf/2609.34677)

**💻 Code:** [⭐ Code](https://github.com/sony/far) • [⭐ Code](https://github.com/huggingface)

> More memory isn’t enough—world models need the right memory at the right time. Most retrieval methods rely on fixed heuristics such as recency, pose proximity, or visual similarity, but the most similar past observation is not always the one that ...

</details>

<details>
<summary><b>75. It Takes Workflows to Evolve Better Workflows</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhenhailong Wang, Yangyi Chen, Haifeng Chen, Haoyu Wang, Xuehang Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01026) • [📄 arXiv](https://arxiv.org/abs/2610.01026) • [📥 PDF](https://arxiv.org/pdf/2610.01026)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://arxiv.org/abs/2610.01026

</details>

<details>
<summary><b>76. Replacing Large Language Models with Jev Decision Models for Low-Latency Edge Service Orchestration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rui Lang, Haochen Gong, Xu Wang, Delong Li, OniReimu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.22753) • [📄 arXiv](https://arxiv.org/abs/2609.22753) • [📥 PDF](https://arxiv.org/pdf/2609.22753)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OniReimu/Edge-Computing-JEV)

> Can a decision model replace an LLM as the intent interpreter in edge service admission? We compare Jev with two self-hosted decision models and three hosted LLMs on 8,280 verified requests and on a live admission path with a real OCR service. Jev...

</details>

<details>
<summary><b>77. Keyword Harnesses Fail Open: A Cheap Diagnostic Ladder for Tool-Use Claims in Small Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** jsantillana

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02142) • [📄 arXiv](https://arxiv.org/abs/2610.02142) • [📥 PDF](https://arxiv.org/pdf/2610.02142)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Keyword-matching benchmarks can credit small models for tool use they never perform. We document such a false positive in a matched-architecture pair of Spanish security language models and propose a ladder of strict, cheap diagnostics. A 661.6M p...

</details>

<details>
<summary><b>78. OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36602) • [📄 arXiv](https://arxiv.org/abs/2609.36602) • [📥 PDF](https://arxiv.org/pdf/2609.36602)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> OTRetarget retargets human demonstrations to a humanoid robot together with the objects it manipulates. Contacts are described by signed distances, closest surface points and relative directions, transferred across human, robot and object geometri...

</details>

<details>
<summary><b>79. Latent-Foresight: End-to-End Learning Predictable Representations for Latent World Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Nikos Komodakis, Spyros Gidaris, Sta8is

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01942) • [📄 arXiv](https://arxiv.org/abs/2610.01942) • [📥 PDF](https://arxiv.org/pdf/2610.01942)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Sta8is/Latent-Foresight)

> No abstract available.

</details>

<details>
<summary><b>80. Before It Fades: Reinforcing Temporal Representations at Inference Time in VideoLLMs</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Junmo Kim, Minseo Kim, Yusung Ro, yshin0917

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01595) • [📄 arXiv](https://arxiv.org/abs/2610.01595) • [📥 PDF](https://arxiv.org/pdf/2610.01595)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Youngwoo-git/Before-It-Fades)

> VideoLLMs do capture temporal information, but it peaks at intermediate layers and progressively fades toward the output. We pinpoint this fading through representation-level analysis and propose Temporal Activation Injection (TAI), which extracts...

</details>

<details>
<summary><b>81. Pay for the Fault, Not the Flow: Label-Free In-Flow Multi-Agent Workflow Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wei Cheng, Zach Chen, Shengyu Chen, Haoyu Wang, Xuehang Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.01017) • [📄 arXiv](https://arxiv.org/abs/2610.01017) • [📥 PDF](https://arxiv.org/pdf/2610.01017)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://arxiv.org/abs/2610.01017

</details>

<details>
<summary><b>82. DexPolicy: Scheduled Exploration for Trajectory-Guided Dexterous Manipulation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00360) • [📄 arXiv](https://arxiv.org/abs/2610.00360) • [📥 PDF](https://arxiv.org/pdf/2610.00360)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIGeeksGroup/DexPolicy)

> Code: https://github.com/AIGeeksGroup/DexPolicy

</details>

<details>
<summary><b>83. Predictive Credit: Measuring What Scientific Explanations Add to Experimental Forecasts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.00314) • [📄 arXiv](https://arxiv.org/abs/2610.00314) • [📥 PDF](https://arxiv.org/pdf/2610.00314)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Measuring What Scientific Explanations Add to Experimental Forecasts

</details>

<details>
<summary><b>84. Honeycomb: Constant-Size Scene Memory Representation for Video World Models</b> ⭐ 26</summary>

<br/>

**👥 Authors:** Keane Ong, Yufeng Weng, Haoyu Chen, Kaichen Zhou, llama2thedog

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37690) • [📄 arXiv](https://arxiv.org/abs/2609.37690) • [📥 PDF](https://arxiv.org/pdf/2609.37690)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kaichen-z/HoneyComb)

> We introduce Honeycomb, a video world model built on our proposed HexMemory. HexMemory represents scene features using a low-rank factorization into three spatial and three spatiotemporal planes, whose dimensions remain fixed throughout generation...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 84 |
| 📅 Today | [`2026-10-04.json`](data/daily/2026-10-04.json) | 84 |
| 📆 This Week | [`2026-W39.json`](data/weekly/2026-W39.json) | 326 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 237 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-04 | 84 | [View JSON](data/daily/2026-10-04.json) |
| 📄 2026-10-03 | 84 | [View JSON](data/daily/2026-10-03.json) |
| 📄 2026-10-02 | 35 | [View JSON](data/daily/2026-10-02.json) |
| 📄 2026-10-01 | 34 | [View JSON](data/daily/2026-10-01.json) |
| 📄 2026-09-30 | 45 | [View JSON](data/daily/2026-09-30.json) |
| 📄 2026-09-29 | 37 | [View JSON](data/daily/2026-09-29.json) |
| 📄 2026-09-28 | 7 | [View JSON](data/daily/2026-09-28.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |
| 📅 2026-W36 | 88 | [View JSON](data/weekly/2026-W36.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 237 | [View JSON](data/monthly/2026-10.json) |
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
