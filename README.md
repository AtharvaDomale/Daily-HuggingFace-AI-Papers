<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-57-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2085+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">57</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">164</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">566</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2085+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 12, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05400) • [📄 arXiv](https://arxiv.org/abs/2602.05400) • [📥 PDF](https://arxiv.org/pdf/2602.05400)

> In this paper, we argue that LLM pre-training is entering a “data-wall” regime where readily available high-quality public text is approaching exhaustion, so progress must shift from more tokens to better tokens chosen at the right time. While mos...

</details>

<details>
<summary><b>2. Code2World: A GUI World Model via Renderable Code Generation</b> ⭐ 131</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09856) • [📄 arXiv](https://arxiv.org/abs/2602.09856) • [📥 PDF](https://arxiv.org/pdf/2602.09856)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/Code2World)

> Project Page: https://amap-ml.github.io/Code2World/ Github: https://github.com/AMAP-ML/Code2World

</details>

<details>
<summary><b>3. UI-Venus-1.5 Technical Report</b> ⭐ 708</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09082) • [📄 arXiv](https://arxiv.org/abs/2602.09082) • [📥 PDF](https://arxiv.org/pdf/2602.09082)

**💻 Code:** [⭐ Code](https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5) • [⭐ Code](https://github.com/inclusionAI/UI-Venus)

> Is your GUI Agent ready for real work? 🔥 We’ve seen many great previous GUI Agents, but making a "stable assistant" for phones and websites is still hard. There are three main problems: 1️⃣ Knowledge Gap: AI often misses less common icons and does...

</details>

<details>
<summary><b>4. Chain of Mindset: Reasoning with Adaptive Cognitive Modes</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10063) • [📄 arXiv](https://arxiv.org/abs/2602.10063) • [📥 PDF](https://arxiv.org/pdf/2602.10063)

**💻 Code:** [⭐ Code](https://github.com/QuantaAlpha/chain-of-mindset)

> CoM is a training-free agentic framework that dynamically orchestrates four step-level mindsets (Spatial, Convergent, Divergent, Algorithmic) via a Meta-Agent and a Context Gate, avoiding one-size-fits-all reasoning and improving accuracy and effi...

</details>

<details>
<summary><b>5. SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning</b> ⭐ 140</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08234) • [📄 arXiv](https://arxiv.org/abs/2602.08234) • [📥 PDF](https://arxiv.org/pdf/2602.08234)

**💻 Code:** [⭐ Code](https://github.com/aiming-lab/SkillRL)

> Skill accumulation is the new paradigm for AI agents. We’re moving from static models to recursive evolution 🧬. SkillRL proves skills > scale, enabling a 7B model to beat GPT-4o 🚀. Evolving > Scaling. 💡 Paper: https://arxiv.org/abs/2602.08234 Code...

</details>

<details>
<summary><b>6. P1-VL: Bridging Visual Perception and Scientific Reasoning in Physics Olympiads</b> ⭐ 13</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09443) • [📄 arXiv](https://arxiv.org/abs/2602.09443) • [📥 PDF](https://arxiv.org/pdf/2602.09443)

**💻 Code:** [⭐ Code](https://github.com/PRIME-RL/P1-VL)

> Project: https://prime-rl.github.io/P1-VL GitHub: https://github.com/PRIME-RL/P1-VL

</details>

<details>
<summary><b>7. Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning</b> ⭐ 44</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10090) • [📄 arXiv](https://arxiv.org/abs/2602.10090) • [📥 PDF](https://arxiv.org/pdf/2602.10090)

**💻 Code:** [⭐ Code](https://github.com/Snowflake-Labs/agent-world-model)

> Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning 🚀 Introducing Agent World Model (AWM) — we synthesized 1,000 code-driven environments with 35K tools and 10K tasks for large-scale agentic reinforcement learning...

</details>

<details>
<summary><b>8. Prism: Spectral-Aware Block-Sparse Attention</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08426) • [📄 arXiv](https://arxiv.org/abs/2602.08426) • [📥 PDF](https://arxiv.org/pdf/2602.08426)

**💻 Code:** [⭐ Code](https://github.com/xinghaow99/prism)

> TL;DR Prism is a training-free method to accelerate long-context LLM pre-filling. It addresses the "blind spot" in standard mean pooling caused by Rotary Positional Embeddings (RoPE) by disentangling attention into high-frequency and low-frequency...

</details>

<details>
<summary><b>9. DLLM-Searcher: Adapting Diffusion Large Language Model for Search Agents</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07035) • [📄 arXiv](https://arxiv.org/abs/2602.07035) • [📥 PDF](https://arxiv.org/pdf/2602.07035)

**💻 Code:** [⭐ Code](https://github.com/bubble65/DLLM-Searcher)

> 🧠🔍 DLLM-Searcher: Adapting Diffusion Large Language Models for Search Agents Diffusion Large Language Models (dLLMs) offer flexible generation but struggle as search agents due to latency and weak tool-use capabilities.  This paper introduces DLLM...

</details>

<details>
<summary><b>10. Olaf-World: Orienting Latent Actions for Video World Modeling</b> ⭐ 33</summary>

<br/>

**👥 Authors:** Mike Zheng Shou, Ivor W. Tsang, Yuchao Gu, YuxinJ

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10104) • [📄 arXiv](https://arxiv.org/abs/2602.10104) • [📥 PDF](https://arxiv.org/pdf/2602.10104)

**💻 Code:** [⭐ Code](https://github.com/showlab/Olaf-World)

> No abstract available.

</details>

<details>
<summary><b>11. Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09084) • [📄 arXiv](https://arxiv.org/abs/2602.09084) • [📥 PDF](https://arxiv.org/pdf/2602.09084)

**💻 Code:** [⭐ Code](https://github.com/taco-group/agent-banana)

> Agent Banana: High-Fidelity Image Editing with Agentic Thinking and Tooling

</details>

<details>
<summary><b>12. Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07022) • [📄 arXiv](https://arxiv.org/abs/2602.07022) • [📥 PDF](https://arxiv.org/pdf/2602.07022)

> This study presents a theoretical analysis of autoregressive image generation with diffusion loss, demonstrating that patch denoising optimization effectively mitigates condition errors and leads to a stable condition distribution. To further addr...

</details>

<details>
<summary><b>13. TokenTrim: Inference-Time Token Pruning for Autoregressive Long Video Generation</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Lior Wolf, Amit Edenzon, Eitan Shaar, shaulov

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00268) • [📄 arXiv](https://arxiv.org/abs/2602.00268) • [📥 PDF](https://arxiv.org/pdf/2602.00268)

**💻 Code:** [⭐ Code](https://github.com/arielshaulov/TokenTrim)

> Project page: https://arielshaulov.github.io/TokenTrim/ Open source code 🥳: https://github.com/arielshaulov/TokenTrim

</details>

<details>
<summary><b>14. SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04208) • [📄 arXiv](https://arxiv.org/abs/2602.04208) • [📥 PDF](https://arxiv.org/pdf/2602.04208)

> We tackle test-time robustness of VLA models without additional training or multiple forward passes, by proposing SCALE: jointly modulate visual attention and action decoding based on self-uncertainty.

</details>

<details>
<summary><b>15. LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00462) • [📄 arXiv](https://arxiv.org/abs/2602.00462) • [📥 PDF](https://arxiv.org/pdf/2602.00462)

**💻 Code:** [⭐ Code](https://github.com/McGill-NLP/latentlens)

> In this paper we propose a new interpretability method LatentLens. With this we can finally show that visual tokens are actually interpretable across all layers in an LLM, something that past methods like logit lens and or using the LLM's embeddin...

</details>

<details>
<summary><b>16. BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaoyu Chen, Yanjiang Guo, Yuanfei Luo, Jianke Zhang, Yucheng Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09849) • [📄 arXiv](https://arxiv.org/abs/2602.09849) • [📥 PDF](https://arxiv.org/pdf/2602.09849)

> BagelVLA is a unified model that integrates linguistic planning, visual forecasting, and action generation within a single framework for long-horizon manipulation tasks. 🧠 Model Architecture BagelVLA utilizes a Mixture-of-Transformers (MoT) archit...

</details>

<details>
<summary><b>17. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Zezhi Liu, Shaojie Ren, Zekun Qi, Wenyao Zhang, Jingwen Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10098) • [📄 arXiv](https://arxiv.org/abs/2602.10098) • [📥 PDF](https://arxiv.org/pdf/2602.10098)

**💻 Code:** [⭐ Code](https://github.com/ginwind/VLA-JEPA)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Ac...

</details>

<details>
<summary><b>18. ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06820) • [📄 arXiv](https://arxiv.org/abs/2602.06820) • [📥 PDF](https://arxiv.org/pdf/2602.06820)

> We introduce ScaleEnv, a framework that constructs fully interactive environments and verifiable tasks entirely from scratch. By enabling agents to learn through exploration within ScaleEnv, we demonstrate significant performance improvements on u...

</details>

<details>
<summary><b>19. Fine-T2I: An Open, Large-Scale, and Diverse Dataset for High-Quality T2I Fine-Tuning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09439) • [📄 arXiv](https://arxiv.org/abs/2602.09439) • [📥 PDF](https://arxiv.org/pdf/2602.09439)

> Dataset: https://huggingface.co/datasets/ma-xu/fine-t2i Space: https://huggingface.co/spaces/ma-xu/fine-t2i-explore

</details>

<details>
<summary><b>20. Contact-Anchored Policies: Contact Conditioning Creates Strong Robot Utility Models</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09017) • [📄 arXiv](https://arxiv.org/abs/2602.09017) • [📥 PDF](https://arxiv.org/pdf/2602.09017)

**💻 Code:** [⭐ Code](https://github.com/jeffacce/cap-policy)

> The prevalent paradigm in robot learning attempts to generalize across environments, embodiments, and tasks with language prompts at runtime. A fundamental tension limits this approach: language is often too abstract to guide the concrete physical...

</details>

<details>
<summary><b>21. Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems</b> ⭐ 53</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08847) • [📄 arXiv](https://arxiv.org/abs/2602.08847) • [📥 PDF](https://arxiv.org/pdf/2602.08847)

**💻 Code:** [⭐ Code](https://github.com/langfengQ/DrMAS)

> Dr. MAS is designed for stable end-to-end RL post-training 🔥 of multi-agent LLM systems. It enables agents to collaborate on complex reasoning tasks with: ✨ Flexible agent registry & multi-agent orchestration ✨ Heterogeneous LLMs (shared/non-share...

</details>

<details>
<summary><b>22. Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Yang Wang, Wei Zhang, Yuyang Song, Yizhi Li, Siwei Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01244) • [📄 arXiv](https://arxiv.org/abs/2602.01244) • [📥 PDF](https://arxiv.org/pdf/2602.01244)

**💻 Code:** [⭐ Code](https://github.com/multimodal-art-projection/TerminalTraj)

> This is a repo for paper "Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments"

</details>

<details>
<summary><b>23. VideoWorld 2: Learning Transferable Knowledge from Real-world Videos</b> ⭐ 685</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10102) • [📄 arXiv](https://arxiv.org/abs/2602.10102) • [📥 PDF](https://arxiv.org/pdf/2602.10102)

**💻 Code:** [⭐ Code](https://github.com/ByteDance-Seed/VideoWorld/tree/main/VideoWorld2)

> 🤖Text is not enough, Visual is the key to AGI！Can Al learn transferable knowledge for complex tasks directly from videos? Just like a child learns to fold a paper airplane or build a LEGO from video tutorials👶 😎Thrilled to introduce VideoWorld 2, ...

</details>

<details>
<summary><b>24. Steer2Adapt: Dynamically Composing Steering Vectors Elicits Efficient Adaptation of LLMs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07276) • [📄 arXiv](https://arxiv.org/abs/2602.07276) • [📥 PDF](https://arxiv.org/pdf/2602.07276)

> Activation steering has emerged as a promising method for efficiently adapting large language models (LLMs) to downstream behaviors. However, most existing steering approaches identify and steer the model from a single static direction for each ta...

</details>

<details>
<summary><b>25. Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08382) • [📄 arXiv](https://arxiv.org/abs/2602.08382) • [📥 PDF](https://arxiv.org/pdf/2602.08382)

> Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning. We introduce LycheeMemory, a cognitively inspired framework that enables efficient long-context inference via chunk-wise compression and selective memory ...

</details>

<details>
<summary><b>26. Rethinking Global Text Conditioning in Diffusion Transformers</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Yuchen Liu, Ilya Drobyshevskiy, Zongze Wu, Daniil Pakhomov, Nikita Starodubcev

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09268) • [📄 arXiv](https://arxiv.org/abs/2602.09268) • [📥 PDF](https://arxiv.org/pdf/2602.09268)

**💻 Code:** [⭐ Code](https://github.com/quickjkee/modulation-guidance)

> GitHub: https://github.com/quickjkee/modulation-guidance

</details>

<details>
<summary><b>27. iGRPO: Self-Feedback-Driven LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09000) • [📄 arXiv](https://arxiv.org/abs/2602.09000) • [📥 PDF](https://arxiv.org/pdf/2602.09000)

> Let's discuss Self-Feedback for RL Reasoning (iGRPO) Motivation. Current RL methods for reasoning (GRPO, DAPO, etc.) treat each generation as a one-shot attempt. The model samples, gets a reward, updates, and moves on. But humans almost never solv...

</details>

<details>
<summary><b>28. Covo-Audio Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09823) • [📄 arXiv](https://arxiv.org/abs/2602.09823) • [📥 PDF](https://arxiv.org/pdf/2602.09823)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Fun-Audio-Chat Technical Report (2025) FlashLabs Chroma 1.0: A Real-Time En...

</details>

<details>
<summary><b>29. Effective Reasoning Chains Reduce Intrinsic Dimensionality</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09276) • [📄 arXiv](https://arxiv.org/abs/2602.09276) • [📥 PDF](https://arxiv.org/pdf/2602.09276)

> No abstract available.

</details>

<details>
<summary><b>30. TreeCUA: Efficiently Scaling GUI Automation with Tree-Structured Verifiable Evolution</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Liming Zheng, Lei Chen, Xuanle Zhao, Jing Huang, Deyang Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09662) • [📄 arXiv](https://arxiv.org/abs/2602.09662) • [📥 PDF](https://arxiv.org/pdf/2602.09662)

**💻 Code:** [⭐ Code](https://github.com/UITron-hub/TreeCUA)

> TreeCUA: Efficiently Scaling GUI Automation with Tree-Structured Verifiable Evolution

</details>

<details>
<summary><b>31. ANCHOR: Branch-Point Data Generation for GUI Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07153) • [📄 arXiv](https://arxiv.org/abs/2602.07153) • [📥 PDF](https://arxiv.org/pdf/2602.07153)

> End-to-end GUI agents for real desktop environments require large amounts of high-quality interaction data, yet collecting human demonstrations is expensive and existing synthetic pipelines often suffer from limited task diversity or noisy, goal-d...

</details>

<details>
<summary><b>32. SAGE: Scalable Agentic 3D Scene Generation for Embodied AI</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10116) • [📄 arXiv](https://arxiv.org/abs/2602.10116) • [📥 PDF](https://arxiv.org/pdf/2602.10116)

> No abstract available.

</details>

<details>
<summary><b>33. Autoregressive Image Generation with Masked Bit Modeling</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09024) • [📄 arXiv](https://arxiv.org/abs/2602.09024) • [📥 PDF](https://arxiv.org/pdf/2602.09024)

**💻 Code:** [⭐ Code](https://github.com/amazon-far/BAR)

> SOTA discrete visual generation defeats diffusion models with 0.99 FID score, project page is available at https://bar-gen.github.io/

</details>

<details>
<summary><b>34. OPE: Overcoming Information Saturation in Parallel Thinking via Outline-Guided Path Exploration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jianfei Zhang, Xiangyu Xi, Jianing Wang, Qi Guo, DeyangKong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08344) • [📄 arXiv](https://arxiv.org/abs/2602.08344) • [📥 PDF](https://arxiv.org/pdf/2602.08344)

> Parallel thinking has emerged as a new paradigm for large reasoning models (LRMs) in tackling complex problems. Recent methods leverage Reinforcement Learning (RL) to enhance parallel thinking, aiming to address the limitations in computational re...

</details>

<details>
<summary><b>35. TodoEvolve: Learning to Architect Agent Planning Systems</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Heng Chang, Zihan Zhang, Guibin Zhang, Yanzuo Jiang, Jiaxi Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07839) • [📄 arXiv](https://arxiv.org/abs/2602.07839) • [📥 PDF](https://arxiv.org/pdf/2602.07839)

**💻 Code:** [⭐ Code](https://github.com/EcthelionLiu/TodoEvolve)

> Planning has become a central capability for contemporary agent systems in navigating complex, long-horizon tasks, yet existing approaches predominantly rely on fixed, hand-crafted planning structures that lack the flexibility to adapt to the stru...

</details>

<details>
<summary><b>36. Secure Code Generation via Online Reinforcement Learning with Vulnerability Reward Model</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07422) • [📄 arXiv](https://arxiv.org/abs/2602.07422) • [📥 PDF](https://arxiv.org/pdf/2602.07422)

**💻 Code:** [⭐ Code](https://github.com/AndrewWTY/SecCoderX)

> Large language models (LLMs) are increasingly used in software development, yet their tendency to generate insecure code remains a major barrier to real-world deployment. Existing secure code alignment methods often suffer from a functionality–sec...

</details>

<details>
<summary><b>37. Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06161) • [📄 arXiv](https://arxiv.org/abs/2602.06161) • [📥 PDF](https://arxiv.org/pdf/2602.06161)

> We found a silly failure mode in Parallel Revocable Diffusion Decoding: flip-flop . A token gets ReMask’ed… then comes back unchanged. In the existing approach, <1% of ReMasks actually change the token (≈99% wasted). We propose COVER which verifie...

</details>

<details>
<summary><b>38. Stable Velocity: A Variance Perspective on Flow Matching</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Xin Tao, Liang Hou, Xin Yu, Yongxing Zhang, Donglin Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05435) • [📄 arXiv](https://arxiv.org/abs/2602.05435) • [📥 PDF](https://arxiv.org/pdf/2602.05435)

**💻 Code:** [⭐ Code](https://github.com/linYDTHU/StableVelocity)

> While flow matching is elegant, its reliance on single-sample conditional velocities leads to high-variance training targets that destabilize optimization and slow convergence. By explicitly characterizing this variance, we identify 1) a high-vari...

</details>

<details>
<summary><b>39. From Directions to Regions: Decomposing Activations in Language Models via Local Geometry</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Atticus Geiger, Shauli Ravfogel, Omri Fahn, Shaked Ronen, Or Shafran

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02464) • [📄 arXiv](https://arxiv.org/abs/2602.02464) • [📥 PDF](https://arxiv.org/pdf/2602.02464)

**💻 Code:** [⭐ Code](https://github.com/ordavid-s/decomposing-activations-local-geometry)

> Activation decomposition methods in language models are tightly coupled to geometric assumptions on how concepts are realized in activation space. Existing approaches search for individual global directions, implicitly assuming linear separability...

</details>

<details>
<summary><b>40. On the Optimal Reasoning Length for RL-Trained Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rio Yokota, Taishi-N324, neodymium6

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09591) • [📄 arXiv](https://arxiv.org/abs/2602.09591) • [📥 PDF](https://arxiv.org/pdf/2602.09591)

> RL-trained reasoning models often produce longer CoT, increasing test-time cost. We compare several length-control methods on Qwen3-1.7B-Base and DeepSeek-R1-Distill-Qwen-1.5B, and characterize when length penalties hurt reasoning acquisition vs w...

</details>

<details>
<summary><b>41. Learning Self-Correction in Vision-Language Models via Rollout Augmentation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruqi Zhang, Bolian Li, Ziliang Qiu, Yi Ding

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08503) • [📄 arXiv](https://arxiv.org/abs/2602.08503) • [📥 PDF](https://arxiv.org/pdf/2602.08503)

> Learning self-correction in Vision-language models via rollout augmentation

</details>

<details>
<summary><b>42. Learning to Continually Learn via Meta-learning Agentic Memory Designs</b> ⭐ 39</summary>

<br/>

**👥 Authors:** Jeff Clune, Shengran Hu, Yiming Xiong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07755) • [📄 arXiv](https://arxiv.org/abs/2602.07755) • [📥 PDF](https://arxiv.org/pdf/2602.07755)

**💻 Code:** [⭐ Code](https://github.com/zksha/alma)

> Can AI agents design better memory mechanisms for themselves? Introducing Learning to Continually Learn via Meta-learning Memory Designs. A meta agent automatically designs memory mechanisms, including what info to store, how to retrieve it, and h...

</details>

<details>
<summary><b>43. ContextBench: A Benchmark for Context Retrieval in Coding Agents</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Jiaming Wang, Rili Feng, Bohan Zhang, Letian Zhu, Han Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05892) • [📄 arXiv](https://arxiv.org/abs/2602.05892) • [📥 PDF](https://arxiv.org/pdf/2602.05892)

**💻 Code:** [⭐ Code](https://github.com/EuniAI/ContextBench)

> Most repo-level benchmarks measure Pass@k ✅ But fixing a bug does not mean the agent understood the code 👀 We built ContextBench 🎉 A benchmark to measure whether coding agents actually retrieve and use the right context 🔍📂 📊 What’s inside 🧩 1,136 ...

</details>

<details>
<summary><b>44. Locas: Your Models are Principled Initializers of Locally-Supported Parametric Memories</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05085) • [📄 arXiv](https://arxiv.org/abs/2602.05085) • [📥 PDF](https://arxiv.org/pdf/2602.05085)

> We introduce Locas, a parametric memory for parameter-efficient Test-Time Training (TTT) and continual learning. Unlike previous methods that only introduce in-place low-rank model updates (such as LoRA) that do not provide expanded capacity or re...

</details>

<details>
<summary><b>45. Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.10099) • [📄 arXiv](https://arxiv.org/abs/2602.10099) • [📥 PDF](https://arxiv.org/pdf/2602.10099)

**💻 Code:** [⭐ Code](https://github.com/amandpkr/RJF)

> Leveraging representation encoders for generative modeling offers a path for efficient, high-fidelity synthesis. However, standard diffusion transformers fail to converge on these representations directly. While recent work attributes this to a ca...

</details>

<details>
<summary><b>46. LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chris Russell, William Bankes, Thomas Foster, William Lugoloobi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09924) • [📄 arXiv](https://arxiv.org/abs/2602.09924) • [📥 PDF](https://arxiv.org/pdf/2602.09924)

**💻 Code:** [⭐ Code](https://github.com/KabakaWilliam/llms_know_difficulty)

> We show that LLMs maintain a linearly accessible internal representation of difficulty that differs from human assessments and varies across decoding settings. We apply this to route queries between models with different reasoning capabilities. Gi...

</details>

<details>
<summary><b>47. Bridging Academia and Industry: A Comprehensive Benchmark for Attributed Graph Clustering</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08519) • [📄 arXiv](https://arxiv.org/abs/2602.08519) • [📥 PDF](https://arxiv.org/pdf/2602.08519)

**💻 Code:** [⭐ Code](https://github.com/Cloudy1225/PyAGC)

> PyAGC is a production-ready, modular library and comprehensive benchmark for Attributed Graph Clustering (AGC), built on PyTorch and PyTorch Geometric. It unifies 20+ state-of-the-art algorithms under a principled Encode-Cluster-Optimize (ECO) fra...

</details>

<details>
<summary><b>48. MIND: Benchmarking Memory Consistency and Action Control in World Models</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.08025) • [📄 arXiv](https://arxiv.org/abs/2602.08025) • [📥 PDF](https://arxiv.org/pdf/2602.08025)

**💻 Code:** [⭐ Code](https://github.com/CSU-JPG/MIND)

> TL;DR: The first open-domain closed-loop revisited benchmark for evaluating memory consistency and action control in world models

</details>

<details>
<summary><b>49. CausalArmor: Efficient Indirect Prompt Injection Guardrails via Causal Attribution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07918) • [📄 arXiv](https://arxiv.org/abs/2602.07918) • [📥 PDF](https://arxiv.org/pdf/2602.07918)

> I'm excited to share our latest work to defend Prompt Injection: "CausalArmor: Efficient Indirect Prompt Injection Guardrails via Causal Attribution". CausalArmor, a selective defense: 🧠 Causal attribution at privileged actions: measure whether th...

</details>

<details>
<summary><b>50. Surprisal-Guided Selection: Compute-Optimal Test-Time Strategies for Execution-Grounded Code Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jarrodbarnes

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07670) • [📄 arXiv](https://arxiv.org/abs/2602.07670) • [📥 PDF](https://arxiv.org/pdf/2602.07670)

**💻 Code:** [⭐ Code](https://github.com/jbarnes850/test-time-training)

> Standard practice selects the most confident model output. I tested the opposite on GPU kernel optimization and found that selecting by surprisal (the model's least confident correct solution) achieves 80% success vs 50% for confidence-guided, wit...

</details>

<details>
<summary><b>51. AgentSys: Secure and Dynamic LLM Agents Through Explicit Hierarchical Memory Management</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Ning Zhang, Chaowei Xiao, Hao Li, Ruoyao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.07398) • [📄 arXiv](https://arxiv.org/abs/2602.07398) • [📥 PDF](https://arxiv.org/pdf/2602.07398)

**💻 Code:** [⭐ Code](https://github.com/ruoyaow/agentsys-memory)

> AgentSys defends against indirect prompt injection through explicit hierarchical memory management, reducing attack surface and preserving agent decision-making by preventing malicious instructions from persisting in the context window.

</details>

<details>
<summary><b>52. VISTA-Bench: Do Vision-Language Models Really Understand Visualized Text as Well as Pure Text?</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Yujie Cheng, Xinzhe Han, Yuhao Wang, Juntong Feng, liuqa

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04802) • [📄 arXiv](https://arxiv.org/abs/2602.04802) • [📥 PDF](https://arxiv.org/pdf/2602.04802)

**💻 Code:** [⭐ Code](https://github.com/QingAnLiu/VISTA-Bench)

> Vision-Language Models (VLMs) have achieved impressive performance in cross-modal understanding across textual and visual inputs, yet existing benchmarks predominantly focus on pure-text queries. In real-world scenarios, language also frequently a...

</details>

<details>
<summary><b>53. C-ΔΘ: Circuit-Restricted Weight Arithmetic for Selective Refusal</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04521) • [📄 arXiv](https://arxiv.org/abs/2602.04521) • [📥 PDF](https://arxiv.org/pdf/2602.04521)

> C-ΔΘ (Circuit-Restricted Weight Arithmetic) shifts selective refusal from inference-time steering to an offline, checkpoint-level edit. It first identifies the refusal-causal circuit via EAP-IG, then applies a circuit-restricted weight update that...

</details>

<details>
<summary><b>54. Temporal Pair Consistency for Variance-Reduced Flow Matching</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jindong Wang, Chikap421

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04908) • [📄 arXiv](https://arxiv.org/abs/2602.04908) • [📥 PDF](https://arxiv.org/pdf/2602.04908)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Stable Velocity: A Variance Perspective on Flow Matching (2026) Rethinking ...

</details>

<details>
<summary><b>55. SafePred: A Predictive Guardrail for Computer-Using Agents via World Models</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01725) • [📄 arXiv](https://arxiv.org/abs/2602.01725) • [📥 PDF](https://arxiv.org/pdf/2602.01725)

**💻 Code:** [⭐ Code](https://github.com/YurunChen/SafePred)

> With the widespread deployment of Computer-using Agents (CUAs) in complex real-world environments, prevalent long-term risks often lead to severe and irreversible consequences. Most existing guardrails for CUAs adopt a reactive approach, constrain...

</details>

<details>
<summary><b>56. SHARP: Social Harm Analysis via Risk Profiles for Measuring Inequities in Large Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lisa Erickson, Tushar Bandopadhyay, alokabhishek

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21235) • [📄 arXiv](https://arxiv.org/abs/2601.21235) • [📥 PDF](https://arxiv.org/pdf/2601.21235)

> Large language models (LLMs) are increasingly deployed in high-stakes domains, where rare but severe failures can result in irreversible harm. However, prevailing evaluation benchmarks often reduce complex social risk to mean-centered scalar score...

</details>

<details>
<summary><b>57. SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes</b> ⭐ 46</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.09153) • [📄 arXiv](https://arxiv.org/abs/2602.09153) • [📥 PDF](https://arxiv.org/pdf/2602.09153)

**💻 Code:** [⭐ Code](https://github.com/nepfaff/scenesmith)

> Meet SceneSmith: An agentic system that generates entire simulation-ready environments from a single text prompt. VLM agents collaborate to build scenes with dozens of objects per room, articulated furniture, and full physics properties.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 57 |
| 📅 Today | [`2026-02-12.json`](data/daily/2026-02-12.json) | 57 |
| 📆 This Week | [`2026-W06.json`](data/weekly/2026-W06.json) | 164 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 566 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-12 | 57 | [View JSON](data/daily/2026-02-12.json) |
| 📄 2026-02-11 | 58 | [View JSON](data/daily/2026-02-11.json) |
| 📄 2026-02-10 | 2 | [View JSON](data/daily/2026-02-10.json) |
| 📄 2026-02-09 | 47 | [View JSON](data/daily/2026-02-09.json) |
| 📄 2026-02-08 | 47 | [View JSON](data/daily/2026-02-08.json) |
| 📄 2026-02-07 | 47 | [View JSON](data/daily/2026-02-07.json) |
| 📄 2026-02-06 | 52 | [View JSON](data/daily/2026-02-06.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W06 | 164 | [View JSON](data/weekly/2026-W06.json) |
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 566 | [View JSON](data/monthly/2026-02.json) |
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
