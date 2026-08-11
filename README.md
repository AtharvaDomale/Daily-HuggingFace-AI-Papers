<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-35-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5624+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">65</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">243</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5624+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 11, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. SFT Conflicts, RL Coexists: A Theoretical and Empirical Analysis of Multi-Task Learning for LLMs</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03573) • [📄 arXiv](https://arxiv.org/abs/2608.03573) • [📥 PDF](https://arxiv.org/pdf/2608.03573)

**💻 Code:** [⭐ Code](https://github.com/GaryStack/Parallel-RL) • [⭐ Code](https://github.com/huggingface)

> This work investigates the fundamental differences between SFT and RL in multi-task reasoning. We find that SFT suffers from severe task conflicts under multi-stage training, while RL allows different task capabilities to coexist and improve stabl...

</details>

<details>
<summary><b>2. Beyond Simply Environment Scaling: Designing Effective Environment Distributions for Multimodal Agent Learning</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03571) • [📄 arXiv](https://arxiv.org/abs/2608.03571) • [📥 PDF](https://arxiv.org/pdf/2608.03571)

**💻 Code:** [⭐ Code](https://github.com/GaryStack/Beyond-MMEnv-Scaling) • [⭐ Code](https://github.com/huggingface)

> This work revisits the common paradigm of scaling up environment pools for multimodal agent learning. We find that simply increasing the number of training environments does not always improve performance, and multimodal environments are particula...

</details>

<details>
<summary><b>3. SimWAM: A Simple World Action Model for End-to-End Autonomous Driving</b> ⭐ 29</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.07468) • [📄 arXiv](https://arxiv.org/abs/2608.07468) • [📥 PDF](https://arxiv.org/pdf/2608.07468)

**💻 Code:** [⭐ Code](https://github.com/H-EmbodVis/SimWAM) • [⭐ Code](https://github.com/huggingface)

> World-Action Models (WAMs) improve end-to-end autonomous driving by transferring video dynamics priors to action prediction, but existing methods require costly future generation at inference. We present SimWAM, a simple yet effective WAM that use...

</details>

<details>
<summary><b>4. MatrAIx: Simulating the World with 8.3 Billion Persona Agents</b> ⭐ 498</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04205) • [📄 arXiv](https://arxiv.org/abs/2608.04205) • [📥 PDF](https://arxiv.org/pdf/2608.04205)

**💻 Code:** [⭐ Code](https://github.com/MatrAIx-ai/MatrAIx-Persona-8B) • [⭐ Code](https://github.com/huggingface)

> MatrAIx: Simulating the World with 8.3 Billion AI Agents A research team of over 200 scientists, organized by researchers from Harvard and MIT and including more than 40 scientists from OpenAI, Anthropic, Google DeepMind, and xAI, has published ne...

</details>

<details>
<summary><b>5. YOLO-PEFT: Parameter-Efficient Fine-Tuning on YOLO Family</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.07051) • [📄 arXiv](https://arxiv.org/abs/2608.07051) • [📥 PDF](https://arxiv.org/pdf/2608.07051)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Generic parameter-efficient fine-tuning (PEFT) methods transferred from language models can fail silently on real-time detectors, whose heterogeneous operators and detection-specific components impose placement constraints absent from regular Tran...

</details>

<details>
<summary><b>6. DCAS: Decoupling CLI Agent Scaffolding to Internalize Planning across Scaffolds</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06113) • [📄 arXiv](https://arxiv.org/abs/2608.06113) • [📥 PDF](https://arxiv.org/pdf/2608.06113)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> SWE-Lego, Nebius, CoderForge, SWE-Gym. Every major open trajectory dataset for CLI coding agents is collected primarily under the same scaffold, OpenHands, but practitioners pick their scaffold on cost, licensing, latency and data privacy. We meas...

</details>

<details>
<summary><b>7. StreamArena: Toward Continuous, Interactive, and Long-Horizon Agentic Streaming Video Understanding</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05703) • [📄 arXiv](https://arxiv.org/abs/2608.05703) • [📥 PDF](https://arxiv.org/pdf/2608.05703)

**💻 Code:** [⭐ Code](https://github.com/JIA-Lab-research/StreamArena) • [⭐ Code](https://github.com/huggingface)

> DataSet: https://huggingface.co/datasets/hkuzxc/StreamArena

</details>

<details>
<summary><b>8. When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots in Fine-Tuned Oracles</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.23379) • [📄 arXiv](https://arxiv.org/abs/2607.23379) • [📥 PDF](https://arxiv.org/pdf/2607.23379)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Activation Oracles (AOs) are language models trained to answer natural-language questions about another model’s internal activations. They turn activation analysis into a chat interface: an auditor asks “what is this hidden state about?” and reads...

</details>

<details>
<summary><b>9. Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03796) • [📄 arXiv](https://arxiv.org/abs/2608.03796) • [📥 PDF](https://arxiv.org/pdf/2608.03796)

**💻 Code:** [⭐ Code](https://github.com/CompactifAI/Full-Chunked-KL-Loss) • [⭐ Code](https://github.com/huggingface)

> Knowledge distillation is how large open LLMs get compressed into deployable smaller models, but the standard approach is expensive: it keeps teacher and student loaded together and recomputes the teacher's full-vocabulary output at every step. Mu...

</details>

<details>
<summary><b>10. Reinforcement Learning with Evolving Rubrics as Rewards for Audio Reasoning</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.02831) • [📄 arXiv](https://arxiv.org/abs/2608.02831) • [📥 PDF](https://arxiv.org/pdf/2608.02831)

**💻 Code:** [⭐ Code](https://github.com/tianyi-lab/AudioRubrics) • [⭐ Code](https://github.com/huggingface)

> Reinforcement learning with verifiable rewards elicits audio reasoning, but existing reward designs are complementary in their limitations: outcome rewards supervise only the final answer and let the model reach it without attending to the audio, ...

</details>

<details>
<summary><b>11. Relevant but Incomplete: Referential Dangling as a Paradigm-Level Failure Mode in Hard Prompt Compression</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04569) • [📄 arXiv](https://arxiv.org/abs/2608.04569) • [📥 PDF](https://arxiv.org/pdf/2608.04569)

**💻 Code:** [⭐ Code](https://github.com/JusperLee/Referential-Dangling) • [⭐ Code](https://github.com/huggingface)

> Most hard prompt compression methods treat context selection as an independent relevance-ranking problem: score each chunk, keep the top-k, and discard the rest. This paper reveals a fundamental blind spot. A compressor can retain highly relevant,...

</details>

<details>
<summary><b>12. Douyin Multimodal Embedding Model Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.02148) • [📄 arXiv](https://arxiv.org/abs/2608.02148) • [📥 PDF](https://arxiv.org/pdf/2608.02148)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Introducing DME , a two-stage framework for industrial multimodal embeddings! 🚀 Stage 1 scales heterogeneous contrastive pre-training across text, image, video, and visual documents. Stage 2 goes beyond pairwise alignment with Evidence-Grounded Ty...

</details>

<details>
<summary><b>13. Uncertainty-Aware World Model for Aerial Image-Goal Navigation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05597) • [📄 arXiv](https://arxiv.org/abs/2608.05597) • [📥 PDF](https://arxiv.org/pdf/2608.05597)

**💻 Code:** [⭐ Code](https://github.com/DurYi/UA-NWM) • [⭐ Code](https://github.com/huggingface)

> 🌐 Project page: https://duryi.github.io/UA-NWM-Project-Page/ 💻 Code: https://github.com/DurYi/UA-NWM 🤗 Data: https://huggingface.co/datasets/DurYi/AirGoal-10k 📦 Models: https://huggingface.co/DurYi/UA-NWM-Checkpoints

</details>

<details>
<summary><b>14. Small Foundation Models of Human Cognition and Behaviour</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05224) • [📄 arXiv](https://arxiv.org/abs/2608.05224) • [📥 PDF](https://arxiv.org/pdf/2608.05224)

**💻 Code:** [⭐ Code](https://github.com/socius-org/Centauri) • [⭐ Code](https://github.com/huggingface)

> The paper fine-tunes small pretrained LLMs with LoRA to predict human choices on the Psych-101 benchmark, extending Centaur-style SFT (Binz et al., 2025) down to sub-billion-parameter models.

</details>

<details>
<summary><b>15. Modular TTT: Rethinking Test-Time Training as Composable Modules</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.07110) • [📄 arXiv](https://arxiv.org/abs/2608.07110) • [📥 PDF](https://arxiv.org/pdf/2608.07110)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Test-Time Training with Next-Token Prediction (2026) Learning What to Remem...

</details>

<details>
<summary><b>16. Addressable Memory for Video World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.07408) • [📄 arXiv](https://arxiv.org/abs/2608.07408) • [📥 PDF](https://arxiv.org/pdf/2608.07408)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API ChronoStitch: Training-Free Composition of Visual KV Memories for Long-Hori...

</details>

<details>
<summary><b>17. Round-Trip Consistency: Bidirectional Diffusion Models Can Predict Their Own Rollout Errors</b> ⭐ 9</summary>

<br/>

**👥 Authors:** AScheinker

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.00675) • [📄 arXiv](https://arxiv.org/abs/2608.00675) • [📥 PDF](https://arxiv.org/pdf/2608.00675)

**💻 Code:** [⭐ Code](https://github.com/alexscheinker/round-trip-consistency) • [⭐ Code](https://github.com/huggingface)

> Consistency-based, test-time self-supervised error estimation with a bidirectional diffusion model: a directional flag c_d = ±1 selects forward or backward rollout; reversing it rolls back to an estimate of the starting point and the amount by whi...

</details>

<details>
<summary><b>18. Skaling: Chinchilla's Exponents Meet Kaplan's Coupling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.07222) • [📄 arXiv](https://arxiv.org/abs/2608.07222) • [📥 PDF](https://arxiv.org/pdf/2608.07222)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Bridging Compute- and Data-Optimal Pretraining (2026) Domain-Aware Scaling ...

</details>

<details>
<summary><b>19. The Optimizer Is the Agent: Reasoning-Driven Search across Prompts, Programs, and ML Workflows</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuxiong He, Yite Wang, Canwen Xu, Boyi Liu, Junbo Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06714) • [📄 arXiv](https://arxiv.org/abs/2608.06714) • [📥 PDF](https://arxiv.org/pdf/2608.06714)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API MemoHarness: Agent Harnesses That Learn from Experience (2026) Knowledge-Ce...

</details>

<details>
<summary><b>20. Characterizing the Quality Profile of AI-Generated C++ in Production</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06640) • [📄 arXiv](https://arxiv.org/abs/2608.06640) • [📥 PDF](https://arxiv.org/pdf/2608.06640)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API From Conversation to Contribution: Characterizing Coding Agent in Open-Sour...

</details>

<details>
<summary><b>21. Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xinyu Sun, Yi Zhou, Kai Li, Shun Zhang, Xuechao Zou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06865) • [📄 arXiv](https://arxiv.org/abs/2608.06865) • [📥 PDF](https://arxiv.org/pdf/2608.06865)

**💻 Code:** [⭐ Code](https://github.com/XavierJiezou/ARGUS) • [⭐ Code](https://github.com/huggingface)

> Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection This paper tackles a critical challenge in deepfake video detection: poor generalization to emerging generative models and the lack of fine-grained, evidence-grounded forens...

</details>

<details>
<summary><b>22. Zero Gap Is Not Restoration: Stratified Per-Question Probability Evaluation and Step-wise Mitigation of Benchmark Contamination</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.07341) • [📄 arXiv](https://arxiv.org/abs/2608.07341) • [📥 PDF](https://arxiv.org/pdf/2608.07341)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The field of contamination mitigation has long been trapped in a dual predicament of "unmeasurable and incurable": metrics conceal failures through cancellation, while strategies gamble on pre-hoc estimation. This paper replaces both the yardstick...

</details>

<details>
<summary><b>23. Do AI Personas Grow? Analyzing and Benchmarking Personality Evolution in LLM Agents After Life Events</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06485) • [📄 arXiv](https://arxiv.org/abs/2608.06485) • [📥 PDF](https://arxiv.org/pdf/2608.06485)

**💻 Code:** [⭐ Code](https://github.com/sci-m-wang/BFI-Adapt) • [⭐ Code](https://github.com/huggingface)

> For lifelong Persona-conditioned Agents (PC-Agents), we systematically analyze the impact of major events on dynamic shifts in personality.

</details>

<details>
<summary><b>24. When Privileged Guidance Misaligns: State-Matched Routing and Contextualized Self-Distillation for Multi-Turn Agents</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Peng Wang, Jun Ling, Weiwei Li, Junzhuo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05219) • [📄 arXiv](https://arxiv.org/abs/2608.05219) • [📥 PDF](https://arxiv.org/pdf/2608.05219)

**💻 Code:** [⭐ Code](https://github.com/liujunzhuo/SMRC-SD) • [⭐ Code](https://github.com/huggingface)

> Code: https://github.com/liujunzhuo/SMRC-SD

</details>

<details>
<summary><b>25. Can MLLMs Decode the Creative Leap? Introducing C4 for Cross-Concept Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06501) • [📄 arXiv](https://arxiv.org/abs/2608.06501) • [📥 PDF](https://arxiv.org/pdf/2608.06501)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Inspired by cognitive science, we used Chinese idioms (or Chengyu) as a vehicle to design a cross-concept comprehension task as an anchor for assessing the creative ability of MLLMs, and we created a leaderboard.

</details>

<details>
<summary><b>26. PrivacyPeek: Auditing What LLM-Based Agents Acquire, Not Just What They Say</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Guanchu Wang, Songze Li, Dadi Guo, Jiahui Han, xuan269

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.00152) • [📄 arXiv](https://arxiv.org/abs/2606.00152) • [📥 PDF](https://arxiv.org/pdf/2606.00152)

**💻 Code:** [⭐ Code](https://github.com/Xuan269/PrivacyPeek-Resource) • [⭐ Code](https://github.com/huggingface)

> LLM-based agents are rapidly advancing, autonomously invoking external tools to complete multi-step tasks for users. However, agents often acquire more sensitive information than the task requires. Existing privacy benchmarks audit what the agent'...

</details>

<details>
<summary><b>27. Adversarial Attacks for Good: A Survey of Proactive Protection across the Visual Content Lifecycle</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fuyao Zhang, Zherui Li, Boyang Chen, Jiaming Zhang, hanxunh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04314) • [📄 arXiv](https://arxiv.org/abs/2608.04314) • [📥 PDF](https://arxiv.org/pdf/2608.04314)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Great survey!

</details>

<details>
<summary><b>28. Complementary Matrix-Gated QKAN Fast-Weight Programmers for Quantum Dynamics Forecasting</b> ⭐ 0</summary>

<br/>

**👥 Authors:** En-Jui Kuo, Chen-Yu Liu, Samuel Yen-Chi Chen, Kuo-Chung Peng, Jim137

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.27945) • [📄 arXiv](https://arxiv.org/abs/2607.27945) • [📥 PDF](https://arxiv.org/pdf/2607.27945)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper introduces Complementary Matrix Gating (CMG), a low-rank, coordinate-wise memory update for quantum-inspired KAN fast-weight programmers. CMG allows each fast-weight coordinate to independently balance retaining past information and wri...

</details>

<details>
<summary><b>29. Capek 0.5: An Execution-Centric Vision-Language Model for Embodied Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06756) • [📄 arXiv](https://arxiv.org/abs/2608.06756) • [📥 PDF](https://arxiv.org/pdf/2608.06756)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Robot execution is iterative: each action changes what must be perceived, reasoned about, grounded, and verified next. This paper presents Capek 0.5, an embodied VLM that organizes post-training around four execution-facing capabilities and consol...

</details>

<details>
<summary><b>30. Enfold: Folding World Model Imagination into Predictive Representations for Ultra-Efficient Embodied Control</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Antao Xiang, Chengqun Yang, Fulong Liu, Yitong Xing, zwl666666

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26657) • [📄 arXiv](https://arxiv.org/abs/2607.26657) • [📥 PDF](https://arxiv.org/pdf/2607.26657)

**💻 Code:** [⭐ Code](https://github.com/zwl666666/enfold) • [⭐ Code](https://github.com/huggingface)

> 🚀 Enfold: Folding World Model Imagination into Predictive Representations for Ultra-Efficient Embodied Control Do world models really need to generate a future at every control step? Enfold explores a different idea: the most reusable asset of a w...

</details>

<details>
<summary><b>31. CLIP-CC-Bench: Evaluating Paragraph-Level Video Descriptions in Video-Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.04302) • [📄 arXiv](https://arxiv.org/abs/2608.04302) • [📥 PDF](https://arxiv.org/pdf/2608.04302)

**💻 Code:** [⭐ Code](https://github.com/Multimodal-Intelligence-Lab/CLIP-CC-Bench) • [⭐ Code](https://github.com/huggingface)

> Can a video-language model write a faithful paragraph about a minute of film? Video benchmarks still lean on short clips, one-sentence captions, and multiple-choice QA — none of which test long-form description. CLIP-CC-Bench does. 🎬 200 movie cli...

</details>

<details>
<summary><b>32. Towards Interpretable Foundation Models for Retinal Fundus Images</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18846) • [📄 arXiv](https://arxiv.org/abs/2603.18846) • [📥 PDF](https://arxiv.org/pdf/2603.18846)

**💻 Code:** [⭐ Code](https://github.com/berenslab/interpretable_FM) • [⭐ Code](https://github.com/huggingface)

> Dual-IFM is a foundation model for retinal fundus images that is interpretable-by-design. Its BagNet backbone with small receptive fields produces class evidence maps faithful to the model's decision-making process, and a 2D projection layer learn...

</details>

<details>
<summary><b>33. FATE: Frame-Level Audio-Visual Temporal Embedding</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.01310) • [📄 arXiv](https://arxiv.org/abs/2608.01310) • [📥 PDF](https://arxiv.org/pdf/2608.01310)

**💻 Code:** [⭐ Code](https://github.com/guankaisi/FATE) • [⭐ Code](https://github.com/huggingface)

> When a dog opens its mouth and barks, humans naturally recognize what the sound is and when it occurs. Building audio-visual models with this same ability requires representations that capture both semantic and temporal alignment. Current approach...

</details>

<details>
<summary><b>34. DuplexGen: Adaptive Synthesis of Human-AI Turn-Taking Dialogues</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.26178) • [📄 arXiv](https://arxiv.org/abs/2607.26178) • [📥 PDF](https://arxiv.org/pdf/2607.26178)

**💻 Code:** [⭐ Code](https://github.com/duplexgen/duplexgen-code) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>35. OneEmo: A Unified Multimodal Reasoning Model for Emotion Perception, Understanding, and Interaction</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Xiaojiang Peng, Zhide Chen, Jingyi Zhang, Zheng Lian, Jiaha0Hu4ng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.06013) • [📄 arXiv](https://arxiv.org/abs/2608.06013) • [📥 PDF](https://arxiv.org/pdf/2608.06013)

**💻 Code:** [⭐ Code](https://github.com/waHAHJIAHAO/OneEmo) • [⭐ Code](https://github.com/huggingface)

> The first unified multimodal emotion reasoning model capable of handling sentiment analysis, basic emotion recognition, open-vocabulary emotion detection, intent identification, sarcasm understanding, humor comprehension, empathetic response, and ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 35 |
| 📅 Today | [`2026-08-11.json`](data/daily/2026-08-11.json) | 35 |
| 📆 This Week | [`2026-W32.json`](data/weekly/2026-W32.json) | 65 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 243 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-11 | 35 | [View JSON](data/daily/2026-08-11.json) |
| 📄 2026-08-10 | 30 | [View JSON](data/daily/2026-08-10.json) |
| 📄 2026-08-09 | 30 | [View JSON](data/daily/2026-08-09.json) |
| 📄 2026-08-08 | 30 | [View JSON](data/daily/2026-08-08.json) |
| 📄 2026-08-07 | 9 | [View JSON](data/daily/2026-08-07.json) |
| 📄 2026-08-06 | 14 | [View JSON](data/daily/2026-08-06.json) |
| 📄 2026-08-05 | 9 | [View JSON](data/daily/2026-08-05.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W32 | 65 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |
| 📅 2026-W30 | 112 | [View JSON](data/weekly/2026-W30.json) |
| 📅 2026-W29 | 70 | [View JSON](data/weekly/2026-W29.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 243 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
