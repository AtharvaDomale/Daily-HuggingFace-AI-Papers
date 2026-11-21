<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-50-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8141+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">50</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">134</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">413</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8141+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 21, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. WeVisForensics: Self-Evolving Agent for Omni AI-Generated Video Detection</b> ⭐ 48</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21373) • [📄 arXiv](https://arxiv.org/abs/2611.21373) • [📥 PDF](https://arxiv.org/pdf/2611.21373)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/WeVisForensics)

> We present WeVisForensics, an agentic framework for AI-generated video detection in which a main investigator agent coordinates a self-evolving library of forensic inspector sub-agents spanning visual and audio cues. Github: https://github.com/Ten...

</details>

<details>
<summary><b>2. Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26444) • [📄 arXiv](https://arxiv.org/abs/2611.26444) • [📥 PDF](https://arxiv.org/pdf/2611.26444)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Mind2Act/Mind2Act_World)

> Excited to share Mind2Act World ! 🤖 We introduce a diagnostic benchmark that disentangles embodied reasoning, physical execution, and their closed-loop composition in robotic manipulation, revealing a critical gap between individual capabilities a...

</details>

<details>
<summary><b>3. Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26590) • [📄 arXiv](https://arxiv.org/abs/2611.26590) • [📥 PDF](https://arxiv.org/pdf/2611.26590)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Vision-language models (VLMs) integrate visual and textual information, but their outputs do not reveal internal concept representations. Global workspace theory (GWT) proposes that information is selected from parallel brain processes, integrated...

</details>

<details>
<summary><b>4. WeMM-Video: Pushing the Frontiers of Open-Source Omni-Modal Video Understanding</b> ⭐ 55</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26409) • [📄 arXiv](https://arxiv.org/abs/2611.26409) • [📥 PDF](https://arxiv.org/pdf/2611.26409)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/WeMM-Video)

> Technical report of the WeMM-Video family.

</details>

<details>
<summary><b>5. Posterior Matching Distillation</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26629) • [📄 arXiv](https://arxiv.org/abs/2611.26629) • [📥 PDF](https://arxiv.org/pdf/2611.26629)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yesiltepe-hidir/PMD/issues/1) • [⭐ Code](https://github.com/yesiltepe-hidir/PMD)

> Project Page: https://pmd-distillation.github.io/

</details>

<details>
<summary><b>6. Coda-DaVinci: Reimagining Video Generation with Code2Video</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26595) • [📄 arXiv](https://arxiv.org/abs/2611.26595) • [📥 PDF](https://arxiv.org/pdf/2611.26595)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/FrankYang-17/Coda) • [⭐ Code](https://github.com/FrankYang-17/Coda/issues/1)

> Code2Video uses agent-authored 3D proxies to guide video generation, enabling explicit control over structure and dynamics. However, how to construct proxies and prompts for optimal downstream quality remains underexplored, as do the capabilities ...

</details>

<details>
<summary><b>7. The Mask Is Not the Object: Volumetric Supervision for 3D Gaussian Segmentation</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21383) • [📄 arXiv](https://arxiv.org/abs/2611.21383) • [📥 PDF](https://arxiv.org/pdf/2611.21383)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jhxcho/voda)

> No abstract available.

</details>

<details>
<summary><b>8. Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth</b> ⭐ 205</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26603) • [📄 arXiv](https://arxiv.org/abs/2611.26603) • [📥 PDF](https://arxiv.org/pdf/2611.26603)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/apple-aiml-research/ml-depth-pro-2)

> Depth Pro 2 estimates the depth of every pixel, in meters, from images taken with an ordinary camera. One network handles a single photo, a whole video, or a live camera feed, at the input's full resolution. Its depth is sharp at object edges and ...

</details>

<details>
<summary><b>9. EgoCaptioner: Towards Comprehensive and Fine-Grained Egocentric Video Captioning</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26380) • [📄 arXiv](https://arxiv.org/abs/2611.26380) • [📥 PDF](https://arxiv.org/pdf/2611.26380)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cactusqq/EgoCaptioner)

> Egocentric video captioning requires fine-grained detail and accurate timing, yet current multimodal large language models (MLLMs) often miss brief actions, confuse their order, or misinterpret the wearer's motion. Existing caption formats also of...

</details>

<details>
<summary><b>10. HiUni: Harmonizing Understanding and Generation in Shared Visual Space</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26107) • [📄 arXiv](https://arxiv.org/abs/2611.26107) • [📥 PDF](https://arxiv.org/pdf/2611.26107)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ernie-research/HiUni)

> HiUni shares the same visual encoder and dense backbone across understanding, generation, and editing. Language understanding is handled by a language modeling head, while image synthesis is performed by a Pixel Diffusion Head. To bridge semantic ...

</details>

<details>
<summary><b>11. TuringSpace: Grounding Spatial Intelligence in Action with Interactive CAPTCHAs</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25387) • [📄 arXiv](https://arxiv.org/abs/2611.25387) • [📥 PDF](https://arxiv.org/pdf/2611.25387)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/os-cua-framework/TuringSpace)

> MLLMs perform well on 2D perception but remain far behind humans in 3D spatial understanding, which makes space a natural basis for CAPTCHAs now that computer-use agents solve conventional ones. Yet existing spatial CAPTCHAs and benchmarks are mos...

</details>

<details>
<summary><b>12. AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26073) • [📄 arXiv](https://arxiv.org/abs/2611.26073) • [📥 PDF](https://arxiv.org/pdf/2611.26073)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> AltRA-Test is a statistical test of whether an LLM judge can replace human annotators, and it needs no shared annotators, only three human annotations per item.

</details>

<details>
<summary><b>13. ZeroVLN: Learning to Navigate without Supervised Fine-Tuning of the VLM Brain</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26414) • [📄 arXiv](https://arxiv.org/abs/2611.26414) • [📥 PDF](https://arxiv.org/pdf/2611.26414)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hzx122/ZeroVLN)

> We introduce ZeroVLN， the first slow–fast vision-language navigation framework with zero navigation-domain supervised fine-tuning of the VLM brain , and the first to jointly optimize the brain and a continuous-action expert through closed-loop age...

</details>

<details>
<summary><b>14. WAM-Policy: World Action Models with On-Policy Distillation for Quadrupedal Loco-Manipulation</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.24011) • [📄 arXiv](https://arxiv.org/abs/2611.24011) • [📥 PDF](https://arxiv.org/pdf/2611.24011)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIGeeksGroup/WAM-Policy)

> open-sourced.

</details>

<details>
<summary><b>15. RL Post-Training in Diffusion Transformers Is Activation Steering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Duen Horng Chau, Atakan Kara, Alec Helbling, Kaan Oktay, tmeral

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26627) • [📄 arXiv](https://arxiv.org/abs/2611.26627) • [📥 PDF](https://arxiv.org/pdf/2611.26627)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> RL post-training of diffusion transformers is mostly a fixed shift. Fixed steering vectors on the block outputs of the frozen base model, identical for every prompt, token and seed, recover 90.0–96.3% of the reward gain of six RL-tuned image model...

</details>

<details>
<summary><b>16. WM^MLLM: Turning MLLMs into World Models</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.24193) • [📄 arXiv](https://arxiv.org/abs/2611.24193) • [📥 PDF](https://arxiv.org/pdf/2611.24193)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ztMotaLee/WM_mllm)

> WM^MLLM rethinks world modeling by moving the world state out of the generator. The scene is stored as an explicit panorama with version history, and MLLMs plan, place and verify each edit. This makes consistency a built-in guarantee: revisits are...

</details>

<details>
<summary><b>17. Training Vision Models for LLM-Driven Weight-Space Programming</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sebastian Kranz, ropinski

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26408) • [📄 arXiv](https://arxiv.org/abs/2611.26408) • [📥 PDF](https://arxiv.org/pdf/2611.26408)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Training Process to improve Edibility in Weight Space Programming by Timo Ropinski & Sebastian Kranz

</details>

<details>
<summary><b>18. Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26268) • [📄 arXiv](https://arxiv.org/abs/2611.26268) • [📥 PDF](https://arxiv.org/pdf/2611.26268)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Tencent/LightningMoBA)

> Sparse attention improves efficiency by reducing unnecessary computation. On GPUs, however, sparse-index construction, irregular memory access, and task scheduling can offset part of those savings. MoBA reduces computation by routing each query to...

</details>

<details>
<summary><b>19. Learning What Actions Change: Action Effect Matching for Dynamics-Faithful World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jinwoo Shin, Yonghoon Dong, periphanes

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25980) • [📄 arXiv](https://arxiv.org/abs/2611.25980) • [📥 PDF](https://arxiv.org/pdf/2611.25980)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Abstract World models aim to predict environment dynamics under actions, enabling the prediction of action consequences without executing them in the real world. Yet despite producing increasingly realistic videos, current world models often fail ...

</details>

<details>
<summary><b>20. On-Policy Morphing into Hybrid Attention Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.24809) • [📄 arXiv](https://arxiv.org/abs/2611.24809) • [📥 PDF](https://arxiv.org/pdf/2611.24809)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hybrid attention models combine full and linear attention to achieve a better quality--efficiency trade-off. Converting pretrained Transformers into such models requires identifying which layers should retain full attention under a prescribed budg...

</details>

<details>
<summary><b>21. AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26510) • [📄 arXiv](https://arxiv.org/abs/2611.26510) • [📥 PDF](https://arxiv.org/pdf/2611.26510)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/RUCAIBox/AutoMaster)

> Task success does not guarantee good teaching . Even successful agent trajectories may contain undesirable behaviors that SFT can inadvertently reinforce. We introduce AutoMaster , an autonomous post-training framework that makes the same model a ...

</details>

<details>
<summary><b>22. Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Ilya Gusev

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26286) • [📄 arXiv](https://arxiv.org/abs/2611.26286) • [📥 PDF](https://arxiv.org/pdf/2611.26286)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/IlyaGusev/twicetold)

> This is an analytical paper about side effects of LLM use in reviewing papers in major conferences and journals. One of the main findings is that reviewers of the same paper now repeat each other’s critiques more often. This overlap rose sharply a...

</details>

<details>
<summary><b>23. Online Mistake Detection in Instruction-Conditioned Procedural Video</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25567) • [📄 arXiv](https://arxiv.org/abs/2611.25567) • [📥 PDF](https://arxiv.org/pdf/2611.25567)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TL;DR: We tackle online, instruction-conditioned mistake detection in egocentric procedural videos—catching errors as they happen rather than offline after the task is done. Key takeaways: Joint vs. Modular: Predicting step ID, temporal boundaries...

</details>

<details>
<summary><b>24. CompoundingKernel: Self-Improving from Zero to First</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.19230) • [📄 arXiv](https://arxiv.org/abs/2611.19230) • [📥 PDF](https://arxiv.org/pdf/2611.19230)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A self-improving agent system that writes GPU kernels by search, starting from only the PyTorch spec. As of Nov 11, 2026 it was #1 on all three public SOL-ExecBench leaderboards (L1, L2, Quantization) on NVIDIA B200: all 235 kernels correct on the...

</details>

<details>
<summary><b>25. Language Models Can Rewrite Their Own Weights</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xi Zhu, Andy Mao, Changyi Yang, Neil He, jiajunfanthu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21885) • [📄 arXiv](https://arxiv.org/abs/2611.21885) • [📥 PDF](https://arxiv.org/pdf/2611.21885)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A language model rewrites its own weights by its own reasoning and gets better, with no gradients and no training. It diagnoses its own failures, keeps improving across sessions, and repairs a damaged copy of itself. Demos and videos: https://www....

</details>

<details>
<summary><b>26. Embeddings vs. Grep: Rethinking the Dense Advantage in Code Search</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.20455) • [📄 arXiv](https://arxiv.org/abs/2611.20455) • [📥 PDF](https://arxiv.org/pdf/2611.20455)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Code-search benchmarks show dense encoders well ahead of BM25, yet coding agents do well with grep. What does this dense advantage measure? Existing benchmarks cannot tell: they use curated queries, old code and a handicapped BM25. To find out, we...

</details>

<details>
<summary><b>27. Accord: Universal Online Metric Localization and Dense Mapping from one Camera and an IMU</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25279) • [📄 arXiv](https://arxiv.org/abs/2611.25279) • [📥 PDF](https://arxiv.org/pdf/2611.25279)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/be2rlab/ACCORD)

> Online Universal Dense Metric Mapping from Any Camera with an IMU

</details>

<details>
<summary><b>28. TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26248) • [📄 arXiv](https://arxiv.org/abs/2611.26248) • [📥 PDF](https://arxiv.org/pdf/2611.26248)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJUSCL/TwinLab-Bench)

> Excited to share TwinLab-Bench! 🔬 We evaluate AI scientists on experiments they design themselves, across 52 tasks from real labs in the physical sciences, life sciences, and engineering. In each task, a TwinLab agent grounded in real lab records ...

</details>

<details>
<summary><b>29. VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.22666) • [📄 arXiv](https://arxiv.org/abs/2611.22666) • [📥 PDF](https://arxiv.org/pdf/2611.22666)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/suho-yoo/VSpark)

> Our aim is to accelerate autoregressive vision-language-action models (VLAs), which generate robot actions one token at a time. Speculative decoding makes this possible: a lightweight draft model proposes several tokens at once, and the VLA verifi...

</details>

<details>
<summary><b>30. System 1 agents: a method for local, reliable and fast agentic AI</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Eleftherios Tzanis, mklont

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.21053) • [📄 arXiv](https://arxiv.org/abs/2611.21053) • [📥 PDF](https://arxiv.org/pdf/2611.21053)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/eltzanis/SystemOneAgents)

> Turning each tool call into small multiple-choice questions, so the agent never has to invent a value, is a neat way to deal with hallucination in clinical settings. Running it locally also keeps patient data off a cloud model. How does it handle ...

</details>

<details>
<summary><b>31. Direct3D-V3: Latent-Free Diffusion Transformers in the Native Voxel Domain</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26511) • [📄 arXiv](https://arxiv.org/abs/2611.26511) • [📥 PDF](https://arxiv.org/pdf/2611.26511)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DreamTechAI/Direct3D-V3)

> Project Page: https://dreamtechai.github.io/Direct3D-V3 Code: https://github.com/DreamTechAI/Direct3D-V3

</details>

<details>
<summary><b>32. A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights</b> ⭐ 0</summary>

<br/>

**👥 Authors:** TanmaySah

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26450) • [📄 arXiv](https://arxiv.org/abs/2611.26450) • [📥 PDF](https://arxiv.org/pdf/2611.26450)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Why teach an LLM arithmetic when you can compile arithmetic directly into its weights? TL;DR: From learning weights to programming weights. We introduce closed-form, zero-gradient compilation primitives that embed deterministic computation directl...

</details>

<details>
<summary><b>33. Phantom Success, Silent Failure: Auditing the Success Checkers of Simulated Manipulation Benchmarks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25179) • [📄 arXiv](https://arxiv.org/abs/2611.25179) • [📥 PDF](https://arxiv.org/pdf/2611.25179)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Are your robot policies really succeeding, or is the success checker lying? 🤖 Simulated manipulation benchmarks rely on hand-written success checkers, which are usually validated only against expert demos. Learned policies behave differently, so c...

</details>

<details>
<summary><b>34. Do Probing and Steering Favor the Same Directions Within and Across Language Models?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.19499) • [📄 arXiv](https://arxiv.org/abs/2611.19499) • [📥 PDF](https://arxiv.org/pdf/2611.19499)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> The direction that best detects a concept in an LLM is rarely the one that best steers it: 59 of 60 settings across 3 model families. Interactive explainer: https://www.afshn.com/research/probing-vs-steering/

</details>

<details>
<summary><b>35. DailySafe: Benchmarking Dynamic Safety Awareness in Everyday Multi-Turn Multimodal Interactions</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Siyuan Liang, Ziheng Peng, Wenzhang Jiang, Xinhao Deng, XuankunRong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26298) • [📄 arXiv](https://arxiv.org/abs/2611.26298) • [📥 PDF](https://arxiv.org/pdf/2611.26298)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/XuankunRong/DailySafe)

> As multimodal large language models (MLLMs) become increasingly integrated into everyday life, their ability to provide safe and reliable assistance in realistic daily activities remains insufficiently explored. Existing safety benchmarks primaril...

</details>

<details>
<summary><b>36. NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.23835) • [📄 arXiv](https://arxiv.org/abs/2611.23835) • [📥 PDF](https://arxiv.org/pdf/2611.23835)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Speculative decoding is widely used to speed up generation in large language models (LLMs), with a small draft model proposing tokens that one forward pass of the target model verifies. Draft models that read the target's hidden states, such as DF...

</details>

<details>
<summary><b>37. Teaching Vision-Language-Action Models to Stop Safely</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26239) • [📄 arXiv](https://arxiv.org/abs/2611.26239) • [📥 PDF](https://arxiv.org/pdf/2611.26239)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/franciscoliu/SAFI)

> Vision-language-action (VLA) models enable robots to perform diverse tasks from visual observations and language instructions. While prior work has explored task suppression and refusal, stopping prohibited behavior once execution is underway and ...

</details>

<details>
<summary><b>38. PAD: Policy-Aware Data Valuation for Autonomous Driving</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Shiyu Fang, Yiming Cui, Jiaqi Liu, Yicheng Guo, ChengkaiXu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25604) • [📄 arXiv](https://arxiv.org/abs/2611.25604) • [📥 PDF](https://arxiv.org/pdf/2611.25604)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/PerfectXu88/PAD)

> A different way to scale. A demonstration’s value depends on the policy learning from it. PAD connects real driving outcomes to the learning signal of each demonstration. The result: a synthetic dataset selected for your planner, with every retain...

</details>

<details>
<summary><b>39. Suppressing the Shortcut Hidden Behind the Gains of VideoLLM Fine-Tuning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Junmo Kim, Jongsuk Kim, yshin0917

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25776) • [📄 arXiv](https://arxiv.org/abs/2611.25776) • [📥 PDF](https://arxiv.org/pdf/2611.25776)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Youngwoo-git/VETO)

> SFT raises the accuracy of VideoLLMs, yet behind this rise the model increasingly relies on textual regularities of the training data, a shortcut that bypasses the video. VETO suppresses this shortcut by projecting the fine-tuning update onto the ...

</details>

<details>
<summary><b>40. Squoosh Bench: Evaluating UX-1 on 318 Historical Website A/B Tests</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Suzan Halim, Viraj Shah, Johari Dramiga, Rohith Pasula, dooskin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25200) • [📄 arXiv](https://arxiv.org/abs/2611.25200) • [📥 PDF](https://arxiv.org/pdf/2611.25200)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> AI has made building a website change take minutes, but finding out whether customers like it still takes weeks of live testing. Squoosh simulates how people use a site, so teams can see what their customers will likely do before anything goes liv...

</details>

<details>
<summary><b>41. In Situ Training of Optical Neural Networks by Coding Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.23875) • [📄 arXiv](https://arxiv.org/abs/2611.23875) • [📥 PDF](https://arxiv.org/pdf/2611.23875)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Training an optical neural network is our first demonstration, but the agent only needs a way to set parameters and a way to measure the result. I am curious what else people would try this on.

</details>

<details>
<summary><b>42. SpecWM: Turning Verification into Refinement for Streaming World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26211) • [📄 arXiv](https://arxiv.org/abs/2611.26211) • [📥 PDF](https://arxiv.org/pdf/2611.26211)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Recently, video world models have advanced interactive generation, with large models offering higher quality and smaller models enabling faster inference. Existing speculative methods combine their strengths through fast draft generation and selec...

</details>

<details>
<summary><b>43. One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26617) • [📄 arXiv](https://arxiv.org/abs/2611.26617) • [📥 PDF](https://arxiv.org/pdf/2611.26617)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/showlab/O2LaF)

> Project page: https://showlab.github.io/o2laf/ Code: https://github.com/showlab/O2LaF

</details>

<details>
<summary><b>44. Inhabitant: An Agent Framework for Long-Horizon Human-Scene Interaction with Spatial Control and Quantitative Feedback</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yu Xiao, Yao Zhang, Ryenhails

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25005) • [📄 arXiv](https://arxiv.org/abs/2611.25005) • [📥 PDF](https://arxiv.org/pdf/2611.25005)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yaozhang182/inhabitant)

> Inhabitant is a training-free agent framework for long-horizon human-scene interaction. An LLM agent writes spatial control for a pretrained motion generator, measures the executed motion in a scene simulator, and revises until the task is done. O...

</details>

<details>
<summary><b>45. S1CogBench: What System One Decision Models Can and Cannot Do on Cognitive Experiments</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kazunori D Yamada, dengzhe-hou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.00911) • [📄 arXiv](https://arxiv.org/abs/2611.00911) • [📥 PDF](https://arxiv.org/pdf/2611.00911)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/dengzhe-hou/S1CogBench)

> S1CogBench tests System One decision models, which return one fast typed decision over the legal options, on 24 established cognitive experiments (27 modes), with seven open models and the Jev API. Acting well does not imply predicting humans. In ...

</details>

<details>
<summary><b>46. PlayCraft: Designing Engaging Games from Agent Play Experience</b> ⭐ 4</summary>

<br/>

**👥 Authors:** tokenid

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26624) • [📄 arXiv](https://arxiv.org/abs/2611.26624) • [📥 PDF](https://arxiv.org/pdf/2611.26624)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/xt4d/PlayCraft)

> PlayCraft is an automated framework for designing engaging games that give players a path from beginner play to mastery. PlayCraft alternates between playing and crafting: A player agent watches the game screen, uses the controls, and pursues the ...

</details>

<details>
<summary><b>47. Time Series Structure Learning with Local Causal Parent Discovery</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26232) • [📄 arXiv](https://arxiv.org/abs/2611.26232) • [📥 PDF](https://arxiv.org/pdf/2611.26232)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jmaasch/itpd)

> Iterative Temporal Parent Discovery (ITPD) is a constraint-based causal discovery algorithm for multivariate time series.

</details>

<details>
<summary><b>48. Lang-SVG: Hierarchical Image Vectorization with Language Priors</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.15941) • [📄 arXiv](https://arxiv.org/abs/2611.15941) • [📥 PDF](https://arxiv.org/pdf/2611.15941)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>49. AURA: AUdio-dRiven streaming Avatar</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.26540) • [📄 arXiv](https://arxiv.org/abs/2611.26540) • [📥 PDF](https://arxiv.org/pdf/2611.26540)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kyutai-labs/aura)

> A live demo is available at https://aura.kyutai.org/

</details>

<details>
<summary><b>50. TwinD: Generalizing Deepfake Detection from Blending to Entire Face Synthesis</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jiri Matas, Mario Fritz, Jan Cech, yermandy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.25935) • [📄 arXiv](https://arxiv.org/abs/2611.25935) • [📥 PDF](https://arxiv.org/pdf/2611.25935)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/yermandy/TwinD)

> TwinD: Generalizing Deepfake Detection from Blending to Entire Face Synthesis

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 50 |
| 📅 Today | [`2026-11-21.json`](data/daily/2026-11-21.json) | 50 |
| 📆 This Week | [`2026-W46.json`](data/weekly/2026-W46.json) | 134 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 413 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-21 | 50 | [View JSON](data/daily/2026-11-21.json) |
| 📄 2026-11-20 | 18 | [View JSON](data/daily/2026-11-20.json) |
| 📄 2026-11-19 | 27 | [View JSON](data/daily/2026-11-19.json) |
| 📄 2026-11-18 | 17 | [View JSON](data/daily/2026-11-18.json) |
| 📄 2026-11-17 | 10 | [View JSON](data/daily/2026-11-17.json) |
| 📄 2026-11-16 | 12 | [View JSON](data/daily/2026-11-16.json) |
| 📄 2026-11-15 | 34 | [View JSON](data/daily/2026-11-15.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W46 | 134 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 413 | [View JSON](data/monthly/2026-11.json) |
| 🗓️ 2026-10 | 1120 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |

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
