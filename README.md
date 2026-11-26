<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-30-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-8307+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">30</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">116</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">579</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">8307+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 26, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. PyRUA-Dex: Learning Programmatic Policies for Dexterous Bimanual Manipulation with Coding Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33144) • [📄 arXiv](https://arxiv.org/abs/2611.33144) • [📥 PDF](https://arxiv.org/pdf/2611.33144)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DAGroup-PKU/PyRUA-Lean/tree/dex)

> Github Repo: https://github.com/DAGroup-PKU/PyRUA-Lean/tree/dex

</details>

<details>
<summary><b>2. Sage-Med: Scaling Medical Multimodal Intelligence</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34575) • [📄 arXiv](https://arxiv.org/abs/2611.34575) • [📥 PDF](https://arxiv.org/pdf/2611.34575)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Sage-Med/Sage-Med-Preview)

> Hi everyone! 👋 We are the Sage-Med Team , excited to share our paper: Sage-Med: Scaling Medical Multimodal Intelligence . TL;DR: We benchmarked 77 models across 238 clinical tasks (61 public + 13 private multi-center benchmarks from 14 hospitals a...

</details>

<details>
<summary><b>3. VibeVideoAgent: Closing the Perceptual Loop for Code-Driven Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34237) • [📄 arXiv](https://arxiv.org/abs/2611.34237) • [📥 PDF](https://arxiv.org/pdf/2611.34237)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Awesome Agent solution for code2video tasks～～～

</details>

<details>
<summary><b>4. Improving and Expanding Video Editing Capabilities Without Videos</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34941) • [📄 arXiv](https://arxiv.org/abs/2611.34941) • [📥 PDF](https://arxiv.org/pdf/2611.34941)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HorizonWind2004/i2v-transfer)

> Fully Open-sourced! We introduce Image-to-Video Transfer (I2VT), which improves pretrained video editors and expands them to new editing tasks using an image editor. Our intuition is that an image is simply a one-frame video, so editing images wel...

</details>

<details>
<summary><b>5. DyLoMo-DiT: Dynamic Looped Mixture-of-Experts Diffusion Transformers</b> ⭐ 35</summary>

<br/>

**👥 Authors:** Wei Zhang, Pengfei Xiong, Yuanliu Liu, OneCreamBox

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30844) • [📄 arXiv](https://arxiv.org/abs/2611.30844) • [📥 PDF](https://arxiv.org/pdf/2611.30844)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OneBoxCream/DyLoMo-DiT)

> DyLoMo-DiT is a new diffusion Transformer architecture that gets more out of the same parameters: the middle layers run in a weight-shared loop, sparse experts decide which parameters each image token uses, and the number of loops changes with the...

</details>

<details>
<summary><b>6. HOG-WAM: Improving Visual Robustness through Future Gradient Prediction</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34811) • [📄 arXiv](https://arxiv.org/abs/2611.34811) • [📥 PDF](https://arxiv.org/pdf/2611.34811)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/iPPLab/HOG-WAM)

> No abstract available.

</details>

<details>
<summary><b>7. AIRIS: Benchmarking Social Embodied Intelligence in Egocentric Streaming Videos</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34347) • [📄 arXiv](https://arxiv.org/abs/2611.34347) • [📥 PDF](https://arxiv.org/pdf/2611.34347)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIRIS-bench/AIRIS)

> Embodied agents operating around people must not only recognize social cues, but also decide when the available evidence is sufficient, whether to intervene, and how to respond without violating social norms. Existing benchmarks typically evaluate...

</details>

<details>
<summary><b>8. Target-Part Preference Optimization for Part-Aware 3D Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Lina Yao, Sevvandi Kandanaarachchi, Zeyu Zhang, Rui Li, Zishan Qin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30472) • [📄 arXiv](https://arxiv.org/abs/2611.30472) • [📥 PDF](https://arxiv.org/pdf/2611.30472)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi everyone! I’m Taylor, the first author of PartDPO. We study how to improve part-aware 3D generation with limited part-level annotations. Our approach learns preferences over individual parts while retaining geometric supervision, without changi...

</details>

<details>
<summary><b>9. FoundSurface: Feedforward Ray-Based 3D Scene Surface Reconstruction from Unposed Images</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34128) • [📄 arXiv](https://arxiv.org/abs/2611.34128) • [📥 PDF](https://arxiv.org/pdf/2611.34128)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/vLAR-group/FoundSurface)

> FoundSurface (NeurIPS'26) is a foundation surface reconstruction model. Jointly trained on 9 diverse large-scale datasets, it recovers 3D scene surfaces from unposed images in a single feed-forward pass. For every pixel of a target view, it predic...

</details>

<details>
<summary><b>10. Agentic LongLive: Asynchronous Agentic Memory for Long Video Generation</b> ⭐ 2.91k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34907) • [📄 arXiv](https://arxiv.org/abs/2611.34907) • [📥 PDF](https://arxiv.org/pdf/2611.34907)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NVlabs/LongLive)

> No abstract available.

</details>

<details>
<summary><b>11. ROAM-Bench: Evaluating Embodied Active Spatial Intelligence through Evidence Verification</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33143) • [📄 arXiv](https://arxiv.org/abs/2611.33143) • [📥 PDF](https://arxiv.org/pdf/2611.33143)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/rxjfighting/ROAM-Bench)

> Introducing ROAM-Bench, a benchmark for evaluating embodied active spatial intelligence through evidence verification! As illustrated in the teaser, counting mugs can require exploring multiple rooms and opening closed storage. We check both the f...

</details>

<details>
<summary><b>12. iMFPO: Improving Mean Flow Policy Optimization with Drift-Aware Velocity Alignment for Embodied Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33922) • [📄 arXiv](https://arxiv.org/abs/2611.33922) • [📥 PDF](https://arxiv.org/pdf/2611.33922)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIGeeksGroup/iMFPO)

> 🚀 We’re excited to introduce iMFPO , a one-step generative policy optimization framework for efficient embodied control! By combining drift-aware velocity alignment with reward-driven policy optimization, iMFPO enables direct action generation wit...

</details>

<details>
<summary><b>13. What Is Redundant at High Resolution? Efficient Attention and Feed-Forward Design for All-in-One Image Restoration</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33636) • [📄 arXiv](https://arxiv.org/abs/2611.33636) • [📥 PDF](https://arxiv.org/pdf/2611.33636)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/dslisleedh/HiGLR)

> Hi everyone! 👋 I'm the first author of this paper. TL;DR: We ask what is actually redundant at high resolution in multi-scale all-in-one image restoration (AIO-IR) networks, and use the answer to build HiGLR , a fast and memory-efficient AIO-IR mo...

</details>

<details>
<summary><b>14. RoboEvo^2: Recursive Robot Self-Improvement via Harness-Policy Co-Evolution</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Haoyuan Li, Xuzhao Huang, Wei Huang, Jiehong Lin, Chenzhi Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34675) • [📄 arXiv](https://arxiv.org/abs/2611.34675) • [📥 PDF](https://arxiv.org/pdf/2611.34675)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/kaiknower/RoboEvo2)

> Policy failures tell a coding agent what to repair; verified repairs become agent skills and policy training data. Over rounds, the policy succeeds more on its own and the agent is called less, with no new human demonstrations.

</details>

<details>
<summary><b>15. EgoUseBench: Benchmarking Computer-Use Agents under Evolving First-Person Contexts</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34466) • [📄 arXiv](https://arxiv.org/abs/2611.34466) • [📥 PDF](https://arxiv.org/pdf/2611.34466)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/YurunChen/EgoUseBench)

> AI wearables, such as smart glasses, extend the context available to agents beyond digital environments to users' everyday physical surroundings. With these devices, computer tasks can originate from users' ongoing real-world activities rather tha...

</details>

<details>
<summary><b>16. High-Fidelity Image Matting with Pixel-Space Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuheng Liu, Baiang Li, Yufan Wei, Xin Lin, Baihan Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34929) • [📄 arXiv](https://arxiv.org/abs/2611.34929) • [📥 PDF](https://arxiv.org/pdf/2611.34929)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/White-YC/pixelmatte)

> Project page: https://white-yc.github.io/pixelmatte/ .

</details>

<details>
<summary><b>17. Loupe: Reinforcing Knowledge-Intensive Video Generation Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.31895) • [📄 arXiv](https://arxiv.org/abs/2611.31895) • [📥 PDF](https://arxiv.org/pdf/2611.31895)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present Loupe, an agentic framework for knowledge-intensive video generation that addresses the challenges of missing factual knowledge, entity appearances, and temporal relationships in user prompts. Loupe autonomously leverages text search, i...

</details>

<details>
<summary><b>18. Beyond Code and Pixels: Co-Generating Editable and Visually Rich Presentations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.28010) • [📄 arXiv](https://arxiv.org/abs/2611.28010) • [📥 PDF](https://arxiv.org/pdf/2611.28010)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Generating PPTX presentations is a structured task that requires semantic planning, visual composition, precise layout, and editability, motivating collaboration among specialized agents. However, programmatic methods offer editability but often p...

</details>

<details>
<summary><b>19. SEEK: Self-Evolution through Execution Knowledge for Agentic Robot Manipulation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33915) • [📄 arXiv](https://arxiv.org/abs/2611.33915) • [📥 PDF](https://arxiv.org/pdf/2611.33915)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZeyangGuan/SEEK-Agentic-Robot-Manipulation)

> General-purpose multimodal foundation models enable robots to adapt during deployment from observations and execution feedback, without task-specific demonstrations or training. However, the robot retains what it learns only in the model's working...

</details>

<details>
<summary><b>20. Beyond Self-Consistency: Recursive Self-Improvement at Test Time</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30965) • [📄 arXiv](https://arxiv.org/abs/2611.30965) • [📥 PDF](https://arxiv.org/pdf/2611.30965)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/uanu2002/TT-RSI)

> Most test-time methods for LLMs vote : sample independent answers and trust the majority, both to pick the final answer and to build training targets. That wastes earlier reasoning, and when the majority is wrong, training on it reinforces the err...

</details>

<details>
<summary><b>21. Beyond the Happy Path: Generalizing Robot World Models from Simulated Failures</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33925) • [📄 arXiv](https://arxiv.org/abs/2611.33925) • [📥 PDF](https://arxiv.org/pdf/2611.33925)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/aim-uofa/WorldReforge)

> Robot world models are increasingly used to evaluate and plan policies, yet robot training data are dominated by successful demonstrations. Off the happy path, when a grasp is misaligned or mistimed and the object is pushed, tilted, or slips, thei...

</details>

<details>
<summary><b>22. OmniPersist: Persistent Dynamic World Modeling via Coupled Panoramic Evolution</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.32473) • [📄 arXiv](https://arxiv.org/abs/2611.32473) • [📥 PDF](https://arxiv.org/pdf/2611.32473)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/cszhilu1998/OmniPersist)

> The camera turns away. The world should go on. OmniPersist couples perspective video generation with an evolving sparse panoramic world to preserve dynamic subjects beyond the field of view. The generated perspective video and corresponding panora...

</details>

<details>
<summary><b>23. Puzzling-Bench: Evaluating Language Models on 23,777 Human-Written Puzzles with Community-Verified Answers</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Bo Ni, Sai Karthik Navuluru, Chien Van Nguyen, Li Li, Abhinav Sharma

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34331) • [📄 arXiv](https://arxiv.org/abs/2611.34331) • [📥 PDF](https://arxiv.org/pdf/2611.34331)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/a1bhinav/puzzling-bench)

> No abstract available.

</details>

<details>
<summary><b>24. Real2SimLab: A Unified Platform for Real-to-Simulation Transfer</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34849) • [📄 arXiv](https://arxiv.org/abs/2611.34849) • [📥 PDF](https://arxiv.org/pdf/2611.34849)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Real2SimHub/Real2SimLab)

> Real2SimHub: a community to build the infrastructure of real-to-simulation together.

</details>

<details>
<summary><b>25. 3D CoCa v3: Contrastive Learners with Scene Graphs for Urban Scene Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33835) • [📄 arXiv](https://arxiv.org/abs/2611.33835) • [📥 PDF](https://arxiv.org/pdf/2611.33835)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/AIGeeksGroup/3DCoCav3)

> We present 3D CoCa v3 , a geometry-aware framework for urban 3D scene understanding. It combines a Pyramid Vision Encoder with Multi-view Scene Graph (MSG) supervision to enhance multi-view geometric representation learning and 3D-language alignme...

</details>

<details>
<summary><b>26. CONCOR: Benchmarking Contrastive Decoding for Visual Hallucination Mitigation Beyond Final-Output Performance</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.30460) • [📄 arXiv](https://arxiv.org/abs/2611.30460) • [📥 PDF](https://arxiv.org/pdf/2611.30460)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce CONCOR (Contrastive Correction Evaluation), a controlled benchmark that directly evaluates the competition between ground-truth and different types of hallucinated candidates using the Contrastive Grounding Margin (CGM).

</details>

<details>
<summary><b>27. LiveVAE: Causal Window Attention Makes Efficient Streaming Video VAE</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhiyuan Li, Hongyu Liu, Le Li, Sijie Zhao, Jiachen An

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34912) • [📄 arXiv](https://arxiv.org/abs/2611.34912) • [📥 PDF](https://arxiv.org/pdf/2611.34912)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MeiGen-AI/LiveVAE)

> LiveVAE: a streaming video VAE built on causal window attention. Distilled from pretrained video VAEs (MiniMax-H3, Wan2.2), it keeps their latent space and replaces both encoder and decoder directly, running up to 13.4× faster end to end.

</details>

<details>
<summary><b>28. SIA: Selective Image Analogy</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.33894) • [📄 arXiv](https://arxiv.org/abs/2611.33894) • [📥 PDF](https://arxiv.org/pdf/2611.33894)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/alimohammadiamirhossein/selective-image-analogy)

> One before/after example often holds several edits, and image analogy methods give you no say in which ones transfer. SIA lets you pick which edits carry over to your image and leave the rest out, from the same example. Try the interactive demo on...

</details>

<details>
<summary><b>29. SPARC: Geometry-Guided Attention Pruning for Long-Horizon Camera-Controlled Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.34064) • [📄 arXiv](https://arxiv.org/abs/2611.34064) • [📥 PDF](https://arxiv.org/pdf/2611.34064)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zju3dv/SPARC)

> We present SPARC, a geometry-guided attention pruning framework that turns the native self-attention of pretrained video diffusion transformers into a unified mechanism for camera control and persistent scene memory. Our key insight is that attent...

</details>

<details>
<summary><b>30. Negative-Only Policy Optimization for One-Sided Verifiable Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17287) • [📄 arXiv](https://arxiv.org/abs/2611.17287) • [📥 PDF](https://arxiv.org/pdf/2611.17287)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> NeurIPS 2026 Spotlight A failed unit test tells us a program is wrong. Passing a few tests still leaves its correctness uncertain. Can we train a policy using only the failures we trust? We introduce Negative-Only Policy Optimization (NOPO), which...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 30 |
| 📅 Today | [`2026-11-26.json`](data/daily/2026-11-26.json) | 30 |
| 📆 This Week | [`2026-W47.json`](data/weekly/2026-W47.json) | 116 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 579 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-26 | 30 | [View JSON](data/daily/2026-11-26.json) |
| 📄 2026-11-25 | 34 | [View JSON](data/daily/2026-11-25.json) |
| 📄 2026-11-24 | 40 | [View JSON](data/daily/2026-11-24.json) |
| 📄 2026-11-23 | 12 | [View JSON](data/daily/2026-11-23.json) |
| 📄 2026-11-22 | 50 | [View JSON](data/daily/2026-11-22.json) |
| 📄 2026-11-21 | 50 | [View JSON](data/daily/2026-11-21.json) |
| 📄 2026-11-20 | 18 | [View JSON](data/daily/2026-11-20.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W47 | 116 | [View JSON](data/weekly/2026-W47.json) |
| 📅 2026-W46 | 184 | [View JSON](data/weekly/2026-W46.json) |
| 📅 2026-W45 | 107 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 579 | [View JSON](data/monthly/2026-11.json) |
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
