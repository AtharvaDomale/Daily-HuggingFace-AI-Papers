<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-39-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7728+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">39</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">110</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">1120</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7728+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 31, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Intrinsic Distribution Matching</b> ⭐ 49</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39633) • [📄 arXiv](https://arxiv.org/abs/2610.39633) • [📥 PDF](https://arxiv.org/pdf/2610.39633)

**💻 Code:** [⭐ Code](https://github.com/inclusionAI/IDM) • [⭐ Code](https://github.com/huggingface)

> Hi everyone! We’re excited to share IDM: Intrinsic Distribution Matching . Can a generator provide both the score supervision and the representation used for distribution matching? Building on internal DMD, IDM retains the full DMD correction and ...

</details>

<details>
<summary><b>2. A-RSI: Adaptive Task Refresh for Zero-Data Multimodal Self-Improvement</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38568) • [📄 arXiv](https://arxiv.org/abs/2610.38568) • [📥 PDF](https://arxiv.org/pdf/2610.38568)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A-RSI improves recursive self-improvement by refreshing training tasks based on declining learning signals and using independently verifiable program-generated answers, raising benchmark performance without external labels.

</details>

<details>
<summary><b>3. Your Benchmark Is an Empirical Measure Over Difficulty</b> ⭐ 34</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.29927) • [📄 arXiv](https://arxiv.org/abs/2610.29927) • [📥 PDF](https://arxiv.org/pdf/2610.29927)

**💻 Code:** [⭐ Code](https://github.com/ASTRAL-Group/benchmark-difficulty-measure) • [⭐ Code](https://github.com/huggingface)

> The paper views a benchmark as an empirical measure over difficulty, which enables it to formalize measure mismatch and coverage gaps and derive theoretical results and practical diagnostics. The paper further introduces BenchFill, a difficulty-ta...

</details>

<details>
<summary><b>4. Youtu-Parsing-Omni: One Encoder, One Schema, All Modalities</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39301) • [📄 arXiv](https://arxiv.org/abs/2610.39301) • [📥 PDF](https://arxiv.org/pdf/2610.39301)

**💻 Code:** [⭐ Code](https://github.com/TencentCloudADP/youtu-parsing/tree/main/youtu_parsing_omni) • [⭐ Code](https://github.com/huggingface)

> Youtu-Parsing-Omni is a compact (5B) omni-modal parsing model. Given a single input — a document page, a natural image, a chart / flowchart, a geometry figure, an audio clip or an audio-visual video — it produces one structured JSON envelope that ...

</details>

<details>
<summary><b>5. Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.36671) • [📄 arXiv](https://arxiv.org/abs/2610.36671) • [📥 PDF](https://arxiv.org/pdf/2610.36671)

**💻 Code:** [⭐ Code](https://github.com/hyeon9698/LOWN) • [⭐ Code](https://github.com/huggingface)

> LOWN (Look Only When Needed) is a training-free decoding method for diffusion vision-language models (DVLMs) that processes the image only when a text-only prediction disagrees with the last image-conditioned prediction. Diffusion VLMs such as LLa...

</details>

<details>
<summary><b>6. Proving and Improving the Platonic Representation Hypothesis with JEPAs</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39302) • [📄 arXiv](https://arxiv.org/abs/2610.39302) • [📥 PDF](https://arxiv.org/pdf/2610.39302)

**💻 Code:** [⭐ Code](https://github.com/galilai-group/prh-proof) • [⭐ Code](https://github.com/huggingface)

> Authors here 👋 We prove the Platonic Representation Hypothesis for JEPAs: with good views (two observations of one scene), encoders trained separately on any modalities learn the same representation, up to one rotation. Next-token prediction is a ...

</details>

<details>
<summary><b>7. Read, Allocate, Route: Role-Coupled Tokenization for Pixel-Space Generative Transformers</b> ⭐ 31</summary>

<br/>

**👥 Authors:** Ziwei Liu, Zhongang Cai, Haiwen Diao, weepiess2383, Gokottaw434

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40647) • [📄 arXiv](https://arxiv.org/abs/2610.40647) • [📥 PDF](https://arxiv.org/pdf/2610.40647)

**💻 Code:** [⭐ Code](https://github.com/JiongfanYang/SpecSlot) • [⭐ Code](https://github.com/huggingface)

> We introduce SpecSlot, a role-coupled tokenization method for pixel-space generative Transformers. SpecSlot decouples fine visual detail from global communication by assigning tokens fixed spectral output roles: low-frequency carriers perform glob...

</details>

<details>
<summary><b>8. L3G-World: Real-Time Infinite World Generation with Latent 3D Gaussian Memory</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40614) • [📄 arXiv](https://arxiv.org/abs/2610.40614) • [📥 PDF](https://arxiv.org/pdf/2610.40614)

**💻 Code:** [⭐ Code](https://github.com/cvlab-kaist/L3G-World) • [⭐ Code](https://github.com/huggingface)

> This is an automated message from the ResearchStudio team. We created an interactive ResearchStudio Reel for this paper. It includes a visual poster , a video , and a blog , all available for download in editable formats. Open the ResearchStudio R...

</details>

<details>
<summary><b>9. WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.37161) • [📄 arXiv](https://arxiv.org/abs/2610.37161) • [📥 PDF](https://arxiv.org/pdf/2610.37161)

**💻 Code:** [⭐ Code](https://github.com/IQuestLab/WorkFactory) • [⭐ Code](https://github.com/huggingface)

> Excited to share WorkFactory, a scalable framework for synthesizing coherent and executable professional tasks to train Working Agents! WorkFactory combines professional intent synthesis, information support graphs, and execution-guided repair to ...

</details>

<details>
<summary><b>10. A Broader Look into MLLM's Retrieval Heads</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.32411) • [📄 arXiv](https://arxiv.org/abs/2610.32411) • [📥 PDF](https://arxiv.org/pdf/2610.32411)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We study how omni-modal LLMs retrieve relevant evidence, focusing on how retrieval mechanisms extend from text and vision to audio-visual inputs. Across 18 model checkpoints, including eight omni-modal models, we find that retrieval relies on a sp...

</details>

<details>
<summary><b>11. Co-Design Bench: Multimodal Collaborative Coding Evaluation for Agentic Models</b> ⭐ 34</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.35138) • [📄 arXiv](https://arxiv.org/abs/2610.35138) • [📥 PDF](https://arxiv.org/pdf/2610.35138)

**💻 Code:** [⭐ Code](https://github.com/Wiselnn570/Co-Design-Bench) • [⭐ Code](https://github.com/huggingface)

> Can a coding agent turn a real user's multimodal request into a complete, working application revision? We're sharing Co-Design Bench: Multimodal Collaborative Coding Evaluation for Agentic Models , a benchmark built around real users' requests to...

</details>

<details>
<summary><b>12. VibeGame: Prompt-to-Game Development with AI-Native Engine and Self-Evolving Adversarial Agent Team</b> ⭐ 294</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.33106) • [📄 arXiv](https://arxiv.org/abs/2610.33106) • [📥 PDF](https://arxiv.org/pdf/2610.33106)

**💻 Code:** [⭐ Code](https://github.com/tettethu/VibeGame) • [⭐ Code](https://github.com/huggingface)

> Excited to share VibeGame, a framework that turns prompts (text and optional reference images) into complete, playable 2D game projects! To let agents build a complete, vivid game with no given assets, we build an AI-native game engine that agents...

</details>

<details>
<summary><b>13. HairDeck: Animation-Ready Stylized Hair Cards from a Single Image</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40599) • [📄 arXiv](https://arxiv.org/abs/2610.40599) • [📥 PDF](https://arxiv.org/pdf/2610.40599)

**💻 Code:** [⭐ Code](https://github.com/hyz317/HairDeck) • [⭐ Code](https://github.com/huggingface)

> Try our demo! https://huggingface.co/spaces/hyz317/HairDeck

</details>

<details>
<summary><b>14. WeVisCut: The Whole Editing Room in One Agent From Narration to Long-Form Video</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38352) • [📄 arXiv](https://arxiv.org/abs/2610.38352) • [📥 PDF](https://arxiv.org/pdf/2610.38352)

**💻 Code:** [⭐ Code](https://github.com/Tencent/WeVisCut) • [⭐ Code](https://github.com/huggingface)

> WeVisCut connects tool execution with policy learning for long-form script-to-video production. Editing inversion and verified replay recover executable supervision from 3,750 finished projects, yielding 134,771 training examples for a single 9B p...

</details>

<details>
<summary><b>15. IntReal: A Real-World Dataset and Benchmark for Multimodal Interleaved Generation</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38425) • [📄 arXiv](https://arxiv.org/abs/2610.38425) • [📥 PDF](https://arxiv.org/pdf/2610.38425)

**💻 Code:** [⭐ Code](https://github.com/yushaohan/IntReal) • [⭐ Code](https://github.com/huggingface)

> Paper: https://arxiv.org/abs/2610.38425 Project Page: http://yushaohan.github.io/IntReal/ Dataset: https://huggingface.co/datasets/yushaohan/IntReal-100K Fine-tuned Models: https://huggingface.co/collections/yushaohan/intreal Leaderboard: https://...

</details>

<details>
<summary><b>16. CoSTA: Compressed Speech Token Adaptation for Efficient Spoken Dialogue Models</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.21143) • [📄 arXiv](https://arxiv.org/abs/2610.21143) • [📥 PDF](https://arxiv.org/pdf/2610.21143)

**💻 Code:** [⭐ Code](https://github.com/yingzhuodeng/CoSTA) • [⭐ Code](https://github.com/huggingface)

> Excited to share CoSTA , a post-hoc adaptation framework for efficient spoken dialogue models! Unlike conventional approaches that reduce speech token rates, CoSTA decouples generation and synthesis token spaces through reversible BPE-based compre...

</details>

<details>
<summary><b>17. WebAnything: From the Wild Web to Scalable Agent Environments</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Bingxuan Li, Pinxue Guo, Chaoyun Zhang, Zikun Cai, Mumuthanks

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.37786) • [📄 arXiv](https://arxiv.org/abs/2610.37786) • [📥 PDF](https://arxiv.org/pdf/2610.37786)

**💻 Code:** [⭐ Code](https://github.com/china-qijizhifeng/Web-Anything) • [⭐ Code](https://github.com/huggingface)

> Making any url as an web agent environment!

</details>

<details>
<summary><b>18. Multi-Token Prediction Done Right</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38331) • [📄 arXiv](https://arxiv.org/abs/2610.38331) • [📥 PDF](https://arxiv.org/pdf/2610.38331)

**💻 Code:** [⭐ Code](https://github.com/JaydenTeoh/Soft-MTP) • [⭐ Code](https://github.com/huggingface)

> We show the standard multi-token prediction (MTP) gradient hides a KL self-distillation gradient signal behind a noisy residual, and introduce the KL gradient–recovery trick to extract it. This is a free lunch that improves MTP across a range of t...

</details>

<details>
<summary><b>19. Investor's Last Exam: Can AI Agents Do Equity Investment Without Seeing the Future?</b> ⭐ 86</summary>

<br/>

**👥 Authors:** Rui Ge, Kai Sheng, Anning Hu, Huangfei Jiang, caiyuchen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38342) • [📄 arXiv](https://arxiv.org/abs/2610.38342) • [📥 PDF](https://arxiv.org/pdf/2610.38342)

**💻 Code:** [⭐ Code](https://github.com/Prophetlab/Investor-Last-Exam) • [⭐ Code](https://github.com/huggingface)

> Evaluating forecasting capability usually means waiting for outcomes to materialize — months, sometimes years — during which market conditions, fundamentals, and the information environment shift enough that scores can no longer be cleanly attribu...

</details>

<details>
<summary><b>20. Glance: The Generator Is Its Own Reranker</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40454) • [📄 arXiv](https://arxiv.org/abs/2610.40454) • [📥 PDF](https://arxiv.org/pdf/2610.40454)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Glance: one model. Judge, rank, generate. RAG usually serves two models: a cross-encoder reranker that encodes every (query, passage) pair, then a generator that reads the winners again. We show the generator can be its own reranker. All k passage...

</details>

<details>
<summary><b>21. AskStream: When and What to Ask Proactively under Uncertainty in Streaming Video Understanding</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38678) • [📄 arXiv](https://arxiv.org/abs/2610.38678) • [📥 PDF](https://arxiv.org/pdf/2610.38678)

**💻 Code:** [⭐ Code](https://github.com/wanglu-cs/AskStream) • [⭐ Code](https://github.com/huggingface)

> Recent advances in multimodal large language models have increased interest in interaction paradigms for video understanding. The evolution from offline to streaming and proactive interaction has gradually shifted models from passive responders to...

</details>

<details>
<summary><b>22. EditAgent: Rethinking Agentic Image Editing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39664) • [📄 arXiv](https://arxiv.org/abs/2610.39664) • [📥 PDF](https://arxiv.org/pdf/2610.39664)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Rubric-CEPR: Self-Evolving Image Editing via Reward-Verified Self-Distillat...

</details>

<details>
<summary><b>23. PhysInsert: Physically Plausible Object Insertion with Geometric Priors and Bidirectional Object-Scene Interactions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaochao Qu, Song Chang, Rui Chen, Le Zhang, Arthuringy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40453) • [📄 arXiv](https://arxiv.org/abs/2610.40453) • [📥 PDF](https://arxiv.org/pdf/2610.40453)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project Page: https://arthuring.github.io/PhysInsert/

</details>

<details>
<summary><b>24. Online Dense Reinforcement Learning with Planning-Grounding Agent for UI Navigation on Live IOS Apps</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39264) • [📄 arXiv](https://arxiv.org/abs/2610.39264) • [📥 PDF](https://arxiv.org/pdf/2610.39264)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> While frontier vision-language models demonstrate strong capabilities in user interface (UI) navigation, their high inference costs prohibit everyday use. Conversely, smaller models trained via supervised fine-tuning (SFT) or offline reinforcement...

</details>

<details>
<summary><b>25. OpenFlyV2: An onboard vision-language-action model for aerial perception and navigation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39287) • [📄 arXiv](https://arxiv.org/abs/2610.39287) • [📥 PDF](https://arxiv.org/pdf/2610.39287)

**💻 Code:** [⭐ Code](https://github.com/Ideal-ljl/AeroBrain) • [⭐ Code](https://github.com/huggingface)

> OpenFlyV2 introduces a 130K-sample VLN+VQA dataset and unifies spatial understanding with vision-language navigation, achieving 10 Hz onboard inference and closed-loop navigation on NVIDIA Jetson Orin NX.

</details>

<details>
<summary><b>26. AB-Attention: Accurate Block Attention for Accelerating Video Diffusion Transformers</b> ⭐ 24</summary>

<br/>

**👥 Authors:** Yixing Ma, Xinfeng Yang, Zhengfeiyang Zhang, Mo Zhang, Bin Xu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39663) • [📄 arXiv](https://arxiv.org/abs/2610.39663) • [📥 PDF](https://arxiv.org/pdf/2610.39663)

**💻 Code:** [⭐ Code](https://github.com/InfRec-Cardinal/AB-Attention) • [⭐ Code](https://github.com/huggingface)

> ..

</details>

<details>
<summary><b>27. SO-101 Bench: Accessible Spatial Manipulation</b> ⭐ 40</summary>

<br/>

**👥 Authors:** 5hadytru

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.36420) • [📄 arXiv](https://arxiv.org/abs/2610.36420) • [📥 PDF](https://arxiv.org/pdf/2610.36420)

**💻 Code:** [⭐ Code](https://github.com/5hadytru/so101_bench) • [⭐ Code](https://github.com/huggingface)

> SO-101 Bench is a new real2sim2real benchmark that allows SO-101 arms to perform interesting tasks out of the box, while providing a window into the arc of progress in robot learning. Check it out!

</details>

<details>
<summary><b>28. Training Foundation Models with a Single Algorithm</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40564) • [📄 arXiv](https://arxiv.org/abs/2610.40564) • [📥 PDF](https://arxiv.org/pdf/2610.40564)

**💻 Code:** [⭐ Code](https://github.com/yudasong/unified-training) • [⭐ Code](https://github.com/huggingface)

> Foundation models are typically trained in stages: pretraining, midtraining, supervised fine-tuning (SFT), and reinforcement learning (RL), with changes in both data and learning algorithms. We show that this algorithmic separation is not fundamen...

</details>

<details>
<summary><b>29. ABF-Bench: A Verifiable Benchmark for Agentic Blockchain Forensics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.23656) • [📄 arXiv](https://arxiv.org/abs/2610.23656) • [📥 PDF](https://arxiv.org/pdf/2610.23656)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> ABF-Bench makes blockchain forensics a verifiable task for LLM agents, one that rewards reasoning and may also support RLVR. This is an early version, and we plan to extend it with more real-world money-laundering cases beyond the Bybit hack.

</details>

<details>
<summary><b>30. Your LR Schedule May Not Be Principled: Effective Learning Rate Scheduling in Weight-Normalized Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.40362) • [📄 arXiv](https://arxiv.org/abs/2610.40362) • [📥 PDF](https://arxiv.org/pdf/2610.40362)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Weight-normalized (WN) training is emerging as a compelling alternative to conventional weight decay (WD), with recent studies reporting substantial pretraining loss gains. We show that these gains are driven primarily by the induced effective lea...

</details>

<details>
<summary><b>31. Unlocking Any-Order Generation in Pretrained Autoregressive Image Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhaozheng Yin, Marco Pedersoli, rishavpramanik

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39293) • [📄 arXiv](https://arxiv.org/abs/2610.39293) • [📥 PDF](https://arxiv.org/pdf/2610.39293)

**💻 Code:** [⭐ Code](https://github.com/Ordered-AR/unlockedadaptation) • [⭐ Code](https://github.com/huggingface)

> Many AI image generators build a picture one piece at a time, always in the same order: left to right, top to bottom, like reading a page. Because of this fixed habit, they cannot fill a missing patch in the middle of a photo, extend an image beyo...

</details>

<details>
<summary><b>32. Matched Capabilities, Different Behaviors: A Controlled Study of MCP and CLI Interfaces for Language Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.34298) • [📄 arXiv](https://arxiv.org/abs/2610.34298) • [📥 PDF](https://arxiv.org/pdf/2610.34298)

**💻 Code:** [⭐ Code](https://github.com/ModalityDance/MICA) • [⭐ Code](https://github.com/huggingface)

> We introduce MICA, a controlled study of MCP and CLI interfaces with matched executable capabilities, spanning six models and three interactive environments. We find substantial differences in performance, interaction cost, and agent behavior. We ...

</details>

<details>
<summary><b>33. Broken on Arrival: Non-Functional Quantized LLM Artifacts in Public Model Registries and How to Catch Them</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.05881) • [📄 arXiv](https://arxiv.org/abs/2609.05881) • [📥 PDF](https://arxiv.org/pdf/2609.05881)

**💻 Code:** [⭐ Code](https://github.com/aditi-p31/quantcheck) • [⭐ Code](https://github.com/huggingface)

> We ran every eligible code-capable GGUF tag in the official Ollama library (305 tags across 15 model lines) plus 22 popular Hugging Face community files through a 15-task code screen, then retested suspects on all 164 HumanEval+ tasks. Seven offic...

</details>

<details>
<summary><b>34. Scaling Lean Theorem Proving for Real-World Code Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38174) • [📄 arXiv](https://arxiv.org/abs/2610.38174) • [📥 PDF](https://arxiv.org/pdf/2610.38174)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present EasyProver, a Lean proving system built from recursive agents with isolated contexts, asynchronous search, parallel tactic probing, and an elastic elaboration service. Using an off-the-shelf frontier model with no additional training, i...

</details>

<details>
<summary><b>35. Holo4D: Holistic 4D Reconstruction as Geometric Control for Video Diffusion</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.25276) • [📄 arXiv](https://arxiv.org/abs/2610.25276) • [📥 PDF](https://arxiv.org/pdf/2610.25276)

**💻 Code:** [⭐ Code](https://github.com/NIRVANALAN/Holo4D) • [⭐ Code](https://github.com/huggingface)

> Excited to share Holo4D, accepted to NeurIPS 2026! 🎥 Give it a video and a new camera path, and Holo4D reshoots the video from that camera. It lifts the source video into 3D tracks (a 4D scaffold) and uses them to steer a pretrained video diffusio...

</details>

<details>
<summary><b>36. After One-Click Video: The Inflow of Service-Type Content, the Failure of the Recommendation Pool Mechanism, and the BoLe System</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.37695) • [📄 arXiv](https://arxiv.org/abs/2610.37695) • [📥 PDF](https://arxiv.org/pdf/2610.37695)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> One-click video tools now let anyone make good-looking content fast. Most of it is service-type content: people watch it to meet a need, not to enjoy the content itself. Before competing, creators, tool builders and model researchers should first ...

</details>

<details>
<summary><b>37. EPSBench: Benchmarking LLM Agents for Earnings Forecasting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38293) • [📄 arXiv](https://arxiv.org/abs/2610.38293) • [📥 PDF](https://arxiv.org/pdf/2610.38293)

**💻 Code:** [⭐ Code](https://github.com/JadeXu16/epsbench) • [⭐ Code](https://github.com/huggingface)

> Excited to share EPSBench: Benchmarking LLM Agents for Earnings Forecasting! EPSBench evaluates whether LLM agents can forecast quarterly earnings by reasoning over financial filings, earnings calls, and news, and estimating income-statement drive...

</details>

<details>
<summary><b>38. Weaken to Strengthen: Weakening-Induced On-Policy Self-Distillation for Large Reasoning Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.38681) • [📄 arXiv](https://arxiv.org/abs/2610.38681) • [📥 PDF](https://arxiv.org/pdf/2610.38681)

**💻 Code:** [⭐ Code](https://github.com/jet-ai-projects/WISD) • [⭐ Code](https://github.com/huggingface)

> WISD improves reasoning models by distilling the contrast between the original model and its quantized or pruned copy, boosting  reasoning performance without external teachers or privileged information.

</details>

<details>
<summary><b>39. FDSML: Formula-Driven Supervised Metric Learning with Parameter-Aware Triplet Loss</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hirokatsu Kataoka, Yoshihiro Fukuhara, Keitaro Tanaka, Hina Otake, ShineiArakawa

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.39423) • [📄 arXiv](https://arxiv.org/abs/2610.39423) • [📥 PDF](https://arxiv.org/pdf/2610.39423)

**💻 Code:** [⭐ Code](https://github.com/ShineiArakawa/fdsml) • [⭐ Code](https://github.com/huggingface)

> FDSML pre-trains vision models without real images, using the generating formula itself as supervision: fractals with more similar parameters should lie closer in the embedding space. We turn this parameter-space ordering into triplets, so the mod...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 39 |
| 📅 Today | [`2026-10-31.json`](data/daily/2026-10-31.json) | 39 |
| 📆 This Week | [`2026-W43.json`](data/weekly/2026-W43.json) | 110 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 1120 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-31 | 39 | [View JSON](data/daily/2026-10-31.json) |
| 📄 2026-10-30 | 10 | [View JSON](data/daily/2026-10-30.json) |
| 📄 2026-10-29 | 5 | [View JSON](data/daily/2026-10-29.json) |
| 📄 2026-10-28 | 27 | [View JSON](data/daily/2026-10-28.json) |
| 📄 2026-10-27 | 19 | [View JSON](data/daily/2026-10-27.json) |
| 📄 2026-10-26 | 10 | [View JSON](data/daily/2026-10-26.json) |
| 📄 2026-10-25 | 49 | [View JSON](data/daily/2026-10-25.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W43 | 110 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |
| 📅 2026-W41 | 287 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 1120 | [View JSON](data/monthly/2026-10.json) |
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
