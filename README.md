<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-29-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3168+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">29</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">120</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">601</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3168+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 29, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. PixelSmile: Toward Fine-Grained Facial Expression Editing</b> ⭐ 68</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25728) • [📄 arXiv](https://arxiv.org/abs/2603.25728) • [📥 PDF](https://arxiv.org/pdf/2603.25728)

**💻 Code:** [⭐ Code](https://github.com/Ammmob/PixelSmile)

> from [@Wildminder](https://x.com/wildmindai) 🌐 Project Page 👉 https://ammmob.github.io/PixelSmile/ 💻 GitHub Repo 👉 https://github.com/Ammmob/PixelSmile 🤖 Model (Hugging Face) 👉 https://huggingface.co/PixelSmile/PixelSmile 📊 Benchmark (FFE-Bench) 👉...

</details>

<details>
<summary><b>2. Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25040) • [📄 arXiv](https://arxiv.org/abs/2603.25040) • [📥 PDF](https://arxiv.org/pdf/2603.25040)

**💻 Code:** [⭐ Code](https://github.com/InternLM/Intern-S1)

> Thanks for submitting our paper to the daily paper. Please add 'Intern Large Models' as the organization. Our github is https://github.com/InternLM/Intern-S1 , and the model is available at https://huggingface.co/internlm/Intern-S1-Pro .

</details>

<details>
<summary><b>3. Calibri: Enhancing Diffusion Transformers via Parameter-Efficient Calibration</b> ⭐ 30</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24800) • [📄 arXiv](https://arxiv.org/abs/2603.24800) • [📥 PDF](https://arxiv.org/pdf/2603.24800)

**💻 Code:** [⭐ Code](https://github.com/v-gen-ai/Calibri)

> Introducing Calibri – a parameter-efficient method for diffusion transformer alignment. By optimizing only ∼ 102 parameters, Calibri substantially improves generation quality while reducing inference time.

</details>

<details>
<summary><b>4. RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models</b> ⭐ 66</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25502) • [📄 arXiv](https://arxiv.org/abs/2603.25502) • [📥 PDF](https://arxiv.org/pdf/2603.25502)

**💻 Code:** [⭐ Code](https://github.com/yfyang007/RealRestorer)

> RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models RealRestorer explores how large-scale image editing models can be used for generalizable real-world image restoration . Instead of focusing on a...

</details>

<details>
<summary><b>5. Voxtral TTS</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25551) • [📄 arXiv](https://arxiv.org/abs/2603.25551) • [📥 PDF](https://arxiv.org/pdf/2603.25551)

> Voxtral TTS is a multilingual expressive TTS with a hybrid autoregressive semantic token generator and flow-matching acoustic tokens, using Voxtral Codec for high-quality voice cloning from 3 seconds of audio.

</details>

<details>
<summary><b>6. MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens</b> ⭐ 2.3k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.23516) • [📄 arXiv](https://arxiv.org/abs/2603.23516) • [📥 PDF](https://arxiv.org/pdf/2603.23516)

**💻 Code:** [⭐ Code](https://github.com/EverMind-AI/MSA)

> 📝 Abstract Long-term memory is essential for general intelligence, yet the full attention bottleneck constrains most LLMs’ effective context length to 128K–1M . Existing attempts，hybrid linear attention, fixed-size state memory (e.g., RNNs), and e...

</details>

<details>
<summary><b>7. MACRO: Advancing Multi-Reference Image Generation with Structured Long-Context Data</b> ⭐ 38</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25319) • [📄 arXiv](https://arxiv.org/abs/2603.25319) • [📥 PDF](https://arxiv.org/pdf/2603.25319)

**💻 Code:** [⭐ Code](https://github.com/HKU-MMLab/Macro)

> We present MACRO, a large-scale multi-reference image generation dataset MacroData with 400K samples and the corresponding multi-image generation metric MacroBench. Our dataset supports the input of up to 10 reference maps, covering the four long-...

</details>

<details>
<summary><b>8. SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks</b> ⭐ 28</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24755) • [📄 arXiv](https://arxiv.org/abs/2603.24755) • [📥 PDF](https://arxiv.org/pdf/2603.24755)

**💻 Code:** [⭐ Code](https://github.com/SprocketLab/slop-code-bench)

> Coding benchmarks tell you if an agent passed the tests. They don't tell you if the code is any good, or if it's going to be workable three features from now. SlopCodeBench makes agents extend their own solutions across multiple steps without pres...

</details>

<details>
<summary><b>9. AVControl: Efficient Framework for Training Audio-Visual Controls</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24793) • [📄 arXiv](https://arxiv.org/abs/2603.24793) • [📥 PDF](https://arxiv.org/pdf/2603.24793)

> AVControl: Efficient Framework for Training Audio-Visual Controls A lightweight, extendable framework built on LTX-2 for training diverse audio-visual controls using LoRA adapters on a parallel canvas. Each control modality is trained independentl...

</details>

<details>
<summary><b>10. VFIG: Vectorizing Complex Figures in SVG with Vision-Language Models</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24575) • [📄 arXiv](https://arxiv.org/abs/2603.24575) • [📥 PDF](https://arxiv.org/pdf/2603.24575)

**💻 Code:** [⭐ Code](https://github.com/RAIVNLab/VFig)

> Ever come across a beautiful Figure 1 in a paper, only to wish you could easily edit and adapt it for your own use? Check out our new work VFig: Vectorizing Complex Figures in SVG with Vision-Language Models! It is a specialized VLM that converts ...

</details>

<details>
<summary><b>11. Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25745) • [📄 arXiv](https://arxiv.org/abs/2603.25745) • [📥 PDF](https://arxiv.org/pdf/2603.25745)

> The decoupling of geometric complexity from rendering resolution is a smart approach — similar to what we've seen work in neural radiance fields with feature grids. The per-primitive textures remind me of UV atlas techniques from traditional graph...

</details>

<details>
<summary><b>12. MuRF: Unlocking the Multi-Scale Potential of Vision Foundation Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25744) • [📄 arXiv](https://arxiv.org/abs/2603.25744) • [📥 PDF](https://arxiv.org/pdf/2603.25744)

**💻 Code:** [⭐ Code](https://github.com/orgs/MuRF-VFM) • [⭐ Code](https://github.com/MuRF-VFM/MuRF-VFM.github.io)

> MuRF: Unlocking the Multi-Scale Potential of Vision Foundation Models Bocheng Zou†, Mu Cai†, Mark Stanley∗, Dingfu Lu∗, and Yong Jae Lee 1 University of Wisconsin-Madison Projects: https://MuRF-VFM.github.io Code Repository: https://github.com/org...

</details>

<details>
<summary><b>13. MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Xianfeng Tang, Hui Liu, Hanqing Lu, Zhiwei Zhang, Minhua Lin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.18718) • [📄 arXiv](https://arxiv.org/abs/2603.18718) • [📥 PDF](https://arxiv.org/pdf/2603.18718)

**💻 Code:** [⭐ Code](https://github.com/ventr1c/memma)

> Code is publicly available at https://github.com/ventr1c/memma

</details>

<details>
<summary><b>14. Representation Alignment for Just Image Transformers is not Easier than You Think</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.14366) • [📄 arXiv](https://arxiv.org/abs/2603.14366) • [📥 PDF](https://arxiv.org/pdf/2603.14366)

**💻 Code:** [⭐ Code](https://github.com/kaist-cvml/PixelREPA)

> REPA for JiTs. Github: https://github.com/kaist-cvml/PixelREPA

</details>

<details>
<summary><b>15. FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24943) • [📄 arXiv](https://arxiv.org/abs/2603.24943) • [📥 PDF](https://arxiv.org/pdf/2603.24943)

**💻 Code:** [⭐ Code](https://github.com/aliyun/qwen-dianjin)

> FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol

</details>

<details>
<summary><b>16. AVO: Agentic Variation Operators for Autonomous Evolutionary Search</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Timmy Liu, Zihao Ye, Bing Xu, Zhifan Ye, Terry Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24517) • [📄 arXiv](https://arxiv.org/abs/2603.24517) • [📥 PDF](https://arxiv.org/pdf/2603.24517)

> No abstract available.

</details>

<details>
<summary><b>17. Vega: Learning to Drive with Natural Language Instructions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jie Zhou, Zheng Zhu, Wenzhao Zheng, Yuxuan Li, Sicheng Zuo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25741) • [📄 arXiv](https://arxiv.org/abs/2603.25741) • [📥 PDF](https://arxiv.org/pdf/2603.25741)

> The shift from scene descriptions to instruction-following is a key evolution for embodied agents. Most VL-AM papers treat language as a static conditioning signal, but personal driving requires dynamic instruction interpretation. Curious if the I...

</details>

<details>
<summary><b>18. S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25702) • [📄 arXiv](https://arxiv.org/abs/2603.25702) • [📥 PDF](https://arxiv.org/pdf/2603.25702)

**💻 Code:** [⭐ Code](https://github.com/phymhan/S2D2)

> S2D2 is a training-free self-speculative decoding method for block-diffusion LLMs: the same pretrained model drafts in diffusion mode and verifies in block-size-1 autoregressive mode, improving the accuracy-speed tradeoff over strong confidence-th...

</details>

<details>
<summary><b>19. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25562) • [📄 arXiv](https://arxiv.org/abs/2603.25562) • [📥 PDF](https://arxiv.org/pdf/2603.25562)

**💻 Code:** [⭐ Code](https://github.com/hhh675597/revisiting_opd)

> On-policy distillation (OPD) trains a student on its own rollouts using teacher feedback[1][2][3]. In long-horizon LLM post-training, the common sampled-token implementation can be brittle. From a bias-variance perspective, token-level OPD is bias...

</details>

<details>
<summary><b>20. BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hiroaki Santo, Kuniaki Saito, Nakamasa Inoue, Kaede Shiohara, Risa Shinoda

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.23883) • [📄 arXiv](https://arxiv.org/abs/2603.23883) • [📥 PDF](https://arxiv.org/pdf/2603.23883)

> Understanding animal species from multimodal data poses an emerging challenge at the intersection of computer vision and ecology. While recent biological models, such as BioCLIP, have demonstrated strong alignment between images and textual taxono...

</details>

<details>
<summary><b>21. Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Byeongju Woo, Byeonghyun Pak, Yunghee Lee, SeokminLee-Chris

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13904) • [📄 arXiv](https://arxiv.org/abs/2603.13904) • [📥 PDF](https://arxiv.org/pdf/2603.13904)

**💻 Code:** [⭐ Code](https://github.com/SeokminLee-Chris/CroBo)

> We propose CroBo, a self-supervised visual representation framework for robotics that encodes both what is in the scene and where it is, all in a single compact token. By reconstructing masked local crops from a global bottleneck token, CroBo lear...

</details>

<details>
<summary><b>22. Electrostatic Photoluminescence Tuning in All-Solid-State Perovskite Transistors</b> ⭐ 37</summary>

<br/>

**👥 Authors:** Vitaly Podzorov, Artem A. Bakulin, Beier Hu, Dmitry Maslennikov, Vladimir Bruevich

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25718) • [📄 arXiv](https://arxiv.org/abs/2603.25718) • [📥 PDF](https://arxiv.org/pdf/2603.25718)

**💻 Code:** [⭐ Code](https://github.com/H-EmbodVis/HyDRA)

> Video world models have shown immense potential in simulating the physical world, yet existing memory mechanisms primarily treat environments as static canvases. When dynamic subjects hide out of sight and later re-emerge, current methods often st...

</details>

<details>
<summary><b>23. PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.25398) • [📄 arXiv](https://arxiv.org/abs/2603.25398) • [📥 PDF](https://arxiv.org/pdf/2603.25398)

**💻 Code:** [⭐ Code](https://github.com/tue-mps/pmt)

> We present the Plain Mask Transformer (PMT), a fast Transformer-based segmentation model that operates on top of frozen Vision Foundation Model (VFM) features. Encoder-only models like EoMT and VidEoMT achieve competitive accuracy with low latency...

</details>

<details>
<summary><b>24. Can MLLMs Read Students' Minds? Unpacking Multimodal Error Analysis in Handwritten Math</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Zhiling Yan, Hang Li, Yi-Fan Zhang, Tianlong Xu, Dingjie Song

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24961) • [📄 arXiv](https://arxiv.org/abs/2603.24961) • [📥 PDF](https://arxiv.org/pdf/2603.24961)

**💻 Code:** [⭐ Code](https://github.com/ai-for-edu/ScratchMath)

> Assessing student handwritten scratchwork is crucial for personalized educational feedback but presents unique challenges due to diverse handwriting, complex layouts, and varied problem-solving approaches. Existing educational NLP primarily focuse...

</details>

<details>
<summary><b>25. Reaching Beyond the Mode: RL for Distributional Reasoning in Language Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24844) • [📄 arXiv](https://arxiv.org/abs/2603.24844) • [📥 PDF](https://arxiv.org/pdf/2603.24844)

**💻 Code:** [⭐ Code](https://github.com/ishapuri/multi_answer_rl)

> Current post-training methods for language models implicitly collapse a rich distribution of possible answers into a single dominant output. While this works for benchmark-style tasks, many real-world settings—like medical diagnosis, coding, and a...

</details>

<details>
<summary><b>26. WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24836) • [📄 arXiv](https://arxiv.org/abs/2603.24836) • [📥 PDF](https://arxiv.org/pdf/2603.24836)

**💻 Code:** [⭐ Code](https://github.com/princeton-vl/WAFT-Stereo)

> New efficient state-of-the-art algorithm for rectified stereo: Best-performing model: #1 on ETH3D, Middlebury, and KITTI, 1.8-6.7x faster than competitive methods, 81% zero-shot error reduction. Fastest model: 21FPS on qHD input, up to 80% zero-sh...

</details>

<details>
<summary><b>27. Nudging Hidden States: Training-Free Model Steering for Chain-of-Thought Reasoning in Large Audio-Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.14636) • [📄 arXiv](https://arxiv.org/abs/2603.14636) • [📥 PDF](https://arxiv.org/pdf/2603.14636)

> Steering LALMs for better CoT Reasoning

</details>

<details>
<summary><b>28. Extending Precipitation Nowcasting Horizons via Spectral Fusion of Radar Observations and Foundation Model Priors</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Yan Liu, Wen Wang, Zhiqing Guo, Qingyong Li, Onemiss

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.21768) • [📄 arXiv](https://arxiv.org/abs/2603.21768) • [📥 PDF](https://arxiv.org/pdf/2603.21768)

**💻 Code:** [⭐ Code](https://github.com/Onemissed/PW-FouCast)

> We introduce PW-FouCast, a frequency-domain fusion framework designed to extend precipitation nowcasting horizons. By leveraging Pangu-Weather forecasts as spectral priors within a Fourier-based backbone, we align spectral magnitudes and phases to...

</details>

<details>
<summary><b>29. IQuest-Coder-V1 Technical Report</b> ⭐ 1.36k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16733) • [📄 arXiv](https://arxiv.org/abs/2603.16733) • [📥 PDF](https://arxiv.org/pdf/2603.16733)

**💻 Code:** [⭐ Code](https://github.com/IQuestLab/IQuest-Coder-V1)

> IQuest-Coder-V1 Technical Report

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 29 |
| 📅 Today | [`2026-03-29.json`](data/daily/2026-03-29.json) | 29 |
| 📆 This Week | [`2026-W12.json`](data/weekly/2026-W12.json) | 120 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 601 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-29 | 29 | [View JSON](data/daily/2026-03-29.json) |
| 📄 2026-03-28 | 29 | [View JSON](data/daily/2026-03-28.json) |
| 📄 2026-03-27 | 6 | [View JSON](data/daily/2026-03-27.json) |
| 📄 2026-03-26 | 4 | [View JSON](data/daily/2026-03-26.json) |
| 📄 2026-03-25 | 11 | [View JSON](data/daily/2026-03-25.json) |
| 📄 2026-03-24 | 37 | [View JSON](data/daily/2026-03-24.json) |
| 📄 2026-03-23 | 4 | [View JSON](data/daily/2026-03-23.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |
| 📅 2026-W10 | 119 | [View JSON](data/weekly/2026-W10.json) |
| 📅 2026-W09 | 201 | [View JSON](data/weekly/2026-W09.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 601 | [View JSON](data/monthly/2026-03.json) |
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
