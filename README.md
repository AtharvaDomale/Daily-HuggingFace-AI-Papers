<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-30-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2539+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">128</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">1020</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2539+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 27, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. HyTRec: A Hybrid Temporal-Aware Attention Architecture for Long Behavior Sequential Recommendation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18283) • [📄 arXiv](https://arxiv.org/abs/2602.18283) • [📥 PDF](https://arxiv.org/pdf/2602.18283)

> HyTRec: The first hybrid attention framework for efficient long behavior sequence modeling

</details>

<details>
<summary><b>2. MolHIT: Advancing Molecular-Graph Generation with Hierarchical Discrete Diffusion Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.17602) • [📄 arXiv](https://arxiv.org/abs/2602.17602) • [📥 PDF](https://arxiv.org/pdf/2602.17602)

> We introduce a hierarchical discrete diffusion framework for molecular graph generation that overcomes long-standing performance limitations in existing methods. We generalize discrete diffusion to additional categories that encode chemical priors...

</details>

<details>
<summary><b>3. DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation</b> ⭐ 54</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.12160) • [📄 arXiv](https://arxiv.org/abs/2602.12160) • [📥 PDF](https://arxiv.org/pdf/2602.12160)

**💻 Code:** [⭐ Code](https://github.com/Guoxu1233/DreamID-Omni)

> We introduce DreamID-Omni, a unified framework for controllable human-centric audio-video generation. Project page: https://guoxu1233.github.io/DreamID-Omni/ Code: https://github.com/Guoxu1233/DreamID-Omni

</details>

<details>
<summary><b>4. SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21818) • [📄 arXiv](https://arxiv.org/abs/2602.21818) • [📥 PDF](https://arxiv.org/pdf/2602.21818)

> SkyReels-V4 is a unified multi-modal video-audio generation model with dual-stream diffusion transformers performing video generation, audio synthesis, inpainting, and editing at cinematic resolutions with efficiency via low-res sequences and keyf...

</details>

<details>
<summary><b>5. ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21534) • [📄 arXiv](https://arxiv.org/abs/2602.21534) • [📥 PDF](https://arxiv.org/pdf/2602.21534)

**💻 Code:** [⭐ Code](https://github.com/WillDreamer/ARL-Arena)

> Agentic Reinforcement Learning

</details>

<details>
<summary><b>6. GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22190) • [📄 arXiv](https://arxiv.org/abs/2602.22190) • [📥 PDF](https://arxiv.org/pdf/2602.22190)

**💻 Code:** [⭐ Code](https://github.com/GUI-Libra/GUI-Libra)

> This paper addresses two core bottlenecks in post-training native GUI agents: (1) the lack of high-quality, action-aligned reasoning data, and (2) generic post-training pipelines that overlook the unique constraints of GUI interaction. GUI-Libra m...

</details>

<details>
<summary><b>7. Solaris: Building a Multiplayer Video World Model in Minecraft</b> ⭐ 56</summary>

<br/>

**👥 Authors:** Timothy Meehan, Suppakit Waiwitlikhit, Daohan Lu, Oscar Michel, Georgy Savva

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22208) • [📄 arXiv](https://arxiv.org/abs/2602.22208) • [📥 PDF](https://arxiv.org/pdf/2602.22208)

**💻 Code:** [⭐ Code](https://github.com/solaris-wm/solaris)

> Solaris develops a multiplayer video world model for Minecraft, enabling coordinated multi-agent observations, scalable data collection, and a staged training pipeline with memory-efficient Checkpointed Self Forcing to beat baselines.

</details>

<details>
<summary><b>8. DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21548) • [📄 arXiv](https://arxiv.org/abs/2602.21548) • [📥 PDF](https://arxiv.org/pdf/2602.21548)

> We present DualPath, an inference system that breaks this bottleneck by introducing dual-path KV-Cache loading. Beyond the traditional storage-to-prefill path, DualPath enables a novel storage-to-decode path, in which the KV-Cache is loaded into d...

</details>

<details>
<summary><b>9. VecGlypher: Unified Vector Glyph Generation with Language Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21461) • [📄 arXiv](https://arxiv.org/abs/2602.21461) • [📥 PDF](https://arxiv.org/pdf/2602.21461)

**💻 Code:** [⭐ Code](https://github.com/xk-huang/VecGlypher)

> Project page: https://xk-huang.github.io/VecGlypher/ Code: https://github.com/xk-huang/VecGlypher Data: https://huggingface.co/VecGlypher

</details>

<details>
<summary><b>10. JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation</b> ⭐ 322</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.19163) • [📄 arXiv](https://arxiv.org/abs/2602.19163) • [📥 PDF](https://arxiv.org/pdf/2602.19163)

**💻 Code:** [⭐ Code](https://github.com/JavisVerse/JavisDiT)

> JavisDiT++ is a concise yet powerful DiT model to generate high-quality and synchronized sounding videos with textual conditions. Built upon the lightweight Wan2.1-1.3B-T2V backbone, JavisDiT++ addresses the key bottleneck of joint audio-video gen...

</details>

<details>
<summary><b>11. Image Generation with a Sphere Encoder</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tom Goldstein, Ji Hou, Menglin Jia, Kaiyu Yue

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.15030) • [📄 arXiv](https://arxiv.org/abs/2602.15030) • [📥 PDF](https://arxiv.org/pdf/2602.15030)

**💻 Code:** [⭐ Code](https://github.com/facebookresearch/sphere-encoder)

> Technical report

</details>

<details>
<summary><b>12. World Guidance: World Modeling in Condition Space for Action Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22010) • [📄 arXiv](https://arxiv.org/abs/2602.22010) • [📥 PDF](https://arxiv.org/pdf/2602.22010)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Ac...

</details>

<details>
<summary><b>13. From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Mohamed Elhoseiny, Hongsheng Li, Sayak Paul, Le Zhuo, Liangbing Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21778) • [📄 arXiv](https://arxiv.org/abs/2602.21778) • [📥 PDF](https://arxiv.org/pdf/2602.21778)

**💻 Code:** [⭐ Code](https://github.com/liangbingzhao/PhysicEdit)

> Instruction-based image editing has achieved remarkable success in semantic alignment, yet state-of-the-art models frequently fail to render physically plausible results when editing involves complex causal dynamics, such as refraction or material...

</details>

<details>
<summary><b>14. NanoKnow: How to Know What Your Language Model Knows</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20122) • [📄 arXiv](https://arxiv.org/abs/2602.20122) • [📥 PDF](https://arxiv.org/pdf/2602.20122)

**💻 Code:** [⭐ Code](https://github.com/castorini/NanoKnow/tree/main) • [⭐ Code](https://github.com/castorini/NanoKnow)

> We release NanoKnow, a benchmark dataset that partitions questions from Natural Questions and SQuAD into "supported" and "unsupported" splits based on whether their answers are present in nanochat's  — a family of small LLMs  — pre-training data. ...

</details>

<details>
<summary><b>15. The Design Space of Tri-Modal Masked Diffusion Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21472) • [📄 arXiv](https://arxiv.org/abs/2602.21472) • [📥 PDF](https://arxiv.org/pdf/2602.21472)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Scaling Beyond Masked Diffusion Language Models (2026) DiMo: Discrete Diffu...

</details>

<details>
<summary><b>16. SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Geonho Cha, Byeongju Han, MinKyu Lee, Sangeek Hyun, wldn0202

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18993) • [📄 arXiv](https://arxiv.org/abs/2602.18993) • [📥 PDF](https://arxiv.org/pdf/2602.18993)

**💻 Code:** [⭐ Code](https://github.com/jiwoogit/SeaCache)

> SeaCache is a training-free acceleration method that utilizes Spectral Evolution to decouple low-frequency content from high-frequency noise. It consistently outperforms existing baselines without requiring additional hyperparameter tuning, showin...

</details>

<details>
<summary><b>17. UniVBench: Towards Unified Evaluation for Video Foundation Models</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Yan Zhang, Yuan Wang, Yichen Li, Xiaotian Zhang, Jianhui Wei

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21835) • [📄 arXiv](https://arxiv.org/abs/2602.21835) • [📥 PDF](https://arxiv.org/pdf/2602.21835)

**💻 Code:** [⭐ Code](https://github.com/JianhuiWei7/UniVBench)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VINO: A Unified Visual Generator with Interleaved OmniModal Context (2026) ...

</details>

<details>
<summary><b>18. Revisiting Text Ranking in Deep Research</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21456) • [📄 arXiv](https://arxiv.org/abs/2602.21456) • [📥 PDF](https://arxiv.org/pdf/2602.21456)

**💻 Code:** [⭐ Code](https://github.com/ChuanMeng/text-ranking-in-deep-research)

> Deep research has emerged as an important task that aims to address hard queries through extensive open-web exploration. To tackle it, most prior work equips large language model (LLM)-based agents with opaque web search APIs, enabling agents to i...

</details>

<details>
<summary><b>19. Model Context Protocol (MCP) Tool Descriptions Are Smelly! Towards Improving AI Agent Efficiency with Augmented MCP Tool Descriptions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ahmed E. Hassan, Bram Adams, Gopi Krishnan Rajbahadur, Mohammed Mehedi Hasan, hao-li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.14878) • [📄 arXiv](https://arxiv.org/abs/2602.14878) • [📥 PDF](https://arxiv.org/pdf/2602.14878)

> The Model Context Protocol (MCP) is rapidly becoming the "USB-C for AI," but the natural-language descriptions powering it are riddled with traditional software "smells."

</details>

<details>
<summary><b>20. NoLan: Mitigating Object Hallucinations in Large Vision-Language Models via Dynamic Suppression of Language Priors</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Xinchao Wang, Runpeng Yu, Weihao Yu, Lingfeng Ren

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.22144) • [📄 arXiv](https://arxiv.org/abs/2602.22144) • [📥 PDF](https://arxiv.org/pdf/2602.22144)

**💻 Code:** [⭐ Code](https://github.com/lingfengren/NoLan)

> Code: https://github.com/lingfengren/NoLan

</details>

<details>
<summary><b>21. Small Language Models for Privacy-Preserving Clinical Information Extraction in Low-Resource Languages</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Sepehr Ghahraei, Ali Akbar Omidvarian, Ebrahim Heidari-Farsani, Nahid Yousefian, Mohammadreza Ghaffarzadeh-Esfahani

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.21374) • [📄 arXiv](https://arxiv.org/abs/2602.21374) • [📥 PDF](https://arxiv.org/pdf/2602.21374)

**💻 Code:** [⭐ Code](https://github.com/mohammad-gh009/Small-language-models-on-clinical-data-extraction)

> We benchmark five open-source small language models (SLMs, 1B–8B parameters) on a two-step pipeline for extracting 13 binary clinical features from 1,221 anonymized Persian palliative care transcripts — no fine-tuning required. Qwen2.5-7B-Instruct...

</details>

<details>
<summary><b>22. Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20933) • [📄 arXiv](https://arxiv.org/abs/2602.20933) • [📥 PDF](https://arxiv.org/pdf/2602.20933)

> Accepted by CVPR 2026

</details>

<details>
<summary><b>23. Functional Continuous Decomposition</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20857) • [📄 arXiv](http://arxiv.org/abs/2602.20857) • [📥 PDF](https://arxiv.org/pdf/2602.20857)

**💻 Code:** [⭐ Code](https://github.com/Tima-a/fcd)

> I am excited to announce my latest research: Functional Continuous Decomposition (FCD) - a JAX-accelerated framework designed for parametric, continuous signal decomposition. Traditional signal processing algorithms like Empirical Mode Decompositi...

</details>

<details>
<summary><b>24. The Truthfulness Spectrum Hypothesis</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.20273) • [📄 arXiv](https://arxiv.org/abs/2602.20273) • [📥 PDF](https://arxiv.org/pdf/2602.20273)

**💻 Code:** [⭐ Code](https://github.com/zfying/truth_spec)

> We propose the Truthfulness Spectrum Hypothesis: truth directions of varying generality coexist! Probe geometry predicts generalization, and post-training reshapes it! We create  FLEED datasets (definitional, empirical, logical, fictional, ethical...

</details>

<details>
<summary><b>25. ISO-Bench: Can Coding Agents Optimize Real-World Inference Workloads?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.19594) • [📄 arXiv](https://arxiv.org/abs/2602.19594) • [📥 PDF](https://arxiv.org/pdf/2602.19594)

> We built ISO-Bench: 54 real optimization tasks from vLLM and SGLang and found that agents often understand the problem but can't execute the fix.

</details>

<details>
<summary><b>26. MoBind: Motion Binding for Fine-Grained IMU-Video Pose Alignment</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.19004) • [📄 arXiv](https://arxiv.org/abs/2602.19004) • [📥 PDF](https://arxiv.org/pdf/2602.19004)

**💻 Code:** [⭐ Code](https://github.com/bbvisual/MoBind)

> Accepted to CVPR26

</details>

<details>
<summary><b>27. DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction</b> ⭐ 13</summary>

<br/>

**👥 Authors:** K. Joost Batenburg, Daniel M. Pelt, jiayangshi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18589) • [📄 arXiv](https://arxiv.org/abs/2602.18589) • [📥 PDF](https://arxiv.org/pdf/2602.18589)

**💻 Code:** [⭐ Code](https://github.com/DM4CT/DM4CT)

> Benchmark on diffusion models for CT reconstruction

</details>

<details>
<summary><b>28. JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18527) • [📄 arXiv](https://arxiv.org/abs/2602.18527) • [📥 PDF](https://arxiv.org/pdf/2602.18527)

> 3D AV-LLM leveraging RGB-D and First-Order Ambisonics for end-to-end grounding and spatial reasoning.

</details>

<details>
<summary><b>29. Intent Laundering: AI Safety Datasets Are Not What They Seem</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.16729) • [📄 arXiv](https://arxiv.org/abs/2602.16729) • [📥 PDF](https://arxiv.org/pdf/2602.16729)

> We systematically evaluate the quality of widely used AI safety datasets from two perspectives: in isolation and in practice. In isolation, we examine how faithfully these datasets represent real-world adversarial behavior and find that they fall ...

</details>

<details>
<summary><b>30. Yor-Sarc: A gold-standard dataset for sarcasm detection in a low-resource African language</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.18964) • [📄 arXiv](https://arxiv.org/abs/2602.18964) • [📥 PDF](https://arxiv.org/pdf/2602.18964)

**💻 Code:** [⭐ Code](https://github.com/toheebadura/yor-sarc)

> Yor-Sarc: the first gold-standard Yorùbá sarcasm dataset (436 instances), annotated by three native speakers with substantial agreement (Fleiss’ κ=0.77). A culturally grounded benchmark to accelerate the growth of fine-grained figurative language ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 30 |
| 📅 Today | [`2026-02-27.json`](data/daily/2026-02-27.json) | 30 |
| 📆 This Week | [`2026-W08.json`](data/weekly/2026-W08.json) | 128 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 1020 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-27 | 30 | [View JSON](data/daily/2026-02-27.json) |
| 📄 2026-02-26 | 32 | [View JSON](data/daily/2026-02-26.json) |
| 📄 2026-02-25 | 25 | [View JSON](data/daily/2026-02-25.json) |
| 📄 2026-02-24 | 18 | [View JSON](data/daily/2026-02-24.json) |
| 📄 2026-02-23 | 23 | [View JSON](data/daily/2026-02-23.json) |
| 📄 2026-02-22 | 23 | [View JSON](data/daily/2026-02-22.json) |
| 📄 2026-02-21 | 23 | [View JSON](data/daily/2026-02-21.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W08 | 128 | [View JSON](data/weekly/2026-W08.json) |
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |
| 📅 2026-W06 | 293 | [View JSON](data/weekly/2026-W06.json) |
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 1020 | [View JSON](data/monthly/2026-02.json) |
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
