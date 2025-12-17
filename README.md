<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-41-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-411+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">41</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">87</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">460</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">411+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** December 17, 2025

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding</b> ⭐ 18</summary>

<br/>

**👥 Authors:** Chongxuan Li, Wei Wu, Jian Guan, JinaLeejnl

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13586) • [📄 arXiv](https://arxiv.org/abs/2512.13586) • [📥 PDF](https://arxiv.org/pdf/2512.13586)

**💻 Code:** [⭐ Code](https://github.com/ML-GSAI/ReFusion)

> ReFusion is a masked diffusion model that achieves superior performance and efficiency, featuring full KV cache reuse while simultaneously supporting any-order generation.

</details>

<details>
<summary><b>2. Towards Scalable Pre-training of Visual Tokenizers for Generation</b> ⭐ 92</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13687) • [📄 arXiv](https://arxiv.org/abs/2512.13687) • [📥 PDF](https://arxiv.org/pdf/2512.13687)

**💻 Code:** [⭐ Code](https://github.com/hustvl) • [⭐ Code](https://github.com/MiniMax-AI/VTP)

> GitHub codes: https://github.com/MiniMax-AI/VTP Huggingface weights: https://huggingface.co/collections/MiniMaxAI/vtp collaborated with HUST Vision Lab: https://github.com/hustvl

</details>

<details>
<summary><b>3. Memory in the Age of AI Agents</b> ⭐ 115</summary>

<br/>

**👥 Authors:** Jeryi, zstanjj, KYLN24, Liusc2020, namespace-ERI

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13564) • [📄 arXiv](https://arxiv.org/abs/2512.13564) • [📥 PDF](https://arxiv.org/pdf/2512.13564)

**💻 Code:** [⭐ Code](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

> Memory has emerged, and will continue to remain, a core capability of foundation model-based agents. As research on agent memory rapidly expands and attracts unprecedented attention, the field has also become increasingly fragmented. Existing work...

</details>

<details>
<summary><b>4. QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management</b> ⭐ 312</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12967) • [📄 arXiv](https://arxiv.org/abs/2512.12967) • [📥 PDF](https://arxiv.org/pdf/2512.12967)

**💻 Code:** [⭐ Code](https://github.com/Tongyi-Zhiwen/Qwen-Doc) • [⭐ Code](https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/main/QwenLong-L1.5)

> We introduce QwenLong-L1.5, a model that achieves superior long-context reasoning capabilities through systematic post-training innovations. The key technical breakthroughs of QwenLong-L1.5 are as follows: (1) Long-Context Data Synthesis Pipeline:...

</details>

<details>
<summary><b>5. LongVie 2: Multimodal Controllable Ultra-Long Video World Model</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xian Liu, Zhaoxi Chen, Jianxiong Gao, ChenyangSi, JunhaoZhuang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13604) • [📄 arXiv](https://arxiv.org/abs/2512.13604) • [📥 PDF](https://arxiv.org/pdf/2512.13604)

**💻 Code:** [⭐ Code](https://github.com/Vchitect/LongVie)

> Page: https://vchitect.github.io/LongVie2-project/ Github: https://github.com/Vchitect/LongVie Huggingface: https://huggingface.co/Vchitect/LongVie2

</details>

<details>
<summary><b>6. Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13168) • [📄 arXiv](https://arxiv.org/abs/2512.13168) • [📥 PDF](https://arxiv.org/pdf/2512.13168)

> Real-world F&A work is messy, spanning heterogeneous and large-scale artifacts such as spreadsheets and PDFs. It's also long-horizon and knowledge-intensive: workflows interleave multiple tasks and span diverse domains such as budgeting, trading, ...

</details>

<details>
<summary><b>7. NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents</b> ⭐ 25</summary>

<br/>

**👥 Authors:** yo37, kkish, YueHou, coffiney, JingzheDing

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12730) • [📄 arXiv](https://arxiv.org/abs/2512.12730) • [📥 PDF](https://arxiv.org/pdf/2512.12730)

**💻 Code:** [⭐ Code](https://github.com/multimodal-art-projection/NL2RepoBench)

> Recent advances in coding agents suggest rapid progress toward autonomous software development, yet existing benchmarks fail to rigorously evaluate the long-horizon capabilities required to build complete software systems. Most prior evaluations f...

</details>

<details>
<summary><b>8. Error-Free Linear Attention is a Free Lunch: Exact Solution from Continuous-Time Dynamics</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12602) • [📄 arXiv](https://arxiv.org/abs/2512.12602) • [📥 PDF](https://arxiv.org/pdf/2512.12602)

**💻 Code:** [⭐ Code](https://github.com/declare-lab/EFLA)

> Error-Free Linear Attention is a Free Lunch!

</details>

<details>
<summary><b>9. KlingAvatar 2.0 Technical Report</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13313) • [📄 arXiv](https://arxiv.org/abs/2512.13313) • [📥 PDF](https://arxiv.org/pdf/2512.13313)

> Avatar video generation models have achieved remarkable progress in recent years. However, prior work exhibits limited efficiency in generating long-duration high-resolution videos, suffering from temporal drifting, quality degradation, and weak p...

</details>

<details>
<summary><b>10. MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09636) • [📄 arXiv](https://arxiv.org/abs/2512.09636) • [📥 PDF](https://arxiv.org/pdf/2512.09636)

**💻 Code:** [⭐ Code](https://github.com/elsa66666/MentraSuite)

> No abstract available.

</details>

<details>
<summary><b>11. Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge</b> ⭐ 148</summary>

<br/>

**👥 Authors:** Jinwei Gu, Qizhi Chen, Yu-Wei Chao, Junjie Bai, delinqu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10071) • [📄 arXiv](https://arxiv.org/abs/2512.10071) • [📥 PDF](https://arxiv.org/pdf/2512.10071)

**💻 Code:** [⭐ Code](https://github.com/mli0603/openpi-comet)

> OpenPi Comet is the submission of Team Comet for the 2025 BEHAVIOR Challenge . We provides a unified framework for pre-training, post-training, data generation and evaluation of π0.5 (Pi05) models on BEHAVIOR-1K. 📄 Arxiv: https://arxiv.org/pdf/251...

</details>

<details>
<summary><b>12. Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13080) • [📄 arXiv](https://arxiv.org/abs/2512.13080) • [📥 PDF](https://arxiv.org/pdf/2512.13080)

**💻 Code:** [⭐ Code](https://github.com/BeingBeyond/VIPA-VLA)

> We propose VIPA-VLA , which learns 2D–to–3D visual–physical grounding from human videos with Spatial-Aware VLA Pretraining, enabling robot policies with stronger spatial understanding and generalization. Website: https://beingbeyond.github.io/VIPA...

</details>

<details>
<summary><b>13. WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Md Rizwan Parvez, Mohammed Eunus Ali, Tanzima Hashem, mahirlabibdihan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12692) • [📄 arXiv](https://arxiv.org/abs/2512.12692) • [📥 PDF](https://arxiv.org/pdf/2512.12692)

**💻 Code:** [⭐ Code](https://github.com/kagnlp/WebOperator)

> We are excited to share our recent work titled "WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment". 📃 Paper: https://arxiv.org/abs/2512.12692 💻 Code: https://github.com/kagnlp/WebOperator 🏠 Homepage: https://kagnlp.git...

</details>

<details>
<summary><b>14. DrivePI: Spatial-aware 4D MLLM for Unified Autonomous Driving Understanding, Perception, Prediction and Planning</b> ⭐ 22</summary>

<br/>

**👥 Authors:** Zining Wang, Siming Yan, Rui Yang, Runhui Huang, happinessqq

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12799) • [📄 arXiv](https://arxiv.org/abs/2512.12799) • [📥 PDF](https://arxiv.org/pdf/2512.12799)

**💻 Code:** [⭐ Code](https://github.com/happinesslz/DrivePI)

> Although multi-modal large language models (MLLMs) have shown strong capabilities across diverse domains, their application in generating fine-grained 3D perception and prediction outputs in autonomous driving remains underexplored. In this paper,...

</details>

<details>
<summary><b>15. V-REX: Benchmarking Exploratory Visual Reasoning via Chain-of-Questions</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Kwesi Cobbina, Shweta Bhardwaj, Yijun Liang, zhoutianyi, Fcr09

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11995) • [📄 arXiv](https://arxiv.org/abs/2512.11995) • [📥 PDF](https://arxiv.org/pdf/2512.11995)

**💻 Code:** [⭐ Code](https://github.com/tianyi-lab/VREX)

> While many vision-language models (VLMs) are developed to answer well-defined, straightforward questions with highly specified targets, as in most benchmarks, they often struggle in practice with complex open-ended tasks, which usually require mul...

</details>

<details>
<summary><b>16. Toward Ambulatory Vision: Learning Visually-Grounded Active View Selection</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13250) • [📄 arXiv](https://arxiv.org/abs/2512.13250) • [📥 PDF](https://arxiv.org/pdf/2512.13250)

**💻 Code:** [⭐ Code](https://github.com/KAIST-Visual-AI-Group/VG-AVS)

> Project page: https://active-view-selection.github.io Arxiv: https://arxiv.org/abs/2512.13250 Code: https://github.com/KAIST-Visual-AI-Group/VG-AVS

</details>

<details>
<summary><b>17. Image Diffusion Preview with Consistency Solver</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13592) • [📄 arXiv](https://arxiv.org/abs/2512.13592) • [📥 PDF](https://arxiv.org/pdf/2512.13592)

**💻 Code:** [⭐ Code](https://github.com/G-U-N/consolver)

> The slow inference process of image diffusion models significantly degrades interactive user experiences. We introduce Diffusion Preview , a novel preview-and-refine paradigm that generates rapid, low-step preliminary outputs for user evaluation, ...

</details>

<details>
<summary><b>18. VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zihan Meng, Jun Cen, Shuang Liu, Zeyi Liu, Songqiao Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11891) • [📄 arXiv](https://arxiv.org/abs/2512.11891) • [📥 PDF](https://arxiv.org/pdf/2512.11891)

> Project Page: https://vlsa-aegis.github.io

</details>

<details>
<summary><b>19. GenieDrive: Towards Physics-Aware Driving World Model with 4D Occupancy Guided Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chenxuan Miao, Liping Hou, Yuxiang Lu, Zhe Liu, ANIYA673

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12751) • [📄 arXiv](https://arxiv.org/abs/2512.12751) • [📥 PDF](https://arxiv.org/pdf/2512.12751)

> No abstract available.

</details>

<details>
<summary><b>20. Aesthetic Alignment Risks Assimilation: How Image Generation and Reward Models Reinforce Beauty Bias and Ideological "Censorship"</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shan Du, Khalad Hasan, Qingyun Qian, weathon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11883) • [📄 arXiv](https://arxiv.org/abs/2512.11883) • [📥 PDF](https://arxiv.org/pdf/2512.11883)

> Over-aligning image generation models to a generalized aesthetic preference conflicts with user intent, particularly when ``anti-aesthetic" outputs are requested for artistic or critical purposes. This adherence prioritizes developer-centered valu...

</details>

<details>
<summary><b>21. Towards Interactive Intelligence for Digital Humans</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yifei Huang, Sitong Gong, Xiwei Gao, Xuangeng Chu, Yiyi Cai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13674) • [📄 arXiv](https://arxiv.org/abs/2512.13674) • [📥 PDF](https://arxiv.org/pdf/2512.13674)

> We introduce Interactive Intelligence, a novel paradigm of digital human that is capable of personality-aligned expression, adaptive interaction, and self-evolution. To realize this, we present Mio (Multimodal Interactive Omni-Avatar), an end-to-e...

</details>

<details>
<summary><b>22. RecTok: Reconstruction Distillation along Rectified Flow</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Yujing Wang, Kaidong Yu, Size Wu, BryanW, QingyuShi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13421) • [📄 arXiv](https://arxiv.org/abs/2512.13421) • [📥 PDF](https://arxiv.org/pdf/2512.13421)

**💻 Code:** [⭐ Code](https://github.com/Shi-qingyu/RecTok)

> arXiv: https://arxiv.org/abs/2512.13421 Project: https://shi-qingyu.github.io/rectok.github.io/ Code: https://github.com/Shi-qingyu/RecTok

</details>

<details>
<summary><b>23. Few-Step Distillation for Text-to-Image Generation: A Practical Guide</b> ⭐ 91</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13006) • [📄 arXiv](https://arxiv.org/abs/2512.13006) • [📥 PDF](https://arxiv.org/pdf/2512.13006)

**💻 Code:** [⭐ Code](https://github.com/alibaba-damo-academy/T2I-Distill.git)

> A Systematic Study of Diffusion Distillation for Text-to-Image Synthesis towards truly applicable few steps distillation, casting existing distillation methods (sCM, MeanFlow and IMM) into a unified framework for fair comparison. Code is available...

</details>

<details>
<summary><b>24. Flowception: Temporally Expansive Flow Matching for Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11438) • [📄 arXiv](https://arxiv.org/abs/2512.11438) • [📥 PDF](https://arxiv.org/pdf/2512.11438)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API TempoMaster: Efficient Long Video Generation via Next-Frame-Rate Prediction...

</details>

<details>
<summary><b>25. What matters for Representation Alignment: Global Information or Spatial Structure?</b> ⭐ 80</summary>

<br/>

**👥 Authors:** Richard Zhang, Liang Zheng, Zongze Wu, Xingjian Leng, Jaskirat Singh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10794) • [📄 arXiv](https://arxiv.org/abs/2512.10794) • [📥 PDF](https://arxiv.org/pdf/2512.10794)

**💻 Code:** [⭐ Code](https://github.com/end2end-diffusion/irepa)

> No abstract available.

</details>

<details>
<summary><b>26. CAPTAIN: Semantic Feature Injection for Memorization Mitigation in Text-to-Image Diffusion Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10655) • [📄 arXiv](https://arxiv.org/abs/2512.10655) • [📥 PDF](https://arxiv.org/pdf/2512.10655)

> Diffusion models can unintentionally reproduce training examples, raising privacy and copyright concerns as these systems are increasingly deployed at scale. Existing inference-time mitigation methods typically manipulate classifier-free guidance ...

</details>

<details>
<summary><b>27. DiffusionBrowser: Interactive Diffusion Previews via Multi-Branch Decoders</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jui-Hsien Wang, Zhifei Zhang, Chongjian Ge, Susung Hong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13690) • [📄 arXiv](https://arxiv.org/abs/2512.13690) • [📥 PDF](https://arxiv.org/pdf/2512.13690)

> Project page: https://susunghong.github.io/DiffusionBrowser

</details>

<details>
<summary><b>28. LitePT: Lighter Yet Stronger Point Transformer</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13689) • [📄 arXiv](https://arxiv.org/abs/2512.13689) • [📥 PDF](https://arxiv.org/pdf/2512.13689)

**💻 Code:** [⭐ Code](https://github.com/prs-eth/LitePT)

> LitePT: Lighter Yet Stronger Point Transformer LitePT is a lightweight, high-performance 3D point cloud architecture for various point cloud processing tasks. It embodies the simple principle "convolutions for low-level geometry, attention for hig...

</details>

<details>
<summary><b>29. I-Scene: 3D Instance Models are Implicit Generalizable Spatial Learners</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Aniket Bera, Yichen Sheng, Yunhao Ge, Lu Ling

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13683) • [📄 arXiv](https://arxiv.org/abs/2512.13683) • [📥 PDF](https://arxiv.org/pdf/2512.13683)

> No abstract available.

</details>

<details>
<summary><b>30. Directional Textual Inversion for Personalized Text-to-Image Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13672) • [📄 arXiv](https://arxiv.org/abs/2512.13672) • [📥 PDF](https://arxiv.org/pdf/2512.13672)

**💻 Code:** [⭐ Code](https://github.com/kunheek/dti)

> Hi everyone! 👋 We investigated why Textual Inversion (TI) often ignores context and traced the issue to embedding norm inflation. We found that standard TI learns tokens with massive magnitudes (often >20) compared to the model's native vocabulary...

</details>

<details>
<summary><b>31. AutoMV: An Automatic Multi-Agent System for Music Video Generation</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12196) • [📄 arXiv](https://arxiv.org/abs/2512.12196v1) • [📥 PDF](https://arxiv.org/pdf/2512.12196)

**💻 Code:** [⭐ Code](https://github.com/multimodal-art-projection/AutoMV)

> arxiv: https://arxiv.org/abs/2512.12196v1 GitHub: https://github.com/multimodal-art-projection/AutoMV Website: https://m-a-p.ai/AutoMV/ Apache-2.0 license

</details>

<details>
<summary><b>32. FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.10927) • [📄 arXiv](https://arxiv.org/abs/2512.10927) • [📥 PDF](https://arxiv.org/pdf/2512.10927)

**💻 Code:** [⭐ Code](https://github.com/Wolfv0/FoundationMotion/tree/main)

> FoundationMotion offers a scalable way to curate detailed motion datasets, enabling effective fine-tuning of diverse models (VLM / VLA / world models) to improve motion and spatial reasoning

</details>

<details>
<summary><b>33. START: Spatial and Textual Learning for Chart Understanding</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.07186) • [📄 arXiv](https://arxiv.org/abs/2512.07186) • [📥 PDF](https://arxiv.org/pdf/2512.07186)

**💻 Code:** [⭐ Code](https://github.com/dragonlzm/START)

> Does visual grounding help visual reasoning in Chart Understanding? 📊🧠 I am excited to share our latest paper, "START: Spatial and Textual Learning for Chart Understanding," which explores how we can teach Multimodal LLMs (MLLMs) to better underst...

</details>

<details>
<summary><b>34. Inferring Compositional 4D Scenes without Ever Seeing One</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.05272) • [📄 arXiv](https://arxiv.org/abs/2512.05272) • [📥 PDF](https://arxiv.org/pdf/2512.05272)

**💻 Code:** [⭐ Code](https://github.com/insait-institute/COM4D)

> Our method turns videos into compositional 4D scenes with explicit meshes.

</details>

<details>
<summary><b>35. FIN-bench-v2: A Unified and Robust Benchmark Suite for Evaluating Finnish Large Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.13330) • [📄 arXiv](https://arxiv.org/abs/2511.01066) • [📥 PDF](https://arxiv.org/pdf/2512.13330)

**💻 Code:** [⭐ Code](https://github.com/LumiOpen/lm-evaluation-harness/tree/main/lm_eval/tasks/finbench_v2)

> Our paper introduces FIN-bench-v2, a unified and robust benchmark suite for evaluating large language models in Finnish, addressing the scarcity of high-quality evaluation resources for low-resource languages. This new suite modernizes the origina...

</details>

<details>
<summary><b>36. State over Tokens: Characterizing the Role of Reasoning Tokens</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yoav Goldberg, Shauli Ravfogel, Zohar Elyoseph, Mosh Levy

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12777) • [📄 arXiv](https://arxiv.org/abs/2512.12777) • [📥 PDF](https://arxiv.org/pdf/2512.12777)

> One of the most captivating features of recent chatbot models is their apparent transparency when they "think" out loud, generating step-by-step text before their answer. This might suggest we can trust them because we can verify their logic, but ...

</details>

<details>
<summary><b>37. CoRe3D: Collaborative Reasoning as a Foundation for 3D Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.12768) • [📄 arXiv](https://arxiv.org/abs/2512.12768) • [📥 PDF](https://arxiv.org/pdf/2512.12768)

> A dual semantic + geometric reasoning framework with octant-based 3D tokens and multi-critic GRPO, achieving SoTA on text-to-3D, image-to-3D, and 3D captioning.

</details>

<details>
<summary><b>38. Rethinking Expert Trajectory Utilization in LLM Post-training</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Qi Zhu, Jiyao Yuan, Jiayang Lv, Yuhan Chen, Bowen Ding

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.11470) • [📄 arXiv](https://arxiv.org/abs/2512.11470) • [📥 PDF](https://arxiv.org/pdf/2512.11470)

**💻 Code:** [⭐ Code](https://github.com/LINs-lab/RETU)

> The systematic study of expert trajectory utilization in LLM post-training.

</details>

<details>
<summary><b>39. Learning Robot Manipulation from Audio World Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Michael Gienger, Fanzhri

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08405) • [📄 arXiv](https://arxiv.org/abs/2512.08405) • [📥 PDF](https://arxiv.org/pdf/2512.08405)

> Paper page: https://arxiv.org/abs/2409.01083

</details>

<details>
<summary><b>40. Towards Visual Re-Identification of Fish using Fine-Grained Classification for Electronic Monitoring in Fisheries</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.08400) • [📄 arXiv](https://arxiv.org/abs/2512.08400) • [📥 PDF](https://arxiv.org/pdf/2512.08400)

**💻 Code:** [⭐ Code](https://github.com/msamdk/Fish_Re_Identification.git)

> Link to the AutoFish dataset: https://huggingface.co/datasets/vapaau/autofish

</details>

<details>
<summary><b>41. KD-OCT: Efficient Knowledge Distillation for Clinical-Grade Retinal OCT Classification</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ali Nourbakhsh, Nasrin Sanjari, Erfan-Nourbakhsh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.09069) • [📄 arXiv](https://arxiv.org/abs/2512.09069) • [📥 PDF](https://arxiv.org/pdf/2512.09069)

**💻 Code:** [⭐ Code](https://github.com/erfan-nourbakhsh/KD-OCT)

> KD-OCT: Efficient Knowledge Distillation for Clinical-Grade Retinal OCT Classification

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 41 |
| 📅 Today | [`2025-12-17.json`](data/daily/2025-12-17.json) | 41 |
| 📆 This Week | [`2025-W50.json`](data/weekly/2025-W50.json) | 87 |
| 🗓️ This Month | [`2025-12.json`](data/monthly/2025-12.json) | 460 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2025-12-17 | 41 | [View JSON](data/daily/2025-12-17.json) |
| 📄 2025-12-16 | 21 | [View JSON](data/daily/2025-12-16.json) |
| 📄 2025-12-15 | 25 | [View JSON](data/daily/2025-12-15.json) |
| 📄 2025-12-14 | 25 | [View JSON](data/daily/2025-12-14.json) |
| 📄 2025-12-13 | 24 | [View JSON](data/daily/2025-12-13.json) |
| 📄 2025-12-12 | 21 | [View JSON](data/daily/2025-12-12.json) |
| 📄 2025-12-11 | 25 | [View JSON](data/daily/2025-12-11.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2025-W50 | 87 | [View JSON](data/weekly/2025-W50.json) |
| 📅 2025-W49 | 186 | [View JSON](data/weekly/2025-W49.json) |
| 📅 2025-W48 | 187 | [View JSON](data/weekly/2025-W48.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2025-12 | 460 | [View JSON](data/monthly/2025-12.json) |

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
