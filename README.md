<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-40-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1649+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">40</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">85</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">130</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1649+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 03, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas</b> ⭐ 84</summary>

<br/>

**👥 Authors:** Hao Zhou, Shuaiting Chen, Haotian Wang, jade0101, Emperorizzis

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21558) • [📄 arXiv](https://arxiv.org/abs/2601.21558) • [📥 PDF](https://arxiv.org/pdf/2601.21558)

**💻 Code:** [⭐ Code](https://github.com/LianjiaTech/astra)

> ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas

</details>

<details>
<summary><b>2. Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22813) • [📄 arXiv](https://arxiv.org/abs/2601.22813) • [📥 PDF](https://arxiv.org/pdf/2601.22813)

**💻 Code:** [⭐ Code](https://github.com/IST-DASLab/Quartet-II)

> A SOTA NVFP4 LLM pre-training method based on MS-EDEN unbiased gradient estimation. Code is available on GitHub .

</details>

<details>
<summary><b>3. Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22975) • [📄 arXiv](https://arxiv.org/abs/2601.22975) • [📥 PDF](https://arxiv.org/pdf/2601.22975)

> TL;DR: We introduce Golden Goose 🦢, a simple method that synthesizes unlimited RLVR tasks from unverifiable internet text by constructing multiple-choice fill-in-the-middle problems. This enables the use of reasoning-rich unverifiable corpora typi...

</details>

<details>
<summary><b>4. THINKSAFE: Self-Generated Safety Alignment for Reasoning Models</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Minki Kang, Gyeongman Kim, YuminChoi, Sangsang, Seanie-lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23143) • [📄 arXiv](https://arxiv.org/abs/2601.23143) • [📥 PDF](https://arxiv.org/pdf/2601.23143)

**💻 Code:** [⭐ Code](https://github.com/seanie12/ThinkSafe.git)

> THINKSAFE: Self-Generated Safety Alignment for Reasoning Models

</details>

<details>
<summary><b>5. TTCS: Test-Time Curriculum Synthesis for Self-Evolving</b> ⭐ 19</summary>

<br/>

**👥 Authors:** Chengsong Huang, Zongpei Teng, Yunbo Tang, Zhishang Xiang, ChengyiYang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22628) • [📄 arXiv](https://arxiv.org/abs/2601.22628) • [📥 PDF](https://arxiv.org/pdf/2601.22628)

**💻 Code:** [⭐ Code](https://github.com/XMUDeepLIT/TTCS)

> TTCS, a new paradigm for self-evolving

</details>

<details>
<summary><b>6. PaperBanana: Automating Academic Illustration for AI Scientists</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23265) • [📄 arXiv](https://arxiv.org/abs/2601.23265) • [📥 PDF](https://arxiv.org/pdf/2601.23265)

> PaperBanana automates publication-ready AI research illustrations via an agentic framework using VLMs and image models, orchestrating reference retrieval, planning, rendering, and self-critique with a benchmarking suite.

</details>

<details>
<summary><b>7. Do Reasoning Models Enhance Embedding Models?</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Elton Chun-Chai Li, Kwun Hang Lau, Huihao Jing, Shaojin Chen, lucaswychan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21192) • [📄 arXiv](https://arxiv.org/abs/2601.21192) • [📥 PDF](https://arxiv.org/pdf/2601.21192)

**💻 Code:** [⭐ Code](https://github.com/HKUST-KnowComp/Reasoning-Embedding)

> Our analysis revealed a phenomenon we term Manifold Realignment. RLVR is a Trajectory Optimizer : We found that RLVR irreversibly reorganizes the local geometry of the latent manifold but largely preserves the global manifold geometry (the overall...

</details>

<details>
<summary><b>8. FourierSampler: Unlocking Non-Autoregressive Potential in Diffusion Language Models via Frequency-Guided Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23182) • [📄 arXiv](https://arxiv.org/abs/2601.23182) • [📥 PDF](https://arxiv.org/pdf/2601.23182)

**💻 Code:** [⭐ Code](https://github.com/ShirleYoung/FourierSampler)

> Despite the non-autoregressive potential of diffusion language models (dLLMs), existing decoding strategies demonstrate positional bias, failing to fully unlock the potential of arbitrary generation. In this work, we delve into the inherent spectr...

</details>

<details>
<summary><b>9. Causal World Modeling for Robot Control</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruilin Wang, Shuai Yang, Yiming Luo, Qihang Zhang, Lin Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21998) • [📄 arXiv](https://arxiv.org/abs/2601.21998) • [📥 PDF](https://arxiv.org/pdf/2601.21998)

> No abstract available.

</details>

<details>
<summary><b>10. ReGuLaR: Variational Latent Reasoning Guided by Rendered Chain-of-Thought</b> ⭐ 15</summary>

<br/>

**👥 Authors:** Zhifeng Gao, Hongteng Xu, Guojiang Zhao, Haotian Liu, FanmengWang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23184) • [📄 arXiv](https://arxiv.org/abs/2601.23184) • [📥 PDF](https://arxiv.org/pdf/2601.23184)

**💻 Code:** [⭐ Code](https://github.com/FanmengWang/ReGuLaR)

> Introduces ReGuLaR, a variational latent reasoning framework that renders reasoning as images to regularize posterior inference, achieving efficient multimodal reasoning beyond traditional chain of thought.

</details>

<details>
<summary><b>11. DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Context Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21716) • [📄 arXiv](https://arxiv.org/abs/2601.21716) • [📥 PDF](https://arxiv.org/pdf/2601.21716)

> Character image animation aims to synthesize high-fidelity videos by transferring motion from a driving sequence to a static reference image. Despite recent advancements, existing methods suffer from two fundamental challenges: (1) suboptimal moti...

</details>

<details>
<summary><b>12. SSL: Sweet Spot Learning for Differentiated Guidance in Agentic Optimization</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bolin Ni, Fangzhi Xu, Yuhao Shen, Jinyang Wu, thkelper

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22491) • [📄 arXiv](https://arxiv.org/abs/2601.22491) • [📥 PDF](https://arxiv.org/pdf/2601.22491)

> No abstract available.

</details>

<details>
<summary><b>13. DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20218) • [📄 arXiv](https://arxiv.org/abs/2601.20218) • [📥 PDF](https://arxiv.org/pdf/2601.20218)

> A dense reward for RL in flow matching models.

</details>

<details>
<summary><b>14. Statistical Estimation of Adversarial Risk in Large Language Models under Best-of-N Sampling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22636) • [📄 arXiv](https://arxiv.org/abs/2601.22636) • [📥 PDF](https://arxiv.org/pdf/2601.22636)

> Real-world jailbreak attackers don’t usually try once. They try many times, in parallel, until the model slips. That’s why adversarial risk can’t be captured by attack success rate on a single attempt (ASR@1). As the number of attempts N grows, ri...

</details>

<details>
<summary><b>15. DINO-SAE: DINO Spherical Autoencoder for High-Fidelity Image Reconstruction and Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jong Chul Ye, Byunghee Cha, Hun Chang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22904) • [📄 arXiv](https://arxiv.org/abs/2601.22904) • [📥 PDF](https://arxiv.org/pdf/2601.22904)

> DINO-SAE bridges semantic directions and pixel fidelity via spherical latent diffusion with hierarchical patch embedding and cosine alignment, achieving state-of-the-art reconstruction while preserving semantic alignment.

</details>

<details>
<summary><b>16. PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zelun Zhang, Tingquan Gao, Suyin Liang, sunflowerting78, ChengCui

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21957) • [📄 arXiv](https://arxiv.org/abs/2601.21957) • [📥 PDF](https://arxiv.org/pdf/2601.21957)

> No abstract available.

</details>

<details>
<summary><b>17. RM -RF: Reward Model for Run-Free Unit Test Evaluation</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Mikhail Klementev, doooori, rmndrnts, dangrebenkin, brucheselena

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.13097) • [📄 arXiv](https://arxiv.org/abs/2601.13097) • [📥 PDF](https://arxiv.org/pdf/2601.13097)

**💻 Code:** [⭐ Code](https://github.com/trndcenter/RM-RF-unit-tests)

> RM -RF: Reward Model for Run-Free Unit Test Evaluation proposes a novel lightweight reward model  that predicts unit test quality without compiling or executing code by inferring three execution-derived signals directly from source and test code: ...

</details>

<details>
<summary><b>18. DIFFA-2: A Practical Diffusion Large Language Model for General Audio Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23161) • [📄 arXiv](https://arxiv.org/abs/2601.23161) • [📥 PDF](https://arxiv.org/pdf/2601.23161)

**💻 Code:** [⭐ Code](https://github.com/NKU-HLT/DIFFA)

> DIFFA-2 provides a practical diffusion-based large audio language model with semantic/acoustic adapters and a four-stage curriculum, improving general audio understanding under practical budgets.

</details>

<details>
<summary><b>19. NativeTok: Native Visual Tokenization for Improved Image Generation</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Zhendong Mao, Weinan Jia, Mengqi Huang, Bin Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22837) • [📄 arXiv](https://arxiv.org/abs/2601.22837) • [📥 PDF](https://arxiv.org/pdf/2601.22837)

**💻 Code:** [⭐ Code](https://github.com/wangbei1/Nativetok)

> Introduces native visual tokenization with causal dependencies, via NativeTok (MIT and MoCET) and hierarchical training for efficient, coherent image reconstruction with relational token constraints.

</details>

<details>
<summary><b>20. Pushing the Boundaries of Natural Reasoning: Interleaved Bonus from Formal-Logic Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22642) • [📄 arXiv](https://arxiv.org/abs/2601.22642) • [📥 PDF](https://arxiv.org/pdf/2601.22642)

> No abstract available.

</details>

<details>
<summary><b>21. MemOCR: Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuxin Chen, Wenyu Mao, Yu Yang, Shugui Liu, Yaorui Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21468) • [📄 arXiv](https://arxiv.org/abs/2601.21468) • [📥 PDF](https://arxiv.org/pdf/2601.21468)

> MemOCR is a multimodal memory agent that enhances long-horizon reasoning by adaptively compressing interaction histories into visual layouts, enabling efficient context utilization under tight budget constraints.

</details>

<details>
<summary><b>22. TAM-Eval: Evaluating LLMs for Automated Unit Test Maintenance</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Vadim Alperovich, dangrebenkin, rmndrnts, doooori, brucheselena

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.18241) • [📄 arXiv](https://arxiv.org/abs/2601.18241) • [📥 PDF](https://arxiv.org/pdf/2601.18241)

**💻 Code:** [⭐ Code](https://github.com/trndcenter/TAM-Eval)

> 🧪 TAM-Eval: Evaluating LLMs for Automated Unit Test Maintenance What’s new: Large Language Models (LLMs) have been widely explored for unit test generation , but real-world test suite maintenance — like creating, updating, and repairing tests as c...

</details>

<details>
<summary><b>23. Scaling Multiagent Systems with Process Rewards</b> ⭐ 43</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23228) • [📄 arXiv](https://arxiv.org/abs/2601.23228) • [📥 PDF](https://arxiv.org/pdf/2601.23228)

**💻 Code:** [⭐ Code](https://github.com/ltjed/multiagent-coaching)

> Define and train your own multiagent system @ our github repo !

</details>

<details>
<summary><b>24. Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21358) • [📄 arXiv](https://arxiv.org/abs/2601.21358) • [📥 PDF](https://arxiv.org/pdf/2601.21358)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Forest Before Trees: Latent Superposition for Efficient Visual Reasoning (2...

</details>

<details>
<summary><b>25. Deep Search with Hierarchical Meta-Cognitive Monitoring Inspired by Cognitive Neuroscience</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23188) • [📄 arXiv](https://arxiv.org/abs/2601.23188) • [📥 PDF](https://arxiv.org/pdf/2601.23188)

> 🚀 New Research Alert! 🧠✨ We’re excited to share our latest work: Deep Search with Hierarchical Meta-Cognitive Monitoring Inspired by Cognitive Neuroscience 🔍 What’s the key idea? Deep search agents powered by LLMs excel at multi-step reasoning and...

</details>

<details>
<summary><b>26. Revisiting Diffusion Model Predictions Through Dimensionality</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chaoyang Wang, Qing Jin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21419) • [📄 arXiv](https://arxiv.org/abs/2601.21419) • [📥 PDF](https://arxiv.org/pdf/2601.21419)

> not sure which to choose: x0 prediction or velocity prediction? this paper provides a universal solution to find the optimal solution for you

</details>

<details>
<summary><b>27. Real-Time Aligned Reward Model beyond Semantics</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jianbin Zheng, Yuxi Ren, Xin Xia, Yikunb, hzxllll

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22664) • [📄 arXiv](https://arxiv.org/abs/2601.22664) • [📥 PDF](https://arxiv.org/pdf/2601.22664)

> RLHF is central to aligning LLMs with human preferences, but it often suffers from reward overoptimization: the policy learns to game the reward model instead of truly following human intent. A key reason? Distribution shift—the policy keeps chang...

</details>

<details>
<summary><b>28. LMK > CLS: Landmark Pooling for Dense Embeddings</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yulong Li, Parul Awasthy, Aashka Trivedi, vishwajeetkumar, meetdoshi90

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21525) • [📄 arXiv](https://arxiv.org/abs/2601.21525) • [📥 PDF](https://arxiv.org/pdf/2601.21525)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in De...

</details>

<details>
<summary><b>29. Continual GUI Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.20732) • [📄 arXiv](https://arxiv.org/abs/2601.20732) • [📥 PDF](https://arxiv.org/pdf/2601.20732)

> Some of the observations founded are :- -- Static GUI training breaks under real world change : GUI agents trained on fixed datasets degrade badly when UI domains (mobile --> desktop --> web) or resolutions (1080p -> 4K) shift, mainly due to unsta...

</details>

<details>
<summary><b>30. Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bin Liang, Zezhong Wang, Rui Wang, Zhiwei Zhang, Hiiamein

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15625) • [📄 arXiv](https://arxiv.org/abs/2601.15625) • [📥 PDF](https://arxiv.org/pdf/2601.15625)

> Robust Tool Use via FISSION-GRPO: Learning to Recover from Execution Errors

</details>

<details>
<summary><b>31. Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Michal Byra, Alberto Presta, GrzegorzStefanski

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22141) • [📄 arXiv](https://arxiv.org/abs/2601.22141) • [📥 PDF](https://arxiv.org/pdf/2601.22141)

> This work challenges a core assumption of the Lottery Ticket Hypothesis: that a single sparse subnetwork can serve all data. The authors show that under heterogeneity, multiple specialized winning tickets outperform a universal one, reframing prun...

</details>

<details>
<summary><b>32. Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving</b> ⭐ 32</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22032) • [📄 arXiv](https://arxiv.org/abs/2601.22032) • [📥 PDF](https://arxiv.org/pdf/2601.22032)

**💻 Code:** [⭐ Code](https://github.com/linhanwang/Drive-JEPA)

> End-to-end autonomous driving increasingly leverages self-supervised video pretraining to learn transferable planning representations. However, pretraining video world models for scene understanding has so far brought only limited improvements. Th...

</details>

<details>
<summary><b>33. Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xialiang Tong, Yinqi Bai, Xing Li, Jie Wang, Qingyue Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21709) • [📄 arXiv](https://arxiv.org/abs/2601.21709) • [📥 PDF](https://arxiv.org/pdf/2601.21709)

> We systematically analyze attention patterns from a unified temporal perspective and find that embedding temporal self-similarity and RoPE are key factors underlying streaming, retrieval, seasonal, and reaccess attention patterns. We further apply...

</details>

<details>
<summary><b>34. Memorization Dynamics in Knowledge Distillation for Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.15394) • [📄 arXiv](https://arxiv.org/abs/2601.15394) • [📥 PDF](https://arxiv.org/pdf/2601.15394)

> We show that knowledge distillation in language models can give both improved generalization and reduced memorization.

</details>

<details>
<summary><b>35. Machine Learning for Energy-Performance-aware Scheduling</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yifei Shi, Peter2023HuggingFace

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23134) • [📄 arXiv](https://arxiv.org/abs/2601.23134) • [📥 PDF](https://arxiv.org/pdf/2601.23134)

**💻 Code:** [⭐ Code](https://github.com/PeterHUistyping/ml-cpu-sched)

> Machine Learning for Energy-Performance-aware Scheduling. @ misc {HuShi2026mlcpusched,
      title={Machine Learning for Energy-Performance-aware Scheduling}, 
      author={Zheyuan Hu and Yifei Shi},
      year={2026},
      eprint={2601.23134},
...

</details>

<details>
<summary><b>36. Visual Personalization Turing Test</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kuan-Chieh Jackson Wang, Sergey Tulyakov, James Burgess, Rameen Abdal

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22680) • [📄 arXiv](https://arxiv.org/abs/2601.22680) • [📥 PDF](https://arxiv.org/pdf/2601.22680)

> No abstract available.

</details>

<details>
<summary><b>37. ExpAlign: Expectation-Guided Vision-Language Alignment for Open-Vocabulary Grounding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22666) • [📄 arXiv](https://arxiv.org/abs/2601.22666) • [📥 PDF](https://arxiv.org/pdf/2601.22666)

> Open-vocabulary grounding requires accurate vision-language alignment under weak supervision, yet existing methods either rely on global sentence embeddings that lack fine-grained expressiveness or introduce token-level alignment with explicit sup...

</details>

<details>
<summary><b>38. Value-Based Pre-Training with Downstream Feedback</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22108) • [📄 arXiv](https://arxiv.org/abs/2601.22108) • [📥 PDF](https://arxiv.org/pdf/2601.22108)

> We’re entering the age of research, not just the age of scaling. Bigger models gave us horsepower. But pretraining still has almost no steering wheel. Today’s foundation models learn in an open loop: pick a proxy objective (next‑token / fixed augm...

</details>

<details>
<summary><b>39. SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21666) • [📄 arXiv](https://arxiv.org/abs/2601.21666) • [📥 PDF](https://arxiv.org/pdf/2601.21666)

> SONIC-O1: A Real-World Benchmark for Evaluating Multimodal LLMs on Audio-Video Understanding SONIC-O1 is a fully human-verified benchmark for real-world audio–video conversations: 13 conversational domains, 4,958 annotated instances, plus demograp...

</details>

<details>
<summary><b>40. KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization</b> ⭐ 56</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21526) • [📄 arXiv](https://arxiv.org/abs/2601.21526) • [📥 PDF](https://arxiv.org/pdf/2601.21526)

**💻 Code:** [⭐ Code](https://github.com/Leeroo-AI/kapso)

> We introduce KAPSO, a modular framework for autonomous program synthesis and optimization. Given a natural language goal and an evaluation method, KAPSO iteratively performs ideation, code synthesis and editing, execution, evaluation, and learning...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 40 |
| 📅 Today | [`2026-02-03.json`](data/daily/2026-02-03.json) | 40 |
| 📆 This Week | [`2026-W05.json`](data/weekly/2026-W05.json) | 85 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 130 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-03 | 40 | [View JSON](data/daily/2026-02-03.json) |
| 📄 2026-02-02 | 45 | [View JSON](data/daily/2026-02-02.json) |
| 📄 2026-02-01 | 45 | [View JSON](data/daily/2026-02-01.json) |
| 📄 2026-01-31 | 45 | [View JSON](data/daily/2026-01-31.json) |
| 📄 2026-01-30 | 21 | [View JSON](data/daily/2026-01-30.json) |
| 📄 2026-01-29 | 21 | [View JSON](data/daily/2026-01-29.json) |
| 📄 2026-01-28 | 37 | [View JSON](data/daily/2026-01-28.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W05 | 85 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 130 | [View JSON](data/monthly/2026-02.json) |
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
