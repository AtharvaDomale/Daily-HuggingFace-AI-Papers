<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-33-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-902+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">33</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">123</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">164</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">902+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 10, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization</b> ⭐ 64</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05242) • [📄 arXiv](https://arxiv.org/abs/2601.05242) • [📥 PDF](https://arxiv.org/pdf/2601.05242)

**💻 Code:** [⭐ Code](https://github.com/NVlabs/GDPO)

> GDPO is a drop-in replacement for GRPO in verl and TRL — only minor code changes needed. We release a slurm-free, easy-to-run implementation supporting multiple RL frameworks (verl / TRL / NeMo-RL) so you can quickly validate GDPO on tool-calling ...

</details>

<details>
<summary><b>2. Learnable Multipliers: Freeing the Scale of Language Model Matrix Layers</b> ⭐ 98</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04890) • [📄 arXiv](https://arxiv.org/abs/2601.04890) • [📥 PDF](https://arxiv.org/pdf/2601.04890)

**💻 Code:** [⭐ Code](https://github.com/tiiuae/falcon-h1)

> Building on the μP multipliers applied in Falcon-H1 pretraining ( https://huggingface.co/papers/2507.22448 ), this work extends the idea to learnable matrix-, row-, and column-wise scaling. We show that the weight-norm equilibrium induced by weigh...

</details>

<details>
<summary><b>3. RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes</b> ⭐ 12</summary>

<br/>

**👥 Authors:** Chia-Che Chang, Kuan-Lin Chen, yulunliu, NeilLeeNTU

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05249) • [📄 arXiv](https://arxiv.org/abs/2601.05249) • [📥 PDF](https://arxiv.org/pdf/2601.05249)

**💻 Code:** [⭐ Code](https://github.com/BrianChen1120/RL-AWB)

> Nighttime color constancy remains a challenging problem in computational photography due to low-light noise and complex illumination conditions. We present RL-AWB, a novel framework combining statistical methods with deep reinforcement learning fo...

</details>

<details>
<summary><b>4. Token-Level LLM Collaboration via FusionRoute</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Furong Huang, Zhaorun Chen, Hanqing Zeng, Nuoya Xiong, zyhang1998

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05106) • [📄 arXiv](https://arxiv.org/abs/2601.05106) • [📥 PDF](https://arxiv.org/pdf/2601.05106)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API LLMBoost: Make Large Language Models Stronger with Boosting (2025) SDA: Ste...

</details>

<details>
<summary><b>5. RoboVIP: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Jia-Zeng, ZhaoyangLyu, matthewmao, wuzhi-hao, HikariDawn

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05241) • [📄 arXiv](https://arxiv.org/abs/2601.05241) • [📥 PDF](https://arxiv.org/pdf/2601.05241)

**💻 Code:** [⭐ Code](https://github.com/RoboVIP/RoboVIP_VDM)

> The project webpage is at: https://robovip.github.io/RoboVIP/

</details>

<details>
<summary><b>6. RelayLLM: Efficient Reasoning via Collaborative Decoding</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Haolin Liu, Jinyuan Li, Tong Zheng, shrango, ChengsongHuang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05167) • [📄 arXiv](https://arxiv.org/abs/2601.05167) • [📥 PDF](https://arxiv.org/pdf/2601.05167)

**💻 Code:** [⭐ Code](https://github.com/Chengsong-Huang/RelayLLM)

> Large Language Models (LLMs) for complex reasoning is often hindered by high computational costs and latency, while resource-efficient Small Language Models (SLMs) typically lack the necessary reasoning capacity. Existing collaborative approaches,...

</details>

<details>
<summary><b>7. AT^2PO: Agentic Turn-based Policy Optimization via Tree Search</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04767) • [📄 arXiv](https://arxiv.org/abs/2601.04767) • [📥 PDF](https://arxiv.org/pdf/2601.04767)

**💻 Code:** [⭐ Code](https://github.com/zzfoutofspace/ATPO)

> Abstract LLM agents have emerged as powerful systems for tackling multi-turn tasks by interleaving internal reasoning and external tool interactions. Agentic Reinforcement Learning has recently drawn significant research attention as a critical po...

</details>

<details>
<summary><b>8. Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.21815) • [📄 arXiv](https://arxiv.org/abs/2512.21815) • [📥 PDF](https://arxiv.org/pdf/2512.21815)

> Vision-language models (VLMs) achieve remarkable performance but remain vulnerable to adversarial attacks. Entropy, a measure of model uncertainty, is strongly correlated with the reliability of VLM. Prior entropy-based attacks maximize uncertaint...

</details>

<details>
<summary><b>9. VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05175) • [📄 arXiv](https://arxiv.org/abs/2601.05175) • [📥 PDF](https://arxiv.org/pdf/2601.05175)

**💻 Code:** [⭐ Code](https://github.com/IVUL-KAUST/VideoAuto-R1/)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Rethinking Chain-of-Thought Reasoning for Videos (2025) LongVT: Incentivizi...

</details>

<details>
<summary><b>10. VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaoyu Li, Wenbo Hu, Minghao Yin, yanweifuture, sxzheng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05138) • [📄 arXiv](https://arxiv.org/abs/2601.05138) • [📥 PDF](https://arxiv.org/pdf/2601.05138)

> Project Page: https://sixiaozheng.github.io/VerseCrafter_page/

</details>

<details>
<summary><b>11. The Illusion of Specialization: Unveiling the Domain-Invariant "Standing Committee" in Mixture-of-Experts Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03425) • [📄 arXiv](https://arxiv.org/abs/2601.03425) • [📥 PDF](https://arxiv.org/pdf/2601.03425)

> Mixture of Experts models are widely assumed to achieve domain specialization through sparse routing. In this work, we question this assumption by introducing COMMITTEEAUDIT, a post hoc framework that analyzes routing behavior at the level of expe...

</details>

<details>
<summary><b>12. Plenoptic Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05239) • [📄 arXiv](https://arxiv.org/abs/2601.05239) • [📥 PDF](https://arxiv.org/pdf/2601.05239)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API ReCamDriving: LiDAR-Free Camera-Controlled Novel Trajectory Video Generatio...

</details>

<details>
<summary><b>13. Agent-as-a-Judge</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Meng Liu, Qiancheng Xu, Caiqi Zhang, HongruCai, dd101bb

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05111) • [📄 arXiv](https://arxiv.org/abs/2601.05111) • [📥 PDF](https://arxiv.org/pdf/2601.05111)

**💻 Code:** [⭐ Code](https://github.com/ModalityDance/Awesome-Agent-as-a-Judge)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Jenius Agent: Towards Experience-Driven Accuracy Optimization in Real-World...

</details>

<details>
<summary><b>14. CoV: Chain-of-View Prompting for Spatial Reasoning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05172) • [📄 arXiv](https://arxiv.org/abs/2601.05172) • [📥 PDF](https://arxiv.org/pdf/2601.05172)

**💻 Code:** [⭐ Code](https://github.com/ziplab/CoV)

> We propose Chain-of-View (CoV) prompting, a training-free, test-time reasoning framework that transforms a VLM into an active viewpoint reasoner through a coarse-to-fine exploration process.

</details>

<details>
<summary><b>15. DocDancer: Towards Agentic Document-Grounded Information Seeking</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05163) • [📄 arXiv](https://arxiv.org/abs/2601.05163) • [📥 PDF](https://arxiv.org/pdf/2601.05163)

> Document Question Answering (DocQA) focuses on answering questions grounded in given documents, yet existing DocQA agents lack effective tool utilization and largely rely on closed-source models. In this work, we introduce DocDancer, an end-to-end...

</details>

<details>
<summary><b>16. Re-Align: Structured Reasoning-guided Alignment for In-Context Image Generation and Editing</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Tiankai Hang, Yiji Cheng, eternaldolphin, Zhiminli, hrz2000

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05124) • [📄 arXiv](https://arxiv.org/abs/2601.05124) • [📥 PDF](https://arxiv.org/pdf/2601.05124)

**💻 Code:** [⭐ Code](https://github.com/hrz2000/realign)

> This paper introduces Re-Align, a unified framework for in-context image generation and editing that bridges the gap between multimodal understanding and image synthesis. Re-Align employs a structured In-Context Chain-of-Thought (IC-CoT) to explic...

</details>

<details>
<summary><b>17. DiffCoT: Diffusion-styled Chain-of-Thought Reasoning in LLMs</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jing Ma, Yuxuan Gu, Shidong Cao, Ziyang, danielhzlin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03559) • [📄 arXiv](https://arxiv.org/abs/2601.03559) • [📥 PDF](https://arxiv.org/pdf/2601.03559)

> DiffCoT improves multi-step LLM reasoning by applying diffusion-based iterative denoising to correct intermediate Chain-of-Thought steps.

</details>

<details>
<summary><b>18. ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04754) • [📄 arXiv](https://arxiv.org/abs/2601.04754) • [📥 PDF](https://arxiv.org/pdf/2601.04754)

**💻 Code:** [⭐ Code](https://github.com/chiou1203/ProFuse)

> We present ProFuse, an efficient context-aware framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). The pipeline enhances cross-view consistency and intra-mask cohesion within a direct registration setup, adding ...

</details>

<details>
<summary><b>19. Guardians of the Hair: Rescuing Soft Boundaries in Depth, Stereo, and Novel Views</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03362) • [📄 arXiv](https://arxiv.org/abs/2601.03362) • [📥 PDF](https://arxiv.org/pdf/2601.03362)

> Soft boundaries, like thin hairs, are commonly observed in natural and computer-generated imagery, but they remain challenging for 3D vision due to the ambiguous mixing of foreground and background cues. This paper introduces Guardians of the Hair...

</details>

<details>
<summary><b>20. One Sample to Rule Them All: Extreme Data Efficiency in RL Scaling</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xuefeng Li, Weixun Wang, Yanan Wu, Zhen Huang, Yiyuan Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03111) • [📄 arXiv](https://arxiv.org/abs/2601.03111) • [📥 PDF](https://arxiv.org/pdf/2601.03111)

> This work discusses the potential of lifting broader reasoning ability by learning from one high-quality sample. In polymath learning, the quality of samples can be selected through the lens of salient math skills and categories. The model learned...

</details>

<details>
<summary><b>21. Memorization in 3D Shape Generation: An Empirical Study</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.23628) • [📄 arXiv](https://arxiv.org/abs/2512.23628) • [📥 PDF](https://arxiv.org/pdf/2512.23628)

**💻 Code:** [⭐ Code](https://github.com/zlab-princeton/3d-gen-mem)

> Our code is available at https://github.com/zlab-princeton/3d-gen-mem.

</details>

<details>
<summary><b>22. Multi-Scale Local Speculative Decoding for Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05149) • [📄 arXiv](https://arxiv.org/abs/2601.05149) • [📥 PDF](https://arxiv.org/pdf/2601.05149)

> Multi-Scale Local Speculative Decoding (MuLo-SD), a new framework to supercharge Autoregressive (AR) image generation! By combining multi-resolution drafting with spatially informed verification, we achieve substantial speedups of up to 1.7x while...

</details>

<details>
<summary><b>23. PyramidalWan: On Making Pretrained Video Model Pyramidal for Efficient Inference</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04792) • [📄 arXiv](https://arxiv.org/abs/2601.04792) • [📥 PDF](https://arxiv.org/pdf/2601.04792)

> We tackle the challenge of quadratic complexity in video generation with a novel Recurrent Hybrid Attention mechanism. By combining the fidelity of softmax attention for local dependencies with the efficiency of linear attention globally, we enabl...

</details>

<details>
<summary><b>24. AgentDevel: Reframing Self-Evolving LLM Agents as Release Engineering</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Di Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04620) • [📄 arXiv](https://arxiv.org/abs/2601.04620) • [📥 PDF](https://arxiv.org/pdf/2601.04620)

> Recent progress in large language model (LLM) agents has largely focused on embedding self-improvement mechanisms inside the agent or searching over many concurrent variants. While these approaches can raise aggregate scores, they often yield unst...

</details>

<details>
<summary><b>25. Scaling Behavior Cloning Improves Causal Reasoning: An Open Model for Real-Time Video Game Playing</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04575) • [📄 arXiv](https://arxiv.org/abs/2601.04575) • [📥 PDF](https://arxiv.org/pdf/2601.04575)

**💻 Code:** [⭐ Code](https://github.com/elefant-ai/open-p2p)

> We introduce Pixels2Play (P2P), an open-source generalist agent designed for real-time control across diverse 3D video games on consumer-grade GPUs. Built on an efficient, decoder-only transformer architecture that predicts keyboard and mouse acti...

</details>

<details>
<summary><b>26. ReHyAt: Recurrent Hybrid Attention for Video Diffusion Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04342) • [📄 arXiv](https://arxiv.org/abs/2601.04342) • [📥 PDF](https://arxiv.org/pdf/2601.04342)

> 🚀 Introducing PyramidalWan! Our paper presents a novel pipeline to convert pretrained video diffusion models (like Wan2.1-1.3B) into efficient pyramidal ones via low-cost finetuning. Key Innovations: Efficiency via Hierarchy: We restructure the di...

</details>

<details>
<summary><b>27. Beyond Binary Preference: Aligning Diffusion Models to Fine-grained Criteria by Decoupling Attributes</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04300) • [📄 arXiv](https://arxiv.org/abs/2601.04300) • [📥 PDF](https://arxiv.org/pdf/2601.04300)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Direct Diffusion Score Preference Optimization via Stepwise Contrastive Pol...

</details>

<details>
<summary><b>28. Enhancing Object Detection with Privileged Information: A Model-Agnostic Teacher-Student Approach</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Carl James Debono, Matthew Montebello, Gabriel Hili, Dylan Seychell, mbar0075

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02016) • [📄 arXiv](https://arxiv.org/abs/2601.02016) • [📥 PDF](https://arxiv.org/pdf/2601.02016)

> No abstract available.

</details>

<details>
<summary><b>29. VERSE: Visual Embedding Reduction and Space Exploration. Clustering-Guided Insights for Training Data Enhancement in Visually-Rich Document Understanding</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05125) • [📄 arXiv](https://arxiv.org/abs/2601.05125) • [📥 PDF](https://arxiv.org/pdf/2601.05125)

**💻 Code:** [⭐ Code](https://github.com/nachoDRT/VrDU-Doctor)

> We usually train VLMs on visual synthetic data that we (as humans) label as photorealistic. We argue that this is an anthropocentric perspective imposed to a model that might not synthetize visual information as we do. VERSE helps to visualize lat...

</details>

<details>
<summary><b>30. Learning User Preferences Through Interaction for Long-Term Collaboration</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Dilek Hakkani-Tür, Tal August, Priyanka Kargupta, Shuhaib Mehri

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02702) • [📄 arXiv](https://arxiv.org/abs/2601.02702) • [📥 PDF](https://arxiv.org/pdf/2601.02702)

> Current long-term conversation benchmarks focus on recall. But this ignores key skills like recognizing what user information is valuable & leveraging it to improve future interactions. In our work, we present MultiSessionCollab to evaluate agents...

</details>

<details>
<summary><b>31. Safety at One Shot: Patching Fine-Tuned LLMs with A Single Instance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jian Liu, Jian Lou, Kejia Chen, Jiawen Zhang, ttttonyhe

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.01887) • [📄 arXiv](https://arxiv.org/abs/2601.01887) • [📥 PDF](https://arxiv.org/pdf/2601.01887)

> Fine-tuning safety-aligned large language models (LLMs) can substantially compromise their safety. Previous approaches require many safety samples or calibration sets, which not only incur significant computational overhead during realignment but ...

</details>

<details>
<summary><b>32. LEMAS: Large A 150K-Hour Large-scale Extensible Multilingual Audio Suite with Generative Speech Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04233) • [📄 arXiv](https://arxiv.org/abs/2601.04233) • [📥 PDF](https://arxiv.org/pdf/2601.04233)

> LEMAS: A 150K-Hour Large-scale Extensible Multilingual Audio Suite with Generative Speech Models LEMAS is a large-scale extensible multilingual audio suite, providing multilingual speech corpus (LEMAS-Dataset) with word-level timestamps, covering ...

</details>

<details>
<summary><b>33. Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset</b> ⭐ 1</summary>

<br/>

**👥 Authors:** YuanFu Yang, ZhenQi Chen, water-fountain

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2512.24160) • [📄 arXiv](https://arxiv.org/abs/2512.24160) • [📥 PDF](https://arxiv.org/pdf/2512.24160)

**💻 Code:** [⭐ Code](https://github.com/NinaNeon/IMDD-1M-Towards-Open-Vocabulary-Industrial-Defect-)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 33 |
| 📅 Today | [`2026-01-10.json`](data/daily/2026-01-10.json) | 33 |
| 📆 This Week | [`2026-W01.json`](data/weekly/2026-W01.json) | 123 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 164 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-10 | 33 | [View JSON](data/daily/2026-01-10.json) |
| 📄 2026-01-09 | 20 | [View JSON](data/daily/2026-01-09.json) |
| 📄 2026-01-08 | 26 | [View JSON](data/daily/2026-01-08.json) |
| 📄 2026-01-07 | 24 | [View JSON](data/daily/2026-01-07.json) |
| 📄 2026-01-06 | 13 | [View JSON](data/daily/2026-01-06.json) |
| 📄 2026-01-05 | 7 | [View JSON](data/daily/2026-01-05.json) |
| 📄 2026-01-04 | 7 | [View JSON](data/daily/2026-01-04.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W01 | 123 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |
| 📅 2025-W51 | 132 | [View JSON](data/weekly/2025-W51.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 164 | [View JSON](data/monthly/2026-01.json) |
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
