<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-45-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3238+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">45</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">70</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">67</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3238+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 04, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models</b> ⭐ 143</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.26164) • [📄 arXiv](https://arxiv.org/abs/2603.26164) • [📥 PDF](https://arxiv.org/pdf/2603.26164)

**💻 Code:** [⭐ Code](https://github.com/OpenDCAI/DataFlex)

> DataFlex is a data-centric training framework that enhances model performance by either selecting the most influential samples, optimizing their weights, or adjusting their mixing ratios.

</details>

<details>
<summary><b>2. The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook</b> ⭐ 489</summary>

<br/>

**👥 Authors:** Yongbo He, Zhangquan Chen, Xinlei Yu, hhj-ai, YangC777

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02029) • [📄 arXiv](https://arxiv.org/abs/2604.02029) • [📥 PDF](https://arxiv.org/pdf/2604.02029)

**💻 Code:** [⭐ Code](https://github.com/YU-deep/Awesome-Latent-Space)

> The Latent Space: Foundation, Evolution, Mechanism, Ability, and Outlook

</details>

<details>
<summary><b>3. Generative World Renderer</b> ⭐ 145</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02329) • [📄 arXiv](https://arxiv.org/abs/2604.02329) • [📥 PDF](https://arxiv.org/pdf/2604.02329)

**💻 Code:** [⭐ Code](https://github.com/ShandaAI/AlayaRenderer)

> Scaling generative inverse and forward rendering to real-world scenarios is bottlenecked by the limited realism and temporal coherence of existing synthetic datasets. To bridge this persistent domain gap, we introduce a large-scale , dynamic datas...

</details>

<details>
<summary><b>4. SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization</b> ⭐ 70</summary>

<br/>

**👥 Authors:** Qi Gu, Chengcheng Han, Jinyang Wu, Zhiyuan Yao, LZXzju

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02268) • [📄 arXiv](https://arxiv.org/abs/2604.02268) • [📥 PDF](https://arxiv.org/pdf/2604.02268)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/SkillZero)

> We propose SKILL0, the first RL framework that formulates skill internalization as an explicit training objective, moving agents from inference-time skill dependence to fully autonomous zero-shot behavior. We introduce in-context reinforcement lea...

</details>

<details>
<summary><b>5. Steerable Visual Representations</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02327) • [📄 arXiv](https://arxiv.org/abs/2604.02327) • [📥 PDF](https://arxiv.org/pdf/2604.02327)

**💻 Code:** [⭐ Code](https://github.com/manugaurdl/SteerViT)

> Check out our interactive demo: https://huggingface.co/spaces/JonaRuthardt/SteerViT

</details>

<details>
<summary><b>6. EgoSim: Egocentric World Simulator for Embodied Interaction Generation</b> ⭐ 23</summary>

<br/>

**👥 Authors:** Ran Yi, Xihui Liu, Ruiyan Wang, Mingda Jia, wuzhi-hao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01001) • [📄 arXiv](https://arxiv.org/abs/2604.01001) • [📥 PDF](https://arxiv.org/pdf/2604.01001)

**💻 Code:** [⭐ Code](https://github.com/jinkun-hao/EgoSim)

> Project page: https://egosimulator.github.io/

</details>

<details>
<summary><b>7. LatentUM: Unleashing the Potential of Interleaved Cross-Modal Reasoning via a Latent-Space Unified Model</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Pengfei Liu, Hao Zhang, Xiao Yang, Zetong Zhou, orres

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02097) • [📄 arXiv](https://arxiv.org/abs/2604.02097) • [📥 PDF](https://arxiv.org/pdf/2604.02097)

**💻 Code:** [⭐ Code](https://github.com/SJTU-DENG-Lab/LatentUM)

> We’re excited to share LatentUM! It unifies multimodal understanding and generation in one shared latent space, which improves cross-modal reasoning and future-state prediction. Questions and feedback are very welcome.

</details>

<details>
<summary><b>8. NearID: Identity Representation Learning via Near-identity Distractors</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01973) • [📄 arXiv](https://arxiv.org/abs/2604.01973) • [📥 PDF](https://arxiv.org/pdf/2604.01973)

**💻 Code:** [⭐ Code](https://github.com/Gorluxor/NearID)

> Code, inference code, and datasets are released. Model is at https://huggingface.co/Aleksandar/nearid-siglip2 .

</details>

<details>
<summary><b>9. Omni-SimpleMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01007) • [📄 arXiv](https://arxiv.org/abs/2604.01007) • [📥 PDF](https://arxiv.org/pdf/2604.01007)

> AI agents increasingly operate over extended time horizons, yet their ability to retain, organize, and recall multimodal experiences remains a critical bottleneck. Building effective lifelong memory requires navigating a vast design space spanning...

</details>

<details>
<summary><b>10. CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery</b> ⭐ 100</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01658) • [📄 arXiv](https://arxiv.org/abs/2604.01658) • [📥 PDF](https://arxiv.org/pdf/2604.01658)

**💻 Code:** [⭐ Code](https://github.com/Human-Agent-Society/CORAL)

> Large language model (LLM)-based evolution is a promising approach for open-ended discovery, where progress requires sustained search and knowledge accumulation. Existing methods still rely heavily on fixed heuristics and hard-coded exploration ru...

</details>

<details>
<summary><b>11. VOID: Video Object and Interaction Deletion</b> ⭐ 167</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02296) • [📄 arXiv](https://arxiv.org/abs/2604.02296) • [📥 PDF](https://arxiv.org/pdf/2604.02296)

**💻 Code:** [⭐ Code](https://github.com/Netflix/void-model)

> We present VOID, a video object removal framework designed to perform physically-plausible inpainting in these complex scenarios. Check out the demo here: https://huggingface.co/spaces/sam-motamed/VOID .

</details>

<details>
<summary><b>12. Therefore I am. I Think</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01202) • [📄 arXiv](https://arxiv.org/abs/2604.01202) • [📥 PDF](https://arxiv.org/pdf/2604.01202)

> We found evidence that detectable, early-encoded decision signals shape chain-of-thought in reasoning models. The primary benchmark, When2Call, is designed to confuse: ambiguous queries, mismatched tools, missing info. Models need to reason carefu...

</details>

<details>
<summary><b>13. UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving</b> ⭐ 45</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02190) • [📄 arXiv](https://arxiv.org/abs/2604.02190) • [📥 PDF](https://arxiv.org/pdf/2604.02190)

**💻 Code:** [⭐ Code](https://github.com/xiaomi-research/unidrivevla)

> Vision-Language-Action (VLA) models have recently emerged in autonomous driving, with the promise of leveraging rich world knowledge to improve the cognitive capabilities of driving systems. However, adapting such models for driving tasks currentl...

</details>

<details>
<summary><b>14. Investigating Autonomous Agent Contributions in the Wild: Activity Patterns and Code Change over Time</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.00917) • [📄 arXiv](https://arxiv.org/abs/2604.00917) • [📥 PDF](https://arxiv.org/pdf/2604.00917)

> Dataset: https://huggingface.co/datasets/AISE-TUDelft/MOSAIC-agentic-3m

</details>

<details>
<summary><b>15. ASI-Evolve: AI Accelerates AI</b> ⭐ 73</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.29640) • [📄 arXiv](https://arxiv.org/abs/2603.29640) • [📥 PDF](https://arxiv.org/pdf/2603.29640)

**💻 Code:** [⭐ Code](https://github.com/GAIR-NLP/ASI-Evolve)

> Can AI accelerate the development of AI itself? While recent agentic systems have shown strong performance on well-scoped tasks with rapid feedback, it remains unclear whether they can tackle the costly, long-horizon, and weakly supervised researc...

</details>

<details>
<summary><b>16. GPA: Learning GUI Process Automation from Demonstrations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01676) • [📄 arXiv](https://arxiv.org/abs/2604.01676) • [📥 PDF](https://arxiv.org/pdf/2604.01676)

> GUI Process Automation (GPA) - from Salesforce AI Research For product or enterprise use cases, please contact zirui.zhao@salesforce.com or junnan.li@salesforce.com . What is GPA? GPA is a demo-based RPA (Robotic Process Automation) framework for ...

</details>

<details>
<summary><b>17. Tex3D: Objects as Attack Surfaces via Adversarial 3D Textures for Vision-Language-Action Models</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01618) • [📄 arXiv](https://arxiv.org/abs/2604.01618) • [📥 PDF](https://arxiv.org/pdf/2604.01618)

**💻 Code:** [⭐ Code](https://github.com/vla-attack/tex3d)

> This paper presents Tex3D, a novel framework for attacking vision-language-action (VLA) models via adversarial 3D textures applied to manipulated objects. To enable end-to-end optimization in non-differentiable simulation environments, the authors...

</details>

<details>
<summary><b>18. Video Models Reason Early: Exploiting Plan Commitment for Maze Solving</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.30043) • [📄 arXiv](https://arxiv.org/abs/2603.30043) • [📥 PDF](https://arxiv.org/pdf/2603.30043)

> Video models surprisingly can solve mazes, but inconsistently. We understand little about how they reason, making it hard to use such abilities. We investigate the denoising process and find models commit to a plan early, letting us screen far mor...

</details>

<details>
<summary><b>19. AIBench: Evaluating Visual-Logical Consistency in Academic Illustration Generation</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.28068) • [📄 arXiv](https://arxiv.org/abs/2603.28068) • [📥 PDF](https://arxiv.org/pdf/2603.28068)

**💻 Code:** [⭐ Code](https://github.com/ali-vilab/AIBench)

> Project Page: https://deep-kaixun.github.io/aibench-page/

</details>

<details>
<summary><b>20. VideoZeroBench: Probing the Limits of Video MLLMs with Spatio-Temporal Evidence Verification</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01569) • [📄 arXiv](https://arxiv.org/abs/2604.01569) • [📥 PDF](https://arxiv.org/pdf/2604.01569)

**💻 Code:** [⭐ Code](https://github.com/marinero4972/VideoZeroBench)

> VideoZeroBench is a hierarchical benchmark designed for extremely challenging long-video question answering that rigorously verifies spatio-temporal evidence.

</details>

<details>
<summary><b>21. AutoMIA: Improved Baselines for Membership Inference Attack via Agentic Self-Exploration</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Xinchao Wang, Qi Li, Weiqi Huang, Ruhao Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01014) • [📄 arXiv](https://arxiv.org/abs/2604.01014) • [📥 PDF](https://arxiv.org/pdf/2604.01014)

**💻 Code:** [⭐ Code](https://github.com/amiya-special/AutoMIA)

> Github Repo: https://github.com/amiya-special/AutoMIA

</details>

<details>
<summary><b>22. LinguDistill: Recovering Linguistic Ability in Vision- Language Models via Selective Cross-Modal Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.00829) • [📄 arXiv](https://arxiv.org/abs/2604.00829) • [📥 PDF](https://arxiv.org/pdf/2604.00829)

> Adapting pretrained language models (LMs) into vision-language models (VLMs) can degrade their native linguistic capability due to representation shift and cross-modal interference introduced during multimodal adaptation. Such loss is difficult to...

</details>

<details>
<summary><b>23. Forecasting Supply Chain Disruptions with Foresight Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01298) • [📄 arXiv](https://arxiv.org/abs/2604.01298) • [📥 PDF](https://arxiv.org/pdf/2604.01298)

> We train an LLM to forecast supply chain disruptions from news using Foresight Learning, an RL-based framework that supervises probabilistic predictions with realized outcomes. Our fine-tuned model outperforms GPT-5 across all metrics on a held-ou...

</details>

<details>
<summary><b>24. MDPBench: A Benchmark for Multilingual Document Parsing in Real-World Scenarios</b> ⭐ 811</summary>

<br/>

**👥 Authors:** Shuo Zhang, Ziyang Zhang, Qiang Liu, Zhibo Lin, Zhang Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.28130) • [📄 arXiv](https://arxiv.org/abs/2603.28130) • [📥 PDF](https://arxiv.org/pdf/2603.28130)

**💻 Code:** [⭐ Code](https://github.com/Yuliang-Liu/MultimodalOCR)

> Cool new OCR benchmark

</details>

<details>
<summary><b>25. Omni123: Exploring 3D Native Foundation Models with Limited 3D Data by Unifying Text to 2D and 3D Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yihao Zhi, Yiming Hao, Chuanyu Pan, Cheng Cao, Chongjie Ye

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02289) • [📄 arXiv](https://arxiv.org/abs/2604.02289) • [📥 PDF](https://arxiv.org/pdf/2604.02289)

> No abstract available.

</details>

<details>
<summary><b>26. DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01666) • [📄 arXiv](https://arxiv.org/abs/2604.01666) • [📥 PDF](https://arxiv.org/pdf/2604.01666)

> Existing video diffusion models still struggle to synthesize realistic videos involving highly dynamic motions or requiring fine-grained motion controllability. A central limitation lies in the scarcity of such examples in commonly used training d...

</details>

<details>
<summary><b>27. T5Gemma-TTS Technical Report</b> ⭐ 284</summary>

<br/>

**👥 Authors:** KiyoshiKurihara, Aratako

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01760) • [📄 arXiv](https://arxiv.org/abs/2604.01760) • [📥 PDF](https://arxiv.org/pdf/2604.01760)

**💻 Code:** [⭐ Code](https://github.com/Aratako/T5Gemma-TTS)

> GitHub Project: https://github.com/Aratako/T5Gemma-TTS Model: https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b Demo: https://huggingface.co/spaces/Aratako/T5Gemma-TTS-Demo

</details>

<details>
<summary><b>28. Apriel-Reasoner: RL Post-Training for General-Purpose and Efficient Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02007) • [📄 arXiv](https://arxiv.org/abs/2604.02007) • [📥 PDF](https://arxiv.org/pdf/2604.02007)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for...

</details>

<details>
<summary><b>29. FlowSlider: Training-Free Continuous Image Editing via Fidelity-Steering Decomposition</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kazuhiko Sumi, Guoqing Hao, Taichi Endo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02088) • [📄 arXiv](https://arxiv.org/abs/2604.02088) • [📥 PDF](https://arxiv.org/pdf/2604.02088)

> Training-free continuous image editing

</details>

<details>
<summary><b>30. Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhenzhi Tan, Zikai Xie, Alexandre Max Maraval, Rasul Tutunov, Zhongwei Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01328) • [📄 arXiv](https://arxiv.org/abs/2604.01328) • [📥 PDF](https://arxiv.org/pdf/2604.01328)

> Scaling generation isn’t enough for Chemistry → you need cost-aware search. We show how Bayesian optimisation can be a tool used in the chemistry toolbox.

</details>

<details>
<summary><b>31. Gated Condition Injection without Multimodal Attention: Towards Controllable Linear-Attention Transformers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.27666) • [📄 arXiv](https://arxiv.org/abs/2603.27666) • [📥 PDF](https://arxiv.org/pdf/2603.27666)

**💻 Code:** [⭐ Code](https://github.com/Carol-lyh/GateControl.git)

> Recent advances in diffusion-based controllable visual generation have led to remarkable improvements in image quality. However, these powerful models are typically deployed on cloud servers due to their large computational demands, raising seriou...

</details>

<details>
<summary><b>32. Working Notes on Late Interaction Dynamics: Analyzing Targeted Behaviors of Late Interaction Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.26259) • [📄 arXiv](https://arxiv.org/abs/2603.26259) • [📥 PDF](https://arxiv.org/pdf/2603.26259)

> We link the poster as a PNG file 🤗 We were very glad to present it at the 1st Late Interaction Workshop at ECIR 2026 !

</details>

<details>
<summary><b>33. ActionParty: Multi-Subject Action Binding in Generative Video Games</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.02330) • [📄 arXiv](https://arxiv.org/abs/2604.02330) • [📥 PDF](https://arxiv.org/pdf/2604.02330)

**💻 Code:** [⭐ Code](https://github.com/action-party/action-party)

> Project page: https://action-party.github.io/

</details>

<details>
<summary><b>34. Woosh: A Sound Effects Foundation Model</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01929) • [📄 arXiv](https://arxiv.org/abs/2604.01929) • [📥 PDF](https://arxiv.org/pdf/2604.01929)

**💻 Code:** [⭐ Code](https://github.com/SonyResearch/Woosh)

> Github: https://github.com/SonyResearch/Woosh

</details>

<details>
<summary><b>35. Executing as You Generate: Hiding Execution Latency in LLM Code Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.00491) • [📄 arXiv](https://arxiv.org/abs/2604.00491) • [📥 PDF](https://arxiv.org/pdf/2604.00491)

> Current LLM-based coding agents follow a serial execution paradigm: the model first generates the complete code, then invokes an interpreter to execute it. This sequential workflow leaves the executor idle during generation and the generator idle ...

</details>

<details>
<summary><b>36. UniRecGen: Unifying Multi-View 3D Reconstruction and Generation</b> ⭐ 25</summary>

<br/>

**👥 Authors:** Chenyu Hu, Cheng Lin, Jiahao Chen, Zhisheng Huang, Yuheng02

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01479) • [📄 arXiv](https://arxiv.org/abs/2604.01479) • [📥 PDF](https://arxiv.org/pdf/2604.01479)

**💻 Code:** [⭐ Code](https://github.com/zsh523/UniRecGen)

> Code: https://github.com/zsh523/UniRecGen

</details>

<details>
<summary><b>37. Memory-Augmented Vision-Language Agents for Persistent and Semantically Consistent Object Captioning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.24257) • [📄 arXiv](https://arxiv.org/abs/2603.24257) • [📥 PDF](https://arxiv.org/pdf/2603.24257)

> Why do VLMs call the same object a "sofa," a "bed," and an "armchair" during a single navigation task? 🛋️ We present EPOS-VLM, which uses a structured episodic object memory to ensure persistent semantic consistency in 3D environments. Our agent d...

</details>

<details>
<summary><b>38. Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.26233) • [📄 arXiv](https://arxiv.org/abs/2603.26233) • [📥 PDF](https://arxiv.org/pdf/2603.26233)

**💻 Code:** [⭐ Code](https://github.com/nedwards99/ask-or-assume)

> We investigate whether LLM agents can independently decide when to ask clarifying questions on underspecified coding tasks. Our uncertainty-aware multi-agent scaffold (UA-Multi) achieves 69.40% on an underspecified variant of SWE-bench Verified — ...

</details>

<details>
<summary><b>39. Automatic Image-Level Morphological Trait Annotation for Organismal Images</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01619) • [📄 arXiv](https://arxiv.org/abs/2604.01619) • [📥 PDF](https://arxiv.org/pdf/2604.01619)

**💻 Code:** [⭐ Code](https://github.com/OSU-NLP-Group/sae-trait-annotation)

> We develop a trait annotation pipeline that leverages sparse autoencoders to generate interpretable morphological trait descriptions from ecological images.

</details>

<details>
<summary><b>40. Signals: Trajectory Sampling and Triage for Agentic Interactions</b> ⭐ 6.14k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.00356) • [📄 arXiv](https://arxiv.org/abs/2604.00356) • [📥 PDF](https://arxiv.org/pdf/2604.00356)

**💻 Code:** [⭐ Code](https://github.com/katanemo/plano)

> Agentic applications based on large language models increasingly rely on multi-step interaction loops involving planning, action execution, and environment feedback. While such systems are now deployed at scale, improving them post-deployment rema...

</details>

<details>
<summary><b>41. Friends and Grandmothers in Silico: Localizing Entity Cells in Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mor Geva, Michael Karasik, Dan Barzilay, tux

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01404) • [📄 arXiv](https://arxiv.org/abs/2604.01404) • [📥 PDF](https://arxiv.org/pdf/2604.01404)

**💻 Code:** [⭐ Code](https://github.com/1tux/in-silico)

> .

</details>

<details>
<summary><b>42. Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.01152) • [📄 arXiv](https://arxiv.org/abs/2604.01152) • [📥 PDF](https://arxiv.org/pdf/2604.01152)

**💻 Code:** [⭐ Code](https://github.com/achelousace/brainstacks)

> "Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning" I built an architecture that adds unlimited domain expertise to any LLM - one domain at a time - with near-zero forgetting. Null-space project...

</details>

<details>
<summary><b>43. An Empirical Recipe for Universal Phone Recognition</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.29042) • [📄 arXiv](https://arxiv.org/abs/2603.29042) • [📥 PDF](https://arxiv.org/pdf/2603.29042)

**💻 Code:** [⭐ Code](https://github.com/changelinglab/PhoneticXeus)

> Phone recognition (PR) is a key enabler of multilingual and low-resource speech processing tasks, yet robust performance remains elusive. Highly performant English-focused models do not generalize across languages, while multilingual models underu...

</details>

<details>
<summary><b>44. MultiGen: Level-Design for Editable Multiplayer Worlds in Diffusion Game Engines</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Neal Wadhwa, Gordon Wetzstein, Amir Hertz, David Junhao Zhang, Ryan Po

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.06679) • [📄 arXiv](https://arxiv.org/abs/2603.06679) • [📥 PDF](https://arxiv.org/pdf/2603.06679)

> No abstract available.

</details>

<details>
<summary><b>45. LOME: Learning Human-Object Manipulation with Action-Conditioned Egocentric World Model</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.27449) • [📄 arXiv](https://arxiv.org/abs/2603.27449) • [📥 PDF](https://arxiv.org/pdf/2603.27449)

**💻 Code:** [⭐ Code](https://github.com/Zerg-Overmind/LOME)

> LOME: Learning Human-Object Manipulation with Action-Conditioned Egocentric World Model. LOME generates videos of contact-rich and fine-grained human-object manipulations without 3D/4D reconstruction or affordance prediction.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 45 |
| 📅 Today | [`2026-04-04.json`](data/daily/2026-04-04.json) | 45 |
| 📆 This Week | [`2026-W13.json`](data/weekly/2026-W13.json) | 70 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 67 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-04 | 45 | [View JSON](data/daily/2026-04-04.json) |
| 📄 2026-04-03 | 7 | [View JSON](data/daily/2026-04-03.json) |
| 📄 2026-04-02 | 5 | [View JSON](data/daily/2026-04-02.json) |
| 📄 2026-04-01 | 10 | [View JSON](data/daily/2026-04-01.json) |
| 📄 2026-03-31 | 1 | [View JSON](data/daily/2026-03-31.json) |
| 📄 2026-03-30 | 2 | [View JSON](data/daily/2026-03-30.json) |
| 📄 2026-03-29 | 29 | [View JSON](data/daily/2026-03-29.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W13 | 70 | [View JSON](data/weekly/2026-W13.json) |
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |
| 📅 2026-W11 | 133 | [View JSON](data/weekly/2026-W11.json) |
| 📅 2026-W10 | 119 | [View JSON](data/weekly/2026-W10.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 67 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |
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
