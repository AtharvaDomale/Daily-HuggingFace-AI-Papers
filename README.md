<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-45-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6608+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">89</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">480</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6608+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** September 30, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. What Makes World Action Models Generalize? An Empirical Study of Test-Time Future Modeling</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34981) • [📄 arXiv](https://arxiv.org/abs/2609.34981) • [📥 PDF](https://arxiv.org/pdf/2609.34981)

**💻 Code:** [⭐ Code](https://github.com/LeapLabTHU/Simple-WAM) • [⭐ Code](https://github.com/huggingface)

> Links 📄 paper: https://arxiv.org/abs/2609.34981 🏠 project page: https://zrporz.github.io/Simple-WAM-Web/ 💻 code: https://github.com/LeapLabTHU/Simple-WAM 🤗 model: https://huggingface.co/rpzhou/Simple-WAM

</details>

<details>
<summary><b>2. Raven: The Harness of Harnesses for Composable Agentic Intelligence</b> ⭐ 4.83k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33439) • [📄 arXiv](https://arxiv.org/abs/2609.33439) • [📥 PDF](https://arxiv.org/pdf/2609.33439)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/EverMind-AI/Raven)

> As large language models advance, AI agents are moving beyond isolated, domain-specific tasks toward long-horizon, cross-domain workflows. This transition exposes two challenges: increasing harness complexity makes manual design difficult to scale...

</details>

<details>
<summary><b>3. MaLiang-Harness: A Programmable Path to Image and Video Generation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34309) • [📄 arXiv](https://arxiv.org/abs/2609.34309) • [📥 PDF](https://arxiv.org/pdf/2609.34309)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/gulucaptain/MaLiang-Harness)

> Executable programs offer explicit control over how images and videos are constructed, but generating runnable code is only the beginning of visual creation. A program can execute correctly while violating the requested composition, appearance, or...

</details>

<details>
<summary><b>4. Beyond the Timeline: Augmenting Long-Video Memory with Grounded Entity Biographies</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38155) • [📄 arXiv](https://arxiv.org/abs/2609.38155) • [📥 PDF](https://arxiv.org/pdf/2609.38155)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/rhfeiyang/GEB)

> We introduce Grounded Entity Biographies (GEB), which links observations of the same physical entity across long videos into retrievable biographies, improving long-horizon video understanding.

</details>

<details>
<summary><b>5. Think Before You Score: Thinking Reward Model for Visual Generation</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Tengfei Liu, Dianyi Wang, Yang Shi, Zhenchen Tang, Xuehai Bai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37372) • [📄 arXiv](https://arxiv.org/abs/2609.37372) • [📥 PDF](https://arxiv.org/pdf/2609.37372)

**💻 Code:** [⭐ Code](https://github.com/bxhsort/Thinking_Reward_Model) • [⭐ Code](https://github.com/huggingface)

> Visual reward models are essential for evaluating and improving visual generation models, yet existing approaches typically map task conditions and candidate outputs directly to scalar rewards, leaving implicit what should be evaluated for each in...

</details>

<details>
<summary><b>6. SAKI: Maximal-Coupling-Routed Teacher Supervision for On-Policy Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36601) • [📄 arXiv](https://arxiv.org/abs/2609.36601) • [📥 PDF](https://arxiv.org/pdf/2609.36601)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> On-policy distillation (OPD) reduces train-test state mismatch by training a student on its own generated trajectories, but weak students may visit teacher-misaligned prefixes where supervision is less representative. We introduce SAKI (Supervisio...

</details>

<details>
<summary><b>7. Learning Beyond What You Sample: Off-Policy-Aware Cross-Model Trajectory Exchange for RLVR</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37868) • [📄 arXiv](https://arxiv.org/abs/2609.37868) • [📥 PDF](https://arxiv.org/pdf/2609.37868)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We explore a simple question: can heterogeneous reasoning models learn from successes that their peers discover but they fail to sample? GRAFT exchanges complementary peer trajectories during RLVR while explicitly controlling off-policy mismatch. ...

</details>

<details>
<summary><b>8. Anisotropic Representations Improve Planning in JEPA World Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37441) • [📄 arXiv](https://arxiv.org/abs/2609.37441) • [📥 PDF](https://arxiv.org/pdf/2609.37441)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>9. OmniTaskonomy: When Does Visual Generation Improve Visual Understanding?</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38079) • [📄 arXiv](https://arxiv.org/abs/2609.38079) • [📥 PDF](https://arxiv.org/pdf/2609.38079)

**💻 Code:** [⭐ Code](https://github.com/para-lost/OmniTaskonomy) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>10. LEGO-Anything: Coding Agents for 3D Scene Reconstruction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36380) • [📄 arXiv](https://arxiv.org/abs/2609.36380) • [📥 PDF](https://arxiv.org/pdf/2609.36380)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> LEGO-Anything uses coding agents to reconstruct a single image as an editable, executable 3D scene through iterative Blender programming. The work introduces LEGO-Bench, comprising 208 images from 104 indoor and outdoor scenes, to evaluate artifac...

</details>

<details>
<summary><b>11. Marathoner: Ultra-Long-Horizon Autonomous Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34378) • [📄 arXiv](https://arxiv.org/abs/2609.34378) • [📥 PDF](https://arxiv.org/pdf/2609.34378)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Humans naturally possess the ability to work persistently toward long-term goals. Given a challenging task, humans can continuously work for months or even years to accomplish a specific objective. Following this spirit, strong proprietary models ...

</details>

<details>
<summary><b>12. WorldAttention: An Efficient Attention Architecture for Interactive Video World Models</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34606) • [📄 arXiv](https://arxiv.org/abs/2609.34606) • [📥 PDF](https://arxiv.org/pdf/2609.34606)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/alibaba-damo-academy/WorldAttention)

> 🌍 We propose WorldAttention, an efficient attention architecture that lets interactive video world models draw on long-range history while generating at 22 FPS on a single NVIDIA H100. ⚡ Its Hybrid Sparse Attention and Hierarchical KV Cache delive...

</details>

<details>
<summary><b>13. WorldLine: Action-Driven Visual Simulation for Robotic Manipulation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38059) • [📄 arXiv](https://arxiv.org/abs/2609.38059) • [📥 PDF](https://arxiv.org/pdf/2609.38059)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Zhengsh123/WorldLine)

> A visual simulator that predicts how robot actions change the scene. WorldLine separates learning robot–object dynamics from learning how each robot’s controls should steer them.

</details>

<details>
<summary><b>14. EasyPPO: Stabilizing the Critic Is Key</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenhao Chai, Dacheng Li, Huanzhi Mao, Qiuyang Mang, Xuanyi Zhou

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36802) • [📄 arXiv](https://arxiv.org/abs/2609.36802) • [📥 PDF](https://arxiv.org/pdf/2609.36802)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 来看

</details>

<details>
<summary><b>15. LongLive-Plug: Once-for-All Distillation for Video Generation</b> ⭐ 2.64k</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38154) • [📄 arXiv](https://arxiv.org/abs/2609.38154) • [📥 PDF](https://arxiv.org/pdf/2609.38154)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/NVlabs/LongLive)

> We introduce LongLive-Plug, a once-for-all distillation framework that learns reusable capabilities as LoRAs for training-free, plug-and-play deployment to compatible downstream video models. We validate deployment on 54 downstream models across M...

</details>

<details>
<summary><b>16. CrossBFM: Distilling a Shared Latent Behavior Space Across Humanoid Embodiments</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Siwei Ju, Cuc T. Trinh, Nico Bohlinger, Tuan Dat Phuong, Tan-Dzung Do

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38087) • [📄 arXiv](https://arxiv.org/abs/2609.38087) • [📥 PDF](https://arxiv.org/pdf/2609.38087)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project Page: https://dotandung.github.io/crossbfm/

</details>

<details>
<summary><b>17. HiRAE: Hierarchical Representation Autoencoding with Residual Budgets</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuanxing Zhang, Yihang Lou, Yang Shi, Yan Bai, Xuanyu Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37775) • [📄 arXiv](https://arxiv.org/abs/2609.37775) • [📥 PDF](https://arxiv.org/pdf/2609.37775)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Pretrained visual representations support image generation, but may not fully preserve the fine-grained details needed for faithful reconstruction. Meanwhile, intermediate encoder layers contain complementary visual details, but learning to fuse t...

</details>

<details>
<summary><b>18. HybridCUA: Learning to Orchestrate GUI and CLI for Computer-Use Agents</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Fei Tang, Niu Lian, Zhengxi Lu, Junbo Niu, Tongbo Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38008) • [📄 arXiv](https://arxiv.org/abs/2609.38008) • [📥 PDF](https://arxiv.org/pdf/2609.38008)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/HybridCUA)

> We develop a data construction pipeline that produces interleaved GUI and CLI trajectories. This pipeline results in HybridCUA-8K, containing 5K hybrid trajectories and 3K verified RLVR tasks. Building on these data, we propose a training framewor...

</details>

<details>
<summary><b>19. EVO-WAM: Evolving World Action Models through Video-Action Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38057) • [📄 arXiv](https://arxiv.org/abs/2609.38057) • [📥 PDF](https://arxiv.org/pdf/2609.38057)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Clausy9/EVO-WAM)

> Really cool idea! 🤖 EVO-WAM shows that a World Action Model can actually learn from its own imagination! Instead of collecting more expert demonstrations, it generates video-action rollouts, uses a VLM to check whether the task is completed, and a...

</details>

<details>
<summary><b>20. When Does Dense Retrieval Need Asymmetric Geometry? A Bias-Variance Theory of Shared and Dual Projections</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32488) • [📄 arXiv](https://arxiv.org/abs/2609.32488) • [📥 PDF](https://arxiv.org/pdf/2609.32488)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> When should dense retrieval use separate query and document projections instead of a shared one? We study this choice through a bias–variance lens, derive a boundary for when the added flexibility is worthwhile, and propose CARS to help select bet...

</details>

<details>
<summary><b>21. ANTMAN: Adaptive Need Tracking for Multi-Agent Navigation in Large Information Spaces</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33326) • [📄 arXiv](https://arxiv.org/abs/2609.33326) • [📥 PDF](https://arxiv.org/pdf/2609.33326)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce ANTMAN, an adaptive coordination framework for information-seeking agents operating over large-scale information spaces. Instead of relying on static decomposition, ANTMAN tracks evolving unresolved information needs and scales coordi...

</details>

<details>
<summary><b>22. VideoLoop: Looped Working Memory Against Semantic Thrashing in Long-Form Video Agents</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Jiebo Luo, Zhengyuan Yang, Jingyang Lin, Jianming Xu, Jinfa Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38119) • [📄 arXiv](https://arxiv.org/abs/2609.38119) • [📥 PDF](https://arxiv.org/pdf/2609.38119)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/philipxjm/videoloop)

> VideoLoop addresses semantic thrashing in long-form video agents with two coupled loops: an outer loop that reasons over video and an inner loop that retrieves past artifacts and rewrites a bounded working memory. It improves four LVLM backbones b...

</details>

<details>
<summary><b>23. ROSS: Relearning from Self-Generated Rollouts through Selective Supervision</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bin Liang, Jiayan Fu, Fei Zhao, Huayu Deng, Zhiwei Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35954) • [📄 arXiv](https://arxiv.org/abs/2609.35954) • [📥 PDF](https://arxiv.org/pdf/2609.35954)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large language model post-training generates self-generated rollouts through reinforcement learning and on-policy distillation, yet this experience is often treated as stale once the policy advances. Historical rollouts can remain compatible with ...

</details>

<details>
<summary><b>24. LIFT: Layout-In-Future Video Generation under Large Viewpoint Change via On-Policy Self-Distillation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38146) • [📄 arXiv](https://arxiv.org/abs/2609.38146) • [📥 PDF](https://arxiv.org/pdf/2609.38146)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jsxzs/LIFT)

> TL;DR Joint Camera and Future-Layout Control: LIFT is a unified video generation framework that enables users to control both camera motion and the semantic-spatial composition of newly revealed regions using only a last-frame layout. Dual-mode OP...

</details>

<details>
<summary><b>25. TGRL: Temperature-Grouped Reinforcement Learning for Efficient Exploration in LLMs</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.33589) • [📄 arXiv](https://arxiv.org/abs/2609.33589) • [📥 PDF](https://arxiv.org/pdf/2609.33589)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/1229095296/TGRL)

> Accepted by NeurIPS2026. Efficient exploration often remains a central bottleneck in reinforcement learning with verifiable rewards (RLVR). Although temperature control and test-time scaling strategies can increase rollout diversity of large langu...

</details>

<details>
<summary><b>26. TabFM: A Zero-Shot Foundation Model for Tabular Data</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37959) • [📄 arXiv](https://arxiv.org/abs/2609.37959) • [📥 PDF](https://arxiv.org/pdf/2609.37959)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TabFM, a tabular foundation model from Google Research

</details>

<details>
<summary><b>27. Real2Gym: Building Gyms from Videos, Bringing Skills to Robots</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37089) • [📄 arXiv](https://arxiv.org/abs/2609.37089) • [📥 PDF](https://arxiv.org/pdf/2609.37089)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/real2gym/Real2Gym)

> Real-world videos provide rich demonstrations of manipulation, but turning them into reusable robot skills requires visually aligned environments, executable physical interactions, and mechanisms for learning from experience. We introduce Real2Gym...

</details>

<details>
<summary><b>28. In-Context Learning for Robots: Methods and Applications</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36012) • [📄 arXiv](https://arxiv.org/abs/2609.36012) • [📥 PDF](https://arxiv.org/pdf/2609.36012)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/JethroJames/awesome-robots-icl)

> Co-author note: I contributed to this survey. A robot can finish a task and still miss what the demonstration taught. This survey follows context all the way to execution: through action distributions, motion references, predicted futures, and ski...

</details>

<details>
<summary><b>29. Where the Model Changes Its Mind: Hindsight-Divergence Localization for Efficient Reinforcement Learning with Verifiable Rewards</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36864) • [📄 arXiv](https://arxiv.org/abs/2609.36864) • [📥 PDF](https://arxiv.org/pdf/2609.36864)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Where the Model Changes Its Mind: Hindsight-Divergence Localization for Efficient Reinforcement Learning with Verifiable Rewards

</details>

<details>
<summary><b>30. TabFM-Auto: Self-Evolving Pipelines for Tabular Foundation Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37989) • [📄 arXiv](https://arxiv.org/abs/2609.37989) • [📥 PDF](https://arxiv.org/pdf/2609.37989)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> TabFM-Auto: an LLM agent search feature engineering around a frozen TabFM — the Tabular Foundation Model by Google Research — to achieve over 2000 Elo on TabArena.

</details>

<details>
<summary><b>31. Beyond Selection: Token Parameterization for Extreme Visual Token Compression</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Cheng Zhuo, Zheyu Yan, Yu Li, zrrraa

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35232) • [📄 arXiv](https://arxiv.org/abs/2609.35232) • [📥 PDF](https://arxiv.org/pdf/2609.35232)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zrrraa/Braco)

> TL;DR: Don’t just decide which visual tokens to keep—change how they are represented. We revisit compression through a token parameterization lens, separating (i) basis transformation and structured truncation (retained subspace/compressibility) f...

</details>

<details>
<summary><b>32. Adversarial Training for Pixel Diffusion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.38170) • [📄 arXiv](https://arxiv.org/abs/2609.38170) • [📥 PDF](https://arxiv.org/pdf/2609.38170)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Pixel diffusion models can generate semantically strong images, yet often miss fine-scale natural image statistics. We find that adversarial post-training consistently restores this missing high-frequency detail, improving fidelity, coverage, prom...

</details>

<details>
<summary><b>33. Language Models Are "Insecure" Reporters</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36139) • [📄 arXiv](https://arxiv.org/abs/2609.36139) • [📥 PDF](https://arxiv.org/pdf/2609.36139)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>34. AutoDataBench: Can Agents Write the Data That Feeds the Self-Improvement Loop?</b> ⭐ 18</summary>

<br/>

**👥 Authors:** Yibo Wang, Huanjin Yao, Zeyu Qin, Haoyu Wang, Haotian Luo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35025) • [📄 arXiv](https://arxiv.org/abs/2609.35025) • [📥 PDF](https://arxiv.org/pdf/2609.35025)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/StarDewXXX/AutoDataBench)

> Recent gains in language model capability have come more from data than from architecture. Frontier labs and data companies produce verifiable agentic tasks, which supervised finetuning and reinforcement learning then turn into capability. This pr...

</details>

<details>
<summary><b>35. PanoVLN: Towards Effective Panoramic Vision-and-Language Navigation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34759) • [📄 arXiv](https://arxiv.org/abs/2609.34759) • [📥 PDF](https://arxiv.org/pdf/2609.34759)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://wangzhen-w.github.io/PanoVLN/

</details>

<details>
<summary><b>36. Principled Thoughts for Latent Recursive LLM Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36159) • [📄 arXiv](https://arxiv.org/abs/2609.36159) • [📥 PDF](https://arxiv.org/pdf/2609.36159)

**💻 Code:** [⭐ Code](https://github.com/FARD-Lab/REST) • [⭐ Code](https://github.com/huggingface)

> Large language models can reason in continuous space instead of decoded text, by recurring on their own hidden states or by passing those states between agents, while training supervises only the Cross-Entropy (CE) of the final decoded answer and ...

</details>

<details>
<summary><b>37. Act First, Reason Later: Accelerating On-Policy Distillation for Multi-Turn Agents via Reference-Conditioned Inverse Dynamics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36608) • [📄 arXiv](https://arxiv.org/abs/2609.36608) • [📥 PDF](https://arxiv.org/pdf/2609.36608)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> ActFirst-OPD accelerates multi-turn agent on-policy distillation by combining reference-conditioned inverse dynamics for fast actions with asynchronous full-response generation.

</details>

<details>
<summary><b>38. FurE: Efficient Instance-Specific 3D Fur Reconstruction without Animal-Fur Datasets</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Alan Yuille, Soumava Paul, Prakhar Kaushik, Srinjay Sarkar

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35770) • [📄 arXiv](https://arxiv.org/abs/2609.35770) • [📥 PDF](https://arxiv.org/pdf/2609.35770)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/toshi2k2/fure)

> Efficient SOTA for 3D animal fur reconstruction.

</details>

<details>
<summary><b>39. EngiWorld: What Can Frontier Agents Deliver in Professional Engineering Environments?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.37686) • [📄 arXiv](https://arxiv.org/abs/2609.37686) • [📥 PDF](https://arxiv.org/pdf/2609.37686)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Hongcheng-Gao/EngiWorld)

> EngiWorld is the first benchmark covering the complete engineering design loop, with 1,301 expert-curated tasks across 6 domains (CAD, CAE, CAM, BIM, EDA, 3D visualization) and 26 professional software platforms. Its artifact-centric evaluation pr...

</details>

<details>
<summary><b>40. One Proposal for Every Margin: Zero-Shot Amortized Sequential Importance Sampling for Binary Matrices</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.35514) • [📄 arXiv](https://arxiv.org/abs/2609.35514) • [📥 PDF](https://arxiv.org/pdf/2609.35514)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi everyone, first author here! 👋 Excited to share MarginFlow: one learned proposal for thousands of counting and sampling problems. 🚀 We tackle a classic challenge: counting and sampling binary matrices with prescribed row and column sums—a found...

</details>

<details>
<summary><b>41. Preference-Guided Adaptation for Open-Vocabulary Semantic Segmentation via Prompt Disagreement</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34528) • [📄 arXiv](https://arxiv.org/abs/2609.34528) • [📥 PDF](https://arxiv.org/pdf/2609.34528)

**💻 Code:** [⭐ Code](https://github.com/blue-531/pref-ovss) • [⭐ Code](https://github.com/huggingface)

> Open-vocabulary semantic segmentation (OVSS) enables pixel-level prediction over arbitrary text-specified vocabularies and has shown strong generalization on common benchmarks. However, OVSS performance often degrades in specialized domains such a...

</details>

<details>
<summary><b>42. LongCat-DeepResearch Technical Report</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Haolin Ren, Wanli Wu, Yue Xu, He Zhu, Meituan LongCat Team

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.36071) • [📄 arXiv](https://arxiv.org/abs/2609.36071) • [📥 PDF](https://arxiv.org/pdf/2609.36071)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>43. PlaylistEval: Can Video-Language Judges Be Trusted at Day Scale and Beyond?</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hwanjun Song, Shayekh Bin Islam

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.34314) • [📄 arXiv](https://arxiv.org/abs/2609.34314) • [📥 PDF](https://arxiv.org/pdf/2609.34314)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We benchmark 17 video-language judges from eight model families on PlaylistEval, covering 630 preference pairs across seven domains, with roughly 100 hours of video per domain. Several findings stood out: The strongest judge reaches only 75.4% acc...

</details>

<details>
<summary><b>44. CaptchaArena: A Large-Scale, Fine-Grained Dataset for Training Computer-Use Agents on Interactive CAPTCHAs</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.31957) • [📄 arXiv](https://arxiv.org/abs/2609.31957) • [📥 PDF](https://arxiv.org/pdf/2609.31957)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/X0X0X00/CaptchaArena)

> We introduce CaptchaArena, a large-scale, fine-grained dataset for interactive CAPTCHA solving, with 50K puzzles across 20 types and 5 interaction modes, including 46K screenshot-action trajectories with step-by-step reasoning. Using CaptchaArena,...

</details>

<details>
<summary><b>45. PrismQuant: Optimal Null-Space Rotations for Grouped Quantizers</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2609.32429) • [📄 arXiv](https://arxiv.org/abs/2609.32429) • [📥 PDF](https://arxiv.org/pdf/2609.32429)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ForeverBlue816/PrismQuant)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 45 |
| 📅 Today | [`2026-09-30.json`](data/daily/2026-09-30.json) | 45 |
| 📆 This Week | [`2026-W39.json`](data/weekly/2026-W39.json) | 89 |
| 🗓️ This Month | [`2026-09.json`](data/monthly/2026-09.json) | 480 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-09-30 | 45 | [View JSON](data/daily/2026-09-30.json) |
| 📄 2026-09-29 | 37 | [View JSON](data/daily/2026-09-29.json) |
| 📄 2026-09-28 | 7 | [View JSON](data/daily/2026-09-28.json) |
| 📄 2026-09-27 | 22 | [View JSON](data/daily/2026-09-27.json) |
| 📄 2026-09-26 | 22 | [View JSON](data/daily/2026-09-26.json) |
| 📄 2026-09-25 | 13 | [View JSON](data/daily/2026-09-25.json) |
| 📄 2026-09-24 | 10 | [View JSON](data/daily/2026-09-24.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W39 | 89 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |
| 📅 2026-W37 | 96 | [View JSON](data/weekly/2026-W37.json) |
| 📅 2026-W36 | 88 | [View JSON](data/weekly/2026-W36.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |

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
