<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-28-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7212+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">28</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">85</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">604</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7212+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** October 15, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Any Time, Any Space: A World-State-Centric Framework for Agentic Video Generation</b> ⭐ 65</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20571) • [📄 arXiv](https://arxiv.org/abs/2610.20571) • [📥 PDF](https://arxiv.org/pdf/2610.20571)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ziplab/ATAS)

> Excited to share ATAS — Any Time, Any Space ! Video models can produce impressive individual shots, but keeping a whole story consistent is harder: a room should keep its layout when revisited, and different cameras should capture the same event a...

</details>

<details>
<summary><b>2. LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Zongliang Wu, Tong Zhu, Xiaoyang Chen, Yunkai Chai, Zefeng He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19975) • [📄 arXiv](https://arxiv.org/abs/2610.19975) • [📥 PDF](https://arxiv.org/pdf/2610.19975)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/lcqysl/LatentLoop)

> We introduce LatentLoop, which improves looped MoE models under matched budgets by compressing the first loop visit, preserving full resolution in the second, and reallocating the saved compute to increase model width.

</details>

<details>
<summary><b>3. Seeing, Maintaining, and Learning: The Evolution of Multimodal Agent Memory</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19366) • [📄 arXiv](https://arxiv.org/abs/2610.19366) • [📥 PDF](https://arxiv.org/pdf/2610.19366)

**💻 Code:** [⭐ Code](https://github.com/Seeing-Maintaining-Learning/multimodal-agent-memory-evolution) • [⭐ Code](https://github.com/huggingface)

> How can multimodal agents remember what they perceive, keep that memory useful as the world changes, and learn to manage it better over time? We are sharing our survey, “Seeing, Maintaining, and Learning: The Evolution of Multimodal Agent Memory.”...

</details>

<details>
<summary><b>4. LensErase: A Foundation Model for Erase Anything</b> ⭐ 29</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18072) • [📄 arXiv](https://arxiv.org/abs/2610.18072) • [📥 PDF](https://arxiv.org/pdf/2610.18072)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/LensFamily/LensErase)

> LensErase: A foundation model for Erase Anything, enabling high-quality removal of objects, text, watermarks, and associated visual effects while precisely preserving unedited regions.

</details>

<details>
<summary><b>5. Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.13521) • [📄 arXiv](https://arxiv.org/abs/2610.13521) • [📥 PDF](https://arxiv.org/pdf/2610.13521)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Developed jointly by JD.com’s OxygenVLM team and Nanjing University, RoadMOPD mitigates learning conflicts among teachers with substantially different response styles and domain expertise, helping a student integrate multiple experts’ capabilities...

</details>

<details>
<summary><b>6. Hybrid Reasoning Agents</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18861) • [📄 arXiv](https://arxiv.org/abs/2610.18861) • [📥 PDF](https://arxiv.org/pdf/2610.18861)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/togethercomputer/HybridAgent)

> Your LLM agent has to think out loud, token by token, before it can search, run code, or act in the world. We humans don’t. We ponder in silence and speak when it matters. Human language was never meant to be the limit of machine thought. Introduc...

</details>

<details>
<summary><b>7. APT: Feedback-Driven Agentic Post-Training for Text-to-Image Generation</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18097) • [📄 arXiv](https://arxiv.org/abs/2610.18097) • [📥 PDF](https://arxiv.org/pdf/2610.18097)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/siruizhang30/APT)

> Better models. Smarter training data. Different image models struggle with different things. A fixed training set cannot keep up with every model’s evolving strengths and weaknesses. APT makes post-training adaptive. Three agents work together to ...

</details>

<details>
<summary><b>8. Recovering Reliable Supervision via Critique-Guided On-Policy Distillation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Amanda Hughes, Yuxiao Yang, Shangzhe Li, Tianrun Yu, Kaixiang Zhao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20572) • [📄 arXiv](https://arxiv.org/abs/2610.20572) • [📥 PDF](https://arxiv.org/pdf/2610.20572)

**💻 Code:** [⭐ Code](https://github.com/kzhao5/CriticOPD) • [⭐ Code](https://github.com/huggingface)

> 🤖 CriticOPD: Recovering Reliable Supervision via Critique-Guided On-Policy Distillation 🔍 Key finding: Teachers can recognize student mistakes yet still follow them when continuing a flawed solution, making OPD supervision unreliable. ✍️ Our appro...

</details>

<details>
<summary><b>9. HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile Execution</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.02089) • [📄 arXiv](https://arxiv.org/abs/2610.02089) • [📥 PDF](https://arxiv.org/pdf/2610.02089)

**💻 Code:** [⭐ Code](https://github.com/SNU-PI/HumanoidToolBench) • [⭐ Code](https://github.com/huggingface)

> Can humanoids choose the right tool and then actually use it? We introduce HumanoidToolBench, an 18-task benchmark evaluating humanoid tool selection and execution across three scenarios, three execution levels, and two tool-set modes. We also int...

</details>

<details>
<summary><b>10. Towards Agentic Robots: Model, Data, Environment, and Harness</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xinlei Yu, Jiaxu Wang, Kaituo Feng, Dongming Wu, Chonghan Qin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20442) • [📄 arXiv](https://arxiv.org/abs/2610.20442) • [📥 PDF](https://arxiv.org/pdf/2610.20442)

**💻 Code:** [⭐ Code](https://github.com/chonghanqin/Awesome-Agentic-Robots) • [⭐ Code](https://github.com/huggingface)

> Repo: https://github.com/chonghanqin/Awesome-Agentic-Robots

</details>

<details>
<summary><b>11. RAR: Rate-Adaptive Rollout for Time-Consistent Interactive World Models</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Zeke Xie, Shengju Qian, Yihua Du, Dongyu Yan, Yaoyang Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.16302) • [📄 arXiv](https://arxiv.org/abs/2610.16302) • [📥 PDF](https://arxiv.org/pdf/2610.16302)

**💻 Code:** [⭐ Code](https://github.com/HKUST-LongGroup/RAR) • [⭐ Code](https://github.com/huggingface)

> Hi everyone! We’re sharing RAR (Rate-Adaptive Rollout), a framework for controllable generation rates in interactive world models. Most world models run at a fixed temporal rate, so differences in hardware throughput can unintentionally speed up o...

</details>

<details>
<summary><b>12. More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19789) • [📄 arXiv](https://arxiv.org/abs/2610.19789) • [📥 PDF](https://arxiv.org/pdf/2610.19789)

**💻 Code:** [⭐ Code](https://github.com/aim-uofa/Self-Distillation-AgentRL) • [⭐ Code](https://github.com/huggingface)

> We present a large-scale, systematic evaluation of self-distillation methods in agentic reinforcement learning, revealing that existing methods struggle to consistently outperform GRPO despite increased computational costs. We hope our work provid...

</details>

<details>
<summary><b>13. GroundGUI: GUI Grounding by High-Quality Data and Distillation-to-Reinforcement Learning</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Madalina Ciobanu, Zhifeng Li, Jiacheng Tang, Jiahao Tian, Qingsong Xie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19100) • [📄 arXiv](https://arxiv.org/abs/2610.19100) • [📥 PDF](https://arxiv.org/pdf/2610.19100)

**💻 Code:** [⭐ Code](https://github.com/PrentisAI/GroundGUI) • [⭐ Code](https://github.com/huggingface)

> ScreenRef and Distillation-to-Reinforcement Learning (D2RL) are the core contributions of GroundGUI. ScreenRef is a high-quality GUI grounding dataset spanning desktop, web, and mobile environments. It pairs diverse screenshots with referring expr...

</details>

<details>
<summary><b>14. Player-1: Learning to Play from Synthetic Experience</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20584) • [📄 arXiv](https://arxiv.org/abs/2610.20584) • [📥 PDF](https://arxiv.org/pdf/2610.20584)

**💻 Code:** [⭐ Code](https://github.com/TencentARC/Player-1) • [⭐ Code](https://github.com/huggingface)

> Synthetic environments improve transferable agent capabilities: Player-1 is an agentic post-training pipeline that learns general game control beyond supervised demonstrations.

</details>

<details>
<summary><b>15. Doc2Harness: Benchmarking LLMs on Long-Horizon Harness Engineering of Coding Agents</b> ⭐ 32</summary>

<br/>

**👥 Authors:** Chenxu Liu, Chendi Ge, Peizhong Gao, Shenghua Liu, Yuyao Ge

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20344) • [📄 arXiv](https://arxiv.org/abs/2610.20344) • [📥 PDF](https://arxiv.org/pdf/2610.20344)

**💻 Code:** [⭐ Code](https://github.com/YuyaoGe/doc2harness) • [⭐ Code](https://github.com/huggingface)

> Agents driven by large language models are increasingly deployed on long-horizon tasks, where they use tools and modify real workspaces. How well they perform depends as much on the harness around the model as on the model itself. Existing benchma...

</details>

<details>
<summary><b>16. Physical Time as a Missing Dimension of Robot Policy Memory Dynamics</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20524) • [📄 arXiv](https://arxiv.org/abs/2610.20524) • [📥 PDF](https://arxiv.org/pdf/2610.20524)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Robot policies act in physical time, but their memories are often organized around observation steps. What happens when these two clocks disagree? The same observation history can require different actions depending on how much time has elapsed. A...

</details>

<details>
<summary><b>17. Are We Testing Too Much? A Survey of Benchmark Compression for Efficient Evaluation of Large Language Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.18248) • [📄 arXiv](https://arxiv.org/abs/2610.18248) • [📥 PDF](https://arxiv.org/pdf/2610.18248)

**💻 Code:** [⭐ Code](https://github.com/MilkThink-Lab/Awesome-Benchmark-Compression) • [⭐ Code](https://github.com/huggingface)

> First Survey of Benchmark Compression for Efficient Evaluation of LLMs

</details>

<details>
<summary><b>18. Long-Horizon RL Needs Muon</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qianshan Wei, Ziqi Zhou, Tao Chen, Jinghao Lin, Kai Ruan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20256) • [📄 arXiv](https://arxiv.org/abs/2610.20256) • [📥 PDF](https://arxiv.org/pdf/2610.20256)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Long-horizon agents must coordinate reasoning, tool use, and environment feedback over extended interactions. Learning from sparse outcome rewards requires weighting trajectories of different lengths and updating the policy while rollouts are stil...

</details>

<details>
<summary><b>19. Acting Through Registers</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17638) • [📄 arXiv](https://arxiv.org/abs/2610.17638) • [📥 PDF](https://arxiv.org/pdf/2610.17638)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Z1zyw/RegisterWAM)

> World Action Models (WAMs) use generative visual representations to condition robot actions, yet how these representations organize action-relevant information remains poorly understood. We identify persistent high-norm spatial outliers in WAM rep...

</details>

<details>
<summary><b>20. Exploring the Limits of Combinatorial Generalization in Embodied Manipulation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.15425) • [📄 arXiv](https://arxiv.org/abs/2610.15425) • [📥 PDF](https://arxiv.org/pdf/2610.15425)

**💻 Code:** [⭐ Code](https://github.com/AutoLab-SAI-SJTU/RoboCombine) • [⭐ Code](https://github.com/huggingface)

> In embodied manipulation, Vision Language Action (VLA) models can execute atomic skills, but long-horizon manipulation requires recombining these skills without demonstrations for every skill combination. The full instruction route treats skill ch...

</details>

<details>
<summary><b>21. Diff Diffusion: Predicting Edit Paths with Diffusion Models for Efficient Code Editing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ningyuan Sun, Zhiwen Deng, Bo Jiang, Hao Jiang, Xiaolong Xu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17831) • [📄 arXiv](https://arxiv.org/abs/2610.17831) • [📥 PDF](https://arxiv.org/pdf/2610.17831)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Diffusion large language models (dLLMs) are appealing for code generation because they can update many tokens in parallel, yet many are optimized for sequence synthesis rather than edit prediction. This setting is poorly matched to modern AI-assis...

</details>

<details>
<summary><b>22. Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Donghoon Kim, Hyesoo Hong, Hamish Ivison, Wooseok Seo, Sangyeon Yoon

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19525) • [📄 arXiv](https://arxiv.org/abs/2610.19525) • [📥 PDF](https://arxiv.org/pdf/2610.19525)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> .

</details>

<details>
<summary><b>23. The COUNT: Counting N-grams Unlocks Scaling on Host Memory and Training-Free Adaptation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20456) • [📄 arXiv](https://arxiv.org/abs/2610.20456) • [📥 PDF](https://arxiv.org/pdf/2610.20456)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/IanMagnusson/COUNT)

> COUNT adds an n-gram embedding to a transformer without adding learned parameters. At each position, it looks up the most frequent next tokens in a table of corpus n-gram counts held in CPU RAM, and adds the model’s own output-head rows for those ...

</details>

<details>
<summary><b>24. DecisionRobust: Benchmarking Robustness for JEV Style Decision Models Across Text and Vision</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yongkang Du, Hanyu Wang, Yurui Chang, Zhongye Liu, Yifan Lan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19118) • [📄 arXiv](https://arxiv.org/abs/2610.19118) • [📥 PDF](https://arxiv.org/pdf/2610.19118)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce DecisionRobust, a benchmark for testing the robustness of JEV-style decision models across text and vision, including classification, routing, tool selection, and robotics scenarios. Our public release contains 1,058 curated attack ex...

</details>

<details>
<summary><b>25. Training LLMs for Multi-Agent Collaboration with Verifiable Information Rewards</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shaofeng Zou, Nathaniel D. Bastian, Yingbin Liang, Sen Lin, CliverChen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.17893) • [📄 arXiv](https://arxiv.org/abs/2610.17893) • [📥 PDF](https://arxiv.org/pdf/2610.17893)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🚀 Can we train LLMs to collaborate, rather than only design better multi-agent workflows? I'm sharing our new preprint, Training LLMs for Multi-Agent Collaboration with Verifiable Information Rewards (CoSIR). Most LLM post-training focuses on chat...

</details>

<details>
<summary><b>26. G^3VLA: Geometric Inductive Bias for Vision-Language-Action Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yanheng Zhu, Khuyen Pham, Artur Habuda, ColinSkywalker, pengyue-polaron

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2606.24472) • [📄 arXiv](https://arxiv.org/abs/2606.24472) • [📥 PDF](https://arxiv.org/pdf/2606.24472)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We introduce G³VLA, a camera-aware module that helps pretrained VLAs use the geometric relationships between multiple camera views. It combines intrinsic-conditioned ray embeddings, projective positional encoding, and bidirectional cross-view fusi...

</details>

<details>
<summary><b>27. MA-Bench: Measuring Operational Planning in Language Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuan Xue, Phil Sage, Robert Sincero, Rodney Lafuente-Mercado

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.19138) • [📄 arXiv](https://arxiv.org/abs/2610.19138) • [📥 PDF](https://arxiv.org/pdf/2610.19138)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>28. Continuum Attention</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Afshin Khadangi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.20586) • [📄 arXiv](https://arxiv.org/abs/2610.20586) • [📥 PDF](https://arxiv.org/pdf/2610.20586)

**💻 Code:** [⭐ Code](https://github.com/akhadangi/continuum-attention) • [⭐ Code](https://github.com/huggingface)

> We introduce Continuum Attention (CA), a causal Transformer attention operator that replaces discrete key and value lookup with integration over a continuous, piecewise linear memory curve.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 28 |
| 📅 Today | [`2026-10-15.json`](data/daily/2026-10-15.json) | 28 |
| 📆 This Week | [`2026-W41.json`](data/weekly/2026-W41.json) | 85 |
| 🗓️ This Month | [`2026-10.json`](data/monthly/2026-10.json) | 604 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-10-15 | 28 | [View JSON](data/daily/2026-10-15.json) |
| 📄 2026-10-14 | 21 | [View JSON](data/daily/2026-10-14.json) |
| 📄 2026-10-13 | 12 | [View JSON](data/daily/2026-10-13.json) |
| 📄 2026-10-12 | 24 | [View JSON](data/daily/2026-10-12.json) |
| 📄 2026-10-11 | 72 | [View JSON](data/daily/2026-10-11.json) |
| 📄 2026-10-10 | 72 | [View JSON](data/daily/2026-10-10.json) |
| 📄 2026-10-09 | 35 | [View JSON](data/daily/2026-10-09.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W41 | 85 | [View JSON](data/weekly/2026-W41.json) |
| 📅 2026-W40 | 282 | [View JSON](data/weekly/2026-W40.json) |
| 📅 2026-W39 | 326 | [View JSON](data/weekly/2026-W39.json) |
| 📅 2026-W38 | 98 | [View JSON](data/weekly/2026-W38.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-10 | 604 | [View JSON](data/monthly/2026-10.json) |
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
