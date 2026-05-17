<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-53-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-4011+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">53</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">218</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">390</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">4011+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** May 17, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling</b> ⭐ 65</summary>

<br/>

**👥 Authors:** Yizhuo Li, Shunkai Zhang, Haoran Zhang, Runzhe Zhan, Yafu Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13301) • [📄 arXiv](https://arxiv.org/abs/2605.13301) • [📥 PDF](https://arxiv.org/pdf/2605.13301)

**💻 Code:** [⭐ Code](https://github.com/Simplified-Reasoning/SU-01) • [⭐ Code](https://github.com/huggingface)

> We have open-sourced our code and model. Please check out our project page and GitHub repository: https://simplified-reasoning.github.io/SU-01/ https://github.com/Simplified-Reasoning/SU-01

</details>

<details>
<summary><b>2. Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15141) • [📄 arXiv](https://arxiv.org/abs/2605.15141) • [📥 PDF](https://arxiv.org/pdf/2605.15141)

**💻 Code:** [⭐ Code](https://github.com/thu-ml/Causal-Forcing) • [⭐ Code](https://github.com/huggingface)

> Try our model here: https://huggingface.co/zhuhz22/Causal-Forcing And the full-stack open-source code: https://github.com/thu-ml/Causal-Forcing We release 2-step frame-wise AR model with 50% latency and even better quality compared to 4-step chunk...

</details>

<details>
<summary><b>3. Self-Distilled Agentic Reinforcement Learning</b> ⭐ 64</summary>

<br/>

**👥 Authors:** Jinyang Wu, Zi-Han Wang, Zhuowen Han, Zhiyuan Yao, Zhengxi Lu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15155) • [📄 arXiv](https://arxiv.org/abs/2605.15155) • [📥 PDF](https://arxiv.org/pdf/2605.15155)

**💻 Code:** [⭐ Code](https://github.com/ZJU-REAL/SDAR) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>4. MemLens: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14906) • [📄 arXiv](https://arxiv.org/abs/2605.14906) • [📥 PDF](https://arxiv.org/pdf/2605.14906)

**💻 Code:** [⭐ Code](https://github.com/xrenaf/MEMLENS) • [⭐ Code](https://github.com/huggingface)

> We have open-sourced our code and dataset. Please check out our GitHub repository: https://github.com/xrenaf/MEMLENS

</details>

<details>
<summary><b>5. SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15178) • [📄 arXiv](https://arxiv.org/abs/2605.15178) • [📥 PDF](https://arxiv.org/pdf/2605.15178)

**💻 Code:** [⭐ Code](https://github.com/NVlabs/Sana/) • [⭐ Code](https://github.com/huggingface)

> A 2.6B open-source world model that turns one image and a camera trajectory into 720p, minute-long, controllable video on a single GPU. Project Page: https://nvlabs.github.io/Sana/WM/ Code: https://github.com/NVlabs/Sana/

</details>

<details>
<summary><b>6. MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory</b> ⭐ 25</summary>

<br/>

**👥 Authors:** Boxuan Zhang, Yihao Quan, Zeru Shi, Qingyue Jiao, Minghao Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15128) • [📄 arXiv](https://arxiv.org/abs/2605.15128) • [📥 PDF](https://arxiv.org/pdf/2605.15128)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MinghoKwok/MemEye)

> MemEye is a vision-centric long-term memory benchmark designed to evaluate how agents remember and reason over long-running image-grounded interactions. The benchmark focuses on assessing agents’ abilities to retain and utilize visual information ...

</details>

<details>
<summary><b>7. Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of Language-Model Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14386) • [📄 arXiv](https://arxiv.org/abs/2605.14386) • [📥 PDF](https://arxiv.org/pdf/2605.14386)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> FINAL Bench introduces a new evaluation paradigm for LLMs: functional metacognitive reasoning — not just "can the model solve it," but "does the model know when, why, and how it solves it." 100 tasks across 15 domains, built on the TICOS framework...

</details>

<details>
<summary><b>8. Beyond Individual Intelligence: Surveying Collaboration, Failure Attribution, and Self-Evolution in LLM-based Multi-Agent Systems</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14892) • [📄 arXiv](https://arxiv.org/abs/2605.14892) • [📥 PDF](https://arxiv.org/pdf/2605.14892)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/mira-ai-lab/awesome-mas-life)

> LLM-based autonomous agents have demonstrated strong capabilities in reasoning, planning, and tool use, yet remain limited when tasks require sustained coordination across roles, tools, and environments. Multi-agent systems address this through st...

</details>

<details>
<summary><b>9. WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation</b> ⭐ 371</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.10912) • [📄 arXiv](https://arxiv.org/abs/2605.10912) • [📥 PDF](https://arxiv.org/pdf/2605.10912)

**💻 Code:** [⭐ Code](https://github.com/internlm/WildClawBench) • [⭐ Code](https://github.com/InternLM/WildClawBench) • [⭐ Code](https://github.com/huggingface)

> Large language and vision-language models increasingly power agents that act on a user's behalf through command-line interface (CLI) harnesses. However, most agent benchmarks still rely on synthetic sandboxes, short-horizon tasks, mock-service API...

</details>

<details>
<summary><b>10. STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06527) • [📄 arXiv](https://arxiv.org/abs/2605.06527) • [📥 PDF](https://arxiv.org/pdf/2605.06527)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We would greatly appreciate it if you could upvote.

</details>

<details>
<summary><b>11. Warp-as-History: Generalizable Camera-Controlled Video Generation from One Training Video</b> ⭐ 78</summary>

<br/>

**👥 Authors:** Tong He, Yifan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15182) • [📄 arXiv](https://arxiv.org/abs/2605.15182) • [📥 PDF](https://arxiv.org/pdf/2605.15182)

**💻 Code:** [⭐ Code](https://github.com/yyfz/Warp-as-History) • [⭐ Code](https://github.com/huggingface)

> Our method enables interactive camera trajectory following and viewpoint manipulation, similar to HappyOyster and Genie 3, using only a single camera-annotated training example.

</details>

<details>
<summary><b>12. RouteProfile: Elucidating the Design Space of LLM Profiles for Routing</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.00180) • [📄 arXiv](https://arxiv.org/abs/2605.00180) • [📥 PDF](https://arxiv.org/pdf/2605.00180)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ulab-uiuc/RouteProfile)

> This paper proposes RouteProfile, a general LLM profile design space spanning four key dimensions, and systematically demonstrates that structured, query-level profiles consistently improve routing performance and generalization to newly introduce...

</details>

<details>
<summary><b>13. PREPING: Building Agent Memory without Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13880) • [📄 arXiv](https://arxiv.org/abs/2605.13880) • [📥 PDF](https://arxiv.org/pdf/2605.13880)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> LLM agents often need memory to solve tasks in new tool environments, but memory is usually built only after seeing human tasks or deployment-time user interactions. This creates a cold-start problem: the agent needs memory most when it has the le...

</details>

<details>
<summary><b>14. EvolveMem:Self-Evolving Memory Architecture via AutoResearch for LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Cihang Xie, Zeyu Zheng, Peng Xia, Xinyu Ye, Jiaqi Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13941) • [📄 arXiv](https://arxiv.org/abs/2605.13941) • [📥 PDF](https://arxiv.org/pdf/2605.13941)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present EvolveMem, a self-evolving memory architecture that exposes its full retrieval configuration as a structured action space optimized by an LLM-powered diagnosis module.

</details>

<details>
<summary><b>15. Realiz3D: 3D Generation Made Photorealistic via Domain-Aware Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13852) • [📄 arXiv](https://arxiv.org/abs/2605.13852) • [📥 PDF](https://arxiv.org/pdf/2605.13852)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Realiz3D is a framework that leverages both real and synthetic data to train diffusion models that generate photorealistic images while faithfully adhering to input conditions and maintaining 3D consistency.

</details>

<details>
<summary><b>16. Long Context Pre-Training with Lighthouse Attention</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.06554) • [📄 arXiv](https://arxiv.org/abs/2605.06554) • [📥 PDF](https://arxiv.org/pdf/2605.06554)

**💻 Code:** [⭐ Code](https://github.com/ighoshsubho/lighthouse-attention) • [⭐ Code](https://github.com/huggingface)

> We propose Lighthouse Attention, a training-only symmetrical selection-based hierarchical attention algorithm that wraps around ordinary SDPA and can be easily removed towards the end of the training.

</details>

<details>
<summary><b>17. ATLAS: Agentic or Latent Visual Reasoning? One Word is Enough for Both</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pheng-Ann Heng, Xinyan Chen, Rain Liu, Ziyu Guo

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15198) • [📄 arXiv](https://arxiv.org/abs/2605.15198) • [📥 PDF](https://arxiv.org/pdf/2605.15198)

**💻 Code:** [⭐ Code](https://github.com/ZiyuGuo99/ATLAS) • [⭐ Code](https://github.com/huggingface)

> Project Page: https://atlas-oneword.github.io Code: https://github.com/ZiyuGuo99/ATLAS

</details>

<details>
<summary><b>18. FrontierSmith: Synthesizing Open-Ended Coding Problems at Scale</b> ⭐ 23</summary>

<br/>

**👥 Authors:** Hanchen Li, Kaiyuan Liu, Shang Zhou, Qiuyang Mang, Runyuan He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14445) • [📄 arXiv](https://arxiv.org/abs/2605.14445) • [📥 PDF](https://arxiv.org/pdf/2605.14445)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/FrontierCS/FrontierSmith)

> https://frontier-cs.org/blog/frontiersmith/

</details>

<details>
<summary><b>19. Learning to Communicate Locally for Large-Scale Multi-Agent Pathfinding</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuri Kuratov, Oleg Bulichev, Anton Andreychuk, Alsu Sagirova, Valeriy Vyaltsev

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.07637) • [📄 arXiv](https://arxiv.org/abs/2605.07637) • [📥 PDF](https://arxiv.org/pdf/2605.07637)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> LC-MAPF provides iterative message exchange among agents to enable progressive refinement of predicted action distributions over multiple communication rounds.

</details>

<details>
<summary><b>20. DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhen Xing, Yujie Wei, Kaixun Jiang, Junqiu Yu, Quanhao Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15055) • [📄 arXiv](https://arxiv.org/abs/2605.15055) • [📥 PDF](https://arxiv.org/pdf/2605.15055)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Project page: https://quanhaol.github.io/DiffusionOPD-site/

</details>

<details>
<summary><b>21. IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14712) • [📄 arXiv](https://arxiv.org/abs/2605.14712) • [📥 PDF](https://arxiv.org/pdf/2605.14712)

**💻 Code:** [⭐ Code](https://github.com/ZGC-EmbodyAI/IntentVLA) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>22. VGGT-Edit: Feed-forward Native 3D Scene Editing with Residual Field Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15186) • [📄 arXiv](https://arxiv.org/abs/2605.15186) • [📥 PDF](https://arxiv.org/pdf/2605.15186)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://arxiv.org/pdf/2605.15186

</details>

<details>
<summary><b>23. Orchard: An Open-Source Agentic Modeling Framework</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15040) • [📄 arXiv](https://arxiv.org/abs/2605.15040) • [📥 PDF](https://arxiv.org/pdf/2605.15040)

**💻 Code:** [⭐ Code](https://github.com/microsoft/Orchard) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>24. PanoWorld: Towards Spatial Supersensing in 360^circ Panorama World</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13169) • [📄 arXiv](https://arxiv.org/abs/2605.13169) • [📥 PDF](https://arxiv.org/pdf/2605.13169)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> PanoWorld

</details>

<details>
<summary><b>25. ViMU: Benchmarking Video Metaphorical Understanding</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Xinchao Wang, Qi Li

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14607) • [📄 arXiv](https://arxiv.org/abs/2605.14607) • [📥 PDF](https://arxiv.org/pdf/2605.14607)

**💻 Code:** [⭐ Code](https://github.com/LiQiiiii/Video-Metaphorical-Understanding) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>26. RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO</b> ⭐ 29</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15190) • [📄 arXiv](https://arxiv.org/abs/2605.15190) • [📥 PDF](https://arxiv.org/pdf/2605.15190)

**💻 Code:** [⭐ Code](https://github.com/mvp-ai-lab/RAVEN) • [⭐ Code](https://github.com/huggingface)

> Causal autoregressive video diffusion models support real-time streaming generation by extrapolating future chunks from previously generated content. Distilling such generators from high-fidelity bidirectional teachers yields competitive few-step ...

</details>

<details>
<summary><b>27. Forcing-KV: Hybrid KV Cache Compression for Efficient Autoregressive Video Diffusion Models</b> ⭐ 69</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.09681) • [📄 arXiv](https://arxiv.org/abs/2605.09681) • [📥 PDF](https://arxiv.org/pdf/2605.09681)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zju-jiyicheng/Forcing-KV)

> No abstract available.

</details>

<details>
<summary><b>28. Does Synthetic Layered Design Data Benefit Layered Design Decomposition?</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15167) • [📄 arXiv](https://arxiv.org/abs/2605.15167) • [📥 PDF](https://arxiv.org/pdf/2605.15167)

**💻 Code:** [⭐ Code](https://github.com/YangHaolin0526/SynLayers) • [⭐ Code](https://github.com/huggingface)

> Pure synthetic layered design dataset can indeed benefit the layer decomposition task.

</details>

<details>
<summary><b>29. CurveBench: A Benchmark for Exact Topological Reasoning over Nested Jordan Curves</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Naser Talebizadeh Saradari, Morteza Saghafian, Mona Mohammadi, Amirreza Mohseni

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14068) • [📄 arXiv](https://arxiv.org/abs/2605.14068) • [📥 PDF](https://arxiv.org/pdf/2605.14068)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Amir-Mohseni/CurveBench)

> We introduce CurveBench, a benchmark for testing whether vision-language models can recover hierarchical region-containment trees from images of non-intersecting Jordan curves. The task targets visual topology and structured reasoning beyond simpl...

</details>

<details>
<summary><b>30. BOOKMARKS: Efficient Active Storyline Memory for Role-playing</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14169) • [📄 arXiv](https://arxiv.org/abs/2605.14169) • [📥 PDF](https://arxiv.org/pdf/2605.14169)

**💻 Code:** [⭐ Code](https://github.com/KomeijiForce/BOOKMARKS_Koishiday_2026) • [⭐ Code](https://github.com/huggingface)

> BOOKMARKS: Efficient Active Storyline Memory for Role-playing

</details>

<details>
<summary><b>31. WildTableBench: Benchmarking Multimodal Foundation Models on Table Understanding In the Wild</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.01018) • [📄 arXiv](https://arxiv.org/abs/2605.01018) • [📥 PDF](https://arxiv.org/pdf/2605.01018)

**💻 Code:** [⭐ Code](https://github.com/hjzhe/WildTableBench) • [⭐ Code](https://github.com/huggingface)

> We introduce WildTableBench, the first QA benchmark for evaluating multimodal foundation models on naturally occurring table images collected from real-world web sources (Reddit, Pinterest, etc.). Unlike prior benchmarks built on structured text o...

</details>

<details>
<summary><b>32. Learning to Build the Environment: Self-Evolving Reasoning RL via Verifiable Environment Synthesis</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Wenhao Yu, Dian Yu, Kishan Panaganti, Zhenwen Liang, Yucheng Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14392) • [📄 arXiv](https://arxiv.org/abs/2605.14392) • [📥 PDF](https://arxiv.org/pdf/2605.14392)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>33. PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14269) • [📄 arXiv](https://arxiv.org/abs/2605.14269) • [📥 PDF](https://arxiv.org/pdf/2605.14269)

**💻 Code:** [⭐ Code](https://github.com/h6kplus/PhyMotion) • [⭐ Code](https://github.com/huggingface)

> Generating realistic human motion is a central yet unsolved challenge in video generation. While reinforcement learning (RL)-based post-training has driven recent gains in general video quality, extending it to human motion remains bottlenecked by...

</details>

<details>
<summary><b>34. PRISM: Prior Rectification and Uncertainty-Aware Structure Modeling for Diffusion-Based Text Image Super-Resolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13027) • [📄 arXiv](https://arxiv.org/abs/2605.13027) • [📥 PDF](https://arxiv.org/pdf/2605.13027)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> .

</details>

<details>
<summary><b>35. Sat3DGen: Comprehensive Street-Level 3D Scene Generation from Single Satellite Image</b> ⭐ 32</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14984) • [📄 arXiv](https://arxiv.org/abs/2605.14984) • [📥 PDF](https://arxiv.org/pdf/2605.14984)

**💻 Code:** [⭐ Code](https://github.com/qianmingduowan/Sat3DGen) • [⭐ Code](https://github.com/huggingface)

> Given a single satellite image, Sat3DGen generates a street-view-renderable NeRF-based 3D scene with strong geometry, enabling large-area meshing, multi-camera surround-view video, semantic-map-to-3D, and single-image DSM estimation. Demo: https:/...

</details>

<details>
<summary><b>36. FutureSim: Replaying World Events to Evaluate Adaptive Agents</b> ⭐ 17</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15188) • [📄 arXiv](https://arxiv.org/abs/2605.15188) • [📥 PDF](https://arxiv.org/pdf/2605.15188)

**💻 Code:** [⭐ Code](https://github.com/OpenForecaster/futuresim) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>37. Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.11458) • [📄 arXiv](https://arxiv.org/abs/2605.11458) • [📥 PDF](https://arxiv.org/pdf/2605.11458)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> @ librarian-bot recommend

</details>

<details>
<summary><b>38. Aligning Latent Geometry for Spherical Flow Matching in Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15193) • [📄 arXiv](https://arxiv.org/abs/2605.15193) • [📥 PDF](https://arxiv.org/pdf/2605.15193)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>39. Topology-Preserving Neural Operator Learning via Hodge Decomposition</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.13834) • [📄 arXiv](https://arxiv.org/abs/2605.13834) • [📥 PDF](https://arxiv.org/pdf/2605.13834)

**💻 Code:** [⭐ Code](https://github.com/ContinuumCoder/Hodge-Spectral-Duality) • [⭐ Code](https://github.com/huggingface)

> In this paper, we study solution operators of physical field equations on geometric meshes from a function-space perspective. We reveal that Hodge orthogonality fundamentally resolves spectral interference by isolating unlearnable topological degr...

</details>

<details>
<summary><b>40. RewardHarness: Self-Evolving Agentic Post-Training</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.08703) • [📄 arXiv](https://arxiv.org/abs/2605.08703) • [📥 PDF](https://arxiv.org/pdf/2605.08703)

**💻 Code:** [⭐ Code](https://github.com/KlingAIResearch/RewardHarness) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/TIGER-AI-Lab/RewardHarness)

> Project: https://rewardharness.com/

</details>

<details>
<summary><b>41. Unlocking Complex Visual Generation via Closed-Loop Verified Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14876) • [📄 arXiv](https://arxiv.org/abs/2605.14876) • [📥 PDF](https://arxiv.org/pdf/2605.14876)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Current text-to-image models still rely heavily on a “single-step generation” paradigm: the model is expected to satisfy all semantic constraints in one denoising process. As prompts become more compositional, this often leads to counting failures...

</details>

<details>
<summary><b>42. Dynamic Latent Routing</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14323) • [📄 arXiv](https://arxiv.org/abs/2605.14323) • [📥 PDF](https://arxiv.org/pdf/2605.14323)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Humans live in a continuous world, yet think in discrete language. LLMs live in a tokenized world, yet think in continuous representations. So is discrete language merely a communication artifact? Our answer: no. Language is not just discrete; it ...

</details>

<details>
<summary><b>43. LLM-based Detection of Manipulative Political Narratives</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14354) • [📄 arXiv](https://arxiv.org/abs/2605.14354) • [📥 PDF](https://arxiv.org/pdf/2605.14354)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/SinclairSchneider/manipulative_narrative_detection)

> We present a new computational framework for detecting and structuring manipulative political narratives. A task that became more important due to the shift of political discussions to social media. One of the primary challenges thereby is differe...

</details>

<details>
<summary><b>44. Ideology Prediction of German Political Texts</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14352) • [📄 arXiv](https://arxiv.org/abs/2605.14352) • [📥 PDF](https://arxiv.org/pdf/2605.14352)

**💻 Code:** [⭐ Code](https://github.com/SinclairSchneider/german_ideology_prediction) • [⭐ Code](https://github.com/huggingface)

> Elections represent a crucial milestone in a nation's ongoing development. To better understand the political rhetoric from various movements, ranging from left to right, we propose a transformer-based model capable of projecting the political ori...

</details>

<details>
<summary><b>45. Boosting Omni-Modal Language Models: Staged Post-Training with Visually Debiased Evaluation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.12034) • [📄 arXiv](https://arxiv.org/abs/2605.12034) • [📥 PDF](https://arxiv.org/pdf/2605.12034)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Omni-modal language models are intended to jointly understand audio, visual inputs, and language, but benchmark gains can be inflated when visual evidence alone is enough to answer a query. We study whether current omni-modal benchmarks separate v...

</details>

<details>
<summary><b>46. Boosting Reinforcement Learning with Verifiable Rewards via Randomly Selected Few-Shot Guidance</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15012) • [📄 arXiv](https://arxiv.org/abs/2605.15012) • [📥 PDF](https://arxiv.org/pdf/2605.15012)

**💻 Code:** [⭐ Code](https://github.com/KaiYan289/FEST) • [⭐ Code](https://github.com/huggingface)

> This is a work aimed for boosting RLVR performance using only minimal amount of SFT data in a unified training paradigm. Check our code at https://github.com/KaiYan289/FEST and checkpoints/dataset at https://huggingface.co/collections/kaiyan289/fe...

</details>

<details>
<summary><b>47. BEAM: Binary Expert Activation Masking for Dynamic Routing in MoE</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14438) • [📄 arXiv](https://arxiv.org/abs/2605.14438) • [📥 PDF](https://arxiv.org/pdf/2605.14438)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Interesting work on practical MoE inference acceleration. BEAM introduces trainable binary expert masks for token-adaptive routing, which helps avoid the redundant computation of fixed Top-K routing while staying more deployment-friendly than meth...

</details>

<details>
<summary><b>48. LiSA: Lifelong Safety Adaptation via Conservative Policy Induction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14454) • [📄 arXiv](https://arxiv.org/abs/2605.14454) • [📥 PDF](https://arxiv.org/pdf/2605.14454)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> As LLM agents read private data, call tools, and run multi-step workflows, guardrail failures stop being answer-quality issues — they become real harms: leaked secrets, unsafe actions, blocked legitimate work. And the hardest failures are contextu...

</details>

<details>
<summary><b>49. Nexus : An Agentic Framework for Time Series Forecasting</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Vishy Tirumalashetty, Nanyun Peng, Mihir Parmar, Palash Goyal, Sarkar Snigdha Sarathi Das

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14389) • [📄 arXiv](https://arxiv.org/abs/2605.14389) • [📥 PDF](https://arxiv.org/pdf/2605.14389)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>50. Quantitative Video World Model Evaluation for Geometric-Consistency</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Xueyan Zou, Yuheng Li, Yinling Zhang, Yihao Pi, Jiaxin Wu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.15185) • [📄 arXiv](https://arxiv.org/abs/2605.15185) • [📥 PDF](https://arxiv.org/pdf/2605.15185)

**💻 Code:** [⭐ Code](https://github.com/AnteaWu/PDI-Bench) • [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>51. SPIN: Structural LLM Planning via Iterative Navigation for Industrial Tasks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.14051) • [📄 arXiv](https://arxiv.org/abs/2605.14051) • [📥 PDF](https://arxiv.org/pdf/2605.14051)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Industrial LLM agent systems often separate planning from execution, yet LLM planners frequently produce structurally invalid or unnecessarily long workflows, leading to brittle failures and avoidable tool and API cost. We propose \texttt{SPIN}, a...

</details>

<details>
<summary><b>52. PreScam: A Benchmark for Predicting Scam Progression from Early Conversations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.12243) • [📄 arXiv](https://arxiv.org/abs/2605.12243) • [📥 PDF](https://arxiv.org/pdf/2605.12243)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Conversational scams, such as romance and investment scams, are emerging as a major form of online fraud. Unlike one-shot scam lures such as fake lottery or unpaid toll messages, they unfold through multi-turn conversations in which scammers gradu...

</details>

<details>
<summary><b>53. Overcoming Dynamics-Blindness: Training-Free Pace-and-Path Correction for VLA Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kai Ye, Xinpeng Li, Vikash Singh, Chaoda Song, Yanyan Zhang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2605.11459) • [📄 arXiv](https://arxiv.org/abs/2605.11459) • [📥 PDF](https://arxiv.org/pdf/2605.11459)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This paper proposes Pace-and-Path Correction, a training-free, closed-form inference-time operator that wraps any chunked-action VLA.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 53 |
| 📅 Today | [`2026-05-17.json`](data/daily/2026-05-17.json) | 53 |
| 📆 This Week | [`2026-W19.json`](data/weekly/2026-W19.json) | 218 |
| 🗓️ This Month | [`2026-05.json`](data/monthly/2026-05.json) | 390 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-05-17 | 53 | [View JSON](data/daily/2026-05-17.json) |
| 📄 2026-05-16 | 53 | [View JSON](data/daily/2026-05-16.json) |
| 📄 2026-05-15 | 26 | [View JSON](data/daily/2026-05-15.json) |
| 📄 2026-05-14 | 22 | [View JSON](data/daily/2026-05-14.json) |
| 📄 2026-05-13 | 22 | [View JSON](data/daily/2026-05-13.json) |
| 📄 2026-05-12 | 16 | [View JSON](data/daily/2026-05-12.json) |
| 📄 2026-05-11 | 26 | [View JSON](data/daily/2026-05-11.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W19 | 218 | [View JSON](data/weekly/2026-W19.json) |
| 📅 2026-W18 | 113 | [View JSON](data/weekly/2026-W18.json) |
| 📅 2026-W17 | 84 | [View JSON](data/weekly/2026-W17.json) |
| 📅 2026-W16 | 74 | [View JSON](data/weekly/2026-W16.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-05 | 390 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
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
