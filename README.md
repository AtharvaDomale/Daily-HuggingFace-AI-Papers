<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-26-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-6052+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">26</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">109</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">671</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">6052+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 27, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. FrontierChallenge: Evaluating Scientific Workflow Completion</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24979) • [📄 arXiv](https://arxiv.org/abs/2608.24979) • [📥 PDF](https://arxiv.org/pdf/2608.24979)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ApodexAI/FrontierAgent/tree/main/benchmarks/frontierchallenge)

> Everyone says AI agents can already analyze data, run code, generate figures, and write reports. But can they actually complete a scientific workflow? Today, we’re introducing FrontierChallenge, a new benchmark evaluating whether AI agents can com...

</details>

<details>
<summary><b>2. VoiceMem: Streaming Dual-Brain Memory for Real-Time Interaction</b> ⭐ 28</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26005) • [📄 arXiv](https://arxiv.org/abs/2608.26005) • [📥 PDF](https://arxiv.org/pdf/2608.26005)

**💻 Code:** [⭐ Code](https://github.com/xzf-thu/VoiceMem) • [⭐ Code](https://github.com/huggingface)

> Memory foundation for real-time voice interaction. Project page: https://github.com/xzf-thu/VoiceMem Code: https://github.com/xzf-thu/VoiceMem Model: https://huggingface.co/zhifeixie/VoiceMem_MF_Qwen3_6_35B_A3B_Qlora Dataset: https://huggingface.c...

</details>

<details>
<summary><b>3. WarpSAC: Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24479) • [📄 arXiv](https://arxiv.org/abs/2608.24479) • [📥 PDF](https://arxiv.org/pdf/2608.24479)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/wzhhasadream/warprl)

> The project page provides an overview of WarpSAC, demonstrations, benchmark results, and sim-to-real evaluations. The GitHub repository contains the source code, installation instructions, training scripts, environment integrations, and experiment...

</details>

<details>
<summary><b>4. VGI-BENCH: Probing Visual Intelligence in Video Generation Models</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yuxuan Zhang, Linrui Ma, Yuhao Cheng, Cong Wei, Xuan He

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19583) • [📄 arXiv](https://arxiv.org/abs/2608.19583) • [📥 PDF](https://arxiv.org/pdf/2608.19583)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hexuan21/VGI-Bench)

> We introduce VGI-Bench, a benchmark for probing visual intelligence in video generation models beyond perceptual quality. It evaluates whether video generators can actually reason through evolving visual processes, with 27 tasks and 810 instances ...

</details>

<details>
<summary><b>5. JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25593) • [📄 arXiv](https://arxiv.org/abs/2608.25593) • [📥 PDF](https://arxiv.org/pdf/2608.25593)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/bingreeky/JIT)

> Model-as-a-Harness; Harness Intelligence Model Website: https://bingreeky.github.io/JIT-site/ GitHub: https://github.com/bingreeky/JIT Hugging Face: https://huggingface.co/datasets/JIT-Agent/jit-meta-harness

</details>

<details>
<summary><b>6. Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds</b> ⭐ 1.95k</summary>

<br/>

**👥 Authors:** Weiyang Jin, Haoyang Huang, Nan Duan, JunhaoZhuang, jahnsonblack

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.23383) • [📄 arXiv](https://arxiv.org/abs/2608.23383) • [📥 PDF](https://arxiv.org/pdf/2608.23383)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/jd-opensource/JoyAI-Echo)

> JoyAI-Echo-1.5: Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds

</details>

<details>
<summary><b>7. D^3-MOPD: Adaptive Dynamic Domain ScheDuling for Efficient Multi-Teacher Distillation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mu Chuan, Juntao Li, Fei Zhao, Zhiwei Zhang, Zechen Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24987) • [📄 arXiv](https://arxiv.org/abs/2608.24987) • [📥 PDF](https://arxiv.org/pdf/2608.24987)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Multi-teacher on-policy distillation trains a single student from several domain-expert teachers, but different domains converge at very different rates, so the fixed data mixtures used in prior work keep spending rollouts on domains that have alr...

</details>

<details>
<summary><b>8. Agent-G^2: Gaussian Guidance for Agentic Reinforcement Learning</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.23318) • [📄 arXiv](https://arxiv.org/abs/2608.23318) • [📥 PDF](https://arxiv.org/pdf/2608.23318)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/Agent-G2)

> We propose Agent-G^2, a Gaussian guidance framework that draws the depth per task from a Gaussian whose center and spread are estimated online from rollouts already collected for policy optimization, requiring no probe rollouts or learned depth pr...

</details>

<details>
<summary><b>9. VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ran Ji, Maijunxian Wang, Fanyi Pu, Ruisi Wang, Junxiang Xu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26105) • [📄 arXiv](https://arxiv.org/abs/2608.26105) • [📥 PDF](https://arxiv.org/pdf/2608.26105)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Video-Reason/VBVR-Pro) • [⭐ Code](https://github.com/Video-Reason/VBVR-Pro-Bench)

> Homepage: https://video-reason.com/ Data and Models: https://huggingface.co/collections/Video-Reason/vbvr-pro-a-scalable-and-verifiable-suite-for-native-visual Training Code: https://github.com/Video-Reason/VBVR-Pro Eval Code: https://github.com/V...

</details>

<details>
<summary><b>10. StreamPI: Streaming Multimodal Temporal Modeling for Vision-Language-Action Models</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Xianzhe Fan, Zhenya Yang, Yuxiang Lu, Jinghua Hou, Zhe Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.26067) • [📄 arXiv](https://arxiv.org/abs/2608.26067) • [📥 PDF](https://arxiv.org/pdf/2608.26067)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hku-sail/StreamPI)

> Vision-Language-Action (VLA) models have demonstrated effectiveness in robot manipulation, yet state-of-the-art models such as pi0.5 operate under a single-frame paradigm, limiting their ability to retain past observations and develop precise spat...

</details>

<details>
<summary><b>11. Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation</b> ⭐ 43</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19098) • [📄 arXiv](https://arxiv.org/abs/2608.19098) • [📥 PDF](https://arxiv.org/pdf/2608.19098)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/BytedTsinghua-SIA/Open-MOPD)

> No abstract available.

</details>

<details>
<summary><b>12. Video-IFBench: Evaluating Instruction Following of Multimodal LLMs in Video Understanding Scenarios</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kai Zou, Peiyuan Zhang, Sihan Liu, Peixian Chen, Hongbo Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25529) • [📄 arXiv](https://arxiv.org/abs/2608.25529) • [📥 PDF](https://arxiv.org/pdf/2608.25529)

**💻 Code:** [⭐ Code](https://github.com/Alexios-hub/Video-IFBench) • [⭐ Code](https://github.com/huggingface)

> Video-IFBench evaluates whether video MLLMs can follow complex user instructions with semantic, format, and conditional constraints, revealing substantial gaps beyond standard video understanding accuracy.

</details>

<details>
<summary><b>13. V-Rubrics: Visual Faithfulness via Rubric-Based Reinforcement Learning</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25580) • [📄 arXiv](https://arxiv.org/abs/2608.25580) • [📥 PDF](https://arxiv.org/pdf/2608.25580)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/shulin16/v-rubrics)

> Vision-language models can produce fluent answers that are visually wrong: a single unsupported object, chart value, or intermediate inference can invalidate an otherwise plausible response. We argue that this is a credit-assignment failure in mul...

</details>

<details>
<summary><b>14. Are Android GUI Agents Robust Against Runtime Anomalies? AnTrap: Evaluating Agents in Dynamic Adversarial Environments</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Tingyu Song, Jinbiao Wei, Cong Chen, Yilun Zhao, Guo Gan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24099) • [📄 arXiv](https://arxiv.org/abs/2608.24099) • [📥 PDF](https://arxiv.org/pdf/2608.24099)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/gguogan/AnTrap)

> We propose AnTrap, a dynamic adversarial evaluation framework that injects realistic runtime anomalies across State, Thinking, Action, and Round levels to stress-test Android GUI agents, uncovering universal performance degradation and intrinsic r...

</details>

<details>
<summary><b>15. Is Next-Chunk Reasoning RL Really Better than SFT? Revisiting Training Strategies under no-CoT Data</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ziyi Wang, Jiangning Liu, Yanan Sun, Youqing Fang, Yinhao Tang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.23256) • [📄 arXiv](https://arxiv.org/abs/2608.23256) • [📥 PDF](https://arxiv.org/pdf/2608.23256)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>16. Rubrics as Visual-Repair Context for Self-Evolving UI-to-Code Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ruichun Ma, Chung-Ching Lin, Xiaofei Wang, Zhengyuan Yang, Tianyi Xiong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24138) • [📄 arXiv](https://arxiv.org/abs/2608.24138) • [📥 PDF](https://arxiv.org/pdf/2608.24138)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Title: Rubrics as Visual-Repair Context for Self-Evolving UI-to-Code Generation This paper introduces RubSE, a rubric-guided self-evolution framework that structures visual feedback into typed, targeted rubrics to achieve stable and effective iter...

</details>

<details>
<summary><b>17. Code World Model: Coding Agent as World Brain</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chi Zhang, Guosheng Lin, Yiwen Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25927) • [📄 arXiv](https://arxiv.org/abs/2608.25927) • [📥 PDF](https://arxiv.org/pdf/2608.25927)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>18. A Programming Paradigm for Spatiotemporal Composability</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25512) • [📄 arXiv](https://arxiv.org/abs/2608.25512) • [📥 PDF](https://arxiv.org/pdf/2608.25512)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>19. The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Ron Litman, Adi Kalyanpur, Mor Shpigel Nacson, Roy Ganz

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24358) • [📄 arXiv](https://arxiv.org/abs/2608.24358) • [📥 PDF](https://arxiv.org/pdf/2608.24358)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We study what happens when coding agents switch models mid-trajectory. Across Claude and GPT model pairs, escalation suffers a substantial “handoff tax,” while downshifting offers a more favorable cost–quality trade-off—and the best handoff interf...

</details>

<details>
<summary><b>20. MA-VLA: Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25864) • [📄 arXiv](https://arxiv.org/abs/2608.25864) • [📥 PDF](https://arxiv.org/pdf/2608.25864)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zhangzaibin/future-robots)

> Multi-arm collaboration is becoming a core capability in embodied manipulation. Recent vision-language-action (VLA) models integrate perception, language, and control, but most represent language as a single global instruction and do not provide a...

</details>

<details>
<summary><b>21. Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation in Transformers</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15062) • [📄 arXiv](https://arxiv.org/abs/2608.15062) • [📥 PDF](https://arxiv.org/pdf/2608.15062)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Amr-Hegazy1/gated-recurrent-transformer)

> Hi, I really liked this paper and all the ablation studies you did, congrats! Do you plan to share the repo you used to train this model and the checkpoints? I wanted to check how does EBT (Energy based transformer) compared with the RecurrentGPT.

</details>

<details>
<summary><b>22. RetrievalRouter: Joint Modality and Architecture Selection for Document Retrieval</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Noel Crespi, Reza Farahbakhsh, Mehmet Onur Keskin, Emre Kuru

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.25625) • [📄 arXiv](https://arxiv.org/abs/2608.25625) • [📥 PDF](https://arxiv.org/pdf/2608.25625)

**💻 Code:** [⭐ Code](https://github.com/emrekuruu/retrieval-router) • [⭐ Code](https://github.com/huggingface)

> Accepted at EMNLP 2026

</details>

<details>
<summary><b>23. FIRM-Video: Check Before You Score for Reliable Text-to-Video Reward Modeling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.21839) • [📄 arXiv](https://arxiv.org/abs/2608.21839) • [📥 PDF](https://arxiv.org/pdf/2608.21839)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> FIRM-Video introduces a checklist-driven, check-before-score framework for reliable and efficient text-to-video reward modeling. It decomposes evaluation into verifiable criteria for instruction following, world coherence, and perceptual quality, ...

</details>

<details>
<summary><b>24. Stream4D: 4D-Consistency for Streaming Autoregressive Diffusion Video Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.19556) • [📄 arXiv](https://arxiv.org/abs/2608.19556) • [📥 PDF](https://arxiv.org/pdf/2608.19556)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://banyuanhao.github.io/Stream4D/

</details>

<details>
<summary><b>25. Super Star: Towards Streaming Real-time Interactive Agents for Digital Humans</b> ⭐ 5</summary>

<br/>

**👥 Authors:** Xin Wang, Yajing Chen, Haidi Fan, Youchen Xie, Wentao Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.24909) • [📄 arXiv](https://arxiv.org/abs/2608.24909) • [📥 PDF](https://arxiv.org/pdf/2608.24909)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/PeterIverson/Super-Star)

> No abstract available.

</details>

<details>
<summary><b>26. Real-TurnTurk: A Multimodal Turkish Corpus for Turn-Taking Prediction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Alper Kaplan, Mustafa Sertaç Türkel, Bekir Berker Türker, Fatma Nur Korkmaz, Ahmet Tuğrul Bayrak

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.22071) • [📄 arXiv](https://arxiv.org/abs/2608.22071) • [📥 PDF](https://arxiv.org/pdf/2608.22071)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Turn-taking is a basic organizational feature of human conversation and remains difficult to model in natural, synchronous dialog systems. While existing research has explored multimodal approaches and large language models for turn-ending predict...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 26 |
| 📅 Today | [`2026-08-27.json`](data/daily/2026-08-27.json) | 26 |
| 📆 This Week | [`2026-W34.json`](data/weekly/2026-W34.json) | 109 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 671 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-27 | 26 | [View JSON](data/daily/2026-08-27.json) |
| 📄 2026-08-26 | 35 | [View JSON](data/daily/2026-08-26.json) |
| 📄 2026-08-25 | 22 | [View JSON](data/daily/2026-08-25.json) |
| 📄 2026-08-24 | 26 | [View JSON](data/daily/2026-08-24.json) |
| 📄 2026-08-23 | 26 | [View JSON](data/daily/2026-08-23.json) |
| 📄 2026-08-22 | 26 | [View JSON](data/daily/2026-08-22.json) |
| 📄 2026-08-21 | 22 | [View JSON](data/daily/2026-08-21.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W34 | 109 | [View JSON](data/weekly/2026-W34.json) |
| 📅 2026-W33 | 213 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 671 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |
| 🗓️ 2026-05 | 782 | [View JSON](data/monthly/2026-05.json) |
| 🗓️ 2026-04 | 450 | [View JSON](data/monthly/2026-04.json) |
| 🗓️ 2026-03 | 604 | [View JSON](data/monthly/2026-03.json) |

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
