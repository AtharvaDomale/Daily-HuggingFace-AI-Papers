<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-29-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-3522+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">99</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">351</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">3522+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** April 19, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds</b> ⭐ 1.13k</summary>

<br/>

**👥 Authors:** Yisu Zhang, Zhenwei Wang, Xuhui Zuo, Chenjie Cao, Team HY-World

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14268) • [📄 arXiv](https://arxiv.org/abs/2604.14268) • [📥 PDF](https://arxiv.org/pdf/2604.14268)

**💻 Code:** [⭐ Code](https://github.com/Tencent-Hunyuan/HY-World-2.0)

> Here is the English PV of HY-World 2.0.

</details>

<details>
<summary><b>2. DR^{3}-Eval: Towards Realistic and Reproducible Deep Research Evaluation</b> ⭐ 22</summary>

<br/>

**👥 Authors:** Tiantian Xia, He Zhu, Qingheng Xiong, Qianqian Xie, Nucleon-17th

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14683) • [📄 arXiv](https://arxiv.org/abs/2604.14683) • [📥 PDF](https://arxiv.org/pdf/2604.14683)

**💻 Code:** [⭐ Code](https://github.com/NJU-LINK/DR3-Eval)

> Interesting breakdown of this paper on arXivLens: https://arxivlens.com/PaperView/Details/dr-3-eval-towards-realistic-and-reproducible-deep-research-evaluation-74-0eb2f238 Covers the executive summary, detailed methodology, and practical applicati...

</details>

<details>
<summary><b>3. RAD-2: Scaling Reinforcement Learning in a Generator-Discriminator Framework</b> ⭐ 207</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.15308) • [📄 arXiv](https://arxiv.org/abs/2604.15308) • [📥 PDF](https://arxiv.org/pdf/2604.15308)

**💻 Code:** [⭐ Code](https://github.com/hustvl/RAD)

> RAD-2 synergizes a Diffusion-based generator 𝒢 and a Transformer-based discriminator 𝒟 within a multi-stage optimization loop: (a) Pre-training Stage: 𝒢 is initialized via imitation learning to capture multi-modal trajectory priors from expert dem...

</details>

<details>
<summary><b>4. How to Fine-Tune a Reasoning Model? A Teacher-Student Cooperation Framework to Synthesize Student-Consistent SFT Data</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Qiming Ge, Feiyang Hao, Xu Huang, Kaichen Yang, njuhzx

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14164) • [📄 arXiv](https://arxiv.org/abs/2604.14164) • [📥 PDF](https://arxiv.org/pdf/2604.14164)

**💻 Code:** [⭐ Code](https://github.com/CoopReason/TESSY)

> 🚀 Motivation Training reasoning models (e.g., Qwen3) is highly sensitive to the data distribution. We observe that: ❗ Using off-policy data (e.g., directly from a strong teacher model) for SFT can lead to severe catastrophic forgetting , especiall...

</details>

<details>
<summary><b>5. GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.15284) • [📄 arXiv](https://arxiv.org/abs/2604.15284) • [📥 PDF](https://arxiv.org/pdf/2604.15284)

> GlobalSplat is a feed-forward 3D Gaussian Splatting method that learns a compact set of global scene tokens instead of allocating primitives per pixel. By aligning first and decoding later, it produces globally consistent reconstructions with as f...

</details>

<details>
<summary><b>6. ASGuard: Activation-Scaling Guard to Mitigate Targeted Jailbreaking Attack</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2509.25843) • [📄 arXiv](https://arxiv.org/abs/2509.25843) • [📥 PDF](https://arxiv.org/pdf/2509.25843)

**💻 Code:** [⭐ Code](https://github.com/dmis-lab/ASGuard)

> Asguard is a novel mechanistic safety framework that mitigates targeted jailbreak vulnerabilities in LLMs by directly intervening in internal activation dynamics rather than relying solely on data-level alignment. (1) Background: Large language mo...

</details>

<details>
<summary><b>7. HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yitian Liu, Zhixuan Liang, Yutian Chen, Guanyu Chen, Tianshuo Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14125) • [📄 arXiv](https://arxiv.org/abs/2604.14125) • [📥 PDF](https://arxiv.org/pdf/2604.14125)

> ✨ HiVLA is a hierarchical embodied manipulation agent system that combines visual grounding, semantic planning, and robust action execution for long-horizon and fine-grained robotic manipulation.

</details>

<details>
<summary><b>8. Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems</b> ⭐ 177</summary>

<br/>

**👥 Authors:** Zhiqiang Shen, Xinyi Shang, Xiaohan Zhao, Jiacheng Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14228) • [📄 arXiv](https://arxiv.org/abs/2604.14228) • [📥 PDF](https://arxiv.org/pdf/2604.14228)

**💻 Code:** [⭐ Code](https://github.com/VILA-Lab/Dive-into-Claude-Code)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures (...

</details>

<details>
<summary><b>9. UniDoc-RL: Coarse-to-Fine Visual RAG with Hierarchical Actions and Dense Rewards</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14967) • [📄 arXiv](https://arxiv.org/abs/2604.14967) • [📥 PDF](https://arxiv.org/pdf/2604.14967)

**💻 Code:** [⭐ Code](https://github.com/deepglint/UniDoc-RL)

> Retrieval-Augmented Generation (RAG) extends Large Vision-Language Models (LVLMs) with external visual knowledge. However, existing visual RAG systems typically rely on generic retrieval signals that overlook the fine-grained visual semantics esse...

</details>

<details>
<summary><b>10. Switch-KD: Visual-Switch Knowledge Distillation for Vision-Language Models</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14629) • [📄 arXiv](https://arxiv.org/abs/2604.14629) • [📥 PDF](https://arxiv.org/pdf/2604.14629)

**💻 Code:** [⭐ Code](https://github.com/haoyi199815/Switch-KD)

> ✨ Switch-KD is the open-source academic release of Li Auto's MindKD technology, published as Switch-KD at CVPR Findings 2026, enabling efficient vision-language model distillation through visual-switch supervision and unified multimodal knowledge ...

</details>

<details>
<summary><b>11. LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.15311) • [📄 arXiv](https://arxiv.org/abs/2604.15311) • [📥 PDF](https://arxiv.org/pdf/2604.15311)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Stepwise Credit Assignment for GRPO on Flow-Matching Models (2026) MotionRF...

</details>

<details>
<summary><b>12. Representations Before Pixels: Semantics-Guided Hierarchical Video Prediction</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Nikos Komodakis, Spyros Gidaris, Efstathios Karypidis

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.11707) • [📄 arXiv](https://arxiv.org/abs/2604.11707) • [📥 PDF](https://arxiv.org/pdf/2604.11707)

**💻 Code:** [⭐ Code](https://github.com/Sta8is/Re2Pix)

> Pixel or latent world models? Video world models fall into two camps: • generate photorealistic frames • predict semantic features of the future (e.g., DINOv2) Why choose one? We introduce Re2Pix, a hierarchical approach that combines both.

</details>

<details>
<summary><b>13. TRACER: Trace-Based Adaptive Cost-Efficient Routing for LLM Classification</b> ⭐ 123</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14531) • [📄 arXiv](https://arxiv.org/abs/2604.14531) • [📥 PDF](https://arxiv.org/pdf/2604.14531)

**💻 Code:** [⭐ Code](https://github.com/adrida/tracer)

> TRACER is an open-source system that replaces LLM calls with a smaller model when they closely agree, reducing cost while keeping control and interpretability.

</details>

<details>
<summary><b>14. OneHOI: Unifying Human-Object Interaction Generation and Editing</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14062) • [📄 arXiv](https://arxiv.org/abs/2604.14062) • [📥 PDF](https://arxiv.org/pdf/2604.14062)

**💻 Code:** [⭐ Code](https://github.com/jiuntian/OneHOI)

> [CVPR2026] OneHOI unifies Human-Object Interaction (HOI) generation and editing in a single, versatile model. It excels at challenging HOI editing, from text-guided changes to novel layout-guided control and novel multi-HOI edits. For generation, ...

</details>

<details>
<summary><b>15. RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.15231) • [📄 arXiv](https://arxiv.org/abs/2604.15231) • [📥 PDF](https://arxiv.org/pdf/2604.15231)

> New agent for CT report generation that unlocks step-by-step, tool-based reasoning for more accurate and transparent reports. "Anthropic's Faithfulness" (37%) compared to 0% in 3D VLM baseline.

</details>

<details>
<summary><b>16. LongAct: Harnessing Intrinsic Activation Patterns for Long-Context Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chenxuan Li, Qize Yu, Tingfeng Hui, Zijun Chen, Bowen Ping

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14922) • [📄 arXiv](https://arxiv.org/abs/2604.14922) • [📥 PDF](https://arxiv.org/pdf/2604.14922)

> ACL 2026

</details>

<details>
<summary><b>17. MM-WebAgent: A Hierarchical Multimodal Web Agent for Webpage Generation</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.15309) • [📄 arXiv](https://arxiv.org/abs/2604.15309) • [📥 PDF](https://arxiv.org/pdf/2604.15309)

**💻 Code:** [⭐ Code](https://github.com/microsoft/MM-webagent)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Please give a thumbs up to this comment if you found it helpful! If you wan...

</details>

<details>
<summary><b>18. Reinforcement Learning via Value Gradient Flow</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Amy Zhang, Somayeh Sojoudi, Kaiwen Hu, Haoran Xu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14265) • [📄 arXiv](https://arxiv.org/abs/2604.14265) • [📥 PDF](https://arxiv.org/pdf/2604.14265)

**💻 Code:** [⭐ Code](https://github.com/ryanxhr/vgf)

> ICLR 2026

</details>

<details>
<summary><b>19. Boosting Visual Instruction Tuning with Self-Supervised Guidance</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.12966) • [📄 arXiv](https://arxiv.org/abs/2604.12966) • [📥 PDF](https://arxiv.org/pdf/2604.12966)

**💻 Code:** [⭐ Code](https://github.com/sirkosophia/V-GIFT)

> Multimodal large language models (MLLMs) perform well on many vision-language tasks but often struggle with vision-centric problems that require fine-grained visual reasoning. Recent evidence suggests that this limitation arises not from weak visu...

</details>

<details>
<summary><b>20. KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.13226) • [📄 arXiv](https://arxiv.org/abs/2604.13226) • [📥 PDF](https://arxiv.org/pdf/2604.13226)

**💻 Code:** [⭐ Code](https://github.com/ChuangtaoChen-TUM/KVPacket)

> KV Packet is a framework for reusing precomputed KV caches across documents without recomputation. Code available at https://github.com/ChuangtaoChen-TUM/KVPacket .

</details>

<details>
<summary><b>21. SuperLocalMemory V3.3: The Living Brain -- Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval for Zero-LLM Agent Memory Systems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.04514) • [📄 arXiv](https://arxiv.org/abs/2604.04514) • [📥 PDF](https://arxiv.org/pdf/2604.04514)

**💻 Code:** [⭐ Code](https://github.com/qualixar/superlocalmemory)

> SuperLocalMemory V3.3 "The Living Brain" — the first unified memory + learning system for AI agents. Zero-cloud. EU AI Act compliant. And it KEEPS learning. This is the sequel to our V3 paper (arXiv:2603.14588) — now with learning, mesh coordinati...

</details>

<details>
<summary><b>22. Beyond Prompts: Unconditional 3D Inversion for Out-of-Distribution Shapes</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Maks Ovsjanikov, Léopold Maillard, Emery Pierson, Victoria Yue Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14914) • [📄 arXiv](https://arxiv.org/abs/2604.14914) • [📥 PDF](https://arxiv.org/pdf/2604.14914)

> The paper studies text-driven inversion of 3D generative models. It establishes the existence of sink traps : the model can become insensitive to prompts during generation, effectively collapsing to a single shape. Despite this property, the model...

</details>

<details>
<summary><b>23. Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Lawrence B. Hsieh, Pengfei Wei, dukesun99

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14572) • [📄 arXiv](https://arxiv.org/abs/2604.14572) • [📥 PDF](https://arxiv.org/pdf/2604.14572)

**💻 Code:** [⭐ Code](https://github.com/dukesun99/Corpus2Skill)

> We built and released Corpus2Skill (C2S), an agentic RAG system that replaces the traditional vector/BM25 retrieval stack with a navigable skill hierarchy the LLM browses directly at query time. On enterprise-style QA benchmarks, C2S matches or be...

</details>

<details>
<summary><b>24. Towards Autonomous Mechanistic Reasoning in Virtual Cells</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.11661) • [📄 arXiv](https://arxiv.org/abs/2604.11661) • [📥 PDF](https://arxiv.org/pdf/2604.11661)

**💻 Code:** [⭐ Code](https://github.com/valence-labs/VCR-Agent)

> VCR-Agent is a multi-agent system that can explain why a perturbation drives a particular cellular response, as opposed to simply describing the cellular response of a perturbation. This work falls under the “Explain” pillar of our Virtual Cell fr...

</details>

<details>
<summary><b>25. Cross-Tokenizer LLM Distillation through a Byte-Level Interface</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Davide Buffelli, Alberto Bernacchia, Alexandru Cioba, Yen-Chen Wu, Avyav Kumar Singh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.07466) • [📄 arXiv](https://arxiv.org/abs/2604.07466) • [📥 PDF](https://arxiv.org/pdf/2604.07466)

> Cross-tokenizer distillation (CTD), the transfer of knowledge from a teacher to a student language model when the two use different tokenizers, remains a largely unsolved problem. Existing approaches rely on heuristic strategies to align mismatche...

</details>

<details>
<summary><b>26. Three-Phase Transformer</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.14430) • [📄 arXiv](https://arxiv.org/abs/2604.14430) • [📥 PDF](https://arxiv.org/pdf/2604.14430)

**💻 Code:** [⭐ Code](https://github.com/achelousace/three-phase-transformer)

> In 1888, Nikola Tesla presented the world with three phase motor.. i did the same but for transformers. 😊 I just published "Three-Phase Transformer (3PT)" A research paper that takes Tesla's polyphase geometry and drops it into a Transformer's res...

</details>

<details>
<summary><b>27. C2: Scalable Rubric-Augmented Reward Modeling from Binary Preferences</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Saku Sugawara, Akira-k

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2604.13618) • [📄 arXiv](https://arxiv.org/abs/2604.13618) • [📥 PDF](https://arxiv.org/pdf/2604.13618)

**💻 Code:** [⭐ Code](https://github.com/asahi-research/C2)

> Rubrics are a powerful harness for aligning reward models with human judgment, but they come with two problems. Human-annotated rubrics are costly. Auto-generated rubrics are often vague or misleading, and can hurt the RM more than help. We introd...

</details>

<details>
<summary><b>28. An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Linh Ngo, Quan Dao, Hoang Phan, Hai Nguyen, Quyen Tran

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2211.16780) • [📄 arXiv](https://arxiv.org/abs/2211.16780) • [📥 PDF](https://arxiv.org/pdf/2211.16780)

> test

</details>

<details>
<summary><b>29. Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3</b> ⭐ 1</summary>

<br/>

**👥 Authors:** natnitaract

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.27844) • [📄 arXiv](https://arxiv.org/abs/2603.27844) • [📥 PDF](https://arxiv.org/pdf/2603.27844)

**💻 Code:** [⭐ Code](https://github.com/nat-nischw/model-capability-dominates-lessons-aimo3)

> Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3 Diverse Prompt Mixer assigns different reasoning strategies to majority-voting members to decorrelate errors. Tested on 50 IMO-level problems (1×H100, 5-hour limit, 3 mode...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 29 |
| 📅 Today | [`2026-04-19.json`](data/daily/2026-04-19.json) | 29 |
| 📆 This Week | [`2026-W15.json`](data/weekly/2026-W15.json) | 99 |
| 🗓️ This Month | [`2026-04.json`](data/monthly/2026-04.json) | 351 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-04-19 | 29 | [View JSON](data/daily/2026-04-19.json) |
| 📄 2026-04-18 | 29 | [View JSON](data/daily/2026-04-18.json) |
| 📄 2026-04-17 | 12 | [View JSON](data/daily/2026-04-17.json) |
| 📄 2026-04-16 | 10 | [View JSON](data/daily/2026-04-16.json) |
| 📄 2026-04-15 | 5 | [View JSON](data/daily/2026-04-15.json) |
| 📄 2026-04-14 | 5 | [View JSON](data/daily/2026-04-14.json) |
| 📄 2026-04-13 | 9 | [View JSON](data/daily/2026-04-13.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W15 | 99 | [View JSON](data/weekly/2026-W15.json) |
| 📅 2026-W14 | 140 | [View JSON](data/weekly/2026-W14.json) |
| 📅 2026-W13 | 115 | [View JSON](data/weekly/2026-W13.json) |
| 📅 2026-W12 | 120 | [View JSON](data/weekly/2026-W12.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-04 | 351 | [View JSON](data/monthly/2026-04.json) |
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
