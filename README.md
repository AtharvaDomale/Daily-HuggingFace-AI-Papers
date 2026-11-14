<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-34-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-7973+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">34</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">73</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">245</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">7973+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** November 14, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.15777) • [📄 arXiv](https://arxiv.org/abs/2611.15777) • [📥 PDF](https://arxiv.org/pdf/2611.15777)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/holdrain/SO-Arena)

> SO-Arena benchmarks the capabilities and decision reliability of System One models across 11 domains, revealing that strong task performance does not guarantee stable decisions or recognition of missing correct options. Homepage: https://lichenlil...

</details>

<details>
<summary><b>2. BirdDreams: Joint Perspective and BEV Video Generation for a Plausible, Reactive Driving World Model</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.14815) • [📄 arXiv](https://arxiv.org/abs/2611.14815) • [📥 PDF](https://arxiv.org/pdf/2611.14815)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> BirdDreams: A driving world model with accurate ego control and traffic that can react. BirdDreams is a driving world model that generates driving video and a bird’s-eye view of the scene from a single camera image.  It follows input ego trajector...

</details>

<details>
<summary><b>3. PhysLoop: Harnessing LLMs' Physics Awareness for Self-Improving Video Generation Models</b> ⭐ 32</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16748) • [📄 arXiv](https://arxiv.org/abs/2611.16748) • [📥 PDF](https://arxiv.org/pdf/2611.16748)

**💻 Code:** [⭐ Code](https://github.com/DAGroup-PKU/PhysLoop) • [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DAGroup-PKU/PhysLoop/issues/1)

> PhysLoop is a physics-aware harness for building controllable video training data and repairing physically inconsistent video at inference time. The project video shows how the harness works and presents examples of video generation and repair. Co...

</details>

<details>
<summary><b>4. G-STAR: Group-Scaled Token Capacity for Deeply Compressed Autoregressive Generation</b> ⭐ 25</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.13004) • [📄 arXiv](https://arxiv.org/abs/2611.13004) • [📥 PDF](https://arxiv.org/pdf/2611.13004)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DAGroup-PKU/G-STAR)

> Discrete autoregressive models offer a scalable framework for image generation. However, they face a fundamental trade-off: long token sequences preserve representation quality but slow generation, while aggressive spatial compression reduces repr...

</details>

<details>
<summary><b>5. Duo: A Reciprocal World Model for Full-Duplex Interaction</b> ⭐ 27</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.15131) • [📄 arXiv](https://arxiv.org/abs/2611.15131) • [📥 PDF](https://arxiv.org/pdf/2611.15131)

**💻 Code:** [⭐ Code](https://github.com/inclusionAI/Duo) • [⭐ Code](https://github.com/huggingface)

> arXiv: https://arxiv.org/abs/2611.15131 Homepage: https://duo-v1.github.io/ Github: https://github.com/inclusionAI/Duo Hugging Face Model: https://huggingface.co/collections/inclusionAI/duo-v1

</details>

<details>
<summary><b>6. Open Vision Banana: A Fully Open Recipe for Generative Visual Perception</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.15844) • [📄 arXiv](https://arxiv.org/abs/2611.15844) • [📥 PDF](https://arxiv.org/pdf/2611.15844)

**💻 Code:** [⭐ Code](https://github.com/OpenVisionBanana/OpenVisionBanana) • [⭐ Code](https://github.com/huggingface)

> A Fully Open Recipe for Generative Visual Perception

</details>

<details>
<summary><b>7. Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents</b> ⭐ 41</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17167) • [📄 arXiv](https://arxiv.org/abs/2609.36012) • [📥 PDF](https://arxiv.org/pdf/2611.17167)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/decision-agents)

> Decision models output typed decisions (e.g., a probability over candidate actions), so the same model can act as an Actor (choose actions), a Judge (evaluate actions already taken), or a Teacher (score every action to provide step-level rewards f...

</details>

<details>
<summary><b>8. Ropedia Xperience 2: A Gold Standard for Scaling Physical AI Environments and Behaviors</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17060) • [📄 arXiv](https://arxiv.org/abs/2609.36012) • [📥 PDF](https://arxiv.org/pdf/2611.17060)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Ropedia/Ropedia-Xperience-2)

> Introducing Xperience 2 from Ropedia: a gold standard for scaling Physical AI environments and behaviors. Real-robot data is expensive. DROID's 350 hours of demonstrations took 50 operators a full year, and a real robot can't retry a task the way ...

</details>

<details>
<summary><b>9. On the Design of Hyper-Connections: Simplification, Stability, and Scalability</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.05984) • [📄 arXiv](https://arxiv.org/abs/2611.05984) • [📥 PDF](https://arxiv.org/pdf/2611.05984)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/aHapBean/SimpleHC)

> We introduce SimpleHC, one of the fastest Hyper-Connections variants to date, reducing repeated memory access through a simpler design. It adds just 2.4% training overhead at the largest tested width while maintaining competitive performance with ...

</details>

<details>
<summary><b>10. TTB: Test-time Baking for Efficient Rendering of Decoder-only View Synthesis Models</b> ⭐ 32</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17331) • [📄 arXiv](https://arxiv.org/abs/2611.17331) • [📥 PDF](https://arxiv.org/pdf/2611.17331)

**💻 Code:** [⭐ Code](https://github.com/NVlabs/DVSM) • [⭐ Code](https://github.com/huggingface)

> Test-time Baking (TTB) turns cross-view attention layers into fixed-size MLP layers at test time, each fit like a light field in latent space. Rendering then needs no 3D representation and costs the same at any number of input views, at the qualit...

</details>

<details>
<summary><b>11. WorldDraft: If Code Builds the World, Can Video Models Just Render It?</b> ⭐ 8</summary>

<br/>

**👥 Authors:** Chenfei Liao, Weilin Ruan, Wen-Jie Shu, Zhiyu Hou, Harold328

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16217) • [📄 arXiv](https://arxiv.org/abs/2611.16217) • [📥 PDF](https://arxiv.org/pdf/2611.16217)

**💻 Code:** [⭐ Code](https://github.com/KevinHuge/WorldDraft) • [⭐ Code](https://github.com/huggingface)

> If Code Builds the World, Can Video Models Just Render It?

</details>

<details>
<summary><b>12. Unbundling Multimodal Vision Encoding</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17334) • [📄 arXiv](https://arxiv.org/abs/2611.17334) • [📥 PDF](https://arxiv.org/pdf/2611.17334)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> How multimodal language models should encode visual input remains up for debate. Most models use a pretrained vision encoder, while some recent recipes train the encoder from scratch with the language model or remove it entirely. Vision encoders b...

</details>

<details>
<summary><b>13. Factored One-sided Adam-Moment for Practical and Scalable SOAP</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16905) • [📄 arXiv](https://arxiv.org/abs/2611.16905) • [📥 PDF](https://arxiv.org/pdf/2611.16905)

**💻 Code:** [⭐ Code](https://github.com/naver-ai/foam) • [⭐ Code](https://github.com/huggingface)

> We introduce FOAM, a lightweight SOAP variant that preconditions only the input side and factors the rotated second moment, halving SOAP’s optimizer state and removing one of its two eigen-decompositions. FOAM matches strong SOAP and Muon baseline...

</details>

<details>
<summary><b>14. UniSpark: Exploring Hybrid Attention for Multimodal Understanding and Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16517) • [📄 arXiv](https://arxiv.org/abs/2611.16517) • [📥 PDF](https://arxiv.org/pdf/2611.16517)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Can hybrid attention support both understanding and visual generation? UniSpark brings full attention and recurrent linear attention into a shared Mixture-of-Transformer-Experts backbone.

</details>

<details>
<summary><b>15. Learning the Task or Learning the Verifier? Rethinking Criteria Exposure in Embodied Self-Improvement</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08318) • [📄 arXiv](https://arxiv.org/abs/2611.08318) • [📥 PDF](https://arxiv.org/pdf/2611.08318)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/DAGroup-PKU/PyRUA-Lean/tree/RSIJudge)

> https://github.com/DAGroup-PKU/PyRUA-Lean/tree/RSIJudge

</details>

<details>
<summary><b>16. Who Remembers What? Multi-Granularity Memory Extraction for Multi-User, Multi-Agent Interaction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.10439) • [📄 arXiv](https://arxiv.org/abs/2611.10439) • [📥 PDF](https://arxiv.org/pdf/2611.10439)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Today, memory in personal coding agents like Claude Code and Codex belongs to one user and one agent. A fix one user discovers never reaches anyone else, so other users, on other agents, solve the same problem again. Cross-user learning shares tho...

</details>

<details>
<summary><b>17. TurtLES: A Large-Scale Benchmark for Turbulent 3D Neural PDE Surrogates</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.09567) • [📄 arXiv](https://arxiv.org/abs/2611.09567) • [📥 PDF](https://arxiv.org/pdf/2611.09567)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/itsakk/TurtLES)

> TurtLES is a 14 TB benchmark of 400 high-fidelity 3D turbulent cylinder wakes (Re 1000–3900, four cross-section shapes, 0–45° angle of attack) on dense irregular meshes with 400 time steps each, built to test neural PDE surrogates on realistic tur...

</details>

<details>
<summary><b>18. Consensus Capture in Reference-Based UAV Video Stabilization: A Benchmark on Dynamic Urban Scenes</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17190) • [📄 arXiv](https://arxiv.org/abs/2611.17190) • [📥 PDF](https://arxiv.org/pdf/2611.17190)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/rfonod/stabilo-benchmark)

> Reference-frame stabilization registers every frame of a hovering-drone video onto one reference frame, and its registration pipelines are usually picked on feature-matching benchmarks where only the camera moves. Over a busy intersection, vehicle...

</details>

<details>
<summary><b>19. JustRUA: A Controlled Study of GPT-6 Robot-Use Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Cheng Yin, Mingqi Gao, Xuanwei Shi, Yufan Dang, ZefanW

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16009) • [📄 arXiv](https://arxiv.org/abs/2609.36012) • [📥 PDF](https://arxiv.org/pdf/2611.16009)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZefanW/JustRUA)

> JustRUA is a controlled study of GPT-6 Astra robot-use agents on RoboDojo. We take the designs recent systems add (harness, action interface, reuse of experience, scene geometry) and test them one at a time in one framework. The harness we end up ...

</details>

<details>
<summary><b>20. DeepStress: Stress-Testing Deep Search Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2607.13920) • [📄 arXiv](https://arxiv.org/abs/2607.17291) • [📥 PDF](https://arxiv.org/pdf/2607.13920)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Hi everyone! This is the first paper of my PhD, which I presented at the Workshop on Agent Behavior earlier this year at COLM! In this paper, we look at how we can use a simulated environment to test our search agents in "difficult" situations and...

</details>

<details>
<summary><b>21. HiVe-Mem: Interleaving Horizontal Memory Expansion and Vertical Evidence Composition for Multimodal Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.12872) • [📄 arXiv](https://arxiv.org/abs/2611.12872) • [📥 PDF](https://arxiv.org/pdf/2611.12872)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ViktorAxelsen/HiVe-Mem)

> As multimodal agents accumulate increasingly long interaction histories, effectively retrieving and utilizing past information becomes critical for long-term interaction. Existing memory systems typically rely on semantic retrieval over compressed...

</details>

<details>
<summary><b>22. Agent-R^2: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.08219) • [📄 arXiv](https://arxiv.org/abs/2611.08219) • [📥 PDF](https://arxiv.org/pdf/2611.08219)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-REAL/Agent-R2)

> Real productivity work means turning heterogeneous files (annual reports, spreadsheets, PDFs) into deliverables with the right format and evidence. LLM agents still struggle here: workflows are long, files are messy, and success criteria are fine-...

</details>

<details>
<summary><b>23. Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chu Wang, Ziyuan Wang, Genliang Zhu, chen-ai-researcher

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2610.41453) • [📄 arXiv](https://arxiv.org/abs/2610.41453) • [📥 PDF](https://arxiv.org/pdf/2610.41453)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Tool timeouts can leave AI agents facing ambiguous tool outcomes and partial execution: an action may have failed, committed or remained in flight while other workflow steps have already completed. We formulate evidence-guided task recovery over a...

</details>

<details>
<summary><b>24. Do Multimodal Models Decide Consistently on Equivalent Optical Spectra?</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Genliang Zhu, Ziyuan Wang, chen-ai-researcher

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.00428) • [📄 arXiv](https://arxiv.org/abs/2611.00428) • [📥 PDF](https://arxiv.org/pdf/2611.00428)

**💻 Code:** [⭐ Code](https://github.com/zwang-devv/spectra-decision-audit) • [⭐ Code](https://github.com/huggingface)

> A spectrum can be given to a model as a table in nanometers or as a plot in photon energy, and the decision should not depend on which one it sees. When the data needed for an answer are missing, the model should say so. We test representation inv...

</details>

<details>
<summary><b>25. Back to Pixels: Lifting Latent Video Diffusion into Pixel Space</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.17336) • [📄 arXiv](https://arxiv.org/abs/2611.17336) • [📥 PDF](https://arxiv.org/pdf/2611.17336)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>26. The Value of Thinking: Learning When to Reason in Hybrid Decision Models</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Weilin Luo, Haitham Ammar, Meng Fang, Ke Xu, Zhengxu Yu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16961) • [📄 arXiv](https://arxiv.org/abs/2611.16961) • [📥 PDF](https://arxiv.org/pdf/2611.16961)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zhengxuyu/HDM-code)

> HDM, a hybrid decision model: one that decides for itself whether an answer needs autoregressive generation at all, so a question costs either no token or a full reasoning pass.

</details>

<details>
<summary><b>27. The Narrative Gap: Can LLMs Help Us Navigate Diverse Narratives Across Languages?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.12750) • [📄 arXiv](https://arxiv.org/abs/2611.12750) • [📥 PDF](https://arxiv.org/pdf/2611.12750)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/isle-dev/Narrative-Bench-NeurIPS-2026)

> What information capabilities do #AI systems need to help us, informavores, navigate diverse, multicultural narratives in our current & future information ecosystems? Do current models support them? No. Introducing NarrativeBench. https://huggingf...

</details>

<details>
<summary><b>28. Experience-Handoff Reinforcement Learning for Recursive Self-Improvement</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16409) • [📄 arXiv](https://arxiv.org/abs/2609.36012) • [📥 PDF](https://arxiv.org/pdf/2611.16409)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/doyoungkim-ml/ehrl)

> Why not train an agent's weights inside the harness it runs in? Reinforcement learning (RL) is the standard way to train LLM agents, and it boils a whole attempt down to one reward number. Everything the agent saw along the way is thrown out: erro...

</details>

<details>
<summary><b>29. Structured Confidence Perturbation via Fisher Geometry for Adversarial Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.14046) • [📄 arXiv](https://arxiv.org/abs/2611.14046) • [📥 PDF](https://arxiv.org/pdf/2611.14046)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Adversarial training is often plagued by "pseudo-attacks"—perturbations that significantly increase the loss value but fail to truly alter the model's confidence structure. To address this, we propose two complementary approaches to directly attac...

</details>

<details>
<summary><b>30. Humanized but Still True? Auditing Facts and Detection in AI-Text Humanizers</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Genliang Zhu, Ziyuan Wang, chen-ai-researcher

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.00359) • [📄 arXiv](https://arxiv.org/abs/2611.00359) • [📥 PDF](https://arxiv.org/pdf/2611.00359)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zwang-devv/humanizer-fact-audit)

> AI "humanizers" rewrite machine-generated text so that detectors classify it as human-written. Public leaderboards score evasion and embedding similarity, but not whether checkable details survive. We check whether numbers, dates, person and organ...

</details>

<details>
<summary><b>31. From Large Language Models to Efficient System One Models via Adaptive Computation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Minh-Anh Nguyen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16321) • [📄 arXiv](https://arxiv.org/abs/2611.16321) • [📥 PDF](https://arxiv.org/pdf/2611.16321)

**💻 Code:** [⭐ Code](https://github.com/minhkks/NestOne) • [⭐ Code](https://github.com/huggingface)

> Transferring LLMs into System One decision models typically replaces the language modeling head with a decision head while retaining the full backbone, making every decision pay for all layers. Yet it remains unclear how much depth such decisions ...

</details>

<details>
<summary><b>32. EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.16728) • [📄 arXiv](https://arxiv.org/abs/2611.16728) • [📥 PDF](https://arxiv.org/pdf/2611.16728)

**💻 Code:** [⭐ Code](https://github.com/egox-dream/egox) • [⭐ Code](https://github.com/huggingface)

> Generalist manipulation policies can benefit from the complementary experience of humans, real robots, and simulated agents. However, differences in morphology, sensing, and control interfaces make it difficult to learn effectively from their comb...

</details>

<details>
<summary><b>33. A frontier machine intelligence for human neuroimaging</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.15094) • [📄 arXiv](https://arxiv.org/abs/2611.15094) • [📥 PDF](https://arxiv.org/pdf/2611.15094)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/MLNeurosurg/neuro-vl) • [⭐ Code](https://github.com/MLNeurosurg/neurobench)

> Neuro-VL is a machine intelligence for human neuroimaging designed to reason over images and clinical context and communicate its interpretations through natural-language conversation. We developed Neuro-VL using nearly 1 million neuroimaging stud...

</details>

<details>
<summary><b>34. A Cross-Scenario Survey and Analysis of Self-Evolving Multimodal Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** JingjingJiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2611.14256) • [📄 arXiv](https://arxiv.org/abs/2609.36012) • [📥 PDF](https://arxiv.org/pdf/2611.14256)

**💻 Code:** [⭐ Code](https://github.com/jingjing12110/SE-MMAgents) • [⭐ Code](https://github.com/huggingface)

> Self-Evolving Multimodal Agents

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 34 |
| 📅 Today | [`2026-11-14.json`](data/daily/2026-11-14.json) | 34 |
| 📆 This Week | [`2026-W45.json`](data/weekly/2026-W45.json) | 73 |
| 🗓️ This Month | [`2026-11.json`](data/monthly/2026-11.json) | 245 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-11-14 | 34 | [View JSON](data/daily/2026-11-14.json) |
| 📄 2026-11-13 | 6 | [View JSON](data/daily/2026-11-13.json) |
| 📄 2026-11-12 | 3 | [View JSON](data/daily/2026-11-12.json) |
| 📄 2026-11-11 | 11 | [View JSON](data/daily/2026-11-11.json) |
| 📄 2026-11-10 | 9 | [View JSON](data/daily/2026-11-10.json) |
| 📄 2026-11-09 | 10 | [View JSON](data/daily/2026-11-09.json) |
| 📄 2026-11-08 | 32 | [View JSON](data/daily/2026-11-08.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W45 | 73 | [View JSON](data/weekly/2026-W45.json) |
| 📅 2026-W44 | 133 | [View JSON](data/weekly/2026-W44.json) |
| 📅 2026-W43 | 149 | [View JSON](data/weekly/2026-W43.json) |
| 📅 2026-W42 | 204 | [View JSON](data/weekly/2026-W42.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-11 | 245 | [View JSON](data/monthly/2026-11.json) |
| 🗓️ 2026-10 | 1120 | [View JSON](data/monthly/2026-10.json) |
| 🗓️ 2026-09 | 480 | [View JSON](data/monthly/2026-09.json) |
| 🗓️ 2026-08 | 747 | [View JSON](data/monthly/2026-08.json) |
| 🗓️ 2026-07 | 366 | [View JSON](data/monthly/2026-07.json) |
| 🗓️ 2026-06 | 612 | [View JSON](data/monthly/2026-06.json) |

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
