<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-52-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2982+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">52</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">67</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">415</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2982+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 19, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. MiroThinker-1.7 & H1: Towards Heavy-Duty Research Agents via Verification</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15726) • [📄 arXiv](https://arxiv.org/abs/2603.15726) • [📥 PDF](https://arxiv.org/pdf/2603.15726)

> We present MiroThinker-1.7, a new research agent designed for complex long-horizon reasoning tasks. Building on this foundation, we further introduce MiroThinker-H1, which extends the agent with heavy-duty reasoning capabilities for more reliable ...

</details>

<details>
<summary><b>2. InCoder-32B: Code Foundation Model for Industrial Scenarios</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16790) • [📄 arXiv](https://arxiv.org/abs/2603.16790) • [📥 PDF](https://arxiv.org/pdf/2603.16790)

**💻 Code:** [⭐ Code](https://github.com/CSJianYang/Industrial-Coder)

> InCoder-32B: Code Foundation Model for Industrial Scenarios InCoder-32B (Industrial-Coder-32B) is the first 32B-parameter code foundation model purpose-built for industrial software engineering. While recent code LLMs have made impressive strides ...

</details>

<details>
<summary><b>3. Qianfan-OCR: A Unified End-to-End Model for Document Intelligence</b> ⭐ 195</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13398) • [📄 arXiv](https://arxiv.org/abs/2603.13398) • [📥 PDF](https://arxiv.org/pdf/2603.13398)

**💻 Code:** [⭐ Code](https://github.com/baidubce/Qianfan-VL)

> Is end-to-end OCR model a future research direction?

</details>

<details>
<summary><b>4. Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding</b> ⭐ 40</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13366) • [📄 arXiv](https://arxiv.org/abs/2603.13366) • [📥 PDF](https://arxiv.org/pdf/2603.13366)

**💻 Code:** [⭐ Code](https://github.com/mlrm-LEAD/mlrm-LEAD)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API SAKED: Mitigating Hallucination in Large Vision-Language Models via Stabili...

</details>

<details>
<summary><b>5. Kinema4D: Kinematic 4D World Modeling for Spatiotemporal Embodied Simulation</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16669) • [📄 arXiv](https://arxiv.org/abs/2603.16669) • [📥 PDF](https://arxiv.org/pdf/2603.16669)

**💻 Code:** [⭐ Code](https://github.com/mutianxu/Kinema4D)

> Code will be released in 1~2 weeks. https://github.com/mutianxu/Kinema4D

</details>

<details>
<summary><b>6. Demystifing Video Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16870) • [📄 arXiv](https://arxiv.org/abs/2603.16870) • [📥 PDF](https://arxiv.org/pdf/2603.16870)

> Homepage: https://www.wruisi.com/demystifying_video_reasoning YouTube Video: https://youtu.be/Gs9TPZmzo-s

</details>

<details>
<summary><b>7. WorldCam: Interactive Autoregressive 3D Gaming Worlds with Camera Pose as a Unifying Geometric Representation</b> ⭐ 69</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16871) • [📄 arXiv](https://arxiv.org/abs/2603.16871) • [📥 PDF](https://arxiv.org/pdf/2603.16871)

**💻 Code:** [⭐ Code](https://github.com/cvlab-kaist/WorldCam)

> Interesting breakdown of this paper on arXivLens: https://arxivlens.com/PaperView/Details/worldcam-interactive-autoregressive-3d-gaming-worlds-with-camera-pose-as-a-unifying-geometric-representation-5437-bf4fae19 Covers the executive summary, deta...

</details>

<details>
<summary><b>8. TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16448) • [📄 arXiv](https://arxiv.org/abs/2603.16448) • [📥 PDF](https://arxiv.org/pdf/2603.16448)

> No full schema. No pre-loaded metadata. Still stronger Text-to-SQL. We present TRUST-SQL, an autonomous agent for the Unknown Schema setting that learns to discover, verify, and reason over only the relevant schema subset. With a four-phase protoc...

</details>

<details>
<summary><b>9. Online Experiential Learning for Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16856) • [📄 arXiv](https://arxiv.org/abs/2603.16856) • [📥 PDF](https://arxiv.org/pdf/2603.16856)

**💻 Code:** [⭐ Code](https://github.com/microsoft/LMOps/tree/main/oel)

> The prevailing paradigm for improving large language models relies on offline training with human annotations or simulated environments, leaving the rich experience accumulated during real-world deployment entirely unexploited. We propose Online E...

</details>

<details>
<summary><b>10. FinToolBench: Evaluating LLM Agents for Real-World Financial Tool Use</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.08262) • [📄 arXiv](https://arxiv.org/abs/2603.08262) • [📥 PDF](https://arxiv.org/pdf/2603.08262)

**💻 Code:** [⭐ Code](https://github.com/Double-wk/FinToolBench)

> We introduce FinToolBench, a benchmark for evaluating LLM agents in realistic financial tool-use scenarios. It focuses not only on tool-calling capability, but also on finance-specific requirements such as timeliness, intent alignment, and domain ...

</details>

<details>
<summary><b>11. WiT: Waypoint Diffusion Transformers via Trajectory Conflict Navigation</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15132) • [📄 arXiv](https://arxiv.org/abs/2603.15132) • [📥 PDF](https://arxiv.org/pdf/2603.15132)

**💻 Code:** [⭐ Code](https://github.com/hainuo-wang/WiT)

> Recent flow matching models avoid VAE reconstruction bottlenecks by operating directly in pixel space, but the pixel manifold lacks semantic continuity. Optimal transport paths for different semantic endpoints overlap and intersect, causing severe...

</details>

<details>
<summary><b>12. Rethinking UMM Visual Generation: Masked Modeling for Efficient Image-Only Pre-training</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Tao Lin, Jun Xie, Peng Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16139) • [📄 arXiv](https://arxiv.org/abs/2603.16139) • [📥 PDF](https://arxiv.org/pdf/2603.16139)

**💻 Code:** [⭐ Code](https://github.com/LINs-lab/IOMM)

> IOMM (Image-Only Training for UMMs) introduces a data-efficient two-stage framework that achieves state-of-the-art multimodal generation by replacing the costly reliance on paired text-image data with a high-performance "image-only" pre-training s...

</details>

<details>
<summary><b>13. GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent</b> ⭐ 5</summary>

<br/>

**👥 Authors:** mbur, irodkin, booydar, mkairov, yurakuratov

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13875) • [📄 arXiv](https://arxiv.org/abs/2603.13875) • [📥 PDF](https://arxiv.org/pdf/2603.13875)

**💻 Code:** [⭐ Code](https://github.com/yurakuratov/gradmem)

> No abstract available.

</details>

<details>
<summary><b>14. MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Zhizhou Sha, Jianzhu Yao, Bobby Cheng, Kevin Wang, Yunfei Xie

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.09022) • [📄 arXiv](https://arxiv.org/abs/2603.09022) • [📥 PDF](https://arxiv.org/pdf/2603.09022)

**💻 Code:** [⭐ Code](https://github.com/openverse-ai/MEMO)

> X post Link: https://x.com/xiynfi1520580/status/2034316890601050565?s=20

</details>

<details>
<summary><b>15. SegviGen: Repurposing 3D Generative Model for Part Segmentation</b> ⭐ 41</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16869) • [📄 arXiv](https://arxiv.org/abs/2603.16869) • [📥 PDF](https://arxiv.org/pdf/2603.16869)

**💻 Code:** [⭐ Code](https://github.com/Nelipot-Lee/SegviGen)

> Code: https://github.com/Nelipot-Lee/SegviGen Web Page: https://fenghora.github.io/SegviGen-Page/ Demo: https://huggingface.co/spaces/fenghora/SegviGen

</details>

<details>
<summary><b>16. AgentProcessBench: Diagnosing Step-Level Process Quality in Tool-Using Agents</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.14465) • [📄 arXiv](https://arxiv.org/abs/2603.14465) • [📥 PDF](https://arxiv.org/pdf/2603.14465)

**💻 Code:** [⭐ Code](https://github.com/RUCBM/AgentProcessBench)

> 😏 𝑨𝒈𝒆𝒏𝒕𝑷𝒓𝒐𝒄𝒆𝒔𝒔𝑩𝒆𝒏𝒄𝒉 𝑨𝒗𝒂𝒊𝒍𝒂𝒃𝒍𝒆 𝑵𝒐𝒘 When utilizing Process Reward Models (PRMs) to guide Reinforcement Learning (RL) training, accurately identifying the impact or contribution of each step within a trajectory is essential for providing precise rewa...

</details>

<details>
<summary><b>17. Efficient Reasoning on the Edge</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16867) • [📄 arXiv](https://arxiv.org/abs/2603.16867) • [📥 PDF](https://arxiv.org/pdf/2603.16867)

> Proposes LoRA-based fine-tuning with supervised learning, plus budgeted reinforcement learning, dynamic adapter switching, and KV-cache sharing to enable efficient, accurate reasoning on small LLMs for on-device edge inference.

</details>

<details>
<summary><b>18. SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models</b> ⭐ 15</summary>

<br/>

**👥 Authors:** YuhuiZeng, lrf-502, bobxmuma, Jinfa, hypocritisis

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16859) • [📄 arXiv](https://arxiv.org/abs/2603.16859) • [📥 PDF](https://arxiv.org/pdf/2603.16859)

**💻 Code:** [⭐ Code](https://github.com/MAC-AutoML/SocialOmni)

> New OmniModel benchmark on social interaction. 🔗Github： github.com/MAC-AutoML/SocialOmni 🔗Dataset： huggingface.co/datasets/alexisty/SocialOmni

</details>

<details>
<summary><b>19. SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering?</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15401) • [📄 arXiv](https://arxiv.org/abs/2603.15401) • [📥 PDF](https://arxiv.org/pdf/2603.15401)

**💻 Code:** [⭐ Code](https://github.com/GeniusHTX/SWE-Skills-Bench)

> Agent skills, structured procedural knowledge packages injected at inference time, are increasingly used to augment LLM agents on software engineering tasks. However, their real utility in end-to-end development settings remains unclear. We presen...

</details>

<details>
<summary><b>20. Semi-Autonomous Formalization of the Vlasov-Maxwell-Landau Equilibrium</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15929) • [📄 arXiv](https://arxiv.org/abs/2603.15929) • [📥 PDF](https://arxiv.org/pdf/2603.15929)

**💻 Code:** [⭐ Code](https://github.com/Vilin97/Clawristotle)

> Clawristotle: Semi-Autonomous Mathematical Research Formalization of plasma equilibrium characterization theorem — achieved in 10 days by a centaur team of AI agents and a human mathematician.

</details>

<details>
<summary><b>21. SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16864) • [📄 arXiv](https://arxiv.org/abs/2603.16864) • [📥 PDF](https://arxiv.org/pdf/2603.16864)

**💻 Code:** [⭐ Code](https://github.com/taco-group/SparkVSR)

> SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation

</details>

<details>
<summary><b>22. One-Eval: An Agentic System for Automated and Traceable LLM Evaluation</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.09821) • [📄 arXiv](https://arxiv.org/abs/2603.09821) • [📥 PDF](https://arxiv.org/pdf/2603.09821)

**💻 Code:** [⭐ Code](https://github.com/OpenDCAI/One-Eval)

> No abstract available.

</details>

<details>
<summary><b>23. M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16844) • [📄 arXiv](https://arxiv.org/abs/2603.16844) • [📥 PDF](https://arxiv.org/pdf/2603.16844)

**💻 Code:** [⭐ Code](https://github.com/InternRobotics/M3)

> Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SL...

</details>

<details>
<summary><b>24. Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16189) • [📄 arXiv](https://arxiv.org/abs/2603.16189) • [📥 PDF](https://arxiv.org/pdf/2603.16189)

**💻 Code:** [⭐ Code](https://github.com/hmwang2002/CTRL-S)

> In this work, we present CTRL-S (Chain-of-Thought Reinforcement Learning for SVG), a unified framework that introduces a chain-of-thought mechanism to explicitly expose the model’s reasoning process during SVG generation. To support this structure...

</details>

<details>
<summary><b>25. Omnilingual MT: Machine Translation for 1,600 Languages</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16309) • [📄 arXiv](https://arxiv.org/abs/2603.16309) • [📥 PDF](https://arxiv.org/pdf/2603.16309)

> Tweet thread: https://x.com/b_alastruey/status/2033917464803697001

</details>

<details>
<summary><b>26. SK-Adapter: Skeleton-Based Structural Control for Native 3D Generation</b> ⭐ 10</summary>

<br/>

**👥 Authors:** Chi-Keung Tang, Shangzhe01, Supramundaner, aawangas

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.14152) • [📄 arXiv](https://arxiv.org/abs/2603.14152) • [📥 PDF](https://arxiv.org/pdf/2603.14152)

**💻 Code:** [⭐ Code](https://github.com/sk-adapter/SK-Adapter)

> We propose SK-Adapter, the first framework that unlocks precise skeletal manipulation for native 3D generation. Extensive experiments confirm that our method achieves robust structural control while preserving the geometry and texture quality of t...

</details>

<details>
<summary><b>27. From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15600) • [📄 arXiv](https://arxiv.org/abs/2603.15600) • [📥 PDF](https://arxiv.org/pdf/2603.15600)

> This paper proposes PRIMO R1, a model that leverages Reinforcement Learning to elicit the zero-shot reasoning capabilities of Video MLLMs, enabling them to estimate task progress and identify robot execution errors without the need for external re...

</details>

<details>
<summary><b>28. FlashSampling: Fast and Memory-Efficient Exact Sampling</b> ⭐ 46</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15854) • [📄 arXiv](https://arxiv.org/abs/2603.15854) • [📥 PDF](https://arxiv.org/pdf/2603.15854)

**💻 Code:** [⭐ Code](https://github.com/FlashSampling/FlashSampling)

> Sampling from a categorical distribution is mathematically simple, but in large-vocabulary decoding, it often triggers extra memory traffic and extra kernels after the LM head. We present FlashSampling, an exact sampling primitive that fuses sampl...

</details>

<details>
<summary><b>29. Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15653) • [📄 arXiv](https://arxiv.org/abs/2603.15653) • [📥 PDF](https://arxiv.org/pdf/2603.15653)

> Recursive Language Models (RLMs) have been a new direction for long context. Ever wondered what is the main driver of RLM? Is it always effective or depends on the context? How is the role of recursion or self-query tool call in this setting? In o...

</details>

<details>
<summary><b>30. MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16861) • [📄 arXiv](https://arxiv.org/abs/2603.16861) • [📥 PDF](https://arxiv.org/pdf/2603.16861)

> MolmoB0T demonstrates zero-shot real-world manipulation via large-scale procedural simulation, releasing MolmoBot-Data and open-source MolmoBot pipelines to train robust policies without real-world fine-tuning.

</details>

<details>
<summary><b>31. V-Co: A Closer Look at Visual Representation Alignment via Co-Denoising</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chu Wang, Yue Zhang, Zun Wang, Xichen Pan, Han Lin

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16792) • [📄 arXiv](https://arxiv.org/abs/2603.16792) • [📥 PDF](https://arxiv.org/pdf/2603.16792)

> No abstract available.

</details>

<details>
<summary><b>32. Mixture of Style Experts for Diverse Image Stylization</b> ⭐ 13</summary>

<br/>

**👥 Authors:** Mi Zhou, Qilong Wang, Yijia Kang, Ziheng Ouyang, Shihao Zhu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16649) • [📄 arXiv](https://arxiv.org/abs/2603.16649) • [📥 PDF](https://arxiv.org/pdf/2603.16649)

**💻 Code:** [⭐ Code](https://github.com/HVision-NKU/StyleExpert)

> Diffusion-based stylization has advanced significantly, yet existing methods are limited to color-driven transformations, neglecting complex semantics and material details. We introduce StyleExpert, a semantic-aware framework based on the Mixture ...

</details>

<details>
<summary><b>33. ViT-AdaLA: Adapting Vision Transformers with Linear Attention</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Viet Dac Lai, Seunghyun Yoon, Yifan Li, xternalz, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16063) • [📄 arXiv](https://arxiv.org/abs/2603.16063) • [📥 PDF](https://arxiv.org/pdf/2603.16063)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs ...

</details>

<details>
<summary><b>34. Sparking Scientific Creativity via LLM-Driven Interdisciplinary Inspiration</b> ⭐ 22</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12226) • [📄 arXiv](https://arxiv.org/abs/2603.12226) • [📥 PDF](https://arxiv.org/pdf/2603.12226)

**💻 Code:** [⭐ Code](https://github.com/pkargupta/idea_catalyst)

> This paper introduces Idea-Catalyst, a metacognition-driven framework for helping humans and language models move beyond within-domain brainstorming. Rather than rushing to propose answers, the system first clarifies what a research problem is rea...

</details>

<details>
<summary><b>35. Anticipatory Planning for Multimodal AI Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hao Tan, Yu Gu, Shijie Zhou, Yongyuan Liang, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16777) • [📄 arXiv](https://arxiv.org/abs/2603.16777) • [📥 PDF](https://arxiv.org/pdf/2603.16777)

> 2603.16777

</details>

<details>
<summary><b>36. Learning Human-Object Interaction for 3D Human Pose Estimation from LiDAR Point Clouds</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16343) • [📄 arXiv](https://arxiv.org/abs/2603.16343) • [📥 PDF](https://arxiv.org/pdf/2603.16343)

> Learning Human-Object Interaction for 3D Human Pose Estimation from LiDAR Point Clouds. Presented at arXiv 2026.

</details>

<details>
<summary><b>37. Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16184) • [📄 arXiv](https://arxiv.org/abs/2603.16184) • [📥 PDF](https://arxiv.org/pdf/2603.16184)

**💻 Code:** [⭐ Code](https://github.com/knoveleng/polyglot-lion)

> Efficient Multilingual ASR for Singapore

</details>

<details>
<summary><b>38. OneWorld: Taming Scene Generation with 3D Unified Representation Autoencoder</b> ⭐ 27</summary>

<br/>

**👥 Authors:** Changhu Wang, Dongdong Yu, Qihang Cao, Zhaoqing Wang, Sensen02

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16099) • [📄 arXiv](https://arxiv.org/abs/2603.16099) • [📥 PDF](https://arxiv.org/pdf/2603.16099)

**💻 Code:** [⭐ Code](https://github.com/SensenGao/OneWorld)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Mode...

</details>

<details>
<summary><b>39. MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Compute-optimal Scaling of Diffusion Language Models</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16077) • [📄 arXiv](https://arxiv.org/abs/2603.16077) • [📥 PDF](https://arxiv.org/pdf/2603.16077)

**💻 Code:** [⭐ Code](https://github.com/chen-hao-chao/mdm-prime-v2)

> No abstract available.

</details>

<details>
<summary><b>40. Residual Stream Duality in Modern Transformer Architectures</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16039) • [📄 arXiv](https://arxiv.org/abs/2603.16039) • [📥 PDF](https://arxiv.org/pdf/2603.16039)

**💻 Code:** [⭐ Code](https://github.com/yifanzhang-pro/residual-stream-duality)

> Recent work has made clear that the residual pathway is not mere optimization plumbing; it is part of the model's representational machinery. We agree, but argue that the cleanest way to organize this design space is through a two-axis view of the...

</details>

<details>
<summary><b>41. CCTU: A Benchmark for Tool Use under Complex Constraints</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15309) • [📄 arXiv](https://arxiv.org/abs/2603.15309) • [📥 PDF](https://arxiv.org/pdf/2603.15309)

**💻 Code:** [⭐ Code](https://github.com/Junjie-Ye/CCTU)

> https://github.com/Junjie-Ye/CCTU

</details>

<details>
<summary><b>42. SuperLocalMemory V3: Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Iamvarun369

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.14588) • [📄 arXiv](https://arxiv.org/abs/2603.14588) • [📥 PDF](https://arxiv.org/pdf/2603.14588)

**💻 Code:** [⭐ Code](https://github.com/qualixar/superlocalmemory)

> First AI agent memory system to report LoCoMo results mwithout any cloud dependency: 74.8% (Mode A, data stays local) and 87.7% (Mode C, full power). Three mathematical contributions, each a first in agent memory: Fisher-Rao geodesic distance for ...

</details>

<details>
<summary><b>43. ECG-Reasoning-Benchmark: A Benchmark for Evaluating Clinical Reasoning Capabilities in ECG Interpretation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.14326) • [📄 arXiv](https://arxiv.org/abs/2603.14326) • [📥 PDF](https://arxiv.org/pdf/2603.14326)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API How Well Do Multimodal Models Reason on ECG Signals? (2026) ECG-R1: Protoco...

</details>

<details>
<summary><b>44. Theoretical Foundations of Latent Posterior Factors: Formal Guarantees for Multi-Evidence Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** aaaEpalea

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15674) • [📄 arXiv](https://arxiv.org/abs/2603.15674) • [📥 PDF](https://arxiv.org/pdf/2603.15674)

> This paper presents the theoretical foundations for LPFs, introduced in a companion paper. You insights and advise are highly sought

</details>

<details>
<summary><b>45. I Know What I Don't Know: Latent Posterior Factor Models for Multi-Evidence Probabilistic Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** aaaEpalea

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15670) • [📄 arXiv](https://arxiv.org/abs/2603.15670) • [📥 PDF](https://arxiv.org/pdf/2603.15670)

**💻 Code:** [⭐ Code](https://github.com/aaaEpalea/epalea)

> This paper is introducing LPFs. Your insights and advise are deeply sought.

</details>

<details>
<summary><b>46. HistoAtlas: A Pan-Cancer Morphology Atlas Linking Histomics to Molecular Programs and Clinical Outcomes</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Pierre-Antoine Bannier

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16587) • [📄 arXiv](https://arxiv.org/abs/2603.16587) • [📥 PDF](https://arxiv.org/pdf/2603.16587)

**💻 Code:** [⭐ Code](https://github.com/HistoAtlas/HistoAtlas)

> Hi! HistoAtlas is a pan-cancer computational histopathology atlas that extracts 38 interpretable morphology features from 6,745 H&E diagnostic slides across 21 TCGA cancer types, then systematically links every feature to survival, gene expression...

</details>

<details>
<summary><b>47. ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.16060) • [📄 arXiv](https://arxiv.org/abs/2603.16060) • [📥 PDF](https://arxiv.org/pdf/2603.16060)

**💻 Code:** [⭐ Code](https://github.com/Skylanding/ARISE)

> The dominant paradigm for improving mathematical reasoning in language models relies on Reinforcement Learning with verifiable rewards. Yet existing methods treat each problem instance in isolation without leveraging the reusable strategies that e...

</details>

<details>
<summary><b>48. VAREX: A Benchmark for Multi-Modal Structured Extraction from Documents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.15118) • [📄 arXiv](https://arxiv.org/abs/2603.15118) • [📥 PDF](https://arxiv.org/pdf/2603.15118)

**💻 Code:** [⭐ Code](https://github.com/udibarzi/varex-bench)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured...

</details>

<details>
<summary><b>49. Chain-of-Trajectories: Unlocking the Intrinsic Generative Optimality of Diffusion Models via Graph-Theoretic Planning</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Xun Gong, Fei Shen, Xingpeng Zhang, Xiang Liu, redcping

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.14704) • [📄 arXiv](https://arxiv.org/abs/2603.14704) • [📥 PDF](https://arxiv.org/pdf/2603.14704)

**💻 Code:** [⭐ Code](https://github.com/UnicomAI/CoTj/blob/main/CoTj_v20260305.pdf) • [⭐ Code](https://github.com/UnicomAI/CoTj)

> CoTj (Chain-of-Trajectories: Unlocking the Intrinsic Generative Optimality of Diffusion Models via Graph-Theoretic Planning) 🧭 Description CoTj (Chain-of-Trajectories) is a graph-theoretic trajectory planning framework for diffusion models. It upg...

</details>

<details>
<summary><b>50. Measuring Primitive Accumulation: An Information-Theoretic Approach to Capitalist Enclosure in PIK2, Indonesia</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13715) • [📄 arXiv](https://arxiv.org/abs/2603.13715) • [📥 PDF](https://arxiv.org/pdf/2603.13715)

**💻 Code:** [⭐ Code](https://github.com/sandyherho/supplPIK2LULC)

> This study introduces a novel statistical-mechanical framework to quantify the kinematics and topology of capitalist land enclosure in the PIK2 coastal mega-development of Indonesia using eight years of 10-meter resolution Sentinel-2 data. By proj...

</details>

<details>
<summary><b>51. BERTology of Molecular Property Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.13627) • [📄 arXiv](https://arxiv.org/abs/2603.13627) • [📥 PDF](https://arxiv.org/pdf/2603.13627)

**💻 Code:** [⭐ Code](https://github.com/molssi-ai/bertology)

> Chemical language models (CLMs) have emerged as promising competitors to popular classical machine learning models for molecular property prediction (MPP) tasks. However, an increasing number of studies have reported inconsistent and contradictory...

</details>

<details>
<summary><b>52. Test-Time Strategies for More Efficient and Accurate Agentic RAG</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Abhinav Sharma, Zhiyang Zuo, Brian Zhang, Franck-Dernoncourt, deeptiguntur

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12396) • [📄 arXiv](https://arxiv.org/abs/2603.12396) • [📥 PDF](https://arxiv.org/pdf/2603.12396)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API SAGE: Steerable Agentic Data Generation for Deep Search with Execution Feed...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 52 |
| 📅 Today | [`2026-03-19.json`](data/daily/2026-03-19.json) | 52 |
| 📆 This Week | [`2026-W11.json`](data/weekly/2026-W11.json) | 67 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 415 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-19 | 52 | [View JSON](data/daily/2026-03-19.json) |
| 📄 2026-03-18 | 8 | [View JSON](data/daily/2026-03-18.json) |
| 📄 2026-03-17 | 1 | [View JSON](data/daily/2026-03-17.json) |
| 📄 2026-03-16 | 6 | [View JSON](data/daily/2026-03-16.json) |
| 📄 2026-03-15 | 48 | [View JSON](data/daily/2026-03-15.json) |
| 📄 2026-03-14 | 48 | [View JSON](data/daily/2026-03-14.json) |
| 📄 2026-03-13 | 4 | [View JSON](data/daily/2026-03-13.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W11 | 67 | [View JSON](data/weekly/2026-W11.json) |
| 📅 2026-W10 | 119 | [View JSON](data/weekly/2026-W10.json) |
| 📅 2026-W09 | 201 | [View JSON](data/weekly/2026-W09.json) |
| 📅 2026-W08 | 184 | [View JSON](data/weekly/2026-W08.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 415 | [View JSON](data/monthly/2026-03.json) |
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
