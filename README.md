<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-33-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5869+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📅 This Week</b><br/><font size="5">139</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">488</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5869+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 20, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Demystifying Agent Skills: Why They Work-Until They Don't</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14036) • [📄 arXiv](https://arxiv.org/abs/2608.14036) • [📥 PDF](https://arxiv.org/pdf/2608.14036)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zhiyuanjiang04/demystify-agent-skills)

> Skills have emerged as a practical and effective approach for enhancing LLM agents at inference time through structured packages of knowledge. However, existing evaluations largely measure whether skills improve aggregated task success, leaving a ...

</details>

<details>
<summary><b>2. Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17310) • [📄 arXiv](https://arxiv.org/abs/2608.17310) • [📥 PDF](https://arxiv.org/pdf/2608.17310)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zz1358m/Agentic-ESOpt)

> Will Agentic ESOpt Be the Future of Long-Horizon Agentic Fine-Tuning? Excited to share our new work on Arxiv. Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements As LLM agents become longer-horizon and more bran...

</details>

<details>
<summary><b>3. FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution</b> ⭐ 48</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16157) • [📄 arXiv](https://arxiv.org/abs/2608.16157) • [📥 PDF](https://arxiv.org/pdf/2608.16157)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/FlashML-org/FreeToken)

> Serve DeepSeek-V4-Flash on your gaming PC

</details>

<details>
<summary><b>4. ASI-Bench: At the Dawn of Artificial Superintelligence</b> ⭐ 32</summary>

<br/>

**👥 Authors:** plafle, zhouxueyang, cmwqfcmwqf, panjose, zjw49246

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17271) • [📄 arXiv](https://arxiv.org/abs/2608.17271) • [📥 PDF](https://arxiv.org/pdf/2608.17271)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/apexin-ai/ASI-Bench)

> ASI-Bench evaluates AI agents' capabilities in innovative scientific exploration and autonomous project-level research across multiple domains.

</details>

<details>
<summary><b>5. Embodied-Navigator: Point, Think, Memorize, and Align for Efficient Navigation</b> ⭐ 215</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17512) • [📄 arXiv](https://arxiv.org/abs/2608.17512) • [📥 PDF](https://arxiv.org/pdf/2608.17512)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/ZJU-OmniAI/Embodied-Omni)

> Efficient embodied navigation requires more than accurate action selection: an agent must also know when to reason, what to remember, and how to learn from environmental feedback. We introduce TAMP-Nav, where the model selects target points direct...

</details>

<details>
<summary><b>6. AVA-Encoder: Towards Agent-Native Video Representation Learning</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.12313) • [📄 arXiv](https://arxiv.org/abs/2608.12313) • [📥 PDF](https://arxiv.org/pdf/2608.12313)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/HBDYW/AVA-Encoder)

> Creative agents still lack an effective way to learn from high-quality human films, limiting their ability to produce cinematic-grade videos. A key challenge is the absence of a structured video representation that is both faithful to film content...

</details>

<details>
<summary><b>7. EDITBRIDGE: Towards Faithful and Efficient Ultra-High-Resolution Image Editing</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Zhenxiong Tan, Yubo Huang, Shijie Huang, Jiayi Song, FangtaiWu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18063) • [📄 arXiv](https://arxiv.org/abs/2608.18063) • [📥 PDF](https://arxiv.org/pdf/2608.18063)

**💻 Code:** [⭐ Code](https://github.com/songyangyifei/editbridge) • [⭐ Code](https://github.com/huggingface)

> EditBridge enables faithful and efficient ultra-high-resolution image editing by reformulating refinement as a structured data-to-data diffusion bridge, explicitly guided by the original high-resolution source. Our prior-guided block-wise sparse a...

</details>

<details>
<summary><b>8. Agent Lightning v1.0: Towards Harnessed Agentic RL</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17528) • [📄 arXiv](https://arxiv.org/abs/2608.17528) • [📥 PDF](https://arxiv.org/pdf/2608.17528)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Modern agents do not operate as standalone LLMs. They run inside agent harnesses that manage tools, context, and control flow, which makes the harness a critical component. Our original Agent Lightning work introduced a disaggregated architecture ...

</details>

<details>
<summary><b>9. DiSCO: Defending text-to-image generation through distribution-guided contrastive prompt optimization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17067) • [📄 arXiv](https://arxiv.org/abs/2608.17067) • [📥 PDF](https://arxiv.org/pdf/2608.17067)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> As text-to-image generative models advance, they raise critical safety concerns, particularly the generation of Not-Safe-For-Work (NSFW) content such as violence and nudity, further exacerbated by red-teaming adversarial attacks. Existing defenses...

</details>

<details>
<summary><b>10. V-RAE: Rethinking Video Latent Spaces for Generation</b> ⭐ 43</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13556) • [📄 arXiv](https://arxiv.org/abs/2608.13556) • [📥 PDF](https://arxiv.org/pdf/2608.13556)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/V-RAE/V-RAE)

> V-RAE: Extending Representation Autoencoders to Video Generation Following the recent Representation Autoencoder (RAE) paradigm in image generation, we explore a natural question: Can pretrained visual representations also serve as effective laten...

</details>

<details>
<summary><b>11. CoinVE-200K: A Large-Scale High-Quality Dataset for Compositional Instruction-Guided Video Editing</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17566) • [📄 arXiv](https://arxiv.org/abs/2608.17566) • [📥 PDF](https://arxiv.org/pdf/2608.17566)

**💻 Code:** [⭐ Code](https://github.com/coinve200k/CoinVE-200K) • [⭐ Code](https://github.com/huggingface)

> CoinVE-200K: A Large-Scale High-Quality Dataset for Compositional Instruction-Guided Video Editing Project Page: https://coinve200k.github.io/ Code: https://github.com/coinve200k/CoinVE-200K Dataset: https://huggingface.co/datasets/FireCRT/CoinVE-...

</details>

<details>
<summary><b>12. Dynamic Multi-Byte Prediction With Hierarchical Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15454) • [📄 arXiv](https://arxiv.org/abs/2608.15454) • [📥 PDF](https://arxiv.org/pdf/2608.15454)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/skai-research/lca-multibyte)

> Byte-level hierarchical language models (LMs) have recently emerged as a robust alternative to their popular counterparts that use subword tokenization. However, generating one byte at a time remains a bottleneck for inference speed. To address th...

</details>

<details>
<summary><b>13. MoE-ViE: Mixture of Experts Vision Encoder for Efficient Image and Video Understanding</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17402) • [📄 arXiv](https://arxiv.org/abs/2608.17402) • [📥 PDF](https://arxiv.org/pdf/2608.17402)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/facebookresearch/moe_vie)

> Tweet: https://x.com/HuggingPapers/status/2089887693799084315

</details>

<details>
<summary><b>14. Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Eric Hanchen Jiang, Yankai Chen, Yuchen Wu, Weizhi Zhang, Wei-Chieh Huang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.15008) • [📄 arXiv](https://arxiv.org/abs/2608.15008) • [📥 PDF](https://arxiv.org/pdf/2608.15008)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We present a controlled evaluation of memory substrates for memory-augmented LLM agents, covering dense and sparse retrieval, text and structural stores, hierarchical and refinement-based memories, parametric updates, and activation-compatible con...

</details>

<details>
<summary><b>15. From Corpora to Co-Evolving Capabilities: Capability-Centric Data Design for Generalist Image Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18076) • [📄 arXiv](https://arxiv.org/abs/2608.18076) • [📥 PDF](https://arxiv.org/pdf/2608.18076)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API VicEdit: Learning to Edit Videos from Visual In-Context Examples (2026) SeF...

</details>

<details>
<summary><b>16. StartupBench: Benchmarking General-Purpose Agents on Market-Validated End-to-End Workflows</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17800) • [📄 arXiv](https://arxiv.org/abs/2608.17800) • [📥 PDF](https://arxiv.org/pdf/2608.17800)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API OmniaBench: Benchmarking General AI Agents Across Diverse Scenarios (2026) ...

</details>

<details>
<summary><b>17. Energy-Guided Flow Matching</b> ⭐ 16</summary>

<br/>

**👥 Authors:** Jingling Fu, Lichen Ma, Fang Li, Yu He, ysng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.05811) • [📄 arXiv](https://arxiv.org/abs/2608.05811) • [📥 PDF](https://arxiv.org/pdf/2608.05811)

**💻 Code:** [⭐ Code](https://github.com/ysng123/EG-FM) • [⭐ Code](https://github.com/huggingface)

> Pixel-space generative models bypass lossy latent compression, yet necessitate joint learning of global structure and fine-grained details in a high-dimensional space. Standard flow matching interpolates noise toward a fixed clean-image endpoint, ...

</details>

<details>
<summary><b>18. HarnessRisk: A Lifecycle-Oriented Benchmark for Agent Harness Safety</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Sijia Liu, Xianfeng Wu, Jie Peng, Jinhao Duan, Yajing Bai

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17597) • [📄 arXiv](https://arxiv.org/abs/2608.17597) • [📥 PDF](https://arxiv.org/pdf/2608.17597)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Baiyajing/HarnessRisk)

> webpage: https://baiyajing.github.io/harness-risk/ , code: https://github.com/Baiyajing/HarnessRisk

</details>

<details>
<summary><b>19. Abra: Scaling Diffusion Image Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17286) • [📄 arXiv](https://arxiv.org/abs/2608.17286) • [📥 PDF](https://arxiv.org/pdf/2608.17286)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transform...

</details>

<details>
<summary><b>20. From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16002) • [📄 arXiv](https://arxiv.org/abs/2608.16002) • [📥 PDF](https://arxiv.org/pdf/2608.16002)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/icip-cas/RUPA)

> RUPA models LLM-agent trajectories as relational graphs and propagates uncertainty across their dependencies, yielding more accurate and earlier failure detection and better uncertainty-guided execution than existing methods across diverse agent b...

</details>

<details>
<summary><b>21. MathForm: Scaling Mathematical Autoformalization with Knowledge Retrieval and Verification-Guided Refinement</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14221) • [📄 arXiv](https://arxiv.org/abs/2608.14221) • [📥 PDF](https://arxiv.org/pdf/2608.14221)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/OpenBMB/MathForm)

> Autoformalization is commonly framed as translating natural-language mathematical statements into machine-verifiable formal languages such as Lean 4. However, faithful formalization requires more than translation. Models must map mathematical conc...

</details>

<details>
<summary><b>22. Security Assessment of DeepSeek Harness with A.I.G: Evaluating Resistance to Indirect Prompt Injection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16393) • [📄 arXiv](https://arxiv.org/abs/2608.16393) • [📥 PDF](https://arxiv.org/pdf/2608.16393)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We assess indirect prompt injection in DeepSeek Harness (DSH), using AI-Infra-Guard (A.I.G) to construct tests, deliver controlled taint, execute DSH, collect traces, and judge outcomes. The study covers 14,560 controlled executions over 16 indire...

</details>

<details>
<summary><b>23. Cross-Model Memory Transfer via Target-Side Reader Adaptation</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17050) • [📄 arXiv](https://arxiv.org/abs/2608.17050) • [📥 PDF](https://arxiv.org/pdf/2608.17050)

**💻 Code:** [⭐ Code](https://github.com/OLAResearch/XMemTransfer) • [⭐ Code](https://github.com/huggingface)

> We study cross-model frozen-memory transfer, showing that target-side readers can extract and align representations across model architectures and tokenizers.

</details>

<details>
<summary><b>24. Personalized Auto-Research: Towards a True AI Co-Scientist</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nesreen K. Ahmed, Yu Wang, Hongjie Chen, Bo Ni, Franck-Dernoncourt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14881) • [📄 arXiv](https://arxiv.org/abs/2608.14881) • [📥 PDF](https://arxiv.org/pdf/2608.14881)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Diversifying Personalized Research Ideation against AI-Induced Homogenizati...

</details>

<details>
<summary><b>25. GS-Voxel: Fitting-Free Structured Latents for Large-Scale 3DGS Generation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17988) • [📄 arXiv](https://arxiv.org/abs/2608.17988) • [📥 PDF](https://arxiv.org/pdf/2608.17988)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> GS-Voxel converts pre-optimized 3D Gaussian Splatting reconstructions into sparse, structured latents without additional per-scene fitting. By combining a factorized geometry-and-attribute VAE with image-conditioned flow matching, it enables 3DGS ...

</details>

<details>
<summary><b>26. LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17393) • [📄 arXiv](https://arxiv.org/abs/2608.17393) • [📥 PDF](https://arxiv.org/pdf/2608.17393)

**💻 Code:** [⭐ Code](https://github.com/LegoX/Lego-RL) • [⭐ Code](https://github.com/huggingface)

> We are happy to announce Lego-RL: an open-source framework for harness-native reinforcement learning of coding agents . All code, data, and models are available on GitHub and Hugging Face.

</details>

<details>
<summary><b>27. aDSL: Agentic 3D Creation via Joint Agent-Program Design</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Baoquan Chen, Heng-Yi Wei, Jia-Qi He, Si-Tong Wei, Rui-Huan Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17975) • [📄 arXiv](https://arxiv.org/abs/2608.17975) • [📥 PDF](https://arxiv.org/pdf/2608.17975)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API WorldClaw: Agentic 3D Open-World Generation at Scale (2026) SimWorlds: A Mu...

</details>

<details>
<summary><b>28. The Problem Is the Problem: Towards Scalable Mathematical Discovery</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16977) • [📄 arXiv](https://arxiv.org/abs/2608.16977) • [📥 PDF](https://arxiv.org/pdf/2608.16977)

**💻 Code:** [⭐ Code](https://github.com/zeyu-zheng/FAR) • [⭐ Code](https://github.com/huggingface)

> A step toward exponentiating mathematical discovery

</details>

<details>
<summary><b>29. Unifying Graph Neural Networks Through a Common Layer Equation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hongjie Chen, Bo Ni, Sai Karthik Navuluru, Franck-Dernoncourt, siddhartha047

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16097) • [📄 arXiv](https://arxiv.org/abs/2608.16097) • [📥 PDF](https://arxiv.org/pdf/2608.16097)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Graph Neural Networks Applications Across Domains: All Insights You Need (2...

</details>

<details>
<summary><b>30. PTXBench: Benchmark and Adapt LLMs for GPU Kernel Optimization with Architecture-specific PTX</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.17379) • [📄 arXiv](https://arxiv.org/abs/2608.17379) • [📥 PDF](https://arxiv.org/pdf/2608.17379)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zhang677/PTXBench)

> We introduce PTXBench, a benchmark for evaluating and adapting large language models (LLMs) to use architecture-specific PTX for GPU kernel optimization. PTXBench measures functional correctness, whether selected target instructions execute at run...

</details>

<details>
<summary><b>31. PixRestore: Unified Image Restoration via Pixel Diffusion Transformer</b> ⭐ 32</summary>

<br/>

**👥 Authors:** Qiaosi Yi, Jixin Zhao, Xiangtao Kong, Rongyuan Wu, Lingchen Sun

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.16793) • [📄 arXiv](https://arxiv.org/abs/2608.16793) • [📥 PDF](https://arxiv.org/pdf/2608.16793)

**💻 Code:** [⭐ Code](https://github.com/csslc/PixRestore) • [⭐ Code](https://github.com/huggingface)

> TL;DR: A VAE-free pixel-space DiT for unified image restoration. It applies flow matching to patchified RGB, uses adaptive DINO guidance, and is finetuned for one-step inference with ~50M parameters.

</details>

<details>
<summary><b>32. CardioState-JEPA: Delay-Aware Cross-Modal Learning of a Shared Cardiac Representation</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Jun Hu, Pan Zhou, Bin Zhu, Hung Manh Pham, hamzashafiq

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.12944) • [📄 arXiv](https://arxiv.org/abs/2608.12944) • [📥 PDF](https://arxiv.org/pdf/2608.12944)

**💻 Code:** [⭐ Code](https://github.com/hamzashafiq28/CardioState-Jepa) • [⭐ Code](https://github.com/huggingface)

> CardioState-JEPA introduces a delay-aware cross-modal JEPA for learning a shared cardiac representation across ECG, PPG, and PCG using a single encoder. The key idea is to account explicitly for physiological timing offsets between electrical, mec...

</details>

<details>
<summary><b>33. FACET: Preserving Source Intent and Executable State in Terminal Task Synthesis</b> ⭐ 2</summary>

<br/>

**👥 Authors:** Ziao Zhang, Shiting Huang, Qisheng Su, Zun Wang, Kou Shi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.18580) • [📄 arXiv](https://arxiv.org/abs/2608.18580) • [📥 PDF](https://arxiv.org/pdf/2608.18580)

**💻 Code:** [⭐ Code](https://github.com/StoKou/FACET-Terminal) • [⭐ Code](https://github.com/huggingface)

> Training terminal agents requires scalable executable supervision, yet synthesizing high-quality terminal tasks remains challenging. Each task couples an instruction, an initialized environment, a reference solution, and an executable verifier; if...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 33 |
| 📅 Today | [`2026-08-20.json`](data/daily/2026-08-20.json) | 33 |
| 📆 This Week | [`2026-W33.json`](data/weekly/2026-W33.json) | 139 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 488 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-20 | 33 | [View JSON](data/daily/2026-08-20.json) |
| 📄 2026-08-19 | 42 | [View JSON](data/daily/2026-08-19.json) |
| 📄 2026-08-18 | 32 | [View JSON](data/daily/2026-08-18.json) |
| 📄 2026-08-17 | 32 | [View JSON](data/daily/2026-08-17.json) |
| 📄 2026-08-16 | 32 | [View JSON](data/daily/2026-08-16.json) |
| 📄 2026-08-15 | 32 | [View JSON](data/daily/2026-08-15.json) |
| 📄 2026-08-14 | 1 | [View JSON](data/daily/2026-08-14.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W33 | 139 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |
| 📅 2026-W30 | 112 | [View JSON](data/weekly/2026-W30.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 488 | [View JSON](data/monthly/2026-08.json) |
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
