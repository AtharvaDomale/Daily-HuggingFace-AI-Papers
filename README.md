<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-47-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-1921+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">47</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">357</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">402</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">1921+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** February 08, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty</b> ⭐ 14</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22027) • [📄 arXiv](https://arxiv.org/abs/2601.22027) • [📥 PDF](https://arxiv.org/pdf/2601.22027)

**💻 Code:** [⭐ Code](https://github.com/CAR-bench/car-bench)

> Why is this gap widening? Frontier models like Claude-Opus-4.6 are crushing base task performance (80%), but hallucination resistance (48%) and disambiguation (46%) lag far behind. What's preventing models from learning when to say 'I need more in...

</details>

<details>
<summary><b>2. Spider-Sense: Intrinsic Risk Sensing for Efficient Agent Defense with Hierarchical Adaptive Screening</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05386) • [📄 arXiv](https://arxiv.org/abs/2602.05386) • [📥 PDF](https://arxiv.org/pdf/2602.05386)

**💻 Code:** [⭐ Code](https://github.com/aifinlab/Spider-Sense)

> Endow AI Agents with "Spider-Sense"! Spider-Sense: Pioneering Intrinsic Risk Sensing, Reducing Defense Delay to 8.3%

</details>

<details>
<summary><b>3. Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Variation in RLVR</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Zhixiong Zeng, Siqi Yang, Peng Shi, Youyang Yin, liufanfanlff

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05261) • [📄 arXiv](https://arxiv.org/abs/2504.06037) • [📥 PDF](https://arxiv.org/pdf/2602.05261)

**💻 Code:** [⭐ Code](https://github.com/murphy4122/LUSPO)

> We introduce Length-Unbiased Sequence Policy Optimization (LUSPO), a novel reinforcement learning algorithm for training large language models. LUSPO consistently outperforms GRPO and GSPO on both dense small-scale models and large-scale MoE model...

</details>

<details>
<summary><b>4. MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents</b> ⭐ 24</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02474) • [📄 arXiv](https://arxiv.org/abs/2602.02474) • [📥 PDF](https://arxiv.org/pdf/2602.02474)

**💻 Code:** [⭐ Code](https://github.com/ViktorAxelsen/MemSkill)

> Most Large Language Model (LLM) agent memory systems rely on a small set of static, hand-designed operations for extracting memory. These fixed procedures hard-code human priors about what to store and how to revise memory, making them rigid under...

</details>

<details>
<summary><b>5. Context Forcing: Consistent Autoregressive Video Generation with Long Context</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06028) • [📄 arXiv](https://arxiv.org/abs/2602.06028) • [📥 PDF](https://arxiv.org/pdf/2602.06028)

**💻 Code:** [⭐ Code](https://github.com/TIGER-AI-Lab/Context-Forcing)

> project page: https://chenshuo20.github.io/Context_Forcing/ code: https://github.com/TIGER-AI-Lab/Context-Forcing

</details>

<details>
<summary><b>6. DFlash: Block Diffusion for Flash Speculative Decoding</b> ⭐ 489</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06036) • [📄 arXiv](https://arxiv.org/abs/2602.06036) • [📥 PDF](https://arxiv.org/pdf/2602.06036)

**💻 Code:** [⭐ Code](https://github.com/z-lab/dflash)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference (2026)...

</details>

<details>
<summary><b>7. RISE-Video: Can Video Generators Decode Implicit World Rules?</b> ⭐ 20</summary>

<br/>

**👥 Authors:** Zicheng Zhang, Xiangyu Zhao, Shibei Meng, Shuran Ma, Mingxin Liu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05986) • [📄 arXiv](https://arxiv.org/abs/2602.05986) • [📥 PDF](https://arxiv.org/pdf/2602.05986)

**💻 Code:** [⭐ Code](https://github.com/VisionXLab/Rise-Video)

> Despite strong visual realism, we find that current text-image-to-video models frequently fail to respect implicit world rules when generating complex scenarios. We introduce RISE-Video to systematically evaluate reasoning fidelity in video genera...

</details>

<details>
<summary><b>8. Accurate Failure Prediction in Agents Does Not Imply Effective Failure Prevention</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03338) • [📄 arXiv](https://arxiv.org/abs/2602.03338) • [📥 PDF](https://arxiv.org/pdf/2602.03338)

> Accurate LLM critics do not guarantee safe intervention: like relentless contradiction, they can derail trajectories that would have succeeded. Despite strong offline accuracy (AUROC 0.94), a binary critic causes outcomes ranging from a 26-pp coll...

</details>

<details>
<summary><b>9. ProAct: Agentic Lookahead in Interactive Environments</b> ⭐ 10</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05327) • [📄 arXiv](https://arxiv.org/abs/2602.05327) • [📥 PDF](https://arxiv.org/pdf/2602.05327)

**💻 Code:** [⭐ Code](https://github.com/GreatX3/ProAct)

> ProAct trains LLM-based agents to perform accurate lookahead planning in interactive environments via Grounded LookAhead Distillation and a Monte-Carlo Critic, improving long-horizon decision accuracy.

</details>

<details>
<summary><b>10. Privileged Information Distillation for Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04942) • [📄 arXiv](https://arxiv.org/abs/2602.04942) • [📥 PDF](https://arxiv.org/pdf/2602.04942)

> Training-time privileged information (PI) can enable language models to succeed on tasks they would otherwise fail, making it a powerful tool for reinforcement learning in hard, long-horizon settings. However, transferring capabilities learned wit...

</details>

<details>
<summary><b>11. Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05885) • [📄 arXiv](https://arxiv.org/abs/2602.05885) • [📥 PDF](https://arxiv.org/pdf/2602.05885)

**💻 Code:** [⭐ Code](https://github.com/hkust-nlp/KernelGYM)

> High-quality kernel is critical for scalable AI systems, and enabling LLMs to generate such code would advance AI development. However, training LLMs for this task requires sufficient data, a robust environment, and the process is often vulnerable...

</details>

<details>
<summary><b>12. InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiaohan Fei, Xialin He, Morteza Ziyadi, Samuel Schulter, xusirui

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06035) • [📄 arXiv](https://arxiv.org/abs/2602.06035) • [📥 PDF](https://arxiv.org/pdf/2602.06035)

> Distillation reconstructs motor skills, while RL fine-tuning interpolates and consolidates the latent space into a coherent skill manifold for versatile whole-body loco-manipulation.

</details>

<details>
<summary><b>13. Reinforcement World Model Learning for LLM-based Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05842) • [📄 arXiv](https://arxiv.org/abs/2602.05842) • [📥 PDF](https://arxiv.org/pdf/2602.05842)

> Large language models (LLMs) have achieved strong performance in language-centric tasks. However, in agentic settings, LLMs often struggle to anticipate action consequences and adapt to environment dynamics, highlighting the need for world-modelin...

</details>

<details>
<summary><b>14. Semantic Search over 9 Million Mathematical Theorems</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05216) • [📄 arXiv](https://arxiv.org/abs/2602.05216) • [📥 PDF](https://arxiv.org/pdf/2602.05216)

> Mathematicians and math prover agents need fast and efficient theorem search. We release Theorem Search over all of arXiv, the Stacks Project, and six other sources. Our search is 2x more accurate than frontier LLMs, with only 4 second latency. Fe...

</details>

<details>
<summary><b>15. Retrieval-Infused Reasoning Sandbox: A Benchmark for Decoupling Retrieval and Reasoning Capabilities</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21937) • [📄 arXiv](https://arxiv.org/abs/2601.21937) • [📥 PDF](https://arxiv.org/pdf/2601.21937)

**💻 Code:** [⭐ Code](https://github.com/Retrieval-Infused-Reasoning-Sandbox/Retrieval-Infused-Reasoning-Sandbox)

> Despite strong performance on existing benchmarks, it remains unclear whether large language models can reason over genuinely novel scientific information. Most evaluations score end-to-end RAG pipelines, where reasoning is confounded with retriev...

</details>

<details>
<summary><b>16. SocialVeil: Probing Social Intelligence of Language Agents under Communication Barriers</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Tal August, Haofei Yu, Chongrui Ye, Pengda Wang, Keyang Xuan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05115) • [📄 arXiv](https://arxiv.org/abs/2602.05115) • [📥 PDF](https://arxiv.org/pdf/2602.05115)

> Interesting work, Keyang

</details>

<details>
<summary><b>17. Steering LLMs via Scalable Interactive Oversight</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04210) • [📄 arXiv](https://arxiv.org/abs/2602.04210) • [📥 PDF](https://arxiv.org/pdf/2602.04210)

> As Large Language Models increasingly automate complex, long-horizon tasks such as \emph{vibe coding}, a supervision gap has emerged. While models excel at execution, users often struggle to guide them effectively due to insufficient domain expert...

</details>

<details>
<summary><b>18. Grounding and Enhancing Informativeness and Utility in Dataset Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21296) • [📄 arXiv](https://arxiv.org/abs/2601.21296) • [📥 PDF](https://arxiv.org/pdf/2601.21296)

> Dataset Distillation (DD) seeks to create a compact dataset from a large, real-world dataset. While recent methods often rely on heuristic approaches to balance efficiency and quality, the fundamental relationship between original and synthetic da...

</details>

<details>
<summary><b>19. Reinforced Attention Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04884) • [📄 arXiv](https://arxiv.org/abs/2602.04884) • [📥 PDF](https://arxiv.org/pdf/2602.04884)

> No abstract available.

</details>

<details>
<summary><b>20. Thinking in Frames: How Visual Context and Test-Time Scaling Empower Video Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.21037) • [📄 arXiv](https://arxiv.org/abs/2601.21037) • [📥 PDF](https://arxiv.org/pdf/2601.21037)

> Project Page: https://thinking-in-frames.github.io/

</details>

<details>
<summary><b>21. LatentMem: Customizing Latent Memory for Multi-Agent Systems</b> ⭐ 17</summary>

<br/>

**👥 Authors:** Zefeng He, Yafu Li, Xiangyuan Xue, Guibin Zhang, Muxin Fu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.03036) • [📄 arXiv](https://arxiv.org/abs/2602.03036) • [📥 PDF](https://arxiv.org/pdf/2602.03036)

**💻 Code:** [⭐ Code](https://github.com/KANABOON1/LatentMem)

> LatentMem: Customizing Latent Memory for Multi-Agent Systems

</details>

<details>
<summary><b>22. SwimBird: Eliciting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06040) • [📄 arXiv](https://arxiv.org/abs/2602.06040) • [📥 PDF](https://arxiv.org/pdf/2602.06040)

**💻 Code:** [⭐ Code](https://github.com/Accio-Lab/SwimBird)

> Project Page: https://accio-lab.github.io/SwimBird Github Repo: https://github.com/Accio-Lab/SwimBird HuggingFace: https://huggingface.co/datasets/Accio-Lab/SwimBird-SFT-92K

</details>

<details>
<summary><b>23. SAGE: Benchmarking and Improving Retrieval for Deep Research Agents</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Chen Zhao, Arman Cohan, Canyu Zhang, yilunzhao, HughieHu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05975) • [📄 arXiv](https://arxiv.org/abs/2602.05975) • [📥 PDF](https://arxiv.org/pdf/2602.05975)

> Deep research agents have emerged as powerful systems for addressing complex queries. Meanwhile, LLM-based retrievers have demonstrated strong capability in following instructions or reasoning. This raises a critical question: can LLM-based retrie...

</details>

<details>
<summary><b>24. Towards Reducible Uncertainty Modeling for Reliable Large Language Model Agents</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05073) • [📄 arXiv](https://arxiv.org/abs/2602.05073) • [📥 PDF](https://arxiv.org/pdf/2602.05073)

> A foundation and perspective for uncertainty quantification of LLM agents.

</details>

<details>
<summary><b>25. BABE: Biology Arena BEnchmark</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05857) • [📄 arXiv](https://arxiv.org/abs/2602.05857) • [📥 PDF](https://arxiv.org/pdf/2602.05857)

> BABE is a biology benchmark that evaluates AI models' experimental reasoning across papers and real studies, stressing cross-scale causal inference and practical scientific reasoning.

</details>

<details>
<summary><b>26. V-Retrver: Evidence-Driven Agentic Reasoning for Universal Multimodal Retrieval</b> ⭐ 19</summary>

<br/>

**👥 Authors:** Zeyu Zhang, Xi Xiao, Dezhao SU, Chaoyang Wang, Dongyang Chen

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06034) • [📄 arXiv](https://arxiv.org/abs/2602.06034) • [📥 PDF](https://arxiv.org/pdf/2602.06034)

**💻 Code:** [⭐ Code](https://github.com/chendy25/V-Retrver)

> Multimodal Large Language Models (MLLMs) have recently been applied to universal multimodal retrieval, where Chain-of-Thought (CoT) reasoning improves candidate reranking. However, existing approaches remain largely language-driven, relying on sta...

</details>

<details>
<summary><b>27. Multi-Task GRPO: Reliable LLM Reasoning Across Tasks</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhiyong Wang, Sangwoong Yoon, Matthieu Zimmer, Xiaotong Ji, Shyam Sundhar Ramesh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05547) • [📄 arXiv](https://arxiv.org/abs/2602.05547) • [📥 PDF](https://arxiv.org/pdf/2602.05547)

> We propose a novel technique for multitask learning with GRPO without forgetting about worst-case tasks.

</details>

<details>
<summary><b>28. DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02016) • [📄 arXiv](https://arxiv.org/abs/2602.02016) • [📥 PDF](https://arxiv.org/pdf/2602.02016)

**💻 Code:** [⭐ Code](https://github.com/IST-DASLab/DASH)

> We propose DASH ( D istributed A ccelerated SH ampoo), a faster and more accurate version of Distributed Shampoo. To make it faster, we stack the blocks extracted from the preconditioners to obtain a 3D tensor, which are inverted efficiently using...

</details>

<details>
<summary><b>29. Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training</b> ⭐ 9</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05933) • [📄 arXiv](https://arxiv.org/abs/2602.05933) • [📥 PDF](https://arxiv.org/pdf/2602.05933)

**💻 Code:** [⭐ Code](https://github.com/horizon-rl/OpenKimi)

> Reproduce Kimi K1.5/K2 RL algorithm and theoretically understand PMD as regularization in LLM post training

</details>

<details>
<summary><b>30. CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs</b> ⭐ 5</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05258) • [📄 arXiv](https://arxiv.org/abs/2602.05258) • [📥 PDF](https://arxiv.org/pdf/2602.05258)

**💻 Code:** [⭐ Code](https://github.com/hrlics/CoPE)

> [Paper] [HF checkpoints] CoPE is a plug-and-play enhancement of RoPE that softly clips the unstable low-frequency components, delivering consistent gains both within the training context and during long-context extrapoaltion . With a simple yet ef...

</details>

<details>
<summary><b>31. Late-to-Early Training: LET LLMs Learn Earlier, So Faster and Better</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05393) • [📄 arXiv](https://arxiv.org/abs/2602.05393) • [📥 PDF](https://arxiv.org/pdf/2602.05393)

> arXivLens breakdown of this paper 👉 https://arxivlens.com/PaperView/Details/late-to-early-training-let-llms-learn-earlier-so-faster-and-better-8353-cc1b8d02 Executive Summary Detailed Breakdown Practical Applications

</details>

<details>
<summary><b>32. Breaking the Static Graph: Context-Aware Traversal for Robust Retrieval-Augmented Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Qintian Guo, Yingli Zhou, Boyu Ruan, Fangyuan Zhang, Jimlkh

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.01965) • [📄 arXiv](https://arxiv.org/abs/2602.01965) • [📥 PDF](https://arxiv.org/pdf/2602.01965)

> This paper addresses a fundamental limitation in graph-based retrieval-augmented generation (RAG) systems, which we characterize as the "Static Graph Fallacy." While recent methods have successfully utilized Knowledge Graphs (KGs) to capture multi...

</details>

<details>
<summary><b>33. Pathwise Test-Time Correction for Autoregressive Long Video Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhe Gao, Haiyu Zhang, Guiyu Zhang, Zixuan Duan, Xunzhi Xiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05871) • [📄 arXiv](https://arxiv.org/abs/2602.05871) • [📥 PDF](https://arxiv.org/pdf/2602.05871)

> Introduces Test-Time Correction (TTC) to stabilize long autoregressive video generation by anchoring intermediate states to the initial frame, enabling longer sequences with minimal overhead.

</details>

<details>
<summary><b>34. Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Mi-Yen Yeh, Pin-Yu Chen, Ching-Yun Ko, Yu-Ang Lee

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04998) • [📄 arXiv](https://arxiv.org/abs/2602.04998) • [📥 PDF](https://arxiv.org/pdf/2602.04998)

> Motivated by the increasing number of LoRA variants and the insufficient hyperparameter tuning in many studies, in this work, we conduct a systematic re-evaluation of five LoRA PEFT methods under a unified evaluation protocol. Based on the compreh...

</details>

<details>
<summary><b>35. Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02393) • [📄 arXiv](https://arxiv.org/abs/2602.02393) • [📥 PDF](https://arxiv.org/pdf/2602.02393)

> Project Page: https://rq-wu.github.io/projects/infinite-world/index.html

</details>

<details>
<summary><b>36. A Unified Framework for Rethinking Policy Divergence Measures in GRPO</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yanning Dai, Simon Sinong Zhan, Yuhui Wang, Qingyuan Wu, zczlsde

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05494) • [📄 arXiv](https://arxiv.org/abs/2602.05494) • [📥 PDF](https://arxiv.org/pdf/2602.05494)

> A Unified Framework for Rethinking Policy Divergence Measures in GRPO

</details>

<details>
<summary><b>37. Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05023) • [📄 arXiv](https://arxiv.org/abs/2602.05023) • [📥 PDF](https://arxiv.org/pdf/2602.05023)

**💻 Code:** [⭐ Code](https://github.com/99starman/VLM-GeoPrivacyBench)

> Our data and code are available at https://github.com/99starman/VLM-GeoPrivacyBench .

</details>

<details>
<summary><b>38. Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Shen Ren, Ruihao Gong, Yumeng Shi, Harahan, mack-williams

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04789) • [📄 arXiv](https://arxiv.org/abs/2602.04789) • [📥 PDF](https://arxiv.org/pdf/2602.04789)

> Light Forcing introduces a novel sparse attention mechanism for autoregressive video generation that improves efficiency while maintaining quality through chunk-aware growth and hierarchical sparse attention strategies.

</details>

<details>
<summary><b>39. Beyond Fixed Frames: Dynamic Character-Aligned Speech Tokenization</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.23174) • [📄 arXiv](https://arxiv.org/abs/2601.23174) • [📥 PDF](https://arxiv.org/pdf/2601.23174)

**💻 Code:** [⭐ Code](https://github.com/lucadellalib/dycast)

> Variable-frame-rate speech tokenization

</details>

<details>
<summary><b>40. Failing to Explore: Language Models on Interactive Tasks</b> ⭐ 9</summary>

<br/>

**👥 Authors:** Zahra Sodagar, Keivan Rezaei, yizecheng, ckodser, AghaTizi

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.22345) • [📄 arXiv](https://arxiv.org/abs/2601.22345) • [📥 PDF](https://arxiv.org/pdf/2601.22345)

**💻 Code:** [⭐ Code](https://github.com/mahdi-jfri/explore-exploit-bench)

> LLMs fail to explore.

</details>

<details>
<summary><b>41. FastVMT: Eliminating Redundancy in Video Motion Transfer</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Hongyu Liu, Mingzhe Zheng, Tianhao Ren, Zhikai Wang, Yue Ma

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05551) • [📄 arXiv](https://arxiv.org/abs/2602.05551) • [📥 PDF](https://arxiv.org/pdf/2602.05551)

> FastVMT speeds up video motion transfer by masking local attention and reusing gradients to remove motion and gradient redundancy, achieving 3.43x speedup without quality loss.

</details>

<details>
<summary><b>42. Fast-SAM3D: 3Dfy Anything in Images but Faster</b> ⭐ 34</summary>

<br/>

**👥 Authors:** Haotong Qin, Chuanguang Yang, Zhiliang Chen, Mingqiang Wu, Weilun Feng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.05293) • [📄 arXiv](https://arxiv.org/abs/2602.05293) • [📥 PDF](https://arxiv.org/pdf/2602.05293)

**💻 Code:** [⭐ Code](https://github.com/wlfeng0509/Fast-SAM3D)

> No abstract available.

</details>

<details>
<summary><b>43. UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization</b> ⭐ 111</summary>

<br/>

**👥 Authors:** Xixin Wu, Songxiang Liu, Dading Chong, Yuanyuan Wang, Dongchao Yang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04683) • [📄 arXiv](https://arxiv.org/abs/2602.04683) • [📥 PDF](https://arxiv.org/pdf/2602.04683)

**💻 Code:** [⭐ Code](https://github.com/yangdongchao/UniAudio2)

> Audio Foundation Models

</details>

<details>
<summary><b>44. Adaptive 1D Video Diffusion Autoencoder</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Xiao Yang, Shuai Wang, Xian Liu, Minxuan Lin, Yao Teng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.04220) • [📄 arXiv](https://arxiv.org/abs/2602.04220) • [📥 PDF](https://arxiv.org/pdf/2602.04220)

> Adaptive 1D Video Diffusion Autoencoder Recent video generation models largely rely on video autoencoders that compress pixel-space videos into latent representations. However, existing video autoencoders suffer from three major limitations: (1) f...

</details>

<details>
<summary><b>45. PhysicsAgentABM: Physics-Guided Generative Agent-Based Modeling</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.06030) • [📄 arXiv](https://arxiv.org/abs/2602.06030) • [📥 PDF](https://arxiv.org/pdf/2602.06030)

> Large language model (LLM)-based multi-agent systems enable expressive agent reasoning but are expensive to scale and poorly calibrated for timestep-aligned state-transition simulation, while classical agent-based models (ABMs) offer interpretabil...

</details>

<details>
<summary><b>46. Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing</b> ⭐ 7</summary>

<br/>

**👥 Authors:** Jun Zhang, Ruihao Gong, Shihao Bai, Lingkun Long, Harahan

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.02159) • [📄 arXiv](https://arxiv.org/abs/2602.02159) • [📥 PDF](https://arxiv.org/pdf/2602.02159)

**💻 Code:** [⭐ Code](https://github.com/Longxmas/Focus-dLLM)

> Diffusion Large Language Models (dLLMs) deliver strong long-context processing capability in a non-autoregressive decoding paradigm. However, the considerable computational cost of bidirectional full attention limits the inference efficiency. Alth...

</details>

<details>
<summary><b>47. Assessing Domain-Level Susceptibility to Emergent Misalignment from Narrow Finetuning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Deepesh Suranjandass, Polina Petrova, Reshma Ashok, Mugilan Arulvanan, abhishek9909

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.00298) • [📄 arXiv](https://arxiv.org/abs/2602.00298) • [📥 PDF](https://arxiv.org/pdf/2602.00298)

**💻 Code:** [⭐ Code](https://github.com/clarifying-EM/model-organisms-for-EM) • [⭐ Code](https://github.com/emergent-misalignment/emergent-misalignment)

> Overview We investigate how fine-tuning LLMs on domain-specific "insecure" datasets can induce emergent misalignment —where narrow harmful objectives generalize into broadly misaligned behavior on unrelated tasks. Our study spans 11 diverse domain...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 47 |
| 📅 Today | [`2026-02-08.json`](data/daily/2026-02-08.json) | 47 |
| 📆 This Week | [`2026-W05.json`](data/weekly/2026-W05.json) | 357 |
| 🗓️ This Month | [`2026-02.json`](data/monthly/2026-02.json) | 402 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-02-08 | 47 | [View JSON](data/daily/2026-02-08.json) |
| 📄 2026-02-07 | 47 | [View JSON](data/daily/2026-02-07.json) |
| 📄 2026-02-06 | 52 | [View JSON](data/daily/2026-02-06.json) |
| 📄 2026-02-05 | 53 | [View JSON](data/daily/2026-02-05.json) |
| 📄 2026-02-04 | 73 | [View JSON](data/daily/2026-02-04.json) |
| 📄 2026-02-03 | 40 | [View JSON](data/daily/2026-02-03.json) |
| 📄 2026-02-02 | 45 | [View JSON](data/daily/2026-02-02.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W05 | 357 | [View JSON](data/weekly/2026-W05.json) |
| 📅 2026-W04 | 214 | [View JSON](data/weekly/2026-W04.json) |
| 📅 2026-W03 | 183 | [View JSON](data/weekly/2026-W03.json) |
| 📅 2026-W02 | 232 | [View JSON](data/weekly/2026-W02.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-02 | 402 | [View JSON](data/monthly/2026-02.json) |
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
