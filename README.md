<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-32-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-5794+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">32</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">64</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">413</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">5794+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** August 18, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Can We Defend Against AI-Generated Video Attacks on Real-World Crisis Events? A Systematic Evaluation of Detectors, Generators and Social Dissemination</b> ⭐ 36</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14391) • [📄 arXiv](https://arxiv.org/abs/2608.14391) • [📥 PDF](https://arxiv.org/pdf/2608.14391)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/24029100313/RA-Bench)

> A Systematic Evaluation of Detectors, Generators and Social Dissemination

</details>

<details>
<summary><b>2. Self-Supervised Visual On-Policy Distillation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14144) • [📄 arXiv](https://arxiv.org/abs/2608.14144) • [📥 PDF](https://arxiv.org/pdf/2608.14144)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Visual on-policy distillation relies heavily on an informative teacher-student asymmetry, through either a larger, stronger teacher or privileged supervision, such as reference answers or ground-truth regions of interest. This raises a fundamental...

</details>

<details>
<summary><b>3. Beyond Final Scores: A Systematic Evaluation of Agents for Long-Horizon AI Research and Development</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13417) • [📄 arXiv](https://arxiv.org/abs/2608.13417) • [📥 PDF](https://arxiv.org/pdf/2608.13417)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> 🚀 We are excited to share Beyond Final Scores: A Systematic Evaluation of Agents for Long-Horizon AI Research and Development . 🔍 As AI agents increasingly tackle long-horizon research and engineering tasks, evaluating them by final scores alone i...

</details>

<details>
<summary><b>4. Intern-S2-Mobius: Foundation Model with Decoupled Knowledge and Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jiaye Ge, Ning Ding, Jifeng Ding, Kai Chen, Youbang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14290) • [📄 arXiv](https://arxiv.org/abs/2608.14290) • [📥 PDF](https://arxiv.org/pdf/2608.14290)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> اصنع فيديو تعليمي لل اطفال لمده 5 دقائق عن الأرقام

</details>

<details>
<summary><b>5. Apodex Discovery: Reality Benchmarks and Environments for Evaluating and Building Discoverative Artificial Intelligence</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.11341) • [📄 arXiv](https://arxiv.org/abs/2608.11341) • [📥 PDF](https://arxiv.org/pdf/2608.11341)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Apodex Discovery is a new paradigm for discoverative AI, which is different from generative AI, with a focus on targeting real-world problems that do not have ground truth answers. More can be learned at our website https://discovery.apodex.com/

</details>

<details>
<summary><b>6. SimpleOPD: Simple Tokenizer-Agnostic On-Policy Distillation for Long-Context Reasoning</b> ⭐ 8</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14277) • [📄 arXiv](https://arxiv.org/abs/2608.14277) • [📥 PDF](https://arxiv.org/pdf/2608.14277)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hhnqqq/SimpleOPD)

> Experiments on both same-family and different-family student models, including Qwen3, Qwen3.5, Intern-S2, GLM-4.7, Gemma-4, show consistent gains in mathematical reasoning, especially natural-language math proving. Notably, Intern-S2-Preview impro...

</details>

<details>
<summary><b>7. Marionette: Predicting World States, Rendering Geometry, Painting Appearance</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kaipeng Zhang, Qiang Li, Chuanhao Li, Zian Meng, Lixsp11

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14530) • [📄 arXiv](https://arxiv.org/abs/2608.14530) • [📥 PDF](https://arxiv.org/pdf/2608.14530)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> https://youtu.be/bLLtwXVcqEc

</details>

<details>
<summary><b>8. DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13517) • [📄 arXiv](https://arxiv.org/abs/2608.13517) • [📥 PDF](https://arxiv.org/pdf/2608.13517)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/schneiderkamplab/HRM-Text)

> Current large language model development relies on massive, often non-permissible datasets, creating a high barrier for researchers committed to open-source and ethically sourced data. We introduce Mimir, a 1-billion-parameter language model based...

</details>

<details>
<summary><b>9. MobileMem: Learning from a Year of Mobile Experiences</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13606) • [📄 arXiv](https://arxiv.org/abs/2608.13606) • [📥 PDF](https://arxiv.org/pdf/2608.13606)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zjunlp/MobileMem)

> MobileMem, a benchmark and framework for studying on-device long-term memory, grounded in a year-scale collection of mobile experiences.

</details>

<details>
<summary><b>10. HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13555) • [📄 arXiv](https://arxiv.org/abs/2608.13555) • [📥 PDF](https://arxiv.org/pdf/2608.13555)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/GalaxyGeneralRobotics/HumanTracker)

> We introduce HumanTracker, a comprehensive and human-aligned benchmark for humanoid motion tracking, together with HumanScore, a preference-aligned metric that better reflects human judgments. Our goal is to move beyond simple kinematic errors and...

</details>

<details>
<summary><b>11. CPI-Bench: A Comprehensive,Practical and Intelligent Benchmark for Real-World Image Editing</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14546) • [📄 arXiv](https://arxiv.org/abs/2608.14546) • [📥 PDF](https://arxiv.org/pdf/2608.14546)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/zqyzzz/CPI-benchmark)

> CPI-Bench: A Comprehensive,Practical and Intelligent Benchmark for Real-World Image Editing

</details>

<details>
<summary><b>12. Latent On-Policy Self-Distillation</b> ⭐ 31</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13040) • [📄 arXiv](https://arxiv.org/abs/2608.13040) • [📥 PDF](https://arxiv.org/pdf/2608.13040)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/bingreeky/LOPD)

> Github: https://github.com/bingreeky/LOPD Enabling agents to learn from experience and internalize it into their policy has become a central problem in self-evolving AI. On-policy self-distillation (OPSD) offers an effective pathway by using a pri...

</details>

<details>
<summary><b>13. Modular Cognitive Architecture Emerges in Large Language Models</b> ⭐ 69</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13567) • [📄 arXiv](https://arxiv.org/abs/2608.13567) • [📥 PDF](https://arxiv.org/pdf/2608.13567)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Pengrui-Han/LLM_Modularity)

> The human brain is strikingly modular, with distinct networks for language, formal reasoning, social reasoning, and physical reasoning. Is this a fundamental principle of how intelligent systems are built, or an accident of biological evolution? I...

</details>

<details>
<summary><b>14. PRM-as-a-Judge 1.5: A Toolkit for Robot Process Assessment</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Ruike Chen, Yanqing Shen, Yuyang Liu, yuheng2000, BubbleQ

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14284) • [📄 arXiv](https://arxiv.org/abs/2608.14284) • [📥 PDF](https://arxiv.org/pdf/2608.14284)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/YuyangLiu2003/PRM-as-a-Judge)

> An easy-to-use toolkit for fine-grained robot process assessment, with progress-aware metrics, progress judge benchmarking, and visualization tools.

</details>

<details>
<summary><b>15. Claim-Level Reliability Assessment for Efficient Test-Time Reasoning</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.11994) • [📄 arXiv](https://arxiv.org/abs/2608.11994) • [📥 PDF](https://arxiv.org/pdf/2608.11994)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/WeiboAI/CLR)

> We introduce CLR (Claim-Level Reliability Assessment) , a training-free test-time scaling method previously used in VibeThinker-3B . CLR is built on a simple idea that exploits the asymmetry between solving and falsification to improve reasoning r...

</details>

<details>
<summary><b>16. Second Thought: Reasoning in Parallel as LLM Agents Act and Observe</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13667) • [📄 arXiv](https://arxiv.org/abs/2608.13667) • [📥 PDF](https://arxiv.org/pdf/2608.13667)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> A new idle window for test-time scaling 😃

</details>

<details>
<summary><b>17. Multimodal Model Diffing for Feature Discovery and Control</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.09928) • [📄 arXiv](https://arxiv.org/abs/2608.09928) • [📥 PDF](https://arxiv.org/pdf/2608.09928)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/hunarbatra/MMDiff)

> We introduce MMDiff, a multimodal model-diffing pipeline to discover task-specific features in MLLMs and enable targeted feature-level control. 🌟 MMDiff diffs a base-LM SAE against a multimodal SAE to isolate vision-adapted features, making it eas...

</details>

<details>
<summary><b>18. Forecast Collapse in Time-Series Foundation Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14106) • [📄 arXiv](https://arxiv.org/abs/2608.14106) • [📥 PDF](https://arxiv.org/pdf/2608.14106)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> We find a surprising failure mode, forecast collapse , of Time Series Foundation Models (TSFMs). Despite their strong performance across general time-series benchmarks, we observe that TSFMs can generate overly smooth or near-constant forecasts th...

</details>

<details>
<summary><b>19. LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13545) • [📄 arXiv](https://arxiv.org/abs/2608.13545) • [📥 PDF](https://arxiv.org/pdf/2608.13545)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> What happens when an LLM never sees material beyond fifth grade? The 5B LittleLearner model trained from scratch on LittleCurriculum, a corpus restricted to K–5 material, answers this question and allows to test the effect of post-training, prompt...

</details>

<details>
<summary><b>20. Scaling Domain Data Repetition in LLM Pretraining</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14071) • [📄 arXiv](https://arxiv.org/abs/2608.14071) • [📥 PDF](https://arxiv.org/pdf/2608.14071)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

<details>
<summary><b>21. Dion3: Full-Stack Orthogonal Updates</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.11612) • [📄 arXiv](https://arxiv.org/abs/2608.11612) • [📥 PDF](https://arxiv.org/pdf/2608.11612)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/microsoft/dion)

> Dion3 is a full stack optimization of muon-like updates yielding up to an observed factor of 6 reduction in optimizer time and an extra centinat of performance with code available at https://github.com/microsoft/dion . The time improvements includ...

</details>

<details>
<summary><b>22. Agents Catching Agents: Shortcut Cascades and Benchmark Gaming in Clinical Multi-Agent Systems</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.03744) • [📄 arXiv](https://arxiv.org/abs/2608.03744) • [📥 PDF](https://arxiv.org/pdf/2608.03744)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/criticaldata/benchmaxxing)

> Multi-agent medical AI fails socially, not visually: agents that resist shortcuts alone adopt wrong answers 38% of the time under peer agreement. Independent referees detect it at 77–88% precision.

</details>

<details>
<summary><b>23. Verifier-Induced Support Reshaping in On-Policy Optimization</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.00220) • [📄 arXiv](https://arxiv.org/abs/2608.00220) • [📥 PDF](https://arxiv.org/pdf/2608.00220)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sylvain-wei/verifier-induced-support-reshaping)

> What happens when on-policy RLVR improves the objective in front of it, but makes successful behavior for the next objective harder to sample? In this paper, we study this effect across mathematical reasoning and constrained instruction following....

</details>

<details>
<summary><b>24. UniProbe: A Learnable Token-Level Hallucination Detector for Large VLMs using Multi-Structural Internal Representations</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.10835) • [📄 arXiv](https://arxiv.org/abs/2608.10835) • [📥 PDF](https://arxiv.org/pdf/2608.10835)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Large Vision-Language Models (LVLMs) achieve impressive visual reasoning and dialogue capabilities, yet frequently hallucinate content unsupported by the visual input. Effective mitigation requires token-level localization, enabling targeted inter...

</details>

<details>
<summary><b>25. A Pathway to General-Purpose Scientific AI: Multimodal Comprehension of Scientific Images</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14075) • [📄 arXiv](https://arxiv.org/abs/2608.14075) • [📥 PDF](https://arxiv.org/pdf/2608.14075)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/sciknoworg/sci-imageminer)

> Scientific figures are more than illustrations—they encode the evidence behind scientific claims. This perspective asks what it would take for multimodal AI to move beyond extracting information from figures toward scientific conceptual understand...

</details>

<details>
<summary><b>26. Nanbeige4.2-3B on Apple Silicon: Fixing Deployment Bugs and Decreasing Looped Transformer Memory Overhead</b> ⭐ 1</summary>

<br/>

**👥 Authors:** John T. Halloran

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13987) • [📄 arXiv](https://arxiv.org/abs/2608.13987) • [📥 PDF](https://arxiv.org/pdf/2608.13987)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/johnhalloran321/Nanbeige4.2-3B-mps-fix)

> Nanbeige4.2-3B (a 3B "Looped Transformer" agentic model) has five bugs that block it from loading/running correctly via Hugging Face transformers out of the box, plus a memory ceiling and a system-prompt regression that surface once it does run. W...

</details>

<details>
<summary><b>27. Amplified Does Not Mean Predictive: Reasoning Behaviors in Thinking Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13760) • [📄 arXiv](https://arxiv.org/abs/2608.13760) • [📥 PDF](https://arxiv.org/pdf/2608.13760)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Amplified Does Not Mean Predictive: Reasoning Behaviors in Thinking Models Thinking models produce more deliberative traces, but this added deliberation is not concentrated in the behaviors most associated with reasoning correctness. The largest l...

</details>

<details>
<summary><b>28. Who Speaks Matters: Authority-Aware Multi-View RAG over Italian Parliamentary Proceedings</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.13410) • [📄 arXiv](https://arxiv.org/abs/2608.13410) • [📥 PDF](https://arxiv.org/pdf/2608.13410)

**💻 Code:** [⭐ Code](https://github.com/huggingface) • [⭐ Code](https://github.com/Emeierkeio/ParliamentRAG)

> ParliamentRAG is a RAG system over the official records of the Italian Chamber of Deputies (19th legislature): 46.8k speech transcripts, 17.3k roll-call votes and 6.9M individual ballots in a Neo4j knowledge graph. Speaker authority is estimated p...

</details>

<details>
<summary><b>29. UNMASK: Discovering and Causally Verifying Spurious Shortcuts in Text Classifiers</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.09209) • [📄 arXiv](https://arxiv.org/abs/2608.09209) • [📥 PDF](https://arxiv.org/pdf/2608.09209)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Language Models are very good at exploiting shortcuts. Existing approaches either require manual specification of the spurious feature or automate discovery only partially. The gap between dataset-level correlation and model-level exploitation is ...

</details>

<details>
<summary><b>30. SPARGen: Unifying Spatial Perception and Reasoning through Native Multimodal Generation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Kewang Deng, Siyi Xie, Jianhua Li, Jinsheng Quan, shixuanke

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.14138) • [📄 arXiv](https://arxiv.org/abs/2608.14138) • [📥 PDF](https://arxiv.org/pdf/2608.14138)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> Interesting approach. SPARGen’s main strength seems to be bringing 3D reconstruction, dense correspondence, and spatial reasoning into one unified generative framework instead of treating them as separate tasks. Sharing representations across thes...

</details>

<details>
<summary><b>31. Is this Citation on Point?</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.12571) • [📄 arXiv](https://arxiv.org/abs/2608.12571) • [📥 PDF](https://arxiv.org/pdf/2608.12571)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> In 2023, a New York judge sanctioned two attorneys in Mata v. Avianca for filing a brief with hallucinated citations generated by ChatGPT. Such failures are largely caught by database lookups; the harder problem is detecting citations that point t...

</details>

<details>
<summary><b>32. Generation as Auxiliary Supervision: Enhancing Visual Understanding at Zero Inference Overhead via Decoupled Embedding Prediction</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2608.12209) • [📄 arXiv](https://arxiv.org/abs/2608.12209) • [📥 PDF](https://arxiv.org/pdf/2608.12209)

**💻 Code:** [⭐ Code](https://github.com/huggingface)

> No abstract available.

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 32 |
| 📅 Today | [`2026-08-18.json`](data/daily/2026-08-18.json) | 32 |
| 📆 This Week | [`2026-W33.json`](data/weekly/2026-W33.json) | 64 |
| 🗓️ This Month | [`2026-08.json`](data/monthly/2026-08.json) | 413 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-08-18 | 32 | [View JSON](data/daily/2026-08-18.json) |
| 📄 2026-08-17 | 32 | [View JSON](data/daily/2026-08-17.json) |
| 📄 2026-08-16 | 32 | [View JSON](data/daily/2026-08-16.json) |
| 📄 2026-08-15 | 32 | [View JSON](data/daily/2026-08-15.json) |
| 📄 2026-08-14 | 1 | [View JSON](data/daily/2026-08-14.json) |
| 📄 2026-08-13 | 3 | [View JSON](data/daily/2026-08-13.json) |
| 📄 2026-08-12 | 38 | [View JSON](data/daily/2026-08-12.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W33 | 64 | [View JSON](data/weekly/2026-W33.json) |
| 📅 2026-W32 | 171 | [View JSON](data/weekly/2026-W32.json) |
| 📅 2026-W31 | 102 | [View JSON](data/weekly/2026-W31.json) |
| 📅 2026-W30 | 112 | [View JSON](data/weekly/2026-W30.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-08 | 413 | [View JSON](data/monthly/2026-08.json) |
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
