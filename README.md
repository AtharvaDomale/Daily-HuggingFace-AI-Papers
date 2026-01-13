<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-30-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-998+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">30</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">63</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">260</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">998+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** January 13, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Thinking with Map: Reinforced Parallel Map-Augmented Agent for Geolocalization</b> ⭐ 107</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05432) • [📄 arXiv](https://arxiv.org/abs/2601.05432) • [📥 PDF](https://arxiv.org/pdf/2601.05432)

**💻 Code:** [⭐ Code](https://github.com/AMAP-ML/Thinking-with-Map)

> Demo video

</details>

<details>
<summary><b>2. MMFormalizer: Multimodal Autoformalization in the Wild</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Huajian Xin, Hui Shen, Yunta Hsieh, Qi Han, Jing Xiong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03017) • [📄 arXiv](https://arxiv.org/abs/2601.03017) • [📥 PDF](https://arxiv.org/pdf/2601.03017)

> Autoformalization, which translates natural language mathematics into formal statements to enable machine reasoning, faces fundamental challenges in the wild due to the multimodal nature of the physical world, where physics requires inferring hidd...

</details>

<details>
<summary><b>3. CaricatureGS: Exaggerating 3D Gaussian Splatting Faces With Gaussian Curvature</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.03319) • [📄 arXiv](https://arxiv.org/abs/2601.03319) • [📥 PDF](https://arxiv.org/pdf/2601.03319)

> Project Page: https://c4ricaturegs.github.io/

</details>

<details>
<summary><b>4. The Molecular Structure of Thought: Mapping the Topology of Long Chain-of-Thought Reasoning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06002) • [📄 arXiv](https://arxiv.org/abs/2601.06002) • [📥 PDF](https://arxiv.org/pdf/2601.06002)

> Glad to share our recent exploratory project: 🧪 Title: The Molecular Structure of Thought: Mapping the Topology of Long Chain-of-Thought Reasoning 🌐 arXiv: 2601.06002 ​ 🧐 Why revisit Long CoT? Recent work often focuses on “making CoT longer,” but ...

</details>

<details>
<summary><b>5. Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards</b> ⭐ 15</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.06021) • [📄 arXiv](https://arxiv.org/abs/2601.06021) • [📥 PDF](https://arxiv.org/pdf/2601.06021)

**💻 Code:** [⭐ Code](https://github.com/THUDM/CaRR)

> Code: https://github.com/THUDM/CaRR Data: https://huggingface.co/datasets/THU-KEG/CaRR-DeepDive

</details>

<details>
<summary><b>6. EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhicheng Dou, Yutao Zhu, Haofei Chang, Xiaoshuai Song, dongguanting

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05808) • [📄 arXiv](https://arxiv.org/abs/2601.05808) • [📥 PDF](https://arxiv.org/pdf/2601.05808)

**💻 Code:** [⭐ Code](https://github.com/RUC-NLPIR/EnvScaler)

> Code: https://github.com/RUC-NLPIR/EnvScaler Data & Model: https://huggingface.co/collections/XXHStudyHard/envscaler

</details>

<details>
<summary><b>7. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking</b> ⭐ 620</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04720) • [📄 arXiv](https://www.arxiv.org/abs/2601.04720) • [📥 PDF](https://arxiv.org/pdf/2601.04720)

**💻 Code:** [⭐ Code](https://github.com/QwenLM/Qwen3-VL-Embedding)

> 🚀 Introducing Qwen3-VL-Embedding and Qwen3-VL-Reranker – advancing the state of the art in multimodal retrieval and cross-modal understanding! ✨ Highlights: ✅ Built upon the robust Qwen3-VL foundation model ✅ Processes text, images, screenshots, v...

</details>

<details>
<summary><b>8. Can We Predict Before Executing Machine Learning Agents?</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05930) • [📄 arXiv](https://arxiv.org/abs/2601.05930) • [📥 PDF](https://arxiv.org/pdf/2601.05930)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/predict-before-execute)

> We replace slow trial-and-error in scientific agents with learned execution prediction, enabling FOREAGENT to think before it runs and achieve 6× faster and better scientific discovery.

</details>

<details>
<summary><b>9. An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Nikolaos Aletras, Constantinos Karouzos, XingweiT

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05882) • [📄 arXiv](https://arxiv.org/abs/2601.05882) • [📥 PDF](https://arxiv.org/pdf/2601.05882)

**💻 Code:** [⭐ Code](https://github.com/ckarouzos/prefadap)

> Our paper presents a systematic study of preference-optimization under domain shift. We compare five popular alignment objectives and various adaptation strategies from source to target, including target-domain supervised fine-tuning and pseudo-la...

</details>

<details>
<summary><b>10. AgentOCR: Reimagining Agent History via Optical Self-Compression</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04786) • [📄 arXiv](https://arxiv.org/abs/2601.04786) • [📥 PDF](https://arxiv.org/pdf/2601.04786)

> We’re introducing AgentOCR, a new way to scale LLM agents by reimagining long interaction histories as compact rendered images, leveraging the higher information density of visual tokens to curb exploding context costs. To make long-horizon rollou...

</details>

<details>
<summary><b>11. VideoAR: Autoregressive Video Generation via Next-Frame & Scale Prediction</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Yu Sun, Shuohuan Wang, Xiaoxiong Liu, Longbin Ji, sjy1203

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05966) • [📄 arXiv](https://arxiv.org/abs/2601.05966) • [📥 PDF](https://arxiv.org/pdf/2601.05966)

> VideoAR presents a scalable autoregressive video-generation framework that combines next-frame scale prediction with a 3D multi-scale tokenizer to improve temporal coherence and efficiency.

</details>

<details>
<summary><b>12. Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05905) • [📄 arXiv](https://arxiv.org/abs/2601.05905) • [📥 PDF](https://arxiv.org/pdf/2601.05905)

**💻 Code:** [⭐ Code](https://github.com/zjunlp/belief)

> We show that many LLM “beliefs” that look confident collapse under small context changes, and propose Neighbor-Consistency Belief (NCB) and Structure-Aware Training to measure and train models to keep their knowledge stable and robust under such i...

</details>

<details>
<summary><b>13. Goal Force: Teaching Video Models To Accomplish Physics-Conditioned Goals</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Evan Luo, Zitian Tang, Yinghua Zhou, dakshces, nate-gillman

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05848) • [📄 arXiv](https://arxiv.org/abs/2601.05848) • [📥 PDF](https://arxiv.org/pdf/2601.05848)

> Goal Force trains a physics-grounded video model to follow explicit force-directed goals, achieving zero-shot planning in real-world tasks by implicit neural physics simulation.

</details>

<details>
<summary><b>14. Orient Anything V2: Unifying Orientation and Rotation Understanding</b> ⭐ 82</summary>

<br/>

**👥 Authors:** Tianyu Pang, Jialei Wang, Jiayang Xu, Zehan Wang, Viglong

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05573) • [📄 arXiv](https://arxiv.org/abs/2601.05573) • [📥 PDF](https://arxiv.org/pdf/2601.05573)

**💻 Code:** [⭐ Code](https://github.com/SpatialVision/Orient-Anything-V2)

> Code: https://github.com/SpatialVision/Orient-Anything-V2 Demo Space: https://huggingface.co/spaces/Viglong/Orient-Anything-V2

</details>

<details>
<summary><b>15. Same Claim, Different Judgment: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05403) • [📄 arXiv](https://arxiv.org/abs/2601.05403) • [📥 PDF](https://arxiv.org/pdf/2601.05403)

> Large language models (LLMs) have been widely applied across various domains of finance. Since their training data are largely derived from human-authored corpora, LLMs may inherit a range of human biases. Behavioral biases can lead to instability...

</details>

<details>
<summary><b>16. AnyDepth: Depth Estimation Made Easy</b> ⭐ 63</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.02760) • [📄 arXiv](https://arxiv.org/abs/2601.02760) • [📥 PDF](https://arxiv.org/pdf/2601.02760)

**💻 Code:** [⭐ Code](https://github.com/AIGeeksGroup/AnyDepth)

> https://aigeeksgroup.github.io/AnyDepth

</details>

<details>
<summary><b>17. SmartSearch: Process Reward-Guided Query Refinement for Search Agents</b> ⭐ 11</summary>

<br/>

**👥 Authors:** Guanting Dong, douzc, vvv111222

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04888) • [📄 arXiv](https://arxiv.org/abs/2601.04888) • [📥 PDF](https://arxiv.org/pdf/2601.04888)

**💻 Code:** [⭐ Code](https://github.com/MYVAE/SmartSearch?tab=readme-ov-file)

> Some of the observations founded are :- i. Dual Level Credit Assessment This mechanism provides a comprehensive evaluation of query quality through both rule-based and model-based assessments. It allows for fine-grained supervision, helping to ide...

</details>

<details>
<summary><b>18. Over-Searching in Search-Augmented Large Language Models</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05503) • [📄 arXiv](https://arxiv.org/abs/2601.05503) • [📥 PDF](https://arxiv.org/pdf/2601.05503)

**💻 Code:** [⭐ Code](https://github.com/ruoyuxie/OversearchQA)

> Systematically analyzes over-search in search-augmented LLMs, showing when retrieval helps or hurts, introducing Tokens Per Correctness and mitigation strategies.

</details>

<details>
<summary><b>19. DR-LoRA: Dynamic Rank LoRA for Mixture-of-Experts Adaptation</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Linqi Song, Huacan Wang, Ronghao Chen, Guanzhi Deng, liboaccn

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04823) • [📄 arXiv](https://arxiv.org/abs/2601.04823) • [📥 PDF](https://arxiv.org/pdf/2601.04823)

> Mixture-of-Experts (MoE) has become a prominent paradigm for scaling Large Language Models (LLMs). Parameter-efficient fine-tuning (PEFT), such as LoRA, is widely adopted to adapt pretrained MoE LLMs to downstream tasks. However, existing approach...

</details>

<details>
<summary><b>20. Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhicheng Dou, Yutao Zhu, Jiejun Tan, Jiongnan Liu, namespace-ERI

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04726) • [📄 arXiv](https://arxiv.org/abs/2601.04726) • [📥 PDF](https://arxiv.org/pdf/2601.04726)

> Large language models (LLMs) are increasingly deployed as intelligent agents that reason, plan, and interact with their environments. To effectively scale to long-horizon scenarios, a key capability for such agents is a memory mechanism that can r...

</details>

<details>
<summary><b>21. GenCtrl -- A Formal Controllability Toolkit for Generative Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05637) • [📄 arXiv](https://arxiv.org/abs/2601.05637) • [📥 PDF](https://arxiv.org/pdf/2601.05637)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API A Reason-then-Describe Instruction Interpreter for Controllable Video Gener...

</details>

<details>
<summary><b>22. TCAndon-Router: Adaptive Reasoning Router for Multi-Agent Collaboration</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04544) • [📄 arXiv](https://arxiv.org/abs/2601.04544) • [📥 PDF](https://arxiv.org/pdf/2601.04544)

**💻 Code:** [⭐ Code](https://github.com/kyegomez/awesome-multi-agent-papers) • [⭐ Code](https://github.com/Tencent/TCAndon-Router)

> Code: https://github.com/Tencent/TCAndon-Router

</details>

<details>
<summary><b>23. Distilling Feedback into Memory-as-a-Tool</b> ⭐ 1</summary>

<br/>

**👥 Authors:** vicgalle

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05960) • [📄 arXiv](https://arxiv.org/abs/2601.05960) • [📥 PDF](https://arxiv.org/pdf/2601.05960)

**💻 Code:** [⭐ Code](https://github.com/vicgalle/feedback-memory-as-a-tool)

> Code: https://github.com/vicgalle/feedback-memory-as-a-tool Data: https://huggingface.co/datasets/vicgalle/rubric-feedback-bench

</details>

<details>
<summary><b>24. TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05899) • [📄 arXiv](https://arxiv.org/abs/2601.05899) • [📥 PDF](https://arxiv.org/pdf/2601.05899)

**💻 Code:** [⭐ Code](https://github.com/tb6147877/TowerMind)

> Some of the observations are :- i. TowerMind is a lightweight RTS-style benchmark for LLM agents It introduces a tower defense based environment that preserves long term planning and decision making challenges of RTS games, while requiring very lo...

</details>

<details>
<summary><b>25. Router-Suggest: Dynamic Routing for Multimodal Auto-Completion in Visually-Grounded Dialogs</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05851) • [📄 arXiv](https://arxiv.org/abs/2601.05851) • [📥 PDF](https://arxiv.org/pdf/2601.05851)

> Real-time multimodal auto-completion is essential for digital assistants, chatbots, design tools, and healthcare consultations, where user inputs rely on shared visual context. We introduce Multimodal Auto-Completion (MAC), a task that predicts up...

</details>

<details>
<summary><b>26. ViTNT-FIQA: Training-Free Face Image Quality Assessment with Vision Transformers</b> ⭐ 3</summary>

<br/>

**👥 Authors:** Marco Huber, Jan Niklas Kolf, Tahar Chettaoui, Eduarda Caldeira, gurayozgur

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05741) • [📄 arXiv](https://arxiv.org/abs/2601.05741) • [📥 PDF](https://arxiv.org/pdf/2601.05741)

**💻 Code:** [⭐ Code](https://github.com/gurayozgur/ViTNT-FIQA)

> https://github.com/gurayozgur/ViTNT-FIQA

</details>

<details>
<summary><b>27. IIB-LPO: Latent Policy Optimization via Iterative Information Bottleneck</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhuoyue Chen, Long Li, Yue Zhu, Hongchen Luo, Huilin Deng

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05870) • [📄 arXiv](https://arxiv.org/abs/2601.05870) • [📥 PDF](https://arxiv.org/pdf/2601.05870)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API ReLaX: Reasoning with Latent Exploration for Large Reasoning Models (2025) ...

</details>

<details>
<summary><b>28. Afri-MCQA: Multimodal Cultural Question Answering for African Languages</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Jesujoba Oluwadara Alabi, Israel Abebe Azime, Emilio Villa-Cueva, Srija Anand, Atnafu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05699) • [📄 arXiv](https://arxiv.org/abs/2601.05699) • [📥 PDF](https://arxiv.org/pdf/2601.05699)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG (2025) R...

</details>

<details>
<summary><b>29. The Persona Paradox: Medical Personas as Behavioral Priors in Clinical Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.05376) • [📄 arXiv](https://arxiv.org/abs/2601.05376) • [📥 PDF](https://arxiv.org/pdf/2601.05376)

> This paper investigates how "persona conditioning" (e.g., instructing an LLM to act as a specific medical professional) impacts clinical decision-making. The authors challenge the assumption that assigning a medical persona consistently improves a...

</details>

<details>
<summary><b>30. Legal Alignment for Safe and Ethical AI</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Rishi Bommasani, Cullen O'Keefe, Jack Boeglin, Nicholas Caputo, Noam Kolt

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2601.04175) • [📄 arXiv](https://arxiv.org/abs/2601.04175) • [📥 PDF](https://arxiv.org/pdf/2601.04175)

> Field-defining paper by researchers from Stanford, MIT, Harvard, Oxford, Princeton, and other leading institutions. More details at: https://www.legal-alignment.ai/

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 30 |
| 📅 Today | [`2026-01-13.json`](data/daily/2026-01-13.json) | 30 |
| 📆 This Week | [`2026-W02.json`](data/weekly/2026-W02.json) | 63 |
| 🗓️ This Month | [`2026-01.json`](data/monthly/2026-01.json) | 260 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-01-13 | 30 | [View JSON](data/daily/2026-01-13.json) |
| 📄 2026-01-12 | 33 | [View JSON](data/daily/2026-01-12.json) |
| 📄 2026-01-11 | 33 | [View JSON](data/daily/2026-01-11.json) |
| 📄 2026-01-10 | 33 | [View JSON](data/daily/2026-01-10.json) |
| 📄 2026-01-09 | 20 | [View JSON](data/daily/2026-01-09.json) |
| 📄 2026-01-08 | 26 | [View JSON](data/daily/2026-01-08.json) |
| 📄 2026-01-07 | 24 | [View JSON](data/daily/2026-01-07.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W02 | 63 | [View JSON](data/weekly/2026-W02.json) |
| 📅 2026-W01 | 156 | [View JSON](data/weekly/2026-W01.json) |
| 📅 2026-W00 | 41 | [View JSON](data/weekly/2026-W00.json) |
| 📅 2025-W52 | 52 | [View JSON](data/weekly/2025-W52.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-01 | 260 | [View JSON](data/monthly/2026-01.json) |
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
