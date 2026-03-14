<div align="center">

# 🤖 Daily HuggingFace AI Papers

### 📊 Your Automated AI Research Companion

> **Never miss groundbreaking AI research again!** Get daily updates on the hottest papers from HuggingFace, automatically curated and archived. Perfect for researchers, ML engineers, and AI enthusiasts. 🔥

[![Update Daily](https://img.shields.io/badge/Update-Daily-brightgreen?style=for-the-badge&logo=github-actions)](https://github.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/actions)
[![Papers Today](https://img.shields.io/badge/Papers%20Today-48-blue?style=for-the-badge&logo=arxiv)](data/latest.json)
[![Total Papers](https://img.shields.io/badge/Total%20Papers-2867+-orange?style=for-the-badge&logo=academia)](data/)
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
<td align="center"><b>📄 Today</b><br/><font size="5">48</font><br/>papers</td>
<td align="center"><b>📅 This Week</b><br/><font size="5">71</font><br/>papers</td>
<td align="center"><b>📆 This Month</b><br/><font size="5">300</font><br/>papers</td>
<td align="center"><b>🗄️ Total Archive</b><br/><font size="5">2867+</font><br/>papers</td>
</tr>
</table>

**Last Updated:** March 14, 2026

---

## 🔥 Today's Trending Papers

> Latest AI research papers from HuggingFace Papers, updated daily

<details>
<summary><b>1. Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training</b> ⭐ 71</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12255) • [📄 arXiv](https://arxiv.org/abs/2603.12255) • [📥 PDF](https://arxiv.org/pdf/2603.12255)

**💻 Code:** [⭐ Code](https://github.com/THU-SI/Spatial-TTT)

> code: https://github.com/THU-SI/Spatial-TTT

</details>

<details>
<summary><b>2. Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections</b> ⭐ 12</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12180) • [📄 arXiv](https://arxiv.org/abs/2603.12180) • [📥 PDF](https://arxiv.org/pdf/2603.12180)

**💻 Code:** [⭐ Code](https://github.com/OxRML/MADQA)

> Dataset: https://huggingface.co/datasets/OxRML/MADQA Baseline code: https://github.com/OxRML/MADQA Leaderboard: https://huggingface.co/spaces/Snowflake/MADQA-Leaderboard

</details>

<details>
<summary><b>3. IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12201) • [📄 arXiv](https://arxiv.org/abs/2603.12201) • [📥 PDF](https://arxiv.org/pdf/2603.12201)

**💻 Code:** [⭐ Code](https://github.com/THUDM/IndexCache)

> We introduce IndexCache, a method to accelerate DeepSeek Sparse Attention (DSA) by exploiting cross-layer redundancy in token selection. In DSA, a lightweight indexer at each layer selects top-k tokens for sparse attention, but adjacent layers pro...

</details>

<details>
<summary><b>4. Video-Based Reward Modeling for Computer-Use Agents</b> ⭐ 6</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.10178) • [📄 arXiv](https://arxiv.org/abs/2603.10178) • [📥 PDF](https://arxiv.org/pdf/2603.10178)

**💻 Code:** [⭐ Code](https://github.com/limenlp/ExeVRM)

> Computer-using agents (CUAs) are becoming increasingly capable; however, it remains difficult to scale evaluation of whether a trajectory truly fulfills a user instruction. In this work, we study reward modeling from execution video: a sequence of...

</details>

<details>
<summary><b>5. DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12257) • [📄 arXiv](https://arxiv.org/abs/2603.12257) • [📥 PDF](https://arxiv.org/pdf/2603.12257)

> Project Page: https://dreamvideo-omni.github.io/

</details>

<details>
<summary><b>6. Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation</b> ⭐ 23</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12247) • [📄 arXiv](https://arxiv.org/abs/2603.12247) • [📥 PDF](https://arxiv.org/pdf/2603.12247)

**💻 Code:** [⭐ Code](https://github.com/VisionXLab/FIRM-Reward)

> Project Page: https://firm-reward.github.io/ Code: https://github.com/VisionXLab/FIRM-Reward Hugging Face: https://huggingface.co/collections/VisionXLab/firm-reward

</details>

<details>
<summary><b>7. DVD: Deterministic Video Depth Estimation with Generative Priors</b> ⭐ 72</summary>

<br/>

**👥 Authors:** Jing He, Chenfei Liao, Hongfei Zhang, haodongli, Harold328

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12250) • [📄 arXiv](https://arxiv.org/abs/2603.12250) • [📥 PDF](https://arxiv.org/pdf/2603.12250)

**💻 Code:** [⭐ Code](https://github.com/EnVision-Research/DVD)

> New SOTA of video depth estimation

</details>

<details>
<summary><b>8. WeEdit: A Dataset, Benchmark and Glyph-Guided Framework for Text-centric Image Editing</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zongkai Liu, Juntao Liu, Hui Zhang, fandong, lqniu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11593) • [📄 arXiv](https://arxiv.org/abs/2603.11593) • [📥 PDF](https://arxiv.org/pdf/2603.11593)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API PosterOmni: Generalized Artistic Poster Creation via Task Distillation and ...

</details>

<details>
<summary><b>9. ShotVerse: Advancing Cinematic Camera Control for Text-Driven Multi-Shot Video Creation</b> ⭐ 26</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11421) • [📄 arXiv](https://arxiv.org/abs/2603.11421) • [📥 PDF](https://arxiv.org/pdf/2603.11421)

**💻 Code:** [⭐ Code](https://github.com/Songlin1998/ShotVerse)

> Text-driven video generation has democratized film creation, but camera control in cinematic multi-shot scenarios remains a significant block. Implicit textual prompts lack precision, while explicit trajectory conditioning imposes prohibitive manu...

</details>

<details>
<summary><b>10. GRADE: Benchmarking Discipline-Informed Reasoning in Image Editing</b> ⭐ 23</summary>

<br/>

**👥 Authors:** Zuica96, NingLiao, Changyao, Glllllly, wzk1015

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12264) • [📄 arXiv](https://arxiv.org/abs/2603.12264) • [📥 PDF](https://arxiv.org/pdf/2603.12264)

**💻 Code:** [⭐ Code](https://github.com/VisionXLab/GRADE)

> Unified multimodal models target joint understanding, reasoning, and generation, but current image editing benchmarks are largely confined to natural images and shallow commonsense reasoning, offering limited assessment of this capability under st...

</details>

<details>
<summary><b>11. One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers</b> ⭐ 11</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12245) • [📄 arXiv](https://arxiv.org/abs/2603.12245) • [📥 PDF](https://arxiv.org/pdf/2603.12245)

**💻 Code:** [⭐ Code](https://github.com/snap-research/elit)

> We found that DiTs waste substantial compute by allocating it uniformly across pixels, despite large variation in regional difficulty. ELIT addresses this by introducing a variable-length set of latent tokens and two lightweight cross-attention la...

</details>

<details>
<summary><b>12. CREATE: Testing LLMs for Associative Creativity</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.09970) • [📄 arXiv](https://arxiv.org/abs/2603.09970) • [📥 PDF](https://arxiv.org/pdf/2603.09970)

**💻 Code:** [⭐ Code](https://github.com/ManyaWadhwa/CREATE)

> we introduce a benchmark for evaluating associative reasoning capabilities in LLMs. We introduce a creative utility metric that measures the diversity and quality of responses.

</details>

<details>
<summary><b>13. EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation</b> ⭐ 20</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12267) • [📄 arXiv](https://arxiv.org/abs/2603.12267) • [📥 PDF](https://arxiv.org/pdf/2603.12267)

**💻 Code:** [⭐ Code](https://github.com/HKU-MMLab/EVATok)

> Accepted by CVPR 2026 Project page: https://silentview.github.io/EVATok/ Github Code: https://github.com/HKU-MMLab/EVATok

</details>

<details>
<summary><b>14. EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models</b> ⭐ 21</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12252) • [📄 arXiv](https://arxiv.org/abs/2603.12252) • [📥 PDF](https://arxiv.org/pdf/2603.12252)

**💻 Code:** [⭐ Code](https://github.com/InternLM/EndoCoT)

> Recently, Multimodal Large Language Models (MLLMs) have been widely integrated into diffusion frameworks primarily as text encoders to tackle complex tasks such as spatial reasoning. However, this paradigm suffers from two critical limitations: (i...

</details>

<details>
<summary><b>15. RubiCap: Rubric-Guided Reinforcement Learning for Dense Image Captioning</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.09160) • [📄 arXiv](https://arxiv.org/abs/2603.09160) • [📥 PDF](https://arxiv.org/pdf/2603.09160)

> Scaling expert-annotated image captions is expensive. Supervised distillation from VLMs helps but has a diversity ceiling: models memorize the teacher's style and generalize poorly. Can RL fix this without a verifiable "ground truth"?

</details>

<details>
<summary><b>16. OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams</b> ⭐ 25</summary>

<br/>

**👥 Authors:** Weidi Xie, Shangzhe Di, haoningwu, Jazzcharles, Go2Heart

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12265) • [📄 arXiv](https://arxiv.org/abs/2603.12265) • [📥 PDF](https://arxiv.org/pdf/2603.12265)

**💻 Code:** [⭐ Code](https://github.com/Go2Heart/OmniStream)

> Project Page: https://go2heart.github.io/omnistream/ Paper: https://arxiv.org/abs/2603.12265 Code: https://github.com/Go2Heart/OmniStream

</details>

<details>
<summary><b>17. XSkill: Continual Learning from Experience and Skills in Multimodal Agents</b> ⭐ 14</summary>

<br/>

**👥 Authors:** Fung, Yi R., Xiaoye Qu, Zhaochen Su, Guanyu Jiang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12056) • [📄 arXiv](https://arxiv.org/abs/2603.12056) • [📥 PDF](https://arxiv.org/pdf/2603.12056)

**💻 Code:** [⭐ Code](https://github.com/XSkill-Agent/XSkill)

> We welcome feedback and discussion from the community 😄.

</details>

<details>
<summary><b>18. Mobile-GS: Real-time Gaussian Splatting for Mobile Devices</b> ⭐ 32</summary>

<br/>

**👥 Authors:** Xin Yu, Kun Zhan, Yida Wang, Xiaobiao Du

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11531) • [📄 arXiv](https://arxiv.org/abs/2603.11531) • [📥 PDF](https://arxiv.org/pdf/2603.11531)

**💻 Code:** [⭐ Code](https://github.com/xiaobiaodu/mobile-gs)

> 3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-quality rendering across a wide range of this http URL, its high computational demands and large storage costs pose significant challenges for deployment on mobile devi...

</details>

<details>
<summary><b>19. Are Video Reasoning Models Ready to Go Outside?</b> ⭐ 7</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.10652) • [📄 arXiv](https://arxiv.org/abs/2603.10652) • [📥 PDF](https://arxiv.org/pdf/2603.10652)

**💻 Code:** [⭐ Code](https://github.com/codepassionor/ROVA)

> Project Page: https://robust-video-reason.github.io arXiv: https://arxiv.org/abs/2603.10652 Github: https://github.com/codepassionor/ROVA

</details>

<details>
<summary><b>20. Understanding by Reconstruction: Reversing the Software Development Process for LLM Pretraining</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11103) • [📄 arXiv](https://arxiv.org/abs/2603.11103) • [📥 PDF](https://arxiv.org/pdf/2603.11103)

> good paper!, any plan releasing the workflow?

</details>

<details>
<summary><b>21. Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12246) • [📄 arXiv](https://arxiv.org/abs/2603.12246) • [📥 PDF](https://arxiv.org/pdf/2603.12246)

> Reasoning LLMs-as-Judges, which can benefit from inference-time scaling, provide a promising path for extending the success of reasoning models to non-verifiable domains where the output correctness/quality cannot be directly checked. However, whi...

</details>

<details>
<summary><b>22. Coarse-Guided Visual Generation via Weighted h-Transform Sampling</b> ⭐ 16</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12057) • [📄 arXiv](https://arxiv.org/abs/2603.12057) • [📥 PDF](https://arxiv.org/pdf/2603.12057)

**💻 Code:** [⭐ Code](https://github.com/HKUST-LongGroup/Coarse-guided-Gen)

> Achieve various conditional visual generation guided by a coarse sample with 1 line of code.

</details>

<details>
<summary><b>23. DIVE: Scaling Diversity in Agentic Task Synthesis for Generalizable Tool Use</b> ⭐ 1</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11076) • [📄 arXiv](https://arxiv.org/abs/2603.11076) • [📥 PDF](https://arxiv.org/pdf/2603.11076)

**💻 Code:** [⭐ Code](https://github.com/sheep333c/DIVE)

> We introduce DIVE, an iterative evidence-driven recipe that scales diversity in agentic task synthesis for generalizable tool use. Unlike prior approaches that synthesize tasks first, DIVE inverts the order — executing diverse, real-world tools fi...

</details>

<details>
<summary><b>24. Automatic Generation of High-Performance RL Environments</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12145) • [📄 arXiv](https://arxiv.org/abs/2603.12145) • [📥 PDF](https://arxiv.org/pdf/2603.12145)

> Automatic Generation of High-Performance RL Environments https://arxiv.org/abs/2603.12145

</details>

<details>
<summary><b>25. Meta-Reinforcement Learning with Self-Reflection for Agentic Search</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11327) • [📄 arXiv](https://arxiv.org/abs/2603.11327) • [📥 PDF](https://arxiv.org/pdf/2603.11327)

> This work introduces MR-Search, an in-context meta reinforcement learning (RL) formulation for agentic search with self-reflection. Instead of optimizing a policy within a single independent episode with sparse rewards, MR-Search trains a policy t...

</details>

<details>
<summary><b>26. PACED: Distillation at the Frontier of Student Competence</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Zhipeng Wang, Ran He, Zhengze Zhou, Hejian Sang, xuyd16

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11178) • [📄 arXiv](https://arxiv.org/abs/2603.11178) • [📥 PDF](https://arxiv.org/pdf/2603.11178)

> We welcome feedback and discussion from the community！

</details>

<details>
<summary><b>27. Geometric Autoencoder for Diffusion Models</b> ⭐ 4</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.10365) • [📄 arXiv](https://arxiv.org/abs/2603.10365) • [📥 PDF](https://arxiv.org/pdf/2603.10365)

**💻 Code:** [⭐ Code](https://github.com/sii-research/GAE)

> Latent diffusion models have established a new state-of-the-art in high-resolution visual generation. Integrating Vision Foundation Model priors improves generative efficiency, yet existing latent designs remain largely heuristic. These approaches...

</details>

<details>
<summary><b>28. Training Language Models via Neural Cellular Automata</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Pulkit Agrawal, Akarsh Kumar, Dan Lee, hanseungwook

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.10055) • [📄 arXiv](https://arxiv.org/abs/2603.10055) • [📥 PDF](https://arxiv.org/pdf/2603.10055)

**💻 Code:** [⭐ Code](https://github.com/danihyunlee/nca-pre-pretraining)

> Neural cellular automata generate synthetic spatiotemporal data for pre-pre-training large language models, achieving better performance and faster convergence than traditional natural language pre-training.

</details>

<details>
<summary><b>29. Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11665) • [📄 arXiv](https://arxiv.org/abs/2603.11665) • [📥 PDF](https://arxiv.org/pdf/2603.11665)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge (2026) Omni-RRM:...

</details>

<details>
<summary><b>30. Tiny Aya: Bridging Scale and Multilingual Depth</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11510) • [📄 arXiv](https://arxiv.org/abs/2603.11510) • [📥 PDF](https://arxiv.org/pdf/2603.11510)

> This is an automated message from the Librarian Bot . I found the following papers similar to this paper. The following papers were recommended by the Semantic Scholar API BYOL: Bring Your Own Language Into LLMs (2026) EstLLM: Enhancing Estonian C...

</details>

<details>
<summary><b>31. FireRedASR2S: A State-of-the-Art Industrial-Grade All-in-One Automatic Speech Recognition System</b> ⭐ 368</summary>

<br/>

**👥 Authors:** Wenpeng Li, Junjie Chen, Kai Huang, Yan Jia, Kaituo Xu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.10420) • [📄 arXiv](https://arxiv.org/abs/2603.10420) • [📥 PDF](https://arxiv.org/pdf/2603.10420)

**💻 Code:** [⭐ Code](https://github.com/FireRedTeam/FireRedASR2S)

> We present FireRedASR2S , a state-of-the-art industrial-grade all-in-one automatic speech recognition (ASR) system. It integrates four modules in a unified pipeline: ASR, Voice Activity Detection (VAD), Spoken Language Identification (LID), and Pu...

</details>

<details>
<summary><b>32. TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size</b> ⭐ 19</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.07988) • [📄 arXiv](https://arxiv.org/abs/2603.07988) • [📥 PDF](https://arxiv.org/pdf/2603.07988)

**💻 Code:** [⭐ Code](https://github.com/sail-sg/TeamHOI)

> TeamHOI is a novel framework for learning a unified decentralized policy for cooperative human-object interactions (HOI) that works seamlessly across varying team sizes and object configurations. We evaluate our framework on a cooperative table tr...

</details>

<details>
<summary><b>33. SoundWeaver: Semantic Warm-Starting for Text-to-Audio Diffusion Serving</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.07865) • [📄 arXiv](https://arxiv.org/abs/2603.07865) • [📥 PDF](https://arxiv.org/pdf/2603.07865)

> Tired of multi-second waits for stunning AI audio? We introduce SoundWeaver, the first training-free, model-agnostic serving system that revolutionizes text-to-audio diffusion by semantically warm-starting from a tiny cache of similar audio clips!...

</details>

<details>
<summary><b>34. Accent Vector: Controllable Accent Manipulation for Multilingual TTS Without Accented Data</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.07534) • [📄 arXiv](https://arxiv.org/abs/2603.07534) • [📥 PDF](https://arxiv.org/pdf/2603.07534)

> Accent is an integral part of society, reflecting multiculturalism and shaping how individuals express identity. The majority of English speakers are non-native (L2) speakers, yet current Text-To-Speech (TTS) systems primarily model American-accen...

</details>

<details>
<summary><b>35. NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.06922) • [📄 arXiv](https://arxiv.org/abs/2603.06922) • [📥 PDF](https://arxiv.org/pdf/2603.06922)

> Hi everyone! I am excited to share our ICLR 2026 paper, NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks. These are some interesting findings about the role of FFN (nonlinearity) in transformer architecture  (we also verified t...

</details>

<details>
<summary><b>36. SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.05483) • [📄 arXiv](https://arxiv.org/abs/2603.05483) • [📥 PDF](https://arxiv.org/pdf/2603.05483)

**💻 Code:** [⭐ Code](https://github.com/Shahriarnz14/SurvHTE-Bench)

> We present SurvHTE-Bench, a comprehensive causal inference benchmark to evaluate methods that estimate heterogeneous treatment effects from censored survival data, enabling rigorous, fair, and reproducible comparison across diverse causal scenarios.

</details>

<details>
<summary><b>37. EmbTracker: Traceable Black-box Watermarking for Federated Language Models</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12089) • [📄 arXiv](https://arxiv.org/abs/2603.12089) • [📥 PDF](https://arxiv.org/pdf/2603.12089)

> Federated Language Model (FedLM) allows a collaborative learning without sharing raw data, yet it introduces a critical vulnerability, as every untrustworthy client may leak the received functional model instance. Current watermarking schemes for ...

</details>

<details>
<summary><b>38. Dr. SHAP-AV: Decoding Relative Modality Contributions via Shapley Attribution in Audio-Visual Speech Recognition</b> ⭐ 3</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12046) • [📄 arXiv](https://arxiv.org/abs/2603.12046) • [📥 PDF](https://arxiv.org/pdf/2603.12046)

**💻 Code:** [⭐ Code](https://github.com/umbertocappellazzo/Dr-SHAP-AV)

> A Shapley-based framework revealing how audio-visual speech recognition models balance what they hear and what they see across architectures, decoding stages, and acoustic conditions.

</details>

<details>
<summary><b>39. Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Bo Liu, Yoonchang Sung, Chen Tang, Jay Shim, Jiaheng Hu

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11653) • [📄 arXiv](https://arxiv.org/abs/2603.11653) • [📥 PDF](https://arxiv.org/pdf/2603.11653)

> CRL for VLAs

</details>

<details>
<summary><b>40. Attention Sinks Are Provably Necessary in Softmax Transformers: Evidence from Trigger-Conditional Tasks</b> ⭐ 2</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11487) • [📄 arXiv](https://arxiv.org/abs/2603.11487) • [📥 PDF](https://arxiv.org/pdf/2603.11487)

**💻 Code:** [⭐ Code](https://github.com/YuvMilo/sinks-are-provably-necessary)

> Why do transformers attend so strongly to the first token? This paper proves that for certain trigger-conditional behaviors, attention sinks are necessary in softmax transformers. The author shows that any softmax model solving the task must devel...

</details>

<details>
<summary><b>41. Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.11045) • [📄 arXiv](https://arxiv.org/abs/2603.11045) • [📥 PDF](https://arxiv.org/pdf/2603.11045)

> Project Page: https://cab-lab-princeton.github.io/nefty/

</details>

<details>
<summary><b>42. A Mixed Diet Makes DINO An Omnivorous Vision Encoder</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2602.24181) • [📄 arXiv](https://arxiv.org/abs/2602.24181) • [📥 PDF](https://arxiv.org/pdf/2602.24181)

> We adapt DINOv2 into an "omnivorous" encoder that produces consistent embeddings for different input modalities like RGB, depth, and segmentation maps. By aligning paired modalities while anchoring to a frozen DINOv2 teacher, we unlock better cros...

</details>

<details>
<summary><b>43. Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights</b> ⭐ 18</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.12228) • [📄 arXiv](https://arxiv.org/abs/2603.12228) • [📥 PDF](https://arxiv.org/pdf/2603.12228)

**💻 Code:** [⭐ Code](https://github.com/sunrainyg/RandOpt)

> Pretraining produces a learned parameter vector that is typically treated as a starting point for further iterative adaptation. In this work, we instead view the outcome of pretraining as a distribution over parameter vectors, whose support alread...

</details>

<details>
<summary><b>44. HyPER-GAN: Hybrid Patch-Based Image-to-Image Translation for Real-Time Photorealism Enhancement</b> ⭐ 4</summary>

<br/>

**👥 Authors:** Nikos Nikolaidis, Stefanos Pasios

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.10604) • [📄 arXiv](https://arxiv.org/abs/2603.10604) • [📥 PDF](https://arxiv.org/pdf/2603.10604)

**💻 Code:** [⭐ Code](https://github.com/stefanos50/HyPER-GAN)

> Generative models are widely employed to enhance the photorealism of synthetic data for training computer vision algorithms. However, they often introduce visual artifacts that degrade the accuracy of these algorithms and require high computationa...

</details>

<details>
<summary><b>45. The Curse and Blessing of Mean Bias in FP4-Quantized LLM Training</b> ⭐ 0</summary>

<br/>

**👥 Authors:** Fanqi Yu, Yifeng Yang, Mengyi Chen, Zhendong Huang, Hengjie Cao

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.10444) • [📄 arXiv](https://arxiv.org/abs/2603.10444) • [📥 PDF](https://arxiv.org/pdf/2603.10444)

> Very insightful low bit training paper

</details>

<details>
<summary><b>46. 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video</b> ⭐ 6</summary>

<br/>

**👥 Authors:** Xiaoying Tang, Yebin Liu, Pujin Cheng, Liang An, luoxue-star

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.10125) • [📄 arXiv](https://arxiv.org/abs/2603.10125) • [📥 PDF](https://arxiv.org/pdf/2603.10125)

**💻 Code:** [⭐ Code](https://github.com/luoxue-star/4DEquine)

> Accepted to CVPR 2026

</details>

<details>
<summary><b>47. WaDi: Weight Direction-aware Distillation for One-step Image Synthesis</b> ⭐ 1</summary>

<br/>

**👥 Authors:** Yaxing Wang, Ge Wu, Senmao Li, Yang Cheng, Lei Wang

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.08258) • [📄 arXiv](https://arxiv.org/abs/2603.08258) • [📥 PDF](https://arxiv.org/pdf/2603.08258)

**💻 Code:** [⭐ Code](https://github.com/gudaochangsheng/WaDi)

> Despite the impressive performance of diffusion models such as Stable Diffusion (SD) in image generation, their slow inference limits practical deployment. Recent works accelerate inference by distilling multi-step diffusion into one-step generato...

</details>

<details>
<summary><b>48. Causal Attribution of Coastal Water Clarity Degradation to Nickel Processing Expansion at the Indonesia Morowali Industrial Park, Sulawesi</b> ⭐ 0</summary>

<br/>

**🔗 Links:** [🤗 HuggingFace](https://huggingface.co/papers/2603.07331) • [📄 arXiv](https://arxiv.org/abs/2603.07331) • [📥 PDF](https://arxiv.org/pdf/2603.07331)

**💻 Code:** [⭐ Code](https://github.com/sandyherho/supplMorowaliOcean)

> The global push for electric vehicles carries a hidden environmental cost, and a compelling new study leverages satellite data to finally quantify its impact on vulnerable marine ecosystems. Researchers investigated the coastal waters adjacent to ...

</details>

---

## 📅 Historical Archives

### 📊 Quick Access

| Type | Link | Papers |
|------|------|--------|
| 🕐 Latest | [`latest.json`](data/latest.json) | 48 |
| 📅 Today | [`2026-03-14.json`](data/daily/2026-03-14.json) | 48 |
| 📆 This Week | [`2026-W10.json`](data/weekly/2026-W10.json) | 71 |
| 🗓️ This Month | [`2026-03.json`](data/monthly/2026-03.json) | 300 |

### 📜 Recent Days

| Date | Papers | Link |
|------|--------|------|
| 📌 2026-03-14 | 48 | [View JSON](data/daily/2026-03-14.json) |
| 📄 2026-03-13 | 4 | [View JSON](data/daily/2026-03-13.json) |
| 📄 2026-03-12 | 7 | [View JSON](data/daily/2026-03-12.json) |
| 📄 2026-03-11 | 5 | [View JSON](data/daily/2026-03-11.json) |
| 📄 2026-03-10 | 1 | [View JSON](data/daily/2026-03-10.json) |
| 📄 2026-03-09 | 6 | [View JSON](data/daily/2026-03-09.json) |
| 📄 2026-03-08 | 24 | [View JSON](data/daily/2026-03-08.json) |

### 📚 Weekly Archives

| Week | Papers | Link |
|------|--------|------|
| 📅 2026-W10 | 71 | [View JSON](data/weekly/2026-W10.json) |
| 📅 2026-W09 | 201 | [View JSON](data/weekly/2026-W09.json) |
| 📅 2026-W08 | 184 | [View JSON](data/weekly/2026-W08.json) |
| 📅 2026-W07 | 197 | [View JSON](data/weekly/2026-W07.json) |

### 🗂️ Monthly Archives

| Month | Papers | Link |
|------|--------|------|
| 🗓️ 2026-03 | 300 | [View JSON](data/monthly/2026-03.json) |
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
