# awesome-video-action-models
🤖 A curated list of Video Action Models (VAMs) — papers using video generation models to produce executable robot actions. Covers UniPi, UVA, mimic-video, Motus, Cosmos Policy, DreamZero, and more.
# Awesome Video Action Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)


## Table of Contents

- [What Are Video Action Models?](#what-are-video-action-models)
- [Inclusion Criteria](#inclusion-criteria)
- [Papers by Category](#papers-by-category)
  - [1. Joint Video-Action Generation](#1-joint-video-action-generation)
  - [2. Video-as-Plan + Inverse Dynamics](#2-video-as-plan--inverse-dynamics)
  - [3. Video Backbone → Action Decoder (Cascaded)](#3-video-backbone--action-decoder-cascaded)
  - [4. Latent Action from Video](#4-latent-action-from-video)
  - [5. Video as Representation for Policy](#5-video-as-representation-for-policy)
  - [6. Interactive World Models with Actions](#6-interactive-world-models-with-actions)
  - [7. Video Generation for Data Augmentation & Sim](#7-video-generation-for-data-augmentation--sim)
- [Timeline](#timeline)
- [Key Comparisons](#key-comparisons)
- [Related Surveys](#related-surveys)
- [Contributing](#contributing)

---

## What Are Video Action Models?

Video Action Models (VAMs) are a class of models that satisfy **two core conditions simultaneously**:

1. **Use a video generation model** (diffusion, flow-matching, autoregressive, etc.) as a central component — either as a pretrained backbone, a jointly-trained module, or a planning engine.
2. **Generate executable actions** — real robot joint commands, end-effector poses, or other control signals that can be deployed on physical or simulated robots.

This paradigm departs from traditional Vision-Language-Action (VLA) models that rely on static image-text pretraining. Instead, VAMs leverage the rich spatiotemporal dynamics captured by video models to achieve better physical understanding and sample efficiency.

---

## Inclusion Criteria

Each paper in this list must meet **both** conditions:

- ✅ Incorporates a **video generation / prediction model** (e.g., video diffusion, video DiT, video flow-matching, autoregressive video generation)
- ✅ Produces **executable robot actions** (joint torques, delta end-effector, waypoints, etc.)

Papers that only do video prediction without action output, or only predict actions without a video generation component, are **not** included.

---

## Papers by Category

### 1. Joint Video-Action Generation

Models that **jointly denoise / generate** both future video frames and robot actions within a unified framework.

| Paper | Venue | Date | Video Model | Action Type | Code |
|-------|-------|------|-------------|-------------|------|
| [**Unified Video Action Model (UVA)**](https://arxiv.org/abs/2503.00200) | RSS 2025 | 2025.02 | MAR-based joint diffusion | Joint latent → decoupled diffusion heads | [✅](https://github.com/ShuangLI59/unified_video_action) |
| [**Unified World Models (UWM)**](https://weirdlabuw.github.io/uwm/) | RSS 2025 | 2025.01 | Diffusion Transformer (from scratch) | Coupled video+action diffusion w/ separate timesteps | [✅](https://github.com/WeirdLabUW/uwm) |
| [**VideoVLA**](https://arxiv.org/abs/2512.06963) | NeurIPS 2025 | 2025.12 | CogVideoX DiT | Joint video-action denoising in unified DiT | — |
| [**Motus**](https://arxiv.org/abs/2512.13030) | arXiv | 2025.12 | Wan2.1 (MoT architecture) | Mixture-of-Transformers: video + action + understanding experts | [✅](https://github.com/thu-ml/Motus) |
| [**Cosmos Policy**](https://arxiv.org/abs/2601.16163) | arXiv | 2026.01 | Cosmos-Predict2-2B | Actions encoded as latent frames in video diffusion | [✅](https://github.com/nvidia-cosmos/cosmos-policy) |
| [**DreamZero**](https://arxiv.org/abs/2602.15922) | arXiv | 2026.02 | Cosmos-based WAM | Joint video + action prediction; zero-shot transfer | [✅](https://github.com/dreamzero0/dreamzero) |
| [**GR-1**](https://arxiv.org/abs/2312.13139) | ICLR 2024 | 2023.12 | GPT-style autoregressive video model | End-to-end video + action autoregressive generation | — |
| [**GR-2**](https://arxiv.org/abs/2410.06158) | arXiv | 2024.10 | Video generation pretrained on 38M clips | Video generation + action fine-tuning | — |
| [**Prediction with Action (PAD)**](https://arxiv.org/abs/2407.09016) | NeurIPS 2025 | 2024.07 | Joint denoising diffusion | Visual prediction + action via joint denoising | — |

**Key Idea:** These models treat video and action as two modalities within a single generative process, enabling mutual information sharing during training and inference.

---

### 2. Video-as-Plan + Inverse Dynamics

Models that first **generate a video plan** (future visual trajectory), then extract actions from it using a separate **Inverse Dynamics Model (IDM)**.

| Paper | Venue | Date | Video Model | Action Extraction | Code |
|-------|-------|------|-------------|-------------------|------|
| [**UniPi**](https://arxiv.org/abs/2302.00111) | NeurIPS 2023 | 2023.02 | Text-conditioned video diffusion | Separate IDM from pixel frames | [✅](https://github.com/HalanJiang/UniPi_reproduce) |
| [**AVDC**](https://arxiv.org/abs/2310.16828) | ICLR 2024 | 2023.10 | Video diffusion model | Optical flow → SE(3) transforms → robot commands | — |
| [**SuSIE**](https://arxiv.org/abs/2310.10639) | ICRA 2024 | 2023.10 | InstructPix2Pix (subgoal image generation) | Goal-conditioned policy from predicted keyframe | [✅](https://github.com/rail-berkeley/susie) |
| [**Dreamitate**](https://arxiv.org/abs/2406.16862) | CoRL 2024 | 2024.06 | Pretrained video generator (fine-tuned) | End-effector tracking in generated video | — |
| [**VILP**](https://arxiv.org/abs/2502.01784) | arXiv | 2025.02 | Latent video diffusion planner | Latent video plan → low-level action policy | — |
| [**RoboDreamer**](https://arxiv.org/abs/2404.12377) | arXiv | 2024.04 | Compositional video diffusion | Compositional video → optical flow → actions | — |
| [**Video Language Planning (VLP)**](https://arxiv.org/abs/2310.10625) | ICLR 2024 | 2023.10 | Tree-search video generation | Goal image generation → goal-conditioned policy | — |

**Key Idea:** Decouple high-level planning (done in video space) from low-level control (done by IDM), leveraging the rich planning capabilities of video models.

---

### 3. Video Backbone → Action Decoder (Cascaded)

Models that use a **pretrained video model backbone** and attach a **separate action decoder** conditioned on video features/latents.

| Paper | Venue | Date | Video Backbone | Action Decoder | Code |
|-------|-------|------|----------------|----------------|------|
| [**mimic-video**](https://arxiv.org/abs/2512.15692) | arXiv | 2025.12 | Cosmos-Predict2 (partial denoising) | Flow-matching IDM on latent video plans | [✅](https://github.com/lucidrains/mimic-video) |
| [**DiT4DiT**](https://dit4dit.github.io/) | arXiv | 2025 | Cosmos-Predict2.5-2B (Video DiT) | Cascaded Action DiT via cross-attention | — |
| [**Video Prediction Policy (VPP)**](https://arxiv.org/abs/2412.14803) | ICML 2025 | 2024.12 | Stable Video Diffusion (fine-tuned) | IDM conditioned on VDM internal representations | [✅](https://github.com/roboterax/video-prediction-policy) |
| [**Video Policy (Video Generators are Robot Policies)**](https://arxiv.org/abs/2508.00795) | arXiv | 2025.08 | SVD (fine-tuned on robot data) | Action head co-trained with video generation | — |
| [**FLARE**](https://arxiv.org/abs/2505.15659) | arXiv | 2025.05 | Diffusion Transformer | Latent future representation alignment → action head | — |
| [**UniVLA**](https://openreview.net/forum?id=PklMD8PwUy) | arXiv | 2025 | Autoregressive video tokens | World modeling supervision → action generation | — |
| [**WorldVLA**](https://arxiv.org/abs/2506.12348) | arXiv | 2025 | Autoregressive world model | Action tokens conditioned on predicted states | — |

**Key Idea:** Use the pretrained video model as a feature extractor or dynamics prior, with a lightweight action decoder that translates visual plans/features into motor commands.

---

### 4. Latent Action from Video

Models that learn a **latent action space** from video data (without ground-truth actions), then map these latent actions to real robot commands.

| Paper | Venue | Date | Video Model | Latent Action Method | Code |
|-------|-------|------|-------------|---------------------|------|
| [**LAPA**](https://arxiv.org/abs/2410.11758) | ICLR 2025 | 2024.10 | VQ-VAE latent action + VLM | Discrete latent actions → fine-tune to real actions | [✅](https://github.com/LatentActionPretraining/LAPA) |
| [**Motus**](https://arxiv.org/abs/2512.13030)* | arXiv | 2025.12 | Optical flow → latent action VAE | Pixel-level "delta action" from optical flow | [✅](https://github.com/thu-ml/Motus) |
| [**ViPRA**](https://vipra-project.github.io/) | arXiv | 2025 | Video-language model | Discrete latent actions via neural quantization | — |
| [**Genie**](https://arxiv.org/abs/2402.15391) | ICML 2024 | 2024.02 | Spatiotemporal video transformer | Latent action model from video-only data | — |
| [**CoMo**](https://arxiv.org/abs/2505.17006) | arXiv | 2025.05 | Continuous latent motion | Latent motion from internet video → robot actions | — |
| [**MOTO**](https://arxiv.org/abs/2412.04445) | arXiv | 2024.12 | Latent motion tokens | Motion token bridging video and robot actions | — |

*Motus appears in multiple categories due to its unified architecture.

**Key Idea:** Since most video data lacks action labels, learn an embodiment-agnostic latent action representation from video transitions, then fine-tune with a small amount of labeled robot data.

---

### 5. Video as Representation for Policy

Models that leverage **video diffusion model internal representations** as visual encoders for policy learning, using the video model's understanding of dynamics.

| Paper | Venue | Date | Video Model | How Video Features Are Used | Code |
|-------|-------|------|-------------|----------------------------|------|
| [**VPP (Video Prediction Policy)**](https://arxiv.org/abs/2412.14803) | ICML 2025 | 2024.12 | Video diffusion model | Internal diffusion features as visual representation | [✅](https://github.com/roboterax/video-prediction-policy) |
| [**FLARE**](https://arxiv.org/abs/2505.15659) | arXiv | 2025.05 | DiT | Future latent representation alignment | — |
| [**GR-1**](https://arxiv.org/abs/2312.13139) | ICLR 2024 | 2023.12 | GPT-style video model | Video generation pretraining → policy fine-tuning | — |

**Key Idea:** The internal representations of video generation models capture rich spatiotemporal features that serve as powerful visual encoders for downstream action prediction.

---

### 6. Interactive World Models with Actions

Video generation models that operate as **interactive world simulators**, accepting actions as input and generating corresponding next-state video frames.

| Paper | Venue | Date | Video Model | Interaction Mode | Code |
|-------|-------|------|-------------|-----------------|------|
| [**Genie**](https://arxiv.org/abs/2402.15391) | ICML 2024 | 2024.02 | Spatiotemporal transformer | Learned latent actions control video generation | — |
| [**Genie 2 / Genie 3**](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | Google DeepMind | 2024-2025 | Large-scale world model | Real-time interactive environment generation | — |
| [**GameGen-X**](https://arxiv.org/abs/2411.00769) | arXiv | 2024.11 | DiT for game video | Interactive game video generation with actions | — |
| [**UVA**](https://arxiv.org/abs/2503.00200)* | RSS 2025 | 2025.02 | MAR-based | Forward dynamics: action → video prediction | [✅](https://github.com/ShuangLI59/unified_video_action) |
| [**UWM**](https://weirdlabuw.github.io/uwm/)* | RSS 2025 | 2025.01 | Diffusion Transformer | Action-conditioned video prediction | [✅](https://github.com/WeirdLabUW/uwm) |

*These also appear in Category 1 due to their dual functionality.

**Key Idea:** These models serve as "neural simulators" that predict how the world evolves given specific actions, enabling planning and rollout-based evaluation.

---

### 7. Video Generation for Data Augmentation & Sim

Models that use video generation to **create synthetic training data** with action labels for downstream policy learning.

| Paper | Venue | Date | Video Model | Data Generation Strategy | Code |
|-------|-------|------|-------------|-------------------------|------|
| [**RoboMaster**](https://openreview.net/forum?id=OeDwYtp8n1) | arXiv | 2025 | Collaborative trajectory video diffusion | Multi-object interaction video synthesis | — |
| [**GenAug**](https://arxiv.org/abs/2302.06671) | ICRA 2023 | 2023.02 | Image generation for augmentation | Augmented visual data for policy training | — |
| [**VidBot**](https://arxiv.org/abs/2503.07135) | CVPR 2025 | 2025.03 | Human video → 3D affordance | 3D hand trajectory from video → robot actions | — |

**Key Idea:** Use video generation to expand training data diversity, enabling policies to generalize to novel objects, scenes, and tasks.

---

## Timeline

```
2023.02 ── UniPi: Text-to-video as universal policy (NeurIPS 2023)
2023.10 ── AVDC: Video diffusion + optical flow for action (ICLR 2024)
         ── SuSIE: Image editing for subgoal generation (ICRA 2024)
         ── VLP: Video Language Planning (ICLR 2024)
2023.12 ── GR-1: GPT-style video generation + action (ICLR 2024)
2024.02 ── Genie: Interactive world model with latent actions (ICML 2024)
2024.06 ── Dreamitate: Video generation → tool tracking → action
2024.07 ── PAD: Prediction with Action via joint denoising
2024.10 ── GR-2: 38M video clips pretraining → robot control
         ── LAPA: Latent action pretraining from video (ICLR 2025)
2024.12 ── VPP: Video Prediction Policy (ICML 2025)
         ── Motus: Unified latent action world model
         ── MOTO: Latent motion tokens for robot learning
         ── mimic-video: Video-Action Model beyond VLAs
         ── VideoVLA: Video generators as robot manipulators (NeurIPS 2025)
2025.01 ── UWM: Coupled video-action diffusion (RSS 2025)
2025.02 ── UVA: Unified Video Action Model (RSS 2025)
         ── VILP: Imitation learning with latent video planning
2025.05 ── FLARE: Future latent representation alignment
         ── CoMo: Continuous latent motion from internet video
2025.08 ── Video Policy: Video Generators are Robot Policies
         ── Genie 3: Real-time interactive world model
2025 ── ViPRA: Video Prediction for Robot Actions
      ── DiT4DiT: Dual DiT for video dynamics + actions
2026.01 ── Cosmos Policy: Fine-tuning video models for control
2026.02 ── DreamZero: World Action Models as zero-shot policies
```

---

## Key Comparisons

### Architecture Paradigms

| Paradigm | Representatives | Pros | Cons |
|----------|----------------|------|------|
| **Joint Generation** | UVA, UWM, VideoVLA, Cosmos Policy | Mutual supervision; unified model | Training complexity; balancing objectives |
| **Video-then-IDM** | UniPi, AVDC, Dreamitate | Modular; leverages pretrained video | Error accumulation; slow inference |
| **Cascaded Backbone** | mimic-video, DiT4DiT, VPP | Preserves pretrained priors; efficient | Requires careful feature extraction |
| **Latent Action** | LAPA, Motus, Genie, ViPRA | Scales to unlabeled video | Latent-to-real action gap |

### Video Backbones Commonly Used

| Backbone | Used By |
|----------|---------|
| Cosmos-Predict2 / 2.5 | mimic-video, DiT4DiT, Cosmos Policy, DreamZero |
| CogVideoX | VideoVLA |
| Wan2.1 | Motus |
| Stable Video Diffusion (SVD) | Video Policy, VPP |
| Custom U-Net Diffusion | UniPi, AVDC, RoboDreamer |
| MAR (Masked Autoregressive) | UVA |
| GPT-style Autoregressive | GR-1, GR-2 |

---

## Related Surveys

- [Video Generation Models in Robotics: Applications, Research Challenges, Future Directions](https://arxiv.org/abs/2601.07823) (2026.01)
- [A Step Toward World Models: A Survey on Robotic Manipulation](https://arxiv.org/abs/2511.02097) (2025.11)
- [Diffusion Models for Robotic Manipulation: A Survey](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1606247) (2025)
- [A Comprehensive Survey on World Models for Embodied AI](https://arxiv.org/abs/2501.xxxxx) (2025)

---

## Related Awesome Lists

- [awesome-embodied-vla-va-vln](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln) — Broader list covering VLA, VA, and VLN models
- [awesome-world-model](https://github.com/GigaAI-research/Awesome-World-Model) — World models for embodied AI

---

## Contributing

Contributions are welcome! Please open a Pull Request with papers that satisfy **both** inclusion criteria:

1. Uses a video generation model
2. Generates executable actions

Please include: paper title, arXiv link, venue, date, and a brief note on how it fits both criteria.

---

## Citation

If you find this list useful, please consider starring ⭐ the repository and citing the relevant papers.

---

## License

This list is released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
