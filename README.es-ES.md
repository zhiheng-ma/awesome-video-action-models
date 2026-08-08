

# Modelos de Acción de Video Destacados [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
🤖 Una lista curada de Modelos de Acción de Video (VAMs, por sus siglas en inglés) — artículos que utilizan modelos de generación de video para producir acciones robóticas ejecutables. Cubre UniPi, UVA, mimic-video, Motus, Cosmos Policy, DreamZero y más.

## Tabla de Contenidos

- [¿Qué son los Modelos de Acción de Video?](#what-are-video-action-models)
- [Criterios de Inclusión](#inclusion-criteria)
- [Artículos por Categoría](#papers-by-category)
  - [1. Generación Conjunta de Video y Acción](#1-joint-video-action-generation)
  - [2. Video como Plan + Dinámica Inversa](#2-video-as-plan--inverse-dynamics)
  - [3. Backbone de Video → Decodificador de Acción (En Cascada)](#3-video-backbone--action-decoder-cascaded)
  - [4. Acción Latente a partir de Video](#4-latent-action-from-video)
  - [5. Video como Representación para la Política](#5-video-as-representation-for-policy)
  - [6. Modelos Mundiales Interactivos con Acciones](#6-interactive-world-models-with-actions)
  - [7. Generación de Video para Aumento de Datos y Simulación](#7-video-generation-for-data-augmentation--sim)
- [Línea de Tiempo](#timeline)
- [Comparaciones Clave](#key-comparisons)
- [Revisión Bibliográfica Relacionada](#related-surveys)
- [Contribuir](#contributing)

---

## ¿Qué son los Modelos de Acción de Video?

Los Modelos de Acción de Video (VAMs, por sus siglas en inglés) son una clase de modelos que satisfacen **dos condiciones fundamentales simultáneamente**:

1. **Utilizar un modelo de generación de video** (difusión, emparejamiento de flujos, autoregresivo, etc.) como componente central — ya sea como backbone preentrenado, un módulo entrenado conjuntamente o un motor de planificación.
2. **Generar acciones ejecutables** — comandos reales de articulaciones de robot, poses del efector final u otras señales de control que se pueden implementar en robots físicos o simulados.

Este paradigma se aparta de los modelos tradicionales de Visión-Lenguaje-Acción (VLA) que dependen del preentrenamiento estático de imagen-texto. En su lugar, los VAMs aprovechan las ricas dinámicas espaciotemporales capturadas por los modelos de video para lograr una mejor comprensión física y mayor eficiencia de muestreo.

---

## Criterios de Inclusión

Cada artículo de esta lista debe cumplir **ambas** condiciones:

- ✅ Incorpora un **modelo de generación/predicción de video** (p. ej., difusión de video, video DiT, emparejamiento de flujos de video, generación de video autoregresiva)
- ✅ Produce **acciones robóticas ejecutables** (torques en articulaciones, deltas del efector final, waypoints, etc.)

Los artículos que solo realizan predicción de video sin salida de acción, o que solo predicen acciones sin un componente de generación de video, son **no** están incluidos.

---

## Artículos por Categoría

### 1. Generación Conjunta de Video y Acción

Modelos que **desruean / generan conjuntamente** tanto los fotogramas futuros de video como las acciones robóticas dentro de un marco unificado.

| Artículo | Conferencia | Fecha | Modelo de Video | Tipo de Acción | Código |
|-------|------|------|-------------|-------------|------|
| [**Unified Video Action Model (UVA)**](https://arxiv.org/abs/2503.00200) | RSS 2025 | 2025.02 | MAR-based joint diffusion | Joint latent → decoupled diffusion heads | [✅](https://github.com/ShuangLI59/unified_video_action) |
| [**Unified World Models (UWM)**](https://weirdlabuw.github.io/uwm/) | RSS 2025 | 2025.01 | Diffusion Transformer (from scratch) | Coupled video+action diffusion w/ separate timesteps | [✅](https://github.com/WeirdLabUW/uwm) |
| [**VideoVLA**](https://arxiv.org/abs/2512.06963) | NeurIPS 2025 | 2025.12 | CogVideoX DiT | Joint video-action denoising in unified DiT | — |
| [**Motus**](https://arxiv.org/abs/2512.13030) | arXiv | 2025.12 | Wan2.1 (MoT architecture) | Mixture-of-Transformers: video + action + understanding experts | [✅](https://github.com/thu-ml/Motus) |
| [**Cosmos Policy**](https://arxiv.org/abs/2601.16163) | arXiv | 2026.01 | Cosmos-Predict2-2B | Actions encoded as latent frames in video diffusion | [✅](https://github.com/nvidia-cosmos/cosmos-policy) |
| [**DreamZero**](https://arxiv.org/abs/2602.15922) | arXiv | 2026.02 | Cosmos-based WAM | Joint video + action prediction; zero-shot transfer | [✅](https://github.com/dreamzero0/dreamzero) |
| [**GR-1**](https://arxiv.org/abs/2312.13139) | ICLR 2024 | 2023.12 | GPT-style autoregressive video model | End-to-end video + action autoregressive generation | — |
| [**GR-2**](https://arxiv.org/abs/2410.06158) | arXiv | 2024.10 | Video generation pretrained on 38M clips | Video generation + action fine-tuning | — |
| [**Prediction with Action (PAD)**](https://arxiv.org/abs/2407.09016) | NeurIPS 2025 | 2024.07 | Joint denoising diffusion | Visual prediction + action via joint denoising | — |

**Idea Clave:** Estos modelos tratan el video y la acción como dos modalidades dentro de un único proceso generativo, permitiendo el intercambio mutuo de información durante el entrenamiento y la inferencia.

---

### 2. Video como Plan + Dinámica Inversa

Modelos que primero **generan un plan de video** (trayectoria visual futura) y luego extraen acciones de este utilizando un **Modelo de Dinámica Inversa (IDM)** separado.

| Artículo | Conferencia | Fecha | Modelo de Video | Extracción de Acción | Código |
|-------|------|------|-------------|-------------------|------|
| [**UniPi**](https://arxiv.org/abs/2302.00111) | NeurIPS 2023 | 2023.02 | Text-conditioned video diffusion | Separate IDM from pixel frames | [✅](https://github.com/HalanJiang/UniPi_reproduce) |
| [**AVDC**](https://arxiv.org/abs/2310.16828) | ICLR 2024 | 2023.10 | Video diffusion model | Optical flow → SE(3) transforms → robot commands | — |
| [**SuSIE**](https://arxiv.org/abs/2310.10639) | ICRA 2024 | 2023.10 | InstructPix2Pix (subgoal image generation) | Goal-conditioned policy from predicted keyframe | [✅](https://github.com/rail-berkeley/susie) |
| [**Dreamitate**](https://arxiv.org/abs/2406.16862) | CoRL 2024 | 2024.06 | Pretrained video generator (fine-tuned) | End-effector tracking in generated video | — |
| [**VILP**](https://arxiv.org/abs/2502.01784) | arXiv | 2025.02 | Latent video diffusion planner | Latent video plan → low-level action policy | — |
| [**RoboDreamer**](https://arxiv.org/abs/2404.12377) | arXiv | 2024.04 | Compositional video diffusion | Compositional video → optical flow → actions | — |
| [**Video Language Planning (VLP)**](https://arxiv.org/abs/2310.10625) | ICLR 2024 | 2023.10 | Tree-search video generation | Goal image generation → goal-conditioned policy | — |

**Idea Clave:** Desacopla la planificación de alto nivel (realizada en el espacio de video) del control de bajo nivel (realizado por el IDM), aprovechando las ricas capacidades de planificación de los modelos de video.

---

### 3. Backbone de Video → Decodificador de Acción (En Cascada)

Modelos que utilizan un **backbone de video preentrenado** y adjuntan un **decodificador de acción separado** condicionado en características/potenciales de video.

| Artículo | Conferencia | Fecha | Backbone de Video | Decodificador de Acción | Código |
|-------|------|------|----------------|----------------|------|
| [**mimic-video**](https://arxiv.org/abs/2512.15692) | arXiv | 2025.12 | Cosmos-Predict2 (partial denoising) | Flow-matching IDM on latent video plans | [✅](https://github.com/lucidrains/mimic-video) |
| [**DiT4DiT**](https://dit4dit.github.io/) | arXiv | 2025 | Cosmos-Predict2.5-2B (Video DiT) | Cascaded Action DiT via cross-attention | — |
| [**Video Prediction Policy (VPP)**](https://arxiv.org/abs/2412.14803) | ICML 2025 | 2024.12 | Stable Video Diffusion (fine-tuned) | IDM conditioned on VDM internal representations | [✅](https://github.com/roboterax/video-prediction-policy) |
| [**Video Policy (Video Generators are Robot Policies)**](https://arxiv.org/abs/2508.00795) | arXiv | 2025.08 | SVD (fine-tuned on robot data) | Action head co-trained with video generation | — |
| [**FLARE**](https://arxiv.org/abs/2505.15659) | arXiv | 2025.05 | Diffusion Transformer | Latent future representation alignment → action head | — |
| [**UniVLA**](https://openreview.net/forum?id=PklMD8PwUy) | arXiv | 2025 | Autoregressive video tokens | World modeling supervision → action generation | — |
| [**WorldVLA**](https://arxiv.org/abs/2506.12348) | arXiv | 2025 | Autoregressive world model | Action tokens conditioned on predicted states | — |

**Idea Clave:** Utilizar el modelo de video preentrenado como extractor de características o prior de dinámica, con un decodificador de acción ligero que traduce los planes/características visuales en comandos motores.

---

### 4. Acción Latente a partir de Video

Modelos que aprenden un **espacio de acción latente** a partir de datos de video (sin datos reales de acción), y luego mapean estas acciones latentes a comandos robóticos reales.

| Artículo | Conferencia | Fecha | Modelo de Video | Método de Acción Latente | Código |
|-------|------|------|-------------|---------------------|------|
| [**LAPA**](https://arxiv.org/abs/2410.11758) | ICLR 2025 | 2024.10 | VQ-VAE latent action + VLM | Discrete latent actions → fine-tune to real actions | [✅](https://github.com/LatentActionPretraining/LAPA) |
| [**Motus**](https://arxiv.org/abs/2512.13030)* | arXiv | 2025.12 | Optical flow → latent action VAE | Pixel-level "delta action" from optical flow | [✅](https://github.com/thu-ml/Motus) |
| [**ViPRA**](https://vipra-project.github.io/) | arXiv | 2025 | Video-language model | Discrete latent actions via neural quantization | — |
| [**Genie**](https://arxiv.org/abs/2402.15391) | ICML 2024 | 2024.02 | Spatiotemporal video transformer | Latent action model from video-only data | — |
| [**CoMo**](https://arxiv.org/abs/2505.17006) | arXiv | 2025.05 | Continuous latent motion | Latent motion from internet video → robot actions | — |
| [**MOTO**](https://arxiv.org/abs/2412.04445) | arXiv | 2024.12 | Latent motion tokens | Motion token bridging video and robot actions | — |

*Motus aparece en múltiples categorías debido a su arquitectura unificada.

**Idea Clave:** Dado que la mayoría de los datos de video carece de etiquetas de acción, aprende una representación de acción latente, agnóstica a la embodiment, a partir de transiciones de video, y luego ajusta finamente con una pequeña cantidad de datos robóticos etiquetados.

---

### 5. Video como Representación para la Política

Modelos que aprovechan **representaciones internas de modelos de difusión de video** como codificadores visuales para el aprendizaje de políticas, utilizando la comprensión de la dinámica por parte del modelo de video.

| Artículo | Conferencia | Fecha | Modelo de Video | Cómo se usan las Características de Video | Código |
|-------|------|------|-------------|----------------------------|------|
| [**VPP (Video Prediction Policy)**](https://arxiv.org/abs/2412.14803) | ICML 2025 | 2024.12 | Video diffusion model | Internal diffusion features as visual representation | [✅](https://github.com/roboterax/video-prediction-policy) |
| [**FLARE**](https://arxiv.org/abs/2505.15659) | arXiv | 2025.05 | DiT | Future latent representation alignment | — |
| [**GR-1**](https://arxiv.org/abs/2312.13139) | ICLR 2024 | 2023.12 | GPT-style video model | Video generation pretraining → policy fine-tuning | — |

**Idea Clave:** Las representaciones internas de los modelos de generación de video capturan características espaciotemporales ricas que sirven como potentes codificadores visuales para la predicción de acciones posteriores.

---

### 6. Modelos Mundiales Interactivos con Acciones

Modelos de generación de video que operan como **simuladores de mundo interactivos**, aceptando acciones como entrada y generando fotogramas de video del siguiente estado correspondientes.

| Artículo | Conferencia | Fecha | Modelo de Video | Modo de Interacción | Código |
|-------|------|------|-------------|-----------------|------|
| [**Genie**](https://arxiv.org/abs/2402.15391) | ICML 2024 | 2024.02 | Spatiotemporal transformer | Learned latent actions control video generation | — |
| [**Genie 2 / Genie 3**](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | Google DeepMind | 2024-2025 | Large-scale world model | Real-time interactive environment generation | — |
| [**GameGen-X**](https://arxiv.org/abs/2411.00769) | arXiv | 2024.11 | DiT for game video | Interactive game video generation with actions | — |
| [**UVA**](https://arxiv.org/abs/2503.00200)* | RSS 2025 | 2025.02 | MAR-based | Forward dynamics: action → video prediction | [✅](https://github.com/ShuangLI59/unified_video_action) |
| [**UWM**](https://weirdlabuw.github.io/uwm/)* | RSS 2025 | 2025.01 | Diffusion Transformer | Action-conditioned video prediction | [✅](https://github.com/WeirdLabUW/uwm) |

*Estos también aparecen en la Categoría 1 debido a su funcionalidad dual.

**Idea Clave:** Estos modelos actúan como "simuladores neuronales" que predicen cómo evoluciona el mundo dadas acciones específicas, lo que permite la planificación y la evaluación basada en rollout (despliegue).

---

### 7. Generación de Video para Aumento de Datos y Simulación

Modelos que utilizan la generación de video para **crear datos de entrenamiento sintéticos** con etiquetas de acción para el aprendizaje de políticas posteriores.

| Artículo | Conferencia | Fecha | Modelo de Video | Estrategia de Generación de Datos | Código |
|-------|------|------|-------------|-------------------------|------|
| [**RoboMaster**](https://openreview.net/forum?id=OeDwYtp8n1) | arXiv | 2025 | Collaborative trajectory video diffusion | Multi-object interaction video synthesis | — |
| [**GenAug**](https://arxiv.org/abs/2302.06671) | ICRA 2023 | 2023.02 | Image generation for augmentation | Augmented visual data for policy training | — |
| [**VidBot**](https://arxiv.org/abs/2503.07135) | CVPR 2025 | 2025.03 | Human video → 3D affordance | 3D hand trajectory from video → robot actions | — |

**Idea Clave:** Utilizar la generación de video para ampliar la diversidad de los datos de entrenamiento, permitiendo que las políticas se generalicen a objetos, escenas y tareas novedosas.

---

## Línea de Tiempo

```
2023.02 ── UniPi: Video a partir de texto como política universal (NeurIPS 2023)
2023.10 ── AVDC: Difusión de video + flujo óptico para acción (ICLR 2024)
         ── SuSIE: Edición de imagen para generación de subobjetivos (ICRA 2024)
         ── VLP: Planificación de Lenguaje de Video (ICLR 2024)
2023.12 ── GR-1: Generación de video estilo GPT + acción (ICLR 2024)
2024.02 ── Genie: Modelo de mundo interactivo con acciones latentes (ICML 2024)
2024.06 ── Dreamitate: Generación de video → seguimiento de herramientas → acción
2024.07 ── PAD: Predicción con Acción mediante desrueación conjunta
2024.10 ── GR-2: Preentrenamiento con 38M clips de video → control robótico
         ── LAPA: Preentrenamiento de acción latente a partir de video (ICLR 2025)
2024.12 ── VPP: Política de Predicción de Video (ICML 2025)
         ── Motus: Modelo de mundo de acción latente unificado
         ── MOTO: Tokens de movimiento latente para aprendizaje robótico
         ── mimic-video: Modelo de Acción-Video más allá de VLA
         ── VideoVLA: Generadores de video como manipuladores robóticos (NeurIPS 2025)
2025.01 ── UWM: Difusión acoplada de video y acción (RSS 2025)
2025.02 ── UVA: Modelo de Acción de Video Unificado (RSS 2025)
         ── VILP: Aprendizaje por imitación con planificación de video latente
2025.05 ── FLARE: Alineación de representación latente futura
         ── CoMo: Movimiento latente continuo a partir de video de internet
2025.08 ── Video Policy: Los Generadores de Video son Políticas Robóticas
         ── Genie 3: Modelo de mundo interactivo en tiempo real
2025 ── ViPRA: Predicción de Video para Acciones Robóticas
      ── DiT4DiT: Dual DiT para dinámica de video + acciones
2026.01 ── Cosmos Policy: Ajuste fino de modelos de video para control
2026.02 ── DreamZero: Modelos de Acción de Mundo como políticas zero-shot
```

---

## Comparaciones Clave

### Paradigmas de Arquitectura

| Paradigma | Representantes | Ventajas | Desventajas |
|----------|----------------|----------|-------------|
| **Generación Conjunta** | UVA, UWM, VideoVLA, Cosmos Policy | Supervisión mutua; modelo unificado | Complejidad del entrenamiento; equilibrio de objetivos |
| **Video después de IDM** | UniPi, AVDC, Dreamitate | Modular; aprovecha video preentrenado | Acumulación de errores; inferencia lenta |
| **Backbone en Cascada** | mimic-video, DiT4DiT, VPP | Preserva priores preentrenados; eficiente | Requiere extracción de características cuidadosa |
| **Acción Latente** | LAPA, Motus, Genie, ViPRA | Escala a video sin etiquetar | Brecha entre acción latente y real |

### Backbones de Video Más Utilizados

| Backbone | Utilizado Por |
|----------|---------------|
| Cosmos-Predict2 / 2.5 | mimic-video, DiT4DiT, Cosmos Policy, DreamZero |
| CogVideoX | VideoVLA |
| Wan2.1 | Motus |
| Stable Video Diffusion (SVD) | Video Policy, VPP |
| Difusión U-Net Personalizada | UniPi, AVDC, RoboDreamer |
| MAR (Autoregresivo enmascarado) | UVA |
| Autoregresivo estilo GPT | GR-1, GR-2 |

---

## Revisión Bibliográfica Relacionada

- [Modelos de Generación de Video en Robótica: Aplicaciones, Desafíos de Investigación y Direcciones Futuras](https://arxiv.org/abs/2601.07823) (2026.01)
- [Un Paso Hacia los Modelos Mundiales: Una Revisión sobre Manipulación Robótica](https://arxiv.org/abs/2511.02097) (2025.11)
- [Modelos de Difusión para Manipulación Robótica: Una Revisión](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1606247) (2025)
- [Una Revisión Exhaustiva sobre Modelos Mundiales para IA Encarnada](https://arxiv.org/abs/2501.xxxxx) (2025)

---

## Listas Awesome Relacionadas

- [awesome-embodied-vla-va-vln](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln) — Lista más amplia que cubre modelos VLA, VA y VLN
- [awesome-world-model](https://github.com/GigaAI-research/Awesome-World-Model) — Modelos Mundiales para IA Encarnada

---

## Contribuir

¡Se aceptan contribuciones! Por favor, abre una Pull Request con artículos que satisfagan **ambos** criterios de inclusión:

1. Utiliza un modelo de generación de video
2. Genera acciones ejecutables

Por favor, incluye: título del artículo, enlace a arXiv, conferencia/revista, fecha y una breve nota sobre cómo cumple ambos criterios.

---

## Cita

Si encuentras útil esta lista, considera ponerle ⭐ al repositorio y citar los artículos relevantes.

---

## Licencia

Esta lista se publica bajo [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
