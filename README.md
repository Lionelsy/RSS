# 今日论文推荐 - 2025-11-11

共 77 篇论文

---

## 1. SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards

**论文链接:** [http://arxiv.org/abs/2511.07403v1](http://arxiv.org/abs/2511.07403v1)

**作者:** Hunar Batra, Haoqin Tu, Hardy Chen, Yuanze Lin, Cihang Xie, Ronald Clark

**发布时间:** 2025-11-10

**备注:** Preprint. Accepted at NeurIPS 2025 Workshops on SPACE in Vision,  Language, and Embodied AI (SpaVLE), Embodied World Models for Decision Making  (EWM), Aligning Reinforcement Learning Experimentalists and Theorists  (ARLET), and Scaling Environments for Agents (SEA)

### GPT解析

### 总结

SpatialThinker是一种3D感知的多模态大语言模型，通过强化学习训练，将结构化空间定位与多步推理相结合，有效解决了现有模型在空间理解方面的局限性。

### 背景

多模态大语言模型在视觉-语言任务中取得了显著进展，但在空间理解方面仍存在困难。现有的空间MLLM通常依赖于显式的3D输入或特定的架构修改，并受限于大规模数据集或稀疏监督。

### 目的

解决现有空间MLLM的局限性，开发一种能够在有限数据下实现稳健3D空间理解的模型，推动MLLMs向人类水平的视觉推理发展。

### 方法

SpatialThinker通过构建与任务相关的对象和空间关系的场景图来模拟类人的空间感知，并通过密集空间奖励推理得出答案。主要贡献包括：(1)数据合成管道生成高质量空间VQA数据集STVQA-7K；(2)使用多目标密集空间奖励的在线强化学习强制执行空间定位。

### 主要发现

SpatialThinker-7B在空间理解和真实世界VQA基准测试上优于监督微调和稀疏强化学习基线，与稀疏强化学习相比几乎使基模型增益翻倍，并超越了GPT-4o的性能。

### 结论

结合空间监督与奖励对齐推理的方法在有限数据下能有效实现稳健的3D空间理解，是推动MLLMs向人类水平视觉推理发展的重要一步。

### 翻译

多模态大语言模型在视觉-语言任务中已取得显著进展，但它们在空间理解方面仍然存在困难。现有的空间MLLM通常依赖于显式的3D输入或特定的架构修改，并受限于大规模数据集或稀疏监督。为解决这些局限性，我们引入了SpatialThinker，一种通过强化学习训练的3D感知MLLM，将结构化空间定位与多步推理相结合。该模型通过构建与任务相关的对象和空间关系的场景图来模拟类人的空间感知，并通过密集空间奖励推理得出答案。SpatialThinker包含两个关键贡献：(1)生成高质量空间VQA数据集STVQA-7K的数据合成管道；(2)使用多目标密集空间奖励强制执行空间定位的在线强化学习。SpatialThinker-7B在空间理解和真实世界VQA基准测试上优于监督微调和稀疏强化学习基线，与稀疏强化学习相比几乎使基模型增益翻倍，并超越了GPT-4o。这些结果表明，结合空间监督与奖励对齐推理在有限数据下实现稳健3D空间理解的有效性，推动MLLMs向人类水平的视觉推理发展。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态大语言模型(MLLMs)在空间理解方面的局限性，特别是在3D空间理解上的困难。这个问题非常重要，因为空间推理是人类智能的核心能力，对于机器人操作、导航、增强现实等具身AI任务至关重要，这些任务需要精确的空间意识作为交互决策的基础。现有方法要么需要大量训练数据，要么需要特定架构修改，要么依赖显式的3D输入，限制了MLLMs在现实世界应用中的能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有MLLMs在空间理解上的局限性，借鉴了人类空间认知过程（先构建场景关系再推理）和现有技术（场景图表示、强化学习与可验证奖励）。他们注意到现有RL方法使用简单的最终正确性奖励，对视觉引导推理指导不足，因此设计了更密集的奖励信号。方法设计上，他们将场景图与端到端推理集成（而非作为外部预处理），并采用多目标奖励框架，结合了格式、计数、准确度和空间奖励，通过字典序排序确保模型先满足格式要求，再优化计数和准确度，最后获得空间奖励。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过强化学习训练一个3D感知的MLLM，将结构化的场景图基础与多步空间推理相结合，模拟人类的空间感知过程。整体流程包括：1) 构建问题聚焦的场景子图，捕捉物体、关系和坐标；2) 设计多目标密集奖励（格式、计数、准确度和空间奖励），使用字典序排序；3) 使用GRPO进行在线RL策略优化；4) 构建STVQA-7K数据集（基于人类标注的场景图）；5) 在高分辨率图像上训练模型，更新所有参数（包括视觉编码器）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个将场景图基础与在线RL结合的MLLM，仅需7K样本就实现强大性能；2) 创建STVQA-7K高质量空间VQA数据集及可扩展生成管道；3) 设计密集多目标空间奖励，提供更丰富的学习信号；4) 展示了高质量数据与适当指导相结合的高效学习。相比之前工作，SPATIALTHINKER在数据效率上高出2-3个数量级（仅需7K样本 vs 数百万样本），使用密集而非稀疏奖励，将场景图与端到端推理集成，并在多个基准测试上超越GPT-4o和Claude 3.5 Sonnet等模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SPATIALTHINKER通过结合场景图基础与密集空间奖励的强化学习，仅用7K训练样本就实现了超越现有多模态大语言模型的3D空间理解能力，展示了高质量数据与适当指导相结合的强大学习效率。'}


### 论文摘要

Multimodal large language models (MLLMs) have achieved remarkable progress in vision-language tasks, but they continue to struggle with spatial understanding. Existing spatial MLLMs often rely on explicit 3D inputs or architecture-specific modifications, and remain constrained by large-scale datasets or sparse supervision. To address these limitations, we introduce SpatialThinker, a 3D-aware MLLM trained with RL to integrate structured spatial grounding with multi-step reasoning. The model simulates human-like spatial perception by constructing a scene graph of task-relevant objects and spatial relations, and reasoning towards an answer via dense spatial rewards. SpatialThinker consists of two key contributions: (1) a data synthesis pipeline that generates STVQA-7K, a high-quality spatial VQA dataset, and (2) online RL with a multi-objective dense spatial reward enforcing spatial grounding. SpatialThinker-7B outperforms supervised fine-tuning and the sparse RL baseline on spatial understanding and real-world VQA benchmarks, nearly doubling the base-model gain compared to sparse RL, and surpassing GPT-4o. These results showcase the effectiveness of combining spatial supervision with reward-aligned reasoning in enabling robust 3D spatial understanding with limited data and advancing MLLMs towards human-level visual reasoning.

---

## 2. Inference-Time Scaling of Diffusion Models for Infrared Data Generation

**论文链接:** [http://arxiv.org/abs/2511.07362v1](http://arxiv.org/abs/2511.07362v1)

**作者:** Kai A. Horstmann, Maxim Clouser, Kia Khezeli

**发布时间:** 2025-11-10

**备注:** Peer-reviewed workshop paper

### GPT解析

### 总结

研究提出了一种利用领域适应的CLIP验证器来改进红外图像生成质量的方法，通过在推理时指导扩散模型，能够在有限数据条件下提高生成效果。

### 背景

红外成像使用被动传感器进行基于温度的场景理解，在低能见度条件下优于传统RGB成像。然而，红外应用开发受限于高质量标注数据的稀缺，因为红外标注需要专业知识。合成红外图像生成可提供大规模训练数据，但受限于数据集不足，难以训练基础级生成扩散模型。

### 目的

在数据限制条件下，探索一种推理时扩展方法，使用领域适应的基于CLIP的验证器来增强红外图像生成质量。

### 方法

采用参数高效技术，在少量红外图像样本上微调FLUX.1-dev（最先进的文本到图像扩散模型）。训练好的验证器在推理过程中用于引导扩散采样，生成更高质量的红外图像，更好地与输入文本提示保持一致。

### 主要发现

该方法在生成质量上取得了一致的改进。与无引导的基线样本相比，在KAIST多光谱行人检测基准数据集上将FID分数降低了10%。

### 结论

推理时指导为弥合低数据红外设置中的领域差距提供了有希望的方向。

### 翻译

红外成像使用被动传感器实现基于温度的场景理解，特别是在传统RGB成像失效的低能见度条件下。然而，为红外应用开发下游视觉模型受到高质量标注数据稀缺的阻碍，因为红外标注需要专业知识。虽然合成红外图像生成有潜力通过提供大规模、多样化的训练数据来加速模型开发，但由于数据集有限，在红外领域训练基础级生成扩散模型一直难以实现。鉴于这些数据限制，我们探索了一种使用领域适应的基于CLIP的验证器的推理时扩展方法，以增强红外图像生成质量。我们使用参数高效技术，在少量红外图像样本上微调FLUX.1-dev（最先进的文本到图像扩散模型），将其适应到红外领域。训练好的验证器随后在推理过程中被使用，以引导扩散采样过程，生成更高质量的红外图像，更好地与输入文本提示保持一致。经验表明，我们发现我们的方法在生成质量上取得了一致的改进，与无引导的基线样本相比，在KAIST多光谱行人检测基准数据集上将FID分数降低了10%。我们的结果表明，推理时指导为弥合低数据红外设置中的领域差距提供了有希望的方向。


### 论文摘要

Infrared imagery enables temperature-based scene understanding using passive sensors, particularly under conditions of low visibility where traditional RGB imaging fails. Yet, developing downstream vision models for infrared applications is hindered by the scarcity of high-quality annotated data, due to the specialized expertise required for infrared annotation. While synthetic infrared image generation has the potential to accelerate model development by providing large-scale, diverse training data, training foundation-level generative diffusion models in the infrared domain has remained elusive due to limited datasets. In light of such data constraints, we explore an inference-time scaling approach using a domain-adapted CLIP-based verifier for enhanced infrared image generation quality. We adapt FLUX.1-dev, a state-of-the-art text-to-image diffusion model, to the infrared domain by finetuning it on a small sample of infrared images using parameter-efficient techniques. The trained verifier is then employed during inference to guide the diffusion sampling process toward higher quality infrared generations that better align with input text prompts. Empirically, we find that our approach leads to consistent improvements in generation quality, reducing FID scores on the KAIST Multispectral Pedestrian Detection Benchmark dataset by 10% compared to unguided baseline samples. Our results suggest that inference-time guidance offers a promising direction for bridging the domain gap in low-data infrared settings.

---

## 3. PlanT 2.0: Exposing Biases and Structural Flaws in Closed-Loop Driving

**论文链接:** [http://arxiv.org/abs/2511.07292v1](http://arxiv.org/abs/2511.07292v1)

**作者:** Simon Gerstenecker, Andreas Geiger, Katrin Renz

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文介绍了PlanT 2.0，一个面向对象的轻量级规划transformer，用于自动驾驶研究。通过系统性扰动模型输入，作者分析了模型失败的根本原因，并在CARLA基准测试中实现了最先进性能。研究揭示了模型存在的偏见和捷径学习问题，并提出了向数据中心开发转变的建议。

### 背景

自动驾驶领域最近的优先事项是基准性能和方法创新，而非对模型失败、偏见和捷径学习的深入分析。这导致了对当前失败缺乏深入理解的渐进式改进。

### 目的

理解自动驾驶模型失败的根本原因，通过系统性扰动模型输入并观察预测结果，进行深入分析，并提出改进方向。

### 方法

引入PlanT 2.0，一个轻量级、面向对象的规划transformer，专为CARLA中的自动驾驶研究设计。对象级表示使受控分析成为可能，因为输入可以轻松扰动。为应对CARLA Leaderboard 2.0的新挑战，对PlanT进行了多项升级。

### 主要发现

分析揭示了模型存在的失败案例，包括：由于障碍物多样性低导致缺乏场景理解；刚性专家行为导致可利用的捷径；以及对固定专家轨迹集的过度拟合。

### 结论

基于研究发现，作者主张向数据中心开发转变，重点是构建更丰富、更健壮、偏见更少的数据集，以提高自动驾驶模型的鲁棒性和可靠性。

### 翻译

大多数最新的自动驾驶研究优先考虑基准性能和方法创新，而不是对模型失败、偏见和捷径学习的深入分析。这导致了在没有深入理解当前失败情况下的渐进式改进。虽然查看模型失效的情况很简单，但要理解根本原因却很困难。这促使我们进行系统性研究，通过扰动模型输入并观察预测结果。我们引入了PlanT 2.0，一个轻量级的、面向对象的规划transformer，专为CARLA中的自动驾驶研究而设计。对象级表示使受控分析成为可能，因为输入可以轻松扰动（例如，通过改变位置或添加或删除某些对象），这与基于传感器的模型形成对比。为了应对CARLA Leaderboard 2.0引入的具有挑战性的新场景，我们对PlanT进行了多项升级，在Longest6 v2、Bench2Drive和CARLA验证路线上实现了最先进的性能。我们的分析揭示了有价值的失败案例，如由于障碍物多样性低导致的场景理解不足，刚性专家行为导致的可利用捷径，以及对固定专家轨迹集的过度拟合。基于这些发现，我们主张向数据中心开发转变，重点是构建更丰富、更健壮、偏见更少的数据集。我们在https://github.com/autonomousvision/plant2开源了我们的代码和模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶领域过分关注基准测试性能和方法创新，而对模型失败、偏见和捷径学习缺乏深入分析的问题。这个问题很重要，因为它导致虽然性能有所提升，但对失败的根本原因缺乏理解，可能使自动驾驶系统在实际应用中出现不可预测的故障，同时模型可能存在未被发现的偏见和捷径学习，在真实世界中可能导致严重安全问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到自动驾驶领域缺乏对模型失败原因的深入分析，并指出传感器端到端模型难以进行受控分析。因此选择重新审视PlanT模型，因为它具有物体中心表示，便于进行受控分析。作者借鉴了原始PlanT模型、PDM-Lite专家策略，以及[33]在输出表示上的方法，使用空间等距路径点进行横向规划和传统时间航点进行纵向规划。同时比较了多种航点生成方法，最终选择了简单线性层方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用物体中心表示而非传感器输入，便于进行受控分析和扰动；通过系统性地扰动模型输入并观察预测结果，来揭示模型的偏见、结构缺陷和捷径学习；设计一个轻量级的物体中心规划器，能够高效训练和评估，同时支持深入分析。整体实现流程包括：扩展输入表示（增加五个新物体类别、添加道路布局信息、增加检测范围）；升级输出表示（使用空间等距路径点进行横向规划，使用传统时间航点进行纵向规划，改进航点生成方法）；在多个基准测试上评估性能；通过系统扰动输入分析模型行为并识别各种失败模式。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出PlanT 2.0轻量级物体中心规划器，专为CARLA自动驾驶研究设计；2) 设计新的物体表示和输出表示，适应CARLA Leaderboard 2.0挑战；3) 实现受控分析框架，能系统扰动输入并观察模型行为；4) 进行深入模型分析，揭示8种关键问题（如缺乏环境理解、轨迹泛化能力差等）；5) 指出当前数据集局限性并提出改进方向。相比之前的工作，PlanT 2.0在多个基准测试上达到最先进性能，同时提供了更深入的分析；相比端到端传感器模型，其物体中心表示便于受控分析；相比其他规划器，它更注重揭示模型偏见和结构缺陷而非仅追求性能指标。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了PlanT 2.0，一个在CARLA自动驾驶基准测试上实现最先进性能的轻量级物体中心规划器，并通过系统性的输入扰动分析揭示了自动驾驶模型中的关键偏见和结构缺陷，为未来更鲁棒、更少偏见的自动驾驶系统开发提供了重要见解。'}


### 论文摘要

Most recent work in autonomous driving has prioritized benchmark performance and methodological innovation over in-depth analysis of model failures, biases, and shortcut learning. This has led to incremental improvements without a deep understanding of the current failures. While it is straightforward to look at situations where the model fails, it is hard to understand the underlying reason. This motivates us to conduct a systematic study, where inputs to the model are perturbed and the predictions observed. We introduce PlanT 2.0, a lightweight, object-centric planning transformer designed for autonomous driving research in CARLA. The object-level representation enables controlled analysis, as the input can be easily perturbed (e.g., by changing the location or adding or removing certain objects), in contrast to sensor-based models. To tackle the scenarios newly introduced by the challenging CARLA Leaderboard 2.0, we introduce multiple upgrades to PlanT, achieving state-of-the-art performance on Longest6 v2, Bench2Drive, and the CARLA validation routes. Our analysis exposes insightful failures, such as a lack of scene understanding caused by low obstacle diversity, rigid expert behaviors leading to exploitable shortcuts, and overfitting to a fixed set of expert trajectories. Based on these findings, we argue for a shift toward data-centric development, with a focus on richer, more robust, and less biased datasets. We open-source our code and model at https://github.com/autonomousvision/plant2.

---

## 4. Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images

**论文链接:** [http://arxiv.org/abs/2511.07222v1](http://arxiv.org/abs/2511.07222v1)

**作者:** JiaKui Hu, Shanshan Zhao, Qing-Guo Chen, Xuerui Qiu, Jialun Liu, Zhao Xu, Weihua Luo, Kaifu Zhang, Yanye Lu

**发布时间:** 2025-11-10

**备注:** Under review

### GPT解析

### 总结

本文提出了Omni-View系统，基于多视图图像将统一的多模态理解和生成扩展到3D场景，探索'生成促进理解'的原理。该系统由理解模型、纹理模块和几何模块组成，能够联合建模场景理解、新视图合成和几何估计，实现3D场景理解和生成任务之间的协同交互。

### 背景

基于多视图图像的3D场景理解和生成是计算机视觉领域的重要研究方向，但现有方法在整合理解和生成任务方面存在局限。

### 目的

开发一个能够同时处理3D场景理解、新视图合成和几何估计的系统，并通过'生成促进理解'的原理提升整体性能。

### 方法

Omni-View系统由理解模型、纹理模块和几何模块组成。纹理模块负责外观合成，利用其空间时间建模能力；几何模块提供显式几何约束。系统采用两阶段训练策略，实现3D场景理解和生成任务的协同交互。

### 主要发现

在VSI-Bench基准测试上，Omni-View达到55.4的最先进分数，超越现有的专用3D理解模型，同时在新型视图合成和3D场景生成方面也表现出色。

### 结论

Omni-View通过整合理解和生成任务，实现了3D场景处理的协同优化，证明了'生成促进理解'原理的有效性，为3D场景理解和生成提供了新的研究方向。

### 翻译

本文提出了Omni-View，它基于多视图图像将统一的多模态理解和生成扩展到3D场景，探索'生成促进理解'的原理。Omni-View由理解模型、纹理模块和几何模块组成，联合建模场景理解、新视图合成和几何估计，实现3D场景理解和生成任务之间的协同交互。通过设计，它利用了负责外观合成的纹理模块的空间时间建模能力，以及专用几何模块提供的显式几何约束，从而丰富了模型对3D场景的整体理解。采用两阶段策略训练后，Omni-View在VSI-Bench基准测试上达到55.4的最先进分数，超越现有的专用3D理解模型，同时在新型视图合成和3D场景生成方面也表现出色。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文旨在解决如何构建一个统一的3D场景理解和生成模型的问题。这个问题很重要，因为3D场景理解是人工智能领域的关键挑战，对机器人导航、自动驾驶和增强现实等应用至关重要。现有的方法通常专注于2D图像或需要明确的3D输入，限制了实际应用场景。同时，探索生成任务如何促进理解能力，对于构建更智能的通用人工智能系统具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者基于'生成促进理解'的原则进行思考，认为几何估计和新视角合成等生成任务具有内在的几何和时空建模能力，可以增强3D场景理解。他们借鉴了Bagel框架的共享多模态自注意力机制，并在此基础上创新性地将生成模型分解为纹理模块和几何模块两个专门组件。训练上采用两阶段策略：第一阶段同时训练所有模块以增强理解能力，第二阶段冻结理解模型优化生成性能。此外，还借鉴了视频扩散模型、点云表示、SigLIP视觉编码器等技术。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'生成促进理解'，即通过几何估计和新视角合成等生成任务来增强3D场景理解能力。整体实现流程包括：1) 架构上分为理解模型、纹理模块和几何模块；2) 两阶段训练策略，第一阶段同时训练所有模块并采用密集到稀疏(D2S)的渐进式训练方法，第二阶段冻结理解模型优化生成性能；3) 推理时，理解模型处理多视角图像执行理解任务，生成模型接收参考图像、相机姿态和提示生成新视角和几何信息。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 首次探索'生成促进理解'的3D统一模型；2) 创新性地将生成模型分离为纹理模块和几何模块；3) 提出两阶段训练策略和密集到稀疏(D2S)的渐进式训练方法；4) 利用几何约束增强时空建模能力。相比之前的工作，不同之处在于：不需要明确的3D输入即可实现高性能3D理解；专注于3D场景而非2D图像；同时实现高质量的场景理解和生成；通过几何约束提高生成的一致性和质量。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "Omni-View通过'生成促进理解'的创新理念，首次实现了仅基于多视角图像的高性能统一3D场景理解和生成，为人工智能在3D空间感知和创造能力的发展奠定了基础。"}


### 论文摘要

This paper presents Omni-View, which extends the unified multimodal understanding and generation to 3D scenes based on multiview images, exploring the principle that "generation facilitates understanding". Consisting of understanding model, texture module, and geometry module, Omni-View jointly models scene understanding, novel view synthesis, and geometry estimation, enabling synergistic interaction between 3D scene understanding and generation tasks. By design, it leverages the spatiotemporal modeling capabilities of its texture module responsible for appearance synthesis, alongside the explicit geometric constraints provided by its dedicated geometry module, thereby enriching the model's holistic understanding of 3D scenes. Trained with a two-stage strategy, Omni-View achieves a state-of-the-art score of 55.4 on the VSI-Bench benchmark, outperforming existing specialized 3D understanding models, while simultaneously delivering strong performance in both novel view synthesis and 3D scene generation.

---

## 5. TrueCity: Real and Simulated Urban Data for Cross-Domain 3D Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.07007v1](http://arxiv.org/abs/2511.07007v1)

**作者:** Duc Nguyen, Yan-Ling Lai, Qilin Zhang, Prabin Gyawali, Benedikt Schwab, Olaf Wysocki, Thomas H. Kolbe

**发布时间:** 2025-11-10

**备注:** The paper accepted for 3DV 2026 (International Conference on 3D  Vision 2026)

### GPT解析

### 总结

论文介绍了TrueCity数据集，这是首个包含真实世界和模拟点云的城市语义分割基准，用于解决3D语义场景理解中的合成到真实域差距问题。

### 背景

3D语义场景理解是3D计算机视觉领域的长期挑战，关键问题是有标注的真实世界数据有限，难以促进可泛化模型的发展。常见做法是模拟新数据，但合成数据虽然具有可扩展性和完美标签，却无法捕捉真实世界的复杂性和传感器噪声，导致合成到真实域的差距。

### 目的

引入TrueCity数据集，这是第一个具有厘米级精确标注的真实世界点云、语义3D城市模型和表示同一城市的标注模拟点云的城市语义分割基准，并提出与国际3D城市建模标准一致的分割类别，以便对合成到真实差距进行一致的评估。

### 方法

开发TrueCity基准数据集，并在常见基线上进行广泛实验，量化域差距并强调利用合成数据增强真实世界3D场景理解的策略。

### 主要发现

通过实验量化了域差距，并确定了利用合成数据增强真实世界3D场景理解的有效策略。

### 结论

TrueCity数据集将促进合成到真实差距量化的进一步发展，并能够实现可泛化的数据驱动模型。

### 翻译

3D语义场景理解仍然是3D计算机视觉界的一个长期挑战。关键问题之一是有标注的真实世界数据有限，难以促进可泛化模型的发展。解决这个问题的常见做法是模拟新数据。尽管合成数据集具有可扩展性和完美标签，但其设计者精心设计的场景无法捕捉真实世界的复杂性和传感器噪声，导致合成到真实域的差距。此外，没有基准提供用于面向分割的域偏移分析的真实和模拟点云的同步数据。我们引入了TrueCity，这是第一个具有厘米级精确标注的真实世界点云、语义3D城市模型和表示同一城市的标注模拟点云的城市语义分割基准。TrueCity提出了与国际3D城市建模标准一致的分割类别，使合成到真实差距的评估保持一致。我们在常见基线上进行的广泛实验量化了域偏移，并强调了利用合成数据增强真实世界3D场景理解的策略。我们相信TrueCity数据集将促进合成到真实差距量化的进一步发展，并实现可泛化的数据驱动模型。数据、代码和3D模型可在网上获取：https://tum-gis.github.io/TrueCity/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决3D语义场景理解中高质量标注数据不足以及模拟数据与真实数据之间存在'域差距'的问题。这个问题很重要，因为真实世界的高质量3D数据稀缺限制了可泛化模型的发展，而模拟数据无法完全捕捉真实世界的复杂性和传感器噪声，导致模型在实际应用中表现不佳。城市场景由于扫描距离变化、物体材料多样性和动态事件等因素尤其具有挑战性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出3D语义分割的两个主要障碍：标注数据稀缺和模拟-真实域差距。他们注意到现有数据集类别定义不一致，难以进行统一比较，因此决定与国际标准（CityGML 2.0和OpenDRIVE 1.4）保持一致。作者借鉴了现有工作：利用国际标准定义语义类别，使用CARLA模拟器进行激光扫描模拟，采用现有点云标注流程（如连通分量分析和布料模拟过滤算法），并使用多种点云分割方法作为基线模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个同步的真实和模拟数据集，使研究者能够精确量化分析模拟-真实域差距，并通过设计与国际标准兼容的语义类别确保数据集与现有工作流程的兼容性。整体流程包括：1)在德国Ingolstadt市中心采集高精度真实点云；2)基于真实点云构建符合CityGML标准的语义3D城市模型；3)使用CARLA模拟器生成模拟点云；4)对点云进行标注和分类；5)在不同比例的混合数据上训练和评估多种分割模型，分析域差距影响。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)TrueCity数据集——首个包含同步真实和模拟点云的城市语义分割基准，提供厘米级精度标注；2)标准化语义类别——提出与国际标准对齐的12个城市场景类别；3)系统化域差距分析——通过不同比例混合训练揭示各类别对域差距的敏感性。相比之前工作，TrueCity的真实和模拟数据针对同一地理位置，提供全面的城市场景类别而非仅关注特定建筑类型，解决了类别定义不一致问题，并提供了更全面的域差距分析。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TrueCity数据集通过提供同步的真实和模拟城市点云以及与国际标准兼容的语义标注，使研究者能够精确量化和分析模拟-真实域差距，从而促进更鲁棒的3D城市场景理解模型的发展。'}


### 论文摘要

3D semantic scene understanding remains a long-standing challenge in the 3D computer vision community. One of the key issues pertains to limited real-world annotated data to facilitate generalizable models. The common practice to tackle this issue is to simulate new data. Although synthetic datasets offer scalability and perfect labels, their designer-crafted scenes fail to capture real-world complexity and sensor noise, resulting in a synthetic-to-real domain gap. Moreover, no benchmark provides synchronized real and simulated point clouds for segmentation-oriented domain shift analysis. We introduce TrueCity, the first urban semantic segmentation benchmark with cm-accurate annotated real-world point clouds, semantic 3D city models, and annotated simulated point clouds representing the same city. TrueCity proposes segmentation classes aligned with international 3D city modeling standards, enabling consistent evaluation of synthetic-to-real gap. Our extensive experiments on common baselines quantify domain shift and highlight strategies for exploiting synthetic data to enhance real-world 3D scene understanding. We are convinced that the TrueCity dataset will foster further development of sim-to-real gap quantification and enable generalizable data-driven models. The data, code, and 3D models are available online: https://tum-gis.github.io/TrueCity/

---

## 6. Video Dataset for Surgical Phase, Keypoint, and Instrument Recognition in Laparoscopic Surgery (PhaKIR)

**论文链接:** [http://arxiv.org/abs/2511.06549v1](http://arxiv.org/abs/2511.06549v1)

**作者:** Tobias Rueckert, Raphaela Maerkl, David Rauber, Leonard Klausmann, Max Gutbrod, Daniel Rueckert, Hubertus Feussner, Dirk Wilhelm, Christoph Palm

**发布时间:** 2025-11-09

**备注:** 9 pages, 5 figures, 4 tables

### GPT解析

### 总结

本文提出了Surgical Procedure Phase, Keypoint, and Instrument Recognition (PhaKIR)数据集，这是一个包含三个医疗中心八例完整腹腔镜胆囊切除手术视频的多机构数据集，提供手术阶段识别、器械关键点估计和器械实例分割三个任务的帧级标注。

### 背景

机器人辅助微创手术(RAMIS)越来越多地依赖计算机视觉方法进行器械识别和手术流程理解，但现有数据集往往存在孤立任务、忽视时间依赖或缺乏多中心变异等问题。

### 目的

开发一个能够满足计算机视觉在手术中应用需求的多机构数据集，提供全面的标注信息以支持手术场景理解研究。

### 方法

收集三个医疗中心记录的八例完整腹腔镜胆囊切除手术视频，并提供三个相互关联任务的帧级标注：手术阶段识别(485,875帧)、器械关键点估计(19,435帧)和器械实例分割(19,435帧)。

### 主要发现

PhaKIR据作者所知是第一个提供阶段标签、器械姿态信息和像素级精确器械分割的多机构数据集，同时能够利用时间上下文，因为提供了完整的手术流程序列。

### 结论

该数据集作为PhaKIR挑战赛的基础，在MICCAI 2024内镜视觉挑战赛中被用作基准，验证了数据集的质量和相关价值，现已通过Zenodo平台公开可用。

### 翻译

机器人辅助和计算机辅助微创手术(RAMIS)越来越多地依赖计算机视觉方法进行可靠的器械识别和手术流程理解。开发此类系统通常需要大量、标注良好的数据集，但现有资源往往只处理孤立任务，忽视时间依赖性，或缺乏多中心变异。我们提出了手术阶段、关键点和器械识别(PhaKIR)数据集，包含三个医疗中心记录的八例完整腹腔镜胆囊切除手术视频。该数据集提供了三个相互关联任务的帧级标注：手术阶段识别(485,875帧)、器械关键点估计(19,435帧)和器械实例分割(19,435帧)。据我们所知，PhaKIR是第一个联合提供阶段标签、器械姿态信息和像素级精确器械分割的多机构数据集，同时由于完整手术流程序列的可用性，能够利用时间上下文。它作为PhaKIR挑战赛的基础，成为MICCAI 2024内镜视觉(EndoVis)挑战赛的一部分，用于评估手术场景理解方法，从而进一步验证了数据集的质量和相关价值。该数据集可通过Zenodo平台公开获取。


### 论文摘要

Robotic- and computer-assisted minimally invasive surgery (RAMIS) is increasingly relying on computer vision methods for reliable instrument recognition and surgical workflow understanding. Developing such systems often requires large, well-annotated datasets, but existing resources often address isolated tasks, neglect temporal dependencies, or lack multi-center variability. We present the Surgical Procedure Phase, Keypoint, and Instrument Recognition (PhaKIR) dataset, comprising eight complete laparoscopic cholecystectomy videos recorded at three medical centers. The dataset provides frame-level annotations for three interconnected tasks: surgical phase recognition (485,875 frames), instrument keypoint estimation (19,435 frames), and instrument instance segmentation (19,435 frames). PhaKIR is, to our knowledge, the first multi-institutional dataset to jointly provide phase labels, instrument pose information, and pixel-accurate instrument segmentations, while also enabling the exploitation of temporal context since full surgical procedure sequences are available. It served as the basis for the PhaKIR Challenge as part of the Endoscopic Vision (EndoVis) Challenge at MICCAI 2024 to benchmark methods in surgical scene understanding, thereby further validating the dataset's quality and relevance. The dataset is publicly available upon request via the Zenodo platform.

---

## 7. TimeSense:Making Large Language Models Proficient in Time-Series Analysis

**论文链接:** [http://arxiv.org/abs/2511.06344v1](http://arxiv.org/abs/2511.06344v1)

**作者:** Zhirui Zhang, Changhua Pei, Tianyi Gao, Zhe Xie, Yibo Hao, Zhaoyang Yu, Longlong Xu, Tong Xiao, Jing Han, Dan Pei

**发布时间:** 2025-11-09

### GPT解析

### 总结

TimeSense是一个多模态框架，通过平衡文本推理和保留时间感知能力，使大型语言模型能够精通时间序列分析，在多个任务上实现了最先进的性能。

### 背景

在时间序列领域，越来越多的研究将文本与时间数据结合，利用大型语言模型(LLMs)的推理能力来进行各种下游时间序列理解任务，使单个模型能够灵活执行以前需要每个领域专用模型才能完成的任务。

### 目的

解决现有方法在训练期间依赖文本标签进行监督所导致的模型偏向文本线索而忽略完整时间特征的问题，防止输出与底层时间序列上下文相矛盾。

### 方法

构建EvalTS基准(包含10个跨越三个难度级别的任务)来评估模型；提出TimeSense多模态框架，包含时间感知模块以在模型上下文中重建输入时间序列，并集成基于坐标的位置嵌入来增强时间序列数据的空间理解。

### 主要发现

TimeSense在多个任务上实现了最先进的性能，特别是在复杂的多维时间序列推理任务上优于现有方法。

### 结论

通过平衡文本推理和保留时间感知能力，TimeSense使大型语言模型能够更有效地进行时间序列分析，解决了现有方法中的文本偏见问题。

### 翻译

在时间序列领域，越来越多的工作将文本与时间数据相结合，利用大型语言模型(LLMs)的推理能力来完成各种下游时间序列理解任务。这使得单个模型能够灵活执行以前需要为每个领域开发专用模型才能完成的任务。然而，这些方法通常在训练期间依赖文本标签进行监督，使模型偏向于文本线索，同时可能忽略完整的时间特征。这种偏差可能导致输出与底层时间序列上下文相矛盾。为解决这一问题，我们构建了EvalTS基准，包含跨越三个难度级别的10个任务，从基本的时间模式识别到复杂的现实世界推理，用于在更具挑战性和现实性的场景下评估模型。我们还提出了TimeSense，一个多模态框架，通过平衡文本推理和保留时间感知能力，使LLM精通时间序列分析。TimeSense包含一个时间感知模块，在模型上下文中重建输入时间序列，确保文本推理基于时间序列动态。此外，为了增强对时间序列数据的空间理解，我们明确集成了基于坐标的位置嵌入，为每个时间点提供空间上下文，使模型能够更有效地捕获结构依赖关系。实验结果表明，TimeSense在多个任务上实现了最先进的性能，特别是在复杂的多维时间序列推理任务上优于现有方法。


### 论文摘要

In the time-series domain, an increasing number of works combine text with temporal data to leverage the reasoning capabilities of large language models (LLMs) for various downstream time-series understanding tasks. This enables a single model to flexibly perform tasks that previously required specialized models for each domain. However, these methods typically rely on text labels for supervision during training, biasing the model toward textual cues while potentially neglecting the full temporal features. Such a bias can lead to outputs that contradict the underlying time-series context. To address this issue, we construct the EvalTS benchmark, comprising 10 tasks across three difficulty levels, from fundamental temporal pattern recognition to complex real-world reasoning, to evaluate models under more challenging and realistic scenarios. We also propose TimeSense, a multimodal framework that makes LLMs proficient in time-series analysis by balancing textual reasoning with a preserved temporal sense. TimeSense incorporates a Temporal Sense module that reconstructs the input time-series within the model's context, ensuring that textual reasoning is grounded in the time-series dynamics. Moreover, to enhance spatial understanding of time-series data, we explicitly incorporate coordinate-based positional embeddings, which provide each time point with spatial context and enable the model to capture structural dependencies more effectively. Experimental results demonstrate that TimeSense achieves state-of-the-art performance across multiple tasks, and it particularly outperforms existing methods on complex multi-dimensional time-series reasoning tasks.

---

## 8. 10 Open Challenges Steering the Future of Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2511.05936v1](http://arxiv.org/abs/2511.05936v1)

**作者:** Soujanya Poria, Navonil Majumder, Chia-Yu Hung, Amir Ali Bagherzadeh, Chuan Li, Kenneth Kwok, Ziwei Wang, Cheston Tan, Jiajun Wu, David Hsu

**发布时间:** 2025-11-08

**备注:** AAAI 2026 (Senior Track)

### GPT解析

### 总结

本文讨论了视觉-语言-动作(VLA)模型在具身人工智能领域的发展，分析了10个主要里程碑和新兴趋势，旨在加速VLA模型的更广泛接受。

### 背景

视觉-语言-动作模型因其遵循自然语言指令的能力，在具身人工智能领域日益普及，这是继大语言模型和视觉语言模型成功之后的自然发展。

### 目的

讨论VLA模型发展的10个主要里程碑，探讨新兴趋势，并引起对可能加速VLA模型更广泛接受的研究途径的关注。

### 方法

通过分析VLA模型发展的关键里程碑和新兴趋势，系统梳理该领域的研究现状和未来方向。

### 主要发现

VLA模型发展的10个主要里程碑包括：多模态能力、推理能力、数据优化、评估方法、跨机器人动作泛化、效率提升、全身协调、安全性、智能体设计以及与人类协调能力。新兴趋势包括：空间理解、建模世界动态、训练后优化和数据合成。

### 结论

通过系统讨论VLA模型的发展里程碑和新兴趋势，可以促进相关研究，加速VLA模型的更广泛接受和应用。

### 翻译

由于遵循自然语言指令的能力，视觉-语言-动作(VLA)模型在具身人工智能领域日益普及，这继其前身——大语言模型和视觉语言模型——的广泛成功之后。在本文中，我们讨论了VLA模型持续发展中的10个主要里程碑——多模态、推理、数据、评估、跨机器人动作泛化、效率、全身协调、安全、智能体以及与人类的协调。此外，我们还探讨了使用空间理解、建模世界动态、训练后优化和数据合成等新兴趋势——所有这些都旨在达到这些里程碑。通过这些讨论，我们希望引起对可能加速VLA模型更广泛接受的研究途径的关注。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉-语言-行动(VLA)模型面临的十大开放性挑战。这些问题在现实中非常重要，因为VLA模型是实现真正智能机器人的关键，能让机器人理解自然语言指令并在复杂环境中执行任务，推动具身AI从实验室走向实际应用，如家庭服务、灾难救援和工业制造等领域。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析当前VLA模型研究现状，识别出10个主要挑战，并针对每个挑战提出可能的解决方向。他们借鉴了多个现有工作，包括大型语言模型(LLMs)、视觉语言模型(VLMs)、机器人控制、强化学习和模仿学习等领域的成果，还整合了分层规划、世界建模和预测等理论框架，这些方法都是基于现有研究的延伸和创新。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合视觉感知、语言理解和行动执行，使机器人能够理解自然语言指令并在复杂环境中执行任务。整体流程包括：1)接收多模态输入(视觉、语言、上下文)；2)通过视觉语言模型理解场景和任务，进行高层次规划；3)生成具体机器人动作序列；4)执行动作并接收环境反馈；5)根据反馈调整后续动作。具体实现分为离散动作模型和连续动作模型两种路径，还提出了分层规划框架和推理前行动等高级方法。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：系统性挑战识别、多模态感知扩展、分层规划框架、空间理解增强、通用动作表示、世界动态建模、数据合成方法和安全与保障机制。相比之前工作，这篇论文提供了全面的视角，强调从感知到行动的完整闭环，重视实际应用中的挑战，提出系统性的解决方案框架，并关注多智能体协作和人机交互等更高级的协作模式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文系统性地识别并探讨了视觉-语言-行动模型面临的十大核心挑战，并提出了一系列创新方法框架，为构建更强大、更实用、更安全的具身AI系统提供了全面的研究路线图。'}


### 论文摘要

Due to their ability of follow natural language instructions, vision-language-action (VLA) models are increasingly prevalent in the embodied AI arena, following the widespread success of their precursors -- LLMs and VLMs. In this paper, we discuss 10 principal milestones in the ongoing development of VLA models -- multimodality, reasoning, data, evaluation, cross-robot action generalization, efficiency, whole-body coordination, safety, agents, and coordination with humans. Furthermore, we discuss the emerging trends of using spatial understanding, modeling world dynamics, post training, and data synthesis -- all aiming to reach these milestones. Through these discussions, we hope to bring attention to the research avenues that may accelerate the development of VLA models into wider acceptability.

---

## 9. Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning

**论文链接:** [http://arxiv.org/abs/2511.05894v1](http://arxiv.org/abs/2511.05894v1)

**作者:** Fei Yu, Quan Deng, Shengeng Tang, Yuehua Li, Lechao Cheng

**发布时间:** 2025-11-08

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种用于开放世界3D场景图生成的统一框架，结合了视觉语言模型和检索增强推理技术，实现了可泛化和交互式的3D场景理解。该方法包含动态场景图生成模块和检索增强推理管道两个关键组件，在多个基准测试和任务中表现出强大的泛化能力和优越性能。

### 背景

在开放世界环境中理解3D场景对视觉和机器人技术构成了根本性挑战，主要受限于封闭词汇监督和静态标注的局限性。

### 目的

提出一个统一框架，解决开放世界3D场景理解中的挑战，实现可泛化和交互式的3D场景理解。

### 方法

提出了一种带有检索增强推理的开放世界3D场景图生成统一框架，整合视觉语言模型与基于检索的推理，支持多模态探索和语言引导交互。方法包含两个关键组件：(1)动态场景图生成模块，无需固定标签集即可检测对象并推断语义关系；(2)检索增强推理管道，将场景图编码为向量数据库以支持文本/图像条件查询。

### 主要发现

在3DSSG和Replica基准上对四个任务（场景问答、视觉定位、实例检索和任务规划）进行的评估中，展示了该方法强大的泛化能力和在不同环境中的优越性能。

### 结论

结合开放词汇感知与基于检索的推理对于可扩展的3D场景理解是有效的，研究结果突显了这种组合方法的价值。

### 翻译

在开放世界环境中理解3D场景对视觉和机器人技术构成了根本性挑战，主要受限于封闭词汇监督和静态标注的局限性。为解决这一问题，我们提出了一个带有检索增强推理的开放世界3D场景图生成统一框架，实现了可泛化和交互式的3D场景理解。我们的方法将视觉语言模型与基于检索的推理相结合，支持多模态探索和语言引导交互。该框架包含两个关键组件：(1)动态场景图生成模块，无需固定标签集即可检测对象并推断语义关系；(2)检索增强推理管道，将场景图编码为向量数据库以支持文本/图像条件查询。我们在3DSSG和Replica基准上对四个任务（场景问答、视觉定位、实例检索和任务规划）评估了我们的方法，展示了强大的泛化能力和在不同环境中的优越性能。我们的研究结果突显了结合开放词汇感知与基于检索的推理对于可扩展3D场景理解的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决开放世界3D场景理解的问题，即如何让计算机系统在动态、非结构化环境中识别未知物体并理解它们之间的关系。这个问题在现实中非常重要，因为自主导航、增强现实等应用需要系统适应新环境，而传统方法依赖于预定义的物体类别和固定场景，无法处理现实世界中经常出现的新物体和场景变化。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统3D场景理解方法的局限性，特别是它们对封闭词汇表和静态注释的依赖。然后，作者借鉴了视觉-语言模型(VLMs)在2D任务中的成功经验，结合检索机制来处理开放世界场景。设计过程中，作者参考了多模态大语言模型(MLLMs)的架构，包括大语言模型骨干、视觉编码器和适配器模块，以及参数高效微调方法。同时，作者也基于3D场景图生成的前期工作进行了改进，消除了对人工注释或固定姿态RGB-D数据的依赖，实现了完全无注释的3D场景图生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将视觉-语言模型与检索机制相结合，实现开放世界3D场景理解和推理。通过动态场景图构建环境表示，支持对未见物体的识别和关系推断，并使用检索增强推理来支持多模态交互。整体流程分为三部分：1)开放世界3D场景图生成，包括多帧目标检测、最佳视角选择、物体过滤和语义关系提取；2)检索增强语义推理，将场景图编码为向量数据库，处理用户查询并进行大语言模型推理；3)场景驱动的多模态交互，支持文本问答、视觉定位、实例检索和任务规划四种任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的开放世界3D场景图生成框架，无需固定注释；2)检索增强推理模块，将场景图编码为支持查询的向量数据库；3)多模态交互能力，支持四种场景交互任务。相比之前工作，不同之处在于：1)相比封闭词汇方法，能识别未知物体且不依赖预定义标签；2)相比开放词汇方法，无需人工注释或固定姿态RGB-D数据，泛化能力更强；3)相比多模态大语言模型，专门针对3D场景设计，结合场景图结构和检索推理，在空间理解和任务规划方面表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合视觉-语言模型和检索增强推理的开放世界3D场景图生成框架，实现了无需人工注释的动态场景理解和多模态交互能力，显著提升了系统在复杂环境中的泛化性和实用性。'}


### 论文摘要

Understanding 3D scenes in open-world settings poses fundamental challenges for vision and robotics, particularly due to the limitations of closed-vocabulary supervision and static annotations. To address this, we propose a unified framework for Open-World 3D Scene Graph Generation with Retrieval-Augmented Reasoning, which enables generalizable and interactive 3D scene understanding. Our method integrates Vision-Language Models (VLMs) with retrieval-based reasoning to support multimodal exploration and language-guided interaction. The framework comprises two key components: (1) a dynamic scene graph generation module that detects objects and infers semantic relationships without fixed label sets, and (2) a retrieval-augmented reasoning pipeline that encodes scene graphs into a vector database to support text/image-conditioned queries. We evaluate our method on 3DSSG and Replica benchmarks across four tasks-scene question answering, visual grounding, instance retrieval, and task planning-demonstrating robust generalization and superior performance in diverse environments. Our results highlight the effectiveness of combining open-vocabulary perception with retrieval-based reasoning for scalable 3D scene understanding.

---

## 10. Lite VLA: Efficient Vision-Language-Action Control on CPU-Bound Edge Robots

**论文链接:** [http://arxiv.org/abs/2511.05642v1](http://arxiv.org/abs/2511.05642v1)

**作者:** Justin Williams, Kishor Datta Gupta, Roy George, Mrinmoy Sarkar

**发布时间:** 2025-11-07

### GPT解析

### 总结

该研究展示了在移动机器人上部署小型视觉语言模型以实现实时场景理解和推理的可行性，使机器人能够在动态环境中同时进行移动和推理，无需云连接支持。

### 背景

在GPS受限环境中运行的自主机器人，其人工智能模型在边缘的部署日益重要，因为本地、资源高效的推理是必不可少的。

### 目的

证明在移动机器人上部署小型视觉语言模型（VLM）的可行性，以在严格的计算约束下实现实时场景理解和推理。

### 方法

提出一种框架，将紧凑的VLM与多模态感知集成，直接在嵌入式硬件上执行上下文解释，消除对云连接的依赖，使机器人在动态环境中仅使用板载硬件就能同时进行移动和推理。

### 主要发现

实验验证了计算效率、任务准确性和系统响应能力之间的平衡，在移动机器人上的实现确认了小型VLM在边缘进行并发推理和移动的首次成功部署之一。

### 结论

这项工作为服务机器人、灾难响应和国防行动等应用中的可扩展、有保障的自主性奠定了基础。

### 翻译

在GPS受限环境中运行的自主机器人，其人工智能模型在边缘的部署日益重要，因为本地、资源高效的推理是必不可少的。这项工作证明了在移动机器人上部署小型视觉语言模型（VLM）的可行性，以在严格的计算约束下实现实时场景理解和推理。与先前分离感知和移动的方法不同，所提出的框架使机器人在动态环境中仅使用板载硬件就能同时进行移动和推理。该系统将紧凑的VLM与多模态感知集成，直接在嵌入式硬件上执行上下文解释，消除对云连接的依赖。实验验证突显了计算效率、任务准确性和系统响应能力之间的平衡。在移动机器人上的实现确认了小型VLM在边缘进行并发推理和移动的首次成功部署之一。这项工作为服务机器人、灾难响应和国防行动等应用中的可扩展、有保障的自主性奠定了基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在资源受限的边缘设备上实现高效的视觉-语言-动作控制问题。这个问题在现实中非常重要，因为在GPS受限环境（如灾难区域、地下设施或国防任务）中运行的机器人需要完全自包含的智能，能够本地推理和行动而不依赖外部基础设施。现有的大型多模态模型严重依赖云端计算，在连接、电源和计算能力有限的边缘场景中不实用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到大型模型在边缘设备上的局限性，然后考虑将感知、规划和控制统一在单一推理循环中。他们借鉴了多项现有工作：SmolVLA作为基础架构，LoRA用于参数高效微调，QLoRA结合4位量化技术，SmolVLM作为基础多模态架构，llama-cpp运行时用于CPU推理，以及ROS 2用于机器人控制。设计方法包括使用小型多模态变换器、LoRA微调、4位量化和混合精度设计，最终集成到ROS 2环境中实现端到端控制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在资源受限的边缘设备上通过模型小型化、量化和参数高效微调实现高效的视觉-语言-动作控制，使机器人能够在本地进行实时推理而不依赖云端。整体流程包括：1)通过远程操作收集RGB图像和对应动作并进行时间同步；2)使用LoRA微调预训练的SmolVLM模型；3)应用4位NF4量化和混合精度设计；4)部署到ROS 2环境；5)执行时捕获RGB帧，通过量化模型处理生成语义动作命令，解析并转换为机器人运动命令。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次实现完全基于CPU的视觉-语言-动作推理；2)使用LoRA进行参数高效微调；3)采用4位NF4量化和混合精度设计；4)实现端到端ROS 2管道；5)提出EDGE-VLA-ROADMAP可扩展路线图。相比之前工作，与SmolVLA不同，LiteVLA针对移动机器人且完全基于CPU运行；与其他边缘VLA系统相比，实现了更低的内存占用和更高的推理效率；与大型多模态模型相比，专为边缘设备设计且无需云端支持；技术上采用混合量化策略和llama-cpp运行时。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文成功实现了在资源受限的边缘设备上运行轻量级视觉-语言-动作模型，通过模型量化、参数高效微调和ROS 2集成，使机器人能够在本地进行实时场景理解和推理，为服务机器人、灾难响应和国防应用等场景提供了可扩展的自主解决方案。'}


### 论文摘要

The deployment of artificial intelligence models at the edge is increasingly critical for autonomous robots operating in GPS-denied environments where local, resource-efficient reasoning is essential. This work demonstrates the feasibility of deploying small Vision-Language Models (VLMs) on mobile robots to achieve real-time scene understanding and reasoning under strict computational constraints. Unlike prior approaches that separate perception from mobility, the proposed framework enables simultaneous movement and reasoning in dynamic environments using only on-board hardware. The system integrates a compact VLM with multimodal perception to perform contextual interpretation directly on embedded hardware, eliminating reliance on cloud connectivity. Experimental validation highlights the balance between computational efficiency, task accuracy, and system responsiveness. Implementation on a mobile robot confirms one of the first successful deployments of small VLMs for concurrent reasoning and mobility at the edge. This work establishes a foundation for scalable, assured autonomy in applications such as service robotics, disaster response, and defense operations.

---

## 11. Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition

**论文链接:** [http://arxiv.org/abs/2511.05622v1](http://arxiv.org/abs/2511.05622v1)

**作者:** Nicholas Babey, Tiffany Gu, Yiheng Li, Cristian Meo, Kevin Zhu

**发布时间:** 2025-11-06

**备注:** Accepted at NeurIPS 2025 SpaVLE, for code see  https://github.com/nbabey20/groundactrec , 9 pages, 1 figure

### GPT解析

### 总结

研究提出了一种基于物理空间的动作识别模型，通过融合V-JEPA2的上下文预测世界动态和CoMotion的抗遮挡人体姿态数据，使具身智能体能够更有效地理解和与周围世界交互。

### 背景

当前的动作识别模型主要依赖RGB视频，学习的是模式与动作标签之间的表面关联，难以捕捉复杂场景中潜在的物理交互动态和人体姿态。

### 目的

提出一种能够基于物理空间进行动作识别的模型架构，使具身智能体能够更有效地理解和与周围世界交互。

### 方法

提出了一种融合两种互补表示的模型架构：V-JEPA2的上下文预测世界动态和CoMotion的明确抗遮挡人体姿态数据，将动作识别建立在物理空间基础上。

### 主要发现

模型在InHARD和UCF-19-Y-OCC基准测试上分别验证了在通用动作识别和高遮挡动作识别任务上的性能，优于其他三种基线方法，特别是在复杂、有遮挡的场景中表现更好。

### 结论

动作识别需要空间理解的支持，而不仅仅是统计模式识别。

### 翻译

为了使具身智能体能够有效地理解和与周围世界互动，他们需要对基于物理空间的人类动作有细致的理解。当前的动作识别模型通常依赖于RGB视频，学习的是模式与动作标签之间的表面关联，因此难以捕捉复杂场景中潜在的物理交互动态和人体姿态。我们提出了一种模型架构，通过融合两种强大且互补的表示将动作识别建立在物理空间基础上：V-JEPA2的上下文预测世界动态和CoMotion的明确抗遮挡人体姿态数据。我们的模型分别在InHARD和UCF-19-Y-OCC基准测试上进行了验证，分别用于通用动作识别和高遮挡动作识别。我们的模型优于其他三种基线方法，特别是在复杂、有遮挡的场景中。我们的研究结果强调，动作识别需要空间理解的支持，而不仅仅是统计模式识别。


### 论文摘要

For embodied agents to effectively understand and interact within the world around them, they require a nuanced comprehension of human actions grounded in physical space. Current action recognition models, often relying on RGB video, learn superficial correlations between patterns and action labels, so they struggle to capture underlying physical interaction dynamics and human poses in complex scenes. We propose a model architecture that grounds action recognition in physical space by fusing two powerful, complementary representations: V-JEPA 2's contextual, predictive world dynamics and CoMotion's explicit, occlusion-tolerant human pose data. Our model is validated on both the InHARD and UCF-19-Y-OCC benchmarks for general action recognition and high-occlusion action recognition, respectively. Our model outperforms three other baselines, especially within complex, occlusive scenes. Our findings emphasize a need for action recognition to be supported by spatial understanding instead of statistical pattern recognition.

---

## 12. Segmentation of Ischemic Stroke Lesions using Transfer Learning on Multi-sequence MRI

**论文链接:** [http://arxiv.org/abs/2511.07281v1](http://arxiv.org/abs/2511.07281v1)

**作者:** R. P. Chowdhury, T. Rahman

**发布时间:** 2025-11-10

**备注:** Ischemic Stroke, Segmentation, Transfer Learning, Magnetic Resonance  Imaging, Deep Learning, Res-UNet

### GPT解析

### 总结

本研究提出了一种基于Res-Unet架构的新框架，用于在各种MRI序列上快速自动分割缺血性中风病变。通过迁移学习和多数投票分类器的集成，该方法在ISLES 2015数据集上实现了80.5%的Dice分数和74.03%的准确率，展示了其有效性。

### 背景

缺血性中风病变的准确理解对中风患者的有效治疗和预后至关重要。磁共振成像（MRI）是诊断中风的常用方法，但专家手动分割病变繁琐、耗时且容易存在观察者不一致性。先前基于手工设计特征的方法无法充分捕捉病变的不规则和生理复杂形状。

### 目的

开发一种快速自动分割各种MRI序列上缺血性中风病变的新框架，以克服手动分割的挑战，并提高分割的准确性和效率。

### 方法

提出一个新框架用于在T1加权、T2加权、DWI和FLAIR等MRI序列上分割缺血性中风病变。在ISLES 2015脑中风序列数据集上使用Res-Unet架构训练模型，分两次进行：使用预训练权重和不使用预训练权重，以探索迁移学习的好处。计算Dice分数和敏感性等评估指标，并集成多数投票分类器来合并每个轴的结果。

### 主要发现

使用迁移学习（预训练权重）的模型表现更好。最终方法实现了80.5%的Dice分数和74.03%的准确率，表明该方法能有效分割缺血性中风病变。

### 结论

所提出的分割方法有效，能够准确分割缺血性中风病变，为中风患者的诊断和治疗提供了有价值的工具。

### 翻译

缺血性中风病变的准确理解对于中风患者的有效治疗和预后至关重要。磁共振成像（MRI）对急性缺血性中风敏感，是中风诊断的常用方法。然而，专家进行的手动病变分割繁琐、耗时，并且容易存在观察者不一致性。已经提出了自动医学图像分析方法来克服这一挑战。然而，先前的方法依赖于手工设计的特征，这些特征可能无法捕捉缺血性中风病变的不规则和生理复杂形状。在本研究中，我们提出了一种新框架，用于在各种MRI序列上快速自动分割缺血性中风病变，包括T1加权、T2加权、DWI和FLAIR。所提出的方法在ISLES 2015脑中风序列数据集上得到验证，我们使用Res-Unet架构两次训练我们的模型：首先使用预训练权重，然后不使用，以探索迁移学习的好处。计算了包括Dice分数和敏感性在内的评估指标，跨3D体积进行计算。最后，集成了多数投票分类器来合并每个轴的结果，形成了一种全面的分割方法。我们的努力最终实现了80.5%的Dice分数和74.03%的准确率，展示了我们分割方法的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何自动、准确地从多序列MRI图像中分割出缺血性脑卒中病灶的问题。这个问题很重要，因为缺血性脑卒中占所有脑卒中病例的80-85%，而准确理解病灶位置和范围对于有效治疗和预后评估至关重要。治疗时间窗口非常窄（通常只有4.5小时），快速准确的诊断直接影响治疗效果。目前依赖专家手动分割的方法繁琐、耗时且容易受观察者主观因素影响，而自动化方法可以克服传统方法无法捕捉病灶不规则和生理复杂形状的局限。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先指出传统手动分割方法的局限性，然后指出早期自动方法的不足在于依赖手工特征，无法捕捉病灶的复杂形状。接着提出利用深度学习特别是卷积神经网络进行自动分割的潜力，考虑到医学数据的特殊性，提出使用迁移学习。作者选择了Res-UNet架构，结合了U-Net的分割能力和ResNet的特征提取能力。为了克服3D计算成本高的问题，采用多平面2D切片方法，并使用多数投票分类器融合三个平面的预测结果。该方法借鉴了U-Net架构、ResNet特征提取器、迁移学习思想和多数投票分类器等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合迁移学习和多平面信息融合来提高缺血性脑卒中病灶分割的准确性，使用Res-UNet架构，并通过三个不同平面分别训练模型后融合结果。整体流程包括：1)数据准备，将3D MRI图像沿三个平面切分为2D切片；2)模型训练，每个平面训练一个Res-UNet模型，使用混合损失函数解决类别不平衡问题；3)预测阶段，每个模型对输入图像进行预测，生成三个平面的分割掩码；4)融合阶段，使用多数投票分类器结合三个模型的预测，生成最终的3D分割掩码；5)评估，使用Dice系数、准确性等指标评估模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)多平面信息融合策略，利用三个平面的信息提高分割准确性；2)迁移学习应用，探索了使用预训练权重的优势；3)混合损失函数，解决类别不平衡问题；4)计算效率优化，采用多平面2D切片方法。相比之前的工作，该方法不同于依赖手工特征的传统方法，不同于仅使用单一平面的方法，不同于大多数使用3D卷积的方法，并且明确评估了迁移学习在脑卒中病灶分割任务中的效果。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于迁移学习和多平面信息融合的Res-UNet框架，实现了从多序列MRI图像中自动、准确地分割缺血性脑卒中病灶，达到了80.5%的Dice分数和74.03%的准确率。'}


### 论文摘要

The accurate understanding of ischemic stroke lesions is critical for efficient therapy and prognosis of stroke patients. Magnetic resonance imaging (MRI) is sensitive to acute ischemic stroke and is a common diagnostic method for stroke. However, manual lesion segmentation performed by experts is tedious, time-consuming, and prone to observer inconsistency. Automatic medical image analysis methods have been proposed to overcome this challenge. However, previous approaches have relied on hand-crafted features that may not capture the irregular and physiologically complex shapes of ischemic stroke lesions. In this study, we present a novel framework for quickly and automatically segmenting ischemic stroke lesions on various MRI sequences, including T1-weighted, T2-weighted, DWI, and FLAIR. The proposed methodology is validated on the ISLES 2015 Brain Stroke sequence dataset, where we trained our model using the Res-Unet architecture twice: first, with pre-existing weights, and then without, to explore the benefits of transfer learning. Evaluation metrics, including the Dice score and sensitivity, were computed across 3D volumes. Finally, a Majority Voting Classifier was integrated to amalgamate the outcomes from each axis, resulting in a comprehensive segmentation method. Our efforts culminated in achieving a Dice score of 80.5\% and an accuracy of 74.03\%, showcasing the efficacy of our segmentation approach.

---

## 13. A Hybrid Autoencoder-Transformer Model for Robust Day-Ahead Electricity Price Forecasting under Extreme Conditions

**论文链接:** [http://arxiv.org/abs/2511.06898v1](http://arxiv.org/abs/2511.06898v1)

**作者:** Boyan Tang, Xuanhao Ren, Peng Xiao, Shunbo Lei, Xiaorong Sun, Jianghua Wu

**发布时间:** 2025-11-10

**备注:** Published in 2025 IEEE 1st International Symposium on the Application  of Artificial Intelligence in Electrical Engineering (AAIEE)  https://ieeexplore.ieee.org/document/11100637

### GPT解析

### 总结

本文提出了一种新型混合深度学习框架，结合蒸馏注意力变换器和自编码器自回归模型，用于解决极端条件和市场异常对日前电力价格预测的挑战。

### 背景

准确的日前电力价格预测对电力系统高效运行至关重要，但极端条件和市场异常对现有预测方法构成显著挑战。

### 目的

开发一种新型混合深度学习框架，提高电力价格预测的准确性、鲁棒性和计算效率。

### 方法

提出结合蒸馏注意力变换器(DAT)和自编码器自回归模型(ASM)的混合框架。DAT利用自注意力机制动态分配权重给历史数据关键部分，捕捉长期趋势和短期波动；ASM采用无监督学习检测和隔离由极端条件引起的异常模式。

### 主要发现

在加利福尼亚和山东省数据集上的实验表明，该框架在预测准确性、鲁棒性和计算效率方面显著优于现有最先进方法。

### 结论

该框架有望增强电网韧性并优化未来电力系统的市场运营。

### 翻译

准确的日前电力价格预测(DAEPF)对电力系统的高效运行至关重要，但极端条件和市场异常对现有预测方法构成重大挑战。为克服这些挑战，本文提出了一种新型混合深度学习框架，整合了蒸馏注意力变换器(DAT)模型和自编码器自回归模型(ASM)。DAT利用自注意力机制动态分配更高权重给历史数据的关键部分，有效捕捉长期趋势和短期波动。同时，ASM采用无监督学习检测和隔离由极端条件(如暴雨、热浪或人类节日)引起的异常模式。在加利福尼亚和山东省数据集上的实验表明，我们的框架在预测准确性、鲁棒性和计算效率方面显著优于最先进的方法。因此，该框架有望增强电网韧性并优化未来电力系统的市场运营。


### 论文摘要

Accurate day-ahead electricity price forecasting (DAEPF) is critical for the efficient operation of power systems, but extreme condition and market anomalies pose significant challenges to existing forecasting methods. To overcome these challenges, this paper proposes a novel hybrid deep learning framework that integrates a Distilled Attention Transformer (DAT) model and an Autoencoder Self-regression Model (ASM). The DAT leverages a self-attention mechanism to dynamically assign higher weights to critical segments of historical data, effectively capturing both long-term trends and short-term fluctuations. Concurrently, the ASM employs unsupervised learning to detect and isolate anomalous patterns induced by extreme conditions, such as heavy rain, heat waves, or human festivals. Experiments on datasets sampled from California and Shandong Province demonstrate that our framework significantly outperforms state-of-the-art methods in prediction accuracy, robustness, and computational efficiency. Our framework thus holds promise for enhancing grid resilience and optimizing market operations in future power systems.

---

## 14. Neyman-Pearson Classification under Both Null and Alternative Distributions Shift

**论文链接:** [http://arxiv.org/abs/2511.06641v1](http://arxiv.org/abs/2511.06641v1)

**作者:** Mohammadreza M. Kalan, Yuyang Deng, Eitan J. Neugut, Samory Kpotufe

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文研究了Neyman-Pearson分类中的迁移学习问题，提出了一种自适应程序，能够在控制两种类型错误的同时处理分布偏移问题。

### 背景

传统分类中的迁移学习已被广泛研究，但在Neyman-Pearson分类等不平衡分类场景下的迁移学习研究较少。现有工作仅考虑了μ₁分布中的偏移，而实际场景中偏移可能同时发生在μ₀和μ₁中。

### 目的

开发一种迁移学习方法，针对分布μ₁最小化误差，同时确保针对分布μ₀的误差保持在规定阈值以下，并适应源数据是否有信息量的情况。

### 方法

推导了一种自适应程序，当源数据具有信息量时保证改进的Type-I和Type-II错误，同时自动适应源数据无信息量的情况以避免负迁移。

### 主要发现

所提出的方法能够在源数据具有信息量时提供改进的错误率，同时避免在源数据无信息量时发生负迁移。

### 结论

该方法不仅提供了统计保证，还在计算上表现出高效性，解决了Neyman-Pearson分类中迁移学习的独特挑战。

### 翻译

我们考虑Neyman-Pearson分类中的迁移学习问题，其目标是在针对分布μ₀的误差保持在规定阈值以下的约束下，最小化针对分布μ₁的误差。虽然传统分类中的迁移学习已被广泛研究，但像Neyman-Pearson分类这样的不平衡分类中的迁移学习受到的关注较少。这种设置带来了独特挑战，因为必须同时控制两种类型的错误。现有工作仅处理了μ₁中的分布偏移情况，而实际场景中偏移可能同时发生在μ₀和μ₁中。我们推导了一种自适应程序，不仅能在源数据具有信息量时保证改进的Type-I和Type-II错误，还能自动适应源数据无信息量的情况，从而避免负迁移。除了这些统计保证外，该程序在计算上也表现出高效性。


### 论文摘要

We consider the problem of transfer learning in Neyman-Pearson classification, where the objective is to minimize the error w.r.t. a distribution $\mu_1$, subject to the constraint that the error w.r.t. a distribution $\mu_0$ remains below a prescribed threshold. While transfer learning has been extensively studied in traditional classification, transfer learning in imbalanced classification such as Neyman-Pearson classification has received much less attention. This setting poses unique challenges, as both types of errors must be simultaneously controlled. Existing works address only the case of distribution shift in $\mu_1$, whereas in many practical scenarios shifts may occur in both $\mu_0$ and $\mu_1$. We derive an adaptive procedure that not only guarantees improved Type-I and Type-II errors when the source is informative, but also automatically adapt to situations where the source is uninformative, thereby avoiding negative transfer. In addition to such statistical guarantees, the procedures is efficient, as shown via complementary computational guarantees.

---

## 15. TriShGAN: Enhancing Sparsity and Robustness in Multivariate Time Series Counterfactuals Explanation

**论文链接:** [http://arxiv.org/abs/2511.06529v1](http://arxiv.org/abs/2511.06529v1)

**作者:** Hongnan Ma, Yiwei Shi, Guanxiong Sun, Mengyue Yang, Weiru Liu

**发布时间:** 2025-11-09

### GPT解析

### 总结

这篇论文提出了TriShGAN方法，用于为多元时间序列数据生成更稳健的反事实解释。该方法结合了三元组损失和形状提取器，通过距离度量学习平衡最小成本和稳健性，提高了解释的稀疏性和训练效率。

### 背景

在决策过程中，利益相关者通常依赖反事实解释，这提供了关于应更改查询实例中的哪些内容以改变AI系统结果的建议。然而，为多元时间序列生成这些解释具有挑战性，因为它们具有复杂的多维性质。传统方法要么直接替换子序列（不现实），要么主要关注最小化成本而忽略将反事实解释远离决策边界的重要性。

### 目的

生成更稳健的反事实解释，使其不仅保持接近查询的时间序列，还捕获具有期望结果的实例的特征分布，从而在最小成本和稳健性之间实现更好的平衡。

### 方法

作者引入了TriShGAN，这是在CounteRGAN框架基础上通过结合三元组损失而增强的方法。这是一种无监督学习方法，使用距离度量学习来鼓励反事实解释保持接近查询的时间序列同时捕获具有期望结果的实例的特征分布。此外，整合了一个形状提取器，战略性地选择高维查询时间序列中最具判别性的部分。

### 主要发现

通过结合三元组损失和形状提取器，TriShGAN方法能够在最小成本和稳健性之间实现更好的平衡，生成更高质量的反事实解释，同时提高了训练效率和解释的稀疏性。

### 结论

TriShGAN方法通过距离度量学习和形状提取器的整合，有效地解决了多元时间序列反事实解释生成中的挑战，提供了更稳健和稀疏的解释，同时提高了训练效率。

### 翻译

在决策过程中，利益相关者通常依赖反事实解释，这提供了关于应更改查询实例中的哪些内容以改变AI系统结果的建议。然而，为多元时间序列生成这些解释具有挑战性，因为它们具有复杂的多维性质。传统的基于最近异邻域的方法通常用来自NUN的有影响力子序列替换查询时间序列中的子序列，由于严格的直接替换，这在现实场景中并不总是现实的。基于残差生成对抗网络的反事实方法旨在通过学习观测数据的分布来生成合成反事实解释来解决这一问题。然而，这些方法主要关注最小化从查询时间序列到反事实解释的成本，而常常忽略将反事实解释远离决策边界的重要性。这种疏忽可能导致在模型内发生微小变化时，解释不再符合反事实条件。为了生成更稳健的反事实解释，我们引入了TriShGAN，这是在CounteRGAN框架基础上通过结合三元组损失而增强的方法。这种无监督学习方法使用距离度量学习来鼓励反事实解释不仅保持接近查询的时间序列，还捕获具有期望结果的实例的特征分布，从而在最小成本和稳健性之间实现更好的平衡。此外，我们整合了一个形状提取器，它战略性地选择高维查询时间序列中最具判别性的部分，以提高反事实解释的稀疏性和训练过程的效率。


### 论文摘要

In decision-making processes, stakeholders often rely on counterfactual explanations, which provide suggestions about what should be changed in the queried instance to alter the outcome of an AI system. However, generating these explanations for multivariate time series presents challenges due to their complex, multi-dimensional nature. Traditional Nearest Unlike Neighbor-based methods typically substitute subsequences in a queried time series with influential subsequences from an NUN, which is not always realistic in real-world scenarios due to the rigid direct substitution. Counterfactual with Residual Generative Adversarial Networks-based methods aim to address this by learning from the distribution of observed data to generate synthetic counterfactual explanations. However, these methods primarily focus on minimizing the cost from the queried time series to the counterfactual explanations and often neglect the importance of distancing the counterfactual explanation from the decision boundary. This oversight can result in explanations that no longer qualify as counterfactual if minor changes occur within the model. To generate a more robust counterfactual explanation, we introduce TriShGAN, under the CounteRGAN framework enhanced by the incorporation of triplet loss. This unsupervised learning approach uses distance metric learning to encourage the counterfactual explanations not only to remain close to the queried time series but also to capture the feature distribution of the instance with the desired outcome, thereby achieving a better balance between minimal cost and robustness. Additionally, we integrate a Shapelet Extractor that strategically selects the most discriminative parts of the high-dimensional queried time series to enhance the sparsity of counterfactual explanation and efficiency of the training process.

---

## 16. Label-Efficient 3D Forest Mapping: Self-Supervised and Transfer Learning for Individual, Structural, and Species Analysis

**论文链接:** [http://arxiv.org/abs/2511.06331v1](http://arxiv.org/abs/2511.06331v1)

**作者:** Aldino Rizaldy, Fabian Ewald Fassnacht, Ahmed Jamal Afifi, Hua Jiang, Richard Gloaguen, Pedram Ghamisi

**发布时间:** 2025-11-09

### GPT解析

### 总结

研究探索使用自监督学习和迁移学习减少对大型标注数据集的依赖，改进从激光扫描点云中提取个体树木信息的方法，开发统一框架提高效率并减少碳排放。

### 背景

个体树木级别的详细结构和物种信息对精准林业、生物多样性保护和碳制图至关重要。激光扫描点云是获取此类信息的主要数据源，但深度学习模型需要大量标注数据，而3D点云标注在复杂森林环境中劳动密集且难以规模化。

### 目的

探索减少对大型标注数据集依赖的策略，使用自监督学习和迁移学习架构，通过真实和可操作的训练集改进实例分割、语义分割和树木分类任务。

### 方法

采用自监督学习与领域适应结合的方法进行实例分割，使用自监督学习进行语义分割，采用分层迁移学习进行树木分类，并将所有任务整合到一个统一框架中。

### 主要发现

结合自监督学习和领域适应使实例分割性能提升16.98%（AP50）；自监督学习使语义分割提升1.79%（mIoU）；分层迁移学习使未见物种分类准确率提升6.07%（Jaccard）；预训练模型减少约21%的能源消耗和碳排放。

### 结论

这项开源贡献加速了从激光扫描点云中提取个体树木信息的操作性过程，支持林业、生物多样性和碳制图，同时提高了效率和可持续性。

### 翻译

个体树木级别的详细结构和物种信息正变得越来越重要，以支持精准林业、生物多样性保护，并为生物量和碳制图提供参考数据。目前，来自机载和地面激光扫描的点云是最适合大规模快速获取此类信息的数据源。深度学习的最新进展改进了对个体树木的分割和分类以及识别语义树木组件的能力。然而，深度学习模型通常需要大量标注的训练数据，这限制了进一步的改进。为三维点云生成密集、高质量的标注，特别是在复杂的森林环境中，是劳动密集型的且难以规模化。我们探索了使用自监督学习和迁移学习架构来减少对大型标注数据集依赖的策略。我们的目标是使用真实和可操作的训练集，改进三个任务：实例分割、语义分割和树木分类的性能。我们的研究结果表明，与从头开始训练相比，结合自监督学习和领域适应显著增强了实例分割（AP50 +16.98%），自监督学习足以用于语义分割（mIoU +1.79%），分层迁移学习能够准确分类未见物种（Jaccard +6.07%）。为了简化使用并鼓励采用，我们将这些任务整合到一个统一框架中，简化了从原始点云到树木描绘、结构分析和物种分类的过程。预训练模型减少了约21%的能源消耗和碳排放。这项开源贡献旨在加速从激光扫描点云中操作性地提取个体树木信息，以支持林业、生物多样性和碳制图。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决深度学习模型在3D森林分析中需要大量标注数据的问题。在复杂森林环境中，对3D点云进行高质量标注是劳动密集型且难以规模化的。这个问题很重要，因为详细的树木结构和物种信息对精准林业、生物多样性保护和碳制图至关重要，而气候变化增加了森林干扰频率，使传统森林调查方法时效性不足。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考到深度学习需要大量标注数据，而森林环境中的点云标注特别困难。他们借鉴了自监督学习（特别是对比学习）来利用未标记数据，领域适应技术解决不同森林区域间的差异，以及分层迁移学习改进细粒度分类。设计上，他们使用对比自监督预训练学习通用特征，然后通过领域适应和分层迁移学习将这些特征应用到三个具体任务：实例分割、语义分割和树木分类。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自监督学习从大量未标记数据中学习通用特征，然后利用领域适应和分层迁移学习将这些特征应用到具体任务。整体流程：1)用掩码场景对比在大型未标记点云上预训练Sparse UNet编码器；2)对实例分割任务，添加偏移头预测点到实例中心，再用聚类算法分组；3)对语义分割，添加分类头对每个点分类；4)对树木分类，先在宽泛类别预训练，再在特定物种微调；5)通过领域适应解决不同森林区域间的差异。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的3D深度学习框架整合预训练、领域适应和分层迁移学习；2)在多样化森林类型的大规模数据上预训练；3)在极少量标注数据(如每棵树仅4-5个点)下实现有效性能；4)跨多个站点的大量验证；5)减少21%的能源消耗和碳排放。相比之前工作，不同之处在于将多种技术整合到一个框架中同时处理三个任务，在极度稀疏标注条件下实现有效性能，并提供从原始点云到最终分析的可扩展工作流程。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合自监督学习和迁移学习的统一框架，显著减少了对标注数据的依赖，实现了在有限监督下从3D点云中进行高效个体树木识别、结构分析和物种分类。'}


### 论文摘要

Detailed structural and species information on individual tree level is increasingly important to support precision forestry, biodiversity conservation, and provide reference data for biomass and carbon mapping. Point clouds from airborne and ground-based laser scanning are currently the most suitable data source to rapidly derive such information at scale. Recent advancements in deep learning improved segmenting and classifying individual trees and identifying semantic tree components. However, deep learning models typically require large amounts of annotated training data which limits further improvement. Producing dense, high-quality annotations for 3D point clouds, especially in complex forests, is labor-intensive and challenging to scale. We explore strategies to reduce dependence on large annotated datasets using self-supervised and transfer learning architectures. Our objective is to improve performance across three tasks: instance segmentation, semantic segmentation, and tree classification using realistic and operational training sets. Our findings indicate that combining self-supervised learning with domain adaptation significantly enhances instance segmentation compared to training from scratch (AP50 +16.98%), self-supervised learning suffices for semantic segmentation (mIoU +1.79%), and hierarchical transfer learning enables accurate classification of unseen species (Jaccard +6.07%). To simplify use and encourage uptake, we integrated the tasks into a unified framework, streamlining the process from raw point clouds to tree delineation, structural analysis, and species classification. Pretrained models reduce energy consumption and carbon emissions by ~21%. This open-source contribution aims to accelerate operational extraction of individual tree information from laser scanning point clouds to support forestry, biodiversity, and carbon mapping.

---

## 17. Cross-Modal Fine-Tuning of 3D Convolutional Foundation Models for ADHD Classification with Low-Rank Adaptation

**论文链接:** [http://arxiv.org/abs/2511.06163v1](http://arxiv.org/abs/2511.06163v1)

**作者:** Jyun-Ping Kao, Shinyeong Rho, Shahar Lazarev, Hyun-Hae Cho, Fangxu Xing, Taehoon Shin, C. -C. Jay Kuo, Jonghye Woo

**发布时间:** 2025-11-08

### GPT解析

### 总结

该研究提出了一种基于3D低秩适应(LoRA)的参数高效迁移学习方法，用于ADHD的神经影像学诊断，显著减少了可训练参数同时实现了高准确率。

### 背景

早期诊断儿童注意缺陷多动障碍(ADHD)对改善教育和心理健康结果至关重要。然而，使用神经影像数据诊断ADHD具有挑战性，因为其表现异质且与其他疾病症状重叠。

### 目的

提出一种参数高效的迁移学习方法，将大规模3D卷积基础模型适配到基于MRI的ADHD分类任务。

### 方法

引入3D低秩适应(LoRA)，将3D卷积核分解为2D低秩更新，减少可训练参数。在公共扩散MRI数据库上进行五折交叉验证评估。

### 主要发现

3D LoRA微调策略达到最先进结果，一个模型变体准确率达71.9%，另一个变体AUC为0.716，两者仅使用164万可训练参数（比完全微调的基础模型少113倍以上）。

### 结论

该研究代表了神经影像学中基础模型跨模态（CT到MRI）适配的首次成功之一，为ADHD分类建立了新基准并大幅提高了效率。

### 翻译

儿童注意缺陷多动障碍(ADHD)的早期诊断在改善教育和心理健康结果方面起着至关重要的作用。然而，使用神经影像数据诊断ADHD仍然具有挑战性，因为表现异质且与其他疾病症状重叠。为解决这一问题，我们提出了一种新颖的参数高效的迁移学习方法，将大规模3D卷积基础模型（在CT图像上预训练）适配到基于MRI的ADHD分类任务。我们的方法通过将3D卷积核分解为2D低秩更新引入了3D低秩适应(LoRA)，显著减少了可训练参数，同时实现了卓越性能。在公共扩散MRI数据库上的五折交叉验证评估中，我们的3D LoRA微调策略取得了最先进的结果，一个模型变体达到71.9%的准确率，另一个变体达到0.716的AUC。两个变体仅使用164万可训练参数（比完全微调的基础模型少113倍以上）。我们的结果代表了神经影像学中基础模型跨模态（CT到MRI）适配的首次成功之一，为ADHD分类建立了新基准，同时大大提高了效率。


### 论文摘要

Early diagnosis of attention-deficit/hyperactivity disorder (ADHD) in children plays a crucial role in improving outcomes in education and mental health. Diagnosing ADHD using neuroimaging data, however, remains challenging due to heterogeneous presentations and overlapping symptoms with other conditions. To address this, we propose a novel parameter-efficient transfer learning approach that adapts a large-scale 3D convolutional foundation model, pre-trained on CT images, to an MRI-based ADHD classification task. Our method introduces Low-Rank Adaptation (LoRA) in 3D by factorizing 3D convolutional kernels into 2D low-rank updates, dramatically reducing trainable parameters while achieving superior performance. In a five-fold cross-validated evaluation on a public diffusion MRI database, our 3D LoRA fine-tuning strategy achieved state-of-the-art results, with one model variant reaching 71.9% accuracy and another attaining an AUC of 0.716. Both variants use only 1.64 million trainable parameters (over 113x fewer than a fully fine-tuned foundation model). Our results represent one of the first successful cross-modal (CT-to-MRI) adaptations of a foundation model in neuroimaging, establishing a new benchmark for ADHD classification while greatly improving efficiency.

---

## 18. LLM Attention Transplant for Transfer Learning of Tabular Data Across Disparate Domains

**论文链接:** [http://arxiv.org/abs/2511.06161v1](http://arxiv.org/abs/2511.06161v1)

**作者:** Ibna Kowsar, Kazi F. Akhter, Manar D. Samad

**发布时间:** 2025-11-08

### GPT解析

### 总结

本文提出了一种名为LATTLE的新型迁移学习方法，通过将大型语言模型的选择性注意力权重移植到专门设计的表格数据模型中，解决了表格数据迁移学习中的异质性问题

### 背景

表格数据的迁移学习具有挑战性，因为不同领域间的特征空间存在异质性；传统深度学习在表格知识迁移方面表现有限；虽然大型语言模型可以提升迁移学习效果，但由于文本提示和上下文学习的限制，其在处理混合数据类型的表格时效果往往停滞不前

### 目的

开发一种轻量级迁移学习框架，消除对共享特征、LLM提示工程和大规模预训练模型的需求，有效解决表格数据间的迁移学习问题

### 方法

使用源表格数据微调LLM，将其选择性key和value投影权重移植到为表格数据设计的门控特征标记变换器(gFTT)中，然后使用目标表格数据微调具有跨域注意力的gFTT模型进行迁移学习

### 主要发现

通过十对源-目标数据集和12个基线的实验证明，LATTLE方法优于传统机器学习模型、最先进的深度表格架构以及从数千到数十亿表格样本训练的迁移学习模型；注意力迁移为在低资源学习环境中使用LLM学习表格间关系提供了有效解决方案

### 结论

所提出的注意力迁移是一种有效方法，可以在低资源环境下利用LLM学习表格数据间的关系；该方法的源代码已公开可用

### 翻译

表格数据的迁移学习具有挑战性，因为不同领域间的特征空间存在异质性。传统深度学习在表格知识迁移方面的有限成功可以通过利用大型语言模型来提升。然而，由于文本提示和上下文学习的限制，LLMs在处理表格中混合数据类型时的效果往往停滞不前。我们提出了一种轻量级迁移学习框架，使用源表格数据微调LLM，并将LLM的选择性key和value投影权重移植到为表格数据构建的门控特征标记变换器(gFTT)中。使用目标表格数据微调具有跨域注意力的gFTT模型进行迁移学习，消除对共享特征、LLM提示工程和大规模预训练模型的需求。我们使用十对源-目标数据集和12个基线进行的实验证明了所提出的LLM-注意力迁移(LATTLE)方法优于传统机器学习模型、最先进的深度表格架构以及从数千到数十亿表格样本训练的迁移学习模型。所提出的注意力迁移为在低资源学习环境中使用LLM学习表格间关系提供了一种有效解决方案。所提出方法的源代码已公开可用。


### 论文摘要

Transfer learning of tabular data is non-trivial due to heterogeneity in the feature space across disparate domains. The limited success of traditional deep learning in tabular knowledge transfer can be advanced by leveraging large language models (LLMs). However, the efficacy of LLMs often stagnates for mixed data types structured in tables due to the limitations of text prompts and in-context learning. We propose a lightweight transfer learning framework that fine-tunes an LLM using source tabular data and transplants the LLM's selective $key$ and $value$ projection weights into a gated feature tokenized transformer (gFTT) built for tabular data. The gFTT model with cross-domain attention is fine-tuned using target tabular data for transfer learning, eliminating the need for shared features, LLM prompt engineering, and large-scale pretrained models. Our experiments using ten pairs of source-target data sets and 12 baselines demonstrate the superiority of the proposed LLM-attention transplant for transfer learning (LATTLE) method over traditional ML models, state-of-the-art deep tabular architectures, and transfer learning models trained on thousands to billions of tabular samples. The proposed attention transfer demonstrates an effective solution to learning relationships between data tables using an LLM in a low-resource learning environment. The source code for the proposed method is publicly available.

---

## 19. FusionLog: Cross-System Log-based Anomaly Detection via Fusion of General and Proprietary Knowledge

**论文链接:** [http://arxiv.org/abs/2511.05878v1](http://arxiv.org/abs/2511.05878v1)

**作者:** Xinlong Zhao, Tong Jia, Minghua He, Xixuan Yang, Ying Li

**发布时间:** 2025-11-08

**备注:** 11 pages, 4 figures, and 2 tables

### GPT解析

### 总结

FusionLog是一种创新的零标记跨系统日志异常检测方法，通过融合通用知识和专有知识，实现了无需标记目标日志的跨系统异常检测，在三个不同系统的公共日志数据集上取得了超过90%的F1分数，显著优于现有方法。

### 背景

基于日志的异常检测对确保Web系统稳定性和可靠性至关重要，但缺乏足够标记日志限制了新系统的快速部署。现有方法利用成熟系统的大规模标记日志和新系统的小量标记日志，通过迁移学习提取通用知识。

### 目的

解决现有方法只关注通用知识迁移而忽视与目标系统专有知识间差异的问题，提出一种无需标记目标日志即可实现跨系统泛化的异常检测方法。

### 方法

设计基于语义相似性的无需训练路由器，动态划分未标记目标日志为'通用日志'和'专有日志'。通用日志采用系统无关表示元学习的小型模型处理，继承共享异常模式；专有日志通过迭代生成伪标签并利用大型语言模型和小型模型的多轮协作知识蒸馏与融合来微调模型。

### 主要发现

在三个不同系统的公共日志数据集上，FusionLog在完全零标记设置下实现超过90%的F1分数，性能显著优于现有最先进的跨系统日志异常检测方法。

### 结论

FusionLog有效解决了跨系统日志异常检测中标记数据不足的问题，通过通用知识和专有知识的融合，实现了高性能的零标记异常检测，具有良好的实际应用价值。

### 翻译

基于日志的异常检测对于确保Web系统的稳定性和可靠性至关重要。这项任务中的一个关键问题是缺乏足够的标记日志，这限制了在新系统中的快速部署。现有工作通常利用来自成熟Web系统的大规模标记日志和新系统的小量标记日志，使用迁移学习来提取和泛化两个领域间的通用知识。然而，这些方法只关注通用知识的迁移，忽视了这种知识与目标系统专有知识之间的差异和潜在不匹配，从而限制了性能。为解决这一局限，我们提出了FusionLog，一种创新的零标记跨系统日志异常检测方法，有效实现了通用知识和专有知识的融合，无需任何标记的目标日志即可实现跨系统泛化。具体而言，我们首先设计了一个基于语义相似性的无需训练的路由器，动态将未标记的目标日志划分为'通用日志'和'专有日志'。对于通用日志，FusionLog使用基于系统无关表示元学习的小型模型进行直接训练和推理，继承源系统和目标系统之间共享的通用异常模式。对于专有日志，我们通过迭代生成伪标签并使用基于大型语言模型和小型模型的多轮协作知识蒸馏和融合来微调小型模型，增强其识别目标系统特定异常模式的能力。在三个来自不同系统的公共日志数据集上的实验结果表明，FusionLog在完全零标记设置下实现了超过90%的F1分数，显著优于最先进的跨系统日志异常检测方法。


### 论文摘要

Log-based anomaly detection is critical for ensuring the stability and reliability of web systems. One of the key problems in this task is the lack of sufficient labeled logs, which limits the rapid deployment in new systems. Existing works usually leverage large-scale labeled logs from a mature web system and a small amount of labeled logs from a new system, using transfer learning to extract and generalize general knowledge across both domains. However, these methods focus solely on the transfer of general knowledge and neglect the disparity and potential mismatch between such knowledge and the proprietary knowledge of target system, thus constraining performance. To address this limitation, we propose FusionLog, a novel zero-label cross-system log-based anomaly detection method that effectively achieves the fusion of general and proprietary knowledge, enabling cross-system generalization without any labeled target logs. Specifically, we first design a training-free router based on semantic similarity that dynamically partitions unlabeled target logs into 'general logs' and 'proprietary logs.' For general logs, FusionLog employs a small model based on system-agnostic representation meta-learning for direct training and inference, inheriting the general anomaly patterns shared between the source and target systems. For proprietary logs, we iteratively generate pseudo-labels and fine-tune the small model using multi-round collaborative knowledge distillation and fusion based on large language model (LLM) and small model (SM) to enhance its capability to recognize anomaly patterns specific to the target system. Experimental results on three public log datasets from different systems show that FusionLog achieves over 90% F1-score under a fully zero-label setting, significantly outperforming state-of-the-art cross-system log-based anomaly detection methods.

---

## 20. ZeroLog: Zero-Label Generalizable Cross-System Log-based Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.05862v1](http://arxiv.org/abs/2511.05862v1)

**作者:** Xinlong Zhao, Tong Jia, Minghua He, Ying Li, Gang Huang

**发布时间:** 2025-11-08

**备注:** 12 pages, 17 figures, and 3 tables; accepted by ISSRE 2025

### GPT解析

### 总结

本研究提出了ZeroLog，一种系统无关的表示元学习方法，实现了零标记条件下的跨系统日志异常检测，无需目标系统的任何标记日志即可达到超过80%的F1分数。

### 背景

基于日志的异常检测是确保软件系统稳定性和可靠性的重要任务，但缺乏标记日志是一个关键问题。现有方法通常利用成熟系统的大规模标记日志通过迁移学习训练目标系统模型，但仍需目标系统的一些标记日志。

### 目的

研究零标记跨系统日志异常检测这一有价值但未被充分探索的设置，即目标系统中没有可用的标记日志。

### 方法

提出ZeroLog，利用无监督域适应在源域和目标域之间进行对抗训练，学习系统无关的通用特征表示，并通过元学习将学习到的表示推广到目标系统，无需任何目标标记。

### 主要发现

在三个不同系统的公共日志数据集上，ZeroLog在无标记情况下达到超过80%的F1分数，可与使用标记日志训练的最先进跨系统方法相媲美，并且在零标记条件下优于现有方法。

### 结论

ZeroLog成功实现了零标记条件下的跨系统日志异常检测，解决了目标系统缺乏标记日志的问题，为实际应用提供了有效的解决方案。

### 翻译

基于日志的异常检测是确保软件系统稳定性和可靠性的重要任务。此任务中的一个关键问题是缺乏标记日志。现有工作通常利用成熟系统的大规模标记日志，基于迁移学习的思想来训练目标系统的异常检测模型。然而，这些工作仍然需要目标系统一定数量的标记日志。在本文中，我们进一步研究了一个有价值但未被充分探索的设置：零标记跨系统日志异常检测，即目标系统中没有可用的标记日志。具体来说，我们提出了ZeroLog，一种系统无关的表示元学习方法，使零标记条件下的跨系统日志异常检测成为可能。为此，我们利用无监督域适应在源域和目标域之间进行对抗训练，旨在学习系统无关的通用特征表示。通过采用元学习，学习到的表示可以进一步推广到目标系统，而无需任何目标标记。在来自不同系统的三个公共日志数据集上的实验结果表明，ZeroLog在无标记情况下达到超过80%的F1分数，可与使用标记日志训练的最先进跨系统方法相媲美，并且在零标记条件下优于现有方法。


### 论文摘要

Log-based anomaly detection is an important task in ensuring the stability and reliability of software systems. One of the key problems in this task is the lack of labeled logs. Existing works usually leverage large-scale labeled logs from mature systems to train an anomaly detection model of a target system based on the idea of transfer learning. However, these works still require a certain number of labeled logs from the target system. In this paper, we take a step forward and study a valuable yet underexplored setting: zero-label cross-system log-based anomaly detection, that is, no labeled logs are available in the target system. Specifically, we propose ZeroLog, a system-agnostic representation meta-learning method that enables cross-system log-based anomaly detection under zero-label conditions. To achieve this, we leverage unsupervised domain adaptation to perform adversarial training between the source and target domains, aiming to learn system-agnostic general feature representations. By employing meta-learning, the learned representations are further generalized to the target system without any target labels. Experimental results on three public log datasets from different systems show that ZeroLog reaches over 80% F1-score without labels, comparable to state-of-the-art cross-system methods trained with labeled logs, and outperforms existing methods under zero-label conditions.

---

## 21. 论文ID: 2511.05728v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.05728v1.json'

---

## 22. Google-MedGemma Based Abnormality Detection in Musculoskeletal radiographs

**论文链接:** [http://arxiv.org/abs/2511.05600v1](http://arxiv.org/abs/2511.05600v1)

**作者:** Soumyajit Maity, Pranjal Kamboj, Sneha Maity, Rajat Singh, Sankhadeep Chatterjee

**发布时间:** 2025-11-06

**备注:** Proceedings of ICICT 2026, London, Springer (Forthcoming, February  2026; Accepted for Publication)

### GPT解析

### 总结

本研究提出了一种基于MedGemma框架的肌肉骨骼放射照片异常自动检测方法，利用现代医学基础模型提高分类性能和泛化能力。

### 背景

传统的异常检测方法主要依赖自编码器和神经网络管道，在肌肉骨骼放射照片分析中存在局限性。

### 目的

开发一种基于MedGemma的高性能异常检测系统，用于肌肉骨骼放射照片的自动分类，提高检测准确性和泛化能力。

### 方法

利用MedGemma基础模型和SigLIP衍生的视觉编码器，将预处理的X光图像编码为高维嵌入，通过轻量级多层感知器进行二元分类，并采用选择性编码器块解冻等模块化训练策略。

### 主要发现

MedGemma驱动的分类器表现出强大的性能，超过了传统卷积和基于自编码器的指标；模型利用迁移学习能力提高了泛化能力和特征工程优化。

### 结论

MedGemma驱动的分类系统可以通过提供可扩展和准确的异常检测来推进临床放射照片分类，并在自动医学图像分析中有更广泛的应用潜力。

### 翻译

本文提出了一种基于MedGemma的框架，用于肌肉骨骼放射照片的异常自动检测。与传统自编码器和神经网络管道不同，该方法利用了MedGemma基础模型，并集成了在多种医学成像模态上预训练的SigLIP衍生的视觉编码器。预处理的X光图像使用MedGemma视觉主干编码为高维嵌入，然后通过轻量级多层感知器进行二元分类。实验评估显示，MedGemma驱动的分类器表现出强大的性能，超过了传统卷积和基于自编码器的指标。此外，该模型利用了MedGemma的迁移学习能力，提高了泛化能力和特征工程优化。集成现代医学基础模型不仅增强了表示学习，还促进了模块化训练策略，如选择性编码器块解冻，以实现高效的领域适应。研究结果表明，MedGemma驱动的分类系统可以通过提供可扩展和准确的异常检测来推进临床放射照片分类，并在自动医学图像分析中有更广泛的应用潜力。关键词：Google MedGemma, MURA, 医学图像, 分类。


### 论文摘要

This paper proposes a MedGemma-based framework for automatic abnormality detection in musculoskeletal radiographs. Departing from conventional autoencoder and neural network pipelines, the proposed method leverages the MedGemma foundation model, incorporating a SigLIP-derived vision encoder pretrained on diverse medical imaging modalities. Preprocessed X-ray images are encoded into high-dimensional embeddings using the MedGemma vision backbone, which are subsequently passed through a lightweight multilayer perceptron for binary classification. Experimental assessment reveals that the MedGemma-driven classifier exhibits strong performance, exceeding conventional convolutional and autoencoder-based metrics. Additionally, the model leverages MedGemma's transfer learning capabilities, enhancing generalization and optimizing feature engineering. The integration of a modern medical foundation model not only enhances representation learning but also facilitates modular training strategies such as selective encoder block unfreezing for efficient domain adaptation. The findings suggest that MedGemma-powered classification systems can advance clinical radiograph triage by providing scalable and accurate abnormality detection, with potential for broader applications in automated medical image analysis.   Keywords: Google MedGemma, MURA, Medical Image, Classification.

---

## 23. LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs

**论文链接:** [http://arxiv.org/abs/2511.07379v1](http://arxiv.org/abs/2511.07379v1)

**作者:** Himanshu Pal, Venkata Sai Pranav Bachina, Ankit Gangwal, Charu Sharma

**发布时间:** 2025-11-10

**备注:** Accepted at AAAI 2026

### GPT解析

### 总结

LoReTTA是一种针对连续时间动态图的新型对抗攻击框架，通过两阶段方法显著降低时间图神经网络(TGNNs)的性能，同时保持不可察觉性和鲁棒性。

### 背景

时间图神经网络(TGNNs)越来越多地应用于金融预测、推荐系统和欺诈检测等高风险领域，但这些模型容易受到中毒攻击，构成严重安全风险。

### 目的

开发一种能够有效降低TGNN性能的对抗攻击方法，同时保持攻击的不可察觉性和对抗防御的鲁棒性。

### 方法

LoReTTA采用两阶段方法：首先使用16种时间重要性指标之一移除高影响力边稀疏化图；然后通过新颖的保持度数的负采样算法用对抗性负样本替换被移除的边。其插件式设计无需昂贵代理模型且符合真实的不可察觉性约束。

### 主要发现

LoReTTA在4个基准数据集和4个SotA模型上平均降低TGNN性能29.47%，具体为MOOC降低42.0%、Wikipedia降低31.5%、UCI降低28.8%、Enron降低15.6%。该方法优于11个攻击基线，对4种异常检测系统保持不可检测，并对4种SotA防御训练方法具有鲁棒性。

### 结论

LoReTTA在有效性、不可察觉性和鲁棒性方面均表现优异，是针对时间图神经网络的强大对抗攻击框架。

### 翻译

时间图神经网络(TGNNs)越来越多地用于高风险领域，如金融预测、推荐系统和欺诈检测。然而，它们对中毒攻击的敏感性构成了严重的安全风险。我们引入LoReTTA（低资源两阶段时间攻击），一种针对连续时间动态图的新型对抗框架，该框架在4个广泛基准数据集和4个最先进模型上平均使TGNN性能降低29.47%。LoReTTA通过两阶段方法运行：(1)使用任何16种测试的时间重要性指标移除高影响力边来稀疏化图，(2)通过LoReTTA新颖的保持度数的负采样算法用对抗性负样本战略性地替换被移除的边。我们的插件式设计消除了对昂贵代理模型的需求，同时遵循真实的不可察觉性约束。LoReTTA在MOOC上降低性能42.0%，在Wikipedia上降低31.5%，在UCI上降低28.8%，在Enron上降低15.6%。LoReTTA优于11个攻击基线，对4种领先的异常检测系统保持不可检测，并对4种SotA对抗防御训练方法具有鲁棒性，确立了其有效性、不可察觉性和鲁棒性。


### 论文摘要

Temporal Graph Neural Networks (TGNNs) are increasingly used in high-stakes domains, such as financial forecasting, recommendation systems, and fraud detection. However, their susceptibility to poisoning attacks poses a critical security risk. We introduce LoReTTA (Low Resource Two-phase Temporal Attack), a novel adversarial framework on Continuous-Time Dynamic Graphs, which degrades TGNN performance by an average of 29.47% across 4 widely benchmark datasets and 4 State-of-the-Art (SotA) models. LoReTTA operates through a two-stage approach: (1) sparsify the graph by removing high-impact edges using any of the 16 tested temporal importance metrics, (2) strategically replace removed edges with adversarial negatives via LoReTTA's novel degree-preserving negative sampling algorithm. Our plug-and-play design eliminates the need for expensive surrogate models while adhering to realistic unnoticeability constraints. LoReTTA degrades performance by upto 42.0% on MOOC, 31.5% on Wikipedia, 28.8% on UCI, and 15.6% on Enron. LoReTTA outperforms 11 attack baselines, remains undetectable to 4 leading anomaly detection systems, and is robust to 4 SotA adversarial defense training methods, establishing its effectiveness, unnoticeability, and robustness.

---

## 24. MG-HGNN: A Heterogeneous GNN Framework for Indoor Wi-Fi Fingerprint-Based Localization

**论文链接:** [http://arxiv.org/abs/2511.07282v1](http://arxiv.org/abs/2511.07282v1)

**作者:** Yibu Wang, Zhaoxin Zhang, Ning Li, Xinlong Zhao, Dong Zhao, Tianzi Zhao

**发布时间:** 2025-11-10

**备注:** 16 pages, 11 figures, 11 tables

### GPT解析

### 总结

该研究提出了一种名为MG-HGNN的多图异质图神经网络框架，用于提高基于接收信号强度指示器(RSSI)的室内定位准确性。该框架通过两个图构建分支进行节点和边嵌入，然后使用异质图神经网络进行图表示学习，从而增强空间感知能力和定位性能。

### 背景

接收信号强度指示器(RSSI)是Wi-Fi指纹的主要表现形式，并在室内定位中扮演着关键角色。然而，现有的基于RSSI的定位方法常常由于环境复杂性以及多源信息处理方面的挑战而导致准确性降低。

### 目的

解决现有RSSI定位方法在复杂环境下准确性的降低问题，以及多源信息处理的挑战，提高室内定位的精度和性能。

### 方法

提出了一种多图异质GNN框架(MG-HGNN)，该框架包含两个图构建分支分别进行节点和边嵌入，以生成信息丰富的图。随后，使用异质图神经网络进行图表示学习，实现准确定位。主要创新包括：1)多类型任务导向的图构建，结合标签估计和特征编码以获取更丰富的图信息；2)异质GNN结构，增强传统GNN模型的性能。

### 主要发现

在UJIIndoorLoc和UTSIndoorLoc公共数据集上的评估表明，MG-HGNN不仅比几种最先进的方法实现了更优的性能，而且为增强基于GNN的定位方法提供了新的视角。消融研究进一步证实了所提出框架的合理性和有效性。

### 结论

MG-HGNN框架通过创新的图构建方法和异质GNN结构，有效解决了复杂环境下基于RSSI的室内定位准确性降低的问题，为室内定位技术提供了新的解决方案和研究方向。

### 翻译

接收信号强度指示器(RSSI)是Wi-Fi指纹的主要表现形式，并在室内定位中扮演着关键角色。然而，现有的基于RSSI的定位方法常常由于环境复杂性以及多源信息处理方面的挑战而导致准确性降低。为解决这些问题，我们提出了一种新颖的多图异质GNN框架(MG-HGNN)，以增强空间感知能力和提高定位性能。在该框架中，两个图构建分支分别进行节点和边嵌入，以生成信息丰富的图。随后，采用异质图神经网络进行图表示学习，实现准确定位。MG-HGNN框架引入了以下关键创新：1)多类型任务导向的图构建，结合标签估计和特征编码以获取更丰富的图信息；2)异质GNN结构，增强传统GNN模型的性能。在UJIIndoorLoc和UTSIndoorLoc公共数据集上的评估表明，MG-HGNN不仅比几种最先进的方法实现了更优的性能，而且为增强基于GNN的定位方法提供了新的视角。消融研究进一步证实了所提出框架的合理性和有效性。


### 论文摘要

Received signal strength indicator (RSSI) is the primary representation of Wi-Fi fingerprints and serves as a crucial tool for indoor localization. However, existing RSSI-based positioning methods often suffer from reduced accuracy due to environmental complexity and challenges in processing multi-source information. To address these issues, we propose a novel multi-graph heterogeneous GNN framework (MG-HGNN) to enhance spatial awareness and improve positioning performance. In this framework, two graph construction branches perform node and edge embedding, respectively, to generate informative graphs. Subsequently, a heterogeneous graph neural network is employed for graph representation learning, enabling accurate positioning. The MG-HGNN framework introduces the following key innovations: 1) multi-type task-directed graph construction that combines label estimation and feature encoding for richer graph information; 2) a heterogeneous GNN structure that enhances the performance of conventional GNN models. Evaluations on the UJIIndoorLoc and UTSIndoorLoc public datasets demonstrate that MG-HGNN not only achieves superior performance compared to several state-of-the-art methods, but also provides a novel perspective for enhancing GNN-based localization methods. Ablation studies further confirm the rationality and effectiveness of the proposed framework.

---

## 25. On Stealing Graph Neural Network Models

**论文链接:** [http://arxiv.org/abs/2511.07170v1](http://arxiv.org/abs/2511.07170v1)

**作者:** Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pręgowska, Tomasz P. Michalak

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文提出了一种在查询数量极其有限的情况下提取图神经网络(GNN)模型的方法，即使存在防御措施也能有效实施。

### 背景

当前GNN模型窃取方法严重依赖对目标模型的查询，假设没有严格的查询限制，但现实中查询数量可能受到严格限制。

### 目的

展示如何通过非常有限的模型交互来提取GNN模型。

### 方法

首先使攻击者能够无需直接查询目标模型就获取模型骨干，然后战略性地利用固定的查询限制来提取最信息丰富的数据。

### 主要发现

在八个真实世界数据集上的实验证明了该方法的有效性，即使在非常有限的查询限制下，并且在已有防御模型提取措施的情况下也是如此。

### 结论

研究结果强调了需要对GNN模型提取威胁采取稳健防御的必要性。

### 翻译

当前的图神经网络(GNN)模型窃取方法严重依赖对目标模型的查询，假设没有严格的查询限制。然而，实际上允许的查询数量可能会受到严格限制。在本文中，我们展示了攻击者如何通过与模型非常有限的交互来提取GNN。我们的方法首先使攻击者能够在不直接查询目标模型的情况下获取模型骨干，然后战略性地利用固定的查询限制来提取最信息丰富的数据。在八个真实世界数据集上的实验证明了攻击的有效性，即使在非常有限的查询限制下，并且在已有防御模型提取措施的情况下也是如此。我们的研究结果强调了对GNN模型提取威胁需要采取稳健防御的必要性。


### 论文摘要

Current graph neural network (GNN) model-stealing methods rely heavily on queries to the victim model, assuming no hard query limits. However, in reality, the number of allowed queries can be severely limited. In this paper, we demonstrate how an adversary can extract the GNN with very limited interactions with the model. Our approach first enables the adversary to obtain the model backbone without making direct queries to the victim model and then to strategically utilize a fixed query limit to extract the most informative data. The experiments on eight real-world datasets demonstrate the effectiveness of the attack, even under a very restricted query limit and under defense against model extraction in place. Our findings underscore the need for robust defenses against GNN model extraction threats.

---

## 26. Direct Molecular Polarizability Prediction with SO(3) Equivariant Local Frame GNNs

**论文链接:** [http://arxiv.org/abs/2511.07087v1](http://arxiv.org/abs/2511.07087v1)

**作者:** Jean Philip Filling, Felix Post, Michael Wand, Denis Andrienko

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文介绍了一种新型的等变图神经网络架构，用于预测分子的张量响应性质，通过局部坐标系保持SO(3)-等变性，并在分子极化率预测中展示了优于传统方法的性能。

### 背景

传统图神经网络框架主要专注于回归标量量并从导数中推导张量性质，缺乏对分子几何结构的有效捕捉能力。

### 目的

开发一种能够直接预测分子张量响应性质的神经网络架构，保持旋转等变性并有效捕获几何信息。

### 方法

设计了一种新型的等变图神经网络，通过使用局部坐标系保持SO(3)-等变性，并在局部消息传递框架内整合标量、向量和张量通道来捕获几何信息。

### 主要发现

在QM7-X数据集上的实验表明，张量消息传递模型在预测分子极化率方面优于标量消息传递模型。

### 结论

这项工作代表了向开发用于分子性质预测的结构化、几何感知神经模型的重要进展，为分子性质预测提供了新的方法。

### 翻译

我们介绍了一种新型的等变图神经网络架构，旨在预测分子的张量响应性质。与专注于回归标量量并从导数中推导张量性质的传统框架不同，我们的方法通过使用局部坐标系来保持SO(3)-等变性。我们的GNN通过在局部消息传递框架内整合标量、向量和张量通道，有效地捕获了几何信息。为了评估模型的准确性，我们将其应用于预测QM7-X数据集中分子的极化率，并表明张量消息传递优于标量消息传递模型。这项工作朝着开发用于分子性质预测的结构化、几何感知神经模型迈出了一步。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何直接预测分子的极化率张量问题。极化率描述分子电子云对外部电场的响应能力，决定了分子间相互作用和介电行为，对理解材料光学性质和介电特性至关重要。传统方法通常通过预测标量量再求导获得张量，不够准确；而密度泛函理论虽然准确但计算成本高，尤其对大分子系统，因此需要高效的机器学习替代方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了现有的等变架构，如张量场网络、Cormorant等，这些工作使用球谐函数和Clebsch-Gordan张量积编码旋转对称性。但作者注意到这些方法通常通过标量导数获得张量，而非直接回归。作者还受局部参考帧概念启发，特别是Lippmann等人的工作，但改进之处在于保持独立的标量、向量和张量通道，并允许它们之间的显式控制，通过设计专门的张量消息传递机制来捕获方向性相互作用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用局部参考帧保持SO(3)等变性，同时结合标量、向量和张量通道捕获分子几何信息。每个原子都有自己的局部坐标系，用于表示和交换张量特征。实现流程：1)为每个原子构建电荷加权PCA局部参考帧；2)初始化原子特征(标量、向量、张量)；3)通过相对旋转在节点间传递特征；4)使用边缘MLP混合张量消息；5)进行邻域聚合；6)预测每个节点的局部3×3贡献；7)旋转到全局坐标并池化，得到最终分子极化率张量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)直接预测极化率张量而非通过标量导数获得；2)使用局部参考帧保持SO(3)等变性，同时处理多种类型特征；3)设计专门的张量消息传递机制；4)在消息传递过程中保持特征形状。相比之前工作的不同：不同于传统框架回归标量再导出张量，本文直接回归秩-2张量；与Lippmann等人不同，本文保持独立通道并允许显式交互；本文通过张量消息传递捕获方向性相互作用，超越了仅使用标量消息的方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于SO(3)等变局部参考帧图神经网络的新架构，通过结合标量、向量和张量通道的消息传递机制，实现了比传统方法更准确地直接预测分子极化率张量。'}


### 论文摘要

We introduce a novel equivariant graph neural network (GNN) architecture designed to predict the tensorial response properties of molecules. Unlike traditional frameworks that focus on regressing scalar quantities and derive tensorial properties from their derivatives, our approach maintains $SO(3)$-equivariance through the use of local coordinate frames. Our GNN effectively captures geometric information by integrating scalar, vector, and tensor channels within a local message-passing framework. To assess the accuracy of our model, we apply it to predict the polarizabilities of molecules in the QM7-X dataset and show that tensorial message passing outperforms scalar message passing models. This work marks an advancement towards developing structured, geometry-aware neural models for molecular property prediction.

---

## 27. CGLE: Class-label Graph Link Estimator for Link Prediction

**论文链接:** [http://arxiv.org/abs/2511.06982v1](http://arxiv.org/abs/2511.06982v1)

**作者:** Ankit Mazumder, Srikanta Bedathur

**发布时间:** 2025-11-10

**备注:** Paper accepted at the IEEE International Conference on Data Mining  (ICDM 2025)

### GPT解析

### 总结

本文提出了CGLE(Class-label Graph Link Estimator)框架，通过整合类级别语义信息来增强图神经网络在链接预测任务中的表现，在多个基准数据集上实现了显著的性能提升。

### 背景

链接预测是图挖掘中的关键任务，在社交网络、推荐系统和知识图谱补全等领域有广泛应用。然而，许多领先的图神经网络模型往往忽略了在类级别聚合的有价值语义信息。

### 目的

解决现有GNN模型忽略类级别语义信息的局限性，提出一种新颖的框架来增强基于GNN的链接预测模型。

### 方法

CGLE通过构建类条件链接概率矩阵，该矩阵可从真实标签或聚类获得的伪标签中推导。然后将基于类的先验与骨干GNN的结构链接嵌入连接，使用多层感知器处理组合表示进行最终预测。该方法封装在高效预处理阶段，不影响底层GNN模型的计算复杂度。

### 主要发现

在广泛的基准数据集上验证了该方法，包括同配性和稀疏异配性图。与NCN和NCNC等强基线相比，CGLE在PubMed和DBLP等同配性数据集上HR@100提高了10多个百分点，在Chameleon稀疏异配性数据集上MRR提高了4%以上。

### 结论

集成全局、数据驱动的语义先验是有效的，为追求越来越复杂的模型架构提供了有吸引力的替代方案。

### 翻译

链接预测是图挖掘中的一个关键任务，在社交网络、推荐系统和知识图谱补全中有广泛应用。然而，许多领先的图神经网络模型通常忽略了在类级别聚合的有价值语义信息。为解决这一局限，本文引入了CGLE(类标签图链接估计器)，一个旨在增强基于GNN的链接预测模型的新颖框架。CGLE通过构建类条件链接概率矩阵来运作，其中每个条目代表两个节点类别之间形成链接的概率。该矩阵从可用的真实标签或通过聚类获得的伪标签中推导得出。然后，将基于类的先验与来自骨干GNN的结构链接嵌入连接，由多层感知器处理组合表示进行最终预测。关键的是，CGLE的逻辑封装在高效的预处理阶段，不影响底层GNN模型的计算复杂度。我们在广泛的基准数据集套件上验证了我们的方法，涵盖了同配性和稀疏异配性图。结果表明，CGLE与NCN和NCNC等强基线相比取得了显著的性能提升，在PubMed和DBLP等同配性数据集上HR@100提高了10多个百分点。在稀疏异配性图上，CGLE在Chameleon数据集上实现了MRR超过4%的提升。我们的工作强调了集成全局、数据驱动的语义先验的有效性，为追求越来越复杂的模型架构提供了有吸引力的替代方案。重现我们研究成果的代码可在https://github.com/data-iitd/cgle-icdm2025获取。


### 论文摘要

Link prediction is a pivotal task in graph mining with wide-ranging applications in social networks, recommendation systems, and knowledge graph completion. However, many leading Graph Neural Network (GNN) models often neglect the valuable semantic information aggregated at the class level. To address this limitation, this paper introduces CGLE (Class-label Graph Link Estimator), a novel framework designed to augment GNN-based link prediction models. CGLE operates by constructing a class-conditioned link probability matrix, where each entry represents the probability of a link forming between two node classes. This matrix is derived from either available ground-truth labels or from pseudo-labels obtained through clustering. The resulting class-based prior is then concatenated with the structural link embedding from a backbone GNN, and the combined representation is processed by a Multi-Layer Perceptron (MLP) for the final prediction. Crucially, CGLE's logic is encapsulated in an efficient preprocessing stage, leaving the computational complexity of the underlying GNN model unaffected. We validate our approach through extensive experiments on a broad suite of benchmark datasets, covering both homophilous and sparse heterophilous graphs. The results show that CGLE yields substantial performance gains over strong baselines such as NCN and NCNC, with improvements in HR@100 of over 10 percentage points on homophilous datasets like Pubmed and DBLP. On sparse heterophilous graphs, CGLE delivers an MRR improvement of over 4% on the Chameleon dataset. Our work underscores the efficacy of integrating global, data-driven semantic priors, presenting a compelling alternative to the pursuit of increasingly complex model architectures. Code to reproduce our findings is available at: https://github.com/data-iitd/cgle-icdm2025.

---

## 28. Dual Mamba for Node-Specific Representation Learning: Tackling Over-Smoothing with Selective State Space Modeling

**论文链接:** [http://arxiv.org/abs/2511.06756v1](http://arxiv.org/abs/2511.06756v1)

**作者:** Xin He, Yili Wang, Yiwei Dai, Xin Wang

**发布时间:** 2025-11-10

**备注:** 11 pages, 4 figures

### GPT解析

### 总结

本文提出了一种双Mamba增强图卷积网络(DMbaGCN)，通过整合Mamba到GNNs中，从局部和全局两个角度解决深度图神经网络中的过度平滑问题。

### 背景

深度图神经网络中存在过度平滑的基本挑战，重复的消息传递导致节点表示变得不可区分。现有解决方案如残差连接和跳跃层虽然在一定程度上缓解了这个问题，但存在局限性。

### 目的

解决现有方法无法明确建模节点表示在层间的节点特定和渐进式演化，以及不考虑全局信息的问题。

### 方法

提出DMbaGCN框架，包含两个模块：局部状态演化Mamba(LSEMba)用于局部邻域聚合和捕获节点特定表示动态；全局上下文感知Mamba(GCAMba)利用Mamba的全局注意力能力为每个节点整合全局上下文。

### 主要发现

DMbaGCN通过结合这些组件，增强了深度GNNs中节点的区分性，从而缓解了过度平滑问题。

### 结论

在多个基准测试上的大量实验证明了DMbaGCN方法的有效性和效率。

### 翻译

过度平滑仍然是深度图神经网络(GNNs)中的一个基本挑战，其中重复的消息传递导致节点表示变得不可区分。虽然现有的解决方案，如残差连接和跳跃层，在一定程度上缓解了这个问题，但它们未能明确地以节点特定和渐进的方式建模节点表示在层之间的演化。此外，这些方法没有考虑全局信息，这对于缓解过度平滑问题也至关重要。为了解决上述问题，在本文中，我们提出了一个双Mamba增强图卷积网络(DMbaGCN)，这是一个将Mamba整合到GNNs中的新框架，从局部和全局两个角度解决过度平滑问题。DMbaGCN包含两个模块：局部状态演化Mamba(LSEMba)用于局部邻域聚合，并利用Mamba的选择性状态空间建模来捕获跨层的节点特定表示动态；全局上下文感知Mamba(GCAMba)利用Mamba的全局注意力能力为每个节点整合全局上下文。通过结合这些组件，DMbaGCN增强了深度GNNs中节点的区分性，从而缓解了过度平滑问题。在多个基准测试上的大量实验证明了我们方法的有效性和效率。


### 论文摘要

Over-smoothing remains a fundamental challenge in deep Graph Neural Networks (GNNs), where repeated message passing causes node representations to become indistinguishable. While existing solutions, such as residual connections and skip layers, alleviate this issue to some extent, they fail to explicitly model how node representations evolve in a node-specific and progressive manner across layers. Moreover, these methods do not take global information into account, which is also crucial for mitigating the over-smoothing problem. To address the aforementioned issues, in this work, we propose a Dual Mamba-enhanced Graph Convolutional Network (DMbaGCN), which is a novel framework that integrates Mamba into GNNs to address over-smoothing from both local and global perspectives. DMbaGCN consists of two modules: the Local State-Evolution Mamba (LSEMba) for local neighborhood aggregation and utilizing Mamba's selective state space modeling to capture node-specific representation dynamics across layers, and the Global Context-Aware Mamba (GCAMba) that leverages Mamba's global attention capabilities to incorporate global context for each node. By combining these components, DMbaGCN enhances node discriminability in deep GNNs, thereby mitigating over-smoothing. Extensive experiments on multiple benchmarks demonstrate the effectiveness and efficiency of our method.

---

## 29. S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning

**论文链接:** [http://arxiv.org/abs/2511.06727v1](http://arxiv.org/abs/2511.06727v1)

**作者:** Jiangwen Dong, Zehui Lin, Wanyu Lin, Mingjin Zhang

**发布时间:** 2025-11-10

### GPT解析

### 总结

该研究提出了一种新型框架，通过在学科层面进行细粒度分析，并配备专门的多智能体协作策略来解决异构问题推理。研究构建了基于学科的有向无环图(S-DAG)，并根据模型的专业评分选择最佳匹配模型，实现图结构的多智能体协作。实验表明该方法在准确性和效率上均优于现有基线。

### 背景

大型语言模型在复杂推理问题上已取得显著成效，但其有效性高度依赖于任务的具体性质，特别是所需的领域知识。现有方法如专家混合模型通常在任务级别操作，对于涉及多个学科的异构问题来说过于粗糙。

### 目的

开发一种能够有效解决涉及多学科的复杂推理问题的框架，通过在学科层面进行细粒度分析和多智能体协作，提高模型在复杂多学科问题上的推理能力。

### 方法

1. 使用图神经网络识别输入查询的相关学科并推断相互依赖关系，生成基于学科的有向无环图(S-DAG)；2. 为每个模型分配学科特定的专业评分，选择最佳模型匹配S-DAG中的相应学科；3. 实现基于图结构的多智能体协作，信息从起始模型通过S-DAG流向结束模型。

### 主要发现

1. 整理并发布了标准基准测试(MMLU-Pro, GPQA, MedMCQA)的多学科子集，更好地反映复杂真实的推理任务；2. 大量实验表明，该方法在准确性和效率上都显著优于现有的任务级模型选择和多智能体协作基线。

### 结论

这些结果强调了在解决复杂和多学科问题时，学科感知推理和结构化协作的有效性。

### 翻译

大型语言模型在复杂推理问题上已取得了令人印象深刻的性能。它们的有效性高度依赖于任务的具体性质，特别是所需的领域知识。现有方法，如专家混合模型，通常在任务级别操作；对于涉及多个学科的异构问题来说，它们过于粗糙。本文提出了一种新型框架，在学科层面进行细粒度分析，并配备专门的多智能体协作策略来解决异构问题推理。具体来说，给定一个输入查询，我们首先使用图神经网络识别相关学科并推断它们之间的相互依赖关系，生成一个基于学科的有向无环图(S-DAG)，其中节点代表学科，边编码信息流。然后，我们通过为每个模型分配学科特定的专业评分来分析大型语言模型，并为S-DAG的相应学科选择表现最佳的模型。这种学科-模型匹配实现了图结构的多智能体协作，信息从起始模型通过S-DAG流向结束模型。我们整理并发布了标准基准测试(MMLU-Pro, GPQA, MedMCQA)的多学科子集，以更好地反映复杂、真实的推理任务。大量实验表明，我们的方法在准确性和效率上都显著优于现有的任务级模型选择和多智能体协作基线。这些结果强调了在解决复杂和多学科问题时，学科感知推理和结构化协作的有效性。


### 论文摘要

Large Language Models (LLMs) have achieved impressive performance in complex reasoning problems. Their effectiveness highly depends on the specific nature of the task, especially the required domain knowledge. Existing approaches, such as mixture-of-experts, typically operate at the task level; they are too coarse to effectively solve the heterogeneous problems involving multiple subjects. This work proposes a novel framework that performs fine-grained analysis at subject level equipped with a designated multi-agent collaboration strategy for addressing heterogeneous problem reasoning. Specifically, given an input query, we first employ a Graph Neural Network to identify the relevant subjects and infer their interdependencies to generate an \textit{Subject-based Directed Acyclic Graph} (S-DAG), where nodes represent subjects and edges encode information flow. Then we profile the LLM models by assigning each model a subject-specific expertise score, and select the top-performing one for matching corresponding subject of the S-DAG. Such subject-model matching enables graph-structured multi-agent collaboration where information flows from the starting model to the ending model over S-DAG. We curate and release multi-subject subsets of standard benchmarks (MMLU-Pro, GPQA, MedMCQA) to better reflect complex, real-world reasoning tasks. Extensive experiments show that our approach significantly outperforms existing task-level model selection and multi-agent collaboration baselines in accuracy and efficiency. These results highlight the effectiveness of subject-aware reasoning and structured collaboration in addressing complex and multi-subject problems.

---

## 30. Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.06696v1](http://arxiv.org/abs/2511.06696v1)

**作者:** Dian Jin, Yancheng Yuan, Xiaoming Tao

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文提出了一种新的等变微调方法MMEA，通过轻量级标量门控按阶和多重性调制特征幅度，在保持严格等变性的同时，提高了能量和力预测性能，且训练参数更少。

### 背景

基于球谐函数的预训练等变图神经网络为计算密集型从头计算方法提供了高效准确的替代方案，但适应新任务需要微调。传统PEFT技术如Adapters和LoRA会破坏对称性，与等变架构不兼容。ELoRA作为首个等变PEFT方法，保留了较高的张量阶自由度，可能扰动预训练特征分布。

### 目的

解决ELoRA保留高自由度导致的问题，提出一种不破坏对称性的等变微调方法。

### 方法

提出幅度调制等变适配器（MMEA），采用轻量级标量门控来按阶和多重性调制特征幅度，保持严格的等变性。

### 主要发现

MMEA在多个基准测试中持续提高了能量和力预测性能，达到最先进水平，同时训练的参数比竞争方法更少。

### 结论

在许多实际场景中，调制通道幅度足以使等变模型适应新的化学环境而不会破坏对称性，为等变PEFT设计指明了新范式。

### 翻译

基于球谐函数的预训练等变图神经网络为计算密集型的从头计算方法提供了高效准确的替代方案，但适应它们到新任务和化学环境仍需要微调。传统的参数高效微调技术如Adapters和LoRA通常会破坏对称性，使其与那些等变架构不兼容。最近提出的ELoRA是首个等变PEFT方法，它在许多基准测试中实现了改进的参数效率和性能。然而，它在每个张量阶中保留的相对较高的自由度仍然会扰动预训练的特征分布并最终降低性能。为解决这一问题，我们提出了幅度调制等变适配器（MMEA），一种新的等变微调方法，它采用轻量级标量门控来按阶和多重性调制特征幅度。我们证明了MMEA保持了严格的等变性，并且在多个基准测试中，持续改进了能量和力预测到最先进水平，同时训练的参数比竞争方法更少。这些结果表明，在许多实际场景中，调制通道幅度足以使等变模型适应新的化学环境而不会破坏对称性，为等变PEFT设计指明了新的范式。


### 论文摘要

Pretrained equivariant graph neural networks based on spherical harmonics offer efficient and accurate alternatives to computationally expensive ab-initio methods, yet adapting them to new tasks and chemical environments still requires fine-tuning. Conventional parameter-efficient fine-tuning (PEFT) techniques, such as Adapters and LoRA, typically break symmetry, making them incompatible with those equivariant architectures. ELoRA, recently proposed, is the first equivariant PEFT method. It achieves improved parameter efficiency and performance on many benchmarks. However, the relatively high degrees of freedom it retains within each tensor order can still perturb pretrained feature distributions and ultimately degrade performance. To address this, we present Magnitude-Modulated Equivariant Adapter (MMEA), a novel equivariant fine-tuning method which employs lightweight scalar gating to modulate feature magnitudes on a per-order and per-multiplicity basis. We demonstrate that MMEA preserves strict equivariance and, across multiple benchmarks, consistently improves energy and force predictions to state-of-the-art levels while training fewer parameters than competing approaches. These results suggest that, in many practical scenarios, modulating channel magnitudes is sufficient to adapt equivariant models to new chemical environments without breaking symmetry, pointing toward a new paradigm for equivariant PEFT design.

---

## 31. GNN-Enabled Robust Hybrid Beamforming with Score-Based CSI Generation and Denoising

**论文链接:** [http://arxiv.org/abs/2511.06663v1](http://arxiv.org/abs/2511.06663v1)

**作者:** Yuhang Li, Yang Lu, Bo Ai, Zhiguo Ding, Dusit Niyato, Arumugam Nallanathan

**发布时间:** 2025-11-10

### GPT解析

### 总结

该研究提出利用图神经网络和基于分数的生成模型解决不完美信道状态信息(CSI)条件下的混合波束成形(HBF)问题，开发了HMGAT、NCSN和DeBERT三种模型，在DeepMIMO数据集上验证了其优越性能。

### 背景

准确的信道状态信息(CSI)对于混合波束成形(HBF)任务至关重要，但在实际无线通信系统中获取高分辨率的CSI仍然具有挑战性。

### 目的

解决在不完美CSI条件下实现鲁棒HBF的问题，提高无线通信系统的性能和可靠性。

### 方法

1. 开发混合消息图注意力网络(HMGAT)，通过节点级和边级消息传递更新节点和边特征；2. 设计基于BERT的噪声条件分数网络(NCSN)，学习高分辨率CSI分布并促进数据增强；3. 提出去噪分数网络(DSN)框架及其实现DeBERT，可在任意信道错误级别下对不完美CSI去噪。

### 主要发现

在DeepMIMO城市数据集上的实验表明，所提模型在各种HBF任务中具有优越的泛化能力、可扩展性和鲁棒性，适用于完美和不完美CSI条件。

### 结论

结合图神经网络和基于分数的生成模型可以有效解决不完美CSI条件下的混合波束成形问题，提高系统鲁棒性和性能。

### 翻译

准确的信道状态信息(CSI)对于混合波束成形(HBF)任务至关重要。然而，在实际无线通信系统中，获取高分辨率的CSI仍然具有挑战性。为了解决这个问题，我们提议利用图神经网络(GNNs)和基于分数的生成模型，在不完美CSI条件下实现鲁棒的HBF。首先，我们开发了混合消息图注意力网络(HMGAT)，通过节点级和边级消息传递来更新节点和边特征。其次，我们设计了一个基于Transformer的双向编码器表示(BERT)的噪声条件分数网络(NCSN)，用于学习高分辨率CSI的分布，促进CSI生成和数据增强，从而进一步提高HMGAT的性能。最后，我们提出了一个去噪分数网络(DSN)框架及其实现形式，称为DeBERT，它可以在任意信道错误级别下对不完美的CSI去噪，从而促进鲁棒的HBF。在DeepMIMO城市数据集上的实验证明了所提出的模型在各种HBF任务中具有优越的泛化能力、可扩展性和鲁棒性，无论是在完美还是不完美的CSI条件下。


### 论文摘要

Accurate Channel State Information (CSI) is critical for Hybrid Beamforming (HBF) tasks. However, obtaining high-resolution CSI remains challenging in practical wireless communication systems. To address this issue, we propose to utilize Graph Neural Networks (GNNs) and score-based generative models to enable robust HBF under imperfect CSI conditions. Firstly, we develop the Hybrid Message Graph Attention Network (HMGAT) which updates both node and edge features through node-level and edge-level message passing. Secondly, we design a Bidirectional Encoder Representations from Transformers (BERT)-based Noise Conditional Score Network (NCSN) to learn the distribution of high-resolution CSI, facilitating CSI generation and data augmentation to further improve HMGAT's performance. Finally, we present a Denoising Score Network (DSN) framework and its instantiation, termed DeBERT, which can denoise imperfect CSI under arbitrary channel error levels, thereby facilitating robust HBF. Experiments on DeepMIMO urban datasets demonstrate the proposed models' superior generalization, scalability, and robustness across various HBF tasks with perfect and imperfect CSI.

---

## 32. Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning

**论文链接:** [http://arxiv.org/abs/2511.06633v1](http://arxiv.org/abs/2511.06633v1)

**作者:** Qinghong Guo, Yu Wang, Ji Cao, Tongya Zheng, Junshu Dai, Bingde Hu, Shunyu Liu, Canghong Jin

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文提出了一种名为DST的双分支时空自监督表示框架，用于解决道路网络表示学习中的空间异质性和时间动态性问题。

### 背景

道路网络表示学习随着各种时空任务的涌现而受到越来越多的关注，现有方法主要利用图神经网络和对比学习来表征路段的空间结构。

### 目的

解决道路网络的空间异质性和时间动态性对自监督GNN邻域平滑机制带来的挑战，提高道路网络表示的效果。

### 方法

DST框架包含空间和时间两个分支：空间分支设计混合跳转过渡矩阵用于图卷积，并通过对比普通道路网络与超图的表示来捕捉长程关系；时间分支使用因果Transformer在交通动态序列上进行下一个令牌预测，并通过区分工作日和周末的交通模式进行正则化。

### 主要发现

与最先进方法的广泛实验验证了DST框架的优越性，全面的时空建模使DST在零学习场景中表现出色。

### 结论

DST框架通过双分支时空自监督表示有效解决了道路网络表示学习中的挑战，在各种应用场景中具有优越性能。

### 翻译

道路网络表示学习随着各种时空任务的涌现而受到研究人员和从业者的越来越多的关注。最近的高级方法利用图神经网络和对比学习来表征路段的空间结构，采用自监督范式。然而，道路网络的空间异质性和时间动态性对自监督GNN的邻域平滑机制提出了严重挑战。为解决这些问题，我们提出了一种双分支时空自监督表示框架DST，用于增强道路表示。一方面，DST设计了一个混合跳转过渡矩阵用于图卷积，从轨迹中整合道路的动态关系。此外，DST通过空间自监督方式对比普通道路网络与超图的表示，超图基于三种类型的超边构建以捕捉长程关系。另一方面，DST在基于因果Transformer的交通动态序列上执行下一个令牌预测作为时间自监督任务，并通过区分工作日和周末的交通模式进行进一步正则化。与最先进方法的广泛实验验证了我们提出框架的优越性。此外，全面的时空建模使DST在零学习场景中表现出色。


### 论文摘要

Road network representation learning (RNRL) has attracted increasing attention from both researchers and practitioners as various spatiotemporal tasks are emerging. Recent advanced methods leverage Graph Neural Networks (GNNs) and contrastive learning to characterize the spatial structure of road segments in a self-supervised paradigm. However, spatial heterogeneity and temporal dynamics of road networks raise severe challenges to the neighborhood smoothing mechanism of self-supervised GNNs. To address these issues, we propose a $\textbf{D}$ual-branch $\textbf{S}$patial-$\textbf{T}$emporal self-supervised representation framework for enhanced road representations, termed as DST. On one hand, DST designs a mix-hop transition matrix for graph convolution to incorporate dynamic relations of roads from trajectories. Besides, DST contrasts road representations of the vanilla road network against that of the hypergraph in a spatial self-supervised way. The hypergraph is newly built based on three types of hyperedges to capture long-range relations. On the other hand, DST performs next token prediction as the temporal self-supervised task on the sequences of traffic dynamics based on a causal Transformer, which is further regularized by differentiating traffic modes of weekdays from those of weekends. Extensive experiments against state-of-the-art methods verify the superiority of our proposed framework. Moreover, the comprehensive spatiotemporal modeling facilitates DST to excel in zero-shot learning scenarios.

---

## 33. Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily

**论文链接:** [http://arxiv.org/abs/2511.06608v1](http://arxiv.org/abs/2511.06608v1)

**作者:** Asela Hevapathige, Asiri Wijesinghe, Ahad N. Zehmakan

**发布时间:** 2025-11-10

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

本文提出了一种自适应深度的图神经网络架构，能够根据节点的局部同质性水平和邻域结构动态选择聚合深度，从而在同类和异类图中都能有效提升节点分类性能。

### 背景

传统图神经网络在节点分类任务中取得了显著成功，但在异类图（连接节点通常属于不同标签或属性）上效果会下降。现有方法存在局限性：大多数模型对所有节点应用固定聚合深度，且多数方法仅针对同类或异类场景设计，缺乏通用性。

### 目的

解决传统GNN在异类图上性能下降的问题，开发一种能够自适应处理同类和异类图结构的GNN架构，通过动态选择节点特定的聚合深度来提升节点分类性能。

### 方法

开发了一个理论框架，将局部结构和标签特征与节点级别的信息传播动力学联系起来；基于此分析提出一种新型自适应深度GNN架构，使用基于理论指标的度量动态选择节点特定的聚合深度，使模型能够同时适应同类和异类模式。

### 主要发现

最优聚合深度因节点而异，且对于保留类别判别信息至关重要；通过动态选择节点特定的聚合深度，模型能够在不同同质性的图中保持良好性能。

### 结论

所提出的自适应深度GNN架构在不同基准测试中一致地提升了标准GNN骨干网络的性能，为处理具有不同同质性的图结构提供了有效解决方案。

### 翻译

图神经网络（GNNs）在解决节点分类任务方面已取得显著成功。然而，传统GNNs在异类图上的效果会下降，在这些图中，连接的节点通常属于不同的标签或属性。尽管近期工作已引入了提高GNN在异质性条件下性能的机制，但仍存在某些关键局限性。大多数现有模型对所有节点应用固定的聚合深度，忽略了节点可能需要根据其局部同质性水平和邻域结构采用不同的传播深度。此外，许多方法专门针对同类或异类场景设计，缺乏在两种模式下的通用性。为解决这些挑战，我们开发了一个理论框架，将局部结构和标签特征与节点级别的信息传播动力学联系起来。我们的分析表明，最优聚合深度因节点而异，对于保留类别判别信息至关重要。受此见解指导，我们提出了一种新型自适应深度GNN架构，使用基于理论的度量动态选择节点特定的聚合深度。我们的方法在统一模型中能够无缝适应同类和异类模式。大量实验证明，我们的方法在不同基准测试中一致地提升了标准GNN骨干网络的性能。


### 论文摘要

Graph Neural Networks (GNNs) have achieved significant success in addressing node classification tasks. However, the effectiveness of traditional GNNs degrades on heterophilic graphs, where connected nodes often belong to different labels or properties. While recent work has introduced mechanisms to improve GNN performance under heterophily, certain key limitations still exist. Most existing models apply a fixed aggregation depth across all nodes, overlooking the fact that nodes may require different propagation depths based on their local homophily levels and neighborhood structures. Moreover, many methods are tailored to either homophilic or heterophilic settings, lacking the flexibility to generalize across both regimes. To address these challenges, we develop a theoretical framework that links local structural and label characteristics to information propagation dynamics at the node level. Our analysis shows that optimal aggregation depth varies across nodes and is critical for preserving class-discriminative information. Guided by this insight, we propose a novel adaptive-depth GNN architecture that dynamically selects node-specific aggregation depths using theoretically grounded metrics. Our method seamlessly adapts to both homophilic and heterophilic patterns within a unified model. Extensive experiments demonstrate that our approach consistently enhances the performance of standard GNN backbones across diverse benchmarks.

---

## 34. Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees

**论文链接:** [http://arxiv.org/abs/2511.06598v1](http://arxiv.org/abs/2511.06598v1)

**作者:** Mohammad Shirzadi, Ali Safarpoor Dehkordi, Ahad N. Zehmakan

**发布时间:** 2025-11-10

**备注:** This is the full version of the paper accepted to the 40th Annual  AAAI Conference on Artificial Intelligence (AAAI-2026)

### GPT解析

### 总结

论文研究了图神经网络中的自适应残差连接方案，证明了这种方法可以防止过度平滑并保持嵌入表达能力，实验验证了该方法在异质图上的优越性，并提出了时间复杂度更低的启发式变体。

### 背景

消息传递是图神经网络的核心操作，每个节点通过聚合邻居信息来更新其嵌入表示。然而，在深度架构中，这个过程常常导致表达能力减弱。

### 目的

研究一种自适应残差方案，其中不同节点具有不同的残差强度，证明这种方法可以防止过度平滑，确保嵌入的Dirichlet能量保持远离零的状态。

### 方法

使用自适应残差连接方案，不同节点具有不同的残差强度。引入一种变体，其中残差强度不是学习得到的，而是启发式设置的，以提高时间复杂度。

### 主要发现

这种方法可以防止过度平滑，嵌入的Dirichlet能量保持远离零；这是自适应设置以及带激活函数的静态残差连接的第一个理论保证；广泛的实验表明，这种方法优于标准和最先进的消息传递机制，特别是在异质图上；启发式设置的残差强度与可学习版本表现相当。

### 结论

自适应残差方案在防止过度平滑方面有效，在异质图上表现尤其出色，启发式设置的残差强度与可学习版本表现相当。

### 翻译

消息传递是图神经网络的核心操作，其中每个节点通过聚合来自邻居的信息来更新其嵌入表示。然而，在深度架构中，这个过程常常导致表达能力减弱。一个流行的解决方案是使用残差连接，即将当前（或初始）层的输入添加到聚合的邻居信息中，以在层之间保留嵌入表示。遵循最近的研究路线，我们研究了一种自适应残差方案，其中不同节点具有不同的残差强度。我们证明这种方法可以防止过度平滑；特别是，我们展示了嵌入的Dirichlet能量保持远离零。这不仅是对自适应设置的理论保证，也是对带激活函数的静态残差连接（残差强度在节点间共享）的第一个理论保证。此外，大量实验表明，这种方法优于标准和最先进的消息传递机制，特别是在异质图上。为了提高我们方法的时间复杂度，我们引入了一种变体，其中残差强度不是学习得到的，而是启发式设置的，这种选择与可学习版本表现相当。


### 论文摘要

Message passing is the core operation in graph neural networks, where each node updates its embeddings by aggregating information from its neighbors. However, in deep architectures, this process often leads to diminished expressiveness. A popular solution is to use residual connections, where the input from the current (or initial) layer is added to aggregated neighbor information to preserve embeddings across layers. Following a recent line of research, we investigate an adaptive residual scheme in which different nodes have varying residual strengths. We prove that this approach prevents oversmoothing; particularly, we show that the Dirichlet energy of the embeddings remains bounded away from zero. This is the first theoretical guarantee not only for the adaptive setting, but also for static residual connections (where residual strengths are shared across nodes) with activation functions. Furthermore, extensive experiments show that this adaptive approach outperforms standard and state-of-the-art message passing mechanisms, especially on heterophilic graphs. To improve the time complexity of our approach, we introduce a variant in which residual strengths are not learned but instead set heuristically, a choice that performs as well as the learnable version.

---

## 35. How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation

**论文链接:** [http://arxiv.org/abs/2511.06443v1](http://arxiv.org/abs/2511.06443v1)

**作者:** Zinuo You, Jin Zheng, John Cartlidge

**发布时间:** 2025-11-09

**备注:** 29 pages, 11 figures. Author manuscript accepted for the 40th Annual  AAAI Conference on Artificial Intelligence (AAAI-26), January 2026

### GPT解析

### 总结

本文提出了一种名为信道容量约束估计(C3E)的新框架，用于解决图神经网络中的过度压缩问题，通过信息论方法优化隐藏维度和传播深度的选择。

### 背景

现有的图神经网络通常依赖于对隐藏维度和传播深度的启发式选择，这往往导致传播过程中的严重信息丢失，称为过度压缩。

### 目的

解决图神经网络中的过度压缩问题，提出一种更科学的方法来选择隐藏维度和传播深度。

### 方法

提出信道容量约束估计(C3E)框架，将隐藏维度和深度的选择构建为基于信息论的非线性规划问题，通过将谱图神经网络建模为通信信道，将信道容量与隐藏维度、传播深度、传播机制和图结构直接联系起来。

### 主要发现

1) C3E估计的隐藏维度和深度可减轻过度压缩并改进表示学习；2) 过度压缩是由于表示矩阵中信息的累积压缩造成的；3) 增加隐藏维度可减轻信息压缩，而传播深度的作用更为微妙；4) 存在信息压缩与表示复杂性之间的基本平衡。

### 结论

C3E框架能够有效解决图神经网络中的过度压缩问题，通过基于信息论的方法优化网络参数，提高表示学习效果。

### 翻译

现有的图神经网络通常依赖于对隐藏维度和传播深度的启发式选择，这通常会导致传播过程中的严重信息丢失，称为过度压缩。为解决这一问题，我们提出了信道容量约束估计(C3E)，一个新颖的框架，将隐藏维度和深度的选择构建为一个基于信息论的非线性规划问题。通过将谱图神经网络建模为通信信道，我们的方法直接将信道容量与隐藏维度、传播深度、传播机制和图结构联系起来。在九个公共数据集上的大量实验表明，通过C3E估计的隐藏维度和深度可以减轻过度压缩并持续改进表示学习。实验结果显示，过度压缩是由于表示矩阵中信息的累积压缩造成的。此外，我们的研究表明增加隐藏维度确实可以减轻信息压缩，而传播深度的作用更为微妙，揭示了信息压缩与表示复杂性之间的基本平衡。


### 论文摘要

Existing graph neural networks typically rely on heuristic choices for hidden dimensions and propagation depths, which often lead to severe information loss during propagation, known as over-squashing. To address this issue, we propose Channel Capacity Constrained Estimation (C3E), a novel framework that formulates the selection of hidden dimensions and depth as a nonlinear programming problem grounded in information theory. Through modeling spectral graph neural networks as communication channels, our approach directly connects channel capacity to hidden dimensions, propagation depth, propagation mechanism, and graph structure. Extensive experiments on nine public datasets demonstrate that hidden dimensions and depths estimated by C3E can mitigate over-squashing and consistently improve representation learning. Experimental results show that over-squashing occurs due to the cumulative compression of information in representation matrices. Furthermore, our findings show that increasing hidden dimensions indeed mitigate information compression, while the role of propagation depth is more nuanced, uncovering a fundamental balance between information compression and representation complexity.

---

## 36. Privacy-Preserving Federated Learning for Fair and Efficient Urban Traffic Optimization

**论文链接:** [http://arxiv.org/abs/2511.06363v1](http://arxiv.org/abs/2511.06363v1)

**作者:** Rathin Chandra Shit, Sharmila Subudhi

**发布时间:** 2025-11-09

**备注:** Under review at IEEE journal

### GPT解析

### 总结

FedFair-Traffic是一种隐私保护的联邦学习框架，能够同时优化出行效率、交通公平性和差分隐私保护。通过整合图神经网络与差分隐私机制及公平约束，该框架在实验中显著减少了平均旅行时间，提高了交通公平性，提供了高隐私保护，并大幅降低了通信开销。

### 背景

城市交通优化面临在交通效率和隐私保护间平衡的挑战，同时需要考虑社会经济多样性的交通公平分配。现有集中式交通管理方案侵犯用户位置隐私并加剧交通不平等，而现有联邦学习框架未考虑多目标交通设置中的公平约束。

### 目的

提出一个名为FedFair-Traffic的隐私保护联邦学习框架，联合并同时优化出行效率、交通公平性和差分隐私保护。

### 方法

首次整合三个冲突目标改善城市交通系统。通过集成图神经网络与差分隐私机制和使用基尼系数的公平约束，实现相关车辆间的协作学习与数据本地化。采用梯度裁剪和噪声注入的联邦聚合方法提供差分隐私，并优化效率-公平权衡的帕累托有效解决方案。

### 主要发现

在METR-LA交通数据集上的实验显示：FedFair-Traffic比集中式基线减少平均旅行时间7%（14.2分钟），提高交通公平性73%（基尼系数0.78），提供高隐私保护（隐私分数0.8），并减少89%的通信开销。

### 结论

FedFair-Traffic是可扩展的隐私感知智能城市基础设施，在都市交通流量控制和联邦交通网络中具有应用潜力。

### 翻译

城市交通优化面临着在交通效率和隐私保护之间取得平衡的复杂性威胁，以及基于社会经济多样化社区的交通公平分配问题。当前集中式交通管理方案侵犯了用户位置隐私，并通过提供劣势路线建议进一步加剧了交通不平等，而当前联邦学习框架在多目标交通设置中未考虑公平约束。本研究提出了一种隐私保护的联邦学习框架，称为FedFair-Traffic，它联合并同时优化出行效率、交通公平性和差分隐私保护。这是首次尝试整合三个冲突目标以改善城市交通系统。所提出的方法通过将图神经网络与差分隐私机制和使用基尼系数的公平约束相结合，实现了相关车辆之间的协作学习和数据本地化，采用多目标优化。该框架使用梯度裁剪和噪声注入的联邦聚合方法提供差分隐私，并优化效率-公平权衡的帕累托有效解决方案。在METR-LA交通数据集上的真实世界综合实验表明，与集中式基线相比，FedFair-Traffic可以减少7%（14.2分钟）的平均旅行时间，提高73%的交通公平性（基尼系数，0.78），并提供高隐私保护（隐私分数，0.8），同时减少89%的通信开销。这些结果表明，FedFair-Traffic是一种可扩展的隐私感知智能城市基础设施，在都市交通流量控制和联邦交通网络中可能有应用案例。


### 论文摘要

The optimization of urban traffic is threatened by the complexity of achieving a balance between transport efficiency and the maintenance of privacy, as well as the equitable distribution of traffic based on socioeconomically diverse neighborhoods. Current centralized traffic management schemes invade user location privacy and further entrench traffic disparity by offering disadvantaged route suggestions, whereas current federated learning frameworks do not consider fairness constraints in multi-objective traffic settings. This study presents a privacy-preserving federated learning framework, termed FedFair-Traffic, that jointly and simultaneously optimizes travel efficiency, traffic fairness, and differential privacy protection. This is the first attempt to integrate three conflicting objectives to improve urban transportation systems. The proposed methodology enables collaborative learning between related vehicles with data locality by integrating Graph Neural Networks with differential privacy mechanisms ($\epsilon$-privacy guarantees) and Gini coefficient-based fair constraints using multi-objective optimization. The framework uses federated aggregation methods of gradient clipping and noise injection to provide differential privacy and optimize Pareto-efficient solutions for the efficiency-fairness tradeoff. Real-world comprehensive experiments on the METR-LA traffic dataset showed that FedFair-Traffic can reduce the average travel time by 7\% (14.2 minutes) compared with their centralized baselines, promote traffic fairness by 73\% (Gini coefficient, 0.78), and offer high privacy protection (privacy score, 0.8) with an 89\% reduction in communication overhead. These outcomes demonstrate that FedFair-Traffic is a scalable privacy-aware smart city infrastructure with possible use-cases in metropolitan traffic flow control and federated transportation networks.

---

## 37. Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective

**论文链接:** [http://arxiv.org/abs/2511.06284v1](http://arxiv.org/abs/2511.06284v1)

**作者:** Bing Wang, Ximing Li, Yanjun Wang, Changchun Li, Lin Yuanbo Wu, Buyu Wang, Shengsheng Wang

**发布时间:** 2025-11-09

**备注:** Accepted by AAAI 2026. 13 pages, 6 figures. Code:  https://github.com/wangbing1416/RETSIMD

### GPT解析

### 总结

本文提出了一种名为RETSIMD的新方法用于多模态虚假信息检测，该方法基于文本比图像提供更多信息的观察，通过分割文本并生成增强图像来提高检测效果。

### 背景

多模态虚假信息检测(MMD)是检测社交媒体中包含虚假信息的任务，这些帖子通常包含文本和图像两种模态。

### 目的

提高多模态虚假信息检测的准确性，通过利用文本信息生成增强图像来弥补图像模态信息不足的问题。

### 方法

将文本分割成多个片段，每个片段描述一个可由图像呈现的场景；将这些片段输入预训练的文本到图像生成器生成增强图像；整合文本-图像和图像-标签互信息的辅助目标；在辅助数据集上后训练生成器；定义图像间的三种启发式关系构建图结构，使用图神经网络生成融合特征。

### 主要发现

文本模态在虚假信息检测中比图像模态提供更多信息；图像模态对MMD任务的贡献较小；通过文本分割和图像增强可以提高虚假信息检测的准确性。

### 结论

RETSIMD方法通过利用文本信息生成增强图像，并结合图神经网络进行特征融合，能够有效提高多模态虚假信息检测的准确性。

### 翻译

多模态虚假信息检测(MMD)是指检测涉及虚假信息的社交媒体帖子，这些帖子通常包含文本和图像模态。然而，通过观察MMD帖子，我们认为文本模态可能比图像模态提供更多信息，因为文本通常描述了当前帖子的整体事件/故事，而图像往往只呈现部分场景。我们初步的经验结果表明图像模态确实对MMD贡献较少。基于这一想法，我们提出了一种名为RETSIMD的新MMD方法。具体来说，我们假设每段文本可以分为几个片段，每个文本片段描述了一个可以通过图像呈现的部分场景。因此，我们将文本分割成一系列片段，并将这些片段输入预训练的文本到图像生成器，以增强一系列图像。我们还整合了两个关于文本-图像和图像-标签互信息的辅助目标，并在辅助的文本到图像生成基准数据集上对生成器进行后训练。此外，我们通过定义图像之间的三种启发式关系提出了一种图结构，并使用图神经网络生成融合特征。大量的经验结果验证了RETSIMD的有效性。


### 论文摘要

Multimodal Misinformation Detection (MMD) refers to the task of detecting social media posts involving misinformation, where the post often contains text and image modalities. However, by observing the MMD posts, we hold that the text modality may be much more informative than the image modality because the text generally describes the whole event/story of the current post but the image often presents partial scenes only. Our preliminary empirical results indicate that the image modality exactly contributes less to MMD. Upon this idea, we propose a new MMD method named RETSIMD. Specifically, we suppose that each text can be divided into several segments, and each text segment describes a partial scene that can be presented by an image. Accordingly, we split the text into a sequence of segments, and feed these segments into a pre-trained text-to-image generator to augment a sequence of images. We further incorporate two auxiliary objectives concerning text-image and image-label mutual information, and further post-train the generator over an auxiliary text-to-image generation benchmark dataset. Additionally, we propose a graph structure by defining three heuristic relationships between images, and use a graph neural network to generate the fused features. Extensive empirical results validate the effectiveness of RETSIMD.

---

## 38. Resilience Inference for Supply Chains with Hypergraph Neural Network

**论文链接:** [http://arxiv.org/abs/2511.06208v1](http://arxiv.org/abs/2511.06208v1)

**作者:** Zetian Shen, Hongjun Wang, Jiyuan Chen, Xuan Song

**发布时间:** 2025-11-09

### GPT解析

### 总结

该论文提出了一种名为SC-RIHN的新型超图网络模型，用于推断供应链韧性，解决了现有方法无法有效捕捉供应链网络中高阶多实体依赖关系的问题。实验表明，该方法在合成基准测试中显著优于传统方法，为复杂供应链系统的早期风险评估提供了实用工具。

### 背景

供应链对全球经济稳定至关重要，但干扰会迅速通过网络传播造成重大经济影响。准确及时地推断供应链韧性（即在干扰期间维持核心功能的能力）对主动风险缓解和稳健网络设计至关重要。

### 目的

定义并解决供应链韧性推断（SCRI）问题，即使用超图拓扑和观察到的库存轨迹预测供应链韧性，而不使用显式动态方程。

### 方法

提出供应链韧性推断超图网络（SC-RIHN），这是一种基于超图的新型模型，利用基于集合的编码和超图消息传递来捕获多方企业-产品交互关系。

### 主要发现

全面的实验表明，SC-RIHN在合成基准测试中显著优于传统的MLP、代表性的图神经网络变体和ResInf基线方法，凸显了其在复杂供应链系统中进行实际、早期风险评估的潜力。

### 结论

SC-RIHN模型为供应链韧性推断提供了有效解决方案，能够捕捉供应链网络中的复杂交互关系，无需明确的系统动力学方程，具有重要的实际应用价值。

### 翻译

供应链是全球经济稳定的重要组成部分，然而干扰可以通过相互关联的网络迅速传播，造成重大经济影响。准确及时地推断供应链韧性——即在干扰期间维持核心功能的能力——对于主动风险缓解和稳健网络设计至关重要。然而，现有方法缺乏在没有明确系统动力学的情况下有效推断供应链韧性的机制，难以表示供应链网络中固有的高阶、多实体依赖关系。这些局限性促使定义了一个新问题并开发针对性的建模解决方案。为解决这些挑战，我们正式定义了一个新问题：供应链韧性推断（SCRI），即使用超图拓扑和观察到的库存轨迹来预测供应链韧性，而不使用显式动态方程。为解决此问题，我们提出了供应链韧性推断超图网络（SC-RIHN），这是一种基于超图的新型模型，利用基于集合的编码和超图消息传递来捕获多方企业-产品交互。全面的实验表明，SC-RIHN在合成基准测试中显著优于传统的MLP、代表性的图神经网络变体和ResInf基线方法，凸显了其在复杂供应链系统中进行实际、早期风险评估的潜力。


### 论文摘要

Supply chains are integral to global economic stability, yet disruptions can swiftly propagate through interconnected networks, resulting in substantial economic impacts. Accurate and timely inference of supply chain resilience the capability to maintain core functions during disruptions is crucial for proactive risk mitigation and robust network design. However, existing approaches lack effective mechanisms to infer supply chain resilience without explicit system dynamics and struggle to represent the higher-order, multi-entity dependencies inherent in supply chain networks. These limitations motivate the definition of a novel problem and the development of targeted modeling solutions. To address these challenges, we formalize a novel problem: Supply Chain Resilience Inference (SCRI), defined as predicting supply chain resilience using hypergraph topology and observed inventory trajectories without explicit dynamic equations. To solve this problem, we propose the Supply Chain Resilience Inference Hypergraph Network (SC-RIHN), a novel hypergraph-based model leveraging set-based encoding and hypergraph message passing to capture multi-party firm-product interactions. Comprehensive experiments demonstrate that SC-RIHN significantly outperforms traditional MLP, representative graph neural network variants, and ResInf baselines across synthetic benchmarks, underscoring its potential for practical, early-warning risk assessment in complex supply chain systems.

---

## 39. Enhancing Robustness of Graph Neural Networks through p-Laplacian

**论文链接:** [http://arxiv.org/abs/2511.06143v1](http://arxiv.org/abs/2511.06143v1)

**作者:** Anuj Kumar Sirohi, Subhanu Halder, Kabir Kumar, Sandeep Kumar

**发布时间:** 2025-11-08

**备注:** Accepted at 5th Workshop on Graphs and more Complex Structures For  Learning and Reasoning (GCLR), The 40th AAAI Conference on Artificial  Intelligence (AAAI-26)

### GPT解析

### 总结

本文提出了一种名为pLAPGNN的计算高效框架，基于加权p-Laplacian，用于增强图神经网络对抗对抗攻击的鲁棒性，并在真实数据集上验证了其有效性和效率。

### 背景

随着数据量增加，企业和利益相关者需要分析数据以做出更好预测。传统关系数据分析已不足以满足需求，图数据分析因其能更真实灵活地建模复杂关系而成为重要工具。图神经网络在社交网络分析、推荐系统、药物发现等领域显示出巨大潜力，但容易受到对抗攻击影响。

### 目的

提高图神经网络对抗对抗攻击的鲁棒性，同时保持计算效率，解决现有鲁棒性方法计算量大且在高强度攻击下表现不佳的问题。

### 方法

提出一种名为pLAPGNN的计算高效框架，基于加权p-Laplacian来增强GNNs的鲁棒性。

### 主要发现

通过在真实数据集上的实证评估，证明了所提出方法在对抗攻击下表现良好，且计算成本较低，有效性和效率均得到验证。

### 结论

pLAPGNN框架能够有效提高GNNs对抗对抗攻击的鲁棒性，同时保持计算效率，解决了现有方法在高强度攻击下表现不佳的问题。

### 翻译

随着日常生活中数据的增加，企业和不同的利益相关者需要分析数据以做出更好的预测。传统上，关系数据一直是各种洞察的来源，但随着计算能力的提高和对实体间更深层次关系理解的需求，设计新技术的需求已经出现。因此，图数据分析已成为理解数据的非凡工具，它揭示了复杂关系更真实和灵活的建模。最近，图神经网络在各种应用中显示出巨大的潜力，如社交网络分析、推荐系统、药物发现等。然而，许多对抗性攻击可能会发生在数据上，无论是在训练期间还是测试期间，都可能对GNN模型期望的结果产生不利影响。因此，使GNNs对这类攻击具有鲁棒性至关重要。现有的鲁棒性方法计算量大，且在攻击强度增加时表现不佳。本文提出了一种计算高效的框架，即基于加权p-Laplacian的pLAPGNN，用于提高GNNs的鲁棒性。在真实数据集上的实证评估确立了所提出方法的有效性和效率。


### 论文摘要

With the increase of data in day-to-day life, businesses and different stakeholders need to analyze the data for better pre- dictions. Traditionally, relational data has been a source of various insights, but with the increase in computational power and the need to understand deeper relationships between en- tities, the need to design new techniques has arisen. For this graph data analysis has become an extraordinary tool for un- derstanding the data, which reveals more realistic and flexible modelling of complex relationships. Recently, Graph Neural Networks (GNNs) have shown great promise in various ap- plications, such as social network analysis, recommendation systems, drug discovery, and more. However, many adversar- ial attacks can happen over the data, whether during training (poisoning attack) or during testing (evasion attack), which can adversely manipulate the desired outcome from the GNN model. Therefore, it is crucial to make the GNNs robust to such attacks. The existing robustness methods are computa- tionally demanding and perform poorly when the intensity of attack increases. This paper presents a computationally ef- ficient framework, namely, pLAPGNN, based on weighted p-Laplacian for making GNNs robust. Empirical evaluation on real datasets establishes the efficacy and efficiency of the proposed method.

---

## 40. Reperio-rPPG: Relational Temporal Graph Neural Networks for Periodicity Learning in Remote Physiological Measurement

**论文链接:** [http://arxiv.org/abs/2511.05946v1](http://arxiv.org/abs/2511.05946v1)

**作者:** Ba-Thinh Nguyen, Thach-Ha Ngoc Pham, Hoang-Long Duc Nguyen, Thi-Duyen Ngo, Thanh-Ha Le

**发布时间:** 2025-11-08

### GPT解析

### 总结

Reperio-rPPG是一种新型框架，通过整合关系卷积网络与图Transformer来有效捕捉生理信号的内在周期性结构，并引入CutMix增强方法提高模型泛化能力，在多种基准数据集上取得了最先进性能且表现出显著鲁棒性。

### 背景

远程光容积脉搏波描记法(rPPG)是一种新兴的无接触式生理传感技术，利用面部视频中的细微颜色变化来估算心率、呼吸率等生命体征，因其可扩展性和便利性在远程医疗、情感计算、驾驶员疲劳检测和健康监测等领域受到关注。

### 目的

解决以往rPPG方法中对生理信号内在周期性特征探索不足或建模不充分的问题，提出能够捕捉细粒度时间动态的新框架，并提高模型在现实条件下的泛化能力。

### 方法

战略性地整合关系卷积网络与图Transformer来捕捉生理信号的周期性结构，引入定制的CutMix数据增强方法，并在PURE、UBFC-rPPG和MMPD三个基准数据集上进行实验验证。

### 主要发现

Reperio-rPPG在三个基准数据集上实现了最先进的性能，并且在各种运动条件(静止、旋转、说话、行走)和光照条件(自然光、低LED、高LED)下表现出显著的鲁棒性。

### 结论

Reperio-rPPG有效解决了以往方法中对生理信号内在周期性建模不足的问题，为远程生理信号测量提供了更强大、更鲁棒的解决方案，代码已在GitHub公开。

### 翻译

远程光容积脉搏波描记法(rPPG)是一种新兴的无接触式生理传感技术，它利用面部视频中的细微颜色变化来估算心率、呼吸率等生命体征。这种非侵入式方法因其可扩展性和便利性，在远程医疗、情感计算、驾驶员疲劳检测和健康监测等多个领域受到关注。尽管在远程生理信号测量方面取得了显著进展，但一个关键特征——内在周期性——在以往的方法中往往被探索不足或建模不充分，限制了它们在现实条件下捕捉细粒度时间动态的能力。为填补这一空白，我们提出了Reperio-rPPG，一个新型框架，通过战略性地整合关系卷积网络与图Transformer来有效捕捉生理信号中固有的周期性结构。此外，认识到现有rPPG数据集的多样性有限，我们进一步引入了定制的CutMix增强方法来提高模型的泛化能力。在三个广泛使用的基准数据集(PURE、UBFC-rPPG和MMPD)上进行的大量实验表明，Reperio-rPPG不仅取得了最先进的性能，而且在各种运动(如静止、旋转、说话、行走)和光照条件(如自然光、低LED、高LED)下表现出显著的鲁棒性。代码已在https://github.com/deconasser/Reperio-rPPG公开。


### 论文摘要

Remote photoplethysmography (rPPG) is an emerging contactless physiological sensing technique that leverages subtle color variations in facial videos to estimate vital signs such as heart rate and respiratory rate. This non-invasive method has gained traction across diverse domains, including telemedicine, affective computing, driver fatigue detection, and health monitoring, owing to its scalability and convenience. Despite significant progress in remote physiological signal measurement, a crucial characteristic - the intrinsic periodicity - has often been underexplored or insufficiently modeled in previous approaches, limiting their ability to capture fine-grained temporal dynamics under real-world conditions. To bridge this gap, we propose Reperio-rPPG, a novel framework that strategically integrates Relational Convolutional Networks with a Graph Transformer to effectively capture the periodic structure inherent in physiological signals. Additionally, recognizing the limited diversity of existing rPPG datasets, we further introduce a tailored CutMix augmentation to enhance the model's generalizability. Extensive experiments conducted on three widely used benchmark datasets - PURE, UBFC-rPPG, and MMPD - demonstrate that Reperio-rPPG not only achieves state-of-the-art performance but also exhibits remarkable robustness under various motion (e.g., stationary, rotation, talking, walking) and illumination conditions (e.g., nature, low LED, high LED). The code is publicly available at https://github.com/deconasser/Reperio-rPPG.

---

## 41. MoEGCL: Mixture of Ego-Graphs Contrastive Representation Learning for Multi-View Clustering

**论文链接:** [http://arxiv.org/abs/2511.05876v1](http://arxiv.org/abs/2511.05876v1)

**作者:** Jian Zhu, Xin Zou, Jun Sun, Cheng Luo, Lei Liu, Lingfang Zeng, Ning Zhang, Bian Wu, Chang Tang, Lirong Dai

**发布时间:** 2025-11-08

**备注:** AAAI'2026 oral paper

### GPT解析

### 总结

本文提出了一种名为MoEGCL的新型多视图聚类方法，通过自我图混合对比表示学习解决了现有方法中的粗粒度图融合问题。

### 背景

图神经网络(GNNs)的进步近年来显著推动了多视图聚类(MVC)的发展，但现有方法存在粗粒度图融合的问题。

### 目的

解决现有多视图聚类方法中在视图级别进行图结构加权融合的粗糙策略，实现更细粒度的图融合。

### 方法

提出MoEGCL方法，包含两个主要模块：1)自我图混合融合(MoEGF)模块，构建自我图并使用专家混合网络在样本级别实现细粒度融合；2)自我图对比学习(EGCL)模块，将融合表示与视图特定表示对齐，增强同一聚类样本的表示相似性。

### 主要发现

通过大量实验验证，MoEGCL在深度多视图聚类任务中达到了最先进的结果。

### 结论

MoEGCL通过在样本级别而非视图级别进行细粒度图融合，以及增强同一聚类样本的表示相似性，有效提升了多视图聚类的性能。

### 翻译

近年来，图神经网络(GNNs)的进步显著推动了多视图聚类(MVC)的发展。然而，现有方法面临粗粒度图融合的问题。具体而言，当前方法通常为每个视图生成单独的图结构，然后在视图级别进行图结构的加权融合，这是一种相对粗糙的策略。为解决这一局限，我们提出了一种新颖的自我图混合对比表示学习(MoEGCL)方法。它主要由两个模块组成。特别是，我们提出了创新的自我图混合融合(MoEGF)方法，该方法构建自我图并利用专家混合网络在样本级别实现自我图的细粒度融合，而非传统的视图级别融合。此外，我们提出了自我图对比学习(EGCL)模块，将融合后的表示与视图特定表示对齐。EGCL模块增强了来自同一聚类而不仅仅是同一样本的样本表示相似性，进一步提升了细粒度图表示。大量实验证明，MoEGCL在深度多视图聚类任务中取得了最先进的结果。源代码已在https://github.com/HackerHyper/MoEGCL公开。


### 论文摘要

In recent years, the advancement of Graph Neural Networks (GNNs) has significantly propelled progress in Multi-View Clustering (MVC). However, existing methods face the problem of coarse-grained graph fusion. Specifically, current approaches typically generate a separate graph structure for each view and then perform weighted fusion of graph structures at the view level, which is a relatively rough strategy. To address this limitation, we present a novel Mixture of Ego-Graphs Contrastive Representation Learning (MoEGCL). It mainly consists of two modules. In particular, we propose an innovative Mixture of Ego-Graphs Fusion (MoEGF), which constructs ego graphs and utilizes a Mixture-of-Experts network to implement fine-grained fusion of ego graphs at the sample level, rather than the conventional view-level fusion. Additionally, we present the Ego Graph Contrastive Learning (EGCL) module to align the fused representation with the view-specific representation. The EGCL module enhances the representation similarity of samples from the same cluster, not merely from the same sample, further boosting fine-grained graph representation. Extensive experiments demonstrate that MoEGCL achieves state-of-the-art results in deep multi-view clustering tasks. The source code is publicly available at https://github.com/HackerHyper/MoEGCL.

---

## 42. Multi-Scale Feature Fusion and Graph Neural Network Integration for Text Classification with Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.05752v1](http://arxiv.org/abs/2511.05752v1)

**作者:** Xiangchen Song, Yulin Huang, Jinxu Guo, Yuchen Liu, Yaxuan Luan

**发布时间:** 2025-11-07

### GPT解析

### 总结

本研究提出了一种用于文本分类的混合方法，整合了大语言模型的深度特征提取、特征金字塔的多尺度融合以及图神经网络的结构化建模，以提升复杂语义上下文中的分类性能。

### 背景

在复杂语义语境下，需要更有效的文本分类方法来处理文本中的深层语义关系和复杂交互。

### 目的

通过整合多种技术方法，构建一个能够平衡全局与局部信息、语义与结构的文本分类框架，提高分类性能。

### 方法

首先使用大语言模型捕获文本的上下文依赖和深层语义表示；然后通过特征金字塔机制融合不同尺度的语义特征，平衡全局信息和局部细节；接着将融合特征转换为图表示，利用图神经网络捕获潜在语义关系和逻辑依赖；最后通过读取和分类模块生成最终类别预测。

### 主要发现

该方法在鲁棒性对齐实验中表现出显著优势，在ACC、F1-Score、AUC和Precision等指标上均优于现有模型，验证了框架的有效性和稳定性。

### 结论

该方法构建了一个平衡全局和局部信息、语义和结构的集成框架，为文本分类任务中的多尺度特征融合和结构化语义建模提供了新视角。

### 翻译

本研究调查了一种用于文本分类的混合方法，该方法整合了大语言模型的深度特征提取、特征金字塔的多尺度融合以及图神经网络的结构化建模，以增强复杂语义上下文中的性能。首先，大语言模型捕获输入文本的上下文依赖和深层语义表示，为后续建模提供丰富的特征基础。然后，基于多级特征表示，特征金字塔机制有效融合不同尺度的语义特征，平衡全局信息和局部细节，构建层次化语义表达。此外，融合的特征被转换为图表示，并采用图神经网络捕获文本中的潜在语义关系和逻辑依赖，实现对语义单元间复杂交互的全面建模。在此基础上，读取和分类模块生成最终的类别预测。该方法在鲁棒性对齐实验中表现出显著优势，在ACC、F1-Score、AUC和Precision等指标上优于现有模型，验证了该框架的有效性和稳定性。本研究不仅构建了一个平衡全局和局部信息以及语义和结构的集成框架，还为文本分类任务中的多尺度特征融合和结构化语义建模提供了新视角。


### 论文摘要

This study investigates a hybrid method for text classification that integrates deep feature extraction from large language models, multi-scale fusion through feature pyramids, and structured modeling with graph neural networks to enhance performance in complex semantic contexts. First, the large language model captures contextual dependencies and deep semantic representations of the input text, providing a rich feature foundation for subsequent modeling. Then, based on multi-level feature representations, the feature pyramid mechanism effectively integrates semantic features of different scales, balancing global information and local details to construct hierarchical semantic expressions. Furthermore, the fused features are transformed into graph representations, and graph neural networks are employed to capture latent semantic relations and logical dependencies in the text, enabling comprehensive modeling of complex interactions among semantic units. On this basis, the readout and classification modules generate the final category predictions. The proposed method demonstrates significant advantages in robustness alignment experiments, outperforming existing models on ACC, F1-Score, AUC, and Precision, which verifies the effectiveness and stability of the framework. This study not only constructs an integrated framework that balances global and local information as well as semantics and structure, but also provides a new perspective for multi-scale feature fusion and structured semantic modeling in text classification tasks.

---

## 43. Personalized Image Editing in Text-to-Image Diffusion Models via Collaborative Direct Preference Optimization

**论文链接:** [http://arxiv.org/abs/2511.05616v1](http://arxiv.org/abs/2511.05616v1)

**作者:** Connor Dunlop, Matthew Zheng, Kavana Venkatesh, Pinar Yanardag

**发布时间:** 2025-11-06

**备注:** Published at NeurIPS'25 Main Conference

### GPT解析

### 总结

本研究提出了首个个性化图像编辑框架，通过协作直接偏好优化(C-DPO)方法使扩散模型能够适应用户特定的美学偏好，同时利用相似品味用户之间的协作信号。

### 背景

文本到图像(T2I)扩散模型在根据文本生成和编辑高保真图像方面取得了显著进展，但这些模型本质上仍然是通用的，无法适应用户细微的美学偏好差异。

### 目的

开发一种能够适应用户特定审美偏好的个性化图像编辑方法，同时利用具有相似视觉品味的用户之间的协作信息来提高编辑质量。

### 方法

提出协作直接偏好优化(C-DPO)框架，将每个用户编码为动态偏好图中的一个节点，通过轻量级图神经网络学习嵌入表示，实现具有重叠视觉品味用户之间的信息共享，并将这些个性化嵌入整合到新的DPO目标中，同时优化个体对齐和邻域一致性。

### 主要发现

通过用户研究和定量基准测试的综合实验表明，该方法在生成符合用户偏好的编辑方面始终优于基线方法。

### 结论

C-DPO框架成功解决了扩散模型无法适应用户特定审美偏好的问题，通过用户协作信号显著提高了个性化图像编辑的质量和用户满意度。

### 翻译

文本到图像(T2I)扩散模型在根据文本生成和编辑高保真图像方面取得了显著进展。然而，这些模型本质上仍然是通用的，无法适应用户细微的美学偏好。在本工作中，我们提出了扩散模型中首个个性化图像编辑框架，引入协作直接偏好优化(C-DPO)，一种新方法，它使图像编辑与用户特定偏好保持一致，同时利用具有相似品味个体的协作信号。我们的方法将每个用户编码为动态偏好图中的一个节点，并通过轻量级图神经网络学习嵌入表示，使具有重叠视觉品味的用户之间能够共享信息。我们将这些个性化嵌入整合到新的DPO目标中，增强扩散模型的编辑能力，同时优化个体对齐和邻域一致性。包括用户研究和定量基准测试的综合实验表明，我们的方法在生成符合用户偏好的编辑方面始终优于基线。


### 论文摘要

Text-to-image (T2I) diffusion models have made remarkable strides in generating and editing high-fidelity images from text. Yet, these models remain fundamentally generic, failing to adapt to the nuanced aesthetic preferences of individual users. In this work, we present the first framework for personalized image editing in diffusion models, introducing Collaborative Direct Preference Optimization (C-DPO), a novel method that aligns image edits with user-specific preferences while leveraging collaborative signals from like-minded individuals. Our approach encodes each user as a node in a dynamic preference graph and learns embeddings via a lightweight graph neural network, enabling information sharing across users with overlapping visual tastes. We enhance a diffusion model's editing capabilities by integrating these personalized embeddings into a novel DPO objective, which jointly optimizes for individual alignment and neighborhood coherence. Comprehensive experiments, including user studies and quantitative benchmarks, demonstrate that our method consistently outperforms baselines in generating edits that are aligned with user preferences.

---

## 44. ArtReg: Visuo-Tactile based Pose Tracking and Manipulation of Unseen Articulated Objects

**论文链接:** [http://arxiv.org/abs/2511.06378v1](http://arxiv.org/abs/2511.06378v1)

**作者:** Prajval Kumar Murali, Mohsen Kaboli

**发布时间:** 2025-11-09

**备注:** Under review

### GPT解析

### 总结

ArtReg是一种创新的视觉-触觉跟踪方法，能够在没有先验知识的情况下跟踪未知和关节式物体，并通过实验验证了其在各种条件下的鲁棒性和精确性。

### 背景

机器人在真实环境中经常遇到具有复杂结构和关节组件的未知物体，如门、抽屉、柜子、工具等。在没有预先了解物体几何或运动学属性的情况下，感知、跟踪和操作这些物体仍然是机器人技术中的一个基本挑战。

### 目的

提出一种新颖的方法，用于机器人在交互过程中对未知物体（单个、多个或关节式物体）进行视觉-触觉跟踪，无需预先假设物体的形状或动力学知识。

### 方法

1. 提出了一种名为ArtReg（关节式配准）的姿态跟踪方法；2. ArtReg在SE(3)李群中使用无迹卡尔曼滤波公式整合视觉-触觉点云进行点云配准；3. 使用两个机器人团队进行有目的的操作动作（如推或握-拉）来检测物体可能的关节；4. 利用ArtReg开发了一个闭环控制器，用于关节物体的目标驱动操作。

### 主要发现

1. 在各种类型的未知物体上通过真实机器人实验广泛评估了该方法；2. 通过评估具有不同质心、低光条件和具有挑战性视觉背景的物体，证明了该方法的鲁棒性；3. 在标准关节物体数据集上进行了基准测试，与最先进方法相比，姿态精度有所提高；4. 实验表明，利用视觉-触觉信息进行鲁棒精确的姿态跟踪，使机器人能够感知和交互未见的复杂关节物体。

### 结论

基于视觉-触觉信息的鲁棒精确姿态跟踪使机器人能够感知和交互未见的复杂关节物体，这是机器人技术的重要进展。

### 翻译

在真实环境中运行的机器人经常遇到具有复杂结构和关节组件的未知物体，如门、抽屉、柜子、工具等。在没有预先了解物体几何或运动学属性的情况下，感知、跟踪和操作这些物体仍然是机器人技术中的一个基本挑战。在这项工作中，我们提出了一种新颖的方法，用于机器人在交互过程中对未知物体（单个、多个或关节式物体）进行视觉-触觉跟踪，无需预先假设物体的形状或动力学知识。我们称之为ArtReg（关节式配准）的新颖姿态跟踪方法，在SE(3)李群中使用无迹卡尔曼滤波公式整合视觉-触觉点云进行点云配准。ArtReg使用两个机器人团队进行有目的的操作动作（如推或握-拉）来检测物体可能的关节。此外，我们利用ArtReg开发了一个闭环控制器，用于关节物体的目标驱动操作，将物体移动到期望的姿态配置。我们通过各种类型的未知物体进行了广泛的真实机器人实验评估。我们还通过评估具有不同质心、低光条件和具有挑战性视觉背景的物体，证明了我们方法的鲁棒性。此外，我们在标准关节物体数据集上对我们的方法进行了基准测试，并证明了与最先进方法相比在姿态精度方面的改进。我们的实验表明，利用视觉-触觉信息进行鲁棒精确的姿态跟踪，使机器人能够感知和交互未见的复杂关节物体（具有旋转或平移关节）。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人如何在没有预先了解物体几何形状或运动特性的情况下，感知、跟踪和操作具有复杂结构和铰接组件的未知物体。这个问题在现实中非常重要，因为机器人经常需要与日常物体如门、抽屉、柜子、工具等交互，这些物体具有多个自由度和非线性动力学特性，使得准确跟踪和操作变得非常困难。当前大多数方法依赖于预先知道的物体模型或仅使用视觉信息，无法处理完全未知的复杂铰接物体。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：大多数铰接物体跟踪方法依赖于几何特征跟踪或基于标记的方法，或者假设预先知道物体模型。作者借鉴了多模态感知的思想，结合视觉和触觉信息，因为触觉信息可以提供关于物体特性的互补信息，并且对遮挡、环境光照和物体透明度具有不变性。作者设计了ArtReg方法，这是一种基于SE(3)李群上的流形无迹卡尔曼滤波器的视觉-触觉点云配准方法，并使用双机器人系统（一个配备RGB-D视觉传感器，另一个配备触觉传感器阵列）来实现交互式感知和操作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合视觉和触觉信息，在没有预先了解物体形状或动力学的情况下，实现对未知单物体、多物体或铰接物体的姿态跟踪和操作。整体实现流程包括：1) 使用ArtReg方法进行视觉-触觉姿态跟踪，基于SE(3)李群上的流形无迹卡尔曼滤波器整合视觉和触觉点云数据；2) 使用交互式感知检测铰接关节，通过有目的的操作动作（如推、握-拉）来检测可能的铰接关节；3) 使用视觉-触觉闭环控制器进行目标驱动的物体操作，将物体移动到期望的姿态配置。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出了ArtReg方法，一种基于SE(3)李群上的流形无迹卡尔曼滤波器的视觉-触觉点云配准方法；2) 基于ArtReg开发了检测物体中运动链（旋转或平移关节）的新方法；3) 基于ArtReg开发了视觉-触觉闭环控制算法；4) 在各种条件下进行了广泛的实验验证。相比之前的工作，这篇论文的主要不同之处在于：不需要预先了解物体的形状或动力学特性；结合了视觉和触觉信息，而不仅仅是视觉信息；可以处理单物体、多物体和铰接物体；提供了从检测、跟踪到操作的完整框架；在各种条件下展示了方法的鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种名为ArtReg的视觉-触觉融合方法，使机器人能够在没有预先了解物体模型的情况下，准确跟踪和操作未知的单物体、多物体和铰接物体。'}


### 论文摘要

Robots operating in real-world environments frequently encounter unknown objects with complex structures and articulated components, such as doors, drawers, cabinets, and tools. The ability to perceive, track, and manipulate these objects without prior knowledge of their geometry or kinematic properties remains a fundamental challenge in robotics. In this work, we present a novel method for visuo-tactile-based tracking of unseen objects (single, multiple, or articulated) during robotic interaction without assuming any prior knowledge regarding object shape or dynamics. Our novel pose tracking approach termed ArtReg (stands for Articulated Registration) integrates visuo-tactile point clouds in an unscented Kalman Filter formulation in the SE(3) Lie Group for point cloud registration. ArtReg is used to detect possible articulated joints in objects using purposeful manipulation maneuvers such as pushing or hold-pulling with a two-robot team. Furthermore, we leverage ArtReg to develop a closed-loop controller for goal-driven manipulation of articulated objects to move the object into the desired pose configuration. We have extensively evaluated our approach on various types of unknown objects through real robot experiments. We also demonstrate the robustness of our method by evaluating objects with varying center of mass, low-light conditions, and with challenging visual backgrounds. Furthermore, we benchmarked our approach on a standard dataset of articulated objects and demonstrated improved performance in terms of pose accuracy compared to state-of-the-art methods. Our experiments indicate that robust and accurate pose tracking leveraging visuo-tactile information enables robots to perceive and interact with unseen complex articulated objects (with revolute or prismatic joints).

---

## 45. Adaptive Agent Selection and Interaction Network for Image-to-point cloud Registration

**论文链接:** [http://arxiv.org/abs/2511.05965v1](http://arxiv.org/abs/2511.05965v1)

**作者:** Zhixin Cheng, Xiaotian Yin, Jiacheng Deng, Bohao Liao, Yujia Chen, Xu Zhou, Baoqun Yin, Tianzhu Zhang

**发布时间:** 2025-11-08

**备注:** Accepted by AAAI2026

### GPT解析

### 总结

提出了一种新的跨模态配准框架，包含迭代代理选择(IAS)和可靠代理交互(RAI)两个模块，解决了无检测图像到点云配准在噪声环境下表现不佳的问题。

### 背景

典型的无检测图像到点云配准方法利用基于Transformer的架构聚合跨模态特征并建立对应关系，但在噪声干扰下容易产生错误对应，且难以有效选择跨模态中的相关信息表示。

### 目的

解决现有方法在挑战性条件下表现不佳的问题，提高配准的鲁棒性和准确性。

### 方法

提出一个由两个关键模块组成的跨模态配准框架：1) 迭代代理选择(IAS)模块，使用相位图增强结构特征感知并采用强化学习选择可靠代理；2) 可靠代理交互(RAI)模块，利用选定的代理引导跨模态交互，减少不匹配。

### 主要发现

在RGB-D Scenes v2和7-Scenes基准测试上进行了大量实验，结果表明该方法始终达到了最先进的性能。

### 结论

通过IAS和RAI模块的组合，有效解决了噪声环境下的配准挑战，提高了跨模态配准的准确性和鲁棒性。

### 翻译

典型的无检测图像到点云配准方法利用基于Transformer的架构来聚合跨模态特征并建立对应关系。然而，它们在具有挑战性的条件下往往表现不佳，其中噪声会干扰相似性计算并导致错误的对应关系。此外，如果没有专门的设计，仍然难以有效选择跨模态中信息丰富且相关的表示，从而限制了配准的鲁棒性和准确性。为解决这些挑战，我们提出了一种新型的跨模态配准框架，由两个关键模块组成：迭代代理选择(IAS)模块和可靠代理交互(RAI)模块。IAS通过相位图增强结构特征感知能力，并采用强化学习原则来高效选择可靠的代理。然后，RAI利用这些选定的代理来引导跨模态交互，有效减少不匹配并提高整体鲁棒性。在RGB-D Scenes v2和7-Scenes基准测试上的大量实验表明，我们的方法始终达到了最先进的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决图像到点云配准（Image-to-point cloud Registration, I2P）问题，特别是在噪声干扰、重复结构和非重叠区域等挑战性条件下的配准鲁棒性和准确性问题。这个问题在现实中非常重要，因为I2P是3D重建、SLAM（同步定位与地图构建）和视觉定位等关键视觉任务的基础步骤，对于机器人导航、自动驾驶和增强现实等领域至关重要。准确配准能够弥合2D图像和3D点云之间的模态差距，提高空间感知能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统transformer-based方法在噪声环境下容易产生错误对应关系的问题，以及难以有效选择跨模态信息丰富表示的挑战。他们借鉴了相位图提取技术（来自NeRF等工作的启发）来增强图像结构特征感知，并引入强化学习原理设计代理选择策略。整体设计思路是通过增强结构特征感知能力，并采用智能代理选择来提高配准质量。该方法确实借鉴了现有工作，包括transformer架构、强化学习、粗到细匹配策略等，但进行了创新性整合和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过相位图增强图像的结构特征感知能力，并采用强化学习原则设计三阶段代理优化策略，从冗余查询中识别可靠代理，然后利用这些代理指导跨模态特征交互，减少噪声影响。整体流程包括：1)使用ResNet和KPFCNN分别提取图像和点云特征；2)应用傅里叶变换提取图像相位信息增强结构感知；3)三阶段代理优化（预热训练、奖励引导训练、最优代理选择）；4)可靠代理交互模块进行跨模态特征融合；5)通过PnP-RANSAC估计最终刚性变换。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出自适应代理选择与交互网络（A2SI）；2)设计迭代代理选择（IAS）模块，利用相位图增强结构特征感知和三阶段代理优化策略；3)设计可靠代理交互（RAI）模块，用代理引导交互替代传统transformer融合。相比之前的工作，A2SI采用无检测设计避免了关键点检测的模态依赖问题；通过专门设计的代理选择机制选择信息丰富且相关的表示；使用强化学习优化代理选择而非简单top-k选择；减少噪声特征影响，提高匹配鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种自适应代理选择与交互网络，通过相位图增强和强化学习引导的代理选择，有效解决了图像到点云配准中跨模态特征匹配的挑战，显著提高了配准的准确性和鲁棒性。'}


### 论文摘要

Typical detection-free methods for image-to-point cloud registration leverage transformer-based architectures to aggregate cross-modal features and establish correspondences. However, they often struggle under challenging conditions, where noise disrupts similarity computation and leads to incorrect correspondences. Moreover, without dedicated designs, it remains difficult to effectively select informative and correlated representations across modalities, thereby limiting the robustness and accuracy of registration. To address these challenges, we propose a novel cross-modal registration framework composed of two key modules: the Iterative Agents Selection (IAS) module and the Reliable Agents Interaction (RAI) module. IAS enhances structural feature awareness with phase maps and employs reinforcement learning principles to efficiently select reliable agents. RAI then leverages these selected agents to guide cross-modal interactions, effectively reducing mismatches and improving overall robustness. Extensive experiments on the RGB-D Scenes v2 and 7-Scenes benchmarks demonstrate that our method consistently achieves state-of-the-art performance.

---

## 46. Do Street View Imagery and Public Participation GIS align: Comparative Analysis of Urban Attractiveness

**论文链接:** [http://arxiv.org/abs/2511.05570v1](http://arxiv.org/abs/2511.05570v1)

**作者:** Milad Malekzadeh, Elias Willberg, Jussi Torkko, Silviya Korpilo, Kamyar Hasanzadeh, Olle Järv, Tuuli Toivonen

**发布时间:** 2025-11-04

### GPT解析

### 总结

本研究比较了街景图像(SVI)和公众参与地理信息系统(PPGIS)两种方法在捕捉城市环境感知方面的一致性，发现两者只有部分匹配，SVI无法完全捕捉PPGIS记录的丰富体验。

### 背景

数字工具越来越多地影响空间规划实践，了解不同数据源如何反映城市环境中的人类体验至关重要。SVI和PPGIS是两种突出的方法，用于收集基于地点的认知以支持城市规划决策，但它们的可比性尚未得到充分探索。

### 目的

研究基于SVI的感知吸引力和通过城市范围的PPGIS调查收集的居民报告经验之间的一致性，研究地点为芬兰赫尔辛基。

### 方法

使用参与者评分的SVI数据和语义图像分割，训练机器学习模型基于视觉特征预测感知吸引力，并将这些预测与PPGIS确定的有吸引力或没有吸引力的位置进行比较，使用两套严格和适中的标准计算一致性。

### 主要发现

两个数据集之间只有部分一致性：使用中等标准时，对于有吸引力和没有吸引力的地方，一致性分别达到67%和77%；使用严格标准时，一致性分别下降到27%和29%。非视觉因素（如噪音、交通、人口存在和土地利用）显著导致了不匹配，模型无法解释图像中不可见的体验维度。

### 结论

虽然SVI提供了可扩展的城市感知视觉代理，但它不能完全替代PPGIS捕捉的体验丰富性。两种方法都有价值但服务于不同目的，需要更综合的方法来全面捕捉人们如何感知城市环境。

### 翻译

随着数字工具日益塑造空间规划实践，了解不同数据源如何反映城市环境中的人类体验至关重要。街景图像(SVI)和公众参与地理信息系统(PPGIS)代表了两种突出的捕捉基于地点认知的方法，可以支持城市规划决策，但它们的可比性尚未得到充分探索。本研究调查了基于SVI的感知吸引力和通过芬兰赫尔辛基全市范围的PPGIS调查收集的居民报告经验之间的一致性。使用参与者评分的SVI数据和语义图像分割，我们训练了一个机器学习模型来基于视觉特征预测感知吸引力。我们将这些预测与PPGIS确定的有吸引力或没有吸引力的位置进行比较，使用两套严格和适中的标准计算一致性。我们的发现显示两个数据集之间只有部分一致性。虽然使用中等标准时，对于有吸引力和没有吸引力的地方，一致性分别达到67%和77%，但使用严格标准时，一致性分别下降到27%和29%。通过分析各种背景变量，包括噪音、交通、人口存在和土地利用，我们发现非视觉因素显著导致了不匹配。模型无法解释塑造感知但图像中不可见的体验维度，如活动水平和环境压力因素。这些结果表明，虽然SVI提供了可扩展的城市感知视觉代理，但它不能完全替代PPGIS捕捉的体验丰富性。我们认为两种方法都有价值但服务于不同目的；因此，需要更综合的方法来全面捕捉人们如何感知城市环境。


### 论文摘要

As digital tools increasingly shape spatial planning practices, understanding how different data sources reflect human experiences of urban environments is essential. Street View Imagery (SVI) and Public Participation GIS (PPGIS) represent two prominent approaches for capturing place-based perceptions that can support urban planning decisions, yet their comparability remains underexplored. This study investigates the alignment between SVI-based perceived attractiveness and residents' reported experiences gathered via a city-wide PPGIS survey in Helsinki, Finland. Using participant-rated SVI data and semantic image segmentation, we trained a machine learning model to predict perceived attractiveness based on visual features. We compared these predictions to PPGIS-identified locations marked as attractive or unattractive, calculating agreement using two sets of strict and moderate criteria. Our findings reveal only partial alignment between the two datasets. While agreement (with a moderate threshold) reached 67% for attractive and 77% for unattractive places, agreement (with a strict threshold) dropped to 27% and 29%, respectively. By analysing a range of contextual variables, including noise, traffic, population presence, and land use, we found that non-visual cues significantly contributed to mismatches. The model failed to account for experiential dimensions such as activity levels and environmental stressors that shape perceptions but are not visible in images. These results suggest that while SVI offers a scalable and visual proxy for urban perception, it cannot fully substitute the experiential richness captured through PPGIS. We argue that both methods are valuable but serve different purposes; therefore, a more integrated approach is needed to holistically capture how people perceive urban environments.

---

## 47. MVU-Eval: Towards Multi-Video Understanding Evaluation for Multimodal LLMs

**论文链接:** [http://arxiv.org/abs/2511.07250v1](http://arxiv.org/abs/2511.07250v1)

**作者:** Tianhao Peng, Haochen Wang, Yuanxing Zhang, Zekun Wang, Zili Wang, Ge Zhang, Jian Yang, Shihao Li, Yanghai Wang, Xintao Wang, Houyi Li, Wei Ji, Pengfei Wan, Wenhao Huang, Zhaoxiang Zhang, Jiaheng Liu

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文介绍了MVU-Eval，这是首个专门用于评估多模态大语言模型在多视频理解能力的基准测试。通过大量多样化的视频和问题对，评估了模型在基础感知和高阶推理方面的能力，研究发现当前MLLMs在多视频理解方面存在明显不足。

### 背景

多模态大语言模型的出现已将AI能力扩展到视觉模态，但现有评估基准仍局限于单视频理解，忽略了现实场景(如体育分析和自动驾驶)中多视频理解的关键需求。

### 目的

引入MVU-Eval，首个用于评估MLLMs多视频理解的全面基准，以解决现有评估基准的显著差距。

### 方法

MVU-Eval通过1,824个精心策划的问题-答案对评估八项核心能力，涵盖4,959个来自不同领域的视频，包括基础感知任务和高阶推理任务，并与现实世界应用严格对齐。

### 主要发现

通过对最先进的开源和闭源模型进行广泛评估，揭示了当前MLLMs在多视频理解能力方面存在显著的性能差异和局限性。

### 结论

该基准将公开发布，以促进未来多视频理解领域的研究发展。

### 翻译

多模态大语言模型的出现已将AI能力扩展到视觉模态，然而现有的评估基准仍然局限于单视频理解，忽略了现实场景(如体育分析和自动驾驶)中多视频理解的关键需求。为解决这一重要差距，我们引入了MVU-Eval，这是首个用于评估MLLMs多视频理解的全面基准。具体而言，我们的MVU-Eval主要通过1,824个精心策划的问题-答案对来评估八项核心能力，这些答案对涵盖来自不同领域的4,959个视频，既包括基础感知任务，也包括高阶推理任务。这些能力与现实世界应用(如自动驾驶系统中的多传感器合成和跨角度体育分析)严格对齐。通过对最先进的开源和闭源模型进行广泛评估，我们揭示了当前MLLMs在多视频理解能力方面存在显著的性能差异和局限性。该基准将公开发布以促进未来研究。


### 论文摘要

The advent of Multimodal Large Language Models (MLLMs) has expanded AI capabilities to visual modalities, yet existing evaluation benchmarks remain limited to single-video understanding, overlooking the critical need for multi-video understanding in real-world scenarios (e.g., sports analytics and autonomous driving). To address this significant gap, we introduce MVU-Eval, the first comprehensive benchmark for evaluating Multi-Video Understanding for MLLMs. Specifically, our MVU-Eval mainly assesses eight core competencies through 1,824 meticulously curated question-answer pairs spanning 4,959 videos from diverse domains, addressing both fundamental perception tasks and high-order reasoning tasks. These capabilities are rigorously aligned with real-world applications such as multi-sensor synthesis in autonomous systems and cross-angle sports analytics. Through extensive evaluation of state-of-the-art open-source and closed-source models, we reveal significant performance discrepancies and limitations in current MLLMs' ability to perform understanding across multiple videos. The benchmark will be made publicly available to foster future research.

---

## 48. 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation

**论文链接:** [http://arxiv.org/abs/2511.07241v1](http://arxiv.org/abs/2511.07241v1)

**作者:** Mengmeng Liu, Jiuming Liu, Yunpeng Zhang, Jiangtao Li, Michael Ying Yang, Francesco Nex, Hao Cheng

**发布时间:** 2025-11-10

**备注:** Accepted by AAAI 2026.The first two authors contributed equally

### GPT解析

### 总结

这篇论文提出了一个名为4DSTR的新型4D生成网络，通过时空校正调制生成式4D高斯飞溅，解决了4D生成中的时空一致性和快速时间变化适应性问题。

### 背景

最近2D图像和3D形状生成的显著进步引起了人们对动态4D内容生成的关注，但现有方法在时空一致性和快速时间变化适应性方面存在不足。

### 目的

解决现有4D生成方法难以保持时空一致性、对快速时间变化适应性差的问题，通过有效的时空建模提高4D生成质量。

### 方法

提出4DSTR网络，通过时空校正调制生成式4D高斯飞溅，设计时间相关性校正可变形尺度和旋转保证时间一致性，并提出自适应空间密集化和修剪策略处理显著时间变化。

### 主要发现

大量实验表明4DSTR在视频到4D生成方面达到最先进性能，在重建质量、时空一致性和快速时间运动适应性方面表现出色。

### 结论

4DSTR通过时空校正和自适应空间密集化/修剪策略有效解决了4D生成中的时空一致性和快速时间变化适应性问题，实现了高质量的4D内容生成。

### 翻译

最近2D图像和3D形状生成的显著进步引起了人们对动态4D内容生成的显著关注。然而，由于缺乏有效的时空建模，先前的4D生成方法通常难以保持时空一致性，并且对快速的时间变化适应不良。为了解决这些问题，我们提出了一种名为4DSTR的新型4D生成网络，它通过时空校正来调制生成式4D高斯飞溅。具体来说，通过设计生成4D序列之间的时间相关性来校正可变形的尺度和旋转，并保证时间一致性。此外，还提出了一种自适应空间密集化和修剪策略，通过动态添加或删除高斯点并考虑它们前一帧的运动，来解决显著的时间变化问题。大量实验表明，4DSTR在视频到4D生成方面达到了最先进的性能，在重建质量、时空一致性和对快速时间运动的适应性方面表现出色。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决4D生成方法中的时空一致性和快速时变变化适应性问题。在现实中，高质量的4D内容对自动驾驶仿真、虚拟现实和数字角色动画等领域至关重要，而现有方法在动态区域经常出现不一致，难以捕捉快速变化的内容，限制了这些应用的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有4D表示方法的局限性，包括独立处理每帧导致缺乏时序相关性，以及固定数量的高斯点难以适应快速变化。他们借鉴了预训练扩散模型、可变形4D高斯溅射表示、Mamba架构和STAG4D的时空锚点设计，并在此基础上创新性地设计了时序相关性和自适应高斯点管理策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过时空校正增强4D生成内容的时空一致性，并动态调整高斯点数量以适应快速变化。整体流程包括：1)使用Zero123++生成多视角帧并初始化3D高斯；2)通过轻量级解码器映射体素特征到4D高斯参数；3)使用Mamba时序编码层关联序列并回归尺度和旋转残差；4)根据累积梯度自适应增密和剪枝高斯点；5)通过多视图SDS损失优化确保时空一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于Mamba的时序相关性模块，建立跨帧关联并回归校正参数；2)自适应高斯增密和剪枝策略，根据纹理变化动态调整点数量。相比STAG4D等方法，4DSTR显式建模时序相关性，考虑同一区域跨帧的快速纹理差异，并动态适应高斯点数量需求，实验显示在FID-VID和FVD指标上分别有15.1%和19.9%的提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '4DSTR通过时空校正和自适应高斯点管理，显著提高了4D生成内容的时空一致性和对快速变化的适应能力，实现了高质量的动态四维内容生成。'}


### 论文摘要

Remarkable advances in recent 2D image and 3D shape generation have induced a significant focus on dynamic 4D content generation. However, previous 4D generation methods commonly struggle to maintain spatial-temporal consistency and adapt poorly to rapid temporal variations, due to the lack of effective spatial-temporal modeling. To address these problems, we propose a novel 4D generation network called 4DSTR, which modulates generative 4D Gaussian Splatting with spatial-temporal rectification. Specifically, temporal correlation across generated 4D sequences is designed to rectify deformable scales and rotations and guarantee temporal consistency. Furthermore, an adaptive spatial densification and pruning strategy is proposed to address significant temporal variations by dynamically adding or deleting Gaussian points with the awareness of their pre-frame movements. Extensive experiments demonstrate that our 4DSTR achieves state-of-the-art performance in video-to-4D generation, excelling in reconstruction quality, spatial-temporal consistency, and adaptation to rapid temporal movements.

---

## 49. Pandar128 dataset for lane line detection

**论文链接:** [http://arxiv.org/abs/2511.07084v1](http://arxiv.org/abs/2511.07084v1)

**作者:** Filip Beránek, Václav Diviš, Ivan Gruber

**发布时间:** 2025-11-10

### GPT解析

### 总结

Pandar128是基于128线激光雷达的最大公开车道线检测数据集，包含52,000多张相机帧和34,000多个激光雷达扫描，在德国各种真实世界条件下采集。

### 背景

车道线检测领域需要高质量数据集和相关方法的研究支持。

### 目的

创建大规模公开数据集，开发简单有效的基线方法，并提出标准化的评估指标。

### 方法

构建Pandar128数据集包含完整传感器校准和同步里程计；开发SimpleLidarLane方法结合BEV分割、聚类和多项式拟合；提出基于多项式的IAM-F1评估指标，采用BEV空间中的插值感知横向匹配。

### 主要发现

SimpleLidarLane方法虽简单，但在各种挑战性条件下（如雨天、稀疏返回）表现优异，表明模块化管道与高质量数据和原则性评估可媲美复杂方法。

### 结论

所有数据和代码已公开发布，以支持基于激光雷达的车道检测研究的可复现性。

### 翻译

我们提出了Pandar128，这是基于128线激光雷达的最大的公开车道线检测数据集。它包含超过52,000张相机帧和34,000个激光雷达扫描，在德国各种真实世界条件下采集。该数据集包含完整的传感器校准（内参、外参）和同步的里程计，支持投影、融合和时间建模等任务。为补充该数据集，我们还引入了SimpleLidarLane，这是一种轻量级的车道线重建基线方法，结合了BEV分割、聚类和多项式拟合。尽管方法简单，但我们的方法在各种挑战性条件下（如雨天、稀疏返回）表现出色，表明模块化管道与高质量数据和原则性评估可以与更复杂的方法竞争。此外，为解决缺乏标准化评估的问题，我们提出了一种新的基于多项式的指标 - 插值感知匹配F1（IAM-F1），该指标在BEV空间中采用插值感知的横向匹配。所有数据和代码均已公开发布，以支持基于激光雷达的车道检测的可复现性。


### 论文摘要

We present Pandar128, the largest public dataset for lane line detection using a 128-beam LiDAR. It contains over 52,000 camera frames and 34,000 LiDAR scans, captured in diverse real-world conditions in Germany. The dataset includes full sensor calibration (intrinsics, extrinsics) and synchronized odometry, supporting tasks such as projection, fusion, and temporal modeling.   To complement the dataset, we also introduce SimpleLidarLane, a light-weight baseline method for lane line reconstruction that combines BEV segmentation, clustering, and polyline fitting. Despite its simplicity, our method achieves strong performance under challenging various conditions (e.g., rain, sparse returns), showing that modular pipelines paired with high-quality data and principled evaluation can compete with more complex approaches.   Furthermore, to address the lack of standardized evaluation, we propose a novel polyline-based metric - Interpolation-Aware Matching F1 (IAM-F1) - that employs interpolation-aware lateral matching in BEV space.   All data and code are publicly released to support reproducibility in LiDAR-based lane detection.

---

## 50. DTTNet: Improving Video Shadow Detection via Dark-Aware Guidance and Tokenized Temporal Modeling

**论文链接:** [http://arxiv.org/abs/2511.06925v1](http://arxiv.org/abs/2511.06925v1)

**作者:** Zhicheng Li, Kunyang Sun, Rui Yao, Hancheng Zhu, Fuyuan Hu, Jiaqi Zhao, Zhiwen Shao, Yong Zhou

**发布时间:** 2025-11-10

### GPT解析

### 总结

这篇论文提出了一种新的视频阴影检测方法，通过视觉语言匹配模块和暗感知语义块解决阴影-背景模糊问题，并使用标记化时间块处理动态阴影变形，实现了高精度和实时效率。

### 背景

视频阴影检测面临两个相互关联的困难：从复杂背景中区分阴影，以及在不同光照条件下建模动态阴影变形。

### 目的

开发一种能够准确区分阴影与背景并有效建模动态阴影变形的视频阴影检测方法，同时保持高效率和实时性能。

### 方法

提出视觉语言匹配模块(VMM)和暗感知语义块(DSB)利用语言先验区分阴影与暗物体；引入自适应掩码重加权弱化半影区域；在解码器阶段应用边缘掩码增强监督；提出标记化时间块(TTB)解耦时空学习，将跨帧阴影语义总结为可学习的时间标记实现高效序列编码。

### 主要发现

在多个基准数据集上的综合实验表明，该方法达到了最先进的准确性，同时保持了实时推理效率。

### 结论

该方法通过结合视觉语言先验和创新的时空建模技术，有效解决了视频阴影检测中的关键挑战，为实际应用提供了高效准确的解决方案。

### 翻译

视频阴影检测面临两个相互关联的困难：从复杂背景中区分阴影以及在各种光照条件下建模动态阴影变形。为了解决阴影-背景模糊问题，我们通过提出的视觉语言匹配模块(VMM)和暗感知语义块(DSB)利用语言先验，提取文本引导的特征以明确区分阴影和暗物体。此外，我们引入自适应掩码重加权以在训练期间弱化半影区域，并在最终解码器阶段应用边缘掩码以获得更好的监督。对于可变阴影形状的时间建模，我们提出了标记化时间块(TTB)，该块解耦了时空学习。TTB将跨帧阴影语义总结为可学习的时间标记，实现了具有最小计算开销的高效序列编码。在多个基准数据集上的综合实验证明了最先进的准确性和实时推理效率。代码可在https://github.com/city-cheng/DTTNet获取。


### 论文摘要

Video shadow detection confronts two entwined difficulties: distinguishing shadows from complex backgrounds and modeling dynamic shadow deformations under varying illumination. To address shadow-background ambiguity, we leverage linguistic priors through the proposed Vision-language Match Module (VMM) and a Dark-aware Semantic Block (DSB), extracting text-guided features to explicitly differentiate shadows from dark objects. Furthermore, we introduce adaptive mask reweighting to downweight penumbra regions during training and apply edge masks at the final decoder stage for better supervision. For temporal modeling of variable shadow shapes, we propose a Tokenized Temporal Block (TTB) that decouples spatiotemporal learning. TTB summarizes cross-frame shadow semantics into learnable temporal tokens, enabling efficient sequence encoding with minimal computation overhead. Comprehensive Experiments on multiple benchmark datasets demonstrate state-of-the-art accuracy and real-time inference efficiency. Codes are available at https://github.com/city-cheng/DTTNet.

---

## 51. TiS-TSL: Image-Label Supervised Surgical Video Stereo Matching via Time-Switchable Teacher-Student Learning

**论文链接:** [http://arxiv.org/abs/2511.06817v1](http://arxiv.org/abs/2511.06817v1)

**作者:** Rui Wang, Ying Zhou, Hao Wang, Wenwei Zhang, Qiang Li, Zhiwei Wang

**发布时间:** 2025-11-10

**备注:** 8 pages, 4 figures, accepted by BiBM2025

### GPT解析

### 总结

本文提出了一种名为TiS-TSL的时间可切换教师-学生学习框架，用于微创手术中的视频立体匹配，通过时空一致性建模解决了现有方法中存在的视差预测不稳定和严重闪烁问题。

### 背景

微创手术中的立体匹配对新一代导航和增强现实至关重要，但由于解剖结构限制，几乎无法提供密集视差监督，通常只能获取少量图像级标签。

### 目的

克服现有教师-学生学习方法仅限于图像级监督、缺乏时间一致性估计的问题，解决视差预测不稳定和视频帧间严重闪烁的问题。

### 方法

提出TiS-TSL框架，核心是统一模型，可在图像预测(IP)、前向视频预测(FVP)和后向视频预测(BVP)三种模式下运行。采用两阶段学习策略：图像到视频(I2V)阶段初始化时域建模；视频到视频(V2V)阶段通过比较前向和后向预测计算双向时空一致性，识别不可靠区域，过滤噪声伪标签，并强制时间相干性。

### 主要发现

在两个公共数据集上的实验表明，TiS-TSL超越了其他基于图像的最先进方法，TEPE和EPE指标分别提高了至少2.11%和4.54%。

### 结论

TiS-TSL通过时空一致性建模有效解决了微创手术中立体匹配的挑战，提高了视差预测的稳定性和准确性。

### 翻译

微创手术中的立体匹配对新一代导航和增强现实至关重要。然而，由于解剖结构的限制，几乎不可能提供密集的视差监督，这通常限制了注释只能在获取一些图像级标签后进行。教师-学生学习通过利用在稀疏标签上训练的教师模型从丰富的未标记手术视频中生成伪标签和相关置信度图，提供了一个有前途的解决方案。然而，现有的TSL方法仅限于图像级监督，仅提供空间置信度，缺乏时间一致性估计。这种时空可靠性的缺失导致视差预测不稳定和视频帧间的严重闪烁伪影。为了克服这些挑战，我们提出了TiS-TSL，一种用于最小监督下视频立体匹配的新型时间可切换教师-学生学习框架。其核心是一个统一模型，可在三种不同模式下运行：图像预测、前向视频预测和后向视频预测，在单一架构中实现灵活的时域建模。通过这一统一模型，TiS-TSL采用两阶段学习策略。图像到视频阶段将稀疏图像级知识转移到初始化时域建模。随后的视频到视频阶段通过比较前向和后向预测来计算双向时空一致性，从而改进时域视差预测。这种一致性可以识别跨帧的不可靠区域，过滤嘈杂的视频级伪标签，并强制执行时间相干性。两个公共数据集上的实验结果表明，TiS-TSL通过将TEPE和EPE分别提高至少2.11%和4.54%，超过了其他基于图像的最先进方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决微创手术中的视频立体匹配问题，特别是在只有少量图像级标签的情况下如何实现高质量且时间一致的立体匹配。这个问题在现实中非常重要，因为微创手术需要精确的3D导航和重建，但受限于解剖结构，难以获得密集的深度标注数据。现有方法在处理视频序列时缺乏时间一致性，导致深度预测在不同帧之间闪烁，这在医疗环境中可能影响手术安全和精度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别到现有教师-学生学习方法在静态图像上表现良好，但直接应用于视频时会导致严重的深度闪烁伪影。他们发现这是因为图像级教师产生时间独立的伪标签，而视频级学生需要学习时间连贯的深度表示。作者借鉴了现有的迭代优化立体匹配方法(如RAFT-Stereo和IGEV-Stereo)的门控循环单元(GRU)架构，以及教师-学生学习框架和双向预测机制，然后设计了时间可切换的三种模式(IP、FVP、BVP)来统一处理图像和视频输入。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一个时间可切换的教师-学生学习框架(TiS-TSL)，通过统一模型在三种模式下运行(图像预测、前向视频预测、后向视频预测)，并采用两阶段学习策略：图像到视频(I2V)阶段初始化时间建模，视频到视频(V2V)阶段通过双向预测一致性过滤噪声伪标签。整体流程是：1)在I2V阶段，教师模型用IP模式为未标记视频生成伪标签，学生模型在FVP模式下学习；2)在V2V阶段，教师模型在FVP和BVP模式下进行双向预测，计算时空一致性置信度图，过滤噪声并强制时间连贯性；3)两个阶段都只更新学生参数，教师参数通过指数移动平均更新。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)时间可切换视频立体匹配模型，统一处理图像和视频输入；2)时间可切换教师-学生学习框架(TiS-TSL)，包含I2V和V2V两个阶段；3)时空置信度过滤机制，通过双向预测评估时间一致性。相比之前工作，不同之处在于：现有图像级TSL方法缺乏时间一致性估计，现有视频方法需要密集标注，而TiS-TSL只需少量图像级标签就能实现高质量时间一致的视频立体匹配；现有半监督方法只考虑空间维度，而TiS-TSL同时处理空间和时间维度的一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TiS-TSL通过时间可切换的教师-学生学习框架和时空置信度过滤机制，实现了在仅有少量图像级标签监督下的高质量、时间一致性的微创手术视频立体匹配，显著提高了预测精度并消除了深度闪烁伪影。'}


### 论文摘要

Stereo matching in minimally invasive surgery (MIS) is essential for next-generation navigation and augmented reality. Yet, dense disparity supervision is nearly impossible due to anatomical constraints, typically limiting annotations to only a few image-level labels acquired before the endoscope enters deep body cavities. Teacher-Student Learning (TSL) offers a promising solution by leveraging a teacher trained on sparse labels to generate pseudo labels and associated confidence maps from abundant unlabeled surgical videos. However, existing TSL methods are confined to image-level supervision, providing only spatial confidence and lacking temporal consistency estimation. This absence of spatio-temporal reliability results in unstable disparity predictions and severe flickering artifacts across video frames. To overcome these challenges, we propose TiS-TSL, a novel time-switchable teacher-student learning framework for video stereo matching under minimal supervision. At its core is a unified model that operates in three distinct modes: Image-Prediction (IP), Forward Video-Prediction (FVP), and Backward Video-Prediction (BVP), enabling flexible temporal modeling within a single architecture. Enabled by this unified model, TiS-TSL adopts a two-stage learning strategy. The Image-to-Video (I2V) stage transfers sparse image-level knowledge to initialize temporal modeling. The subsequent Video-to-Video (V2V) stage refines temporal disparity predictions by comparing forward and backward predictions to calculate bidirectional spatio-temporal consistency. This consistency identifies unreliable regions across frames, filters noisy video-level pseudo labels, and enforces temporal coherence. Experimental results on two public datasets demonstrate that TiS-TSL exceeds other image-based state-of-the-arts by improving TEPE and EPE by at least 2.11% and 4.54%, respectively..

---

## 52. Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV

**论文链接:** [http://arxiv.org/abs/2511.06741v1](http://arxiv.org/abs/2511.06741v1)

**作者:** Wenbo Huang, Jinghui Zhang, Zhenghao Chen, Guang Li, Lei Zhang, Yang Cao, Fang Dong, Takahiro Ogawa, Miki Haseyama

**发布时间:** 2025-11-10

**备注:** Accepted by AAAI 2026 Oral

### GPT解析

### 总结

本文提出了一种名为Otter的模型，用于解决宽视角视频少样本动作识别中的背景干扰和时间关系重建问题。该模型结合了复合分割模块和时间重建模块，能够有效突出主体并重建时间关系，在多个基准数据集上取得了最先进的性能。

### 背景

宽视角视频在少样本动作识别中能有效表达特定场景中的动作，但由于背景干扰，缺乏对主体和背景的全局理解使得识别具有挑战性。RWKV虽然适合全局建模，但直接应用于宽视角FSAR时无法突出主体，且相似背景帧会降低时间关系。

### 目的

设计一种能够有效分割和强调关键区域、重建时间关系的模型，提高宽视角视频少样本动作识别的性能。

### 方法

提出了CompOund SegmenTation and Temporal REconstructing RWKV (Otter)，包含复合分割模块(CSM)用于分割和强调每帧中的关键区域，时间重建模块(TRM)用于实现双向扫描重建时间关系，并将常规原型与时间增强原型结合。

### 主要发现

复合分割模块能有效突出主体信息，时间重建模块能更好地重建时间关系，两者的结合显著提高了宽视角少样本动作识别的性能。

### 结论

Otter模型在SSv2、Kinetics、UCF101和HMDB51等基准测试上取得了最先进的结果，并在VideoBadminton数据集上的额外评估进一步验证了其优越性。

### 翻译

宽视角视频在少样本动作识别中能有效表达特定场景中的动作。然而，缺乏对主体和背景的全局理解使得识别此类样本中的动作具有挑战性，因为背景干扰。学习不同维度之间交互的RWKV在全局建模方面显示出潜力。然而，将RWKV直接应用于宽视角FSAR可能因过多的背景信息而无法突出主体。此外，相似背景帧降低的时间关系难以重建，进一步影响性能。因此，我们设计了复合分割和时间重建RWKV(Otter)。具体而言，设计了复合分割模块来分割和强调每帧中的关键区域，有效突出主体信息。将时间重建模块整合到时间增强的原型构建中，实现双向扫描，更好地重建时间关系。此外，将常规原型与时间增强原型结合，同时增强主体强调和时间建模，提高宽视角FSAR性能。在SSv2、Kinetics、UCF101和HMDB51等基准上的大量实验表明，Otter达到了最先进的性能。在VideoBadminton数据集上的额外评估进一步验证了Otter在宽视角FSAR中的优越性。


### 论文摘要

Wide-angle videos in few-shot action recognition (FSAR) effectively express actions within specific scenarios. However, without a global understanding of both subjects and background, recognizing actions in such samples remains challenging because of the background distractions. Receptance Weighted Key Value (RWKV), which learns interaction between various dimensions, shows promise for global modeling. While directly applying RWKV to wide-angle FSAR may fail to highlight subjects due to excessive background information. Additionally, temporal relation degraded by frames with similar backgrounds is difficult to reconstruct, further impacting performance. Therefore, we design the CompOund SegmenTation and Temporal REconstructing RWKV (Otter). Specifically, the Compound Segmentation Module~(CSM) is devised to segment and emphasize key patches in each frame, effectively highlighting subjects against background information. The Temporal Reconstruction Module (TRM) is incorporated into the temporal-enhanced prototype construction to enable bidirectional scanning, allowing better reconstruct temporal relation. Furthermore, a regular prototype is combined with the temporal-enhanced prototype to simultaneously enhance subject emphasis and temporal modeling, improving wide-angle FSAR performance. Extensive experiments on benchmarks such as SSv2, Kinetics, UCF101, and HMDB51 demonstrate that Otter achieves state-of-the-art performance. Extra evaluation on the VideoBadminton dataset further validates the superiority of Otter in wide-angle FSAR.

---

## 53. VideoSSR: Video Self-Supervised Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.06281v1](http://arxiv.org/abs/2511.06281v1)

**作者:** Zefeng He, Xiaoye Qu, Yafu Li, Siyuan Huang, Daizong Liu, Yu Cheng

**发布时间:** 2025-11-09

### GPT解析

### 总结

本文提出了一种名为VideoSSR的视频自监督强化学习框架，用于提升多模态大语言模型(MLLMs)的视频理解能力。通过引入三种自监督预训练任务并构建相关数据集，实验表明该方法在17个跨四大视频领域的基准测试上平均提升性能超过5%。

### 背景

可验证奖励的强化学习(RLVR)已显著提升MLLMs的视频理解能力，但MLLMs的快速发展已超出现有视频数据集的复杂性，且手动标注新的高质量数据成本过高。

### 目的

研究是否能够利用视频内部丰富的内在信息来自生成高质量、可验证的训练数据，以解决数据标注成本高的问题。

### 方法

引入三种自监督预训练任务(异常定位、目标计数和时间拼图)，构建视频内在理解基准(VIUBench)验证任务难度，开发VideoSSR-30K数据集，并提出VideoSSR框架用于视频自监督强化学习。

### 主要发现

当前最先进的MLLMs在提出的预训练任务上表现不佳；VideoSSR在17个跨四大视频领域(通用视频QA、长视频QA、时间定位和复杂推理)的基准测试上一致提升模型性能，平均改进超过5%。

### 结论

VideoSSR被确立为开发更先进MLLMs视频理解能力的强大基础框架，有效利用视频内在信息生成高质量训练数据。

### 翻译

可验证奖励的强化学习(RLVR)已经显著提升了多模态大语言模型(MLLMs)的视频理解能力。然而，MLLMs的快速发展超出了现有视频数据集的复杂性，而手动标注新的高质量数据仍然成本过高。这项研究探讨了一个关键问题：能否利用视频内部丰富的内在信息来自生成高质量、可验证的训练数据？为了研究这一点，我们引入了三种自监督的预训练任务：异常定位、目标计数和时间拼图。我们构建了视频内在理解基准(VIUBench)来验证这些任务的难度，揭示当前最先进的MLLMs在这些任务上表现显著不佳。基于这些预训练任务，我们开发了VideoSSR-30K数据集，并提出了VideoSSR，一种用于RLVR的新型视频自监督强化学习框架。在跨越四大视频领域(通用视频QA、长视频QA、时间定位和复杂推理)的17个基准测试上的广泛实验表明，VideoSSR一致提升了模型性能，平均改进超过5%。这些结果确立了VideoSSR作为开发MLLMs更先进视频理解能力的强大基础框架。代码可在https://github.com/lcqysl/VideoSSR获取。


### 论文摘要

Reinforcement Learning with Verifiable Rewards (RLVR) has substantially advanced the video understanding capabilities of Multimodal Large Language Models (MLLMs). However, the rapid progress of MLLMs is outpacing the complexity of existing video datasets, while the manual annotation of new, high-quality data remains prohibitively expensive. This work investigates a pivotal question: Can the rich, intrinsic information within videos be harnessed to self-generate high-quality, verifiable training data? To investigate this, we introduce three self-supervised pretext tasks: Anomaly Grounding, Object Counting, and Temporal Jigsaw. We construct the Video Intrinsic Understanding Benchmark (VIUBench) to validate their difficulty, revealing that current state-of-the-art MLLMs struggle significantly on these tasks. Building upon these pretext tasks, we develop the VideoSSR-30K dataset and propose VideoSSR, a novel video self-supervised reinforcement learning framework for RLVR. Extensive experiments across 17 benchmarks, spanning four major video domains (General Video QA, Long Video QA, Temporal Grounding, and Complex Reasoning), demonstrate that VideoSSR consistently enhances model performance, yielding an average improvement of over 5\%. These results establish VideoSSR as a potent foundational framework for developing more advanced video understanding in MLLMs. The code is available at https://github.com/lcqysl/VideoSSR.

---

## 54. Temporal-Guided Visual Foundation Models for Event-Based Vision

**论文链接:** [http://arxiv.org/abs/2511.06238v1](http://arxiv.org/abs/2511.06238v1)

**作者:** Ruihao Xia, Junhong Cai, Luziwei Leng, Liuyi Wang, Chengju Liu, Ran Cheng, Yang Tang, Pan Zhou

**发布时间:** 2025-11-09

### GPT解析

### 总结

本文提出了一种名为时间引导的视觉基础模型（TGVFM）的新框架，成功地将图像预训练的视觉基础模型应用于事件相机视觉任务，通过引入时间上下文融合块解决了异步事件流处理的挑战，在多个视觉任务上取得了最先进的性能。

### 背景

事件相机在具有挑战性的环境视觉任务中具有独特优势，但处理异步事件流仍然是一个开放的挑战。现有方法依赖于专门的架构或资源密集型训练，而利用在图像数据上预训练的现代视觉基础模型在基于事件的视觉方面的潜力尚未被充分探索。

### 目的

提出一个名为时间引导的视觉基础模型（TGVFM）的新框架，将VFMs与时间上下文融合块无缝集成，以弥合图像预训练模型与事件相机视觉任务之间的差距。

### 方法

TGVFM框架包含三个关键的时间块组件：1) 长程时间注意力用于建模全局时间依赖关系；2) 双时空注意力用于多尺度帧相关性；3) 深度特征引导机制用于融合语义-时间特征。通过在真实数据上重新训练事件到视频模型，并利用基于transformer的VFMs，保留时空动态性同时利用预训练表示。

### 主要发现

在语义分割、深度估计和目标检测等任务上，TGVFM分别比现有方法提高了16%、21%和16%，展示了最先进的性能。这证明了基于图像的VFMs在具有时间推理能力的基于事件的视觉中的跨模态潜力。

### 结论

这项工作成功释放了基于图像的视觉基础模型在事件相机视觉任务中的跨模态潜力，通过时间引导的框架有效解决了异步事件流处理的挑战，为未来研究提供了新的方向。

### 翻译

事件相机在具有挑战性环境中的视觉任务方面提供独特优势，但处理异步事件流仍然是一个开放的挑战。虽然现有方法依赖于专门的架构或资源密集型训练，但在基于事件的视觉中利用在图像数据上预训练的现代视觉基础模型的潜力仍未被充分探索。为此，我们提出了时间引导的视觉基础模型（TGVFM），这是一个新颖的框架，将VFMs与我们的时间上下文融合块无缝集成以弥合这一差距。我们的时间块引入了三个关键组件：（1）长程时间注意力用于建模全局时间依赖关系，（2）双时空注意力用于多尺度帧相关性，（3）深度特征引导机制用于融合语义-时间特征。通过在真实数据上重新训练事件到视频模型并利用基于transformer的VFMs，TGVFM保留了时空动态性同时利用了预训练表示。实验在语义分割、深度估计和目标检测等任务上展示了最先进的性能，分别比现有方法提高了16%、21%和16%。总体而言，这项工作释放了基于图像的VFMs在具有时间推理能力的基于事件的视觉中的跨模态潜力。代码可在https://github.com/XiaRho/TGVFM获取。


### 论文摘要

Event cameras offer unique advantages for vision tasks in challenging environments, yet processing asynchronous event streams remains an open challenge. While existing methods rely on specialized architectures or resource-intensive training, the potential of leveraging modern Visual Foundation Models (VFMs) pretrained on image data remains under-explored for event-based vision. To address this, we propose Temporal-Guided VFM (TGVFM), a novel framework that integrates VFMs with our temporal context fusion block seamlessly to bridge this gap. Our temporal block introduces three key components: (1) Long-Range Temporal Attention to model global temporal dependencies, (2) Dual Spatiotemporal Attention for multi-scale frame correlation, and (3) Deep Feature Guidance Mechanism to fuse semantic-temporal features. By retraining event-to-video models on real-world data and leveraging transformer-based VFMs, TGVFM preserves spatiotemporal dynamics while harnessing pretrained representations. Experiments demonstrate SoTA performance across semantic segmentation, depth estimation, and object detection, with improvements of 16%, 21%, and 16% over existing methods, respectively. Overall, this work unlocks the cross-modality potential of image-based VFMs for event-based vision with temporal reasoning. Code is available at https://github.com/XiaRho/TGVFM.

---

## 55. TYrPPG: Uncomplicated and Enhanced Learning Capability rPPG for Remote Heart Rate Estimation

**论文链接:** [http://arxiv.org/abs/2511.05833v1](http://arxiv.org/abs/2511.05833v1)

**作者:** Taixi Chen, Yiu-ming Cheung

**发布时间:** 2025-11-08

**备注:** The 6th International Workshop on AI for Social Good in the Connected  World (AI4SG)@ IEEE WI-IAT 2025

### GPT解析

### 总结

本研究提出了一种名为TYrPPG的新型rPPG算法，通过基于Mambaout结构的门控视频理解块(GVB)和综合监督损失函数(CSL)，实现了在远程心率估计方面的最先进性能。

### 背景

rPPG技术可以从RGB视频中远程提取生理信号，在心率检测方面具有低成本和非侵入性的优势。现有基于transformer的rPPG模型计算效率较低，而Mamba模型在自然语言处理中表现出高效性能，但研究表明其核心SSM模块对视觉任务并非必需。

### 目的

证明使用基于Mambaout的模块远程学习心率的可行性，并开发一种高效的rPPG算法。

### 方法

提出TYrPPG算法，包含创新的门控视频理解块(GVB)，基于Mambaout结构整合2D-CNN和3D-CNN增强视频理解，同时提出综合监督损失函数(CSL)及其弱监督变体来提高模型学习能力。

### 主要发现

TYrPPG在常用数据集上实现了最先进的性能，证明了其在远程心率估计方面的前景和优越性。

### 结论

TYrPPG算法在远程心率估计方面表现出色，具有实际应用价值，源代码已公开可供使用。

### 翻译

远程光电容积描记(rPPG)可以从RGB视频中远程提取生理信号，在心率检测方面具有许多优势，如低成本和非侵入性。现有的rPPG模型通常基于transformer模块，计算效率较低。最近，Mamba模型在自然语言处理任务中因其高效的性能而获得越来越多的关注，显示出作为基于transformer算法替代品的潜力。然而，Mambaout模型及其变体证明，Mamba模型的核心组件SSM模块对于视觉任务不是必需的。因此，我们希望证明使用基于Mambaout的模块远程学习心率的可行性。具体来说，我们提出了一种名为TYrPPG(简单且增强学习能力的rPPG)的新型rPPG算法。本文介绍了一个创新的门控视频理解块(GVB)，专为高效分析RGB视频而设计。基于Mambaout结构，该块整合了2D-CNN和3D-CNN以增强视频理解用于分析。此外，我们提出了一个综合监督损失函数(CSL)来提高模型的学习能力，以及其弱监督变体。实验表明，我们的TYrPPG在常用数据集上可以实现最先进的性能，表明其在远程心率估计方面具有前景和优越性。源代码可在https://github.com/Taixi-CHEN/TYrPPG获取。


### 论文摘要

Remote photoplethysmography (rPPG) can remotely extract physiological signals from RGB video, which has many advantages in detecting heart rate, such as low cost and no invasion to patients. The existing rPPG model is usually based on the transformer module, which has low computation efficiency. Recently, the Mamba model has garnered increasing attention due to its efficient performance in natural language processing tasks, demonstrating potential as a substitute for transformer-based algorithms. However, the Mambaout model and its variants prove that the SSM module, which is the core component of the Mamba model, is unnecessary for the vision task. Therefore, we hope to prove the feasibility of using the Mambaout-based module to remotely learn the heart rate. Specifically, we propose a novel rPPG algorithm called uncomplicated and enhanced learning capability rPPG (TYrPPG). This paper introduces an innovative gated video understanding block (GVB) designed for efficient analysis of RGB videos. Based on the Mambaout structure, this block integrates 2D-CNN and 3D-CNN to enhance video understanding for analysis. In addition, we propose a comprehensive supervised loss function (CSL) to improve the model's learning capability, along with its weakly supervised variants. The experiments show that our TYrPPG can achieve state-of-the-art performance in commonly used datasets, indicating its prospects and superiority in remote heart rate estimation. The source code is available at https://github.com/Taixi-CHEN/TYrPPG.

---

## 56. MARAuder's Map: Motion-Aware Real-time Activity Recognition with Layout-Based Trajectories

**论文链接:** [http://arxiv.org/abs/2511.05773v1](http://arxiv.org/abs/2511.05773v1)

**作者:** Zishuai Liu, Weihang You, Jin Lu, Fei Dou

**发布时间:** 2025-11-08

### GPT解析

### 总结

本文提出了一种名为MARAuder's Map的新框架，用于从原始、未分割的传感器流中进行实时人类活动识别，通过将传感器激活投影到物理平面图上，生成轨迹感知的类图像序列，并结合混合深度学习模型处理空间结构和时间依赖性。

### 背景

基于环境传感器的人类活动识别在智能家居中面临挑战，需要实时推理、空间定位推理和上下文感知的时间建模。现有方法通常依赖预分割的活动内数据，忽略环境物理布局，限制了在连续真实世界部署中的鲁棒性。

### 目的

开发一个能够从原始、未分割传感器流中进行实时活动识别的框架，克服现有方法的局限性，提高在智能家居环境中的活动识别准确性。

### 方法

将传感器激活投影到物理平面图生成轨迹感知的类图像序列；使用混合深度学习模型同时捕获空间结构和时间依赖性；引入可学习的时间嵌入模块编码上下文线索；采用基于注意力的编码器选择性关注信息段，处理跨活动转换和时间模糊情况。

### 主要发现

在多个真实世界智能家居数据集上的实验表明，该方法优于强基线方法，为环境传感器环境中的实时HAR提供了实用的解决方案。

### 结论

MARAuder's Map框架通过考虑空间布局和时间上下文，有效提高了智能家居中实时活动识别的准确性和鲁棒性，为实际应用提供了可行的解决方案。

### 翻译

基于环境传感器的人类活动识别在智能家居中仍然具有挑战性，因为需要实时推理、空间定位推理和上下文感知的时间建模。现有方法通常依赖于预分割的活动内数据，并忽略环境的物理布局，限制了它们在连续、真实世界部署中的鲁棒性。在本文中，我们提出了MARAuder's Map，一种用于从原始、未分割的传感器流中进行实时活动识别的新框架。我们的方法将传感器激活投影到物理平面图上，生成轨迹感知的类图像序列，捕获人类运动的空间流动。这些表示由混合深度学习模型处理，该模型同时捕获空间结构和时间依赖性。为了增强时间感知，我们引入了一个可学习的时间嵌入模块，编码上下文线索，如一天中的小时和一周中的天。此外，基于注意力的编码器选择性地关注每个观察窗口中的信息段，即使在跨活动转换和时间模糊的情况下也能实现准确的识别。在多个真实世界智能家居数据集上的广泛实验表明，我们的方法优于强基线方法，为环境传感器环境中的实时HAR提供了实用的解决方案。


### 论文摘要

Ambient sensor-based human activity recognition (HAR) in smart homes remains challenging due to the need for real-time inference, spatially grounded reasoning, and context-aware temporal modeling. Existing approaches often rely on pre-segmented, within-activity data and overlook the physical layout of the environment, limiting their robustness in continuous, real-world deployments. In this paper, we propose MARAuder's Map, a novel framework for real-time activity recognition from raw, unsegmented sensor streams. Our method projects sensor activations onto the physical floorplan to generate trajectory-aware, image-like sequences that capture the spatial flow of human movement. These representations are processed by a hybrid deep learning model that jointly captures spatial structure and temporal dependencies. To enhance temporal awareness, we introduce a learnable time embedding module that encodes contextual cues such as hour-of-day and day-of-week. Additionally, an attention-based encoder selectively focuses on informative segments within each observation window, enabling accurate recognition even under cross-activity transitions and temporal ambiguity. Extensive experiments on multiple real-world smart home datasets demonstrate that our method outperforms strong baselines, offering a practical solution for real-time HAR in ambient sensor environments.

---

## 57. Sign language recognition from skeletal data using graph and recurrent neural networks

**论文链接:** [http://arxiv.org/abs/2511.05772v1](http://arxiv.org/abs/2511.05772v1)

**作者:** B. Mederos, J. Mejía, A. Medina-Reyes, Y. Espinosa-Almeyda, J. D. Díaz-Roman, I. Rodríguez-Mederos, M. Mejía-Carreon, F. Gonzalez-Lopez

**发布时间:** 2025-11-08

**备注:** 15 pages, 2 figures

### GPT解析

### 总结

这项研究提出了一种基于骨架姿态数据的手语手势识别方法，使用图门控循环单元网络建模时空依赖关系，在土耳其手语数据集上实现了高准确率的识别。

### 背景

手语识别是帮助听障人士与人交流的重要技术，传统方法可能存在局限性，需要更有效的识别方法。

### 目的

开发一种能够准确识别孤立手语手势的方法，利用骨架姿态数据并有效建模时空关系。

### 方法

使用从视频中提取的骨架姿态数据，提出图门控循环单元（Graph-GRU）时序网络来建模帧间的空间和时间依赖关系，并在AUTSL数据集上进行训练和评估。

### 主要发现

实验结果表明，将基于图的空间表示与时序建模相结合是有效的，在手语识别任务中取得了高准确率。

### 结论

姿态驱动的方法在手语理解方面具有巨大潜力，所提出的框架为手语识别提供了可扩展的解决方案。

### 翻译

这项工作提出了一种使用从视频序列中提取的基于骨架的姿态数据来识别孤立手语手势的方法。提出了一种图门控循环单元（Graph-GRU）时序网络，用于建模帧之间的空间和时间依赖关系，实现准确分类。该模型在AUTSL（安卡拉大学土耳其手语）数据集上进行训练和评估，取得了高准确率。实验结果表明，将基于图的空间表示与时序建模相结合是有效的，为手语识别提供了可扩展的框架。这种方法的结果强调了姿态驱动方法在手语理解方面的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决孤立手语识别（ISLR）问题，即识别单个、无上下文的手语手势。这个问题在现实中非常重要，因为它可以为听障人士提供无障碍交流方式，支持手语语言表达的理解。在研究中，它是连续手语理解的基础，且传统方法在处理不同表演者、光照变化和背景差异时泛化能力有限，需要更鲁棒的解决方案。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了手语识别面临的挑战，包括类内变化和表演者差异。他们评估了现有方法（如基于RGB视频、3D-CNN和transformer架构）的局限性，发现基于姿势的方法更有潜力。作者借鉴了图神经网络处理图结构数据的能力，将人体关节表示为节点；借鉴了循环神经网络（特别是GRU）建模时间序列的能力；还借鉴了ResNet的残差连接和注意力机制。通过这些创新性组合，设计了Graph-GRU时间网络，同时捕获空间和时间依赖关系。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将每个手语手势建模为图序列，其中每个图表示特定帧的人体姿势，使用图神经网络捕获空间结构，使用GRU建模时间依赖，并通过残差连接和注意力机制增强特征保存。整体流程包括：1)从视频中提取2D骨骼关键点；2)将输入表示为T个时间步的图序列；3)通过K个由GNN和GRU组成的处理块，每块间有残差连接；4)应用时间注意力机制强调重要帧；5)通过多层分类器将特征映射到手语类别。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)混合Graph-GRU架构同时建模空间和时间依赖；2)在时空处理块间应用残差连接提高训练稳定性；3)时间注意力机制自动识别并强调手势序列中信息量最大的帧；4)高效的骨骼表示减少计算需求。相比之前的工作，该方法不同于传统RGB方法（计算量大且易受环境影响），区别于其他骨骼方法（要么缺乏时间建模，要么未充分利用关节间空间关系），性能上达到90.04%的准确率，训练时间仅约1小时，推理时间0.95秒，显著优于基于视频的方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合图神经网络和门控循环单元的混合架构，通过骨骼数据高效准确地识别孤立手语手势，为听障人士提供了更便捷的无障碍交流方式。'}


### 论文摘要

This work presents an approach for recognizing isolated sign language gestures using skeleton-based pose data extracted from video sequences. A Graph-GRU temporal network is proposed to model both spatial and temporal dependencies between frames, enabling accurate classification. The model is trained and evaluated on the AUTSL (Ankara university Turkish sign language) dataset, achieving high accuracy. Experimental results demonstrate the effectiveness of integrating graph-based spatial representations with temporal modeling, providing a scalable framework for sign language recognition. The results of this approach highlight the potential of pose-driven methods for sign language understanding.

---

## 58. TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.05489v1](http://arxiv.org/abs/2511.05489v1)

**作者:** Junwen Pan, Qizhe Zhang, Rui Zhang, Ming Lu, Xin Wan, Yuan Zhang, Chang Liu, Qi She

**发布时间:** 2025-11-07

**备注:** 22 pages, 17 figures. Official code:  https://github.com/Time-Search/TimeSearch-R

### GPT解析

### 总结

本文提出TimeSearch-R方法，通过将时间搜索重新表述为交错文本-视频思考过程，使用强化学习将视频片段搜索整合到推理过程中，并引入GRPO-CSV方法解决无监督中间搜索决策问题，显著提升了时间搜索和长视频理解能力。

### 背景

时间搜索旨在根据给定查询从数万个帧中识别最小相关帧集，作为长视频理解的基础。现有工作尝试逐步缩小搜索空间，但这些方法通常依赖手工设计的搜索过程，缺乏端到端优化来学习最优搜索策略。

### 目的

解决现有时间搜索方法中缺乏端到端优化的问题，提出一种能够学习最优搜索策略的方法，并通过强化学习将视频片段搜索整合到推理过程中，同时解决无监督中间搜索决策带来的探索不充分和推理不一致问题。

### 方法

1. 提出TimeSearch-R，将时间搜索重新表述为交错文本-视频思考过程；2. 使用强化学习将视频片段搜索无缝整合到推理过程中；3. 引入GRPO-CSV方法，通过收集交错推理过程中搜索的视频帧，并使用相同的策略模型验证搜索帧的充分性；4. 构建专门设计的数据集，用于GRPO-CSV的SFT冷启动和RL训练，过滤掉时间依赖性弱的样本。

### 主要发现

1. TimeSearch-R在Haystack-LVBench和Haystack-Ego4D等时间搜索基准测试上取得了显著改进；2. 在VideoMME和MLVU等长视频理解基准测试上也表现出色；3. 在LongVideoBench上创造了新的最先进水平，比基础模型Qwen2.5-VL提高了4.1%，比先进视频推理模型Video-R1提高了2.0%。

### 结论

TimeSearch-R通过将时间搜索与交错文本-视频思考相结合，并引入GRPO-CSV方法，有效解决了现有方法中的局限性，显著提高了时间搜索和长视频理解的能力。

### 翻译

时间搜索旨在根据给定查询从数万个帧中识别最小相关帧集，作为准确长视频理解的基础。现有工作尝试逐步缩小搜索空间。然而，这些方法通常依赖手工设计的搜索过程，缺乏端到端优化来学习最优搜索策略。在本文中，我们提出TimeSearch-R，它将时间搜索重新表述为交错文本-视频思考，通过强化学习将视频片段搜索无缝整合到推理过程中。然而，将RL训练方法应用于视频推理可能导致无监督的中间搜索决策，这导致视频内容探索不充分和逻辑推理不一致。为解决这些问题，我们引入了带完整性自验证的GRPO，它从交错推理过程中收集搜索的视频帧，并使用相同的策略模型验证搜索帧的充分性，从而提高视频推理的完整性。此外，我们构建了专门为GRPO-CSV的SFT冷启动和RL训练设计的数据集，过滤掉时间依赖性弱的样本，以增强任务难度并提高时间搜索能力。


### 论文摘要

Temporal search aims to identify a minimal set of relevant frames from tens of thousands based on a given query, serving as a foundation for accurate long-form video understanding. Existing works attempt to progressively narrow the search space. However, these approaches typically rely on a hand-crafted search process, lacking end-to-end optimization for learning optimal search strategies. In this paper, we propose TimeSearch-R, which reformulates temporal search as interleaved text-video thinking, seamlessly integrating searching video clips into the reasoning process through reinforcement learning (RL). However, applying RL training methods, such as Group Relative Policy Optimization (GRPO), to video reasoning can result in unsupervised intermediate search decisions. This leads to insufficient exploration of the video content and inconsistent logical reasoning. To address these issues, we introduce GRPO with Completeness Self-Verification (GRPO-CSV), which gathers searched video frames from the interleaved reasoning process and utilizes the same policy model to verify the adequacy of searched frames, thereby improving the completeness of video reasoning. Additionally, we construct datasets specifically designed for the SFT cold-start and RL training of GRPO-CSV, filtering out samples with weak temporal dependencies to enhance task difficulty and improve temporal search capabilities. Extensive experiments demonstrate that TimeSearch-R achieves significant improvements on temporal search benchmarks such as Haystack-LVBench and Haystack-Ego4D, as well as long-form video understanding benchmarks like VideoMME and MLVU. Notably, TimeSearch-R establishes a new state-of-the-art on LongVideoBench with 4.1% improvement over the base model Qwen2.5-VL and 2.0% over the advanced video reasoning model Video-R1. Our code is available at https://github.com/Time-Search/TimeSearch-R.

---

## 59. Canonical Space Representation for 4D Panoptic Segmentation of Articulated Objects

**论文链接:** [http://arxiv.org/abs/2511.05356v1](http://arxiv.org/abs/2511.05356v1)

**作者:** Manuel Gomes, Bogdan Raducanu, Miguel Oliveira

**发布时间:** 2025-11-07

**备注:** 32 pages, 6 figures, 4 tables, submitted to Expert Systems With  Applications

### GPT解析

### 总结

该论文提出了一种新的4D全景分割框架CanonSeg4D，以及配套的Artic4D数据集，用于解决关节物体感知中的时间动态性问题。

### 背景

关节物体感知在计算机视觉中具有重大挑战，因为大多数现有方法忽略了时间动态性，尽管这类物体本质上是动态的。4D时间数据在关节物体感知中尚未得到充分探索，在全景分割领域也未被研究，且缺乏基准数据集。

### 目的

为了解决关节物体感知中忽略时间动态性的问题，作者引入Artic4D数据集并提出CanonSeg4D框架，用于4D全景分割。

### 方法

Artic4D数据集源自PartNet Mobility，并增加了合成传感器数据，具有4D全景标注和关节参数。CanonSeg4D框架通过估计每帧偏移量，将观察到的物体部分映射到学习的规范空间，实现部分级分割增强，并使用规范表示实现跨连续帧的物体部分一致对齐。

### 主要发现

在Artic4D上的全面实验表明，所提出的CanonSeg4D在更复杂场景下的全景分割准确性优于最先进的方法，验证了时间建模和规范对齐在动态物体理解中的有效性。

### 结论

时间建模和规范对齐对动态物体理解至关重要，该研究为4D关节物体感知的未来进展铺平了道路。

### 翻译

关节物体感知在计算机视觉中提出了重大挑战，特别是因为大多数现有方法忽略了时间动态性，尽管这类物体本质上是动态的。4D时间数据在关节物体感知中尚未得到充分探索，在全景分割领域也尚未被研究。缺乏基准数据集进一步阻碍了该领域的发展。为此，我们引入了Artic4D作为从PartNet Mobility派生的新数据集，并增加了合成传感器数据，具有4D全景标注和关节参数。基于此数据集，我们提出了CanonSeg4D，一种新颖的4D全景分割框架。该方法明确估计每帧偏移量，将观察到的物体部分映射到学习的规范空间，从而增强部分级分割。该框架使用这种规范表示来实现跨连续帧的物体部分的一致对齐。在Artic4D上的全面实验表明，所提出的CanonSeg4D在更复杂场景下的全景分割准确性优于最先进的方法。这些发现强调了时间建模和规范对齐在动态物体理解中的有效性，并为4D关节物体感知的未来进展铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决铰接物体的4D全景分割问题。铰接物体（如柜子、抽屉、洗衣机等由多个部分通过关节连接而成）在日常生活中非常常见，理解它们的动态特性对机器人应用（如服务机器人、自动驾驶、医疗机器人）至关重要。现有方法主要针对刚性物体，忽略了铰接物体的时间动态特性，且缺乏专门的基准数据集，导致研究结果难以比较。解决这一问题对于机器人抓取、操作等任务非常重要，能够提升机器人与环境中常见铰接物体的交互能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：大多数4D分割方法是为自动驾驶场景设计的，假设物体是刚性的，不适用于铰接物体；现有变换方法基于实例质心，但铰接物体的质心会随铰接状态移动，导致表示不一致。作者借鉴了PST-Transformer用于特征提取，规范空间表示的概念（在铰接物体姿态估计中已有应用），以及Lovász-Softmax损失函数处理类别不平衡问题。核心创新是设计了规范模块，将点从4D点云序列变换到学习的规范空间，实现与铰接状态无关的一致表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用规范空间表示，学习一个与铰接状态无关的规范表示，使得不同铰接状态下的同一物体部分在规范空间中具有一致的表示，从而实现一致的实例分割。整体实现流程：1) 输入4D点云序列；2) 使用PST-Transformer backbone提取时空特征；3) 通过语义头预测每个点的语义标签；4) 通过规范模块将点变换到规范空间；5) 在规范空间中使用聚类算法将点分组为实例；6) 输出每个点在每一帧的语义和实例标签。训练时结合语义损失和规范变换损失进行优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Artic4D数据集：首个专门用于铰接物体4D全景分割的基准数据集；2) CanonSeg4D方法：首个专门为铰接物体设计的4D全景分割框架；3) 规范空间表示：使用与铰接状态无关的规范表示实现一致的实例分割；4) 时空特征建模：充分利用时间信息建模铰接物体的动态特性；5) 专门的损失函数：使用Lovász-Softmax损失处理类别不平衡问题。相比之前的工作，CanonSeg4D专门针对铰接物体设计，规范空间提供了一致的参考框架，而非常规的质心方法；充分利用时间信息而非仅使用静态数据或两个时间戳；具有更好的泛化能力，不需要多个类别特定模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了首个专门针对铰接物体的4D全景分割方法CanonSeg4D，通过规范空间表示和时空特征建模，显著提升了铰接物体在复杂场景下的分割性能，并发布了新的基准数据集Artic4D推动该领域研究。'}


### 论文摘要

Articulated object perception presents significant challenges in computer vision, particularly because most existing methods ignore temporal dynamics despite the inherently dynamic nature of such objects. The use of 4D temporal data has not been thoroughly explored in articulated object perception and remains unexamined for panoptic segmentation. The lack of a benchmark dataset further hurt this field. To this end, we introduce Artic4D as a new dataset derived from PartNet Mobility and augmented with synthetic sensor data, featuring 4D panoptic annotations and articulation parameters. Building on this dataset, we propose CanonSeg4D, a novel 4D panoptic segmentation framework. This approach explicitly estimates per-frame offsets mapping observed object parts to a learned canonical space, thereby enhancing part-level segmentation. The framework employs this canonical representation to achieve consistent alignment of object parts across sequential frames. Comprehensive experiments on Artic4D demonstrate that the proposed CanonSeg4D outperforms state of the art approaches in panoptic segmentation accuracy in more complex scenarios. These findings highlight the effectiveness of temporal modeling and canonical alignment in dynamic object understanding, and pave the way for future advances in 4D articulated object perception.

---

## 60. LiveStar: Live Streaming Assistant for Real-World Online Video Understanding

**论文链接:** [http://arxiv.org/abs/2511.05299v1](http://arxiv.org/abs/2511.05299v1)

**作者:** Zhenyu Yang, Kairui Zhang, Yuhang Hu, Bing Wang, Shengsheng Qian, Bin Wen, Fan Yang, Tingting Gao, Weiming Dong, Changsheng Xu

**发布时间:** 2025-11-07

**备注:** NeurIPS 2025 Accepted

### GPT解析

### 总结

LiveStar是一个创新的直播助手，通过自适应流式解码实现持续主动响应，解决了现有在线视频大语言模型在同时处理连续帧输入和确定最佳响应时间方面的局限性。

### 背景

现有的在线视频大语言模型(Video-LLMs)通常难以同时处理连续帧输入和确定最佳响应时间，这往往损害了实时响应能力和叙事连贯性。

### 目的

引入LiveStar，一个通过自适应流式解码实现持续主动响应的直播助手，以解决现有在线Video-LLMs的局限性。

### 方法

LiveStar包含三个主要创新：(1)一种训练策略，实现可变长度视频流的增量视频-语言对齐，保持动态演化帧序列的时间一致性；(2)一种响应-静默解码框架，通过单次前向验证确定最佳主动响应时间；(3)一种通过峰值结束内存压缩实现的内存感知加速，结合流式键值缓存，实现10+分钟视频的在线推理，推理速度提升1.53倍。同时构建了OmniStar数据集，包含15个多样化的真实场景和5个评估任务。

### 主要发现

在三个基准测试中，LiveStar展示了最先进的性能，与现有在线Video-LLMs相比，语义正确性平均提高19.5%，时间差异减少18.1%，所有五个OmniStar任务的FPS提高12.0%。

### 结论

LiveStar成功解决了在线视频理解中的实时响应和叙事连贯性问题，同时提高了推理速度，为在线视频理解领域提供了新的解决方案。

### 翻译

尽管离线视频理解中的视频大语言模型(Video-LLMs)取得了显著进展，但现有的在线Video-LLMs通常难以同时处理连续的逐帧输入并确定最佳响应时间，常常牺牲实时响应能力和叙事连贯性。为解决这些局限性，我们引入了LiveStar，这是一个开创性的直播助手，通过自适应流式解码实现持续主动响应。具体而言，LiveStar包含：(1)一种训练策略，实现可变长度视频流的增量视频-语言对齐，在动态演化的帧序列中保持时间一致性；(2)一种响应-静默解码框架，通过单次前向验证确定最佳主动响应时间；(3)通过峰值结束内存压缩实现的内存感知加速，用于10+分钟视频的在线推理，结合流式键值缓存实现1.53倍的更快推理速度。我们还构建了OmniStar数据集，这是一个全面的训练和基准测试数据集，包含15个多样化的真实场景和5个用于在线视频理解的评估任务。在三个基准测试中的大量实验证明了LiveStar的最先进性能，与现有在线Video-LLMs相比，语义正确性平均提高19.5%，时间差异减少18.1%，同时在所有五个OmniStar任务中FPS提高12.0%。我们的模型和数据集可在https://github.com/yzy-bupt/LiveStar获取。


### 论文摘要

Despite significant progress in Video Large Language Models (Video-LLMs) for offline video understanding, existing online Video-LLMs typically struggle to simultaneously process continuous frame-by-frame inputs and determine optimal response timing, often compromising real-time responsiveness and narrative coherence. To address these limitations, we introduce LiveStar, a pioneering live streaming assistant that achieves always-on proactive responses through adaptive streaming decoding. Specifically, LiveStar incorporates: (1) a training strategy enabling incremental video-language alignment for variable-length video streams, preserving temporal consistency across dynamically evolving frame sequences; (2) a response-silence decoding framework that determines optimal proactive response timing via a single forward pass verification; (3) memory-aware acceleration via peak-end memory compression for online inference on 10+ minute videos, combined with streaming key-value cache to achieve 1.53x faster inference. We also construct an OmniStar dataset, a comprehensive dataset for training and benchmarking that encompasses 15 diverse real-world scenarios and 5 evaluation tasks for online video understanding. Extensive experiments across three benchmarks demonstrate LiveStar's state-of-the-art performance, achieving an average 19.5% improvement in semantic correctness with 18.1% reduced timing difference compared to existing online Video-LLMs, while improving FPS by 12.0% across all five OmniStar tasks. Our model and dataset can be accessed at https://github.com/yzy-bupt/LiveStar.

---

## 61. M2S2L: Mamba-based Multi-Scale Spatial-temporal Learning for Video Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.05564v1](http://arxiv.org/abs/2511.05564v1)

**作者:** Yang Liu, Boan Chen, Xiaoguang Zhu, Jing Liu, Peng Sun, Wei Zhou

**发布时间:** 2025-11-04

**备注:** IEEE VCIP 2025

### GPT解析

### 总结

本文提出了一种基于Mamba的多尺度空间-时间学习框架（M2S2L），用于解决视频异常检测中检测精度与计算效率的平衡问题。

### 背景

视频异常检测是图像处理领域的重要任务，在视频监控中有广泛应用前景。随着视频内容日益复杂，包含多样的行为模式和情境场景，传统VAD方法难以提供稳健评估，且现有方法要么缺乏全面的空间-时间建模，要么需要过多计算资源用于实时应用。

### 目的

开发一种既能保持高检测精度又能满足实时应用计算效率的视频异常检测方法。

### 方法

提出Mamba-based多尺度空间-时间学习（M2S2L）框架，包含分层空间编码器在多个粒度上操作，多时间编码器捕获不同时间尺度的运动动态，以及特征分解机制实现外观和运动重建的任务特定优化。

### 主要发现

在三个基准数据集上取得优异性能：UCSD Ped2上帧级AUC达98.5%，CUHK Avenue上达92.1%，ShanghaiTech上达77.9%，同时保持20.1 GFLOPs计算量和45 FPS推理速度的高效率。

### 结论

M2S2L框架在保持高效率的同时实现了高精度，适合实际监控部署。

### 翻译

视频异常检测（VAD）是图像处理社区中的一项重要任务，在视频监控领域具有应用前景，但在平衡检测精度与计算效率方面面临基本挑战。随着视频内容因多样的行为模式和情境场景而变得越来越复杂，传统的VAD方法难以对现代监控系统提供稳健评估。现有方法要么缺乏全面的空间-时间建模，要么需要过多的计算资源用于实时应用。在这方面，本文提出了一个基于Mamba的多尺度空间-时间学习（M2S2L）框架。所提出的方法采用在多个粒度上操作的分层空间编码器和捕获不同时间尺度运动动态的多时间编码器。我们还引入了一种特征分解机制，使外观和运动重建能够进行任务特定优化，促进更细致的行为建模和质量感知的异常评估。在三个基准数据集上的实验表明，M2S2L框架在UCSD Ped2、CUHK Avenue和ShanghaiTech上分别实现了98.5%、92.1%和77.9%的帧级AUC，同时保持20.1 GFLOPs的计算效率和45 FPS的推理速度，使其适用于实际监控部署。


### 论文摘要

Video anomaly detection (VAD) is an essential task in the image processing community with prospects in video surveillance, which faces fundamental challenges in balancing detection accuracy with computational efficiency. As video content becomes increasingly complex with diverse behavioral patterns and contextual scenarios, traditional VAD approaches struggle to provide robust assessment for modern surveillance systems. Existing methods either lack comprehensive spatial-temporal modeling or require excessive computational resources for real-time applications. In this regard, we present a Mamba-based multi-scale spatial-temporal learning (M2S2L) framework in this paper. The proposed method employs hierarchical spatial encoders operating at multiple granularities and multi-temporal encoders capturing motion dynamics across different time scales. We also introduce a feature decomposition mechanism to enable task-specific optimization for appearance and motion reconstruction, facilitating more nuanced behavioral modeling and quality-aware anomaly assessment. Experiments on three benchmark datasets demonstrate that M2S2L framework achieves 98.5%, 92.1%, and 77.9% frame-level AUCs on UCSD Ped2, CUHK Avenue, and ShanghaiTech respectively, while maintaining efficiency with 20.1G FLOPs and 45 FPS inference speed, making it suitable for practical surveillance deployment.

---

## 62. Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection

**论文链接:** [http://arxiv.org/abs/2511.07301v1](http://arxiv.org/abs/2511.07301v1)

**作者:** Huizai Yao, Sicheng Zhao, Pengteng Li, Yi Cui, Shuo Lu, Weiyu Guo, Yunfan Lu, Yijie Xu, Hui Xiong

**发布时间:** 2025-11-10

**备注:** Accepted to AAAI 2026. Extended version with full Appendix

### GPT解析

### 总结

本文提出了一种新颖的源域免费目标检测(SFOD)框架，利用视觉基础模型(VFMs)作为外部知识源来增强特征对齐和标签质量，通过三个专门设计的模块解决了现有SFOD方法在泛化和伪标签质量方面的局限性。

### 背景

现有SFOD方法主要依赖源模型的内部知识，限制了跨域泛化能力，并常常导致有偏差的伪标签，影响可转移性和判别性。视觉基础模型(VFMs)虽具有强大的感知能力和广泛的泛化性，但在SFOD场景中的潜力尚未被充分利用。

### 目的

提出一个新的SFOD框架，利用VFMs作为外部知识源来联合增强特征对齐和标签质量，提高目标检测器在目标域上的性能。

### 方法

设计了三个基于VFMs的模块：(1)Patch-weighted Global Feature Alignment (PGFA)使用基于patch相似性的权重从VFMs中提取全局特征；(2)Prototype-based Instance Feature Alignment (PIFA)通过动量更新的VFM原型进行实例级对比学习；(3)Dual-source Enhanced Pseudo-label Fusion (DEPF)通过熵感知策略融合检测VFMs和教师模型的预测。

### 主要发现

在六个基准测试上的广泛实验表明，该方法实现了最先进的SFOD性能，验证了集成VFMs可以同时提高可转移性和判别性的有效性。

### 结论

利用VFMs作为外部知识源可以有效提升SFOD的性能，通过三个专门设计的模块解决了现有方法在泛化和伪标签质量方面的局限性。

### 翻译

无源域目标检测(SFOD)旨在将源预训练的目标检测器适应到目标域而无需访问源数据。然而，现有的SFOD方法主要依赖源模型的内部知识，这限制了它们跨域泛化的能力，并常常导致有偏差的伪标签，从而阻碍了可转移性和判别性。相比之下，在大量多样化数据上预训练的视觉基础模型(VFMs)表现出强大的感知能力和广泛的泛化性，但在SFOD设置中它们的潜力仍未被充分利用。在本文中，我们提出了一种新颖的SFOD框架，利用VFMs作为外部知识源来联合增强特征对齐和标签质量。具体来说，我们设计了三个基于VFMs的模块：(1)基于补丁权重的全局特征对齐(PGFA)使用基于补丁相似性的权重从VFMs中提取全局特征，以增强全局特征的可转移性；(2)基于原型的实例特征对齐(PIFA)由动量更新的VFM原型引导，执行实例级对比学习；(3)双源增强的伪标签融合(DEPF)通过熵感知策略融合检测VFMs和教师模型的预测，以获得更可靠的监督。在六个基准测试上的广泛实验证明，我们的方法实现了最先进的SFOD性能，验证了集成VFMs可以同时提高可转移性和判别性的有效性。


### 论文摘要

Source-Free Object Detection (SFOD) aims to adapt a source-pretrained object detector to a target domain without access to source data. However, existing SFOD methods predominantly rely on internal knowledge from the source model, which limits their capacity to generalize across domains and often results in biased pseudo-labels, thereby hindering both transferability and discriminability. In contrast, Vision Foundation Models (VFMs), pretrained on massive and diverse data, exhibit strong perception capabilities and broad generalization, yet their potential remains largely untapped in the SFOD setting. In this paper, we propose a novel SFOD framework that leverages VFMs as external knowledge sources to jointly enhance feature alignment and label quality. Specifically, we design three VFM-based modules: (1) Patch-weighted Global Feature Alignment (PGFA) distills global features from VFMs using patch-similarity-based weighting to enhance global feature transferability; (2) Prototype-based Instance Feature Alignment (PIFA) performs instance-level contrastive learning guided by momentum-updated VFM prototypes; and (3) Dual-source Enhanced Pseudo-label Fusion (DEPF) fuses predictions from detection VFMs and teacher models via an entropy-aware strategy to yield more reliable supervision. Extensive experiments on six benchmarks demonstrate that our method achieves state-of-the-art SFOD performance, validating the effectiveness of integrating VFMs to simultaneously improve transferability and discriminability.

---

## 63. Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.07295v1](http://arxiv.org/abs/2511.07295v1)

**作者:** Tianrui Song, Wen-Shuo Chao, Hao Liu

**发布时间:** 2025-11-10

**备注:** Accepted by AAAI2026

### GPT解析

### 总结

本文提出了一种名为LLMHNI的框架，利用大型语言模型生成的辅助信号来区分推荐系统中的困难样本和噪声样本，解决了传统方法中的困难-噪声混淆问题，显著提高了去噪和推荐性能。

### 背景

推荐系统训练中使用的隐式反馈不可避免地面临噪声问题，如误点击和位置偏差等。先前研究试图通过数据模式识别噪声样本并通过样本丢弃或重新加权来减轻其影响。

### 目的

解决噪声样本和困难样本之间的混淆问题，避免因错误去除困难样本而影响用户偏好建模，从而提高推荐系统的去噪和推荐性能。

### 方法

提出LLMHNI框架，利用大型语言模型生成两种辅助信号：1)用户-项目语义相关性，用于负采样中选择困难负样本并过滤噪声负样本；2)用户-项目交互中的逻辑相关性，用于识别困难样本和噪声样本。还提出目标对齐策略将LLM嵌入投影到优化空间，以及图对比学习策略抑制不可靠边。

### 主要发现

LLMHNI框架能有效区分困难样本和噪声样本，显著提高去噪效果和推荐系统性能。

### 结论

通过利用大型语言模型的能力，可以有效解决推荐系统中的困难-噪声混淆问题，提升推荐质量。

### 翻译

在训练推荐系统时使用的隐式反馈不可避免地因误点击和位置偏差等因素而面临噪声问题。先前研究试图通过数据模式（如更高的损失值）识别噪声样本，并通过样本丢弃或重新加权来减轻其影响。然而，我们观察到噪声样本和困难样本显示相似模式，导致困难-噪声混淆问题。这种混淆是有问题的，因为困难样本对建模用户偏好至关重要。为解决这个问题，我们提出了LLMHNI框架，利用大型语言模型生成的两种辅助用户-项目相关性信号来区分困难样本和噪声样本。LLMHNI从LLM编码的嵌入中获取用户-项目语义相关性，用于负采样中选择困难负样本，同时过滤掉噪声负样本。提出了一种目标对齐策略，将原本用于通用语言任务的LLM编码嵌入投影到针对用户-项目相关性建模优化的表示空间。LLMHNI还利用LLM推断的用户-项目交互中的逻辑相关性来识别困难样本和噪声样本。这些LLM推断的交互被整合到交互图中，并通过跨图对比对齐指导去噪。为消除LLM幻觉引起的不可靠交互的影响，我们提出了一种图对比学习策略，通过对随机边丢弃视图中的表示进行对齐来抑制不可靠边。实验结果表明，LLMHNI显著提高了去噪和推荐性能。


### 论文摘要

Implicit feedback, employed in training recommender systems, unavoidably confronts noise due to factors such as misclicks and position bias. Previous studies have attempted to identify noisy samples through their diverged data patterns, such as higher loss values, and mitigate their influence through sample dropping or reweighting. However, we observed that noisy samples and hard samples display similar patterns, leading to hard-noisy confusion issue. Such confusion is problematic as hard samples are vital for modeling user preferences. To solve this problem, we propose LLMHNI framework, leveraging two auxiliary user-item relevance signals generated by Large Language Models (LLMs) to differentiate hard and noisy samples. LLMHNI obtains user-item semantic relevance from LLM-encoded embeddings, which is used in negative sampling to select hard negatives while filtering out noisy false negatives. An objective alignment strategy is proposed to project LLM-encoded embeddings, originally for general language tasks, into a representation space optimized for user-item relevance modeling. LLMHNI also exploits LLM-inferred logical relevance within user-item interactions to identify hard and noisy samples. These LLM-inferred interactions are integrated into the interaction graph and guide denoising with cross-graph contrastive alignment. To eliminate the impact of unreliable interactions induced by LLM hallucination, we propose a graph contrastive learning strategy that aligns representations from randomly edge-dropped views to suppress unreliable edges. Empirical results demonstrate that LLMHNI significantly improves denoising and recommendation performance.

---

## 64. From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models Without Task Knowledge

**论文链接:** [http://arxiv.org/abs/2511.07049v1](http://arxiv.org/abs/2511.07049v1)

**作者:** Hui Lu, Yi Yu, Song Xia, Yiming Yang, Deepu Rajan, Boon Poh Ng, Alex Kot, Xudong Jiang

**发布时间:** 2025-11-10

**备注:** AAAI 2026 (Oral presentation)

### GPT解析

### 总结

本研究提出了一种名为可迁移视频攻击(TVA)的新型对抗攻击方法，利用视频基础模型的时间表示动力学创建有效扰动，无需访问目标任务或训练代理模型，实验证明该方法对下游模型和多模态大语言模型有效，揭示了视频模型部署中的安全漏洞。

### 背景

大规模视频基础模型(VFMs)已显著推进各种视频相关任务的发展，但其开源性也带来了严重的安全风险，攻击者可利用对VFMs的完整知识发起强大攻击。

### 目的

研究一种新颖且实用的对抗威胁场景：攻击基于开源VFMs微调的下游模型或MLLMs，无需访问目标任务、训练数据、模型查询和架构信息。

### 方法

提出可迁移视频攻击(TVA)，一种时间感知的对抗攻击方法，整合双向对比学习机制最大化干净和对抗特征差异，并引入时间一致性损失利用运动线索增强扰动顺序影响。

### 主要发现

TVA避免了训练昂贵的代理模型或访问特定领域数据的需求，提供更实用高效的攻击策略，在24个视频相关任务上证明了对下游模型和MLLMs的有效性。

### 结论

研究揭示了视频模型部署中一个先前未被充分探索的安全漏洞，强调了在开发和部署视频基础模型时需要考虑安全问题。

### 翻译

大规模视频基础模型(VFMs)已显著推进了各种视频相关任务的发展，无论是通过特定任务模型还是多模态大语言模型(MLLMs)。然而，VFMs的开源性也引入了严重的安全风险，因为攻击者可以利用对VFMs的完整知识来发起强大攻击。本文研究了一种新颖且实用的对抗威胁场景：攻击基于开源VFMs微调的下游模型或MLLMs，无需访问目标任务、训练数据、模型查询和架构信息。与依赖任务对齐代理模型的传统基于迁移的攻击不同，我们证明对抗性漏洞可以直接从VFMs中利用。为此，我们提出了可迁移视频攻击(TVA)，一种时间感知的对抗攻击方法，利用VFMs的时间表示动力学来创建有效扰动。TVA整合了双向对比学习机制，以最大化干净和对抗特征之间的差异，并引入了时间一致性损失，利用运动线索来增强扰动的顺序影响。TVA避免了训练昂贵的代理模型或访问特定领域数据的需求，从而提供了一种更实用和高效的攻击策略。在24个视频相关任务上的广泛实验证明了TVA对下游模型和MLLMs的有效性，揭示了视频模型部署中一个先前未被充分探索的安全漏洞。


### 论文摘要

Large-scale Video Foundation Models (VFMs) has significantly advanced various video-related tasks, either through task-specific models or Multi-modal Large Language Models (MLLMs). However, the open accessibility of VFMs also introduces critical security risks, as adversaries can exploit full knowledge of the VFMs to launch potent attacks. This paper investigates a novel and practical adversarial threat scenario: attacking downstream models or MLLMs fine-tuned from open-source VFMs, without requiring access to the victim task, training data, model query, and architecture. In contrast to conventional transfer-based attacks that rely on task-aligned surrogate models, we demonstrate that adversarial vulnerabilities can be exploited directly from the VFMs. To this end, we propose the Transferable Video Attack (TVA), a temporal-aware adversarial attack method that leverages the temporal representation dynamics of VFMs to craft effective perturbations. TVA integrates a bidirectional contrastive learning mechanism to maximize the discrepancy between the clean and adversarial features, and introduces a temporal consistency loss that exploits motion cues to enhance the sequential impact of perturbations. TVA avoids the need to train expensive surrogate models or access to domain-specific data, thereby offering a more practical and efficient attack strategy. Extensive experiments across 24 video-related tasks demonstrate the efficacy of TVA against downstream models and MLLMs, revealing a previously underexplored security vulnerability in the deployment of video models.

---

## 65. Design Principles of Zero-Shot Self-Supervised Unknown Emitter Detectors

**论文链接:** [http://arxiv.org/abs/2511.07026v1](http://arxiv.org/abs/2511.07026v1)

**作者:** Mikhail Krasnov, Ljupcho Milosheski, Mihael Mohorčič, Carolina Fortuna

**发布时间:** 2025-11-10

### GPT解析

### 总结

该研究针对无线设备激增背景下的未知发射器检测问题，提出了全面的系统评估方法，并引入了2D-星座数据模态、KANs网络和SVD初始化程序，显著提升了检测性能。

### 背景

无线设备的快速增长使得关键任务如频谱管理和网络安全需要更强大可靠的发射器检测与识别技术。

### 目的

对未知发射器检测系统进行全面评估，重点关注数据模态、学习方法和特征学习模块等设计空间的关键方面。

### 方法

提出2D-星座数据模态处理不同消息场景；引入可解释的Kolmogorov-Arnold Networks增强模型透明度；提出基于奇异值分解的特征学习模块初始化程序处理稀疏2D-星座数据；评估深度聚类、自编码器和对比学习三种学习方法。

### 主要发现

先前方法通常使用相同传输消息的数据集；2D-星座数据模态相比传统I/Q数据在ROC-AUC、NMI和F1指标上提升高达40%；SVD初始化的特征学习模块使深度聚类方法性能提升高达40%。

### 结论

通过系统评估和创新方法，显著提升了未知发射器检测系统的性能和可靠性，为频谱管理和网络安全提供了更有效的解决方案。

### 翻译

无线设备的激增对关键任务（如频谱管理和网络安全）中的发射器检测和识别提出了更强大可靠的需求。然而，探索未知发射器识别方法的现有研究通常受到标记数据或专有数据集的限制，存在不切实际的假设（如所有样本具有相同的传输消息），或缺乏对不同架构和设计维度的系统评估。在这项工作中，我们对未知发射器检测系统在设计空间的关键方面进行了全面评估，重点关注数据模态、学习方法和特征学习模块。我们证明了先前自监督、零样本发射器检测方法通常使用具有相同传输消息的数据集。为解决这一限制，我们提出了用于不同消息场景的2D-星座数据模态，相比传统原始I/Q数据，在ROC-AUC、NMI和F1指标上实现了高达40%的性能提升。此外，我们引入了可解释的Kolmogorov-Arnold Networks以增强模型透明度，以及基于奇异值分解的特征学习模块初始化程序，用于处理稀疏的2D-星座数据，使深度聚类方法的性能相比无SVD初始化的模块提升了高达40%。我们在三种学习方法（深度聚类、自编码器和对比学习）下评估了所有数据模态和学习模块。


### 论文摘要

The proliferation of wireless devices necessitates more robust and reliable emitter detection and identification for critical tasks such as spectrum management and network security. Existing studies exploring methods for unknown emitters identification, however, are typically hindered by their dependence on labeled or proprietary datasets, unrealistic assumptions (e.g. all samples with identical transmitted messages), or deficiency of systematic evaluations across different architectures and design dimensions. In this work, we present a comprehensive evaluation of unknown emitter detection systems across key aspects of the design space, focusing on data modality, learning approaches, and feature learn- ing modules. We demonstrate that prior self-supervised, zero-shot emitter detection approaches commonly use datasets with identical transmitted messages. To address this limitation, we propose a 2D- Constellation data modality for scenarios with varying messages, achieving up to a 40\% performance improvement in ROC-AUC, NMI, and F1 metrics compared to conventional raw I/Q data. Furthermore, we introduce interpretable Kolmogorov--Arnold Net- works (KANs) to enhance model transparency, and a Singular Value Decomposition (SVD)-based initialization procedure for feature learning modules operating on sparse 2D-Constellation data, which improves the performance of Deep Clustering approaches by up to 40\% across the same metrics comparing to the modules without SVD initialization. We evaluate all data modalities and learning modules across three learning approaches: Deep Clustering, Auto Encoder and Contrastive Learning.

---

## 66. S$^2$Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening

**论文链接:** [http://arxiv.org/abs/2511.07006v1](http://arxiv.org/abs/2511.07006v1)

**作者:** Bowei He, Bowen Gao, Yankai Chen, Yanyan Lan, Chen Ma, Philip S. Yu, Ya-Qin Zhang, Wei-Ying Ma

**发布时间:** 2025-11-10

**备注:** Accepted by AAAI 2026 Main Technical Track

### GPT解析

### 总结

研究提出了S²Drug两阶段框架，整合蛋白质序列信息和3D结构上下文，通过对比表示学习提高虚拟筛选性能，结合预训练和微调两个阶段，并引入辅助结合位点预测任务增强蛋白质-配体匹配。

### 背景

虚拟筛选是药物发现中的关键任务，专注于识别能与特定蛋白质口袋结合的小分子配体。现有深度学习方法主要依赖结构数据，忽略了更易获取且能提高泛化能力的蛋白质序列信息。然而，直接整合蛋白质序列面临大规模蛋白质-配体数据集中的冗余和噪声挑战。

### 目的

解决现有虚拟筛选方法忽视蛋白质序列信息的问题，处理大规模数据集中的冗余和噪声，开发能有效整合蛋白质序列和结构信息的框架，提高虚拟筛选性能。

### 方法

S²Drug采用两阶段框架：第一阶段使用基于ESM2的主干在ChemBL上进行蛋白质序列预训练，结合定制数据采样策略减少冗余和噪声；第二阶段通过残基级门控模块融合序列和结构信息，在PDBBind上微调，并引入辅助结合位点预测任务，指导模型定位结合残基并捕获其3D空间排列。

### 主要发现

S²Drug在多个基准测试中一致提高了虚拟筛选性能，并在结合位点预测上取得良好结果，证明了在对比学习中桥接序列和结构的价值。

### 结论

通过整合蛋白质序列信息和3D结构上下文，S²Drug有效解决了现有虚拟筛选方法的局限性，提高了虚拟筛选性能，展示了结合序列和结构信息在药物发现中的潜力。

### 翻译

虚拟筛选是药物发现中的基本任务，专注于识别能与特定蛋白质口袋结合的小分子配体。从早期的回归模型到最近的对比学习方法，现有的深度学习方法主要依赖结构数据，而忽略了更易获取且能提高泛化能力的蛋白质序列。然而，由于大规模蛋白质-配体数据集中的冗余和噪声，直接整合蛋白质序列存在挑战。为解决这些限制，我们提出了S²Drug，这是一个两阶段框架，明确地将蛋白质序列信息和3D结构上下文整合到蛋白质-配体对比表示学习中。在第一阶段，我们使用基于ESM2的主干在ChemBL上进行蛋白质序列预训练，结合定制的数据采样策略，减少蛋白质和配体两方面的冗余和噪声。在第二阶段，我们通过残基级门控模块融合序列和结构信息，并在PDBBind上进行微调，同时引入辅助结合位点预测任务。这个辅助任务指导模型准确地在蛋白质序列中定位结合残基并捕获其3D空间排列，从而优化蛋白质-配体匹配。在多个基准测试中，S²Drug一致提高了虚拟筛选性能，并在结合位点预测上取得了显著结果，证明了在对比学习中桥接序列和结构的价值。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有虚拟筛选方法过度依赖蛋白质3D结构而忽略蛋白质序列信息的问题。这个问题很重要，因为蛋白质序列比3D结构更容易获取，能增强模型泛化能力，且蛋白质序列包含蛋白质折叠和功能的基本信息，影响蛋白质与配体的相互作用。此外，确定3D结构技术复杂且昂贵，限制了大规模训练数据集的扩展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者基于'序列决定结构，结构决定功能'的基本原理，认识到现有方法仅利用预训练模型编码的序列表示而没有显式学习序列-配体相互作用。他们发现大规模蛋白质序列-配体数据集(如ChemBL)的潜力尚未被充分利用，但直接整合这些数据集面临冗余和噪声挑战。设计了两阶段框架：第一阶段在ChemBL上进行序列预训练，结合数据采样策略减少冗余和噪声；第二阶段在PDBBind上微调，通过门控模块融合序列和结构信息，并引入辅助结合位点预测任务。借鉴了ESM2模型、Uni-Mol编码器和对比学习等现有工作。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将蛋白质序列和3D结构信息结合起来进行虚拟筛选，通过两阶段训练框架和辅助任务增强模型对蛋白质空间结构的理解。整体流程分为两阶段：第一阶段是序列模型预训练，包括双边数据采样(减少蛋白质侧冗余和配体侧噪声)、使用ESM2和Uni-Mol编码器进行表示学习、以及对比训练对齐蛋白质和配体表示；第二阶段是序列-结构融合微调，包括通过残基级门控模块自适应融合序列和结构信息、引入辅助结合位点预测任务、以及结合对比损失和结合位点预测损失进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：两阶段对比表示学习框架结合序列和结构信息、双边数据采样策略减少数据集冗余和噪声、序列-结构融合模块通过门控机制自适应融合信息、辅助结合位点预测任务增强空间结构理解。相比之前工作，S2Drug是首个将蛋白质序列和3D结构信息显式结合用于虚拟筛选的对比学习框架，解决了之前方法过度依赖3D结构或仅简单利用序列表示的问题，通过创新的数据采样策略和融合机制显著提高了虚拟筛选的准确性和泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'S2Drug通过创新的两阶段对比学习框架有效整合蛋白质序列和3D结构信息，显著提高了虚拟筛选的准确性和泛化能力，同时为结合位点预测提供了新思路。'}


### 论文摘要

Virtual screening (VS) is an essential task in drug discovery, focusing on the identification of small-molecule ligands that bind to specific protein pockets. Existing deep learning methods, from early regression models to recent contrastive learning approaches, primarily rely on structural data while overlooking protein sequences, which are more accessible and can enhance generalizability. However, directly integrating protein sequences poses challenges due to the redundancy and noise in large-scale protein-ligand datasets. To address these limitations, we propose \textbf{S$^2$Drug}, a two-stage framework that explicitly incorporates protein \textbf{S}equence information and 3D \textbf{S}tructure context in protein-ligand contrastive representation learning. In the first stage, we perform protein sequence pretraining on ChemBL using an ESM2-based backbone, combined with a tailored data sampling strategy to reduce redundancy and noise on both protein and ligand sides. In the second stage, we fine-tune on PDBBind by fusing sequence and structure information through a residue-level gating module, while introducing an auxiliary binding site prediction task. This auxiliary task guides the model to accurately localize binding residues within the protein sequence and capture their 3D spatial arrangement, thereby refining protein-ligand matching. Across multiple benchmarks, S$^2$Drug consistently improves virtual screening performance and achieves strong results on binding site prediction, demonstrating the value of bridging sequence and structure in contrastive learning.

---

## 67. Beyond Observations: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning

**论文链接:** [http://arxiv.org/abs/2511.06854v1](http://arxiv.org/abs/2511.06854v1)

**作者:** Jiexi Liu, Meng Cao, Songcan Chen

**发布时间:** 2025-11-10

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

iTimER是一种自监督预训练框架，通过利用重建误差作为学习信号，为不规则采样时间序列表示学习提供了一种简单有效的方法。

### 背景

不规则采样时间序列在现实应用中普遍存在，特点是时间间隔不均匀且自然缺失。现有方法主要依赖观测值来推断未观测值或潜在动态，但忽略了训练过程中产生的重建误差这一重要学习信号源。

### 目的

提出iTimER框架，利用重建误差作为学习信号，提高不规则采样时间序列的表示学习效果。

### 方法

iTimER建模重建误差在观测值上的分布，通过混合策略为未观测时间戳生成伪观测值，使未观测时间戳成为噪声感知的训练目标。使用Wasserstein度量对齐不同区域间的重建误差分布，并通过对比学习增强表示的判别性。

### 主要发现

重建误差能反映模型捕获底层数据结构的程度，可作为未观测值的信息代理。通过将未观测时间戳转化为噪声感知的训练目标，可以实现有意义的重建信号。

### 结论

在分类、插值和预测任务上的实验表明，iTimER在ISTS设置下持续优于最先进的方法，证明了其有效性和实用性。

### 翻译

不规则采样时间序列具有非均匀时间间隔和自然缺失的特点，在现实世界中很常见。现有的ISTS建模方法主要依赖观测值来推断未观测值或潜在动态。然而，这些方法忽略了一个关键的学习信号来源：模型训练过程中自然产生的重建误差。这种误差隐式地反映了模型捕获底层数据结构的程度，并可作为未观测值的信息代理。为利用这一见解，我们提出了iTimER，一个简单而有效的自监督预训练框架，用于ISTS表示学习。iTimER建模重建误差在观测值上的分布，并通过采样误差和最后可用观测值之间的混合策略为未观测时间戳生成伪观测值。这使未观测时间戳成为噪声感知的训练目标，实现有意义的重建信号。Wasserstein度量对齐观测区域和伪观测区域之间的重建误差分布，同时对比学习目标增强了学习表示的判别性。在分类、插值和预测任务上的广泛实验表明，iTimER在ISTS设置下持续优于最先进的方法。


### 论文摘要

Irregularly sampled time series (ISTS), characterized by non-uniform time intervals with natural missingness, are prevalent in real-world applications. Existing approaches for ISTS modeling primarily rely on observed values to impute unobserved ones or infer latent dynamics. However, these methods overlook a critical source of learning signal: the reconstruction error inherently produced during model training. Such error implicitly reflects how well a model captures the underlying data structure and can serve as an informative proxy for unobserved values. To exploit this insight, we propose iTimER, a simple yet effective self-supervised pre-training framework for ISTS representation learning. iTimER models the distribution of reconstruction errors over observed values and generates pseudo-observations for unobserved timestamps through a mixup strategy between sampled errors and the last available observations. This transforms unobserved timestamps into noise-aware training targets, enabling meaningful reconstruction signals. A Wasserstein metric aligns reconstruction error distributions between observed and pseudo-observed regions, while a contrastive learning objective enhances the discriminability of learned representations. Extensive experiments on classification, interpolation, and forecasting tasks demonstrate that iTimER consistently outperforms state-of-the-art methods under the ISTS setting.

---

## 68. Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra

**论文链接:** [http://arxiv.org/abs/2511.06259v1](http://arxiv.org/abs/2511.06259v1)

**作者:** Yiwen Zhang, Keyan Ding, Yihang Wu, Xiang Zhuang, Yi Yang, Qiang Zhang, Huajun Chen

**发布时间:** 2025-11-09

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

GLMR是一种基于生成语言模型的检索框架，通过两阶段处理过程解决了现有方法中的质谱库覆盖不足和模态不匹配问题，大幅提高了检索准确性。

### 背景

从串联质谱中检索分子结构是快速化合物识别的关键步骤。现有的检索方法存在局限性：传统的质谱库匹配方法受限于质谱库覆盖范围，而最近的跨模态表示学习框架常常遇到模态不匹配问题，导致次优的检索精度和泛化能力。

### 目的

解决现有方法的局限性，提高分子结构检索的准确性和泛化能力。

### 方法

提出了GLMR框架，通过两阶段过程减轻跨模态不匹配：1) 预检索阶段：基于对比学习的模型识别候选分子作为输入质谱的上下文先验；2) 生成检索阶段：将候选分子与输入质谱整合，引导生成模型产生精细化的分子结构，然后基于分子相似度重新排序候选分子。

### 主要发现

在MassSpecGym和MassRET-20k数据集上的实验表明，GLMR显著优于现有方法，在top-1准确性上实现了超过40%的改进，并表现出强大的泛化能力。

### 结论

GLMR框架有效解决了跨模态不匹配问题，显著提高了分子结构检索的准确性和泛化能力。

### 翻译

从串联质谱中检索分子结构是快速化合物识别的关键步骤。现有的检索方法，如传统的质谱库匹配，受限于质谱库覆盖范围，而最近的跨模态表示学习框架常常遇到模态不匹配问题，导致次优的检索精度和泛化能力。为解决这些限制，我们提出了GLMR，一种基于生成语言模型的检索框架，通过两阶段过程减轻跨模态不匹配。在预检索阶段，基于对比学习的模型将候选分子识别为输入质谱的上下文先验。在生成检索阶段，这些候选分子与输入质谱整合，引导生成模型产生精细化的分子结构，然后基于分子相似度重新排序候选分子。在MassSpecGym和我们提出的MassRET-20k数据集上的实验表明，GLMR显著优于现有方法，在top-1准确性上实现了超过40%的改进，并表现出强大的泛化能力。


### 论文摘要

Retrieving molecular structures from tandem mass spectra is a crucial step in rapid compound identification. Existing retrieval methods, such as traditional mass spectral library matching, suffer from limited spectral library coverage, while recent cross-modal representation learning frameworks often encounter modality misalignment, resulting in suboptimal retrieval accuracy and generalization. To address these limitations, we propose GLMR, a Generative Language Model-based Retrieval framework that mitigates the cross-modal misalignment through a two-stage process. In the pre-retrieval stage, a contrastive learning-based model identifies top candidate molecules as contextual priors for the input mass spectrum. In the generative retrieval stage, these candidate molecules are integrated with the input mass spectrum to guide a generative model in producing refined molecular structures, which are then used to re-rank the candidates based on molecular similarity. Experiments on both MassSpecGym and the proposed MassRET-20k dataset demonstrate that GLMR significantly outperforms existing methods, achieving over 40% improvement in top-1 accuracy and exhibiting strong generalizability.

---

## 69. Adaptive Multi-view Graph Contrastive Learning via Fractional-order Neural Diffusion Networks

**论文链接:** [http://arxiv.org/abs/2511.06216v1](http://arxiv.org/abs/2511.06216v1)

**作者:** Yanan Zhao, Feng Ji, Jingyang Dai, Jiaze Ma, Keyue Jiang, Kai Zhao, Wee Peng Tay

**发布时间:** 2025-11-09

**备注:** Submitted to TPAMI

### GPT解析

### 总结

本文提出了一种基于分数阶连续动力学的无需增强的多视图图对比学习框架，通过可学习的分数导数阶数自动生成多样化的视图表示，实验证明该方法能产生更鲁棒和更具表达力的嵌入，优于现有最先进的图对比学习方法。

### 背景

现有的图对比学习方法通常依赖于固定的、手工制作的视图（通常是局部和全局视角），这限制了它们捕捉多尺度结构模式的能力。

### 目的

提出一种无需增强的、多视图的图对比学习框架，能够自动发现信息量大的视图，提高表示学习的鲁棒性和表达力。

### 方法

基于分数阶连续动力学，通过改变分数导数阶数使编码器产生连续的视图谱：小阶数产生局部特征，大阶数诱导更广泛的全球聚合。将阶数视为可学习参数，使模型能够适应数据的扩散尺度并自动发现信息量大的视图。

### 主要发现

通过将分数导数阶数作为可学习参数，模型能够自动适应数据的扩散尺度并发现信息量大的视图，无需手动增强即可生成多样化、互补的表示。

### 结论

在标准基准上的大量实验表明，该方法产生更鲁棒和更具表达力的嵌入，并且优于最先进的图对比学习基线。

### 翻译

图对比学习通过对同一图的多个视图进行对比来学习节点和图表示。现有方法通常依赖于固定的、手工制作的视图——通常是局部和全局视角，这限制了它们捕捉多尺度结构模式的能力。我们提出了一种无需增强的、多视图的图对比学习框架，该框架基于分数阶连续动力学。通过改变分数导数阶数，我们的编码器产生连续的视图谱：小阶数产生局部特征，而大阶数诱导更广泛的全局聚合。我们将阶数视为可学习参数，使模型能够适应数据的扩散尺度并自动发现信息量大的视图。这种原则性方法无需手动增强即可生成多样化、互补的表示。在标准基准上的大量实验证明，我们的方法产生更鲁棒和更具表达力的嵌入，并优于最先进的图对比学习基线。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决图对比学习(GCL)中多视图生成的局限性问题。现有方法通常依赖固定的、手工设计的视图(如局部和全局视角)，限制了捕获多尺度结构模式的能力。这个问题很重要，因为真实世界图数据往往包含复杂的多尺度结构信息，自适应生成多样化视图对于提升图表示学习性能至关重要，特别是在异质图(heterophilic graphs)中，简单方法往往效果不佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有图对比学习方法受限于固定视图，无法充分捕获多尺度结构。他们借鉴分数阶微分方程(FDEs)理论，特别是其在图扩散建模中的应用。作者的关键洞察是分数阶参数α可以控制扩散尺度，通过改变α产生连续视图谱(小α强调局部，大α强调全局)。基于此，他们设计了FD-MVGCL框架，将α作为可学习参数，并解决了维度坍塌和视图坍塌两个核心挑战。该方法借鉴了图神经网络、分数阶微分方程在GNN中的应用以及对比学习理论，但创新性地将这些思想整合用于自适应多视图生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用分数阶微分方程(FDEs)的连续特性，通过分数阶导数阶数α∈(0,1]控制图扩散的尺度，将α作为可学习参数使模型自适应地发现信息丰富的视图。小α值产生局部特征视图，大α值产生全局特征视图，从而生成多样互补的表示。整体流程：1)输入图数据；2)每个编码器进行特征投影、分数阶扩散和输出；3)对连续视图对应用正则化对比损失；4)使用自适应视图学习算法动态选择编码器数量和α值；5)下游任务中计算多个视图的加权平均作为最终表示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次利用分数阶动力学的多尺度行为进行对比学习，通过α参数生成连续视图谱；2)提供了严格的理论分析，证明不同α值产生的特征表示的可区分性；3)通过小分数阶编码器缓解维度坍塌，通过正则化对比目标防止视图坍塌；4)将α作为可学习参数，提出自适应视图学习算法；5)提供了稳定性分析，量化了扰动影响。相比之前工作，FD-MVGCL无需手工设计视图、无需负样本、无需手动调参，在多种图数据上实现了更优性能和更强鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了基于分数阶神经扩散网络的自适应多视图图对比学习框架，通过可学习的分数阶参数α生成连续的多尺度视图，无需手动增强或负样本，在多种图数据上实现了最先进的性能并表现出更强的鲁棒性。'}


### 论文摘要

Graph contrastive learning (GCL) learns node and graph representations by contrasting multiple views of the same graph. Existing methods typically rely on fixed, handcrafted views-usually a local and a global perspective, which limits their ability to capture multi-scale structural patterns. We present an augmentation-free, multi-view GCL framework grounded in fractional-order continuous dynamics. By varying the fractional derivative order $\alpha \in (0,1]$, our encoders produce a continuous spectrum of views: small $\alpha$ yields localized features, while large $\alpha$ induces broader, global aggregation. We treat $\alpha$ as a learnable parameter so the model can adapt diffusion scales to the data and automatically discover informative views. This principled approach generates diverse, complementary representations without manual augmentations. Extensive experiments on standard benchmarks demonstrate that our method produces more robust and expressive embeddings and outperforms state-of-the-art GCL baselines.

---

## 70. EMOD: A Unified EEG Emotion Representation Framework Leveraging V-A Guided Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2511.05863v1](http://arxiv.org/abs/2511.05863v1)

**作者:** Yuning Chen, Sha Zhao, Shijian Li, Gang Pan

**发布时间:** 2025-11-08

### GPT解析

### 总结

本文提出了EMOD框架，一种利用效价-唤醒(V-A)引导的对比学习的统一EEG情感表示框架，解决了现有深度学习方法在跨数据集泛化上的局限性，通过统一表示和对比学习提高了模型对不同EEG格式的适应能力。

### 背景

情感识别在情感计算中至关重要，深度学习已被广泛应用于EEG信号的情感识别。现有深度学习方法在单一EEG情感数据集上表现良好，但在跨数据集泛化能力上有限，这种局限性源于标注方案和数据格式的异质性。现有模型通常需要针对特定数据集的架构，缺乏跨不同情感标签的语义对齐。

### 目的

解决现有模型在跨数据集泛化上的局限性，弥合语义和结构差距，创建一个统一的EEG情感表示框架。

### 方法

提出了EMOD框架：利用效价-唤醒(V-A)引导的对比学习的统一EEG情感表示框架。通过将离散和连续情感标签投影到统一的V-A空间，学习可迁移和情感感知的表示。设计了软监督对比损失函数，鼓励情感相似的样本在潜在空间中聚类。采用灵活的主干网络，包括三域编码器和时空Transformer，以适应可变的EEG格式，能够稳健地提取和整合时间、频谱和空间特征。

### 主要发现

在八个公共EEG数据集上预训练EMOD，在三个基准数据集上评估其性能。EMOD实现了最先进的性能，展示了在多样化EEG情感识别场景中的强适应性和泛化能力。

### 结论

EMOD框架有效解决了EEG情感识别中的跨数据集泛化问题，通过统一表示和对比学习，提高了模型对不同EEG格式的适应能力。

### 翻译

从脑电信号中进行情感识别对于情感计算至关重要，并已使用深度学习得到广泛探索。虽然最近的深度学习方法在单一脑电情感数据集上取得了强大性能，但由于标注方案和数据格式的异质性，它们在跨数据集泛化方面仍然有限。现有模型通常需要针对输入结构定制特定数据集的架构，并且缺乏跨多样化情感标签的语义对齐。为了解决这些挑战，我们提出了EMOD：一种利用效价-唤醒(V-A)引导的对比学习的统一脑电情感表示框架。EMOD通过弥合语义和结构差距，从异构数据集中学习可迁移和情感感知的表示。具体而言，我们将离散和连续情感标签投影到统一的V-A空间，并制定了一个软监督对比损失，鼓励情感相似的样本在潜在空间中聚类。为了适应可变的脑电格式，EMOD采用了一个灵活的主干网络，包括三域编码器和时空Transformer，能够稳健地提取和整合时间、频谱和空间特征。我们在八个公共脑电数据集上预训练EMOD，并在三个基准数据集上评估其性能。实验结果表明，EMOD实现了最先进的性能，展示了在多样化基于脑电的情感识别场景中的强适应性和泛化能力。


### 论文摘要

Emotion recognition from EEG signals is essential for affective computing and has been widely explored using deep learning. While recent deep learning approaches have achieved strong performance on single EEG emotion datasets, their generalization across datasets remains limited due to the heterogeneity in annotation schemes and data formats. Existing models typically require dataset-specific architectures tailored to input structure and lack semantic alignment across diverse emotion labels. To address these challenges, we propose EMOD: A Unified EEG Emotion Representation Framework Leveraging Valence-Arousal (V-A) Guided Contrastive Learning. EMOD learns transferable and emotion-aware representations from heterogeneous datasets by bridging both semantic and structural gaps. Specifically, we project discrete and continuous emotion labels into a unified V-A space and formulate a soft-weighted supervised contrastive loss that encourages emotionally similar samples to cluster in the latent space. To accommodate variable EEG formats, EMOD employs a flexible backbone comprising a Triple-Domain Encoder followed by a Spatial-Temporal Transformer, enabling robust extraction and integration of temporal, spectral, and spatial features. We pretrain EMOD on eight public EEG datasets and evaluate its performance on three benchmark datasets. Experimental results show that EMOD achieves state-of-the-art performance, demonstrating strong adaptability and generalization across diverse EEG-based emotion recognition scenarios.

---

## 71. Cross-domain EEG-based Emotion Recognition with Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2511.05293v1](http://arxiv.org/abs/2511.05293v1)

**作者:** Rui Yan, Yibo Li, Han Ding, Fei Wang

**发布时间:** 2025-11-07

**备注:** 5 pages

### GPT解析

### 总结

本研究提出了EmotionCLIP模型，将EEG情感识别重新表述为CLIP框架内的EEG-文本匹配任务，使用SST-LegoViT骨干网络捕获多维度特征，实验结果显著优于现有模型。

### 背景

基于脑电图(EEG)的情感识别对于情感计算至关重要，但面临特征利用和跨领域泛化的挑战。

### 目的

引入EmotionCLIP模型，将识别任务重新表述为CLIP框架内的EEG-文本匹配任务，提高EEG情感识别的准确性和泛化能力。

### 方法

设计了专门的骨干网络SST-LegoViT，使用多尺度卷积和Transformer模块捕获空间、频谱和时间特征，采用多模态对比学习方法进行训练。

### 主要发现

在SEED和SEED-IV数据集上，跨主体准确率分别达到88.69%和73.50%，跨时间准确率分别达到88.46%和77.54%，显著优于现有模型。

### 结论

多模态对比学习对稳健的EEG情感识别有效，EmotionCLIP模型在跨主体和跨时间识别任务中表现优异。

### 翻译

基于脑电图(EEG)的情感识别对情感计算至关重要，但在特征利用和跨领域泛化方面面临挑战。本研究引入了EmotionCLIP，它将识别任务重新表述为CLIP框架内的EEG-文本匹配任务。一个专门的骨干网络SST-LegoViT使用多尺度卷积和Transformer模块捕获空间、频谱和时间特征。在SEED和SEED-IV数据集上的实验显示出了卓越的跨主体准确率88.69%和73.50%，以及跨时间准确率88.46%和77.54%，优于现有模型。结果表明多模态对比学习对稳健的EEG情感识别有效。


### 论文摘要

Electroencephalogram (EEG)-based emotion recognition is vital for affective computing but faces challenges in feature utilization and cross-domain generalization. This work introduces EmotionCLIP, which reformulates recognition as an EEG-text matching task within the CLIP framework. A tailored backbone, SST-LegoViT, captures spatial, spectral, and temporal features using multi-scale convolution and Transformer modules. Experiments on SEED and SEED-IV datasets show superior cross-subject accuracies of 88.69% and 73.50%, and cross-time accuracies of 88.46% and 77.54%, outperforming existing models. Results demonstrate the effectiveness of multimodal contrastive learning for robust EEG emotion recognition.

---

## 72. A Dual-stage Prompt-driven Privacy-preserving Paradigm for Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.05092v1](http://arxiv.org/abs/2511.05092v1)

**作者:** Ruolin Li, Min Liu, Yuan Bian, Zhaoyang Li, Yuzhen Li, Xueping Wang, Yaonan Wang

**发布时间:** 2025-11-07

**备注:** 10 pages, 6 figures

### GPT解析

### 总结

本文提出了一种双阶段提示驱动隐私保护范式(DPPP)，用于解决虚拟数据集在行人再识别模型训练中面临的构建复杂和域泛化能力差的问题。通过生成丰富提示和提示驱动解缠机制，构建了大规模虚拟数据集GenePerson，并实现了最先进的泛化性能。

### 背景

随着对数据隐私的关注增加，研究人员开始使用虚拟数据替代敏感的真实世界图像来训练行人再识别(Re-ID)模型。然而，现有的游戏引擎生成的虚拟数据集面临构建复杂和域泛化能力差等挑战，难以在实际场景中应用。

### 目的

解决现有虚拟数据集在行人再识别模型训练中面临的构建复杂和域泛化能力差的问题，提高模型在实际场景中的应用能力。

### 方法

提出双阶段提示驱动隐私保护范式(DPPP)：第一阶段生成包含行人外观、光照和视点等多维属性的丰富提示，驱动扩散模型端到端合成多样化数据，构建GenePerson虚拟数据集；第二阶段提出提示驱动解缠机制(PDM)，通过对比学习和文本反转网络将图像映射为风格和内容的伪词，构建风格解缠的内容提示，引导模型学习域不变内容特征。

### 主要发现

在GenePerson数据集上使用PDM训练的模型达到了最先进的泛化性能，超过了在流行真实和虚拟Re-ID数据集上的表现。

### 结论

通过DPPP范式和PDM机制，有效解决了虚拟数据集在行人再识别模型训练中的域泛化问题，提高了模型在实际场景中的应用能力。

### 翻译

随着对数据隐私问题的日益关注，研究人员开始使用虚拟数据作为替代敏感真实世界图像的方案，用于训练行人再识别(Re-ID)模型。然而，现有由游戏引擎生成的虚拟数据集仍面临构建复杂和域泛化能力差等挑战，难以在实际场景中应用。为解决这些挑战，我们提出了一种双阶段提示驱动隐私保护范式(DPPP)。在第一阶段，我们生成包含行人外观、光照和视点等多维属性的丰富提示，驱动扩散模型端到端合成多样化数据，构建了一个包含130,519张图像和6,641个身份的大规模虚拟数据集GenePerson。在第二阶段，我们提出了一种提示驱动解缠机制(PDM)，学习域不变泛化特征。借助对比学习，我们使用两个文本反转网络将图像分别映射为表示风格和内容的伪词，从而构建风格解缠的内容提示，引导模型在图像层面学习域不变内容特征。实验证明，在GenePerson上使用PDM训练的模型实现了最先进的泛化性能，超过了在流行真实和虚拟Re-ID数据集上的表现。


### 论文摘要

With growing concerns over data privacy, researchers have started using virtual data as an alternative to sensitive real-world images for training person re-identification (Re-ID) models. However, existing virtual datasets produced by game engines still face challenges such as complex construction and poor domain generalization, making them difficult to apply in real scenarios. To address these challenges, we propose a Dual-stage Prompt-driven Privacy-preserving Paradigm (DPPP). In the first stage, we generate rich prompts incorporating multi-dimensional attributes such as pedestrian appearance, illumination, and viewpoint that drive the diffusion model to synthesize diverse data end-to-end, building a large-scale virtual dataset named GenePerson with 130,519 images of 6,641 identities. In the second stage, we propose a Prompt-driven Disentanglement Mechanism (PDM) to learn domain-invariant generalization features. With the aid of contrastive learning, we employ two textual inversion networks to map images into pseudo-words representing style and content, respectively, thereby constructing style-disentangled content prompts to guide the model in learning domain-invariant content features at the image level. Experiments demonstrate that models trained on GenePerson with PDM achieve state-of-the-art generalization performance, surpassing those on popular real and virtual Re-ID datasets.

---

## 73. Medical Referring Image Segmentation via Next-Token Mask Prediction

**论文链接:** [http://arxiv.org/abs/2511.05044v1](http://arxiv.org/abs/2511.05044v1)

**作者:** Xinyu Chen, Yiran Wang, Gaoyang Pang, Jiafu Hao, Chentao Yue, Luping Zhou, Yonghui Li

**发布时间:** 2025-11-07

**备注:** This work has been submitted to the IEEE Transactions on Medical  Imaging for possible publication

### GPT解析

### 总结

本文提出了一种名为NTP-MRISeg的新型医学图像分割框架，将医学图像分割任务重新表述为自回归下一个令牌预测任务，并通过三种创新策略解决了该表述下的挑战，在多个数据集上实现了最先进的性能。

### 背景

医学图像分割(MRIS)涉及基于自然语言描述分割医学图像中的目标区域。虽然现有方法取得了有前景的结果，但通常涉及复杂的多模态融合或多阶段解码器设计。

### 目的

开发一种简化的医学图像分割框架，消除对模态特定融合和外部分割模型的需求，支持统一的端到端训练架构，并提高模型的泛化能力和适应性。

### 方法

将MRIS重新表述为对标记化的图像、文本和掩码表示的统一多模态序列的自回归下一个令牌预测任务。提出三种创新策略：(1)Next-k Token Prediction (NkTP)方案减少累积预测误差；(2)Token-level Contrastive Learning (TCL)增强边界敏感性并缓解长尾分布效应；(3)基于内存的Hard Error Token (HET)优化策略在训练中强调困难令牌。

### 主要发现

在QaTa-COV19和MosMedData+数据集上的大量实验表明，NTP-MRISeg实现了新的最先进性能，为传统MRIS流程提供了一种简化且有效的替代方案。

### 结论

NTP-MRISeg框架通过统一的多模态序列自回归预测方法，简化了医学图像分割任务的设计，同时提高了性能，是一种有前景的替代传统MRIS流程的方法。

### 翻译

医学图像分割涉及基于自然语言描述分割医学图像中的目标区域。虽然取得了有前景的结果，但最近的方法通常涉及复杂的多模态融合或多阶段解码器设计。在这项工作中，我们提出了NTP-MRISeg，一种新型框架，将MRIS重新表述为对标记化的图像、文本和掩码表示的统一多模态序列的自回归下一个令牌预测任务。这种表述通过消除对模态特定融合和外部分割模型的需求，简化了模型设计，支持统一的端到端训练架构。它还允许使用新兴的大规模多模态模型的预训练令牌化器，提高泛化能力和适应性。更重要的是，为了解决这种表述下的挑战，如曝光偏差、长尾令牌分布和细粒度病变边缘，我们提出了三种创新策略：(1)Next-k Token Prediction (NkTP)方案减少累积预测误差；(2)Token-level Contrastive Learning (TCL)增强边界敏感性并缓解长尾分布效应；(3)基于内存的Hard Error Token (HET)优化策略在训练中强调困难令牌。在QaTa-COV19和MosMedData+数据集上的大量实验表明，NTP-MRISeg实现了新的最先进性能，为传统MRIS流程提供了一种简化且有效的替代方案。


### 论文摘要

Medical Referring Image Segmentation (MRIS) involves segmenting target regions in medical images based on natural language descriptions. While achieving promising results, recent approaches usually involve complex design of multimodal fusion or multi-stage decoders. In this work, we propose NTP-MRISeg, a novel framework that reformulates MRIS as an autoregressive next-token prediction task over a unified multimodal sequence of tokenized image, text, and mask representations. This formulation streamlines model design by eliminating the need for modality-specific fusion and external segmentation models, supports a unified architecture for end-to-end training. It also enables the use of pretrained tokenizers from emerging large-scale multimodal models, enhancing generalization and adaptability. More importantly, to address challenges under this formulation-such as exposure bias, long-tail token distributions, and fine-grained lesion edges-we propose three novel strategies: (1) a Next-k Token Prediction (NkTP) scheme to reduce cumulative prediction errors, (2) Token-level Contrastive Learning (TCL) to enhance boundary sensitivity and mitigate long-tail distribution effects, and (3) a memory-based Hard Error Token (HET) optimization strategy that emphasizes difficult tokens during training. Extensive experiments on the QaTa-COV19 and MosMedData+ datasets demonstrate that NTP-MRISeg achieves new state-of-the-art performance, offering a streamlined and effective alternative to traditional MRIS pipelines.

---

## 74. Dynamic Residual Encoding with Slide-Level Contrastive Learning for End-to-End Whole Slide Image Representation

**论文链接:** [http://arxiv.org/abs/2511.05034v1](http://arxiv.org/abs/2511.05034v1)

**作者:** Jing Jin, Xu Liu, Te Gao, Zhihong Shi, Yixiong Liang, Ruiqing Zheng, Hulin Kuang, Min Zeng, Shichao Kan

**发布时间:** 2025-11-07

**DOI:** 10.1145/3746027.3755469

**备注:** 8pages, 3figures, published to ACM Digital Library

### GPT解析

### 总结

本文提出了一种动态残差编码与幻灯片级别对比学习(DRE-SLCL)方法，用于解决全幻灯片图像(WSI)端到端表示训练中的挑战，并在多种癌症相关任务上验证了其有效性。

### 背景

全幻灯片图像(WSI)表示对于癌症亚型分型、癌症识别和突变预测至关重要。然而，训练端到端的WSI表示模型存在重大挑战，因为一个标准的千兆像素幻灯片可包含数万个图像块，受当前GPU限制，难以在单个小批量中计算所有图像块的梯度。

### 目的

提出一种动态残差编码与幻灯片级别对比学习(DRE-SLCL)方法，用于端到端的WSI表示，以解决GPU计算限制问题。

### 方法

使用内存库存储所有WSI的图像块特征；训练时，对于小批量中的每个WSI，随机采样部分图像块并计算其特征，同时从内存库中检索同一WSI的额外图像块特征；采用残差编码技术结合这些特征生成个体WSI表示；最后基于WSI表示和组织病理学报告计算幻灯片级别的对比损失。

### 主要发现

在癌症亚型分型、癌症识别和突变预测任务上进行的实验证明了所提出的DRE-SLCL方法的有效性。

### 结论

DRE-SLCL方法是一种有效的端到端WSI表示方法，能够克服GPU计算限制，在多种癌症相关任务上表现良好。

### 翻译

全幻灯片图像(WSI)表示对于癌症亚型分型、癌症识别和突变预测至关重要。训练端到端的WSI表示模型存在重大挑战，因为一个标准的千兆像素幻灯片可以包含数万个图像块，受当前GPU限制，难以在单个小批量中计算所有图像块的梯度。为应对这一挑战，我们提出了一种动态残差编码与幻灯片级别对比学习(DRE-SLCL)方法用于端到端的WSI表示。我们的方法使用内存库存储数据集中所有WSI的图像块特征。训练时，一个小批量通常包含多个WSI。对于批量中的每个WSI，随机采样一个子集的图像块，并使用图像块编码器计算其特征。然后从内存库中选择来自同一WSI的额外图像块特征。使用残差编码技术生成每个个体WSI的表示，该技术结合了采样特征和从内存库中检索的特征。最后，基于小批量内WSI的表示和组织病理学报告计算幻灯片级别的对比损失。在癌症亚型分型、癌症识别和突变预测任务上进行的实验证明了所提出的DRE-SLCL方法的有效性。


### 论文摘要

Whole Slide Image (WSI) representation is critical for cancer subtyping, cancer recognition and mutation prediction.Training an end-to-end WSI representation model poses significant challenges, as a standard gigapixel slide can contain tens of thousands of image tiles, making it difficult to compute gradients of all tiles in a single mini-batch due to current GPU limitations. To address this challenge, we propose a method of dynamic residual encoding with slide-level contrastive learning (DRE-SLCL) for end-to-end WSI representation. Our approach utilizes a memory bank to store the features of tiles across all WSIs in the dataset. During training, a mini-batch usually contains multiple WSIs. For each WSI in the batch, a subset of tiles is randomly sampled and their features are computed using a tile encoder. Then, additional tile features from the same WSI are selected from the memory bank. The representation of each individual WSI is generated using a residual encoding technique that incorporates both the sampled features and those retrieved from the memory bank. Finally, the slide-level contrastive loss is computed based on the representations and histopathology reports ofthe WSIs within the mini-batch. Experiments conducted over cancer subtyping, cancer recognition, and mutation prediction tasks proved the effectiveness of the proposed DRE-SLCL method.

---

## 75. RCMCL: A Unified Contrastive Learning Framework for Robust Multi-Modal (RGB-D, Skeleton, Point Cloud) Action Understanding

**论文链接:** [http://arxiv.org/abs/2511.04351v2](http://arxiv.org/abs/2511.04351v2)

**作者:** Hasan Akgul, Mari Eplik, Javier Rojas, Akira Yamamoto, Rajesh Kumar, Maya Singh

**发布时间:** 2025-11-06

**备注:** 11 pages, 6 figures,

### GPT解析

### 总结

本文提出了一种名为鲁棒跨模态对比学习(RCMCL)的自监督框架，用于解决多模态人类动作识别在传感器故障或噪声情况下的性能下降问题。该框架通过联合优化跨模态对比目标、模态内自蒸馏目标和退化模拟目标，学习模态不变表示，并引入自适应模态门控网络实现鲁棒融合。

### 背景

多模态输入（RGB-D、骨骼、点云）的人类动作识别虽然可以实现高精度，但通常依赖于大型标记数据集，并且在传感器故障或噪声情况下性能会急剧下降。

### 目的

开发一个自监督框架，学习模态不变表示，并在模态丢失和损坏情况下保持可靠性，以解决多模态HAR在实际应用中的脆弱性问题。

### 方法

RCMCL框架联合优化三个目标：(1)跨模态对比目标对齐异构流，(2)模态内自蒸馏目标提高视图不变性并减少冗余，(3)退化模拟目标训练模型从掩码或损坏输入中恢复。同时，引入自适应模态门控网络为每个模态分配数据驱动的可靠性权重。

### 主要发现

在NTU RGB+D 120 (CS/CV)和UWA3D-II数据集上，RCMCL在标准设置中达到最先进准确率，并且在严重双模态丢失情况下仅显示11.5%的性能下降，显著优于监督融合基线。

### 结论

自监督跨模态对齐，结合明确的退化建模和自适应融合，是开发可靠可部署的多模态HAR系统的关键。

### 翻译

多模态输入（RGB-D、骨骼、点云）的人类动作识别(HAR)可以实现高精度，但通常依赖于大型标记数据集，并且在传感器故障或噪声情况下性能会急剧下降。我们提出了鲁棒跨模态对比学习(RCMCL)，这是一个自监督框架，学习模态不变表示，并在模态丢失和损坏情况下保持可靠性。RCMCL联合优化了(i)跨模态对比目标，对齐异构流；(ii)模态内自蒸馏目标，提高视图不变性并减少冗余；以及(iii)退化模拟目标，明确训练模型从掩码或损坏输入中恢复。在推理时，自适应模态门控(AMG)网络为每个模态分配数据驱动的可靠性权重以实现鲁棒融合。在NTU RGB+D 120 (CS/CV)和UWA3D-II上，RCMCL在标准设置中达到最先进准确率，并表现出明显更好的鲁棒性：在严重的双模态丢失情况下仅显示11.5%的性能下降，显著优于强大的监督融合基线。这些结果表明，自监督跨模态对齐，结合明确的退化建模和自适应融合，是可部署的多模态HAR的关键。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态人体动作识别系统的两个关键挑战：一是严重依赖大量标注数据，二是实际部署中因传感器故障或噪声导致的性能严重下降。这个问题在现实中非常重要，因为人体动作识别系统需要在各种条件下可靠工作，包括传感器可能失效或数据被噪声污染的情况；同时，数据标注成本高昂，且多模态系统在部分模态丢失时性能会显著下降，限制了其在实际应用中的部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到多模态动作识别的优势（互补信息）和挑战（异构数据融合、标注需求、鲁棒性问题）。然后借鉴了对比学习在自监督表征学习中的成功经验，特别是跨模态对比学习。作者设计了三个核心组件：跨模态一致性损失（LCM）用于特征对齐，模态内自蒸馏损失（LIM）提高内部表征质量，退化模拟损失（Ldeg）和自适应模态门控（AMG）提高鲁棒性。这些设计借鉴了现有工作中的对比学习、跨模态一致性学习、自蒸馏、退化模拟和自适应门控机制等思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是学习一个统一的、模态不变的特征空间，使表征对噪声和模态缺失具有内在鲁棒性；通过跨模态对比学习强制不同模态的特征对齐；显式训练模型以应对和补偿模态丢失和损坏；使用自适应门控机制动态调整不同模态的权重。整体实现流程分为四个阶段：1）模态特定特征编码（使用三种不同编码器处理RGB-D、骨骼和点云数据）；2）自监督跨模态预训练（包括LCM、LIM和Ldeg三种损失函数）；3）模态自适应鲁棒融合（通过AMG网络动态加权融合特征）；4）训练流程（预训练阶段优化所有组件，微调阶段训练分类器）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）统一的三模态自监督对比学习框架，同时处理RGB-D、骨骼和点云三种异构模态；2）创新的对比损失设计，结合跨模态一致性和模态内自蒸馏；3）退化模拟与自适应门控机制，处理模态丢失和噪声问题；4）联合优化策略，统一训练所有组件。相比之前工作，RCMCL首次统一处理三种不同的3D数据类型，显式解决实际部署中的模态丢失和噪声问题，结合多种自监督技术在一个框架中，并引入自适应门控机制根据输入质量动态调整模态权重。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RCMCL通过统一的自监督跨模态对比学习框架，结合创新的退化模拟和自适应门控机制，显著提高了多模态人体动作识别系统在数据标注有限和传感器故障条件下的准确性和鲁棒性。'}


### 论文摘要

Human action recognition (HAR) with multi-modal inputs (RGB-D, skeleton, point cloud) can achieve high accuracy but typically relies on large labeled datasets and degrades sharply when sensors fail or are noisy. We present Robust Cross-Modal Contrastive Learning (RCMCL), a self-supervised framework that learns modality-invariant representations and remains reliable under modality dropout and corruption. RCMCL jointly optimizes (i) a cross-modal contrastive objective that aligns heterogeneous streams, (ii) an intra-modal self-distillation objective that improves view-invariance and reduces redundancy, and (iii) a degradation simulation objective that explicitly trains models to recover from masked or corrupted inputs. At inference, an Adaptive Modality Gating (AMG) network assigns data-driven reliability weights to each modality for robust fusion. On NTU RGB+D 120 (CS/CV) and UWA3D-II, RCMCL attains state-of-the-art accuracy in standard settings and exhibits markedly better robustness: under severe dual-modality dropout it shows only an 11.5% degradation, significantly outperforming strong supervised fusion baselines. These results indicate that self-supervised cross-modal alignment, coupled with explicit degradation modeling and adaptive fusion, is key to deployable multi-modal HAR.

---

## 76. C3-Diff: Super-resolving Spatial Transcriptomics via Cross-modal Cross-content Contrastive Diffusion Modelling

**论文链接:** [http://arxiv.org/abs/2511.05571v1](http://arxiv.org/abs/2511.05571v1)

**作者:** Xiaofei Wang, Stephen Price, Chao Li

**发布时间:** 2025-11-04

### GPT解析

### 总结

该研究提出了一个名为C3-Diff的跨模态跨内容对比扩散框架，用于空间转录组学增强，通过整合组织学图像提高基因表达分辨率

### 背景

空间转录组学的发展使在原始组织中测量基因表达成为可能，但当前ST平台常面临分辨率低的问题，限制了空间基因表达的深入理解

### 目的

提高ST地图分辨率，通过建模组织学图像与基因表达之间的相互作用，实现有效的ST增强

### 方法

提出C3-Diff框架，改进传统对比学习范式提取模态不变和内容不变特征，在特征单元超球面上进行噪声信息增强，并采用动态跨模态插值训练策略缓解数据稀缺问题

### 主要发现

在四个公共数据集上测试，C3-Diff性能显著优于竞争方法，并在细胞类型定位、基因表达相关性和单细胞水平基因表达预测等下游任务上表现良好

### 结论

C3-Diff促进了人工智能增强的生物技术在生物医学研究和临床应用中的应用

### 翻译

空间转录组学（ST）的快速发展，即空间基因表达，使得在原始组织中测量基因表达成为可能，使我们能够发现分子机制。然而，当前的ST平台经常面临分辨率低的问题，限制了空间基因表达的深入理解。超分辨率方法通过整合组织学图像和已分析组织斑点的基因表达，有望增强ST地图。然而，建模组织学图像和基因表达之间的相互作用以实现有效的ST增强仍然是一个挑战。本研究提出了一个名为C3-Diff的跨模态跨内容对比扩散框架，用于以组织学图像为指导的ST增强。在C3-Diff中，我们首先分析了传统对比学习范式的不足，然后加以改进，以提取ST地图和组织学图像的模态不变和内容不变特征。此外，为了克服ST地图中测序灵敏度低的问题，我们在特征单元超球面上进行了基于噪声的信息增强。最后，我们提出了一种动态跨模态插值训练策略，以缓解ST数据稀缺问题。我们在四个公共数据集上测试了C3-Diff，其性能显著优于竞争方法。此外，我们在细胞类型定位、基因表达相关性和单细胞水平基因表达预测等下游任务上评估了C3-Diff，促进了人工智能增强的生物技术在生物医学研究和临床应用中的应用。代码可在https://github.com/XiaofeiWang2018/C3-Diff获取。


### 论文摘要

The rapid advancement of spatial transcriptomics (ST), i.e., spatial gene expressions, has made it possible to measure gene expression within original tissue, enabling us to discover molecular mechanisms. However, current ST platforms frequently suffer from low resolution, limiting the in-depth understanding of spatial gene expression. Super-resolution approaches promise to enhance ST maps by integrating histology images with gene expressions of profiled tissue spots. However, it remains a challenge to model the interactions between histology images and gene expressions for effective ST enhancement. This study presents a cross-modal cross-content contrastive diffusion framework, called C3-Diff, for ST enhancement with histology images as guidance. In C3-Diff, we firstly analyze the deficiency of traditional contrastive learning paradigm, which is then refined to extract both modal-invariant and content-invariant features of ST maps and histology images. Further, to overcome the problem of low sequencing sensitivity in ST maps, we perform nosing-based information augmentation on the surface of feature unit hypersphere. Finally, we propose a dynamic cross-modal imputation-based training strategy to mitigate ST data scarcity. We tested C3-Diff by benchmarking its performance on four public datasets, where it achieves significant improvements over competing methods. Moreover, we evaluate C3-Diff on downstream tasks of cell type localization, gene expression correlation and single-cell-level gene expression prediction, promoting AI-enhanced biotechnology for biomedical research and clinical applications. Codes are available at https://github.com/XiaofeiWang2018/C3-Diff.

---

## 77. Point Cloud Segmentation of Integrated Circuits Package Substrates Surface Defects Using Causal Inference: Dataset Construction and Methodology

**论文链接:** [http://arxiv.org/abs/2511.05853v1](http://arxiv.org/abs/2511.05853v1)

**作者:** Bingyang Guo, Qiang Zuo, Ruiyun Yu

**发布时间:** 2025-11-08

### GPT解析

### 总结

本研究构建了用于陶瓷封装基板(CPS)表面缺陷3D分割的高质量点云数据集CPS3D-Seg，并提出了一种基于因果推断的新型3D分割方法CINet，实验表明该方法在性能上显著优于现有算法。

### 背景

3D数据的有效分割对工业应用至关重要，特别是在集成电路(IC)领域检测微小缺陷。陶瓷封装基板(CPS)作为重要电子材料，因其优异的物理化学性质在IC封装中必不可少。然而，CPS的复杂结构和微小缺陷，加上缺乏公开可用的数据集，严重阻碍了CPS表面缺陷检测的发展。

### 目的

构建一个高质量的CPS表面缺陷3D分割点云数据集，并开发一种有效的3D分割方法来提高检测性能。

### 方法

1. 构建了名为CPS3D-Seg的高质量点云数据集，包含20个产品类别下的1300个点云样本，每个样本提供精确的点级标注；2. 基于最先进的点云分割算法进行了全面的基准测试；3. 提出了一种基于因果推断的新型3D分割方法CINet，通过结构化精炼(SR)和质量评估(QA)模块量化点云中的潜在混杂因素。

### 主要发现

1. CPS3D-Seg数据集在点分辨率和精度上优于现有的3D工业数据集；2. CINet在mIoU和准确率方面显著优于现有算法。

### 结论

通过构建高质量数据集和提出创新的分割方法，有效解决了CPS表面缺陷检测的挑战，为工业应用提供了可靠的解决方案。

### 翻译

3D数据的有效分割对广泛的工业应用至关重要，特别是在集成电路(IC)领域检测微小缺陷。陶瓷封装基板(CPS)作为一种重要的电子材料，由于其优异的物理和化学性质，在IC封装中必不可少。然而，CPS的复杂结构和微小缺陷，加上缺乏公开可用的数据集，严重阻碍了CPS表面缺陷检测的发展。在本研究中，我们构建了一个用于CPS表面缺陷3D分割的高质量点云数据集，即CPS3D-Seg，与现有的3D工业数据集相比，它具有最佳的点分辨率和精度。CPS3D-Seg包含20个产品类别下的1300个点云样本，每个样本都提供精确的点级标注。同时，我们基于最先进的点云分割算法进行了全面的基准测试，以验证CPS3D-Seg的有效性。此外，我们提出了一种基于因果推断的新型3D分割方法(CINet)，通过结构化精炼(SR)和质量评估(QA)模块量化点云中的潜在混杂因素。大量实验证明，CINet在mIoU和准确率方面显著优于现有算法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决集成电路封装基板表面缺陷的高精度3D点云分割问题。这个问题很重要，因为封装基板是电子设备的关键组成部分，其表面微小缺陷可能导致设备故障；同时，现有2D方法缺乏深度信息，无法准确检测深度方向的缺陷，而高精度3D数据集的缺乏也限制了相关算法的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有3D工业数据集与陶瓷封装基板之间存在显著差异，后者具有复杂的三维结构和表面电路。作者发现高密度点云在预处理过程中可能导致表示效果差异，这些差异可概括为点云的复杂性。受此启发，作者将因果推断引入点云分割。作者借鉴了现有的点云采集技术、数据集构建方法（如MVTec 3D-AD、Real3D-AD）、多种分割方法（CNN、图神经网络、Transformer、Mamba）以及计算机视觉中的因果推断应用，但将其专门应用于集成电路领域的高精度缺陷检测。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建高质量的3D点云数据集并基于因果推断进行分割。具体流程：1) 使用四个高精度线激光扫描仪构建数据采集平台，从实际生产线收集20种产品样本并进行点级标注；2) 提出CINet方法，包含三个模块：质量评估模块(QA)使用GMM捕获点云特征，结构精炼模块(SR)通过FPS和K-NN处理点云结构，映射注意力检测模块(MAD)整合特征；3) 构建结构因果模型(SCM)处理点云中的潜在混杂因素，通过后门调整方法控制混杂因素的影响。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 构建CPS3D-Seg数据集，包含1300个样本，点云分辨率0.0025毫米，精度0.0003毫米，比现有数据集高一个数量级；2) 提出基于因果推断的CINet分割方法，通过结构因果模型解决点云中的潜在混杂因素；3) 构建全面的基准测试，涵盖最新点云分割算法。相比之前工作，CPS3D-Seg全部来自实际生产线而非合成数据，具有最高分辨率和精度；CINet专门针对陶瓷封装基板复杂数据设计，引入因果推断而非仅关注统计相关性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文构建了目前最高精度的集成电路封装基板表面缺陷3D点云数据集，并提出了一种基于因果推断的创新分割方法，显著提高了微小缺陷检测的准确性。'}


### 论文摘要

The effective segmentation of 3D data is crucial for a wide range of industrial applications, especially for detecting subtle defects in the field of integrated circuits (IC). Ceramic package substrates (CPS), as an important electronic material, are essential in IC packaging owing to their superior physical and chemical properties. However, the complex structure and minor defects of CPS, along with the absence of a publically available dataset, significantly hinder the development of CPS surface defect detection. In this study, we construct a high-quality point cloud dataset for 3D segmentation of surface defects in CPS, i.e., CPS3D-Seg, which has the best point resolution and precision compared to existing 3D industrial datasets. CPS3D-Seg consists of 1300 point cloud samples under 20 product categories, and each sample provides accurate point-level annotations. Meanwhile, we conduct a comprehensive benchmark based on SOTA point cloud segmentation algorithms to validate the effectiveness of CPS3D-Seg. Additionally, we propose a novel 3D segmentation method based on causal inference (CINet), which quantifies potential confounders in point clouds through Structural Refine (SR) and Quality Assessment (QA) Modules. Extensive experiments demonstrate that CINet significantly outperforms existing algorithms in both mIoU and accuracy.

---

