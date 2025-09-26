# 今日论文推荐 - 2025-09-26

共 66 篇论文

---

## 1. GeoRef: Referring Expressions in Geometry via Task Formulation, Synthetic Supervision, and Reinforced MLLM-based Solutions

**论文链接:** [http://arxiv.org/abs/2509.21050v1](http://arxiv.org/abs/2509.21050v1)

**作者:** Bing Liu, Wenqiang Yv, Xuzheng Yang, Shichang Wang, Junzhuo Liu, Peng Wang, Guoqing Wang, Yang Yang, Heng Tao Shen

**发布时间:** 2025-09-25

### GPT解析

### 总结

该研究引入了几何问题中的指代表达式理解(REC)任务，构建了GeoRef基准数据集，探索了监督微调(SFT)和组相对策略优化(GRPO)两种微调方法，并提出了验证和重新生成机制以提高准确性。研究结果表明，即使是先进的多模态大语言模型在几何基础方面也存在困难，但适当的训练可以改善下游几何推理任务的表现。

### 背景

AI驱动的几何问题解决是一个复杂的视觉-语言任务，需要准确的图表解释、数学推理和强大的跨模态基础能力。根据自然语言查询识别和解释几何元素是一个基础但未被充分探索的能力。

### 目的

引入几何问题中的指代表达式理解(REF)任务，评估模型是否能够根据文本提示在图表中定位点、形状和空间关系，并构建一个基准数据集(GeoRef)。

### 方法

从现有几何问题语料库构建GeoRef基准数据集，使用结构化的几何形式语言生成大规模合成训练数据；探索监督微调(SFT)和组相对策略优化(GRPO)两种微调方法；提出验证和重新生成机制来检测不正确预测并使用上下文推理历史重新推断答案。

### 主要发现

GRPO显著优于SFT，能更好地使模型行为与任务特定奖励保持一致；验证和重新生成机制进一步提高了准确性；即使是最先进的多模态大语言模型也难以完成此任务；在GeoRef上训练的模型在下游几何推理任务上显示出可衡量的改进。

### 结论

明确评估和加强几何基础能力对于强大的几何问题解决是必要的；REC作为多模态数学理解的基础具有更广泛的价值。

### 翻译

AI驱动的几何问题解决是一个复杂的视觉-语言任务，需要准确的图表解释、数学推理和强大的跨模态基础能力。这个任务的一个基础但未被充分探索的能力是根据自然语言查询识别和解释几何元素。为此，我们引入了几何问题中的指代表达式理解(REF)任务，评估模型是否能够根据文本提示在图表中定位点、形状和空间关系。我们提出了从现有几何问题语料库构建的GeoRef基准数据集，具有多样化、高质量的注释和查询。由于缺乏此任务的标注数据，我们使用结构化的几何形式语言生成了大规模合成训练数据，使几何概念覆盖广泛并促进模型适应。我们探索了两种微调方法：监督微调(SFT)和组相对策略优化(GRPO)。我们的结果表明，GRPO通过更好地使模型行为与任务特定奖励保持一致，显著优于SFT。此外，我们提出了一种验证和重新生成机制，可以检测不正确的预测并使用上下文推理历史重新推断答案，从而进一步提高准确性。值得注意的是，即使是最先进的多模态大语言模型也难以完成此任务，这凸显了明确评估和加强几何基础能力作为强大几何问题解决先决条件的必要性。此外，在GeoRef上训练的模型在下游几何推理任务上显示出可衡量的改进，突显了REC作为多模态数学理解基础的更广泛价值。


### 论文摘要

AI-driven geometric problem solving is a complex vision-language task that requires accurate diagram interpretation, mathematical reasoning, and robust cross-modal grounding. A foundational yet underexplored capability for this task is the ability to identify and interpret geometric elements based on natural language queries. To address this, we introduce the task of Referring Expression Comprehension (REC) for geometric problems, which evaluates whether models can localize points, shapes, and spatial relations in diagrams in response to textual prompts. We present GeoRef, a benchmark dataset constructed from existing geometric problem corpora, featuring diverse, high-quality annotations and queries. Due to the lack of annotated data for this task, we generate a large-scale synthetic training dataset using a structured geometric formal language, enabling broad coverage of geometric concepts and facilitating model adaptation. We explore two fine-tuning approaches: Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). Our results show that GRPO significantly outperforms SFT by better aligning model behavior with task-specific rewards. Furthermore, we propose a verify-and-regenerate mechanism that detects incorrect predictions and re-infers answers using contextual reasoning history, further boosting accuracy. Notably, even state-of-the-art Multimodal Large Language Models (MLLMs) struggle with this task, underscoring the necessity of explicitly evaluating and strengthening geometric grounding as a prerequisite for robust geometric problem solving. Moreover, models trained on GeoRef demonstrate measurable improvements on downstream geometric reasoning tasks, highlighting the broader value of REC as a foundation for multimodal mathematical understanding.

---

## 2. DENet: Dual-Path Edge Network with Global-Local Attention for Infrared Small Target Detection

**论文链接:** [http://arxiv.org/abs/2509.20701v1](http://arxiv.org/abs/2509.20701v1)

**作者:** Jiayi Zuo, Songwei Pei, Qian Li

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了一种新颖的双路径边缘网络，用于解决红外小目标检测中的挑战，通过解耦边缘增强和语义建模，实现了精确的目标检测和定位。

### 背景

红外小目标检测对于遥感应用如灾害预警和海事监视至关重要。然而，由于缺乏独特的纹理和形态特征，红外小目标很容易融入杂乱和嘈杂的背景中。现有方法通常依赖于固定的梯度算子或简单的注意力机制，在低对比度和高噪声条件下无法准确提取目标边缘。

### 目的

解决红外小目标检测中捕获微小目标的高分辨率空间细节与提取较大目标的鲁棒语义上下文之间的内在冲突，提高检测准确性和鲁棒性。

### 方法

提出双路径边缘网络，包含两条互补处理路径：第一条路径采用双向交互模块，结合局部自注意力和全局自注意力捕获多尺度特征依赖性；第二条路径引入多边缘细化器，使用多级泰勒有限差分算子增强边缘细节，结合注意力驱动的门控机制实现精确定位和特征增强。

### 主要发现

双路径边缘网络能够有效解决特征错位问题，实现不同大小目标的精确定位和特征增强，同时有效抑制噪声。

### 结论

该方法为精确的红外小目标检测和定位提供了有希望的解决方案，在一个统一的框架中结合了结构语义和边缘细化。

### 翻译

红外小目标检测对于遥感应用（如灾害预警和海事监视）至关重要。然而，由于缺乏独特的纹理和形态特征，红外小目标很容易融入杂乱和嘈杂的背景中。为此任务设计深度模型的一个基本挑战在于，捕获微小目标的高分辨率空间细节与提取较大目标的鲁棒语义上下文之间存在内在冲突，这通常会导致特征错位和次优性能。现有方法通常依赖于固定的梯度算子或简单的注意力机制，这些方法在低对比度和高噪声条件下无法准确提取目标边缘。在本文中，我们提出了一种新颖的双路径边缘网络，通过将边缘增强和语义建模解耦为两个互补的处理路径，明确地解决了这一挑战。第一条路径采用双向交互模块，该模块使用局部自注意力和全局自注意力来捕获多尺度的局部和全局特征依赖性。基于Transformer架构的全局注意力机制整合了长程语义关系和上下文信息，确保了鲁棒的场景理解。第二条路径引入了多边缘细化器，该细化器使用多级泰勒有限差分算子在多个尺度上增强细粒度的边缘细节。这种数学方法，结合注意力驱动的门控机制，能够实现不同大小目标的精确定位和特征增强，同时有效抑制噪声。我们的方法为精确的红外小目标检测和定位提供了有希望的解决方案，在一个统一的框架中结合了结构语义和边缘细化。


### 论文摘要

Infrared small target detection is crucial for remote sensing applications like disaster warning and maritime surveillance. However, due to the lack of distinctive texture and morphological features, infrared small targets are highly susceptible to blending into cluttered and noisy backgrounds. A fundamental challenge in designing deep models for this task lies in the inherent conflict between capturing high-resolution spatial details for minute targets and extracting robust semantic context for larger targets, often leading to feature misalignment and suboptimal performance. Existing methods often rely on fixed gradient operators or simplistic attention mechanisms, which are inadequate for accurately extracting target edges under low contrast and high noise. In this paper, we propose a novel Dual-Path Edge Network that explicitly addresses this challenge by decoupling edge enhancement and semantic modeling into two complementary processing paths. The first path employs a Bidirectional Interaction Module, which uses both Local Self-Attention and Global Self-Attention to capture multi-scale local and global feature dependencies. The global attention mechanism, based on a Transformer architecture, integrates long-range semantic relationships and contextual information, ensuring robust scene understanding. The second path introduces the Multi-Edge Refiner, which enhances fine-grained edge details using cascaded Taylor finite difference operators at multiple scales. This mathematical approach, along with an attention-driven gating mechanism, enables precise edge localization and feature enhancement for targets of varying sizes, while effectively suppressing noise. Our method provides a promising solution for precise infrared small target detection and localization, combining structural semantics and edge refinement in a unified framework.

---

## 3. OmniScene: Attention-Augmented Multimodal 4D Scene Understanding for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2509.19973v2](http://arxiv.org/abs/2509.19973v2)

**作者:** Pei Liu, Hongliang Lu, Haichao Liu, Haipeng Liu, Xin Liu, Ruoyu Yao, Shengbo Eben Li, Jun Ma

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种名为OmniScene的类人框架，通过视觉语言模型和分层融合策略，使自动驾驶系统能够实现更接近人类的三维场景理解能力。

### 背景

人类视觉能够将二维观察转化为以自我为中心的三维场景理解，支持复杂场景转换和适应性行为。然而，当前自动驾驶系统主要依赖基于深度的3D重建，缺乏真正的场景理解能力。

### 目的

开发一种类人框架OmniScene，弥补当前自动驾驶系统在场景理解方面的不足，实现更接近人类视觉系统的感知和理解能力。

### 方法

1) 引入OmniScene视觉语言模型(OmniVLM)，整合多视图和时间感知实现4D场景理解；2) 采用教师-学生架构和知识蒸馏，将文本表示嵌入3D实例特征进行语义监督；3) 提出分层融合策略(HFS)，自适应校准几何和语义特征的相对重要性，解决模态贡献不平衡问题。

### 主要发现

OmniScene在nuScenes数据集上与十多种最先进模型进行对比测试，在感知、预测、规划和视觉问答等任务上均取得优越结果，建立了新的性能基准。

### 结论

OmniScene框架通过模拟人类视觉系统的场景理解能力，结合有效的多模态融合策略，显著提升了自动驾驶系统的场景理解和适应能力，为未来自动驾驶技术的发展提供了新方向。

### 翻译

人类视觉能够将二维观察转化为以自我为中心的三维场景理解，这支持了复杂场景的转换和适应性行为的能力。然而，当前自动驾驶系统仍缺乏这种能力，主流方法主要依赖基于深度的3D重建而非真正的场景理解。为解决这一局限，我们提出了一种名为OmniScene的新型类人框架。首先，我们引入了OmniScene视觉语言模型(OmniVLM)，这是一个整合多视图和时间感知以实现整体4D场景理解的视觉语言框架。然后，利用教师-学生OmniVLM架构和知识蒸馏，我们将文本表示嵌入3D实例特征中进行语义监督，丰富特征学习并明确捕获类人的注意力语义。这些特征表示进一步与人类驾驶行为对齐，形成更类人的感知-理解-行动架构。此外，我们提出了一种分层融合策略(HFS)来解决多模态融合过程中模态贡献不平衡的问题。我们的方法在多个抽象层次上自适应校准几何和语义特征的相对重要性，实现了来自视觉和文本模态互补线索的协同使用。这种可学习的动态融合能够更细致有效地利用异构信息。我们在nuScenes数据集上全面评估了OmniScene，在各种任务上与十多种最先进模型进行基准测试。我们的方法 consistently取得了优越的结果，在感知、预测、规划和视觉问答方面建立了新的基准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决当前自动驾驶系统缺乏人类视觉能力的问题，即无法像人类一样将二维观察转化为以自我为中心的三维场景理解。这个问题很重要，因为人类视觉能够处理复杂场景并做出适应性决策，而现有系统主要依赖基于深度的3D重建，缺乏真正的场景理解能力，导致在复杂和模糊场景下表现有限，无法有效整合交通动态和导航约束等关键上下文信息。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析人类视觉系统的能力来设计方法，借鉴了视觉-语言模型(VLM)、注意力机制、端到端自动驾驶和多模态融合等现有工作。作者设计了OmniScene框架，包括OmniVLM视觉-语言模型和分层融合策略(HFS)，并通过教师-学生架构提高计算效率。他们结合了现有VLM技术(如Qwen2.5VL)和知识蒸馏方法，同时参考了UniAD和SparseDrive等端到端自动驾驶系统，但针对自动驾驶场景进行了专门优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是模拟人类视觉系统，将二维观察转化为三维场景理解，并通过整合视觉、文本和3D几何特征实现全面的场景理解。整体流程包括：1)接收多视角图像流、操作命令和用户提示；2)学生OmniVLM生成场景文本注释；3)提取视觉特征和文本特征；4)通过分层融合策略(HFS)整合多模态信息，包括3D实例初始化、4D时空聚合、视觉变形聚合、文本条件聚合和深度细化；5)进行多模态预测和规划；6)输出感知结果、预测轨迹和规划决策。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)OmniScene框架实现人类般的场景理解；2)OmniVLM架构整合多视角和时间感知；3)教师-学生知识蒸馏将文本表示嵌入3D实例特征；4)分层融合策略(HFS)解决模态不平衡问题；5)全局多模态对齐策略。相比之前工作，OmniScene专注于真正的场景理解而非简单的3D重建，实现了视觉和文本模态的深度整合而非独立处理，具有明确的人类注意力建模能力，通过教师-学生架构提高效率，并在更全面的评估指标上表现优异。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OmniScene通过整合多模态信息、模拟人类注意力机制和采用分层融合策略，实现了更接近人类视觉能力的自动驾驶场景理解，在nuScenes数据集上取得了多项任务的最新性能。'}


### 论文摘要

Human vision is capable of transforming two-dimensional observations into an egocentric three-dimensional scene understanding, which underpins the ability to translate complex scenes and exhibit adaptive behaviors. This capability, however, remains lacking in current autonomous driving systems, where mainstream approaches primarily rely on depth-based 3D reconstruction rather than true scene understanding. To address this limitation, we propose a novel human-like framework called OmniScene. First, we introduce the OmniScene Vision-Language Model (OmniVLM), a vision-language framework that integrates multi-view and temporal perception for holistic 4D scene understanding. Then, harnessing a teacher-student OmniVLM architecture and knowledge distillation, we embed textual representations into 3D instance features for semantic supervision, enriching feature learning, and explicitly capturing human-like attentional semantics. These feature representations are further aligned with human driving behaviors, forming a more human-like perception-understanding-action architecture. In addition, we propose a Hierarchical Fusion Strategy (HFS) to address imbalances in modality contributions during multimodal integration. Our approach adaptively calibrates the relative significance of geometric and semantic features at multiple abstraction levels, enabling the synergistic use of complementary cues from visual and textual modalities. This learnable dynamic fusion enables a more nuanced and effective exploitation of heterogeneous information. We evaluate OmniScene comprehensively on the nuScenes dataset, benchmarking it against over ten state-of-the-art models across various tasks. Our approach consistently achieves superior results, establishing new benchmarks in perception, prediction, planning, and visual question answering.

---

## 4. SGAligner++: Cross-Modal Language-Aided 3D Scene Graph Alignment

**论文链接:** [http://arxiv.org/abs/2509.20401v1](http://arxiv.org/abs/2509.20401v1)

**作者:** Binod Singh, Sayan Deb Sarkar, Iro Armeni

**发布时间:** 2025-09-23

### GPT解析

### 总结

该论文提出了SGAligner++，一种跨模态、语言辅助的3D场景图对齐框架，能有效处理不完整或有噪声的输入，在机器人导航和具身感知应用中实现准确的场景图对齐。

### 背景

3D场景图对齐是机器人导航和具身感知应用中的关键初始步骤，但当前方法通常依赖单模态点云数据，在处理不完整或有噪声的输入时表现不佳。

### 目的

开发一种能够处理异构模态间部分重叠场景观测对齐挑战的方法，特别是在低重叠条件和传感器噪声环境下实现准确对齐。

### 方法

SGAligner++通过学习统一的联合嵌入空间来处理跨异构模态的部分重叠场景观测对齐问题，采用轻量级单模态编码器和基于注意力的融合技术，增强视觉定位、3D重建和导航等任务的场景理解能力。

### 主要发现

在真实世界数据集上的大量评估表明，SGAligner++在有噪声的真实世界重建任务上比最先进的方法性能提高高达40%，同时实现了跨模态泛化能力。

### 结论

SGAligner++是一种有效的跨模态、语言辅助框架，能够处理不完整或有噪声的输入，在3D场景图对齐任务上取得了显著性能提升，同时确保了可扩展性和最小计算开销。

### 翻译

对齐3D场景图是机器人导航和具身感知中多个应用的关键初始步骤。当前3D场景图对齐方法通常依赖单模态点云数据，难以处理不完整或有噪声的输入。我们提出了SGAligner++，一种用于3D场景图对齐的跨模态、语言辅助框架。我们的方法通过学习统一的联合嵌入空间，解决了跨异构模态对齐部分重叠场景观测的挑战，即使在低重叠条件和传感器噪声下也能实现准确对齐。通过采用轻量级单模态编码器和基于注意力的融合，SGAligner++增强了视觉定位、3D重建和导航等任务的场景理解能力，同时确保了可扩展性和最小计算开销。在真实世界数据集上的大量评估表明，SGAligner++在有噪声的真实世界重建任务上比最先进的方法性能提高高达40%，同时实现了跨模态泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D场景图在跨模态条件下的对齐问题，特别是当输入数据不完整或有噪声时的对齐挑战。这个问题在机器人导航和具身感知中至关重要，因为机器人需要在动态环境中将不同传感器获取的信息（如点云、CAD模型、文本描述等）整合成一致的空间理解，以便进行准确的定位、导航和交互。现有方法主要依赖单一模态数据，难以处理现实世界中常见的部分重叠场景和传感器噪声问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法依赖单一模态（如点云）和固定标签词汇表的局限性，难以处理不完整重建和跨模态场景。他们借鉴了SGAligner的单模态编码器设计、MCLEA的多模态表示学习以及EVA的跨模态对齐思想，但进行了创新改进。作者设计了一个融合结构、几何和语言信息的统一框架，使用轻量级单模态编码器和基于注意力的机制，使模型能够从语言中推理空间关系，并处理不同环境中的缺失数据。他们还采用图注意力网络捕获空间关系，并使用对比学习损失函数优化嵌入空间。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是学习一个统一的联合嵌入空间，使不同模态的数据能够在同一空间中表示和比较，从而实现跨模态对齐。整体流程包括：1) 使用单模态编码器分别处理点云、CAD网格、结构图、文本描述和空间参考；2) 将单模态特征投影到共享潜在空间并使用注意力权重融合；3) 通过对比学习损失函数优化嵌入空间；4) 在联合嵌入空间中使用余弦相似度匹配节点；5) 合并匹配节点的属性和融合多模态数据，构建统一的3D场景图。这种方法即使在部分重叠和噪声条件下也能保持鲁棒性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次实现跨模态3D场景图对齐，支持点云、CAD网格、文本描述等多种模态；2) 不仅对齐节点，还构建统一的场景图，合并重叠对象；3) 使用轻量级设计和基于注意力的融合确保计算效率；4) 在噪声和低重叠条件下保持准确对齐；5) 模块化设计支持新模态集成；6) 利用语言信息增强场景理解。相比之前工作，SGAligner++不再局限于单一模态或固定语义标签，而是支持开放词汇语义和跨模态泛化，同时保持低计算成本和高鲁棒性，在真实世界数据上表现显著优于现有方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SGAligner++通过融合几何、语义和空间关系信息，实现了一种轻量级、鲁棒的跨模态3D场景图对齐方法，即使在部分重叠和噪声条件下也能准确对齐场景并生成统一的场景表示。'}


### 论文摘要

Aligning 3D scene graphs is a crucial initial step for several applications in robot navigation and embodied perception. Current methods in 3D scene graph alignment often rely on single-modality point cloud data and struggle with incomplete or noisy input. We introduce SGAligner++, a cross-modal, language-aided framework for 3D scene graph alignment. Our method addresses the challenge of aligning partially overlapping scene observations across heterogeneous modalities by learning a unified joint embedding space, enabling accurate alignment even under low-overlap conditions and sensor noise. By employing lightweight unimodal encoders and attention-based fusion, SGAligner++ enhances scene understanding for tasks such as visual localization, 3D reconstruction, and navigation, while ensuring scalability and minimal computational overhead. Extensive evaluations on real-world datasets demonstrate that SGAligner++ outperforms state-of-the-art methods by up to 40% on noisy real-world reconstructions, while enabling cross-modal generalization.

---

## 5. SlideMamba: Entropy-Based Adaptive Fusion of GNN and Mamba for Enhanced Representation Learning in Digital Pathology

**论文链接:** [http://arxiv.org/abs/2509.21239v1](http://arxiv.org/abs/2509.21239v1)

**作者:** Shakib Khan, Fariba Dambandkhameneh, Nazim Shaikh, Yao Nie, Raghavan Venugopal, Xiao Li

**发布时间:** 2025-09-25

### GPT解析

### 总结

该研究提出了一种名为SlideMamba的深度学习框架，结合Mamba架构与图神经网络(GNNs)用于全切片图像(WSI)分析，通过自适应融合策略有效捕捉局部空间关系和长程上下文依赖关系。

### 背景

计算病理学发展越来越依赖从全切片图像中提取有意义的表示来支持临床和生物学任务，但现有方法在捕捉多尺度信息方面存在局限。

### 目的

开发一个可推广的深度学习框架，整合Mamba架构与图神经网络，增强全切片图像分析能力，同时捕捉局部空间关系和长程上下文依赖关系。

### 方法

设计结合Mamba模块(擅长捕捉长程全局依赖)与GNNs(强调细粒度短程空间交互)的框架，引入基于熵的置信度加权机制的自适应融合策略，动态平衡两个分支的贡献。

### 主要发现

在预测基因融合和突变状态的任务中，SlideMamba达到0.751±0.05的PRAUC，显著优于MIL、Trans-MIL、Mamba-only、GNN-only和GAT-Mamba等对比方法，在ROC AUC、敏感性和特异性方面也表现优异。

### 结论

集成架构通过基于熵的自适应融合策略得到增强，在计算病理学中的空间分辨预测建模任务具有广阔的应用前景。

### 翻译

计算病理学的进步越来越依赖于从全切片图像中提取有意义的表示来支持各种临床和生物学任务。在本研究中，我们提出了一个可推广的深度学习框架，将Mamba架构与图神经网络(GNNs)相结合，以增强WSI分析。我们的方法旨在捕捉局部空间关系和长程上下文依赖关系，为数字病理分析提供灵活的架构。Mamba模块擅长捕捉长程全局依赖关系，而GNNs强调细粒度的短程空间交互。为了有效结合这些互补信号，我们引入了一种自适应融合策略，使用基于熵的置信度加权机制。这种方法通过根据上下文重要性为不同下游任务分配更高权重，动态平衡两个分支的贡献，将更高权重分配给具有更置信(低熵)预测的分支。我们在代表性任务上证明了我们方法的有效性：从WSIs预测基因融合和突变状态。我们的框架SlideMamba达到了0.751±0.05的精确召回曲线下面积(PRAUC)，优于MIL (0.491±0.042)、Trans-MIL (0.39±0.017)、仅使用Mamba (0.664±0.063)、仅使用GNN (0.748±0.091)以及先前类似工作GAT-Mamba (0.703±0.075)。SlideMamba在ROC AUC (0.738±0.055)、敏感性(0.662±0.083)和特异性(0.725±0.094)方面也取得了具有竞争力的结果。这些结果突显了集成架构的优势，通过提出的基于熵的自适应融合策略得到增强，并表明在计算病理学中空间分辨预测建模任务具有应用潜力。


### 论文摘要

Advances in computational pathology increasingly rely on extracting meaningful representations from Whole Slide Images (WSIs) to support various clinical and biological tasks. In this study, we propose a generalizable deep learning framework that integrates the Mamba architecture with Graph Neural Networks (GNNs) for enhanced WSI analysis. Our method is designed to capture both local spatial relationships and long-range contextual dependencies, offering a flexible architecture for digital pathology analysis. Mamba modules excels in capturing long-range global dependencies, while GNNs emphasize fine-grained short-range spatial interactions. To effectively combine these complementary signals, we introduce an adaptive fusion strategy that uses an entropy-based confidence weighting mechanism. This approach dynamically balances contributions from both branches by assigning higher weight to the branch with more confident (lower-entropy) predictions, depending on the contextual importance of local versus global information for different downstream tasks. We demonstrate the utility of our approach on a representative task: predicting gene fusion and mutation status from WSIs. Our framework, SlideMamba, achieves an area under the precision recall curve (PRAUC) of 0.751 \pm 0.05, outperforming MIL (0.491 \pm 0.042), Trans-MIL (0.39 \pm 0.017), Mamba-only (0.664 \pm 0.063), GNN-only (0.748 \pm 0.091), and a prior similar work GAT-Mamba (0.703 \pm 0.075). SlideMamba also achieves competitive results across ROC AUC (0.738 \pm 0.055), sensitivity (0.662 \pm 0.083), and specificity (0.725 \pm 0.094). These results highlight the strength of the integrated architecture, enhanced by the proposed entropy-based adaptive fusion strategy, and suggest promising potential for application of spatially-resolved predictive modeling tasks in computational pathology.

---

## 6. Embodied Representation Alignment with Mirror Neurons

**论文链接:** [http://arxiv.org/abs/2509.21136v1](http://arxiv.org/abs/2509.21136v1)

**作者:** Wentao Zhu, Zhining Zhang, Yuwei Ren, Yin Huang, Hao Xu, Yizhou Wang

**发布时间:** 2025-09-25

**备注:** ICCV 2025

### GPT解析

### 总结

本研究通过镜像神经元的机制，提出了一种统一的表示学习方法来建模动作理解和执行，实现了两种能力之间的协同效应，提高了表示质量和泛化能力。

### 背景

镜像神经元是一类在个体观察和执行同一动作时都会激活的神经元，揭示了动作理解和具身执行之间的基本联系。然而，现有机器学习方法大多忽视这种联系，将这两种能力视为独立任务。

### 目的

通过表示学习的视角为动作理解和执行提供统一的建模方法，探索两种能力之间的协同效应。

### 方法

首先观察到动作理解和执行的中间表示会自发对齐；受镜像神经元启发，引入方法明确对齐观察和执行动作的表示；使用两个线性层将表示映射到共享的潜在空间，通过对比学习强制相应表示的对齐，最大化它们之间的互信息。

### 主要发现

实验表明，这种简单的方法促进了动作理解和执行两个任务之间的相互协同，有效提高了表示质量和泛化能力。

### 结论

通过模拟镜像神经元机制，证明了动作理解和执行可以通过统一的表示学习方法进行有效建模，且两种能力之间的协同效应可以提升整体性能。

### 翻译

镜像神经元是一类在个体观察动作和执行同一动作时都会激活的神经元。这种机制揭示了动作理解和具身执行之间的基本互动，表明这两种能力本质上是相互连接的。尽管如此，现有的机器学习方法大多忽视了这种互动，将这些能力视为独立任务。在本研究中，我们通过表示学习的视角为它们建模提供了一个统一的视角。我们首先观察到它们的中间表示会自发对齐。受镜像神经元启发，我们进一步引入了一种方法，明确地对齐观察和执行动作的表示。具体来说，我们使用两个线性层将表示映射到共享的潜在空间，其中对比学习强制相应表示的对齐，有效最大化它们的互信息。实验证明，这种简单的方法促进了两个任务之间的相互协同，有效提高了表示质量和泛化能力。


### 论文摘要

Mirror neurons are a class of neurons that activate both when an individual observes an action and when they perform the same action. This mechanism reveals a fundamental interplay between action understanding and embodied execution, suggesting that these two abilities are inherently connected. Nonetheless, existing machine learning methods largely overlook this interplay, treating these abilities as separate tasks. In this study, we provide a unified perspective in modeling them through the lens of representation learning. We first observe that their intermediate representations spontaneously align. Inspired by mirror neurons, we further introduce an approach that explicitly aligns the representations of observed and executed actions. Specifically, we employ two linear layers to map the representations to a shared latent space, where contrastive learning enforces the alignment of corresponding representations, effectively maximizing their mutual information. Experiments demonstrate that this simple approach fosters mutual synergy between the two tasks, effectively improving representation quality and generalization.

---

## 7. UniTransfer: Video Concept Transfer via Progressive Spatial and Timestep Decomposition

**论文链接:** [http://arxiv.org/abs/2509.21086v1](http://arxiv.org/abs/2509.21086v1)

**作者:** Guojun Lei, Rong Zhang, Chi Wang, Tianhang Liu, Hong Li, Zhiyuan Ma, Weiwei Xu

**发布时间:** 2025-09-25

**备注:** NeuriIPS 2025

### GPT解析

### 总结

本研究提出了UniTransfer架构，通过空间和时间步分解的渐进范式实现精确可控的视频概念迁移。该方法将视频解耦为前景主体、背景和运动流三个组件，采用双流到单流的DiT架构进行细粒度控制，并引入提示链机制和自监督预训练策略。同时创建了OpenAnimal数据集，实验证明该方法在视觉保真度和可编辑性方面优于现有方法。

### 背景

视频概念迁移是一个具有挑战性的任务，需要精确控制和高质量生成。现有方法在处理视频的空间和时间维度时缺乏足够的灵活性和控制能力。

### 目的

开发一种能够实现精确和可控视频概念迁移的新型架构，通过分解视频的不同组件并提供细粒度控制，提高生成视频的质量和可编辑性。

### 方法

1. 提出UniTransfer架构，引入空间和时间步分解的渐进范式；2. 将视频解耦为前景主体、背景和运动流三个关键组件；3. 采用双流到单流的DiT架构支持对不同组件的细粒度控制；4. 提出基于随机掩码的自监督预训练策略增强分解表示学习；5. 引入提示链机制将去噪过程分解为三个不同粒度的阶段；6. 利用大型语言模型进行特定阶段的指导；7. 创建OpenAnimal视频数据集用于研究和基准测试。

### 主要发现

1. 空间和时间步分解能够有效提高视频概念迁移的精确性和可控性；2. 双流到单流的DiT架构能够实现对视频不同组件的细粒度控制；3. 提示链机制和大型语言模型的指导能够改善生成过程；4. 所提出的方法在视觉保真度和可编辑性方面优于现有基线方法。

### 结论

UniTransfer架构通过创新的空间和时间分解方法，实现了高质量和可控的视频概念迁移。该方法不仅提高了生成视频的质量，还提供了更好的控制能力，为视频概念迁移领域的研究开辟了新方向。OpenAnimal数据集的创建也为该领域的研究提供了宝贵的资源。

### 翻译

我们提出了一种新型架构UniTransfer，它在渐进范式中引入了空间和扩散时间步分解，实现了精确可控的视频概念迁移。具体来说，在空间分解方面，我们将视频解耦为三个关键组件：前景主体、背景和运动流。基于这种分解公式，我们进一步引入了双流到单流的基于DiT的架构，以支持对视频中不同组件的细粒度控制。我们还引入了一种基于随机掩码的自监督预训练策略，以增强从大规模未标记视频数据中学习分解表示。受思维链推理范式的启发，我们重新审视了去噪扩散过程，并提出了提示链机制来实现时间步分解。我们将去噪过程分解为三个不同粒度的阶段，并利用大型语言模型进行特定阶段的指导，以逐步引导生成。我们还整理了一个以动物为中心的视频数据集，称为OpenAnimal，以促进和基准化视频概念转移研究。大量实验表明，我们的方法在不同参考图像和场景下实现了高质量和可控的视频概念转移，在视觉保真度和可编辑性方面都超越了现有基线。


### 论文摘要

We propose a novel architecture UniTransfer, which introduces both spatial and diffusion timestep decomposition in a progressive paradigm, achieving precise and controllable video concept transfer. Specifically, in terms of spatial decomposition, we decouple videos into three key components: the foreground subject, the background, and the motion flow. Building upon this decomposed formulation, we further introduce a dual-to-single-stream DiT-based architecture for supporting fine-grained control over different components in the videos. We also introduce a self-supervised pretraining strategy based on random masking to enhance the decomposed representation learning from large-scale unlabeled video data. Inspired by the Chain-of-Thought reasoning paradigm, we further revisit the denoising diffusion process and propose a Chain-of-Prompt (CoP) mechanism to achieve the timestep decomposition. We decompose the denoising process into three stages of different granularity and leverage large language models (LLMs) for stage-specific instructions to guide the generation progressively. We also curate an animal-centric video dataset called OpenAnimal to facilitate the advancement and benchmarking of research in video concept transfer. Extensive experiments demonstrate that our method achieves high-quality and controllable video concept transfer across diverse reference images and scenes, surpassing existing baselines in both visual fidelity and editability. Web Page: https://yu-shaonian.github.io/UniTransfer-Web/

---

## 8. Alignment Unlocks Complementarity: A Framework for Multiview Circuit Representation Learning

**论文链接:** [http://arxiv.org/abs/2509.20968v1](http://arxiv.org/abs/2509.20968v1)

**作者:** Zhengyuan Shi, Jingxin Wang, Wentao Jiang, Chengyu Ma, Ziyang Zheng, Zhufei Chu, Weikang Qian, Qiang Xu

**发布时间:** 2025-09-25

### GPT解析

### 总结

论文提出了一种名为MixGate的框架，通过功能对齐解决了多视图学习中视图间结构异质性问题，使掩模建模技术能够有效工作并提高性能。

### 背景

多视图学习在布尔电路上具有巨大潜力，不同的基于图的表示（如AIG与XMG）提供互补的结构和语义信息，但视图之间的巨大结构异质性是有效融合的关键障碍。

### 目的

解决视图间结构异质性导致的多视图学习融合困难问题，特别是对于自监督技术如掩模建模。

### 方法

提出MixGate框架，基于有原则的训练课程，首先通过等价对齐损失教导模型共享的、感知功能的表示空间，然后引入多视图掩模建模目标。

### 主要发现

功能对齐是解锁多视图自监督能力的关键先决条件；对齐优先策略将掩模建模从无效技术转变为强大的性能驱动因素。

### 结论

对齐优先策略有效解决了视图间结构异质性问题，使掩模建模能够有效工作并提高性能。

### 翻译

布尔电路上的多视图学习具有巨大潜力，因为不同的基于图的表示提供了互补的结构和语义信息。然而，视图之间的巨大结构异质性，如与-反相图（AIG）与异或-多数图（XMG）之间的异质性，是有效融合的关键障碍，特别是对于掩模建模等自监督技术。简单地应用这些方法会失败，因为跨视图上下文被视为噪声。我们的关键见解是，功能对齐是解锁多视图自监督能力的关键先决条件。我们引入了MixGate，这是一个基于有原则的训练课程的框架，首先通过等价对齐损失教导模型一个共享的、感知功能的表示空间。然后我们才引入多视图掩模建模目标，现在可以利用对齐的视图作为丰富、互补的信号。包括关键消融研究在内的广泛实验证明，我们的对齐优先策略将掩模建模从无效技术转变为强大的性能驱动因素。


### 论文摘要

Multiview learning on Boolean circuits holds immense promise, as different graph-based representations offer complementary structural and semantic information. However, the vast structural heterogeneity between views, such as an And-Inverter Graph (AIG) versus an XOR-Majority Graph (XMG), poses a critical barrier to effective fusion, especially for self-supervised techniques like masked modeling. Naively applying such methods fails, as the cross-view context is perceived as noise. Our key insight is that functional alignment is a necessary precondition to unlock the power of multiview self-supervision. We introduce MixGate, a framework built on a principled training curriculum that first teaches the model a shared, function-aware representation space via an Equivalence Alignment Loss. Only then do we introduce a multiview masked modeling objective, which can now leverage the aligned views as a rich, complementary signal. Extensive experiments, including a crucial ablation study, demonstrate that our alignment-first strategy transforms masked modeling from an ineffective technique into a powerful performance driver.

---

## 9. Flow Matching in the Low-Noise Regime: Pathologies and a Contrastive Remedy

**论文链接:** [http://arxiv.org/abs/2509.20952v1](http://arxiv.org/abs/2509.20952v1)

**作者:** Weili Zeng, Yichao Yan

**发布时间:** 2025-09-25

### GPT解析

### 总结

Flow matching作为一种强大的生成模型框架，在低噪声情况下存在根本性不稳定问题。作者首次分析了这一现象（称为低噪声病理现象），并提出了Local Contrastive Flow (LCF)混合训练协议来解决此问题，提高了收敛速度并稳定了表示质量。

### 背景

Flow matching最近已成为扩散模型的一个强大替代方案，为生成建模和表示学习提供了连续时间公式。

### 目的

解决Flow matching在低噪声regime下的不稳定性问题，提高其收敛速度和表示质量，以充分发挥Flow matching在生成和表示学习方面的潜力。

### 方法

作者提出了Local Contrastive Flow (LCF)，这是一种混合训练协议，在小噪声水平下用对比特征对齐替代直接速度回归，同时在中等和高噪声水平下保留标准Flow matching。

### 主要发现

1. 在低噪声情况下，输入的微小扰动会导致速度目标的大变化，导致学习问题的条件数发散。2. 这种不良条件不仅减缓了优化过程，还迫使编码器将其有限的Jacobian容量重新分配到噪声方向，从而降低了语义表示的质量。3. 作者首次对这种现象（称为低噪声病理现象）进行了理论分析，建立了它与Flow matching目标结构的内在联系。

### 结论

解决低噪声病理现象对于充分发挥Flow matching在生成和表示学习方面的潜力至关重要。提出的LCF方法不仅提高了收敛速度，还稳定了表示质量。

### 翻译

Flow matching最近已成为扩散模型的一个强大替代方案，为生成建模和表示学习提供了连续时间公式。然而，我们表明该框架在低噪声regime下存在根本不稳定性。当噪声水平接近零时，输入的任意小扰动会导致速度目标的大变化，导致学习问题的条件数发散。这种不良条件不仅减缓了优化过程，还迫使编码器将其有限的Jacobian容量重新分配到噪声方向，从而降低了语义表示的质量。我们首次对这种现象进行了理论分析，我们称之为低噪声病理现象，建立了它与Flow matching目标结构的内在联系。基于这些见解，我们提出了Local Contrastive Flow (LCF)，这是一种混合训练协议，在小噪声水平下用对比特征对齐替代直接速度回归，同时在中等和高噪声水平下保留标准Flow matching。从经验上看，LCF不仅提高了收敛速度，还稳定了表示质量。我们的发现强调了解决低噪声病理现象对于充分发挥Flow matching在生成和表示学习方面潜力的重要性。


### 论文摘要

Flow matching has recently emerged as a powerful alternative to diffusion models, providing a continuous-time formulation for generative modeling and representation learning. Yet, we show that this framework suffers from a fundamental instability in the low-noise regime. As noise levels approach zero, arbitrarily small perturbations in the input can induce large variations in the velocity target, causing the condition number of the learning problem to diverge. This ill-conditioning not only slows optimization but also forces the encoder to reallocate its limited Jacobian capacity toward noise directions, thereby degrading semantic representations. We provide the first theoretical analysis of this phenomenon, which we term the low-noise pathology, establishing its intrinsic link to the structure of the flow matching objective. Building on these insights, we propose Local Contrastive Flow (LCF), a hybrid training protocol that replaces direct velocity regression with contrastive feature alignment at small noise levels, while retaining standard flow matching at moderate and high noise. Empirically, LCF not only improves convergence speed but also stabilizes representation quality. Our findings highlight the critical importance of addressing low-noise pathologies to unlock the full potential of flow matching for both generation and representation learning.

---

## 10. Latent Twins

**论文链接:** [http://arxiv.org/abs/2509.20615v1](http://arxiv.org/abs/2509.20615v1)

**作者:** Matthias Chung, Deepanshu Verma, Max Collins, Amit N. Subrahmanya, Varuni Katti Sastry, Vishwas Rao

**发布时间:** 2025-09-24

**备注:** 38 pages, 22 figures, 1 table

### GPT解析

### 总结

本文提出了Latent Twins，一个统一数学框架，为基础方程在潜在空间中创建隐藏的替代模型，将经典建模、反演、模型简化和算子近似统一为单一原则的特殊情况。

### 背景

过去十年科学机器学习转变了分析和预测复杂系统的数学和计算框架，从反问题到数值PDEs、动力系统和模型简化，但这些进展往往并行发展，表示学习和算法求解方法作为独立管道演进。

### 目的

创建一个统一数学框架，为基础方程在潜在空间中创建隐藏的替代模型，就像数字双胞胎镜像物理系统一样，Latent Twins镜像数学系统。

### 方法

建立Latent Twins对ODEs和PDEs的基本近似特性，并在三个设置中展示：规范ODEs捕获多样化动力学状态；使用浅水方程的PDE基准测试；真实数据地势再分析数据集的重建和预测。

### 主要发现

Latent Twins为解算子提供紧凑、可解释的替代模型，可在单次评估中跨越任意时间间隔，同时保持与科学管道的兼容性。

### 结论

该框架提供可扩展、有理论基础的替代模型，跨越学科连接数据驱动的表示学习和经典科学建模。

### 翻译

在过去的十年中，科学机器学习已经转变了分析和预测复杂系统的数学和计算框架的发展。从反问题到数值PDEs、动力系统和模型简化，这些进展已经推动了模拟能力的边界。然而，它们常常并行发展，表示学习和算法求解方法在很大程度上作为独立的管道演进。通过Latent Twins，我们提出了一个统一数学框架，为基础方程在潜在空间中创建隐藏的替代模型。而数字双胞胎在数字世界中镜像物理系统，Latent Twins在由算子控制的潜在空间中镜像数学系统。通过这种视角，经典建模、反演、模型简化和算子近似都作为单一原则的特殊情况出现。我们建立了Latent Twins对ODEs和PDEs的基本近似特性，并在三个代表性设置中展示了该框架：(i) 规范ODEs，捕获多样化的动力学状态；(ii) 使用浅水方程的PDE基准测试，将Latent Twin模拟与DeepONet对比，并将预测与4D-Var基线对比；(iii) 具有挑战性的真实数据地势再分析数据集，从稀疏、嘈杂的观测中重建和预测。Latent Twins为解算子提供了紧凑、可解释的替代模型，可以在单次评估中跨越任意时间间隔，同时保持与科学管道的兼容性，如数据同化、控制和不确定性量化。展望未来，该框架提供了可扩展的、有理论基础的替代模型，跨越学科连接数据驱动的表示学习和经典科学建模。


### 论文摘要

Over the past decade, scientific machine learning has transformed the development of mathematical and computational frameworks for analyzing, modeling, and predicting complex systems. From inverse problems to numerical PDEs, dynamical systems, and model reduction, these advances have pushed the boundaries of what can be simulated. Yet they have often progressed in parallel, with representation learning and algorithmic solution methods evolving largely as separate pipelines. With \emph{Latent Twins}, we propose a unifying mathematical framework that creates a hidden surrogate in latent space for the underlying equations. Whereas digital twins mirror physical systems in the digital world, Latent Twins mirror mathematical systems in a learned latent space governed by operators. Through this lens, classical modeling, inversion, model reduction, and operator approximation all emerge as special cases of a single principle. We establish the fundamental approximation properties of Latent Twins for both ODEs and PDEs and demonstrate the framework across three representative settings: (i) canonical ODEs, capturing diverse dynamical regimes; (ii) a PDE benchmark using the shallow-water equations, contrasting Latent Twin simulations with DeepONet and forecasts with a 4D-Var baseline; and (iii) a challenging real-data geopotential reanalysis dataset, reconstructing and forecasting from sparse, noisy observations. Latent Twins provide a compact, interpretable surrogate for solution operators that evaluate across arbitrary time gaps in a single-shot, while remaining compatible with scientific pipelines such as assimilation, control, and uncertainty quantification. Looking forward, this framework offers scalable, theory-grounded surrogates that bridge data-driven representation learning and classical scientific modeling across disciplines.

---

## 11. SwasthLLM: a Unified Cross-Lingual, Multi-Task, and Meta-Learning Zero-Shot Framework for Medical Diagnosis Using Contrastive Representations

**论文链接:** [http://arxiv.org/abs/2509.20567v1](http://arxiv.org/abs/2509.20567v1)

**作者:** Ayan Sar, Pranav Singh Puri, Sumit Aich, Tanupriya Choudhury, Abhijit Kumar

**发布时间:** 2025-09-24

**备注:** Submitted to International Conference on Big Data 2025

### GPT解析

### 总结

SwasthLLM是一个统一的、零样本的、跨语言的和多任务学习的医疗诊断框架，能够在英语、印地语和孟加拉语等多种语言环境中有效工作，无需针对特定语言进行微调。

### 背景

在多语言医疗环境中，由于低资源语言的标注医疗数据稀缺以及不同人群间的语言变异性，从临床文本中自动进行疾病诊断仍然是一个具有挑战性的任务。

### 目的

开发一个能够在多种语言（英语、印地语和孟加拉语）上有效工作的医疗诊断框架，无需针对特定语言进行微调，特别是在低资源语言环境下也能表现良好。

### 方法

SwasthLLM核心使用多语言XLM-RoBERTa编码器，并增强了一个语言感知注意力机制和疾病分类头；引入Siamese对比学习模块对齐不同语言的语义表示；使用翻译一致性模块和对比投影头强化语言不变表示学习；采用多任务学习策略联合优化多个目标；使用MAML使模型能够快速适应新语言或任务；采用分阶段训练流程强调稳健的表示对齐。

### 主要发现

在监督设置下，SwasthLLM实现了97.22%的测试准确率和97.17%的F1分数；在零样本场景下，在印地语医疗文本上达到92.78%的准确率，在孟加拉语医疗文本上达到73.33%的准确率；表明在低资源背景下具有强大的泛化能力。

### 结论

SwasthLLM是一个有效的跨语言医疗诊断框架，能够在低资源语言环境下实现高性能诊断，无需针对特定语言进行微调。

### 翻译

在多语言医疗环境中，由于低资源语言的标注医疗数据稀缺以及不同人群间的语言变异性，从临床文本中自动进行疾病诊断仍然是一个具有挑战性的任务。本文提出了SwasthLLM，一个统一的、零样本的、跨语言的和多任务学习的医疗诊断框架，能够在英语、印地语和孟加拉语上有效工作，无需针对特定语言进行微调。SwasthLLM核心使用多语言XLM-RoBERTa编码器，并增强了一个语言感知注意力机制和疾病分类头，使模型能够提取与医学相关的信息，无论语言结构如何。为了对齐不同语言的语义表示，引入了一个Siamese对比学习模块，确保不同语言中的等效医学文本产生相似的嵌入。此外，翻译一致性模块和对比投影头强化了语言不变的表示学习。SwasthLLM使用多任务学习策略进行训练，联合优化疾病分类、翻译对齐和对比学习目标。此外，我们采用模型无关元学习（MAML）使模型具备快速适应未见语言或任务的能力，只需最少的数据。我们的分阶段训练流程在任务特定微调前强调稳健的表示对齐。广泛的评估显示，SwasthLLM在监督设置下实现了高诊断性能，测试准确率为97.22%，F1分数为97.17%。关键的是，在零样本场景下，它在印地语医疗文本上达到92.78%的准确率，在孟加拉语医疗文本上达到73.33%的准确率，展示了在低资源环境下的强大泛化能力。


### 论文摘要

In multilingual healthcare environments, automatic disease diagnosis from clinical text remains a challenging task due to the scarcity of annotated medical data in low-resource languages and the linguistic variability across populations. This paper proposes SwasthLLM, a unified, zero-shot, cross-lingual, and multi-task learning framework for medical diagnosis that operates effectively across English, Hindi, and Bengali without requiring language-specific fine-tuning. At its core, SwasthLLM leverages the multilingual XLM-RoBERTa encoder augmented with a language-aware attention mechanism and a disease classification head, enabling the model to extract medically relevant information regardless of the language structure. To align semantic representations across languages, a Siamese contrastive learning module is introduced, ensuring that equivalent medical texts in different languages produce similar embeddings. Further, a translation consistency module and a contrastive projection head reinforce language-invariant representation learning. SwasthLLM is trained using a multi-task learning strategy, jointly optimizing disease classification, translation alignment, and contrastive learning objectives. Additionally, we employ Model-Agnostic Meta-Learning (MAML) to equip the model with rapid adaptation capabilities for unseen languages or tasks with minimal data. Our phased training pipeline emphasizes robust representation alignment before task-specific fine-tuning. Extensive evaluation shows that SwasthLLM achieves high diagnostic performance, with a test accuracy of 97.22% and an F1-score of 97.17% in supervised settings. Crucially, in zero-shot scenarios, it attains 92.78% accuracy on Hindi and 73.33% accuracy on Bengali medical text, demonstrating strong generalization in low-resource contexts.

---

## 12. Beyond Visual Similarity: Rule-Guided Multimodal Clustering with explicit domain rules

**论文链接:** [http://arxiv.org/abs/2509.20501v1](http://arxiv.org/abs/2509.20501v1)

**作者:** Kishor Datta Gupta, Mohd Ariful Haque, Marufa Kamal, Ahmed Rafi Hasan, Md. Mahfuzur Rahman, Roy George

**发布时间:** 2025-09-24

**备注:** 12 pages, 9 figures

### GPT解析

### 总结

该论文提出了一种名为DARTVAE的规则引导多模态聚类框架，将领域特定约束直接整合到表示学习过程中，通过在损失函数中强制执行规则一致性和违规惩罚，实现了比传统聚类方法更有意义和可解释的聚类结果。

### 背景

传统的聚类技术通常仅依赖于输入数据中的相似性，限制了它们捕捉许多领域中至关重要的结构或语义约束的能力。

### 目的

引入DARTVAE框架，将领域特定约束直接整合到表示学习过程中，以实现更有意义和可解释的聚类结果。

### 方法

DARTVAE扩展了VAE架构，将显式规则、语义表示和数据驱动特征嵌入到统一的潜在空间中，并通过损失函数中的规则一致性和违规惩罚来强制执行约束合规性。规则由LLMs生成，结构化为知识图谱，并通过结合重建、KL散度、一致性和违规惩罚的损失函数来执行。

### 主要发现

在飞机和汽车数据集上的实验表明，规则引导的聚类产生了更具操作意义和可解释的聚类结果，例如隔离UAVs、统一隐形飞机或将SUV与轿车分开，同时改进了传统聚类指标。然而，该框架面临挑战：LLM生成的规则可能产生幻觉或冲突，过多的规则有过度拟合的风险，并且扩展到复杂领域会增加计算和一致性难度。

### 结论

通过将规则编码与学习表示相结合，DARTVAE实现了比纯数据驱动模型更有意义和一致的聚类结果，突显了约束引导多模态聚类在复杂、知识密集型环境中的实用性。

### 翻译

传统的聚类技术通常仅依赖于输入数据中的相似性，限制了它们捕捉许多领域中至关重要的结构或语义约束的能力。我们引入了领域感知规则触发变分自编码器（DARTVAE），这是一种规则引导的多模态聚类框架，将领域特定约束直接整合到表示学习过程中。DARTVAE通过将显式规则、语义表示和数据驱动特征嵌入到统一的潜在空间中来扩展VAE架构，同时通过损失函数中的规则一致性和违规惩罚来强制执行约束合规性。与仅依赖视觉相似性或将规则作为后处理过滤器应用的常规聚类方法不同，DARTVAE将规则视为第一类学习信号。规则由LLMs生成，结构化为知识图谱，并通过结合重建、KL散度、一致性和违规惩罚的损失函数来执行。在飞机和汽车数据集上的实验表明，规则引导的聚类产生了更具操作意义和可解释的聚类结果，例如隔离UAVs、统一隐形飞机或将SUV与轿车分开，同时改进了传统聚类指标。然而，该框架面临挑战：LLM生成的规则可能产生幻觉或冲突，过多的规则有过度拟合的风险，并且扩展到复杂领域会增加计算和一致性难度。通过将规则编码与学习表示相结合，DARTVAE实现了比纯数据驱动模型更有意义和一致的聚类结果，突显了约束引导多模态聚类在复杂、知识密集型环境中的实用性。


### 论文摘要

Traditional clustering techniques often rely solely on similarity in the input data, limiting their ability to capture structural or semantic constraints that are critical in many domains. We introduce the Domain Aware Rule Triggered Variational Autoencoder (DARTVAE), a rule guided multimodal clustering framework that incorporates domain specific constraints directly into the representation learning process. DARTVAE extends the VAE architecture by embedding explicit rules, semantic representations, and data driven features into a unified latent space, while enforcing constraint compliance through rule consistency and violation penalties in the loss function. Unlike conventional clustering methods that rely only on visual similarity or apply rules as post hoc filters, DARTVAE treats rules as first class learning signals. The rules are generated by LLMs, structured into knowledge graphs, and enforced through a loss function combining reconstruction, KL divergence, consistency, and violation penalties. Experiments on aircraft and automotive datasets demonstrate that rule guided clustering produces more operationally meaningful and interpretable clusters for example, isolating UAVs, unifying stealth aircraft, or separating SUVs from sedans while improving traditional clustering metrics. However, the framework faces challenges: LLM generated rules may hallucinate or conflict, excessive rules risk overfitting, and scaling to complex domains increases computational and consistency difficulties. By combining rule encodings with learned representations, DARTVAE achieves more meaningful and consistent clustering outcomes than purely data driven models, highlighting the utility of constraint guided multimodal clustering for complex, knowledge intensive settings.

---

## 13. Offline Goal-conditioned Reinforcement Learning with Quasimetric Representations

**论文链接:** [http://arxiv.org/abs/2509.20478v1](http://arxiv.org/abs/2509.20478v1)

**作者:** Vivek Myers, Bill Chunyuan Zheng, Benjamin Eysenbach, Sergey Levine

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种统一目标条件强化学习中对比表示和时间距离框架的新方法，通过拟度量表示空间结构和适当约束学习后续表示，实现最优目标达到。

### 背景

目标条件强化学习中，两种有效的表示结构框架是对比表示(学习'后续特征')和时间距离(将表示空间距离与状态到目标的过渡时间关联)。

### 目的

统一对比表示和时间距离两种框架，利用拟度量表示空间结构学习最优目标达到的后续表示，即使在次优数据和随机环境中也能工作。

### 方法

使用拟度量表示空间(三角形不等式)结构，添加适当约束来学习后续表示，实现最优目标达到。

### 主要发现

该方法能够利用拟度量距离参数化学习最优目标达到距离，即使在次优数据和随机环境中也能工作；保留了蒙特卡罗对比RL方法的稳定性和长期能力，同时获得了拟度量网络参数化的自由拼接能力。

### 结论

在离线GCRL基准测试中，该方法在拼接任务和嘈杂高维环境中均优于现有方法，结合了两种框架的优势。

### 翻译

目标条件强化学习(GCRL)的方法通常使用学习到的状态表示来提取目标达到策略。两种表示结构框架产生了特别有效的GCRL算法：(1)对比表示，其中方法使用对比目标学习'后续特征'，对未来结果进行推理；(2)时间距离，将表示空间中的(拟度量)距离与从状态到目标的过渡时间联系起来。我们提出了一种统一这两种框架的方法，使用拟度量表示空间的结构(三角形不等式)和适当的额外约束来学习后续表示，实现最优目标达到。与过去的工作不同，我们的方法能够利用拟度量距离参数化来学习最优目标达到距离，即使在次优数据和随机环境中也是如此。这使我们两全其美：我们保留了蒙特卡罗对比RL方法的稳定性和长期能力，同时获得了拟度量网络参数化的自由拼接能力。在现有的离线GCRL基准测试中，我们的表示学习目标提高了拼接任务的性能，而基于对比学习的方法在这些任务上表现不佳，并且在嘈杂、高维环境中也提高了性能，而基于拟度量网络的方法在这些环境中表现不佳。


### 论文摘要

Approaches for goal-conditioned reinforcement learning (GCRL) often use learned state representations to extract goal-reaching policies. Two frameworks for representation structure have yielded particularly effective GCRL algorithms: (1) *contrastive representations*, in which methods learn "successor features" with a contrastive objective that performs inference over future outcomes, and (2) *temporal distances*, which link the (quasimetric) distance in representation space to the transit time from states to goals. We propose an approach that unifies these two frameworks, using the structure of a quasimetric representation space (triangle inequality) with the right additional constraints to learn successor representations that enable optimal goal-reaching. Unlike past work, our approach is able to exploit a **quasimetric** distance parameterization to learn **optimal** goal-reaching distances, even with **suboptimal** data and in **stochastic** environments. This gives us the best of both worlds: we retain the stability and long-horizon capabilities of Monte Carlo contrastive RL methods, while getting the free stitching capabilities of quasimetric network parameterizations. On existing offline GCRL benchmarks, our representation learning objective improves performance on stitching tasks where methods based on contrastive learning struggle, and on noisy, high-dimensional environments where methods based on quasimetric networks struggle.

---

## 14. Predictive Coding-based Deep Neural Network Fine-tuning for Computationally Efficient Domain Adaptation

**论文链接:** [http://arxiv.org/abs/2509.20269v2](http://arxiv.org/abs/2509.20269v2)

**作者:** Matteo Cardoni, Sam Leroux

**发布时间:** 2025-09-24

**备注:** 20 pages, 4 figures

### GPT解析

### 总结

本文提出了一种结合反向传播和预测编码的混合训练方法，用于实现高效的设备域适应，使深度神经网络能够在动态环境中持续适应数据分布变化。

### 背景

深度神经网络在动态、真实世界环境中部署时，依靠单一静态模型往往不够。传感器漂移或光照变化导致的输入数据分布变化需要模型持续适应。

### 目的

提出一种混合训练方法，通过结合反向传播和预测编码的优势，实现高效的设备域适应。

### 方法

首先使用反向传播离线训练深度神经网络以获得高初始性能，然后使用预测编码进行在线适应，使模型能够恢复因输入数据分布变化而损失的准确性。

### 主要发现

这种方法利用反向传播在初始表示学习中的稳健性和预测编码在持续学习中的计算效率，特别适合资源受限的边缘设备或未来的神经形态加速器。在MNIST和CIFAR-10数据集上的实验结果表明，这种混合策略能够实现有效的适应，同时减少计算开销。

### 结论

这种混合策略为在动态环境中保持模型性能提供了有希望的解决方案。

### 翻译

随着深度神经网络在动态、真实世界环境中的日益部署，依靠单一静态模型通常是不够的。传感器漂移或光照变化引起的输入数据分布变化需要模型持续适应。在本文中，我们提出了一种混合训练方法，通过结合反向传播和预测编码的优势，实现高效的设备域适应。该方法首先使用反向传播离线训练深度神经网络以获得高初始性能。随后，采用预测编码进行在线适应，使模型能够恢复因输入数据分布变化而损失的准确性。这种方法利用了反向传播在初始表示学习中的稳健性和预测编码在持续学习中的计算效率，使其特别适合资源受限的边缘设备或未来的神经形态加速器。在MNIST和CIFAR-10数据集上的实验结果表明，这种混合策略能够实现有效的适应，同时减少计算开销，为在动态环境中保持模型性能提供了有希望的解决方案。


### 论文摘要

As deep neural networks are increasingly deployed in dynamic, real-world environments, relying on a single static model is often insufficient. Changes in input data distributions caused by sensor drift or lighting variations necessitate continual model adaptation. In this paper, we propose a hybrid training methodology that enables efficient on-device domain adaptation by combining the strengths of Backpropagation and Predictive Coding. The method begins with a deep neural network trained offline using Backpropagation to achieve high initial performance. Subsequently, Predictive Coding is employed for online adaptation, allowing the model to recover accuracy lost due to shifts in the input data distribution. This approach leverages the robustness of Backpropagation for initial representation learning and the computational efficiency of Predictive Coding for continual learning, making it particularly well-suited for resource-constrained edge devices or future neuromorphic accelerators. Experimental results on the MNIST and CIFAR-10 datasets demonstrate that this hybrid strategy enables effective adaptation with a reduced computational overhead, offering a promising solution for maintaining model performance in dynamic environments.

---

## 15. MolPILE - large-scale, diverse dataset for molecular representation learning

**论文链接:** [http://arxiv.org/abs/2509.18353v2](http://arxiv.org/abs/2509.18353v2)

**作者:** Jakub Adamczyk, Jakub Poziemski, Franciszek Job, Mateusz Król, Maciej Makowski

**发布时间:** 2025-09-22

### GPT解析

### 总结

这篇论文介绍了MolPILE，一个包含2.22亿化合物的预训练数据集，旨在解决分子表示学习中的数据集限制问题，通过在MolPILE上重新训练模型可以提高泛化性能。

### 背景

预训练数据集的大小、多样性和质量对基础模型的泛化能力至关重要。在化学信息学领域，现有小分子数据集的限制阻碍了分子表示学习的有效性。

### 目的

解决现有分子数据集的局限性，创建一个类似于ImageNet规模的标准化学数据集，以提高分子表示学习的效果。

### 方法

构建了MolPILE数据集，这是一个包含2.22亿化合物的集合，通过自动化筛选流程从6个大型数据库构建而成。对当前预训练数据集进行全面分析，并重新训练现有模型以评估性能提升。

### 主要发现

当前预训练数据集在训练机器学习模型方面存在显著不足；在MolPILE上重新训练现有模型可以改善泛化性能。

### 结论

MolPILE为模型训练提供了标准化资源，解决了分子化学领域对类ImageNet数据集的迫切需求。

### 翻译

预训练数据集的大小、多样性和质量决定了基础模型的泛化能力。尽管它们在化学信息学中的重要性日益增加，但由于现有小分子数据集的限制，分子表示学习的有效性受到了阻碍。为了解决这一差距，我们提出了MolPILE，这是一个大规模、多样化且经过严格筛选的2.22亿化合物集合，使用自动化筛选流程从6个大型数据库构建而成。我们对当前的预训练数据集进行了全面分析，指出了它们在训练机器学习模型方面的显著不足，并展示了如何在MolPILE上重新训练现有模型以提高泛化性能。这项工作为模型训练提供了标准化资源，解决了分子化学领域对类ImageNet数据集的迫切需求。


### 论文摘要

The size, diversity, and quality of pretraining datasets critically determine the generalization ability of foundation models. Despite their growing importance in chemoinformatics, the effectiveness of molecular representation learning has been hindered by limitations in existing small molecule datasets. To address this gap, we present MolPILE, large-scale, diverse, and rigorously curated collection of 222 million compounds, constructed from 6 large-scale databases using an automated curation pipeline. We present a comprehensive analysis of current pretraining datasets, highlighting considerable shortcomings for training ML models, and demonstrate how retraining existing models on MolPILE yields improvements in generalization performance. This work provides a standardized resource for model training, addressing the pressing need for an ImageNet-like dataset in molecular chemistry.

---

## 16. LAVA: Explainability for Unsupervised Latent Embeddings

**论文链接:** [http://arxiv.org/abs/2509.21149v1](http://arxiv.org/abs/2509.21149v1)

**作者:** Ivan Stresec, Joana P. Gonçalves

**发布时间:** 2025-09-25

**备注:** 28 pages, including references and appendix

### GPT解析

### 总结

本文介绍了一种名为局部感知变量关联(LAVA)的新方法，用于解释无监督黑盒模型的潜在空间组织及其与输入特征的关系。

### 背景

无监督黑盒模型可以推动科学发现但难以解释。现有监督学习的可解释性方法关注输入特征与预测目标的关系，而无监督对应方法应关注输入特征与学习到的潜在空间结构的联系。现有无监督学习解释方法要么过于细致，要么过于简化，难以提供有意义的信息。

### 目的

开发一种能够自动根据潜在邻近性关联相似样本的策略，以解释局部嵌入组织及其与输入特征的关系，特别适用于不产生映射函数的流形学习方法。

### 方法

引入LAVA(局部感知变量关联)，一种后期模型无关方法。该方法将潜在空间表示为一系列用原始特征间相关性描述的局部性(邻域)，然后揭示整个潜在空间中重复出现的相关性模式。

### 主要发现

基于MNIST和单细胞肾脏数据集的UMAP嵌入测试表明，LVA能够捕获相关的特征关联，并在潜在空间的看似遥远区域之间共享具有视觉和生物学相关性的局部模式。

### 结论

LVA提供了一种有效的方法来解释无监督黑盒模型的潜在空间结构，通过揭示特征关联和局部模式，帮助理解无监督学习的结果，促进科学发现。

### 翻译

无监督黑盒模型可以成为科学发现的驱动力，但仍然难以解释。关键在于，发现依赖于理解模型输出，这通常是多维潜在嵌入，而非明确定义的目标。虽然监督学习的可解释性通常试图揭示输入特征如何用于预测目标，但其无监督对应方法应该将输入特征与学习到的潜在空间结构联系起来。为无监督学习调整的监督模型可解释性方法要么提供单个样本解释，要么提供整个数据集的摘要解释。然而，如果没有自动策略根据样本的潜在邻近性将相似样本相互关联，解释要么过于细致，要么过于简化而缺乏意义。这对于不产生映射函数的流形学习方法尤其相关，因为我们只有其嵌入的相对空间组织。我们引入了局部感知变量关联(LAVA)，一种后期模型无关方法，旨在通过其与输入特征的关系解释局部嵌入组织。为此，LAVA将潜在空间表示为一系列用原始特征间相关性描述的局部性(邻域)，然后揭示整个潜在空间中重复出现的相关性模式。基于MNIST和单细胞肾脏数据集的UMAP嵌入，我们表明LVA捕获了相关的特征关联，并在潜在空间的看似遥远区域之间共享具有视觉和生物学相关性的局部模式。


### 论文摘要

Unsupervised black-box models can be drivers of scientific discovery, but remain difficult to interpret. Crucially, discovery hinges on understanding the model output, which is often a multi-dimensional latent embedding rather than a well-defined target. While explainability for supervised learning usually seeks to uncover how input features are used to predict a target, its unsupervised counterpart should relate input features to the structure of the learned latent space. Adaptations of supervised model explainability for unsupervised learning provide either single-sample or dataset-wide summary explanations. However, without automated strategies of relating similar samples to one another guided by their latent proximity, explanations remain either too fine-grained or too reductive to be meaningful. This is especially relevant for manifold learning methods that produce no mapping function, leaving us only with the relative spatial organization of their embeddings. We introduce Locality-Aware Variable Associations (LAVA), a post-hoc model-agnostic method designed to explain local embedding organization through its relationship with the input features. To achieve this, LAVA represents the latent space as a series of localities (neighborhoods) described in terms of correlations between the original features, and then reveals reoccurring patterns of correlations across the entire latent space. Based on UMAP embeddings of MNIST and a single-cell kidney dataset, we show that LAVA captures relevant feature associations, with visually and biologically relevant local patterns shared among seemingly distant regions of the latent spaces.

---

## 17. An Improved Quantum Software Challenges Classification Approach using Transfer Learning and Explainable AI

**论文链接:** [http://arxiv.org/abs/2509.21068v1](http://arxiv.org/abs/2509.21068v1)

**作者:** Nek Dil Khan, Javed Ali Khan, Mobashir Husain, Muhammad Sohail Khan, Arif Ali Khan, Muhammad Azeem Akbar, Shahid Hussain

**发布时间:** 2025-09-25

### GPT解析

### 总结

该研究分析了量子软件工程(QSE)领域的挑战，通过机器学习方法对Stack Overflow上的量子相关讨论进行分类，实现了95%的高准确率，并使用SHAP增强了模型可解释性。

### 背景

量子软件工程(QSE)是科技公司实践的研究领域。量子开发者在优化量子计算和QSE概念时面临挑战，他们使用Stack Overflow讨论问题并使用专门的量子标签标记帖子，但这些标签通常指向技术方面而非开发者帖子。

### 目的

通过分类问题来识别量子软件工程中的常见挑战，并开发一种基于机器学习的分类方法来准确对这些讨论进行分类。

### 方法

从Q&A平台提取2829个量子相关问题，分析帖子识别六种主要挑战类型(工具、理论、学习、概念、错误和API使用)，使用内容分析和扎根理论构建数据集，通过ChatGPT验证人工注释，微调transformer算法(BERT、DistilBERT和RoBERTa)进行分类，并与D&ML分类器比较，最后使用SHAP进行模型可解释性分析。

### 主要发现

BERT DistilBERT实现了95%的平均准确率，比D&ML分类器(FNN、CNN、LSTM)分别高出6%、9%和11%。Transformer方法在处理实际讨论(无需数据增强)时表现优异，SHAP分析揭示了语言特征如何驱动预测，提高了分类透明度。

### 结论

这些发现可以帮助量子供应商和论坛更好地组织讨论，提高可访问性和可读性。然而，需要与实际开发者和供应商进行实证评估研究。

### 翻译

量子软件工程(QSE)是科技公司实践的研究领域。量子开发者在优化量子计算和QSE概念时面临挑战。他们使用Stack Overflow讨论问题并使用专门的量子标签标记帖子，这些标签通常指向技术方面而非开发者帖子。基于量子概念对问题进行分类可以帮助识别频繁的QSE挑战。我们进行了研究将问题分类为各种挑战。我们使用量子相关标签从Q&A平台提取了2829个问题。分析了帖子以识别常见挑战并开发了一种新的扎根理论。挑战包括工具、理论、学习、概念、错误和API使用。通过内容分析和扎根理论，使用常见挑战对讨论进行注释，构建了一个真实数据集。ChatGPT验证了人工注释并解决了分歧。微调的transformer算法，包括BERT、DistilBERT和RoBERTa，将讨论分类为常见挑战。我们使用BERT DistilBERT实现了95%的平均准确率，而微调的深度和机器学习分类器，包括前馈神经网络、卷积神经网络和长短期记忆网络，分别实现了89%、86%和84%的准确率。基于Transformer的方法比基于D&ML的方法提高了6%的准确率，通过处理实际讨论，即无需数据增强。我们应用了SHAP进行模型可解释性分析，揭示了语言特征如何驱动预测，提高了分类的透明度。这些发现可以帮助量子供应商和论坛更好地组织讨论，提高可访问性和可读性。然而，需要与实际开发者和供应商进行实证评估研究。


### 论文摘要

Quantum Software Engineering (QSE) is a research area practiced by tech firms. Quantum developers face challenges in optimizing quantum computing and QSE concepts. They use Stack Overflow (SO) to discuss challenges and label posts with specialized quantum tags, which often refer to technical aspects rather than developer posts. Categorizing questions based on quantum concepts can help identify frequent QSE challenges. We conducted studies to classify questions into various challenges. We extracted 2829 questions from Q&A platforms using quantum-related tags. Posts were analyzed to identify frequent challenges and develop a novel grounded theory. Challenges include Tooling, Theoretical, Learning, Conceptual, Errors, and API Usage. Through content analysis and grounded theory, discussions were annotated with common challenges to develop a ground truth dataset. ChatGPT validated human annotations and resolved disagreements. Fine-tuned transformer algorithms, including BERT, DistilBERT, and RoBERTa, classified discussions into common challenges. We achieved an average accuracy of 95% with BERT DistilBERT, compared to fine-tuned Deep and Machine Learning (D&ML) classifiers, including Feedforward Neural Networks (FNN), Convolutional Neural Networks (CNN), and Long Short-Term Memory networks (LSTM), which achieved accuracies of 89%, 86%, and 84%, respectively. The Transformer-based approach outperforms the D&ML-based approach with a 6\% increase in accuracy by processing actual discussions, i.e., without data augmentation. We applied SHAP (SHapley Additive exPlanations) for model interpretability, revealing how linguistic features drive predictions and enhancing transparency in classification. These findings can help quantum vendors and forums better organize discussions for improved access and readability. However,empirical evaluation studies with actual developers and vendors are needed.

---

## 18. A Real-Time On-Device Defect Detection Framework for Laser Power-Meter Sensors via Unsupervised Learning

**论文链接:** [http://arxiv.org/abs/2509.20946v1](http://arxiv.org/abs/2509.20946v1)

**作者:** Dongqi Zheng, Wenjin Fu, Guangzong Chen

**发布时间:** 2025-09-25

### GPT解析

### 总结

提出了一种基于视觉的自动化系统，用于激光功率计传感器涂层的缺陷检测和分类，能够识别热损伤和划痕等影响激光能量测量精度的缺陷。

### 背景

激光功率计传感器涂层缺陷（如热损伤和划痕）会影响医疗和工业应用中激光能量测量的准确性，需要有效的检测方法。

### 目的

开发一种自动化的视觉系统，用于检测和分类激光功率计传感器涂层的缺陷，无需大量标记的缺陷数据集。

### 方法

使用无监督异常检测框架，仅在良好传感器图像上训练；包含三个关键组件：(1)基于拉普拉斯边缘检测和K-means聚类的预处理管道；(2)通过StyleGAN2进行合成数据增强；(3)基于UFlow的神经网络架构进行多尺度特征提取和异常图生成。

### 主要发现

在366个真实传感器图像上的实验评估显示，对缺陷样本的准确率为93.8%，对良好样本的准确率为89.3%，图像级AUROC为0.957，像素级AUROC为0.961。

### 结论

该系统通过自动化质量控制提供潜在年度成本节约，在设备实现中每张图像处理时间为0.5秒。

### 翻译

我们提出了一种基于视觉的自动化系统，用于激光功率计传感器涂层的缺陷检测和分类。我们的方法解决了识别涂层缺陷的关键挑战，如热损伤和划痕，这些缺陷会影响医疗和工业应用中激光能量测量的准确性。该系统采用无监督异常检测框架，仅在良好传感器图像上训练，学习正常涂层分布模式，能够检测已知和新型的缺陷类型，而无需大量标记的缺陷数据集。我们的方法包含三个关键组件：(1)使用拉普拉斯边缘检测和K-means聚类的稳健预处理管道，分割感兴趣区域；(2)通过StyleGAN2进行合成数据增强；(3)基于UFlow的神经网络架构，用于多尺度特征提取和异常图生成。在366个真实传感器图像上的实验评估显示，对缺陷样本的准确率为93.8%，对良好样本的准确率为89.3%，图像级AUROC为0.957，像素级AUROC为0.961。该系统通过自动化质量控制提供潜在年度成本节约，在设备实现中每张图像处理时间为0.5秒。


### 论文摘要

We present an automated vision-based system for defect detection and classification of laser power meter sensor coatings. Our approach addresses the critical challenge of identifying coating defects such as thermal damage and scratches that can compromise laser energy measurement accuracy in medical and industrial applications. The system employs an unsupervised anomaly detection framework that trains exclusively on ``good'' sensor images to learn normal coating distribution patterns, enabling detection of both known and novel defect types without requiring extensive labeled defect datasets. Our methodology consists of three key components: (1) a robust preprocessing pipeline using Laplacian edge detection and K-means clustering to segment the area of interest, (2) synthetic data augmentation via StyleGAN2, and (3) a UFlow-based neural network architecture for multi-scale feature extraction and anomaly map generation. Experimental evaluation on 366 real sensor images demonstrates $93.8\%$ accuracy on defective samples and $89.3\%$ accuracy on good samples, with image-level AUROC of 0.957 and pixel-level AUROC of 0.961. The system provides potential annual cost savings through automated quality control and processing times of 0.5 seconds per image in on-device implementation.

---

## 19. A Deep Transfer Learning-Based Low-overhead Beam Prediction in Vehicle Communications

**论文链接:** [http://arxiv.org/abs/2509.20659v1](http://arxiv.org/abs/2509.20659v1)

**作者:** Zhiqiang Xiao, Yuwen Cao, Mondher Bouazizi, Tomoaki Ohtsuki, Shahid Mumtaz

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了一种结合微调和域适应的迁移学习方法，用于解决源域和目标域数据分布差异大时的波束预测问题。

### 背景

现有的基于迁移学习的波束预测方法主要依赖于简单的微调，当目标域和源域之间的数据分布存在显著差异时，简单的微调限制了模型在目标域的性能。

### 目的

解决源域和目标域数据分布差异大时模型性能受限的问题，提高模型在目标域的表现。

### 方法

将域分类器集成到预训练模型的微调过程中，通过对抗训练使模型提取域不变特征，从而增强模型在目标域的性能。

### 主要发现

模拟结果表明，所提出的迁移学习方法在目标域中比纯微调方法实现了更好的可达速率性能，并且接近在目标域上从头开始训练时的性能。

### 结论

结合微调和域适应的迁移学习方法可以有效解决源域和目标域数据分布差异大的问题，显著提高模型在目标域的性能。

### 翻译

现有的基于迁移学习的波束预测方法主要依赖于简单的微调。当目标域和源域之间的数据分布存在显著差异时，简单的微调限制了模型在目标域的性能。为了解决这个问题，我们提出了一种结合微调和域适应的基于迁移学习的波束预测方法。我们将域分类器集成到预训练模型的微调过程中，模型通过域分类器进行对抗训练，提取域不变特征，从而增强模型在目标域的性能。模拟结果表明，所提出的基于迁移学习的波束预测方法在目标域中比纯微调方法实现了更好的可达速率性能，并且接近在目标域上从头开始训练时的性能。


### 论文摘要

Existing transfer learning-based beam prediction approaches primarily rely on simple fine-tuning. When there is a significant difference in data distribution between the target domain and the source domain, simple fine-tuning limits the model's performance in the target domain. To tackle this problem, we propose a transfer learning-based beam prediction method that combines fine-tuning with domain adaptation. We integrate a domain classifier into fine-tuning the pre-trained model. The model extracts domain-invariant features in adversarial training with domain classifier, which can enhance model performance in the target domain. Simulation results demonstrate that the proposed transfer learning-based beam prediction method achieves better achievable rate performance than the pure fine-tuning method in the target domain, and close to those when the training is done from scratch on the target domain.

---

## 20. Building Information Models to Robot-Ready Site Digital Twins (BIM2RDT): An Agentic AI Safety-First Framework

**论文链接:** [http://arxiv.org/abs/2509.20705v1](http://arxiv.org/abs/2509.20705v1)

**作者:** Reza Akhavian, Mani Amani, Johannes Mootz, Robert Ashe, Behrad Beheshti

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文介绍了BIM2RDT框架，这是一个代理人工智能系统，将静态建筑信息模型转变为动态、机器人就绪的数字孪生，优先考虑施工安全，通过整合多种数据流弥合BIM与实时现场条件的差距。

### 背景

建筑行业采用赛博物理系统和工地智能连接设计模型、实时现场传感和自主现场操作，可显著增强数字管理，但现有BIM数据与实时现场条件之间存在差距。

### 目的

开发一个框架将静态BIM模型转变为动态、机器人就绪的数字孪生，优先考虑执行过程中的安全，并通过整合多种数据流弥合BIM数据与实时现场条件之间的差距。

### 方法

引入语义重力ICP点云配准算法利用大型语言模型推理；创建反馈循环使机器人收集的数据更新数字孪生并优化任务路径；采用YOLOE目标检测和Shi-Tomasi角点检测识别和跟踪建筑元素；集成实时手臂振动监测并使用IFC标准映射安全事件。

### 主要发现

SG-ICP算法在遮挡特征场景中的对齐优于标准ICP，实现64.3%至88.3%的均方根误差降低；HAV集成在超过暴露限制时触发警告，提高了对ISO 5349-1标准的合规性。

### 结论

BIM2RDT框架成功将静态BIM转变为动态数字孪生，通过整合多种数据流和先进算法提高了建筑工地的安全性和效率，SG-ICP和HAV监测显著提升了系统准确性和安全性。

### 翻译

采用连接设计模型、实时现场传感和自主现场操作的赛博物理系统和工地智能可以显著增强建筑行业的数字管理。本文介绍了BIM2RDT框架，这是一个代理人工智能系统，旨在将静态建筑信息建模转变为动态的、机器人就绪的数字孪生，优先考虑执行过程中的安全。该框架通过整合三个关键数据流来弥合现有BIM数据和实时现场条件之间的差距：来自BIM模型的几何和语义信息、来自物联网传感器网络的活动数据以及机器人在现场遍历期间收集的视觉空间数据。该方法引入了语义重力ICP点云配准算法，利用大型语言模型推理。与传统方法相比，SG-ICP利用大型语言模型基于BIM语义推断特定对象、合理的方向先验，通过避免收敛到局部最小值提高对齐精度。这创建了一个反馈循环，机器人收集的数据更新数字孪生，进而优化任务路径。该框架采用YOLOE目标检测和Shi-Tomasi角点检测来识别和跟踪建筑元素，同时使用BIM几何作为先验地图。该框架还集成了实时手臂振动监测，使用IFC标准将传感器检测到的安全事件映射到数字孪生，以便干预。实验证明SG-ICP优于标准ICP，在遮挡特征场景中的对齐实现了64.3%至88.3%的均方根误差降低，确保了合理的方向。HAV集成在超过暴露限制时触发警告，提高了对ISO 5349-1标准的合规性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何将静态的建筑信息模型转变为动态的、机器人可用的工地数字孪生，并在施工过程中优先考虑安全。这个问题很重要，因为建筑行业正经历数字化转型，但传统BIM模型无法反映工地的实时变化；同时随着机器人技术在工地应用增加，需要将现有BIM数据与实时工地条件结合，确保机器人安全高效地导航和工作；此外施工安全监测和干预也是建筑行业的关键挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过整合三个关键数据流来设计方法：BIM模型的几何和语义信息、物联网传感器网络的实时活动数据、四足机器人在工地穿行期间收集的视觉-空间数据。作者借鉴了现有工作如传统ICP算法用于点云配准，但进行了创新，开发了语义-重力ICP算法，利用大型语言模型的推理能力来推断物体特定的、物理上合理的方向先验知识，避免收敛到局部最小值。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建智能反馈循环，机器人收集的数据更新数字孪生，数字孪生又为后续任务优化路径。整体流程包括：使用YOLOE开放词汇目标检测器和Shi-Tomasi角点检测识别和跟踪建筑元素；将BIM几何作为先验地图；应用语义-重力ICP算法对齐机器人捕获的点云与BIM模型；集成实时手臂振动监测将安全事件映射到数字孪生；使用智能代理AI引擎处理数据并做出决策。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：开发了语义-重力ICP算法利用大型语言模型推理；将静态BIM转变为动态机器人就绪的数字孪生；整合三种关键数据流（BIM信息、物联网数据、机器人数据）；集成实时手臂振动监测；使用智能代理AI作为系统认知引擎。相比之前工作，这个框架同时利用现有BIM数据、实时传感器流和自主决策，而之前工作通常只关注单一方面的技术。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文贡献了一个名为BIM2RDT的智能AI框架，通过整合BIM数据、机器人收集的传感器信息和大型语言模型的推理能力，将静态建筑模型转变为动态的、安全优先的工地数字孪生，显著提高了建筑机器人导航的准确性和施工安全性。'}


### 论文摘要

The adoption of cyber-physical systems and jobsite intelligence that connects design models, real-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT (Building Information Models to Robot-Ready Site Digital Twins), an agentic artificial intelligence (AI) framework designed to transform static Building Information Modeling (BIM) into dynamic, robot-ready digital twins (DTs) that prioritize safety during execution. The framework bridges the gap between pre-existing BIM data and real-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual-spatial data collected by robots during site traversal. The methodology introduces Semantic-Gravity ICP (SG-ICP), a point cloud registration algorithm that leverages large language model (LLM) reasoning. Unlike traditional methods, SG-ICP utilizes an LLM to infer object-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real-time Hand-Arm Vibration (HAV) monitoring, mapping sensor-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%--88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349-1.

---

## 21. OmniPlantSeg: Species Agnostic 3D Point Cloud Organ Segmentation for High-Resolution Plant Phenotyping Across Modalities

**论文链接:** [http://arxiv.org/abs/2509.21038v1](http://arxiv.org/abs/2509.21038v1)

**作者:** Andreas Gilson, Lukas Meyer, Oliver Scholz, Ute Schmid

**发布时间:** 2025-09-25

### GPT解析

### 总结

研究提出了一种名为KDSS的简单有效算法，用于植物器官点云分割，能够保留全分辨率数据并适用于不同植物种类和传感器模态。

### 背景

现有植物器官点云分割解决方案针对特定问题设计，专注于特定植物种类或特定传感器模态，且通常需要大量预处理和下采样以满足硬件或神经网络输入要求。

### 目的

开发一种独立于传感器数据和植物种类的点云下采样算法，无需下采样输入数据，从而实现对全分辨率点云的分割。

### 方法

提出KDSS算法并将其与当前最先进的分割模型结合，在不同模态（如摄影测量、激光三角测量和LiDAR）和各种植物种类上进行评估。

### 主要发现

KDSS与先进分割模型结合在不同传感器模态和植物种类上取得了令人满意的结果，成为一种轻量级保留分辨率的替代方案。

### 结论

KDSS是密集预处理和下采样方法的有效替代方案，可用于植物器官分割，不受植物种类和传感器模态限制。

### 翻译

准确的植物器官点云分割对于3D植物表型分析至关重要。现有解决方案是针对特定问题设计的，专注于特定植物种类或用于数据获取的指定传感器模态。此外，通常使用大量的预处理和对植物点云进行下采样，以满足硬件或神经网络输入大小的要求。我们提出了一种简单而有效的KDSS算法，用于生物点云的下采样，该算法独立于传感器数据和植物种类。这种方法的主要优点是我们不需要下采样输入数据，从而能够对全分辨率点云进行分割。将KD-SS与当前最先进的分割模型结合，在不同的模态（如摄影测量、激光三角测量和LiDAR）和各种植物种类上评估，显示出令人满意的结果。我们提出KD-SS作为轻量级的保留分辨率的替代方案，用于密集的预处理和下采样方法进行植物器官分割，无论使用的种类和传感器模态如何。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决植物器官点云分割的通用性和分辨率问题。现有方法通常针对特定植物种类或传感器设计，且需要降采样点云以适应神经网络输入，这会导致信息丢失和细节丢失。这一问题在植物表型研究中至关重要，因为高分辨率扫描能捕捉微小特征和细节，而准确的器官分割是提取有意义信息的基础，对农业数字化转型和精细作物管理具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性，特别是球形子采样(Spherical Sub Sampling)方法因固定半径导致点云密度分布失真的问题。作者借鉴了Scholz等人的球形子采样概念，但引入KD-tree算法进行改进，创建了KD-SS算法。这种方法优化了运行时间，同时保持原始点云的所有点。作者还使用了DGCNN作为分割模型，这是点云分割领域的先进方法，通过结合KD-SS和DGCNN，实现了无需降采样的全分辨率分割。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是KD-SS子采样算法，它可以将任意大小的点云分割成适合神经网络输入大小的子样本，同时保持原始分辨率。整体流程包括：1)初始化点云数据和每个子样本的点数；2)创建KD-tree；3)随机选择中心点并选取其N个最近邻居作为子样本；4)保存子样本和特征向量；5)从原始数据中移除已采样点；6)重复直到剩余点不足；7)使用DGCNN对子样本进行分割；8)合并所有分割后的子样本，得到全分辨率的带标签点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出KD-SS子采样算法，能在保持全分辨率的同时将点云分割成适合神经网络的大小；2)开发OmniPlantSeg流程，实现跨植物种类和传感器模态的通用分割；3)展示方法在不同数据集上的有效性。相比之前工作，不同之处在于：不需要降采样输入数据，保留了全分辨率信息；具有更好的通用性，能处理不同植物和传感器数据；在保持高分辨率的同时实现了有竞争力的分割性能；共享权重模型表明单个模型可泛化到不同物种。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了KD-SS子采样算法和OmniPlantSeg流程，实现了无需降采样的全分辨率植物器官分割，能够在不同植物种类和传感器模式下提供准确分割结果，为高分辨率植物表型分析提供了新方法。'}


### 论文摘要

Accurate point cloud segmentation for plant organs is crucial for 3D plant phenotyping. Existing solutions are designed problem-specific with a focus on certain plant species or specified sensor-modalities for data acquisition. Furthermore, it is common to use extensive pre-processing and down-sample the plant point clouds to meet hardware or neural network input size requirements. We propose a simple, yet effective algorithm KDSS for sub-sampling of biological point clouds that is agnostic to sensor data and plant species. The main benefit of this approach is that we do not need to down-sample our input data and thus, enable segmentation of the full-resolution point cloud. Combining KD-SS with current state-of-the-art segmentation models shows satisfying results evaluated on different modalities such as photogrammetry, laser triangulation and LiDAR for various plant species. We propose KD-SS as lightweight resolution-retaining alternative to intensive pre-processing and down-sampling methods for plant organ segmentation regardless of used species and sensor modality.

---

## 22. EvoMail: Self-Evolving Cognitive Agents for Adaptive Spam and Phishing Email Defense

**论文链接:** [http://arxiv.org/abs/2509.21129v1](http://arxiv.org/abs/2509.21129v1)

**作者:** Wei Huang, De-Tian Chu, Lin-Yuan Bai, Wei Kang, Hai-Tao Zhang, Bo Li, Zhi-Mo Han, Jing Ge, Hai-Feng Lin

**发布时间:** 2025-09-25

### GPT解析

### 总结

EvoMail是一种自进化的认知智能体框架，用于稳健检测垃圾邮件和网络钓鱼攻击，通过构建异构电子邮件图和对抗性自进化循环实现高性能检测。

### 背景

现代垃圾邮件和网络钓鱼攻击已超越关键词黑名单或简单启发式方法，攻击者创建多模态活动结合文本、混淆URL、伪造头和恶意附件，并在几天内调整策略以绕过过滤器。传统垃圾邮件检测系统依赖静态规则或单模态模型，难以整合异构信号或持续适应，导致性能迅速下降。

### 目的

提出EvoMail，一个用于稳健检测垃圾邮件和网络钓鱼的自进化认知智能体框架。

### 方法

EvoMail构建统一的异构电子邮件图，融合文本内容、元数据和嵌入资源；使用由大型语言模型增强的认知图神经网络进行上下文感知推理；通过红队生成新的规避策略，蓝队从失败中学习并将经验压缩到内存模块中实现对抗性自进化循环。

### 主要发现

在多个真实世界数据集和合成对抗变体上的实验表明，EvoMail在检测准确性、对 evolving 垃圾邮件策略的适应性以及推理轨迹的可解释性方面 consistently 超越最先进的基线。

### 结论

EvoMail作为抵御下一代垃圾邮件和网络钓鱼威胁的弹性且可解释的防御框架具有显著潜力。

### 翻译

现代垃圾邮件和网络钓鱼攻击已经远远超出了关键词黑名单或简单启发式方法。攻击者现在创建多模态活动，将自然语言文本与混淆URL、伪造头和恶意附件相结合，在几天内调整策略以绕过过滤器。依赖静态规则或单模态模型的传统垃圾邮件检测系统难以整合异构信号或持续适应，导致性能迅速下降。我们提出了EvoMail，一种用于稳健检测垃圾邮件和网络钓鱼的自进化认知智能体框架。EvoMail首先构建统一的异构电子邮件图，融合文本内容、元数据（头、发件人、域名）和嵌入资源（URL、附件）。由大型语言模型增强的认知图神经网络在这些源之间执行上下文感知推理，以识别协调的垃圾邮件活动。最重要的是，EvoMail参与对抗性自进化循环：红队智能体生成新的规避策略，如字符混淆或AI生成的钓鱼文本，而蓝队检测器从失败中学习，将经验压缩到内存模块中，并在未来推理中重用这些经验。在真实世界数据集（Enron-Spam、Ling-Spam、SpamAssassin和TREC）和合成对抗变体上的广泛实验表明，EvoMail在检测准确性、对 evolving 垃圾邮件策略的适应性以及推理轨迹的可解释性方面 consistently 超越最先进的基线。这些结果突显了EvoMail作为抵御下一代垃圾邮件和网络钓鱼威胁的弹性且可解释的防御框架的潜力。


### 论文摘要

Modern email spam and phishing attacks have evolved far beyond keyword blacklists or simple heuristics. Adversaries now craft multi-modal campaigns that combine natural-language text with obfuscated URLs, forged headers, and malicious attachments, adapting their strategies within days to bypass filters. Traditional spam detection systems, which rely on static rules or single-modality models, struggle to integrate heterogeneous signals or to continuously adapt, leading to rapid performance degradation.   We propose EvoMail, a self-evolving cognitive agent framework for robust detection of spam and phishing. EvoMail first constructs a unified heterogeneous email graph that fuses textual content, metadata (headers, senders, domains), and embedded resources (URLs, attachments). A Cognitive Graph Neural Network enhanced by a Large Language Model (LLM) performs context-aware reasoning across these sources to identify coordinated spam campaigns. Most critically, EvoMail engages in an adversarial self-evolution loop: a ''red-team'' agent generates novel evasion tactics -- such as character obfuscation or AI-generated phishing text -- while the ''blue-team'' detector learns from failures, compresses experiences into a memory module, and reuses them for future reasoning.   Extensive experiments on real-world datasets (Enron-Spam, Ling-Spam, SpamAssassin, and TREC) and synthetic adversarial variants demonstrate that EvoMail consistently outperforms state-of-the-art baselines in detection accuracy, adaptability to evolving spam tactics, and interpretability of reasoning traces. These results highlight EvoMail's potential as a resilient and explainable defense framework against next-generation spam and phishing threats.

---

## 23. GraphUniverse: Enabling Systematic Evaluation of Inductive Generalization

**论文链接:** [http://arxiv.org/abs/2509.21097v1](http://arxiv.org/abs/2509.21097v1)

**作者:** Louis Van Langendonck, Guillermo Bernárdez, Nina Miolane, Pere Barlet-Ros

**发布时间:** 2025-09-25

### GPT解析

### 总结

本研究提出了GraphUniverse框架，用于生成整个图家族以评估图模型在归纳泛化方面的大规模性能，发现强大的同构性能是归纳泛化的不良预测因素，且对分布转移的鲁棒性对模型架构和初始图区域都很敏感。

### 背景

图学习中的一个基本挑战是理解模型如何推广到新的、未见过的图上。现有方法局限于单图、同构设置，即模型在同一图结构上训练和测试，缺乏对归纳泛化的系统评估。

### 目的

开发一个框架来生成整个图家族，实现归纳泛化的大规模系统评估，并研究不同图模型架构的泛化能力和鲁棒性。

### 方法

提出GraphUniverse框架，生成具有持久语义社区的图，确保概念一致性同时允许对同配性和度分布等结构特性进行细粒度控制。对多种架构（包括GNN、图变换器和拓扑架构）进行基准测试。

### 主要发现

1. 强大的同构性能是归纳泛化的不良预测因素；2. 对分布转移的鲁棒性不仅对模型架构选择敏感，而且对初始图区域（如高同配性与低同配性）也很敏感。

### 结论

GraphUniverse的灵活性和可扩展性可以促进鲁棒和真正可推广架构的开发，包括下一代图基础模型，为图学习领域提供了新的评估基准和工具。

### 翻译

图学习中的一个基本挑战是理解模型如何推广到新的、未见过的图上。虽然合成基准为分析提供了受控环境，但现有方法局限于单图、同构设置，即模型在同一图结构上训练和测试。为解决这一差距，我们引入了GraphUniverse，这是一个用于生成整个图家族的框架，以实现归纳泛化的大规模系统评估。我们的核心创新是生成具有持久语义社区的图，确保概念一致性的同时允许对同配性和度分布等结构特性进行细粒度控制。这使得关键但未被充分探索的鲁棒性测试成为可能，例如在受控分布转移下的性能。对各种架构（从GNN到图变换器和拓扑架构）的基准测试表明，强大的同构性能是归纳泛化的不良预测因素。此外，我们发现对分布转移的鲁棒性不仅对模型架构选择高度敏感，而且对初始图区域（如高同配性与低同配性）也很敏感。除了基准测试外，GraphUniverse的灵活性和可扩展性可以促进鲁棒和真正可推广架构的开发，包括下一代图基础模型。交互式演示可在https://graphuniverse.streamlit.app获取。


### 论文摘要

A fundamental challenge in graph learning is understanding how models generalize to new, unseen graphs. While synthetic benchmarks offer controlled settings for analysis, existing approaches are confined to single-graph, transductive settings where models train and test on the same graph structure. Addressing this gap, we introduce GraphUniverse, a framework for generating entire families of graphs to enable the first systematic evaluation of inductive generalization at scale. Our core innovation is the generation of graphs with persistent semantic communities, ensuring conceptual consistency while allowing fine-grained control over structural properties like homophily and degree distributions. This enables crucial but underexplored robustness tests, such as performance under controlled distribution shifts. Benchmarking a wide range of architectures -- from GNNs to graph transformers and topological architectures -- reveals that strong transductive performance is a poor predictor of inductive generalization. Furthermore, we find that robustness to distribution shift is highly sensitive not only to model architecture choice but also to the initial graph regime (e.g., high vs. low homophily). Beyond benchmarking, GraphUniverse's flexibility and scalability can facilitate the development of robust and truly generalizable architectures -- including next-generation graph foundation models. An interactive demo is available at https://graphuniverse.streamlit.app.

---

## 24. Feature Augmentation of GNNs for ILPs: Local Uniqueness Suffices

**论文链接:** [http://arxiv.org/abs/2509.21000v1](http://arxiv.org/abs/2509.21000v1)

**作者:** Qingyu Han, Qian Li, Linxin Yang, Qian Chen, Qingjiang Shi, Ruoyu Sun

**发布时间:** 2025-09-25

**备注:** 9 pages, 6 Tables

### GPT解析

### 总结

该研究提出了一种基于d跳唯一性着色的局部唯一标识符方案及其衍生的ColorGNN和ColorUID方法，解决了整数线性规划中图神经网络表达能力和泛化能力之间的权衡问题。

### 背景

整数线性规划是现实世界优化的核心但 notoriously 难以解决。学习优化已成为有前景的范式，图神经网络作为标准骨干。然而，标准匿名GNNs在表达ILPs方面存在局限性，而添加全局唯一标识符的常见方法会引入虚假相关性，严重损害泛化能力。

### 目的

解决标准匿名GNNs在ILPs中的表达能力限制，以及全局唯一标识符引入的虚假相关性和泛化能力下降问题。

### 方法

提出基于d跳唯一性着色的局部唯一标识符方案，确保标识符仅在节点的d跳邻域内唯一。基于此方案，引入ColorGNN通过颜色条件嵌入整合颜色信息，以及ColorUID这一轻量级特征级变体。

### 主要发现

对于d层网络，Local-UIDs能达到全局唯一标识符的表达能力，同时提供更强的泛化能力。实验表明该方法在三个ILP基准测试上取得显著提升，在线性规划数据集上表现出良好的OOD泛化能力，并与最先进方法配对时改进了图级任务。

### 结论

所提出的Local-UID方案及其衍生的ColorGNN和ColorUID方法有效解决了ILPs中表达能力和泛化能力之间的权衡问题，在多个任务和基准测试中表现出色。

### 翻译

整数线性规划是现实世界优化的核心，但 notoriously 难以解决。学习优化已成为有前景的范式，图神经网络作为标准骨干。然而，标准匿名GNNs在表达ILPs方面存在局限性，而添加全局唯一标识符的常见方法会引入虚假相关性，严重损害泛化能力。为了解决这种权衡，我们提出基于d跳唯一性着色的局部唯一标识符方案，确保标识符仅在节点的d跳邻域内唯一。基于此方案，我们引入ColorGNN通过颜色条件嵌入整合颜色信息，以及ColorUID这一轻量级特征级变体。我们证明对于d层网络，Local-UIDs能达到全局唯一标识符的表达能力，同时提供更强的泛化能力。大量实验表明，我们的方法在三个ILP基准测试上取得显著提升，在线性规划数据集上表现出良好的OOD泛化能力，并与最先进方法配对时进一步改进了图级任务。


### 论文摘要

Integer Linear Programs (ILPs) are central to real-world optimizations but notoriously difficult to solve. Learning to Optimize (L2O) has emerged as a promising paradigm, with Graph Neural Networks (GNNs) serving as the standard backbone. However, standard anonymous GNNs are limited in expressiveness for ILPs, and the common enhancement of augmenting nodes with globally unique identifiers (UIDs) typically introduces spurious correlations that severely harm generalization. To address this tradeoff, we propose a parsimonious Local-UID scheme based on d-hop uniqueness coloring, which ensures identifiers are unique only within each node's d-hop neighborhood. Building on this scheme, we introduce ColorGNN, which incorporates color information via color-conditioned embeddings, and ColorUID, a lightweight feature-level variant. We prove that for d-layer networks, Local-UIDs achieve the expressive power of Global-UIDs while offering stronger generalization. Extensive experiments show that our approach (i) yields substantial gains on three ILP benchmarks, (ii) exhibits strong OOD generalization on linear programming datasets, and (iii) further improves a general graph-level task when paired with a state-of-the-art method.

---

## 25. FracAug: Fractional Augmentation boost Graph-level Anomaly Detection under Limited Supervision

**论文链接:** [http://arxiv.org/abs/2509.20978v1](http://arxiv.org/abs/2509.20978v1)

**作者:** Xiangyu Dong, Xingyi Zhang, Sibo Wang

**发布时间:** 2025-09-25

### GPT解析

### 总结

论文提出了FracAug，一个创新的插件式图增强框架，用于解决图级别异常检测中的高标注成本和数据不平衡问题。该框架通过生成语义一致的图变体和相互验证的伪标记来增强GNN性能，在多种GNN和数据集上展现出显著的通用性和有效性。

### 背景

图级别异常检测在药物发现等多个领域至关重要，但高标注成本和数据集不平衡问题限制了图神经网络(GNNs)的性能表现。

### 目的

解决图级别异常检测中的高标注成本和数据集不平衡问题，提高GNN的性能。

### 方法

提出FracAug框架，通过在给定图中学习语义，利用加权距离感知边际损失引导合成部分变体，捕获多尺度拓扑生成多样且保持语义的图，不受数据不平衡影响。然后利用原始图和增强图的预测为未标记数据分配伪标记，迭代扩大训练集。作为模型无关模块，可与各种GNN兼容。

### 主要发现

在12个真实数据集上对14种GNN进行的实验显示了一致的性能提升，平均AUROC提高5.72%，AUPRC提高7.23%，F1分数提高4.18%。

### 结论

FracAug是一个有效的插件式增强框架，能够显著提高图级别异常检测任务的性能，具有广泛的适用性。

### 翻译

图级别异常检测(GAD)在药物发现等多个领域至关重要，然而高昂的标注成本和数据集不平衡问题限制了图神经网络(GNNs)的性能。为解决这些问题，我们提出了FracAug，一个创新的插件式增强框架，通过生成语义一致的图变体和相互验证的伪标记来增强GNN。与之前的启发式方法不同，FracAug在给定图中学习语义，并由一种新颖的加权距离感知边际损失引导，合成部分变体，捕获多尺度拓扑以生成多样且保持语义的图，不受数据不平衡影响。然后，FracAug利用原始图和增强图的预测来为未标记数据分配伪标记，迭代扩大训练集。作为一个与各种GNN兼容的模型无关模块，FracAug展示了显著的通用性和有效性：在12个真实数据集上对14种GNN进行的实验显示了一致的性能提升，平均AUROC、AUPRC和F1分数分别提高了5.72%、7.23%和4.18%。


### 论文摘要

Graph-level anomaly detection (GAD) is critical in diverse domains such as drug discovery, yet high labeling costs and dataset imbalance hamper the performance of Graph Neural Networks (GNNs). To address these issues, we propose FracAug, an innovative plug-in augmentation framework that enhances GNNs by generating semantically consistent graph variants and pseudo-labeling with mutual verification. Unlike previous heuristic methods, FracAug learns semantics within given graphs and synthesizes fractional variants, guided by a novel weighted distance-aware margin loss. This captures multi-scale topology to generate diverse, semantic-preserving graphs unaffected by data imbalance. Then, FracAug utilizes predictions from both original and augmented graphs to pseudo-label unlabeled data, iteratively expanding the training set. As a model-agnostic module compatible with various GNNs, FracAug demonstrates remarkable universality and efficacy: experiments across 14 GNNs on 12 real-world datasets show consistent gains, boosting average AUROC, AUPRC, and F1-score by up to 5.72%, 7.23%, and 4.18%, respectively.

---

## 26. Decoding the Surgical Scene: A Scoping Review of Scene Graphs in Surgery

**论文链接:** [http://arxiv.org/abs/2509.20941v1](http://arxiv.org/abs/2509.20941v1)

**作者:** Angelo Henriques, Korab Hoxha, Daniel Zapp, Peter C. Issa, Nassir Navab, M. Ali Nasseri

**发布时间:** 2025-09-25

**备注:** Submitted to Medical Image Analysis. Under review. 49 pages, 9  figures. An interactive version of the summary tables is available at  osf.io/fruq8

### GPT解析

### 总结

这篇综述系统地研究了场景图在手术领域的应用发展，揭示了研究中的数据鸿沟问题，并展示了从基础图神经网络到专业基础模型的演进过程。场景图已成为手术分析和工作流识别等任务的关键技术，同时也在可控手术模拟等生成任务中展现出潜力。

### 背景

场景图提供结构化的关系表示，对于解码复杂、动态的手术环境至关重要。随着医疗技术的发展，手术场景的智能化分析需求日益增长。

### 目的

系统地绘制手术中场景图研究的演变图景，包括其应用、方法学进展和未来方向。

### 方法

采用PRISMA-ScR指导的范围综述方法，对手术场景图研究进行系统性分析。

### 主要发现

1. 研究领域呈现快速增长，但存在关键的数据鸿沟：内部视图研究(如三元组识别)几乎仅使用真实世界2D视频，而外部视图4D建模则严重依赖模拟数据。2. 方法学上，该领域已从基础图神经网络发展到专业基础模型，这些模型在手术环境中显著优于通用大型视觉语言模型。3. 场景图已成为手术工作流识别和自动化安全监控等分析任务，以及可控手术模拟等生成任务的核心技术。4. 数据标注和实时实施方面仍存在挑战，但新兴技术正在积极解决这些问题。

### 结论

手术场景图正在发展为一种重要的语义桥梁，使新一代智能系统能够提高手术安全性、效率和培训质量。

### 翻译

场景图提供了解码复杂、动态手术环境所必需的结构化关系表示。这篇基于PRISMA-ScR指导的范围综述系统地绘制了手术中场景图研究的演变图景，记录了其应用、方法学进展和未来方向。分析揭示了快速增长，但发现了一个关键的数据鸿沟：内部视图研究(如三元组识别)几乎 exclusively 使用真实世界2D视频，而外部视图4D建模则严重依赖模拟数据，暴露了一个关键的转化研究差距。在方法学上，该领域已从基础图神经网络发展到专业基础模型，这些模型在手术环境中现在显著优于通用大型视觉语言模型。这一进展已使场景图成为分析(如工作流识别和自动化安全监控)和生成任务(如可控手术模拟)的核心技术。尽管数据标注和实时实施方面的挑战仍然存在，但它们正通过新兴技术得到积极解决。手术场景图正在发展为一种必要的语义桥梁，使新一代智能系统能够提高手术安全性、效率和培训质量。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决的问题是系统性地梳理和总结'场景图'在手术领域的研究现状、应用和发展趋势。这个问题在现实中非常重要，因为手术环境极其复杂动态，手术中多个器械同时操作，关键解剖结构距操作点仅毫米级距离；现有方法常单独分析特定组件而忽略丰富关系背景；准确理解手术场景中实体间的相互作用对提高手术安全性至关重要，手术情境意识不足是导致可预防不良事件的重要因素。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者采用范围综述(Scoping Review)方法，设计明确的纳入和排除标准，在三个数据库系统搜索，辅以Google Scholar关键词搜索、引用搜索和专家咨询。他们使用PRISMA-ScR框架指导综述过程，构建了三维分类系统（时间、维度、视角）来组织研究。这种方法借鉴了系统综述的现有方法，但针对手术场景图这一新兴领域进行了调整和创新，确保方法透明和可重复。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是系统性地梳理手术场景图研究，通过构建全面文献地图，揭示该领域发展趋势、方法学进步和未来方向。整体流程包括：背景介绍(解释手术场景图动机和历史)；方法学框架(介绍范围综述方法)；分类系统(提出三维分类框架)；场景图构建方法(分析数据源、数据集和计算方法)；应用和演变(追踪从内部场景图到与AI融合的发展)；结果分析(回答四个研究问题)；讨论(总结差距、挑战和未来方向)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：首个专注于手术场景图的系统性综述；提出三维分类框架(时间、空间、视角)；揭示'数据鸿沟'(内部视图用真实2D视频，外部视图依赖模拟数据)；追踪方法学演变(从图神经网络到专用基础模型)；识别研究热点和冷点。相比之前工作，这篇论文专注于场景图表示本身而非特定下游应用，采用更系统全面的方法梳理新兴领域，不仅总结现状还通过识别差距为未来研究提供方向，强调多模态融合和基础模型等最新趋势。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统性地梳理手术场景图研究，揭示了该领域从基础图模型到专用基础模型的演变轨迹，指出了关键的数据鸿沟和研究空白，为未来手术AI的发展提供了重要指导。'}


### 论文摘要

Scene graphs (SGs) provide structured relational representations crucial for decoding complex, dynamic surgical environments. This PRISMA-ScR-guided scoping review systematically maps the evolving landscape of SG research in surgery, charting its applications, methodological advancements, and future directions. Our analysis reveals rapid growth, yet uncovers a critical 'data divide': internal-view research (e.g., triplet recognition) almost exclusively uses real-world 2D video, while external-view 4D modeling relies heavily on simulated data, exposing a key translational research gap. Methodologically, the field has advanced from foundational graph neural networks to specialized foundation models that now significantly outperform generalist large vision-language models in surgical contexts. This progress has established SGs as a cornerstone technology for both analysis, such as workflow recognition and automated safety monitoring, and generative tasks like controllable surgical simulation. Although challenges in data annotation and real-time implementation persist, they are actively being addressed through emerging techniques. Surgical SGs are maturing into an essential semantic bridge, enabling a new generation of intelligent systems to improve surgical safety, efficiency, and training.

---

## 27. GALAX: Graph-Augmented Language Model for Explainable Reinforcement-Guided Subgraph Reasoning in Precision Medicine

**论文链接:** [http://arxiv.org/abs/2509.20935v1](http://arxiv.org/abs/2509.20935v1)

**作者:** Heming Zhang, Di Huang, Wenyu Li, Michael Province, Yixin Chen, Philip Payne, Fuhai Li

**发布时间:** 2025-09-25

### GPT解析

### 总结

该论文提出了GALAX框架，通过图过程奖励模型(GPRM)引导的强化学习，将预训练图神经网络(GNNs)整合到大语言模型(LLMs)中，实现逐步生成疾病相关子图并无需显式中间推理注释。同时引入了Target-QA基准测试，结合CRISPR识别靶点、多组学特征和生物医学图知识，支持长上下文推理。

### 背景

在精准医疗中，定量多组学特征、拓扑背景和文本生物知识对识别疾病关键信号通路和靶点至关重要。现有方法存在局限性：数值组学忽略拓扑背景，文本中心LLMs缺乏定量基础推理，图模型未充分利用节点语义和LLMs泛化能力，限制了机制可解释性。过程奖励模型(PRM)受限于不可靠的中间评估和奖励黑客攻击，且计算成本高。

### 目的

通过LLMs整合定量多组学信号、拓扑结构和节点注释以及文献规模文本，使用子图推理作为连接数字证据、拓扑知识和语言上下文的原则桥梁，实现可靠和可解释的靶点和通路发现。

### 方法

提出GALAX框架，通过图过程奖励模型(GPRM)引导的强化学习，将预训练图神经网络(GNNs)整合到大语言模型(LLMs)中，逐步生成疾病相关子图并由预训练GNN迭代评估，实现过程级监督。引入Target-QA基准测试，结合CRISPR识别靶点、多组学特征和跨多种癌细胞系生物医学图知识，支持GNN预训练和长上下文推理。

### 主要发现

GALAX框架通过子图推理有效整合数字证据、拓扑知识和语言上下文，克服现有方法局限性。Target-QA基准测试提供了可扩展、生物有据可依的框架，支持强化引导的子图推理，实现可靠和可解释的靶点和通路发现。

### 结论

GALAX框架和Target-QA基准测试为精准医学中可靠和可解释的靶点和通路发现提供了可扩展和生物有据可依的框架，通过整合定量多组学特征、拓扑结构和文本生物知识，提升机制可解释性。

### 翻译

在精准医疗中，定量多组学特征、拓扑背景和文本生物知识在识别疾病关键信号通路和靶点中发挥着至关重要的作用。现有的流程仅捕捉了其中的一部分——数值组学忽略了拓扑背景，以文本为中心的大语言模型缺乏定量基础推理，而仅图模型未充分利用节点语义和大语言模型的泛化能力——限制了机制可解释性。尽管过程奖励模型(PRM)旨在指导大语言模型中的推理，但它们仍然受到不可靠的中间评估和奖励黑客攻击的困扰，且计算成本高。这些差距促使我们通过大语言模型整合定量多组学信号、带有节点注释的拓扑结构和文献规模的文本，使用子图推理作为连接数字证据、拓扑知识和语言上下文的原则桥梁。因此，我们提出了GALAX，这是一个创新框架，通过图过程奖励模型(GPRM)引导的强化学习，将预训练图神经网络(GNNs)整合到大语言模型(LLMs)中，该模型以逐步方式生成疾病相关的子图，并由预训练的GNN迭代评估，实现无需显式中间推理注释的过程级监督。作为应用，我们还引入了Target-QA，这是一个基准测试，结合了CRISPR识别的靶点、多组学特征和跨多种癌细胞系的生物医学图知识，它支持GNN预训练以监督逐步图构建，并支持在文本-数字图上的长上下文推理，为精准医学中可靠和可解释的靶点和通路发现提供了可扩展和生物有据可依的框架。


### 论文摘要

In precision medicine, quantitative multi-omic features, topological context, and textual biological knowledge play vital roles in identifying disease-critical signaling pathways and targets. Existing pipelines capture only part of these-numerical omics ignore topological context, text-centric LLMs lack quantitative grounded reasoning, and graph-only models underuse node semantics and the generalization of LLMs-limiting mechanistic interpretability. Although Process Reward Models (PRMs) aim to guide reasoning in LLMs, they remain limited by unreliable intermediate evaluation, and vulnerability to reward hacking with computational cost. These gaps motivate integrating quantitative multi-omic signals, topological structure with node annotations, and literature-scale text via LLMs, using subgraph reasoning as the principle bridge linking numeric evidence, topological knowledge and language context. Therefore, we propose GALAX (Graph Augmented LAnguage model with eXplainability), an innovative framework that integrates pretrained Graph Neural Networks (GNNs) into Large Language Models (LLMs) via reinforcement guided by a Graph Process Reward Model (GPRM), which generates disease-relevant subgraphs in a step-wise manner initiated by an LLM and iteratively evaluated by a pretrained GNN, enabling process-level supervision without explicit intermediate reasoning annotations. As an application, we also introduced Target-QA, a benchmark combining CRISPR-identified targets, multi-omic profiles, and biomedical graph knowledge across diverse cancer cell lines, which enables GNN pretraining for supervising step-wise graph construction and supports long-context reasoning over text-numeric graphs (TNGs), providing a scalable and biologically grounded framework for explainable, reinforcement-guided subgraph reasoning toward reliable and interpretable target and pathway discovery in precision medicine.

---

## 28. Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting

**论文链接:** [http://arxiv.org/abs/2509.20911v1](http://arxiv.org/abs/2509.20911v1)

**作者:** Zinan Zheng, Yang Liu, Jia Li

**发布时间:** 2025-09-25

**备注:** NeurIPS 2025 main track

### GPT解析

### 总结

本文提出了一种通用的网格插值图网络(MIGN)，用于解决全局天气预报中不规则分布和动态变化的挑战，实现了对未观测位置的泛化。

### 背景

图神经网络在天气预报中显示出有前景的结果，这对人类活动如农业规划和极端天气准备至关重要。然而，大多数研究集中在有限和局部区域进行训练，忽视了更广泛区域的影响，限制了有效泛化的能力。

### 目的

研究实践中不规则分布和动态变化的全局天气预报，开发能够泛化到未观测位置的模型。

### 方法

提出了一种通用的网格插值图网络(MIGN)，包含两个关键设计：(1)使用规则网格插值网络学习空间不规则数据以对齐数据；(2)利用参数化球谐位置嵌入进一步增强空间泛化能力。

### 主要发现

在最新的观测数据集上进行的大量实验表明，MIGN显著优于现有的数据驱动模型，并且具有空间泛化能力，能够泛化到以前未见过的站点。

### 结论

MIGN通过网格插值和球谐位置嵌入解决了全局天气预报中不规则分布和动态变化的挑战，实现了对未观测位置的泛化。

### 翻译

图神经网络在天气预报中显示出有前景的结果，这对人类活动如农业规划和极端天气准备至关重要。然而，大多数研究集中在有限和局部区域进行训练，忽视了更广泛区域的影响，限制了它们有效泛化的能力。因此，在本文中，我们研究实践中不规则分布和动态变化的全局天气预报，需要模型能够泛化到未观测的位置。为解决这些挑战，我们提出了一种通用的网格插值图网络(MIGN)，用于建模不规则的气象站预测，包含两个关键设计：(1)使用规则网格插值网络学习空间不规则数据以对齐数据；(2)利用参数化球谐位置嵌入进一步增强空间泛化能力。在最新的观测数据集上进行的大量实验表明，MIGN显著优于现有的数据驱动模型。此外，我们证明MIGN具有空间泛化能力，能够泛化到以前未见过的站点。


### 论文摘要

Graph neural networks have shown promising results in weather forecasting, which is critical for human activity such as agriculture planning and extreme weather preparation. However, most studies focus on finite and local areas for training, overlooking the influence of broader areas and limiting their ability to generalize effectively. Thus, in this work, we study global weather forecasting that is irregularly distributed and dynamically varying in practice, requiring the model to generalize to unobserved locations. To address such challenges, we propose a general Mesh Interpolation Graph Network (MIGN) that models the irregular weather station forecasting, consisting of two key designs: (1) learning spatially irregular data with regular mesh interpolation network to align the data; (2) leveraging parametric spherical harmonics location embedding to further enhance spatial generalization ability. Extensive experiments on an up-to-date observation dataset show that MIGN significantly outperforms existing data-driven models. Besides, we show that MIGN has spatial generalization ability, and is capable of generalizing to previous unseen stations.

---

## 29. MolCluster: Integrating Graph Neural Network with Community Detection for Coarse-Grained Mapping

**论文链接:** [http://arxiv.org/abs/2509.20893v1](http://arxiv.org/abs/2509.20893v1)

**作者:** Zhixuan Zhong, Linbo Ma, Jian Jiang

**发布时间:** 2025-09-25

**备注:** 17 pages; 4 figures; 5 tables

### GPT解析

### 总结

MolCluster是一种无监督模型，结合图神经网络和社区检测算法，实现了无需标签的粗粒度表示提取，并支持可定制的分辨率，在MARTINI2数据集上表现优于传统方法和监督模型。

### 背景

传统的粗粒度（CG）建模通过将原子组映射为代表性单元来简化分子系统，但依赖固定映射规则，限制了处理多样化化学系统的能力且需要大量人工干预。基于监督学习的CG方法虽然更自动化和适应性强，但受限于标记数据集有限且无法控制映射分辨率的问题。

### 目的

开发一种无监督模型，克服传统CG方法和监督学习方法在处理多样化化学系统、控制映射分辨率方面的局限性，实现无需标签训练和可定制的分辨率。

### 方法

提出MolCluster，一种结合图神经网络和社区检测算法的无监督模型，用于提取粗粒度表示。引入预定义组对损失确保目标组的保留，采用二分策略实现不同分子系统上的精确、可定制分辨率。

### 主要发现

在MARTINI2数据集上的评估表明，MolCluster受益于其无标签预训练策略，在下游任务中表现优于传统聚类方法和监督模型。

### 结论

MolCluster作为可定制和化学一致的CG映射的核心模型具有潜力，能够有效解决传统CG方法和监督学习方法中的局限性。

### 翻译

粗粒度（CG）建模通过将原子组映射为代表性单元来简化分子系统。然而，传统的CG方法依赖固定的映射规则，这限制了它们处理多样化化学系统的能力，并需要大量人工干预。因此，已经提出了基于监督学习的CG方法，使映射更加自动化和适应性强。尽管如此，这些方法受限于标记数据集有限且无法控制映射分辨率，这对于多尺度建模至关重要。为了克服这些局限性，我们提出了MolCluster，这是一种无监督模型，结合了图神经网络和社区检测算法来提取CG表示。此外，预定义的组对损失确保了目标组的保留，而二分策略能够在不同分子系统上实现精确、可定制的分辨率。在下游任务方面，在MARTINI2数据集上的评估表明，MolCluster受益于其无标签预训练策略，在性能上优于传统聚类和监督模型。总体而言，这些结果突显了MolCluster作为可定制和化学一致的CG映射核心模型的潜力。


### 论文摘要

Coarse-grained (CG) modeling simplifies molecular systems by mapping groups of atoms into representative units. However, traditional CG approaches rely on fixed mapping rules, which limit their ability to handle diverse chemical systems and require extensive manual intervention. Thus, supervised learning-based CG methods have been proposed, enabling more automated and adaptable mapping. Nevertheless, these methods suffer from limited labeled datasets and the inability to control mapping resolution, which is essential for multiscale modeling. To overcome these limitations, we propose MolCluster, an unsupervised model that integrates a graph neural network and a community detection algorithm to extract CG representations. Additionally, a predefined group pair loss ensures the preservation of target groups, and a bisection strategy enables precise, customizable resolution across different molecular systems. In the case of the downstream task, evaluations on the MARTINI2 dataset demonstrate that MolCluster, benefiting from its label-free pretraining strategy, outperforms both traditional clustering and supervised models. Overall, these results highlight the potential of MolCluster as a core model for customizable and chemically consistent CG mapping.

---

## 30. Enhancing Molecular Property Prediction with Knowledge from Large Language Models

**论文链接:** [http://arxiv.org/abs/2509.20664v1](http://arxiv.org/abs/2509.20664v1)

**作者:** Peng Zhou, Lai Hou Tim, Zhixiang Cheng, Kun Xie, Chaoyi Li, Wei Liu, Xiangxiang Zeng

**发布时间:** 2025-09-25

**备注:** 9 pages, 5 figures

### GPT解析

### 总结

本研究提出了一种新型框架，将大型语言模型提取的知识与预训练分子模型的结构特征相结合，以增强分子性质预测(MPP)性能。

### 背景

分子性质预测是药物发现的关键环节。深度学习特别是图神经网络(GNNs)的发展使得从分子结构进行端到端学习成为可能，减少了对手动特征工程的依赖。然而，尽管GNNs和自监督学习方法取得了进展，但人类先验知识的整合仍然不可或缺，如最近利用大型语言模型进行知识提取的方法所示。尽管LLMs有优势，但它们存在知识空白和幻觉问题，尤其是对于研究较少的分子性质。

### 目的

提出一种新型框架，首次整合从LLMs提取的知识与从预训练分子模型衍生的结构特征，以增强分子性质预测(MPP)。

### 方法

该方法提示LLMs生成领域相关知识和分子向量化的可执行代码，产生基于知识的特征，随后与结构表示融合。研究使用了三种最先进的LLMs（GPT-4o、GPT-4.1和DeepSeek-R1）进行知识提取。

### 主要发现

广泛的实验证明，集成方法优于现有方法，确认了LLMs衍生的知识与结构信息的结合为MPP提供了强大而有效的解决方案。

### 结论

结合LLMs提取的知识和结构信息的方法在分子性质预测中表现优越，为药物发现提供了新的有效途径。

### 翻译

预测分子性质是药物发现的关键组成部分。深度学习的最新进展，特别是图神经网络(GNNs)，使得从分子结构进行端到端学习成为可能，减少了对手动特征工程的依赖。然而，尽管GNNs和自监督学习方法在分子性质预测(MPP)方面取得了进展，人类先验知识的整合仍然是不可或缺的，正如最近利用大型语言模型(LLMs)进行知识提取的方法所证明的那样。尽管LLMs具有优势，但它们受限于知识空白和幻觉问题，尤其是对于研究较少的分子性质。在这项工作中，我们提出了一个新型框架，首次将从LLMs提取的知识与从预训练分子模型衍生的结构特征相结合，以增强MPP。我们的方法提示LLMs生成领域相关知识和分子向量化的可执行代码，产生基于知识的特征，随后与结构表示融合。我们使用三种最先进的LLMs（GPT-4o、GPT-4.1和DeepSeek-R1）进行知识提取。广泛的实验证明我们的集成方法优于现有方法，确认了LLMs衍生的知识与结构信息的结合为MPP提供了强大而有效的解决方案。


### 论文摘要

Predicting molecular properties is a critical component of drug discovery. Recent advances in deep learning, particularly Graph Neural Networks (GNNs), have enabled end-to-end learning from molecular structures, reducing reliance on manual feature engineering. However, while GNNs and self-supervised learning approaches have advanced molecular property prediction (MPP), the integration of human prior knowledge remains indispensable, as evidenced by recent methods that leverage large language models (LLMs) for knowledge extraction. Despite their strengths, LLMs are constrained by knowledge gaps and hallucinations, particularly for less-studied molecular properties. In this work, we propose a novel framework that, for the first time, integrates knowledge extracted from LLMs with structural features derived from pre-trained molecular models to enhance MPP. Our approach prompts LLMs to generate both domain-relevant knowledge and executable code for molecular vectorization, producing knowledge-based features that are subsequently fused with structural representations. We employ three state-of-the-art LLMs, GPT-4o, GPT-4.1, and DeepSeek-R1, for knowledge extraction. Extensive experiments demonstrate that our integrated method outperforms existing approaches, confirming that the combination of LLM-derived knowledge and structural information provides a robust and effective solution for MPP.

---

## 31. Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding

**论文链接:** [http://arxiv.org/abs/2509.21223v1](http://arxiv.org/abs/2509.21223v1)

**作者:** Muxin Pu, Mei Kuan Lim, Chun Yong Chong, Chen Change Loy

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了名为Sigma的统一骨架SLU框架，通过标志感知早期融合机制、分层对齐学习策略和统一预训练框架解决了当前手语理解方法面临的三个关键限制：弱语义基础、局部细节与全局上下文不平衡以及低效跨模态学习。

### 背景

预训练在手语理解(SLU)任务中被证明是有效的，基于骨架的方法因其能稳健处理主体和背景变化而不受外观或环境因素影响而日益受到关注。

### 目的

解决当前SLU方法面临的三个关键限制：1)弱语义基础，模型难以将骨骼数据的运动模式与语言意义联系起来；2)局部细节与全局上下文之间的不平衡；3)低效的跨模态学习，难以构建跨模态语义对齐表示。

### 方法

提出Sigma框架，包含：1)标志感知早期融合机制促进视觉与文本模态深度交互；2)分层对齐学习策略同时捕获细粒度细节和高级语义关系；3)统一预训练框架结合对比学习、文本匹配和语言建模。

### 主要发现

Sigma在多个基准测试上取得最先进结果，包括孤立手语识别、连续手语识别和无词汇表手语翻译，证明了语义信息丰富预训练的影响和骨骼数据作为SLU独立解决方案的有效性。

### 结论

语义信息丰富的预训练和骨骼数据作为独立解决方案对SLU任务具有重要影响，Sigma框架有效解决了当前方法的局限性。

### 翻译

预训练已被证明对手语理解(SLU)任务中学习可迁移特征是有效的。最近，基于骨架的方法越来越受到关注，因为它们可以稳健地处理主体和背景的变化，而不受外观或环境因素的影响。当前的SLU方法仍然面临三个关键限制：1)弱语义基础，因为模型通常从骨骼数据中捕获低级运动模式，但难以将其与语言意义联系起来；2)局部细节和全局上下文之间的不平衡，模型要么过于关注细粒度线索，要么为了更广泛的上下文而忽略它们；3)低效的跨模态学习，因为在不同模态间构建语义对齐表示仍然很困难。为解决这些问题，我们提出了Sigma，一个统一的基于骨架的SLU框架，具有：1)标志感知的早期融合机制，促进视觉和文本模态之间的深度交互，用语言上下文丰富视觉特征；2)分层对齐学习策略，共同最大化来自不同模态的不同级别配对特征之间的一致性，有效捕获细粒度细节和高级语义关系；3)统一预训练框架，结合对比学习、文本匹配和语言建模，促进语义一致性和泛化能力。Sigma在多个基准测试上取得了新的最先进结果，包括孤立手语识别、连续手语识别和无词汇表手语翻译，这些测试涵盖了不同的手语和口语语言，证明了语义信息丰富预训练的影响以及骨骼数据作为SLU独立解决方案的有效性。


### 论文摘要

Pre-training has proven effective for learning transferable features in sign language understanding (SLU) tasks. Recently, skeleton-based methods have gained increasing attention because they can robustly handle variations in subjects and backgrounds without being affected by appearance or environmental factors. Current SLU methods continue to face three key limitations: 1) weak semantic grounding, as models often capture low-level motion patterns from skeletal data but struggle to relate them to linguistic meaning; 2) imbalance between local details and global context, with models either focusing too narrowly on fine-grained cues or overlooking them for broader context; and 3) inefficient cross-modal learning, as constructing semantically aligned representations across modalities remains difficult. To address these, we propose Sigma, a unified skeleton-based SLU framework featuring: 1) a sign-aware early fusion mechanism that facilitates deep interaction between visual and textual modalities, enriching visual features with linguistic context; 2) a hierarchical alignment learning strategy that jointly maximises agreements across different levels of paired features from different modalities, effectively capturing both fine-grained details and high-level semantic relationships; and 3) a unified pre-training framework that combines contrastive learning, text matching and language modelling to promote semantic consistency and generalisation. Sigma achieves new state-of-the-art results on isolated sign language recognition, continuous sign language recognition, and gloss-free sign language translation on multiple benchmarks spanning different sign and spoken languages, demonstrating the impact of semantically informative pre-training and the effectiveness of skeletal data as a stand-alone solution for SLU.

---

## 32. Adversarially Robust MIMO Physical Layer Authentication for Non-Stationary Channels

**论文链接:** [http://arxiv.org/abs/2509.21171v1](http://arxiv.org/abs/2509.21171v1)

**作者:** Ali Khandan Boroujeni, Ghazal Bagheri, Kuranage Roche Rayan Ranasinghe, Giuseppe Thadeu Freitas de Abreu, Stefan Köpsell, Rafael F. Schaefer

**发布时间:** 2025-09-25

**备注:** Submitted to an IEEE journal

### GPT解析

### 总结

提出了一种针对非平稳MIMO无线信道的抗干扰物理层认证框架，整合了序列贝叶斯决策、深度特征提取和生成对抗建模，有效处理了时间空间相关性、视距阻塞和动态欺骗策略等问题。

### 背景

传统物理层认证方法假设信道平稳或观测独立，难以应对实际无线环境中存在的时间与空间相关性、视距阻塞和动态欺骗策略等挑战。

### 目的

开发一种能够适应非平稳MIMO无线信道特性的鲁棒物理层认证框架，提高认证系统对各种欺骗攻击的抵抗能力。

### 方法

整合序列贝叶斯决策、通过对比学习进行深度特征提取，以及生成对抗建模来模拟自适应欺骗者；使用2状态和3状态隐马尔可夫模型结合移动平均在线适应对认证性能进行综合分析。

### 主要发现

通过闭式递归推导出的对数似然比、检测概率和稳态近似值表明，该方法相比经典序列认证方案具有显著的鲁棒性改进。

### 结论

所提出的AR-PLA框架能够有效处理非平稳MIMO信道中的复杂挑战，为无线通信系统提供更安全的物理层认证解决方案。

### 翻译

我们提出了一种针对非平稳多输入多输出(MIMO)无线信道的抗干扰物理层认证(AR-PLA)框架。该框架整合了序列贝叶斯决策、通过对比学习进行深度特征提取以及生成对抗建模来模拟自适应欺骗者。与假设信道平稳或观测独立性的传统方法不同，我们的方法明确考虑了时间与空间相关性、视距(LoS)阻塞和动态欺骗策略。同时，我们使用2状态和3状态隐马尔可夫模型(HMMs)结合移动平均在线适应对认证性能进行了全面分析，给出了对数似然比、检测概率和稳态近似值的闭式递归，这些结果表明与经典序列认证方案相比具有显著的鲁棒性提升。


### 论文摘要

We propose an adversarially robust physical layer authentication (AR-PLA) framework tailored for non-stationary multiple-input multiple-output (MIMO) wireless channels. The framework integrates sequential Bayesian decision-making, deep feature extraction via contrastive learning, and generative adversarial modeling to simulate adaptive spoofers. Unlike conventional methods that assume stationary channels or independent observations, our approach explicitly accounts for temporal and spatial correlations, line-of-sight (LoS) blockages, and dynamic spoofing strategies. A comprehensive analytical characterization of the authentication performance using both 2-state and 3-state hidden Markov models (HMMs) with moving-average online adaptation is also provided, with closed-form recursions for loglikelihood ratios, detection probabilities, and steady-state approximations, which demonstrate significant robustness improvement over classical sequential authentication schemes.

---

## 33. Retrieval over Classification: Integrating Relation Semantics for Multimodal Relation Extraction

**论文链接:** [http://arxiv.org/abs/2509.21151v1](http://arxiv.org/abs/2509.21151v1)

**作者:** Lei Hei, Tingjing Liao, Yingxin Pei, Yiyang Qi, Jiaqi Wang, Ruiting Li, Feiliang Ren

**发布时间:** 2025-09-25

**备注:** Accepted by EMNLP 2025 Main Conference

### GPT解析

### 总结

本文提出了ROC框架，将多模态关系抽取从分类任务转变为检索任务，解决了传统方法的两个主要局限性：忽略结构约束和缺乏语义表达能力。

### 背景

关系抽取(RE)旨在识别非结构化文本中实体间的语义关系。虽然最近研究将传统RE扩展到多模态场景，但大多数方法仍采用基于分类的范式，将关系表示为离散标签。

### 目的

解决传统多模态关系抽取方法的两个主要局限性：忽略结构约束和缺乏细粒度关系理解的语义表达能力。

### 方法

提出ROC框架，将多模态RE重新定义为关系语义驱动的检索任务。通过多模态编码器集成实体类型和位置信息，使用大型语言模型将关系标签扩展为自然语言描述，并通过基于语义相似性的对比学习对齐实体-关系对。

### 主要发现

ROC在基准数据集MNRE和MORE上取得了最先进的性能，表现出更强的鲁棒性和可解释性。

### 结论

ROC框架有效解决了传统分类范式的问题，在多模态关系抽取任务中表现优异。

### 翻译

关系抽取(RE)旨在识别非结构化文本中实体之间的语义关系。尽管最近的工作将传统RE扩展到多模态场景，但大多数方法仍然采用基于分类的范式，融合多模态特征，将关系表示为离散标签。这种范式有两个显著局限性：(1)它忽略了实体类型和位置线索等结构约束，(2)它缺乏对细粒度关系理解的语义表达能力。我们提出了ROC(检索优于分类)框架，这是一个将多模态RE重新定义为关系语义驱动的检索任务的新框架。ROC通过多模态编码器集成实体类型和位置信息，使用大型语言模型将关系标签扩展为自然语言描述，并通过基于语义相似性的对比学习对齐实体-关系对。实验表明，我们的方法在基准数据集MNRE和MORE上取得了最先进的性能，并表现出更强的鲁棒性和可解释性。


### 论文摘要

Relation extraction (RE) aims to identify semantic relations between entities in unstructured text. Although recent work extends traditional RE to multimodal scenarios, most approaches still adopt classification-based paradigms with fused multimodal features, representing relations as discrete labels. This paradigm has two significant limitations: (1) it overlooks structural constraints like entity types and positional cues, and (2) it lacks semantic expressiveness for fine-grained relation understanding. We propose \underline{R}etrieval \underline{O}ver \underline{C}lassification (ROC), a novel framework that reformulates multimodal RE as a retrieval task driven by relation semantics. ROC integrates entity type and positional information through a multimodal encoder, expands relation labels into natural language descriptions using a large language model, and aligns entity-relation pairs via semantic similarity-based contrastive learning. Experiments show that our method achieves state-of-the-art performance on the benchmark datasets MNRE and MORE and exhibits stronger robustness and interpretability.

---

## 34. SupCLAP: Controlling Optimization Trajectory Drift in Audio-Text Contrastive Learning with Support Vector Regularization

**论文链接:** [http://arxiv.org/abs/2509.21033v1](http://arxiv.org/abs/2509.21033v1)

**作者:** Jiehui Luo, Yuguo Yin, Yuxin Xie, Jinghan Ru, Xianwei Zhuang, Minghua He, Aofan Liu, Zihan Xiong, Dongchao Yang

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出支持向量正则化（SVR）方法，用于解决对比语言-音频预训练中负样本推力的垂直分量导致的优化轨迹漂移和训练不稳定问题。通过引入辅助支持向量控制垂直分量，并探索语义半径的无监督建模策略，该方法在多个任务上超越了现有基线。

### 背景

对比语言-音频预训练旨在将多模态表示统一到共享嵌入空间中，是构建从跨模态检索到前沿多模态大语言模型等各种应用的基础。

### 目的

解决对比学习中负样本推力的垂直分量带来的优化轨迹漂移和训练不稳定问题，同时保留负样本中的丰富信息。

### 方法

提出支持向量正则化（SVR）方法，引入辅助支持向量控制推力的垂直分量；探索语义半径的无监督建模策略，包括直接参数化和带有约束的自适应半径预测器模块。

### 主要发现

SVR方法在标准音频-文本数据集上的分类、单语检索和多语检索任务中超越了InfoNCE和SigLIP损失等广泛使用的基线；关于优化轨迹漂移的理论分析和实验结果验证了SVR方法的正确性和有效性。

### 结论

支持向量正则化方法能够有效控制对比学习中负样本推力的垂直分量，减轻优化轨迹漂移，提高训练稳定性，同时保留负样本中的丰富信息，在各种多模态任务上表现优异。

### 翻译

对比语言-音频预训练旨在将多模态表示统一到共享嵌入空间中，是构建从跨模态检索到前沿多模态大语言模型等各种应用的基础。然而，我们发现对比学习中负样本推力的垂直分量是一把双刃剑：它包含来自负样本的丰富补充信息，但其无约束的性质会导致优化轨迹漂移和训练不稳定。为此，我们提出支持向量正则化（SVR）方法，通过引入辅助支持向量来控制这个垂直分量，旨在利用其丰富信息同时减轻相关的轨迹漂移。SVR的功效由其语义半径决定，我们探索了两种无监督建模策略：直接参数化和带有约束的自适应半径预测器模块，以提高其预测准确性。大量实验结果表明，在标准音频-文本数据集上的分类、单语检索和多语检索任务中，我们的方法超越了InfoNCE和SigLIP损失等广泛使用的基线。关于优化轨迹漂移的理论分析和实验结果都验证了SVR方法的正确性和有效性。


### 论文摘要

Contrastive language-audio pretraining, which aims to unify multimodal representations in a shared embedding space, serves as a cornerstone for building a wide range of applications, from cross-modal retrieval to cutting-edge multimodal large language models. However, we find that the perpendicular component of the pushing force from negative samples in contrastive learning is a double-edged sword: it contains rich supplementary information from negative samples, yet its unconstrained nature causes optimization trajectory drift and training instability. To address this, we propose Support Vector Regularization (SVR), a method that introduces an auxiliary support vector to control this perpendicular component, aiming to harness its rich information while mitigating the associated trajectory drift. The efficacy of SVR is critically governed by its semantic radius, for which we explore two unsupervised modeling strategies: direct parameterization and an adaptive radius predictor module enhanced with constraints to improve its predicting accuracy. Extensive experimental results demonstrate that our method surpasses widely used baselines like InfoNCE and SigLIP loss across classification, monolingual retrieval, and multilingual retrieval on standard audio-text datasets. Both the theoretical analysis and the experimental results on optimizing trajectory drift validate the correctness and effectiveness of our SVR method.

---

## 35. FlowXpert: Context-Aware Flow Embedding for Enhanced Traffic Detection in IoT Network

**论文链接:** [http://arxiv.org/abs/2509.20861v1](http://arxiv.org/abs/2509.20861v1)

**作者:** Chao Zha, Haolin Pan, Bing Bai, Jiangxing Wu, Ruyun Zhang

**发布时间:** 2025-09-25

### GPT解析

### 总结

针对物联网环境中复杂动态网络流量的检测挑战，本研究提出了一种新型特征提取工具和嵌入训练框架，通过消除传统时间和长度特征，采用上下文感知语义特征，并结合DBSCAN聚类算法和对比学习策略，有效提高了检测准确性、鲁棒性和泛化能力。

### 背景

物联网环境中大量设备持续交互产生复杂动态的网络流量，对基于规则的检测方法构成重大挑战。基于机器学习的流量检测技术能够识别异常模式和潜在威胁，是确保网络安全的关键组件。

### 目的

解决现有特征提取工具使用时间和长度相关特征导致高稀疏性影响模型收敛的问题，以及现有流量检测方法缺乏高效捕获网络流量语义特征的嵌入机制的挑战。

### 方法

提出新型特征提取工具，消除传统时间和长度特征，采用与源主机相关的上下文感知语义特征；设计集成无监督DBSCAN聚类算法和对比学习策略的嵌入训练框架，有效捕获流量的细粒度语义表示。

### 主要发现

在真实Mawi数据集上的评估表明，与几种最先进模型的比较实验证明了所提出方法的优越性能，且该方法在实际实时场景中具有良好的适用性和可部署性。

### 结论

提出的特征提取工具和嵌入训练框架在物联网网络流量检测方面表现优越，具有实际应用价值。

### 翻译

在物联网环境中，大量设备的持续交互产生复杂动态的网络流量，这对基于规则的检测方法构成了重大挑战。基于机器学习的流量检测技术能够识别此流量中的异常模式和潜在威胁，是确保网络安全的关健组件。本研究首先确定了广泛采用的特征提取工具的一个显著问题：大量使用时间和长度相关特征导致高稀疏性，这 adversely 影响模型收敛。此外，现有流量检测方法通常缺乏能够高效全面捕获网络流量语义特征的嵌入机制。为解决这些挑战，我们提出了一种新型特征提取工具，它消除了传统的时间和长度特征，转而采用与源主机相关的上下文感知语义特征，从而提高模型的泛化能力。此外，我们设计了一个嵌入训练框架，将无监督DBSCAN聚类算法与对比学习策略相结合，以有效捕获流量的细粒度语义表示。我们在真实的Mawi数据集上进行了大量经验评估，以验证所提出方法在检测准确性、鲁棒性和泛化方面的有效性。与几种最先进模型的比较实验证明了我们方法的优越性能。此外，我们确认了它在实时场景中的适用性和可部署性。


### 论文摘要

In the Internet of Things (IoT) environment, continuous interaction among a large number of devices generates complex and dynamic network traffic, which poses significant challenges to rule-based detection approaches. Machine learning (ML)-based traffic detection technology, capable of identifying anomalous patterns and potential threats within this traffic, serves as a critical component in ensuring network security. This study first identifies a significant issue with widely adopted feature extraction tools (e.g., CICMeterFlow): the extensive use of time- and length-related features leads to high sparsity, which adversely affects model convergence. Furthermore, existing traffic detection methods generally lack an embedding mechanism capable of efficiently and comprehensively capturing the semantic characteristics of network traffic. To address these challenges, we propose a novel feature extraction tool that eliminates traditional time and length features in favor of context-aware semantic features related to the source host, thus improving the generalizability of the model. In addition, we design an embedding training framework that integrates the unsupervised DBSCAN clustering algorithm with a contrastive learning strategy to effectively capture fine-grained semantic representations of traffic. Extensive empirical evaluations are conducted on the real-world Mawi data set to validate the proposed method in terms of detection accuracy, robustness, and generalization. Comparative experiments against several state-of-the-art (SOTA) models demonstrate the superior performance of our approach. Furthermore, we confirm its applicability and deployability in real-time scenarios.

---

## 36. Revolutionizing Precise Low Back Pain Diagnosis via Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2509.20813v1](http://arxiv.org/abs/2509.20813v1)

**作者:** Thanh Binh Le, Hoang Nhat Khang Vo, Tan-Ha Mai, Trong Nhan Phan

**发布时间:** 2025-09-25

**备注:** 12 pages, 4 figures

### GPT解析

### 总结

本文提出了一种名为LumbarCLIP的新型多模态框架，利用对比语言-图像预训练技术将腰椎MRI扫描与相应的放射学描述对齐，实现了在下游分类任务上高达95.00%的准确率和94.75%的F1分数，为自动化肌肉骨骼诊断和临床决策支持提供了有前景的基础。

### 背景

腰背痛影响全球数百万人，这推动了对能够联合分析复杂医学图像和相关文本报告的强大诊断模型的需求。

### 目的

开发一种能够将腰椎MRI扫描与相应的放射学描述对齐的多模态诊断框架。

### 方法

提出LumbarCLIP框架，基于包含轴向MRI视图与专家撰写的报告配对的数据集，集成视觉编码器（ResNet-50、Vision Transformer、Swin Transformer）与基于BERT的文本编码器提取密集表示，通过可学习的投影头将这些表示投影到共享嵌入空间，并使用软CLIP损失进行对比训练。

### 主要发现

模型在测试集上达到95.00%的准确率和94.75%的F1分数，消融研究表明线性投影头比非线性变体产生更有效的跨模态对齐。

### 结论

LumbarCLIP为自动化肌肉骨骼诊断和临床决策支持提供了有前景的基础。

### 翻译

腰背痛影响全球数百万人，推动了对能够联合分析复杂医学图像和相关文本报告的强大诊断模型的需求。我们提出了LumbarCLIP，一种新颖的多模态框架，利用对比语言-图像预训练来将腰椎MRI扫描与相应的放射学描述对齐。基于一个包含轴向MRI视图与专家撰写的报告配对的数据集构建，LumbarCLIP集成了视觉编码器（ResNet-50、Vision Transformer、Swin Transformer）与基于BERT的文本编码器来提取密集表示。这些表示通过可学习的投影头（可配置为线性或非线性）投影到共享的嵌入空间，并进行归一化以使用软CLIP损失促进稳定的对比训练。尽管存在固有的类别不平衡，我们的模型在下游分类任务上取得了最先进的性能，在测试集上达到高达95.00%的准确率和94.75%的F1分数。大量的消融研究表明，线性投影头比非线性变体产生更有效的跨模态对齐。LumbarCLIP为自动化肌肉骨骼诊断和临床决策支持提供了有前景的基础。


### 论文摘要

Low back pain affects millions worldwide, driving the need for robust diagnostic models that can jointly analyze complex medical images and accompanying text reports. We present LumbarCLIP, a novel multimodal framework that leverages contrastive language-image pretraining to align lumbar spine MRI scans with corresponding radiological descriptions. Built upon a curated dataset containing axial MRI views paired with expert-written reports, LumbarCLIP integrates vision encoders (ResNet-50, Vision Transformer, Swin Transformer) with a BERT-based text encoder to extract dense representations. These are projected into a shared embedding space via learnable projection heads, configurable as linear or non-linear, and normalized to facilitate stable contrastive training using a soft CLIP loss. Our model achieves state-of-the-art performance on downstream classification, reaching up to 95.00% accuracy and 94.75% F1-score on the test set, despite inherent class imbalance. Extensive ablation studies demonstrate that linear projection heads yield more effective cross-modal alignment than non-linear variants. LumbarCLIP offers a promising foundation for automated musculoskeletal diagnosis and clinical decision support.

---

## 37. CoSupFormer : A Contrastive Supervised learning approach for EEG signal Classification

**论文链接:** [http://arxiv.org/abs/2509.20489v1](http://arxiv.org/abs/2509.20489v1)

**作者:** D. Darankoum, C. Habermacher, J. Volle, S. Grudinin

**发布时间:** 2025-09-24

**备注:** 20 pages (14 pages Main text and 6 pages Supplementary Material)

### GPT解析

### 总结

该研究提出了一种新型端到端深度学习框架，用于处理脑电图信号中的多尺度信息，解决噪声和通道变化性问题。

### 背景

脑电图信号包含丰富的多尺度信息，对理解大脑状态至关重要，在诊断和药物开发领域有潜在应用。然而，从原始EEG信号中提取有意义特征同时处理噪声和通道变化性仍然是一个主要挑战。

### 目的

开发一种新颖的端到端深度学习框架来解决EEG信号处理中的挑战，通过几个关键创新来提高特征提取的可靠性。

### 方法

设计了一种能够捕获多尺度频率振荡的编码器；引入了基于注意力的编码器来学习EEG通道间和通道内局部区域的交互；集成了门控网络以动态过滤噪声通道；使用结合监督学习和对比学习的新型损失函数指导整个编码过程。

### 主要发现

该学习范式能从原始EEG信号中提取生物学上有意义的模式；可跨不同物种自主选择高质量通道；通过创新的架构和损失设计实现强大的泛化能力。

### 结论

该方法在多种应用中得到验证，从中枢神经系统障碍治疗的分类到帕金森病和阿尔茨海默病的诊断，结果表明能有效处理EEG数据的挑战。

### 翻译

脑电图信号包含丰富的多尺度信息，对理解大脑状态至关重要，在诊断和推进药物开发领域具有潜在应用。然而，从原始脑电图信号中提取有意义特征同时处理噪声和通道变化性仍然是一个主要挑战。这项工作提出了一种新颖的端到端深度学习框架，通过几个关键创新解决了这些问题。首先，我们设计了一种能够明确捕获多尺度频率振荡的编码器，覆盖不同脑电图相关任务的广泛特征。其次，为了建模复杂依赖关系并处理脑电图的高时间分辨率，我们引入了一种基于注意力的编码器，同时学习脑电图通道之间的相互作用和单个通道内的局部区域。我们在注意力编码器顶部集成了一个专门的门控网络，以动态过滤掉嘈杂和非信息性通道，提高脑电图数据的可靠性。整个编码过程由一种新颖的损失函数指导，该函数利用监督学习和对比学习，显著提高了模型泛化能力。我们在多种应用中验证了我们的方法，从中枢神经系统障碍治疗的分类到帕金森病和阿尔茨海默病的诊断。我们的结果表明，所提出的学习范式能够从原始脑电图信号中提取跨不同物种的生物学上有意义的模式，自主选择高质量通道，并通过创新的架构和损失设计实现强大的泛化能力。


### 论文摘要

Electroencephalography signals (EEGs) contain rich multi-scale information crucial for understanding brain states, with potential applications in diagnosing and advancing the drug development landscape. However, extracting meaningful features from raw EEG signals while handling noise and channel variability remains a major challenge. This work proposes a novel end-to-end deep-learning framework that addresses these issues through several key innovations. First, we designed an encoder capable of explicitly capturing multi-scale frequency oscillations covering a wide range of features for different EEG-related tasks. Secondly, to model complex dependencies and handle the high temporal resolution of EEGs, we introduced an attention-based encoder that simultaneously learns interactions across EEG channels and within localized {\em patches} of individual channels. We integrated a dedicated gating network on top of the attention encoder to dynamically filter out noisy and non-informative channels, enhancing the reliability of EEG data. The entire encoding process is guided by a novel loss function, which leverages supervised and contrastive learning, significantly improving model generalization. We validated our approach in multiple applications, ranging from the classification of effects across multiple Central Nervous System (CNS) disorders treatments to the diagnosis of Parkinson's and Alzheimer's disease. Our results demonstrate that the proposed learning paradigm can extract biologically meaningful patterns from raw EEG signals across different species, autonomously select high-quality channels, and achieve robust generalization through innovative architectural and loss design.

---

## 38. A Contrastive Learning Framework for Breast Cancer Detection

**论文链接:** [http://arxiv.org/abs/2509.20474v1](http://arxiv.org/abs/2509.20474v1)

**作者:** Samia Saeed, Khuram Naveed

**发布时间:** 2025-09-24

### GPT解析

### 总结

该研究提出了一种基于对比学习(CL)框架的乳腺癌检测方法，通过半监督学习在大量未标记数据上训练Resnet-50模型，并在小规模标记数据上调整，最终在基准数据集上实现了96.7%的检测准确率，优于现有方法。

### 背景

乳腺癌是全球第二大癌症死亡原因，占所有癌症病例的四分之一。早期检测对降低死亡率至关重要，而非侵入性成像技术和计算机辅助检测系统为早期检测提供了可能。传统图像分析方法存在局限性，而深度学习方法因标记数据有限而面临准确性挑战。

### 目的

解决深度学习方法在乳腺癌检测中因标记数据有限而导致的准确性问题，开发一种能够在小规模标记数据上高效训练的模型，提高乳腺癌检测的准确性。

### 方法

研究引入了一种对比学习(CL)框架，采用半监督对比学习方法，使用相似度指数在大量未标记的乳腺X线照片数据上训练Resnet-50模型。研究中使用了各种数据增强和变换技术来提高方法的性能，最后在一小组标记数据上调整模型。

### 主要发现

在INbreast和MIAS基准数据集上，所提出的模型达到了96.7%的乳腺癌检测准确率，优于现有的最先进方法。

### 结论

通过对比学习框架和半监督学习方法，研究成功地解决了深度学习在乳腺癌检测中因标记数据有限而导致的准确性问题，为早期乳腺癌检测提供了更有效的工具。

### 翻译

乳腺癌是全球第二大癌症相关死亡原因，占所有癌症病例的四分之一[1]。为了降低这一死亡率，早期检测肿瘤至关重要，因为早期检测显著改善治疗效果。非侵入性成像技术的进步使得通过计算机辅助检测(CAD)系统进行早期检测成为可能，这些系统依赖于传统的图像分析来识别恶性肿瘤。然而，由于深度学习方法在处理有限的大型标记训练数据集时往往面临准确性挑战，因此正逐渐转向深度学习方法。尽管有这种潜力，但我们的研究引入了一种对比学习(CL)框架，该框架在小型标记数据集上表现出色。在这方面，我们使用相似度指数在大量未标记的乳腺X线照片数据上以半监督对比学习方法训练Resnet-50。在这方面，我们使用了各种增强和变换技术，有助于提高我们方法的性能。最后，我们在一小部分标记数据上调整了我们的模型，优于现有的最先进方法。具体而言，我们在基准数据集INbreast和MIAS上观察到乳腺癌检测的准确率为96.7%。


### 论文摘要

Breast cancer, the second leading cause of cancer-related deaths globally, accounts for a quarter of all cancer cases [1]. To lower this death rate, it is crucial to detect tumors early, as early-stage detection significantly improves treatment outcomes. Advances in non-invasive imaging techniques have made early detection possible through computer-aided detection (CAD) systems which rely on traditional image analysis to identify malignancies. However, there is a growing shift towards deep learning methods due to their superior effectiveness. Despite their potential, deep learning methods often struggle with accuracy due to the limited availability of large-labeled datasets for training. To address this issue, our study introduces a Contrastive Learning (CL) framework, which excels with smaller labeled datasets. In this regard, we train Resnet-50 in semi supervised CL approach using similarity index on a large amount of unlabeled mammogram data. In this regard, we use various augmentation and transformations which help improve the performance of our approach. Finally, we tune our model on a small set of labelled data that outperforms the existing state of the art. Specifically, we observed a 96.7% accuracy in detecting breast cancer on benchmark datasets INbreast and MIAS.

---

## 39. Hunyuan3D-Omni: A Unified Framework for Controllable Generation of 3D Assets

**论文链接:** [http://arxiv.org/abs/2509.21245v1](http://arxiv.org/abs/2509.21245v1)

**作者:** Team Hunyuan3D, :, Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jingwei Huang, Junlin Yu, Kunhong Li, Linus, Penghao Wang, Qingxiang Lin, Sicong Liu, Xianghui Yang, Yixuan Tang, Yunfei Zhao, Zeqiang Lai, Zhihao Liang, Zibo Zhao

**发布时间:** 2025-09-25

**备注:** Technical Report; 3D Generation

### GPT解析

### 总结

本文提出了Hunyuan3D-Omni，一个统一的细粒度可控3D资产生成框架，通过多种条件信号实现精确控制，解决了现有方法依赖单一模态输入的限制。

### 背景

3D原生生成模型的最新进展加速了游戏、电影和设计中的资产创建，但大多数方法主要依赖图像或文本条件，缺乏细粒度的跨模态控制，限制了可控性和实际应用。

### 目的

解决现有3D生成方法在细粒度、跨模态控制方面的不足，开发一个统一的框架，实现对几何、拓扑和姿态的精确控制。

### 方法

基于Hunyuan3D 2.1构建，接受点云、体素、边界框和骨骼姿态先验等多种条件信号。采用统一的跨模态架构而非为每种模态使用单独的头部。使用渐进式、难度感知的采样策略，偏向于处理更难的信号，同时降低较容易信号的权重，鼓励稳健的多模态融合和对缺失输入的优雅处理。

### 主要发现

额外的控制提高了生成准确性，实现了几何感知的转换，并增强了生产工作流程的稳健性。

### 结论

Hunyuan3D-Omni通过统一的跨模态架构和渐进式训练策略，解决了现有3D生成模型在细粒度控制方面的局限性，为游戏、电影和设计行业提供了更强大的3D资产生成工具。

### 翻译

3D原生生成模型的最新进展加速了游戏、电影和设计中的资产创建。然而，大多数方法仍然主要依赖图像或文本条件，缺乏细粒度的跨模态控制，这限制了可控性和实际应用。为解决这一差距，我们提出了Hunyuan3D-Omni，一个基于Hunyuan3D 2.1构建的、用于细粒度可控3D资产生成的统一框架。除了图像外，Hunyuan3D-Omni还接受点云、体素、边界框和骨骼姿态先验作为条件信号，能够对几何、拓扑和姿态进行精确控制。我们的模型采用统一的跨模态架构，而非为每种模态使用单独的头部。我们采用渐进式、难度感知的采样策略，为每个示例选择一种控制模态，偏向于采样更难的信号(如骨骼姿态)，同时降低较容易信号的权重(如点云)，鼓励稳健的多模态融合和对缺失输入的优雅处理。实验表明，这些额外的控制提高了生成准确性，实现了几何感知的转换，并增强了生产工作流程的稳健性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有3D生成模型过度依赖图像或文本条件输入，缺乏细粒度跨模态控制能力的问题。这个问题在现实中很重要，因为缺乏控制性导致生成的3D资产在几何精度、拓扑结构和姿态控制上不够准确，限制了3D生成技术在游戏、电影和设计等领域的实际应用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于Hunyuan3D 2.1模型进行扩展，引入点云、体素、边界框和骨骼四种控制信号，设计了一个统一的控制编码器来处理这些不同信号。训练时采用渐进式、难度感知的采样策略，优先处理难度较高的信号。作者借鉴了PoseMaster的骨骼表示方法，参考了2D可控生成模型如ControlNet的思想，并采用了点云补全方法中的随机丢弃采样策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将多种控制信号统一表示为点云形式，通过一个统一的控制编码器处理不同类型的控制信号，并将控制特征与图像特征结合输入扩散模型实现可控3D生成。整体流程包括：1)图像通过DINO编码器提取特征；2)不同控制信号通过统一控制编码器处理；3)将图像特征和控制特征连接形成联合特征；4)输入扩散模型生成3D潜在表示；5)通过VAE解码器生成最终3D模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的多模态控制框架，首次支持四种控制信号；2)轻量级统一控制编码器；3)渐进式难度感知训练策略；4)最小化训练成本。相比之前工作，Hunyuan3D-Omni通过统一架构实现了多模态控制，减少了模型复杂度，能够优雅处理缺失输入，而之前的方法通常只支持单一条件或需要针对每种信号单独训练。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Hunyuan3D-Omni通过统一的控制框架和轻量级编码器，实现了基于点云、体素、边界框和骨骼的细粒度可控3D资产生成，显著提高了生成模型的控制性和实用性。'}


### 论文摘要

Recent advances in 3D-native generative models have accelerated asset creation for games, film, and design. However, most methods still rely primarily on image or text conditioning and lack fine-grained, cross-modal controls, which limits controllability and practical adoption. To address this gap, we present Hunyuan3D-Omni, a unified framework for fine-grained, controllable 3D asset generation built on Hunyuan3D 2.1. In addition to images, Hunyuan3D-Omni accepts point clouds, voxels, bounding boxes, and skeletal pose priors as conditioning signals, enabling precise control over geometry, topology, and pose. Instead of separate heads for each modality, our model unifies all signals in a single cross-modal architecture. We train with a progressive, difficulty-aware sampling strategy that selects one control modality per example and biases sampling toward harder signals (e.g., skeletal pose) while downweighting easier ones (e.g., point clouds), encouraging robust multi-modal fusion and graceful handling of missing inputs. Experiments show that these additional controls improve generation accuracy, enable geometry-aware transformations, and increase robustness for production workflows.

---

## 40. DAGDiff: Guiding Dual-Arm Grasp Diffusion to Stable and Collision-Free Grasps

**论文链接:** [http://arxiv.org/abs/2509.21145v1](http://arxiv.org/abs/2509.21145v1)

**作者:** Md Faizal Karim, Vignesh Vembar, Keshab Patra, Gaurav Singh, K Madhava Krishna

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了DAGDiff框架，一个端到端的系统，能够直接在SE(3) x SE(3)空间中去噪生成抓取对，有效解决双臂抓取中的稳定性、碰撞和泛化问题。

### 背景

可靠的双臂抓取对于操作大型和复杂物体至关重要，但由于稳定性、碰撞和泛化要求，这仍然是一个具有挑战性的问题。先前方法通常将任务分解为两个独立的抓取提议，依赖于区域先验或启发式方法，限制了泛化能力且无法提供稳定性的原则性保证。

### 目的

开发一个能够直接生成稳定且无碰撞的双臂抓取对的方法，不依赖于显式的区域检测或物体先验，提高抓取的泛化能力和可靠性。

### 方法

提出DAGDiff框架，通过分类器信号引导扩散过程，而非依赖区域检测或物体先验。该框架集成了几何、稳定性和碰撞感知的引导项，引导生成过程朝向物理有效且力封闭合规的抓取方向发展。

### 主要发现

通过分析力封闭检查、碰撞分析和大规模基于物理的模拟全面评估了DAGDiff，在各项指标上显示出比先前工作的一致性改进。框架能够直接在先前未见过的物体的真实世界点云上生成双臂抓取，并在异构双臂设置上可靠执行。

### 结论

DAGDiff提供了一个有效的端到端解决方案，能够生成稳定、无碰撞且具有良好泛化能力的双臂抓取，可直接应用于真实世界场景中的未知物体。

### 翻译

可靠的双臂抓取对于操作大型和复杂物体至关重要，但由于稳定性、碰撞和泛化要求，这仍然是一个具有挑战性的问题。先前方法通常将任务分解为两个独立的抓取提议，依赖于区域先验或启发式方法，这些方法限制了泛化能力且无法提供稳定性的原则性保证。我们提出了DAGDiff，一个端到端的框架，直接在SE(3) x SE(3)空间中去噪以生成抓取对。我们的主要见解是通过分类器信号引导扩散过程，可以更有效地强制执行稳定性和避免碰撞，而不是依赖显式区域检测或物体先验。为此，DAGDiff集成了几何、稳定性和碰撞感知的引导项，引导生成过程朝向物理有效且力封闭合规的抓取方向发展。我们通过分析力封闭检查、碰撞分析和大规模基于物理的模拟全面评估了DAGDiff，在这些指标上显示出比先前工作的一致性改进。最后，我们展示了该框架可以直接在先前未见过的物体的真实世界点云上生成双臂抓取，这些抓取在异构双臂设置上执行，两个机械臂可靠地抓取和提升它们。


### 论文摘要

Reliable dual-arm grasping is essential for manipulating large and complex objects but remains a challenging problem due to stability, collision, and generalization requirements. Prior methods typically decompose the task into two independent grasp proposals, relying on region priors or heuristics that limit generalization and provide no principled guarantee of stability. We propose DAGDiff, an end-to-end framework that directly denoises to grasp pairs in the SE(3) x SE(3) space. Our key insight is that stability and collision can be enforced more effectively by guiding the diffusion process with classifier signals, rather than relying on explicit region detection or object priors. To this end, DAGDiff integrates geometry-, stability-, and collision-aware guidance terms that steer the generative process toward grasps that are physically valid and force-closure compliant. We comprehensively evaluate DAGDiff through analytical force-closure checks, collision analysis, and large-scale physics-based simulations, showing consistent improvements over previous work on these metrics. Finally, we demonstrate that our framework generates dual-arm grasps directly on real-world point clouds of previously unseen objects, which are executed on a heterogeneous dual-arm setup where two manipulators reliably grasp and lift them.

---

## 41. CHARM: Control-point-based 3D Anime Hairstyle Auto-Regressive Modeling

**论文链接:** [http://arxiv.org/abs/2509.21114v1](http://arxiv.org/abs/2509.21114v1)

**作者:** Yuze He, Yanning Zhou, Wang Zhao, Jingwen Ye, Yushi Bai, Kaiwen Xiao, Yong-Jin Liu, Zhongqian Sun, Wei Yang

**发布时间:** 2025-09-25

**备注:** SIGGRAPH Asia 2025. 17 pages, 15 figures

### GPT解析

### 总结

本文介绍了CHARM，一种用于动漫发型建模的新型参数化表示和生成框架，解决了传统方法难以处理动漫发型高度风格化、分段结构几何形状的问题。

### 背景

传统头发建模方法专注于真实感头发，使用基于发丝或体积的表示方法，而动漫发型具有高度风格化和分段结构的几何特征，挑战了现有技术。现有工作通常依赖于密集网格建模或手工制作的样条曲线，使得它们在编辑方面效率低下，不适合可扩展的学习。

### 目的

开发一个高效、准确且可扩展的动漫发型建模解决方案，既能支持艺术家友好的设计，也能支持基于学习的生成。

### 方法

CHARM引入了一种紧凑的、可逆的基于控制点的参数化方法，每个发片由一系列控制点表示，每个点仅用五个几何参数编码。基于这种表示，CHARM引入了一个自回归生成框架，可以从输入图像或点云有效地生成动漫发型。通过将动漫发型解释为顺序的'头发语言'，自回归Transformer能够捕获局部几何形状和全局发型拓扑结构。

### 主要发现

研究团队构建了AnimeHair数据集，包含37K个高质量动漫发型，带有分离的发片和处理的网格数据。实验证明CHARM在重建准确性和生成质量方面都达到了最先进的性能。

### 结论

CHARM为动漫发型建模提供了一种具有表现力和可扩展性的解决方案，在重建准确性和生成质量方面都表现出色。

### 翻译

我们提出了CHARM，一种用于动漫发型建模的新型参数化表示和生成框架。虽然传统的头发建模方法专注于使用基于发丝或体积的表示方法来模拟真实头发，但动漫发型具有高度风格化、分段结构的几何特征，挑战了现有技术。现有工作通常依赖于密集网格建模或手工制作的样条曲线，使得它们在编辑方面效率低下，不适合可扩展的学习。CHARM引入了一种紧凑的、可逆的基于控制点的参数化方法，其中一系列控制点表示每个发片，每个点仅用五个几何参数编码。这种高效且准确的表示既支持艺术家友好的设计，也支持基于学习的生成。基于这种表示，CHARM引入了一个自回归生成框架，能够有效地从输入图像或点云生成动漫发型。通过将动漫发型解释为顺序的'头发语言'，我们的自回归Transformer捕获了局部几何形状和全局发型拓扑结构，实现了高保真度的动漫发型创建。项目页面：https://hyzcluster.github.io/charm/


### 论文摘要

We present CHARM, a novel parametric representation and generative framework for anime hairstyle modeling. While traditional hair modeling methods focus on realistic hair using strand-based or volumetric representations, anime hairstyle exhibits highly stylized, piecewise-structured geometry that challenges existing techniques. Existing works often rely on dense mesh modeling or hand-crafted spline curves, making them inefficient for editing and unsuitable for scalable learning. CHARM introduces a compact, invertible control-point-based parameterization, where a sequence of control points represents each hair card, and each point is encoded with only five geometric parameters. This efficient and accurate representation supports both artist-friendly design and learning-based generation. Built upon this representation, CHARM introduces an autoregressive generative framework that effectively generates anime hairstyles from input images or point clouds. By interpreting anime hairstyles as a sequential "hair language", our autoregressive transformer captures both local geometry and global hairstyle topology, resulting in high-fidelity anime hairstyle creation. To facilitate both training and evaluation of anime hairstyle generation, we construct AnimeHair, a large-scale dataset of 37K high-quality anime hairstyles with separated hair cards and processed mesh data. Extensive experiments demonstrate state-of-the-art performance of CHARM in both reconstruction accuracy and generation quality, offering an expressive and scalable solution for anime hairstyle modeling. Project page: https://hyzcluster.github.io/charm/

---

## 42. SeamCrafte: Enhancing Mesh Seam Generation for Artist UV Unwrapping via Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2509.20725v1](http://arxiv.org/abs/2509.20725v1)

**作者:** Duoteng Xu, Yuguang Chen, Jing Li, Xinhai Liu, Xueqi Ma, Zhuo Chen, Dongyu Zhang, Chunchao Guo

**发布时间:** 2025-09-25

### GPT解析

### 总结

该论文提出了SeamCrafter，一种基于GPT风格的自回归接缝生成器，用于解决3D表面UV参数化和纹理映射中的接缝放置问题。

### 背景

网格接缝在3D表面的UV参数化和纹理映射中起关键作用，不良接缝会导致严重的UV扭曲或过度碎片化，阻碍纹理合成并干扰艺术家工作流程。现有方法通常在两种失败模式间权衡：高扭曲或多分散碎片。

### 目的

开发一种能够同时减少UV扭曲和碎片化的接缝生成方法，避免现有方法的权衡问题。

### 方法

提出SeamCrafter，一种基于点云输入条件化的自回归GPT风格接缝生成器。采用双分支点云编码器在预训练中解耦并捕获拓扑和几何线索，使用直接偏好优化(DPO)在基于新接缝评估框架的偏好数据集上微调模型。

### 主要发现

SeamCrafter产生的接缝比先前方法的扭曲和碎片化程度显著降低，同时保持了拓扑一致性和视觉保真度。

### 结论

SeamCrafter是一种有效的接缝生成解决方案，能够平衡UV扭曲和碎片化问题，优于现有方法。

### 翻译

网格接缝在将3D表面分割用于UV参数化和纹理映射中起着关键作用。放置不当的接缝通常会导致严重的UV扭曲或过度碎片化，从而阻碍纹理合成并干扰艺术家工作流程。现有方法经常在一种失败模式和另一种失败模式之间权衡——要么产生高扭曲，要么产生许多分散的碎片。为了解决这个问题，我们引入了SeamCrafter，一种基于点云输入条件化的自回归GPT风格接缝生成器。SeamCrafter采用双分支点云编码器，在预训练过程中解耦并捕获互补的拓扑和几何线索。为了进一步提高接缝质量，我们在基于新型接缝评估框架派生的偏好数据集上使用直接偏好优化(DPO)对模型进行微调。该框架主要根据UV扭曲和碎片化评估接缝，并提供成对偏好标签来指导优化。大量实验表明，SeamCrafter产生的接缝比先前方法的扭曲和碎片化程度低得多，同时保持了拓扑一致性和视觉保真度。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D网格接缝生成的问题。接缝是3D表面上定义切割并展开成2D UV域的边缘，在UV参数化和纹理映射中起关键作用。接缝放置不当会导致UV失真或过度碎片化，影响纹理合成和艺术家工作流程。这个问题很重要，因为良好的接缝选择能实现忠实纹理对齐并最小化失真，支持高效的纹理合成和艺术家友好的编辑工作流程。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性：传统方法产生碎片化UV图谱，几何图像方法计算成本高，而神经方法如SeamGPT过度依赖拓扑线索。作者借鉴了SeamGPT的自回归方法、VecSet-based点云编码器和Direct Preference Optimization(DPO)方法，但创新性地设计了双分支编码器同时处理拓扑和几何信息，并使用基于偏好的优化来对齐人类判断。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是同时利用输入网格的拓扑和几何线索，并将预测与人类偏好对齐。流程分为三阶段：1)预训练阶段：将接衣表示为线段序列，用双分支编码器处理拓扑和几何点，用沙漏Transformer生成接缝；2)后训练阶段：用接缝评估系统构建偏好数据集，通过DPO优化失真和碎片化权衡；3)推理阶段：预测接缝坐标，映射到网格表面，标记为接缝并展开网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)双分支点云编码器解耦拓扑和几何信息；2)接缝评估框架提供成对偏好标签；3)使用DPO进行后训练对齐人类偏好。相比之前工作，SeamCrafter不会产生过度碎片化UV图谱，计算成本更低，鲁棒性更强，同时考虑拓扑和几何信息而非仅依赖拓扑，无需大量场景微调且能保持接缝语义连贯性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SeamCrafter通过结合双分支点云编码器和直接偏好优化，实现了能同时考虑拓扑和几何信息并符合人类偏好的高质量3D网格接缝生成，显著降低了UV失真和碎片化。'}


### 论文摘要

Mesh seams play a pivotal role in partitioning 3D surfaces for UV parametrization and texture mapping. Poorly placed seams often result in severe UV distortion or excessive fragmentation, thereby hindering texture synthesis and disrupting artist workflows. Existing methods frequently trade one failure mode for another-producing either high distortion or many scattered islands. To address this, we introduce SeamCrafter, an autoregressive GPT-style seam generator conditioned on point cloud inputs. SeamCrafter employs a dual-branch point-cloud encoder that disentangles and captures complementary topological and geometric cues during pretraining. To further enhance seam quality, we fine-tune the model using Direct Preference Optimization (DPO) on a preference dataset derived from a novel seam-evaluation framework. This framework assesses seams primarily by UV distortion and fragmentation, and provides pairwise preference labels to guide optimization. Extensive experiments demonstrate that SeamCrafter produces seams with substantially lower distortion and fragmentation than prior approaches, while preserving topological consistency and visual fidelity.

---

## 43. Reflect3r: Single-View 3D Stereo Reconstruction Aided by Mirror Reflections

**论文链接:** [http://arxiv.org/abs/2509.20607v1](http://arxiv.org/abs/2509.20607v1)

**作者:** Jing Wu, Zirui Wang, Iro Laina, Victor Adrian Prisacariu

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种利用镜面反射进行3D重建的方法，通过将反射视为辅助视图并设计虚拟相机变换，实现了从单图像的多视图立体设置，简化了成像过程，并与强大的前馈重建模型兼容。

### 背景

镜面反射在日常环境中很常见，可以在单次捕获中提供立体信息，因为真实和反射的虚拟视图同时可见。

### 目的

利用镜面反射的性质将其视为辅助视图，设计变换构建物理有效的虚拟相机，实现从单图像生成多视图立体设置，使其与强大的前馈重建模型兼容以实现通用且稳健的3D重建。

### 方法

将反射视为辅助视图并设计变换构建物理有效的虚拟相机，直接在像素域生成虚拟视图同时遵循真实世界成像过程；提出对称感知损失细化姿态估计以利用镜面引入的几何对称性；框架自然扩展到动态场景实现逐帧几何恢复。

### 主要发现

该方法简化了成像过程，实现了从单图像的多视图立体设置，与强大的前馈重建模型兼容，能够实现通用且稳健的3D重建，并有效处理动态场景。

### 结论

在真实世界数据和合成数据上的广泛实验证明了该方法的有效性，提供了一个完全可定制的16个Blender场景的合成数据集，每个场景都有真实的点云和相机姿态。

### 翻译

镜面反射在日常环境中很常见，可以在单次捕获中提供立体信息，因为真实和反射的虚拟视图同时可见。我们通过将反射视为辅助视图并设计一种变换来构建物理上有效的虚拟相机来利用这一特性，允许在像素域直接生成虚拟视图，同时遵循真实世界的成像过程。这使我们可以从单图像实现多视图立体设置，简化了成像过程，使其与强大的前馈重建模型兼容，实现通用且稳健的3D重建。为了进一步利用镜面引入的几何对称性，我们提出了一种对称感知损失来细化姿态估计。我们的框架还自然地扩展到动态场景，其中每一帧都包含镜面反射，实现高效的逐帧几何恢复。为了进行定量评估，我们提供了一个完全可定制的16个Blender场景的合成数据集，每个场景都有真实的点云和相机姿态。在真实世界数据和合成数据上进行了广泛的实验，以说明我们方法的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何利用镜面反射来辅助单视图3D重建的问题。这个问题重要是因为镜面反射在日常环境中很常见，可以在单次拍摄中同时提供真实场景和反射的虚拟视图，形成立体信息。利用这些反射可以简化成像过程，减少硬件需求，无需跨相机同步，还能自然地扩展到动态场景，实现高效的逐帧几何恢复。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到镜面反射可以提供多视图线索，形成极线几何，且真实-虚拟视图对共享相同的内部参数。早期基于镜子的方法只能处理简单形状且需要高度控制的环境，而前馈重建模型又缺乏对反射的感知。作者借鉴了DUSt3R框架作为主干网络，使用Detect Any Mirror进行镜子检测，并利用多视图立体几何原理。通过将反射重新解释为辅助视角，将单视图问题转化为多视图问题，设计了像素域操作来创建虚拟视图。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将镜面反射重新解释为辅助视角，使单视图图像形成立体设置，并利用镜子场景的对称性来优化重建。整体流程包括：1)镜子检测和多视图设置，检测反射区域并水平翻转模拟虚拟相机；2)初始预测，使用DUSt3R从虚拟-真实对生成初始点云；3)后优化，应用对称感知损失细化相机位姿；4)镜子平面恢复，从结果点云中恢复镜子平面信息。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)将镜面反射重新解释为辅助视角，实现单视图立体重建；2)设计像素域操作的多视图设置过程，创建符合物理成像的虚拟视图；3)利用对称性提出对称感知损失优化位姿；4)构建包含16个场景的完全可定制合成数据集。相比之前工作，本文方法不局限于简单形状或特定场景，能处理大角度差异的反射，且不将反射视为噪声而是有用信息，显著提高了重建质量和完整性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Reflect3r通过将镜面反射重新解释为虚拟相机捕获的辅助视图，成功地将单视图3D重建转化为多视图立体重建，显著提高了在含有镜子场景中的3D重建质量和完整性。'}


### 论文摘要

Mirror reflections are common in everyday environments and can provide stereo information within a single capture, as the real and reflected virtual views are visible simultaneously. We exploit this property by treating the reflection as an auxiliary view and designing a transformation that constructs a physically valid virtual camera, allowing direct pixel-domain generation of the virtual view while adhering to the real-world imaging process. This enables a multi-view stereo setup from a single image, simplifying the imaging process, making it compatible with powerful feed-forward reconstruction models for generalizable and robust 3D reconstruction. To further exploit the geometric symmetry introduced by mirrors, we propose a symmetric-aware loss to refine pose estimation. Our framework also naturally extends to dynamic scenes, where each frame contains a mirror reflection, enabling efficient per-frame geometry recovery. For quantitative evaluation, we provide a fully customizable synthetic dataset of 16 Blender scenes, each with ground-truth point clouds and camera poses. Extensive experiments on real-world data and synthetic data are conducted to illustrate the effectiveness of our method.

---

## 44. Fast Estimation of Wasserstein Distances via Regression on Sliced Wasserstein Distances

**论文链接:** [http://arxiv.org/abs/2509.20508v1](http://arxiv.org/abs/2509.20508v1)

**作者:** Khai Nguyen, Hai Nguyen, Nhat Ho

**发布时间:** 2025-09-24

**备注:** 35 pages, 20 figures, 4 tables

### GPT解析

### 总结

该研究提出了一种基于切片Wasserstein距离回归的快速估计方法，用于高效计算多对分布之间的Wasserstein距离。该方法结合标准SW距离和提升SW距离，通过两个线性模型实现精确预测，在多种任务和数据集上优于现有方法，并能加速Wasserstein Wormhole训练。

### 背景

Wasserstein距离是衡量分布间差异的重要工具，但计算多对分布之间的Wasserstein距离通常计算成本高昂。现有方法在低数据情况下表现不佳，需要更高效的计算方法。

### 目的

开发一种能够高效、准确估计多对分布之间Wasserstein距离的方法，特别是在数据有限的情况下，同时探索该方法在加速现有Wasserstein计算框架中的应用潜力。

### 方法

提出基于切片Wasserstein距离回归的估计方法，同时利用标准SW距离（提供下界）和提升SW距离（提供上界）作为预测因子。设计了两个线性模型：一个具有闭式最小二乘解的无约束模型，和一个参数减半的约束模型。模型可以从少量分布对中学习，然后通过SW距离的线性组合快速预测任意分布对的Wasserstein距离。

### 主要发现

在多种任务（高斯混合、点云分类、3D点云可视化）和数据集（MNIST点云、ShapeNetV2、MERFISH Cell Niches、scRNA-seq）上验证了该方法的有效性。结果表明，在低数据情况下，该方法比最先进的Wasserstein嵌入模型Wasserstein Wormhole提供更好的Wasserstein距离近似。此外，该估计器可以加速Wormhole训练，产生RG-Wormhole。

### 结论

该研究提出的方法显著提高了Wasserstein距离的计算效率，同时保持了准确性，特别是在数据有限的情况下。该方法不仅独立应用时表现优异，还能作为加速器改进现有Wasserstein计算框架，为分布间距离计算提供了新的实用工具。

### 翻译

我们解决了从元分布中抽取的多对分布之间高效计算Wasserstein距离的问题。为此，我们提出了一种基于对切片Wasserstein距离回归的快速估计方法。具体来说，我们利用标准SW距离（提供下界）和提升SW距离（提供上界）作为真实Wasserstein距离的预测因子。为确保简洁性，我们引入了两个线性模型：一个具有闭式最小二乘解的无约束模型，以及一个参数减半的约束模型。我们证明可以从少量分布对中学习准确的模型。一旦估计完成，模型可以通过SW距离的线性组合预测任意分布对的Wasserstein距离，使其具有很高的效率。我们在多种任务上经验性地验证了我们的方法，包括高斯混合、点云分类和3D点云的Wasserstein空间可视化。在MNIST点云、ShapeNetV2、MERFISH Cell Niches和scRNA-seq等各种数据集上，我们的方法始终比最先进的Wasserstein嵌入模型Wasserstein Wormhole更好地近似Wasserstein距离，特别是在低数据情况下。最后，我们证明我们的估计器也可以加速Wormhole训练，产生RG-Wormhole。


### 论文摘要

We address the problem of efficiently computing Wasserstein distances for multiple pairs of distributions drawn from a meta-distribution. To this end, we propose a fast estimation method based on regressing Wasserstein distance on sliced Wasserstein (SW) distances. Specifically, we leverage both standard SW distances, which provide lower bounds, and lifted SW distances, which provide upper bounds, as predictors of the true Wasserstein distance. To ensure parsimony, we introduce two linear models: an unconstrained model with a closed-form least-squares solution, and a constrained model that uses only half as many parameters. We show that accurate models can be learned from a small number of distribution pairs. Once estimated, the model can predict the Wasserstein distance for any pair of distributions via a linear combination of SW distances, making it highly efficient. Empirically, we validate our approach on diverse tasks, including Gaussian mixtures, point-cloud classification, and Wasserstein-space visualizations for 3D point clouds. Across various datasets such as MNIST point clouds, ShapeNetV2, MERFISH Cell Niches, and scRNA-seq, our method consistently provides a better approximation of Wasserstein distance than the state-of-the-art Wasserstein embedding model, Wasserstein Wormhole, particularly in low-data regimes. Finally, we demonstrate that our estimator can also accelerate Wormhole training, yielding \textit{RG-Wormhole}.

---

## 45. MOSS-ChatV: Reinforcement Learning with Process Reasoning Reward for Video Temporal Reasoning

**论文链接:** [http://arxiv.org/abs/2509.21113v1](http://arxiv.org/abs/2509.21113v1)

**作者:** Sicheng Tao, Jungang Li, Yibo Yan, Junyan Zhang, Yubo Gao, Hanqian Li, ShuHang Xun, Yuxuan Fan, Hong Chen, Jianxiang He, Xuming Hu

**发布时间:** 2025-09-25

### GPT解析

### 总结

MOSS-ChatV是一种基于强化学习的视频推理框架，通过动态时间规整(DTW)过程奖励解决了多模态大语言模型在视频推理中的过程不一致性问题，提高了模型的解释性和鲁棒性。

### 背景

视频推理已成为多模态大语言模型的关键能力，要求模型超越静态感知，对复杂场景中的时间动态进行连贯理解。然而现有MLLMs常表现出过程不一致性，即使最终答案正确，中间推理也会偏离视频动态。

### 目的

解决现有MLLMs在视频推理过程中的过程不一致性问题，提高模型的解释性和鲁棒性。

### 方法

引入MOSS-ChatV框架，使用基于动态时间规整的过程奖励对齐推理轨迹与时间锚定的参考，无需辅助奖励模型即可实现高效过程监督；构建MOSS-Video基准，带有注释的推理轨迹，训练集用于微调MOSS-ChatV，保留集用于评估。

### 主要发现

MOSS-ChatV在MOSS-Video测试集上达到87.2%的准确率，并在MVBench和MMVU等通用视频基准上提升性能；该框架在不同架构(包括Qwen2.5-VL和Phi-2)中均一致取得提升；使用GPT-4o作为评判者评估表明，MOSS-ChatV产生更一致和稳定的推理轨迹。

### 结论

MOSS-ChatV框架有效解决了MLLMs在视频推理中的过程不一致性问题，提高了模型的解释性和鲁棒性，具有广泛的适用性。

### 翻译

视频推理已成为多模态大语言模型的关键能力，要求模型超越静态感知，对复杂场景中的时间动态进行连贯理解。然而现有MLLMs常表现出过程不一致性，即使最终答案正确，中间推理也会偏离视频动态，损害了模型的解释性和鲁棒性。为解决这一问题，我们引入MOSS-ChatV，这是一个基于强化学习的框架，使用基于动态时间规整的过程奖励。这种基于规则的奖励将推理轨迹与时间上锚定的参考对齐，无需辅助奖励模型即可实现高效的过程监督。我们进一步将动态状态预测识别为视频推理的关键度量，并构建了MOSS-Video基准，带有注释的推理轨迹，其中训练集用于微调MOSS-ChatV，保留集用于评估。MOSS-ChatV在MOSS-Video(测试集)上达到87.2%的准确率，并在MVBench和MMVU等通用视频基准上提高了性能。该框架在不同架构(包括Qwen2.5-VL和Phi-2)中均一致地取得提升，证实了其广泛的适用性。使用GPT-4o作为评判者的进一步评估表明，MOSS-ChatV产生了更一致和稳定的推理轨迹。


### 论文摘要

Video reasoning has emerged as a critical capability for multimodal large language models (MLLMs), requiring models to move beyond static perception toward coherent understanding of temporal dynamics in complex scenes. Yet existing MLLMs often exhibit process inconsistency, where intermediate reasoning drifts from video dynamics even when the final answer is correct, undermining interpretability and robustness. To address this issue, we introduce MOSS-ChatV, a reinforcement learning framework with a Dynamic Time Warping (DTW)-based process reward. This rule-based reward aligns reasoning traces with temporally grounded references, enabling efficient process supervision without auxiliary reward models. We further identify dynamic state prediction as a key measure of video reasoning and construct MOSS-Video, a benchmark with annotated reasoning traces, where the training split is used to fine-tune MOSS-ChatV and the held-out split is reserved for evaluation. MOSS-ChatV achieves 87.2\% on MOSS-Video (test) and improves performance on general video benchmarks such as MVBench and MMVU. The framework consistently yields gains across different architectures, including Qwen2.5-VL and Phi-2, confirming its broad applicability. Evaluations with GPT-4o-as-judge further show that MOSS-ChatV produces more consistent and stable reasoning traces.

---

## 46. A sub-hourly spatio-temporal statistical model for solar irradiance in Ireland using open-source data

**论文链接:** [http://arxiv.org/abs/2509.21041v1](http://arxiv.org/abs/2509.21041v1)

**作者:** Maeve Upton, Eamonn Organ, Amanda Lenzi, James Sweeney

**发布时间:** 2025-09-25

### GPT解析

### 总结

该研究开发了一种新的贝叶斯时空建模框架，用于预测爱尔兰全境的小时和亚小时（10分钟）分辨率的太阳辐照度，该模型在预测精度、不确定性量化、可扩展性和实时实施能力方面表现出色。

### 背景

爱尔兰具有高度变化的海洋性气候，地面测量站分布稀疏，这使得选择合适的太阳辐照度数据集成为一个重大挑战。准确的太阳辐照度估计对可靠建模太阳能光伏（PV）发电量至关重要。

### 目的

开发一种新的贝叶斯时空建模框架，用于预测爱尔兰全境的小时和亚小时（10分钟）分辨率的太阳辐照度。

### 方法

使用贝叶斯时空建模框架进行太阳辐照度预测，通过交叉验证评估模型的统计鲁棒性，并与替代数据源（包括再分析数据集和最近站点插值）进行比较。

### 主要发现

模型在所有时间分辨率上都具有统计鲁棒性；小时分辨率显示最高的预测精度，而10分钟分辨率遇到更高的误差但更好的不确定性量化；与替代数据源相比，模型始终提供更优的特定站点准确性；在小时尺度上，模型优于ERA5与地面观测的一致性；在亚小时尺度上，10分钟分辨率估计与爱尔兰住宅和工业太阳能光伏装置的太阳能光伏电力输出一致。

### 结论

该模型不仅超越了现有数据集，还提供了完整的不确定性量化、可扩展性和实时实施能力，为太阳能预测和由于逆变器尺寸不足导致的过载削波损失估计提供了强大的工具。

### 翻译

准确的太阳辐照度估计对于可靠地建模太阳能光伏（PV）发电量至关重要。在爱尔兰高度变化的海洋性气候中，地面测量站分布稀疏，选择合适的太阳辐照度数据集是一个重大挑战。本研究引入了一种新的贝叶斯时空建模框架，用于预测爱尔兰全境小时和亚小时（10分钟）分辨率的太阳辐照度。交叉验证表明，我们的模型在所有时间分辨率上都具有统计鲁棒性，其中小时分辨率显示出最高的预测精度，而10分钟分辨率遇到更高的误差但更好的不确定性量化。在单独的评估中，我们将我们的模型与替代数据源（包括再分析数据集和最近站点插值）进行比较，发现它始终提供更优的特定站点准确性。在小时尺度上，我们的模型在符合地面观测方面优于ERA5。在亚小时尺度上，10分钟分辨率估计与爱尔兰住宅和工业太阳能光伏装置的太阳能光伏电力输出一致。除了超越现有数据集外，我们的模型提供了完整的不确定性量化、可扩展性和实时实施能力，为太阳能预测和由于逆变器尺寸不足导致的过载削波损失估计提供了强大的工具。


### 论文摘要

Accurate estimation of solar irradiance is essential for reliable modelling of solar photovoltaic (PV) power production. In Ireland's highly variable maritime climate, where ground-based measurement stations are sparsely distributed, selecting an appropriate solar irradiance dataset presents a significant challenge. This study introduces a novel Bayesian spatio-temporal modelling framework for predicting solar irradiance at hourly and sub-hourly (10-minute) resolutions across Ireland. Cross-validation demonstrates that our model is statistically robust across all temporal resolutions with hourly showing highest prediction precision whereas 10-minute resolution encounters higher errors but better uncertainty quantification. In separate evaluations, we compare our model against alternative data sources, including reanalysis datasets and nearest-station interpolation, and find that it consistently provides superior site-specific accuracy. At the hourly scale, our model outperforms ERA5 in agreement with ground-based observations. At the sub-hourly scale, 10-minute resolution estimates provide solar PV power outputs consistent with residential and industrial solar PV installations in Ireland. Beyond surpassing existing datasets, our model delivers full uncertainty quantification, scalability and the capacity for real-time implementation, offering a powerful tool for solar energy prediction and the estimation of losses due to overload clipping from inverter undersizing.

---

## 47. Nuclear Diffusion Models for Low-Rank Background Suppression in Videos

**论文链接:** [http://arxiv.org/abs/2509.20886v1](http://arxiv.org/abs/2509.20886v1)

**作者:** Tristan S. W. Stevens, Oisín Nolan, Jean-Luc Robert, Ruud J. G. van Sloun

**发布时间:** 2025-09-25

**备注:** 5 pages, 4 figures, preprint

### GPT解析

### 总结

提出了一种名为Nuclear Diffusion的混合框架，结合低秩时间建模与扩散后验采样，用于视频序列的去噪和修复，特别是在医学成像领域取得了改进的性能。

### 背景

视频序列通常包含结构化噪声和背景伪影，这些会掩盖动态内容，对准确分析和修复构成挑战。

### 目的

克服传统稳健主成分方法(RPCA)中稀疏性假设无法捕捉真实视频数据中丰富变异性的局限。

### 方法

开发了一种混合框架，整合低秩时间建模与扩散后验采样，称为Nuclear Diffusion方法，并在心脏超声去雾问题上进行了评估。

### 主要发现

与传统RPCA相比，Nuclear Diffusion在对比度增强(gCNR)和信号保留(KS统计)方面展示了改进的去雾性能。

### 结论

结合基于模型的时间模型与深度生成先验在高保真视频修复方面具有巨大潜力。

### 翻译

视频序列通常包含结构化噪声和背景伪影，这些会掩盖动态内容，对准确分析和修复构成挑战。稳健的主成分方法通过将数据分解为低秩和稀疏成分来解决此问题。然而，稀疏性假设通常无法捕捉真实视频数据中存在的丰富变异性。为了克服这一限制，提出了一种混合框架，该框架结合了低秩时间建模和扩散后验采样。提出的方法（Nuclear Diffusion）在真实世界的医学成像问题（即心脏超声去雾）上进行了评估，与传统RPCA相比，在对比度增强（gCNR）和信号保留（KS统计）方面展示了改进的去雾性能。这些结果突显了结合基于模型的时间模型与深度生成先验进行高保真视频修复的潜力。


### 论文摘要

Video sequences often contain structured noise and background artifacts that obscure dynamic content, posing challenges for accurate analysis and restoration. Robust principal component methods address this by decomposing data into low-rank and sparse components. Still, the sparsity assumption often fails to capture the rich variability present in real video data. To overcome this limitation, a hybrid framework that integrates low-rank temporal modeling with diffusion posterior sampling is proposed. The proposed method, Nuclear Diffusion, is evaluated on a real-world medical imaging problem, namely cardiac ultrasound dehazing, and demonstrates improved dehazing performance compared to traditional RPCA concerning contrast enhancement (gCNR) and signal preservation (KS statistic). These results highlight the potential of combining model-based temporal models with deep generative priors for high-fidelity video restoration.

---

## 48. MoCLIP-Lite: Efficient Video Recognition by Fusing CLIP with Motion Vectors

**论文链接:** [http://arxiv.org/abs/2509.17084v2](http://arxiv.org/abs/2509.17084v2)

**作者:** Binhua Huang, Ni Wang, Arjun Pakrashi, Soumyabrata Dev

**发布时间:** 2025-09-21

**备注:** 6 pages, 3 figures

### GPT解析

### 总结

该论文提出了MoCLIP-Lite，一种简单而强大的双流晚期融合框架，用于高效视频动作识别，结合了CLIP图像编码器和运动向量的优势，实现了89.2%的Top-1准确率。

### 背景

视频动作识别是计算机视觉的基础任务，但现有先进模型计算成本高且依赖大量视频预训练。同时，CLIP等视觉语言模型在静态图像上具有强大零样本能力，运动向量则从压缩视频流中提供高效时间信息。

### 目的

结合CLIP模型和运动向量的优势，开发一种高效的视频识别框架，减少计算成本并提高性能。

### 方法

MoCLIP-Lite将冻结的CLIP图像编码器特征与在原始运动向量上训练的轻量级监督网络特征相结合。融合过程中，两个主干网络均保持冻结状态，仅训练一个小型多层感知机(MLP)头部，确保高效率。

### 主要发现

在UCF101数据集上，该方法达到89.2%的Top-1准确率，显著优于零样本基线(65.0%)和仅使用运动向量的基线(66.5%)。

### 结论

该工作为视频理解提供了新的高效基线，有效连接了大型静态模型和动态低成本运动线索。

### 翻译

视频动作识别是计算机视觉的基础任务，但最先进的模型通常计算成本高且依赖大量视频预训练。同时，像CLIP这样的大规模视觉语言模型在静态图像上提供了强大的零样本能力，而运动向量(MV)则直接从压缩视频流中提供高效的时间信息。为了结合这些范式的优势，我们提出了MoCLIP-Lite，一种简单而强大的双流晚期融合框架，用于高效的视频识别。我们的方法将冻结的CLIP图像编码器的特征与在原始运动向量上训练的轻量级监督网络的特征相结合。在融合过程中，两个主干网络都被冻结，只训练一个微小的多层感知机(MLP)头部，确保极高的效率。在UCF101数据集上的全面实验表明，我们的方法实现了89.2%的Top-1准确率，显著优于强大的零样本(65.0%)和仅使用运动向量(66.5%)的基线。我们的工作为视频理解提供了一个新的、高效的基线，有效地连接了大型静态模型和动态、低成本的运动线索。我们的代码和模型可在https://github.com/microa/MoCLIP-Lite获取。


### 论文摘要

Video action recognition is a fundamental task in computer vision, but state-of-the-art models are often computationally expensive and rely on extensive video pre-training. In parallel, large-scale vision-language models like Contrastive Language-Image Pre-training (CLIP) offer powerful zero-shot capabilities on static images, while motion vectors (MV) provide highly efficient temporal information directly from compressed video streams. To synergize the strengths of these paradigms, we propose MoCLIP-Lite, a simple yet powerful two-stream late fusion framework for efficient video recognition. Our approach combines features from a frozen CLIP image encoder with features from a lightweight, supervised network trained on raw MV. During fusion, both backbones are frozen, and only a tiny Multi-Layer Perceptron (MLP) head is trained, ensuring extreme efficiency. Through comprehensive experiments on the UCF101 dataset, our method achieves a remarkable 89.2% Top-1 accuracy, significantly outperforming strong zero-shot (65.0%) and MV-only (66.5%) baselines. Our work provides a new, highly efficient baseline for video understanding that effectively bridges the gap between large static models and dynamic, low-cost motion cues. Our code and models are available at https://github.com/microa/MoCLIP-Lite.

---

## 49. SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines

**论文链接:** [http://arxiv.org/abs/2509.21320v1](http://arxiv.org/abs/2509.21320v1)

**作者:** Yizhou Wang, Chen Tang, Han Deng, Jiabei Xiao, Jiaqi Liu, Jianyu Wu, Jun Yao, Pengze Li, Encheng Su, Lintao Wang, Guohang Zhuang, Yuchen Ren, Ben Fei, Ming Hu, Xin Chen, Dongzhan Zhou, Junjun He, Xiangyu Yue, Zhenfei Yin, Jiamin Wu, Qihao Zheng, Yuhao Zhou, Huihui Xu, Chenglong Ma, Yan Lu, Wenlong Zhang, Chunfeng Song, Philip Torr, Shixiang Tang, Xinzhu Ma, Wanli Ouyang, Lei Bai

**发布时间:** 2025-09-25

**备注:** technical report

### GPT解析

### 总结

研究团队开发了一个科学推理基础模型，能够将自然语言与异构科学表示对齐，支持多种科学推理任务，并在多个方面优于专业系统。

### 背景

科学领域需要能够处理多种科学表示形式并具备复杂推理能力的系统，而现有专业系统可能覆盖范围有限，跨领域泛化能力不足或保真度不够。

### 目的

开发一个能够将自然语言与异构科学表示对齐的基础模型，支持多种科学推理任务，提高跨领域泛化能力，并增强保真度。

### 方法

在206B-token的多模态科学语料库上预训练，使用40M指令进行监督微调，采用退火冷启动自举技术引发长链思维，通过任务特定的奖励塑造进行强化学习，支持四种能力家族涵盖103项任务，并开源相关资源。

### 主要发现

该方法比专业系统具有更广泛的指令覆盖范围，提高了跨领域泛化能力，增强了保真度，且跨学科学习增强了迁移能力和下游可靠性。

### 结论

该科学推理基础模型能有效处理多种科学表示形式，支持广泛的科学推理任务，研究团队已开源模型、指令调优数据集和评估代码。

### 翻译

我们提出了一个科学推理基础模型，该模型将自然语言与异构科学表示对齐。该模型在包含科学文本、纯序列和序列-文本对的206B-token语料库上进行了预训练，然后通过40M指令的监督微调进行对齐，采用退火冷启动自举来引发长链思维，并通过任务特定的奖励塑造进行强化学习，从而培养深思熟虑的科学推理能力。它支持四种能力家族，涵盖工作流中的103项任务：(i)文本与科学格式之间的忠实转换，(ii)文本/知识提取，(iii)属性预测，(iv)属性分类，(v)无条件和条件序列生成与设计。与专业系统相比，我们的方法扩展了指令覆盖范围，提高了跨领域泛化能力，并增强了保真度。我们详细介绍了数据整理和训练过程，并证明跨学科学习增强了迁移能力和下游可靠性。该模型、指令调优数据集和评估代码已在https://huggingface.co/SciReason和https://github.com/open-sciencelab/SciReason开源。


### 论文摘要

We present a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations. The model is pretrained on a 206B-token corpus spanning scientific text, pure sequences, and sequence-text pairs, then aligned via SFT on 40M instructions, annealed cold-start bootstrapping to elicit long-form chain-of-thought, and reinforcement learning with task-specific reward shaping, which instills deliberate scientific reasoning. It supports four capability families, covering up to 103 tasks across workflows: (i) faithful translation between text and scientific formats, (ii) text/knowledge extraction, (iii) property prediction, (iv) property classification, (v) unconditional and conditional sequence generation and design. Compared with specialist systems, our approach broadens instruction coverage, improves cross-domain generalization, and enhances fidelity. We detail data curation and training and show that cross-discipline learning strengthens transfer and downstream reliability. The model, instruct tuning datasets and the evaluation code are open-sourced at https://huggingface.co/SciReason and https://github.com/open-sciencelab/SciReason.

---

## 50. A Sentinel-3 foundation model for ocean colour

**论文链接:** [http://arxiv.org/abs/2509.21273v1](http://arxiv.org/abs/2509.21273v1)

**作者:** Geoffrey Dawson, Remy Vandaele, Andrew Taylor, David Moffat, Helen Tamura-Wicks, Sarah Jackson, Rosie Lickorish, Paolo Fraccaro, Hywel Williams, Chunbo Luo, Anne Jones

**发布时间:** 2025-09-25

**备注:** 15 pages, 8 figures

### GPT解析

### 总结

本研究介绍了一种基于Prithvi-EO Vision Transformer架构的新型基础模型，该模型在Sentinel-3 OLCI数据上预训练，用于海洋科学应用。研究通过在两个下游海洋地球观测任务上微调来评估模型性能，结果表明该基础模型在海洋监测中具有显著效用，能够利用少量高质量标记数据并捕捉海洋颜色的详细空间模式。

### 背景

在海洋科学领域，标记数据通常稀少且收集成本高昂，这限制了人工智能应用的发展。基础模型(FMs)是在大量未标记数据上预训练的AI模型，有潜力改变这一状况。

### 目的

开发并评估一种新型基础模型，用于海洋科学应用，特别是在量化叶绿素浓度和改进海洋初级生产力估算方面的应用。

### 方法

使用Prithvi-EO Vision Transformer架构构建基础模型，在Sentinel-3海洋和陆地色彩仪器(OLCI)数据上预训练用于数据重建。然后通过在两个下游海洋地球观测任务上微调来评估模型：1)与当前用于量化叶绿素浓度的基线模型比较性能；2)评估改进基于遥感的海洋初级生产力估算的能力。

### 主要发现

自训练的基础模型在海洋监测中表现出显著效用，能够有效利用少量高质量标记数据，同时捕捉海洋颜色的详细空间模式并与点观测相匹配。

### 结论

新一代地理空间AI模型有潜力为海洋生态系统及其在全球气候过程中的作用提供更稳健、数据驱动的见解，为海洋科学研究提供了新的工具和方法。

### 翻译

人工智能(AI)基础模型(FMs)在大量未标记数据上预训练，有可能彻底改变海洋科学中的AI应用，因为标记数据通常稀少且收集成本高昂。在本工作中，我们描述了一种使用Prithvi-EO Vision Transformer架构的新型基础模型，该模型已在Sentinel-3海洋和陆地色彩仪器(OLCI)数据上预训练用于数据重建。我们通过在两个下游海洋地球观测任务上进行微调来评估模型。首先，我们评估了模型与当前用于量化叶绿素浓度的基线模型相比的性能。然后，我们评估了该基础模型改进基于遥感的海洋初级生产力估算的能力。我们的结果证明了自训练基础模型在海洋监测中的效用，特别是能够利用少量高质量标记数据并捕捉海洋颜色的详细空间模式，同时与点观测相匹配。我们得出结论，新一代地理空间AI模型有可能为海洋生态系统及其在全球气候过程中的作用提供更稳健、数据驱动的见解。


### 论文摘要

Artificial Intelligence (AI) Foundation models (FMs), pre-trained on massive unlabelled datasets, have the potential to drastically change AI applications in ocean science, where labelled data are often sparse and expensive to collect. In this work, we describe a new foundation model using the Prithvi-EO Vision Transformer architecture which has been pre-trained to reconstruct data from the Sentinel-3 Ocean and Land Colour Instrument (OLCI). We evaluate the model by fine-tuning on two downstream marine earth observation tasks. We first assess model performance compared to current baseline models used to quantify chlorophyll concentration. We then evaluate the FMs ability to refine remote sensing-based estimates of ocean primary production. Our results demonstrate the utility of self-trained FMs for marine monitoring, in particular for making use of small amounts of high quality labelled data and in capturing detailed spatial patterns of ocean colour whilst matching point observations. We conclude that this new generation of geospatial AI models has the potential to provide more robust, data-driven insights into ocean ecosystems and their role in global climate processes.

---

## 51. Dense Semantic Matching with VGGT Prior

**论文链接:** [http://arxiv.org/abs/2509.21263v1](http://arxiv.org/abs/2509.21263v1)

**作者:** Songlin Yang, Tianyi Wei, Yushi Lan, Zeqi Xiao, Anyi Rao, Xingang Pan

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了一种基于VGGT的语义匹配方法，解决了现有方法在几何模糊性和最近邻规则方面的局限性，实现了优越的几何感知能力、匹配可靠性和流形保持。

### 背景

语义匹配是计算机视觉中的基础任务，旨在建立同一类别实例之间的像素级对应关系。现有方法存在两个局限性：(i) 几何模糊性：依赖于2D基础模型特征往往无法区分对称结构，需要额外微调但缺乏泛化能力；(ii) 最近邻规则：逐像素匹配忽略了跨图像不可见性和流形保持。

### 目的

开发具有几何感知能力的像素描述符和整体密集对应机制，以改进语义匹配性能。

### 方法

受3D几何基础模型启发，利用VGGT的基于几何的特征和整体密集匹配能力。通过重用早期特征阶段、微调后期阶段和添加双向对应的语义头保留VGGT优势；并通过循环一致训练策略、合成数据增强和渐进式训练方案（带有混叠伪影缓解）使VGGT适应数据稀缺的语义匹配场景。

### 主要发现

大量实验表明，该方法实现了优越的几何感知能力、匹配可靠性和流形保持，性能优于以前的基线方法。

### 结论

所提出的方法有效解决了现有语义匹配方法的局限性，通过利用并适应VGGT的优势，显著提升了语义匹配的性能。

### 翻译

语义匹配旨在建立同一类别实例之间的像素级对应关系，是计算机视觉中的基础任务。现有方法存在两种局限性：(i) 几何模糊性：它们对2D基础模型特征（如Stable Diffusion、DINO）的依赖往往无法区分对称结构，需要额外微调但缺乏泛化能力；(ii) 最近邻规则：它们的逐像素匹配忽略了跨图像不可见性和流形保持。这些挑战需要具有几何感知能力的像素描述符和整体密集对应机制。受3D几何基础模型最新进展的启发，我们转向使用VGGT，它提供了基于几何的特征和整体密集匹配能力，很好地满足了这些需求。然而，直接迁移VGGT具有挑战性，因为它原本是为单个实例的跨视图几何匹配设计的，与跨实例语义匹配不匹配，并且受到密集语义注释稀缺的阻碍。为此，我们提出了一种方法：(i) 通过重用早期特征阶段、微调后期阶段和添加双向对应的语义头，保留VGGT的内在优势；(ii) 通过循环一致训练策略、合成数据增强和带有混叠伪影缓解的渐进式训练方案，使VGGT适应数据稀缺的语义匹配场景。大量实验证明，我们的方法实现了优越的几何感知能力、匹配可靠性和流形保持，优于以前的基线。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决语义匹配中的两个关键问题：几何歧义性（难以区分对称结构如左右眼）和最近邻规则限制（忽略跨图像不可见性和流形保持）。这个问题在计算机视觉中很重要，因为语义匹配是建立同类实例间像素级对应关系的基础技术，广泛应用于2D操作（如风格迁移）、3D分析（如变形）和机器人技术（如功能学习）等领域，现有方法的局限性严重影响了这些应用的效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到语义匹配本质上是3D问题，但现有方法主要使用2D特征，缺乏3D几何感知。他们发现3D几何基础模型VGGT提供基于几何的特征和整体密集匹配能力，与需求高度匹配。然而，直接迁移VGGT存在挑战：它原本设计用于单个实例的跨视图几何匹配，不完全符合跨实例语义匹配的目标；同时语义匹配面临密集标注数据稀缺问题。因此，作者设计架构调整（重用早期特征、微调后期特征、添加语义头）和循环一致训练策略来解决这些问题。该方法借鉴了VGGT的3D几何基础模型工作，同时也吸收了语义匹配领域的循环一致性和弱监督学习策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用VGGT的3D几何先验解决语义匹配中的几何歧义性问题，同时保持流形映射特性，并通过循环一致训练策略处理跨图像不可见性问题。整体流程包括：1)架构设计：重用VGGT早期块提取几何特征，微调后期块获取语义特征，添加语义匹配头预测双向采样网格和置信度图；2)训练策略：采用循环一致训练（匹配-重建一致性和误差-置信度相关性）、合成数据生成和渐进式训练（四个阶段逐步训练）；3)损失函数：包括监督损失、循环一致性损失、平滑损失和不确定性损失；4)推理过程：输入图像通过特征提取、主干细化和语义匹配头输出对应关系和置信度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将VGGT适应于密集语义匹配，利用其3D几何先验解决几何歧义并保持流形特性；2)提出循环一致训练策略，通过匹配-重建一致性和误差-置信度相关性解决跨图像不可见性；3)创建合成数据管道和渐进式训练方案，减少对密集标注数据的依赖。相比之前工作，该方法具有更强的几何感知能力（能更好区分对称结构）、保持流形结构（大多数现有方法无法做到）、处理非刚性变形（优于SpaceJAM的全局变换）、预测置信度图（解决跨图像不可见性问题）以及更高的训练效率（减少标注数据依赖）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过将VGGT的3D几何先验与创新的循环一致训练策略相结合，首次实现了具有强大几何感知能力、流形保持特性和可靠置信度预测的密集语义匹配，解决了现有方法在几何歧义性和跨图像不可见性方面的关键局限。'}


### 论文摘要

Semantic matching aims to establish pixel-level correspondences between instances of the same category and represents a fundamental task in computer vision. Existing approaches suffer from two limitations: (i) Geometric Ambiguity: Their reliance on 2D foundation model features (e.g., Stable Diffusion, DINO) often fails to disambiguate symmetric structures, requiring extra fine-tuning yet lacking generalization; (ii) Nearest-Neighbor Rule: Their pixel-wise matching ignores cross-image invisibility and neglects manifold preservation. These challenges call for geometry-aware pixel descriptors and holistic dense correspondence mechanisms. Inspired by recent advances in 3D geometric foundation models, we turn to VGGT, which provides geometry-grounded features and holistic dense matching capabilities well aligned with these needs. However, directly transferring VGGT is challenging, as it was originally designed for geometry matching within cross views of a single instance, misaligned with cross-instance semantic matching, and further hindered by the scarcity of dense semantic annotations. To address this, we propose an approach that (i) retains VGGT's intrinsic strengths by reusing early feature stages, fine-tuning later ones, and adding a semantic head for bidirectional correspondences; and (ii) adapts VGGT to the semantic matching scenario under data scarcity through cycle-consistent training strategy, synthetic data augmentation, and progressive training recipe with aliasing artifact mitigation. Extensive experiments demonstrate that our approach achieves superior geometry awareness, matching reliability, and manifold preservation, outperforming previous baselines.

---

## 52. Decipher-MR: A Vision-Language Foundation Model for 3D MRI Representations

**论文链接:** [http://arxiv.org/abs/2509.21249v1](http://arxiv.org/abs/2509.21249v1)

**作者:** Zhijian Yang, Noel DSouza, Istvan Megyeri, Xiaojian Xu, Amin Honarmandi Shandiz, Farzin Haddadpour, Krisztian Koos, Laszlo Rusko, Emanuele Valeriano, Bharadwaj Swaninathan, Lei Wu, Parminder Bhatia, Taha Kass-Hout, Erhan Bas

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了Decipher-MR，一个3D MRI特定的视觉语言基础模型，通过大规模数据集训练和模块化设计，实现了在多样化临床任务中的高效应用和性能提升。

### 背景

MRI是临床诊断和研究的关键医学成像方式，但其复杂性和异质性对自动化分析构成挑战。基础模型在自然语言和视觉任务中表现优异，但在MRI应用中受限于数据稀缺和狭窄的解剖学焦点。

### 目的

开发一个可扩展、可泛化的MRI基础模型，能够在广泛的临床和研究任务中有效应用，克服现有方法的局限性。

### 方法

构建了Decipher-MR模型，在包含200,000个MRI系列（来自22,000多项研究）的大规模数据集上训练，涵盖多样解剖区域、序列和病理。结合自监督视觉学习和报告引导的文本监督，采用模块化设计，支持轻量级任务特定解码器附加到冻结的预训练编码器。

### 主要发现

在疾病分类、人口统计预测、解剖定位和跨模态检索等多样化基准测试中，Decipher-MR展现出超越现有基础模型和特定任务方法的性能，证明了其有效性和通用性。

### 结论

Decipher-MR确立了作为基于MRI的AI的可扩展和多功能基础，能够促进临床和研究领域的高效发展，为MRI分析提供了新的可能性。

### 翻译

磁共振成像（MRI）是临床诊断和研究中的关键医学成像方式，但其复杂性和异质性对自动化分析构成挑战，特别是在可扩展和可泛化的机器学习应用方面。虽然基础模型已经革命化了自然语言和视觉任务，但由于数据稀缺和狭窄的解剖学焦点，它们在MRI中的应用仍然有限。在这项工作中，我们提出了Decipher-MR，这是一个3D MRI特定的视觉语言基础模型，在包含来自22,000多项研究的200,000个MRI系列的大规模数据集上训练，涵盖了多样的解剖区域、序列和病理。Decipher-MR将自监督视觉学习与报告引导的文本监督相结合，构建强大、可泛化的表示，实现跨广泛应用的有效适应。为了以最小的计算开销实现强大和多样的临床任务，Decipher-MR支持模块化设计，能够将轻量级、特定任务的解码器附加到冻结的预训练编码器上。在此设置下，我们在疾病分类、人口统计预测、解剖定位和跨模态检索等多种基准上评估了Decipher-MR，展示了与现有基础模型和特定任务方法相比的一致性能提升。我们的结果确立了Decipher-MR作为基于MRI的AI的可扩展和多功能基础，促进了临床和研究领域的高效发展。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决MRI（磁共振成像）分析的自动化挑战，特别是开发能够处理MRI数据复杂性和异质性的基础模型。这个问题很重要，因为MRI是临床诊断和研究中的关键医学成像方式，但其数据的复杂性和多样性限制了机器学习应用的可扩展性和泛化能力。传统方法依赖大量标记数据且难以跨不同扫描仪、数据源和临床任务泛化，影响诊断准确性、临床工作流程效率和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到MRI分析面临的挑战，特别是缺乏专门针对MRI的基础模型。他们分析了现有基础模型在自然语言处理和计算机视觉领域的成功应用，以及医学成像领域（如X光、CT）的基础模型经验。设计思路包括构建专门针对3D MRI的基础模型，使用大规模多样化数据集，结合自监督视觉学习和文本监督，采用模块化设计。他们借鉴了DINOv2的学生-教师自监督学习、CLIP的对比学习方法，以及PubMedBERT作为文本编码器基础，但专门针对MRI数据特点进行了优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是开发一个专门针对3D MRI的视觉-语言基础模型，通过大规模多样化MRI数据和相应放射学报告进行训练，结合自监督视觉学习和文本监督构建强大可泛化的表示，并采用模块化设计支持轻量级任务特定解码器的灵活适配。整体流程包括：1)收集大规模多样化MRI数据集；2)两阶段预训练（第一阶段独立预训练图像和文本编码器，第二阶段进行图像-报告对比预训练）；3)使用3D Vision Transformer作为图像编码器，PubMedBERT作为文本编码器；4)自定义数据增强和基于器官的采样策略；5)冻结预训练编码器，附加和微调轻量级任务特定解码器进行多种任务评估。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)专门针对3D MRI的基础模型；2)使用超过20万个MRI系列的大规模多样化数据集；3)结合自监督视觉学习和文本监督的两阶段预训练策略；4)支持冻结预训练编码器仅微调轻量级解码器的模块化设计；5)实现零样本跨模态检索能力。相比之前工作，不同之处在于：比现有MRI基础模型使用的数据集更大更多样化；专门针对MRI特性优化而非通用医学成像模型；结合图像自监督和文本监督比单一阶段方法更有效；模块化设计比端到端训练更高效；在跨模态检索等方面表现出色。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Decipher-MR通过结合大规模多样化3D MRI数据与两阶段预训练策略，创建了一个强大的视觉-语言基础模型，能够高效适配多种临床任务，显著提升了MRI分析的自动化能力和准确性。'}


### 论文摘要

Magnetic Resonance Imaging (MRI) is a critical medical imaging modality in clinical diagnosis and research, yet its complexity and heterogeneity pose challenges for automated analysis, particularly in scalable and generalizable machine learning applications. While foundation models have revolutionized natural language and vision tasks, their application to MRI remains limited due to data scarcity and narrow anatomical focus. In this work, we present Decipher-MR, a 3D MRI-specific vision-language foundation model trained on a large-scale dataset comprising 200,000 MRI series from over 22,000 studies spanning diverse anatomical regions, sequences, and pathologies. Decipher-MR integrates self-supervised vision learning with report-guided text supervision to build robust, generalizable representations, enabling effective adaptation across broad applications. To enable robust and diverse clinical tasks with minimal computational overhead, Decipher-MR supports a modular design that enables tuning of lightweight, task-specific decoders attached to a frozen pretrained encoder. Following this setting, we evaluate Decipher-MR across diverse benchmarks including disease classification, demographic prediction, anatomical localization, and cross-modal retrieval, demonstrating consistent performance gains over existing foundation models and task-specific approaches. Our results establish Decipher-MR as a scalable and versatile foundation for MRI-based AI, facilitating efficient development across clinical and research domains.

---

## 53. Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy

**论文链接:** [http://arxiv.org/abs/2509.21190v1](http://arxiv.org/abs/2509.21190v1)

**作者:** Tian Lan, Hao Duong Le, Jinbo Li, Wenjun He, Meng Wang, Chenghao Liu, Chen Zhang

**发布时间:** 2025-09-25

### GPT解析

### 总结

TimeRCD是一种基于相对上下文差异(RCD)范式的新型时间序列异常检测基础模型，通过检测相邻时间窗口之间的显著差异来识别异常，而非依赖重建目标。该模型使用标准Transformer架构实现，在大规模多样化合成语料库上预训练，实验证明其在零样本时间序列异常检测任务中显著优于现有模型。

### 背景

时间序列异常检测是一项关键任务，但在零样本方式下开发能泛化到未见数据的模型仍然是一个重大挑战。现有基础模型主要依赖重建目标，存在目标不匹配问题：难以识别细微异常，同时经常误判复杂正常模式，导致高假阴性和假阳性率。

### 目的

克服现有重建基础模型的局限性，开发一种能够在零样本方式下有效识别时间序列异常的新型基础模型。

### 方法

提出TimeRCD模型，基于新的相对上下文差异(RCD)预训练范式。模型通过检测相邻时间窗口之间的显著差异来训练识别异常，而非重建输入。使用标准Transformer架构实现，在大规模多样化合成语料库(具有令牌级异常标签)上进行预训练，提供丰富的监督信号。

### 主要发现

大量实验表明，TimeRCD在跨不同数据集的零样本时间序列异常检测任务中，显著优于现有通用和特定于异常的基础模型。结果验证了RCD范式的优越性。

### 结论

相对上下文差异(RCD)范式为构建健壮且可泛化的时间序列异常检测基础模型建立了新的有效路径。

### 翻译

时间序列异常检测(TSAD)是一项关键任务，但开发以零样本方式泛化到未见数据的模型仍然是一个重大挑战。现有的TSAD基础模型主要依赖基于重建的目标，存在基本目标不匹配问题：它们难以识别细微异常，同时经常误判复杂的正常模式，导致高假阴性和假阳性率。为克服这些限制，我们引入了TimeRCD，一种基于新预训练范式——相对上下文差异(RCD)的时间序列异常检测基础模型。TimeRCD不是学习重建输入，而是通过检测相邻时间窗口之间的显著差异来明确训练识别异常。这种使用标准Transformer架构实现的关系方法，使模型能够捕获重建方法经常忽略的异常指示性上下文变化。为支持这一范式，我们开发了一个大规模、多样化的合成语料库，具有令牌级异常标签，提供了有效预训练所需的丰富监督信号。大量实验表明，TimeRCD在跨不同数据集的零样本时间序列异常检测中，显著优于现有通用和特定于异常的基础模型。我们的结果验证了RCD范式的优越性，并为构建健壮且可泛化的时间序列异常检测基础模型建立了新的有效路径。


### 论文摘要

Time series anomaly detection (TSAD) is a critical task, but developing models that generalize to unseen data in a zero-shot manner remains a major challenge. Prevailing foundation models for TSAD predominantly rely on reconstruction-based objectives, which suffer from a fundamental objective mismatch: they struggle to identify subtle anomalies while often misinterpreting complex normal patterns, leading to high rates of false negatives and positives. To overcome these limitations, we introduce \texttt{TimeRCD}, a novel foundation model for TSAD built upon a new pre-training paradigm: Relative Context Discrepancy (RCD). Instead of learning to reconstruct inputs, \texttt{TimeRCD} is explicitly trained to identify anomalies by detecting significant discrepancies between adjacent time windows. This relational approach, implemented with a standard Transformer architecture, enables the model to capture contextual shifts indicative of anomalies that reconstruction-based methods often miss. To facilitate this paradigm, we develop a large-scale, diverse synthetic corpus with token-level anomaly labels, providing the rich supervisory signal necessary for effective pre-training. Extensive experiments demonstrate that \texttt{TimeRCD} significantly outperforms existing general-purpose and anomaly-specific foundation models in zero-shot TSAD across diverse datasets. Our results validate the superiority of the RCD paradigm and establish a new, effective path toward building robust and generalizable foundation models for time series anomaly detection.

---

## 54. Expanding Reasoning Potential in Foundation Model by Learning Diverse Chains of Thought Patterns

**论文链接:** [http://arxiv.org/abs/2509.21124v1](http://arxiv.org/abs/2509.21124v1)

**作者:** Xuemiao Zhang, Can Ren, Chengying Tu, Rongxiang Weng, Shuo Wang, Hongfei Yan, Jingang Wang, Xunliang Cai

**发布时间:** 2025-09-25

### GPT解析

### 总结

本研究提出了一种通过选择高价值思维链数据来提升大型推理模型数学推理能力的方法，仅使用10B-token精选数据即可显著提高模型在AIME测试中的表现。

### 背景

大型推理模型在数学推理领域的进展主要由强化学习驱动，中期训练中加入长思维链(CoT)数据能提高推理深度，但当前方法通常不加区分地使用所有CoT数据。

### 目的

确定哪些类型的数据最能有效增强模型推理能力，并开发一种方法来筛选和利用这些高价值数据。

### 方法

首次定义推理潜能为解决问题所需尝试次数的倒数；从CoT序列中抽象出原子推理模式构建核心参考集；提出双粒度算法选择与核心集一致的高价值CoT数据(CoTP)；使用这些数据训练模型。

### 主要发现

10B-token的CoTP数据使85A6B MoE模型在AIME 2024和2025测试中提高9.58%，将下游RL性能上限提高7.81%。

### 结论

通过有选择地使用富含高价值推理模式的数据，可以显著提高大型推理模型在数学推理任务上的性能，而无需使用大量训练数据。

### 翻译

近期在具有挑战性的数学推理领域大型推理模型的进展是由强化学习(RL)驱动的。在中期训练中加入长思维链(CoT)数据也被证明可以显著提高推理深度。然而，当前方法往往不加区分地使用CoT数据，关于哪种数据类型最有效地增强模型推理能力的关键问题仍未解决。在本文中，我们首次将基础模型的推理潜能定义为正确回答问题所需独立尝试次数的倒数，这与最终模型性能强相关。我们随后提出利用富含高价值推理模式的多样化数据来扩展推理潜能。具体而言，我们从CoT序列中抽象出具有共性和归纳能力的原子推理模式，并用它们构建富含有价值推理模式的核心参考集。此外，我们提出了一种涉及推理模式链和令牌熵的双粒度算法，从数据池中高效选择与核心集一致的高价值CoT数据(CoTP)，从而有效训练模型掌握推理能力。仅10B-token的CoTP数据就使85A6B Mixture-of-Experts (MoE)模型在具有挑战性的AIME 2024和2025上提高了9.58%，并将下游RL性能的上限提高了7.81%。


### 论文摘要

Recent progress in large reasoning models for challenging mathematical reasoning has been driven by reinforcement learning (RL). Incorporating long chain-of-thought (CoT) data during mid-training has also been shown to substantially improve reasoning depth. However, current approaches often utilize CoT data indiscriminately, leaving open the critical question of which data types most effectively enhance model reasoning capabilities. In this paper, we define the foundation model's reasoning potential for the first time as the inverse of the number of independent attempts required to correctly answer the question, which is strongly correlated with the final model performance. We then propose utilizing diverse data enriched with high-value reasoning patterns to expand the reasoning potential. Specifically, we abstract atomic reasoning patterns from CoT sequences, characterized by commonality and inductive capabilities, and use them to construct a core reference set enriched with valuable reasoning patterns. Furthermore, we propose a dual-granularity algorithm involving chains of reasoning patterns and token entropy, efficiently selecting high-value CoT data (CoTP) from the data pool that aligns with the core set, thereby training models to master reasoning effectively. Only 10B-token CoTP data enables the 85A6B Mixture-of-Experts (MoE) model to improve by 9.58% on the challenging AIME 2024 and 2025, and to raise the upper bound of downstream RL performance by 7.81%.

---

## 55. SoM-1K: A Thousand-Problem Benchmark Dataset for Strength of Materials

**论文链接:** [http://arxiv.org/abs/2509.21079v1](http://arxiv.org/abs/2509.21079v1)

**作者:** Qixin Wan, Zilong Wang, Jingwen Zhou, Wanting Wang, Ziheng Geng, Jiachen Liu, Ran Cao, Minghui Cheng, Lu Cheng

**发布时间:** 2025-09-25

### GPT解析

### 总结

研究团队创建了首个大规模多模态基准数据集SoM-1K，用于评估基础模型在材料力学问题上的性能，并提出图像描述(DoI)提示策略，发现当前基础模型在工程问题上表现不佳，但文本描述比直接图像输入更有效。

### 背景

基础模型已在多个领域展现出显著能力，但它们在复杂的多模态工程问题上的性能仍 largely 未被探索。特别是，当前基础模型在理解复杂视觉信息方面的能力有限。

### 目的

创建首个专门用于评估基础模型在材料力学问题上表现的大规模多模态基准数据集SoM-1K；提出一种新型提示策略(DoI)来改善基础模型对视觉信息的理解；评估多种基础模型在工程问题上的表现。

### 方法

构建包含1,065个带注释的材料力学问题的SoM-1K数据集，包含文本问题陈述和示意图；提出图像描述(DoI)策略，为视觉图表提供专家生成的严格文本描述作为上下文；评估八种代表性的基础模型，包括大型语言模型(LLMs)和视觉语言模型(VLMs)；进行详细的错误分析，比较DoI与直接图像输入的效果。

### 主要发现

当前基础模型在工程问题上表现显著不佳，最佳模型仅达到56.6%的准确率；当提供DoI时，大型语言模型通常比提供视觉图表的视觉语言模型表现更好；DoI在减轻视觉误解错误方面起着关键作用，表明准确的基于文本的描述对当前基础模型比直接图像输入更有效。

### 结论

这项工作为工程AI建立了严格的基准，并突显出基础模型(特别是在科学和工程背景下)需要开发更强大的多模态推理能力的迫切需求。

### 翻译

基础模型已在各个领域展现出显著能力，但它们在复杂的多模态工程问题上的性能 largely 未被探索。我们介绍了SoM-1K，这是首个专门用于评估基础模型在材料力学问题上表现的大规模多模态基准数据集。该数据集包含1,065个带注释的材料力学问题，通过包含文本问题陈述和示意图来模拟真实工程任务。由于当前基础模型在理解复杂视觉信息方面的能力有限，我们提出了一种名为图像描述(DoI)的新型提示策略，它提供了视觉图表的专家生成文本描述作为上下文。我们评估了八种代表性的基础模型，包括大型语言模型(LLMs)和视觉语言模型(VLMs)。我们的结果显示，当前基础模型在解决这些工程问题时遇到很大困难，表现最佳的模型仅达到56.6%的准确率。有趣的是，我们发现当提供DoI时，大型语言模型通常比提供视觉图表的视觉语言模型表现更好。详细的错误分析显示，DoI在减轻视觉误解错误方面起着关键作用，表明准确的基于文本的描述对当前基础模型比直接图像输入更有效。这项工作为工程AI建立了严格的基准，并突显出基础模型(特别是在科学和工程背景下)需要开发更强大的多模态推理能力的迫切需求。


### 论文摘要

Foundation models have shown remarkable capabilities in various domains, but their performance on complex, multimodal engineering problems remains largely unexplored. We introduce SoM-1K, the first large-scale multimodal benchmark dataset dedicated to evaluating foundation models on problems in the strength of materials (SoM). The dataset, which contains 1,065 annotated SoM problems, mirrors real-world engineering tasks by including both textual problem statements and schematic diagrams. Due to the limited capabilities of current foundation models in understanding complicated visual information, we propose a novel prompting strategy called Descriptions of Images (DoI), which provides rigorous expert-generated text descriptions of the visual diagrams as the context. We evaluate eight representative foundation models, including both large language models (LLMs) and vision language models (VLMs). Our results show that current foundation models struggle significantly with these engineering problems, with the best-performing model achieving only 56.6% accuracy. Interestingly, we found that LLMs, when provided with DoI, often outperform VLMs provided with visual diagrams. A detailed error analysis reveals that DoI plays a crucial role in mitigating visual misinterpretation errors, suggesting that accurate text-based descriptions can be more effective than direct image input for current foundation models. This work establishes a rigorous benchmark for engineering AI and highlights a critical need for developing more robust multimodal reasoning capabilities in foundation models, particularly in scientific and engineering contexts.

---

## 56. Measuring Audio's Impact on Correctness: Audio-Contribution-Aware Post-Training of Large Audio Language Models

**论文链接:** [http://arxiv.org/abs/2509.21060v1](http://arxiv.org/abs/2509.21060v1)

**作者:** Haolin He, Xingjian Du, Renhe Sun, Zheqi Dai, Yujia Xiao, Mingru Yang, Jiayi Zhou, Xiquan Li, Zhengxi Liu, Zining Liang, Chunyat Wu, Qianhua He, Tan Lee, Xie Chen, Weilong Zheng, Weiqiang Wang, Mark Plumbley, Jian Liu, Qiuqiang Kong

**发布时间:** 2025-09-25

### GPT解析

### 总结

这篇论文研究了大型音频语言模型(LALMs)的多阶段后训练方法，提出了新的数据集和训练范式，解决了零音频贡献问题，并在多个基准测试上取得了最先进性能。

### 背景

大型音频语言模型(LALMs)是多模态AI的重要前沿，能够处理多样化的音频任务。后训练可以显著提高基础模型的性能，但多阶段训练方法(如监督微调后跟强化学习)的效果不如单阶段强化学习。

### 目的

解决多阶段训练中数据分配问题，探索最大化LALMs能力的方法，并提供大规模高质量数据集支持相关研究。

### 方法

提出了AudioMCQ数据集，研究了零音频贡献现象，设计了音频贡献过滤方法，开发了弱到强和混合到强两种后训练范式。

### 主要发现

LALMs存在零音频贡献现象，模型可能仅从文本信息获取正确答案而不处理音频内容；通过音频贡献过滤和数据集划分，可以显著提升模型性能。

### 结论

通过提出的AudioMCQ数据集和两种训练范式，在DCASE 2025音频问答挑战中获得第一名，并在MMAU-test-mini、MMAU、MMAR和MMSU等多个基准测试上取得了最先进性能。

### 翻译

大型音频语言模型(LALMs)代表了多模态AI的重要前沿，能够处理多样化的音频任务。最近，LALMs的后训练受到越来越多的关注，因为它能显著提高基础模型的性能。虽然单阶段后训练如强化学习已经显示出有希望的结果，但多阶段方法如监督微调后跟强化学习仍然不够理想。跨多个训练阶段分配数据以最大化LALMs能力尚未得到充分探索，此类研究的大规模高质量数据集也缺乏。为解决这些问题，我们首先提出了AudioMCQ，一个包含57.1万样本的综合音频多选题数据集，具有两种思维链注释。其次，我们研究了LALMs中普遍存在的零音频贡献现象，即模型仅从文本信息中获取正确答案而不处理音频内容。我们提出了音频贡献过滤方法，将数据划分为弱音频贡献和强音频贡献子集。基于这些见解，我们开发了两种有效的后训练范式：弱到强(在弱音频贡献数据上进行监督微调，然后在强音频贡献数据上进行强化学习)和混合到强(在混合音频贡献数据上进行监督微调，然后在强音频贡献数据上进行强化学习)。通过使用AudioMCQ，我们在DCASE 2025音频问答挑战中获得第一名。此外，利用我们的数据集和不同的训练策略，我们在MMAU-test-mini上达到78.2%，在MMAU上达到75.6%，在MMAR上达到67.1%，在MMSU上达到70.7%，在这些基准测试上建立了新的最先进性能。


### 论文摘要

Large Audio Language Models (LALMs) represent an important frontier in multimodal AI, addressing diverse audio tasks. Recently, post-training of LALMs has received increasing attention due to significant performance improvements over foundation models. While single-stage post-training such as reinforcement learning (RL) has demonstrated promising results, multi-stage approaches such as supervised fine-tuning (SFT) followed by RL remain suboptimal. The allocation of data across multiple training stages to maximize LALM capabilities has not been fully explored, and large-scale, high-quality datasets for such research are also lacking. To address these problems, we firstly present AudioMCQ, a comprehensive audio multiple-choice question dataset comprising 571k samples with two kinds of chain-of-thought annotations. Secondly, we investigate the prevalent zero audio-contribution phenomenon in LALMs, where models derive correct answers solely from textual information without processing audio content. We propose Audio-Contribution Filtering to partition data into weak and strong audio-contribution subsets. Based on these insights, we develop two effective post-training paradigms: Weak-to-Strong (SFT on weak audio-contribution data followed by RL on strong audio-contribution data) and Mixed-to-Strong (SFT on mixed audio-contribution data followed by RL on strong audio-contribution data). We achieve first place in the DCASE 2025 Audio-Question-Answering challenge by using AudioMCQ. Additionally, leveraging our dataset with different training strategies, we achieve 78.2\% on MMAU-test-mini, 75.6\% on MMAU, 67.1\% on MMAR, and 70.7\% on MMSU, establishing new state-of-the-art performance across these benchmarks.

---

## 57. SiNGER: A Clearer Voice Distills Vision Transformers Further

**论文链接:** [http://arxiv.org/abs/2509.20986v1](http://arxiv.org/abs/2509.20986v1)

**作者:** Geunhyeok Yu, Sunjae Jeong, Yoonyoung Choi, Jaeseung Kim, Hyoseok Hwang

**发布时间:** 2025-09-25

**备注:** Main paper: 12 pages (including 3 pages of references), 6 figures, 6  tables. Appendix: 9 pages, 7 figures

### GPT解析

### 总结

本文提出了Singular Nullspace-Guided Energy Reallocation (SiNGER)框架，解决了Vision Transformers在知识蒸馏中的高范数伪影问题，通过教师特征精炼和零空间引导的扰动，实现了抑制伪影同时保留信息信号的效果。

### 背景

Vision Transformers被广泛用作视觉基础模型的骨干网络，但它们会产生高范数伪影，降低表示质量。在知识蒸馏中，这些高范数伪影会主导目标函数，导致学生模型过度拟合伪影而低估信息信号。

### 目的

开发一种新的知识蒸馏框架，能够在抑制Vision Transformers产生的高范数伪影的同时，保留教师模型中的信息信号，从而提高学生模型的性能。

### 方法

引入Singular Nullspace-Guided Energy Reallocation (SiNGER)框架，其核心思想是有原则的教师特征精炼：在精炼过程中利用零空间引导的扰动来保留信息同时抑制伪影，然后将精炼后的教师特征蒸馏给学生。使用基于LoRA的适配器高效实现这种扰动，只需要最小的结构修改。

### 主要发现

大量实验表明SiNGER能持续改进学生模型，在多个下游任务中达到最先进性能，并产生更清晰和可解释的表示。

### 结论

SiNGER框架成功解决了知识蒸馏中高范数伪影和信息信号保留之间的权衡问题，为学生模型提供了更有效的知识转移方式。

### 翻译

Vision Transformers被广泛用作视觉基础模型的骨干网络，但它们会产生高范数伪影，降低表示质量。当知识蒸馏将这些特征转移到学生模型时，高范数伪影主导了目标函数，导致学生模型过度拟合伪影而低估信息信号，从而削弱了大模型的收益。先前的工作尝试移除伪影，但在抑制伪影和保留教师信息信号之间遇到了固有权衡。为解决这一问题，我们引入了Singular Nullspace-Guided Energy Reallocation (SiNGER)，这是一种新的蒸馏框架，可以在抑制伪影的同时保留信息信号。核心思想是有原则的教师特征精炼：在精炼过程中，我们利用零空间引导的扰动来保留信息同时抑制伪影。然后将精炼后的教师特征蒸馏给学生。我们使用基于LoRA的适配器高效实现这种扰动，只需要最小的结构修改。大量实验表明，SiNGER能持续改进学生模型，在多个下游任务中达到最先进性能，并产生更清晰和可解释的表示。


### 论文摘要

Vision Transformers are widely adopted as the backbone of vision foundation models, but they are known to produce high-norm artifacts that degrade representation quality. When knowledge distillation transfers these features to students, high-norm artifacts dominate the objective, so students overfit to artifacts and underweight informative signals, diminishing the gains from larger models. Prior work attempted to remove artifacts but encountered an inherent trade-off between artifact suppression and preserving informative signals from teachers. To address this, we introduce Singular Nullspace-Guided Energy Reallocation (SiNGER), a novel distillation framework that suppresses artifacts while preserving informative signals. The key idea is principled teacher feature refinement: during refinement, we leverage the nullspace-guided perturbation to preserve information while suppressing artifacts. Then, the refined teacher's features are distilled to a student. We implement this perturbation efficiently with a LoRA-based adapter that requires minimal structural modification. Extensive experiments show that \oursname consistently improves student models, achieving state-of-the-art performance in multiple downstream tasks and producing clearer and more interpretable representations.

---

## 58. Revisiting Data Challenges of Computational Pathology: A Pack-based Multiple Instance Learning Framework

**论文链接:** [http://arxiv.org/abs/2509.20923v1](http://arxiv.org/abs/2509.20923v1)

**作者:** Wenhao Tang, Heng Fang, Ge Wu, Xiang Li, Ming-Ming Cheng

**发布时间:** 2025-09-25

**备注:** 26 pages, 5 figures

### GPT解析

### 总结

本文提出了一种基于包装的多实例学习(PackMIL)框架，用于解决计算病理学中全切片图像(WSIs)分析面临的挑战，实现了更高的准确率和更快的训练速度。

### 背景

计算病理学将病理切片数字化为全切片图像(WSIs)用于癌症诊断和预后，但WSIs具有极长序列长度(高达200K)、显著长度变化(从200到200K)和有限监督，导致数据异构性和冗余性高，传统方法难以有效处理。

### 目的

全面解决计算病理学中WSIs分析面临的超长序列长度、显著长度变化和有限监督等挑战，提高模型性能和训练效率。

### 方法

1. 提出PackMIL框架，将多个采样的可变长度特征序列打包为固定长度序列，实现批量训练同时保留数据异构性；2. 引入残差分支，将多个切片中被丢弃的特征组合成超切片，使用定制标签进行多切片监督；3. 设计注意力驱动的下采样器，压缩特征以减少冗余。

### 主要发现

该方法在PANDA(UNI)数据集上实现了高达8%的准确率提升，同时仅使用12%的训练时间，实验表明关注计算病理学中的数据挑战在基础模型时代具有巨大潜力。

### 结论

专注于解决计算病理学中的数据挑战对于提高模型性能和训练效率至关重要，所提出的PackMIL框架为WSIs分析提供了有效解决方案。

### 翻译

计算病理学(CPath)将病理切片数字化为全切片图像(WSIs)，使得能够进行分析以用于癌症诊断和预后等关键医疗任务。然而，WSIs具有极长的序列长度(高达200K)、显著的长度变化(从200到200K)和有限的监督。这些序列长度的极端变化导致数据异构性和冗余性高。传统方法通常需要在训练效率和优化之间做出妥协，以在有限监督下保留这种异构性。为了全面解决这些挑战，我们提出了一种基于包装的多实例学习框架。它将多个采样的可变长度特征序列打包为固定长度序列，实现批量训练同时保留数据异构性。此外，我们引入了一个残差分支，将多个切片中被丢弃的特征组合成超切片，使用定制标签进行训练。它提供多切片监督，同时减少采样带来的特征损失。同时，引入了注意力驱动的下采样器来压缩两个分支中的特征以减少冗余。通过缓解这些挑战，我们的方法在PANDA(UNI)上实现了高达8%的准确率提升，同时仅使用12%的训练时间。大量实验表明，关注计算病理学中的数据挑战在基础模型时代具有巨大潜力。代码位于https://github.com/FangHeng/PackMIL


### 论文摘要

Computational pathology (CPath) digitizes pathology slides into whole slide images (WSIs), enabling analysis for critical healthcare tasks such as cancer diagnosis and prognosis. However, WSIs possess extremely long sequence lengths (up to 200K), significant length variations (from 200 to 200K), and limited supervision. These extreme variations in sequence length lead to high data heterogeneity and redundancy. Conventional methods often compromise on training efficiency and optimization to preserve such heterogeneity under limited supervision. To comprehensively address these challenges, we propose a pack-based MIL framework. It packs multiple sampled, variable-length feature sequences into fixed-length ones, enabling batched training while preserving data heterogeneity. Moreover, we introduce a residual branch that composes discarded features from multiple slides into a hyperslide which is trained with tailored labels. It offers multi-slide supervision while mitigating feature loss from sampling. Meanwhile, an attention-driven downsampler is introduced to compress features in both branches to reduce redundancy. By alleviating these challenges, our approach achieves an accuracy improvement of up to 8% while using only 12% of the training time in the PANDA(UNI). Extensive experiments demonstrate that focusing data challenges in CPath holds significant potential in the era of foundation models. The code is https://github.com/FangHeng/PackMIL

---

## 59. TasselNetV4: A vision foundation model for cross-scene, cross-scale, and cross-species plant counting

**论文链接:** [http://arxiv.org/abs/2509.20857v1](http://arxiv.org/abs/2509.20857v1)

**作者:** Xiaonan Hu, Xuebing Li, Jinyu Xu, Abdulkadir Duran Adan, Letian Zhou, Xuhui Zhu, Yanan Li, Wei Guo, Shouyang Liu, Wenzhong Liu, Hao Lu

**发布时间:** 2025-09-25

**备注:** 13 figures, 7 tables, code is available at  https://github.com/tiny-smart/tasselnetv4

### GPT解析

### 总结

本文提出了一种名为TasselNetV4的新型跨物种植物计数模型，结合了TasselNet的局部计数思想和类别无关计数(CAC)的提取与匹配范式，实现了跨场景、跨尺度和跨物种的高效植物计数。

### 背景

准确的植物计数对农业有价值，如作物产量预测、植物密度评估和表型量化。现有视觉方法通常使用检测或回归模型计数特定植物，但植物具有生物多样性，每年都有新品种培育，几乎不可能穷尽并构建所有物种依赖的计数模型。

### 目的

重新思考植物计数的问题表述，从'计数什么植物'转向'如何计数植物'，开发一种不依赖于特定物种的通用植物计数方法。

### 方法

继承TasselNet植物计数模型的思路，引入TasselNetV4扩展，基于普通视觉变换器构建，融合了多分支框感知局部计数器以增强跨尺度鲁棒性，并构建了两个具有挑战性的数据集PAC-105和PAC-Somalia进行测试。

### 主要发现

与最先进的CAC模型相比，TasselNetV4实现了卓越的计数性能和高效率。植物是动态的，随时间和空间变化，其非刚性结构导致当前CAC和开放世界检测模型在植物计数方面表现不佳。

### 结论

TasselNetV4已成为跨场景、跨尺度和跨物种植物计数的视觉基础模型，为解决植物生物多样性带来的计数挑战提供了新思路。

### 翻译

准确的植物计数为农业提供了有价值的信息，如作物产量预测、植物密度评估和表型量化。基于视觉的方法是目前的主流解决方案。现有技术通常使用检测或回归模型来计数特定植物。然而，植物具有生物多样性，每年都有新品种不断培育。几乎不可能穷尽并构建所有物种依赖的计数模型。受计算机视觉中类别无关计数(CAC)的启发，我们认为现在是重新思考植物计数问题表述的时候了，从'计数什么植物'转向'如何计数植物'。与大多数具有时空不变性的日常物体不同，植物是动态的，随时间和空间变化。它们的非刚性结构通常比计数头部和汽车等刚性实例导致更差的性能，因此当前的CAC和开放世界检测模型对于植物计数不是最优的。在这项工作中，我们继承了TasselNet植物计数模型的思路，并引入了一个新的扩展TasselNetV4，从物种特定计数转向跨物种计数。TasselNetV4结合了TasselNet的局部计数思想和CAC的提取与匹配范式。它基于普通视觉变换器构建，并融合了新颖的多分支框感知局部计数器，用于增强跨尺度鲁棒性。我们收集了两个具有挑战性的数据集：PAC-105和PAC-Somalia。与最先进的CAC模型进行的广泛实验表明，TasselNetV4不仅实现了卓越的计数性能，而且具有高效率。我们的结果表明，TasselNetV4已成为跨场景、跨尺度和跨物种植物计数的视觉基础模型。


### 论文摘要

Accurate plant counting provides valuable information for agriculture such as crop yield prediction, plant density assessment, and phenotype quantification. Vision-based approaches are currently the mainstream solution. Prior art typically uses a detection or a regression model to count a specific plant. However, plants have biodiversity, and new cultivars are increasingly bred each year. It is almost impossible to exhaust and build all species-dependent counting models. Inspired by class-agnostic counting (CAC) in computer vision, we argue that it is time to rethink the problem formulation of plant counting, from what plants to count to how to count plants. In contrast to most daily objects with spatial and temporal invariance, plants are dynamic, changing with time and space. Their non-rigid structure often leads to worse performance than counting rigid instances like heads and cars such that current CAC and open-world detection models are suboptimal to count plants. In this work, we inherit the vein of the TasselNet plant counting model and introduce a new extension, TasselNetV4, shifting from species-specific counting to cross-species counting. TasselNetV4 marries the local counting idea of TasselNet with the extract-and-match paradigm in CAC. It builds upon a plain vision transformer and incorporates novel multi-branch box-aware local counters used to enhance cross-scale robustness. Two challenging datasets, PAC-105 and PAC-Somalia, are harvested. Extensive experiments against state-of-the-art CAC models show that TasselNetV4 achieves not only superior counting performance but also high efficiency.Our results indicate that TasselNetV4 emerges to be a vision foundation model for cross-scene, cross-scale, and cross-species plant counting.

---

## 60. CaTS-Bench: Can Language Models Describe Numeric Time Series?

**论文链接:** [http://arxiv.org/abs/2509.20823v1](http://arxiv.org/abs/2509.20823v1)

**作者:** Luca Zhou, Pratham Yashwante, Marshall Fisher, Alessio Sampieri, Zihao Zhou, Fabio Galasso, Rose Yu

**发布时间:** 2025-09-25

**备注:** 9 pages, 4 images, 4 tables in the main paper. Many more in the  appendix

### GPT解析

### 总结

这篇论文介绍了CaTS-Bench，这是第一个大规模、真实世界的上下文感知时间序列描述基准测试，解决了现有基准测试的局限性，提供了丰富的数据集和评估方法，为时间序列分析和基础模型的交叉研究奠定了基础。

### 背景

时间序列描述是将数值时间序列用自然语言描述的任务，需要数值推理、趋势解释和上下文理解。然而，现有的基准测试通常依赖合成数据或过于简单的描述，并且通常忽略元数据和视觉表示。

### 目的

引入CaTS-Bench，这是第一个大规模、真实世界的上下文感知时间序列描述基准测试，以弥补现有基准测试的不足。

### 方法

CaTS-Bench来源于11个多样化数据集，重新构造成描述和问答任务，包含约465k训练和105k测试时间戳。每个样本包括数值序列片段、上下文元数据、线图图像和描述。提出了可扩展的参考描述生成流程，使用oracle LLM生成大多数参考描述并通过多种验证方法确保质量，同时提供579个测试描述的人类修订子集。

### 主要发现

CaTS-Bench还提供460个针对时间序列推理更深层次方面的多项选择题。提出了新的定制评估指标，并对领先的VLMs进行了基准测试，突显了它们的优势和持续存在的局限性。

### 结论

这些贡献共同将CaTS-Bench及其描述流程建立为时间序列分析和基础模型交叉领域未来研究的可靠且可扩展的基础。

### 翻译

时间序列描述是将数值时间序列用自然语言描述的任务，需要数值推理、趋势解释和上下文理解。然而，现有的基准测试通常依赖合成数据或过于简单的描述，并且通常忽略元数据和视觉表示。为了弥补这一差距，我们引入了CaTS-Bench，这是第一个大规模、真实世界的上下文感知时间序列描述基准测试。CaTS-Bench来源于11个多样化数据集，重新构造成描述和问答任务，包含约465k训练和105k测试时间戳。每个样本包括数值序列片段、上下文元数据、线图图像和描述。这项工作的一个关键贡献是用于生成参考描述的可扩展流程：虽然大多数参考描述由oracle LLM生成并通过事实检查、人类不可区分性研究和多样性分析进行验证，但我们还提供了579个测试描述的人类修订子集，从LLM输出中提炼以确保准确性和人类风格。除了描述外，CaTS-Bench还提供460个针对时间序列推理更深层次方面的多项选择题。我们进一步提出了新的定制评估指标，并对领先的VLMs进行了基准测试，突显了它们的优势和持续存在的局限性。这些贡献共同将CaTS-Bench及其描述流程建立为时间序列分析和基础模型交叉领域未来研究的可靠且可扩展的基础。


### 论文摘要

Time series captioning, the task of describing numeric time series in natural language, requires numerical reasoning, trend interpretation, and contextual understanding. Existing benchmarks, however, often rely on synthetic data or overly simplistic captions, and typically neglect metadata and visual representations. To close this gap, we introduce CaTS-Bench, the first large-scale, real-world benchmark for Context-aware Time Series captioning. CaTS-Bench is derived from 11 diverse datasets reframed as captioning and Q&A tasks, comprising roughly 465k training and 105k test timestamps. Each sample includes a numeric series segment, contextual metadata, a line-chart image, and a caption. A key contribution of this work is the scalable pipeline used to generate reference captions: while most references are produced by an oracle LLM and verified through factual checks, human indistinguishability studies, and diversity analyses, we also provide a human-revisited subset of 579 test captions, refined from LLM outputs to ensure accuracy and human-like style. Beyond captioning, CaTS-Bench offers 460 multiple-choice questions targeting deeper aspects of time series reasoning. We further propose new tailored evaluation metrics and benchmark leading VLMs, highlighting both their strengths and persistent limitations. Together, these contributions establish CaTS-Bench and its captioning pipeline as a reliable and extensible foundation for future research at the intersection of time series analysis and foundation models.

---

## 61. RAPTOR-GEN: RApid PosTeriOR GENerator for Bayesian Learning in Biomanufacturing

**论文链接:** [http://arxiv.org/abs/2509.20753v1](http://arxiv.org/abs/2509.20753v1)

**作者:** Wandi Xu, Wei Xie

**发布时间:** 2025-09-25

**备注:** 80 pages, 6 figures

### GPT解析

### 总结

RAPTOR-GEN是一个基于机制的贝叶斯学习框架，旨在从稀疏和异构的实验数据中加速智能数字孪生的发展，用于生物制药制造的快速按需生产。

### 背景

生物制药制造对公共健康至关重要，但由于生物过程的复杂性和变异性，缺乏快速按需生产生物治疗药物的灵活性。

### 目的

介绍RAPTOR-GEN框架，从稀疏和异构的实验数据中加速智能数字孪生的发展。

### 方法

RAPTOR-GEN建立在多尺度概率知识图谱基础上，表述为基于随机微分方程的基础模型，包含两个组成部分：(1)整合线性噪声近似的可解释元模型，利用生物处理机制结构信息和顺序学习策略融合数据；(2)利用朗之万扩散的高效贝叶斯后验采样方法。

### 主要发现

RAPTOR-GEN将LNA方法推广以避免步长选择的挑战，促进具有可证明有限样本性能保证的机制参数的稳健学习，开发了一种具有可控误差的快速稳健算法。

### 结论

数值实验证明了RAPTOR-GEN在揭示生物制造过程潜在调控机制方面的有效性。

### 翻译

生物制药制造对公共健康至关重要，但由于生物过程的复杂性和变异性，缺乏快速按需生产生物治疗药物的灵活性。为克服这一点，我们引入了RAPTOR-GEN（RApid PosTeriOR GENenerator），这是一个基于机制的贝叶斯学习框架，旨在从稀疏和异构的实验数据中加速智能数字孪生的发展。该框架建立在多尺度概率知识图谱(pKG)的基础上，被表述为基于随机微分方程(SDE)的基础模型，能够捕捉生物过程的非线性动力学。RAPTOR-GEN包含两个组成部分：(i)一个可解释的元模型，整合了线性噪声近似(LNA)，利用生物处理机制的结构信息，并采用顺序学习策略融合异构和稀疏数据，能够推断潜在状态变量并明确逼近难以处理的似然函数；(ii)一种高效的贝叶斯后验采样方法，利用朗之万扩散(LD)通过利用导出的似然的梯度来加速后验探索。它将LNA方法推广以避免步长选择的挑战，促进具有可证明有限样本性能保证的机制参数的稳健学习。我们开发了一种具有可控误差的快速稳健RAPTOR-GEN算法。数值实验证明了其在揭示生物制造过程潜在调控机制方面的有效性。


### 论文摘要

Biopharmaceutical manufacturing is vital to public health but lacks the agility for rapid, on-demand production of biotherapeutics due to the complexity and variability of bioprocesses. To overcome this, we introduce RApid PosTeriOR GENerator (RAPTOR-GEN), a mechanism-informed Bayesian learning framework designed to accelerate intelligent digital twin development from sparse and heterogeneous experimental data. This framework is built on a multi-scale probabilistic knowledge graph (pKG), formulated as a stochastic differential equation (SDE)-based foundational model that captures the nonlinear dynamics of bioprocesses. RAPTOR-GEN consists of two ingredients: (i) an interpretable metamodel integrating linear noise approximation (LNA) that exploits the structural information of bioprocessing mechanisms and a sequential learning strategy to fuse heterogeneous and sparse data, enabling inference of latent state variables and explicit approximation of the intractable likelihood function; and (ii) an efficient Bayesian posterior sampling method that utilizes Langevin diffusion (LD) to accelerate posterior exploration by exploiting the gradients of the derived likelihood. It generalizes the LNA approach to circumvent the challenge of step size selection, facilitating robust learning of mechanistic parameters with provable finite-sample performance guarantees. We develop a fast and robust RAPTOR-GEN algorithm with controllable error. Numerical experiments demonstrate its effectiveness in uncovering the underlying regulatory mechanisms of biomanufacturing processes.

---

## 62. Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation

**论文链接:** [http://arxiv.org/abs/2509.20681v1](http://arxiv.org/abs/2509.20681v1)

**作者:** Wei-Teng Chu, Tianyi Zhang, Matthew Johnson-Roberson, Weiming Zhi

**发布时间:** 2025-09-25

### GPT解析

### 总结

本文提出了一种名为Fast Image-to-Neural Surface (FINS)的轻量级框架，能够从单张或少量图像重建高保真表面和SDF场。该方法结合多分辨率哈希网格编码器与轻量级几何和颜色头部，通过近似二阶优化器实现高效训练，能在几秒内收敛。研究还展示了FINS在机器人表面跟随任务中的适用性及其在各种基准数据集上的可扩展性。

### 背景

隐式表示已在机器人领域广泛应用于避障和路径规划。然而，现有的隐式表面重建方法（如NeuS及其变体）通常需要大量多视角图像作为输入，并且需要很长的训练时间。

### 目的

探索从单张图像构建隐式距离表示的问题，开发一种能够从单张或少量图像重建高保真表面和SDF场的高效方法。

### 方法

提出Fast Image-to-Neural Surface (FINS)框架，该方法集成了多分辨率哈希网格编码器，使用轻量级的几何和颜色头部，通过近似二阶优化器实现高效训练。此外，利用预训练的基础模型来估计图像中固有的几何信息，从而仅使用单张RGB图像构建神经表面。

### 主要发现

实验表明，在相同条件下，该方法在表面重建和SDF场估计的收敛速度和准确性上都优于最先进的基线方法。此外，FINS在机器人表面跟随任务中表现出适用性，并且在各种基准数据集上具有可扩展性。

### 结论

FINS是一种高效的方法，能够从单张或少量图像重建高质量的隐式表面和SDF场，相比现有方法具有更快的收敛速度和更高的准确性，且在实际应用中具有良好的适用性和可扩展性。

### 翻译

隐式表示已在机器人领域被广泛应用于避障和路径规划。在本文中，我们探索了从单张图像构建隐式距离表示的问题。过去的隐式表面重建方法，如NeuS及其变体通常需要大量多视角图像作为输入，并且需要很长的训练时间。在这项工作中，我们提出了Fast Image-to-Neural Surface (FINS)，一个轻量级框架，可以根据单张或少量图像重建高保真表面和SDF场。FINS集成了多分辨率哈希网格编码器与轻量级的几何和颜色头部，使得通过近似二阶优化器进行训练非常高效，并且能够在几秒内收敛。此外，我们通过利用预训练的基础模型来估计图像中固有的几何信息，实现了仅使用单张RGB图像构建神经表面。我们的实验证明，在相同条件下，我们的方法在表面重建和SDF场估计的收敛速度和准确性上都优于最先进的基线方法。此外，我们展示了FINS在机器人表面跟随任务中的适用性，并证明了其在各种基准数据集上的可扩展性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从单张图像快速构建隐式表面模型的问题。这个问题很重要，因为自主机器人需要快速理解周围环境几何以安全导航和交互，而传统方法需要大量多视角图像和长时间训练，不适合机器人实时应用场景。准确的环境几何表示对障碍物避让和路径规划至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有隐式表面重建方法的局限性：需要密集多视角监督和长时间训练。然后借鉴了预训练3D基础模型（如DUSt3R和VGGT）来估计单张图像几何信息，采用多分辨率哈希网格编码高效表示空间特征，并使用近似二阶优化器加速收敛。作者将这些技术创新性地组合，设计了轻量级框架，实现了从单张图像快速重建高保真表面。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合预训练3D基础模型、多分辨率哈希编码和混合优化策略，实现从单张图像快速重建隐式表面。流程包括：1)用预训练3D模型将输入图像转换为3D点云；2)使用多分辨率哈希编码器将3D坐标编码为特征；3)通过轻量级几何和颜色头预测SDF值和颜色；4)采用分阶段优化策略（先用一阶优化器预热，再用二阶优化器快速收敛）；5)最后提取等值面生成网格模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次实现从单张图像在消费级硬件上约10秒内重建高保真表面；2)利用预训练3D基础模型生成点云监督信号；3)采用多分辨率哈希编码和轻量级头设计；4)使用混合优化策略实现快速收敛。相比之前工作，FINS大幅减少了输入图像需求（从5-49张减少到1张）和训练时间（从18-600秒减少到10秒），同时保持竞争性的重建质量，更适合实时机器人应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FINS通过结合预训练3D基础模型、多分辨率哈希编码和混合优化策略，首次实现了从单张图像在消费级硬件上仅需约10秒即可重建高保真隐式表面模型，为机器人实时感知和导航提供了高效解决方案。'}


### 论文摘要

Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as \emph{NeuS} and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets.

---

## 63. Boosting Zero-Shot VLN via Abstract Obstacle Map-Based Waypoint Prediction with TopoGraph-and-VisitInfo-Aware Prompting

**论文链接:** [http://arxiv.org/abs/2509.20499v1](http://arxiv.org/abs/2509.20499v1)

**作者:** Boqi Li, Siyuan Li, Weiyi Wang, Anran Li, Zhong Cao, Henry X. Liu

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种零样本框架，结合简化的航路点预测器和多模态大语言模型，用于解决连续环境中的视觉语言导航问题，在R2R-CE和RxR-CE数据集上实现了最先进的零样本性能。

### 背景

随着基础模型和机器人技术的快速发展，视觉语言导航已成为具身智能体的关键任务，具有广泛的应用前景。连续环境中的VLN尤其具有挑战性，因为智能体需要同时解释自然语言指令、感知周围环境并规划低级别动作。

### 目的

解决具身智能体在连续环境中需要联合处理自然语言指令、环境感知和低级别动作规划的挑战性问题。

### 方法

提出一个零样本框架，整合了在抽象障碍地图上操作的航路点预测器和多模态大语言模型。预测器产生线性可达的航路点，这些航路点被整合到具有访问记录的动态拓扑图中。图和访问信息被编码到提示中，使模型能够对空间结构和探索历史进行推理，鼓励探索并实现局部路径规划用于错误纠正。

### 主要发现

在R2R-CE和RxR-CE数据集上的广泛实验表明，该方法实现了最先进的零样本性能，成功率分别达到41%和36%，优于之前的最先进方法。

### 结论

所提出的方法在视觉语言导航任务中表现优异，特别是在零样本设置下，为具身智能体在连续环境中的导航提供了有效解决方案。

### 翻译

随着基础模型和机器人技术的快速发展，视觉语言导航已成为具身智能体的关键任务，具有广泛的应用前景。我们解决了连续环境中的视觉语言导航问题，这是一个特别具挑战性的设置，因为智能体必须同时解释自然语言指令、感知周围环境并规划低级别动作。我们提出一个零样本框架，集成了简化的 yet 有效的航路点预测器与多模态大语言模型。预测器在抽象障碍地图上操作，产生线性可达的航路点，这些航路点被整合到具有明确访问记录的动态更新的拓扑图中。图和访问信息被编码到提示中，使模型能够对空间结构和探索历史进行推理，鼓励探索并使MLLM具备局部路径规划能力用于错误纠正。在R2R-CE和RxR-CE上的广泛实验表明，我们的方法实现了最先进的零样本性能，成功率分别为41%和36%，优于之前的最先进方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决连续环境中的视觉语言导航(VLN)问题，即智能体需要理解自然语言指令并在自由移动的环境中导航到指定目标。这个问题在现实中非常重要，因为它涉及具身AI的核心能力，可应用于搜索救援、自主导航和日常人机交互等领域。连续环境中的VLN特别具有挑战性，因为智能体需要同时处理语言理解、环境感知和低级动作规划，而现有方法在零样本设置下性能有限。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有航点预测方法的局限性：复杂的RGB-D输入会引入无关信息，且预测的航点可能不可达。因此，他们提出简化输入表示，将深度图像转换为抽象障碍物地图，使模型专注于空间可通行性。作者借鉴了ETPNav的拓扑图构建方法和MapGPT的自然语言表示方式，同时参考了AO-Planner的提示设计思想，但加入了拓扑图和访问信息以增强空间推理能力。这种设计思路体现了从简化输入到增强推理的系统性思考。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过抽象障碍物地图简化航点预测，并结合拓扑图和访问信息感知提示增强多模态大语言模型的导航能力。整体流程包括：1)将深度图像转换为障碍物地图；2)使用轻量级模型预测线性可达的航点；3)构建动态更新的拓扑图，标记已访问和未访问节点；4)将拓扑图和访问信息编码为提示，提供给MLLM进行导航决策；5)MLLM基于提示推理并决定下一步动作，支持探索和错误纠正；6)循环执行直到到达目标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于抽象障碍物地图的轻量级航点预测器，使用梯度方法检测障碍物而非固定阈值；2)拓扑图和访问信息感知提示系统，明确记录探索历史和空间结构；3)集成框架实现零样本VLN。相比之前工作，不同之处在于：使用简化的障碍物地图代替复杂RGB-D输入；确保航点线性可达；利用拓扑图而非仅顺序轨迹记录空间关系；提供局部路径规划能力进行错误纠正；在保持模型轻量化的同时实现了更优性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于抽象障碍物地图的航点预测与拓扑图和访问信息感知提示相结合的零样本视觉语言导航框架，显著提升了连续环境中的导航性能并实现了最先进的零样本结果。'}


### 论文摘要

With the rapid progress of foundation models and robotics, vision-language navigation (VLN) has emerged as a key task for embodied agents with broad practical applications. We address VLN in continuous environments, a particularly challenging setting where an agent must jointly interpret natural language instructions, perceive its surroundings, and plan low-level actions. We propose a zero-shot framework that integrates a simplified yet effective waypoint predictor with a multimodal large language model (MLLM). The predictor operates on an abstract obstacle map, producing linearly reachable waypoints, which are incorporated into a dynamically updated topological graph with explicit visitation records. The graph and visitation information are encoded into the prompt, enabling reasoning over both spatial structure and exploration history to encourage exploration and equip MLLM with local path planning for error correction. Extensive experiments on R2R-CE and RxR-CE show that our method achieves state-of-the-art zero-shot performance, with success rates of 41% and 36%, respectively, outperforming prior state-of-the-art methods.

---

## 64. Are Foundation Models Ready for Industrial Defect Recognition? A Reality Check on Real-World Data

**论文链接:** [http://arxiv.org/abs/2509.20479v1](http://arxiv.org/abs/2509.20479v1)

**作者:** Simon Baeuerle, Pratik Khanna, Nils Friederich, Angelo Jovin Yamachui Sitcheu, Damir Shakirov, Andreas Steimer, Ralf Mikut

**发布时间:** 2025-09-24

### GPT解析

### 总结

基础模型在文本和图像处理任务中表现出色，具有跨领域泛化能力，但在真实工业数据上的应用存在局限性。

### 背景

基础模型在零样本设置下可跨领域和数据集泛化，这使其成为系列制造过程中自动化质量检查的潜在解决方案。

### 目的

探索基础模型在工业图像质量检查中的应用，用简单文本提示代替繁琐标记任务，利用相同模型处理多种产品以节省模型设置和实施的工作量。

### 方法

测试多个最新的基础模型，使用自定义真实工业图像数据和公共图像数据，比较模型在两种数据集上的表现。

### 主要发现

所有测试的基础模型在真实世界工业数据上表现不佳，但相同的模型在公共基准数据集上表现良好。

### 结论

基础模型虽然在公共数据集上表现良好，但在真实工业环境中存在应用局限性，需要进一步研究其在实际应用中的问题。

### 翻译

基础模型在文本和图像处理任务上表现出色。它们可以在零样本设置下跨领域和数据集泛化。这可能使它们适合在系列制造过程中的自动化质量检查，其中正在评估各种类型图像以检查许多不同产品。用简单的文本提示描述异常来代替繁琐的标记任务，并在许多产品上使用相同的模型，将在模型设置和实施过程中节省大量工作。这与监督式人工智能模型相比是一个显著优势，监督式AI模型针对单个应用程序进行训练，需要标记的训练数据。我们在自定义的真实工业图像数据和公共图像数据上测试了多个最新的基础模型。我们表明所有这些模型在我们的真实数据上都失败了，而完全相同的模型在公共基准数据集上表现良好。


### 论文摘要

Foundation Models (FMs) have shown impressive performance on various text and image processing tasks. They can generalize across domains and datasets in a zero-shot setting. This could make them suitable for automated quality inspection during series manufacturing, where various types of images are being evaluated for many different products. Replacing tedious labeling tasks with a simple text prompt to describe anomalies and utilizing the same models across many products would save significant efforts during model setup and implementation. This is a strong advantage over supervised Artificial Intelligence (AI) models, which are trained for individual applications and require labeled training data. We test multiple recent FMs on both custom real-world industrial image data and public image data. We show that all of those models fail on our real-world data, while the very same models perform well on public benchmark datasets.

---

## 65. Discovering Association Rules in High-Dimensional Small Tabular Data

**论文链接:** [http://arxiv.org/abs/2509.20113v2](http://arxiv.org/abs/2509.20113v2)

**作者:** Erkan Karabulut, Daniel Daza, Paul Groth, Victoria Degeler

**发布时间:** 2025-09-24

**备注:** This paper was accepted at ECAI 2025 Workshop: 1st International  Workshop on Advanced Neuro-Symbolic Applications (ANSyA)

### GPT解析

### 总结

本文提出了一种改进的关联规则挖掘方法，特别针对高维、低数据场景，通过结合神经符号方法和表格基础模型微调技术，有效解决了规则爆炸问题并提高了规则质量。

### 背景

关联规则挖掘旨在发现数据集中特征间的模式，支持高风险决策中的知识发现和可解释机器学习。但在高维数据中，规则爆炸和计算开销使传统方法不切实际，神经符号方法如Aerial+虽解决高维问题，但在低数据场景下性能受限。

### 目的

解决高维数据中的关联规则挖掘问题，特别是在低数据场景下的挑战，提高算法的扩展性和规则质量。

### 方法

提出两种基于表格基础模型的Aerial+微调方法，用于高维、低数据场景下的关联规则挖掘。

### 主要发现

Aerial+比最先进的算法和神经符号基线在五个真实世界数据集上扩展性好一到两个数量级；提出了高维、低数据设置下的ARM问题；提出的微调方法显著提高了五个真实世界数据集上的规则质量。

### 结论

结合神经符号方法和表格基础模型微调的技术在高维、低数据场景中能有效提升关联规则挖掘的性能和质量。

### 翻译

关联规则挖掘旨在以命题规则的形式发现数据集中特征之间的模式，支持高风险决策中的知识发现和可解释机器学习。然而，在高维设置中，规则爆炸和计算开销使得没有有效搜索空间减少的流行算法方法不切实际，这些挑战会传递到下游任务。神经符号方法，如Aerial+，最近被提出以解决ARM中的规则爆炸问题。虽然它们处理了数据的高维性，但也继承了神经网络的局限性，特别是在低数据环境下的性能降低。本文对高维表格数据中的关联规则发现做出了三项关键贡献。首先，我们在五个真实世界数据集上 empirically 证明 Aerial+ 比最先进的算法和神经符号基线扩展性好一到两个数量级。其次，我们提出了高维、低数据设置下的ARM新问题，例如生物医学领域具有约18k特征和50个样本的基因表达数据。第三，我们提出了两种使用表格基础模型对Aerial+进行微调的方法。我们提出的方法在五个真实世界数据集上被证明显著提高了规则质量，展示了它们在低数据、高维场景中的有效性。


### 论文摘要

Association Rule Mining (ARM) aims to discover patterns between features in datasets in the form of propositional rules, supporting both knowledge discovery and interpretable machine learning in high-stakes decision-making. However, in high-dimensional settings, rule explosion and computational overhead render popular algorithmic approaches impractical without effective search space reduction, challenges that propagate to downstream tasks. Neurosymbolic methods, such as Aerial+, have recently been proposed to address the rule explosion in ARM. While they tackle the high dimensionality of the data, they also inherit limitations of neural networks, particularly reduced performance in low-data regimes.   This paper makes three key contributions to association rule discovery in high-dimensional tabular data. First, we empirically show that Aerial+ scales one to two orders of magnitude better than state-of-the-art algorithmic and neurosymbolic baselines across five real-world datasets. Second, we introduce the novel problem of ARM in high-dimensional, low-data settings, such as gene expression data from the biomedicine domain with around 18k features and 50 samples. Third, we propose two fine-tuning approaches to Aerial+ using tabular foundation models. Our proposed approaches are shown to significantly improve rule quality on five real-world datasets, demonstrating their effectiveness in low-data, high-dimensional scenarios.

---

## 66. Hyperspectral Adapter for Semantic Segmentation with Vision Foundation Models

**论文链接:** [http://arxiv.org/abs/2509.20107v2](http://arxiv.org/abs/2509.20107v2)

**作者:** Juana Valeria Hurtado, Rohit Mohan, Abhinav Valada

**发布时间:** 2025-09-24

### GPT解析

### 总结

该研究提出了一种新的高光谱适配器架构，利用预训练视觉基础模型有效学习高光谱数据，在自动驾驶场景中实现了最先进的语义分割性能。

### 背景

高光谱成像能够捕获空间信息和密集的光谱测量，在复杂材料成分、光照变化等视觉挑战性环境中具有促进机器人感知的潜力。然而，当前HSI语义分割方法表现不佳，因为它们依赖于为RGB输入优化的架构和学习框架。

### 目的

提出一种新的高光谱适配器，利用预训练的视觉基础模型有效学习高光谱数据，解决当前HSI语义分割方法性能不足的问题。

### 方法

提出了一种包含光谱变换器和光谱感知空间先验模块的架构，以提取丰富的空间-光谱特征。同时引入了一种模态感知交互块，通过专门的提取和注入机制促进高光谱表示和冻结视觉Transformer特征的有效集成。

### 主要发现

在三个基准自动驾驶数据集上的广泛评估表明，该架构在使用HSI输入直接时实现了最先进的语义分割性能，优于基于视觉和高光谱分割的方法。

### 结论

该研究成功解决了HSI语义分割中的性能问题，通过新的适配器架构实现了与视觉方法相当或更好的性能，为高光谱成像在机器人感知中的应用提供了新途径。

### 翻译

高光谱成像捕获空间信息以及众多窄波长波段上的密集光谱测量。这种丰富的光谱内容有可能促进强大的机器人感知，特别是在具有复杂材料成分、变化光照或其他视觉挑战性条件的环境中。然而，当前的高光谱语义分割方法表现不佳，因为它们依赖于为RGB输入优化的架构和学习框架。在这项工作中，我们提出了一种新的高光谱适配器，它利用预训练的视觉基础模型来有效学习高光谱数据。我们的架构集成了光谱变换器和光谱感知空间先验模块，以提取丰富的空间-光谱特征。此外，我们引入了一种模态感知交互块，通过专门的提取和注入机制促进高光谱表示和冻结视觉Transformer特征的有效集成。在三个基准自动驾驶数据集上的广泛评估表明，我们的架构在使用HSI输入直接时实现了最先进的语义分割性能，优于基于视觉和高光谱的分割方法。我们在https://hsi-adapter.cs.uni-freiburg.de提供代码。


### 论文摘要

Hyperspectral imaging (HSI) captures spatial information along with dense spectral measurements across numerous narrow wavelength bands. This rich spectral content has the potential to facilitate robust robotic perception, particularly in environments with complex material compositions, varying illumination, or other visually challenging conditions. However, current HSI semantic segmentation methods underperform due to their reliance on architectures and learning frameworks optimized for RGB inputs. In this work, we propose a novel hyperspectral adapter that leverages pretrained vision foundation models to effectively learn from hyperspectral data. Our architecture incorporates a spectral transformer and a spectrum-aware spatial prior module to extract rich spatial-spectral features. Additionally, we introduce a modality-aware interaction block that facilitates effective integration of hyperspectral representations and frozen vision Transformer features through dedicated extraction and injection mechanisms. Extensive evaluations on three benchmark autonomous driving datasets demonstrate that our architecture achieves state-of-the-art semantic segmentation performance while directly using HSI inputs, outperforming both vision-based and hyperspectral segmentation methods. We make the code available at https://hsi-adapter.cs.uni-freiburg.de.

---

