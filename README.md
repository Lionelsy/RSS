# 今日论文推荐 - 2025-12-04

共 214 篇论文

---

## 1. PosterCopilot: Toward Layout Reasoning and Controllable Editing for Professional Graphic Design

**论文链接:** [http://arxiv.org/abs/2512.04082v1](http://arxiv.org/abs/2512.04082v1)

**作者:** Jiazhe Wei, Ken Li, Tianyu Lao, Haofan Wang, Liang Wang, Caifeng Shan, Chenyang Si

**发布时间:** 2025-12-03

**备注:** Project page: https://postercopilot.github.io/

### GPT解析

### 总结

本文介绍了PosterCopilot框架，通过渐进式三阶段训练策略和完整工作流程，实现了专业图形设计的几何准确布局和分层可控编辑。

### 背景

图形设计是现代视觉沟通的基石，但现有使用大型多模态模型（LMMs）的自动化方法往往产生几何不准确的布局，且缺乏专业工作流程所需的分层迭代编辑功能。

### 目的

解决现有自动化图形设计方法的局限性，提高布局的几何准确性，并实现专业工作流程所需的分层迭代编辑功能。

### 方法

提出PosterCopilot框架，采用渐进式三阶段训练策略（扰动监督微调、视觉-现实对齐的强化学习、美学反馈的强化学习），并将基于LMM的设计模型与生成模型结合，实现分层可控、迭代编辑。

### 主要发现

大量实验证明，PosterCopilot能够实现几何准确且美学优越的布局设计，为专业迭代设计提供了前所未有的可控性。

### 结论

PosterCopilot框架成功解决了现有自动化图形设计方法的局限性，提高了布局的几何准确性，并提供了专业工作流程所需的分层迭代编辑功能。

### 翻译

图形设计构成了现代视觉沟通的基石，作为推广文化和商业活动的重要媒介。最近的进展探索了使用大型多模态模型（LMMs）来自动化这一过程，但现有方法往往产生几何不准确的布局，并且缺乏专业工作流程所需的分层、迭代编辑。为解决这些局限性，我们提出了PosterCopilot，一个推进专业图形设计布局推理和可控编辑的框架。具体而言，我们引入了一个渐进式三阶段训练策略，使LMMs具备布局设计的几何理解和美学推理能力，包括扰动监督微调、视觉-现实对齐的强化学习以及美学反馈的强化学习。此外，我们开发了一个完整的工作流程，将基于LMM的设计模型与生成模型相结合，实现了分层可控、迭代编辑，从而在保持全局视觉一致性的同时进行精确的元素细化。大量实验证明，PosterCopilot实现了几何准确且美学优越的布局设计，为专业迭代设计提供了前所未有的可控性。


### 论文摘要

Graphic design forms the cornerstone of modern visual communication, serving as a vital medium for promoting cultural and commercial events. Recent advances have explored automating this process using Large Multimodal Models (LMMs), yet existing methods often produce geometrically inaccurate layouts and lack the iterative, layer-specific editing required in professional workflows. To address these limitations, we present PosterCopilot, a framework that advances layout reasoning and controllable editing for professional graphic design. Specifically, we introduce a progressive three-stage training strategy that equips LMMs with geometric understanding and aesthetic reasoning for layout design, consisting of Perturbed Supervised Fine-Tuning, Reinforcement Learning for Visual-Reality Alignment, and Reinforcement Learning from Aesthetic Feedback. Furthermore, we develop a complete workflow that couples the trained LMM-based design model with generative models, enabling layer-controllable, iterative editing for precise element refinement while maintaining global visual consistency. Extensive experiments demonstrate that PosterCopilot achieves geometrically accurate and aesthetically superior layouts, offering unprecedented controllability for professional iterative design.

---

## 2. SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL

**论文链接:** [http://arxiv.org/abs/2512.04069v1](http://arxiv.org/abs/2512.04069v1)

**作者:** Siyi Chen, Mikaela Angelina Uy, Chan Hee Song, Faisal Ladhak, Adithyavairavan Murali, Qing Qu, Stan Birchfield, Valts Blukis, Jonathan Tremblay

**发布时间:** 2025-12-03

### GPT解析

### 总结

这篇论文提出了双重交互式强化学习(DIRL)框架，解决了视觉语言模型(VLMs)在多工具协调方面的挑战，显著提升了空间推理能力，在空间理解基准测试上实现了最先进的性能。

### 背景

视觉语言模型(VLMs)在视觉理解方面表现出色，但在需要精确空间推理的实体应用中表现不佳。代理范式表明VLMs可以使用各种工具(如深度估计器、分割模型和姿态估计器)来增强这些能力。

### 目的

开发一种不依赖手工提示策略或固定工具管道的方法，使VLMs能够发现最优的工具使用模式，特别是在多工具协调方面。

### 方法

提出了双重交互式强化学习(DIRL)，一个两阶段训练框架。在教学阶段，结合单一工具专家的演示和前沿模型使用所有工具的轨迹；在探索阶段，模型通过持续的强化学习进一步改进多工具协调能力。

### 主要发现

SpaceTools模型在空间理解基准测试(RoboSpatial-Home, BLINK, BOP-ASK)上实现了最先进的性能，并在使用7-DOF机器人的实际操作中表现出可靠的性能。DIRL比普通的SFT(+12%在RoboSpatial)和RL(+16%在RoboSpatial)基线有显著改进。

### 结论

DIRL框架有效解决了VLMs在多工具协调方面的挑战，显著提升了空间推理能力，为实体应用提供了新的可能性。

### 翻译

视觉语言模型(VLMs)展示了强大的定性视觉理解能力，但在需要精确空间推理的实体应用中表现不佳。代理范式表明VLMs可以使用各种工具来增强这些能力，如深度估计器、分割模型和姿态估计器。然而，如何不依赖手工提示策略或强制执行固定的预定义工具管道(这些管道限制了VLMs发现最优工具使用模式的能力)来实现这一愿景仍然是一个开放的挑战。强化学习可以弥补这一差距，但由于多工具推理中的巨大搜索空间，迄今为止仅限于使用单个视觉工具进行推理。我们引入了双重交互式强化学习(DIRL)，这是一个两阶段训练框架，VLMs通过交互式探索和反馈学习协调多个工具。在教学阶段，我们结合了通过交互式RL训练的单一工具专家的演示和使用所有工具的前沿模型的轨迹。在探索阶段，模型通过持续的RL进一步改进多工具协调能力。我们的模型SpaceTools具有工具增强的空间推理能力，在空间理解基准测试(RoboSpatial-Home, BLINK, BOP-ASK)上实现了最先进的性能，并展示了使用7-DOF机器人作为工具的可靠实际操作。DIRL比普通的SFT(+12%在RoboSpatial)和RL(+16%在RoboSpatial)基线提供了显著改进。项目页面：https://spacetools.github.io/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉语言模型（VLMs）在精确空间推理方面的不足。虽然VLMs在开放性视觉问答上表现良好，但在需要精确几何理解、3D感知和复杂多步推理的空间推理任务中仍然存在挑战。这个问题在机器人应用中尤为重要，因为机器人需要精确理解物体间的空间关系、距离和方向才能有效执行任务，如判断物体能否放置在一起或确定抓取部位。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到VLMs在空间推理上的局限性，考虑引入外部工具增强能力。借鉴了ViGoRL等使用强化学习让VLM学习与单个工具交互的工作，但发现直接应用于多工具会导致搜索空间过大。因此设计了双交互强化学习（DIRL）框架，分为教学阶段（结合单工具教师模型和全工具前沿模型的演示进行监督微调）和探索阶段（使用全工具集进行交互式强化学习）。此外，还开发了Toolshed平台解决大规模工具交互的系统挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过双阶段训练框架让VLMs学会协调使用多种视觉和机器人工具解决空间推理问题。教学阶段先建立基本工具使用能力，探索阶段再优化多工具协调。整体流程：模型接收用户查询，根据需要调用工具执行任务，将工具输出与历史对话合并继续推理，最终生成答案。Toolshed平台提供各种工具的高效部署，支持资源隔离和异步执行。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1）双交互强化学习（DIRL）框架，将多工具强化学习分解为可管理的两个阶段；2）Toolshed平台，提供高效工具部署和管理系统；3）SpaceTools模型，能动态选择、组合和协调多种工具，支持错误恢复。相比之前工作，DIRL允许模型自主发现最优工具使用模式，而非依赖固定管道或手工提示；能处理多工具协调而非仅单工具；支持交互式工具使用而非预计算输出；通过实际工具反馈提供更有效学习信号。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了双交互强化学习框架和工具平台，使视觉语言模型能够通过协调多种工具实现精确的空间推理，在多个基准测试中达到最先进性能，同时在真实机器人操作中表现出色。'}


### 论文摘要

Vision Language Models (VLMs) demonstrate strong qualitative visual understanding, but struggle with metrically precise spatial reasoning required for embodied applications. The agentic paradigm promises that VLMs can use a wide variety of tools that could augment these capabilities, such as depth estimators, segmentation models, and pose estimators. Yet it remains an open challenge how to realize this vision without solely relying on handcrafted prompting strategies or enforcing fixed, predefined tool pipelines that limit VLMs' ability to discover optimal tool-use patterns. Reinforcement Learning could overcome this gap, but has so far been limited to reasoning with a single visual tool due to the large search space in multi-tool reasoning. We introduce Double Interactive Reinforcement Learning (DIRL), a two-phase training framework where VLMs learn to coordinate multiple tools through interactive exploration and feedback. In the teaching phase, we combine demonstrations from a single tool specialist trained via interactive RL with traces from a frontier model using all tools. In the exploration phase, the model further refines multi-tool coordination through continued RL. Our model, SpaceTools, with tool-augmented spatial reasoning ability, achieves state-of-the-art performance on spatial understanding benchmarks (RoboSpatial-Home, BLINK, BOP-ASK) and demonstrates reliable real-world manipulation using a 7-DOF robot as a tool. DIRL provides substantial improvements over the vanilla SFT (+12% on RoboSpatial) and RL (+16% on RoboSpatial) baselines. Project page: https://spacetools.github.io/.

---

## 3. RELIC: Interactive Video World Model with Long-Horizon Memory

**论文链接:** [http://arxiv.org/abs/2512.04040v1](http://arxiv.org/abs/2512.04040v1)

**作者:** Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, Hao Tan

**发布时间:** 2025-12-03

**备注:** 22 pages

### GPT解析

### 总结

本文提出RELIC框架，一个统一解决方案，同时解决交互式世界模型中的三个关键挑战：实时长时程流式传输、一致的空间记忆和精确的用户控制。RELIC能在给定单张图像和文本描述的情况下，实时实现具有记忆感知的任意场景长时间探索。

### 背景

真正的交互式世界模型需要三个关键要素：实时长时程流式传输、一致的空间记忆和精确的用户控制。然而，大多数现有方法仅单独解决其中一方面，因为同时实现这三个方面极具挑战性。例如，长期记忆机制通常会降低实时性能。

### 目的

开发一个统一框架RELIC，同时解决交互式世界模型中的三个关键挑战：实时长时程流式传输、一致的空间记忆和精确的用户控制。

### 方法

RELIC基于自回归视频扩散蒸馏技术，使用高度压缩的历史潜在令牌表示长时程记忆，这些令牌在KV缓存中编码了相对动作和绝对相机姿态。这种紧凑的记忆结构支持3D一致内容检索并强制执行长期一致性。同时，微调双向教师视频模型并使用新的内存高效自强制范式将其转换为因果学生生成器。RELIC实现为140亿参数模型，在Unreal Engine渲染数据集上训练。

### 主要发现

RELIC能够以16FPS的速度实现实时生成，在动作跟随准确性、长时程流式传输稳定性和空间记忆检索鲁棒性方面优于先前工作。

### 结论

RELIC的能力使其成为下一代交互式世界建模的坚实基础。

### 翻译

一个真正的交互式世界模型需要三个关键要素：实时长时程流式传输、一致的空间记忆和精确的用户控制。然而，大多数现有方法仅单独解决其中一方面，因为同时实现这三个方面极具挑战性——例如，长期记忆机制通常会降低实时性能。在这项工作中，我们提出了RELIC，这是一个统一框架，同时解决这三个挑战。给定单张图像和文本描述，RELIC能够实时实现具有记忆感知的任意场景长时间探索。基于最近的自回归视频扩散蒸馏技术，我们的模型使用高度压缩的历史潜在令牌表示长时程记忆，这些令牌在KV缓存中编码了相对动作和绝对相机姿态。这种紧凑的、具有相机感知能力的记忆结构支持隐式的3D一致内容检索，并以最小的计算开销强制执行长期一致性。同时，我们微调了一个双向教师视频模型，以生成超出其原始5秒训练时序的序列，并使用一种新的内存高效自强制范式将其转换为因果学生生成器，从而实现长时间教师和长时间学生自滚动的完整上下文蒸馏。作为一个140亿参数的模型并在精心筛选的Unreal Engine渲染数据集上训练，RELIC以16FPS的速度实现实时生成，同时表现出比先前工作更准确的动作跟随、更稳定的长时程流式传输和更鲁棒的空间记忆检索。这些能力使RELIC成为下一代交互式世界建模的坚实基础。


### 论文摘要

A truly interactive world model requires three key ingredients: real-time long-horizon streaming, consistent spatial memory, and precise user control. However, most existing approaches address only one of these aspects in isolation, as achieving all three simultaneously is highly challenging-for example, long-term memory mechanisms often degrade real-time performance. In this work, we present RELIC, a unified framework that tackles these three challenges altogether. Given a single image and a text description, RELIC enables memory-aware, long-duration exploration of arbitrary scenes in real time. Built upon recent autoregressive video-diffusion distillation techniques, our model represents long-horizon memory using highly compressed historical latent tokens encoded with both relative actions and absolute camera poses within the KV cache. This compact, camera-aware memory structure supports implicit 3D-consistent content retrieval and enforces long-term coherence with minimal computational overhead. In parallel, we fine-tune a bidirectional teacher video model to generate sequences beyond its original 5-second training horizon, and transform it into a causal student generator using a new memory-efficient self-forcing paradigm that enables full-context distillation over long-duration teacher as well as long student self-rollouts. Implemented as a 14B-parameter model and trained on a curated Unreal Engine-rendered dataset, RELIC achieves real-time generation at 16 FPS while demonstrating more accurate action following, more stable long-horizon streaming, and more robust spatial-memory retrieval compared with prior work. These capabilities establish RELIC as a strong foundation for the next generation of interactive world modeling.

---

## 4. PSA: Pyramid Sparse Attention for Efficient Video Understanding and Generation

**论文链接:** [http://arxiv.org/abs/2512.04025v1](http://arxiv.org/abs/2512.04025v1)

**作者:** Xiaolong Li, Youping Gu, Xi Lin, Weijie Wang, Bohan Zhuang

**发布时间:** 2025-12-03

**备注:** Tech report

### GPT解析

### 总结

本文提出了一种名为金字塔稀疏注意力(PSA)的高效注意力机制，通过多级池化的KV表示替代传统的二元掩码，实现了在保持计算效率的同时减少信息损失。

### 背景

注意力机制是基础模型的核心，但其二次复杂度成为扩展规模的关键瓶颈。高效注意力机制的发展因此受到推动，稀疏性已成为主要范式，但当前方法在高稀疏度下会导致大量信息损失。

### 目的

缓解现有稀疏注意力方法在高稀疏度下的信息损失问题，提出一种适用于视频理解和生成任务的通用模块。

### 方法

PSA引入多级池化的KV表示，每个查询块动态分配较低池化级别给关键KV块和较高池化级别给不太重要的块，创建完整保留和完全修剪之间的信息插值。该方法类似于定点量化和计算机视觉中的特征金字塔网络，使用硬件友好的内核和解耦块瓦片设计确保高效执行。

### 主要发现

PSA在视频理解和生成基准测试中保留了上下文信息和视觉保真度，比现有稀疏注意力基线有更好或相当的性能，实现了更优的效率-质量权衡。

### 结论

PSA有效减轻了信息损失，同时在低计算预算下保持计算效率，代码和模型权重已公开。

### 翻译

注意力机制是基础模型的核心，但其二次复杂度仍然是扩展规模的关键瓶颈。这一挑战推动了高效注意力机制的发展，稀疏性已成为主要范式。当前方法通常使用二元掩码保留或丢弃整个键值块，在高稀疏度下导致大量信息损失。为了缓解这一差距，我们提出了金字塔稀疏注意力(PSA)，一个适用于视频理解和生成任务的通用模块。PSA使用多级池化的KV表示而非二元掩码，实现更细粒度的掩码粒度。具体而言，每个查询块动态分配较低池化级别给关键KV块，较高池化级别给不太重要的块，创建完整保留和完全修剪之间的信息插值。这种设计类似于计算机视觉中的定点量化和经典特征金字塔网络，在低计算预算下有效减轻信息损失同时保持计算效率。PSA使用原生、硬件友好的内核，利用解耦块瓦片设计确保高效执行。在视频理解和生成基准测试中，PSA保留了上下文信息和视觉保真度，始终优于或达到与现有稀疏注意力基线相当的性能，具有更优的效率-质量权衡。我们的代码和模型权重可在http://ziplab.co/PSA公开获取。


### 论文摘要

Attention mechanisms are the core of foundation models, but their quadratic complexity remains a critical bottleneck for scaling. This challenge has driven the development of efficient attention mechanisms, with sparsity emerging as the dominant paradigm. Current methods typically retain or discard entire key-value blocks with binary masks, resulting in substantial information loss under high sparsity. To mitigate this gap, we present Pyramid Sparse Attention (PSA), a versatile module applicable to both video understanding and generation tasks. Instead of binary masking, PSA introduces multi-level pooled KV representations, enabling finer mask granularity. Specifically, each query block dynamically allocates lower pooling levels to critical KV blocks and higher levels to less important ones, creating an informative interpolation between full retention and complete pruning. This design, analogous to fixed-point quantization and classical feature pyramid networks in computer vision, effectively mitigates information loss while preserving computational efficiency under a low compute budget. It works with a native, hardware-friendly kernel that leverages decoupled block-tile design to ensure efficient execution. Across video understanding and generation benchmarks, PSA preserves contextual information and visual fidelity, consistently outperforming or achieving comparable performance over existing sparse attention baselines with superior efficiency-quality trade-offs. Our code and model weights are publicly available at: http://ziplab.co/PSA

---

## 5. Divide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.04000v1](http://arxiv.org/abs/2512.04000v1)

**作者:** Jialuo Li, Bin Li, Jiahao Li, Yan Lu

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为DIG的无需训练的帧选择框架，能够根据查询类型自适应选择帧，解决了大型多模态模型在长视频理解中面临的上下文长度限制和计算成本高的问题。

### 背景

大型多模态模型(LMMs)在长视频理解中应用受限，主要受限于上下文长度和处理密集视频令牌的高计算成本。最近研究集中在感知查询的帧选择方法上，但这些方法通常需要大量计算开销。

### 目的

质疑复杂搜索机制的普遍必要性，提出一种更高效的帧选择方法，针对不同类型查询优化长视频理解性能。

### 方法

识别并验证区分全局查询和局部查询的查询类型学；提出DIG框架，对全局查询采用均匀采样，对局部查询激活专门管道提取查询相关帧；无需训练即可自适应调整策略。

### 主要发现

均匀采样对全局查询既有效又高效；局部查询确实需要感知查询的选择才能获得最佳性能；DIG在三个基准测试上优于现有基线，并能稳健提高LMM性能，即使输入帧数扩展到256。

### 结论

DIG是一种有效的自适应帧选择框架，能够根据查询类型选择最优策略，显著提高长视频理解的效率和效果，解决了LMM在长视频处理中的关键挑战。

### 翻译

将大型多模态模型(LMMs)应用于长视频理解受到有限上下文长度和处理密集视频令牌计算成本高昂的限制。因此，最近的研究集中在感知查询的帧选择方法上，这些方法通常会产生大量计算开销。本文质疑了这类复杂搜索机制普遍必要的假设。我们首先识别并验证了区分全局查询和局部查询的查询类型学。我们证明，对于全局查询，均匀采样既有效又高效；而对于局部查询，确实需要感知查询的选择才能获得最佳性能。基于这一见解，我们提出了DIG，一个无需训练的帧选择框架，能根据查询类型调整其策略。具体来说，DIG对全局查询采用高效的均匀采样，同时对局部查询激活专门管道来提取查询相关帧。在三个长视频理解基准测试上的实验表明，DIG始终优于现有基线，并且能够稳健地提高LMM性能，即使将输入帧数扩展到256。


### 论文摘要

The application of Large Multimodal Models (LMMs) to long-form video understanding is constrained by limited context lengths and the computationally prohibitive cost of processing dense video tokens. Consequently, recent research has focused on query-aware frame selection, methods that often incur significant computational overhead. This paper challenges the assumption that such complex search mechanisms are universally necessary. We first identify and validate a query typology distinguishing between global query and localized query. We demonstrate that while uniform sampling is both effective and efficient for global queries, localized queries indeed necessitate query-aware selection for optimal performance. Building on this insight, we propose DIG, a training-free frame selection framework that adapts its strategy based on the query type. Specifically,DIG employs efficient uniform sampling for global queries while activating a specialized pipeline to extract query-relevant frames for localized queries. Experiments on three long-form video understanding benchmarks demonstrate that DIG consistently outperforms existing baselines and robustly improves LMM performance, even when scaling the input frame count to 256.

---

## 6. Adapting Large Language Models to Low-Resource Tibetan: A Two-Stage Continual and Supervised Fine-Tuning Study

**论文链接:** [http://arxiv.org/abs/2512.03976v1](http://arxiv.org/abs/2512.03976v1)

**作者:** Lifeng Chen, Ryan Lai, Tianming Liu

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究提出了一种将Qwen2.5-3B大型语言模型适应藏语的两阶段方法，解决了低资源语言适应中的数据稀缺和跨语言漂移问题。

### 背景

将大型语言模型适应低资源语言仍然是一个重大挑战，主要由于数据稀缺和跨语言漂移问题。

### 目的

将Qwen2.5-3B模型适应藏语，这是一种形态丰富但资源匮乏的语言。

### 方法

采用两阶段适应方法：首先使用持续预训练(CPT)建立藏语语言基础，然后使用监督微调(SFT)进行任务和翻译专业化。

### 主要发现

困惑度从2.98降低到1.54；中文到藏语翻译质量显著提高(BLEU从0.046提高到0.261；chrF从2.2提高到6.6)；对Qwen3-4B的435层分析显示，适应主要集中在嵌入层和输出头，中后期MLP投影编码领域特定转换。

### 结论

CPT构建藏语语义流形，而SFT使任务对齐最小化表示干扰。这项研究首次提供了LLM适应藏语动力学的定量探索，并为将多语言基础模型扩展到低资源环境提供了一个开放、可复现的框架。

### 翻译

本研究首次提供了大型语言模型适应藏语动力学的定量探索，并为将多语言基础模型扩展到低资源环境提供了一个开放、可复现的框架。


### 论文摘要

Adapting large language models (LLMs) to low-resource languages remains a major challenge due to data scarcity and cross-lingual drift. This work presents a two-stage adaptation of Qwen2.5-3B to Tibetan, a morphologically rich and underrepresented language. We employ Continual Pretraining (CPT) to establish Tibetan linguistic grounding, followed by Supervised Fine-Tuning (SFT) for task and translation specialization. Empirical evaluations demonstrate a consistent decrease in perplexity (from 2.98 $\rightarrow$ 1.54) and substantial improvements in Chinese$\rightarrow$Tibetan translation quality (BLEU: 0.046 $\rightarrow$ 0.261; chrF: 2.2 $\rightarrow$ 6.6). Layer-wise analysis across 435 layers in Qwen3-4B reveals that adaptation primarily concentrates on embedding and output heads, with mid--late MLP projections encoding domain-specific transformations. Our findings suggest that CPT constructs a Tibetan semantic manifold while SFT sharpens task alignment with minimal representational disruption. This study provides the first quantitative exploration of Tibetan adaptation dynamics for LLMs, and offers an open, reproducible framework for extending multilingual foundation models to low-resource settings.

---

## 7. Refining Machine Learning Potentials through Thermodynamic Theory of Phase Transitions

**论文链接:** [http://arxiv.org/abs/2512.03974v1](http://arxiv.org/abs/2512.03974v1)

**作者:** Paul Fuchs, Julija Zavadlav

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种自上而下的微调策略，通过可微分轨迹重加权算法校正基础机器学习势能函数的相变温度预测，使其与实验数据精确匹配。

### 背景

基础机器学习势能函数虽能解决经典力场的准确性和可转移性限制，并加速材料设计，但其参考数据不够广泛且存在系统性偏差，导致相变温度预测常与实验值有数百开尔文的偏差。

### 目的

开发一种微调策略，直接纠正错误的相变温度预测，使其与实验参考数据精确匹配。

### 方法

利用可微分轨迹重加权算法，最小化在实验目标压力和温度下各相之间的自由能差。

### 主要发现

该方法能准确校正纯钛在高达5 GPa压力范围内的相图，匹配精度达十分之几开尔文，并改进了液态扩散常数；该方法与模型无关，适用于多组分系统，且符合对其他实验属性进行自上而下训练的要求。

### 结论

该方法是实现高精度专用和基础机器学习势能函数的重要步骤。

### 翻译

基础机器学习势能函数可以解决经典力场的准确性和可转移性限制。它们通过分子动力学模拟使人们能够深入了解材料行为，从而 crucially 加速材料设计和发现。然而，不够广泛且存在系统性偏差的参考数据会影响学习模型的预测质量。这些模型在预测相变温度时常常与实验观察值有数百开尔文的偏差。因此，在许多实际问题中，微调是必要的。本文提出了一种通过自上而下学习的微调策略，直接纠正错误的相变温度预测，使其与实验参考数据匹配。我们的方法利用可微分轨迹重加权算法，最小化在实验目标压力和温度下各相之间的自由能差。我们证明，该方法能够准确校正纯钛在高达5 GPa压力范围内的相图，与实验参考值的匹配精度达十分之几开尔文，并改进了液态扩散常数。该方法与模型无关，适用于具有固-固和固-液转变的多组分系统，并符合对其他实验属性进行自上而下训练的要求。因此，该方法可以作为实现高精度专用和基础机器学习势能函数的重要步骤。


### 论文摘要

Foundational Machine Learning Potentials can resolve the accuracy and transferability limitations of classical force fields. They enable microscopic insights into material behavior through Molecular Dynamics simulations, which can crucially expedite material design and discovery. However, insufficiently broad and systematically biased reference data affect the predictive quality of the learned models. Often, these models exhibit significant deviations from experimentally observed phase transition temperatures, in the order of several hundred kelvins. Thus, fine-tuning is necessary to achieve adequate accuracy in many practical problems. This work proposes a fine-tuning strategy via top-down learning, directly correcting the wrongly predicted transition temperatures to match the experimental reference data. Our approach leverages the Differentiable Trajectory Reweighting algorithm to minimize the free energy differences between phases at the experimental target pressures and temperatures. We demonstrate that our approach can accurately correct the phase diagram of pure Titanium in a pressure range of up to 5 GPa, matching the experimental reference within tenths of kelvins and improving the liquid-state diffusion constant. Our approach is model-agnostic, applicable to multi-component systems with solid-solid and solid-liquid transitions, and compliant with top-down training on other experimental properties. Therefore, our approach can serve as an essential step towards highly accurate application-specific and foundational machine learning potentials.

---

## 8. Approximate Optimal Active Learning of Decision Trees

**论文链接:** [http://arxiv.org/abs/2512.03971v1](http://arxiv.org/abs/2512.03971v1)

**作者:** Zunchen Huang, Chenglu Jin

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究提出了一种使用成员查询主动学习未知二叉决策树的符号方法，通过将有界深度决策树的整个空间符号编码为SAT公式，并使用近似模型计数实现近似最优查询选择。

### 背景

在主动学习二叉决策树的场景中，学习者需要在大假设空间中进行推理并保持形式保证，传统的枚举候选树或依赖启发式纯度/熵度量的方法存在局限性。

### 目的

开发一种无需枚举候选树或依赖启发式方法的主动学习算法，能够在保持形式保证的同时高效学习二叉决策树。

### 方法

将有界深度决策树的整个空间符号编码为SAT公式，使用近似模型计数(ApproxMC)估计每个潜在查询导致的假设空间减少，实现近似最优查询选择；学习者根据查询结果逐步增强CNF表示，并在ApproxMC停滞时执行功能等价性检查。

### 主要发现

实验表明该方法仅使用少量查询就能可靠收敛到正确模型，同时保留了基于SAT的严格基础，适合形式分析和验证。

### 结论

通过符号编码和近似模型计数相结合的方法，可以有效解决主动学习二叉决策树的问题，在保证形式正确性的同时实现高效学习。

### 翻译

我们考虑仅使用成员查询主动学习未知二叉决策树的问题，在这种设定中，学习者必须在大假设空间中进行推理并保持形式保证。我们不是枚举候选树或依赖启发式纯度或熵度量，而是将有界深度决策树的整个空间符号编码为SAT公式。我们提出了一种用于决策树主动学习的符号方法，其中使用近似模型计数来估计每个潜在查询导致的假设空间减少，从而无需完整模型枚举即可实现近似最优查询选择。所得的学习者根据观察到的查询结果逐步增强CNF表示，并调用近似模型计数器ApproxMC以可扩展且可靠的方式量化剩余的版本空间。此外，当ApproxMC停滞时，执行功能等价性检查以验证所有剩余假设在功能上相同。在决策树上的实验表明，该方法仅使用少量查询就能可靠收敛到正确模型，同时保留了适合形式分析和验证的基于SAT的严格基础。


### 论文摘要

We consider the problem of actively learning an unknown binary decision tree using only membership queries, a setting in which the learner must reason about a large hypothesis space while maintaining formal guarantees. Rather than enumerating candidate trees or relying on heuristic impurity or entropy measures, we encode the entire space of bounded-depth decision trees symbolically in SAT formulas. We propose a symbolic method for active learning of decision trees, in which approximate model counting is used to estimate the reduction of the hypothesis space caused by each potential query, enabling near-optimal query selection without full model enumeration. The resulting learner incrementally strengthens a CNF representation based on observed query outcomes, and approximate model counter ApproxMC is invoked to quantify the remaining version space in a sound and scalable manner. Additionally, when ApproxMC stagnates, a functional equivalence check is performed to verify that all remaining hypotheses are functionally identical. Experiments on decision trees show that the method reliably converges to the correct model using only a handful of queries, while retaining a rigorous SAT-based foundation suitable for formal analysis and verification.

---

## 9. Technical Report on Text Dataset Distillation

**论文链接:** [http://arxiv.org/abs/2512.03967v1](http://arxiv.org/abs/2512.03967v1)

**作者:** Keith Ando Ogawa, Bruno Lopes Yamamoto, Lucas Lauton de Alcantara, Victor Zacarias, Edson Bollis, Lucas Pellicer, Rosimeire Pereira Costa, Anna Helena Reali Costa, Artur Jordao

**发布时间:** 2025-12-03

### GPT解析

### 总结

文本数据集蒸馏是一种将大型数据集压缩成小型合成数据集的技术，尽管在视觉领域有广泛应用，但文本领域的相关工作仍较少且处于成熟阶段。该领域经历了从视觉领域方法适应性应用到独立研究方向的发展过程，已取得多项里程碑式进展，但仍面临基准标准化、克服文本离散性等挑战。

### 背景

数据集蒸馏技术旨在将大型数据集压缩成小型合成数据集，使其在训练过程中能产生相似结果。视觉领域有大量相关文献，而文本数据集蒸馏研究相对较少。文本数据集蒸馏最初是视觉领域方法的适应性应用，随着文本模态的特殊性成为明显障碍，该领域逐渐发展成为一个独立的研究分支。

### 目的

回顾文本数据集蒸馏的过去和最新进展，强调不同的蒸馏策略、关键贡献和一般性挑战，为该领域的研究提供系统性综述。

### 方法

文本数据集蒸馏领域的发展包括多项里程碑，如引入使用transformer模型的方法、生成离散合成文本技术，以及扩展到参数超过1B的仅解码器模型。这些方法代表了该领域技术演进的主要方向。

### 主要发现

尽管现代方法在文本数据集蒸馏领域取得了重大进展，但该领域仍处于成熟阶段，存在多项需要改进的方面：基准标准化、克服文本离散性的方法、处理复杂任务的能力，以及提供实际应用案例的明确示例。

### 结论

文本数据集蒸馏是一个正在发展的研究领域，从视觉领域的适应性应用逐渐发展为独立研究方向，已取得多项技术突破，但仍面临标准化、离散性处理等挑战，需要进一步研究和探索。

### 翻译

在视觉领域，数据集蒸馏作为一种技术出现，可以将大型数据集压缩成较小的合成数据集，在训练过程中表现出相似的结果。虽然图像数据有大量的蒸馏方法文献，但文本数据集蒸馏的相关工作相对较少。文本数据集蒸馏最初作为视觉领域工作的适应性应用而发展，随着模态特殊性成为明显障碍，它逐渐成为一个独立的研究分支。该领域的几个发展里程碑包括：引入使用transformer模型的方法、生成离散合成文本，以及扩展到参数超过1B的仅解码器模型。尽管现代方法取得了重大进展，该领域仍处于成熟阶段，在基准标准化、克服文本离散性、处理复杂任务以及提供实际应用实例方面仍有改进空间。在本报告中，我们回顾了文本数据集蒸馏的过去和最新进展，突出了不同的蒸馏策略、关键贡献和一般性挑战。


### 论文摘要

In the vision domain, dataset distillation arises as a technique to condense a large dataset into a smaller synthetic one that exhibits a similar result in the training process. While image data presents an extensive literature of distillation methods, text dataset distillation has fewer works in comparison. Text dataset distillation initially grew as an adaptation of efforts from the vision universe, as the particularities of the modality became clear obstacles, it rose into a separate branch of research. Several milestones mark the development of this area, such as the introduction of methods that use transformer models, the generation of discrete synthetic text, and the scaling to decoder-only models with over 1B parameters. Despite major advances in modern approaches, the field remains in a maturing phase, with room for improvement on benchmarking standardization, approaches to overcome the discrete nature of text, handling complex tasks, and providing explicit examples of real-world applications. In this report, we review past and recent advances in dataset distillation for text, highlighting different distillation strategies, key contributions, and general challenges.

---

## 10. TempR1: Improving Temporal Understanding of MLLMs via Temporal-Aware Multi-Task Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.03963v1](http://arxiv.org/abs/2512.03963v1)

**作者:** Tao Wu, Li Yang, Gen Zhan, Yiting Liao, Junlin Li, Deliang Fu, Li Zhang, Limin Wang

**发布时间:** 2025-12-03

### GPT解析

### 总结

TempR1是一个时间感知的多任务强化学习框架，通过系统性地增强多模态大语言模型(MLLMs)的时间理解能力，解决了现有方法在多样化时间理解场景中泛化能力有限的问题。

### 背景

增强MLLMs的时间理解对长视频分析至关重要，可支持时间定位、动作检测和时间敏感问答等任务。现有强化学习方法通常局限于有限任务类型和数据，限制了跨场景泛化能力。

### 目的

提出TempR1框架，系统性地增强MLLMs的时间理解能力，使其能够处理多样化的时间结构和语义。

### 方法

整理多任务语料库暴露模型于多样化时间结构；基于GRPO算法实现跨任务优化；将时间任务分为三类预测区间与真实实例对应关系，设计特定定位奖励；使模型能捕获细粒度时间依赖并适应不同时间模式。

### 主要发现

TempR1在多个基准测试中取得最先进性能；互补任务联合优化产生强大协同效应，增强泛化能力和单任务性能。

### 结论

TempR1为MLLMs中的时间推理建立了可扩展的、有原则的范式。

### 翻译

增强多模态大语言模型(MLLMs)的时间理解对于推进长视频分析至关重要，能够支持时间定位、动作检测和时间敏感问答等任务。虽然强化学习(RL)最近被探索用于改进时间推理，但现有方法通常局限于有限的任务类型和数据，限制了它们在多样化时间理解场景中的泛化能力。为应对这一挑战，我们提出了TempR1，一个时间感知的多任务强化学习框架，系统性地增强MLLMs的时间理解能力。我们整理了一个多任务语料库，使模型能够接触多样化的时间结构和语义，并基于组相对策略优化(GRPO)算法实现稳定有效的跨任务优化。具体而言，我们将时间任务分为预测区间与真实实例之间的三种对应类型，并为每种类型设计特定的定位奖励，使TempR1能够捕获细粒度的时间依赖性并适应不同的时间模式。大量实验表明，TempR1在多个基准测试中取得了最先进的性能。此外，互补任务的联合优化产生了强大的协同效应，增强了泛化能力和单任务性能，为MLLMs中的时间推理建立了可扩展的、有原则的范式。


### 论文摘要

Enhancing the temporal understanding of Multimodal Large Language Models (MLLMs) is essential for advancing long-form video analysis, enabling tasks such as temporal localization, action detection, and time-sensitive question answering. While reinforcement learning (RL) has recently been explored for improving temporal reasoning, existing approaches are often confined to limited task types and data, restricting their generalization across diverse temporal understanding scenarios. To address this challenge, we present TempR1, a temporal-aware multi-task reinforcement learning framework that systematically strengthens MLLMs' temporal comprehension. We curate a multi-task corpus that exposes the model to diverse temporal structures and semantics, and build upon the Group Relative Policy Optimization (GRPO) algorithm to achieve stable and effective cross-task optimization. Specifically, we categorize temporal tasks into three correspondence types between predicted intervals and ground-truth instances, and design tailored localization rewards for each, enabling TempR1 to capture fine-grained temporal dependencies and adapt to different temporal patterns. Extensive experiments demonstrate that TempR1 attains state-of-the-art performance across multiple benchmarks. Moreover, its joint optimization over complementary tasks yields a strong synergistic effect, enhancing both generalization and single-task performance, establishing a scalable and principled paradigm for temporal reasoning in MLLMs.

---

## 11. MUT3R: Motion-aware Updating Transformer for Dynamic 3D Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.03939v1](http://arxiv.org/abs/2512.03939v1)

**作者:** Guole Shen, Tianchen Deng, Xingrui Qin, Nailin Wang, Jianyu Wang, Yanbo Wang, Yongtao Chen, Hesheng Wang, Jingchuan Wang

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为MUT3R的无需训练的框架，用于解决动态场景中的3D重建问题，通过利用预训练变压器自身的注意力机制来抑制动态内容，提高时间一致性和相机姿态鲁棒性。

### 背景

现有的状态循环神经网络在静态3D重建方面取得了显著进展，但对运动引起的伪影仍然很脆弱，非刚性区域会破坏空间记忆和图像特征之间的注意力传播。

### 目的

解决动态场景中的3D重建问题，提高时间一致性和相机姿态鲁棒性，同时避免重新训练或微调模型。

### 方法

分析状态和图像令牌更新机制的内部行为，发现跨层聚合自注意力图会揭示动态区域被自然降低权重的模式；基于此，引入MUT3R框架，在推理期间将注意力导出的运动线索应用于变压器早期层来抑制动态内容，通过注意力级门控模块在动态区域的伪影传播前抑制其影响。

### 主要发现

预训练的变压器已经编码了运动线索但从未明确使用；跨层聚合自注意力图中动态区域被自然地降低权重，这提供了隐式的运动线索。

### 结论

MUT3R框架让预训练的变压器自行诊断运动线索并自我修正，稳定了流式场景中的几何推理，在多个动态基准测试中改进了时间一致性和相机姿态鲁棒性，为运动感知的流式重建提供了一条简单且无需训练的途径。

### 翻译

最近的有状态循环神经网络在静态3D重建方面取得了显著进展，但对运动引起的伪影仍然很脆弱，非刚性区域会破坏空间记忆和图像特征之间的注意力传播。通过分析状态和图像令牌更新机制的内部行为，我们发现跨层聚合自注意力图揭示了一致的模式：动态区域被自然地降低权重，暴露了预训练变压器已经编码但从未明确使用的隐式运动线索。受此观察启发，我们引入了MUT3R，一个无需训练的框架，在推理期间将注意力导出的运动线索应用于变压器的早期层来抑制动态内容。我们的注意力级门控模块在动态区域的伪影通过特征层次传播之前抑制其影响。值得注意的是，我们没有重新训练或微调模型；我们让预训练的变压器自行诊断其运动线索并自我修正。这种早期调节稳定了流式场景中的几何推理，并在多个动态基准测试中提高了时间一致性和相机姿态鲁棒性，为运动感知的流式重建提供了一条简单且无需训练的途径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决动态3D重建中的运动引起的伪影问题，即非刚性区域会破坏空间记忆和图像特征之间的注意力传播，导致重建质量下降。这个问题在AR/VR、自主导航、机器人等领域非常重要，因为这些应用需要在动态环境中重建3D场景，而传统方法在处理运动、遮挡和光照变化时表现脆弱。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析CUT3R模型的内部行为，发现预训练transformer的自注意力图中自然表现出运动敏感性，动态区域被自然降权。基于这一观察，作者设计了无需训练的框架，在推理过程中应用注意力派生的运动线索抑制动态内容。他们借鉴了CUT3R的流式设计，但不同于需要额外时间融合的Easi3R，MUT3R直接利用CUT3R状态令牌中的时间历史来获取稳定的运动线索。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用预训练transformer中自注意力图隐含的运动信息，在transformer早期层应用注意力门控机制抑制动态区域干扰。整体流程包括：1)从冻结的CUT3R解码器提取多层自注意力图；2)聚合这些图生成运动分数图；3)通过注意力级门控机制将运动分数作为软偏差注入早期解码器层；4)根据注意力方向选择性减弱不稳定查询或键；5)防止运动干扰向前传播，保持深层几何一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)无监督运动线索发现，利用预训练transformer隐式编码的运动显著性；2)无需训练的动态信息抑制，通过注意力门控机制实现；3)简洁有效的实现方式。相比之前工作，MUT3R不同于需要重新训练的MonST3R，保持模型权重固定；不同于需要额外时间融合的Easi3R，直接利用CUT3R状态令牌的时间历史；不同于其他流式方法，专注于早期层注意力抑制而非改变整体架构。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种无需训练的MUT3R框架，通过利用预训练transformer中隐含的运动线索和注意力门控机制，有效抑制了动态场景中的运动干扰，显著提高了3D重建的时间一致性和几何稳定性，而无需重新训练或修改模型架构。'}


### 论文摘要

Recent stateful recurrent neural networks have achieved remarkable progress on static 3D reconstruction but remain vulnerable to motion-induced artifacts, where non-rigid regions corrupt attention propagation between the spatial memory and image feature. By analyzing the internal behaviors of the state and image token updating mechanism, we find that aggregating self-attention maps across layers reveals a consistent pattern: dynamic regions are naturally down-weighted, exposing an implicit motion cue that the pretrained transformer already encodes but never explicitly uses. Motivated by this observation, we introduce MUT3R, a training-free framework that applies the attention-derived motion cue to suppress dynamic content in the early layers of the transformer during inference. Our attention-level gating module suppresses the influence of dynamic regions before their artifacts propagate through the feature hierarchy. Notably, we do not retrain or fine-tune the model; we let the pretrained transformer diagnose its own motion cues and correct itself. This early regulation stabilizes geometric reasoning in streaming scenarios and leads to improvements in temporal consistency and camera pose robustness across multiple dynamic benchmarks, offering a simple and training-free pathway toward motion-aware streaming reconstruction.

---

## 12. Probabilistic Foundations of Fuzzy Simplicial Sets for Nonlinear Dimensionality Reduction

**论文链接:** [http://arxiv.org/abs/2512.03899v1](http://arxiv.org/abs/2512.03899v1)

**作者:** Janis Keck, Lukas Silvester Barth, Fatemeh, Fahimi, Parvaneh Joharinad, Jürgen Jost

**发布时间:** 2025-12-03

**备注:** 47 pages (including appendix), 11 figures

### GPT解析

### 总结

本文提出了一种新框架，将模糊单纯复形解释为单纯复形上概率测度的边缘分布，为模糊单纯复形提供了统一的概率理论基础，明确了UMAP在此框架中的角色，并能够系统推导新的降维方法。

### 背景

模糊单纯复形在降维和流形学习中受到关注，特别是在UMAP算法中的应用。然而，它们通过代数拓扑工具定义，缺乏明确的概率解释，与这些领域常用的理论框架脱节。

### 目的

引入一个框架，将模糊单纯复形解释为单纯复形上概率测度的边缘分布，为模糊单纯复形提供概率理论基础，并据此开发新的嵌入方法。

### 方法

作者提出将模糊单纯复形视为单纯复形上概率测度的边缘分布。这一视角表明UMAP的模糊权重来自一个生成模型，该模型在随机尺度上采样Vietoris-Rips过滤，产生成对距离的累积分布函数。

### 主要发现

该框架将模糊单纯复形与面偏序集上的概率模型联系起来；阐明了在此设置下Kullback-Leibler散度与模糊交叉熵之间的关系；通过底层单纯复形上的布尔运算恢复了标准的t-范数和t-共范数；使用Čech过滤和三元组采样推广了UMAP。

### 结论

这种概率观点为模糊单纯复形提供了统一的概率理论基础，明确了UMAP在此框架中的角色，并能够系统地推导新的降维方法。

### 翻译

模糊单纯复形已成为降维和流形学习的关注对象，特别是在UMAP中的作用。然而，它们通过代数拓扑工具定义，缺乏明确的概率解释，与这些领域常用的理论框架脱节。在这项工作中，我们引入了一个框架，将模糊单纯复形解释为单纯复形上概率测度的边缘分布。特别是，这一视角表明UMAP的模糊权重来自一个生成模型，该模型在随机尺度上采样Vietoris-Rips过滤，产生成对距离的累积分布函数。更一般地，该框架将模糊单纯复形与面偏序集上的概率模型联系起来，阐明了在此设置下Kullback-Leibler散度与模糊交叉熵之间的关系，并通过底层单纯复形上的布尔运算恢复了标准的t-范数和t-共范数。然后，我们展示了如何从该框架推导出新的嵌入方法，并通过一个使用Čech过滤和三元组采样推广UMAP的例子来说明这一点。总之，这种概率观点为模糊单纯复形提供了统一的概率理论基础，明确了UMAP在此框架中的角色，并能够系统地推导新的降维方法。


### 论文摘要

Fuzzy simplicial sets have become an object of interest in dimensionality reduction and manifold learning, most prominently through their role in UMAP. However, their definition through tools from algebraic topology without a clear probabilistic interpretation detaches them from commonly used theoretical frameworks in those areas. In this work we introduce a framework that explains fuzzy simplicial sets as marginals of probability measures on simplicial sets. In particular, this perspective shows that the fuzzy weights of UMAP arise from a generative model that samples Vietoris-Rips filtrations at random scales, yielding cumulative distribution functions of pairwise distances. More generally, the framework connects fuzzy simplicial sets to probabilistic models on the face poset, clarifies the relation between Kullback-Leibler divergence and fuzzy cross-entropy in this setting, and recovers standard t-norms and t-conorms via Boolean operations on the underlying simplicial sets. We then show how new embedding methods may be derived from this framework and illustrate this on an example where we generalize UMAP using Čech filtrations with triplet sampling. In summary, this probabilistic viewpoint provides a unified probabilistic theoretical foundation for fuzzy simplicial sets, clarifies the role of UMAP within this framework, and enables the systematic derivation of new dimensionality reduction methods.

---

## 13. Parameter efficient hybrid spiking-quantum convolutional neural network with surrogate gradient and quantum data-reupload

**论文链接:** [http://arxiv.org/abs/2512.03895v1](http://arxiv.org/abs/2512.03895v1)

**作者:** Luu Trong Nhan, Luu Trung Duong, Pham Ngoc Nam, Truong Cong Thang

**发布时间:** 2025-12-03

**备注:** Work under review

### GPT解析

### 总结

本研究提出了一种新的尖峰-量子数据重上传卷积神经网络(SQDR-CNN)架构，实现了卷积SNN和量子电路在单一反向传播框架内的联合训练，无需依赖预训练的尖峰编码器和数据子集即可达到合理性能。

### 背景

人工智能和深度学习的快速发展催生了几种优化驱动的子领域，特别是神经形态计算和量子机器学习。研究人员利用混合模型的可微分性，通过统一的优化策略解决复杂问题。

### 目的

解决现有尖峰量子神经网络(SQNN)实现通常依赖预训练SNN的问题，提出一种能够联合训练卷积SNN和量子电路的新架构。

### 方法

设计SQDR-CNN架构，阐明理论基础，测试使用量子数据重上传的新设计，采用不同的训练算法-初始化方法，在模拟的嘈杂量子环境中评估模型性能。

### 主要发现

该模型能够达到最先进的SNN基线平均最佳准确率的86%，但仅使用最小尖峰模型0.5%的参数。

### 结论

通过整合神经形态和量子范式，为多模态、可学习系统开辟新的研究方向并促进技术进步。

### 翻译

人工智能和深度学习的快速发展催生了几种优化驱动的子领域，特别是神经形态计算和量子机器学习。利用混合模型的可微分性，研究人员探索了通过统一优化策略解决复杂问题的潜力。尖峰量子神经网络(SQNN)就是这样的发展之一，它结合了尖峰神经网络(SNN)和量子计算的原则。然而，现有的SQNN实现通常依赖于预训练的SNN，这是由于尖峰活动的非可微分性质和当前SNN编码器的有限可扩展性。在本工作中，我们提出了一种新颖的架构——尖峰-量子数据重上传卷积神经网络(SQDR-CNN)，它能够在单一反向传播框架内实现卷积SNN和量子电路的联合训练。与它的前身不同，SQDR-CNN无需依赖预训练的尖峰编码器和数据子集即可收敛到合理性能。我们还阐明了一些理论基础，测试了使用量子数据重上传的新设计，并采用不同的训练算法-初始化方法，在模拟的嘈杂量子环境中评估了所提出模型的性能。结果表明，我们能够达到最先进的SNN基线平均最佳准确率的86%，而仅使用最小尖峰模型0.5%的参数。通过整合神经形态和量子范式，我们旨在为多模态、可学习系统开辟新的研究方向并促进技术进步。


### 论文摘要

The rapid advancement of artificial intelligence (AI) and deep learning (DL) has catalyzed the emergence of several optimization-driven subfields, notably neuromorphic computing and quantum machine learning. Leveraging the differentiable nature of hybrid models, researchers have explored their potential to address complex problems through unified optimization strategies. One such development is the Spiking Quantum Neural Network (SQNN), which combines principles from spiking neural networks (SNNs) and quantum computing. However, existing SQNN implementations often depend on pretrained SNNs due to the non-differentiable nature of spiking activity and the limited scalability of current SNN encoders. In this work, we propose a novel architecture, Spiking-Quantum Data Re-upload Convolutional Neural Network (SQDR-CNN), that enables joint training of convolutional SNNs and quantum circuits within a single backpropagation framework. Unlike its predecessor, SQDR-CNN allow convergence to reasonable performance without the reliance of pretrained spiking encoder and subsetting datasets. We also clarified some theoretical foundations, testing new design using quantum data-reupload with different training algorithm-initialization and evaluate the performance of the proposed model under noisy simulated quantum environments. As a result, we were able to achieve 86% of the mean top-performing accuracy of the SOTA SNN baselines, yet uses only 0.5% of the smallest spiking model's parameters. Through this integration of neuromorphic and quantum paradigms, we aim to open new research directions and foster technological progress in multi-modal, learnable systems.

---

## 14. Dual Cross-Attention Siamese Transformer for Rectal Tumor Regrowth Assessment in Watch-and-Wait Endoscopy

**论文链接:** [http://arxiv.org/abs/2512.03883v1](http://arxiv.org/abs/2512.03883v1)

**作者:** Jorge Tapias Gomez, Despoina Kanata, Aneesh Rangnekar, Christina Lee, Julio Garcia-Aguilar, Joshua Jesse Smith, Harini Veeraraghavan

**发布时间:** 2025-12-03

**备注:** 6 pages, 5 figures, 1 table, submitted to ISBI conference

### GPT解析

### 总结

本研究开发了一种具有双交叉注意力的Siamese Swin Transformer(SSDCA)模型，用于结合直肠癌患者在再分期和随访时的纵向内镜图像，区分临床完全缓解(cCR)和局部复发(LR)，实现了高准确率的早期检测。

### 背景

越来越多的证据支持对接受全新辅助治疗后达到临床完全缓解(cCR)的直肠癌患者采取观察等待(WW)策略，但需要客观准确的方法从随访内镜图像中早期检测局部复发(LR)。

### 目的

开发一种方法来结合再分期和随访时的纵向内镜图像，区分cCR和LR，以便管理治疗并防止远处转移。

### 方法

开发具有双交叉注意力的Siamese Swin Transformer(SSDCA)，利用预训练的Swin变换器提取领域无关特征，实现双交叉注意力机制强调两次扫描特征而不需要图像空间对齐，使用135名患者的图像对训练，并在62名患者的保留图像对上评估。

### 主要发现

SSDCA产生了最佳平衡准确率(81.76% ± 0.04)、敏感性(90.07% ± 0.08)和特异性(72.86% ± 0.05)；鲁棒性分析显示，无论存在血液、粪便、毛细血管扩张和图像质量差等伪影，性能都保持稳定；UMAP聚类显示SSDCA具有最大的簇间分离和最小的簇内离散，证实了判别性表示学习。

### 结论

SSDCA模型能够有效区分直肠癌患者的临床完全缓解和局部复发，在早期检测局部复发方面表现出色，有助于管理患者治疗并防止远处转移。

### 翻译

越来越多的证据支持对接受全新辅助治疗后达到临床完全缓解(cCR)的直肠癌患者采取观察等待(WW)策略。然而，在WW期间需要客观准确的方法从随访内镜图像中早期检测局部复发(LR)，以管理治疗并防止远处转移。因此，我们开发了具有双交叉注意力的Siamese Swin Transformer(SSDCA)，结合再分期和随访时的纵向内镜图像，区分cCR和LR。SSDCA利用预训练的Swin变换器提取领域无关特征，增强对成像变化的鲁棒性。实现双交叉注意力机制，强调两次扫描的特征，不需要图像的空间对齐来预测响应。使用135名患者的图像对训练SSDCA和基于Swin的基线模型，并在62名患者的保留图像对上评估。SSDCA产生了最佳的平衡准确率、敏感性和特异性。鲁棒性分析显示，无论存在血液、粪便、毛细血管扩张和图像质量差等伪影，性能都保持稳定。对提取的特征进行UMAP聚类显示，SSDCA具有最大的簇间分离和最小的簇内离散，证实了判别性表示学习。


### 论文摘要

Increasing evidence supports watch-and-wait (WW) surveillance for patients with rectal cancer who show clinical complete response (cCR) at restaging following total neoadjuvant treatment (TNT). However, objectively accurate methods to early detect local regrowth (LR) from follow-up endoscopy images during WW are essential to manage care and prevent distant metastases. Hence, we developed a Siamese Swin Transformer with Dual Cross-Attention (SSDCA) to combine longitudinal endoscopic images at restaging and follow-up and distinguish cCR from LR. SSDCA leverages pretrained Swin transformers to extract domain agnostic features and enhance robustness to imaging variations. Dual cross attention is implemented to emphasize features from the two scans without requiring any spatial alignment of images to predict response. SSDCA as well as Swin-based baselines were trained using image pairs from 135 patients and evaluated on a held-out set of image pairs from 62 patients. SSDCA produced the best balanced accuracy (81.76\% $\pm$ 0.04), sensitivity (90.07\% $\pm$ 0.08), and specificity (72.86\% $\pm$ 0.05). Robustness analysis showed stable performance irrespective of artifacts including blood, stool, telangiectasia, and poor image quality. UMAP clustering of extracted features showed maximal inter-cluster separation (1.45 $\pm$ 0.18) and minimal intra-cluster dispersion (1.07 $\pm$ 0.19) with SSDCA, confirming discriminative representation learning.

---

## 15. PULSE: A Unified Multi-Task Architecture for Cardiac Segmentation, Diagnosis, and Few-Shot Cross-Modality Clinical Adaptation

**论文链接:** [http://arxiv.org/abs/2512.03848v1](http://arxiv.org/abs/2512.03848v1)

**作者:** Hania Ghouse, Maryam Alsharqi, Farhad R. Nezami, Muzammil Behzad

**发布时间:** 2025-12-03

### GPT解析

### 总结

PULSE是一个多任务视觉-语言框架，能够统一心脏图像分析中的解剖分割、疾病分类和临床报告生成任务，并在单一架构中实现从像素到临床推理的过渡。

### 背景

心脏图像分析在不同任务中是分散的：解剖分割、疾病分类和基于临床的报告生成通常由单独的网络处理，这些网络在不同的数据条件下训练，缺乏统一的跨模态和数据集的框架。

### 目的

开发一个能够统一多种心脏图像分析任务的单一架构框架，确保对成像方式和数据集的泛化能力。

### 方法

PULSE基于自监督表示构建，使用复合监督策略平衡区域重叠学习、逐像素分类保真度和边界感知IoU细化；多尺度令牌重建解码器实现解剖分割，共享全局表示支持疾病分类和临床文本输出。

### 主要发现

PULSE能够学习任务不变的心脏先验知识，在数据集上具有强大的泛化能力，可用最少监督适应新的成像方式。

### 结论

PULSE推动心脏图像分析领域向可扩展的、基础风格的分析框架发展。

### 翻译

心脏图像分析在不同任务中仍然分散：解剖分割、疾病分类和基于临床的报告生成通常由在不同数据条件下训练的独立网络处理。目前没有现有框架能够在单一架构中统一这些目标，同时保持对成像方式和数据集的泛化能力。我们介绍了PULSE，一个基于自监督表示构建的多任务视觉-语言框架，通过复合监督策略进行优化，该策略平衡了区域重叠学习、逐像素分类保真度和边界感知IoU细化。多尺度令牌重建解码器实现解剖分割，而共享的全局表示支持疾病分类和基于临床的文本输出，使模型能够在单一架构中从像素过渡到结构，最后到临床推理。与先前的特定任务流水线不同，PULSE学习任务不变的心脏先验知识，在数据集上稳健泛化，并且可以用最少的监督适应新的成像方式。这使该领域更接近一个可扩展的、基础风格的心脏分析框架。


### 论文摘要

Cardiac image analysis remains fragmented across tasks: anatomical segmentation, disease classification, and grounded clinical report generation are typically handled by separate networks trained under different data regimes. No existing framework unifies these objectives within a single architecture while retaining generalization across imaging modalities and datasets. We introduce PULSE, a multi-task vision-language framework built on self-supervised representations and optimized through a composite supervision strategy that balances region overlap learning, pixel wise classification fidelity, and boundary aware IoU refinement. A multi-scale token reconstruction decoder enables anatomical segmentation, while shared global representations support disease classification and clinically grounded text output allowing the model to transition from pixels to structures and finally clinical reasoning within one architecture. Unlike prior task-specific pipelines, PULSE learns task-invariant cardiac priors, generalizes robustly across datasets, and can be adapted to new imaging modalities with minimal supervision. This moves the field closer to a scalable, foundation style cardiac analysis framework.

---

## 16. CoDA: From Text-to-Image Diffusion Models to Training-Free Dataset Distillation

**论文链接:** [http://arxiv.org/abs/2512.03844v1](http://arxiv.org/abs/2512.03844v1)

**作者:** Letian Zhou, Songhua Liu, Xinchao Wang

**发布时间:** 2025-12-03

**备注:** 34 pages, 24 figures

### GPT解析

### 总结

本文提出了Core Distribution Alignment (CoDA)框架，解决了数据集蒸馏方法中依赖目标数据集预训练生成模型和分布不匹配的问题，仅使用现成文本到图像模型就能实现高效数据集蒸馏，在ImageNet-1K上达到60.4%的最先进准确率。

### 背景

当前主流的数据集蒸馏方法利用生成模型面临两个基本限制：一是大多数方法需要预先在完整目标数据集上训练扩散模型，这与DD目的相悖且成本高昂；二是转向通用文本到图像模型的方法存在显著分布不匹配问题，因为网络规模先验无法忠实捕获目标特定语义。

### 目的

解决现有数据集蒸馏方法的两个主要限制，提出一种仅使用现成文本到图像模型实现有效数据集蒸馏的框架。

### 方法

提出Core Distribution Alignment (CoDA)框架，首先使用稳健的基于密度的发现机制识别目标数据集的'内在核心分布'，然后引导生成过程使生成的样本与此核心分布对齐，从而弥合通用生成先验与目标语义之间的差距。

### 主要发现

不依赖特定于目标数据集的生成模型，CoDA在所有基准测试(包括ImageNet-1K及其子集)上实现了与或优于先前依赖此类模型方法的性能；在ImageNet-1K的50-IPC设置下，CoDA建立了60.4%的新最先进准确率。

### 结论

CoDA框架解决了现有数据集蒸馏方法的两个主要限制，能够仅使用通用文本到图像模型实现高效的数据集蒸馏，并且在性能上达到了或超过了需要特定训练的先前方法。

### 翻译

当前主流的数据集蒸馏(DD)方法利用生成模型面临两个基本限制。首先，尽管开创性地在DD中使用扩散模型并取得了令人印象深刻的性能，但大多数方法仍然需要预先在完整目标数据集上训练扩散模型，这与DD的目的相悖，并且带来了高昂的训练成本。其次，虽然一些方法转向不依赖目标特定训练的通用文本到图像模型，但它们存在显著的分布不匹配问题，因为这些基础模型中包含的网络规模先验无法忠实捕获目标特定语义，导致性能不佳。为应对这些挑战，我们提出了核心分布对齐(CoDA)框架，该框架仅使用现成的文本到图像模型就能实现有效的DD。我们的核心思想是首先使用稳健的基于密度的发现机制识别目标数据集的'内在核心分布'，然后引导生成过程使生成的样本与此核心分布对齐。通过这样做，CoDA有效地弥合了通用生成先验与目标语义之间的差距，产生了高度代表性的蒸馏数据集。大量实验表明，在不依赖特定于目标数据集的生成模型的情况下，CoDA在所有基准测试(包括ImageNet-1K及其子集)上实现了与或优于先前依赖此类模型方法的性能。值得注意的是，在ImageNet-1K的每类50张图像(50-IPC)设置下，CoDA建立了60.4%的新最先进准确率。我们的代码可在项目网页获取：https://github.com/zzzlt422/CoDA


### 论文摘要

Prevailing Dataset Distillation (DD) methods leveraging generative models confront two fundamental limitations. First, despite pioneering the use of diffusion models in DD and delivering impressive performance, the vast majority of approaches paradoxically require a diffusion model pre-trained on the full target dataset, undermining the very purpose of DD and incurring prohibitive training costs. Second, although some methods turn to general text-to-image models without relying on such target-specific training, they suffer from a significant distributional mismatch, as the web-scale priors encapsulated in these foundation models fail to faithfully capture the target-specific semantics, leading to suboptimal performance. To tackle these challenges, we propose Core Distribution Alignment (CoDA), a framework that enables effective DD using only an off-the-shelf text-to-image model. Our key idea is to first identify the "intrinsic core distribution" of the target dataset using a robust density-based discovery mechanism. We then steer the generative process to align the generated samples with this core distribution. By doing so, CoDA effectively bridges the gap between general-purpose generative priors and target semantics, yielding highly representative distilled datasets. Extensive experiments suggest that, without relying on a generative model specifically trained on the target dataset, CoDA achieves performance on par with or even superior to previous methods with such reliance across all benchmarks, including ImageNet-1K and its subsets. Notably, it establishes a new state-of-the-art accuracy of 60.4% at the 50-images-per-class (IPC) setup on ImageNet-1K. Our code is available on the project webpage: https://github.com/zzzlt422/CoDA

---

## 17. From Micro-Distributions to Macro-Regularities: A Critique and Reconstruction of the Production Function Based on the Maximum Entropy Principle

**论文链接:** [http://arxiv.org/abs/2512.03812v1](http://arxiv.org/abs/2512.03812v1)

**作者:** Jihyuan Liuh

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文基于统计物理学为柯布-道格拉斯生产函数提供了微观基础，并批判了其政治经济含义。通过引入最大熵原理和尺度不变性公理，证明了在不完全信息的经济系统中，微观技术系数的最无偏分布呈现截断幂律形式，统计聚合自然导致宏观层面规模收益不变的柯布-道格拉斯函数的出现。研究揭示了聚合生产函数是微观信息的有损压缩，其中社会历史关系被'自然化'为技术定律，体现了马克思对'拜物教'的批判。

### 背景

新古典经济学中的柯布-道格拉斯生产函数缺乏微观基础，其政治经济含义需要批判性分析。研究基于统计物理学视角，考虑不完全信息的经济系统。

### 目的

为柯布-道格拉斯生产函数提供基于统计物理学的微观基础，批判其政治经济含义，深化对生产函数作为统计现象而非技术定律的理解。

### 方法

引入最大熵原理和尺度不变性公理，证明微观技术系数分布形式，通过统计聚合推导宏观生产函数，并与马克思、剑桥学派和Shaikh的理论进行对话。

### 主要发现

1) 微观层面技术系数呈现截断幂律分布；2) 统计聚合自然导致宏观柯布-道格拉斯函数出现；3) 聚合生产函数是微观信息的有损压缩；4) 社会历史关系在压缩过程中被'自然化'为技术定律。

### 结论

研究为不依赖代表性主体或资本价值聚合的新古典增长模型提供了微观基础，揭示了生产函数的统计本质，展示了马克思'拜物教'批判在数学逻辑层面的体现。

### 翻译

本文旨在基于统计物理学为柯布-道格拉斯生产函数提供微观基础，并对其政治经济含义展开批判。通过引入最大熵原理和尺度不变性公理，我们证明在具有不完全信息的经济系统中，微观层面技术系数的最无偏分布必须采取截断幂律的形式。基于这一点，统计聚合自然导致宏观层面规模收益不变的柯布-道格拉斯函数的出现。这一结果不仅为不依赖于代表性主体或资本价值聚合的新古典增长模型提供了微观基础，更重要的是揭示了聚合生产函数本质上是微观信息的有损压缩。在这一压缩过程中，嵌入在分配参数中的社会历史关系被'自然化'为看似永恒的技术定律，这是马克思对'拜物教'批判在数学逻辑层面的体现。本文通过与马克思、剑桥学派和Shaikh的对话，进一步深化了对生产函数作为统计现象而非技术定律的理解。


### 论文摘要

This paper aims to provide a micro-foundation for the Cobb-Douglas production function based on statistical physics, and to launch a critique of its political-economic implications. By introducing the Maximum Entropy Principle and an axiom of scale invariance, we prove that in an economic system with incomplete information, the most unbiased distribution of micro-level technical coefficients must take the form of a truncated power law. Based on this, statistical aggregation naturally leads to the emergence of a constant-returns-to-scale Cobb-Douglas function at the macro level. This result not only provides a micro-foundation for neoclassical growth models that does not rely on a representative agent or value aggregation of capital but, more importantly, reveals that the aggregate production function is essentially a lossy compression of micro-level information. In this compression process, the social-historical relations embedded in distribution parameters are 'naturalized' into seemingly eternal technical laws, which is the manifestation of Marx's critique of 'fetishism' at the level of mathematical logic. This paper further deepens the understanding of the production function as a statistical phenomenon rather than a technical law through dialogues with Marx, the Cambridge School, and Shaikh.

---

## 18. Algorithms for Boolean Matrix Factorization using Integer Programming and Heuristics

**论文链接:** [http://arxiv.org/abs/2512.03807v1](http://arxiv.org/abs/2512.03807v1)

**作者:** Christos Kolomvakis, Thomas Bobille, Arnaud Vandaele, Nicolas Gillis

**发布时间:** 2025-12-03

**备注:** 24 pages, 12 tables, 3 figures, code and data available from https://gitlab.com/ckolomvakis/boolean-matrix-factorization-ip-and-heuristics

### GPT解析

### 总结

本文提出了一种改进的布尔矩阵分解方法，包括基于交替优化的算法、选择最优秩一因子的策略、贪心和局部搜索启发式方法，以及高效的C++数据结构，并在多个真实数据集上验证了其性能。

### 背景

布尔矩阵分解(BMF)将二元输入矩阵近似为两个较小二元因子的乘积，使用布尔OR和AND运算而非标准算术，提高可解释性并减少近似误差。BMF应用于角色挖掘和计算机视觉等领域。

### 目的

开发高效的BMF算法，解决可扩展性问题，构建快速数据结构，并在有缺失数据和无缺失数据的真实数据集上评估性能。

### 方法

提出基于交替优化的BMF算法，使用整数规划解决子问题；设计从多次运行中选择最优秩一因子的方法；引入贪心和局部搜索启发式算法；构建高效的C++数据结构用于布尔向量和矩阵操作。

### 主要发现

所提出的BMF算法在处理各种真实数据集(包括有缺失数据和无缺失数据)时表现良好，在主题建模和成像等应用中优于或与最先进方法相当。

### 结论

新的算法和数据结构使BMF能够扩展到大型数据集，在多个应用领域具有实用价值。

### 翻译

布尔矩阵分解(BMF)将给定的二元输入矩阵近似为两个较小的二元因子的乘积。与基于标准算术的二元矩阵分解不同，BMF使用布尔OR和AND运算进行矩阵乘法，这提高了可解释性并减少了近似误差。它也用于角色挖掘和计算机视觉。在本文中，我们首先提出了一种用于BMF的算法，该算法对因子矩阵执行交替优化(AO)，其中每个子问题通过整数规划(IP)解决。然后，我们设计了不同的方法，通过从多次运行中选择一组最优的秩一因子来进一步增强基于AO的算法。为了解决基于IP方法的可扩展性限制，我们引入了新的贪心和局部搜索启发式算法。我们还构建了一个新的C++数据结构用于布尔向量和矩阵，它比现有结构快得多，并且具有独立的研究价值，使我们的启发式算法能够扩展到大型数据集。我们展示了所有提出方法的性能，并在各种真实数据集上(包括有缺失数据和无缺失数据的情况)将它们与最先进的方法进行了比较，这些应用包括主题建模和成像。


### 论文摘要

Boolean matrix factorization (BMF) approximates a given binary input matrix as the product of two smaller binary factors. Unlike binary matrix factorization based on standard arithmetic, BMF employs the Boolean OR and AND operations for the matrix product, which improves interpretability and reduces the approximation error. It is also used in role mining and computer vision. In this paper, we first propose algorithms for BMF that perform alternating optimization (AO) of the factor matrices, where each subproblem is solved via integer programming (IP). We then design different approaches to further enhance AO-based algorithms by selecting an optimal subset of rank-one factors from multiple runs. To address the scalability limits of IP-based methods, we introduce new greedy and local-search heuristics. We also construct a new C++ data structure for Boolean vectors and matrices that is significantly faster than existing ones and is of independent interest, allowing our heuristics to scale to large datasets. We illustrate the performance of all our proposed methods and compare them with the state of the art on various real datasets, both with and without missing data, including applications in topic modeling and imaging.

---

## 19. AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition

**论文链接:** [http://arxiv.org/abs/2512.03794v1](http://arxiv.org/abs/2512.03794v1)

**作者:** Zichuan Lin, Yicheng Liu, Yang Yang, Lvfang Tao, Deheng Ye

**发布时间:** 2025-12-03

**备注:** 15 pages, 9 figures

### GPT解析

### 总结

本研究提出了一种名为AdaptVision的高效视觉语言模型范式，通过自适应视觉token获取机制，在保持高性能的同时显著减少视觉token的使用量。

### 背景

视觉语言模型在视觉问答任务中取得了显著成功，但依赖大量视觉token带来显著计算开销。现有高效方法通过固定比例压缩减少视觉token，但无法适应不同任务需求。

### 目的

探索视觉语言模型能否自主确定每个样本所需的最少视觉token数量。

### 方法

提出AdaptVision范式，采用从粗到细的方法实现自适应视觉token获取，初始处理压缩视觉token，必要时通过边界框工具获取额外信息。使用强化学习框架训练，核心是解耦回合策略优化(DTPO)，将学习目标解耦为工具学习和准确性改进两部分，并计算与每个目标相关联的token的单独优势。

### 主要发现

在多个VQA基准上的综合实验表明，AdaptVision实现了卓越性能，同时消耗的视觉token显著少于最先进的高效VLM方法。

### 结论

AdaptVision是一种有效的视觉语言模型范式，能够在保持高性能的同时显著减少视觉token的使用量，解决了现有方法被动压缩的问题。

### 翻译

视觉语言模型在视觉问答任务中取得了显著成功，但它们对大量视觉token的依赖带来了显著的计算开销。虽然现有高效VLM方法通过固定比例压缩减少视觉token，但它们是被动的，缺乏适应不同任务需求的能力。这促使我们思考一个基本问题：VLMs能否自主确定每个样本所需的最少视觉token数量？受人类主动视觉机制的启发，我们引入了AdaptVision，这是一种高效VLM范式，通过从粗到细的方法实现自适应视觉token获取。我们的模型初始处理来自低分辨率图像的压缩视觉token，并在必要时通过调用边界框工具来裁剪关键区域以获取额外的视觉信息。我们使用强化学习框架训练AdaptVision，仔细平衡准确性和效率。我们方法的核心是解耦回合策略优化(DTPO)，它将学习目标解耦为两个部分：(1)工具学习，优化正确工具的使用；(2)准确性改进，完善生成的回答以提高答案正确性。基于这一公式，我们通过计算与每个目标相关联的token的单独优势来进一步解耦优势估计。与普通GRPO相比，这种公式使AdaptVision的优化更加有效。在多个VQA基准上的综合实验表明，AdaptVision实现了卓越性能，同时消耗的视觉token远少于最先进的高效VLM方法。


### 论文摘要

Vision-Language Models (VLMs) have achieved remarkable success in visual question answering tasks, but their reliance on large numbers of visual tokens introduces significant computational overhead. While existing efficient VLM approaches reduce visual tokens through fixed-ratio compression, they operate passively and lack the ability to adapt to varying task requirements. This motivates a fundamental question: Can VLMs autonomously determine the minimum number of visual tokens required for each sample? Inspired by human active vision mechanisms, we introduce AdaptVision, an efficient VLM paradigm that enables adaptive visual token acquisition through a coarse-to-fine approach. Our model initially processes compressed visual tokens from low-resolution images and selectively acquires additional visual information by invoking a bounding box tool to crop key regions when necessary. We train AdaptVision using a reinforcement learning framework that carefully balances accuracy and efficiency. Central to our approach is Decoupled Turn Policy Optimization (DTPO), which decouples the learning objective into two components: (1) tool learning, which optimizes correct tool utilization, and (2) accuracy improvement, which refines the generated responses to improve answer correctness. Based on this formulation, we further decouple advantage estimation by computing separate advantages for tokens associated with each objective. This formulation enables more effective optimization for AdaptVision compared to vanilla GRPO. Comprehensive experiments across multiple VQA benchmarks demonstrate that AdaptVision achieves superior performance while consuming substantially fewer visual tokens than state-of-the-art efficient VLM methods.

---

## 20. "MCP Does Not Stand for Misuse Cryptography Protocol": Uncovering Cryptographic Misuse in Model Context Protocol at Scale

**论文链接:** [http://arxiv.org/abs/2512.03775v1](http://arxiv.org/abs/2512.03775v1)

**作者:** Biwei Yan, Yue Zhang, Minghui Xu, Hao Wu, Yechao Zhang, Kun Li, Guoming Zhang, Xiuzhen Cheng

**发布时间:** 2025-12-03

### GPT解析

### 总结

MICRYSCOPE是首个专门用于检测MCP实现中加密误用的领域特定框架，通过跨语言中间表示、混合依赖分析和基于污点的误用检测器来识别安全问题。

### 背景

MCP作为LLM应用的中间件正在快速发展，但其内置安全机制非常有限，无法保证真实性和保密性，迫使开发者自行实现加密，这种临时做法历史上容易被误用。

### 目的

开发MICRYSCOPE框架来检测MCP实现中的加密误用，提高MCP生态系统的安全性，解决其安全机制不足的问题。

### 方法

MICRYSCOPE结合三个关键创新：跨语言中间表示规范化不同生态系统中的加密API；混合依赖分析揭示显式和隐式函数关系；基于污点的误用检测器跟踪敏感数据流并标记加密规则违规。

### 主要发现

在9,403个MCP服务器中，发现720个包含加密逻辑，其中19.7%存在误用；这些缺陷集中在特定市场(如Smithery Registry)、语言(Python)和类别(开发者工具和数据科学与机器学习)；案例研究显示真实世界后果包括泄露API密钥和不安全加密工具。

### 结论

该研究首次建立了MCP中加密误用的生态系统级视图，提供了工具和见解，以加强这一快速增长协议的安全基础。

### 翻译

模型上下文协议(MCP)正在迅速成为基于LLM的应用程序的中间件，为工具集成提供标准化接口。然而，其内置的安全机制非常有限：虽然模式和声明可以防止格式错误的请求，但MCP无法保证真实性和保密性，迫使开发者自己实现加密。这种临时做法历史上容易被误用，在MCP中威胁敏感数据和服务的安全。我们提出了MICRYSCOPE，这是首个用于检测MCP实现中加密误用的领域特定框架。MICRYSCOPE结合了三个关键创新：一种跨语言中间表示，规范化不同生态系统中的加密API；一种混合依赖分析，揭示显式和隐式函数关系（包括由LLM编排的不安全运行时组合）；以及一种基于污点的误用检测器，跟踪敏感数据流并标记已建立的加密规则违规。将MICRYSCOPE应用于9,403个MCP服务器，我们发现720个包含加密逻辑，其中19.7%存在误用。这些缺陷集中在某些市场（例如Smithery Registry，42%的服务器不安全）、语言（Python，34%的误用率）和类别（开发者工具和数据科学与机器学习占所有误用的50%以上）。案例研究揭示了真实世界的后果，包括泄露的API密钥、不安全的DES/ECB工具和基于MD5的身份验证绕过。我们的研究首次建立了MCP中加密误用的生态系统级视图，并提供了工具和见解，以加强这一快速增长协议的安全基础。


### 论文摘要

The Model Context Protocol (MCP) is rapidly emerging as the middleware for LLM-based applications, offering a standardized interface for tool integration. However, its built-in security mechanisms are minimal: while schemas and declarations prevent malformed requests, MCP provides no guarantees of authenticity or confidentiality, forcing developers to implement cryptography themselves. Such ad hoc practices are historically prone to misuse, and within MCP they threaten sensitive data and services. We present MICRYSCOPE, the first domain-specific framework for detecting cryptographic misuses in MCP implementations. MICRYSCOPE combines three key innovations: a cross-language intermediate representation that normalizes cryptographic APIs across diverse ecosystems, a hybrid dependency analysis that uncovers explicit and implicit function relationships (including insecure runtime compositions orchestrated by LLMs) and a taint-based misuse detector that tracks sensitive data flows and flags violations of established cryptographic rules. Applying MICRYSCOPE to 9,403 MCP servers, we identified 720 with cryptographic logic, of which 19.7% exhibited misuses. These flaws are concentrated in certain markets (e.g., Smithery Registry with 42% insecure servers), languages (Python at 34% misuse rate), and categories (Developer Tools and Data Science & ML accounting for over 50% of all misuses). Case studies reveal real-world consequences, including leaked API keys, insecure DES/ECB tools, and MD5-based authentication bypasses. Our study establishes the first ecosystem-wide view of cryptographic misuse in MCP and provides both tools and insights to strengthen the security foundations of this rapidly growing protocol.

---

## 21. Deep Unfolding: Recent Developments, Theory, and Design Guidelines

**论文链接:** [http://arxiv.org/abs/2512.03768v1](http://arxiv.org/abs/2512.03768v1)

**作者:** Nir Shlezinger, Santiago Segarra, Yi Zhang, Dvir Avrahami, Zohar Davidov, Tirza Routtenberg, Yonina C. Eldar

**发布时间:** 2025-12-03

**备注:** under review for publication in the IEEE

### GPT解析

### 总结

本研究探讨了深度展开（deep unfolding）框架，该框架通过将迭代优化算法转化为结构化的、可训练的机器学习架构，弥合了经典优化方法和机器学习之间的差距。文章提供了深度展开的教程式概述，介绍了相关方法论、设计范式和训练方案，并讨论了理论进展和实证研究。

### 背景

优化方法在信号处理中扮演核心角色，是推理、估计和控制的基础。经典迭代优化算法提供可解释性和理论保证，但依赖代理目标、需要仔细调整超参数，且计算延迟大。机器学习提供强大的数据驱动建模能力，但缺乏优化驱动推理所需的结构、透明度和效率。

### 目的

提供深度展开的教程式概述，提出将优化求解器转化为机器学习模型的方法论统一视角，强调其概念、理论和实践意义，介绍设计范式和训练方案，并讨论理论进展和实证研究。

### 方法

系统地将迭代优化算法转化为结构化的、可训练的机器学习架构。文章回顾了优化基础，介绍了四种代表性的深度展开设计范式，讨论了其迭代性质产生的特殊训练方案，并综述了建立展开优化器收敛性和泛化保证的理论进展。

### 主要发现

深度展开框架成功弥合了经典优化方法和机器学习之间的差距。文章通过定性和实证研究表明，展开优化器在复杂性、可解释性和鲁棒性方面具有特定的权衡，并建立了这些方法的收敛性和泛化保证。

### 结论

深度展开为优化驱动的推理提供了一种有前景的方法，结合了传统优化算法的可解释性和机器学习的强大建模能力。这一框架在信号处理和相关领域具有广泛的应用潜力，为优化和机器学习的融合提供了新视角。

### 翻译

优化方法在信号处理中扮演核心角色，作为推理、估计和控制的数学基础。虽然经典迭代优化算法提供了可解释性和理论保证，但它们通常依赖代理目标，需要仔细调整超参数，并表现出大量的计算延迟。相反，机器学习提供了强大的数据驱动建模能力，但缺乏优化驱动推理所需的结构、透明度和效率。深度展开最近出现了一个引人注目的框架，通过系统地将迭代优化算法转化为结构化的、可训练的机器学习架构，弥合了这两种范式之间的差距。本文提供了深度展开的教程式概述，提出了将优化求解器转化为机器学习模型的方法论统一视角，并强调了它们的概念、理论和实践意义。我们回顾了用于推理和学习的优化基础，介绍了深度展开的四种代表性设计范式，并讨论了其迭代性质产生的特殊训练方案。此外，我们还综述了最近的理论进展，建立了展开优化器的收敛性和泛化保证，并提供了比较性的定性和实证研究，说明了它们在复杂性、可解释性和鲁棒性方面的相对权衡。


### 论文摘要

Optimization methods play a central role in signal processing, serving as the mathematical foundation for inference, estimation, and control. While classical iterative optimization algorithms provide interpretability and theoretical guarantees, they often rely on surrogate objectives, require careful hyperparameter tuning, and exhibit substantial computational latency. Conversely, machine learning (ML ) offers powerful data-driven modeling capabilities but lacks the structure, transparency, and efficiency needed for optimization-driven inference. Deep unfolding has recently emerged as a compelling framework that bridges these two paradigms by systematically transforming iterative optimization algorithms into structured, trainable ML architectures. This article provides a tutorial-style overview of deep unfolding, presenting a unified perspective of methodologies for converting optimization solvers into ML models and highlighting their conceptual, theoretical, and practical implications. We review the foundations of optimization for inference and for learning, introduce four representative design paradigms for deep unfolding, and discuss the distinctive training schemes that arise from their iterative nature. Furthermore, we survey recent theoretical advances that establish convergence and generalization guarantees for unfolded optimizers, and provide comparative qualitative and empirical studies illustrating their relative trade-offs in complexity, interpretability, and robustness.

---

## 22. 论文ID: 2512.03750v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03750v1.json'

---

## 23. Thinking with Programming Vision: Towards a Unified View for Thinking with Images

**论文链接:** [http://arxiv.org/abs/2512.03746v1](http://arxiv.org/abs/2512.03746v1)

**作者:** Zirun Guo, Minjie Hong, Feng Zhang, Kai Jia, Tao Jin

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究提出了CodeVision，一个灵活可扩展的代码作为工具框架，解决了多模态大语言模型在图像处理中的脆弱性和工具限制问题，显著提高了模型性能并促进了新兴能力的出现。

### 背景

多模态大语言模型能够通过图像进行交互式工具使用来推理视觉输入，但当前方法通常依赖于一组有限的工具，这些工具在现实世界的必要性和可扩展性方面存在局限。

### 目的

解决当前MLLMs在面对图像方向变化或自然损坏时的脆弱性问题，提出一个更灵活、可扩展的框架，使模型能够生成代码作为调用任何图像操作的通用接口。

### 方法

提出CodeVision框架，采用两阶段训练方法：第一阶段在有质量保证的数据集上进行监督微调(SFT)，专注于复杂的多工具组合和错误恢复；第二阶段使用新颖的密集过程奖励函数进行强化学习(RL)，鼓励战略性和高效的工具使用。同时构建了新的SFT和RL数据集，并引入了新的基准测试套件。

### 主要发现

最先进的MLLMs在图像方向变化或自然损坏时表现出显著的性能下降；CodeVision显著提高了模型性能；促进了灵活的工具组合、高效的链式执行和从运行时反馈中恢复的健壮错误恢复等新兴能力的出现。

### 结论

CodeVision通过代码作为工具的框架，有效解决了MLLMs的脆弱性和工具限制问题，提高了模型的性能和可扩展性，为视觉推理提供了更强大的解决方案。

### 翻译

多模态大语言模型(MLLMs)能够通过图像进行交互式工具使用来推理视觉输入，但当前方法通常依赖于一组有限的工具，这些工具在现实世界的必要性和可扩展性方面有限。在本研究中，我们首先揭示了一个关键且先前被忽视的弱点：即使是最先进的MLLMs也异常脆弱，在图像方向简单变化或自然损坏时表现出显著的性能下降，突显了对更基于工具的鲁棒推理的需求。为此，我们提出了CodeVision，一个灵活可扩展的代码作为工具框架，模型在其中生成代码作为调用任何图像操作的通用接口，超越了固定的工具注册表。我们使用两阶段方法训练模型，首先在有质量保证的数据集上进行监督微调(SFT)，该数据集针对复杂的多工具组合和错误恢复进行精心设计，然后使用新颖且密集的过程奖励函数进行强化学习(RL)，以鼓励战略性和高效的工具使用。为促进这项研究，我们构建了新的SFT和RL数据集，并引入了一个具有挑战性的新基准测试套件，旨在严格评估对方向变化的鲁棒性和多工具推理能力。在Qwen2.5-VL和Qwen3-VL系列上的实验表明，我们的方法显著提高了模型性能，并促进了灵活工具组合、高效链式执行和从运行时反馈中恢复的健壮错误恢复等新兴能力的出现。代码可在https://github.com/ByteDance-BandAI/CodeVision获取。


### 论文摘要

Multimodal large language models (MLLMs) that think with images can interactively use tools to reason about visual inputs, but current approaches often rely on a narrow set of tools with limited real-world necessity and scalability. In this work, we first reveal a critical and previously overlooked weakness: even state-of-the-art MLLMs are surprisingly brittle, showing significant performance degradation on images with simple orientation changes or natural corruptions, underscoring the need for more robust tool-based reasoning. To address this, we propose CodeVision, a flexible and scalable code-as-tool framework where the model generates code as a universal interface to invoke any image operation, moving beyond fixed tool registries. We train our model using a two-stage methodology, beginning with Supervised Fine-Tuning (SFT) on a high-quality dataset curated for complex, multi-turn tool composition and error recovery, followed by Reinforcement Learning (RL) with a novel and dense process reward function to encourage strategic and efficient tool use. To facilitate this research, we construct new SFT and RL datasets and introduce a challenging new benchmark suite designed to rigorously evaluate robustness to orientation changes and multi-tool reasoning. Experiments on Qwen2.5-VL and Qwen3-VL series show that our approach significantly improves model performance and fosters emergent capabilities such as flexible tool composition, efficient chained execution, and robust error recovery from runtime feedback. Code is available at https://github.com/ByteDance-BandAI/CodeVision.

---

## 24. Colored Markov Random Fields for Probabilistic Topological Modeling

**论文链接:** [http://arxiv.org/abs/2512.03727v1](http://arxiv.org/abs/2512.03727v1)

**作者:** Lorenzo Marinucci, Leonardo Di Nino, Gabriele D'Acunto, Mario Edoardo Pandolfo, Paolo Di Lorenzo, Sergio Barbarossa

**发布时间:** 2025-12-03

**备注:** Proceeding of 2025 Asilomar Conference on Signals, Systems, and Computers

### GPT解析

### 总结

论文提出了彩色马尔可夫随机场(CMRFs)，这是一种扩展的概率图模型，能够同时建模拓扑空间中高斯边变量之间的条件依赖和边际依赖，基于Hodge理论构建。

### 背景

概率图模型(PGMs)通过图结构编码随机变量间的条件依赖，适合分析复杂系统；拓扑信号处理的进展表明拓扑空间上的变量在多领域应用中很重要，但底层拓扑限制了经典PGMs的表达能力。

### 目的

克服经典PGMs在处理拓扑空间变量时的局限性，开发能同时建模条件依赖和边际依赖的模型。

### 方法

引入彩色马尔可夫随机场(CMRFs)，通过链接着色扩展经典高斯马尔可夫随机场，其中连接性编码条件独立性，颜色编码边际独立性。

### 主要发现

通过物理网络上的分布式估计案例研究，量化了CMRFs的优势，并与具有不同拓扑先验的基线方法进行了比较。

### 结论

CMRFs能有效处理拓扑空间中的变量，同时考虑条件依赖和边际依赖，在分布式估计任务中表现出优势。

### 翻译

概率图模型(PGMs)使用图结构(节点表示变量，链接表示依赖关系)来编码随机变量之间的条件依赖，并将联合分布分解为低维组件。这使得PGMs非常适合分析复杂系统和支持决策。拓扑信号处理的最新进展强调了在多个应用领域中定义在拓扑空间上的变量的重要性。在这种情况下，底层拓扑塑造了统计关系，限制了经典PGMs的表达能力。为了克服这一限制，我们引入了彩色马尔可夫随机场(CMRFs)，它基于Hodge理论，能够建模拓扑空间中高斯边变量之间的条件依赖和边际依赖。CMRFs通过包含链接着色来扩展经典的高斯马尔可夫随机场：连接性编码条件独立性，而颜色编码边际独立性。我们通过物理网络上的分布式估计案例研究量化了CMRFs的优势，并将其与具有不同拓扑先验的基线方法进行了比较。


### 论文摘要

Probabilistic Graphical Models (PGMs) encode conditional dependencies among random variables using a graph -nodes for variables, links for dependencies- and factorize the joint distribution into lower-dimensional components. This makes PGMs well-suited for analyzing complex systems and supporting decision-making. Recent advances in topological signal processing highlight the importance of variables defined on topological spaces in several application domains. In such cases, the underlying topology shapes statistical relationships, limiting the expressiveness of canonical PGMs. To overcome this limitation, we introduce Colored Markov Random Fields (CMRFs), which model both conditional and marginal dependencies among Gaussian edge variables on topological spaces, with a theoretical foundation in Hodge theory. CMRFs extend classical Gaussian Markov Random Fields by including link coloring: connectivity encodes conditional independence, while color encodes marginal independence. We quantify the benefits of CMRFs through a distributed estimation case study over a physical network, comparing it with baselines with different levels of topological prior.

---

## 25. Over-the-Air Federated Learning: Rethinking Edge AI Through Signal Processing

**论文链接:** [http://arxiv.org/abs/2512.03719v1](http://arxiv.org/abs/2512.03719v1)

**作者:** Seyed Mohammad Azimi-Abarghouyi, Carlo Fischione, Kaibin Huang

**发布时间:** 2025-12-03

### GPT解析

### 总结

空中联合学习（AirFL）是一种新兴范式，将无线信号处理与分布式机器学习紧密结合，通过无线信号的叠加特性同时执行通信和模型聚合，显著降低延迟、带宽和能源消耗。

### 背景

AirFL旨在通过网络边缘实现可扩展的人工智能，解决传统机器学习在资源受限环境中的挑战。

### 目的

提供AirFL的教程性处理，并提出一种新的分类方法，将AirFL设计分为三种不同的方法。

### 方法

利用无线信号的叠加特性同时执行通信和模型聚合；提供理论基础的全面指南；进行性能分析；考虑复杂性；探讨实际限制；提出前瞻性研究方向。

### 主要发现

AirFL通过同时执行通信和模型聚合，能够显著降低延迟、带宽和能源消耗，提高边缘AI系统的效率。

### 结论

AirFL代表了边缘计算和机器学习融合的重要发展方向，具有广阔的应用前景和研究价值。

### 翻译

空中联合学习（AirFL）是一种新兴范式，它将无线信号处理和分布式机器学习紧密结合，以实现网络边缘的可扩展人工智能。通过利用无线信号的叠加特性，AirFL同时执行学习过程中的通信和模型聚合，显著降低了延迟、带宽和能源消耗。本文提供了AirFL的教程性处理，提出了一种新颖的分类方法，将其分为三种设计方法：信道状态信息感知型、盲型和加权型AirFL。我们全面介绍了理论基础、性能分析、复杂性考虑、实际限制和前瞻性研究方向。


### 论文摘要

Over-the-Air Federated Learning (AirFL) is an emerging paradigm that tightly integrates wireless signal processing and distributed machine learning to enable scalable AI at the network edge. By leveraging the superposition property of wireless signals, AirFL performs communication and model aggregation of the learning process simultaneously, significantly reducing latency, bandwidth, and energy consumption. This article offers a tutorial treatment of AirFL, presenting a novel classification into three design approaches: CSIT-aware, blind, and weighted AirFL. We provide a comprehensive guide to theoretical foundations, performance analysis, complexity considerations, practical limitations, and prospective research directions.

---

## 26. A theory-agnostic hierarchical Bayesian framework for black-hole spectroscopy: a case study on GW250114 in Einstein-dilaton-Gauss-Bonnet gravity

**论文链接:** [http://arxiv.org/abs/2512.03713v1](http://arxiv.org/abs/2512.03713v1)

**作者:** Shitong Guo, Yan-Gang Miao

**发布时间:** 2025-12-03

**备注:** v1: 18 pages, 11 figures, 4 tables

### GPT解析

### 总结

该研究开发了一种理论无关的分层贝叶斯框架，用于黑洞谱学分析，在频谱层面直接比较环荡观测与理论准正模频谱，避免了传统方法中的模型依赖系统误差。

### 背景

黑洞谱学作为强引力场的有力探测手段，在引力波天文学时代已经兴起。目前对修改或扩展引力的测试通常是通过寻找预测的、作为相对论论波形微扰校正的信号特征来实现的。

### 目的

开发一种不依赖于特定理论的测试方法，避免传统方法的模型依赖系统误差，扩展到更广泛的引力理论类别。

### 方法

构建理论无关的分层贝叶斯框架，将阻尼正弦波形式的环荡观测直接与理论准正模频谱联系起来，在频谱层面进行比较。框架包含软截断模块处理理论参数空间有限性，并配备定量诊断工具识别稳定分析时间窗口。

### 主要发现

在爱因斯坦-标量-高斯-博内特引力中应用该框架于GW250114事件，得到的ζ参数后验分布对先验假设具有鲁棒性但信息量有限。受控环荡注入研究表明非零耦合可被检测，但基于Kerr的先验可能部分吸收替代引力理论的频谱偏差。

### 结论

该工作为未来强引力场测试建立了透明且可扩展的基础，与下一代引力波探测器不断提高的精度和模态分辨率自然兼容。

### 翻译

黑洞谱学已成为引力波天文学时代强引力场的有力探测手段。在此背景下，许多当前对修改或扩展引力的测试是通过寻找预测的、作为相对论论波形微扰校正的信号特征来实现的；然而，这种方法可能引入依赖于模型的系统误差，并限制其在更广泛理论类别中的适用性。为了补充此类方法，我们开发了一种理论无关的分层贝叶斯框架，将阻尼正弦波形式的环荡观测直接与理论准正模频谱联系起来，在频谱层面而非通过特定理论的波形匹配进行比较。该框架包含软截断模块，用于考虑理论参数空间有效域的有限性，并配备定量诊断工具以识别稳定的分析时间窗口。作为说明性应用，我们在爱因斯坦-标量-高斯-博内特引力中实现了该框架，并应用于引力波事件GW250114，发现得到的ζ无量纲耦合参数后验分布对先验假设具有鲁棒性，但在本研究考虑的范围内仍然信息量有限。我们进一步在不同ζ值下进行受控环荡注入研究，确认非零耦合可以被恢复，同时也表明了一种潜在的系统效应：基于Kerr的先验可能在ζ推断中部分吸收替代引力理论中出现的频谱偏差。这项工作为未来的强引力场测试建立了一个透明且可扩展的基础，自然地与下一代引力波探测器不断提高的精度和模态分辨率相兼容。


### 论文摘要

Black-hole spectroscopy has emerged as a powerful probe of strong-field gravity in the era of gravitational-wave astronomy. In this context, many current tests of modified or extended gravity are implemented by searching for predicted signatures modeled as perturbative corrections to general-relativistic waveforms; however, this approach may introduce model-dependent systematics and limit applicability to broader classes of theories. To complement such methods, we develop a theory-agnostic hierarchical Bayesian framework that connects ringdown observations -- modeled as damped sinusoids -- directly with theoretical quasinormal mode spectra, performing the comparison at the spectral level rather than through theory-specific waveform matching. The framework incorporates a soft-truncation module to account for the finite domain of validity in the theory's parameter space and is equipped with quantitative diagnostics that identify stable analysis time windows. As an illustrative application, we implement the framework within Einstein-dilaton-Gauss-Bonnet gravity and apply it to the gravitational-wave event GW250114, finding that the resulting posterior for the dimensionless coupling $ζ$ is robust against prior assumptions yet remains only weakly informative over the range considered in this work. We further perform controlled ringdown injection studies across different values of $ζ$, confirming that nonzero couplings can be recovered while also indicating a potential systematic effect: Kerr-based priors in the $ζ$ inference may partially absorb spectral deviations arising in alternative theories of gravity. This work establishes a transparent and extensible foundation for future strong-field gravity tests, naturally compatible with the growing precision and modal resolution of next-generation gravitational-wave detectors.

---

## 27. 论文ID: 2512.03673v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03673v1.json'

---

## 28. Colon-X: Advancing Intelligent Colonoscopy from Multimodal Understanding to Clinical Reasoning

**论文链接:** [http://arxiv.org/abs/2512.03667v1](http://arxiv.org/abs/2512.03667v1)

**作者:** Ge-Peng Ji, Jingyi Liu, Deng-Ping Fan, Nick Barnes

**发布时间:** 2025-12-03

**备注:** Technical report

### GPT解析

### 总结

Colon-X是一个开放性计划，旨在推进结肠镜检查中的多模态智能。研究构建了最全面的多模态数据集ColonVQA，并探索了从多模态理解到临床推理的转变，开发了ColonR1模型，在数据稀缺条件下表现优异。

### 背景

结肠镜检查中多模态智能的发展面临从理解到临床推理的转变挑战，现有多模态大语言模型在临床应用中不够稳健和可信。

### 目的

推进结肠镜检查中的多模态智能，特别是从多模态理解到临床推理的转变，提高模型的可靠性和临床应用价值。

### 方法

构建了包含110多万条目、76种临床发现和18种多模态任务的ColonVQA数据集；评估了22种多模态大语言模型的泛化能力和可靠性；创建了通过多专家辩论流程注释的ColonReason数据集；开发了采用任务自适应奖励和梯度稳定优化技术的ColonR1模型。

### 主要发现

领先的多模态大语言模型临床输出不够稳健和可信；在数据稀缺条件下，ColonR1模型实现了56.61%的整体准确率，比监督微调高出25.22%，为多模态结肠镜分析建立了新的推理能力基线。

### 结论

Colon-X计划通过构建全面的数据集和开发专门的推理模型，有效推进了结肠镜检查中的多模态智能发展，特别是在临床推理方面取得了显著进展。

### 翻译

在这项研究中，我们提出了Colon-X，一个旨在推进结肠镜检查多模态智能的开放性计划。我们首先构建了ColonVQA，这是迄今为止为结肠镜构建的最全面的多模态数据集，包含超过110万个视觉问答条目，涵盖76种临床发现和18种多模态任务。除了作为全社区的数据基础外，我们还进一步研究了结肠镜检查中一个关键但未被充分探索的转变——从多模态理解向临床推理的转变：(a)为了捕捉当前多模态理解行为的现状，我们系统评估了22种多模态大语言模型的泛化能力，并检查了它们在人为干扰下的可靠性。结果表明，领先MLLMs的临床输出仍然远不够稳健和可信。(b)为了缩小这一差距，我们进一步探索了专门针对结肠镜检查的以推理为中心的智能。具体而言，我们整理了ColonReason，一个通过多专家辩论流程进行注释的临床基础推理数据集，并开发了ColonR1，这是第一个采用任务自适应奖励和梯度稳定优化技术的R1风格模型。在数据稀缺条件下，我们的ColonR1实现了56.61%的整体准确率，比监督微调高出25.22%，并为多模态结肠镜分析建立了新的推理能力基线。所有数据和模型资源可在https://github.com/ai4colonoscopy/Colon-X公开获取。


### 论文摘要

In this study, we present Colon-X, an open initiative aimed at advancing multimodal intelligence in colonoscopy. We begin by constructing ColonVQA, the most comprehensive multimodal dataset ever built for colonoscopy, featuring over 1.1M+ visual question answering entries across 76 clinical findings and 18 multimodal tasks. Beyond serving as a community-wide data foundation, we further investigate a critical yet underexplored transition in colonoscopy - evolving from multimodal understanding to clinical reasoning: (a) To capture the current landscape of multimodal understanding behaviors, we systematically assess the generalizability of 22 multimodal large language models and examine their reliability under human-induced perturbations. The results reveal that clinical outputs from leading MLLMs remain far from robust and trustworthy. (b) To narrow this gap, we further explore reasoning-centric intelligence tailored for colonoscopy. Specifically, we curate ColonReason, a clinically grounded reasoning dataset annotated through a multi-expert debating pipeline, and develop ColonR1, the first R1-styled model incorporating task-adaptive rewarding and gradient-stable optimization techniques. Under data-scarce conditions, our ColonR1 achieves 56.61% overall accuracy, outperforming supervised fine-tuning by 25.22%, and sets a new reasoning-enabled baseline for multimodal colonoscopy analysis. All data and model resources are publicly available at https://github.com/ai4colonoscopy/Colon-X.

---

## 29. ToG-Bench: Task-Oriented Spatio-Temporal Grounding in Egocentric Videos

**论文链接:** [http://arxiv.org/abs/2512.03666v1](http://arxiv.org/abs/2512.03666v1)

**作者:** Qi'ao Xu, Tianwen Qian, Yuqian Fu, Kailing Li, Yang Jiao, Jiacheng Zhang, Xiaoling Wang, Liang He

**发布时间:** 2025-12-03

**备注:** 26 pages

### GPT解析

### 总结

本文提出了ToG-Bench，这是第一个针对第一人称视频的任务导向空间时间视频定位基准，解决了现有研究忽略任务导向推理的问题。

### 背景

空间时间视频定位(STVG)是通用具身智能的核心能力，但现有研究主要集中在物体中心和描述性指令上，忽略了任务导向推理，这对具身智能体完成目标导向交互至关重要。

### 目的

引入ToG-Bench基准，填补现有研究与任务导向推理之间的差距，促进具身智能中感知和交互的结合。

### 方法

ToG-Bench具有三个关键特征：任务导向定位、显式-隐式双重定位和一对多定位。基于ScanNet视频构建，包含100个带注释片段和2,704个任务导向定位指令，使用半自动化流程构建。引入了针对多物体和显式-隐式物体定位的任务级评估指标，并对七个最先进的MLLM进行了系统基准测试。

### 主要发现

揭示了任务导向STVG的内在挑战，显式-隐式定位和多物体定位之间存在显著性能差距，强调了在具身场景中弥合感知和交互的难度。

### 结论

ToG-Bench为研究任务导向的空间时间视频定位提供了新基准，数据和代码将在GitHub上发布。

### 翻译

通用具身智能的核心能力在于从第一人称视角定位任务相关物体，这被表述为空间时间视频定位(STVG)。尽管近期取得了进展，但现有的STVG研究仍然主要局限于物体中心和描述性指令，忽略了具身智能体完成目标导向交互至关重要的任务导向推理。为了弥合这一差距，我们引入了ToG-Bench，这是第一个针对第一人称视频的任务导向空间时间视频定位基准。ToG-Bench具有三个关键特征：(1)任务导向定位，要求基于预期任务而非直接描述来识别和定位物体；(2)显式-隐式双重定位，目标物体可以是通过上下文推理显式提及或隐式推断；(3)一对多定位，单个指令可能对应多个参与任务执行的物体。基于ScanNet来源的视频构建，ToG-Bench包含100个带注释片段和2,704个任务导向定位指令，通过结合基础模型注释和人工优化的半自动化流程构建。此外，我们引入了一套针对多物体和显式-隐式物体定位的任务级评估指标，并对七个最先进的MLLM进行了系统基准测试。大量实验揭示了任务导向STVG的内在挑战以及显式-隐式和多物体定位之间的显著性能差距，突显了在具身场景中弥合感知和交互的难度。数据和代码将在以下地址发布：https://github.com/qaxuDev/ToG-Bench。


### 论文摘要

A core capability towards general embodied intelligence lies in localizing task-relevant objects from an egocentric perspective, formulated as Spatio-Temporal Video Grounding (STVG). Despite recent progress, existing STVG studies remain largely confined to object-centric and descriptive instructions, neglecting the task-oriented reasoning that is crucial for embodied agents to accomplish goal-directed interactions. To bridge this gap, we introduce \textbf{ToG-Bench}, the first task-oriented spatio-temporal video grounding benchmark for egocentric videos. ToG-Bench is characterized by three key features: (1) \textbf{Task-oriented Grounding}, which requires identifying and localizing objects based on intended tasks rather than straightforward descriptions; (2) \textbf{Explicit-Implicit Dual Grounding}, where target objects can be either explicitly mentioned or implicitly inferred by contextual reasoning; (3) \textbf{One-to-Many Grounding}, where a single instruction may correspond to multiple objects involved in task execution. Built upon videos sourced from ScanNet, ToG-Bench comprises 100 annotated clips with 2,704 task-oriented grounding instructions, constructed via a semi-automated pipeline that combines foundation model annotation and human refinement. In addition, we introduce a set of task-level evaluation metrics tailored for multi-object and explicit-implicit object grounding, and systematically benchmark seven state-of-the-art MLLMs. Extensive experiments reveal the intrinsic challenges of task-oriented STVG and substantial performance gaps across explicit-implicit and multi-object grounding, highlighting the difficulty of bridging perception and interaction in embodied scenarios. Data and code will be released at: \href{https://github.com/qaxuDev/ToG-Bench}{https://github.com/qaxuDev/ToG-Bench}..

---

## 30. Multi-Scale Visual Prompting for Lightweight Small-Image Classification

**论文链接:** [http://arxiv.org/abs/2512.03663v1](http://arxiv.org/abs/2512.03663v1)

**作者:** Salim Khazem

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种多尺度视觉提示（MSVP）方法，用于在小图像基准上提升视觉模型性能，该方法通过轻量级参数与输入图像融合，在CNN和Vision Transformer骨干网络上均表现出色。

### 背景

视觉提示是一种新兴的视觉模型适应策略，但先前研究主要关注大型Vision Transformer和高分辨率数据集，而MNIST、Fashion-MNIST和CIFAR-10等小型图像基准在提示方面受到的关注较少。

### 目的

研究视觉提示在小图像基准上的应用效果，提出一种适用于小型图像数据集的视觉提示方法。

### 方法

引入多尺度视觉提示（MSVP），学习一组全局、中尺度和局部提示图，通过轻量级的1×1卷积与输入图像融合，该方法与骨干网络无关，参数增加少于0.02%。

### 主要发现

在MNIST、Fashion-MNIST和CIFAR-10上的实验表明，MSVP在计算开销可忽略的情况下带来了一致的性能提升；对提示尺度、融合策略和骨干架构的消融研究以及定性分析证明了多尺度提示在低分辨率图像上的有效性。

### 结论

多尺度视觉提示（MSVP）是一种有效的方法，可以在小型图像数据集上提升视觉模型的性能，即使在低分辨率图像上也能提供有效的归纳偏置。

### 翻译

视觉提示最近已成为一种高效的策略，通过在输入空间注入轻量级、可学习的参数来适应视觉模型。然而，先前的工作主要针对大型视觉变压器和高分辨率数据集，如ImageNet。相比之下，MNIST、Fashion-MNIST和CIFAR-10等小型图像基准在教育、原型设计和研究中仍被广泛使用，但在提示方面却很少受到关注。在本文中，我们引入了多尺度视觉提示（MSVP），这是一个简单通用的模块，学习一组全局、中尺度和局部提示图，通过轻量级的1×1卷积与输入图像融合。MSVP与骨干网络无关，参数增加少于0.02%，并且显著提高了CNN和视觉变压器骨干网络的性能。我们使用简单CNN、ResNet-18和小型视觉变压器在MNIST、Fashion-MNIST和CIFAR-10上提供了统一基准。我们的方法在计算开销可忽略的情况下带来了一致的改进。我们进一步对提示尺度、融合策略和骨干架构进行了消融研究，以及使用提示可视化和Grad-CAM进行了定性分析。我们的结果表明，即使在低分辨率图像上，多尺度提示也能提供有效的归纳偏置。


### 论文摘要

Visual prompting has recently emerged as an efficient strategy to adapt vision models using lightweight, learnable parameters injected into the input space. However, prior work mainly targets large Vision Transformers and high-resolution datasets such as ImageNet. In contrast, small-image benchmarks like MNIST, Fashion-MNIST, and CIFAR-10 remain widely used in education, prototyping, and research, yet have received little attention in the context of prompting. In this paper, we introduce \textbf{Multi-Scale Visual Prompting (MSVP)}, a simple and generic module that learns a set of global, mid-scale, and local prompt maps fused with the input image via a lightweight $1 \times 1$ convolution. MSVP is backbone-agnostic, adds less than $0.02\%$ parameters, and significantly improves performance across CNN and Vision Transformer backbones.   We provide a unified benchmark on MNIST, Fashion-MNIST, and CIFAR-10 using a simple CNN, ResNet-18, and a small Vision Transformer. Our method yields consistent improvements with negligible computational overhead. We further include ablations on prompt scales, fusion strategies, and backbone architectures, along with qualitative analyzes using prompt visualizations and Grad-CAM. Our results demonstrate that multi-scale prompting provides an effective inductive bias even on low-resolution images.

---

## 31. Unique Lives, Shared World: Learning from Single-Life Videos

**论文链接:** [http://arxiv.org/abs/2512.04085v1](http://arxiv.org/abs/2512.04085v1)

**作者:** Tengda Han, Sayna Ebrahimi, Dilara Gokay, Li Yang Ku, Maks Ovsjanikov, Iva Babukova, Daniel Zoran, Viorica Patraucean, Joao Carreira, Andrew Zisserman, Dima Damen

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种'单生命'学习范式，通过使用单个个体的第一人称视频训练视觉模型，并利用单个人生活中自然捕捉的多个视角进行自监督学习。研究表明，这种方法能够学习到可推广的几何表示，且性能可与使用多样化网络数据训练的模型相媲美。

### 背景

传统的视觉表示学习通常使用多样化的网络数据，而本文探索了仅使用一个人的生活视频进行视觉模型训练的可能性。

### 目的

探索仅使用一个人的生活视频训练视觉模型的可行性，并研究这种模型是否能学习到可推广的几何表示。

### 方法

利用单个人生活中自然捕捉的多个视角进行自监督学习；在不同数据集上（室内和室外）训练独立的视觉编码器；引入基于交叉注意力的新指标量化不同模型内部表示的功能一致性；评估单生命模型在下游任务（如深度估计）上的泛化能力；比较使用单个人一周生活数据（最多30小时）与多样化网络数据（30小时）训练的性能。

### 主要发现

1) 在不同生活数据上独立训练的模型发展出高度一致的几何理解；2) 单生命模型学习到可推广的几何表示，能有效迁移到未见环境的下游任务；3) 使用同一个人一周生活数据（最多30小时）训练能达到与多样化网络数据（30小时）训练相当的性能。

### 结论

共享的世界结构既导致了在个体生活上训练的模型间的一致性，也为视觉表示学习提供了强大的信号。

### 翻译

我们引入了'单生命'学习范式，其中我们训练一个独特的视觉模型，专门用于捕捉由单个个体拍摄的第一人称视频。我们利用在单个人生命中自然捕捉的多个视角，以自监督方式学习视觉编码器。我们的实验证明了三个关键发现。首先，在不同生命数据上独立训练的模型发展出高度一致的几何理解。我们通过在不同数据集上训练视觉编码器来证明这一点，这些数据集捕捉了不同的生命，包括室内和室外，同时引入了一种新颖的基于交叉注意力的指标来量化不同模型开发的内部表示的功能一致性。其次，我们表明单生命模型学习到可推广的几何表示，能有效迁移到未见环境的下游任务，如深度估计。第三，我们证明使用同一个人一周生活中的最多30小时数据进行训练，可以达到与使用30小时多样化网络数据进行训练相当的性能，突显了单生命表示学习的优势。总体而言，我们的研究结果表明，世界的共享结构既导致了在个体生命上训练的模型间的一致性，也为视觉表示学习提供了强大的信号。


### 论文摘要

We introduce the "single-life" learning paradigm, where we train a distinct vision model exclusively on egocentric videos captured by one individual. We leverage the multiple viewpoints naturally captured within a single life to learn a visual encoder in a self-supervised manner. Our experiments demonstrate three key findings. First, models trained independently on different lives develop a highly aligned geometric understanding. We demonstrate this by training visual encoders on distinct datasets each capturing a different life, both indoors and outdoors, as well as introducing a novel cross-attention-based metric to quantify the functional alignment of the internal representations developed by different models. Second, we show that single-life models learn generalizable geometric representations that effectively transfer to downstream tasks, such as depth estimation, in unseen environments. Third, we demonstrate that training on up to 30 hours from one week of the same person's life leads to comparable performance to training on 30 hours of diverse web data, highlighting the strength of single-life representation learning. Overall, our results establish that the shared structure of the world, both leads to consistency in models trained on individual lives, and provides a powerful signal for visual representation learning.

---

## 32. Fast & Efficient Normalizing Flows and Applications of Image Generative Models

**论文链接:** [http://arxiv.org/abs/2512.04039v1](http://arxiv.org/abs/2512.04039v1)

**作者:** Sandeep Nagar

**发布时间:** 2025-12-03

**备注:** PhD Thesis

### GPT解析

### 总结

该论文在两个主要领域做出了创新性贡献：推进生成模型（特别是归一化流）的效率，以及将生成模型应用于解决实际计算机视觉问题。论文提出了六项归一化流架构的关键改进，以及五项生成模型在计算机视觉领域的应用创新。

### 背景

生成模型，特别是归一化流，在计算机视觉领域有广泛应用，但其效率和实用性仍有提升空间。同时，实际计算机视觉问题如农产品质量评估、地质制图、隐私保护和艺术修复等需要更先进的生成模型解决方案。

### 目的

提高生成模型的效率，特别是归一化流模型的性能，并将这些技术应用于解决实际计算机视觉挑战，包括质量评估、地质制图、隐私保护和艺术修复等领域。

### 方法

第一部分提出了六项归一化流架构的改进：可逆3x3卷积层、四耦合层、k×k卷积层的并行逆算法、卷积逆的反向传播算法、逆流中的卷积逆应用，以及Affine-StableSR超分辨率模型。第二部分应用生成模型解决实际问题：条件GAN用于农产品质量评估、堆叠自编码器用于地质制图、人脸检测和图像修复用于隐私保护、稳定扩散用于图像修复和隐私保护，以及扩散模型用于艺术修复。

### 主要发现

归一化流模型的六项改进显著提高了模型效率和性能。在应用方面，条件GAN在农产品质量评估中表现良好，堆叠自编码器在地质制图中改进了特征提取，隐私保护方法有效保护了敏感信息，扩散模型在艺术修复中能有效处理多种退化类型。

### 结论

论文通过创新性地改进归一化流架构并将其应用于实际计算机视觉问题，为生成模型领域的发展做出了重要贡献。这些技术进步不仅提高了生成模型的效率，还拓展了其在解决实际挑战中的应用范围。

### 翻译

这篇论文在两个主要领域做出了创新性贡献：推进生成模型（特别是归一化流）的效率，以及将生成模型应用于解决实际计算机视觉挑战。第一部分通过六项关键创新显著改进了归一化流架构：1) 开发了具有数学证明的可逆条件的3x3可逆卷积层；2) 引入了更高效的四耦合层；3) 设计了用于k×k卷积层的快速高效并行逆算法；4) 卷积逆的快速高效反向传播算法；5) 在逆流中使用卷积逆进行前向传播，并使用提出的反向传播算法进行训练；6) Affine-StableSR，一种紧凑高效的超分辨率模型，利用预训练权重和归一化流层减少参数数量同时保持性能。第二部分包括：1) 使用条件GAN开发农产品自动化质量评估系统，解决类别不平衡、数据稀缺和标注挑战，在种子纯度测试中取得良好准确性；2) 利用堆叠自编码器进行降维的无监督地质制图框架，相比传统方法显示出改进的特征提取；3) 使用人脸检测和图像修复提出自动驾驶数据集的隐私保护方法；4) 利用基于稳定扩散的图像修复替换检测到的人脸和车牌，推进隐私保护技术和道德考量；5) 适应扩散模型用于艺术修复，通过统一微调有效处理多种类型的退化。


### 论文摘要

This thesis presents novel contributions in two primary areas: advancing the efficiency of generative models, particularly normalizing flows, and applying generative models to solve real-world computer vision challenges. The first part introduce significant improvements to normalizing flow architectures through six key innovations: 1) Development of invertible 3x3 Convolution layers with mathematically proven necessary and sufficient conditions for invertibility, (2) introduction of a more efficient Quad-coupling layer, 3) Design of a fast and efficient parallel inversion algorithm for kxk convolutional layers, 4) Fast & efficient backpropagation algorithm for inverse of convolution, 5) Using inverse of convolution, in Inverse-Flow, for the forward pass and training it using proposed backpropagation algorithm, and 6) Affine-StableSR, a compact and efficient super-resolution model that leverages pre-trained weights and Normalizing Flow layers to reduce parameter count while maintaining performance.   The second part: 1) An automated quality assessment system for agricultural produce using Conditional GANs to address class imbalance, data scarcity and annotation challenges, achieving good accuracy in seed purity testing; 2) An unsupervised geological mapping framework utilizing stacked autoencoders for dimensionality reduction, showing improved feature extraction compared to conventional methods; 3) We proposed a privacy preserving method for autonomous driving datasets using on face detection and image inpainting; 4) Utilizing Stable Diffusion based image inpainting for replacing the detected face and license plate to advancing privacy-preserving techniques and ethical considerations in the field.; and 5) An adapted diffusion model for art restoration that effectively handles multiple types of degradation through unified fine-tuning.

---

## 33. Domain Feature Collapse: Implications for Out-of-Distribution Detection and Solutions

**论文链接:** [http://arxiv.org/abs/2512.04034v1](http://arxiv.org/abs/2512.04034v1)

**作者:** Hong Yang, Devroop Kar, Qi Yu, Alex Ororbia, Travis Desell

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究从信息论角度解释了先进的OOD检测方法在单域数据集上训练的模型出现灾难性失败的原因，并通过理论证明和实验验证了这一现象。

### 背景

最先进的OOD检测方法在单域数据集上训练的模型会出现灾难性失败，这种现象缺乏理论解释。

### 目的

从信息论角度为单域训练模型的OOD检测失败提供理论解释，并探索解决方案。

### 方法

通过信息瓶颈优化证明监督学习在单域数据上会导致域特征坍塌；使用Fano不等式量化实际场景中的部分坍塌；引入Domain Bench基准测试；通过域过滤保留域信息。

### 主要发现

单域训练的模型会丢弃域特定信息，仅保留类特定特征，导致在检测分布外样本时失败；域过滤能有效解决这一问题，提供经验证据支持信息论框架。

### 结论

揭示了监督学习在狭窄域中的基本局限性，对迁移学习和预训练模型使用策略有广泛影响。

### 翻译

为什么最先进的OOD检测方法在单域数据集上训练的模型会表现出灾难性失败？我们通过信息论的角度为这一现象提供了首个理论解释。我们证明，在单域数据上的监督学习不可避免地会导致域特征坍塌——表示中域特定信息被完全丢弃。这是信息瓶颈优化的基本结果：在单域（如医学图像）上训练的模型学会仅依赖类特定特征，同时丢弃域特征，导致在检测分布外样本时灾难性失败（例如在MNIST上仅达到53%的FPR@95）。我们使用Fano不等式扩展分析，量化实际场景中的部分坍塌。为验证我们的理论，我们引入了Domain Bench，一个单域数据集基准，并证明通过域过滤（使用预训练表示）保持域信息可以解决这种失败模式。虽然域过滤本身在概念上很简单，但其有效性为我们的信息论框架提供了强有力的经验证据。我们的工作解释了一个令人困惑的经验现象，揭示了监督学习在狭窄域中的基本局限性，并对迁移学习以及何时微调而非冻结预训练模型有更广泛的影响。


### 论文摘要

Why do state-of-the-art OOD detection methods exhibit catastrophic failure when models are trained on single-domain datasets? We provide the first theoretical explanation for this phenomenon through the lens of information theory. We prove that supervised learning on single-domain data inevitably produces domain feature collapse -- representations where I(x_d; z) = 0, meaning domain-specific information is completely discarded. This is a fundamental consequence of information bottleneck optimization: models trained on single domains (e.g., medical images) learn to rely solely on class-specific features while discarding domain features, leading to catastrophic failure when detecting out-of-domain samples (e.g., achieving only 53% FPR@95 on MNIST). We extend our analysis using Fano's inequality to quantify partial collapse in practical scenarios. To validate our theory, we introduce Domain Bench, a benchmark of single-domain datasets, and demonstrate that preserving I(x_d; z) > 0 through domain filtering (using pretrained representations) resolves the failure mode. While domain filtering itself is conceptually straightforward, its effectiveness provides strong empirical evidence for our information-theoretic framework. Our work explains a puzzling empirical phenomenon, reveals fundamental limitations of supervised learning in narrow domains, and has broader implications for transfer learning and when to fine-tune versus freeze pretrained models.

---

## 34. Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models

**论文链接:** [http://arxiv.org/abs/2512.03989v1](http://arxiv.org/abs/2512.03989v1)

**作者:** Taido Purason, Pavel Chizhov, Ivan P. Yamshchikov, Mark Fishel

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种分词器适配方法，通过继续BPE训练和基于叶子的词汇剪枝技术，解决了预训练语言模型迁移到新领域或语言时的词汇扩展问题，提高了分词效率和词汇利用率。

### 背景

分词器适配在将预训练语言模型转移到新领域或新语言中扮演着重要角色，但现有方法常导致许多标记无法到达或从未被使用。

### 目的

解决分词器适配过程中的两个互补方面：词汇扩展和剪枝，提高分词效率并更好地利用添加的词汇。

### 方法

提出继续BPE训练方法，通过在新数据上继续BPE合并学习过程来适配预训练分词器；同时引入基于叶子的词汇剪枝技术，在保持模型质量的同时移除冗余标记。

### 主要发现

在多种语言和模型系列上的实验表明，继续BPE训练方法提高了分词效率，并更好地利用了添加的词汇；基于叶子的词汇剪枝可以在保持模型质量的同时移除冗余标记。

### 结论

这些方法为受控词汇修改提供了实用工具，已作为开源包发布，可用于预训练语言模型在新领域或语言上的迁移。

### 翻译

分词器适配在将预训练语言模型转移到新领域或新语言中扮演着重要角色。在这项工作中，我们解决了这一过程中的两个互补方面：词汇扩展和剪枝。扩展的常见方法是在领域特定文本上训练新的分词器，并将不与现有词汇重叠的标记追加进去，这往往导致许多标记无法到达或从未被使用。我们提出了继续BPE训练，通过在新数据上继续BPE合并学习过程来适配预训练分词器。在多种语言和模型系列上的实验表明，这种方法提高了分词效率，并更好地利用了添加的词汇。我们还引入了基于叶子的词汇剪枝，在保持模型质量的同时移除冗余标记。这些方法共同为受控词汇修改提供了实用工具，我们将其作为开源包发布。


### 论文摘要

Tokenizer adaptation plays an important role in transferring pre-trained language models to new domains or languages. In this work, we address two complementary aspects of this process: vocabulary extension and pruning. The common approach to extension trains a new tokenizer on domain-specific text and appends the tokens that do not overlap with the existing vocabulary, which often results in many tokens that are unreachable or never used. We propose continued BPE training, which adapts a pre-trained tokenizer by continuing the BPE merge learning process on new data. Experiments across multiple languages and model families show that this approach improves tokenization efficiency and leads to better utilization of added vocabulary. We also introduce leaf-based vocabulary pruning, which removes redundant tokens while preserving model quality. Together, these methods provide practical tools for controlled vocabulary modification, which we release as an open-source package.

---

## 35. Fully Unsupervised Self-debiasing of Text-to-Image Diffusion Models

**论文链接:** [http://arxiv.org/abs/2512.03749v1](http://arxiv.org/abs/2512.03749v1)

**作者:** Korada Sri Vardhana, Shrikrishna Lolla, Soma Biswas

**发布时间:** 2025-12-03

**备注:** Accepted at WACV 2026

### GPT解析

### 总结

本文介绍了SelfDebias，一种完全无监督的测试时去偏方法，适用于任何使用UNet作为噪声预测器的扩散模型，能够减少生成图像中的刻板印象同时保持视觉保真度。

### 背景

Text-to-image扩散模型因其能生成高分辨率、照片级真实图像而取得广泛成功，但这些模型通常在从互联网收集的大型数据集（如LAION-5B）上训练，数据中包含的偏见导致模型学习并复制这些偏见，产生刻板印象输出。

### 目的

开发一种不需要人工标注数据集或外部分类器的去偏方法，能够自动识别语义模式，减少扩散模型生成图像中的偏见同时保持图像的视觉质量。

### 方法

SelfDebias通过在图像编码器的嵌入空间中识别语义簇，并使用这些簇来指导推理过程中的扩散过程，最小化输出分布和均匀分布之间的KL散差。

### 主要发现

SelfDebias能够跨提示和扩散模型架构泛化，包括条件模型和非条件模型；它不仅有效减少关键人口统计维度上的偏见同时保持生成图像的视觉保真度，还能处理更抽象的概念。

### 结论

SelfDebias是一种有效的、完全无监督的去偏方法，不需要额外的训练数据或分类器，能够自动识别并减少扩散模型生成图像中的各种偏见同时保持图像质量。

### 翻译

Text-to-image (T2I)扩散模型因其能够生成高分辨率、照片级真实的图像而取得了广泛成功。这些模型通常在从互联网收集的大型数据集（如LAION-5B）上进行训练。然而，由于这些数据包含许多偏见，模型会内在地学习和复制它们，导致产生刻板印象的输出。我们引入了SelfDebias，一种完全无监督的测试时去偏方法，适用于任何使用UNet作为噪声预测器的扩散模型。SelfDebias在图像编码器的嵌入空间中识别语义簇，并使用这些簇来指导推理过程中的扩散过程，最小化输出分布和均匀分布之间的KL散度。与监督方法不同，SelfDebias不需要人工标注的数据集或为每个生成概念训练的外部分类器。相反，它被设计为自动识别语义模式。大量实验表明，SelfDebias能够跨提示和扩散模型架构泛化，包括条件模型和非条件模型。它不仅能够有效地减少关键人口统计维度上的偏见，同时保持生成图像的视觉保真度，还能处理更抽象的概念，对于这些概念识别偏见也具有挑战性。


### 论文摘要

Text-to-image (T2I) diffusion models have achieved widespread success due to their ability to generate high-resolution, photorealistic images. These models are trained on large-scale datasets, like LAION-5B, often scraped from the internet. However, since this data contains numerous biases, the models inherently learn and reproduce them, resulting in stereotypical outputs. We introduce SelfDebias, a fully unsupervised test-time debiasing method applicable to any diffusion model that uses a UNet as its noise predictor. SelfDebias identifies semantic clusters in an image encoder's embedding space and uses these clusters to guide the diffusion process during inference, minimizing the KL divergence between the output distribution and the uniform distribution. Unlike supervised approaches, SelfDebias does not require human-annotated datasets or external classifiers trained for each generated concept. Instead, it is designed to automatically identify semantic modes. Extensive experiments show that SelfDebias generalizes across prompts and diffusion model architectures, including both conditional and unconditional models. It not only effectively debiases images along key demographic dimensions while maintaining the visual fidelity of the generated images, but also more abstract concepts for which identifying biases is also challenging.

---

## 36. Dual-level Modality Debiasing Learning for Unsupervised Visible-Infrared Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2512.03745v1](http://arxiv.org/abs/2512.03745v1)

**作者:** Jiaze Li, Yan Lu, Bin Liu, Guojun Yin, Mang Ye

**发布时间:** 2025-12-03

### GPT解析

### 总结

论文提出了一种双层模态去偏学习(DMDL)框架，用于解决无监督可见光-红外行人重识别任务中的模态偏差问题，通过在模型和优化两个层面实施去偏策略，实现了模态不变的特征学习和更泛化的模型。

### 背景

两阶段学习流水线在无监督可见光-红外行人重识别领域取得了有前景的结果，该流水线首先执行单模态学习，然后进行跨模态学习来解决模态差异问题。

### 目的

解决两阶段学习流水线中不可避免的模态偏差问题，防止模态特定特征在单模态训练后传播到跨模态学习，从而改善身份判别能力和模型泛化能力。

### 方法

提出双层模态去偏学习(DMDL)框架：1)在模型层面，引入受因果理论启发的调整干预(CAI)模块，用因果建模替代基于可能性的建模；2)在优化层面，提出协作无偏训练(CBT)策略，通过整合模态特定增强、标签细化和特征对齐，中断数据、标签和特征之间的模态偏差传播。

### 主要发现

在基准数据集上的大量实验表明，DMDL框架能够有效实现模态不变的特征学习，构建出更泛化的模型。

### 结论

DMDL框架通过在模型和优化两个层面实施去偏策略，成功解决了两阶段学习流水线中的模态偏差问题，提高了无监督可见光-红外行人重识别任务的性能。

### 翻译

两阶段学习流水线在无监督可见光-红外行人重识别(USL-VI-ReID)领域取得了有前景的结果。它首先执行单模态学习，然后进行跨模态学习来解决模态差异问题。尽管有前景，但这种流水线不可避免地引入了模态偏差：在单模态训练中学到的模态特定特征自然地传播到后续的跨模态学习中，损害了身份判别和泛化能力。为解决这一问题，我们提出了一个双层模态去偏学习(DMDL)框架，在模型和优化两个层面实施去偏。在模型层面，我们提出了一个受因果理论启发的调整干预(CAI)模块，用因果建模替代基于可能性的建模，防止引入由模态引起的虚假模式，从而得到低偏差模型。在优化层面，引入了协作无偏训练(CBT)策略，通过整合模态特定增强、标签细化和特征对齐，中断数据、标签和特征之间的模态偏差传播。在基准数据集上的大量实验表明，DMDL能够实现模态不变的特征学习，并构建一个更泛化的模型。


### 论文摘要

Two-stage learning pipeline has achieved promising results in unsupervised visible-infrared person re-identification (USL-VI-ReID). It first performs single-modality learning and then operates cross-modality learning to tackle the modality discrepancy. Although promising, this pipeline inevitably introduces modality bias: modality-specific cues learned in the single-modality training naturally propagate into the following cross-modality learning, impairing identity discrimination and generalization. To address this issue, we propose a Dual-level Modality Debiasing Learning (DMDL) framework that implements debiasing at both the model and optimization levels. At the model level, we propose a Causality-inspired Adjustment Intervention (CAI) module that replaces likelihood-based modeling with causal modeling, preventing modality-induced spurious patterns from being introduced, leading to a low-biased model. At the optimization level, a Collaborative Bias-free Training (CBT) strategy is introduced to interrupt the propagation of modality bias across data, labels, and features by integrating modality-specific augmentation, label refinement, and feature alignment. Extensive experiments on benchmark datasets demonstrate that DMDL could enable modality-invariant feature learning and a more generalized model.

---

## 37. Crossing the Sim2Real Gap Between Simulation and Ground Testing to Space Deployment of Autonomous Free-flyer Control

**论文链接:** [http://arxiv.org/abs/2512.03736v1](http://arxiv.org/abs/2512.03736v1)

**作者:** Kenneth Stewart, Samantha Chapin, Roxana Leontie, Carl Glen Henshaw

**发布时间:** 2025-12-03

**备注:** published at iSpaRo 2025

### GPT解析

### 总结

该研究首次展示了在国际空间站上使用强化学习控制NASA Astrobee自由飞行机器人的可行性，通过模拟训练成功实现了机器人在微重力环境中的自主导航。

### 背景

强化学习在太空机器人控制领域具有变革性潜力，但如何将模拟训练有效转移到实际太空环境一直是一个挑战。

### 目的

验证一种能够弥合模拟到现实差距的训练流程，证明在地面训练的强化学习策略可以成功应用于太空机器人控制。

### 方法

使用NVIDIA的Omniverse物理模拟器和课程学习策略训练深度神经网络，替代Astrobee的标准姿态和平移控制系统，并在国际空间站上进行在轨测试。

### 主要发现

成功验证了新型训练流程的有效性，证明GPU加速的科学级模拟环境可以高效进行蒙特卡洛强化学习训练，并将训练结果成功转移到太空应用中。

### 结论

强化学习可以有效地应用于太空机器人的自主控制，为未来的在轨服务、组装和制造工作铺平了道路，使太空机器人能够快速适应动态任务需求。

### 翻译

强化学习（RL）为太空机器人控制提供了变革性潜力。我们首次展示了在国际空间站（ISS）上基于强化学习的NASA Astrobee自由飞行机器人的自主控制在轨演示。使用NVIDIA的Omniverse物理模拟器和课程学习，我们训练了一个深度神经网络来替代Astrobee的标准姿态和平移控制，使其能够在微重力环境中导航。我们的结果验证了一种新型训练流程，弥合了模拟到现实（Sim2Real）的差距，利用GPU加速的科学级模拟环境进行高效的蒙特卡洛强化学习训练。这次成功的部署证明了在地面训练强化学习策略并将其转移到太空应用的可行性。这为未来的在轨服务、组装和制造（ISAM）工作铺平了道路，使机器人能够快速适应动态任务需求。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决的是'模拟到现实'(Sim2Real)的差距问题，即如何将强化学习训练出的控制策略从模拟环境成功转移到真实的太空环境中。这个问题在太空探索中至关重要，因为太空环境复杂且危险，需要高度自主的机器人系统；传统控制方法适应性有限；未来太空任务需要能动态适应不确定条件的机器人；而太空任务成本高、风险大，需要在地面充分验证控制算法的有效性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了强化学习在机器人控制领域的现有工作，特别是Proximal Policy Optimization (PPO)算法和NVIDIA的Omniverse物理模拟器。设计思路包括：在模拟器中使用大规模并行训练提高效率；采用课程学习逐步增加任务难度和环境变化；通过渐进式验证流程确保策略可靠性，从模拟训练到地面测试再到太空实际部署。这种系统化的方法确保了RL策略能够在真实太空环境中可靠工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是用强化学习替代传统控制方法提高自主性，通过课程学习增强策略鲁棒性，利用大规模并行模拟提高训练效率，并采用渐进式验证确保可靠性。整体流程分为四个阶段：1)模拟训练：在Omniverse中使用PPO算法和课程学习训练RL策略；2)模拟验证：在相同和不同的模拟环境中评估策略性能并与传统控制器对比；3)地面测试：在Granite实验室使用空气轴承模拟微重力环境测试实际硬件；4)太空测试：在ISS微重力环境中部署策略执行各种机动任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首个在轨演示强化学习控制自由飞行机器人；开发完整训练管道跨越Sim2Real差距；应用课程学习提高策略鲁棒性；利用大规模并行训练提高效率；特别关注质量变化等太空环境不确定性。相比之前工作，本文实现了从模拟到太空部署的完整流程；将PPO应用于太空微重力环境并针对性优化；通过课程学习解决实际太空环境中的不确定性问题；使用更先进的模拟工具和更大规模训练提高效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次成功在太空中部署了强化学习控制的自由飞行机器人，开发了一个完整的训练和验证流程，有效解决了模拟到现实的差距问题，为未来自主太空机器人系统铺平了道路。'}


### 论文摘要

Reinforcement learning (RL) offers transformative potential for robotic control in space. We present the first on-orbit demonstration of RL-based autonomous control of a free-flying robot, the NASA Astrobee, aboard the International Space Station (ISS). Using NVIDIA's Omniverse physics simulator and curriculum learning, we trained a deep neural network to replace Astrobee's standard attitude and translation control, enabling it to navigate in microgravity. Our results validate a novel training pipeline that bridges the simulation-to-reality (Sim2Real) gap, utilizing a GPU-accelerated, scientific-grade simulation environment for efficient Monte Carlo RL training. This successful deployment demonstrates the feasibility of training RL policies terrestrially and transferring them to space-based applications. This paves the way for future work in In-Space Servicing, Assembly, and Manufacturing (ISAM), enabling rapid on-orbit adaptation to dynamic mission requirements.

---

## 38. Cross-Stain Contrastive Learning for Paired Immunohistochemistry and Histopathology Slide Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.03577v1](http://arxiv.org/abs/2512.03577v1)

**作者:** Yizhi Zhang, Lei Fan, Zhulin Tao, Donglin Di, Yang Song, Sidong Liu, Cong Cong

**发布时间:** 2025-12-03

**备注:** 6 pages, 2 figures. Camera-ready version accepted for IEEE BIBM 2025

### GPT解析

### 总结

本文提出了一种跨染色对比学习方法(CSCL)，用于解决多染色病理图像中的对齐问题，生成高质量的H&E全玻片图像表示。

### 背景

全玻片图像(WSI)的通用、可迁移表示是计算病理学的核心。结合多种标记物(如免疫组化IHC)与H&E染色可以提供多样化的生物学信息，但进展受限于良好对齐的多染色数据集稀缺。染色间错位会导致相应组织在切片间移动，阻碍一致的块级特征并降低切片级嵌入质量。

### 目的

创建一个切片级对齐的五染色数据集(H&E, HER2, KI67, ER, PGR)，用于成对H&E-IHC学习和稳健的跨染色表示，解决染色间错位问题。

### 方法

提出Cross-Stain Contrastive Learning (CSCL)，一个两阶段预训练框架：1)使用轻量级适配器，通过块级对比对齐训练，提高H&E特征与相应IHC衍生上下文线索的兼容性；2)使用多实例学习(MIL)进行切片级表示学习，包括跨染色注意力融合模块整合染色特定的块特征，以及跨染色全局对齐模块强制不同染色切片级嵌入之间的一致性。

### 主要发现

在癌症亚型分类、IHC生物标志物状态分类和生存预测实验中，CSCL方法显示出一致的性能提升，产生高质量的、可迁移的H&E切片级表示。

### 结论

CSCL方法有效解决了染色间错位问题，提高了跨染色表示的质量，所提出的数据集和方法有助于计算病理学中WSI表示的学习。

### 翻译

通用的、可迁移的全玻片图像(WSI)表示是计算病理学的核心。将多种标记物(如免疫组化IHC)与H&E结合，可以用多样化的、具有生物学意义的信息丰富基于H&E的特征。然而，进展受到良好对齐的多染色数据集稀缺的限制。染色间错位会导致相应组织在切片间移动，阻碍一致的块级特征并降低切片级嵌入质量。为此，我们整理了一个切片级对齐的五染色数据集(H&E, HER2, KI67, ER, PGR)，以实现成对H&E-IHC学习和稳健的跨染色表示。利用此数据集，我们提出了跨染色对比学习(CSCL)，这是一个两阶段预训练框架，使用轻量级适配器通过块级对比对齐进行训练，以提高H&E特征与相应IHC衍生上下文线索的兼容性，并使用多实例学习(MIL)进行切片级表示学习，该学习使用跨染色注意力融合模块整合染色特定的块特征，并使用跨染色全局对齐模块强制不同染色切片级嵌入之间的一致性。在癌症亚型分类、IHC生物标志物状态分类和生存预测实验中，显示出一致的改进，产生高质量的、可迁移的H&E切片级表示。代码和数据可在https://github.com/lily-zyz/CSCL获取。


### 论文摘要

Universal, transferable whole-slide image (WSI) representations are central to computational pathology. Incorporating multiple markers (e.g., immunohistochemistry, IHC) alongside H&E enriches H&E-based features with diverse, biologically meaningful information. However, progress is limited by the scarcity of well-aligned multi-stain datasets. Inter-stain misalignment shifts corresponding tissue across slides, hindering consistent patch-level features and degrading slide-level embeddings. To address this, we curated a slide-level aligned, five-stain dataset (H&E, HER2, KI67, ER, PGR) to enable paired H&E-IHC learning and robust cross-stain representation. Leveraging this dataset, we propose Cross-Stain Contrastive Learning (CSCL), a two-stage pretraining framework with a lightweight adapter trained using patch-wise contrastive alignment to improve the compatibility of H&E features with corresponding IHC-derived contextual cues, and slide-level representation learning with Multiple Instance Learning (MIL), which uses a cross-stain attention fusion module to integrate stain-specific patch features and a cross-stain global alignment module to enforce consistency among slide-level embeddings across different stains. Experiments on cancer subtype classification, IHC biomarker status classification, and survival prediction show consistent gains, yielding high-quality, transferable H&E slide-level representations. The code and data are available at https://github.com/lily-zyz/CSCL.

---

## 39. M3DR: Towards Universal Multilingual Multimodal Document Retrieval

**论文链接:** [http://arxiv.org/abs/2512.03514v1](http://arxiv.org/abs/2512.03514v1)

**作者:** Adithya S Kolavi, Vyoman Jain

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出M3DR（多语言多模态文档检索）框架，解决了现有多模态文档检索系统过度以英语为中心的问题，使其能够在多种语言和文化环境中有效工作。

### 背景

多模态文档检索系统在视觉和文本内容对齐方面取得了显著进展，但大多数现有方法仍以英语为中心，限制了其在多语言环境中的有效性。

### 目的

设计M3DR框架，弥合语言差距，使多模态文档检索系统能够适用于不同的语言和文化环境。

### 方法

利用合成多语言文档数据，适用于不同的视觉语言架构和模型规模；使用对比训练学习文本和文档图像的统一表示；在22种类型多样的语言上验证能力；引入全面的基准测试，在单语言、多语言和混合语言设置下评估模型；支持单密集向量和ColBERT风格的令牌级多向量检索范式。

### 主要发现

模型能够有效跨语言迁移；在语言和书写系统变化中表现出一致的性能和适应性；提出的NetraEmbed和ColNetraEmbed模型在跨语言检索上实现了最先进性能，相对改进约150%。

### 结论

M3DR框架成功解决了多语言多模态文档检索的挑战，通过合成多语言数据和对比训练实现了跨语言和跨模态对齐，为多语言环境下的文档检索提供了有效解决方案。

### 翻译

多模态文档检索系统在对齐视觉和文本内容以进行语义搜索方面显示出强劲进展。然而，大多数现有方法仍然严重以英语为中心，限制了它们在多语言环境中的有效性。在这项工作中，我们提出了M3DR（多语言多模态文档检索），这是一个旨在弥合语言差距的框架，使其能够适用于不同的语言和文化环境。M3DR利用合成的多语言文档数据，并适用于不同的视觉语言架构和模型规模，实现了强大的跨语言和跨模态对齐。通过对比训练，我们的模型学习了文本和文档图像的统一表示，这些表示能够有效跨语言迁移。我们在22种类型多样的语言上验证了这一能力，展示了在语言和书写系统变化中的一致性能和适应性。我们进一步引入了一个捕捉真实世界多语言场景的综合基准，在单语言、多语言和混合语言设置下评估模型。M3DR适用于单密集向量和ColBERT风格的令牌级多向量检索范式。我们的模型NetraEmbed和ColNetraEmbed在跨语言检索上实现了最先进的性能，相对改进约150%。


### 论文摘要

Multimodal document retrieval systems have shown strong progress in aligning visual and textual content for semantic search. However, most existing approaches remain heavily English-centric, limiting their effectiveness in multilingual contexts. In this work, we present M3DR (Multilingual Multimodal Document Retrieval), a framework designed to bridge this gap across languages, enabling applicability across diverse linguistic and cultural contexts. M3DR leverages synthetic multilingual document data and generalizes across different vision-language architectures and model sizes, enabling robust cross-lingual and cross-modal alignment. Using contrastive training, our models learn unified representations for text and document images that transfer effectively across languages. We validate this capability on 22 typologically diverse languages, demonstrating consistent performance and adaptability across linguistic and script variations. We further introduce a comprehensive benchmark that captures real-world multilingual scenarios, evaluating models under monolingual, multilingual, and mixed-language settings. M3DR generalizes across both single dense vector and ColBERT-style token-level multi-vector retrieval paradigms. Our models, NetraEmbed and ColNetraEmbed achieve state-of-the-art performance with ~150% relative improvements on cross-lingual retrieval.

---

## 40. Learning From Limited Data and Feedback for Cell Culture Process Monitoring: A Comparative Study

**论文链接:** [http://arxiv.org/abs/2512.03460v1](http://arxiv.org/abs/2512.03460v1)

**作者:** Johnny Peng, Thanh Tung Khuat, Ellen Otte, Katarzyna Musial, Bogdan Gabrys

**发布时间:** 2025-12-03

**备注:** This is a pre-print for submitting to computers & chemical engineering journal

### GPT解析

### 总结

本研究针对细胞培养生物加工中的实时批次过程监测(BPM)挑战，对机器学习方法进行了全面基准分析，特别是在有限历史数据条件下如何有效学习和预测关键工艺变量。

### 背景

在细胞培养生物加工中，实时批次过程监测对跟踪活细胞密度、营养水平、代谢物浓度和产品滴度等关键变量至关重要，可确保产品质量和监管合规。然而，开发准确的软传感器面临历史数据有限、反馈频率低、工艺条件异构和高维感官输入等挑战。

### 目的

评估和比较多种机器学习方法在生物过程监测中的应用，特别是在历史数据量有限且相关性有限的情况下如何有效学习。

### 方法

评估了多种机器学习方法，包括特征降维、在线学习和及时学习，使用了一个计算机生成数据集和两个真实世界实验数据集进行测试。

### 主要发现

批量学习在均质环境中有效，而及时学习和在线学习在冷启动场景中表现出更强的适应性；喂料培养基成分和工艺控制策略等关键元特征显著影响模型的可转移性；将拉曼预测与滞后的离线测量相结合可以提高监测准确性。

### 结论

整合拉曼预测与滞后的离线测量为未来生物过程软传感器开发提供了有前景的方向，可有效应对数据有限和反馈不足的挑战。

### 翻译

在细胞培养生物加工中，实时批次过程监测(BPM)指的是在整个批次运行过程中对活细胞密度、营养水平、代谢物浓度和产品滴度等关键工艺变量的连续跟踪和分析。这 enables 早期发现偏差并支持及时的控制行动，以确保最佳的细胞生长和产品质量。BPM在确保生物制药制造过程的质量和监管合规性方面起着关键作用。然而，开发用于BPM的准确软传感器面临关键挑战，包括历史数据有限、反馈频率低、工艺条件异构和高维感官输入。本研究提出了针对这些挑战的机器学习方法的全面基准分析，重点关注在生物过程监测背景下从历史数据中学习，即使数据量有限且相关性有限。我们在三个数据集上评估了多种机器学习方法，包括一个计算机生成的数据集和两个真实世界的实验数据集。我们的研究结果表明，训练策略在处理有限数据和反馈方面的重要性，批量学习在均质环境中被证明是有效的，而及时学习和在线学习在冷启动场景中表现出更强的适应性。此外，我们确定了关键的元特征，如喂料培养基成分和工艺控制策略，这些特征显著影响模型的可转移性。研究结果表明，将基于拉曼的预测与滞后的离线测量相结合可以提高监测准确性，为未来生物过程软传感器开发提供了有前景的方向。


### 论文摘要

In cell culture bioprocessing, real-time batch process monitoring (BPM) refers to the continuous tracking and analysis of key process variables such as viable cell density, nutrient levels, metabolite concentrations, and product titer throughout the duration of a batch run. This enables early detection of deviations and supports timely control actions to ensure optimal cell growth and product quality. BPM plays a critical role in ensuring the quality and regulatory compliance of biopharmaceutical manufacturing processes. However, the development of accurate soft sensors for BPM is hindered by key challenges, including limited historical data, infrequent feedback, heterogeneous process conditions, and high-dimensional sensory inputs. This study presents a comprehensive benchmarking analysis of machine learning (ML) methods designed to address these challenges, with a focus on learning from historical data with limited volume and relevance in the context of bioprocess monitoring. We evaluate multiple ML approaches including feature dimensionality reduction, online learning, and just-in-time learning across three datasets, one in silico dataset and two real-world experimental datasets. Our findings highlight the importance of training strategies in handling limited data and feedback, with batch learning proving effective in homogeneous settings, while just-in-time learning and online learning demonstrate superior adaptability in cold-start scenarios. Additionally, we identify key meta-features, such as feed media composition and process control strategies, that significantly impact model transferability. The results also suggest that integrating Raman-based predictions with lagged offline measurements enhances monitoring accuracy, offering a promising direction for future bioprocess soft sensor development.

---

## 41. KeyPointDiffuser: Unsupervised 3D Keypoint Learning via Latent Diffusion Models

**论文链接:** [http://arxiv.org/abs/2512.03450v1](http://arxiv.org/abs/2512.03450v1)

**作者:** Rhys Newbury, Juyan Zhang, Tin Tran, Hanna Kurniawati, Dana Kulić

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种无监督框架，用于从点云数据中学习空间结构化的3D关键点，这些关键点可以条件化扩散模型来重建完整形状，并在不同对象类别上表现出色，关键点一致性比先前方法提高6个百分点。

### 背景

理解和以无监督方式表示3D对象的结构是计算机视觉和图形学中的一个核心挑战。大多数现有的无监督关键点方法不是为无条件生成环境设计的，限制了它们在现代3D生成管道中的应用。

### 目的

提出一个无监督框架，从点云数据中学习空间结构化的3D关键点，这些关键点作为紧凑且可解释的表示，条件化一个阐明扩散模型来重建完整形状。

### 方法

开发一个无监督框架学习空间结构化的3D关键点，并使用阐明扩散模型(EDM)进行形状重建。

### 主要发现

学习到的关键点在对象实例之间表现出可重复的空间结构，支持关键点空间中的平滑插值，表明它们捕获了几何变化，在不同对象类别上取得了强大的性能。

### 结论

相比先前的方法，关键点一致性提高了6个百分点。

### 翻译

理解和以无监督方式表示3D对象的结构仍然是计算机视觉和图形学中的一个核心挑战。大多数现有的无监督关键点方法不是为无条件生成环境设计的，限制了它们在现代3D生成管道中的使用；我们的公式明确地弥合了这一差距。我们提出了一个无监督框架，用于从点云数据中学习空间结构化的3D关键点。这些关键点作为一种紧凑且可解释的表示，条件化一个阐明扩散模型来重建完整形状。学习到的关键点在对象实例之间表现出可重复的空间结构，并支持关键点空间中的平滑插值，表明它们捕获了几何变化。我们的方法在不同对象类别上取得了强大的性能，与先前的方法相比，关键点一致性提高了6个百分点。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是如何在没有监督的情况下学习3D物体的空间结构化关键点问题。这个问题在计算机视觉和图形学中非常重要，因为它能够帮助机器理解3D物体的结构语义，实现物体对应、重建和操作等任务。现有的无监督关键点方法大多不是为无条件生成设计的，限制了它们在现代3D生成管道中的应用。而这项工作的关键点能够在不同物体实例间保持可重复的空间结构，并支持平滑插值，从而捕获几何变化。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了扩散模型在3D形状生成方面的成功，引入了一个以学习的关键点为条件的潜在扩散模型。他们发现大多数现有方法依赖显式结构先验或确定性解码器，因此设计了一个概率生成解码器直接从关键点重建形状。方法的核心是使用Transformer-based编码器提取关键点，结合Elucidated Diffusion Model进行形状重建，并应用多种几何正则化技术（如Chamfer损失和变形一致性损失）确保关键点的空间意义和一致性。他们的设计受到了点云处理、无监督表示学习和扩散生成模型等多个领域工作的启发。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将提取的关键点作为潜在表示，条件化去噪扩散过程以生成完整的3D形状，从而学习结构先验并提取语义一致且空间信息丰富的关键点。整体流程包括：1) 使用Transformer编码器将输入点云编码为几何关键点和辅助特征，通过注意力机制确保关键点位于点云凸包内；2) 采用Elucidated Diffusion Model从关键点生成密集形状，使用课程学习噪声调度策略；3) 设计多种损失函数（形状重建、Chamfer、变形一致性、最远点采样和KL散度）来训练模型，确保关键点的几何一致性和语义意义；4) 通过分离训练VAE和扩散模型来避免潜在空间崩溃问题。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出利用去噪扩散学习无监督3D关键点的生成性公式；2) 引入由显式关键点和辅助特征组成的结构化潜在空间，并应用几何正则化技术；3) 实现强关键点一致性和高保真度形状生成。相比之前的工作，KeyPointDiffuser不依赖于参考网格（如KeypointDeformer），直接生成点云实现更灵活的形状合成；与KeyGrid不同，它不需要输入点特征支持无条件生成；相比Skeleton Merger，它不直接依赖输入形状；与SC3K相比，它不仅能够发现关键点还能进行形状重建；与点云扩散模型(DPM)相比，它使用更紧凑的结构化表示而非高维潜在代码。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'KeyPointDiffuser通过结合无监督关键点学习和潜在扩散模型，实现了在无需标注数据的情况下学习语义一致且空间结构化的3D关键点，并利用这些关键点进行高质量的3D形状重建和生成。'}


### 论文摘要

Understanding and representing the structure of 3D objects in an unsupervised manner remains a core challenge in computer vision and graphics. Most existing unsupervised keypoint methods are not designed for unconditional generative settings, restricting their use in modern 3D generative pipelines; our formulation explicitly bridges this gap. We present an unsupervised framework for learning spatially structured 3D keypoints from point cloud data. These keypoints serve as a compact and interpretable representation that conditions an Elucidated Diffusion Model (EDM) to reconstruct the full shape. The learned keypoints exhibit repeatable spatial structure across object instances and support smooth interpolation in keypoint space, indicating that they capture geometric variation. Our method achieves strong performance across diverse object categories, yielding a 6 percentage-point improvement in keypoint consistency compared to prior approaches.

---

## 42. An Analysis of LIGO Glitches Using t-SNE During the First Part of the Fourth LIGO-Virgo-KAGRA Observing Run

**论文链接:** [http://arxiv.org/abs/2512.03440v1](http://arxiv.org/abs/2512.03440v1)

**作者:** Tabata Aira Ferreira, Gabriela González, Osvaldo Salas

**发布时间:** 2025-12-03

**备注:** 30 pages, 26 figures

### GPT解析

### 总结

本文分析了LIGO在第四次观测运行初期数据中观察到的噪声瞬态现象，使用机器学习技术研究故障群组行为，并确定其与环境或仪器条件的相关性。

### 背景

LIGO（激光干涉引力波天文台）在第四次观测运行期间记录了包含噪声瞬态的数据，这些噪声可能影响引力波探测的准确性。

### 目的

识别和分析LIGO数据中的故障群组，研究它们随时间的变化模式，以及与外部环境或仪器条件的关联性。

### 方法

使用t-SNE无监督机器学习技术分析故障群组行为，应用层次聚类结合轮廓系数确定最佳分组数量，然后跟踪这些组组并分析其与环境或仪器条件的相关性。

### 主要发现

Livingston天文台最常见的故障是季节性的，与地面运动有关；而Hanford天文台最常见的故障则与仪器条件相关。

### 结论

不同LIGO观测站的主要故障类型存在差异，Livingston的故障主要受环境因素影响，Hanford的故障则主要与仪器条件相关。

### 翻译

本文分析了在第四次观测运行初期LIGO数据中观察到的噪声瞬态现象，使用无监督机器学习技术t-分布随机邻域嵌入（t-SNE）来检查故障群组的行为。基于t-SNE输出，我们应用层次聚类结合轮廓系数来确定最佳组数。然后我们随时间跟踪这些组组，并研究它们的发生与环境或仪器条件之间的相关性。在Livingston天文台，O4a期间最常见的故障是季节性的，并与地面运动有关，而在Hanford，最常见的故障则与仪器条件相关。


### 论文摘要

This paper presents an analysis of noise transients observed in LIGO data during the first part of the fourth observing run, using the unsupervised machine learning technique t-distributed Stochastic Neighbor Embedding (t-SNE) to examine the behavior of glitch groups. Based on the t-SNE output, we apply Agglomerative Clustering in combination with the Silhouette Score to determine the optimal number of groups. We then track these groups over time and investigate correlations between their occurrence and environmental or instrumental conditions. At the Livingston observatory, the most common glitches during O4a were seasonal and associated with ground motion, whereas at Hanford, the most prevalent glitches were related to instrumental conditions.

---

## 43. Label-Efficient Hyperspectral Image Classification via Spectral FiLM Modulation of Low-Level Pretrained Diffusion Features

**论文链接:** [http://arxiv.org/abs/2512.03430v1](http://arxiv.org/abs/2512.03430v1)

**作者:** Yuzhen Hu, Biplab Banerjee, Saurabh Prasad

**发布时间:** 2025-12-03

**备注:** Accepted to the ICML 2025 TerraBytes Workshop (June 9, 2025)

### GPT解析

### 总结

提出了一种标签高效的框架，利用预训练扩散模型的空间特征和光谱信息融合方法，解决了高光谱成像中的低空间分辨率和稀疏标注问题，在仅使用稀疏标签的情况下实现了优于最先进方法的性能。

### 背景

高光谱成像能够实现详细的地物覆盖分类，但面临低空间分辨率和稀疏标注的显著挑战，限制了现有方法的性能。

### 目的

开发一种标签高效的方法，有效利用预训练扩散模型的空间特征，并实现光谱和空间信息的融合，以解决高光谱成像中的分类问题。

### 方法

利用冻结在自然图像上预训练的扩散模型，从高分辨率解码器层的早期去噪时间步提取低级表示；引入基于FiLM的轻量级融合模块，自适应地利用光谱线索调制空间特征，实现光谱和空间信息的有效整合。

### 主要发现

在两个超光谱数据集上，该方法仅使用提供的稀疏训练标签就优于最先进方法；消融研究证明了扩散衍生特征和光谱感知融合的有效性；预训练扩散模型能够支持遥感任务的领域无关表示学习。

### 结论

预训练扩散模型可以支持领域无关、标签高效的表示学习，适用于遥感和其他科学成像任务；光谱和空间信息的有效融合是解决高光谱成像分类问题的关键。

### 翻译

高光谱成像(HSI)能够实现详细的地物覆盖分类，然而低空间分辨率和稀疏标注带来了显著挑战。我们提出了一种标签高效的框架，利用自然图像预训练的冻结扩散模型的空间特征。我们的方法从高分辨率解码器层的早期去噪时间步提取低级表示，这些表示有效地转移到HSI的低纹理结构中。为了整合光谱和空间信息，我们引入了一种基于FiLM的轻量级融合模块，利用光谱线索自适应地调制冻结的空间特征，从而在稀疏监督下实现鲁棒的多模态学习。在两个最近的超光谱数据集上的实验表明，我们的方法仅使用提供的稀疏训练标签就优于最先进的方法。消融研究进一步强调了扩散衍生特征和光谱感知融合的优势。总体而言，我们的结果表明预训练扩散模型可以支持遥感和其他科学成像任务的领域无关、标签高效的表示学习。


### 论文摘要

Hyperspectral imaging (HSI) enables detailed land cover classification, yet low spatial resolution and sparse annotations pose significant challenges. We present a label-efficient framework that leverages spatial features from a frozen diffusion model pretrained on natural images. Our approach extracts low-level representations from high-resolution decoder layers at early denoising timesteps, which transfer effectively to the low-texture structure of HSI. To integrate spectral and spatial information, we introduce a lightweight FiLM-based fusion module that adaptively modulates frozen spatial features using spectral cues, enabling robust multimodal learning under sparse supervision. Experiments on two recent hyperspectral datasets demonstrate that our method outperforms state-of-the-art approaches using only the provided sparse training labels. Ablation studies further highlight the benefits of diffusion-derived features and spectral-aware fusion. Overall, our results indicate that pretrained diffusion models can support domain-agnostic, label-efficient representation learning for remote sensing and broader scientific imaging tasks.

---

## 44. Associating Healthcare Teamwork with Patient Outcomes for Predictive Analysis

**论文链接:** [http://arxiv.org/abs/2512.03296v1](http://arxiv.org/abs/2512.03296v1)

**作者:** Hsiao-Ying Lu, Kwan-Liu Ma

**发布时间:** 2025-12-02

### GPT解析

### 总结

这项研究探讨了医疗专业人员合作对癌症治疗效果的影响，通过电子健康记录系统捕获的合作数据，应用人工智能和机器学习方法分析合作网络，发现能够预测患者生存率的关键合作特征，并验证了这些特征在改善医疗结果方面的实际应用价值。

### 背景

癌症治疗效果不仅受临床和人口统计学因素影响，还受医疗团队协作的影响。然而，以往的研究大多忽视了人类协作在塑造患者生存方面的潜在作用。

### 目的

本研究旨在揭示通过电子健康记录系统捕获的医疗专业人员合作对癌症患者结果的影响，开发一种应用人工智能方法来发现合作网络中预测患者生存率的信号。

### 方法

研究将电子健康记录介导的医疗专业人员互动建模为网络，并应用机器学习技术来检测嵌入在这些合作中的患者生存预测信号。模型经过交叉验证以确保可推广性，并通过识别与改善结果相关的关键网络特征来解释预测结果。

### 主要发现

研究识别出与改善结果相关的关键网络特征，这些特征得到了临床专家和文献的验证，证实了它们在现实世界应用中的潜在价值。

### 结论

这项工作为利用协作的数字踪迹和人工智能来评估和改进基于团队的医疗保健提供了实用的工作流程。该方法可能适用于涉及复杂协作的其他领域，并为支持医疗保健交付中的数据驱动干预措施提供可行的见解。

### 翻译

癌症治疗效果不仅受临床和人口统计学因素的影响，还受医疗团队协作的影响。然而，先前的工作大多忽视了人类协作在塑造患者生存方面的潜在作用。本文提出了一种应用人工智能方法，揭示通过电子健康记录系统捕获的医疗专业人员协作对癌症患者结果的影响。我们将EHR介导的医疗专业人员互动建模为网络，并应用机器学习技术来检测嵌入在这些协作中的患者生存预测信号。我们的模型经过交叉验证以确保可推广性，并通过识别与改善结果相关的关键网络特征来解释预测结果。重要的是，临床专家和文献验证了所识别的关键协作特征的相关性，强化了它们在现实世界应用中的潜力。这项工作为利用协作的数字踪迹和人工智能来评估和改进基于团队的医疗保健做出了贡献。该方法可能适用于涉及复杂协作的其他领域，并为支持医疗保健交付中的数据驱动干预措施提供可行的见解。


### 论文摘要

Cancer treatment outcomes are influenced not only by clinical and demographic factors but also by the collaboration of healthcare teams. However, prior work has largely overlooked the potential role of human collaboration in shaping patient survival. This paper presents an applied AI approach to uncovering the impact of healthcare professionals' (HCPs) collaboration-captured through electronic health record (EHR) systems-on cancer patient outcomes. We model EHR-mediated HCP interactions as networks and apply machine learning techniques to detect predictive signals of patient survival embedded in these collaborations. Our models are cross validated to ensure generalizability, and we explain the predictions by identifying key network traits associated with improved outcomes. Importantly, clinical experts and literature validate the relevance of the identified crucial collaboration traits, reinforcing their potential for real-world applications. This work contributes to a practical workflow for leveraging digital traces of collaboration and AI to assess and improve team-based healthcare. The approach is potentially transferable to other domains involving complex collaboration and offers actionable insights to support data-informed interventions in healthcare delivery.

---

## 45. Perch 2.0 transfers 'whale' to underwater tasks

**论文链接:** [http://arxiv.org/abs/2512.03219v1](http://arxiv.org/abs/2512.03219v1)

**作者:** Andrea Burns, Lauren Harrell, Bart van Merriënboer, Vincent Dumoulin, Jenny Hamer, Tom Denton

**发布时间:** 2025-12-02

**备注:** 8 pages, 3 figures, 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: AI for Non-Human Animal Communication

### GPT解析

### 总结

Perch 2.0是一个监督式生物声学基础模型，预训练数据包含14,597个物种，在多个基准测试中表现优异。尽管训练数据中几乎没有海洋哺乳动物音频，但通过少样本迁移学习，该模型在海洋哺乳动物分类任务上表现优于其他模型。

### 背景

Perch 2.0的训练数据中几乎没有海洋哺乳动物音频或类别，需要评估其在海洋哺乳动物和水下音频任务上的性能。

### 目的

通过少样本迁移学习评估Perch 2.0在海洋哺乳动物和水下音频任务上的性能，并与其它预训练生物声学模型进行比较。

### 方法

使用从Perch 2.0基础模型生成的嵌入进行线性探测，并与Perch 1.0、SurfPerch、AVES-bio、BirdAVES、Birdnet V2.3等模型进行比较。

### 主要发现

Perch 2.0模型生成的嵌入在少样本迁移学习中表现一致良好，在大多数任务上通常优于其他嵌入模型。

### 结论

推荐在开发具有少量标记示例的海洋哺乳动物分类器新线性分类器时使用Perch 2.0。

### 翻译

Perch 2.0是一个监督式生物声学基础模型，在14,597个物种上进行了预训练，包括鸟类、哺乳动物、两栖动物和昆虫，并在多个基准测试中取得了最先进的性能。鉴于Perch 2.0的训练数据中几乎不包含海洋哺乳动物音频或类别，我们通过少样本迁移学习评估了Perch 2.0在海洋哺乳动物和水下音频任务上的性能。我们使用从该基础模型生成的嵌入进行线性探测，并将性能与其他预训练的生物声学模型进行比较。特别是，我们将Perch 2.0与之前的多物种鲸鱼模型、Perch 1.0、SurfPerch、AVES-bio、BirdAVES和Birdnet V2.3模型进行了比较，这些模型都有用于迁移学习和敏捷建模的开源工具。我们表明，Perch 2.0模型生成的嵌入在少样本迁移学习中具有一致的高性能，通常在大多数任务上优于其他嵌入模型，因此推荐在开发具有少量标记示例的海洋哺乳动物分类新线性分类器时使用。


### 论文摘要

Perch 2.0 is a supervised bioacoustics foundation model pretrained on 14,597 species, including birds, mammals, amphibians, and insects, and has state-of-the-art performance on multiple benchmarks. Given that Perch 2.0 includes almost no marine mammal audio or classes in the training data, we evaluate Perch 2.0 performance on marine mammal and underwater audio tasks through few-shot transfer learning. We perform linear probing with the embeddings generated from this foundation model and compare performance to other pretrained bioacoustics models. In particular, we compare Perch 2.0 with previous multispecies whale, Perch 1.0, SurfPerch, AVES-bio, BirdAVES, and Birdnet V2.3 models, which have open-source tools for transfer-learning and agile modeling. We show that the embeddings from the Perch 2.0 model have consistently high performance for few-shot transfer learning, generally outperforming alternative embedding models on the majority of tasks, and thus is recommended when developing new linear classifiers for marine mammal classification with few labeled examples.

---

## 46. 论文ID: 2512.03210v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03210v1.json'

---

## 47. Contrastive Deep Learning for Variant Detection in Wastewater Genomic Sequencing

**论文链接:** [http://arxiv.org/abs/2512.03158v1](http://arxiv.org/abs/2512.03158v1)

**作者:** Adele Chinda, Richmond Azumah, Hemanth Demakethepalli Venkateswara

**发布时间:** 2025-12-02

**备注:** 13 pages, 4 figures

### GPT解析

### 总结

本文提出了一种使用向量量化变分自编码器进行无监督病毒变异检测的综合框架，能够从k-mer标记化序列中学习基因组模式的离散码本，无需参考基因组或变异标签。该方法通过掩码重建预训练和对比学习提高了鲁棒性和判别性。在SARS-CoV-2废水测序数据上评估，表现出高准确率，且嵌入维度显著影响变异判别能力。

### 背景

废水基因组监测已成为人口水平病毒监测的有力工具，可提供整个社区循环病毒变异的全面见解。然而，该方法面临重大计算挑战，包括高测序噪声、低病毒覆盖率、片段化读长和完全缺乏标记的变异注释。传统基于参考的变异 calling 管道难以处理新型突变，需要大量计算资源。

### 目的

开发一个全面的无监督病毒变异检测框架，使用向量量化变分自编码器学习基因组模式的离散码本，无需参考基因组或变异标签。

### 方法

使用VQ-VAE架构，从k-mer标记化序列中学习。扩展基础VQ-VAE架构，包括掩码重建预训练（提高对缺失数据的鲁棒性）和对比学习（实现高度判别性的嵌入）。在约100,000个读长的SARS-CoV-2废水测序数据上进行评估。

### 主要发现

VQ-VAE达到99.52%的平均标记级别准确率和56.33%的精确序列匹配率，同时保持19.73%的码本利用率（512个码中101个激活），证明了高效的离散表示学习。对比微调显示，64维嵌入实现+35%的轮廓分数改进（从0.31到0.42），128维嵌入实现+42%的改进（从0.31到0.44），清楚展示了嵌入维度对变异判别能力的影响。

### 结论

该无参考框架为基因组监测提供了可扩展、可解释的方法，直接应用于公共卫生监测。

### 翻译

废水基因组监测已成为人口水平病毒监测的有力工具，能够提供整个社区循环病毒变异的全面见解。然而，这种方法面临重大计算挑战，源于高测序噪声、低病毒覆盖率、片段化读长以及完全缺乏标记的变异注释。传统的基于参考的变异 calling 管道难以处理新型突变，需要大量计算资源。我们提出了一个使用向量量化变分自编码器的无监督病毒变异检测综合框架，它从k-mer标记化序列中学习基因组模式的离散码本，不需要参考基因组或变异标签。我们的方法通过掩码重建预训练扩展基础VQ-VAE架构，提高对缺失数据的鲁棒性，并通过对比学习实现高度判别性的嵌入。在包含约100,000个读长的SARS-CoV-2废水测序数据上评估，我们的VQ-VAE实现了99.52%的平均标记级别准确率和56.33%的精确序列匹配率，同时保持19.73%的码本利用率（512个码中101个激活），证明了高效的离散表示学习。不同投影维度的对比微调带来显著的聚类改进：64维嵌入实现+35%的轮廓分数改进（从0.31到0.42），而128维嵌入实现+42%的改进（从0.31到0.44），清楚展示了嵌入维度对变异判别能力的影响。我们的无参考框架为基因组监测提供了可扩展、可解释的方法，直接应用于公共卫生监测。


### 论文摘要

Wastewater-based genomic surveillance has emerged as a powerful tool for population-level viral monitoring, offering comprehensive insights into circulating viral variants across entire communities. However, this approach faces significant computational challenges stemming from high sequencing noise, low viral coverage, fragmented reads, and the complete absence of labeled variant annotations. Traditional reference-based variant calling pipelines struggle with novel mutations and require extensive computational resources. We present a comprehensive framework for unsupervised viral variant detection using Vector-Quantized Variational Autoencoders (VQ-VAE) that learns discrete codebooks of genomic patterns from k-mer tokenized sequences without requiring reference genomes or variant labels. Our approach extends the base VQ-VAE architecture with masked reconstruction pretraining for robustness to missing data and contrastive learning for highly discriminative embeddings. Evaluated on SARS-CoV-2 wastewater sequencing data comprising approximately 100,000 reads, our VQ-VAE achieves 99.52% mean token-level accuracy and 56.33% exact sequence match rate while maintaining 19.73% codebook utilization (101 of 512 codes active), demonstrating efficient discrete representation learning. Contrastive fine-tuning with different projection dimensions yields substantial clustering improvements: 64-dimensional embeddings achieve +35% Silhouette score improvement (0.31 to 0.42), while 128-dimensional embeddings achieve +42% improvement (0.31 to 0.44), clearly demonstrating the impact of embedding dimensionality on variant discrimination capability. Our reference-free framework provides a scalable, interpretable approach to genomic surveillance with direct applications to public health monitoring.

---

## 48. 论文ID: 2512.03043v2

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03043v2.json'

---

## 49. Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning

**论文链接:** [http://arxiv.org/abs/2512.02914v1](http://arxiv.org/abs/2512.02914v1)

**作者:** Zhonghao He, Tianyi Qiu, Hirokazu Shirado, Maarten Sap

**发布时间:** 2025-12-02

**备注:** NeurIPS 2025

### GPT解析

### 总结

本研究提出了一种基于贝叶斯统计中鞅属性的系统评估框架，用于测量大型语言模型推理中的信念固化问题。研究发现在多个开放式问题领域中，模型普遍存在信念固化现象，即当前信念正向预测未来信念更新。提出的鞅分数能有效衡量这一问题，并能作为推理过程真实寻求能力的代理指标。

### 背景

推理技术的最新进展显著提高了大型语言模型的性能，提高了人们对它们提供准确、真实和可靠信息能力的期望。然而，新出现的证据表明，迭代推理可能导致信念固化和确认偏见，而非增强真实寻求行为。

### 目的

提出一个系统性的评估框架，用于衡量大型语言模型推理中的信念固化问题，利用贝叶斯统计中的鞅属性，并开发一种无监督的测量方法。

### 方法

利用贝叶斯统计中的鞅属性，即理性信念更新下未来信念的期望值应等于当前信念。提出无监督的、基于回归的鞅分数来测量这一属性的违反，检测偏离贝叶斯更新新证据能力的情况。在多个开放式问题领域中进行测试，包括事件预测、价值判断问题和学术论文评审。

### 主要发现

在多个开放式问题领域中，发现信念固化现象普遍存在于各种模型和设置中，当前信念正向预测未来信念更新。研究确定了更容易出现信念固化的模型、推理技术和领域。验证了鞅分数可以预测在有真实标签可用的问题领域上的真实准确度。

### 结论

虽然鞅分数设计为无监督指标，即使在无法获取真实标签的领域也能工作，但它能有效作为推理过程真实寻求能力的代理指标，帮助评估大型语言模型的推理质量。

### 翻译

最近的推理技术进展显著提高了大型语言模型的性能，提高了人们对它们提供准确、真实和可靠信息能力的期望。然而，新出现的证据表明，迭代推理可能导致信念固化和确认偏见，而非增强真实寻求行为。在本研究中，我们通过利用贝叶斯统计中的鞅属性，提出了一个用于评估大型语言模型推理中信念固化的系统框架。这一属性意味着，在理性信念更新下，未来信念的期望值应保持等于当前信念，即信念更新从当前信念来看是不可预测的。我们提出了无监督的、基于回归的鞅分数来测量这一属性的违反，这表明偏离了贝叶斯更新新证据的能力。在事件预测、价值判断问题和学术论文评审等开放式问题领域中，我们发现这种违反现象在模型和设置中普遍存在，其中当前信念正向预测未来信念更新，这种现象我们称为信念固化。我们确定了更容易出现信念固化的模型、推理技术和领域。最后，我们通过显示鞅分数在有真实标签可用的问题领域上预测真实准确度，验证了鞅分数的有效性。这表明，虽然设计为无监督指标，即使在无法获取真实标签的领域也能工作，但鞅分数是推理过程真实寻求能力的一个有用代理指标。


### 论文摘要

Recent advances in reasoning techniques have substantially improved the performance of large language models (LLMs), raising expectations for their ability to provide accurate, truthful, and reliable information. However, emerging evidence suggests that iterative reasoning may foster belief entrenchment and confirmation bias, rather than enhancing truth-seeking behavior. In this study, we propose a systematic evaluation framework for belief entrenchment in LLM reasoning by leveraging the Martingale property from Bayesian statistics. This property implies that, under rational belief updating, the expected value of future beliefs should remain equal to the current belief, i.e., belief updates are unpredictable from the current belief. We propose the unsupervised, regression-based Martingale Score to measure violations of this property, which signal deviation from the Bayesian ability of updating on new evidence. In open-ended problem domains including event forecasting, value-laden questions, and academic paper review, we find such violations to be widespread across models and setups, where the current belief positively predicts future belief updates, a phenomenon which we term belief entrenchment. We identify the models, reasoning techniques, and domains more prone to belief entrenchment. Finally, we validate the Martingale Score by showing that it predicts ground-truth accuracy on problem domains where ground truth labels are available. This indicates that, while designed as an unsupervised metric that operates even in domains without access to ground truth, the Martingale Score is a useful proxy of the truth-seeking ability of a reasoning process.

---

## 50. The future of AI in critical mineral exploration

**论文链接:** [http://arxiv.org/abs/2512.02879v1](http://arxiv.org/abs/2512.02879v1)

**作者:** Jef Caers

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种基于人工智能的矿产勘探新科学方法，旨在解决能源转型背景下关键矿产新发现减少的问题。

### 背景

全球能源转型通过增加电气化使人们关注关键矿产勘探，尽管投资增加，但过去二十年间新发现数量却在减少。

### 目的

实施AI作为矿产勘探严格科学方法的推动者，减少认知偏差和假阳性，降低勘探成本。

### 方法

提出基于贝叶斯主义和证伪原则的新科学方法，将数据获取视为证伪人类生成假设的手段，使用可验证指标和理性决策决定数据获取策略，提供可在任何勘探活动中使用的实用协议，并需要无监督学习方法和人机循环AI算法支持。

### 主要发现

无明确具体发现，主要提出了一种创新方法框架。

### 结论

通过AI实施的新科学方法可以有效解决矿产勘探中新发现减少的问题，提高勘探效率和准确性。

### 翻译

能源转型通过增加电气化已将全球注意力转向关键矿产勘探。尽管投资增加，但过去二十年间新发现数量却在减少。我在此提出解决此问题的方案，即实施AI作为矿产勘探严格科学方法的推动者，旨在减少认知偏差和假阳性，降低勘探成本。我提出了一种基于贝叶斯主义和证伪原则的新科学方法。在此方法中，数据获取首先被视为证伪人类生成假设的手段。决定下一步获取什么数据通过可验证指标进行量化，并基于理性决策。提供了可在任何勘探活动中使用的实用协议。然而，为了使该协议实用化，需要各种形式的人工智能。我将论证最重要的形式是：一种新颖的无监督学习方法，与领域专家合作以更好地理解数据并生成多个竞争性的地质假设；以及人机循环AI算法，能够最优规划各种地质、地球物理、地球化学和钻探数据获取，其中地质假设的不确定性减少优先于品位和吨位的不确定性减少。


### 论文摘要

The energy transition through increased electrification has put the worlds attention on critical mineral exploration Even with increased investments a decrease in new discoveries has taken place over the last two decades Here I propose a solution to this problem where AI is implemented as the enabler of a rigorous scientific method for mineral exploration that aims to reduce cognitive bias and false positives drive down the cost of exploration I propose a new scientific method that is based on a philosophical approach founded on the principles of Bayesianism and falsification In this approach data acquisition is in the first place seen as a means to falsify human generated hypothesis Decision of what data to acquire next is quantified with verifiable metrics and based on rational decision making A practical protocol is provided that can be used as a template in any exploration campaign However in order to make this protocol practical various form of artificial intelligence are needed I will argue that the most important form are one novel unsupervised learning methods that collaborate with domain experts to better understand data and generate multiple competing geological hypotheses and two humanintheloop AI algorithms that can optimally plan various geological geophysical geochemical and drilling data acquisition where uncertainty reduction of geological hypothesis precedes the uncertainty reduction on grade and tonnage

---

## 51. SwarmDiffusion: End-To-End Traversability-Guided Diffusion for Embodiment-Agnostic Navigation of Heterogeneous Robots

**论文链接:** [http://arxiv.org/abs/2512.02851v2](http://arxiv.org/abs/2512.02851v2)

**作者:** Iana Zhura, Sausar Karaf, Faryal Batool, Nipun Dhananjaya Weerakkodi Mudalige, Valerii Serpiva, Ali Alridha Abdulkarim, Aleksey Fedoseev, Didar Seyidov, Hajira Amjad, Dzmitry Tsetserukou

**发布时间:** 2025-12-02

**备注:** This work has been submitted for publication and is currently under review

### GPT解析

### 总结

SwarmDiffusion是一种轻量级端到端扩散模型，能够从单个RGB图像联合预测可通行性和生成可行轨迹，无需手工提示工程，并能跨不同机器人平台迁移。

### 背景

视觉可通行性估计对自主导航至关重要，但现有的基于VLM的方法依赖手工制作的提示，泛化能力差，且只输出可通行性地图，轨迹生成留给外部规划器。

### 目的

开发一个无需提示工程、能够跨机器人平台迁移的统一可通行性推理和轨迹生成方法。

### 方法

SwarmDiffusion模型结合了基于随机航点采样、贝塞尔平滑和正则化的轨迹构建管道，确保路径的连通性、安全性、方向性和稀疏性，无需注释或规划器生成路径。

### 主要发现

该方法在室内环境和两种机器人形态上实现了80-100%的导航成功率，0.09秒的推理时间，仅使用500个额外视觉样本即可适应新机器人，在模拟和真实世界试验中能可靠推广到未见环境。

### 结论

SwarmDiffusion提供了一种可扩展的、无需提示的统一可通行性推理和轨迹生成方法，能够物理一致且可通行地生成路径。

### 翻译

视觉可通行性估计对自主导航至关重要，但现有的基于VLM的方法依赖手工制作的提示，泛化能力差，且只输出可通行性地图，轨迹生成留给缓慢的外部规划器。我们提出了SwarmDiffusion，一种轻量级端到端扩散模型，能够从单个RGB图像联合预测可通行性并生成可行轨迹。为消除对注释或规划器生成路径的需求，我们引入了一个基于随机航点采样、贝塞尔平滑和正则化的无规划器轨迹构建管道，确保连通性、安全性、方向性和路径稀疏性。这使得无需演示即可学习稳定的运动先验。SwarmDiffusion利用VLM衍生的监督而不需要提示工程，并将扩散过程条件化为紧凑的 embodiment 状态，产生物理一致且可通行的路径，并能跨不同机器人平台迁移。在室内环境和两种形态(四足和空中)上，该方法实现了80-100%的导航成功率和0.09秒的推理时间，并且仅使用500个额外的视觉样本就能适应新机器人。它在模拟和真实世界试验中能可靠地推广到未见环境，提供了一种可扩展的、无需提示的统一可通行性推理和轨迹生成方法。


### 论文摘要

Visual traversability estimation is critical for autonomous navigation, but existing VLM-based methods rely on hand-crafted prompts, generalize poorly across embodiments, and output only traversability maps, leaving trajectory generation to slow external planners. We propose SwarmDiffusion, a lightweight end-to-end diffusion model that jointly predicts traversability and generates a feasible trajectory from a single RGB image. To remove the need for annotated or planner-produced paths, we introduce a planner-free trajectory construction pipeline based on randomized waypoint sampling, Bezier smoothing, and regularization enforcing connectivity, safety, directionality, and path thinness. This enables learning stable motion priors without demonstrations. SwarmDiffusion leverages VLM-derived supervision without prompt engineering and conditions the diffusion process on a compact embodiment state, producing physically consistent, traversable paths that transfer across different robot platforms. Across indoor environments and two embodiments (quadruped and aerial), the method achieves 80-100% navigation success and 0.09s inference, and adapts to a new robot using only-500 additional visual samples. It generalizes reliably to unseen environments in simulation and real-world trials, offering a scalable, prompt-free approach to unified traversability reasoning and trajectory generation.

---

## 52. Defense That Attacks: How Robust Models Become Better Attackers

**论文链接:** [http://arxiv.org/abs/2512.02830v2](http://arxiv.org/abs/2512.02830v2)

**作者:** Mohamed Awad, Mahmoud Akrm, Walid Gomaa

**发布时间:** 2025-12-02

### GPT解析

### 总结

本研究发现对抗训练虽然提高了模型鲁棒性，但意外地增加了对抗样本的可转移性，引入了新的生态系统风险。

### 背景

深度学习在计算机视觉领域取得了巨大成功，但仍然容易受到对抗攻击。对抗训练是提高模型鲁棒性的主要防御方法。

### 目的

研究对抗训练是否无意中增加了对抗样本的可转移性。

### 方法

训练了一个包含36种不同模型（包括CNNs和ViTs）的多样化模型集合，并进行了全面的可转移性实验。

### 主要发现

存在明显的悖论：对抗训练的模型产生的扰动比标准模型更有效地转移，这引入了新的生态系统风险。

### 结论

鲁棒性评估应该不仅评估模型对转移攻击的抵抗力，还应该评估其产生可转移对抗样本的倾向。

### 翻译

深度学习在计算机视觉领域取得了巨大成功，但仍然容易受到对抗攻击。对抗训练是旨在提高模型鲁棒性的主要防御方法。然而，它对攻击可转移性的影响尚未得到充分探索。在这项工作中，我们问对抗训练是否无意中增加了对抗样本的可转移性。为了回答这个问题，我们训练了一个包含36种模型（包括CNNs和ViTs）的多样化模型集合，并进行了全面的可转移性实验。我们的结果揭示了一个明显的悖论：对抗训练的模型产生的扰动比标准模型更有效地转移，这引入了新的生态系统风险。为了实现可重复性和进一步研究，我们发布了所有模型、代码和实验脚本。此外，我们主张鲁棒性评估应该不仅评估模型对转移攻击的抵抗力，还应该评估其产生可转移对抗样本的倾向。


### 论文摘要

Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples.

---

## 53. 论文ID: 2512.02712v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.02712v1.json'

---

## 54. CREST: Universal Safety Guardrails Through Cluster-Guided Cross-Lingual Transfer

**论文链接:** [http://arxiv.org/abs/2512.02711v1](http://arxiv.org/abs/2512.02711v1)

**作者:** Lavish Bansal, Naman Mishra

**发布时间:** 2025-12-02

**备注:** 8 Pages, 5 Figures, Under Review

### GPT解析

### 总结

这篇论文介绍了一种名为CREST的参数高效多语言安全分类模型，该模型仅用0.5B参数支持100种语言，通过在13种高资源语言上训练并利用跨语言传递技术，有效解决了低资源语言的安全护栏问题。

### 背景

确保大型语言模型的内容安全对其在现实世界应用中的部署至关重要，但现有安全护栏主要针对高资源语言设计，导致使用低资源语言沟通的世界人口大部分没有得到充分代表。

### 目的

解决低资源语言安全护栏不足的问题，开发一个能够支持多种语言的安全分类模型，特别是针对那些资源有限的低资源语言。

### 方法

研究人员提出了CREST（跨语言高效安全传递），一个参数高效的多语言安全分类模型，通过在13种高资源语言子集上进行训练，利用基于聚类的跨语言传递技术，从少数语言扩展到100种语言，实现对低资源语言的有效泛化。

### 主要发现

CREST在六个安全基准测试上的综合评估表明，它优于现有同规模的最先进安全护栏，并且在参数量显著更大（2.5B参数及以上）的模型上取得了有竞争力的结果。

### 结论

研究结果强调了语言特定安全护栏的局限性，并强调了开发能够有效扩展以服务全球人口的通用、语言无关安全系统的重要性。

### 翻译

确保大型语言模型的内容安全对于它们在现实世界应用中的部署至关重要。然而，现有的安全护栏主要针对高资源语言进行定制，导致使用低资源语言沟通的世界人口大部分没有得到充分代表。为此，我们引入了CREST（跨语言高效安全传递），这是一个参数高效的多语言安全分类模型，仅用0.5B参数就支持100种语言。通过在仅选择的13种高资源语言子集上进行训练，我们的模型利用基于聚类的跨语言传递技术，从少数语言扩展到100种语言，从而能够有效推广到未见过的低资源和高资源语言。这种方法解决了低资源环境中的训练数据有限这一挑战。我们在六个安全基准测试上进行了全面评估，证明CREST优于现有同规模的最先进安全护栏，并且在参数量显著更大（2.5B参数及以上）的模型上取得了有竞争力的结果。我们的研究结果强调了语言特定安全护栏的局限性，并强调了开发能够有效扩展以服务全球人口的通用、语言无关安全系统的重要性。


### 论文摘要

Ensuring content safety in large language models (LLMs) is essential for their deployment in real-world applications. However, existing safety guardrails are predominantly tailored for high-resource languages, leaving a significant portion of the world's population underrepresented who communicate in low-resource languages. To address this, we introduce CREST (CRoss-lingual Efficient Safety Transfer), a parameter-efficient multilingual safety classification model that supports 100 languages with only 0.5B parameters. By training on a strategically chosen subset of only 13 high-resource languages, our model utilizes cluster-based cross-lingual transfer from a few to 100 languages, enabling effective generalization to both unseen high-resource and low-resource languages. This approach addresses the challenge of limited training data in low-resource settings. We conduct comprehensive evaluations across six safety benchmarks to demonstrate that CREST outperforms existing state-of-the-art guardrails of comparable scale and achieves competitive results against models with significantly larger parameter counts (2.5B parameters and above). Our findings highlight the limitations of language-specific guardrails and underscore the importance of developing universal, language-agnostic safety systems that can scale effectively to serve global populations.

---

## 55. Unsupervised Structural Scene Decomposition via Foreground-Aware Slot Attention with Pseudo-Mask Guidance

**论文链接:** [http://arxiv.org/abs/2512.02685v1](http://arxiv.org/abs/2512.02685v1)

**作者:** Huankun Sheng, Ming Li, Yixiang Wei, Yeying Fan, Yu-Hui Wen, Tieliang Gong, Yong-Jin Liu

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种前景感知槽注意力(FASA)方法，通过两阶段框架明确分离前景和背景，解决了现有方法 indiscriminately 处理前景和背景导致的背景干扰问题，显著提高了物体发现性能。

### 背景

物体中心表征学习的最新进展表明，基于槽注意力的方法能够在无监督情况下将视觉场景分解为物体槽表征。然而，现有方法通常 indiscriminately 处理前景和背景区域，导致背景干扰和在真实数据上实例发现性能不佳。

### 目的

解决现有方法无法区分前景和背景的问题，提出前景感知槽注意力(FASA)框架，实现精确的物体发现。

### 方法

FASA采用两阶段框架：第一阶段通过双槽竞争机制进行粗略场景分解，基于聚类策略初始化槽；第二阶段引入掩蔽槽注意力机制，第一个槽捕获背景，其余槽竞争表示前景物体；同时加入自监督图像特征构建的补丁亲和图推导的伪掩蔽指导前景槽学习，解决前景物体过度分割问题。

### 主要发现

在合成和真实数据集上的广泛实验表明，FASA持续优于最先进方法，验证了明确前景建模和伪掩蔽指导对于鲁棒场景分解和物体一致表征的有效性。

### 结论

前景感知槽注意力(FASA)通过明确分离前景和背景，显著提高了物体发现性能，代码将公开可用。

### 翻译

物体中心表征学习的最新进展表明，基于槽注意力的方法能够在无监督情况下将视觉场景分解为物体槽表征。然而，现有方法通常 indiscriminately 处理前景和背景区域，往往导致背景干扰和在真实数据上实例发现性能不佳。为解决这一限制，我们提出前景感知槽注意力(FASA)，一个两阶段框架，明确分离前景和背景以实现精确的物体发现。在第一阶段，FASA通过双槽竞争机制进行粗略场景分解，以区分前景和背景区域。这些槽通过基于聚类的策略初始化，产生显著区域的良好结构化表示。在第二阶段，我们引入掩蔽槽注意力机制，其中第一个槽捕获背景，而其余槽竞争表示单个前景物体。为解决前景物体的过度分割问题，我们加入了从自监督图像特征构建的补丁亲和图推导的伪掩蔽指导，以引导前景槽的学习。在合成和真实数据集上的广泛实验表明，FASA持续优于最先进方法，验证了明确前景建模和伪掩蔽指导对于鲁棒场景分解和物体一致表征的有效性。代码将公开提供。


### 论文摘要

Recent advances in object-centric representation learning have shown that slot attention-based methods can effectively decompose visual scenes into object slot representations without supervision. However, existing approaches typically process foreground and background regions indiscriminately, often resulting in background interference and suboptimal instance discovery performance on real-world data. To address this limitation, we propose Foreground-Aware Slot Attention (FASA), a two-stage framework that explicitly separates foreground from background to enable precise object discovery. In the first stage, FASA performs a coarse scene decomposition to distinguish foreground from background regions through a dual-slot competition mechanism. These slots are initialized via a clustering-based strategy, yielding well-structured representations of salient regions. In the second stage, we introduce a masked slot attention mechanism where the first slot captures the background while the remaining slots compete to represent individual foreground objects. To further address over-segmentation of foreground objects, we incorporate pseudo-mask guidance derived from a patch affinity graph constructed with self-supervised image features to guide the learning of foreground slots. Extensive experiments on both synthetic and real-world datasets demonstrate that FASA consistently outperforms state-of-the-art methods, validating the effectiveness of explicit foreground modeling and pseudo-mask guidance for robust scene decomposition and object-coherent representation. Code will be made publicly available.

---

## 56. Adapting Tensor Kernel Machines to Enable Efficient Transfer Learning for Seizure Detection

**论文链接:** [http://arxiv.org/abs/2512.02626v1](http://arxiv.org/abs/2512.02626v1)

**作者:** Seline J. S. de Rooij, Borbála Hunyadi

**发布时间:** 2025-12-02

**备注:** This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

本研究提出了一种基于张量核机器的高效迁移学习方法（Adapt-TKM），通过低秩张量网络学习紧凑的非线性模型，实现知识迁移。应用于耳后脑电图癫痫检测时，该方法通过少量患者特定数据个性化模型，在性能优于传统模型的同时，参数量减少约100倍，推理速度显著提高，特别适合资源受限的可穿戴设备。

### 背景

迁移学习旨在通过学习相关源问题的知识来优化目标任务性能。在资源受限的可穿戴设备上实现高效迁移学习是一个挑战，特别是对于需要实时处理的应用如癫痫检测。

### 目的

开发一种高效的迁移学习方法，利用张量核机器实现知识迁移，同时保持模型紧凑高效，特别适合资源受限的可穿戴设备应用。

### 方法

提出了一种自适应张量核机器（Adapt-TKM）方法，受自适应SVM启发，通过正则化将源域知识迁移到适应模型中。该方法利用低秩张量网络在原始域中学习紧凑的非线性模型，无需增加更多参数即可实现更高效的适应。

### 主要发现

1. 将Adapt-TKM应用于耳后脑电图癫痫检测
2. 通过少量患者特定数据个性化患者独立模型
3. 患者适应模型相比患者独立模型和完全患者特定模型表现更好
4. Adapt-TKM需要的参数量比自适应SVM模型少约100倍
5. 推理速度相应提高
6. 该方法特别适用于资源受限的可穿戴设备

### 结论

Adapt-TKM是一种高效的迁移学习方法，能够在保持高性能的同时显著减少模型参数量，提高推理速度，特别适合资源受限的可穿戴设备应用，如癫痫检测等实时医疗监测系统。

### 翻译

迁移学习旨在通过学习相关源问题的知识来优化目标任务性能。在本研究中，我们提出了一种使用张量核机器的高效迁移学习方法。我们的方法受自适应SVM启发，通过正则化将源域知识迁移到'适应'模型中。使用张量核机器的主要优势在于它们利用低秩张量网络在原始域中学习紧凑的非线性模型。这使得能够更有效地适应模型，而无需向模型添加更多参数。为了证明我们方法的有效性，我们将自适应张量核机器（Adapt-TKM）应用于耳后脑电图癫痫检测。通过使用少量患者特定数据个性化患者独立模型，患者适应模型（使用Adapt-TKM）相比患者独立模型和完全患者特定模型实现了更好的性能。值得注意的是，它在需要比自适应SVM模型少约100倍参数的同时实现了这一点，从而带来了相应更快的推理速度。这使得Adapt-TKM特别适用于资源受限的可穿戴设备。


### 论文摘要

Transfer learning aims to optimize performance in a target task by learning from a related source problem. In this work, we propose an efficient transfer learning method using a tensor kernel machine. Our method takes inspiration from the adaptive SVM and hence transfers 'knowledge' from the source to the 'adapted' model via regularization. The main advantage of using tensor kernel machines is that they leverage low-rank tensor networks to learn a compact non-linear model in the primal domain. This allows for a more efficient adaptation without adding more parameters to the model. To demonstrate the effectiveness of our approach, we apply the adaptive tensor kernel machine (Adapt-TKM) to seizure detection on behind-the-ear EEG. By personalizing patient-independent models with a small amount of patient-specific data, the patient-adapted model (which utilizes the Adapt-TKM), achieves better performance compared to the patient-independent and fully patient-specific models. Notably, it is able to do so while requiring around 100 times fewer parameters than the adaptive SVM model, leading to a correspondingly faster inference speed. This makes the Adapt-TKM especially useful for resource-constrained wearable devices.

---

## 57. Modeling and Inverse Identification of Interfacial Heat Conduction in Finite Layer and Semi-Infinite Substrate Systems via a Physics-Guided Neural Framework

**论文链接:** [http://arxiv.org/abs/2512.02618v1](http://arxiv.org/abs/2512.02618v1)

**作者:** Wenhao Sha, Tienchong Chang

**发布时间:** 2025-12-02

### GPT解析

### 总结

该研究提出了一种名为HeatTransFormer的物理引导Transformer架构，用于解决半导体设备中界面主导的热传递问题，能够准确处理陡峭的温度梯度并保持物理一致性。

### 背景

半导体设备中的热传递主要由芯片和基板组件主导，热量在有限芯片层内产生并消散到具有更高热物理性质的半无限基板中，这种不匹配产生陡峭的界面温度梯度，使瞬态热响应对界面高度敏感。

### 目的

解决传统数值求解器需要过度离散化以及物理信息神经网络(PINNs)在材料界面附近表现不稳定的问题，为界面主导的扩散问题开发新的解决方案。

### 方法

HeatTransFormer是一种物理引导的Transformer架构，集成了物理信息时空采样、基于拉普拉斯的激活函数（模拟扩散分析解）和无掩码注意力机制（支持双向时空耦合）。

### 主要发现

HeatTransFormer能够解决陡峭梯度，保持物理一致性，并在PINNs通常失败的地方保持稳定；应用于有限层和半无限基板配置时能产生跨界面的连贯温度场；结合物理约束逆策略，仅使用外部测量就能可靠地同时识别三个未知的热特性。

### 结论

物理引导的Transformer架构为界面主导的热系统中的正向和逆向建模提供了统一框架。

### 翻译

半导体设备中的热传递主要由芯片和基板组件主导，其中在有限芯片层内产生的热量会消散到热物理性质高得多的半无限基板中。这种不匹配产生了陡峭的界面温度梯度，使瞬态热响应对界面高度敏感。传统的数值求解器需要过度离散化来解决这些动态问题，而物理信息神经网络(PINNs)在材料界面附近通常表现出不稳定的收敛性和物理一致性的丧失。为应对这些挑战，我们引入了HeatTransFormer，一种用于界面主导扩散问题的物理引导Transformer架构。该框架集成了物理信息时空采样、模拟扩散分析解的基于拉普拉斯的激活函数，以及支持双向时空耦合的无掩码注意力机制。这些组件使模型能够解决陡峭梯度，保持物理一致性，并在PINNs通常失败的地方保持稳定。HeatTransFormer应用于有限层和半无限基板配置时，能够产生跨界面的连贯温度场。结合物理约束逆策略，它仅使用外部测量就能可靠地同时识别三个未知的热特性。总之，这项工作表明，物理引导的Transformer架构为界面主导热系统中的正向和逆向建模提供了统一框架。


### 论文摘要

Heat transfer in semiconductor devices is dominated by chip and substrate assemblies, where heat generated within a finite chip layer dissipates into a semi-infinite substrate with much higher thermophysical properties. This mismatch produces steep interfacial temperature gradients, making the transient thermal response highly sensitive to the interface. Conventional numerical solvers require excessive discretization to resolve these dynamics, while physics-informed neural networks (PINNs) often exhibit unstable convergence and loss of physical consistency near the material interface. To address these challenges, we introduce HeatTransFormer, a physics-guided Transformer architecture for interface-dominated diffusion problems. The framework integrates physically informed spatiotemporal sampling, a Laplace-based activation emulating analytical diffusion solutions, and a mask-free attention mechanism supporting bidirectional spatiotemporal coupling. These components enable the model to resolve steep gradients, maintain physical consistency, and remain stable where PINNs typically fail. HeatTransFormer produces coherent temperature fields across the interface when applied to a finite layer and semi-infinite substrate configuration. Coupled with a physics-constrained inverse strategy, it further enables reliable identification of three unknown thermal properties simultaneously using only external measurements. Overall, this work demonstrates that physics-guided Transformer architectures provide a unified framework for forward and inverse modeling in interface-dominated thermal systems.

---

## 58. UniCom: Towards a Unified and Cohesiveness-aware Framework for Community Search and Detection

**论文链接:** [http://arxiv.org/abs/2512.02460v1](http://arxiv.org/abs/2512.02460v1)

**作者:** Yifan Zhu, Hanchen Wang, Wenjie Zhang, Alexander Zhou, Ying Zhang

**发布时间:** 2025-12-02

**备注:** 14 pages (12 for content, 2 for reference)

### GPT解析

### 总结

本文提出了UniCom框架，一个统一的方法来解决社区搜索和社区检测任务，通过跨领域知识转移缓解了现有方法的局限性。

### 背景

在现实世界图中搜索和检测社区是许多应用的基础。现有学习方法将社区搜索和社区检测视为独立问题，需要针对特定任务和数据集重新训练，限制了模型的适用性和泛化能力。

### 目的

提出一个统一框架同时解决社区搜索和社区检测任务，通过跨领域知识转移缓解单数据集学习的局限性，消除昂贵的重新训练需求。

### 方法

UniCom框架包含领域感知专门化(DAS)程序和通用图学习(UGL)主干。DAS能够即时适应未见过的图或任务，通过轻量级基于提示的范式保持框架紧凑。UGL通过全面预训练从多个源领域提炼可转移的语义和拓扑知识。两者都由局部邻域信号和凝聚子图结构提供一致指导。

### 主要发现

在16个基准数据集和22个基线上的实验表明，UniCom在监督稀缺或无监督设置下，在所有任务中始终优于所有最先进的基线，同时保持运行时效率。

### 结论

UniCom通过统一框架和跨领域知识转移，成功解决了社区搜索和社区检测作为独立处理的问题，提高了模型在有限监督下的性能和适用性。

### 翻译

在现实世界图中搜索和检测社区支撑着广泛的应用范围。尽管取得了成功，但当前基于学习的解决方案将社区搜索（即定位给定查询的最佳社区）和社区检测（即划分整个图）视为独立问题，需要针对特定任务和数据集重新训练。这种策略限制了现有模型的适用性和泛化能力。此外，这些方法严重依赖目标数据集的信息，导致在监督有限或不可用时性能次优化。为缓解这一局限性，我们提出了UniCom，一个统一框架，通过跨领域知识转移同时解决社区搜索和检测任务，从而缓解单数据集学习的限制。UniCom以领域感知专门化(DAS)程序为核心，能够即时适应未见过的图或任务，消除昂贵的重新训练，同时通过轻量级的基于提示的范式保持框架紧凑性。这由通用图学习(UGL)主干支持，通过全面预训练从多个源领域提炼可转移的语义和拓扑知识。DAS和UGL都由局部邻域信号和凝聚子图结构提供信息，为整个框架提供一致的指导。已在16个基准数据集上的两项任务和22个基线进行了大量实验，确保全面和公平的评估。在监督稀缺或无设置下，UniCom在所有任务中始终优于所有最先进的基线，同时保持运行时效率。


### 论文摘要

Searching and detecting communities in real-world graphs underpins a wide range of applications. Despite the success achieved, current learning-based solutions regard community search, i.e., locating the best community for a given query, and community detection, i.e., partitioning the whole graph, as separate problems, necessitating task- and dataset-specific retraining. Such a strategy limits the applicability and generalization ability of the existing models. Additionally, these methods rely heavily on information from the target dataset, leading to suboptimal performance when supervision is limited or unavailable. To mitigate this limitation, we propose UniCom, a unified framework to solve both community search and detection tasks through knowledge transfer across multiple domains, thus alleviating the limitations of single-dataset learning. UniCom centers on a Domain-aware Specialization (DAS) procedure that adapts on the fly to unseen graphs or tasks, eliminating costly retraining while maintaining framework compactness with a lightweight prompt-based paradigm. This is empowered by a Universal Graph Learning (UGL) backbone, which distills transferable semantic and topological knowledge from multiple source domains via comprehensive pre-training. Both DAS and UGL are informed by local neighborhood signals and cohesive subgraph structures, providing consistent guidance throughout the framework. Extensive experiments on both tasks across 16 benchmark datasets and 22 baselines have been conducted to ensure a comprehensive and fair evaluation. UniCom consistently outperforms all state-of-the-art baselines across all tasks under settings with scarce or no supervision, while maintaining runtime efficiency.

---

## 59. SimFlow: Simplified and End-to-End Training of Latent Normalizing Flows

**论文链接:** [http://arxiv.org/abs/2512.04084v1](http://arxiv.org/abs/2512.04084v1)

**作者:** Qinyu Zhao, Guangting Zheng, Tao Yang, Rui Zhu, Xingjian Leng, Stephen Gould, Liang Zheng

**发布时间:** 2025-12-03

**备注:** Project Page: https://qinyu-allen-zhao.github.io/SimFlow/

### GPT解析

### 总结

本文提出了一种简单方法，通过固定VAE编码器的方差为常数，解决了标准化流研究中存在的两个主要问题，显著提高了模型在ImageNet生成任务上的性能。

### 背景

标准化流学习数据与高斯分布之间的可逆映射。先前的研究通常受两个限制：一是通过添加随机噪声作为数据增强引入复杂流程；二是使用预训练且冻结的VAE编码器导致次优的重建和生成质量。

### 目的

解决先前研究中存在的两个问题，避免复杂的噪声处理流程，并提高重建和生成质量。

### 方法

将VAE编码器预测的方差固定为一个常数（例如0.5）。这种方法允许编码器输出更广泛的令牌分布，解码器学习从增强的令牌分布重建干净图像，同时简化了VAE证据下界，使与VAE联合训练NF变得稳定。

### 主要发现

在ImageNet 256×256生成任务上，提出的SimFlow模型获得了2.15的gFID分数，优于STARFlow（gFID 2.40）。SimFlow与REPA-E方法集成后，gFID分数提升至1.91，树立了新的最先进水平。

### 结论

通过简单固定VAE编码器的方差为常数，可以解决先前研究中存在的两个主要问题，提高模型性能，并在ImageNet生成任务上取得最先进的结果。

### 翻译

标准化流学习数据与高斯分布之间的可逆映射。先前的研究通常受两个限制。首先，它们向训练样本或VAE潜在变量添加随机噪声作为数据增强，引入了包含额外加噪和去噪步骤的复杂流程。其次，它们使用预训练和冻结的VAE编码器，导致次优的重建和生成质量。在本文中，我们发现这两个问题可以通过一个非常简单的方式解决：只需将VAE编码器预测的方差固定为一个常数（例如0.5）。一方面，这种方法允许编码器输出更广泛的令牌分布，解码器学习从增强的令牌分布重建干净图像，避免额外的噪声或去噪设计。另一方面，固定方差简化了VAE证据下界，使得与VAE联合训练NF变得稳定。在ImageNet 256×256生成任务上，我们的模型SimFlow获得了2.15的gFID分数，优于最先进的方法STARFlow（gFID 2.40）。此外，SimFlow可以与端到端表示对齐方法无缝集成，并实现改进的gFID分数1.91，在NF中树立了新的最先进水平。


### 论文摘要

Normalizing Flows (NFs) learn invertible mappings between the data and a Gaussian distribution. Prior works usually suffer from two limitations. First, they add random noise to training samples or VAE latents as data augmentation, introducing complex pipelines including extra noising and denoising steps. Second, they use a pretrained and frozen VAE encoder, resulting in suboptimal reconstruction and generation quality. In this paper, we find that the two issues can be solved in a very simple way: just fixing the variance (which would otherwise be predicted by the VAE encoder) to a constant (e.g., 0.5). On the one hand, this method allows the encoder to output a broader distribution of tokens and the decoder to learn to reconstruct clean images from the augmented token distribution, avoiding additional noise or denoising design. On the other hand, fixed variance simplifies the VAE evidence lower bound, making it stable to train an NF with a VAE jointly. On the ImageNet $256 \times 256$ generation task, our model SimFlow obtains a gFID score of 2.15, outperforming the state-of-the-art method STARFlow (gFID 2.40). Moreover, SimFlow can be seamlessly integrated with the end-to-end representation alignment (REPA-E) method and achieves an improved gFID of 1.91, setting a new state of the art among NFs.

---

## 60. MarkTune: Improving the Quality-Detectability Trade-off in Open-Weight LLM Watermarking

**论文链接:** [http://arxiv.org/abs/2512.04044v1](http://arxiv.org/abs/2512.04044v1)

**作者:** Yizhou Zhao, Zhiwei Steven Wu, Adam Block

**发布时间:** 2025-12-03

### GPT解析

### 总结

MarkTune是一种改进的水印技术，解决了开放权重语言模型中水印检测与文本质量之间的权衡问题，通过基于策略的微调框架实现了更高质量的检测结果，同时保持文本生成质量。

### 背景

水印技术旨在为生成的文本嵌入隐藏信号，这些信号在拥有密钥的情况下可以被可靠检测。开放权重语言模型给水印方案带来特殊挑战，因为当代方法主要依赖推理时干预，而一旦模型权重公开就无法强制执行。现有的开放权重模型水印技术（如GaussMark）通常依赖对模型权重的微小修改，这些修改可以被拥有密钥的人检测到，但达到与推理时水印相当的检测能力通常需要明显降低生成质量的权重扰动。

### 目的

开发一种理论上有原则的、基于策略的微调框架，能够改善水印的检测能力与文本质量之间的权衡。

### 方法

提出了MarkTune框架，将GaussMark信号视为奖励，同时正则化以防止文本质量下降。MarkTune作为GaussMark的改进，通过在模型表示空间内引导更细粒度的、水印感知的权重更新，同时保持生成质量，来改善质量-检测能力权衡。

### 主要发现

MarkTune将GaussMark的质量-检测能力边界推向接近推理时水印的水平。MarkTune对释义和微调攻击具有鲁棒性。MarkTune表现出强大的泛化能力：在一个数据集上微调的模型在未见过的数据集上仍保留实质性的水印检测能力。

### 结论

这些结果确立了MarkTune作为向开放权重语言模型嵌入稳健、高质量水印的一般策略。

### 翻译

水印旨在在生成的文本中嵌入隐藏信号，当能够访问密钥时可以可靠检测。开放权重语言模型对此类水印方案构成了严峻挑战，因为主导当代方法的推理时干预一旦模型权重公开就无法强制执行。现有的开放权重模型水印技术，如最近提出的GaussMark，通常依赖于对模型权重的微小修改，这些修改可以为拥有密钥的人提供可检测的信号，但要达到与推理时水印相当的检测能力，通常需要明显降低生成质量的权重扰动。我们引入了MarkTune，这是一种理论上合理的、基于策略的微调框架，它将GaussMark信号视为奖励，同时正则化以防止文本质量下降。我们将MarkTune推导为GaussMark的改进，并证明MarkTune通过在模型表示空间内引导更细粒度的、水印感知的权重更新，同时保持生成质量，从而一致地改善了GaussMark的质量-检测能力权衡。经验上，我们表明MarkTune将GaussMark的质量-检测能力边界推向接近推理时水印的水平，对释义和微调攻击保持鲁棒性，并表现出强大的泛化能力：在一个数据集上微调的模型在未见过的数据集上保留了实质性的水印检测能力。总之，这些结果确立了MarkTune作为向开放权重语言模型嵌入稳健、高质量水印的一般策略。


### 论文摘要

Watermarking aims to embed hidden signals in generated text that can be reliably detected when given access to a secret key. Open-weight language models pose acute challenges for such watermarking schemes because the inference-time interventions that dominate contemporary approaches cannot be enforced once model weights are public. Existing watermaking techniques for open-weight models, such as the recently proposed GaussMark, typically rely on small modifications to model weights, which can yield signals detectable to those equipped with a secret key, but achieving detection power comparable to inference-time watermarks generally requires weight perturbations that noticeably reduce generation quality. We introduce MarkTune, a theoretically principled, on-policy fine-tuning framework that treats the GaussMark signal as a reward while simultaneously regularizing against degradation in text quality. We derive MarkTune as an improvement on GaussMark and demonstrate that MarkTune consistently improves the quality-detectability trade-off over GaussMark by steering finer-grained, watermark-aware weight updates within the model's representation space while preserving generation quality. Empirically, we show that MarkTune pushes the quality-detectability frontier of GaussMark close to that of inference-time watermarking, remains robust to paraphrasing and fine-tuning attacks, and exhibits strong generalization: a model fine-tuned on one dataset retains substantial watermark detection power on unseen datasets. Together, these results establish MarkTune as a general strategy for embedding robust, high-quality watermarks into open-weight LMs.

---

## 61. C3G: Learning Compact 3D Representations with 2K Gaussians

**论文链接:** [http://arxiv.org/abs/2512.04021v1](http://arxiv.org/abs/2512.04021v1)

**作者:** Honggyu An, Jaewoo Jung, Mungyeom Kim, Sunghwan Hong, Chaehyun Kim, Kazumi Fukuda, Minkyeong Jeon, Jisang Han, Takuya Narihira, Hyuna Ko, Junsu Kim, Yuki Mitsufuji, Seungryong Kim

**发布时间:** 2025-12-03

**备注:** Project Page : https://cvlab-kaist.github.io/C3G/

### GPT解析

### 总结

该研究提出了一种名为C3G的新型前馈框架，用于从未标定的稀疏视图中重建和理解3D场景。该方法通过在关键空间位置估计紧凑的3D高斯分布来减少冗余，并利用可学习的标记通过自注意力机制聚合多视图特征，从而实现高效的场景重建与理解。

### 背景

从未标定的稀疏视图中以前馈方式重建和理解3D场景是3D计算机视觉中的挑战性任务。现有方法使用逐像素3D高斯散射进行重建，然后通过2D到3D的特征提升阶段进行场景理解。然而，这些方法会产生过多冗余的高斯分布，导致高内存开销和次优的多视图特征聚合，进而影响新视图合成和场景理解性能。

### 目的

解决现有方法中产生过多冗余高斯分布的问题，降低内存需求并提高多视图特征聚合的效率，最终实现更高质量的新视图合成和场景理解。

### 方法

研究者提出了C3G框架，主要包括三个关键部分：1）仅在必要的空间位置估计紧凑的3D高斯分布，最小化冗余；2）引入可学习的标记，通过自注意力机制聚合多视图特征，以指导高斯生成；3）利用学习到的注意力模式进行高斯解码，以高效地提升特征。

### 主要发现

1）在无姿态新视图合成、3D开放词汇分割和视图不变特征聚合方面的广泛实验证明了该方法的有效性；2）紧凑且具有几何意义的表示足以实现高质量的场景重建和理解；3）与现有方法相比，该方法在内存效率和特征保真度方面表现更优。

### 结论

C3G框架通过生成紧凑的3D高斯表示并有效地聚合多视图特征，显著提高了3D场景重建和理解的效率和质量，证明了在关键位置使用紧凑表示的优越性。

### 翻译

从未标定的稀疏视图中以前馈方式重建和理解3D场景仍然是3D计算机视觉中的一个挑战性任务。最近的方法使用逐像素3D高斯散射进行重建，然后通过2D到3D的特征提升阶段进行场景理解。然而，它们会产生过多冗余的高斯分布，导致高内存开销和次优的多视图特征聚合，从而降低新视图合成和场景理解性能。我们提出了C3G，一种新型前馈框架，仅在关键空间位置估计紧凑的3D高斯分布，最小化冗余同时实现有效的特征提升。我们引入了可学习的标记，通过自注意力机制聚合多视图特征，以指导高斯生成，确保每个高斯都能整合跨视图的相关视觉特征。然后，我们利用学习到的注意力模式进行高斯解码，以高效地提升特征。在无姿态新视图合成、3D开放词汇分割和视图不变特征聚合方面的广泛实验证明了我们方法的有效性。结果表明，紧凑且具有几何意义的表示足以实现高质量的场景重建和理解，与现有方法相比，在内存效率和特征保真度方面表现更优。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从稀疏、无姿态的多视角图像中高效重建和理解3D场景的挑战。现有方法使用密集的每像素高斯表示，导致内存开销大、计算效率低，限制了在实际应用中的部署。这个问题在机器人、场景理解、虚拟现实等领域至关重要，因为高效准确的3D表示可以加速这些技术的发展并降低硬件要求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从人类视觉认知获得灵感——人类通过形成紧凑、语义上有意义的抽象来理解环境，而非像素级重建。他们设计了基于查询的高斯解码方法，使用可学习的查询令牌替代每像素预测。借鉴了3D高斯飞溅作为表示基础，transformer架构用于处理多视图特征，VGGT作为视觉编码器提供几何先验，以及渐进式低通滤波策略来稳定训练。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用一组紧凑的可学习查询令牌来发现和解码只在关键空间位置的3D高斯元，而非为每个像素预测。整体流程：1)使用预训练视觉编码器提取多视图特征；2)通过transformer架构处理查询令牌和多视图特征；3)查询令牌通过自注意力机制聚合跨视图信息；4)将精炼的查询令牌解码为3D高斯元；5)通过新视角合成目标函数训练；6)利用学习到的注意力模式将任意2D特征提升为视图不变的3D特征。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)仅使用2K个高斯元(比现有方法少约65倍)实现紧凑表示；2)基于查询的高斯解码避免冗余和对齐问题；3)自注意力机制自然发现跨视图对应关系；4)任意特征3D提升方法(C3G-F)；5)重用注意力图的视点不变特征解码器。相比之前工作：从根本上解决了输入视图偏差问题，避免了自动编码器的信息损失，减少了内存使用，提高了计算效率，同时在新视角合成和场景理解任务上取得更好性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'C3G提出了一种基于查询的紧凑3D高斯表示方法，仅使用2K个高斯元实现高效的3D场景重建和理解，显著降低内存需求并提高计算效率，同时在新视角合成和场景理解任务上取得优于现有方法的性能。'}


### 论文摘要

Reconstructing and understanding 3D scenes from unposed sparse views in a feed-forward manner remains as a challenging task in 3D computer vision. Recent approaches use per-pixel 3D Gaussian Splatting for reconstruction, followed by a 2D-to-3D feature lifting stage for scene understanding. However, they generate excessive redundant Gaussians, causing high memory overhead and sub-optimal multi-view feature aggregation, leading to degraded novel view synthesis and scene understanding performance. We propose C3G, a novel feed-forward framework that estimates compact 3D Gaussians only at essential spatial locations, minimizing redundancy while enabling effective feature lifting. We introduce learnable tokens that aggregate multi-view features through self-attention to guide Gaussian generation, ensuring each Gaussian integrates relevant visual features across views. We then exploit the learned attention patterns for Gaussian decoding to efficiently lift features. Extensive experiments on pose-free novel view synthesis, 3D open-vocabulary segmentation, and view-invariant feature aggregation demonstrate our approach's effectiveness. Results show that a compact yet geometrically meaningful representation is sufficient for high-quality scene reconstruction and understanding, achieving superior memory efficiency and feature fidelity compared to existing methods.

---

## 62. Learning Group Actions In Disentangled Latent Image Representations

**论文链接:** [http://arxiv.org/abs/2512.04015v1](http://arxiv.org/abs/2512.04015v1)

**作者:** Farhana Hossain Swarnali, Miaomiao Zhang, Tonmoy Hossain

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究提出了一种新颖的端到端框架，首次在潜在图像流形上学习群作用，无需人工干预即可自动发现与变换相关的结构，有效解决了现有方法在分离变换子空间方面的局限性。

### 背景

现有方法在高维数据空间操作时群作用均匀应用于整个输入，难以分离变换子空间；而潜在空间方法虽灵活，但需手动划分潜在变量为等变和不变子空间，限制了群作用的稳健学习和操作。

### 目的

引入一个端到端框架，首次在潜在图像流形上学习群作用，自动发现变换相关结构，无需人工干预，并能与任何标准编码器-解码器架构无缝集成。

### 方法

使用可学习的二进制掩码和直通估计动态将潜在表示划分为变换敏感和不变组件，在统一优化框架内联合学习潜在解缠和群变换映射。

### 主要发现

在五个2D/3D图像数据集上验证了该方法，证明其能自动学习群作用的解缠潜在因子，下游分类任务证实了学习表示的有效性。

### 结论

该框架成功解决了现有方法的局限性，能够自动发现变换相关结构，有效应用于图像表示学习，代码已公开。

### 翻译

在潜在表示上对群作用进行建模能够实现对高维图像数据的可控变换。先前应用群论先验或建模变换的方法通常在高维数据空间中操作，群作用在整个输入上均匀应用，难以分离在变换下变化的子空间。虽然潜在空间方法提供了更大的灵活性，但仍需手动将潜在变量划分为等变和不变子空间，限制了在表示空间内稳健学习和操作群作用的能力。为解决这一问题，我们引入了一种新颖的端到端框架，首次在潜在图像流形上学习群作用，无需人工干预即可自动发现与变换相关的结构。我们的方法使用可学习的二进制掩码和直通估计来动态将潜在表示划分为变换敏感和不变组件。我们在统一的优化框架内制定该方法，联合学习潜在解缠和群变换映射。该框架可以与任何标准编码器-解码器架构无缝集成。我们在五个2D/3D图像数据集上验证了我们的方法，证明了它能够自动学习群作用的解缠潜在因子在各种数据中，同时下游分类任务证实了学习表示的有效性。我们的代码已在https://github.com/farhanaswarnali/Learning-Group-Actions-In-Disentangled-Latent-Image-Representations公开。


### 论文摘要

Modeling group actions on latent representations enables controllable transformations of high-dimensional image data. Prior works applying group-theoretic priors or modeling transformations typically operate in the high-dimensional data space, where group actions apply uniformly across the entire input, making it difficult to disentangle the subspace that varies under transformations. While latent-space methods offer greater flexibility, they still require manual partitioning of latent variables into equivariant and invariant subspaces, limiting the ability to robustly learn and operate group actions within the representation space. To address this, we introduce a novel end-to-end framework that for the first time learns group actions on latent image manifolds, automatically discovering transformation-relevant structures without manual intervention. Our method uses learnable binary masks with straight-through estimation to dynamically partition latent representations into transformation-sensitive and invariant components. We formulate this within a unified optimization framework that jointly learns latent disentanglement and group transformation mappings. The framework can be seamlessly integrated with any standard encoder-decoder architecture. We validate our approach on five 2D/3D image datasets, demonstrating its ability to automatically learn disentangled latent factors for group actions in diverse data, while downstream classification tasks confirm the effectiveness of the learned representations. Our code is publicly available at https://github.com/farhanaswarnali/Learning-Group-Actions-In-Disentangled-Latent-Image-Representations .

---

## 63. 论文ID: 2512.04007v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04007v1.json'

---

## 64. Training and Evaluation of Guideline-Based Medical Reasoning in LLMs

**论文链接:** [http://arxiv.org/abs/2512.03838v1](http://arxiv.org/abs/2512.03838v1)

**作者:** Michael Staniek, Artem Sokolov, Stefan Riezler

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究旨在教会大语言模型遵循医学共识指南进行推理和预测，通过微调模型学习共识规则及其例外，并自动评估推理过程的正确性。研究发现小型微调模型优于大型模型，早期预测的瓶颈是预测未来临床变量而非分布外泛化问题。

### 背景

医学领域的机器学习在早期预测方面取得突破性进展，但过于关注提高预测准确率，忽视了获得医疗从业者信任所需的忠实解释。

### 目的

教会大语言模型在推理和预测过程中逐步遵循医学共识指南，提高预测的可解释性和可信度。

### 方法

利用共识指南将语言化的医疗推理规则实例化为电子健康记录数据微调LLMs；自动评估模型推理的推导正确性和数值正确性；使用Sepsis-3共识定义作为示例；通过多模态集成时间序列预测模型和LLM改进对未来临床变量的预测。

### 主要发现

小型微调模型优于大型模型和医学文本训练模型；在特定领域微调可实现规则和例外的近乎完美推导正确性；早期预测的瓶颈是预测未来临床变量而非分布外泛化问题。

### 结论

通过多模态集成时间序列预测模型和LLM，可以改进对未来临床变量的预测，提高医学早期预测的准确性。

### 翻译

医学领域的机器学习在早期预测方面最近显示出突破性性能，然而，专注于提高预测准确率导致忽视了获得医疗从业者信任所需的忠实解释。本文的目标是教会大语言模型在其推理和预测过程中逐步遵循医学共识指南。由于共识指南在医学中无处不在，将语言化的医疗推理规则实例化为电子健康记录数据，可以为微调LLMs学习许多医学领域的共识规则及其可能的例外提供数据。共识规则还能够自动评估模型的推理过程，包括其推导正确性（评估从给定前提得出结论的正确和忠实演绎）和数值正确性（将预测值与真实世界测量值进行比较）。我们使用复杂的Sepsis-3共识定义来说明我们的工作。我们的实验表明，小型微调模型优于通过显式定义提示的较大LLMs的一次性学习模型，也优于在包含共识定义的医学文本上训练的模型。由于在特定医疗领域对语言化规则实例进行微调，可以在该领域未见过的患者数据上实现规则（及其例外）的近乎完美的推导正确性，因此早期预测的瓶颈不是分布外泛化，而是泛化到未来的正交问题，即预测稀疏和不规则采样的临床变量。我们表明，通过在多模态设置中集成时间序列预测模型的输出表示与LLM，可以改进后者的结果。


### 论文摘要

Machine learning for early prediction in medicine has recently shown breakthrough performance, however, the focus on improving prediction accuracy has led to a neglect of faithful explanations that are required to gain the trust of medical practitioners. The goal of this paper is to teach LLMs to follow medical consensus guidelines step-by-step in their reasoning and prediction process. Since consensus guidelines are ubiquitous in medicine, instantiations of verbalized medical inference rules to electronic health records provide data for fine-tuning LLMs to learn consensus rules and possible exceptions thereof for many medical areas. Consensus rules also enable an automatic evaluation of the model's inference process regarding its derivation correctness (evaluating correct and faithful deduction of a conclusion from given premises) and value correctness (comparing predicted values against real-world measurements). We exemplify our work using the complex Sepsis-3 consensus definition. Our experiments show that small fine-tuned models outperform one-shot learning of considerably larger LLMs that are prompted with the explicit definition and models that are trained on medical texts including consensus definitions. Since fine-tuning on verbalized rule instantiations of a specific medical area yields nearly perfect derivation correctness for rules (and exceptions) on unseen patient data in that area, the bottleneck for early prediction is not out-of-distribution generalization, but the orthogonal problem of generalization into the future by forecasting sparsely and irregularly sampled clinical variables. We show that the latter results can be improved by integrating the output representations of a time series forecasting model with the LLM in a multimodal setup.

---

## 65. MPCFormer: A physics-informed data-driven approach for explainable socially-aware autonomous driving

**论文链接:** [http://arxiv.org/abs/2512.03795v1](http://arxiv.org/abs/2512.03795v1)

**作者:** Jia Hu, Zhexi Lian, Xuerun Yan, Ruiang Bi, Dou Shen, Yu Ruan, Haoran Wang

**发布时间:** 2025-12-03

**备注:** 17 pages, 18 figures

### GPT解析

### 总结

MPCFormer是一种结合物理信息和数据驱动的可解释自动驾驶方法，通过建模多车辆社交互动动力学，实现了更安全、高效且类人的驾驶行为。

### 背景

自动驾驶车辆在动态和交互式交通场景中仍难以表现出类似人类的行为，主要挑战在于与周围车辆的交互能力有限，这主要是因为缺乏对社会互动机制的理解。

### 目的

提出MPCFormer，一种可解释的、具有社会意识的自动驾驶方法，结合物理信息和数据驱动的耦合社交互动动力学，以解决自动驾驶车辆的社交互动问题。

### 方法

将动力学公式化为离散空间状态表示，嵌入物理先验以提高模型可解释性；通过基于Transformer的编码器-解码器架构从自然驾驶数据中学习动力学系数；明确建模多车辆社交互动动力学。

### 主要发现

学习到的社交互动动力学使规划器能够产生多样化的类人行为；在NGSIM数据集上实现了最低的轨迹预测误差；在密集交互场景中实现了94.67%的最高规划成功率，提高驾驶效率15.75%，将碰撞率从21.25%降至0.5%。

### 结论

MPCFormer通过结合物理信息和数据驱动的方法，成功解决了自动驾驶车辆在社交互动方面的挑战，实现了更安全、高效且接近人类行为的自动驾驶。

### 翻译

自动驾驶车辆在高度动态和交互式的交通场景中仍难以表现出类似人类的行为。关键挑战在于自动驾驶与周围车辆交互的能力有限，这主要是由于缺乏对社会互动基本机制的理解。为解决这一问题，我们引入了MPCFormer，一种可解释的、具有社会意识的自动驾驶方法，结合了物理信息和数据驱动的耦合社交互动动力学。在该模型中，动力学被公式化为离散空间状态表示，嵌入物理先验以增强建模的可解释性。动力学系数通过基于Transformer的编码器-解码器架构从自然驾驶数据中学习。据我们所知，MPCFormer是第一个明确建模多车辆社交互动动力学的方法。学习到的社交互动动力学使规划器在与周围交通互动时能够产生多样化的、类人的行为。通过利用MPC框架，该方法减轻了与纯学习方法相关的潜在安全风险。在NGSIM数据集上的开环评估表明，MPCFormer实现了卓越的社交互动意识，与其他最先进的方法相比，产生了最低的轨迹预测误差。在5秒的长预测范围内，预测的ADE低至0.86米。在高强度交互场景下的闭环实验（需要连续变道以驶出匝道）进一步验证了MPCFormer的有效性。结果显示，MPCFormer实现了94.67%的最高规划成功率，提高了15.75%的驾驶效率，并将碰撞率从21.25%降至0.5%，优于前沿的基于强化学习(RL)的规划器。


### 论文摘要

Autonomous Driving (AD) vehicles still struggle to exhibit human-like behavior in highly dynamic and interactive traffic scenarios. The key challenge lies in AD's limited ability to interact with surrounding vehicles, largely due to a lack of understanding the underlying mechanisms of social interaction. To address this issue, we introduce MPCFormer, an explainable socially-aware autonomous driving approach with physics-informed and data-driven coupled social interaction dynamics. In this model, the dynamics are formulated into a discrete space-state representation, which embeds physics priors to enhance modeling explainability. The dynamics coefficients are learned from naturalistic driving data via a Transformer-based encoder-decoder architecture. To the best of our knowledge, MPCFormer is the first approach to explicitly model the dynamics of multi-vehicle social interactions. The learned social interaction dynamics enable the planner to generate manifold, human-like behaviors when interacting with surrounding traffic. By leveraging the MPC framework, the approach mitigates the potential safety risks typically associated with purely learning-based methods. Open-looped evaluation on NGSIM dataset demonstrates that MPCFormer achieves superior social interaction awareness, yielding the lowest trajectory prediction errors compared with other state-of-the-art approach. The prediction achieves an ADE as low as 0.86 m over a long prediction horizon of 5 seconds. Close-looped experiments in highly intense interaction scenarios, where consecutive lane changes are required to exit an off-ramp, further validate the effectiveness of MPCFormer. Results show that MPCFormer achieves the highest planning success rate of 94.67%, improves driving efficiency by 15.75%, and reduces the collision rate from 21.25% to 0.5%, outperforming a frontier Reinforcement Learning (RL) based planner.

---

## 66. In-Context Representation Hijacking

**论文链接:** [http://arxiv.org/abs/2512.03771v1](http://arxiv.org/abs/2512.03771v1)

**作者:** Itay Yona, Amir Sarid, Michael Karasik, Yossi Gandelsman

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文介绍了一种名为'Doublespeak'的简单上下文表示劫持攻击方法，通过将有害关键词替换为良性标记，使模型内部表示发生转变，从而绕过安全对齐机制。

### 背景

大型语言模型(LLMs)存在安全对齐问题，需要探索新的攻击方式来评估其安全性。

### 目的

提出一种针对LLMs的新型攻击方法，探索其潜在空间中的新攻击面，揭示当前对齐策略的不足。

### 方法

通过系统性地将有害关键词(如'bomb')替换为良性标记(如'carrot')，并在有害请求前提供多个上下文示例，导致良性标记的内部表示向有害表示收敛，在委婉语下嵌入有害语义。

### 主要发现

表面无害的提示被内部解释为被禁止的指令；语义覆盖逐层出现，从早期层的良性含义在后期层收敛为有害语义；该攻击无需优化，可跨模型家族广泛转移；在Llama-3.3-70B-Instruct上单句上下文覆盖达到74%的攻击成功率。

### 结论

LLMs的潜在空间中存在新的攻击面，当前对齐策略不足，应该在表示层面进行操作。

### 翻译

我们引入了'Doublespeak'，这是一种针对大型语言模型(LLMs)的简单上下文表示劫持攻击。该攻击通过在多个上下文示例中有系统地用良性标记(如'胡萝卜')替换有害关键词(如'炸弹')，并提供有害请求的前缀，来实现攻击。这种替换导致良性标记的内部表示向有害表示收敛，有效地在委婉语下嵌入有害语义。结果，表面上无害的提示(如'如何建造一个胡萝卜?')被内部解释为被禁止的指令(如'如何建造一个炸弹?')，从而绕过模型的安全对齐。我们使用可解释性工具表明，这种语义覆盖逐层出现，早期层的良性含义在后期层收敛为有害语义。Doublespeak无需优化，可跨模型家族广泛转移，在闭源和开源系统上取得高成功率，在Llama-3.3-70B-Instruct上单句上下文覆盖达到74%的攻击成功率。我们的发现突显了LLMs潜在空间中的新攻击面，揭示当前对齐策略不足，应该在表示层面进行操作。


### 论文摘要

We introduce \textbf{Doublespeak}, a simple \emph{in-context representation hijacking} attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., \textit{bomb}) with a benign token (e.g., \textit{carrot}) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., ``How to build a carrot?'') are internally interpreted as disallowed instructions (e.g., ``How to build a bomb?''), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74\% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level.

---

## 67. Origin-Conditional Trajectory Encoding: Measuring Urban Configurational Asymmetries through Neural Decomposition

**论文链接:** [http://arxiv.org/abs/2512.03755v1](http://arxiv.org/abs/2512.03755v1)

**作者:** Stephen Law, Tao Yang, Nanjiang Chen, Xuhui Lin

**发布时间:** 2025-12-03

### GPT解析

### 总结

该论文提出了一种条件轨迹编码器，解决了城市分析中空间和时间表示整合、方向不对称性和过度依赖辅助数据的问题。通过双向LSTM和对比学习，能够分解城市导航为共享模式和起点特定特征。实证研究表明城市形态导致系统性认知不平等，为城市规划、建筑设计和导航系统提供了新见解。

### 背景

城市分析越来越依赖人工智能驱动的轨迹分析，但当前方法存在方法论碎片化问题。轨迹学习方法捕捉移动模式但忽略空间上下文，而空间嵌入方法编码街道网络但错过时间动态。存在三个持续差距：缺乏整合空间和时间表示的联合训练；忽略导航中方向不对称性的起点无关处理；过度依赖辅助数据而非城市空间的基本几何属性。

### 目的

引入一个条件轨迹编码器，联合学习空间和移动表示，使用几何特征保持依赖于起点的方向不对称性。将城市导航分解为共享认知模式和起点特定空间叙事，实现跨起点的认知不对称性的定量测量。

### 方法

使用双向LSTM处理可见性比率和曲率特征，基于可学习的起点嵌入进行条件处理，通过对比学习将表示分解为共享的城市模式和起点特定特征。

### 主要发现

在六个合成城市和北京西城区的真实世界验证中，城市形态产生系统性认知不平等。该框架为城市规划者提供评估体验公平性的定量工具，为建筑师提供布局决策认知影响的洞见，并实现导航系统的起点感知分析。

### 结论

该框架解决了城市分析中的三个关键差距，提供了更全面、更准确的城市导航理解方法，有助于城市规划、建筑设计和导航系统的改进。

### 翻译

城市分析越来越依赖人工智能驱动的轨迹分析，然而当前方法存在方法论碎片化问题：轨迹学习捕捉移动模式但忽略空间上下文，而空间嵌入方法编码街道网络但错过时间动态。三个持续存在的差距：(1)缺乏整合空间和时间表示的联合训练，(2)忽略导航中方向不对称性的起点无关处理，(3)过度依赖辅助数据而非城市空间的基本几何属性。我们引入一个条件轨迹编码器，联合学习空间和移动表示，同时使用几何特征保持依赖于起点的方向不对称性。该框架将城市导航分解为共享认知模式和起点特定空间叙事，实现了跨起点的认知不对称性的定量测量。我们的双向LSTM处理可见性比率和曲率特征，条件是可学习的起点嵌入，通过对比学习将表示分解为共享的城市模式和起点特定特征。六个合成城市和北京西城区的真实世界验证结果表明，城市形态产生系统性认知不平等。这为城市规划者提供了评估体验公平性的定量工具，为建筑师提供了布局决策认知影响的洞见，并实现了导航系统的起点感知分析。


### 论文摘要

Urban analytics increasingly relies on AI-driven trajectory analysis, yet current approaches suffer from methodological fragmentation: trajectory learning captures movement patterns but ignores spatial context, while spatial embedding methods encode street networks but miss temporal dynamics. Three gaps persist: (1) lack of joint training that integrates spatial and temporal representations, (2) origin-agnostic treatment that ignores directional asymmetries in navigation ($A \to B \ne B \to A$), and (3) over-reliance on auxiliary data (POIs, imagery) rather than fundamental geometric properties of urban space. We introduce a conditional trajectory encoder that jointly learns spatial and movement representations while preserving origin-dependent asymmetries using geometric features. This framework decomposes urban navigation into shared cognitive patterns and origin-specific spatial narratives, enabling quantitative measurement of cognitive asymmetries across starting locations. Our bidirectional LSTM processes visibility ratio and curvature features conditioned on learnable origin embeddings, decomposing representations into shared urban patterns and origin-specific signatures through contrastive learning. Results from six synthetic cities and real-world validation on Beijing's Xicheng District demonstrate that urban morphology creates systematic cognitive inequalities. This provides urban planners quantitative tools for assessing experiential equity, offers architects insights into layout decisions' cognitive impacts, and enables origin-aware analytics for navigation systems.

---

## 68. 论文ID: 2512.03744v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03744v1.json'

---

## 69. GaussianBlender: Instant Stylization of 3D Gaussians with Disentangled Latent Spaces

**论文链接:** [http://arxiv.org/abs/2512.03683v1](http://arxiv.org/abs/2512.03683v1)

**作者:** Melis Ocal, Xiaoyan Xing, Yue Li, Ngo Anh Vien, Sezer Karaoglu, Theo Gevers

**发布时间:** 2025-12-03

### GPT解析

### 总结

GaussianBlender是一种创新的文本驱动3D风格化前馈框架，通过学习结构化的解耦潜在空间和利用潜在扩散模型实现即时编辑，提供高质量、多视角一致的3D风格化，无需每项资产的密集优化。

### 背景

3D风格化在游戏开发、虚拟现实和数字艺术中至关重要，对多样化资产的需求需要可扩展的方法。现有文本到3D风格化方法通常从2D图像编辑器中提取，需要每项资产密集的时间优化，并因当前文本到图像模型的局限性而表现出多视角不一致，不适用于大规模生产。

### 目的

引入GaussianBlender，一种用于文本驱动的3D风格化的前馈框架，实现推理时的即时编辑，解决现有方法的局限性，提供实用的大规模3D风格化解决方案。

### 方法

GaussianBlender学习结构化的解耦潜在空间，具有受控的信息共享，用于几何和外观，从空间分组的3D高斯体中获取这些表示。然后使用潜在扩散模型在这些学习到的表示上应用文本条件的编辑。

### 主要发现

全面的评估显示GaussianBlender提供即时、高保真、保持几何形状、多视角一致的3D风格化，超越了需要每实例测试时间优化的方法，解锁了实用的大规模民主化3D风格化。

### 结论

GaussianBlender为3D风格化提供了一种实用、高效、高质量的解决方案，克服了现有方法的局限性，特别是需要密集优化和多视角不一致的问题，使大规模3D风格化变得可行和民主化。

### 翻译

3D风格化是游戏开发、虚拟现实和数字艺术的核心，其中对多样化资产的需求需要支持快速、高保真操作的可扩展方法。现有的文本到3D风格化方法通常从2D图像编辑器中提取，需要每项资产密集的时间优化，并由于当前文本到图像模型的局限性而表现出多视角不一致，这使得它们不适用于大规模生产。在本文中，我们介绍了GaussianBlender，这是一种用于文本驱动3D风格化的开创性前馈框架，可以在推理时即时执行编辑。我们的方法从空间分组的3D高斯体中学习结构化的解耦潜在空间，具有受控的信息共享，用于几何和外观。然后潜在扩散模型在这些学习到的表示上应用文本条件的编辑。全面的评估表明，GaussianBlender不仅提供即时、高保真、保持几何形状、多视角一致的风格化，而且超越了需要每实例测试时间优化的方法——解锁了实用的大规模民主化3D风格化。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D风格化(editing)的效率和一致性问题。现有方法通常需要针对每个3D资产进行耗时的测试时优化，导致它们不适合大规模生产和交互式应用。这个问题在游戏开发、虚拟现实和数字艺术领域至关重要，因为这些领域需要多样化的3D资产和快速的风格探索能力，而传统3D风格化是一个繁琐、专家驱动且耗时的过程，限制了可扩展性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D编辑方法的局限性，包括需要逐个资产优化、多视角一致性不足以及几何保持不佳等问题。他们确定了三个主要挑战：3DGS的非结构性质、高斯参数分布不均匀以及需要解耦外观和几何。设计上，作者采用分组结构潜在空间来处理非结构问题，使用双分支架构实现几何和外观的解耦，并通过跨分支特征共享模块实现受控信息交换。他们借鉴了图像编辑中的扩散模型技术，使用Shap-E的文本条件降噪器，采用类似GaussianMAE的transformer架构，并利用InstructPix2Pix作为2D图像编辑器。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过学习结构化的、解耦的潜在空间，分别表示几何和外观，同时保持受控的信息共享，实现即时前馈的3D风格化编辑。整体实现分为三个阶段：1) 潜在空间学习：将输入高斯按空间邻近性分组，使用双分支3D VAE编码为几何和外观的解耦潜在表示，并通过跨分支特征共享实现信息交换；2) 潜在扩散预训练：使用文本条件潜在扩散模型学习去噪外观潜在表示，以几何潜在和文本提示为条件；3) 潜在编辑：适应预训练降噪器学习编辑函数，将源外观潜在映射到编辑后的潜在。推理时，给定3D高斯资产和编辑提示，GaussianBlender在单个前向传递中即时生成修改后的高质量、3D一致的资产。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个基于扩散的前馈3D高斯风格化编辑器，完全消除测试时优化；2) 分组结构的解耦潜在空间学习策略，通过空间邻近性分组并学习几何和外观的解耦表示；3) 双分支架构与跨分支特征共享，实现几何保持的、可控制的编辑。相比之前的工作，GaussianBlender与IN2N等优化方法相比，将推理时间从几分钟缩短到约0.26秒，且具有更好的多视角一致性和几何保持能力；与Shap-Editor相比，明确解耦了外观和几何，避免了共享表示带来的控制限制和结构失真问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GaussianBlender引入了一种基于扩散的前馈3D高斯风格化编辑方法，通过学习解耦的几何-外观潜在空间，实现了即时、高保真、几何保持且多视角一致的3D风格化编辑，完全消除了传统方法所需的耗时的每个资产测试时优化。'}


### 论文摘要

3D stylization is central to game development, virtual reality, and digital arts, where the demand for diverse assets calls for scalable methods that support fast, high-fidelity manipulation. Existing text-to-3D stylization methods typically distill from 2D image editors, requiring time-intensive per-asset optimization and exhibiting multi-view inconsistency due to the limitations of current text-to-image models, which makes them impractical for large-scale production. In this paper, we introduce GaussianBlender, a pioneering feed-forward framework for text-driven 3D stylization that performs edits instantly at inference. Our method learns structured, disentangled latent spaces with controlled information sharing for geometry and appearance from spatially-grouped 3D Gaussians. A latent diffusion model then applies text-conditioned edits on these learned representations. Comprehensive evaluations show that GaussianBlender not only delivers instant, high-fidelity, geometry-preserving, multi-view consistent stylization, but also surpasses methods that require per-instance test-time optimization - unlocking practical, democratized 3D stylization at scale.

---

## 70. Feature-aware Modulation for Learning from Temporal Tabular Data

**论文链接:** [http://arxiv.org/abs/2512.03678v1](http://arxiv.org/abs/2512.03678v1)

**作者:** Hao-Run Cai, Han-Jia Ye

**发布时间:** 2025-12-03

**备注:** 17 pages, 6 figures, 8 tables. NeurIPS 2025

### GPT解析

### 总结

该论文提出了一种基于特征的时域调制机制，通过条件化特征表示于时域上下文，调制统计属性如尺度和偏度，从而在表格数据中有效处理时间分布转移，平衡泛化能力和适应性。

### 背景

表格机器学习在现实部署中面临时间分布转移的挑战，因为特征和标签之间的关系会持续变化。静态模型假设固定映射以确保泛化能力，而自适应模型可能对瞬时模式过拟合，导致鲁棒性和适应性之间的两难困境。

### 目的

分析构建有效的时间表格数据动态映射的关键因素，提出一种能够平衡泛化能力和适应性的方法。

### 方法

提出一种基于特征的时域调制机制，该机制将特征表示条件化于时域上下文，调制统计属性如尺度和偏度，通过对齐跨时间的特征语义来实现轻量而强大的自适应。

### 主要发现

演化的特征语义(特别是客观和主观含义)随时间引入概念漂移；特征转换策略能够缓解跨时间阶段的特征表示差异；通过调制特征的统计属性可以实现对齐特征语义。

### 结论

所提出的方法在处理表格数据中的时间转移方面是有效的，能够平衡泛化能力和适应性。

### 翻译

虽然表格机器学习已经取得了显著的成功，但时间分布转移在现实部署中带来了重大挑战，因为特征和标签之间的关系在不断演变。静态模型假设固定映射以确保泛化能力，而自适应模型可能对瞬时模式过拟合，在鲁棒性和适应性之间产生两难困境。在本文中，我们分析了构建有效的时间表格数据动态映射的关键因素。我们发现演化的特征语义-特别是客观和主观含义-随时间引入概念漂移。关键的是，我们确定特征转换策略能够缓解跨时间阶段的特征表示差异。受这些见解的启发，我们提出了一种基于特征的时域调制机制，该机制将特征表示条件化于时域上下文，调制尺度、偏度等统计属性。通过对齐跨时间的特征语义，我们的方法实现了轻量而强大的自适应，有效地平衡了泛化能力和适应性。基准评估验证了我们的方法在处理表格数据中时间转移方面的有效性。


### 论文摘要

While tabular machine learning has achieved remarkable success, temporal distribution shifts pose significant challenges in real-world deployment, as the relationships between features and labels continuously evolve. Static models assume fixed mappings to ensure generalization, whereas adaptive models may overfit to transient patterns, creating a dilemma between robustness and adaptability. In this paper, we analyze key factors essential for constructing an effective dynamic mapping for temporal tabular data. We discover that evolving feature semantics-particularly objective and subjective meanings-introduce concept drift over time. Crucially, we identify that feature transformation strategies are able to mitigate discrepancies in feature representations across temporal stages. Motivated by these insights, we propose a feature-aware temporal modulation mechanism that conditions feature representations on temporal context, modulating statistical properties such as scale and skewness. By aligning feature semantics across time, our approach achieves a lightweight yet powerful adaptation, effectively balancing generalizability and adaptability. Benchmark evaluations validate the effectiveness of our method in handling temporal shifts in tabular data.

---

## 71. Cyclical Temporal Encoding and Hybrid Deep Ensembles for Multistep Energy Forecasting

**论文链接:** [http://arxiv.org/abs/2512.03656v1](http://arxiv.org/abs/2512.03656v1)

**作者:** Salim Khazem, Houssam Kanso

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种统一的深度学习框架，结合循环时间编码和混合LSTM-CNN架构，以提高电力消耗预测的准确性。通过使用正弦余弦编码处理日历属性，并采用集成模型同时利用长期季节效应和短期局部模式，实验结果表明该方法在七个预测范围内均优于现有方法。

### 背景

准确的电力消耗预测对需求管理和智能电网运营至关重要，但现有方法在多步能源预测中仍有改进空间。

### 目的

开发一个统一的深度学习框架，整合循环时间编码与混合LSTM-CNN架构，以提高多步能源预测的准确性。

### 方法

1) 使用正弦余弦编码转换日历属性以保留周期结构；2) 通过相关性分析评估预测相关性；3) 构建集成模型(LSTM、CNN和针对各预测范围的MLP回归器元学习器)；4) 使用一年国家级消耗数据集进行实验，包括消融分析和与基线方法的比较。

### 主要发现

混合模型在所有七个预测范围内均表现出一致的改进，实现了比单独架构和先前方法更低的RMSE和MAE误差指标。

### 结论

结合循环时间表示与互补的深度学习结构能够有效提升短期能源预测性能。

### 翻译

准确的电力消耗预测对需求管理和智能电网运营至关重要。本文引入了一个统一的深度学习框架，结合循环时间编码和混合LSTM-CNN架构，以提高多步能源预测能力。我们使用正弦余弦编码系统地将基于日历的属性进行转换，以保留周期结构，并通过相关性分析评估它们的预测相关性。为利用长期季节效应和短期局部模式，我们采用了一个由LSTM、CNN和针对每个预测范围专门化的MLP回归器元学习器组成的集成模型。使用一年国家级消耗数据集，我们进行了广泛的实验研究，包括包含和不包含循环编码和日历特征的消融分析，以及与文献中既定基线的比较。结果表明，在所有七个预测范围内都表现出一致的改进，我们的混合模型实现了比单独架构和先前方法更低的RMSE和MAE。这些发现证实了结合循环时间表示与互补深度学习结构的好处。据我们所知，这是第一个在统一的短期能源预测框架内联合评估时间编码、基于日历的特征和混合集成架构的工作。


### 论文摘要

Accurate electricity consumption forecasting is essential for demand management and smart grid operations. This paper introduces a unified deep learning framework that integrates cyclical temporal encoding with hybrid LSTM-CNN architectures to enhance multistep energy forecasting. We systematically transform calendar-based attributes using sine cosine encodings to preserve periodic structure and evaluate their predictive relevance through correlation analysis. To exploit both long-term seasonal effects and short-term local patterns, we employ an ensemble model composed of an LSTM, a CNN, and a meta-learner of MLP regressors specialized for each forecast horizon. Using a one year national consumption dataset, we conduct an extensive experimental study including ablation analyses with and without cyclical encodings and calendar features and comparisons with established baselines from the literature. Results demonstrate consistent improvements across all seven forecast horizons, with our hybrid model achieving lower RMSE and MAE than individual architectures and prior methods. These findings confirm the benefit of combining cyclical temporal representations with complementary deep learning structures. To our knowledge, this is the first work to jointly evaluate temporal encodings, calendar-based features, and hybrid ensemble architectures within a unified short-term energy forecasting framework.

---

## 72. Optical Context Compression Is Just (Bad) Autoencoding

**论文链接:** [http://arxiv.org/abs/2512.03643v1](http://arxiv.org/abs/2512.03643v1)

**作者:** Ivan Yee Lee, Cheng Yang, Taylor Berg-Kirkpatrick

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究质疑了DeepSeek-OCR提出的视觉上下文压缩方法对语言建模的有效性，通过对比实验发现简单方法在重建和语言建模任务上均表现更优。

### 背景

DeepSeek-OCR展示了从少量视觉token中高保真重建渲染文本的能力，引发了对基于视觉的上下文压缩在语言模型中应用的兴奋，但评估仅限于重建阶段。

### 目的

测试光学压缩叙事中隐含的两个假设：基于视觉的压缩为文本重建提供独特优势，以及DeepSeek-OCR的重建结果证明视觉压缩对语言建模有用。

### 方法

将DeepSeek-OCR的视觉编码器与简单替代方案（无参数平均池化和学习到的分层编码器）进行比较，在相同压缩比率下评估重建和语言建模性能。

### 主要发现

在相同压缩比率下，简单方法在重建方面匹配或超越了视觉方法；在语言建模方面，简单方法优于视觉方法；基于视觉的压缩无法超越简单的截断方法。

### 结论

对光学上下文压缩的热情超过了现有证据支持，需要更谨慎评估此类方法的优势。

### 翻译

DeepSeek-OCR证明可以从少量视觉token中高保真重建渲染文本。这一发现引发了人们对基于视觉的上下文压缩在语言模型中应用的兴奋。但评估仅停留在重建阶段；这些表示是否有助于语言建模尚未得到检验。我们测试了光学压缩叙事中隐含的两个假设：基于视觉的压缩为从压缩表示中重建文本提供了独特优势，以及DeepSeek-OCR的重建结果是视觉压缩对语言建模有用的证据。将其视觉编码器与简单替代方案（无参数平均池化和学习到的分层编码器）进行比较，我们发现在相同压缩比率下，这些简单方法在重建方面匹配或超越了视觉方法，并且在语言建模方面优于视觉方法——基于视觉的压缩无法超越截断。对光学上下文压缩的热情超过了现有证据。代码和模型可在https://github.com/ivnle/bad-autoencoding获取。


### 论文摘要

DeepSeek-OCR demonstrates that rendered text can be reconstructed with high fidelity from a small number of vision tokens. This finding has sparked excitement about vision-based context compression for language models. But the evaluation stops at reconstruction; whether these representations help language modeling remains untested. We test two assumptions implicit in the optical-compression narrative: that vision-based compression provides unique advantages for text reconstruction from compressed representations, and that DeepSeek-OCR's reconstruction results are evidence that vision-based compression will be useful for language modeling. Comparing their vision encoder against simple alternatives--parameter-free mean pooling and a learned hierarchical encoder--we find that these simple approaches match or surpass vision for reconstruction at matched compression ratios, and outperform it for language modeling--where vision-based compression fails to beat truncation. The excitement around optical context compression outpaces the evidence. Code and checkpoints are available at https://github.com/ivnle/bad-autoencoding

---

## 73. AaPE: Aliasing-aware Patch Embedding for Self-Supervised Audio Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.03637v1](http://arxiv.org/abs/2512.03637v1)

**作者:** Kohei Yamamoto, Kosuke Okusa

**发布时间:** 2025-12-03

**备注:** 11 pages, 4 figures

### GPT解析

### 总结

本文提出了一种名为Aliasing-aware Patch Embedding (AaPE)的新型音频处理方法，用于解决Transformer-based自监督学习模型中频谱图处理导致的混叠问题，同时保留高频信息。

### 背景

基于Transformer的音频自监督学习模型通常将频谱图视为图像，采用带有大量时间下采样的卷积块划分方法，这降低了有效奈奎斯特频率并引入混叠现象，而简单的低通滤波会去除任务相关的高频线索。

### 目的

开发一种能够减轻混叠同时保留高频信息的插入式块嵌入方法，提升音频自监督学习模型的性能。

### 方法

AaPE使用双边指数窗口的带限复正弦核增强标准块标记，动态针对容易混叠的频带；从输入估计核参数，实现并行自适应子带分析；将子带输出与标准块标记融合；无缝集成到掩码教师-学生自监督学习中；结合多掩码策略和对比目标确保不同掩码模式间的一致性，稳定训练。

### 主要发现

在AudioSet上预训练后，在多种下游音频任务（包括环境声音等）上微调，在部分任务上达到最先进性能，其他任务具有竞争力；线性探测评估也显示在多个基准测试上有明显提升。

### 结论

AaPE方法能够有效减轻混叠效应，同时不丢弃有用的高频内容，提升了音频自监督学习模型的性能。

### 翻译

基于Transformer的音频自监督学习模型通常将频谱图视为图像，应用带有大量时间下采样的卷积块划分方法。这降低了有效的奈奎斯特频率并引入混叠，而简单的低通滤波会去除任务相关的高频线索。在本研究中，我们提出了Aliasing-aware Patch Embedding (AaPE)，一种可以减轻混叠同时保留高频信息的插入式块嵌入方法。AaPE使用双面指数窗口的带限复正弦核来增强标准块标记，该窗口动态针对容易混叠的频带。从输入中估计核的频率和衰减参数，实现并行自适应子带分析，其输出与标准块标记融合。AaPE无缝集成到掩码教师-学生自监督学习中。此外，我们将多掩码策略与对比目标相结合，确保不同掩码模式间的一致性，稳定训练。在AudioSet上预训练，然后在各种下游基准测试（包括环境声音和其他常见音频领域）上进行微调评估。该方法在部分任务上取得了最先进的性能，在其他任务上具有竞争力。互补的线性探测评估也反映了这一模式，在几个基准测试上取得了明显提升，在其他地方也表现出强劲性能。对这些结果的总体分析表明，AaPE有助于减轻混叠效应，同时不丢弃有用的高频内容。


### 论文摘要

Transformer-based audio SSL (self-supervised learning) models often treat spectrograms as images, applying convolutional patchification with heavy temporal downsampling. This lowers the effective Nyquist frequency and introduces aliasing, while naïve low-pass filtering removes task-relevant high-frequency cues. In this study, we present Aliasing-aware Patch Embedding (AaPE), a drop-in patch stem that mitigates aliasing while preserving high-frequency information. AaPE augments standard patch tokens with features produced by a band-limited complex sinusoidal kernel using a two-sided exponential window that dynamically targets alias-prone bands. Frequency and decay parameters of the kernel are estimated from the input, enabling parallel, adaptive subband analysis whose outputs are fused with the standard patch tokens. AaPE integrates seamlessly into the masked teacher-student self-supervised learning. In addition, we combine a multi-mask strategy with a contrastive objective to enforce consistency across diverse mask patterns, stabilizing training. Pre-training on AudioSet followed by fine-tuning evaluation across diverse downstream benchmarks, which spanned categories, such as environmental sounds and other common audio domains. This approach yields state-of-the-art performance on a subset of tasks and competitive results across the remainder. Complementary linear probing evaluation mirrors this pattern, yielding clear gains on several benchmarks and strong performance elsewhere. The collective analysis of these results indicates that AaPE serves to mitigate the effects of aliasing without discarding of informative high-frequency content.

---

## 74. Observation-driven correction of numerical weather prediction for marine winds

**论文链接:** [http://arxiv.org/abs/2512.03606v1](http://arxiv.org/abs/2512.03606v1)

**作者:** Matteo Peduto, Qidong Yang, Jonathan Giezendanner, Devis Tuia, Sherrie Wang

**发布时间:** 2025-12-03

### GPT解析

### 总结

这项研究提出了一种基于Transformer的深度学习方法，通过整合现场观测数据来校正全球数值天气预报模型，显著提高了海洋风预报的准确性。

### 背景

准确的海洋风预报对安全航行、船舶路线规划和能源运营至关重要，但由于海洋上的观测数据稀少、异构且随时间变化，风预报仍然具有挑战性。

### 目的

将风预报重新表述为全球数值天气预报模型的观测信息校正，通过整合最新现场观测数据来调整全球预报系统输出，学习局部校正模式。

### 方法

提出一种基于Transformer的深度学习架构，通过掩码和基于集合的注意力机制处理不规则观测集合，利用交叉注意力机制基于观测-预报对进行条件预测，采用循环时间嵌入和坐标感知的位置表示实现任意空间坐标的单次推理。

### 主要发现

在大西洋地区评估显示，模型在所有提前时间(最多48小时)内都降低了GFS 10米风RMSE，1小时提前时间实现45%改进，48小时提前时间实现13%改进；沿岸和航运路线上的改进最为持久；该架构能自然适应异构观测平台，同时产生站点预测和盆地规模网格化产品。

### 结论

这是一种实用的低延迟后处理方法，通过学习校正系统性预报误差来补充数值天气预报模型。

### 翻译

准确的海洋风预报对安全航行、船舶路线规划和能源运营至关重要，但由于海洋上的观测数据稀少、异构且随时间变化，风预报仍然具有挑战性。我们将风预报重新表述为全球数值天气预报模型的观测信息校正。我们不直接预报风，而是通过整合最新的现场观测数据来调整全球预报系统输出，从而学习局部校正模式。我们提出了一种基于Transformer的深度学习架构，(i)通过掩码和基于集合的注意力机制处理不规则且随时间变化的观测集合，(ii)通过交叉注意力机制，基于最近的观测-预报对进行条件预测，(iii)采用循环时间嵌入和感知坐标的位置表示，能够在任意空间坐标上进行单次推理。我们使用国际综合海洋-大气数据集的观测数据作为参考，在大西洋地区评估了我们的模型。该模型在所有提前时间(最多48小时)内都降低了GFS 10米风的RMSE，在1小时提前时间实现了45%的改进，在48小时提前时间实现了13%的改进。空间分析显示，沿岸和航运路线上的改进最为持久，这些区域的观测数据最为丰富。分词架构自然适应异构观测平台(船舶、浮标、潮汐计和沿海站)，并在单次前向传递中产生特定站点的预测和盆地规模的网格化产品。这些结果表明，这是一种实用的低延迟后处理方法，通过学习校正系统性预报误差来补充NWP。


### 论文摘要

Accurate marine wind forecasts are essential for safe navigation, ship routing, and energy operations, yet they remain challenging because observations over the ocean are sparse, heterogeneous, and temporally variable. We reformulate wind forecasting as observation-informed correction of a global numerical weather prediction (NWP) model. Rather than forecasting winds directly, we learn local correction patterns by assimilating the latest in-situ observations to adjust the Global Forecast System (GFS) output. We propose a transformer-based deep learning architecture that (i) handles irregular and time-varying observation sets through masking and set-based attention mechanisms, (ii) conditions predictions on recent observation-forecast pairs via cross-attention, and (iii) employs cyclical time embeddings and coordinate-aware location representations to enable single-pass inference at arbitrary spatial coordinates. We evaluate our model over the Atlantic Ocean using observations from the International Comprehensive Ocean-Atmosphere Data Set (ICOADS) as reference. The model reduces GFS 10-meter wind RMSE at all lead times up to 48 hours, achieving 45% improvement at 1-hour lead time and 13% improvement at 48-hour lead time. Spatial analyses reveal the most persistent improvements along coastlines and shipping routes, where observations are most abundant. The tokenized architecture naturally accommodates heterogeneous observing platforms (ships, buoys, tide gauges, and coastal stations) and produces both site-specific predictions and basin-scale gridded products in a single forward pass. These results demonstrate a practical, low-latency post-processing approach that complements NWP by learning to correct systematic forecast errors.

---

## 75. Motion4D: Learning 3D-Consistent Motion and Semantics for 4D Scene Understanding

**论文链接:** [http://arxiv.org/abs/2512.03601v1](http://arxiv.org/abs/2512.03601v1)

**作者:** Haoran Zhou, Gim Hee Lee

**发布时间:** 2025-12-03

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

Motion4D是一个新框架，通过整合基础模型的2D先验到4D高斯溅射表示中，解决了现有模型在3D一致性问题上的缺陷，实现了在复杂3D环境中的稳定表现。

### 背景

基础模型在2D视觉领域的进展显著改善了单目视频中动态场景的分析能力，但这些模型通常缺乏3D一致性，无法满足理解场景几何和运动的基本要求。

### 目的

开发一个新框架，解决现有模型在3D一致性方面的缺陷，减少复杂3D环境中的空间错位和时间闪烁问题。

### 方法

Motion4D框架整合了2D先验到统一的4D高斯溅射表示中，采用两部分迭代优化：1)序列优化更新运动和语义场保持局部一致性；2)全局优化联合优化所有属性实现长期一致性。同时引入3D置信图调整运动先验，自适应重采样过程处理表示不足区域，以及迭代细化过程增强语义一致性。

### 主要发现

Motion4D在点跟踪、视频对象分割和新视图合成等多种场景理解任务上，显著优于2D基础模型和现有3D方法，解决了空间错位和时间闪烁问题。

### 结论

Motion4D成功地将2D基础模型的先验知识与3D一致性需求相结合，为动态场景分析提供了一个强大的新框架，代码已开源。

### 翻译

最近在2D视觉基础模型方面的进展显著改善了对单目视频中动态场景的分析。然而，尽管这些模型具有强大的泛化能力，但它们通常缺乏3D一致性，这是理解场景几何和运动的基本要求，从而导致复杂3D环境中的严重空间错位和时间闪烁。在本文中，我们提出了Motion4D，一个新颖的框架，通过将基础模型的2D先验整合到统一的4D高斯溅射表示中解决了这些挑战。我们的方法包含一个两部分迭代优化框架：1)序列优化，在连续阶段更新运动和语义场以保持局部一致性；2)全局优化，联合优化所有属性以实现长期一致性。为了提高运动准确性，我们引入了3D置信图，动态调整运动先验，以及一个自适应重采样过程，基于每像素RGB和语义错误将新高斯插入表示不足的区域。此外，我们通过迭代细化过程增强语义一致性，通过交替优化语义场和更新SAM2的提示来解决语义不一致问题。大量评估表明，在点跟踪、视频对象分割和新视图合成等多种场景理解任务上，我们的Motion4D显著优于2D基础模型和现有的3D方法。我们的代码可在https://hrzhou2.github.io/motion4d-web/获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决2D视觉基础模型在处理动态场景时缺乏3D一致性的问题，导致空间错位和时间闪烁现象。这个问题在计算机视觉领域非常重要，因为3D一致性是理解场景几何和运动的基本要求，对于机器人、自动驾驶、增强现实等应用至关重要。现有方法在复杂3D环境中表现不佳，限制了动态场景分析的质量和应用范围。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到2D视觉基础模型（如SAM2）虽然泛化能力强但缺乏3D一致性，因此考虑将2D先验整合到3D表示中。他们选择了3D高斯溅射作为基础表示方法，并设计了两部分迭代优化框架（顺序优化和全局优化）。为了提高运动准确性，引入了3D置信图和自适应重采样；为了增强语义一致性，设计了迭代细化过程。该方法借鉴了SAM2的视频分割、Track Any Point的点跟踪、Depth Anything的深度估计以及3D高斯溅射的表示方法，但针对动态场景进行了创新性改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'Motion4D的核心思想是将2D视觉基础模型的先验知识整合到统一的4D高斯溅射表示中，通过迭代优化提高3D一致性。整体流程包括：1) 输入视频序列和2D先验（掩码、点轨迹、深度）；2) 使用带运动和语义字段的3D高斯溅射表示场景；3) 迭代运动细化（使用3D置信图和自适应重采样）；4) 迭代语义细化（通过渲染3D掩码并更新SAM2提示）；5) 两部分优化（顺序优化先局部后全局，确保长期一致性）；6) 输出时空一致的运动和语义预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的动态表示将2D先验整合到3D高斯溅射中；2) 两部分迭代优化框架（顺序和全局优化）；3) 使用3D置信图和自适应重采样的迭代运动细化；4) 通过迭代更新SAM2提示的语义细化；5) 新的DyCheck-VOS基准数据集。相比之前的工作，Motion4D提供了3D一致性，减少了空间错位和时间闪烁；联合建模语义和运动而非分离处理；通过迭代优化策略而非简单整合；专门针对动态场景而非仅静态场景进行优化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Motion4D通过将2D视觉基础模型先验整合到4D高斯溅射表示中并采用迭代优化策略，实现了动态场景中运动和语义的3D一致性建模，显著提升了视频对象分割、点跟踪和新视图合成的性能。'}


### 论文摘要

Recent advancements in foundation models for 2D vision have substantially improved the analysis of dynamic scenes from monocular videos. However, despite their strong generalization capabilities, these models often lack 3D consistency, a fundamental requirement for understanding scene geometry and motion, thereby causing severe spatial misalignment and temporal flickering in complex 3D environments. In this paper, we present Motion4D, a novel framework that addresses these challenges by integrating 2D priors from foundation models into a unified 4D Gaussian Splatting representation. Our method features a two-part iterative optimization framework: 1) Sequential optimization, which updates motion and semantic fields in consecutive stages to maintain local consistency, and 2) Global optimization, which jointly refines all attributes for long-term coherence. To enhance motion accuracy, we introduce a 3D confidence map that dynamically adjusts the motion priors, and an adaptive resampling process that inserts new Gaussians into under-represented regions based on per-pixel RGB and semantic errors. Furthermore, we enhance semantic coherence through an iterative refinement process that resolves semantic inconsistencies by alternately optimizing the semantic fields and updating prompts of SAM2. Extensive evaluations demonstrate that our Motion4D significantly outperforms both 2D foundation models and existing 3D-based approaches across diverse scene understanding tasks, including point-based tracking, video object segmentation, and novel view synthesis. Our code is available at https://hrzhou2.github.io/motion4d-web/.

---

## 76. Harnessing Hypergraphs in Geometric Deep Learning for 3D RNA Inverse Folding

**论文链接:** [http://arxiv.org/abs/2512.03592v1](http://arxiv.org/abs/2512.03592v1)

**作者:** Guang Yang, Lei Fan

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为HyperRNA的框架，用于解决RNA逆向折叠问题，这是一个基于超图的生成模型，采用编码器-解码器架构来设计RNA序列。

### 背景

RNA逆向折叠问题是RNA设计中的一个关键挑战，涉及识别能够折叠成期望二级结构的核苷酸序列，这对确保分子稳定性和功能至关重要。该任务的复杂性源于序列和结构之间的复杂关系。

### 目的

提出一个名为HyperRNA的框架，作为解决RNA逆向折叠问题的生成模型，利用超图来设计RNA序列。

### 方法

HyperRNA模型包含三个主要组件：预处理阶段基于3珠粗粒化表示构建RNA骨架的图结构；编码阶段使用注意力嵌入模块和基于超图的编码器处理图结构，捕获高阶依赖性和复杂的生物分子相互作用；解码阶段以自回归方式生成RNA序列。研究者在PDBBind和RNAsolo数据集上进行了定量和定性实验。

### 主要发现

实验结果表明，HyperRNA不仅优于现有的RNA设计方法，还突显了在RNA工程中利用超图的潜力。

### 结论

HyperRNA框架有效地解决了RNA逆向折叠问题，通过超图捕获RNA序列和结构之间复杂关系的能力使其成为RNA设计的强大工具。

### 翻译

RNA逆向折叠问题是RNA设计中的一个关键挑战，涉及识别能够折叠成期望二级结构的核苷酸序列，这对确保分子稳定性和功能至关重要。这个任务的固有复杂性来自于序列和结构之间的复杂关系，使其特别具有挑战性。在本文中，我们提出了一个名为HyperRNA的框架，这是一个具有编码器-解码器架构的生成模型，利用超图来设计RNA序列。具体来说，我们的HyperRNA模型包含三个主要组件：预处理、编码和解码。在预处理阶段，基于3珠粗粒化表示提取RNA骨架的原子坐标来构建图结构。编码阶段处理这些图，使用注意力嵌入模块和基于超图的编码器捕获高阶依赖性和复杂的生物分子相互作用。最后，解码阶段以自回归方式生成RNA序列。我们在PDBBind和RNAsolo数据集上进行了定量和定性实验，评估了RNA序列生成和RNA-蛋白质复合物序列生成的逆向折叠任务。实验结果表明，HyperRNA不仅优于现有的RNA设计方法，还突显了在RNA工程中利用超图的潜力。


### 论文摘要

The RNA inverse folding problem, a key challenge in RNA design, involves identifying nucleotide sequences that can fold into desired secondary structures, which are critical for ensuring molecular stability and function. The inherent complexity of this task stems from the intricate relationship between sequence and structure, making it particularly challenging. In this paper, we propose a framework, named HyperRNA, a generative model with an encoder-decoder architecture that leverages hypergraphs to design RNA sequences. Specifically, our HyperRNA model consists of three main components: preprocessing, encoding and decoding.   In the preprocessing stage, graph structures are constructed by extracting the atom coordinates of RNA backbone based on 3-bead coarse-grained representation. The encoding stage processes these graphs, capturing higher order dependencies and complex biomolecular interactions using an attention embedding module and a hypergraph-based encoder. Finally, the decoding stage generates the RNA sequence in an autoregressive manner. We conducted quantitative and qualitative experiments on the PDBBind and RNAsolo datasets to evaluate the inverse folding task for RNA sequence generation and RNA-protein complex sequence generation. The experimental results demonstrate that HyperRNA not only outperforms existing RNA design methods but also highlights the potential of leveraging hypergraphs in RNA engineering.

---

## 77. GAOT: Generating Articulated Objects Through Text-Guided Diffusion Models

**论文链接:** [http://arxiv.org/abs/2512.03566v1](http://arxiv.org/abs/2512.03566v1)

**作者:** Hao Sun, Lei Fan, Donglin Di, Shaohui Liu

**发布时间:** 2025-12-03

**备注:** Accepted by ACM MM Asia2026

### GPT解析

### 总结

本文提出GAOT框架，通过结合扩散模型和超图学习，实现了从文本提示生成3D关节物体的能力，解决了文本描述与3D关节物体表示之间的差距问题。

### 背景

关节物体生成领域虽有进展，但现有模型通常缺乏根据文本提示生成的能力。

### 目的

弥合文本描述与3D关节物体表示之间的显著差距。

### 方法

提出GAOT，一个三阶段框架：1)微调点云生成模型从文本提示产生物体粗略表示；2)设计基于超图的学习方法精细化表示，将物体部分表示为图顶点；3)利用扩散模型基于物体部分生成关节物体的关节（表示为图边）。

### 主要发现

在PartNet-Mobility数据集上的定性和定量实验表明，GAOT方法有效且性能优于先前方法。

### 结论

GAOT框架成功实现了从文本提示生成3D关节物体的目标，解决了文本到3D关节物体生成的挑战。

### 翻译

关节物体生成已取得越来越多的进展，然而现有模型往往缺乏基于文本提示生成的能力。为了解决文本描述与3D关节物体表示之间的显著差距，我们提出了GAOT，一个三阶段框架，利用扩散模型和超图学习，通过三步过程从文本提示生成关节物体。首先，我们微调点云生成模型，从文本提示产生物体的粗略表示。鉴于关节物体与图结构之间的固有联系，我们设计了一种基于超图的学习方法来细化这些粗略表示，将物体部分表示为图顶点。最后，利用扩散模型，基于物体部分生成关节物体的关节（表示为图边）。在PartNet-Mobility数据集上的大量定性和定量实验证明了我们方法的有效性，实现了优于先前方法的性能。


### 论文摘要

Articulated object generation has seen increasing advancements, yet existing models often lack the ability to be conditioned on text prompts. To address the significant gap between textual descriptions and 3D articulated object representations, we propose GAOT, a three-phase framework that generates articulated objects from text prompts, leveraging diffusion models and hypergraph learning in a three-step process. First, we fine-tune a point cloud generation model to produce a coarse representation of objects from text prompts. Given the inherent connection between articulated objects and graph structures, we design a hypergraph-based learning method to refine these coarse representations, representing object parts as graph vertices. Finally, leveraging a diffusion model, the joints of articulated objects-represented as graph edges-are generated based on the object parts. Extensive qualitative and quantitative experiments on the PartNet-Mobility dataset demonstrate the effectiveness of our approach, achieving superior performance over previous methods.

---

## 78. Parameter-Efficient Augment Plugin for Class-Incremental Learning

**论文链接:** [http://arxiv.org/abs/2512.03537v1](http://arxiv.org/abs/2512.03537v1)

**作者:** Zhiming Xu, Baile Xu, Jian Zhao, Furao Shen, Suorong Yang

**发布时间:** 2025-12-03

**备注:** 10 pages, 6 figures, 2 tables

### GPT解析

### 总结

这篇论文提出了一种名为DLC(部署额外LoRA组件)的即插即用扩展范式，用于非预训练的增量学习场景。该方法通过LoRA技术向基模型的深层注入任务特定残差，并引入轻量级加权单元减轻干扰，在保持低参数量的同时显著提升了模型性能。

### 背景

现有的基于重放或知识蒸馏的增量学习方法通常受到遗忘问题或稳定性-可塑性困境的限制。而一些基于扩展的方法虽然能实现更高精度，但往往需要显著增加参数量。

### 目的

提出一种高效的参数扩展方法，解决非预训练增量学习场景中的遗忘和稳定性-可塑性问题，实现在不大幅增加参数量的情况下提升模型性能。

### 方法

将经过重放或蒸馏训练的特征提取器视为具有丰富知识的基模型，通过LoRA技术向基模型的深层注入任务特定的残差。在推理时，聚合具有任务特定残差的表示以生成分类预测。引入轻量级加权单元来分配不同LoRA调整表示的重要性分数，减轻非目标LoRA组件的干扰。

### 主要发现

在ImageNet-100上，仅使用标准ResNet-18参数量的4%，DLC模型实现了8%的精度提升，显示出卓越的效率。在固定内存预算下，该方法能够超越最先进的方法。

### 结论

DLC方法作为一种即插即用的增强方式，能够高效扩展基方法，解决增量学习中的遗忘问题，并在保持低参数量的同时显著提升模型性能。

### 翻译

现有的基于重放或知识蒸馏的增量学习方法通常受到遗忘或稳定性-可塑性困境的限制。一些基于扩展的方法可以实现更高的准确性，但它们总是需要显著增加参数量。在本文中，我们提出了一种称为'部署额外LoRA组件(DLC)'的插件扩展范式，用于非预训练的增量学习场景。我们将通过重放或蒸馏训练的特征提取器视为一个具有丰富知识的基模型。对于每个任务，我们使用低秩自适应(LoRA)向基模型的深层注入任务特定的残差。在推理时，聚合具有任务特定残差的表示以产生分类预测。为了减轻来自非目标LoRA插件的干扰，我们引入了一个轻量级加权单元。该单元学习为不同的LoRA调整表示分配重要性分数。像软件中的可下载内容一样，我们的方法作为一种即插即用的增强，有效地扩展了基方法。值得注意的是，在大型ImageNet-100上，仅使用标准ResNet-18参数量的4%，我们的DLC模型实现了8%的显著精度提升，展现了卓越的效率。此外，在固定内存预算下，它可以超越最先进的方法。


### 论文摘要

Existing class-incremental learning (CIL) approaches based on replay or knowledge distillation are often constrained by forgetting or the stability-plasticity dilemma. Some expansion-based approaches could achieve higher accuracy. However, they always require significant parameter increases. In this paper, we propose a plugin extension paradigm termed the Deployment of extra LoRA Components (DLC) for non-pre-trained CIL scenarios.We treat the feature extractor trained through replay or distillation as a base model with rich knowledge. For each task, we use Low-Rank Adaptation (LoRA) to inject task-specific residuals into the base model's deep layers. During inference, representations with task-specific residuals are aggregated to produce classification predictions. To mitigate interference from non-target LoRA plugins, we introduce a lightweight weighting unit. This unit learns to assign importance scores to different LoRA-tuned representations. Like downloadable contents in software, our method serves as a plug-and-play enhancement that efficiently extends the base methods. Remarkably, on the large-scale ImageNet-100, with merely 4 % of the parameters of a standard ResNet-18, our DLC model achieves a significant 8 % improvement in accuracy, demonstrating exceptional efficiency. Moreover, it could surpass state-of-the-art methods under the fixed memory budget.

---

## 79. Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation

**论文链接:** [http://arxiv.org/abs/2512.03521v1](http://arxiv.org/abs/2512.03521v1)

**作者:** Xiaosen Lyu, Jiayu Xiong, Yuren Chen, Wanlong Wang, Xiaoqing Dai, Jing Wang

**发布时间:** 2025-12-03

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

本文提出了一种名为Cross-Space Synergy (CSS)的新方法，用于解决多模态情感识别中跨模态交互捕捉困难和训练不稳定的问题。

### 背景

多模态情感识别(MERC)旨在通过整合文本、声学和视觉线索来预测说话者的情绪。现有方法要么难以捕捉复杂的跨模态交互，要么在使用更深层次架构时遇到梯度冲突和训练不稳定的问题。

### 目的

解决现有方法中难以捕捉复杂跨模态交互以及梯度冲突和训练不稳定的问题。

### 方法

提出Cross-Space Synergy (CSS)，结合表示组件与优化组件：1) Synergistic Polynomial Fusion (SPF)作为表示组件，利用低秩张量分解捕捉高阶跨模态交互；2) Pareto Gradient Modulator (PGM)作为优化组件，引导更新沿着竞争目标间的帕累托最优方向进行，以减轻梯度冲突并提高稳定性。

### 主要发现

CSS在IEMOCAP和MELD数据集上，在准确性和训练稳定性方面都优于现有的代表性方法。

### 结论

CSS在复杂的多模态场景中表现出有效性。

### 翻译

多模态对话情感识别(MERC)旨在通过整合文本、声学和视觉线索来预测说话者的情绪。现有方法要么难以捕捉复杂的跨模态交互，要么在使用更深层次架构时遇到梯度冲突和训练不稳定的问题。为解决这些问题，我们提出了Cross-Space Synergy (CSS)，它将表示组件与优化组件相结合。协同多项式融合(SPF)作为表示组件，利用低秩张量分解来有效捕捉高阶跨模态交互。帕累托梯度调制器(PGM)作为优化组件，引导更新沿着竞争目标间的帕累托最优方向进行，以减轻梯度冲突并提高稳定性。实验表明，CSS在IEMOCAP和MELD数据集上的准确性和训练稳定性都优于现有的代表性方法，证明了其在复杂多模态场景中的有效性。


### 论文摘要

Multimodal Emotion Recognition in Conversation (MERC) aims to predict speakers' emotions by integrating textual, acoustic, and visual cues. Existing approaches either struggle to capture complex cross-modal interactions or experience gradient conflicts and unstable training when using deeper architectures. To address these issues, we propose Cross-Space Synergy (CSS), which couples a representation component with an optimization component. Synergistic Polynomial Fusion (SPF) serves the representation role, leveraging low-rank tensor factorization to efficiently capture high-order cross-modal interactions. Pareto Gradient Modulator (PGM) serves the optimization role, steering updates along Pareto-optimal directions across competing objectives to alleviate gradient conflicts and improve stability. Experiments show that CSS outperforms existing representative methods on IEMOCAP and MELD in both accuracy and training stability, demonstrating its effectiveness in complex multimodal scenarios.

---

## 80. Procedural Mistake Detection via Action Effect Modeling

**论文链接:** [http://arxiv.org/abs/2512.03474v1](http://arxiv.org/abs/2512.03474v1)

**作者:** Wenliang Guo, Yujiang Pu, Yu Kong

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究提出了动作效果建模（AEM）框架，用于在程序性任务中检测错误，通过同时关注动作执行和结果效果来提高错误检测的可靠性。

### 背景

现有错误检测方法主要关注动作如何执行，而忽略了动作产生的结果（动作效果）。然而许多错误不是在执行过程中出现，而是在结果中显现。

### 目的

开发一个统一框架，能够同时捕捉动作执行和其结果，以实现更可靠的错误检测。

### 方法

提出动作效果建模（AEM）框架：1)通过语义相关性和视觉质量选择信息量最大的效果帧；2)从视觉基础和符号场景图中提取互补线索；3)在共享潜在空间中对齐这些线索形成效果感知表示；4)设计基于提示的检测器，结合任务特定提示，将动作段与预期执行语义对齐。

### 主要发现

在EgoPER和CaptainCook4D基准测试中，AEM在具有挑战性的单类分类（OCC）设置下取得了最先进的性能。

### 结论

同时建模动作执行和结果可以产生更可靠的错误检测，效果感知表示有利于更广泛的下游应用。

### 翻译

程序性任务中的错误检测对于构建支持学习和任务执行的智能系统至关重要。现有方法主要分析动作如何执行，而忽略了动作产生的结果，即动作效果。然而许多错误不是在执行过程中出现，而是在结果中显现，如意外的物体状态或不正确的空间排列。为解决这一差距，我们提出了动作效果建模（AEM）框架，通过概率公式同时捕捉动作执行和其结果。AEM首先基于语义相关性和视觉质量选择信息量最大的效果帧来识别动作结果。然后从视觉基础和符号场景图中提取互补线索，在共享潜在空间中对齐它们，形成强大的效果感知表示。为检测错误，我们进一步设计了一个基于提示的检测器，结合任务特定提示，并将每个动作段与其预期执行语义对齐。在具有挑战性的单类分类（OCC）设置下，我们的方法在EgoPER和CaptainCook4D基准测试中取得了最先进的性能。这些结果表明，对执行和结果进行建模可以产生更可靠的错误检测，并突显了效果感知表示有利于更广泛的下游应用的潜力。


### 论文摘要

Mistake detection in procedural tasks is essential for building intelligent systems that support learning and task execution. Existing approaches primarily analyze how an action is performed, while overlooking what it produces, i.e., the \textbf{action effect}. Yet many errors manifest not in the execution itself but in the resulting outcome, such as an unintended object state or incorrect spatial arrangement. To address this gap, we propose Action Effect Modeling (AEM), a unified framework that jointly captures action execution and its outcomes through a probabilistic formulation. AEM first identifies the outcome of an action by selecting the most informative effect frame based on semantic relevance and visual quality. It then extracts complementary cues from visual grounding and symbolic scene graphs, aligning them in a shared latent space to form robust effect-aware representations. To detect mistakes, we further design a prompt-based detector that incorporates task-specific prompts and aligns each action segment with its intended execution semantics. Our approach achieves state-of-the-art performance on the EgoPER and CaptainCook4D benchmarks under the challenging one-class classification (OCC) setting. These results demonstrate that modeling both execution and outcome yields more reliable mistake detection, and highlight the potential of effect-aware representations to benefit a broader range of downstream applications.

---

## 81. OpenTrack3D: Towards Accurate and Generalizable Open-Vocabulary 3D Instance Segmentation

**论文链接:** [http://arxiv.org/abs/2512.03532v1](http://arxiv.org/abs/2512.03532v1)

**作者:** Zhishan Zhou, Siyuan Wei, Zengran Wang, Chunjie Wang, Xiaosheng Yan, Xiao Liu

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了OpenTrack3D框架，解决了开放词汇3D实例分割在无网格环境中的泛化问题，通过视觉-空间跟踪器和多模态大语言模型提高了性能和推理能力。

### 背景

开放词汇3D实例分割(OV-3DIS)在机器人技术和AR/VR领域中至关重要，但将其推广到多样化、非结构化和无网格环境仍然是一个重大挑战。

### 目的

开发一个可泛化且准确的框架，解决现有方法在无网格场景中应用受限和文本推理能力弱的问题。

### 方法

OpenTrack3D采用新颖的视觉-空间跟踪器在线构建跨视图一致的对象提案；利用2D开放词汇分割器生成掩码并提升到3D点云；使用DINO特征图提取实例特征；融合视觉和空间线索保持实例一致性；用多模态大语言模型替换CLIP增强组合推理能力。

### 主要发现

在ScanNet200、Replica、ScanNet++和SceneFun3D等多样化基准测试上，OpenTrack3D展示了最先进的性能和强大的泛化能力。

### 结论

OpenTrack3D框架成功解决了现有方法在无网格环境中的局限性，显著提高了开放词汇3D实例分割的准确性和泛化能力。

### 翻译

将开放词汇3D实例分割(OV-3DIS)推广到多样化、非结构化和无网格环境中对机器人和AR/VR至关重要，但仍然是一个重大挑战。我们将这归因于现有方法的两个关键局限性：(1)提案生成依赖于数据集特定的提案网络或基于网格的超点，使它们在无网格场景中不适用，并限制了向新场景的泛化；(2)基于CLIP的分类器文本推理能力弱，难以识别组合性和功能性用户查询。为解决这些问题，我们引入了OpenTrack3D，这是一个可泛化且准确的框架。与依赖预生成提案的方法不同，OpenTrack3D采用新颖的视觉-空间跟踪器在线构建跨视图一致的对象提案。给定RGB-D流，我们的管道首先利用2D开放词汇分割器生成掩码，然后使用深度提升到3D点云。使用DINO特征图提取掩码引导的实例特征，我们的跟踪器融合视觉和空间线索以保持实例一致性。核心管道完全无网格，但我们还提供了可选的超点细化模块，当场景网格可用时可以进一步提高性能。最后，我们用多模态大语言模型(MLLM)替换CLIP，显著增强了复杂用户查询的组合推理能力。在ScanNet200、Replica、ScanNet++和SceneFun3D等多样化基准上的广泛实验展示了最先进的性能和强大的泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决开放词汇3D实例分割在多样化、非结构化和无网格环境中的泛化问题。现有方法有两个局限：提案生成依赖数据集特定网络或网格结构，在无网格场景中不适用；基于CLIP的分类器文本推理能力弱，难以识别组合和功能性查询。这个问题很重要，因为3D实例分割是VR/AR、机器人等应用的基础，而精确3D感知依赖昂贵的多视图标注，开放词汇能力对于处理现实世界中的多样化物体类别至关重要，且复杂查询理解能力是人机交互的关键。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于对现有方法局限性的分析进行设计：针对提案生成局限，设计了视觉-空间跟踪器在线构建跨视图一致提案；针对文本推理局限，用MLLM替代CLIP。借鉴了现有工作如使用SAM生成初始mask、DINO特征提取、多视图一致性过滤等技术，但创新性地将这些技术整合为一个无网格的框架，并引入MLLM增强文本理解能力。整体设计思路是建立一个无需训练的框架，通过融合视觉和空间线索来维护实例一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用新颖的视觉-空间跟踪器在线构建跨视图一致的物体提案，无需依赖预生成提案或网格，同时用MLLM替代CLIP增强文本理解能力。整体流程分为三阶段：1)提案生成：2D实例初始化(开放词汇检测器生成边界框，SAM生成掩码，提升到3D)和视觉-空间跟踪(融合DINO特征和3D占用线索)；2)提案细化：一致性过滤消除噪声，可选几何细化增强边界，合并重复提案；3)提案分类：选择信息量最大的视图，用MLLM进行开放词汇分类。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)新颖的视觉-空间跟踪器在线构建跨视图一致提案；2)用MLLM替代CLIP增强复杂查询理解；3)无网格核心流程提高泛化性；4)无需训练的框架避免数据集特定学习。相比之前工作，OpenTrack3D不依赖监督3D提案或网格，使用MLLM而非CLIP进行分类，在线构建提案而非依赖预生成，在多个基准测试上实现了最先进性能，特别擅长处理细粒度和功能性元素分割。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OpenTrack3D提出了一种无需训练的开放词汇3D实例分割框架，通过视觉-空间跟踪器在线构建跨视图一致提案，并利用多模态大语言模型增强复杂查询理解，在多种基准测试上实现了最先进的性能和强泛化能力。'}


### 论文摘要

Generalizing open-vocabulary 3D instance segmentation (OV-3DIS) to diverse, unstructured, and mesh-free environments is crucial for robotics and AR/VR, yet remains a significant challenge. We attribute this to two key limitations of existing methods: (1) proposal generation relies on dataset-specific proposal networks or mesh-based superpoints, rendering them inapplicable in mesh-free scenarios and limiting generalization to novel scenes; and (2) the weak textual reasoning of CLIP-based classifiers, which struggle to recognize compositional and functional user queries. To address these issues, we introduce OpenTrack3D, a generalizable and accurate framework. Unlike methods that rely on pre-generated proposals, OpenTrack3D employs a novel visual-spatial tracker to construct cross-view consistent object proposals online. Given an RGB-D stream, our pipeline first leverages a 2D open-vocabulary segmenter to generate masks, which are lifted to 3D point clouds using depth. Mask-guided instance features are then extracted using DINO feature maps, and our tracker fuses visual and spatial cues to maintain instance consistency. The core pipeline is entirely mesh-free, yet we also provide an optional superpoints refinement module to further enhance performance when scene mesh is available. Finally, we replace CLIP with a multi-modal large language model (MLLM), significantly enhancing compositional reasoning for complex user queries. Extensive experiments on diverse benchmarks, including ScanNet200, Replica, ScanNet++, and SceneFun3D, demonstrate state-of-the-art performance and strong generalization capabilities.

---

## 82. DM3D: Deformable Mamba via Offset-Guided Gaussian Sequencing for Point Cloud Understanding

**论文链接:** [http://arxiv.org/abs/2512.03424v1](http://arxiv.org/abs/2512.03424v1)

**作者:** Bin Liu, Chunyang Wang, Xuelian Liu

**发布时间:** 2025-12-03

### GPT解析

### 总结

DM3D是一种可变形的Mamba架构，用于点云理解，通过偏置引导的高斯序列化机制解决了SSM对输入顺序的依赖与点云不规则性之间的冲突。

### 背景

状态空间模型(SSMs)在长序列建模方面显示出巨大潜力，但其对输入顺序的依赖与点云的不规则性质相冲突。现有方法通常依赖于预定义的序列化策略，无法根据不同的几何结构进行调整。

### 目的

克服现有方法的限制，提出一种能够适应点云结构的序列化方法，释放SSM在点云理解方面的潜力。

### 方法

提出DM3D架构，引入偏置引导的高斯序列化机制，统一局部重采样和全局重新排序；包含Gaussian-based KNN Resampling(GKR)增强结构感知能力；Gaussian-based Differentiable Reordering(GDR)实现端到端优化的序列化顺序；以及三路径频率融合模块增强特征互补性并减少混叠。

### 主要发现

自适应序列化能够有效释放SSM在点云理解方面的潜力；DM3D在分类、少样本学习和部分分割任务上达到了最先进的性能。

### 结论

DM3D实现了点云的结构自适应序列化，证明了自适应序列化能有效解锁SSM在点云理解方面的潜力。

### 翻译

状态空间模型(SSMs)在长序列建模方面显示出巨大潜力，但它们对输入顺序的依赖与点云的不规则性质相冲突。现有方法通常依赖于预定义的序列化策略，无法根据不同的几何结构进行调整。为了克服这一限制，我们提出了DM3D，一种用于点云理解的可变形Mamba架构。具体来说，DM3D引入了偏置引导的高斯序列化机制，在可变形扫描中统一了局部重采样和全局重新排序。基于高斯的KNN重采样(GKR)通过自适应地重新组织邻近点增强了结构感知能力，而基于高斯的可重新排序(GDR)使序列化顺序能够端到端优化。此外，三路径频率融合模块增强了特征互补性并减少了混叠。这些组件共同实现了点云的结构自适应序列化。在基准数据集上的大量实验表明，DM3D在分类、少样本学习和部分分割任务上取得了最先进的性能，证明了自适应序列化有效地释放了SSM在点云理解方面的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何将状态空间模型（特别是Mamba）有效应用于点云理解的问题。点云是自动驾驶、机器人、AR等应用中的重要数据表示，但它缺乏规则结构，而现有的序列化方法使用静态预定义策略，无法适应点云的多样几何结构。这限制了模型捕捉点云中局部连续性和几何细节的能力，影响了点云理解的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到人类视觉系统自适应关注几何边缘和细节的启发，思考模型能否动态调整扫描路径以适应点云几何结构。他们借鉴了DefMamba的可变形机制思想，但将其从2D规则网格扩展到点云领域；参考了DAT的偏移预测设计；采用了PointNet等点云处理网络的基本架构；并融入了Mamba的状态空间模型框架。在此基础上，作者创新性地设计了偏移引导的高斯序列化机制，统一了局部重采样和全局重新排序。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过可学习的空间和序列偏移来引导高斯加权，实现双域变形，使模型能够动态调整扫描路径以适应点云几何结构。整体流程包括：1)输入点云进行FPS采样和KNN分组；2)提取局部特征并添加位置编码；3)沿Hilbert曲线序列化形成初始令牌序列；4)通过编码器阶段处理，每个阶段包含GFCP模块和可变形Mamba块；5)在可变形Mamba块中，通过局部上下文特征聚合获取几何上下文，预测偏移，并应用高斯KNN重采样和高斯可微分重新排序；6)通过三路径频率融合增强特征互补性；7)最终输出处理得到分类或分割结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个针对点云设计的可变形Mamba架构；2)偏移引导的高斯序列化机制，统一局部重采样和全局重新排序；3)三路径频率融合模块，在频域进行特征融合以减少混叠；4)将序列化从静态预处理转变为可学习模块。相比之前工作，不同之处在于：传统序列化方法使用静态预定义规则，而DM3D实现动态自适应序列化；相比DefMamba等2D可变形方法，DM3D扩展到点云这种不规则数据结构；相比其他Mamba变体，DM3D引入了可调整的序列化机制；相比空间域特征融合，DM3D在频域进行特征融合更有效。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DM3D通过引入偏移引导的高斯序列化机制，将点云序列化从静态预处理转变为可学习模块，使状态空间模型能够动态适应点云的几何结构，从而在点云理解任务中实现了最先进的性能。'}


### 论文摘要

State Space Models (SSMs) demonstrate significant potential for long-sequence modeling, but their reliance on input order conflicts with the irregular nature of point clouds. Existing approaches often rely on predefined serialization strategies, which cannot adjust based on diverse geometric structures. To overcome this limitation, we propose \textbf{DM3D}, a deformable Mamba architecture for point cloud understanding. Specifically, DM3D introduces an offset-guided Gaussian sequencing mechanism that unifies local resampling and global reordering within a deformable scan. The Gaussian-based KNN Resampling (GKR) enhances structural awareness by adaptively reorganizing neighboring points, while the Gaussian-based Differentiable Reordering (GDR) enables end-to-end optimization of serialization order. Furthermore, a Tri-Path Frequency Fusion module enhances feature complementarity and reduces aliasing. Together, these components enable structure-adaptive serialization of point clouds. Extensive experiments on benchmark datasets show that DM3D achieves state-of-the-art performance in classification, few-shot learning, and part segmentation, demonstrating that adaptive serialization effectively unlocks the potential of SSMs for point cloud understanding.

---

## 83. LLM-Guided Material Inference for 3D Point Clouds

**论文链接:** [http://arxiv.org/abs/2512.03237v1](http://arxiv.org/abs/2512.03237v1)

**作者:** Nafiseh Izadyar, Teseo Schneider

**发布时间:** 2025-12-02

### GPT解析

### 总结

该研究提出了一种基于大型语言模型的两阶段方法，用于从3D点云直接推断材料组成，实现了高语义和材料合理性。

### 背景

大多数现有的3D形状数据集和模型只关注几何形状，忽略了决定物体外观的材料属性。

### 目的

引入一种基于大型语言模型的方法，直接从具有粗略分割的3D点云中推断材料组成。

### 方法

将物体是什么的推理与它是由什么构成的推理解耦。第一阶段，LLM预测物体语义；第二阶段，基于推断语义为每个几何段分配合理材料。两个阶段都以零样本方式运行，无需特定任务训练。

### 主要发现

使用DeepEval中实现的LLM-as-a-Judge评估方法，在来自Fusion/ABS和ShapeNet的1000个形状上实现了高语义和材料合理性。

### 结论

语言模型可以作为通用先验，用于桥接3D数据中的几何推理和材料理解。

### 翻译

大多数现有的3D形状数据集和模型仅关注几何，忽略了决定物体外观的材料属性。我们引入了一种基于大型语言模型的两阶段方法，直接从具有粗略分割的3D点云中推断材料组成。我们的关键见解是将关于物体是什么的推理与它是由什么构成的推理解耦。在第一阶段，LLM预测物体的语义；在第二阶段，基于推断的语义为每个几何段分配合理的材料。两个阶段都以零样本方式运行，无需特定任务训练。由于现有数据集缺乏可靠的材料标注，我们使用DeepEval中实现的LLM-as-a-Judge评估该方法。在来自Fusion/ABS和ShapeNet的1000个形状上，我们的方法实现了高语义和材料合理性。这些结果表明，语言模型可以作为通用先验，用于桥接3D数据中的几何推理和材料理解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从3D点云数据中推断物体材料组成的问题。这个问题在现实中很重要，因为材料属性决定了物体的外观，对于实现逼真渲染（如机器人强化学习中的视觉真实性）至关重要；在研究中也很重要，因为现有3D数据集大多缺乏材料标注，限制了依赖材料理解的应用发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过分析直接从几何推断材料的困难性，认识到需要解耦'物体是什么'与'物体由什么材料制成'的推理过程。他们借鉴了视觉语言模型中利用LLM进行零样本识别的思想，以及部分感知3D理解的方法，但将其扩展到材料推理领域。方法设计上，作者利用LLM的常识知识，结合几何渲染和特定提示词引导，实现了两阶段的材料推断框架。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将材料推理分解为语义提取和材料分配两个阶段，利用大语言模型作为通用先验来结合几何推理。整体流程：1)输入分割后的3D点云；2)第一阶段渲染多视图图像，通过LLM预测物体语义；3)第二阶段为每个几何段选择最佳视角，结合语义信息和候选材料列表，通过LLM推断最合理的材料；4)输出每个段的材料标签。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)首个基于LLM的3D材料推断框架；2)两阶段解耦方法，分离语义理解和材料分配；3)零样本方法，无需特定训练数据；4)采用'LLM作为评判者'的评估框架。相比之前工作，不同之处在于：传统3D方法仅关注几何结构，本文扩展到材料理解；传统材料估计多基于RGB图像，本文直接从无纹理3D几何推断；现有语言驱动3D理解处理类别级语义，本文专注于物理材料推理。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了首个基于大语言模型的两阶段框架，能够直接从分割的3D点云中推断物体的材料组成，通过分离语义理解和材料推理解决了材料推断中的歧义性问题。'}


### 论文摘要

Most existing 3D shape datasets and models focus solely on geometry, overlooking the material properties that determine how objects appear. We introduce a two-stage large language model (LLM) based method for inferring material composition directly from 3D point clouds with coarse segmentations. Our key insight is to decouple reasoning about what an object is from what it is made of. In the first stage, an LLM predicts the object's semantic; in the second stage, it assigns plausible materials to each geometric segment, conditioned on the inferred semantics. Both stages operate in a zero-shot manner, without task-specific training. Because existing datasets lack reliable material annotations, we evaluate our method using an LLM-as-a-Judge implemented in DeepEval. Across 1,000 shapes from Fusion/ABS and ShapeNet, our method achieves high semantic and material plausibility. These results demonstrate that language models can serve as general-purpose priors for bridging geometric reasoning and material understanding in 3D data.

---

## 84. Reproducing and Extending RaDelft 4D Radar with Camera-Assisted Labels

**论文链接:** [http://arxiv.org/abs/2512.02394v1](http://arxiv.org/abs/2512.02394v1)

**作者:** Kejia Hu, Mohammed Alsakabi, John M. Dolan, Ozan K. Tonguz

**发布时间:** 2025-12-02

### GPT解析

### 总结

本研究提出了一种相机引导的雷达标注管道，能够生成准确的4D雷达点云标签，无需人工标注，并研究了不同雾水平对雷达标注性能的影响。

### 背景

4D雷达在恶劣条件下的环境感知方面具有潜力，但雷达语义分割的进展受限于开源数据集和标签的稀缺。RaDelft数据集虽然是开创性的，但仅提供LiDAR注释，没有公开代码生成雷达标签，限制了可重复性和下游研究。

### 目的

重现RaDelft团队的数值结果；展示相机引导雷达标注管道可生成准确的雷达点云标签；建立可重复框架供研究社区训练和评估标记的4D雷达数据；研究和量化不同雾水平对雷达标注性能的影响。

### 方法

重现RaDelft团队的数值结果；开发相机引导的雷达标注管道；将雷达点云投影到基于相机的语义分割；应用空间聚类创建标签；研究不同雾水平对雷达标注性能的影响。

### 主要发现

相机引导的雷达标注管道能够生成准确的雷达点云标签，无需人工标注；通过投影和空间聚类创建的标签显著提高了雷达标签的准确性；不同雾水平会影响雷达标注性能。

### 结论

建立了一个可重复的框架，允许研究社区训练和评估标记的4D雷达数据，促进了4D雷达语义分割领域的研究进展。

### 翻译

最近的4D雷达进展突显了其在恶劣条件下进行稳健环境感知的潜力，然而雷达语义分割的进展仍受限于开源数据集和标签的稀缺。RaDelft数据集虽然是开创性的，但仅提供LiDAR注释，没有公开代码来生成雷达标签，限制了可重复性和下游研究。在这项工作中，我们重现了RaDelft团队的数值结果，并证明了一种相机引导的雷达标注管道可以生成准确的雷达点云标签，无需依赖人工标注。通过将雷达点云投影到基于相机的语义分割并应用空间聚类，我们创建了能显著提高雷达标签准确性的标签。这些结果建立了一个可重复的框架，允许研究社区训练和评估标记的4D雷达数据。此外，我们研究了不同雾水平如何影响雷达标注性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决4D雷达语义分割数据集稀缺和可重复性问题。具体来说，RaDelft数据集只提供了LiDAR标注而没有公开代码来生成雷达标签，这限制了研究的可重复性和后续研究。这个问题在现实中很重要，因为4D雷达在恶劣天气条件下具有环境感知优势，缺乏标注好的雷达数据集限制了自动驾驶和机器人领域的发展，研究人员难以验证方法、公平比较架构或扩展管道到新环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先重现了RaDelft的数值结果，不使用原作者的代码或专有标签，仅依赖公开的雷达张量和校准元数据。他们认识到相机可以提供丰富的语义信息，而雷达提供几何信息，两者可以互补。作者借鉴了现有工作：Yan等人将语义线索从相机图像转移到LiDAR点云的方法，以及Sun等人使用类似方法标注RaDelft数据集中的LiDAR点云。作者将这些思想扩展到雷达点云，并特别关注了恶劣天气条件下的性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用相机的语义分割能力来辅助生成雷达点云的标签，通过跨模态监督提高检测概率，并在恶劣视觉条件下融合雷达和相机信息以提高鲁棒性。整体实现流程包括：1)重现RaDelft基线，将雷达帧转换为RAE体积并应用语义分割；2)语义融合，生成不同强度的雾天图像，将雷达点云转换为雷达深度图并融合相机和雷达的语义分割结果；3)相机辅助标注，通过坐标对齐、3D到2D投影、语义转移和后处理细化（使用DBSCAN聚类）来实现。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)完全重现RaDelft雷达语义分割基线并发布完整的开源工具包；2)提出相机引导的雷达标注管道，通过几何投影和空间聚类实现高质量标签；3)研究不同雾度水平对雷达标注性能的影响。相比之前的工作，本文专注于雷达点云而非LiDAR，研究了退化环境对基于相机的监督的影响，提出了完整的可重复性框架，并开发了相机-雷达融合策略，特别关注了恶劣天气条件下的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过开发一个可重复的相机引导标注框架解决了4D雷达语义分割数据集稀缺问题，该框架能生成高质量的雷达点云标签，在恶劣天气条件下表现出色，并为研究社区提供了完整的开源工具包。'}


### 论文摘要

Recent advances in 4D radar highlight its potential for robust environment perception under adverse conditions, yet progress in radar semantic segmentation remains constrained by the scarcity of open source datasets and labels. The RaDelft data set, although seminal, provides only LiDAR annotations and no public code to generate radar labels, limiting reproducibility and downstream research. In this work, we reproduce the numerical results of the RaDelft group and demonstrate that a camera-guided radar labeling pipeline can generate accurate labels for radar point clouds without relying on human annotations. By projecting radar point clouds into camera-based semantic segmentation and applying spatial clustering, we create labels that significantly enhance the accuracy of radar labels. These results establish a reproducible framework that allows the research community to train and evaluate the labeled 4D radar data. In addition, we study and quantify how different fog levels affect the radar labeling performance.

---

## 85. PointCNN++: Performant Convolution on Native Points

**论文链接:** [http://arxiv.org/abs/2511.23227v2](http://arxiv.org/abs/2511.23227v2)

**作者:** Lihan Li, Haofeng Zhong, Rui Bu, Mingchao Sun, Wenzheng Chen, Baoquan Chen, Yangyan Li

**发布时间:** 2025-11-28

### GPT解析

### 总结

PointCNN++是一种新颖的3D点云卷积学习架构，通过将稀疏卷积从体素推广到点，解决了基于点的方法和基于体素的方法之间的精度-性能权衡问题。

### 背景

现有的3D点云数据卷积学习方法分为两种范式：基于点的方法保持几何精度但面临性能挑战，基于体素的方法通过量化实现高效但以几何保真度为代价。这种精度损失对于点云配准等任务是一个关键瓶颈。

### 目的

提出PointCNN++，一种新颖的架构设计，从根本上缓解精度与性能之间的权衡问题。

### 方法

将稀疏卷积从体素推广到点；引入以原始高精度点坐标为中心的卷积；设计在点上原生运行的计算策略；将点上的卷积表述为矩阵向量乘法和归约问题；开发专用的、高度优化的GPU内核。

### 主要发现

PointCNN++比代表性基于点的方法内存使用量少一个数量级，速度快几倍；作为基于体素主干网络的简单替代，PointCNN++显著提高了点云配准精度，同时更节省内存且速度更快。

### 结论

PointCNN++表明保持几何细节和高性能不是互斥的，为高保真度和高效率的3D学习开辟了新途径。代码将开源。

### 翻译

现有的3D点云数据卷积学习方法分为两种范式：基于点的方法保持几何精度但经常面临性能挑战，基于体素的方法通过量化实现高效但以几何保真度为代价。这种精度损失对于点云配准等任务是一个关键瓶颈。我们提出了PointCNN++，一种新颖的架构设计，从根本上缓解了这种精度与性能的权衡问题。它将稀疏卷积从体素推广到点，将基于体素的卷积视为我们更通用的基于点卷积的专业化、退化特例。首先，我们引入了一个以原始高精度点坐标为中心的点中心卷积。其次，为了使这种高保真操作高效运行，我们设计了一种在点上原生运行的计算策略。我们将点上的卷积表述为矩阵向量乘法和归约问题，为此我们开发了一个专用的、高度优化的GPU内核。实验证明，PointCNN++比代表性基于点的方法内存使用量少一个数量级，速度快几倍。此外，当用作它推广的基于体素主干网络的简单替代时，PointCNN++显著提高了点云配准精度，同时证明更节省内存且速度更快。PointCNN++表明保持几何细节和实现高性能不是互斥的，为高保真度和高效率的3D学习开辟了新途径。我们的代码将开源。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云数据卷积学习中长期存在的精度与性能之间的权衡问题。现有方法分为两类：点云方法保留几何精度但性能差，体素方法效率高但会损失几何细节。这个问题在自动驾驶、机器人技术和增强现实等领域至关重要，因为这些应用既需要高精度的几何信息，也需要高效的计算性能来实时处理大量点云数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有两种点云处理范式各有妥协，然后提出不应将这种权衡视为永久问题，而是可以通过整体计算设计来缓解。他们借鉴了图像卷积中cuDNN的高效算法经验，特别是处理内存使用问题的方法；还从矩阵向量乘法的高效算法中获得灵感，将其应用到点云卷积中。作者设计了专门的GPU内核，实现了从数据表示到计算优化的系统级协同设计，使高保真点云卷积在计算上可行。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是直接在原始点云上执行卷积操作，将卷积中心放在原始高精度点坐标上，通过局部自适应体素化实现高效计算。实现流程包括：1)点中心卷积，将操作中心放在原始点坐标上；2)基于精确位置构建邻域；3)在每个邻域应用局部体素化；4)将卷积表述为矩阵向量乘法和归约(MVMR)问题；5)设计高效GPU算法优化内存访问；6)实现前向和后向传播的高效梯度计算。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)新的计算范式，直接在原始点云上执行卷积；2)点中心卷积，保留原始几何精度；3)局部自适应体素化，替代全局体素化；4)MVMR计算抽象及专门GPU内核；5)系统级协同设计。相比体素方法，PointCNN++避免了全局量化损失，解耦了核分辨率与体素分辨率；相比点云方法，它避免了不规则到规则的转换开销，减少了内存使用和计算时间，实现了既高效又高精度的点云处理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PointCNN++通过直接在原始点云上执行高效卷积，解决了3D点云处理中长期存在的几何精度与计算性能之间的权衡，实现了既保留高保真空间信息又具备高效计算性能的新一代3D学习方法。'}


### 论文摘要

Existing convolutional learning methods for 3D point cloud data are divided into two paradigms: point-based methods that preserve geometric precision but often face performance challenges, and voxel-based methods that achieve high efficiency through quantization at the cost of geometric fidelity. This loss of precision is a critical bottleneck for tasks such as point cloud registration. We propose PointCNN++, a novel architectural design that fundamentally mitigates this precision-performance trade-off. It $\textbf{generalizes sparse convolution from voxels to points}$, treating voxel-based convolution as a specialized, degraded case of our more general point-based convolution. First, we introduce a point-centric convolution where the receptive field is centered on the original, high-precision point coordinates. Second, to make this high-fidelity operation performant, we design a computational strategy that operates $\textbf{natively}$ on points. We formulate the convolution on native points as a Matrix-Vector Multiplication and Reduction (MVMR) problem, for which we develop a dedicated, highly-optimized GPU kernel. Experiments demonstrate that PointCNN++ $\textbf{uses an order of magnitude less memory and is several times faster}$ than representative point-based methods. Furthermore, when used as a simple replacement for the voxel-based backbones it generalizes, it $\textbf{significantly improves point cloud registration accuracies while proving both more memory-efficient and faster}$. PointCNN++ shows that preserving geometric detail and achieving high performance are not mutually exclusive, paving the way for a new class of 3D learning with high fidelity and efficiency. Our code will be open sourced.

---

## 86. Efficient Transferable Optimal Transport via Min-Sliced Transport Plans

**论文链接:** [http://arxiv.org/abs/2511.19741v2](http://arxiv.org/abs/2511.19741v2)

**作者:** Xinran Liu, Elaheh Akbari, Rocio Diaz Martin, Navid NaderiAlizadeh, Soheil Kolouri

**发布时间:** 2025-11-24

### GPT解析

### 总结

这篇论文研究了min-Sliced Transport Plan (min-STP)框架，探讨了优化切片器的可转移性，并提出了小批量形式以提高可扩展性。

### 背景

最优传输(OT)是计算机视觉中寻找分布间对应关系的强大框架，但计算成本限制了其可扩展性。基于切片的传输计划通过利用一维OT问题的闭式解降低计算成本，但这些方法在分布变化下优化切片器的可转移性仍是一个开放问题。

### 目的

研究min-Sliced Transport Plan框架，调查优化切片器的可转移性：即在一个分布对上训练的切片器是否能对新的、未见过的分布对产生有效的传输计划。

### 方法

理论上证明了优化切片器在数据分布轻微扰动下保持接近，从而能够在相关任务间高效转移；引入了min-STP的小批量形式，并提供了其准确性的统计保证。

### 主要发现

理论上证明了优化切片器在数据分布轻微变化下仍保持有效；实证表明可转移的min-STP在一次性匹配任务中表现良好，并促进了点云配准和基于流的生成模型的分摊训练。

### 结论

可转移的min-STP框架能够在不同分布间有效传输，提高了最优传输方法的可扩展性和实用性。

### 翻译

最优传输(OT)为寻找分布间对应关系及解决计算机视觉各领域(包括形状分析、图像生成和多模态任务)中的匹配和 alignment 问题提供了强大框架。然而，OT的计算成本阻碍了其可扩展性。基于切片的传输计划最近通过利用一维OT问题的闭式解显示出降低计算成本的潜力。这些方法优化一维投影(切片)以获得在空间中传输成本最小的条件传输计划。尽管这些方法效率高，但它们留下了一个开放问题：学习到的最优切片器是否能在分布变化下转移到新的分布对。在数据不断变化或需要在 closely 相关分布间重复进行OT计算的场景中，理解这种可转移性至关重要。在本文中，我们研究了min-Sliced Transport Plan(min-STP)框架，并调查了优化切片器的可转移性：在一个分布对上训练的切片器能否对新的、未见过的分布对产生有效的传输计划？理论上，我们证明了优化切片器在数据分布轻微扰动下保持接近，从而能够在相关任务间高效转移。为了进一步提高可扩展性，我们引入了min-STP的小批量形式，并提供了其准确性的统计保证。实证上，我们证明可转移的min-STP实现了强大的一次性匹配性能，并促进了点云配准和基于流的生成模型的分摊训练。


### 论文摘要

Optimal Transport (OT) offers a powerful framework for finding correspondences between distributions and addressing matching and alignment problems in various areas of computer vision, including shape analysis, image generation, and multimodal tasks. The computation cost of OT, however, hinders its scalability. Slice-based transport plans have recently shown promise for reducing the computational cost by leveraging the closed-form solutions of 1D OT problems. These methods optimize a one-dimensional projection (slice) to obtain a conditional transport plan that minimizes the transport cost in the ambient space. While efficient, these methods leave open the question of whether learned optimal slicers can transfer to new distribution pairs under distributional shift. Understanding this transferability is crucial in settings with evolving data or repeated OT computations across closely related distributions. In this paper, we study the min-Sliced Transport Plan (min-STP) framework and investigate the transferability of optimized slicers: can a slicer trained on one distribution pair yield effective transport plans for new, unseen pairs? Theoretically, we show that optimized slicers remain close under slight perturbations of the data distributions, enabling efficient transfer across related tasks. To further improve scalability, we introduce a minibatch formulation of min-STP and provide statistical guarantees on its accuracy. Empirically, we demonstrate that the transferable min-STP achieves strong one-shot matching performance and facilitates amortized training for point cloud alignment and flow-based generative modeling.

---

## 87. Multimodal Reinforcement Learning with Agentic Verifier for AI Agents

**论文链接:** [http://arxiv.org/abs/2512.03438v1](http://arxiv.org/abs/2512.03438v1)

**作者:** Reuben Tan, Baolin Peng, Zhengyuan Yang, Hao Cheng, Oier Mees, Theodore Zhao, Andrea Tupini, Isar Meijier, Qianhui Wu, Yuncong Yang, Lars Liden, Yu Gu, Sheng Zhang, Xiaodong Liu, Lijuan Wang, Marc Pollefeys, Yong Jae Lee, Jianfeng Gao

**发布时间:** 2025-12-03

### GPT解析

### 总结

这篇论文介绍了Argos，一个用于训练多模态推理模型的智能体奖励系统，通过从多种评分函数中选择来评估最终响应准确性、实体定位和推理过程质量，从而在多个智能体任务上取得了最先进的结果。

### 背景

当前多模态强化学习训练的智能体推理模型几乎都是基于最终答案的稀疏、结果导向的奖励进行优化的。从推理标记中计算更丰富的奖励可以提供更细粒度的指导，但存在挑战，因为不同样本需要不同评分函数，且教师模型可能提供嘈杂的奖励信号。

### 目的

开发一个更有效的奖励系统来训练多模态推理模型，解决现有方法中奖励信息不足和噪声问题，提高模型在智能体任务上的性能。

### 方法

作者提出了Argos（用于基础和客观评分的智能体奖励），这是一个奖励智能体，它从基于教师模型和基于规则的评分函数池中选择，同时评估三个方面：(i) 最终响应的准确性，(ii) 所指实体和动作的时空定位，(iii) 推理过程的质量。

### 主要发现

利用Argos进行SFT数据整理和RL训练，模型在多个智能体任务上取得了最先进的结果；仅依靠SFT后训练是不够的，因为在没有在线验证的情况下，智能体会崩溃为无基础的解决方案；Argos可以帮助减少MMRL中的奖励黑客行为；通过pareto最优的概念为Argos的有效性提供了理论依据。

### 结论

Argos是一个有效的奖励系统，能够显著提高多模态推理模型在智能体任务上的性能，解决了传统奖励方法的局限性，并提供了理论支持。

### 翻译

通过多模态强化学习(MMRL)训练的智能体推理模型能力越来越强，但它们几乎都是基于最终答案计算的稀疏、结果导向的奖励进行优化的。从推理标记中计算的更丰富的奖励可以通过提供更细粒度的指导来显著改善学习。然而，在MMRL中计算超越基于结果的更信息丰富的奖励具有挑战性，因为不同的样本可能需要不同的评分函数，教师模型也可能提供嘈杂的奖励信号。在本文中，我们介绍了Argos（用于基础和客观评分的智能体奖励），这是一个用于训练多模态推理模型的奖励智能体。对于每个样本，Argos从基于教师模型和基于规则的评分函数池中选择，同时评估：(i) 最终响应的准确性，(ii) 所指实体和动作的时空定位，以及 (iii) 推理过程的质量。我们发现，通过在SFT数据整理和RL训练中利用我们的智能体验证器，我们的模型在多个智能体任务（如空间推理、视觉幻觉以及机器人和 embodied AI 基准测试）上取得了最先进的结果。关键的是，我们证明仅依赖于在高度整理的推理数据上进行SFT后训练是不够的，因为在没有在线验证的情况下，智能体在RL期间不可避免地会崩溃为无基础的解决方案。我们还表明，我们的智能体验证器可以帮助减少MMRL中的奖励黑客行为。最后，我们通过pareto最优的概念为Argos的有效性提供了理论依据。


### 论文摘要

Agentic reasoning models trained with multimodal reinforcement learning (MMRL) have become increasingly capable, yet they are almost universally optimized using sparse, outcome-based rewards computed based on the final answers. Richer rewards computed from the reasoning tokens can improve learning significantly by providing more fine-grained guidance. However, it is challenging to compute more informative rewards in MMRL beyond those based on outcomes since different samples may require different scoring functions and teacher models may provide noisy reward signals too. In this paper, we introduce the Argos (Agentic Reward for Grounded & Objective Scoring), a principled reward agent to train multimodal reasoning models for agentic tasks. For each sample, Argos selects from a pool of teacher-model derived and rule-based scoring functions to simultaneously evaluate: (i) final response accuracy, (ii) spatiotemporal localization of referred entities and actions, and (iii) the quality of the reasoning process. We find that by leveraging our agentic verifier across both SFT data curation and RL training, our model achieves state-of-the-art results across multiple agentic tasks such as spatial reasoning, visual hallucination as well as robotics and embodied AI benchmarks. Critically, we demonstrate that just relying on SFT post-training on highly curated reasoning data is insufficient, as agents invariably collapse to ungrounded solutions during RL without our online verification. We also show that our agentic verifier can help to reduce reward-hacking in MMRL. Finally, we also provide a theoretical justification for the effectiveness of Argos through the concept of pareto-optimality.

---

## 88. ProtoEFNet: Dynamic Prototype Learning for Inherently Interpretable Ejection Fraction Estimation in Echocardiography

**论文链接:** [http://arxiv.org/abs/2512.03339v1](http://arxiv.org/abs/2512.03339v1)

**作者:** Yeganeh Ghamary, Victoria Wu, Hooman Vaseli, Christina Luong, Teresa Tsang, Siavash Bigdeli, Purang Abolmaesumi

**发布时间:** 2025-12-03

**备注:** 11 pages, Accepted in IMIMIC Workshop at MICCAI 2025

### GPT解析

### 总结

本文提出了一种名为ProtoEFNet的新型基于视频的原型学习模型，用于连续射血分数(EF)回归，实现了与不可解释模型相当的准确性，同时提供临床相关的可解释性。

### 背景

射血分数(EF)是评估心脏功能和诊断心力衰竭等疾病的关键指标。传统EF估计需要手动描画和专业知识，耗时且存在观察者间差异。目前大多数深度学习方法为黑盒模型，透明度有限，降低了临床信任度。事后解释方法虽被提出，但不能指导模型内部推理，临床应用可靠性有限。

### 目的

开发一种既准确又可解释的EF评估模型，解决黑盒模型透明度低和事后解释方法可靠性有限的问题。

### 方法

ProtoEFNet是一种基于视频的原型学习模型，学习动态时空原型捕捉临床上有意义的心脏运动模式。同时提出原型角度分离(PAS)损失函数，强制在连续EF谱上实现有区分性的表示。

### 主要发现

在EchonetDynamic数据集上的实验表明，ProtoEFNet达到与不可解释模型相当的准确性，同时提供临床相关见解。消融研究显示，提出的损失函数将F1分数从77.67±2.68提高了2个百分点至79.64±2.10。

### 结论

ProtoEFNet提供了一种既准确又可解释的EF评估方法，有助于提高临床信任度和可靠性。

### 翻译

射血分数(EF)是评估心脏功能和诊断心力衰竭等疾病的关键指标。传统EF估计需要手动描画和专业知识，使过程耗时且存在观察者间差异。目前大多数用于EF预测的深度学习方法都是黑盒模型，透明度有限，降低了临床信任度。一些事后的可解释性方法被提出用于在预测后解释决策过程，但这些解释不能指导模型的内部推理，因此在临床应用中可靠性有限。为此，我们引入ProtoEFNet，一种新颖的基于视频的原型学习模型，用于连续EF回归。该模型学习动态时空原型，捕捉临床上有意义的心脏运动模式。此外，提出的新型原型角度分离(PAS)损失函数强制在连续EF谱上实现有区分性的表示。在EchonetDynamic数据集上的实验表明，ProtoEFNet能够达到与不可解释模型相当的准确性，同时提供临床相关的见解。消融研究表明，提出的损失函数将F1分数从77.67±2.68提高了2个百分点至79.64±2.10。我们的源代码可在以下网址获取：https://github.com/DeepRCL/ProtoEF


### 论文摘要

Ejection fraction (EF) is a crucial metric for assessing cardiac function and diagnosing conditions such as heart failure. Traditionally, EF estimation requires manual tracing and domain expertise, making the process time-consuming and subject to interobserver variability. Most current deep learning methods for EF prediction are black-box models with limited transparency, which reduces clinical trust. Some post-hoc explainability methods have been proposed to interpret the decision-making process after the prediction is made. However, these explanations do not guide the model's internal reasoning and therefore offer limited reliability in clinical applications. To address this, we introduce ProtoEFNet, a novel video-based prototype learning model for continuous EF regression. The model learns dynamic spatiotemporal prototypes that capture clinically meaningful cardiac motion patterns. Additionally, the proposed Prototype Angular Separation (PAS) loss enforces discriminative representations across the continuous EF spectrum. Our experiments on the EchonetDynamic dataset show that ProtoEFNet can achieve accuracy on par with its non-interpretable counterpart while providing clinically relevant insight. The ablation study shows that the proposed loss boosts performance with a 2% increase in F1 score from 77.67$\pm$2.68 to 79.64$\pm$2.10. Our source code is available at: https://github.com/DeepRCL/ProtoEF

---

## 89. U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences

**论文链接:** [http://arxiv.org/abs/2512.02982v1](http://arxiv.org/abs/2512.02982v1)

**作者:** Xiang Xu, Ao Liang, Youquan Liu, Linfeng Li, Lingdong Kong, Ziwei Liu, Qingshan Liu

**发布时间:** 2025-12-02

**备注:** Preprint; 19 pages, 7 figures, 8 tables

### GPT解析

### 总结

本文提出了U4D框架，一种用于4D LiDAR世界建模的不确定性感知方法，通过处理空间不确定性来提高生成结果的真实性和时间稳定性。

### 背景

从激光雷达序列建模动态3D环境对于构建自动驾驶和具身AI的可靠4D世界至关重要。现有生成框架通常对所有空间区域进行均匀处理，忽略了真实场景中不同的不确定性。

### 目的

解决现有方法在复杂或模糊区域产生伪影的问题，提高生成结果的真实性和时间稳定性。

### 方法

U4D框架首先从预训练分割模型估计空间不确定性图，定位语义挑战区域；然后通过两阶段'从难到易'的生成过程：(1)不确定性区域建模，用精细几何保真度重建高熵区域；(2)不确定性条件完成，在学习的结构先验下合成剩余区域；同时引入时空混合(MoST)块确保时间一致性。

### 主要发现

大量实验表明，U4D能够产生几何上真实且时间一致的激光雷达序列，有效提高了4D世界建模的可靠性。

### 结论

U4D通过考虑空间不确定性并采用分层生成策略，显著提升了4D LiDAR世界建模的质量，为自主感知和仿真提供了更可靠的4D世界表示。

### 翻译

从激光雷达序列建模动态3D环境对于构建自动驾驶和具身AI的可靠4D世界至关重要。然而，现有的生成框架通常对所有空间区域进行均匀处理，忽略了真实场景中不同的不确定性。这种均匀生成方式在复杂或模糊区域会产生伪影，限制了真实性和时间稳定性。在本工作中，我们提出了U4D，一种用于4D LiDAR世界建模的不确定性感知框架。我们的方法首先从预训练的分割模型估计空间不确定性图，定位语义上具有挑战性的区域。然后通过两个顺序阶段进行'从难到易'的生成：(1)不确定性区域建模，用精细的几何保真度重建高熵区域；(2)不确定性条件完成，在学习的结构先验下合成剩余区域。为了进一步确保时间一致性，U4D集成了时空混合(MoST)块，在扩散过程中自适应融合空间和时间表示。大量实验表明，U4D产生了几何上真实且时间一致的激光雷达序列，提高了自主感知和仿真的4D世界建模的可靠性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有激光雷达场景生成方法对所有空间区域采用统一处理方式的问题，忽略了真实世界中不同区域存在的不确定性差异。这个问题在现实中非常重要，因为激光雷达建模是构建可靠4D世界模型的基础，对自动驾驶和具身AI至关重要。真实场景中远处物体、遮挡区域、小规模物体等高不确定性区域难以处理，影响生成场景的质量和下游感知任务的可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到真实激光雷达观测存在非均匀难度，受人类先解决模糊区域再感知全局的启发，提出显式建模不确定性的方法。他们借鉴了激光雷达场景理解中的点云转换技术、不确定性建模中的贝叶斯推理方法、扩散模型框架以及视频生成中的时空特征融合技术，将这些现有工作与新的不确定性感知理念相结合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是：1)不确定性感知，先估计空间不确定性图；2)'从难到易'生成策略，先重建高不确定性区域；3)时空一致性保证，自适应融合空间和时间特征。整体流程分三步：1)使用预训练分割模型和香农熵计算不确定性，选择高熵点形成稀疏点云；2)通过无条件扩散模型重建不确定性区域；3)以重建区域为条件，用条件扩散模型合成完整场景，同时使用MoST块确保时间一致性。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点有：1)首个不确定性感知的激光雷达生成框架；2)'从难到易'的两阶段生成范式；3)时空混合(MoST)块。相比之前工作，不同之处在于：从均匀处理转向不确定性感知的差异化处理；从一次性生成转向分阶段生成；从简单时间连接转向自适应时空特征融合；不仅关注生成质量，还注重下游任务的鲁棒性。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "本文提出了U4D，首个不确定性感知的激光雷达4D世界建模框架，通过'从难到易'的两阶段生成策略和时空混合机制，显著提升了生成场景的几何保真度、时间一致性和下游任务性能。"}


### 论文摘要

Modeling dynamic 3D environments from LiDAR sequences is central to building reliable 4D worlds for autonomous driving and embodied AI. Existing generative frameworks, however, often treat all spatial regions uniformly, overlooking the varying uncertainty across real-world scenes. This uniform generation leads to artifacts in complex or ambiguous regions, limiting realism and temporal stability. In this work, we present U4D, an uncertainty-aware framework for 4D LiDAR world modeling. Our approach first estimates spatial uncertainty maps from a pretrained segmentation model to localize semantically challenging regions. It then performs generation in a "hard-to-easy" manner through two sequential stages: (1) uncertainty-region modeling, which reconstructs high-entropy regions with fine geometric fidelity, and (2) uncertainty-conditioned completion, which synthesizes the remaining areas under learned structural priors. To further ensure temporal coherence, U4D incorporates a mixture of spatio-temporal (MoST) block that adaptively fuses spatial and temporal representations during diffusion. Extensive experiments show that U4D produces geometrically faithful and temporally consistent LiDAR sequences, advancing the reliability of 4D world modeling for autonomous perception and simulation.

---

## 90. Pruning AMR: Efficient Visualization of Implicit Neural Representations via Weight Matrix Analysis

**论文链接:** [http://arxiv.org/abs/2512.02967v1](http://arxiv.org/abs/2512.02967v1)

**作者:** Jennifer Zvonek, Andrew Gillette

**发布时间:** 2025-12-02

### GPT解析

### 总结

这篇论文提出了一种名为PruningAMR的算法，用于基于隐式神经表示(INR)的自适应网格生成，能够根据INR编码的几何特征调整分辨率，实现内存高效的变量分辨率可视化。

### 背景

隐式神经表示(INR)是一种近似时空函数的神经网络，许多内存密集型可视化任务(如现代4D CT扫描方法)将数据原生表示为INR。尽管INR比存储在晶格上的传统数据更节省内存，但许多可视化任务仍需要将其离散化为规则网格。

### 目的

开发一种能够根据INR编码的几何特征自适应调整分辨率的算法，以减少内存使用并提高可视化效率。

### 方法

提出PruningAMR算法，使用插值分解修剪方法对INR的权重矩阵进行处理，以识别几何特征，然后利用修剪后的网络引导自适应网格细化，实现针对函数底层分辨率的自适应网格生成。

### 主要发现

通过从预训练的INR开始(无需访问其训练数据)，可以生成具有显著内存节省的变量分辨率可视化。

### 结论

PruningAMR算法能够有效利用INR的特性，生成自适应分辨率的网格可视化，同时大幅减少内存需求。

### 翻译

隐式神经表示(INR)是一种近似时空函数的神经网络。许多内存密集型可视化任务，包括现代4D CT扫描方法，将数据原生表示为INR。虽然INR比存储在晶格上的传统数据更节省内存，但许多可视化任务仍需要将其离散化为规则网格。我们提出了PruningAMR，一种构建具有适应INR编码几何特征的分辨率的网格的算法。为了识别这些几何特征，我们在INR的权重矩阵上使用插值分解修剪方法。修剪后的网络用于引导自适应网格细化，实现针对函数底层分辨率的自适应网格生成。从预训练的INR开始(无需访问其训练数据)，我们生成了具有显著内存节省的变量分辨率可视化。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何高效可视化隐式神经表示（INR）的问题。INR虽然比传统网格数据存储更高效，但在可视化时通常需要离散化为规则网格，这消除了其计算优势。这个问题在现实中非常重要，因为现代4D CT扫描等技术将数据原生存储为INR格式，一个几兆字节的INR文件可能对应数TB的均匀网格数据。许多区域变化很小，而某些区域需要高分辨率供专家评估，因此需要能保留函数细粒度几何特征的自适应采样方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将传统的自适应网格细化（AMR）方法与现代神经网络压缩技术相结合。他们借鉴了两个领域的工作：1) 自适应网格细化技术，用于根据几何特征细化网格元素；2) 神经网络剪枝技术，特别是Chee等人提出的基于插值分解的剪枝方法。作者的创新在于将这两种技术结合，使用神经网络权重矩阵分析来确定哪些区域需要更高分辨率，从而指导自适应网格细化，无需访问INR的训练数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过分析INR权重矩阵的复杂性来确定哪些区域包含更复杂的几何特征，从而指导自适应网格细化，使网格分辨率与函数的底层分辨率相匹配。整体流程是：1) 从粗略均匀网格开始；2) 对每个元素，使用插值分解剪枝方法分析INR在该元素上的表现；3) 记录剪枝后神经元比例(p)和相对误差(error)；4) 如果p和error都小于阈值，则认为元素已充分细化；5) 否则标记需要细化；6) 对标记元素进行几何细化；7) 重复直到达到停止条件；8) 输出自适应网格及其顶点处的INR值供可视化使用。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出衡量INR编码几何特征复杂度的通用方法；2) 设计基于权重矩阵分析的自适应网格细化算法，无需训练数据；3) 在2D、3D和4D INR上证明概念并量化内存节省；4) 应用于物理信息神经网络(PINN)；5) 应用于最先进CT扫描仪的4D数据。相比之前工作，这篇论文不专注于训练INR，而是处理预训练的INR；不假设数据已以自适应网格格式提供；不依赖底层微分方程，区别于传统基于PDE的AMR技术。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了PruningAMR算法，通过分析神经网络权重矩阵指导自适应网格细化，实现了对预训练隐式神经表示的高效可视化，在保持类似精度的同时显著减少了内存使用。'}


### 论文摘要

An implicit neural representation (INR) is a neural network that approximates a spatiotemporal function. Many memory-intensive visualization tasks, including modern 4D CT scanning methods, represent data natively as INRs. While INRs are prized for being more memory-efficient than traditional data stored on a lattice, many visualization tasks still require discretization to a regular grid. We present PruningAMR, an algorithm that builds a mesh with resolution adapted to geometric features encoded by the INR. To identify these geometric features, we use an interpolative decomposition pruning method on the weight matrices of the INR. The resulting pruned network is used to guide adaptive mesh refinement, enabling automatic mesh generation tailored to the underlying resolution of the function. Starting from a pre-trained INR--without access to its training data--we produce a variable resolution visualization with substantial memory savings.

---

## 91. A Discrete Neural Operator with Adaptive Sampling for Surrogate Modeling of Parametric Transient Darcy Flows in Porous Media

**论文链接:** [http://arxiv.org/abs/2512.03113v1](http://arxiv.org/abs/2512.03113v1)

**作者:** Zhenglong Chen, Zhao Zhang, Xia Yan, Jiayu Zhai, Piyang Liu, Kai Zhang

**发布时间:** 2025-12-02

### GPT解析

### 总结

本研究提出了一种新的离散神经算子，用于模拟具有随机参数的非均质多孔介质中的瞬态达西流场，该方法结合时间编码、算子学习和UNet，实现了比现有方法更高的预测精度。

### 背景

非均质多孔介质中的瞬态达西流场模拟在石油工程和地下水流动等领域具有重要应用，但由于参数随机性和介质非均质性，传统模拟方法面临计算成本高和精度有限的问题。

### 目的

开发一种新的离散神经算子作为替代模型，用于高效准确地预测具有随机参数的非均质多孔介质中的瞬态达西流场。

### 方法

提出整合时间编码、算子学习和UNet的新离散神经算子；采用源自有限体积法的传导率矩阵而非渗透率作为输入；开发基于生成潜在空间的自适应采样方法，使用高斯混合模型估计泛化误差；在2D/3D单相和两相达西流场预测测试案例中进行验证。

### 主要发现

新离散神经算子比现有的注意力残差UNet结构具有更高的预测精度；使用传导率矩阵作为输入可进一步提高预测精度；开发的自适应采样方法提高了采样效率；在有限的训练集条件下，预测准确性得到一致提升。

### 结论

该研究成功开发了一种新的离散神经算子，能够高效准确地模拟具有随机参数的非均质多孔介质中的瞬态达西流场，在有限训练数据条件下实现了比现有方法更高的预测精度。

### 翻译

本研究提出了一种新的离散神经算子，用于具有随机参数的非均质多孔介质中瞬态达西流场的替代建模。新方法整合了时间编码、算子学习和UNet，以近似随机参数向量空间与时空流场向量空间之间的映射关系。新的离散神经算子比最先进的注意力残差UNet结构能够实现更高的预测精度。源自有限体积法，传导率矩阵而非渗透率被用作替代模型的输入，以进一步提高预测精度。为提高采样效率，开发了一种生成潜在空间自适应采样方法，采用高斯混合模型进行泛化误差的密度估计。在2D/3D单相和两相达西流场预测的测试案例中进行了验证。结果表明，在有限的训练集条件下，预测准确性得到了一致提升。


### 论文摘要

This study proposes a new discrete neural operator for surrogate modeling of transient Darcy flow fields in heterogeneous porous media with random parameters. The new method integrates temporal encoding, operator learning and UNet to approximate the mapping between vector spaces of random parameter and spatiotemporal flow fields. The new discrete neural operator can achieve higher prediction accuracy than the SOTA attention-residual-UNet structure. Derived from the finite volume method, the transmissibility matrices rather than permeability is adopted as the inputs of surrogates to enhance the prediction accuracy further. To increase sampling efficiency, a generative latent space adaptive sampling method is developed employing the Gaussian mixture model for density estimation of generalization error. Validation is conducted on test cases of 2D/3D single- and two-phase Darcy flow field prediction. Results reveal consistent enhancement in prediction accuracy given limited training set.

---

## 92. Spatiotemporal Pyramid Flow Matching for Climate Emulation

**论文链接:** [http://arxiv.org/abs/2512.02268v1](http://arxiv.org/abs/2512.02268v1)

**作者:** Jeremy Andrew Irvin, Jiaqi Han, Zikui Wang, Abdulaziz Alharbi, Yufei Zhao, Nomin-Erdene Bayarsaikhan, Daniele Visioni, Andrew Y. Ng, Duncan Watson-Parris

**发布时间:** 2025-12-01

### GPT解析

### 总结

论文介绍了一种名为时空金字塔流(SPF)的新型生成模型，用于高效模拟地球气候变化。该模型通过时空金字塔结构在不同时空尺度上分层建模数据，实现了并行气候模拟，并在多个时间尺度上表现出色。

### 背景

现有的生成式方法主要依赖天气尺度的自回归进行气候模拟，但对于长期气候预测存在速度慢的问题，并且在非平稳强迫条件下尚未展示出稳定的运行能力。

### 目的

开发一种新的生成模型方法，能够高效、准确地在多个时间尺度上进行气候模拟，并能够处理非平稳强迫条件下的气候预测。

### 方法

作者提出了时空金字塔流(SPF)，一种新的流匹配方法，通过时空金字塔结构对数据进行分层建模。该方法将生成轨迹划分为时空金字塔，逐步提高空间分辨率以减少计算量，并将每个阶段与相关时间尺度关联，允许在金字塔的任何时间级别直接采样。此外，每个阶段都基于预设的物理强迫条件（如温室气体或气溶胶）进行条件化，实现了在多个时间尺度上的高效并行气候模拟。

### 主要发现

在ClimateBench基准测试中，SPF在年度和月度时间尺度上优于强大的流匹配基线模型和预训练模型，并提供快速采样，特别是在较粗的时间级别上。作者还创建了ClimateSuite，这是迄今为止最大的地球系统模拟集合，包含超过33,000个模拟年，涵盖十个气候模型，并且是首个包含气候干预模拟的数据集。研究发现，扩展后的SPF模型在跨气候模型的保留场景上表现出良好的泛化能力。

### 结论

SPF和ClimateSuite为在不同时间尺度和真实未来场景下进行准确、高效、概率性的气候模拟提供了基础。数据和代码已在GitHub上公开。

### 翻译

生成式模型有可能改变我们模拟地球气候变化的方式。先前的生成式方法依赖于天气尺度的自回归进行气候模拟，但对于长期气候预测本质上很慢，并且在非平稳强迫条件下尚未展示出稳定的运行能力。在此，我们引入了时空金字塔流(SPF)，这是一类新的流匹配方法，在时空尺度上分层建模数据。受级联视频模型的启发，SPF将生成轨迹划分为时空金字塔，逐步提高空间分辨率以减少计算量，并将每个阶段与相关时间尺度关联，以允许在金字塔的任何时间级别直接采样。这种设计，结合将每个阶段基于预设的物理强迫条件（如温室气体或气溶胶）进行条件化，使得能够在多个时间尺度上进行高效、并行的气候模拟。在ClimateBench上，SPF在年度和月度时间尺度上优于强大的流匹配基线模型和预训练模型，同时提供快速采样，特别是在较粗的时间级别上。为了扩展SPF，我们创建了ClimateSuite，这是迄今为止最大的地球系统模拟集合，包含超过33,000个模拟年，涵盖十个气候模型，也是首个包含气候干预模拟的数据集。我们发现，扩展后的SPF模型在跨气候模型的保留场景上表现出良好的泛化能力。SPF和ClimateSuite共同为在不同时间尺度和真实未来场景下进行准确、高效、概率性的气候模拟提供了基础。数据和代码可在https://github.com/stanfordmlgroup/spf 公开获取。


### 论文摘要

Generative models have the potential to transform the way we emulate Earth's changing climate. Previous generative approaches rely on weather-scale autoregression for climate emulation, but this is inherently slow for long climate horizons and has yet to demonstrate stable rollouts under nonstationary forcings. Here, we introduce Spatiotemporal Pyramid Flows (SPF), a new class of flow matching approaches that model data hierarchically across spatial and temporal scales. Inspired by cascaded video models, SPF partitions the generative trajectory into a spatiotemporal pyramid, progressively increasing spatial resolution to reduce computation and coupling each stage with an associated timescale to enable direct sampling at any temporal level in the pyramid. This design, together with conditioning each stage on prescribed physical forcings (e.g., greenhouse gases or aerosols), enables efficient, parallel climate emulation at multiple timescales. On ClimateBench, SPF outperforms strong flow matching baselines and pre-trained models at yearly and monthly timescales while offering fast sampling, especially at coarser temporal levels. To scale SPF, we curate ClimateSuite, the largest collection of Earth system simulations to date, comprising over 33,000 simulation-years across ten climate models and the first dataset to include simulations of climate interventions. We find that the scaled SPF model demonstrates good generalization to held-out scenarios across climate models. Together, SPF and ClimateSuite provide a foundation for accurate, efficient, probabilistic climate emulation across temporal scales and realistic future scenarios. Data and code is publicly available at https://github.com/stanfordmlgroup/spf .

---

## 93. 论文ID: 2512.03598v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03598v1.json'

---

## 94. 论文ID: 2512.03010v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03010v1.json'

---

## 95. Learning Steerable Clarification Policies with Collaborative Self-play

**论文链接:** [http://arxiv.org/abs/2512.04068v1](http://arxiv.org/abs/2512.04068v1)

**作者:** Jonathan Berant, Maximillian Chen, Adam Fisch, Reza Aghajani, Fantine Huot, Mirella Lapata, Jacob Eisenstein

**发布时间:** 2025-12-03

### GPT解析

### 总结

该论文提出了一种通过自我对弈训练AI助手处理模糊查询的可调节策略，使AI能够根据不同情境决定如何响应不确定的查询。

### 背景

AI助手需要处理模糊或不确定的查询，需要一种管理不确定性的策略来决定何时猜测用户意图直接回答，何时列举多种可能意图，何时提出澄清问题。这些策略依赖于上下文因素，如用户偏好或模态。例如，在小屏幕或语音环境中列举多种可能的用户意图会很麻烦。

### 目的

训练一种可调节的策略，使AI助手能够根据不同情境（如成本因素）灵活决定如何响应模糊查询，提高响应的准确性和效率。

### 方法

使用自我对弈训练可调节策略，涉及两个代理（模拟用户和AI助手）。模型将每个澄清问题的数值成本和每个生成单词的成本作为输入，采取能够最大化成本惩罚准确性的行动。使用强化自训练（ReST）来训练模型以获得高奖励。

### 主要发现

通过ReST训练的模型实现了可调节的策略，该策略根据提供的成本条件性地改变其行为，从而获得更高的奖励和准确性。该方法还可以推广到训练时未观察到的数值成本值。

### 结论

所提出的方法能够训练出适应不同情境的可调节策略，使AI助手能够根据成本因素灵活地决定如何响应模糊查询，提高响应的准确性和效率。

### 翻译

为了处理模糊或不确定的查询，AI助手需要一种管理不确定性的策略，以决定(a)何时猜测用户意图并直接回答，(b)何时列举并回答多种可能的意图，以及(c)何时提出澄清问题。然而，这样的策略依赖于上下文因素，如用户偏好或模态。例如，在小屏幕或语音环境中列举多种可能的用户意图会很麻烦。在这项工作中，我们提出使用自我对弈训练管理这种不确定性的可调节策略。给定两个代理，一个模拟用户，另一个模拟AI助手，我们生成用户发出可能模糊的查询，而AI助手需要决定如何响应的对话。重要的是，模型将每个澄清问题的数值成本和每个生成单词的成本作为输入，并被要求采取能够最大化其最终奖励的行动，即成本惩罚的准确性。我们使用强化自训练（ReST）来训练我们的模型以获得高奖励，并表明这会导致一个可调节的策略，该策略根据提供的成本条件性地改变其行为，从而获得更高的奖励和准确性。此外，我们的程序也可以推广到训练时未观察到的数值成本值。


### 论文摘要

To handle underspecified or ambiguous queries, AI assistants need a policy for managing their uncertainty to determine (a) when to guess the user intent and answer directly, (b) when to enumerate and answer multiple possible intents, and (c) when to ask a clarifying question. However, such policies are contextually dependent on factors such as user preferences or modality. For example, enumerating multiple possible user intentions is cumbersome on small screens or in a voice setting. In this work, we propose to train steerable policies for managing this uncertainty using self-play. Given two agents, one simulating a user and the other an AI assistant, we generate conversations where the user issues a potentially ambiguous query, and the assistant needs to determine how to respond. Importantly, the model takes as input the numerical cost of each clarification question, and each generated word, and is asked to take the action that will maximize its final reward, which is the cost-penalized accuracy. We use Reinforced Self-Training (ReST) to train our model to achieve high reward and show this leads to a steerable policy that changes its behavior predictably conditioned on the provided costs, leading to higher reward and accuracy. Moreover, our procedure also generalizes to numerical cost values that were unobserved at training time.

---

## 96. Fully quantum theory of strong-field driven tunable entangled multi-photon states in HHG

**论文链接:** [http://arxiv.org/abs/2512.03987v1](http://arxiv.org/abs/2512.03987v1)

**作者:** Sebastián de-la-Peña, Heiko Appel, Angel Rubio, Ofer Neufeld

**发布时间:** 2025-12-03

**备注:** 8 pages, 4 figures

### GPT解析

### 总结

本文开发了一个用于高次谐波生成中纠缠度量的完整量子理论，精确求解了光-物质相互作用哈密顿量，成功解释了实验观察到的纠缠特性随激光功率变化的规律。

### 背景

量子高次谐波生成是一个提供高光子数纠缠光态的研究领域，但如何正确描述HHG发射的量子方面（如压缩或纠缠）存在开放性争论。先前的方法使用了非相互作用的经典轨迹系综或微扰理论，遗漏了关键的纠缠特性。

### 目的

开发一个完整的量子理论来描述高次谐波生成中的纠缠特性，并解释实验观察到的现象。

### 方法

精确求解光-物质相互作用哈密顿量，评估不同谐波之间发射光子的纠缠，并考虑焦距平均效应对纠缠度量的影响。

### 主要发现

1) 对于阈值以下谐波，R纠缠参数随激光功率增加而减小；2) 微调激光功率可以增强HHG纠缠特性，这些特性随驱动功率振荡并表现出局域非经典极大值结构；3) 阈值以上谐波的纠缠也表现出类似的振荡行为；4) 驱动电子轨迹的长程行为可以定性改变纠缠特性；5) 对经典变量的焦距平均在纠缠度量中起关键作用，可以改变可观测量的定性行为。

### 结论

该工作建立了探索HHG中纠缠特性的最先进方法，为在XUV和超快领域分析和工程'真正量子'的多光子态铺平了道路，可用于更复杂的物质系统。

### 翻译

量子高次谐波生成是一个不断发展的研究领域，能够提供高光子数的纠缠光态。然而，关于正确描述HHG发射的量子方面（如压缩或纠缠）所需的理论水平存在开放性争论。先前的方法采用了非相互作用的经典轨迹系综，或以经典轨迹为起点使用微扰理论，遗漏了关键的纠缠特性。在本信中，我们开发了一个用于HHG中纠缠度量的完整量子理论，精确求解了光-物质相互作用哈密顿量，并利用它来评估不同谐波之间发射光子的纠缠。我们首次使理论与最近的实验达到定性一致，表明对于阈值以下谐波，R纠缠参数随激光功率增加而减小。我们的结果表明，微调激光功率可以增强HHG纠缠特性，这些特性随驱动功率振荡并表现出局域非经典极大值结构。同样，我们的理论预测，对于阈值以下谐波观察到的纠缠振荡行为也出现在涉及阈值以上谐波的纠缠中。我们还表明，驱动电子轨迹的长程行为可以定性改变 resulting entanglement。最后，我们表明，对经典变量的焦距平均（迄今为止在量子HHG理论中被忽略）在纠缠度量中起关键作用，可以改变可观测量的定性行为。我们的工作建立了探索HHG中纠缠特性的最先进方法，并为在XUV和超快领域分析和工程'真正量子'的多光子态铺平了道路，可用于更复杂的物质系统。


### 论文摘要

Quantum high-harmonic generation (HHG) is a growing field of research with capabilities of providing high photon-number entangled states of light. However, there is an open debate regarding the theory level required for correctly describing the quantum aspects of HHG emission, such as squeezing or entanglement. Previous approaches have employed non-interacting classical ensembles of trajectories, or perturbation theory utilizing the classical trajectories as a starting point, missing out key entanglement features. In this Letter, we develop a full quantum theory for entanglement measures in HHG solving exactly the light-matter interaction Hamiltonian and employ it for evaluating the entanglement between emitted photons of different harmonics. For the first time, we reach qualitative agreement of theory with recent experiments showing that the R entanglement parameter decreases with increasing laser power for below-threshold harmonics. Our results indicate that fine-tuning the laser power could enhance HHG entanglement features, which are observed to oscillate with the driving power and exhibit local non-classical maxima structures. Similarly, our theory predicts that the oscillatory behavior of entanglement observed for below-threshold harmonics also appears for entanglement involving above-threshold harmonics. We also show that the long-range behavior of driven electronic trajectories can qualitatively change the resulting entanglement. Lastly, we show that focal averaging over classical degrees of freedom, which has thus far been ignored in quantum HHG theories, plays a key role in entanglement measures and can change the qualitative behavior of observables. Our work establishes the state-of-the art in exploring entanglement features in HHG, and paves way for analysis and engineering of 'truly-quantum' multi-photon states in the XUV and ultrafast regime for more complex matter systems.

---

## 97. OOPredictor: Predicting Object-Oriented Accesses using Static Analysis

**论文链接:** [http://arxiv.org/abs/2512.03972v1](http://arxiv.org/abs/2512.03972v1)

**作者:** Hassan Arafat, David Bremner, Kenneth B. Kent, Julian Wang

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种使用编译时静态分析预测程序运行时访问模式的方法，以解决面向对象编程中指针追逐导致的缓存性能问题。

### 背景

面向对象编程因其关注点分离和设计适应性已成为主导设计范式，降低了开发和维护成本。然而，这种设计的间接性导致过多的指针追逐，影响局部性，降低缓存性能。现代硬件预取器难以处理这种不可预测的访问模式。

### 目的

开发一种方法来预测程序运行时的访问模式，以缓解指针追逐导致的缓存性能问题，而无需运行时分析带来的开销。

### 方法

使用编译时静态分析预测程序运行时最常见的访问模式。在OpenJ9 JVM的OMR优化器基础设施中实现了原型，输出马尔可夫链来建模程序的预期行为。

### 主要发现

实验表明，所提出的预测器具有良好的准确性，可以用于指导最小侵入性的负载停滞缓解策略，例如指导复制GC采用更友好的局部性复制顺序。

### 结论

编译时静态分析可以有效预测面向对象程序中的访问模式，为优化缓存性能提供了一种无需运行时开销的方法。

### 翻译

面向对象编程已成为一种主导设计范式，因为其关注点分离和设计适应性降低了开发和维护成本。然而，这种便利并非没有代价。此类设计中固有的间接性导致过多的指针追逐，负面影响局部性，进而降低缓存结构的性能。此外，现代硬件预取器主要是步长预取器，难以处理指针追逐产生的不可预测的访问模式。大多数寻求解决此问题的软件方法依赖于分析运行中的程序，这带来了显著的运行时开销或需要之前运行的数据。在本文中，我们提出使用编译时静态分析来预测程序运行时显示的最常见访问模式。由于Java是最流行的面向对象语言之一，我们在OpenJ9 JVM内的OMR优化器基础设施中实现了我们的原型。我们提出的预测器的输出是马尔可夫链，用于建模程序的预期行为。通过与使用仪器化解释器测量的程序实际运行时行为进行比较，评估了所提出预测器的有效性。我们的实验表明，所提出的预测器具有良好的准确性，可用于通知最小侵入性的负载停滞缓解策略，例如指导复制GC采用更友好的局部性复制顺序。


### 论文摘要

Object-oriented Programming has become one of the most dominant design paradigms as the separation of concerns and adaptability of design reduce development and maintenance costs. However, the convenience is not without cost. The added indirection inherent in such designs causes excessive pointer chasing, negatively affecting locality, which in turn degrades the performance of cache structures. Furthermore, modern hardware prefetchers are mostly stride prefetchers that are ill-equipped to handle the unpredictability of access patterns generated by pointer chasing. Most software approaches that seek to address this problem resort to profiling the program as it runs, which comes with a significant run-time overhead or requires data from previous runs. In this paper, we propose the use of compile-time static analysis to predict the most common access patterns displayed by a program during run time. Since Java is one of the most popular object-oriented languages, we implement our prototype within the OpenJ9 JVM, inside the OMR optimizer infrastructure. The outputs of our proposed predictor are Markov chains that model the expected behavior of the program. The effectiveness of the proposed predictor is evaluated by comparing the model with the actual run-time behavior of the program measured using an instrumented interpreter. Our experiments show that the proposed predictor exhibits good accuracy and can be used to inform minimally intrusive load stall mitigation strategies, e.g. informing copying GCs on more locality-friendly copying orders

---

## 98. Driving is a Game: Combining Planning and Prediction with Bayesian Iterative Best Response

**论文链接:** [http://arxiv.org/abs/2512.03936v1](http://arxiv.org/abs/2512.03936v1)

**作者:** Aron Distelzweig, Yiwei Wang, Faris Janjoš, Marcel Hallgarten, Mihai Dobre, Alexander Langmann, Joschka Boedecker, Johannes Betz

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为贝叶斯迭代最佳响应(BIBeR)的框架，将运动预测和博弈论规划统一为单一交互感知过程，实现了自动驾驶在密集城市交通环境中的性能提升。

### 背景

自动驾驶规划系统在常规场景下表现良好，但在密集城市交通中仍面临挑战。现代运动预测器虽提供准确预测，但与规划的集成基础；端到端模型提供单向集成，避免了联合预测和规划建模的挑战；博弈论公式虽为有原则的替代方案，但在自动驾驶中采用有限。

### 目的

提出一种将运动预测和博弈论规划统一为单一交互感知过程的方法，解决自动驾驶在密集城市交通中的规划问题。

### 方法

介绍贝叶斯迭代最佳响应(BIBeR)框架，将最先进的预测器集成到迭代最佳响应循环中，通过贝叶斯置信估计量化预测可靠性并调节更新强度，实现低置信度下更保守、高置信度下更果断的决策。

### 主要发现

BIBeR实现了双向适应，使自我车辆既能对他人行为做出反应，又能塑造他人行为。实验显示，在高度交互的interPlan车道变更场景中，BIBeR比最先进的规划器提高了11%的性能，在标准的nuPlan基准测试中也优于现有方法。

### 结论

BIBeR与现代预测器和规划器兼容，结合了结构化规划的透明性和学习模型的灵活性，有效提升了自动驾驶在密集城市交通环境中的性能。

### 翻译

自动驾驶规划系统在常规场景中使用轻量级基于规则的方法表现近乎完美，但在密集城市交通中仍然面临挑战，车道变更和汇入需要预测和影响其他智能体。现代运动预测器提供高度准确的预测，但它们与规划的集成大多基础：仅丢弃不安全的计划。同样，端到端模型提供单向集成，避免了在不确定性下进行联合预测和规划建模的挑战。相比之下，博弈论公式提供了一个有原则的替代方案，但在自动驾驶中采用有限。我们提出贝叶斯迭代最佳响应(BIBeR)，一个将运动预测和博弈论规划统一为单一交互感知过程的框架。BIBeR首次将最先进的预测器集成到迭代最佳响应(IBR)循环中，不断优化自我车辆和周围智能体的策略。这种重复的最佳响应过程近似纳什均衡，实现双向适应，使自我车辆既能对他人行为做出反应，又能塑造他人行为。此外，我们提出的贝叶斯置信估计量化预测可靠性并调节更新强度，在低置信度下更保守，在高置信度下更果断。BIBeR与现代预测器和规划器兼容，结合了结构化规划的透明性和学习模型的灵活性。实验表明，BIBeR在高度交互的interPlan车道变更场景中比最先进的规划器提高了11%的性能，同时在标准的nuPlan基准测试上也优于现有方法。


### 论文摘要

Autonomous driving planning systems perform nearly perfectly in routine scenarios using lightweight, rule-based methods but still struggle in dense urban traffic, where lane changes and merges require anticipating and influencing other agents. Modern motion predictors offer highly accurate forecasts, yet their integration into planning is mostly rudimental: discarding unsafe plans. Similarly, end-to-end models offer a one-way integration that avoids the challenges of joint prediction and planning modeling under uncertainty. In contrast, game-theoretic formulations offer a principled alternative but have seen limited adoption in autonomous driving. We present Bayesian Iterative Best Response (BIBeR), a framework that unifies motion prediction and game-theoretic planning into a single interaction-aware process. BIBeR is the first to integrate a state-of-the-art predictor into an Iterative Best Response (IBR) loop, repeatedly refining the strategies of the ego vehicle and surrounding agents. This repeated best-response process approximates a Nash equilibrium, enabling bidirectional adaptation where the ego both reacts to and shapes the behavior of others. In addition, our proposed Bayesian confidence estimation quantifies prediction reliability and modulates update strength, more conservative under low confidence and more decisive under high confidence. BIBeR is compatible with modern predictors and planners, combining the transparency of structured planning with the flexibility of learned models. Experiments show that BIBeR achieves an 11% improvement over state-of-the-art planners on highly interactive interPlan lane-change scenarios, while also outperforming existing approaches on standard nuPlan benchmarks.

---

## 99. 论文ID: 2512.03806v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03806v1.json'

---

## 100. Safety Reinforced Model Predictive Control (SRMPC): Improving MPC with Reinforcement Learning for Motion Planning in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.03774v1](http://arxiv.org/abs/2512.03774v1)

**作者:** Johannes Fischer, Marlon Steiner, Ömer Sahin Tas, Christoph Stiller

**发布时间:** 2025-12-03

**DOI:** 10.1109/ITSC57777.2023.10422605

### GPT解析

### 总结

该研究提出了一种结合模型预测控制(MPC)和安全强化学习(SRL)的新方法，用于自动驾驶中的运动规划，以提高安全性和性能。

### 背景

模型预测控制(MPC)被广泛用于自动驾驶的运动规划，但实时规划器需要利用最优控制问题(OCPs)的凸近似，这种近似将解限制在子空间中，可能不包含全局最优解。

### 目的

通过结合安全强化学习(SRL)与MPC，使规划器能够探索超出先前解邻近区域的解空间，从而找到全局最优解，同时确保自动驾驶的安全性。

### 方法

使用约束强化学习(CRL)确保自动驾驶安全，采用基于手工制作能量函数的安全指标作为约束目标来建模安全和不安全区域，并通过学习状态依赖的拉格朗日乘子与安全策略同时来解决CRL问题。

### 主要发现

在高速公路场景中的实验表明，所提出的方法在安全性和性能指标上都优于单独使用MPC或SRL的方法。

### 结论

结合MPC和SRL的方法能够有效提高自动驾驶系统的安全性和整体性能，为自动驾驶规划提供了新的解决方案。

### 翻译

模型预测控制(MPC)被广泛用于运动规划，特别是在自动驾驶领域。规划器的实时性要求利用最优控制问题(OCPs)的凸近似。然而，这种近似将解限制在子空间中，可能不包含全局最优解。为此，我们提出使用安全强化学习(SRL)在MPC框架内获得新的安全参考轨迹。通过采用基于学习的方法，MPC可以探索超出先前解邻近区域的解，从而可能找到全局最优解。我们整合了约束强化学习(CRL)以确保自动驾驶的安全性，使用手工制作的基于能量函数的安全指标作为约束目标来建模安全和不安全区域。我们的方法利用状态依赖的拉格朗日乘子，与安全策略同时学习，以解决CRL问题。通过在高速公路场景中的实验，我们证明了在安全性和性能指标方面，我们的方法优于MPC和SRL。


### 论文摘要

Model predictive control (MPC) is widely used for motion planning, particularly in autonomous driving. Real-time capability of the planner requires utilizing convex approximation of optimal control problems (OCPs) for the planner. However, such approximations confine the solution to a subspace, which might not contain the global optimum. To address this, we propose using safe reinforcement learning (SRL) to obtain a new and safe reference trajectory within MPC. By employing a learning-based approach, the MPC can explore solutions beyond the close neighborhood of the previous one, potentially finding global optima. We incorporate constrained reinforcement learning (CRL) to ensure safety in automated driving, using a handcrafted energy function-based safety index as the constraint objective to model safe and unsafe regions. Our approach utilizes a state-dependent Lagrangian multiplier, learned concurrently with the safe policy, to solve the CRL problem. Through experimentation in a highway scenario, we demonstrate the superiority of our approach over both MPC and SRL in terms of safety and performance measures.

---

## 101. Prediction-Driven Motion Planning: Route Integration Strategies in Attention-Based Prediction Models

**论文链接:** [http://arxiv.org/abs/2512.03756v1](http://arxiv.org/abs/2512.03756v1)

**作者:** Marlon Steiner, Royden Wagner, Ömer Sahin Tas, Christoph Stiller

**发布时间:** 2025-12-03

**备注:** In Proceedings of the IEEE International Conference on Intelligent Transportation Systems (ITSC), Gold Coast, AUSTRALIA, 18-21 November 2025

### GPT解析

### 总结

本文研究了将导航信息整合到基于注意力的运动预测模型中，以弥合多智能体运动预测与基于目标的运动规划之间的差距，从而提升自动驾驶车辆与其他交通参与者的交互能力。

### 背景

结合运动预测和运动规划为增强自动驾驶车辆与其他交通参与者之间的互动提供了有前景的框架，但存在根据导航目标进行预测条件设定以及确保稳定、运动学可行轨迹的挑战。

### 目的

解决将导航信息融入预测模型的问题，弥合多智能体运动预测和基于目标的运动规划之间的差距。

### 方法

提出并评估了几种架构导航集成策略，并在nuPlan数据集上进行了测试。

### 主要发现

预测驱动的运动规划具有潜力，导航信息可以同时增强预测和规划任务的效果。

### 结论

通过将导航信息整合到运动预测模型中，可以有效连接预测和规划两个任务，提升自动驾驶系统的整体性能。

### 翻译

结合运动预测和运动规划为增强自动驾驶车辆与其他交通参与者之间的互动提供了有前景的框架。然而，这带来了根据导航目标进行预测条件设定以及确保稳定、运动学可行轨迹的挑战。针对前一个挑战，本文研究了将导航信息扩展到基于注意力的运动预测模型中。通过将自车的预定路线和目标姿态整合到模型架构中，我们弥合了多智能体运动预测和基于目标的运动规划之间的差距。我们在nuPlan数据集上提出并评估了几种架构导航集成策略。我们的结果展示了预测驱动的运动规划的潜力，突显了导航信息如何增强预测和规划任务。我们的实现位于：https://github.com/KIT-MRT/future-motion。


### 论文摘要

Combining motion prediction and motion planning offers a promising framework for enhancing interactions between automated vehicles and other traffic participants. However, this introduces challenges in conditioning predictions on navigation goals and ensuring stable, kinematically feasible trajectories. Addressing the former challenge, this paper investigates the extension of attention-based motion prediction models with navigation information. By integrating the ego vehicle's intended route and goal pose into the model architecture, we bridge the gap between multi-agent motion prediction and goal-based motion planning. We propose and evaluate several architectural navigation integration strategies to our model on the nuPlan dataset. Our results demonstrate the potential of prediction-driven motion planning, highlighting how navigation information can enhance both prediction and planning tasks. Our implementation is at: https://github.com/KIT-MRT/future-motion.

---

## 102. A Lyapunov-based MPC for Distributed Multi Agent Systems with Time Delays and Packet Dropouts using Hidden Markov Models

**论文链接:** [http://arxiv.org/abs/2512.03708v1](http://arxiv.org/abs/2512.03708v1)

**作者:** Loaie Solyman, Aamir Ahmad, Ayman El-Badawy

**发布时间:** 2025-12-03

**备注:** 12 pages, 12 figures

### GPT解析

### 总结

本文提出了一种SCHMM LMPC框架，将半连续隐马尔可夫模型与基于李雅普诺夫的模型预测控制相结合，用于在不完美网络条件下多智能体系统的分布式最优控制。

### 背景

多智能体系统在分布式控制中面临网络不完美条件下的挑战，包括网络延迟、丢包以及通信拓扑限制等问题。

### 目的

开发一种能够同时最小化控制输入、网络误差和拓扑误差的分布式最优控制框架，并实现在线学习以适应网络变化。

### 方法

提出SCHMM LMPC框架，结合半连续隐马尔可夫模型(SCHMM)实时捕获随机网络行为，使用基于李雅普诺夫的模型预测控制(LMPC)通过线性矩阵不等式(LMIs)确保一致性和最优性。引入增量期望最大化(EM)算法实现SCHMM的在线学习。

### 主要发现

开发的优化控制问题同时最小化三个要素：控制输入、网络引起的误差(由时间延迟和丢包引起)以及拓扑引起的误差(由分布式图限制智能体获取全局信息引起)。拓扑误差是通信图固有的，无法通过离线学习解决。

### 结论

所提出的SCHMM LMPC框架能够有效减轻网络和拓扑误差，同时通过MPC保持最优性，在具有不同拓扑的多智能体系统中表现出良好的适应性和有效性。

### 翻译

我们提出了一种SCHMM LMPC框架，将半连续隐马尔可夫模型与基于李雅普诺夫的模型预测控制相结合，用于在不完美网络条件下多智能体系统的分布式最优控制。SCHMM实时捕获随机网络行为，而LMPC通过线性矩阵不等式(LMIs)确保一致性和最优性。开发的优化控制问题同时最小化三个要素：首先，减少控制输入以避免激进输入；其次，减少由时间延迟和丢包引起的网络误差；第三，减少拓扑误差，因为分布式图限制了智能体获取全局信息的能力。这种误差是通信图固有的，无法通过离线学习解决。为克服这一点，研究还引入了增量期望最大化(EM)算法，实现了SCHMM的在线学习。这种适应性使框架能够减轻网络和拓扑误差，同时通过MPC保持最优性。模拟验证了所提出的SCHMM LMPC的有效性，证明了其在具有不同拓扑的多智能体系统中的适应性。


### 论文摘要

We propose a SCHMM LMPC framework, integrating Semi Continuous Hidden Markov Models with Lyapunov based Model Predictive Control, for distributed optimal control of multi agent systems under network imperfections. The SCHMM captures the stochastic network behavior in real time, while LMPC ensures consensus and optimality via Linear Matrix Inequalities LMIs. The developed optimal control problem simultaneously minimizes three elements. First, the control effort is reduced to avoid aggressive inputs and second, the network induced error caused by time delays and packet dropouts. Third, the topology-induced error, as the distributed graph restricts agents access to global information. This error is inherent to the communication graph and cannot be addressed through offline learning. To overcome this, the study also introduces the incremental Expectation Maximization EM algorithm, enabling online learning of the SCHMM. This adaptation allows the framework to mitigate both network and topology errors while maintaining optimality through MPC. Simulations validate the effectiveness of the proposed SCHMM LMPC, demonstrating adaptability in multi agent systems with diverse topologies.

---

## 103. 论文ID: 2512.03698v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03698v1.json'

---

## 104. Linking Aneurysmal Geometry and Hemodynamics Using Computational Fluid Dynamics

**论文链接:** [http://arxiv.org/abs/2512.03660v1](http://arxiv.org/abs/2512.03660v1)

**作者:** Spyridon C. Katsoudas, Konstantina C. Kyriakoudi, Grigorios T. Chrimatopoulos, Panagiotis D. Linardopoulos, Christoforos T. Chrimatopoulos, Anastasios A. Raptis, Konstantinos G. Moulakakis, John D. Kakisis, Christos G. Manopoulos, Michail A. Xenos, Efstratios E. Tzirtzilakis

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究通过计算流体动力学分析，探讨了腹主动脉瘤几何形状与血流动力学之间的关系，发现特定几何特征可塑造剪切应力模式，可作为风险评估的生物标志物。

### 背景

腹主动脉瘤的发展和进展与复杂的血流模式和壁剪切驱动的机械生物刺激有关，但动脉瘤几何形状与血流动力学之间的定量关系仍不明确。

### 目的

研究腹主动脉瘤的几何形状如何影响血流行为，并探索几何驱动的血流特征作为患者特异性风险评估生物标志物的潜力。

### 方法

对74个患者特异性的腹主动脉进行全面的血流动力学分析，使用多尺度框架结合0D-1D全身循环模型与3D稳定有限元模拟，提取时间平均壁面剪切应力、振荡剪切指数、相对停留时间和局部归一化螺旋度等指标，以及描述直径、曲率和扭转的几何描述符。

### 主要发现

特定的几何特征可靠地塑造了剪切应力模式，几何驱动的血流特征可能作为患者特异性风险评估的有价值的生物标志物。

### 结论

将详细的几何描述符纳入未来旨在预测腹主动脉瘤生长和破裂的模型中具有潜力。

### 翻译

腹主动脉瘤的发展和进展与复杂的血流模式和壁剪切驱动的机械生物刺激有关，但动脉瘤几何形状与血流动力学之间的定量关系仍不明确。在这项研究中，我们对74个患者特异性的腹主动脉进行了全面的血流动力学分析，这是迄今为止报道的最大计算流体动力学队列之一。使用多尺度框架将0D-1D全身循环模型与3D稳定有限元模拟相结合，以生成生理一致的边界条件和高保真流场。从每个模型中，我们提取了时间平均壁面剪切应力、振荡剪切指数、相对停留时间和局部归一化螺旋度指标，以及一组扩展的几何描述符，描述直径、曲率和扭转。本研究提供了动脉瘤形状如何影响血流行为的清晰而全面的视图，得到了迄今为止最大规模的系统性分析AAA的CFD数据集的支持。我们的结果表明，特定的几何特征可靠地塑造了剪切应力模式，表明这些几何驱动的血流特征可能作为患者特异性风险评估的有价值的生物标志物。总之，这些见解强调了将详细的几何描述符纳入未来旨在预测AAA生长和破裂的模型中的潜力。


### 论文摘要

The development and progression of abdominal aortic aneurysms (AAA) are related to complex flow patterns and wall-shear-driven mechanobiological stimuli, yet the quantitative relationship between aneurysmal geometry and hemodynamics remains poorly defined. In this study, we conducted a comprehensive hemodynamic analysis of 74 patient-specific abdominal aortas, representing one of the largest Computational Fluid Dynamics (CFD) cohorts reported to date. A multiscale framework coupling 0D-1D systemic circulation models with 3D stabilized finite-element simulations is used to generate physiologically consistent boundary conditions and high-fidelity flow fields. From each model, we extract Time Averaged Wall Shear Stress (TAWSS), Oscillatory Shear Index (OSI), Relative Residence Time (RRT) and Local Normalized Helicity (LNH) indicators alongside an extended set of geometric descriptors characterizing diameter, curvature and torsion. This study provides a clear and comprehensive view of how aneurysm shape influences blood-flow behavior, supported by one of the largest systematically analyzed CFD datasets of AAAs to date. Our results show that specific geometric features reliably shape shear-stress patterns, suggesting that these geometry-driven flow signatures could serve as valuable biomarkers for patient-specific risk assessment. Together, these insights highlight the potential of incorporating detailed geometric descriptors into future models that aim to predict AAA growth and rupture.

---

## 105. V-Reactor Dynamics: Dual Chaotic System and Rewriting the Antiviral Response History

**论文链接:** [http://arxiv.org/abs/2512.03655v1](http://arxiv.org/abs/2512.03655v1)

**作者:** Yong-Shou Chen

**发布时间:** 2025-12-03

**备注:** 9 pages, 4 figures

### GPT解析

### 总结

V-Dynamics是一个基于物理学的框架，将宿主-病毒相互作用建模为同步双混沌系统，通过参数反应性(ρ)预测病毒进化、免疫反应、传播和毒力。

### 背景

COVID-19大流行暴露了我们在预测新型病毒威胁方面的失败，传统的描述性病毒学方法已不足以应对这一挑战。

### 目的

引入V-Reactor Dynamics (V-Dynamics)框架，用于更好地理解和预测病毒-宿主相互作用，从而预防未来大流行。

### 方法

将宿主-病毒相互作用建模为同步双混沌系统；通过参数反应性(ρ)的方程预测病毒特征；量化感染阶段(峰值、平台期、清除期)；引入可测量的病毒复制、免疫逃逸和药物吸收横截面；分析微观混沌和宏观混沌之间的对偶性。

### 主要发现

正确预测了SARS-CoV-2比SARS-CoV具有更高的传播性；预测了Omicron浪潮；揭示了封锁-社会经济成本权衡；发现ρ的符号决定了流行轨迹。

### 结论

V-Dynamics通过统一动力学、跨尺度动力学和混沌理论，为预防未来大流行提供了定量路线图。

### 翻译

COVID-19大流行揭示了一个关键弱点：我们未能预见新型病毒威胁。超越描述性病毒学，我们引入了V-Reactor Dynamics (V-Dynamics)，这是一个基于物理学的框架，将宿主-病毒相互作用建模为同步双混沌系统。该范式通过由参数反应性(ρ)控制的方程预测病毒进化、免疫反应、传播和毒力。它通过参数反应性(ρ)量化感染阶段：峰值(ρ>0)、平台期(ρ≈0)、清除期(ρ<0)，以及通过ρ/ℓ(反应性/代际时间)量化传播和模态。回顾性分析显示，它正确预测了SARS-CoV-2比SARS-CoV更高的传播性，并预测了Omicron浪潮，揭示了封锁-社会经济成本的权衡。我们引入了可测量的常数：病毒复制、免疫逃逸和药物吸收横截面，这些来自体外病毒粒相互作用的量子力学类比与ρ和ℓ相关，使疫情爆发前能够进行预测性监测。V-Dynamics揭示了一种对偶性：微观混沌(病毒产生)和宏观混沌(人群传播)通过标度定律相连。ρ的符号与Lyapunov指数相关，决定了流行轨迹(ρ>0表示爆发，ρ<0表示终止)，提供了控制机制。通过统一动力学、跨尺度动力学和混沌理论，该框架为预防未来大流行提供了定量路线图。


### 论文摘要

The COVID-19 pandemic revealed a key vulnerability: our failure to anticipate novel viral threats. Moving beyond descriptive virology, we introduce V-Reactor Dynamics (V-Dynamics), a physics-based framework modeling host-virus interaction as a synchronized dual chaotic system. This paradigm predicts viral evolution, immune response, transmission, and virulence through equations governed by the parameter reactivity ($ρ$). It quantifies infection phases, peak ($ρ>0$), plateau ($ρ\approx0$), clearance ($ρ<0$), transmission, and modality via $ρ/\ell$ (Reactivity/Generation time). Retrospectively, it correctly predicted SARS-CoV-2's higher transmissibility versus SARS-CoV's lethality and forecasted Omicron waves, exposing the lockdown-socioeconomic cost trade-off. We introduce measurable constants, viral replication, immune evasion, and drug absorption cross sections, derived from in vitro virion interactions. These quantum mechanical analogues relate to $ρ$ and $\ell$, enabling pre-outbreak predictive surveillance.V-Dynamics reveals a duality: microscopic chaos in viral production and macroscopic chaos in population transmission, linked by a scaling law. The sign of $ρ$, tied to the Lyapunov Exponent, dictates pandemic trajectory ($ρ>0$ for outbreak, $ρ<0$ for termination), offering a control mechanism. By unifying kinetics, cross-scale dynamics, and chaos theory, this framework provides a quantitative roadmap to preempt future pandemics.

---

## 106. Synthetic Cognitive Walkthrough: Aligning Large Language Model Performance with Human Cognitive Walkthrough

**论文链接:** [http://arxiv.org/abs/2512.03568v1](http://arxiv.org/abs/2512.03568v1)

**作者:** Ruican Zhong, David W. McDonald, Gary Hsieh

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究探索了大型语言模型(GPT-4和Gemini-2.5-pro)能否模拟人类在认知 walkthrough(CW)中的行为，发现虽然LLM行为不完全等同于人类，但通过适当提示可以使其表现接近人类，并能作为传统可用性测试的有价值补充。

### 背景

传统认知 walkthrough(CW)可用性测试成本高昂，而具有视觉推理和UI导航能力的大型语言模型(LLMs)为自动化CW提供了可能性。

### 目的

探索LLMs能否通过模拟人类行为来执行认知walkthrough，以及其表现与人类参与者的差异。

### 方法

比较GPT-4和Gemini-2.5-pro模型与人类参与者在认知walkthrough中的表现，包括导航界面、提供理由和识别潜在故障点的能力。

### 主要发现

LLMs能够导航界面并提供合理理由，但行为与人类不同；LLM提示的CW比人类实现更高的任务完成率和更优化的导航路径，但识别的潜在故障点较少；通过额外提示，LLMs可以预测人类识别的故障点，使其性能与人类参与者保持一致。

### 结论

虽然LLMs可能无法完全复制人类行为，但可以利用它们扩展可用性walkthrough并提供UI洞察，为传统可用性测试提供有价值的补充。

### 翻译

进行像认知 walkthrough (CW) 这样的可用性测试可能很昂贵。最近大型语言模型 (LLMs) 的发展，具有视觉推理和 UI 导航能力，为自动化 CW 提供了机会。我们探索了 LLMs (GPT-4 和 Gemini-2.5-pro) 能否通过将其 walkthrough 与人类参与者比较来模拟人类在 CW 中的行为。虽然 LLMs 能够导航界面并提供合理的理由，但它们的行为与人类不同。LLM 提示的 CW 实现了比人类更高的任务完成率，并遵循了更优化的导航路径，同时识别出更少的潜在故障点。然而，后续研究表明，通过额外提示，LLMs 可以预测人类识别的故障点，使其性能与人类参与者保持一致。我们的研究强调，虽然 LLMs 可能无法完全复制人类行为，但可以利用它们扩展可用性 walkthrough 并提供 UI 洞察，为传统可用性测试提供有价值的补充。


### 论文摘要

Conducting usability testing like cognitive walkthrough (CW) can be costly. Recent developments in large language models (LLMs), with visual reasoning and UI navigation capabilities, present opportunities to automate CW. We explored whether LLMs (GPT-4 and Gemini-2.5-pro) can simulate human behavior in CW by comparing their walkthroughs with human participants. While LLMs could navigate interfaces and provide reasonable rationales, their behavior differed from humans. LLM-prompted CW achieved higher task completion rates than humans and followed more optimal navigation paths, while identifying fewer potential failure points. However, follow-up studies demonstrated that with additional prompting, LLMs can predict human-identified failure points, aligning their performance with human participants. Our work highlights that while LLMs may not replicate human behaviors exactly, they can be leveraged for scaling usability walkthroughs and providing UI insights, offering a valuable complement to traditional usability testing.

---

## 107. Parameters Optimization in Trajectory Planning Using Diffrentiable Convex Programing

**论文链接:** [http://arxiv.org/abs/2512.03557v1](http://arxiv.org/abs/2512.03557v1)

**作者:** Ziqi Xu, Lin Cheng, Shengping Gong

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种可微序列凸规划框架，通过结合可微凸优化与序列凸规划，实现了端到端的参数优化，解决了传统方法对参数高度敏感的问题。

### 背景

序列凸编程已被证明是解决非凸轨迹规划问题的有效框架，但其性能对问题参数（包括轨迹变量、算法超参数和物理车辆参数）高度敏感。

### 目的

引入一种可微序列凸规划框架，将可微凸优化与序列凸规划相结合，实现端到端的参数优化，提高方法的鲁棒性和效率。

### 方法

通过推导二阶锥规划解关于问题数据的一阶灵敏度关系，获得轨迹性能指标关于任意参数的精确梯度，并通过迭代进行传播。

### 主要发现

通过三个应用（动力着陆的最优终端时间预测、子问题中的信赖域惩罚优化、高超声速滑翔飞行器的表面积与质量比优化）验证了框架的有效性，结果表明该方法能实现可靠的基于梯度的参数学习。

### 结论

可微序列凸规划框架为航空航天轨迹规划中的车辆设计、任务优化和超参数选择提供了强大而通用的工具，显著提高了数值性能、收敛行为和设计效率。

### 翻译

序列凸编程已被确立为解决非凸轨迹规划问题的有效框架。然而，其性能对问题参数高度敏感，包括轨迹变量、算法超参数和物理车辆参数。本文引入了一种可微序列凸编程框架，将可微凸优化与序列凸编程相结合，实现端到端的参数优化。通过推导二阶锥规划解关于问题数据的一阶灵敏度关系，获得了轨迹性能指标关于任意参数的精确梯度，并通过迭代进行传播。通过三个代表性应用验证了所提框架的有效性：动力着陆的最优终端时间预测、子问题中的信赖域惩罚优化以及高超声速滑翔飞行器的表面积与质量比优化。仿真结果表明，所提框架能够实现可靠的基于梯度的参数学习，并显著提高了数值性能、收敛行为和设计效率。这些结果表明，可微序列凸编程框架为航空航天轨迹规划中的车辆设计、任务优化和超参数选择提供了强大而通用的工具。


### 论文摘要

Sequential convex programming has been established as an effective framework for solving nonconvex trajectory planning problems. However, its performance is highly sensitive to problem parameters, including trajectory variables, algorithmic hyperparameters, and physical vehicle parameters. This paper introduces a differentiable sequential convex programming framework that integrates differentiable convex optimization with sequential convex programming to enable end-to-end parameter optimization. By deriving first-order sensitivity relations of second-order cone programming solutions with respect to problem data, exact gradients of trajectory performance metrics with respect to arbitrary parameters are obtained and propagated through iterations. The effectiveness of the proposed framework is validated through three representative applications: optimal terminal-time prediction for powered landing, trust-region penalty optimization in subproblems, and surface-to-mass ratio optimization for hypersonic gliding vehicles. Simulation results show that the proposed framework enables reliable gradient-based parameter learning and significantly improves numerical performance, convergence behavior, and design efficiency. These results indicate that differentiable sequential convex programming framework provides a powerful and general tool for vehicle design, mission optimization, and hyperparameter selection in aerospace trajectory planning.

---

## 108. Mean-Square Stability of Continuous-Time Stochastic Model Predictive Control

**论文链接:** [http://arxiv.org/abs/2512.03516v1](http://arxiv.org/abs/2512.03516v1)

**作者:** Qi Lü, Bowen Ma, Enrique Zuazua

**发布时间:** 2025-12-03

### GPT解析

### 总结

提出了一种随机模型预测控制框架，适用于广泛的随机微分方程，并建立了其在无限时间域内的均方指数稳定性。

### 背景

随机模型预测控制在理论稳定性方面存在空白，特别是对于具有延迟状态信息的随机微分方程系统。

### 目的

为随机微分方程系统提供具有均方稳定性保证的随机模型预测控制框架，推进随机预测控制的理论基础。

### 方法

在每个预测步骤中将非线性随机微分方程线性化，求解有限时间域随机线性二次型最优控制问题，并将结果应用于原始系统，构建延迟随机模型预测控制方案。

### 主要发现

通过Riccati方程的指数收敛性，证明了线性和中度非线性随机微分方程的全局均方指数稳定性；对于强非线性随机微分方程，建立了局部均方指数稳定性；随机微分方程的非线性项允许有多项式增长但不能有指数增长。

### 结论

首次为具有延迟状态信息的随机微分方程系统的随机模型预测控制提供了严格的均方稳定性保证，扩展了随机预测控制的理论基础。

### 翻译

我们提出了一种适用于广泛无约束随机微分方程的随机模型预测控制框架，并建立了其在无限时间域内的均方指数稳定性。在随机模型预测控制迭代的每个预测步骤中，非线性受控随机微分方程通过其在原点的线性化来近似，使用非线性系统的采样状态作为初始条件，产生有限时间域随机线性二次型最优控制问题。然后将得到的最优控制应用于原始非线性随机动力系统，直到下一个采样时刻。这种构造导致了一个延迟的随机模型预测控制方案，其闭环行为由耦合的时间延迟随机微分方程系统控制，这是一个尚未分析过的设置。通过利用Riccati方程向代数Riccati方程的指数收敛性，我们证明了线性和中度非线性随机微分方程的全局均方指数稳定性。对于强非线性随机微分方程，通过结合指数Riccati收敛性与停止时间技术和Grönwall型估计，我们建立了局部均方指数稳定性。观察到，为确保所需的局部稳定性特性，随机微分方程的非线性项允许有多项式增长但不能有指数增长，这使随机模型预测控制与其确定性 counterpart有所区别。这些结果为具有延迟状态信息的随机微分方程系统的随机模型预测控制提供了首次严格的均方稳定性保证，从而推进了随机预测控制的理论基础。


### 论文摘要

We propose a stochastic model predictive control (SMPC) framework for a broad class of unconstrained controlled stochastic differential equations (SDEs) and establish its mean-square exponential stability in the infinite-horizon limit. At each prediction step of the MPC iteration, the nonlinear controlled SDE is approximated by its linearization at the origin, with the sampled state of the nonlinear system as initial condition, yielding a finite-horizon stochastic linear-quadratic (SLQ) optimal control problem. The resulting optimal control is then applied to the original nonlinear stochastic dynamics until the next sampling instant. This construction leads to a delayed SMPC scheme whose closed-loop behavior is governed by a coupled time-delay SDE system, a setting that has not been analyzed before. We prove global mean-square exponential stability for linear and mildly nonlinear SDEs by exploiting the exponential convergence of the Riccati equation to the algebraic Riccati equation (ARE). For strongly nonlinear SDEs, we establish local mean-square exponential stability by combining exponential Riccati convergence with stopping-time techniques and Grönwall-type estimates. It is observed that, to ensure the desired local stability properties, the nonlinearities of the SDE are allowed to have polynomial growth but not exponential growth, distinguishing SMPC from its deterministic counterpart.   These results provide the first rigorous mean-square stability guarantees for SMPC of SDE systems with delayed state information, thereby advancing the theoretical foundations of stochastic predictive control.

---

## 109. GeoVideo: Introducing Geometric Regularization into Video Generation Model

**论文链接:** [http://arxiv.org/abs/2512.03453v1](http://arxiv.org/abs/2512.03453v1)

**作者:** Yunpeng Bai, Shaoheng Fang, Chaohui Yu, Fan Wang, Qixing Huang

**发布时间:** 2025-12-03

**备注:** Project Page: https://geovideo.github.io/GeoVideo/

### GPT解析

### 总结

本文提出了一种通过引入几何正则化损失和深度预测来改进视频生成质量的方法，解决了现有方法在3D结构建模上的不足。

### 背景

视频生成领域的最新进展已使用扩散变压器模型合成了高质量视频片段，但大多数现有方法完全在2D像素空间操作，缺乏对3D结构的显式建模，导致时间上不一致的几何结构、不合理运动和结构伪影。

### 目的

引入几何正则化损失到视频生成中，通过为潜在扩散模型添加每帧深度预测来增强模型，提高视频的时空一致性、形状一致性和物理合理性。

### 方法

采用深度作为几何表示，利用深度预测的进展及其与基于图像的潜在编码器的兼容性；提出多视图几何损失，将预测的深度图在共享的3D坐标系中对齐，强制时间上的结构一致性。

### 主要发现

该方法弥合了外观生成和3D结构建模之间的差距，显著提高了时空一致性、形状一致性和物理合理性；在多个数据集上的实验表明，该方法比现有基线产生更稳定和几何上一致的结果。

### 结论

通过引入几何正则化损失和深度预测，有效改进了视频生成的质量和一致性，解决了现有方法在3D结构建模方面的不足。

### 翻译

视频生成的最新进展已使用扩散变压器模型实现了高质量和视觉上真实的视频片段的合成。然而，大多数现有方法完全在2D像素空间中操作，缺乏对3D结构的显式建模机制，常常导致时间上不一致的几何结构、不合理的运动和结构伪影。在这项工作中，我们通过为潜在扩散模型添加每帧深度预测，将几何正则化损失引入视频生成。我们选择深度作为几何表示，是因为深度预测取得了显著进展，并且它与基于图像的潜在编码器兼容。具体来说，为了强制时间上的结构一致性，我们提出了一种多视图几何损失，该损失将预测的深度图在共享的3D坐标系中对齐。我们的方法弥合了外观生成和3D结构建模之间的差距，从而提高了时空一致性、形状一致性和物理合理性。在多个数据集上的实验表明，我们的方法比现有基线产生明显更稳定和几何上一致的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有视频生成模型缺乏3D结构建模的问题，导致生成的视频在时间上几何不一致、运动不合理并存在结构伪影。这个问题很重要，因为高质量视频生成对内容创作、虚拟现实等应用至关重要，而具有物理合理性的世界建模在3D场景合成、机器人和具身AI等领域有重要价值，推动视频生成从单纯视觉合成向结构化3D世界建模转变。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析当前视频生成模型的局限性，认为真实视频生成需要超越视觉连贯性，需要对3D世界有结构化理解。他们选择深度图作为几何表示，因为深度预测进展显著且与图像编码器兼容。方法设计上引入几何正则化损失，通过多视图几何损失对齐深度图。该方法借鉴了扩散模型、潜在扩散架构、深度预测技术、多视图几何原理、ControlNet的输入输出扩展和VGGT等视觉几何基础模型等多项现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过引入几何正则化到视频生成过程，使模型在生成视频时不仅考虑外观一致性，还确保3D几何结构在时间上的连贯性。整体流程包括：1) 修改模型架构支持RGB和深度双模态输入；2) 两阶段训练——先训练RGB-D联合生成，再引入几何正则化；3) 实现几何正则化通过构建全局点云和深度重投影一致性来监督跨帧深度一致性；4) 对动态视频使用运动概率图区分动态和静态内容，特殊处理几何一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次在视频生成中引入显式几何建模；2) 提出跨帧几何一致性损失确保全局几何一致性；3) 设计两阶段训练策略平衡几何约束和视觉质量；4) 针对动态场景引入运动概率图特殊处理。相比之前工作，不同于纯2D视频生成模型仅依赖时序注意力；与其他引入额外先验的方法不同，GeoVideo专注于几何一致性；相比深度估计工作，GeoVideo将深度预测整合到生成过程中；与3D场景生成方法不同，GeoVideo专注于改进视频生成同时支持3D重建；比WVD等方法更通用，适用于各种场景和视频生成任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GeoVideo通过在视频生成过程中引入几何正则化，利用深度预测和跨帧一致性约束，显著提升了生成视频的时空连贯性、形状一致性和物理合理性，同时使生成的视频能够支持高质量的3D场景重建。'}


### 论文摘要

Recent advances in video generation have enabled the synthesis of high-quality and visually realistic clips using diffusion transformer models. However, most existing approaches operate purely in the 2D pixel space and lack explicit mechanisms for modeling 3D structures, often resulting in temporally inconsistent geometries, implausible motions, and structural artifacts. In this work, we introduce geometric regularization losses into video generation by augmenting latent diffusion models with per-frame depth prediction. We adopted depth as the geometric representation because of the great progress in depth prediction and its compatibility with image-based latent encoders. Specifically, to enforce structural consistency over time, we propose a multi-view geometric loss that aligns the predicted depth maps across frames within a shared 3D coordinate system. Our method bridges the gap between appearance generation and 3D structure modeling, leading to improved spatio-temporal coherence, shape consistency, and physical plausibility. Experiments across multiple datasets show that our approach produces significantly more stable and geometrically consistent results than existing baselines.

---

## 110. PretrainZero: Reinforcement Active Pretraining

**论文链接:** [http://arxiv.org/abs/2512.03442v1](http://arxiv.org/abs/2512.03442v1)

**作者:** Xingrun Xing, Zhiyuan Fan, Jie Lou, Guoqi Li, Jiajun Zhang, Debing Zhang

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为PretrainZero的强化主动学习框架，旨在扩展强化学习从特定领域的后训练扩展到通用预训练，以突破通用推理能力的性能边界。

### 背景

当前基于强化学习(RL)的大型思维模型在特定领域展现出专家级能力，但仍严重依赖特定领域的可验证奖励，限制了通用推理能力性能边界的扩展。

### 目的

开发一种能够模仿人类行为，从一般经验中主动学习，从而突破通用推理能力瓶颈的强化学习框架。

### 方法

PretrainZero是一种建立在预训练语料上的强化主动学习框架，具有三个特点：1)主动预训练，学习统一推理策略从语料中识别合理内容；2)自监督学习，无需可验证标签直接在通用语料上预训练；3)验证扩展，通过解决挑战性掩码跨度增强推理能力。

### 主要发现

在强化预训练中，PretrainZero在MMLU-Pro、SuperGPQA和数学平均基准上分别提升了Qwen3-4B-Base模型8.43、5.96和10.60。在后训练中，预训练模型可作为下游RLVR任务的推理基础模型。

### 结论

PretrainZero成功突破了通用推理的验证数据壁垒，显著提升了预训练基础模型的通用推理能力，为扩展强化学习从特定领域到通用领域提供了有效方法。

### 翻译

模仿人类行为从一般经验中主动学习以实现通用人工智能一直是人类的梦想。最近的基于强化学习(RL)的大型思维模型展示了令人印象深刻的专家级能力，即软件和数学，但仍严重依赖特定领域的可验证奖励，这对扩展通用推理能力的性能边界构成了重大瓶颈。在这项工作中，我们提出了PretrainZero，一种建立在预训练语料上的强化主动学习框架，将强化学习从特定领域的后训练扩展到通用预训练。PretrainZero具有以下特点：1)主动预训练：受人类主动学习能力启发，PretrainZero学习统一的推理策略，从预训练语料中主动识别合理且信息丰富的内容，并通过RL推理预测这些内容。2)自监督学习：无需任何可验证标签、预训练奖励模型或监督微调，我们直接在通用Wikipedia语料上使用RL预训练3到30B基础模型推理器，显著突破了通用推理的验证数据壁垒。3)验证扩展：通过解决越来越具挑战性的掩码跨度，PretrainZero大幅增强了预训练基础模型的通用推理能力。在强化预训练中，PretrainZero在MMLU-Pro、SuperGPQA和数学平均基准上分别提升了Qwen3-4B-Base模型8.43、5.96和10.60。在后训练中，预训练模型也可作为下游RLVR任务的推理基础模型。


### 论文摘要

Mimicking human behavior to actively learning from general experience and achieve artificial general intelligence has always been a human dream. Recent reinforcement learning (RL) based large-thinking models demonstrate impressive expert-level abilities, i.e., software and math, but still rely heavily on verifiable rewards in specific domains, placing a significant bottleneck to extend the performance boundary of general reasoning capabilities. In this work, we propose PretrainZero, a reinforcement active learning framework built on the pretraining corpus to extend RL from domain-specific post-training to general pretraining. PretrainZero features the following characteristics: 1) Active pretraining: inspired by the active learning ability of humans, PretrainZero learns a unified reasoning policy to actively identify reasonable and informative contents from pretraining corpus, and reason to predict these contents by RL. 2) Self-supervised learning: without any verifiable labels, pretrained reward models, or supervised fine-tuning, we directly pretrain reasoners from 3 to 30B base models on the general Wikipedia corpus using RL, significantly breaking the verification data-wall for general reasoning. 3) Verification scaling: by tackling increasingly challenging masked spans, PretrainZero substantially enhances the general reasoning abilities of pretrained base models. In reinforcement pretraining, PretrainZero improves Qwen3-4B-Base for 8.43, 5.96 and 10.60 on MMLU-Pro, SuperGPQA and math average benchmarks. In post-training, the pretrained models can also serve as reasoning foundation models for downstream RLVR tasks.

---

## 111. 论文ID: 2512.03436v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03436v1.json'

---

## 112. 论文ID: 2512.03411v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03411v1.json'

---

## 113. Tensor renormalization group calculations of partition-function ratios

**论文链接:** [http://arxiv.org/abs/2512.03395v1](http://arxiv.org/abs/2512.03395v1)

**作者:** Satoshi Morita, Naoki Kawashima

**发布时间:** 2025-12-03

**备注:** 9 pages, 7 figures

### GPT解析

### 总结

本研究分析了定义为配分函数比的无量纲量的行为，以研究相变和临界现象。通过数值计算验证了这些比率在临界点的普适值与共形场论预测的一致性，并观察到了四态Potts模型中的对数修正。

### 背景

相变和临界现象是统计物理中的重要研究课题。共形场理论(CFT)提供了一种预测临界点普适值的理论框架，特别是在环面上模不变配分函数的基础上。

### 目的

验证作为配分函数比的无量纲量在临界点的行为是否符合共形场论的预测，并研究不同普适类模型中这些比率的系统大小依赖性。

### 方法

使用键加权张量重正化群方法，对三个属于不同普适类的二维模型(伊辛模型、三态Potts模型和四态Potts模型)进行数值计算，分析配分函数比率的有限尺寸标度行为。

### 主要发现

1. 配分函数比率遵循与Binder参数相同的有限尺寸标度形式；2. 这些比率的临界值与共形场论预测的普适值非常吻合；3. 在四态Potts模型中，观察到这些比率的系统大小依赖性存在对数修正。

### 结论

作为配分函数比的无量纲量是研究相变和临界现象的有效工具，其临界行为符合共形场论的预测，但在某些情况下可能存在对数修正等非平凡标度行为。

### 翻译

本研究分析了定义为配分函数比的无量纲量的行为，以研究相变和临界现象。在临界点，这些比率的普适值可以通过共形场论(CFT)在环面上的模不变配分函数来预测。我们使用键加权张量重正化群对三个属于不同普适类的二维模型进行了数值计算：伊辛模型、三态Potts模型和四态Potts模型。配分函数比率遵循与Binder参数相同的有限尺寸标度形式，它们的临界值与共形场论预测的普适值非常吻合。在四态Potts模型中，我们观察到这些比率的系统大小依赖性存在对数修正。


### 论文摘要

The behavior of dimensionless quantities defined as ratios of partition functions is analyzed to investigate phase transitions and critical phenomena. At criticality, the universal values of these ratios can be predicted from conformal field theory (CFT) through the modular-invariant partition functions on a torus. We perform numerical calculations using the bond-weighted tensor renormalization group for three two-dimensional models belonging to different universality classes: the Ising model, the three-state Potts model, and the four-state Potts model. The partition-function ratios obey the same finite-size scaling form as the Binder parameter, and their critical values agree well with the universal values predicted by CFT. In the four-state Potts model, we observe logarithmic corrections in the system-size dependence of these ratios.

---

## 114. When does Gaussian equivalence fail and how to fix it: Non-universal behavior of random features with quadratic scaling

**论文链接:** [http://arxiv.org/abs/2512.03325v1](http://arxiv.org/abs/2512.03325v1)

**作者:** Garrett G. Wen, Hong Hu, Yue M. Lu, Zhou Fan, Theodor Misiakiewicz

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文研究了高斯等价理论在高维随机特征模型二次缩放规则下的失效问题，并提出条件高斯等价模型来准确描述模型行为。

### 背景

现代高维统计学致力于分析通过经验风险最小化在非线性特征嵌入上训练的线性预测器。高斯等价理论作为这一领域的普遍性原理，指出高维复杂特征的行为可通过高斯代理来捕捉。

### 目的

探究高斯等价理论在随机特征模型二次缩放规则下的失效原因，并提出能准确描述模型行为的替代框架。

### 方法

在随机特征模型的二次缩放规则下进行研究，引入条件高斯等价模型，结合Wiener混沌展开的中心极限定理和两阶段Lindeberg交换论证进行分析。

### 主要发现

当目标函数依赖于数据的低维投影时，高斯等价理论会产生错误预测；提出的条件高斯等价模型能准确描述二次缩放规则下的随机特征模型，并推导出与数值模拟一致的训练和测试误差渐近性质。

### 结论

条件高斯等价模型作为高斯框架的混合扩展，保留了高斯框架的可处理性，并能准确描述高维ERM中的普遍性现象，特别是在高斯等价理论失效的情况下。

### 翻译

现代高维统计学的一个主要努力方向是致力于通过经验风险最小化(ERM)分析在非线性特征嵌入上训练的线性预测器。高斯等价理论(GET)已成为这一背景下的有力普遍性原理：它指出高维、复杂特征的行为可以通过高斯代理来捕捉，这些代理更易于分析。尽管取得了显著成功，但数值实验表明，即使在多项式映射等简单嵌入情况下，这种等价关系在一般缩放规则下也可能失效。我们在随机特征(RF)模型的二次缩放规则下研究这种失效现象，其中特征数量和样本量都随数据维度二次增长。我们表明，当目标函数依赖于数据的低维投影(如广义线性模型)时，GET会产生错误预测。为了捕捉正确的渐近性质，我们引入了条件高斯等价(CGE)模型，可以将其视为在高维高斯模型上附加一个低维非高斯组件。这种混合模型保留了高斯框架的可处理性，并能准确描述二次缩放规则下的RF模型。我们推导了该设置下的训练和测试误差的精确渐近性质，即使在GET失效的情况下，这些结果仍与数值模拟一致。我们的分析结合了关于Wiener混沌展开的中心极限定理的一般结果和仔细的两阶段Lindeberg交换论证。除了RF模型和二次缩放规则外，我们的工作暗示了在高维ERM中存在丰富的普遍性现象景观。


### 论文摘要

A major effort in modern high-dimensional statistics has been devoted to the analysis of linear predictors trained on nonlinear feature embeddings via empirical risk minimization (ERM). Gaussian equivalence theory (GET) has emerged as a powerful universality principle in this context: it states that the behavior of high-dimensional, complex features can be captured by Gaussian surrogates, which are more amenable to analysis. Despite its remarkable successes, numerical experiments show that this equivalence can fail even for simple embeddings -- such as polynomial maps -- under general scaling regimes.   We investigate this breakdown in the setting of random feature (RF) models in the quadratic scaling regime, where both the number of features and the sample size grow quadratically with the data dimension. We show that when the target function depends on a low-dimensional projection of the data, such as generalized linear models, GET yields incorrect predictions. To capture the correct asymptotics, we introduce a Conditional Gaussian Equivalent (CGE) model, which can be viewed as appending a low-dimensional non-Gaussian component to an otherwise high-dimensional Gaussian model. This hybrid model retains the tractability of the Gaussian framework and accurately describes RF models in the quadratic scaling regime. We derive sharp asymptotics for the training and test errors in this setting, which continue to agree with numerical simulations even when GET fails.   Our analysis combines general results on CLT for Wiener chaos expansions and a careful two-phase Lindeberg swapping argument. Beyond RF models and quadratic scaling, our work hints at a rich landscape of universality phenomena in high-dimensional ERM.

---

## 115. Numerical optimization for the compatibility constant of the lasso

**论文链接:** [http://arxiv.org/abs/2512.03321v1](http://arxiv.org/abs/2512.03321v1)

**作者:** Kei Hirose

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究提出了一种计算兼容性常数的数值方法，解决了当变量数量超过观测数量时lasso预测误差评估中的计算难题。

### 背景

兼容性条件和兼容性常数常用于评估变量数量超过观测数量时lasso的预测误差，但兼容性常数的计算通常困难，因为它是一个复杂的非线性优化问题。

### 目的

开发一种当真实回归系数的零/非零模式已知时计算兼容性常数的方法，并研究其有限样本行为。

### 方法

提出一种数值方法，当非零系数的符号被指定时，将优化问题简化为二次规划(QP)问题；对于真实非零系数数量适中的情况，提出混合整数二次规划(MIQP)方法；通过模拟和真实数据分析研究兼容性常数的有限样本行为。

### 主要发现

兼容性常数可以通过解决所有可能符号组合的QP问题获得；混合整数二次规划方法适用于真实非零系数数量适中的情况；兼容性常数的有限样本行为在多种参数设置下得到了验证；均方误差与基于兼容性常数的理论误差边界进行了比较。

### 结论

所提出的方法有效解决了兼容性常数计算的困难问题，为评估高维数据中lasso的预测误差提供了实用工具。

### 翻译

兼容性条件和兼容性常数已被广泛用于评估当变量数量超过观测数量时lasso的预测误差。然而，兼容性常数的计算通常困难，因为它是一个复杂的非线性优化问题。在本研究中，我们提出了一种数值方法来计算当真实回归系数的零/非零模式已知时的兼容性常数。我们表明，一旦指定了非零系数的符号，优化问题可以简化为一个二次规划(QP)问题。在这种情况下，可以通过解决所有可能符号组合的QP来获得兼容性常数。我们还制定了一种混合整数二次规划(MIQP)方法，适用于真实非零系数数量适中的情况。我们在各种参数设置下通过模拟数据研究了兼容性常数的有限样本行为，并将均方误差与基于兼容性常数的理论误差边界进行了比较。通过真实数据分析也研究了兼容性常数在有限样本中的行为。


### 论文摘要

Compatibility condition and compatibility constant have been commonly used to evaluate the prediction error of the lasso when the number of variables exceeds the number of observations. However, the computation of the compatibility constant is generally difficult because it is a complicated nonlinear optimization problem. In this study, we present a numerical approach to compute the compatibility constant when the zero/nonzero pattern of true regression coefficients is given. We show that the optimization problem reduces to a quadratic program (QP) once the signs of the nonzero coefficients are specified. In this case, the compatibility constant can be obtained by solving QPs for all possible sign combinations. We also formulate a mixed-integer quadratic programming (MIQP) approach that can be applied when the number of true nonzero coefficients is moderately large. We investigate the finite-sample behavior of the compatibility constant for simulated data under a wide variety of parameter settings and compare the mean squared error with its theoretical error bound based on the compatibility constant. The behavior of the compatibility constant in finite samples is also investigated through a real data analysis.

---

## 116. Unlocking hidden biomolecular conformational landscapes in diffusion models at inference time

**论文链接:** [http://arxiv.org/abs/2512.03312v1](http://arxiv.org/abs/2512.03312v1)

**作者:** Daniel D. Richman, Jessica Karaguesian, Carl-Mikael Suomivuori, Ron O. Dror

**发布时间:** 2025-12-02

**备注:** Project page: https://github.com/drorlab/conformix

### GPT解析

### 总结

ConforMix是一种推理时算法，通过结合分类器引导、过滤和自由能估计，增强构象分布的采样效率，能够更有效地发现构象变异性而无需预先了解主要自由度。

### 背景

生物分子（如蛋白质）的功能取决于它们在不同构象之间的转化能力。研究人员几十年来一直致力于开发计算方法来预测构象分布，这比确定静态折叠结构更难通过实验确定。

### 目的

开发一种能够增强构象分布采样效率的算法，使扩散模型能够更好地捕捉生物分子的构象变异性。

### 方法

ConforMix是一种推理时算法，结合分类器引导、过滤和自由能估计来升级扩散模型，无论这些模型是为静态结构预测还是构象生成训练的。该方法与模型预训练的改进是正交的，适用于各种扩散模型。

### 主要发现

当应用于为静态结构预测训练的扩散模型时，ConforMix能够捕捉结构变化，包括结构域运动、隐蔽口袋的灵活性和转运蛋白的循环，同时避免非物理状态。对生物关键蛋白质的案例研究证明了该方法的可扩展性、准确性和实用性。

### 结论

Confor是一种有效的算法，可以增强生物分子构象分布的采样，适用于各种扩散模型，能够捕捉重要的生物学结构变化，即使对于完美再现玻尔兹曼分布的假设模型也有益。

### 翻译

生物分子（如蛋白质）的功能取决于它们在不同结构或'构象'之间转化的能力。研究人员几十年来一直致力于开发计算方法来预测构象分布，这比确定静态折叠结构更难通过实验确定。我们提出了ConforMix，一种推理时算法，通过结合分类器引导、过滤和自由能估计来增强构象分布的采样。我们的方法升级了扩散模型——无论这些模型是为静态结构预测还是构象生成训练的——以更有效地发现构象变异性，而无需预先了解主要自由度。ConforMix与模型预训练的改进是正交的，即使对于完美再现玻尔兹曼分布的假设模型也是有利的。值得注意的是，当应用于为静态结构预测训练的扩散模型时，Confor捕捉了包括结构域运动、隐蔽口袋的灵活性和转运蛋白循环在内的结构变化，同时避免了非物理状态。对生物关键蛋白质的案例研究证明了该方法的可扩展性、准确性和实用性。


### 论文摘要

The function of biomolecules such as proteins depends on their ability to interconvert between a wide range of structures or "conformations." Researchers have endeavored for decades to develop computational methods to predict the distribution of conformations, which is far harder to determine experimentally than a static folded structure. We present ConforMix, an inference-time algorithm that enhances sampling of conformational distributions using a combination of classifier guidance, filtering, and free energy estimation. Our approach upgrades diffusion models -- whether trained for static structure prediction or conformational generation -- to enable more efficient discovery of conformational variability without requiring prior knowledge of major degrees of freedom. ConforMix is orthogonal to improvements in model pretraining and would benefit even a hypothetical model that perfectly reproduced the Boltzmann distribution. Remarkably, when applied to a diffusion model trained for static structure prediction, ConforMix captures structural changes including domain motion, cryptic pocket flexibility, and transporter cycling, while avoiding unphysical states. Case studies of biologically critical proteins demonstrate the scalability, accuracy, and utility of this method.

---

## 117. 论文ID: 2512.03295v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03295v1.json'

---

## 118. The Origins of the Bulk flow

**论文链接:** [http://arxiv.org/abs/2512.03168v1](http://arxiv.org/abs/2512.03168v1)

**作者:** Richard Watkins, Hume A. Feldman

**发布时间:** 2025-12-02

**备注:** 7 pages, 6 figures

### GPT解析

### 总结

研究使用CosmicFlows 4目录分析大尺度体流起源，发现观测体流主要由200Mpc/h之外的结构驱动，挑战了空间均匀流动的假设。

### 背景

使用CosmicFlows 4 (CF4)特殊速度目录研究大尺度体流的起源问题。

### 目的

分析大尺度体流的起源，特别是确定内部和外部结构对体流的贡献。

### 方法

将观测运动分解为内部和外部分量；开发加权平均技术测试模型一致性；使用最小方差形式论分离体流的内外贡献。

### 主要发现

CF4速度与预测内部场吻合(beta=0.31±0.01)；确定哈勃常数H0=75.9±0.1 km/s/Mpc；观测体流主要由200Mpc/h之外源主导；外部驱动流动幅度随尺度单调增加。

### 结论

CF4速度场可靠性得到强化；对外部源产生空间均匀流动的假设提出质疑；挑战了局部体积流动可建模为空间均匀的常见假设。

### 翻译

我们使用CosmicFlows 4 (CF4)特殊速度目录分析大尺度体流的起源。我们将观测到的运动分解为内部分量(由200Mpc/h内的质量波动产生)和外部分量(源于这一体积之外的结构)。开发了一种加权平均技术来测试模型的自我一致性，同时最小化非高斯距离误差的影响。CF4速度与预测的内部场极好地吻合，得到beta = 0.31 ± 0.01。我们还确定用于从CF4计算特殊速度的哈勃常数H0 = 75.9 ± 0.1 km/s/Mpc，与CF4校准一致。使用最小方差形式论，我们将体流进一步分离为内部和外部贡献，发现观测到的大尺度体流主要由200Mpc/h之外的源主导。这种外部驱动的流动幅度随尺度单调增加，与远处巨大过密度的影响一致。这些发现强化了CF4速度场的可靠性，同时对外部源产生的空间均匀流动的假设提出质疑。我们的结果挑战了通常认为由外部质量浓度引起的局部体积流动可以建模为空间均匀的常见假设。


### 论文摘要

We analyze the origin of the large scale bulk flow using the CosmicFlows 4 (CF4) peculiar velocity catalog. We decompose the observed motions into internal components, generated by mass fluctuations within 200Mpc/h, and external ones arising from structures beyond this volume. A weighted average technique is developed to test the model's self consistency while minimizing the impact of non Gaussian distance errors. The CF4 velocities show excellent agreement with the predicted internal field, yielding beta = 0.31 pm 0.01. We also determine that the value of the Hubble constant that should be used for calculating peculiar velocities from the CF4 to be H0 = 75.9 pm 0.1 km/s/ Mpc, consistent with CF4 calibrations. Using the minimum variance formalism, we further separate the bulk flow into its internal and external contributions and find that the observed large scale bulk flow is dominated by sources beyond 200Mpc/h. The amplitude of this externally driven flow increases monotonically with scale, consistent with the influence of a distant, massive overdensity. These findings reinforce the reliability of the CF4 velocity field while calling into question the assumption of a spatially uniform flow generated by external sources. Our results challenge the commonly made assumption that the flow in our local volume due to external mass concentrations can be modeled as being spatially uniform.

---

## 119. DGGT: Feedforward 4D Reconstruction of Dynamic Driving Scenes using Unposed Images

**论文链接:** [http://arxiv.org/abs/2512.03004v1](http://arxiv.org/abs/2512.03004v1)

**作者:** Xiaoxue Chen, Ziyi Xiong, Yuantao Chen, Gen Li, Nan Wang, Hongcheng Luo, Long Chen, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Hongyang Li, Ya-Qin Zhang, Hao Zhao

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种名为Driving Gaussian Grounded Transformer (DGGT)的前馈框架，用于自动驾驶场景的无姿态动态场景重建，实现了快速、可扩展的4D重建和重模拟。

### 背景

自动驾驶需要快速、可扩展的4D重建和重模拟方法用于训练和评估，但现有方法大多依赖单场景优化、已知相机校准或短帧窗口，导致速度慢且不实用。

### 目的

重新从前馈视角解决动态驾驶场景重建问题，引入一个统一的框架，实现无姿态的动态场景重建。

### 方法

将相机姿态作为模型输出而非输入，直接从稀疏、无姿态图像进行重建；联合预测每帧3D高斯图和相机参数；使用轻量级动态头解耦动态特性；通过寿命头保持时间一致性；采用基于扩散的渲染细化减少伪影并改善新视图质量。

### 主要发现

在Waymo、nuScenes和Argoverse2等大规模驾驶基准数据集上训练和评估，该方法在每个数据集训练时和跨数据集零样本迁移中均优于先前工作，且随着输入帧数增加具有良好的可扩展性。

### 结论

提出的单通道、无姿态算法实现了最先进的性能和速度，为自动驾驶场景重建提供了高效解决方案。

### 翻译

自动驾驶需要快速、可扩展的4D重建和重模拟用于训练和评估，然而大多数动态驾驶场景的方法仍然依赖单场景优化、已知相机校准或短帧窗口，使它们速度慢且不实用。我们从前馈视角重新审视这个问题，引入了Driving Gaussian Grounded Transformer (DGGT)，这是一个用于无姿态动态场景重建的统一框架。我们注意到现有公式将相机姿态视为必需输入，限制了灵活性和可扩展性。相反，我们将姿态重新建模为模型的输出，使能够直接从稀疏、无姿态图像进行重建，并支持任意数量的视图用于长序列。我们的方法联合预测每帧3D高斯图和相机参数，使用轻量级动态头解耦动态特性，并通过寿命头保持时间一致性，该头随时间调节可见性。基于扩散的渲染细化进一步减少了运动/插值伪影，并在稀疏输入下改善了新视图质量。结果是单通道、无姿态的算法，实现了最先进的性能和速度。在大型驾驶基准数据集（Waymo、nuScenes、Argoverse2）上训练和评估，我们的方法在各自数据集训练时和跨数据集零样本迁移中均优于先前工作，并且随着输入帧数的增加具有良好的可扩展性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何快速、可扩展地进行动态驾驶场景的4D重建和重模拟，而无需依赖场景特定优化、已知相机校准或短帧窗口限制。这个问题在自主驾驶领域至关重要，因为训练和评估驾驶系统需要快速将原始驾驶日志转换为可编辑和重新渲染的场景表示，支持假设分析。现有方法速度慢且不实用，无法作为常规预处理步骤，限制了重建技术在自动驾驶中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从重新审视问题出发，采用前馈而非优化的视角，注意到现有方法将相机位姿作为必需输入限制了灵活性和可扩展性。因此他们将位姿重新定义为模型的输出，使重建可以直接从稀疏未标定图像进行。方法借鉴了ViT架构、DINO特征提取、3D高斯溅射表示和可微分渲染器等现有技术，但创新性地结合了多个预测头来处理动态场景的不同方面。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将相机位姿作为输出而非输入，使用像素对齐的高斯图表示每个场景帧，通过动态头解耦动态元素，通过寿命头保持时间一致性，并使用基于扩散的渲染细化减少运动伪影。整体流程包括：输入未标定图像序列→ViT编码器提取特征→多个预测头输出相机参数、高斯表示、动态信息等→动态分解静态和动态元素→运动插值处理中间帧→可微分渲染→扩散细化提高质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)无位姿前馈框架，首次实现从未标定图像直接重建动态场景；2)统一表示，联合预测相机参数、3D高斯和动态信息；3)动态分解，解耦静态背景和动态实体；4)寿命参数，捕捉静态区域随时间的外观变化；5)扩散细化，减少运动伪影。相比之前工作，该方法不依赖相机位姿输入，支持任意数量的输入视图和长序列，实现单次前馈计算无需每场景优化，并在多个数据集上表现出良好的零样本泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了DGGT，一种无位姿输入的前馈框架，能够从未标定图像中快速重建动态驾驶场景的4D表示，支持任意视图数量和长序列，并实现了高质量的实例级场景编辑能力。'}


### 论文摘要

Autonomous driving needs fast, scalable 4D reconstruction and re-simulation for training and evaluation, yet most methods for dynamic driving scenes still rely on per-scene optimization, known camera calibration, or short frame windows, making them slow and impractical. We revisit this problem from a feedforward perspective and introduce \textbf{Driving Gaussian Grounded Transformer (DGGT)}, a unified framework for pose-free dynamic scene reconstruction. We note that the existing formulations, treating camera pose as a required input, limit flexibility and scalability. Instead, we reformulate pose as an output of the model, enabling reconstruction directly from sparse, unposed images and supporting an arbitrary number of views for long sequences. Our approach jointly predicts per-frame 3D Gaussian maps and camera parameters, disentangles dynamics with a lightweight dynamic head, and preserves temporal consistency with a lifespan head that modulates visibility over time. A diffusion-based rendering refinement further reduces motion/interpolation artifacts and improves novel-view quality under sparse inputs. The result is a single-pass, pose-free algorithm that achieves state-of-the-art performance and speed. Trained and evaluated on large-scale driving benchmarks (Waymo, nuScenes, Argoverse2), our method outperforms prior work both when trained on each dataset and in zero-shot transfer across datasets, and it scales well as the number of input frames increases.

---

## 120. Stable Signer: Hierarchical Sign Language Generative Model

**论文链接:** [http://arxiv.org/abs/2512.04048v1](http://arxiv.org/abs/2512.04048v1)

**作者:** Sen Fang, Yalin Feng, Hongbin Zhong, Yanxin Zhang, Dimitris N. Metaxas

**发布时间:** 2025-12-03

**备注:** 12 pages, 7 figures. More Demo at https://stablesigner.github.io

### GPT解析

### 总结

本文介绍了一种名为'Stable Signer'的新手语生成模型，它重新定义了手语生成任务为分层生成的端到端任务，只包括文本理解和姿势到视频转换两个阶段。

### 背景

大多数先前的研究关注Text2Gloss、Gloss2Pose、Pose2Vid等阶段，有些集中在Prompt2Gloss和Text2Avatar阶段。然而，由于文本转换、姿势生成以及将姿势渲染为真实人类视频的不准确性，该领域进展缓慢，导致误差逐渐累积。

### 目的

简化传统冗余结构，简化并优化任务目标，设计一种新的手语生成模型以提高生成质量。

### 方法

重新定义SLP任务为只包括文本理解(Prompt2Gloss, Text2Gloss)和Pose2Vid的分层生成端到端任务；设计新的手语理解链接器SLUL执行文本理解；使用SLP-MoE手部姿势渲染专家块生成手部姿势；采用新开发的语义感知词汇掩码损失(SAGM Loss)训练SLUL。

### 主要发现

与当前最先进的生成方法相比，新模型的性能提高了48.6%。

### 结论

通过简化传统结构和优化任务目标，Stable Signer模型能够端到端生成高质量、多风格的手语视频，解决了传统方法中误差累积的问题。

### 翻译

手语生成是将复杂输入文本转换为实际视频的过程。大多数先前的工作集中在文本到手语词汇、手语词汇到姿势、姿势到视频等阶段，有些专注于提示到手语词汇、文本到虚拟形象等阶段。然而，由于这些阶段中文本转换、姿势生成以及将姿势渲染为真实人类视频的不准确性，该领域进展缓慢，导致误差逐渐累积。因此，本文简化了传统冗余结构，简化和优化了任务目标，并设计了一种名为'Stable Signer'的新手语生成模型。它将SLP任务重新定义为仅包括文本理解(提示到手语词汇、文本到手语词汇)和姿势到视频的分层生成端到端任务，并通过我们提出的新的手语理解链接器SLUL执行文本理解，使用名为SLP-MoE的手部姿势渲染专家块生成手部姿势，以端到端方式生成高质量、多风格的手语视频。SLUL使用新开发的语义感知词汇掩码损失进行训练，其性能比当前最先进的生成方法提高了48.6%。


### 论文摘要

Sign Language Production (SLP) is the process of converting the complex input text into a real video. Most previous works focused on the Text2Gloss, Gloss2Pose, Pose2Vid stages, and some concentrated on Prompt2Gloss and Text2Avatar stages. However, this field has made slow progress due to the inaccuracy of text conversion, pose generation, and the rendering of poses into real human videos in these stages, resulting in gradually accumulating errors. Therefore, in this paper, we streamline the traditional redundant structure, simplify and optimize the task objective, and design a new sign language generative model called Stable Signer. It redefines the SLP task as a hierarchical generation end-to-end task that only includes text understanding (Prompt2Gloss, Text2Gloss) and Pose2Vid, and executes text understanding through our proposed new Sign Language Understanding Linker called SLUL, and generates hand gestures through the named SLP-MoE hand gesture rendering expert block to end-to-end generate high-quality and multi-style sign language videos. SLUL is trained using the newly developed Semantic-Aware Gloss Masking Loss (SAGM Loss). Its performance has improved by 48.6% compared to the current SOTA generation methods.

---

## 121. Machine Learning Pipeline for Denoising Low Signal-To-Noise Ratio and Out-of-Distribution Transmission Electron Microscopy Datasets

**论文链接:** [http://arxiv.org/abs/2512.04045v1](http://arxiv.org/abs/2512.04045v1)

**作者:** Brian Lee, Meng Li, Judith C Yang, Dmitri N Zakharov, Xiaohui Qu

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究提出了一种针对时间序列高分辨率透射电子显微镜图像的自监督机器学习去噪流程，解决了现有方法计算成本高、推理速度慢以及泛化能力有限的问题。

### 背景

高分辨率透射电子显微镜对于观察材料在埃尺度上的结构和形态演化至关重要，但电子束会改变这些过程。基于CMOS的直接电子探测器可减少电子剂量，但会导致图像信噪比低，需要帧积分而牺牲时间分辨率。现有去噪模型计算成本高，推理速度跟不上先进探测器的成像速度，且在偏离训练数据集的成像条件下性能未评估。

### 目的

开发一种计算效率高、推理速度快且具有良好泛化能力的去噪模型，专门用于时间序列HRTEM图像，实现原位分析。

### 方法

提出一种新的自监督机器学习去噪流程，整合盲点卷积神经网络，并包含预处理和后处理步骤，如漂移校正和低通滤波。

### 主要发现

该模型在降噪和对比度增强方面优于各种其他机器学习和非机器学习去噪方法，提高了原子特征的可视清晰度。模型比基于U-Net的机器学习模型快得多，表现出优秀的分布外泛化能力，计算推理速度达到每幅图像毫秒级。

### 结论

所提出的模型计算效率高，推理速度快，适用于原位HRTEM实验，能够实时处理高分辨率透射电子显微镜图像。

### 翻译

高分辨率透射电子显微镜对于观察材料在埃尺度上的结构和形态演化至关重要，但电子束可能会改变这些过程。基于CMOS的直接电子探测器在电子计数模式下工作可以显著减少电子剂量。然而，这会导致图像信噪比低，需要帧积分，牺牲了时间分辨率。最近开发的几种机器学习模型已成功用于高分辨率透射电子显微镜图像去噪。但这些模型通常计算成本高，在GPU上的推理速度落后于先进探测器的成像速度，阻碍了原位分析。此外，这些去噪模型在偏离训练数据集的成像条件下的性能尚未得到评估。为弥补这些差距，我们提出了一种专门针对时间序列高分辨率透射电子显微镜图像的新型自监督机器学习去噪流程。该流程整合了盲点卷积神经网络，包括漂移校正和低通滤波等预处理和后处理步骤。结果表明，我们的模型在降噪和对比度增强方面优于各种其他机器学习和非机器学习去噪方法，提高了原子特征的可视清晰度。此外，该模型比基于U-Net的机器学习模型快得多，并表现出优秀的分布外泛化能力。模型的计算推理速度达到每幅图像毫秒级，使其适用于原位高分辨率透射电子显微镜实验。


### 论文摘要

High-resolution transmission electron microscopy (HRTEM) is crucial for observing material's structural and morphological evolution at Angstrom scales, but the electron beam can alter these processes. Devices such as CMOS-based direct-electron detectors operating in electron-counting mode can be utilized to substantially reduce the electron dosage. However, the resulting images often lead to low signal-to-noise ratio, which requires frame integration that sacrifices temporal resolution. Several machine learning (ML) models have been recently developed to successfully denoise HRTEM images. Yet, these models are often computationally expensive and their inference speeds on GPUs are outpaced by the imaging speed of advanced detectors, precluding in situ analysis. Furthermore, the performance of these denoising models on datasets with imaging conditions that deviate from the training datasets have not been evaluated. To mitigate these gaps, we propose a new self-supervised ML denoising pipeline specifically designed for time-series HRTEM images. This pipeline integrates a blind-spot convolution neural network with pre-processing and post-processing steps including drift correction and low-pass filtering. Results demonstrate that our model outperforms various other ML and non-ML denoising methods in noise reduction and contrast enhancement, leading to improved visual clarity of atomic features. Additionally, the model is drastically faster than U-Net-based ML models and demonstrates excellent out-of-distribution generalization. The model's computational inference speed is in the order of milliseconds per image, rendering it suitable for application in in-situ HRTEM experiments.

---

## 122. Needle beams and structured space-time wavepackets

**论文链接:** [http://arxiv.org/abs/2512.03993v1](http://arxiv.org/abs/2512.03993v1)

**作者:** Ruediger Grunwald, Martin Bock

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文介绍了关于针束和时空波包的最新研究，探讨了它们的特性、组成和应用前景。

### 背景

关于针束和时空波包的研究进展

### 目的

探索STWPs的特性及其潜在应用

### 方法

在简单模型中将STWPs解释为差分针束组成，并通过模拟进行研究

### 主要发现

STWPs能在扩展距离上保持稳定的空间和时间局域化；脉冲贝塞尔状针束比空谱整形高斯聚焦束有更高的光谱和时间均匀性；飞秒针束阵列干涉导致时空非衍射自成像

### 结论

STWPs在高频切换、结合光学系统、轨道角动量生成和自扭矩等新兴研究领域有应用前景

### 翻译

关于针束和时空波包的最新研究被呈现。准非衍射STWPs在扩展距离上传播时保持稳定的空间和时间局域化。在简单模型中，STWPs被解释为由差分针束组成。模拟表明，脉冲贝塞尔状针束相比空谱整形的高斯聚焦束可以达到更高的光谱和时间均匀性。飞秒针束阵列的干涉导致空间和时间的非衍射自成像。STWPs的高速切换、结合光学系统、轨道角动量生成与自扭矩以及其他新兴研究领域被提及


### 论文摘要

Recent research on needle beams and space-time wavepackets (STWPs) is presented. Quasi-nondiffracting STWPs propagate at stable spatial and temporal localization over extended distances. In a simple model, STWPs are interpreted as being composed of differential needle beams. Simulations indicate that pulsed Bessel-like needle beams can reach higher spectral and temporal homogeneity compared to spatio-spectrally shaped focused Gaussian beams. The interference of femtosecond needle beam arrays leads to nondiffracting self-imaging in space and time. High-speed switching of STWPs with combined optical systems, orbital angular momentum generation with self-torque and other emerging fields of research are addressed

---

## 123. DIQ-H: Evaluating Hallucination Persistence in VLMs Under Temporal Visual Degradation

**论文链接:** [http://arxiv.org/abs/2512.03992v1](http://arxiv.org/abs/2512.03992v1)

**作者:** Zexin Lin, Hawen Wan, Yebin Zhong, Xiaoqiang

**发布时间:** 2025-12-03

### GPT解析

### 总结

研究团队提出了DIQ-H基准测试，用于评估视觉语言模型在时间序列动态视觉退化下的鲁棒性，通过物理损坏和多轮问答任务测量模型性能，并提出了不确定性引导迭代细化方法提升标注质量。

### 背景

视觉语言模型在自动驾驶等安全关键应用中需要处理不完美条件下的连续视觉流，但现有基准测试专注于静态高质量图像，忽略了时间退化和错误传播等关键故障模式。

### 目的

引入DIQ-H作为首个评估时间序列中动态视觉退化下视觉语言模型鲁棒性的基准测试。

### 方法

DIQ-H应用基于物理的损坏（包括运动模糊、传感器噪声和压缩伪影），通过多轮问答任务测量幻觉持久性、错误恢复和时间一致性；提出不确定性引导迭代细化方法，使用轻量级VLM和不确定性过滤生成可靠的伪真实标注，实现15.3%准确率提升。

### 主要发现

在16个最先进VLM的实验中发现了显著的鲁棒性差距；即使GPT-4o等先进模型也只有78.5%的恢复率；开源模型在时间一致性方面表现不佳，低于60%。

### 结论

DIQ-H为评估现实部署中视觉语言模型的可靠性提供了全面平台。

### 翻译

视觉语言模型部署在自动驾驶等安全关键应用中时，必须在不完美条件下处理连续视觉流。然而，现有基准测试专注于静态、高质量图像，忽略了时间退化和错误传播，这些是关键的故障模式，其中瞬时的视觉损坏会导致幻觉持续到后续帧。我们引入了DIQ-H，这是首个用于评估时间序列中动态视觉退化下视觉语言模型鲁棒性的基准测试。DIQ-H应用基于物理的损坏，包括运动模糊、传感器噪声和压缩伪影，并通过多轮问答任务测量幻觉持久性、错误恢复和时间一致性。为实现可扩展的标注，我们提出了不确定性引导迭代细化方法，使用带有不确定性过滤的轻量级VLM生成可靠的伪真实标注，实现了15.3%的准确率提升。在16个最先进视觉语言模型上的实验揭示了巨大的鲁棒性差距：即使GPT-4o等先进模型也只有78.5%的恢复率，而开源模型在时间一致性方面表现不佳，低于60%。DIQ-H为评估现实部署中视觉语言模型的可靠性提供了全面平台。


### 论文摘要

Vision-Language Models (VLMs) deployed in safety-critical applications such as autonomous driving must handle continuous visual streams under imperfect conditions. However, existing benchmarks focus on static, high-quality images and ignore temporal degradation and error propagation, which are critical failure modes where transient visual corruption induces hallucinations that persist across subsequent frames. We introduce DIQ-H, the first benchmark for evaluating VLM robustness under dynamic visual degradation in temporal sequences. DIQ-H applies physics-based corruptions including motion blur, sensor noise, and compression artifacts, and measures hallucination persistence, error recovery, and temporal consistency through multi-turn question-answering tasks. To enable scalable annotation, we propose Uncertainty-Guided Iterative Refinement (UIR), which generates reliable pseudo-ground-truth using lightweight VLMs with uncertainty filtering, achieving a 15.3 percent accuracy improvement. Experiments on 16 state-of-the-art VLMs reveal substantial robustness gaps: even advanced models such as GPT-4o achieve only a 78.5 percent recovery rate, while open-source models struggle with temporal consistency at less than 60 percent. DIQ-H provides a comprehensive platform for evaluating VLM reliability in real-world deployments.

---

## 124. 论文ID: 2512.03918v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03918v1.json'

---

## 125. Zero-Shot Video Translation and Editing with Frame Spatial-Temporal Correspondence

**论文链接:** [http://arxiv.org/abs/2512.03905v1](http://arxiv.org/abs/2512.03905v1)

**作者:** Shuai Yang, Junxin Lin, Yifan Zhou, Ziwei Liu, Chen Change Loy

**发布时间:** 2025-12-03

**备注:** Code: https://github.com/Sunnycookies/FRESCO-v2, Project: https://williamyang1991.github.io/projects/FRESCOv2/

### GPT解析

### 总结

本文提出了一种名为FRESCO的新方法，用于解决视频扩散模型中的时空一致性问题，通过结合帧内和帧间对应关系形成更强大的时空约束，显著提升了生成视频的视觉连贯性。

### 背景

文本到图像扩散模型的显著成功促使研究人员广泛探索其在视频应用中的潜力。零样本技术旨在无需进一步模型训练的情况下将图像扩散模型适应于视频应用。

### 目的

解决现有零样本方法中通过注意力机制整合帧间对应关系时，软约束不足以识别有效特征而导致的时间不一致问题。

### 方法

提出FRESCO方法，整合帧内对应关系与帧间对应关系，形成更强大的时空约束，确保帧间语义相似内容的一致性转换，并通过显式优化特征而不仅仅是注意力指导来实现高时空一致性。

### 主要发现

FRESCO方法能够实现高时空一致性并显著增强操作视频的视觉连贯性，在视频到视频翻译和文本引导视频编辑两个零样本任务上表现出色。

### 结论

FRESCO框架能够生成高质量、连贯的视频，相比当前零样本方法有显著进步，为视频扩散模型提供了新的解决方案。

### 翻译

文本到图像扩散模型的显著成功促使研究人员广泛探索其在视频应用中的潜力。零样本技术旨在无需进一步模型训练的情况下将图像扩散模型适应于视频。最近的方法主要强调将帧间对应关系整合到注意力机制中。然而，用于识别要关注的有效特征的软约束不足，可能导致时间不一致性。在本文中，我们提出了FRESCO，它将帧内对应关系与帧间对应关系相结合，形成更强大的时空约束。这种增强确保了帧间语义相似内容的一致性转换。我们的方法超越了注意力指导，通过显式优化特征实现了与输入视频的高时空一致性，显著增强了操作视频的视觉连贯性。我们在视频到视频翻译和文本引导视频编辑两个零样本任务上验证了FRESCO的适应性。全面的实验证明了我们框架在生成高质量、连贯视频方面的有效性，突显了相比当前零样本方法的显著进步。


### 论文摘要

The remarkable success in text-to-image diffusion models has motivated extensive investigation of their potential for video applications. Zero-shot techniques aim to adapt image diffusion models for videos without requiring further model training. Recent methods largely emphasize integrating inter-frame correspondence into attention mechanisms. However, the soft constraint applied to identify the valid features to attend is insufficient, which could lead to temporal inconsistency. In this paper, we present FRESCO, which integrates intra-frame correspondence with inter-frame correspondence to formulate a more robust spatial-temporal constraint. This enhancement ensures a consistent transformation of semantically similar content between frames. Our method goes beyond attention guidance to explicitly optimize features, achieving high spatial-temporal consistency with the input video, significantly enhancing the visual coherence of manipulated videos. We verify FRESCO adaptations on two zero-shot tasks of video-to-video translation and text-guided video editing. Comprehensive experiments demonstrate the effectiveness of our framework in generating high-quality, coherent videos, highlighting a significant advance over current zero-shot methods.

---

## 126. Generating a Contact Matrix for Aged Care Settings in Australia: an agent-based model study

**论文链接:** [http://arxiv.org/abs/2512.03866v1](http://arxiv.org/abs/2512.03866v1)

**作者:** Haley Stone, C. Raina MacIntyre, Mohana Kunasekaran, Chris Poulos, David Heslop

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究开发了一个基于智能体的模型来模拟养老院中工作人员和居民之间的互动，分析了不同护理级别居民与不同班次工作人员之间的接触模式，并评估了疫苗接种对传播的影响。

### 背景

养老院是高风险的封闭环境，了解工作人员和居民之间的接触模式对于制定有效的感染控制策略至关重要。

### 目的

开发一个基于智能体的模型来模拟养老院中工作人员和居民之间的互动，分析接触模式，并评估感染风险和疫苗接种效果。

### 方法

使用基于智能体的模型模拟养老院中的互动，通过空间阈值(1.5米和3米)和累积持续时间定义接触，生成接触矩阵，并使用基于泊松的回归建模进行分析。还集成了一个空气传播模块来评估感染风险。

### 主要发现

低护理和中等护理水平的居民与工作人员接触频率最高，特别是与早班和下午班工作人员；高护理居民和夜班工作人员接触较少；接触率因护理级别和班次而显著不同；高风险接触在结构化日常活动期间呈聚集性；在高接触班次和中等护理居民中感染风险最高；疫苗接种可减少高达68%的传播，当工作人员和居民都接种疫苗时效果最显著。

### 结论

在养老院中考虑接触异质性非常重要，基于智能体的模型对于评估高风险封闭环境中的针对性感染控制策略具有实用价值。

### 翻译

本研究提出了一个基于智能体的模型，用于模拟合成养老院中工作人员和居民的互动，捕捉三个班次工作人员和不同护理级别居民的移动、任务执行和基于接近度的接触事件。接触通过空间阈值(1.5米和3米)和累积持续时间定义，能够生成详细的接触矩阵。模拟结果显示，低护理和中等护理水平的居民经历了最高频率的互动，特别是与早班和下午班的工作人员，而高护理水平的居民和夜班工作人员的接触明显较少。接触率因护理级别和班次而有显著差异，通过基于泊松的回归建模得到证实。时间分析显示，在结构化的日常活动期间，高风险接触呈聚集性，特别是公共区域和护理活动。结合空气传播模块(以一名感染工作人员为起点)表明，在高接触班次和中等护理居民中感染风险最高。疫苗接种情景可将预测的传播减少高达68%，当工作人员和居民都接种疫苗时效果最显著。这些发现强调了在养老院中考虑接触异质性的重要性，并展示了基于智能体的模型在评估高风险封闭环境中针对性感染控制策略方面的效用。


### 论文摘要

This study presents an agent-based model (ABM) developed to simulate staff and resident interactions within a synthetic aged care facility, capturing movement, task execution, and proximity-based contact events across three staff shifts and varying levels of resident care. Contacts were defined by spatial thresholds (1.5 m and 3 m) and cumulative duration, enabling the generation of detailed contact matrices. Simulation results showed that low and medium care residents experienced the highest frequency of interactions, particularly with staff on morning and afternoon shifts, while high care residents and night staff had substantially fewer contacts. Contact rates varied significantly by care level and shift, confirmed through Poisson-based regression modelling. Temporal analyses revealed clustering of high-risk contacts during structured daily routines, especially communal and care activities. An integrated airborne transmission module, seeded with a single infectious staff member, demonstrated that infection risk was highest during high-contact shifts and among medium care residents. Vaccination scenarios reduced predicted transmission by up to 68\%, with the greatest impact observed when both staff and residents were vaccinated. These findings highlight the importance of accounting for contact heterogeneity in aged care and demonstrate the utility of ABMs for evaluating targeted infection control strategies in high-risk, enclosed environments.

---

## 127. A decay-adjusted spatio-temporal model to account for the impact of mass drug administration on neglected tropical disease prevalence

**论文链接:** [http://arxiv.org/abs/2512.03760v1](http://arxiv.org/abs/2512.03760v1)

**作者:** Emanuele Giorgi, Claudio Fronterre, Peter J. Diggle

**发布时间:** 2025-12-03

**备注:** Under review

### GPT解析

### 总结

本研究提出了一种衰减调整时空(DAST)模型，用于监测控制被忽视热带病的群体药物 administrations(MDA)项目的有效性，为从稀疏调查数据中估计干预效果提供了新方法。

### 背景

患病率调查通常用于监测控制被忽视热带病(NTDs)的群体药物 administrations(MDA)项目的有效性。

### 目的

提出衰减调整时空(DAST)模型，明确考虑MDA对NTDs患病率的时间变化影响，提供灵活且可解释的框架来估计干预效果。

### 方法

开发衰减调整时空(DAST)模型，并使用土源性蠕虫和淋巴丝虫病的案例研究进行验证。

### 主要发现

当目标包括量化MDA影响和支持短期项目预测时，DAST是标准地质统计模型的实用替代方案。

### 结论

在可用数据稀疏的情况下，应采用数据驱动的简约性而非复杂性，讨论了模型扩展和可识别性挑战。

### 翻译

患病率调查通常用于监测控制被忽视热带病(NTDs)的群体药物 administrations(MDA)项目的有效性。我们提出了一个衰减调整时空(DAST)模型，明确考虑了MDA对NTDs患病率的时间变化影响，为从稀疏调查数据中估计干预效果提供了一个灵活且可解释的框架。通过土源性蠕虫和淋巴丝虫病的案例研究，我们表明当目标包括量化MDA影响和支持短期项目预测时，DAST是标准地质统计模型的一个实用替代方案。我们还讨论了扩展和可识别性挑战，提倡在可用数据太稀疏而无法支持高度参数化模型的估计的情况下，应采用数据驱动的简约性而非复杂性。


### 论文摘要

Prevalence surveys are routinely used to monitor the effectiveness of mass drug administration (MDA) programmes for controlling neglected tropical diseases (NTDs). We propose a decay-adjusted spatio-temporal (DAST) model that explicitly accounts for the time-varying impact of MDA on NTD prevalence, providing a flexible and interpretable framework for estimating intervention effects from sparse survey data. Using case studies on soil-transmitted helminths and lymphatic filariasis, we show that DAST offers a practical alternative to standard geostatistical models when the objective includes quantifying MDA impact and supporting short-term programmatic forecasting. We also discuss extensions and identifiability challenges, advocating for data-driven parsimony over complexity in settings where the available data are too sparse to support the estimation of highly parameterised models.

---

## 128. A BTR-Based Approach for Detection of Infrared Small Targets

**论文链接:** [http://arxiv.org/abs/2512.03752v1](http://arxiv.org/abs/2512.03752v1)

**作者:** Ke-Xin Li

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种基于双边张量环分解的红外小目标检测模型BTR-ISTD，通过将数据重构为四阶张量，有效解决了现有低秩稀疏方法在处理低对比度小目标和复杂动态背景时的高计算复杂度问题。

### 背景

红外小目标检测在军事侦察和防空系统中扮演着重要角色，但现有基于低秩稀疏的方法在处理低对比度小目标和包含类目标干扰的复杂动态背景时仍面临高计算复杂度的挑战。

### 目的

解决现有低秩稀疏方法在处理低对比度小目标和复杂动态背景时的高计算复杂度问题，提高红外小目标检测的性能。

### 方法

将数据重构为四阶张量，提出基于双边张量环分解的红外小目标检测模型BTR-ISTD。该方法从图像序列构建四维红外张量，利用BTR分解区分弱空间相关性和强时空块相关性，同时捕获这两个组成部分间的相互作用，并在近端交替最小化框架下高效求解。

### 主要发现

实验结果表明，所提出的BTR-ISTD方法在检测精度、背景抑制能力和计算速度方面均优于几种现有最先进的方法。

### 结论

基于双边张量环分解的红外小目标检测模型能有效解决现有方法在高计算复杂度方面的问题，并在检测性能上表现优异。

### 翻译

红外小目标检测在军事侦察和防空系统中起着至关重要的作用。然而，现有的基于低秩稀疏的方法在处理低对比度小目标和包含类目标干扰的复杂动态背景时，仍然面临高计算复杂度的问题。为解决这一局限，我们将数据重构为四阶张量，并提出了一种基于双边张量环分解的新型红外小目标检测模型，称为BTR-ISTD。该方法首先从图像序列构建四维红外张量，然后利用BTR分解有效区分弱空间相关性和强时空块相关性，同时捕获这两个组成部分之间的相互作用。该模型在近端交替最小化框架下得到高效求解。实验结果表明，所提出的方法在检测精度、背景抑制能力和计算速度方面优于几种最先进的方法。


### 论文摘要

Infrared small target detection plays a crucial role in military reconnaissance and air defense systems. However,existing low-rank sparse based methods still face high computational complexity when dealing with low-contrast small targets and complex dynamic backgrounds mixed with target-like interference. To address this limitation, we reconstruct the data into a fourth-order tensor and propose a new infrared small target detection model based on bilateral tensor ring decomposition, called BTR-ISTD. The approach begins by constructing a four-dimensional infrared tensor from an image sequence, then utilizes BTR decomposition to effectively distinguish weak spatial correlations from strong temporal-patch correlations while simultaneously capturing interactions between these two components. This model is efficiently solved under the proximal alternating minimization (PAM) framework. Experimental results demonstrate that the proposed approach outperforms several state-of-the-art methods in terms of detection accuracy, background suppression capability, and computational speed.

---

## 129. DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue

**论文链接:** [http://arxiv.org/abs/2512.03704v1](http://arxiv.org/abs/2512.03704v1)

**作者:** Yijun Liao

**发布时间:** 2025-12-03

**备注:** 22 pages, 2 figures, 13 tables. Code available at https://github.com/lyj20071013/DZ-TDPO

### GPT解析

### 总结

该研究针对长对话系统中的状态惯性问题，提出DZ-TDPO框架，通过动态KL约束和时间注意力偏差解决用户意图与历史上下文冲突，实现了高性能与低困惑度的平衡。

### 背景

长上下文对话系统面临状态惯性问题，即静态约束阻止模型解决 evolving 用户意图与已建立历史上下文之间的冲突。

### 目的

解决长对话系统中的状态惯性问题，使模型能够更好地处理 evolving 用户意图与历史上下文之间的冲突。

### 方法

提出DZ-TDPO，一种非破坏性对齐框架，结合了冲突感知的动态KL约束和可学习的时间注意力偏差。

### 主要发现

1) 在MSC数据集上实现最先进胜率(Phi-3.5上86.2%)；2) 保持强大的零样本泛化能力；3) 揭示'能力-稳定性权衡'：小模型需承担'对齐税'克服历史惯性，而大模型(Qwen2.5-7B)实现近乎完美对齐(99.4%胜率)且困惑度开销可忽略；4) 确认可通过精确注意力调节而非破坏性权重更新缓解TAI，保持模型通用能力。

### 结论

DZ-TDPO通过精确的注意力调节而非破坏性权重更新来缓解状态惯性问题，实现了高性能与低困惑度的平衡，同时保持了模型的通用能力。

### 翻译

长上下文对话系统遭受状态惯性的困扰，静态约束阻止模型解决 evolving 用户意图与已建立历史上下文之间的冲突。为此，我们提出了DZ-TDPO，一种非破坏性对齐框架，结合了冲突感知的动态KL约束和可学习的时间注意力偏差。在多轮对话(MSC)数据集上的实验表明，DZ-TDPO在Phi-3.5上实现了最先进的胜率(86.2%)，同时保持了强大的零样本泛化能力。关键的是，我们的规模分析揭示了'能力-稳定性权衡'：虽然较小模型需要承担'对齐税'(困惑度激增)来克服历史惯性，但较大的Qwen2.5-7B模型实现了近乎完美的对齐(99.4%胜率)，困惑度开销可以忽略不计。这证实了TAI可以通过精确的注意力调节而非破坏性权重更新来缓解，从而在不同模型规模上保持通用能力(MMLU)。代码和数据可在以下网址获取：https://github.com/lyj20071013/DZ-TDPO


### 论文摘要

Long-context dialogue systems suffer from State Inertia, where static constraints prevent models from resolving conflicts between evolving user intents and established historical context. To address this, we propose DZ-TDPO, a non-destructive alignment framework that synergizes conflict-aware dynamic KL constraints with a learnable temporal attention bias. Experiments on the Multi-Session Chat (MSC) dataset demonstrate that DZ-TDPO achieves state-of-the-art win rates (86.2% on Phi-3.5) while maintaining robust zero-shot generalization. Crucially, our scaling analysis reveals a "Capacity-Stability Trade-off": while smaller models incur an "alignment tax" (perplexity surge) to overcome historical inertia, the larger Qwen2.5-7B model achieves near-perfect alignment (99.4% win rate) with negligible perplexity overhead. This confirms that TAI can be alleviated via precise attention regulation rather than destructive weight updates, preserving general capabilities (MMLU) across model scales. Code and data are available: https://github.com/lyj20071013/DZ-TDPO

---

## 130. Conditional updates of neural network weights for increased out of training performance

**论文链接:** [http://arxiv.org/abs/2512.03653v1](http://arxiv.org/abs/2512.03653v1)

**作者:** Jan Saynisch-Wagner, Saran Rajendran Sari

**发布时间:** 2025-12-03

### GPT解析

### 总结

这项研究提出了一种方法，用于增强神经网络在训练数据与应用数据不相似情况下的性能，通过三个步骤实现：重新训练神经网络并记录权重异常、建立预测因子与权重异常的回归关系、外推权重到应用数据。该方法在气候科学的三个用例中成功实现了时间、空间和跨域外推。

### 背景

神经网络在训练数据与应用数据不相似的情况下（如分布外问题、模式和 regime 变化）面临性能挑战。

### 目的

提高神经网络在训练数据与应用数据不相似情况下的性能。

### 方法

通过三个步骤实现：1) 重新训练神经网络朝向训练数据集的合理子集并记录权重异常；2) 选择合理的预测因子并推导预测因子与权重异常之间的回归关系；3) 外推权重和神经网络到应用数据。

### 主要发现

该方法在气候科学的三个用例中成功实现了神经网络的时间、空间和跨域外推。

### 结论

所提出的方法能够有效处理神经网络在训练数据与应用数据不相似情况下的性能问题。

### 翻译

这项研究提出了一种方法，用于增强神经网络在训练数据与应用数据不太相似情况下的性能，例如分布外问题、模式和 regime 变化。该方法包含三个主要步骤：1) 重新训练神经网络朝向训练数据集的合理子集，并记录权重异常。2) 选择合理的预测因子，推导预测因子与权重异常之间的回归关系。3) 外推权重，从而外推神经网络到应用数据。研究人员在气候科学的三个用例中展示了并讨论了这种方法，包括神经网络在时间、空间和跨域方面的成功外推。


### 论文摘要

This study proposes a method to enhance neural network performance when training data and application data are not very similar, e.g., out of distribution problems, as well as pattern and regime shifts. The method consists of three main steps: 1) Retrain the neural network towards reasonable subsets of the training data set and note down the resulting weight anomalies. 2) Choose reasonable predictors and derive a regression between the predictors and the weight anomalies. 3) Extrapolate the weights, and thereby the neural network, to the application data. We show and discuss this method in three use cases from the climate sciences, which include successful temporal, spatial and cross-domain extrapolations of neural networks.

---

## 131. 论文ID: 2512.03639v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03639v1.json'

---

## 132. Beyond Boundary Frames: Audio-Visual Semantic Guidance for Context-Aware Video Interpolation

**论文链接:** [http://arxiv.org/abs/2512.03590v1](http://arxiv.org/abs/2512.03590v1)

**作者:** Yuchen Deng, Xiuyang Wu, Hai-Tao Zheng, Jie Wang, Feidiao Yang, Yuxing Han

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究提出了BBF（Beyond Boundary Frames）框架，一个具有上下文感知能力的视频帧插值方法，可通过音频/视觉语义进行指导，解决了现有方法在处理复杂非线性运动模式时难以产生清晰、时间一致帧的问题。

### 背景

处理快速、复杂且高度非线性的运动模式一直是视频帧插值的挑战。虽然最近的基于扩散的方法改进了传统的光流方法，但它们仍然难以覆盖多样化的应用场景，在音视频同步插值等细粒度运动任务中往往无法生成清晰、时间一致的帧。

### 目的

开发一个能够处理多种条件模态（文本、音频、图像和视频）的视频帧插值框架，提高在通用插值和音视频同步插值任务上的性能，建立统一的多通道条件视频帧插值框架。

### 方法

1) 增强插值模型的输入设计，使其能灵活处理多种条件模态；2) 提出解耦的多模态融合机制，将不同条件信号顺序注入到DiT主干网络中；3) 采用渐进式多阶段训练范式，使用开始-结束帧差异嵌入动态调整数据采样和损失权重。

### 主要发现

BBF在通用插值和音视频同步插值任务上都优于专门的最先进方法，证明了其在处理多样化运动模式方面的优越性，并成功建立了在协调多通道条件下视频帧插值的统一框架。

### 结论

BBF框架通过增强输入设计、解耦多模态融合机制和渐进式多阶段训练范式，有效解决了现有方法在处理复杂非线性运动模式时的局限性，为视频帧插值提供了新的统一解决方案。

### 翻译

处理快速、复杂且高度非线性的运动模式长期以来一直是视频帧插值的挑战。尽管最近的基于扩散的方法改进了传统的基于光流的方法，但它们仍然难以覆盖多样化的应用场景，并且在音视频同步插值等细粒度运动任务中往往无法产生清晰、时间一致的帧。为了解决这些局限性，我们引入了BBF（Beyond Boundary Frames），一个具有上下文感知能力的视频帧插值框架，可通过音频/视觉语义进行指导。首先，我们增强了插值模型的输入设计，使其能够灵活处理多种条件模态，包括文本、音频、图像和视频。其次，我们提出了一种解耦的多模态融合机制，将不同的条件信号顺序注入到DiT主干网络中。最后，为了保持基础模型的生成能力，我们采用渐进式多阶段训练范式，其中开始-结束帧差异嵌入用于动态调整数据采样和损失权重。广泛的实验结果表明，BBF在通用插值和音视频同步插值任务上都优于专门的最先进方法，建立了在协调多通道条件下视频帧插值的统一框架。


### 论文摘要

Handling fast, complex, and highly non-linear motion patterns has long posed challenges for video frame interpolation. Although recent diffusion-based approaches improve upon traditional optical-flow-based methods, they still struggle to cover diverse application scenarios and often fail to produce sharp, temporally consistent frames in fine-grained motion tasks such as audio-visual synchronized interpolation. To address these limitations, we introduce BBF (Beyond Boundary Frames), a context-aware video frame interpolation framework, which could be guided by audio/visual semantics. First, we enhance the input design of the interpolation model so that it can flexibly handle multiple conditional modalities, including text, audio, images, and video. Second, we propose a decoupled multimodal fusion mechanism that sequentially injects different conditional signals into a DiT backbone. Finally, to maintain the generation abilities of the foundation model, we adopt a progressive multi-stage training paradigm, where the start-end frame difference embedding is used to dynamically adjust both the data sampling and the loss weighting. Extensive experimental results demonstrate that BBF outperforms specialized state-of-the-art methods on both generic interpolation and audio-visual synchronized interpolation tasks, establishing a unified framework for video frame interpolation under coordinated multi-channel conditioning.

---

## 133. A Coupled IMEX Domain Decomposition Method for High-Order Time Integration of the ES-BGK Model of the Boltzmann Equation

**论文链接:** [http://arxiv.org/abs/2512.03586v1](http://arxiv.org/abs/2512.03586v1)

**作者:** Domenico Caparello, Tommaso Tenna

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种针对Boltzmann方程ES-BGK模型的高阶区域分解方法，能够动态识别平衡和非平衡区域，并实现宏观求解器与动力学求解器之间的耦合，保持高精度计算效率。

### 背景

在计算流体力学中，Boltzmann方程的ES-BGK模型用于模拟气体动力学行为，但在不同区域可能需要不同的求解方法以提高计算效率。

### 目的

开发一种能够动态切换使用欧拉方程和ES-BGK模型的高阶区域分解方法，同时保持整体时间精度和计算效率。

### 方法

提出了一种耦合IMEX方法，在分解的子区域和求解器之间实现切换，平衡区域使用欧拉方程，非平衡区域使用ES-BGK模型，并确保两种求解器之间的耦合策略保持整体时间精度。

### 主要发现

通过二维空间中的多个数值模拟，验证了该方法具有鲁棒性，并实现了预期的高阶时间收敛性，同时保持了高精度和计算效率。

### 结论

该方法有效结合了宏观和微观求解器的优势，实现了高精度和计算效率的平衡，为复杂流体动力学问题提供了新的求解思路。

### 翻译

在本文中，我们针对Boltzmann方程的ES-BGK模型提出了一种高阶区域分解方法，该方法能够动态检测平衡和非平衡区域。我们的实现在流体处于平衡的区域自动切换到欧拉方程，在其他区域使用ES-BGK模型。本文解决的主要挑战是开发宏观求解器和动力学求解器之间的耦合策略，保持方案的整体时间精度。引入了一种跨分解子区域和求解器的耦合IMEX方法。这种方法基于耦合IMEX方法，能够实现高精度和计算效率。进行了二维空间中的多个数值模拟，以验证我们方法的鲁棒性和预期的高阶时间收敛性。


### 论文摘要

In this paper, we propose a high-order domain decomposition method for the ES-BGK model of the Boltzmann equation, which dynamically detects regions of equilibrium and non-equilibrium. Our implementation automatically switches between Euler equations in regions where the fluid is at equilibrium, and the ES-BGK model elsewhere. The main challenge addressed in this work is the development of a coupled strategy between the macroscopic and the kinetic solvers, which preserves the overall temporal order of accuracy of the scheme. A coupled IMEX method is introduced across decomposed subdomains and solvers. This approach is based on a coupled IMEX method and allows high accuracy and computational efficiency. Several numerical simulations in two space dimensions are performed, in order to validate the robustness of our approach and the expected temporal high-order convergence.

---

## 134. When, How Long and How Much? Interpretable Neural Networks for Time Series Regression by Learning to Mask and Aggregate

**论文链接:** [http://arxiv.org/abs/2512.03578v1](http://arxiv.org/abs/2512.03578v1)

**作者:** Florent Forest, Amaury Wei, Olga Fink

**发布时间:** 2025-12-03

**备注:** 12 pages, 5 figures, 4 tables

### GPT解析

### 总结

MAGNETS是一种新的内在可解释神经架构，用于时间序列外回归(TSER)，能够学习人类可理解的概念，不需要注释，同时提供透明度和可解释性。

### 背景

时间序列外回归(TSER)是从输入时间序列预测连续目标变量的任务，应用于医疗保健、金融、环境监测和工程等领域。虽然最先进的TSER模型能实现强大的预测性能，但通常作为黑盒操作，难以理解哪些时间模式驱动决策。事后的可解释性技术往往产生粗糙、嘈杂或不稳定的解释。现有的内在可解释方法存在局限性：需要概念本身的明确监督、无法捕获时间序列特征间的交互、缺乏对复杂时间模式的表达能力、难以扩展到高维多元数据。

### 目的

解决现有TSER可解释方法的局限性，提出一种内在可解释的神经架构，使模型能够学习人类可理解的概念并提供透明的决策过程。

### 方法

提出MAGNETS(时间序列的掩码和聚合网络)，一种内在可解释的神经架构。它学习紧凑的人类可理解概念集，不需要任何注释。每个概念对应于对选定输入特征的基于掩码的聚合，明确揭示哪些特征驱动预测以及它们在序列中的重要性。预测通过透明的加性结构组合这些学习到的概念形成。

### 主要发现

现有内在可解释方法存在多种局限性：需要概念本身的明确监督、通常无法捕获时间序列特征之间的交互、缺乏对复杂时间模式的表达能力、难以扩展到高维多元数据。MAGNETS能够克服这些限制，提供更好的可解释性。

### 结论

MAGNETS是一种有效的内在可解释神经架构，能够学习人类可理解的概念，不需要注释，同时明确显示哪些特征驱动预测以及它们在序列中的重要性，通过透明的加性结构形成预测，能够清晰洞察模型的决策过程。

### 翻译

时间序列外回归(TSER)指的是从输入时间序列预测连续目标变量的任务。它出现在许多领域，包括医疗保健、金融、环境监测和工程。在这些环境中，准确的预测和可信的推理都是必不可少的。尽管最先进的TSER模型实现了强大的预测性能，但它们通常作为黑盒操作，使得难以理解哪些时间模式驱动了它们的决策。事后可解释性技术，如特征归因，旨在解释模型如何得出其预测，但通常产生粗糙、嘈杂或不稳定的解释。最近，基于概念、加性分解或符号回归的内在可解释方法已成为有前途的替代方案。然而，这些方法仍然存在局限性：它们需要概念本身的明确监督，通常无法捕获时间序列特征之间的交互，缺乏对复杂时间模式的表现力，并且难以扩展到高维多元数据。为了解决这些局限性，我们提出了MAGNETS(时间序列的掩码和聚合网络)，一种用于TSER的内在可解释神经架构。MAGNETS学习紧凑的人类可理解概念集，不需要任何注释。每个概念对应于对选定输入特征的基于掩码的聚合，明确揭示哪些特征驱动预测以及它们在序列中的重要性。预测通过这些学习概念的透明加性结构形成，使能够清晰洞察模型的决策过程。


### 论文摘要

Time series extrinsic regression (TSER) refers to the task of predicting a continuous target variable from an input time series. It appears in many domains, including healthcare, finance, environmental monitoring, and engineering. In these settings, accurate predictions and trustworthy reasoning are both essential. Although state-of-the-art TSER models achieve strong predictive performance, they typically operate as black boxes, making it difficult to understand which temporal patterns drive their decisions. Post-hoc interpretability techniques, such as feature attribution, aim to to explain how the model arrives at its predictions, but often produce coarse, noisy, or unstable explanations. Recently, inherently interpretable approaches based on concepts, additive decompositions, or symbolic regression, have emerged as promising alternatives. However, these approaches remain limited: they require explicit supervision on the concepts themselves, often cannot capture interactions between time-series features, lack expressiveness for complex temporal patterns, and struggle to scale to high-dimensional multivariate data.   To address these limitations, we propose MAGNETS (Mask-and-AGgregate NEtwork for Time Series), an inherently interpretable neural architecture for TSER. MAGNETS learns a compact set of human-understandable concepts without requiring any annotations. Each concept corresponds to a learned, mask-based aggregation over selected input features, explicitly revealing both which features drive predictions and when they matter in the sequence. Predictions are formed as combinations of these learned concepts through a transparent, additive structure, enabling clear insight into the model's decision process.

---

## 135. CookAnything: A Framework for Flexible and Consistent Multi-Step Recipe Image Generation

**论文链接:** [http://arxiv.org/abs/2512.03540v1](http://arxiv.org/abs/2512.03540v1)

**作者:** Ruoxuan Zhang, Bin Wen, Hongxia Xie, Yi Yao, Songhan Zuo, Jian-Yu Jiang-Lin, Hong-Han Shuai, Wen-Huang Cheng

**发布时间:** 2025-12-03

**DOI:** 10.1145/3746027.3755174

**备注:** Accepted by ACM Multimedia 2025

### GPT解析

### 总结

CookAnything是一个灵活且一致的基于扩散的框架，可以从任意长度的文本烹饪指令生成连贯、语义不同的图像序列。该框架通过三个关键组件解决现有方法在处理食谱插图时的局限性。

### 背景

烹饪是一个连续且视觉基础的活动，每个步骤都有程序逻辑和视觉语义。当前扩散模型难以处理多步骤结构化场景如食谱插图，且现有方法无法适应食谱长度的自然变化，总是生成固定数量的图像。

### 目的

提出CookAnything框架，解决现有方法在处理食谱插图时的局限性，实现从任意长度文本烹饪指令生成连贯、语义不同的图像序列。

### 方法

框架引入三个关键组件：(1)步骤区域控制(SRC)：在单个去噪过程中将文本步骤与相应图像区域对齐；(2)灵活的RoPE：步骤感知的位置编码机制，增强时间一致性和空间多样性；(3)跨步骤一致性控制(CSCC)：在步骤间保持精细的成分一致性。

### 主要发现

在食谱插图基准测试中，CookAnything在基于训练和无需训练的设置中都优于现有方法。

### 结论

CookAnything框架支持复杂多步骤指令的可扩展、高质量视觉合成，在指导媒体和程序内容创作领域具有广泛的应用潜力。

### 翻译

烹饪是一个连续且视觉基础的活动，其中每个步骤如切、混合或煎炸都包含程序逻辑和视觉语义。虽然最近的扩散模型在文本到图像生成方面表现出强大的能力，但它们难以处理食谱插图等结构化多步骤场景。此外，当前食谱插图方法无法适应食谱长度的自然变化，无论实际指令结构如何，都生成固定数量的图像。为解决这些局限性，我们提出了CookAnything，这是一个灵活且一致的基于扩散的框架，可以从任意长度的文本烹饪指令生成连贯、语义不同的图像序列。该框架引入了三个关键组件：(1)步骤区域控制(SRC)，在单个去噪过程中将文本步骤与相应的图像区域对齐；(2)灵活的RoPE，一种步骤感知的位置编码机制，增强时间一致性和空间多样性；(3)跨步骤一致性控制(CSCC)，在步骤间保持精细的成分一致性。在食谱插图基准测试上的实验结果表明，CookAnything在基于训练和无需训练的设置中都优于现有方法。所提出的框架支持复杂多步骤指令的可扩展、高质量视觉合成，并在指导媒体和程序内容创作领域具有巨大的应用潜力。


### 论文摘要

Cooking is a sequential and visually grounded activity, where each step such as chopping, mixing, or frying carries both procedural logic and visual semantics. While recent diffusion models have shown strong capabilities in text-to-image generation, they struggle to handle structured multi-step scenarios like recipe illustration. Additionally, current recipe illustration methods are unable to adjust to the natural variability in recipe length, generating a fixed number of images regardless of the actual instructions structure. To address these limitations, we present CookAnything, a flexible and consistent diffusion-based framework that generates coherent, semantically distinct image sequences from textual cooking instructions of arbitrary length. The framework introduces three key components: (1) Step-wise Regional Control (SRC), which aligns textual steps with corresponding image regions within a single denoising process; (2) Flexible RoPE, a step-aware positional encoding mechanism that enhances both temporal coherence and spatial diversity; and (3) Cross-Step Consistency Control (CSCC), which maintains fine-grained ingredient consistency across steps. Experimental results on recipe illustration benchmarks show that CookAnything performs better than existing methods in training-based and training-free settings. The proposed framework supports scalable, high-quality visual synthesis of complex multi-step instructions and holds significant potential for broad applications in instructional media, and procedural content creation.

---

## 136. AdaPower: Specializing World Foundation Models for Predictive Manipulation

**论文链接:** [http://arxiv.org/abs/2512.03538v1](http://arxiv.org/abs/2512.03538v1)

**作者:** Yuhang Huang, Shilong Zou, Jiazhao Zhang, Xinwang Liu, Ruizhen Hu, Kai Xu

**发布时间:** 2025-12-03

### GPT解析

### 总结

AdaPower是一种轻量级适应框架，通过时空测试时训练和内存持久性两个组件，将通用世界模型转变为专业世界模型，实现了在机器人控制任务中无需重新训练策略就能显著提高任务成功率。

### 背景

世界模型具有出色的视觉动力学模拟能力，但在精确机器人控制中的应用受限于生成式真实感与控制精确性之间的差距。现有方法将世界模型作为合成数据生成器使用，但计算成本高且未充分利用预训练的VLA策略。

### 目的

开发一种轻量级适应框架，将通用世界模型转变为专业世界模型，以解决在精确机器人控制中的应用限制。

### 方法

提出AdaPower框架，包含两个核心组件：时空测试时训练（TS-TTT）用于推理时适应，内存持久性（MP）用于长程一致性。该框架集成在模型预测控制框架中，使适应后的世界模型能够赋能预训练的VLA策略。

### 主要发现

在LIBERO基准测试中，使用AdaPower框架的任务成功率提高了41%以上，无需重新训练策略，同时保持了计算效率和通用能力。

### 结论

AdaPower框架成功解决了世界模型在精确机器人控制中的应用限制，通过轻量级适应实现了显著的性能提升，同时保留了计算效率和通用能力。

### 翻译

世界模型（WFMs）提供了出色的视觉动力学模拟能力，然而它们在精确机器人控制中的应用仍然受限于生成式真实感与面向控制的精确性之间的差距。虽然现有方法使用世界模型作为合成数据生成器，但它们面临高计算成本和未充分利用预训练VLA策略的问题。我们介绍了AdaPower（Adapt and Empower），这是一个轻量级适应框架，通过两个新颖组件将通用世界模型转变为专业世界模型：用于推理时适应的时空测试时训练（TS-TTT）和用于长程一致性的内存持久性（MP）。集成在模型预测控制框架中，我们适应后的世界模型赋能了预训练的VLA策略，在LIBERO基准测试中实现了超过41%的任务成功率提升，无需重新训练策略，同时保持了计算效率和通用能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决世界基础模型（WFMs）在机器人控制应用中的局限性问题。WFMs虽然擅长生成视觉上合理的场景预测，但机器人控制需要基于动作的、精确可执行的动态预测，两者之间存在差距。这个问题很重要，因为它限制了WFMs在机器人操作领域的应用潜力，而机器人操作需要连续、细粒度的环境交互，对精确预测有高要求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法（如DreamGen）的局限性：计算成本高、适应周期长、未充分利用预训练策略。然后借鉴了多个现有技术：适配器技术（如ControlNet、LoRA）用于参数高效更新；测试时训练（TTT）技术用于推理时优化；世界模型用于模型预测控制（MPC）的方法。作者针对机器人控制场景的特殊需求，设计了两个关键组件：时空测试时训练（TS-TTT）和记忆持久性（MP）模块，以解决测试时分布变化和长时间一致性问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过轻量级适配器框架将通用WFMs转换为专业化的世界模型（SWM），直接提高预训练VLA的零样本能力，避免传统方法中高计算成本的合成数据生成和新策略训练。整体流程包括：1) 基于DiT架构的WFMs插入MP和TS-TTT模块，并将文本编码器替换为动作编码器；2) 只训练新添加层，保持预训练参数冻结；3) 将SWM集成到MPC框架中，预训练VLA生成候选动作序列，SWM进行滚动预测，奖励模型评估选择最优轨迹；4) 推理时TS-TTT进行自监督优化，MP通过交叉注意力保持历史上下文。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) AdaPower框架，通过轻量级模型适应而非合成数据生成利用WFMs；2) 时空测试时训练（TS-TTT），扩展传统TTT利用视频数据的时空低秩结构；3) 记忆持久性（MP）模块，通过交叉注意力保持历史信息确保长时间一致性；4) 协作MPC系统，SWM与预训练VLA协同工作。相比之前工作，AdaPower避免了高计算成本的数据生成，更充分利用预训练策略，针对视频动态和动作条件预测优化，并专注于将大型WFMs适配为控制就绪的专家模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AdaPower通过轻量级适配器框架将通用世界基础模型转化为专业化的预测模型，显著提升了预训练视觉-语言-行动模型在机器人操作任务中的零样本泛化能力，实现了超过40%的成功率提升，同时保持了计算效率和通用能力。'}


### 论文摘要

World Foundation Models (WFMs) offer remarkable visual dynamics simulation capabilities, yet their application to precise robotic control remains limited by the gap between generative realism and control-oriented precision. While existing approaches use WFMs as synthetic data generators, they suffer from high computational costs and underutilization of pre-trained VLA policies. We introduce \textbf{AdaPower} (\textbf{Ada}pt and Em\textbf{power}), a lightweight adaptation framework that transforms general-purpose WFMs into specialist world models through two novel components: Temporal-Spatial Test-Time Training (TS-TTT) for inference-time adaptation and Memory Persistence (MP) for long-horizon consistency. Integrated within a Model Predictive Control framework, our adapted world model empowers pre-trained VLAs, achieving over 41\% improvement in task success rates on LIBERO benchmarks without policy retraining, while preserving computational efficiency and generalist capabilities.

---

## 137. EEA: Exploration-Exploitation Agent for Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.03500v1](http://arxiv.org/abs/2512.03500v1)

**作者:** Te Yang, Xiangyu Zhu, Bo Wang, Quan Chen, Peng Jiang, Zhen Lei

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为EEA的新型视频智能体框架，通过语义引导的分层树搜索过程实现探索-利用平衡，有效解决了长视频理解中的信息覆盖不完整和计算效率低下的问题。

### 背景

长视频理解需要高效处理大量视觉数据以找到关键信息，但现有方法要么因密集预处理导致计算开销大，要么无法平衡探索与利用，导致信息覆盖不完整和效率低下。

### 目的

开发一种能够有效平衡探索和利用的视频理解框架，实现对长视频的高效处理和关键信息的准确提取。

### 方法

EEA框架通过自主发现和动态更新任务相关语义查询，收集匹配的视频帧作为语义锚点，在树搜索过程中优先探索语义相关帧同时确保未知段覆盖，并通过建模不确定性将VLMs的内在奖励与语义先验知识自适应结合。

### 主要发现

EEA在各种长视频基准上表现出优越的性能和计算效率，能够有效平衡探索和利用，实现对视频段的稳定而精确的评估。

### 结论

EEA框架为长视频理解提供了一种有效解决方案，通过语义引导的分层树搜索过程和自适应奖励机制，实现了信息覆盖完整性和计算效率的平衡。

### 翻译

长视频理解需要高效导航大量视觉数据以定位稀疏但关键的信息。当前长视频理解方法要么因密集预处理导致严重的计算开销，要么无法有效平衡探索和利用，导致信息覆盖不完整和效率低下。在这项工作中，我们引入了EEA，一种新颖的视频智能体框架，通过分层树搜索过程的语义指导实现探索-利用平衡。EEA自主发现并动态更新任务相关的语义查询，并将与这些查询密切匹配的视频帧收集为语义锚点。在树搜索过程中，EEA不是均匀扩展，而是优先探索语义相关的帧，同时确保在未知段内有足够的覆盖。此外，EEA通过明确建模不确定性，将视觉语言模型(VLMs)的内在奖励与语义先验知识自适应结合，从而实现对视频段的稳定而精确的评估。在各种长视频基准上的实验验证了我们提出方法的优越性能和计算效率。


### 论文摘要

Long-form video understanding requires efficient navigation of extensive visual data to pinpoint sparse yet critical information. Current approaches to longform video understanding either suffer from severe computational overhead due to dense preprocessing, or fail to effectively balance exploration and exploitation, resulting in incomplete information coverage and inefficiency. In this work, we introduce EEA, a novel video agent framework that archives exploration-exploitation balance through semantic guidance with hierarchical tree search process. EEA autonomously discovers and dynamically updates task-relevant semantic queries, and collects video frames closely matched to these queries as semantic anchors. During the tree search process, instead of uniform expansion, EEA preferentially explores semantically relevant frames while ensuring sufficient coverage within unknown segments. Moreover, EEA adaptively combines intrinsic rewards from visionlanguage models (VLMs) with semantic priors by explicitly modeling uncertainty to achieve stable and precise evaluation of video segments. Experiments across various long-video benchmarks validate the superior performance and computational efficiency of our proposed method.

---

## 138. 论文ID: 2512.03370v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03370v1.json'

---

## 139. nuScenes Revisited: Progress and Challenges in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.02448v1](http://arxiv.org/abs/2512.02448v1)

**作者:** Whye Kit Fong, Venice Erin Liong, Kok Seang Tan, Holger Caesar

**发布时间:** 2025-12-02

**备注:** 18 pages, 17 figures

### GPT解析

### 总结

该研究重新审视了广泛使用的自动驾驶数据集nuScenes，详细分析了其创建过程、技术细节、影响及在自动驾驶领域的重要性。

### 背景

深度学习正在革新自动驾驶汽车和高级驾驶辅助系统，作为数据驱动方法，深度学习依赖大量详细标记的驾驶数据，数据集、硬件和算法是自动驾驶开发的基础组成部分。

### 目的

提供nuScenes数据集及其扩展nuImages和Panoptic nuScenes创建过程的深入分析，追踪nuScenes对其他数据集的影响，展示使用该数据集的任务概览，并回顾主要方法发展。

### 方法

分析nuScenes数据集及其扩展，追踪其对后续数据集的影响，整合官方和非官方使用nuScenes的任务，回顾主要方法发展，提供自动驾驶文献的综合调查。

### 主要发现

nuScenes代表了自动驾驶开发的关键趋势，是第一个包含雷达数据的数据集，收集了来自两个大陆的多样化城市驾驶场景，使用完全自动驾驶车辆在公共道路上收集，促进了多模态传感器融合、标准化基准和广泛任务，对后续数据集产生了重大影响并定义了众多行业标准。

### 结论

nuScenes数据集及其扩展对自动驾驶领域产生了深远影响，该研究提供了基于nuScenes的自动驾驶文献综合调查，展示了其在自动驾驶研究中的核心地位。

### 翻译

自动驾驶汽车和高级驾驶辅助系统已被深度学习所革新。作为数据驱动方法，深度学习依赖于大量驾驶数据，这些数据通常被详细标记。因此，数据集与硬件和算法一起，成为自动驾驶开发的基础构建模块。在本研究中，我们重新审视了最广泛使用的自动驾驶数据集之一：nuScenes数据集。nuScenes体现了自动驾驶开发的关键趋势，它是第一个包含雷达数据的数据集，具有来自两个大陆的多样化城市驾驶场景，并使用在公共道路上运行的完全自动驾驶车辆收集，同时促进多模态传感器融合、标准化基准以及包括感知、定位与地图构建、预测和规划在内的广泛任务。我们前所未有地深入了解了nuScenes的创建过程，以及其扩展nuImages和Panoptic nuScenes，总结了许多迄今为止在学术出版物中尚未披露的技术细节。此外，我们追踪了nuScenes的影响如何影响了后来发布的大量其他数据集，以及它如何定义了社区至今仍在使用的众多标准。最后，我们介绍了使用nuScenes数据集的官方和非官方任务概览，并回顾了主要方法发展，从而提供了自动驾驶文献的全面调查，特别关注nuScenes。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决对nuScenes自动驾驶数据集进行系统回顾和评估的问题。这个问题很重要，因为数据集是自动驾驶研究和开发的基础设施，nuScenes作为首个包含雷达数据、具有多样化城市场景且使用自动驾驶车辆收集的数据集，其质量直接影响自动驾驶算法的性能评估和进步，同时它定义了行业标准并影响了后续数据集的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者没有提出新方法，而是对现有nuScenes数据集进行回顾分析。他们的思考过程包括：回顾自动驾驶数据集发展历史（从KITTI开始），分析nuScenes的创建背景和设计考量（解决KITTI的局限性），详细描述数据收集、处理和标注流程，并与其他数据集进行比较。作者借鉴了现有工作，包括基于KITTI等早期数据集的经验，利用现有传感器技术和标注方法，参考Google街景的隐私保护方法，以及借鉴地图构建和SLAM技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '由于本文主要是回顾分析现有数据集而非提出新方法，nuScenes的核心思想是创建一个全面、多样化的自动驾驶数据集，解决现有数据集的局限性，提供多模态传感器数据支持多种任务，包含高质量标注和地图数据。其实现流程包括：1)使用配备多种传感器的车辆在多个城市收集数据；2)进行数据后处理（图像调整、自运动补偿、同步、校准）；3)进行详细的数据标注；4)实施隐私保护措施；5)进行严格的质量保证；6)构建高精度地图；7)创建扩展数据集（Panoptic nuScenes和nuImages）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)数据集创新（首个包含雷达数据、跨大洲收集、使用自动驾驶车辆收集的数据集）；2)技术创新（提供高精度地图、丰富物体属性标注、精确传感器校准同步、隐私保护系统）；3)评估方法创新（引入NDS综合指标、单一排行榜设计、可分解指标）。相比KITTI等早期工作，nuScenes规模更大、环境更多样化、传感器配置更完整、提供详细的地图数据、标注质量更高、评估方法更综合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过对nuScenes自动驾驶数据集的全面回顾、分析和评估，揭示了其在推动自动驾驶技术发展和标准化方面的关键作用，并指出了未来研究和发展的方向。'}


### 论文摘要

Autonomous Vehicles (AV) and Advanced Driver Assistance Systems (ADAS) have been revolutionized by Deep Learning. As a data-driven approach, Deep Learning relies on vast amounts of driving data, typically labeled in great detail. As a result, datasets, alongside hardware and algorithms, are foundational building blocks for the development of AVs. In this work we revisit one of the most widely used autonomous driving datasets: the nuScenes dataset. nuScenes exemplifies key trends in AV development, being the first dataset to include radar data, to feature diverse urban driving scenes from two continents, and to be collected using a fully autonomous vehicle operating on public roads, while also promoting multi-modal sensor fusion, standardized benchmarks, and a broad range of tasks including perception, localization \& mapping, prediction and planning. We provide an unprecedented look into the creation of nuScenes, as well as its extensions nuImages and Panoptic nuScenes, summarizing many technical details that have hitherto not been revealed in academic publications. Furthermore, we trace how the influence of nuScenes impacted a large number of other datasets that were released later and how it defined numerous standards that are used by the community to this day. Finally, we present an overview of both official and unofficial tasks using the nuScenes dataset and review major methodological developments, thereby offering a comprehensive survey of the autonomous driving literature, with a particular focus on nuScenes.

---

## 140. Towards Modeling Road Access Deprivation in Sub-Saharan Africa Based on a New Accessibility Metric and Road Quality

**论文链接:** [http://arxiv.org/abs/2512.02190v1](http://arxiv.org/abs/2512.02190v1)

**作者:** Sebastian Hafner, Qunshan Zhao, Bunmi Alugbin, Kehinde Baruwa, Caleb Cheruiyot, Sabitu Sa'adu Da'u, Xingyi Du, Peter Elias, Helen Elsey, Ryan Engstrom, Serkan Girgin, Diego F. P. Grajales, Esther Judith, Caroline Kabaria, Monika Kuffer, Oluwatoyin Odulana, Francis C. Onyambu, Adenike Shonowo, Dana R. Thomson, Mingyu Zhu, João Porto de Albuquerque

**发布时间:** 2025-12-01

**备注:** 20 pages, 21 figures, submitted to Habitat International

### GPT解析

### 总结

该研究提出了一种道路获取剥夺模型，结合可达性指标和道路质量数据，用于评估城市道路获取情况，并在三个非洲城市进行了应用，结果显示该模型能有效识别不同剥夺水平的区域。

### 背景

在快速城市化地区，特别是撒哈拉以南非洲，可通行的道路是城市基础设施的关键维度。然而，许多城市社区，尤其是非正规定居点，仍然与道路网络隔绝。

### 目的

开发一个结合可达性指标和道路质量数据的道路获取剥夺模型，用于分类城市地区的剥夺水平，帮助识别与道路网络连接不畅的区域，为数据稀缺环境下的城市规划提供支持。

### 方法

模型结合了新的可达性指标（捕捉建筑物与道路网络连接程度）和道路类型数据（作为道路质量的代理），将城市区域分为低、中或高剥夺水平。研究使用开放地理空间数据集对内罗毕（肯尼亚）、拉各斯（尼日利亚）和卡诺（尼日利亚）三个城市应用了该模型。

### 主要发现

在三个城市中，大多数建成区域属于低和中等剥夺水平，高度剥夺区域比例从内罗毕的11.8%到卡诺的27.7%不等。模型评估显示，识别低剥夺区域表现良好（F1 > 0.74），中等剥夺区域在内罗毕和拉各斯具有中等准确性（F1 > 0.52），高剥夺区域结果更不稳定（F1从0.26到0.69）。社区验证显示不同剥夺水平间存在分歧，主要源于概念模型与社区认知不匹配及模型操作化问题。

### 结论

道路获取剥夺建模方法作为一种可扩展、可解释的工具，在识别连接不畅区域和数据稀缺环境下的城市规划方面显示出潜力。

### 翻译

可通行的道路是城市基础设施的关键维度，特别是在快速城市化的地区，如撒哈拉以南非洲。然而，许多城市社区，特别是那些非正规定居点，仍然与道路网络隔绝。本研究提出了一种道路获取剥夺模型，该模型结合了一种新的可达性指标（捕捉建筑物与道路网络的连接程度）和道路类型数据（作为道路质量的代理）。这两个组件共同将城市区域分为低、中或高剥夺水平。使用开放地理空间数据集，该模型被应用于内罗毕（肯尼亚）、拉各斯（尼日利亚）和卡诺（尼日利亚）。在所有三个城市中，大多数建成区域属于低和中等道路获取剥夺水平，而高度剥夺区域相对有限。然而，高度剥夺区域的比例差异显著，从内罗毕的11.8%到卡诺的27.7%不等。与社区来源的验证数据相比，模型评估显示识别低剥夺区域表现良好（F1 > 0.74），内罗毕和拉各斯的中等剥夺区域具有中等准确性（F1 > 0.52，卡诺较低），而高剥夺区域的结果更不稳定（F1从卡诺的0.26到内罗毕的0.69）。此外，对具有多个验证的网格单元的分析显示，社区成员之间有很强的一致性，分歧主要发生在相邻剥夺水平之间。最后，我们讨论了与社区验证分歧的两种来源：（1）概念模型与社区认知之间的不匹配，以及（2）概念模型的操作化。总之，我们的道路获取剥夺建模方法作为一种可扩展、可解释的工具，在识别连接不畅区域和数据稀缺环境下的城市规划方面显示出潜力。


### 论文摘要

Access to motorable roads is a critical dimension of urban infrastructure, particularly in rapidly urbanizing regions such as Sub-Saharan Africa. Yet, many urban communities, especially those in informal settlements, remain disconnected from road networks. This study presents a road access deprivation model that combines a new accessibility metric, capturing how well buildings are connected to the road network, with road surface type data as a proxy for road quality. These two components together enable the classification of urban areas into low, medium, or high deprivation levels. The model was applied to Nairobi (Kenya), Lagos (Nigeria), and Kano (Nigeria) using open geospatial datasets. Across all three cities, the majority of built-up areas fall into the low and medium road access deprivation levels, while highly deprived areas are comparatively limited. However, the share of highly deprived areas varies substantially, ranging from only 11.8 % in Nairobi to 27.7 % in Kano. Model evaluation against community-sourced validation data indicates good performance for identifying low deprivation areas (F1 > 0.74), moderate accuracy for medium deprivation in Nairobi and Lagos (F1 > 0.52, lower in Kano), and more variable results for high deprivation (F1 ranging from 0.26 in Kano to 0.69 in Nairobi). Furthermore, analysis of grid cells with multiple validations showed strong agreement among community members, with disagreements occurring mainly between adjacent deprivation levels. Finally, we discussed two types of sources for disagreement with community validations: (1) misalignment between the conceptual model and community perceptions, and (2) the operationalization of the conceptual model. In summary, our road access deprivation modeling approach demonstrates promise as a scalable, interpretable tool for identifying disconnected areas and informing urban planning in data-scarce contexts.

---

## 141. Transmit Weights, Not Features: Orthogonal-Basis Aided Wireless Point-Cloud Transmission

**论文链接:** [http://arxiv.org/abs/2512.03819v1](http://arxiv.org/abs/2512.03819v1)

**作者:** Junlin Chang, Yubo Han, Hnag Yue, John S Thompson, Rongke Liu

**发布时间:** 2025-12-03

**备注:** 5 pages, 5 figures

### GPT解析

### 总结

该研究提出了一种基于深度联合信源信道编码的三维点云语义无线传输框架，通过在接收端预测特征池的组合权重而非传输原始特征，实现了高效的点云传输，在带宽受限环境下表现出色，同时保持了较高的重建质量。

### 背景

深度传感器的广泛应用大大降低了点云获取的门槛，使得三维点云数据的传输变得更为重要。

### 目的

设计一种基于深度联合信源信道编码的三维点云语义无线传输框架，实现高效、鲁棒的点云传输。

### 方法

发送端在接收端的语义正交特征池上预测组合权重而非传输原始特征；使用基于折叠的解码器将2D网格变形为3D以保持流形连续性和几何保真度；系统使用Chamfer距离和正交性正则化器进行训练；在ModelNet40数据集上通过变化的信噪比和带宽进行评估。

### 主要发现

在高带宽下，性能与SEPT相当；在带宽受限的情况下有明显的性能提升；在峰值信噪比和CD方面都有持续改进；消融实验证实了正交化和折叠先验的好处。

### 结论

该语义无线传输框架在带宽受限情况下优于现有方法，同时在高带宽下与现有方法性能相当，为点云无线传输提供了新思路。

### 翻译

深度传感器的广泛应用大大降低了点云获取的门槛。本文提出了一种基于深度联合信源信道编码的三维点云语义无线传输框架。与发送原始特征不同，发送端在接收端的语义正交特征池上预测组合权重，实现紧凑表示和鲁棒重建。基于折叠的解码器将2D网格变形为3D，同时保持流形连续性和几何保真度。系统使用Chamfer距离和正交性正则化器进行训练，并在ModelNet40数据集上通过变化的信噪比和带宽进行评估。结果显示，在高带宽下性能与SEPT相当，在带宽受限的情况下有明显的性能提升，在峰值信噪比和CD方面都有持续改进。消融实验证实了正交化和折叠先验的好处。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云数据无线传输效率低、抗噪声能力差的问题。随着深度传感器在自动驾驶、虚拟现实和机器人等领域的广泛应用，高效传输点云数据变得至关重要。传统方法忽略了数据中的语义信息，导致在低带宽和低信噪比环境下性能不佳，限制了这些关键应用的实用性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到自然数据具有近似低秩特性，并受此启发设计了新方法。他们借鉴了Point-BERT作为语义编码器利用预训练知识，参考了扩散模型中的去噪思想设计信道解码器，并采用了点云处理中的折叠机制。作者基于Deep联合信源信道编码框架，创新性地提出在接收端构建正交特征池，发射端传输基的权重而非原始特征，结合折叠解码器处理点云不规则性，从而提高传输效率和鲁棒性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是传输预定义正交基的权重而非原始特征，利用自然数据的低秩特性，结合流形连续性约束实现高效鲁棒的点云传输。整体流程：1)发射端用Point-BERT提取语义特征，压缩为带宽受限表示并功率归一化；2)信号通过AWGN信道传输；3)接收端用残差校正机制去噪，通过加权聚合正交基重建语义特征，预测中心坐标，最后用折叠解码器将2D网格变形为3D点云；4)训练时使用Chamfer Distance和正交正则化联合优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)传输权重而非原始特征的机制；2)接收端正交特征池设计与正交正则化；3)基于折叠的解码器引入2D网格先验；4)联合信源信道编码优化。相比SEPT等前工作，本文方法在传输内容上从特征变为权重，表示方式从多尺度编码变为正交基组合，解码机制从基于偏移变为折叠解码，且在带宽受限条件下性能明显更优，对信道噪声具有更强鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种通过传输正交基权重而非原始特征，结合折叠解码器的语义无线点云传输方法，实现了在带宽受限条件下高效且鲁棒的三维点云数据重建。'}


### 论文摘要

The widespread adoption of depth sensors has substantially lowered the barrier to point-cloud acquisition. This letter proposes a semantic wireless transmission framework for three dimension (3D) point clouds built on Deep Joint Source - Channel Coding (DeepJSCC). Instead of sending raw features, the transmitter predicts combination weights over a receiver-side semantic orthogonal feature pool, enabling compact representations and robust reconstruction. A folding-based decoder deforms a 2D grid into 3D, enforcing manifold continuity while preserving geometric fidelity. Trained with Chamfer Distance (CD) and an orthogonality regularizer, the system is evaluated on ModelNet40 across varying Signal-to-Noise Ratios (SNRs) and bandwidths. Results show performance on par with SEmantic Point cloud Transmission (SEPT) at high bandwidth and clear gains in bandwidth-constrained regimes, with consistent improvements in both Peak Signal-to-Noise Ratio (PSNR) and CD. Ablation experiments confirm the benefits of orthogonalization and the folding prior.

---

## 142. Towards Privacy-Preserving Range Queries with Secure Learned Spatial Index over Encrypted Data

**论文链接:** [http://arxiv.org/abs/2512.03669v1](http://arxiv.org/abs/2512.03669v1)

**作者:** Zuan Wang, Juntao Lu, Jiazhuang Wu, Youliang Tian, Wei Song, Qiuxian Li, Duo Zhang

**发布时间:** 2025-12-03

**备注:** IEEE TrustCom-2025

### GPT解析

### 总结

本文提出了一种名为SLRQ的新型隐私保护范围查询方案，通过安全学习空间索引(SLS-INDEX)和安全协议，在确保数据集、查询、结果和访问模式隐私的同时，显著提高了查询效率。

### 背景

随着云服务在大规模数据管理中的依赖日益增长，保护外包数据集的安全和隐私变得至关重要。虽然加密数据和查询可以防止直接内容暴露，但对手仍可通过访问模式和搜索路径分析推断敏感信息。现有提供强访问模式隐私保护的方案通常存在较大的性能开销。

### 目的

设计一种新颖的加密数据集上的隐私保护范围查询方案，在提供强大安全保证的同时保持高效率。

### 方法

开发了安全学习空间索引(SLS-INDEX)，将Paillier密码系统与分层预测架构和注入噪声的桶相结合，实现加密域中的数据感知查询加速；SLRQ采用基于置换的安全桶预测协议来模糊查询执行路径；引入安全点提取协议生成候选结果，减少安全计算开销。

### 主要发现

在现实泄漏函数下提供了形式化安全分析；实现原型评估实际性能；在真实和合成数据集上的实验表明，SLRQ在查询效率方面显著优于现有解决方案。

### 结论

SLRQ方案能够在确保数据集、查询、结果和访问模式隐私的同时，实现高效的查询性能。

### 翻译

随着云服务在大规模数据管理中的依赖日益增长，保护外包数据集的安全和隐私变得越来越重要。虽然加密数据和查询可以防止直接内容暴露，但最近的研究表明，对手仍然可以通过访问模式和搜索路径分析推断敏感信息。然而，现有提供强访问模式隐私保护的解决方案通常会产生巨大的性能开销。在本文中，我们提出了一种新颖的加密数据集上的隐私保护范围查询方案，在提供强大安全保证的同时保持高效率。为此，我们开发了安全学习空间索引(SLS-INDEX)，这是一种安全学习索引，将Paillier密码系统与分层预测架构和注入噪声的桶相结合，实现加密域中的数据感知查询加速。为了进一步模糊查询执行路径，基于SLS-INDEX的范围查询(SLRQ)采用基于置换的安全桶预测协议。此外，我们引入了安全点提取协议，生成候选结果以减少安全计算的开销。我们在现实泄漏函数下提供了形式化安全分析，并实现了原型来评估其实际性能。在真实和合成数据集上的广泛实验表明，SLRQ在查询效率方面显著优于现有解决方案，同时确保了数据集、查询、结果和访问模式隐私。


### 论文摘要

With the growing reliance on cloud services for large-scale data management, preserving the security and privacy of outsourced datasets has become increasingly critical. While encrypting data and queries can prevent direct content exposure, recent research reveals that adversaries can still infer sensitive information via access pattern and search path analysis. However, existing solutions that offer strong access pattern privacy often incur substantial performance overhead. In this paper, we propose a novel privacy-preserving range query scheme over encrypted datasets, offering strong security guarantees while maintaining high efficiency. To achieve this, we develop secure learned spatial index (SLS-INDEX), a secure learned index that integrates the Paillier cryptosystem with a hierarchical prediction architecture and noise-injected buckets, enabling data-aware query acceleration in the encrypted domain. To further obfuscate query execution paths, SLS-INDEXbased Range Queries (SLRQ) employs a permutation-based secure bucket prediction protocol. Additionally, we introduce a secure point extraction protocol that generates candidate results to reduce the overhead of secure computation. We provide formal security analysis under realistic leakage functions and implement a prototype to evaluate its practical performance. Extensive experiments on both real-world and synthetic datasets demonstrate that SLRQ significantly outperforms existing solutions in query efficiency while ensuring dataset, query, result, and access pattern privacy.

---

## 143. What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.03422v1](http://arxiv.org/abs/2512.03422v1)

**作者:** Tianchen Deng, Yue Pan, Shenghai Yuan, Dong Li, Chen Wang, Mingrui Li, Long Chen, Lihua Xie, Danwei Wang, Jingchuan Wang, Javier Civera, Hesheng Wang, Weidong Chen

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提供了机器人场景表示方法的全面概述，涵盖传统表示方法（点云、体素、符号距离函数、场景图）和新兴神经表示方法（NeRF、3D高斯溅射、基础模型）。文章探讨了这些方法在机器人核心模块（感知、建图、定位、导航、操作）中的应用与优缺点。

### 背景

当前SLAM和定位系统主要依赖稀疏表示方法，而密集场景表示在导航和避障等下游任务中预计将发挥关键作用。神经表示方法能够很好地集成高级语义特征和基于语言的先验知识，实现更全面的3D场景理解和具身智能。

### 目的

回答'什么是机器人学中最佳的三维场景表示方法？'这一问题，探讨三维场景表示的未来发展趋势，特别是三维基础模型如何可能成为未来机器人应用的统一解决方案。

### 方法

将机器人核心模块分为五个部分，介绍不同场景表示方法的标准公式，并比较这些表示在不同模块中的优缺点。

### 主要发现

神经表示方法（NeRF、3DGS和基础模型）非常适合集成高级语义特征和基于语言的先验知识，能够实现更全面的3D场景理解和具身智能。三维基础模型有潜力取代当前方法，成为未来机器人应用的统一解决方案。

### 结论

为研究人员提供有价值的资源，探索三维场景表示的未来及其在机器人学中的应用。作者已在GitHub上发布开源项目，并计划持续更新。

### 翻译

在本文中，我们提供了机器人现有场景表示方法的全面概述，涵盖了传统表示方法，如点云、体素、符号距离函数和场景图，以及更近期的神经表示方法，如神经辐射场、3D高斯溅射和新兴的基础模型。虽然当前的SLAM和定位系统主要依赖于点云和体素等稀疏表示方法，但密集场景表示预计将在导航和避障等下游任务中发挥关键作用。此外，神经表示方法如NeRF、3DGS和基础模型非常适合集成高级语义特征和基于语言的先验知识，从而实现更全面的3D场景理解和具身智能。在本文中，我们将机器人的核心模块分为五个部分（感知、建图、定位、导航和操作）。我们首先介绍不同场景表示方法的标准公式，并比较场景表示在不同模块中的优缺点。本综述围绕以下问题展开：什么是机器人学中最佳的三维场景表示？然后，我们讨论了三维场景表示的未来发展趋势，特别关注三维基础模型如何可能取代当前方法，成为未来机器人应用的统一解决方案。同时，我们也探讨了完全实现这一模型所面临的挑战。我们旨在为新手和有经验的研究人员提供有价值的资源，探索三维场景表示的未来及其在机器人学中的应用。我们已在GitHub上发布了一个开源项目，并将继续向该项目添加新的工作和技术。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要想解决的问题是：为机器人确定最佳的3D场景表示方法。这个问题非常重要，因为3D场景表示是机器人理解周围环境的基础，直接影响机器人的导航、避障、操作和智能交互等核心能力。随着深度学习和计算机图形学的发展，新的表示方法不断涌现，需要系统性地评估它们在不同机器人任务中的适用性，以帮助研究人员和工程师选择最适合特定应用场景的表示方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者采用系统性的综述方法，将机器人核心模块分为五个部分（感知、建图、定位、导航、操作），并从多个维度（数据形式、连续性、内存效率、保真度、灵活性、几何表示能力）评估不同场景表示方法。论文借鉴了大量现有工作，引用了100多篇相关论文，涵盖了从传统几何表示到最新神经表示的完整技术谱系。作者对这些现有工作进行了分类、比较和分析，形成了一个全面的评估框架，而非提出全新的方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '由于这是一篇综述文章，其核心思想是全面评估和比较不同3D场景表示方法在机器人学各个模块中的适用性，探索从传统几何表示到神经表示再到基础模型的演进趋势。整体实现流程包括：1）引言部分介绍3D场景表示的重要性和论文贡献；2）背景与问题定义部分介绍传统和神经场景表示的基本概念；3）3D场景表示综述部分回顾不同表示方法的发展历程并进行多维度比较；4）感知、建图与定位、交互等模块分析不同表示方法在各任务中的应用；5）结论与未来展望部分总结研究发现并指出未来方向。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '论文的关键创新点包括：1）建立了全面的机器人模块分类体系（感知、建图、定位、导航、操作）；2）提出了多维度的比较框架评估不同场景表示方法；3）整合了最新的3D场景表示技术（NeRF、3DGS和基础模型）；4）深入分析了3D基础模型作为未来机器人统一解决方案的发展前景。相比之前的工作，这篇论文范围更广（涵盖完整技术谱系）、视角更系统（从整体机器人系统角度评估）、实用性更强（关注实际任务适用性）且前瞻性更强（探讨前沿技术应用前景）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统性地综述和比较从传统几何表示到神经表示再到基础模型的3D场景表示方法，为机器人学应用选择最合适的场景表示提供了全面的指导框架，并指明了3D基础模型作为未来机器人统一解决方案的发展方向。'}


### 论文摘要

In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models. While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance. Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence. In this paper, we categorized the core modules of robotics into five parts (Perception, Mapping, Localization, Navigation, Manipulation). We start by presenting the standard formulation of different scene representation methods and comparing the advantages and disadvantages of scene representation across different modules. This survey is centered around the question: What is the best 3D scene representation for robotics? We then discuss the future development trends of 3D scene representations, with a particular focus on how the 3D Foundation Model could replace current methods as the unified solution for future robotic applications. The remaining challenges in fully realizing this model are also explored. We aim to offer a valuable resource for both newcomers and experienced researchers to explore the future of 3D scene representations and their application in robotics. We have published an open-source project on GitHub and will continue to add new works and technologies to this project.

---

## 144. GraphFusion3D: Dynamic Graph Attention Convolution with Adaptive Cross-Modal Transformer for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.02991v1](http://arxiv.org/abs/2512.02991v1)

**作者:** Md Sohag Mia, Md Nahid Hasan, Tawhid Ahmed, Muhammad Abdullah Adnan

**发布时间:** 2025-12-02

### GPT解析

### 背景

尽管3D目标检测取得了显著进展，点云数据仍然面临挑战，包括稀疏数据、不完整结构和有限的语义信息。此外，捕捉远处物体之间的上下文关系也存在额外困难。

### 目的

提出GraphFusion3D框架，结合多模态融合和高级特征学习，解决点云数据面临的挑战。

### 方法

引入自适应跨模态转换器(ACMT)，自适应地将图像特征整合到点表示中以丰富几何和语义信息；引入图推理模块(GRM)，建模邻域关系同时捕获局部几何结构和全局语义上下文，使用多尺度图注意力动态加权提案之间的空间邻近性和特征相似性；采用级联解码器通过多阶段预测逐步改进检测。

### 主要发现

在SUN RGB-D数据集上达到70.6% AP25和51.2% AP50的性能，在ScanNetV2数据集上达到75.1% AP25和60.8% AP50的性能，相比现有方法有显著的性能提升。

### 结论

GraphFusion3D框架有效解决了点云数据面临的挑战，通过多模态融合和高级特征学习提高了3D目标检测性能。

### 翻译

尽管3D目标检测取得了显著进展，点云数据仍然面临挑战，原因是数据稀疏、结构不完整和语义信息有限。捕捉远处物体之间的上下文关系带来了额外的困难。为应对这些挑战，我们提出了GraphFusion3D，这是一个结合多模态融合与高级特征学习的统一框架。我们的方法引入了自适应跨模态转换器(ACMT)，它自适应地将图像特征整合到点表示中，以丰富几何和语义信息。对于提案精炼，我们引入了图推理模块(GRM)，这是一种新颖的机制，用于建模邻域关系，同时捕获局部几何结构和全局语义上下文。该模块采用多尺度图注意力来动态加权提案之间的空间邻近性和特征相似性。我们进一步采用了一个级联解码器，通过多阶段预测逐步改进检测结果。在SUN RGB-D(70.6% AP25和51.2% AP50)和ScanNetV2(75.1% AP25和60.8% AP50)上的大量实验表明，与现有方法相比，性能有显著提升。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D物体检测中点云数据面临的挑战，包括数据稀疏、结构不完整和语义信息有限，以及难以捕捉远处物体间上下文关系的问题。这个问题在现实中非常重要，因为3D物体检测是自动驾驶、家庭机器人和增强现实等应用的核心技术，准确的3D感知使机器能够安全、智能地与三维真实世界环境交互，例如帮助自动驾驶车辆检测行人和其他车辆，或使服务机器人理解物体位置进行抓取和导航。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性来设计新方法，发现单纯依赖点云数据存在不规则、稀疏问题，而单纯依赖图像数据缺乏深度信息。作者借鉴了ImVoteNet和EPNet++等多模态融合方法，并结合了DeformDETR等先进技术。作者设计了一个统一框架，通过图推理模块捕捉局部几何结构和全局语义上下文，利用自适应跨模态Transformer动态融合图像和点云特征，并通过级联解码器逐步优化检测结果。这种方法特别考虑了室内场景的特点，如物体间的空间关系（如椅子在桌子旁）。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过图神经网络和Transformer的结合，有效融合点云的几何信息和图像的语义信息，提升3D物体检测的准确性。整体流程分为四个阶段：1) 特征提取，分别通过点云和图像骨干网络提取特征；2) 图推理模块处理，使用多尺度图注意力建模邻域关系；3) 自适应跨模态Transformer融合，动态平衡2D和3D特征；4) 渐进式级联细化解码，通过多阶段预测逐步优化边界框定位。这种方法能够同时捕捉物体的几何结构和语义上下文，提高在复杂场景中的检测性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 图推理模块(GRM)，使用多尺度图注意力同时考虑空间接近度和特征相似性；2) 自适应跨模态Transformer(ACMT)，通过跨模态门控机制动态调整图像和点云特征的贡献权重；3) 渐进式级联细化解码器，多阶段优化边界框定位；4) 专门的GraphFusion3D分配器处理细长物体。相比之前工作，不同之处在于实现了更动态、上下文感知的特征融合，显式建模物体间的空间和语义关系，以及设计了专门的级联解码过程，显著提升了在室内场景中的检测性能，特别是在SUN RGB-D数据集上达到了最先进水平。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GraphFusion3D通过结合自适应跨模态Transformer、多尺度图推理和渐进式级联解码，实现了高效的三维物体检测，显著提升了室内场景中点云和RGB图像融合的检测性能。'}


### 论文摘要

Despite significant progress in 3D object detection, point clouds remain challenging due to sparse data, incomplete structures, and limited semantic information. Capturing contextual relationships between distant objects presents additional difficulties. To address these challenges, we propose GraphFusion3D, a unified framework combining multi-modal fusion with advanced feature learning. Our approach introduces the Adaptive Cross-Modal Transformer (ACMT), which adaptively integrates image features into point representations to enrich both geometric and semantic information. For proposal refinement, we introduce the Graph Reasoning Module (GRM), a novel mechanism that models neighborhood relationships to simultaneously capture local geometric structures and global semantic context. The module employs multi-scale graph attention to dynamically weight both spatial proximity and feature similarity between proposals. We further employ a cascade decoder that progressively refines detections through multi-stage predictions. Extensive experiments on SUN RGB-D (70.6\% AP$_{25}$ and 51.2\% AP$_{50}$) and ScanNetV2 (75.1\% AP$_{25}$ and 60.8\% AP$_{50}$) demonstrate a substantial performance improvement over existing approaches.

---

## 145. BEVDilation: LiDAR-Centric Multi-Modal Fusion for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.02972v1](http://arxiv.org/abs/2512.02972v1)

**作者:** Guowen Zhang, Chenhang He, Liyi Chen, Lei Zhang

**发布时间:** 2025-12-02

**备注:** Accept by AAAI26

### GPT解析

### 总结

BEVDilation是一种以LiDAR为中心的新型框架，通过将图像BEV特征作为隐式指导而非简单连接，有效缓解了图像深度估计错误引起的空间不匹配问题，同时利用图像先验解决点云的稀疏性和语义局限性。

### 背景

将LiDAR和相机信息集成到鸟瞰图(BEV)表示中已被证明在3D目标检测中有效。但由于这些传感器在几何精度上的根本差异，先前方法的 indiscriminate 融合往往导致性能下降。

### 目的

提出一种名为BEVDilation的新型以LiDAR为中心的框架，在融合中优先考虑LiDAR信息，并通过隐式指导方式解决空间不匹配问题。

### 方法

1) 将图像BEV特征表述为隐式指导而非简单连接；2) 提出稀疏体素膨胀块，通过图像先验增加前景体素密度，减轻点云稀疏性；3) 引入语义引导的BEV膨胀块，利用图像语义指导和长距离上下文捕获增强LiDAR特征扩散处理。

### 主要发现

在nuScenes基准测试上，BEVDilation比最先进方法实现了更好的性能，同时保持有竞争力的计算效率；与简单融合相比，以LiDAR为中心的策略对深度噪声具有更强的鲁棒性。

### 结论

BEVDilation通过隐式指导方式融合LiDAR和相机信息，有效解决了空间不匹配问题，同时利用图像信息增强点云特征，实现了比现有方法更优的性能和鲁棒性。

### 翻译

将LiDAR和相机信息集成到鸟瞰图(BEV)表示中已被证明在3D目标检测中有效。然而，由于这些传感器在几何精度上存在根本差异，先前方法的 indiscriminate 融合往往导致性能下降。在本文中，我们提出了BEVDilation，一种新型以LiDAR为中心的框架，在融合中优先考虑LiDAR信息。通过将图像BEV特征表述为隐式指导而非简单连接，我们的策略有效缓解了图像深度估计错误引起的空间不匹配问题。此外，图像指导可以有效帮助以LiDAR为中心的范式解决点云的稀疏性和语义局限性。具体而言，我们提出了一个稀疏体素膨胀块，通过图像先验增加前景体素的密度，减轻点云的固有稀疏性。此外，我们引入了语义引导的BEV膨胀块，利用图像语义指导和长距离上下文捕获增强LiDAR特征扩散处理。在具有挑战性的nuScenes基准测试上，BEVDilation比最先进方法实现了更好的性能，同时保持有竞争力的计算效率。重要的是，与简单融合相比，我们的以LiDAR为中心的策略对深度噪声表现出更强的鲁棒性。源代码可在https://github.com/gwenzhang/BEVDilation获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR和相机信息在鸟瞰图(BEV)表示中的融合问题。由于这两种传感器在几何精度上存在根本差异，之前的简单融合方法往往导致性能下降。这个问题在自动驾驶、虚拟现实等领域非常重要，因为LiDAR提供准确的几何信息，而相机提供丰富的语义信息，两者的有效结合能提高感知系统的准确性和鲁棒性，但传感器间的差异使得简单融合效果不佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了LiDAR和相机传感器间的差异：LiDAR提供准确位置信息但点云稀疏，相机提供丰富语义但深度估计易出错。基于此，作者设计了以LiDAR为中心的融合框架，将图像特征作为隐式指导而非直接融合。方法借鉴了现有工作：使用Lift-Splat-Shot进行图像特征到BEV的转换，采用VoxelNet作为稀疏体素编码器，利用Mamba模型的全局感受野处理体素特征，并应用可变形卷积提高特征扩散效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是以LiDAR为中心的融合框架，优先考虑LiDAR的几何精度，同时利用图像语义信息增强点云表示。整体流程包括：1)输入处理 - 分别处理LiDAR点云和多视图图像；2)稀疏体素膨胀块(SVDB) - 预测前景掩码并填充可学习体素解决稀疏问题；3)语义引导的BEV膨胀块(SBDB) - 使用多模态条件变形进行特征扩散；4)检测头 - 仅使用LiDAR特征进行回归保持几何精度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)LiDAR为中心的融合框架，将图像作为隐式指导；2)稀疏体素膨胀块(SVDB)填充前景区域解决点云稀疏问题；3)语义引导的BEV膨胀块(SBDB)使用多模态条件变形进行特征扩散。相比之前工作，BEVDilation优先考虑LiDAR几何信息而非简单融合，使用语义引导的可变形卷积提高特征扩散效率，对深度噪声具有更强鲁棒性，在nuScenes基准上实现了更高的检测精度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BEVDilation提出了一种以LiDAR为中心的多模态融合框架，通过稀疏体素膨胀和语义引导的特征扩散，有效解决了LiDAR和相机在3D目标检测中的融合挑战，显著提高了检测性能和对深度噪声的鲁棒性。'}


### 论文摘要

Integrating LiDAR and camera information in the bird's eye view (BEV) representation has demonstrated its effectiveness in 3D object detection. However, because of the fundamental disparity in geometric accuracy between these sensors, indiscriminate fusion in previous methods often leads to degraded performance. In this paper, we propose BEVDilation, a novel LiDAR-centric framework that prioritizes LiDAR information in the fusion. By formulating image BEV features as implicit guidance rather than naive concatenation, our strategy effectively alleviates the spatial misalignment caused by image depth estimation errors. Furthermore, the image guidance can effectively help the LiDAR-centric paradigm to address the sparsity and semantic limitations of point clouds. Specifically, we propose a Sparse Voxel Dilation Block that mitigates the inherent point sparsity by densifying foreground voxels through image priors. Moreover, we introduce a Semantic-Guided BEV Dilation Block to enhance the LiDAR feature diffusion processing with image semantic guidance and long-range context capture. On the challenging nuScenes benchmark, BEVDilation achieves better performance than state-of-the-art methods while maintaining competitive computational efficiency. Importantly, our LiDAR-centric strategy demonstrates greater robustness to depth noise compared to naive fusion. The source code is available at https://github.com/gwenzhang/BEVDilation.

---

## 146. Asymptotics for additive functionals of particle systems via Stein's method

**论文链接:** [http://arxiv.org/abs/2512.02922v1](http://arxiv.org/abs/2512.02922v1)

**作者:** Arturo Jaramillo, Antonio Murillo-Salas

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文研究了具有泊松点过程初始配置的随机测度系统的加性泛函，在基本矩界假设下，建立了归一化泛函的通用三阶矩定理，并获得了多种移动测度模型在Wasserstein距离上的定量收敛界，将定性中心极限定理转化为明确的收敛速率。

### 背景

研究关注由泊松点过程初始化的随机测度系统，其组件遵循任意马尔可夫或非马尔可夫测值动力学，且除了基本矩界外没有其他结构假设。

### 目的

建立随机测度系统加性泛函的收敛理论，提供定量收敛速率，并将定性中心极限定理转化为明确的收敛界。

### 方法

结合Stein方法与Mecke公式，采用泊松Malliavin-Stein方法论，在基本矩界条件下进行分析。

### 主要发现

1) 建立了归一化泛函的通用三阶矩定理；2) 获得了多种移动测度模型在Wasserstein距离上的第一个定量界；3) 将定性中心极限定理转化为明确的收敛速率；4) 方法适用于多种系统，包括由分数布朗运动、α-稳定过程、一致椭圆扩散和Dyson布朗运动谱经验测度驱动的系统。

### 结论

所提出的方法在广泛假设下有效，为泊松驱动的随机测度系统提供了定量收敛分析框架，扩展了泊松Malliavin-Stein方法论的应用范围。

### 翻译

我们考虑随机测度系统的加性泛函，其初始配置由泊松点过程给出，且各个组件遵循任意马尔可夫或非马尔可夫的测值动力学，除了基本矩界外没有其他结构假设。在此设定和适当条件下，我们建立了归一化泛函的通用三阶矩定理。基于这一结果，我们获得了多种由泊松驱动点云初始化的移动测度模型在Wasserstein距离上的第一个定量界，将定性中心极限定理转化为明确的收敛速率。随后通过几个例子展示了该方法的应用范围，包括由分数布朗运动、α-稳定过程、一致椭圆扩散和Dyson布朗运动产生的谱经验测度驱动的系统，所有这些都基于对初始泊松配置控制测度的广泛假设。分析依赖于Stein方法与Mecke公式的结合，遵循泊松Malliavin-Stein方法论的精神。


### 论文摘要

We consider additive functionals of systems of random measures whose initial configuration is given by a Poisson point process, and whose individual components evolve according to arbitrary Markovian or non-Markovian measure valued dynamics, with no structural assumptions beyond basic moment bounds. In this setting and under adequate conditions, we establish a general third moment theorem for the normalized functionals. Building on this result, we obtain the first quantitative bounds in the Wasserstein distance for a variety of moving-measure models initialized by Poisson-driven clouds of points, turning qualitative central limit theorems into explicit rates of convergence. The scope of the approach is then demonstrated through several examples, including systems driven by fractional Brownian motion, $α$-stable processes, uniformly elliptic diffusions, and spectral empirical measures arising from Dyson Brownian motion, all under broad assumptions on the control measure of the initial Poisson configuration. The analysis relies on a combination of Stein's method with Mecke's formula, in the spirit of the Poisson Malliavin-Stein methodology.

---

## 147. 论文ID: 2512.02375v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.02375v1.json'

---

## 148. Mapping of Lesion Images to Somatic Mutations

**论文链接:** [http://arxiv.org/abs/2512.02162v1](http://arxiv.org/abs/2512.02162v1)

**作者:** Rahul Mehta

**发布时间:** 2025-12-01

**备注:** https://dl.acm.org/doi/abs/10.1145/3340531.3414074#sec-terms

### GPT解析

### 总结

该研究构建了一个深度潜变量模型(LLOST)，通过医学影像预测患者的体细胞突变谱，实现癌症的早期诊断和治疗。

### 背景

医学影像是临床医生确定癌症诊断的关键工具，随后的诊断阶段会提取遗传信息以指导治疗方案。癌症治疗效果依赖于早期诊断和治疗。

### 目的

构建一个深度潜变量模型，基于患者的医学影像确定其体细胞突变谱，从而实现更早的诊断和治疗。

### 方法

引入病变图像的点云表示，提出LLOST模型(具有双变分自编码器和共享潜在空间)，模型包含三个使用条件归一化流先验学习的潜在空间。在去标识化的医学图像和相应的体细胞突变数据上进行实验。

### 主要发现

模型能够准确预测特定突变的数量和突变的发生；发现了影像和体细胞突变域之间的共享模式，这些模式反映了癌症类型。

### 结论

讨论了模型改进方法和未来研究方向，包括纳入其他遗传域的可能性。

### 翻译

医学影像是临床医生确定患者癌症诊断的关键初始工具，能够实现更快干预和更可靠的患者预后。在患者诊断的后续阶段，会提取遗传信息以帮助选择特定的患者治疗方案。由于癌症治疗的疗效通常依赖于早期诊断和治疗，我们构建了一个深度潜变量模型，根据患者相应的医学影像确定其体细胞突变谱。我们首先引入了病变图像的点云表示，使其对成像模具具有不变性。然后我们提出了LLOST模型，这是一个具有双变分自编码器的模型，通过一个独立的共享潜在空间将病变点云特征和不同体细胞突变计数统一起来。因此我们的模型包含三个潜在空间，每个都使用条件归一化流先验进行学习，以考虑每个域的多样化分布。我们在去标识化的医学图像（来自癌症影像档案库）和相应的体细胞突变（来自泛癌症数据集）上进行了定性和定量实验。我们展示了模型在预测特定突变数量方面的预测性能，以及准确预测突变发生的能力。特别是，我们发现影像和体细胞突变域之间的共享模式反映了癌症类型。我们最后讨论了如何改进模型以及可能的研究方向，包括纳入其他遗传域。


### 论文摘要

Medical imaging is a critical initial tool used by clinicians to determine a patient's cancer diagnosis, allowing for faster intervention and more reliable patient prognosis. At subsequent stages of patient diagnosis, genetic information is extracted to help select specific patient treatment options. As the efficacy of cancer treatment often relies on early diagnosis and treatment, we build a deep latent variable model to determine patients' somatic mutation profiles based on their corresponding medical images. We first introduce a point cloud representation of lesions images to allow for invariance to the imaging modality. We then propose, LLOST, a model with dual variational autoencoders coupled together by a separate shared latent space that unifies features from the lesion point clouds and counts of distinct somatic mutations. Therefore our model consists of three latent space, each of which is learned with a conditional normalizing flow prior to account for the diverse distributions of each domain. We conduct qualitative and quantitative experiments on de-identified medical images from The Cancer Imaging Archive and the corresponding somatic mutations from the Pan Cancer dataset of The Cancer Genomic Archive. We show the model's predictive performance on the counts of specific mutations as well as it's ability to accurately predict the occurrence of mutations. In particular, shared patterns between the imaging and somatic mutation domain that reflect cancer type. We conclude with a remark on how to improve the model and possible future avenues of research to include other genetic domains.

---

## 149. Tada-DIP: Input-adaptive Deep Image Prior for One-shot 3D Image Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.03962v1](http://arxiv.org/abs/2512.03962v1)

**作者:** Evan Bell, Shijun Liang, Ismail Alkhouri, Saiprasad Ravishankar

**发布时间:** 2025-12-03

**备注:** 6 pages, 8 figures, 2025 Asilomar Conference on Signals, Systems, and Computers. Code is available at github.com/evanbell02/Tada-DIP/

### GPT解析

### 总结

这项研究提出了Tada-DIP，一种用于3D图像重建的高效且完全3D的DIP方法，通过结合输入适应和去噪正则化，能够产生高质量重建结果并避免过拟合现象。

### 背景

Deep Image Prior(DIP)是一种基于神经网络的一次性图像重建方法，前景良好，但在3D图像重建问题上的应用有限。

### 目的

开发一种用于解决3D逆问题的高效且完全3D的DIP方法，以克服现有DIP方法在3D重建中的局限性。

### 方法

Tada-DIP是一种结合了输入适应和去噪正则化的3D DIP方法，用于解决3D逆问题，能够产生高质量的3D重建结果并避免过拟合现象。

### 主要发现

在稀视角X射线计算机断层扫描重建实验中，Tada-DIP比无需训练数据的基线方法产生更好的重建结果，并且性能与使用完全采样体积的大数据集训练的有监督网络相当。

### 结论

Tada-DIP是一种有效的3D图像重建方法，无需大量训练数据即可达到与有监督网络相当的重建质量，为3D逆问题提供了新的解决方案。

### 翻译

深度图像优先(DIP)最近已成为一种有前景的基于神经网络的一次性图像重建方法。然而，DIP在3D图像重建问题上的应用有限。在这项工作中，我们介绍了Tada-DIP，一种用于解决3D逆问题的高效且完全3D的DIP方法。通过结合输入适应和去噪正则化，Tada-DIP能够产生高质量的3D重建结果，同时避免了DIP中常见的过拟合现象。在稀视角X射线计算机断层扫描重建上的实验验证了所提出方法的有效性，表明Tada-DIP比无需训练数据的基线方法产生更好的重建结果，并且性能与使用完全采样体积的大数据集训练的有监督网络相当。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D图像重建中的挑战，特别是针对稀疏采样数据的3D图像重建问题。这个问题在现实中非常重要，因为在医疗成像等领域（如CT和MRI），减少辐射暴露或扫描时间对患者安全至关重要，但会导致数据不足的问题。开发不需要大量训练数据的重建方法可以降低应用门槛，提高医疗诊断的准确性和效率，对患者护理有直接影响。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有DIP方法的局限性（容易过拟合、在3D重建上应用有限）和成功DIP算法的关键组件（正则化和输入更新）来设计方法。他们借鉴了多个现有工作：DeepRED使用外部去噪器进行正则化，SGLD-DIP在优化过程中注入噪声，Self-Guided DIP训练网络作为去噪器，aSeq-DIP通过更新网络输入来加速优化。作者结合这些思想，专门针对3D重建问题设计了Tada-DIP方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合输入适应和去噪正则化来产生高质量的3D重建，同时避免DIP中常见的过拟合现象。整体流程包括：初始化3D U-Net和随机输入；在每次迭代中计算自适应噪声水平，生成高斯噪声并注入到网络输入；计算包含数据保真度和去噪正则化的损失函数；更新网络参数和输入（将输入设置为前一个输入和当前输出的线性组合）；经过多次迭代后输出最终重建结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：完全3D的DIP实现，处理256³大小的体积；输入适应机制，在每次迭代中更新网络输入；自适应噪声注入，噪声水平根据输入规模动态调整；去噪正则化，结合数据保真度和去噪正则化。相比之前工作，Tada-DIP解决了原始DIP在3D重建上的局限性，处理了更大的体积，使用了更先进的正则化方法，实现了输入适应机制，在性能上与使用大量数据训练的监督方法相当，同时不需要训练数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Tada-DIP通过结合输入适应和去噪正则化，实现了无需训练数据的高质量3D图像重建，在稀疏视图CT重建中达到了与监督方法相当的性能，同时解决了传统DIP方法在3D重建中的过拟合问题。'}


### 论文摘要

Deep Image Prior (DIP) has recently emerged as a promising one-shot neural-network based image reconstruction method. However, DIP has seen limited application to 3D image reconstruction problems. In this work, we introduce Tada-DIP, a highly effective and fully 3D DIP method for solving 3D inverse problems. By combining input-adaptation and denoising regularization, Tada-DIP produces high-quality 3D reconstructions while avoiding the overfitting phenomenon that is common in DIP. Experiments on sparse-view X-ray computed tomography reconstruction validate the effectiveness of the proposed method, demonstrating that Tada-DIP produces much better reconstructions than training-data-free baselines and achieves reconstruction performance on par with a supervised network trained using a large dataset with fully-sampled volumes.

---

## 150. Automatic Attack Discovery for Few-Shot Class-Incremental Learning via Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.03882v1](http://arxiv.org/abs/2512.03882v1)

**作者:** Haidong Kang, Wei Wu, Hanling Wang

**发布时间:** 2025-12-03

### GPT解析

### 总结

该论文研究了少样本类增量学习(FSCIL)中的安全问题，提出了一种名为ACraft的自动攻击方法，利用大语言模型和强化学习来生成针对FSCIL的有效攻击方法。

### 背景

少样本类增量学习(FSCIL)是一个更加现实且具有挑战性的持续学习范式，它旨在仅用少量训练样本增量地学习未见过的类别，同时克服对基础类别的灾难性遗忘。以往的研究主要集中在开发更有效的FSCIL方法上，而对FSCIL中的安全问题关注较少。

### 目的

提供对攻击方法影响FSCIL的全面研究，并开发一种专门针对FSCIL的有效攻击方法，解决现有攻击方法无法有效攻击基础类别或依赖大量专家知识导致成本高昂的问题。

### 方法

提出了一种名为ACraft的简单而有效的方法，利用大语言模型(LLMs)自动引导和发现针对FSCIL的最优攻击方法，无需人类专家参与。此外，引入了一种基于近端策略优化(PPO)的新型强化学习方法来优化学习过程，通过建立正反馈机制使LLMs在下一代生成更好的攻击方法。

### 主要发现

人类专家设计的攻击方法(如PGD、FGSM)要么无法攻击基础类别，要么由于依赖大量专家知识而面临巨大的劳动成本，这表明需要为FSCIL设计专门的攻击方法。

### 结论

ACraft方法能够显著降低最先进FSCIL方法的性能，远超人类专家设计的攻击方法，同时保持最低的攻击成本，证明了其在FSCIL安全研究中的有效性。

### 翻译

少样本类增量学习(FSCIL)是持续学习中一个更加现实且具有挑战性的范式，它旨在仅用少量训练样本增量地学习未见过的类别，同时克服对基础类别的灾难性遗忘。以往的努力主要集中在研究更有效的FSCIL方法上。相比之下，较少关注思考对FSCIL安全问题的贡献。本文旨在全面研究攻击对FSCIL的影响。我们首先通过系统探索人类专家设计的攻击方法(即PGD、FGSM)如何影响FSCIL来获得见解。我们发现这些方法要么无法攻击基础类别，要么由于依赖大量专家知识而面临巨大劳动成本。这凸显了需要为FSCIL设计专门的攻击方法。基于这些见解，本文提出了一种简单而有效的ACraft方法，利用大语言模型(LLMs)自动引导和发现针对FSCIL的最优攻击方法，无需人类专家参与。此外，为了提高LLMs与FSCIL之间的推理能力，我们引入了一种基于近端策略优化(PPO)的新型强化学习来优化学习过程，通过建立正反馈使LLMs在下一代生成更好的攻击方法。主流基准实验表明，我们的ACraft显著降低了最先进FSCIL方法的性能，远超人类专家设计的攻击方法，同时保持最低的攻击成本。


### 论文摘要

Few-shot class incremental learning (FSCIL) is a more realistic and challenging paradigm in continual learning to incrementally learn unseen classes and overcome catastrophic forgetting on base classes with only a few training examples. Previous efforts have primarily centered around studying more effective FSCIL approaches. By contrast, less attention was devoted to thinking the security issues in contributing to FSCIL. This paper aims to provide a holistic study of the impact of attacks on FSCIL. We first derive insights by systematically exploring how human expert-designed attack methods (i.e., PGD, FGSM) affect FSCIL. We find that those methods either fail to attack base classes, or suffer from huge labor costs due to relying on huge expert knowledge. This highlights the need to craft a specialized attack method for FSCIL. Grounded in these insights, in this paper, we propose a simple yet effective ACraft method to automatically steer and discover optimal attack methods targeted at FSCIL by leveraging Large Language Models (LLMs) without human experts. Moreover, to improve the reasoning between LLMs and FSCIL, we introduce a novel Proximal Policy Optimization (PPO) based reinforcement learning to optimize learning, making LLMs generate better attack methods in the next generation by establishing positive feedback. Experiments on mainstream benchmarks show that our ACraft significantly degrades the performance of state-of-the-art FSCIL methods and dramatically beyond human expert-designed attack methods while maintaining the lowest costs of attack.

---

## 151. SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting

**论文链接:** [http://arxiv.org/abs/2512.03620v1](http://arxiv.org/abs/2512.03620v1)

**作者:** Hanxiu Zhang, Yue Zheng

**发布时间:** 2025-12-03

### GPT解析

### 总结

SELF是一种新型基于内在权重的指纹识别方案，用于解决大型语言模型知识产权保护问题，无需依赖输入且能抵抗虚假声明攻击。

### 背景

在当代AI研究中，大型语言模型的知识产权保护面临关键挑战，现有指纹识别技术（无论是行为还是结构方法）存在虚假声明攻击和易受权重操作等漏洞。

### 目的

克服现有方法的局限性，提出一种能消除对输入依赖且本质上抵抗虚假声明的知识产权保护方案。

### 方法

SELF通过两个关键创新实现：1) 通过对LLM注意力权重的奇异值和特征值分解实现独特、可扩展且变换不变的指纹提取；2) 基于少样本学习和数据增强的有效神经网络指纹相似度比较。

### 主要发现

实验表明SELF在保持高知识产权侵权检测准确度的同时，对量化、剪枝和微调攻击等下游修改表现出强大的鲁棒性。

### 结论

SELF代码已在GitHub平台开源，可供进一步研究和应用。

### 翻译

大型语言模型中的知识产权保护代表了当代AI研究中的一个关键挑战。虽然指纹识别技术已成为检测未授权模型使用的基本机制，但现有方法——无论是基于行为还是基于结构——都存在虚假声明攻击等漏洞或容易受到权重操作的影响。为了克服这些局限性，我们提出了SELF，一种新颖的基于内在权重的指纹识别方案，它消除了对输入的依赖并本质上能抵抗虚假声明。SELF通过两个关键创新实现了强大的知识产权保护：1) 通过对LLM注意力权重的奇异值和特征值分解，实现独特、可扩展且变换不变的指纹提取；2) 基于少样本学习和数据增强的有效神经网络指纹相似度比较。实验结果表明，SELF在保持高知识产权侵权检测准确度的同时，对各种下游修改（包括量化、剪枝和微调攻击）表现出强大的鲁棒性。我们的代码可在https://github.com/HanxiuZhang/SELF_v2获取。


### 论文摘要

The protection of Intellectual Property (IP) in Large Language Models (LLMs) represents a critical challenge in contemporary AI research. While fingerprinting techniques have emerged as a fundamental mechanism for detecting unauthorized model usage, existing methods -- whether behavior-based or structural -- suffer from vulnerabilities such as false claim attacks or susceptible to weight manipulations. To overcome these limitations, we propose SELF, a novel intrinsic weight-based fingerprinting scheme that eliminates dependency on input and inherently resists false claims. SELF achieves robust IP protection through two key innovations: 1) unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights, and 2) effective neural network-based fingerprint similarity comparison based on few-shot learning and data augmentation. Experimental results demonstrate SELF maintains high IP infringement detection accuracy while showing strong robustness against various downstream modifications, including quantization, pruning, and fine-tuning attacks. Our code is available at https://github.com/HanxiuZhang/SELF_v2.

---

## 152. Continuous Prompts: LLM-Augmented Pipeline Processing over Unstructured Streams

**论文链接:** [http://arxiv.org/abs/2512.03389v1](http://arxiv.org/abs/2512.03389v1)

**作者:** Shu Chen, Deepti Raghavan, Uğur Çetintemel

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文介绍了Continuous Prompts (CPs)框架，首次将LLM推理引入连续流处理，扩展RAG到流设置，定义连续语义操作符，提供多种实现方法，并通过优化技术和动态框架平衡效率与精度。

### 背景

监控非结构化流数据越来越需要持久、语义感知的计算，但当前的LLM框架是无状态的且一次性使用，限制了它们在长期分析中的实用性。

### 目的

开发一个将LLM推理引入连续流处理的框架，解决当前LLM框架在长期分析中的局限性，并优化效率与精度的权衡。

### 方法

引入Continuous Prompts框架，扩展RAG到流设置，定义连续语义操作符，提供多种实现方法，研究元组批处理和操作符融合两种优化技术，提出使用轻量级影子执行和感知成本的多目标贝叶斯优化的动态框架，并在VectraFlow流处理系统中实现。

### 主要发现

VectraFlow能够适应工作负载动态，导航精度-效率权衡，并在不断演化的非结构化流上维持持久的语义查询。

### 结论

Continuous Prompts框架成功将LLM推理引入连续流处理，通过优化技术和动态框架实现了效率与精度的平衡，使系统能够适应不断变化的数据流并维持持久的语义分析。

### 翻译

监控非结构化流越来越需要持久、语义感知的计算，然而当今的LLM框架仍然是无状态的和一次性的，限制了它们在长期分析中的实用性。我们引入了Continuous Prompts (CPs)，这是第一个将LLM推理引入连续流处理的框架。CPs将RAG扩展到流设置，定义了连续语义操作符，并提供了多种实现方法，主要关注基于LLM的方法，但也报告了一种基于嵌入的变体。此外，我们研究了两种以LLM为中心的优化：元组批处理和操作符融合，在显著提高效率的同时管理精度损失。因为这些优化本质上以精度换取速度，我们提出了一个动态优化框架，使用轻量级影子执行和感知成本的多目标贝叶斯优化来学习吞吐量-精度前沿并在探测预算下调整计划。我们在VectraFlow流处理系统中实现了CPs。通过真实数据集上的操作级微基准测试和流管道，我们表明VectraFlow能够适应工作负载动态，导航精度-效率权衡，并在不断演化的非结构化流上维持持久的语义查询。


### 论文摘要

Monitoring unstructured streams increasingly requires persistent, semantics-aware computation, yet today's LLM frameworks remain stateless and one-shot, limiting their usefulness for long-running analytics. We introduce Continuous Prompts (CPs), the first framework that brings LLM reasoning into continuous stream processing. CPs extend RAG to streaming settings, define continuous semantic operators, and provide multiple implementations, primarily focusing on LLM-based approaches but also reporting one embedding-based variants. Furthermore, we study two LLM-centric optimizations, tuple batching and operator fusion, to significantly improve efficiency while managing accuracy loss.   Because these optimizations inherently trade accuracy for speed, we present a dynamic optimization framework that uses lightweight shadow executions and cost-aware multi-objective Bayesian optimization (MOBO) to learn throughput-accuracy frontiers and adapt plans under probing budgets.   We implement CPs in the VectraFlow stream processing system. Using operator-level microbenchmarks and streaming pipelines on real datasets, we show that VectraFlow can adapt to workload dynamics, navigate accuracy-efficiency trade-offs, and sustain persistent semantic queries over evolving unstructured streams.

---

## 153. DAWZY: A New Addition to AI powered "Human in the Loop" Music Co-creation

**论文链接:** [http://arxiv.org/abs/2512.03289v1](http://arxiv.org/abs/2512.03289v1)

**作者:** Aaron C Elkins, Sanchit Singh, Adrian Kieback, Sawyer Blankenship, Uyiosa Philip Amadasun, Aman Chadha

**发布时间:** 2025-12-02

### GPT解析

### 总结

DAWZY是一个开源助手，可将自然语言（文本/语音/哼唱）请求转换为REAPER中的可逆操作。它使用基于LLM的代码生成简化用户界面，用聊天框替代传统按钮和下拉菜单，并提供三种工具进行状态查询、参数调整和节拍生成。评估显示DAWZY在常见制作任务上表现可靠，用户评价积极。

### 背景

数字音频工作站（DAWs）提供精细控制，但将高级意图（如'温暖人声'）映射到低级编辑会破坏创作流程。现有AI音乐生成器通常是单次的，限制了迭代开发和人类贡献的机会。

### 目的

开发一个助手，将自然语言请求转换为可逆操作，保持DAW作为创意中心，简化用户界面，减少学习时间，并支持迭代开发。

### 方法

DAWZY使用基于LLM的代码生成，用聊天框替代传统界面，使用三种模型上下文协议工具进行实时状态查询、参数调整和AI节拍生成。通过在变更前刷新状态保持基础，并使用原子脚本和撤销功能确保安全性和可逆性。

### 主要发现

在评估中，DAWZY在常见制作任务上表现可靠，并在可用性、控制、学习、协作和享受等方面获得了用户积极评价。

### 结论

DAWZY成功解决了DAW中高级意图映射到低级编辑的问题，提供了更直观、高效的音乐创作工具，支持迭代开发和人类贡献。

### 翻译

数字音频工作站（DAWs）提供精细控制，但将高级意图（如'温暖人声'）映射到低级编辑会破坏创作流程。现有的人工智能音乐生成器通常是单次的，限制了迭代开发和人类贡献的机会。我们提出了DAWZY，这是一个开源助手，可将自然语言（文本/语音/哼唱）请求转换为REAPER中的可逆操作。DAWZY保持DAW作为创意中心，具有最小图形界面和以语音为主的界面。DAWZY使用基于LLM的代码生成作为创新方式，显著减少用户熟悉大型界面的时间，用聊天框替代数百个按钮和下拉菜单。DAWZY还使用三种模型上下文协议工具进行实时状态查询、参数调整和AI节拍生成。它通过在变更前刷新状态来保持基础，并使用原子脚本和撤销功能确保安全性和可逆性。在评估中，DAWZY在常见制作任务上表现可靠，并在可用性、控制、学习、协作和享受等方面获得了用户积极评价。


### 论文摘要

Digital Audio Workstations (DAWs) offer fine control, but mapping high-level intent (e.g., "warm the vocals") to low-level edits breaks creative flow. Existing artificial intelligence (AI) music generators are typically one-shot, limiting opportunities for iterative development and human contribution. We present DAWZY, an open-source assistant that turns natural-language (text/voice/hum) requests into reversible actions in REAPER. DAWZY keeps the DAW as the creative hub with a minimal GUI and voice-first interface. DAWZY uses LLM-based code generation as a novel way to significantly reduce the time users spend familiarizing themselves with large interfaces, replacing hundreds of buttons and drop-downs with a chat box. DAWZY also uses three Model Context Protocol tools for live state queries, parameter adjustment, and AI beat generation. It maintains grounding by refreshing state before mutation and ensures safety and reversibility with atomic scripts and undo. In evaluations, DAWZY performed reliably on common production tasks and was rated positively by users across Usability, Control, Learning, Collaboration, and Enjoyment.

---

## 154. SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.03244v1](http://arxiv.org/abs/2512.03244v1)

**作者:** Salman Rahman, Sruthi Gorantla, Arpit Gupta, Swastik Roy, Nanyun Peng, Yang Liu

**发布时间:** 2025-12-02

### GPT解析

### 总结

SPARK是一个三阶段框架，通过生成器产生多样化解决方案，验证器使用并行和顺序扩展评估，利用合成数据训练过程奖励模型(PRM)，实现无参考的强化学习训练，超越基于真实结果的方法。

### 背景

过程奖励模型(PRM)提供密集的步骤级反馈，对强化学习有潜力，但其应用受到需要昂贵步骤级标注或真实参考的限制。

### 目的

提出一种不需要真实参考的强化学习方法，解决PRM应用受限问题。

### 方法

SPARK三阶段框架：1)生成器产生解决方案，验证器使用并行扩展(自一致性)和顺序扩展(元批判)评估；2)用验证输出作为合成数据微调生成PRM作为奖励信号；3)在数学推理RL实验中应用PRM-CoT作为奖励模型，引入格式约束防止奖励黑客攻击。

### 主要发现

在ProcessBench上达到67.5 F1分数，超过参考引导训练(66.4)和GPT-4o(61.9)；在六个数学推理基准上平均准确率47.4%，超过基于真实结果的RLVR(43.9%)。

### 结论

SPARK使无参考的强化学习训练成为可能，超越了真实结果方法，为缺乏可验证答案的领域开辟新可能性。

### 翻译

提供密集步骤级反馈的过程奖励模型(PRM)已显示出对强化学习的潜力，但其应用仍受到需要昂贵步骤级标注或真实参考的限制。我们提出SPARK：一个三阶段框架，在第一阶段，生成器模型产生多样化解决方案，验证器模型使用并行扩展(自一致性)和顺序扩展(元批判)评估它们。在第二阶段，我们使用这些验证输出作为合成训练数据来微调生成过程奖励模型，这些模型随后作为训练期间的奖励信号。我们表明，在步骤级别聚合多个独立验证产生的训练数据，超越了基于真实结果的监督，在ProcessBench(一个识别数学推理中错误步骤的基准)上达到67.5 F1，相比参考引导训练的66.4和GPT-4o的61.9。在最后阶段，我们在数学推理的强化学习实验中应用生成PRM与思维链验证(PRM-CoT)作为奖励模型，并引入格式约束以防止奖励黑客攻击。使用Qwen2.5-Math-7B，我们在六个数学推理基准测试上平均准确率达到47.4%，超过了基于真实结果的RLVR(43.9%)。我们的工作使无参考的强化学习训练成为可能，并超越了真实结果方法，为缺乏可验证答案或可访问真实答案的领域开辟了新的可能性。


### 论文摘要

Process reward models (PRMs) that provide dense, step-level feedback have shown promise for reinforcement learning, yet their adoption remains limited by the need for expensive step-level annotations or ground truth references. We propose SPARK: a three-stage framework where in the first stage a generator model produces diverse solutions and a verifier model evaluates them using parallel scaling (self-consistency) and sequential scaling (meta-critique). In the second stage, we use these verification outputs as synthetic training data to fine-tune generative process reward models, which subsequently serve as reward signals during training. We show that aggregating multiple independent verifications at the step level produces training data for process reward models that surpass ground-truth outcome supervision, achieving 67.5 F1 on ProcessBench (a benchmark for identifying erroneous steps in mathematical reasoning) compared to 66.4 for reference-guided training and 61.9 for GPT-4o. In the final stage, we apply our generative PRM with chain-of-thought verification (PRM-CoT) as the reward model in RL experiments on mathematical reasoning, and introduce format constraints to prevent reward hacking. Using Qwen2.5-Math-7B, we achieve 47.4% average accuracy across six mathematical reasoning benchmarks, outperforming ground-truth-based RLVR (43.9%). Our work enables reference-free RL training that exceeds ground-truth methods, opening new possibilities for domains lacking verifiable answers or accessible ground truth.

---

## 155. VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling

**论文链接:** [http://arxiv.org/abs/2512.02902v1](http://arxiv.org/abs/2512.02902v1)

**作者:** Weiqi Li, Quande Zhang, Ruifeng Zhai, Liang Lin, Guangrun Wang

**发布时间:** 2025-12-02

### GPT解析

### 总结

Vision-language-action (VLA) 模型在新相机视角和视觉干扰下性能急剧下降，主要原因是空间建模不对齐而非物理建模问题。

### 背景

Vision-language-action (VLA) 模型在分布内表现良好，但在新的相机视角和视觉干扰下性能急剧下降。

### 目的

提出一种一次性适应框架，通过轻量级、可学习的更新来重新校准视觉表示，提高模型的视角泛化能力。

### 方法

提出两种方法：1) 特征标记调制(FTM)，对视觉标记应用全局仿射变换；2) 特征线性适应(FLA)，为ViT编码器引入低秩更新。

### 主要发现

FTM方法仅用4K参数就将Libero视角准确率从48.5%提高到87.1%；FLA方法用4.7M参数实现了90.8%的成功率，以远低于微调的成本匹配了LoRA规模。

### 结论

预训练的VLA模型中存在大量未被充分利用的鲁棒性，有针对性的、最小化的视觉适应足以恢复视角泛化能力。

### 翻译

视觉-语言-动作(VLA)模型在分布内表现强劲，但在新颖的相机视角和视觉干扰下性能急剧下降。我们表明，这种脆弱性主要源于空间建模的不对齐，而非物理建模。为解决此问题，我们提出了一种一次性适应框架，通过轻量级、可学习的更新来重新校准视觉表示。我们的第一种方法，特征标记调制(FTM)，对视觉标记应用全局仿射变换，仅用4K参数就将Libero视角准确率从48.5%提高到87.1%。基于此，特征线性适应(FLA)为ViT编码器引入低秩更新，以4.7M参数实现90.8%的成功率——以远低于微调的成本匹配了LoRA规模。这些结果共同揭示了预训练VLA模型中存在大量未被充分利用的鲁棒性，并表明有针对性的、最小化的视觉适应足以恢复视角泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决Vision-Language-Action (VLA)模型在新的相机视角和视觉干扰下性能急剧下降的问题。这个问题在现实中很重要，因为它限制了VLA模型在真实世界环境中的应用，因为真实世界的视觉条件本质上是动态和不可预测的。研究上，它挑战了现有方法需要大规模数据收集或复杂架构来提高鲁棒性的假设，表明可以通过轻量级适应激活预训练模型中潜在的鲁棒性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将VLA模型分解为空间建模(Spatial Modeling)和物理建模(Physical Modeling)两个组件，假设视角变化主要影响空间建模而非物理建模。基于这一假设，他们设计了轻量级的视觉表示适应方法。作者借鉴了元学习思想，认为有效适应比大规模重训练更有效；借鉴了参数高效微调技术如LoRA，但将其专门应用于视觉编码器；还参考了几何一致性学习的方法，但采取了更轻量级的适应策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是VLA模型的脆弱性主要源于空间建模中的表示失准，而非物理建模的不足，通过轻量级的视觉表示适应可以重新激活预训练模型中潜在的鲁棒性。整体流程包括两种方法：1)特征标记调制(FTM)：在视觉标记上应用全局仿射变换(F̂ = (1 + γ) ⊙ F + β)，仅需4K参数；2)特征线性适应(FLA)：在ViT编码器上应用低秩更新(W' = W + ΔW)，使用4.7M参数。适应过程只需一个新视角的人类演示，然后应用FTM或FLA进行一次性适应。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)重新定义问题根源为空间建模失准而非物理建模不足；2)提出轻量级适应框架(FTM和FLA)；3)参数效率极高(比LoRA少99倍参数)；4)性能优异(90.8%成功率)；5)构建Libero-V基准进行统一评估。相比之前工作，我们的方法在保持相似性能的同时参数效率大幅提高；不需要替换整个视觉编码器；直接调制视觉标记而非添加可学习标记；不需要大规模多视图数据收集。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过轻量级的视觉表示适应方法，揭示了预训练VLA模型中潜在的鲁棒性，并展示了针对性的最小视觉适应足以恢复视角泛化能力，无需大规模重训练或复杂架构。'}


### 论文摘要

Vision-language-action (VLA) models achieve strong in-distribution performance but degrade sharply under novel camera viewpoints and visual perturbations. We show that this brittleness primarily arises from misalignment in Spatial Modeling, rather than Physical Modeling. To address this, we propose a one-shot adaptation framework that recalibrates visual representations through lightweight, learnable updates. Our first method, Feature Token Modulation (FTM), applies a global affine transformation to visual tokens and improves Libero viewpoint accuracy from 48.5% to 87.1% with only 4K parameters. Building on this, Feature Linear Adaptation (FLA) introduces low-rank updates to the ViT encoder, achieving 90.8% success with 4.7M parameters -- matching LoRA-scale finetuning at far lower cost. Together, these results reveal substantial untapped robustness in pretrained VLA models and demonstrate that targeted, minimal visual adaptation is sufficient to restore viewpoint generalization.

---

## 156. Distill, Forget, Repeat: A Framework for Continual Unlearning in Text-to-Image Diffusion Models

**论文链接:** [http://arxiv.org/abs/2512.02657v1](http://arxiv.org/abs/2512.02657v1)

**作者:** Naveen George, Naoki Murata, Yuhta Takida, Konda Reddy Mopuri, Yuki Mitsufuji

**发布时间:** 2025-12-02

**备注:** Preprint

### GPT解析

### 总结

本文提出了一种基于生成蒸馏的持续遗忘框架，解决了视觉生成模型在处理连续删除请求时的稳定性问题，实现了目标明确且稳定的遗忘效果。

### 背景

视觉生成模型在基于大规模网络数据集的训练上迅速增长，这与数据隐私法规和版权法（如GDPR的'被遗忘权'）产生显著冲突，需要机器遗忘技术来移除特定概念而无需重新训练整个模型。

### 目的

解决现有机器遗忘技术无法处理连续删除请求的问题，确保在一系列删除请求下进行目标明确且稳定的遗忘，避免稳定性危机和级联退化。

### 方法

引入一种新颖的基于生成蒸馏的持续遗忘框架，将每个遗忘步骤重新构造成多目标、师生蒸馏过程，利用持续学习的原则来保持模型完整性。

### 主要发现

在10步顺序基准测试中，该方法能够以更好的保真度遗忘被遗忘的概念，同时在保留概念的整体性能和图像质量方面没有显著干扰，明显优于基线方法。

### 结论

该框架为大规模生成模型的有责任部署和维护提供了可行途径，使行业能够以实用有效的方式遵守持续的数据删除请求。

### 翻译

最近基于大规模网络数据集训练的视觉生成模型的快速增长与数据隐私法规和版权法（如GDPR的'被遗忘权'）产生了显著冲突。这需要机器遗忘(MU)来移除特定概念，而无需付出昂贵的重新训练成本。然而，现有的MU技术从根本上无法处理删除请求连续出现的真实场景，这种设置被称为持续遗忘(CUL)。在持续环境中一次性应用单次方法会引发稳定性危机，导致级联退化特征：保留崩溃、对相关概念造成附带损害、生成质量急剧下降。为解决这一关键挑战，我们引入了一种新颖的基于生成蒸馏的持续遗忘框架，确保在一系列删除请求下进行目标明确且稳定的遗忘。通过将每个遗忘步骤重新构造成多目标、师生蒸馏过程，该框架利用持续学习的原则来保持模型完整性。在10步顺序基准测试中的实验表明，我们的方法能够以更好的保真度遗忘被遗忘的概念，且在保留概念的整体性能或图像质量方面没有显著干扰，明显优于基线方法。该框架为大规模生成模型的有责任部署和维护提供了可行途径，使行业能够以实用有效的方式遵守持续的数据删除请求。


### 论文摘要

The recent rapid growth of visual generative models trained on vast web-scale datasets has created significant tension with data privacy regulations and copyright laws, such as GDPR's ``Right to be Forgotten.'' This necessitates machine unlearning (MU) to remove specific concepts without the prohibitive cost of retraining. However, existing MU techniques are fundamentally ill-equipped for real-world scenarios where deletion requests arrive sequentially, a setting known as continual unlearning (CUL). Naively applying one-shot methods in a continual setting triggers a stability crisis, leading to a cascade of degradation characterized by retention collapse, compounding collateral damage to related concepts, and a sharp decline in generative quality. To address this critical challenge, we introduce a novel generative distillation based continual unlearning framework that ensures targeted and stable unlearning under sequences of deletion requests. By reframing each unlearning step as a multi-objective, teacher-student distillation process, the framework leverages principles from continual learning to maintain model integrity. Experiments on a 10-step sequential benchmark demonstrate that our method unlearns forget concepts with better fidelity and achieves this without significant interference to the performance on retain concepts or the overall image quality, substantially outperforming baselines. This framework provides a viable pathway for the responsible deployment and maintenance of large-scale generative models, enabling industries to comply with ongoing data removal requests in a practical and effective manner.

---

## 157. Leveraging Large-Scale Pretrained Spatial-Spectral Priors for General Zero-Shot Pansharpening

**论文链接:** [http://arxiv.org/abs/2512.02643v1](http://arxiv.org/abs/2512.02643v1)

**作者:** Yongchuan Cui, Peng Liu, Yi Zeng

**发布时间:** 2025-12-02

### GPT解析

### 总结

本研究提出了一种基于基础模型的预训练策略，通过利用大规模模拟数据集学习鲁棒的空间-光谱先验，解决了遥感图像融合中现有深度学习方法泛化能力差的问题。该方法在不同网络架构和卫星传感器上均表现出优异的性能。

### 背景

现有的遥感图像融合深度学习方法在应用于未见过的数据集时往往表现不佳，这主要归因于真实训练数据的有限性以及不同卫星传感器之间的域差异。

### 目的

探索基础模型在遥感图像融合中的潜力，提出一种新的预训练策略，利用大规模模拟数据集学习鲁棒的空间-光谱先验，以提高模型的泛化能力。

### 方法

1. 构建多样化的模拟数据集：通过将各种退化操作（模糊、噪声、下采样）和增强（波段生成、通道洗牌、高通滤波、颜色抖动等）应用于ImageNet中的自然图像和SkyScript中的遥感图像。2. 在模拟数据上预训练融合模型，学习可泛化的空间-光谱表示。3. 在六个数据集上评估预训练模型，使用零样本和单样本范式，并采用完全微调和冻结微调两种方法进行微调。

### 主要发现

1. 预训练策略显著提高了各种融合模型在不同卫星传感器和成像条件下的泛化性能。2. 预训练模型在零样本场景中取得优异结果。3. 预训练模型在单样本设置中展现出使用最少真实数据的显著适应能力。

### 结论

该研究为跨域全色锐化提供了实用解决方案，为遥感图像融合任务中的泛化建立了新基准，并为通过先进训练策略利用基础模型铺平了道路。

### 翻译

现有的遥感图像融合深度学习方法在应用于未见过的数据集时，由于真实训练数据的有限性以及不同卫星传感器之间的域差异，往往泛化能力较差。为应对这一挑战，我们通过提出一种新的预训练策略来探索基础模型的潜力，该策略利用大规模模拟数据集学习鲁棒的空间-光谱先验。具体而言，我们的方法首先通过将各种退化操作（模糊、噪声、下采样）和增强（波段生成、通道洗牌、高通滤波、颜色抖动等）应用于ImageNet中的自然图像和SkyScript中的遥感图像，构建多样化的模拟数据集。然后，我们在这些模拟数据上预训练融合模型，以学习可泛化的空间-光谱表示。随后，我们使用零样本和单样本范式，在六个数据集（WorldView-2/3/4、IKONOS、QuickBird、GaoFen-2）上评估预训练模型，并采用完全微调和冻结微调两种方法进行微调。在不同网络架构（包括卷积神经网络、Transformer和Mamba）上的大量实验表明，我们的预训练策略显著提高了各种融合模型在不同卫星传感器和成像条件下的泛化性能。预训练模型在零样本场景中取得优异结果，并在单样本设置中展现出使用最少真实数据的显著适应能力。我们的工作为跨域全色锐化提供了实用解决方案，为遥感图像融合任务中的泛化建立了新基准，并为通过先进训练策略利用基础模型铺平了道路。


### 论文摘要

Existing deep learning methods for remote sensing image fusion often suffer from poor generalization when applied to unseen datasets due to the limited availability of real training data and the domain gap between different satellite sensors. To address this challenge, we explore the potential of foundation models by proposing a novel pretraining strategy that leverages large-scale simulated datasets to learn robust spatial-spectral priors. Specifically, our approach first constructs diverse simulated datasets by applying various degradation operations (blur, noise, downsampling) and augmentations (bands generation, channel shuffling, high-pass filtering, color jittering, etc.) to natural images from ImageNet and remote sensing images from SkyScript. We then pretrain fusion models on these simulated data to learn generalizable spatial-spectral representations. The pretrained models are subsequently evaluated on six datasets (WorldView-2/3/4, IKONOS, QuickBird, GaoFen-2) using zero-shot and one-shot paradigms, with both full- and freeze-tuning approaches for fine-tuning. Extensive experiments on different network architectures including convolutional neural networks, Transformer, and Mamba demonstrate that our pretraining strategy significantly improves generalization performance across different satellite sensors and imaging conditions for various fusion models. The pretrained models achieve superior results in zero-shot scenarios and show remarkable adaptation capability with minimal real data in one-shot settings. Our work provides a practical solution for cross-domain pansharpening, establishes a new benchmark for generalization in remote sensing image fusion tasks, and paves the way for leveraging foundation models through advanced training strategies.

---

## 158. Quantum-Based Self-Attention Mechanism for Hardware-Aware Differentiable Quantum Architecture Search

**论文链接:** [http://arxiv.org/abs/2512.02476v1](http://arxiv.org/abs/2512.02476v1)

**作者:** Yuxiang Liu, Sixuan Li, Fanxu Meng, Zaichen Zhang, Xutao Yu

**发布时间:** 2025-12-02

**备注:** 15 pages,9 figures,1 table

### GPT解析

### 总结

本文提出了一种基于量化的自注意力可微分量子架构搜索(QBSA-DQAS)框架，用于解决NISQ时代变分算法中参数化量子电路自动设计的局限性，该框架在VQE任务和无线传感器网络路由中表现出优越性能。

### 背景

在NISQ时代，为变分算法设计参数化量子电路面临根本性限制，因为传统的可微分架构搜索依赖于经典模型，这些模型无法在硬件噪声下充分表示量子门相互作用。

### 目的

开发一种能够克服传统方法局限性的量子架构搜索框架，以实现更高效、更健壮的量子电路设计，特别是在存在硬件噪声的情况下。

### 方法

提出了QBSA-DQAS元学习框架，采用两阶段量子自注意力模块：通过参数化量子电路映射架构参数计算上下文依赖性，用量子导出的注意力分数替代经典相似度度量，并应用逐位置量子变换进行特征增强；架构搜索由与任务无关的多目标函数指导，共同优化噪声表达能力和成功试验概率；后搜索优化阶段应用门交换、融合和消除来减少电路复杂性。

### 主要发现

在VQE任务上表现出优越性能，H2的VQE上达到0.9的准确率(标准DQAS为0.89)；后搜索优化将电路复杂性在门数量上减少44%，深度上减少47%，而准确率没有下降；在三个分子和五个IBM量子硬件噪声模型上保持稳健性能；对于WSN路由，发现的电路比QAOA节省8.6%的能量，比经典贪心方法节省40.7%。

### 结论

量子原生架构搜索对NISQ应用的有效性得到证实，所提出的QBSA-DQAS框架能够有效处理硬件噪声并生成高效量子电路。

### 翻译

在NISQ时代，为变分算法设计参数化量子电路面临根本性限制，因为传统的可微分架构搜索依赖于经典模型，这些模型无法在硬件噪声下充分表示量子门相互作用。我们引入了基于量化的自注意力可微分量子架构搜索(QBSA-DQAS)，这是一个元学习框架，具有基于量化的自注意力和硬件感知的多目标搜索功能。该框架采用两阶段量子自注意力模块，通过参数化量子电路映射架构参数来计算上下文依赖性，用量子导出的注意力分数替代经典相似度度量，然后应用逐位置量子变换进行特征增强。架构搜索由与任务无关的多目标函数指导，共同优化噪声表达能力和成功试验概率(PST)。后搜索优化阶段应用门交换、融合和消除来减少电路复杂性。实验验证在VQE任务和大规模无线传感器网络上表现出优越性能。对于H2的VQE，QBSA-DQAS达到0.9的准确率，而标准DQAS为0.89。后搜索优化将发现的电路复杂性在门数量上减少高达44%，在深度上减少47%，而准确率没有下降。该框架在三个分子和五个IBM量子硬件噪声模型上保持稳健性能。对于WSN路由，发现的电路比QAOA节省8.6%的能量，比经典贪心方法节省40.7%，确立了量子原生架构搜索对NISQ应用的有效性。


### 论文摘要

The automated design of parameterized quantum circuits for variational algorithms in the NISQ era faces a fundamental limitation, as conventional differentiable architecture search relies on classical models that fail to adequately represent quantum gate interactions under hardware noise. We introduce the Quantum-Based Self-Attention for Differentiable Quantum Architecture Search (QBSA-DQAS), a meta-learning framework featuring quantum-based self-attention and hardware-aware multi-objective search. The framework employs a two-stage quantum self-attention module that computes contextual dependencies by mapping architectural parameters through parameterized quantum circuits, replacing classical similarity metrics with quantum-derived attention scores, then applies position-wise quantum transformations for feature enrichment. Architecture search is guided by a task-agnostic multi-objective function jointly optimizing noisy expressibility and Probability of Successful Trials (PST). A post-search optimization stage applies gate commutation, fusion, and elimination to reduce circuit complexity. Experimental validation demonstrates superior performance on VQE tasks and large-scale Wireless Sensor Networks. For VQE on H$_2$, QBSA-DQAS achieves 0.9 accuracy compared to 0.89 for standard DQAS. Post-search optimization reduces discovered circuit complexity by up to 44% in gate count and 47% in depth without accuracy degradation. The framework maintains robust performance across three molecules and five IBM quantum hardware noise models. For WSN routing, discovered circuits achieve 8.6% energy reduction versus QAOA and 40.7% versus classical greedy methods, establishing the effectiveness of quantum-native architecture search for NISQ applications.

---

## 159. Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation

**论文链接:** [http://arxiv.org/abs/2512.02441v1](http://arxiv.org/abs/2512.02441v1)

**作者:** Junghwan Park, Woojin Cho, Junhyuk Heo, Darongsae Kwon, Kookjin Lee

**发布时间:** 2025-12-02

### GPT解析

### 总结

BOLT框架通过提取正交的任务感知谱基并在该子空间中进行适应，实现了在有限资源下将预训练模型有效迁移到新任务，避免了元学习方法的高成本和不稳定性问题。

### 背景

在有限的数据和计算预算下，将大型预训练模型适应到未见任务具有挑战性。元学习方法需要额外的元训练阶段，成本高且可能不稳定。针对特定任务的预训练模型不断增加，但如何以最少额外训练将它们迁移到新任务的问题相对未被充分探索。

### 目的

提出一种框架，能够在有限资源下有效地将预训练模型适应到新任务，解决元学习方法的高成本和不稳定性问题，并探索如何利用现有的针对特定任务的预训练模型。

### 方法

BOLT（Basis-Oriented Low-rank Transfer）框架不通过合并权重来重用现有的微调模型，而是提取正交的、任务感知的谱基，并在该子空间中进行适应。离线阶段收集主导奇异方向并正交化形成可重用基；在线阶段冻结这些基，只为新任务的每层训练少量对角系数，实现秩受控的更新。

### 主要发现

在实验中，BOLT与常见的PEFT基线以及代表性的元学习初始化相比实现了稳健性能。将适应限制在任务感知的正交子空间中，为未见任务的迁移提供了有效的替代方案。

### 结论

BOLT框架通过任务感知的正交子空间约束，实现了在有限资源下有效适应预训练模型到新任务，提供了与元学习方法相当或更好的性能，同时避免了高训练成本和不稳定性问题。

### 翻译

在严格的数据和计算预算下，将大型预训练模型适应到未见任务仍然具有挑战性。元学习方法明确学习良好的初始化，但它们需要在许多任务上进行额外的元训练阶段，导致高训练成本，并且可能不稳定。同时，针对特定任务的预训练模型数量不断增加，然而如何以最少的额外训练将它们迁移到新任务的问题仍然相对未被充分探索。我们提出了BOLT（Basis-Oriented Low-rank Transfer），一个框架，它不通过合并权重来重用现有的微调模型，而是提取一个正交的、任务感知的谱基，并在该子空间中进行适应。在离线阶段，BOLT从多个任务向量中收集主导奇异方向，并对每层进行正交化以形成可重用的基。在线阶段，我们冻结这些基，只为新任务的每层训练一小组对角系数，产生秩受控的更新，且只有很少的可训练参数。这种设计为未见任务提供了(i)一个强大的、无需训练的初始化，通过汇集源任务系数，同时利用共享的正交基进行轻量级重新缩放；(ii)一种参数高效的微调(PEFT)路径，在我们的实验中，与常见的PEFT基线以及代表性的元学习初始化相比，实现了稳健的性能。我们的结果表明，将适应限制在任务感知的正交子空间中，为未见任务的迁移提供了一种有效的替代方案。


### 论文摘要

Adapting large pre-trained models to unseen tasks under tight data and compute budgets remains challenging. Meta-learning approaches explicitly learn good initializations, but they require an additional meta-training phase over many tasks, incur high training cost, and can be unstable. At the same time, the number of task-specific pre-trained models continues to grow, yet the question of how to transfer them to new tasks with minimal additional training remains relatively underexplored. We propose BOLT (Basis-Oriented Low-rank Transfer), a framework that reuses existing fine-tuned models not by merging weights, but instead by extracting an orthogonal, task-informed spectral basis and adapting within that subspace. In the offline phase, BOLT collects dominant singular directions from multiple task vectors and orthogonalizes them per layer to form reusable bases. In the online phase, we freeze these bases and train only a small set of diagonal coefficients per layer for the new task, yielding a rank-controlled update with very few trainable parameters. This design provides (i) a strong, training-free initialization for unseen tasks, obtained by pooling source-task coefficients, along with a lightweight rescaling step while leveraging the shared orthogonal bases, and (ii) a parameter-efficient fine-tuning (PEFT) path that, in our experiments, achieves robust performance compared to common PEFT baselines as well as a representative meta-learned initialization. Our results show that constraining adaptation to a task-informed orthogonal subspace provides an effective alternative for unseen-task transfer.

---

## 160. Boosting Medical Vision-Language Pretraining via Momentum Self-Distillation under Limited Computing Resources

**论文链接:** [http://arxiv.org/abs/2512.02438v1](http://arxiv.org/abs/2512.02438v1)

**作者:** Phuc Pham, Nhu Pham, Ngoc Quoc Ly

**发布时间:** 2025-12-02

**备注:** WACV 2026

### GPT解析

### 总结

该研究提出了一种结合动量方法和知识蒸馏的视觉语言模型训练方法，旨在解决医疗保健领域数据标注困难和计算资源限制的问题。通过动量自蒸馏和梯度积累技术，该方法在保持高性能的同时显著提高了训练效率。

### 背景

在医疗保健领域，获取详细标注数据具有挑战性，需要强大的视觉语言模型(VLMs)。预训练VLMs可在小数据集上微调或进行零样本推理，达到与特定任务模型相当的性能。对比学习虽是训练VLMs的关键范式，但需要大批量学习，计算量大，且医疗数据有限，训练过程中需要优先从数据和模型中提取知识。

### 目的

解决计算效率问题，提高知识利用效率，在资源有限的情况下实现高效的多模态学习，同时提升模型性能。

### 方法

利用动量方法结合知识蒸馏；使用动量自蒸馏增强多模态学习；将动量机制与梯度积累相结合，在不增加资源消耗的情况下扩大有效批量大小。

### 主要发现

在零样本分类中取得与最先进方法具有竞争力的性能；在少样本适应方面提供显著提升，AUC-ROC超过90%；检索任务性能提高2-3%；使用单个GPU实现高训练效率，同时保持合理的训练时间。

### 结论

该方法通过减少资源需求同时提高性能，推进了高效的多模态学习，为医疗保健领域提供了实用的解决方案。

### 翻译

在医疗保健领域，获取详细标注具有挑战性，凸显了对强大视觉语言模型(VLMs)的需求。预训练VLMs能够在小数据集上进行微调或进行零样本推理，实现与特定任务模型相当的性能。对比学习(CL)是训练VLMs的关键范式，但本质上需要大批量才能有效学习，这使得计算量大，通常仅限于资源丰富的机构。此外，在医疗保健数据有限的情况下，训练过程中优先从数据和模型中提取知识以提高性能至关重要。因此，我们专注于利用动量方法结合蒸馏来同时解决计算效率和知识利用问题。我们的贡献可总结如下：(1)利用动量自蒸馏增强多模态学习，(2)将动量机制与梯度积累相结合，在不增加资源消耗的情况下扩大有效批量大小。我们的方法在零样本分类中取得了与最先进(SOTA)方法具有竞争力的性能，同时在少样本适应方面提供了显著提升，AUC-ROC超过90%，检索任务性能提高了2-3%。重要的是，我们的方法使用单个GPU实现高训练效率，同时保持合理的训练时间。我们的方法旨在通过减少资源需求同时超越SOTA方法性能来推进高效的多模态学习。我们方法的实现可在https://github.com/phphuc612/MSD获取。


### 论文摘要

In medical healthcare, obtaining detailed annotations is challenging, highlighting the need for robust Vision-Language Models (VLMs). Pretrained VLMs enable fine-tuning on small datasets or zero-shot inference, achieving performance comparable to task-specific models. Contrastive learning (CL) is a key paradigm for training VLMs but inherently requires large batch sizes for effective learning, making it computationally demanding and often limited to well-resourced institutions. Moreover, with limited data in healthcare, it is important to prioritize knowledge extraction from both data and models during training to improve performance. Therefore, we focus on leveraging the momentum method combined with distillation to simultaneously address computational efficiency and knowledge exploitation. Our contributions can be summarized as follows: (1) leveraging momentum self-distillation to enhance multimodal learning, and (2) integrating momentum mechanisms with gradient accumulation to enlarge the effective batch size without increasing resource consumption. Our method attains competitive performance with state-of-the-art (SOTA) approaches in zero-shot classification, while providing a substantial boost in the few-shot adaption, achieving over 90% AUC-ROC and improving retrieval tasks by 2-3%. Importantly, our method achieves high training efficiency with a single GPU while maintaining reasonable training time. Our approach aims to advance efficient multimodal learning by reducing resource requirements while improving performance over SOTA methods. The implementation of our method is available at https://github.com/phphuc612/MSD .

---

## 161. Few-shot Protein Fitness Prediction via In-context Learning and Test-time Training

**论文链接:** [http://arxiv.org/abs/2512.02315v1](http://arxiv.org/abs/2512.02315v1)

**作者:** Felix Teufel, Aaron W. Kollasch, Yining Huang, Ole Winther, Kevin K. Yang, Pascal Notin, Debora S. Marks

**发布时间:** 2025-12-02

**备注:** AI for Science Workshop (NeurIPS 2025)

### GPT解析

### 总结

PRIMO是一种基于transformer的蛋白质适应性预测框架，通过上下文学习和测试时训练，仅需少量实验数据就能准确预测蛋白质适应性，在多种蛋白质家族和属性中表现优于现有方法。

### 背景

准确预测蛋白质适应性且使用最少的实验数据是蛋白质工程中持续面临的挑战。

### 目的

引入PRIMO框架，利用上下文学习和测试时训练，快速适应新的蛋白质和检测方法，而无需大量特定任务的数据集。

### 方法

PRIMO通过在预训练的掩码语言建模范式中，将序列信息、辅助的零样本预测以及来自多种检测的稀疏实验标签编码为统一的标记集合，并通过基于偏好的损失函数来优先选择有前景的变异体。

### 主要发现

在多种蛋白质家族和属性中，包括替换和插入缺失突变，PRIMO的性能优于零样本和完全监督的基线方法。

### 结论

将大规模预训练与高效的测试时适应相结合，可以有效解决数据收集昂贵且标签可用性有限的蛋白质设计挑战。

### 翻译

准确预测蛋白质适应性且使用最少的实验数据是蛋白质工程中持续面临的挑战。我们引入PRIMO（蛋白质上下文突变预言家），这是一个基于transformer的框架，它利用上下文学习和测试时训练，能够快速适应新的蛋白质和检测方法，而无需大量特定任务的数据集。通过在预训练的掩码语言建模范式中，将序列信息、辅助的零样本预测以及来自多种检测的稀疏实验标签编码为统一的标记集合，PRIMO学会通过基于偏好的损失函数来优先选择有前景的变异体。在多种蛋白质家族和属性中，包括替换和插入缺失突变，PRIMO的性能优于零样本和完全监督的基线方法。这项工作强调了将大规模预训练与高效的测试时适应相结合，以解决数据收集昂贵且标签可用性有限的蛋白质设计挑战的强大能力。


### 论文摘要

Accurately predicting protein fitness with minimal experimental data is a persistent challenge in protein engineering. We introduce PRIMO (PRotein In-context Mutation Oracle), a transformer-based framework that leverages in-context learning and test-time training to adapt rapidly to new proteins and assays without large task-specific datasets. By encoding sequence information, auxiliary zero-shot predictions, and sparse experimental labels from many assays as a unified token set in a pre-training masked-language modeling paradigm, PRIMO learns to prioritize promising variants through a preference-based loss function. Across diverse protein families and properties-including both substitution and indel mutations-PRIMO outperforms zero-shot and fully supervised baselines. This work underscores the power of combining large-scale pre-training with efficient test-time adaptation to tackle challenging protein design tasks where data collection is expensive and label availability is limited.

---

## 162. End-to-end machine-learned interatomic potentials for modeling functionalized mesoporous aluminosilicates

**论文链接:** [http://arxiv.org/abs/2512.02309v1](http://arxiv.org/abs/2512.02309v1)

**作者:** Jong Hyun Jung, Tom Schächtel, Yongliang Ou, Selina Itzigehl, Marc Högler, Niels Hansen, Johanna R. Bruckner, Blazej Grabowski

**发布时间:** 2025-12-02

**备注:** 15 pages, 6 figures

### GPT解析

### 总结

这篇论文开发了一种端到端的工作流程，用于为结构复杂且化学性质多样的系统（如金属硅酸盐）量身定制精确高效的机器学习势能函数，特别是矩张量势能函数(MTPs)。通过将新结构生成、表面功能化和属性评估相结合，并采用特定领域的训练策略，成功应用于铝硅酸盐系统，实验验证了模拟结果的准确性，并展示了其在预测功能化多孔铝硅酸盐红外光谱方面的适用性。

### 背景

金属硅酸盐的结构层次和化学灵活性使其具有广泛的技术应用，但也使得揭示结构-性质关系变得具有挑战性。先前的大规模原子模拟提供了机理见解，但其准确性和可实现的模型复杂性仍受限于现有的原子间势能函数。

### 目的

开发一种端到端的工作流程，用于为结构复杂且化学性质多样的系统（如金属硅酸盐）量身定制精确高效的机器学习势能函数，特别是矩张量势能函数(MTPs)，以便更准确地模拟这些材料。

### 方法

整合了从头结构生成、表面功能化和属性评估的工作流程。采用特定领域的训练策略：与熔体-淬火生成和后续功能化相关的配置训练syn-MTP，而接近平衡的配置训练eq-MTP。将工作流程应用于典型金属硅酸盐（铝硅酸盐），并实验合成和表征这些材料以用于模拟基准测试。

### 主要发现

syn-MTP可靠地生成与实验密度和同步X射线衍射测得的配对分布函数相匹配的非晶铝硅酸盐。eq-MTP再现了实验红外光谱和表面羟基密度，以及密度泛函理论推导的脱氢能量，展示了meta-GGA级别的准确性，验证了端到端工作流程。最后，通过预测功能化多孔铝硅酸盐的红外光谱，展示了所开发势能函数的适用性。

### 结论

这项研究为在操作相关条件下准确模拟真实金属硅酸盐建立了一条稳健的路径。

### 翻译

金属硅酸盐的结构层次和化学灵活性使其具有广泛的技术应用，但也使得揭示结构-性质关系变得具有挑战性。先前的大规模原子模拟提供了机理见解，但其准确性和可实现的模型复杂性仍受限于现有的原子间势能函数。在此，我们提出了一种端到端的工作流程，用于开发为结构复杂且化学性质多样的系统（如金属硅酸盐）量身定制的精确高效的机器学习势能函数，特别是矩张量势能函数(MTPs)。该工作流程整合了从头结构生成、表面功能化和属性评估。采用特定领域的训练策略：与熔体-淬火生成和后续功能化相关的配置训练syn-MTP，而接近平衡的配置训练eq-MTP。我们将该工作流程应用于典型金属硅酸盐，即铝硅酸盐，我们也实验性地合成和表征了这些材料，以用于模拟基准测试。syn-MTP可靠地生成与实验密度和通过同步X射线衍射测得的配对分布函数相匹配的非晶铝硅酸盐。eq-MTP再现了实验红外光谱和表面羟基密度，以及通过密度泛函理论推导的脱氢能量，展示了meta-GGA级别的准确性，并验证了端到端工作流程。最后，我们通过预测功能化多孔铝硅酸盐的红外光谱，展示了所开发势能函数的适用性。这项研究为在操作相关条件下准确模拟真实金属硅酸盐建立了一条稳健的路径。


### 论文摘要

The structural hierarchy and chemical flexibility of metallosilicates enable broad technological applications, yet they also make it challenging to uncover structure--property relations. Previous large-scale atomistic simulations have provided mechanistic insight, but their accuracy and achievable model complexity remain constrained by the available interatomic potentials. Here, we present an end-to-end workflow for developing accurate and efficient machine-learning potentials, specifically moment tensor potentials (MTPs), tailored for structurally and chemically complex systems such as metallosilicates. The workflow integrates de novo structure generation, surface functionalization, and property evaluation. A domain-specific training strategy is employed: Configurations associated with melt--quench generation and subsequent functionalization train the syn-MTP, whereas configurations near equilibrium train the eq-MTP. We apply the workflow to prototypical metallosilicates, i.e., aluminosilicates, which we also experimentally synthesize and characterize for benchmarking the simulations. The syn-MTP reliably generates amorphous aluminosilicates that match experimental density and pair distribution functions measured with synchrotron X-ray diffraction. The eq-MTP reproduces experimental infrared spectra and surface hydroxyl densities, along with density-functional-theory-derived dehydrogenation energies, demonstrating meta-GGA-level accuracy and validating the end-to-end workflow. Finally, we showcase the applicability of the developed potentials by predicting infrared spectra of functionalized porous aluminosilicates. This study establishes a robust path toward accurate modeling of realistic metallosilicates under operando-relevant conditions.

---

## 163. Bridging the Gap: Toward Cognitive Autonomy in Artificial Intelligence

**论文链接:** [http://arxiv.org/abs/2512.02280v1](http://arxiv.org/abs/2512.02280v1)

**作者:** Noorbakhsh Amiri Golilarz, Sindhuja Penchala, Shahram Rahimi

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文探讨了当前AI系统的局限性，特别是在自我监控、自我纠正和自主调节方面的不足，分析了七个核心缺陷，并提出了未来AI发展的展望。

### 背景

人工智能在感知、语言、推理和多模态领域取得了快速进展，但现代AI系统在动态环境中自主监控、自我纠正和调节行为的能力仍然受到根本性限制。

### 目的

识别和分析限制当代AI模型的七个核心缺陷，并展望AI如何通过模拟神经认知原则的架构来超越这些限制。

### 方法

通过比较人工系统和生物认知的分析，整合AI研究、认知科学和神经科学的见解，探讨当前模型中缺失的能力以及为什么仅靠扩展规模无法解决这些问题。

### 主要发现

当代AI模型存在七个核心缺陷：缺乏内在自我监控、缺乏元认知意识、固定且非自适应的学习机制、无法重构目标、缺乏表征维护、缺乏具身反馈以及缺乏内在能动性。

### 结论

倡导向基于认知的AI（认知自主性）转变，使其能够进行自我导向的适应、动态表征管理和有意图的、目标导向的行为，并配备改革性的监督机制，确保自主系统保持可解释性、可治理性并与人类价值观保持一致。

### 翻译

人工智能在感知、语言、推理和多模态领域取得了快速发展。然而，尽管取得了这些成就，现代AI系统在动态环境中自主监控、自我纠正和调节行为的能力仍然受到根本性限制。本文确定并分析了限制当代AI模型的七个核心缺陷：缺乏内在自我监控、缺乏元认知意识、固定且非自适应的学习机制、无法重构目标、缺乏表征维护、缺乏具身反馈以及缺乏内在能动性。在识别这些局限性的同时，我们还概述了AI如何通过模拟神经认知原则的架构超越这些局限性的前瞻性观点。我们认为，这些结构性限制阻止了当前架构（包括深度学习和基于transformer的系统）实现强大的泛化能力、终身适应性和现实世界自主性。通过对人工系统和生物认知的比较分析[7]，并整合AI研究、认知科学和神经科学的见解，我们概述了这些能力在当前模型中缺失的原因以及为什么仅靠扩展规模无法解决这些问题。最后，我们倡导向基于认知的AI（认知自主性）转变，使其具备自我导向的适应能力、动态表征管理能力和有意图的、目标导向的行为，并配备改革性的监督机制[8]，确保自主系统保持可解释性、可治理性并与人类价值观保持一致。


### 论文摘要

Artificial intelligence has advanced rapidly across perception, language, reasoning, and multimodal domains. Yet despite these achievements, modern AI systems remain fundamentally limited in their ability to self-monitor, self-correct, and regulate their behavior autonomously in dynamic contexts. This paper identifies and analyzes seven core deficiencies that constrain contemporary AI models: the absence of intrinsic self-monitoring, lack of meta-cognitive awareness, fixed and non-adaptive learning mechanisms, inability to restructure goals, lack of representational maintenance, insufficient embodied feedback, and the absence of intrinsic agency. Alongside identifying these limitations, we also outline a forward-looking perspective on how AI may evolve beyond them through architectures that mirror neurocognitive principles. We argue that these structural limitations prevent current architectures, including deep learning and transformer-based systems, from achieving robust generalization, lifelong adaptability, and real-world autonomy. Drawing on a comparative analysis of artificial systems and biological cognition [7], and integrating insights from AI research, cognitive science, and neuroscience, we outline how these capabilities are absent in current models and why scaling alone cannot resolve them. We conclude by advocating for a paradigmatic shift toward cognitively grounded AI (cognitive autonomy) capable of self-directed adaptation, dynamic representation management, and intentional, goal-oriented behavior, paired with reformative oversight mechanisms [8] that ensure autonomous systems remain interpretable, governable, and aligned with human values.

---

## 164. TT-Stack: A Transformer-Based Tiered-Stacking Ensemble Framework with Meta-Learning for Automated Breast Cancer Detection in Mammography

**论文链接:** [http://arxiv.org/abs/2512.02091v1](http://arxiv.org/abs/2512.02091v1)

**作者:** Showkat Osman, Md. Tajwar Munim Turzo, Maher Ali Rusho, Md. Makid Haider, Sazzadul Islam Sajin, Ayatullah Hasnat Behesti, Ahmed Faizul Haque Dhrubo, Md. Khurshid Jahan, Mohammad Abdul Qayum

**发布时间:** 2025-12-01

**备注:** This paper contains 15 pages with 23 figures and 4 tables. This Paper is already accepted in IEEE Computational Intelligence Magazine (CIM)

### GPT解析

### 总结

本文提出了一种名为TT-Stack（两级Transformer堆叠）的新型集成框架，结合七种轻量级视觉Transformer架构，用于在乳腺X光片中自动识别乳腺癌，取得了优异的性能。

### 背景

乳腺癌是全球癌症相关死亡的第二大常见原因，早期检测对提高患者生存率至关重要。传统计算机辅助诊断系统在特征表示能力和泛化到各种乳腺X光图像范围方面存在局限性。

### 目的

开发一种新的计算机辅助诊断系统，能够更有效地在乳腺X光片中识别乳腺癌，克服传统系统的局限性，提高检测准确率和临床实用性。

### 方法

提出TT-Stack集成框架，整合七种视觉Transformer（RepViT、DaViT、EfficientViT、MobileViT、FasterViT、MViT和PVT v2），采用两级元学习方法进行集成，通过获取基础模型logits并应用逻辑回归进行二元分类。每个模型针对单通道灰度乳腺X光片开发，同时利用ImageNet预训练的迁移学习。训练过程包括分层80/20分割、类平衡上采样、早停和自适应学习率调度。

### 主要发现

EfficientViT和PVT-v2表现最佳，达到99.33%验证准确率、97.96% F1分数和完美ROC-AUC。TT-Stack集成模型最终达到99.33%准确率、100%精确率、96%召回率、97.96% F1分数和99.97% ROC-AUC，由于架构多样性表现出稳健性能。

### 结论

TT-Stack集成框架在乳腺癌检测任务上表现出色，结合多种轻量级视觉Transformer优势，提供参数高效方法，可能适用于临床实践，且性能稳健。

### 翻译

乳腺癌继续是全球范围内癌症相关死亡的第二大常见原因，早期检测对提高患者的生存率至关重要。传统的计算机辅助诊断系统在特征表示能力和泛化到各种乳腺X光图像范围方面存在局限性。我们提出了一种新的基于使用异构轻量级视觉Transformer架构的两级Transformer堆叠（TT-Stack）集成框架，用于自动识别乳腺X光片中的乳腺癌。具体来说，我们整合了七种最先进的视觉Transformer：RepViT、DaViT、EfficientViT、MobileViT、FasterViT、MViT和PVT v2，同时设计了一个两级元学习方法进行集成，通过简单获取基础模型的logits并应用逻辑回归进行二元分类（癌症vs非癌症）。每个Transformer主干模型都开发用于处理单通道灰度乳腺X光片，同时仍然利用在ImageNet上预训练的迁移学习，以便提供一种参数高效的方法，可以在临床实践中合理应用且方差最小。


### 论文摘要

Breast cancer continues to be the second most common cause of cancer-related deaths around the world, with early detection being important to improve survival rates for patients. Traditional computer-aided diagnosis systems have limitations in their ability to represent features and generalize to the range of mammographic images. We present a new two-level Stack of Transformers (TT-Stack) ensemble framework based on using heterogeneous lightweight vision transformer architectures to automatically identify breast cancer in mammograms. Specifically, we integrate seven state-of-the-art vision transformers: RepViT, DaViT, EfficientViT, MobileViT, FasterViT, MViT, and PVT v2 while also designing a two-tier meta-learning approach for the ensemble by simply taking the logits from the base model and applying logistic regression for binary classification (Cancer vs. Non-Cancer). Each of the transformer backbone models was developed to process single-channel grayscale mammograms while still taking advantage of transfer learning from pre-training on ImageNet so that they would offer a parameter-efficient approach that may reasonably be applied in clinical practice with minimal variance. The training process included stratified 80/20 splits when necessary, class-balanced upsampling, early stopping, and an adaptive learning rate schedule on the public Mammogram Mastery dataset. In separate evaluations here, it was determined that EfficientViT and PVT-v2 were the top per-forming models achieving 99.33% validation, 97.96% F1-score, and perfect 1.000:0 ROC-AUC with only small train/validation gaps. Finally, the TT-Stack ensemble model by the end of the evaluation reached 99.33% accuracy with 100% precision, 96% recall, 97.96% F1-score and a 99.97% ROC-AUC, and demonstrated robustness in performance due to the diversity of the architecture.

---

## 165. ZIP-RC: Optimizing Test-Time Compute via Zero-Overhead Joint Reward-Cost Prediction

**论文链接:** [http://arxiv.org/abs/2512.01457v2](http://arxiv.org/abs/2512.01457v2)

**作者:** Rohin Manvi, Joey Hong, Tim Seyde, Maxime Labonne, Mathias Lechner, Sergey Levine

**发布时间:** 2025-12-01

**备注:** Code coming soon

### GPT解析

### 总结

ZIP-RC是一种自适应推理方法，使大型语言模型能够在零额外开销的情况下进行实时奖励和成本预测，从而提高推理效率并做出更智能的元认知决策。

### 背景

大型语言模型在推理方面表现出色，但缺乏内省能力，无法预测自己的成功和所需的计算量。人类使用实时内省来决定投入多少努力、何时多次尝试等，而LLM难以做出智能的元认知决策。

### 目的

开发一种能够实现自适应推理的方法，使模型能够实时评估奖励和成本，从而提高推理效率并做出更好的决策。

### 方法

ZIP-RC方法在每个token处重用保留或未使用的logits，与下一个token预测在同一前向传递中输出最终奖励和剩余长度的联合分布，不需要额外模型、架构更改或推理开销。该方法使用联合分布计算采样效用，并在推理过程中通过元动作最大化此效用。

### 主要发现

在混合难度数学基准测试中，ZIP-RC在相等或更低平均成本下，比多数投票提高多达12%的准确率，并在质量、计算和延迟之间绘制平滑的帕累托前沿。

### 结论

通过提供实时奖励-成本内省，ZIP-RC能够实现自适应、高效的推理，解决了大型语言模型缺乏内省能力的问题。

### 翻译

大型语言模型在推理方面表现出色，但缺乏内省的关键方面，包括预测自己的成功和实现成功所需的计算量。人类使用实时内省来决定投入多少努力、何时多次尝试、何时停止以及何时表明成功或失败。没有这种能力，LLM难以做出智能的元认知决策。测试时扩展方法如Best-of-N使用固定数量的样本，不考虑每个样本在任何生成点的边际收益，从而增加了成本和延迟。缺乏置信度信号可能导致误导、无法适当地升级到更好的工具，并降低可信度。学习验证器或奖励模型可以提供置信度估计，但不能实现自适应推理，并且需要额外模型或前向传递，增加了大量成本。我们提出了ZIP-RC，一种自适应推理方法，使模型能够在零额外开销的情况下预测奖励和成本。在每个token处，ZIP-RC重用保留或未使用的logits，与下一个token预测在同一前向传递中输出最终奖励和剩余长度的联合分布，不需要额外模型、架构更改或推理开销。使用这个完整的联合分布计算采样效用，即如果生成完成，一组样本的最大期望奖励、总计算和延迟的线性组合。在推理过程中，我们通过确定继续哪个token前缀或从哪里开始采样的元动作来最大化此效用。在混合难度数学基准测试中，ZIP-RC在相等或更低平均成本下，比多数投票提高多达12%的准确率，并在质量、计算和延迟之间绘制平滑的帕累托前沿。通过提供实时奖励-成本内省，ZIP-RC能够实现自适应、高效的推理。


### 论文摘要

Large language models excel at reasoning but lack key aspects of introspection, including anticipating their own success and the computation required to achieve it. Humans use real-time introspection to decide how much effort to invest, when to make multiple attempts, when to stop, and when to signal success or failure. Without this, LLMs struggle to make intelligent meta-cognition decisions. Test-time scaling methods like Best-of-N drive up cost and latency by using a fixed budget of samples regardless of the marginal benefit of each one at any point in generation, and the absence of confidence signals can mislead people, prevent appropriate escalation to better tools, and undermine trustworthiness. Learned verifiers or reward models can provide confidence estimates, but do not enable adaptive inference and add substantial cost by requiring extra models or forward passes. We present ZIP-RC, an adaptive inference method that equips models with zero-overhead inference-time predictions of reward and cost. At every token, ZIP-RC reuses reserved or unused logits in the same forward pass as next-token prediction to output a joint distribution over final reward and remaining length -- no extra models, architecture change, or inference overhead. This full joint distribution is used to compute a sampling utility which is the linear combination of the expected maximum reward, total compute, and latency of set of samples if generated to completion. During inference, we maximize this utility with meta-actions that determine which prefix of tokens to continue or initiate sampling from. On mixed-difficulty mathematical benchmarks, ZIP-RC improves accuracy by up to 12% over majority voting at equal or lower average cost, and traces smooth Pareto frontiers between quality, compute, and latency. By providing real-time reward-cost introspection, ZIP-RC enables adaptive, efficient reasoning.

---

## 166. Data-Dependent Complexity of First-Order Methods for Binary Classification

**论文链接:** [http://arxiv.org/abs/2512.03947v1](http://arxiv.org/abs/2512.03947v1)

**作者:** Matthew Hough, Stephen A. Vavasis

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究探讨了如何使用FISTA算法优化解决数据科学中的二元分类问题，通过提前停止策略显著提高计算效率。

### 背景

数据科学中的大规模问题通常通过优化建模，而优化模型通常使用一阶方法求解，这些方法可能以次线性速率收敛，导致不必要的计算开销。

### 目的

研究如何尽早终止优化算法，一旦底层的数据科学任务完成就停止，以提高计算效率。

### 方法

将FISTA算法应用于两个二元分类问题：椭球分离问题(ESP)和软间隔支持向量机(SVM)。对于ESP，将二阶锥规划转化为适合FISTA的形式；对于SVM，提出一个强凹扰动对偶，支持高效的FISTA更新。

### 主要发现

提前停止的迭代能够识别良好分类的点并产生精确分离它们的超平面；所需精度由数据几何特性控制；提出的提前停止标准减少了对难以选择的基于容差的停止条件的需求。

### 结论

在MNIST数据和SVM基准测试上的实验表明，该方法具有竞争力的运行时间，提前停止策略能带来显著的速度提升。

### 翻译

数据科学中的大规模问题通常通过优化建模，优化模型通常使用一阶方法求解，这些方法可能以次线性速率收敛。因此，一旦底层的数据科学任务完成就终止优化算法是有意义的。我们考虑使用FISTA解决两个二元分类问题：椭球分离问题(ESP)和软间隔支持向量机(SVM)。对于ESP，我们将对偶二阶锥规划转化为适合FISTA的形式，并证明FISTA残差收敛到原始-对偶混合梯度(PDHG)算法的最小位移向量，该向量直接编码了一个分离超平面。我们进一步导出一个与数据相关的迭代上界，缩放为O(1/δ_A²)，其中δ_A是破坏可分性的最小扰动。对于SVM，我们提出一个强凹扰动对偶，在线性时间投影方案下支持高效的FISTA更新，通过我们的参数选择，目标函数具有小的条件数，实现快速收敛。我们证明，在合理的数据模型下，提前停止的迭代能够识别良好分类的点并产生一个精确分离它们的超平面，其中所需的对偶迭代精度由数据的几何特性控制。特别是，所提出的提前停止标准减少了对难以选择的基于容差的停止条件的需求。我们在从MNIST数据导出的ESP实例和软间隔SVM基准测试上的数值实验表明，运行时间具有竞争力，提前停止带来了显著的速度提升。


### 论文摘要

Large-scale problems in data science are often modeled with optimization, and the optimization model is usually solved with first-order methods that may converge at a sublinear rate. Therefore, it is of interest to terminate the optimization algorithm as soon as the underlying data science task is accomplished. We consider FISTA for solving two binary classification problems: the ellipsoid separation problem (ESP), and the soft-margin support-vector machine (SVM). For the ESP, we cast the dual second-order cone program into a form amenable to FISTA and show that the FISTA residual converges to the infimal displacement vector of the primal-dual hybrid gradient (PDHG) algorithm, that directly encodes a separating hyperplane. We further derive a data-dependent iteration upper bound scaling as $\mathcal{O}(1/δ_{\mathcal{A}}^2)$, where $δ_{\mathcal{A}}$ is the minimal perturbation that destroys separability. For the SVM, we propose a strongly-concave perturbed dual that admits efficient FISTA updates under a linear time projection scheme, and with our parameter choices, the objective has small condition number, enabling rapid convergence. We prove that, under a reasonable data model, early-stopped iterates identify well-classified points and yield a hyperplane that exactly separates them, where the accuracy required of the dual iterate is governed by geometric properties of the data. In particular, the proposed early-stopping criteria diminish the need for hard-to-select tolerance-based stopping conditions. Our numerical experiments on ESP instances derived from MNIST data and on soft-margin SVM benchmarks indicate competitive runtimes and substantial speedups from stopping early.

---

## 167. A 3D virtual geographic environment for flood representation towards risk communication

**论文链接:** [http://arxiv.org/abs/2512.03839v1](http://arxiv.org/abs/2512.03839v1)

**作者:** Weilian Li, Jun Zhu, Saied Pirasteh, Qing Zhu, Yukun Guo, Lan Luo, Youness Dehbi

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种基于三维虚拟地理环境的洪水风险沟通方法，通过集成洪水建模、并行计算和三维表示技术，为非专业利益相关者提供直观易懂的洪水可视化方案。

### 背景

风险沟通旨在促进利益相关者对灾害的共同理解，提高公众意识，增强应对紧急情况的能力。然而，现有研究过分强调专业数值建模，导致非研究利益相关者难以理解和应用专业输出。

### 目的

开发一种三维虚拟地理环境，用于洪水风险沟通，使非专业利益相关者能够直观理解洪水风险及其时空变化过程。

### 方法

将洪水建模、并行计算和三维表示技术集成到一个流程中，并选择德国波恩的莱茵河一段区域进行实验验证。

### 主要发现

该方法能在几小时内完成洪水建模和三维表示，并行加速比达到6.45；三维城市模型的直观洪水场景有助于促进风险沟通，尤其对无直接洪水经验者理解洪水时空过程特别有帮助；可嵌入地理基础设施管理生态系统云应用中支持智能洪水系统。

### 结论

三维虚拟地理环境是一种有效的洪水风险沟通工具，能够提高公众对洪水风险的理解，增强应对能力。

### 翻译

风险沟通旨在促进利益相关者对灾害的共同理解，从而提高公众意识，使他们能够更有效地应对紧急情况。然而，现有研究过分强调专业数值建模，使专业输出难以被非研究利益相关者理解和使用。在此背景下，本文提出了一种用于洪水表现的三维虚拟地理环境，用于风险沟通，将洪水建模、并行计算和三维表示集成到一个流程中。最后，选择德国波恩的莱茵河一段区域进行实验分析。实验结果表明，所提出的方法能够在几小时内完成洪水建模和三维表示，并行加速比达到6.45。具有三维城市模型的直观洪水场景有助于促进洪水风险沟通，特别有助于没有直接洪水经验的参与者理解其时空过程。它还可以嵌入到地理基础设施管理生态系统(GeoIME)云应用中，用于智能洪水系统。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决洪水风险沟通中的专业信息难以被非专业人士理解的问题。这个问题很重要，因为洪水是最频繁和破坏性强的自然灾害，而有效的风险沟通能提高公众风险意识，帮助他们更好地应对紧急情况。现有的洪水信息传递方式（如2D洪水图）不够直观，难以让没有直接洪水经验的人理解洪水过程和风险。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了洪水风险管理从结构防御到非结构'软'管理的演变过程，认识到欧盟在洪水预警系统方面的领先经验。然后指出现有洪水建模工具（如MIKE、HEC-RAS）参数配置复杂，可视化与地理场景耦合不足。方法借鉴了细胞自动机(CA)模型进行洪水模拟，OpenMP实现并行计算优化，以及WebGL技术进行3D可视化，将这三者集成到一个虚拟地理环境框架中。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个3D虚拟地理环境，将洪水时空建模、并行计算和3D表示集成到一个工作流程中，既提高计算效率又提供直观可视化。整体流程包括：1)数据准备(收集DEM、卫星图像等)；2)基于CA模型的洪水时空模拟；3)使用OpenMP并行计算优化效率；4)基于WebGL的3D可视化展示洪水过程；5)最终可嵌入GeoIME云应用中。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)集成框架设计，简化了洪水建模的复杂性；2)基于OpenMP的并行计算优化，动态调度考虑负载平衡，计算效率提高(加速比达6.45)；3)WebGL-based 3D表示，无需插件即可在多种设备上使用；4)优化的数据结构支持高效洪水信息存储和传输。相比传统方法，它简化了参数配置，从2D洪水图升级为直观3D场景，计算速度更快，可访问性更强，风险沟通效果更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文开发了一个集成洪水模拟、并行计算和3D可视化的虚拟地理环境，显著提高了洪水风险沟通的效率和可理解性，使非专业人士也能直观认识洪水风险。'}


### 论文摘要

Risk communication seeks to develop a shared understanding of disaster among stakeholders, thereby amplifying public awareness and empowering them to respond more effectively to emergencies. However, existing studies have overemphasized specialized numerical modelling, making the professional output challenging to understand and use by non-research stakeholders. In this context, this article proposes a 3D virtual geographic environment for flood representation towards risk communication, which integrates flood modelling, parallel computation, and 3D representation in a pipeline. Finally, a section of the Rhine River in Bonn, Germany, is selected for experiment analysis. The experimental results show that the proposed approach is capable of flood modelling and 3D representation within a few hours, the parallel speedup ratio reached 6.45. The intuitive flood scene with 3D city models is beneficial for promoting flood risk communication and is particularly helpful for participants without direct experience of floods to understand its spatiotemporal process. It also can be embedded in the Geospatial Infrastructure Management Ecosystem (GeoIME) cloud application for intelligent flood systems.

---

## 168. Inaccessibility in Public Transit Networks

**论文链接:** [http://arxiv.org/abs/2512.03766v1](http://arxiv.org/abs/2512.03766v1)

**作者:** Katherine Betz

**发布时间:** 2025-12-03

**备注:** 12 figures, 4 tables

### GPT解析

### 总结

本研究探讨了两个主要交通系统中的基础设施可及性，通过网络分析方法揭示了系统间的显著差异，并证明了数学和网络理论方法在改善现代交通基础设施中的实用性。

### 背景

基础设施系统网络研究已受到相当关注，但这类系统的可及性（尤其是公共交通网络中的可及性）相对未被充分探索。可及性涵盖广泛考量因素，从基于基础设施的特征（如电梯和无障碍通道）到空间因素（如无障碍站的地理分布）。

### 目的

研究两个主要交通系统中的基于基础设施的可及性，分析其网络结构特性，识别主要可及性枢纽，并评估社会经济和旅游因素对无障碍站点普及程度的影响。

### 方法

构建网络模型，节点代表无障碍站点，边代表沿交通线的相邻关系；使用网络分析工具检查可及性网络的结构特性；运用中心性措施确定主要可及性枢纽；分析社会经济和旅游相关变量对无障碍站点普及度的影响。

### 主要发现

两个系统中存在显著的可及性差异，数学和网络理论方法在理解和改善现代交通基础设施方面具有实用性。

### 结论

数学和网络理论方法有助于理解和改善现代交通基础设施。

### 翻译

对基础设施系统网络的研究已受到相当关注，但这类系统的可及性，特别是公共交通网络中的可及性，仍然相对未被充分探索。可及性涵盖广泛的考量因素，从基于基础设施的特征（如电梯和无障碍通道）到空间因素（如无障碍站的地理分布）。在本研究中，我们研究两个主要交通系统中的基于基础设施的可及性：伦敦地铁和纽约地铁。我们构建网络模型，其中节点代表无障碍站点，边代表沿交通线的相邻关系。使用网络分析工具，我们检查这些可及性网络的结构特性，包括聚类模式和可及性节点的空间分布。我们进一步采用中心性措施来确定作为主要可及性枢纽的站点。最后，我们分析社会经济和旅游相关变量，评估社区财富和流行度对无障碍站点普及程度的影响。我们的研究突显了两个系统中存在显著的可及性差异，并证明了数学和网络理论方法在理解和改善现代交通基础设施方面的实用性。


### 论文摘要

The study of networks derived from infrastructure systems has received considerable attention, yet the accessibility of such systems, particularly within public transit networks, remains comparatively underexplored. Accessibility encompasses a broad range of considerations, from infrastructure-based features such as elevators and step-free access to spatial factors such as the geographic distribution of accessible stations. In this work, we investigate infrastructure-based accessibility in two major transit systems: the London Underground and the New York City Subway. We construct network models in which nodes represent accessible stations and edges represent adjacency along transit lines. Using tools from network analysis, we examine the structural properties of these accessibility networks, including clustering patterns and the spatial distribution of accessible nodes. We further employ centrality measures to identify stations that serve as major accessible hubs. Finally, we analyze socioeconomic and tourism-related variables to assess the influence of neighborhood wealth and popularity on the prevalence of accessible stations. Our findings highlight significant disparities in accessibility across both systems and demonstrate the utility of mathematical and network-theoretic methods in understanding and improving modern transit infrastructure.

---

## 169. HBFormer: A Hybrid-Bridge Transformer for Microtumor and Miniature Organ Segmentation

**论文链接:** [http://arxiv.org/abs/2512.03597v1](http://arxiv.org/abs/2512.03597v1)

**作者:** Fuchen Zheng, Xinyi Chen, Weixuan Li, Quanjun Li, Junhua Zhou, Xiaojiao Guo, Xuhang Chen, Chi-Man Pun, Shoujun Zhou

**发布时间:** 2025-12-03

**备注:** 6 pages, 4 figures, 3 tables

### GPT解析

### 总结

本文提出了一种名为HBFormer的新型混合桥接Transformer架构，用于解决医学图像分割中局部注意力机制难以有效融合局部细节与全局上下文的问题，特别适用于微肿瘤和微型器官分割等具有挑战性的任务。

### 背景

医学图像分割是现代临床诊断的基石，基于移位窗口自注意力的Vision Transformer虽然在该领域建立了新基准，但其局部注意力机制难以有效融合局部细节与全局上下文，这对需要细粒度边界定义和广泛上下文理解的微肿瘤和微型器官分割任务尤为不利。

### 目的

开发一种新型架构，能够有效融合局部细节与全局上下文，提高医学图像分割特别是微肿瘤和微型器官分割的准确性。

### 方法

提出HBFormer架构，结合了经典的U型编码器-解码器框架与Swin Transformer骨干网络，核心创新是'桥接'机制，通过多尺度特征融合(MFF)解码器实现多尺度特征与全局上下文信息的融合，使用通道和空间注意模块（由膨胀和深度卷积构建）来捕获长距离依赖关系并精确细化对象边界。

### 主要发现

在多器官、肝脏肿瘤和膀胱肿瘤等具有挑战性的医学图像分割数据集上进行的全面实验表明，HBFormer实现了最先进的结果，特别在微肿瘤和微型器官分割任务上展现出卓越能力。

### 结论

HBFormer通过其创新的混合桥接设计，有效解决了传统Vision Transformer在医学图像分割中的局限性，为临床诊断提供了更精确的分割工具，特别是在处理微小结构时表现突出。

### 翻译

医学图像分割是现代临床诊断的基石。虽然利用基于移位窗口自注意力的Vision Transformer已在该领域建立了新基准，但它们常常受到一个关键限制：其局部注意力机制难以有效融合局部细节与全局上下文。这种缺陷对微肿瘤和微型器官分割等具有挑战性的任务尤为不利，因为细粒度的边界定义和广泛的上下文理解在这些任务中至关重要。为解决这一差距，我们提出了HBFormer，一种新型混合桥接Transformer架构。HBFormer的'混合'设计将经典的U型编码器-解码器框架与强大的Swin Transformer骨干网络相结合，用于稳健的分层特征提取。核心创新在于其'桥接'机制，这是多尺度特征集成的复杂枢纽。这一桥接结构通过我们新颖的多尺度特征融合(MFF)解码器得以体现。与传统对称设计不同，MFF解码器被设计用来融合来自编码器的多尺度特征和全局上下文信息。它通过通道和空间注意模块的协同组合实现这一点，这些模块由一系列膨胀和深度卷积构建。这些组件协同工作，创建了一个强大的特征桥接，能够明确捕获长距离依赖关系，并以极高的精度细化对象边界。在具有挑战性的医学图像分割数据集（包括多器官、肝脏肿瘤和膀胱肿瘤基准）上的全面实验表明，HBFormer实现了最先进的结果，展示了其在微肿瘤和微型器官分割方面的出色能力。代码和模型可在以下网址获取：https://github.com/lzeeorno/HBFormer。


### 论文摘要

Medical image segmentation is a cornerstone of modern clinical diagnostics. While Vision Transformers that leverage shifted window-based self-attention have established new benchmarks in this field, they are often hampered by a critical limitation: their localized attention mechanism struggles to effectively fuse local details with global context. This deficiency is particularly detrimental to challenging tasks such as the segmentation of microtumors and miniature organs, where both fine-grained boundary definition and broad contextual understanding are paramount. To address this gap, we propose HBFormer, a novel Hybrid-Bridge Transformer architecture. The 'Hybrid' design of HBFormer synergizes a classic U-shaped encoder-decoder framework with a powerful Swin Transformer backbone for robust hierarchical feature extraction. The core innovation lies in its 'Bridge' mechanism, a sophisticated nexus for multi-scale feature integration. This bridge is architecturally embodied by our novel Multi-Scale Feature Fusion (MFF) decoder. Departing from conventional symmetric designs, the MFF decoder is engineered to fuse multi-scale features from the encoder with global contextual information. It achieves this through a synergistic combination of channel and spatial attention modules, which are constructed from a series of dilated and depth-wise convolutions. These components work in concert to create a powerful feature bridge that explicitly captures long-range dependencies and refines object boundaries with exceptional precision. Comprehensive experiments on challenging medical image segmentation datasets, including multi-organ, liver tumor, and bladder tumor benchmarks, demonstrate that HBFormer achieves state-of-the-art results, showcasing its outstanding capabilities in microtumor and miniature organ segmentation. Code and models are available at: https://github.com/lzeeorno/HBFormer.

---

## 170. RoboScape-R: Unified Reward-Observation World Models for Generalizable Robotics Training via RL

**论文链接:** [http://arxiv.org/abs/2512.03556v1](http://arxiv.org/abs/2512.03556v1)

**作者:** Yinzhou Tang, Yu Shang, Yinuo Chen, Bingwen Wei, Xin Zhang, Shu'ang Yu, Liangzhi Shi, Chao Yu, Chen Gao, Wei Wu, Yong Li

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为RoboScape-R的框架，利用世界模型作为强化学习中的通用代理环境，通过内源性奖励机制显著提升具身策略在多样化场景中的泛化能力，相比基线方法平均实现37.5%的性能提升。

### 背景

实现可泛化的具身策略仍然是一个关键挑战。传统的策略学习范式，包括模仿学习（IL）和强化学习（RL），都难以在多样化场景中培养泛化能力。IL策略通常过度拟合特定的专家轨迹，而RL则缺乏有效的多场景泛化所需的统一且通用的奖励信号。

### 目的

提出一个利用世界模型作为强化学习范式中的通用、多功能代理环境的框架，以解决传统RL方法的局限性。

### 方法

提出RoboScape-R框架，利用世界模型作为具身环境的通用代理。引入了一种基于世界模型的新型通用奖励机制，该机制从模型对现实世界状态转换动力学的内在理解中生成'内源性'奖励。

### 主要发现

广泛的实验表明，RoboScape-R通过提供高效且通用的训练环境，有效解决了传统RL方法的局限性，显著增强了具身策略的泛化能力。相比基线方法，该方法在域外场景下平均实现了37.5%的性能提升。

### 结论

该方法为利用世界模型作为在线训练策略提供了关键见解，并证明了其在提高具身策略泛化能力方面的有效性。

### 翻译

实现可泛化的具身策略仍然是一个关键挑战。传统的策略学习范式，包括模仿学习（IL）和强化学习（RL），都难以在多样化场景中培养泛化能力。虽然IL策略通常过度拟合特定的专家轨迹，但RL却因为缺乏有效多场景泛化所需的统一通用奖励信号而受到限制。我们认为世界模型具有独特的能力，可以作为通用环境代理来解决这个问题。然而，当前的世界模型主要关注其预测观测的能力，仍然依赖于特定任务的手工设计奖励函数，因此无法提供真正通用的训练环境。针对这个问题，我们提出了RoboScape-R框架，利用世界模型作为强化学习范式内具身环境的多功能通用代理。我们引入了一种基于世界模型的新型通用奖励机制，该机制从模型对现实世界状态转换动力学的内在理解中生成'内源性'奖励。广泛的实验证明，RoboScape-R通过提供高效且通用的训练环境，有效解决了传统RL方法的局限性，显著增强了具身策略的泛化能力。我们的方法为利用世界模型作为在线训练策略提供了关键见解，并在域外场景下相比基线方法平均实现了37.5%的性能提升。


### 论文摘要

Achieving generalizable embodied policies remains a key challenge. Traditional policy learning paradigms, including both Imitation Learning (IL) and Reinforcement Learning (RL), struggle to cultivate generalizability across diverse scenarios. While IL policies often overfit to specific expert trajectories, RL suffers from the inherent lack of a unified and general reward signal necessary for effective multi-scene generalization. We posit that the world model is uniquely capable of serving as a universal environment proxy to address this limitation. However, current world models primarily focus on their ability to predict observations and still rely on task-specific, handcrafted reward functions, thereby failing to provide a truly general training environment. Toward this problem, we propose RoboScape-R, a framework leveraging the world model to serve as a versatile, general-purpose proxy for the embodied environment within the RL paradigm. We introduce a novel world model-based general reward mechanism that generates ''endogenous'' rewards derived from the model's intrinsic understanding of real-world state transition dynamics. Extensive experiments demonstrate that RoboScape-R effectively addresses the limitations of traditional RL methods by providing an efficient and general training environment that substantially enhances the generalization capability of embodied policies. Our approach offers critical insights into utilizing the world model as an online training strategy and achieves an average 37.5% performance improvement over baselines under out-of-domain scenarios.

---

## 171. Dynamical and Photometric Analysis of NGC 146 and King 14: Evidence for a Co-Moving, Unbound Cluster Pair

**论文链接:** [http://arxiv.org/abs/2512.03552v1](http://arxiv.org/abs/2512.03552v1)

**作者:** D. Bisht, Ing-Guey Jiang, W. H. Elsanhoury, K. Belwal, D. C. Cınar, A. Raj, Shraddha Biswas, Arvind K. Dattatrey, Geeta Rangwal, Devesh P. Sariya, Mohit Singh Bisht, Alok Durgapal

**发布时间:** 2025-12-03

**备注:** 22 pages, 12 figures, 6 Tables, accepted for publication in The Astronomical Journal

### GPT解析

### 总结

对NGC 146和King 14星团对进行了多波段观测研究，确定了它们的物理特性和动力学关系

### 背景

研究NGC 146-King 14星团对的性质

### 目的

理解NGC 146和King 14星团对的本质和关系

### 方法

使用Gaia DR3、Pan-STARRS1、WISE和TESS等多波长数据进行光度测量、天体测量和动力学研究，采用概率方法确定成员，进行等时线拟合和轨道积分

### 主要发现

确定了两个星团的高概率成员数量；星团年龄分别为20±5 Myr和50±10 Myr；距离分别为2.98±0.33 kpc和2.51±0.23 kpc；质量函数斜率接近Salpeter值；投影分离约9 pc，动力学分离约32 pc；两个星团共享共同的空间和运动学关联，但相对速度超过逃逸速度，不是引力束缚的

### 结论

这两个星团可能是在同一个巨大分子云中形成的，现在作为一个无引力的共同运动对存在

### 翻译

为了理解NGC 146-King 14星团对的性质，我们使用Gaia DR3、Pan-STARRS1、WISE和TESS的多波长数据进行了详细的光度测量、天体测量和动力学研究。使用概率方法，我们确定了NGC 146和King 14的770和690个高概率成员。两个星团都表现出明确的径向密度分布，符合King模型。我们从等时线拟合估计星团年龄为20±5 Myr和50±10 Myr，从视差估计距离为2.98±0.33 kpc和2.51±0.23 kpc（应用Bailer-Jones标准后）。星团表现出一致的平均自行。质量函数斜率（1.51±0.18和1.50±0.15）接近Salpeter值，消光遵循正常的银河系红化规律（RV ~ 3.1）。三维映射给出投影分离约为9 pc。使用galpy MWPotential2014模型的轨道积分显示，NGC 146和King 14在近圆形、盘状轨道上运动，具有相似的平均轨道半径（Rm ~ 9 kpc）和约255 Myr的轨道周期。约32 pc的动力学分离表明两个星团共享共同的空间和运动学关联，是一对共同运动的星团。然而，它们的相对速度超过了它们组合质量设定的逃逸速度，表明它们不是引力束缚的。TESS光曲线揭示了七颗变星，包括γ Doradus星、SPB星和食双星，但只有一颗可能是星团成员。总体而言，这些星团可能是在同一个巨大分子云中形成的，现在作为一个无引力的共同运动对存在。


### 论文摘要

To understand the nature of the NGC 146-King 14 cluster pair, we conducted a detailed photometric, astrometric, and dynamical study using multiwavelength data from Gaia DR3, Pan-STARRS1, WISE, and TESS. Using a probabilistic approach, we identified 770 and 690 high-probability members of NGC 146 and King 14, respectively. Both clusters exhibit well-defined radial density profiles consistent with King models. We estimate the cluster ages as 20 $\pm$ 5 Myr and 50 $\pm$ 10 Myr from isochrone fitting, and distances of 2.98 $\pm$ 0.33 kpc and 2.51 $\pm$ 0.23 kpc from parallaxes after applying the Bailer-Jones criteria. The clusters show consistent mean proper motions. The mass function slopes (1.51 $\pm$ 0.18 and 1.50 $\pm$ 0.15) are close to the Salpeter value, and the extinction follows a normal Galactic reddening law (RV ~ 3.1). Three-dimensional mapping gives a projected separation of ~ 9 pc. Orbit integration using the galpy MWPotential2014 model shows that NGC 146 and King 14 move in nearly circular, disk-like orbits with similar mean orbital radii (Rm ~ 9 kpc) and orbital periods of roughly 255 Myr. A dynamical separation of ~ 32 pc indicates that both clusters share a common spatial and kinematic association, consistent with a co-moving pair. However, their relative velocity exceeds the escape velocity set by their combined mass, indicating they are not gravitationally bound. TESS light curves reveal seven variable stars, including $γ$ Doradus, SPB stars, and eclipsing binaries, though only one is a likely member. Overall, the clusters likely formed within the same giant molecular cloud and now exist as an unbound co-moving pair.

---

## 172. ViDiC: Video Difference Captioning

**论文链接:** [http://arxiv.org/abs/2512.03405v1](http://arxiv.org/abs/2512.03405v1)

**作者:** Jiangtao Wu, Shihao Li, Zhaozhou Bian, Yuanxing Zhang, Jialu Chen, Runzhe Wen, An Ping, Yiwen He, Jiakai Wang, Jiaheng Liu

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究引入了视频差异描述（ViDiC）任务及其ViDiC-1K数据集，用于评估多模态大语言模型对视频对之间相似性和差异的细粒度描述能力。研究提出了双重检查清单框架，并通过对19个多模态模型的实验，揭示了它们在比较描述和差异感知上的显著性能差距。

### 背景

理解动态场景的视觉差异需要组合、空间和时间变化的比较感知能力，而现有视觉语言系统在这方面探索不足。现有的图像差异描述方法虽能描述静态图像间的语义变化，但无法捕捉运动连续性、事件演变或时间上的一致性。

### 目的

引入视频差异描述任务及其数据集，评估多模态大语言模型对视频对相似性和差异的描述能力，为视频理解、编辑感知和多模态智能中的比较推理奠定基础。

### 方法

创建ViDiC-1K数据集（包含1000个视频对和4000多个标注的比较项目，涵盖七个类别）；提出双重检查清单框架，基于LLM-as-a-Judge协议分别测量相似性和差异的准确性；对19个代表性多模态模型进行实验评估。

### 主要发现

实验表明，现有的19个代表性多模态模型在比较描述和差异感知能力上存在显著的性能差距。

### 结论

ViDiC-1K可作为具有挑战性的基准，为推进视频理解、编辑感知和多模态智能中的比较推理奠定坚实基础。

### 翻译

理解动态场景之间的视觉差异需要组合、空间和时间变化的比较感知能力——这一能力在现有的视觉语言系统中仍探索不足。虽然先前关于图像差异描述的工作使模型能够描述静态图像之间的语义变化，但这些方法无法捕捉时间上的运动连续性、事件演变或编辑一致性。我们引入了视频差异描述任务及其相应的ViDiC-1K数据集，旨在评估多模态大语言模型对视频对之间相似性和差异的细粒度描述能力。ViDiC-1K包含1000个精心挑选的视频对，标注了4000多个比较检查清单项目，涵盖七个类别：主体、风格、背景、摄影技术、运动、位置和播放技术。为确保可靠的评估，我们提出了一个双重检查清单框架，基于LLM-as-a-Judge协议，分别测量相似性和差异的准确性。对19个代表性多模态模型的实验揭示了它们在比较描述和差异感知能力上存在显著的性能差距。我们希望ViDiC-1K能成为一个具有挑战性的基准，为推进视频理解、编辑感知和多模态智能中的比较推理奠定坚实基础。


### 论文摘要

Understanding visual differences between dynamic scenes requires the comparative perception of compositional, spatial, and temporal changes--a capability that remains underexplored in existing vision-language systems. While prior work on Image Difference Captioning (IDC) has enabled models to describe semantic changes between static images, these approaches fail to capture motion continuity, event evolution, or editing consistency over time. We introduce the ViDiC (Video Difference Captioning) task and its corresponding ViDiC-1K dataset, designed to evaluate the ability of Multimodal Large Language Models (MLLMs) to provide fine-grained descriptions of similarities and differences between video pairs. ViDiC-1K comprises 1,000 curated video pairs annotated with over 4,000 comparative checklist items, covering seven categories: subject, style, background, cinematography, motion, location, and playback techniques. To ensure reliable evaluation, we propose a dual-checklist framework that measures the accuracy of similarity and difference separately, based on the LLM-as-a-Judge protocol. Experiments on nineteen representative multimodal models reveal a significant performance gap in their comparative description and difference perception abilities. We hope ViDiC-1K can be a challenging benchmark that lays a solid foundation for advancing video understanding, edit awareness, and comparative reasoning in multimodal intelligence.

---

## 173. SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation

**论文链接:** [http://arxiv.org/abs/2512.03350v1](http://arxiv.org/abs/2512.03350v1)

**作者:** Yu Yuan, Tharindu Wickremasinghe, Zeeshan Nadir, Xijun Wang, Yiheng Chi, Stanley H. Chan

**发布时间:** 2025-12-03

**备注:** Project Page: https://yuyuanspace.com/SeeU/

### GPT解析

### 总结

SeeU是一种新的视觉生成方法，通过学习4D世界的连续动态来生成未见视觉内容，克服了传统2D视觉处理的局限性。

### 背景

图像和视频是四维世界（三维空间+时间）的离散二维投影。大多数视觉理解、预测和生成方法直接在二维观察上操作，导致性能不佳。

### 目的

提出一种能够学习连续四维动态并生成未见视觉内容的新方法，实现更准确和一致的视觉理解与生成。

### 方法

SeeU采用一种新的2D→4D→2D学习框架，包括三个步骤：从稀疏和单目2D帧重建四维世界；在低秩表示和物理约束下学习连续的四维动态；将世界向前推进时间，重新投影回2D，并基于时空上下文感知生成未见区域。

### 主要发现

通过在四维中建模动态，SeeU实现了连续且物理一致的新视觉生成，在多个任务中展示了强大的潜力，包括未见时间生成、未见空间生成和视频编辑。

### 结论

SeeU方法通过建模四维动态，克服了传统二维视觉处理的局限性，实现了更准确和一致的视觉理解与生成，在多个视觉任务中表现出强大的潜力。

### 翻译

图像和视频是四维世界（三维空间+时间）的离散二维投影。大多数视觉理解、预测和生成方法直接在二维观察上操作，导致性能不佳。我们提出了SeeU，一种新颖的方法，可以学习连续的四维动态并生成未见视觉内容。SeeU背后的原理是一种新的2D→4D→2D学习框架。SeeU首先从稀疏和单目的二维帧重建四维世界（2D→4D）。然后在低秩表示和物理约束下学习连续的四维动态（离散4D→连续4D）。最后，SeeU将世界向前推进时间，在采样的时间和视点重新投影回二维，并基于时空上下文感知生成未见区域（4D→2D）。通过在四维中建模动态，SeeU实现了连续且物理一致的新视觉生成，在多个任务中展示了强大的潜力，包括未见时间生成、未见空间生成和视频编辑。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决如何从有限的2D图像/视频中理解和生成'看不见的世界'，包括看不见的时间（过去、未来、帧之间）和看不见的空间（新的视角、被遮挡区域）。这个问题很重要，因为现有方法在需要精确3D推理的场景中表现不佳，无法保持物理一致性，且难以处理相机运动与场景运动混合的复杂情况。理解动态场景对视频编辑、增强现实、机器人导航等多个应用领域至关重要。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法在2D空间建模动态场景的局限性，然后提出应该建模连续的4D动力学。他们借鉴了Shape-of-Motion方法处理稀疏输入和分离静态/动态元素，借鉴了NeRF和3DGS作为3D重建骨干，还借鉴了密集3D点跟踪和物理约束方法。在此基础上，作者创新设计了连续4D动力学模型(C4DD)使用B样条曲线和低秩参数化，以及空间-时间上下文生成阶段，形成了完整的2D→4D→2D学习框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过学习连续4D动力学（3D空间+时间）来理解和生成动态场景，超越传统2D视觉处理方法。整体流程分为三阶段：1) 动态场景重建(2D→4D)：从稀疏单目帧重建包含运动轨迹的4D世界；2) 连续4D动力学建模(离散4D→连续4D)：使用低秩参数化和B样条曲线建模平滑物理合理的运动；3) 空间-时间上下文生成(4D→2D)：利用学习动力学演化场景，结合上下文视频生成模型修复不确定区域。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出全新的2D→4D→2D学习框架；2) 设计连续4D动力学模型(C4DD)使用B样条曲线和物理约束；3) 开发空间-时间上下文生成方法。相比之前工作，SeeU超越2D建模直接在4D空间操作，明确融入物理约束确保一致性，能够解缠相机和场景运动，处理稀疏输入，并支持广泛的时空生成任务，而非仅限于特定任务如帧间插值。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "这篇论文提出了SeeU，一种通过学习连续4D动力学来理解和生成动态场景的新方法，能够在时间和空间上生成'看不见'的内容，同时确保物理一致性和3D几何准确性。"}


### 论文摘要

Images and videos are discrete 2D projections of the 4D world (3D space + time). Most visual understanding, prediction, and generation operate directly on 2D observations, leading to suboptimal performance. We propose SeeU, a novel approach that learns the continuous 4D dynamics and generate the unseen visual contents. The principle behind SeeU is a new 2D$\to$4D$\to$2D learning framework. SeeU first reconstructs the 4D world from sparse and monocular 2D frames (2D$\to$4D). It then learns the continuous 4D dynamics on a low-rank representation and physical constraints (discrete 4D$\to$continuous 4D). Finally, SeeU rolls the world forward in time, re-projects it back to 2D at sampled times and viewpoints, and generates unseen regions based on spatial-temporal context awareness (4D$\to$2D). By modeling dynamics in 4D, SeeU achieves continuous and physically-consistent novel visual generation, demonstrating strong potentials in multiple tasks including unseen temporal generation, unseen spatial generation, and video editing.

---

## 174. Push-broom Mapping of Galaxies and Supernova Remnants with the SPRITE CubeSat

**论文链接:** [http://arxiv.org/abs/2512.03329v1](http://arxiv.org/abs/2512.03329v1)

**作者:** Elena Carlson, Brian Fleming, Yi Hang Valerie Wong, Briana Indahl, Dmitry Vorobiev, Maitland Bowen, Donal O'Sullivan, Kevin France, Anne Jaskot, Jason Tumlinson, Sanchayeeta Borthakur, Michael Rutkowski, Stephan McCandliss, Ravi Sankrit, John M. O'Meara

**发布时间:** 2025-12-03

**DOI:** 10.1117/1.JATIS.11.4.045001

**备注:** Accepted for publication in the Journal of Astronomical Telescopes, Instruments, and Systems on August 28, 2025. 28 pages with 11 figures and 4 tables

### GPT解析

### 总结

SPRITE CubeSat任务将使用首个长缝轨道光谱仪，以亚角分分辨率覆盖远紫外波长，研究超新星遗迹与星际介质的相互作用，并测量低红移星形成星系的电离逃逸。

### 背景

超新星丰富并激发周围的星际介质，是星系反馈循环中的关键机制。超新星冲击波对星际介质的加热及其随后的冷却对未来恒星形成至关重要。弥散冲击加热星际介质的冷却主要由紫外发射线主导，这些冷却区域和界面在秒差距尺度上具有复杂的空间结构。映射这一冷却过程对于理解星系反馈循环至关重要，这也是2020年天体物理学十年调查的主要目标之一。

### 目的

SPRITE任务旨在通过映射超新星遗迹与周围星际介质相互作用处的关键FUV发射线，提供关于驱动星系演化的恒星反馈的新见解。同时测量约50个低红移星形成星系的电离逃逸。

### 方法

SPRITE将搭载首个长缝轨道光谱仪，具有亚角分角分辨率，覆盖远紫外波长(1000-1750埃)并可获取莱曼紫外区域。SPRITE SNR调查将使用其长缝对扩展源进行推扫映射，产生首个亚角分扩展源FUV三维数据立方体的大样本。

### 主要发现

当前模型预测SPRITE能够以10-20角秒的角分辨率检测强O VI、O IV]和C IV发射线。通过对大麦哲伦云超新星遗迹的模拟SPRITE观测，展示了SPRITE仪器在发射前的有效性，这些模型作为关键规划工具，并纳入了最终飞行前预测性能和早期扩展源数据减少管道。

### 结论

SPRITE任务将提供对星系演化中恒星反馈的新见解，通过映射超新星遗迹与星际介质相互作用处的关键FUV发射线，并测量低红移星形成星系的电离逃逸，为理解星系反馈循环提供重要数据。

### 翻译

超新星丰富并激发周围的星际介质，是星系反馈循环中的关键机制。超新星冲击波对星际介质的加热及其随后的冷却对未来恒星形成至关重要。弥散冲击加热星际介质的冷却主要由紫外发射线主导，这些冷却区域和界面在秒差距尺度上具有复杂的空间结构。映射这一冷却过程对于理解星系反馈循环至关重要，这也是2020年天体物理学十年调查的主要目标之一。超新星遗迹和再电离测试台实验CubeSat任务将搭载首个长缝轨道光谱仪，具有亚角分角分辨率，覆盖远紫外波长(1000-1750埃)并可获取莱曼紫外区域。SPRITE旨在通过映射超新星遗迹与周围星际介质相互作用处的关键FUV发射线，提供关于驱动星系演化的恒星反馈的新见解。SPRITE还将测量约50个低红移星形成星系的电离逃逸。当前模型预测SPRITE能够以10-20角秒的角分辨率检测强O VI、O IV]和C IV发射线。SPRITE SNR调查将使用其长缝对扩展源进行推扫映射，产生首个亚角分扩展源FUV三维数据立方体的大样本。在本文中，我们提出了对大麦哲伦云超新星遗迹的模拟SPRITE观测，以展示SPRITE仪器在发射和仪器调试前的有效性。这些模型作为关键规划工具，并纳入了最终飞行前预测性能和早期扩展源数据减少管道。


### 论文摘要

Supernovae (SNe) enrich and energize the surrounding interstellar medium (ISM) and are a key mechanism in the galaxy feedback cycle. The heating of the ISM by supernova shocks, and its subsequent cooling is critical to future star formation. The cooling of the diffuse shock-heated ISM is dominated by ultraviolet (UV) emission lines. These cooling regions and interfaces have complex spatial structure on sub-parsec scales. Mapping this cooling process is essential to understanding the feedback cycle of galaxies, a major goal of the 2020 Astrophysics Decadal Survey. The Supernova remnants and Proxies for ReIonization Testbed Experiment (SPRITE) CubeSat Mission will house the first long-slit orbital spectrograph with sub-arcminute angular resolution covering far ultraviolet wavelengths (FUV; 1000 - 1750 angstroms) and access to the Lyman UV (lambda < 1216 angstroms). SPRITE aims to provide new insights into the stellar feedback that drives galaxy evolution by mapping key FUV emission lines at the interaction lines between supernova remnants (SNRs) and the ambient interstellar medium (ISM). SPRITE will also measure the ionizing escape from approximately 50 low-redshift (0.16 < z < 0.4) star-forming galaxies. Current models predict SPRITE capable of detecting strong O VI, O IV], and C IV emission lines with angular resolution from 10 - 20 arcseconds. The SPRITE SNR survey will use push-broom mapping of its long-slit on extended sources to produce the first large sample of sub-arcminute 3D data cubes of extended sources in the FUV. In this paper, we present simulated SPRITE observations of Large Magellanic Cloud (LMC) SNRs to demonstrate the efficacy of the SPRITE instrument ahead of launch and instrument commissioning. These models serve as critical planning tools and incorporate the final pre-flight predicted performance of the instrument and the early extended source data reduction pipeline.

---

## 175. SpatialReasoner: Active Perception for Large-Scale 3D Scene Understanding

**论文链接:** [http://arxiv.org/abs/2512.03284v1](http://arxiv.org/abs/2512.03284v1)

**作者:** Hongpei Zheng, Shijie Li, Yanran Li, Hujun Yin

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了H²U3D数据集和SpatialReasoner框架，解决了大规模3D环境中的空间推理挑战，实现了高效的房屋规模场景理解。

### 背景

当前视觉语言模型在大规模3D环境中的空间推理仍然具有挑战性，这些模型通常局限于房间规模的场景。

### 目的

引入H²U3D数据集用于房屋规模场景理解，并提出SpatialReasoner框架实现基于文本查询的3D场景主动探索。

### 方法

H²U3D包含多层环境（最多三层楼和10-20个房间，覆盖超过300平方米），通过自动化注释流程构建层次化视觉表示并生成多样化问答对。SpatialReasoner采用两阶段训练：监督式冷启动和自适应探索奖励的强化学习。

### 主要发现

SpatialReasoner在H²U3D上取得最先进性能，优于GPT-4o和Gemini-2.5-Pro等基线模型，仅需平均3-4张图像即可获得优越结果，而基线模型需要16+张图像。

### 结论

从粗到细的主动探索范式在3D场景理解中非常有效，能够显著提高空间推理效率。

### 翻译

在大型3D环境中的空间推理对当前的视觉语言模型仍然具有挑战性，这些模型通常局限于房间规模的场景。我们引入了H²U3D（三维整体房屋理解），这是一个为房屋规模场景理解而设计的三维视觉问答数据集。H²U3D具有跨越最多三层楼和10-20个房间的多层环境，覆盖超过300平方米。通过自动化注释流程，它构建了层次化的从粗到细的视觉表示，并生成了带有思维链注释的多样化问答对。我们进一步提出了SpatialReasoner，这是一个主动感知框架，能够根据文本查询自主调用空间工具来探索3D场景。SpatialReasoner通过两阶段策略进行训练：首先是监督式冷启动，然后是使用自适应探索奖励的强化学习，该奖励促进高效探索同时避免冗余操作。大量实验表明，SpatialReasoner在H²U3D上取得了最先进的性能，优于包括GPT-4o和Gemini-2.5-Pro在内的强大基线模型。值得注意的是，与需要16+图像的基线模型相比，我们的方法平均仅使用3-4张图像就获得了优越的结果，突显了我们从粗到细的主动探索范式的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决当前视觉-语言模型在大型3D环境中的空间推理能力有限的问题，尤其是从房间规模扩展到建筑规模（多楼层、多房间）的场景。这个问题很重要，因为现实应用如智能家居、机器人导航等需要在整个建筑级别进行空间理解，而现有方法效率低下，需要处理大量不相关的视觉信息，计算负担重。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到随着3D空间扩大，与任务相关的区域通常保持不变，因此提出主动定位这些区域的解决方案。他们借鉴了HM3D数据集、Vision-Language Model（如Gemini-2.5 Pro）和Qwen3-VL-Instruct模型等现有工作，但创新性地设计了分层粗到细的视觉表示和两阶段训练策略（监督式冷启动+强化学习），以及自适应探索奖励机制来实现主动感知能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'主动感知'，即模型能自主调用空间工具（如聚焦特定区域或从特定视角渲染图像）来探索3D场景，根据文本查询定位相关信息，而非被动处理整个场景。整体流程包括：1)构建H2U3D数据集，提供分层视觉表示和思维链标注；2)两阶段训练：监督式冷启动学习正确格式和空间操作模式，强化学习阶段使用GRPO和自适应奖励机制优化探索策略；3)推理时从全局鸟瞰图开始，逐步聚焦到相关区域，形成分层探索路径。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)H2U3D数据集，首个针对大规模3D场景理解的数据集；2)SpatialReasoner主动感知框架；3)自适应探索奖励机制；4)分层粗到细的探索策略。相比之前工作，不同之处在于：从房间规模扩展到建筑规模（300+平方米），从被动感知转变为主动感知，效率显著提升（平均只需3-4张图像 vs 基线方法16+张），支持长期规划和多步推理轨迹。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SpatialReasoner通过引入H2U3D大规模3D场景数据集和主动感知框架，实现了高效的大规模3D场景理解，仅需少量图像就能超越现有方法，为具身AI在复杂现实环境中的应用奠定了基础。'}


### 论文摘要

Spatial reasoning in large-scale 3D environments remains challenging for current vision-language models, which are typically constrained to room-scale scenarios. We introduce H$^2$U3D (Holistic House Understanding in 3D), a 3D visual question answering dataset designed for house-scale scene understanding. H$^2$U3D features multi-floor environments spanning up to three floors and 10-20 rooms, covering more than 300 m$^2$. Through an automated annotation pipeline, it constructs hierarchical coarse-to-fine visual representations and generates diverse question-answer pairs with chain-of-thought annotations. We further propose SpatialReasoner, an active perception framework that autonomously invokes spatial tools to explore 3D scenes based on textual queries. SpatialReasoner is trained through a two-stage strategy: a supervised cold start followed by reinforcement learning with an adaptive exploration reward that promotes efficient exploration while discouraging redundant operations. Extensive experiments demonstrate that SpatialReasoner achieves state-of-the-art performance on H$^2$U3D, outperforming strong baselines including GPT-4o and Gemini-2.5-Pro. Notably, our method attains superior results while using only 3-4 images in total on average, compared to baselines requiring 16+ images, highlighting the effectiveness of our coarse-to-fine active exploration paradigm.

---

## 176. Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling

**论文链接:** [http://arxiv.org/abs/2512.03044v1](http://arxiv.org/abs/2512.03044v1)

**作者:** Yueru Jia, Jiaming Liu, Shengbang Liu, Rui Zhou, Wanhe Yu, Yuyang Yan, Xiaowei Chi, Yandong Guo, Boxin Shi, Shanghang Zhang

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了Video2Act框架，通过整合空间和感知运动表征来引导机器人动作学习，利用视频扩散模型(VDMs)的固有表征提取前景边界和帧间运动变化，过滤背景噪声，并将这些表征作为扩散变换器(DiT)动作头的条件输入，采用异步双系统设计提高效率。

### 背景

稳健的感知和动力学建模是现实世界机器人策略学习的基础。最近的方法使用视频扩散模型(VDMs)来增强机器人策略，提高它们对物理世界的理解和建模能力。然而，现有方法忽视了VDMs中固有的跨帧编码的连贯且物理一致的运动表征。

### 目的

解决现有方法忽视VDMs中固有的连贯且物理一致的运动表征的问题，提出Video2Act框架，通过显式整合空间和感知运动表征来有效引导机器人动作学习。

### 方法

Video2Act框架基于VDMs的固有表征，提取前景边界和帧间运动变化，同时过滤掉背景噪声和任务无关偏差。这些精细化的表征被用作扩散变换器(DiT)动作头的附加条件输入。为解决推理效率问题，提出了异步双系统设计，其中VDM作为慢速系统2，DiT头作为快速系统1，协同工作生成自适应动作。

### 主要发现

Video2Act在仿真和真实世界任务中的平均成功率分别比之前的最先进VLA方法高出7.7%和21.7%，并且表现出强大的泛化能力。

### 结论

Video2Act通过有效利用VDMs中的空间和运动表征，显著提高了机器人策略学习的性能，特别是在现实世界任务中表现出色，同时保持了高效的操作。

### 翻译

稳健的感知和动力学建模是现实世界机器人策略学习的基础。最近的方法采用视频扩散模型(VDMs)来增强机器人策略，提高它们对物理世界的理解和建模能力。然而，现有方法忽视了VDMs中固有的跨帧编码的连贯且物理一致的运动表征。为此，我们提出了Video2Act，一个通过显式整合空间和感知运动表征来有效引导机器人动作学习的框架。基于VDMs的固有表征，我们提取前景边界和帧间运动变化，同时过滤掉背景噪声和任务无关偏差。这些精细化的表征随后被用作扩散变换器(DiT)动作头的附加条件输入，使其能够推理要操作什么以及如何移动。为解决推理效率问题，我们提出了异步双系统设计，其中VDM作为慢速系统2，DiT头作为快速系统1，协同工作生成自适应动作。通过向系统1提供感知运动条件，Video2Act即使在VDM低频更新时也能保持稳定的操作。在评估方面，Video2Act在仿真和真实世界任务中的平均成功率分别比之前的最先进VLA方法高出7.7%和21.7%，进一步表现出强大的泛化能力。


### 论文摘要

Robust perception and dynamics modeling are fundamental to real-world robotic policy learning. Recent methods employ video diffusion models (VDMs) to enhance robotic policies, improving their understanding and modeling of the physical world. However, existing approaches overlook the coherent and physically consistent motion representations inherently encoded across frames in VDMs. To this end, we propose Video2Act, a framework that efficiently guides robotic action learning by explicitly integrating spatial and motion-aware representations. Building on the inherent representations of VDMs, we extract foreground boundaries and inter-frame motion variations while filtering out background noise and task-irrelevant biases. These refined representations are then used as additional conditioning inputs to a diffusion transformer (DiT) action head, enabling it to reason about what to manipulate and how to move. To mitigate inference inefficiency, we propose an asynchronous dual-system design, where the VDM functions as the slow System 2 and the DiT head as the fast System 1, working collaboratively to generate adaptive actions. By providing motion-aware conditions to System 1, Video2Act maintains stable manipulation even with low-frequency updates from the VDM. For evaluation, Video2Act surpasses previous state-of-the-art VLA methods by 7.7% in simulation and 21.7% in real-world tasks in terms of average success rate, further exhibiting strong generalization capabilities.

---

## 177. Video4Spatial: Towards Visuospatial Intelligence with Context-Guided Video Generation

**论文链接:** [http://arxiv.org/abs/2512.03040v1](http://arxiv.org/abs/2512.03040v1)

**作者:** Zeqi Xiao, Yiwei Zhao, Lingxiao Li, Yushi Lan, Yu Ning, Rahul Garg, Roshni Cooper, Mohammad H. Taghavi, Xingang Pan

**发布时间:** 2025-12-02

**备注:** Project page at https://xizaoqu.github.io/video4spatial/

### GPT解析

### 总结

研究仅使用视觉数据的视频生成模型是否能够表现出视觉空间智能，提出了Video4Spatial框架，验证了其在场景导航和对象定位两个任务上的能力，结果表明该模型能从视频上下文中展示强大的空间理解能力。

### 背景

视觉空间智能是人类认知的核心能力，研究视频生成模型是否能够表现出这种智能具有重要意义。

### 目的

探究仅使用视觉数据，视频生成模型是否能够表现出视觉空间智能，并验证其在复杂空间任务上的能力。

### 方法

提出Video4Spatial框架，该框架基于仅视频条件化的视频扩散模型，在场景导航和对象定位两个任务上进行验证，仅使用视频输入而不使用深度或姿态等辅助模态，并在框架和数据整理方面做出简单而有效的设计选择。

### 主要发现

Video4Spatial能够从视频上下文中展示强大的空间理解能力，能够端到端地规划导航和定位目标对象，能够遵循相机姿态指令同时保持空间一致性，能够推广到长上下文和域外环境。

### 结论

这些结果推进了视频生成模型向通用视觉空间推理方向发展。

### 翻译

我们研究视频生成模型是否仅使用视觉数据就能表现出视觉空间智能，这是人类认知的核心能力。为此，我们提出了Video4Spatial框架，表明仅基于视频场景条件化的视频扩散模型可以执行复杂的空间任务。我们在两个任务上进行了验证：场景导航 - 在遵循相机姿态指令的同时保持与场景3D几何的一致性，以及对象定位 - 这需要语义定位、指令遵循和规划。两个任务仅使用视频输入，没有深度或姿态等辅助模态。通过在框架和数据整理方面做出简单而有效的设计选择，Video4Spatial展示了从视频上下文中强大的空间理解能力：它端到端地规划导航和定位目标对象，遵循相机姿态指令同时保持空间一致性，并能推广到长上下文和域外环境。总而言之，这些结果推进了视频生成模型向通用视觉空间推理方向发展。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何让视频生成模型仅通过视觉数据展现视觉空间智能（visuospatial intelligence）的问题。这个问题很重要，因为视觉空间智能是人类认知的核心能力，涉及记忆、理解和在空间环境中行动的能力。现有方法通常依赖辅助信号（如深度图、相机姿态）进行空间理解，而仅从RGB视频学习空间理解具有挑战性。如果能仅通过视频实现空间智能，将使模型更接近人类感知世界的方式，并简化系统架构。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将视频生成重新定义为空间推理问题，而非简单的神经渲染任务。他们借鉴了DiT架构处理上下文和目标帧的方式，将上下文帧的扩散时间步设置为t=0。受History Guidance启发，将classifier-free guidance扩展到视频上下文。为了减少连续视频中的冗余，采用非连续帧采样和非连续RoPE技术。还借鉴了语言模型中的明确推理模式，引入辅助边界框作为推理先验以提高目标定位准确性。整体设计简洁，仅使用标准视频扩散架构和扩散目标进行训练。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将视频生成模型视为空间推理器，而非仅作为神经渲染器。给定视频上下文和指令，模型生成符合指令的连贯视频，同时保持场景几何和时间一致性。整体流程：1)输入视频上下文和指令；2)将上下文帧与噪声目标帧沿时间轴连接，上下文帧保持无噪声；3)通过交叉注意力等方式注入指令；4)模型生成符合上下文和指令的视频。主要评估两个任务：基于视频的目标定位（移动到指定对象）和场景导航（沿指定轨迹生成新视角视频）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)VIDEO4SPATIAL框架，仅依赖视频上下文执行空间任务；2)联合CFG在上下文和指令上应用；3)辅助边界框作为推理先验提高定位准确性；4)非连续上下文采样提高效率。相比之前工作，本文方法：1)完全基于原始视频，不需要显式3D信号（深度、姿态等）；2)不依赖外部3D重建或估计器；3)能从短训练上下文外推到更长推理上下文；4)尽管在室内场景训练，但能泛化到室外环境和未见对象类别；5)实现了端到端的视频空间推理，无需额外模态辅助。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Video4Spatial展示了仅通过视频上下文引导的视频生成模型能够展现强大的视觉空间智能，成功执行场景导航和目标定位任务，同时保持几何一致性和时间连贯性。'}


### 论文摘要

We investigate whether video generative models can exhibit visuospatial intelligence, a capability central to human cognition, using only visual data. To this end, we present Video4Spatial, a framework showing that video diffusion models conditioned solely on video-based scene context can perform complex spatial tasks. We validate on two tasks: scene navigation - following camera-pose instructions while remaining consistent with 3D geometry of the scene, and object grounding - which requires semantic localization, instruction following, and planning. Both tasks use video-only inputs, without auxiliary modalities such as depth or poses. With simple yet effective design choices in the framework and data curation, Video4Spatial demonstrates strong spatial understanding from video context: it plans navigation and grounds target objects end-to-end, follows camera-pose instructions while maintaining spatial consistency, and generalizes to long contexts and out-of-domain environments. Taken together, these results advance video generative models toward general visuospatial reasoning.

---

## 178. Hierarchical Process Reward Models are Symbolic Vision Learners

**论文链接:** [http://arxiv.org/abs/2512.03126v1](http://arxiv.org/abs/2512.03126v1)

**作者:** Shan Zhang, Aotian Chen, Kai Zou, Jindong Gu, Yuan Xue, Anton van den Hengel

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种新颖的自监督符号自编码器，用于将图表编码为结构化基元及其相互关系，并通过可执行引擎重建输入图表。该系统通过符号分层过程奖励建模实现一致性约束，并引入稳定机制平衡探索与利用，最终开发了一个结合神经网络推理能力和符号模型可解释性的神经符号系统。

### 背景

符号计算机视觉通过显式逻辑规则和结构化表示来表示图表，实现机器视觉的可解释理解。这需要不同于基于像素的视觉模型的学习范式，因为符号视觉学习者将图表解析为几何基元（点、线和形状），而基于像素的学习者则在纹理和颜色上操作。

### 目的

开发自监督符号自编码器，将图表编码为结构化基元及其相互关系并通过可执行引擎重建；通过符号分层过程奖励建模强制执行几何一致性；引入稳定机制平衡探索与利用；构建结合神经网络推理能力和符号模型可解释性的神经符号系统。

### 方法

提出自监督符号自编码器架构，将图表编码为结构化基元及其在潜在空间中的相互关系；实现符号分层过程奖励建模，应用分层步骤级解析奖励确保点在线上、线在形状上和形状在关系上的一致性；引入稳定机制解决强化学习在图表重建中的探索问题；在下游任务上微调符号编码器，通过基于推理的视觉奖励开发神经符号系统。

### 主要发现

在多个任务上评估显示该方法有效性：几何图表重建实现98.2%的MSE减少；使用7B模型在图表重建上超越GPT-4o达0.6%；在MathGlance感知基准测试中提高13%；在MathVerse和GeoQA推理基准测试中分别提高3%。

### 结论

该神经符号系统成功结合了神经网络的推理能力与符号模型的可解释性，在图表重建、感知和推理任务上展现出显著优势，特别是在几何理解和数学推理方面表现突出。

### 翻译

符号计算机视觉通过显式逻辑规则和结构化表示来表示图表，使机器视觉能够实现可解释的理解。这需要从根本上不同于基于像素的视觉模型的学习范式。符号视觉学习者将图表解析为几何基元（点、线和形状），而基于像素的学习者则在纹理和颜色上操作。我们提出了一种新颖的自监督符号自编码器，将图表编码为结构化基元及其在潜在空间中的相互关系，并通过我们的可执行引擎解码它们以重建输入图表。该架构的核心是符号分层过程奖励建模，它应用分层步骤级解析奖励来强制执行点在线上、线在形状上和形状在关系上的一致性。由于标准强化学习在图表重建过程中对策略空间的探索表现不佳；因此我们引入了稳定机制来平衡探索和利用。我们在下游任务上微调我们的符号编码器，开发了一个神经符号系统，通过基于推理的视觉奖励将神经网络的推理能力与符号模型的可解释性结合起来。在重建、感知和推理任务上的评估证明了我们方法的有效性：在几何图表重建中实现了98.2%的MSE减少，使用7B模型在图表重建方面比GPT-4o高出0.6%，在MathGlance感知基准测试中提高了+13%，在MathVerse和GeoQA推理基准测试中提高了+3%。


### 论文摘要

Symbolic computer vision represents diagrams through explicit logical rules and structured representations, enabling interpretable understanding in machine vision. This requires fundamentally different learning paradigms from pixel-based visual models. Symbolic visual learners parse diagrams into geometric primitives-points, lines, and shapes-whereas pixel-based learners operate on textures and colors. We propose a novel self-supervised symbolic auto-encoder that encodes diagrams into structured primitives and their interrelationships within the latent space, and decodes them through our executable engine to reconstruct the input diagrams. Central to this architecture is Symbolic Hierarchical Process Reward Modeling, which applies hierarchical step-level parsing rewards to enforce point-on-line, line-on-shape, and shape-on-relation consistency. Since vanilla reinforcement learning exhibits poor exploration in the policy space during diagram reconstruction; we thus introduce stabilization mechanisms to balance exploration and exploitation. We fine-tune our symbolic encoder on downstream tasks, developing a neuro-symbolic system that integrates the reasoning capabilities of neural networks with the interpretability of symbolic models through reasoning-grounded visual rewards. Evaluations across reconstruction, perception, and reasoning tasks demonstrate the effectiveness of our approach: achieving a 98.2% reduction in MSE for geometric diagram reconstruction, surpassing GPT-4o by 0.6% with a 7B model on chart reconstruction, and improving by +13% on the MathGlance perception benchmark, and by +3% on MathVerse and GeoQA reasoning benchmarks.

---

## 179. 论文ID: 2512.02983v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.02983v1.json'

---

## 180. 论文ID: 2512.02896v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.02896v1.json'

---

## 181. Action Anticipation at a Glimpse: To What Extent Can Multimodal Cues Replace Video?

**论文链接:** [http://arxiv.org/abs/2512.02846v1](http://arxiv.org/abs/2512.02846v1)

**作者:** Manuel Benavent-Lledo, Konstantinos Bacharidis, Victoria Manousaki, Konstantinos Papoutsakis, Antonis Argyros, Jose Garcia-Rodriguez

**发布时间:** 2025-12-02

**备注:** Accepted in WACV 2026 - Applications Track

### GPT解析

### 总结

本文提出了一种名为AAG（Action Anticipation at a Glimpse）的方法，能够在单帧图像上预测即将发生的动作，并通过多模态信息实现了与时间聚合视频方法相当的性能。

### 背景

在动作理解研究中，预测即将发生的动作是一个核心挑战。传统方法依赖于从视频中提取和聚合时间信息，而人类通常可以通过观察场景中的单一时刻来预测即将发生的动作，只要有足够的上下文。

### 目的

研究视频聚合可以在多大程度上被替代模态所取代，并开发一种能够在单一帧上预测动作的方法。

### 方法

AAG方法结合了RGB特征和来自单帧的深度线索以增强空间推理，并融入先验动作信息以提供长期上下文。这种上下文通过视觉语言模型的文本摘要或单帧动作识别器生成的预测来获取。

### 主要发现

使用AAG的多模态单帧动作预测在三个指导活动数据集（IKEA-ASM、Meccano和Assembly101）上能够与时间聚合视频基线方法和最先进方法竞争性地表现。

### 结论

模型确实可以像人类一样通过单帧预测动作，尽管其有效性取决于任务的复杂性。

### 翻译

预测即将发生的动作是动作理解研究中的一个核心挑战。虽然传统方法依赖于从视频中提取和聚合时间信息，但作为人类，我们通常可以通过观察场景中的单一时刻来预测即将发生的动作，只要有足够的上下文。模型能否实现这种能力？简短的回答是肯定的，尽管其有效性取决于任务的复杂性。在这项工作中，我们研究视频可以在多大程度上被替代模态取代。为此，基于视觉特征提取和语言推理的最新进展，我们引入了AAG，一种用于瞬间动作预测的方法。AAG将RGB特征与来自单帧的深度线索相结合以增强空间推理，并融入先前的动作信息以提供长期上下文。这种上下文是通过视觉语言模型的文本摘要或单帧动作识别器生成的预测获得的。我们的结果表明，在三个指导活动数据集（IKEA-ASM、Meccano和Assembly101）上，使用AAG的多模态单帧动作预测与时间聚合视频基线方法和最先进方法相比具有竞争力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要研究能否仅使用单帧图像（而非视频序列）结合多模态线索来有效预测人类即将执行的动作。这个问题在现实中非常重要，因为传统的视频动作预测方法计算量大，而单帧方法可以大幅减少计算成本，特别适用于自动驾驶、工业安全、人机协作等需要实时响应的场景，同时也为资源受限环境提供了可行的解决方案。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将动作预测问题分解为短期观察（单帧）和长期上下文（过去动作序列）两部分。他们借鉴了现有工作中的多种技术：使用DINOv2作为视觉编码器提取高质量特征，采用Depth Anything V2处理深度信息并转换为RGB表示，使用DistilBERT编码文本描述，以及应用注意力机制进行特征融合。作者设计了三种动作历史获取策略：VLM提示、已见动作编码和已见动作聚合，并通过实验验证了不同策略的效果。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过融合单帧视觉信息（RGB和深度）与长期上下文（过去动作序列），实现高效的动作预测，避免传统方法中对视频序列的时间聚合需求。整体流程包括：1) 视觉编码 - 使用DINOv2提取RGB特征，处理深度信息，并通过交叉注意力融合两者；2) 动作历史编码 - 获取过去3-7个动作并编码为文本表示；3) 多模态融合与分类 - 使用自注意力Transformer融合视觉和文本特征，最后通过线性分类器预测未来动作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出AAG框架实现单帧动作预测；2) 有效融合RGB、深度信息和动作历史；3) 设计三种动作历史编码方法，特别是单独编码动作类别的方法表现最佳；4) 创新应用深度信息。相比之前的工作，AAG减少了时间依赖，显著降低了计算成本和可训练参数，探索了更高效的动作历史表示方法，并使用自监督视觉编码器而非需要大量标注数据的专用模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AAG证明了通过融合单帧视觉信息与长期动作上下文，可以在保持竞争力的同时显著提高动作预测的计算效率，为实时应用提供了可行的解决方案。'}


### 论文摘要

Anticipating actions before they occur is a core challenge in action understanding research. While conventional methods rely on extracting and aggregating temporal information from videos, as humans we can often predict upcoming actions by observing a single moment from a scene, when given sufficient context. Can a model achieve this competence? The short answer is yes, although its effectiveness depends on the complexity of the task. In this work, we investigate to what extent video aggregation can be replaced with alternative modalities. To this end, based on recent advances in visual feature extraction and language-based reasoning, we introduce AAG, a method for Action Anticipation at a Glimpse. AAG combines RGB features with depth cues from a single frame for enhanced spatial reasoning, and incorporates prior action information to provide long-term context. This context is obtained either through textual summaries from Vision-Language Models, or from predictions generated by a single-frame action recognizer. Our results demonstrate that multimodal single-frame action anticipation using AAG can perform competitively compared to both temporally aggregated video baselines and state-of-the-art methods across three instructional activity datasets: IKEA-ASM, Meccano, and Assembly101.

---

## 182. SR-GRPO: Stable Rank as an Intrinsic Geometric Reward for Large Language Model Alignment

**论文链接:** [http://arxiv.org/abs/2512.02807v1](http://arxiv.org/abs/2512.02807v1)

**作者:** Yixuan Tang, Yi Yang

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种名为'stable rank'的新方法，用于对齐大型语言模型与人类偏好，该方法无需外部监督，而是从模型表示中提取内在质量信号。

### 背景

当前对齐大型语言模型与人类偏好通常依赖外部监督，但面临几个关键限制：人工标注稀缺且主观，奖励模型容易受到奖励攻击，自我评估方法则容易受到提示敏感性和偏见的影响。

### 目的

提出一种不需要外部监督的方法来评估和改进大型语言模型的质量，解决当前对齐方法中的局限性。

### 方法

1. 提出'stable rank'方法，从模型表示中推导内在质量信号；2. 通过计算总方差与主导方向方差的比率来衡量隐藏状态的有效维度；3. 引入Stable Rank Group Relative Policy Optimization (SR-GRPO)，使用stable rank作为强化学习的奖励信号。

### 主要发现

1. 在RewardBench上，stable rank达到84.04%的准确率；2. 通过Best-of-N采样，比贪婪解码平均提高11.3个百分点的任务准确率；3. 使用SR-GRPO，Qwen2.5-1.5B-Instruct模型在STEM领域提高10%，在数学推理方面提高19%，优于已学习的奖励模型和自我评估基线。

### 结论

质量信号可以从内部模型几何结构中提取，为无需外部监督的可扩展对齐提供了一条路径。

### 翻译

将大型语言模型与人类偏好对齐通常依赖于外部监督，这面临关键限制：人工标注稀缺且主观，奖励模型容易受到奖励攻击，而自我评估方法则受到提示敏感性和偏见的影响。在这项工作中，我们提出了stable rank，一种从模型表示中推导出的内在、无需标注的质量信号。stable rank通过计算总方差与主导方向方差的比率来衡量隐藏状态的有效维度，通过信息在表示维度上的分布来捕捉质量。实验表明，stable rank在RewardBench上达到84.04%的准确率，并通过Best-of-N采样比贪婪解码平均提高11.3个百分点的任务准确率。利用这一见解，我们引入了Stable Rank Group Relative Policy Optimization (SR-GRPO)，它使用stable rank作为强化学习的奖励信号。在没有外部监督的情况下，SR-GRPO使Qwen2.5-1.5B-Instruct在STEM领域提高10%，在数学推理方面提高19%，优于已学习的奖励模型和自我评估基线。我们的研究证明，质量信号可以从内部模型几何结构中提取，为无需外部监督的可扩展对齐提供了一条路径。


### 论文摘要

Aligning Large Language Models (LLMs) with human preferences typically relies on external supervision, which faces critical limitations: human annotations are scarce and subjective, reward models are vulnerable to reward hacking, and self-evaluation methods suffer from prompt sensitivity and biases. In this work, we propose stable rank, an intrinsic, annotation-free quality signal derived from model representations. Stable rank measures the effective dimensionality of hidden states by computing the ratio of total variance to dominant-direction variance, capturing quality through how information distributes across representation dimensions. Empirically, stable rank achieves 84.04% accuracy on RewardBench and improves task accuracy by an average of 11.3 percentage points over greedy decoding via Best-of-N sampling. Leveraging this insight, we introduce Stable Rank Group Relative Policy Optimization (SR-GRPO), which uses stable rank as a reward signal for reinforcement learning. Without external supervision, SR-GRPO improves Qwen2.5-1.5B-Instruct by 10% on STEM and 19% on mathematical reasoning, outperforming both learned reward models and self-evaluation baselines. Our findings demonstrate that quality signals can be extracted from internal model geometry, offering a path toward scalable alignment without external supervision.

---

## 183. Statistical hypothesis testing for differences between layers in dynamic multiplex networks

**论文链接:** [http://arxiv.org/abs/2512.03983v1](http://arxiv.org/abs/2512.03983v1)

**作者:** Maximilian Baum, Francesco Sanna Passino, Axel Gandy

**发布时间:** 2025-12-03

**备注:** 11 pages, 2 figures

### GPT解析

### 总结

本文提出了一种在动态多路网络中检验不同边类型层之间连接性差异的假设检验框架，基于潜在空间网络模型，通过谱嵌入方法构建检验统计量，并在模拟数据和生物数据集上验证了其有效性。

### 背景

随着动态多路网络的出现（对应于多种边类型随时间演化的图），一个关键的推断任务是确定与不同边类型相关的层在连接性上是否存在差异。

### 目的

引入一个假设检验框架，在潜在空间网络模型下，评估这些层是否共享共同的潜在表示。

### 方法

提出了一种基于图邻接矩阵展开表示的谱嵌入构建检验统计量的方法，扩展了先前关于随机图成对检验的文献，能够检验多路图中层之间的全局差异。

### 主要发现

该方法在图中节点数量趋于无穷大的渐近情况下能够有效检测层间差异，且通过在模拟数据和描述果蝇幼虫神经活动的生物数据集上的性能评估，证明了其良好的有限样本特性。

### 结论

该方法不仅可以用来检验层之间的差异，还可以轻松调整用于检验时间点之间的差异。

### 翻译

随着动态多路网络的出现，对应于多种边类型随时间演化的图，一个关键的推断任务是确定与不同边类型相关的层在连接性上是否存在差异。在这项工作中，我们引入了一个假设检验框架，在潜在空间网络模型下，评估这些层是否共享共同的潜在表示。我们提出的方法扩展了先前关于随机图成对检验问题的相关文献，并能够检验多路图中层之间的全局差异。虽然我们介绍该方法作为检验层间差异的工具，但它也可以轻松调整用于检验时间点之间的差异。我们基于图邻接矩阵展开表示的谱嵌入构建了一个检验统计量，并证明了它在图中节点数量趋于无穷大的渐近情况下检测层间差异的能力。通过在模拟数据和描述果蝇幼虫神经活动的生物数据集上评估其性能， empirically 证明了该检验的有限样本特性。


### 论文摘要

With the emergence of dynamic multiplex networks, corresponding to graphs where multiple types of edges evolve over time, a key inferential task is to determine whether the layers associated with different edge types differ in their connectivity. In this work, we introduce a hypothesis testing framework, under a latent space network model, for assessing whether the layers share a common latent representation. The method we propose extends previous literature related to the problem of pairwise testing for random graphs and enables global testing of differences between layers in multiplex graphs. While we introduce the method as a test for differences between layers, it can easily be adapted to test for differences between time points. We construct a test statistic based on a spectral embedding of an unfolded representation of the graph adjacency matrices and demonstrate its ability to detect differences across layers in the asymptotic regime where the number of nodes in each graph tends to infinity. The finite-sample properties of the test are empirically demonstrated by assessing its performance on both simulated data and a biological dataset describing the neural activity of larval Drosophila.

---

## 184. Quantum Topological Graph Neural Networks for Detecting Complex Fraud Patterns

**论文链接:** [http://arxiv.org/abs/2512.03696v1](http://arxiv.org/abs/2512.03696v1)

**作者:** Mohammad Doost, Mohammad Manthouri

**发布时间:** 2025-12-03

### GPT解析

### 总结

QTGNN框架是一种用于检测大规模金融网络中欺诈交易的新型方法，结合量子嵌入、变分图卷积和拓扑数据分析，能够捕捉复杂交易动态和结构异常。

### 背景

金融欺诈检测面临挑战，特别是在大规模金融网络中，需要能够捕捉复杂交易模式和结构异常的方法。

### 目的

开发一种新的QTGNN框架，有效检测金融交易中的欺诈行为，利用量子计算在处理复杂网络结构方面的优势。

### 方法

QTGNN方法包括：具有纠缠增强的量子数据嵌入、具有非线性动力学的变分量子图卷积、高阶拓扑不变量提取、具有自适应优化的混合量子-经典异常学习、通过拓扑归因实现可解释决策制定，以及针对NISQ硬件的电路简化和图采样优化。

### 主要发现

QTGNN在PaySim和Elliptic等金融数据集上表现优于经典和量子基线，使用ROC-AUC、精度和假阳性率等指标评估；消融研究验证了各组件的贡献；拓扑特征的稳定性提供了稳健的欺诈检测能力。

### 结论

QTGNN为金融欺诈检测提供了理论健全、可解释且实用的解决方案，桥接了量子机器学习、图论和拓扑分析。

### 翻译

我们提出了一种新型的QTGNN框架，用于检测大规模金融网络中的欺诈交易。通过集成量子嵌入、变分图卷积和拓扑数据分析，QTGNN能够捕捉复杂的交易动态和结构异常，这些异常指示着欺诈行为。该方法包括具有纠缠增强的量子数据嵌入、具有非线性动力学的变分量子图卷积、高阶拓扑不变量的提取、具有自适应优化的混合量子-经典异常学习，以及通过拓扑归因实现可解释的决策制定。严格的收敛保证确保在噪声中等规模量子(NISQ)设备上稳定训练，而拓扑特征的稳定性提供了稳健的欺诈检测。通过电路简化和图采样针对NISQ硬件进行优化，该框架能够扩展到大型交易网络。


### 论文摘要

We propose a novel QTGNN framework for detecting fraudulent transactions in large-scale financial networks. By integrating quantum embedding, variational graph convolutions, and topological data analysis, QTGNN captures complex transaction dynamics and structural anomalies indicative of fraud. The methodology includes quantum data embedding with entanglement enhancement, variational quantum graph convolutions with non-linear dynamics, extraction of higher-order topological invariants, hybrid quantum-classical anomaly learning with adaptive optimization, and interpretable decision-making via topological attribution. Rigorous convergence guarantees ensure stable training on noisy intermediate-scale quantum (NISQ) devices, while stability of topological signatures provides robust fraud detection. Optimized for NISQ hardware with circuit simplifications and graph sampling, the framework scales to large transaction networks. Simulations on financial datasets, such as PaySim and Elliptic, benchmark QTGNN against classical and quantum baselines, using metrics like ROC-AUC, precision, and false positive rate. An ablation study evaluates the contributions of quantum embeddings, topological features, non-linear channels, and hybrid learning. QTGNN offers a theoretically sound, interpretable, and practical solution for financial fraud detection, bridging quantum machine learning, graph theory, and topological analysis.

---

## 185. Comparative algorithm performance evaluation and prediction for the maximum clique problem using instance space analysis

**论文链接:** [http://arxiv.org/abs/2512.03419v1](http://arxiv.org/abs/2512.03419v1)

**作者:** Bharat Sharman, Elkafi Hassini

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究使用实例空间分析方法系统分析了最大团问题的实例空间，并评估了包括精确算法、启发式算法和基于图神经网络方法在内的最先进算法的性能。

### 背景

最大团问题是著名的基于图的组合优化问题，尽管已有多种算法方法，但对问题实例的系统分析仍然不足。

### 目的

系统分析最大团问题的实例空间，并评估和预测最先进算法的性能。

### 方法

使用来自TWITTER、COLLAB和IMDB-BINARY基准的图实例构建数据集；采用33个通用和2个问题特定的多项式时间可计算的基于图的特征；使用包含解决方案质量和算法运行时间的复合性能指标；应用实例空间分析方法。

### 主要发现

精确算法Mixed Order Maximum Clique (MOMC)在约74.7%的实例空间中表现最佳；Gurobi和CliSAT分别在13.8%和11%的实例空间中表现最佳；基于ISA的算法性能预测模型在34个测试实例上，排名第一和第二的最佳算法预测准确率分别为88%和97%。

### 结论

实例空间分析方法可以有效地预测和评估不同算法在最大团问题上的性能，有助于选择最适合特定实例的算法。

### 翻译

最大团问题是一种著名的基于图的组合优化问题，已通过多种算法方法得到解决，但对问题实例的系统分析仍然稀少。本研究采用实例空间分析方法系统地分析了该问题的实例空间，并评估和预测了最先进算法（包括精确算法、启发式算法和基于图神经网络的方法）的性能。使用来自图机器学习研究中常用的TWITTER、COLLAB和IMDB-BINARY基准的图实例构建了一个数据集。采用了33个通用和2个问题特定的多项式时间可计算的基于图的特征，包括几个谱特性。使用了结合了解决方案质量和算法运行时间的复合性能指标。比较分析表明，精确算法Mixed Order Maximum Clique (MOMC)在所构建数据集构成的约74.7%的实例空间中表现出优越性能。Gurobi和CliSAT分别约占13.8%和11%的实例空间。在BHOSLIB和DIMACS数据集中编译的34个具有挑战性的测试实例上运行的基于ISA的算法性能预测模型，排名第一和第二的最佳算法预测准确率分别为88%和97%。


### 论文摘要

The maximum clique problem, a well-known graph-based combinatorial optimization problem, has been addressed through various algorithmic approaches, though systematic analyses of the problem instances remain sparse. This study employs the instance space analysis (ISA) methodology to systematically analyze the instance space of this problem and assess & predict the performance of state-of-the-art (SOTA) algorithms, including exact, heuristic, and graph neural network (GNN)-based methods. A dataset was compiled using graph instances from TWITTER, COLLAB and IMDB-BINARY benchmarks commonly used in graph machine learning research. A set of 33 generic and 2 problem-specific polynomial-time-computable graph-based features, including several spectral properties, was employed for the ISA. A composite performance mea- sure incorporating both solution quality and algorithm runtime was utilized. The comparative analysis demonstrated that the exact algorithm Mixed Order Maximum Clique (MOMC) exhib- ited superior performance across approximately 74.7% of the instance space constituted by the compiled dataset. Gurobi & CliSAT accounted for superior performance in 13.8% and 11% of the instance space, respectively. The ISA-based algorithm performance prediction model run on 34 challenging test instances compiled from the BHOSLIB and DIMACS datasets yielded top-1 and top-2 best performing algorithm prediction accuracies of 88% and 97%, respectively.

---

## 186. VS-Graph: Scalable and Efficient Graph Classification Using Hyperdimensional Computing

**论文链接:** [http://arxiv.org/abs/2512.03394v1](http://arxiv.org/abs/2512.03394v1)

**作者:** Hamed Poursiami, Shay Snyder, Guojing Cong, Thomas Potok, Maryam Parsa

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出VS-Graph，一种向量符号图学习框架，结合了超维计算的高效性和图神经网络的表达能力，在不使用梯度优化的情况下实现了与GNNs相当的性能，同时大幅提升了训练速度。

### 背景

图分类是分子性质预测到材料设计等领域的基本任务。图神经网络通过消息传递学习强大表示但计算成本高，而超维计算是一种轻量级、受大脑启动的替代方案，但现有基于HDC的图方法通常难以匹配GNNs的性能。

### 目的

提出VS-Graph框架，缩小超维计算效率和消息传递表达能力之间的差距。

### 方法

VS-Graph引入了尖峰扩散机制用于拓扑驱动的节点识别，以及关联消息传递方案用于多跳邻域聚合，完全在高维向量空间内进行操作，不依赖梯度优化或反向传播。

### 主要发现

VS-Graph在MUTAG和DD等标准基准测试上比先前的HDC基线提高4-5%的性能；在多个数据集上匹配或超过GNN基线的性能，同时将训练速度提高最多450倍；即使在超维维度降至128的情况下仍保持高准确性，表现出在激进维度压缩下的鲁棒性。

### 结论

VS-Graph为在边缘和神经形态硬件上实现超高效执行铺平了道路。

### 翻译

图分类是从分子性质预测到材料设计等各个领域的基本任务。虽然图神经网络通过消息传递学习强大的表示而取得优异性能，但它们会产生高计算成本，限制了其在资源受限设备上的可扩展性和部署。超维计算，也称为向量符号架构，提供了一种轻量级、受大脑启发的替代方案，然而现有的基于HDC的图方法通常难以匹配GNNs的预测性能。在这项工作中，我们提出了VS-Graph，一种向量符号图学习框架，它缩小了HDC效率和消息传递表达能力之间的差距。VS-Graph引入了一种尖峰扩散机制用于拓扑驱动的节点识别，以及一种关联消息传递方案用于多跳邻域聚合，完全在高维向量空间内进行。在没有基于梯度优化或反向传播的情况下，我们的方法在与现代GNNs竞争的准确率上实现了竞争力，在MUTAG和DD等标准基准测试上比先前的HDC基线高出4-5%。它在多个数据集上匹配或超过了GNN基线的性能，同时将训练速度提高了最多450倍。此外，即使超维维度降低到D=128，VS-Graph仍保持高准确性，证明了其在激进维度压缩下的鲁棒性，为在边缘和神经形态硬件上实现超高效执行铺平了道路。


### 论文摘要

Graph classification is a fundamental task in domains ranging from molecular property prediction to materials design. While graph neural networks (GNNs) achieve strong performance by learning expressive representations via message passing, they incur high computational costs, limiting their scalability and deployment on resource-constrained devices. Hyperdimensional Computing (HDC), also known as Vector Symbolic Architectures (VSA), offers a lightweight, brain-inspired alternative, yet existing HDC-based graph methods typically struggle to match the predictive performance of GNNs. In this work, we propose VS-Graph, a vector-symbolic graph learning framework that narrows the gap between the efficiency of HDC and the expressive power of message passing. VS-Graph introduces a Spike Diffusion mechanism for topology-driven node identification and an Associative Message Passing scheme for multi-hop neighborhood aggregation entirely within the high-dimensional vector space. Without gradient-based optimization or backpropagation, our method achieves competitive accuracy with modern GNNs, outperforming the prior HDC baseline by 4-5% on standard benchmarks such as MUTAG and DD. It also matches or exceeds the performance of the GNN baselines on several datasets while accelerating the training by a factor of up to 450x. Furthermore, VS-Graph maintains high accuracy even with the hypervector dimensionality reduced to D=128, demonstrating robustness under aggressive dimension compression and paving the way for ultra-efficient execution on edge and neuromorphic hardware.

---

## 187. GRAND: Guidance, Rebalancing, and Assignment for Networked Dispatch in Multi-Agent Path Finding

**论文链接:** [http://arxiv.org/abs/2512.03194v1](http://arxiv.org/abs/2512.03194v1)

**作者:** Johannes Gaber, Meshal Alharbi, Daniele Gammelli, Gioele Zardini

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种混合方法，结合基于学习的全局指导和轻量级优化，用于解决终身多机器人取送任务调度问题，在大型机器人舰队中实现了更高的吞吐量和实时执行。

### 背景

大型机器人舰队在仓库和其他物流环境中已变得普遍，微小的控制增益会带来显著的运营影响。随着舰队规模扩大，任务调度问题变得复杂且关键。

### 目的

设计一种高效的任务调度方法，能够在保持实时执行的同时，提高大型机器人舰队的整体吞吐量，减少拥堵情况。

### 方法

采用图神经网络策略通过强化学习训练，输出自由代理在聚合仓库图上的期望分布；通过最小成本流将信号转换为区域间重新平衡；最后通过小型本地分配问题完成最终调度，同时确保每步计算延迟不超过1秒。

### 主要发现

在拥挤仓库基准测试中(最多500个代理)，该方法相比2024年最佳调度器能提高高达10%的吞吐量，同时保持实时执行能力。

### 结论

将图结构的学习指导与可解算器相结合可以有效减少拥堵，为大型机器人舰队的高吞吐量调度提供实用且可扩展的解决方案。

### 翻译

大型机器人舰队现在在仓库和其他物流环境中很常见，其中小的控制增益会转化为大的运营影响。在本文中，我们解决了终身多机器人取送任务的调度问题，并提出了一种混合方法，将基于学习的全局指导与轻量级优化相结合。通过强化学习训练的图神经网络策略输出自由代理在聚合仓库图上的期望分布。该信号通过最小成本流转换为区域间重新平衡，并通过小型本地分配问题最终确定，在保持准确性的同时将每步延迟控制在1秒计算预算内。在机器人联盟拥挤仓库基准测试中(最多500个代理)，我们的方法相比2024年获胜调度器将吞吐量提高了高达10%，同时保持实时执行。结果表明，将图结构的学习指导与可解算器耦合可以减少拥堵，并为大型舰队的高吞吐量调度提供实用、可扩展的蓝图。


### 论文摘要

Large robot fleets are now common in warehouses and other logistics settings, where small control gains translate into large operational impacts. In this article, we address task scheduling for lifelong Multi-Agent Pickup-and-Delivery (MAPD) and propose a hybrid method that couples learning-based global guidance with lightweight optimization. A graph neural network policy trained via reinforcement learning outputs a desired distribution of free agents over an aggregated warehouse graph. This signal is converted into region-to-region rebalancing through a minimum-cost flow, and finalized by small, local assignment problems, preserving accuracy while keeping per-step latency within a 1 s compute budget. On congested warehouse benchmarks from the League of Robot Runners (LRR) with up to 500 agents, our approach improves throughput by up to 10% over the 2024 winning scheduler while maintaining real-time execution. The results indicate that coupling graph-structured learned guidance with tractable solvers reduces congestion and yields a practical, scalable blueprint for high-throughput scheduling in large fleets.

---

## 188. BD-Index: Scalable Biharmonic Distance Queries on Large Graphs via Divide-and-Conquer Indexing

**论文链接:** [http://arxiv.org/abs/2512.02929v1](http://arxiv.org/abs/2512.02929v1)

**作者:** Yueyang Pan, Meihao Liao, Rong-Hua Li

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种名为BIndex的新型索引结构，用于高效计算图中的双调和距离，解决了现有随机游走方法在大型图上效率低下的问题。

### 背景

双调和距离是一种强大的图距离度量，在识别道路网络中的关键链路和缓解图神经网络中的过度挤压问题等方面有广泛应用。然而，计算双调和距离极其困难，特别是在大型图上。

### 目的

专注于解决单对双调和距离查询问题，提高在大型图上计算双调和距离的效率。

### 方法

作者提出了新的双调和距离公式，用两个独立随机游走来表示距离，并设计了BIndex索引结构，采用分治策略。图首先被分割成易于处理的部分，然后从下到上确定性计算所需的随机游走概率。查询时只需访问索引的一小部分。

### 主要发现

双调和距离可以解释为从两个节点开始的随机游走分布之间的距离；当图容易被分割成小块时，传统随机游走方法需要很长的游走长度；通过分割图和层次化处理，可以显著提高计算效率。

### 结论

BIndex索引结构能够在O(n·h)空间中构建，构建时间为O(n·h·(h+d_max))，每次查询时间为O(n·h)，其中h是层次分区树的高度，d_max是最大度数，两者通常远小于图的大小n。

### 翻译

双调和距离是一种强大的图距离度量，在识别道路网络中的关键链路和缓解图神经网络中的过度挤压问题等方面有许多应用。然而，计算双调和距离极其困难，特别是在大型图上。在本文中，我们专注于单对双调和距离查询问题。现有方法主要依赖于基于随机游走的方法，这些方法在某些图上表现良好，但当随机游走不能快速混合时效率低下。为解决这个问题，我们首先证明两个节点s,t之间的双调和距离，记为b(s,t)，可以解释为从s和t开始的两个随机游走分布之间的距离。当底层图容易被分割成更小的部分时，估计这些分布所需的随机游走长度很大。受此观察启发，我们提出了双调和距离的新公式，通过在由小切割集V_cut分隔的两个节点集V_s和V_t内的独立随机游走来表示b(s,t)，其中V_s∪V_t∪V_cut=V是图节点集。基于这一思想，我们提出了BIndex，一种遵循分治策略的新型索引结构。图首先被分割成易于处理的部分，然后所有需要的随机游走概率可以从下到上确定性计算。当查询到来时，只需访问索引的一小部分。我们证明BIndex需要O(n·h)空间，可以在O(n·h·(h+d_max))时间内构建，并在O(n·h)时间内回答每个查询，其中h是层次分区树的高度，d_max是最大度数，两者通常远小于n。


### 论文摘要

Biharmonic distance (\bd) is a powerful graph distance metric with many applications, including identifying critical links in road networks and mitigating over-squashing problem in \gnn. However, computing \bd\ is extremely difficult, especially on large graphs. In this paper, we focus on the problem of \emph{single-pair} \bd\ query. Existing methods mainly rely on random walk-based approaches, which work well on some graphs but become inefficient when the random walk cannot mix rapidly.To overcome this issue, we first show that the biharmonic distance between two nodes $s,t$, denoted by $b(s,t)$, can be interpreted as the distance between two random walk distributions starting from $s$ and $t$. To estimate these distributions, the required random walk length is large when the underlying graph can be easily cut into smaller pieces. Inspired by this observation, we present novel formulas of \bd to represent $b(s,t)$ by independent random walks within two node sets $\mathcal{V}_s$, $\mathcal{V}_t$ separated by a small \emph{cut set} $\mathcal{V}_{cut}$, where $\mathcal{V}_s\cup\mathcal{V}_t\cup\mathcal{V}_{cut}=\mathcal{V}$ is the set of graph nodes. Building upon this idea, we propose \bindex, a novel index structure which follows a divide-and-conquer strategy. The graph is first cut into pieces so that each part can be processed easily. Then, all the required random walk probabilities can be deterministically computed in a bottom-top manner. When a query comes, only a small part of the index needs to be accessed. We prove that \bindex\ requires $O(n\cdot h)$ space, can be built in $O(n\cdot h\cdot (h+d_{max}))$ time, and answers each query in $O(n\cdot h)$ time, where $h$ is the height of a hierarchy partition tree and $d_{max}$ is the maximum degree, which are both usually much smaller than $n$.

---

## 189. Learning Multimodal Embeddings for Traffic Accident Prediction and Causal Estimation

**论文链接:** [http://arxiv.org/abs/2512.02920v1](http://arxiv.org/abs/2512.02920v1)

**作者:** Ziniu Zhang, Minxuan Duan, Haris N. Koutsopoulos, Hongyang R. Zhang

**发布时间:** 2025-12-02

**备注:** 17 pages. To appear in KDD'26 Datasets

### GPT解析

### 总结

本研究通过结合道路网络数据和卫星图像分析交通事故模式，构建了大型多模态数据集，并评估了多模态学习方法的有效性。

### 背景

以往交通事故预测工作主要依赖道路网络结构特征，忽略了道路表面及其周围环境的物理和环境信息。

### 目的

构建结合道路网络数据和卫星图像的多模态数据集，评估多模态学习方法，并分析影响交通事故的关键因素。

### 方法

构建横跨美国六个州的大型多模态数据集，包含九百万交通事故记录和一百万高分辨率卫星图像；每个节点标注天气统计和道路类型特征，每条边标注交通量信息；使用多模态学习方法整合视觉和网络嵌入，并应用匹配估计器进行因果分析。

### 主要发现

整合两种数据模态提高了预测准确性，平均AUROC达到90.1%，比仅使用图结构的模型提高3.7%；调整混杂因素后，事故率在降水量增加时上升24%，在高速公路等高速道路上上升22%，由于季节性模式上升29%；消融研究证实卫星图像特征对准确预测至关重要。

### 结论

结合卫星图像和道路网络数据的多模态方法能有效提高交通事故预测准确性，卫星图像特征是实现准确预测的关键要素。

### 翻译

我们考虑使用道路网络数据和与道路图节点对齐的卫星图像来分析交通事故模式。以往预测交通事故发生的工作主要依赖道路网络结构特征，而忽略了道路表面及其周围环境的物理和环境信息。在本工作中，我们构建了一个横跨美国六个州的大型多模态数据集，包含来自官方来源的九百万交通事故记录，以及道路网络每个节点的一百万张高分辨率卫星图像。此外，每个节点都标注了区域天气统计和道路类型等特征，每条边都标注了交通量信息。利用此数据集，我们对整合视觉和网络嵌入的多模态学习方法进行了全面评估。研究结果表明，整合两种数据模态提高了预测准确性，平均AUROC达到90.1%，比仅使用图结构的图神经网络模型提高了3.7%。通过改进的嵌入方法，我们基于匹配估计器进行了因果分析，以估计影响交通事故的关键因素。在调整其他混杂因素后，我们发现事故率在降水量较高时上升24%，在高速公路等高速道路上上升22%，由于季节性模式上升29%。消融研究证实卫星图像特征对于实现准确预测是必要的。


### 论文摘要

We consider analyzing traffic accident patterns using both road network data and satellite images aligned to road graph nodes. Previous work for predicting accident occurrences relies primarily on road network structural features while overlooking physical and environmental information from the road surface and its surroundings. In this work, we construct a large multimodal dataset across six U.S. states, containing nine million traffic accident records from official sources, and one million high-resolution satellite images for each node of the road network. Additionally, every node is annotated with features such as the region's weather statistics and road type (e.g., residential vs. motorway), and each edge is annotated with traffic volume information (i.e., Average Annual Daily Traffic). Utilizing this dataset, we conduct a comprehensive evaluation of multimodal learning methods that integrate both visual and network embeddings. Our findings show that integrating both data modalities improves prediction accuracy, achieving an average AUROC of $90.1\%$, which is a $3.7\%$ gain over graph neural network models that only utilize graph structures. With the improved embeddings, we conduct a causal analysis based on a matching estimator to estimate the key contributing factors influencing traffic accidents. We find that accident rates rise by $24\%$ under higher precipitation, by $22\%$ on higher-speed roads such as motorways, and by $29\%$ due to seasonal patterns, after adjusting for other confounding factors. Ablation studies confirm that satellite imagery features are essential for achieving accurate prediction.

---

## 190. GraphMatch: Fusing Language and Graph Representations in a Dynamic Two-Sided Work Marketplace

**论文链接:** [http://arxiv.org/abs/2512.02849v1](http://arxiv.org/abs/2512.02849v1)

**作者:** Mikołaj Sacha, Hammad Jafri, Mattie Terzolo, Ayan Sinha, Andrew Rabinovich

**发布时间:** 2025-12-02

### GPT解析

### 总结

GraphMatch是一个新的大规模推荐框架，通过融合预训练语言模型和图神经网络，解决内容丰富、动态双边市场中的匹配推荐问题。该方法在保持高效运行的同时，在匹配任务上表现优于传统方法。

### 背景

内容丰富、动态的双边市场中的匹配推荐面临独特挑战，主要是由于内容和交互图的不断演变。传统方法通常使用独立模型，难以同时捕捉文本的细粒度语义和图的时序敏感结构。

### 目的

开发一个能够有效融合文本和图信息的大规模推荐框架，以应对动态环境中内容不断变化的挑战，提高匹配推荐的准确性和效率。

### 方法

GraphMatch框架结合了强大的文本编码器和图神经网络，采用对抗性负采样和时序子图训练策略，学习能够同时捕捉文本语义和图结构的表示。该框架在大规模数据上进行评估，并针对实时应用进行了低延迟推理优化。

### 主要发现

在Upwork平台的交互数据上进行的实验表明，GraphMatch在匹配任务上优于仅使用语言或仅使用图的基线方法，同时保持了较高的运行效率。这证明了统一语言和图表示的有效性。

### 结论

统一语言和图表示是解决内容丰富、动态双边推荐问题的有效方法，GraphMatch框架弥合了强大预训练语言模型和大规模图之间的实际应用差距，为动态环境中的推荐系统提供了新的解决方案。

### 翻译

在内容丰富、动态的双边市场中推荐匹配面临独特挑战，这是由于内容和交互图的不断演变。我们引入了GraphMatch，这是一个新的推荐框架，它融合了预训练语言模型和图神经网络来克服这些挑战。与之前以独立模型为中心的方法不同，GraphMatch是一个建立在强大文本编码器和图神经网络协同工作基础上的综合方案。它采用对抗性负采样和时序子图训练来学习能够捕捉演变文本的细粒度语义和图的时序敏感结构的表示。我们在Upwork（一个领先的劳动力市场）的交互数据上进行了大规模评估，并讨论了适合实时使用的低延迟推理方法。在我们的实验中，GraphMatch在匹配任务上优于仅使用语言和仅使用图的基线方法，同时运行效率高。这些结果表明，统一语言和图表示是解决内容丰富、动态双边推荐问题的有效方法，在实践中弥合了强大预训练语言模型和大规模图之间的差距。


### 论文摘要

Recommending matches in a text-rich, dynamic two-sided marketplace presents unique challenges due to evolving content and interaction graphs. We introduce GraphMatch, a new large-scale recommendation framework that fuses pre-trained language models with graph neural networks to overcome these challenges. Unlike prior approaches centered on standalone models, GraphMatch is a comprehensive recipe built on powerful text encoders and GNNs working in tandem. It employs adversarial negative sampling alongside point-in-time subgraph training to learn representations that capture both the fine-grained semantics of evolving text and the time-sensitive structure of the graph. We evaluated extensively on interaction data from Upwork, a leading labor marketplace, at large scale, and discuss our approach towards low-latency inference suitable for real-time use. In our experiments, GraphMatch outperforms language-only and graph-only baselines on matching tasks while being efficient at runtime. These results demonstrate that unifying language and graph representations yields a highly effective solution to text-rich, dynamic two-sided recommendations, bridging the gap between powerful pretrained LMs and large-scale graphs in practice.

---

## 191. Credal Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.02722v1](http://arxiv.org/abs/2512.02722v1)

**作者:** Matteo Tolloso, Davide Bacciu

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文引入了第一种置信集图神经网络（CGNNs），将置信学习扩展到图域，通过训练GNN输出以置信集形式表示的集合值预测，实现了更可靠的认识不确定性表示。

### 背景

不确定性量化对于部署可靠的图神经网络至关重要，现有方法主要依赖贝叶斯推理或集成方法。

### 目的

开发一种新的不确定性量化方法，能够捕捉图神经网络中的认识不确定性，特别是在分布外条件下。

### 方法

提出了一种互补的置信学习方法，利用层间信息传播的不同方面，适应GNN中消息传递的独特性质，训练GNN输出集合值预测。

### 主要发现

图同质性假设在塑造不确定性估计有效性方面扮演关键角色，CGNNs在异配图上的分布偏移情况下表现优异。

### 结论

大量实验证明，CGNNs能够提供更可靠的认识不确定性表示，并在分布偏移情况下实现最先进的性能。

### 翻译

不确定性量化对于部署可靠的图神经网络至关重要，现有方法主要依赖贝叶斯推理或集成方法。在本文中，我们引入了第一种置信集图神经网络（CGNNs），通过将置信学习扩展到图域，训练GNN输出以置信集形式表示的集合值预测。为了适应GNN中消息传递的独特性质，我们开发了一种互补的置信学习方法，利用层间信息传播的不同方面。我们在分布外条件下的节点分类任务中评估了我们的方法。我们的分析强调了图同质性假设在塑造不确定性估计有效性方面的关键作用。大量实验表明，CGNNs能够提供更可靠的认识不确定性表示，并在异配图上的分布偏移情况下实现最先进的性能。


### 论文摘要

Uncertainty quantification is essential for deploying reliable Graph Neural Networks (GNNs), where existing approaches primarily rely on Bayesian inference or ensembles. In this paper, we introduce the first credal graph neural networks (CGNNs), which extend credal learning to the graph domain by training GNNs to output set-valued predictions in the form of credal sets. To account for the distinctive nature of message passing in GNNs, we develop a complementary approach to credal learning that leverages different aspects of layer-wise information propagation. We assess our approach on uncertainty quantification in node classification under out-of-distribution conditions. Our analysis highlights the critical role of the graph homophily assumption in shaping the effectiveness of uncertainty estimates. Extensive experiments demonstrate that CGNNs deliver more reliable representations of epistemic uncertainty and achieve state-of-the-art performance under distributional shift on heterophilic graphs.

---

## 192. Zero-Shot Instruction Following in RL via Structured LTL Representations

**论文链接:** [http://arxiv.org/abs/2512.02633v1](http://arxiv.org/abs/2512.02633v1)

**作者:** Mattia Giuri, Mathias Jackermeier, Alessandro Abate

**发布时间:** 2025-12-02

**备注:** ICML 2025 Workshop on Programmatic Representations for Agent Learning

### GPT解析

### 总结

该论文提出了一种新的方法，用于学习遵循任意线性时序逻辑(LTL)指令的多任务策略，解决了现有方法在多个高级事件同时可能为真且复杂交互情况下的不足。

### 背景

线性时序逻辑(LTL)是指定强化学习(RL)智能体复杂结构化任务的有力框架。最近的工作表明，将LTL指令解释为有限自动机(可视为监控任务进度的高级程序)，使得能够学习一个通用的单一策略，能够在测试时执行任意指令。然而，现有方法在多个高级事件(即原子命题)可能同时为真且可能复杂交互的环境中表现不佳。

### 目的

提出一种新的学习方法，用于学习遵循任意LTL指令的多任务策略，解决现有方法的不足。

### 方法

提出一种新方法，将策略基于简单布尔公式序列进行条件化，这些公式直接与自动机中的转换对齐，并通过图神经网络(GNN)编码，以产生结构化的任务表示。

### 主要发现

在基于国际象棋的复杂环境中的实验证明了该方法的优势。

### 结论

通过在策略条件化中使用简单布尔公式序列并通过GNN编码，解决了多高级事件同时为真且复杂交互情况下的LTL指令跟随问题。

### 翻译

线性时序逻辑(LTL)是指定强化学习(RL)智能体复杂结构化任务的有力框架。最近的工作表明，将LTL指令解释为有限自动机(可以看作是监控任务进度的高级程序)，使得能够学习一个通用的单一策略，能够在测试时执行任意指令。然而，现有方法在多个高级事件(即原子命题)可能同时为真且可能复杂交互的环境中表现不佳。在这项工作中，我们提出了一种新的学习方法，用于学习遵循任意LTL指令的多任务策略，以解决这一不足。我们的方法将策略基于简单布尔公式序列进行条件化，这些公式直接与自动机中的转换对齐，并通过图神经网络(GNN)编码，以产生结构化的任务表示。在基于国际象棋的复杂环境中的实验证明了我们方法的优势。


### 论文摘要

Linear temporal logic (LTL) is a compelling framework for specifying complex, structured tasks for reinforcement learning (RL) agents. Recent work has shown that interpreting LTL instructions as finite automata, which can be seen as high-level programs monitoring task progress, enables learning a single generalist policy capable of executing arbitrary instructions at test time. However, existing approaches fall short in environments where multiple high-level events (i.e., atomic propositions) can be true at the same time and potentially interact in complicated ways. In this work, we propose a novel approach to learning a multi-task policy for following arbitrary LTL instructions that addresses this shortcoming. Our method conditions the policy on sequences of simple Boolean formulae, which directly align with transitions in the automaton, and are encoded via a graph neural network (GNN) to yield structured task representations. Experiments in a complex chess-based environment demonstrate the advantages of our approach.

---

## 193. Temporal Graph Neural Networks for Early Anomaly Detection and Performance Prediction via PV System Monitoring Data

**论文链接:** [http://arxiv.org/abs/2512.03114v1](http://arxiv.org/abs/2512.03114v1)

**作者:** Srijani Mukherjee, Laurent Vuillon, Liliane Bou Nassif, Stéphanie Giroux-Julien, Hervé Pabiou, Denys Dutykh, Ionnasis Tsanakas

**发布时间:** 2025-12-02

### GPT解析

### 总结

本研究提出了一种利用时间图神经网络进行太阳能光伏系统性能监控和异常检测的新方法。

### 背景

太阳能光伏系统的快速增长需要先进的方法来确保最佳运行性能监控和异常检测。

### 目的

提出一种利用时间图神经网络预测太阳能光伏输出功率并使用环境和运营参数检测异常的新方法。

### 方法

提出的模型利用关键光伏系统参数（包括辐照度、模块温度和环境温度）之间的基于时间图的关系来预测电功率输出。

### 主要发现

摘要中未明确提及主要发现。

### 结论

摘要中未明确提及结论。

### 翻译

太阳能光伏(PV)系统的快速增长需要先进的方法进行性能监控和异常检测以确保最佳运行。在本研究中，我们提出了一种新方法，利用时间图神经网络(Temporal GNN)来预测太阳能光伏输出功率并使用环境和运营参数检测异常。所提出的模型利用关键光伏系统参数（包括辐照度、模块温度和环境温度）之间的基于时间图的关系来预测电功率输出。本研究基于在法国里昂屋顶户外设施收集的数据，包括光伏模块的功率测量和气象参数。


### 论文摘要

The rapid growth of solar photovoltaic (PV) systems necessitates advanced methods for performance monitoring and anomaly detection to ensure optimal operation. In this study, we propose a novel approach leveraging Temporal Graph Neural Network (Temporal GNN) to predict solar PV output power and detect anomalies using environmental and operational parameters. The proposed model utilizes graph-based temporal relationships among key PV system parameters, including irradiance, module and ambient temperature to predict electrical power output. This study is based on data collected from an outdoor facility located on a rooftop in Lyon (France) including power measurements from a PV module and meteorological parameters.

---

## 194. Detection of Crowdsourcing Cryptocurrency Laundering via Multi-Task Collaboration

**论文链接:** [http://arxiv.org/abs/2512.02534v1](http://arxiv.org/abs/2512.02534v1)

**作者:** Guang Li, Litong Sun, Jieying Zhou, Weigang Wu

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种基于多任务协作的众包洗钱检测框架(MCCLD)，用于解决USDT稳定币新型洗钱方式的检测问题。该框架利用交易组作为辅助信息，通过端到端图神经网络实现洗钱交易检测和交易组检测的协同工作，有效提高了对多样化众包洗钱模式的检测性能。

### 背景

USDT作为一种与美元挂钩的稳定币，因其稳定性、匿名性和易用性而成为洗钱活动的首选工具。近年来，一种名为'众包洗钱'的新型洗钱方式迅速兴起，它通过招募大量普通个人来分散资金，已成为一个重大威胁。由于精细的分工，这类洗钱交易表现出多样化的模式和中心化结构，给传统检测方法带来重大挑战。

### 目的

开发一种有效的检测方法，能够识别和应对USDT稳定币中的众包洗钱活动，解决其交易模式多样化和结构中心化带来的检测难题。

### 方法

本文引入交易组作为辅助信息，提出了多任务协作众包洗钱检测(MCCLD)框架。该框架采用端到端图神经网络，实现洗钱交易检测和交易组检测任务之间的协作。两个任务通过共享分类器进行联合优化，共享特征编码器融合多级特征嵌入，提供丰富的交易语义和潜在的组信息。

### 主要发现

MCCLD框架在众包洗钱和一般洗钱检测任务上都表现出色，证明了其有效性和良好的泛化能力。通过多任务协作和特征共享，框架能够更准确地识别多样化的众包洗钱模式，克服了传统方法面临的挑战。

### 结论

MCCLD框架为USDT稳定币中的众包洗钱检测提供了一种有效解决方案。通过整合交易组信息和多任务学习策略，该框架能够显著提高检测性能，为金融监管和反洗钱工作提供了新的技术手段。

### 翻译

USDT是一种与美元挂钩的稳定币，由于其稳定性、匿名性和易用性，已成为洗钱的首选选择。值得注意的是，稳定币上出现了一种新型的洗钱形式——我们称之为'众包洗钱'，它通过招募大量普通个人分散资金，并迅速成为一个重大威胁。然而，由于精细的分工，众包洗钱交易表现出多样化的模式和中心化结构，给检测带来重大挑战。在本文中，我们引入交易组作为辅助信息，提出了多任务协作众包洗钱检测(MCCLD)框架。MCCLD采用端到端图神经网络，实现洗钱交易检测和交易组检测任务之间的协作，提高对众包洗钱组内多样化模式的检测性能。这两个任务通过共享分类器进行联合优化，共享特征编码器融合多级特征嵌入，提供丰富的交易语义和潜在的组信息。在众包和一般洗钱上的大量实验证明了MCCLD的有效性和泛化能力。据我们所知，这是第一篇关于众包洗钱检测的研究。


### 论文摘要

USDT, a stablecoin pegged to dollar, has become a preferred choice for money laundering due to its stability, anonymity, and ease of use. Notably, a new form of money laundering on stablecoins -- we refer to as crowdsourcing laundering -- disperses funds through recruiting a large number of ordinary individuals, and has rapidly emerged as a significant threat. However, due to the refined division of labor, crowdsourcing laundering transactions exhibit diverse patterns and a polycentric structure, posing significant challenges for detection. In this paper, we introduce transaction group as auxiliary information, and propose the Multi-Task Collaborative Crowdsourcing Laundering Detection (MCCLD) framework. MCCLD employs an end-to-end graph neural network to realize collaboration between laundering transaction detection and transaction group detection tasks, enhancing detection performance on diverse patterns within crowdsourcing laundering group. These two tasks are jointly optimized through a shared classifier, with a shared feature encoder that fuses multi-level feature embeddings to provide rich transaction semantics and potential group information. Extensive experiments on both crowdsourcing and general laundering demonstrate MCCLD's effectiveness and generalization. To the best of our knowledge, this is the first work on crowdsourcing laundering detection.

---

## 195. Cross-View Topology-Aware Graph Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.02130v1](http://arxiv.org/abs/2512.02130v1)

**作者:** Ahmet Sami Korkmaz, Selim Coskunuzer, Md Joshem Uddin

**发布时间:** 2025-12-01

### GPT解析

### 总结

本文提出了GraphTCL，一种双视图对比学习框架，用于图分类任务。该方法结合了图神经网络的结构嵌入和持续同调导出的拓扑嵌入，通过跨视图对比损失对齐这些互补视图，从而增强表示质量并提高分类性能。

### 背景

图分类在化学、社交网络和生物信息学等领域受到广泛关注。虽然图神经网络能有效捕获局部结构模式，但它们常常忽略对鲁棒表示学习至关重要的全局拓扑特征。

### 目的

提出一种能够同时考虑局部结构特征和全局拓扑特征的图分类方法，以增强表示质量并提高分类性能。

### 方法

GraphTCL是一种双视图对比学习框架，它集成了来自图神经网络的结构嵌入和来自持续同调的拓扑嵌入，通过跨视图对比损失对齐这些互补视图。

### 主要发现

在包括TU和OGB分子图在内的基准数据集上的大量实验表明，GraphTCL始终优于最先进的基线方法。

### 结论

这项研究强调了拓扑感知对比学习对于推进图表示方法的重要性。

### 翻译

图分类由于其化学、社交网络和生物信息学中的应用而受到广泛关注。虽然图神经网络能有效捕获局部结构模式，但它们常常忽略对鲁棒表示学习至关重要的全局拓扑特征。在这项工作中，我们提出了GraphTCL，一种双视图对比学习框架，它集成了来自图神经网络的结构嵌入和来自持续同调导出的拓扑嵌入。通过跨视图对比损失对齐这些互补视图，我们的方法增强了表示质量并提高了分类性能。在包括TU和OGB分子图在内的基准数据集上的大量实验表明，GraphTCL始终优于最先进的基线方法。这项研究强调了拓扑感知对比学习对于推进图表示方法的重要性。


### 论文摘要

Graph classification has gained significant attention due to its applications in chemistry, social networks, and bioinformatics. While Graph Neural Networks (GNNs) effectively capture local structural patterns, they often overlook global topological features that are critical for robust representation learning. In this work, we propose GraphTCL, a dual-view contrastive learning framework that integrates structural embeddings from GNNs with topological embeddings derived from persistent homology. By aligning these complementary views through a cross-view contrastive loss, our method enhances representation quality and improves classification performance. Extensive experiments on benchmark datasets, including TU and OGB molecular graphs, demonstrate that GraphTCL consistently outperforms state-of-the-art baselines. This study highlights the importance of topology-aware contrastive learning for advancing graph representation methods.

---

## 196. QGShap: Quantum Acceleration for Faithful GNN Explanations

**论文链接:** [http://arxiv.org/abs/2512.03099v1](http://arxiv.org/abs/2512.03099v1)

**作者:** Haribandhu Jena, Jyotirmaya Shivottam, Subhankar Mishra

**发布时间:** 2025-12-01

**备注:** Accepted in the QC+AI Workshop at AAAI 2026

### GPT解析

### 总结

本文介绍了QGShap，一种基于量子计算的图神经网络解释方法，通过振幅放大技术实现联盟评估的二次加速，同时保持精确的Shapley计算。

### 背景

图神经网络在药物发现、社交网络分析和推荐系统等关键领域变得不可或缺，但其黑盒性质阻碍了在需要透明度和责任追究场景中的部署。基于Shapley值的方法虽然能提供数学上有原则的解释，但计算精确值需要评估2^n个联盟，对于现实世界的图来说计算不可行。

### 目的

开发一种既能保持精确Shapley计算又能提高效率的解释方法，解决现有近似策略要么保真度低要么效率低的问题。

### 方法

引入QGShap，一种利用振幅放大实现联盟评估二次加速的量子计算方法。与经典采样或代理方法不同，该方法在可处理的图大小下提供完全忠实的解释，无需近似权衡。

### 主要发现

在合成图数据集上的经验评估表明，QGShap实现了持续的高保真度和解释准确性，在所有评估指标上匹配或超过经典方法。QGShap保留了精确的Shapley保真度，并提供可解释、稳定且与GNN底层图推理一致的结构一致解释。

### 结论

QGShap不仅保留了精确的Shapley保真度，还提供了与GNN底层图推理一致的解释，解决了现有方法在效率和保真度之间的权衡问题。

### 翻译

图神经网络(GNNs)已成为药物发现、社交网络分析和推荐系统等关键领域不可或缺的工具，然而它们的黑盒性质阻碍了在需要透明度和责任追究的场景中的部署。虽然基于Shapley值的方法通过量化每个组件对预测的贡献提供了数学上有原则的解释，但计算精确值需要评估2^n个联盟（或聚合n!个排列），这对于现实世界的图来说是不可行的。现有近似策略要么牺牲保真度要么牺牲效率，限制了它们的实际应用价值。我们引入了QGShap，一种利用振幅放大实现联盟评估二次加速的量子计算方法，同时保持精确的Shapley计算。与经典采样或代理方法不同，我们的方法在可处理的图大小下提供完全忠实的解释，无需在近似之间进行权衡。我们在合成图数据集上进行了经验评估，证明QGShap实现了持续的高保真度和解释准确性，在所有评估指标上匹配或超过经典方法的性能。这些结果共同表明，QGShap不仅保留了精确的Shapley保真度，还提供了可解释、稳定且与GNN底层图推理一致的结构一致解释。QGShap的实现可在https://github.com/smlab-niser/qgshap获取。


### 论文摘要

Graph Neural Networks (GNNs) have become indispensable in critical domains such as drug discovery, social network analysis, and recommendation systems, yet their black-box nature hinders deployment in scenarios requiring transparency and accountability. While Shapley value-based methods offer mathematically principled explanations by quantifying each component's contribution to predictions, computing exact values requires evaluating $2^n$ coalitions (or aggregating over $n!$ permutations), which is intractable for real-world graphs. Existing approximation strategies sacrifice either fidelity or efficiency, limiting their practical utility. We introduce QGShap, a quantum computing approach that leverages amplitude amplification to achieve quadratic speedups in coalition evaluation while maintaining exact Shapley computation. Unlike classical sampling or surrogate methods, our approach provides fully faithful explanations without approximation trade-offs for tractable graph sizes. We conduct empirical evaluations on synthetic graph datasets, demonstrating that QGShap achieves consistently high fidelity and explanation accuracy, matching or exceeding the performance of classical methods across all evaluation metrics. These results collectively demonstrate that QGShap not only preserves exact Shapley faithfulness but also delivers interpretable, stable, and structurally consistent explanations that align with the underlying graph reasoning of GNNs. The implementation of QGShap is available at https://github.com/smlab-niser/qgshap.

---

## 197. Morphling: Fast, Fused, and Flexible GNN Training at Scale

**论文链接:** [http://arxiv.org/abs/2512.01678v2](http://arxiv.org/abs/2512.01678v2)

**作者:** Anubhab, Rupesh Nasre

**发布时间:** 2025-12-01

### GPT解析

### 总结

Morphling是一个领域特定的代码合成器，旨在解决图神经网络(GNNs)的硬件挑战，通过编译高级GNN规范为针对不同执行环境的优化实现，显著提高了训练吞吐量并减少了内存消耗。

### 背景

图神经网络(GNNs)将不规则、内存绑定的图遍历与规则、计算密集的密集矩阵操作融合，形成基本硬件挑战。现有框架如PyG和DGL优先考虑高级可用性，但未能解决这些不同的执行特性，导致通用内核存在缓存局部性差、内存移动过多和大量中间分配的问题。

### 目的

开发Morphling，一个领域特定的代码合成器，弥合GNN高级规范与底层硬件执行之间的差距，提高GNN训练性能。

### 方法

Morphling通过将高级GNN规范编译为针对OpenMP、CUDA和MPI的便携式、后端专门化实现。它利用为每个执行环境定制的优化、架构感知原语库，并包含一个运行时稀疏感知执行引擎，动态选择密集或稀疏执行路径以减少不必要计算。

### 主要发现

在11个真实世界数据集上的评估显示，Morphling将每个epoch训练吞吐量平均提高了20倍(CPU)和19倍(GPU)，峰值加速比达66倍。内存效率布局减少了最多15倍的峰值内存消耗，使大规模GNN训练可在商品硬件上进行。

### 结论

专门的、架构感知的代码合成是跨不同并行和分布式平台实现高性能GNN执行的有效且可扩展的途径。

### 翻译

图神经网络(GNNs)通过将不规则、内存绑定的图遍历与规则、计算密集的密集矩阵操作融合，提出了基本的硬件挑战。虽然PyTorch Geometric (PyG)和Deep Graph Library (DGL)等框架优先考虑高级可用性，但它们未能解决这些不同的执行特性。因此，它们依赖于通用内核，这些内核存在缓存局部性差、内存移动过多和大量中间分配的问题。为了解决这些限制，我们提出了Morphling，一个领域特定的代码合成器，旨在弥合这一差距。Morphling通过实例化为每个执行环境定制的优化、架构感知原语库，将高级GNN规范编译为针对OpenMP、CUDA和MPI的便携式、后端专门化实现。Morphling还包含一个运行时稀疏感知执行引擎，它使用输入特征统计动态选择密集或稀疏执行路径，减少对零值条目不必要计算。我们在11个涵盖不同图结构、特征维度和稀疏性领域的真实世界数据集上评估了Morphling。结果表明，与PyG和DGL相比，Morphling在CPU上的每个epoch训练吞吐量平均提高了20倍，在GPU上提高了19倍，峰值加速比达到66倍。Morphling的内存高效布局进一步将峰值内存消耗减少了最多15倍，使得能够在商品硬件上进行大规模GNN训练。这些发现表明，专门的、架构感知的代码合成是跨不同并行和分布式平台实现高性能GNN执行的有效且可扩展的途径。


### 论文摘要

Graph Neural Networks (GNNs) present a fundamental hardware challenge by fusing irregular, memory-bound graph traversals with regular, compute-intensive dense matrix operations. While frameworks such as PyTorch Geometric (PyG) and Deep Graph Library (DGL) prioritize high-level usability, they fail to address these divergent execution characteristics. As a result, they rely on generic kernels that suffer from poor cache locality, excessive memory movement, and substantial intermediate allocations. To address these limitations, we present Morphling, a domain-specific code synthesizer designed to bridge this gap. Morphling compiles high-level GNN specifications into portable, backend-specialized implementations targeting OpenMP, CUDA, and MPI. It achieves this by instantiating a library of optimized, architecture-aware primitives tailored to each execution environment. Morphling also incorporates a runtime sparsity-aware execution engine that dynamically selects dense or sparse execution paths using input feature statistics, reducing unnecessary computation on zero-valued entries. We evaluate Morphling on eleven real-world datasets spanning diverse graph structures, feature dimensionalities, and sparsity regimes. The results show that Morphling improves per-epoch training throughput by an average of 20X on CPUs and 19X on GPUs over PyG and DGL, with peak speedups reaching 66X. Morphling's memory-efficient layouts further reduce peak memory consumption by up to 15X, enabling large-scale GNN training on commodity hardware. These findings demonstrate that specialized, architecture-aware code synthesis provides an effective and scalable path toward high-performance GNN execution across diverse parallel and distributed platforms.

---

## 198. Approximate Bayesian Inference on Mechanisms of Network Growth and Evolution

**论文链接:** [http://arxiv.org/abs/2512.03092v1](http://arxiv.org/abs/2512.03092v1)

**作者:** Maxwell H Wang, Till Hoffmann, Jukka-Pekka Onnela

**发布时间:** 2025-11-30

**备注:** 24 pages, 8 figures

### GPT解析

### 总结

该论文提出了一种结合图神经网络的条件密度估计器方法，用于推断网络形成机制的相对贡献，通过将机制分配给每条边形成事件，实现了对网络生成过程和动态演化的解释。

### 背景

机制模型可通过指定生成规则提供对网络增长的直观解释。现实世界网络形成通常涉及多种机制同时作用，理解各机制的相对贡献具有重要意义。

### 目的

开发一种能够灵活推断多种网络形成机制相对贡献的方法，以解释网络生成过程和动态演化。

### 方法

提出一种基于条件密度估计器和图神经网络的混合机制模型，该模型将机制分配给每条边形成事件，而非节点级机制，并采用近似贝叶斯方法进行推断。

### 主要发现

近似贝叶斯方法可有效推断模型中各机制的相对权重，该方法成功应用于调查多种现实世界网络形成背后的机制。

### 结论

所提出的基于事件的机制混合模型能够有效解释网络生成过程和动态演化，为理解复杂网络形成提供了新的分析框架。

### 翻译

机制模型可以通过指定一组生成规则，为网络增长提供直观且可解释的解释。这些规则可以根据关于现实世界网络增长机制的领域知识来定义，也可以设计为促进特定网络模体的出现。在现实世界网络的形成中，可能同时涉及多种机制；因此，理解每种机制的相对贡献非常重要。在本文中，我们提出使用条件密度估计器（结合图神经网络）对灵活的网络形成机制混合体进行推断。这种基于事件的机制混合模型将机制分配给每条边形成事件，而不是规定节点级机制，从而能够解释网络生成过程以及网络随时间的动态演化。我们证明了我们的近似贝叶斯方法对模型中机制的相对权重推断有效，并利用该方法调查了多种现实世界网络形成背后的机制。


### 论文摘要

Mechanistic models can provide an intuitive and interpretable explanation of network growth by specifying a set of generative rules. These rules can be defined by domain knowledge about real-world mechanisms governing network growth or may be designed to facilitate the appearance of certain network motifs. In the formation of real-world networks, multiple mechanisms may be simultaneously involved; it is then important to understand the relative contribution of each of these mechanisms. In this paper, we propose the use of a conditional density estimator, augmented with a graph neural network, to perform inference on a flexible mixture of network-forming mechanisms. This event-wise mixture-of-mechanisms model assigns mechanisms to each edge formation event rather than stipulating node-level mechanisms, thus allowing for an explanation of the network generation process, as well as the dynamic evolution of the network over time. We demonstrate that our approximate Bayesian approach yields valid inferences for the relative weights of the mechanisms in our model, and we utilize this method to investigate the mechanisms behind the formation of a variety of real-world networks.

---

## 199. On the Problem of Consistent Anomalies in Zero-Shot Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2512.02520v1](http://arxiv.org/abs/2512.02520v1)

**作者:** Tai Le-Gia

**发布时间:** 2025-12-02

**备注:** PhD Dissertation

### GPT解析

### 总结

该研究解决了零样本异常分类和分割(AC/AS)的核心挑战，提出了基于理论和算法设计的解决方案，包括一致性异常问题分析、CoDeGraph框架开发、3D医学影像扩展以及批处理与文本方法融合。

### 背景

零样本异常分类和分割(AC/AS)能够在没有训练数据的情况下检测异常样本和区域，这在工业检测和医学影像领域变得越来越重要。

### 目的

调查零样本AC/AS的核心挑战，并提出基于理论和算法设计的解决方案。

### 方法

1) 分析预训练视觉变换器中补丁表示的统计和几何行为；2) 开发CoDeGraph图框架，通过多阶段图构建、社区检测和结构化细化过滤一致性异常；3) 为3D医学影像提出无训练、计算高效的体素标记化策略；4) 使用CoDeGraph派生的伪掩模监督提示驱动的视觉-语言模型。

### 主要发现

1) 一致性异常是一种系统性地影响基于距离方法的失败模式；2) 确定了两个关键现象：相似度缩放和邻居耗尽，描述了在高度相似对象环境中正常补丁关系的变化；3) 体素异常分割可以在没有任何3D训练样本的情况下实现；4) CoDeGraph派生的伪掩模可以连接批处理和文本零样本方法。

### 结论

该研究为零样本AC/AS问题提供了理论理解和实际解决方案，有效解决了工业检测和医学影像中的异常检测挑战。

### 翻译

零样本异常分类和分割(AC/AS)旨在没有任何训练数据的情况下检测异常样本和区域，这一能力在工业检测和医学影像中变得越来越重要。本文旨在研究零样本AC/AS的核心挑战，并提出基于理论和算法设计的解决方案。我们首先正式化了一致性异常的问题，这是一种失败模式，其中重复出现的相似异常会系统性地影响基于距离的方法。通过分析预训练视觉变换器中补丁表示的统计和几何行为，我们确定了两个关键现象——相似度缩放和邻居耗尽——这些现象描述了在高度相似对象的环境中，正常补丁之间的关系如何随着和没有一致性异常而变化。然后，我们引入了CoDeGraph，一个基于图的一致性异常过滤框架，建立在相似度缩放和邻居耗尽现象的基础上。通过多阶段图构建、社区检测和结构化细化，CoDeGraph有效抑制了一致性异常的影响。接下来，我们通过提出一种针对MRI数据的无训练、计算高效的体素标记化策略，将此框架扩展到3D医学影像。这使得真正的零样本3D异常检测管道成为可能，并表明体素异常分割可以在没有任何3D训练样本的情况下实现。最后，我们通过证明CoDeGraph派生的伪掩模可以监督提示驱动的视觉-语言模型，弥合了基于批处理和基于文本的零样本方法之间的差距。总之，本文为零样本AC/AS问题提供了理论理解和实际解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决零样本异常检测中的'一致异常'问题，即相似的异常模式在多个图像中重复出现的情况。这个问题在现实中很重要，因为在工业检测中，系统性的生产错误会导致相似的缺陷在多个产品上出现；在医学影像中，相同的病理特征可能在不同患者的扫描中重复出现。现有的零样本异常检测方法无法有效处理这类一致异常，会导致误判（将异常分类为正常），影响检测的准确性和可靠性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过观察两个关键现象设计了这个方法：1）相似度缩放现象（正常元素的互相似度向量遵循幂律衰减模式）；2）邻居耗尽现象（一致异常的互相似度向量会在某个点出现突然的增长）。作者借鉴了现有的批量零样本异常检测方法（如MuSc），但针对其无法处理一致异常的局限性进行了改进。同时，也借鉴了图神经网络和社区检测的思想，用于处理和过滤一致异常，设计了CoDeGraph这个三阶段框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'CoDeGraph方法的核心思想是通过分析图像块之间的相似度关系，识别出在多个图像中重复出现的异常模式，并将其与真正的正常模式区分开来。整体实现流程分为三个阶段：1）可疑链接的局部检测：利用邻居耗尽现象检测可疑链接；2）异常相似度图的构建：将可疑链接聚合为图像级别的图结构，节点代表图像，边表示共享可疑链接的频率；3）基于社区的过滤：检测异常密集的社区，从基集中移除这些异常图像，得到精炼的基集，并重新计算异常分数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "论文的关键创新点包括：1）识别并定义了'一致异常'问题；2）发现了相似度缩放和邻居耗尽两个关键现象；3）提出了CoDeGraph图框架；4）提供了理论解释。相比之前的工作，CoDeGraph的不同之处在于：专门处理重复异常模式，而不仅仅是假设异常是稀有的；利用图像之间的全局相似性结构，而不仅仅是局部相似性；通过多阶段处理流程更准确地识别和过滤一致异常；完全无训练，不需要额外标注数据或微调。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文识别并解决了零样本异常检测中的一致异常问题，提出了创新的CoDeGraph图框架，能够有效处理工业和医学影像中重复出现的异常模式，显著提升了零样本异常检测的准确性和实用性。'}


### 论文摘要

Zero-shot anomaly classification and segmentation (AC/AS) aim to detect anomalous samples and regions without any training data, a capability increasingly crucial in industrial inspection and medical imaging. This dissertation aims to investigate the core challenges of zero-shot AC/AS and presents principled solutions rooted in theory and algorithmic design.   We first formalize the problem of consistent anomalies, a failure mode in which recurring similar anomalies systematically bias distance-based methods. By analyzing the statistical and geometric behavior of patch representations from pre-trained Vision Transformers, we identify two key phenomena - similarity scaling and neighbor-burnout - that describe how relationships among normal patches change with and without consistent anomalies in settings characterized by highly similar objects.   We then introduce CoDeGraph, a graph-based framework for filtering consistent anomalies built on the similarity scaling and neighbor-burnout phenomena. Through multi-stage graph construction, community detection, and structured refinement, CoDeGraph effectively suppresses the influence of consistent anomalies.   Next, we extend this framework to 3D medical imaging by proposing a training-free, computationally efficient volumetric tokenization strategy for MRI data. This enables a genuinely zero-shot 3D anomaly detection pipeline and shows that volumetric anomaly segmentation is achievable without any 3D training samples.   Finally, we bridge batch-based and text-based zero-shot methods by demonstrating that CoDeGraph-derived pseudo-masks can supervise prompt-driven vision-language models. Together, this dissertation provides theoretical understanding and practical solutions for the zero-shot AC/AS problem.

---

## 200. A Co-evolutionary Approach for Heston Calibration

**论文链接:** [http://arxiv.org/abs/2512.03922v1](http://arxiv.org/abs/2512.03922v1)

**作者:** Julian Gutierrez

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文评估了Heston模型的一种协同进化校准框架，结合了遗传算法和演化的神经逆映射，探讨了数据集多样性对校准效果的影响。

### 背景

Heston模型是金融领域中常用的随机波动率模型，其参数校准对于期权定价至关重要。

### 目的

评估一种结合遗传算法和神经逆映射的协同进化校准框架，并研究数据集多样性对校准效果的影响。

### 方法

采用遗传算法(GA)优化参数，同时使用演化的神经逆映射从期权曲面映射到参数。比较了GA历史采样和拉丁超立方采样(LHS)两种数据生成方法的效果。

### 主要发现

GA历史采样能快速减少训练损失并产生强样本内拟合，但导致严重的过拟合；LHS生成的广泛数据集实现了几乎相当的校准精度，同时提供了更好的样本外稳定性。协同进化数据生成的改善主要反映目标特定专业化而非更可靠的全局逆映射。

### 结论

保持数据集多样性对于鲁棒的摊销校准至关重要，在Heston模型校准中应优先考虑数据多样性而非仅追求样本内拟合效果。

### 翻译

我们评估了一种Heston模型的协同进化校准框架，其中一种针对参数的遗传算法与一个从期权曲面到参数的演化的神经逆映射相结合。虽然GA历史采样可以快速减少训练损失并产生对目标曲面的强样本内拟合，但学习曲线诊断显示随着代数增加，训练-验证差距扩大，表明数据集集中且多样性不足导致了严重的过拟合。相比之下，通过拉丁超立方采样生成的广泛、空间填充的数据集实现了几乎相当的校准精度，同时在保留的曲面外提供了更好的样本外稳定性。这些结果表明，协同进化数据生成带来的改善主要反映了目标特定的专业化，而不是更可靠的全局逆映射，并且保持数据集多样性对于鲁棒的摊销校准至关重要。


### 论文摘要

We evaluate a co-evolutionary calibration framework for the Heston model in which a genetic algorithm (GA) over parameters is coupled to an evolving neural inverse map from option surfaces to parameters. While GA-history sampling can reduce training loss quickly and yields strong in-sample fits to the target surface, learning-curve diagnostics show a widening train--validation gap across generations, indicating substantial overfitting induced by the concentrated and less diverse dataset. In contrast, a broad, space-filling dataset generated via Latin hypercube sampling (LHS) achieves nearly comparable calibration accuracy while delivering markedly better out-of-sample stability across held-out surfaces. These results suggest that apparent improvements from co-evolutionary data generation largely reflect target-specific specialization rather than a more reliable global inverse mapping, and that maintaining dataset diversity is critical for robust amortized calibration.

---

## 201. Diminishing Returns in Self-Supervised Learning

**论文链接:** [http://arxiv.org/abs/2512.03862v1](http://arxiv.org/abs/2512.03862v1)

**作者:** Oli Bridge, Huey Sun, Botond Branyicskai-Nagy, Charles D'Ornano, Shomit Basu

**发布时间:** 2025-12-03

### GPT解析

### 总结

研究不同预训练、中间微调和下游数据集及训练目标对小型视觉Transformer模型性能的影响

### 背景

Transformer架构在计算机视觉和自然语言处理领域表现出色，但通常需要大量参数和训练数据才能获得良好性能

### 目的

探索三种不同的预训练、中间微调和下游数据集及训练目标对小型500万参数视觉Transformer的边际效益

### 方法

实验三种不同的预训练、中间微调和下游数据集及训练目标

### 主要发现

预训练和微调总是有帮助的但收益递减；中间微调可能对下游性能产生负面影响，可能是由于任务机制不相似

### 结论

小型ViTs从有针对性的预训练和仔细的数据选择中获益最多，不加选择地堆叠中间任务会浪费计算资源甚至降低性能

### 翻译

虽然基于Transformer的架构已经在计算机视觉和自然语言处理领域掀起风暴，但它们通常需要大量参数和训练数据才能获得强大的性能。在这项工作中，我们实验了三种不同的预训练、中间微调和下游数据集及训练目标，以探索它们对小型500万参数视觉Transformer的边际效益。我们发现，虽然预训练和微调总是对我们的模型有帮助，但收益递减，而中间微调实际上可能对下游性能产生有害影响，可能是由于任务机制的不相似性。综合来看，我们的结果表明，小型ViTs从有针对性的预训练和仔细的数据选择中获益最多，而不加选择地堆叠中间任务会浪费计算资源甚至降低性能。


### 论文摘要

While transformer-based architectures have taken computer vision and NLP by storm, they often require a vast amount of parameters and training data to attain strong performance. In this work, we experiment with three distinct pre-training, intermediate fine-tuning, and downstream datasets and training objectives to explore their marginal benefits on a small 5M-parameter vision transformer. We find that while pre-training and fine-tuning always help our model but have diminishing returns, intermediate fine-tuning can actually show harmful impact on downstream performance, potentially due to dissimilarity in task mechanics. Taken together, our results suggest that small-scale ViTs benefit most from targeted pre-training and careful data selection, while indiscriminate stacking of intermediate tasks can waste compute and even degrade performance.

---

## 202. DINO-RotateMatch: A Rotation-Aware Deep Framework for Robust Image Matching in Large-Scale 3D Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.03715v1](http://arxiv.org/abs/2512.03715v1)

**作者:** Kaichen Zhang, Tianxiang Sheng, Xuanming Shi

**发布时间:** 2025-12-03

**备注:** 9 pages, 5 figures, 1 table

### GPT解析

### 总结

本文提出DINO-RotateMatch框架，通过结合数据自适应图像配对策略与旋转感知的关键点提取和匹配，解决了从非结构化互联网图像进行大规模3D重建中的图像匹配挑战。

### 背景

从非结构化互联网图像进行大规模3D重建面临着图像匹配的挑战，需要有效的方法来处理大规模图像集合中的匹配问题。

### 目的

开发一种深度学习框架，能够有效解决大规模3D重建中的图像匹配问题，提高匹配的准确性和效率。

### 方法

DINO-RotateMatch框架集成了数据自适应图像配对策略与旋转感知的关键点提取和匹配。具体包括：使用DINO在大规模集合中检索语义相关的图像对；使用基于旋转的增强，通过ALIKED和Light Glue捕获方向依赖的局部特征。

### 主要发现

在Kaggle Image Matching Challenge 2025上的实验表明，该方法在平均平均准确率上有持续改进，获得了银奖(在943支队伍中排名第47位)。

### 结论

结合自监督全局描述符与旋转增强的局部匹配，为大规模3D重建提供了一种稳健且可扩展的解决方案。

### 翻译

本文提出了DINO-RotateMatch，这是一个深度学习框架，旨在解决从非结构化互联网图像进行大规模3D重建中的图像匹配挑战。该方法集成了数据自适应图像配对策略与旋转感知的关键点提取和匹配。DINO用于在大规模集合中检索语义相关的图像对，而基于旋转的增强使用ALIKED和Light Glue捕获方向依赖的局部特征。在Kaggle Image Matching Challenge 2025上的实验表明，平均平均准确率有持续改进，获得了银奖(943支队伍中排名第47位)。结果证实，将自监督全局描述符与旋转增强的局部匹配相结合，为大规模3D重建提供了一种稳健且可扩展的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从非结构化互联网图像进行大规模3D重建时的图像匹配问题。这个问题很重要，因为互联网上有数十亿张图像，这些图像是文化、艺术和科学研究的重要资源，而传统方法在处理这些光照、视角、分辨率和上下文差异巨大的图像时表现不佳，现有的深度学习方法又面临计算资源需求大、缺乏可解释性和依赖图像重叠等问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，然后借鉴了多个现有元素：DINO用于自监督特征学习和图像配对，ALIKED用于关键点提取，LightGlue用于特征匹配，以及COLMAP用于3D重建。作者的创新在于将这些元素整合到一个新框架中，并引入了旋转增强策略，根据数据集大小自适应选择图像配对方法，同时在关键点提取和匹配阶段应用旋转策略以提高鲁棒性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合自监督全局描述符与旋转增强的局部匹配，为大规模3D重建提供鲁棒且可扩展的解决方案。整体流程分为四个阶段：1)图像配对阶段，小数据集使用穷举搜索，大数据集使用DINO提取全局描述符；2)关键点提取阶段，对图像进行四个方向旋转后使用ALIKED提取关键点；3)特征匹配阶段，使用LightGlue建立跨旋转方向的对应关系；4)3D重建阶段，使用COLMAP生成密集点云或纹理网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于DINO的自适应图像配对策略；2)关键点提取中的旋转增强；3)关键点匹配中的旋转策略。相比之前的工作，不同之处在于：传统方法在复杂场景下鲁棒性不足，之前的学习方法在可扩展性和效率方面存在挑战，而本文方法通过旋转策略增加特征多样性，提高匹配鲁棒性，并结合自适应配对策略平衡了小数据集的准确性和大数据集的效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DINO-RotateMatch通过结合自监督的DINO图像配对策略和旋转增强的关键点提取与匹配方法，显著提高了从非结构化互联网图像进行大规模3D重建的准确性和鲁棒性。'}


### 论文摘要

This paper presents DINO-RotateMatch, a deep-learning framework designed to address the chal lenges of image matching in large-scale 3D reconstruction from unstructured Internet images. The   method integrates a dataset-adaptive image pairing strategy with rotation-aware keypoint extraction and   matching. DINO is employed to retrieve semantically relevant image pairs in large collections, while   rotation-based augmentation captures orientation-dependent local features using ALIKED and Light Glue. Experiments on the Kaggle Image Matching Challenge 2025 demonstrate consistent improve ments in mean Average Accuracy (mAA), achieving a Silver Award (47th of 943 teams). The results   confirm that combining self-supervised global descriptors with rotation-enhanced local matching offers   a robust and scalable solution for large-scale 3D reconstruction.

---

## 203. Structured Uncertainty Similarity Score (SUSS): Learning a Probabilistic, Interpretable, Perceptual Metric Between Images

**论文链接:** [http://arxiv.org/abs/2512.03701v1](http://arxiv.org/abs/2512.03701v1)

**作者:** Paula Seidler, Neill D. F. Campbell, Ivor J A Simpson

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为结构化不确定性相似性评分(SUSS)的新方法，用于评估图像之间的感知相似性。该方法通过生成式自监督方式训练，能够与人类感知判断保持高度一致，并提供可解释的评估结果。

### 背景

与人类视觉一致的感知相似性评分对于训练和评估计算机视觉模型至关重要。现有的深度感知损失(如LPIPS)虽然与人类视觉对齐良好，但依赖于复杂的高度非线性判别特征，其不变性未知；而手工设计的度量(如SSIM)虽然可解释，但缺乏关键的感知特性。

### 目的

开发一种新的感知相似性评分方法，既能与人类感知判断保持一致，又能提供可解释性，并且能够应用于下游图像处理任务。

### 方法

作者提出了结构化不确定性相似性评分(SUSS)，该方法将每个图像建模为一组感知组件，每个组件由结构化多元正态分布表示。这些组件通过生成式自监督方式训练，为人类无法感知的增强分配高可能性。最终得分是组件对数概率的加权和，权重从人类感知数据集中学习。与基于特征的方法不同，SUSS学习像素空间中残差图像特定的线性变换，可以通过去相关残差和采样进行透明检查。

### 主要发现

1. SUSS与人类感知判断高度一致；2. SUSS在不同类型的失真中表现出强大的感知校准能力；3. SUSS能够提供局部化、可解释的相似性评估解释；4. 当用作下游图像任务的感知损失时，SUSS表现出稳定的优化行为和竞争性能。

### 结论

SUSS是一种新的感知相似性评分方法，它结合了深度方法的准确性和手工设计方法的可解释性，为计算机视觉模型的训练和评估提供了有价值的工具。

### 翻译

与人类视觉一致的感知相似性评分对于训练和评估计算机视觉模型至关重要。深度感知损失(如LPIPS)实现了良好的对齐，但依赖于复杂的高度非线性判别特征，其不变性未知；而手工设计的度量(如SSIM)虽然可解释，但缺乏关键的感知特性。我们引入了结构化不确定性相似性评分(SUSS)；它通过一组感知组件对每个图像进行建模，每个组件由结构化多元正态分布表示。这些组件以生成式、自监督方式训练，为人类无法感知的增强分配高可能性。最终得分是组件对数概率的加权和，权重从人类感知数据集中学习。与基于特征的方法不同，SUSS学习像素空间中残差图像特定的线性变换，能够通过去相关残差和采样进行透明检查。SUSS与人类感知判断高度一致，在不同类型的失真中表现出强大的感知校准能力，并提供局部化、可解释的相似性评估解释。我们进一步证明了当SUSS用作下游图像任务的感知损失时，具有稳定的优化行为和竞争性能。


### 论文摘要

Perceptual similarity scores that align with human vision are critical for both training and evaluating computer vision models. Deep perceptual losses, such as LPIPS, achieve good alignment but rely on complex, highly non-linear discriminative features with unknown invariances, while hand-crafted measures like SSIM are interpretable but miss key perceptual properties.   We introduce the Structured Uncertainty Similarity Score (SUSS); it models each image through a set of perceptual components, each represented by a structured multivariate Normal distribution. These are trained in a generative, self-supervised manner to assign high likelihood to human-imperceptible augmentations. The final score is a weighted sum of component log-probabilities with weights learned from human perceptual datasets. Unlike feature-based methods, SUSS learns image-specific linear transformations of residuals in pixel space, enabling transparent inspection through decorrelated residuals and sampling.   SUSS aligns closely with human perceptual judgments, shows strong perceptual calibration across diverse distortion types, and provides localized, interpretable explanations of its similarity assessments. We further demonstrate stable optimization behavior and competitive performance when using SUSS as a perceptual loss for downstream imaging tasks.

---

## 204. State Space Models for Bioacoustics: A comparative Evaluation with Transformers

**论文链接:** [http://arxiv.org/abs/2512.03563v1](http://arxiv.org/abs/2512.03563v1)

**作者:** Chengyu Tang, Sanjeev Baskiyar

**发布时间:** 2025-12-03

### GPT解析

### 总结

本研究评估了Mamba模型在生物声学领域的有效性，通过预训练和微调BioMamba模型，并在BEANS基准测试上与AVES等基线模型进行比较，结果表明BioMamba在保持相当性能的同时显著减少了VRAM消耗。

### 背景

生物声学是研究生物声音信号的科学领域，随着深度学习技术的发展，基于Transformer的模型已成为该领域的先进方法，但通常需要大量计算资源。

### 目的

评估Mamba模型在生物声学任务中的有效性，特别是在保持性能的同时提高计算效率。

### 方法

首先在大规模音频数据上使用自监督学习预训练一个基于Mamba的音频大语言模型，然后在BEANS基准测试（包含分类和检测等多样化生物声学任务）上进行微调和评估，并与包括AVES在内的多个基线模型进行比较。

### 主要发现

BioMamba模型在BEANS基准测试上达到了与AVES相当的性能水平，同时VRAM消耗显著减少，表明Mamba架构在生物声学领域具有潜力。

### 结论

Mamba模型在生物声学领域展现出良好的应用前景，BioMamba能够在保持高性能的同时显著降低计算资源需求，为生物声学研究提供了更高效的选择。

### 翻译

在这项研究中，我们评估了Mamba模型在生物声学领域的有效性。我们首先在大规模音频数据语料库上使用自监督学习预训练了一个基于Mamba的音频大语言模型。我们在BEANS基准测试（一个包含分类和检测等多样化生物声学任务的集合）上对BioMamba进行微调和评估，并将其性能和效率与包括最先进的基于Transformer的模型AVES在内的多个基线模型进行比较。结果表明，BioMamba实现了与AVES相当的性能，同时VRAM消耗显著减少，展示了其在此领域的潜力。


### 论文摘要

In this study, we evaluate the efficacy of the Mamba model in the field of bioacoustics. We first pretrain a Mamba-based audio large language model (LLM) on a large corpus of audio data using self-supervised learning. We fine-tune and evaluate BioMamba on the BEANS benchmark, a collection of diverse bioacoustic tasks including classification and detection, and compare its performance and efficiency with multiple baseline models, including AVES, a state-of-the-art Transformer-based model. The results show that BioMamba achieves comparable performance with AVES while consumption significantly less VRAM, demonstrating its potential in this domain.

---

## 205. A Learning-based Control Methodology for Transitioning VTOL UAVs

**论文链接:** [http://arxiv.org/abs/2512.03548v1](http://arxiv.org/abs/2512.03548v1)

**作者:** Zexin Lin, Yebin Zhong, Hanwen Wan, Jiu Cheng, Zhenglong Sun, Xiaoqiang Ji

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种基于强化学习的耦合过渡控制方法，用于解决垂直起降无人机在过渡过程中的控制挑战，通过将巡航视为悬停的特殊情况，实现了更精确的控制并减少了振动。

### 背景

垂直起降无人机(VTOL UAV)在开发过程中面临过渡控制的关键挑战，倾斜旋翼机制在过渡过程中会改变重心和推力方向。

### 目的

开发一种新的耦合过渡控制方法，以解决当前控制方法中高度和位置解耦控制导致的振动和适应性限制问题。

### 方法

提出了一种基于强化学习(RL)驱动的控制器耦合过渡控制方法，采用ST3M方法将巡航模式视为悬停的特殊情况，而非传统的阶段过渡方式。

### 主要发现

该方法在仿真和真实环境中均表现出可行性，实现了高效的控制器开发和迁移，能够准确控制UAV的位置和姿态，实现出色的轨迹跟踪，并在过渡过程中显著减少振动。

### 结论

基于强化学习的耦合过渡控制方法为VTOL UAV的过渡控制提供了有效解决方案，克服了传统方法的局限性。

### 翻译

由于过渡控制中的倾斜旋翼机制会在过渡过程中改变重心和推力方向，垂直起降无人机(VTOL UAV)的开发面临关键挑战。当前控制方法对高度和位置的解耦控制导致显著振动，并限制了交互考虑和适应性。在本研究中，我们提出了一种基于强化学习(RL)驱动的控制器的新的耦合过渡控制方法。此外，与传统阶段过渡方法相比，ST3M方法通过将巡航模式视为悬停的特殊情况，展示了一种新视角。我们在仿真和真实环境中验证了应用我们方法的可行性，展示了高效的控制器开发和迁移，同时准确控制UAV的位置和姿态，在过渡过程中表现出出色的轨迹跟踪和减少的振动。


### 论文摘要

Transition control poses a critical challenge in Vertical Take-Off and Landing Unmanned Aerial Vehicle (VTOL UAV) development due to the tilting rotor mechanism, which shifts the center of gravity and thrust direction during transitions. Current control methods' decoupled control of altitude and position leads to significant vibration, and limits interaction consideration and adaptability. In this study, we propose a novel coupled transition control methodology based on reinforcement learning (RL) driven controller. Besides, contrasting to the conventional phase-transition approach, the ST3M method demonstrates a new perspective by treating cruise mode as a special case of hover. We validate the feasibility of applying our method in simulation and real-world environments, demonstrating efficient controller development and migration while accurately controlling UAV position and attitude, exhibiting outstanding trajectory tracking and reduced vibrations during the transition process.

---

## 206. Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.03508v1](http://arxiv.org/abs/2512.03508v1)

**作者:** Seogkyu Jeon, Kibeom Hong, Hyeran Byun

**发布时间:** 2025-12-03

**备注:** ICCV 2025 (poster)

### GPT解析

### 总结

本文提出了一个名为Domain-aware Prompt-driven Masked Transformer (DPMFormer)的新型领域泛化语义分割框架，通过解决视觉和文本上下文之间的语义不对齐问题，在多个DGSS基准测试上取得了最先进的性能。

### 背景

近期领域泛化语义分割研究通过从视觉语言模型中提取语义知识取得了显著进展，但这些研究忽视了视觉和文本上下文之间的语义不对齐问题，这源于在单一源域上学习的固定上下文提示的刚性。

### 目的

开发一个能够处理视觉和文本上下文语义不对齐问题的领域泛化语义分割框架，提高模型在不同领域环境下的适应性和鲁棒性。

### 方法

1) 引入领域感知提示学习促进视觉和文本线索的语义对齐；2) 提出领域感知对比学习和纹理扰动，在单一源数据集中捕获各种领域特定属性；3) 设计领域鲁棒一致性学习，指导模型最小化原始图像和增强图像预测之间的差异。

### 主要发现

通过实验和分析证明了DPMFormer框架的优越性，在各种DGSS基准测试上建立了新的最先进水平，有效解决了语义不对齐问题。

### 结论

DPMFormer框架通过创新的提示学习和对比学习方法，成功解决了领域泛化语义分割中的语义不对齐挑战，为处理多样化环境变化提供了有效的解决方案。

### 翻译

近期的领域泛化语义分割研究通过从视觉语言模型中提取语义知识取得了显著进展。然而，它们忽视了视觉和文本上下文之间的语义不对齐问题，这是由于在单一源域上学习的固定上下文提示的刚性导致的。为此，我们提出了一个用于语义分割的新型领域泛化框架，即领域感知提示驱动掩码Transformer(DPMFormer)。首先，我们引入领域感知提示学习，促进视觉和文本线索之间的语义对齐。为了在单一源数据集中捕获各种领域特定属性，我们提出了领域感知对比学习和纹理扰动，以多样化可观察的领域。最后，为了建立一个对各种环境变化具有韧性的框架，我们提出了领域鲁棒一致性学习，指导模型最小化原始图像和增强图像预测之间的差异。通过实验和分析，我们证明了所提出框架的优越性，在各种DGSS基准测试上建立了新的最先进水平。代码可在https://github.com/jone1222/DPMFormer获取。


### 论文摘要

Recent domain generalized semantic segmentation (DGSS) studies have achieved notable improvements by distilling semantic knowledge from Vision-Language Models (VLMs). However, they overlook the semantic misalignment between visual and textual contexts, which arises due to the rigidity of a fixed context prompt learned on a single source domain. To this end, we present a novel domain generalization framework for semantic segmentation, namely Domain-aware Prompt-driven Masked Transformer (DPMFormer). Firstly, we introduce domain-aware prompt learning to facilitate semantic alignment between visual and textual cues. To capture various domain-specific properties with a single source dataset, we propose domain-aware contrastive learning along with the texture perturbation that diversifies the observable domains. Lastly, to establish a framework resilient against diverse environmental changes, we have proposed the domain-robust consistency learning which guides the model to minimize discrepancies of prediction from original and the augmented images. Through experiments and analyses, we demonstrate the superiority of the proposed framework, which establishes a new state-of-the-art on various DGSS benchmarks. The code is available at https://github.com/jone1222/DPMFormer.

---

## 207. Rethinking Security in Semantic Communication: Latent Manipulation as a New Threat

**论文链接:** [http://arxiv.org/abs/2512.03361v1](http://arxiv.org/abs/2512.03361v1)

**作者:** Zhiyuan Xi, Kun Zhu

**发布时间:** 2025-12-03

**备注:** 8 pages, 6 figures

### GPT解析

### 总结

该论文研究了深度学习语义通信系统中的安全漏洞，提出了两种针对潜在空间语义的攻击方法，这些攻击可以改变传输的语义同时保持潜在表示的统计特性，使攻击难以检测。

### 背景

深度学习语义通信（SemCom）作为新一代无线网络的有前景范式，通过提取和传递任务相关的语义潜在表示而非原始数据，提供了卓越的传输效率。然而，无线介质的开放性和语义潜在表示的固有脆弱性使这类系统面临以前未被识别的安全风险。

### 目的

揭示语义通信系统中潜在空间的基本安全漏洞，展示中间人（MitM）攻击者如何在保持传输潜在表示统计特性的同时秘密操纵传输的语义。

### 方法

论文提出了两种攻击方法：1）基于扩散的重新编码攻击（DiR）：攻击者使用扩散模型合成攻击者设计的语义变体，并将其重新编码为与语义通信解码器兼容的有效潜在表示；2）与模型无关且无需训练的测试时适应潜在操作攻击（TTA-LM）：攻击者通过利用目标损失函数的梯度，干扰和引导拦截的潜在表示向攻击者指定的语义目标转变。

### 主要发现

两种攻击都能显著改变解码的语义，同时保持自然潜在空间分布，使攻击具有隐蔽性和难以检测的特点。TTA-LM不依赖任何生成模型，不施加模态特定或任务特定的假设，从而能够高效且广泛地应用于各种语义通信架构的潜在空间篡改。

### 结论

语义通信系统存在潜在空间的基本安全漏洞，攻击者可以在不改变潜在表示统计特性的情况下操纵语义。提出的两种攻击方法在不同语义通信架构上都被证明是有效的，这对语义通信系统的安全性提出了重要挑战。

### 翻译

基于深度学习的语义通信（SemCom）已成为下一代无线网络的有前景范式，通过提取和传递任务相关的语义潜在表示而非原始数据，提供卓越的传输效率。然而，无线介质的开放性和语义潜在表示的固有脆弱性使此类系统面临以前未被识别的安全风险。在本文中，我们揭示了一个基本的潜在空间漏洞，使中间人（MitM）攻击者能够在保持传输潜在表示统计特性的同时秘密操纵传输的语义。我们首先提出一种基于扩散的重新编码攻击（DiR），其中攻击者使用扩散模型合成攻击者设计的语义变体，并将其重新编码为与语义通信解码器兼容的有效潜在表示。除了这种依赖于模型的途径外，我们还进一步提出了一种与模型无关且无需训练的测试时适应潜在操作攻击（TTA-LM），其中攻击者通过利用目标损失函数的梯度，干扰和引导拦截的潜在表示向攻击者指定的语义目标转变。与基于扩散的操作不同，TTA-LM不依赖任何生成模型，不施加模态特定或任务特定的假设，从而能够高效且广泛地应用于各种语义通信架构的潜在空间篡改。在代表性语义通信架构上的广泛实验表明，两种攻击都能显著改变解码的语义，同时保持自然潜在空间分布，使攻击具有隐蔽性和难以检测。


### 论文摘要

Deep learning-based semantic communication (SemCom) has emerged as a promising paradigm for next-generation wireless networks, offering superior transmission efficiency by extracting and conveying task-relevant semantic latent representations rather than raw data. However, the openness of the wireless medium and the intrinsic vulnerability of semantic latent representations expose such systems to previously unrecognized security risks. In this paper, we uncover a fundamental latent-space vulnerability that enables Man-in-the-Middle (MitM) attacker to covertly manipulate the transmitted semantics while preserving the statistical properties of the transmitted latent representations. We first present a Diffusion-based Re-encoding Attack (DiR), wherein the attacker employs a diffusion model to synthesize an attacker-designed semantic variant, and re-encodes it into a valid latent representation compatible with the SemCom decoder. Beyond this model-dependent pathway, we further propose a model-agnostic and training-free Test-Time Adaptation Latent Manipulation attack (TTA-LM), in which the attacker perturbs and steers the intercepted latent representation toward an attacker-specified semantic target by leveraging the gradient of a target loss function. In contrast to diffusion-based manipulation, TTA-LM does not rely on any generative model and does not impose modality-specific or task-specific assumptions, thereby enabling efficient and broadly applicable latent-space tampering across diverse SemCom architectures. Extensive experiments on representative semantic communication architectures demonstrate that both attacks can significantly alter the decoded semantics while preserving natural latent-space distributions, making the attacks covert and difficult to detect.

---

## 208. ASPEN: An Adaptive Spectral Physics-Enabled Network for Ginzburg-Landau Dynamics

**论文链接:** [http://arxiv.org/abs/2512.03290v1](http://arxiv.org/abs/2512.03290v1)

**作者:** Julian Evan Chrisnanto, Nurfauzi Fadillah, Yulison Herry Chrisnanto

**发布时间:** 2025-12-02

**备注:** 15 pages, 7 figures

### GPT解析

### 总结

本文介绍了ASPEN（自适应频谱物理使能网络），一种新型神经网络架构，用于解决传统物理信息神经网络（PINNs）在处理刚性、多尺度非线性系统时的局限性。

### 背景

物理信息神经网络（PINNs）已成为求解偏微分方程（PDEs）的强大无网格方法，但它们在处理刚性、多尺度、非线性系统时存在困难，这是由于标准多层感知器（MLP）架构的内在频谱偏差导致的，这种偏差使它们无法充分表示高频分量。

### 目的

开发一种新的网络架构来克服传统PINNs在处理高频分量时的局限性，使其能够有效解决复杂的动态系统问题。

### 方法

作者提出了ASPEN，它将自适应频谱层与可学习的傅里叶特征直接集成到网络的输入阶段。这种机制使模型能够在训练过程中动态调整自身的频谱基，从而有效地学习和表示解决方案所需的精确频率内容。

### 主要发现

在应用于复杂的Ginzburg-Landau方程（CGLE）这一非线性、刚性时空动力学的标准基准测试中，标准PINN架构完全失败，产生非物理振荡。相比之下，ASPEN以极高的精度成功求解了CGLE，预测的解与高分辨率真实解在视觉上无法区分，达到了5.10 x 10^-3的低中值物理残差。此外，ASPEN的解不仅在点上是准确的，而且在物理上是一致的，正确地捕获了快速自由能松弛和域壁前沿的长期稳定性等涌现物理特性。

### 结论

通过结合自适应频谱基，该框架为标准PINNs失败的复杂动态系统提供了稳健且物理上一致的求解器，为机器学习在具有挑战性的物理领域开辟了新的可能性。

### 翻译

物理信息神经网络（PINNs）已成为求解偏微分方程（PDEs）的强大无网格范式。然而，由于标准多层感知器（MLP）架构的内在频谱偏差，它们在处理刚性、多尺度、非线性系统时 notoriously 困难，这种偏差使它们无法充分表示高频分量。在本工作中，我们引入了自适应频谱物理使能网络（ASPEN），这是一种旨在克服这一关键限制的新型架构。ASPEN将自适应频谱层与可学习的傅里叶特征直接集成到网络的输入阶段。这种机制使模型能够在训练过程中动态调整自身的频谱基，从而有效地学习和表示解决方案所需的精确频率内容。我们通过将ASPEN应用于复杂的Ginzburg-Landau方程（CGLE）来证明其有效性，该方程是非线性、刚性时空动力学的典型且具有挑战性的基准测试。我们的结果表明，标准PINN架构在该问题上完全失败，发散为非物理振荡。相比之下，ASPEN以极高的精度成功求解了CGLE。预测的解与高分辨率真实解在视觉上无法区分，实现了5.10 x 10^-3的低中值物理残差。此外，我们验证了ASPEN的解不仅在点上是准确的，而且在物理上是一致的，正确地捕获了涌现的物理特性，包括快速自由能松弛和域壁前沿的长期稳定性。这项工作表明，通过结合自适应频谱基，我们的框架为标准PINNs失败的复杂动态系统提供了稳健且物理上一致的求解器，为机器学习在具有挑战性的物理领域开辟了新的选择。


### 论文摘要

Physics-Informed Neural Networks (PINNs) have emerged as a powerful, mesh-free paradigm for solving partial differential equations (PDEs). However, they notoriously struggle with stiff, multi-scale, and nonlinear systems due to the inherent spectral bias of standard multilayer perceptron (MLP) architectures, which prevents them from adequately representing high-frequency components. In this work, we introduce the Adaptive Spectral Physics-Enabled Network (ASPEN), a novel architecture designed to overcome this critical limitation. ASPEN integrates an adaptive spectral layer with learnable Fourier features directly into the network's input stage. This mechanism allows the model to dynamically tune its own spectral basis during training, enabling it to efficiently learn and represent the precise frequency content required by the solution. We demonstrate the efficacy of ASPEN by applying it to the complex Ginzburg-Landau equation (CGLE), a canonical and challenging benchmark for nonlinear, stiff spatio-temporal dynamics. Our results show that a standard PINN architecture catastrophically fails on this problem, diverging into non-physical oscillations. In contrast, ASPEN successfully solves the CGLE with exceptional accuracy. The predicted solution is visually indistinguishable from the high-resolution ground truth, achieving a low median physics residual of 5.10 x 10^-3. Furthermore, we validate that ASPEN's solution is not only pointwise accurate but also physically consistent, correctly capturing emergent physical properties, including the rapid free energy relaxation and the long-term stability of the domain wall front. This work demonstrates that by incorporating an adaptive spectral basis, our framework provides a robust and physically-consistent solver for complex dynamical systems where standard PINNs fail, opening new options for machine learning in challenging physical domains.

---

## 209. Assumption-Lean Differential Variance Inference for Heterogeneous Treatment Effect Detection

**论文链接:** [http://arxiv.org/abs/2512.03254v1](http://arxiv.org/abs/2512.03254v1)

**作者:** Philippe A. Boileau, Hani Zaki, Gabriele Lileikyte, Niklas Nielsen, Patrick R. Lawler, Mireille E. Schnitzer

**发布时间:** 2025-12-02

### GPT解析

### 总结

本研究提出了一种通过推断潜在结果方差的对比来评估同质处理效应假设的方法，解决了当效果修饰符缺失或测量错误时CATE-based技术无法检测异质性处理效应的问题。

### 背景

条件性平均处理效应(CATE)常用于反驳同质处理效应假设，后者认为研究人群中的所有单位从给定处理中获得的益处是相同的。然而，基于CATE的技术在效果修饰符缺失或测量错误时无法检测异质性处理效应。

### 目的

开发一种方法，即使在效果修饰符缺失或测量错误的情况下，也能评估同质处理效应假设并进行正式假设检验。

### 方法

推导了潜在结果方差的对比的因果机器学习估计量，并研究其渐近性质。建立了这些估计量在温和条件下是双重稳健和渐近线性的。

### 主要发现

数值实验表明，这些估计量的渐近保证在实验数据和观察数据中都能近似实现。这些推断程序成功用于检测心脏骤停患者目标温度管理随机对照试验中的异质性处理效应。

### 结论

所提出的方法允许在效果修饰符缺失或测量错误的情况下对同质处理效应假设进行正式假设检验，扩展了异质性处理效应检测的适用范围。

### 翻译

条件性平均处理效应(CATE)经常被估计以反驳同质处理效应假设。在该假设下，构成研究人群的所有单位从给定处理中经历相同的益处。然而，通过关于CATE的推断来揭示异质性处理效应，要求在基线时真正可靠地收集修改处理效果的协变量。当由于资源限制等原因效果修饰符从数据中遗漏时，基于CATE的技术将必然无法检测到违反假设的情况。严重的测量误差有类似的影响。为了解决这些限制，我们证明可以通过关于潜在结果方差的对比的推断来评估同质处理效应假设。我们推导了这些对比的因果机器学习估计量，并研究了它们的渐近性质。我们建立了这些估计量在温和条件下是双重稳健和渐近线性的，允许在效果修饰符缺失或测量错误的情况下对同质处理效应假设进行正式假设检验。数值实验表明，这些估计量的渐近保证在实验数据和观察数据中都能近似实现。然后使用这些推断程序来重新分析研究心脏骤停患者目标温度管理的随机对照试验，以检测异质性处理效应。


### 论文摘要

The conditional average treatment effect (CATE) is frequently estimated to refute the homogeneous treatment effect assumption. Under this assumption, all units making up the population under study experience identical benefit from a given treatment. Uncovering heterogeneous treatment effects through inference about the CATE, however, requires that covariates truly modifying the treatment effect be reliably collected at baseline. CATE-based techniques will necessarily fail to detect violations when effect modifiers are omitted from the data due to, for example, resource constraints. Severe measurement error has a similar impact. To address these limitations, we prove that the homogeneous treatment effect assumption can be gauged through inference about contrasts of the potential outcomes' variances. We derive causal machine learning estimators of these contrasts and study their asymptotic properties. We establish that these estimators are doubly robust and asymptotically linear under mild conditions, permitting formal hypothesis testing about the homogeneous treatment effect assumption even when effect modifiers are missing or mismeasured. Numerical experiments demonstrate that these estimators' asymptotic guarantees are approximately achieved in experimental and observational data alike. These inference procedures are then used to detect heterogeneous treatment effects in the re-analysis of randomized controlled trials investigating targeted temperature management in cardiac arrest patients.

---

## 210. 论文ID: 2512.03196v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03196v1.json'

---

## 211. 论文ID: 2512.03176v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.03176v1.json'

---

## 212. Unrolled Networks are Conditional Probability Flows in MRI Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.03020v1](http://arxiv.org/abs/2512.03020v1)

**作者:** Kehan Qi, Saumya Gupta, Qingqiao Hu, Weimin Lyu, Chao Chen

**发布时间:** 2025-12-02

### GPT解析

### 总结

本文提出了一种名为Flow-Aligned Training (FLAT)的新方法，通过将展开网络与条件概率流ODEs理论连接，改进了磁共振成像(MRI)重建的稳定性和收敛性。

### 背景

磁共振成像提供优秀的软组织对比度且无电离辐射，但长采集时间限制了临床应用。最近的方法通过欠采样k空间并使用深度学习重建图像来加速MRI。展开网络虽被广泛使用，但因中间步骤参数不稳定而受限；扩散模型虽稳定但计算成本高。

### 目的

引入流ODEs到MRI重建中，通过理论证明展开网络是条件概率流ODEs的离散实现，提供参数的显式公式，并改进重建的稳定性和收敛性。

### 方法

提出Flow-Aligned Training (FLAT)方法，从ODE离散化推导展开参数，并将中间重建与理想的ODE轨迹对齐，以提高稳定性和收敛性。

### 主要发现

在三个MRI数据集上的实验表明，FLAT实现了高质量重建，比基于扩散的生成模型少3倍的迭代次数，且比展开网络具有显著更高的稳定性。

### 结论

通过将展开网络与流ODEs理论连接，FLAT方法在MRI重建中实现了更好的性能和稳定性，为加速MRI提供了一种有效的方法。

### 翻译

磁共振成像(MRI)提供优秀的软组织对比度而无电离辐射，但其长采集时间限制了临床应用。最近的方法通过欠采样k空间并使用深度学习重建图像来加速MRI。展开网络因其效率被广泛用于重建任务，但中间步骤中可自由学习的参数导致不稳定演化。相比之下，基于随机微分方程的扩散模型在医学和自然图像任务中提供理论稳定性，但计算成本高。在这项工作中，我们通过理论证明展开网络是条件概率流ODEs的离散实现，将流ODEs引入到MRI重建中。这种连接提供了参数的显式公式，并阐明了中间状态应该如何演化。基于这一见解，我们提出了Flow-Aligned Training (FLAT)，它从ODE离散化推导展开参数，并将中间重建与理想的ODE轨迹对齐，以提高稳定性和收敛性。在三个MRI数据集上的实验表明，FLAT实现了高质量重建，比基于扩散的生成模型少3倍的迭代次数，且比展开网络具有显著更高的稳定性。


### 论文摘要

Magnetic Resonance Imaging (MRI) offers excellent soft-tissue contrast without ionizing radiation, but its long acquisition time limits clinical utility. Recent methods accelerate MRI by under-sampling $k$-space and reconstructing the resulting images using deep learning. Unrolled networks have been widely used for the reconstruction task due to their efficiency, but suffer from unstable evolving caused by freely-learnable parameters in intermediate steps. In contrast, diffusion models based on stochastic differential equations offer theoretical stability in both medical and natural image tasks but are computationally expensive. In this work, we introduce flow ODEs to MRI reconstruction by theoretically proving that unrolled networks are discrete implementations of conditional probability flow ODEs. This connection provides explicit formulations for parameters and clarifies how intermediate states should evolve. Building on this insight, we propose Flow-Aligned Training (FLAT), which derives unrolled parameters from the ODE discretization and aligns intermediate reconstructions with the ideal ODE trajectory to improve stability and convergence. Experiments on three MRI datasets show that FLAT achieves high-quality reconstructions with up to $3\times$ fewer iterations than diffusion-based generative models and significantly greater stability than unrolled networks.

---

## 213. Layout Anything: One Transformer for Universal Room Layout Estimation

**论文链接:** [http://arxiv.org/abs/2512.02952v1](http://arxiv.org/abs/2512.02952v1)

**作者:** Md Sohag Mia, Muhammad Abdullah Adnan

**发布时间:** 2025-12-02

**备注:** Published at WACV 2026

### GPT解析

### 总结

本文提出了Layout Anything，一个基于Transformer的室内布局估计框架，将OneFormer的通用分割架构适应到几何结构预测中，通过整合布局退化策略和可微分几何损失，实现了高性能和高效率的室内布局估计。

### 背景

室内布局估计是计算机视觉和3D场景重建中的重要任务，需要精确预测房间的几何结构和边界，而现有方法可能存在复杂后处理流程或计算效率低的问题。

### 目的

开发一个高效准确的室内布局估计框架，能够直接预测平面一致性和边界，同时保持高推理速度，适用于增强现实应用和大规模3D场景重建。

### 方法

采用基于Transformer的架构，结合OneFormer的任务条件查询和对比学习，实现两个关键模块：(1)布局退化策略，通过拓扑感知变换增强训练数据同时保留曼哈顿世界约束；(2)可微分几何损失，在训练过程中直接强制执行平面一致性和边界预测；构建端到端框架消除复杂的后处理流程。

### 主要发现

在多个标准基准测试中实现了最先进的性能：LSUN数据集上像素误差5.43%，角点误差4.02%；Hedau数据集上像素误差7.04%，角点误差5.17%；Matterport3D-Layout数据集上像素误差4.03%，角点误差3.15%；推理速度达到114毫秒。

### 结论

Layout Anything框架结合了几何感知和计算效率，特别适合增强现实应用和大规模3D场景重建任务。

### 翻译

我们提出了Layout Anything，一个基于Transformer的室内布局估计框架，它将OneFormer的通用分割架构适应到几何结构预测中。我们的方法将OneFormer的任务条件查询和对比学习与两个关键模块相结合：(1)一种布局退化策略，通过拓扑感知变换增强训练数据，同时保留曼哈顿世界约束；(2)可微分几何损失，在训练过程中直接强制执行平面一致性和边界预测。通过在端到端框架中统一这些组件，该模型消除了复杂的后处理流程，同时以114毫秒的速度实现高速推理。大量实验证明了在标准基准测试中的最先进性能，在LSUN上的像素误差为5.43%，角点误差为4.02%，在Hedau上的像素误差为7.04%(角点误差为5.17%)，在Matterport3D-Layout上的像素误差为4.03%(角点误差为3.15%)。该框架结合了几何感知和计算效率，使其特别适合增强现实应用和大规模3D场景重建任务。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决室内房间布局估计问题，即从单张RGB图像中推断房间的空间结构，包括墙壁、地板和天花板等。这个问题在计算机视觉中非常重要，它是场景理解的基础，广泛应用于增强现实、机器人导航、虚拟展示和3D场景重建等领域。现有方法要么需要复杂且缓慢的后处理，要么在严重遮挡或复杂布局下会牺牲精度，或者难以处理非立方体结构的房间。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了OneFormer的通用分割架构，将其适应几何结构预测。他们结合了任务条件查询和对比学习技术，并引入了两个关键模块：布局退化策略和可微分几何损失。作者受到经典立方体布局模型的启发设计了几何感知的正则化损失，同时借鉴了拓扑推理思想设计了保持拓扑的退化策略。整体思路是将布局估计视为结构化分割任务，由平面表面语义引导并由几何和拓扑约束加强。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将布局估计视为结构化分割任务，统一任务条件分割与几何先验，直接推断几何一致的布局而不依赖立方体先验或昂贵后处理。整体流程包括：1)使用基于transformer的OneFormer作为骨干网络；2)采用任务条件联合分割，通过自然语言提示条件化；3)生成对象查询并对应布局表面；4)生成各类别掩码并分配像素标签；5)使用多种损失函数监督分割；6)应用几何正则化损失；7)在训练时使用拓扑保持退化策略；8)所有组件集成到端到端系统中。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)利用基于transformer的OneFormer框架进行条件化布局解析；2)引入布局退化策略增加数据多样性；3)设计可微分几何损失强制执行平面一致性和尖锐边界；4)统一组件到端到端框架消除复杂后处理。相比之前工作，不同之处在于：早期方法依赖强几何先验和手工特征；许多深度学习方法需要复杂后处理；立方体布局方法泛化能力有限；图形化方法依赖线框质量；一些方法难以平衡准确性和速度。Layout Anything实现了高精度和高速推理(114ms/图像)，同时保持几何一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Layout Anything提出了一种基于transformer的统一框架，通过结合任务条件分割、几何正则化和拓扑保持退化策略，实现了高效、准确的室内房间布局估计，在保持高推理速度的同时达到了最先进的性能。'}


### 论文摘要

We present Layout Anything, a transformer-based framework for indoor layout estimation that adapts the OneFormer's universal segmentation architecture to geometric structure prediction. Our approach integrates OneFormer's task-conditioned queries and contrastive learning with two key modules: (1) a layout degeneration strategy that augments training data while preserving Manhattan-world constraints through topology-aware transformations, and (2) differentiable geometric losses that directly enforce planar consistency and sharp boundary predictions during training. By unifying these components in an end-to-end framework, the model eliminates complex post-processing pipelines while achieving high-speed inference at 114ms. Extensive experiments demonstrate state-of-the-art performance across standard benchmarks, with pixel error (PE) of 5.43% and corner error (CE) of 4.02% on the LSUN, PE of 7.04% (CE 5.17%) on the Hedau and PE of 4.03% (CE 3.15%) on the Matterport3D-Layout datasets. The framework's combination of geometric awareness and computational efficiency makes it particularly suitable for augmented reality applications and large-scale 3D scene reconstruction tasks.

---

## 214. GNSS Array-Based Multipath Detection Employing UKF on Manifolds

**论文链接:** [http://arxiv.org/abs/2512.02994v1](http://arxiv.org/abs/2512.02994v1)

**作者:** Abdelgabar Ahmed, Tarig Ballal, Xing Liu, Mohanad Ahmed, Tareq Y. Al-Naffouri

**发布时间:** 2025-12-02

**DOI:** 10.1109/PLANS61210.2025.11028455

**备注:** The paper, was presented at the ION PLANS 2025 meeting (Position, Location, and Navigation Symposium) in Session C1: Multisensor Integrated Systems and Sensor Fusion Technologies, and is published in the conference proceedings

### GPT解析

### 总结

本文提出了一种基于GNSS阵列的多路径检测算法，结合实时姿态估计和RANSAC算法，有效解决了城市环境中多路径干扰问题，显著提高了定位精度。

### 背景

全球导航卫星系统(GNSS)应用常受到各种误差源的影响，其中多路径干扰是最具挑战性的问题之一，特别是在城市环境中。

### 目的

实现一种GNSS阵列式多路径检测算法，结合实时姿态估计以应对动态场景，提高定位和姿态确定的准确性。

### 方法

在流形上使用无迹卡尔曼滤波(UKF)融合GNSS和IMU数据，利用卫星组合的姿态信息识别并排除受多路径影响的卫星，并采用随机抽样一致性(RANSAC)算法减少计算量。

### 主要发现

该方法能有效检测受多路径干扰影响的卫星，在定位精度上有显著改善，特别是在大部分可见卫星受到严重多路径污染的场景中效果尤为明显。

### 结论

提出的GNSS阵列多路径检测算法结合实时姿态估计和RANSAC算法，能够有效识别和排除多路径干扰，显著提高GNSS在复杂环境中的定位和姿态确定性能。

### 翻译

全球导航卫星系统(GNSS)应用常受到各种误差源的影响，其中多路径干扰是最具挑战性的问题之一，特别是在城市环境中。本文在先前研究基础上，实现了基于GNSS阵列的多路径检测算法，并结合实时姿态估计以应对动态场景。该方法在流形上使用无迹卡尔曼滤波(UKF)融合GNSS和IMU数据，实现连续姿态跟踪。所提出的方法利用卫星组合的姿态信息来识别并排除受多路径影响的卫星，提高了定位和姿态确定的准确性。为解决评估大量卫星组合带来的计算挑战，我们提出了使用随机抽样一致性(RANSAC)算法，减少评估的组合数量同时保持高检测性能。性能评估使用KITTI数据集的轨迹和IMU读数进行，GNSS观测基于真实位置和卫星星历进行模拟。结果表明，所提出的方法在检测受多路径干扰影响的卫星方面是有效的。在大部分可见卫星受到严重多路径污染的场景中，观察到定位精度有显著提高。


### 论文摘要

Global Navigation Satellite Systems (GNSS) applications are often hindered by various sources of error, with multipath interference being one of the most challenging, particularly in urban environments. In this work, we build on previous research by implementing a GNSS array-based multipath detection algorithm, incorporating real-time attitude estimation for dynamic scenarios. The method fuses GNSS and IMU data using an Unscented Kalman Filter (UKF) on a manifold, enabling continuous attitude tracking. The proposed approach utilizes attitude information from satellite combinations to identify and exclude multipath-affected satellites, improving the accuracy of both positioning and attitude determination. To address computational challenges associated with evaluating large numbers of satellite combinations, we propose the use of the Random Sample Consensus (RANSAC) algorithm, which reduces the number of combinations assessed while maintaining high detection performance. Performance evaluations are conducted using trajectories and IMU readings from the KITTI dataset. GNSS observations are simulated based on ground truth positions and satellite ephemeris. The results demonstrate the effectiveness of the proposed approach in detecting satellites affected by multipath interference. Significant improvements in positioning accuracy are observed, particularly in scenarios where a large portion of the visible satellites are contaminated by severe multipath.

---

