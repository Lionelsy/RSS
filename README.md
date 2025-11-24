# 今日论文推荐 - 2025-11-24

共 151 篇论文

---

## 1. Improving Multimodal Distillation for 3D Semantic Segmentation under Domain Shift

**论文链接:** [http://arxiv.org/abs/2511.17455v1](http://arxiv.org/abs/2511.17455v1)

**作者:** Björn Michele, Alexandre Boulch, Gilles Puy, Tuan-Hung Vu, Renaud Marlet, Nicolas Courty

**发布时间:** 2025-11-21

**备注:** Accepted at BMVC 2025

### GPT解析

### 总结

本研究探讨了在无监督领域适应中利用视觉基础模型(VFMs)进行激光雷达点云语义分割的最佳方法，提出了一种新的管道，在四个具有挑战性的设置中实现了最先进的结果。

### 背景

在完全监督下为一种激光雷达训练的语义分割网络无法在没有干预的情况下推广到未见过的激光雷达上，导致在领域转移时性能差距较大。

### 目的

减少领域转移下的性能差距，研究如何有效利用视觉基础模型(VFMs)提供的跨领域鲁棒特征，以实现激光雷达点云语义分割的无监督领域适应。

### 方法

基于无监督图像到激光雷达知识蒸馏方法，进行详尽研究以确定在激光雷达点云语义分割的无监督领域适应中利用VFMs的最佳方案。

### 主要发现

(1)激光雷达主干架构是最大化目标领域泛化性能的关键；(2)可以预训练单个主干一次性，并用于解决多种领域转移；(3)通过保持预训练主干冻结并训练MLP头进行语义分割可获得最佳结果。

### 结论

所提出的管道在四个广泛认可且具有挑战性的设置中实现了最先进的结果，代码将在https://github.com/valeoai/muddos上公开。

### 翻译

为一种激光雷达训练的语义分割网络在完全监督下无法在没有干预的情况下推广到未见过的激光雷达上。为了减少领域转移下的性能差距，最近的趋势是利用提供跨领域鲁棒特征的视觉基础模型(VFMs)。在本工作中，我们进行了详尽的研究，以确定在激光雷达点云语义分割的无监督领域适应中利用VFMs的方案。基于无监督图像到激光雷达知识蒸馏，我们的研究揭示：(1)激光雷达主干架构是最大化目标领域泛化性能的关键；(2)可以预训练单个主干一次性，并用于解决多种领域转移；(3)通过保持预训练主干冻结并训练MLP头进行语义分割可获得最佳结果。所提出的管道在四个广泛认可且具有挑战性的设置中实现了最先进的结果。代码将在https://github.com/valeoai/muddos上公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决激光雷达语义分割网络在不同传感器之间的域偏移问题。在自动驾驶和机器人应用中，这个问题至关重要，因为它限制了在不同激光雷达传感器之间部署训练好的模型，而不同传感器(如不同数量的激光束、分辨率)会导致性能显著下降。有效的域适应可以减少对大量标注数据的依赖，使模型能够更好地适应新环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过系统性研究沿着四个方向探索如何最好地利用视觉基础模型(VFMs)进行3D无监督域适应：(1)分析3D主干架构对域差距的影响；(2)评估不同的VFMs；(3)评估不同的下游训练方法；(4)研究预训练数据集选择的影响。作者借鉴了现有工作如ScaLR知识蒸馏方法、xMUDA多模态域适应方法，以及MinkowskiUNet等点云处理架构，但通过系统性研究发现了更有效的配置，如使用WaffleIron架构、层归一化而非批归一化、DINOv2而非SAM等。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视觉基础模型提供的跨域鲁棒特征，通过多模态知识蒸馏技术将这些特征转移到激光雷达点云处理网络中，实现有效的域适应。整体实现流程分为三步：1)多模态蒸馏阶段：使用冻结的DINOv2提取图像特征，通过相似度损失对齐2D图像特征和3D点云特征，在多个数据集上进行预训练；2)源域分类器训练阶段：冻结预训练的激光雷达主干网络，在源域数据上训练MLP分类头；3)自训练阶段(可选)：使用教师-学生机制生成高置信度伪标签，进一步优化目标域性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：(1)架构选择：发现WaffleIron架构比传统MinkowskiUNet更有效，层归一化优于批归一化，不使用强度作为输入特征；(2)预训练策略：证明DINOv2比SAM提供更鲁棒特征，可在多数据集上进行一次性预训练；(3)下游训练方法：发现冻结主干并训练MLP头比完全微调更有效，顺序训练优于联合训练。相比之前工作(如Adapt-SAM)，主要不同在于架构选择、归一化方法、输入特征处理、视觉模型选择、训练策略和预训练方法等方面。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '通过系统性地研究多模态蒸馏方法，本文提出MuDDoS方法，显著提高了3D语义分割在域偏移场景下的性能，实现了比之前最先进方法高出15% mIoU的改进，并简化了训练流程。'}


### 论文摘要

Semantic segmentation networks trained under full supervision for one type of lidar fail to generalize to unseen lidars without intervention. To reduce the performance gap under domain shifts, a recent trend is to leverage vision foundation models (VFMs) providing robust features across domains. In this work, we conduct an exhaustive study to identify recipes for exploiting VFMs in unsupervised domain adaptation for semantic segmentation of lidar point clouds. Building upon unsupervised image-to-lidar knowledge distillation, our study reveals that: (1) the architecture of the lidar backbone is key to maximize the generalization performance on a target domain; (2) it is possible to pretrain a single backbone once and for all, and use it to address many domain shifts; (3) best results are obtained by keeping the pretrained backbone frozen and training an MLP head for semantic segmentation. The resulting pipeline achieves state-of-the-art results in four widely-recognized and challenging settings. The code will be available at: https://github.com/valeoai/muddos.

---

## 2. Mesh RAG: Retrieval Augmentation for Autoregressive Mesh Generation

**论文链接:** [http://arxiv.org/abs/2511.16807v1](http://arxiv.org/abs/2511.16807v1)

**作者:** Xiatao Sun, Chen Liang, Qian Wang, Daniel Rakita

**发布时间:** 2025-11-20

### GPT解析

### 总结

Mesh RAG是一种新型的、无需训练的即插即用框架，用于自回归网格生成模型，通过基于检索的方法克服了传统方法的顺序依赖性限制，提高了生成质量、加速了生成速度，并支持增量编辑。

### 背景

3D网格是工业设计、游戏、仿真和机器人应用等领域的核心组成部分。传统上，网格由艺术家手工制作，这是一个耗时且难以扩展的过程。自回归模型已成为艺术网格生成的强大范式，但当前提高质量的方法通常依赖更大或更长的序列，导致生成时间更长，且其固有的顺序性质造成了质量与速度之间的严重权衡。

### 目的

为了克服现有自回归网格生成模型的局限性，特别是质量与速度的权衡以及增量编辑困难的问题，作者提出了Mesh RAG框架。

### 方法

Mesh RAG受语言模型RAG启发，通过点云分割、空间变换和点云配准来检索、生成和整合网格组件。这种基于检索的方法将生成与其严格的顺序依赖性解耦，实现了高效和可并行化的推理。

### 主要发现

Mesh RAG在各种基础自回归网格生成模型上表现出广泛的适用性，显著提高了网格质量，相比顺序部分预测加速了生成速度，并支持增量编辑，所有这些都不需要重新训练模型。

### 结论

Mesh RAG成功解决了自回归网格生成中的质量-速度权衡问题，通过引入基于检索的方法消除了严格的顺序依赖，使生成过程更加高效、可并行，并支持增量编辑，为3D网格生成提供了新的可能性。

### 翻译

3D网格是从工业设计、游戏到仿真和机器人等应用的关键组成部分。传统上，网格由艺术家手工制作，这是一个耗时且难以扩展的过程。为了自动化和加速这种资产创建，自回归模型已成为艺术网格生成的强大范式。然而，当前提高质量的方法通常依赖更大或更长的序列，导致生成时间更长，并且它们固有的顺序性质造成了质量与速度之间的严重权衡。这种顺序依赖也显著增加了增量编辑的复杂性。为了克服这些限制，我们提出了Mesh RAG，一种新型的、无需训练的即插即用框架，用于自回归网格生成模型。受语言模型的RAG启发，我们的方法通过利用点云分割、空间变换和点云配准来增强生成过程，检索、生成和整合网格组件。这种基于检索的方法将生成与其严格的顺序依赖性解耦，促进了高效和可并行化的推理。我们展示了Mesh RAG在各种基础自回归网格生成模型上的广泛适用性，表明它显著提高了网格质量，相比顺序部分预测加速了生成速度，并支持增量编辑，所有这些都无需重新训练模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自回归3D网格生成模型中的质量-效率权衡问题以及增量编辑困难的问题。这个问题很重要，因为3D网格是工业设计、游戏、模拟和机器人等应用的关键构建块，传统手工创建网格耗时且难以扩展，而现有自动方法在生成复杂网格时面临速度与质量的严重权衡，同时难以支持局部编辑功能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到语言模型中检索增强生成(RAG)方法的启发，将其应用于3D网格生成领域。他们采用分而治之策略，将复杂3D对象分解为独立部件，设计了一个无需重新训练的即插即用框架。方法借鉴了点云分割技术(P3-SAM模型)、变换检索技术(包括粗对齐和ICP精细对齐)，以及TRELLIS模型在图像到点云转换方面的应用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将自回归网格生成分解为独立部件生成和空间变换检索两个阶段，通过检索每个部件的空间信息实现并行生成和正确集成。整体流程包括：1)点云分割：使用P3-SAM将输入点云分割成不同部分；2)变换检索：通过边界框匹配粗对齐，再用ICP算法精细对齐；3)并行生成：将分割后的点云段批量输入自回归模型生成对应网格部分；4)增量编辑：对初始网格采样并与编辑后点云对齐，生成新部件并组合成最终网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出Mesh RAG检索增强框架，无需重新训练即可即插即用；2)通过点云分割和变换检索实现并行生成，提高质量和效率；3)首次实现自回归网格生成模型的增量编辑功能。相比之前工作，Mesh RAG打破了自回归模型的顺序生成限制，生成更清洁规则的拓扑结构，更适合实时应用和艺术家编辑，且无需重新训练即可与现有模型兼容。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Mesh RAG通过引入检索增强机制，解决了自回归3D网格生成中的质量-效率权衡问题，实现了并行生成和增量编辑，无需重新训练即可显著提升生成质量和效率。'}


### 论文摘要

3D meshes are a critical building block for applications ranging from industrial design and gaming to simulation and robotics. Traditionally, meshes are crafted manually by artists, a process that is time-intensive and difficult to scale. To automate and accelerate this asset creation, autoregressive models have emerged as a powerful paradigm for artistic mesh generation. However, current methods to enhance quality typically rely on larger models or longer sequences that result in longer generation time, and their inherent sequential nature imposes a severe quality-speed trade-off. This sequential dependency also significantly complicates incremental editing. To overcome these limitations, we propose Mesh RAG, a novel, training-free, plug-and-play framework for autoregressive mesh generation models. Inspired by RAG for language models, our approach augments the generation process by leveraging point cloud segmentation, spatial transformation, and point cloud registration to retrieve, generate, and integrate mesh components. This retrieval-based approach decouples generation from its strict sequential dependency, facilitating efficient and parallelizable inference. We demonstrate the wide applicability of Mesh RAG across various foundational autoregressive mesh generation models, showing it significantly enhances mesh quality, accelerates generation speed compared to sequential part prediction, and enables incremental editing, all without model retraining.

---

## 3. RL-AD-Net: Reinforcement Learning Guided Adaptive Displacement in Latent Space for Refined Point Cloud Completion

**论文链接:** [http://arxiv.org/abs/2511.17054v1](http://arxiv.org/abs/2511.17054v1)

**作者:** Bhanu Pratap Paregi, Vaibhav Kumar

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为RL-AD-Net的强化学习细化框架，用于解决点云补全模型中的局部几何不一致性问题。该框架在预训练点自编码器的潜在空间中操作，通过强化学习代理调整全局特征向量来提高几何保真度，并结合轻量级非参数PointNN选择器确保鲁棒性。

### 背景

现有的点云补全模型（包括基于Transformer、基于去噪以及其他最先进的方法）可以从部分输入生成全局合理的形状，但常常存在局部几何不一致的问题。此外，基线补全网络在其训练风格的裁剪下表现合理，但在随机裁剪场景下表现不佳。

### 目的

开发一种能够改善点云补全模型局部几何一致性的方法，特别是在随机裁剪场景下提高补全质量，同时保持框架的轻量级、模块化和模型无关性。

### 方法

提出RL-AD-Net框架，在预训练点自编码器潜在空间中操作；自编码器将补全结果编码为全局特征向量(GFVs)；通过强化学习代理选择性地调整GFVs；使用PointNN选择器评估并保留几何一致性更好的结果；当真实数据可用时使用Chamfer距离和几何一致性指标指导细化；按类别单独训练以应对强化学习的无监督和动态特性。

### 主要发现

基线补全网络在训练风格裁剪下表现合理但在随机裁剪场景下表现不佳；RL-AD-Net在两种设置下都能持续改进性能；该方法是轻量级、模块化和模型无关的，适用于各种补全网络而无需重新训练。

### 结论

RL-AD-Net框架有效地解决了点云补全中的局部几何不一致性问题，通过强化学习在潜在空间中进行细化，结合选择机制确保最佳结果，适用于多种补全网络且无需重新训练。

### 翻译

近期的点云补全模型，包括基于Transformer的、基于去噪的以及其他最先进的方法，可以从部分输入生成全局合理的形状，但常常留下局部几何不一致性。我们提出了RL-AD-Net，一个在预训练点自编码器的潜在空间中操作的强化学习(RL)细化框架。该自编码器将补全结果编码为紧凑的全局特征向量(GFVs)，这些GFVs被强化学习代理选择性地调整以提高几何保真度。为确保鲁棒性，一个轻量级的非参数PointNN选择器评估原始补全和RL细化输出的几何一致性，保留更好的重建结果。当真实数据可用时，Chamfer距离和几何一致性指标都指导细化。训练按类别单独执行，因为强化学习的无监督和动态特性使得跨高度多样化类别的收敛具有挑战性。尽管如此，该框架可以在未来工作中扩展到多类别细化。在ShapeNetCore-2048上的实验表明，虽然基线补全网络在其训练风格的裁剪下表现合理，但在随机裁剪场景下表现不佳。相比之下，RL-AD-Net在这两种设置下都能持续改进，突显了强化学习引导的集成细化的有效性。该方法轻量级、模块化且模型无关，使其适用于各种补全网络而无需重新训练。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云补全模型中存在的局部几何不一致问题。这个问题很重要，因为准确的点云补全是机器人、自动驾驶、AR/VR内容创建和数字遗产保护等应用的基础能力，而现实世界的扫描数据经常存在遮挡、传感器噪声和有限视角等问题，局部几何不一致会影响精确任务（如抓取或碰撞推断）的可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有补全方法能生成全局连贯的形状但存在局部瑕疵，因此提出轻量级后处理解决方案。他们借鉴了RL-GAN-Net在潜在空间使用强化学习的思想，但改为在轻量级点云自编码器的嵌入空间中操作。作者还借鉴了PointNet架构设计自编码器，使用TD3强化学习算法，并采用PointNN评估几何一致性。此外，作者采用类别特定设计，为每个对象类别单独训练自编码器和强化学习智能体，以解决不同类别几何分布冲突的问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在预训练点云自编码器的潜在空间中，通过强化学习智能体对补全结果进行自适应调整，以提高几何保真度。整体流程包括：1)使用任何预训练点云补全网络生成初始补全；2)用特定类别自编码器将补全结果编码为128维全局特征向量；3)训练特定类别强化学习智能体预测调整向量；4)用冻结的自编码器解码器将调整后的特征向量解码为精炼点云；5)使用PointNN评估基线和精炼输出的几何一致性，选择更好的结果作为最终输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出类别特定潜在空间强化学习精炼方法；2)证明在128维GFV空间操作可实现高效有效的强化学习训练；3)提出几何感知选择性精炼策略，确保精炼不会降低性能；4)采用类别特定训练策略。相比之前工作，本方法不与生成对抗模型耦合，而是仅在自编码器潜在空间操作；不是确定性过滤或回归校正，而是数据驱动的实例特定校正；采用类别特定设计而非联合训练，以解决不同类别几何分布冲突问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出RL-AD-Net，一种基于强化学习的后处理框架，通过在自编码器潜在空间中进行自适应调整，显著提高了点云补全的几何一致性，同时保持全局结构完整性，且无需重新训练基础模型。'}


### 论文摘要

Recent point cloud completion models, including transformer-based, denoising-based, and other state-of-the-art approaches, generate globally plausible shapes from partial inputs but often leave local geometric inconsistencies. We propose RL-AD-Net, a reinforcement learning (RL) refinement framework that operates in the latent space of a pretrained point autoencoder. The autoencoder encodes completions into compact global feature vectors (GFVs), which are selectively adjusted by an RL agent to improve geometric fidelity. To ensure robustness, a lightweight non-parametric PointNN selector evaluates the geometric consistency of both the original completion and the RL-refined output, retaining the better reconstruction. When ground truth is available, both Chamfer Distance and geometric consistency metrics guide refinement. Training is performed separately per category, since the unsupervised and dynamic nature of RL makes convergence across highly diverse categories challenging. Nevertheless, the framework can be extended to multi-category refinement in future work. Experiments on ShapeNetCore-2048 demonstrate that while baseline completion networks perform reasonable under their training-style cropping, they struggle in random cropping scenarios. In contrast, RL-AD-Net consistently delivers improvements across both settings, highlighting the effectiveness of RL-guided ensemble refinement. The approach is lightweight, modular, and model-agnostic, making it applicable to a wide range of completion networks without requiring retraining.

---

## 4. The MeerKAT Fornax Survey VI. The collapse of the galaxy HI Mass Function in Fornax

**论文链接:** [http://arxiv.org/abs/2511.15795v2](http://arxiv.org/abs/2511.15795v2)

**作者:** D. Kleiner, P. Serra, A. Loni, S. H. A. Rajohnson, F. M. Maccagni, W. J. G. de Blok, P. Kamphuis, R. C. Kraan-Korteweg, M. A. W. Verheijen

**发布时间:** 2025-11-19

**备注:** Accepted in Astronomy & Astrophysics. 12 pages, 6 figures

### GPT解析

### 总结

本研究测量了本地星系群外最深的一次氢原子质量函数，发现了在低质量端HIMF的崩塌现象，这可能是由于低质量星系中氢的迅速去除造成的。

### 背景

研究使用MeerKAT Fornax Survey的数据，覆盖了一个4×4平方度的区域，对应约Rvir。观测检测到了35个星系和44个没有光学对应物的氢云。

### 目的

测量深氢原子质量函数，探索低质量星系的氢分布特征，特别是在低质量端HIMF的行为。

### 方法

使用MeerKAT望远镜进行HI观测，结合Fornax Deep Survey的深光学图像，通过信噪比分析和Rauzy统计量验证目录完整性，并使用修正的最大似然方法拟合Schechter函数。

### 主要发现

1) 在log(MHI/Msun) = 7处检测到HI检测星系的数量密度急剧下降，表明在6 < log(MHI/Msun) < 7之间明显缺少星系；2) 低质量斜率α = -1.31 ± 0.13与场域中的斜率匹配；3) 在log(MHI/Msun) = 7以下，HIMF与Schechter函数存在明显偏离；4) 为了使低质量端HIMF遵循幂律定律，需要比观察到的数量多约六倍的星系。

### 结论

Fornax星系团的HIMF在低质量端呈现崩塌现象，这种崩塌可能是由低质量星系中氢的迅速去除造成的，这为理解环境对星系演化的影响提供了重要线索。

### 翻译

我们展示了迄今为止在本地星系群外测量的最深氢原子质量函数。观测是MeerKAT Fornax调查的一部分，覆盖了一个4×4平方度的区域，对应约Rvir。对于50 km/s宽的点源，3σ检测限为log(MHI/Msun) = 5.7。我们在35个星系和44个没有光学对应物的云中检测到HI。使用来自Fornax Deep Survey的深光学图像，我们表明这些云是一个独特的种群，与最暗的HI检测星系之间存在四个星等差距。大多数(44个中的33个)云与星系团中HI最多的两个星系——NGC 1365和NGC 1427A相关联，尽管云对总MHI预算的贡献可忽略不计。通过对HI检测进行SNR分析和计算Rauzy统计量，我们证明我们的目录在log(MHI/Msun) = 6以下完整，因此能够探测到此水平的HIMF。我们在log(MHI/Msun) = 7处发现HI检测星系的数量密度急剧下降，表明在6 < log(MHI/Msun) < 7之间明显缺少星系。我们使用修正的最大似然方法将Schechter函数拟合到log(MHI/Msun) > 7的范围，即HIMF遵循幂律定律的范围。测得的低质量斜率为α = -1.31 ± 0.13，特征膝盖质量为log(M*/Msun) = 10.52 ± 1.89。低质量斜率与场域中的斜率匹配，而膝盖质量由单个星系定义且不受约束。在log(MHI/Msun) = 7以下，与Schechter函数存在明显偏离，我们报告了HIMF崩塌的第一个稳健测量。为了使log(MHI/Msun) = 7以下的HIMF遵循幂律定律，需要数十个星系——比观察到的数量高约六倍。Fornax HIMF的崩塌可能是由于低质量星系中HI的迅速去除造成的。


### 论文摘要

We present the deepest HI mass Function (HIMF) ever measured, outside the Local Group. The observations are part of the MeerKAT Fornax Survey and cover a 4 x 4 deg^2 field, corresponding to ~ Rvir. The 3$σ$ detection limit is log(MHI/Msun) = 5.7 for a 50 km/s-wide point source. We detect HI in 35 galaxies and 44 clouds with no optical counterparts. Using deep optical images from the Fornax Deep Survey, we show that the clouds are a distinct population, separated by a four magnitude gap from the faintest HI-detected galaxies. The majority (33 out of 44) of the clouds are associated with the two galaxies with the most HI in the cluster -- NGC 1365 and NGC 1427A, although the clouds contribute a negligible amount to the total MHI budget. By performing a SNR analysis and computing the Rauzy statistic on the HI detections, we demonstrate that our catalogue is complete down log(MHI/Msun) = 6, and we are therefore able to probe the HIMF down to this level. We find an abrupt drop of the number density of HI-detected galaxies at log(MHI/Msun) = 7, signifying a clear absence of galaxies between 6 < log(MHI/Msun) < 7. We use the modified maximum likelihood method to fit a Schechter function down to log(MHI/Msun) > 7, the range where the HIMF follows a power-law. The measured low-mass slope is $α$ = -1.31 $\pm$ 0.13, with a characteristic knee mass of log(M*/Msun) = 10.52 $\pm$ 1.89. The low-mass slope matches the slope in the field, while the knee is defined by a single galaxy and is unconstrained. Below log(MHI/Msun) = 7, there is a sharp departure from a Schechter function, and we report the first robust measurement of the collapse of a HIMF. For the HIMF below log(MHI/Msun) = 7 to follow a power-law, tens of galaxies are needed -- a factor ~ six higher than what is observed. The collapse of the Fornax HIMF is likely due to the rapid removal of HI from low-mass galaxies.

---

## 5. Joint Design of Protein Surface and Structure Using a Diffusion Bridge Model

**论文链接:** [http://arxiv.org/abs/2511.16675v1](http://arxiv.org/abs/2511.16675v1)

**作者:** Guanlue Li, Xufeng Zhao, Fang Wu, Sören Laue

**发布时间:** 2025-11-08

**备注:** 21 pages, 4 figures

### GPT解析

### 总结

PepBridge是一种创新的蛋白质设计框架，通过整合受体表面几何形状和生化特性，实现了蛋白质表面和结构的联合设计，解决了计算蛋白质设计中的关键挑战。

### 背景

蛋白质-蛋白质相互作用由蛋白质界面的表面互补性和疏水相互作用控制。然而，设计多样且物理上真实的蛋白质结构和表面，使其能够精确互补目标受体，仍然是计算蛋白质设计中的一个重大挑战。

### 目的

引入PepBridge框架，用于蛋白质表面和结构的联合设计，能够无缝整合受体表面几何形状和生化特性。

### 方法

PepBridge从表示为3D点云的受体表面开始，通过多步过程生成完整的蛋白质结构。首先使用去噪扩散桥模型将受体表面映射到配体表面，然后通过多模型扩散模型预测相应结构，同时利用Shape-Frame Matching Networks确保表面几何形状和主干架构之间的对齐。

### 主要发现

在各种蛋白质设计场景中的广泛验证证明了PepBridge在生成结构上可行的蛋白质方面的有效性。

### 结论

PepBridge代表了自上而下蛋白质结构联合设计的重要进展。

### 翻译

蛋白质-蛋白质相互作用由蛋白质界面的表面互补性和疏水相互作用控制。然而，设计多样且物理上真实的蛋白质结构和表面，使其能够精确互补目标受体，仍然是计算蛋白质设计中的一个重大挑战。在这项工作中，我们引入了PepBridge，这是一种用于蛋白质表面和结构联合设计的新框架，能够无缝整合受体表面几何形状和生化特性。从表示为3D点云的受体表面开始，PepBridge通过多步过程生成完整的蛋白质结构。首先，它使用去噪扩散桥模型将受体表面映射到配体表面。接下来，多模型扩散模型预测相应的结构，同时Shape-Frame Matching Networks确保表面几何形状和主干架构之间的对齐。这种集成方法促进了表面互补性、构象稳定性和化学可行性。在各种蛋白质设计场景中的广泛验证证明了PepBridge在生成结构上可行的蛋白质方面的有效性，代表了自上而下蛋白质结构联合设计的重要进展。


### 论文摘要

Protein-protein interactions (PPIs) are governed by surface complementarity and hydrophobic interactions at protein interfaces. However, designing diverse and physically realistic protein structure and surfaces that precisely complement target receptors remains a significant challenge in computational protein design. In this work, we introduce PepBridge, a novel framework for the joint design of protein surface and structure that seamlessly integrates receptor surface geometry and biochemical properties. Starting with a receptor surface represented as a 3D point cloud, PepBridge generates complete protein structures through a multi-step process. First, it employs denoising diffusion bridge models (DDBMs) to map receptor surfaces to ligand surfaces. Next, a multi-model diffusion model predicts the corresponding structure, while Shape-Frame Matching Networks ensure alignment between surface geometry and backbone architecture. This integrated approach facilitates surface complementarity, conformational stability, and chemical feasibility. Extensive validation across diverse protein design scenarios demonstrates PepBridge's efficacy in generating structurally viable proteins, representing a significant advancement in the joint design of top-down protein structure.

---

## 6. Automobile demand forecasting: Spatiotemporal and hierarchical modeling, life cycle dynamics, and user-generated online information

**论文链接:** [http://arxiv.org/abs/2511.17275v1](http://arxiv.org/abs/2511.17275v1)

**作者:** Tom Nahrendorf, Stefan Minner, Helfried Binder, Richard Zinck

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究解决了高端汽车制造商在多产品、多市场、多层级架构下的月度汽车需求预测挑战，结合了点预测和概率预测方法，使用LightGBM模型集合、分位数回归和混合整数线性规划协调方法。

### 背景

高端汽车制造商面临日益复杂的预测挑战，包括高产品多样性、变体级别数据稀疏以及市场动态波动。

### 目的

研究多产品、多市场、多层级架构下的月度汽车需求预测问题，基于德国高端制造商的数据进行分析。

### 方法

结合战略和运营规划层级的点预测和概率预测，使用LightGBM模型集合与共享训练集，应用分位数回归和混合整数线性规划协调方法。

### 主要发现

时空依赖性和舍入偏差显著影响预测准确性，整数预测对运营可行性很重要；短期需求是反应性的，受生命周期成熟度等因素影响；中期需求反映预期驱动因素；在线行为数据在细粒度级别显著提高了预测准确性。

### 结论

所提出的方法有效解决了高端汽车行业面临的复杂预测挑战，特别是在处理产品多样性和数据稀疏问题上表现突出。

### 翻译

高端汽车制造商由于产品多样性高、变体级别数据稀疏以及市场动态波动，面临着日益复杂的预测挑战。本研究使用德国一家高端制造商的数据，解决了多产品、多市场、多层级架构下的月度汽车需求预测问题。该方法结合了战略和运营规划层级的点预测和概率预测，利用具有共享训练集的LightGBM模型集合、分位数回归和混合整数线性规划协调方法。研究结果表明，时空依赖性和舍入偏差显著影响预测准确性，突显了整数预测对运营可行性的重要性。Shapley分析显示，短期需求是反应性的，受生命周期成熟度、自回归动量和运营信号影响，而中期需求则反映在线参与度、规划目标和竞争指标等预期驱动因素，在线行为数据在细粒度级别显著提高了准确性。


### 论文摘要

Premium automotive manufacturers face increasingly complex forecasting challenges due to high product variety, sparse variant-level data, and volatile market dynamics. This study addresses monthly automobile demand forecasting across a multi-product, multi-market, and multi-level hierarchy using data from a German premium manufacturer. The methodology combines point and probabilistic forecasts across strategic and operational planning levels, leveraging ensembles of LightGBM models with pooled training sets, quantile regression, and a mixed-integer linear programming reconciliation approach. Results highlight that spatiotemporal dependencies, as well as rounding bias, significantly affect forecast accuracy, underscoring the importance of integer forecasts for operational feasibility. Shapley analysis shows that short-term demand is reactive, shaped by life cycle maturity, autoregressive momentum, and operational signals, whereas medium-term demand reflects anticipatory drivers such as online engagement, planning targets, and competitive indicators, with online behavioral data considerably improving accuracy at disaggregated levels.

---

## 7. Flow-Guided Implicit Neural Representation for Motion-Aware Dynamic MRI Reconstruction

**论文链接:** [http://arxiv.org/abs/2511.16948v1](http://arxiv.org/abs/2511.16948v1)

**作者:** Baoqing Li, Yuanyuan Liu, Congcong Liu, Qingyong Zhu, Jing Cheng, Yihang Zhou, Hao Chen, Zhuo-Xu Cui, Dong Liang

**发布时间:** 2025-11-21

**备注:** 10 pages, 7 figures

### GPT解析

### 总结

本文提出了一种新颖的隐式神经表示(INR)框架，用于动态磁共振成像(dMRI)的重建，能够同时恢复时间连贯的图像和运动场，无需预先估计光流。

### 背景

动态磁共振成像(dMRI)能够捕捉时间分辨的解剖结构，但常面临采样有限和运动引起的伪影挑战。传统运动补偿重建方法依赖于预估计的光流，在欠采样情况下不准确，会降低重建质量。

### 目的

开发一种能够同时重建动态图像序列和运动场的方法，克服传统方法在欠采样情况下的局限性，提高dMRI的重建质量。

### 方法

提出一种隐式神经表示(INR)框架，使用一个INR参数化时空图像内容，另一个INR表示光流。两者通过光流方程耦合，作为物理启发的正则化，同时结合数据一致性损失强制与k空间测量保持一致，实现联合优化。

### 主要发现

在动态心脏MRI数据集上的实验表明，所提出的方法优于最先进的运动补偿和深度学习方法，实现了更好的重建质量、准确的运动估计和改进的时间保真度。

### 结论

具有流约束的隐式联合建模在推进dMRI重建方面具有潜力，能够有效解决传统方法在欠采样情况下的局限性。

### 翻译

动态磁共振成像(dMRI)能够捕捉时间分辨的解剖结构，但常面临采样有限和运动引起的伪影挑战。传统运动补偿重建通常依赖于预估计的光流，但在欠采样情况下不准确，会降低重建质量。在本文中，我们提出了一种新颖的隐式神经表示(INR)框架，联合建模动态图像序列及其底层运动场。具体来说，一个INR用于参数化时空图像内容，另一个INR表示光流。两者通过光流方程耦合，作为物理启发的正则化，同时还有数据一致性损失，强制与k空间测量保持一致。这种联合优化能够同时恢复时间连贯的图像和运动场，无需先前的流估计。在动态心脏MRI数据集上的实验表明，所提出的方法优于最先进的运动补偿和深度学习方法，实现了更好的重建质量、准确的运动估计和改进的时间保真度。这些结果强调了具有流约束的隐式联合建模在推进dMRI重建方面的潜力。


### 论文摘要

Dynamic magnetic resonance imaging (dMRI) captures temporally-resolved anatomy but is often challenged by limited sampling and motion-induced artifacts. Conventional motion-compensated reconstructions typically rely on pre-estimated optical flow, which is inaccurate under undersampling and degrades reconstruction quality. In this work, we propose a novel implicit neural representation (INR) framework that jointly models both the dynamic image sequence and its underlying motion field. Specifically, one INR is employed to parameterize the spatiotemporal image content, while another INR represents the optical flow. The two are coupled via the optical flow equation, which serves as a physics-inspired regularization, in addition to a data consistency loss that enforces agreement with k-space measurements. This joint optimization enables simultaneous recovery of temporally coherent images and motion fields without requiring prior flow estimation. Experiments on dynamic cardiac MRI datasets demonstrate that the proposed method outperforms state-of-the-art motion-compensated and deep learning approaches, achieving superior reconstruction quality, accurate motion estimation, and improved temporal fidelity. These results highlight the potential of implicit joint modeling with flow-regularized constraints for advancing dMRI reconstruction.

---

## 8. GeoPTH: A Lightweight Approach to Category-Based Trajectory Retrieval via Geometric Prototype Trajectory Hashing

**论文链接:** [http://arxiv.org/abs/2511.16258v2](http://arxiv.org/abs/2511.16258v2)

**作者:** Yang Xu, Zuliang Yang, Kai Ming Ting

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了几何原型轨迹哈希(GeoPTH)，一种新型轻量级非学习框架，用于高效的基于类别的轨迹相似性检索。

### 背景

轨迹相似性检索是时空数据挖掘的重要组成部分，但现有方法存在局限性：传统方法计算成本高，基于学习的方法训练成本高且可能不稳定。

### 目的

解决现有方法的局限性，提出一种轻量级、非学习框架，实现高效的基于类别的轨迹检索。

### 方法

GeoPTH通过使用代表性轨迹原型（保留几何特征的小点集）作为锚点构建数据依赖的哈希函数，利用鲁棒性的Hausdorff度量将新轨迹映射到最近的原型。

### 主要发现

GeoPTH的检索精度与传统方法和最先进的学习方法具有竞争力，显著优于通过简单二值化学习嵌入生成的二进制码，在效率方面始终优于所有竞争方法。

### 结论

轻量级、以原型为中心的方法提供了一种实用且强大的替代方案，实现了卓越的检索性能和计算效率。

### 翻译

轨迹相似性检索是时空数据挖掘的重要组成部分，然而，现有方法存在以下局限性：传统度量计算成本高，而基于学习的方法则存在大量训练成本和潜在的不稳定性。本文通过提出几何原型轨迹哈希(GeoPTH)来解决这些问题，这是一种新颖的轻量级非学习框架，用于高效的基于类别的轨迹检索。GeoPTH使用代表性轨迹原型（即保留几何特征的小点集）作为锚点来构建数据依赖的哈希函数。哈希过程高效，涉及通过鲁棒性的Hausdorff度量将新轨迹映射到其最近的原型。大量实验表明，GeoPTH的检索精度与传统方法和最先进的学习方法高度竞争，并且显著优于通过简单二值化学习嵌入生成的二进制码。关键的是，GeoPTH在效率方面始终优于所有竞争方法。我们的工作表明，轻量级、以原型为中心的方法提供了一种实用且强大的替代方案，实现了卓越的检索性能和计算效率。


### 论文摘要

Trajectory similarity retrieval is an important part of spatiotemporal data mining, however, existing methods have the following limitations: traditional metrics are computationally expensive, while learning-based methods suffer from substantial training costs and potential instability. This paper addresses these problems by proposing Geometric Prototype Trajectory Hashing (GeoPTH), a novel, lightweight, and non-learning framework for efficient category-based trajectory retrieval. GeoPTH constructs data-dependent hash functions by using representative trajectory prototypes, i.e., small point sets preserving geometric characteristics, as anchors. The hashing process is efficient, which involves mapping a new trajectory to its closest prototype via a robust, Hausdorff metric. Extensive experiments show that GeoPTH's retrieval accuracy is highly competitive with both traditional metrics and state-of-the-art learning methods, and it significantly outperforms binary codes generated through simple binarization of the learned embeddings. Critically, GeoPTH consistently outperforms all competitors in terms of efficiency. Our work demonstrates that a lightweight, prototype-centric approach offers a practical and powerful alternative, achieving an exceptional retrieval performance and computational efficiency.

---

## 9. FisheyeGaussianLift: BEV Feature Lifting for Surround-View Fisheye Camera Perception

**论文链接:** [http://arxiv.org/abs/2511.17210v1](http://arxiv.org/abs/2511.17210v1)

**作者:** Shubham Sonarghare, Prasad Deshpande, Ciaran Hogan, Deepika-Rani Kaliappan-Mahalingam, Ganesh Sistu

**发布时间:** 2025-11-21

**备注:** 8 pages, 3 figures, published in IMVIP 2025 conference

### GPT解析

### 总结

本文提出了一种畸变感知的BEV分割框架，直接处理多相机高分辨率鱼眼图像，通过高斯参数化和可微分溅射技术实现准确的语义分割，无需传统去畸变处理。

### 背景

从鱼眼图像进行准确的BEV语义分割具有挑战性，原因是极端非线性畸变、遮挡和广角投影固有的深度模糊性。

### 目的

开发一个能够直接处理鱼眼图像并准确生成BEV语义分割的框架，解决传统方法中的畸变问题。

### 方法

使用校准的几何反投影和逐像素深度分布估计，通过高斯参数化将图像像素提升到3D空间，预测空间均值和各向异性协方差建模几何不确定性，然后通过可微分溅射将3D高斯融合到BEV表示中。

### 主要发现

在复杂停车和城市驾驶场景中，该框架在严重鱼眼畸变和多样化环境条件下取得了优异性能，可行驶区域的IoU得分为87.75%，车辆的IoU得分为57.26%。

### 结论

该框架能够有效处理鱼眼图像的BEV语义分割问题，产生连续的、不确定性感知的语义地图，无需传统的去畸变或透视校正步骤。

### 翻译

从鱼眼图像进行准确的BEV语义分割仍然具有挑战性，这是由于广角投影固有的极端非线性畸变、遮挡和深度模糊性。我们提出了一个畸变感知的BEV分割框架，直接处理多相机高分辨率鱼眼图像，利用校准的几何反投影和逐像素深度分布估计。通过高斯参数化将每个图像像素提升到3D空间，预测空间均值和各向异性协方差以明确建模几何不确定性。通过可微分溅射将投影的3D高斯融合到BEV表示中，产生连续的、不确定性感知的语义地图，无需去畸变或透视校正。大量实验表明，在复杂的停车和城市驾驶场景中具有强大的分割性能，在严重的鱼眼畸变和多样化环境条件下，可行驶区域的IoU得分为87.75%，车辆的IoU得分为57.26%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从鱼眼相机图像中准确提取鸟瞰图特征的问题。这个问题很重要，因为鱼眼相机在自动驾驶系统中被广泛使用（提供360度视野），但它们产生的极端非线性畸变使得传统方法难以处理。BEV表示对自动驾驶和停车系统至关重要，因为它能提供统一的空间视角，帮助车辆理解周围环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有BEV框架（如LSS、BEVDet）在处理鱼眼相机时的局限性。他们借鉴了Gaussian Splatting用于3D场景表示的思想，以及GaussianLSS用于BEV构建的方法。作者将这些技术扩展到鱼眼相机模型，通过结合3DGUT和Fisheye-GS处理非线性畸变的方法，设计了一个新的管道，直接处理鱼眼图像而不需要先进行畸变校正。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用高斯参数化将图像像素提升到3D空间，预测空间均值和各向异性协方差来建模几何不确定性，然后通过可微的高斯投影将3D高斯融合到BEV表示中。整体流程包括：1)鱼眼特征提取，使用EfficientNet-Lite处理图像；2)高斯深度投影，将像素反向投影到3D空间并参数化为高斯分布；3)BEV编码，聚合多视图特征到统一网格；4)多类分割头，预测语义地图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)畸变感知的BEV分割框架，直接处理鱼眼图像；2)高斯参数化明确建模几何不确定性；3)可微高斯投影产生连续语义地图；4)基于查找表的鱼眼反向投影。相比之前工作，该方法不需要去畸变或透视校正，直接处理鱼眼畸变，使用连续表示而非离散网格，并明确建模深度不确定性，特别适用于鱼眼相机和停车场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FisheyeGaussianLift通过结合高斯参数化和可微投影技术，首次实现了直接从鱼眼图像中生成高质量、不确定感知的BEV语义分割，解决了自动驾驶中鱼眼相机感知的关键挑战。'}


### 论文摘要

Accurate BEV semantic segmentation from fisheye imagery remains challenging due to extreme non-linear distortion, occlusion, and depth ambiguity inherent to wide-angle projections. We present a distortion-aware BEV segmentation framework that directly processes multi-camera high-resolution fisheye images,utilizing calibrated geometric unprojection and per-pixel depth distribution estimation. Each image pixel is lifted into 3D space via Gaussian parameterization, predicting spatial means and anisotropic covariances to explicitly model geometric uncertainty. The projected 3D Gaussians are fused into a BEV representation via differentiable splatting, producing continuous, uncertainty-aware semantic maps without requiring undistortion or perspective rectification. Extensive experiments demonstrate strong segmentation performance on complex parking and urban driving scenarios, achieving IoU scores of 87.75% for drivable regions and 57.26% for vehicles under severe fisheye distortion and diverse environmental conditions.

---

## 10. Edge-ANN: Storage-Efficient Edge-Based Remote Sensing Feature Retrieval

**论文链接:** [http://arxiv.org/abs/2511.16938v1](http://arxiv.org/abs/2511.16938v1)

**作者:** Xianwei Lv, Debin Tang, Zhecheng Shi, Wang Wang, Yujiao Zheng, Xiatian Zhu

**发布时间:** 2025-11-21

### GPT解析

### 总结

Edge-ANN是一个创新的ANN框架，专为边缘设备的存储效率设计，通过利用'锚点'对而非传统的高维超平面来定义空间分区，显著减少了存储需求，同时保持了较高的检索性能。

### 背景

高性能近似最近邻(ANN)搜索在遥感边缘设备上满足实时约束仍然是一个关键挑战，这些设备如微型卫星和无人机本质上是融合系统，主要受限于主存(RAM)和辅助存储(磁盘)的严格限制。

### 目的

提出一个针对存储效率优化的创新ANN框架，解决边缘设备上的存储限制问题，实现大规模、高性能、实时的遥感特征检索。

### 方法

提出了Edge-ANN框架，利用现有的数据项对（称为'锚点'）来隐式定义空间分区，开发了新的二元锚点优化算法确保分区既平衡又有效，这种架构转变消除了空间复杂度的维度依赖性。

### 主要发现

在三个多源数据集上的实验表明，在模拟的边缘环境和双重存储约束下，Edge-ANN实现了30-40%的辅助存储减少，以3-5%的检索精度轻微下降为代价，整体检索性能超过了其他主流方法。

### 结论

这些结果确立了Edge-ANN作为最先进的解决方案，使具有极有限存储的边缘设备能够实现大规模、高性能、实时的遥感特征检索。

### 翻译

在遥感边缘设备上满足高性能近似最近邻(ANN)搜索的实时约束仍然是一个关键挑战，这些设备本质上像微型卫星和无人机这样的融合系统，主要由于主存(RAM)和辅助存储(磁盘)的严格限制。为应对这一挑战，我们提出了Edge-ANN，一个专为存储效率设计的创新ANN框架。Edge-ANN的核心创新在于它偏离了存储高维超平面的传统基于树的方法。相反，它利用现有的数据项对，称为'锚点'，来隐式定义空间分区。为确保这些分区既平衡又有效，我们开发了一种新的二元锚点优化算法。这种架构转变消除了空间复杂度的维度依赖性。在MillionAID、高分辨率城市复杂数据集和GlobalUrbanNet数据集三个多源数据集上的严格实验表明，在具有双重存储约束的模拟边缘环境下，Edge-ANN与基线相比实现了30-40%的辅助存储减少，代价是检索精度轻微下降3-5%。此外，在这些受限场景下，其整体检索性能超过了其他主流方法。总之，这些结果确立了Edge-ANN作为最先进的解决方案，使具有极有限存储的边缘设备能够实现大规模、高性能、实时的遥感特征检索。Edge-ANN的代码可在https://github.com/huaijiao666/Edge-ANN获取。


### 论文摘要

Meeting real-time constraints for high-performance Approximate Nearest Neighbor (ANN) search remains a critical challenge in remote sensing edge devices, which are essentially fusion systems like micro-satellites and UAVs, largely due to stringent limitations in primary (RAM) and secondary (disk) storage. To address this challenge, we propose Edge-ANN, an innovative ANN framework specifically engineered for storage efficiency. The core innovation of Edge-ANN lies in its departure from traditional tree-based methods that store high-dimensional hyperplanes. Instead, it leverages pairs of existing data items, termed "anchors," to implicitly define spatial partitions. To ensure these partitions are both balanced and effective, we have developed a novel Binary Anchor Optimization algorithm.This architectural shift eliminates the dimension-dependence of the space complexity. Rigorous experiments on three multi-source datasets, MillionAID, High-resolution Urban Complex Dataset, and GlobalUrbanNet Dataset, demonstrate that under simulated edge environments with dual storage constraints, Edge-ANN achieves a 30-40% reduction in secondary storage compared to the baseline, at the cost of a minor 3-5% drop in retrieval accuracy. Furthermore, its overall retrieval performance surpasses that of other mainstream methods in these constrained scenarios. Collectively, these results establish Edge-ANN as a state-of-the-art solution for enabling large-scale, high-performance, real-time remote sensing feature retrieval on edge devices with exceptionally constrained storage. The codes of Edge-ANN are available at https://github.com/huaijiao666/Edge-ANN.

---

## 11. SPEAR-1: Scaling Beyond Robot Demonstrations via 3D Understanding

**论文链接:** [http://arxiv.org/abs/2511.17411v1](http://arxiv.org/abs/2511.17411v1)

**作者:** Nikolay Nikolov, Giuliano Albanese, Sombit Dey, Aleksandar Yanev, Luc Van Gool, Jan-Nico Zaech, Danda Pani Paudel

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文介绍了SPEAR-1，一个集成了3D感知与语言指导具身控制的机器人基础模型。通过为非机器人图像数据添加3D注释，作者训练了SPEAR-VLM，并在此基础上构建了SPEAR-1，该模型在较少机器人演示数据下达到或超越最先进模型的性能。

### 背景

机器人基础模型(RFMs)在机器人控制方面有很大潜力，但泛化能力有限。大多数RFMs基于微调互联网预训练的视觉语言模型(VLMs)构建，但这些VLMs缺乏3D空间推理能力，这是3D世界具身控制所必需的。

### 目的

解决RFMs在3D空间推理方面的局限性，通过增强VLM的3D理解能力，提高机器人在新环境、任务和形态上的泛化能力。

### 方法

为非机器人图像数据添加3D注释，训练SPEAR-VLM模型使其能从单张2D图像推断3D空间坐标；基于SPEAR-VLM构建SPEAR-1，在约4500万帧数据上训练，结合3D感知与语言指导的具身控制。

### 主要发现

SPEAR-1性能优于或匹敌π_0-FAST和π_{0.5}等最先进模型，同时使用的机器人演示数据少20倍。这种训练策略解锁了VLM的新能力，提高了具身控制的可靠性。

### 结论

通过将3D理解能力集成到VLM中，可显著提高机器人基础模型的泛化能力和性能，同时减少对机器人演示数据的依赖。作者公开了模型权重和3D注释数据集。

### 翻译

机器人基础模型(RFMs)作为通用端到端系统在机器人控制方面有很大前景。然而，它们在新环境、任务和形态上的泛化能力仍然有限。我们认为主要瓶颈在于它们的基础：大多数RFMs是通过微调互联网预训练的视觉语言模型(VLMs)构建的。然而，这些VLMs在2D图像语言任务上训练，缺乏3D世界具身控制 inherently 所需的3D空间推理能力。直接使用大规模机器人数据来弥合这一差距成本高昂且难以扩展。相反，我们提出通过为易于收集的非机器人图像数据添加3D注释，并增强预训练VLM的3D理解能力。遵循这一策略，我们训练了SPEAR-VLM，一个能够从单张2D图像推断3D空间中物体坐标的3D感知VLM。基于SPEAR-VLM，我们引入了主要贡献SPEAR-1：一个集成了基于3D感知与语言指导的具身控制的机器人基础模型。在约4500万帧来自24个开放形态数据集的数据上训练，SPEAR-1的性能优于或匹敌最先进的模型(如π_0-FAST和π_{0.5})，同时使用的机器人演示数据少20倍。这种精心设计的训练策略解锁了VLM的新能力，并因此提高了具身控制的可靠性，超越了仅使用机器人数据所能实现的效果。我们公开了模型权重和3D注释数据集。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人基础模型在跨新环境、任务和机器人形态方面的泛化能力有限的问题。这个问题很重要，因为收集大规模机器人演示数据成本高昂且难以扩展，而现有模型在未见环境中的零样本性能有限，机器人需要在各种不同环境中执行任务，需要更好的泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有视觉语言模型缺乏3D空间理解能力，然后提出通过引入显式3D意识到视觉语言骨干网络中，而不是仅仅依赖大规模机器人演示数据。方法设计分为两阶段：先开发SPEAR-VLM增强3D理解，再构建SPEAR-1实现实体控制。借鉴了PaliGemma VLM架构、MoGe深度估计器、π0架构和Flow Matching方法等现有工作，但进行了针对机器人控制的定制改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过引入显式3D理解能力增强视觉语言模型，减少对昂贵机器人演示数据的依赖，使用非机器人的2D图像数据添加3D注释来增强空间推理。整体流程分两阶段：第一阶段训练SPEAR-VLM，扩展预训练VLM并添加深度编码器，在非机器人图像数据上进行3D感知训练；第二阶段训练SPEAR-1，添加动作专家模块，在机器人演示数据上将视觉语言表示映射到电机动作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：SPEAR-VLM具有3D能力的VLM；SPEAR-1使用20倍更少机器人数据的机器人基础模型；通过非机器人3D注释数据替代机器人演示数据的能力。相比之前工作，SPEAR-1引入显式3D意识而非依赖隐式学习；在基础模型级别实现跨环境和机器人的端到端控制；在更少数据上达到相当或更好的性能；与需要推理的实时控制方法不同，SPEAR-1可直接部署。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了SPEAR-1，一种通过3D理解能力减少对机器人演示数据依赖的机器人基础模型，在多种机器人平台上实现了与使用20倍更多机器人数据的模型相当的性能，显著提升了机器人在未见环境中的泛化能力。'}


### 论文摘要

Robotic Foundation Models (RFMs) hold great promise as generalist, end-to-end systems for robot control. Yet their ability to generalize across new environments, tasks, and embodiments remains limited. We argue that a major bottleneck lies in their foundations: most RFMs are built by fine-tuning internet-pretrained Vision-Language Models (VLMs). However, these VLMs are trained on 2D image-language tasks and lack the 3D spatial reasoning inherently required for embodied control in the 3D world. Bridging this gap directly with large-scale robotic data is costly and difficult to scale. Instead, we propose to enrich easy-to-collect non-robotic image data with 3D annotations and enhance a pretrained VLM with 3D understanding capabilities. Following this strategy, we train SPEAR-VLM, a 3D-aware VLM that infers object coordinates in 3D space from a single 2D image. Building on SPEAR-VLM, we introduce our main contribution, $~\textbf{SPEAR-1}$: a robotic foundation model that integrates grounded 3D perception with language-instructed embodied control. Trained on $\sim$45M frames from 24 Open X-Embodiment datasets, SPEAR-1 outperforms or matches state-of-the-art models such as $π_0$-FAST and $π_{0.5}$, while it uses 20$\times$ fewer robot demonstrations. This carefully-engineered training strategy unlocks new VLM capabilities and as a consequence boosts the reliability of embodied control beyond what is achievable with only robotic data. We make our model weights and 3D-annotated datasets publicly available.

---

## 12. SuperQuadricOcc: Multi-Layer Gaussian Approximation of Superquadrics for Real-Time Self-Supervised Occupancy Estimation

**论文链接:** [http://arxiv.org/abs/2511.17361v1](http://arxiv.org/abs/2511.17361v1)

**作者:** Seamie Hayes, Reenu Mohandas, Tim Brophy, Alexandre Boulch, Ganesh Sistu, Ciaran Eising

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种基于超二次曲面的语义占用估计方法SuperQuadricOcc，通过减少基元数量显著降低内存需求并提高推理速度，同时保持较高的估计精度。

### 背景

语义占用估计对自动驾驶的全面场景理解至关重要，提供密集的空间和语义信息。高斯表示虽被广泛采用，但大量高斯基元会增加内存需求，不适合实时推理。超二次曲面可减少基元数量，但在自监督占用模型中实现困难，因为缺乏超二次曲面光栅化器来支持模型监督。

### 目的

开发一种能够实现实时推理且保持竞争力的占用估计模型，解决超二次曲面在自监督占用模型中的实现难题。

### 方法

提出SuperQuadricOcc方法，采用基于超二次曲面的场景表示。通过利用多层二十面体细分高斯近似来表示超二次曲面，从而在训练期间通过高斯光栅化实现模型监督。

### 主要发现

在Occ3D数据集上，SuperQuadricAchie相比之前的高斯方法实现了75%的内存占用减少、124%的推理速度提升和5.9%的mIoU改进，且不使用时间标签。超二次曲面将场景建模所需的基元数量减少了84%。研究还开发了快速超二次曲面体素化模块，便于与先前方法进行比较评估。

### 结论

SuperQuadricOcc是首个能够实现实时推理同时保持竞争力的占用模型，证明了超二次曲面在占用估计中的优势，代码将作为开源发布。

### 翻译

语义占用估计实现了自动驾驶的全面场景理解，提供了对感知和规划至关重要的密集空间和语义信息。虽然高斯表示已在自监督占用估计中被广泛采用，但大量高斯基元的部署会大幅增加内存需求，不适合实时推理。相比之下，超二次曲面由于其多样化的形状集，允许减少基元数量和降低内存需求。然而，由于缺乏超二次曲面光栅化器来支持模型监督，将其实现到自监督占用模型中并非易事。我们提出的方法SuperQuadricOcc采用基于超二次曲面的场景表示。通过利用超二次曲面的多层二十面体细分高斯近似，我们实现了训练期间的高斯光栅化监督。在Occ3D数据集上，SuperQuadricOcc相比之前的高斯方法实现了75%的内存占用减少、124%更快的推理速度和5.9%的mIoU提升，且不使用时间标签。据我们所知，这是首个能够实现实时推理同时保持竞争力的占用模型。超二次曲面的使用将场景建模所需的基元数量减少了84%。最后，我们开发的快速超二次曲面体素化模块便于与先前方法进行比较评估。代码将作为开源发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶场景中实时语义占用估计的问题，即在保持高精度的同时实现快速推理。这个问题很重要，因为自动驾驶系统需要全面理解周围环境，提供密集的空间和语义信息用于感知和规划。现有的高斯表示方法需要大量基本元素来表示场景，导致内存需求高，无法满足实时推理的需求，限制了自动驾驶系统的实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到高斯表示在自监督占用估计中的优势，但注意到其需要大量基本元素导致内存和计算负担。他们发现超二次曲面可以用更少的元素表示更复杂的形状，但缺乏将其集成到自监督模型中的方法。解决方案是通过多层高斯近似来表示超二次曲面，利用高斯光栅化进行训练监督。作者借鉴了GaussianFlowOcc的Transformer架构、PartGS和GaussianBlock的二十面体细分策略，以及QuadricFormer的超二次曲面应用思想，创造性地结合了这些方法的优势。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用超二次曲面作为场景表示的基本元素，通过多层高斯近似实现训练监督，并设计高效的体素化模块用于推理。整体流程：1)输入多摄像头图像；2)用ResNet-50提取图像特征；3)初始化超二次曲面特征和位置；4)通过Transformer优化特征；5)用MLP预测超二次曲面属性；6)训练时通过超二次曲面到高斯模块生成多层高斯近似并渲染进行监督；7)推理时通过超二次曲面体素化模块将占用概率聚合到体素网格生成最终预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次将超二次曲面应用于自监督语义占用估计；2)提出多层高斯近似方法实现训练监督；3)实现实时推理能力。相比不同：相比体素方法，内存减少94%；相比高斯方法(如GaussianFlowOcc)，基本元素减少84%，内存减少75%，推理速度提高124%(达21.5 FPS)，mIoU提高5.9%；相比依赖基础模型的方法(如GaussTR)，推理速度提高7066%，不依赖外部模型。SuperQuadricOcc在保持竞争力的同时大幅提高了效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SuperQuadricOcc通过创新的超二次曲面场景表示和多层高斯近似方法，首次实现了实时自监督语义占用估计，在大幅减少内存需求和加速推理的同时，提高了场景理解的准确性。'}


### 论文摘要

Semantic occupancy estimation enables comprehensive scene understanding for automated driving, providing dense spatial and semantic information essential for perception and planning. While Gaussian representations have been widely adopted in self-supervised occupancy estimation, the deployment of a large number of Gaussian primitives drastically increases memory requirements and is not suitable for real-time inference. In contrast, superquadrics permit reduced primitive count and lower memory requirements due to their diverse shape set. However, implementation into a self-supervised occupancy model is nontrivial due to the absence of a superquadric rasterizer to enable model supervision. Our proposed method, SuperQuadricOcc, employs a superquadric-based scene representation. By leveraging a multi-layer icosphere-tessellated Gaussian approximation of superquadrics, we enable Gaussian rasterization for supervision during training. On the Occ3D dataset, SuperQuadricOcc achieves a 75\% reduction in memory footprint, 124\% faster inference, and a 5.9\% improvement in mIoU compared to previous Gaussian-based methods, without the use of temporal labels. To our knowledge, this is the first occupancy model to enable real-time inference while maintaining competitive performance. The use of superquadrics reduces the number of primitives required for scene modeling by 84\% relative to Gaussian-based approaches. Finally, evaluation against prior methods is facilitated by our fast superquadric voxelization module. The code will be released as open source.

---

## 13. Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM

**论文链接:** [http://arxiv.org/abs/2511.17335v1](http://arxiv.org/abs/2511.17335v1)

**作者:** Chiori Hori, Yoshiki Masuyama, Siddarth Jain, Radu Corcodel, Devesh Jha, Diego Romeres, Jonathan Le Roux

**发布时间:** 2025-11-21

**备注:** Accepted to ASRU 2025

### GPT解析

### 总结

本文提出了一种基于长上下文Q-former和文本条件化方法的人机交互框架，用于提升机器人对人类动作的理解和动作规划能力。

### 背景

人机协作需要机器人理解人类动作及与环境的交互。当前方法主要基于多模态transformers从单一片段生成与机器人动作确认对齐的动作步骤，但忽视了长视频中的上下文信息。

### 目的

解决当前方法主要关注片段级处理而未利用长上下文信息的问题，提高人机协作中机器人动作确认和动作规划的性能。

### 方法

提出了一种包含完整视频左右上下文依赖关系的长上下文Q-former，以及一种文本条件化方法，将文本嵌入直接输入到LLM解码器中，以减轻Q-former中文本信息的高抽象性。

### 主要发现

在YouCook2语料库上的实验表明，确认生成的准确性是动作规划性能的主要因素；长上下文Q-former通过集成VideoLLaMA3提高了确认和动作规划的性能。

### 结论

通过引入长上下文信息和文本条件化方法，可以改善人机协作中机器人对人类动作的理解和响应能力。

### 翻译

人机协作朝向共同目标需要机器人理解人类动作以及与周围环境的交互。本文关注基于人机对话的人机交互(HRI)，该交互依赖于机器人动作确认和使用多模态场景理解的动作步骤生成。最先进的方法使用多模态transformers从显示由多个微步骤组成的任务的单一片段中生成与机器人动作确认对齐的机器人动作步骤。尽管朝向长时程任务的动作在整个视频中相互依赖，但当前方法主要关注片段级处理且不利用长上下文信息。本文提出了一种包含完整视频中左右上下文依赖关系的长上下文Q-former。此外，本文提出了一种文本条件化方法，将文本嵌入直接输入到LLM解码器中，以减轻Q-former中文本信息的高抽象性。使用YouCook2语料库的实验表明，确认生成的准确性是动作规划性能的主要因素。此外，我们证明了通过集成VideoLLaMA3，长上下文Q-former提高了确认和动作规划性能。


### 论文摘要

Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3.

---

## 14. MuM: Multi-View Masked Image Modeling for 3D Vision

**论文链接:** [http://arxiv.org/abs/2511.17309v1](http://arxiv.org/abs/2511.17309v1)

**作者:** David Nordström, Johan Edstedt, Fredrik Kahl, Georg Bökman

**发布时间:** 2025-11-21

### GPT解析

### 总结

提出了一种名为MuM的新方法，将掩码自编码扩展到多视图场景，通过统一屏蔽所有视图和使用带帧间注意力的轻量级解码器，实现了比CroCo更简单且更具可扩展性的解决方案，并在多个下游任务上表现优异。

### 背景

自监督学习在图像领域旨在从未标记数据中提取有意义的视觉表示，当扩展到大型数据集时已达到最先进性能，如DINOv3模型得到广泛应用。然而，大多数先前工作针对语义理解而非几何推理，CroCo是为3D理解定制的例外。

### 目的

沿着CroCo提出的路径，专注于学习为3D视觉定制的特征表示。

### 方法

将掩码自编码扩展到同一场景的任意多个视图，通过统一屏蔽所有视图并使用带有帧间注意力的轻量级解码器，使方法本质上比CroCo更简单且更具可扩展性。

### 主要发现

创建的模型MuM在前馈重建、密集图像匹配和相对位姿估计等多个下游任务上进行了广泛评估，结果表明它优于最先进的视觉编码器DINOv3和CroCo v2。

### 结论

MuM模型通过多视图掩码自编码方法在3D视觉理解任务上取得了优异性能，为3D视觉表示学习提供了更简单且可扩展的解决方案。

### 翻译

图像上的自监督学习旨在从未标记的数据中提取有意义的视觉表示。当扩展到大型数据集时，这种范式已达到最先进的性能， resulting trained models such as DINOv3 have seen widespread adoption. 然而，大多数先前工作都是针对语义理解而非几何推理。一个重要的例外是Cross-View Completion, CroCo，它是一种为3D理解定制的掩码自编码形式。在这项工作中，我们沿着CroCo提出的路径继续前进，专注于学习为3D视觉定制的特征。简而言之，我们将MAE扩展到同一场景的任意多个视图。通过统一屏蔽所有视图并使用带有帧间注意力的轻量级解码器，我们的方法本质上比CroCo更简单且更具可扩展性。我们在包括前馈重建、密集图像匹配和相对位姿估计在内的下游任务上对 resulting model, MuM 进行了广泛评估，发现它优于最先进的视觉编码器DINOv3和CroCo v2。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的是在3D视觉任务中学习几何特征的问题。现有自监督学习方法（如DINOv3）主要针对语义理解而非几何推理，而专门的3D方法（如CroCo）又存在数据采样脆弱、计算成本高等限制。这个问题很重要，因为3D视觉任务（如重建、匹配、姿态估计）需要模型理解几何关系，而不仅仅是语义内容，现有方法要么几何表现不佳，要么计算资源要求过高，限制了实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从MAE（掩码自编码器）和CroCo（跨视图完成）获得启发。注意到MAE在单视图上表现良好，而CroCo通过引入参考视图增强3D理解但存在局限性。作者思考如何结合两者的优点，避免缺点，最终决定将MAE扩展到多视图场景，不依赖参考视图，使用统一的掩码策略和轻量级解码器处理多个视图。这借鉴了MAE的掩码重建思想和CroCo的多视图处理，但避免了其复杂性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将掩码自编码扩展到多视图场景，通过同时处理多个视角的图像来学习几何特征，不依赖参考视图。整体流程：输入同一场景的2-24个图像→将每个图像分割成patch→对所有图像的patch进行75%均匀掩码→将可见patch通过ViT-L编码器处理→添加可学习的掩码token→使用ViT-B解码器处理所有token（包含帧内和全局交替注意力）→通过线性预测头映射回像素空间→计算重建损失。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)将MAE扩展到任意数量的视图；2)使用统一的掩码策略避免参考视图选择；3)设计轻量级多视图解码器；4)提出对称架构；5)能灵活混合单视图和多视图数据。不同之处：相比CroCo，MuM不需参考视图，可处理任意数量视图，架构更简单，数据采样更灵活；相比DINOv3，MuM专注几何理解而非语义，计算成本低约30倍，但在3D任务上表现更好；相比标准MAE，MuM能学习几何特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MuM通过将掩码自编码扩展到多视图场景，提供了一种简单、可扩展的自监督方法，能够学习对3D视觉任务有用的几何特征，在多种下游任务上优于现有方法同时大幅降低计算成本。'}


### 论文摘要

Self-supervised learning on images seeks to extract meaningful visual representations from unlabeled data. When scaled to large datasets, this paradigm has achieved state-of-the-art performance and the resulting trained models such as DINOv3 have seen widespread adoption. However, most prior efforts are optimized for semantic understanding rather than geometric reasoning. One important exception is Cross-View Completion, CroCo, which is a form of masked autoencoding (MAE) tailored for 3D understanding. In this work, we continue on the path proposed by CroCo and focus on learning features tailored for 3D vision. In a nutshell, we extend MAE to arbitrarily many views of the same scene. By uniformly masking all views and employing a lightweight decoder with inter-frame attention, our approach is inherently simpler and more scalable than CroCo. We evaluate the resulting model, MuM, extensively on downstream tasks including feedforward reconstruction, dense image matching and relative pose estimation, finding that it outperforms the state-of-the-art visual encoders DINOv3 and CroCo v2.

---

## 15. TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making

**论文链接:** [http://arxiv.org/abs/2511.17225v1](http://arxiv.org/abs/2511.17225v1)

**作者:** Shanshan Li, Da Huang, Yu He, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue

**发布时间:** 2025-11-21

**备注:** Accepted at NeurIPS 2025

### GPT解析

### 总结

本文提出了一种新的任务偏好多需求驱动导航(TP-MDDN)基准和AWMSystem自主决策系统，用于解决具身AI中涉及多种需求和现实世界任务复杂性的导航挑战。

### 背景

日常生活中，人们经常在空间中移动以满足需求，这是具身AI的关键挑战。传统的需求驱动导航(DDN)一次只处理一个需求，不能反映涉及多种需求和个人选择的现实世界任务的复杂性。

### 目的

引入任务偏好多需求驱动导航(TP-MDDN)基准，解决涉及具有明确任务偏好的多个子需求的长时程导航问题，以更准确地模拟现实世界中的导航任务。

### 方法

提出AWMSystem自主决策系统，包含三个关键模块：BreakLLM(指令分解)、LocateLLM(目标选择)和StatusMLLM(任务监控)；设计MASMap空间记忆系统，结合3D点云积累和2D语义映射；采用双节奏动作生成框架，集成零样本规划和基于策略的精细控制；配备自适应错误校正器，实时处理失败情况。

### 主要发现

实验证明，所提出的方法在感知准确性和导航鲁棒性方面优于最先进的基线方法，有效解决了传统方法在处理多需求导航任务时的局限性。

### 结论

TP-MDDN基准和AWMSystem系统能够有效处理现实世界中的多需求导航任务，在复杂场景中表现出色，为具身AI领域提供了新的解决方案。

### 翻译

在日常生活中，人们经常在空间中移动以满足需求，这构成了具身AI中的一个关键挑战。传统的需求驱动导航(DDN)一次只处理一个需求，但不能反映涉及多种需求和个人选择的现实世界任务的复杂性。为了弥补这一差距，我们引入了任务偏好多需求驱动导航(TP-MDDN)，这是一个涉及具有明确任务偏好的多个子需求的长时程导航新基准。为解决TP-MDDN，我们提出了AWMSystem，这是一个由三个关键模块组成的自主决策系统：BreakLLM(指令分解)、LocateLLM(目标选择)和StatusMLLM(任务监控)。对于空间记忆，我们设计了MASMap，它将3D点云积累与2D语义映射相结合，以实现准确高效的环境理解。我们的双节奏动作生成框架将零样本规划与基于策略的精细控制相结合，并得到了自适应错误校正器的进一步支持，该校正器可实时处理失败情况。实验证明，我们的方法在感知准确性和导航鲁棒性方面都优于最先进的基线方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决任务偏好多需求驱动导航问题，即如何让智能体在复杂环境中同时处理多个需求并考虑个人偏好。这个问题很重要，因为在日常生活中，人们经常需要完成一系列相关任务（如先清洁、然后休息、再进食），而不是简单的单一需求，现有方法无法有效处理这种复杂场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析传统需求驱动导航(DDN)的局限性，发现其无法处理多需求场景。他们借鉴了WMNav的自主决策系统设计，参考了InstructNav的零样本场景理解能力，并采用了双系统机器人的双节奏动作生成策略。在此基础上，作者设计了一个模块化的自主决策系统，结合大型语言模型进行指令分解、目标选择和任务监控，同时开发了高效的空间记忆方案和错误纠正机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将复杂指令分解为带偏好的子任务，使用大型语言模型进行自主决策，结合高效的空间记忆和双节奏动作生成来平衡性能与效率。整体流程包括：1)使用BreakLLM分解指令为子任务；2)通过RGB-D图像感知环境并构建MASMap空间记忆；3)使用LocateLLM选择下一个目标；4)采用双节奏动作生成框架执行导航；5)通过StatusMLLM监控任务进度；6)使用自适应错误校正器处理失败情况。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出TP-MDDN新基准，首次处理多子需求任务和明确任务偏好；2)设计AWMSystem自主决策系统，包含指令分解、目标选择和任务监控三个模块；3)开发MASMap空间记忆方案，结合3D点云和2D语义映射；4)提出双节奏动作生成框架平衡效率与性能；5)实现自适应错误校正器提高鲁棒性。相比之前工作，本文能同时处理多个需求并考虑个人偏好，采用模块化设计而非端到端训练，在长期导航任务中表现出更强的鲁棒性和环境适应性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了TP-MDDN基准和AWMSystem系统，通过结合大型语言模型的推理能力与高效的空间记忆方案，实现了在复杂环境中处理多需求任务偏好的长期导航，显著提升了导航的准确性和鲁棒性。'}


### 论文摘要

In daily life, people often move through spaces to find objects that meet their needs, posing a key challenge in embodied AI. Traditional Demand-Driven Navigation (DDN) handles one need at a time but does not reflect the complexity of real-world tasks involving multiple needs and personal choices. To bridge this gap, we introduce Task-Preferenced Multi-Demand-Driven Navigation (TP-MDDN), a new benchmark for long-horizon navigation involving multiple sub-demands with explicit task preferences. To solve TP-MDDN, we propose AWMSystem, an autonomous decision-making system composed of three key modules: BreakLLM (instruction decomposition), LocateLLM (goal selection), and StatusMLLM (task monitoring). For spatial memory, we design MASMap, which combines 3D point cloud accumulation with 2D semantic mapping for accurate and efficient environmental understanding. Our Dual-Tempo action generation framework integrates zero-shot planning with policy-based fine control, and is further supported by an Adaptive Error Corrector that handles failure cases in real time. Experiments demonstrate that our approach outperforms state-of-the-art baselines in both perception accuracy and navigation robustness.

---

## 16. FireScope: Wildfire Risk Prediction with a Chain-of-Thought Oracle

**论文链接:** [http://arxiv.org/abs/2511.17171v1](http://arxiv.org/abs/2511.17171v1)

**作者:** Mario Markov, Stefan Maria Ailuro, Luc Van Gool, Konrad Schindler, Danda Pani Paudel

**发布时间:** 2025-11-21

### GPT解析

### 总结

该研究提出了FireScope框架，一个基于视觉语言模型的推理到生成框架，用于预测野火风险。通过结合Sentinel-2图像、气候数据和专家定义的风险栅格，该框架实现了跨大陆的野火风险预测，并显著提高了模型的泛化能力和可解释性。

### 背景

预测野火风险是一个推理密集型的空间问题，需要整合视觉、气候和地理因素来推断连续的风险地图。现有方法缺乏可靠泛化所需的因果推理和多模态理解。

### 目的

开发一个能够进行可靠泛化的野火风险预测框架，提高模型的泛化能力和可解释性。

### 方法

引入了FireScope-Bench数据集和基准，结合Sentinel-2图像、气候数据、美国专家定义的风险栅格以及欧洲的实际野火事件；提出基于VLM的FireScope推理到生成框架，通过强化学习和视觉监督学习预测风险栅格。

### 主要发现

FireScope在美国训练并在欧洲测试时取得了显著的性能提升；其推理轨迹是忠实且语义上有意义的；推理可以为基础栅格预测模型提供支持，提高泛化能力和可解释性。

### 结论

FireScope-Bench有潜力成为推动推理驱动、可解释和可泛化空间建模的基础；该研究首次证明基于语言的推理可以改善视觉生成泛化，提出可跨大陆应用的高分辨率野火风险模型，并 enable对多模态火灾风险模型的稳健跨大陆泛化进行系统性研究。

### 翻译

预测野火风险是一个推理密集型的空间问题，需要整合视觉、气候和地理因素来推断连续的风险地图。现有方法缺乏可靠泛化所需的因果推理和多模态理解。我们引入了FireScope-Bench，这是一个大规模数据集和基准，它将Sentinel-2图像和气候数据与美国专家定义的风险栅格以及欧洲的实际野火事件相结合，用于跨大陆评估。基于此数据集，我们提出了FireScope，这是一个基于VLM的推理到生成框架，它通过强化学习和视觉监督学习来预测具有互补推理轨迹的风险栅格。在美国训练并在欧洲测试时，FireScope取得了显著的性能提升，而专家反馈和自动分析证实其推理轨迹是忠实且语义上有意义的。我们的研究结果表明，推理可以为基础栅格预测模型提供支持，同时提高泛化能力和可解释性。据我们所知，这是第一个证明基于语言的推理可以改善视觉生成泛化的框架，提出可跨大陆应用的高分辨率野火风险模型，并 enable对多模态火灾风险模型的稳健跨大陆泛化进行系统性研究。我们认为FireScope-Bench有潜力成为推动推理驱动、可解释和可泛化空间建模的基础。数据和源代码将公开提供。


### 论文摘要

Predicting wildfire risk is a reasoning-intensive spatial problem that requires the integration of visual, climatic, and geographic factors to infer continuous risk maps. Existing methods lack the causal reasoning and multimodal understanding required for reliable generalization. We introduce $\textbf{FireScope-Bench}$, a large-scale dataset and benchmark that couples Sentinel-2 imagery and climate data with expert-defined risk rasters across the USA, and real wildfire events in Europe for cross-continental evaluation. Building on this dataset, we propose $\textbf{FireScope}$, a VLM-based reasoning-to-generation framework that learns from both reinforcement learning and visual supervision to predict risk rasters with complementary reasoning traces. When trained in the USA and tested in Europe, $\textbf{FireScope}$ achieves substantial performance gains, while expert feedback and automated analysis confirm that its reasoning traces are faithful and semantically meaningful. Our findings demonstrate that reasoning can ground raster prediction models, improving both generalization and interpretability. To our knowledge, this is the first framework to (1) demonstrate that language-based reasoning can improve generalization in visual generation, (2) propose a high-resolution wildfire risk model that can be applied across continents, and (3) enable systematic studies of robust cross-continental generalization for multimodal fire risk models. We believe that $\textbf{FireScope-Bench}$ has the potential to serve as a foundation for advancing reasoning-driven, interpretable and generalizable spatial modeling. Data and source code will be made publicly available.

---

## 17. Four decades of circumpolar super-resolved satellite land surface temperature data

**论文链接:** [http://arxiv.org/abs/2511.17134v1](http://arxiv.org/abs/2511.17134v1)

**作者:** Sonia Dupuis, Nando Metzger, Konrad Schindler, Frank Göttsche, Stefan Wunderle

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文介绍了一个新的42年全北极地表温度数据集，通过深度学习技术将AVHRR GAC数据从低分辨率提升至1公里分辨率，为北极地区提供了高分辨率的地表温度观测，支持多种气候相关研究。

### 背景

地表温度是理解地-大气能量交换和监测气候变化的关键气候变量，在迅速变暖的北极地区尤为重要。基于卫星的长期LST记录对于检测气候趋势至关重要，但AVHRR全球覆盖数据的粗空间分辨率限制了其在分析北极细尺度冻土动力学和其他地表过程中的应用。

### 目的

开发一个新的高分辨率全北极地表温度数据集，提高空间分辨率，以便更好地分析北极地区的细尺度地表过程，特别是冻土动力学，并支持多种气候相关研究。

### 方法

使用基于深度各向异性扩散模型的超分辨率算法，将AVHRR GAC数据降尺度至1公里。该模型使用MODIS LST数据进行训练，采用降尺度的输入和原生分辨率的输出，并由高分辨率土地覆盖、数字高程和植被高度图指导。

### 主要发现

成功生成了覆盖全北极地区、长达42年的数据集，提供每日两次、1公里分辨率的地表温度观测。这个增强的数据集改进了冻土建模、近地表气温重建和格陵兰冰盖物质平衡评估的准确性，并支持MODIS时代前的气候监测工作。

### 结论

通过深度学习技术将低分辨率AVHRR数据提升至1公里分辨率，显著提高了北极地区地表温度数据的空间分辨率和应用价值，为多种气候相关研究提供了高质量的数据支持，并建立了一个可扩展到未来卫星任务的框架。

### 翻译

地表温度是理解地-大气能量交换和监测气候变化的关键气候变量，在迅速变暖的北极地区尤为重要。基于卫星的长期LST记录对于检测气候趋势至关重要。然而，AVHRR全球覆盖数据的粗空间分辨率限制了其在分析北极细尺度冻土动力学和其他地表过程中的应用。本文介绍了一个新的42年全北极LST数据集，通过基于深度各向异性扩散模型的超分辨率算法将AVHRR GAC数据降尺度至1公里。该模型使用MODIS LST数据进行训练，采用降尺度的输入和原生分辨率的输出，并由高分辨率土地覆盖、数字高程和植被高度图指导。生成的数据集提供了整个北极地区长达四十年、每日两次的1公里LST观测。这个增强的数据集改进了冻土建模、近地表气温重建和格陵兰冰盖物质平衡评估。此外，它还支持MODIS时代前的气候监测工作，并为未来热红外观测卫星任务和气候数据记录连续性提供了可适应的框架。


### 论文摘要

Land surface temperature (LST) is an essential climate variable (ECV) crucial for understanding land-atmosphere energy exchange and monitoring climate change, especially in the rapidly warming Arctic. Long-term satellite-based LST records, such as those derived from the Advanced Very High Resolution Radiometer (AVHRR), are essential for detecting climate trends. However, the coarse spatial resolution of AVHRR's global area coverage (GAC) data limit their utility for analyzing fine-scale permafrost dynamics and other surface processes in the Arctic. This paper presents a new 42 years pan-Arctic LST dataset, downscaled from AVHRR GAC to 1 km with a super-resolution algorithm based on a deep anisotropic diffusion model. The model is trained on MODIS LST data, using coarsened inputs and native-resolution outputs, guided by high-resolution land cover, digital elevation, and vegetation height maps. The resulting dataset provides twice-daily, 1 km LST observations for the entire pan-Arctic region over four decades. This enhanced dataset enables improved modelling of permafrost, reconstruction of near-surface air temperature, and assessment of surface mass balance of the Greenland Ice Sheet. Additionally, it supports climate monitoring efforts in the pre-MODIS era and offers a framework adaptable to future satellite missions for thermal infrared observation and climate data record continuity.

---

## 18. R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios

**论文链接:** [http://arxiv.org/abs/2511.16901v1](http://arxiv.org/abs/2511.16901v1)

**作者:** Lu Zhu, Tiantian Geng, Yangye Chen, Teng Wang, Ping Lu, Feng Zheng

**发布时间:** 2025-11-21

**备注:** Accepted by AAAI 2026. Project page: https://github.com/zhlllau/R-AVST

### GPT解析

### 总结

该研究提出了R-AVST数据集和AVST-Zero模型，用于提升真实世界音频-视觉事件的时空推理能力。

### 背景

多模态大语言模型在视频理解任务中进展迅速，但现有研究仅关注简单视频场景，无法反映真实世界音频-视觉事件的复杂性和多样性。

### 目的

弥补当前研究的不足，专注于真实世界的音频-视觉事件，引入具有细粒度时空标注的音频-视觉推理数据集。

### 方法

构建R-AVST数据集，采用基于LLLM的关键对象提取、自动空间标注和人工质量检查的流程，定义三个核心任务，并生成超过8K个问题-答案对；提出AVST-Zero模型，一种基于强化学习的模型，通过多维度奖励直接优化行为。

### 主要发现

实验验证了R-AVST数据集在推进音频-视觉时空推理方面的有效性，AVST-Zero模型与现有模型相比表现出有竞争力的性能。

### 结论

R-AVST是首个专为真实世界音频-视觉时空推理设计的数据集，AVST-Zero为解决该领域的未来挑战提供了新的视角。

### 翻译

最近，多模态大语言模型在视频理解任务中取得了快速进展，特别是在视频理解方面。然而，当前研究集中在简单的视频场景，无法反映视频中真实世界音频-视觉事件的复杂性和多样性。为了弥补这一差距，我们首先引入了R-AVST，这是一个具有细粒度时空标注的音频-推理数据集。在构建过程中，我们设计了一个包含基于LLLM的关键对象提取、自动空间标注和人工质量检查的流程，产生了超过5K个未修剪视频，包含100种音频-视觉事件类型的27K个对象。基于此数据集，我们定义了音频-视觉场景中时空推理的三个核心任务，并生成了超过8K个高质量、均匀分布的问题-答案对，以有效评估模型性能。为进一步增强推理能力，我们提出了AVST-Zero，一种基于强化学习的模型，避免了中间监督，通过精心设计的多维度奖励直接优化行为。大量实验验证了我们的R-AVST在推进音频-视觉时空推理方面的有效性，基于此，AVST-Zero与现有模型相比表现出有竞争力的性能。据我们所知，R-AVST是首个专为真实世界音频-视觉时空推理设计的数据集，AVST-Zero为解决该领域的未来挑战提供了新的视角。


### 论文摘要

Recently, rapid advancements have been made in multimodal large language models (MLLMs), especially in video understanding tasks. However, current research focuses on simple video scenarios, failing to reflect the complex and diverse nature of real-world audio-visual events in videos. To bridge this gap, we firstly introduce R-AVST, a dataset for audio-visual reasoning featuring fine-grained spatio-temporal annotations. In constructing this, we design a pipeline consisting of LLM-based key object extraction, automatic spatial annotation and manual quality inspection, resulting in over 5K untrimmed videos with 27K objects across 100 types of audio-visual events. Building on this dataset, we define three core tasks for spatio-temporal reasoning in audio-visual scenes and generate more than 8K high-quality, evenly distributed question-answer pairs to effectively benchmark model performance. To further enhance reasoning, we propose AVST-Zero, a reinforcement learning-based model that avoids intermediate supervision, directly optimizing behavior via carefully designed multi-dimensional rewards. Extensive experiments validate the effectiveness of our R-AVST in advancing audio-visual spatio-temporal reasoning, upon which AVST-Zero demonstrates competitive performance compared to existing models. To the best of our knowledge, R-AVST is the first dataset designed for real-world audio-visual spatio-temporal reasoning, and AVST-Zero offers a novel perspective for tackling future challenges in this domain.

---

## 19. BOP-ASK: Object-Interaction Reasoning for Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.16857v1](http://arxiv.org/abs/2511.16857v1)

**作者:** Vineet Bhat, Sungsu Kim, Valts Blukis, Greg Heinrich, Prashanth Krishnamurthy, Ramesh Karri, Stan Birchfield, Farshad Khorrami, Jonathan Tremblay

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文介绍了BOP-ASK，一个用于物体交互推理的大规模数据集，包含超过15万张图像和3300万个问答对，涵盖六个任务（四个为新任务），用于训练和评估视觉语言模型在细粒度空间理解方面的能力。

### 背景

视觉语言模型在空间推理基准测试上表现出色，但这些评估掩盖了理解物体交互方面的关键弱点。当前基准测试只测试高层次空间关系（如'左边'、'后面'等），忽略了现实世界应用中需要的细粒度空间理解：精确的3D定位、物体间的物理兼容性、物体可供性和多步骤空间规划。

### 目的

创建一个新颖的大规模数据集BOP-ASK，用于物体交互推理的训练和基准测试，以弥补当前评估方法的不足，促进视觉语言模型在细粒度空间理解方面的发展。

### 方法

利用来自物体姿态估计基准测试（BOP）数据集的6D物体姿态构建数据生成管道，从中派生出细粒度注释，如抓取姿态、被引用物体姿态、路径规划轨迹、相对空间和深度关系以及物体间关系。同时创建了BOP-ASK-core（测试基准）和BOP-ASK-lab（分布外基准，包含不来自BOP的图像）。

### 主要发现

在BOP-ASK上训练的模型优于基线模型，并展现出涌现能力，如精确的物体和抓取姿态估计、轨迹规划和在杂乱环境中的细粒度物体中心空间推理。

### 结论

BOP-ASK数据集为训练和评估视觉语言模型提供了丰富的资源，能够更好地理解和处理物体交互的细粒度空间关系。作者计划公开发布其数据集和数据集生成管道。

### 翻译

视觉语言模型在空间推理基准测试上取得了令人印象深刻的性能，但这些评估掩盖了理解物体交互方面的关键弱点。当前的基准测试测试高层次的关系（如'左边'、'后面'等），但忽略了现实世界应用中需要的细粒度空间理解：精确的3D定位、物体之间的物理兼容性、物体可供性和多步骤空间规划。在这项工作中，我们提出了BOP-ASK，这是一个用于物体交互推理的新型大规模数据集，用于训练和基准测试。我们的数据生成管道利用了来自物体姿态估计基准测试（BOP）数据集的6D物体姿态，从中我们派生出细粒度注释，如抓取姿态、被引用物体姿态、路径规划轨迹、相对空间和深度关系以及物体间关系。BOP-ASK包含超过15万张图像和3300万个问答对，涵盖六个任务（四个是新任务），为训练和评估视觉语言模型提供了丰富的资源。我们评估了专有和开源的视觉语言模型，并在BOP-ASK-core（一个贡献的测试基准）上进行了人类评估。我们还发布了BOP-ASK-lab，这是一个分布外基准，包含不来自BOP的图像，能够测试泛化能力。我们的实验表明，在BOP-ASK上训练的模型优于基线模型，并展现出涌现能力，如精确的物体和抓取姿态估计、轨迹规划以及在杂乱环境中的细粒度物体中心空间推理。我们将公开发布我们的数据集和数据集生成管道。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决视觉语言模型在物体交互推理方面的不足问题。现有模型虽然能识别物体间的简单空间关系（如'左边'、'后面'），但缺乏理解物体间精细物理交互的能力，如精确抓取位置、物体间物理兼容性、多步骤空间规划等。这个问题在机器人操作、增强现实等实际应用中至关重要，因为这些应用需要模型不仅识别物体，还要理解如何精确抓取、移动它们，以及如何在复杂环境中规划路径。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有方法的局限性，如SpatialVLM和RoboPoint等在物体间关系和长期推理问题上表现不佳。作者选择基于BOP数据集构建新数据集，因为BOP提供高质量的3D标注图像和3D模型。作者借鉴了现有工作的元素，如RRT规划器用于生成无碰撞路径、M2T2模型用于计算抓取姿态，以及使用LLM生成问答对。但作者将这些元素整合到一个新框架中，专注于物体交互推理，这是之前工作没有系统解决的。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过创建大规模、几何一致的数据集，增强视觉语言模型对物体交互的理解能力，使模型能理解物体间的精确空间关系，从而在机器人操作等实际应用中表现出色。实现流程包括：1)从BOP数据集获取RGB-D图像和3D物体姿态；2)构建世界坐标系；3)生成几何先验（3D边界框、运动轨迹、抓取姿态）；4)使用模板和LLM生成多样化的问答对；5)组织数据集为训练集(BOP-Ask)和两个测试集(BOP-Ask-core和BOP-Ask-lab)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专注于物体交互推理的大规模数据集，包含150k图像和33M问答对；2)提供精确的3D几何基础，支持毫米级精度；3)定义六项核心技能的全面物体交互推理框架；4)使用像素级问答对，要求输出高精度的像素位置；5)自动化数据生成流程。相比之前工作，BOP-Ask规模更大、任务更全面、精确性更高，并提供了互补的评估基准，使模型能在已见和未见的环境中进行评估。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BOP-Ask通过创建首个专注于物体交互推理的大规模、精确几何数据集，显著提升了视觉语言模型在物体姿态估计、抓取预测、运动规划和场景理解等复杂任务上的表现，为机器人在现实世界中的操作应用提供了新的能力。'}


### 论文摘要

Vision Language Models (VLMs) have achieved impressive performance on spatial reasoning benchmarks, yet these evaluations mask critical weaknesses in understanding object interactions. Current benchmarks test high level relationships ('left of,' 'behind', etc.) but ignore fine-grained spatial understanding needed for real world applications: precise 3D localization, physical compatibility between objects, object affordances and multi step spatial planning. In this work, we present BOP-ASK, a novel large scale dataset for object interaction reasoning for both training and benchmarking. Our data generation pipeline leverages 6D object poses from the Benchmark for Object Pose Estimation (BOP) datasets from which we derive fine grained annotations such as grasp poses, referred object poses, path planning trajectories, relative spatial and depth relationships, and object-to-object relationships. BOP-ASK comprises over 150k images and 33M question answer pairs spanning six tasks (four novel), providing a rich resource for training and evaluating VLMs. We evaluate proprietary and open sourced VLMs, and conduct human evaluations on BOP-ASK-core, a contributed test benchmark. We also release BOP-ASK-lab, an out-of-distribution benchmark with images not sourced from BOP, enabling testing of generalization. Our experiments demonstrate that models trained on BOP-ASK outperform baselines and exhibit emergent capabilities such as precise object and grasp pose estimation, trajectory planning, and fine-grained object-centric spatial reasoning in cluttered environments. We will publicly release our datasets and dataset generation pipeline.

---

## 20. WorldGen: From Text to Traversable and Interactive 3D Worlds

**论文链接:** [http://arxiv.org/abs/2511.16825v1](http://arxiv.org/abs/2511.16825v1)

**作者:** Dilin Wang, Hyunyoung Jung, Tom Monnier, Kihyuk Sohn, Chuhang Zou, Xiaoyu Xiang, Yu-Ying Yeh, Di Liu, Zixuan Huang, Thu Nguyen-Phuoc, Yuchen Fan, Sergiu Oprea, Ziyan Wang, Roman Shapovalov, Nikolaos Sarafianos, Thibault Groueix, Antoine Toisoul, Prithviraj Dhar, Xiao Chu, Minghao Chen, Geon Yeong Park, Mahima Gupta, Yassir Azziz, Rakesh Ranjan, Andrea Vedaldi

**发布时间:** 2025-11-20

### GPT解析

### 总结

WorldGen系统可以从文本提示自动创建大规模交互式3D世界，将自然语言描述转换为可探索和编辑的3D环境。

### 背景

3D世界构建通常需要专业知识和手动建模，限制了创意表达的效率。

### 目的

开发一个系统，使创作者能够设计连贯、可导航的虚拟世界，无需手动建模或专业3D技能。

### 方法

结合LLM驱动的场景布局推理、程序生成、基于扩散的3D生成和对象感知场景分解技术。

### 主要发现

该系统能够生成几何一致、视觉丰富且实时渲染高效的3D世界，支持细粒度控制。

### 结论

WorldGen代表了向大规模、可访问的生成式世界构建迈出的一步，推动了3D生成AI在游戏、模拟和沉浸式社交环境中的应用。

### 翻译

我们引入了WorldGen，一个可以直接从文本提示自动创建大规模交互式3D世界的系统。我们的方法将自然语言描述转换为可穿越的、完全纹理化的环境，这些环境可以在标准游戏引擎中立即探索或编辑。通过结合LLM驱动的场景布局推理、程序生成、基于扩散的3D生成和对象感知场景分解，WorldGen弥合了创意意图和功能虚拟空间之间的差距，使创作者能够设计连贯、可导航的世界，无需手动建模或专业的3D知识。该系统完全模块化，支持对布局、规模和风格的细粒度控制，生成几何一致、视觉丰富且实时渲染高效的世界。这项工作向大规模、可访问的生成式世界构建迈出了一步，推进了3D生成AI的前沿，应用于游戏、模拟和沉浸式社交环境。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从简单的文本描述自动生成大规模、可交互的3D世界的问题。这个问题在现实中很重要，因为创建3D内容（如视频游戏）复杂、耗时且需要专业知识，而自动化生成可以显著减少创作时间，让更多人成为创作者，支持即时生成和个性化内容。在研究中，这代表了3D生成AI的前沿进展，弥合了创意意图和功能性虚拟空间之间的差距。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过问题分解思路，将直接从文本到3D场景的映射转化为先生成场景图像再进行图像到3D重建的两阶段任务。他们借鉴了多项现有工作：大型语言模型（LLM）用于解析文本提示，扩散模型用于图像生成，AssetGen2用于图像到3D转换，AutoPartGen用于场景分解，TRELLIS用于纹理生成，以及PartPacker用于加速部分提取。在此基础上，作者创新性地结合这些技术，引入导航网格确保场景可遍历性，并设计了四阶段流程（场景规划、重建、分解和增强）来实现从文本到完整3D世界的生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合大型语言模型、程序生成、扩散模型和3D重建技术，从文本描述自动生成功能性和组成性的3D场景，确保生成的环境既符合文本描述又具有可交互性。整体流程分为四个阶段：1）场景规划：使用LLM解析文本，程序生成粗略布局，提取导航网格，生成参考图像；2）场景重建：结合参考图像和导航网格，使用AssetGen2生成整体3D场景网格；3）场景分解：使用改进的AutoPartGen将场景分解为单独的低分辨率3D资产；4）场景增强：为每个对象生成高质量图像，重建形状，生成高分辨率纹理，同时保持整体一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）文本控制的可遍历场景生成，首次实现从文本生成功能性3D世界；2）导航网格引导的3D重建，确保场景可遍历性；3）场景分解与增强，允许局部编辑和提升质量；4）多阶段生成流程，保证全局一致性和局部可控性；5）改进的AutoPartGen，高效处理场景中的大量部件。相比之前工作，WorldGen不仅生成单个3D对象，而是完整场景；通过LLM和图像生成器实现了更大的风格和主题多样性；通过导航网格确保场景功能性；提供更细粒度的控制，允许对单个对象进行编辑和增强。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'WorldGen通过结合大型语言模型、程序生成、扩散模型和3D重建技术，首次实现了从简单文本描述自动生成可导航、可交互且视觉丰富的大规模3D世界，为游戏开发、模拟和沉浸式社交环境提供了强大的创作工具。'}


### 论文摘要

We introduce WorldGen, a system that enables the automatic creation of large-scale, interactive 3D worlds directly from text prompts. Our approach transforms natural language descriptions into traversable, fully textured environments that can be immediately explored or edited within standard game engines. By combining LLM-driven scene layout reasoning, procedural generation, diffusion-based 3D generation, and object-aware scene decomposition, WorldGen bridges the gap between creative intent and functional virtual spaces, allowing creators to design coherent, navigable worlds without manual modeling or specialized 3D expertise. The system is fully modular and supports fine-grained control over layout, scale, and style, producing worlds that are geometrically consistent, visually rich, and efficient to render in real time. This work represents a step towards accessible, generative world-building at scale, advancing the frontier of 3D generative AI for applications in gaming, simulation, and immersive social environments.

---

## 21. POMA-3D: The Point Map Way to 3D Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.16567v2](http://arxiv.org/abs/2511.16567v2)

**作者:** Ye Mao, Weixun Luo, Ranran Huang, Junpeng Jing, Krystian Mikolajczyk

**发布时间:** 2025-11-20

**备注:** 11 pages, 6 tables, 5 figures

### GPT解析

### 总结

本文介绍了POMA-3D，这是一种创新的3D表示模型，它通过点图(point maps)实现自监督学习，解决了3D表示学习中预训练先验稀缺和数据有限的问题。

### 背景

3D场景理解领域面临预训练先验稀缺和数据有限的挑战。现有的3D表示方法难以有效利用2D基础模型的知识。

### 目的

开发一种新的3D表示模型，能够从点图中学习自监督表示，并有效利用2D先验知识，提升3D场景理解能力。

### 方法

1. 引入点图(point maps)在结构化2D网格上编码3D坐标；2. 设计视图到场景的对齐策略(view-to-scene alignment)转移2D先验；3. 提出POMA-JEPA联合嵌入-预测架构确保多视图几何一致性；4. 构建ScenePoint数据集(6.5K房间级RGB-D场景和1M 2D图像场景)支持大规模预训练。

### 主要发现

POMA-3D可作为专业和通用3D理解的强大骨干网络，在3D问答、具身导航、场景检索和具身定位等多种任务中表现出色，且仅使用几何输入(3D坐标)即可实现。

### 结论

POMA-3D通过点图方式探索了3D场景理解的新路径，有效解决了3D表示学习中的先验稀缺和数据有限问题，为3D场景理解提供了新的可能性。

### 翻译

在本文中，我们介绍了POMA-3D，这是第一个从点图(point maps)学习的自监督3D表示模型。点图在结构化2D网格上编码显式3D坐标，保留了全局3D几何形状，同时与2D基础模型的输入格式兼容。为了将丰富的2D先验知识转移到POMA-3D，设计了一个视图到场景的对齐策略。此外，由于点图相对于规范空间是视图依赖的，我们引入了POMA-JEPA，这是一种联合嵌入-预测架构，强制跨多个视图的几何一致的点图特征。另外，我们引入了ScenePoint，这是一个由6.5K个房间级RGB-D场景和1M个2D图像场景构建的点图数据集，用于促进大规模POMA-3D预训练。实验表明，POMA-3D可以作为专业和通用3D理解的强大骨干网络。它有利于多种任务，包括3D问答、具身导航、场景检索和具身定位，所有这些仅使用几何输入（即3D坐标）实现。总体而言，我们的POMA-3D探索了点图方式用于3D场景理解，解决了3D表示学习中预训练先验稀缺和数据有限的问题。项目页面：https://matchlab-imperial.github.io/poma3d/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D场景理解中缺乏强大预训练模型的问题，具体表现为预训练先验稀缺和3D数据有限。这个问题很重要，因为3D场景理解是物理世界感知和交互的基础，是AR系统和具身智能体中情境智能的基础，影响着机器人在真实环境中的导航、交互和决策能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有3D视觉语言学习方法的局限性，指出它们使用的点云、深度图等表示与2D基础模型差异大。作者借鉴了CLIP的跨模态对齐思想和JEPA架构设计，创新性地提出使用点图作为2D-3D中间模态，因为它能同时保留3D几何信息并与2D模型兼容。作者还利用VGGT模型将2D图像转换为3D点图，并参考了现有3D视觉语言数据集的构建方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用点图作为3D场景的表示，通过大规模预训练学习通用3D场景表示，并利用2D视觉语言模型的知识迁移到3D领域。整体流程包括：1)构建ScenePoint数据集(6.5K房间级场景+1M图像场景)；2)两阶段预训练(温暖阶段在图像衍生的单视图点图上进行视觉-语言对齐，主要阶段在房间级多视图点图上进行联合优化)；3)将预训练模型应用于3D问答、具身导航等下游任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出POMA-3D，第一个基于点图的自监督3D表示学习模型；2)构建ScenePoint大规模数据集；3)设计视图到场景的视觉-语言对齐策略；4)提出POMA-JEPA架构强制多视图点图特征几何一致。相比之前工作，POMA-3D使用点图而非点云/深度图作为表示，专门预训练点图编码器而非仅作为位置编码，利用2D图像数据扩展3D预训练规模，解决了3D数据稀缺问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'POMA-3D通过引入点图表示和两阶段预训练策略，首次实现了从大规模2D视觉语言数据到3D场景理解的有效知识迁移，显著提升了多种3D理解任务的性能。'}


### 论文摘要

In this paper, we introduce POMA-3D, the first self-supervised 3D representation model learned from point maps. Point maps encode explicit 3D coordinates on a structured 2D grid, preserving global 3D geometry while remaining compatible with the input format of 2D foundation models. To transfer rich 2D priors into POMA-3D, a view-to-scene alignment strategy is designed. Moreover, as point maps are view-dependent with respect to a canonical space, we introduce POMA-JEPA, a joint embedding-predictive architecture that enforces geometrically consistent point map features across multiple views. Additionally, we introduce ScenePoint, a point map dataset constructed from 6.5K room-level RGB-D scenes and 1M 2D image scenes to facilitate large-scale POMA-3D pretraining. Experiments show that POMA-3D serves as a strong backbone for both specialist and generalist 3D understanding. It benefits diverse tasks, including 3D question answering, embodied navigation, scene retrieval, and embodied localization, all achieved using only geometric inputs (i.e., 3D coordinates). Overall, our POMA-3D explores a point map way to 3D scene understanding, addressing the scarcity of pretrained priors and limited data in 3D representation learning. Project Page: https://matchlab-imperial.github.io/poma3d/

---

## 22. Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small Multimodal Models

**论文链接:** [http://arxiv.org/abs/2511.17487v1](http://arxiv.org/abs/2511.17487v1)

**作者:** Mark Endo, Serena Yeung-Levy

**发布时间:** 2025-11-21

**备注:** Website at https://web.stanford.edu/~markendo/projects/downscaling_intelligence

### GPT解析

### 总结

本研究分析了多模态模型中智能缩减的影响，发现LLM缩减 disproportionately影响视觉能力而非语言能力，并提出了Extract+Think方法来解决这一瓶颈

### 背景

多模态模型的扩展已在视觉理解和推理方面取得显著进展，但实际需求需要更小、更高效的系统

### 目的

对多模态模型中的智能缩减进行有原则的分析，研究降低大型语言模型容量如何影响多模态能力

### 方法

引入视觉提取调优，明确训练模型跨任务一致地提取与指令相关的视觉细节，并应用逐步推理生成答案，形成Extract+Think方法

### 主要发现

LLM缩减 disproportionately影响视觉能力而非从LLM继承的能力；将LLM缩减对感知的影响分离出来，发现性能急剧下降，通常匹配或超过对推理的影响

### 结论

Extract+Think方法为多模态模型的效率和性能设定了新标准

### 翻译

扩展多模态模型已在视觉理解和推理方面取得了显著进展，但实际需求需要更小、更高效的系统。在这项工作中，我们对多模态模型中的智能缩减进行了有原则的分析，研究了降低大型语言模型容量如何影响多模态能力。我们的初步发现揭示了一个有趣的趋势：LLM缩减 disproportionately影响视觉能力，而不是从LLM继承的能力。然后我们研究这种下降是否主要反映了视觉推理的预期下降，还是感知能力的更根本性丧失。将LLM缩减对感知的影响分离出来，我们发现性能仍然急剧下降，通常匹配或超过对推理的影响。为解决这一瓶颈，我们引入了视觉提取调优，明确训练模型跨任务一致地提取与指令相关的视觉细节。使用这些提取的视觉细节，我们然后应用逐步推理来生成答案。这些组件共同构成了我们的Extract+Think方法，为该领域的效率和性能设定了新标准。


### 论文摘要

Scaling up multimodal models has enabled remarkable advances in visual understanding and reasoning, but practical demands call for smaller, efficient systems. In this work, we conduct a principled analysis of downscaling intelligence in multimodal models, examining how reduced large language model (LLM) capacity affects multimodal capabilities. Our initial findings reveal an interesting trend: LLM downscaling disproportionately affects visual capabilities, rather than abilities inherited from the LLM. We then examine whether this drop mainly reflects the expected decline in visual reasoning or a more fundamental loss of perceptual abilities. Isolating the effect of LLM downscaling on perception, we find performance still drops sharply, often matching or exceeding the impact on reasoning. To address this bottleneck, we introduce visual extraction tuning, which explicitly trains the model to extract instruction-relevant visual details consistently across tasks. With these extracted visual details, we then apply step-by-step reasoning to generate answers. Together, these components form our Extract+Think approach, setting a new standard for efficiency and performance in this space.

---

## 23. Counterfactual World Models via Digital Twin-conditioned Video Diffusion

**论文链接:** [http://arxiv.org/abs/2511.17481v1](http://arxiv.org/abs/2511.17481v1)

**作者:** Yiqing Shen, Aiza Maksutova, Chenjia Li, Mathias Unberath

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了CWMDT框架，将标准视频扩散模型转化为反事实世界模型，使模型能够回答'如果移除这个物体会发生什么'等反事实查询。

### 背景

当前世界模型基于事实观察进行预测，但在需要评估物理AI行为等应用中，反事实查询能力变得日益重要。

### 目的

正式化反事实世界模型，使其能够接受干预作为显式输入，预测在假设修改场景属性下的时序序列。

### 方法

CWMDT通过三步实现：1)构建场景的数字孪生体以结构化文本编码对象关系；2)使用大型语言模型推理反事实干预的传播；3)用修改后的表示调节视频扩散模型生成反事实视觉序列。

### 主要发现

在两个基准测试上，CWMDT方法实现了最先进的性能。

### 结论

替代视频的表示（如数字孪生体）为基于视频前向模拟的世界模型提供了强大的控制信号。

### 翻译

世界模型学习根据控制信号预测视觉观察的时序演变，可能使代理能够通过前向模拟来推理环境。由于专注于前向模拟，当前世界模型基于事实观察生成预测。对于许多新兴应用，如在不同条件下对物理AI行为进行全面评估，世界模型回答反事实查询（如'如果移除这个物体会发生什么？'）的能力变得越来越重要。我们正式化了反事实世界模型，它们额外将干预作为显式输入，预测在假设修改观察到的场景属性下的时序序列。传统世界模型直接在纠缠的像素空间表示上操作，无法选择性地修改特定场景属性。我们引入CWMDT框架来克服这些限制，将标准视频扩散模型转化为有效的反事实世界模型。首先，CWMDT构建观察场景的数字孪生体以显式编码对象及其关系，表示为结构化文本。其次，CWMDT应用大型语言模型对这些表示进行推理，预测反事实干预如何随时间传播以改变观察场景。第三，CWMDT使用修改后的表示调节视频扩散模型以生成反事实视觉序列。在两个基准测试上的评估显示，CWMDT方法实现了最先进的性能，这表明视频的替代表示（如本文考虑的数字孪生体）为基于视频前向模拟的世界模型提供了强大的控制信号。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文解决传统世界模型无法进行反事实推理的问题，即无法回答'如果...会怎样？'的假设性问题。这一问题在自主驾驶、机器人控制和物理AI评估等场景中至关重要，因为系统需要评估不同条件下的可能结果，而不仅仅是基于当前事实进行预测。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析传统世界模型的局限性（直接在纠缠的像素空间操作，无法选择性修改特定场景属性），设计出三阶段方法：感知（构建数字孪生表示）、干预（LLM推理）、合成（视频生成）。该方法借鉴了多种现有工作：世界模型理论、视频扩散模型（如SORA）、数字孪生表示概念，以及多种视觉模型（SAM-2、DepthAnything等）用于构建结构化场景表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过数字孪生表示分离推理与合成过程，使AI系统能在结构化文本表示上进行明确反事实推理，然后转化为视觉序列。整体流程分为三阶段：1)数字孪生构建：使用多种视觉模型将视频帧转换为结构化文本表示；2)反事实推理：LLM分析干预查询，预测场景变化并生成修改后的数字孪生序列；3)视频合成：基于修改后的数字孪生表示和编辑后的初始帧，通过微调的视频扩散模型生成反事实视频序列。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)正式化反事实世界模型概念；2)提出CWMDT三阶段框架；3)利用数字孪生表示实现推理与生成分离；4)结合LLM进行场景级反事实推理。相比之前的工作，该方法能处理复杂反事实查询，而传统世界模型只能基于事实预测，视频扩散模型无法进行有针对性干预，其他视频编辑方法直接在像素空间操作难以处理多步推理问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出CWMDT框架，通过数字孪生表示将视频扩散模型转换为能够回答反事实查询的世界模型，实现了推理与生成的分离，使AI系统能够评估假设场景下的可能结果。'}


### 论文摘要

World models learn to predict the temporal evolution of visual observations given a control signal, potentially enabling agents to reason about environments through forward simulation. Because of the focus on forward simulation, current world models generate predictions based on factual observations. For many emerging applications, such as comprehensive evaluations of physical AI behavior under varying conditions, the ability of world models to answer counterfactual queries, such as "what would happen if this object was removed?", is of increasing importance. We formalize counterfactual world models that additionally take interventions as explicit inputs, predicting temporal sequences under hypothetical modifications to observed scene properties. Traditional world models operate directly on entangled pixel-space representations where object properties and relationships cannot be selectively modified. This modeling choice prevents targeted interventions on specific scene properties. We introduce CWMDT, a framework to overcome those limitations, turning standard video diffusion models into effective counterfactual world models. First, CWMDT constructs digital twins of observed scenes to explicitly encode objects and their relationships, represented as structured text. Second, CWMDT applies large language models to reason over these representations and predict how a counterfactual intervention propagates through time to alter the observed scene. Third, CWMDT conditions a video diffusion model with the modified representation to generate counterfactual visual sequences. Evaluations on two benchmarks show that the CWMDT approach achieves state-of-the-art performance, suggesting that alternative representations of videos, such as the digital twins considered here, offer powerful control signals for video forward simulation-based world models.

---

## 24. Scaling Conditional Autoencoders for Portfolio Optimization via Uncertainty-Aware Factor Selection

**论文链接:** [http://arxiv.org/abs/2511.17462v1](http://arxiv.org/abs/2511.17462v1)

**作者:** Ryan Engel, Yu Chen, Pawel Polak, Ioana Boier

**发布时间:** 2025-11-21

**DOI:** 10.1145/3768292.3770415

**备注:** 9 pages, 6 figures. Published in Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25)

### GPT解析

### 总结

本文提出了一种可扩展框架，将高维条件自编码器与不确定性感知的因子选择过程相结合，克服了传统方法中潜在因子维度限制的问题，显著提升了风险调整后性能。

### 背景

条件自编码器(CAEs)提供了一种灵活、可解释的方法，用于从公司特征中估计潜在资产定价因子。然而，现有研究通常将潜在因子维度限制在K=5左右，因为担心更大的K会降低性能。

### 目的

克服高维CAE的性能挑战，提出一个可扩展框架，将高维CAE与不确定性感知的因子选择过程相结合。

### 方法

使用三种模型进行分位数预测：零样本Chronos(ZS-Chronos)、梯度提升分位数回归树(Q-Boost)和基于I.I.D自助采样的样本均值模型(IID-BS)。通过预测不确定性对因子排序，保留前k个最可预测的因子进行投资组合构建。

### 主要发现

剪枝策略在所有预测模型中都带来了风险调整后性能的显著提升。由于各模型预测不相关，性能加权的集合始终优于单个模型，具有更高的夏普比率、索提诺比率和欧米伽比率。

### 结论

所提出的框架成功克服了高维CAE的性能问题，通过不确定性感知的因子选择提高了预测性能。

### 翻译

条件自编码器(CAEs)提供了一种灵活、可解释的方法，用于从公司特征中估计潜在资产定价因子。然而，现有研究通常将潜在因子维度限制在K=5左右，因为担心更大的K会降低性能。为克服这一挑战，我们提出了一种可扩展框架，将高维CAE与不确定性感知的因子选择过程相结合。我们使用三种模型进行分位数预测：零样本Chronos，一个预训练的时间序列基础模型(ZS-Chronos)、使用XGBoost和RAPIDS的梯度提升分位数回归树(Q-Boost)以及基于I.I.D自助采样的样本均值模型(IID-BS)。对于每个模型，我们通过预测不确定性对因子进行排序，并为投资组合构建保留前k个最可预测的因子，其中k表示所选的因子子集。这种剪枝策略在所有预测模型中都带来了风险调整后性能的显著提升。此外，由于每个模型的预测都不相关，性能加权的集合始终优于单个模型，具有更高的夏普比率、索提诺比率和欧米伽比率。


### 论文摘要

Conditional Autoencoders (CAEs) offer a flexible, interpretable approach for estimating latent asset-pricing factors from firm characteristics. However, existing studies usually limit the latent factor dimension to around K=5 due to concerns that larger K can degrade performance. To overcome this challenge, we propose a scalable framework that couples a high-dimensional CAE with an uncertainty-aware factor selection procedure. We employ three models for quantile prediction: zero-shot Chronos, a pretrained time-series foundation model (ZS-Chronos), gradient-boosted quantile regression trees using XGBoost and RAPIDS (Q-Boost), and an I.I.D bootstrap-based sample mean model (IID-BS). For each model, we rank factors by forecast uncertainty and retain the top-k most predictable factors for portfolio construction, where k denotes the selected subset of factors. This pruning strategy delivers substantial gains in risk-adjusted performance across all forecasting models. Furthermore, due to each model's uncorrelated predictions, a performance-weighted ensemble consistently outperforms individual models with higher Sharpe, Sortino, and Omega ratios.

---

## 25. MMT-ARD: Multimodal Multi-Teacher Adversarial Distillation for Robust Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.17448v1](http://arxiv.org/abs/2511.17448v1)

**作者:** Yuqi Li, Junhao Dong, Chuanguang Yang, Shiping Wen, Piotr Koniusz, Tingwen Huang, Yingli Tian, Yew-Soon Ong

**发布时间:** 2025-11-21

**备注:** 10 pages

### GPT解析

### 总结

本文提出了MMT-ARD框架，通过多模态多教师对抗知识蒸馏提升视觉-语言模型的对抗鲁棒性，实验证明该方法在提高模型鲁棒性和准确性的同时，显著提升了训练效率。

### 背景

视觉-语言模型(VLMs)越来越多地部署在安全关键应用中，其对抗鲁棒性成为重要问题。传统单教师对抗知识蒸馏方法存在知识多样性有限、收敛速度慢以及难以平衡鲁棒性和准确性的挑战。

### 目的

解决传统单教师对抗知识蒸馏方法的局限性，提高视觉-语言模型的对抗鲁棒性和训练效率。

### 方法

提出MMT-ARD多模态多教师对抗鲁棒蒸馏框架，包含双教师知识融合架构、基于教师置信度的动态权重分配策略以及基于自适应sigmoid的加权函数，以协同优化特征保存和增强，平衡跨模态知识转移。

### 主要发现

在ImageNet和零样本基准上，MMT-ARD在ViT-B-32模型上将鲁棒准确性提高4.32%，零样本准确性提高3.5%，同时训练效率比传统单教师方法提高2.3倍。

### 结论

MMT-ARD框架在增强多模态大模型对抗鲁棒性方面具有显著的有效性和可扩展性，为安全关键应用中的视觉-语言模型提供了鲁棒性解决方案。

### 翻译

视觉-语言模型(VLMs)越来越多地部署在安全关键应用中，使其对抗鲁棒性成为一个重要问题。虽然对抗知识蒸馏在将鲁棒性从教师模型转移到学生模型方面显示出潜力，但传统的单教师方法存在知识多样性有限、收敛速度慢以及难以平衡鲁棒性和准确性的问题。为了解决这些挑战，我们提出了MMT-ARD：一个多模态多教师对抗鲁棒蒸馏框架。我们的关键创新是一个双教师知识融合架构，协同优化清洁特征保存和鲁棒特征增强。为了更好地处理具有挑战性的对抗样本，我们引入了一种基于教师置信度的动态权重分配策略，使模型能够自适应地关注更难的样本。此外，为了减轻教师之间的偏差，我们设计了一种基于自适应sigmoid的加权函数，平衡跨模态的知识转移强度。在ImageNet和零样本基准上的大量实验表明，MMT-ARD在ViT-B-32模型上将鲁棒准确性提高了4.32%，零样本准确性提高了3.5%，同时与传统单教师方法相比，训练效率提高了2.3倍。这些结果突显了MMT-ARD在增强多模态大模型对抗鲁棒性方面的有效性和可扩展性。我们的代码可在https://github.com/itsnotacie/MMT-ARD获取。


### 论文摘要

Vision-Language Models (VLMs) are increasingly deployed in safety-critical applications, making their adversarial robustness a crucial concern. While adversarial knowledge distillation has shown promise in transferring robustness from teacher to student models, traditional single-teacher approaches suffer from limited knowledge diversity, slow convergence, and difficulty in balancing robustness and accuracy. To address these challenges, we propose MMT-ARD: a Multimodal Multi-Teacher Adversarial Robust Distillation framework. Our key innovation is a dual-teacher knowledge fusion architecture that collaboratively optimizes clean feature preservation and robust feature enhancement. To better handle challenging adversarial examples, we introduce a dynamic weight allocation strategy based on teacher confidence, enabling adaptive focus on harder samples. Moreover, to mitigate bias among teachers, we design an adaptive sigmoid-based weighting function that balances the strength of knowledge transfer across modalities. Extensive experiments on ImageNet and zero-shot benchmarks demonstrate that MMT-ARD improves robust accuracy by +4.32% and zero-shot accuracy by +3.5% on the ViT-B-32 model, while achieving a 2.3x increase in training efficiency over traditional single-teacher methods. These results highlight the effectiveness and scalability of MMT-ARD in enhancing the adversarial robustness of multimodal large models. Our codes are available at https://github.com/itsnotacie/MMT-ARD.

---

## 26. REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing

**论文链接:** [http://arxiv.org/abs/2511.17442v1](http://arxiv.org/abs/2511.17442v1)

**作者:** Binger Chen, Tacettin Emre Bök, Behnood Rasti, Volker Markl, Begüm Demir

**发布时间:** 2025-11-21

**备注:** Code and data available at https://github.com/be-chen/REMSA

### GPT解析

### 总结

本研究介绍了一种基于大语言模型的自动化遥感基础模型选择代理REMSA，以及一个涵盖150多个遥感基础模型的数据库RS-FMD。这些工具解决了遥感领域中选择合适基础模型的困难问题，通过自然语言查询实现模型自动选择，并提供透明化的理由。

### 背景

基础模型在遥感领域被越来越多地用于环境监测、灾害评估和土地利用制图等任务。这些模型包括单模态视觉编码器和多模态架构，支持多种遥感任务。然而，由于文档分散、格式异构和部署约束各异，选择合适的遥感基础模型仍然很困难。

### 目的

开发一个结构化的遥感基础模型数据库和一种基于大语言模型的自动化选择代理，以解决遥感领域中选择合适基础模型的困难问题。

### 方法

研究团队构建了RSFM数据库(RS-FMD)，涵盖150多个跨越多种数据模态、分辨率和学习范式的遥感基础模型。基于此数据库，开发了REMSA代理，它能够解释用户需求、解决缺失约束、使用上下文学习对候选模型进行排序，并提供透明化的理由。同时，研究团队提出了一个包含75个专家验证的RS查询场景的基准测试。

### 主要发现

REMSA在多个基线方法上表现更优，包括朴素代理、密集检索和非结构化RAG-based LLM。该方法完全基于公开可用的元数据运行，不访问私有或敏感数据。研究团队通过专家为中心的评估协议，测试了900种配置。

### 结论

REMSA和RS-FMD为遥感领域提供了一种高效、透明的遥感基础模型选择方法，有助于推动遥感基础模型的应用和研究。

### 翻译

基础模型(FMs)越来越多地用于遥感(RS)领域，用于环境监测、灾害评估和土地利用制图等任务。这些模型包括在单一数据模态上训练的单模态视觉编码器以及在SAR、多光谱、高光谱和图像文本数据组合上训练的多模态架构。它们支持多种遥感任务，包括语义分割、图像分类、变化检测和视觉问答。然而，由于文档分散、格式异构和部署约束各异，选择合适的遥感基础模型(RSFM)仍然很困难。我们引入了RSFM数据库(RS-FMD)，这是一个结构化资源，涵盖150多个跨越多种数据模态、分辨率和学习范式的RSFM。基于RS-FMD，我们提出了REMSA，这是首个基于大语言模型(LLM)的代理，用于从自然语言查询中自动选择RSFM。REMSA解释用户需求，解决缺失约束，使用上下文学习对候选模型进行排序，并提供透明化的理由。我们还提出了一个包含75个专家验证的RS查询场景的基准测试，在以专家为中心的评估协议下产生了900种配置。REMSA在多个基线上表现更优，包括朴素代理、密集检索和非结构化RAG-based LLM。它完全基于公开可用的元数据运行，不访问私有或敏感数据。


### 论文摘要

Foundation Models (FMs) are increasingly used in remote sensing (RS) for tasks such as environmental monitoring, disaster assessment, and land-use mapping. These models include unimodal vision encoders trained on a single data modality and multimodal architectures trained on combinations of SAR, multispectral, hyperspectral, and image-text data. They support diverse RS tasks including semantic segmentation, image classification, change detection, and visual question answering. However, selecting an appropriate remote sensing foundation model (RSFM) remains difficult due to scattered documentation, heterogeneous formats, and varied deployment constraints. We introduce the RSFM Database (RS-FMD), a structured resource covering over 150 RSFMs spanning multiple data modalities, resolutions, and learning paradigms. Built on RS-FMD, we present REMSA, the first LLM-based agent for automated RSFM selection from natural language queries. REMSA interprets user requirements, resolves missing constraints, ranks candidate models using in-context learning, and provides transparent justifications. We also propose a benchmark of 75 expert-verified RS query scenarios, producing 900 configurations under an expert-centered evaluation protocol. REMSA outperforms several baselines, including naive agents, dense retrieval, and unstructured RAG-based LLMs. It operates entirely on publicly available metadata and does not access private or sensitive data.

---

## 27. Semantic and Semiotic Interplays in Text-to-Audio AI: Exploring Cognitive Dynamics and Musical Interactions

**论文链接:** [http://arxiv.org/abs/2511.17429v1](http://arxiv.org/abs/2511.17429v1)

**作者:** Guilherme Coelho

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文探讨了人工智能中新兴的文本到音频范式，分析其对音乐创作、诠释和认知的变革性影响，研究AI系统如何重新配置音乐符号化过程并建立认知框架。

### 背景

文本到音频AI技术作为人工智能领域的新兴范式，能够将自然语言描述转化为声音对象，正在改变音乐创作和聆听的方式。

### 目的

探索文本到音频AI系统如何影响音乐的创作、诠释和认知过程，以及这些系统如何重新配置音乐符号化过程并影响听众的聆听方式和音乐认知。

### 方法

采用结构主义和后结构主义理论框架，结合认知理论中的图式动态和元认知理论，以Udio为主要案例研究，分析AI中介的音乐活动中的认知动态。

### 主要发现

文本到音频AI模型作为音乐符号化的准客体，既稳定又颠覆传统形式，促进新的聆听方式和审美反思；这些模型在语言提示和声音输出之间的过渡空间中导航，产生新的音乐表达并促使批判性聆听。

### 结论

文本到音频AI模型可作为认识论工具和准客体，促进音乐互动的重大转变，邀请用户发展对音乐认知和文化基础更细微的理解。

### 翻译

这篇论文探讨了人工智能中新兴的文本到音频范式，考察其对音乐创作、诠释和认知的变革性影响。研究探索了描述性自然语言提示在文本到音频模态中被转化为细微声音对象时发生的复杂语义和符号互动。借鉴结构主义和后结构主义视角，以及图式动态和元认知的认知理论，本文探讨了这些AI系统如何重新配置音乐符号化过程并建立既定的认知框架。研究分析了AI中介音乐活动中的一些认知动态，包括图式同化和适应、元认知反思和建构性感知的过程。本文认为，文本到音频AI模型作为音乐符号化的准客体，同时稳定和颠覆传统形式，同时促进新的聆听方式和审美反思。以Udio为主要案例研究，本研究探讨了这些模型如何在语言提示和声音输出之间的过渡空间中导航。这一过程不仅产生新的音乐表达，还促使听众进行批判性和'结构化感知'的聆听形式，鼓励更深入地理解音乐结构、符号细微差别以及塑造我们音乐认知的社会文化背景。


### 论文摘要

This paper investigates the emerging text-to-audio paradigm in artificial intelligence (AI), examining its transformative implications for musical creation, interpretation, and cognition. I explore the complex semantic and semiotic interplays that occur when descriptive natural language prompts are translated into nuanced sound objects across the text-to-audio modality. Drawing from structuralist and post-structuralist perspectives, as well as cognitive theories of schema dynamics and metacognition, the paper explores how these AI systems reconfigure musical signification processes and navigate established cognitive frameworks. The research analyzes some of the cognitive dynamics at play in AI-mediated musicking, including processes of schema assimilation and accommodation, metacognitive reflection, and constructive perception. The paper argues that text-to-audio AI models function as quasi-objects of musical signification, simultaneously stabilizing and destabilizing conventional forms while fostering new modes of listening and aesthetic reflexivity.Using Udio as a primary case study, this study explores how these models navigate the liminal spaces between linguistic prompts and sonic outputs. This process not only generates novel musical expressions but also prompts listeners to engage in forms of critical and "structurally-aware listening.", encouraging a deeper understanding of music's structures, semiotic nuances, and the socio-cultural contexts that shape our musical cognition. The paper concludes by reflecting on the potential of text-to-audio AI models to serve as epistemic tools and quasi-objects, facilitating a significant shift in musical interactions and inviting users to develop a more nuanced comprehension of the cognitive and cultural foundations of music.

---

## 28. Preventing Shortcut Learning in Medical Image Analysis through Intermediate Layer Knowledge Distillation from Specialist Teachers

**论文链接:** [http://arxiv.org/abs/2511.17421v1](http://arxiv.org/abs/2511.17421v1)

**作者:** Christopher Boland, Sotirios Tsaftaris, Sonia Dahdouh

**发布时间:** 2025-11-21

**DOI:** 10.59275/j.melba.2025-8888

**备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) https://melba-journal.org/2025:020

### GPT解析

### 总结

该研究提出了一种新颖的知识蒸馏框架，用于减轻深度学习模型中的捷径学习问题，特别是在医学图像分析等高风险应用中。

### 背景

深度学习模型倾向于使用训练数据中虚假相关但不相关的特征来学习问题的捷径解决方案。在医学图像分析等高风险应用中，这种现象可能阻止模型使用临床上有意义的特征进行预测，导致鲁棒性差并可能对患者造成伤害。

### 目的

证明不同类型的捷径在网络层中有不同的表现，并提出一种通过知识蒸馏框架减轻捷径学习的方法。

### 方法

提出了一种新颖的知识蒸馏框架，利用在少量任务相关数据上微调的教师网络来减轻在大型数据集上训练的学生网络中的捷径学习，这些数据集被偏差特征所污染。

### 主要发现

在CheXpert、ISIC 2017和SimBA数据集上使用多种架构进行的实验表明，该方法相比传统方法有持续改进，在许多情况下即使是在分布外测试数据上也能达到与无偏差数据训练的基线模型相当的性能。

### 结论

该方法在实际医学成像场景中具有实际适用性，特别是在偏差标注有限且捷径特征难以预先识别的情况下。

### 翻译

深度学习模型倾向于使用训练数据中虚假相关但不相关的特征来学习问题的捷径解决方案。在高风险应用如医学图像分析中，这种现象可能阻止模型在做出预测时使用临床上有意义的特征，可能导致鲁棒性差并对患者造成伤害。我们证明不同类型的捷径（在整个图像中分散的捷径以及集中在特定区域的捷径）在网络层中有不同的表现，因此可以通过针对中间层的缓解策略更有效地针对它们。我们提出了一种新颖的知识蒸馏框架，利用在少量任务相关数据上微调的教师网络来减轻在大型数据集上训练的学生网络中的捷径学习，这些数据集被偏差特征所污染。在CheXpert、ISIC 2017和SimBA数据集上使用多种架构（ResNet-18、AlexNet、DenseNet-121和3D CNN）进行的广泛实验中，我们证明了与传统经验风险最小化、基于增强的偏差缓解和基于组的偏差缓解方法相比，我们的方法有持续改进。在许多情况下，即使是在分布外测试数据上，我们也实现了与在无偏差数据上训练的基线模型相当的性能。我们的结果表明，我们的方法在实际医学成像场景中具有实际适用性，在这些场景中偏差标注有限且捷径特征难以预先识别。


### 论文摘要

Deep learning models are prone to learning shortcut solutions to problems using spuriously correlated yet irrelevant features of their training data. In high-risk applications such as medical image analysis, this phenomenon may prevent models from using clinically meaningful features when making predictions, potentially leading to poor robustness and harm to patients. We demonstrate that different types of shortcuts (those that are diffuse and spread throughout the image, as well as those that are localized to specific areas) manifest distinctly across network layers and can, therefore, be more effectively targeted through mitigation strategies that target the intermediate layers. We propose a novel knowledge distillation framework that leverages a teacher network fine-tuned on a small subset of task-relevant data to mitigate shortcut learning in a student network trained on a large dataset corrupted with a bias feature. Through extensive experiments on CheXpert, ISIC 2017, and SimBA datasets using various architectures (ResNet-18, AlexNet, DenseNet-121, and 3D CNNs), we demonstrate consistent improvements over traditional Empirical Risk Minimization, augmentation-based bias-mitigation, and group-based bias-mitigation approaches. In many cases, we achieve comparable performance with a baseline model trained on bias-free data, even on out-of-distribution test data. Our results demonstrate the practical applicability of our approach to real-world medical imaging scenarios where bias annotations are limited and shortcut features are difficult to identify a priori.

---

## 29. Momentum-Resolved Electronic Structure and Orbital Hybridization in the Layered Antiferromagnet CrPS$_4$

**论文链接:** [http://arxiv.org/abs/2511.17403v1](http://arxiv.org/abs/2511.17403v1)

**作者:** Lasse Sternemann, David Maximilian Janas, Eshan Banerjee, Richard Leven, Jonah Elias Nitschke, Marco Marino, Leon Becker, Ahmet Can Ademoğlu, Frithjof Anders, Stefan Tappertzhofen, Mirko Cinchetti

**发布时间:** 2025-11-21

### GPT解析

### 总结

铬硫磷酸盐(CrPS₄)是一种层状二维反铁磁半导体，具有有趣的自旋电子学和磁光特性，但其电子能带结构尚未通过实验表征。

### 背景

铬硫磷酸盐(CrPS₄)作为一种层状二维反铁磁半导体，展现出引人注目的自旋电子学和磁光特性，但其电子能带结构一直缺乏实验表征。

### 目的

揭示铬硫磷酸盐的电子能带结构，特别是其价带特性和带隙类型，以理解其磁性和光学性质的基础。

### 方法

使用动量分辨的光电子能谱技术(在Néel温度上下进行测量)，并结合带有Hubbard U修正的密度泛函理论(DFT+U)计算进行分析。

### 主要发现

价带主要由铬的3d态和硫的3p态组成，存在配体到金属的电荷转移带隙；弱杂化的t₂g轨道负责磁有序，而强杂化的e_g轨道使偶极选择规则松弛，从而能够实现光活性的轨道跃迁。

### 结论

这些发现建立了对铬硫磷酸盐电子结构的基础理解，为理论模型提供了基准，并为其轨道物理研究和潜在器件应用提供了指导。

### 翻译

铬硫磷酸盐(CrPS₄)是一种层状二维反铁磁半导体，展现出引人注目的自旋电子学和磁光特性，但其电子能带结构一直缺乏实验表征。本文采用动量分辨的光电子能谱技术(在Néel温度上下进行)，并结合带有Hubbard U修正的密度泛函理论(DFT+U)，揭示了主要由铬3d态和硫3p态组成的价带，以及配体到金属的电荷转移带隙。我们识别出负责磁有序的弱杂化t₂g轨道和使偶极选择规则松弛从而实现光活性轨道跃迁的强杂化e_g轨道。这些发现建立了对铬硫磷酸盐电子结构的基础理解，为理论模型提供了基准，并为其轨道物理研究和潜在器件应用提供了信息。


### 论文摘要

Chromium thiophosphate (CrPS$_4$) is a layered two-dimensional antiferromagnetic semiconductor exhibiting intriguing spintronic and magneto-optical properties, yet its electronic band structure has remained experimentally uncharacterized. Here, we employ momentum-resolved photoemission spectroscopy above and below the Néel temperature, complemented by density functional theory with Hubbard U corrections (DFT+U), to reveal a valence band dominated by Cr $3d$ and S $3p$ states with a ligand-to-metal charge-transfer band gap. We identify weakly hybridized t$_{2g}$ orbitals responsible for magnetic ordering and strongly hybridized e$_{g}$ orbitals that relax dipole selection rules, enabling optically active orbital transitions. These findings establish a foundational understanding of CrPS$_4$'s electronic structure, providing a benchmark for theoretical models and informing future investigations into its orbital physics and potential device applications.

---

## 30. Sparse Mixture-of-Experts for Multi-Channel Imaging: Are All Channel Interactions Required?

**论文链接:** [http://arxiv.org/abs/2511.17400v1](http://arxiv.org/abs/2511.17400v1)

**作者:** Sukwon Yun, Heming Yao, Burkhard Hoeckendorf, David Richmond, Aviv Regev, Russell Littman

**发布时间:** 2025-11-21

**备注:** This has been accepted at the NeurIPS AI4Science Workshop 2025

### GPT解析

### 总结

本文提出了一种名为MoE-ViT的新型架构，用于解决多通道图像处理中的效率问题，通过专家混合方法显著减少了计算复杂度同时保持了或提高了性能。

### 背景

Vision Transformers已成为视觉基础模型的主干架构，但在多通道领域(如细胞绘画或卫星图像)的优化研究不足。这些领域的关键挑战是捕捉通道间的相互作用，因为每个通道携带不同信息。

### 目的

从有效性转向跨通道注意力中被忽视的效率挑战，探究是否需要建模所有通道交互，以解决现有方法中通道比较导致的计算瓶颈问题。

### 方法

受稀疏专家混合(MoE)理念启发，提出MoE-ViT架构，将每个通道视为一个专家，并使用轻量级路由器为每个图像补丁选择最相关的专家进行注意力计算，避免不必要的通道间交互。

### 主要发现

在JUMP-CP和So2Sat真实世界数据集上的实验证明，MoE-ViT实现了显著的效率提升，没有牺牲性能，在某些情况下甚至增强了性能表现。

### 结论

MoE-ViT是多通道成像领域的一个实用且有吸引力的主干架构，有效平衡了计算效率和模型性能。

### 翻译

视觉变换器(ViTs)已成为视觉基础模型的主干架构，然而它们在多通道领域(如细胞绘画或卫星图像)的优化仍研究不足。这些领域的一个关键挑战是捕捉通道间的相互作用，因为每个通道携带不同的信息。虽然现有工作通过在标记化过程中独立处理每个通道已显示出有效性，但这种方法在注意力块中自然引入了主要的计算瓶颈—通道间的比较导致注意力计算呈二次增长，产生过多的FLOPs和高训练成本。在这项工作中，我们将重点从有效性转向跨通道注意力中被忽视的效率挑战，并提问：'是否需要建模所有通道交互？'受稀疏专家混合(MoE)理念启发，我们提出了MoE-ViT，这是一种用于ViTs中多通道图像的专家混合架构，它将每个通道视为一个专家，并使用轻量级路由器为每个补丁选择最相关的专家进行注意力计算。在JUMP-CP和So2Sat真实世界数据集上的概念验证实验表明，MoE-ViT实现了显著的效率提升，没有牺牲性能，在某些情况下甚至增强了性能，使其成为多通道成像的一个实用且有吸引力的主干架构。


### 论文摘要

Vision Transformers ($\text{ViTs}$) have become the backbone of vision foundation models, yet their optimization for multi-channel domains - such as cell painting or satellite imagery - remains underexplored. A key challenge in these domains is capturing interactions between channels, as each channel carries different information. While existing works have shown efficacy by treating each channel independently during tokenization, this approach naturally introduces a major computational bottleneck in the attention block - channel-wise comparisons leads to a quadratic growth in attention, resulting in excessive $\text{FLOPs}$ and high training cost. In this work, we shift focus from efficacy to the overlooked efficiency challenge in cross-channel attention and ask: "Is it necessary to model all channel interactions?". Inspired by the philosophy of Sparse Mixture-of-Experts ($\text{MoE}$), we propose MoE-ViT, a Mixture-of-Experts architecture for multi-channel images in $\text{ViTs}$, which treats each channel as an expert and employs a lightweight router to select only the most relevant experts per patch for attention. Proof-of-concept experiments on real-world datasets - JUMP-CP and So2Sat - demonstrate that $\text{MoE-ViT}$ achieves substantial efficiency gains without sacrificing, and in some cases enhancing, performance, making it a practical and attractive backbone for multi-channel imaging.

---

## 31. Designing and Generating Diverse, Equitable Face Image Datasets for Face Verification Tasks

**论文链接:** [http://arxiv.org/abs/2511.17393v1](http://arxiv.org/abs/2511.17393v1)

**作者:** Georgia Baltsou, Ioannis Sarridis, Christos Koutlis, Symeon Papadopoulos

**发布时间:** 2025-11-21

### GPT解析

### 总结

研究提出了一种解决人脸验证数据集偏差问题的方法，通过生成多样化的合成人脸图像，并创建了DIF-V数据集作为基准，发现现有模型存在性别和种族偏差，身份样式修改会降低模型性能。

### 背景

人脸验证是身份认证的重要组件，应用于网上银行和个人设备安全访问等场景，但现有人脸图像数据集存在种族、性别和其他人口统计特征的显著偏差，限制了人脸验证系统的有效性和公平性。

### 目的

提出一种综合方法，通过整合先进的生成模型创建多样化、高质量的人脸合成图像，确保代表多样化的面部特征并遵守身份证件照片允许的特征，解决现有数据集中的固有不平等问题。

### 方法

整合先进的生成模型创建多样化、高质量的人脸合成图像，强调代表多样化的面部特征并确保符合身份证件照片的要求，引入了包含27,780张图像、涉及926个独特身份的DIF-V数据集作为未来人脸验证研究的基准。

### 主要发现

现有的验证模型对某些性别和种族存在偏差，应用身份样式修改会对模型性能产生负面影响。

### 结论

这项工作丰富了人工智能中多样性和伦理的讨论，为开发更具包容性和可靠性的人脸验证技术奠定了基础。

### 翻译

人脸验证是网上银行和个人设备安全访问等各种应用中身份认证的重要组成部分。现有的大多数人脸图像数据集通常存在与种族、性别和其他人口统计特征相关的显著偏差，限制了人脸验证系统的有效性和公平性。为应对这些挑战，我们提出了一种综合方法，整合先进的生成模型来创建多样化且高质量的人脸合成图像。该方法强调代表多样化的面部特征，确保遵守身份证件照片允许的特征。此外，我们引入了用于验证的多样化包容性人脸（DIF-V）数据集，包含27,780张图像，涉及926个独特身份，旨在作为未来人脸验证研究的基准。我们的分析显示，现有的验证模型对某些性别和种族存在偏差，并且应用身份样式修改会对模型性能产生负面影响。通过解决现有数据集中的固有不平等问题，这项工作不仅丰富了人工智能中多样性和伦理的讨论，也为开发更具包容性和可靠性的人脸验证技术奠定了基础。


### 论文摘要

Face verification is a significant component of identity authentication in various applications including online banking and secure access to personal devices. The majority of the existing face image datasets often suffer from notable biases related to race, gender, and other demographic characteristics, limiting the effectiveness and fairness of face verification systems. In response to these challenges, we propose a comprehensive methodology that integrates advanced generative models to create varied and diverse high-quality synthetic face images. This methodology emphasizes the representation of a diverse range of facial traits, ensuring adherence to characteristics permissible in identity card photographs. Furthermore, we introduce the Diverse and Inclusive Faces for Verification (DIF-V) dataset, comprising 27,780 images of 926 unique identities, designed as a benchmark for future research in face verification. Our analysis reveals that existing verification models exhibit biases toward certain genders and races, and notably, applying identity style modifications negatively impacts model performance. By tackling the inherent inequities in existing datasets, this work not only enriches the discussion on diversity and ethics in artificial intelligence but also lays the foundation for developing more inclusive and reliable face verification technologies

---

## 32. IndustryNav: Exploring Spatial Reasoning of Embodied Agents in Dynamic Industrial Navigation

**论文链接:** [http://arxiv.org/abs/2511.17384v1](http://arxiv.org/abs/2511.17384v1)

**作者:** Yifan Li, Lichi Li, Anh Dao, Xinyu Zhou, Yicheng Qiao, Zheda Mai, Daeun Lee, Zichen Chen, Zhen Tan, Mohit Bansal, Yu Kong

**发布时间:** 2025-11-21

### GPT解析

### 总结

这篇论文介绍了IndustryNav，第一个用于评估视觉大语言模型作为具身智能体在动态环境中空间推理能力的工业导航基准测试。

### 背景

现有的具身智能基准测试主要关注静态家庭环境和孤立能力评估，无法捕捉动态、真实世界复杂性中的整体性能。视觉大语言模型在空间推理方面仍面临重大挑战。

### 目的

填补现有基准测试空白，提出IndustryNav作为第一个用于主动空间推理的动态工业导航基准测试。

### 方法

创建12个高保真Unity仓库场景，包含动态对象和人体移动；采用PointGoal导航流水线结合以自我为中心的视觉和全局里程计；引入'碰撞率'和'警告率'指标衡量安全行为和距离估计；对九个先进VLLMs进行综合研究。

### 主要发现

闭源模型保持一致优势，但所有智能体在稳健路径规划、碰撞避免和主动探索方面都表现出明显缺陷。

### 结论

具身研究需要超越被动感知，转向要求在动态、真实环境中进行稳定规划、主动探索和安全行为的任务。

### 翻译

虽然视觉大语言模型作为具身智能体展现出巨大潜力，但它们在空间推理方面继续面临重大挑战。现有的具身基准测试主要关注被动、静态的家庭环境，仅评估孤立能力，无法捕捉在动态、真实世界复杂性中的整体性能。为了填补这一空白，我们提出了IndustryNav，这是第一个用于主动空间推理的动态工业导航基准测试。IndustryNav利用12个手动创建的高保真Unity仓库场景，其中包含动态对象和人体移动。我们的评估采用PointGoal导航流水线，有效结合以自我为中心的视觉和全局里程计来评估整体局部-全局规划。关键的是，我们引入了'碰撞率'和'警告率'指标来衡量安全导向的行为和距离估计。对九个最先进的VLLMs（包括GPT-5-mini、Claude-4.5和Gemini-2.5等模型）的综合研究表明，闭源模型保持一致的优势；然而，所有智能体在稳健路径规划、碰撞避免和主动探索方面都表现出明显的缺陷。这突显了具身研究需要超越被动感知，转向要求在动态、真实环境中进行稳定规划、主动探索和安全行为的任务的迫切需求。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉大语言模型(VLLMs)作为具身智能体在动态工业环境中的空间推理能力不足的问题。现有基准测试主要关注静态家庭环境和孤立能力评估，无法准确评估智能体在复杂动态环境中的整体表现。这个问题很重要，因为工业导航等实际应用需要在动态环境中进行稳定的规划、主动探索和安全行为，而当前模型在这些方面表现欠佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有空间推理基准的两个关键局限性：静态环境和孤立能力评估。为解决这些问题，他们设计了一个基于Unity的动态工业导航环境，包含12个手动创建的仓库场景，引入动态对象和人类活动。他们借鉴了PointGoal导航的概念，结合以自我为中心的视觉和全局里程计信息，并参考了现有视觉语言模型在空间推理方面的研究方法。此外，还参考了智能体导航的研究，包括指令跟随、探索基础和结构化推理等范式。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个动态工业导航环境，评估具身智能体在主动空间推理方面的能力，特别是在处理动态障碍物和复杂空间关系时的表现。整体实现流程包括：1)使用Unity创建包含动态对象的12个仓库场景；2)设计PointGoal导航管道，结合局部以自我为中心的图像和全局里程计信息；3)提供当前状态信息让智能体选择动作；4)使用五个指标(任务成功率、距离率、平均步数、碰撞率和警告率)评估性能。每个场景测试4个不同难度的起点-目标点对，每次运行70步，智能体每步输出动作和推理过程。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)第一个动态工业导航基准，包含12个高保真Unity仓库场景；2)综合评估方法，结合局部和全局信息评估整体规划能力；3)新的安全评估指标(碰撞率和警告率)；4)对9个先进VLLMs的全面实验评估。相比之前工作，IndustryNav的最大不同是引入了动态对象和环境，提供了对碰撞意识、动态障碍物处理等安全行为的综合评估，而现有基准主要关注静态家庭环境和孤立能力评估。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IndustryNav提出了第一个动态工业导航基准，通过创新的评估指标和全面的实验分析，揭示了当前视觉大语言模型在动态环境中的空间推理局限性，并为具身智能体在复杂工业环境中的导航研究指明了方向。'}


### 论文摘要

While Visual Large Language Models (VLLMs) show great promise as embodied agents, they continue to face substantial challenges in spatial reasoning. Existing embodied benchmarks largely focus on passive, static household environments and evaluate only isolated capabilities, failing to capture holistic performance in dynamic, real-world complexity. To fill this gap, we present IndustryNav, the first dynamic industrial navigation benchmark for active spatial reasoning. IndustryNav leverages 12 manually created, high-fidelity Unity warehouse scenarios featuring dynamic objects and human movement. Our evaluation employs a PointGoal navigation pipeline that effectively combines egocentric vision with global odometry to assess holistic local-global planning. Crucially, we introduce the "collision rate" and "warning rate" metrics to measure safety-oriented behaviors and distance estimation. A comprehensive study of nine state-of-the-art VLLMs (including models such as GPT-5-mini, Claude-4.5, and Gemini-2.5) reveals that closed-source models maintain a consistent advantage; however, all agents exhibit notable deficiencies in robust path planning, collision avoidance and active exploration. This highlights a critical need for embodied research to move beyond passive perception and toward tasks that demand stable planning, active exploration, and safe behavior in dynamic, real-world environment.

---

## 33. Exploring Scientific Debt: Harnessing AI for SATD Identification in Scientific Software

**论文链接:** [http://arxiv.org/abs/2511.17368v1](http://arxiv.org/abs/2511.17368v1)

**作者:** Eric L. Melin, Ahmed Musa Awon, Nasir U. Eisty, Neil A. Ernst, Shurui Zhou

**发布时间:** 2025-11-21

**备注:** 11 pages, 2 figures, 6 tables

### GPT解析

### 总结

本研究探讨了科学软件中的自承认技术债务(SATD)，发现科学软件中的SATD比通用软件高出4.93倍，研究还评估了基于transformer的模型用于SATD识别，并提出了管理科学软件中技术债务的策略。

### 背景

开发者在代码中经常会留下线索承认代码不足，形成自承认技术债务(SATD)。在科学软件领域，这种债务不仅常见而且影响深远，可能威胁科学发现的基础，但SATD与科学软件的关系尚未被充分探索。

### 目的

探索科学软件仓库中的SATD，比较科学软件与通用开源软件中的SATD差异，并评估基于transformer的模型用于SATD识别的效果。

### 方法

分析27个科学和通用软件仓库中的SATD(涵盖多个领域和语言)，在67,066个标记的代码注释上微调和比较10个基于transformer的模型(参数量从100M到7B)。

### 主要发现

科学软件包含比通用软件多9.25倍的科学债务和4.93倍的SATD，这源于复杂计算、领域约束和不断发展的研究需求；研究开发的最佳模型性能优于现有模型。

### 结论

科学软件中的SATD与通用软件存在显著差异，影响软件质量和科学有效性；认识到这些挑战有助于开发者和研究人员采用更智能的策略管理债务，保护科学发现的完整性。

### 翻译

开发人员经常在他们的代码中留下线索，承认其不足之处，这被称为自承认技术债务(SATD)。在科学软件(SSW)领域，创新快速且协作是关键，这种债务不仅常见而且影响深远。由于研究依赖于准确和可重复的结果，积累的SATD可能威胁科学发现的基础。然而，尽管其重要性，SATD与SSW之间的关系在很大程度上仍未被探索，留下了理解如何在关键领域管理SATD的重要空白。本研究探讨了SSW仓库中的SATD，比较了科学软件与通用开源软件中的SATD，并评估了基于transformer的模型用于SATD识别。我们分析了多个领域和语言中27个科学和通用仓库中的SATD。我们在67,066个标记的代码注释上微调和比较了10个基于transformer的模型(参数量从100M到7B)。由于复杂计算、领域约束和不断发展的研究需求，SSW包含比通用软件多9.25倍的科学债务和4.93倍的SATD。此外，我们的最佳模型优于现有模型。本研究揭示了SSW中的SATD与通用软件的不同之处，揭示了其对质量和科学有效性的影响。通过认识到这些挑战，开发者和研究人员可以采用更智能的策略来管理债务，保护科学发现的完整性。


### 论文摘要

Developers often leave behind clues in their code, admitting where it falls short, known as Self-Admitted Technical Debt (SATD). In the world of Scientific Software (SSW), where innovation moves fast and collaboration is key, such debt is not just common but deeply impactful. As research relies on accurate and reproducible results, accumulating SATD can threaten the very foundations of scientific discovery. Yet, despite its significance, the relationship between SATD and SSW remains largely unexplored, leaving a crucial gap in understanding how to manage SATD in this critical domain. This study explores SATD in SSW repositories, comparing SATD in scientific versus general-purpose open-source software and evaluating transformer-based models for SATD identification. We analyzed SATD in 27 scientific and general-purpose repositories across multiple domains and languages. We fine-tuned and compared 10 transformer-based models (100M-7B parameters) on 67,066 labeled code comments. SSW contains 9.25x more Scientific Debt and 4.93x more SATD than general-purpose software due to complex computations, domain constraints, and evolving research needs. Furthermore, our best model outperforms existing ones. This study uncovers how SATD in SSW differs from general software, revealing its impact on quality and scientific validity. By recognizing these challenges, developers and researchers can adopt smarter strategies to manage debt and safeguard the integrity of scientific discovery.

---

## 34. METIS: Multi-Source Egocentric Training for Integrated Dexterous Vision-Language-Action Model

**论文链接:** [http://arxiv.org/abs/2511.17366v1](http://arxiv.org/abs/2511.17366v1)

**作者:** Yankai Fu, Ning Chen, Junkai Zhao, Shaozhe Shan, Guocai Yao, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为METIS的视觉-语言-动作模型，通过整合多源人类和机器人数据解决了灵巧操作中数据稀缺的问题，实现了在多样化任务中的高效部署和优异表现。

### 背景

构建能够感知、推理和执行多样化任务的通用机器人仍然是一个开放挑战，特别是对于灵巧操作。主要瓶颈在于缺乏大规模、带有动作标注的灵巧技能数据，因为远程操作既困难又昂贵。

### 目的

利用人类数据为机器人学习提供丰富的先验知识，解决灵巧操作中数据稀缺的问题，开发一个能够有效部署到多样化灵巧操作任务的通用模型。

### 方法

提出了METIS，一个在多源第一人称数据集上预训练的用于灵巧操作的视觉-语言-动作模型。构建了EgoAtlas，整合了来自多个来源的大规模人类和机器人数据，并在统一的动作空间下统一。提取了运动感知动力学，这是一种紧凑和离散化的运动表示，为VLA训练提供了高效且富有表现力的监督。METIS将推理和行动整合到一个统一的框架中。

### 主要发现

该方法展示了出色的灵巧操作能力，在六个现实世界任务中取得了最高的平均成功率。实验结果还突显了其在分布外场景中的优越泛化能力和鲁棒性。

### 结论

METIS是迈向灵巧操作通用模型的有希望的一步，为解决灵巧操作中的数据稀缺和泛化问题提供了有效方案。

### 翻译

构建能够感知、推理和执行多样化任务的通用机器人仍然是一个开放挑战，特别是对于灵巧操作。主要瓶颈在于缺乏大规模、带有动作标注的灵巧技能数据，因为远程操作既困难又昂贵。人类数据以其大规模和多样化的操作行为，为学习机器人动作提供了丰富的先验知识。虽然先前的工作已经探索了利用人类演示，但它们通常受到场景有限以及人类和机器人之间巨大视觉差距的限制。为了消除这些限制，我们提出了METIS，一个在多源第一人称数据集上预训练的用于灵巧操作的视觉-语言-动作模型。我们首先构建了EgoAtlas，它整合了来自多个来源的大规模人类和机器人数据，所有数据都在统一的动作空间下统一。我们进一步提取了运动感知动力学，这是一种紧凑和离散化的运动表示，为VLA训练提供了高效且富有表现力的监督。基于这些，METIS将推理和行动整合到一个统一的框架中，能够有效部署到下游的灵巧操作任务。我们的方法展示了出色的灵巧操作能力，在六个现实世界任务中取得了最高的平均成功率。实验结果也突显了其在分布外场景中的优越泛化能力和鲁棒性。这些发现强调了METIS作为迈向灵巧操作通用模型的有希望的一步。


### 论文摘要

Building a generalist robot that can perceive, reason, and act across diverse tasks remains an open challenge, especially for dexterous manipulation. A major bottleneck lies in the scarcity of large-scale, action-annotated data for dexterous skills, as teleoperation is difficult and costly. Human data, with its vast scale and diverse manipulation behaviors, provides rich priors for learning robotic actions. While prior works have explored leveraging human demonstrations, they are often constrained by limited scenarios and a large visual gap between human and robots. To eliminate these limitations, we propose METIS, a vision-language-action (VLA) model for dexterous manipulation pretrained on multi-source egocentric datasets. We first construct EgoAtlas, which integrates large-scale human and robotic data from multiple sources, all unified under a consistent action space. We further extract motion-aware dynamics, a compact and discretized motion representation, which provides efficient and expressive supervision for VLA training. Built upon them, METIS integrates reasoning and acting into a unified framework, enabling effective deployment to downstream dexterous manipulation tasks. Our method demonstrates exceptional dexterous manipulation capabilities, achieving highest average success rate in six real-world tasks. Experimental results also highlight the superior generalization and robustness to out-of-distribution scenarios. These findings emphasize METIS as a promising step toward a generalist model for dexterous manipulation.

---

## 35. UAM: A Unified Attention-Mamba Backbone of Multimodal Framework for Tumor Cell Classification

**论文链接:** [http://arxiv.org/abs/2511.17355v1](http://arxiv.org/abs/2511.17355v1)

**作者:** Taixi Chen, Jingyun Chen, Nancy Guo

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究引入了一种统一的注意力-Mamba（UAM）骨干网络，用于细胞级别的放射组学特征分析，显著提高了肿瘤诊断的准确性和AI的可解释性。

### 背景

细胞水平放射组学特征可提供肿瘤表型的精细见解，提高H&E图像诊断准确性，但现有研究多集中于幻灯片或补丁级别，缺乏专门为放射组学数据设计的骨干网络。

### 目的

开发一种统一的骨干网络架构，用于细胞级别的放射组学特征分析，提高肿瘤诊断的准确性和AI的可解释性。

### 方法

受Mamba架构启发，设计了统一的注意力-Mamba（UAM）骨干网络，灵活结合注意力和Mamba模块的能力，开发了两种UAM变体，并构建了多模态UAM框架联合执行细胞分类和图像分割。

### 主要发现

UAM在公共基准上取得了最先进性能，细胞分类准确率从74%提高到78%（n=349,882个细胞），肿瘤分割精度从75%提高到80%（n=406个补丁）。

### 结论

UAM作为放射组学驱动的癌症诊断的统一且可扩展的多模态基础具有有效性和广阔前景。

### 翻译

细胞水平的放射组学特征为肿瘤表型提供精细化的见解，并有可能显著提高苏木精和伊红（H&E）图像上的诊断准确性。通过捕捉微观形态和强度模式，这些特征支持更精确的肿瘤识别，并通过突出显示对病理学家审查具有诊断相关性的细胞，提高AI的可解释性。然而，大多数现有研究专注于幻灯片级别或补丁级别的肿瘤分类，细胞水平的放射组学分析在很大程度上仍未被探索。此外，目前还没有专门为放射组学数据设计的骨干网络。受Mamba架构在视觉和语言领域最近成功的启发，我们引入了一个统一的注意力-Mamba（UAM）骨干网络，用于使用放射组学特征进行细胞级别分类。与之前以固定比例集成注意力和Mamba模块的混合方法不同，我们的统一设计在单一连贯的架构中灵活结合它们的能力，消除了手动比例调整的需要，提高了编码能力。我们开发了两种UAM变体，以全面评估这种统一结构的好处。基于此骨干网络，我们进一步提出了一个多模态UAM框架，联合执行细胞级别分类和图像分割。实验结果表明，UAM在公共基准上的两项任务中都取得了最先进的性能，超越了领先的基于图像的基础模型。它将细胞分类准确率从74%提高到78%（n=349,882个细胞），肿瘤分割精度从75%提高到80%（n=406个补丁）。这些发现强调了UAM作为放射组学驱动的癌症诊断的统一且可扩展的多模态基础的有效性和前景。


### 论文摘要

Cell-level radiomics features provide fine-grained insights into tumor phenotypes and have the potential to significantly enhance diagnostic accuracy on hematoxylin and eosin (H&E) images. By capturing micro-level morphological and intensity patterns, these features support more precise tumor identification and improve AI interpretability by highlighting diagnostically relevant cells for pathologist review. However, most existing studies focus on slide-level or patch-level tumor classification, leaving cell-level radiomics analysis largely unexplored. Moreover, there is currently no dedicated backbone specifically designed for radiomics data. Inspired by the recent success of the Mamba architecture in vision and language domains, we introduce a Unified Attention-Mamba (UAM) backbone for cell-level classification using radiomics features. Unlike previous hybrid approaches that integrate Attention and Mamba modules in fixed proportions, our unified design flexibly combines their capabilities within a single cohesive architecture, eliminating the need for manual ratio tuning and improving encode capability. We develop two UAM variants to comprehensively evaluate the benefits of this unified structure. Building on this backbone, we further propose a multimodal UAM framework that jointly performs cell-level classification and image segmentation. Experimental results demonstrate that UAM achieves state-of-the-art performance across both tasks on public benchmarks, surpassing leading image-based foundation models. It improves cell classification accuracy from 74% to 78% ($n$=349,882 cells), and tumor segmentation precision from 75% to 80% ($n$=406 patches). These findings highlight the effectiveness and promise of UAM as a unified and extensible multimodal foundation for radiomics-driven cancer diagnosis.

---

## 36. ReBaPL: Repulsive Bayesian Prompt Learning

**论文链接:** [http://arxiv.org/abs/2511.17339v1](http://arxiv.org/abs/2511.17339v1)

**作者:** Yassir Bendou, Omar Ezzahir, Eduardo Fernandes Montesuma, Gabriel Mahuas, Victoria Shevchenko, Mike Gartrell

**发布时间:** 2025-11-21

**备注:** Under review

### GPT解析

### 总结

本文介绍了一种名为Repulsive Bayesian Prompt Learning (ReBaPL)的新方法，用于解决传统提示调整方法中的过拟合和分布外泛化问题。该方法通过在表示空间中引入排斥力，结合循环步长计划和随机梯度哈密顿蒙特卡洛算法，更有效地探索提示的后验分布，提高模型的鲁棒性和泛化能力。

### 背景

提示学习已成为调整大规模基础模型以适应下游任务的有效技术。然而，传统的提示调整方法容易过拟合，并且在分布外泛化方面表现不佳。贝叶斯提示学习被提出作为解决方案，将提示优化视为贝叶斯推断问题，以增强鲁棒性。

### 目的

为了解决传统提示学习方法中的过拟合和分布外泛化问题，本文提出了一种新的贝叶斯提示学习方法ReBaPL，旨在高效探索提示的复杂且通常是多模态的后验景观，从而提高模型的泛化能力。

### 方法

ReBaPL方法结合了循环步长计划和随机梯度哈密顿蒙特卡洛（SGHMC）算法，实现探索新模态和细化现有模态的交替阶段。此外，方法引入了一种基于不同提示产生的表示分布计算的度量（包括最大均值差异和Wasserstein距离）的势函数推导出的排斥力，这种表示空间排斥力使探索多样化，并防止过早收敛到单一模态。

### 主要发现

ReBaPL方法能够对提示后验分布进行更全面的表征，从而提高泛化能力。与先前的贝叶斯提示学习方法相比，ReBaPL提供了一种模块化的即插即用贝叶斯扩展，可以基于任何现有的最大似然估计提示学习方法进行扩展。在多个基准数据集上的实验证明，ReBaPL在提示学习方面优于最先进的方法。

### 结论

ReBaPL通过在表示空间中引入排斥力和结合先进的采样算法，有效地解决了传统提示学习方法中的过拟合和分布外泛化问题，为提示学习提供了一种更鲁棒和泛化能力更强的解决方案。

### 翻译

提示学习已成为调整大规模基础模型以适应下游任务的有效技术。然而，传统的提示调整方法容易过拟合，并且在分布外泛化方面表现不佳。为了解决这些局限性，提出了贝叶斯提示学习，它将提示优化视为贝叶斯推断问题，以增强鲁棒性。本文介绍了Repulsive Bayesian Prompt Learning（ReBaPL），这是一种新颖的贝叶斯提示学习方法，旨在高效探索提示的复杂且通常是多模态的后验景观。我们的方法将循环步长计划与随机梯度哈密顿蒙特卡洛（SGHMC）算法相结合，实现了探索新模态和细化现有模态的交替阶段。此外，我们引入了一种基于不同提示产生的表示分布计算的度量（包括最大均值差异和Wasserstein距离）的势函数推导出的排斥力。这种表示空间排斥力使探索多样化，并防止过早收敛到单一模态。我们的方法能够对提示后验分布进行更全面的表征，从而提高泛化能力。与先前的贝叶斯提示学习方法相比，我们的方法为任何基于最大似然估计的现有提示学习方法提供了模块化的即插即用贝叶斯扩展。我们在几个基准数据集上证明了ReBaPL的有效性，显示出在提示学习方面优于最先进方法的性能。


### 论文摘要

Prompt learning has emerged as an effective technique for fine-tuning large-scale foundation models for downstream tasks. However, conventional prompt tuning methods are prone to overfitting and can struggle with out-of-distribution generalization. To address these limitations, Bayesian prompt learning has been proposed, which frames prompt optimization as a Bayesian inference problem to enhance robustness. This paper introduces Repulsive Bayesian Prompt Learning (ReBaPL), a novel method for Bayesian prompt learning, designed to efficiently explore the complex and often multimodal posterior landscape of prompts. Our method integrates a cyclical step-size schedule with a stochastic gradient Hamiltonian Monte Carlo (SGHMC) algorithm, enabling alternating phases of exploration to discover new modes, and exploitation to refine existing modes. Furthermore, we introduce a repulsive force derived from a potential function over probability metrics (including Maximum Mean Discrepancy and Wasserstein distance) computed on the distributions of representations produced by different prompts. This representation-space repulsion diversifies exploration and prevents premature collapse to a single mode. Our approach allows for a more comprehensive characterization of the prompt posterior distribution, leading to improved generalization. In contrast to prior Bayesian prompt learning methods, our method provides a modular plug-and-play Bayesian extension of any existing prompt learning method based on maximum likelihood estimation. We demonstrate the efficacy of ReBaPL on several benchmark datasets, showing superior performance over state-of-the-art methods for prompt learning.

---

## 37. Agentifying Agentic AI

**论文链接:** [http://arxiv.org/abs/2511.17332v1](http://arxiv.org/abs/2511.17332v1)

**作者:** Virginia Dignum, Frank Dignum

**发布时间:** 2025-11-21

**备注:** 10 pages; 1 figure

### GPT解析

### 总结

这篇论文探讨了如何通过结合认知、合作和治理模型来实现Agentic AI的愿景，并提出了自主代理和多代理系统社区的概念工具作为实现这一目标的基础。

### 背景

Agentic AI旨在为系统提供持续的自主性、推理和交互能力，但需要明确的认知、合作和治理模型来补充其关于代理的假设。

### 目的

论证自主代理和多代理系统(AAMAS)社区开发的概念工具（如BDI架构、通信协议、机制设计和制度建模）为实现Agentic AI提供了基础。

### 方法

将自适应、数据驱动的方法与推理和协调的结构化模型相结合，构建通往透明、合作和负责的代理系统的路径。

### 主要发现

通过结合自适应、数据驱动的方法与结构化的推理和协调模型，可以创建出不仅有能力且灵活，而且透明、合作和负责的代理系统。

### 结论

提出了一种能够连接形式理论和实际自主性的代理观点。

### 翻译

代理AI旨在赋予系统持续的自主性、推理和交互能力。为了实现这一愿景，其关于代理的假设必须通过明确的认知、合作和治理模型来补充。本文认为，自主代理和多代理系统(AAMAS)社区开发的概念工具，如BDI架构、通信协议、机制设计和制度建模，恰恰提供了这样的基础。通过将自适应、数据驱动的方法与推理和协调的结构化模型相结合，我们概述了一条通往不仅有能力且灵活，而且透明、合作和负责的代理系统的道路。结果是形成了一种能够连接形式理论和实际自主性的代理观点。


### 论文摘要

Agentic AI seeks to endow systems with sustained autonomy, reasoning, and interaction capabilities. To realize this vision, its assumptions about agency must be complemented by explicit models of cognition, cooperation, and governance. This paper argues that the conceptual tools developed within the Autonomous Agents and Multi-Agent Systems (AAMAS) community, such as BDI architectures, communication protocols, mechanism design, and institutional modelling, provide precisely such a foundation. By aligning adaptive, data-driven approaches with structured models of reasoning and coordination, we outline a path toward agentic systems that are not only capable and flexible, but also transparent, cooperative, and accountable. The result is a perspective on agency that bridges formal theory and practical autonomy.

---

## 38. Agentic Program Verification

**论文链接:** [http://arxiv.org/abs/2511.17330v1](http://arxiv.org/abs/2511.17330v1)

**作者:** Haoxin Tu, Huan Zhao, Yahui Song, Mehtab Zafar, Ruijie Meng, Abhik Roychoudhury

**发布时间:** 2025-11-21

**备注:** 21 pages, 8 figures

### GPT解析

### 总结

本文介绍了一个名为AutoRocq的LLM代理，用于程序验证，它通过迭代细化循环和与定理证明器的自主协作来改进证明，实验证明其在自动化程序验证方面有良好效果。

### 背景

自动生成的代码因大型语言模型(LLMs)的普及而受到关注；AlphaProof倡议展示了AI用于通用数学推理的可能性；对计算机程序的推理虽可通过数学推理完成，但更具结构性和上下文丰富性。

### 目的

开发一个名为AutoRocq的LLM代理进行程序验证，不同于过去依赖大量训练证明示例的方法，该代理即时学习并通过迭代循环改进证明。

### 方法

证明代理与Rocq定理证明器通信获取上下文和反馈，实现证明的迭代改进；最终产生由Rocq检查的证明推导；涉及证明代理和定理证明器间的自主协作，促进证明搜索和结构决策。

### 主要发现

在SV-COMP基准测试和Linux内核模块上的实验评估显示该代理在自动化程序验证方面有希望的效果；可与AI编码代理集成形成生成和验证循环。

### 结论

随着代码生成自动化普及，该证明代理可集成到AI编码系统中，更接近可信自动编程的愿景。

### 翻译

最近自动生成的代码日益普及，这得益于大型语言模型(LLMs)的广泛应用。此外，AlphaProof倡议已经展示了AI用于通用数学推理的可能性。对计算机程序(软件)的推理可以通过通用数学推理完成；然而，它往往更具结构性和上下文丰富性。这形成了一个有吸引力的前景，因为AI代理可用于推理由AI生成的大量代码。在本工作中，我们提出了第一个用于进行程序验证的LLM代理AutoRocq。与过去依赖在证明示例上大量训练LLM的工作不同，我们的代理即时学习并通过迭代细化循环改进证明。证明的迭代改进是通过证明代理与Rocq(前身为Coq)定理证明器通信以获取额外上下文和反馈来实现的。迭代的最终结果是由Rocq定理证明器检查的证明推导。通过这种方式，我们的证明构建涉及证明代理和定理证明器之间的自主协作。这种自主性促进了证明搜索和决定证明树结构的决策。在SV-COMP基准测试和Linux内核模块上的实验评估显示出在实现自动化程序验证方面的有希望的效果。随着代码生成自动化变得更加普及，我们认为我们的证明代理可以潜在地与AI编码代理集成，以实现生成和验证循环，从而更接近可信自动编程的愿景。


### 论文摘要

Automatically generated code is gaining traction recently, owing to the prevalence of Large Language Models (LLMs). Further, the AlphaProof initiative has demonstrated the possibility of using AI for general mathematical reasoning. Reasoning about computer programs (software) can be accomplished via general mathematical reasoning; however, it tends to be more structured and richer in contexts. This forms an attractive proposition, since then AI agents can be used to reason about voluminous code that gets generated by AI.   In this work, we present a first LLM agent, AutoRocq, for conducting program verification. Unlike past works, which rely on extensive training of LLMs on proof examples, our agent learns on-the-fly and improves the proof via an iterative refinement loop. The iterative improvement of the proof is achieved by the proof agent communicating with the Rocq (formerly Coq) theorem prover to get additional context and feedback. The final result of the iteration is a proof derivation checked by the Rocq theorem prover. In this way, our proof construction involves autonomous collaboration between the proof agent and the theorem prover. This autonomy facilitates the search for proofs and decision-making in deciding on the structure of the proof tree.   Experimental evaluation on SV-COMP benchmarks and on Linux kernel modules shows promising efficacy in achieving automated program verification. As automation in code generation becomes more widespread, we posit that our proof agent can be potentially integrated with AI coding agents to achieve a generate and validate loop, thus moving closer to the vision of trusted automatic programming.

---

## 39. SpatialGeo:Boosting Spatial Reasoning in Multimodal LLMs via Geometry-Semantics Fusion

**论文链接:** [http://arxiv.org/abs/2511.17308v1](http://arxiv.org/abs/2511.17308v1)

**作者:** Jiajie Guo, Qingpeng Zhu, Jin Zeng, Xiaolong Wu, Changyong He, Weida Wang

**发布时间:** 2025-11-21

### GPT解析

### 总结

该研究提出了一种名为SpatialGeo的新型视觉编码器，通过几何和语义特征的层次融合，提升了多模态大语言模型在空间推理方面的能力，在保持高效的同时显著提高了准确性。

### 背景

多模态大语言模型（MLLMs）在大语言模型（LLMs）的强大推理能力下，在图像和语言任务中取得了显著进展。然而，大多数MLLMs在三维空间中解释和推断空间排列方面的空间推理能力有限。

### 目的

提出一种基于几何和语义特征层次融合的新型视觉编码器，生成具有空间感知能力的视觉嵌入，提升MLLMs的空间定位能力。

### 方法

揭示大多数现有MLLMs使用的视觉编码器的空间不足源于其仅限于实例级语义特征的有损嵌入；通过层次适配器补充来自纯视觉自监督学习的几何特征；使用预训练的LLaVA模型高效训练SpatialGeo网络；使用随机特征丢弃进行优化，避免仅依赖CLIP编码器的平凡解。

### 主要发现

实验结果表明，SpatialGeo提高了空间推理任务的准确性，在SpatialRGPT-Bench上将最先进模型的准确率提高了至少8.0%，同时推理时的内存消耗减少了约50%。

### 结论

SpatialGeo通过结合几何和语义特征，有效解决了MLLMs在空间推理方面的局限性，同时保持了较高的效率。

### 翻译

多模态大语言模型（MLLMs）由于大语言模型（LLMs）的强大推理能力，在图像和语言任务中取得了显著进展。然而，大多数MLLMs在三维空间中解释和推断空间排列方面的空间推理能力有限。在这项工作中，我们提出了一种基于几何和语义特征层次融合的新型视觉编码器，生成具有空间感知能力的视觉嵌入，提升MLLMs的空间定位能力。具体来说，我们首先揭示了空间不足的缺点源于大多数现有MLLMs（如CLIP）使用的视觉编码器的有损嵌入，仅限于实例级语义特征。这促使我们通过层次适配器补充来自纯视觉自监督学习的几何特征，增强提出的SpatialGeo中的空间感知能力。该网络使用预训练的LLaVA模型高效训练，并通过随机特征丢弃进行优化，避免仅依赖CLIP编码器的平凡解。实验结果表明，SpatialGeo提高了空间推理任务的准确性，在SpatialRGPT-Bench上将最先进模型的准确率提高了至少8.0%，同时推理时的内存消耗减少了约50%。源代码可通过https://ricky-plus.github.io/SpatialGeoPages/获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态大语言模型（MLLMs）在空间推理能力上的不足。这个问题很重要，因为空间推理是智能体（如机器人）与物理世界交互的基础能力，对于机器人导航、操作和视觉规划等关键任务至关重要。现有模型虽然能识别物体，但难以准确理解物体间的空间关系，限制了它们在实际应用中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有MLLMs空间推理能力不足的原因，发现主要源于视觉编码器（如CLIP）在嵌入空间中存在空间模糊问题，导致空间信息丢失。他们借鉴了MMVP工作对嵌入空间的分析方法，并利用了MoGe编码器（一个专门用于从单目图像重建3D几何结构的自监督学习模型）。基于LLaVA模型，作者设计了分层适配器来融合几何和语义特征，并开发了高效的训练策略，包括预训练初始化和随机特征丢弃机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过在视觉编码器中融合几何和语义特征，增强MLLMs的空间推理能力。整体流程包括：1) 架构由CLIP模块（提取语义特征）、MoGe模块（提取几何和语义混合特征）和LLM模块组成；2) 使用分层适配器处理MoGe编码器的特征，并将CLIP和MoGe特征交错融合；3) 采用两阶段训练：第一阶段预训练分层适配器，第二阶段进行指令微调并使用随机特征丢弃策略；4) 使用Open Spatial Dataset进行训练，并调整边界框标注以适应不同图像尺寸。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次揭示并分析了MLLMs视觉编码器中的空间模糊问题；2) 提出在视觉编码器中融合几何和语义特征的创新方法；3) 设计了特殊的分层适配器有效融合特征；4) 开发了基于预训练模型的高效训练策略。相比之前的工作，SpatialGeo不需要额外的3D特征提取模块，简化了模型结构；不需要深度图作为输入，减少了计算复杂度；实验显示它在空间推理任务上表现更佳，同时内存消耗减少约50%，且保持了通用视觉问答的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SpatialGeo通过在视觉编码器中融合几何和语义特征，有效提升了多模态大语言模型的空间推理能力，在保持通用视觉问答性能的同时显著提高了空间理解和定量空间关系判断的准确性，并且大幅降低了计算资源需求。'}


### 论文摘要

Multimodal large language models (MLLMs) have achieved significant progress in image and language tasks due to the strong reasoning capability of large language models (LLMs). Nevertheless, most MLLMs suffer from limited spatial reasoning ability to interpret and infer spatial arrangements in three-dimensional space. In this work, we propose a novel vision encoder based on hierarchical fusion of geometry and semantics features, generating spatial-aware visual embedding and boosting the spatial grounding capability of MLLMs. Specifically, we first unveil that the spatial ambiguity shortcoming stems from the lossy embedding of the vision encoder utilized in most existing MLLMs (e.g., CLIP), restricted to instance-level semantic features. This motivates us to complement CLIP with the geometry features from vision-only self-supervised learning via a hierarchical adapter, enhancing the spatial awareness in the proposed SpatialGeo. The network is efficiently trained using pretrained LLaVA model and optimized with random feature dropping to avoid trivial solutions relying solely on the CLIP encoder. Experimental results show that SpatialGeo improves the accuracy in spatial reasoning tasks, enhancing state-of-the-art models by at least 8.0% in SpatialRGPT-Bench with approximately 50% less memory cost during inference. The source code is available via https://ricky-plus.github.io/SpatialGeoPages/.

---

## 40. SlsReuse: LLM-Powered Serverless Function Reuse

**论文链接:** [http://arxiv.org/abs/2511.17262v1](http://arxiv.org/abs/2511.17262v1)

**作者:** Jinfeng Wen, Yuehan Sun

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了SlsReuse，首个基于大型语言模型的无服务器函数重用框架，旨在解决Serverless计算中函数开发面临的挑战。

### 背景

Serverless计算允许开发者实现函数级任务而无需管理基础设施，但对新手开发者提出了挑战。从零开始开发函数需要适应异构的、平台特定的编程风格，使过程耗时且容易出错。

### 目的

解决Serverless计算中缺乏专门函数推荐方法的问题，弥合任务描述与异构函数实现之间的语义差距。

### 方法

SlsReuse构建可重用函数存储库作为基础知识库，通过提示工程和少样本提示学习异构函数的统一语义增强表示，捕获代码意图、目标平台等信息，并执行意图感知发现和多级剪枝策略。

### 主要发现

在110个任务查询的数据集上评估，基于ChatGPT-4o的SlsReuse实现了91.20%的Recall@10，超越了最先进的基线24.53个百分点。

### 结论

SlsReuse是首个基于LLM的无服务器函数重用框架，有效弥合了开发人员需求与函数语义之间的差距，在函数推荐任务上表现出色。

### 翻译

无服务器计算已迅速成为一种流行的云计算范式。它使开发者能够实现函数级任务，即无服务器函数，而无需管理基础设施。虽然减少了运营开销，但它对新手开发者提出了挑战。从零开始开发函数需要适应异构的、平台特定的编程风格，使过程耗时且容易出错。函数重用为解决这些挑战提供了有希望的解决方案。然而，关于无服务器计算的研究缺乏专门的函数推荐方法。由于任务描述和异构函数实现之间的语义差距，传统上下文中的技术仍然不足。在大型语料库上预训练的大型语言模型的进步，通过将开发人员需求与函数语义对齐，为弥合这一差距创造了机会。本文提出了SlsReuse，这是第一个用于无服务器函数重用的基于LLM的框架。具体而言，SlsReuse首先构建一个可重用函数存储库作为基础知识库。然后，它通过有效的提示工程和少样本提示学习异构函数的统一语义增强表示，捕获隐式代码意图、目标平台、编程语言和云服务。最后，给定自然语言任务查询，SlsReuse执行意图感知发现，结合多级剪枝策略和相似性匹配。我们在110个任务查询的数据集上评估了SlsReuse。基于最具代表性的LLM之一ChatGPT-4o，SlsReuse实现了91.20%的Recall@10，超过了最先进的基线24.53个百分点。


### 论文摘要

Serverless computing has rapidly emerged as a popular cloud computing paradigm. It enables developers to implement function-level tasks, i.e., serverless functions, without managing infrastructure. While reducing operational overhead, it poses challenges, especially for novice developers. Developing functions from scratch requires adapting to heterogeneous, platform-specific programming styles, making the process time-consuming and error-prone. Function reuse offers a promising solution to address these challenges. However, research on serverless computing lacks a dedicated approach for function recommendation. Existing techniques from traditional contexts remain insufficient due to the semantic gap between task descriptions and heterogeneous function implementations. Advances in large language models (LLMs), pre-trained on large-scale corpora, create opportunities to bridge this gap by aligning developer requirements with function semantics.   This paper presents SlsReuse, the first LLM-powered framework for serverless function reuse. Specifically, SlsReuse first constructs a reusable function repository serving as a foundational knowledge base. Then, it learns unified semantic-enhanced representations of heterogeneous functions through effective prompt engineering with few-shot prompting, capturing implicit code intent, target platforms, programming languages, and cloud services. Finally, given a natural language task query, SlsReuse performs intent-aware discovery combined with a multi-level pruning strategy and similarity matching. We evaluate SlsReuse on a curated dataset of 110 task queries. Built on ChatGPT-4o, one of the most representative LLMs, SlsReuse achieves Recall@10 of 91.20%, exceeding the state-of-the-art baseline by 24.53 percentage points.

---

## 41. A Little More Like This: Text-to-Image Retrieval with Vision-Language Models Using Relevance Feedback

**论文链接:** [http://arxiv.org/abs/2511.17255v1](http://arxiv.org/abs/2511.17255v1)

**作者:** Bulat Khaertdinov, Mirela Popa, Nava Tintarev

**发布时间:** 2025-11-21

**备注:** Accepted to WACV'26

### GPT解析

### 总结

该研究提出了一种受传统文本搜索启发的相关性反馈机制，用于改进视觉-语言模型的检索性能，无需进行微调。研究团队提出了四种反馈策略，并在Flickr30k和COCO数据集上进行了实验验证。

### 背景

大型视觉-语言模型(VLMs)允许使用自然语言查询进行直观的视觉搜索，但提高性能通常需要微调和扩展到更大的模型变体。

### 目的

提出一种在推理时提高检索性能的机制，作为微调的替代方案，同时也能与已微调的VLMs一起使用。

### 方法

研究团队提出了并评估了四种反馈策略：修改的经典伪相关性反馈(PRF)，基于排名靠前的结果优化查询嵌入；生成式相关性反馈(GRF)，使用合成标题进行查询优化；注意力反馈摘要器(AFS)，一个基于transformer的模型，整合了相关项目的多模态细粒度特征；以及使用真实标题模拟的显式反馈作为基线。

### 主要发现

实验表明，GRF、AFS和显式反馈使较小VLMs的MRR@5检索性能提高了3-5%，使较大VLMs提高了1-3%。AFS和显式反馈都能减轻查询漂移，并且在迭代、多轮检索设置中比GRF更稳健。

### 结论

相关性反馈可以一致地提高VLMs的检索性能，并为交互式和自适应视觉搜索开辟了机会。

### 翻译

大型视觉-语言模型(VLMs)使使用自然语言查询进行直观的视觉搜索成为可能。然而，提高它们的性能通常需要微调和扩展到更大的模型变体。在这项工作中，我们提出了一种受传统基于文本的搜索启发的机制，用于在推理时提高检索性能：相关性反馈。虽然相关性反馈可以作为微调的替代方案，但其与模型无关的设计也使其可以与已微调的VLMs一起使用。具体来说，我们介绍并评估了四种基于VLM的检索反馈策略。首先，我们修改了经典的伪相关性反馈(PRF)，它基于排名靠前的结果优化查询嵌入。为了解决其局限性，我们提出了生成式相关性反馈(GRF)，它使用合成标题进行查询优化。此外，我们引入了一个注意力反馈摘要器(AFS)，这是一个基于transformer的自定义模型，整合了来自相关项目的多模态细粒度特征。最后，我们使用真实标题模拟显式反馈作为上限基线。在VLM骨干网络上对Flickr30k和COCO的实验表明，与无反馈的检索相比，GRF、AFS和显式反馈使较小VLMs的MRR@5检索性能提高了3-5%，使较大VLMs提高了1-3%。此外，AFS与显式反馈类似，能够减轻查询漂移，并且在迭代、多轮检索设置中比GRF更稳健。我们的研究结果表明，相关性反馈可以一致地提高VLMs的检索性能，并为交互式和自适应视觉搜索开辟了机会。


### 论文摘要

Large vision-language models (VLMs) enable intuitive visual search using natural language queries. However, improving their performance often requires fine-tuning and scaling to larger model variants. In this work, we propose a mechanism inspired by traditional text-based search to improve retrieval performance at inference time: relevance feedback. While relevance feedback can serve as an alternative to fine-tuning, its model-agnostic design also enables use with fine-tuned VLMs. Specifically, we introduce and evaluate four feedback strategies for VLM-based retrieval. First, we revise classical pseudo-relevance feedback (PRF), which refines query embeddings based on top-ranked results. To address its limitations, we propose generative relevance feedback (GRF), which uses synthetic captions for query refinement. Furthermore, we introduce an attentive feedback summarizer (AFS), a custom transformer-based model that integrates multimodal fine-grained features from relevant items. Finally, we simulate explicit feedback using ground-truth captions as an upper-bound baseline. Experiments on Flickr30k and COCO with the VLM backbones show that GRF, AFS, and explicit feedback improve retrieval performance by 3-5% in MRR@5 for smaller VLMs, and 1-3% for larger ones, compared to retrieval with no feedback. Moreover, AFS, similarly to explicit feedback, mitigates query drift and is more robust than GRF in iterative, multi-turn retrieval settings. Our findings demonstrate that relevance feedback can consistently enhance retrieval across VLMs and open up opportunities for interactive and adaptive visual search.

---

## 42. Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats

**论文链接:** [http://arxiv.org/abs/2511.17254v1](http://arxiv.org/abs/2511.17254v1)

**作者:** Jiaye Qian, Ge Zheng, Yuchen Zhu, Sibei Yang

**发布时间:** 2025-11-21

**备注:** Accepted to NeurIPS 2025, Project Page: https://github.com/SooLab/AllPath

### GPT解析

### 总结

本研究提出了一种针对大型视觉语言模型幻觉问题的综合干预框架，通过分析不同干预路径的相互作用，有效减少了幻觉现象。

### 背景

大型视觉语言模型虽然在多种任务中表现出色，但仍然容易出现幻觉问题。

### 目的

开发一个与transformer因果架构一致的干预框架，以减少大型视觉语言模型中的幻觉现象。

### 方法

提出一个综合干预框架，整合不同干预路径对幻觉的影响，并针对判别性和生成性格式定制方法来识别和干预关键幻觉头。

### 主要发现

幻觉并非来自单一因果路径，而是来自图像到输入文本、图像到输出文本和文本到文本路径之间的相互作用；大型视觉语言模型根据问答对齐格式依赖不同路径；基于这些发现提出了针对性的干预方法。

### 结论

在多个基准测试上的实验表明，该方法一致地减少了不同对齐类型中的幻觉。

### 翻译

尽管大型视觉语言模型在广泛任务中表现出令人印象深刻的性能，但它们仍然容易出现幻觉。在本研究中，我们提出了一种与transformer因果架构一致的综合干预框架，整合了不同干预路径对幻觉的影响。我们发现幻觉并非来自单一因果路径，而是来自图像到输入文本、图像到输出文本和文本到文本路径之间的相互作用。我们还首次发现，大型视觉语言模型根据问答对齐格式依赖不同的路径。基于这些见解，我们提出了简单但有效的方法来识别和干预每个路径中的关键幻觉头，这些方法针对判别性和生成性格式进行了定制。在多个基准测试上的实验表明，我们的方法一致地减少了不同对齐类型中的幻觉。


### 论文摘要

Despite their impressive performance across a wide range of tasks, Large Vision-Language Models (LVLMs) remain prone to hallucination. In this study, we propose a comprehensive intervention framework aligned with the transformer's causal architecture in LVLMs, integrating the effects of different intervention paths on hallucination. We find that hallucinations in LVLMs do not arise from a single causal path, but rather from the interplay among image-to-input-text, image-to-output-text, and text-to-text pathways. For the first time, we also find that LVLMs rely on different pathways depending on the question-answer alignment format. Building on these insights, we propose simple yet effective methods to identify and intervene on critical hallucination heads within each pathway, tailored to discriminative and generative formats. Experiments across multiple benchmarks demonstrate that our approach consistently reduces hallucinations across diverse alignment types.

---

## 43. Signed Networks: theory, methods, and applications

**论文链接:** [http://arxiv.org/abs/2511.17247v1](http://arxiv.org/abs/2511.17247v1)

**作者:** Fernando Diaz-Diaz, Elena Candellone, Miguel A. Gonzalez-Casado, Emma Fraxanet, Antoine Vendeville, Irene Ferri, Andreia Sofia Teixeira

**发布时间:** 2025-11-21

### GPT解析

### 总结

这篇论文是一篇关于符号网络理论的全面综述，涵盖了符号网络的数学原理、特定度量、结构分析、动态过程以及实际应用挑战，为研究复杂系统中的符号网络提供了结构化的入门点和参考框架。

### 背景

在复杂系统中，相互作用往往具有质的区别（如友好或敌对、支持或冲突、兴奋或抑制）。这种极性改变了我们对复杂系统中结构和动力学的理解，负面联系不仅仅是缺失的正面联系，而是一种产生张力和可能不对称性的约束。符号网络为不同学科提供了研究二元性、平衡和对立的共享语言。

### 目的

提供一个全面和基础的符号网络理论总结，整合理论基础、方法方法和跨领域例子，为对复杂系统中符号网络研究感兴趣的研究人员提供结构化的入门点和参考框架。

### 方法

综述通过形式化符号图的数学原理、调查特定于符号网络的度量、回顾平衡理论、检查符号网络结构方面、讨论动态过程、强调实际挑战、提供数据集概述以及解决常见陷阱等方法展开研究。

### 主要发现

负面联系是产生张力和不对称性的约束；符号网络为不同学科提供研究二元性的共享语言；符号网络有特定度量方法；平衡理论与挫折感有密切联系；符号网络结构分析包括多个关键主题；符号网络上有多种动态过程；从真实系统构建符号数据面临挑战；建模分析符号数据存在常见陷阱。

### 结论

该综述整合了理论基础、方法方法和跨领域例子，符号网络为理解和分析具有极性相互作用的复杂系统提供了强大的框架，跨越了多个学科领域。

### 翻译

符号网络提供了一个原则性的框架，用于表示相互作用不仅是存在或不存在，而且在性质上截然不同的系统：友好或敌对，支持或冲突，兴奋或抑制。这种极性改变了我们对复杂系统中结构和动力学的思考方式。跨学科领域，符号网络提供了一个共享的语言，将二元性、平衡和对立形式化为系统行为的组成部分。这篇综述提供了符号网络理论和全面基础的总结，形式化了符号图的数学原理，调查了特定于符号网络的度量，回顾了平衡理论，检查了符号网络的结构方面，讨论了动态过程，强调了实际挑战，提供了数据集概述，并解决了常见陷阱和挑战。


### 论文摘要

Signed networks provide a principled framework for representing systems in which interactions are not merely present or absent but qualitatively distinct: friendly or antagonistic, supportive or conflicting, excitatory or inhibitory. This polarity reshapes how we think about structure and dynamics in complex systems: a negative tie is not simply a missing positive one but a constraint that generates tension, and possibly asymmetry. Across disciplines, from sociology to neuroscience and machine learning, signed networks provide a shared language to formalise duality, balance, and opposition as integral components of system behaviour. This review provides a comprehensive and foundational summary of signed network theory. It formalises the mathematical principles of signed graphs and surveys signed-network-specific measures, including signed degree distributions, clustering, centralities, motifs, and Laplacians. It revisits balance theory, tracing its cognitive and structural formulations and their connections to frustration. Structural aspects of signed networks are examined, analysing key topics such as null models, node embeddings, sign prediction, and community detection. Subsequent sections address dynamical processes on and of signed networks, such as opinion dynamics, contagion models, and data-driven approaches for studying evolving networks. Practical challenges in constructing, inferring and validating signed data from real-world systems are also highlighted, and we offer an overview of currently available datasets. We also address common pitfalls and challenges that arise when modelling or analysing signed data. Overall, this review integrates theoretical foundations, methodological approaches, and cross-domain examples, providing a structured entry point and a reference framework for researchers interested in the study of signed networks in complex systems.

---

## 44. QueryOcc: Query-based Self-Supervision for 3D Semantic Occupancy

**论文链接:** [http://arxiv.org/abs/2511.17221v1](http://arxiv.org/abs/2511.17221v1)

**作者:** Adam Lilja, Ji Lan, Junsheng Fu, Lars Hammarstrand

**发布时间:** 2025-11-21

### GPT解析

### 总结

QueryOcc是一种基于查询的自监督框架，通过4D时空查询直接学习连续3D语义占用，超越了现有方法，实现了高效的自监督占用学习。

### 背景

从图像学习3D场景几何和语义是计算机视觉的核心挑战和自动驾驶的关键能力。由于大规模3D标注成本高昂，研究者们转向自监督学习方法。

### 目的

开发一种能够直接从传感器数据学习连续3D语义占用的自监督框架，克服现有方法在空间精度和可扩展性方面的限制。

### 方法

Query框架通过在相邻帧中采样的独立4D时空查询学习连续3D语义占用，支持来自视觉基础模型派生的伪点云或原始lidar数据的监督。引入可收缩场景表示，在保持近场细节的同时平滑压缩远距离区域，实现长距离监督和推理。

### 主要发现

QueryOcc在自监督的Occ3D-nuScenes基准测试中，在语义RayIoU上比之前的基于相机的方法高出26%，同时以11.6 FPS的速度运行。

### 结论

直接4D查询监督可以实现强大的自监督占用学习，为3D场景理解提供了新的有效方法。

### 翻译

从图像学习3D场景几何和语义是计算机视觉的核心挑战，也是自动驾驶的关键能力。由于大规模3D标注成本过高，近期工作探索直接从传感器数据进行自监督学习而无需人工标签。现有方法要么依赖2D渲染一致性（3D结构仅隐式出现），要么依赖于从累积的lidar点云获得的离散体素网格，这限制了空间精度和可扩展性。我们引入QueryOcc，一种基于查询的自监督框架，通过在相邻帧中采样的独立4D时空查询直接学习连续3D语义占用。该框架支持来自视觉基础模型派生的伪点云或原始lidar数据的监督。为了在保持内存恒定的同时实现长距离监督和推理，我们引入了一种可收缩的场景表示，在保持近场细节的同时平滑压缩远距离区域。QueryOcc在自监督的Occ3D-nuScenes基准测试中，在语义RayIoU上比之前的基于相机的方法高出26%，同时以11.6 FPS的速度运行，证明直接4D查询监督可以实现强大的自监督占用学习。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从图像中学习3D语义占用而不依赖昂贵的3D标注数据。这个问题在现实中很重要，因为3D场景理解是自动驾驶的核心能力，而大规模3D标注极其昂贵（例如标注nuScenes数据集需约4000小时人工工作），限制了自动驾驶感知系统的泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性（基于渲染的方法提供间接几何信号，基于激光雷达的方法受限于离散体素表示），假设直接在连续4D空间时间中进行监督可以提供更清晰的几何反馈。他们设计Query框架时借鉴了Lift-Splat-Show方法进行特征提升，结合查询式学习预测场景属性，并采用BEV表示的效率，通过空间收缩扩展到无界场景。同时利用视觉基础模型生成的伪点云和语义标签作为监督信号。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过直接的4D时空查询监督学习连续的3D语义占用场，避免依赖渲染损失或激光雷达聚合体素化。实现流程分为四个阶段：1)图像编码：用预训练骨干网络处理图像并提取特征；2)提升-收缩-喷射：将特征提升到收缩的BEV空间，使用对数线性深度分箱平衡近远分辨率；3)BEV特征处理：用残差块和可变形注意力捕获空间依赖；4)基于查询的解码：统一解码器预测任意查询点的占用和语义，支持多任务输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于查询的自监督框架，直接在4D时空监督；2)无界收缩场景表示，保持近场细节同时压缩远场；3)提升-收缩-喷射机制，将图像特征提升到收缩空间；4)灵活的监督信号来源，支持仅相机或相机-激光雷达组合。与之前工作不同：相比渲染方法，直接在3D空间监督而非间接通过2D图像；相比激光雷达方法，避免体素化限制；结合BEV效率与无界场景表示；将查询学习引入仅相机语义占用预测。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'QueryOcc通过基于查询的直接4D时空监督和无界收缩场景表示，实现了从多视图图像高效学习连续3D语义占用，在自监督设置下达到最先进性能同时保持实时推理能力。'}


### 论文摘要

Learning 3D scene geometry and semantics from images is a core challenge in computer vision and a key capability for autonomous driving. Since large-scale 3D annotation is prohibitively expensive, recent work explores self-supervised learning directly from sensor data without manual labels. Existing approaches either rely on 2D rendering consistency, where 3D structure emerges only implicitly, or on discretized voxel grids from accumulated lidar point clouds, limiting spatial precision and scalability. We introduce QueryOcc, a query-based self-supervised framework that learns continuous 3D semantic occupancy directly through independent 4D spatio-temporal queries sampled across adjacent frames. The framework supports supervision from either pseudo-point clouds derived from vision foundation models or raw lidar data. To enable long-range supervision and reasoning under constant memory, we introduce a contractive scene representation that preserves near-field detail while smoothly compressing distant regions. QueryOcc surpasses previous camera-based methods by 26% in semantic RayIoU on the self-supervised Occ3D-nuScenes benchmark while running at 11.6 FPS, demonstrating that direct 4D query supervision enables strong self-supervised occupancy learning. https://research.zenseact.com/publications/queryocc/

---

## 45. Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers

**论文链接:** [http://arxiv.org/abs/2511.17209v1](http://arxiv.org/abs/2511.17209v1)

**作者:** Cris Claessens, Christiaan Viviers, Giacomo D'Amicantonio, Egor Bondarev, Fons van der Sommen

**发布时间:** 2025-11-21

### GPT解析

### 总结

介绍SPECTRE，一个完全基于Transformer的体积CT基础模型，通过自监督和跨模态预训练学习通用CT表示，解决了体积CT处理的独特挑战，并在多个基准测试中表现优异。

### 背景

体积CT面临极端令牌缩放、几何各向异性和弱或有噪声的临床监督等独特挑战，使标准Transformer和对比学习方法无法直接有效应用。

### 目的

开发一个可扩展的3D Vision Transformer架构，使用现代自监督和视觉语言预训练策略学习通用CT表示，解决体积CT处理的独特挑战。

### 方法

提出SPECTRE方法，联合优化局部Transformer用于高分辨率体积特征提取和全局Transformer用于全扫描上下文建模；仅在公开可用的CT数据集上训练；预训练结合DINO风格的自蒸馏和基于SigLIP的视觉语言对齐，使用配对的放射学报告。

### 主要发现

SPECTRE在多个CT基准测试中，在零样本和微调设置下始终优于先前的CT基础模型；仅使用公开数据集即可实现高性能、可泛化的表示。

### 结论

SPECTRE是3D医学成像的可扩展、开放和完全基于Transformer的基础模型。

### 翻译

我们引入SPECTRE，一个完全基于Transformer的体积计算机断层扫描（CT）基础模型。我们的自监督和跨模态预训练用于CT表示提取（SPECTRE）方法利用可扩展的3D Vision Transformer架构和现代自监督及视觉语言预训练策略来学习通用CT表示。体积CT面临独特挑战，如极端令牌缩放、几何各向异性和弱或有噪声的临床监督，这使得标准Transformer和对比学习配方无法直接有效应用。该框架联合优化了用于高分辨率体积特征提取的局部Transformer和用于全扫描上下文建模的全局Transformer，使大规模3D注意力计算上可行。值得注意的是，SPECTRE仅在公开可用的CT数据集上进行训练，证明无需依赖私有数据即可实现高性能、可泛化的表示。预训练结合了DINO风格的自蒸馏和基于SigLIP的视觉语言对齐，使用配对的放射学报告，产生既几何一致又有临床意义的特征。在多个CT基准测试中，SPECTRE在零样本和微调设置下均一致优于先前的CT基础模型，确立了SPECTRE作为3D医学成像的可扩展、开放和完全基于Transformer的基础模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何构建可扩展的、通用的3D医学基础模型，特别是针对CT影像数据的问题。CT体积数据面临独特挑战，如极端的标记缩放、几何各向异性、弱或有噪声的临床监督等，导致标准transformer和对比学习方法效果不佳。这个问题在现实中非常重要，因为CT是医学影像诊断中的关键工具，而现有的基础模型主要针对2D图像设计，无法直接有效应用于3D医学影像；同时医疗领域通常缺乏大量标注数据，需要有效的自监督学习方法来充分利用未标注数据。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者设计方法时借鉴了多项现有工作：利用了Vision Transformer架构但进行了3D适配；参考了DINOv3的自监督学习但进行了修改以适应医学影像；采用了SigLIP进行视觉-语言对齐；参考了SwinUNETR、nnFormer等3D医学影像transformer模型。作者的设计思路是针对CT体积数据的特性设计特定的标记化方法，采用两阶段transformer架构平衡计算效率和上下文建模能力，并通过3D旋转位置编码处理体积几何关系，最终采用两阶段预训练策略结合几何特征学习和临床语义注入。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个完全基于transformer的基础模型SPECTRE，通过结合自监督学习和视觉-语言对齐，学习既几何一致又临床有意义的特征表示。整体流程包括：1)数据预处理：从多个公开数据集收集CT扫描，进行标准化、重采样和裁剪；2)模型架构：设计局部ViT提取细粒度特征，全局ViT捕获扫描级语义，3D RoPE处理几何关系；3)两阶段预训练：第一阶段使用DINOv3风格的自监督学习，第二阶段使用SigLIP进行视觉-语言对齐；4)下游任务评估：包括癌症生物标志物预测、语义分割和零样本文本到图像检索等任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)专为CT体积数据设计的transformer架构，采用各向异性3D标记嵌入和两阶段注意力设计；2)两阶段预训练策略，先学习几何特征再注入临床语义；3)可扩展的训练方法，解决3D体积数据计算复杂度高的问题；4)完全开源发布。相比之前工作，SPECTRE不仅学习图像特征还注入临床语义；专为3D体积数据设计，处理了各向异性和大体积问题；在多个基准测试中表现更好，特别是在零样本设置下；使用更多样化的数据集进行训练；更注重几何一致性和临床语义的结合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SPECTRE通过结合专为CT体积数据设计的transformer架构和两阶段预训练策略，实现了在多个医学影像任务上超越现有方法的性能，为3D医学影像分析提供了一个可扩展、开源的基础模型。'}


### 论文摘要

We introduce SPECTRE, a fully transformer-based foundation model for volumetric computed tomography (CT). Our Self-Supervised & Cross-Modal Pretraining for CT Representation Extraction (SPECTRE) approach utilizes scalable 3D Vision Transformer architectures and modern self-supervised and vision-language pretraining strategies to learn general-purpose CT representations. Volumetric CT poses unique challenges, such as extreme token scaling, geometric anisotropy, and weak or noisy clinical supervision, that make standard transformer and contrastive learning recipes ineffective out of the box. The framework jointly optimizes a local transformer for high-resolution volumetric feature extraction and a global transformer for whole-scan context modeling, making large-scale 3D attention computationally tractable. Notably, SPECTRE is trained exclusively on openly available CT datasets, demonstrating that high-performing, generalizable representations can be achieved without relying on private data. Pretraining combines DINO-style self-distillation with SigLIP-based vision-language alignment using paired radiology reports, yielding features that are both geometrically consistent and clinically meaningful. Across multiple CT benchmarks, SPECTRE consistently outperforms prior CT foundation models in both zero-shot and fine-tuned settings, establishing SPECTRE as a scalable, open, and fully transformer-based foundation model for 3D medical imaging.

---

## 46. Continual Alignment for SAM: Rethinking Foundation Models for Medical Image Segmentation in Continual Learning

**论文链接:** [http://arxiv.org/abs/2511.17201v1](http://arxiv.org/abs/2511.17201v1)

**作者:** Jiayi Wang, Wei Dai, Haoyu Wang, Sihan Yang, Haixia Bi, Jian Sun

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为CA-SAM的持续学习策略，通过引入Alignment Layer模块使Segment Anything Model能够高效适应医学图像分割任务，解决了异构隐私政策下的数据共享和持续学习问题，在多个医学数据集上实现了最先进的性能。

### 背景

医学图像分割中，不同机构间的异构隐私政策使得联合训练在共享数据集上不可行，这促使了从数据流中进行持续图像分割学习的研究，以避免灾难性遗忘问题。

### 目的

展示SAM（Segment Anything Model）范式在平衡计算效率和性能方面的潜力，并开发一种方法使SAM能够有效适应特定医学图像分割任务。

### 方法

引入Alignment Layer，一个轻量级即插即用模块，用于对齐编码器-解码器特征分布；基于SAM和Alignment Layer，提出CA-SAM（Continual Alignment for SAM）持续学习策略，该策略能自动适应适当的Alignment Layer以减轻灾难性遗忘，同时利用SAM的零样本先验知识保持对未见医学数据集的强性能。

### 主要发现

在九个持续学习场景下的医学分割数据集上，CA-SAM达到了最先进的性能，证明了该方法的有效性。

### 结论

SAM范式在平衡计算效率和性能后具有巨大潜力，而Alignment Layer和CA-SAM策略有效解决了医学图像分割中的隐私保护和持续学习挑战。

### 翻译

在医学图像分割中，跨机构的异构隐私政策通常使得在共享数据集上进行联合训练不可行，这促使了从数据流中进行持续图像分割学习的研究，以避免灾难性遗忘。虽然Segment Anything Model（SAM）提供了强大的零样本先验知识，并已在各种下游任务中广泛微调，但其大量的参数和计算开销对实际部署构成了挑战。本文证明，一旦平衡了计算效率和性能，SAM范式就具有巨大潜力。为此，我们引入了Alignment Layer，这是一个轻量级即插即用模块，它对齐编码器-解码器特征分布，使SAM能够高效适应特定医学图像，在提高准确性的同时减少计算量。基于SAM和Alignment Layer，我们随后提出了SAM的持续对齐（CA-SAM），这是一种持续学习策略，能够自动适应适当的Alignment Layer以减轻灾难性遗忘，同时利用SAM的零样本先验知识保持对未见医学数据集的强性能。在持续学习场景下的九个医学分割数据集进行实验，CA-SAM达到了最先进的性能。我们的代码、模型和数据集将在https://github.com/azzzzyo/Continual-Alignment-for-SAM上发布。


### 论文摘要

In medical image segmentation, heterogeneous privacy policies across institutions often make joint training on pooled datasets infeasible, motivating continual image segmentation-learning from data streams without catastrophic forgetting. While the Segment Anything Model (SAM) offers strong zero-shot priors and has been widely fine-tuned across downstream tasks, its large parameter count and computational overhead challenge practical deployment. This paper demonstrates that the SAM paradigm is highly promising once its computational efficiency and performance can be balanced. To this end, we introduce the Alignment Layer, a lightweight, plug-and-play module which aligns encoder-decoder feature distributions to efficiently adapt SAM to specific medical images, improving accuracy while reducing computation. Building on SAM and the Alignment Layer, we then propose Continual Alignment for SAM (CA-SAM), a continual learning strategy that automatically adapts the appropriate Alignment Layer to mitigate catastrophic forgetting, while leveraging SAM's zero-shot priors to preserve strong performance on unseen medical datasets. Experimented across nine medical segmentation datasets under continual-learning scenario, CA-SAM achieves state-of-the-art performance. Our code, models and datasets will be released on \mbox{https://github.com/azzzzyo/Continual-Alignment-for-SAM.}

---

## 47. Navigating in the Dark: A Multimodal Framework and Dataset for Nighttime Traffic Sign Recognition

**论文链接:** [http://arxiv.org/abs/2511.17183v1](http://arxiv.org/abs/2511.17183v1)

**作者:** Aditya Mishra, Akshay Agarwal, Haroon Lone

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究针对夜间交通标志识别问题，提出了一个大规模数据集INTSD和一个新的方法LENS-Net，有效解决了低光照条件下的交通标志识别挑战。

### 背景

交通标志对道路安全和智能交通系统至关重要，但夜间识别仍面临视觉噪声和公共夜间数据集稀缺的挑战。现有方法在低光照条件下鲁棒性不足，且未能有效利用多模态线索。

### 目的

解决夜间交通标志识别的难题，通过构建大规模数据集和开发新的方法来提高夜间交通标志的检测和分类性能。

### 方法

首先引入INTSD数据集，包含印度不同地区41类交通标志的夜间图像，涵盖不同光照和天气条件；其次提出LENS-Net方法，结合自适应图像增强检测器进行光照校正和标志定位，以及结构化多模态CLIP-GCNN分类器利用跨模态注意力和基于图的推理进行鲁棒识别。

### 主要发现

LENS-Net方法超越了现有框架，消融研究证实了其关键组件的有效性，证明了自适应图像增强和多模态融合在夜间交通标志识别中的重要作用。

### 结论

INTSD数据集和LENS-Net方法为夜间交通标志识别提供了新的基准和解决方案，相关数据和代码已公开可用，促进了该领域的研究发展。

### 翻译

交通标志对道路安全和智能交通系统至关重要，能够支持导航和自动驾驶。然而，由于视觉噪声和公共夜间数据集的稀缺，夜间识别交通标志仍然具有挑战性。尽管视觉架构有所进步，但现有方法在低光照条件下的鲁棒性不足，且未能有效利用互补的多模态线索。为克服这些局限，首先，我们引入了INTSD，这是一个大规模数据集，包含印度不同地区收集的街道级别夜间交通标志图像。该数据集涵盖41类交通标志，在不同光照和天气条件下采集，为检测和分类任务提供了全面的基准。为了对INTSD进行夜间标志识别基准测试，我们使用最先进的检测和分类模型进行了广泛评估。其次，我们提出了LENS-Net，它集成了一个自适应图像增强检测器，用于联合光照校正和标志定位，随后是一个结构化多模态CLIP-GCNN分类器，利用跨模态注意力和基于图的推理进行鲁棒且语义一致的识别。我们的方法超越了现有框架，消融研究证实了其关键组件的有效性。INTSD数据集和LENS-Net的代码已公开可供研究使用。


### 论文摘要

Traffic signboards are vital for road safety and intelligent transportation systems, enabling navigation and autonomous driving. Yet, recognizing traffic signs at night remains challenging due to visual noise and scarcity of public nighttime datasets. Despite advances in vision architectures, existing methods struggle with robustness under low illumination and fail to leverage complementary mutlimodal cues effectively. To overcome these limitations, firstly, we introduce INTSD, a large-scale dataset comprising street-level night-time images of traffic signboards collected across diverse regions of India. The dataset spans 41 traffic signboard classes captured under varying lighting and weather conditions, providing a comprehensive benchmark for both detection and classification tasks. To benchmark INTSD for night-time sign recognition, we conduct extensive evaluations using state-of-the-art detection and classification models. Secondly, we propose LENS-Net, which integrates an adaptive image enhancement detector for joint illumination correction and sign localization, followed by a structured multimodal CLIP-GCNN classifier that leverages cross-modal attention and graph-based reasoning for robust and semantically consistent recognition. Our method surpasses existing frameworks, with ablation studies confirming the effectiveness of its key components. The dataset and code for LENS-Net is publicly available for research.

---

## 48. Toward Sustainable Generative AI: A Scoping Review of Carbon Footprint and Environmental Impacts Across Training and Inference Stages

**论文链接:** [http://arxiv.org/abs/2511.17179v1](http://arxiv.org/abs/2511.17179v1)

**作者:** Min-Kyu Kim, Tae-An Yoo, Ji-Bum Chung

**发布时间:** 2025-11-21

**备注:** 43 pages, 5 figure, 3 tables

### GPT解析

### 总结

这是一项关于生成式AI碳足迹评估的范围综述研究，分析了AI训练和推理阶段的环境影响，指出了当前碳核算实践的局限性，并提出了未来研究和治理方向。

### 背景

生成式AI正在迅速传播，创造了巨大的社会经济价值，但也引发了对其高能源使用和环境可持续性的担忧。虽然之前的研究主要集中在训练阶段的能源密集型特性上，但在大规模服务运营中（特别是在推理阶段）产生的累积环境足迹受到的关注较少。

### 目的

弥合这一研究空白，对AI碳足迹评估的方法论和研究趋势进行范围综述，分析现有AI碳测量工具和方法的分类和标准化状况，比较训练和推理阶段产生的环境影响，并确定多维因素如何塑造碳足迹。

### 方法

进行范围综述分析AI碳足迹评估的方法论和研究趋势，分析现有AI碳测量工具和方法的分类和标准化状况，比较训练和推理阶段的环境影响，研究模型大小、提示复杂性、服务环境和系统边界定义等多维因素对碳足迹的影响。

### 主要发现

当前AI碳核算实践存在关键局限性，包括方法论不一致、技术特定偏差以及对端到端系统视角的关注不足。研究揭示了模型大小、提示复杂性、服务环境和系统边界定义等多维因素如何影响碳足迹。

### 结论

基于这些见解，提出了未来研究和治理方向：(1)建立标准化和透明的通用测量协议；(2)设计纳入用户行为的动态评估框架；(3)开发包含隐含排放的生命周期监测系统；(4)推进多维可持续性评估框架，平衡模型性能与环境效率。

### 翻译

生成式AI正在迅速传播，创造了巨大的社会经济价值，同时也引发了对其高能源使用和环境可持续性的担忧。虽然先前的研究主要集中在训练阶段的能源密集型特性上，但在大规模服务运营中产生的累积环境足迹，特别是在推理阶段，受到的关注相对较少。为了弥合这一空白，本研究对AI碳足迹评估的方法论和研究趋势进行了范围综述。我们分析了现有AI碳测量工具和方法的分类及标准化状况，并比较了训练和推理阶段产生的环境影响。此外，我们确定了模型大小、提示复杂性、服务环境和系统边界定义等多维因素如何塑造最终的碳足迹。我们的综述揭示了当前AI碳核算实践的关键局限性，包括方法学不一致、技术特定偏差以及对端到端系统视角的关注不足。基于这些见解，我们提出了未来研究和治理方向：(1)建立标准化和透明的通用测量协议，(2)设计纳入用户行为的动态评估框架，(3)开发包含隐含排放的生命周期监测系统，以及(4)推进多维可持续性评估框架，平衡模型性能与环境效率。本文为旨在构建可持续AI生态系统的跨学科对话提供了基础，并为寻求从技术、社会和运营维度理解AI环境影响的研究人员提供了基线指导。


### 论文摘要

Generative AI is spreading rapidly, creating significant social and economic value while also raising concerns about its high energy use and environmental sustainability. While prior studies have predominantly focused on the energy-intensive nature of the training phase, the cumulative environmental footprint generated during large-scale service operations, particularly in the inference phase, has received comparatively less attention. To bridge this gap this study conducts a scoping review of methodologies and research trends in AI carbon footprint assessment. We analyze the classification and standardization status of existing AI carbon measurement tools and methodologies, and comparatively examine the environmental impacts arising from both training and inference stages. In addition, we identify how multidimensional factors such as model size, prompt complexity, serving environments, and system boundary definitions shape the resulting carbon footprint. Our review reveals critical limitations in current AI carbon accounting practices, including methodological inconsistencies, technology-specific biases, and insufficient attention to end-to-end system perspectives. Building on these insights, we propose future research and governance directions: (1) establishing standardized and transparent universal measurement protocols, (2) designing dynamic evaluation frameworks that incorporate user behavior, (3) developing life-cycle monitoring systems that encompass embodied emissions, and (4) advancing multidimensional sustainability assessment framework that balance model performance with environmental efficiency. This paper provides a foundation for interdisciplinary dialogue aimed at building a sustainable AI ecosystem and offers a baseline guideline for researchers seeking to understand the environmental implications of AI across technical, social, and operational dimensions.

---

## 49. Enhancing Quranic Learning: A Multimodal Deep Learning Approach for Arabic Phoneme Recognition

**论文链接:** [http://arxiv.org/abs/2511.17477v1](http://arxiv.org/abs/2511.17477v1)

**作者:** Ayhan Kucukmanisa, Derya Gelmez, Sukru Selim Calik, Zeynep Hilal Kilimci

**发布时间:** 2025-11-21

**备注:** 11 pages, 2 figures, 3 tables

### GPT解析

### 总结

本研究提出了一种基于transformer的多模态框架，用于阿拉伯语音素发音错误检测，结合声学和文本表示以提高精度和鲁棒性。

### 背景

多模态深度学习的进步增强了语音分析和发音评估系统能力，但阿拉伯语发音检测仍是一个关键挑战，特别是在《古兰经》朗诵背景下，细微的语音差异可能改变意义。

### 目的

开发一个结合声学和文本表示的多模态框架，实现更精确和鲁棒的阿拉伯语音素发音错误检测。

### 方法

框架整合了从UniSpeech衍生的声学嵌入和从Whisper转录中提取的基于BERT的文本嵌入；实现了早期、中期和晚期融合方法；在包含29个阿拉伯语音素的数据集上评估；从YouTube收集额外语音样本以提高数据多样性。

### 主要发现

UniSpeech-BERT多模态配置提供了强有力的结果；基于融合的transformer架构对音素级发音错误检测有效。

### 结论

该研究有助于开发智能的、说话人独立的和多模态的计算机辅助语言学习系统，为技术支持的《古兰经》发音培训和基于语音的教育应用提供了实际步骤。

### 翻译

最近多模态深度学习的进展大大增强了系统进行语音分析和发音评估的能力。准确的发音检测在阿拉伯语中仍然是一个关键挑战，特别是在《古兰经》朗诵的背景下，细微的语音差异可能会改变意义。为应对这一挑战，本研究提出了一个基于transformer的多模态框架，用于阿拉伯语音素发音错误检测，该框架结合声学和文本表示，以实现更高的精度和鲁棒性。该框架集成了从UniSpeech衍生的声学嵌入和从Whisper转录中提取的基于BERT的文本嵌入，创建了一个统一表示，同时捕捉语音细节和语言上下文。为了确定最有效的集成策略，实现了早期、中期和晚期融合方法，并在两个包含29个阿拉伯语音素（包括八个哈菲兹音）的数据集上进行了评估，这些音素由11名母语发音者发出。还从公开的YouTube录音中收集了额外的语音样本，以提高数据多样性和泛化能力。使用标准评估指标（准确率、精确率、召回率和F1分数）评估模型性能，允许对融合策略进行详细比较。实验结果表明，UniSpeech-BERT多模态配置提供了强有力的结果，基于融合的transformer架构对音素级发音错误检测有效。该研究有助于开发智能的、说话人独立的和多模态的计算机辅助语言学习系统，为技术支持的《古兰经》发音培训和更广泛的基于语音的教育应用提供了实际步骤。


### 论文摘要

Recent advances in multimodal deep learning have greatly enhanced the capability of systems for speech analysis and pronunciation assessment. Accurate pronunciation detection remains a key challenge in Arabic, particularly in the context of Quranic recitation, where subtle phonetic differences can alter meaning. Addressing this challenge, the present study proposes a transformer-based multimodal framework for Arabic phoneme mispronunciation detection that combines acoustic and textual representations to achieve higher precision and robustness. The framework integrates UniSpeech-derived acoustic embeddings with BERT-based textual embeddings extracted from Whisper transcriptions, creating a unified representation that captures both phonetic detail and linguistic context. To determine the most effective integration strategy, early, intermediate, and late fusion methods were implemented and evaluated on two datasets containing 29 Arabic phonemes, including eight hafiz sounds, articulated by 11 native speakers. Additional speech samples collected from publicly available YouTube recordings were incorporated to enhance data diversity and generalization. Model performance was assessed using standard evaluation metrics: accuracy, precision, recall, and F1-score, allowing a detailed comparison of the fusion strategies. Experimental findings show that the UniSpeech-BERT multimodal configuration provides strong results and that fusion-based transformer architectures are effective for phoneme-level mispronunciation detection. The study contributes to the development of intelligent, speaker-independent, and multimodal Computer-Aided Language Learning (CALL) systems, offering a practical step toward technology-supported Quranic pronunciation training and broader speech-based educational applications.

---

## 50. InTAct: Interval-based Task Activation Consolidation for Continual Learning

**论文链接:** [http://arxiv.org/abs/2511.17439v1](http://arxiv.org/abs/2511.17439v1)

**作者:** Patryk Krukowski, Jan Miksa, Piotr Helm, Jacek Tabor, Paweł Wawrzyński, Przemysław Spurek

**发布时间:** 2025-11-21

### GPT解析

### 总结

这篇论文介绍了一种名为InTAct的新方法，用于解决持续学习中的表示漂移问题，使神经网络能够在学习新任务时保持已学知识。

### 背景

持续学习旨在使神经网络获取新知识而不忘记已学信息。虽然基于提示的方法在类别增量设置中表现良好，但在领域迁移(输入分布变化但标签空间固定)情况下仍存在表示漂移问题，导致共享表示覆盖先前有用特征。

### 目的

解决持续学习中的表示漂移问题，使神经网络在领域迁移情况下保持已学知识并继续学习新任务。

### 方法

InTAct方法捕获与先前学习任务相关的特征激活范围，约束更新以确保网络在这些区域内保持一致，同时允许其他区域灵活适应。它通过稳定重要神经元的功能作用而非直接限制参数值，实现稳定性和可塑性的平衡。

### 主要发现

InTAct能稳定重要神经元功能作用，与架构无关，可无缝集成到现有基于提示的持续学习框架中。在DomainNet和ImageNet-R等基准测试中，一致减少表示漂移并提高性能，将平均准确率提升最多8个百分点。

### 结论

InTAct通过在共享层中保持功能行为，有效解决了持续学习中的表示漂移问题，实现了稳定性和可塑性之间的平衡，在各种领域增量任务中取得显著性能提升。

### 翻译

持续学习旨在使神经网络能够获取新知识而不会忘记之前学习的信息。虽然最近的基于提示的方法在类别增量设置中表现强劲，但在领域迁移(输入分布变化但标签空间保持固定)的情况下仍然很脆弱。这暴露了一个持续存在的问题，称为表示漂移。共享表示以覆盖先前有用特征的方式演变，即使提示隔离了任务特定参数，也会导致遗忘。为了解决这个问题，我们引入了InTAct，一种在不冻结参数或存储过去数据的情况下保持共享层中功能行为的方法。InTAct捕获与先前学习任务相关的特征激活范围，并约束更新以确保网络在这些区域内保持一致，同时仍然允许在其他地方灵活适应。通过这样做，InTAct稳定了重要神经元的功能作用，而不是直接限制参数值。该方法与架构无关，并可以无缝集成到现有的基于提示的持续学习框架中。通过调节过去知识编码处的表示变化，InTAct实现了稳定性和可塑性之间的平衡。在DomainNet和ImageNet-R等多样化的领域增量基准测试中，InTAct一致地减少了表示漂移并提高了性能，将平均准确率比最先进的基线提高了最多8个百分点。


### 论文摘要

Continual learning aims to enable neural networks to acquire new knowledge without forgetting previously learned information. While recent prompt-based methods perform strongly in class-incremental settings, they remain vulnerable under domain shifts, where the input distribution changes but the label space remains fixed. This exposes a persistent problem known as representation drift. Shared representations evolve in ways that overwrite previously useful features and cause forgetting even when prompts isolate task-specific parameters. To address this issue, we introduce InTAct, a method that preserves functional behavior in shared layers without freezing parameters or storing past data. InTAct captures the characteristic activation ranges associated with previously learned tasks and constrains updates to ensure the network remains consistent within these regions, while still allowing for flexible adaptation elsewhere. In doing so, InTAct stabilizes the functional role of important neurons rather than directly restricting parameter values. The approach is architecture-agnostic and integrates seamlessly into existing prompt-based continual learning frameworks. By regulating representation changes where past knowledge is encoded, InTAct achieves a principled balance between stability and plasticity. Across diverse domain-incremental benchmarks, including DomainNet and ImageNet-R, InTAct consistently reduces representation drift and improves performance, increasing Average Accuracy by up to 8 percentage points over state-of-the-art baselines.

---

## 51. Multi-Agent Pointer Transformer: Seq-to-Seq Reinforcement Learning for Multi-Vehicle Dynamic Pickup-Delivery Problems

**论文链接:** [http://arxiv.org/abs/2511.17435v1](http://arxiv.org/abs/2511.17435v1)

**作者:** Zengyu Zou, Jingyuan Wang, Yixuan Huang, Junjie Wu

**发布时间:** 2025-11-21

**备注:** 15 pages

### GPT解析

### 总结

本文提出了一种名为多智能体指针变换器（MAPT）的端到端集中式决策框架，用于解决合作式多车辆动态随机取送货问题（MVDPDPSR），该框架通过Transformer编码器提取实体表示，结合Transformer解码器和指针网络生成联合动作序列，并引入关系感知注意力模块捕捉实体间关系。

### 背景

MVDPDPSR是车辆路径问题的扩展，是一个时空系统优化问题，广泛应用于按需配送等场景。经典运筹学方法在处理大规模动态问题时面临计算复杂度和时间效率的瓶颈，现有强化学习方法也存在多车辆独立解码无法建模联合动作分布、特征提取网络难以捕捉实体间关系以及联合动作空间呈指数级增长等挑战。

### 目的

解决合作式多车辆动态随机取送货问题，设计一个能够有效处理大规模动态问题的决策框架，克服现有方法的局限性。

### 方法

提出名为多智能体指针变换器（MAPT）的框架，该框架使用Transformer编码器提取实体表示，结合Transformer解码器和指针网络以自回归方式生成联合动作序列，引入关系感知注意力模块捕捉实体间关系，并使用信息先验指导模型决策，促进有效探索。

### 主要发现

在8个数据集上的实验表明，MAPT在性能上显著优于现有基线方法，与经典运筹学方法相比具有明显的计算时间优势。

### 结论

MAPT框架为解决合作式多车辆动态随机取送货问题提供了一种有效的方法，能够克服现有方法的局限性，在性能和计算效率方面均表现出色。

### 翻译

本文解决了合作式多车辆动态随机取送货问题（MVDPDPSR），并提出了一种基于序列到序列的端到端集中式决策框架，名为多智能体指针变换器（MAPT）。MVDPDPSR是车辆路径问题的扩展，是一个时空系统优化问题，广泛应用于按需配送等场景。当处理大规模动态问题时，经典运筹学方法在计算复杂度和时间效率方面面临瓶颈。尽管现有的强化学习方法已取得一定进展，但仍面临几个挑战：1）多车辆独立解码无法建模联合动作分布；2）特征提取网络难以捕捉实体间关系；3）联合动作空间呈指数级增长。为解决这些问题，我们设计了MAPT框架，该框架采用Transformer编码器提取实体表示，结合Transformer解码器和指针网络以自回归方式生成联合动作序列，并引入关系感知注意力模块来捕捉实体间关系。此外，我们使用信息先验指导模型决策，促进有效探索。在8个数据集上的实验表明，MAPT在性能上显著优于现有基线方法，与经典运筹学方法相比具有明显的计算时间优势。


### 论文摘要

This paper addresses the cooperative Multi-Vehicle Dynamic Pickup and Delivery Problem with Stochastic Requests (MVDPDPSR) and proposes an end-to-end centralized decision-making framework based on sequence-to-sequence, named Multi-Agent Pointer Transformer (MAPT). MVDPDPSR is an extension of the vehicle routing problem and a spatio-temporal system optimization problem, widely applied in scenarios such as on-demand delivery. Classical operations research methods face bottlenecks in computational complexity and time efficiency when handling large-scale dynamic problems. Although existing reinforcement learning methods have achieved some progress, they still encounter several challenges: 1) Independent decoding across multiple vehicles fails to model joint action distributions; 2) The feature extraction network struggles to capture inter-entity relationships; 3) The joint action space is exponentially large. To address these issues, we designed the MAPT framework, which employs a Transformer Encoder to extract entity representations, combines a Transformer Decoder with a Pointer Network to generate joint action sequences in an AutoRegressive manner, and introduces a Relation-Aware Attention module to capture inter-entity relationships. Additionally, we guide the model's decision-making using informative priors to facilitate effective exploration. Experiments on 8 datasets demonstrate that MAPT significantly outperforms existing baseline methods in terms of performance and exhibits substantial computational time advantages compared to classical operations research methods.

---

## 52. Self-Supervised Learning by Curvature Alignment

**论文链接:** [http://arxiv.org/abs/2511.17426v1](http://arxiv.org/abs/2511.17426v1)

**作者:** Benyamin Ghojogh, M. Hadi Sepanj, Paul Fieguth

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了CurvSSL，一种基于曲率正则化的自监督学习框架及其RKHS扩展kernel CurvSSL，通过考虑数据流形的局部几何结构来提高表征质量。

### 背景

自监督学习(SSL)最近通过非对比方法取得了进展，这些方法将不变性项与方差、协方差或冗余减少惩罚相结合。然而，这些方法主要关注表征的一阶和二阶统计特性，而忽略了底层数据流形的局部几何结构。

### 目的

引入一个考虑局部几何结构的自监督学习框架，通过曲率正则化来提高表征质量。

### 方法

保留了标准的双视图编码器-投影器架构，并在投影特征上使用类似Barlow Twins的冗余减少损失，同时增加了基于曲率的正则化项。每个嵌入被视为一个顶点，其k个最近邻通过单位超球面上的余弦交互定义离散曲率分数；在核变体中，曲率从RKHS中的归一化局部Gram矩阵计算。这些分数通过基于曲率派生矩阵的Barlow风格损失在增强之间进行对齐和解相关。

### 主要发现

在MNIST和CIFAR-10数据集上使用ResNet-18骨干网络的实验表明，曲率正则化的自监督学习与Barlow Twins和VICReg相比具有竞争性或改进的线性评估性能。

### 结论

显式地塑造局部几何结构是纯粹统计自监督正则化器的一个简单而有效的补充。

### 翻译

自监督学习(SSL)最近通过非对比方法取得了进展，这些方法将不变性项与方差、协方差或冗余减少惩罚相结合。虽然这类目标塑造了表征的一阶和二阶统计特性，但它们 largely 忽略了底层数据流形的局部几何结构。在本文中，我们引入了CurvSSL，一种曲率正则化的自监督学习框架，及其RKHS扩展kernel CurvSSL。我们的方法保留了标准的双视图编码器-投影器架构，并在投影特征上使用了类似Barlow Twins的冗余减少损失，但通过基于曲率的正则化项增强了它。每个嵌入被视为一个顶点，其k个最近邻通过单位超球面上的余弦交互定义离散曲率分数；在核变体中，曲率从RKHS中的归一化局部Gram矩阵计算。这些分数通过基于曲率派生矩阵的Barlow风格损失在增强之间进行对齐和解相关，同时鼓励视图不变性和局部流形弯曲的一致性。在MNIST和CIFAR-10数据集上使用ResNet-18骨干网络的实验表明，曲率正则化的自监督学习与Barlow Twins和VICReg相比具有竞争性或改进的线性评估性能。我们的结果表明，显式地塑造局部几何结构是纯粹统计自监督正则化器的一个简单而有效的补充。


### 论文摘要

Self-supervised learning (SSL) has recently advanced through non-contrastive methods that couple an invariance term with variance, covariance, or redundancy-reduction penalties. While such objectives shape first- and second-order statistics of the representation, they largely ignore the local geometry of the underlying data manifold. In this paper, we introduce CurvSSL, a curvature-regularized self-supervised learning framework, and its RKHS extension, kernel CurvSSL. Our approach retains a standard two-view encoder-projector architecture with a Barlow Twins-style redundancy-reduction loss on projected features, but augments it with a curvature-based regularizer. Each embedding is treated as a vertex whose $k$ nearest neighbors define a discrete curvature score via cosine interactions on the unit hypersphere; in the kernel variant, curvature is computed from a normalized local Gram matrix in an RKHS. These scores are aligned and decorrelated across augmentations by a Barlow-style loss on a curvature-derived matrix, encouraging both view invariance and consistency of local manifold bending. Experiments on MNIST and CIFAR-10 datasets with a ResNet-18 backbone show that curvature-regularized SSL yields competitive or improved linear evaluation performance compared to Barlow Twins and VICReg. Our results indicate that explicitly shaping local geometry is a simple and effective complement to purely statistical SSL regularizers.

---

## 53. DS-Span: Single-Phase Discriminative Subgraph Mining for Efficient Graph Embeddings

**论文链接:** [http://arxiv.org/abs/2511.17419v1](http://arxiv.org/abs/2511.17419v1)

**作者:** Yeamin Kaiser, Muhammed Tasnim Bin Anwar, Bholanath Das, Chowdhury Farhan Ahmed, Md. Tanvir Alam

**发布时间:** 2025-11-21

### GPT解析

### 总结

DS-Span是一种单阶段判别式子图挖掘框架，通过统一模式生长、剪枝和监督驱动评分，在一次搜索空间遍历中高效生成具有强类别分离能力的子图特征。

### 背景

图表示学习旨在将复杂的高维图结构转换为保持拓扑和语义的紧凑向量空间。基于子图的方法提供了符号模式发现与连续嵌入学习之间的可解释桥梁，但现有方法常存在冗余多阶段流程、高计算成本以及挖掘结构与判别相关性之间弱耦合的问题。

### 目的

提出一个单阶段的判别式子图挖掘框架，统一模式生长、剪枝和监督驱动评分，在一次搜索空间遍历中完成。

### 方法

DS-Span框架引入了覆盖限制资格机制，一旦图被充分表示就动态限制探索；以及信息增益引导的选择，促进具有强类别分离能力的子图同时最小化冗余。所得子图集作为下游图嵌入和分类的高效、可解释基础。

### 主要发现

在多个基准测试上的大量实验表明，DS-Span比之前的多阶段方法生成更紧凑和判别性的子图特征，以显著减少的运行时间实现更高或相当的准确性。

### 结论

这些结果突出了统一、单阶段判别式挖掘作为可扩展和可解释图表示学习基础的潜力。

### 翻译

图表示学习寻求将复杂的高维图结构转换为保持拓扑和语义的紧凑向量空间。在各种策略中，基于子图的方法提供了符号模式发现与连续嵌入学习之间的可解释桥梁。然而，现有的频繁或判别式子图挖掘方法通常遭受冗余的多阶段流程、高计算成本以及挖掘结构与判别相关性之间弱耦合的问题。我们提出了DS-Span，这是一个单阶段判别式子图挖掘框架，在一次搜索空间遍历中统一了模式生长、剪枝和监督驱动评分。DS-Span引入了覆盖限制资格机制，一旦图被充分表示就动态限制探索，以及信息增益引导的选择，促进具有强类别分离能力的子图同时最小化冗余。所得的子图集作为下游图嵌入和分类的高效、可解释基础。在多个基准测试上的大量实验表明，DS-Span比之前的多阶段方法生成更紧凑和判别性的子图特征，以显著减少的运行时间实现更高或相当的准确性。这些结果突出了统一、单阶段判别式挖掘作为可扩展和可解释图表示学习基础的潜力。


### 论文摘要

Graph representation learning seeks to transform complex, high-dimensional graph structures into compact vector spaces that preserve both topology and semantics. Among the various strategies, subgraph-based methods provide an interpretable bridge between symbolic pattern discovery and continuous embedding learning. Yet, existing frequent or discriminative subgraph mining approaches often suffer from redundant multi-phase pipelines, high computational cost, and weak coupling between mined structures and their discriminative relevance. We propose DS-Span, a single-phase discriminative subgraph mining framework that unifies pattern growth, pruning, and supervision-driven scoring within one traversal of the search space. DS-Span introduces a coverage-capped eligibility mechanism that dynamically limits exploration once a graph is sufficiently represented, and an information-gain-guided selection that promotes subgraphs with strong class-separating ability while minimizing redundancy. The resulting subgraph set serves as an efficient, interpretable basis for downstream graph embedding and classification. Extensive experiments across benchmarks demonstrate that DS-Span generates more compact and discriminative subgraph features than prior multi-stage methods, achieving higher or comparable accuracy with significantly reduced runtime. These results highlight the potential of unified, single-phase discriminative mining as a foundation for scalable and interpretable graph representation learning.

---

## 54. Feasibility of Embodied Dynamics Based Bayesian Learning for Continuous Pursuit Motion Control of Assistive Mobile Robots in the Built Environment

**论文链接:** [http://arxiv.org/abs/2511.17401v1](http://arxiv.org/abs/2511.17401v1)

**作者:** Xiaoshan Zhou, Carol C. Menassa, Vineet R. Kamat

**发布时间:** 2025-11-21

**备注:** 37 pages, 9 figures, and 7 tables

### GPT解析

### 总结

本研究提出了一种基于脑电图(EEG)的非侵入式脑机接口(BCI)系统，用于连续追踪控制轮椅运动，解决传统BCI仅支持离散命令的局限性。

### 背景

当前大多数BCI运动控制系统仅支持离散命令，而非连续追踪控制，这种自然运动控制能力对于轮椅使用者在复杂公共空间中导航至关重要。

### 目的

解决BCI中连续追踪运动控制的空白，提出并验证一个受大脑启发的贝叶斯推理框架，实现更自然、灵活的轮椅控制。

### 方法

提出基于加速度运动表征的具身动力学解码方法，使用自动相关性确定进行特征选择，采用持续在线学习，并利用包含四个受试者执行基于运动想象的目标跟随任务的十六小时脑电图数据集进行验证。

### 主要发现

与基线方法相比，预测速度与真实速度之间的归一化均方误差降低了72%，这些发现经验性地支持了具身认知理论，揭示了大脑的内在运动控制动力学。

### 结论

将脑电图解码建立在支配生物运动的相同动力学原则上，为更稳定和直观的BCI控制提供了有希望的路径。

### 翻译

非侵入式脑电图(EEG)脑机接口为严重运动障碍者提供了直观方式，使其能独立操作辅助轮椅并在建筑环境中导航。尽管BCI研究取得显著进展，但当前大多数运动控制系统仅限于离散命令，而非支持连续追踪，用户无法实时自由调整速度和方向。然而，这种自然运动控制能力对于轮椅使用者在复杂公共空间（如交通枢纽、机场、医院和室内走廊）中导航、与动态人群灵活互动以及随着自动驾驶技术的完善实现自由移动至关重要。本研究通过提出并验证一个受大脑启发的贝叶斯推理框架，解决了BCI中连续追踪运动控制的空白，该框架解码基于加速度运动表征的具身动力学。这种方法与传统运动学级解码和基于深度学习的方法形成对比。利用包含四个受试者执行基于运动想象的目标跟随任务的十六小时脑电图公共数据集，我们证明，在会话累积迁移学习设置下，使用自动相关性确定进行特征选择和持续在线学习的方法，与自回归和基于EEGNet的方法相比，将预测速度与真实速度之间的归一化均方误差降低了72%。理论上，这些发现经验性地支持了具身认知理论，并揭示了大脑在具身和预测性质中的内在运动控制动力学。实践上，将脑电图解码建立在支配生物运动的相同动力学原则上，为更稳定和直观的BCI控制提供了有希望的路径。


### 论文摘要

Non-invasive electroencephalography (EEG)-based brain-computer interfaces (BCIs) offer an intuitive means for individuals with severe motor impairments to independently operate assistive robotic wheelchairs and navigate built environments. Despite considerable progress in BCI research, most current motion control systems are limited to discrete commands, rather than supporting continuous pursuit, where users can freely adjust speed and direction in real time. Such natural mobility control is, however, essential for wheelchair users to navigate complex public spaces, such as transit stations, airports, hospitals, and indoor corridors, to interact socially with the dynamic populations with agility, and to move flexibly and comfortably as autonomous driving is refined to allow movement at will. In this study, we address the gap of continuous pursuit motion control in BCIs by proposing and validating a brain-inspired Bayesian inference framework, where embodied dynamics in acceleration-based motor representations are decoded. This approach contrasts with conventional kinematics-level decoding and deep learning-based methods. Using a public dataset with sixteen hours of EEG from four subjects performing motor imagery-based target-following, we demonstrate that our method, utilizing Automatic Relevance Determination for feature selection and continual online learning, reduces the normalized mean squared error between predicted and true velocities by 72% compared to autoregressive and EEGNet-based methods in a session-accumulative transfer learning setting. Theoretically, these findings empirically support embodied cognition theory and reveal the brain's intrinsic motor control dynamics in an embodied and predictive nature. Practically, grounding EEG decoding in the same dynamical principles that govern biological motion offers a promising path toward more stable and intuitive BCI control.

---

## 55. MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment

**论文链接:** [http://arxiv.org/abs/2511.17397v1](http://arxiv.org/abs/2511.17397v1)

**作者:** Huangbiao Xu, Huanqi Wu, Xiao Ke, Junyi Wu, Rui Xu, Jinglin Xu

**发布时间:** 2025-11-21

**备注:** AAAI 2026

### GPT解析

### 总结

本文提出了一种基于专家混合的缺失完成框架(MCMoE)，用于解决多模态动作质量评估中模态缺失问题，在单阶段训练中统一了单模态和联合表示学习，实现了在完整和不完整多模态学习中的最先进性能。

### 背景

多模态动作质量评估(AQA)是一个新兴的研究范式，通过利用共享上下文线索中的互补信息，增强了高度相似动作序列中细微类内差异的判别性评估。然而，在实际推理阶段，部分模态经常不可用。

### 目的

解决多模态模型在模态缺失情况下无法工作的问题，提高模型在完整和不完整多模态学习中的性能。

### 方法

提出基于专家混合的缺失完成框架(MCMoE)，包含自适应门控模态生成器来重建缺失模态，设计模态专家学习单模态知识并动态混合提取跨模态联合表示，在训练阶段挖掘完整多模态特征和单模态专家知识指导模型学习。

### 主要发现

在三个公共AQA基准测试上，MCMoE在完整和不完整多模态学习中都取得了最先进的成果。

### 结论

MCMoE框架有效解决了多模态模型在模态缺失情况下的不可用问题，通过专家混合机制实现了对缺失模态的重建和补充，显著提升了模型在模态不完整条件下的性能。

### 翻译

多模态动作质量评估(AQA)最近已成为一个有前景的研究范式。通过利用共享上下文线索中的互补信息，它增强了高度相似动作序列中细微类内差异的判别性评估。然而，在实际推理阶段，部分模态经常不可用。任何模态的缺失往往使现有的多模态模型无法操作。此外，由于跨模态交互中断，它会导致灾难性的性能下降。为了解决这个问题，我们提出了一个基于专家混合的缺失完成框架(MCMoE)，在单阶段训练中统一了单模态和联合表示学习。具体来说，我们提出了一个自适应门控模态生成器，动态融合可用信息来重建缺失模态。然后我们设计了模态专家来学习单模态知识，并动态混合所有专家的知识以提取跨模态联合表示。通过专家混合，缺失模态得到进一步细化和补充。最后，在训练阶段，我们挖掘完整的多模态特征和单模态专家知识来指导模态生成和基于生成的联合表示提取。大量实验表明，我们的MCMoE在三个公共AQA基准上，在完整和不完整多模态学习中都取得了最先进的结果。代码可在 https://github.com/XuHuangbiao/MCMoE 获取。


### 论文摘要

Multimodal Action Quality Assessment (AQA) has recently emerged as a promising paradigm. By leveraging complementary information across shared contextual cues, it enhances the discriminative evaluation of subtle intra-class variations in highly similar action sequences. However, partial modalities are frequently unavailable at the inference stage in reality. The absence of any modality often renders existing multimodal models inoperable. Furthermore, it triggers catastrophic performance degradation due to interruptions in cross-modal interactions. To address this issue, we propose a novel Missing Completion Framework with Mixture of Experts (MCMoE) that unifies unimodal and joint representation learning in single-stage training. Specifically, we propose an adaptive gated modality generator that dynamically fuses available information to reconstruct missing modalities. We then design modality experts to learn unimodal knowledge and dynamically mix the knowledge of all experts to extract cross-modal joint representations. With a mixture of experts, missing modalities are further refined and complemented. Finally, in the training phase, we mine the complete multimodal features and unimodal expert knowledge to guide modality generation and generation-based joint representation extraction. Extensive experiments demonstrate that our MCMoE achieves state-of-the-art results in both complete and incomplete multimodal learning on three public AQA benchmarks. Code is available at https://github.com/XuHuangbiao/MCMoE.

---

## 56. MorphSeek: Fine-grained Latent Representation-Level Policy Optimization for Deformable Image Registration

**论文链接:** [http://arxiv.org/abs/2511.17392v1](http://arxiv.org/abs/2511.17392v1)

**作者:** Runxun Zhang, Yizhou Liu, Li Dongrui, Bo XU, Jingwei Wei

**发布时间:** 2025-11-21

### GPT解析

### 总结

MorphSeek是一种新的可变形图像注册方法，通过在潜在特征空间中进行空间连续优化，解决了高维变形空间和监督稀缺的问题。它结合了无监督预热和弱监督微调，在多个医学图像注册任务中取得了优于现有方法的性能。

### 背景

可变形图像注册是医学图像分析中的一个基础且具有挑战性的问题，主要挑战在于密集位移场的高维变形空间和体素级监督的稀缺性。现有的强化学习框架通常将高维空间投影到粗糙的低维表示，限制了捕捉空间变化变形的能力。

### 目的

提出一种细粒度的表示级策略优化范式来解决DIR问题，将DIR重新表述为潜在特征空间中的空间连续优化过程。

### 方法

MorphSeek框架在编码器之上引入随机高斯策略头来建模潜在特征的分布，促进高效探索和粗到细的细化。通过组相对策略优化整合无监督预热和弱监督微调，使用多轨迹采样稳定训练并提高标签效率。

### 主要发现

在三个3D注册基准测试上(OASIS脑部MRI、LiTS肝脏CT和腹部MR-CT)，MorphSeek实现了比竞争基线一致更高的Dice系数，在保持高标签效率的同时，具有最小的参数成本和低级别的步骤延迟开销。

### 结论

MorphSeek推动了表示级策略学习范式，实现了空间连贯和数据高效的变形优化，为高维设置中的可扩展视觉对齐提供了有原则的、与主干无关的、与优化器无关的解决方案。

### 翻译

可变形图像注册在医学图像分析中仍然是一个基础但具有挑战性的问题，这主要是由于密集位移场的高维变形空间和体素级监督的稀缺性。现有的强化学习框架通常将这个空间投影为粗糙的低维表示，限制了它们捕捉空间变化变形的能力。我们提出了MorphSeek，一种细粒度的表示级策略优化范式，它将DIR重新表述为潜在特征空间中的空间连续优化过程。MorphSeek在编码器之上引入了一个随机高斯策略头来建模潜在特征的分布，促进高效探索和粗到细的细化。该框架通过组相对策略优化整合了无监督预热和弱监督微调，其中多轨迹采样稳定了训练并提高了标签效率。在三个3D注册基准测试(OASIS脑部MRI、LiTS肝脏CT和腹部MR-CT)中，MorphSeek实现了比竞争基线一致的Dice改进，同时保持高标签效率，具有最小的参数成本和低级别的步骤延迟开销。除了优化器的具体细节外，MorphSeek推动了表示级策略学习范式，实现了空间连贯和数据高效的变形优化，为高维设置中可扩展的视觉对齐提供了有原则的、与主干无关的、与优化器无关的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决可变形图像配准中的两个核心挑战：1) 高维变形空间和体素级监督稀缺问题，导致现有强化学习框架难以捕获空间变化变形；2) 单次推理难以处理复杂大变形情况，限制了配准精度。这些问题在医学图像分析中至关重要，因为准确的配准对于图像引导手术、放射治疗规划和疾病监测等临床应用具有决定性影响，直接影响诊断和治疗质量。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有深度学习配准方法的局限性，特别是监督稀缺和单次推理处理复杂变形的不足。他们借鉴了强化学习在决策优化中的优势，但认识到直接在高维变形场上应用RL不可行。因此，作者创新性地提出将配准问题转化为潜在空间中的策略优化，通过在编码器顶部引入可采样高维潜在表示，既保留了空间细节又降低了维度复杂性。方法借鉴了VoxelMorph等编码器-解码器架构、PPO/TRPO等策略优化方法，以及无监督预训练和粗到细配准的思想，但进行了根本性重构以适应高维配准场景。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将可变形图像配准重新表述为潜在空间中的策略优化问题，通过在编码器顶层引入高维潜在空间并建模高斯分布，实现高效探索和粗到细细化。整体流程分为三阶段：1) RL友好重构：解耦编码器和解码器，添加高斯参数头以采样潜在向量；2) 无监督预训练：使用图像相似性、变形正则化和KL损失预训练模型，建立稳定潜在空间；3) GRPO弱监督微调：通过多轨迹采样和分组相对策略优化，在有限标签下进行粗到细细化，使用潜在维度方差归一化(LDVN)稳定训练，并贪婪选择最优路径更新变形场。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 潜在空间策略优化范式，首次在编码器特征空间而非直接在变形场上定义策略；2) 潜在维度方差归一化(LDVN)，解决高维GRPO训练不稳定问题；3) 多轨迹多步弱监督框架，高效重用有限标签。相比之前工作，MorphSeek突破了传统单次推理限制，解决了现有RL方法局限于低维刚性变换的问题，避免了传统粗到细方法缺乏自适应探索的缺陷，实现了更精确的复杂大变形处理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MorphSeek通过潜在空间策略优化和创新的LDVN技术，解决了医学图像配准中监督稀缺和复杂大变形处理的挑战，实现了高精度、高效率且资源友好的可变形图像配准。'}


### 论文摘要

Deformable image registration (DIR) remains a fundamental yet challenging problem in medical image analysis, largely due to the prohibitively high-dimensional deformation space of dense displacement fields and the scarcity of voxel-level supervision. Existing reinforcement learning frameworks often project this space into coarse, low-dimensional representations, limiting their ability to capture spatially variant deformations. We propose MorphSeek, a fine-grained representation-level policy optimization paradigm that reformulates DIR as a spatially continuous optimization process in the latent feature space. MorphSeek introduces a stochastic Gaussian policy head atop the encoder to model a distribution over latent features, facilitating efficient exploration and coarse-to-fine refinement. The framework integrates unsupervised warm-up with weakly supervised fine-tuning through Group Relative Policy Optimization, where multi-trajectory sampling stabilizes training and improves label efficiency. Across three 3D registration benchmarks (OASIS brain MRI, LiTS liver CT, and Abdomen MR-CT), MorphSeek achieves consistent Dice improvements over competitive baselines while maintaining high label efficiency with minimal parameter cost and low step-level latency overhead. Beyond optimizer specifics, MorphSeek advances a representation-level policy learning paradigm that achieves spatially coherent and data-efficient deformation optimization, offering a principled, backbone-agnostic, and optimizer-agnostic solution for scalable visual alignment in high-dimensional settings.

---

## 57. Deep Learning Analysis of Ions Accelerated at Shocks

**论文链接:** [http://arxiv.org/abs/2511.17363v1](http://arxiv.org/abs/2511.17363v1)

**作者:** Paxson Swierc, Damiano Caprioli, Luca Orusa, Miha Cernetic

**发布时间:** 2025-11-21

**备注:** 21 pages, 8 figures. Submitted to JCAP

### GPT解析

### 总结

本研究探讨了深度学习技术在分析混合模拟中碰撞激波加速离子的应用，展示了机器学习从动力学等离子体模拟中提取物理洞察力的潜力。

### 背景

研究聚焦于在混合（离子动力学-电子流体）模拟中分析碰撞激波加速的离子，这类模拟在等离子体物理学中具有重要意义。

### 目的

应用深度学习技术对离子进行分类，并预测哪些粒子被高效注入到加速过程中，为未来构建流体方法中的子网格模型奠定基础。

### 方法

根据离子获得的能量和所处加速机制将离子分为热离子、超热离子和非热离子；使用离子与激波初始相互作用期间经历的局部磁场时间序列训练深度学习模型；测试自编码器架构用于参数时间序列重建。

### 主要发现

深度学习模型能够以超过90%的准确率预测哪些粒子被注入到加速过程中，仅使用局部磁场的时间序列即可实现高精度预测。

### 结论

机器学习技术可有效从动力学等离子体模拟中提取物理洞察力，为等离子体物理研究提供了新的分析工具和方法。

### 翻译

我们研究了深度学习技术在分析混合（离子动力学-电子流体）模拟中碰撞激波加速离子的应用和分类。根据离子获得的能量和所处的加速机制，离子被分为热离子、超热离子或非热离子。这些分类被用于训练深度学习模型，以高精度（>90%）预测哪些粒子被注入到加速过程中，仅使用粒子在初始与激波相互作用期间经历的局部磁场时间序列。还测试了自编码器架构，用于从编码表示中重建各种参数的时间序列。这项研究展示了应用机器学习技术从动力学等离子体模拟中提取物理洞察力的潜力，并为未来应用（包括在流体方法中构建子网格模型）奠定了基础。


### 论文摘要

We study the application of deep learning techniques to the analysis and classification of ions accelerated at collisionless shocks in hybrid (kinetic ions--fluid electrons) simulations. Ions were classified as thermal, suprathermal, or nonthermal, depending on the energy they achieved and the acceleration regime they fell under. These classifications were used to train deep learning models to predict which particles are injected into the acceleration process with high accuracy (>90%), using only time series of the local magnetic field they experienced during their initial interaction with the shock. An autoencoder architecture was also tested, for which time series of various parameters were reconstructed from encoded representations. This study shows the potential of applying machine learning techniques to extract physical insights from kinetic plasma simulations and sets the groundwork for future applications, including the construction of sub-grid models in fluid approaches.

---

## 58. DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture

**论文链接:** [http://arxiv.org/abs/2511.17354v1](http://arxiv.org/abs/2511.17354v1)

**作者:** Xiangteng He, Shunsuke Sakai, Kun Yuan, Nicolas Padoy, Tatsuhito Hasegawa, Leonid Sigal

**发布时间:** 2025-11-21

**备注:** Project page: https://github.com/SkyShunsuke/DSeq-JEPA

### GPT解析

### 总结

DSeq-JEPA是一种新的视觉表示学习架构，结合了预测性和自回归自监督学习，通过识别主要判别区域并按顺序预测后续区域，形成课程式语义进展，在多种视觉任务上表现优于I-JEPA。

### 背景

I-JEPA通过从可见上下文中预测掩码区域的潜在嵌入来学习视觉表示，但它对所有区域统一且独立地处理，缺乏关于预测位置或顺序的明确概念。

### 目的

受人类视觉感知启发，提出DSeq-JEPA，整合JEPA式潜在预测与GPT式顺序推理，选择性且顺序地处理视觉区域。

### 方法

DSeq-JEPA首先基于转换器导出的显著性图识别主要判别区域，然后按此判别顺序预测后续区域，形成从主要到次要的课程式语义进展。

### 主要发现

在图像分类、细粒度视觉分类、检测和分割、低级推理任务等多种任务上，DSeq-JEPA比I-JEPA变体更专注于判别性和可泛化的表示。

### 结论

DSeq-JEPA通过整合预测性和自回归自监督学习方法，有效提升了视觉表示学习的性能。

### 翻译

基于图像的联合嵌入预测架构(I-JEPA)通过从可见上下文中预测掩码区域的潜在嵌入来学习视觉表示。然而，它对所有区域统一且独立地处理，缺乏关于预测位置或顺序的明确概念。受人类视觉感知启发，它选择性地且顺序地从最信息丰富的区域到次要区域部署注意力，我们提出了DSeq-JEPA，一种判别性顺序联合嵌入预测架构，它连接了预测性和自回归自监督学习，整合了JEPA式的潜在预测与GPT式的顺序推理。具体而言，DSeq-JEPA首先基于转换器导出的显著性图识别主要判别区域，然后按此判别顺序预测后续区域，逐渐形成从主要到次要线索的课程式语义进展——一种GPT式的预训练形式。在图像分类(iNaturalist21、CUB-200-2011、Stanford-Cars)、检测和分割(MS-COCO、ADE20K)以及低级推理任务(Clevr/Count、Clevr/Dist)等多种任务上的广泛实验表明，DSeq-JEPA比I-JEPA变体更专注于判别性和可泛化的表示。项目页面：https://github.com/SkyShunsuke/DSeq-JEPA。


### 论文摘要

Image-based Joint-Embedding Predictive Architecture (I-JEPA) learns visual representations by predicting latent embeddings of masked regions from visible context. However, it treats all regions uniformly and independently, lacking an explicit notion of where or in what order predictions should be made. Inspired by human visual perception, which deploys attention selectively and sequentially from the most informative to secondary regions, we propose DSeq-JEPA, a Discriminative Sequential Joint-Embedding Predictive Architecture that bridges predictive and autoregressive self-supervised learning, integrating JEPA-style latent prediction with GPT-style sequential reasoning. Specifically, DSeq-JEPA (i) first identifies primary discriminative regions based on a transformer-derived saliency map, emphasizing the distribution of visual importance, and then (ii) predicts subsequent regions in this discriminative order, progressively forming a curriculum-like semantic progression from primary to secondary cues -- a form of GPT-style pre-training. Extensive experiments across diverse tasks, including image classification (ImageNet), fine-grained visual categorization (iNaturalist21, CUB-200-2011, Stanford-Cars), detection and segmentation (MS-COCO, ADE20K), and low-level reasoning tasks (Clevr/Count, Clevr/Dist), demonstrate that DSeq-JEPA consistently focuses on more discriminative and generalizable representations than I-JEPA variants. Project page: https://github.com/SkyShunsuke/DSeq-JEPA.

---

## 59. MolSight: Optical Chemical Structure Recognition with SMILES Pretraining, Multi-Granularity Learning and Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.17300v1](http://arxiv.org/abs/2511.17300v1)

**作者:** Wenrui Zhang, Xinggang Wang, Bin Feng, Wenyu Liu

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了MolSight，一个用于光学化学结构识别(OCSR)的全面学习框架，通过三阶段训练范式解决了现有系统在识别立体化学信息方面的挑战，并取得了最先进的性能。

### 背景

OCSR在现代化学信息学中扮演关键角色，能将科学文献、专利和教育材料中的化学结构图像自动转换为机器可读的分子表示，这对大规模化学数据挖掘、药物发现流程和相关领域的大语言模型应用至关重要。然而，现有系统在准确识别立体化学信息方面面临挑战，因为立体异构体间的区别在于微妙的视觉线索。

### 目的

开发一个能够准确识别立体化学信息的光学化学结构识别系统，解决现有系统在识别立体化学信息方面的挑战。

### 方法

提出MolSight框架，采用三阶段训练范式：第一阶段在大型但有噪声的数据集上进行预训练，使模型具备基本感知能力；第二阶段使用更丰富监督信号的数据集进行多粒度微调，探索辅助任务（化学键分类和原子定位）对分子式识别的贡献；第三阶段使用强化学习进行后训练优化，并引入新的立体化学结构数据集。

### 主要发现

即使MolSight的参数相对紧凑，群相对策略优化(GRPO)算法也能进一步提高模型在立体分子识别方面的性能。通过在多个数据集上的广泛实验，MolSight在(立体)化学光学结构识别方面取得了最先进的性能。

### 结论

MolSight框架通过三阶段训练范式和强化学习优化，有效解决了OCSR系统在识别立体化学信息方面的挑战，实现了最先进的性能。

### 翻译

光学化学结构识别(OCSR)在现代化学信息学中扮演着关键角色，能够将科学文献、专利和教育材料中的化学结构图像自动转换为机器可读的分子表示。这种能力对大规模化学数据挖掘、药物发现流程以及相关领域的大型语言模型应用至关重要。然而，由于区分立体异构体的微妙视觉线索（如楔形和虚线键、环构象和空间排列），现有OCSR系统在准确识别立体化学信息方面面临重大挑战。为应对这些挑战，我们提出了MolSight，一个用于OCSR的全面学习框架，采用三阶段训练范式。在第一阶段，我们在大型但有噪声的数据集上进行预训练，赋予模型对化学结构图像的基本感知能力。在第二阶段，我们使用具有更丰富监督信号的数据集进行多粒度微调，系统探索辅助任务（特别是化学键分类和原子定位）如何贡献于分子式识别。最后，我们使用强化学习进行后训练优化，并引入了一个新的立体化学结构数据集。值得注意的是，我们发现即使MolSight的参数相对紧凑，群相对策略优化(GRPO)算法也能进一步提高模型在立体分子识别方面的性能。通过在多个数据集上的广泛实验，我们的结果表明MolSight在(立体)化学光学结构识别方面取得了最先进的性能。


### 论文摘要

Optical Chemical Structure Recognition (OCSR) plays a pivotal role in modern chemical informatics, enabling the automated conversion of chemical structure images from scientific literature, patents, and educational materials into machine-readable molecular representations. This capability is essential for large-scale chemical data mining, drug discovery pipelines, and Large Language Model (LLM) applications in related domains. However, existing OCSR systems face significant challenges in accurately recognizing stereochemical information due to the subtle visual cues that distinguish stereoisomers, such as wedge and dash bonds, ring conformations, and spatial arrangements. To address these challenges, we propose MolSight, a comprehensive learning framework for OCSR that employs a three-stage training paradigm. In the first stage, we conduct pre-training on large-scale but noisy datasets to endow the model with fundamental perception capabilities for chemical structure images. In the second stage, we perform multi-granularity fine-tuning using datasets with richer supervisory signals, systematically exploring how auxiliary tasks-specifically chemical bond classification and atom localization-contribute to molecular formula recognition. Finally, we employ reinforcement learning for post-training optimization and introduce a novel stereochemical structure dataset. Remarkably, we find that even with MolSight's relatively compact parameter size, the Group Relative Policy Optimization (GRPO) algorithm can further enhance the model's performance on stereomolecular. Through extensive experiments across diverse datasets, our results demonstrate that MolSight achieves state-of-the-art performance in (stereo)chemical optical structure recognition.

---

## 60. A First Full Physics Benchmark for Highly Granular Calorimeter Surrogates

**论文链接:** [http://arxiv.org/abs/2511.17293v1](http://arxiv.org/abs/2511.17293v1)

**作者:** Thorsten Buss, Henry Day-Hall, Frank Gaede, Gregor Kasieczka, Katja Krüger, Anatolii Korol, Thomas Madlener, Peter McKeown

**发布时间:** 2025-11-21

**备注:** 26 pages, 15 figures

### GPT解析

### 总结

本研究探讨了在高粒度量热器模拟中使用生成式替代模拟器的应用，引入了DDML库，比较了两种生成模型（规则网格和点云），并通过多种基准测试评估了它们的性能。

### 背景

当前和未来的对撞机实验物理项目需要开发量热器簇射的替代模拟器。虽然生成模型在此领域已有进展，但通常在简化的场景和单粒子情况下进行评估，高粒度量热器模拟尤其面临挑战。

### 目的

首次评估高粒度量热器替代模拟器在真实模拟应用中的使用，寻找速度和准确性之间的平衡点。

### 方法

引入DDML库结合生成式量热器替代模拟器与DD4hep实现的真实探测器；比较规则网格和点云两种生成模型；提供与理想化模拟器的对比；在电磁簇射模拟的后重建基准上评估模型性能，包括单粒子、双光子分离和τ轻子强子衰变等场景。

### 主要发现

点云生成模型在高粒度量热器模拟中相比规则网格模型能更好地平衡速度和准确性。

### 结论

点云生成模型在高粒度量热器模拟中具有优势，能够更好地平衡计算速度和模拟精度，对未来对撞机实验的物理程序具有重要意义。

### 翻译

当前和未来对撞机实验的物理项目需要开发量热器簇射的替代模拟器。尽管在此任务的生成模型开发方面已取得很大进展，但它们通常在简化的场景和单粒子情况下进行评估。对于具有挑战性的高粒度量热器模拟任务尤其如此。本研究首次探讨了在高粒度量热器替代模拟器在真实模拟应用中的使用。我们引入了DDML，一个通用库，它能够将生成式量热器替代模拟器与使用DD4hep工具包实现的真实探测器相结合。我们比较了两种不同的生成模型 - 一种在规则网格表示上运行，另一种使用不太常见的点云方法。为了将方法细节与模型性能区分开，我们提供了与理想化模拟器的比较，这些模拟器直接从全模拟真实数据中采样不同分辨率的表示。然后，我们在电磁簇射模拟的后重建基准上系统地评估了模型性能。从典型的单粒子研究开始，我们引入了基于双光子分离的第一个多粒子基准，然后研究了基于τ轻子强子衰变的第一个全物理基准。我们的结果表明，与在规则网格表示上运行的模型相比，在点云上运行的模型在高粒度量热器模拟中可以实现速度和准确性之间的良好平衡。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决高粒度量热器模拟中生成替代模型的准确性和效率问题。这个问题在现实中很重要，因为现代和未来对撞机实验需要准确高效的粒子探测器模拟，传统蒙特卡洛方法虽然准确但计算成本极高，而高粒度量热器能够解析簇射的更精细子结构，需要生成替代器在更高维度的数据上运行，但这些模型在真实实验环境中的表现尚未得到充分评估。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别到现有生成模型评估的局限性，即主要在简化的单粒子场景下评估，缺乏真实实验环境中的测试。他们借鉴了现有的生成模型架构（如ConvL2LFlows和CaloClouds3）、DD4hep工具包以及CaloChallenge 2022的比较框架。通过系统性地设计从单粒子到多粒子再到全物理基准的测试方法，并引入DDML库将生成模型与真实探测器集成，解决了高粒度量热器模拟中的挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是开发能够处理高粒度量热器复杂几何结构的生成模型，并通过系统性的基准测试评估其在真实物理场景中的性能。整体流程包括：1) 使用Geant4生成大量光子簇射样本；2) 开发两种不同数据表示的生成模型（规则网格的ConvL2LFlows和点云的CaloClouds3）；3) 通过DDML库将模型集成到ILD探测器模拟链中；4) 进行三种基准测试（单粒子、双光子分离和τ衰变物理过程）；5) 评估关键指标并与理想化模型和Geant4参考进行比较。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次在高粒度量热器模拟中进行全物理基准测试；开发DDML通用库实现生成模型与真实探测器的集成；系统比较规则网格和点云两种数据表示方法；引入多层次的基准测试方法；提供理想化模型作为参考分离数据表示和建模假设的影响。相比之前工作，本研究不仅限于简化的单粒子场景，而是在真实实验环境中评估模型；探索了点云方法在高粒度量热器中的优势；提供了包括复杂τ衰变过程的全面物理基准测试；考虑了实际探测器几何不规则性的影响。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次为高粒度量热器替代模拟器建立了全物理基准测试，证明了点云模型在速度和准确性之间能取得比传统网格模型更有利的平衡，为未来对撞机实验的快速模拟提供了实用解决方案。'}


### 论文摘要

The physics programs of current and future collider experiments necessitate the development of surrogate simulators for calorimeter showers. While much progress has been made in the development of generative models for this task, they have typically been evaluated in simplified scenarios and for single particles. This is particularly true for the challenging task of highly granular calorimeter simulation. For the first time, this work studies the use of highly granular generative calorimeter surrogates in a realistic simulation application. We introduce DDML, a generic library which enables the combination of generative calorimeter surrogates with realistic detectors implemented using the DD4hep toolkit. We compare two different generative models - one operating on a regular grid representation, and the other using a less common point cloud approach. In order to disentangle methodological details from model performance, we provide comparisons to idealized simulators which directly sample representations of different resolutions from the full simulation ground-truth. We then systematically evaluate model performance on post-reconstruction benchmarks for electromagnetic shower simulation. Beginning with a typical single particle study, we introduce a first multi-particle benchmark based on di-photon separations, before studying a first full-physics benchmark based on hadronic decays of the tau lepton. Our results indicate that models operating on a point cloud can achieve a favorable balance between speed and accuracy for highly granular calorimeter simulation compared to those which operate on a regular grid representation.

---

## 61. Cross-cultural value alignment frameworks for responsible AI governance: Evidence from China-West comparative analysis

**论文链接:** [http://arxiv.org/abs/2511.17256v1](http://arxiv.org/abs/2511.17256v1)

**作者:** Haijiang Liu, Jinguang Gu, Xun Wu, Daniel Hershcovich, Qiaoling Xiao

**发布时间:** 2025-11-21

**备注:** Presented on Academic Conference "Technology for Good: Driving Social Impact" (2025)

### GPT解析

### 总结

该研究提出了一个负责任AI的多层次审计平台，通过四种集成方法论系统性评估中国和西方起源的大型语言模型在跨文化价值观对齐方面的表现，揭示了普遍性挑战和区域发展差异。

### 背景

大型语言模型(LLMs)在全球范围内越来越多地影响高风险决策，确保这些模型与多元文化价值观保持一致已成为关键的治理挑战。

### 目的

提出一个负责任AI的多层次审计平台，系统性评估中国和西方起源的LLMs在跨文化价值观对齐方面的表现。

### 方法

采用四种集成方法论：伦理困境语料库评估时间稳定性，多样性增强框架量化文化保真度，首令牌概率对齐评估分布准确性，多阶段推理框架提供可解释的决策过程。研究对20多个领先模型进行了比较分析，包括Qwen、GPT-4o、Claude、LLaMA和DeepSeek等。

### 主要发现

发现了普遍性挑战：价值观体系的基本不稳定性、年轻人口群体的系统性代表性不足、模型规模与对齐质量之间的非线性关系。区域发展轨迹存在差异：中国起源的模型强调多语言数据集成，西方模型展示更多架构实验但存在以美国为中心的偏见。两种范式都未能实现强大的跨文化泛化能力。Mistral系列架构在跨文化对齐方面显著优于LLaMA3系列，全参数微调比基于人类反馈的强化学习更能保留文化多样性。

### 结论

大型语言模型的跨文化对齐能力需要进一步研究和改进，不同地区的模型发展路径和特点各异，架构选择和训练方法对文化对齐有显著影响。

### 翻译

随着大型语言模型(LLMs)在全球范围内越来越多地影响高风险决策，确保它们与多元文化价值观保持一致已成为关键的治理挑战。本研究提出了一个负责任AI的多层次审计平台，通过四种集成方法论系统性评估中国和西方起源的LLMs的跨文化价值观对齐情况：伦理困境语料库用于评估时间稳定性，多样性增强框架用于量化文化保真度，首令牌概率对齐用于评估分布准确性，多阶段推理框架用于提供可解释的决策过程。我们对20多个领先模型（如Qwen、GPT-4o、Claude、LLaMA和DeepSeek）的比较分析揭示了普遍性挑战——价值观体系的基本不稳定性、年轻人口群体的系统性代表性不足，以及模型规模与对齐质量之间的非线性关系——同时区域发展轨迹也存在差异。虽然中国起源的模型越来越强调多语言数据集成以实现特定上下文的优化，但西方模型展示了更多的架构实验，同时持续存在以美国为中心的偏见。两种范式都未能实现强大的跨文化泛化能力。我们确定Mistral系列架构在跨文化对齐方面显著优于LLaMA3系列，并且在多样化数据集上进行全参数微调比基于人类反馈的强化学习更能保留文化多样性。


### 论文摘要

As Large Language Models (LLMs) increasingly influence high-stakes decision-making across global contexts, ensuring their alignment with diverse cultural values has become a critical governance challenge. This study presents a Multi-Layered Auditing Platform for Responsible AI that systematically evaluates cross-cultural value alignment in China-origin and Western-origin LLMs through four integrated methodologies: Ethical Dilemma Corpus for assessing temporal stability, Diversity-Enhanced Framework (DEF) for quantifying cultural fidelity, First-Token Probability Alignment for distributional accuracy, and Multi-stAge Reasoning frameworK (MARK) for interpretable decision-making. Our comparative analysis of 20+ leading models, such as Qwen, GPT-4o, Claude, LLaMA, and DeepSeek, reveals universal challenges-fundamental instability in value systems, systematic under-representation of younger demographics, and non-linear relationships between model scale and alignment quality-alongside divergent regional development trajectories. While China-origin models increasingly emphasize multilingual data integration for context-specific optimization, Western models demonstrate greater architectural experimentation but persistent U.S.-centric biases. Neither paradigm achieves robust cross-cultural generalization. We establish that Mistral-series architectures significantly outperform LLaMA3-series in cross-cultural alignment, and that Full-Parameter Fine-Tuning on diverse datasets surpasses Reinforcement Learning from Human Feedback in preserving cultural variation...

---

## 62. PostCam: Camera-Controllable Novel-View Video Generation with Query-Shared Cross-Attention

**论文链接:** [http://arxiv.org/abs/2511.17185v1](http://arxiv.org/abs/2511.17185v1)

**作者:** Yipeng Chen, Zhichao Ye, Zhenzhou Fang, Xinyu Chen, Xiaoyu Zhang, Jialing Liu, Nan Wang, Haomin Liu, Guofeng Zhang

**发布时间:** 2025-11-21

### GPT解析

### 总结

PostCam是一种新颖视角视频生成框架，能够对动态场景中的相机轨迹进行后捕获编辑，解决了现有方法相机运动注入策略次优的问题。

### 背景

现有视频重新捕获方法存在次优的相机运动注入策略，限制了相机控制精确度，导致生成的视频无法保留源视频的精细视觉细节。

### 目的

实现更准确和灵活的运动操作，提高相机控制精确度和视频生成质量。

### 方法

PostCam引入查询共享的交叉注意力模块，整合6自由度相机姿态和2D渲染视频帧两种控制信号，通过在共享特征空间中融合它们提取潜在运动线索。采用两阶段训练策略：先学习粗略相机控制，再整合视觉信息优化运动精度和视觉保真度。

### 主要发现

在真实世界和合成数据集上的实验表明，PostCam在相机控制精确度和视图一致性方面比最先进方法提高20%以上，同时实现最高视频生成质量。

### 结论

PostCam是一种有效的视频生成框架，能够实现相机轨迹的后捕获编辑，并在性能上显著优于现有方法。

### 翻译

我们提出了PostCam，一种用于新颖视角视频生成的框架，能够对动态场景中的相机轨迹进行后捕获编辑。我们发现现有的视频重新捕获方法存在次优的相机运动注入策略；这种次优设计不仅限制了相机控制的精确度，还导致生成的视频无法保留源视频的精细视觉细节。为实现更准确和灵活的运动操作，PostCam引入了一个查询共享的交叉注意力模块。它整合了两种不同的控制信号：6自由度相机姿态和2D渲染的视频帧。通过将它们融合到共享特征空间中的统一表示中，我们的模型可以提取潜在的运动线索，从而提高控制精确度和生成质量。此外，我们采用两阶段训练策略：模型首先从姿态输入学习粗略的相机控制，然后整合视觉信息以优化运动精度和提高视觉保真度。在真实世界和合成数据集上的实验表明，PostCam在相机控制精确度和视图一致性方面比最先进方法提高20%以上，同时实现最高的视频生成质量。我们的项目网页可在 https://cccqaq.github.io/PostCam.github.io/ 公开获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何实现相机可控的新视角视频生成问题，特别是允许在拍摄后编辑动态场景中的相机运动轨迹。这个问题在现实中非常重要，因为电影和视频制作中的相机运动效果通常需要昂贵的硬件和专业技能，普通创作者难以实现。此外，现有的视频重新拍摄方法存在相机控制精度有限且难以保留源视频精细视觉细节的问题，限制了其在增强现实、虚拟现实和电影制作等领域的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性：基于姿势的方法（如ReCamMaster）泛化能力有限，常导致模糊和身份漂移；基于渲染的方法（如TrajectoryCrafter）高度依赖深度估计准确性；简单的模态融合方案不够优化。基于这些分析，作者借鉴了视频扩散模型（Wan2.1）作为基础架构，相机控制视频生成中的条件注入机制，以及3D点云渲染提供视觉信息的思想。但创新性地设计了查询共享交叉注意力模块和两阶段训练策略来解决现有方法的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过查询共享交叉注意力模块将6自由度相机姿势和2D渲染视频帧融合到统一的共享特征表示中，并采用两阶段训练策略：第一阶段仅从姿势输入学习粗略相机控制，第二阶段整合视觉信息以细化运动精度并增强视觉保真度。整体流程包括：1)预处理源视频、相机参数和渲染视频；2)通过查询共享交叉注意力模块融合这些信息；3)两阶段训练学习相机控制和视觉细节；4)根据源视频和目标相机轨迹生成新视角视频。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)查询共享交叉注意力模块，有效融合相机姿势和渲染视频到共享特征空间；2)两阶段训练策略，先学习基本相机运动再整合视觉信息；3)轻量级架构，仅13亿参数实现高质量生成。相比之前工作，PostCam不仅融合了视觉和数值信号，解决了基于姿势方法的弱视觉-姿势对齐问题；而且即使在不准确的深度图情况下也能生成正确运动，克服了基于渲染方法的深度敏感性；通过查询共享机制提取共享运动线索，实现了更精确的控制和更好的细节保留。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PostCam通过创新的查询共享交叉注意力和两阶段训练策略，实现了在动态场景中精确后拍摄相机轨迹编辑的同时保持高保真度视频生成，显著超越了现有方法在相机控制精度和视角一致性方面的性能。'}


### 论文摘要

We propose PostCam, a framework for novel-view video generation that enables post-capture editing of camera trajectories in dynamic scenes. We find that existing video recapture methods suffer from suboptimal camera motion injection strategies; such suboptimal designs not only limit camera control precision but also result in generated videos that fail to preserve fine visual details from the source video. To achieve more accurate and flexible motion manipulation, PostCam introduces a query-shared cross-attention module. It integrates two distinct forms of control signals: the 6-DoF camera poses and the 2D rendered video frames. By fusing them into a unified representation within a shared feature space, our model can extract underlying motion cues, which enhances both control precision and generation quality. Furthermore, we adopt a two-stage training strategy: the model first learns coarse camera control from pose inputs, and then incorporates visual information to refine motion accuracy and enhance visual fidelity. Experiments on both real-world and synthetic datasets demonstrate that PostCam outperforms state-of-the-art methods by over 20% in camera control precision and view consistency, while achieving the highest video generation quality. Our project webpage is publicly available at: https://cccqaq.github.io/PostCam.github.io/

---

## 63. Attention-Guided Feature Fusion (AGFF) Model for Integrating Statistical and Semantic Features in News Text Classification

**论文链接:** [http://arxiv.org/abs/2511.17184v1](http://arxiv.org/abs/2511.17184v1)

**作者:** Mohammad Zare

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文介绍了一种注意力引导特征融合（AGFF）模型，用于新闻文本分类，该模型结合统计特征和语义特征，通过注意力机制动态确定各类特征的重要性，在基准数据集上表现出优越性能。

### 背景

新闻文本分类是自然语言处理中的重要任务，对组织和过滤大量数字内容至关重要。传统方法依赖统计特征（如词频或TF-IDF值），能捕捉词级重要性但无法反映上下文含义；现代深度学习方法利用语义特征理解上下文，但可能忽略简单而影响重大的统计指标。

### 目的

开发一种能够结合统计特征和语义特征优势，提高新闻文本分类准确性的方法。

### 方法

提出注意力引导特征融合（AGFF）模型，在统一框架中结合统计和语义特征，应用基于注意力的机制动态确定每种特征类型的相对重要性，以实现更明智的分类决策。

### 主要发现

在基准新闻数据集上，AGFF模型相比传统统计模型和纯语义深度学习模型表现出优越性能；消融研究验证了融合过程中每个组件的贡献；研究结果强调了模型平衡和利用统计和语义表示互补优势的能力。

### 结论

战略性整合不同类型的特征可以显著提高分类准确性；该模型能够平衡和利用统计和语义表示的互补优势，是实际新闻分类任务中实用且有效的解决方案。

### 翻译

新闻文本分类是自然语言处理中的关键任务，对于组织和过滤大量数字内容至关重要。传统方法通常依赖词频或TF-IDF值等统计特征，这些特征能有效捕捉词级重要性，但往往无法反映上下文含义。相比之下，现代深度学习方法利用语义特征来理解单词在上下文中的使用，但可能会忽略简单且影响重大的统计指标。本文引入了一种注意力引导特征融合（AGFF）模型，该模型在一个统一框架中结合统计和语义特征。该模型应用基于注意力的机制来动态确定每种特征类型的相对重要性，从而能够做出更明智的分类决策。通过在基准新闻数据集上的评估，AGFF模型相比传统统计模型和纯语义深度学习模型表现出优越的性能。结果证实，战略性整合不同类型的特征可以显著提高分类准确性。此外，消融研究验证了融合过程中每个组件的贡献。研究结果强调了模型平衡和利用统计和语义表示互补优势的能力，使其成为实际新闻分类任务中实用且有效的解决方案。


### 论文摘要

News text classification is a crucial task in natural language processing, essential for organizing and filtering the massive volume of digital content. Traditional methods typically rely on statistical features like term frequencies or TF-IDF values, which are effective at capturing word-level importance but often fail to reflect contextual meaning. In contrast, modern deep learning approaches utilize semantic features to understand word usage within context, yet they may overlook simple, high-impact statistical indicators. This paper introduces an Attention-Guided Feature Fusion (AGFF) model that combines statistical and semantic features in a unified framework. The model applies an attention-based mechanism to dynamically determine the relative importance of each feature type, enabling more informed classification decisions. Through evaluation on benchmark news datasets, the AGFF model demonstrates superior performance compared to both traditional statistical models and purely semantic deep learning models. The results confirm that strategic integration of diverse feature types can significantly enhance classification accuracy. Additionally, ablation studies validate the contribution of each component in the fusion process. The findings highlight the model's ability to balance and exploit the complementary strengths of statistical and semantic representations, making it a practical and effective solution for real-world news classification tasks.

---

## 64. Investigating self-supervised representations for audio-visual deepfake detection

**论文链接:** [http://arxiv.org/abs/2511.17181v1](http://arxiv.org/abs/2511.17181v1)

**作者:** Dragos-Alexandru Boldisor, Stefan Smeu, Dan Oneata, Elisabeta Oneata

**发布时间:** 2025-11-21

### GPT解析

### 总结

该研究系统评估了自监督表示在音频-视觉深度伪造检测中的应用，发现大多数自监督特征能捕获与深度伪造相关的互补信息，模型主要关注语义区域而非伪影，但缺乏跨数据集的泛化能力。

### 背景

自监督表示在视觉和语音任务中表现出色，但在音频-视觉深度伪造检测中的潜力尚未被充分探索。

### 目的

系统性地评估自监督特征在不同模态（音频、视频、多模态）和领域（唇部运动、通用视觉内容）中的应用，并评估三个关键维度：检测有效性、编码信息的可解释性和跨模态互补性。

### 方法

与之前单独使用这些特征或将它们埋藏在复杂架构中的研究不同，本研究系统地评估了自监督特征在不同模态和领域的应用。

### 主要发现

大多数自监督特征捕获了与深度伪造相关的互补信息；模型主要关注语义上有意义的区域而非伪影；但没有一个模型能在不同数据集上可靠地泛化；这种泛化失败可能源于数据集特征而非特征本身依附于表面模式。

### 结论

自监督表示在深度伪造检测中既有前景也面临基本挑战：虽然它们学习有意义的模式，但实现稳健的跨域性能仍然难以实现。

### 翻译

自监督表示在许多视觉和语音任务中表现出色，但它们在音频-视觉深度伪造检测中的潜力仍未被充分探索。与之前孤立使用这些特征或将它们埋藏在复杂架构中的工作不同，我们系统地评估了它们在不同模态（音频、视频、多模态）和领域（唇部运动、通用视觉内容）中的应用。我们评估了三个关键维度：检测有效性、编码信息的可解释性和跨模态互补性。我们发现大多数自监督特征捕获了与深度伪造相关的信息，并且这些信息是互补的。此外，模型主要关注语义上有意义的区域，而非伪影。然而，没有一个模型能在不同数据集上可靠地泛化。这种泛化失败可能源于数据集特征，而非特征本身依附于表面模式。这些结果揭示了自监督表示在深度伪造检测中的前景和基本挑战：虽然它们学习有意义的模式，但实现稳健的跨域性能仍然难以实现。


### 论文摘要

Self-supervised representations excel at many vision and speech tasks, but their potential for audio-visual deepfake detection remains underexplored. Unlike prior work that uses these features in isolation or buried within complex architectures, we systematically evaluate them across modalities (audio, video, multimodal) and domains (lip movements, generic visual content). We assess three key dimensions: detection effectiveness, interpretability of encoded information, and cross-modal complementarity. We find that most self-supervised features capture deepfake-relevant information, and that this information is complementary. Moreover, models primarily attend to semantically meaningful regions rather than spurious artifacts. Yet none generalize reliably across datasets. This generalization failure likely stems from dataset characteristics, not from the features themselves latching onto superficial patterns. These results expose both the promise and fundamental challenges of self-supervised representations for deepfake detection: while they learn meaningful patterns, achieving robust cross-domain performance remains elusive.

---

## 65. Learning to Compress: Unlocking the Potential of Large Language Models for Text Representation

**论文链接:** [http://arxiv.org/abs/2511.17129v1](http://arxiv.org/abs/2511.17129v1)

**作者:** Yeqin Zhang, Yizheng Zhao, Chen Hu, Binxing Jiao, Daxin Jiang, Ruihang Miao, Cam-Tu Nguyen

**发布时间:** 2025-11-21

**备注:** Accepted by AAAI'26

### GPT解析

### 总结

本文提出了一种使用上下文压缩作为预训练任务的无监督方法来增强大语言模型的文本表示能力，通过生成紧凑的记忆令牌替代整个上下文，并进一步结合对比学习创建了强大的表示模型LLM2Comp。

### 背景

文本表示在聚类、检索等下游任务中至关重要，大多数大语言模型本质上是因果模型，针对下一个词预测优化，因此不适合生成整体表示。现有方法大多依赖令牌级预测目标，如掩码下一个词预测(MNTP)。

### 目的

探索上下文压缩作为无监督适配大语言模型的预训练任务的潜力，以生成更有效的文本表示。

### 方法

在压缩预训练过程中，模型学习生成紧凑的记忆令牌，这些令牌替代整个上下文用于下游序列预测。随后通过对比学习进一步改进，创建了表示模型LLM2Comp。

### 主要发现

设计良好的压缩目标可以显著增强基于大语言模型的文本表示；使用压缩目标训练的模型优于使用令牌级预训练任务的模型；LLM2Comp在广泛任务上优于当代基于大语言模型的文本编码器，且更高效，需要显著更少的训练数据。

### 结论

上下文压缩作为预训练任务能有效提升大语言模型的文本表示能力，LLM2Comp模型在性能和效率方面均优于现有方法。

### 翻译

文本表示在聚类、检索和其他下游应用中起着关键作用。随着大语言模型的出现，人们越来越有兴趣利用它们的能力来实现这一目的。然而，大多数大语言模型本质上是因果模型，并针对下一个词预测进行了优化，因此它们在生成整体表示方面并非最佳选择。为了解决这个问题，最近的研究引入了预训练任务来使大语言模型适应文本表示。然而，这些任务大多依赖于令牌级预测目标，例如LLM2Vec中使用的掩码下一个词预测(MNTP)。在这项工作中，我们探索了上下文压缩作为无监督适配大语言模型的预训练任务的未开发潜力。在压缩预训练期间，模型学习生成紧凑的记忆令牌，这些令牌替代整个上下文用于下游序列预测。实验表明，设计良好的压缩目标可以显著增强基于大语言模型的文本表示，优于使用令牌级预训练任务训练的模型。通过对比学习的进一步改进产生了一个强大的表示模型(LLM2Comp)，它在广泛任务上优于当代基于大语言模型的文本编码器，同时更高效，需要显著更少的训练数据。


### 论文摘要

Text representation plays a critical role in tasks like clustering, retrieval, and other downstream applications. With the emergence of large language models (LLMs), there is increasing interest in harnessing their capabilities for this purpose. However, most of the LLMs are inherently causal and optimized for next-token prediction, making them suboptimal for producing holistic representations. To address this, recent studies introduced pretext tasks to adapt LLMs for text representation. Most of these tasks, however, rely on token-level prediction objectives, such as the masked next-token prediction (MNTP) used in LLM2Vec. In this work, we explore the untapped potential of context compression as a pretext task for unsupervised adaptation of LLMs. During compression pre-training, the model learns to generate compact memory tokens, which substitute the whole context for downstream sequence prediction. Experiments demonstrate that a well-designed compression objective can significantly enhance LLM-based text representations, outperforming models trained with token-level pretext tasks. Further improvements through contrastive learning produce a strong representation model (LLM2Comp) that outperforms contemporary LLM-based text encoders on a wide range of tasks while being more sample-efficient, requiring significantly less training data.

---

## 66. OmniLens++: Blind Lens Aberration Correction via Large LensLib Pre-Training and Latent PSF Representation

**论文链接:** [http://arxiv.org/abs/2511.17126v1](http://arxiv.org/abs/2511.17126v1)

**作者:** Qi Jiang, Xiaolong Qian, Yao Gao, Lei Sun, Kailun Yang, Zhonghua Yi, Wenyong Li, Ming-Hsuan Yang, Luc Van Gool, Kaiwei Wang

**发布时间:** 2025-11-21

**备注:** The source code and datasets will be made publicly available at https://github.com/zju-jiangqi/OmniLens2

### GPT解析

### 总结

本文提出了一种名为OmniLens++的新框架，用于解决现有管道在泛化能力方面面临的两个挑战：数据扩展困难和缺乏表征光学退化的先验指导。通过扩展设计规范和引入潜在点扩散函数表示(LPR)，该框架在真实世界镜头和合成LensLib上的实验中显示出最先进的泛化能力，可用于盲像差校正。

### 背景

新兴的基于深度学习的镜头库预训练(LensLib-PT)管道为盲镜头像差校正提供了新途径，通过训练通用神经网络来处理各种未知的光学退化。然而，现有管道的泛化能力受到两个主要挑战的限制：数据扩展困难和缺乏表征光学退化的先验指导。

### 目的

提出OmniLens++框架，解决阻碍现有管道泛化能力的两个挑战：数据扩展困难和缺乏表征光学退化的先验指导，从而提高盲像差校正的性能和泛化能力。

### 方法

为提高数据可扩展性，扩展设计规范以增加镜头源的退化多样性，并通过量化光学退化的空间变化模式和严重程度来采样更均匀的分布；在模型设计方面，提出潜在点扩散函数表示(LPR)，利用直观描述光学退化的点扩散函数(PSF)作为盲模式下的指导；引入VQVAE框架学习LensLib的PSF的潜在特征，通过建模光学退化过程来约束退化先验的学习。

### 主要发现

在真实世界镜头的各种像差和合成LensLib上的实验表明，OmniLens++在盲像差校正方面展现出最先进的泛化能力；AODLibpro被验证为跨越各种像差进行更有效训练的可扩展基础；LPR可以进一步挖掘大规模LensLib的潜力。

### 结论

OmniLens++框架通过解决数据扩展困难和缺乏表征光学退化的先验指导这两个挑战，显著提高了盲像差校正的泛化能力。该框架不仅实现了最先进的性能，还为未来研究提供了可扩展的基础和潜力。

### 翻译

新兴的基于深度学习的镜头库预训练(LensLib-PT)管道通过训练通用神经网络为盲镜头像差校正提供了新途径，显示出处理各种未知光学退化的强大能力。这项工作提出了OmniLens++框架，解决了阻碍现有管道泛化能力的两个挑战：数据扩展困难和表征光学退化的先导指导的缺失。为了提高数据可扩展性，我们扩展了设计规范以增加镜头源的退化多样性，并通过量化光学退化的空间变化模式和严重程度来采样更均匀的分布。在模型设计方面，为了在盲模式下直观描述光学退化的点扩散函数(PSF)作为指导，我们提出了潜在PSF表示(LPR)。引入VQVAE框架学习LensLib的PSF的潜在特征，通过建模光学退化过程来约束退化先验的学习。在真实世界镜头的各种像差和合成LensLib上的实验表明，OmniLens++在盲像差校正方面展现出最先进的泛化能力。除了性能之外，AODLibpro还被验证为跨越各种像差进行更有效训练的可扩展基础，而LPR可以进一步挖掘大规模LensLib的潜力。源代码和数据集将在https://github.com/zju-jiangqi/OmniLens2上公开提供。


### 论文摘要

Emerging deep-learning-based lens library pre-training (LensLib-PT) pipeline offers a new avenue for blind lens aberration correction by training a universal neural network, demonstrating strong capability in handling diverse unknown optical degradations. This work proposes the OmniLens++ framework, which resolves two challenges that hinder the generalization ability of existing pipelines: the difficulty of scaling data and the absence of prior guidance characterizing optical degradation. To improve data scalability, we expand the design specifications to increase the degradation diversity of the lens source, and we sample a more uniform distribution by quantifying the spatial-variation patterns and severity of optical degradation. In terms of model design, to leverage the Point Spread Functions (PSFs), which intuitively describe optical degradation, as guidance in a blind paradigm, we propose the Latent PSF Representation (LPR). The VQVAE framework is introduced to learn latent features of LensLib's PSFs, which is assisted by modeling the optical degradation process to constrain the learning of degradation priors. Experiments on diverse aberrations of real-world lenses and synthetic LensLib show that OmniLens++ exhibits state-of-the-art generalization capacity in blind aberration correction. Beyond performance, the AODLibpro is verified as a scalable foundation for more effective training across diverse aberrations, and LPR can further tap the potential of large-scale LensLib. The source code and datasets will be made publicly available at https://github.com/zju-jiangqi/OmniLens2.

---

## 67. Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation

**论文链接:** [http://arxiv.org/abs/2511.17097v1](http://arxiv.org/abs/2511.17097v1)

**作者:** Shuo Wang, Yucheng Wang, Guoxin Lian, Yongcai Wang, Maiyue Chen, Kaihui Wang, Bo Zhang, Zhizhong Su, Yutian Zhou, Wanting Li, Deying Li, Zhaoxin Fan

**发布时间:** 2025-11-21

### GPT解析

### 总结

这项研究提出了Progress-Think方法，通过语义进展推理改进视觉语言导航任务中的长期连贯性。研究团队开发了一个三阶段框架，在没有昂贵标注的情况下实现了最先进的导航性能。

### 背景

Vision-Language Navigation要求代理在长时程内保持连贯行动，不仅要理解局部视觉上下文，还要理解在多步指令中已经进展了多远。然而，现有的视觉语言行动模型专注于直接行动预测，而早期的方法预测数字成就，两者都忽略了观察和指令序列的单调共进特性。

### 目的

通过语义进展推理，从视觉观察中预测指令风格的进展，从而实现更准确的视觉语言导航，而不需要昂贵的标注数据。

### 方法

提出了一个三阶段框架：1) 自对齐进展预训练，通过视觉历史和指令前缀之间的新型可微分对齐引导推理模块；2) 进展引导策略预训练，将学习到的进展状态注入导航上下文，引导策略采取一致行动；3) 进展-策略共同微调，使用定制的进展感知强化目标共同优化两个模块。

### 主要发现

在R2R-CE和RxR-CE数据集上的实验展示了最先进的成功率和效率，表明语义进展为导航进展提供了更一致的表示。

### 结论

语义进展推理能够有效提升视觉语言导航的性能，特别是在需要长时程连贯性的任务中。所提出的三阶段框架在没有昂贵标注的情况下实现了这一目标，为视觉语言导航领域提供了新的思路。

### 翻译

视觉语言导航要求代理通过不仅理解局部视觉上下文，还理解在多步指令中已经前进了多远，从而在长时程内保持连贯行动。然而，最近的视觉语言行动模型专注于直接行动预测，而早期的方法预测数字成就；两者都忽略了观察和指令序列的单调共进特性。基于这一见解，Progress-Think引入了语义进展推理，从视觉观察中预测指令风格的进展，从而实现更准确的导航。为了在没有昂贵标注的情况下实现这一目标，我们提出了一个三阶段框架。在初始阶段，自对齐进展预训练通过视觉历史和指令前缀之间的新型可微分对齐引导推理模块。然后，进展引导策略预训练将学习到的进展状态注入导航上下文，引导策略采取一致行动。最后，进展-策略共同微调使用定制的进展感知强化目标共同优化两个模块。在R2R-CE和RxR-CE上的实验展示了最先进的成功率和效率，表明语义进展为导航进展提供了更一致的表示。


### 论文摘要

Vision-Language Navigation requires agents to act coherently over long horizons by understanding not only local visual context but also how far they have advanced within a multi-step instruction. However, recent Vision-Language-Action models focus on direct action prediction and earlier progress methods predict numeric achievements; both overlook the monotonic co-progression property of the observation and instruction sequences. Building on this insight, Progress-Think introduces semantic progress reasoning, predicting instruction-style progress from visual observations to enable more accurate navigation. To achieve this without expensive annotations, we propose a three-stage framework. In the initial stage, Self-Aligned Progress Pretraining bootstraps a reasoning module via a novel differentiable alignment between visual history and instruction prefixes. Then, Progress-Guided Policy Pretraining injects learned progress states into the navigation context, guiding the policy toward consistent actions. Finally, Progress-Policy Co-Finetuning jointly optimizes both modules with tailored progress-aware reinforcement objectives. Experiments on R2R-CE and RxR-CE show state-of-the-art success and efficiency, demonstrating that semantic progress yields a more consistent representation of navigation advancement.

---

## 68. Morphological Image Similarity Search on the ALMA Science Archive Query Interface Using Deep Unsupervised Contrastive Representation Learning

**论文链接:** [http://arxiv.org/abs/2511.17061v1](http://arxiv.org/abs/2511.17061v1)

**作者:** Felix Stoehr, Andrea Farago, Stefan Curiban, Alisdair Manning, Jorge Garcia, Pei-Ying Hsieh, Andrew Lipnicky, Adele Plunkett

**发布时间:** 2025-11-21

**DOI:** 10.18727/0722-6691/5396

**备注:** 3 pages, 2 figures

### GPT解析

### 总结

研究团队在ALMA科学档案中实现了形态学图像相似性搜索功能，使用自监督对比学习方法和深度神经网络，使天文学家能够基于观测内容而非仅元数据进行数据检索和探索。

### 背景

随着天文数据的指数级增长，在海量数据中找到特定信息变得日益困难。科学档案的下一个前沿是不仅能够基于观测元数据进行搜索，还能基于观测内容本身进行搜索。

### 目的

在ALMA科学档案(ASA)中实现形态学图像相似性搜索功能，帮助天文学家更有效地找到相关的天文观测数据。

### 方法

使用自监督对比学习，采用仿射变换无关的表示学习方法，通过深度神经网络学习源形态学的表示。在ASA网页界面上，给定一个图像后，系统会显示形态学上最相似的图像摘要视图，用户选择额外图像后，显示会立即更新，展示与所选图像组合最相似的图像。

### 主要发现

这是第一次在天文科学档案中提供图像相似性搜索功能，为天文学家提供了基于内容的数据探索新途径。

### 结论

这种形态学图像相似性搜索方法能够根据天文学家的科学需求不断完善搜索结果，提高了天文数据检索的相关性和效率。

### 翻译

随着天文数据的指数级增长，在海量数据中找到特定信息变得越来越困难。科学档案的下一个前沿是不仅能够基于观测元数据进行搜索，还能基于观测内容本身进行搜索。作为迈向这一方向的一步，我们在ALMA科学档案(ASA)中实现了形态学图像相似性搜索。为此，我们使用深度神经网络进行自监督对比仿射变换无关的源形态学表示学习。对于ASA网页界面上的给定图像，天文学家会看到形态学上最相似图像的摘要视图。每当天文学家从该视图中选择额外图像时，显示会立即更新，以显示与所选图像组合最相似的图像。每次选择都根据天文学家的科学需求来完善相似性显示。这是第一次在天文科学档案中提供图像相似性搜索。


### 论文摘要

With the exponential growth of astronomical data over time, finding the needles in the haystack is becoming increasingly difficult. The next frontier for science archives is to enable searches not only on observational metadata, but also on the content of the observations themselves. As a step in this direction, we have implemented morphological image similarity search into the ALMA Science Archive (ASA). To achieve this we use self-supervised contrastive affine-transformation-independent representation learning of source morphologies with a deep neural network. For a given image on the ASA web interface, astronomers are presented with a summary view of the morphologically most similar images. Each time an astronomer selects an additional image from that view, the display is instantly updated to show the images most similar to the combination of the selected images. Each selection thus refines the similarity display according to the scientific needs of the astronomer. This is the first time image similarity search has been offered in an astronomical science archive.

---

## 69. CLLMRec: LLM-powered Cognitive-Aware Concept Recommendation via Semantic Alignment and Prerequisite Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2511.17041v1](http://arxiv.org/abs/2511.17041v1)

**作者:** Xiangrui Xiong, Yichuan Lu, Zifei Pan, Chang Sun

**发布时间:** 2025-11-21

### GPT解析

### 总结

CLLMRec是一种新型框架，利用大语言模型解决MOOC中概念推荐问题，不依赖高质量结构化知识图谱，通过语义对齐和先验知识蒸馏两个技术支柱，结合深度知识追踪，实现认知感知和个性化推荐。

### 背景

大规模开放在线课程（MOOCs）的增长给个性化学习带来了挑战，概念推荐在个性化学习中至关重要。现有方法通常依赖异构信息网络或知识图谱捕捉概念关系，并结合知识追踪模型评估学习者认知状态，但这些方法面临严重局限性，因为它们依赖于高质量的结构化知识图谱，而现实教育场景中这类图谱往往稀缺。

### 目的

解决对高质量结构化知识图谱依赖的根本挑战，实现无需明确结构先验的认知感知和个性化概念推荐。

### 方法

提出了CLLMRec框架，利用大语言模型通过两个协同技术支柱：1) 语义对齐：通过编码学习者和概念的非结构化文本描述构建统一的表示空间；2) 先验知识蒸馏：采用教师-学生架构，大型教师LLM从其内部化世界知识中提取概念先验关系，并将其蒸馏为软标签以训练高效的学生排序器。框架还包含细粒度排序机制，通过深度知识追踪明确建模学习者的实时认知状态。

### 主要发现

在两个真实世界MOOC数据集上的大量实验表明，CLLMRec在多个评估指标上显著优于现有的基线方法。

### 结论

CLLMRec验证了在生成真正认知感知和个性化概念推荐方面的有效性，无需依赖明确的结构先验。

### 翻译

大规模开放在线课程（MOOCs）的增长给个性化学习带来了重大挑战，其中概念推荐至关重要。现有方法通常依赖异构信息网络或知识图谱来捕捉概念关系，并结合知识追踪模型来评估学习者的认知状态。然而，由于这些方法依赖于高质量的结构化知识图谱，而现实教育场景中这类图谱往往稀缺，因此它们面临着显著的局限性。为了解决这一根本挑战，本文提出了CLLMRec，一种利用大语言模型通过两个协同技术支柱的新框架：语义对齐和先验知识蒸馏。语义对齐组件通过编码学习者和概念的非结构化文本描述来构建统一的表示空间。先验知识蒸馏范式采用教师-学生架构，其中大型教师LLM（作为先验知识感知组件实现）从其内部化的世界知识中提取概念先验关系，并将其蒸馏为软标签以训练高效的学生排序器。基于这些基础，我们的框架纳入了一个细粒度排序机制，通过深度知识追踪明确建模学习者的实时认知状态，确保推荐在结构上合理且在认知上适当。在两个真实世界MOOC数据集上的大量实验表明，CLLMRec在多个评估指标上显著优于现有的基线方法，验证了其在生成真正认知感知和个性化概念推荐方面的有效性，无需依赖明确的结构先验。


### 论文摘要

The growth of Massive Open Online Courses (MOOCs) presents significant challenges for personalized learning, where concept recommendation is crucial. Existing approaches typically rely on heterogeneous information networks or knowledge graphs to capture conceptual relationships, combined with knowledge tracing models to assess learners' cognitive states. However, these methods face significant limitations due to their dependence on high-quality structured knowledge graphs, which are often scarce in real-world educational scenarios. To address this fundamental challenge, this paper proposes CLLMRec, a novel framework that leverages Large Language Models through two synergistic technical pillars: Semantic Alignment and Prerequisite Knowledge Distillation. The Semantic Alignment component constructs a unified representation space by encoding unstructured textual descriptions of learners and concepts. The Prerequisite Knowledge Distillation paradigm employs a teacher-student architecture, where a large teacher LLM (implemented as the Prior Knowledge Aware Component) extracts conceptual prerequisite relationships from its internalized world knowledge and distills them into soft labels to train an efficient student ranker. Building upon these foundations, our framework incorporates a fine-ranking mechanism that explicitly models learners' real-time cognitive states through deep knowledge tracing, ensuring recommendations are both structurally sound and cognitively appropriate. Extensive experiments on two real-world MOOC datasets demonstrate that CLLMRec significantly outperforms existing baseline methods across multiple evaluation metrics, validating its effectiveness in generating truly cognitive-aware and personalized concept recommendations without relying on explicit structural priors.

---

## 70. Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering

**论文链接:** [http://arxiv.org/abs/2511.17008v1](http://arxiv.org/abs/2511.17008v1)

**作者:** Zexi Tan, Xiaopeng Luo, Yunlin Liu, Yiqun Zhang

**发布时间:** 2025-11-21

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

本文提出了一种名为EMTC的进化掩码多元时间序列聚类方法，通过感知变量重要性的掩码(IVM)和多内源视图(MEV)表示学习模块，有效解决了时间序列数据中的冗余问题，提高了聚类性能。

### 背景

多元时间序列聚类能发现时间数据样本的内在分组模式，但时间序列中的冗余(如稳态机器运行记录和零输出期)降低了表示学习中判别性时间戳的关注度，导致聚类性能瓶颈。现有掩码策略多为独立预处理步骤，无法动态适应聚类关键时间戳的重要性。

### 目的

开发一种能够自适应学习更具判别性表示的MTS聚类方法，解决现有掩码策略的局限性，提高聚类性能。

### 方法

提出EMTC方法，包含两个核心模块：1)感知变量重要性的掩码(IVM)，自适应指导模型学习更具判别性的聚类表示；2)多内源视图(MEV)表示学习模块，通过基于MEV的重建促进多视角互补防止过早收敛，以及聚类引导的对比学习促进表示和聚类的联合优化。

### 主要发现

在15个真实基准数据集上的实验表明，EMTC与8种最先进方法相比表现出优越性，比最强基线方法平均提高了4.85%的聚类性能。

### 结论

EMTC方法通过自适应掩码和多视图表示学习有效解决了MTS聚类中的冗余问题，能够更好地关注判别性时间戳，显著提高聚类性能。

### 翻译

多元时间序列(MTS)聚类能够发现时间数据样本的内在分组模式。虽然时间序列提供丰富的判别信息，但也包含大量冗余，如稳态机器运行记录和太阳能发电的零输出期。这种冗余降低了表示学习中判别性时间戳的关注度，从而导致MTS聚类性能瓶颈。掩码已被广泛采用以增强MTS表示，其中时间重建任务旨在从MTS中捕获关键信息。然而，大多数现有掩码策略似乎是独立的预处理步骤，与学习过程隔离，这阻碍了对聚类关键时间戳重要性的动态适应。因此，本文提出了进化掩码MTS聚类(EMTC)方法，其模型架构由感知变量重要性的掩码(IVM)和多内源视图(MEV)表示学习模块组成。IVM自适应地指导模型学习更具判别性的聚类表示，而基于MEV的重建和对比学习途径增强了泛化能力。也就是说，MEV重建促进了多视角互补，防止掩码过早收敛，聚类引导的对比学习促进了表示和聚类的联合优化。在15个真实基准数据集上的广泛实验证明了EMTC与八种最先进方法相比的优越性，EMTC比最强基线方法平均提高了4.85%。


### 论文摘要

Multivariate Time-Series (MTS) clustering discovers intrinsic grouping patterns of temporal data samples. Although time-series provide rich discriminative information, they also contain substantial redundancy, such as steady-state machine operation records and zero-output periods of solar power generation. Such redundancy diminishes the attention given to discriminative timestamps in representation learning, thus leading to performance bottlenecks in MTS clustering. Masking has been widely adopted to enhance the MTS representation, where temporal reconstruction tasks are designed to capture critical information from MTS. However, most existing masking strategies appear to be standalone preprocessing steps, isolated from the learning process, which hinders dynamic adaptation to the importance of clustering-critical timestamps. Accordingly, this paper proposes the Evolving-masked MTS Clustering (EMTC) method, with its model architecture composed of Importance-aware Variate-wise Masking (IVM) and Multi-Endogenous Views (MEV) representation learning modules. IVM adaptively guides the model in learning more discriminative representations for clustering, while the MEV-based reconstruction and contrastive learning pathways enhance the generalization. That is, the MEV reconstruction facilitates multi-perspective complementary to prevent the masking from premature convergence, and the clustering-guided contrastive learning facilitates the joint optimization of representation and clustering. Extensive experiments on 15 real benchmark datasets demonstrate the superiority of EMTC in comparison with eight SOTA methods, where the EMTC achieves an average improvement of 4.85% over the strongest baselines.

---

## 71. Generative MIMO Beam Map Construction for Location Recovery and Beam Tracking

**论文链接:** [http://arxiv.org/abs/2511.17007v1](http://arxiv.org/abs/2511.17007v1)

**作者:** Wangqian Chen, Junting Chen, Shuguang Cui

**发布时间:** 2025-11-21

### GPT解析

### 总结

该论文提出了一种生成框架，能够从稀疏信道状态信息(CSI)测量序列中直接恢复位置标签，无需显式的位置标签即可构建无线电地图，从而提高定位精度和通信容量。

### 背景

机器学习在无线通信系统的数据驱动信道建模和资源优化方面取得了很大进展，但现有的大多数基于ML的方法依赖于带有位置信息的大型、准确标记的数据集，这些数据集通常难以获取且成本高昂。

### 目的

提出一个生成框架，直接从稀疏信道状态信息(CSI)测量序列中恢复位置标签，无需显式的位置标签即可构建无线电地图。

### 方法

学习紧凑的低维无线电地图嵌入而非直接存储原始CSI；设计双尺度特征提取方案增强特征表示；开发混合循环卷积编码器学习移动模式；在潜在空间中嵌入可学习的无线电图捕获位置信息；使用基于扩散的生成解码器重建完整CSI。

### 主要发现

与基于模型的卡尔曼滤波方法相比，所提出的模型可以将定位精度提高30%以上，在非视距(NLOS)场景中实现20%的容量增益。

### 结论

该生成框架能够有效地从稀疏CSI中恢复位置信息，提高定位精度和通信容量，解决了无线通信系统中标记数据获取困难的问题。

### 翻译

机器学习(ML)极大地推动了无线通信系统中数据驱动的信道建模和资源优化。然而，大多数现有的基于ML的方法依赖于带有位置信息的大型、准确标记的数据集，这些数据集通常难以获取且成本高昂。本文提出了一种生成框架，可以直接从稀疏信道状态信息(CSI)测量序列中恢复位置标签，无需显式的位置标签即可构建无线电地图。我们不直接存储原始CSI，而是学习一个紧凑的低维无线电地图嵌入，并利用生成模型重建高维CSI。具体来说，为解决稀疏CSI的不确定性问题，我们设计了一种双尺度特征提取方案，通过联合利用角度空间和相关样本的关联来增强特征表示。我们开发了一种混合循环卷积编码器来学习移动模式，该编码器在循环神经网络(RNN)中结合了截断策略和多尺度卷积，以确保特征对短期波动的鲁棒性。与潜在空间中的传统高斯先验不同，我们嵌入了一个可学习的无线电图，通过从CSI测量中编码高级位置特征来捕获位置信息。最后，基于扩散的生成解码器根据无线电图中的位置特征以高保真度重建完整的CSI。数值实验表明，与基于模型的卡尔曼滤波方法相比，所提出的模型可以将定位精度提高30%以上，并在非视距(NLOS)场景中实现20%的容量增益。


### 论文摘要

Machine learning (ML) has greatly advanced data-driven channel modeling and resource optimization in wireless communication systems. However, most existing ML-based methods rely on large, accurately labeled datasets with location information, which are often difficult and costly to obtain. This paper proposes a generative framework to recover location labels directly from sequences of sparse channel state information (CSI) measurements, without explicit location labels for radio map construction. Instead of directly storing raw CSI, we learn a compact low-dimensional radio map embedding and leverage a generative model to reconstruct the high-dimensional CSI. Specifically, to address the uncertainty of sparse CSI, a dual-scale feature extraction scheme is designed to enhance feature representation by jointly exploiting correlations from angular space and across neighboring samples. We develop a hybrid recurrent-convolutional encoder to learn mobility patterns, which combines a truncation strategy and multi-scale convolutions in the recurrent neural network (RNN) to ensure feature robustness against short-term fluctuations. Unlike conventional Gaussian priors in latent space, we embed a learnable radio map to capture the location information by encoding high-level positional features from CSI measurements. Finally, a diffusion-based generative decoder reconstructs the full CSI with high fidelity by conditioning on the positional features in the radio map. Numerical experiments demonstrate that the proposed model can improve localization accuracy by over 30% and achieve a 20% capacity gain in non-line-of-sight (NLOS) scenarios compared with model-based Kalman filter approaches.

---

## 72. EvDiff: High Quality Video with an Event Camera

**论文链接:** [http://arxiv.org/abs/2511.17492v1](http://arxiv.org/abs/2511.17492v1)

**作者:** Weilun Li, Lei Sun, Ruixi Gao, Qi Jiang, Yuqin Ma, Kaiwei Wang, Ming-Hsuan Yang, Luc Van Gool, Danda Pani Paudel

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了EvDiff，一种基于事件的扩散模型，通过替代训练框架和单步前向扩散策略，从单色事件流生成高质量彩色视频，无需配对事件-图像数据集。

### 背景

神经形态相机作为事件相机，异步记录亮度变化，具有高时间分辨率和高动态范围优势，但从事件重建强度图像是一个高度不适定的问题。

### 目的

解决早期端到端回归方法在感知质量、模型容量扩展和训练数据依赖方面的局限性，实现从单色事件流生成高质量彩色视频。

### 方法

提出EvDiff事件扩散模型，采用替代训练框架，设计仅执行单步前向扩散的事件扩散模型并配备时间一致的EvEncoder，消除对配对数据集的依赖。

### 主要发现

EvDiff能够仅从单色事件流生成高质量彩色视频，在真实数据集实验中，在保真度和真实度之间取得平衡，在像素级和感知指标上优于现有方法。

### 结论

EvDiff通过创新的扩散模型架构和训练策略，成功解决了事件到视频转换中的关键挑战，为神经形态视觉处理提供了新思路。

### 翻译

作为神经形态传感器，事件相机异步记录亮度变化，以稀疏事件流的形式，具有高时间分辨率和高动态范围的优势。由于绝对亮度的内在模糊性，从事件重建强度图像是一个高度不适定的任务。早期方法通常遵循端到端回归范式，以确定性方式直接将事件映射到强度帧。虽然在一定程度上有效，但这些方法通常产生感知质量较差的结果，并且在模型容量和训练数据扩展方面存在困难。在本工作中，我们提出了EvDiff，一种基于事件的扩散模型，遵循替代训练框架来生成高质量视频。为了减少高帧率视频生成的计算成本，我们设计了一种仅执行单步前向扩散的事件扩散模型，配备了时间一致的EvEncoder。此外，我们新颖的替代训练框架消除了对配对事件-图像数据集的依赖，使模型能够利用大规模图像数据集实现更高容量。所提出的EvDiff能够仅从单色事件流生成高质量彩色视频。在真实数据集上的实验表明，我们的方法在保真度和真实感之间取得了良好的平衡，在像素级和感知指标上都优于现有方法。


### 论文摘要

As neuromorphic sensors, event cameras asynchronously record changes in brightness as streams of sparse events with the advantages of high temporal resolution and high dynamic range. Reconstructing intensity images from events is a highly ill-posed task due to the inherent ambiguity of absolute brightness. Early methods generally follow an end-to-end regression paradigm, directly mapping events to intensity frames in a deterministic manner. While effective to some extent, these approaches often yield perceptually inferior results and struggle to scale up in model capacity and training data. In this work, we propose EvDiff, an event-based diffusion model that follows a surrogate training framework to produce high-quality videos. To reduce the heavy computational cost of high-frame-rate video generation, we design an event-based diffusion model that performs only a single forward diffusion step, equipped with a temporally consistent EvEncoder. Furthermore, our novel Surrogate Training Framework eliminates the dependence on paired event-image datasets, allowing the model to leverage large-scale image datasets for higher capacity. The proposed EvDiff is capable of generating high-quality colorful videos solely from monochromatic event streams. Experiments on real-world datasets demonstrate that our method strikes a sweet spot between fidelity and realism, outperforming existing approaches on both pixel-level and perceptual metrics.

---

## 73. Video-R4: Reinforcing Text-Rich Video Reasoning with Visual Rumination

**论文链接:** [http://arxiv.org/abs/2511.17490v1](http://arxiv.org/abs/2511.17490v1)

**作者:** Yolo Yunlong Tang, Daiki Shimada, Hang Hua, Chao Huang, Jing Bi, Rogerio Feris, Chenliang Xu

**发布时间:** 2025-11-21

### GPT解析

### 总结

这篇论文提出了Video-R4，一种通过视觉沉思增强富含文本视频推理的视频推理LMM，通过迭代选择帧、放大信息丰富区域、重新编码像素和更新推理状态，提高了视频问答的准确性。

### 背景

理解富含文本的视频需要反复检查小型、短暂的文本线索，但大多数视频问答模型依赖于对固定帧的单次感知，导致在细粒度证据上出现幻觉和失败。

### 目的

开发一种能够像人类一样暂停、放大和重新阅读关键区域的视频推理模型，以提高富含文本视频问答的准确性和细粒度理解能力。

### 方法

提出Video-R4执行视觉沉思的视频推理LMM；构建两个数据集（Video-R4-CoT-17k用于监督学习和Video-R4-RL-30k用于强化学习）；提出多阶段沉思学习框架，通过SFT和基于GRPO的RL逐步微调7B LMM。

### 主要发现

Video-R4-7B在M4-ViteVQA上取得了最先进的结果，并能推广到多页文档问答、幻灯片问答和通用视频问答任务。

### 结论

迭代沉思是一种有效的像素基础多模态推理范式，能够显著提高富含文本视频的理解和问答能力。

### 翻译

理解富含文本的视频需要阅读小型、短暂的文本线索，这些线索通常需要反复检查。然而，大多数视频问答模型依赖于对固定帧的单次感知，导致在细粒度证据上出现幻觉和失败。受人类如何暂停、放大和重新阅读关键区域的启发，我们引入了Video-R4（通过视觉沉思强化富含文本视频推理），这是一种执行视觉沉思的视频推理LMM：迭代选择帧、放大信息丰富区域、重新编码检索到的像素，并更新其推理状态。我们构建了两个具有可执行沉思轨迹的数据集：用于监督学习的Video-R4-CoT-17k和用于强化学习的Video-R4-RL-30k。我们提出了一种多阶段沉思学习框架，通过SFT和基于GRPO的RL逐步微调一个7B LMM，以学习原子和混合视觉操作。Video-R4-7B在M4-ViteVQA上取得了最先进的结果，并进一步推广到多页文档问答、幻灯片问答和通用视频问答，证明了迭代沉思是一种有效的像素基础多模态推理范式。


### 论文摘要

Understanding text-rich videos requires reading small, transient textual cues that often demand repeated inspection. Yet most video QA models rely on single-pass perception over fixed frames, leading to hallucinations and failures on fine-grained evidence. Inspired by how humans pause, zoom, and re-read critical regions, we introduce Video-R4 (Reinforcing Text-Rich Video Reasoning with Visual Rumination), a video reasoning LMM that performs visual rumination: iteratively selecting frames, zooming into informative regions, re-encoding retrieved pixels, and updating its reasoning state. We construct two datasets with executable rumination trajectories: Video-R4-CoT-17k for supervised practice and Video-R4-RL-30k for reinforcement learning. We propose a multi-stage rumination learning framework that progressively finetunes a 7B LMM to learn atomic and mixing visual operations via SFT and GRPO-based RL. Video-R4-7B achieves state-of-the-art results on M4-ViteVQA and further generalizes to multi-page document QA, slides QA, and generic video QA, demonstrating that iterative rumination is an effective paradigm for pixel-grounded multimodal reasoning.

---

## 74. Structure-Function Coherent Coarsening for Cross-Resolution Ecohydrological Modeling

**论文链接:** [http://arxiv.org/abs/2511.17482v1](http://arxiv.org/abs/2511.17482v1)

**作者:** Long Jiang, Yang Yang, Morgan Thornwell, Tiantian Yang, Hoshin Vijai Gupta

**发布时间:** 2025-11-21

**备注:** 24 pages, 8 figures

### GPT解析

### 总结

本研究开发了一个结构与功能协调粗化(SFCC)框架，用于在水文生态模型粗化过程中保持水文连通性和功能异质性，有效解决了模型应用中计算成本高和跨尺度建模结构不一致的问题。

### 背景

水文生态模型被越来越多地应用于多种场景，但其应用受到精细分辨率模拟计算成本高和跨尺度建模结构不一致的限制。

### 目的

开发一个结构与功能协调粗化框架，在模型输入粗化过程中同时保持水文连通性和功能异质性。

### 方法

将VELMA模型应用于美国萨利什海盆地的24个子流域，研究三种类型输入的粗化方法：(i)使用水文感知方法粗化的DEM；(ii)使用功能保持方法粗化的土地利用和土壤类型数据；(iii)使用水文、土地覆盖和土壤感知策略粗化的初始条件。

### 主要发现

水文感知方法有效保持流域形态，产生更一致的径流和硝酸盐预测；功能保持方法减轻了多数聚类的主导类别偏差；长期模拟显示水文变量快速平衡，生物地球化学过程调整较缓慢，但两者偏差随时间减小并趋近稳态；结构一致性和功能保持通过时空反馈共同维持动态稳定性。

### 结论

与现有工作相比，SFCC框架直接在数据输入层面操作，能够更协调地整合多源数据集，并最大限度地保留高分辨率信息。

### 翻译

水文生态模型正被越来越多地应用于多种场景，但其应用仍然受到精细分辨率模拟计算成本高和跨尺度建模结构不一致的限制。本研究开发了一个结构与功能协调粗化(SFCC)框架，在模型输入粗化过程中保持水文连通性和功能异质性。我们将VELMA模型应用于美国萨利什海盆地的24个子流域，并检查三种类型的输入：(i)使用保持排水拓扑结构的水文感知方法粗化的DEM；(ii)使用保留小型但过程主导类别的功能保持方法(自动权重和重新分配)粗化的土地利用和土壤类型数据；(iii)使用水文、土地覆盖和土壤感知策略粗化的初始条件，以增强时间稳定性。结果显示，水文感知方法有效保持流域形态，并在跨尺度上比基于平均的粗化方法产生更一致的径流和硝酸盐预测。对于分类输入，功能保持方法减轻了多数聚类的主导类别偏差，特别是在小型高影响斑块驱动氮输出的流域中。长期模拟进一步表明，尽管水文变量快速平衡而生物地球化学过程调整较缓慢，但两者的偏差随时间减小并趋近稳态。这些表明结构一致性和功能保持通过时空反馈共同维持动态稳定性。与现有工作相比，提出的SFCC框架直接在数据输入层面操作，能够更协调地整合多源数据集，并最大限度地保留高分辨率信息。


### 论文摘要

Ecohydrological models are increasingly applied across multiple scenarios, yet their application remains constrained by high computational costs of fine-resolution simulations and structural inconsistencies in cross-scale modeling. This study develops a Structure-Function Coherent Coarsening (SFCC) framework that preserves both hydrological connectivity and functional heterogeneity during model input coarsening. We apply the VELMA model to 24 subbasins in the Salish Sea Basin, U.S. and examine three types of inputs: (i) DEM coarsened with a Hydro-aware approach that preserves drainage topology; (ii) land-use and soil-type datasets coarsened with function-preserving methods (Auto-weight and Auto-reassign) that retain small but process-dominant classes; and (iii) initial conditions coarsened with hydrology-, land-cover-, and soil-aware strategies to enhance temporal stability. Results show that the Hydro-aware method effectively preserves watershed morphology and yields more consistent runoff and nitrate predictions than mean-based coarsening across scales. For categorical inputs, the function-preserving methods alleviate the dominant-class bias of majority aggregation, particularly in basins where small high-impact patches drive nitrogen export. Long-term simulations further show that although hydrological variables equilibrate rapidly and biogeochemical processes adjust more gradually, deviations in both decrease over time and converge toward a steady state. These demonstrate that structural consistency and functional preservation together maintain dynamic stability through spatiotemporal feedback. Compared with existing work, the proposed SFCC framework operates directly at the data-input level, enabling more coherent integration of multi-source datasets and maximizing the retention of high-resolution information.

---

## 75. SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation

**论文链接:** [http://arxiv.org/abs/2511.17432v1](http://arxiv.org/abs/2511.17432v1)

**作者:** Shrikant Kendre, Austin Xu, Honglu Zhou, Michael Ryoo, Shafiq Joty, Juan Carlos Niebles

**发布时间:** 2025-11-21

**备注:** 23 pages, 6 tables, 9 figures

### GPT解析

### 总结

SMILE是一种创新的评估指标，结合了句子级语义理解、关键词级语义理解和简单关键词匹配，平衡了词汇精确性和语义相关性，提供了更全面的评估方法。

### 背景

传统评估指标如ROUGE、METEOR和精确匹配(EM)主要依赖n-gram词汇相似性，缺乏深层语义理解；BERTScore和MoverScore虽利用上下文嵌入但仍缺乏灵活性且忽略词汇相似性；基于大型语言模型的评估器存在成本高、偏见、不一致和幻觉等问题。

### 目的

解决现有评估方法的局限性，提出一种结合词汇精确性和语义理解的综合评估指标。

### 方法

SMILE（结合词汇精确性的语义指标）结合了句子级语义理解、关键词级语义理解和简单关键词匹配，形成一种复合评估方法。

### 主要发现

在文本、图像和视频问答任务上的广泛基准测试表明，SMILE与人类判断高度相关，且计算轻量级，有效弥合了词汇评估和语义评估之间的差距。

### 结论

SMILE成功结合了词汇精确性和语义理解，为问答任务提供了一种更全面、更高效的评估解决方案。

### 翻译

传统的文本和视觉问答评估指标，如ROUGE、METEOR和精确匹配(EM)，严重依赖基于n-gram的词汇相似性，通常缺乏准确评估所需的更深层次的语义理解。虽然BERTScore和MoverScore等措施利用上下文嵌入来解决这一局限性，但它们在平衡句子级和关键词级语义方面缺乏灵活性，并忽略了仍然重要的词汇相似性。基于大型语言模型(LLM)的评估器虽然功能强大，但存在高成本、偏见、不一致和幻觉等缺点。为了解决这些问题，我们引入了SMILE：结合词汇精确性的语义指标，这是一种新方法，结合了句子级语义理解、关键词级语义理解和简单的关键词匹配。这种复合方法平衡了词汇精确性和语义相关性，提供了全面的评估。在文本、图像和视频问答任务上的广泛基准测试表明，SMILE与人类判断高度相关，并且计算轻量级，弥合了词汇评估和语义评估之间的差距。


### 论文摘要

Traditional evaluation metrics for textual and visual question answering, like ROUGE, METEOR, and Exact Match (EM), focus heavily on n-gram based lexical similarity, often missing the deeper semantic understanding needed for accurate assessment. While measures like BERTScore and MoverScore leverage contextual embeddings to address this limitation, they lack flexibility in balancing sentence-level and keyword-level semantics and ignore lexical similarity, which remains important. Large Language Model (LLM) based evaluators, though powerful, come with drawbacks like high costs, bias, inconsistency, and hallucinations. To address these issues, we introduce SMILE: Semantic Metric Integrating Lexical Exactness, a novel approach that combines sentence-level semantic understanding with keyword-level semantic understanding and easy keyword matching. This composite method balances lexical precision and semantic relevance, offering a comprehensive evaluation. Extensive benchmarks across text, image, and video QA tasks show SMILE is highly correlated with human judgments and computationally lightweight, bridging the gap between lexical and semantic evaluation.

---

## 76. DoS Dos and Don'ts

**论文链接:** [http://arxiv.org/abs/2511.17360v1](http://arxiv.org/abs/2511.17360v1)

**作者:** Lucas Warwaruk, Konstantinos Zinelis, Randy H. Ewoldt, Christopher W. Macosko, Gareth H. McKinley

**发布时间:** 2025-11-21

**备注:** 24 pages, 7 figures

### GPT解析

### 总结

本研究定义了DoS流变法的操作限制和模型界限，证明了可测量低至0.1毫秒的拉伸松弛时间，并提出了细丝捕获率作为新的评估指标。

### 背景

Dripping-onto-Substrate (DoS) rheometry是一种用于测量低粘度液体拉伸流变学的成熟方法，但缺乏关于其能力和局限性的明确指导原则。

### 目的

明确DoS流变法的操作限制，提供基于模型计算液体粘度和拉伸松弛时间的界限，并确定可测量的松弛时间下限。

### 方法

使用聚乙烯氧化物(PEO)和聚丙烯酰胺(PAM)的稀溶液进行实验，研究不同喷嘴半径和邦德数(Bo)下的流变特性，并定义了细丝捕获率作为新的评估指标。

### 主要发现

1) 可测量低至0.1毫秒的拉伸松弛时间，需满足固有Deborah数De ≥ O(0.1)和仪器约束条件；2) 细丝捕获率可作为评估数据可用性的指标；3) 在0.2 < Bo < 0.7范围内，同一流体的拉伸松弛时间变化小于±16%；4) Bo > 0.5时，低粘度流体表现出阻尼重力振荡，影响早期动力学。

### 结论

研究结果为可靠的DoS流变学提供了定量路线图，并确认了其在测量弱弹性流体中亚毫秒松弛时间的适用性。

### 翻译

滴落到基底(Dripping-onto-Substrate, DoS)流变学是一种测量低粘度液体拉伸流变学的成熟方法。然而，关于该技术能力和局限性的明确指导原则仍然缺乏。在本工作中，我们定义了直接通过观察细丝变薄速率来测量瞬态拉伸粘度的操作限制，以及基于模型计算液体粘度η和拉伸松弛时间τ_E的界限。使用聚乙烯氧化物(PEO)和聚丙烯酰胺(PAM)的稀溶液来探测可测量的τ_E的下限，表明可以解析低至0.1毫秒的值，前提是(a)固有Deborah数(基于松弛时间与瑞利断裂时间尺度的比率)De ≥ O(0.1)，以及(b)与空间和时间分辨率相关的仪器约束得到满足。这种仪器约束通过我们定义的新指标'细丝捕获率'进行量化，这是一个'品质因数'(以赫兹为单位)，可用于量化可用于提取τ_E的弹毛细管区域内的数据点数量。我们还研究了其他实验参数的敏感性，包括喷嘴半径和邦德数(Bo)的变化。在测试范围内(0.2 < Bo < 0.7)，同一流体的拉伸松弛时间变化小于±16%；然而，在Bo > 0.5时，低粘度流体表现出阻尼重力振荡，影响早期动力学。总的来说，这些结果为可靠的DoS流变学提供了定量路线图，并确认了其在测量弱弹性流体中亚毫秒松弛时间的用途。


### 论文摘要

Dripping-onto-Substrate (DoS) rheometry is a well-established method for measuring the extensional rheology of low-viscosity liquids. However, clear guidelines on the capabilities and limitations of the technique are lacking. In the present work, we define operational limits for measuring a transient extensional viscosity directly from observation of the rate of filament thinning, as well as model-based bounds on calculating a viscosity $η$ and extensional relaxation time $τ_E$ of a liquid using DoS. Dilute solutions of polyethylene oxide (PEO) and polyacrylamide (PAM) are used to probe the lower limit of measurable $τ_E$, demonstrating that values as low as 0.1 ms can be resolved, provided (a) the intrinsic Deborah number (based on the ratio of the relaxation time and the Rayleigh breakup time scale) is $De \geq \mathcal{O}(0.1)$ and (b) an instrumental constraint related to spatial and temporal resolution is satisfied. This instrumental constraint is quantified through a new metric we define as the \textit{filament capture rate}, a ``figure of merit'' (expressed in Hz) that can be used to quantify the number of data points within the elasto-capillary regime that are available for extraction of $τ_E$. We also investigate the sensitivity to other experimental parameters including variations in nozzle radius and Bond number ($Bo$). Across the tested range ($0.2 < Bo < 0.7$), extensional relaxation times for the same fluid vary by less than $\pm16$ \%; however, experiments with low viscosity fluids at $Bo > 0.5$ exhibit damped gravitational oscillations that affect early-time dynamics. Collectively, these results provide a quantitative roadmap for reliable DoS rheometry and affirm its use for measuring sub-millisecond relaxation times in weakly elastic fluids.

---

## 77. Loomis Painter: Reconstructing the Painting Process

**论文链接:** [http://arxiv.org/abs/2511.17344v1](http://arxiv.org/abs/2511.17344v1)

**作者:** Markus Pobitzer, Chang Liu, Chenyi Zhuang, Teng Long, Bin Ren, Nicu Sebe

**发布时间:** 2025-11-21

### GPT解析

### 总结

这篇论文提出了一个统一框架，用于多媒体系画过程生成，具有语义驱动的风格控制机制，能够实现跨媒体的纹理演化和过程转移，并确保生成过程与人类创作流程对齐。

### 背景

分步绘画教程对于学习艺术技巧至关重要，但现有视频资源缺乏互动性和个性化。现有生成模型在跨媒体泛化方面存在困难，常表现出时间或结构不一致性，妨碍了对人类创作流程的忠实再现。

### 目的

解决现有生成模型在跨媒体泛化方面的问题，创建能够实现跨媒体一致性和过程转移的框架，确保生成的绘画过程与人类创作流程对齐。

### 方法

提出统一的多媒体系画过程生成框架，使用语义驱动的风格控制机制，将多种媒体嵌入到扩散模型的条件空间中，使用跨媒体风格增强，采用反向绘画训练策略，构建大规模真实绘画过程数据集，评估跨媒体一致性、时间连贯性和最终图像保真度，使用LPIPS、DINO和CLIP指标进行评估，引入感知距离曲线定量建模创作序列。

### 主要发现

该框架实现了跨媒体的纹理演化和过程转移，确保了生成过程与人类创作流程的平滑对齐，在多个评估指标上取得了强结果，感知距离曲线能够有效建模艺术创作过程。

### 结论

该框架解决了现有生成模型在跨媒体泛化和一致性方面的问题，通过语义驱动的风格控制和跨媒体风格增强实现了跨媒体的纹理演化和过程转移，反向绘画训练策略确保了生成过程与人类创作流程的对齐，感知距离曲线能够有效建模艺术创作过程。

### 翻译

分步绘画教程对于学习艺术技巧至关重要，但现有的视频资源（如YouTube）缺乏互动性和个性化。虽然最近的生成模型在艺术图像合成方面取得了进展，但它们在跨媒体泛化方面存在困难，并且常常表现出时间或结构不一致性，妨碍了对人类创作流程的忠实再现。为了解决这个问题，我们提出了一个统一的多媒体系画过程生成框架，该框架具有语义驱动的风格控制机制，将多种媒体嵌入到扩散模型的条件空间中，并使用跨媒体风格增强。这实现了跨风格的纹理演化和过程转移的一致性。反向绘画训练策略进一步确保了平滑、与人类对齐的生成。我们还构建了一个大规模的真实绘画过程数据集，并评估了跨媒体一致性、时间连贯性和最终图像保真度，在LPIPS、DINO和CLIP指标上取得了强结果。最后，我们的感知距离曲线（PDP）定量建模了创作序列，即构图、色块和细节精炼，反映了人类艺术进步过程。


### 论文摘要

Step-by-step painting tutorials are vital for learning artistic techniques, but existing video resources (e.g., YouTube) lack interactivity and personalization. While recent generative models have advanced artistic image synthesis, they struggle to generalize across media and often show temporal or structural inconsistencies, hindering faithful reproduction of human creative workflows. To address this, we propose a unified framework for multi-media painting process generation with a semantics-driven style control mechanism that embeds multiple media into a diffusion models conditional space and uses cross-medium style augmentation. This enables consistent texture evolution and process transfer across styles. A reverse-painting training strategy further ensures smooth, human-aligned generation. We also build a large-scale dataset of real painting processes and evaluate cross-media consistency, temporal coherence, and final-image fidelity, achieving strong results on LPIPS, DINO, and CLIP metrics. Finally, our Perceptual Distance Profile (PDP) curve quantitatively models the creative sequence, i.e., composition, color blocking, and detail refinement, mirroring human artistic progression.

---

## 78. Social-Media Based Personas Challenge: Hybrid Prediction of Common and Rare User Actions on Bluesky

**论文链接:** [http://arxiv.org/abs/2511.17241v1](http://arxiv.org/abs/2511.17241v1)

**作者:** Benjamin White, Anastasia Shimorina

**发布时间:** 2025-11-21

**备注:** 1st place at SocialSim: Social-Media Based Personas challenge 2025

### GPT解析

### 总结

本文提出了一种混合方法用于社交媒体用户行为预测，同时处理常见和罕见行为，通过特定人格模型和混合神经架构实现了良好性能，并在相关挑战赛中获得第一名。

### 背景

理解和预测社交媒体平台上的用户行为对内容推荐和平台设计至关重要。现有方法主要关注常见行为如转发和点赞，但对罕见但重要行为的预测研究较少。

### 目的

开发一种混合方法，能够处理多样化行为词汇中的频繁和罕见行为，提高社交媒体用户行为预测的准确性。

### 方法

结合四种互补方法：(1)基于历史响应模式的查找数据库系统；(2)针对常见行为的特定人格LightGBM模型，使用工程化时间和语义特征；(3)用于罕见行为分类的混合神经架构，融合文本和时间表示；(4)生成文本回复。在包含640万条对话线程、跨越12种不同用户行为和25个人格集群的Bluesky数据集上进行评估。

### 主要发现

特定人格模型在常见行为预测上达到0.64的平均宏观F1分数；罕见行为分类器在10种罕见行为上达到0.56的宏观F1分数。有效社交媒体行为预测需要针对不同行为类型的基本差异量身定制的建模策略。

### 结论

混合方法能够有效处理社交媒体上的常见和罕见行为预测，在SocialSim: Social-Media Based Personas挑战赛中获得第一名，证明了该方法的实用性和优越性。

### 翻译

理解和预测社交媒体平台上的用户行为对内容推荐和平台设计至关重要。虽然现有方法主要关注转发和点赞等常见行为，但对罕见但重要行为的预测在很大程度上仍未被探索。本文提出了一种社交媒体用户行为预测的混合方法，解决了多样化行为词汇中频繁和不频繁行为的预测问题。我们在一个包含640万条对话线程的大规模Bluesky数据集上评估了我们的方法，这些线程跨越12种不同的用户行为和25个人格集群。我们的方法结合了四种互补的方法：(i)基于历史响应模式的查找数据库系统；(ii)针对常见行为的、具有工程化时间和语义特征的特定人格LightGBM模型；(iii)用于罕见行为分类的特殊混合神经架构，融合文本和时间表示；(iv)生成文本回复。我们的特定人格模型在常见行为预测上实现了0.64的平均宏观F1分数，而我们的罕见行为分类器在10种罕见行为上实现了0.56的宏观F1分数。这些结果表明，有效的社交媒体行为预测需要针对不同行为类型的基本差异量身定制的建模策略。我们的方法在COLM 2025 Social Simulation with LLMs研讨会上组织的SocialSim: Social-Media Based Personas挑战赛中获得第一名。


### 论文摘要

Understanding and predicting user behavior on social media platforms is crucial for content recommendation and platform design. While existing approaches focus primarily on common actions like retweeting and liking, the prediction of rare but significant behaviors remains largely unexplored. This paper presents a hybrid methodology for social media user behavior prediction that addresses both frequent and infrequent actions across a diverse action vocabulary. We evaluate our approach on a large-scale Bluesky dataset containing 6.4 million conversation threads spanning 12 distinct user actions across 25 persona clusters. Our methodology combines four complementary approaches: (i) a lookup database system based on historical response patterns; (ii) persona-specific LightGBM models with engineered temporal and semantic features for common actions; (iii) a specialized hybrid neural architecture fusing textual and temporal representations for rare action classification; and (iv) generation of text replies. Our persona-specific models achieve an average macro F1-score of 0.64 for common action prediction, while our rare action classifier achieves 0.56 macro F1-score across 10 rare actions. These results demonstrate that effective social media behavior prediction requires tailored modeling strategies recognizing fundamental differences between action types. Our approach achieved first place in the SocialSim: Social-Media Based Personas challenge organized at the Social Simulation with LLMs workshop at COLM 2025.

---

## 79. Reconstruction of Surface EMG Signal using IMU data for Upper Limb Actions

**论文链接:** [http://arxiv.org/abs/2511.17200v1](http://arxiv.org/abs/2511.17200v1)

**作者:** Shubhranil Basak, Mada Hemanth, Madhav Rao

**发布时间:** 2025-11-21

**备注:** 5 pages, 5 figures

### GPT解析

### 总结

该研究探索了使用深度学习方法从惯性测量单元数据合成表面肌电信号的可能性，结果表明该方法在肌肉意图检测方面具有可行性。

### 背景

表面肌电图能提供肌肉功能的重要见解，但可能存在噪声且难以获取；而惯性测量单元为动作捕捉系统提供了稳健且可穿戴的替代方案。

### 目的

研究使用深度学习方法从6轴IMU数据合成归一化sEMG信号。

### 方法

收集了各种手臂运动的同步sEMG和IMU数据，采样率为1 KHz；基于膨胀因果卷积的滑动窗口波网模型被训练用来将IMU数据映射到sEMG信号。

### 主要发现

模型成功预测了肌肉激活的时间和大致形状，尽管峰值幅度常常被低估，但具有高时间保真度。

### 结论

该方法在假肢和康复生物反馈等应用中用于肌肉意图检测是可行的。

### 翻译

表面肌电图为肌肉功能提供了重要见解，但它可能存在噪声且难以获取。惯性测量单元为动作捕捉系统提供了稳健且可穿戴的替代方案。本文研究了使用深度学习方法从6轴IMU数据合成归一化sEMG信号。我们收集了各种手臂运动的同步sEMG和IMU数据，采样率为1 KHz。基于膨胀因果卷积的滑动窗口波网模型被训练用来将IMU数据映射到sEMG信号。结果表明，该模型成功预测了肌肉激活的时间和大致形状。尽管峰值幅度常常被低估，但高时间保真度证明了将此方法用于假肢和康复生物反馈等应用中的肌肉意图检测是可行的。


### 论文摘要

Surface Electromyography (sEMG) provides vital insights into muscle function, but it can be noisy and challenging to acquire. Inertial Measurement Units (IMUs) provide a robust and wearable alternative to motion capture systems. This paper investigates the synthesis of normalized sEMG signals from 6-axis IMU data using a deep learning approach. We collected simultaneous sEMG and IMU data sampled at 1~KHz for various arm movements. A Sliding-Window-Wave-Net model, based on dilated causal convolutions, was trained to map the IMU data to the sEMG signal. The results show that the model successfully predicts the timing and general shape of muscle activations. Although peak amplitudes were often underestimated, the high temporal fidelity demonstrates the feasibility of using this method for muscle intent detection in applications such as prosthetics and rehabilitation biofeedback.

---

## 80. VLA-4D: Embedding 4D Awareness into Vision-Language-Action Models for SpatioTemporally Coherent Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2511.17199v1](http://arxiv.org/abs/2511.17199v1)

**作者:** Hanyu Zhou, Chuanhao Ma, Gim Hee Lee

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了VLA-4D模型，一种具有4D感知能力的通用视觉-语言-动作模型，用于实现时空一致的机器人操作。

### 背景

视觉-语言-动作模型在通用机器人任务中显示出潜力，但在时空一致的操作方面仍面临挑战。现有方法通过将3D位置嵌入到视觉表示中来增强动作的空间精度，但难以实现对动作执行的时序一致性控制。

### 目的

开发一个能够实现时空一致机器人操作的通用视觉-语言-动作模型。

### 方法

1) 4D感知视觉表示：提取视觉特征，将1D时间嵌入到3D位置形成4D嵌入，并通过交叉注意力机制融合为统一视觉表示；2) 时空动作表示：扩展传统空间动作表示加入时间信息，将多模态表示与LLM对齐用于时空动作预测；3) 扩展VLA数据集添加时间动作注释用于模型微调。

### 主要发现

通过大量实验验证了VLA-4D方法在不同机器人操作任务中的优越性，能够实现空间上平滑且时间上连贯的机器人操作。

### 结论

VLA-4D模型通过设计的视觉和动作表示，能够在统一框架内实现空间上平滑且时间上连贯的机器人操作。

### 翻译

视觉-语言-动作模型在通用机器人任务中显示出潜力，但在时空一致的操作方面仍然具有挑战性，这需要细粒度的表示。通常，现有方法将3D位置嵌入到视觉表示中以增强动作的空间精度。然而，这些方法难以实现对动作执行的时序一致性控制。在这项工作中，我们提出了VLA-4D，一个具有4D感知能力的通用VLA模型，用于时空一致的机器人操作。我们的模型由两个关键设计指导：1) 4D感知视觉表示。我们提取视觉特征，将1D时间嵌入到3D位置中形成4D嵌入，并通过交叉注意力机制将它们融合为统一的视觉表示。2) 时空动作表示。我们扩展传统的空间动作表示，加入时间信息以实现时空规划，并将多模态表示与LLM对齐用于时空动作预测。在这个统一框架中，设计的视觉和动作表示共同使机器人操作在空间上平滑且在时间上连贯。此外，我们扩展了VLA数据集，添加了时间动作注释用于微调我们的模型。已经进行了大量实验来验证我们的方法在不同机器人操作任务中的优越性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人操作中时空不连贯的问题。具体来说，现有的视觉-语言-行动模型在执行机器人任务时，往往动作不连续、出现停顿或抖动，无法流畅完成精细操作。这个问题在现实中很重要，因为机器人需要在家庭、工厂等环境中执行精确且连贯的任务，如抓取、放置、组装等，不连贯的动作会导致任务失败、物体掉落或效率低下，限制了机器人在实际应用中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有VLA模型的局限性：2D模型使用单张图像导致视觉推理粗糙；3D模型虽然引入了3D位置信息提高了空间精度，但缺乏时间维度的连贯性控制。基于这些观察，作者提出需要在视觉表示和行动表示中同时嵌入时空信息。他们借鉴了现有工作：利用Qwen2.5-VL-7B作为视觉-语言模型基础架构，使用VGGT几何编码器提取3D位置信息，采用交叉注意力机制融合特征，并参考多模态对齐方法整合视觉、语言和本体感觉状态。在此基础上，作者创新性地将3D位置和1D时间结合成4D嵌入，并在行动参数中增加时间控制变量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将4D时空感知嵌入到视觉-语言-行动模型中，使机器人同时理解场景的几何结构和动态变化。整体实现流程分为两个主要阶段：1) 4D感知的视觉表示阶段：编码视频序列获取视觉和几何特征，将3D位置和1D时间编码成4D时空嵌入，通过交叉注意力机制融合到视觉特征中；2) 时空行动表示阶段：将融合后的视觉特征和本体感觉状态投影到语言嵌入空间，使用大语言模型预测包含空间控制(Δx, Δθ, Grip)和时间控制(Δt)的行动参数。训练流程也分为两个阶段：第一阶段进行4D视觉-语言对齐，第二阶段使用扩展的LIBERO数据集进行机器人任务微调。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有：1) 4D感知的视觉表示：同时融合3D位置和1D时间信息到视觉特征中，使用交叉注意力机制实现动态融合；2) 时空行动表示：在传统空间行动参数基础上增加时间控制变量(Δt)，使机器人动作既精确又连贯；3) 数据集扩展：为LIBERO数据集添加时间动作标注。相比之前的工作，2D模型缺乏空间和时间维度的精确控制；3D模型缺乏时间连贯性；其他4D模型没有同时在视觉和行动表示中嵌入时空信息，也没有明确的时间控制变量。VLA-4D通过双重时空嵌入实现了操作的时空连贯性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VLA-4D通过在视觉表示中嵌入4D时空感知和在行动表示中增加时间控制变量，实现了机器人操作的时空连贯性，显著提高了机器人任务的完成质量和效率。'}


### 论文摘要

Vision-language-action (VLA) models show potential for general robotic tasks, but remain challenging in spatiotemporally coherent manipulation, which requires fine-grained representations. Typically, existing methods embed 3D positions into visual representations to enhance the spatial precision of actions. However, these methods struggle to achieve temporally coherent control over action execution. In this work, we propose VLA-4D, a general VLA model with 4D awareness for spatiotemporally coherent robotic manipulation. Our model is guided by two key designs: 1) 4D-aware visual representation. We extract visual features, embed 1D time into 3D positions for 4D embeddings, and fuse them into a unified visual representation via a cross-attention mechanism. 2) Spatiotemporal action representation. We extend conventional spatial action representations with temporal information to enable the spatiotemporal planning, and align the multimodal representations into the LLM for spatiotemporal action prediction. Within this unified framework, the designed visual and action representations jointly make robotic manipulation spatially-smooth and temporally-coherent. In addition, we extend the VLA dataset with temporal action annotations for fine-tuning our model. Extensive experiments have been conducted to verify the superiority of our method across different tasks of robotic manipulation.

---

## 81. Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.17170v1](http://arxiv.org/abs/2511.17170v1)

**作者:** Vy Nguyen, Ziqi Xu, Jeffrey Chan, Estrid He, Feng Xia, Xiuzhen Zhang

**发布时间:** 2025-11-21

**备注:** Accepted to AAAI 2026 (Main Technical Track)

### GPT解析

### 总结

本文提出了一种名为基于方面的因果回避(ABCA)的新框架，通过分析大型语言模型内部知识多样性实现早期回避，提高回答可靠性并减少幻觉现象。

### 背景

大型语言模型(LLMs)经常产生流畅但不准确的回答，这种现象被称为'幻觉'。现有的回避方法通常依赖于生成后的信号，限制了它们提前防止不可靠回答的能力。

### 目的

开发一种能够实现早期回避的新框架，通过分析LLM内部知识多样性来评估知识可靠性，从而防止产生不可靠的回答。

### 方法

ABCA通过因果推理分析LLM内部知识多样性，这种多样性反映了从各种来源获取的参数知识的多方面性质。ABCA基于不同方面(如学科、法律背景或时间框架)估计条件因果效应，评估与给定查询相关的知识可靠性，并实现两种类型的回避：类型1(知识冲突)和类型2(知识不足)。

### 主要发现

在标准基准上的实验表明，ABCA提高了回避的可靠性，实现了最先进的性能，并增强了回避决策的可解释性。

### 结论

ABCA框架能够有效提高大型语言模型的回答质量，通过提前识别不可靠的知识并实施回避策略，减少了幻觉现象的发生。

### 翻译

大型语言模型(LLMs)经常产生流畅但不准确的回答，这种现象被称为'幻觉'。回避(abstention)是一种常见的保护措施，即模型选择不回答并输出'我不知道'等短语。然而，现有的回避方法通常依赖于生成后的信号，这限制了它们提前防止不可靠回答的能力。在本文中，我们引入了基于方面的因果回避(ABCA)，这是一个新框架，通过因果推理分析LLM内部知识多样性来实现早期回避。这种多样性反映了从各种来源获取的参数知识的多方面性质，代表了不同的方面，如学科、法律背景或时间框架。ABCA基于这些方面估计条件因果效应，以评估与给定查询相关的知识可靠性。基于这些估计，我们实现了两种类型的回避：类型1，当方面效应不一致时(知识冲突)；类型2，当方面效应一致支持回避时(知识不足)。在标准基准上的实验表明，ABCA提高了回避的可靠性，实现了最先进的性能，并增强了回避决策的可解释性。


### 论文摘要

Large Language Models (LLMs) often produce fluent but factually incorrect responses, a phenomenon known as hallucination. Abstention, where the model chooses not to answer and instead outputs phrases such as "I don't know", is a common safeguard. However, existing abstention methods typically rely on post-generation signals, such as generation variations or feedback, which limits their ability to prevent unreliable responses in advance. In this paper, we introduce Aspect-Based Causal Abstention (ABCA), a new framework that enables early abstention by analysing the internal diversity of LLM knowledge through causal inference. This diversity reflects the multifaceted nature of parametric knowledge acquired from various sources, representing diverse aspects such as disciplines, legal contexts, or temporal frames. ABCA estimates causal effects conditioned on these aspects to assess the reliability of knowledge relevant to a given query. Based on these estimates, we enable two types of abstention: Type-1, where aspect effects are inconsistent (knowledge conflict), and Type-2, where aspect effects consistently support abstention (knowledge insufficiency). Experiments on standard benchmarks demonstrate that ABCA improves abstention reliability, achieves state-of-the-art performance, and enhances the interpretability of abstention decisions.

---

## 82. A spatiotemporal Bayesian hierarchical model of heat-related mortality in Catalonia, Spain (2012--2022): The role of environmental and socioeconomic modifiers

**论文链接:** [http://arxiv.org/abs/2511.17148v1](http://arxiv.org/abs/2511.17148v1)

**作者:** David Solano, Marta Solans, Xavier Perafita, Anna Ruiz-Comellas, Marc Saez, Maria A. Barceló

**发布时间:** 2025-11-21

### GPT解析

### 总结

该研究探讨了极端热浪与死亡率的关系，发现极端热浪本身与死亡率没有独立关联，其效应被臭氧水平和社会经济因素所混杂。臭氧浓度高和收入不平等加剧了死亡风险，尤其是对老年人。

### 背景

极端热浪是主要的公共健康风险，但其与死亡率的关系可能受到空气污染和社会决定因素的混杂或修饰。

### 目的

量化极端最高温度和热浪在加泰罗尼亚（2012-2022年）对每日死亡率的影响，并评估空气污染物和社会经济因素的修饰和混杂作用。

### 方法

在夏季期间进行跨越379个基本健康区域的时间序列生态学研究，将死亡率数据与气象和空气污染数据关联，使用分层贝叶斯时空模型分析。

### 主要发现

极端热浪本身与死亡率没有独立关联，其效应完全被高臭氧水平所混杂，部分被社会经济指标所混杂。臭氧浓度高显著增加死亡风险，特别是对85岁以上人群。收入不平等和老年人口比例高也增加了脆弱性。

### 结论

加泰罗尼亚极端热浪导致的死亡风险受到臭氧水平和社会决定因素的强烈影响。适应策略应同时解决复合环境暴露和社会经济脆弱性，以更好地保护老年和弱势人群。

### 翻译

背景：极端热浪是主要的公共健康风险，但其与死亡率的关系可能受到空气污染和社会决定因素的混杂或修饰。目的：我们旨在量化极端最高温度和热浪在加泰罗尼亚（2012-2022年）对每日死亡率的影响，并评估空气污染物和社会经济因素的修饰和混杂作用。方法：我们在夏季月份期间进行了跨越379个基本健康区域的时间序列生态学研究。将来自西班牙国家统计局的死亡率数据与气象和空气污染数据相关联。使用分层贝叶斯时空模型，包含结构化和非结构化随机效应，以考虑时空依赖性以及观察到的社会经济混杂因素。结果：总共发生了730,634例死亡，其中216,989例发生在夏季。极端热浪本身与死亡率没有独立关联，因为其效应完全被高臭氧水平所混杂，部分被社会经济指标所混杂。臭氧浓度显著增加了死亡风险，特别是在85岁以上的个体中。更大的收入不平等和更高比例的老年居民也增加了脆弱性。结论：加泰罗尼亚极端热浪导致的死亡风险受到臭氧水平和社会决定因素的强烈影响。适应策略应同时解决复合环境暴露和社会经济脆弱性，以更好地保护和弱势老年人群。


### 论文摘要

Background: Extreme heat is a major public health risk, yet its relationship with mortality may be confounded or modified by air pollution and social determinants. Objectives: We aimed to quantify the effects of extreme maximum temperatures and heatwaves on daily mortality in Catalonia (2012--2022), and to assess the modifying and confounding roles of air pollutants and socioeconomic factors. Methods: We conducted a time--series ecological study across 379 basic health areas (ABS) during summer months. Mortality data from the Spanish National Statistics Institute were linked with meteorological and air pollution data. A hierarchical Bayesian spatiotemporal model, incorporating structured and unstructured random effects, was used to account for spatial and temporal dependencies, as well as observed socioeconomic confounders. Results: In total, 730,634 deaths occurred, with 216,989 in summer. Extreme heat alone was not independently associated with mortality, as its effect was fully confounded by high ozone levels and partly by socioeconomic indicators. Ozone concentrations ($\ge 120 μg/m^3$) significantly increased mortality risk, especially among individuals aged $\ge 85$ years. Greater income inequality and higher proportions of older residents also amplified vulnerability. Conclusion: Mortality risks from extreme heat in Catalonia were strongly influenced by ozone levels and social determinants. Adaptation strategies should address both compound environmental exposures together with socioeconomic vulnerability to better protect older and disadvantaged populations.

---

## 83. From Ground to Space: An Overview of the JEM-EUSO Program for the Study of UHECRs and Astrophysical Neutrinos

**论文链接:** [http://arxiv.org/abs/2511.17139v1](http://arxiv.org/abs/2511.17139v1)

**作者:** Zbigniew Plebaniak

**发布时间:** 2025-11-21

**备注:** Proceedings for the 39th International Cosmic Ray Conference, 15-24 July 2025, Geneva, Switzerland

### GPT解析

### 总结

JEM-EUSO是一个国际合作项目，研究超高能宇宙射线及相关现象，通过开发先进的UV检测技术和多平台观测策略，从太空探测这些能量超过10^20 eV的罕见粒子。

### 背景

超高能宇宙射线能量超过10^20 eV，能提供对极端天体物理过程的理解，但由于其通量极低，检测难度很大。

### 目的

开发从太空检测超高能宇宙射线的技术，通过空间观测显著增加对这些罕见现象的曝光量，详细研究宇宙射线和中微子相互作用产生的荧光和切伦科夫光。

### 方法

使用超高速、高灵敏度的UV检测大气中的扩展空气簇射；开发专门的切伦科夫相机评估地球掠射技术；结合荧光和切伦科夫探测器创建混合检测表面；采用多平台策略包括地面实验(如EUSO-TA)、气球任务(如EUSO-Balloon和EUSO-SPB1/SPB2)以及空间任务(如Mini-EUSO)进行验证和测试。

### 主要发现

荧光和切伦科夫探测器可以结合使用，创建混合检测表面；空间观测能显著增加对罕见现象的曝光量；Mini-EUSO任务提供了关于UV背景、瞬态发光事件和流星体的宝贵数据，证明了空间检测的可行性。

### 结论

正在开发跨平台方法论，但最终目标是实现空间测量。未来计划包括POEMMA空间任务进行立体观测，PBR实验整合无线电检测技术，以及向ESA提议的M-EUSO卫星任务。

### 翻译

JEM-EUSO(极端宇宙空间天文台联合探索任务)合作是一个国际倡议，研究超高能宇宙射线及相关现象。这些能量超过10^20 eV的粒子为极端天体物理过程提供了见解，但由于其低通量，检测仍然具有挑战性。JEM-EUSO技术的核心是一个超高速、高灵敏度的UV相机，能够以出色的空间和时间分辨率检测大气中的扩展空气簇射。已经开发了一个专门的切伦科夫相机来评估从高空进行地球掠射技术的可行性。荧光和切伦科夫探测器可以一起使用，创建混合检测表面。这种创新方法能够详细研究来自宇宙射线和中微子相互作用的荧光和切伦科夫光。JEM-EUSO技术将允许从太空进行观测，显著增加对这些罕见现象的曝光量。该合作采用多平台策略，地面实验如EUSO-TA校准检测系统和验证模型，气球任务如EUSO-Balloon和EUSO-SPB1/SPB2演示了从平流层的观测并测试了技术。空间任务，特别是国际空间站上的Mini-EUSO，提供了关于UV背景、瞬态发光事件和流星体的宝贵数据，并展示了未来空间检测的潜力。虽然我们正在开发跨平台方法论，但我们最终正在转向空间测量。未来的努力包括POEMMA空间任务，设计用于对超高能宇宙射线和多信使现象进行立体观测；PBR实验，整合无线电检测，计划于2027年飞行；以及提议给ESA的M-EUSO卫星任务。


### 论文摘要

The JEM-EUSO (Joint Exploratory Missions for Extreme Universe Space Observatory) collaboration is an international initiative studying ultra-high-energy cosmic rays and related phenomena. These particles, with energies exceeding 10$^{20}$~eV, provide insights into extreme astrophysical processes but remain challenging to detect due to their low flux. At the heart of JEM-EUSO's technology is an ultra-fast, highly sensitive UV camera capable of detecting EASs in the atmosphere with exceptional spatial and temporal resolution. A dedicated Cherenkov camera has been developed to evaluate the viability of the Earth-skimming technique from high altitudes. Fluorescence and Cherenkov detectors can be used together to create a hybrid detection surface. This innovative approach enables detailed studies of fluorescence and Cherenkov light from cosmic ray and neutrino interactions. The JEM-EUSO technology will allow for observations from space to significantly increase the exposure to these rare phenomena. The collaboration employs a multi-platform strategy with ground-based experiments like EUSO-TA calibrating detection systems and validating models, and balloon-borne missions such as EUSO-Balloon and EUSO-SPB1/SPB2 demonstrating observations from the stratosphere and testing technologies. Space-based missions, particularly Mini-EUSO on the ISS, have provided valuable data on UV backgrounds, TLEs, and meteoroids, as well as demonstrating the potential for future space-based detection. While we are developing a cross-platform methodology, we are ultimately moving towards space-based measurements. Future efforts include the POEMMA space mission, designed for stereoscopic observations of UHECRs and multi-messenger phenomena, the PBR experiment, which integrates radio detection and is scheduled to fly in 2027, and the M-EUSO satellite mission, proposed to ESA.

---

## 84. PEGS: Physics-Event Enhanced Large Spatiotemporal Motion Reconstruction via 3D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2511.17116v1](http://arxiv.org/abs/2511.17116v1)

**作者:** Yijun Xu, Jingrui Zhang, Hongyi Liu, Yuhan Chen, Yuanyang Wang, Qingyao Guo, Dingwen Wang, Lei Yu, Chu He

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为PEGS的框架，通过整合物理先验和事件流增强，在3D高斯溅射管道中实现去模糊的目标聚焦建模和运动恢复，解决了刚体运动在大时空尺度上重建的挑战。

### 背景

刚体运动在大时空尺度上的重建仍然是一个具有挑战性的任务，主要受限于建模范式、严重的运动模糊和物理一致性不足。

### 目的

提出PEGS框架，整合物理先验和事件流增强，在3D高斯溅射管道中执行去模糊的目标聚焦建模和运动恢复。

### 方法

提出统一的三级监督方案，通过加速度约束强制物理可行性，利用事件流进行高时间分辨率指导，采用卡尔曼正则化器融合多源观测；设计基于运动感知的模拟退火策略，根据实时运动状态自适应安排训练过程；贡献首个针对自然、快速刚体运动的RGB-Event配对数据集。

### 主要发现

实验表明，与主流动态方法相比，PEGS在大时空尺度上重建运动方面具有优越性能。

### 结论

PEGS框架有效地解决了刚体运动重建的挑战，特别是在大时空尺度上的重建。

### 翻译

在大时空尺度上的刚体运动重建由于建模范式的限制、严重的运动模糊和物理一致性不足，仍然是一个具有挑战性的任务。在这项工作中，我们提出了PEGS，一个在3D高斯溅射管道中整合物理先验和事件流增强的框架，以执行去模糊的目标聚焦建模和运动恢复。我们引入了一个统一的三级监督方案，通过加速度约束强制物理可行性，利用事件流进行高时间分辨率指导，并采用卡尔曼正则化器融合多源观测。此外，我们设计了一种基于运动感知的模拟退火策略，根据实时运动状态自适应地安排训练过程。我们还贡献了首个针对自然、快速刚体运动的RGB-Event配对数据集，涵盖多种场景。实验表明，与主流动态方法相比，PEGS在大时空尺度上重建运动方面具有优越性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决大时空尺度上的刚体运动重建问题。这个问题在现实中非常重要，因为它是现代电影、互动游戏和虚拟现实的核心驱动力。在研究中，现有方法大多针对非刚性运动，难以处理大范围、长时间的物体运动，且面临运动模糊和物理一致性不足等挑战，导致重建结果不准确或产生伪影。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有工作的局限性来设计方法。他们发现现有动态渲染方法难以处理大位移，物理增强方法依赖黑盒模型导致误差难以追溯，而基于事件的方法局限于小范围运动。作者借鉴了3D高斯飞溅的基本表示方法，但创新性地整合了物理先验和事件流增强，设计了三级监督方案和运动感知训练策略，解决了现有方法的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将物理先验与事件流增强集成到3D高斯飞溅管道中，通过三级监督方案确保物理可行性、利用事件流提高时间分辨率、融合多源观测数据。整体流程分为两个阶段：目标建模（图像去模糊、目标中心化、3D高斯重建、场景配准）和运动恢复（通过三级监督方案估计SE-3姿态变换）。此外还引入了运动感知模拟退火策略自适应调整训练过程。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出PEGS框架并贡献首个针对自然快速刚体运动的RGB-Event配对数据集；2) 开发具有加速度约束、事件增强和卡尔曼正则化的三级监督方案；3) 设计运动感知模拟退火策略自适应调度训练。相比之前工作，PEGS直接基于物理第一原理而非依赖外部模型，针对更具挑战性的大跨度快速运动场景，并解决了物理一致性不足、运动模糊和多源观测融合等关键问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PEGS通过整合物理先验和事件流增强，结合创新的三级监督方案和训练策略，有效解决了大时空尺度刚体运动重建中的关键挑战，并贡献了首个相关领域的RGB-Event配对数据集。'}


### 论文摘要

Reconstruction of rigid motion over large spatiotemporal scales remains a challenging task due to limitations in modeling paradigms, severe motion blur, and insufficient physical consistency. In this work, we propose PEGS, a framework that integrates Physical priors with Event stream enhancement within a 3D Gaussian Splatting pipeline to perform deblurred target-focused modeling and motion recovery. We introduce a cohesive triple-level supervision scheme that enforces physical plausibility via an acceleration constraint, leverages event streams for high-temporal resolution guidance, and employs a Kalman regularizer to fuse multi-source observations. Furthermore, we design a motion-aware simulated annealing strategy that adaptively schedules the training process based on real-time kinematic states. We also contribute the first RGB-Event paired dataset targeting natural, fast rigid motion across diverse scenarios. Experiments show PEGS's superior performance in reconstructing motion over large spatiotemporal scales compared to mainstream dynamic methods.

---

## 85. Modeling memory in time-respecting paths on temporal networks

**论文链接:** [http://arxiv.org/abs/2511.17108v1](http://arxiv.org/abs/2511.17108v1)

**作者:** Silvia Guerrini, Ciro Cattuto, Lorenzo Dall'Amico

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究提出了一种量化时间尊重路径中记忆的框架，并通过实证数据验证了记忆效应对扩散过程的影响。

### 背景

近距离人际互动是知识传播、规范采纳和传染病传播等扩散过程的关键决定因素，这些动态过程可以通过时间网络上的时间尊重路径进行建模。

### 目的

提出并验证一个量化时间尊重路径中记忆的框架，评估记忆效应对扩散过程的影响。

### 方法

提出一个量化记忆的框架，在多个不同环境中收集的编码人与人之间接近关系的经验数据集上进行评估，并创建一个生成模型来构建具有记忆的合成时间图。

### 主要发现

结果显示强烈的记忆效应，这种效应在不同环境、模型参数下都很稳健，并且与无记忆的零模型相比具有统计显著性；时间尊重路径中的记忆会降低扩散速度，影响时间网络上扩散过程的动态。

### 结论

记忆在时间尊重路径中扮演着重要角色，对理解扩散过程的动态机制具有重要意义，并为预测和控制扩散过程提供了新的视角。

### 翻译

人类近距离互动是知识扩散、规范采纳和传染病传播等扩散过程的关键决定因素。这些动态过程可以通过时间网络上的时间尊重路径进行建模。在此，我们提出了一种量化时间尊重路径中记忆的框架，并在几个不同环境中收集的编码人与人之间接近关系的经验数据集上对其进行了评估。我们的结果显示了强烈的记忆效应，这种效应在不同环境、模型参数下都很稳健，并且与无记忆的零模型相比具有统计显著性。我们进一步提出了一个生成模型来创建具有记忆的合成时间图，并使用它证明时间尊重路径中的记忆会降低扩散速度，影响时间网络上扩散过程的动态。


### 论文摘要

Human close-range proximity interactions are the key determinant for spreading processes like knowledge diffusion, norm adoption, and infectious disease transmission. These dynamical processes can be modeled with time-respecting paths on temporal networks. Here, we propose a framework to quantify memory in time-respecting paths and evaluate it on several empirical datasets encoding proximity between humans collected in different settings. Our results show strong memory effects, robust across settings, model parameters, and statistically significant when compared to memoryless null models. We further propose a generative model to create synthetic temporal graphs with memory and use it to show that memory in time-respecting paths decreases the diffusion speed, affecting the dynamics of spreading processes on temporal networks.

---

## 86. KNN and Time Series Based Prediction of Power Generation from Renewable Resources

**论文链接:** [http://arxiv.org/abs/2511.17102v1](http://arxiv.org/abs/2511.17102v1)

**作者:** Ismum Ul Hossain, Mohammad Nahidul Islam

**发布时间:** 2025-11-21

### GPT解析

### 总结

该研究提供了一个机器学习环境用于可再生能源预测，解决了间歇性、非线性和复杂性等常见问题。通过使用30年的综合数据集，研究评估了KNN和SARIMA两种模型，发现它们在误差指标上同样显著，但在某些情况下表现出不同趋势。长期数据有助于校准模型以应对各种影响因素，提高预测可靠性，对电网运营商、能源交易商和政策制定者具有重要价值。

### 背景

世界正在转向利用自然资源发电，需要增强预测系统以保证稳定的电力供应，并将产生的电力整合到网络系统中。

### 目的

提供一个机器学习环境用于可再生能源预测，解决实际过程中常见的缺陷：间歇性、非线性和复杂性，这些特性难以通过现有预测程序把握。

### 方法

使用约30年的综合数据集，涵盖多种可再生能源来源；评估K-最近邻(KNN)模型和带有季节性自回归积分移动平均(SARIMA)的非线性自回归分布模型；使用高时间分辨率和环境的多重参数来改进预测；利用长期历史数据进行模型校准，以应对时间波动、季节和气候对发电的影响。

### 主要发现

两种模型在误差指标方面同样显著；在某些情况下，两种模型表现出独特的趋势；30年数据的历史记录有助于更好地校准模型，应对时间波动、季节和气候对发电的影响。

### 结论

预测功能的可靠性增强得益于30年的数据，这对电网运营商、能源交易商以及制定可再生能源政策和标准的机构具有价值。

### 翻译

随着世界转向利用自然资源发电，需要增强预测系统以确保稳定的电力供应，并将产生的电力整合到网络系统中。这项工作为可再生能源预测提供了一个机器学习环境，解决了实际过程中常见的缺陷：间歇性、非线性和难以通过现有预测程序把握的复杂性。利用一个涵盖多种可再生能源来源的约30年综合数据集，我们的研究评估了两种不同方法：K-最近邻(KNN)模型和带有季节性自回归积分移动平均(SARIMA)的非线性自回归分布模型，以预测太阳能、风能和水力资源产生的总发电量。该框架使用高时间分辨率和环境的多重参数来改进预测。事实上，从误差指标来看，两种模型同样显著，并且在某些情况下表现出独特的趋势。长期历史记录允许更好地校准模型，以应对时间波动、季节和气候对发电的影响。预测功能的可靠性增强得益于30年的数据，这对电网运营商、能源交易商以及那些制定可再生能源政策和可靠性标准的机构具有价值。


### 论文摘要

As the world shifts towards utilizing natural resources for electricity generation, there is need to enhance forecasting systems to guarantee a stable electricity provision and to incorporate the generated power into the network systems. This work provides a machine learning environment for renewable energy forecasting that prevents the flaws which are usually experienced in the actual process; intermittency, nonlinearity and intricacy in nature which is difficult to grasp by ordinary existing forecasting procedures. Leveraging a comprehensive approximately 30-year dataset encompassing multiple renewable energy sources, our research evaluates two distinct approaches: K-Nearest Neighbors (KNN) model and Non-Linear Autoregressive distributed called with Seasonal Autoregressive Integrated Moving Average (SARIMA) model to forecast total power generation using the solar, wind, and hydroelectric resources. The framework uses high temporal resolution and multiple parameters of the environment to improve the predictions. The fact that both the models in terms of error metrics were equally significant and had some unique tendencies at certain circumstances. The long history allows for better model calibration of temporal fluctuations and seasonal and climatic effects on power generation. The reliability enhancement in the prediction function, which benefits from 30 years of data, has value to grid operators, energy traders, and those establishing renewable energy policies and standards concerning reliability

---

## 87. H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2511.17079v1](http://arxiv.org/abs/2511.17079v1)

**作者:** Yijie Zhu, Rui Shao, Ziyang Liu, Jie He, Jizhihui Liu, Jiuru Wang, Zitong Yu

**发布时间:** 2025-11-21

**备注:** Accepted to AAAI 2026 (Oral), Project Page: https://github.com/JiuTian-VL/H-GAR

### GPT解析

### 总结

本文提出了一种名为H-GAR的层次化交互框架，通过目标驱动的观察-动作细化方法，解决了统一视频和动作预测模型在机器人操作中的语义不匹配问题，实现了更准确的操作预测。

### 背景

统一视频和动作预测模型在机器人操作方面具有巨大潜力，因为未来观察为规划提供了上下文线索，而动作揭示了交互如何塑造环境。然而，大多数现有方法以单一且与目标无关的方式处理观察和动作生成，常常导致语义不匹配的预测和不一致的行为。

### 目的

提出一个名为H-GAR的层次化交互框架，通过目标驱动的观察-动作细化，使预测与任务目标保持一致，实现观察和动作之间的明确交互，以支持更连贯的决策。

### 方法

H-GAR首先生成目标观察和粗略动作草图，勾勒出通往目标的高层次路线，并包含两个协同模块：(1)目标条件观察合成器(GOS)基于粗粒度动作和预测的目标观察合成中间观察；(2)交互感知动作细化器(IAAR)通过利用中间观察的反馈和历史动作记忆库，将粗略动作细化为细粒度、目标一致的动作。

### 主要发现

通过将目标锚定与显式的动作-观察交互以从粗到细的方式集成，H-GAR实现了更精确的机器人操作。在模拟和真实世界机器人操作任务上的大量实验表明，H-GAR达到了最先进的性能。

### 结论

H-GAR框架通过整合目标锚定和显式的动作-观察交互，解决了现有方法的问题，实现了更精确的机器人操作，在模拟和真实世界任务中均表现出色。

### 翻译

统一视频和动作预测模型在机器人操作方面具有巨大潜力，因为未来观察为规划提供了上下文线索，而动作揭示了交互如何塑造环境。然而，大多数现有方法以单一且与目标无关的方式处理观察和动作生成，常常导致语义不匹配的预测和不一致的行为。为此，我们提出了H-GAR，一种通过目标驱动的观察-动作细化的层次化交互框架。为了将预测锚定到任务目标，H-GAR首先生成一个目标观察和一个粗略动作草图，勾勒出通往目标的高层次路线。为了在目标观察的指导下实现观察和动作之间的明确交互以支持更连贯的决策，我们设计了两个协同模块。(1) 目标条件观察合成器(GOS)基于粗粒度动作和预测的目标观察合成中间观察。(2) 交互感知动作细化器(IAAR)通过利用中间观察的反馈和一个编码先前动作以确保时间一致性的历史动作记忆库，将粗略动作细化为细粒度、目标一致的动作。通过以从粗到细的方式将目标锚定与显式的动作-观察交互相结合，H-GAR实现了更精确的操作。在模拟和真实世界机器人操作任务上的大量实验表明，H-GAR达到了最先进的性能。


### 论文摘要

Unified video and action prediction models hold great potential for robotic manipulation, as future observations offer contextual cues for planning, while actions reveal how interactions shape the environment. However, most existing approaches treat observation and action generation in a monolithic and goal-agnostic manner, often leading to semantically misaligned predictions and incoherent behaviors. To this end, we propose H-GAR, a Hierarchical interaction framework via Goal-driven observation-Action Refinement.To anchor prediction to the task objective, H-GAR first produces a goal observation and a coarse action sketch that outline a high-level route toward the goal. To enable explicit interaction between observation and action under the guidance of the goal observation for more coherent decision-making, we devise two synergistic modules. (1) Goal-Conditioned Observation Synthesizer (GOS) synthesizes intermediate observations based on the coarse-grained actions and the predicted goal observation. (2) Interaction-Aware Action Refiner (IAAR) refines coarse actions into fine-grained, goal-consistent actions by leveraging feedback from the intermediate observations and a Historical Action Memory Bank that encodes prior actions to ensure temporal consistency. By integrating goal grounding with explicit action-observation interaction in a coarse-to-fine manner, H-GAR enables more accurate manipulation. Extensive experiments on both simulation and real-world robotic manipulation tasks demonstrate that H-GAR achieves state-of-the-art performance.

---

## 88. REArtGS++: Generalizable Articulation Reconstruction with Temporal Geometry Constraint via Planar Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2511.17059v1](http://arxiv.org/abs/2511.17059v1)

**作者:** Di Wu, Liu Liu, Anran Huang, Yuyan Liu, Qiaoyu Jun, Shaofan Liu, Liangtu Song, Cewu Lu

**发布时间:** 2025-11-21

**备注:** 10 pages, 7 figures

### GPT解析

### 总结

本文提出了REArtGS++方法，通过时间几何约束和平面高斯飞溅技术改进关节式物体的可推广重建

### 背景

关节式物体如抽屉和冰箱在日常环境中普遍存在，REArtGS方法使用多视角RGB图像进行部件级表面重建和关节参数估计

### 目的

解决REArtGS在处理螺钉关节或多部件物体时的困难，并添加对未见状态的几何约束

### 方法

为每个关节建模解耦螺旋运动，通过部件运动混合联合优化部件感知高斯与关节参数，引入时间连续几何约束鼓励高斯平面化，并通过泰勒一阶展开实现平面法线和深度的时间一致规则化

### 主要发现

在合成和真实世界的关节式物体上的实验表明，REArtGS++在可推广的部件级表面重建和关节参数估计方面优于现有方法

### 结论

REArtGS++是一种改进的方法，能够更好地处理关节式物体的重建和关节参数估计

### 翻译

关节式物体在日常环境中普遍存在，如抽屉和冰箱。针对它们的部件级表面重建和关节参数估计，REArtGS引入了一种使用两种不同状态多视角RGB图像的类别无关方法。然而，我们发现REArtGS在处理螺钉关节或多部件物体时仍有困难，并且缺乏对未见状态的几何约束。在本文中，我们提出了REArtGS++，一种具有时间几何约束和平面高斯飞溅的可推广关节式物体重建新方法。我们首先为每个关节建模解耦的螺旋运动，无需类型先验，并通过部件运动混合联合优化部件感知高斯与关节参数。为了为关节式建模引入时间连续的几何约束，我们鼓励高斯平面化，并通过泰勒一阶展开提出平面法线和深度之间的时间一致规则化。在合成和真实世界关节式物体上的大量实验表明，与现有方法相比，我们在可推广的部件级表面重建和关节参数估计方面具有优势。项目网站：https://sites.google.com/view/reartgs2/home

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决可关节物体的部分级别表面重建和关节参数估计问题，特别是处理螺旋关节和多部分物体的局限性，以及对未见状态缺乏几何约束的挑战。这个问题在现实中很重要，因为可关节物体（如家具、电器、工具等）在日常生活中无处不在，准确重建它们的几何结构和运动特性对机器人技术、具身智能、虚拟现实等领域的交互操作、数字孪生和虚拟仿真应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从现有方法的局限性出发进行思考，发现REArtGS等方法在螺旋关节和多部分物体上表现不佳，且依赖关节类型先验限制了泛化能力。作者借鉴了多项现有工作：采用3D高斯溅射(3DGS)的显式表示方法；参考PGSR通过压缩3D高斯为平面获得准确法线和深度；受REACTO启发使用马氏距离进行部分分割；借鉴ArtGS的初始化策略但进行了改进。基于这些思考，作者设计了REArtGS++方法，包括解耦的螺旋运动模型、部分感知的高斯表示、时间几何约束和局部一致投票机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用平面高斯溅射表示可关节物体的几何结构，建立无需关节类型先验的解耦螺旋运动模型，通过部分感知的高斯表示和部分运动混合联合优化部分分割和关节参数，并引入时间几何约束实现整个运动过程中的几何一致性。整体流程包括：1)初始化阶段，对两个状态的高斯进行优化并识别动态高斯；2)优化阶段，将3D高斯压缩为2D平面，计算部分分割掩码，通过部分运动混合计算高斯位置变化，应用时间几何约束和局部一致投票机制；3)动态网格提取阶段，根据优化后的高斯和关节参数在任何状态提取动态表面网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)解耦的螺旋运动模型，无需关节类型先验可处理任意类型关节运动；2)时间几何约束，通过泰勒一阶扩展实现整个时间区间的几何一致性；3)局部一致投票机制，解决部分边界区域的分割模糊问题。相比REArtGS和ArtGS等前工作，REArtGS++不依赖关节类型先验，能处理螺旋关节；引入了时间连续的几何约束而非仅离散状态约束；通过局部投票机制提高部分分割准确性；计算效率更高(16-20分钟vs 70分钟)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'REArtGS++通过引入解耦的螺旋运动模型和时间几何约束，实现了对任意类型关节物体的高质量部分级别表面重建和精确的关节参数估计，显著提升了在螺旋关节和多部分物体上的表现。'}


### 论文摘要

Articulated objects are pervasive in daily environments, such as drawers and refrigerators. Towards their part-level surface reconstruction and joint parameter estimation, REArtGS~\cite{wu2025reartgs} introduces a category-agnostic approach using multi-view RGB images at two different states. However, we observe that REArtGS still struggles with screw-joint or multi-part objects and lacks geometric constraints for unseen states. In this paper, we propose REArtGS++, a novel method towards generalizable articulated object reconstruction with temporal geometry constraint and planar Gaussian splatting. We first model a decoupled screw motion for each joint without type prior, and jointly optimize part-aware Gaussians with joint parameters through part motion blending. To introduce time-continuous geometric constraint for articulated modeling, we encourage Gaussians to be planar and propose a temporally consistent regularization between planar normal and depth through Taylor first-order expansion. Extensive experiments on both synthetic and real-world articulated objects demonstrate our superiority in generalizable part-level surface reconstruction and joint parameter estimation, compared to existing approaches. Project Site: https://sites.google.com/view/reartgs2/home.

---

## 89. Stable Offline Hand-Eye Calibration for any Robot with Just One Mark

**论文链接:** [http://arxiv.org/abs/2511.17001v1](http://arxiv.org/abs/2511.17001v1)

**作者:** Sicheng Xie, Lingchen Meng, Zhiying Du, Shuyuan Tu, Haidong Cao, Jiaqi Leng, Zuxuan Wu, Yu-Gang Jiang

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出CalibAll方法，仅需在机器人末端执行器上做一个标记，即可实现训练自由、稳定且准确的相机外参估计。

### 背景

模仿学习通过学习从相机空间观察到机器人空间动作的映射函数，在多种机器人任务中取得成功。研究表明使用机器人到相机的变换信息（相机外参）有助于学习过程，但相机外参通常不可用，现有估计方法易陷入局部最小值且泛化能力差。

### 目的

开发一种简单而有效的方法，只需要一个标记，就能在不同机器人和数据集上实现训练自由、稳定且准确的相机外参估计。

### 方法

CalibAll方法包括：1)在末端执行器上标注一个标记；2)利用视觉基础模型的对应能力自动定位对应标记；3)通过点跟踪和3D EEF轨迹，使用时间透视点PnP获得粗略相机外参；4)通过基于渲染的优化精炼估计，使渲染和真实掩码对齐。

### 主要发现

实验结果表明，该方法优于最先进方法，在三个机器人平台上表现出强大的鲁棒性和通用有效性。同时能产生深度图、链式掩码和末端执行器2D轨迹等有用辅助标注。

### 结论

CalibAll是一种简单而有效的相机外参估计方法，仅需一个标记即可实现训练自由、稳定且准确的估计，在不同机器人和数据集上都表现出色。

### 翻译

模仿学习通过学习从相机空间观察到机器人空间动作的映射函数，在多种机器人任务中取得了显著成功。最近的研究表明，使用机器人到相机的变换信息（即相机外参）有利于学习过程并产生更好的结果。然而，相机外参通常不可用，且估计方法通常容易陷入局部最小值且泛化能力差。在本文中，我们提出了CalibAll，这是一种简单而有效的方法，只需要一个标记，就能通过粗到细的校准流程，在不同机器人和数据集上实现训练自由、稳定且准确的相机外参估计。特别是，我们在末端执行器上标注一个标记，并利用视觉基础模型的对应能力，自动在不同机器人的不同数据集中定位对应标记。使用这个标记，结合点跟踪和3D EEF轨迹，我们通过时间透视点PnP获得粗略的相机外参。这个估计通过基于渲染的优化进一步精炼，使渲染和真实掩码对齐，从而获得准确且稳定的相机外参。实验结果表明，我们的方法优于最先进的方法，在三个机器人平台上表现出强大的鲁棒性和通用有效性。它还能产生有用的辅助标注，如深度图、链式掩码和末端执行器2D轨迹，这些可以进一步支持下游任务。


### 论文摘要

Imitation learning has achieved remarkable success in a variety of robotic tasks by learning a mapping function from camera-space observations to robot-space actions. Recent work indicates that the use of robot-to-camera transformation information ({\ie}, camera extrinsics) benefits the learning process and produces better results. However, camera extrinsics are oftentimes unavailable and estimation methods usually suffer from local minima and poor generalizations. In this paper, we present CalibAll, a simple yet effective method that \textbf{requires only a single mark} and performs training-free, stable, and accurate camera extrinsic estimation across diverse robots and datasets through a coarse-to-fine calibration pipeline. In particular, we annotate a single mark on an end-effector (EEF), and leverage the correspondence ability emerged from vision foundation models (VFM) to automatically localize the corresponding mark across robots in diverse datasets. Using this mark, together with point tracking and the 3D EEF trajectory, we obtain a coarse camera extrinsic via temporal Perspective-n-Point (PnP). This estimate is further refined through a rendering-based optimization that aligns rendered and ground-true masks, yielding accurate and stable camera extrinsic. Experimental results demonstrate that our method outperforms state-of-the-art approaches, showing strong robustness and general effectiveness across three robot platforms. It also produces useful auxiliary annotations such as depth maps, link-wise masks, and end-effector 2D trajectories, which can further support downstream tasks.

---

## 90. Real-Time Cooked Food Image Synthesis and Visual Cooking Progress Monitoring on Edge Devices

**论文链接:** [http://arxiv.org/abs/2511.16965v1](http://arxiv.org/abs/2511.16965v1)

**作者:** Jigyasa Gupta, Soumya Goyal, Anil Kumar, Ishan Jindal

**发布时间:** 2025-11-21

**备注:** 13 pages, 11 figures

### GPT解析

### 总结

该研究提出了一种边缘高效的食谱和烹饪状态引导的生成器，用于从原始食物图像合成真实的烹饪食物图像，并引入了烹饪图像相似性(CIS)指标来确保时间一致性和烹饪合理性。

### 背景

在边缘设备上从原始输入合成真实的烹饪食物图像是一项具有挑战性的生成任务，需要模型捕捉烹饪过程中纹理、颜色和结构的复杂变化。现有的图像到图像生成方法通常产生不真实的结果或对边缘部署来说资源消耗太大。

### 目的

引入第一个基于烤箱的烹饪进度数据集，包含厨师标注的熟度级别；提出一种边缘高效的生成器，根据原始食物图像合成真实的食物图像，使用户能够选择视觉目标而非固定预设。

### 方法

提出边缘高效的食谱和烹饪状态引导的生成器；引入特定领域的烹饪图像相似性(CIS)指标，该指标既作为训练损失也作为进度监控信号。

### 主要发现

该模型在FID分数上显著优于现有基线，在作者的数据集上改进30%，在公共数据集上改进60%。

### 结论

该方法能够合成真实的烹饪食物图像，适合边缘设备部署，同时保持时间一致性和烹饪合理性。

### 翻译

从原始输入在边缘设备上合成真实的烹饪食物图像是一项具有挑战性的生成任务，需要模型在烹饪过程中捕捉纹理、颜色和结构的复杂变化。现有的图像到图像生成方法通常产生不真实的结果，或者对于边缘部署来说资源消耗太大。我们引入了第一个基于烤箱的烹饪进度数据集，包含厨师标注的熟度级别，并提出了一种边缘高效的食谱和烹饪状态引导的生成器，根据原始食物图像合成真实的食物图像。这种表述使用户能够选择偏好的视觉目标，而不是固定的预设。为确保时间一致性和烹饪合理性，我们引入了一个特定领域的烹饪图像相似性(CIS)指标，该指标既作为训练损失，也作为进度监控信号。我们的模型在FID分数上显著优于现有基线，在我们的数据集上改进30%，在公共数据集上改进60%。


### 论文摘要

Synthesizing realistic cooked food images from raw inputs on edge devices is a challenging generative task, requiring models to capture complex changes in texture, color and structure during cooking. Existing image-to-image generation methods often produce unrealistic results or are too resource-intensive for edge deployment. We introduce the first oven-based cooking-progression dataset with chef-annotated doneness levels and propose an edge-efficient recipe and cooking state guided generator that synthesizes realistic food images conditioned on raw food image. This formulation enables user-preferred visual targets rather than fixed presets. To ensure temporal consistency and culinary plausibility, we introduce a domain-specific \textit{Culinary Image Similarity (CIS)} metric, which serves both as a training loss and a progress-monitoring signal. Our model outperforms existing baselines with significant reductions in FID scores (30\% improvement on our dataset; 60\% on public datasets)

---

## 91. Point-Supervised Facial Expression Spotting with Gaussian-Based Instance-Adaptive Intensity Modeling

**论文链接:** [http://arxiv.org/abs/2511.16952v1](http://arxiv.org/abs/2511.16952v1)

**作者:** Yicheng Deng, Hideaki Hayashi, Hajime Nagahara

**发布时间:** 2025-11-21

### GPT解析

### 总结

该论文提出了一种点监督面部表情定位方法，通过双分支框架和GIM模块解决了传统方法对时序边界标注的依赖问题，实现了更高效的面部表情分析。

### 背景

自动面部表情定位旨在未修剪视频中识别面部表情实例，对面部表情分析至关重要。现有方法主要依赖全监督学习和昂贵耗时的时序边界标注。

### 目的

研究点监督面部表情定位(P-FES)，其中每个实例仅需一个时间戳标注，并提出一个独特的双分支框架用于P-FES。

### 方法

提出基于高斯的自适应实例强度建模(GIM)模块生成软伪标签；设计类感知顶点分类分支区分宏观和微观表情；引入强度感知对比损失增强特征学习；两个分支在推理阶段独立工作分别负责表情提案生成和分类。

### 主要发现

在SAMM-LV、CAS(ME)²和CAS(ME)³数据集上的大量实验证明了所提出框架的有效性。

### 结论

提出的双分支框架和GIM模块能够在仅使用点标注的情况下有效进行面部表情定位。

### 翻译

自动面部表情定位旨在识别未修剪视频中的面部表情实例，对面部表情分析至关重要。现有方法主要专注于全监督学习，并依赖于昂贵、耗时的时序边界标注。本文研究点监督面部表情定位(P-FES)，其中每个实例只需要一个时间戳标注用于训练。我们提出了一个独特的双分支框架用于P-FES。首先，为缓解硬伪标签的局限性，我们提出了基于高斯的自适应实例强度建模(GIM)模块，用于实例级表情强度分布建模以生成软伪标签。通过检测每个点标签周围的伪顶点帧，估计持续时间，并构建实例级高斯分布，GIM为表情帧分配软伪标签以提供更可靠的强度监督。将GIM模块整合到我们的框架中以优化类无关表情强度分支。其次，我们设计了一个类感知顶点分类分支，仅基于伪顶点帧区分宏观和微观表情。在推理过程中，两个分支独立工作：类无关表情强度分支生成表情提案，而类感知顶点分类分支负责宏观和微观表情分类。此外，我们引入了强度感知对比损失，通过对比中性帧与不同强度的表情帧来增强判别性特征学习并抑制中性噪声。在SAMM-LV、CAS(ME)²和CAS(ME)³数据集上的大量实验证明了我们提出框架的有效性。


### 论文摘要

Automatic facial expression spotting, which aims to identify facial expression instances in untrimmed videos, is crucial for facial expression analysis. Existing methods primarily focus on fully-supervised learning and rely on costly, time-consuming temporal boundary annotations. In this paper, we investigate point-supervised facial expression spotting (P-FES), where only a single timestamp annotation per instance is required for training. We propose a unique two-branch framework for P-FES. First, to mitigate the limitation of hard pseudo-labeling, which often confuses neutral and expression frames with various intensities, we propose a Gaussian-based instance-adaptive intensity modeling (GIM) module to model instance-level expression intensity distribution for soft pseudo-labeling. By detecting the pseudo-apex frame around each point label, estimating the duration, and constructing an instance-level Gaussian distribution, GIM assigns soft pseudo-labels to expression frames for more reliable intensity supervision. The GIM module is incorporated into our framework to optimize the class-agnostic expression intensity branch. Second, we design a class-aware apex classification branch that distinguishes macro- and micro-expressions solely based on their pseudo-apex frames. During inference, the two branches work independently: the class-agnostic expression intensity branch generates expression proposals, while the class-aware apex-classification branch is responsible for macro- and micro-expression classification.Furthermore, we introduce an intensity-aware contrastive loss to enhance discriminative feature learning and suppress neutral noise by contrasting neutral frames with expression frames with various intensities. Extensive experiments on the SAMM-LV, CAS(ME)$^2$, and CAS(ME)$^3$ datasets demonstrate the effectiveness of our proposed framework.

---

## 92. FingerCap: Fine-grained Finger-level Hand Motion Captioning

**论文链接:** [http://arxiv.org/abs/2511.16951v1](http://arxiv.org/abs/2511.16951v1)

**作者:** Xin Shen, Rui Zhu, Lei Shen, Xinyu Wang, Kaihao Zhang, Tianqing Zhu, Shuchen Wu, Chenxi Miao, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang, Xin Yu

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了FingerCap方法，用于生成细粒度手指级手部运动的文本描述，并构建了FingerCap-40K数据集。针对时间稀疏性问题，作者开发了FiGOP方法，将RGB关键帧与手部关键点配对，有效捕捉细微手指运动而不增加RGB密度。实验表明现有Video-MLLMs在手指级推理上仍有困难，而FiGOP增强的模型表现更优。

### 背景

理解细粒度的人手运动对于视觉感知、具身智能和多模态通信至关重要。当前视频多模态大模型面临时间稀疏性的根本瓶颈，因为稀疏RGB采样不足以捕捉细微手指运动下的微妙高频动态。

### 目的

提出细粒度手指级手部运动描述生成方法(FingerCap)，生成能够捕捉手部动作详细手指级语义的文本描述，并解决现有方法在捕捉细微手指运动上的局限性。

### 方法

构建FingerCap-40K数据集，包含4万对手部运动视频和描述；使用基于大模型的评估标准HandJudge评估手指级正确性和运动完整性；开发FiGOP方法，将RGB关键帧与后续手部关键点配对，使用轻量级时间编码器将关键点转换为运动嵌入并与RGB特征集成。

### 主要发现

当前开源和闭源Video-MLLMs在手指级推理方面仍有困难；FiGOP增强的模型在HandJudge评估和人类研究中表现一致提升，能够有效恢复细粒度时间线索而不增加RGB密度。

### 结论

FiGOP方法通过将经典GOP概念适应于手指运动，解决了时间稀疏性问题，有效捕捉了细微的高频手指运动，为细粒度手部动作理解提供了新的解决方案。

### 翻译

理解细粒度的人手运动是视觉感知、具身智能和多模态通信的基础。在本工作中，我们提出了细粒度手指级手部运动描述生成(FingerCap)，旨在生成能够捕捉手部动作详细手指级语义的文本描述。为支持这一任务，我们整理了FingerCap-40K，这是一个包含4万对配对手部运动视频和描述的大规模语料库，涵盖两个互补来源：简洁的指令式手指动作和多样化的自然手部物体交互。为实现有效评估，我们采用HandJudge，这是一种基于大模型的评估标准，用于测量手指级正确性和运动完整性。时间稀疏性仍然是当前视频多模态大模型的基本瓶颈，因为稀疏RGB采样不足以捕捉细微手指运动下的微妙高频动态。作为一种简单且计算友好的解决方案，我们引入了FiGOP(手指组图)，它将每个RGB关键帧与后续手部关键点配对，直到下一个关键帧。轻量级时间编码器将关键点转换为运动嵌入，并将其与RGB特征集成。FiGOP将经典GOP概念适应于手指运动，在不增加RGB密度的情况下恢复细粒度时间线索。在FingerCap-40K上的实验表明，强大的开源和闭源视频多模态大模型在手指级推理方面仍有困难，而我们的FiGOP增强模型在HandJudge和人类研究中表现一致提升。


### 论文摘要

Understanding fine-grained human hand motion is fundamental to visual perception, embodied intelligence, and multimodal communication. In this work, we propose Fine-grained Finger-level Hand Motion Captioning (FingerCap), which aims to generate textual descriptions that capture detailed finger-level semantics of hand actions. To support this task, we curate FingerCap-40K, a large-scale corpus of 40K paired hand-motion videos and captions spanning two complementary sources: concise instruction-style finger motions and diverse, naturalistic hand-object interactions. To enable effective evaluation, we employ HandJudge, a LLM-based rubric that measures finger-level correctness and motion completeness. Temporal sparsity remains a fundamental bottleneck for current Video-MLLMs, since sparse RGB sampling is insufficient to capture the subtle, high-frequency dynamics underlying fine finger motions. As a simple and compute-friendly remedy, we introduce FiGOP (Finger Group-of-Pictures), which pairs each RGB keyframe with subsequent hand keypoints until the next keyframe. A lightweight temporal encoder converts the keypoints into motion embeddings and integrates them with RGB features. FiGOP adapts the classic GOP concept to finger motion, recovering fine temporal cues without increasing RGB density. Experiments on FingerCap-40K show that strong open- and closed-source Video-MLLMs still struggle with finger-level reasoning, while our FiGOP-augmented model yield consistent gains under HandJudge and human studies.

---

## 93. OmniGround: A Comprehensive Spatio-Temporal Grounding Benchmark for Real-World Complex Scenarios

**论文链接:** [http://arxiv.org/abs/2511.16937v1](http://arxiv.org/abs/2511.16937v1)

**作者:** Hong Gao, Jingyu Wu, Xiangkai Xu, Kangni Xie, Yunchen Zhang, Bin Zhong, Xurui Gao, Min-Ling Zhang

**发布时间:** 2025-11-21

**备注:** 20 pages

### GPT解析

### 总结

本文提出了解决时空视频定位(STVG)模型在现实世界场景中性能不足的问题，通过引入OmniGround基准测试和DeepSTG评估框架，并提出了PG-TAF框架显著提升了模型性能。

### 背景

时空视频定位(STVG)旨在基于自然语言描述在视频中定位目标对象。尽管多模态大语言模型有所进展，但当前模型与涉及多样对象和复杂查询的现实世界需求之间存在显著差距，表现为类别偏差、过度简化的推理和较差的语言鲁棒性。

### 目的

解决现有STVG模型在复杂现实世界场景中的局限性，包括类别偏差、过度简化的推理和较差的语言鲁棒性问题。

### 方法

1) 引入OmniGround基准测试，包含3475个视频，涵盖81个类别和复杂查询；2) 提出Forward-Backward-Refinement标注流程，结合多方向跟踪和智能错误校正；3) 引入DeepSTG评估框架，从四个互补维度量化数据集质量；4) 提出PG-TAF框架，将STVG分解为高级时空定位和细粒度时空传播的两阶段方法。

### 主要发现

1) 评估显示在复杂现实场景中性能平均下降10.4%，特别是在小/遮挡对象和复杂空间关系方面；2) PG-TAF框架在OmniGround上实现了显著性能提升；3) PG-TAF在四个基准测试中保持一致的性能提升。

### 结论

通过引入更全面的基准测试和评估框架，以及提出PG-TAF框架，有效解决了STVG模型在复杂现实场景中的局限性，显著提升了模型性能。

### 翻译

时空视频定位(STVG)旨在基于自然语言描述在视频中定位目标对象。尽管多模态大语言模型最近有所进展，但当前模型与涉及多样对象和复杂查询的现实世界需求之间仍存在显著差距。我们将此归因于有限的基准测试范围，导致模型表现出类别偏差、过度简化的推理和较差的语言鲁棒性。为解决这些局限性，我们引入了OmniGround，一个包含3475个视频、涵盖81个类别和复杂现实世界查询的全面基准测试。我们提出了Forward-Backward-Refinement标注流程，结合多方向跟踪和智能错误校正以获得高质量标签。我们进一步引入了DeepSTG，一个系统评估框架，从四个互补维度量化数据集质量，超越表面统计数据。评估显示在复杂现实场景中性能平均下降10.4%，特别是在小/遮挡对象和复杂空间关系方面。基于这些发现，我们提出了PG-TAF，一个无需训练的两阶段框架，将STVG分解为高级时空定位和细粒度时空传播。实验证明PG-TAF在OmniGround上实现了显著性能提升，并在四个基准测试中保持一致的性能提升。


### 论文摘要

Spatio-Temporal Video Grounding (STVG) aims to localize target objects in videos based on natural language descriptions. Despite recent advances in Multimodal Large Language Models, a significant gap remains between current models and real-world demands involving diverse objects and complex queries. We attribute this to limited benchmark scope, causing models to exhibit category bias, oversimplified reasoning, and poor linguistic robustness. To address these limitations, we introduce OmniGround, a comprehensive benchmark with 3,475 videos spanning 81 categories and complex real-world queries. We propose the Forward-Backward-Refinement annotation pipeline that combines multi-directional tracking with intelligent error correction for high-quality labels. We further introduce DeepSTG, a systematic evaluation framework quantifying dataset quality across four complementary dimensions beyond superficial statistics. Evaluations reveal performance average drop of 10.4% on complex real-world scenes, particularly with small/occluded objects and intricate spatial relations. Motivated by these, we propose PG-TAF, a training-free two-stage framework decomposing STVG into high-level temporal grounding and fine-grained spatio-temporal propagation. Experiments demonstrate PG-TAF achieves 25.6% and 35.6% improvements in m\_tIoU and m\_vIoU on OmniGround with consistent gains across four benchmarks.

---

## 94. Leveraging CVAE for Joint Configuration Estimation of Multifingered Grippers from Point Cloud Data

**论文链接:** [http://arxiv.org/abs/2511.17276v1](http://arxiv.org/abs/2511.17276v1)

**作者:** Julien Merand, Boris Meden, Mathieu Grossard

**发布时间:** 2025-11-21

**DOI:** 10.1109/CASE58245.2025.11164060

### GPT解析

### 总结

论文提出了一种仅通过多指夹爪的多关节链点云数据来确定关节配置的高效方法，相比传统逆运动学方法具有计算速度快、精度高的优势。

### 背景

传统逆运动学(IK)技术在处理复杂运动学时面临挑战，需要考虑所有中间指骨位置进行事后决策，或依赖算法进行数值近似解。

### 目的

开发一种仅从点云数据确定多指夹爪关节配置的高效方法，避免传统方法的计算复杂性和决策需求。

### 方法

使用条件变分自编码器(CVAE)架构，将关键结构元素的点云数据作为输入，重建相应的关节配置，实现机器学习驱动的关节配置估计。

### 主要发现

该方法在MultiDex抓取数据集上使用Allegro Hand验证，仅需0.05毫秒即可完成计算，达到与最先进方法相当的准确性。

### 结论

该方法在AI驱动的抓取规划技术背景下，对于关节配置估计是有效的，为实时抓取系统提供了实用解决方案。

### 翻译

本文提出了一种仅通过多指夹爪的多关节链点云数据来确定关节配置的高效方法，这些点云数据可由视觉传感器、仿真甚至生成神经网络产生。众所周知的逆运动学(IK)技术可以基于指尖姿态提供数学上精确的解（当解存在时），但通常需要通过考虑夹爪手指中所有中间指骨的位置来进行事后决策，或依赖于算法对更复杂的运动学进行数值近似解。相比之下，我们的方法利用机器学习隐式地克服了这些挑战。这是通过一个条件变分自编码器(CVAE)实现的，它将关键结构元素的点云数据作为输入，并重建相应的关节配置。我们在MultiDex抓取数据集上使用Allegro Hand验证了我们的方法，在0.05毫秒内运行，并达到与最先进方法相当的准确性。这突显了我们的管道在AI驱动的抓取规划技术更广泛背景下，用于关节配置估计的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何仅从点云数据确定多指夹爪的关节配置问题。这个问题在现实中很重要，因为精确的关节控制对于将指尖准确放置在物体上的期望接触点至关重要，有效的抓取依赖于对可达接触点的知识，且所选配置必须是手指运动学可达且与环境无碰撞的。传统逆运动学方法在动态环境中往往不够可靠和高效，可能需要大量模拟试验或面临多解问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统逆运动学方法的局限性，包括多解问题和数值近似困难。他们回顾了早期使用神经网络、遗传算法等方法解决IK问题的尝试，发现这些方法存在误差大且通常只返回单个解的问题。作者借鉴了近期生成网络提供完整解决方案空间的思想，设计了基于条件变分自编码器(CVAE)的方法，直接推断关节配置。他们还借鉴了PointNet架构处理点云数据和变分自编码器的基本原理，包括编码器-解码器结构和KL散度正则化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用条件变分自编码器直接从夹爪的点云数据重建关节配置，隐式处理逆运动学问题的复杂性，如多解问题，并自动选择最合适配置。整体流程包括：1)数据生成：采样关节配置，过滤碰撞配置，创建点云；2)模型架构：使用PointNet编码器提取点云特征，与关节值特征连接后映射到潜在空间，解码器使用另一个PointNet和潜在变量重建关节角度；3)训练：通过最大化证据下界进行训练，使用包含重建损失和KL散度的损失函数；4)推理：输入点云数据，快速预测关节配置，推理时间小于0.05毫秒。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用CVAE直接从点云推断关节配置，无需显式逆运动学计算；2)考虑点云中的中间指节，识别与特定关节配置对应的模式；3)提出三种点云数据集生成方法(完全密集、聚类和手印点云)适应不同应用；4)机器人中心的数据生成方法，仅需夹爪的URDF文件；5)使用动态调整的β参数缓解KL消失问题。相比之前工作，本文方法直接从点云推断关节配置，隐式处理多解问题，不需要显式逆运动学计算或额外的后处理步骤，同时实现了实时处理速度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于条件变分自编码器的高效方法，能够直接从点云数据推断多指夹爪的关节配置，在保持与最先进方法相当精度的同时实现了实时处理速度，为机器人抓取规划和控制提供了新的AI驱动解决方案。'}


### 论文摘要

This paper presents an efficient approach for determining the joint configuration of a multifingered gripper solely from the point cloud data of its poly-articulated chain, as generated by visual sensors, simulations or even generative neural networks. Well-known inverse kinematics (IK) techniques can provide mathematically exact solutions (when they exist) for joint configuration determination based solely on the fingertip pose, but often require post-hoc decision-making by considering the positions of all intermediate phalanges in the gripper's fingers, or rely on algorithms to numerically approximate solutions for more complex kinematics. In contrast, our method leverages machine learning to implicitly overcome these challenges. This is achieved through a Conditional Variational Auto-Encoder (CVAE), which takes point cloud data of key structural elements as input and reconstructs the corresponding joint configurations. We validate our approach on the MultiDex grasping dataset using the Allegro Hand, operating within 0.05 milliseconds and achieving accuracy comparable to state-of-the-art methods. This highlights the effectiveness of our pipeline for joint configuration estimation within the broader context of AI-driven techniques for grasp planning.

---

## 95. Range-Edit: Semantic Mask Guided Outdoor LiDAR Scene Editing

**论文链接:** [http://arxiv.org/abs/2511.17269v1](http://arxiv.org/abs/2511.17269v1)

**作者:** Suchetan G. Uppur, Hemant Kumar, Vaibhav Kumar

**发布时间:** 2025-11-21

**备注:** 8 pages, 9 figures

### GPT解析

### 总结

本研究提出了一种基于语义掩码引导的LiDAR点云生成方法，通过编辑真实世界的LiDAR扫描数据来创建多样化的合成点云数据集，从而提高自动驾驶系统的鲁棒性和泛化能力。

### 背景

训练自动驾驶和导航系统需要大量且多样化的点云数据集来捕捉复杂边缘情况，但从真实世界获取此类数据具有挑战性，特别是关键边缘情况。当前基于手工制作3D虚拟环境的模拟方法耗时、计算成本高，且难以完全模拟真实场景的复杂性。

### 目的

解决获取多样化LiDAR点云数据的难题，特别是关键边缘情况，以提高自动驾驶系统的泛化能力和鲁棒性。

### 方法

提出一种基于扩散生成的点云生成方法，将点云转换为2D范围视图图像作为中间表示，使用基于凸包的语义掩码进行语义编辑，通过提供环境中物体的尺寸、方向和位置信息来指导生成过程，确保几何一致性和真实性。

### 主要发现

该方法能够生成高质量的LiDAR点云，能够产生复杂的边缘情况和动态场景，在KITTI-360数据集上验证了其有效性。

### 结论

提供了一种经济且可扩展的解决方案来生成多样化的LiDAR数据，是提高自动驾驶系统鲁棒性的一步。

### 翻译

训练自动驾驶和导航系统需要大量且多样化的点云数据集，这些数据集应捕捉来自各种动态城市环境的复杂边缘情况场景。从真实世界的点云数据获取如此多样的场景具有挑战性，尤其是关键的边缘情况，这限制了系统的泛化能力和鲁棒性。当前方法依赖于在手工制作的3D虚拟环境中模拟点云数据，这些方法耗时、计算成本高，且往往无法完全捕捉真实场景的复杂性。为解决这些问题，本研究提出了一种新方法，通过基于语义掩码的指导来编辑真实的LiDAR扫描，生成新的合成LiDAR点云。我们结合范围图像投影和语义掩码条件实现基于扩散的生成。点云被转换为2D范围视图图像，作为中间表示来使用基于凸包的语义掩码进行语义编辑。这些掩码通过提供关于环境中物体尺寸、方向和位置的信息来指导生成过程，确保几何一致性和真实性。该方法在KITTI-360数据集上验证了能够生成高质量的LiDAR点云，能够产生复杂的边缘情况和动态场景。这为生成多样化的LiDAR数据提供了一种经济且可扩展的解决方案，是提高自动驾驶系统鲁棒性的一步。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何生成多样化的LiDAR点云数据，特别是那些在真实世界中难以获取的边缘案例。这个问题很重要，因为训练自动驾驶和导航系统需要大量复杂的场景数据，而真实世界数据获取困难，现有模拟方法耗时且无法完全捕捉真实场景复杂性，导致系统泛化能力和鲁棒性受限。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先考虑将3D点云转换为2D表示以便使用成熟的图像编辑技术，选择了range view表示。他们借鉴了扩散模型在图像生成领域的成功应用，引入语义掩码作为条件输入来控制生成过程。还使用了凸包增强几何一致性，并引入区域聚焦损失简化训练。该方法确实借鉴了range view表示、扩散模型和语义图像编辑等多个现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D LiDAR点云转换为2D range view图像，利用基于扩散模型的语义图像编辑技术进行编辑，最后转换回3D点云。流程包括：1)将点云转换为2D range view图像；2)构建语义掩码定义编辑区域；3)计算凸包增强几何一致性；4)使用扩散模型进行条件生成；5)将生成的range view图像转换回3D点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次不依赖LiDAR扫描模拟或预定义3D模型进行对象级别点云语义编辑；2)首次在range view图像上执行基于扩散的语义编辑；3)探索凸包掩码作为编辑定位手段；4)在KITTI-360数据集上建立语义编辑基准。相比之前工作，该方法通过编辑真实世界数据而非模拟来生成边缘案例，使用2D表示提高效率，结合语义掩码和扩散模型更好地控制生成物体的几何和位置信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于语义掩码引导的range view图像编辑方法，通过扩散模型高效生成多样化且高质量的LiDAR点云数据，特别是难以获取的边缘案例，为提高自动驾驶系统鲁棒性提供了成本效益高的解决方案。'}


### 论文摘要

Training autonomous driving and navigation systems requires large and diverse point cloud datasets that capture complex edge case scenarios from various dynamic urban settings. Acquiring such diverse scenarios from real-world point cloud data, especially for critical edge cases, is challenging, which restricts system generalization and robustness. Current methods rely on simulating point cloud data within handcrafted 3D virtual environments, which is time-consuming, computationally expensive, and often fails to fully capture the complexity of real-world scenes. To address some of these issues, this research proposes a novel approach that addresses the problem discussed by editing real-world LiDAR scans using semantic mask-based guidance to generate novel synthetic LiDAR point clouds. We incorporate range image projection and semantic mask conditioning to achieve diffusion-based generation. Point clouds are transformed to 2D range view images, which are used as an intermediate representation to enable semantic editing using convex hull-based semantic masks. These masks guide the generation process by providing information on the dimensions, orientations, and locations of objects in the real environment, ensuring geometric consistency and realism. This approach demonstrates high-quality LiDAR point cloud generation, capable of producing complex edge cases and dynamic scenes, as validated on the KITTI-360 dataset. This offers a cost-effective and scalable solution for generating diverse LiDAR data, a step toward improving the robustness of autonomous driving systems.

---

## 96. RoomPlanner: Explicit Layout Planner for Easier LLM-Driven 3D Room Generation

**论文链接:** [http://arxiv.org/abs/2511.17048v1](http://arxiv.org/abs/2511.17048v1)

**作者:** Wenzhuo Sun, Mingjian Liang, Wenxuan Song, Xuelian Cheng, Zongyuan Ge

**发布时间:** 2025-11-21

### GPT解析

### 总结

RoomPlanner是一个全自动的3D房间生成框架，只需简短文本输入即可创建逼真室内场景，无需手动布局设计或全景图像指导，能在30分钟内生成几何合理、视觉质量高且可编辑的3D室内场景。

### 背景

现有3D室内场景生成方法可能需要手动设计或全景图像指导，或存在生成速度慢、质量不高等问题。

### 目的

开发一个仅通过简短文本输入就能全自动生成高质量3D室内场景的框架，提高生成速度并保持场景的可编辑性。

### 方法

使用语言驱动的智能体规划器层次结构解析提示为详细场景描述，初始化3D点云，实现两种约束条件优化空间安排，提出AnyReach采样策略和ITFS策略优化3D高斯场景表示。

### 主要发现

RoomPlanner能生成几何合理的3D室内场景，在渲染速度和视觉质量上优于先前方法，同时保持场景的可编辑性，总生成时间控制在30分钟以内。

### 结论

RoomPlanner是首个全自动的3D房间生成框架，仅通过简短文本输入就能创建逼真室内场景，无需额外指导，在速度和质量方面均优于现有方法。

### 翻译

在本文中，我们提出了RoomPlanner，这是第一个全自动的3D房间生成框架，可以轻松创建逼真的室内场景，仅需简短文本作为输入。无需任何手动布局设计或全景图像指导，我们的框架可以生成明确的标准来合理布置空间。我们首先引入了一个语言驱动的智能体规划器的层次结构，该结构可以自动将简短且模糊的提示解析为详细的场景描述。这些描述包括每个对象和背景的原始空间和语义属性，然后用于初始化3D点云。为了在有限环境中定位对象，我们实现了两种排列约束，迭代优化空间安排，确保无碰撞且可访问的布局解决方案。在最终渲染阶段，我们提出了用于相机轨迹的AnyReach采样策略，以及ITFS策略，以有效优化粗略的3D高斯场景表示。这些方法帮助将总生成时间减少到30分钟以内。大量实验表明，我们的方法可以生成几何合理的3D室内场景，在渲染速度和视觉质量上都优于先前方法，同时保留了可编辑性。代码即将公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从简短文本描述自动生成逼真3D室内场景的问题。这个问题在现实中很重要，因为它能降低3D场景创建的成本和专业知识需求，适用于游戏开发、虚拟现实、电影制作和具身AI等多种应用；在研究中，它解决了现有方法（视觉引导和基于规则方法）的局限性，如缺乏对象级解耦、需要人工干预或物理布局不合理等问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有3D场景生成方法的局限性，设计了RoomPlanner框架。他们借鉴了多项现有工作：使用LLM进行场景布局规划（如LayoutGPT），利用约束求解器增强物理有效性（如Holodeck），采用物理引擎模拟物体动力学（如Physcene），以及使用Shap-E等文本到3D对象生成器创建初始3D资产。在此基础上，作者创新性地设计了分层LLM代理规划器、碰撞与可达性约束、AnyReach相机轨迹采样和区间时间步流采样等组件，解决了现有方法的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过'推理-接地、排列和优化'的管道，将复杂3D场景生成任务分解为可控子任务，结合视觉引导和基于规则方法的优点。整体流程：1)推理和接地阶段：使用分层LLM解析文本提示，生成详细场景描述并创建3D点云；2)布局排列阶段：应用碰撞约束检测对象重叠，使用可达性约束确保路径可行性；3)优化渲染阶段：采用AnyReach相机采样策略和区间时间步流采样技术，生成高质量3D场景。整个过程完全自动化，无需人工干预。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)分层LLM驱动代理规划器，能解析模糊文本为详细场景描述；2)碰撞和可达性双重约束，确保布局物理合理且可访问；3)AnyReach相机轨迹采样策略，高效捕获全局和局部视图；4)区间时间步流采样技术，优化场景质量和一致性；5)完全自动化管道，保持对象级解耦和可编辑性。相比之前工作，RoomPlanner结合了视觉引导方法的高质量渲染和基于规则方法的布局控制能力，生成速度更快（30分钟内），场景质量更高，且支持多样化编辑操作。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RoomPlanner通过创新的分层LLM规划、物理约束布局优化和高效渲染采样策略，实现了从简短文本快速生成高质量、可编辑的3D室内场景，显著提升了文本到3D场景生成的实用性和效率。'}


### 论文摘要

In this paper, we propose RoomPlanner, the first fully automatic 3D room generation framework for painlessly creating realistic indoor scenes with only short text as input. Without any manual layout design or panoramic image guidance, our framework can generate explicit layout criteria for rational spatial placement. We begin by introducing a hierarchical structure of language-driven agent planners that can automatically parse short and ambiguous prompts into detailed scene descriptions. These descriptions include raw spatial and semantic attributes for each object and the background, which are then used to initialize 3D point clouds. To position objects within bounded environments, we implement two arrangement constraints that iteratively optimize spatial arrangements, ensuring a collision-free and accessible layout solution. In the final rendering stage, we propose a novel AnyReach Sampling strategy for camera trajectory, along with the Interval Timestep Flow Sampling (ITFS) strategy, to efficiently optimize the coarse 3D Gaussian scene representation. These approaches help reduce the total generation time to under 30 minutes. Extensive experiments demonstrate that our method can produce geometrically rational 3D indoor scenes, surpassing prior approaches in both rendering speed and visual quality while preserving editability. The code will be available soon.

---

## 97. Adaptive Receiver-Side Scheduling for Smooth Interactive Delivery

**论文链接:** [http://arxiv.org/abs/2511.16902v1](http://arxiv.org/abs/2511.16902v1)

**作者:** Michael Luby

**发布时间:** 2025-11-21

**备注:** 25 pages, 6 figures, 1 table

### GPT解析

### 总结

本文提出了一种轻量级接收方调度方法，用于解决网络延迟变化导致的交互式应用播放质量问题，通过自适应估计路径延迟并不对称调整释放时间，实现平滑播放且不增加额外延迟。

### 背景

交互式应用如云游戏、XR流传输和实时推理依赖于数据对象以稳定节奏到达，但网络延迟变化和接收方恢复动态会导致明显的抖动、卡顿和不稳定播放问题。

### 目的

开发一种接收端调度机制，在数据恢复后规范化释放时间，保持播放流畅度而不增加额外延迟，且无需反馈或同步机制。

### 方法

调度器维护有效路径延迟的自适应估计，对延迟到达快速响应，对提前到达仅逐渐响应，使释放与最近的延迟峰值保持一致，完全在接收方时钟上运行。

### 主要发现

在云游戏工作负载测试中，该调度器几乎消除了所有大的抖动波动，产生紧密聚集的释放间隔，显著提高了播放的可见流畅度。

### 结论

接收方调度可模块化集成到各种传输堆栈如TCP、QUIC、WebRTC、UDP或RTP中，这些是未来部署的自然点，能有效解决网络延迟变化导致的播放质量问题。

### 翻译

交互式应用如云游戏、XR流传输和实时推理依赖于数据对象以稳定的节奏到达。实际上，即使传输层正确传递所有数据包，网络延迟变化和接收方的恢复动态也会扭曲这种节奏，从而产生明显的抖动、卡顿和不稳定的播放。我们提出了一种轻量级的接收方调度方法，在恢复后规范释放时间。该调度器维护有效路径延迟的自适应估计，并不对称地调整释放时间，对延迟到达快速响应，对提前到达仅逐渐响应。这种上包线行为使释放与最近的延迟峰值保持一致，以最小的额外延迟维持流畅播放。该调度器完全在接收方时钟上运行，无需反馈或同步。作为具体示例，我们将接收方调度集成到BitRipple Tunnel (BRT)覆盖层中，这是一个不更改底层传输协议的应用层软件系统。在BRT中，调度器作为独立模块，调节转发对象的交付时间。在云游戏工作负载上评估带有接收方调度的BRT显示，该调度器几乎消除了所有大的抖动波动，并产生紧密聚集的释放间隔，提高了可见的流畅度。更广泛的延迟改进来自于整个BRT覆盖层的行为。接收方调度也可以模块化地集成到传输堆栈中，如TCP、QUIC、WebRTC、UDP或RTP，这些是未来部署的自然点。


### 论文摘要

Interactive applications such as cloud gaming, XR streaming, and real-time inference depend on data objects arriving at a steady cadence. In practice, network delay variation and recovery dynamics at the receiver distort this cadence even when transports deliver all packets correctly, which produces visible jitter, stalls, and unstable playback.   We present a lightweight receiver-side scheduling approach that regularizes release timing after recovery. The scheduler maintains an adaptive estimate of effective path delay and adjusts release times asymmetrically, responding quickly to late arrivals and only gradually to early ones. This upper-envelope behavior keeps release aligned with recent delay peaks and maintains smooth playback with minimal added latency. The scheduler runs entirely on the receiver clock and requires no feedback or synchronization.   As a concrete example, we integrate receiver-side scheduling into the BitRipple Tunnel (BRT) overlay, an application-layer software system that forwards traffic without altering the underlying transport protocol. Within BRT, the scheduler functions as an independent module that regulates delivery timing for forwarded objects.   Evaluating BRT with receiver-side scheduling on a cloud-gaming workload shows that the scheduler removes virtually all large jitter excursions and yields tightly clustered release intervals that improve visible smoothness. Broader latency improvements arise from the behavior of the full BRT overlay. Receiver-side scheduling can also be integrated modularly into transport stacks such as TCP, QUIC, WebRTC, UDP, or RTP, which are natural deployment points for future work.

---

## 98. The Joint Gromov Wasserstein Objective for Multiple Object Matching

**论文链接:** [http://arxiv.org/abs/2511.16868v1](http://arxiv.org/abs/2511.16868v1)

**作者:** Aryan Tajmir Riahi, Khanh Dao Duc

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种新的联合Gromov-Wasserstein (JGW)目标，扩展了传统GW框架以实现对象集合间的同时匹配，解决了传统GW距离局限于单对单匹配的问题。

### 背景

Gromov-Wasserstein (GW)距离是度量空间中对象匹配的有力工具，但其传统形式仅限于单对单的对象匹配，限制了在需要多对一或多对多匹配场景中的应用。

### 目的

引入联合Gromov-Wasserstein (JGW)目标，扩展原始GW框架，实现对象集合之间的同时匹配。

### 方法

提出JGW目标，提供非负不相似性度量，能够识别部分同构的m-空间分布，具有点采样收敛性；通过调整最优传输中的传统算法（包括熵正则化）为点云对象表示制定并求解目标。

### 主要发现

与其他GW变体在部分匹配上的基准测试表明，该方法在准确性和计算效率方面表现更优；在合成和真实数据集上的实验证明了其在多种形状匹配中的有效性，适用于几何形状和生物分子复合体的匹配。

### 结论

该方法为解决跨领域（包括计算机图形学和结构生物学）的复杂匹配问题提供了有前景的应用。

### 翻译

Gromov-Wasserstein (GW)距离是度量空间中对象匹配的有力工具。然而，其传统形式局限于单对单的对象匹配，限制了在需要多对一或多对多对象匹配场景和应用中的效用。在本文中，我们引入了联合Gromov-Wasserstein (JGW)目标，并扩展了原始GW框架，实现了对象集合之间的同时匹配。我们的公式提供了一个非负不相似性度量，能够识别部分同构的m-空间分布，具有点采样收敛性。我们还展示了该目标可以通过调整最优传输中的传统算法（包括熵正则化）为点云对象表示制定和求解。与其他GW变体在部分匹配上的基准测试表明，我们的方法在准确性和计算效率方面表现更优，而在合成和真实数据集上的实验展示了其在多种形状匹配中的有效性，包括几何形状和生物分子复合体，这表明该方法在解决跨领域（包括计算机图形学和结构生物学）的复杂匹配问题方面具有应用前景。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是多对象匹配问题，特别是传统的Gromov-Wasserstein距离方法只能处理单对象之间的成对匹配，而无法处理多对一或多对多的对象匹配场景。这个问题在现实中非常重要，因为在蛋白质模型构建、合并部分3D扫描、解决2D和3D拼图等多个应用中，需要同时匹配多个对象。现有的顺序成对匹配策略会导致错误累积和计算成本增加，而其他多对多方法（如Z-Gromov-Wasserstein）又要求数据具有特定结构，限制了它们的适用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了传统GW距离的局限性，然后通过扩展度量测度空间(mm-space)的概念到分布的mm空间，引入了'嵌入'的概念将多个mm空间组合到一个统一的度量空间中。基于此设计了联合Gromov-Wasserstein目标函数。作者确实借鉴了现有工作：基于传统GW理论框架，利用最优传输理论中的熵正则化技术，参考了部分匹配方法（如mPGW、PGW、UGW）的设计思路，并借鉴了Z-Gromov-Wasserstein处理多对象匹配的部分思想，但避免了其对特定数据结构的要求。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将传统的单对象匹配扩展到多对象集合的匹配，通过'嵌入'将多个度量测度空间组合到一个统一的度量空间中，并定义联合Gromov-Wasserstein距离来衡量对象集合间的不相似性。整体流程：1)准备输入，为每个mm空间定义距离矩阵和分布；2)构造嵌入，将各mm空间嵌入更大的度量空间；3)计算嵌入空间距离矩阵并构建块矩阵；4)定义JGW目标函数并应用熵正则化；5)使用类似Sinkhorn的迭代算法求解最优传输计划；6)输出结果表示对象间的对应关系。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出联合Gromov-Wasserstein目标，专门处理多对象集合匹配；2)证明了JGW在部分同构和点采样收敛方面的理论性质；3)设计不要求特定数据结构的灵活框架；4)提出基于熵正则化的高效算法。相比之前工作：与传统GW不同，JGW能处理多对象而非单对象；与部分匹配方法相比，JGW可处理多对多情况且结构保持更好；与Z-Gromov-Wasserstein相比，JGW不要求特定数据结构；实验显示JGW在准确性和效率上均优于其他方法，如生物分子对齐快11倍以上。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了联合Gromov-Wasserstein目标，将传统的单对象匹配方法扩展为能够同时匹配多个对象集合的高效框架，显著提升了在部分匹配、形状匹配和生物分子对齐等任务中的性能。'}


### 论文摘要

The Gromov-Wasserstein (GW) distance serves as a powerful tool for matching objects in metric spaces. However, its traditional formulation is constrained to pairwise matching between single objects, limiting its utility in scenarios and applications requiring multiple-to-one or multiple-to-multiple object matching. In this paper, we introduce the Joint Gromov-Wasserstein (JGW) objective and extend the original framework of GW to enable simultaneous matching between collections of objects. Our formulation provides a non-negative dissimilarity measure that identifies partially isomorphic distributions of mm-spaces, with point sampling convergence. We also show that the objective can be formulated and solved for point cloud object representations by adapting traditional algorithms in Optimal Transport, including entropic regularization. Our benchmarking with other variants of GW for partial matching indicates superior performance in accuracy and computational efficiency of our method, while experiments on both synthetic and real-world datasets show its effectiveness for multiple shape matching, including geometric shapes and biomolecular complexes, suggesting promising applications for solving complex matching problems across diverse domains, including computer graphics and structural biology.

---

## 99. Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.15311v2](http://arxiv.org/abs/2511.15311v2)

**作者:** Mehran Tamjidi, Hamidreza Dastmalchi, Mohammadreza Alimoradijazi, Ali Cheraghian, Aijun An, Morteza Saberi

**发布时间:** 2025-11-19

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种名为Uni-Adapter的测试时适应策略，用于提升3D视觉语言基础模型在噪声、不完整或分布不同的数据上的表现，无需重新训练即可实现显著性能提升。

### 背景

3D视觉语言基础模型在开放世界点云处理任务中表现出强大的泛化和零样本识别能力，但在实际场景中，当数据存在噪声、不完整或与训练数据分布不同时，这些模型往往表现不佳。

### 目的

开发一种无需训练的在线测试时适应策略，解决3D视觉语言基础模型在实际应用场景中的性能下降问题，特别是在面对分布偏移时的适应能力。

### 方法

提出Uni-Adapter策略，基于动态原型学习：1)定义3D缓存存储类特定聚类中心作为动态原型，持续更新以捕捉异构数据分布中的类内变异性；2)通过相似度评分实现基于缓存的logit计算；3)引入基于图的标签平滑模块，强制相似原型间保持标签一致性；4)使用熵加权聚合统一原始模型和改进缓存的预测。

### 主要发现

Uni-Adapter无需重新训练即可有效缓解分布偏移问题；在多个3D基准测试上实现了最先进性能，在ModelNet-40C上提升10.55%，在ScanObjectNN-C上提升8.26%，在ShapeNet-C上提升4.49%。

### 结论

Uni-Adapter是一种有效的测试时适应策略，能够显著提升3D视觉语言基础模型在实际场景中的鲁棒性和性能，为处理分布偏移问题提供了新思路。

### 翻译

3D视觉语言基础模型在开放世界点云处理任务中展现出强大的泛化和零样本识别能力。然而，在实际场景中，当数据存在噪声、不完整或与训练数据分布不同时，这些模型往往表现不佳。为此，我们提出了Uni-Adapter，一种基于动态原型学习的3D视觉语言基础模型的无需训练的在线测试时适应策略。我们定义了一个3D缓存来存储类特定的聚类中心作为原型，这些原型被持续更新以捕捉异构数据分布中的类内变异性。这些动态原型通过相似度评分作为基于缓存的logit计算的锚点。同时，基于图的标签平滑模块捕获原型间的相似性，以强制相似原型间保持标签一致性。最后，我们使用熵加权聚合统一原始3D视觉语言基础模型和改进的3D缓存的预测，实现可靠的适应。无需重新训练，Uni-Adapter有效缓解了分布偏移问题，在不同的3D基准测试上实现了最先进的性能，在ModelNet-40C上比源3D视觉语言基础模型提高了10.55%，在ScanObjectNN-C上提高了8.26%，在ShapeNet-C上提高了4.49%。项目页面：https://mehran-tam.github.io/Uni-Adapter

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D视觉语言基础模型(VLFMs)在实际场景中表现不佳的问题，特别是当数据有噪声、不完整或与训练数据分布不同时模型性能下降的问题。这个问题在现实中很重要，因为实际应用中的3D点云数据常受传感器限制和环境因素影响，存在噪声、稀疏性和低分辨率等问题，研究这个问题有助于提高3D视觉模型在真实世界环境中的鲁棒性和泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有测试时适应(TTA)方法的局限性，特别是基于缓存的策略在3D数据上无法充分捕获分布多样性的问题。现有方法通常依赖高置信度样本作为原型，但这种方法无法代表3D数据的全部分布模式。作者借鉴了现有的缓存策略和在线聚类技术，但改进了原型选择机制，使用基于聚类的动态原型而非仅基于高置信度的样本。此外，作者还引入了图平滑技术来处理噪声伪标签问题，以及基于熵的融合策略来结合不同模型的预测。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用动态原型学习来改进3D视觉语言基础模型的测试时适应能力，通过维护一个3D缓存来存储类特定的聚类中心作为原型，这些原型会不断更新以捕获异构数据分布中的类内变化。整体实现流程包括：1)在线原型模块：采用在线聚类策略动态更新类特定原型；2)原型重新分配模块：通过基于图的标签平滑捕获原型间相似性，强制相似原型间标签一致性；3)缓存计算：基于相似性评分计算基于缓存的logit；4)基于熵的融合：使用熵加权聚合统一原始模型和缓存模型的预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于聚类的缓存策略：使用多个聚类中心作为每个类的原型，捕获类内变异性；2)基于图的标签平滑：利用原型间相似性来平滑噪声伪标签；3)基于熵的融合策略：自适应地结合原始模型和缓存模型的预测。相比之前的工作，特别是Point-Cache等基于高置信度样本的缓存方法，Uni-Adapter能更好地捕获3D数据中的分布多样性，因为它不局限于高置信度样本，而是使用聚类来代表数据分布的不同模式，从而实现更鲁棒的适应能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Uni-Adapter通过引入基于聚类的动态原型学习和基于图的标签平滑，实现了无需训练的在线测试时适应，显著提高了3D视觉语言基础模型在噪声和不完整数据上的性能。'}


### 论文摘要

3D Vision-Language Foundation Models (VLFMs) have shown strong generalization and zero-shot recognition capabilities in open-world point cloud processing tasks. However, these models often underperform in practical scenarios where data are noisy, incomplete, or drawn from a different distribution than the training data. To address this, we propose Uni-Adapter, a novel training-free online test-time adaptation (TTA) strategy for 3D VLFMs based on dynamic prototype learning. We define a 3D cache to store class-specific cluster centers as prototypes, which are continuously updated to capture intra-class variability in heterogeneous data distributions. These dynamic prototypes serve as anchors for cache-based logit computation via similarity scoring. Simultaneously, a graph-based label smoothing module captures inter-prototype similarities to enforce label consistency among similar prototypes. Finally, we unify predictions from the original 3D VLFM and the refined 3D cache using entropy-weighted aggregation for reliable adaptation. Without retraining, Uni-Adapter effectively mitigates distribution shifts, achieving state-of-the-art performance on diverse 3D benchmarks over different 3D VLFMs, improving ModelNet-40C by 10.55%, ScanObjectNN-C by 8.26%, and ShapeNet-C by 4.49% over the source 3D VLFMs. Project page: https://mehran-tam.github.io/Uni-Adapter

---

## 100. Distributed Switching Model Predictive Control Meets Koopman Operator for Dynamic Obstacle Avoidance

**论文链接:** [http://arxiv.org/abs/2511.17186v1](http://arxiv.org/abs/2511.17186v1)

**作者:** Ali Azarbahram, Chrystian Pool Yuca Huanca, Gian Paolo Incremona, Patrizio Colaneri

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种基于Koopman增强的分布式切换模型预测控制框架，用于四旋翼无人机在动态环境中安全可扩展的导航，实现实时无碰撞协调。

### 背景

四旋翼无人机在动态环境中（有移动障碍物）的安全导航面临挑战，需要实时、无碰撞的协调控制方法。

### 目的

开发一种能够实现无人机在动态环境中安全导航的方法，特别关注实时性和可扩展性。

### 方法

结合切换运动模式和数据驱动预测，使用局部Koopman算子将非线性障碍动力学近似为基于在线测量的线性模型，并将预测嵌入分布式SMPC结构中，使每架无人机能够使用本地和基于集群的信息做出自主决策。

### 主要发现

模拟结果表明实现了可靠的编队控制和实时障碍避障，该框架对智能和协作移动系统具有广泛的适用性。

### 结论

所提出的方法为无人机在动态环境中的安全导航提供了一种有效的解决方案，特别适用于表面交通应用场景，包括协调车流、与行人或骑行者共享基础设施以及城市无人机交通。

### 翻译

本文介绍了一种用于四旋翼无人机在动态环境中（有移动障碍物）进行安全和可扩展导航的Koopman增强型分布式切换模型预测控制框架。所提出的方法结合了切换运动模式和数据驱动预测，以实现实时、无碰撞的协调。局部Koopman算子基于在线测量将非线性障碍动力学近似为线性模型，从而实现准确的轨迹预测。这些预测被嵌入到分布式SMPC结构中，每架无人机使用本地和基于集群的信息做出自主决策。这种计算高效的架构在表面交通应用中特别有前景，包括协调车辆流、与行人或骑行者共享基础设施以及城市无人机交通。模拟结果表明实现了可靠的编队控制和实时障碍避障，突显了该框架对智能和协作移动系统的广泛相关性。


### 论文摘要

This paper introduces a Koopman-enhanced distributed switched model predictive control (SMPC) framework for safe and scalable navigation of quadrotor unmanned aerial vehicles (UAVs) in dynamic environments with moving obstacles. The proposed method integrates switched motion modes and data-driven prediction to enable real-time, collision-free coordination. A localized Koopman operator approximates nonlinear obstacle dynamics as linear models based on online measurements, enabling accurate trajectory forecasting. These predictions are embedded into a distributed SMPC structure, where each UAV makes autonomous decisions using local and cluster-based information. This computationally efficient architecture is particularly promising for applications in surface transportation, including coordinated vehicle flows, shared infrastructure with pedestrians or cyclists, and urban UAV traffic. Simulation results demonstrate reliable formation control and real-time obstacle avoidance, highlighting the frameworks broad relevance for intelligent and cooperative mobility systems.

---

## 101. Masked-and-Reordered Self-Supervision for Reinforcement Learning from Verifiable Rewards

**论文链接:** [http://arxiv.org/abs/2511.17473v1](http://arxiv.org/abs/2511.17473v1)

**作者:** Zhen Wang, Zhifeng Gao, Guolin Ke

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究提出了一种名为MR-RLVR的新方法，通过过程级自监督奖励来增强大型语言模型的数学推理能力，特别是在只有最终答案可验证的数学任务中。该方法结合了掩码后填充和步骤重排序技术，在两个阶段进行训练：首先在数学计算和证明数据上进行自监督训练，然后在只有结果可验证的数据集上进行RLVR微调。实验表明，该方法相比原始RLVR在多个指标上实现了显著提升。

### 背景

测试时扩展已被证明能提高大型语言模型的数学推理能力，但对于许多数学语料库特别是定理证明，RLVR的可扩展性有限，因为中间推理很重要而最终答案难以直接可靠地验证。同时，基于标记级的监督微调往往会退化为死记硬背，而不是诱导更长的思维链。

### 目的

解决RLVR在数学推理任务中可扩展性有限的问题，特别是在只有最终答案可验证的情况下，通过引入过程感知的自监督信号来增强模型性能。

### 方法

提出MR-RLVR方法，灵感来自BERT的自监督任务，通过'掩码后填充'和'步骤重排序'构建过程级自监督奖励，从中间推理中提取可学习信号。训练流程包括两个阶段：首先在采样的数学计算和证明数据上进行自监督训练，然后在只有结果可验证的数学计算数据集上进行RLVR微调。在Qwen2.5-3B和DeepSeek-R1-Distill-Qwen-1.5B模型上实现，并在AIME24、AIME25、AMC23和MATH500数据集上评估。

### 主要发现

在固定的采样和解码预算下，MR-RLVR相比原始RLVR实现了平均相对增益：Pass@1提升9.86%，Pass@5提升5.27%，Pass@8提升4.00%。

### 结论

融合过程感知的自监督信号可以有效增强RLVR在只有结果可验证设置中的可扩展性和性能。

### 翻译

测试时扩展已被证明能显著提高大型语言模型的数学推理能力。然而，对于大量数学语料库，特别是定理证明，RLVR的可扩展性有限：中间推理至关重要，而最终答案难以直接可靠地验证。同时，基于标记级的监督微调往往会退化为死记硬背，而不是诱导更长的思维链。受BERT自监督任务的启发，我们提出了MR-RLVR（掩码和重排序RLVR），通过'掩码后填充'和'步骤重排序'构建过程级自监督奖励，从中间推理中提取可学习信号。我们的训练流程包括两个阶段：首先在采样的数学计算和证明数据上进行自监督训练；然后在只有结果可验证的数学计算数据集上进行RLVR微调。我们在Qwen2.5-3B和DeepSeek-R1-Distill-Qwen-1.5B上实现了MR-RLVR，并在AIME24、AIME25、AMC23和MATH500上进行了评估。在固定的采样和解码预算下，MR-RLVR相比原始RLVR实现了平均相对增益：Pass@1提升9.86%，Pass@5提升5.27%，Pass@8提升4.00%。这些结果表明，在只有结果可验证的设置中，融入过程感知的自监督信号可以有效增强RLVR的可扩展性和性能。


### 论文摘要

Test-time scaling has been shown to substantially improve large language models' (LLMs) mathematical reasoning. However, for a large portion of mathematical corpora, especially theorem proving, RLVR's scalability is limited: intermediate reasoning is crucial, while final answers are difficult to directly and reliably verify. Meanwhile, token-level SFT often degenerates into rote memorization rather than inducing longer chains of thought. Inspired by BERT's self-supervised tasks, we propose MR-RLVR (Masked-and-Reordered RLVR), which constructs process-level self-supervised rewards via "masked-then-fill" and "step reordering" to extract learnable signals from intermediate reasoning. Our training pipeline comprises two stages: we first perform self-supervised training on sampled mathematical calculation and proof data; we then conduct RLVR fine-tuning on mathematical calculation datasets where only outcomes are verifiable. We implement MR-RLVR on Qwen2.5-3B and DeepSeek-R1-Distill-Qwen-1.5B, and evaluate on AIME24, AIME25, AMC23, and MATH500. Under a fixed sampling and decoding budget, MR-RLVR achieves average relative gains over the original RLVR of +9.86% Pass@1, +5.27% Pass@5, and +4.00% Pass@8. These results indicate that incorporating process-aware self-supervised signals can effectively enhance RLVR's scalability and performance in only outcome-verifiable settings.

---

## 102. Self-supervised denoising of raw tomography detector data for improved image reconstruction

**论文链接:** [http://arxiv.org/abs/2511.17312v1](http://arxiv.org/abs/2511.17312v1)

**作者:** Israt Jahan Tulin, Sebastian Starke, Dominic Windisch, André Bieberle, Peter Steinbach

**发布时间:** 2025-11-21

### GPT解析

### 总结

研究比较了两种基于自监督深度学习的原始探测器数据去噪方法，发现它们能有效提高信噪比并改善重建图像质量，优于非基于学习的方法。

### 背景

超快电子束X射线计算机断层扫描由于测量时间短会产生噪声数据，导致重建伪影并限制整体图像质量。

### 目的

研究并比较两种基于自监督深度学习的原始探测器数据去噪方法，并与非基于学习的去噪方法进行对比。

### 方法

采用两种自监督深度学习方法和一种非基于学习的去噪方法对原始探测器数据进行处理。

### 主要发现

基于深度学习的方法能够提高探测器数据的信噪比，并一致改善重建图像质量，优于非基于学习的方法。

### 结论

基于深度学习的去噪方法在超快电子束X射线计算机断层扫描数据去噪方面表现更好。

### 翻译

超快电子束X射线计算机断层扫描由于测量时间短会产生噪声数据，导致重建伪影并限制整体图像质量。为了解决这些问题，研究并比较了两种用于原始探测器数据去噪的自监督深度学习方法，并与一种非基于学习的去噪方法进行了对比。我们发现，应用基于深度学习的方法能够提高探测器数据的信噪比，并一致改善重建图像质量，优于非基于学习的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决超快电子束X射线计算机断层扫描产生的噪声数据问题。由于测量时间短，导致重建图像出现伪影并降低质量。这个问题很重要，因为UFXCT是研究高动态两相流的重要工具，能提供每秒8000个高分辨率图像，但快速成像导致每个投影获取的光子数少，使图像噪声严重，限制了其在工业设备（如化学反应器、热交换器）中的应用。传统降噪方法虽能减少噪声但会降低图像清晰度，造成空间分辨率损失。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到图像重建过程涉及多个非线性相关步骤，改变任何参数都可能使训练好的降噪网络失效。因此，他们选择在图像处理前对原始正弦图数据进行降噪。该方法借鉴了三种现有工作：Noise2Void（N2V）自监督深度学习方法，只需嘈杂数据训练；N2V2作为N2V的改进版，减少棋盘格伪影；以及传统的非学习方法BM3D。作者创新性地将这些方法应用于CT探测器数据的降噪，并比较它们在正弦图和重建图像空间中的表现。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是在图像重建前对原始探测器数据进行降噪，使用自监督深度学习方法捕获系统固有噪声特征。流程包括：1) 从两个探测器平面和三个位置收集数据，形成12,000个正弦图样本；2) 将数据分为训练/验证集和测试集；3) 通过平均多个样本创建参考'地面真实值'；4) 使用五折交叉验证策略；5) 在嘈杂正弦图上训练N2V和N2V2模型；6) 使用PSNR指标在正弦图和重建图像空间评估性能；7) 比较深度学习方法与传统BM3D方法的性能差异。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 将自监督深度学习方法应用于原始CT探测器数据降噪；2) 在重建前而非重建后进行降噪；3) 开发能捕获系统固有噪声特征的模型。相比之前工作，本文发现：N2V表现出最稳定的性能提升，重建图像PSNR提高约4.1分贝；N2V2虽在某些情况下表现良好但不稳定；传统BM3D方法在CT数据上表现不佳，反而降低了PSNR；深度学习方法在重建空间表现优于正弦图空间，因为重建结合了多投影信息；传统基于补丁的滤波方法不适合CT数据的特定噪声特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文证明了自监督深度学习方法（特别是N2V）能有效降噪超快X射线CT的原始探测器数据，显著提高重建图像质量，且性能优于传统非学习方法。'}


### 论文摘要

Ultrafast electron beam X-ray computed tomography produces noisy data due to short measurement times, causing reconstruction artifacts and limiting overall image quality. To counteract these issues, two self-supervised deep learning methods for denoising of raw detector data were investigated and compared against a non-learning based denoising method. We found that the application of the deep-learning-based methods was able to enhance signal-to-noise ratios in the detector data and also led to consistent improvements of the reconstructed images, outperforming the non-learning based method.

---

## 103. SAVeD: Semantic Aware Version Discovery

**论文链接:** [http://arxiv.org/abs/2511.17298v1](http://arxiv.org/abs/2511.17298v1)

**作者:** Artem Frenk, Roee Shraga

**发布时间:** 2025-11-21

**备注:** 11 pages, 6 figures

### GPT解析

### 总结

SAVeD是一种基于对比学习的框架，用于识别结构化数据集的版本，无需依赖元数据、标签或集成假设，解决了数据科学中因难以识别相似工作或数据集转换导致的重复劳动问题。

### 背景

数据科学中常见的一个挑战是，由于难以识别相似工作或对数据集进行转换，导致重复劳动。

### 目的

开发一种能够识别结构化数据集版本的框架，无需依赖元数据、标签或集成假设。

### 方法

SAVeD采用改进的SimCLR流程，通过随机转换（如行删除、编码扰动）生成增强的表视图，使用自定义transformer编码器嵌入这些视图，并在潜在空间中进行对比优化语义相似性，学习最小化同一数据集增强视图间的距离，最大化无关表间的距离。

### 主要发现

实验表明，SAVeD在五个典型数据集上实现了显著提升，在完全看不见的表上达到更高准确度，显著提高分离度分数，确认了其区分语义改变版本的能力；与未训练基线和Starmie等先前方法相比，自定义编码器实现了具有竞争力或更优的结果。

### 结论

SAVeD框架有效解决了数据集版本识别问题，无需元数据或标签，能够准确区分语义改变的数据集版本，为数据科学中的重复劳动问题提供了解决方案。

### 翻译

我们的工作引入了SAVeD（语义感知版本检测），一种基于对比学习的框架，用于识别结构化数据集的版本，而不依赖元数据、标签或基于集成的假设。SAVeD解决了数据科学中因难以识别相似工作或对数据集进行转换而导致的重复劳动这一常见挑战。SAVeD采用改进的SimCLR流程，通过随机转换（如行删除、编码扰动）生成增强的表视图。这些视图通过自定义的transformer编码器嵌入，并在潜在空间中进行对比以优化语义相似性。我们的模型学习最小化同一数据集增强视图之间的距离，最大化无关表之间的距离。我们使用验证准确度和分离度来评估性能，分别定义为在保留集上正确分类版本/非版本对的比例，以及版本表和非版本表平均相似度之间的差异（由基准定义，不提供给模型）。我们的实验来自Semantic Versioning in Databases基准测试中的五个典型数据集，展示了训练后的显著提升。SAVeD在完全看不见的表上实现了显著更高的准确度，并显著提高了分离度分数，确认了其区分语义改变版本的能力。与未训练的基线和之前最先进的数据集发现方法（如Starmie）相比，我们的自定义编码器实现了具有竞争力或更优的结果。


### 论文摘要

Our work introduces SAVeD (Semantically Aware Version Detection), a contrastive learning-based framework for identifying versions of structured datasets without relying on metadata, labels, or integration-based assumptions. SAVeD addresses a common challenge in data science of repeated labor due to a difficulty of similar work or transformations on datasets. SAVeD employs a modified SimCLR pipeline, generating augmented table views through random transformations (e.g., row deletion, encoding perturbations). These views are embedded via a custom transformer encoder and contrasted in latent space to optimize semantic similarity. Our model learns to minimize distances between augmented views of the same dataset and maximize those between unrelated tables. We evaluate performance using validation accuracy and separation, defined respectively as the proportion of correctly classified version/non-version pairs on a hold-out set, and the difference between average similarities of versioned and non-versioned tables (defined by a benchmark, and not provided to the model). Our experiments span five canonical datasets from the Semantic Versioning in Databases Benchmark, and demonstrate substantial gains post-training. SAVeD achieves significantly higher accuracy on completely unseen tables in, and a significant boost in separation scores, confirming its capability to distinguish semantically altered versions. Compared to untrained baselines and prior state-of-the-art dataset-discovery methods like Starmie, our custom encoder achieves competitive or superior results.

---

## 104. AutoGraphAD: A novel approach using Variational Graph Autoencoders for anomalous network flow detection

**论文链接:** [http://arxiv.org/abs/2511.17113v1](http://arxiv.org/abs/2511.17113v1)

**作者:** Georgios Anyfantis, Pere Barlet-Ros

**发布时间:** 2025-11-21

**备注:** 11 pages, 9 figures

### GPT解析

### 总结

AutoGraphAD是一种基于异构变分图自编码器的无监督异常检测方法，在不需要标记数据的情况下实现了高效的网络入侵检测，性能与现有方法相当或更好，且训练和推理速度显著提升。

### 背景

网络入侵检测系统(NIDS)是检测网络攻击的重要工具，现有基于监督式机器学习的方法需要准确标记的数据集，获取成本高，且公共数据集存在攻击类型有限/过时和数据标记错误的问题。

### 目的

减少对标记数据的依赖，提出一种新的无监督异常检测方法来检测网络入侵。

### 方法

AutoGraphAD在由连接节点和IP节点组成的异构图上运行，捕获时间窗口内的网络活动；使用无监督和对比学习进行模型训练，不依赖任何标记数据；通过重建、结构损失和KL散度的加权组合形成异常分数用于异常检测。

### 主要发现

AutoGraphAD取得了与之前无监督方法(如Anomal-E)相同或更好的结果，不需要昂贵的下游异常检测器；训练速度提高了约1.18个数量级，推理速度提高了约1.03个数量级。

### 结论

AutoGraphAD在操作部署方面具有显著优势，是一种高效的网络入侵检测解决方案。

### 翻译

网络入侵检测系统(NIDS)是检测网络攻击和入侵的重要工具。虽然大量研究已经探索了使用监督式机器学习进行攻击检测和分类，但这些方法需要准确标记的数据集，获取成本很高。此外，现有的公共数据集存在攻击类型有限/过时，并且许多数据集存在标记错误的问题。为了减少对标记数据的依赖，我们提出了AutoGraphAD，这是一种基于异构变分图自编码器的新型无监督异常检测方法。AutoGraphAD在异构图上运行，这些图由连接节点和IP节点组成，捕获时间窗口内的网络活动。该模型使用无监督和对比学习进行训练，不依赖任何标记数据。然后，重建、结构损失和KL散度被加权组合成一个异常分数，用于异常检测。总体而言，AutoGraphAD取得了与之前无监督方法(如Anomal-E)相同，在某些情况下更好的结果，但不需要昂贵的下游异常检测器。因此，AutoGraphAD实现了约1.18个数量级更快的训练速度和1.03个数量级更快的推理速度，这对于操作部署来说是一个显著优势。


### 论文摘要

Network Intrusion Detection Systems (NIDS) are essential tools for detecting network attacks and intrusions. While extensive research has explored the use of supervised Machine Learning for attack detection and characterisation, these methods require accurately labelled datasets, which are very costly to obtain. Moreover, existing public datasets have limited and/or outdated attacks, and many of them suffer from mislabelled data. To reduce the reliance on labelled data, we propose AutoGraphAD, a novel unsupervised anomaly detection approach based on a Heterogeneous Variational Graph Autoencoder. AutoGraphAD operates on heterogeneous graphs, made from connection and IP nodes that capture network activity within a time window. The model is trained using unsupervised and contrastive learning, without relying on any labelled data. The reconstruction, structural loss, and KL divergence are then weighted and combined in an anomaly score that is then used for anomaly detection. Overall, AutoGraphAD yields the same, and in some cases better, results than previous unsupervised approaches, such as Anomal-E, but without requiring costly downstream anomaly detectors. As a result, AutoGraphAD achieves around 1.18 orders of magnitude faster training and 1.03 orders of magnitude faster inference, which represents a significant advantage for operational deployment.

---

## 105. Bridging Visual Affective Gap: Borrowing Textual Knowledge by Learning from Noisy Image-Text Pairs

**论文链接:** [http://arxiv.org/abs/2511.17103v1](http://arxiv.org/abs/2511.17103v1)

**作者:** Daiqing Wu, Dongbao Yang, Yu Zhou, Can Ma

**发布时间:** 2025-11-21

**DOI:** 10.1145/3664647.3680875

**备注:** Accepted by ACM MM 2024

### GPT解析

### 总结

这篇论文提出了一种名为PACL（分区自适应对比学习）的方法，通过借鉴预训练文本模型的知识来增强预训练视觉模型的情感感知能力，以解决视觉情感识别中的'情感鸿沟'问题，从而提高下游情感相关任务的性能。

### 背景

视觉情感识别是一个长期存在的领域，随着深度神经网络的发展而受到越来越多的关注。尽管最近的研究通过利用预训练视觉模型中嵌入的知识取得了显著进展，但事实级特征与情感类别之间缺乏直接联系，即所谓的'情感鸿沟'，限制了预训练知识在视觉情感识别任务中的应用。相比之下，文本模态中的明确情感表达和高信息密度消除了'情感鸿沟'。

### 目的

研究目的是通过借鉴预训练文本模型的知识来增强预训练视觉模型的情感感知能力，解决视觉情感识别中的'情感鸿沟'问题，从而提高下游情感相关任务的性能。

### 方法

研究提出了分区自适应对比学习（PACL）方法，专注于嘈杂社交媒体数据中图像和文本之间的事实和情感联系。该方法分离不同类型的样本，并为每种类型设计不同的对比学习策略。通过动态构建负样本和正样本对，充分利用嘈杂样本的潜力。

### 主要发现

通过综合实验证明，弥合'情感鸿沟'显著提高了各种预训练视觉模型在下游情感相关任务中的性能。

### 结论

通过借鉴文本模态的知识来解决视觉情感识别中的'情感鸿沟'问题是有效的。PACL方法能够充分利用嘈杂社交媒体数据中的信息，提高预训练视觉模型在情感相关任务中的表现，为视觉情感识别领域提供了新的思路。

### 翻译

视觉情感识别是一个长期存在的领域，随着深度神经网络的发展而受到越来越多的关注。尽管最近的研究通过利用预训练视觉模型中嵌入的知识取得了显著进展，但事实级特征与情感类别之间缺乏直接联系，即所谓的'情感鸿沟'，限制了预训练知识在视觉情感识别任务中的应用。相比之下，文本模态中的明确情感表达和高信息密度消除了'情感鸿沟'。因此，我们提出借鉴预训练文本模型的知识来增强预训练视觉模型的情感感知能力。我们专注于嘈杂社交媒体数据中图像和文本之间的事实和情感联系，并提出了分区自适应对比学习（PACL）来利用这些联系。具体来说，我们成功分离了不同类型的样本，并为每种类型设计了不同的对比学习策略。通过动态构建负样本和正样本对，我们充分利用了嘈杂样本的潜力。通过综合实验，我们证明弥合'情感鸿沟'显著提高了各种预训练视觉模型在下游情感相关任务中的性能。我们的代码已在https://github.com/wdqqdw/PACL上发布。


### 论文摘要

Visual emotion recognition (VER) is a longstanding field that has garnered increasing attention with the advancement of deep neural networks. Although recent studies have achieved notable improvements by leveraging the knowledge embedded within pre-trained visual models, the lack of direct association between factual-level features and emotional categories, called the "affective gap", limits the applicability of pre-training knowledge for VER tasks. On the contrary, the explicit emotional expression and high information density in textual modality eliminate the "affective gap". Therefore, we propose borrowing the knowledge from the pre-trained textual model to enhance the emotional perception of pre-trained visual models. We focus on the factual and emotional connections between images and texts in noisy social media data, and propose Partitioned Adaptive Contrastive Learning (PACL) to leverage these connections. Specifically, we manage to separate different types of samples and devise distinct contrastive learning strategies for each type. By dynamically constructing negative and positive pairs, we fully exploit the potential of noisy samples. Through comprehensive experiments, we demonstrate that bridging the "affective gap" significantly improves the performance of various pre-trained visual models in downstream emotion-related tasks. Our code is released on https://github.com/wdqqdw/PACL.

---

## 106. The Finer the Better: Towards Granular-aware Open-set Domain Generalization

**论文链接:** [http://arxiv.org/abs/2511.16979v1](http://arxiv.org/abs/2511.16979v1)

**作者:** Yunyun Wang, Zheng Duan, Xinyue Liao, Ke-Jia Chen, Songcan Chen

**发布时间:** 2025-11-21

**备注:** 9 pages,3 figures,aaai2026

### GPT解析

### 总结

本文提出了一种名为SeeCLIP的语义增强CLIP框架，通过细粒度语义增强解决开放领域泛化中已知类别结构风险和未知类别开放空间风险之间的困境，特别是针对与已知类别具有细粒度视觉相似性的'困难未知'类别。

### 背景

开放集领域泛化(OSDG)处理部署模型同时遇到域偏移和新对象类别的现实场景。尽管CLIP等视觉-语言模型取得显著进展，但现有方法仍面临已知类别结构风险和未知类别开放空间风险之间的困境，在区分'困难未知'类别时容易过度自信。

### 目的

提出一个语义增强的CLIP框架(SeeCLIP)，通过细粒度语义增强明确解决已知类别和未知类别之间的困境，提高模型区分已知类别和困难未知类别的能力。

### 方法

SeeCLIP包含三个核心组件：1)语义感知提示增强模块，将图像分解为判别性语义标记；2)双重对比学习，通过排斥和内聚目标有效定位未知提示；3)语义引导扩散模块，合成与已知类别视觉相似但具有关键局部差异的伪未知样本。

### 主要发现

在五个基准测试上的广泛实验表明，SeeCLIP比最先进的方法 consistently提高了3%的准确率和5%的H-score。

### 结论

SeeCLIP框架通过细粒度语义增强、双重对比学习和语义引导扩散模块，有效解决了开放领域泛化中的困境，特别是在处理与已知类别视觉相似的困难未知类别时表现出色。

### 翻译

开放集领域泛化(OSDG)处理的是部署模型同时遇到域偏移和新对象类别的现实场景。尽管像CLIP这样的视觉-语言模型取得了令人印象深刻的进展，但现有方法仍然陷入已知类别的结构风险和来自未知类别的开放空间风险之间的困境，特别是在区分与已知类别共享细粒度视觉相似性的'困难未知'类别时容易过度自信。为此，我们提出了一个语义增强的CLIP(Semantic-enhanced CLIP, SeeCLIP)框架，通过细粒度语义增强明确解决这一困境。在SeeCLIP中，我们提出了一个语义感知的提示增强模块，将图像分解为判别性语义标记，实现超越粗略类别标签的细粒度视觉-语言对齐。为了有效定位未知提示，我们引入了具有互补目标的双重对比学习，即排斥以保持与已知类别的可分离性，以及内聚以保持语义相似性。此外，我们的语义引导扩散模块通过扰动提取的语义标记合成伪未知样本，生成与已知类别视觉相似但表现出关键局部差异的挑战性样本。这些困难负样本迫使模型学习更精细的决策边界。在五个基准测试上的广泛实验表明，SeeCLIP比最先进的方法 consistently提高了3%的准确率和5%的H-score。


### 论文摘要

Open-Set Domain Generalization (OSDG) tackles the realistic scenario where deployed models encounter both domain shifts and novel object categories. Despite impressive progress with vision-language models like CLIP, existing methods still fall into the dilemma between structural risk of known-classes and open-space risk from unknown-classes, and easily suffers from over-confidence, especially when distinguishing ``hard unknowns" that share fine-grained visual similarities with known classes. To this end, we propose a Semantic-enhanced CLIP (SeeCLIP) framework that explicitly addresses this dilemma through fine-grained semantic enhancement. In SeeCLIP, we propose a semantic-aware prompt enhancement module to decompose images into discriminative semantic tokens, enabling nuanced vision-language alignment beyond coarse category labels. To position unknown prompts effectively, we introduce duplex contrastive learning with complementary objectives, that is, repulsion to maintain separability from known classes, and cohesion to preserve semantic proximity. Further, our semantic-guided diffusion module synthesizes pseudo-unknowns by perturbing extracted semantic tokens, generating challenging samples that are visually similar to known classes yet exhibit key local differences. These hard negatives force the model to learn finer decision boundaries. Extensive experiments across five benchmarks demonstrate consistent improvements of 3% accuracy and 5% H-score over state-of-the-art methods.

---

## 107. Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models

**论文链接:** [http://arxiv.org/abs/2511.16955v1](http://arxiv.org/abs/2511.16955v1)

**作者:** Dailan He, Guanlin Feng, Xingtong Ge, Yazhe Niu, Yi Zhang, Bingqi Ma, Guanglu Song, Yu Liu, Hongsheng Li

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为Neighbor GRPO的新型对齐算法，用于解决Group Relative Policy Optimization在应用于现代flow matching模型时的挑战。该方法通过扰动ODE的初始噪声条件生成候选轨迹，并使用基于softmax距离的代理跳跃策略优化模型，完全避开了对SDE的需求。

### 背景

Group Relative Policy Optimization (GRPO)在图像和视频生成模型与人类偏好对齐方面显示出潜力，但将其应用于现代flow matching模型时面临挑战，因为其确定性采样范式。

### 目的

开发一种新的对齐算法，能够有效解决GRPO在应用于flow matching模型时的问题，同时保持确定性ODE采样的优势。

### 方法

Neighbor GRPO通过扰动ODE的初始噪声条件生成多样化的候选轨迹集合，并使用基于softmax距离的代理跳跃策略优化模型。此外，引入了对称锚采样提高计算效率，以及组-wise准范数重加权来处理奖励扁平化问题。

### 主要发现

从距离优化角度重新解释现有的基于SDE的GRPO方法揭示了它们作为对比学习形式的内在机制。建立了基于距离的目标与策略梯度优化之间的理论联系，证明了Neighbor GRPO与GRPO框架的严格集成。

### 结论

Neighbor GRPO完全保留了确定性ODE采样的优势，包括效率和对高阶求解器的兼容性。实验表明，该方法在训练成本、收敛速度和生成质量方面显著优于基于SDE的对应方法。

### 翻译

Group Relative Policy Optimization (GRPO)在使图像和视频生成模型与人类偏好保持一致方面显示出潜力。然而，将其应用于现代flow matching模型具有挑战性，因为其确定性采样范式。当前方法通过将常微分方程(ODE)转换为随机微分方程(SDE)来解决这个问题，这引入了随机性。然而，这种基于SDE的GRPO存在信用分配效率低下以及与少步采样的高阶求解器不兼容的问题。在本文中，我们首先从距离优化的角度重新解释现有的基于SDE的GRPO方法，揭示了它们作为对比学习形式的内在机制。基于这一见解，我们提出了Neighbor GRPO，一种完全避开SDE需求的新型对齐算法。Neighbor GRPO通过扰动ODE的初始噪声条件生成多样化的候选轨迹集合，并使用基于softmax距离的代理跳跃策略优化模型。我们建立了这种基于距离的目标与策略梯度优化之间的理论联系，将我们的方法严格集成到GRPO框架中。我们的方法完全保留了确定性ODE采样的优势，包括效率和对高阶求解器的兼容性。我们进一步引入了对称锚采样以提高计算效率，以及组-wise准范数重加权来解决奖励扁平化问题。大量实验表明，Neighbor GRPO在训练成本、收敛速度和生成质量方面显著优于基于SDE的对应方法。


### 论文摘要

Group Relative Policy Optimization (GRPO) has shown promise in aligning image and video generative models with human preferences. However, applying it to modern flow matching models is challenging because of its deterministic sampling paradigm. Current methods address this issue by converting Ordinary Differential Equations (ODEs) to Stochastic Differential Equations (SDEs), which introduce stochasticity. However, this SDE-based GRPO suffers from issues of inefficient credit assignment and incompatibility with high-order solvers for fewer-step sampling. In this paper, we first reinterpret existing SDE-based GRPO methods from a distance optimization perspective, revealing their underlying mechanism as a form of contrastive learning. Based on this insight, we propose Neighbor GRPO, a novel alignment algorithm that completely bypasses the need for SDEs. Neighbor GRPO generates a diverse set of candidate trajectories by perturbing the initial noise conditions of the ODE and optimizes the model using a softmax distance-based surrogate leaping policy. We establish a theoretical connection between this distance-based objective and policy gradient optimization, rigorously integrating our approach into the GRPO framework. Our method fully preserves the advantages of deterministic ODE sampling, including efficiency and compatibility with high-order solvers. We further introduce symmetric anchor sampling for computational efficiency and group-wise quasi-norm reweighting to address reward flattening. Extensive experiments demonstrate that Neighbor GRPO significantly outperforms SDE-based counterparts in terms of training cost, convergence speed, and generation quality.

---

## 108. CroTad: A Contrastive Reinforcement Learning Framework for Online Trajectory Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.16929v1](http://arxiv.org/abs/2511.16929v1)

**作者:** Rui Xue, Dan He, Fengmei Jin, Chen Zhang, Xiaofang Zhou

**发布时间:** 2025-11-21

**备注:** 18 pages, 4 figures, will be submitted to VLDBJ

### GPT解析

### 总结

该论文提出了一种名为CroTad的对比强化学习框架，用于在线轨迹异常检测，无需阈值设置且对噪声和不规则采样数据具有鲁棒性。

### 背景

在智能交通系统中，检测轨迹异常是识别不安全、低效或不规则旅行行为的重要任务。尽管深度学习已成为主导方法，但存在子轨迹异常检测研究不足、方法依赖精心调整的阈值、数据不规则采样和噪声导致模型性能下降等挑战。

### 目的

解决轨迹异常检测中的关键挑战，特别是子轨迹异常检测、阈值依赖性、不规则采样和噪声问题，开发一个无需阈值且对噪声和不规则采样数据具有鲁棒性的在线轨迹异常检测框架。

### 方法

提出CroTad对比强化学习框架，通过对比学习提取不同行程的多样化正常旅行模式，在子轨迹和点级别区分异常行为，并利用深度强化学习进行在线、实时异常评分。

### 主要发现

在两个真实世界数据集上的大量实验证明了该框架在各种评估场景中的有效性和鲁棒性。

### 结论

CroTad框架成功解决了轨迹异常检测中的关键挑战，提供了一个无需阈值、对噪声和不规则采样数据具有鲁棒性的解决方案，能够进行在线、实时异常检测并精确定位异常发生的子轨迹和点。

### 翻译

检测轨迹异常是现代智能交通系统中的重要任务，能够识别不安全、低效或不规则的旅行行为。虽然深度学习已成为主导方法，但仍有几个关键挑战未解决。首先，与全轨迹分析相比，子轨迹异常检测（能够精确定位异常发生的精确片段）研究不足。其次，许多现有方法依赖于精心调整的阈值，限制了它们在实际应用中的适应性。此外，轨迹数据的不规则采样和训练集中的噪声进一步降低了模型性能，使其难以学习正常路线的可靠表示。为解决这些挑战，我们提出了一个用于在线轨迹异常检测的对比强化学习框架CroTad。我们的方法无需阈值设置，对噪声和不规则采样的数据具有鲁棒性。通过融入对比学习，CroTad学习提取不同行程的多样化正常旅行模式，并在子轨迹和点级别有效区分异常行为。检测模块利用深度强化学习进行在线、实时的异常评分，实现及时和细粒度的异常片段识别。在两个真实世界数据集上的大量实验证明了我们框架在各种评估场景中的有效性和鲁棒性。


### 论文摘要

Detecting trajectory anomalies is a vital task in modern Intelligent Transportation Systems (ITS), enabling the identification of unsafe, inefficient, or irregular travel behaviours. While deep learning has emerged as the dominant approach, several key challenges remain unresolved. First, sub-trajectory anomaly detection, capable of pinpointing the precise segments where anomalies occur, remains underexplored compared to whole-trajectory analysis. Second, many existing methods depend on carefully tuned thresholds, limiting their adaptability in real-world applications. Moreover, the irregular sampling of trajectory data and the presence of noise in training sets further degrade model performance, making it difficult to learn reliable representations of normal routes. To address these challenges, we propose a contrastive reinforcement learning framework for online trajectory anomaly detection, CroTad. Our method is threshold-free and robust to noisy, irregularly sampled data. By incorporating contrastive learning, CroTad learns to extract diverse normal travel patterns for different itineraries and effectively distinguish anomalous behaviours at both sub-trajectory and point levels. The detection module leverages deep reinforcement learning to perform online, real-time anomaly scoring, enabling timely and fine-grained identification of abnormal segments. Extensive experiments on two real-world datasets demonstrate the effectiveness and robustness of our framework across various evaluation scenarios.

---

## 109. Glass Surface Detection: Leveraging Reflection Dynamics in Flash/No-flash Imagery

**论文链接:** [http://arxiv.org/abs/2511.16887v1](http://arxiv.org/abs/2511.16887v1)

**作者:** Tao Yan, Hao Huang, Yiwei Lu, Zeyu Wang, Ke Xu, Yinghui Wang, Xiaojun Chang, Rynson W. H. Lau

**发布时间:** 2025-11-21

**备注:** 13 pages, 12 figures

### GPT解析

### 总结

本文提出了一种名为NFGlassNet的新型玻璃表面检测方法，通过利用闪光/无闪光图像中的反射动力学特性，结合反射对比度挖掘模块(RCMM)和反射引导注意力模块(RGAM)，实现了比现有方法更准确的玻璃表面检测。

### 背景

玻璃表面在日常生活中无处不在，通常呈现无色、透明且缺乏鲜明特征的特点，这使得玻璃表面检测成为一项具有挑战性的计算机视觉任务。现有的玻璃表面检测方法依赖于边界线索（如门窗框）或反射线索来定位玻璃表面，但未能充分利用玻璃本身的固有特性进行精确定位。

### 目的

提出一种新的玻璃表面检测方法，能够利用玻璃本身的特性进行更准确的检测。

### 方法

作者观察到，在大多数真实场景中，玻璃表面前方的光照强度与后方不同，导致玻璃表面上的反射发生变化。基于这一现象，他们提出了NFGlassNet，一种利用闪光/无闪光图像中反射动力学的玻璃表面检测新方法。具体包括：反射对比度挖掘模块(RCMM)用于提取反射，反射引导注意力模块(RGAM)用于融合反射和玻璃表面特征。此外，他们还构建了一个包含3.3K对无闪光和闪光图像的数据集，来自各种场景并有相应标注。

### 主要发现

通过实验证明，他们的方法优于最先进的方法。

### 结论

他们的代码、模型和数据集将在稿件接受后提供。

### 翻译

玻璃表面在日常生活中无处不在，通常呈现无色、透明且缺乏鲜明特征。这些特点使得玻璃表面检测成为一项具有挑战性的计算机视觉任务。现有的玻璃表面检测方法总是依赖于边界线索（例如门窗框）或反射线索来定位玻璃表面，但它们未能充分利用玻璃本身的固有特性进行精确定位。我们观察到，在大多数真实场景中，玻璃表面前方的光照强度与后方不同，这导致玻璃表面上的反射发生变化。具体来说，当站在玻璃较亮的一侧并向较暗的一侧应用闪光时，玻璃表面上的现有反射往往会消失。相反，当站在较暗的一侧并向较亮的一侧应用闪光时，玻璃表面会出现明显的反射。基于这一现象，我们提出了NFGlassNet，一种利用闪光/无闪光图像中反射动力学的新型玻璃表面检测方法。具体来说，我们提出了一个反射对比度挖掘模块(RCMM)用于提取反射，以及一个反射引导注意力模块(RGAM)用于融合反射和玻璃表面特征，以实现准确的玻璃表面检测。为了学习我们的网络，我们还构建了一个数据集，包含从各种场景捕获的3.3K对无闪光和闪光图像，并有相应的真实标注。大量实验证明，我们的方法优于最先进的方法。我们的代码、模型和数据集将在稿件接受后提供。


### 论文摘要

Glass surfaces are ubiquitous in daily life, typically appearing colorless, transparent, and lacking distinctive features. These characteristics make glass surface detection a challenging computer vision task. Existing glass surface detection methods always rely on boundary cues (e.g., window and door frames) or reflection cues to locate glass surfaces, but they fail to fully exploit the intrinsic properties of the glass itself for accurate localization. We observed that in most real-world scenes, the illumination intensity in front of the glass surface differs from that behind it, which results in variations in the reflections visible on the glass surface. Specifically, when standing on the brighter side of the glass and applying a flash towards the darker side, existing reflections on the glass surface tend to disappear. Conversely, while standing on the darker side and applying a flash towards the brighter side, distinct reflections will appear on the glass surface. Based on this phenomenon, we propose NFGlassNet, a novel method for glass surface detection that leverages the reflection dynamics present in flash/no-flash imagery. Specifically, we propose a Reflection Contrast Mining Module (RCMM) for extracting reflections, and a Reflection Guided Attention Module (RGAM) for fusing features from reflection and glass surface for accurate glass surface detection. For learning our network, we also construct a dataset consisting of 3.3K no-flash and flash image pairs captured from various scenes with corresponding ground truth annotations. Extensive experiments demonstrate that our method outperforms the state-of-the-art methods. Our code, model, and dataset will be available upon acceptance of the manuscript.

---

## 110. Align & Invert: Solving Inverse Problems with Diffusion and Flow-based Models via Representational Alignment

**论文链接:** [http://arxiv.org/abs/2511.16870v1](http://arxiv.org/abs/2511.16870v1)

**作者:** Loukas Sfountouris, Giannis Daras, Paris Giampouras

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为REPA（表示对齐）的方法，通过在扩散模型或流模型与预训练自监督视觉编码器之间建立表示对齐，来提高逆问题中的重建质量和效率。

### 背景

研究表明，在扩散模型或流模型与预训练自监督编码器的内部表示之间强制对齐，可以提供强大的归纳偏置，改善收敛性和样本质量。然而，这种方法尚未在逆问题中得到充分探索。

### 目的

将表示对齐的思想扩展到逆问题中，利用预训练生成模型作为先验，提高重建过程的效率和结果质量。

### 方法

提出REPA方法，在推理过程中对扩散模型或流模型与预训练自监督视觉编码器（如DINOv2）之间的表示进行对齐。尽管在逆问题中无法获得真实信号，但通过将模型表示与近似目标特征对齐，来增强重建保真度和感知真实感。

### 主要发现

1. 将模型表示与近似目标特征对齐可以显著提高重建保真度和感知真实度；2. 提供了理论结果，展示了REPA正则化与DINOv2嵌入空间中散度度量的关系；3. 证明了REPA更新如何将模型的内部表示引导向干净图像的表示；4. 在超分辨率、盒状修复、高斯去模糊和运动去模糊等多个任务上验证了方法的有效性。

### 结论

REPA方法在多种逆问题求解器中表现出通用性，能够一致地提高重建质量，同时通过减少所需的离散化步骤提供显著的效率提升，而不损害底层求解器的性能。

### 翻译

在扩散模型或流模型的内部表示与预训练自监督编码器的表示之间强制对齐，最近已被证明可以提供强大的归纳偏置，改善收敛性和样本质量。在这项工作中，我们将这一思想扩展到逆问题，其中预训练的生成模型被用作先验。我们提出在扩散模型或流模型与预训练的自监督视觉编码器（如DINOv2）之间应用表示对齐（REPA），以指导推理时的重建过程。尽管在逆问题中无法获得真实信号，但我们表明将模型表示与近似目标特征对齐可以显著提高重建保真度和感知真实感。我们提供了理论结果，展示了（a）REPA正则化与DINOv2嵌入空间中散度度量的关系，以及（b）REPA更新如何将模型的内部表示引导向干净图像的表示。这些结果为了解REPA在提高感知保真度方面的作用提供了见解。最后，我们通过将我们的方法集成到多个最先进的逆问题求解器中，证明了其通用性。在超分辨率、盒状修复、高斯去模糊和运动去模糊上的大量实验证实，我们的方法在各项任务中一致地提高了重建质量，同时通过减少所需的离散化步骤提供了显著的效率提升，而不损害底层求解器的性能。


### 论文摘要

Enforcing alignment between the internal representations of diffusion or flow-based generative models and those of pretrained self-supervised encoders has recently been shown to provide a powerful inductive bias, improving both convergence and sample quality. In this work, we extend this idea to inverse problems, where pretrained generative models are employed as priors. We propose applying representation alignment (REPA) between diffusion or flow-based models and a pretrained self-supervised visual encoder, such as DINOv2, to guide the reconstruction process at inference time. Although ground-truth signals are unavailable in inverse problems, we show that aligning model representations with approximate target features can substantially enhance reconstruction fidelity and perceptual realism. We provide theoretical results showing (a) the relation between the REPA regularization and a divergence measure in the DINOv2 embedding space, and (b) how REPA updates steer the model's internal representations toward those of the clean image. These results offer insights into the role of REPA in improving perceptual fidelity. Finally, we demonstrate the generality of our approach by integrating it into multiple state-of-the-art inverse problem solvers. Extensive experiments on super-resolution, box inpainting, Gaussian deblurring, and motion deblurring confirm that our method consistently improves reconstruction quality across tasks, while also providing substantial efficiency gains by reducing the number of required discretization steps without compromising the performance of the underlying solver.

---

## 111. Better audio representations are more brain-like: linking model-brain alignment with performance in downstream auditory tasks

**论文链接:** [http://arxiv.org/abs/2511.16849v1](http://arxiv.org/abs/2511.16849v1)

**作者:** Leonardo Pepino, Pablo Riera, Juan Kamienkowski, Luciana Ferrer

**发布时间:** 2025-11-20

### GPT解析

### 总结

研究表明高性能的人工神经网络音频模型其内部表示与大脑听觉活动更为相似，且这种相似性随着模型性能提升而增强，即使在预训练早期也会自然涌现。

### 背景

人工神经网络(ANNs)日益成为大脑计算的强大模型，但尚不清楚提高模型任务性能是否也会使其内部表示更类似于大脑信号。本研究聚焦于听觉领域探索这一问题。

### 目的

量化36种不同音频模型的内部表示与两个独立fMRI数据集大脑活动之间的对齐程度，评估提高模型性能是否会导致其内部表示更类似于大脑信号。

### 方法

使用体素级和组件级回归以及表示相似性分析(RSA)来评估模型；在HEAREval基准的6个听觉任务(涵盖音乐、语音和环境声音)中评估模型性能；分析EnCodecMAE预训练过程中音频和大脑表示相似性的演变。

### 主要发现

具有强大下游任务性能的最新自监督音频模型比旧的和更专业的模型能更好地预测听觉皮层活动；模型的整体任务性能与其大脑表示对齐之间存在强正相关(Pearson相关系数r>0.7)；在EnCodecMAE的预训练过程中，大脑相似性逐渐增加并在预训练早期出现。

### 结论

大脑般的表示可以从学习重建自然音频数据中的缺失信息中涌现出来，即使模型没有为此目标进行明确优化。

### 翻译

人工神经网络(ANNs)日益成为大脑计算的强大模型，但尚不清楚提高它们的任务性能是否也会使其内部表示更类似于大脑信号。为了在听觉领域解决这个问题，我们量化了36种不同音频模型的内部表示与两个独立fMRI数据集大脑活动之间的对齐程度。使用体素级和组件级回归以及表示相似性分析(RSA)，我们发现具有强大下游任务性能的最新自监督音频模型比旧的和更专业的模型能更好地预测听觉皮层活动。为了评估音频表示的质量，我们在HEAREval基准的6个听觉任务中评估了这些模型，涵盖音乐、语音和环境声音。这揭示了模型的整体任务性能与其大脑表示对齐之间存在强正相关(Pearson相关系数r>0.7)。最后，我们分析了EnCodecMAE预训练过程中音频和大脑表示相似性的演变。我们发现大脑相似性逐渐增加并在预训练早期出现，尽管模型没有为此目标进行明确优化。这表明大脑般的表示可能是从自然音频数据中学习重建缺失信息的涌现副产品。


### 论文摘要

Artificial neural networks (ANNs) are increasingly powerful models of brain computation, yet it remains unclear whether improving their task performance also makes their internal representations more similar to brain signals. To address this question in the auditory domain, we quantified the alignment between the internal representations of 36 different audio models and brain activity from two independent fMRI datasets. Using voxel-wise and component-wise regression, and representation similarity analysis (RSA), we found that recent self-supervised audio models with strong performance in diverse downstream tasks are better predictors of auditory cortex activity than older and more specialized models. To assess the quality of the audio representations, we evaluated these models in 6 auditory tasks from the HEAREval benchmark, spanning music, speech, and environmental sounds. This revealed strong positive Pearson correlations ($r>0.7$) between a model's overall task performance and its alignment with brain representations. Finally, we analyzed the evolution of the similarity between audio and brain representations during the pretraining of EnCodecMAE. We discovered that brain similarity increases progressively and emerges early during pretraining, despite the model not being explicitly optimized for this objective. This suggests that brain-like representations can be an emergent byproduct of learning to reconstruct missing information from naturalistic audio data.

---

## 112. GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs

**论文链接:** [http://arxiv.org/abs/2511.16778v1](http://arxiv.org/abs/2511.16778v1)

**作者:** Yating Ren, Yikun Ban, Huobin Tan

**发布时间:** 2025-11-20

**备注:** AAAI 2026

### GPT解析

### 总结

本文提出了一种名为GCL-OT的新型图对比学习框架，通过最优传输技术解决文本属性图中的结构-文本对齐问题，特别针对多粒度异质性（完全异质性、部分异质性和潜在同质性）设计了特定机制，实验证明该方法在多个基准测试上优于现有方法。

### 背景

结构-文本对比学习在文本属性图上展现出良好性能，但现有方法通常依赖同质性假设和硬优化目标，限制了其在异质性图上的应用。现有方法虽可通过结构调整或邻居聚合缓解异质性，但通常将文本嵌入视为静态目标，导致对齐效果不佳。

### 目的

实现灵活和双向的结构-文本对齐，解决文本属性图中的多粒度异质性导致的对齐挑战，提高模型在异质性图上的适用性和性能。

### 方法

提出GCL-OT框架，针对不同类型异质性设计专门机制：1)对部分异质性，使用基于RealSoftMax的相似性估计器强调关键邻居-词交互；2)对完全异质性，引入基于提示的过滤器自适应排除无关噪声；3)对潜在同质性，结合OT引导的软监督发现相似语义的潜在邻居。

### 主要发现

理论分析表明GCL-OT能改善互信息界限和贝叶斯误差保证；在九个基准测试上的实验显示，GCL-OT持续优于最先进方法，验证了其有效性和鲁棒性。

### 结论

GCL-OT通过针对不同类型异质性设计的专门机制，有效解决了文本属性图中的结构-文本对齐问题，是一种有效且稳健的图对比学习方法。

### 翻译

最近，结构-文本对比学习通过结合图神经网络和语言模型的优势，在文本属性图上展现出有前景的性能。然而，现有方法通常依赖于相似性估计中的同质性假设和硬优化目标，这限制了它们在异质性图上的适用性。尽管现有方法可以通过结构调整或邻居聚合来缓解异质性，但它们通常将文本嵌入视为静态目标，导致次优对齐。在这项工作中，我们识别出文本属性图中的多粒度异质性，包括完全异质性、部分异质性和潜在同质性，由于混合、有噪声和缺失的语义相关性，这使得结构-文本对齐特别具有挑战性。为了实现灵活和双向的对齐，我们提出了GCL-OT，这是一种基于最优传输的新型图对比学习框架，为每种异质性类型配备了定制机制。具体来说，对于部分异质性，我们设计了一个基于RealSoftMax的相似性估计器，以强调关键的邻居-词交互，同时减轻背景噪声。对于完全异质性，我们引入了一个基于提示的过滤器，在最优传输对齐期间自适应地排除无关噪声。此外，我们结合了OT引导的软监督，以发现具有相似语义的潜在邻居，增强潜在同质性的学习。理论分析表明，GCL-OT可以改善互信息界限和贝叶斯误差保证。在九个基准测试上的大量实验表明，GCL-OT持续优于最先进的方法，验证了其有效性和稳健性。


### 论文摘要

Recently, structure-text contrastive learning has shown promising performance on text-attributed graphs by leveraging the complementary strengths of graph neural networks and language models. However, existing methods typically rely on homophily assumptions in similarity estimation and hard optimization objectives, which limit their applicability to heterophilic graphs. Although existing methods can mitigate heterophily through structural adjustments or neighbor aggregation, they usually treat textual embeddings as static targets, leading to suboptimal alignment. In this work, we identify the multi-granular heterophily in text-attributed graphs, including complete heterophily, partial heterophily, and latent homophily, which makes structure-text alignment particularly challenging due to mixed, noisy, and missing semantic correlations. To achieve flexible and bidirectional alignment, we propose GCL-OT, a novel graph contrastive learning framework with optimal transport, equipped with tailored mechanisms for each type of heterophily. Specifically, for partial heterophily, we design a RealSoftMax-based similarity estimator to emphasize key neighbor-word interactions while easing background noise. For complete heterophily, we introduce a prompt-based filter that adaptively excludes irrelevant noise during optimal transport alignment. Furthermore, we incorporate OT-guided soft supervision to uncover potential neighbors with similar semantics, enhancing the learning of latent homophily. Theoretical analysis shows that GCL-OT can improve the mutual information bound and Bayes error guarantees. Extensive experiments on nine benchmarks show that GCL-OT consistently outperforms state-of-the-art methods, verifying its effectiveness and robustness.

---

## 113. Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation

**论文链接:** [http://arxiv.org/abs/2511.16757v1](http://arxiv.org/abs/2511.16757v1)

**作者:** Wei-Cheng Tseng, Xuanru Zhou, Mingyue Huo, Yiwen Shao, Hao Zhang, Dong Yu

**发布时间:** 2025-11-20

**备注:** Work in progress

### GPT解析

### 总结

本研究探讨了音频语言预训练在通用音频理解中的应用，识别了三个关键障碍，并提出了CaptionStew数据集进行系统性评估，发现对比学习和标题化目标在不同规模下各有优势。

### 背景

音频语言预训练对通用音频理解有潜力，但相比视觉语言模型（如CLIP）的研究仍然不足。现有音频语言模型主要在检索任务上表现出色，作为通用编码器的采用有限。

### 目的

识别音频语言模型发展的三个关键障碍：大规模音频文本语料库有限、标题多样性不足、缺乏系统性的探索和评估。引入CaptionStew数据集以促进音频语言预训练研究。

### 方法

创建包含1070万标题的CaptionStew数据集，首次全面对比了对比学习和标题化目标在音频表征学习中的效果，涵盖语音、音乐和环境声音任务，并进行系统性的数据扩展实验。

### 主要发现

音频语言预训练能产生具有竞争力的可转移表征；对比学习在较小规模上数据效率更高；标题化在涉及语言理解的音频任务上可扩展性更好；常见的监督初始化实践在规模扩大时收益递减。

### 结论

音频语言预训练是通用音频表征的可行途径，为未来研究提供指导。研究团队发布了数据准备方案、训练协议和预训练模型，推动通用音频理解的发展。

### 翻译

音频语言预训练对通用音频理解具有潜力，但相比其视觉对应领域的研究仍然不足。虽然视觉语言模型如CLIP被广泛采用作为基础，但现有的音频语言模型主要在检索任务上表现出色，作为通用编码器的采用有限。我们确定了三个关键障碍：大规模音频文本语料库有限、标题多样性不足，以及缺乏系统性的探索和评估。为此，我们引入了CaptionStew，一个包含1070万标题的数据集，汇集了多个领域和标题风格的开放音频文本语料库。利用这一资源，我们进行了首次全面评估，比较了对比学习和标题化目标在音频表征学习中的效果，涵盖语音、音乐和环境声音任务。我们的结果表明，音频语言预训练能够产生具有竞争力的、可转移的表征。通过系统性的数据扩展实验，我们揭示了互补目标的优势：对比学习在较小规模上实现更好的数据效率，而标题化在涉及语言理解的音频任务上表现出更好的可扩展性。我们还发现，常见的监督初始化实践在规模扩大时收益递减，挑战了当前的方法。这些发现确立了音频语言预训练作为通用音频表征的可行途径，为未来研究提供指导。为加速进展，我们发布了数据准备方案、训练协议和预训练模型，为通用音频理解铺平道路。


### 论文摘要

Audio-language pretraining holds promise for general-purpose audio understanding, yet remains underexplored compared to its vision counterpart. While vision-language models like CLIP serve as widely adopted foundations, existing audio-language models primarily excel at retrieval tasks with limited adoption as general-purpose encoders. We identify three key barriers: limited large-scale audio-text corpora, insufficient caption diversity, and lack of systematic exploration and evaluation. To this end, we introduce CaptionStew, a 10.7M caption dataset aggregating diverse open-source audio-text corpora across multiple domains and captioning styles. Using this resource, we conduct the first comprehensive evaluation comparing contrastive and captioning objectives for audio representation learning across speech, music, and environmental sound tasks. Our results demonstrate that audio-language pretraining yields competitive, transferable representations. Through systematic data-scaling experiments, we reveal complementary objective strengths: contrastive learning achieves superior data efficiency at smaller scales, while captioning demonstrates better scalability on language-involved audio understanding tasks. We also find that common supervised initialization practices provide diminishing returns at scale, challenging current approaches. These findings establish audio-language pretraining as a viable pathway toward general-purpose audio representations, guiding future research. To accelerate progress, we release data preparation recipes, training protocols, and pretrained models, paving the way toward universal audio understanding.

---

## 114. Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data

**论文链接:** [http://arxiv.org/abs/2511.17373v1](http://arxiv.org/abs/2511.17373v1)

**作者:** Yixuan Pan, Ruoyi Qiao, Li Chen, Kashyap Chitta, Liang Pan, Haoguang Mai, Qingwen Bu, Hao Zhao, Cunyuan Zheng, Ping Luo, Hongyang Li

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究提出了AMS（敏捷性与稳定性相结合）框架，这是首个将动态运动跟踪和极端平衡维护统一在单一策略中的控制器，使人形机器人能够同时执行敏捷技能和稳定性关键行为。

### 背景

人形机器人需要在以人为中心的环境中执行各种任务，这需要结合敏捷性和稳健平衡的控制器。现有方法在运动和全身跟踪方面虽有进展，但仍然专业化，专注于一种能力而牺牲另一种能力，无法同时满足敏捷性和稳定性的需求。

### 目的

开发一个统一的控制框架，使人形机器人能够同时具备动态运动跟踪的能力和极端平衡维护的能力，解决现有方法专业化、单一化的问题。

### 方法

1. 引入AMS框架，统一动态运动跟踪和极端平衡维护；2. 利用异构数据源：人体运动捕捉数据集（提供敏捷行为）和物理约束合成平衡运动（捕获稳定性配置）；3. 设计混合奖励方案，对所有数据应用一般跟踪目标，同时仅将平衡特定先验注入合成运动中；4. 采用自适应学习策略，包括性能驱动采样和运动特定奖励塑造，实现跨不同运动分布的高效训练。

### 主要发现

1. 单一策略可以执行多种敏捷技能，如跳舞和跑步；2. 同一策略能够执行零样本极端平衡动作，如叶问蹲；3. AMS框架在仿真和真实宇树G1人形机器人上都得到了验证；4. 该框架展示了作为未来人形应用通用控制范式的潜力。

### 结论

AMS框架成功解决了人形机器人控制器中敏捷性与稳定性难以兼顾的问题，通过统一策略和混合奖励方案，实现了对动态运动和极端平衡的同时控制，为未来人形机器人在复杂环境中的应用提供了新的控制范式。

### 翻译

人形机器人被设想在以人为中心的环境中执行各种任务，这需要结合敏捷性和稳健平衡的控制器。最近在运动和全身跟踪方面的进展使得在敏捷动态技能或稳定性关键行为方面取得了令人印象深刻的进展，但现有方法仍然专业化，专注于一种能力而牺牲另一种能力。在这项工作中，我们引入了AMS（敏捷性与稳定性相结合），这是第一个将动态运动跟踪和极端平衡维护统一在单一策略中的框架。我们的关键见解是利用异构数据源：提供丰富敏捷行为的人体运动捕捉数据集，以及捕获稳定性配置的物理约束合成平衡运动。为了协调敏捷性和稳定性的不同优化目标，我们设计了一种混合奖励方案，对所有数据应用一般跟踪目标，同时仅将平衡特定先验注入合成运动中。此外，一种具有性能驱动采样和特定运动奖励塑造的自适应学习策略，使得能够在不同的运动分布上实现高效训练。我们在仿真和真实的宇树G1人形机器人上广泛验证了AMS。实验证明，单一策略可以执行跳舞和跑步等敏捷技能，同时也能执行如叶问蹲等零样本极端平衡动作，突显了AMS作为未来人形应用的通用控制范式。


### 论文摘要

Humanoid robots are envisioned to perform a wide range of tasks in human-centered environments, requiring controllers that combine agility with robust balance. Recent advances in locomotion and whole-body tracking have enabled impressive progress in either agile dynamic skills or stability-critical behaviors, but existing methods remain specialized, focusing on one capability while compromising the other. In this work, we introduce AMS (Agility Meets Stability), the first framework that unifies both dynamic motion tracking and extreme balance maintenance in a single policy. Our key insight is to leverage heterogeneous data sources: human motion capture datasets that provide rich, agile behaviors, and physically constrained synthetic balance motions that capture stability configurations. To reconcile the divergent optimization goals of agility and stability, we design a hybrid reward scheme that applies general tracking objectives across all data while injecting balance-specific priors only into synthetic motions. Further, an adaptive learning strategy with performance-driven sampling and motion-specific reward shaping enables efficient training across diverse motion distributions. We validate AMS extensively in simulation and on a real Unitree G1 humanoid. Experiments demonstrate that a single policy can execute agile skills such as dancing and running, while also performing zero-shot extreme balance motions like Ip Man's Squat, highlighting AMS as a versatile control paradigm for future humanoid applications.

---

## 115. Step-E: A Differentiable Data Cleaning Framework for Robust Learning with Noisy Labels

**论文链接:** [http://arxiv.org/abs/2511.17040v1](http://arxiv.org/abs/2511.17040v1)

**作者:** Wenzhang Du

**发布时间:** 2025-11-21

**备注:** 12 pages, 4 figures

### GPT解析

### 总结

Step-E是一种新框架，将样本选择和模型学习整合到单个优化过程中，通过在线课程学习处理野外收集训练数据中的噪声标签和异常值，显著提升了深度神经网络性能。

### 背景

在野外收集的训练数据通常包含噪声标签和异常值，这些会显著降低深度神经网络的性能和可靠性。目前常用的数据清理作为独立预处理阶段的两阶段管道，既无法充分利用下游模型的反馈，也难以适应未知的噪声模式。

### 目的

开发一种能够整合样本选择和模型学习的框架，充分利用下游模型反馈，并适应未知噪声模式，以提高深度神经网络在噪声数据上的性能和可靠性。

### 方法

提出Step-E框架，在每个epoch中按损失对样本进行排序，在短暂预热阶段后逐渐增加被排除在梯度更新之外的高损失样本比例，形成在线课程，专注于简单和一致的样本，最终忽略持续的异常值。

### 主要发现

在CIFAR-100N上，Step-E将ResNet-18模型的测试准确率从43.3%提高到50.4%，明显优于损失截断、自 paced学习和一次性过滤，接近60.5%的清洁标签oracle。在CIFAR-10N (aggre)上，Step-E也优于噪声基线（85.3% vs. 83.9%）并几乎匹配清洁标签oracle（85.9%），仅具有适度的训练时间开销。

### 结论

Step-E通过整合样本选择和模型学习的单一优化过程，有效处理了训练数据中的噪声标签和异常值问题，在多个数据集上表现出色，接近清洁标签的性能水平。

### 翻译

在野外收集的训练数据通常包含噪声标签和异常值，这些会显著降低深度神经网络的性能和可靠性。虽然数据清理通常作为独立的预处理阶段应用，但这种两阶段管道既不能充分利用下游模型的反馈，也不能适应未知的噪声模式。我们提出了Step-E，一个将样本选择和模型学习整合到单个优化过程中的简单框架。在每个epoch中，Step-E按损失对样本进行排序，并在短暂的预热阶段后逐渐增加被排除在梯度更新之外的高损失样本的比例，产生一个专注于简单和一致样本的在线课程，并最终忽略持续的异常值。在CIFAR-100N上，Step-E将ResNet-18模型的测试准确率从43.3% (+/- 0.7%)提高到50.4% (+/- 0.9%)，明显优于损失截断、自 paced学习和一次性过滤，同时接近60.5% (+/- 0.2%)的清洁标签oracle。在CIFAR-10N (aggre)上，Step-E也优于噪声基线（85.3% vs. 83.9%）并几乎匹配清洁标签oracle（85.9%），仅具有适度的训练时间开销。


### 论文摘要

Training data collected in the wild often contain noisy labels and outliers that substantially degrade the performance and reliability of deep neural networks. While data cleaning is commonly applied as a separate preprocessing stage, such two-stage pipelines neither fully exploit feedback from the downstream model nor adapt to unknown noise patterns. We propose Step-E, a simple framework that integrates sample selection and model learning into a single optimization process. At each epoch, Step-E ranks samples by loss and gradually increases the fraction of high-loss examples that are excluded from gradient updates after a brief warm-up stage, yielding an online curriculum that focuses on easy and consistent examples and eventually ignores persistent outliers. On CIFAR-100N, Step-E improves the test accuracy of a ResNet-18 model from 43.3% (+/- 0.7%) to 50.4% (+/- 0.9%), clearly outperforming loss truncation, self-paced learning, and one-shot filtering while approaching the clean-label oracle at 60.5% (+/- 0.2%). On CIFAR-10N (aggre), Step-E also improves over the noisy baseline (85.3% vs. 83.9%) and nearly matches the clean-label oracle (85.9%), with only moderate training-time overhead.

---

## 116. Continuous Resilience in Cyber-Physical Systems of Systems: Extending Architectural Models through Adaptive Coordination and Learning

**论文链接:** [http://arxiv.org/abs/2511.17017v1](http://arxiv.org/abs/2511.17017v1)

**作者:** Elisabeth Vogel, Peter Langendörfer

**发布时间:** 2025-11-21

**备注:** 27 pages, 6 tables, 1 figure

### GPT解析

### 总结

该论文提出了一种新的自适应协调层(ACL)和重新定义的适应与学习层(AL)架构，用于解决网络物理系统(CPSoS)中的弹性适应问题。该架构结合了短期响应能力和长期学习能力，将弹性从静态属性转变为持续的数据驱动过程。

### 背景

网络物理系统(CPSoS)是高度复杂、动态的环境，技术、控制论和组织子系统紧密互动。需要动态、持续可适应的弹性来确保它们在变化条件下的功能。

### 目的

解决现有弹性架构主要保持静态的问题，通过引入新的架构模型实现CPSoS的动态适应能力。

### 方法

提出自适应协调层(ACL)作为操作控制层，实时检测风险并动态协调对策；重新定义适应与学习层(AL)为战略协作层，评估ACL决策并学习推导长期调整。描述了从基于规则到AI支持的多种实现变体，可根据系统复杂性、数据可用性和监管程度进行组合。

### 主要发现

所提出的架构模型将弹性理解为持续的数据驱动过程，涉及相互协调和系统学习，而非静态的系统属性。

### 结论

该架构为下一代自适应和弹性的网络物理系统提供了方法基础，实现了短期响应与长期学习的结合。

### 翻译

网络物理系统(CPSoS)是高度复杂、动态的环境，其中技术、控制论和组织子系统紧密互动。需要动态、持续可适应的弹性来确保它们在变化条件下的功能。然而，现有的弹性架构通常只隐式处理适应性问题，因此主要保持静态。本文通过引入新的自适应协调层(ACL)和概念性地重新定义适应与学习层(AL)来解决这一差距。ACL作为操作控制层，实时检测风险，优先考虑对策并动态协调它们。AL被重新解释为战略协作层，评估ACL的操作决策，从中学习，并在政策、治理和架构层面推导长期调整。两个层次共同实现适应的弹性原则，结合短期响应能力与长期学习和开发能力。论文描述了两个层次的多种实现变体——从基于规则和KPI驱动的方法到AI支持和元学习机制，并展示了如何根据系统复杂性、数据可用性和监管程度组合这些方法。所提出的架构模型不再将弹性理解为静态的系统属性，而是理解为持续的数据驱动过程，涉及相互协调和系统学习。这为下一代自适应和弹性的CPSoS创造了方法基础。


### 论文摘要

Cyber-physical systems of systems (CPSoS) are highly complex, dynamic environments in which technical, cybernetic and organisational subsystems interact closely with one another. Dynamic, continuously adaptable resilience is required to ensure their functionality under variable conditions. However, existing resilience architectures usually only deal with adaptation implicitly and thus remain predominantly static. This paper addresses this gap by introducing a new Adaptive Coordination Layer (ACL) and conceptually redefining the Adaptation & Learning Layer (AL). The ACL acts as an operational control layer that detects risks in real time, prioritises countermeasures and coordinates them dynamically. The AL is reinterpreted as a strategic-cooperative layer that evaluates the operational decisions of the ACL, learns from them, and derives long-term adjustments at the policy, governance, and architecture levels. Together, both layers operationalise the resilience principle of adaptation and combine short-term responsiveness with long-term learning and development capabilities. The paper describes various implementation variants of both levels - from rule-based and KPI-driven approaches to AI-supported and meta-learning mechanisms - and shows how these can be combined depending on system complexity, data availability and degree of regulation. The proposed architecture model no longer understands resilience as a static system property, but as a continuous, data-driven process of mutual coordination and systemic learning. This creates a methodological basis for the next generation of adaptive and resilient CPSoS.

---

## 117. ARQUSUMM: Argument-aware Quantitative Summarization of Online Conversations

**论文链接:** [http://arxiv.org/abs/2511.16985v1](http://arxiv.org/abs/2511.16985v1)

**作者:** An Quang Tang, Xiuzhen Zhang, Minh Ngoc Dinh, Zhuang Li

**发布时间:** 2025-11-21

**备注:** Paper accepted to AAAI2026 Main Technical Track

### GPT解析

### 总结

本文提出了一种新的论证感知定量摘要方法，用于揭示在线对话中论点的主张-理由结构，并通过数量测量论证强度。

### 背景

在线对话在公共讨论平台上越来越普遍，随着争议性话题的增加，需要总结多样化的论点及其理由和论证。早期文本摘要研究忽视了对话的论证性质，最近的对话摘要研究虽然考虑了句子间的论证关系，但没有阐明句子内部的深层论证结构。

### 目的

提出一种新的'论证感知定量摘要'任务，揭示对话中论点的'主张-理由'结构，并用数量测量论证强度。

### 方法

提出ARQUSUMM框架，利用基于论证理论的LLM少样本学习来识别句子中的命题及其主张-理由关系，并使用论证结构感知的聚类算法来聚合论点并量化其支持度。

### 主要发现

实验表明ARQUSUMM优于现有的对话和定量摘要模型，生成的摘要更能代表论证结构，对用户更有帮助，文本质量和量化准确性更高。

### 结论

ARQUSUMM框架能够有效揭示对话中的论证结构，生成高质量的定量摘要，为在线对话分析提供了新方法。

### 翻译

在线对话在公共讨论平台上（如Reddit）变得越来越普遍。随着争议性话题的增加，人们期望能够总结多样化的论点及其理由和论证。早期关于文本摘要的研究专注于捕捉源文档中的显著一般信息，忽视了在线对话的论证性质。最近的对话摘要研究虽然考虑了句子间的论证关系，但在摘要中未能阐明句子内部的深层论证结构。在本文中，我们提出了一个新颖的论证感知定量摘要任务，以揭示对话中论点的主张-理由结构，并用数量测量论证强度。我们进一步提出了ARQUSUMM，一个解决该任务的新颖框架。为了揭示句子内部的潜在论证结构，ARQUSUMM利用基于论证理论的LLM少样本学习来识别句子中的命题及其主张-理由关系。对于定量摘要，ARQUSUMM采用论证结构感知的聚类算法来聚合论点并量化其支持度。实验表明，ARQUSUMM优于现有的对话和定量摘要模型，生成的摘要更能代表论证结构，对用户更有帮助，文本质量和量化准确性更高。


### 论文摘要

Online conversations have become more prevalent on public discussion platforms (e.g. Reddit). With growing controversial topics, it is desirable to summarize not only diverse arguments, but also their rationale and justification. Early studies on text summarization focus on capturing general salient information in source documents, overlooking the argumentative nature of online conversations. Recent research on conversation summarization although considers the argumentative relationship among sentences, fail to explicate deeper argument structure within sentences for summarization. In this paper, we propose a novel task of argument-aware quantitative summarization to reveal the claim-reason structure of arguments in conversations, with quantities measuring argument strength. We further propose ARQUSUMM, a novel framework to address the task. To reveal the underlying argument structure within sentences, ARQUSUMM leverages LLM few-shot learning grounded in the argumentation theory to identify propositions within sentences and their claim-reason relationships. For quantitative summarization, ARQUSUMM employs argument structure-aware clustering algorithms to aggregate arguments and quantify their support. Experiments show that ARQUSUMM outperforms existing conversation and quantitative summarization models and generate summaries representing argument structures that are more helpful to users, of high textual quality and quantification accuracy.

---

## 118. ToC: Tree-of-Claims Search with Multi-Agent Language Models

**论文链接:** [http://arxiv.org/abs/2511.16972v1](http://arxiv.org/abs/2511.16972v1)

**作者:** Shuyang Yu, Jianan Liang, Hui Hu

**发布时间:** 2025-11-21

**备注:** Accepted by AAAI 2026 (Oral)

### GPT解析

### 总结

论文介绍了一种名为声明树(ToC)的创新框架，将专利声明优化重新定义为有引导的搜索问题，结合蒙特卡洛树搜索和多代理系统，显著优于传统大型语言模型

### 背景

优化专利声明需平衡新颖性与法律范围，手动声明起草劳动密集、成本高且不一致，传统大型语言模型缺乏结构化迭代推理能力

### 目的

解决专利声明优化挑战，提供能平衡新颖性、范围保留和语义连贯性的系统

### 方法

ToC框架整合蒙特卡洛树搜索(MCTS)与协作多代理系统，包括基于LLM的EditorAgent提出上下文编辑，ExaminerAgent模拟专利审查员分析，由多目标奖励函数驱动共同优化

### 主要发现

在1145个声明的基准测试中，ToC在零样本和少样本场景中平均综合得分提高8%，某些情况下达9%，消融研究验证其生成法律健壮声明修订的有效性

### 结论

ToC建立了透明、可控、可解释的方法论，将先进LLM推理能力与战略MCTS规划结合，实现结构化专利声明优化

### 翻译

优化专利声明是一项关键且具有挑战性的任务，需要在最大程度地提高新颖性和保留法律范围之间进行谨慎平衡。手动声明起草劳动密集、成本高昂且本质上不一致，而传统大型语言模型通常缺乏精确声明完善所必需的结构化、迭代推理能力。为解决这些挑战，我们引入了声明树(ToC)，一个将声明编辑重新定义为有引导搜索问题的创新框架。ToC协同整合了蒙特卡洛树搜索(MCTS)与协作多代理系统，包括一个基于LLM的EditorAgent，提出基于上下文的编辑，以及一个ExaminerAgent，通过结构化的、思维链分析模拟专利审查员对新颖性和现有技术披露的批评。由精心设计的多目标奖励函数驱动，ToC共同优化新颖性、范围保留和语义连贯性。在包含1145个声明的基准测试中，实验证明，ToC在零样本和少样本场景中显著优于标准LLMs，平均综合得分提高8%，某些情况下高达9%。包括详细消融研究在内的广泛实验验证了ToC在生成优越的、法律健壮的声明修订方面的有效性。总体而言，ToC建立了一种透明、可控和可解释的方法论，有效地将先进的LLM推理能力与战略性的MCTS规划结合起来，用于结构化的专利声明优化。源代码可在https://github.com/ysy2003/ToC获取。


### 论文摘要

Optimizing patent claims is a critical yet challenging task, demanding careful balance between maximizing novelty and preserving legal scope. Manual claim drafting is labor-intensive, costly, and inherently inconsistent, while conventional Large Language Models (LLMs) often lack the structured, iterative reasoning essential for precise claim refinement. To address these challenges, we introduce Tree of Claims (ToC), an innovative framework that redefines claim editing as a guided search problem. ToC synergistically integrates Monte Carlo Tree Search (MCTS) with a collaborative multi-agent system, comprising an LLM-based EditorAgent that proposes contextually grounded edits, and an ExaminerAgent that mimics patent examiner critiques through structured, chain-of-thought analyses of novelty and prior art disclosure. Driven by a carefully designed multi-objective reward function, ToC jointly optimizes novelty, scope retention, and semantic coherence. Experimental evaluation on a benchmark of 1145 claims demonstrates that ToC significantly outperforms standard LLMs in zero-shot and few-shot scenarios, achieving an average composite score improvement of 8\%, and up to 9\% in certain cases. Extensive experiments, including detailed ablation studies, validate ToC's efficacy in generating superior, legally robust claim revisions. Overall, ToC establishes a transparent, controllable, and interpretable methodology that effectively bridges advanced LLM reasoning capabilities with strategic MCTS planning for structured patent claim optimization.The source code is available at https://github.com/ysy2003/ToC.

---

## 119. Accelerated Materials Discovery through Cost-Aware Bayesian Optimization of Real-World Indentation Workflows

**论文链接:** [http://arxiv.org/abs/2511.16930v1](http://arxiv.org/abs/2511.16930v1)

**作者:** Vivek Chawla, Stephen Puplampu, Haochen Zhu, Philip D. Rack, Dayakar Penumadu, Sergei Kalinin

**发布时间:** 2025-11-21

**备注:** 29 pages, 8 figures, 2 figures in SI

### GPT解析

### 总结

研究开发了一种自动化纳米压痕框架，通过结合异方差高斯过程建模和成本感知贝叶斯优化，实现了组合薄膜库的高效自适应机械映射，在保持测量精度的同时显著减少了测试时间。

### 背景

加速组合材料中机械性能的发现需要考虑仪器行为和实验成本的自主实验方法，传统方法常忽略横向运动、漂移稳定化和重新配置等因素。

### 目的

开发并验证一种用于组合薄膜库自适应机械映射的自动化纳米压痕框架，优化实验效率并保持测量精度。

### 方法

结合异方差高斯过程建模和成本感知贝叶斯优化动态选择压痕位置和保持时间；使用详细仿真器和成本模型捕获内在惩罚；引入分层元测试工作流程结合局部网格和全局探索以防止核长度尺度崩溃。

### 主要发现

在Ta-Ti-Hf-Zr薄膜库实验中，该框架实现了比基于网格的压痕近30倍的属性映射效率提升，证明将成本和漂移模型纳入概率规划可显著提高性能。

### 结论

该研究建立了自主材料表征中优化实验工作流程的可推广策略，并可扩展到其他高精度、漂移受限的仪器。

### 翻译

加速组合材料中机械性能的发现需要同时考虑仪器行为和实验成本的自主实验。本文开发并验证了一种用于组合薄膜库自适应机械映射的自动化纳米压痕框架。该方法将异方差高斯过程建模与成本感知贝叶斯优化相结合，动态选择压痕位置和保持时间，在保持测量精度的同时最小化总测试时间。详细的仿真器和成本模型捕获了与横向运动、漂移稳定化和重新配置相关的内在惩罚，这些因素在传统主动学习方法中常被忽略。为防止因不同时间尺度导致的核长度尺度崩溃，引入了结合局部网格和全局探索的分层元测试工作流程。在Ta-Ti-Hf-Zr薄膜库实验中展示了该工作流程的实现。所提出的框架实现了比基于网格的压痕近30倍的属性映射效率提升，证明将成本和漂移模型纳入概率规划可显著提高性能。该研究建立了自主材料表征中优化实验工作流程的可推广策略，并可扩展到其他高精度、漂移受限的仪器。


### 论文摘要

Accelerating the discovery of mechanical properties in combinatorial materials requires autonomous experimentation that accounts for both instrument behavior and experimental cost. Here, an automated nanoindentation (AE-NI) framework is developed and validated for adaptive mechanical mapping of combinatorial thin-film libraries. The method integrates heteroskedastic Gaussian-process modeling with cost-aware Bayesian optimization to dynamically select indentation locations and hold times, minimizing total testing time while preserving measurement accuracy. A detailed emulator and cost model capture the intrinsic penalties associated with lateral motion, drift stabilization, and reconfiguration-factors often neglected in conventional active-learning approaches. To prevent kernel-length-scale collapse caused by disparate time scales, a hierarchical meta-testing workflow combining local grid and global exploration is introduced. Implementation of the workflow is shown on a experimental Ta-Ti-Hf-Zr thin-film library. The proposed framework achieves nearly a thirty-fold improvement in property-mapping efficiency relative to grid-based indentation, demonstrating that incorporating cost and drift models into probabilistic planning substantially improves performance. This study establishes a generalizable strategy for optimizing experimental workflows in autonomous materials characterization and can be extended to other high-precision, drift-limited instruments.

---

## 120. PersonalizedRouter: Personalized LLM Routing via Graph-based User Preference Modeling

**论文链接:** [http://arxiv.org/abs/2511.16883v1](http://arxiv.org/abs/2511.16883v1)

**作者:** Zhongjie Dai, Tao Feng, Jiaxuan You

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究提出了PersonalizedRouter，一个基于图的个性化大型语言模型选择框架，能够从交互数据中学习用户偏好并进行个性化LLM选择。

### 背景

随着具有不同能力和响应风格的大型语言模型数量增加，用户面临选择合适LLM的挑战，因为用户在性能、成本和响应风格方面的偏好各不相同。

### 目的

解决当前LLM选择方法通常只优化单一固定目标，无法从交互数据中学习个体用户偏好的问题。

### 方法

提出PersonalizedRouter框架，通过建模多样化用户档案并利用交互数据进行个性化LLM选择。将交互数据转换为异构图捕获上下文信息，设计了多成本效率模拟和LLM-as-a-Judge两种评估策略，并构建了包含1000个模拟用户和10个LLM的PersonaRoute-Bench基准。

### 主要发现

实验表明PersonalizedRouter显著优于现有方法，在两种模拟策略下分别以15.38%和9.83%的优势超越最强方法；在PersonaRoute-Bench上以16.19%和59.69%的优势超越最佳方法，同时保持更高效率；在适应新用户和新LLM时，分别达到完全训练模型性能的64.81%和85.80%。

### 结论

PersonalizedRouter通过基于图的个性化选择框架有效解决了LLM选择中考虑用户偏好的问题，并在各种评估策略下表现出色。

### 翻译

随着具有不同能力和响应风格的大型语言模型数量增加，用户面临选择合适LLM的挑战，因为用户在性能、成本和响应风格方面的偏好各不相同。当前LLM选择方法通常优化单一固定目标，如性能、成本或它们的权衡，无法从交互数据中学习个体用户偏好。为解决这些问题，我们提出了PersonalizedRouter，一个基于图的框架，通过建模多样化用户档案并利用包含任务上下文、查询、候选LLM和用户决策的交互数据进行个性化LLM选择。为捕获用户查询与最优LLM之间的上下文信息，PersonalizedRouter将交互数据转换为异构图，其中不同类型节点之间的关系通过边表示。为评估跨用户适应性，我们设计了两种策略：多成本效率模拟策略和LLM-as-a-Judge策略。此外，我们构建了PersonaRoute-Bench，一个包含1000个模拟用户和10个LLM的大规模基准。实验结果表明，PersonalizedRouter显著优于现有LLM选择方法，在两种模拟策略下分别以15.38%和9.83%的大幅优势超越最强方法。在包含1000个用户的PersonaRoute-Bench上，它进一步以16.19%和59.69%的优势超越最佳方法，同时保持更高效率。此外，PersonalizedRouter展示了强大的少样本泛化能力，在适应新用户和新LLM时，分别达到完全训练模型性能的64.81%和85.80%。


### 论文摘要

The growing number of Large Language Models (LLMs) with diverse capabilities and response styles provides users with a wider range of choices, which presents challenges in selecting appropriate LLMs, as user preferences vary in terms of performance, cost, and response style. Current LLM selection methods typically optimize for a single fixed objective, such as performance, cost, or a trade-off between them, and fail to learn individual user preferences from interaction data. To address these limitations, we propose PersonalizedRouter, a graph-based framework that models diverse user profiles and performs personalized LLM selection by leveraging interaction data that includes task context, queries, candidate LLMs, and user decisions. To capture contextual information between user queries and optimal LLMs, PersonalizedRouter converts the interaction data into a heterogeneous graph, where the relationships between different types of nodes are represented by edges. To evaluate adaptability across users, we design two strategies: the multi-cost-efficiency simulation strategy and the LLM-as-a-Judge strategy. In addition, we construct PersonaRoute-Bench, a large-scale benchmark with 1,000 simulated users and 10 LLMs. Experimental results show that PersonalizedRouter significantly outperforms existing LLM selection methods and surpasses the strongest methods by a large margin of 15.38% and 9.83% under two simulation strategies. On the PersonaRoute-Bench with 1,000 users, it further surpasses the best methods by 16.19% and 59.69% while maintaining higher efficiency. Moreover, PersonalizedRouter demonstrates strong few-shot generalization, achieving 64.81% and 85.80% of the fully trained model's performance when adapting to new users and new LLMs.

---

## 121. Supervised Contrastive Learning for Few-Shot AI-Generated Image Detection and Attribution

**论文链接:** [http://arxiv.org/abs/2511.16541v2](http://arxiv.org/abs/2511.16541v2)

**作者:** Jaime Álvarez Urueña, David Camacho, Javier Huertas Tato

**发布时间:** 2025-11-20

**备注:** 17 pages, 6 figures, 6 tables

### GPT解析

### 总结

本文提出了一种新的两阶段检测框架，用于解决生成式AI创建的合成图像检测中的泛化挑战。该方法在少样本学习条件下实现了高准确率，并能有效适应新型生成模型，无需频繁重新训练。

### 背景

生成式人工智能的快速发展使得合成的图像越来越难以与真实内容区分，这对数字媒体完整性构成了重大挑战。新型生成模型的加速发布周期使传统的依赖定期重新训练的检测方法在计算上不可行且操作上不切实际。

### 目的

设计一种新的两阶段检测框架，解决合成图像检测中固有的泛化挑战，使检测系统能够适应不断发展的生成式AI环境。

### 方法

第一阶段使用通过监督对比学习训练的视觉深度学习模型提取图像判别性嵌入，在战略性划分的生成器子集上训练并保留特定架构不参与训练以评估跨生成器泛化能力；第二阶段使用k-NN分类器在学习的嵌入空间上操作，采用少样本学习范式训练，包含来自未见过的测试生成器的有限样本。

### 主要发现

在少样本学习模式下，每类仅使用150张图像，框架实现了91.3%的平均检测准确率，比现有方法提高5.2个百分点；对于来源归属任务，在开放集分类背景下，AUC和OSCR分别提高了14.70%和4.27%。

### 结论

该研究代表了向健壮、可扩展的取证归属系统的重要进展，这些系统能够适应不断发展的生成式AI环境，无需彻底的重新训练协议。

### 翻译

生成式人工智能的快速发展使得能够创建与真实内容越来越难以区分的合成图像，对数字媒体完整性构成了重大挑战。新型生成模型的加速发布周期使传统的依赖定期重新训练的检测方法在计算上不可行且操作上不切实际。本研究提出了一种新的两阶段检测框架，旨在解决合成图像检测中固有的泛化挑战。第一阶段采用通过监督对比学习训练的视觉深度学习模型，从输入图像中提取判别性嵌入。关键的是，该模型在可用的生成器的战略性划分子集上训练，并特意保留特定架构不参与训练，以严格评估跨生成器泛化能力。第二阶段使用在学习的嵌入空间上操作的k最近邻(k-NN)分类器，在少样本学习范式中训练，包含来自先前未见过的测试生成器的有限样本。在少样本学习模式下，每类仅使用150张图像（这些图像很容易从当前生成模型中获取），所提出的框架实现了91.3%的平均检测准确率，比现有方法提高了5.2个百分点。对于来源归属任务，在开放集分类背景下，所提出的方法在AUC和OSCR上分别提高了14.70%和4.27%，这标志着向健壮、可扩展的取证归属系统的重要进展，这些系统能够适应不断发展的生成式AI环境，无需彻底的重新训练协议。


### 论文摘要

The rapid advancement of generative artificial intelligence has enabled the creation of synthetic images that are increasingly indistinguishable from authentic content, posing significant challenges for digital media integrity. This problem is compounded by the accelerated release cycle of novel generative models, which renders traditional detection approaches (reliant on periodic retraining) computationally infeasible and operationally impractical.   This work proposes a novel two-stage detection framework designed to address the generalization challenge inherent in synthetic image detection. The first stage employs a vision deep learning model trained via supervised contrastive learning to extract discriminative embeddings from input imagery. Critically, this model was trained on a strategically partitioned subset of available generators, with specific architectures withheld from training to rigorously ablate cross-generator generalization capabilities. The second stage utilizes a k-nearest neighbors (k-NN) classifier operating on the learned embedding space, trained in a few-shot learning paradigm incorporating limited samples from previously unseen test generators.   With merely 150 images per class in the few-shot learning regime, which are easily obtainable from current generation models, the proposed framework achieves an average detection accuracy of 91.3%, representing a 5.2 percentage point improvement over existing approaches . For the source attribution task, the proposed approach obtains improvements of of 14.70% and 4.27% in AUC and OSCR respectively on an open set classification context, marking a significant advancement toward robust, scalable forensic attribution systems capable of adapting to the evolving generative AI landscape without requiring exhaustive retraining protocols.

---

## 122. R2PS: Worst-Case Robust Real-Time Pursuit Strategies under Partial Observability

**论文链接:** [http://arxiv.org/abs/2511.17367v1](http://arxiv.org/abs/2511.17367v1)

**作者:** Runyu Lu, Ruochuan Shi, Yuanheng Zhu, Dongbin Zhao

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种在部分可观察性下针对追逐逃避游戏的最坏情况鲁棒实时追逐策略(R2PS)方法，解决了追逐者只有不完美信息时的实时策略问题。

### 背景

计算追逐逃避游戏中的最坏情况鲁棒策略非常耗时，尤其当考虑部分可观察性等现实因素时。现有基于图的追逐逃避游戏中，当追逐者只有关于逃逸者位置的不完美信息时，缺少实时适用的追逐策略。现有强化学习方法如EPG和Grasper仅适用于完美信息场景，未考虑逃逸者可能预测追逐者行动的情况。

### 目的

引入首个在部分可观察性下针对最坏情况鲁棒实时追逐策略(R2PS)的方法。

### 方法

首先证明传统动态规划算法在解决马尔可夫追逐逃避游戏时即使逃逸者采取异步移动也能保持最优性；然后提出关于逃逸者可能位置的信念保存机制，将DP追逐策略扩展到部分可观察设置；最后将信念保存机制嵌入EPG框架，通过跨图强化学习实现实时追逐者策略。

### 主要发现

通过强化学习，该策略实现了对未见过的真实世界图结构的鲁棒零样本泛化，且持续优于现有游戏RL方法直接在测试图上训练的策略。

### 结论

该方法成功解决了部分可观察性下的实时追逐策略问题，在真实场景中表现出更好的泛化能力和性能。

### 翻译

在追逐逃避游戏(PEGs)中计算最坏情况鲁棒策略非常耗时，特别是当考虑部分可观察性等现实因素时。虽然这对一般安全目的很重要，但当追逐者只有关于逃逸者位置的不完美信息时，目前仍缺少基于图的PEGs中实时适用的追逐策略。虽然最先进的强化学习方法如EPG和Grasper为学习对各种游戏动态具有鲁棒性的图神经网络策略提供了指导，但它们仅限于完美信息场景，没有考虑逃逸者可能预测追逐者行动的情况。本文首次引入了部分可观察性下最坏情况鲁棒实时追逐策略(R2PS)的方法。我们首先证明了解决马尔可夫PEGs的传统动态规划算法在逃逸者异步移动的情况下仍保持最优性。然后，我们提出了关于逃逸者可能位置的信念保存机制，将DP追逐策略扩展到部分可观察设置。最后，我们将信念保存嵌入到最先进的EPG框架中，完成了我们的R2PS学习方案，通过跨图强化学习对抗异步移动DP逃避策略，实现了实时追逐者策略。经过强化学习，我们的策略实现了对未见过的真实世界图结构的鲁棒零样本泛化，并持续优于现有游戏RL方法直接在测试图上训练的策略。


### 论文摘要

Computing worst-case robust strategies in pursuit-evasion games (PEGs) is time-consuming, especially when real-world factors like partial observability are considered. While important for general security purposes, real-time applicable pursuit strategies for graph-based PEGs are currently missing when the pursuers only have imperfect information about the evader's position. Although state-of-the-art reinforcement learning (RL) methods like Equilibrium Policy Generalization (EPG) and Grasper provide guidelines for learning graph neural network (GNN) policies robust to different game dynamics, they are restricted to the scenario of perfect information and do not take into account the possible case where the evader can predict the pursuers' actions. This paper introduces the first approach to worst-case robust real-time pursuit strategies (R2PS) under partial observability. We first prove that a traditional dynamic programming (DP) algorithm for solving Markov PEGs maintains optimality under the asynchronous moves by the evader. Then, we propose a belief preservation mechanism about the evader's possible positions, extending the DP pursuit strategies to a partially observable setting. Finally, we embed the belief preservation into the state-of-the-art EPG framework to finish our R2PS learning scheme, which leads to a real-time pursuer policy through cross-graph reinforcement learning against the asynchronous-move DP evasion strategies. After reinforcement learning, our policy achieves robust zero-shot generalization to unseen real-world graph structures and consistently outperforms the policy directly trained on the test graphs by the existing game RL approach.

---

## 123. Topologic Attention Networks: Attending to Direct and Indirect Neighbors through Gaussian Belief Propagation

**论文链接:** [http://arxiv.org/abs/2511.16871v1](http://arxiv.org/abs/2511.16871v1)

**作者:** Marshall Rosenhoover, Huaming Zhang

**发布时间:** 2025-11-21

**备注:** 15 pages, 13 Figures

### GPT解析

### 总结

本文提出了一种名为拓扑注意力网络的新型图神经网络框架，通过拓扑注意力机制解决传统GNN在建模长程依赖关系方面的局限性，同时保持计算效率和可扩展性，并在多个基准测试中取得了最先进的性能。

### 背景

图神经网络依赖于局部消息传递，这限制了它们建模图中长程依赖关系的能力。现有方法通过连续时间动力学或密集自注意力来扩展这一范围，但两者都存在高计算成本和可扩展性有限的问题。

### 目的

开发一种新的框架，能够有效建模图中的长程依赖关系，同时保持较低的 computational cost 和良好的可扩展性。

### 方法

提出拓扑注意力，一种概率机制，学习信息应该如何通过图中的直接和间接连接流动。与传统注意力不同，拓扑注意力是从图中学习的信息传播中涌现的，能够实现对局部和全局关系的统一推理。

### 主要发现

拓扑注意力网络在所有测量的基线模型上提供了最先进的性能。

### 结论

拓扑注意力网络是一种有效的方法，可以克服传统GNNs在建模长程依赖关系方面的局限性，同时保持计算效率和可扩展性。

### 翻译

图神经网络依赖于局部消息传递，这限制了它们建模图中长程依赖关系的能力。现有的方法通过连续时间动力学或密集自注意力来扩展这一范围，但两者都存在高计算成本和可扩展性有限的问题。我们提出了拓扑注意力网络，这是一种新框架，应用了拓扑注意力——一种概率机制，学习信息应该如何通过图中的直接和间接连接流动。与依赖于显式成对交互的传统注意力不同，拓扑注意力是从图中学习的信息传播中涌现的，能够实现对局部和全局关系的统一推理。该方法在所有测量的基线模型上提供了最先进的性能。我们的实现可在 https://github.com/Marshall-Rosenhooper/Topologic-Attention-Networks 获取。


### 论文摘要

Graph Neural Networks rely on local message passing, which limits their ability to model long-range dependencies in graphs. Existing approaches extend this range through continuous-time dynamics or dense self-attention, but both suffer from high computational cost and limited scalability. We propose Topologic Attention Networks, a new framework that applies topologic attention, a probabilistic mechanism that learns how information should flow through both direct and indirect connections in a graph. Unlike conventional attention that depends on explicit pairwise interactions, topologic attention emerges from the learned information propagation of the graph, enabling unified reasoning over local and global relationships. This method achieves provides state-of-the-art performance across all measured baseline models. Our implementation is available at https://github.com/Marshall-Rosenhoover/Topologic-Attention-Networks.

---

## 124. When Structure Doesn't Help: LLMs Do Not Read Text-Attributed Graphs as Effectively as We Expected

**论文链接:** [http://arxiv.org/abs/2511.16767v1](http://arxiv.org/abs/2511.16767v1)

**作者:** Haotian Xu, Yuning You, Tengfei Ma

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究探讨了图结构编码策略对大语言模型(LLM)在文本属性图上性能的影响，发现仅利用节点文本描述的LLM已经能实现强大性能，而大多数结构编码策略收益甚微甚至有害，挑战了传统图学习范式中结构先验的必要性。

### 背景

图能够统一表示语义内容和关系结构，适用于分子建模、引用网络和社会图等领域。大语言模型在理解自然语言和整合跨模态信号方面表现出色，引发了对它们在图推理中应用的兴趣。现有研究主要通过模板设计或图神经网络(GNNs)来编码结构信息。

### 目的

调查不同的图结构编码策略如何影响大语言模型在文本属性图上的性能，评估结构信息对LLM图推理的必要性。

### 方法

进行系统实验，比较不同图结构编码策略下LLM在文本属性图任务上的表现，包括仅使用节点文本描述的策略和结合显式结构信息的策略。

### 主要发现

(i) 仅利用节点文本描述的LLM已经在各种任务上实现了强大的性能；(ii) 大多数结构编码策略带来的收益很小，甚至有负面影响，表明显式的结构先验在强大的语言模型参与时通常是不必要的，有时甚至会产生反效果。

### 结论

这一发现代表了对传统图学习范式的重要偏离，突显了在LLM时代需要重新思考如何表示和利用结构。研究系统地挑战了'结构对基于LLM的图推理本质上有益'这一基础假设，为图学习开辟了新的、语义驱动的方法。

### 翻译

图提供了语义内容和关系结构的统一表示，使它们成为分子建模、引用网络和社会图等领域的自然选择。同时，大语言模型(LLMs)在理解自然语言和整合跨模态信号方面表现出色，引发了人们对它们在图推理中潜力的兴趣。最近的工作通过设计基于模板的图模板或使用图神经网络(GNNs)来编码结构信息来探索这一点。在本研究中，我们调查了不同的图结构编码策略如何影响LLM在文本属性图上的性能。令人惊讶的是，我们的系统实验揭示：(i) 仅利用节点文本描述的LLM已经在各种任务上实现了强大性能；(ii) 大多数结构编码策略带来的收益很小，甚至是负面的。我们表明，当涉及强大的语言模型时，显式的结构先验通常是不必要的，在某些情况下甚至会产生反效果。这代表了对传统图学习范式的重要偏离，并突显了在LLM时代需要重新思考如何表示和利用结构。我们的研究系统地挑战了'结构对基于LLM的图推理本质上有益'这一基础假设，为图学习开辟了新的、语义驱动的方法。


### 论文摘要

Graphs provide a unified representation of semantic content and relational structure, making them a natural fit for domains such as molecular modeling, citation networks, and social graphs. Meanwhile, large language models (LLMs) have excelled at understanding natural language and integrating cross-modal signals, sparking interest in their potential for graph reasoning. Recent work has explored this by either designing template-based graph templates or using graph neural networks (GNNs) to encode structural information. In this study, we investigate how different strategies for encoding graph structure affect LLM performance on text-attributed graphs. Surprisingly, our systematic experiments reveal that: (i) LLMs leveraging only node textual descriptions already achieve strong performance across tasks; and (ii) most structural encoding strategies offer marginal or even negative gains. We show that explicit structural priors are often unnecessary and, in some cases, counterproductive when powerful language models are involved. This represents a significant departure from traditional graph learning paradigms and highlights the need to rethink how structure should be represented and utilized in the LLM era. Our study is to systematically challenge the foundational assumption that structure is inherently beneficial for LLM-based graph reasoning, opening the door to new, semantics-driven approaches for graph learning.

---

## 125. MDG: Masked Denoising Generation for Multi-Agent Behavior Modeling in Traffic Environments

**论文链接:** [http://arxiv.org/abs/2511.17496v1](http://arxiv.org/abs/2511.17496v1)

**作者:** Zhiyu Huang, Zewei Zhou, Tianhui Cai, Yun Zhang, Jiaqi Ma

**发布时间:** 2025-11-21

### GPT解析

### 总结

论文提出了Masked Denoising Generation (MDG)，一种统一的多智能体行为建模框架，通过连续的按智能体和按时间步的噪声掩码实现高效、一致和可控的轨迹生成。

### 背景

真实和交互式多智能体行为建模对自动驾驶和交通仿真至关重要，但现有方法存在效率低和重用性差的问题。

### 目的

开发一个统一的生成框架，克服现有扩散和自回归方法的局限性，提高效率和重用性。

### 方法

提出Masked Denoising Generation (MDG)，将多智能体行为建模重新表述为对独立噪声化的时空张量的重构，应用连续的、按智能体和按时间步的噪声掩码。

### 主要发现

MDG能够在单次或几次前向传递中实现局部去噪和可控轨迹生成，并能在一个模型中概括开环预测、闭环仿真、运动规划和条件生成。

### 结论

MDG在Waymo Sim Agents和nuPlan Planning基准测试上实现了具有竞争力的闭环性能，同时提供了高效、一致和可控的开环多智能体轨迹生成，成为多智能体行为建模的一种简单而多功能的范式。

### 翻译

对自动驾驶和交通仿真中的多智能体行为进行真实且交互式的建模至关重要。然而，现有的扩散和自回归方法受限于迭代采样、顺序解码或特定任务设计，这阻碍了效率和重用。我们提出了掩码去噪生成（MDG），一种统一的生成框架，将多智能体行为建模重新表述为对独立噪声化的时空张量的重构。MDG不依赖于扩散时间步或离散标记化，而是应用连续的、按智能体和按时间步的噪声掩码，使在单次或几次前向传递中实现局部去噪和可控轨迹生成成为可能。这种掩码驱动的表述在一个模型内概括了开环预测、闭环仿真、运动规划和条件生成。在大型真实驾驶数据集上训练后，MDG在Waymo Sim Agents和nuPlan Planning基准测试上实现了具有竞争力的闭环性能，同时提供了高效、一致和可控的开环多智能体轨迹生成。这些结果将MDG定位为多智能体行为建模的一种简单而多功能的范式。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文旨在解决现有多智能体行为建模方法（特别是扩散模型和自回归方法）的效率、一致性和可控性问题。这些问题在自动驾驶和交通模拟中至关重要，因为准确且可控的行为生成支持从运动预测到交通模拟和闭环规划等多种下游任务。现有方法的局限性阻碍了模型在不同任务间的泛化和重用，影响了可扩展自主系统的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：扩散模型需要迭代采样，自回归方法需要顺序解码且难以处理连续动态。他们设计了一个统一的生成框架，将多智能体行为建模重新表述为独立加噪时空张量的重建。该方法借鉴了扩散模型的基本思想，但摒弃了迭代过程；参考了掩码生成建模在语言、视频和图像领域的应用；扩展了Diffusion Forcing的每个元素独立噪声水平思想；并改进了掩码离散扩散，使用连续高斯损坏代替离散掩码令牌。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将多智能体轨迹生成重新表述为独立加噪时空张量的重建，每个智能体-时间位置有独立的掩码强度，支持局部去噪和有条件生成。整体流程包括：1)掩码驱动的损坏过程，为每个元素分配噪声水平并应用高斯噪声；2)通过掩码去噪生成，从完全加噪状态开始逐步减少噪声；3)模型结构包含场景编码器和去噪器，处理多模态场景上下文和噪声轨迹；4)训练时使用自适应掩蔽策略联合优化去噪和预测损失；5)推理支持多种模式，如单步去噪、沿时间/智能体轴去噪、长程指导和闭环重用。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的生成框架，无需任务特定设计；2)连续、结构化的噪声掩码场，支持局部去噪和条件生成；3)高效的单步或少量前向传播生成；4)灵活的推理模式，适应不同下游任务。相比之前工作：不同于扩散模型的迭代去噪，MDG在单步中生成完整序列；区别于自回归模型的顺序解码，MDG处理连续动态；扩展了Diffusion Forcing，将其推广到结构化时空张量；改进了掩码离散扩散，使用连续高斯损坏而非离散令牌。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MDG提出了一种掩码去噪生成框架，通过连续、结构化的噪声掩码实现了高效、一致且可控的多智能体行为建模，统一了开放循环预测、闭环模拟、运动规划和条件生成等多种任务。'}


### 论文摘要

Modeling realistic and interactive multi-agent behavior is critical to autonomous driving and traffic simulation. However, existing diffusion and autoregressive approaches are limited by iterative sampling, sequential decoding, or task-specific designs, which hinder efficiency and reuse. We propose Masked Denoising Generation (MDG), a unified generative framework that reformulates multi-agent behavior modeling as the reconstruction of independently noised spatiotemporal tensors. Instead of relying on diffusion time steps or discrete tokenization, MDG applies continuous, per-agent and per-timestep noise masks that enable localized denoising and controllable trajectory generation in a single or few forward passes. This mask-driven formulation generalizes across open-loop prediction, closed-loop simulation, motion planning, and conditional generation within one model. Trained on large-scale real-world driving datasets, MDG achieves competitive closed-loop performance on the Waymo Sim Agents and nuPlan Planning benchmarks, while providing efficient, consistent, and controllable open-loop multi-agent trajectory generation. These results position MDG as a simple yet versatile paradigm for multi-agent behavior modeling.

---

## 126. Emergence of Randomness in Temporally Aggregated Financial Tick Sequences

**论文链接:** [http://arxiv.org/abs/2511.17479v1](http://arxiv.org/abs/2511.17479v1)

**作者:** Silvia Onofri, Andrey Shternshis, Stefano Marmi

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究提出了一种新方法来评估超高频金融数据的随机性，发现随着时间聚合水平的增加，金融tick数据的随机性程度也增加，并发现了某些资产预测性的非单调行为。

### 背景

市场效率意味着股票收益本质上不可预测，这使得市场类似于随机数生成器。

### 目的

开发一种新方法来研究超高频金融数据并评估逐笔回报与随机序列的相似程度。

### 方法

应用全面的随机性测试集（包括NIST统计测试套件和TestU01测试套件中的Rabbit和Alphabit子测试）来分析超高频股票市场数据，而不仅仅依赖传统的序列相关或熵度量。

### 主要发现

时间聚合可以将高度相关的高频交易数据转化为随机流，随着交易时间聚合水平的增加，金融tick数据的随机性程度也增加。此外，测试还发现了某些资产预测性的非单调行为。

### 结论

该研究提出了一种无需模型的方法来评估金融时间序列中的随机性并从中生成伪随机序列，在多个应用中具有潜在相关性。

### 翻译

市场效率意味着股票回报本质上不可预测，这一特性使市场类似于随机数生成器。我们提出了一种新方法来研究超高频金融数据，并评估逐笔回报与随机序列的相似程度。我们通过应用全面的随机性测试集扩展了对超高频股票市场数据的分析，超越了通常对序列相关或熵度量的依赖。我们的目的是使用标准测试套件中评估随机性不同方面的统计测试，广泛分析这些数据的随机性。我们说明了时间聚合在将高度相关的高频交易数据转换为随机流方面的作用。更具体地说，我们使用NIST统计测试套件和TestU01测试套件中的许多测试（特别是Rabbit和Alphabit子测试），证明金融tick数据的随机性程度随着交易时间聚合水平的增加而增加。此外，我们测试的全面性还发现了新的模式，例如某些资产预测性的非单调行为。该研究展示了一种无需模型的方法，用于评估金融时间序列中的随机性并从中生成伪随机序列，在多个应用中具有潜在的相关性。


### 论文摘要

Markets efficiency implies that the stock returns are intrinsically unpredictable, a property that makes markets comparable to random number generators. We present a novel methodology to investigate ultra-high frequency financial data and to evaluate the extent to which tick by tick returns resemble random sequences. We extend the analysis of ultra high-frequency stock market data by applying comprehensive sets of randomness tests, beyond the usual reliance on serial correlation or entropy measures. Our purpose is to extensively analyze the randomness of these data using statistical tests from standard batteries that evaluate different aspects of randomness.   We illustrate the effect of time aggregation in transforming highly correlated high-frequency trade data to random streams. More specifically, we use many of the tests in the NIST Statistical Test Suite and in the TestU01 battery (in particular the Rabbit and Alphabit sub-batteries), to prove that the degree of randomness of financial tick data increases together with the increase of the aggregation level in transaction time. Additionally, the comprehensive nature of our tests also uncovers novel patterns, such as non-monotonic behaviors in predictability for certain assets. This study demonstrates a model-free approach for both assessing randomness in financial time series and generating pseudo-random sequences from them, with potential relevance in several applications.

---

## 127. Planning with Sketch-Guided Verification for Physics-Aware Video Generation

**论文链接:** [http://arxiv.org/abs/2511.17450v1](http://arxiv.org/abs/2511.17450v1)

**作者:** Yidong Huang, Zun Wang, Han Lin, Dong-Ki Kim, Shayegan Omidshafiei, Jaehong Yoon, Yue Zhang, Mohit Bansal

**发布时间:** 2025-11-21

**备注:** website: https://sketchverify.github.io/

### GPT解析

### 总结

SketchVerify是一种无需训练、基于草图验证的视频生成运动规划框架，通过测试时采样和验证循环提高运动规划质量，在保证物理合理性和指令一致性的同时显著提升计算效率。

### 背景

现有视频生成方法主要依赖规划中间控制信号（如物体轨迹）来提高时间一致性和运动保真度，但这些方法要么使用单次计划（仅适用于简单运动），要么需要多次迭代调用视频生成器（计算成本高）。

### 目的

克服现有方法的局限性，提出一种无需训练的规划框架，在完整视频生成前生成更动态连贯的轨迹，提高运动规划质量。

### 方法

SketchVerify框架通过预测多个候选运动计划并使用视觉语言验证器进行排序（评估语义一致性和物理合理性），将每个轨迹渲染为轻量级视频草图以避免昂贵的重复扩散合成，迭代改进直至找到满意计划，最后传递给轨迹条件生成器进行最终合成。

### 主要发现

在WorldModelBench和PhyWorldBench上的实验表明，与竞争基线相比，该方法显著提高了运动质量、物理真实性和长期一致性，同时效率更高；增加轨迹候选数量可持续提高整体性能。

### 结论

SketchVerify框架能够在保证运动质量的同时大幅提高计算效率，为视频生成中的运动规划提供了有效解决方案。

### 翻译

近期的视频生成方法越来越依赖于规划中间控制信号（如物体轨迹）来提高时间一致性和运动保真度。然而，这些方法大多采用单次计划，通常仅限于简单运动，或需要迭代改进，这需要多次调用视频生成器，导致高计算成本。为克服这些局限性，我们提出了SketchVerify，一种无需训练、基于草图验证的规划框架，通过引入测试时采样和验证循环，在完整视频生成前提高运动规划质量，生成更动态连贯的轨迹（即物理上合理且与指令一致的运动）。给定提示和参考图像，我们的方法预测多个候选运动计划，并使用视觉语言验证器对其进行排序，该验证器同时评估与指令的语义一致性和物理合理性。为了高效评分候选运动计划，我们将每个轨迹渲染为轻量级视频草图，通过在静态背景上合成物体实现，这避免了昂贵的重复扩散合成，同时实现了可比的性能。我们迭代改进运动计划直到找到满意的计划，然后将其传递给轨迹条件生成器进行最终合成。在WorldModelBench和PhyWorldBench上的实验表明，与竞争基线相比，我们的方法显著提高了运动质量、物理真实性和长期一致性，同时效率更高。我们的消融研究进一步表明，增加轨迹候选数量可持续提高整体性能。


### 论文摘要

Recent video generation approaches increasingly rely on planning intermediate control signals such as object trajectories to improve temporal coherence and motion fidelity. However, these methods mostly employ single-shot plans that are typically limited to simple motions, or iterative refinement which requires multiple calls to the video generator, incuring high computational cost. To overcome these limitations, we propose SketchVerify, a training-free, sketch-verification-based planning framework that improves motion planning quality with more dynamically coherent trajectories (i.e., physically plausible and instruction-consistent motions) prior to full video generation by introducing a test-time sampling and verification loop. Given a prompt and a reference image, our method predicts multiple candidate motion plans and ranks them using a vision-language verifier that jointly evaluates semantic alignment with the instruction and physical plausibility. To efficiently score candidate motion plans, we render each trajectory as a lightweight video sketch by compositing objects over a static background, which bypasses the need for expensive, repeated diffusion-based synthesis while achieving comparable performance. We iteratively refine the motion plan until a satisfactory one is identified, which is then passed to the trajectory-conditioned generator for final synthesis. Experiments on WorldModelBench and PhyWorldBench demonstrate that our method significantly improves motion quality, physical realism, and long-term consistency compared to competitive baselines while being substantially more efficient. Our ablation study further shows that scaling up the number of trajectory candidates consistently enhances overall performance.

---

## 128. Minimalist machine-learned interatomic potentials can predict complex structural behaviors accurately

**论文链接:** [http://arxiv.org/abs/2511.17449v1](http://arxiv.org/abs/2511.17449v1)

**作者:** Iñigo Robredo-Magro, Binayak Mukherjee, Hugo Aramberri, Jorge Íñiguez-González

**发布时间:** 2025-11-21

**备注:** 13 pages, 5 figures

### GPT解析

### 总结

机器学习原子间势(MLIPs)在过去十年取得了显著发展，已成为原子模拟研究的首选方法。本研究挑战了MLIPs仅适用于插值计算的传统观点，使用极简训练集成功预测了非平凡结构效应，扩展了MLIPs的应用范围，表明简单模型可成为发现复杂物理现象的有效工具。

### 背景

过去十年，机器学习原子间势(MLIPs)取得了显著发展，已成为大多数不需要明确处理电子的原子模拟研究的首选方法。典型的MLIP使用指南强调需要全面的训练集，并警告不要将模型应用于训练空间中未考虑的情况。

### 目的

采用更乐观的观点，挑战两种代表性且广泛可用的MLIP方法的预测能力，探索MLIPs在发现新现象方面的潜力。

### 方法

使用极简训练集，这些训练集依赖于对所研究材料的先验知识较少。采用定义超参数的适度/默认选择，构建MLIP模型。

### 主要发现

这些模型在预测非平凡的结构效应（竞争多晶型、结构转变的能垒、非平凡拓扑的出现）方面非常成功，其预测在定性和准定量上都是正确的。

### 结论

研究结果扩展了现代MLIP方法的应用范围，表明有些简单且容易计算的模型可以成为发现新颖复杂物理现象的有效工具。

### 翻译

过去十年见证了机器学习原子间势(MLIPs)的显著发展，以至于它们已经成为大多数不需要明确处理电子的原子模拟研究的首选方法。典型的MLIP使用指南强调需要全面的训练集，并警告不要将模型应用于相应训练空间中未考虑的情况。这限制了MLIPs的范围仅限于插值计算，基本上否定了以偶然方式使用它们来发现新现象的可能性。虽然谨慎是有理由的，但这里我们采用更乐观的观点，挑战两种代表性且广泛可用的MLIP方法的预测能力。我们使用极简训练集，这些训练集依赖于对所研究材料的先验知识较少。我们表明，由此产生的模型——我们采用了定义超参数的适度/默认选择——在预测非平凡的结构效应（竞争多晶型、结构转变的能垒、非平凡拓扑的出现）方面非常成功，其预测在定性和准定量上都是正确的。我们的研究结果表明，现代MLIP方法的应用范围得到了扩展，证明有些简单且容易计算的模型可以成为发现新颖复杂物理现象的有效工具。


### 论文摘要

The past decade has witnessed a spectacular development of machine-learned interatomic potentials (MLIPs), to the extent that they are already the approach of choice for most atomistic simulation studies not requiring an explicit treatment of electrons. Typical MLIP usage guidelines emphasize the need for exhaustive training sets and warn against applying the models to situations not considered in the corresponding training space. This restricts the scope of MLIPs to interpolative calculations, essentially denying the possibility of using them to discover new phenomena in a serendipitous way. While there are reasons to be cautious, here we adopt a more sanguine view and challenge the predictive power of two representative and widely available MLIP approaches. We work with minimalist training sets that rely on little prior knowledge of the investigated materials. We show that the resulting models -- for which we adopt modest/default choices of the defining hyperparameters -- are very successful in predicting non-trivial structural effects (competing polymorphs, energy barriers for structural transformations, occurrence of non-trivial topologies) in a way that is qualitatively and quasi-quantitatively correct. Our results thus suggest an expanded scope of modern MLIP approaches, evidencing that somewhat trivial -- and easy to compute -- models can be an effective tool for the discovery of novel and complex physical phenomena.

---

## 129. Circulation of Elites in an Adaptive Network Model

**论文链接:** [http://arxiv.org/abs/2511.17434v1](http://arxiv.org/abs/2511.17434v1)

**作者:** Alexander Jochim, Stefan Bornholdt

**发布时间:** 2025-11-21

**备注:** 6 pages, 6 figures

### GPT解析

### 总结

本研究探讨了社会精英结构动力学如何影响长期政治行为，通过引入一个基于局部规则的自适应网络模型，研究了政治权力的传递和竞争性政治理念的扩散。

### 背景

社会在历史上经历政治稳定和不稳定的阶段，政治权力通常通过这些变化传递给新的精英群体。社会精英的结构动力学被认为是塑造长期行为的核心驱动因素之一。然而，当前模型和数据主要是宏观层面的，从微观动力学涌现宏观行为的过程尚不清楚。

### 目的

旨在通过一个基于局部动力学规则的自适应网络模型，理解从微观动力学到宏观政治行为的涌现过程，特别是政治权力的累积优势效应和精英内部冲突如何影响社会政治动态。

### 方法

引入了一个自适应网络模型，使用有向链接代表政治权力和竞争性政治理念，基于两种社会驱动行为：政治权力的累积优势效应和精英内部冲突。

### 主要发现

1. 观察到断续平衡作为一种涌现行为；2. 发现向无序相的相变；3. 定义了精英崩溃的前期预警指标；4. 发现只有少数最大节点的状态适合作为具有预测信息的代理。

### 结论

通过基于局部规则的自适应网络模型，揭示了社会精英结构动力学如何导致宏观政治行为的涌现，包括断续平衡和相变，并提供了精英崩溃的预测指标。

### 翻译

社会在历史上经历政治稳定和不稳定的阶段，而政治权力通常通过这些变化传递给新的精英群体。社会精英的结构动力学被认为是塑造长期行为的核心驱动因素之一。由于当前模型和数据主要是宏观层面的，从微观动力学涌现宏观行为的过程尚不清楚。在这里，我们引入了一个基于局部动力学规则的自适应网络模型，使用有向链接代表政治权力和竞争性政治理念。该模型基于两种社会驱动行为：政治权力的累积优势效应和精英内部冲突。我们观察到断续平衡作为一种涌现行为，并发现向无序相的相变。我们定义了精英崩溃的前期预警指标，并发现只有少数最大节点的状态适合作为具有预测信息的代理。


### 论文摘要

Societies experience politically stable and unstable phases along history, whereas political power is usually passed to new elite groups by these changes. Structural dynamics of the elites in a society have been proposed to be one of the core drivers shaping long term behavior. As current models and data are rather macroscopic, the emergence of macroscopic behavior from microscopic dynamics is largely unclear. Here, we introduce an adaptive network model of directed links representing political power and competing political ideas, based on local dynamical rules, only. The model is based on two socially motivated behaviors: the cumulative advantage effect of political power and intra-elite conflict. We observe punctuated equilibria as an emergent behavior and find a phase transition towards a disordered phase. We define an advance warning measure for elite collapse and find that the states of only a few largest nodes are suitable as a proxy with predictive information.

---

## 130. Algorithmic design and implementation considerations of deep MPC

**论文链接:** [http://arxiv.org/abs/2511.17233v1](http://arxiv.org/abs/2511.17233v1)

**作者:** Prabhat K. Mishra, Mateus V. Gasparino, Girish Chowdhary

**发布时间:** 2025-11-21

### GPT解析

### 总结

这篇论文讨论深度模型预测控制(Deep MPC)，这是一种结合模型预测控制和深度学习的新兴领域，特别关注将深度神经网络与MPC控制器结合使用的方法。

### 背景

Deep MPC是一个不断发展的领域，整合了模型预测控制和深度学习两种技术。

### 目的

解释深度MPC的实现挑战，介绍如何分配控制权限，并论证控制权限分配不当可能导致性能不佳的问题。

### 方法

将控制权限分配给神经网络和MPC控制器，其中神经网络学习模型不确定性，MPC处理约束。优势在于可以使用系统运行时收集的训练数据微调神经网络，同时MPC在学习过程中防止不安全行为。

### 主要发现

控制权限分配不当可能导致性能不佳，这一点通过四轮滑转转向动力学的数值实验得到了验证。

### 结论

在深度MPC中合理分配控制权限对系统性能至关重要。

### 翻译

深度模型预测控制(Deep MPC)是一个不断发展的领域，它整合了模型预测控制和深度学习。本文专注于一种特定方法，该方法将深度神经网络与MPC结合使用。这类方法将控制权限分配给神经网络和MPC控制器，使得神经网络学习模型不确定性，而MPC处理约束。这种方法具有吸引力，因为可以在系统运行时收集的训练数据用于微调神经网络，同时MPC可以防止学习过程中的不安全行为。本文解释了深度MPC的实现挑战，介绍了分配控制权限的算法方法，并论证了控制权限分配不当可能导致性能不佳。通过一个关于四轮滑转转向动力学的数值实验，解释了性能不佳的原因。


### 论文摘要

Deep Model Predictive Control (Deep MPC) is an evolving field that integrates model predictive control and deep learning. This manuscript is focused on a particular approach, which employs deep neural network in the loop with MPC. This class of approaches distributes control authority between a neural network and an MPC controller, in such a way that the neural network learns the model uncertainties while the MPC handles constraints. The approach is appealing because training data collected while the system is in operation can be used to fine-tune the neural network, and MPC prevents unsafe behavior during those learning transients. This manuscript explains implementation challenges of Deep MPC, algorithmic way to distribute control authority and argues that a poor choice in distributing control authority may lead to poor performance. A reason of poor performance is explained through a numerical experiment on a four-wheeled skid-steer dynamics.

---

## 131. Collective early-time spontaneous decay of a strongly driven cold atomic ensemble

**论文链接:** [http://arxiv.org/abs/2511.17187v1](http://arxiv.org/abs/2511.17187v1)

**作者:** Daniel Benedicto Orenes, Naudson Lucas Lopes Matias, Apoorva Apoorva, Antoine Glicenstein, Raphaël Saint-Jalm, Robin Kaiser

**发布时间:** 2025-11-21

### GPT解析

### 总结

本研究对强驱动和光学稠密冷原子云的集体早期衰减率进行了数值和实验研究

### 背景

研究通过不同的Rabi频率(从弱驱动到强驱动)将原子系综驱动到稳态，其中Γ代表单原子衰减率

### 目的

调查强驱动和弱驱动之间转变过程中的早期动力学

### 方法

使用角度相关观测(如云发射的光)和全局观测(激发态布居)两种方法

### 主要发现

在共振驱动云时，作为驱动频率的函数，在某些角度收集到的光的行为从单光子亚辐射转变为超辐射，而激发态布居的行为没有显示出超辐射

### 结论

实验结果在所研究的参数范围内与数值预测一致

### 翻译

本研究中，我们提出了对强驱动和光学稠密冷原子云的集体早期衰减率的数值和实验研究。我们通过使用不同的Rabi频率(Ω)将系统驱动到稳态来制备原子系综，这些频率从弱驱动(Ω远小于Γ)到强驱动(Ω远大于Γ)不等，其中Γ是单原子衰减率。我们使用以下方法调查了强驱动和弱驱动之间转变过程中的早期动力学：i) 角度相关观测，如云发射的光；ii) 全局观测，即激发态布居。当在共振驱动云时，我们发现作为驱动频率的函数，在某些角度收集到的光的行为从单光子亚辐射转变为超辐射，而激发态布居的行为没有显示出超辐射。实验结果在所研究的参数范围内与数值预测一致。


### 论文摘要

In this work we present a numerical and experimental investigation of the collective early-time decay rates of a strongly driven and optically dense cold atomic cloud. We prepare the atomic ensemble by driving the system to its steady state with varying Rabi frequencies $Ω$ that go from the weak $Ω\ll Γ$ to the strong driving regime $Ω\gg Γ$, where $Γ$ is the single-atom decay rate. We investigate the early-time dynamics in the transition between the strong and weak driving regimes using: i) angular-dependent observables such as the light emitted by the cloud, and ii) global observables, i.e., the excited state population. When driving the cloud on-resonance, we find that as a function of the driving frequency, the behavior of the collected light at certain angles transitions from the single-photon subradiant regime to a superradiant regime while the behavior of the excited state population does not show superradiance. The experiment shows good agreement with numerical predictions in the regime of parameters under study.

---

## 132. DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.17150v1](http://arxiv.org/abs/2511.17150v1)

**作者:** Liuhan Yin, Runkun Ju, Guodong Guo, Erkang Cheng

**发布时间:** 2025-11-21

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

本研究提出了一种名为DiffRefiner的新型两阶段轨迹预测框架，结合判别式轨迹提案模块和生成式扩散模型，实现了最先进的轨迹预测性能，在多个公共基准测试上创造了新记录。

### 背景

自主驾驶中，判别式方法预测自车固定候选轨迹集，而生成式方法(如扩散模型)学习未来运动的潜在分布，提供更灵活的轨迹预测。然而，这些方法通常依赖人工设计的轨迹锚点或随机噪声进行去噪，仍有改进空间。

### 目的

提出DiffRefiner框架，提高基于扩散的规划性能，通过增强与周围环境的对齐，实现更准确的轨迹预测。

### 方法

DiffRefiner采用两阶段框架：第一阶段使用基于Transformer的提案解码器从传感器输入回归并使用预定义轨迹锚点生成粗略轨迹预测；第二阶段应用扩散精炼器迭代去噪和精炼初始预测。同时设计细粒度去噪解码器增强场景适应性，并结合判别式轨迹提案模块为生成式精炼过程提供强引导。

### 主要发现

DiffRefiner在NAVSIM v2上达到87.4 EPDMS，在Bench2Drive上达到87.1 DS和71.4 SR，在两个公共基准测试上都创造了新记录。消融研究验证了每个组件的有效性。

### 结论

通过结合判别式轨迹提案模块和生成式扩散模型，DiffRefiner实现了最先进的轨迹预测性能，证明了该方法在自主驾驶轨迹预测中的有效性。

### 翻译

与自主驾驶中预测自车固定候选轨迹的判别式方法不同，生成式方法(如扩散模型)学习未来运动的潜在分布，能够实现更灵活的轨迹预测。然而，由于这些方法通常依赖于去噪人工设计的轨迹锚点或随机噪声，仍有显著的改进空间。在本文中，我们提出了DiffRefiner，一种新颖的两阶段轨迹预测框架。第一阶段使用基于Transformer的提案解码器，通过从传感器输入回归并使用预定义轨迹锚点生成粗略轨迹预测。第二阶段应用扩散精炼器，迭代去噪和精炼这些初始预测。通过这种方式，我们通过整合提供强引导的判别式轨迹提案模块，提高了基于扩散的规划性能。此外，我们设计了细粒度去噪解码器以增强场景适应性，通过与周围环境增强对齐实现更准确的轨迹预测。实验结果表明，DiffRefiner实现了最先进的性能，在NAVSIM v2上达到87.4 EPDMS，在Bench2Drive上达到87.1 DS和71.4 SR，从而在两个公共基准测试上都创造了新记录。每个组件的有效性也通过消融研究得到了验证。


### 论文摘要

Unlike discriminative approaches in autonomous driving that predict a fixed set of candidate trajectories of the ego vehicle, generative methods, such as diffusion models, learn the underlying distribution of future motion, enabling more flexible trajectory prediction. However, since these methods typically rely on denoising human-crafted trajectory anchors or random noise, there remains significant room for improvement. In this paper, we propose DiffRefiner, a novel two-stage trajectory prediction framework. The first stage uses a transformer-based Proposal Decoder to generate coarse trajectory predictions by regressing from sensor inputs using predefined trajectory anchors. The second stage applies a Diffusion Refiner that iteratively denoises and refines these initial predictions. In this way, we enhance the performance of diffusion-based planning by incorporating a discriminative trajectory proposal module, which provides strong guidance for the generative refinement process. Furthermore, we design a fine-grained denoising decoder to enhance scene compliance, enabling more accurate trajectory prediction through enhanced alignment with the surrounding environment. Experimental results demonstrate that DiffRefiner achieves state-of-the-art performance, attaining 87.4 EPDMS on NAVSIM v2, and 87.1 DS along with 71.4 SR on Bench2Drive, thereby setting new records on both public benchmarks. The effectiveness of each component is validated via ablation studies as well.

---

## 133. Magnetized particle motion and accretion process with shock cone morphology around a decoupled hairy black holes

**论文链接:** [http://arxiv.org/abs/2511.17137v1](http://arxiv.org/abs/2511.17137v1)

**作者:** G. Mustafa, Faisal Javed, S. K. Maurya, A. Ditta, Orhan Donmez, Tayyab Naseer, Abdelmalek Bouzenada, Farruh Atamurotov

**发布时间:** 2025-11-21

**备注:** 28 pages, 25 figures

### GPT解析

### 总结

本研究通过扩展几何变形研究了具有'毛发'参数的黑洞周围的磁化粒子运动和相对论性吸积过程，发现这种黑洞模型在能量效率和辐射率方面优于标准黑洞，并为未来引力波和X射线天文观测提供了明显的观测特征。

### 背景

相对论性吸积是已知的最有效的将引力势能转化为辐射的机制之一，发生在黑洞和中子星等致密天体上。对于快速旋转的黑洞，最多可以释放吸积物质静质量能量的40%，远超核聚变的效率。

### 目的

通过扩展几何变形研究解耦'毛发'黑洞周围的磁化粒子运动和相对论性吸积过程，探索其能量效率和观测特征。

### 方法

发展了一种包含两个'毛发'参数的几何结构，这些参数保留了视界结构并在事件视界外满足弱能量条件。为磁化粒子运动提供必要的形式论基础，推导有效势和最内稳定圆轨道，并获得径向速度剖面、质量吸积率等的精确解析表达式。

### 主要发现

1) 在'毛发'参数作用下，磁化粒子的最内稳定圆轨道半径显著减小；2) 与标准黑洞相比，该模型显示出改进的能量效率和辐射率；3) 解耦参数对振荡有强烈影响；4) 分析预测与数值模拟之间表现出极好的一致性。

### 结论

具有'毛发'参数的黑洞模型在吸积效率和观测特征方面优于标准黑洞模型，为未来引力波和X射线天文观测提供了新的可能性和明显的观测特征。

### 翻译

相对论性吸积到黑洞和中子星等致密天体上是已知的最有效的将引力势能转化为辐射的机制之一。对于快速旋转的黑洞，最多可以释放吸积物质静质量能量的40%，远超核聚变的效率。在本工作中，我们通过扩展几何变形研究了围绕解耦'毛发'黑洞的磁化粒子运动和相对论性吸积过程。所发展的几何结构包含两个'毛发'参数，它们保留了视界结构，并在事件视界外满足弱能量条件。我们为围绕解耦黑洞的磁化粒子运动提供了必要的形式论基础。随后，我们推导出有效势和最内稳定圆轨道，结果表明在'毛发'参数作用下，磁化粒子的最内稳定圆轨道半径显著减小。之后，我们获得了径向速度剖面、质量吸积率等的精确解析表达式，与标准黑洞相比，这些结果显示出改进的能量效率和辐射率。此外，解耦参数对振荡有强烈影响，吸积过程表现出分析预测与数值模拟之间极好的一致性，从而为未来的引力波和X射线天文学提供了明显的观测特征。


### 论文摘要

Relativistic accretion onto compact objects such as black holes and neutron stars is one of the most efficient known mechanisms for converting gravitational potential energy into radiation. In the case of rapidly spinning black holes, up to $40\%$ of the rest-mass energy of accreting matter can be released, far exceeding the efficiency of nuclear fusion. In this work, we investigate magnetized particle motion and relativistic accretion processes around a decoupled hairy black hole via extended geometric deformation. The developed geometry involves two hairy parameters that preserve the horizon structure with the additional feature of the fulfillment of weak energy conditions outside the event horizon. We provide the foundation with necessary formalism for magnetized particle motion around a decoupled black hole. The effective potential and innermost stable circular orbits are then derived, which demonstrate a significant reduction of the radius of the latter quantity under the hairy parameters for the magnetized particle. Afterwards, we obtain exact analytical expressions for radial velocity profiles, mass accretion rates, and a few others which reveal improved energy efficiency and emissivity as compared to the standard black hole. Furthermore, the decoupling parameter shows strong influence on oscillations, accretion presenting fantastic agreement between analytical predictions and numerical simulations, and thus offering noticeable observational signatures for future gravitational wave and X-ray astronomy.

---

## 134. RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis

**论文链接:** [http://arxiv.org/abs/2511.17045v1](http://arxiv.org/abs/2511.17045v1)

**作者:** Linfeng Dong, Yuchen Yang, Hao Wu, Wei Wang, Yuenan HouZhihang Zhong, Xiao Sun

**发布时间:** 2025-11-21

**备注:** Accepted to AAAI 2026 (Oral)

### GPT解析

### 总结

RacketVision是一个新的数据集和基准，用于推进体育分析中的计算机视觉研究，涵盖乒乓球、网球和羽毛球。

### 背景

目前缺乏针对球拍类运动的细粒度计算机视觉数据集，特别是同时包含球拍姿态和球位置的大规模标注数据。

### 目的

设计一个数据集来解决三个相互关联的任务：细粒度球跟踪、关节式球拍姿态估计和预测性球轨迹预测。

### 方法

评估了现有基线，并发现对于多模态融合，简单地将球拍姿态特征连接起来会降低性能，而使用CrossAttention机制对于释放这些特征的价值至关重要。

### 主要发现

使用CrossAttention机制进行轨迹预测的结果超越了强大的单模态基线。

### 结论

RacketVision为未来在动态物体跟踪、条件运动预测和体育多模态分析方面的研究提供了多功能资源和强有力的起点。

### 翻译

我们引入了RacketVision，这是一个新颖的数据集和基准，用于推进体育分析中的计算机视觉研究，涵盖乒乓球、网球和羽毛球。该数据集首次提供了大规模、细粒度的球拍姿态标注，以及传统的球位置标注，使研究复杂的人-物体交互成为可能。它旨在解决三个相互关联的任务：细粒度球跟踪、关节式球拍姿态估计和预测性球轨迹预测。我们对现有基线的评估揭示了多模态融合的关键见解：虽然简单连接球拍姿态特征会降低性能，但CrossAttention机制对于释放这些特征的价值至关重要，从而产生了超越强大单模态基线的轨迹预测结果。RacketVision为未来在动态物体跟踪、条件运动预测和体育多模态分析方面的研究提供了多功能资源和强有力的起点。项目页面位于https://github.com/OrcustD/RacketVision


### 论文摘要

We introduce RacketVision, a novel dataset and benchmark for advancing computer vision in sports analytics, covering table tennis, tennis, and badminton. The dataset is the first to provide large-scale, fine-grained annotations for racket pose alongside traditional ball positions, enabling research into complex human-object interactions. It is designed to tackle three interconnected tasks: fine-grained ball tracking, articulated racket pose estimation, and predictive ball trajectory forecasting. Our evaluation of established baselines reveals a critical insight for multi-modal fusion: while naively concatenating racket pose features degrades performance, a CrossAttention mechanism is essential to unlock their value, leading to trajectory prediction results that surpass strong unimodal baselines. RacketVision provides a versatile resource and a strong starting point for future research in dynamic object tracking, conditional motion forecasting, and multimodal analysis in sports. Project page at https://github.com/OrcustD/RacketVision

---

## 135. MfNeuPAN: Proactive End-to-End Navigation in Dynamic Environments via Direct Multi-Frame Point Constraints

**论文链接:** [http://arxiv.org/abs/2511.17013v1](http://arxiv.org/abs/2511.17013v1)

**作者:** Yiwen Ying, Hanjing Ye, Senzi Luo, Luyao Liu, Yu Zhan, Li He, Hong Zhang

**发布时间:** 2025-11-21

**备注:** 6 pages, 9 figures, accepted at IEEE ROBIO 2025

### GPT解析

### 总结

本文提出了一种利用多帧点约束的主动端到端导航框架，通过预测模块预测移动障碍物的未来路径，显著提高了机器人在未知动态环境中的导航鲁棒性和效率。

### 背景

在复杂和动态环境中进行障碍物避免是实时机器人导航的关键挑战。

### 目的

克服传统方法和基于学习方法在高度动态场景中的局限性，提高机器人导航的适应性和效率。

### 方法

提出一种利用多帧点约束（包括当前帧和专用模块预测的未来帧）的框架，通过预测模块基于多帧观测预测移动障碍物的未来路径，实现主动的端到端导航。

### 主要发现

主动规划能力显著提高了机器人在未知动态环境中的导航鲁棒性和效率。

### 结论

通过模拟和真实实验验证了该方法的有效性。

### 翻译

在复杂和动态环境中进行障碍物避免是实时机器人导航的关键挑战。基于模型和基于学习的方法在高度动态的场景中往往失败，因为传统方法假设环境是静态的，无法适应实时变化，而基于学习的方法依赖单帧观测进行运动约束估计，限制了它们的适应性。为了克服这些局限性，本文提出了一种新颖的框架，利用多帧点约束（包括专用模块预测的当前和未来帧）来实现主动的端到端导航。通过包含一个预测模块，该模块基于多帧观测预测移动障碍物的未来路径，我们的方法使机器人能够主动预测和避免潜在危险。这种主动规划能力显著提高了未知动态环境中的导航鲁棒性和效率。模拟和真实实验验证了我们方法的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人在复杂动态环境中进行实时导航时的障碍物避让问题。传统基于模型的方法假设环境静态，无法适应实时变化；而基于学习的方法依赖单帧观测，限制了适应性。这个问题在现实中至关重要，因为随着机器人在城市街道、购物中心等动态环境中的应用增加，机器人需要能够实时预测和避让移动障碍物，以确保安全高效的导航。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统方法和基于学习方法的局限性：传统方法无法快速重新规划路径应对环境变化，而学习方法对环境变化敏感且泛化能力有限。他们特别指出NeuPAN虽尝试通过直接从点云估计运动约束来缓解问题，但主要使用当前帧点，在处理动态障碍物时效果不佳。作者借鉴了NeuPAN的端到端框架，但增加了主动规划能力来处理动态物体。他们结合了DBSCAN聚类进行障碍物检测、卡尔曼滤波进行状态估计、高斯混合模型进行轨迹预测等技术，设计了多帧点约束方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用多帧点约束（包括当前帧和预测的未来帧）实现主动端到端导航，通过预测模块预测移动障碍物的未来路径，使机器人能够主动预期和避免潜在危险。整体流程分为三部分：1)感知模块处理点云数据，应用高斯滤波、DBSCAN聚类、最近邻匹配、卡尔曼滤波和指数平滑来估计障碍物运动状态；2)预测模块使用高斯混合模型(GMM)预测障碍物未来位置，生成虚拟障碍点；3)规划控制模块使用增强的NeuPAN框架，结合动态点云编码(DUNE)和神经正则化运动规划(NRMP)生成平滑的控制命令。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用多帧点约束而非仅单帧信息；2)专门预测模块预测障碍物未来路径；3)基于NeuPAN的端到端框架但增加主动规划能力；4)使用高斯混合模型(GMM)表示障碍物未来位置的概率分布。相比之前工作，它不依赖预定义环境模型，能处理高度动态环境；不依赖单帧观测，提高了对动态环境的适应性和泛化能力；相比NeuPAN，具有主动规划而非仅反应式避障能力，在高度动态场景中表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MfNeuPAN通过引入多帧点约束和障碍物预测模块，使机器人能够在动态环境中实现主动端到端导航，显著提高了导航的鲁棒性和效率。'}


### 论文摘要

Obstacle avoidance in complex and dynamic environments is a critical challenge for real-time robot navigation. Model-based and learning-based methods often fail in highly dynamic scenarios because traditional methods assume a static environment and cannot adapt to real-time changes, while learning-based methods rely on single-frame observations for motion constraint estimation, limiting their adaptability. To overcome these limitations, this paper proposes a novel framework that leverages multi-frame point constraints, including current and future frames predicted by a dedicated module, to enable proactive end-to-end navigation. By incorporating a prediction module that forecasts the future path of moving obstacles based on multi-frame observations, our method allows the robot to proactively anticipate and avoid potential dangers. This proactive planning capability significantly enhances navigation robustness and efficiency in unknown dynamic environments. Simulations and real-world experiments validate the effectiveness of our approach.

---

## 136. RadioKMoE: Knowledge-Guided Radiomap Estimation with Kolmogorov-Arnold Networks and Mixture-of-Experts

**论文链接:** [http://arxiv.org/abs/2511.16986v1](http://arxiv.org/abs/2511.16986v1)

**作者:** Fupei Guo, Kerry Pan, Songyang Zhang, Yue Wang, Zhi Ding

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为RadioKMoE的知识引导无线地图估计框架，结合Kolmogorov-Arnold Networks和Mixture-of-Experts技术，用于提高无线网络信号传播和覆盖范围估计的准确性和鲁棒性。

### 背景

Radiemap是无线网络管理和部署的重要工具，提供信号传播和覆盖的强大空间知识。然而，日益复杂的无线电传播行为和周围环境对无线地图估计(RME)提出了严峻挑战。

### 目的

开发一种新的无线地图估计方法，提高在复杂环境和多频段情况下的估计准确性和鲁棒性。

### 方法

提出RadioKMoE框架，包含KAN模块预测初始粗略覆盖地图，利用KAN在近似物理模型方面的优势；将初始粗略地图与环境信息一起输入MoE网络进行精确估计；MoE模块由处理不同无线地图模式的专家网络组成，提高局部细节同时保持全局一致性。

### 主要发现

在多频段和单频段无线地图估计实验中，RadioKMoE方法展示了增强的准确性和鲁棒性。

### 结论

RadioKMoE框架通过结合KAN和MoE的优势，有效解决了复杂环境下的无线地图估计挑战，提高了估计的准确性和鲁棒性。

### 翻译

Radiemap作为无线网络管理和部署的重要工具，通过提供信号传播和覆盖的强大空间知识发挥着关键作用。然而，日益复杂的无线电传播行为和周围环境对无线地图估计(RME)提出了严峻挑战。在这项工作中，我们提出了一种知识引导的RME框架，将Kolmogorov-Arnold Networks(KAN)与Mixture-of-Experts(MoE)相结合，即RadioKMoE。具体而言，我们设计了一个KAN模块来预测初始的粗略覆盖地图，利用KAN在近似物理模型和全局无线电传播模式方面的优势。初始粗略地图与环境信息一起驱动我们的MoE网络进行精确的无线地图估计。与传统深度学习模型不同，MoE模块由专门处理不同无线地图模式的专家网络组成，以提高局部细节同时保持全局一致性。在多频段和单频段RME实验中，结果表明所提出的RadioKMoE在无线地图估计中具有增强的准确性和鲁棒性。


### 论文摘要

Radiomap serves as a vital tool for wireless network management and deployment by providing powerful spatial knowledge of signal propagation and coverage. However, increasingly complex radio propagation behavior and surrounding environments pose strong challenges for radiomap estimation (RME). In this work, we propose a knowledge-guided RME framework that integrates Kolmogorov-Arnold Networks (KAN) with Mixture-of-Experts (MoE), namely RadioKMoE. Specifically, we design a KAN module to predict an initial coarse coverage map, leveraging KAN's strength in approximating physics models and global radio propagation patterns. The initial coarse map, together with environmental information, drives our MoE network for precise radiomap estimation. Unlike conventional deep learning models, the MoE module comprises expert networks specializing in distinct radiomap patterns to improve local details while preserving global consistency. Experimental results in both multi- and single-band RME demonstrate the enhanced accuracy and robustness of the proposed RadioKMoE in radiomap estimation.

---

## 137. Real Option AI: Reversibility, Silence, and the Release Ladder

**论文链接:** [http://arxiv.org/abs/2511.16958v1](http://arxiv.org/abs/2511.16958v1)

**作者:** I. Sebastian Buhai

**发布时间:** 2025-11-21

### GPT解析

### 总结

该研究将AI产品发布的节奏模式（静默期、可逆补丁和较少见的转向）建模为在声誉学习环境下战略实物期权的最优行使。企业通过控制两种升级选项（廉价补丁和高成本转向）以及发布频率时钟来优化信息披露策略，形成特定的市场信号模式。

### 背景

AI产品发布存在特定的节奏模式，包括静默期、可逆补丁和较少见的转向。这些发布模式反映了企业在技术状态不确定情况下的战略决策。

### 目的

研究AI产品发布的节奏模式，分析企业在不同技术状态下的最优发布策略，以及这些策略如何影响市场或平台采用。

### 方法

使用扩散模型描述私人观察到的技术状态，企业控制两种升级选项（廉价补丁和高成本转向）和发布频率时钟。采用Cox过程描述公共信息披露时机，在静态马尔可夫策略中分析最优发布阶梯，并将市场采用内化为公共信念中的阈值规则。

### 主要发现

在时钟成本足够低时，最优策略会在关键区域设置时钟关闭窗口，形成两级发布阶梯。杠杆作用创造了不可逆性楔子，最优与杠杆化剩余之间的差距由最低可逆阶梯的收购转换成本限制。补丁对债务不敏感，转向可能会扭曲但有限制。预测了企业披露中的特定遥测特征，如发布前强度下降和月内离散性，以及两个发布后平台期。

### 结论

AI产品发布节奏可视为战略实物期权的最优行使，企业通过控制技术信息披露时机和方式来优化市场表现。杠杆效应主要影响高成本转向决策，而对廉价补丁的影响有限。这些模式反映了企业自身对技术信号的限制，而非市场对事件风险的定价。

### 翻译

我们将AI产品发布的节奏（即静默期、可逆补丁和较少见的转向）建模为在声誉学习下战略实物期权的最优行使。一个私人观察到的技术状态遵循扩散过程。企业控制两种具有不对称成本和可逆性的升级选项（廉价补丁和高成本转向）以及一个发布频率时钟（Cox过程，其强度决定了何时披露嘈杂的公共性能和安全信号）。在时钟成本足够低的情况下，最优策略会在关键区域周围设置可观察的时钟关闭窗口。这些窗口关闭了公共信念的鞅部分，消除了关键区域的混合，并将行为简化为一个具有内生触发点、跳跃目标和无内部混合的两级发布阶梯。


### 论文摘要

We model the cadence of AI product releases, i.e. quiet spells, reversible patches, and rarer pivots, as optimal exercise of strategic real options under reputational learning. A privately observed technical state follows a diffusion. The firm controls two upgrade options with asymmetric costs and reversibility (a cheap patch and a costly pivot) and a publication-frequency clock, a Cox process whose intensity governs when noisy public performance and safety signals are disclosed. For sufficiently low clock costs the optimal policy posts observable clock-off windows around knife-edge regions. These windows shut down the martingale part of public beliefs, eliminate knife-edge mixing, and collapse behavior to a two-rung release ladder with endogenous triggers, jump targets, and no interior mixing. Within stationary Markov strategies we show that this ladder is uniquely characterized by a boundary-value system with value matching and smooth pasting at triggers and target optimality at jump targets. We endogenize market or platform adoption as a threshold rule in public beliefs and show that leverage creates an irreversibility wedge: the gap between first-best and levered surplus is bounded by the takeover switching cost of the least reversible rung. Patches are debt-insensitive; pivots can be distorted, but only up to that bound. The framework predicts telemetry signatures in firm-authored disclosures: a pre-release cadence dip in publication intensity and intra-month dispersion as the clock is shut off before a major reset; two post-release plateaus in disclosed performance, consistent with patch versus pivot jump targets; and debt-insensitive patch timing in high-reversibility regimes, with leverage effects concentrated in pivots. Unlike option-implied volatility spikes, these patterns reflect the firm's own throttling of technical signals rather than market pricing of event risk.

---

## 138. Effects of Distance Metrics and Scaling on the Perturbation Discrimination Score

**论文链接:** [http://arxiv.org/abs/2511.16954v1](http://arxiv.org/abs/2511.16954v1)

**作者:** Qiyuan Liu, Qirui Zhang, Jinhong Du, Siming Zhao, Jingshu Wang

**发布时间:** 2025-11-21

### GPT解析

### 总结

该论文研究了扰动判别分数（PDS）在高维基因表达环境下的行为，发现PDS对相似性/距离度量和效应规模的选择高度敏感，不同类型的PDS度量表现出显著差异，作者提供了几何见解并讨论了对未来评估度量的影响。

### 背景

PDS（扰动判别分数）被越来越多地用于评估预测的扰动效应是否仍然可区分，包括在Systema和Virtual Cell Challenge中使用。然而，在高维基因表达设置中，PDS的行为尚未得到详细研究。

### 目的

研究PDS在高维基因表达环境下的行为特征，特别是其对不同相似性或距离度量以及预测效应规模的敏感性。

### 方法

作者分析了观察到的扰动响应，比较了基于L1和L2的PDS与基于余弦的度量之间的差异，即使在范数匹配后。

### 主要发现

PDS对相似性或距离度量的选择以及预测效应的规模高度敏感。基于L1和L2的PDS与基于余弦的度量表现出显著不同的行为，即使在范数匹配后也是如此。

### 结论

作者提供了几何见解，并讨论了未来基于判别的评估度量的影响。

### 翻译

扰动判别分数（PDS）越来越多地用于评估预测的扰动效应是否仍然可区分，包括在Systema和Virtual Cell Challenge中使用。然而，在高维基因表达设置中，其行为尚未得到详细研究。我们表明，PDS对相似性或距离度量的选择以及预测效应的规模高度敏感。对观察到的扰动响应的分析显示，基于L1和L2的PDS与基于余弦的度量表现非常不同，即使在范数匹配后也是如此。我们提供了几何见解并讨论了对未来基于判别的评估度量的影响。


### 论文摘要

The Perturbation Discrimination Score (PDS) is increasingly used to evaluate whether predicted perturbation effects remain distinguishable, including in Systema and the Virtual Cell Challenge. However, its behavior in high-dimensional gene-expression settings has not been examined in detail. We show that PDS is highly sensitive to the choice of similarity or distance measure and to the scale of predicted effects. Analysis of observed perturbation responses reveals that $\ell_1$ and $\ell_2$-based PDS behave very differently from cosine-based measures, even after norm matching. We provide geometric insight and discuss implications for future discrimination-based evaluation metrics.

---

## 139. Warm Diffusion: Recipe for Blur-Noise Mixture Diffusion Models

**论文链接:** [http://arxiv.org/abs/2511.16904v1](http://arxiv.org/abs/2511.16904v1)

**作者:** Hao-Chien Hsueh, Chi-En Yen, Wen-Hsiao Peng, Ching-Chun Huang

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文提出了一种名为Warm Diffusion的统一模糊-噪声混合扩散模型，整合了热扩散和冷扩散的优势，通过控制模糊和噪声共同作用，提高了图像生成质量。

### 背景

扩散概率模型在各种生成任务中取得了显著成功，但存在两种主要范式：热扩散（完全依赖噪声）和冷扩散（仅使用模糊）。热扩散无法利用高频图像细节和低频结构间的强相关性，冷扩散则忽视了噪声在塑造数据流形中的作用。

### 目的

整合热扩散和冷扩散的优势，提出一种统一的扩散模型，解决现有方法的局限性，提高图像生成性能。

### 方法

提出Warm Diffusion（BNMD模型），采用分治策略利用图像频谱依赖性，通过解耦去噪和去模糊过程简化分数模型估计，并使用Blur-to-Noise Ratio分析模型学习动态与数据流形变化间的权衡。

### 主要发现

热扩散在生成早期步骤呈现随机行为，冷扩散存在流形外问题，两者结合可优势互补。通过频谱分析揭示了模糊和噪声在数据生成过程中的不同作用。

### 结论

大量跨基准实验验证了Warm Diffusion方法在图像生成任务中的有效性和优越性。

### 翻译

扩散概率模型在各种数据类型的生成任务中取得了显著成功。虽然最近的研究探索了高斯噪声之外的替代退化过程，但本文连接了两个关键的扩散范式：热扩散（完全依赖噪声）和冷扩散（仅使用模糊而不使用噪声）。我们认为热扩散无法利用高频图像细节和低频结构之间的强相关性，导致生成过程中早期步骤的随机行为。相反，虽然冷扩散利用图像相关性进行预测，但它忽视了噪声（随机性）在塑造数据流形中的作用，导致流形外问题并部分解释了其性能下降。为了整合两者的优势，我们提出了Warm Diffusion，一个统一的模糊-噪声混合扩散模型（BNMD），共同控制模糊和噪声。我们的分治策略利用了图像中的频谱依赖性，通过解耦去噪和去模糊过程简化了分数模型估计。我们进一步使用频谱分析分析了模糊-噪声比（BNR），以研究模型学习动态和数据流形变化之间的权衡。跨基准的大量实验验证了我们方法在图像生成中的有效性。


### 论文摘要

Diffusion probabilistic models have achieved remarkable success in generative tasks across diverse data types. While recent studies have explored alternative degradation processes beyond Gaussian noise, this paper bridges two key diffusion paradigms: hot diffusion, which relies entirely on noise, and cold diffusion, which uses only blurring without noise. We argue that hot diffusion fails to exploit the strong correlation between high-frequency image detail and low-frequency structures, leading to random behaviors in the early steps of generation. Conversely, while cold diffusion leverages image correlations for prediction, it neglects the role of noise (randomness) in shaping the data manifold, resulting in out-of-manifold issues and partially explaining its performance drop. To integrate both strengths, we propose Warm Diffusion, a unified Blur-Noise Mixture Diffusion Model (BNMD), to control blurring and noise jointly. Our divide-and-conquer strategy exploits the spectral dependency in images, simplifying score model estimation by disentangling the denoising and deblurring processes. We further analyze the Blur-to-Noise Ratio (BNR) using spectral analysis to investigate the trade-off between model learning dynamics and changes in the data manifold. Extensive experiments across benchmarks validate the effectiveness of our approach for image generation.

---

## 140. From Representation to Enactment: The ABC Framework of the Translating Mind

**论文链接:** [http://arxiv.org/abs/2511.16811v1](http://arxiv.org/abs/2511.16811v1)

**作者:** Michael Carl, Takanori Mizowaki, Aishvarya Raj, Masaru Yamada, Devi Sri Bandaru, Yuxiang Wei, Xinyue Ren

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文基于扩展心智理论和激进具身认知理论，提出了一种替代基于表征的心智模型的翻译心智新框架

### 背景

扩展心智理论和激进具身认知理论，反对基于表征的心智模型

### 目的

提出一种翻译心智的新框架，将翻译重新概念化为具身活动而非静态双语对应关系的操作

### 方法

提出ABC框架（情感、行为、认知过程的动态整合），借鉴预测处理和（激活）推理理论

### 主要发现

翻译者的心智通过脑-体-环境互动的循环而产生，而非简单的外部延伸；翻译是一种非表征性的活动

### 结论

翻译是对社会文化实践的熟练参与，意义是通过与文本、工具和环境的具身互动实时共同创造的

### 翻译

本文将翻译定义为动态整合情感、行为和认知过程的具身活动，而非静态双语对应关系的操作


### 论文摘要

Building on the Extended Mind (EM) theory and radical enactivism, this article suggests an alternative to representation-based models of the mind. We lay out a novel ABC framework of the translating mind, in which translation is not the manipulation of static interlingual correspondences but an enacted activity, dynamically integrating affective, behavioral, and cognitive (ABC) processes. Drawing on Predictive Processing and (En)Active Inference, we argue that the translator's mind emerges, rather than being merely extended, through loops of brain-body-environment interactions. This non-representational account reframes translation as skillful participation in sociocultural practice, where meaning is co-created in real time through embodied interaction with texts, tools, and contexts.

---

## 141. Membership Inference Attacks Beyond Overfitting

**论文链接:** [http://arxiv.org/abs/2511.16792v1](http://arxiv.org/abs/2511.16792v1)

**作者:** Mona Khalil, Alberto Blanco-Justicia, Najeeb Jebreel, Josep Domingo-Ferrer

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文研究了机器学习模型中的成员推理攻击(MIAs)及其防御策略。研究超越传统过拟合解释，探索了MIAs漏洞的根本原因，并针对易受攻击的样本提出了防御方法。

### 背景

成员推理攻击(MIAs)旨在确定特定数据点是否属于模型训练数据，对使用敏感数据进行训练的个体构成隐私风险。虽然过拟合是MIAs成功的主要因素，但研究表明即使非过拟合的ML模型也可能泄露部分训练数据信息。

### 目的

调查传统过拟合关注之外的成员推理漏洞的根本原因，并提出有针对性的防御策略。

### 方法

经验分析非过拟合模型(能够泛化的模型)中易受MIAs影响的训练数据样本的特征。

### 主要发现

易受MIAs影响的样本通常是它们类别中的异常值，例如有噪声或难以分类的样本。

### 结论

提出保护这些易受攻击样本的潜在防御策略，以增强ML模型的隐私保护能力。相关代码可在https://github.com/najeebjebreel/mia_analysis获取。

### 翻译

成员推理攻击(MIAs)针对机器学习(ML)模型，旨在确定特定数据点是否属于模型训练数据。这些攻击可能对使用敏感数据进行训练的个体构成重大隐私风险，促使人们使用差分隐私等防御措施，但这通常以高精度损失为代价。MIAs利用模型在预测训练中已见过的样本(成员)与未见过的样本(非成员)时的行为差异。多项研究表明，模型过拟合是导致这些行为差异的主要因素，也是MIAs成功的原因。然而，文献也表明，即使非过拟合的ML模型也可能泄露其训练数据的一小部分信息。在本文中，我们调查了传统过拟合关注之外的成员推理漏洞的根本原因，并提出有针对性的防御策略。我们经验分析了非过拟合模型(因此能够泛化)中易受MIAs影响的训练数据样本的特征。我们的发现揭示，这些样本通常是它们类别中的异常值(例如，有噪声或难以分类的样本)。我们随后提出了保护这些易受攻击样本的潜在防御策略，以增强ML模型的隐私保护能力。我们的代码可在https://github.com/najeebjebreel/mia_analysis获取。


### 论文摘要

Membership inference attacks (MIAs) against machine learning (ML) models aim to determine whether a given data point was part of the model training data. These attacks may pose significant privacy risks to individuals whose sensitive data were used for training, which motivates the use of defenses such as differential privacy, often at the cost of high accuracy losses. MIAs exploit the differences in the behavior of a model when making predictions on samples it has seen during training (members) versus those it has not seen (non-members). Several studies have pointed out that model overfitting is the major factor contributing to these differences in behavior and, consequently, to the success of MIAs. However, the literature also shows that even non-overfitted ML models can leak information about a small subset of their training data. In this paper, we investigate the root causes of membership inference vulnerabilities beyond traditional overfitting concerns and suggest targeted defenses. We empirically analyze the characteristics of the training data samples vulnerable to MIAs in models that are not overfitted (and hence able to generalize). Our findings reveal that these samples are often outliers within their classes (e.g., noisy or hard to classify). We then propose potential defensive strategies to protect these vulnerable samples and enhance the privacy-preserving capabilities of ML models. Our code is available at https://github.com/najeebjebreel/mia_analysis.

---

## 142. Landau-Lifshitz-Bloch simulations of the magnetocaloric effect in continuous ferromagnetic-paramagnetic transitions

**论文链接:** [http://arxiv.org/abs/2511.16756v1](http://arxiv.org/abs/2511.16756v1)

**作者:** Luis M. Moreno-Ramírez, Luis Sánchez-Tejerina, Óscar Alejos, Victorino Franco, Víctor Raposo

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出使用基于Landau-Lifshitz-Bloch方程的微磁模拟方法研究磁热材料在居里转变附近的磁热效应，该方法能够可靠预测单晶和多晶配置的等温熵变曲线，验证了微磁模拟是分析复杂微观结构磁热材料的有效工具。

### 背景

磁热材料建模不仅有助于理解其行为，还能预测新材料，对优化其性能至关重要。然而，与其他磁性材料研究相比，磁热材料的微磁模拟很少，主要原因是难以在材料转变附近进行建模。

### 目的

解决磁热材料在转变附近难以建模的限制，提出使用基于Landau-Lifshitz-Bloch方程的微磁模拟来研究铁磁材料在其居里转变附近的磁热效应。

### 方法

采用基于Landau-Lifshitz-Bloch方程的微磁模拟方法，研究铁磁材料在居里转变附近的磁热效应，考虑不同各向异性贡献，获取单晶和多晶配置的等温熵变曲线。

### 主要发现

所提出的方法能够获得可靠的单晶和多晶配置的等温熵变曲线，考虑了不同的各向异性贡献，结果与之前的实验和理论观察一致，验证了方法的稳健性。

### 结论

微磁模拟是分析具有复杂微观结构的磁热材料的强大工具，为理解和优化磁热材料性能提供了新的研究途径。

### 翻译

磁热材料建模的实用性从理解其行为扩展到预测新材料，在优化其性能中起着基本作用。与其他磁性材料研究领域相比，由于在转变附近建模材料的困难，磁热材料的微磁模拟很少。为了解决这一限制，我们提出使用基于Landau-Lifshitz-Bloch方程的微磁模拟来研究铁磁材料在其居里转变附近的磁热效应。按照我们提出的方法，我们获得了单晶和多晶配置的可靠等温熵变曲线，其中考虑了不同的各向异性贡献。评估了方法的稳健性，产生的结果与之前的实验和理论观察一致。我们的研究表明，微磁模拟是分析具有复杂微观结构的磁热材料的强大工具。


### 论文摘要

The usefulness of modeling magnetocaloric materials expands from the understanding of their behavior to the prediction of new materials, playing a fundamental role in the optimization of their performance. In contrast with other areas of magnetic materials research, micromagnetic simulations of magnetocaloric materials are scarce due to the difficulty of modeling the material in the vicinity of the transition. To solve this limitation, we propose the use of micromagnetic simulations based on the Landau-Lifshitz-Bloch equation to study the magnetocaloric effect of a ferromagnetic material around its Curie transition. Following our proposed methodology, we obtain reliable isothermal entropy change curves for both monocrystalline and polycrystalline configurations, where we consider different anisotropic contributions. The robustness of the method was evaluated, yielding results that agreed with previous experimental and theoretical observations. Our study shows that micromagnetic simulations are a powerful tool for analyzing magnetocaloric materials with complex microstructures.

---

## 143. DDTime: Dataset Distillation with Spectral Alignment and Information Bottleneck for Time-Series Forecasting

**论文链接:** [http://arxiv.org/abs/2511.16715v1](http://arxiv.org/abs/2511.16715v1)

**作者:** Yuqi Li, Kuiye Ding, Chuanguang Yang, Hao Wang, Haoxuan Wang, Huiran Duan, Junming Liu, Yingli Tian

**发布时间:** 2025-11-20

**备注:** 36 pages

### GPT解析

### 总结

本文提出DDTime框架，解决时间序列预测中的数据蒸馏问题，通过时序统计和频域对齐机制减轻自相关偏差，并利用信息瓶颈原理增强样本多样性，在保持计算效率的同时显著提高预测准确性。

### 背景

时间序列预测在许多领域都很基础，但训练准确模型通常需要大规模数据集和大量计算资源。数据蒸馏通过合成紧凑数据集提供了一种有前景的替代方案，但将其扩展到时间序列预测面临两个基本挑战：强自相关导致的时序偏差和合成样本多样性不足。

### 目的

解决时间序列预测中的数据蒸馏问题，克服时序偏差和样本多样性不足的挑战，提高预测准确性同时降低计算资源需求。

### 方法

提出DDTime框架，基于一阶凝聚分解构建轻量级即插即用蒸馏系统。通过时序统计重新审视值项对齐，引入频域对齐机制减轻自相关引起的偏差；设计受信息瓶颈原理启发的样本间正则化增强样本多样性。该框架理论上与多种凝聚范式兼容并支持稳定的一阶优化。

### 主要发现

在20个基准数据集和多样化预测架构上的实验表明，DDTime始终优于现有蒸馏方法，实现约30%的相对准确率提升，同时仅引入约2.49%的计算开销。

### 结论

DDTime有效解决了时间序列预测中的数据蒸馏问题，在保持计算效率的同时显著提高了预测准确性，为时间序列预测提供了一种高效的数据处理方法。

### 翻译

时间序列预测在许多领域都很基础，但训练准确模型通常需要大规模数据集和大量计算资源。数据蒸馏通过合成保留完整数据学习行为的紧凑数据集，提供了一种有前景的替代方案。然而，将数据蒸馏扩展到时间序列预测并不容易，面临两个基本挑战：1)强自相关导致的时序偏差，引起教师模型和学生模型之间值项对齐失真；2)合成样本多样性不足，由于缺乏明确的类别先验来规范轨迹变化。在本工作中，我们提出了DDTime，一种基于一阶凝聚分解构建的轻量级即插即用蒸馏框架。为解决挑战1，它通过时序统计重新审视值项对齐，并引入频域对齐机制来减轻自相关引起的偏差，确保频谱一致性和时间保真度。为解决挑战2，我们进一步设计了受信息瓶颈原理启发的样本间正则化，增强多样性并最大化合成轨迹的信息密度。组合目标理论上与多种凝聚范式兼容，并支持稳定的一阶优化。在20个基准数据集和多样化预测架构上的大量实验表明，DDTime始终优于现有蒸馏方法，实现约30%的相对准确率提升，同时仅引入约2.49%的计算开销。所有代码和蒸馏数据集将发布。


### 论文摘要

Time-series forecasting is fundamental across many domains, yet training accurate models often requires large-scale datasets and substantial computational resources. Dataset distillation offers a promising alternative by synthesizing compact datasets that preserve the learning behavior of full data. However, extending dataset distillation to time-series forecasting is non-trivial due to two fundamental challenges: 1.temporal bias from strong autocorrelation, which leads to distorted value-term alignment between teacher and student models; and 2.insufficient diversity among synthetic samples, arising from the absence of explicit categorical priors to regularize trajectory variety.   In this work, we propose DDTime, a lightweight and plug-in distillation framework built upon first-order condensation decomposition. To tackle Challenge 1, it revisits value-term alignment through temporal statistics and introduces a frequency-domain alignment mechanism to mitigate autocorrelation-induced bias, ensuring spectral consistency and temporal fidelity. To address Challenge 2, we further design an inter-sample regularization inspired by the information bottleneck principle, which enhances diversity and maximizes information density across synthetic trajectories. The combined objective is theoretically compatible with a wide range of condensation paradigms and supports stable first-order optimization. Extensive experiments on 20 benchmark datasets and diverse forecasting architectures demonstrate that DDTime consistently outperforms existing distillation methods, achieving about 30% relative accuracy gains while introducing about 2.49% computational overhead. All code and distilled datasets will be released.

---

## 144. Incorporating Bayesian Transfer Learning into Particle Filter for Dual-Tracking System with Asymmetric Noise Intensities

**论文链接:** [http://arxiv.org/abs/2511.17440v1](http://arxiv.org/abs/2511.17440v1)

**作者:** Omar A. Alotaibi, Brian L. Mark, Mohammad Reza Fasihi

**发布时间:** 2025-11-21

**备注:** 20 pages, 8 figures

### GPT解析

### 总结

该研究提出了一种基于贝叶斯迁移学习的粒子滤波方法，用于双跟踪系统中非线性动态模型的跟踪，其中两个传感器的测量噪声强度不对称。

### 背景

研究针对双跟踪系统中两个传感器测量噪声强度不对称的情况，主传感器比源传感器经历更高的噪声强度。

### 目的

开发一种粒子滤波方法，通过贝叶斯迁移学习提高主传感器在双跟踪系统中的跟踪性能。

### 方法

使用贝叶斯迁移学习，通过加权和粒子的总和来近似贝叶斯迁移学习的密度，从而改进粒子滤波方法。

### 主要发现

1. 与独立粒子滤波以及其他卡尔曼滤波变体的迁移学习相比，所提出的方法更有效；2. 增加粒子数量可提高迁移学习性能，提高率高于独立粒子滤波，但会增加计算时间；3. 性能增益与传感器噪声强度差值大致成线性比例。

### 结论

贝叶斯迁移学习方法在双跟踪系统中有效，特别是在主传感器噪声强度较高的情况下，为非线性动态模型跟踪提供了有效解决方案。

### 翻译

我们使用贝叶斯迁移学习，开发了一种粒子滤波方法，用于跟踪双跟踪系统中非线性动态模型，其中两个传感器的测量噪声强度不对称。使用加权和粒子的总和来近似贝叶斯迁移学习的密度，以提高比源传感器经历更高噪声强度的主传感器的跟踪性能。我们通过仿真结果验证了与独立粒子滤波以及应用于无迹卡尔曼滤波和立方卡尔曼滤波的迁移学习相比，所提出方法的有效性。此外，增加粒子数量可以提高应用于粒子滤波的迁移学习性能，提高率高于独立粒子滤波。然而，增加粒子数量会增加每一步的计算时间。而且，整合贝叶斯迁移学习的性能增益与双跟踪系统中传感器噪声强度之间的绝对差值大致成线性比例。


### 论文摘要

Using Bayesian transfer learning, we develop a particle filter approach for tracking a nonlinear dynamical model in a dual-tracking system where intensities of measurement noise for both sensors are asymmetric. The densities for Bayesian transfer learning are approximated with the sum of weighted particles to improve the tracking performance of the primary sensor, which experiences a higher noise intensity compared to the source sensor. We present simulation results that validate the effectiveness of the proposed approach compared to an isolated particle filter and transfer learning applied to the unscented Kalman filter and the cubature Kalman filter. Furthermore, increasing the number of particles shows an improvement in the performance of transfer learning applied to the particle filter with a higher rate compared to the isolated particle filter. However, increasing the number of particles raises computational time per step. Moreover, the performance gain from incorporating Bayesian transfer learning is approximately linearly proportional to the absolute difference value between the noise intensities of the sensors in the dual-tracking system.

---

## 145. Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal

**论文链接:** [http://arxiv.org/abs/2511.17353v1](http://arxiv.org/abs/2511.17353v1)

**作者:** Xiaolong Qian, Qi Jiang, Lei Sun, Zongxi Yu, Kailun Yang, Peixuan Wu, Jiacheng Zhou, Yao Gao, Yaoguang Ma, Ming-Hsuan Yang, Kaiwei Wang

**发布时间:** 2025-11-21

**备注:** All code and datasets will be publicly released at https://github.com/XiaolongQian/DeVeiler

### GPT解析

### 总结

该研究提出了一种处理紧凑型光学系统中眩光遮蔽问题的新方法，包括VeilGen生成模型和DeVeiler恢复网络，能够有效模拟和去除由杂散光散射导致的眩光，提高成像质量。

### 背景

紧凑型光学系统（如单透镜和金属透镜）的成像性能不仅受常见光学像差影响，还常被非理想光学表面和涂层的杂散光散射导致的眩光遮蔽所 degrade，这种复合退化削弱了传统镜头像差校正效果，但研究不足。

### 目的

开发一种方法来模拟和去除紧凑型光学系统中的眩光遮蔽问题，解决传统散射模型无法适应眩光的空间变化和深度独立特性导致的成对高质量数据难以准备的问题。

### 方法

提出VeilGen生成模型，通过无监督方式从目标图像估计光学传输和眩光图，并由Stable Diffusion先验正则化；引入DeVeiler恢复网络，利用预测的潜在图指导散射模型的逆过程，并应用可逆性约束进行训练。

### 主要发现

在具有挑战性的紧凑型光学系统上的实验表明，该方法相比现有方法提供了更好的恢复质量和物理保真度，VeilGen能可靠合成真实眩光，其学习的潜在图能有效指导恢复过程。

### 结论

VeilGen和DeVeiler的组合能有效处理紧凑型光学系统中的眩光遮蔽问题，通过生成模型模拟眩光和恢复网络去除眩光，提高了成像质量和物理保真度。

### 翻译

除了公认的光学像差外，紧凑型光学系统（包括单透镜和金属透镜设计）的成像性能通常还因非理想光学表面和涂层的杂散光散射导致的眩光遮蔽而进一步降质，特别是在复杂的真实环境中。这种复合削弱了传统镜头像差校正的效果，但研究不足。主要挑战是传统散射模型（如去雾模型）无法适应眩光的空间变化和深度独立特性。因此，难以通过仿真准备成对的高质量数据，阻碍了数据驱动的眩光去除模型的应用。为此，我们提出了VeilGen，一种生成模型，它通过无监督方式从目标图像中学习模拟眩光，估计其基础光学传输和眩光图，并由基于Stable Diffusion的先验进行正则化。VeilGen能够生成具有真实复合退化（光学像差和眩光）的成对数据集，同时提供估计的潜在光学传输和眩光图来指导眩光去除过程。我们还引入了DeVeiler，一个带有可逆性约束的恢复网络，它利用预测的潜在图来指导学习到的散射模型的逆过程。在具有挑战性的紧凑型光学系统上的广泛实验表明，与现有方法相比，我们的方法提供了更好的恢复质量和物理保真度。这表明VeilGen能够可靠地合成真实的眩光，其学习的潜在图有效地指导了DeVeiler中的恢复过程。所有代码和数据集将在https://github.com/XiaolongQian/DeVeiler上公开发布。


### 论文摘要

Beyond the commonly recognized optical aberrations, the imaging performance of compact optical systems-including single-lens and metalens designs-is often further degraded by veiling glare caused by stray-light scattering from non-ideal optical surfaces and coatings, particularly in complex real-world environments. This compound degradation undermines traditional lens aberration correction yet remains underexplored. A major challenge is that conventional scattering models (e.g., for dehazing) fail to fit veiling glare due to its spatial-varying and depth-independent nature. Consequently, paired high-quality data are difficult to prepare via simulation, hindering application of data-driven veiling glare removal models. To this end, we propose VeilGen, a generative model that learns to simulate veiling glare by estimating its underlying optical transmission and glare maps in an unsupervised manner from target images, regularized by Stable Diffusion (SD)-based priors. VeilGen enables paired dataset generation with realistic compound degradation of optical aberrations and veiling glare, while also providing the estimated latent optical transmission and glare maps to guide the veiling glare removal process. We further introduce DeVeiler, a restoration network trained with a reversibility constraint, which utilizes the predicted latent maps to guide an inverse process of the learned scattering model. Extensive experiments on challenging compact optical systems demonstrate that our approach delivers superior restoration quality and physical fidelity compared with existing methods. These suggest that VeilGen reliably synthesizes realistic veiling glare, and its learned latent maps effectively guide the restoration process in DeVeiler. All code and datasets will be publicly released at https://github.com/XiaolongQian/DeVeiler.

---

## 146. MedImageInsight for Thoracic Cavity Health Classification from Chest X-rays

**论文链接:** [http://arxiv.org/abs/2511.17043v1](http://arxiv.org/abs/2511.17043v1)

**作者:** Rama Krishna Boya, Mohan Kireeti Magalanadu, Azaruddin Palavalli, Rupa Ganesh Tekuri, Amrit Pattanayak, Prasanthi Enuga, Vignesh Esakki Muthu, Vivek Aditya Boya

**发布时间:** 2025-11-21

**备注:** 9 pages, 5 figures and 3 tables

### GPT解析

### 总结

该研究评估了使用MedImageInsight医学基础模型对胸部X光片进行自动化二元分类（正常和异常）的两种方法：微调模型和作为特征提取器的迁移学习。微调的分类器表现最佳，ROC-AUC为0.888，性能与CheXNet相当。该系统可集成到临床工作流程中以支持分诊并减轻放射科医生负担。

### 背景

胸部X光摄影是最广泛使用的胸部诊断成像方式之一，但不断增加的成像量和放射科医生的工作量继续对及时解释带来挑战。

### 目的

研究使用MedImageInsight医学成像基础模型，对胸部X光片进行自动化二元分类（正常和异常）。

### 方法

评估了两种方法：(1)微调MedImageInsight进行端到端分类，(2)将模型作为特征提取器，使用传统机器学习分类器进行迁移学习。实验使用了ChestX-ray14数据集和来自合作医院的真实临床数据的组合。

### 主要发现

微调的分类器取得了最佳性能，ROC-AUC为0.888，并且在校准方面优于迁移学习模型，其性能与CheXNet等成熟架构相当。

### 结论

基础医学成像模型在减少特定任务培训要求的同时保持诊断可靠性方面是有效的。该系统设计用于集成到基于Web和医院PACS工作流程中，以支持分诊并减少放射科医生负担。未来的工作将扩展模型进行多标签病理分类，以在临床环境中提供初步诊断解释。

### 翻译

胸部X光摄影仍然是胸部诊断中最广泛使用的成像方式之一，然而不断增加的成像量和放射科医生的工作量继续对及时解释构成挑战。在这项工作中，我们研究了使用MedImageInsight（一种医学成像基础模型）对胸部X光片进行自动化二元分类（正常和异常类别）的应用。评估了两种方法：(1)微调MedImageInsight进行端到端分类，(2)将模型作为特征提取器，使用传统机器学习分类器进行迁移学习。实验使用了ChestX-ray14数据集和来自合作医院的真实临床数据的组合。微调分类器取得了最佳性能，ROC-AUC为0.888，并且在校准方面优于迁移学习模型，展示了与CheXNet等成熟架构相当的性能。这些结果突显了基础医学成像模型在减少特定任务培训要求的同时保持诊断可靠性方面的有效性。该系统设计用于集成到基于Web和医院PACS工作流程中，以支持分诊并减轻放射科医生负担。未来的工作将扩展模型进行多标签病理分类，以在临床环境中提供初步诊断解释。


### 论文摘要

Chest radiography remains one of the most widely used imaging modalities for thoracic diagnosis, yet increasing imaging volumes and radiologist workload continue to challenge timely interpretation. In this work, we investigate the use of MedImageInsight, a medical imaging foundational model, for automated binary classification of chest X-rays into Normal and Abnormal categories. Two approaches were evaluated: (1) fine-tuning MedImageInsight for end-to-end classification, and (2) employing the model as a feature extractor for a transfer learning pipeline using traditional machine learning classifiers. Experiments were conducted using a combination of the ChestX-ray14 dataset and real-world clinical data sourced from partner hospitals. The fine-tuned classifier achieved the highest performance, with an ROC-AUC of 0.888 and superior calibration compared to the transfer learning models, demonstrating performance comparable to established architectures such as CheXNet. These results highlight the effectiveness of foundational medical imaging models in reducing task-specific training requirements while maintaining diagnostic reliability. The system is designed for integration into web-based and hospital PACS workflows to support triage and reduce radiologist burden. Future work will extend the model to multi-label pathology classification to provide preliminary diagnostic interpretation in clinical environments.

---

## 147. On a synergistic learning phenomenon in nonparametric domain adaptation

**论文链接:** [http://arxiv.org/abs/2511.17009v1](http://arxiv.org/abs/2511.17009v1)

**作者:** Ling Zhou, Yuhong Yang

**发布时间:** 2025-11-21

### GPT解析

### 总结

本文研究了非参数领域适应回归问题，提出了一种新的协同学习现象(SLP)，表明当目标样本量小于但不是远小于源样本量时，结合两种数据的最小极大收敛速率可能比仅使用任一数据时的更好速率更快。

### 背景

研究假设给定协变量的响应条件分布相同，但协变量的边缘分布不同，特别关注目标数据和源数据的协变量边缘分布的似然比无界的情况。

### 目的

理解源数据如何提高学习回归函数的最小极大收敛速率，特别是在似然比无界的情况下。

### 方法

通过理论分析和最小极大收敛速率研究，探索源数据和目标数据的协同效应。

### 主要发现

发现协同学习现象(SLP)发生在当且仅当目标样本量小于但不是远小于源样本量时，与回归函数的光滑度以及源分布和目标分布的协变量密度的性质有关。SLP以两种方式发生：目标数据帮助估计源数据密度接近零的点处的回归函数，源数据帮助估计源数据密度不小的点处的回归函数。

### 结论

结合源数据和目标数据可以比单独使用任一数据更有效地估计回归函数，特别是在目标样本量适当小于源样本量的情况下。

### 翻译

考虑用于回归的非参数领域适应，它假设给定协变量的响应条件分布相同，但协变量的边缘分布不同。一个重要的目标是理解当目标数据和源数据的协变量边缘分布的似然比无界时，源数据如何提高学习回归函数的最小极大收敛速率。Pathak等人(2022)的先前研究表明，最小极大迁移学习速率仅由单独使用源数据或目标数据时的较快速率决定。在本文中，我们提出了一种新的协同学习现象(SLP)，即基于两种数据的最小极大收敛速率有时可能比仅基于源数据或目标数据时的更好速率更快(甚至快得多)。SLP当且仅当目标样本量小于(数量级上)但不是远小于源样本量时发生，与回归函数的光滑度以及源分布和目标分布的协变量密度的性质有关。有趣的是，根据两个样本量之间的关系，SLP以两种不同的方式发生。一种是目标数据有助于缓解在源数据密度接近零的点处估计回归函数的困难，另一种是源数据(具有比目标数据更大的样本量)有助于在源数据密度不小的点处的估计。还获得了处理未知源和目标参数以及回归函数的光滑度的扩展。


### 论文摘要

Consider nonparametric domain adaptation for regression, which assumes the same conditional distribution of the response given the covariates but different marginal distributions of the covariates. An important goal is to understand how the source data may improve the minimax convergence rate of learning the regression function when the likelihood ratio of the covariate marginal distributions of the target data and the source data are unbounded. A previous work of Pathak et al. (2022) show that the minimax transfer learning rate is simply determined by the faster rate of using either the source or the target data alone. In this paper, we present a new synergistic learning phenomenon (SLP) that the minimax convergence rate based on both data may sometimes be faster (even much faster) than the better rate of convergence based on the source or target data only. The SLP occurs when and only when the target sample size is smaller (in order) than but not too much smaller than the source sample size in relation to the smoothness of the regression function and the nature of the covariate densities of the source and target distributions. Interestingly, the SLP happens in two different ways according to the relationship between the two sample sizes. One is that the target data help alleviate the difficulty in estimating the regression function at points where the density of the source data is close to zero and the other is that the source data (with its larger sample size than that of the target data) help the estimation at points where the density of the source data is not small. Extensions to handle unknown source and target parameters and smoothness of the regression function are also obtained.

---

## 148. Quantum Data Learning of Topological-to-Ferromagnetic Phase Transitions in the 2+1D Toric Code Loop Gas Model

**论文链接:** [http://arxiv.org/abs/2511.16851v1](http://arxiv.org/abs/2511.16851v1)

**作者:** Shamminuj Aktar, Rishabh Bhardwaj, Andreas Bärtschi, Tanmoy Bhattacharya, Stephan Eidenbenz

**发布时间:** 2025-11-20

### GPT解析

### 总结

量子数据学习(QDL)提供了一种直接从量子态中提取物理洞察的框架，本研究将其应用于2+1维环码环气模型在磁场中的相变研究，通过监督和无监督两种方法成功识别了量子相结构并定位了相变点，性能优于经典方法。

### 背景

多体物理的核心挑战在于量子相（尤其是具有拓扑序的相）的身份通常无法通过局部可观测量或简单的对称性破缺诊断来识别。传统方法难以直接从量子态中提取物理洞察。

### 目的

应用QDL技术研究2+1维环码环气模型在磁场中的行为，实现相分类并捕获整体相结构，评估QDL方法在表征拓扑量子物质、研究有限体积效应和探测高维系统相图方面的有效性。

### 方法

使用参数化环气电路(PLGC)和变分量子特征求解器(VQE)生成多个晶格尺寸下的基态；训练量子卷积神经网络(QCNN)进行相分类；采用物理感知的训练协议，排除相变点附近的临界区域用于测试；实现基于态重叠的无监督量子k均值方法进行相分类。

### 主要发现

监督式QDL方法恢复了相结构并准确定位了相变，与先前报道值高度一致；无监督式QDL方法恢复了相结构并定位了相变，在有限体积中存在小的偏移；两种QDL方法都优于经典替代方法。

### 结论

这些发现确立了QDL作为表征拓扑量子物质、研究有限体积效应和探测高维系统相图的有效框架。

### 翻译

量子数据学习(QDL)提供了一种直接从量子态中提取物理洞察的框架，绕过了对理论经典可观测量识别的需求。多体物理中的一个中心挑战是量子相的身份，特别是那些具有拓扑序的相，通常无法通过局部可观测量或简单的对称性破缺诊断来获得。在这里，我们将QDL技术应用于2+1维环码环气模型在磁场中的研究。使用参数化环气电路(PLGC)和变分量子特征求解器(VQE)方法生成多个晶格尺寸下的基态。然后我们在整个场参数范围内训练量子卷积神经网络(QCNN)进行相分类并捕获整体相结构。我们还采用了一种物理感知的训练协议，排除了量子蒙特卡洛估计的相变点(x_c = 0.25)附近的临界区域(0.2 <= x <= 0.4)，保留这个窗口用于测试，以评估模型学习相变的能力。同时，我们实现了一种基于态重叠的无监督量子k均值方法，无需预先标记即可将数据集分为两个相。我们的监督式QDL方法恢复了相结构并准确定位了相变，与先前报道的值高度一致；无监督式QDL方法恢复了相结构并定位了相变，在有限体积中存在小的偏移（符合预期）；两种QDL方法都优于经典替代方法。这些发现确立了QDL作为表征拓扑量子物质、研究有限体积效应和探测高维系统相图的有效框架。


### 论文摘要

Quantum data learning (QDL) provides a framework for extracting physical insights directly from quantum states, bypassing the need for any identification of the classical observable of the theory. A central challenge in many-body physics is that the identity of quantum phases, especially those with topological order, are often inaccessible through local observables or simple symmetry-breaking diagnostics. Here, we apply QDL techniques to the 2+1-dimensional toric-code loop-gas model in a magnetic field. Ground states are generated across multiple lattice sizes using a parametrized loop-gas circuit (PLGC) with a variational quantum-eigensolver (VQE) approach. We then train a quantum convolutional neural network (QCNN) across the full field-parameter range to perform phase classification and capture the overall phase structure. We also employ a physics-aware training protocol that excludes the near-critical region (0.2 <= x <= 0.4)) around (x_c = 0.25), the phase-transition point estimated by quantum Monte Carlo, reserving this window for testing to evaluate the ability of the model to learn the phase transition. In parallel, we implement an unsupervised quantum k-means method based on state overlaps, which partitions the dataset into two phases without prior labeling. Our supervised QDL approach recovers the phase structure and accurately locates the phase transition, in close agreement with previously reported values; the unsupervised QDL approach recovers the phase structure and locates the phase transition with a small offset as expected in finite volumes; both QDL methods outperform classical alternatives. These findings establish QDL as an effective framework for characterizing topological quantum matter, studying finite volume effects, and probing phase diagrams of higher-dimensional systems.

---

## 149. Trust in AI emerges from distrust in humans: A machine learning study on decision-making guidance

**论文链接:** [http://arxiv.org/abs/2511.16769v1](http://arxiv.org/abs/2511.16769v1)

**作者:** Johan Sebastián Galindez-Acosta, Juan José Giraldo-Huertas

**发布时间:** 2025-11-20

**备注:** 36 pages, 6 figures

### GPT解析

### 总结

本研究探讨了人工智能代理（特别是大型语言模型）中的信任动态，引入了'延迟信任'概念，即对人类代理的不信任会转向被认为更中立或更有能力的AI。

### 背景

现有社会心理学和技术接受模型框架中，关于用户因素影响AI信任的研究存在空白。

### 目的

研究用户在决策场景中选择不同类型向导（AI代理、语音助手、同龄人、成人或牧师）的信任因素，并验证延迟信任理论。

### 方法

55名本科生参与了30个决策场景（事实性、情感性、道德性）的实验；使用K-Modes和K-Means聚类分析模式；应用XGBoost模型和SHAP解释来预测基于社会人口统计和先前信任变量的AI选择。

### 主要发现

成人(35.05%)和AI(28.29%)是最常选择的代理；AI在事实场景中占主导，而人类在社会/道德场景中占主导；对人类代理的先前信任较低预测更高的AI选择，支持延迟信任作为补偿性转移；高AI信任特征是对人类不信任、技术使用较低和较高社会经济地位；模型表现一致（平均精度高达0.863）。

### 结论

研究结果挑战了传统的TAM/UTAUT模型，强调了AI信任的关系和认知维度；指出了流利效应导致的过度依赖风险；强调需要透明度来保持警惕性。

### 翻译

本研究通过引入'延迟信任'概念（一种认知机制，即对人类代理的不信任会转向被认为更中立或更有能力的AI），探讨了人工智能代理（特别是大型语言模型）的信任动态。基于社会心理学和技术接受模型框架，研究解决了影响AI信任的用户因素空白。55名本科生参与了涉及30个决策场景（事实性、情感性、道德性）的实验，参与者从AI代理（如ChatGPT）、语音助手、同龄人、成人或牧师中选择向导。数据使用K-Modes和K-Means聚类分析模式，并应用XGBoost模型和SHAP解释来预测基于社会人口统计和先前信任变量的AI选择。结果显示成人(35.05%)和AI(28.29%)是最常选择的代理。聚类分析揭示了情境特定偏好：AI在事实场景中占主导，而人类在社会/道德场景中占主导。对人类代理（牧师、同龄人、成人）的先前信任较低 consistently预测更高的AI选择，支持延迟信任作为补偿性转移。具有更高AI信任的参与者特征是对人类不信任、技术使用较低和较高社会经济地位。模型表现一致（例如，平均精度高达0.863）。研究结果挑战了传统的TAM/UTAUT模型，强调了AI信任的关系和认知维度。它们指出了流利效应导致的过度依赖风险，并强调了需要透明度来保持警惕性。局限性包括样本同质性和静态场景；未来工作应纳入多样化人群和多模态数据，以在不同情境中完善延迟信任理论。


### 论文摘要

This study explores the dynamics of trust in artificial intelligence (AI) agents, particularly large language models (LLMs), by introducing the concept of "deferred trust", a cognitive mechanism where distrust in human agents redirects reliance toward AI perceived as more neutral or competent. Drawing on frameworks from social psychology and technology acceptance models, the research addresses gaps in user-centric factors influencing AI trust. Fifty-five undergraduate students participated in an experiment involving 30 decision-making scenarios (factual, emotional, moral), selecting from AI agents (e.g., ChatGPT), voice assistants, peers, adults, or priests as guides. Data were analyzed using K-Modes and K-Means clustering for patterns, and XGBoost models with SHAP interpretations to predict AI selection based on sociodemographic and prior trust variables.   Results showed adults (35.05\%) and AI (28.29\%) as the most selected agents overall. Clustering revealed context-specific preferences: AI dominated factual scenarios, while humans prevailed in social/moral ones. Lower prior trust in human agents (priests, peers, adults) consistently predicted higher AI selection, supporting deferred trust as a compensatory transfer. Participant profiles with higher AI trust were distinguished by human distrust, lower technology use, and higher socioeconomic status. Models demonstrated consistent performance (e.g., average precision up to 0.863).   Findings challenge traditional models like TAM/UTAUT, emphasizing relational and epistemic dimensions in AI trust. They highlight risks of over-reliance due to fluency effects and underscore the need for transparency to calibrate vigilance. Limitations include sample homogeneity and static scenarios; future work should incorporate diverse populations and multimodal data to refine deferred trust across contexts.

---

## 150. EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards

**论文链接:** [http://arxiv.org/abs/2511.16672v2](http://arxiv.org/abs/2511.16672v2)

**作者:** Omkar Thawakar, Shravan Venkatraman, Ritesh Thawkar, Abdelrahman Shaker, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Fahad Khan

**发布时间:** 2025-11-20

**备注:** 9 Pages, 6 Figures, 4 Tables

### GPT解析

### 总结

本文提出了一个名为EvoLMM的自进化框架，通过两个合作代理(提议者和求解者)在完全无监督的方式下提升大型多模态模型的推理能力，无需任何标注数据或奖励蒸馏。该框架通过自我奖励过程促进信息丰富查询的生成和结构化推理的完善。实验表明，在多模态数学推理基准测试上取得了约3%的一致性提升。

### 背景

大型多模态模型(LMMs)最近取得了显著进展，具有强大的推理和感知能力，但大多数现有的训练流程仍然依赖于人工整理的数据或外部验证的奖励模型，这限制了它们的自主性和可扩展性。

### 目的

以完全无监督的方式(没有任何标注数据或奖励蒸馏)提高LMM的推理能力。

### 方法

提出一个名为EvoLMM的自进化框架，从单个骨干模型中实例化两个合作的代理：提议者(Proposer)生成多样化的、基于图像的问题；求解者(Solver)通过内部一致性解决这些问题。学习通过持续的自我奖励过程进行，这种动态反馈鼓励生成信息丰富的查询和完善结构化推理，而不依赖于真实答案或人类判断。

### 主要发现

当使用流行的Qwen2.5-VL作为基础模型时，EvoLMM仅使用原始训练图像，就在多模态数学推理基准测试(包括ChartQA、MathVista和MathVision)上取得了高达约3%的一致性提升。

### 结论

简单而有效的方法将成为一个坚实的基础，促进未来在完全无监督方式下自我改进的LMM研究。代码和模型已公开可用。

### 翻译

大型多模态模型(LMMs)的最新进展已经令人印象深刻地展示了推理和感知能力，然而大多数现有的训练流程仍然依赖于人工整理的数据或外部验证的奖励模型，限制了它们的自主性和可扩展性。在这项工作中，我们努力以完全无监督的方式(没有任何标注数据或奖励蒸馏)提高LMM的推理能力。为此，我们提出了一个名为EvoLMM的自进化框架，该框架从单个骨干模型中实例化两个合作的代理：一个提议者，它生成多样化的、基于图像的问题，以及一个求解者，它通过内部一致性解决这些问题，学习通过持续的自我奖励过程进行。这种动态反馈鼓励生成信息丰富的查询和完善结构化推理，而不依赖于真实答案或人类判断。当使用流行的Qwen2.5-VL作为基础模型时，我们的EvoLMM仅使用原始训练图像，就在多模态数学推理基准测试(包括ChartQA、MathVista和MathVision)上取得了高达约3%的一致性提升。我们希望这个简单而有效的方法将成为一个坚实的基础，促进未来在完全无监督方式下自我改进的LMM研究。我们的代码和模型可在https://github.com/mbzuai-oryx/EvoLMM获取。


### 论文摘要

Recent advances in large multimodal models (LMMs) have enabled impressive reasoning and perception abilities, yet most existing training pipelines still depend on human-curated data or externally verified reward models, limiting their autonomy and scalability. In this work, we strive to improve LMM reasoning capabilities in a purely unsupervised fashion (without any annotated data or reward distillation). To this end, we propose a self-evolving framework, named EvoLMM, that instantiates two cooperative agents from a single backbone model: a Proposer, which generates diverse, image-grounded questions, and a Solver, which solves them through internal consistency, where learning proceeds through a continuous self-rewarding process. This dynamic feedback encourages both the generation of informative queries and the refinement of structured reasoning without relying on ground-truth or human judgments. When using the popular Qwen2.5-VL as the base model, our EvoLMM yields consistent gains upto $\sim$3\% on multimodal math-reasoning benchmarks, including ChartQA, MathVista, and MathVision, using only raw training images. We hope our simple yet effective approach will serve as a solid baseline easing future research in self-improving LMMs in a fully-unsupervised fashion. Our code and models are available at https://github.com/mbzuai-oryx/EvoLMM.

---

## 151. A Machine Learning-Driven Solution for Denoising Inertial Confinement Fusion Images

**论文链接:** [http://arxiv.org/abs/2511.16717v1](http://arxiv.org/abs/2511.16717v1)

**作者:** Asya Y. Akkus, Bradley T. Wolfe, Pinghan Chu, Chengkun Huang, Chris S. Campbell, Mariana Alvarado Alvarez, Petr Volegov, David Fittinghoff, Robert Reinovsky, Zhehui Wang

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究实现了一种结合CDF 97小波变换的无监督自编码器，用于中子成像中混合高斯-泊松噪声的有效去除，表现出优于传统方法的性能。

### 背景

中子成像在分析惯性约束聚变(ICF)事件中至关重要，特别是在国家点火装置(NIF)等设施。然而，中子源图像常被高斯和泊松噪声污染，这些噪声掩盖细节并模糊边缘，传统滤波方法难以有效处理。过去由于缺乏真实训练数据，机器学习方法应用受限。

### 目的

开发一种能够有效去除混合高斯-泊松噪声同时保持图像保真度的去噪技术，提升中子源图像的分析和解释能力。

### 方法

在潜在空间中实现结合Cohen-Daubechies-Feauveau (CDF 97)小波变换的无监督自编码器，用于混合高斯-泊松噪声去除。

### 主要发现

该网络成功去噪了中子成像数据，与基于前向模型生成的数据相比，展示了比非机器学习方法如BM3D更低的重建误差和更好的边缘保持指标。

### 结论

该方法为中子图像降噪和ICF实验的三维重建分析提供了有前景的技术进步。

### 翻译

中子成像在优化分析惯性约束聚变(ICF)事件（如国家点火装置(NIF)的事件）以及改进当前和未来的ICF平台方面非常重要。然而，中子源的图像经常被各种类型的噪声退化。最常见的是，高斯噪声和泊松噪声经常在一个图像中共存，掩盖了精细细节并模糊了边缘。这些噪声类型经常重叠，使得使用传统滤波和阈值方法难以区分和去除。因此，保持图像保真度的噪声去除技术对于分析和解释中子源图像非常重要。当前解决方案包括滤波和阈值方法的组合。过去，由于缺乏ICF过程的中子成像真实数据，机器学习方法很少被实施。然而，合成数据生产的最新进展，特别是在融合成像领域，为使用监督和无监督机器学习方法研究新的去噪程序提供了机会。在这项研究中，我们在潜在空间中实现了一个结合Cohen-Daubechies-Feauveau (CDF 97)小波变换的无监督自编码器，用于混合高斯-泊松去噪。该网络成功去噪了中子成像数据。此外，与通过前向模型生成的数据相比，与基于非机器学习的滤波机制（如块匹配和3D滤波(BM3D)）相比，它展示了更低的重建误差和更好的边缘保持指标。这种方法在中子图像降噪和ICF实验的三维重建分析方面是一个有前景的进步。


### 论文摘要

Neutron imaging is important in optimizing analysis of inertial confinement fusion (ICF) events such as those at the National Ignition Facility (NIF) and improving current and future ICF platforms. However, images of neutron sources are often degraded by various types of noise. Most commonly, Gaussian and Poisson noise often coexist within one image, obscuring fine details and blurring edges. These noise types often overlap, making them difficult to distinguish and remove using conventional filtering and thresholding methods. As a result, noise removal techniques that preserve image fidelity are important for analyzing and interpreting images of a neutron source. Current solutions include a combination of filtering and thresholding methodologies. In the past, machine learning approaches were rarely implemented due to a lack of ground truth neutron imaging data for ICF processes. However, recent advances in synthetic data production, particularly in the fusion imaging field, have opened opportunities to investigate new denoising procedures using both supervised and unsupervised machine learning methods. In this study, we implement an unsupervised autoencoder with a Cohen-Daubechies- Feauveau (CDF 97) wavelet transform in the latent space for mixed Gaussian-Poisson denoising. The network successfully denoises neutron imaging data. Additionally, it demonstrates lower reconstruction error and superior edge preservation metrics when benchmarked with data generated by a forward model and compared to non-ML-based filtering mechanisms such as Block-matching and 3D filtering (BM3D). This approach presents a promising advancement in neutron image noise reduction and three-dimensional reconstruction analysis of ICF experiments.

---

