# 今日论文推荐 - 2025-10-14

共 172 篇论文

---

## 1. Pre-Training and Personalized Fine-Tuning via Over-the-Air Federated Meta-Learning: Convergence-Generalization Trade-Offs

**论文链接:** [http://arxiv.org/abs/2406.11569v5](http://arxiv.org/abs/2406.11569v5)

**作者:** Haifeng Wen, Hong Xing, Osvaldo Simeone

**发布时间:** 2024-06-17

**备注:** 40 pages, 10 figures, submitted for possible journal publication

### GPT解析

### 总结

该研究探讨了基于元学习的个性化联邦学习(meta-pFL)在无线环境中的泛化性能，分析了泛化到新代理和任务与收敛性之间的权衡关系。

### 背景

现代AI应用如大型语言模型的训练范式已转变为预训练后微调；由于开放数据减少和AI模型访问民主化，预训练正从集中式部署转向联邦学习实现。

### 目的

研究meta-pFL在无线环境中的泛化性能，探索对新代理和任务的泛化与收敛性之间的权衡。

### 方法

采用空中计算技术，研究通过共享无线信道连接到服务器的无线环境中参与预训练阶段的代理的情况。

### 主要发现

信道损伤可能会增强泛化能力同时降低收敛性，存在泛化与收敛性之间的权衡关系。

### 结论

大量的数值结果验证了所提出的理论。

### 翻译

对于现代人工智能应用（如大型语言模型），训练范式已转变为预训练后微调。此外，由于开放数据存储库减少以及AI模型访问民主化的努力，预训练预计将从当前集中式部署转向联邦学习实现。元学习提供了一个可以将预训练和微调形式化的通用框架。基于元学习的个性化联邦学习(meta-pFL)通过针对新代理和任务的泛化能力，超越了基本个性化。本文研究了meta-pFL在无线环境中的泛化性能，其中参与预训练阶段的代理通过共享无线信道连接到服务器。采用空中计算，我们研究了泛化到新代理和任务与收敛性之间的权衡。这种权衡源于信道损伤可能增强泛化同时降低收敛性的事实。大量的数值结果验证了该理论。


### 论文摘要

For modern artificial intelligence (AI) applications such as large language models (LLMs), the training paradigm has recently shifted to pre-training followed by fine-tuning. Furthermore, owing to dwindling open repositories of data and thanks to efforts to democratize access to AI models, pre-training is expected to increasingly migrate from the current centralized deployments to federated learning (FL) implementations. Meta-learning provides a general framework in which pre-training and fine-tuning can be formalized. Meta-learning-based personalized FL (meta-pFL) moves beyond basic personalization by targeting generalization to new agents and tasks. This paper studies the generalization performance of meta-pFL for a wireless setting in which the agents participating in the pre-training phase, i.e., meta-learning, are connected via a shared wireless channel to the server. Adopting over-the-air computing, we study the trade-off between generalization to new agents and tasks, on the one hand, and convergence, on the other hand. The trade-off arises from the fact that channel impairments may enhance generalization, while degrading convergence. Extensive numerical results validate the theory.

---

## 2. High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation

**论文链接:** [http://arxiv.org/abs/2510.11017v1](http://arxiv.org/abs/2510.11017v1)

**作者:** Runyang Feng, Hyung Jin Chang, Tze Ho Elden Tse, Boeun Kim, Yi Chang, Yixing Gao

**发布时间:** 2025-10-13

**备注:** This paper is accepted to ICCV 2025

### GPT解析

### 总结

本文提出了一种新框架，通过扩展Mamba模型来分别学习全局和局部高分辨率时空表示，用于视频人体姿态估计(VHPE)。该框架包含全局时空Mamba和局部细化Mamba两个组件，有效解决了现有方法在平衡全局和局部动态建模方面的困难。

### 背景

高分辨率时空表示建模对于视频人体姿态估计至关重要，需要同时考虑全局动态上下文和局部运动细节。当前最先进方法在单一建模结构中统一时空学习，难以平衡全局和局部动态建模，且在捕获全局依赖时具有二次复杂度，限制了在高分辨率序列中的应用。

### 目的

提出一个新框架，从两个方面扩展Mamba模型，分别学习VHPE的全局和局部高分辨率时空表示，以解决现有方法的局限性。

### 方法

提出全局时空Mamba，执行6D选择性时空扫描和时空调制扫描合并，从高分辨率序列中高效提取全局表示；引入基于窗口时空扫描的局部细化Mamba，增强局部关键点运动的高频细节。这种方法结合了Mamba的线性复杂度和处理长程上下文的能力。

### 主要发现

在四个基准数据集上的大量实验表明，所提出的模型优于最先进的VHPE方法，同时实现了更好的计算权衡。

### 结论

通过分别建模全局和局部时空表示，可以提高VHPE性能。扩展Mamba框架可以有效地处理高分辨率视频数据，解决了现有方法在计算效率和表示能力之间的平衡问题。

### 翻译

建模高分辨率时空表示，包括全局动态上下文(如整体人体运动趋势)和局部运动细节(如关键点的高频变化)，对于基于视频的人体姿态估计(VHPE)至关重要。当前最先进方法通常在单一类型的建模结构(卷积或基于注意力的块)中统一时空学习，这些方法本质上难以平衡全局和局部动态建模，可能导致网络偏向其中一种，从而产生次优性能。此外，现有VHPE模型在捕获全局依赖时具有二次复杂度，限制了它们在高分辨率序列中的应用。最近，状态空间模型(称为Mamba)在建模具有线性复杂度的长程上下文方面显示出巨大潜力；然而，它们仅限于1D序列数据。在本文中，我们提出了一种新框架，从两个方面扩展Mamba，分别学习VHPE的全局和局部高分辨率时空表示。具体而言，我们首先提出了全局时空Mamba，它执行6D选择性时空扫描和时空调制扫描合并，从高分辨率序列中高效提取全局表示。我们进一步引入了基于窗口时空扫描的局部细化Mamba，以增强局部关键点运动的高频细节。在四个基准数据集上的大量实验表明，所提出的模型优于最先进的VHPE方法，同时实现了更好的计算权衡。


### 论文摘要

Modeling high-resolution spatiotemporal representations, including both global dynamic contexts (e.g., holistic human motion tendencies) and local motion details (e.g., high-frequency changes of keypoints), is essential for video-based human pose estimation (VHPE). Current state-of-the-art methods typically unify spatiotemporal learning within a single type of modeling structure (convolution or attention-based blocks), which inherently have difficulties in balancing global and local dynamic modeling and may bias the network to one of them, leading to suboptimal performance. Moreover, existing VHPE models suffer from quadratic complexity when capturing global dependencies, limiting their applicability especially for high-resolution sequences. Recently, the state space models (known as Mamba) have demonstrated significant potential in modeling long-range contexts with linear complexity; however, they are restricted to 1D sequential data. In this paper, we present a novel framework that extends Mamba from two aspects to separately learn global and local high-resolution spatiotemporal representations for VHPE. Specifically, we first propose a Global Spatiotemporal Mamba, which performs 6D selective space-time scan and spatial- and temporal-modulated scan merging to efficiently extract global representations from high-resolution sequences. We further introduce a windowed space-time scan-based Local Refinement Mamba to enhance the high-frequency details of localized keypoint motions. Extensive experiments on four benchmark datasets demonstrate that the proposed model outperforms state-of-the-art VHPE approaches while achieving better computational trade-offs.

---

## 3. PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion

**论文链接:** [http://arxiv.org/abs/2510.10365v1](http://arxiv.org/abs/2510.10365v1)

**作者:** Linlian Jiang, Rui Ma, Li Gu, Ziqiang Wang, Xinxin Zuo, Yang Wang

**发布时间:** 2025-10-11

**备注:** NeurIPS 2025

### GPT解析

### 总结

PointMAC是一种创新的元学习框架，用于点云补全的测试时适应，通过自监督辅助目标和元辅助学习策略，解决了现有模型无法适应新结构模式和传感器失真的问题，实现了高质量的个体样本补全。

### 背景

点云补全对机器人和增强现实等安全关键应用中的鲁棒3D感知至关重要。现有模型执行静态推理，严重依赖训练期间归纳偏置，限制了它们适应新结构模式和传感器引起失真的能力。

### 目的

解决现有模型无法适应测试时新结构模式和传感器失真的问题，提出PointMAC框架实现测试时鲁棒适应。

### 方法

提出PointMAC框架，通过两个自监督辅助目标模拟结构和传感器级别的不完整性；基于模型无关元学习的元辅助学习策略确保适应与主要任务一致；推理时通过优化辅助损失实时适应共享编码器；引入自适应λ校准机制平衡主要和辅助目标间的梯度。

### 主要发现

在合成、模拟和真实世界数据集上的广泛实验表明，PointMAC通过单独细化每个样本来产生高质量补全，实现了最先进的结果。

### 结论

据我们所知，这是首次将元辅助测试时适应应用于点云补全的工作。

### 翻译

点云补全对于机器人和增强现实等安全关键应用中的鲁棒3D感知至关重要。然而，现有模型执行静态推理，并严重依赖训练期间学习的归纳偏置，限制了它们在测试时适应新结构模式和传感器引起失真的能力。为了解决这一限制，我们提出了PointMAC，一种用于点云补全的元学习框架，实现测试时的鲁棒适应。它无需额外监督即可实现样本特定细化。我们的方法在两个自监督辅助目标下优化补全模型，这些目标模拟结构和传感器级别的不完整性。基于模型无关元学习的元辅助学习策略确保由辅助目标驱动的适应与主要补全任务保持一致。在推理过程中，我们通过优化辅助损失实时适应共享编码器，同时保持解码器固定。为了进一步稳定适应，我们引入了自适应λ校准，一种用于平衡主要和辅助目标之间梯度的元学习机制。在合成、模拟和真实世界数据集上的广泛实验表明，PointMAC通过单独细化每个样本来产生高质量补全，实现了最先进的结果。据我们所知，这是首次将元辅助测试时适应应用于点云补全的工作。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文解决点云补全中的静态推理问题，即现有模型在测试时使用固定推理方式，难以适应新的结构模式和传感器引起的失真。这个问题在现实中很重要，因为点云补全对机器人、自动驾驶和增强现实等安全关键应用至关重要，而现实世界中的点云常因遮挡、有限覆盖和传感器噪声而不完整，现有模型在处理这些情况时表现不佳，生成'通用补全'而非针对特定样本的'样本特定补全'。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从静态推理转向动态、样本特定适应的思路，利用测试时适应框架让模型自我调整。他们设计PointMAC框架，包含Bi-Aux Units执行自监督任务，并采用MAML规范适应过程。该方法借鉴了TTA在动态场景去模糊等领域的应用，元学习在少样本学习中的成功经验，以及点云补全领域的编码器-解码器架构和transformer模型设计思路。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过元辅助学习实现测试时样本特定适应，使模型能根据每个输入的独特几何形状和噪声动态调整内部表示。流程包括：1)网络架构(共享编码器、主要解码器和Bi-Aux Units)；2)训练流程(内部辅助适应、外部主要对齐和自适应λ校准)；3)推理流程(对每个测试样本执行自监督梯度步骤，细化共享编码器，生成样本特定补全)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个将元辅助学习和测试时适应应用于点云补全的框架；2)Bi-Aux Units设计自监督双辅助任务；3)基于MAML的元辅助学习策略；4)自适应λ校准机制。相比之前工作，PointMAC实现了从静态到动态的转变，采用自监督适应而非额外监督，解决了辅助任务与主要任务对齐问题，提高了泛化能力，并提供了端到端的框架。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PointMAC通过元辅助学习的测试时适应框架，使点云补全模型能动态调整每个输入样本的内部表示，在不依赖额外监督的情况下生成高质量、样本特定的补全结果，显著提高了模型对未见过的结构模式和传感器噪声的鲁棒性。'}


### 论文摘要

Point cloud completion is essential for robust 3D perception in safety-critical applications such as robotics and augmented reality. However, existing models perform static inference and rely heavily on inductive biases learned during training, limiting their ability to adapt to novel structural patterns and sensor-induced distortions at test time. To address this limitation, we propose PointMAC, a meta-learned framework for robust test-time adaptation in point cloud completion. It enables sample-specific refinement without requiring additional supervision. Our method optimizes the completion model under two self-supervised auxiliary objectives that simulate structural and sensor-level incompleteness. A meta-auxiliary learning strategy based on Model-Agnostic Meta-Learning (MAML) ensures that adaptation driven by auxiliary objectives is consistently aligned with the primary completion task. During inference, we adapt the shared encoder on-the-fly by optimizing auxiliary losses, with the decoder kept fixed. To further stabilize adaptation, we introduce Adaptive $\lambda$-Calibration, a meta-learned mechanism for balancing gradients between primary and auxiliary objectives. Extensive experiments on synthetic, simulated, and real-world datasets demonstrate that PointMAC achieves state-of-the-art results by refining each sample individually to produce high-quality completions. To the best of our knowledge, this is the first work to apply meta-auxiliary test-time adaptation to point cloud completion.

---

## 4. Beyond 'Templates': Category-Agnostic Object Pose, Size, and Shape Estimation from a Single View

**论文链接:** [http://arxiv.org/abs/2510.11687v1](http://arxiv.org/abs/2510.11687v1)

**作者:** Jinyu Zhang, Haitao Lin, Jiashu Hou, Xiangyang Xue, Yanwei Fu

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了一种统一的、类别不可知的框架，可以从单个RGB-D图像同时预测物体的6D姿态、大小和密集形状，无需模板、CAD模型或类别标签，在多个基准数据集上实现了最先进性能，并展现出强大的零样本泛化能力。

### 背景

从视觉输入估计物体的6D姿态、大小和形状是计算机视觉中的基础问题，在机器人抓取和操作中有关键应用。现有方法要么依赖特定于对象的先验，要么由于姿态-形状纠缠和多阶段管道而泛化能力有限。

### 目的

开发一个无需对象特定先验的统一框架，能够从单个RGB-D图像同时预测6D姿态、大小和密集形状，并实现跨类别的强泛化能力。

### 方法

使用Transformer编码器（由Mixture-of-Experts增强）融合来自视觉基础模型的密集2D特征和部分3D点云，采用并行解码器进行姿态-大小估计和形状重建，实现28 FPS的实时推理。仅在SOPE数据集的149个类别的合成数据上进行训练。

### 主要发现

在四个不同基准数据集（涵盖300多个类别）上评估，在已见类别上达到最先进精度，同时对未见到的真实世界物体表现出强大的零样本泛化能力。

### 结论

该框架为机器人和具身AI中的开放集6D理解建立了新标准，无需对象特定先验即可实现高精度和强泛化能力。

### 翻译

从视觉输入估计物体的6D姿态、大小和形状是计算机视觉中的一个基础问题，在机器人抓取和操作中具有关键应用。现有方法要么依赖于特定于对象的先验（如CAD模型或模板），要么由于姿态-形状纠缠和多阶段管道而在跨类别泛化方面受到限制。在这项工作中，我们提出了一个统一的、类别不可知的框架，可以从单个RGB-D图像同时预测6D姿态、大小和密集形状，测试时不需要模板、CAD模型或类别标签。我们的模型使用由Mixture-of-Experts增强的Transformer编码器融合来自视觉基础模型的密集2D特征和部分3D点云，并采用并行解码器进行姿态-大小估计和形状重建，实现28 FPS的实时推理。仅在SOPE数据集的149个类别的合成数据上进行训练后，我们的框架在四个不同的基准数据集SOPE、ROPE、ObjaversePose和HANDAL上进行了评估，涵盖300多个类别。它在已见类别上实现了最先进的精度，同时展现出对未见真实世界物体 remarkably强的零样本泛化能力，为机器人和具身AI中的开放集6D理解建立了新标准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从单张RGB-D图像中估计物体6D位姿、大小和形状的问题，而不需要依赖特定模板、CAD模型或类别标签。这个问题在机器人抓取、操作以及具身AI领域至关重要，因为现有方法要么需要物体特定先验知识，要么在跨类别泛化能力上有限，限制了在开放场景下的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：实例级方法需要参考图像或CAD模型，类别级方法存在姿态-形状纠缠和多阶段流水线问题。他们设计了一个统一框架，融合视觉基础模型的密集2D特征与3D点云，使用增强Mixture-of-Experts的Transformer编码器，并采用并行解码器。该方法借鉴了DGCNN处理特征、Transformer架构、RADIOv2.5基础模型提取语义特征、DenseFusion的特征融合方式以及NOCS坐标系表示等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的、类别无关的框架，同时预测6D位姿、大小和形状，避免多阶段流水线，实现测试时无需模板或类别标签。整体流程：1)使用RADIOv2.5提取RGB图像的密集2D特征；2)将2D特征与3D点云坐标融合；3)通过DGCNN处理融合特征；4)使用带有Mixture-of-Experts的Transformer编码器生成全局表示；5)通过并行解码器进行姿态-大小直接回归和形状重建(采用粗到细策略)；6)结合多种损失函数进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个统一类别无关框架同时估计6D位姿、大小和形状；2)设计可扩展架构融合2D特征与3D点云，使用增强Mixture-of-Experts的Transformer；3)实现28 FPS实时推理；4)仅合成数据训练但展现强大零样本泛化；5)构建ObjaversePose数据集。不同之处：无需测试时模板/CAD模型/类别标签；统一端到端框架避免多阶段处理；同时处理位姿、大小和形状捕获相互依赖；使用MoE提高对不同形状分布建模能力；合成数据训练但能泛化到真实世界和未见类别。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种统一的、类别无关的框架，能够从单张RGB-D图像中实时估计物体的6D位姿、大小和完整形状，无需测试时的模板或类别标签，并在多种未见过的真实物体上展示了强大的零样本泛化能力。'}


### 论文摘要

Estimating an object's 6D pose, size, and shape from visual input is a fundamental problem in computer vision, with critical applications in robotic grasping and manipulation. Existing methods either rely on object-specific priors such as CAD models or templates, or suffer from limited generalization across categories due to pose-shape entanglement and multi-stage pipelines. In this work, we propose a unified, category-agnostic framework that simultaneously predicts 6D pose, size, and dense shape from a single RGB-D image, without requiring templates, CAD models, or category labels at test time. Our model fuses dense 2D features from vision foundation models with partial 3D point clouds using a Transformer encoder enhanced by a Mixture-of-Experts, and employs parallel decoders for pose-size estimation and shape reconstruction, achieving real-time inference at 28 FPS. Trained solely on synthetic data from 149 categories in the SOPE dataset, our framework is evaluated on four diverse benchmarks SOPE, ROPE, ObjaversePose, and HANDAL, spanning over 300 categories. It achieves state-of-the-art accuracy on seen categories while demonstrating remarkably strong zero-shot generalization to unseen real-world objects, establishing a new standard for open-set 6D understanding in robotics and embodied AI.

---

## 5. NV3D: Leveraging Spatial Shape Through Normal Vector-based 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2510.11632v1](http://arxiv.org/abs/2510.11632v1)

**作者:** Krittin Chaowakarn, Paramin Sangwongngam, Nang Htet Htet Aung, Chalie Charoenlarpnopparut

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了一种名为NV3D的新型3D物体检测模型，利用从体素邻居获取的局部法向量特征来提高检测性能，并通过两种采样策略减少数据量同时保持性能。

### 背景

近期自动驾驶车辆3D物体检测研究试图通过多模态设置或从LiDAR点云中提取局部模式来丰富特征，但多模态方法面临特征对齐挑战，局部特征获取对于复杂任务可能过于简化。

### 目的

开发一种新的3D物体检测模型NV3D，解决现有方法的局限性，提高检测性能并减少数据量。

### 方法

NV3D利用基于K近邻和主成分分析计算每个体素的法向量作为局部特征，确定表面与目标实体间关系；提供两种采样策略：基于法向量密度的采样和基于视场感知的基于bin的采样；应用元素级注意力融合机制，将体素特征作为查询和值，法向量特征作为键。

### 主要发现

使用两种采样策略可消除高达55%的数据同时保持性能；在KITTI数据集上，NV3D在汽车和骑行者检测方面优于基线；不使用采样时，NV3D比Voxel R-CNN分别高出2.61%和4.23% mAP；使用采样后，仍比基线高1.56% mAP，同时过滤了约55%体素。

### 结论

NV3D模型通过利用法向量特征和有效采样策略，能够在减少数据量的同时提高3D物体检测性能，特别是在具有特定空间形状的物体检测上表现出色。

### 翻译

近期关于自动驾驶车辆3D物体检测的研究试图通过多模态设置或从LiDAR点云中提取局部模式来丰富特征。然而，多模态方法面临特征对齐的重大挑战，而局部特征获取对于复杂的3D物体检测任务可能过于简化。在本文中，我们提出了一种新型模型NV3D，它利用从体素邻居获取的局部特征，作为使用K近邻和主成分分析按每个体素基础计算的法向量。这种信息丰富的特征使NV3D能够确定表面与相关目标实体之间的关系，包括汽车、行人或骑行者。在法向量提取过程中，NV3D提供两种不同的采样策略：基于法向量密度的采样和基于视场感知的基于bin的采样，允许消除高达55%的数据同时保持性能。此外，我们应用了元素级注意力融合，将体素特征作为查询和值，法向量特征作为键，类似于注意力机制。我们的方法在KITTI数据集上训练，并在汽车和骑行者检测方面表现出优越性能，这得益于它们的空间形状。在验证集上，不使用采样的NV3D达到86.60%和80.18%的平均精度均值(mAP)，分别比基线Voxel R-CNN高出2.61%和4.23% mAP。使用两种采样后，NV3D在汽车检测中达到85.54% mAP，比基线高出1.56% mAP，尽管大约55%的体素被过滤掉。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D目标检测中的特征提取效率和计算复杂度问题。在自动驾驶领域，精确检测车辆、行人和骑行者等对象至关重要，但现有方法要么依赖多模态数据融合面临特征对齐挑战，要么仅使用局部特征提取对复杂任务过于简化。提高检测精度和效率对于确保自动驾驶系统的安全性和可靠性具有现实意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过观察发现点云数据中存在冗余，特别是近距离密集点云集群，并提出几何特征可从邻近点提取的假设。他们借鉴了Voxel R-CNN作为基础架构，采用KNN和PCA方法提取法向量特征，受VirConv中关于远距离点影响更大的启发，并参考了注意力机制设计了元素级注意力融合。这种方法结合了传统几何处理和深度学习的优势，既减少了计算复杂度又保留了关键空间信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用局部体素特征计算法向量作为新特征表示，通过采样减少冗余数据，并融合体素和法向量特征。实现流程包括：1)将LiDAR点云转换为体素；2)使用KNN和PCA提取法向量特征；3)应用法向量密度采样(去除密度>0.7的50%体素)和FOV感知的基于bin采样(保持空间连续性)；4)通过元素级注意力机制融合两种特征；5)将融合特征输入Voxel R-CNN架构进行目标检测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入法向量作为3D目标检测的新特征表示；2)提出两种创新采样策略(法向量密度采样和FOV感知的基于bin采样)；3)设计元素级注意力融合机制。相比之前工作，NV3D专注于单模态LiDAR数据处理避免了特征对齐问题；使用法向量而非直接处理点云减少了计算复杂度；采样策略考虑了法向量和视野因素保留了更多有用信息；将局部特征转换为法向量表示提高了效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NV3D通过引入法向量特征和创新的采样策略，在保持高性能的同时显著提高了3D目标检测的效率，特别是在车辆和骑行者检测任务中表现优异。'}


### 论文摘要

Recent studies in 3D object detection for autonomous vehicles aim to enrich features through the utilization of multi-modal setups or the extraction of local patterns within LiDAR point clouds. However, multi-modal methods face significant challenges in feature alignment, and gaining features locally can be oversimplified for complex 3D object detection tasks. In this paper, we propose a novel model, NV3D, which utilizes local features acquired from voxel neighbors, as normal vectors computed per voxel basis using K-nearest neighbors (KNN) and principal component analysis (PCA). This informative feature enables NV3D to determine the relationship between the surface and pertinent target entities, including cars, pedestrians, or cyclists. During the normal vector extraction process, NV3D offers two distinct sampling strategies: normal vector density-based sampling and FOV-aware bin-based sampling, allowing elimination of up to 55% of data while maintaining performance. In addition, we applied element-wise attention fusion, which accepts voxel features as the query and value and normal vector features as the key, similar to the attention mechanism. Our method is trained on the KITTI dataset and has demonstrated superior performance in car and cyclist detection owing to their spatial shapes. In the validation set, NV3D without sampling achieves 86.60% and 80.18% mean Average Precision (mAP), greater than the baseline Voxel R-CNN by 2.61% and 4.23% mAP, respectively. With both samplings, NV3D achieves 85.54% mAP in car detection, exceeding the baseline by 1.56% mAP, despite roughly 55% of voxels being filtered out.

---

## 6. SNAP: Towards Segmenting Anything in Any Point Cloud

**论文链接:** [http://arxiv.org/abs/2510.11565v1](http://arxiv.org/abs/2510.11565v1)

**作者:** Aniket Gupta, Hanhui Wang, Charles Saunders, Aruni RoyChowdhury, Hanumant Singh, Huaizu Jiang

**发布时间:** 2025-10-13

**备注:** Project Page, https://neu-vi.github.io/SNAP/

### GPT解析

### 总结

SNAP是一个统一的交互式3D点云分割模型，支持跨不同领域的点和文本提示，通过多数据集训练和领域自适应归一化实现跨领域通用性，在多个基准测试上取得了最先进或具有竞争力的结果。

### 背景

交互式3D点云分割通过用户引导提示能够高效标注复杂3D场景，但当前方法通常局限于单一领域（室内或室外）和单一形式的用户交互（空间点击或文本提示）。在多个数据集上训练通常会导致负迁移，导致缺乏通用性的领域特定工具。

### 目的

提出一个统一模型，支持跨不同领域的基于点和基于文本的提示进行交互式3D分割，解决现有方法的局限性。

### 方法

提出SNAP（Segment Anything in Any Point cloud）模型，通过在涵盖室内、室外和空中环境的7个数据集上训练实现跨领域通用性，使用领域自适应归一化来防止负迁移。对于文本提示的分割，自动生成掩码提案并与文本查询的CLIP嵌入进行匹配，支持全景和开放词汇分割。

### 主要发现

SNAP在空间提示分割的9个零样本基准测试中，有8个达到了最先进的性能，在所有5个文本提示基准测试中展示了具有竞争力的结果。

### 结论

统一模型可以匹配或超越专门的领域特定方法，为可扩展的3D标注提供实用工具。

### 翻译

交互式3D点云分割通过用户引导提示能够高效标注复杂3D场景。然而，当前方法通常在范围上局限于单一领域（室内或室外），并且局限于单一形式的用户交互（空间点击或文本提示）。此外，在多个数据集上训练通常会导致负迁移，产生缺乏通用性的领域特定工具。为解决这些限制，我们提出了SNAP（Segment Anything in Any Point cloud），这是一个统一的交互式3D分割模型，支持跨不同领域的基于点和基于文本的提示。我们的方法通过在涵盖室内、室外和空中环境的7个数据集上训练实现跨领域通用性，同时采用领域自适应归一化来防止负迁移。对于文本提示的分割，我们无需人工干预自动生成掩码提案，并将其与文本查询的CLIP嵌入进行匹配，实现全景和开放词汇分割。大量实验证明，SNAP始终提供高质量的分割结果。我们在空间提示分割的9个零样本基准测试中的8个上取得了最先进的性能，并在所有5个文本提示基准测试上展示了具有竞争力的结果。这些结果表明，统一模型可以匹配或超越专门的领域特定方法，为可扩展的3D标注提供实用工具。项目页面位于https://neu-vi.github.io/SNAP/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决当前3D点云交互分割方法局限于单一领域（室内或室外）和单一交互形式（空间点击或文本提示）的问题，以及多数据集训练导致的负迁移问题。这个问题很重要，因为3D场景标注需要大量人工努力，而缺乏通用性的工具限制了它们作为高效标注工具的采用，领域特定工具需要单独训练增加了使用成本，缺乏灵活性也限制了用户适应不同的标注需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到2D图像中SAM成功的启发，认识到当前方法缺乏泛化能力和交互灵活性，因此决定创建一个统一的跨领域模型。他们设计时借鉴了现有工作：使用SAM的掩码解码器设计，借鉴AGILE3D和Interactive4D的点击采样策略，利用CLIP模型处理文本提示和嵌入匹配。核心创新是提出领域自适应归一化来解决跨领域训练的负迁移问题，并设计自动提示点生成算法实现无需人工干预的分割。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的模型，支持在不同领域（室内、室外、空中）的3D点云上进行多模态（空间和文本）交互式分割。整体流程包括：1)点云编码：使用Point Transformer V3提取点嵌入并应用领域自适应归一化；2)空间提示分割：编码点击点，通过掩码解码器生成掩码；3)文本提示分割：自动生成提示点，生成掩码提案并匹配CLIP嵌入；4)训练：结合多种损失函数优化模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)领域自适应归一化解决跨领域训练负迁移；2)统一的多领域模型支持室内、室外和空中场景；3)多模态提示同时支持空间点击和文本描述；4)自动提示点生成算法实现无需人工干预的分割；5)开放词汇分割支持新类别。相比之前工作，SNAP突破了单一领域限制，支持多种提示方式，解决了负迁移问题，可直接处理点云无需RGB图像，并在多个零样本测试中达到最先进性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SNAP是一个统一的、支持多模态提示的3D点云交互分割模型，通过领域自适应归一化实现了跨领域泛化能力，并在多种场景的分割任务中达到了最先进的性能。'}


### 论文摘要

Interactive 3D point cloud segmentation enables efficient annotation of complex 3D scenes through user-guided prompts. However, current approaches are typically restricted in scope to a single domain (indoor or outdoor), and to a single form of user interaction (either spatial clicks or textual prompts). Moreover, training on multiple datasets often leads to negative transfer, resulting in domain-specific tools that lack generalizability. To address these limitations, we present \textbf{SNAP} (\textbf{S}egment a\textbf{N}ything in \textbf{A}ny \textbf{P}oint cloud), a unified model for interactive 3D segmentation that supports both point-based and text-based prompts across diverse domains. Our approach achieves cross-domain generalizability by training on 7 datasets spanning indoor, outdoor, and aerial environments, while employing domain-adaptive normalization to prevent negative transfer. For text-prompted segmentation, we automatically generate mask proposals without human intervention and match them against CLIP embeddings of textual queries, enabling both panoptic and open-vocabulary segmentation. Extensive experiments demonstrate that SNAP consistently delivers high-quality segmentation results. We achieve state-of-the-art performance on 8 out of 9 zero-shot benchmarks for spatial-prompted segmentation and demonstrate competitive results on all 5 text-prompted benchmarks. These results show that a unified model can match or exceed specialized domain-specific approaches, providing a practical tool for scalable 3D annotation. Project page is at, https://neu-vi.github.io/SNAP/

---

## 7. Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Model

**论文链接:** [http://arxiv.org/abs/2510.11509v1](http://arxiv.org/abs/2510.11509v1)

**作者:** Ruiping Liu, Junwei Zheng, Yufan Chen, Zirui Wang, Kunyu Peng, Kailun Yang, Jiaming Zhang, Marc Pollefeys, Rainer Stiefelhagen

**发布时间:** 2025-10-13

**备注:** Accepted to NeurIPS 2025 Datasets and Benchmarks Track. Dataset and  Code: https://github.com/RuipingL/Situat3DChange

### GPT解析

### 总结

本文介绍了Situat3DChange数据集和SCReasoner方法，用于解决3D动态场景和情境理解的不完整问题，通过大规模数据集和高效3D MLLM方法提升动态环境变化的理解能力。

### 背景

物理环境和情况本质上是动态的，但当前3D数据集和评估基准往往只专注于动态场景或动态情况的孤立研究，导致对动态环境的理解不完整。

### 目的

克服现有3D数据集的局限性，引入支持情境感知变化理解任务的大规模数据集，并开发高效方法进行点云比较和动态场景理解。

### 方法

构建Situat3DChange数据集，包含121K问答对、36K变化描述和17K重排指令；利用11K人类环境变化观察建立共享心智模型；融合自我中心和他者中心视角及空间关系；提出SCReasoner方法进行高效点云比较。

### 主要发现

Situat3DChange任务上的评估突显了MLLMs在动态场景和情境理解方面的进展和局限性；数据扩展和跨域迁移实验证明了使用Situat3DChange作为训练数据集的任务无关有效性。

### 结论

Situat3DChange数据集和SCReasoner方法为动态场景和情境理解提供了新的工具和视角，有助于提升AI对环境动态变化的理解能力。

### 翻译

物理环境和情况本质上是动态的，然而当前的3D数据集和评估基准往往只专注于动态场景或动态情况的孤立研究，导致理解不完整。为克服这些限制，我们引入Situat3DChange，一个支持三种情境感知变化理解任务的大规模数据集：121K问答对，36K用于感知任务的变化描述，以及17K用于行动任务的重排指令。为构建这一大规模数据集，Situat3DChange利用11K人类对环境变化的观察来建立人类-AI协作的共享心智模型和共享情境感知。这些观察融合了自我中心和他者中心视角以及分类和坐标空间关系，通过LLM集成以支持对情境变化的理解。为解决比较同一场景中具有微小变化的点云对这一挑战，我们提出SCReasoner，一种高效的3D MLLM方法，能够以最小的参数开销进行有效的点云比较，且语言解码器不需要额外令牌。在Situat3DChange任务上的全面评估突显了MLLMs在动态场景和情境理解方面的进展和局限性。关于数据扩展和跨域迁移的额外实验证明了使用Situat3DChange作为MLLMs训练数据集的任务无关有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决当前3D数据集和评估基准孤立关注动态场景或动态情境的问题，导致对环境变化的理解不完整。这个问题很重要，因为物理环境和情境本质上是动态的，即使是微小的位置变化对视障人士也可能造成障碍，有效的人机协作需要建立共享的心理模型和情境感知能力，而现有方法无法同时捕捉动态场景和情境感知。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析人类与机器人认知差异发现人类以柱面坐标感知环境而机器人以笛卡尔坐标感知，这导致无法共享心理地图。作者收集11K人类注释建立基于人类感知的共享模型，并整合自我中心和异中心视角以及分类和坐标空间关系。作者借鉴了3RScan数据集、3DSSG场景图、MSQA的情境采样方法、LEO框架，并使用GPT-4生成类人文本，同时采用Mamba的选择性状态空间模型和星操作进行token选择与融合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建情境感知的3D变化理解数据集并开发高效的SCReasoner架构，通过结合人类注释和LLM生成保留人类感知框架。数据集构建流程包括情境采样、长文本生成、查询生成、问答生成和数据质量控制。SCReasoner实现流程使用共同编码器将两个点云嵌入token，从前一场景选择信息丰富token并与当前场景token融合，使用Mamba进行token选择，星操作进行token融合，构建在LEO框架上仅添加少量额外参数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Situat3DChange数据集首个整合动态场景和情境感知，包含121K问答对、36K变化描述和17K重新排列指令；2) SCReasoner架构专门处理成对点云，使用Mamba和星操作实现高效比较；3) 基于人类感知框架整合自我中心和异中心视角。相比之前工作，该数据集同时关注动态场景和情境感知，SCReasoner专门设计用于点云比较并关注差异而非冗余，评估方法更全面且包含特殊距离评估指标。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了首个情境感知的3D变化理解数据集和高效的SCReasoner架构，通过整合人类感知与AI系统，实现了对动态环境和情境变化的全面理解，为人机协作在动态环境中的适应性交互提供了新的基础。'}


### 论文摘要

Physical environments and circumstances are fundamentally dynamic, yet current 3D datasets and evaluation benchmarks tend to concentrate on either dynamic scenarios or dynamic situations in isolation, resulting in incomplete comprehension. To overcome these constraints, we introduce Situat3DChange, an extensive dataset supporting three situation-aware change understanding tasks following the perception-action model: 121K question-answer pairs, 36K change descriptions for perception tasks, and 17K rearrangement instructions for the action task. To construct this large-scale dataset, Situat3DChange leverages 11K human observations of environmental changes to establish shared mental models and shared situational awareness for human-AI collaboration. These observations, enriched with egocentric and allocentric perspectives as well as categorical and coordinate spatial relations, are integrated using an LLM to support understanding of situated changes. To address the challenge of comparing pairs of point clouds from the same scene with minor changes, we propose SCReasoner, an efficient 3D MLLM approach that enables effective point cloud comparison with minimal parameter overhead and no additional tokens required for the language decoder. Comprehensive evaluation on Situat3DChange tasks highlights both the progress and limitations of MLLMs in dynamic scene and situation understanding. Additional experiments on data scaling and cross-domain transfer demonstrate the task-agnostic effectiveness of using Situat3DChange as a training dataset for MLLMs.

---

## 8. Into the Unknown: Towards using Generative Models for Sampling Priors of Environment Uncertainty for Planning in Configuration Spaces

**论文链接:** [http://arxiv.org/abs/2510.11014v1](http://arxiv.org/abs/2510.11014v1)

**作者:** Subhransu S. Bhattacharjee, Hao Lu, Dylan Campbell, Rahul Shome

**发布时间:** 2025-10-13

**备注:** Under Review

### GPT解析

### 总结

本文提出了一种基于采样的管道，利用大规模预训练生成模型在零样本方式下产生概率先验，捕捉环境不确定性和空间-语义关系，用于部分可观察性条件下的机器人规划。

### 背景

在部分可观察性条件下进行规划时，先验信息非常重要，但在实践中难以获取。

### 目的

开发一种利用生成模型产生概率先验的方法，以零样本方式捕捉环境不确定性和空间-语义关系，并直接用于配置空间规划。

### 方法

提出一个基于采样的管道，基于部分观察条件恢复完整的RGB-D点云样本，包含占用信息和目标语义；建立Matterport3D基准测试，场景为只能通过门看到部分区域的房间，机器人需要导航到未观察到的目标物体。

### 主要发现

有效先验必须表示未观察区域中的占用和目标位置不确定性；方法恢复了与真实情况一致的常识空间语义，生成了多样化的、干净的3D点云，可用于运动规划。

### 结论

生成模型作为机器人规划中先验信息的丰富来源具有很大潜力。

### 翻译

先验信息对于部分可观察性条件下的规划至关重要，但在实践中难以获取。我们提出了一种基于采样的管道，利用大规模预训练生成模型以零样本方式产生概率先验，捕捉环境不确定性和空间-语义关系。基于部分观察条件，该管道能够恢复包含占用信息和目标语义的完整RGB-D点云样本，可直接用于配置空间规划。我们建立了一个Matterport3D基准测试，场景为只能通过门看到部分区域的房间，机器人必须导航到未观察到的目标物体。此场景的有效先验必须表示未观察区域中的占用和目标位置不确定性。实验表明，我们的方法恢复了与真实情况一致的常识空间语义，生成了多样化的、干净的3D点云，可用于运动规划，突显了生成模型作为机器人规划中丰富先验来源的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人如何在部分可观测环境中（如只能通过门缝看到房间一部分）获取环境不确定性先验信息的问题。这个问题很重要，因为随着机器人应用扩展到真实世界环境，环境不确定性不可避免，而传统的规划方法依赖于难以获取且可能不准确的手工或预编程先验信息。准确的环境不确定性先验对机器人在未知区域进行有效导航和规划至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先指出在部分可观测环境下的规划需要先验信息但难以获取，然后注意到生成式视觉模型能生成符合数据分布的内容并可根据条件输入。他们设计了一个基于采样的流程，借鉴了多项现有工作：VLM用于图像分类和物体提示生成、FLUX图像外推模型用于场景扩展、单目深度估计器用于3D重建、以及现有的采样规划算法。作者将这些现有技术整合成一个完整的流水线，用于生成条件于部分观测的完整环境样本。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用大规模预训练的生成模型作为环境采样器，根据部分观测生成捕捉环境不确定性和空间-语义关系的3D环境样本，然后使用这些样本作为先验信息进行配置空间规划。整体流程包括：1) VLM提示机制对输入图像分类并生成相关物体提示；2) 基于图像的生成使用FLUX模型扩展场景图像；3) 物体分割和地板估计进行语义分割和对齐；4) 深度估计和对齐将RGB-D转换为点云；5) 配置空间规划使用采样先验进行运动规划。整个流程约需10.5秒生成一个样本。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 利用预训练生成模型采样环境不确定性先验的流程；2) 零样本方法直接生成有用先验；3) 空间-语义先验的形式化方法；4) 基于Matterport3D的新数据集；5) 通过模拟运动规划验证实用性。相比之前工作，本文与传统场景完成方法不同，后者追求单一一致重建，而本文捕捉多样性同时确保干净几何；区别于现有生成模型应用，后者专注于目标分布或策略优化，而本文构建环境先验；不同于显式环境先验，后者是空间-语义地图或场景完成模型，而本文采样扩展视野上的分布；也区别于高计算需求的3D生成方法，本文使用更高效的2D生成模型结合单目深度估计。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种利用预训练生成模型从部分观测采样环境不确定性先验的创新方法，使机器人能够在未观测区域进行有效的空间-语义推理和规划。'}


### 论文摘要

Priors are vital for planning under partial observability, yet difficult to obtain in practice. We present a sampling-based pipeline that leverages large-scale pretrained generative models to produce probabilistic priors capturing environmental uncertainty and spatio-semantic relationships in a zero-shot manner. Conditioned on partial observations, the pipeline recovers complete RGB-D point cloud samples with occupancy and target semantics, formulated to be directly useful in configuration-space planning. We establish a Matterport3D benchmark of rooms partially visible through doorways, where a robot must navigate to an unobserved target object. Effective priors for this setting must represent both occupancy and target-location uncertainty in unobserved regions. Experiments show that our approach recovers commonsense spatial semantics consistent with ground truth, yielding diverse, clean 3D point clouds usable in motion planning, highlight the promise of generative models as a rich source of priors for robotic planning.

---

## 9. rareboost3d: a synthetic lidar dataset with enhanced rare classes

**论文链接:** [http://arxiv.org/abs/2510.10876v1](http://arxiv.org/abs/2510.10876v1)

**作者:** Shutong Lin, Zhengkang Xiang, Jianzhong Qi, Kourosh Khoshelham

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文通过引入合成数据集和跨域语义对齐方法解决了点云数据中的长尾问题，提高了激光雷达感知技术的性能。

### 背景

真实世界点云数据集对基于激光雷达的感知技术（如自动驾驶中的物体分割）有重要贡献，但某些罕见类别的实例数量有限导致长尾问题仍然存在。

### 目的

解决现有数据集中罕见类别实例不足的问题，通过合成数据补充真实世界数据。

### 方法

提出了名为RareBoost3D的合成点云数据集，以及名为CS Loss的跨域语义对齐方法，用于对齐不同域中相同类别的特征表示。

### 主要发现

实验结果表明，跨域语义对齐方法显著提高了激光雷达点云分割模型在真实世界数据上的性能。

### 结论

结合合成数据和真实世界数据，并应用跨域语义对齐方法，可以有效解决长尾问题，提升模型性能。

### 翻译

真实世界的点云数据集对基于激光雷达的感知技术的发展做出了重大贡献，例如自动驾驶中的物体分割。然而，由于某些罕见类别中的实例数量有限，长尾问题仍然是现有数据集的主要挑战。为了解决这个问题，我们引入了一个名为RareBoost3D的新型合成点云数据集，通过为真实世界数据集中稀少的物体类别提供更多实例来补充现有的真实世界数据集。为了有效利用合成和真实世界数据，我们进一步提出了一个名为CS Loss的跨域语义对齐方法，用于对齐不同域中相同类别的特征表示。实验结果表明，这种对齐方法显著提高了激光雷达点云分割模型在真实世界数据上的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决的是LiDAR点云数据集中的'长尾问题'，即常见类别（如汽车）的实例数量远多于稀有类别（如行人、自行车等）。这个问题在自动驾驶领域非常重要，因为模型需要准确识别各种类别的对象，而稀有类别样本不足会导致对这些关键对象的识别能力下降，可能影响自动驾驶系统的安全性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到真实世界点云数据集的局限性：获取标注成本高、耗时长且存在类别不平衡。他们借鉴了多数据集联合训练和合成到真实领域迁移学习的思路，选择使用CARLA模拟器生成合成数据。特别的是，他们没有简单复制真实数据的分布，而是主动增加稀有类别的实例数量。为了解决合成与真实数据间的域差距，他们基于PointDR的跨域特征对齐方法，采用对比学习技术设计了CSC损失函数，这借鉴了对比学习在特征对齐方面的成功应用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个专门增强稀有类别的大规模合成LiDAR数据集，并通过对比学习对齐合成与真实数据中相同类别的特征表示。实现流程包括：1) 使用CARLA模拟器生成8个不同地图的LiDAR序列，特别增加稀有类别的实例；2) 为真实和合成数据分别构建类别特征原型并存储在记忆库中；3) 使用对比学习使相同类别的特征在共享语义空间中接近，不同类别相互分离；4) 结合语义分割损失和对比损失进行模型训练，减少域差距的影响。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出了RareBoost3D数据集，不仅规模大，还特别增加了稀有类别的实例数量；2) 设计了跨域语义一致性(CSC)损失函数，通过对比学习而非复杂的对抗训练来实现域对齐；3) 实验证明调整稀有类别在合成数据中的比例可以显著提升这些类别的分割性能。相比之前的工作，不同之处在于：与SynthmanticLiDAR不同，RareBoost3D不复制而是主动改变类别分布；与SynLiDAR和ePointDA不同，CSC损失不需要对抗训练；传统数据增强方法主要在几何层面操作，而RareBoost3D引入了全新样本。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RareBoost3D数据集和跨域语义一致性损失函数通过增加稀有类别的样本数量并有效对齐合成与真实数据的特征表示，显著提升了LiDAR点云分割模型在稀有类别上的性能。'}


### 论文摘要

Real-world point cloud datasets have made significant contributions to the development of LiDAR-based perception technologies, such as object segmentation for autonomous driving. However, due to the limited number of instances in some rare classes, the long-tail problem remains a major challenge in existing datasets. To address this issue, we introduce a novel, synthetic point cloud dataset named RareBoost3D, which complements existing real-world datasets by providing significantly more instances for object classes that are rare in real-world datasets. To effectively leverage both synthetic and real-world data, we further propose a cross-domain semantic alignment method named CSC loss that aligns feature representations of the same class across different domains. Experimental results demonstrate that this alignment significantly enhances the performance of LiDAR point cloud segmentation models over real-world data.

---

## 10. MATStruct: High-Quality Medial Mesh Computation via Structure-aware Variational Optimization

**论文链接:** [http://arxiv.org/abs/2510.10751v1](http://arxiv.org/abs/2510.10751v1)

**作者:** Ningna Wang, Rui Xu, Yibo Yin, Zichun Zhong, Taku Komura, Wenping Wang, Xiaohu Guo

**发布时间:** 2025-10-12

**DOI:** 10.1145/3757377.3763840

### GPT解析

### 总结

该论文提出了一种新的优化框架，用于计算中轴变换，同时保留中轴结构并确保高质量的中轴网格。该方法基于结构感知的粒子优化流程，由限制性幂图引导，通过球形二次误差度量和高斯核能量来约束和优化中轴球的分布。相比现有方法，该技术产生更清洁、准确的中轴结构，具有更好的几何保真度、拓扑正确性和明确的结构分解。

### 背景

中轴结构由相互连接的薄片、接缝和连接点组成，为3D形状提供自然的体积分解。现有的中轴变换计算方法存在一些局限性，如特征保持方法(MATFP和MATTopo)产生的中轴结构不够清洁和准确，而基于体素、点云和变分的方法未能将结构感知集成到优化过程中。

### 目的

开发一种新的优化框架，能够同时保留中轴结构并确保高质量的中轴网格，克服现有方法的局限性，产生具有更好几何保真度、拓扑正确性和明确结构分解的中轴网格。

### 方法

提出了一种基于结构感知的粒子优化流程，由限制性幂图(RPD)引导，将输入体积划分为凸单元，其对偶编码了中轴网格的连通性。通过球形二次误差度量(SQEM)投影强制执行结构感知，约束中轴球的移动，同时使用高斯核能量鼓励均匀的空间分布。

### 主要发现

1. 与特征保持方法(MATFP和MATTopo)相比，新方法产生更清洁、更准确的中轴结构，网格质量显著提高；2. 与基于体素、点云和变分的方法相比，该框架首次将结构感知集成到优化过程中；3. 产生的中轴网格具有优越的几何保真度、拓扑正确性和明确的结构分解。

### 结论

该研究成功开发了一种新的优化框架，能够有效计算中轴变换并生成高质量的中轴网格。该方法通过结构感知的粒子优化流程和限制性幂图引导，克服了现有方法的局限性，为中轴变换计算提供了更有效的解决方案。

### 翻译

我们提出了一种用于计算中轴变换的新型优化框架，该框架同时保留中轴结构并确保高质量的中轴网格。中轴结构由相互连接的薄片、接缝和连接点组成，为3D形状提供自然的体积分解。我们的方法引入了一种基于结构感知的粒子优化流程，由限制性幂图(RPD)引导，该图将输入体积划分为凸单元，其对偶编码了中轴网格的连通性。通过球形二次误差度量(SQEM)投影强制执行结构感知，约束中轴球的移动，同时高斯核能量鼓励均匀的空间分布。与特征保持方法如MATFP和MATTopo相比，我们的方法产生更清洁、更准确的中轴结构，网格质量显著提高。与基于体素、点云和变分的方法相比，我们的框架是第一个将结构感知集成到优化过程中的方法，产生具有优越几何保真度、拓扑正确性和明确结构分解的中轴网格。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决计算高质量中轴网格（medial mesh）时难以同时保持中轴结构（medial structure）和网格质量的问题。中轴变换作为形状分析的基础，能捕捉形状的拓扑和几何特性，在形状分析、识别、匹配等下游应用中至关重要。特别是在CAD模型中，清晰的中轴结构对工程设计、制造和模拟有重要价值，而现有方法无法同时保证结构清晰度和几何质量，限制了中轴变换的实际应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法如MATFP和MATTopo的局限性，特别是在球体分类和球体过度拥挤方面的问题。作者发现基于表面的RPD分类方法在球体偏离中轴时会导致误分类，因此提出使用体积RPD而非表面RPD来解决分类问题。作者借鉴了粒子优化方法来促进球体均匀分布，并设计了球形二次误差度量（SQEM）来约束球体移动，确保结构感知的优化。同时借鉴了MATTopo的拓扑保持策略和球形收缩算法用于投影回中轴，但进行了创新性改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结构感知的变分优化同时保持中轴结构和确保高质量的中轴网格，使用体积RPD进行更准确的球体分类，并利用基于粒子的优化方案结合SQEM约束确保球体沿着正确的子结构移动。整体流程包括：1)初始化：使用球形收缩算法在形状表面均匀采样初始球体；2)迭代优化：计算RPD、采样RPC、计算球体间作用力和能量、梯度投影约束移动、L-BFGS优化；3)投影步骤：将优化后的球体投影回中轴；4)重复优化直到结构收敛；5)计算中轴网格作为RPD对偶；6)后处理修剪无效连接。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出全面的RPD-based优化框架生成高质量结构感知的3D中轴网格；2)提出基于体积RPD的鲁棒球体分类策略提高分类准确性；3)引入中轴结构误差率（MSER）作为新的定量评估指标。相比之前工作，不同之处在于：优化了插入球体的位置而非固定它们；使用体积RPD而非表面RPD进行分类；通过结构感知梯度投影确保球体沿正确子结构移动；在优化过程中促进球体均匀分布减少过度拥挤；引入MSER指标评估中轴结构准确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MATStruct通过结合结构感知的变分优化和基于体积RPD的球体分类，首次实现了高质量中轴网格的生成，同时保持了中轴结构的完整性和几何保真度。'}


### 论文摘要

We propose a novel optimization framework for computing the medial axis transform that simultaneously preserves the medial structure and ensures high medial mesh quality. The medial structure, consisting of interconnected sheets, seams, and junctions, provides a natural volumetric decomposition of a 3D shape. Our method introduces a structure-aware, particle-based optimization pipeline guided by the restricted power diagram (RPD), which partitions the input volume into convex cells whose dual encodes the connectivity of the medial mesh. Structure-awareness is enforced through a spherical quadratic error metric (SQEM) projection that constrains the movement of medial spheres, while a Gaussian kernel energy encourages an even spatial distribution. Compared to feature-preserving methods such as MATFP and MATTopo, our approach produces cleaner and more accurate medial structures with significantly improved mesh quality. In contrast to voxel-based, point-cloud-based, and variational methods, our framework is the first to integrate structural awareness into the optimization process, yielding medial meshes with superior geometric fidelity, topological correctness, and explicit structural decomposition.

---

## 11. WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting

**论文链接:** [http://arxiv.org/abs/2510.10726v1](http://arxiv.org/abs/2510.10726v1)

**作者:** Yifan Liu, Zhiyuan Min, Zhenwei Wang, Junta Wu, Tengfei Wang, Yixuan Yuan, Yawei Luo, Chunchao Guo

**发布时间:** 2025-10-12

**备注:** Project page, code, and models will be publicly available soon

### GPT解析

### 总结

这篇论文提出了WorldMirror，一个用于多样化3D几何预测任务的一体化前馈模型，能够整合多种几何先验信息并同时生成多种3D表示，在各种任务中取得了最先进的性能。

### 背景

现有的3D几何预测方法要么仅限于图像输入，要么针对特定任务定制，缺乏灵活性和通用性。

### 目的

开发一个能够灵活整合多种几何先验信息，同时生成多种3D表示的统一框架，解决结构歧义问题，并实现高效的3D几何预测。

### 方法

提出WorldMirror模型，一种前馈架构，能够整合相机位姿、内参和深度图等几何先验信息，同时生成密集点云、多视角深度图、相机参数、表面法向量和3D高斯等多种3D表示。

### 主要发现

WorldMirror在各种基准测试中实现了最先进的性能，包括相机估计、点图估计、深度估计、表面法向量估计和新视角合成，同时保持了前向推理的效率。

### 结论

WorldMirror提供了一个优雅且统一的解决方案，能够利用可用的先验信息解决结构歧义，并在单次前向传播中生成几何一致的3D输出，为多样化3D几何预测任务提供了高效工具。

### 翻译

我们提出了WorldMirror，这是一个用于多样化3D几何预测任务的一体化前馈模型。与仅限于图像输入或针对特定任务定制的现有方法不同，我们的框架灵活整合了多种几何先验信息，包括相机位姿、内参和深度图，同时生成多种3D表示：密集点云、多视角深度图、相机参数、表面法向量和3D高斯。这种优雅且统一的架构利用可用的先验信息解决结构歧义，并在单次前向传播中生成几何一致的3D输出。WorldMirror在从相机、点图、深度和表面法向量估计到新视角合成的各种基准测试中实现了最先进的性能，同时保持了前向推理的效率。代码和模型即将公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有3D重建方法的两个局限性：一是输入限制，大多数方法只能处理原始图像，无法利用校准内参、相机位姿和深度测量等有用的先验信息；二是输出限制，方法通常只针对单一任务优化，很少在统一框架内整合多个任务。这些问题在现实中很重要，因为先验信息可以解决尺度模糊、确保多视图一致性，并在图像线索不足区域提供基础；统一框架能更高效处理各种3D重建任务，确保不同输出间的几何一致性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有方法在输入和输出空间都存在局限性，提出关键问题：能否通过有效利用多样化先验知识，在通用3D重建架构中解决这些挑战？他们设计了多模态先验提示机制和通用几何预测架构，借鉴了DUSt3R、VGGT等前馈3D重建模型的思想，以及3D高斯溅射在新视图合成中的应用，同时参考了传统优化方法利用已知相机参数提高重建质量的思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的前馈模型，能灵活整合多种几何先验信息，同时生成多种3D表示（点云、深度图、相机参数、表面法线、3D高斯），利用先验解决结构歧义，提供几何一致的3D输出。整体流程：1）多模态先验提示 - 将相机位姿、内参、深度图转换为令牌并整合；2）通用几何预测 - 使用DPT头和Transformer层预测各种几何属性；3）动态训练策略 - 随机采样不同先验组合，采用课程学习从简单到复杂优化训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1）多模态先验提示机制，首次系统探索多模态几何先验注入；2）通用几何预测架构，支持全方位3D重建任务；3）动态先验注入方案，适应任意先验组合；4）系统课程学习策略，优化训练效率。相比之前工作：输入上能灵活利用多种先验而非仅图像；输出上能同时处理多种任务而非单一优化；性能上在多个基准测试实现最先进结果；同时保持前向推理效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'WorldMirror是一个统一的前馈3D重建模型，通过灵活整合多种几何先验信息，在单次前向传播中同时生成多种高质量3D表示，实现了在各种3D重建任务上的最先进性能。'}


### 论文摘要

We present WorldMirror, an all-in-one, feed-forward model for versatile 3D geometric prediction tasks. Unlike existing methods constrained to image-only inputs or customized for a specific task, our framework flexibly integrates diverse geometric priors, including camera poses, intrinsics, and depth maps, while simultaneously generating multiple 3D representations: dense point clouds, multi-view depth maps, camera parameters, surface normals, and 3D Gaussians. This elegant and unified architecture leverages available prior information to resolve structural ambiguities and delivers geometrically consistent 3D outputs in a single forward pass. WorldMirror achieves state-of-the-art performance across diverse benchmarks from camera, point map, depth, and surface normal estimation to novel view synthesis, while maintaining the efficiency of feed-forward inference. Code and models will be publicly available soon.

---

## 12. dN/dx Reconstruction with Deep Learning for High-Granularity TPCs

**论文链接:** [http://arxiv.org/abs/2510.10628v1](http://arxiv.org/abs/2510.10628v1)

**作者:** Guang Zhao, Yue Chang, Jinxian Zhang, Linghui Wu, Huirong Qi, Xin She, Mingyi Dong, Shengsen Sun, Jianchun Wang, Yifang Wang, Chunxu Yu

**发布时间:** 2025-10-12

**备注:** 18 pages, 8 figures

### GPT解析

### 总结

本研究介绍了一种名为GraphPT的深度学习模型，用于粒子物理实验中的dN/dx重建，该模型在粒子识别性能上超过了传统方法，特别是在K/π分离方面有显著提升。

### 背景

粒子识别对未来的粒子物理实验（如圆形正负电子对撞机和未来圆形对撞机）至关重要。高granularity时间投影室能提供精确跟踪和dN/dx测量用于粒子识别，但准确重建仍是一大挑战。

### 目的

开发一种深度学习模型（GraphPT）来解决dN/dx重建的挑战，提高粒子识别性能，特别是改善K/π粒子的分离能力。

### 方法

将TPC数据表示为点云，采用基于图神经网络的U-Net架构作为网络主干，并融入针对点云处理优化的注意力机制用于节点聚合。

### 主要发现

所提出的GraphPT模型在粒子识别性能上超过了传统的截断均值方法。在动量区间5到20 GeV/c时，K/π分离能力提高了约10%到20%。

### 结论

GraphPT模型是一种有效的dN/dx重建方法，能够显著提高粒子物理实验中的粒子识别性能，特别是对于动量在5到20 GeV/c范围内的K/π粒子分离。

### 翻译

粒子识别对未来的粒子物理实验（如圆形正负电子对撞机和未来圆形对撞机）至关重要。高granularity时间投影室不仅提供精确的跟踪，还能实现dN/dx测量用于粒子识别。dN/dx方法估计初级电离电子的数量，显著提高了粒子识别性能。然而，准确的重建对于这种方法仍然是一个主要挑战。在本文中，我们介绍了一种深度学习模型——图点变换器，用于dN/dx重建。在我们的方法中，TPC数据被表示为点云。网络主干采用基于图神经网络的U-Net架构，并融入了针对点云处理优化的注意力机制用于节点聚合。所提出的GraphPT模型在粒子识别性能上超过了传统的截断均值方法。特别是在动量区间从5到20 GeV/c时，K/π分离能力提高了约10%到20%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决高粒度时间投影室（TPC）中dN/dx重建的挑战。dN/dx指单位长度轨迹上的初级电离电子数，用于粒子识别（PID）。这个问题很重要，因为准确的PID对下一代粒子物理实验（如CEPC和FCC）至关重要，特别是高动量下的强子识别。dN/dx方法相比传统dE/dx能显著提高PID性能，因为它直接测量初级电离簇数量，抑制了次级电离和能量波动的干扰，但高粒度TPC中的电子漂移和扩散使得准确重建变得困难。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到传统规则方法难以应对高粒度TPC中的dN/dx重建挑战，而深度学习可以提取数据中的复杂特征。他们将TPC数据表示为点云，设计了基于图神经网络的U-Net架构，并引入针对点云处理优化的注意力机制。该方法借鉴了多个现有工作：自注意力机制和Transformer架构、PointNet和PointNet++的点云处理方法、PointTransformer的自适应自注意力机制，以及之前在dN/dx重建中使用LSTM和DGCNN的研究。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将TPC数据表示为点云，使用基于图神经网络的U-Net架构处理这些点云，并通过注意力机制优化节点聚合。整体流程包括：1) 将TPC轨迹表示为点云，每个点包含电荷和定时信息；2) 构建k近邻图；3) 使用U-Net编码器-解码器结构处理图数据；4) 在每个层使用Transformer层进行节点间信息聚合；5) 使用端到端训练优化模型；6) 根据输出概率进行dN/dx重建，计算轨迹上被分类为正的命中数除以轨迹长度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出GraphPT模型专门用于高粒度TPC的dN/dx重建；2) 将TPC数据表示为点云并用图神经网络处理；3) 引入针对点云优化的注意力机制；4) 研究减法和点积两种注意力操作符，发现点积结合多头机制效果更好；5) 将之前的两步处理统一为单一模型。相比传统截断均值法，它不依赖人工规则，能处理复杂数据模式，K/π分离能力提高10-20%；相比之前基于神经网络的dN/dx工作，它专门针对高粒度TPC的三维点云数据，使用更先进的图神经网络和Transformer架构。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '该论文提出了一种基于图神经网络和Transformer的GraphPT深度学习方法，显著提高了高粒度TPC中dN/dx重建的粒子识别性能，相比传统方法实现了10-20%的K/π分离能力提升。'}


### 论文摘要

Particle identification (PID) is essential for future particle physics experiments such as the Circular Electron-Positron Collider and the Future Circular Collider. A high-granularity Time Projection Chamber (TPC) not only provides precise tracking but also enables dN/dx measurements for PID. The dN/dx method estimates the number of primary ionization electrons, offering significant improvements in PID performance. However, accurate reconstruction remains a major challenge for this approach. In this paper, we introduce a deep learning model, the Graph Point Transformer (GraphPT), for dN/dx reconstruction. In our approach, TPC data are represented as point clouds. The network backbone adopts a U-Net architecture built upon graph neural networks, incorporating an attention mechanism for node aggregation specifically optimized for point cloud processing. The proposed GraphPT model surpasses the traditional truncated mean method in PID performance. In particular, the $K/\pi$ separation power improves by approximately 10% to 20% in the momentum interval from 5 to 20 GeV/c.

---

## 13. SpikeGrasp: A Benchmark for 6-DoF Grasp Pose Detection from Stereo Spike Streams

**论文链接:** [http://arxiv.org/abs/2510.10602v1](http://arxiv.org/abs/2510.10602v1)

**作者:** Zhuoheng Gao, Jiyao Zhang, Zhiyong Xie, Hao Dong, Zhaofei Yu, Rongmei Chen, Guozhang Chen, Tiejun Huang

**发布时间:** 2025-10-12

### GPT解析

### 总结

本文介绍了一种名为SpikeGrasp的神经启发的机器人抓取框架，它模仿生物视觉运动通路，直接从立体尖峰摄像机的原始异步事件推断抓取姿态，无需重建点云，在杂乱和无纹理场景中表现优于传统方法，且数据效率高。

### 背景

大多数机器人抓取系统依赖于将传感器数据转换为显式的3D点云，这是生物智能中不存在的计算步骤，表明现有方法与生物智能有根本差异。

### 目的

探索一种 fundamentally different、神经启发的6-DoF抓取检测范式，模仿生物视觉运动通路，实现更高效、更自然的机器人抓取。

### 方法

引入SpikeGrasp框架，处理来自立体尖峰摄像机的原始异步事件，融合这些立体尖峰流，使用循环尖峰神经网络迭代改进抓取假设，构建大规模合成基准数据集进行验证。

### 主要发现

SpikeGrasp超越了基于点云的传统基线方法，特别在杂乱和无纹理场景中表现更好，展示了卓越的数据效率。

### 结论

通过建立这种端到端的神经启发方法的可行性，SpikeGrasp为未来能够实现自然界中流畅高效操作的系统铺平了道路，特别是对于动态物体。

### 翻译

大多数机器人抓取系统依赖于将传感器数据转换为显式的3D点云，这是生物智能中不存在的计算步骤。本文探索了一种根本不同的、神经启发的6-DoF抓取检测范式。我们引入了SpikeGrasp框架，它模仿生物视觉运动通路，处理来自立体尖峰摄像机的原始、异步事件（类似于视网膜），直接推断抓取姿态。我们的模型融合这些立体尖峰流，并使用循环尖峰神经网络（类似于高级视觉处理）来迭代改进抓取假设，而无需重建点云。为验证这一方法，我们构建了一个大规模合成基准数据集。实验表明，SpikeGrasp超越了传统的基于点云的基线方法，特别是在杂乱和无纹理场景中，并表现出卓越的数据效率。通过建立这种端到端的神经启发方法的可行性，SpikeGrasp为未来能够实现自然界中流畅高效操作的系统铺平了道路，特别是对于动态物体。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从立体尖峰相机的原始异步事件流中直接检测6自由度抓取姿态，而不需要将传感器数据转换为3D点云。这个问题很重要，因为当前大多数机器人抓取系统依赖点云重建，这是一个计算密集且易受噪声影响的步骤，而生物系统并不依赖显式点云表示来抓取物体。直接从原始事件流推断抓取姿态可以减少计算负担，提高系统在杂乱场景中的鲁棒性，更接近生物系统的效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到生物视觉运动系统的启发，将系统分为生物启发的组件：人工视网膜（立体尖峰相机）、视觉通路（左右尖峰流处理和融合）、整合皮层（循环尖峰神经网络）和运动系统。他们借鉴了尖峰相机在图像重建、目标检测等任务中的应用，以及基于点云的抓取检测方法的评估框架和表示方法，但避免了显式的点云处理步骤，直接从原始事件流推断抓取姿态。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是模仿生物视觉运动系统，直接从原始异步事件流中推断抓取姿态，而不需要显式的3D点云重建。整体流程分为三部分：1）视觉通路网络：从左右尖峰流提取特征，计算相关性，使用循环尖峰神经网络迭代更新；2）可抓取网络：解码隐藏状态生成物体存在概率和可抓取性概率图；3）抓取检测网络：从隐藏状态和可抓取位置预测完整的6-DoF抓取配置，选择最高分数的抓取作为最终估计。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）生物启发的端到端框架，直接从原始立体尖峰流检测抓取姿态；2）构建了第一个用于6-DoF抓取姿态检测的大规模合成尖峰流数据集；3）使用循环尖峰神经网络处理时空信息并迭代完善抓取假设；4）表现出强大的数据效率。相比之前工作，SpikeGrasp避免了点云重建的中间步骤，能处理完整6-DoF抓取姿态，在杂乱场景中表现更好，计算效率更高。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SpikeGrasp通过引入一种受生物启发的端到端框架，首次实现了从原始立体尖峰流直接检测6-DoF抓取姿态，避免了点云重建的中间步骤，并在杂乱场景中表现出更高的准确性和数据效率。'}


### 论文摘要

Most robotic grasping systems rely on converting sensor data into explicit 3D point clouds, which is a computational step not found in biological intelligence. This paper explores a fundamentally different, neuro-inspired paradigm for 6-DoF grasp detection. We introduce SpikeGrasp, a framework that mimics the biological visuomotor pathway, processing raw, asynchronous events from stereo spike cameras, similarly to retinas, to directly infer grasp poses. Our model fuses these stereo spike streams and uses a recurrent spiking neural network, analogous to high-level visual processing, to iteratively refine grasp hypotheses without ever reconstructing a point cloud. To validate this approach, we built a large-scale synthetic benchmark dataset. Experiments show that SpikeGrasp surpasses traditional point-cloud-based baselines, especially in cluttered and textureless scenes, and demonstrates remarkable data efficiency. By establishing the viability of this end-to-end, neuro-inspired approach, SpikeGrasp paves the way for future systems capable of the fluid and efficient manipulation seen in nature, particularly for dynamic objects.

---

## 14. DAGLFNet:Deep Attention-Guided Global-Local Feature Fusion for Pseudo-Image Point Cloud Segmentation

**论文链接:** [http://arxiv.org/abs/2510.10471v1](http://arxiv.org/abs/2510.10471v1)

**作者:** Chuang Chen, Wenyi Ge

**发布时间:** 2025-10-12

### GPT解析

### 总结

本文提出了一种名为DAGLFNet的伪图像语义分割框架，用于高效处理非结构化点云并提取结构化语义信息，在保持实时性能的同时实现了高精度。

### 背景

环境感知系统在高精度测绘和自主导航中扮演关键角色，LiDAR作为核心传感器提供准确的3D点云数据。然而，如何高效处理非结构化点云并提取结构化语义信息仍是一个重大挑战。现有的基于伪图像的表示方法往往忽略了点云的结构和语义细节，导致特征融合和判别性有限。

### 目的

设计一个伪图像语义分割框架，提取判别性特征，平衡处理效率和性能，同时保持点云的结构和语义细节。

### 方法

提出DAGLFNet框架，包含三个主要模块：1) 全局-局部特征融合编码模块，增强局部特征相关性并捕获全局上下文信息；2) 多分支特征提取网络，捕获更多邻域信息并增强轮廓特征的判别性；3) 基于深度特征引导的注意力机制的特征融合，提高跨通道特征融合的精度。

### 主要发现

实验评估显示，DAGLFNet在SemanticKITTI和nuScenes验证集上分别达到了69.83%和78.65%的性能，平衡了高性能与实时能力，展示了基于LiDAR实时应用的巨大潜力。

### 结论

DAGLFNet方法通过创新的特征提取和融合机制，有效解决了点云处理中的效率和性能平衡问题，为基于LiDAR的实时应用提供了有效解决方案。

### 翻译

环境感知系统在高精度测绘和自主导航中起着关键作用，LiDAR作为提供准确3D点云数据的核心传感器。如何高效处理非结构化点云同时提取结构化语义信息仍是一个重大挑战，近年来，出现了许多基于伪图像的表示方法以在效率和性能之间取得平衡。然而，它们通常忽略了点云的结构和语义细节，导致特征融合和判别性有限。在这项工作中，我们提出了DAGLFNet，一种基于伪图像的语义分割框架，旨在提取判别性特征。首先，使用全局-局部特征融合编码模块来增强集合内局部特征之间的相关性并捕获全局上下文信息。其次，采用多分支特征提取网络来捕获更多邻域信息并增强轮廓特征的判别性。最后，引入基于深度特征引导的注意力机制进行特征融合，以提高跨通道特征融合的精度。实验评估表明，DAGLFNet在SemanticKITTI和nuScenes的验证集上分别达到了69.83%和78.65%。该方法平衡了高性能与实时能力，展示了基于LiDAR实时应用的巨大潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何高效处理非结构化的LiDAR点云数据并提取结构化语义信息的问题，特别是在基于伪图像的点云分割方法中，如何解决结构扭曲、边界模糊和语义模糊等挑战。这个问题在现实世界中非常重要，因为环境感知系统是高精度地图和自主导航的核心，LiDAR作为关键传感器提供的3D点云数据需要被准确理解和解析，这对自动驾驶、机器人等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了当前点云分割的三种主要方法（点方法、体素方法和混合策略）及其局限性，然后关注到基于范围图像的方法在计算效率和性能之间的平衡。作者发现现有伪图像方法忽略了点云的结构和语义细节，导致特征融合和判别能力有限。针对这些问题，作者设计了DAGLFNet框架，借鉴了现有点云处理的基本方法、范围图像表示方式以及深度学习中的注意力机制和多分支网络等设计理念，但进行了创新性改进以解决特定问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过全局-局部特征融合增强局部特征间的相关性并捕获全局上下文信息，使用多分支特征提取网络捕获更多邻域信息并增强边界特征的判别能力，以及引入深度特征引导的注意力机制提高跨通道特征融合的精度。整体实现流程包括：1)特征编码器将点云分组并提取点级和组级特征；2)图像特征提取器使用多分支架构捕获不同感受野的特征；3)特征更新模块通过深度引导的注意力机制融合点级和组级特征；4)融合头模块聚合多阶段特征并生成最终语义预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出了DAGLFNet网络架构，将点云几何特征与二维伪图像表示紧密集成；2)提出了包含三个关键模块的全面特征增强策略：GL-FFE模块捕获长程依赖并稳定局部几何表示，MB-FE网络扩大感受野并加强边界特征表达，FFDFA机制利用距离感知加权提高跨通道特征集成精度。相比之前的工作，DAGLFNet不仅处理投影到图像上的点，还解决了多点映射冲突问题，考虑了遮挡点，保留了完整三维结构，并强调了对子集内特征关系的一致建模同时系统考虑了空间距离的影响。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DAGLFNet通过全局-局部特征融合、多分支特征提取和深度特征引导的注意力机制，有效解决了伪图像点云分割中特征表达不足的问题，在保持实时性能的同时显著提高了语义分割精度。'}


### 论文摘要

Environmental perception systems play a critical role in high-precision mapping and autonomous navigation, with LiDAR serving as a core sensor that provides accurate 3D point cloud data. How to efficiently process unstructured point clouds while extracting structured semantic information remains a significant challenge, and in recent years, numerous pseudo-image-based representation methods have emerged to achieve a balance between efficiency and performance. However, they often overlook the structural and semantic details of point clouds, resulting in limited feature fusion and discriminability. In this work, we propose DAGLFNet, a pseudo-image-based semantic segmentation framework designed to extract discriminative features. First, the Global-Local Feature Fusion Encoding module is used to enhance the correlation among local features within a set and capture global contextual information. Second, the Multi-Branch Feature Extraction network is employed to capture more neighborhood information and enhance the discriminability of contour features. Finally, a Feature Fusion via Deep Feature-guided Attention mechanism is introduced to improve the precision of cross-channel feature fusion. Experimental evaluations show that DAGLFNet achieves 69.83\% and 78.65\% on the validation sets of SemanticKITTI and nuScenes, respectively. The method balances high performance with real-time capability, demonstrating great potential for LiDAR-based real-time applications.

---

## 15. Multi-View Graph Learning with Graph-Tuple

**论文链接:** [http://arxiv.org/abs/2510.10341v1](http://arxiv.org/abs/2510.10341v1)

**作者:** Shiyu Chen, Ningyuan, Huang, Soledad Villar

**发布时间:** 2025-10-11

**备注:** Submitted to TAG workshop

### GPT解析

### 总结

提出了一种多视图图元组框架，解决了传统图神经网络在密集图上的效率问题

### 背景

图神经网络通常随图边数扩展，适合稀疏图但在密集图(如点云或分子相互作用)上效率较低

### 目的

克服传统稀疏化方法强制选择单一交互尺度并丢弃其他尺度信息的限制

### 方法

将图划分为不相交的子图，捕获主要局部相互作用和远程连接；通过受非交换算子理论启发的异构消息传递架构学习多视图表示

### 主要发现

多视图图元组模型在分子属性预测和宇宙学参数推断两个应用中均优于单图基线模型

### 结论

多视图方法具有强大功能和通用性，能有效处理密集图数据

### 翻译

图神经网络(GNNs)通常随图边数扩展，使其适合稀疏图但在密集图(如点云或分子相互作用)上效率较低。常见补救措施是通过相似度阈值或距离修剪稀疏化图，但这强制选择单一交互尺度并丢弃其他尺度的重要信息。为克服这一限制，我们引入了多视图图元组框架。我们的图元组框架将图划分为不相交的子图，捕获主要局部相互作用和较弱的远程连接。然后，我们通过受非交换算子理论启发的异构消息传递架构从图元组学习多视图表示，理论上证明这比单图消息传递模型更具表现力，并能保证更低的oracle风险。我们在两个科学领域实现了我们的框架：特征稀缺的库仑矩阵的分子属性预测和几何点云的宇宙学参数推断。在这两种应用中，我们的多视图图元组模型都表现出比单图基线模型更好的性能，突显了我们多视图方法的强大功能和通用性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决图神经网络（GNNs）在处理密集图（如点云或分子相互作用）时的效率问题。传统方法需要通过稀疏化图来提高计算效率，但这会强制选择单一交互尺度并丢失其他尺度的重要信息。这个问题在科学应用中尤为重要，因为分子、宇宙学等领域的密集数据包含多种尺度的交互信息，单一尺度方法无法同时捕捉局部强相互作用和全局弱相互作用，导致信息损失和性能下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了GNNs在密集图上的计算瓶颈和传统稀疏化方法的局限性。他们注意到现有多视图方法和异构图学习方法不适用于具有连续边特征的同构图。受异构图神经网络（如R-GCN、HAN）的启发，作者扩展了这些方法到同构图；借鉴了多视图表示学习的思想，但基于物理交互强度构建视图；受到多尺度GNNs的启发，但使用相同节点集的不同边集。作者还从数学上证明了新框架的理论优势，设计了异构消息传递架构，同时进行视图内和视图间的消息传递，以捕获不同尺度的交互信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将单一密集图分解为多个视图（子图），每个视图捕捉不同尺度的交互，通过异构消息传递架构同时学习这些视图的表示。实现流程包括：1) 构建图元组，将原始图分解为不相交的子图（如强连接图和弱连接图）；2) 进行异构消息传递，包括视图内消息传递（每个子图内计算节点表示）和视图间消息传递（跨子图计算表示）；3) 融合多视图表示，使用可学习的标量权重组合所有视图信息；4) 具体实现分为GINE-Gt（用于一般图）和EGNN-Gt（用于几何数据）两种架构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出多视图图元组表示，将单一密集图分解为多个子图，保留多尺度信息；2) 设计异构消息传递架构，同时进行视图内和视图间的消息传递，考虑操作顺序敏感性；3) 提供理论保证，证明新框架比单图模型更具表现力且风险更低；4) 在分子性质预测和宇宙学参数推断等科学应用中验证了框架的有效性。相比之前工作，不同之处在于：不丢弃任何尺度信息，避免任意选择单一交互尺度；扩展异构图方法到同构图；基于物理交互强度而非自监督构建视图；在相同节点集上定义多个图，避免跨级别对齐的复杂性；不依赖低秩假设，能更好保留原始数据信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一个多视图图元组框架，通过分解密集图为多个子图并使用异构消息传递架构同时学习不同尺度的交互，有效解决了图神经网络在密集图上的计算效率与信息保留之间的权衡问题，并在科学应用中展示了优越性能。'}


### 论文摘要

Graph Neural Networks (GNNs) typically scale with the number of graph edges, making them well suited for sparse graphs but less efficient on dense graphs, such as point clouds or molecular interactions. A common remedy is to sparsify the graph via similarity thresholding or distance pruning, but this forces an arbitrary choice of a single interaction scale and discards crucial information from other scales. To overcome this limitation, we introduce a multi-view graph-tuple framework. Instead of a single graph, our graph-tuple framework partitions the graph into disjoint subgraphs, capturing primary local interactions and weaker, long-range connections. We then learn multi-view representations from the graph-tuple via a heterogeneous message-passing architecture inspired by the theory of non-commuting operators, which we formally prove is strictly more expressive and guarantees a lower oracle risk compared to single-graph message-passing models. We instantiate our framework on two scientific domains: molecular property prediction from feature-scarce Coulomb matrices and cosmological parameter inference from geometric point clouds. On both applications, our multi-view graph-tuple models demonstrate better performance than single-graph baselines, highlighting the power and versatility of our multi-view approach.

---

## 16. Gesplat: Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2510.10097v1](http://arxiv.org/abs/2510.10097v1)

**作者:** Jiahui Lu, Haihong Xiao, Xueyan Zhao, Wenxiong Kang

**发布时间:** 2025-10-11

### GPT解析

### 总结

本文提出了Gesplat，一个基于3DGS的框架，能够从未配准的稀疏图像中进行鲁棒的新型视图合成和几何一致的3D重建

### 背景

NeRF和3DGS已推动3D重建和新型视图合成发展，但严重依赖准确的相机姿态和密集视角覆盖，限制了在稀疏视图场景中的应用

### 目的

克服现有方法在稀疏视图场景中的局限性，实现无需准确相机姿态的鲁棒3D重建和视图合成

### 方法

利用VGGT基础模型替代COLMAP进行初始姿态估计；采用混合高斯表示结合视图间匹配一致性进行双位置-形状优化；引入图引导的属性细化模块增强场景细节；使用基于流的深度正则化提高深度估计准确性

### 主要发现

通过定量和定性实验证明，相比其他无姿态方法，Ges在前向-facing和大规模复杂数据集上实现了更鲁棒的性能

### 结论

Gesplat框架能够在稀疏视图条件下实现更鲁棒的3D重建和视图合成，拓展了NeRF和3DGS的应用范围

### 翻译

神经辐射场和3D高斯飞溅已经推动了3D重建和新型视图合成的发展，但仍然严重依赖准确的相机姿态和密集的视角覆盖。这些要求限制了它们在稀疏视图场景中的应用，在这些场景中姿态估计变得不可靠且监督不足。为了克服这些挑战，我们引入了Gesplat，一个基于3DGS的框架，能够从未配准的稀疏图像中进行鲁棒的新型视图合成和几何一致的重建。与之前依赖COLMAP进行稀疏点云初始化的工作不同，我们利用VGGT基础模型获得更可靠的初始姿态和密集点云。我们的方法整合了几个关键创新：1) 通过视图间匹配一致性增强的双位置-形状优化的混合高斯表示；2) 增强场景细节的图引导属性细化模块；3) 基于流的深度正则化，提高深度估计准确性以实现更有效的监督。全面的定量和定性实验表明，与其他无姿态方法相比，我们的方法在前向-facing和大规模复杂数据集上实现了更鲁棒的性能

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从稀疏视角且没有相机位姿信息的图像中进行鲁棒的3D重建和新视角合成的问题。这个问题在现实中很重要，因为在实际场景中获取密集、覆盖良好的图像集通常不切实际且成本高昂，而稀疏视角的3D重建在自主导航、VR/AR和机器人技术等应用中至关重要。有限视角导致训练期间监督不足，会造成伪影和有缺陷的重建。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有NeRF和3DGS方法在稀疏视角和无位姿设置下的局限性，特别是传统COLMAP方法在稀疏视角下的不可靠性。他们引入VGGT基础模型替代COLMAP进行初始点云和位姿估计，并设计了混合高斯表示结合普通和基于射线的高斯，通过多视图匹配一致性进行优化。作者借鉴了VGGT进行初始场景重建，受[28]启发采用混合高斯表示，使用图神经网络优化属性，并参考[31]的方法利用光流进行深度估计。整个设计思路是在保留3DGS高效性的同时，解决其在稀疏视角下的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过引入适当的几何先验约束场景结构，同时采用优化和正则化技术细化场景细节。整体流程包括：1)使用VGGT生成初始密集点云和相机位姿；2)采用混合高斯表示(普通高斯和基于射线的高斯)；3)利用多视图匹配一致性进行位置和形状双重优化；4)应用图神经网络优化高斯属性；5)使用基于流的深度正则化提高渲染质量；6)联合优化高斯参数和相机位姿。测试阶段使用训练好的高斯模型细化测试相机位姿。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)混合高斯表示，结合普通和基于射线的高斯，通过绑定高斯到匹配射线增强多视图一致性；2)图引导属性优化模块，使用图神经网络优化高斯属性；3)基于流的深度正则化，在极线几何框架内通过光流估计可靠深度图；4)使用VGGT替代COLMAP进行初始化，在稀疏视角下更可靠。相比之前工作，Gesplat不依赖密集视角覆盖和已知准确位姿，计算成本更低，能处理更稀疏的输入，且在几何一致性和细节质量上表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Gesplat通过引入混合高斯表示、图引导属性优化和基于流的深度正则化，实现了从稀疏视角无位姿图像中进行鲁棒3D重建和新视角合成，显著提高了场景几何一致性和细节质量。'}


### 论文摘要

Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have advanced 3D reconstruction and novel view synthesis, but remain heavily dependent on accurate camera poses and dense viewpoint coverage. These requirements limit their applicability in sparse-view settings, where pose estimation becomes unreliable and supervision is insufficient. To overcome these challenges, we introduce Gesplat, a 3DGS-based framework that enables robust novel view synthesis and geometrically consistent reconstruction from unposed sparse images. Unlike prior works that rely on COLMAP for sparse point cloud initialization, we leverage the VGGT foundation model to obtain more reliable initial poses and dense point clouds. Our approach integrates several key innovations: 1) a hybrid Gaussian representation with dual position-shape optimization enhanced by inter-view matching consistency; 2) a graph-guided attribute refinement module to enhance scene details; and 3) flow-based depth regularization that improves depth estimation accuracy for more effective supervision. Comprehensive quantitative and qualitative experiments demonstrate that our approach achieves more robust performance on both forward-facing and large-scale complex datasets compared to other pose-free methods.

---

## 17. Minkowski-MambaNet: A Point Cloud Framework with Selective State Space Models for Forest Biomass Quantification

**论文链接:** [http://arxiv.org/abs/2510.09367v1](http://arxiv.org/abs/2510.09367v1)

**作者:** Jinxiang Tu, Dayong Ren, Fei Shi, Zhenhong Jia, Yahong Ren, Jiwei Qin, Fang He

**发布时间:** 2025-10-10

### GPT解析

### 总结

Minkowski-MambaNet是一种创新的深度学习框架，能够直接从原始LiDAR数据估算森林木材体积和地上生物量，显著提高了森林生物量量化的准确性。

### 背景

准确的森林生物量量化对碳循环监测至关重要。虽然机载LiDAR在捕捉森林三维结构方面表现出色，但由于难以建模区分树木所需的长程依赖关系，直接从点云估算木材体积和地上生物量(AGB)具有挑战性。

### 目的

开发一种新的深度学习框架，能够直接从原始LiDAR数据估算体积和AGB，提高森林生物量量化的准确性和鲁棒性。

### 方法

提出Minkowski-MambaNet，将Mamba模型的选择性状态空间模型(SSM)集成到Minkowski网络中，有效编码全局上下文和长程依赖关系以提高树木区分能力，并融入跳跃连接以增强特征并加速收敛。

### 主要发现

在丹麦国家森林清单LiDAR数据上评估，Minkowski-MambaNet显著优于最先进的方法，提供了更准确和稳健的估计。该方法不需要数字地形模型(DTM)，并且对边界伪影具有鲁棒性。

### 结论

Minkowski-MambaNet为大规模森林生物量分析提供了强大的工具，推动了基于LiDAR的森林清查的发展。

### 翻译

准确的森林生物量量化对碳循环监测至关重要。虽然机载LiDAR在捕捉森林三维结构方面表现出色，但由于难以建模区分树木所需的长程依赖关系，直接从点云估算木材体积和地上生物量(AGB)具有挑战性。我们提出了Minkowski-MambaNet，一种创新的深度学习框架，可直接从原始LiDAR估算体积和AGB。其关键创新是将Mamba模型的选择性状态空间模型(SSM)集成到Minkowski网络中，从而有效编码全局上下文和长程依赖关系，以提高树木区分能力。融入了跳跃连接以增强特征并加速收敛。在丹麦国家森林清单LiDAR数据上评估，Minkowski-MambaNet显著优于最先进的方法，提供了更准确和稳健的估计。重要的是，它不需要数字地形模型(DTM)，并且对边界伪影具有鲁棒性。这项工作为大规模森林生物量分析提供了强大的工具，推动了基于LiDAR的森林清查的发展。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何直接从原始LiDAR点云数据中准确估算森林生物量（包括木材体积和地上生物量AGB）的问题。这个问题非常重要，因为森林是陆地生态系统中最大的碳库，约占全球陆地碳储量的40%，准确量化森林生物量对于全球碳循环监测、气候变化研究和森林管理至关重要。传统方法要么成本高昂、难以大范围应用，要么无法充分捕捉森林的三维结构信息。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统CNN处理点云效率低，基于局部卷积的方法难以捕获长距离依赖关系，而自注意力机制计算复杂度高。作者借鉴了Minkowski引擎（高效处理稀疏数据）和Mamba状态空间模型（高效序列建模）的工作，以MSENet50为基础骨干网络，设计了两个关键模块：Mamba-SEBottleneck（解决长距离依赖问题）和特征融合修改层（解决多尺度特征利用问题），形成了一个端到端的直接回归框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合Mamba的选择性状态空间模型与Minkowski稀疏卷积，有效捕获点云中的长距离依赖关系和多层次特征，直接从原始LiDAR点云估算森林生物量。实现流程包括：1)数据预处理（保留异质性、排除不一致样本、设置高度阈值）；2)基于MSENet50构建网络架构；3)Mamba-SEBottleneck模块将点特征转换为序列并处理，生成动态注意力权重；4)特征融合修改层通过跳跃连接融合中间层与深层特征；5)训练与评估使用丹麦国家森林调查数据，通过RMSE、R2等指标比较性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)Mamba-SEBottleneck模块，将Mamba状态空间模型与Minkowski稀疏卷积结合，以线性复杂度捕获长距离依赖；2)特征融合修改层，通过跳跃连接保留多尺度特征；3)端到端直接回归方法，避免中间步骤。相比之前工作，它比传统方法不依赖手工特征，比基于局部卷积的方法更好地捕获全局结构，比自注意力机制计算效率更高，且无需数字地形模型预处理，对边界噪声更鲁棒。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Minkowski-MambaNet通过创新性地结合Mamba状态空间模型与Minkowski稀疏卷积，首次实现了从原始LiDAR点云中高效准确地直接估算森林生物量，为森林碳汇监测和资源管理提供了强大的新工具。'}


### 论文摘要

Accurate forest biomass quantification is vital for carbon cycle monitoring. While airborne LiDAR excels at capturing 3D forest structure, directly estimating woody volume and Aboveground Biomass (AGB) from point clouds is challenging due to difficulties in modeling long-range dependencies needed to distinguish trees.We propose Minkowski-MambaNet, a novel deep learning framework that directly estimates volume and AGB from raw LiDAR. Its key innovation is integrating the Mamba model's Selective State Space Model (SSM) into a Minkowski network, enabling effective encoding of global context and long-range dependencies for improved tree differentiation. Skip connections are incorporated to enhance features and accelerate convergence.Evaluated on Danish National Forest Inventory LiDAR data, Minkowski-MambaNet significantly outperforms state-of-the-art methods, providing more accurate and robust estimates. Crucially, it requires no Digital Terrain Model (DTM) and is robust to boundary artifacts. This work offers a powerful tool for large-scale forest biomass analysis, advancing LiDAR-based forest inventories.

---

## 18. Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes

**论文链接:** [http://arxiv.org/abs/2510.09364v1](http://arxiv.org/abs/2510.09364v1)

**作者:** Yikang Zhang, Rui Fan

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文提出了VAD-GS，一种针对具有挑战性城市场景的3DGS框架，通过体素可视性推理、多样性感知视图选择和基于补丁匹配的多视立体重建来解决3DGS在无界动态环境中初始化点云不完整导致的几何失真和伪影问题。

### 背景

3D Gaussian splatting (3DGS)在合成高质量新视角方面表现出色，但其效果严重依赖于初始化点云的质量。在无界、动态的城市环境中，实现场景结构的均匀和完整点覆盖需要重叠的观察视锥，这一假设常常不成立，导致训练的高斯模型出现失真和伪影。

### 目的

解决3DGS在无界动态城市环境中因初始化点云不完整导致的几何失真和伪影问题，提高静态和动态对象的几何重建质量。

### 方法

提出VAD-GS框架，包含三个关键组件：1) 基于体素的可视性推理识别不可靠的几何结构；2) 通过多样性感知的视图选择选择信息量大的支持视图；3) 通过基于补丁匹配的多视立体重建恢复缺失结构。这种设计能够在缺乏初始点的区域中，由可靠的几何先验引导生成新的高斯基元。

### 主要发现

在Waymo和nuScenes数据集上的实验表明，VAD-GS优于最先进的3DGS方法，并显著提高了静态和动态对象重建几何的质量。

### 结论

VAD-GS框架能够有效解决3DGS在具有挑战性的城市环境中的几何恢复问题，即使在没有初始点的区域也能生成高质量的几何重建。

### 翻译

3D高斯溅射(3DGS)在合成高质量新视角方面展示了令人印象深刻的性能。尽管如此，其有效性严重依赖于初始化点云的质量。具体来说，在底层场景结构上实现均匀和完整的点覆盖需要重叠的观察视锥，这一假设在无界、动态的城市环境中常常被违反。使用部分初始化的点云训练高斯模型通常会导致失真和伪影，因为相机射线可能无法与有效表面相交，导致与被遮挡或不可见几何相关联的高斯基元出现错误的梯度传播。此外，现有的密集化策略只是简单地从现有高斯基元克隆和分割，无法重建缺失的结构。为了解决这些局限性，我们提出了VAD-GS，一种针对具有挑战性的城市场景几何恢复的3DGS框架。我们的方法通过基于体素的可视性推理识别不可靠的几何结构，通过多样性感知的视图选择选择信息量大的支持视图，并通过基于补丁匹配的多视立体重建恢复缺失结构。这种设计使得即使在缺乏初始点的区域中，也能够由可靠的几何先验引导生成新的高斯基元。在Waymo和nuScenes数据集上的大量实验表明，VAD-GS优于最先进的3DGS方法，并显著提高了静态和动态对象重建几何的质量。源代码将在发表后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D高斯泼溅(3DGS)在动态城市场景中重建完整几何结构的问题。具体来说，3DGS方法的有效性依赖于初始点云质量，但在无边界的动态城市环境中，难以实现均匀且完整的点覆盖，导致训练出的模型出现失真和伪影。这个问题在现实中对自动驾驶系统至关重要，因为它们需要高质量的场景重建来进行模拟和验证，而传统模拟器缺乏场景多样性和可扩展性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了3DGS在动态城市场景中的局限性，特别是初始点云不足导致的几何失真问题。他们注意到现有密集化方法只是简单克隆和分割现有高斯原语，无法处理未初始化区域。作者借鉴了多视图立体视觉(MVS)技术、体素化技术和z-buffer可见性推理，并结合实例分割方法来处理动态对象。通过整合这些技术，他们设计了一个主动评估结构完整性并选择性重建不完整区域的框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过可见性感知的密集化策略主动评估并重建缺失的几何结构，即使在初始点云不完整的区域也能生成新的高斯原语。整体流程包括：1)对初始点云进行体素化以获得均匀空间密度；2)进行基于体素的可见性推理，识别不可靠几何结构；3)使用多样性感知的视图选择策略选择信息量大的支持视图；4)通过基于块匹配的MVS算法重建深度和法线信息；5)使用这些几何先验指导高斯密集化和优化；6)结合颜色、法线和深度损失进行模型训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)针对动态城市场景的高斯泼溅框架，主动使用多摄像头、跨帧观测完成缺失几何；2)基于体素表面可见性推理方法，识别不可靠的静态和动态对象几何；3)多样性感知的采样策略，提高MVS重建质量；4)将MVS重建扩展到动态多摄像头驾驶场景。相比之前工作，VAD-GS不局限于现有高斯原语区域，能处理未初始化区域；能处理动态对象而非仅限于静态场景；利用多摄像头和跨帧观测而非仅单摄像头相邻帧；通过可见性推理区分可见和被遮挡几何，避免错误更新被遮挡的高斯原语。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VAD-GS通过可见性感知的密集化策略和多视图立体视觉重建，显著提高了3D高斯泼溅在动态城市场景中的几何质量和渲染保真度，解决了初始点云不完整导致的几何失真问题。'}


### 论文摘要

3D Gaussian splatting (3DGS) has demonstrated impressive performance in synthesizing high-fidelity novel views. Nonetheless, its effectiveness critically depends on the quality of the initialized point cloud. Specifically, achieving uniform and complete point coverage over the underlying scene structure requires overlapping observation frustums, an assumption that is often violated in unbounded, dynamic urban environments. Training Gaussian models with partially initialized point clouds often leads to distortions and artifacts, as camera rays may fail to intersect valid surfaces, resulting in incorrect gradient propagation to Gaussian primitives associated with occluded or invisible geometry. Additionally, existing densification strategies simply clone and split Gaussian primitives from existing ones, incapable of reconstructing missing structures. To address these limitations, we propose VAD-GS, a 3DGS framework tailored for geometry recovery in challenging urban scenes. Our method identifies unreliable geometry structures via voxel-based visibility reasoning, selects informative supporting views through diversity-aware view selection, and recovers missing structures via patch matching-based multi-view stereo reconstruction. This design enables the generation of new Gaussian primitives guided by reliable geometric priors, even in regions lacking initial points. Extensive experiments on the Waymo and nuScenes datasets demonstrate that VAD-GS outperforms state-of-the-art 3DGS approaches and significantly improves the quality of reconstructed geometry for both static and dynamic objects. Source code will be released upon publication.

---

## 19. Obstacle Avoidance using Dynamic Movement Primitives and Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.09254v1](http://arxiv.org/abs/2510.09254v1)

**作者:** Dominik Urbaniak, Alejandro Agostini, Pol Ramon, Jan Rosell, Raúl Suárez, Michael Suppa

**发布时间:** 2025-10-10

**备注:** 8 pages, 7 figures

### GPT解析

### 总结

该研究提出了一种从单个人工演示快速生成平滑、次优、无碰撞的三维笛卡尔轨迹的方法，通过结合动态运动基元和强化学习技术，实现了对多种障碍配置的高效轨迹规划。

### 背景

基于学习的运动规划虽能快速生成次优轨迹，但通常需要大型训练数据集或昂贵的人类演示收集，限制了其在实际应用中的可行性。

### 目的

开发一种替代方法，仅从单个人工演示就能快速生成高质量的三维笛卡尔轨迹，减少对大量训练数据的依赖。

### 方法

将演示编码为动态运动基元(DMP)，使用基于策略的强化学习迭代重塑DMP以创建多样化轨迹数据集，然后训练神经网络输入障碍物参数并输出相应的DMP参数。

### 主要发现

在仿真和真实机器人实验中，该方法在计算时间、执行时间和轨迹长度方面均优于RRT-Connect基线算法，并能支持针对不同障碍几何形状和末端执行器尺寸的多模态轨迹生成。

### 结论

该方法有效解决了传统基于学习的运动规划对大量训练数据的依赖问题，为机器人轨迹规划提供了一种高效、实用的解决方案。

### 翻译

基于学习的运动规划可以快速生成次优轨迹。然而，它通常需要大型训练数据集或昂贵的人类演示收集工作。本文提出了一种替代方法，从单个人工演示快速生成平滑、次优、无碰撞的三维笛卡尔轨迹。该演示被编码为动态运动基元(DMP)，并使用基于策略的强化学习进行迭代重塑，为不同的障碍物配置创建多样化的轨迹数据集。该数据集用于训练神经网络，输入是从点云自动导出的描述障碍物尺寸和位置的任务参数，输出生成轨迹的DMP参数。该方法在仿真和真实机器人实验中得到验证，在计算和执行时间以及轨迹长度方面优于RRT-Connect基线，同时支持针对不同障碍几何形状和末端执行器尺寸的多模态轨迹生成。视频和实现代码可在https://github.com/DominikUrbaniak/obst-avoid-dmp-pi2获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人快速生成平滑、接近最优的无碰撞轨迹问题，特别是在避障场景中的应用。这个问题在现实中很重要，因为机器人需要在复杂环境中自主导航和操作，快速生成轨迹对于实时应用至关重要，而平滑的轨迹可以减少机械磨损并提高执行效率。此外，减少对大量演示数据的依赖可以降低部署成本，同时能够处理多种障碍物配置和末端执行器尺寸变化，增强了机器人的适应性和实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到基于学习的方法在运动规划中的优势，但注意到它们通常需要大量训练数据。因此，他们想到使用动态运动基元（DMP）来编码和泛化演示轨迹，利用其良好的泛化能力。然后采用策略改进与路径积分（PI²）强化学习算法来迭代调整演示轨迹，生成多样化的避障轨迹数据集。最后将生成的轨迹数据集映射到一个神经网络中，实现快速在线轨迹生成。该方法借鉴了现有工作，包括使用DMP作为运动基元、PI²作为强化学习算法、点云处理技术来检测障碍物，以及参考现有避障方法作为比较基准。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用单个人工演示作为初始轨迹，通过强化学习迭代调整生成多样化的避障轨迹，训练神经网络将任务参数映射到DMP参数，并从点云中自动提取任务参数以适应新场景。整体实现流程分为两个阶段：1）离线训练阶段：将演示编码为DMP参数，使用PI²算法根据不同成本函数调整参数生成多样化轨迹数据集，训练神经网络将任务参数映射到DMP参数；2）在线执行阶段：从点云中提取任务参数，使用神经网络推断适合当前场景的DMP参数，生成并执行无碰撞轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）数据效率高，只需一个人工演示；2）能自动从点云提取任务参数；3）支持多模态轨迹生成；4）确保轨迹平滑；5）支持多达三个连续任务参数；6）实时性能好，在线生成时间仅0.2秒。相比之前的工作，该方法比传统采样方法（如RRT-Connect）计算更快，轨迹更平滑；比其他基于学习的方法需要更少训练数据；比仅使用PI²优化的方法将计算负担转移到离线阶段；比其他DMP方法支持更多任务参数和更复杂场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合动态运动基元和强化学习的高效避障方法，只需一个人工演示即可快速生成平滑、接近最优的无碰撞轨迹，并能自动适应不同的障碍物配置和末端执行器尺寸。'}


### 论文摘要

Learning-based motion planning can quickly generate near-optimal trajectories. However, it often requires either large training datasets or costly collection of human demonstrations. This work proposes an alternative approach that quickly generates smooth, near-optimal collision-free 3D Cartesian trajectories from a single artificial demonstration. The demonstration is encoded as a Dynamic Movement Primitive (DMP) and iteratively reshaped using policy-based reinforcement learning to create a diverse trajectory dataset for varying obstacle configurations. This dataset is used to train a neural network that takes as inputs the task parameters describing the obstacle dimensions and location, derived automatically from a point cloud, and outputs the DMP parameters that generate the trajectory. The approach is validated in simulation and real-robot experiments, outperforming a RRT-Connect baseline in terms of computation and execution time, as well as trajectory length, while supporting multi-modal trajectory generation for different obstacle geometries and end-effector dimensions. Videos and the implementation code are available at https://github.com/DominikUrbaniak/obst-avoid-dmp-pi2.

---

## 20. MambaH-Fit: Rethinking Hyper-surface Fitting-based Point Cloud Normal Estimation via State Space Modelling

**论文链接:** [http://arxiv.org/abs/2510.09088v1](http://arxiv.org/abs/2510.09088v1)

**作者:** Weijia Wang, Yuanzhi Su, Pei-Gen Ye, Yuan-Gen Wang, Xuequan Lu

**发布时间:** 2025-10-10

**备注:** 11 pages, 12 figures

### GPT解析

### 总结

本文提出了MambaH-Fit，一种专门用于基于超曲面拟合的点云法线估计的状态空间建模框架。该方法通过注意力驱动的分层特征融合和基于块的状态空间模型，有效解决了现有方法在建模细粒度几何结构方面的不足，显著提高了点云法线估计的准确性、鲁棒性和灵活性。

### 背景

现有的点云法线估计方法在建模细粒度几何结构方面存在不足，限制了预测法线的准确性。虽然状态空间模型(特别是Mamba)已显示出强大的建模能力，能够以线性复杂度捕捉长程依赖关系，但现有的基于Mamba的方法主要关注全局形状结构的理解，对局部、细粒度几何细节的建模探索不足。

### 目的

开发一种能够有效建模局部、细粒度几何细节的点云法线估计方法，以提高预测法线的准确性、鲁棒性和灵活性，解决现有方法在精细几何结构建模方面的局限性。

### 方法

首先提出了一种注意力驱动的分层特征融合(AHFF)方案，用于自适应融合多尺度点云块特征，显著增强局部点云邻域中的几何上下文学习。在此基础上，进一步提出了基于块的状态空间模型(PSSM)，通过状态动力学将点云块建模为隐式超曲面，实现法线预测的有效细粒度几何理解。

### 主要发现

在基准数据集上的大量实验表明，MambaH-Fit方法在准确性、鲁棒性和灵活性方面均优于现有方法。消融研究进一步验证了所提出的AHFF和PSSM组件对方法性能的重要贡献。

### 结论

MambaH-Fit通过结合注意力驱动的分层特征融合和基于块的状态空间模型，成功解决了点云法线估计中细粒度几何结构建模的挑战，为点云处理提供了新的思路和方法，具有重要的理论和实践意义。

### 翻译

我们提出了MambaH-Fit，一种专门用于基于超曲面拟合的点云法线估计的状态空间建模框架。现有的法线估计方法在建模细粒度几何结构方面往往表现不足，从而限制了预测法线的准确性。最近，状态空间模型(SSMs)，特别是Mamba，已经展示了强大的建模能力，能够以线性复杂度捕捉长程依赖关系，并启发了点云处理的适应性方法。然而，现有的基于Mamba的方法主要关注理解全局形状结构，而对局部、细粒度几何细节的建模探索不足。为了解决上述问题，我们首先引入了一种注意力驱动的分层特征融合(AHFF)方案，用于自适应融合多尺度点云块特征，显著增强了局部点云邻域中的几何上下文学习。在此基础上，我们进一步提出了基于块的状态空间模型(PSSM)，通过状态动力学将点云块建模为隐式超曲面，实现了法线预测的有效细粒度几何理解。在基准数据集上的大量实验表明，我们的方法在准确性、鲁棒性和灵活性方面优于现有方法。消融研究进一步验证了所提出组件的贡献。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云法向量估计问题，即从3D点云数据中准确预测每个点的表面法线方向。这个问题在3D视觉领域非常重要，因为准确的法向量是许多应用的基础，包括点云过滤、点云配准和表面重建等。原始点云缺乏连接信息且通常带有噪声，使得法向量估计变得困难，而现有方法在建模细粒度几何结构方面存在不足，限制了预测法向量的准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，指出它们在建模细粒度几何结构方面的不足。他们借鉴了状态空间模型（特别是Mamba）在建模长程依赖关系方面的能力，这种模型最初在自然语言处理领域表现出色。同时，他们参考了HSurf-Net的隐式超表面拟合思想，但发现其残差块结构独立处理点特征，没有明确建模补丁内的点间关系。作者还注意到Transformer的自注意力机制虽然有效，但其二次方计算复杂度不适合处理大规模点云。基于这些观察，他们设计了两个关键模块：注意力驱动的分层特征融合模块和基于补丁的状态空间模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用状态空间模型（特别是Mamba）来建模点云补丁内的隐式超表面，从而更准确地估计点云法向量。方法通过注意力机制自适应地融合多尺度几何特征，并利用Mamba的高效序列建模能力捕捉点云补丁内的长程依赖关系。整体流程包括：1)从点云中提取局部邻域并进行归一化处理；2)使用点特征编码器提取几何特征；3)通过AHFF模块融合多尺度特征；4)使用PSSM模块将点特征作为序列输入，通过Mamba块建模隐式超表面；5)估计点权重并预测法向量；6)对预测结果进行归一化和方向调整。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)注意力驱动的分层特征融合模块，能自适应融合多尺度特征并学习相关几何区域；2)基于补丁的状态空间模型，首次将Mamba应用于点云法向量估计，有效建模补丁内点间关系。相比之前的工作，不同之处在于：不需要预定义多项式阶数（优于n-jet拟合）；明确建模点间关系（优于HSurf-Net的独立点处理）；计算复杂度为线性（优于Transformer的二次复杂度）；专注于局部细粒度几何细节（优于现有Mamba方法的全局关注）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MambaH-Fit首次将状态空间模型引入点云法向量估计领域，通过创新的注意力驱动的分层特征融合和基于补丁的状态空间模型，显著提高了法向量估计的准确性、鲁棒性和对复杂几何结构的适应性。'}


### 论文摘要

We present MambaH-Fit, a state space modelling framework tailored for hyper-surface fitting-based point cloud normal estimation. Existing normal estimation methods often fall short in modelling fine-grained geometric structures, thereby limiting the accuracy of the predicted normals. Recently, state space models (SSMs), particularly Mamba, have demonstrated strong modelling capability by capturing long-range dependencies with linear complexity and inspired adaptations to point cloud processing. However, existing Mamba-based approaches primarily focus on understanding global shape structures, leaving the modelling of local, fine-grained geometric details largely under-explored. To address the issues above, we first introduce an Attention-driven Hierarchical Feature Fusion (AHFF) scheme to adaptively fuse multi-scale point cloud patch features, significantly enhancing geometric context learning in local point cloud neighbourhoods. Building upon this, we further propose Patch-wise State Space Model (PSSM) that models point cloud patches as implicit hyper-surfaces via state dynamics, enabling effective fine-grained geometric understanding for normal prediction. Extensive experiments on benchmark datasets show that our method outperforms existing ones in terms of accuracy, robustness, and flexibility. Ablation studies further validate the contribution of the proposed components.

---

## 21. Exploring Single Domain Generalization of LiDAR-based Semantic Segmentation under Imperfect Labels

**论文链接:** [http://arxiv.org/abs/2510.09035v1](http://arxiv.org/abs/2510.09035v1)

**作者:** Weitong Kong, Zichao Zeng, Di Wen, Jiale Wei, Kunyu Peng, June Moh Goo, Jan Boehm, Rainer Stiefelhagen

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文提出了一种名为DuNe的双视图框架，用于解决带噪声标签的LiDAR语义分割领域泛化问题，在不同数据集上取得了最先进的性能。

### 背景

准确感知对车辆安全至关重要，LiDAR是自动驾驶的关键技术。LiDAR标注常因传感器不完美、遮挡和人为错误而存在噪声，降低分割精度并在领域转移时进一步放大，威胁系统可靠性。点云的稀疏和不规则结构限制了2D噪声学习方法在3D LiDAR分割中的直接应用。

### 目的

引入带噪声标签的LiDAR语义分割领域泛化(DGLSS-NL)这一新任务，建立首个基准，并提出有效方法解决现有噪声标签学习方法对LiDAR数据适应性差的问题。

### 方法

提出DuNe双视图框架，包含强分支和弱分支，强制执行特征级别的一致性，并基于置信感知的预测过滤应用交叉熵损失。

### 主要发现

在10%对称标签噪声下，在SemanticKITTI上达到56.86% mIoU，在nuScenes上达到42.28% mIoU，在SemanticPOSS上达到52.58% mIoU，总体算术平均(AM)为49.57%，调和平均(HM)为48.50%。

### 结论

DuNe框架展示了在DGLSS-NL任务中具有强大的领域泛化能力，代码已在项目页面公开。

### 翻译

准确的感知对车辆安全至关重要，LiDAR是自动驾驶的关键使能技术。为确保在不同环境、传感器类型和天气条件下的鲁棒性能且无需昂贵的重新标注，基于LiDAR的3D语义分割中的领域泛化是必不可少的。然而，由于传感器不完美、遮挡和人为错误，LiDAR标注通常存在噪声。这种噪声会降低分割精度，并在领域转移时进一步放大，威胁系统可靠性。虽然噪声标签学习在图像中已被广泛研究，但其扩展到领域泛化下的3D LiDAR分割基本上仍未被探索，因为点云的稀疏和不规则结构限制了2D方法的直接使用。为解决这一差距，我们引入了带噪声标签的LiDAR语义分割领域泛化这一新任务，并通过将三种代表性的噪声标签学习策略从图像分类调整到3D分割，建立了首个基准。然而，我们发现现有的噪声标签学习方法对LiDAR数据的适应性较差。因此，我们提出了DuNe，一个具有强分支和弱分支的双视图框架，强制执行特征级别的一致性，并基于置信感知的预测过滤应用交叉熵损失。我们的方法在SemanticKITTI上实现了56.86% mIoU，在nuScenes上实现了42.28%，在SemanticPOSS上实现了52.58%，在10%对称标签噪声下展示了最先进的性能，总体算术平均(AM)为49.57%，调和平均(HM)为48.50%，从而证明了在DGLSS-NL任务中具有强大的领域泛化能力。代码可在我们的项目页面上获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在带有不完美标签（噪声标签）条件下实现激光雷达语义分割的单域泛化问题。这个问题在现实中非常重要，因为自动驾驶安全依赖于准确的3D环境感知，而激光雷达是核心传感器；实际应用中激光雷达标注不可避免地存在噪声，且现有方法大多假设完美标注；噪声标签会降低分割精度，在域转移情况下这种影响会被放大，威胁系统可靠性；同时，重新标注不同环境下的数据成本高昂，域泛化可以避免这一成本。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有域泛化方法在噪声标签下的局限性，并观察到图像领域的噪声标签学习方法不能直接迁移到3D点云，因为点云是稀疏、不规则和无序的。作者借鉴了图像领域的三种代表性噪声标签学习方法（TCL、DISC、NPN），以及DGLSS框架中的稀疏不变特征一致性和语义相关性一致性，还借鉴了PolarMix数据增强技术。基于这些现有工作，作者设计了DuNe双视图框架，结合了几何感知的强视图和互补的弱视图，通过瓶颈一致性对齐它们，并采用基于置信度过滤的部分和负监督，针对性地解决了点云特性带来的挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过双视图学习（强视图和弱视图）来处理噪声标签下的域泛化问题。强视图使用几何混合生成增强点云用于构建噪声鲁棒标签，弱视图使用原始和稀疏增强版本形成配对输入并强制执行一致性损失。整体流程包括：1)使用PolarMix将点云增强为强视图和弱视图；2)通过稀疏增强生成四个派生视图；3)使用稀疏卷积网络编码特征；4)在强视图和弱视图中分别生成预测；5)结合DGLSS损失、NPN损失和双视图特征一致性损失进行训练；6)推理时仅使用强分支以提高效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出DGLSS-NL新任务并建立首个基准；2)设计DuNe双视图框架，结合几何感知视图、一致性学习和噪声感知监督；3)实现噪声感知域泛化，能抵抗标签污染和域转移。相比之前工作，不同之处在于：不同于DGLSS假设完美标注，本文处理噪声标签；不同于图像噪声学习方法，本文针对3D点云特性；不同于大多数3D方法只关注单一问题，本文同时处理域转移和噪声标签；方法设计上创新性地采用双视图、自适应选择策略和多种损失函数结合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了DuNe双视图框架，通过结合几何感知视图、一致性学习和噪声感知监督，首次实现了在噪声标签下具有强域泛化能力的激光雷达语义分割，并建立了首个DGLSS-NL基准，显著提升了自动驾驶感知系统在真实世界复杂环境中的可靠性和鲁棒性。'}


### 论文摘要

Accurate perception is critical for vehicle safety, with LiDAR as a key enabler in autonomous driving. To ensure robust performance across environments, sensor types, and weather conditions without costly re-annotation, domain generalization in LiDAR-based 3D semantic segmentation is essential. However, LiDAR annotations are often noisy due to sensor imperfections, occlusions, and human errors. Such noise degrades segmentation accuracy and is further amplified under domain shifts, threatening system reliability. While noisy-label learning is well-studied in images, its extension to 3D LiDAR segmentation under domain generalization remains largely unexplored, as the sparse and irregular structure of point clouds limits direct use of 2D methods. To address this gap, we introduce the novel task Domain Generalization for LiDAR Semantic Segmentation under Noisy Labels (DGLSS-NL) and establish the first benchmark by adapting three representative noisy-label learning strategies from image classification to 3D segmentation. However, we find that existing noisy-label learning approaches adapt poorly to LiDAR data. We therefore propose DuNe, a dual-view framework with strong and weak branches that enforce feature-level consistency and apply cross-entropy loss based on confidence-aware filtering of predictions. Our approach shows state-of-the-art performance by achieving 56.86% mIoU on SemanticKITTI, 42.28% on nuScenes, and 52.58% on SemanticPOSS under 10% symmetric label noise, with an overall Arithmetic Mean (AM) of 49.57% and Harmonic Mean (HM) of 48.50%, thereby demonstrating robust domain generalization in DGLSS-NL tasks. The code is available on our project page.

---

## 22. MagicDock: Toward Docking-oriented De Novo Ligand Design via Gradient Inversion

**论文链接:** [http://arxiv.org/abs/2510.09020v1](http://arxiv.org/abs/2510.09020v1)

**作者:** Zekai Chen, Xunkai Li, Sirui Zhang, Henan Sun, Jia Li, Zhenjun Li, Bing Zhou, Rong-Hua Li, Guoren Wang

**发布时间:** 2025-10-10

**备注:** 52 pages, 14 figures, 12 tables

### GPT解析

### 总结

MagicDock是一个创新的前瞻性框架，基于渐进流程和可微表面建模，解决了从头配体设计中的伪从头设计、有限对接建模和不灵活配体类型等限制问题。

### 背景

从头配体设计是一项基础任务，旨在从头开始生成能够有效对接蛋白质受体并实现强结合亲和力的蛋白质或分子候选物。它对广泛的生物医学应用具有极其重要的意义。

### 目的

解决现有从头配体设计研究中存在的伪从头设计、有限对接建模和不灵活配体类型三大限制问题。

### 方法

MagicDock采用精心设计的梯度反转框架，整合受体和配体的通用对接知识；强调对接过程中的可微表面建模，利用可学习的3D点云表示精确捕获结合细节；为不同类型配体引入定制设计并整合到统一框架中。

### 主要发现

在9种场景中的广泛实验表明，MagicDock比专门针对蛋白质或分子配体设计的最先进基线方法分别实现了27.1%和11.7%的平均改进。

### 结论

MagicDock通过创新的方法解决了从头配体设计中的关键限制，在多种场景中表现出优越的性能，为生物医学应用提供了更有效的配体设计解决方案。

### 翻译

从头配体设计是一项基础任务，旨在从头开始生成能够有效对接蛋白质受体并实现强结合亲和力的蛋白质或分子候选物。它对广泛的生物医学应用具有极其重要的意义。然而，大多数现有研究受限于伪从头设计、有限的对接建模和不灵活的配体类型。为解决这些问题，我们提出了MagicDock，一个基于渐进流程和可微表面建模的前瞻性框架。我们采用精心设计的梯度反转框架，首先将受体和配体的通用对接知识整合到骨干模型中，然后通过结合预测将对接知识实例化为反向梯度流，迭代指导配体的从头生成。我们强调对接过程中的可微表面建模，利用可学习的3D点云表示来精确捕获结合细节，确保生成的配体通过直接和可解释的空间指纹保持对接有效性。我们为不同类型的配体引入定制设计，并将它们整合到具有灵活触发器的统一梯度反转框架中，确保广泛适用性。此外，我们为MagicDock的每个组件提供严格的理论保证。在9种场景中的广泛实验表明，MagicDock比专门针对蛋白质或分子配体设计的最先进基线方法分别实现了27.1%和11.7%的平均改进。


### 论文摘要

De novo ligand design is a fundamental task that seeks to generate protein or molecule candidates that can effectively dock with protein receptors and achieve strong binding affinity entirely from scratch. It holds paramount significance for a wide spectrum of biomedical applications. However, most existing studies are constrained by the \textbf{Pseudo De Novo}, \textbf{Limited Docking Modeling}, and \textbf{Inflexible Ligand Type}. To address these issues, we propose MagicDock, a forward-looking framework grounded in the progressive pipeline and differentiable surface modeling. (1) We adopt a well-designed gradient inversion framework. To begin with, general docking knowledge of receptors and ligands is incorporated into the backbone model. Subsequently, the docking knowledge is instantiated as reverse gradient flows by binding prediction, which iteratively guide the de novo generation of ligands. (2) We emphasize differentiable surface modeling in the docking process, leveraging learnable 3D point-cloud representations to precisely capture binding details, thereby ensuring that the generated ligands preserve docking validity through direct and interpretable spatial fingerprints. (3) We introduce customized designs for different ligand types and integrate them into a unified gradient inversion framework with flexible triggers, thereby ensuring broad applicability. Moreover, we provide rigorous theoretical guarantees for each component of MagicDock. Extensive experiments across 9 scenarios demonstrate that MagicDock achieves average improvements of 27.1\% and 11.7\% over SOTA baselines specialized for protein or molecule ligand design, respectively.

---

## 23. FOLK: Fast Open-Vocabulary 3D Instance Segmentation via Label-guided Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2510.08849v1](http://arxiv.org/abs/2510.08849v1)

**作者:** Hongrui Wu, Zhicheng Gao, Jin Cao, Kelu Yao, Wen Shen, Zhihua Wei

**发布时间:** 2025-10-09

### GPT解析

### 总结

本文提出了一种名为FOLK的快速开放词汇3D实例分割方法，通过标签引导的知识蒸馏技术，解决了现有方法中因2D遮挡引入的噪声问题，并显著提高了推理速度。

### 背景

开放词汇3D实例分割旨在分割和分类超出标注标签空间的实例。现有方法通常将3D实例映射到2D RGB-D图像，然后使用视觉语言模型进行分类，但这种映射策略会引入来自2D遮挡的噪声，并且在推理过程中需要大量计算和内存成本，降低了推理速度。

### 目的

解决现有方法中由2D遮挡引入的噪声问题，减少计算和内存成本，加速推理过程。

### 方法

提出了一种基于标签引导知识蒸馏的快速开放词汇3D实例分割方法(FOLK)。设计一个教师模型提取高质量的实例嵌入，并将其开放词汇知识蒸馏到3D学生模型中。教师模型为每个3D实例生成2D CLIP嵌入，结合可见性和视角多样性，作为蒸馏的学习目标。开发3D学生模型直接为每个3D实例生成3D嵌入。提出标签引导的蒸馏算法，将标签一致的2D嵌入中的开放词汇知识蒸馏到学生模型中。

### 主要发现

在ScanNet200和Replica数据集上进行了实验，在ScanNet200数据集上达到了最先进的性能，AP50得分为35.7，比之前的方法运行速度大约快6.0倍到152.2倍。

### 结论

FOLK方法有效解决了现有方法中的噪声和计算效率问题，代码将在论文被接受后发布。

### 翻译

开放词汇3D实例分割(Open-vocabulary 3D instance segmentation)：指能够分割和分类训练时未见过类别的3D实例的技术。知识蒸馏(Knowledge distillation)：将大型教师模型的知识转移到小型学生模型的技术。CLIP嵌入：由CLIP模型生成的表示文本和图像之间关系的向量。AP50：在IoU阈值为0.5时的平均精度，常用于评估实例分割性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决开放词汇3D实例分割中的效率和准确性问题。现有方法通常将3D实例映射到2D图像再进行分类，但这种方式会受到遮挡引入噪声，同时计算量大导致推理速度慢。这个问题在现实中很重要，因为开放词汇3D实例分割能识别训练中未见过的物体类别，这对自动驾驶、机器人导航等需要处理多样化场景的应用至关重要，而现有方法的速度瓶颈限制了这些技术的实际部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：2D映射引入噪声和计算效率低。他们提出核心思路是通过知识蒸馏将2D视觉语言模型(如CLIP)的知识转移到3D模型中，使3D模型能直接从点云分类。具体设计包括：1)教师模型使用多视图选择和密度引导掩码完成算法生成高质量2D嵌入；2)学生模型直接生成3D嵌入；3)标签引导蒸馏算法确保知识传递质量。作者借鉴了Mask3D用于3D提议生成、CLIP的表示能力、MaskCLIP++的掩码特征提取以及知识蒸馏技术，但进行了创新整合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过标签引导的知识蒸馏，将2D视觉语言模型的知识转移到3D学生模型中，使3D模型能直接从点云提取嵌入并分类，避免2D映射带来的噪声和计算开销。整体流程：1)教师模型阶段：对每个3D实例，选择多视图图像，生成精确掩码，提取高质量2D嵌入；2)学生模型阶段：从点云提取特征，生成3D实例嵌入；3)蒸馏阶段：过滤语义不一致的2D嵌入，通过对比损失和标签损失训练学生模型；4)推理阶段：仅用训练好的3D学生模型直接处理点云，高效生成分割结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)多视图选择算法：同时考虑可见性和视角多样性，选择代表性图像；2)密度引导掩码完成算法：从稀疏掩码生成精确密集掩码，减少背景噪声；3)标签引导蒸馏算法：过滤语义不一致的嵌入，确保知识质量；4)端到端3D嵌入提取：训练后直接从点云分类，无需2D映射。不同之处：传统方法需将3D映射到2D再分类，易受遮挡影响且计算量大；FOLK直接处理3D数据，避免了噪声问题，推理速度提高了6-152倍，同时保持了高精度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出FOLK方法，通过标签引导的知识蒸馏将2D视觉语言模型的知识转移到3D学生模型中，实现了直接从点云进行高效准确的开放词汇3D实例分割，避免了传统2D映射方法的噪声和计算瓶颈问题。'}


### 论文摘要

Open-vocabulary 3D instance segmentation seeks to segment and classify instances beyond the annotated label space. Existing methods typically map 3D instances to 2D RGB-D images, and then employ vision-language models (VLMs) for classification. However, such a mapping strategy usually introduces noise from 2D occlusions and incurs substantial computational and memory costs during inference, slowing down the inference speed. To address the above problems, we propose a Fast Open-vocabulary 3D instance segmentation method via Label-guided Knowledge distillation (FOLK). Our core idea is to design a teacher model that extracts high-quality instance embeddings and distills its open-vocabulary knowledge into a 3D student model. In this way, during inference, the distilled 3D model can directly classify instances from the 3D point cloud, avoiding noise caused by occlusions and significantly accelerating the inference process. Specifically, we first design a teacher model to generate a 2D CLIP embedding for each 3D instance, incorporating both visibility and viewpoint diversity, which serves as the learning target for distillation. We then develop a 3D student model that directly produces a 3D embedding for each 3D instance. During training, we propose a label-guided distillation algorithm to distill open-vocabulary knowledge from label-consistent 2D embeddings into the student model. FOLK conducted experiments on the ScanNet200 and Replica datasets, achieving state-of-the-art performance on the ScanNet200 dataset with an AP50 score of 35.7, while running approximately 6.0x to 152.2x faster than previous methods. All codes will be released after the paper is accepted.

---

## 24. SatDreamer360: Multiview-Consistent Generation of Ground-Level Scenes from Satellite Imagery

**论文链接:** [http://arxiv.org/abs/2506.00600v2](http://arxiv.org/abs/2506.00600v2)

**作者:** Xianghui Ze, Beiyi Zhu, Zhenbo Song, Jianfeng Lu, Yujiao Shi

**发布时间:** 2025-05-31

### GPT解析

### 总结

本文提出了一种名为SatDreamer360的框架，能够从单张卫星图像生成几何上多视角一致的地面全景图，解决了现有方法难以产生多视角一致序列的问题。

### 背景

生成多视角一致的360度地面场景卫星影像是一个具有挑战性的任务，在模拟、自主导航和数字孪生城市等领域有广泛应用。现有方法主要专注于合成单个地面全景图，通常依赖高度图或手工制作的投影等辅助输入，难以产生多视角一致的序列。

### 目的

提出SatDreamer360框架，从单张卫星图像生成几何上多视角一致的地面全景图，给定预定义的位置轨迹。

### 方法

采用三平面表示法编码场景特征；设计基于射线的像素注意力机制从三平面检索特定视角特征；引入全景极线约束注意力模块根据已知相对姿态对齐跨帧特征；扩展VIGOR数据集创建VIGOR++，包含更多地面图像及其姿态标注。

### 主要发现

实验表明SatDreamer360在卫星到地面对齐和多视角一致性方面都优于现有方法。

### 结论

SatDreamer360能够有效地从卫星图像生成多视角一致的360度地面场景，解决了卫星与地面图像之间的大视角差异问题。

### 翻译

从卫星影像生成多视角一致的360度地面场景是一项具有挑战性的任务，在模拟、自主导航和数字孪生城市等领域有广泛应用。现有方法主要专注于合成单个地面全景图，通常依赖高度图或手工制作的投影等辅助输入，难以产生多视角一致的序列。本文提出SatDreamer360框架，能够从单张卫星图像生成几何上多视角一致的地面全景图，给定预定义的位置轨迹。为解决地面与卫星图像之间的大视角差异问题，我们采用三平面表示法编码场景特征，并设计基于射线的像素注意力机制从三平面中检索特定视角特征。为保持多帧一致性，我们引入全景极线约束注意力模块，根据已知的相对姿态对齐跨帧特征。为支持评估，我们通过增加更多地面图像及其姿态标注扩展了原始VIGOR数据集，创建了VIGOR++数据集。实验表明，SatDreamer360在卫星到地面对齐和多视角一致性方面都优于现有方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从卫星图像生成多视角一致的地面全景场景问题。这个问题在现实世界中非常重要，因为它有广泛的应用，包括模拟环境、自动驾驶和数字孪生城市建设。卫星图像覆盖范围广且获取成本低，但与地面视角差异巨大，现有方法难以生成连续、几何一致的地面场景序列，限制了这些应用的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，包括依赖辅助输入(如高度图)和难以保证多视角一致性。他们借鉴了三平面表示技术来编码3D场景特征，并从针孔相机的极线约束概念中获得灵感，将其扩展到全景图像。方法设计基于扩散模型，特别是Stable Diffusion 1.5，并添加了两个关键模块：基于射线的像素注意力机制和全景极线约束注意力模块。这些创新使模型能够从单个卫星图像生成连续且几何一致的地面场景序列。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用三平面表示编码卫星图像中的3D场景特征，并通过基于射线的像素注意力机制从三平面中检索视图特定特征，同时利用全景极线约束注意力模块确保多帧之间的一致性。整体流程是：首先将卫星图像转换为三平面表示；然后为每个地面像素定义3D射线并沿射线采样点；接着从三平面中提取这些点的特征；再利用极线约束对齐不同帧的特征；最后通过扩散模型迭代去噪生成连续的地面全景图像序列。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一框架SatDreamer360，可从单个卫星图像生成连续地面场景；2) 三平面表示编码场景几何，避免依赖高度图；3) 基于射线的像素注意力机制，实现像素级几何感知；4) 全景极线约束注意力模块，确保多视角一致性；5) 新建VIGOR++数据集。相比之前工作，不同之处在于：不需要辅助输入；只需单个卫星图像；能生成多视角一致序列；显式处理几何一致性；支持大规模场景生成。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SatDreamer360通过创新的三平面表示和极线约束注意力机制，实现了从单个卫星图像生成多视角一致的地面全景场景，为模拟、自动驾驶和数字孪生城市等应用提供了新工具。'}


### 论文摘要

Generating multiview-consistent $360^\circ$ ground-level scenes from satellite imagery is a challenging task with broad applications in simulation, autonomous navigation, and digital twin cities. Existing approaches primarily focus on synthesizing individual ground-view panoramas, often relying on auxiliary inputs like height maps or handcrafted projections, and struggle to produce multiview consistent sequences. In this paper, we propose SatDreamer360, a framework that generates geometrically consistent multi-view ground-level panoramas from a single satellite image, given a predefined pose trajectory. To address the large viewpoint discrepancy between ground and satellite images, we adopt a triplane representation to encode scene features and design a ray-based pixel attention mechanism that retrieves view-specific features from the triplane. To maintain multi-frame consistency, we introduce a panoramic epipolar-constrained attention module that aligns features across frames based on known relative poses. To support the evaluation, we introduce {VIGOR++}, a large-scale dataset for generating multi-view ground panoramas from a satellite image, by augmenting the original VIGOR dataset with more ground-view images and their pose annotations. Experiments show that SatDreamer360 outperforms existing methods in both satellite-to-ground alignment and multiview consistency.

---

## 25. Scaling Language-Centric Omnimodal Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.11693v1](http://arxiv.org/abs/2510.11693v1)

**作者:** Chenghao Xiao, Hou Pong Chan, Hao Zhang, Weiwen Xu, Mahani Aljunied, Yu Rong

**发布时间:** 2025-10-13

**备注:** NeurIPS 2025

### GPT解析

### 总结

本研究探讨了基于多模态大语言模型(MLLMs)并通过对比学习(CL)微调的多模态嵌入方法优越性的根本原因，提出了一种以语言为中心的全模态嵌入框架LCO-Emb，并通过实验验证了其有效性。

### 背景

最近基于多模态大语言模型(MLLMs)并通过对比学习(CL)微调的多模态嵌入方法显示出有希望的结果，但它们优越性的根本原因尚未被充分探索。

### 目的

探究MLLM方法优越性的根本原因，并基于此提出一种新的嵌入框架，提高多模态表示的性能。

### 方法

通过各向异性和核相似性结构的分析，确认MLLM表示中存在潜在的对齐，使CL能够作为一个轻量级的精炼阶段。基于这一见解，提出了一个以语言为中心的全模态嵌入框架LCO-Emb，并在不同骨干网络和基准上进行了广泛实验。

### 主要发现

1. MLLM方法的关键优势来自于生成预训练过程中实现的隐式跨模态对齐；2. 提出了表示能力-生成能力缩放定律(GRSL)，表明通过对比精炼获得的表示能力与MLLM的生成能力呈正相关；3. 提供了GRSL的理论解释，正式将MLLM的生成质量与其表示性能的上限联系起来。

### 结论

提高生成能力是提升表示质量的有效范式，在CL之前进行持续的生成预训练可以进一步增强模型的嵌入能力潜力。

### 翻译

最近利用通过对比学习(CL)微调的多模态大语言模型(MLLMs)的多模态嵌入方法显示出有希望的结果，但它们优越性的根本原因仍未被充分探索。本文认为，基于MLLM方法的关键优势来自于生成预训练过程中实现的隐式跨模态对齐，其中语言解码器学习在共享表示空间中利用多模态信号来生成单模态输出。通过各向异性和核相似性结构的分析，我们经验上确认潜在对齐出现在MLLM表示中，使CL能够作为一个轻量级的精炼阶段。利用这一见解，我们提出了一个以语言为中心的全模态嵌入框架，称为LCO-Emb。在不同骨干网络和基准上的广泛实验证明了其有效性，在各模态上实现了最先进的性能。此外，我们确定了表示能力-生成能力缩放定律(GRSL)，表明通过对比精炼获得的表示能力与MLLM的生成能力呈正相关。这表明提高生成能力是提升表示质量的有效范式。我们提供了GRSL的理论解释，正式将MLLM的生成质量与其表示性能的上限联系起来，并在一个具有挑战性的低资源视觉文档检索任务上验证了这一点，表明在CL之前进行持续的生成预训练可以进一步增强模型嵌入能力的潜力。代码、模型和资源可在 https://github.com/LCO-Embedding/LCO-Embedding 获取。


### 论文摘要

Recent multimodal embedding approaches leveraging multimodal large language models (MLLMs) fine-tuned with contrastive learning (CL) have shown promising results, yet the underlying reasons behind their superiority remain underexplored. This work argues that a crucial advantage of MLLM-based approaches stems from implicit cross-modal alignment achieved during generative pretraining, where the language decoder learns to exploit multimodal signals within a shared representation space for generating unimodal outputs. Through analysis of anisotropy and kernel similarity structure, we empirically confirm that latent alignment emerges within MLLM representations, allowing CL to serve as a lightweight refinement stage. Leveraging this insight, we propose a Language-Centric Omnimodal Embedding framework, termed LCO-Emb. Extensive experiments across diverse backbones and benchmarks demonstrate its effectiveness, achieving state-of-the-art performance across modalities. Furthermore, we identify a Generation-Representation Scaling Law (GRSL), showing that the representational capabilities gained through contrastive refinement scales positively with the MLLM's generative capabilities. This suggests that improving generative abilities evolves as an effective paradigm for enhancing representation quality. We provide a theoretical explanation of GRSL, which formally links the MLLM's generative quality to the upper bound on its representation performance, and validate it on a challenging, low-resource visual-document retrieval task, showing that continual generative pretraining before CL can further enhance the potential of a model's embedding capabilities. Codes, models, and resources are available at https://github.com/LCO-Embedding/LCO-Embedding.

---

## 26. Query-Specific GNN: A Comprehensive Graph Representation Learning Method for Retrieval Augmented Generation

**论文链接:** [http://arxiv.org/abs/2510.11541v1](http://arxiv.org/abs/2510.11541v1)

**作者:** Yuchen Yan, Zhihua Liu, Hao Wang, Weiming Li, Xiaoshuai Hao

**发布时间:** 2025-10-13

### GPT解析

### 总结

该论文提出了一种用于多跳问题检索的新型图表示学习框架，通过多信息级知识图和基于查询的图神经网络解决了RAG系统在处理复杂多跳问题时面临的挑战。

### 背景

检索增强生成(RAG)能够通过整合外部知识源增强大语言模型的能力，但在处理需要识别多个知识目标形成综合答案的多跳问题时面临新挑战。

### 目的

提出一种图表示学习框架，解决多跳问题中现有方法难以理解复杂语义结构和易受噪声影响的问题。

### 方法

引入多信息级知识图(Multi-L KG)建模不同信息级别，设计基于查询的图神经网络(QSGNN)进行表示学习，采用层内/层间消息传递机制并由查询引导信息聚合，同时提出两种综合数据生成策略用于预训练QSGNN。

### 主要发现

实验结果表明该框架在多跳场景中有效，特别是在高跳问题上改进可达33.8%。

### 结论

所提出的框架能有效解决多跳问题中的挑战，提高RAG系统在复杂问题上的性能。

### 翻译

检索增强生成(RAG)已证明其通过整合外部知识源增强大语言模型的能力。然而，需要识别多个知识目标以形成综合答案的多跳问题为RAG系统带来了新的挑战。在多跳设置下，现有方法往往难以完全理解具有复杂语义结构的问题，并且在检索多个信息目标时容易受到无关噪声的影响。为解决这些局限性，我们提出了一种用于多跳问题检索的新型图表示学习框架。我们首先引入多信息级知识图(Multi-L KG)来建模不同信息级别，以更全面地理解多跳问题。基于此，我们设计了基于查询的图神经网络(QSGNN)在Multi-L KG上进行表示学习。QSGNN采用层内/层间消息传递机制，每次信息聚合都由查询引导，这不仅促进了多粒度信息聚合，还显著减少了噪声的影响。为增强其学习鲁棒表示的能力，我们进一步提出了两种综合数据生成策略用于预训练QSGNN。广泛的实验结果证明了我们的框架在多跳场景中的有效性，特别是在高跳问题上改进可达33.8%。代码可在以下网址获取：https://github.com/Jerry2398/QSGNN。


### 论文摘要

Retrieval-augmented generation (RAG) has demonstrated its ability to enhance Large Language Models (LLMs) by integrating external knowledge sources. However, multi-hop questions, which require the identification of multiple knowledge targets to form a synthesized answer, raise new challenges for RAG systems. Under the multi-hop settings, existing methods often struggle to fully understand the questions with complex semantic structures and are susceptible to irrelevant noise during the retrieval of multiple information targets. To address these limitations, we propose a novel graph representation learning framework for multi-hop question retrieval. We first introduce a Multi-information Level Knowledge Graph (Multi-L KG) to model various information levels for a more comprehensive understanding of multi-hop questions. Based on this, we design a Query-Specific Graph Neural Network (QSGNN) for representation learning on the Multi-L KG. QSGNN employs intra/inter-level message passing mechanisms, and in each message passing the information aggregation is guided by the query, which not only facilitates multi-granular information aggregation but also significantly reduces the impact of noise. To enhance its ability to learn robust representations, we further propose two synthesized data generation strategies for pre-training the QSGNN. Extensive experimental results demonstrate the effectiveness of our framework in multi-hop scenarios, especially in high-hop questions the improvement can reach 33.8\%. The code is available at: https://github.com/Jerry2398/QSGNN.

---

## 27. Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment

**论文链接:** [http://arxiv.org/abs/2510.11369v1](http://arxiv.org/abs/2510.11369v1)

**作者:** Shijie Zhao, Xuanyu Zhang, Weiqi Li, Junlin Li, Li Zhang, Tianfan Xue, Jian Zhang

**发布时间:** 2025-10-13

### GPT解析

### 总结

本研究提出了一种名为RALI的新算法，用于解决基于推理的图像质量评估模型的高能耗和高延迟问题。通过对比学习直接对齐图像与可泛化文本表示，该方法实现了与基于推理的模型相当的泛化性能，同时显著减少了模型参数和推理时间。

### 背景

基于强化学习的推理图像质量评估模型表现出出色的泛化能力，但其背后的机制和关键因素尚未被充分探索。此外，这些模型虽然性能优越，但推理能耗和延迟比早期模型高出几个数量级，限制了它们在特定场景中的应用。

### 目的

本研究旨在阐明基于推理的IQA模型泛化能力的来源，并提出一种高效的新算法，以减少模型参数和推理时间，同时保持相当的泛化性能。

### 方法

通过大量实验验证，研究揭示了多模态大语言模型(MLLMs)通过强化学习训练，利用推理能力将冗余的视觉表示转换为紧凑的、跨域对齐的文本表示，这是泛化能力的来源。基于此，作者提出了RALI算法，采用对比学习直接将图像与通过RL学习到的可泛化文本表示对齐，消除了对推理过程的依赖和加载大语言模型的必要性。

### 主要发现

1. 通过RL训练，MLLMs利用推理能力将冗余的视觉表示转换为紧凑的、跨域对齐的文本表示。2. 这种转换是基于推理的IQA模型泛化能力的来源。3. RALI算法通过对比学习直接对齐图像与可泛化文本表示，消除了推理过程的依赖。4. RALI实现了与基于推理的模型相当的泛化性能，同时只需要不到5%的模型参数和推理时间。

### 结论

RALI算法成功地解决了基于推理的IQA模型的高能耗和高延迟问题，通过直接对齐图像与文本表示，显著减少了模型复杂度和推理时间，同时保持了相当的泛化性能，为图像质量评估领域提供了一种高效的新方法。

### 翻译

基于强化学习训练的推理图像质量评估模型表现出出色的泛化能力，但其背后的机制和关键驱动因素在当前研究中仍未得到充分探索。此外，尽管这些模型性能优越，但它们的推理能耗和延迟比早期模型高出几个数量级，限制了它们在特定场景中的部署。通过大量实验，本文验证并阐述，通过RL训练，多模态大语言模型利用其推理能力将冗余的视觉表示转换为紧凑的、跨域对齐的文本表示。这种转换正是这些基于推理的IQA模型所表现出的泛化能力的来源。基于这一基本洞察，我们提出了RALI这一新颖算法，它采用对比学习直接将图像与通过RL学习到的可泛化文本表示对齐。这种方法消除了对推理过程的依赖，甚至不需要加载大语言模型。对于质量评分任务，该框架实现了与基于推理的模型相当的泛化性能，同时只需要不到5%的模型参数和推理时间。


### 论文摘要

Reasoning-based image quality assessment (IQA) models trained through reinforcement learning (RL) exhibit exceptional generalization, yet the underlying mechanisms and critical factors driving this capability remain underexplored in current research. Moreover, despite their superior performance, these models incur inference energy usage and latency orders of magnitude higher than their earlier counterparts, restricting their deployment in specific scenarios. Through extensive experiments, this paper verifies and elaborates that through RL training, MLLMs leverage their reasoning capability to convert redundant visual representations into compact, cross-domain aligned text representations. This conversion is precisely the source of the generalization exhibited by these reasoning-based IQA models. Building on this fundamental insight, we propose a novel algorithm, RALI, which employs contrastive learning to directly align images with these generalizable text representations learned by RL. This approach eliminates the reliance on reasoning processes and even obviates the need to load an LLM. For the quality scoring task, this framework achieves generalization performance comparable to reasoning-based models while requiring less than 5% of their model parameters and inference time.

---

## 28. HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data

**论文链接:** [http://arxiv.org/abs/2510.11321v1](http://arxiv.org/abs/2510.11321v1)

**作者:** Ruizhe Liu, Pei Zhou, Qian Luo, Li Sun, Jun Cen, Yibing Song, Yanchao Yang

**发布时间:** 2025-10-13

**备注:** Accepted at 39th Conference on Neural Information Processing Systems  (NeurIPS 2025)

### GPT解析

### 总结

该研究提出了一种自监督框架，用于学习层次化的操作概念，通过跨模态感官相关性和多级时间抽象捕捉不变的操作模式，无需人工标注。结合跨模态相关网络和多时间尺度预测器，使策略能够专注于可转移的关系模式，同时保持对即时行动和长期目标的意识。实验证明概念增强策略在模拟和实际环境中表现显著提升，学习到的概念类似于人类可解释的操作基元。

### 背景

机器人操作中的有效泛化需要能够捕捉环境和任务间不变交互模式的表示。传统的机器人学习方法通常需要大量人工标注，且难以在不同环境和任务间有效迁移。

### 目的

开发一种自监督学习方法，使机器人能够学习层次化的操作概念，捕捉跨环境和任务的不变交互模式，无需人工标注，从而提高机器人在复杂场景中的性能。

### 方法

提出结合两种主要组件的自监督框架：1) 跨模态相关网络：识别跨感官模态的持久模式；2) 多时间尺度预测器：在不同时间尺度上组织表示层次结构。这种双重结构使策略能够专注于可转移的关系模式，同时保持对即时行动和长期目标的意识。

### 主要发现

1) 概念增强的策略在模拟基准测试和实际部署中表现出显著的性能改进；2) 学习到的概念类似于人类可解释的操作基元，尽管没有接受语义监督；3) 该框架能够捕捉跨环境和任务的不变操作模式。

### 结论

这项研究不仅推进了对操作表示学习的理解，还为在复杂场景中增强机器人性能提供了实用方法。通过自监督学习层次化操作概念，机器人能够在无需人工标注的情况下实现更好的泛化能力。

### 翻译

机器人操作中的有效泛化需要能够捕捉环境和任务间不变交互模式的表示。我们提出了一个自监督框架，用于学习层次化的操作概念，这些概念通过跨模态感官相关性和多级时间抽象来编码这些不变模式，无需人工标注。我们的方法结合了跨模态相关网络（识别跨感官模态的持久模式）和多时间尺度预测器（在不同时间尺度上组织表示层次结构）。通过这种双重结构学习到的操作概念，使策略能够专注于可转移的关系模式，同时保持对即时行动和长期目标的意识。在模拟基准测试和实际部署中的经验评估表明，我们的概念增强策略具有显著的性能改进。分析显示，学习到的概念类似于人类可解释的操作基元，尽管没有接受语义监督。这项工作不仅推进了对操作表示学习的理解，还为在复杂场景中增强机器人性能提供了实用方法。


### 论文摘要

Effective generalization in robotic manipulation requires representations that capture invariant patterns of interaction across environments and tasks. We present a self-supervised framework for learning hierarchical manipulation concepts that encode these invariant patterns through cross-modal sensory correlations and multi-level temporal abstractions without requiring human annotation. Our approach combines a cross-modal correlation network that identifies persistent patterns across sensory modalities with a multi-horizon predictor that organizes representations hierarchically across temporal scales. Manipulation concepts learned through this dual structure enable policies to focus on transferable relational patterns while maintaining awareness of both immediate actions and longer-term goals. Empirical evaluation across simulated benchmarks and real-world deployments demonstrates significant performance improvements with our concept-enhanced policies. Analysis reveals that the learned concepts resemble human-interpretable manipulation primitives despite receiving no semantic supervision. This work advances both the understanding of representation learning for manipulation and provides a practical approach to enhancing robotic performance in complex scenarios.

---

## 29. Causal Disentanglement Learning for Accurate Anomaly Detection in Multivariate Time Series

**论文链接:** [http://arxiv.org/abs/2510.11084v1](http://arxiv.org/abs/2510.11084v1)

**作者:** Wonah Kim, Jeonghyeon Park, Dongsan Jun, Jungkyu Han, Sejin Chun

**发布时间:** 2025-10-13

**备注:** 20 pages, 4 Figures,

### GPT解析

### 总结

本文提出了一种名为CDRL4AD的因果解缠结表示学习方法，用于在多元时间序列中检测异常并识别因果关系，解决了传统方法无法在不同时间段明确推断因果关系的问题。

### 背景

在多元时间序列分析中，数据变量之间的动态交互随时间变化，使因果关系的解释变得复杂。传统方法在无监督设置中假设变量间的统计独立性，而最近的方法通过图表示学习捕获特征相关性，但这些表示无法在不同时间段明确推断因果关系。

### 目的

开发一种能够检测异常并识别其因果关系的方法，特别是在多元时间序列数据中，解决现有方法无法明确推断不同时间段因果关系的问题。

### 方法

提出CDRL4AD（用于异常检测的因果解缠结表示学习）方法，设计因果过程作为模型输入（包括时间异质图和因果关系），使表示能够识别不同时间段的因果关系并解缠结潜在变量以推断相应的因果因子。

### 主要发现

在真实世界数据集上的实验表明，CDRL4AD在准确性和根本原因分析方面优于最先进的方法；模型分析验证了超参数敏感性和CDRL4AD的时间复杂度；案例研究展示了该方法如何帮助人类专家诊断异常的根本原因。

### 结论

CDRL4AD方法有效解决了多元时间序列中异常检测和因果关系推断的挑战，为异常诊断提供了更准确的工具，并能帮助人类专家理解异常的根本原因。

### 翻译

解缠结复杂的因果关系对于准确检测异常很重要。在多元时间序列分析中，数据变量之间的动态交互随时间变化，使因果关系的解释变得复杂。传统方法在无监督设置中假设变量间的统计独立性，而最近的方法通过图表示学习捕获特征相关性。然而，这些表示无法在不同时间段明确推断因果关系。为解决这个问题，我们提出了用于异常检测的因果解缠结表示学习（CDRL4AD），用于在多元时间序列中检测异常并识别其因果关系。首先，我们将因果过程设计为模型输入，包括时间异质图和因果关系。其次，我们的表示能够识别不同时间段的因果关系，并解缠结潜在变量以推断相应的因果因子。第三，我们在真实世界数据集上的实验表明，CDRL4AD在准确性和根本原因分析方面优于最先进的方法。第四，我们的模型分析验证了超参数敏感性和CDRL4AD的时间复杂度。最后，我们进行了案例研究，展示我们的方法如何帮助人类专家诊断异常的根本原因。


### 论文摘要

Disentangling complex causal relationships is important for accurate detection of anomalies. In multivariate time series analysis, dynamic interactions among data variables over time complicate the interpretation of causal relationships. Traditional approaches assume statistical independence between variables in unsupervised settings, whereas recent methods capture feature correlations through graph representation learning. However, their representations fail to explicitly infer the causal relationships over different time periods. To solve the problem, we propose Causally Disentangled Representation Learning for Anomaly Detection (CDRL4AD) to detect anomalies and identify their causal relationships in multivariate time series. First, we design the causal process as model input, the temporal heterogeneous graph, and causal relationships. Second, our representation identifies causal relationships over different time periods and disentangles latent variables to infer the corresponding causal factors. Third, our experiments on real-world datasets demonstrate that CDRL4AD outperforms state-of-the-art methods in terms of accuracy and root cause analysis. Fourth, our model analysis validates hyperparameter sensitivity and the time complexity of CDRL4AD. Last, we conduct a case study to show how our approach assists human experts in diagnosing the root causes of anomalies.

---

## 30. Decoupled Multimodal Fusion for User Interest Modeling in Click-Through Rate Prediction

**论文链接:** [http://arxiv.org/abs/2510.11066v1](http://arxiv.org/abs/2510.11066v1)

**作者:** Alin Fan, Hanqing Li, Sihan Lu, Jingsong Yuan, Jiandong Zhang

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了解耦多模态融合(DMF)方法，通过模态增强建模策略实现基于ID的协同表示和多模态表示的细粒度交互，用于用户兴趣建模。DMF构建目标感知特征桥接不同嵌入空间，设计推理优化注意力机制减轻计算瓶颈，并全面整合多模态表示。实验证明DMF有效，已在Lazada平台部署并取得显著业务指标提升。

### 背景

现代工业推荐系统通过整合预训练模型的多模态表示到基于ID的点击率预测框架中来提高推荐性能。

### 目的

解决现有方法无法捕获内容语义和行为信号之间细粒度交互的问题，提高推荐系统的性能。

### 方法

提出解耦多模态融合(DMF)方法，包括：1)构建目标感知特征桥接不同嵌入空间的语义差距；2)设计推理优化注意力机制解耦计算；3)结合模态中心和模态增强建模策略下的用户兴趣表示实现全面多模态整合。

### 主要发现

在公共和工业数据集上的离线实验证明了DMF的有效性；在Lazada平台部署后，CTCVR提升5.30%，GMV提升7.43%，且计算开销可忽略不计。

### 结论

DMF通过模态增强建模策略有效解决了现有方法无法捕获细粒度交互的问题，显著提升了推荐系统性能，且计算效率高。

### 翻译

现代工业推荐系统通过将预训练模型的多模态表示整合到基于ID的点击率预测框架中来提高推荐性能。然而，现有方法通常采用模态中心建模策略，独立处理基于ID和多模态的嵌入，无法捕获内容语义和行为信号之间的细粒度交互。本文提出了解耦多模态融合(DMF)，它引入了模态增强建模策略，使基于ID的协同表示和多模态表示能够进行细粒度交互，用于用户兴趣建模。具体而言，我们构建目标感知特征来桥接不同嵌入空间之间的语义差距，并将其作为辅助信息来增强用户兴趣建模的效果。此外，我们设计了一种推理优化的注意力机制，在注意力层之前解耦目标感知特征和基于ID的嵌入的计算，从而减轻了引入目标感知特征带来的计算瓶颈。为了实现全面的多模态整合，DMF结合了在模态中心和模态增强建模策略下学习到的用户兴趣表示。在公共和工业数据集上的离线实验证明了DMF的有效性。此外，DMF已被部署在跨境电商平台Lazada的产品推荐系统上，实现了CTCVR提升5.30%和GMV提升7.43%，且计算开销可忽略不计。


### 论文摘要

Modern industrial recommendation systems improve recommendation performance by integrating multimodal representations from pre-trained models into ID-based Click-Through Rate (CTR) prediction frameworks. However, existing approaches typically adopt modality-centric modeling strategies that process ID-based and multimodal embeddings independently, failing to capture fine-grained interactions between content semantics and behavioral signals. In this paper, we propose Decoupled Multimodal Fusion (DMF), which introduces a modality-enriched modeling strategy to enable fine-grained interactions between ID-based collaborative representations and multimodal representations for user interest modeling. Specifically, we construct target-aware features to bridge the semantic gap across different embedding spaces and leverage them as side information to enhance the effectiveness of user interest modeling. Furthermore, we design an inference-optimized attention mechanism that decouples the computation of target-aware features and ID-based embeddings before the attention layer, thereby alleviating the computational bottleneck introduced by incorporating target-aware features. To achieve comprehensive multimodal integration, DMF combines user interest representations learned under the modality-centric and modality-enriched modeling strategies. Offline experiments on public and industrial datasets demonstrate the effectiveness of DMF. Moreover, DMF has been deployed on the product recommendation system of the international e-commerce platform Lazada, achieving relative improvements of 5.30% in CTCVR and 7.43% in GMV with negligible computational overhead.

---

## 31. Instruction-aware User Embedding via Synergistic Language and Representation Modeling

**论文链接:** [http://arxiv.org/abs/2510.11016v1](http://arxiv.org/abs/2510.11016v1)

**作者:** Ziyi Gao, Yike Xu, Jiahao Yuan, Baokun Wang, Jinyong Wen, Xiaotong Lin, Yun Liu, Xing Fu, Yu Cheng, Yongchao Liu, Weiqiang Wang, Zhongle Xie

**发布时间:** 2025-10-13

### GPT解析

### 总结

InstructUE是一种指令感知的用户嵌入基础模型，利用大型语言模型生成通用和指令感知的用户表示，通过多编码器架构和对比-自回归训练框架解决了现有方法在跨领域泛化和噪声敏感性方面的问题。

### 背景

用户表示建模对个性化应用日益重要，但现有方法在跨领域泛化能力和对噪声行为信号的敏感性方面存在困难。

### 目的

提出InstructUE模型，利用大型语言模型生成通用和指令感知的用户表示，提高用户建模的泛化能力和噪声鲁棒性。

### 方法

引入多编码器架构配备轻量级适配器处理异构数据；提出对比-自回归训练框架，通过UserQA数据集连接语言和表示空间，同时利用自回归学习捕获领域知识和对比学习对齐用户-文本嵌入。

### 主要发现

通过现实世界应用的广泛实验，证明InstructUE在用户预测、营销和推荐等多个场景中显著优于现有方法，实现了指令引导的用户信息去噪。

### 结论

指令感知的用户建模可有效实现特定场景下用户信息的指令引导去噪，为更具泛化性和鲁棒性的用户表示学习铺平道路。

### 翻译

用户表示建模已成为个性化应用中日益重要的环节，然而现有方法在跨领域泛化能力和对噪声行为信号的敏感性方面存在挑战。我们提出了InstructUE，一种指令感知的用户嵌入基础模型，它利用大型语言模型生成通用且具有指令感知能力的用户表示。InstructUE引入了一个多编码器架构，配备轻量级适配器，能够高效处理来自六个不同来源的异构数据，同时保留其结构特征。此外，它提出了一种新颖的对比-自回归训练框架，通过精心策划的UserQA数据集连接语言和表示空间。该对比-自回归训练框架同时利用自回归学习捕获语言空间中的领域知识，以及对比学习对齐表示空间中的用户-文本嵌入，从而增强了用户嵌入的指令感知能力和噪声鲁棒性。通过在现实世界应用中的广泛实验，我们证明InstructUE在用户预测、营销和推荐等多个场景中显著优于现有方法。我们的结果表明，指令感知的用户建模可以在特定场景下有效实现用户信息的指令引导去噪，为更具泛化性和鲁棒性的用户表示学习铺平道路。


### 论文摘要

User representation modeling has become increasingly crucial for personalized applications, yet existing approaches struggle with generalizability across domains and sensitivity to noisy behavioral signals. We present InstructUE, an instruction-aware user embedding foundation model that leverages large language models (LLMs) to generate general and instruction-aware user representations. InstructUE introduces a multi-encoder architecture with a lightweight adapter that efficiently processes heterogeneous data from six different sources while preserving their structural characteristics. Additionally, it proposes a novel contrastive-autoregressive training framework that bridges language and representation spaces through a curated UserQA dataset. The contrastive-autoregressive training framework simultaneously leverages autoregressive learning to capture domain knowledge in language space and contrastive learning to align user-text embeddings in representation space, thereby enhancing the instruction-awareness and noise-robustness of user embeddings. Through extensive experiments on real-world applications, we demonstrate that InstructUE significantly outperforms existing methods across multiple domains including user prediction, marketing, and recommendation scenarios. Our results show that instruction-aware user modeling can effectively achieve instruction-guided denoising of user information in specific scenarios, paving the way for more generalizable and robust user representation learning.

---

## 32. Unify Variables in Neural Scaling Laws for General Audio Representations via Embedding Effective Rank

**论文链接:** [http://arxiv.org/abs/2510.10948v1](http://arxiv.org/abs/2510.10948v1)

**作者:** Xuyao Deng, Yanjie Sun, Yong Dou, Kele Xu

**发布时间:** 2025-10-13

### GPT解析

### 总结

本研究系统探讨了通用音频表示学习中的缩放定律，引入嵌入有效秩（RankMe）作为统一指标，揭示了RankMe与表示质量之间的幂律关系，为音频基础模型的缩放策略提供了理论依据和实践框架。

### 背景

缩放定律在计算机视觉和自然语言处理中对模型性能的理解有深远影响，但在通用音频表示学习中的应用尚未充分探索。

### 目的

研究通用音频表示学习中的缩放定律，探索如何评估和预测模型性能。

### 方法

利用嵌入有效秩（RankMe）作为统一指标，封装各种变量对表示质量的影响，在广泛的超参数空间（包括模型大小、训练数据量、计算预算、架构配置等）中检查缩放行为。

### 主要发现

实证研究表明RankMe与表示质量之间存在一致的关系，表明嵌入有效秩可作为评估和预测音频表示学习中模型性能的可靠代理。

### 结论

验证了经典缩放原理适用于通用音频领域，为音频基础模型的未来模型缩放策略提供了理论依据和经验上稳健的框架。

### 翻译

缩放定律已经深刻地改变了我们在计算机视觉和自然语言处理中对模型性能的理解，但它们在通用音频表示学习中的应用仍然探索不足。一个关键挑战在于通用音频表示的多因素性质——表示质量受到音频长度、嵌入维度、模型深度、模型架构、数据量等多种变量的共同影响，其中许多变量难以分离或用分析方式表达。在这项工作中，我们通过使用嵌入有效秩（RankMe）作为统一指标，系统地研究了通用音频表示的缩放定律，该指标封装了各种变量对表示质量的影响。RankMe实现了对音频嵌入的无标签、信息论量化，使我们能够检查包括模型大小、训练数据量、计算预算、架构配置等在内的广泛超参数空间中的缩放行为。我们的实证发现揭示了RankMe与表示质量之间的一致的幂律关系，表明嵌入有效秩可作为评估和预测音频表示学习中模型性能的可靠代理。这项工作不仅验证了经典缩放原理适用于通用音频领域，还为音频基础模型的未来模型缩放策略提供了理论依据和经验上稳健的框架。


### 论文摘要

Scaling laws have profoundly shaped our understanding of model performance in computer vision and natural language processing, yet their application to general audio representation learning remains underexplored. A key challenge lies in the multifactorial nature of general audio representation-representation quality is jointly influenced by variables such as audio length, embedding dimensionality, model depth, model architecture, data volume, etc., many of which are difficult to isolate or express analytically. In this work, we present a systematic study of scaling laws for general audio representations by utilizing embedding effective rank (RankMe) as a unifying metric that encapsulates the impact of diverse variables on representation quality. RankMe enables a label-free, information-theoretic quantification of audio embeddings, allowing us to examine scaling behaviors across a wide hyper-parameter space, including model size, training data volume, computational budget, architectural configurations, etc. Our empirical findings reveal a consistent power-law relationship between RankMe and representation quality, suggesting that embedding effective rank serves as a reliable proxy for assessing and predicting model performance in audio representation learning. This work not only validates the applicability of classical scaling principles to the general audio domain but also offers a theoretically grounded and empirically robust framework for guiding future model scaling strategies in audio foundation models.

---

## 33. GapDNER: A Gap-Aware Grid Tagging Model for Discontinuous Named Entity Recognition

**论文链接:** [http://arxiv.org/abs/2510.10927v1](http://arxiv.org/abs/2510.10927v1)

**作者:** Yawen Yang, Fukun Ma, Shiao Meng, Aiwei Liu, Lijie Wen

**发布时间:** 2025-10-13

**备注:** Accepted by IJCNN 2025

### GPT解析

### 总结

该论文提出了一种名为GapDNER的新型模型，用于生物医学领域中的不连续命名实体识别，通过关注实体片段间的上下文间隙解决了传统方法中的错误传播和解码歧义问题。

### 背景

在生物医学领域，一个命名实体可能由一系列不相邻的标记组成，并与其他实体重叠。先前的方法通过连接实体片段或内部标记来识别不连续实体，但由于跨度或单词组合的多样性，面临着错误传播和解码歧义的挑战。

### 目的

为了解决这些问题，作者深入探索不连续实体的结构，并提出一种有效的间隙感知网格标记模型（GapDNER）来提升不连续命名实体识别的性能。

### 方法

GapDNER创新性地在实体片段间的上下文间隙上应用表示学习，将上下文间隙视为额外跨度类型，将跨度分类转换为标记对网格标记任务。设计了两个交互组件：内部跨度规则提取模块使用双仿射机制和线性注意力捕获每个跨度的内部规则；跨跨度关系增强模块利用交叉交叉注意力获取不同跨度间的语义关系。在推理阶段，为每个实体片段和上下文间隙分配有向边，使用BFS算法搜索有效路径。

### 主要发现

在三个数据集上的实验结果表明，GapDNER在不连续NER方面取得了新的最先进性能，并且在识别复杂实体结构方面表现出显著优势。

### 结论

GapDNER模型通过创新地处理上下文间隙和设计专门的组件来建模实体间关系，有效解决了不连续命名实体识别中的挑战，显著提高了性能。

### 翻译

在生物医学领域，一个命名实体可能由一系列不相邻的标记组成，并与其他实体重叠。先前的方法通过连接实体片段或内部标记来识别不连续实体，但由于跨度或单词组合的多样性，面临着错误传播和解码歧义的挑战。为了解决这些问题，我们深入探索了不连续实体的结构，并提出了一种有效的间隙感知网格标记模型用于不连续命名实体识别，称为GapDNER。我们的GapDNER创新性地在实体片段之间的上下文间隙上应用表示学习，以解决解码歧义并增强不连续NER性能。具体来说，我们将上下文间隙视为额外的跨度类型，并将跨度分类转换为标记对网格标记任务。随后，我们设计了两个交互组件，从内部和跨跨度两个角度全面建模标记对网格特征。内部跨度规则提取模块使用双仿射机制和线性注意力来捕获每个跨度的内部规则，而跨跨度关系增强模块则利用交叉交叉注意力来获取不同跨度之间的语义关系。在实体解码的推理阶段，我们为每个实体片段和上下文间隙分配有向边，然后使用BFS算法搜索网格中从头部到尾部的所有带有实体标签的有效路径。在三个数据集上的实验结果表明，我们的GapDNER在不连续NER方面取得了新的最先进性能，并且在识别复杂实体结构方面表现出显著优势。


### 论文摘要

In biomedical fields, one named entity may consist of a series of non-adjacent tokens and overlap with other entities. Previous methods recognize discontinuous entities by connecting entity fragments or internal tokens, which face challenges of error propagation and decoding ambiguity due to the wide variety of span or word combinations. To address these issues, we deeply explore discontinuous entity structures and propose an effective Gap-aware grid tagging model for Discontinuous Named Entity Recognition, named GapDNER. Our GapDNER innovatively applies representation learning on the context gaps between entity fragments to resolve decoding ambiguity and enhance discontinuous NER performance. Specifically, we treat the context gap as an additional type of span and convert span classification into a token-pair grid tagging task. Subsequently, we design two interactive components to comprehensively model token-pair grid features from both intra- and inter-span perspectives. The intra-span regularity extraction module employs the biaffine mechanism along with linear attention to capture the internal regularity of each span, while the inter-span relation enhancement module utilizes criss-cross attention to obtain semantic relations among different spans. At the inference stage of entity decoding, we assign a directed edge to each entity fragment and context gap, then use the BFS algorithm to search for all valid paths from the head to tail of grids with entity tags. Experimental results on three datasets demonstrate that our GapDNER achieves new state-of-the-art performance on discontinuous NER and exhibits remarkable advantages in recognizing complex entity structures.

---

## 34. Topological Alignment of Shared Vision-Language Embedding Space

**论文链接:** [http://arxiv.org/abs/2510.10889v1](http://arxiv.org/abs/2510.10889v1)

**作者:** Junwon You, Dasol Kang, Jae-Hun Jung

**发布时间:** 2025-10-13

**备注:** 24 pages, 5 figures, 19 tables

### GPT解析

### 总结

该研究提出了一种名为ToMCLIP的拓扑感知框架，用于改进多语言视觉语言模型的跨模态对齐，通过拓扑保持约束增强嵌入空间的结构连贯性，提高零样本准确率和多语言检索性能。

### 背景

对比视觉语言模型(VLMs)展示了强大的零样本能力，但由于有限的多语言多模态数据，其跨模态对齐偏向英语。现有多语言扩展虽缓解了这一问题，但仅关注实例级对齐，忽略了共享嵌入空间的几何结构。

### 目的

解决多语言视觉语言模型中嵌入空间的全局几何结构对齐问题，通过拓扑感知方法提升多语言表示的结构连贯性和性能。

### 方法

提出ToMCLIP框架，应用持续同调定义拓扑对齐损失，并利用图稀疏化策略近似持久图，确保在理论误差边界内实现拓扑保持的嵌入空间对齐。

### 主要发现

ToMCLIP增强了多语言表示的结构连贯性，在CIFAR-100上提高了零样本准确率，在xFlickr&CO上增强了多语言检索性能。

### 结论

所提出的拓扑对齐方法不仅适用于视觉语言模型，还为表示学习中融入拓扑对齐提供了通用方法。

### 翻译

对比视觉语言模型(VLMs)已经展示了强大的零样本能力。然而，由于有限的多语言多模态数据，它们的跨模态对齐仍然偏向英语。最近的多语言扩展缓解了这一差距，但强制执行实例级对齐，同时忽略了共享嵌入空间的几何结构。我们通过引入ToMCLIP(多语言CLIP的拓扑对齐)，一种拓扑感知的框架，使用拓扑保持约束对齐嵌入空间，来解决这一问题。所提出的方法应用持续同调来定义拓扑对齐损失，并使用图稀疏化策略近似持久图，具有理论误差边界。这项工作验证了所提出的方法，展示了多语言表示增强的结构连贯性，在CIFAR-100上更高的零样本准确率，以及在xFlickr&CO上更强的多语言检索性能。除了VLMs，所提出的方法为在表示学习中融入拓扑对齐提供了通用方法。


### 论文摘要

Contrastive Vision-Language Models (VLMs) have demonstrated strong zero-shot capabilities. However, their cross-modal alignment remains biased toward English due to limited multilingual multimodal data. Recent multilingual extensions have alleviated this gap but enforce instance-level alignment while neglecting the global geometry of the shared embedding space. We address this problem by introducing ToMCLIP (Topological Alignment for Multilingual CLIP), a topology-aware framework aligning embedding spaces with topology-preserving constraints. The proposed method applies persistent homology to define a topological alignment loss and approximates persistence diagram with theoretical error bounds using graph sparsification strategy. This work validates the proposed approach, showing enhanced structural coherence of multilingual representations, higher zero-shot accuracy on the CIFAR-100, and stronger multilingual retrieval performance on the xFlickr&CO. Beyond VLMs, the proposed approach provides a general method for incorporating topological alignment into representation learning.

---

## 35. UniCoD: Enhancing Robot Policy via Unified Continuous and Discrete Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.10642v1](http://arxiv.org/abs/2510.10642v1)

**作者:** Jianke Zhang, Yucheng Hu, Yanjiang Guo, Xiaoyu Chen, Yichen Liu, Wenna Chen, Chaochao Lu, Jianyu Chen

**发布时间:** 2025-10-12

### GPT解析

### 总结

本文提出了UniCoD模型，结合了理解、规划和连续未来表示学习的优势，通过大规模预训练和微调，显著提升了机器人策略学习在多样化任务中的表现。

### 背景

构建能够在开放环境中处理多样化任务的全能机器人策略是机器人领域的核心挑战。现有方法通常基于视觉语言理解模型或生成模型，但语义理解和视觉动力学建模对具身机器人都至关重要。

### 目的

利用最近出现的统一生成和理解模型的优势，结合理解、规划和连续未来表示学习，提升机器人策略学习的效果。

### 方法

提出UniCoD模型，通过在超过100万个互联网规模的 instructional manipulation 视频上进行预训练，获得动态建模高维视觉特征的能力，然后在机器人具身收集的数据上进行微调，学习从预测表征到动作令牌的映射。

### 主要发现

大量实验表明，该方法在模拟环境和真实世界分布外任务中，分别比基线方法高出9%和12%的性能。

### 结论

UniCoD通过结合理解、规划和连续未来表示学习的优势，在机器人策略学习方面取得了显著的性能提升，为构建全能机器人策略提供了新思路。

### 翻译

构建能够在开放环境中处理多样化任务的全能机器人策略是机器人领域的中心挑战。为了利用大规模预训练的知识，先前的工作通常在视觉语言理解模型(VLMs)或生成模型的基础上构建全能策略。然而，来自视觉语言预训练的语义理解和来自视觉生成预训练的视觉动力学建模对具身机器人都至关重要。最近的统一生成和理解模型通过大规模预训练在理解和生成方面都展示了强大的能力。我们认为机器人策略学习同样可以从理解、规划和连续未来表示学习的综合优势中受益。基于这一见解，我们引入了UniCoD，它通过在超过100万个互联网规模的 instructional manipulation 视频上进行预训练，获得动态建模高维视觉特征的能力。随后，UniCoD在从机器人具身收集的数据上进行微调，实现了从预测表征到动作令牌的映射学习。大量实验表明，我们的方法在模拟环境和真实世界分布外任务中，始终比基线方法高出9%和12%的性能。


### 论文摘要

Building generalist robot policies that can handle diverse tasks in open-ended environments is a central challenge in robotics. To leverage knowledge from large-scale pretraining, prior work has typically built generalist policies either on top of vision-language understanding models (VLMs) or generative models. However, both semantic understanding from vision-language pretraining and visual dynamics modeling from visual-generation pretraining are crucial for embodied robots. Recent unified models of generation and understanding have demonstrated strong capabilities in both comprehension and generation through large-scale pretraining. We posit that robotic policy learning can likewise benefit from the combined strengths of understanding, planning and continuous future representation learning. Building on this insight, we introduce UniCoD, which acquires the ability to dynamically model high-dimensional visual features through pretraining on over 1M internet-scale instructional manipulation videos. Subsequently, UniCoD is fine-tuned on data collected from the robot embodiment, enabling the learning of mappings from predictive representations to action tokens. Extensive experiments show our approach consistently outperforms baseline methods in terms of 9\% and 12\% across simulation environments and real-world out-of-distribution tasks.

---

## 36. FusionGen: Feature Fusion-Based Few-Shot EEG Data Generation

**论文链接:** [http://arxiv.org/abs/2510.10604v1](http://arxiv.org/abs/2510.10604v1)

**作者:** Yuheng Chen, Dingkun Liu, Xinyao Yang, Xinping Xu, Baicheng Chen, Dongrui Wu

**发布时间:** 2025-10-12

### GPT解析

### 总结

本文提出了一种名为FusionGen的新型EEG数据生成框架，通过解耦表征学习和特征融合技术解决脑机接口领域的数据稀缺和受试者间变异性问题，显著提高了EEG解码模型的分类准确性。

### 背景

脑机接口(BCIs)通过脑电图(EEG)在大脑和外部设备间建立直接通信，应用范围从医疗康复到认知状态评估。然而，基于EEG的BCI受到数据稀缺和显著受试者间变异性的严重限制，阻碍了EEG解码模型在实际环境中的泛化和应用。

### 目的

解决EEG数据稀缺和受试者间变异性问题，提高EEG解码模型在实际环境中的泛化能力和适用性。

### 方法

提出FusionGen框架，基于解耦表征学习和特征融合技术。通过特征匹配融合模块整合跨试验特征，并结合轻量级特征提取和重建管道，确保在有限数据条件下的数据多样性和可训练性。

### 主要发现

在多个公开EEG数据集上的实验表明，FusionGen显著优于现有增强技术，在分类准确性方面取得了明显改进。

### 结论

FusionGen有效解决了BCI领域的数据稀缺和受试者间变异性挑战，是一种有前景的EEG数据生成方法。

### 翻译

脑机接口(BCIs)通过脑电图(EEG)在大脑和外部设备之间建立直接通信途径，其应用范围从医疗康复到认知状态评估。然而，基于EEG的BCI受到数据稀缺和显著受试者间变异性的严重限制，这阻碍了EEG解码模型在实际环境中的泛化能力和适用性。为应对这些挑战，我们提出了FusionGen，一种基于解耦表征学习和特征融合的新型EEG数据生成框架。通过特征匹配融合模块整合跨试验特征，并与轻量级特征提取和重建管道相结合，FusionGen确保了在有限数据条件下的数据多样性和可训练性。在多个公开可用的EEG数据集上进行的大量实验表明，FusionGen显著优于现有的增强技术，在分类准确性方面取得了显著改进。


### 论文摘要

Brain-computer interfaces (BCIs) provide potential for applications ranging from medical rehabilitation to cognitive state assessment by establishing direct communication pathways between the brain and external devices via electroencephalography (EEG). However, EEG-based BCIs are severely constrained by data scarcity and significant inter-subject variability, which hinder the generalization and applicability of EEG decoding models in practical settings. To address these challenges, we propose FusionGen, a novel EEG data generation framework based on disentangled representation learning and feature fusion. By integrating features across trials through a feature matching fusion module and combining them with a lightweight feature extraction and reconstruction pipeline, FusionGen ensures both data diversity and trainability under limited data constraints. Extensive experiments on multiple publicly available EEG datasets demonstrate that FusionGen significantly outperforms existing augmentation techniques, yielding notable improvements in classification accuracy.

---

## 37. Understanding Self-supervised Contrastive Learning through Supervised Objectives

**论文链接:** [http://arxiv.org/abs/2510.10572v1](http://arxiv.org/abs/2510.10572v1)

**作者:** Byeongchan Lee

**发布时间:** 2025-10-12

**备注:** Accepted at TMLR 2025

### GPT解析

### 总结

本文提供了一种理论视角，将自监督表示学习表述为监督表示学习目标的近似，推导出与流行对比损失相关的损失函数，并引入原型表示偏差和平衡对比损失的概念，以解释和改进自监督学习算法的行为。

### 背景

自监督表示学习已经取得了显著的实证成功，但其理论理解仍然有限。

### 目的

提供理论视角，将自监督表示学习表述为监督表示学习目标的近似，深入理解对比损失函数的原理。

### 方法

基于自监督表示学习作为监督表示学习目标近似的表述，推导损失函数，引入原型表示偏差和平衡对比损失的概念，并对应到对比学习的既定实践。

### 主要发现

原型表示偏差和平衡对比损失的概念有助于解释和改进自监督学习算法的行为，理论框架的组成部分对应于对比学习中的既定实践，平衡正负样本对的交互具有实证效果。

### 结论

通过理论推导和实证验证，本文提供了自监督表示学习的理论框架，所有理论证明在附录中提供，代码包含在补充材料中。

### 翻译

自监督表示学习已经取得了令人印象深刻的实证成功，但其理论理解仍然有限。在这项工作中，我们通过将自监督表示学习表述为监督表示学习目标的近似，提供了一种理论视角。基于这一表述，我们推导出一个与流行的对比损失（如InfoNCE）密切相关的损失函数，揭示了它们的基本原理。我们的推导自然地引入了原型表示偏差和平衡对比损失的概念，这些概念有助于解释和改进自监督学习算法的行为。我们进一步展示了理论框架的组成部分如何对应于对比学习中的既定实践。最后，我们通过实证验证了平衡正负样本对交互的效果。所有理论证明都在附录中提供，我们的代码包含在补充材料中。


### 论文摘要

Self-supervised representation learning has achieved impressive empirical success, yet its theoretical understanding remains limited. In this work, we provide a theoretical perspective by formulating self-supervised representation learning as an approximation to supervised representation learning objectives. Based on this formulation, we derive a loss function closely related to popular contrastive losses such as InfoNCE, offering insight into their underlying principles. Our derivation naturally introduces the concepts of prototype representation bias and a balanced contrastive loss, which help explain and improve the behavior of self-supervised learning algorithms. We further show how components of our theoretical framework correspond to established practices in contrastive learning. Finally, we empirically validate the effect of balancing positive and negative pair interactions. All theoretical proofs are provided in the appendix, and our code is included in the supplementary material.

---

## 38. Self-Supervised Representation Learning with ID-Content Modality Alignment for Sequential Recommendation

**论文链接:** [http://arxiv.org/abs/2510.10556v1](http://arxiv.org/abs/2510.10556v1)

**作者:** Donglin Zhou, Weike Pan, Zhong Ming

**发布时间:** 2025-10-12

### GPT解析

### 总结

SICSRec是一种创新的顺序推荐模型，通过自监督表示学习和ID-Content模态对齐解决了内容顺序推荐中的三个关键挑战，在有限交互历史情况下表现优异。

### 背景

顺序推荐模型通常基于用户历史交互的物品ID捕捉用户偏好，但在交互历史有限时表现不佳。基于内容的顺序推荐利用物品的文本和视觉特征增强偏好学习，但仍面临语义差距、偏好联合建模和表示对齐等挑战。

### 目的

解决内容顺序推荐中的三个关键挑战：减少不同内容模态表示间的语义差距；联合建模用户行为偏好和内容偏好；设计有效训练策略对齐ID表示和内容表示。

### 方法

提出SICSRec模型，包含：基于LLM的样本构建方法和监督微调方法对齐物品级模态表示；基于Transformer的顺序模型，包括ID模态序列编码器、内容模态序列编码器和混合模态序列解码器；两步训练策略结合内容感知对比学习任务对齐模态表示和ID表示。

### 主要发现

在四个公共视频流数据集上，SICSRec在NDCG@5上平均比最先进的ID模态顺序推荐器高出8.04%，在NDCG@10上平均高出6.62%。

### 结论

SICSRec通过有效对齐ID表示和内容表示，成功解决了内容顺序推荐中的关键挑战，在有限交互历史情况下提供了更优的推荐性能。

### 翻译

顺序推荐(SR)模型通常基于历史交互的物品ID来捕捉用户偏好，当交互历史有限时通常表现不佳。基于内容的顺序推荐最近已成为一种有前途的方向，利用物品的文本和视觉特征来增强偏好学习。然而，仍存在三个关键挑战：(i)如何减少不同内容模态表示之间的语义差距；(ii)如何联合建模用户行为偏好和内容偏好；(iii)如何设计有效的训练策略来对齐ID表示和内容表示。为应对这些挑战，我们提出了一种新模型，即带有ID-Content模态对齐的自监督表示学习，名为SICSRec。首先，我们提出了一种基于LLM的样本构建方法，并开发了监督微调方法来对齐物品级模态表示。其次，我们设计了一种新颖的基于Transformer的顺序模型，其中ID模态序列编码器捕捉用户行为偏好，内容模态序列编码器学习用户内容偏好，混合模态序列解码器把握这两种偏好之间的内在关系。第三，我们提出了一个包含内容感知对比学习任务的两步训练策略，用于对齐模态表示和ID表示，从而解耦内容模态依赖和物品协同依赖的训练过程。在四个公共视频流数据集上进行的广泛实验表明，我们的SICSRec在NDCG@5上平均比最先进的ID模态顺序推荐器高出8.04%，在NDCG@10上平均高出6.62%。


### 论文摘要

Sequential recommendation (SR) models often capture user preferences based on the historically interacted item IDs, which usually obtain sub-optimal performance when the interaction history is limited. Content-based sequential recommendation has recently emerged as a promising direction that exploits items' textual and visual features to enhance preference learning. However, there are still three key challenges: (i) how to reduce the semantic gap between different content modality representations; (ii) how to jointly model user behavior preferences and content preferences; and (iii) how to design an effective training strategy to align ID representations and content representations. To address these challenges, we propose a novel model, self-supervised representation learning with ID-Content modality alignment, named SICSRec. Firstly, we propose a LLM-driven sample construction method and develop a supervised fine-tuning approach to align item-level modality representations. Secondly, we design a novel Transformer-based sequential model, where an ID-modality sequence encoder captures user behavior preferences, a content-modality sequence encoder learns user content preferences, and a mix-modality sequence decoder grasps the intrinsic relationship between these two types of preferences. Thirdly, we propose a two-step training strategy with a content-aware contrastive learning task to align modality representations and ID representations, which decouples the training process of content modality dependency and item collaborative dependency. Extensive experiments conducted on four public video streaming datasets demonstrate our SICSRec outperforms the state-of-the-art ID-modality sequential recommenders and content-modality sequential recommenders by 8.04% on NDCG@5 and 6.62% on NDCD@10 on average, respectively.

---

## 39. Unified Open-World Segmentation with Multi-Modal Prompts

**论文链接:** [http://arxiv.org/abs/2510.10524v1](http://arxiv.org/abs/2510.10524v1)

**作者:** Yang Liu, Yufei Yin, Chenchen Jing, Muzhi Zhu, Hao Chen, Yuling Xi, Bo Feng, Hao Wang, Shiyu Li, Chunhua Shen

**发布时间:** 2025-10-12

**备注:** Accepted to ICCV2025

### GPT解析

### 总结

本文提出了COSINE，一个统一的开放世界分割模型，整合了开放词汇分割和上下文分割功能，支持多模态提示。

### 背景

现有的开放词汇分割和上下文分割方法存在架构差异、不同的学习目标和表示学习策略问题。

### 目的

开发一个统一的模型来解决开放词汇分割和上下文分割的架构和策略不一致问题。

### 方法

COSINE利用基础模型提取图像和多模态提示的表示，并通过SegDecoder对齐这些表示、建模交互，生成不同粒度的掩码。

### 主要发现

COSINE在开放词汇和上下文分割任务中表现出显著的性能提升，且多模态提示的协同合作相比单模态方法提高了泛化能力。

### 结论

COSINE通过统一架构和策略，成功解决了现有开放世界分割方法的局限性，实现了更强大的分割性能。

### 翻译

在这项工作中，我们提出了COSINE，一个统一的开放世界分割模型，它整合了开放词汇分割和上下文分割功能，并支持多模态提示（例如文本和图像）。COSINE利用基础模型提取输入图像和对应多模态提示的表示，并通过SegDecoder对齐这些表示、建模它们的交互，并获得不同粒度下由输入提示指定的掩码。这样，COSINE克服了先前开放词汇分割和上下文分割管道的架构差异、不同的学习目标和表示学习策略。全面的实验证明COSINE在开放词汇和上下文分割任务中都有显著的性能提升。我们的探索性分析强调，使用视觉和文本提示之间的协同合作相比单模态方法显著提高了泛化能力。


### 论文摘要

In this work, we present COSINE, a unified open-world segmentation model that consolidates open-vocabulary segmentation and in-context segmentation with multi-modal prompts (e.g., text and image). COSINE exploits foundation models to extract representations for an input image and corresponding multi-modal prompts, and a SegDecoder to align these representations, model their interaction, and obtain masks specified by input prompts across different granularities. In this way, COSINE overcomes architectural discrepancies, divergent learning objectives, and distinct representation learning strategies of previous pipelines for open-vocabulary segmentation and in-context segmentation. Comprehensive experiments demonstrate that COSINE has significant performance improvements in both open-vocabulary and in-context segmentation tasks. Our exploratory analyses highlight that the synergistic collaboration between using visual and textual prompts leads to significantly improved generalization over single-modality approaches.

---

## 40. Mesh-Gait: A Unified Framework for Gait Recognition Through Multi-Modal Representation Learning from 2D Silhouettes

**论文链接:** [http://arxiv.org/abs/2510.10406v1](http://arxiv.org/abs/2510.10406v1)

**作者:** Zhao-Yang Wang, Jieneng Chen, Jiang Liu, Yuxiang Guo, Rama Chellappa

**发布时间:** 2025-10-12

### GPT解析

### 总结

Mesh-Gait是一种创新的端到端多模态步态识别框架，通过从2D剪影重建3D热图作为中间表示，有效结合了2D和3D表示的优势，在保持计算效率的同时实现了最先进的识别准确性。

### 背景

步态识别是一种利用独特行走模式进行个人识别的生物识别技术，传统方法使用2D表示如剪影或骨架，但在视角变化、遮挡和噪声方面存在困难。结合3D身体形状信息的多模态方法虽能提高鲁棒性，但计算成本高，限制了实时应用的可能性。

### 目的

开发一种能够有效解决现有步态识别方法局限性，同时结合2D和3D优势的步态识别框架，提高识别准确性和计算效率。

### 方法

Mesh-Gait直接从2D剪影重建3D表示，使用3D热图作为中间表示，在训练过程中逐步重建并提高准确性，通过计算重建的3D关节、虚拟标记和3D网格与真实值之间的损失来确保精确对齐。该方法从剪影和重建的3D热图中提取判别性特征，使网络专注于运动动力学而非无关视觉细节。

### 主要发现

Mesh-Gait能够以计算高效的方式捕获空间和结构步态特征，避免了从RGB视频直接进行3D重建的巨大开销，使网络能够专注于运动动力学而非无关的视觉细节。

### 结论

大量实验证明Mesh-Gait达到了最先进的准确性，代码将在论文接受后发布。

### 翻译

步态识别是一种基础的生物识别技术，利用独特的行走模式进行个人识别，通常使用剪影或骨架等二维表示。然而，这些方法往往难以处理视角变化、遮挡和噪声问题。结合三维身体形状信息的多模态方法虽能提高鲁棒性，但计算成本高，限制了其在实时应用中的可行性。为解决这些挑战，我们引入了Mesh-Gait，一种新颖的端到端多模态步态识别框架，直接从二维剪影重建三维表示，有效结合了两种模态的优势。与现有方法相比，直接从三维关节或网格学习三维特征复杂且难以与基于剪影的步态特征融合。为克服这一问题，Mesh-Gait将三维热图重建为中间表示，使模型能够有效捕获三维几何信息，同时保持简单性和计算效率。在训练过程中，中间的三维热图被逐步重建，并在监督学习下变得越来越准确，通过计算重建的三维关节、虚拟标记和三维网格与其对应真实值之间的损失，确保精确的空间对齐和一致的三维结构。Mesh-Gait以计算高效的方式从剪影和重建的三维热图中提取判别性特征。这种设计使模型能够捕获空间和结构步态特征，同时避免了从RGB视频直接进行三维重建的巨大开销，使网络能够专注于运动动力学而非无关的视觉细节。大量实验证明Mesh-Gait达到了最先进的准确性。代码将在论文接受后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决步态识别(gait recognition)中传统2D方法在视角变化、遮挡和环境噪声方面表现不佳，以及多模态3D方法计算成本高、难以实时应用的问题。这个问题在现实中很重要，因为步态识别是一种非接触式生物识别技术，可在远距离识别个人，适用于监控、安全认证和法医分析等场景，但现有方法难以在实际复杂环境中保持高准确率和实时性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了2D方法的局限性和3D方法的计算成本问题，然后提出使用3D热图作为中间表示来桥接2D和3D信息。他们设计了一个双分支架构：一个处理2D轮廓特征，另一个处理重建的3D特征。在训练过程中使用监督学习逐步优化3D热图。该方法借鉴了HRNet作为3D估计器，受到虚拟标记概念的启发，并参考了现有的步态识别方法如GaitSet、GaitPart等，同时使用了多种损失函数的组合进行训练。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过3D热图作为中间表示，直接从2D轮廓重建3D信息，结合2D和3D特征进行多模态步态识别，并在推理阶段避免网格重建以提高效率。整体流程：1)输入2D轮廓序列；2)双分支处理：2D分支提取轮廓特征，3D分支使用HRNet估计3D热图；3)从热图重建3D关节、虚拟标记和网格；4)分别从2D轮廓和3D热图提取特征；5)拼接融合2D和3D特征；6)通过时间池化和金字塔池化处理；7)使用全连接层计算嵌入并进行识别；8)应用多种损失函数进行训练优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)统一框架直接从2D轮廓重建3D表示，消除复杂多视角需求；2)使用3D热图作为中间表示，便于特征提取和融合；3)监督学习渐进式优化3D热图，确保精确重建；4)推理阶段不需网格重建，计算效率高(比传统方法快72倍)。相比之前工作的不同：传统2D方法难以处理视角变化和遮挡；现有多模态方法需要额外3D重建模型且计算成本高；直接3D特征学习方法处理点云复杂且难以与2D特征融合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Mesh-Gait提出了一种创新的多模态步态识别框架，通过直接从2D轮廓重建3D热图表示，结合了2D和3D信息的优势，显著提高了识别准确率和鲁棒性，同时大幅降低了计算成本，使实时步态识别成为可能。'}


### 论文摘要

Gait recognition, a fundamental biometric technology, leverages unique walking patterns for individual identification, typically using 2D representations such as silhouettes or skeletons. However, these methods often struggle with viewpoint variations, occlusions, and noise. Multi-modal approaches that incorporate 3D body shape information offer improved robustness but are computationally expensive, limiting their feasibility for real-time applications. To address these challenges, we introduce Mesh-Gait, a novel end-to-end multi-modal gait recognition framework that directly reconstructs 3D representations from 2D silhouettes, effectively combining the strengths of both modalities. Compared to existing methods, directly learning 3D features from 3D joints or meshes is complex and difficult to fuse with silhouette-based gait features. To overcome this, Mesh-Gait reconstructs 3D heatmaps as an intermediate representation, enabling the model to effectively capture 3D geometric information while maintaining simplicity and computational efficiency. During training, the intermediate 3D heatmaps are gradually reconstructed and become increasingly accurate under supervised learning, where the loss is calculated between the reconstructed 3D joints, virtual markers, and 3D meshes and their corresponding ground truth, ensuring precise spatial alignment and consistent 3D structure. Mesh-Gait extracts discriminative features from both silhouettes and reconstructed 3D heatmaps in a computationally efficient manner. This design enables the model to capture spatial and structural gait characteristics while avoiding the heavy overhead of direct 3D reconstruction from RGB videos, allowing the network to focus on motion dynamics rather than irrelevant visual details. Extensive experiments demonstrate that Mesh-Gait achieves state-of-the-art accuracy. The code will be released upon acceptance of the paper.

---

## 41. Text2Token: Unsupervised Text Representation Learning with Token Target Prediction

**论文链接:** [http://arxiv.org/abs/2510.10224v1](http://arxiv.org/abs/2510.10224v1)

**作者:** Ruize An, Richong Zhang, Zhijie Nie, Zhanyu Wu, Yanzhao Zhang, Dingkun Long

**发布时间:** 2025-10-11

### GPT解析

### 总结

本文提出了一种名为Text2Token的无监督文本表示学习框架，通过token目标预测任务构建高质量的目标token分布，在MTEB v2基准测试上取得了与最先进方法相媲美的性能。

### 背景

无监督文本表示学习是自然语言处理的基础任务，对利用网络未标记文本改进搜索和推荐系统至关重要。研究表明，高质量的文本表示与输入文本的关键词对齐，揭示了表示空间和词汇空间之间的潜在联系。

### 目的

开发一个无监督生成框架Text2Token，用于文本表示学习，探索表示空间和词汇空间之间的联系，并通过token目标预测任务提升表示学习性能。

### 方法

Text2Token框架基于token目标预测任务，利用精心构建的目标token分布作为监督信号。作者确定了两种关键token类别：文本中有意义的token和文本外语义派生的token，并提出了数据驱动和模型派生两种方法来构建合成token目标。

### 主要发现

在MTEB v2基准测试上，Text2Token的性能与最先进的无监督对比学习方法LLM2Vec具有竞争力。词汇和表示空间在训练过程中共同优化并趋向最优解。

### 结论

Text2Token框架成功探索了表示空间和词汇空间之间的联系，为无监督文本表示学习提供了新思路，证明了通过精心设计的token目标预测任务可以有效提升表示学习性能。

### 翻译

无监督文本表示学习(TRL)是自然语言处理的一项基础任务，它有助于利用网络上的未标记文本改进搜索和推荐系统。最近的一项实证研究发现，高质量的表示与输入文本的关键词对齐，揭示了表示空间和词汇空间之间的潜在联系。受这一发现的启发，我们重新审视了生成任务，并开发了一个用于TRL的无监督生成框架Text2Token。该框架基于token目标预测任务，利用精心构建的目标token分布作为监督信号。为了构建高质量的目标token分布，我们分析了与高级嵌入器的token对齐特性，并确定了两种关键token类别：(1)文本中有意义的token和(2)文本外语义派生的token。基于这些见解，我们提出了两种方法——数据驱动和模型派生——从数据或LLM骨干构建合成token目标。在MTEB v2基准测试上的实验表明，Text2Token的性能与最先进的无监督对比学习方法LLM2Vec具有竞争力。我们的进一步分析表明，词汇和表示空间在训练过程中共同优化并趋向最优解，为未来工作提供了新的思路和见解。


### 论文摘要

Unsupervised text representation learning (TRL) is a fundamental task in natural language processing, which is beneficial for improving search and recommendations with the web's unlabeled texts. A recent empirical study finds that the high-quality representation aligns with the key token of the input text, uncovering the potential connection between representation space and vocabulary space. Inspired by the findings, we revisit the generative tasks and develop an unsupervised generative framework for TRL, Text2Token. The framework is based on the token target prediction task, utilizing carefully constructed target token distribution as supervisory signals. To construct the high-quality target token distribution, we analyze the token-alignment properties with advanced embedders and identify two essential categories of key tokens: (1) the meaningful tokens in the text and (2) semantically derived tokens beyond the text. Based on these insights, we propose two methods -- data-driven and model-derived -- to construct synthetic token targets from data or the LLM backbone. Experiments on the MTEB v2 benchmark demonstrate that Text2Token achieves performance competitive with the state-of-the-art embedder with unsupervised contrastive learning, LLM2Vec. Our analysis further shows that vocabulary and representation spaces optimize together and toward the optimum solution during training, providing new ideas and insights for future work.

---

## 42. PANTHER: Generative Pretraining Beyond Language for Sequential User Behavior Modeling

**论文链接:** [http://arxiv.org/abs/2510.10102v1](http://arxiv.org/abs/2510.10102v1)

**作者:** Guilin Li, Yun Zhang, Xiuyuan Chen, Chengqi Li, Bo Wang, Linghe Kong, Wenjia Wang, Weiran Huang, Matthias Hwai Yong Tan

**发布时间:** 2025-10-11

### GPT解析

### 总结

该研究提出了PANTHER，一个混合生成-判别框架，用于用户行为建模和表示学习。通过生成式预训练从无标签行为数据中学习可迁移的表示，结合结构化标记化、序列模式识别、统一用户画像嵌入和实时可扩展性等技术，在微信支付的实际应用中取得了显著效果。

### 背景

大型语言模型能够通过生成式预训练将大量世界知识压缩到紧凑的标记表示中。然而，在建模用户交互历史中的行为知识方面存在局限。用户行为形成独特模态，每个行为（由时间、上下文和交易类型等多维属性定义）构成行为标记，建模这些高基数序列具有挑战性，判别模型在监督有限时表现不佳。

### 目的

将生成式预训练扩展到用户行为领域，类似于LLMs从文本中学习的方式，从无标签行为数据中学习可迁移的表示，开发能够实现大规模序列用户表示学习和实时推理的框架。

### 方法

提出了PANTHER框架，包含四个主要组件：(1)结构化标记化：将多维交易属性压缩为可解释的词汇表；(2)序列模式识别模块(SPRM)：用于建模周期性交易模式；(3)统一用户画像嵌入：融合静态人口统计信息和动态交易历史；(4)实时可扩展性：通过预训练嵌入的离线缓存实现毫秒级推理。

### 主要发现

在微信支付上部署PANTHER后，实现了下一交易预测的HitRate@1提升25.6%，欺诈检测召回率相对提高38.6%。在公共基准测试上显示出强大泛化能力，相比transformer基线最高实现21%的HitRate@1提升。

### 结论

PANTHER被确立为一种可扩展、高性能的工业级序列用户行为建模框架，通过结合生成式预训练和判别式建模，有效解决了用户行为建模中的挑战，并在实际应用中取得了显著效果。

### 翻译

大型语言模型已经证明，生成式预训练可以将大量世界知识压缩到紧凑的标记表示中。虽然LLMs包含广泛的世界知识，但在建模用户交互历史中包含的行为知识方面仍然有限。用户行为形成一种独特的模态，其中每个行为（由时间、上下文和交易类型等多维属性定义）构成一个行为标记。建模这些高基数序列具有挑战性，判别模型在监督有限的情况下往往表现不佳。为了填补这一空白，我们将生成式预训练扩展到用户行为，类似于LLMs从文本中学习的方式，从无标签行为数据中学习可迁移的表示。我们提出了PANTHER，一个混合生成-判别框架，统一了用户行为预训练和下游适应，实现了大规模序列用户表示学习和实时推理。PANTHER引入了：(1)结构化标记化，将多维交易属性压缩为可解释的词汇表；(2)序列模式识别模块(SPRM)，用于建模周期性交易模式；(3)统一用户画像嵌入，融合静态人口统计信息和动态交易历史；(4)实时可扩展性，通过预训练嵌入的离线缓存实现毫秒级推理。在微信支付上全面部署和在线运行后，PANTHER相比基线实现了下一交易预测HitRate@1提升25.6%，欺诈检测召回率相对提高38.6%。在公共基准上的跨领域评估显示出强大的泛化能力，相比transformer基线最高实现21%的HitRate@1提升，确立了PANTHER作为工业级序列用户行为建模的可扩展、高性能框架。


### 论文摘要

Large language models (LLMs) have shown that generative pretraining can distill vast world knowledge into compact token representations. While LLMs encapsulate extensive world knowledge, they remain limited in modeling the behavioral knowledge contained within user interaction histories. User behavior forms a distinct modality, where each action, defined by multi-dimensional attributes such as time, context, and transaction type, constitutes a behavioral token. Modeling these high-cardinality sequences is challenging, and discriminative models often falter under limited supervision. To bridge this gap, we extend generative pretraining to user behavior, learning transferable representations from unlabeled behavioral data analogous to how LLMs learn from text. We present PANTHER, a hybrid generative-discriminative framework that unifies user behavior pretraining and downstream adaptation, enabling large-scale sequential user representation learning and real-time inference. PANTHER introduces: (1) Structured Tokenization to compress multi-dimensional transaction attributes into an interpretable vocabulary; (2) Sequence Pattern Recognition Module (SPRM) for modeling periodic transaction motifs; (3) a Unified User-Profile Embedding that fuses static demographics with dynamic transaction histories; and (4) Real-time scalability enabled by offline caching of pretrained embeddings for millisecond-level inference. Fully deployed and operational online at WeChat Pay, PANTHER delivers a 25.6 percent boost in next-transaction prediction HitRate@1 and a 38.6 percent relative improvement in fraud detection recall over baselines. Cross-domain evaluations on public benchmarks show strong generalization, achieving up to 21 percent HitRate@1 gains over transformer baselines, establishing PANTHER as a scalable, high-performance framework for industrial sequential user behavior modeling.

---

## 43. Cooperative Pseudo Labeling for Unsupervised Federated Classification

**论文链接:** [http://arxiv.org/abs/2510.10100v1](http://arxiv.org/abs/2510.10100v1)

**作者:** Kuangpu Guo, Lijun Sheng, Yongcan Yu, Jian Liang, Zilei Wang, Ran He

**发布时间:** 2025-10-11

**备注:** Accepted by ICCV 2025

### GPT解析

### 总结

本文提出了一种名为FedCoPL（联邦合作伪标签）的新方法，首次将无监督联邦学习(UFL)扩展到分类问题，利用CLIP模型的零样本预测能力，通过客户端上传伪标签分布和服务器重新分配来解决类别不平衡问题，并引入部分提示聚合协议促进协作和个性化。

### 背景

无监督联邦学习(UFL)旨在分布式客户端之间协作训练全局模型而不共享数据或标签信息。之前的UFL工作主要集中在表示学习和聚类任务上。视觉语言模型(如CLIP)因其强大的零样本预测能力而受到广泛关注，使UFL范式下的分类问题成为可能，但这一领域仍 largely未被探索。

### 目的

将UFL扩展到分类问题，利用CLIP模型解决UFL中的分类挑战，并提出一种新的联邦学习方法来有效处理此类问题。

### 方法

提出了FedCoPL方法：客户端估计并上传伪标签分布，服务器调整并重新分配以避免类别不平衡；引入部分提示聚合协议实现有效协作和个性化，其中视觉提示在服务器端聚合，文本提示保留在本地。

### 主要发现

大量实验证明，FedCoPL与基线方法相比具有优越性能，成功解决了UFL范式下的分类问题。

### 结论

FedCoPL成功将UFL扩展到分类问题，通过伪标签分布的协作和部分提示聚合，实现了有效的联邦学习，为UFL范式下的分类问题提供了新解决方案。

### 翻译

无监督联邦学习(UFL)旨在分布式客户端之间协作训练全局模型，而不共享数据或访问标签信息。之前的UFL工作主要集中在表示学习和聚类任务上。最近，视觉语言模型（如CLIP）因其强大的零样本预测能力而受到广泛关注。利用这一进展，之前在UFL范式下被认为不可行的分类问题现在呈现出有希望的新机会，但仍然 largely未被探索。在本文中，我们首次将UFL扩展到使用CLIP的分类问题，并提出了一种新方法，联邦合作伪标签(FedCoPL)。具体来说，客户端估计并上传其伪标签分布，服务器调整并重新分配它们以避免类别之间的全局不平衡。此外，我们引入了部分提示聚合协议以实现有效的协作和个性化。特别是，包含通用图像特征的视觉提示在服务器端聚合，而编码个性化知识的文本提示则保留在本地。大量实验证明了我们的FedCoPL与基线方法相比的优越性能。我们的代码可在https://github.com/krumpguo/FedCoPL获取。


### 论文摘要

Unsupervised Federated Learning (UFL) aims to collaboratively train a global model across distributed clients without sharing data or accessing label information. Previous UFL works have predominantly focused on representation learning and clustering tasks. Recently, vision language models (e.g., CLIP) have gained significant attention for their powerful zero-shot prediction capabilities. Leveraging this advancement, classification problems that were previously infeasible under the UFL paradigm now present promising new opportunities, yet remain largely unexplored. In this paper, we extend UFL to the classification problem with CLIP for the first time and propose a novel method, \underline{\textbf{Fed}}erated \underline{\textbf{Co}}operative \underline{\textbf{P}}seudo \underline{\textbf{L}}abeling (\textbf{FedCoPL}). Specifically, clients estimate and upload their pseudo label distribution, and the server adjusts and redistributes them to avoid global imbalance among classes. Moreover, we introduce a partial prompt aggregation protocol for effective collaboration and personalization. In particular, visual prompts containing general image features are aggregated at the server, while text prompts encoding personalized knowledge are retained locally. Extensive experiments demonstrate the superior performance of our FedCoPL compared to baseline methods. Our code is available at \href{https://github.com/krumpguo/FedCoPL}{https://github.com/krumpguo/FedCoPL}.

---

## 44. Translution: Unifying Self-attention and Convolution for Adaptive and Relative Modeling

**论文链接:** [http://arxiv.org/abs/2510.10060v1](http://arxiv.org/abs/2510.10060v1)

**作者:** Hehe Fan, Yi Yang, Mohan Kankanhalli, Fei Wu

**发布时间:** 2025-10-11

**备注:** technical report

### GPT解析

### 总结

本研究提出了一种名为Translution的新型操作，它结合了自注意力和卷积的优势，能够在计算机视觉和自然语言处理任务上实现更高的准确性。

### 背景

在建模数据时，现有方法如自注意力和卷积各有优缺点。自注意力可以自适应识别相关元素，但依赖绝对位置嵌入；卷积以相对方式编码元素，但固定的核大小限制了其自适应选择能力。

### 目的

开发一种统一自注意力自适应识别能力和卷积相对编码优势的操作，同时解决参数数量过多的问题。

### 方法

提出Translution操作，并结合其轻量级变体α-Translution，以减少参数数量，使其适合实际应用。

### 主要发现

Translution（包括α-Translution）在计算机视觉和自然语言处理任务上实现了比自注意力更高的准确性，同时保持了计算效率。

### 结论

Translution操作成功统一了自注意力和卷积的优势，为数据建模提供了新的有效方法，其轻量级变体使其能够在实际应用中部署。

### 翻译

在建模给定类型的数据时，我们认为它涉及两个关键方面：1)识别与中心元素相关的元素（如卷积感受野中的图像像素）或与查询元素相关的元素（如自注意力中的文本单词），以及2)有效编码这些标记。自注意力可以自适应地识别这些元素，但依赖于绝对位置嵌入来进行结构表示学习。相比之下，卷积以相对方式编码元素，但其固定的核大小限制了它们自适应选择相关元素的能力。在本文中，我们引入了Translution，这是一种统一了自注意力自适应识别能力和卷积相对编码优势的操作。然而，这种整合导致参数数量大幅增加，超过了大多数现有计算资源的能力。因此，我们提出了Translution的轻量级变体，称为α-Translution。在计算机视觉和自然语言处理任务上的实验表明，Translution（包括α-Translution）实现了比自注意力更高的准确性。代码可在https://github.com/hehefan/Translution获取。


### 论文摘要

When modeling a given type of data, we consider it to involve two key aspects: 1) identifying relevant elements (e.g., image pixels or textual words) to a central element, as in a convolutional receptive field, or to a query element, as in self-attention, and 2) encoding these tokens effectively. Self-attention can adaptively identify these elements but relies on absolute positional embedding for structural representation learning. In contrast, convolution encodes elements in a relative manner, yet their fixed kernel size limits their ability to adaptively select the relevant elements. In this paper, we introduce Translution, an operation that unifies the adaptive identification capability of self-attention and the relative encoding advantage of convolution. However, this integration leads to a substantial increase in the number of parameters, exceeding most currently available computational resources. Therefore, we propose a lightweight variant of Translution, named {\alpha}-Translution. Experiments on computer vision and natural language processing tasks show that Translution (including {\alpha}-Translution) achieves superior accuracy compared to self-attention. The code is available at https://github.com/hehefan/Translution.

---

## 45. Beyond AlphaEarth: Toward Human-Centered Spatial Representation via POI-Guided Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.09894v1](http://arxiv.org/abs/2510.09894v1)

**作者:** Junyuan Liu, Quan Qin, Guangsheng Dong, Xinglei Wang, Jiazhuang Feng, Zichao Zeng, Tao Cheng

**发布时间:** 2025-10-10

### GPT解析

### 总结

AETHER是一个轻量级框架，通过兴趣点(POIs)引导的多模态对齐，将AlphaEarth模型扩展到以人为中心的urban分析，在土地使用分类和社会经济映射方面取得显著改进。

### 背景

通用空间表示对构建可迁移的地理空间基础模型(GFMs)至关重要。AlphaEarthFoundation(AE)代表了地球表面全球统一表示的重要进展，但它主要编码物理和光谱模式，难以捕捉城市的功能和社会经济维度。

### 目的

提出AETHER框架，通过兴趣点(POIs)引导的多模态对齐，使AlphaEarth适应以人为中心的urban分析，丰富其物理特征与语义线索。

### 方法

AETHER将AE嵌入与POIs的文本表示对齐，用关于城市功能和社会经济语境的语义线索丰富基于物理的EO特征。它构建在预训练的AE之上，利用轻量级多模态对齐来丰富以人为中心的语义，同时保持计算效率。

### 主要发现

在大伦敦地区，AETHER相对于AE基线实现了一致的改进：土地使用分类F1相对提高7.2%，社会经济映射的Kullback-Leibler散度相对减少23.6%。

### 结论

通过将地球观测数据与以人为中心的语义相结合，AETHER推进了地理空间基础模型向通用城市表示发展，这些表示同时整合物理形态和功能意义。

### 翻译

通用空间表示对于构建可迁移的地理空间基础模型(GFMs)至关重要。其中，AlphaEarthFoundation(AE)代表了地球表面全球统一表示的重要进展，它从多源地球观测(EO)数据中学习10米嵌入，捕捉不同景观中丰富的物理和环境模式。然而，这类EO驱动的表示在捕捉城市的功能和社会经济维度方面仍然有限，因为它们主要编码物理和光谱模式，而不是人类活动或空间功能。我们提出了AETHER(AlphaEarth-POI Enriched Representation Learning)，这是一个轻量级框架，通过兴趣点(POIs)引导的多模态对齐，使AlphaEarth适应以人为中心的urban分析。AETHER将AE嵌入与POIs的文本表示对齐，用关于城市功能和社会经济语境的语义线索丰富基于物理的EO特征。在大伦敦地区，AETHER相对于AE基线实现了一致的改进，土地使用分类F1相对提高7.2%，社会经济映射的Kullback-Leibler散度相对减少23.6%。构建在预训练的AE之上，AETHER利用轻量级多模态对齐来丰富它以人为中心的语义，同时保持计算效率和对城市应用的可扩展性。通过将EO与以人为中心的语义相结合，AETHER推进了地理空间基础模型向通用城市表示发展，这些表示同时整合物理形态和功能意义。


### 论文摘要

General-purpose spatial representations are essential for building transferable geospatial foundation models (GFMs). Among them, the AlphaEarth Foundation (AE) represents a major step toward a global, unified representation of the Earth's surface, learning 10-meter embeddings from multi-source Earth Observation (EO) data that capture rich physical and environmental patterns across diverse landscapes. However, such EO-driven representations remain limited in capturing the functional and socioeconomic dimensions of cities, as they primarily encode physical and spectral patterns rather than human activities or spatial functions. We propose AETHER (AlphaEarth-POI Enriched Representation Learning), a lightweight framework that adapts AlphaEarth to human-centered urban analysis through multimodal alignment guided by Points of Interest (POIs). AETHER aligns AE embeddings with textual representations of POIs, enriching physically grounded EO features with semantic cues about urban functions and socioeconomic contexts. In Greater London, AETHER achieves consistent gains over the AE baseline, with a 7.2% relative improvement in land-use classification F1 and a 23.6% relative reduction in Kullback-Leibler divergence for socioeconomic mapping. Built upon pretrained AE, AETHER leverages a lightweight multimodal alignment to enrich it with human-centered semantics while remaining computationally efficient and scalable for urban applications. By coupling EO with human-centered semantics, it advances geospatial foundation models toward general-purpose urban representations that integrate both physical form and functional meaning.

---

## 46. TAWRMAC: A Novel Dynamic Graph Representation Learning Method

**论文链接:** [http://arxiv.org/abs/2510.09884v1](http://arxiv.org/abs/2510.09884v1)

**作者:** Soheila Farokhi, Xiaojun Qi, Hamid Karimi

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文提出了TAWRMAC框架，通过整合带重启的临时匿名游走、内存增强和邻居共现嵌入技术，解决了动态图表示学习中的三个关键挑战：嵌入过时、上下文感知不足和结构动态捕捉不充分。该方法在动态链接预测和节点分类任务中表现优异。

### 背景

动态图表示学习对于分析社交网络、推荐系统和交通分析等领域中的演化网络至关重要。

### 目的

解决现有连续时间动态图表示学习方法面临的三个关键挑战：节点嵌入过时、邻域相关性捕获不足以及结构动态捕捉不充分。

### 方法

提出TAWRMAC框架，整合带重启的临时匿名游走、内存增强GNN和邻居共现嵌入技术。通过固定时间编码的内存增强GNN提高嵌入稳定性，通过明确捕获邻居相关性改善上下文表示，通过区分重复交互节点和形成新连接的节点来更好地捕获结构动态。

### 主要发现

在多个基准数据集上的实验表明，TAWRMAC在三种不同的负采样策略下，无论是在归纳还是直推设置中的动态链接预测和节点分类任务中，都始终优于最先进的方法。

### 结论

TAWRMAC通过提供稳定、可推广和上下文感知的嵌入，推进了连续时间动态图学习的最新技术水平。

### 翻译

动态图表示学习已成为分析社交网络分析、推荐系统和交通分析等领域中演化网络的关键技术。然而，现有的连续时间方法面临三个关键挑战：(1)一些方法仅依赖节点特定内存而未有效整合邻居节点信息，导致嵌入过时；(2)大多数未能明确捕获节点邻域间的相关性，限制了上下文感知能力；(3)许多方法在缺乏丰富链接属性的情况下无法完全捕捉演化图的结构动态。为解决这些局限性，我们引入了TAWRMAC——一个整合了带重启的临时匿名游走、内存增强和邻居共现嵌入的新颖框架。TAWRMAC通过具有固定时间编码的内存增强GNN提高嵌入稳定性，并通过明确捕获邻居相关性来改善上下文表示。此外，其带重启的临时匿名游走机制区分了展示重复交互的节点和形成超出其直接邻域新连接的节点。这种方法能更好地捕获结构动态并支持强归纳学习。在多个基准数据集上的广泛实验表明，TAWRMAC在三种不同负采样策略下，无论是在归纳还是直推设置中的动态链接预测和节点分类任务中，都始终优于最先进的方法。通过提供稳定、可推广和上下文感知的嵌入，TAWRMAC推进了连续时间动态图学习的最新技术水平。代码可在https://anonymous.4open.science/r/tawrmac-A253获取。


### 论文摘要

Dynamic graph representation learning has become essential for analyzing evolving networks in domains such as social network analysis, recommendation systems, and traffic analysis. However, existing continuous-time methods face three key challenges: (1) some methods depend solely on node-specific memory without effectively incorporating information from neighboring nodes, resulting in embedding staleness; (2) most fail to explicitly capture correlations between node neighborhoods, limiting contextual awareness; and (3) many fail to fully capture the structural dynamics of evolving graphs, especially in absence of rich link attributes. To address these limitations, we introduce TAWRMAC-a novel framework that integrates Temporal Anonymous Walks with Restart, Memory Augmentation, and Neighbor Co-occurrence embedding. TAWRMAC enhances embedding stability through a memory-augmented GNN with fixedtime encoding and improves contextual representation by explicitly capturing neighbor correlations. Additionally, its Temporal Anonymous Walks with Restart mechanism distinguishes between nodes exhibiting repetitive interactions and those forming new connections beyond their immediate neighborhood. This approach captures structural dynamics better and supports strong inductive learning. Extensive experiments on multiple benchmark datasets demonstrate that TAWRMAC consistently outperforms state-of-the-art methods in dynamic link prediction and node classification under both transductive and inductive settings across three different negative sampling strategies. By providing stable, generalizable, and context-aware embeddings, TAWRMAC advances the state of the art in continuous-time dynamic graph learning. The code is available at https://anonymous.4open.science/r/tawrmac-A253 .

---

## 47. Temporal Lifting as Latent-Space Regularization for Continuous-Time Flow Models in AI Systems

**论文链接:** [http://arxiv.org/abs/2510.09805v1](http://arxiv.org/abs/2510.09805v1)

**作者:** Jeffrey Camlin

**发布时间:** 2025-10-10

**备注:** 6 pages, 1 figure, 1 table, 1 algorithm

### GPT解析

### 总结

论文提出了一种针对连续时间动力系统的自适应时间重参数化的隐空间公式，称为时间提升方法。

### 背景

连续时间动力系统在处理近奇异行为时存在挑战，特别是在湍流等复杂系统中。

### 目的

开发一种方法来规范底层流的近奇异行为，同时保持其守恒定律，使轨迹变得全局平滑。

### 方法

引入一个平滑单调映射作为时间提升操作符，可以看作是连续时间归一化或时间扭曲算子。

### 主要发现

在提升坐标中，不可压缩Navier-Stokes方程在环面上的轨迹变得全局平滑，可以稳定物理信息神经网络和其他AI系统中使用的潜在流架构。

### 结论

该框架将解析正则性理论与用于刚性或湍流过程的表示学习方法联系起来。

### 翻译

我们提出了连续时间动力系统的自适应时间重参数化的隐空间公式。这种方法称为时间提升，引入了一个平滑单调映射，它可以规范底层流的近奇异行为，同时保持其守恒定律。在提升坐标中，如环面上不可压缩Navier-Stokes方程的轨迹变得全局平滑。从机器学习动力学的角度来看，时间提升作为连续时间归一化或时间扭曲算子，可以稳定物理信息神经网络和其他AI系统中使用的潜在流架构。该框架将解析正则性理论与用于处理刚性或湍流过程的表示学习方法联系起来。


### 论文摘要

We present a latent-space formulation of adaptive temporal reparametrization for continuous-time dynamical systems. The method, called *temporal lifting*, introduces a smooth monotone mapping $t \mapsto \tau(t)$ that regularizes near-singular behavior of the underlying flow while preserving its conservation laws. In the lifted coordinate, trajectories such as those of the incompressible Navier-Stokes equations on the torus $\mathbb{T}^3$ become globally smooth. From the standpoint of machine-learning dynamics, temporal lifting acts as a continuous-time normalization or time-warping operator that can stabilize physics-informed neural networks and other latent-flow architectures used in AI systems. The framework links analytic regularity theory with representation-learning methods for stiff or turbulent processes.

---

## 48. Combined Representation and Generation with Diffusive State Predictive Information Bottleneck

**论文链接:** [http://arxiv.org/abs/2510.09784v1](http://arxiv.org/abs/2510.09784v1)

**作者:** Richard John, Yunrui Qiu, Lukas Herron, Pratyush Tiwary

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文提出了一种名为Diffusive State Predictive Information Bottleneck (D-SPIB)的新方法，结合时间延迟信息瓶颈和扩散模型，用于分子科学中的生成建模，实现表征学习和生成目标的平衡。

### 背景

生成建模在高维空间中变得日益数据密集型，而在分子科学中，数据收集成本高且重要事件稀少，因此压缩到低维流形对各种下游任务（包括生成）特别重要。

### 目的

结合时间延迟信息瓶颈和扩散模型，在一个联合训练目标中实现表征学习和生成目标的平衡，构建灵活的架构，并学习热力学的连贯内部表征。

### 方法

将时间延迟信息瓶颈与扩散模型结合在一个联合训练目标中，创建名为D-SPIB的协议，使模型能够结合来自不同分子模拟轨迹的温度信息。

### 主要发现

D-SPIB能够在表征学习和生成目标之间取得平衡，且模型能够学习热力学的连贯且有用的内部表征。

### 结论

在多个分子任务上对D-SPIB进行了基准测试，展示了其探索训练集外物理条件的潜力。

### 翻译

在高维空间中，生成建模变得越来越数据密集型。在分子科学领域，数据收集成本高昂且重要事件稀少，压缩到低维流形对各种下游任务（包括生成）尤为重要。我们将一种旨在表征分子重要表示的时间延迟信息瓶颈与扩散模型结合在一个联合训练目标中。由此产生的协议，我们称之为扩散状态预测信息瓶颈，能够在一种灵活的架构中平衡表征学习和生成目标。此外，该模型能够结合来自不同分子模拟轨迹的温度信息，学习热力学的连贯且有用的内部表征。我们在多个分子任务上对D-SPIB进行了基准测试，展示了其探索训练集外物理条件的潜力。


### 论文摘要

Generative modeling becomes increasingly data-intensive in high-dimensional spaces. In molecular science, where data collection is expensive and important events are rare, compression to lower-dimensional manifolds is especially important for various downstream tasks, including generation. We combine a time-lagged information bottleneck designed to characterize molecular important representations and a diffusion model in one joint training objective. The resulting protocol, which we term Diffusive State Predictive Information Bottleneck (D-SPIB), enables the balancing of representation learning and generation aims in one flexible architecture. Additionally, the model is capable of combining temperature information from different molecular simulation trajectories to learn a coherent and useful internal representation of thermodynamics. We benchmark D-SPIB on multiple molecular tasks and showcase its potential for exploring physical conditions outside the training set.

---

## 49. HeSRN: Representation Learning On Heterogeneous Graphs via Slot-Aware Retentive Network

**论文链接:** [http://arxiv.org/abs/2510.09767v1](http://arxiv.org/abs/2510.09767v1)

**作者:** Yifan Lu, Ziyun Zou, Belal Alsinglawi, Islam Al-Qudah, Izzat Alsmadi, Feilong Tang, Pengfei Jiao, Shoaib Jameel

**发布时间:** 2025-10-10

### GPT解析

### 总结

HeSRN是一种新型异构图表示学习网络，通过槽感知结构和基于保留的编码器解决了图变换器计算复杂度高和无法有效建模异构语义的问题，在节点分类任务上取得了优异性能。

### 背景

图变换器通过自注意力机制在图表示学习中取得了显著进展，但其二次方计算复杂度和无法有效建模异构语义严重限制了其在真实世界异构图上的可扩展性和泛化能力。

### 目的

提出HeSRN，一种新型异构图槽感知保留网络，用于高效且表达性强的异构图表示学习，解决图变换器的计算复杂度和异构语义建模问题。

### 方法

HeSRN引入了槽感知结构编码器，通过将异构特征投影到独立槽并使用槽归一化和基于保留的融合来分离节点类型语义；用基于保留的编码器取代自注意力机制，在线性时间复杂度内建模依赖关系；采用异构保留编码器通过多尺度保留层联合捕获局部结构信号和全局异构语义。

### 主要发现

在四个真实世界异构图数据集上的实验表明，HeSRN在节点分类任务上始终优于最先进的异构图神经网络和图变换器基线，同时具有显著更低的计算复杂度。

### 结论

HeSRN通过创新的槽感知结构和基于保留的编码机制，有效解决了图变换器在异构图表示学习中的局限性，实现了高效且表达性强的学习性能。

### 翻译

图变换器最近通过自注意力机制捕获长距离依赖关系，在图表示学习中取得了显著进展。然而，它们的二次方计算复杂度和无法有效建模异构语义严重限制了它们在真实世界异构图上的可扩展性和泛化能力。为了解决这些问题，我们提出了HeSRN，一种用于高效且表达性强的异构图表示学习的新型异构槽感知保留网络。HeSRN引入了一种槽感知结构编码器，通过将异构特征投影到独立槽并通过槽归一化和基于保留的融合来对齐它们的分布，从而显式地分离节点类型语义，有效缓解了先前基于Transformer模型中强制特征空间统一引起的语义纠缠。此外，我们用基于保留的编码器取代了自注意力机制，该编码器在线性时间复杂度内建模结构和上下文依赖关系，同时保持强大的表达能力。进一步采用异构保留编码器通过多尺度保留层联合捕获局部结构信号和全局异构语义。在四个真实世界异构图数据集上的大量实验表明，HeSRN在节点分类任务上始终优于最先进的异构图神经网络和图变换器基线，以显著更低的计算复杂度实现了更高的准确性。


### 论文摘要

Graph Transformers have recently achieved remarkable progress in graph representation learning by capturing long-range dependencies through self-attention. However, their quadratic computational complexity and inability to effectively model heterogeneous semantics severely limit their scalability and generalization on real-world heterogeneous graphs. To address these issues, we propose HeSRN, a novel Heterogeneous Slot-aware Retentive Network for efficient and expressive heterogeneous graph representation learning. HeSRN introduces a slot-aware structure encoder that explicitly disentangles node-type semantics by projecting heterogeneous features into independent slots and aligning their distributions through slot normalization and retention-based fusion, effectively mitigating the semantic entanglement caused by forced feature-space unification in previous Transformer-based models. Furthermore, we replace the self-attention mechanism with a retention-based encoder, which models structural and contextual dependencies in linear time complexity while maintaining strong expressive power. A heterogeneous retentive encoder is further employed to jointly capture both local structural signals and global heterogeneous semantics through multi-scale retention layers. Extensive experiments on four real-world heterogeneous graph datasets demonstrate that HeSRN consistently outperforms state-of-the-art heterogeneous graph neural networks and Graph Transformer baselines on node classification tasks, achieving superior accuracy with significantly lower computational complexity.

---

## 50. 论文ID: 2510.09764v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.09764v1.json'

---

## 51. STaTS: Structure-Aware Temporal Sequence Summarization via Statistical Window Merging

**论文链接:** [http://arxiv.org/abs/2510.09593v1](http://arxiv.org/abs/2510.09593v1)

**作者:** Disharee Bhowmick, Ranjith Ramanathan, Sathyanarayanan N. Aakur

**发布时间:** 2025-10-10

**备注:** 10 pages, 5 figures, 4 tables. Under Review

### GPT解析

### 总结

STaTS是一种轻量级无监督框架，用于结构感知的时间序列总结，能自适应压缩时间序列为保留信息的令牌序列，实现高达30倍的压缩率，同时保留核心时间动态。

### 背景

时间序列数据常含潜在时间结构、状态转换、重复模式和变异性爆发，但现有模型通常处理原始或固定窗口序列，将所有时间步视为同等重要，导致在长序列或噪声序列中效率低下、鲁棒性差和可扩展性有限。

### 目的

提出一个轻量级无监督框架STaTS，用于结构感知的时间序列总结，自适应压缩单变量和多变量时间序列为紧凑的信息保留令牌序列。

### 方法

STaTS使用基于BIC的统计散度标准在多个时间分辨率上检测变化点，然后用简单函数（如均值）或生成模型（如GMM）对每个段进行总结，作为模型无关的预处理器可集成到现有无监督时间序列编码器中无需重新训练。

### 主要发现

在150多个数据集上的实验表明，STaTS可实现全模型85-90%的性能，同时大幅降低计算成本；提高噪声下的鲁棒性，保留判别性结构；优于均匀和基于聚类的压缩基线。

### 结论

STaTS是一种原则性的、通用的解决方案，用于高效、结构感知的时间序列建模。

### 翻译

时间序列数据通常包含潜在的时间结构、局部平稳状态之间的转换、重复的motifs和变异性爆发，这些特征在标准表示学习管道中很少被利用。现有模型通常处理原始或固定窗口序列，将所有时间步视为同等重要，这导致在长序列或有噪声序列中效率低下、鲁棒性差和可扩展性有限。我们提出了STaTS，一种用于结构感知时间总结的轻量级无监督框架，能自适应地将单变量和多变量时间序列压缩为紧凑的、保留信息的令牌序列。STaTS使用基于BIC的统计散度标准在多个时间分辨率上检测变化点，然后使用简单函数如均值或生成模型如GMM对每个段进行总结。这一过程实现了高达30倍的序列压缩，同时保留核心时间动态。STaTS作为模型无关的预处理器，可以与现有无监督时间序列编码器集成而无需重新训练。在150多个数据集上的广泛实验，包括UCR-85、UCR-128和UEA-30档案上的分类任务，以及ETTh1和ETTh2、ETTm1和Electricity上的预测，表明STaTS可实现全模型85-90%的性能，同时显著降低计算成本。此外，STaTS提高了噪声下的鲁棒性并保留了判别性结构，优于均匀和基于聚类的压缩基线。这些结果将STaTS定位为一种原则性的、通用的解决方案，用于高效、结构感知的时间序列建模。


### 论文摘要

Time series data often contain latent temporal structure, transitions between locally stationary regimes, repeated motifs, and bursts of variability, that are rarely leveraged in standard representation learning pipelines. Existing models typically operate on raw or fixed-window sequences, treating all time steps as equally informative, which leads to inefficiencies, poor robustness, and limited scalability in long or noisy sequences. We propose STaTS, a lightweight, unsupervised framework for Structure-Aware Temporal Summarization that adaptively compresses both univariate and multivariate time series into compact, information-preserving token sequences. STaTS detects change points across multiple temporal resolutions using a BIC-based statistical divergence criterion, then summarizes each segment using simple functions like the mean or generative models such as GMMs. This process achieves up to 30x sequence compression while retaining core temporal dynamics. STaTS operates as a model-agnostic preprocessor and can be integrated with existing unsupervised time series encoders without retraining. Extensive experiments on 150+ datasets, including classification tasks on the UCR-85, UCR-128, and UEA-30 archives, and forecasting on ETTh1 and ETTh2, ETTm1, and Electricity, demonstrate that STaTS enables 85-90\% of the full-model performance while offering dramatic reductions in computational cost. Moreover, STaTS improves robustness under noise and preserves discriminative structure, outperforming uniform and clustering-based compression baselines. These results position STaTS as a principled, general-purpose solution for efficient, structure-aware time series modeling.

---

## 52. IF-D: A High-Frequency, General-Purpose Inertial Foundation Dataset for Self-Supervised Learning

**论文链接:** [http://arxiv.org/abs/2510.09539v1](http://arxiv.org/abs/2510.09539v1)

**作者:** Patrick Ferreira, Paula Costa

**发布时间:** 2025-10-10

**备注:** 5 pages, 5 figures. Submitted to IEEE ICASSP 2026. Copyright 2026  IEEE. Personal use of this material is permitted. Permission from IEEE must  be obtained for all other uses

### GPT解析

### 总结

本研究介绍了一个名为IF-D的大型惯性数据集，旨在支持IMU时间序列的自监督学习和基础学习。该数据集包含连续、长时间、多通道的记录，经过精心采集和校准，为鲁棒表示学习和下游任务提供了高质量数据支持。

### 背景

惯性测量单元(IMU)时间序列数据在自动驾驶、运动分析和导航等领域有广泛应用。然而，现有数据集可能存在平台特定偏差，且缺乏多样化的运动模式，限制了模型的学习效果和泛化能力。

### 目的

开发一个大型惯性数据集IF-D，以减轻平台特定运动偏差，让模型接触物理动力学和典型测量噪声，从而促进鲁棒表示学习和支持下游任务如事件检测、运动模式识别和惯性导航。

### 方法

使用UM7 IMU传感器，安装在3D打印的球形外壳内，在车辆行驶过程中采集加速度计、陀螺仪和磁力计数据。采样率为200Hz，收集时长约135分钟，产生约160万个样本。采用六方向加速度计校准、静止陀螺仪偏差估计和磁力计硬/软铁校正的椭球拟合等方法进行数据预处理和校准。

### 主要发现

成功构建了一个包含九个传感器通道、约160万个样本的大型惯性数据集。通过精心设计的校准程序，有效减轻了平台特定偏差，并提供了定量的校准结果，验证了数据集的质量。

### 结论

IF-D数据集通过多样化的自由旋转运动和全面的校准过程，为IMU时间序列的自监督学习和基础学习提供了高质量数据支持。该数据集能够促进鲁棒表示学习，支持事件检测、运动模式识别和惯性导航等多种下游任务。

### 翻译

我们提出了IF-D，这是一个大规模惯性数据集，旨在支持IMU时间序列的自监督学习和基础学习。IF-D包含连续、长时间、多通道记录（加速度计、陀螺仪、磁力计），采样率为200Hz，使用安装在3D打印球形外壳内的UM7 IMU采集，该外壳在车辆行驶期间促进多样化的自由旋转。收集时长约135分钟，产生约160万个样本，涵盖九个传感器通道。我们描述了数据采集设置、预处理和校准程序（六方向加速度计校准、静止陀螺仪偏差估计，以及磁力计硬/软铁校正的椭球拟合），并提供了定量的校准结果。IF-D旨在减轻平台特定运动偏差，让模型接触物理动力学和典型测量噪声，从而促进鲁棒表示学习和下游任务，如事件检测、运动模式识别和惯性导航。


### 论文摘要

We present IF-D, a large-scale inertial dataset designed to enable self-supervised and foundational learning for IMU time series. IF-D comprises continuous, long-duration multichannel recordings (accelerometer, gyroscope, magnetometer) sampled at 200Hz using a UM7 IMU mounted inside a 3D-printed spherical enclosure that promotes diverse, free rotations during vehicle traversal. The collection spans approximately 135 minutes of recording, yielding around 1.6 million samples across nine sensor channels. We describe the data acquisition setup, preprocessing, and calibration procedures (six-orientation accelerometer calibration, stationary gyroscope bias estimation, and ellipsoid fitting for magnetometer hard-/soft-iron correction), and provide quantitative calibration results. IF-D is designed to mitigate platform specific motion bias and expose models to both physical dynamics and typical measurement noise, thereby facilitating robust representation learning and downstream tasks such as event detection, motion mode recognition, and inertial navigation.

---

## 53. What Do Temporal Graph Learning Models Learn?

**论文链接:** [http://arxiv.org/abs/2510.09416v1](http://arxiv.org/abs/2510.09416v1)

**作者:** Abigail J. Hayes, Tobias Schumacher, Markus Strohmaier

**发布时间:** 2025-10-10

### GPT解析

### 总结

本研究系统评估了七种时间图学习模型捕捉时间图链接结构八种基本属性的能力，揭示了模型在捕捉某些属性上的优势和局限性，为时间图学习模型的应用提供了实践见解。

### 背景

时间图学习已成为图表示学习的中心主题，最先进的模型在多个基准测试中表现出色。然而，最近的研究对基准测试结果的可靠性提出了质疑，指出了常用评估协议的问题以及简单启发式方法的竞争力，引发了对模型实际利用底层图哪些特性的疑问。

### 目的

探究时间图学习模型实际利用了底层图的哪些特性进行预测，系统评估模型捕捉时间图链接结构相关基本属性的能力。

### 方法

系统地评估七种时间图学习模型在捕捉八种与时间图链接结构相关的基本属性方面的能力。这些属性包括结构特性（如密度）、时间模式（如近期性）和边形成机制（如同质性）。研究使用了合成和真实世界数据集进行分析。

### 主要发现

研究结果呈现出复杂的图景：模型能够很好地捕捉某些属性，但无法重现其他属性。这暴露了模型在时间图学习方面的重要局限性。

### 结论

研究结果为时间图学习模型的应用提供了实践见解，并激励时间图学习研究进行更多可解释性驱动的评估，以更好地理解模型的工作原理和局限性。

### 翻译

时间图学习已成为图表示学习的中心主题，众多基准测试表明最先进模型的强大性能。然而，最近的工作对基准测试结果的可靠性提出了担忧，指出了常用评估协议的问题以及简单启发式方法的惊人竞争力。这种对比引发了关于时间图学习模型实际上利用了底层图的哪些特性来进行预测的问题。我们通过系统地评估七种模型捕捉时间图链接结构相关八种基本属性的能力来解决这一问题。这些属性包括结构特性如密度、时间模式如近期性，以及边形成机制如同质性。使用合成和真实世界数据集，我们分析了模型学习这些属性的程度。我们的研究结果呈现出复杂的图景：模型能够很好地捕捉某些属性，但无法重现其他属性。通过这一点，我们暴露了重要的局限性。总体而言，我们相信我们的结果为时间图学习模型的应用提供了实践见解，并激励时间图学习研究进行更多可解释性驱动的评估。


### 论文摘要

Learning on temporal graphs has become a central topic in graph representation learning, with numerous benchmarks indicating the strong performance of state-of-the-art models. However, recent work has raised concerns about the reliability of benchmark results, noting issues with commonly used evaluation protocols and the surprising competitiveness of simple heuristics. This contrast raises the question of which properties of the underlying graphs temporal graph learning models actually use to form their predictions. We address this by systematically evaluating seven models on their ability to capture eight fundamental attributes related to the link structure of temporal graphs. These include structural characteristics such as density, temporal patterns such as recency, and edge formation mechanisms such as homophily. Using both synthetic and real-world datasets, we analyze how well models learn these attributes. Our findings reveal a mixed picture: models capture some attributes well but fail to reproduce others. With this, we expose important limitations. Overall, we believe that our results provide practical insights for the application of temporal graph learning models, and motivate more interpretability-driven evaluations in temporal graph learning research.

---

## 54. Cross-Receiver Generalization for RF Fingerprint Identification via Feature Disentanglement and Adversarial Training

**论文链接:** [http://arxiv.org/abs/2510.09405v1](http://arxiv.org/abs/2510.09405v1)

**作者:** Yuhao Pan, Xiucheng Wang, Nan Cheng, Wenchao Xu

**发布时间:** 2025-10-10

### GPT解析

### 总结

射频指纹识别(RFFI)是一种利用设备制造过程中引入的硬件级不完善性来实现精确发射器识别的关键技术。然而，接收器引起的变异性限制了深度神经网络在实际部署中的应用。作者提出了一种对抗训练和风格转移相结合的框架，能够分离发射器和接收器特征，提高跨接收器变化的鲁棒性。

### 背景

射频指纹识别是无线网络安全的关键技术，它利用设备制造过程中引入的硬件级不完善性来实现精确的发射器识别。深度神经网络在提取判别性特征方面表现出色，但它们的实际部署受到接收器引起的变化性的阻碍。

### 目的

解决接收器引起的特征偏移问题，防止RFFI模型过度拟合接收器特定模式，提高模型在不同接收器环境下的鲁棒性。

### 方法

提出一种对抗接收器变化的RFFI框架，整合对抗训练和风格转移，明确分离发射器和接收器特征。通过强制执行域不变表示学习，将真实的硬件签名与接收器伪影隔离，确保对接收器变化的鲁棒性。

### 主要发现

在多接收器数据集上的广泛实验表明，该方法始终优于最先进的基线方法，在各种接收器设置下平均准确率提高了高达10%。

### 结论

所提出的方法有效解决了接收器变化导致的性能下降问题，显著提高了RFFI系统在实际部署中的鲁棒性和准确性。

### 翻译

射频指纹识别是一种关键技术，用于无线网络安全，它利用设备制造过程中引入的硬件级内在不完善性来实现精确的发射器识别。虽然深度神经网络在提取判别性特征方面表现出色，但它们的实际部署受到接收器引起的变化性的阻碍。在实践中，射频指纹信号包含发射器特定特征以及信道失真和接收器引起的偏差。尽管信道均衡可以减轻信道噪声，但接收器引起的特征偏移仍然在很大程度上未得到解决，导致RFFI模型过度拟合接收器特定模式。当训练和评估使用相同的接收器时，这一限制尤其成问题，因为在部署中更换接收器可能导致性能大幅下降。为了应对这一挑战，我们提出了一种对抗接收器变化的RFFI框架，整合对抗训练和风格转移，明确分离发射器和接收器特征。通过强制执行域不变表示学习，我们的方法将真实的硬件签名与接收器伪影隔离，确保对接收器变化的鲁棒性。在多接收器数据集上的广泛实验表明，我们的方法始终优于最先进的基线方法，在各种接收器设置下平均准确率提高了高达10%。


### 论文摘要

Radio frequency fingerprint identification (RFFI) is a critical technique for wireless network security, leveraging intrinsic hardware-level imperfections introduced during device manufacturing to enable precise transmitter identification. While deep neural networks have shown remarkable capability in extracting discriminative features, their real-world deployment is hindered by receiver-induced variability. In practice, RF fingerprint signals comprise transmitter-specific features as well as channel distortions and receiver-induced biases. Although channel equalization can mitigate channel noise, receiver-induced feature shifts remain largely unaddressed, causing the RFFI models to overfit to receiver-specific patterns. This limitation is particularly problematic when training and evaluation share the same receiver, as replacing the receiver in deployment can cause substantial performance degradation. To tackle this challenge, we propose an RFFI framework robust to cross-receiver variability, integrating adversarial training and style transfer to explicitly disentangle transmitter and receiver features. By enforcing domain-invariant representation learning, our method isolates genuine hardware signatures from receiver artifacts, ensuring robustness against receiver changes. Extensive experiments on multi-receiver datasets demonstrate that our approach consistently outperforms state-of-the-art baselines, achieving up to a 10% improvement in average accuracy across diverse receiver settings.

---

## 55. Automatic Music Sample Identification with Multi-Track Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.11507v1](http://arxiv.org/abs/2510.11507v1)

**作者:** Alain Riou, Joan Serrà, Yuki Mitsufuji

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了一种基于自监督学习的音频采样识别方法，通过多轨数据集创建人工混合正样本对，并设计新型对比学习目标，显著提高了采样内容检测和原始素材检索的准确性和鲁棒性。

### 背景

采样是现代音乐制作中常见的技术，指利用现有音频片段创建新的音乐内容。自动识别采样内容是一个具有挑战性的任务。

### 目的

开发一种自动样本识别系统，能够检测采样内容并检索其原始素材来源。

### 方法

采用自监督学习方法，利用多轨数据集创建人工混合的正样本对，设计了一种新的对比学习目标函数。

### 主要发现

该方法显著优于先前最先进的基线方法，对各种音乐类型具有鲁棒性，并在参考数据库规模扩大时表现出良好的可扩展性。

### 结论

详细分析了训练流程中不同组件的贡献，特别强调了高质量分离音轨对于此任务的必要性。

### 翻译

采样，即利用现有音频片段创建新音乐内容的技术，在现代音乐制作中非常普遍。在本文中，我们解决了自动样本识别这一具有挑战性的任务，即检测采样内容并检索其原始素材。为此，我们采用了一种自监督学习方法，利用多轨数据集创建人工混合的正样本对，并设计了一种新颖的对比学习目标。我们证明该方法显著优于先前最先进的基线方法，对各种音乐类型具有鲁棒性，并且在增加参考数据库中的噪音歌曲数量时具有良好的可扩展性。此外，我们详细分析了训练流程中不同组件的贡献，并特别强调了高质量分离音轨对于此任务的必要性。


### 论文摘要

Sampling, the technique of reusing pieces of existing audio tracks to create new music content, is a very common practice in modern music production. In this paper, we tackle the challenging task of automatic sample identification, that is, detecting such sampled content and retrieving the material from which it originates. To do so, we adopt a self-supervised learning approach that leverages a multi-track dataset to create positive pairs of artificial mixes, and design a novel contrastive learning objective. We show that such method significantly outperforms previous state-of-the-art baselines, that is robust to various genres, and that scales well when increasing the number of noise songs in the reference database. In addition, we extensively analyze the contribution of the different components of our training pipeline and highlight, in particular, the need for high-quality separated stems for this task.

---

## 56. Investigating Identity Signals in Conversational Facial Dynamics via Disentangled Expression Features

**论文链接:** [http://arxiv.org/abs/2510.11223v1](http://arxiv.org/abs/2510.11223v1)

**作者:** Masoumeh Chapariniya, Pierre Vuillecard, Jean-Marc Odobez, Volker Dellwo, Teodora Vukovic

**发布时间:** 2025-10-13

### GPT解析

### 总结

本研究探讨了仅通过面部表情的动态成分而非静态面部外观是否能够识别个体。研究使用FLAME 3D可变形模型分离面部形状和表情动态，从对话视频中提取参数。在包含1,429名说话者的CANDOR数据集上，使用带有监督对比学习的Conformer模型实现61.14%的识别准确率，证明面部动态包含强烈的身份特征。研究还引入了漂移-噪声比率(DNR)指标来量化形状-表情分离的可靠性，发现DNR与识别性能呈负相关。研究结果表明对话面部动态中存在特定于个人的特征，对社会感知和临床评估具有重要意义。

### 背景

面部识别研究通常关注静态面部特征，而对面部表情动态与身份识别的关系研究较少。

### 目的

探究是否仅通过面部表情的动态成分就能识别个体，而不依赖于静态面部外观。

### 方法

使用FLAME 3D可变形模型实现面部形状和表情动态的分离，从对话视频中逐帧提取参数并仅保留表情和下颌系数。应用带有监督对比学习的Conformer模型进行1,429路分类任务。引入漂移-噪声比率(DNR)指标量化形状-表情分离的可靠性。

### 主要发现

1) 面部动态携带强烈的身份特征，识别准确率达到61.14%，是随机猜测的458倍；2) 漂移-噪声比率(DNR)与识别性能呈强负相关，表明不稳定的形状估计会损害动态识别能力；3) 对话面部动态中存在特定于个人的特征。

### 结论

面部表情的动态成分包含足够的信息用于个体识别，这种能力对社会感知研究和临床评估具有重要应用价值。

### 翻译

本研究调查个体是否仅能通过面部表情的纯动态成分被识别，而独立于静态面部外观。我们利用FLAME 3D可变形模型来实现面部形状和表情动态之间的明确分离，从对话视频中逐帧提取参数，同时仅保留表情和下颌系数。在包含1,429名说话者在自然对话中的CANDOR数据集上，我们采用监督对比学习的Conformer模型在1,429路分类任务上达到61.14%的准确率——比随机猜测高458倍——证明了面部动态携带强烈的身份特征。我们引入了漂移-噪声比率(DNR)，通过测量跨会话形状变化相对于会话内变异性来量化形状-表情分离的可靠性。DNR与识别性能呈强负相关，证实不稳定的形状估计会损害动态识别。我们的发现揭示了对话面部动态中存在特定于个人的特征，对社会感知和临床评估有启示意义。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要研究是否可以仅通过面部表情的动态成分（即面部运动的方式）来识别一个人，而不依赖于静态的面部外观特征。这个问题在临床评估中很重要，因为可以区分个人特定的表情模式与神经系统症状；在社会互动研究中可以区分个人风格与情感内容；在技术系统中可以创建更自然的个性化虚拟角色动画。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过使用FLAME 3D可变形模型来实现面部形状和表情动态的显式分离，从对话视频中逐帧提取参数，只保留表情和下颌系数。他们借鉴了FLAME模型用于参数化面部，VGGHeads用于从2D视频估计参数，Conformer模型结合自注意力和卷积进行时间建模，以及监督对比学习来学习判别性表示。这些现有技术被创新性地组合以解决身份识别问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是每个人的面部表情动态（如微笑、说话和情绪表达的方式）具有独特的个体特征，这些特征可以作为身份识别的信号，即使不考虑静态的面部外观。整体流程包括：1)使用VGGHeads从视频帧提取FLAME参数；2)只保留动态成分（表情系数和下颌旋转参数）；3)使用监督对比学习训练Conformer模型处理这些动态序列；4)训练线性分类器进行身份识别；5)引入漂移-噪声比(DNR)量化特征分离质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次大规模证明纯面部动态（完全分离外观后）包含身份信号；2)证明Conformer的混合架构能最优捕获多尺度面部模式；3)提出漂移-噪声比(DNR)度量量化分离质量；4)全面分析基于动态识别的上下文和数据需求。相比之前工作，早期方法混合了外观和运动，无法确定识别是基于运动还是几何，而本文通过FLAME模型实现数学上的形状和表情分离，确保分析仅关注动态行为。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次证明可以通过纯面部动态特征（独立于静态面部外观）实现高精度身份识别，并提出了新的度量方法评估特征分离质量。'}


### 论文摘要

This work investigates whether individuals can be identified solely through the pure dynamical components of their facial expressions, independent of static facial appearance. We leverage the FLAME 3D morphable model to achieve explicit disentanglement between facial shape and expression dynamics, extracting frame-by-frame parameters from conversational videos while retaining only expression and jaw coefficients. On the CANDOR dataset of 1,429 speakers in naturalistic conversations, our Conformer model with supervised contrastive learning achieves 61.14\%accuracy on 1,429-way classification -- 458 times above chance -- demonstrating that facial dynamics carry strong identity signatures. We introduce a drift-to-noise ratio (DNR) that quantifies the reliability of shape expression separation by measuring across-session shape changes relative to within-session variability. DNR strongly negatively correlates with recognition performance, confirming that unstable shape estimation compromises dynamic identification. Our findings reveal person-specific signatures in conversational facial dynamics, with implications for social perception and clinical assessment.

---

## 57. Class Prototypes based Contrastive Learning for Classifying Multi-Label and Fine-Grained Educational Videos

**论文链接:** [http://arxiv.org/abs/2510.11204v1](http://arxiv.org/abs/2510.11204v1)

**作者:** Rohit Gupta, Anirban Roy, Claire Christensen, Sujeong Kim, Sarah Gerard, Madeline Cincebeaux, Ajay Divakaran, Todd Grindal, Mubarak Shah

**发布时间:** 2025-10-13

**备注:** Published at CVPR 2023

### GPT解析

### 总结

本文提出了一种基于类原型的监督对比学习方法，用于检测在线视频中的教育内容，特别是识字和数学两个类别。该方法使用多模态Transformer网络捕捉视频中的视觉和音频线索交互，并在新创建的APPROVE数据集上进行了验证，结果表明该方法优于现有基线。

### 背景

儿童早期在线媒体消费的增长需要数据驱动工具来帮助教育工作者筛选适合的教育内容。

### 目的

开发一种能够检测在线视频中教育内容的方法，特别关注识字和数学两个广泛使用的教育内容类别。

### 方法

提出一种基于类原型的监督对比学习方法，将其视为细粒度多标签分类问题。使用多模态Transformer网络捕捉视频中的视觉和音频线索之间的交互。学习每个类别的类原型，通过损失函数最小化类原型与其样本之间的距离，同时最大化类原型与其他类别样本之间的距离。

### 主要发现

提出的方法在APPROVE数据集（包含193小时专家标注的视频，共19个类别）和其他基准测试（如Youtube-8M和COIN）上优于强大的基线方法。

### 结论

该方法能有效处理多标签细粒度样本，为教育工作者筛选适合的教育内容提供了有效工具。APPROVE数据集已在https://github.com/rohit-gupta/MMContrast/tree/main/APPROVE公开可用。

### 翻译

儿童早期在线媒体消费的近期增长需要数据驱动工具，使教育工作者能够为幼儿筛选适当的教育内容。本文提出了一种检测在线视频中教育内容的方法。我们专注于两个广泛使用的教育内容类别：识字和数学。对于每个类别，我们基于共同核心标准选择突出的代码（子类）。例如，识字代码包括'字母名称'、'字母发音'，数学代码包括'计数'、'分类'。我们将此视为细粒度多标签分类问题，因为视频可能包含多种类型的教育内容，且内容类别可能在视觉上相似（例如，'字母名称'与'字母发音'）。我们提出了一种新颖的基于类原型的监督对比学习方法，能够处理与多个标签相关的细粒度样本。我们为每个类别学习一个类原型，并采用损失函数来最小化类原型与其类别样本之间的距离。同样，类原型与其他类别样本之间的距离被最大化。由于视觉和音频线索的 alignment 对有效理解至关重要，我们考虑使用多模态Transformer网络在视频学习嵌入的同时捕捉视频中视觉和音频线索之间的交互。为了评估，我们提出了一个数据集APPROVE，使用YouTube上的教育视频，由教育研究人员用细粒度教育类别进行标注。APPROVE包含193小时专家标注的视频，共19个类别。提出的方法在APPROVE和其他基准测试（如Youtube-8M和COIN）上优于强大的基线方法。数据集可在https://github.com/rohit-gupta/MMContrast/tree/main/APPROVE获取。


### 论文摘要

The recent growth in the consumption of online media by children during early childhood necessitates data-driven tools enabling educators to filter out appropriate educational content for young learners. This paper presents an approach for detecting educational content in online videos. We focus on two widely used educational content classes: literacy and math. For each class, we choose prominent codes (sub-classes) based on the Common Core Standards. For example, literacy codes include `letter names', `letter sounds', and math codes include `counting', `sorting'. We pose this as a fine-grained multilabel classification problem as videos can contain multiple types of educational content and the content classes can get visually similar (e.g., `letter names' vs `letter sounds'). We propose a novel class prototypes based supervised contrastive learning approach that can handle fine-grained samples associated with multiple labels. We learn a class prototype for each class and a loss function is employed to minimize the distances between a class prototype and the samples from the class. Similarly, distances between a class prototype and the samples from other classes are maximized. As the alignment between visual and audio cues are crucial for effective comprehension, we consider a multimodal transformer network to capture the interaction between visual and audio cues in videos while learning the embedding for videos. For evaluation, we present a dataset, APPROVE, employing educational videos from YouTube labeled with fine-grained education classes by education researchers. APPROVE consists of 193 hours of expert-annotated videos with 19 classes. The proposed approach outperforms strong baselines on APPROVE and other benchmarks such as Youtube-8M, and COIN. The dataset is available at https://github.com/rohit-gupta/MMContrast/tree/main/APPROVE

---

## 58. PhysioME: A Robust Multimodal Self-Supervised Framework for Physiological Signals with Missing Modalities

**论文链接:** [http://arxiv.org/abs/2510.11110v1](http://arxiv.org/abs/2510.11110v1)

**作者:** Cheol-Hui Lee, Hwa-Yeon Lee, Min-Kyung Jung, Dong-Joo Kim

**发布时间:** 2025-10-13

**备注:** 9 pages, 2 figures

### GPT解析

### 总结

PhysioME是一种鲁棒框架，能够在生理信号医疗应用中处理模态缺失问题，通过多模态自监督学习、专门设计的网络架构和恢复解码器，确保在各种缺失场景下的可靠性能。

### 背景

生理信号在医疗应用中经常出现缺失或损坏的情况，这主要是由于硬件限制或运动伪影造成的。然而，大多数现有方法假设所有模态都可用，这导致在任何模态缺失时性能会显著下降。

### 目的

克服模态缺失导致性能下降的局限性，提出PhysioME框架，确保在模态缺失条件下保持可靠性能。

### 方法

PhysioME采用了三种主要方法：(1) 多模态自监督学习方法，结合对比学习和掩码预测；(2) 专门用于捕捉每种生理信号模态时间动态的Dual-Path NeuroNet主干网络；(3) 恢复解码器，用于重建缺失的模态令牌，能够灵活处理不完整的输入。

### 主要发现

实验结果表明，PhysioME在各种模态缺失场景下都实现了高度一致性和泛化性能。

### 结论

这些发现强调了PhysioME作为可靠工具的潜力，可以在数据不完善的现实环境中支持临床决策。

### 翻译

缺失或损坏的模态在基于生理信号的医疗应用中很常见，这是由于硬件限制或运动伪影造成的。然而，大多数现有方法假设所有模态都可用，这导致在任何模态缺失时性能显著下降。为了克服这一局限性，本研究提出了PhysioME，一个鲁棒框架，旨在确保在模态缺失条件下保持可靠性能。PhysioME采用：(1) 多模态自监督学习方法，结合对比学习和掩码预测；(2) 专门用于捕捉每种生理信号模态时间动态的Dual-Path NeuroNet主干网络；(3) 恢复解码器，用于重建缺失的模态令牌，能够灵活处理不完整的输入。实验结果表明，PhysioME在各种模态缺失场景下都实现了高度一致性和泛化性能。这些发现强调了PhysioME作为可靠工具的潜力，可以在数据不完善的现实环境中支持临床决策。


### 论文摘要

Missing or corrupted modalities are common in physiological signal-based medical applications owing to hardware constraints or motion artifacts. However, most existing methods assume the availability of all modalities, resulting in substantial performance degradation in the absence of any modality. To overcome this limitation, this study proposes PhysioME, a robust framework designed to ensure reliable performance under missing modality conditions. PhysioME adopts: (1) a multimodal self-supervised learning approach that combines contrastive learning with masked prediction; (2) a Dual-PathNeuroNet backbone tailored to capture the temporal dynamics of each physiological signal modality; and (3) a restoration decoder that reconstructs missing modality tokens, enabling flexible processing of incomplete inputs. The experimental results show that PhysioME achieves high consistency and generalization performance across various missing modality scenarios. These findings highlight the potential of PhysioME as a reliable tool for supporting clinical decision-making in real-world settings with imperfect data availability.

---

## 59. Source-Free Object Detection with Detection Transformer

**论文链接:** [http://arxiv.org/abs/2510.11090v1](http://arxiv.org/abs/2510.11090v1)

**作者:** Huizai Yao, Sicheng Zhao, Shuo Lu, Hui Chen, Yangyang Li, Guoping Liu, Tengfei Xing, Chenggang Yan, Jianhua Tao, Guiguang Ding

**发布时间:** 2025-10-13

**DOI:** 10.1109/TIP.2025.3607621

**备注:** IEEE Transactions on Image Processing

### GPT解析

### 总结

本文提出了FRANCK(Feature Reweighting ANd Contrastive Learning NetworK)，一种专门为DETR架构设计的源域无目标检测(SFOD)框架，通过四个关键组件实现查询为中心的特征增强，有效提升了模型在目标域的鲁棒性和泛化能力。

### 背景

源域无目标检测(SFOD)允许在不访问源数据的情况下，将知识从源域转移到无监督的目标域进行目标检测。然而，现有方法要么局限于传统目标检测模型(如Faster R-CNN)，要么缺乏针对新型目标检测架构(尤其是Detection Transformer/DETR)的专门适配。

### 目的

开发一个专门为DETR模型设计的SFOD框架，解决现有方法在新型目标检测架构上的局限性，提高目标检测在目标域的性能。

### 方法

FRANCK框架包含四个关键组件：(1)基于目标性评分的样本重加权(OSSR)模块，通过计算注意力目标性评分并重加权检测损失来强调难以识别区域；(2)基于匹配的记忆库对比学习(CMMB)模块，集成多级特征到记忆库中增强类别对比学习；(3)不确定性加权的查询融合特征蒸馏(UQFD)模块，通过预测质量重加权和查询特征融合改进特征蒸馏；(4)改进的自训练流程，具有动态教师更新间隔(DTUI)以优化伪标签质量。

### 主要发现

通过这些组件，FRANCK有效将源预训练的DETR模型适应到目标域，显著增强了模型的鲁棒性和泛化能力。在多个基准测试上的广泛实验表明，该方法达到了最先进的性能。

### 结论

FRANCK是一种有效的、与基于DETR的SFOD模型兼容的方法，为目标检测领域特别是DETR架构的源域自适应提供了创新解决方案，具有显著的应用价值和兼容性。

### 翻译

源域无目标检测(SFOD)使知识能够从源域转移到无监督的目标域进行目标检测，而无需访问源数据。大多数现有的SFOD方法要么局限于传统的目标检测(OD)模型(如Faster R-CNN)，要么被设计为通用解决方案，没有针对新型OD架构(尤其是Detection Transformer/DETR)进行专门适配。在本文中，我们引入了特征重加权与对比学习网络(FRANCK)，一种新型SFOD框架，专门为DETR执行以查询为中心的特征增强。FRANCK包含四个关键组件：(1)基于目标性评分的样本重加权(OSSR)模块，在多尺度编码器特征图上计算基于注意力的目标性评分，对检测损失进行重加权，以强调难以识别的区域；(2)基于匹配的记忆库对比学习(CMMB)模块，将多级特征集成到记忆库中，增强类别的对比学习；(3)不确定性加权的查询融合特征蒸馏(UQFD)模块，通过预测质量重加权和查询特征融合来改进特征蒸馏；(4)改进的自训练流程，具有动态教师更新间隔(DTUI)，优化伪标签质量。通过利用这些组件，FRANCK有效地将源预训练的DETR模型适应到目标域，增强了鲁棒性和泛化能力。在几个广泛使用的基准测试上的广泛实验表明，我们的方法达到了最先进的性能，突显了其有效性与基于DETR的SFOD模型的兼容性。


### 论文摘要

Source-Free Object Detection (SFOD) enables knowledge transfer from a source domain to an unsupervised target domain for object detection without access to source data. Most existing SFOD approaches are either confined to conventional object detection (OD) models like Faster R-CNN or designed as general solutions without tailored adaptations for novel OD architectures, especially Detection Transformer (DETR). In this paper, we introduce Feature Reweighting ANd Contrastive Learning NetworK (FRANCK), a novel SFOD framework specifically designed to perform query-centric feature enhancement for DETRs. FRANCK comprises four key components: (1) an Objectness Score-based Sample Reweighting (OSSR) module that computes attention-based objectness scores on multi-scale encoder feature maps, reweighting the detection loss to emphasize less-recognized regions; (2) a Contrastive Learning with Matching-based Memory Bank (CMMB) module that integrates multi-level features into memory banks, enhancing class-wise contrastive learning; (3) an Uncertainty-weighted Query-fused Feature Distillation (UQFD) module that improves feature distillation through prediction quality reweighting and query feature fusion; and (4) an improved self-training pipeline with a Dynamic Teacher Updating Interval (DTUI) that optimizes pseudo-label quality. By leveraging these components, FRANCK effectively adapts a source-pre-trained DETR model to a target domain with enhanced robustness and generalization. Extensive experiments on several widely used benchmarks demonstrate that our method achieves state-of-the-art performance, highlighting its effectiveness and compatibility with DETR-based SFOD models.

---

## 60. XGrasp: Gripper-Aware Grasp Detection with Multi-Gripper Data Generation

**论文链接:** [http://arxiv.org/abs/2510.11036v1](http://arxiv.org/abs/2510.11036v1)

**作者:** Yeonseo Lee, Jungwook Mun, Hyosup Shin, Guebin Hwang, Junhee Nam, Taeyeop Lee, Sungho Jo

**发布时间:** 2025-10-13

### GPT解析

### 总结

XGrasp是一个实时的夹爪感知抓取检测框架，能够高效处理多种夹爪配置，解决了传统机器人抓取方法仅针对单一夹爪类型的问题。

### 背景

大多数机器人抓取方法通常针对单一夹爪类型设计，这在需要多样化末端执行器的现实世界场景中限制了它们的适用性。

### 目的

开发一个能够处理多种夹爪配置的实时抓取检测框架，提高机器人在多样化环境中的抓取能力。

### 方法

通过系统性地为现有数据集添加多夹爪注释解决数据稀缺问题；采用分层两阶段架构，包括使用全局场景信息和夹爪规格确定最佳位置的抓取点预测器(GPP)，以及使用局部特征细化抓取角度和宽度的角度-宽度预测器(AWP)；在AWP模块中使用对比学习实现零样本泛化；模块化框架与视觉基础模型无缝集成。

### 主要发现

实验结果显示XGrasp在各种夹爪类型上具有竞争力的抓取成功率，同时与现有的夹爪感知方法相比，在推理速度上有了显著提高。

### 结论

XGrasp提供了一个有效的解决方案，使机器人能够适应不同的夹爪配置，并在性能和计算效率上优于现有方法。

### 翻译

大多数机器人抓取方法通常针对单一夹爪类型设计，这限制了它们在需要多样化末端执行器的现实世界场景中的适用性。我们提出了XGrasp，一个实时的夹爪感知抓取检测框架，能够高效处理多种夹爪配置。该方法通过系统性地为现有数据集添加多夹爪注释来解决数据稀缺问题。XGrasp采用分层两阶段架构。在第一阶段，抓取点预测器(GPP)使用全局场景信息和夹爪规格确定最佳位置。在第二阶段，角度-宽度预测器(AWP)使用局部特征细化抓取角度和宽度。AWP模块中的对比学习通过学习基本抓取特征，实现对未见夹爪的零样本泛化。模块化框架与视觉基础模型无缝集成，为未来的视觉语言能力提供途径。实验结果表明，在各种夹爪类型上具有竞争力的抓取成功率，同时与现有的夹爪感知方法相比，在推理速度上取得了显著提高。项目页面：https://sites.google.com/view/xgrasp


### 论文摘要

Most robotic grasping methods are typically designed for single gripper types, which limits their applicability in real-world scenarios requiring diverse end-effectors. We propose XGrasp, a real-time gripper-aware grasp detection framework that efficiently handles multiple gripper configurations. The proposed method addresses data scarcity by systematically augmenting existing datasets with multi-gripper annotations. XGrasp employs a hierarchical two-stage architecture. In the first stage, a Grasp Point Predictor (GPP) identifies optimal locations using global scene information and gripper specifications. In the second stage, an Angle-Width Predictor (AWP) refines the grasp angle and width using local features. Contrastive learning in the AWP module enables zero-shot generalization to unseen grippers by learning fundamental grasping characteristics. The modular framework integrates seamlessly with vision foundation models, providing pathways for future vision-language capabilities. The experimental results demonstrate competitive grasp success rates across various gripper types, while achieving substantial improvements in inference speed compared to existing gripper-aware methods. Project page: https://sites.google.com/view/xgrasp

---

## 61. A Joint Learning Approach to Hardware Caching and Prefetching

**论文链接:** [http://arxiv.org/abs/2510.10862v1](http://arxiv.org/abs/2510.10862v1)

**作者:** Samuel Yuan, Divyanshu Saxena, Jiayi Chen, Nihal Sharma, Aditya Akella

**发布时间:** 2025-10-13

**备注:** Accepted at ML for Systems Workshop at the 39th Conference on Neural  Information Processing Systems (NeurIPS 2025)

### GPT解析

### 总结

本文研究了硬件缓存领域中缓存替换和预取策略之间的相互依赖关系，提出了一种联合学习方法，通过共享表示来同时训练这两种策略，并展示了两种实现方法的初步结果。

### 背景

现代系统中已提出多种学习方法替代启发式算法用于调度、缓存等系统组件，这些模型利用多样特征、历史学习和行为预测来应对工作负载动态性和硬件发展。

### 目的

研究缓存替换和预取策略之间的双向相互依赖关系，提出并验证联合训练这两种策略的方法。

### 方法

提出基于特征共享表示的联合学习方法，包括两种实现方式：基于联合编码器和基于嵌入对比学习的方法。

### 主要发现

两种共享表示开发方法都展示了有希望的初步结果，证明了联合训练策略的有效性。

### 结论

为基于相互依赖关系的系统策略联合研究方向制定了未来研究议程。

### 翻译

已经提出了多种学习策略来替代现代系统中的调度、缓存和其他系统组件的启发式算法。通过利用多样特征、从历史趋势学习和预测未来行为，这类模型有望跟上不断增加的工作负载动态性和持续的硬件发展。然而，单独训练的策略在组合使用时可能仍然表现不佳。本文在硬件缓存领域研究了一个这样的实例——针对缓存替换和预取策略。我们认为这两种策略是双向相互依赖的，并论证了联合训练这两种策略的必要性。我们提出了一种联合学习方法，基于为两种策略使用的特征开发共享表示。我们介绍了两种开发这些共享表示的方法，一种基于联合编码器，另一种基于嵌入的对比学习，并展示了这两种方法都取得了有希望的初步结果。最后，我们为这一方向的未来研究制定了议程。


### 论文摘要

Several learned policies have been proposed to replace heuristics for scheduling, caching, and other system components in modern systems. By leveraging diverse features, learning from historical trends, and predicting future behaviors, such models promise to keep pace with ever-increasing workload dynamism and continuous hardware evolution. However, policies trained in isolation may still achieve suboptimal performance when placed together. In this paper, we inspect one such instance in the domain of hardware caching -- for the policies of cache replacement and prefetching. We argue that these two policies are bidirectionally interdependent and make the case for training the two jointly. We propose a joint learning approach based on developing shared representations for the features used by the two policies. We present two approaches to develop these shared representations, one based on a joint encoder and another based on contrastive learning of the embeddings, and demonstrate promising preliminary results for both of these. Finally, we lay down an agenda for future research in this direction.

---

## 62. SS-DPPN: A self-supervised dual-path foundation model for the generalizable cardiac audio representation

**论文链接:** [http://arxiv.org/abs/2510.10719v1](http://arxiv.org/abs/2510.10719v1)

**作者:** Ummy Maria Muna, Md Mehedi Hasan Shawon, Md Jobayer, Sumaiya Akter, Md Rakibul Hasan, Md. Golam Rabiul Alam

**发布时间:** 2025-10-12

### GPT解析

### 总结

本文提出了一种自监督双路径原型网络（SS-DPPN），用于从无标签数据中进行心脏音频表示和分类的基础模型，解决了监督式深度学习受限于专家标注数据稀缺性的问题。

### 背景

心音图的自动分析对心血管疾病的早期诊断至关重要，但监督式深度学习常受限于专家标注数据的稀缺性。

### 目的

开发一种能够从无标签数据中进行心脏音频表示和分类的基础模型，减少对专家标注数据的依赖。

### 方法

提出自监督双路径原型网络（SS-DPPN），采用基于双路径对比学习的架构，同时处理一维波形和二维频谱图，使用新型混合损失函数；下游任务采用基于度量学习的原型网络方法，提高敏感性并产生校准良好且可信赖的预测。

### 主要发现

SS-DPPN在四个心脏音频基准测试上达到最先进性能；在数据效率方面表现出色，标注数据减少三倍；学习到的表示在肺部声音分类和心率估计任务上成功泛化。

### 结论

实验和发现验证了SS-DPPN作为生理信号的一种强大、可靠且可扩展的基础模型。

### 翻译

心音图的自动分析对心血管疾病的早期诊断至关重要，然而监督式深度学习常受限于专家标注数据的稀缺性。在本文中，我们提出了自监督双路径原型网络（SS-DPPN），这是一种用于从无标签数据进行心脏音频表示和分类的基础模型。该框架引入了一种基于双路径对比学习的架构，同时使用新型混合损失函数处理一维波形和二维频谱图。对于下游任务，使用基于度量学习的原型网络方法，提高了敏感性并产生校准良好且可信赖的预测。SS-DPPN在四个心脏音频基准测试上实现了最先进的性能。该框架在数据效率方面表现出色，与全监督模型相比，标注数据减少三倍。最后，学习到的表示在肺部声音分类和心率估计任务上成功泛化。我们的实验和发现验证了SS-DPPN作为生理信号的一种强大、可靠且可扩展的基础模型。


### 论文摘要

The automated analysis of phonocardiograms is vital for the early diagnosis of cardiovascular disease, yet supervised deep learning is often constrained by the scarcity of expert-annotated data. In this paper, we propose the Self-Supervised Dual-Path Prototypical Network (SS-DPPN), a foundation model for cardiac audio representation and classification from unlabeled data. The framework introduces a dual-path contrastive learning based architecture that simultaneously processes 1D waveforms and 2D spectrograms using a novel hybrid loss. For the downstream task, a metric-learning approach using a Prototypical Network was used that enhances sensitivity and produces well-calibrated and trustworthy predictions. SS-DPPN achieves state-of-the-art performance on four cardiac audio benchmarks. The framework demonstrates exceptional data efficiency with a fully supervised model on three-fold reduction in labeled data. Finally, the learned representations generalize successfully across lung sound classification and heart rate estimation. Our experiments and findings validate SS-DPPN as a robust, reliable, and scalable foundation model for physiological signals.

---

## 63. Collaborative Text-to-Image Generation via Multi-Agent Reinforcement Learning and Semantic Fusion

**论文链接:** [http://arxiv.org/abs/2510.10633v1](http://arxiv.org/abs/2510.10633v1)

**作者:** Jiabao Shi, Minfeng Qi, Lefeng Zhang, Di Wang, Yingjie Zhao, Ziying Li, Yalong Xing, Ningran Li

**发布时间:** 2025-10-12

**备注:** 16 pages, 13 figures

### GPT解析

### 总结

该研究提出了一种多智能体强化学习框架，用于解决多模态文本到图像生成中语义对齐和专业细节保持的难题。

### 背景

多模态文本到图像生成面临保持语义对齐和专业级别细节的困难，特别是在不同视觉领域之间。

### 目的

开发一个能够协调领域专业化智能体的多智能体强化学习框架，提升跨视觉域的语义对齐和专业细节生成能力。

### 方法

构建包含文本增强模块和图像生成模块的双耦合子系统，每个模块配备多模态集成组件；使用近端策略优化(PPO)训练智能体，通过复合奖励函数平衡语义相似性、视觉质量和内容多样性；采用对比学习、双向注意和迭代反馈实现跨模态对齐。

### 主要发现

系统显著丰富了生成内容（字数增加1614%），同时降低ROUGE-1分数69.7%；基于Transformer的融合方法获得最高复合分数(0.521)，但存在稳定性问题；多模态集成一致性中等(0.444-0.481)，反映跨模态语义基础的持续挑战。

### 结论

协作的、专业化驱动的架构在推进可靠的多模态生成系统方面显示出广阔前景。

### 翻译

多模态文本到图像生成仍然受到在多样化视觉领域中保持语义对齐和专业级别细节的困难限制。我们提出了一种多智能体强化学习框架，在两个耦合子系统中协调领域专业化智能体（例如，专注于建筑、肖像和风景图像）：文本增强模块和图像生成模块，每个模块都增加了多模态集成组件。智能体在复合奖励函数下使用近端策略优化(PPO)进行训练，该函数平衡语义相似性、语言视觉质量和内容多样性。通过对比学习、双向注意和文本与图像之间的迭代反馈来强制跨模态对齐。在六种实验设置中，我们的系统显著丰富了生成内容（字数增加了1614%），同时将ROUGE-1分数降低了69.7%。在融合方法中，基于Transformer的策略获得最高复合分数(0.521)，尽管偶尔存在稳定性问题。多模态集成产生中等一致性（范围从0.444到0.481），反映了跨模态语义基础的持续挑战。这些研究结果强调了协作的、专业化驱动的架构在推进可靠多模态生成系统方面的前景。


### 论文摘要

Multimodal text-to-image generation remains constrained by the difficulty of maintaining semantic alignment and professional-level detail across diverse visual domains. We propose a multi-agent reinforcement learning framework that coordinates domain-specialized agents (e.g., focused on architecture, portraiture, and landscape imagery) within two coupled subsystems: a text enhancement module and an image generation module, each augmented with multimodal integration components. Agents are trained using Proximal Policy Optimization (PPO) under a composite reward function that balances semantic similarity, linguistic visual quality, and content diversity. Cross-modal alignment is enforced through contrastive learning, bidirectional attention, and iterative feedback between text and image. Across six experimental settings, our system significantly enriches generated content (word count increased by 1614%) while reducing ROUGE-1 scores by 69.7%. Among fusion methods, Transformer-based strategies achieve the highest composite score (0.521), despite occasional stability issues. Multimodal ensembles yield moderate consistency (ranging from 0.444 to 0.481), reflecting the persistent challenges of cross-modal semantic grounding. These findings underscore the promise of collaborative, specialization-driven architectures for advancing reliable multimodal generative systems.

---

## 64. Multi-Granularity Sequence Denoising with Weakly Supervised Signal for Sequential Recommendation

**论文链接:** [http://arxiv.org/abs/2510.10564v1](http://arxiv.org/abs/2510.10564v1)

**作者:** Liang Li, Zhou Yang, Xiaofei Zhu

**发布时间:** 2025-10-12

### GPT解析

### 总结

本文提出了一种名为MGSD-WSS的多粒度序列去噪方法，通过弱监督信号同时处理项目粒度和兴趣粒度噪声，显著提升了序列推荐性能。

### 背景

序列推荐基于用户历史交互序列预测下一个项目，但历史序列中的不相关噪声项严重影响推荐效果。现有无监督方法缺乏明确噪声标签，容易误判用户感兴趣项目，且仅关注项目粒度噪声而忽略兴趣粒度噪声。

### 目的

解决现有方法中噪声标签缺乏导致的误判问题，以及忽略兴趣粒度噪声的问题，实现更全面有效的序列去噪推荐。

### 方法

提出MGSD-WSS方法，包含多核感知器模块映射序列到共同表示空间，利用弱监督信号识别噪声项，通过带噪声加权对比学习的项目粒度去噪模块处理项目噪声，提取目标兴趣表示处理兴趣粒度噪声，最后基于去噪后的项目和兴趣表示预测下一项目。

### 主要发现

在五个数据集上的大量实验表明，所提出的方法显著优于最先进的序列推荐和去噪模型。

### 结论

MGSD-WSS能有效解决现有序列推荐中的噪声问题，提升推荐性能，代码已公开在GitHub上。

### 翻译

Sequential recommendation aims to predict the next item based on user interests in historical interaction sequences. Historical interaction sequences often contain irrelevant noisy items, which significantly hinders the performance of recommendation systems. Existing research employs unsupervised methods that indirectly identify item-granularity irrelevant noise by predicting the ground truth item. Since these methods lack explicit noise labels, they are prone to misidentify users' interested items as noise. Additionally, while these methods focus on removing item-granularity noise driven by the ground truth item, they overlook interest-granularity noise, limiting their ability to perform broader denoising based on user interests. To address these issues, we propose Multi-Granularity Sequence Denoising with Weakly Supervised Signal for Sequential Recommendation (MGSD-WSS). MGSD-WSS first introduces the Multiple Gaussian Kernel Perceptron module to map the original and enhance sequence into a common representation space and utilizes weakly supervised signals to accurately identify noisy items in the historical interaction sequence. Subsequently, it employs the item-granularity denoising module with noise-weighted contrastive learning to obtain denoised item representations. Then, it extracts target interest representations from the ground truth item and applies noise-weighted contrastive learning to obtain denoised interest representations. Finally, based on the denoised item and interest representations, MGSD-WSS predicts the next item. Extensive experiments on five datasets demonstrate that the proposed method significantly outperforms state-of-the-art sequence recommendation and denoising models. Our code is available at https://github.com/lalunex/MGSD-WSS.


### 论文摘要

Sequential recommendation aims to predict the next item based on user interests in historical interaction sequences. Historical interaction sequences often contain irrelevant noisy items, which significantly hinders the performance of recommendation systems. Existing research employs unsupervised methods that indirectly identify item-granularity irrelevant noise by predicting the ground truth item. Since these methods lack explicit noise labels, they are prone to misidentify users' interested items as noise. Additionally, while these methods focus on removing item-granularity noise driven by the ground truth item, they overlook interest-granularity noise, limiting their ability to perform broader denoising based on user interests. To address these issues, we propose Multi-Granularity Sequence Denoising with Weakly Supervised Signal for Sequential Recommendation(MGSD-WSS). MGSD-WSS first introduces the Multiple Gaussian Kernel Perceptron module to map the original and enhance sequence into a common representation space and utilizes weakly supervised signals to accurately identify noisy items in the historical interaction sequence. Subsequently, it employs the item-granularity denoising module with noise-weighted contrastive learning to obtain denoised item representations. Then, it extracts target interest representations from the ground truth item and applies noise-weighted contrastive learning to obtain denoised interest representations. Finally, based on the denoised item and interest representations, MGSD-WSS predicts the next item. Extensive experiments on five datasets demonstrate that the proposed method significantly outperforms state-of-the-art sequence recommendation and denoising models. Our code is available at https://github.com/lalunex/MGSD-WSS.

---

## 65. VOLTAGE: A Versatile Contrastive Learning based OCR Methodology for ultra low-resource scripts through Auto Glyph Feature Extraction

**论文链接:** [http://arxiv.org/abs/2510.10490v1](http://arxiv.org/abs/2510.10490v1)

**作者:** Prawaal Sharma, Poonam Goyal, Vidisha Sharma, Navneet Goyal

**发布时间:** 2025-10-12

**DOI:** 10.18653/v1/2024.eacl-long.53

**备注:** 9 Pages, Plus Appendices, EACL 2024

### GPT解析

### 总结

研究提出了VOLTAGE，一种基于对比学习的OCR方法，用于低资源语言（特别是Takri文字）的数字化保护，以防止语言灭绝。

### 背景

全球7000种语言中有2500种被列为濒危语言，语言流失导致传统智慧、民间文学和社区本质的丧失。低资源语言面临更高的灭绝风险，而缺乏针对低资源语言的无监督OCR方法是阻碍其数字化的原因之一。

### 目的

为低资源语言提供数字包容性，避免语言灭绝，开发适用于低资源语言的OCR方法。

### 方法

提出VOLTAGE - 一种基于对比学习的OCR方法，利用自动字形特征推荐进行基于聚类的标记，使用图像转换和生成对抗网络增加标记数据的多样性和数量，使用Takri文字进行设计，并在多种印度文字上测试。

### 主要发现

在Takri文字上实现了95%的机器印刷样本和87%的手写样本的准确率，进行了基线和消融研究，为Takri构建了下游应用案例，证明了工作的实用性。

### 结论

VOLTAGE方法具有普适性，适用于不同类型的印度文字，有助于保护低资源语言的数字化。

### 翻译

联合国教科文组织已将全球使用的7000种语言中的2500种列为濒危语言。语言的流失导致传统智慧、民间文学和使用该语言的社区本质的丧失。因此，必须为这些语言提供数字包容性，避免其灭绝。低资源语言面临更高的灭绝风险。缺乏针对低资源语言的无监督光学字符识别方法是阻碍其数字化的原因之一。我们提出了VOLTAGE - 一种基于对比学习的OCR方法，利用自动字形特征推荐进行基于聚类的标记。我们使用图像转换和生成对抗网络来增加标记数据的多样性和数量。VOLTAGE是使用Takri文字设计的 - 这是一组在16至20世纪印度喜马拉雅地区使用的文字。我们展示了Takri文字以及其他印度文字（包括低资源和高资源）的结果，以证明该方法的普适性。在Takri文字上，机器印刷样本的准确率达到95%，手写样本达到87%。我们进行了基线和消融研究，并为Takri构建了下游应用案例，证明了我们工作的实用性。


### 论文摘要

UNESCO has classified 2500 out of 7000 languages spoken worldwide as endangered. Attrition of a language leads to loss of traditional wisdom, folk literature, and the essence of the community that uses it. It is therefore imperative to bring digital inclusion to these languages and avoid its extinction. Low resource languages are at a greater risk of extinction. Lack of unsupervised Optical Character Recognition(OCR) methodologies for low resource languages is one of the reasons impeding their digital inclusion. We propose VOLTAGE - a contrastive learning based OCR methodology, leveraging auto-glyph feature recommendation for cluster-based labelling. We augment the labelled data for diversity and volume using image transformations and Generative Adversarial Networks. Voltage has been designed using Takri - a family of scripts used in 16th to 20th century in the Himalayan regions of India. We present results for Takri along with other Indic scripts (both low and high resource) to substantiate the universal behavior of the methodology. An accuracy of 95% for machine printed and 87% for handwritten samples on Takri script has been achieved. We conduct baseline and ablation studies along with building downstream use cases for Takri, demonstrating the usefulness of our work.

---

## 66. Complementary and Contrastive Learning for Audio-Visual Segmentation

**论文链接:** [http://arxiv.org/abs/2510.10051v1](http://arxiv.org/abs/2510.10051v1)

**作者:** Sitong Gong, Yunzhi Zhuge, Lu Zhang, Pingping Zhang, Huchuan Lu

**发布时间:** 2025-10-11

**备注:** Accepted to IEEE Transactions on Multimedia

### GPT解析

### 总结

本文提出了一种名为CCFormer的新型音频-视觉分割框架，结合了CNN和Transformer的优势，能够有效处理局部和全局信息并全面捕捉空间-时间上下文。

### 背景

音频-视觉分割(AVS)旨在生成与物体听觉信号相关的像素级分割图。传统CNN方法通过基本操作处理音频-视觉交互但受限于局部感受野；较新的Transformer方法将听觉作为查询增强帧内协作，但难以充分提取多模态系数和时间动态。

### 目的

克服现有方法的局限性，开发一个能同时处理局部和全局信息并全面捕捉空间-时间上下文的音频-视觉分割框架。

### 方法

提出CCFormer框架，包含：1)早期集成模块(EIM)，采用并行双边架构融合多尺度视觉特征与音频数据；2)多查询Transformer模块(MTM)，动态赋予音频查询学习能力并建模帧级和视频级关系；3)双模态对比学习(BCL)，促进统一特征空间中跨模态对齐。

### 主要发现

通过有效结合这些设计，该方法在S4、MS3和AVSS数据集上设立了新的最先进基准，源代码和模型权重将在GitHub公开。

### 结论

CCFormer框架通过创新设计成功解决了传统方法和现有Transformer方法在音频-视觉分割中的局限性，实现了最先进的性能。

### 翻译

音频-视觉分割(AVS)旨在生成与物体听觉信号相关的像素级分割图。该领域已取得显著进展，许多基于CNN和Transformer的方法提高了分割的准确性和鲁棒性。传统CNN方法通过填充和乘法等基本操作处理音频-视觉交互，但受限于CNN的有限局部感受野。较新的基于Transformer的方法将听觉线索作为查询，利用注意力机制增强帧内音频-视觉协作，但通常难以充分提取多模态系数和时间动态。为克服这些局限性，我们提出了互补和对比Transformer(CCFormer)，这是一个新型框架，擅长处理局部和全局信息，并全面捕捉空间-时间上下文。我们的CCFormer首先采用早期集成模块(EIM)，该模块采用并行双边架构，将多尺度视觉特征与音频数据融合，以增强跨模态互补性。为了提取帧内空间特征并促进时间相干性的感知，我们引入了多查询Transformer模块(MTM)，该模块动态赋予音频查询学习能力，同时建模帧级和视频级关系。此外，我们提出了双模态对比学习(BCL)，以促进统一特征空间中跨模态的对齐。通过有效结合这些设计，我们的方法在S4、MS3和AVSS数据集上设立了新的最先进基准。我们的源代码和模型权重将在https://github.com/SitongGong/CCFormer公开提供。


### 论文摘要

Audio-Visual Segmentation (AVS) aims to generate pixel-wise segmentation maps that correlate with the auditory signals of objects. This field has seen significant progress with numerous CNN and Transformer-based methods enhancing the segmentation accuracy and robustness. Traditional CNN approaches manage audio-visual interactions through basic operations like padding and multiplications but are restricted by CNNs' limited local receptive field. More recently, Transformer-based methods treat auditory cues as queries, utilizing attention mechanisms to enhance audio-visual cooperation within frames. Nevertheless, they typically struggle to extract multimodal coefficients and temporal dynamics adequately. To overcome these limitations, we present the Complementary and Contrastive Transformer (CCFormer), a novel framework adept at processing both local and global information and capturing spatial-temporal context comprehensively. Our CCFormer initiates with the Early Integration Module (EIM) that employs a parallel bilateral architecture, merging multi-scale visual features with audio data to boost cross-modal complementarity. To extract the intra-frame spatial features and facilitate the perception of temporal coherence, we introduce the Multi-query Transformer Module (MTM), which dynamically endows audio queries with learning capabilities and models the frame and video-level relations simultaneously. Furthermore, we propose the Bi-modal Contrastive Learning (BCL) to promote the alignment across both modalities in the unified feature space. Through the effective combination of those designs, our method sets new state-of-the-art benchmarks across the S4, MS3 and AVSS datasets. Our source code and model weights will be made publicly available at https://github.com/SitongGong/CCFormer

---

## 67. Enhancing Faithfulness in Abstractive Summarization via Span-Level Fine-Tuning

**论文链接:** [http://arxiv.org/abs/2510.09915v1](http://arxiv.org/abs/2510.09915v1)

**作者:** Sicong Huang, Qianqi Yan, Shengze Wang, Ian Lane

**发布时间:** 2025-10-10

### GPT解析

### 总结

本研究探讨了通过微调大型语言模型来减少生成摘要中的不忠实片段，引入了一种新的数据集和三种微调技术，实验结果表明所有方法都能提高摘要忠实度，其中似然训练最有效。

### 背景

使用大型语言模型进行摘要生成已成为信息压缩的重要工具，但这些模型有时会产生不忠实的摘要，包含单词、短语或概念级别的幻觉。

### 目的

研究微调策略以减少生成摘要中不忠实片段的出现，提高摘要的忠实度。

### 方法

首先使用多种LLM为训练集中的源文档自动生成摘要，然后使用GPT-4o标注检测到的片段级幻觉；利用这些标注，使用无幻觉摘要和标注的不忠实片段微调LLM；引入一个包含忠实和不忠实摘要以及片段级标签的新数据集；评估三种微调技术：梯度上升、似然训练和任务向量否定。

### 主要发现

所有三种方法都成功利用片段级标注提高了忠实度，其中似然训练最有效。

### 结论

通过使用片段级标注进行微调，可以有效提高LLM生成摘要的忠实度。

### 翻译

使用大型语言模型进行抽象式摘要已成为信息压缩的重要工具。然而，尽管这些模型能够生成流畅的摘要，但有时会产生不忠实的摘要，在单词、短语或概念层面引入幻觉。现有的缓解策略，如后处理校正或使用合成生成的负样本进行对比学习，无法完全解决LLM生成摘要中可能出现的各种错误。在本文中，我们研究了微调策略以减少生成摘要中不忠实片段的出现。首先，我们使用多种LLM为训练集中的源文档自动生成摘要，然后使用GPT-4o标注其检测到的片段级幻觉。利用这些标注，我们使用无幻觉摘要和标注的不忠实片段来微调LLM，以提高模型的忠实度。在本文中，我们引入了一个包含忠实和不忠实摘要以及片段级标签的新数据集，并评估了三种微调技术来提高LLM生成摘要的忠实度：梯度上升、似然训练和任务向量否定。实验结果表明，所有三种方法都成功利用片段级标注提高了忠实度，其中似然训练最为有效。


### 论文摘要

Abstractive summarization using large language models (LLMs) has become an essential tool for condensing information. However, despite their ability to generate fluent summaries, these models sometimes produce unfaithful summaries, introducing hallucinations at the word, phrase, or concept level. Existing mitigation strategies, such as post-processing corrections or contrastive learning with synthetically generated negative samples, fail to fully address the diverse errors that can occur in LLM-generated summaries. In this paper, we investigate fine-tuning strategies to reduce the occurrence of unfaithful spans in generated summaries. First, we automatically generate summaries for the set of source documents in the training set with a variety of LLMs and then use GPT-4o to annotate any hallucinations it detects at the span-level. Leveraging these annotations, we fine-tune LLMs with both hallucination-free summaries and annotated unfaithful spans to enhance model faithfulness. In this paper, we introduce a new dataset that contains both faithful and unfaithful summaries with span-level labels and we evaluate three techniques to fine-tuning a LLM to improve the faithfulness of the resulting summarization: gradient ascent, unlikelihood training, and task vector negation. Experimental results show that all three approaches successfully leverage span-level annotations to improve faithfulness, with unlikelihood training being the most effective.

---

## 68. One Sentence, Two Embeddings: Contrastive Learning of Explicit and Implicit Semantic Representations

**论文链接:** [http://arxiv.org/abs/2510.09293v1](http://arxiv.org/abs/2510.09293v1)

**作者:** Kohei Oda, Po-Min Chuang, Kiyoaki Shirai, Natthawut Kertkeidkachorn

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文提出了一种名为DualCSE的句子嵌入方法，通过为每个句子分配两个嵌入向量（分别代表显式语义和隐式语义）来解决传统方法难以捕捉句子隐式语义的问题。

### 背景

句子嵌入方法虽然取得了显著进展，但仍难以捕捉句子中的隐式语义，这归因于传统方法为每个句子只分配一个向量的固有局限性。

### 目的

克服传统句子嵌入方法的局限性，开发一种能够同时捕捉句子显式和隐式语义的方法。

### 方法

提出DualCSE方法，为每个句子分配两个嵌入：一个代表显式语义，另一个代表隐式语义，这两个嵌入共存于共享空间中，可根据特定任务需求选择合适的语义表示。

### 主要发现

实验结果表明，DualCSE能够有效编码句子的显式和隐式含义，并提高下游任务（如信息检索和文本分类）的性能。

### 结论

DualCSE是一种有效的句子嵌入方法，通过双嵌入机制同时捕捉句子的显式和隐式语义，提升了下游任务的性能。

### 翻译

句子嵌入方法已经取得了显著进展，但仍然难以捕捉句子中的隐式语义。这可以归因于传统句子嵌入方法的固有局限性，即为每个句子只分配一个向量。为了克服这一局限性，我们提出了DualCSE，一种为每个句子分配两个嵌入的句子嵌入方法：一个代表显式语义，另一个代表隐式语义。这些嵌入共存于共享空间中，使得能够为特定目的（如信息检索和文本分类）选择所需的语义。实验结果表明，DualCSE能够有效编码显式和隐式含义，并提高下游任务的性能。


### 论文摘要

Sentence embedding methods have made remarkable progress, yet they still struggle to capture the implicit semantics within sentences. This can be attributed to the inherent limitations of conventional sentence embedding methods that assign only a single vector per sentence. To overcome this limitation, we propose DualCSE, a sentence embedding method that assigns two embeddings to each sentence: one representing the explicit semantics and the other representing the implicit semantics. These embeddings coexist in the shared space, enabling the selection of the desired semantics for specific purposes such as information retrieval and text classification. Experimental results demonstrate that DualCSE can effectively encode both explicit and implicit meanings and improve the performance of the downstream task.

---

## 69. Generative Data Augmentation in Graph Contrastive Learning for Recommendation

**论文链接:** [http://arxiv.org/abs/2510.09129v1](http://arxiv.org/abs/2510.09129v1)

**作者:** Yansong Wang, Qihui Lin, Junjie Huang, Tao Jia

**发布时间:** 2025-10-10

**DOI:** 10.1145/3746252.3761248

**备注:** The 34th ACM International Conference on Information and Knowledge  Management

### GPT解析

### 总结

本文提出了一种名为GDA4Rec的新颖框架，用于在推荐系统中通过图对比学习和生成数据增强来学习高质量的用户-物品嵌入表示。

### 背景

推荐系统已成为各种在线平台不可或缺的组成部分，但面临从稀疏用户-物品交互中学习有效嵌入表示的基本挑战。

### 目的

解决现有随机数据增强方法在对比学习中常常改变原始语义信息的问题，提供高质量的增强视图和强大的自监督信号。

### 方法

GDA4Rec采用噪声生成模块利用深度生成模型近似原始数据分布进行数据增强，提取物品互补矩阵表征物品间潜在相关性，并使用整合推荐、数据增强和对比学习的联合目标函数。

### 主要发现

在三个公共数据集上的广泛实验证明了该模型的优越性。

### 结论

GDA4Rec通过生成高质量增强视图和提供强大自监督信号，解决了推荐系统中从稀疏交互学习有效嵌入的挑战。

### 翻译

推荐系统已成为各种在线平台中不可或缺的组成部分，从电子商务到流媒体服务。该领域的一个基本挑战是如何从稀疏的用户-物品交互中学习有效的嵌入表示。虽然对比学习最近已成为解决这个问题的有前景的方法，但通过大多数现有随机数据增强方法为对比学习生成增强视图常常导致原始语义信息的改变。在本文中，我们提出了一个新颖的框架GDA4Rec（用于推荐的图对比学习中的生成数据增强），以生成高质量的增强视图并提供强大的自监督信号。具体来说，我们采用了一个噪声生成模块，利用深度生成模型来近似原始数据的分布用于数据增强。此外，GDA4Rec进一步提取了一个物品互补矩阵来表征物品之间的潜在相关性，并提供额外的自监督信号。最后，使用一个整合了推荐、数据增强和对比学习的联合目标函数，强制模型学习更有效和信息量更大的嵌入。在三个公共数据集上进行了广泛的实验，以证明该模型的优越性。代码可在以下网址获取：https://github.com/MrYansong/GDA4Rec。


### 论文摘要

Recommendation systems have become indispensable in various online platforms, from e-commerce to streaming services. A fundamental challenge in this domain is learning effective embeddings from sparse user-item interactions. While contrastive learning has recently emerged as a promising solution to this issue, generating augmented views for contrastive learning through most existing random data augmentation methods often leads to the alteration of original semantic information. In this paper, we propose a novel framework, GDA4Rec (Generative Data Augmentation in graph contrastive learning for Recommendation) to generate high-quality augmented views and provide robust self-supervised signals. Specifically, we employ a noise generation module that leverages deep generative models to approximate the distribution of original data for data augmentation. Additionally, GDA4Rec further extracts an item complement matrix to characterize the latent correlations between items and provide additional self-supervised signals. Lastly, a joint objective that integrates recommendation, data augmentation and contrastive learning is used to enforce the model to learn more effective and informative embeddings. Extensive experiments are conducted on three public datasets to demonstrate the superiority of the model. The code is available at: https://github.com/MrYansong/GDA4Rec.

---

## 70. A Unified Biomedical Named Entity Recognition Framework with Large Language Models

**论文链接:** [http://arxiv.org/abs/2510.08902v1](http://arxiv.org/abs/2510.08902v1)

**作者:** Tengxiao Lv, Ling Luo, Juntao Li, Yanhua Wang, Yuchen Pan, Chao Liu, Yanan Wang, Yan Jiang, Huiyi Lv, Yuanyuan Sun, Jian Wang, Hongfei Lin

**发布时间:** 2025-10-10

**备注:** Accepted as a short paper at BIBM2025

### GPT解析

### 总结

本文提出了一种基于大型语言模型的统一生物医学命名实体识别框架，解决了嵌套实体、实体边界模糊和跨语言泛化问题，在多个数据集上实现了最先进性能。

### 背景

生物医学命名实体识别对医学信息提取和知识发现至关重要，但现有方法在处理嵌套实体、实体边界模糊和跨语言泛化方面存在困难。

### 目的

开发一个基于大型语言模型的统一生物医学命名实体识别框架，提高识别准确性和跨语言泛化能力。

### 方法

将BioNER重新表述为文本生成任务，设计符号标记策略处理扁平和嵌套实体，通过中英文数据集进行双语联合微调，引入基于对比学习的实体选择器过滤错误预测。

### 主要发现

在四个基准数据集和两个未见语料库上的实验结果表明，该方法实现了最先进的性能，并在跨语言场景下展现出强大的零样本泛化能力。

### 结论

所提出的基于大型语言模型的BioNER框架能有效处理复杂的生物医学实体识别任务，具有出色的跨语言泛化能力。

### 翻译

准确的生物医学命名实体识别对医学信息提取和知识发现至关重要。然而，现有方法通常难以处理嵌套实体、实体边界模糊和跨语言泛化问题。在本文中，我们提出了一种基于大型语言模型的统一生物医学命名实体识别框架。我们首先将BioNER重新表述为文本生成任务，并设计了一种符号标记策略，通过明确的边界标注联合处理扁平实体和嵌套实体。为了增强多语言和多任务泛化能力，我们在多个中英文数据集上进行双语联合微调。此外，我们引入了一种基于对比学习的实体选择器，利用边界敏感的正负样本过滤不正确或虚假的预测。在四个基准数据集和两个未见语料库上的实验结果表明，我们的方法实现了最先进的性能，并在跨语言场景下展现出强大的零样本泛化能力。源代码可在https://github.com/dreamer-tx/LLMNER免费获取。


### 论文摘要

Accurate recognition of biomedical named entities is critical for medical information extraction and knowledge discovery. However, existing methods often struggle with nested entities, entity boundary ambiguity, and cross-lingual generalization. In this paper, we propose a unified Biomedical Named Entity Recognition (BioNER) framework based on Large Language Models (LLMs). We first reformulate BioNER as a text generation task and design a symbolic tagging strategy to jointly handle both flat and nested entities with explicit boundary annotation. To enhance multilingual and multi-task generalization, we perform bilingual joint fine-tuning across multiple Chinese and English datasets. Additionally, we introduce a contrastive learning-based entity selector that filters incorrect or spurious predictions by leveraging boundary-sensitive positive and negative samples. Experimental results on four benchmark datasets and two unseen corpora show that our method achieves state-of-the-art performance and robust zero-shot generalization across languages. The source codes are freely available at https://github.com/dreamer-tx/LLMNER.

---

## 71. On the Alignment Between Supervised and Self-Supervised Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.08852v1](http://arxiv.org/abs/2510.08852v1)

**作者:** Achleshwar Luthra, Priyadarsi Mishra, Tomer Galanti

**发布时间:** 2025-10-09

### GPT解析

### 总结

本研究探讨了自监督对比学习(CL)与负样本监督对比学习(NSCL)在表示层面的一致性，发现尽管两种方法的损失函数相似，但在训练过程中它们的表示确实保持对齐，而参数空间则可能出现指数级差异。

### 背景

自监督对比学习已取得显著成功，其表示能力可与监督预训练相媲美。最近理论表明，当类别数量增加时，CL损失函数近似于监督代理NSCL损失函数。

### 目的

探究CL和NSCL是否不仅在目标函数层面，而且在表示层面也保持一致，以及这种对齐如何随训练条件变化。

### 方法

分析在相同初始化、批量和增强条件下训练的CL和NSCL模型的表示对齐情况，提供理论证明和实验验证。

### 主要发现

1) CL和NSCL的表示保持相似，相似度矩阵在现实条件下保持接近；2) 提供了CKA和RSA等对齐度量的高概率保证；3) 对齐随类别数量增加、温度升高而改善，且依赖于批量大小；4) 参数空间耦合不稳定，权重差异可能随训练时间呈指数级增长；5) 实验验证显示CL-NSCL对齐随规模和温度增强，NSCL比其他监督目标更紧密跟踪CL。

### 结论

NSCL可作为自监督学习和监督学习之间的原则性桥梁，为理解自监督学习提供了理论框架。

### 翻译

自监督对比学习(CL)已取得显著的实证成功，其产生的表示通常能在下游任务上与监督预训练相媲美。最近的理论表明，当类别数量增加时，CL损失函数近似于监督代理NSCL损失函数。然而，这种损失层面的相似性留下了一个开放问题：CL和NSCL在训练过程中是否也在表示层面保持对齐，而不仅仅是在目标函数层面？我们通过分析在共享随机性(相同初始化、批量和增强)条件下训练的CL和NSCL模型的表示对齐来解决这个问题。首先，我们证明它们诱导的表示保持相似：具体而言，我们在现实条件下证明了CL和NSCL的相似度矩阵保持接近。我们的边界为CKA和RSA等对齐度量提供了高概率保证，并阐明了对齐如何随类别数量增加、温度升高而改善，以及其对批量大小的依赖性。相比之下，我们证明了参数空间耦合本质上是不稳定的：CL和NSCL权重之间的差异可能随训练时间呈指数级增长。最后，我们通过实验验证了这些预测，表明CL-NSCL对齐随规模和温度增强，NSCL比其他监督目标更紧密地跟踪CL。这使NSCL成为自监督学习和监督学习之间的原则性桥梁。我们的代码和项目页面可在[链接]获取。


### 论文摘要

Self-supervised contrastive learning (CL) has achieved remarkable empirical success, often producing representations that rival supervised pre-training on downstream tasks. Recent theory explains this by showing that the CL loss closely approximates a supervised surrogate, Negatives-Only Supervised Contrastive Learning (NSCL) loss, as the number of classes grows. Yet this loss-level similarity leaves an open question: {\em Do CL and NSCL also remain aligned at the representation level throughout training, not just in their objectives?}   We address this by analyzing the representation alignment of CL and NSCL models trained under shared randomness (same initialization, batches, and augmentations). First, we show that their induced representations remain similar: specifically, we prove that the similarity matrices of CL and NSCL stay close under realistic conditions. Our bounds provide high-probability guarantees on alignment metrics such as centered kernel alignment (CKA) and representational similarity analysis (RSA), and they clarify how alignment improves with more classes, higher temperatures, and its dependence on batch size. In contrast, we demonstrate that parameter-space coupling is inherently unstable: divergence between CL and NSCL weights can grow exponentially with training time.   Finally, we validate these predictions empirically, showing that CL-NSCL alignment strengthens with scale and temperature, and that NSCL tracks CL more closely than other supervised objectives. This positions NSCL as a principled bridge between self-supervised and supervised learning. Our code and project page are available at [\href{https://github.com/DLFundamentals/understanding_ssl_v2}{code}, \href{https://dlfundamentals.github.io/cl-nscl-representation-alignment/}{project page}].

---

## 72. Alignment, Mining and Fusion: Representation Alignment with Hard Negative Mining and Selective Knowledge Fusion for Medical Visual Question Answering

**论文链接:** [http://arxiv.org/abs/2510.08791v1](http://arxiv.org/abs/2510.08791v1)

**作者:** Yuanhao Zou, Zhaozheng Yin

**发布时间:** 2025-10-09

**备注:** CVPR2025 Paper

### GPT解析

### 总结

本文提出了一种医学视觉问答(Med-VQA)框架，通过统一模态对齐、难例挖掘和门控交叉注意力模块解决现有方法的局限性，并在多个标准数据集上实现了最先进的性能。

### 背景

医学视觉问答(Med-VQA)需要对医学图像和文本问题有深入理解，尽管近期基于医学视觉语言预训练(Med-VLP)的方法表现良好，但仍存在模态对齐不统一、难例探索不足以及知识融合可能引入无关信息等问题。

### 目的

开发一个框架解决医学视觉问答中的模态对齐、难例处理和知识融合挑战。

### 方法

提出三个关键贡献：(1)统一解决方案处理多级别、多模态、多视图和多阶段的异构模态对齐，利用对比学习和最优传输理论；(2)难例挖掘方法，使用软标签进行多模态对齐并强制区分难例对；(3)门控交叉注意力模块，将答案词汇作为先验知识集成并选择相关信息。

### 主要发现

该框架在RAD-VQA、SLAKE、PathVQA和VQA-2019等广泛使用的Med-VQA数据集上超越了之前的最佳性能。

### 结论

通过统一模态对齐策略、有效的难例挖掘机制和门控交叉注意力模块，该框架显著提升了医学视觉问答任务的性能。

### 翻译

医学视觉问答(Med-VQA)是一个具有挑战性的任务，需要对医学图像和文本问题有深入理解。尽管近期利用医学视觉语言预训练(Med-VLP)的工作已在Med-VQA任务上展现出强大性能，但仍没有统一的模态对齐解决方案，且难例问题尚未得到充分探索。此外，Med-VQA常用的知识融合技术可能引入无关信息。在这项工作中，我们通过三个关键贡献提出一个框架来解决这些挑战：(1)一种统一解决方案，用于处理多级别、多模态、多视图和多阶段的异构模态对齐，利用对比学习和最优传输理论等方法；(2)一种难例挖掘方法，使用软标签进行多模态对齐，并强制区分难例对；(3)一种用于Med-VQA的门控交叉注意力模块，将答案词汇作为先验知识集成，并从中选择相关信息。我们的框架在广泛使用的Med-VQA数据集(如RAD-VQA、SLAKE、PathVQA和VQA-2019)上超越了之前的最佳性能。


### 论文摘要

Medical Visual Question Answering (Med-VQA) is a challenging task that requires a deep understanding of both medical images and textual questions. Although recent works leveraging Medical Vision-Language Pre-training (Med-VLP) have shown strong performance on the Med-VQA task, there is still no unified solution for modality alignment, and the issue of hard negatives remains under-explored. Additionally, commonly used knowledge fusion techniques for Med-VQA may introduce irrelevant information. In this work, we propose a framework to address these challenges through three key contributions: (1) a unified solution for heterogeneous modality alignments across multiple levels, modalities, views, and stages, leveraging methods like contrastive learning and optimal transport theory; (2) a hard negative mining method that employs soft labels for multi-modality alignments and enforces the hard negative pair discrimination; and (3) a Gated Cross-Attention Module for Med-VQA that integrates the answer vocabulary as prior knowledge and selects relevant information from it. Our framework outperforms the previous state-of-the-art on widely used Med-VQA datasets like RAD-VQA, SLAKE, PathVQA and VQA-2019.

---

## 73. Lecture Notes on Verifying Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.11617v1](http://arxiv.org/abs/2510.11617v1)

**作者:** François Schwarzentruber

**发布时间:** 2025-10-13

### GPT解析

### 总结

这篇讲义回顾了图神经网络、Weisfeiler-Lehman测试与逻辑之间的联系，提出了一种包含计数模态的模态逻辑，用于图神经网络验证，并描述了相应的可满足性问题算法。

### 背景

图神经网络、Weisfeiler-Lehman测试与一阶逻辑、分级模态逻辑之间的联系

### 目的

提出一种模态逻辑，其中计数模态以线性不等式形式出现，用于解决图神经网络的验证任务

### 方法

描述了该逻辑可满足性问题的算法，该方法基于普通模态逻辑的tableau方法，并扩展了对无量化子句布尔代数与Presburger算术的推理

### 主要发现

计数模态可以以线性不等式形式出现在模态逻辑中，用于解决图神经网络验证问题

### 结论

通过扩展的tableau方法，可以有效地解决所提出模态逻辑的可满足性问题

### 翻译

在这些讲义中，我们首先回顾了图神经网络、Weisfeiler-Lehman测试与一阶逻辑和分级模态逻辑等逻辑之间的联系。然后，我们提出了一个模态逻辑，其中计数模态以线性不等式的形式出现，用于解决图神经网络的验证任务。我们描述了该逻辑可满足性问题的算法。该算法受到普通模态逻辑的tableau方法的启发，并扩展了对无量化子句布尔代数与Presburger算术的推理。


### 论文摘要

In these lecture notes, we first recall the connection between graph neural networks, Weisfeiler-Lehman tests and logics such as first-order logic and graded modal logic. We then present a modal logic in which counting modalities appear in linear inequalities in order to solve verification tasks on graph neural networks. We describe an algorithm for the satisfiability problem of that logic. It is inspired from the tableau method of vanilla modal logic, extended with reasoning in quantifier-free fragment Boolean algebra with Presburger arithmetic.

---

## 74. Multi-View Graph Feature Propagation for Privacy Preservation and Feature Sparsity

**论文链接:** [http://arxiv.org/abs/2510.11347v1](http://arxiv.org/abs/2510.11347v1)

**作者:** Etzion Harari, Moshe Unger

**发布时间:** 2025-10-13

### GPT解析

### 总结

论文提出了一种新颖的多视图特征传播(MFP)框架，用于解决图神经网络在节点分类任务中因特征稀疏和隐私问题导致的性能下降。该框架通过将特征划分为多个添加高斯噪声的视图，独立传播信息并聚合结果，提高了分类性能同时保护隐私。

### 背景

图神经网络(GNNs)在节点分类任务中表现出色，但其效果通常依赖于完整的节点特征。在现实场景中，特征矩阵往往高度稀疏或包含敏感信息，导致性能下降和隐私风险增加。直接暴露信息还可能导致意外数据泄露，使攻击者能够推断敏感信息。

### 目的

解决特征稀疏问题，提高节点分类性能，同时促进隐私保护，平衡效用与隐私的关系。

### 方法

提出多视图特征传播(MFP)框架，将可用特征划分为多个添加高斯噪声的视图，每个视图独立通过图拓扑传播信息，聚合后的表示生成具有表达能力和鲁棒性的节点嵌入。该框架创新性地提高了极端稀疏条件下的鲁棒性，并提供了平衡效用与隐私的原则性方法。

### 主要发现

在图数据集上的大量实验表明，MFP在节点分类方面优于最先进的基线方法，同时显著减少了隐私泄露。传播的输出作为原始特征的替代插补值而非重建值，保留了效用而不损害隐私。全面的敏感性分析证实了MFP在不同场景下的稳定性和实际适用性。

### 结论

MFP为具有缺失或敏感特征的图学习领域提供了一个有效且具有隐私意识的框架。

### 翻译

图神经网络(GNNs)在关系数据上的节点分类任务中已经显示出显著的成功，然而它们的有效性往往依赖于完整节点特征的可用性。然而，在许多现实场景中，特征矩阵高度稀疏或包含敏感信息，导致性能下降和隐私风险增加。此外，直接暴露信息可能导致意外数据泄露，使攻击者能够推断敏感信息。为应对这些挑战，我们提出了一种新颖的多视图特征传播(MFP)框架，该框架在特征稀疏条件下增强节点分类同时促进隐私保护。MFP通过将可用特征划分为多个添加高斯噪声的视图来扩展传统特征传播(FP)，每个视图独立通过图拓扑传播信息。聚合后的表示产生具有表达能力和鲁棒性的节点嵌入。该框架在两个方面具有创新性：它引入了一种在极端稀疏条件下提高鲁棒性的机制，并提供了一种平衡效用与隐私的原则性方法。在图数据集上进行的大量实验表明，MFP在节点分类方面优于最先进的基线方法，同时显著减少了隐私泄露。此外，我们的分析表明传播的输出作为原始特征的替代插补值而非重建值，保留了效用而不损害隐私。全面的敏感性分析进一步证实了MFP在不同场景下的稳定性和实际适用性。总体而言，MFP为具有缺失或敏感特征的图学习领域提供了一个有效且具有隐私意识的框架。


### 论文摘要

Graph Neural Networks (GNNs) have demonstrated remarkable success in node classification tasks over relational data, yet their effectiveness often depends on the availability of complete node features. In many real-world scenarios, however, feature matrices are highly sparse or contain sensitive information, leading to degraded performance and increased privacy risks. Furthermore, direct exposure of information can result in unintended data leakage, enabling adversaries to infer sensitive information. To address these challenges, we propose a novel Multi-view Feature Propagation (MFP) framework that enhances node classification under feature sparsity while promoting privacy preservation. MFP extends traditional Feature Propagation (FP) by dividing the available features into multiple Gaussian-noised views, each propagating information independently through the graph topology. The aggregated representations yield expressive and robust node embeddings. This framework is novel in two respects: it introduces a mechanism that improves robustness under extreme sparsity, and it provides a principled way to balance utility with privacy. Extensive experiments conducted on graph datasets demonstrate that MFP outperforms state-of-the-art baselines in node classification while substantially reducing privacy leakage. Moreover, our analysis demonstrates that propagated outputs serve as alternative imputations rather than reconstructions of the original features, preserving utility without compromising privacy. A comprehensive sensitivity analysis further confirms the stability and practical applicability of MFP across diverse scenarios. Overall, MFP provides an effective and privacy-aware framework for graph learning in domains characterized by missing or sensitive features.

---

## 75. Event-Aware Prompt Learning for Dynamic Graphs

**论文链接:** [http://arxiv.org/abs/2510.11339v1](http://arxiv.org/abs/2510.11339v1)

**作者:** Xingtong Yu, Ruijuan Liang, Xinming Zhang, Yuan Fang

**发布时间:** 2025-10-13

**备注:** Under review

### GPT解析

### 总结

本文提出了一种名为EVP的事件感知动态图提示学习框架，可作为现有方法的插件，增强其利用历史事件知识的能力。

### 背景

现实世界中的图通常通过一系列事件演化，建模跨领域对象之间的动态交互。动态图神经网络(DGNNs)已成为动态图学习的流行解决方案，而提示学习方法最近也被探索应用于动态图。

### 目的

解决现有方法专注于捕捉节点与时间关系而忽视历史事件影响的问题，提出能够增强现有方法利用历史事件知识能力的框架。

### 方法

首先为每个节点提取一系列历史事件并引入事件适应机制对齐事件特征与下游任务；其次提出事件聚合机制将历史知识整合到节点表示中；最后在四个公共数据集上进行实验评估。

### 主要发现

通过实验验证了EVP框架能够有效利用历史事件知识增强动态图学习性能。

### 结论

EVP框架作为插件可以增强现有动态图学习方法的能力，特别在利用历史事件知识方面表现出色。

### 翻译

现实世界中的图通常通过一系列事件演化，建模跨领域对象之间的动态交互。在动态图学习中，动态图神经网络(DGNNs)已成为流行的解决方案。最近，提示学习方法已被探索应用于动态图。然而，现有方法通常专注于捕捉节点与时间之间的关系，而忽视了历史事件的影响。在本文中，我们提出了EVP，一种事件感知的动态图提示学习框架，可作为现有方法的插件，增强它们利用历史事件知识的能力。首先，我们为每个节点提取一系列历史事件，并引入事件适应机制将这些事件的细粒度特征与下游任务对齐。其次，我们提出事件聚合机制，有效将历史知识整合到节点表示中。最后，我们在四个公共数据集上进行广泛的实验来评估和分析EVP。


### 论文摘要

Real-world graph typically evolve via a series of events, modeling dynamic interactions between objects across various domains. For dynamic graph learning, dynamic graph neural networks (DGNNs) have emerged as popular solutions. Recently, prompt learning methods have been explored on dynamic graphs. However, existing methods generally focus on capturing the relationship between nodes and time, while overlooking the impact of historical events. In this paper, we propose EVP, an event-aware dynamic graph prompt learning framework that can serve as a plug-in to existing methods, enhancing their ability to leverage historical events knowledge. First, we extract a series of historical events for each node and introduce an event adaptation mechanism to align the fine-grained characteristics of these events with downstream tasks. Second, we propose an event aggregation mechanism to effectively integrate historical knowledge into node representations. Finally, we conduct extensive experiments on four public datasets to evaluate and analyze EVP.

---

## 76. Edge-to-Cloud Computations-as-a-Service in Software-Defined Energy Networks for Smart Grids

**论文链接:** [http://arxiv.org/abs/2510.11286v1](http://arxiv.org/abs/2510.11286v1)

**作者:** Jack Jackman, David Ryan, Arun Narayanan, Pedro Nardelli, Indrakshi Dey

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了一种名为SDEN（软件定义能源网络）的创新架构，用于解决现代电网中数据生成与处理位置不匹配的问题，通过统一边缘、雾和云计算资源，结合5G URLLC、SDN和NFV技术，实现能源、延迟和可靠性的协同优化。

### 背景

现代电网面临数据生成位置与数据处理位置严重不匹配的挑战：保护继电器、电动汽车充电和分布式可再生能源需要在边缘进行毫秒级分析，而耗能大的工作负载通常位于远程云中，导致错过实时截止日期和浪费电力。

### 目的

提出首个SDEN（软件定义能源网络）用于CaaS（计算即服务），将碎片化的电网计算转变为单一的、可编程的基础设施，提供可靠、节能的实时分析。

### 方法

统一边缘、雾和云计算与5G URLLC（超可靠低延迟通信）、SDN（软件定义网络）和NFV（网络功能虚拟化）技术；提出联合任务卸载公式，在明确URLLC约束下将计算放置与网络容量耦合；开发可行性保持的轻量级贪婪启发式算法；构建分层AI管道，包括边缘反应式、雾预测性和云战略性层，使用隐私保护的联邦GNN（图神经网络）进行故障检测和微电网协调。

### 主要发现

SDEN架构能够有效解决电网计算资源碎片化问题，通过协同优化能源消耗、网络延迟和系统可靠性，提供端到端的实时分析能力；所提出的轻量级贪婪启发式算法能够在可扩展性的同时，紧密跟踪最佳能源和延迟权衡；分层AI管道能够在不同层级提供不同特性的智能分析能力。

### 结论

SDEN建立了首个软件定义的路径，实现了实用的、电网规模的CaaS（计算即服务），与仅边缘或仅云的方案不同，它能够将碎片化的电网计算资源整合为单一、可编程的基础设施，提供可靠、节能的实时分析能力。

### 翻译

现代电网面临数据生成位置与数据处理位置严重不匹配的挑战：保护继电器、电动汽车充电和分布式可再生能源需要在边缘进行毫秒级分析，而耗能大的工作负载通常位于远程云中，导致错过实时截止日期和浪费电力。我们通过提出据我们所知首个用于CaaS（计算即服务）的SDEN（软件定义能源网络）来解决这一问题，该架构统一了边缘、雾和云计算资源，并结合5G URLLC（超可靠低延迟通信）、SDN（软件定义网络）和NFV（网络功能虚拟化）技术，共同优化端到端的能源、延迟和可靠性。我们的贡献有三方面：（i）联合任务卸载公式，在明确URLLC约束下将计算放置与网络容量耦合；（ii）可行性保持的轻量级贪婪启发式算法，可扩展并紧密跟踪最佳能源和延迟权衡；（iii）分层AI管道，边缘层具有反应性，雾层具有预测性，云层具有战略性，具有隐私保护的联邦GNN（图神经网络）用于故障检测和微电网协调。与仅边缘或仅云的方案不同，SDEN将碎片化的电网计算转变为单一的、可编程的基础设施，提供可靠、节能的实时分析，建立了首个软件定义的路径，实现实用的、电网规模的CaaS。


### 论文摘要

Modern power grids face an acute mismatch between where data is generated and where it can be processed: protection relays, EV (Electric Vehicle) charging, and distributed renewables demand millisecond analytics at the edge, while energy-hungry workloads often sit in distant clouds leading to missed real-time deadlines and wasted power. We address this by proposing, to our knowledge, the first-ever SDEN (Software Defined Energy Network) for CaaS (Computations-as-a-Service) that unifies edge, fog, and cloud compute with 5G URLLC (Ultra-Reliable Low-Latency Communications), SDN (Software Defined Networking), and NFV (Network Functions Virtualization) to co-optimize energy, latency, and reliability end-to-end. Our contributions are threefold: (i) a joint task offloading formulation that couples computation placement with network capacity under explicit URLLC constraints; (ii) a feasibility preserving, lightweight greedy heuristic that scales while closely tracking optimal energy and latency trade-offs; and (iii) a tiered AI (Artificial Intelligence) pipeline-reactive at the edge, predictive in the fog, strategic in the cloud-featuring privacy-preserving, federated GNNs (Graph Neural Networks) for fault detection and microgrid coordination. Unlike prior edge-only or cloud-only schemes, SDEN turns fragmented grid compute into a single, programmable substrate that delivers dependable, energy-aware, real time analytics establishing a first-ever, software defined path to practical, grid-scale CaaS.

---

## 77. Enforcing convex constraints in Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.11227v1](http://arxiv.org/abs/2510.11227v1)

**作者:** Ahmed Rashwan, Keith Briggs, Chris Budd, Lisa Kreusser

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文介绍了ProjNet，一个满足输入依赖约束的图神经网络框架，结合了稀疏向量裁剪和CAD算法，并通过GPU加速实现高效处理大规模输入的能力。

### 背景

许多机器学习应用需要满足复杂、动态约束的输出，但在图神经网络模型中，由于图结构数据的可变输出大小，这一任务尤其具有挑战性。

### 目的

开发一个能够满足输入依赖约束的图神经网络框架。

### 方法

ProjNet结合稀疏向量裁剪方法和Component-Averaged Dykstra (CAD)算法，建立CAD的收敛结果，开发GPU加速实现，并引入计算效率高且适合优化的替代梯度用于端到端训练。

### 主要发现

在四类约束优化问题（线性规划、两类非凸二次规划和无线发射功率优化）上验证了ProjNet的有效性。

### 结论

ProjNet在各种问题设置中表现出有效性，能够满足复杂、动态的约束要求。

### 翻译

许多机器学习应用需要满足复杂、动态约束的输出。在图神经网络模型中，由于图结构数据的可变输出大小，这一任务尤其具有挑战性。本文介绍了ProjNet，一个满足输入依赖约束的图神经网络框架。ProjNet结合了稀疏向量裁剪方法和Component-Averaged Dykstra (CAD)算法，一种解决最佳逼近问题的迭代方案。我们建立了CAD的收敛结果，并开发了能够高效处理大规模输入的GPU加速实现。为了实现端到端训练，我们引入了一个计算效率高且比精确梯度更适合优化的替代梯度。我们在四类约束优化问题上验证了ProjNet：线性规划、两类非凸二次规划和无线发射功率优化，证明了其在各种问题设置中的有效性。


### 论文摘要

Many machine learning applications require outputs that satisfy complex, dynamic constraints. This task is particularly challenging in Graph Neural Network models due to the variable output sizes of graph-structured data. In this paper, we introduce ProjNet, a Graph Neural Network framework which satisfies input-dependant constraints. ProjNet combines a sparse vector clipping method with the Component-Averaged Dykstra (CAD) algorithm, an iterative scheme for solving the best-approximation problem. We establish a convergence result for CAD and develop a GPU-accelerated implementation capable of handling large-scale inputs efficiently. To enable end-to-end training, we introduce a surrogate gradient for CAD that is both computationally efficient and better suited for optimization than the exact gradient. We validate ProjNet on four classes of constrained optimisation problems: linear programming, two classes of non-convex quadratic programs, and radio transmit power optimization, demonstrating its effectiveness across diverse problem settings.

---

## 78. Graph Neural Network-Based Multicast Routing for On-Demand Streaming Services in 6G Networks

**论文链接:** [http://arxiv.org/abs/2510.11109v1](http://arxiv.org/abs/2510.11109v1)

**作者:** Xiucheng Wang, Zien Wang, Nan Cheng, Wenchao Xu, Wei Quan, Xuemin Shen

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了一种基于图神经网络(GNN)的多播路由框架，用于解决6G网络中带宽密集型应用的路由问题，能够同时最小化传输成本并支持用户特定的视频质量需求。

### 背景

6G无线网络中带宽密集型应用（如实时体积流和多感官扩展现实）的增长需要智能多播路由解决方案，能够大规模提供差异化服务质量。

### 目的

解决传统路由算法计算复杂、结构僵化、无法支持异构用户需求的问题，以及基于神经网络的方法缺乏拓扑泛化能力和可扩展性的局限。

### 方法

将路由问题表述为约束最小流优化任务，开发强化学习算法顺序构建高效多播树；使用图注意力网络(GAT)作为编码器提取上下文感知节点嵌入，使用长短期记忆(LSTM)模块建模路由决策中的序列依赖关系。

### 主要发现

该方法接近基于动态规划的最优解，同时显著降低计算复杂度；对大规模和动态网络拓扑具有强泛化能力，适合6G多媒体交付场景的实时部署。

### 结论

提出的GNN-based多播路由框架能有效解决6G网络中带宽密集型应用的路由问题，平衡了性能和计算效率。

### 翻译

随着第六代（6G）无线网络中带宽密集型应用的增加，如实时体积流和多感官扩展现实，需要能够大规模提供差异化服务质量（QoS）的智能多播路由解决方案。传统的最短路径和多播路由算法要么计算上不可行，要么结构上僵化，它们通常无法支持异构用户需求，导致资源利用次优。基于神经网络的方法虽然提供了改进的推理速度，但通常缺乏拓扑泛化能力和可扩展性。为了解决这些限制，本文提出了一个基于图神经网络（GNN）的多播路由框架，该框架同时最小化总传输成本并支持用户特定的视频质量要求。路由问题被表述为约束最小流优化任务，并开发了一种强化学习算法，通过重用路径和适应网络动态来顺序构建高效的多播树。采用图注意力网络（GAT）作为编码器来提取上下文感知的节点嵌入，同时使用长短期记忆（LSTM）模块来建模路由决策中的序列依赖关系。大量模拟表明，该方法接近最优的基于动态规划的解决方案，同时显著降低了计算复杂度。结果还证实了该方法对大规模和动态网络拓扑的强泛化能力，突显了该方法在6G多媒体交付场景中实时部署的潜力。代码可在https://github.com/UNIC-Lab/GNN-Routing获取。


### 论文摘要

The increase of bandwidth-intensive applications in sixth-generation (6G) wireless networks, such as real-time volumetric streaming and multi-sensory extended reality, demands intelligent multicast routing solutions capable of delivering differentiated quality-of-service (QoS) at scale. Traditional shortest-path and multicast routing algorithms are either computationally prohibitive or structurally rigid, and they often fail to support heterogeneous user demands, leading to suboptimal resource utilization. Neural network-based approaches, while offering improved inference speed, typically lack topological generalization and scalability. To address these limitations, this paper presents a graph neural network (GNN)-based multicast routing framework that jointly minimizes total transmission cost and supports user-specific video quality requirements. The routing problem is formulated as a constrained minimum-flow optimization task, and a reinforcement learning algorithm is developed to sequentially construct efficient multicast trees by reusing paths and adapting to network dynamics. A graph attention network (GAT) is employed as the encoder to extract context-aware node embeddings, while a long short-term memory (LSTM) module models the sequential dependencies in routing decisions. Extensive simulations demonstrate that the proposed method closely approximates optimal dynamic programming-based solutions while significantly reducing computational complexity. The results also confirm strong generalization to large-scale and dynamic network topologies, highlighting the method's potential for real-time deployment in 6G multimedia delivery scenarios. Code is available at https://github.com/UNIC-Lab/GNN-Routing.

---

## 79. Comparative Evaluation of Neural Network Architectures for Generalizable Human Spatial Preference Prediction in Unseen Built Environments

**论文链接:** [http://arxiv.org/abs/2510.10954v1](http://arxiv.org/abs/2510.10954v1)

**作者:** Maral Doctorarastoo, Katherine A. Flanigan, Mario Bergés, Christopher McComb

**发布时间:** 2025-10-13

**备注:** The 15th International Workshop on Structural Health Monitoring  (IWSHM)

### GPT解析

### 总结

本研究比较了图神经网络、卷积神经网络和标准前馈神经网络在预测人类空间偏好方面的可推广性，使用合成数据评估它们在未见环境中的表现。

### 背景

预测人类在建成环境中的空间偏好对开发赛博物理社会基础设施系统至关重要，但偏好模型的可推广性是一个重大挑战，特别是在预测训练过程中未遇到的环境配置时。

### 目的

确定哪种神经网络架构在推广到未见过的布局方面最有效，并进行不同神经网络架构的比较研究。

### 方法

使用从简化的合成口袋公园环境生成的合成数据，评估模型预测受异构物理、环境和社会特征影响的偏好的能力，使用精确率-召回率曲线下面积计算可推广性分数。

### 主要发现

深度学习模型在学习复杂的空间和上下文依赖关系方面显示出潜力，但不同神经网络架构在推广到未见过的空间场景方面存在差异。

### 结论

可推广性分数提供了关于每种神经网络架构在未见过的建成环境中进行偏好感知人类行为建模的适用性的见解。

### 翻译

预测建成环境中人类空间偏好的能力对于开发赛博物理社会基础设施系统(CPSIS)至关重要。该领域的一个重大挑战是偏好模型的可推广性，特别是在预测训练过程中未遇到的环境配置时的有效性。虽然深度学习模型在学习复杂的空间和上下文依赖关系方面显示出潜力，但目前尚不清楚哪种神经网络架构在推广到未见过的布局方面最有效。为此，我们使用从简化的合成口袋公园环境生成的合成数据，对图神经网络、卷积神经网络和标准前馈神经网络进行了比较研究。从这个案例研究开始，可以控制分析每种模型将学习到的偏好模式转移到未见空间场景的能力。模型根据其预测受异构物理、环境和社会特征影响的偏好的能力进行评估。使用精确率-召回率曲线下面积计算可推广性分数，这种方法适用于不平衡数据，提供了关于每种神经网络架构在未见过的建成环境中进行偏好感知人类行为建模的适用性的见解。


### 论文摘要

The capacity to predict human spatial preferences within built environments is instrumental for developing Cyber-Physical-Social Infrastructure Systems (CPSIS). A significant challenge in this domain is the generalizability of preference models, particularly their efficacy in predicting preferences within environmental configurations not encountered during training. While deep learning models have shown promise in learning complex spatial and contextual dependencies, it remains unclear which neural network architectures are most effective at generalizing to unseen layouts. To address this, we conduct a comparative study of Graph Neural Networks, Convolutional Neural Networks, and standard feedforward Neural Networks using synthetic data generated from a simplified and synthetic pocket park environment. Beginning with this illustrative case study, allows for controlled analysis of each model's ability to transfer learned preference patterns to unseen spatial scenarios. The models are evaluated based on their capacity to predict preferences influenced by heterogeneous physical, environmental, and social features. Generalizability score is calculated using the area under the precision-recall curve for the seen and unseen layouts. This generalizability score is appropriate for imbalanced data, providing insights into the suitability of each neural network architecture for preference-aware human behavior modeling in unseen built environments.

---

## 80. HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations

**论文链接:** [http://arxiv.org/abs/2510.10864v1](http://arxiv.org/abs/2510.10864v1)

**作者:** Shuaicheng Zhang, Haohui Wang, Junhong Lin, Xiaojie Guo, Yada Zhu, Si Zhang, Dongqi Fu, Dawei Zhou

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文研究了图异质性与谱滤波器之间的关系，发现它们之间的联系比之前认为的更为复杂，提出了自适应滤波的必要性，并提出了一个简单而强大的GNN方法[METHOD NAME]，在实验中表现出色。

### 背景

图异质性（连接节点具有不同标签）最近引起了广泛关注。大多数现有工作采用简化的方法——对同质性图使用低通滤波器，对异质性图使用高通滤波器。

### 目的

探究图异质性与谱滤波器之间的复杂关系，设计能够适应不同异质性连接的自适应滤波器，以提高GNN在各类图上的性能。

### 方法

提出了[METHOD NAME]，一个简单而强大的GNN方法，它能够提取异质性谱中的信息，并通过自适应混合来结合显著的表示。

### 主要发现

图异质性与谱滤波器之间的关系更为复杂——最优滤波器响应在不同频率分量上有所不同，并且与异质性程度没有严格的单调相关性。GNNs的平均频率响应和图异质性程度之间没有严格的单调相关性。

### 结论

需要自适应图滤波器来保证良好的泛化性能。[METHOD NAME]在同类质性和异质性图上相比领先基线实现了高达9.2%的准确率提升。

### 翻译

图异质性，即连接的节点具有不同标签，最近引起了广泛关注。大多数现有工作采用简化的方法——对同质性图使用低通滤波器，对异质性图使用高通滤波器。然而，我们发现图异质性与谱滤波器之间的关系更为复杂——最优滤波器响应在不同频率分量上有所不同，并且与异质性程度没有严格的单调相关性。这一发现挑战了传统的固定滤波器设计，并表明需要自适应滤波来保持图嵌入的表达能力。正式地说，自然产生的问题有：给定一个异质性图G，G的异质性程度的变化如何以及会在多大程度上影响GNNs的性能？如何设计自适应滤波器来适应这些变化的异质性连接？我们的理论分析表明，GNNs的平均频率响应和图异质性程度之间没有严格的单调相关性，这需要自适应图滤波器来保证良好的泛化性能。因此，我们提出了[METHOD NAME]，一个简单而强大的GNN，它提取异质性谱中的信息，并通过自适应混合来结合显著的表示。[METHOD NAME]的优越性能在同类质性和异质性图上相比领先基线实现了高达9.2%的准确率提升。


### 论文摘要

Graph heterophily, where connected nodes have different labels, has attracted significant interest recently. Most existing works adopt a simplified approach - using low-pass filters for homophilic graphs and high-pass filters for heterophilic graphs. However, we discover that the relationship between graph heterophily and spectral filters is more complex - the optimal filter response varies across frequency components and does not follow a strict monotonic correlation with heterophily degree. This finding challenges conventional fixed filter designs and suggests the need for adaptive filtering to preserve expressiveness in graph embeddings. Formally, natural questions arise: Given a heterophilic graph G, how and to what extent will the varying heterophily degree of G affect the performance of GNNs? How can we design adaptive filters to fit those varying heterophilic connections? Our theoretical analysis reveals that the average frequency response of GNNs and graph heterophily degree do not follow a strict monotonic correlation, necessitating adaptive graph filters to guarantee good generalization performance. Hence, we propose [METHOD NAME], a simple yet powerful GNN, which extracts information across the heterophily spectrum and combines salient representations through adaptive mixing. [METHOD NAME]'s superior performance achieves up to 9.2% accuracy improvement over leading baselines across homophilic and heterophilic graphs.

---

## 81. Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion

**论文链接:** [http://arxiv.org/abs/2510.10849v1](http://arxiv.org/abs/2510.10849v1)

**作者:** Donald Loveland, Yao-An Yang, Danai Koutra

**发布时间:** 2025-10-12

### GPT解析

### 总结

本文提出了一种名为GLANCE的自适应GNN-LLM融合框架，通过轻量级路由器选择性调用LLM来优化GNN预测，在异质节点上获得显著性能提升，同时保持整体性能最优。

### 背景

文本属性图的学习促进了大型语言模型(LLMs)在图学习中的应用。然而，大多数融合策略在所有节点上统一应用，只获得较小的整体性能提升。

### 目的

重新设计LLM-GNN融合框架，专注于GNN通常表现不佳的节点，提高图学习性能。

### 方法

提出GLANCE(GNN with LLM Assistance for Neighbor- and Context-aware Embeddings)框架，使用轻量级路由器根据每个节点的低成本信号决定是否查询LLM。路由器使用基于优势的目标进行训练，比较查询LLM与仅依赖GNN的效用。

### 主要发现

GNN和LLM在性能上存在显著差异，各自在不同的结构模式上表现出色；GLANCE在节点子组之间实现了最佳的性能平衡，在异质节点上获得最高13%的性能提升，同时实现顶级整体性能。

### 结论

自适应的、节点感知的GNN-LLM架构具有重要价值，选择性调用LLM enables在大图上的可扩展部署，而不会产生高计算成本。

### 翻译

文本属性图的学习激发了大型语言模型(LLMs)在图学习中的应用。然而，大多数融合策略统一应用于所有节点，仅获得较小的整体性能提升。我们认为这一结果源于聚合指标掩盖了LLMs何时提供益处，阻碍了新策略的可操作信号。在这项工作中，我们围绕GNN通常表现不佳的节点重新设计了LLM-GNN融合。我们首先展示了GNN和LLM在性能上可以显著不同，各自在不同的结构模式上表现出色，例如局部同质性。为了利用这一发现，我们提出了GLANCE(GNN with LLM Assistance for Neighbor- and Context-aware Embeddings)框架，该框架调用LLM来优化GNN的预测。GLANCE采用一个轻量级路由器，根据每个节点的低成本信号，决定是否查询LLM。由于LLM调用是不可微分的，路由器使用基于优势的目标进行训练，比较查询LLM与仅依赖GNN的效用。在多个基准测试中，GLANCE在节点子组之间实现了最佳的性能平衡，在异质节点上获得显著提升(最高+13%)，同时实现顶级整体性能。我们的研究结果表明自适应的、节点感知的GNN-LLM架构的价值，选择性调用LLM enables在大图上的可扩展部署，而不会产生高计算成本。


### 论文摘要

Learning on text-attributed graphs has motivated the use of Large Language Models (LLMs) for graph learning. However, most fusion strategies are applied uniformly across all nodes and attain only small overall performance gains. We argue this result stems from aggregate metrics that obscure when LLMs provide benefit, inhibiting actionable signals for new strategies. In this work, we reframe LLM-GNN fusion around nodes where GNNs typically falter. We first show that performance can significantly differ between GNNs and LLMs, with each excelling on distinct structural patterns, such as local homophily. To leverage this finding, we propose GLANCE (GNN with LLM Assistance for Neighbor- and Context-aware Embeddings), a framework that invokes an LLM to refine a GNN's prediction. GLANCE employs a lightweight router that, given inexpensive per-node signals, decides whether to query the LLM. Since the LLM calls are non-differentiable, the router is trained with an advantage-based objective that compares the utility of querying the LLM against relying solely on the GNN. Across multiple benchmarks, GLANCE achieves the best performance balance across node subgroups, achieving significant gains on heterophilous nodes (up to $+13\%$) while simultaneously achieving top overall performance. Our findings highlight the value of adaptive, node-aware GNN-LLM architectures, where selectively invoking the LLM enables scalable deployment on large graphs without incurring high computational costs.

---

## 82. Fast and the Furious: Hot Starts in Pursuit-Evasion Games

**论文链接:** [http://arxiv.org/abs/2510.10830v1](http://arxiv.org/abs/2510.10830v1)

**作者:** Gabriel Smithline, Scott Nivison

**发布时间:** 2025-10-12

**备注:** Presented at AAMAS Workshop on Autonomous Robots and Multirobot  Systems (ARMS)

### GPT解析

### 总结

本研究提出了一种结合博弈论控制理论与图神经网络的新方法，用于解决在没有事先了解逃亡者位置情况下的追捕者部署问题。通过构建图特征空间和训练图卷积网络生成战略有效的初始配置，显著提高了追捕效率。

### 背景

在追逃游戏中，没有事先了解逃亡者位置的情况下有效部署追捕者仍然是一个重大挑战。

### 目的

开发一种新方法，使追捕者能够在没有先验知识的情况下有效地部署，提高追捕效率。

### 方法

结合博弈论控制理论与图神经网络，将追捕者配置表示为图，通过多目标优化构建图特征空间识别帕累托最优配置，并在这些最优图上训练图卷积网络生成'热启动'配置。

### 主要发现

经验评估表明，GCN生成的热启动相比随机配置具有显著优势；在多追捕者和多逃亡者场景中，该方法加速了逃亡者生存率下降，减少了追捕者移动距离，并增强了围捕效果。

### 结论

该方法在追逃游戏中展示了明显的战略优势，能有效提高追捕效率。

### 翻译

在没有逃亡者位置先验知识的情况下，在追逃游戏中有效部署追捕者仍然是一个重大挑战。本文介绍了一种结合博弈论控制理论与图神经网络的新方法。通过将追捕者配置概念化为战略安排并表示为图，通过多目标优化构建图特征空间以识别帕累托最优配置。在这些帕累托最优图上训练图卷积网络(GCN)，生成战略上有效的初始配置，称为'热启动'。经验评估表明，GCN生成的热启动相比随机配置具有显著优势。在考虑多个追捕者和逃亡者的场景中，这种方法加速了逃亡者生存率的下降，减少了追捕者的移动距离，并增强了围捕效果，展示了明显的战略优势。


### 论文摘要

Effectively positioning pursuers in pursuit-evasion games without prior knowledge of evader locations remains a significant challenge. A novel approach that combines game-theoretic control theory with Graph Neural Networks is introduced in this work. By conceptualizing pursuer configurations as strategic arrangements and representing them as graphs, a Graph Characteristic Space is constructed via multi-objective optimization to identify Pareto-optimal configurations. A Graph Convolutional Network (GCN) is trained on these Pareto-optimal graphs to generate strategically effective initial configurations, termed "hot starts". Empirical evaluations demonstrate that the GCN-generated hot starts provide a significant advantage over random configurations. In scenarios considering multiple pursuers and evaders, this method hastens the decline in evader survival rates, reduces pursuer travel distances, and enhances containment, showcasing clear strategic benefits.

---

## 83. Structure Over Signal: A Globalized Approach to Multi-relational GNNs for Stock Prediction

**论文链接:** [http://arxiv.org/abs/2510.10775v1](http://arxiv.org/abs/2510.10775v1)

**作者:** Amber Li, Aruzhan Abil, Juno Marques Oda

**发布时间:** 2025-10-12

### GPT解析

### 总结

本文提出了OmniGNN，一个基于注意力的多关系动态图神经网络，通过异构节点和边类型整合宏观经济背景，实现稳健的消息传递，特别在宏观经济冲击期间表现出色。

### 背景

图神经网络在金融市场中已被成功应用于建模关系数据，有效捕捉股票间的非线性依赖关系。

### 目的

解决现有模型在宏观经济冲击期间无法有效传播消息的问题，提出一个能够整合宏观经济背景的稳健图神经网络模型。

### 方法

引入一个作为全局中介的行业节点实现快速冲击传播；利用图注意力网络(GAT)加权邻居贡献；采用Transformer捕捉多关系间的时间动态。

### 主要发现

OmniGNN在公共数据集上的股票预测模型中表现优于现有模型，特别是在COVID-19期间表现出强大的鲁棒性。

### 结论

OmniGNN通过整合宏观经济背景和优化的消息传递机制，显著提升了股票预测模型的性能，特别是在宏观经济冲击期间的鲁棒性。

### 翻译

在金融市场中，图神经网络已被成功应用于建模关系数据，有效捕捉股票间的非线性依赖关系。然而，现有模型通常在宏观经济冲击期间无法有效传播消息。在本文中，我们提出了OmniGNN，一个基于注意力的多关系动态图神经网络，通过异构节点和边类型整合宏观经济背景，实现稳健的消息传递。OmniGNN的核心是一个作为全局中介的行业节点，使冲击能够在图中快速传播，而不依赖长距离多跳扩散。该模型利用图注意力网络(GAT)来加权邻居的贡献，并采用Transformer来捕捉多关系间的时间动态。实验表明，OmniGNN在公共数据集上的股票预测模型中优于现有模型，特别是在COVID-19期间表现出强大的鲁棒性。


### 论文摘要

In financial markets, Graph Neural Networks have been successfully applied to modeling relational data, effectively capturing nonlinear inter-stock dependencies. Yet, existing models often fail to efficiently propagate messages during macroeconomic shocks. In this paper, we propose OmniGNN, an attention-based multi-relational dynamic GNN that integrates macroeconomic context via heterogeneous node and edge types for robust message passing. Central to OmniGNN is a sector node acting as a global intermediary, enabling rapid shock propagation across the graph without relying on long-range multi-hop diffusion. The model leverages Graph Attention Networks (GAT) to weigh neighbor contributions and employs Transformers to capture temporal dynamics across multiplex relations. Experiments show that OmniGNN outperforms existing stock prediction models on public datasets, particularly demonstrating strong robustness during the COVID-19 period.

---

## 84. Mapping the Urban Mobility Intelligence Frontier: A Scientometric Analysis of Data-Driven Pedestrian Trajectory Prediction and Simulation

**论文链接:** [http://arxiv.org/abs/2510.10327v1](http://arxiv.org/abs/2510.10327v1)

**作者:** Junhao Xu, Hui Zeng

**发布时间:** 2025-10-11

**备注:** 5 figures

### GPT解析

### 总结

本研究对数据驱动的行人轨迹预测和人群模拟进行了全面的文献计量分析，揭示了人工智能、城市信息和人群行为建模之间的融合趋势，并探讨了这些技术如何应用于城市移动设计、公共安全规划和智慧城市数字孪生开发。

### 背景

理解和预测行人动力学对于塑造更安全、更响应性、以人为中心的城市环境至关重要。随着人工智能技术的发展，数据驱动的行人轨迹预测和人群模拟研究日益增多，需要系统梳理该领域的发展脉络。

### 目的

通过文献计量分析，绘制数据驱动的行人轨迹预测和人群模拟研究的知识演进图和跨学科结构，识别主要趋势、有影响力的贡献者和新兴前沿领域。

### 方法

使用Web of Science核心合集的文献计量数据，采用SciExplorer和Bibliometrix工具进行分析，识别该领域的主要研究趋势、有影响力的贡献者和新兴前沿领域。

### 主要发现

研究发现人工智能、城市信息和人群行为建模之间存在强烈的融合趋势，这种融合由图神经网络、transformers和生成模型驱动。除了技术进步外，该领域越来越多地为城市移动设计、公共安全规划和智慧城市数字孪生开发提供信息。然而，该领域在确保可解释性、包容性和跨领域可转移性方面仍然存在挑战。

### 结论

通过将方法轨迹与城市应用联系起来，这项工作强调了数据驱动方法如何丰富城市治理，并为未来城市的自适应、社会负责的移动智能铺平道路。

### 翻译

理解和预测行人动力学对于塑造更安全、更具响应性、以人为中心的城市环境已成为必不可少的工作。本研究对数据驱动的行人轨迹预测和人群模拟研究进行了全面的文献计量分析，绘制了其知识演进和跨学科结构。利用Web of Science核心合集的文献计量数据，我们采用SciExplorer和Bibliometrix来识别主要趋势、有影响力的贡献者和新兴前沿。结果表明，人工智能、城市信息和人群行为建模之间存在强烈的融合趋势，这种趋势由图神经网络、transformers和生成模型驱动。除了技术进步外，该领域越来越多地影响着城市移动设计、公共安全规划和智慧城市数字孪生开发。然而，在确保可解释性、包容性和跨领域可转移性方面仍然存在挑战。通过将方法轨迹与城市应用联系起来，这项工作强调了数据驱动方法如何丰富城市治理，并为未来城市的自适应、社会负责的移动智能铺平道路。


### 论文摘要

Understanding and predicting pedestrian dynamics has become essential for shaping safer, more responsive, and human-centered urban environments. This study conducts a comprehensive scientometric analysis of research on data-driven pedestrian trajectory prediction and crowd simulation, mapping its intellectual evolution and interdisciplinary structure. Using bibliometric data from the Web of Science Core Collection, we employ SciExplorer and Bibliometrix to identify major trends, influential contributors, and emerging frontiers. Results reveal a strong convergence between artificial intelligence, urban informatics, and crowd behavior modeling--driven by graph neural networks, transformers, and generative models. Beyond technical advances, the field increasingly informs urban mobility design, public safety planning, and digital twin development for smart cities. However, challenges remain in ensuring interpretability, inclusivity, and cross-domain transferability. By connecting methodological trajectories with urban applications, this work highlights how data-driven approaches can enrich urban governance and pave the way for adaptive, socially responsible mobility intelligence in future cities.

---

## 85. Preference-driven Knowledge Distillation for Few-shot Node Classification

**论文链接:** [http://arxiv.org/abs/2510.10116v1](http://arxiv.org/abs/2510.10116v1)

**作者:** Xing Wei, Chunchun Chen, Rui Fan, Xiaofeng Cao, Sourav Medya, Wei Ye

**发布时间:** 2025-10-11

**备注:** Accepted at NeurIPS 2025

### GPT解析

### 总结

本文提出了一种偏好驱动的知识蒸馏(PKD)框架，结合大型语言模型和图神经网络的优势，用于少样本节点分类任务。

### 背景

图神经网络能高效处理带文本属性的图，但训练依赖人工标注标签；现实世界中TAGs节点的复杂多样局部拓扑结构使单一机制难以处理；大型语言模型在TAGs的零样本/少样本学习中表现良好，但面临可扩展性挑战。

### 目的

协同大型语言模型和多种图神经网络的优势，解决少样本节点分类问题。

### 方法

提出偏好驱动的知识蒸馏框架，包括：(1)GNN偏好驱动的节点选择器，促进从LLMs到教师GNN的预测蒸馏；(2)节点偏好驱动的GNN选择器，为每个节点识别最合适的教师GNN，实现定制化知识蒸馏。

### 主要发现

所提出的框架在真实世界TAGs的少样本节点分类任务中表现出色，能够有效处理节点复杂的局部拓扑结构。

### 结论

PKD框架能够有效结合大型语言模型和图神经网络的优势，解决少样本节点分类问题，并在真实数据集上验证了其有效性。

### 翻译

图神经网络(GNNs)由于其消息传递机制能够高效处理带文本属性的图(TAGs)，但它们的训练严重依赖人工标注的标签。此外，现实世界中TAGs节点的复杂多样的局部拓扑结构使得单一机制难以处理。大型语言模型(LLMs)在TAGs的零样本/少样本学习中表现良好，但面临可扩展性挑战。因此，我们提出了一种偏好驱动的知识蒸馏(PKD)框架，协同大型语言模型和多种图神经网络的优势用于少样本节点分类。具体而言，我们开发了一个GNN偏好驱动的节点选择器，有效促进从LLMs到教师GNN的预测蒸馏。为进一步处理节点复杂的局部拓扑结构，我们开发了一个节点偏好驱动的GNN选择器，为每个节点识别最合适的教师GNN，从而促进从教师GNN到学生GNN的定制化知识蒸馏。大量实验验证了我们所提出的框架在真实世界TAGs的少样本节点分类中的有效性。


### 论文摘要

Graph neural networks (GNNs) can efficiently process text-attributed graphs (TAGs) due to their message-passing mechanisms, but their training heavily relies on the human-annotated labels. Moreover, the complex and diverse local topologies of nodes of real-world TAGs make it challenging for a single mechanism to handle. Large language models (LLMs) perform well in zero-/few-shot learning on TAGs but suffer from a scalability challenge. Therefore, we propose a preference-driven knowledge distillation (PKD) framework to synergize the complementary strengths of LLMs and various GNNs for few-shot node classification. Specifically, we develop a GNN-preference-driven node selector that effectively promotes prediction distillation from LLMs to teacher GNNs. To further tackle nodes' intricate local topologies, we develop a node-preference-driven GNN selector that identifies the most suitable teacher GNN for each node, thereby facilitating tailored knowledge distillation from teacher GNNs to the student GNN. Extensive experiments validate the efficacy of our proposed framework in few-shot node classification on real-world TAGs.

---

## 86. Integrating Structure-Aware Attention and Knowledge Graphs in Explainable Recommendation Systems

**论文链接:** [http://arxiv.org/abs/2510.10109v1](http://arxiv.org/abs/2510.10109v1)

**作者:** Shuangquan Lyu, Ming Wang, Huajun Zhang, Jiasen Zheng, Junjiang Lin, Xiaoxuan Sun

**发布时间:** 2025-10-11

### GPT解析

### 总结

这篇论文设计并实现了一个结合知识图谱和结构感知注意力机制的可解释推荐模型，该模型基于图神经网络，采用多跳邻居聚合策略，能够捕获用户隐式偏好关系。

### 背景

推荐系统领域需要更有效地利用知识图谱信息来提高推荐性能，同时需要模型具有可解释性。

### 目的

设计一个能够整合知识图谱结构信息，并通过注意力机制动态分配邻居重要性的推荐模型，以提高推荐的准确性和可解释性。

### 方法

构建基于图神经网络的推荐模型，将用户和项目嵌入统一图结构，利用知识图谱构建多级语义路径提取上下文信息，通过用户和项目表示交互生成推荐，使用二元交叉熵损失函数优化模型。

### 主要发现

在Amazon Books数据集上的实验表明，所提出模型在各种评估指标上表现优越，具有良好的收敛性和稳定性。

### 结论

结构感知注意力机制在知识图谱增强推荐中具有有效性和实用性，能够提高推荐性能并提供更好的可解释性。

### 翻译

这篇论文设计并实现了一个可解释的推荐模型，该模型将知识图谱与结构感知注意力机制相结合。该模型基于图神经网络构建，并采用了多跳邻居聚合策略。通过整合知识图谱的结构信息，并通过注意力机制动态分配不同邻居的重要性，模型增强了捕获隐式偏好关系的能力。在所提出的方法中，用户和项目被嵌入到统一的图结构中。基于知识图谱中的实体和关系构建多级语义路径，以提取更丰富的上下文信息。在评分预测阶段，通过用户和目标项目表示之间的交互生成推荐。模型使用二元交叉熵损失函数进行优化。在Amazon Books数据集上进行的实验验证了所提出模型在各种评估指标上的优越性能。该模型还表现出良好的收敛性和稳定性。这些结果进一步证明了结构感知注意力机制在知识图谱增强推荐中的有效性和实用性。


### 论文摘要

This paper designs and implements an explainable recommendation model that integrates knowledge graphs with structure-aware attention mechanisms. The model is built on graph neural networks and incorporates a multi-hop neighbor aggregation strategy. By integrating the structural information of knowledge graphs and dynamically assigning importance to different neighbors through an attention mechanism, the model enhances its ability to capture implicit preference relationships. In the proposed method, users and items are embedded into a unified graph structure. Multi-level semantic paths are constructed based on entities and relations in the knowledge graph to extract richer contextual information. During the rating prediction phase, recommendations are generated through the interaction between user and target item representations. The model is optimized using a binary cross-entropy loss function. Experiments conducted on the Amazon Books dataset validate the superior performance of the proposed model across various evaluation metrics. The model also shows good convergence and stability. These results further demonstrate the effectiveness and practicality of structure-aware attention mechanisms in knowledge graph-enhanced recommendation.

---

## 87. Lighter-X: An Efficient and Plug-and-play Strategy for Graph-based Recommendation through Decoupled Propagation

**论文链接:** [http://arxiv.org/abs/2510.10105v1](http://arxiv.org/abs/2510.10105v1)

**作者:** Yanping Zheng, Zhewei Wei, Frank de Hoog, Xu Chen, Hongteng Xu, Yuhang Ye, Jiadeng Huang

**发布时间:** 2025-10-11

### GPT解析

### 总结

该研究提出了Lighter-X框架，解决了传统图神经网络推荐系统在大规模部署中的可扩展性问题，通过高效压缩和解耦框架实现了参数和计算复杂度的显著降低。

### 背景

图神经网络在推荐系统中表现优异，但传统方法如LightGCN需要为每个节点维护嵌入向量，导致参数复杂度为O(n×d)，其中n是用户和物品总数，这在大规模应用中面临挑战。

### 目的

开发一个高效且模块化的框架，能够无缝集成到现有GNN推荐器架构中，减少参数大小和计算复杂度，同时保持理论保证和经验性能，实现大规模实际部署。

### 方法

分析原始结构和参数中的固有冗余，提出稀疏邻接结构和高维嵌入矩阵的高效压缩方案，将参数复杂度从O(n×d)降低到O(h×d)（h<<n），并通过解耦框架优化模型，减少训练过程中的计算复杂度。

### 主要发现

Lighter-X在保持与基线模型相当性能的同时，显著减少了参数需求；在有数百万条边的大规模交互图上，仅使用LightGCN 1%的参数就能获得更好的结果。

### 结论

Lighter-X框架能够在保持推荐性能的同时大幅降低参数需求，使大规模图神经网络推荐系统的实际部署成为可能。

### 翻译

图神经网络在推荐系统中已展现出显著的有效性。然而，传统的基于图的推荐系统，如LightGCN，需要为每个节点维护大小为d的嵌入，导致参数复杂度为O(n×d)，其中n代表用户和物品的总数。这种扩展模式在实际应用的大规模图部署中构成了重大挑战。为解决这一可扩展性限制，我们提出了Lighter-X，这是一个高效且模块化的框架，可以无缝集成到现有的基于GNN的推荐器架构中。我们的方法显著减少了参数大小和计算复杂度，同时保留了基础模型的理论保证和经验性能，从而实现了大规模的实际部署。具体而言，我们分析了原始结构和参数中的固有冗余，识别了优化机会。基于这一洞察，我们提出了稀疏邻接结构和高维嵌入矩阵的高效压缩方案，实现了O(h×d)的参数复杂度，其中h<<n。此外，模型通过解耦框架进行优化，减少了训练过程中的计算复杂度并提高了可扩展性。大量实验表明，Lighter-X以显著更少的参数实现了与基线模型相当的性能。特别是在有数百万条边的大规模交互图上，我们仅使用LightGCN 1%的参数就能获得更好的结果。


### 论文摘要

Graph Neural Networks (GNNs) have demonstrated remarkable effectiveness in recommendation systems. However, conventional graph-based recommenders, such as LightGCN, require maintaining embeddings of size $d$ for each node, resulting in a parameter complexity of $\mathcal{O}(n \times d)$, where $n$ represents the total number of users and items. This scaling pattern poses significant challenges for deployment on large-scale graphs encountered in real-world applications. To address this scalability limitation, we propose \textbf{Lighter-X}, an efficient and modular framework that can be seamlessly integrated with existing GNN-based recommender architectures. Our approach substantially reduces both parameter size and computational complexity while preserving the theoretical guarantees and empirical performance of the base models, thereby enabling practical deployment at scale. Specifically, we analyze the original structure and inherent redundancy in their parameters, identifying opportunities for optimization. Based on this insight, we propose an efficient compression scheme for the sparse adjacency structure and high-dimensional embedding matrices, achieving a parameter complexity of $\mathcal{O}(h \times d)$, where $h \ll n$. Furthermore, the model is optimized through a decoupled framework, reducing computational complexity during the training process and enhancing scalability. Extensive experiments demonstrate that Lighter-X achieves comparable performance to baseline models with significantly fewer parameters. In particular, on large-scale interaction graphs with millions of edges, we are able to attain even better results with only 1\% of the parameter over LightGCN.

---

## 88. Rademacher Meets Colors: More Expressivity, but at What Cost ?

**论文链接:** [http://arxiv.org/abs/2510.10101v1](http://arxiv.org/abs/2510.10101v1)

**作者:** Martin Carrasco, Caio Deberaldini Netto, Vahan A. Martirosyan, Aneeqa Mehrab, Ehimare Okoyomon, Caterina Graziani

**发布时间:** 2025-10-11

### GPT解析

### 总结

本研究通过着色算法的视角，揭示了图神经网络表达能力和泛化能力之间的权衡关系，证明WL着色诱导的等价类数量直接限制了GNN的Rademacher复杂度，从而解释了为什么更强的表达能力往往导致更弱的泛化保证。

### 背景

图神经网络(GNNs)的表达能力通常通过它们与图同构测试(如Weisfeiler-Leman层次结构)的对应关系来理解。更具表达能力的GNN能够区分更多种类的图，但也观察到它们有更高的泛化误差。

### 目的

提供图神经网络表达能力和泛化能力之间权衡关系的理论解释，统一表达能力和泛化的研究，为增加表达能力以泛化为代价的现象提供原则性理解。

### 方法

通过着色算法的视角，将表达能力和泛化能力联系起来，分析WL着色诱导的等价类数量对GNN的Rademacher复杂度的影响，并研究Rademacher复杂度在不同样本的颜色计数扰动下的稳定性。

### 主要发现

1) WL着色诱导的等价类数量直接限制了GNN的Rademacher复杂度；2) 更强的表达能力导致更高的复杂度，从而更弱的泛化保证；3) Rademacher复杂度在不同样本的颜色计数扰动下是稳定的，确保了数据集间采样变异性的鲁棒性；4) 该框架适用于任意GNN架构和表达能力度量。

### 结论

图神经网络的表达能力和泛化能力之间存在权衡关系，增加表达能力通常以泛化为代价。这一发现为理解和设计GNN提供了重要指导，强调了在追求高表达能力的同时需要考虑泛化性能的重要性。

### 翻译

图神经网络(GNNs)的表达能力通常通过它们与图同构测试(如Weisfeiler-Leman(WL)层次结构)的对应关系来理解。虽然更具表达能力的GNN能够区分更多种类的图，但也观察到它们有更高的泛化误差。这项工作通过着色算法的视角为这种权衡提供了理论解释。具体来说，我们证明WL着色诱导的等价类数量直接限制了GNN的Rademacher复杂度——这是泛化的一个关键数据相关度量。我们的分析表明，更强的表达能力导致更高的复杂度，从而更弱的泛化保证。此外，我们证明了Rademacher复杂度在不同样本的颜色计数扰动下是稳定的，确保了数据集间采样变异性的鲁棒性。重要的是，我们的框架不仅限于消息传递GNN或1-WL，还扩展到将图划分为等价类的任意GNN架构和表达能力度量。这些结果统一了GNN中表达能力和泛化的研究，为为什么增加表达能力通常以泛化为代价提供了原则性理解。


### 论文摘要

The expressive power of graph neural networks (GNNs) is typically understood through their correspondence with graph isomorphism tests such as the Weisfeiler-Leman (WL) hierarchy. While more expressive GNNs can distinguish a richer set of graphs, they are also observed to suffer from higher generalization error. This work provides a theoretical explanation for this trade-off by linking expressivity and generalization through the lens of coloring algorithms. Specifically, we show that the number of equivalence classes induced by WL colorings directly bounds the GNNs Rademacher complexity -- a key data-dependent measure of generalization. Our analysis reveals that greater expressivity leads to higher complexity and thus weaker generalization guarantees. Furthermore, we prove that the Rademacher complexity is stable under perturbations in the color counts across different samples, ensuring robustness to sampling variability across datasets. Importantly, our framework is not restricted to message-passing GNNs or 1-WL, but extends to arbitrary GNN architectures and expressivity measures that partition graphs into equivalence classes. These results unify the study of expressivity and generalization in GNNs, providing a principled understanding of why increasing expressive power often comes at the cost of generalization.

---

## 89. Learning Joint Embeddings of Function and Process Call Graphs for Malware Detection

**论文链接:** [http://arxiv.org/abs/2510.09984v1](http://arxiv.org/abs/2510.09984v1)

**作者:** Kartikeya Aneja, Nagender Aneja, Murat Kantarcioglu

**发布时间:** 2025-10-11

### GPT解析

### 总结

本文提出了一种名为GeminiNet的统一神经网络方法，用于从函数调用图(FCGs)和进程调用图(PCGs)中学习联合嵌入，实现对软件系统的多视角分析。

### 背景

软件系统可以表示为图来捕获函数和进程间的依赖关系，根据不同目标可构建不同类型的图，如函数调用图和进程交互图。虽然这些图相关但视角不同，提供互补信息。先前研究多关注单一图表示，联合建模两种图的方法研究不足。

### 目的

探索对函数调用图和进程调用图进行联合建模，实现对软件系统更深层次、多视角的分析。

### 方法

提出GeminiNet统一神经网络方法，构建包含635个Windows可执行文件(318个恶意和317个良性)的数据集，使用Ghidra提取FCGs，Any.Run沙箱提取PCGs。采用双图卷积分支和自适应门控机制，平衡静态和动态视图的贡献。

### 主要发现

联合嵌入方法优于单图模型，能够提供更全面的软件系统分析。

### 结论

通过联合建模函数调用图和进程调用图，可以实现对软件系统更全面、准确的分析和理解，有助于软件行为分析和安全检测。

### 翻译

软件系统可以表示为图，捕获函数和进程之间的依赖关系。软件系统的一个有趣方面是，根据提取目标和优先级的不同，它们可以表示为不同类型的图。例如，可以捕获软件内的函数调用以创建函数调用图，突出函数之间的关系和依赖。或者，可以对软件生成的进程进行建模，生成进程交互图，关注运行时行为和进程间通信。虽然这些图表示相关，但每个都捕获了系统的不同视角，提供了对其结构和操作的互补见解。虽然先前的研究利用图神经网络分析软件行为，但大多数工作只关注单一类型的图表示。函数调用图和进程交互图的联合建模在很大程度上仍未被探索，留下了对软件系统进行更深层次、多视角分析的机会。本文提出了一个构建和训练函数调用图和进程调用图以及学习联合嵌入的流程。我们证明联合嵌入优于单图模型。在本文中，我们提出了GeminiNet，一种统一的神经网络方法，用于从函数调用图和进程调用图中学习联合嵌入。我们构建了一个包含635个Windows可执行文件的新数据集，使用Ghidra提取函数调用图，使用Any.Run沙箱提取进程调用图。GeminiNet采用双图卷积分支和自适应门控机制，以平衡静态和动态视图的贡献。


### 论文摘要

Software systems can be represented as graphs, capturing dependencies among functions and processes. An interesting aspect of software systems is that they can be represented as different types of graphs, depending on the extraction goals and priorities. For example, function calls within the software can be captured to create function call graphs, which highlight the relationships between functions and their dependencies. Alternatively, the processes spawned by the software can be modeled to generate process interaction graphs, which focus on runtime behavior and inter-process communication. While these graph representations are related, each captures a distinct perspective of the system, providing complementary insights into its structure and operation. While previous studies have leveraged graph neural networks (GNNs) to analyze software behaviors, most of this work has focused on a single type of graph representation. The joint modeling of both function call graphs and process interaction graphs remains largely underexplored, leaving opportunities for deeper, multi-perspective analysis of software systems. This paper presents a pipeline for constructing and training Function Call Graphs (FCGs) and Process Call Graphs (PCGs) and learning joint embeddings. We demonstrate that joint embeddings outperform a single-graph model. In this paper, we propose GeminiNet, a unified neural network approach that learns joint embeddings from both FCGs and PCGs. We construct a new dataset of 635 Windows executables (318 malicious and 317 benign), extracting FCGs via Ghidra and PCGs via Any.Run sandbox. GeminiNet employs dual graph convolutional branches with an adaptive gating mechanism that balances contributions from static and dynamic views.

---

## 90. Phase-Aware Deep Learning with Complex-Valued CNNs for Audio Signal Applications

**论文链接:** [http://arxiv.org/abs/2510.09926v1](http://arxiv.org/abs/2510.09926v1)

**作者:** Naman Agrawal

**发布时间:** 2025-10-10

### GPT解析

### 总结

本研究探讨了复值卷积神经网络(CVCNNs)在音频信号处理中的设计与应用，重点关注实值网络中常被忽略的相位信息的保留和利用。通过理论基础介绍、训练技术调整和三个阶段的实验评估，证明了复值架构的表现能力和相位作为音频处理中可利用特征的价值。

### 背景

在音频信号处理中，相位信息通常被实值神经网络所忽略，而复值神经网络为保留和利用这些信息提供了可能。

### 目的

研究复值卷积神经网络(CVCNNs)在音频信号处理中的设计和应用，探索如何有效利用通常被实值网络忽略的相位信息。

### 方法

介绍CVCNNs的基础理论概念(复卷积、池化层、基于Wirtinger的微分和复值激活函数)，调整训练技术(复值批归一化和权重初始化方案)，并通过三个阶段实验评估：在图像数据集上基准测试、使用MFCC进行音频分类、引入GNN通过边权重建模相位信息。

### 主要发现

CVCNNs在图像数据集上表现与实值CNN相当；在音频分类中，CVCNNs在实值MFCC上训练时略微优于实值CNN，但保留相位存在挑战；包含相位的GNNs在二元和多流派分类中带来可衡量的性能提升；心脏形激活函数等方法显示出前景。

### 结论

复值架构具有强大的表达能力，相位是音频处理中一个有意义且可利用的特征。未来在相位感知设计方面的进步对于利用复表示在神经网络中的潜力至关重要。

### 翻译

本研究探讨了复值卷积神经网络(CVCNNs)在音频信号处理中的设计与应用，重点关注实值网络中常被忽略的相位信息的保留和利用。我们首先介绍CVCNNs的基础理论概念，包括复卷积、池化层、基于Wirtinger的微分以及各种复值激活函数。这些理论概念辅以关键的训练技术调整，包括复值批归一化和权重初始化方案，以确保训练动力学的稳定性。实证评估分为三个阶段进行。首先，在标准图像数据集上对CVCNNs进行基准测试，它们表现出与实值CNNs相当的竞争力，即使在合成复扰动下也是如此。在第二个实验中，我们专注于使用梅尔频率倒谱系数(MFCC)进行音频分类。在实值MFCC上训练的CVCNNs略微优于实值CNN，而在输入工作流中保留相位则凸显了在没有架构修改的情况下利用相位的挑战。最后，第三个实验引入了GNNs通过边权重建模相位信息，其中包含相位在二元和多流派分类中都带来了可衡量的性能提升。这些结果强调了复值架构的表现能力，并确认了相位是音频处理应用中一个有意义且可利用的特征。


### 论文摘要

This study explores the design and application of Complex-Valued Convolutional Neural Networks (CVCNNs) in audio signal processing, with a focus on preserving and utilizing phase information often neglected in real-valued networks. We begin by presenting the foundational theoretical concepts of CVCNNs, including complex convolutions, pooling layers, Wirtinger-based differentiation, and various complex-valued activation functions. These are complemented by critical adaptations of training techniques, including complex batch normalization and weight initialization schemes, to ensure stability in training dynamics. Empirical evaluations are conducted across three stages. First, CVCNNs are benchmarked on standard image datasets, where they demonstrate competitive performance with real-valued CNNs, even under synthetic complex perturbations. Although our focus is audio signal processing, we first evaluate CVCNNs on image datasets to establish baseline performance and validate training stability before applying them to audio tasks. In the second experiment, we focus on audio classification using Mel-Frequency Cepstral Coefficients (MFCCs). CVCNNs trained on real-valued MFCCs slightly outperform real CNNs, while preserving phase in input workflows highlights challenges in exploiting phase without architectural modifications. Finally, a third experiment introduces GNNs to model phase information via edge weighting, where the inclusion of phase yields measurable gains in both binary and multi-class genre classification. These results underscore the expressive capacity of complex-valued architectures and confirm phase as a meaningful and exploitable feature in audio processing applications. While current methods show promise, especially with activations like cardioid, future advances in phase-aware design will be essential to leverage the potential of complex representations in neural networks.

---

## 91. NG-Router: Graph-Supervised Multi-Agent Collaboration for Nutrition Question Answering

**论文链接:** [http://arxiv.org/abs/2510.09854v1](http://arxiv.org/abs/2510.09854v1)

**作者:** Kaiwen Shi, Zheyuan Zhang, Zhengqing Yuan, Keerthiram Murugesan, Vincent Galass, Chuxu Zhang, Yanfang Ye

**发布时间:** 2025-10-10

### GPT解析

### 总结

这篇论文介绍了一个名为NG-Router的新框架，用于解决营养问答系统中的推理能力有限和上下文过载问题，通过知识图引导的多智能体协作提高系统性能。

### 背景

饮食在人类健康中起核心作用，营养问答系统为个性化饮食指导和预防饮食相关慢性疾病提供了有前景的路径。然而，现有方法面临单智能体系统推理能力有限、设计有效多智能体架构复杂以及上下文过载阻碍准确决策等挑战。

### 目的

开发一个能够有效处理复杂营养健康任务的多智能体推理框架，解决现有营养问答系统的局限性。

### 方法

提出Nutritional-Graph Router (NG-Router)框架，将营养问答视为有监督的、知识图引导的多智能体协作问题。该框架将智能体节点整合到异构知识图中，使用图神经网络学习任务感知的路由分布，并采用基于梯度的子图检索机制来解决上下文过载问题。

### 主要发现

在多个基准测试和骨干模型上的实验表明，NG-Router始终优于单智能体和集成基线方法，能够有效增强多跳和关系推理能力。

### 结论

NG-Router为复杂营养健康任务提供了一种有原则的领域感知多智能体推理方法，代表了营养问答系统的重要进步。

### 翻译

饮食在人类健康中扮演核心角色，营养问答为个性化饮食指导和预防饮食相关慢性疾病提供了有前景的路径。然而，现有方法面临两个基本挑战：单智能体系统的有限推理能力以及设计有效多智能体架构的复杂性，还有阻碍准确决策的上下文过载。我们引入了营养图路由器，这是一个新框架，将营养问答制定为一个有监督的、知识图引导的多智能体协作问题。营养图路由器将智能体节点整合到异构知识图中，并采用图神经网络来学习智能体上的任务感知路由分布，利用从经验智能体性能中获得的软监督。为了进一步解决上下文过载，我们提出了一种基于梯度的子图检索机制，在训练过程中识别显著证据，从而增强多跳和关系推理。在多个基准测试和骨干模型上的广泛实验表明，营养图路由器始终优于单智能体和集成基线，为复杂营养健康任务提供了一种有原则的领域感知多智能体推理方法。


### 论文摘要

Diet plays a central role in human health, and Nutrition Question Answering (QA) offers a promising path toward personalized dietary guidance and the prevention of diet-related chronic diseases. However, existing methods face two fundamental challenges: the limited reasoning capacity of single-agent systems and the complexity of designing effective multi-agent architectures, as well as contextual overload that hinders accurate decision-making. We introduce Nutritional-Graph Router (NG-Router), a novel framework that formulates nutritional QA as a supervised, knowledge-graph-guided multi-agent collaboration problem. NG-Router integrates agent nodes into heterogeneous knowledge graphs and employs a graph neural network to learn task-aware routing distributions over agents, leveraging soft supervision derived from empirical agent performance. To further address contextual overload, we propose a gradient-based subgraph retrieval mechanism that identifies salient evidence during training, thereby enhancing multi-hop and relational reasoning. Extensive experiments across multiple benchmarks and backbone models demonstrate that NG-Router consistently outperforms both single-agent and ensemble baselines, offering a principled approach to domain-aware multi-agent reasoning for complex nutritional health tasks.

---

## 92. Geo-Aware Models for Stream Temperature Prediction across Different Spatial Regions and Scales

**论文链接:** [http://arxiv.org/abs/2510.09500v1](http://arxiv.org/abs/2510.09500v1)

**作者:** Shiyuan Luo, Runlong Yu, Shengyu Chen, Yingda Fan, Yiqun Xie, Yanhua Li, Xiaowei Jia

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文提出了Geo-STARS，一个地理感知时空建模框架，用于预测不同流域和空间尺度的河流水温，通过引入地理感知嵌入解决了现有模型在跨区域和尺度推广方面的问题。

### 背景

理解环境生态系统对地球可持续管理至关重要，但现有基于物理和数据驱动的模型因数据异质性和有限观测样本而难以推广到不同空间区域和尺度。

### 目的

开发Geo-STARS框架，实现跨流域和空间尺度的河流水温预测，解决现有模型的泛化问题。

### 方法

Geo-STARS引入地理感知嵌入来捕捉跨空间区域和尺度的共享原则和模式，并将其整合到门控时空图神经网络中，使模型能够在地理和水文背景下学习复杂时空模式，即使数据稀疏或缺失。

### 主要发现

在美国东部海岸多个流域37年真实数据集上评估，Geo-STARS在区域和尺度上都表现出优于最先进基线的泛化性能。

### 结论

Geo-STARS为可扩展、数据高效的环境监测和决策制定提供了有前景的解决方案。

### 翻译

理解环境生态系统对我们星球的可持续管理至关重要。然而，现有的基于物理和数据驱动的模型往往由于现实环境生态系统中固有的数据异质性而无法推广到不同的空间区域和尺度。由于可用于模型训练的观测样本有限，这种推广问题进一步加剧。为解决这些问题，我们提出了Geo-STARS，一个用于预测不同流域和空间尺度河流水温的地理感知时空建模框架。Geo-STARS的主要创新是引入地理感知嵌入，它利用地理信息来明确捕捉跨空间区域和尺度的共享原则和模式。我们将地理感知嵌入进一步整合到门控时空图神经网络中。这种设计使模型能够在地理和水文背景的指导下学习复杂的时空模式，即使有稀疏或无观测数据也是如此。我们在预测河流水温方面评估了Geo-STARS的有效性，河流水质是水质量的主导因素。使用美国东部海岸多个流域跨越37年的真实世界数据集，Geo-STARS展示了其在区域和尺度上的优越泛化性能，优于最先进的基线。这些结果突显了Geo-STARS在可扩展、数据高效的环境监测和决策制定方面的前景。


### 论文摘要

Understanding environmental ecosystems is vital for the sustainable management of our planet. However,existing physics-based and data-driven models often fail to generalize to varying spatial regions and scales due to the inherent data heterogeneity presented in real environmental ecosystems. This generalization issue is further exacerbated by the limited observation samples available for model training. To address these issues, we propose Geo-STARS, a geo-aware spatio-temporal modeling framework for predicting stream water temperature across different watersheds and spatial scales. The major innovation of Geo-STARS is the introduction of geo-aware embedding, which leverages geographic information to explicitly capture shared principles and patterns across spatial regions and scales. We further integrate the geo-aware embedding into a gated spatio-temporal graph neural network. This design enables the model to learn complex spatial and temporal patterns guided by geographic and hydrological context, even with sparse or no observational data. We evaluate Geo-STARS's efficacy in predicting stream water temperature, which is a master factor for water quality. Using real-world datasets spanning 37 years across multiple watersheds along the eastern coast of the United States, Geo-STARS demonstrates its superior generalization performance across both regions and scales, outperforming state-of-the-art baselines. These results highlight the promise of Geo-STARS for scalable, data-efficient environmental monitoring and decision-making.

---

## 93. Precoder Design in Multi-User FDD Systems with VQ-VAE and GNN

**论文链接:** [http://arxiv.org/abs/2510.09495v1](http://arxiv.org/abs/2510.09495v1)

**作者:** Srikar Allaparapu, Michael Baur, Benedikt Böck, Michael Joham, Wolfgang Utschick

**发布时间:** 2025-10-10

**备注:** Submitted to IEEE ICASSP 2026

### GPT解析

### 总结

本文提出了一种基于向量量化-变分自编码器(VQ-VAE)的鲁棒预编码方法，用于频率双工系统，解决了传统高斯混合模型(GMM)组件数量随反馈比特数指数增长的问题。

### 背景

在频率双工系统中，鲁棒预编码的有效实现需要结合传播环境的统计信息，传统方法使用高斯混合模型和图神经网络设计特定站点的预编码器。

### 目的

开发一种新的预编码框架，解决GMM组件数量随反馈比特数指数增长的问题，并实现端到端训练，以提高多用户无线系统的性能。

### 方法

利用向量量化-变分自编码器(VQ-VAE)替代GMM，构建结合图神经网络(GNN)、VQ-VAE和pilot优化的端到端(E2E)模型。

### 主要发现

所提出的方法在多用户无线系统中实现了显著的速率增益，性能优于传统的子离散傅里叶变换(DFT) pilot 矩量和迭代预编码算法。

### 结论

所提出的框架支持使用更少pilot或反馈比特的系统部署，具有实际应用价值。

### 翻译

通过生成模型整合传播环境的学习统计信息，频率双工系统中的鲁棒预编码可以有效实现。我们基于先前成功结合高斯混合模型(GMM)和图神经网络(GNN)设计特定站点预编码器的工作。本文通过使用向量量化-变分自编码器(VQ-VAE)，避免了GMM的一个关键缺点，即GMM组件数量随反馈比特数呈指数增长。此外，VQ-VAE的深度学习架构允许我们将GNN与VQ-VAE以及pilot优化联合训练，形成一个端到端(E2E)模型，从而在多用户无线系统中实现显著的速率增益。仿真结果表明，所提出的框架优于涉及子离散傅里叶变换(DFT) pilot矩阵和迭代预编码算法的传统方法，能够支持部署具有更少pilot或反馈比特的系统。


### 论文摘要

Robust precoding is efficiently feasible in frequency division duplex (FDD) systems by incorporating the learnt statistics of the propagation environment through a generative model. We build on previous work that successfully designed site-specific precoders based on a combination of Gaussian mixture models (GMMs) and graph neural networks (GNNs). In this paper, by utilizing a vector quantized-variational autoencoder (VQ-VAE), we circumvent one of the key drawbacks of GMMs, i.e., the number of GMM components scales exponentially to the feedback bits. In addition, the deep learning architecture of the VQ-VAE allows us to jointly train the GNN together with VQ-VAE along with pilot optimization forming an end-to-end (E2E) model, resulting in considerable performance gains in sum rate for multi-user wireless systems. Simulations demonstrate the superiority of the proposed frameworks over the conventional methods involving the sub-discrete Fourier transform (DFT) pilot matrix and iterative precoder algorithms enabling the deployment of systems characterized by fewer pilots or feedback bits.

---

## 94. InterCorpRel-LLM: Enhancing Financial Relational Understanding with Graph-Language Models

**论文链接:** [http://arxiv.org/abs/2510.09735v1](http://arxiv.org/abs/2510.09735v1)

**作者:** Qianyou Sun, Jiexin Zheng, Bohan Jin, Lihua Chen, Yijie Peng

**发布时间:** 2025-10-10

### GPT解析

### 总结

这篇论文提出了一种结合图神经网络和大语言模型的跨模态框架InterCorpRel-LLM，用于识别企业间的供应链和竞争关系，在关系识别任务上显著优于基线模型。

### 背景

识别企业间的供应链和竞争关系对财务分析和公司治理至关重要，但由于企业数据的规模、稀疏性和上下文依赖性，这一任务具有挑战性。基于图的方法能捕捉结构但缺乏语义深度，而大语言模型擅长文本处理但在表示关系依赖方面能力有限。

### 目的

开发一个能够有效建模企业网络结构和语义信息的框架，以准确识别企业间的供应链和竞争关系。

### 方法

提出InterCorpRel-LLM，一个结合GNNs和LLMs的跨模态框架，使用来自FactSet供应链记录的专有数据集，并设计了三个定制训练任务：公司图匹配、行业分类和供应链关系预测。

### 主要发现

InterCorpRel-LLM在供应链关系识别任务上显著优于强基线模型(包括GPT-5)，仅使用7B参数主干和轻量级训练就达到了0.8543的F分数(基线为0.2287)。该模型还能推广到零样本竞争者识别，展示了捕捉微妙企业间动态的能力。

### 结论

InterCorpRel-LLM为分析师和战略家提供了绘制和推理复杂企业网络的强大工具，增强了动态市场中的决策制定和风险管理能力。

### 翻译

识别企业间的供应链和竞争关系对财务分析和公司治理至关重要，但由于企业数据的规模、稀疏性和上下文依赖性，这一任务仍然具有挑战性。基于图的方法能捕捉结构但缺乏语义深度，而大语言模型擅长文本但在表示关系依赖方面能力有限。为此，我们提出了InterCorpRel-LLM，这是一个结合GNNs和LLMs的跨模态框架，支持来自FactSet供应链记录的专有数据集和三个定制训练任务：公司图匹配、行业分类和供应链关系预测。这种设计能够有效建模结构和语义的联合表示。实验表明，在供应链关系识别任务上，InterCorpRel-LLM显著优于强基线模型(包括GPT-5)，仅使用7B参数主干和轻量级训练就达到了0.8543的F分数(对比基线的0.2287)。该模型还能推广到零样本竞争者识别，突显了其捕捉微妙企业间动态的能力。因此，我们的框架为分析师和战略家提供了绘制和推理复杂企业网络的强大工具，增强了动态市场中的决策制定和风险管理。


### 论文摘要

Identifying inter-firm relationships such as supply and competitive ties is critical for financial analysis and corporate governance, yet remains challenging due to the scale, sparsity, and contextual dependence of corporate data. Graph-based methods capture structure but miss semantic depth, while large language models (LLMs) excel at text but remain limited in their ability to represent relational dependencies. To address this, we propose InterCorpRel-LLM, a cross-modal framework that integrates GNNs with LLMs, supported by a proprietary dataset derived from FactSet supply chain records and three tailored training tasks: company graph matching, industry classification, and supply relation prediction. This design enables effective joint modeling of structure and semantics. Experiments show that InterCorpRel-LLM substantially outperforms strong baselines, including GPT-5, on a supply relation identification task, achieving an F-score of 0.8543 vs. 0.2287 with only a 7B-parameter backbone and lightweight training. The model also generalizes to zero-shot competitor identification, underscoring its ability to capture nuanced inter-firm dynamics. Our framework thus provides analysts and strategists with a robust tool for mapping and reasoning about complex corporate networks, enhancing decision-making and risk management in dynamic markets.

---

## 95. A Multimodal Approach to SME Credit Scoring Integrating Transaction and Ownership Networks

**论文链接:** [http://arxiv.org/abs/2510.09407v1](http://arxiv.org/abs/2510.09407v1)

**作者:** Sahab Zandi, Kamesh Korangi, Juan C. Moreno-Paredes, María Óskarsdóttir, Christophe Mues, Cristián Bravo

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文提出了一种基于图神经网络的中小企业信贷风险评估新方法，通过结合企业网络数据与传统数据提高了预测准确性，并揭示了企业间风险传染机制。

### 背景

中小企业在经济增长、就业和创新中发挥重要作用，但面临获取信贷的挑战，包括有限财务历史、抵押品约束和宏观经济冲击风险。中小企业常在相互关联的网络中运营，违约风险可能通过网络传播。

### 目的

提出并测试一种新的中小企业信贷风险建模方法，准确评估信贷风险，特别关注企业网络间的违约风险传播问题。

### 方法

使用来自知名金融机构的独特大型中小企业贷款数据集，采用图神经网络预测中小企业违约，基于企业间共同所有权和金融交易的多层网络数据进行建模，并将网络数据与传统结构化数据结合。

### 主要发现

结合网络数据和传统数据提高了申请评分性能；明确模拟了公司间的风险传染；连接的方向性和强度影响金融风险传染；网络数据具有预测能力；供应链网络使中小企业面临相关违约风险。

### 结论

网络数据对预测中小企业违约风险具有重要作用，供应链网络在使中小企业面临相关违约风险方面扮演关键角色。

### 翻译

中小企业(SMEs)在经济增长、就业和创新方面发挥着至关重要的作用。然而，由于有限的财务历史、抵押品约束和暴露于宏观经济冲击，它们在获取信贷方面往往面临重大挑战。这些挑战使得贷款人进行准确的信贷风险评估变得至关重要，特别是因为中小企业经常在相互关联的企业网络中运营，违约风险可以通过这些网络传播。本文提出并测试了一种新的中小企业信贷风险建模方法，使用来自知名金融机构的独特大型中小企业贷款数据集。具体而言，我们的方法采用图神经网络来预测中小企业违约，使用来自企业间共同所有权和金融交易的多层网络数据。我们表明，将此信息与传统结构化数据相结合不仅提高了申请评分性能，还明确模拟了公司之间的风险传染风险。进一步分析显示，这些连接的方向性和强度如何影响金融风险传染，从而提供了对潜在过程的更深入理解。我们的研究结果强调了网络数据的预测能力，以及供应链网络在使中小企业面临相关违约风险方面的作用。


### 论文摘要

Small and Medium-sized Enterprises (SMEs) are known to play a vital role in economic growth, employment, and innovation. However, they tend to face significant challenges in accessing credit due to limited financial histories, collateral constraints, and exposure to macroeconomic shocks. These challenges make an accurate credit risk assessment by lenders crucial, particularly since SMEs frequently operate within interconnected firm networks through which default risk can propagate. This paper presents and tests a novel approach for modelling the risk of SME credit, using a unique large data set of SME loans provided by a prominent financial institution. Specifically, our approach employs Graph Neural Networks to predict SME default using multilayer network data derived from common ownership and financial transactions between firms. We show that combining this information with traditional structured data not only improves application scoring performance, but also explicitly models contagion risk between companies. Further analysis shows how the directionality and intensity of these connections influence financial risk contagion, offering a deeper understanding of the underlying processes. Our findings highlight the predictive power of network data, as well as the role of supply chain networks in exposing SMEs to correlated default risk.

---

## 96. Deep Learning to Identify the Spatio-Temporal Cascading Effects of Train Delays in a High-Density Network

**论文链接:** [http://arxiv.org/abs/2510.09350v1](http://arxiv.org/abs/2510.09350v1)

**作者:** Vu Duc Anh Nguyen, Ziyue Li

**发布时间:** 2025-10-10

**DOI:** 10.1145/3764912.3770828

**备注:** Accepted at SIGSPATIAL 2025 - GeoAI Workshop

### GPT解析

### 总结

本文提出了一种名为XGeoAI的新框架，用于实时、可解释、多步列车延误预测，解决了现有研究中在网络规模多步自回归预测和实时可解释性方面的不足。

### 背景

铁路网络是现代经济的基石，但其运营效率持续受到列车延误级联效应的影响。准确预测延误传播对实时交通管理至关重要。

### 目的

开发并评估一种新的XGeoAI框架，用于实时、可解释、多步列车延误预测，为决策支持提供可靠工具。

### 方法

构建了一个两阶段自回归图注意力网络(GAT)模型，使用覆盖荷兰铁路网络40%以上的真实世界数据集训练。该模型将系统表示为操作事件的时空图，并增加了站台和车站拥堵等细粒度特征。通过顺序的k步前预测协议进行评估，模拟真实世界中预测误差可能累积的情况。

### 主要发现

提出的GATv2模型在纯误差指标(MAE)上比简单的持久性基线更具挑战性，但在分类延误事件方面实现了一贯的更高精度，这对可靠的决策支持工具至关重要。

### 结论

XGeoAI框架能够提供实时、可解释的多步列车延误预测，特别适合作为决策支持工具，在延误事件分类方面表现优异。

### 翻译

铁路网络的运营效率作为现代经济的基石，持续受到列车延误级联效应的破坏。准确预测这种延误传播是实时交通管理的关键挑战。尽管最近的研究利用图神经网络(GNNs)对铁路网络结构进行建模，但在开发能提供网络规模多步自回归预测的框架方面仍存在显著差距，同时缺乏决策支持所需的实时、可解释的解释。本文通过开发和评估一种新的XGeoAI框架来解决这一差距，该框架用于实时、可解释、多步列车延误预测。这项工作的核心是一个两阶段自回归图注意力网络(GAT)模型，在覆盖荷兰铁路网络40%以上的真实世界数据集上进行训练。该模型将系统表示为操作事件(到达和出发)的时空图，并增加了包括站台和车站拥堵在内的细粒度特征。为测试其在实时部署中的可行性，使用顺序的k步前预测协议对模型进行了严格评估，该协议模拟了预测误差可能累积的真实世界条件。结果表明，虽然提出的GATv2模型在纯误差指标(MAE)上比简单的持久性基线更具挑战性，但在分类延误事件方面实现了一贯的更高精度，这对可靠的决策支持工具至关重要。


### 论文摘要

The operational efficiency of railway networks, a cornerstone of modern economies, is persistently undermined by the cascading effects of train delays. Accurately forecasting this delay propagation is a critical challenge for real-time traffic management. While recent research has leveraged Graph Neural Networks (GNNs) to model the network structure of railways, a significant gap remains in developing frameworks that provide multi-step autoregressive forecasts at a network-wide scale, while simultaneously offering the live, interpretable explanations needed for decision support. This paper addresses this gap by developing and evaluating a novel XGeoAI framework for live, explainable, multi-step train delay forecasting. The core of this work is a two-stage, autoregressive Graph Attention Network (GAT) model, trained on a real-world dataset covering over 40% of the Dutch railway network. The model represents the system as a spatio-temporal graph of operational events (arrivals and departures) and is enriched with granular features, including platform and station congestion. To test its viability for live deployment, the model is rigorously evaluated using a sequential, k-step-ahead forecasting protocol that simulates real-world conditions where prediction errors can compound. The results demonstrate that while the proposed GATv2 model is challenged on pure error metrics (MAE) by a simpler Persistence baseline, it achieves consistently higher precision in classifying delay events -- a crucial advantage for a reliable decision support tool.

---

## 97. Physics-Informed High-order Graph Dynamics Identification Learning for Predicting Complex Networks Long-term Dynamics

**论文链接:** [http://arxiv.org/abs/2510.09082v2](http://arxiv.org/abs/2510.09082v2)

**作者:** Bicheng Wang, Junping Wang, Yibo Xue

**发布时间:** 2025-10-10

### GPT解析

### 总结

该研究提出了一种用于复杂网络长期动态预测的高阶网络动力学识别方法，解决了现有方法只能处理成对关系且预测模型要么缺乏准确性要么缺乏可解释性的问题。

### 背景

学习复杂网络动力学对于理解、建模和控制现实世界复杂系统至关重要。现有方法通常使用简单图来描述复杂网络中的关系，只能捕获成对关系，而网络中可能存在丰富的非成对结构化关系。

### 目的

提出一种用于复杂网络长期动态预测的高阶网络动力学识别方法，解决传统图机器学习只能处理成对关系的问题，同时提高预测的准确性和可解释性。

### 方法

引入动态超图学习来捕获复杂网络中的高阶非成对关系，提高复杂网络建模的准确性；提出物理数据双驱动动态预测模块，利用Koopman算子理论将非线性动力学微分方程转化为线性系统求解，同时利用物理信息神经微分方程方法确保动态演化符合物理定律。

### 主要发现

实验结果表明，该方法在公共数据集和自建产业链网络数据集上具有良好的预测精度和长期预测性能。

### 结论

该方法通过动态超图学习和双驱动动态预测模块，有效解决了复杂网络动态预测中的成对关系限制和预测准确性-可解释性权衡问题。

### 翻译

学习复杂网络动力学对于理解、建模和控制现实世界复杂系统至关重要。在预测复杂网络动态演化的任务中存在两个主要问题：一方面，现有方法通常使用简单图来描述复杂网络中的关系，然而这种方法只能捕获成对关系，而网络中可能存在丰富的非成对结构化关系。一阶GNN难以捕获动态非成对关系。另一方面，理论预测模型缺乏准确性，数据驱动预测模型缺乏可解释性。为解决上述问题，本文提出了一种用于复杂网络长期动态预测的高阶网络动力学识别方法。首先，为解决传统图机器学习只能处理成对关系的问题，引入动态超图学习来捕获复杂网络中的高阶非成对关系，提高复杂网络建模的准确性。然后，提出了物理数据双驱动动态预测模块。引入Koopman算子理论将复杂网络动态演化的非线性动力学微分方程转化为线性系统求解。同时，利用物理信息神经微分方程方法确保动态演化符合物理定律。双驱动动态预测模块确保了预测的准确性和可解释性。在公共数据集和自建产业链网络数据集上验证的实验结果表明，本文方法具有良好的预测精度和长期预测性能。


### 论文摘要

Learning complex network dynamics is fundamental to understanding, modelling and controlling real-world complex systems. There are two main problems in the task of predicting the dynamic evolution of complex networks: on the one hand, existing methods usually use simple graphs to describe the relationships in complex networks; however, this approach can only capture pairwise relationships, while there may be rich non-pairwise structured relationships in the network. First-order GNNs have difficulty in capturing dynamic non-pairwise relationships. On the other hand, theoretical prediction models lack accuracy and data-driven prediction models lack interpretability. To address the above problems, this paper proposes a higher-order network dynamics identification method for long-term dynamic prediction of complex networks. Firstly, to address the problem that traditional graph machine learning can only deal with pairwise relations, dynamic hypergraph learning is introduced to capture the higher-order non-pairwise relations among complex networks and improve the accuracy of complex network modelling. Then, a dual-driven dynamic prediction module for physical data is proposed. The Koopman operator theory is introduced to transform the nonlinear dynamical differential equations for the dynamic evolution of complex networks into linear systems for solving. Meanwhile, the physical information neural differential equation method is utilised to ensure that the dynamic evolution conforms to the physical laws. The dual-drive dynamic prediction module ensures both accuracy and interpretability of the prediction. Validated on public datasets and self-built industrial chain network datasets, the experimental results show that the method in this paper has good prediction accuracy and long-term prediction performance.

---

## 98. A review of cultural heritage inspection: Toward terahertz from mid-infrared region

**论文链接:** [http://arxiv.org/abs/2510.11521v1](http://arxiv.org/abs/2510.11521v1)

**作者:** Pengfei Zhu, Hai Zhang, Stefano Sfarra, Dazhi Yang, Xavier Maldague

**发布时间:** 2025-10-13

### GPT解析

### 总结

这篇综述探讨了用于检测和分析文化遗产文物的非侵入式成像方法，覆盖中远红外到太赫兹光谱区域，并总结了这些技术的应用和最新信号处理进展。

### 背景

文化遗产文物的保护和修复需要先进的无损检测技术，非侵入式成像(NII)方法在文化遗产研究中具有广泛应用。

### 目的

总结NII技术在文化遗产研究中的应用，以及信号处理技术的最新进展，特别是深度学习在自动化分析中的革命性作用。

### 方法

热红外域利用材料自发射特性；近红外通过外部照明增强表面细节；远红外和太赫兹技术以透射和反射模式穿透表面层；整合可见光和红外成像丰富诊断能力；应用深度学习进行自动分类、特征提取和缺陷检测。

### 主要发现

深度学习可通过监督和非监督学习可靠识别细微异常和材料变化，这些变化可能指示过去的修复或早期退化阶段；先进光谱成像、信号处理和神经网络的融合提供了更准确的数据驱动分析方法。

### 结论

先进光谱成像、复杂信号处理和深度神经网络的结合为文化遗产分析提供了更准确、高效和数据驱动的途径，最终支持更明智的保护和修复决策。

### 翻译

本综述探讨了覆盖中远红外至太赫兹光谱区域(最高约1000微米)的非侵入式成像(NII)方法，用于检测和分析文化遗产文物。在遵循普朗克定律的热红外域，材料的自发射揭示了材料的内在特性和内部退化。相比之下，在近红外范围内，外部照明增强了表面细节和颜料区分。远红外和太赫兹技术以透射和反射模式工作，通过穿透表面层提供亚表面结构和隐藏特征的补充见解。整合可见光和红外成像通过关联常规视觉评估与光谱信息进一步丰富了诊断能力。除了综述这些NII技术在文化遗产研究中的广泛应用外，本文还总结了信号处理的最新进展，包括硬件和软件发展。特别是，深度学习通过实现自动分类、特征提取、缺陷检测和超分辨率成像彻底改变了该领域。通过监督和非监督学习策略，神经网络可以可靠地识别指示过去修复或早期退化阶段的细微异常和材料变化。总之，先进光谱成像、复杂信号处理和深度神经网络的融合为文化遗产分析提供了更准确、高效和数据驱动的变革性途径，最终支持更明智的保护和修复决策。


### 论文摘要

This review explores non-invasive imaging (NII) methods covering the mid- and far-infrared to the terahertz spectral regions (up to approximately 1000 um) for the detection and analysis of cultural heritage artifacts. In the thermal infrared domain, where radiation follows Planck's law, the self-emission of materials reveals intrinsic properties and internal degradation. By contrast, in the near-infrared range, external illumination enhances surface details and pigment differentiation. Far-infrared and terahertz techniques, operating in both transmission and reflection modes, provide complementary insights by penetrating surface layers to uncover subsurface structures and concealed features. Integrating visible and infrared imaging further enriches diagnostic capabilities by correlating conventional visual assessments with spectral information. Beyond reviewing the wide applications of these NII techniques in cultural heritage research, this work also summarizes recent advances in signal processing, encompassing both hardware and software developments. In particular, deep learning has revolutionized the field by enabling automated classification, feature extraction, defect detection, and super-resolution imaging. Through supervised and unsupervised learning strategies, neural networks can reliably identify subtle anomalies and material variations indicative of past restorations or early stages of deterioration. In conclusion, the convergence of advanced spectral imaging, sophisticated signal processing, and deep neural networks offers a transformative pathway toward more accurate, efficient, and data-driven cultural heritage analysis, ultimately supporting more informed conservation and restoration decisions.

---

## 99. Nepali Sign Language Characters Recognition: Dataset Development and Deep Learning Approaches

**论文链接:** [http://arxiv.org/abs/2510.11243v1](http://arxiv.org/abs/2510.11243v1)

**作者:** Birat Poudel, Satyam Ghimire, Sijan Bhattarai, Saurav Bhandari, Suramya Sharma Dahal

**发布时间:** 2025-10-13

**备注:** 6 pages, 9 figures

### GPT解析

### 总结

这篇论文介绍了尼泊尔手语的首个基准数据集，并评估了深度学习方法在手语识别中的有效性。

### 背景

手语是听力和言语障碍人群的重要交流系统，但像尼泊尔手语这样的代表性不足的手语，其数字语言数据集资源仍然稀缺。

### 目的

创建尼泊尔手语的基准数据集，并评估深度学习方法在手语识别任务中的有效性。

### 方法

创建了一个包含36个手势类别，每类1500个样本的数据集，并使用MobileNetV2和ResNet50架构进行微调来评估分类性能。

### 主要发现

MobileNetV2达到了90.45%的分类准确率，ResNet50达到了88.78%的分类准确率，证明卷积神经网络在手语识别任务中有效，特别是在资源有限的环境中。

### 结论

这项工作代表了构建尼泊尔手语基准数据集和评估深度学习方法的第一系统性努力，突出了迁移学习和微调在推进代表性不足手语研究中的潜力。

### 翻译

手语是听力和言语障碍人士的重要交流系统。然而，对于代表性不足的手语，如尼泊尔手语，其数字语言数据集资源仍然稀缺。本研究引入了尼泊尔手语的第一个基准数据集，包含36个手势类别，每类1500个样本，旨在捕捉该语言的结构和视觉特征。为了评估识别性能，我们在数据集上微调了MobileNetV2和ResNet50架构，分别达到了90.45%和88.78%的分类准确率。这些发现证明了卷积神经网络在手语识别任务中的有效性，特别是在资源有限的环境中。据我们所知，这项工作代表了构建尼泊尔手语识别基准数据集和评估深度学习方法的第一系统性努力，突出了迁移学习和微调在推进代表性不足手语研究中的潜力。


### 论文摘要

Sign languages serve as essential communication systems for individuals with hearing and speech impairments. However, digital linguistic dataset resources for underrepresented sign languages, such as Nepali Sign Language (NSL), remain scarce. This study introduces the first benchmark dataset for NSL, consisting of 36 gesture classes with 1,500 samples per class, designed to capture the structural and visual features of the language. To evaluate recognition performance, we fine-tuned MobileNetV2 and ResNet50 architectures on the dataset, achieving classification accuracies of 90.45% and 88.78%, respectively. These findings demonstrate the effectiveness of convolutional neural networks in sign recognition tasks, particularly within low-resource settings. To the best of our knowledge, this work represents the first systematic effort to construct a benchmark dataset and assess deep learning approaches for NSL recognition, highlighting the potential of transfer learning and fine-tuning for advancing research in underexplored sign languages.

---

## 100. Transfer Learning with Distance Covariance for Random Forest: Error Bounds and an EHR Application

**论文链接:** [http://arxiv.org/abs/2510.10870v1](http://arxiv.org/abs/2510.10870v1)

**作者:** Chenze Li, Subhadeep Paul

**发布时间:** 2025-10-13

### GPT解析

### 总结

该研究提出了一种基于距离协方差的中心随机森林(CRF)方法，用于非参数回归的转移学习，特别是在源域和目标域回归函数在某些特征上稀疏不同的情况下。该方法通过两阶段CRF拟合过程，结合距离协方差特征权重，理论上证明了随机森林中转移学习的优势，并在模拟和实际医疗数据应用中验证了其有效性。

### 背景

随机森林是机器学习中的重要方法，在结构化表格数据方面优于其他竞争方法。然而，在源域和目标域分布不同的情况下，如何有效地应用随机森林进行转移学习仍然是一个挑战。

### 目的

开发一种基于中心随机森林的转移学习方法，利用距离协方差的特征权重，以解决源域和目标域回归函数在某些特征上稀疏不同的问题，提高跨域预测性能。

### 方法

首先使用源域训练的CRF预测目标域响应值并获取残差；然后使用另一个CRF拟合这些残差，特征分割概率与特征和残差之间的样本距离协方差成比例；推导均方误差率上界作为样本大小和差异维度的函数。

### 主要发现

理论上证明了随机森林中转移学习的好处；模拟结果表明CRF的结果在数值上也适用于标准随机森林；基于距离协方差的特征权重能提升RF性能；在预测ICU患者死亡率的应用中显示出显著优势。

### 结论

所提出的CRF方法结合距离协方差特征权重，能有效解决非参数回归中的转移学习问题，特别是在源域和目标域回归函数稀疏不同的情况下，为跨域预测提供了一种有效解决方案。

### 翻译

随机森林是一种重要的机器学习方法，由于其在对结构化表格数据的广泛应用中优于其他竞争方法。我们提出了一种使用基于距离协方差的中心随机森林(CRF)进行非参数回归转移学习的方法，假设未知源域和目标域的回归函数在某些特征上有所不同(稀疏不同)。我们的方法首先使用源域训练的CRF预测目标域的响应值并获取残差。然后，我们使用另一个CRF拟合这些残差，但特征分割概率与特征和残差之间的样本距离协方差成比例。我们推导了该方法的均方误差率上界作为样本大小和差异维度的函数，理论上证明了随机森林中转移学习的好处。在模拟中，我们证明CRF的结果在数值上也适用于具有数据驱动特征分割选择的标准随机森林(SRF)方法。除了转移学习，我们的结果还显示了基于距离协方差的权重在某些情况下对RF性能的益处。我们的方法在使用包含20万ICU患者电子健康记录的大型多医院数据集预测小型床位目标医院ICU患者死亡率方面显示出显著优势。


### 论文摘要

Random forest is an important method for ML applications due to its broad outperformance over competing methods for structured tabular data. We propose a method for transfer learning in nonparametric regression using a centered random forest (CRF) with distance covariance-based feature weights, assuming the unknown source and target regression functions are different for a few features (sparsely different). Our method first obtains residuals from predicting the response in the target domain using a source domain-trained CRF. Then, we fit another CRF to the residuals, but with feature splitting probabilities proportional to the sample distance covariance between the features and the residuals in an independent sample. We derive an upper bound on the mean square error rate of the procedure as a function of sample sizes and difference dimension, theoretically demonstrating transfer learning benefits in random forests. In simulations, we show that the results obtained for the CRFs also hold numerically for the standard random forest (SRF) method with data-driven feature split selection. Beyond transfer learning, our results also show the benefit of distance-covariance-based weights on the performance of RF in some situations. Our method shows significant gains in predicting the mortality of ICU patients in smaller-bed target hospitals using a large multi-hospital dataset of electronic health records for 200,000 ICU patients.

---

## 101. Quantifying Dataset Similarity to Guide Transfer Learning

**论文链接:** [http://arxiv.org/abs/2510.10866v1](http://arxiv.org/abs/2510.10866v1)

**作者:** Shudong Sun, Hao Helen Zhang

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了一种名为交叉学习分数(CLS)的新度量标准，用于衡量数据集相似性并提供迁移学习可迁移性的定量指导。该指标通过领域间的双向泛化性能评估数据集相似性，具有理论依据且计算高效。

### 背景

迁移学习已成为现代机器学习的基石，可通过利用相关领域知识提高模型学习效果。然而，从对齐不良的数据进行迁移可能损害性能而非提高，因此在实施前确定迁移是否 beneficial 至关重要。

### 目的

提出一种创新的度量标准来衡量数据集相似性，为迁移学习的可迁移性提供定量指导。

### 方法

提出交叉学习分数(CLS)度量标准，通过领域间双向泛化性能衡量数据集相似性；建立CLS与目标/源数据集决策边界余弦相似性的理论联系；引入通用框架将源数据集分类为正/模糊/负迁移区；扩展至深度学习编码器-头部架构。

### 主要发现

现有方法主要关注特征分布而忽略标签信息和预测关系；CLS可可靠预测迁移是提高还是降低性能；CLS为迁移学习数据选择提供原则性工具。

### 结论

通过在多种合成和真实世界任务上的广泛实验，证明CLS可有效预测迁移学习性能变化，指导数据选择决策。

### 翻译

迁移学习已成为现代机器学习的基石，因为它可以通过利用相关领域的知识来提高模型的学习效果。然而，从对齐不良的数据进行迁移可能会损害而非提高性能，因此在实施前确定迁移是否 beneficial 至关重要。本研究旨在通过提出一种创新的度量标准来衡量数据集相似性并提供可迁移性的定量指导来解决这一挑战。在现有文献中，现有方法主要关注特征分布而忽略了标签信息和预测关系，可能错失关键的可迁移性见解。相比之下，我们提出的度量标准交叉学习分数(CLS)通过领域之间的双向泛化性能来衡量数据集相似性。我们通过建立CLS与目标数据集和源数据集决策边界之间余弦相似性的联系，为CLS提供了理论依据。从计算角度看，CLS高效且计算快速，因为它绕过了高维问题中昂贵的分布估计问题。我们进一步引入了一个通用框架，根据CLS相对于基线误差将源数据集分类为正迁移区、模糊迁移区或负迁移区，从而能够做出明智的决策。此外，我们将这种方法扩展到深度学习中的编码器-头部架构，以更好地反映现代迁移流程。在多种合成和真实世界任务上的广泛实验表明，CLS可以可靠地预测迁移是提高还是降低性能，为迁移学习中的数据选择提供了原则性工具。


### 论文摘要

Transfer learning has become a cornerstone of modern machine learning, as it can empower models by leveraging knowledge from related domains to improve learning effectiveness. However, transferring from poorly aligned data can harm rather than help performance, making it crucial to determine whether the transfer will be beneficial before implementation. This work aims to address this challenge by proposing an innovative metric to measure dataset similarity and provide quantitative guidance on transferability. In the literature, existing methods largely focus on feature distributions while overlooking label information and predictive relationships, potentially missing critical transferability insights. In contrast, our proposed metric, the Cross-Learning Score (CLS), measures dataset similarity through bidirectional generalization performance between domains. We provide a theoretical justification for CLS by establishing its connection to the cosine similarity between the decision boundaries for the target and source datasets. Computationally, CLS is efficient and fast to compute as it bypasses the problem of expensive distribution estimation for high-dimensional problems. We further introduce a general framework that categorizes source datasets into positive, ambiguous, or negative transfer zones based on their CLS relative to the baseline error, enabling informed decisions. Additionally, we extend this approach to encoder-head architectures in deep learning to better reflect modern transfer pipelines. Extensive experiments on diverse synthetic and real-world tasks demonstrate that CLS can reliably predict whether transfer will improve or degrade performance, offering a principled tool for guiding data selection in transfer learning.

---

## 102. Image-to-Video Transfer Learning based on Image-Language Foundation Models: A Comprehensive Survey

**论文链接:** [http://arxiv.org/abs/2510.10671v1](http://arxiv.org/abs/2510.10671v1)

**作者:** Jinxuan Li, Chaolei Tan, Haoxuan Chen, Jianxin Ma, Jian-Fang Hu, Wei-Shi Zheng, Jianhuang Lai

**发布时间:** 2025-10-12

**备注:** Draft version, work in progress

### GPT解析

### 总结

本文是对图像到视频迁移学习这一新兴领域的首次全面综述，探讨了如何将图像-语言基础模型(ILFM)的能力扩展到视频领域，以减轻从头训练视频-语言基础模型的数据和计算需求。

### 背景

图像-语言基础模型在图像-文本理解和生成任务中表现出色，视频-文本研究的发展促使人们将基于图像的模型扩展到视频领域。图像到视频迁移学习范式可以显著降低训练成本。

### 目的

提供对图像到视频迁移学习领域的全面回顾，建立基于现有ILFM推进视频-文本学习的结构化路线图，并启发未来的研究方向。

### 方法

系统总结广泛使用的ILFM及其能力；将现有策略分为冻结特征和修改特征两类；详细阐述这些策略在从细粒度到粗粒度的各种视频-文本学习任务中的应用；通过实验分析不同迁移学习范式的有效性。

### 主要发现

图像到视频迁移学习能有效扩展ILFM能力到视频领域；不同迁移策略在各种视频任务上表现各异；从时空视频定位到视频问答等多种任务均可受益于这种迁移学习范式。

### 结论

图像到视频迁移学习是视频-文本学习领域的有前途方向，现有ILFM可作为视频-文本学习的基础，该领域仍面临挑战需要进一步研究。

### 翻译

图像-语言基础模型(ILFM)在图像-文本理解/生成任务中展示了显著的成功，提供了可迁移的多模态表示，能够泛化到各种下游基于图像的任务。视频-文本研究的进步促使人们越来越有兴趣将基于图像的模型扩展到视频领域。这种被称为图像到视频迁移学习的范式，成功减轻了从头开始训练视频-语言基础模型以满足视频-文本学习的巨大数据和计算需求。本调查提供了这一新兴领域的首次全面回顾，首先总结了广泛使用的ILFM及其能力。然后我们将现有的图像到视频迁移学习策略系统地分为两类：冻结特征和修改特征，取决于是否保留ILFM的原始表示或进行修改。基于图像到视频迁移的任务特定性质，本调查系统地阐述这些策略，并详细说明了它们在从细粒度(如时空视频定位)到粗粒度(如视频问答)的多种视频-文本学习任务中的应用。我们进一步提供了详细的实验分析，研究不同的图像到视频迁移学习范式在多种下游视频理解任务上的有效性。最后，我们确定了当前面临的挑战并指出了未来研究的有希望方向。通过提供全面和结构化的概述，本调查旨在建立基于现有ILFM推进视频-文本学习的结构化路线图，并启发这一快速发展领域的未来研究方向。


### 论文摘要

Image-Language Foundation Models (ILFM) have demonstrated remarkable success in image-text understanding/generation tasks, providing transferable multimodal representations that generalize across diverse downstream image-based tasks. The advancement of video-text research has spurred growing interest in extending image-based models to the video domain. This paradigm, known as image-to-video transfer learning, succeeds in alleviating the substantial data and computational requirements associated with training video-language foundation models from scratch for video-text learning. This survey provides the first comprehensive review of this emerging field, which begins by summarizing the widely used ILFM and their capabilities. We then systematically classify existing image-to-video transfer learning strategies into two categories: frozen features and modified features, depending on whether the original representations from ILFM are preserved or undergo modifications. Building upon the task-specific nature of image-to-video transfer, this survey methodically elaborates these strategies and details their applications across a spectrum of video-text learning tasks, ranging from fine-grained (e.g., spatio-temporal video grounding) to coarse-grained (e.g., video question answering). We further present a detailed experimental analysis to investigate the efficacy of different image-to-video transfer learning paradigms on a range of downstream video understanding tasks. Finally, we identify prevailing challenges and highlight promising directions for future research. By offering a comprehensive and structured overview, this survey aims to establish a structured roadmap for advancing video-text learning based on existing ILFM, and to inspire future research directions in this rapidly evolving domain.

---

## 103. Towards Cybersickness Severity Classification from VR Gameplay Videos Using Transfer Learning and Temporal Modeling

**论文链接:** [http://arxiv.org/abs/2510.10422v1](http://arxiv.org/abs/2510.10422v1)

**作者:** Jyotirmay Nag Setu, Kevin Desai, John Quarles

**发布时间:** 2025-10-12

### GPT解析

### 总结

本研究提出了一种利用迁移学习和长短期记忆网络预测虚拟现实网络疾病严重程度的方法，通过分析VR游戏视频实现68.4%的分类准确率。

### 背景

虚拟现实技术快速发展并在医疗、教育和娱乐等领域广泛应用，但网络疾病(类似晕动症的症状)持续阻碍VR的广泛接受。

### 目的

利用视频数据预测网络疾病严重程度，填补现有研究中基于视频特征预测网络疾病的空白。

### 方法

使用在ImageNet数据集上预训练的InceptionV3模型从VR游戏视频中提取高级视觉特征，然后将这些特征传递给LSTM网络以捕捉VR体验的时间动态并预测网络疾病严重程度随时间的变化。

### 主要发现

该方法有效利用了视频数据的时间序列特性，实现了68.4%的网络疾病严重度分类准确率，超越了仅使用视频数据训练的现有模型的性能。

### 结论

该研究为VR开发者提供了评估和减轻虚拟环境中网络疾病的实用工具，并为未来基于视频的时间建模研究奠定基础，以增强VR应用中的用户舒适度。

### 翻译

随着虚拟现实(VR)技术的快速发展，其在医疗、教育和娱乐等领域的应用显著增长。然而，持续存在的网络疾病问题(其症状类似于晕动症)继续阻碍VR的广泛接受。虽然近期研究探索了利用集成VR传感器(如眼睛和头部跟踪)数据的多模态深度学习方法，但使用基于视频特征预测网络疾病的研究仍然有限。在本研究中，我们通过使用在ImageNet数据集上预训练的InceptionV3模型，利用迁移学习从VR游戏视频中提取高级视觉特征，解决了这一研究空白。然后将这些特征传递给长短期记忆(LSTM)网络，以捕捉VR体验的时间动态并预测网络疾病严重程度随时间的变化。我们的方法有效利用了视频数据的时间序列特性，实现了68.4%的网络疾病严重度分类准确率。这超越了仅使用视频数据训练的现有模型的性能，为VR开发者提供了评估和减轻虚拟环境中网络疾病的实用工具。此外，这项工作为未来基于视频的时间建模研究奠定了基础，以增强VR应用中的用户舒适度。


### 论文摘要

With the rapid advancement of virtual reality (VR) technology, its adoption across domains such as healthcare, education, and entertainment has grown significantly. However, the persistent issue of cybersickness, marked by symptoms resembling motion sickness, continues to hinder widespread acceptance of VR. While recent research has explored multimodal deep learning approaches leveraging data from integrated VR sensors like eye and head tracking, there remains limited investigation into the use of video-based features for predicting cybersickness. In this study, we address this gap by utilizing transfer learning to extract high-level visual features from VR gameplay videos using the InceptionV3 model pretrained on the ImageNet dataset. These features are then passed to a Long Short-Term Memory (LSTM) network to capture the temporal dynamics of the VR experience and predict cybersickness severity over time. Our approach effectively leverages the time-series nature of video data, achieving a 68.4% classification accuracy for cybersickness severity. This surpasses the performance of existing models trained solely on video data, providing a practical tool for VR developers to evaluate and mitigate cybersickness in virtual environments. Furthermore, this work lays the foundation for future research on video-based temporal modeling for enhancing user comfort in VR applications.

---

## 104. Unveiling Gamer Archetypes through Multi modal feature Correlations and Unsupervised Learning

**论文链接:** [http://arxiv.org/abs/2510.10263v1](http://arxiv.org/abs/2510.10263v1)

**作者:** Moona Kanwal, Muhammad Sami Siddiqui, Syed Anael Ali

**发布时间:** 2025-10-11

**备注:** Submitted to Peer Review Journal

### GPT解析

### 总结

本研究提出了一种综合数据驱动框架，结合心理测量、行为分析和机器学习来识别游戏玩家人格类型，通过结构化调查和多种分析方法确定了四种玩家原型。

### 背景

游戏玩家画像研究对自适应游戏设计、行为理解和数字福祉至关重要。

### 目的

开发一个整合框架，揭示潜在游戏者人格类型，连接游戏动机与心理和健康结果。

### 方法

对250名参与者（含113名活跃游戏玩家）进行结构化调查；整合特征工程、关联网络、知识图谱分析和无监督聚类；使用多种相关性统计和网络中心度分析；结合多种降维技术和聚类算法；使用多种评估指标优化模型。

### 主要发现

PCA与K-Means（k=4）模型达到最佳聚类质量，确定了四种玩家原型：沉浸式社交故事追求者、自律优化者、战略系统导航者和竞争团队建设者。

### 结论

该研究提供了将相关性驱动网络洞察与无监督学习联系的可复现流程，行为相关性网络与聚类的结合提高了分类准确性，并提供了理解游戏动机与心理健康结果的整体视角。

### 翻译

游戏玩家画像为自适应游戏设计、行为理解和数字福祉提供了关键见解。本研究提出了一种综合的、数据驱动的框架，结合心理测量、行为分析和机器学习来揭示潜在的游戏者人格类型。对250名参与者（包括113名活跃游戏玩家）的结构化调查捕获了多维行为、动机和社会数据。分析流程整合了特征工程、关联网络、知识图谱分析和无监督聚类，以提取有意义的模式。相关性统计量化特征关联，网络中心度指导特征选择。将降维技术与聚类算法相结合，使用多种指数进行评估。PCA与K-Means模型实现了最佳聚类质量，确定了四种原型。这项研究贡献了一个可复现的流程，将相关性驱动的网络洞察与无监督学习联系起来。行为相关性网络与聚类的整合不仅提高了分类准确性，还提供了整体视角，将游戏动机与心理和健康结果联系起来。


### 论文摘要

Profiling gamers provides critical insights for adaptive game design, behavioral understanding, and digital well-being. This study proposes an integrated, data-driven framework that combines psychological measures, behavioral analytics, and machine learning to reveal underlying gamer personas. A structured survey of 250 participants, including 113 active gamers, captured multidimensional behavioral, motivational, and social data. The analysis pipeline integrated feature engineering, association-network, knowledge-graph analysis, and unsupervised clustering to extract meaningful patterns. Correlation statistics uses Cramers V, Tschuprows T, Theils U, and Spearmans quantified feature associations, and network centrality guided feature selection. Dimensionality-reduction techniques such as PCA, SVD, t-SNE are coupled with clustering algorithms like K-Means, Agglomerative, Spectral, DBSCAN, evaluated using Silhouette, Calinski Harabasz, and Davies Bouldin indices. The PCA with K-Means with k = 4 model achieved optimal cluster quality with Silhouette = 0.4, identifying four archetypes as Immersive Social Story-Seekers, Disciplined Optimizers, Strategic Systems Navigators, and Competitive Team-Builders. This research contributes a reproducible pipeline that links correlation-driven network insights with unsupervised learning. The integration of behavioral correlation networks with clustering not only enhances classification accuracy but also offers a holistic lens to connect gameplay motivations with psychological and wellness outcomes.

---

## 105. Deep Learning of the Biswas-Chatterjee-Sen Model

**论文链接:** [http://arxiv.org/abs/2510.09446v1](http://arxiv.org/abs/2510.09446v1)

**作者:** J. F. Silva Neto, D. S. M. Alencar, L. T. Brito, G. A. Alves, F. W. S. Lima, A. Macedo-Filho, R. S. Ferreira, T. F. A. Alves

**发布时间:** 2025-10-10

**备注:** 11 pages, 8 figures. arXiv admin note: text overlap with  arXiv:2509.14155

### GPT解析

### 总结

本研究使用深度学习技术研究了动力学连续意见动力学模型的临界性质，通过神经网络、主成分分析和变分自编码器等方法成功识别了临界点并研究了相变行为。

### 背景

动力学连续意见动力学模型是研究集体行为和相变的重要模型，其中系统由连续自旋变量组成，取值区间为[-1,1]，类似于自旋系统中的意见形成过程。

### 目的

探究动力学连续意见动力学模型的临界性质，准确识别临界点，并研究相变行为，特别是使用深度学习技术来发现传统方法可能难以捕捉的规律。

### 方法

使用深度神经网络在动力学蒙特卡洛模拟生成的自旋构型数据上进行训练；采用主成分分析进行无监督学习；实现变分自编码器并通过损失函数研究相变；定义真实数据与重构数据之间的相关函数。

### 主要发现

深度神经网络能够准确识别方形和三角形晶格上的临界点；主成分分析可以重现磁化现象并估计临界指数；变分自编码器的损失函数可作为序参量；真实数据与重构数据之间的相关函数在临界点表现出普适性。

### 结论

深度学习技术为研究动力学连续意见动力学模型的临界性质提供了有效工具，能够准确识别临界点并揭示相变过程中的普适性行为。

### 翻译

我们使用深度学习技术研究动力学连续意见动力学模型的临界性质。该系统由区间[-1,1]内的N个连续自旋变量组成。在通过动力学蒙特卡洛模拟生成的自旋构型数据上训练密集神经网络，能够准确识别方形和三角形晶格上的临界点。使用主成分分析进行经典无监督学习重现了磁化现象，并允许估计临界指数。此外，实现变分自编码器通过损失函数研究相变，该损失函数表现为序参量。定义了真实数据和重构数据之间的相关函数，发现在临界点具有普适性。


### 论文摘要

We investigate the critical properties of kinetic continuous opinion dynamics using deep learning techniques. The system consists of $N$ continuous spin variables in the interval $[-1,1]$. Dense neural networks are trained on spin configuration data generated via kinetic Monte Carlo simulations, accurately identifying the critical point on both square and triangular lattices. Classical unsupervised learning with principal component analysis reproduces the magnetization and allows estimation of critical exponents. Additionally, variational autoencoders are implemented to study the phase transition through the loss function, which behaves as an order parameter. A correlation function between real and reconstructed data is defined and found to be universal at the critical point.

---

## 106. deep-REMAP: Probabilistic Parameterization of Stellar Spectra Using Regularized Multi-Task Learning

**论文链接:** [http://arxiv.org/abs/2510.09362v1](http://arxiv.org/abs/2510.09362v1)

**作者:** Sankalp Gilda

**发布时间:** 2025-10-10

**备注:** 14 pages. Accepted for publication in RASTI

### GPT解析

### 总结

该研究开发了deep-REMAP，一种用于从观测光谱预测恒星大气参数的深度学习框架，结合正则化多任务学习和迁移学习，能够准确预测恒星的有效温度、表面重力和金属licity，具有可解释性和鲁棒性，可扩展到其他调查和合成库。

### 背景

在天文学调查数据量爆炸式增长的时代，传统的光谱分析方法已达到其处理能力的极限。

### 目的

开发一种新的深度学习框架，用于从观测光谱中预测恒星大气参数。

### 方法

在PHOENIX合成光谱库上训练深度卷积神经网络，使用迁移学习在MARVELS调查的一小部分观测FGK矮星光谱上微调模型，然后应用于732个未表征的FGK巨星候选体。结合非对称损失函数和嵌入损失，构建回归分类框架。

### 主要发现

在30个MARVELS校准恒星上验证时，deep-REMAP准确恢复了有效温度、表面重力和金属licity，例如在有效温度上实现了约75 K的精度。该框架具有可解释性，对参数不平衡具有鲁棒性，能够捕捉非高斯不确定性。

### 结论

虽然最初为MARVELS调查开发，但deep-REMAP框架可扩展到其他调查和合成库，为恒星特征表征提供了一种强大且自动化的方法。

### 翻译

在调查量爆炸式增长的时代，传统的光谱分析方法已达到其极限。为此，我们开发了deep-REMAP，一种新颖的深度学习框架，利用正则化多任务方法从观测光谱预测恒星大气参数。我们在PHOENIX合成光谱库上训练深度卷积神经网络，并使用迁移学习在MARVELS调查的一小部分观测FGK矮星光谱上微调模型。然后我们将该模型应用于同一调查中的732个未表征的FGK巨星候选体。在30个MARVELS校准恒星上进行验证时，deep-REMAP准确恢复了有效温度、表面重力和金属licity，例如在有效温度上实现了约75 K的精度。通过结合非对称损失函数和嵌入损失，我们的回归分类框架具有可解释性，对参数不平衡具有鲁棒性，并且能够捕捉非高斯不确定性。虽然是为MARVELS开发的，但deep-REMAP框架可扩展到其他调查和合成库，展示了一种强大且自动化的恒星特征表征途径。


### 论文摘要

In the era of exploding survey volumes, traditional methods of spectroscopic analysis are being pushed to their limits. In response, we develop deep-REMAP, a novel deep learning framework that utilizes a regularized, multi-task approach to predict stellar atmospheric parameters from observed spectra. We train a deep convolutional neural network on the PHOENIX synthetic spectral library and use transfer learning to fine-tune the model on a small subset of observed FGK dwarf spectra from the MARVELS survey. We then apply the model to 732 uncharacterized FGK giant candidates from the same survey. When validated on 30 MARVELS calibration stars, deep-REMAP accurately recovers the effective temperature ($T_{\rm{eff}}$), surface gravity ($\log \rm{g}$), and metallicity ([Fe/H]), achieving a precision of, for instance, approximately 75 K in $T_{\rm{eff}}$. By combining an asymmetric loss function with an embedding loss, our regression-as-classification framework is interpretable, robust to parameter imbalances, and capable of capturing non-Gaussian uncertainties. While developed for MARVELS, the deep-REMAP framework is extensible to other surveys and synthetic libraries, demonstrating a powerful and automated pathway for stellar characterization.

---

## 107. MPA-DNN: Projection-Aware Unsupervised Learning for Multi-period DC-OPF

**论文链接:** [http://arxiv.org/abs/2510.09349v1](http://arxiv.org/abs/2510.09349v1)

**作者:** Yeomoon Kim, Minsoo Kim, Jip Kim

**发布时间:** 2025-10-10

### GPT解析

### 总结

该研究提出了一种MPA-DNN方法，解决了深度神经网络在最优潮流问题中无法满足关键操作约束的问题，特别是在涉及时间间耦合的情况下。该方法通过引入投影层强制物理可行性，实现了无需标记数据的端到端学习，实验表明其在变化负载条件下能实现接近最优性能并严格满足所有约束。

### 背景

现代电力系统在高比例可再生能源和储能的情况下，最优潮流(OPF)操作的可行性和效率变得越来越重要。深度神经网络作为OPF求解器的快速代理很有前景，但常常无法满足关键的操作约束，特别是涉及时间间耦合的约束。

### 目的

解决深度神经网络在最优潮流问题中无法满足关键操作约束的问题，特别是涉及时间间耦合的约束，如发电机爬坡限制和储能操作。

### 方法

提出一个多周期投影感知深度神经网络(MPA-DNN)，它在网络中集成了一个用于多周期调度的投影层，通过投影强制物理可行性，实现端到端的符合约束的调度轨迹学习，无需依赖标记数据。

### 主要发现

实验结果表明，所提出的方法在变化的负载条件下实现了接近最优的性能，同时严格满足所有约束。

### 结论

MPA-DNN方法能够确保最优潮流操作的可行性和效率，特别是在高比例可再生能源和储能的现代电力系统中。

### 翻译

确保现代电力系统中高比例可再生能源和储能情况下的最优潮流(OPF)操作的可行性和效率变得越来越重要。虽然深度神经网络(DNN)已成为OPF求解器有前景的快速代理，但它们常常无法满足关键操作约束，特别是涉及时间间耦合的约束，如发电机爬坡限制和储能操作。为了解决这些问题，我们提出了一种多周期投影感知深度神经网络(MPA-DNN)，它在网络中集成了一个用于多周期调度的投影层。通过这样做，我们的模型通过投影强制物理可行性，使得无需依赖标记数据即可进行端到端的符合约束的调度轨迹学习。实验结果表明，所提出的方法在变化的负载条件下实现了接近最优的性能，同时严格满足所有约束。


### 论文摘要

Ensuring both feasibility and efficiency in optimal power flow (OPF) operations has become increasingly important in modern power systems with high penetrations of renewable energy and energy storage. While deep neural networks (DNNs) have emerged as promising fast surrogates for OPF solvers, they often fail to satisfy critical operational constraints, especially those involving inter-temporal coupling, such as generator ramping limits and energy storage operations. To deal with these issues, we propose a Multi-Period Projection-Aware Deep Neural Network (MPA-DNN) that incorporates a projection layer for multi-period dispatch into the network. By doing so, our model enforces physical feasibility through the projection, enabling end-to-end learning of constraint-compliant dispatch trajectories without relying on labeled data. Experimental results demonstrate that the proposed method achieves near-optimal performance while strictly satisfying all constraints in varying load conditions.

---

## 108. Rewiring Development in Brain Segmentation: Leveraging Adult Brain Priors for Enhancing Infant MRI Segmentation

**论文链接:** [http://arxiv.org/abs/2510.09306v1](http://arxiv.org/abs/2510.09306v1)

**作者:** Alemu Sisay Nigru, Michele Svanera, Austin Dibble, Connor Dalby, Mattia Savardi, Sergio Benini

**发布时间:** 2025-10-10

### GPT解析

### 总结

LODi是一种利用成人脑部MRI分割模型先验知识增强婴儿脑部MRI分割性能的新框架，通过迁移学习和领域适应策略，实现了快速、准确、年龄自适应的分割，减轻了扫描仪和特定地点的偏差。

### 背景

婴儿脑部MRI分割对于研究早期神经发育和诊断神经系统疾病至关重要，但面临受试者解剖结构不断变化、运动伪影以及高质量标记数据稀缺等挑战。

### 目的

开发一种能够利用成人脑部MRI分割模型先验知识来提高婴儿脑部MRI分割性能的方法，解决婴儿脑部MRI分割中的关键挑战。

### 方法

LODi框架首先在大量成人脑部MRI数据上预训练分割模型，然后通过迁移学习和领域适应策略逐步适应0-2岁人群，利用弱监督学习和FreeSurfer获得的银标准真实标签进行调整，并引入分层特征细化和多级别一致性约束的新训练策略。

### 主要发现

在内部和外部数据集上的广泛实验表明，LODi方法优于传统监督学习和领域特定模型，能够实现快速、准确、年龄自适应的分割，同时减轻扫描仪和特定地点的偏差。

### 结论

利用成人脑部先验作为年龄灵活神经成像分析的基础具有显著优势，为整个生命周期内更可靠和通用的脑部MRI分割铺平了道路。

### 翻译

婴儿脑部MRI的精确分割对于研究早期神经发育和诊断神经系统疾病至关重要。然而，由于受试者解剖结构的持续变化、运动伪影以及高质量标记数据的稀缺，它仍然是一个基本挑战。在这项工作中，我们提出了LODi，一个新颖的框架，利用成人脑部MRI分割模型的先验知识来增强婴儿扫描的分割性能。鉴于大量公开可用的成人脑部MRI数据，我们在大型成人数据集上预训练分割模型作为起点。通过迁移学习和领域适应策略，我们将模型逐步适应0-2岁人群，使其能够考虑婴儿扫描中典型的解剖和成像变异性。成人模型的调整是通过在婴儿脑部扫描上进行弱监督学习进行的，利用使用FreeSurfer获得的银标准真实标签。通过引入一种结合分层特征细化和多级别一致性约束的新训练策略，我们的方法能够实现快速、准确、年龄自适应的分割，同时减轻扫描仪和特定地点的偏差。在内部和外部数据集上的广泛实验证明了我们的方法优于传统监督学习和领域特定模型。我们的研究结果突显了利用成人脑部先验作为年龄灵活神经成像分析基础的优势，为整个生命周期内更可靠和通用的脑部MRI分割铺平了道路。


### 论文摘要

Accurate segmentation of infant brain MRI is critical for studying early neurodevelopment and diagnosing neurological disorders. Yet, it remains a fundamental challenge due to continuously evolving anatomy of the subjects, motion artifacts, and the scarcity of high-quality labeled data. In this work, we present LODi, a novel framework that utilizes prior knowledge from an adult brain MRI segmentation model to enhance the segmentation performance of infant scans. Given the abundance of publicly available adult brain MRI data, we pre-train a segmentation model on a large adult dataset as a starting point. Through transfer learning and domain adaptation strategies, we progressively adapt the model to the 0-2 year-old population, enabling it to account for the anatomical and imaging variability typical of infant scans. The adaptation of the adult model is carried out using weakly supervised learning on infant brain scans, leveraging silver-standard ground truth labels obtained with FreeSurfer. By introducing a novel training strategy that integrates hierarchical feature refinement and multi-level consistency constraints, our method enables fast, accurate, age-adaptive segmentation, while mitigating scanner and site-specific biases. Extensive experiments on both internal and external datasets demonstrate the superiority of our approach over traditional supervised learning and domain-specific models. Our findings highlight the advantage of leveraging adult brain priors as a foundation for age-flexible neuroimaging analysis, paving the way for more reliable and generalizable brain MRI segmentation across the lifespan.

---

## 109. Modern Deep Learning Approaches for Cricket Shot Classification: A Comprehensive Baseline Study

**论文链接:** [http://arxiv.org/abs/2510.09187v1](http://arxiv.org/abs/2510.09187v1)

**作者:** Sungwoo Kang

**发布时间:** 2025-10-10

### GPT解析

### 总结

本研究对板球击球分类进行了全面的基线研究，比较了七种不同的深度学习方法，发现在学术文献中报告的性能与实际实现结果之间存在显著差距，而现代架构结合系统优化可实现92.25%的准确率。

### 背景

板球击球分类在体育视频分析中仍然是一个具有挑战性的问题，需要有效地建模空间和时间特征。

### 目的

进行一项全面的基线研究，比较七种不同的深度学习方法，用于板球击球分类。

### 方法

在统一的基准上实现了和系统评估了传统的CNN-LSTM架构、基于注意力的模型、视觉变换器、迁移学习方法以及现代的EfficientNet-GRU组合。

### 主要发现

学术文献中报告的准确率（96%、99.2%和93%）与实际重新实现的准确率（46.0%、55.6%和57.7%）之间存在显著差距；现代最先进的方法（EfficientNet-B0与基于GRU的时间模型组合）达到了92.25%的准确率。

### 结论

现代架构和系统优化可以带来实质性的改进；遵循现代MLOps实践并提供可重现的研究平台突显了标准化评估协议在体育视频分析研究中的重要性。

### 翻译

从视频序列中进行板球击球分类在体育视频分析中仍然是一个具有挑战性的问题，需要有效地建模空间和时间特征。本文首次进行了全面的基线研究，比较了四种不同研究范式下的七种深度学习方法。我们在统一的基准上实现了并系统评估了传统的CNN-LSTM架构、基于注意力的模型、视觉变换器、迁移学习方法以及现代的EfficientNet-GRU组合。我们研究的一个关键发现是学术文献中的声明与实际实现结果之间存在显著性能差距。虽然之前的论文报告了96%（Balaji LRCN）、99.2%（IJERCSE）和93%（Sensors）的准确率，但我们的标准化重新实现分别实现了46.0%、55.6%和57.7%的准确率。我们的现代最先进方法，结合EfficientNet-B0和基于GRU的时间模型，实现了92.25%的准确率，证明使用现代架构和系统优化可以实现实质性改进。所有实现都遵循使用PyTorch Lightning的现代MLOps实践，提供了一个可重现的研究平台，突显了标准化评估协议在体育视频分析研究中的重要性。


### 论文摘要

Cricket shot classification from video sequences remains a challenging problem in sports video analysis, requiring effective modeling of both spatial and temporal features. This paper presents the first comprehensive baseline study comparing seven different deep learning approaches across four distinct research paradigms for cricket shot classification. We implement and systematically evaluate traditional CNN-LSTM architectures, attention-based models, vision transformers, transfer learning approaches, and modern EfficientNet-GRU combinations on a unified benchmark. A critical finding of our study is the significant performance gap between claims in academic literature and practical implementation results. While previous papers reported accuracies of 96\% (Balaji LRCN), 99.2\% (IJERCSE), and 93\% (Sensors), our standardized re-implementations achieve 46.0\%, 55.6\%, and 57.7\% respectively. Our modern SOTA approach, combining EfficientNet-B0 with a GRU-based temporal model, achieves 92.25\% accuracy, demonstrating that substantial improvements are possible with modern architectures and systematic optimization. All implementations follow modern MLOps practices with PyTorch Lightning, providing a reproducible research platform that exposes the critical importance of standardized evaluation protocols in sports video analysis research.

---

## 110. A Novel Multi-branch ConvNeXt Architecture for Identifying Subtle Pathological Features in CT Scans

**论文链接:** [http://arxiv.org/abs/2510.09107v1](http://arxiv.org/abs/2510.09107v1)

**作者:** Irash Perera, Uthayasanker Thayasivam

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文介绍了一种专门用于医学图像分析的新型多分支ConvNeXt架构，通过整合三种并行分支特征提取方法，在COVID-19诊断任务上取得了优异性能，超越了之前报道的所有模型。

### 背景

智能医学影像分析在辅助临床诊断中起着至关重要的作用，特别是在识别细微病理特征方面，但医学图像分析面临着独特的挑战。

### 目的

开发一种针对医学图像分析独特挑战的新型多分支ConvNeXt架构，应用于COVID-19诊断，并提供一个可推广的框架用于从CT扫描中分类广泛的病理。

### 方法

提出包含严格端到端流程的模型，包括细致的数据预处理和增强，采用两阶段训练策略有效利用迁移学习，架构整合了全局平均池化、全局最大池化和新的注意力加权池化三个并行分支的特征，在2609个CT切片的组合数据集上训练和验证。

### 主要发现

模型在验证集上表现出色，COVID-19病例的ROC-AUC达到0.9937，验证准确率为0.9757，F1得分为0.9825，超越了此数据集上之前报道的所有模型。

### 结论

现代多分支架构与谨慎的数据处理相结合，可以实现与当代最先进模型相当或更好的性能，证明先进的深度学习技术为稳健医疗诊断提供了有效解决方案。

### 翻译

医学影像的智能分析在辅助临床诊断中起着至关重要的作用，特别是对于识别细微的病理特征。本文介绍了一种新型的多分支ConvNeXt架构，专门为医学图像分析的细微挑战而设计。虽然在此应用于COVID-19诊断的具体问题，但该方法为从CT扫描中广泛分类病理提供了一种可推广的框架。所提出的模型包含严格的端到端流程，从细致的数据预处理和增强到利用迁移学习有效利用的纪律性两阶段训练策略。该架构独特地整合了从三个并行分支提取的特征：全局平均池化、全局最大池化以及新的注意力加权池化机制。该模型在来自两个不同数据集的2609个CT切片的组合数据集上进行了训练和验证。实验结果表明，在验证集上表现出色，COVID-19病例的最终ROC-AUC达到0.9937，验证准确率为0.9757，F1得分为0.9825，超越了这个数据集上之前报道的所有模型。


### 论文摘要

Intelligent analysis of medical imaging plays a crucial role in assisting clinical diagnosis, especially for identifying subtle pathological features. This paper introduces a novel multi-branch ConvNeXt architecture designed specifically for the nuanced challenges of medical image analysis. While applied here to the specific problem of COVID-19 diagnosis, the methodology offers a generalizable framework for classifying a wide range of pathologies from CT scans. The proposed model incorporates a rigorous end-to-end pipeline, from meticulous data preprocessing and augmentation to a disciplined two-phase training strategy that leverages transfer learning effectively. The architecture uniquely integrates features extracted from three parallel branches: Global Average Pooling, Global Max Pooling, and a new Attention-weighted Pooling mechanism. The model was trained and validated on a combined dataset of 2,609 CT slices derived from two distinct datasets. Experimental results demonstrate a superior performance on the validation set, achieving a final ROC-AUC of 0.9937, a validation accuracy of 0.9757, and an F1-score of 0.9825 for COVID-19 cases, outperforming all previously reported models on this dataset. These findings indicate that a modern, multi-branch architecture, coupled with careful data handling, can achieve performance comparable to or exceeding contemporary state-of-the-art models, thereby proving the efficacy of advanced deep learning techniques for robust medical diagnostics.

---

## 111. Transfer Learning-Enabled Efficient Raman Pump Tuning under Dynamic Launch Power for C+L Band Transmission

**论文链接:** [http://arxiv.org/abs/2510.09047v1](http://arxiv.org/abs/2510.09047v1)

**作者:** Jiaming Liu, Rui Wang, JinJiang Li, Hong Lin, Jing Zhang, Kun Qiu

**发布时间:** 2025-10-10

**备注:** Asia Communications and Photonics Conference 2025

### GPT解析

### 总结

该研究提出了一种基于迁移学习的Transformer框架，用于C+L波段系统中的精确建模和拉曼泵浦设计。

### 背景

C+L波段系统中的精确建模和拉曼泵浦设计面临挑战，需要新的方法来提高性能。

### 目的

开发一种能够同时实现精确建模和拉曼泵浦设计的框架，应用于C+L波段系统。

### 方法

使用基于迁移学习的Transformer框架。

### 主要发现

建模的均方根误差在0.22分贝以内，峰峰值光信噪比变化/偏差在0.86/0.1分贝以内。

### 结论

该框架能够有效实现C+L波段系统中的精确建模和拉曼泵浦设计，性能指标达到预期。

### 翻译

我们提出了一种基于迁移学习的Transformer框架，用于在C+L波段系统中同时实现精确建模和拉曼泵浦设计。建模的均方根误差和峰峰值光信噪比变化/偏差分别在0.22分贝和0.86/0.1分贝以内。


### 论文摘要

We propose a transfer learning-enabled Transformer framework to simultaneously realize accurate modeling and Raman pump design in C+L-band systems. The RMSE for modeling and peak-to-peak GSNR variation/deviation is within 0.22 dB and 0.86/0.1 dB, respectively.

---

## 112. Exploring Cross-Lingual Knowledge Transfer via Transliteration-Based MLM Fine-Tuning for Critically Low-resource Chakma Language

**论文链接:** [http://arxiv.org/abs/2510.09032v1](http://arxiv.org/abs/2510.09032v1)

**作者:** Adity Khisa, Nusrat Jahan Lia, Tasnim Mahfuz Nafis, Zarif Masud, Tanzir Pial, Shebuti Rayana, Ahmedul Kabir

**发布时间:** 2025-10-10

### GPT解析

### 总结

本研究针对数据有限的Chakma语言，引入了一个新的孟加拉语转写的Chakma语料库，并对多种transformer模型进行了微调。实验表明，微调后的多语言模型在适应Chakma时表现优异，且数据质量对模型性能有重要影响。

### 背景

Chakma作为一种达罗毗荼语系语言，可用数据有限，在语言模型中代表性不足。

### 目的

引入一个新的上下文连贯的孟加拉语转写的Chakma语料库，并利用该数据集对多种模型进行微调，以提高Chakma语言在语言模型中的表现。

### 方法

从Chakma文学中整理了一个经过母语人士验证的语料库，并使用该数据集在掩码语言建模任务上微调了六种基于编码器的多语言和区域transformer模型（mBERT、XLM-RoBERTa、DistilBERT、DeBERTaV3、BanglaBERT和IndicBERT）。

### 主要发现

微调后的多语言模型在适应孟加拉语转写的Chakma时优于其预训练版本，达到高达73.54%的标记准确度和低至2.90的困惑度。分析还强调了数据质量对模型性能的影响，以及OCR管道在形态丰富的印度文字方面的局限性。

### 结论

孟加拉语转写的Chakma对于Chakma语言的迁移学习非常有效，研究人员发布了手动验证的单语数据集以鼓励对低资源语言的多语言语言建模进行进一步研究。

### 翻译

作为一种达罗毗荼语系语言且可用数据有限，Chakma在语言模型中仍然代表性不足。在这项工作中，我们引入了一个新颖的上下文连贯的孟加拉语转写的Chakma语料库，该语料库从Chakma文学中精心整理，并由母语人士验证。使用这个数据集，我们在掩码语言建模任务上微调了六种基于编码器的多语言和区域transformer模型（mBERT、XLM-RoBERTa、DistilBERT、DeBERTaV3、BanglaBERT和IndicBERT）。我们的实验表明，当适应到孟加拉语转写的Chakma时，微调后的多语言模型优于其预训练版本，达到高达73.54%的标记准确度和低至2.90的困惑度。我们的分析进一步强调了数据质量对模型性能的影响，并展示了OCR管道在形态丰富的印度文字方面的局限性。我们的研究证明了孟加拉语转写的Chakma对于Chakma语言的迁移学习非常有效，我们发布了手动验证的单语数据集以鼓励对低资源语言的多语言语言建模进行进一步研究。


### 论文摘要

As an Indo-Aryan language with limited available data, Chakma remains largely underrepresented in language models. In this work, we introduce a novel corpus of contextually coherent Bangla-transliterated Chakma, curated from Chakma literature, and validated by native speakers. Using this dataset, we fine-tune six encoder-based multilingual and regional transformer models (mBERT, XLM-RoBERTa, DistilBERT, DeBERTaV3, BanglaBERT, and IndicBERT) on masked language modeling (MLM) tasks. Our experiments show that fine-tuned multilingual models outperform their pre-trained counterparts when adapted to Bangla-transliterated Chakma, achieving up to 73.54% token accuracy and a perplexity as low as 2.90. Our analysis further highlights the impact of data quality on model performance and shows the limitations of OCR pipelines for morphologically rich Indic scripts. Our research demonstrates that Bangla-transliterated Chakma can be very effective for transfer learning for Chakma language, and we release our manually validated monolingual dataset to encourage further research on multilingual language modeling for low-resource languages.

---

## 113. Denoised Diffusion for Object-Focused Image Augmentation

**论文链接:** [http://arxiv.org/abs/2510.08955v1](http://arxiv.org/abs/2510.08955v1)

**作者:** Nisha Pillai, Aditi Virupakshaiah, Harrison W. Smith, Amanda J. Ashworth, Prasanna Gowda, Phillip R. Owens, Adam R. Rivers, Bindu Nanduri, Mahalingam Ramkumar

**发布时间:** 2025-10-10

### GPT解析

### 总结

该研究提出了一种面向对象的数据增强框架，专门用于解决数据受限条件下动物健康监测的问题，通过从背景中分割动物并进行变换和基于扩散的合成，创建真实多样的场景，提高动物检测和监测性能。

### 背景

现代农业操作依赖集成监控系统进行农场优化，基于无人机的动物健康监测是关键组成部分，但面临数据有限的问题，包括动物小、被遮挡或部分可见等场景特定问题，且迁移学习方法因缺乏反映特定农场条件的大型数据集而效果有限。

### 目的

开发针对特定问题、以动物为中心的数据增强策略，应对数据有限条件下动物健康监测的挑战，弥合有限数据与实际应用之间的差距。

### 方法

提出一种面向对象的数据增强框架，专门为数据受限环境下的动物健康监测设计，通过从背景中分割动物，并利用变换和基于扩散的合成技术来增强动物，创建真实多样的场景。

### 主要发现

初步实验表明，与基线模型相比，增强数据集在动物检测任务上表现出更好的性能；通过生成领域特定数据，即使在数据稀缺的情况下，该方法也能支持实时动物健康监测解决方案。

### 结论

该数据增强方法成功解决了数据稀缺条件下动物健康监测的挑战，弥合了有限数据与实际应用之间的差距。

### 翻译

现代农业操作越来越多地依赖集成监控系统，结合多种数据源进行农场优化。基于空中无人机的动物健康监测是关键组成部分，但面临数据有限的问题，加之场景特定问题如动物小、被遮挡或部分可见。迁移学习方法通常无法解决这一限制，因为缺乏反映特定农场条件（包括动物品种、环境和行为变化）的大型数据集。因此，需要开发针对特定问题、以动物为中心的数据增强策略，应对这些独特挑战。为解决这一差距，我们提出了一种面向对象的数据增强框架，专门为数据受限条件下的动物健康监测设计。我们的方法从背景中分割动物，并通过变换和基于扩散的合成来增强它们，创建真实、多样的场景，提高动物检测和监测性能。初步实验表明，与基线模型相比，我们的增强数据集在动物检测任务上表现出更好的性能。通过生成领域特定数据，我们的方法即使在数据稀缺的情况下也能支持实时动物健康监测解决方案，弥合了有限数据与实际应用之间的差距。


### 论文摘要

Modern agricultural operations increasingly rely on integrated monitoring systems that combine multiple data sources for farm optimization. Aerial drone-based animal health monitoring serves as a key component but faces limited data availability, compounded by scene-specific issues such as small, occluded, or partially visible animals. Transfer learning approaches often fail to address this limitation due to the unavailability of large datasets that reflect specific farm conditions, including variations in animal breeds, environments, and behaviors. Therefore, there is a need for developing a problem-specific, animal-focused data augmentation strategy tailored to these unique challenges. To address this gap, we propose an object-focused data augmentation framework designed explicitly for animal health monitoring in constrained data settings. Our approach segments animals from backgrounds and augments them through transformations and diffusion-based synthesis to create realistic, diverse scenes that enhance animal detection and monitoring performance. Our initial experiments demonstrate that our augmented dataset yields superior performance compared to our baseline models on the animal detection task. By generating domain-specific data, our method empowers real-time animal health monitoring solutions even in data-scarce scenarios, bridging the gap between limited data and practical applicability.

---

## 114. Structured Output Regularization: a framework for few-shot transfer learning

**论文链接:** [http://arxiv.org/abs/2510.08728v1](http://arxiv.org/abs/2510.08728v1)

**作者:** Nicolas Ewen, Jairo Diaz-Rodriguez, Kelly Ramsay

**发布时间:** 2025-10-09

### GPT解析

### 总结

该研究提出了结构化输出正则化（SOR）框架，通过冻结网络结构并使用正则化方法，解决了传统迁移学习在适应领域特定特征和防止过拟合方面的局限性，在少样本医学影像分类任务中取得了与基准相当的结果。

### 背景

传统迁移学习方法通过冻结预训练网络的部分权重并添加任务特定层来重用大型预训练网络，这种方法计算效率高，但限制了模型适应领域特定特征的能力，并且在数据有限时仍可能导致过拟合。

### 目的

解决传统迁移学习方法在适应领域特定特征和防止过拟合方面的局限性。

### 方法

提出结构化输出正则化（SOR）框架，冻结内部网络结构（如卷积滤波器），同时使用组套索和L1惩罚的组合，使模型能够以最少的额外参数适应特定数据，并可以轻松应用于各种网络组件。

### 主要发现

在三个少样本医学影像分类任务上评估了SOR，使用DenseNet121和EfficientNetB4作为基础模型，与已建立的基准相比取得了具有竞争力的结果。

### 结论

SOR框架是一种简单有效的方法，可以解决传统迁移学习在适应领域特定特征和防止过拟合方面的局限性，并且具有广泛的适用性。

### 翻译

传统的迁移学习通常通过冻结大型预训练网络的一些权重并添加任务特定层来重用这些网络。虽然这种方法计算效率高，但它限制了模型适应领域特定特征的能力，并且在数据非常有限的情况下仍可能导致过拟合。为了解决这些局限性，我们提出了结构化输出正则化（SOR），这是一个简单而有效的框架，它冻结内部网络结构（例如卷积滤波器），同时使用组套索和L1惩罚的组合。该框架使模型能够以最少的额外参数适应特定数据，并且可以轻松应用于各种网络组件，例如神经网络中的卷积滤波器或各种块，从而为迁移学习任务提供了广泛的适用性。我们在三个少样本医学影像分类任务上评估了SOR，使用DenseNet121和EfficientNetB4作为基础模型，与已建立的基准相比取得了具有竞争力的结果。


### 论文摘要

Traditional transfer learning typically reuses large pre-trained networks by freezing some of their weights and adding task-specific layers. While this approach is computationally efficient, it limits the model's ability to adapt to domain-specific features and can still lead to overfitting with very limited data. To address these limitations, we propose Structured Output Regularization (SOR), a simple yet effective framework that freezes the internal network structures (e.g., convolutional filters) while using a combination of group lasso and $L_1$ penalties. This framework tailors the model to specific data with minimal additional parameters and is easily applicable to various network components, such as convolutional filters or various blocks in neural networks enabling broad applicability for transfer learning tasks. We evaluate SOR on three few shot medical imaging classification tasks and we achieve competitive results using DenseNet121, and EfficientNetB4 bases compared to established benchmarks.

---

## 115. Deploying Tiny LVLM Judges for Real-World Evaluation of Chart Models: Lessons Learned and Best Practices

**论文链接:** [http://arxiv.org/abs/2510.07545v2](http://arxiv.org/abs/2510.07545v2)

**作者:** Md Tahmid Rahman Laskar, Mohammed Saidul Islam, Ridwan Mahbub, Mizanur Rahman, Amran Bhuiyan, Israt Jahan, Mir Tafseer Nayeem, Shafiq Joty, Enamul Hoque, Jimmy Huang

**发布时间:** 2025-10-08

**备注:** Accepted to the EMNLP 2025 Industry Track

### GPT解析

### 总结

本研究探讨了大型视觉-语言模型(LVLMs)作为图表理解任务中自动化评判者的能力，提出多标准提示和领域自适应迁移学习两种方法，成功使小型模型(2B参数)成为有效的图表评判者(ChartJudge)，实现了资源受限环境下的低成本评估。

### 背景

具有70亿参数的大型视觉-语言模型(LVLMs)已显示出作为图表理解任务中自动化评判者的潜力。然而，小型模型(参数量≤2B)作为评判者时表现仍然不佳，限制了它们在资源受限环境中的实际应用。

### 目的

解决小型模型在图表理解任务中作为评判者表现不佳的问题，确保评估过程具有成本效益，使小型模型能够在资源受限环境中有效应用。

### 方法

作者提出两种方法：(i)多标准提示：将单独的评估标准组合到一个查询中；(ii)领域自适应迁移学习：在图表数据集上的合成判断上微调一个具有20亿参数的LVLM，创建出ChartJudge模型。

### 主要发现

多标准提示暴露了模型的鲁棒性差距，导致70亿参数模型(包括专业的LVLM评判者如LLaVA-Critic)性能大幅下降；小型LVLM(ChartJudge)可以有效将知识从一个数据集迁移到另一个数据集，使其成为更专业的模型；对不同图表类型和查询复杂性的细粒度分析提供了关于模型大小、提示设计和可迁移性之间权衡的可操作见解。

### 结论

通过结合多标准提示和领域自适应迁移学习，ChartJudge模型能够在资源受限环境中实现可扩展的低成本评估，为图表推理任务提供有效的解决方案。

### 翻译

仅具有70亿参数的大型视觉-语言模型(LVLMs)已显示出作为图表理解任务中自动化评判者的潜力。然而，小型模型(参数量≤2B)作为评判者时仍然表现不佳，限制了它们在资源受限环境中的实际应用。为此，我们提出两种方法来确保成本高效的评估：(i)多标准提示，将单独的评估标准组合到一个查询中；(ii)领域自适应迁移学习，我们在图表数据集上的合成判断上微调一个20亿参数的LVLM，创建出ChartJudge。实验表明，多标准提示暴露了模型的鲁棒性差距，导致70亿参数模型(包括专业的LVLM评判者如LLaVA-Critic)性能大幅下降。此外，我们发现我们的小型LVLM(ChartJudge)可以有效地将知识从一个数据集迁移到另一个数据集，使其成为更专业的模型。我们对不同图表类型和查询复杂性的细粒度分析提供了关于模型大小、提示设计和可迁移性之间权衡的可操作见解，使图表推理任务能够实现可扩展的低成本评估。


### 论文摘要

Large Vision-Language Models (LVLMs) with only 7B parameters have shown promise as automated judges in chart comprehension tasks. However, tiny models (<=2B parameters) still perform poorly as judges, limiting their real-world use in resource-constrained settings. To address this, we propose two approaches to ensure cost-efficient evaluation: (i) multi-criteria prompting, which combines separate evaluation criteria into a single query, and (ii) domain-adaptive transfer learning, in which we fine-tune a 2B-parameter LVLM on synthetic judgments in a chart dataset to create the ChartJudge. Experiments show that multi-criteria prompting exposes robustness gaps, which led to a huge drop in performance for 7B models, including specialized LVLM judges like LLaVA-Critic. In addition, we find that our tiny LVLM (ChartJudge) can effectively transfer knowledge from one dataset to another to make it a more specialized model. Our fine-grained analysis across chart types and query complexities offers actionable insights into trade-offs between model size, prompt design, and transferability, enabling scalable, low-cost evaluation for chart reasoning tasks.

---

## 116. Bridging Perspectives: Foundation Model Guided BEV Maps for 3D Object Detection and Tracking

**论文链接:** [http://arxiv.org/abs/2510.10287v1](http://arxiv.org/abs/2510.10287v1)

**作者:** Markus Käppeler, Özgün Çiçek, Daniele Cattaneo, Claudius Gläser, Yakov Miron, Abhinav Valada

**发布时间:** 2025-10-11

### GPT解析

### 总结

论文提出了DualViewDistill混合检测和跟踪框架，结合透视视图和鸟瞰图特征，提高自动驾驶中的3D物体检测和跟踪性能。

### 背景

基于相机的3D物体检测和跟踪对自动驾驶感知至关重要，但当前最先进方法通常仅依赖透视视图或鸟瞰图特征，限制了利用精细物体细节和空间结构化场景表示的能力。

### 目的

开发能够同时利用透视视图和鸟瞰图特征优势的混合检测和跟踪框架，提升3D物体检测和跟踪性能。

### 方法

提出DualViewDistill框架，引入由基础模型指导的BEV地图，利用DINOv2特征通过新颖蒸馏过程生成BEV表示，并将PV特征与增强的BEV地图集成，通过可变形聚合增强3D物体检测和跟踪。

### 主要发现

在nuScenes和Argoverse 2基准测试上，DualViewDistill达到最先进性能，展示了基础模型BEV地图在实现更可靠自动驾驶感知方面的潜力。

### 结论

结合透视视图和鸟瞰图特征的优势，并利用基础模型提供的丰富语义和几何信息，显著提高了3D物体检测和跟踪性能，为自动驾驶感知提供了更可靠解决方案。

### 翻译

基于相机的3D物体检测和跟踪对于自动驾驶感知至关重要。当前最先进的方法通常仅依赖于透视视图或鸟瞰图特征，限制了它们利用精细物体细节和空间结构化场景表示的能力。在这项工作中，我们提出了DualViewDistill，一个结合了PV和BEV相机图像特征的混合检测和跟踪框架，以利用它们的互补优势。我们的方法引入了由基础模型指导的BEV地图，利用描述性的DINOv2特征，并通过一种新颖的蒸馏过程将其蒸馏成BEV表示。通过将PV特征与由DINOv2的语义和几何特征增强的BEV地图集成，我们的模型通过可变形聚合利用这种混合表示来增强3D物体检测和跟踪。在nuScenes和Argoverse 2基准测试上的大量实验表明，DualViewDistill达到了最先进的性能。结果展示了基础模型BEV地图在实现更可靠自动驾驶感知方面的潜力。我们在https://dualviewdistill.cs.uni-freiburg.de提供了代码和预训练模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决基于相机的3D目标检测和跟踪方法中只依赖单一视图特征(透视视图PV或鸟瞰视图BEV)的问题，无法同时利用细粒度物体细节和空间结构化场景表示。这个问题在自动驾驶领域非常重要，因为可靠的3D感知系统需要既能识别物体细节，又能理解空间布局，这对安全驾驶至关重要。单一视图特征的限制导致现有方法难以同时满足这两方面的需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了当前方法的局限性，发现BEV-based方法提供结构化空间表示但缺乏物体细节，而PV-based方法有丰富细节但空间推理能力有限。他们借鉴了BEVFormer、LSS等BEV方法的表示思想，以及Sparse4Dv3等查询设计方法，还参考了DINOv2等基础模型的一般特征能力和HDNet等地图感知方法的空间先验思想。在此基础上，作者创新性地设计了双视图融合框架和基础模型引导的BEV地图蒸馏方法，同时利用两种视图的优势。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是双视图融合与基础模型引导的特征蒸馏。方法同时利用透视视图(PV)的物体细节优势和鸟瞰视图(BEV)的空间结构优势，并通过DINOv2基础模型的特征蒸馏增强BEV表示。整体流程包括：1)输入多视角RGB图像；2)提取PV特征和DINOv2特征；3)使用LSS机制将PV特征提升到BEV空间；4)通过Transformer处理对象查询，结合可变形聚合与PV和BEV特征交互；5)将DINOv2特征投影到点云生成BEV伪标签并蒸馏；6)输出3D边界框和对象轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)双视图检测框架，首次统一PV和BEV特征；2)基础模型引导的BEV地图，利用DINOv2特征蒸馏提供丰富的在线神经地图；3)可变形聚合机制有效融合两种视图特征；4)混合监督策略结合检测/跟踪监督和BEV特征蒸馏。相比之前工作，传统方法只使用单一视图特征或依赖手动标注的高清地图，而本文同时使用两种视图并通过基础模型在线生成神经地图，无需人工标注，在nuScenes和Argoverse 2上实现了最先进的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了DualViewDistill，一种结合透视视图和鸟瞰视图特征并利用DINOv2引导的BEV地图进行3D目标检测和跟踪的混合框架，显著提升了自动驾驶系统的感知性能。'}


### 论文摘要

Camera-based 3D object detection and tracking are essential for perception in autonomous driving. Current state-of-the-art approaches often rely exclusively on either perspective-view (PV) or bird's-eye-view (BEV) features, limiting their ability to leverage both fine-grained object details and spatially structured scene representations. In this work, we propose DualViewDistill, a hybrid detection and tracking framework that incorporates both PV and BEV camera image features to leverage their complementary strengths. Our approach introduces BEV maps guided by foundation models, leveraging descriptive DINOv2 features that are distilled into BEV representations through a novel distillation process. By integrating PV features with BEV maps enriched with semantic and geometric features from DINOv2, our model leverages this hybrid representation via deformable aggregation to enhance 3D object detection and tracking. Extensive experiments on the nuScenes and Argoverse 2 benchmarks demonstrate that DualViewDistill achieves state-of-the-art performance. The results showcase the potential of foundation model BEV maps to enable more reliable perception for autonomous driving. We make the code and pre-trained models available at https://dualviewdistill.cs.uni-freiburg.de .

---

## 117. Through the Perspective of LiDAR: A Feature-Enriched and Uncertainty-Aware Annotation Pipeline for Terrestrial Point Cloud Segmentation

**论文链接:** [http://arxiv.org/abs/2510.06582v2](http://arxiv.org/abs/2510.06582v2)

**作者:** Fei Zhang, Rob Chancia, Josie Clapp, Amirhossein Hassanzadeh, Dimah Dera, Richard MacKenzie, Jan van Aardt

**发布时间:** 2025-10-08

**备注:** 40 pages (28 main text), 20 figures, 4 supplementary materials; links  to 3D point animations are included in the last table

### GPT解析

### 总结

本文提出了一种半自动的、不确定性感知的地面激光扫描点云语义分割管道，通过球面投影、特征丰富、集成学习和目标标注相结合，减少标注工作量的同时保持高准确性。构建了Mangrove3D数据集并评估了数据效率和特征重要性。

### 背景

准确的地面激光扫描点云语义分割受限于昂贵的手动标注，需要一种方法来减少标注工作量同时保持高准确性。

### 目的

开发一种半自动的、不确定性感知的管道，构建红树林森林语义分割TLS数据集(Mangrove3D)，并评估数据效率和特征重要性，回答需要多少标注数据以及哪些特征最重要的问题。

### 方法

将3D点投影到2D球面网格，使用多源特征丰富像素，训练分割网络集成生成伪标签和不确定性地图，不确定性地图指导模糊区域标注，将2D输出投影回3D，开发三层可视化套件，构建Mangrove3D数据集，评估数据效率和特征重要性，通过跨数据集测试验证特征丰富化策略的泛化能力。

### 主要发现

性能在约12个标注扫描后趋于饱和，几何特征贡献最大，紧凑的九通道堆叠捕获了几乎所有的判别能力，平均交并比(mIoU)稳定在约0.76。

### 结论

提出了稳健的、不确定性感知的TLS标注管道和可视化工具，构建了Mangrove3D数据集，提供了关于数据效率和特征重要性的经验指导，使得生态监测等领域的可扩展、高质量的TLS点云分割成为可能。

### 翻译

准确的地面激光扫描点云语义分割受限于昂贵的手动标注。我们提出了一种半自动的、不确定性感知的管道，结合球面投影、特征丰富、集成学习和目标标注，以减少标注工作量，同时保持高准确性。我们的方法将3D点投影到2D球面网格，用多源特征丰富像素，并训练分割网络集成来生成伪标签和不确定性地图，后者指导模糊区域的标注。将2D输出投影回3D，得到密集标注的点云，并配有三层可视化套件(2D特征图、3D着色点云和紧凑虚拟球体)用于快速分类和审阅者指导。使用此管道，我们构建了Mangrove3D，一个用于红树林森林的语义分割TLS数据集。我们进一步评估数据效率和特征重要性，以解决两个关键问题：(1)需要多少标注数据，(2)哪些特征最重要。结果表明，性能在约12个标注扫描后趋于饱和，几何特征贡献最大，紧凑的九通道堆叠捕获了几乎所有的判别能力，平均交并比(mIoU)稳定在约0.76。最后，我们通过在ForestSemantic和Semantic3D上的跨数据集测试，验证了我们的特征丰富化策略的泛化能力。我们的贡献包括：(i)带有可视化工具的稳健的、不确定性感知的TLS标注管道；(ii)Mangrove3D数据集；以及(iii)关于数据效率和特征重要性的经验指导，从而使得生态监测等领域的可扩展、高质量的TLS点云分割成为可能。数据集和处理脚本可在https://fz-rit.github.io/through-the-lidars-eye/公开获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决地面激光扫描(TLS)点云语义分割中高质量标注数据集稀缺的问题。这个问题很重要，因为手动标注全分辨率TLS扫描非常耗费人力，特别是在生态场景中，由于严重的遮挡、不规则的几何形状和交织的树结构，这一问题尤为突出。这阻碍了自动分析在林业和环境监测中的采用，尽管这些领域有迫切需求，而准确的语义分割是生态研究中分析树木指标、生物量和栖息地特征的基础。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考如何降低标注复杂度并提高效率，考虑将不规则的3D数据转换为结构化的2D地图，集成特征增强的分割方法，并纳入不确定性分析来指导人工校正。作者借鉴了现有工作：球面投影概念来自地图学，应用于将3D点云转换为2D图像；主动学习范式用于迭代查询标注员获取信息量最大的样本；自训练策略让模型重用高置信度预测；特征融合方法利用几何结构、辐射度响应和位置上下文增强分割。作者的创新在于将这些方法整合并专门针对TLS点云和生态场景进行了优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过将3D点云投影到2D球面网格上，在2D空间中进行高效的标注和分割，然后再将结果投影回3D，从而降低标注复杂度并提高效率。整体流程分为三阶段：1)球面投影与可视化：将3D点云转换为2D球面网格，创建多通道特征图和三种可视化(2D特征图、3D彩色点云、虚拟球体)；2)混合标注：使用少量初始标注训练三个分割模型集合，生成伪标签和不确定性图，高不确定性区域人工精修，高置信度预测保留为伪标签；3)反向投影与精修：将2D分割结果投影回3D，应用几何平滑和特征驱动的修复，生成最终标注结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)不确定性感知的TLS标注流程，集成球面投影、特征增强和集成学习；2)三级可视化套件，便于标注和检查；3)系统化的特征增强策略，识别最优特征组合；4)创建首个针对复杂红树林生态系统的TLS数据集Mangrove3D。相比之前工作，不同之处在于：现有3D标注工具主要为简单场景设计；早期球面投影方法使用传统聚类而非深度学习；现有主动学习方法主要在RGB数据集上应用；PointPainting等方法依赖多传感器数据，而本文方法仅使用LiDAR数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合球面投影、特征增强和不确定性分析的半自动标注流程，显著降低了地面激光扫描点云语义分割的标注成本，同时创建了首个针对复杂红树林生态系统的TLS数据集，为生态监测等领域提供了高效、高质量的分析工具。'}


### 论文摘要

Accurate semantic segmentation of terrestrial laser scanning (TLS) point clouds is limited by costly manual annotation. We propose a semi-automated, uncertainty-aware pipeline that integrates spherical projection, feature enrichment, ensemble learning, and targeted annotation to reduce labeling effort, while sustaining high accuracy. Our approach projects 3D points to a 2D spherical grid, enriches pixels with multi-source features, and trains an ensemble of segmentation networks to produce pseudo-labels and uncertainty maps, the latter guiding annotation of ambiguous regions. The 2D outputs are back-projected to 3D, yielding densely annotated point clouds supported by a three-tier visualization suite (2D feature maps, 3D colorized point clouds, and compact virtual spheres) for rapid triage and reviewer guidance. Using this pipeline, we build Mangrove3D, a semantic segmentation TLS dataset for mangrove forests. We further evaluate data efficiency and feature importance to address two key questions: (1) how much annotated data are needed and (2) which features matter most. Results show that performance saturates after ~12 annotated scans, geometric features contribute the most, and compact nine-channel stacks capture nearly all discriminative power, with the mean Intersection over Union (mIoU) plateauing at around 0.76. Finally, we confirm the generalization of our feature-enrichment strategy through cross-dataset tests on ForestSemantic and Semantic3D.   Our contributions include: (i) a robust, uncertainty-aware TLS annotation pipeline with visualization tools; (ii) the Mangrove3D dataset; and (iii) empirical guidance on data efficiency and feature importance, thus enabling scalable, high-quality segmentation of TLS point clouds for ecological monitoring and beyond. The dataset and processing scripts are publicly available at https://fz-rit.github.io/through-the-lidars-eye/.

---

## 118. RangeSAM: Leveraging Visual Foundation Models for Range-View repesented LiDAR segmentation

**论文链接:** [http://arxiv.org/abs/2509.15886v2](http://arxiv.org/abs/2509.15886v2)

**作者:** Paul Julius Kühn, Duc Anh Nguyen, Arjan Kuijper, Holger Graf, Dieter Fellner, Saptarshi Neil Sinha

**发布时间:** 2025-09-19

### GPT解析

### 总结

该研究探讨了将视觉基础模型SAM2应用于LiDAR点云分割的range-view方法，通过结合2D特征提取与投影/反投影操作，实现了在保持2D方法效率的同时获得有竞争力的3D分割性能。

### 背景

点云分割是自动驾驶和3D场景理解的核心技术。目前基于体素和点的方法虽能捕获细粒度几何信息，但计算成本高，内存访问不规则，实时效率有限。相比之下，range-view方法相对未被充分探索，但可利用成熟的2D语义分割技术进行快速准确预测。

### 目的

研究当前最先进的视觉基础模型SAM2是否可作为LiDAR点云在range view中的强大backbone，探索VFMs作为3D感知通用backbone的可行性。

### 方法

提出了首个适应SAM2进行3D分割的range-view框架，结合高效2D特征提取与标准投影/反投影操作处理点云。对编码器进行了三种架构修改：(1)强调LiDAR范围图像中水平空间依赖性的新模块；(2)针对球面投影几何特性的定制配置；(3)专门设计用于捕获range-view伪图像中独特空间模式和间断性的适应机制。

### 主要发现

该方法在SemanticKITTI数据集上实现了具有竞争力的性能，同时受益于2D为中心的管道的速度、可扩展性和部署简单性。

### 结论

VFMs作为3D感知通用backbone具有可行性，为统一的基础模型驱动的LiDAR分割开辟了道路，使用VFMs的range-view分割方法取得了有希望的结果。

### 翻译

点云分割是自动驾驶和3D场景理解的核心。虽然最近的体素和基于点的方法因其与深度架构的兼容性和捕获细粒度几何的能力而主导研究，但它们通常带来高计算成本、不规则的内存访问和有限的实时效率。相比之下，range-view方法虽然相对未被充分探索，但可以利用成熟的2D语义分割技术进行快速准确的预测。受视觉基础模型(VFMs)在描述、零样本识别和多模态任务中快速进展的启发，我们研究了当前最先进的分割VFM SAM2是否可以作为LiDAR点云在range view中的强大backbone。据我们所知，我们提出了第一个适应SAM2进行3D分割的range-view框架，将高效的2D特征提取与标准投影/反投影相结合以处理点云。为了优化SAM2对range-view表示的处理，我们对编码器实现了几种架构修改：(1)一个强调LiDAR范围图像中固有水平空间依赖性的新模块；(2)针对球面投影几何特性的定制配置；(3)编码器主干中的一种适应机制，专门设计用于捕获range-view伪图像中存在的独特空间模式和间断性。我们的方法在SemanticKITTI上实现了具有竞争力的性能，同时受益于2D为中心的管道的速度、可扩展性和部署简单性。这项工作证明了VFMs作为3D感知通用backbone的可行性，并为统一的基础模型驱动的LiDAR分割开辟了道路。结果让我们得出结论，使用VFMs的range-view分割方法取得了有希望的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云分割的效率和准确性问题。现有的基于体素和点的方法计算成本高、内存访问不规则、运行效率有限。这个问题在自动驾驶和3D场景理解中至关重要，因为准确分割点云可以帮助车辆识别道路、车辆、行人等关键元素，确保安全导航。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，注意到范围视图方法可以利用成熟的2D语义分割技术。受SAM2等视觉基础模型在图像分割中成功的启发，作者将其应用于3D点云分割。为了适应范围视图表示，作者对编码器进行了关键修改：设计了新的Stem模块、自定义了Hiera块配置、调整了窗口注意力机制。该方法借鉴了SAM2模型、Receptive Field Blocks、k-NN插值和多种损失函数等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将SAM2视觉基础模型应用于3D点云分割，通过将点云转换为范围视图表示，利用2D分割模型的能力，再投影回3D空间。流程包括：1)范围投影预处理，将3D点云转换为64×2048像素的范围图像；2)模型架构，包含Stem模块、基于Hiera的编码器和Receptive Field Block解码器；3)后处理，使用k-NN插值将标签传播回3D点云；4)训练，使用复合损失函数和数据增强技术。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个将SAM2适应3D分割的范围视图框架；2)针对范围视图的编码器修改(Stem模块、自定义Hiera块、调整的注意力机制)；3)多组件编码器架构结合感受野块；4)有效的k-NN插值后处理技术；5)复合损失函数。相比之前的工作，RangeSAM利用了最先进的SAM2模型，设计了非对称注意力窗口强调水平空间关系，发现多数据集训练比2D预训练更有效，实现了与现有方法相竞争的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RangeSAM创新性地将SAM2视觉基础模型适应到范围视图表示的LiDAR点云分割中，通过专门设计的架构修改和训练策略，实现了与现有方法相竞争的性能，同时利用了2D分割模型的效率和可扩展性。'}


### 论文摘要

Point cloud segmentation is central to autonomous driving and 3D scene understanding. While voxel- and point-based methods dominate recent research due to their compatibility with deep architectures and ability to capture fine-grained geometry, they often incur high computational cost, irregular memory access, and limited real-time efficiency. In contrast, range-view methods, though relatively underexplored - can leverage mature 2D semantic segmentation techniques for fast and accurate predictions. Motivated by the rapid progress in Visual Foundation Models (VFMs) for captioning, zero-shot recognition, and multimodal tasks, we investigate whether SAM2, the current state-of-the-art VFM for segmentation tasks, can serve as a strong backbone for LiDAR point cloud segmentation in the range view. We present , to our knowledge, the first range-view framework that adapts SAM2 to 3D segmentation, coupling efficient 2D feature extraction with standard projection/back-projection to operate on point clouds. To optimize SAM2 for range-view representations, we implement several architectural modifications to the encoder: (1) a novel module that emphasizes horizontal spatial dependencies inherent in LiDAR range images, (2) a customized configuration of tailored to the geometric properties of spherical projections, and (3) an adapted mechanism in the encoder backbone specifically designed to capture the unique spatial patterns and discontinuities present in range-view pseudo-images. Our approach achieves competitive performance on SemanticKITTI while benefiting from the speed, scalability, and deployment simplicity of 2D-centric pipelines. This work highlights the viability of VFMs as general-purpose backbones for 3D perception and opens a path toward unified, foundation-model-driven LiDAR segmentation. Results lets us conclude that range-view segmentation methods using VFMs leads to promising results.

---

## 119. PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Image

**论文链接:** [http://arxiv.org/abs/2510.11649v1](http://arxiv.org/abs/2510.11649v1)

**作者:** Pradyumna Yalandur Muralidhar, Yuxuan Xue, Xianghui Xie, Margaret Kostyrko, Gerard Pons-Moll

**发布时间:** 2025-10-13

**DOI:** 10.1145/3757377.3763862

**备注:** Accepted to ACM SIGGraphAsia 2025. Project website:  https://yuxuan-xue.com/physic

### GPT解析

### 总结

PhySIC是一个用于物理合理的人体-场景交互和接触重建的框架，可以从单张RGB图像中恢复度量一致的3D人体和场景，处理遮挡和深度模糊等问题，实现高效且高质量的重建。

### 背景

从单张图像重建具有度量准确性的三维人体及其周围场景对于虚拟现实、机器人和全面的3D场景理解至关重要。然而，现有方法面临深度模糊、遮挡和物理不一致接触等挑战。

### 目的

解决现有方法面临的深度模糊、遮挡和物理不一致接触等问题，实现从单张图像中重建物理合理的人体-场景交互。

### 方法

PhySIC从单张RGB图像恢复度量一致的SMPL-X人体网格、密集场景表面和顶点级接触图；执行遮挡感知的图像修复；融合可见深度与未缩放几何形状；合成缺失支撑表面；通过加权置信度优化优化人体姿态、相机参数和全局尺度；使用显式遮挡掩码保护不可见区域；高效处理多个人体交互。

### 主要发现

PhySIC在单图像基线上表现更优，将场景平均每顶点误差从641毫米降低到227毫米，将PA-MPJPE减半至42毫米，并将接触F1从0.09提高到0.51；定性结果显示了真实的脚-地面交互、自然的坐姿以及对严重遮挡家具的合理重建。

### 结论

通过将单张图像转换为物理合理的3D人体-场景对，PhySIC推动了可扩展的3D场景理解，实现仅需9秒的联合人体-场景优化和不到27秒的端到端处理。

### 翻译

从单张图像重建具有度量准确性的人体及其周围场景对于虚拟现实、机器人和全面的3D场景理解至关重要。然而，现有方法难以处理深度模糊、遮挡和物理不一致接触等问题。为应对这些挑战，我们引入了PhySIC，一个用于物理合理的人体-场景交互和接触重建的框架。PhySIC从单张RGB图像中恢复度量一致的SMPL-X人体网格、密集场景表面和顶点级接触图，并在共享坐标系中表示。从粗略的单目深度和人体估计开始，PhySIC执行遮挡感知的图像修复，将可见深度与未缩放的几何形状融合以获得稳健的度量支架，并合成缺失的支撑表面如地板。通过加权置信度优化，通过联合强制深度对齐、接触先验、避免穿透和2D重投影一致性来优化人体姿态、相机参数和全局尺度。显式遮挡掩码保护不可见区域避免不合理的配置。PhySIC效率高，仅需9秒进行联合人体-场景优化，端到端时间不到27秒。自然处理多个人体，能够重建多样化的交互。经验表明，PhySIC优于单图像基线方法，将场景平均每顶点误差从641毫米降低到227毫米，将PA-MPJPE减半至42毫米，并将接触F1从0.09提高到0.51。定性结果显示了真实的脚-地面交互、自然的坐姿以及对严重遮挡家具的合理重建。通过将单张图像转换为物理合理的3D人体-场景对，PhySIC推动了可扩展的3D场景理解。我们的实现已在https://yuxuan-xue.com/physic公开提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从单张RGB图像中重建度量准确的3D人体和场景几何，以及它们之间的物理交互关系的问题。这个问题在现实和研究中非常重要，因为对于虚拟现实、机器人和全面的3D场景理解至关重要，但现有方法难以处理深度歧义、遮挡和物理上不一致的接触等挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到人体和场景是相互约束的：场景在物理上限制了可能的人体姿势，而人体姿势为估计场景几何和规模提供了重要线索。基于这一观察，作者借鉴了多种现有技术，包括使用SAM2进行人体分割、OmniEraser进行图像修复、DepthPro预测度量深度、MoGe获取详细几何、SMPL-X表示人体等。作者设计了一个三阶段方法：首先估计度量规模场景，然后重建人体并与场景对齐，最后通过联合优化确保物理合理性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是联合优化人体姿势、场景几何和全局尺度，产生物理上合理的人体-场景对，同时利用人体和场景之间的物理约束关系提高重建质量。整体流程分为三阶段：1)度量规模场景估计：通过图像修复、结合度量深度和详细几何、地面平面拟合和组合场景点来获取完整场景；2)人体重建与场景对齐：将人体点与场景点对齐，使用SMPL-X模型表示人体，并优化网格与场景对齐；3)联合人体-场景优化：通过接触损失、遮挡感知的穿透损失和正则化项确保物理合理性，并能处理多个人体情况。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个能处理多个人体、多样场景和交互类型的度量规模人体-场景重建方法；2)引入了强大的初始化策略和遮挡感知的联合优化；3)高效的重建管道，27秒内完成端到端重建；4)能处理复杂交互如坐姿和脚-地面接触。相比之前的工作，PhySIC不需要视频输入或多视图图像(如HSR和HSfM)，也不需要预定义的场景扫描(如PROX)，能处理室内外多种场景，而不仅限于特定室内环境(如HolisticMesh)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PhySIC通过联合优化人体姿势、场景几何和接触信息，实现了从单张RGB图像中快速重建物理上合理的3D人体-场景交互，显著提高了重建的准确性和效率，为虚拟现实、机器人和3D场景理解提供了新的解决方案。'}


### 论文摘要

Reconstructing metrically accurate humans and their surrounding scenes from a single image is crucial for virtual reality, robotics, and comprehensive 3D scene understanding. However, existing methods struggle with depth ambiguity, occlusions, and physically inconsistent contacts. To address these challenges, we introduce PhySIC, a framework for physically plausible Human-Scene Interaction and Contact reconstruction. PhySIC recovers metrically consistent SMPL-X human meshes, dense scene surfaces, and vertex-level contact maps within a shared coordinate frame from a single RGB image. Starting from coarse monocular depth and body estimates, PhySIC performs occlusion-aware inpainting, fuses visible depth with unscaled geometry for a robust metric scaffold, and synthesizes missing support surfaces like floors. A confidence-weighted optimization refines body pose, camera parameters, and global scale by jointly enforcing depth alignment, contact priors, interpenetration avoidance, and 2D reprojection consistency. Explicit occlusion masking safeguards invisible regions against implausible configurations. PhySIC is efficient, requiring only 9 seconds for joint human-scene optimization and under 27 seconds end-to-end. It naturally handles multiple humans, enabling reconstruction of diverse interactions. Empirically, PhySIC outperforms single-image baselines, reducing mean per-vertex scene error from 641 mm to 227 mm, halving PA-MPJPE to 42 mm, and improving contact F1 from 0.09 to 0.51. Qualitative results show realistic foot-floor interactions, natural seating, and plausible reconstructions of heavily occluded furniture. By converting a single image into a physically plausible 3D human-scene pair, PhySIC advances scalable 3D scene understanding. Our implementation is publicly available at https://yuxuan-xue.com/physic.

---

## 120. A Framework for Low-Effort Training Data Generation for Urban Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2510.11567v1](http://arxiv.org/abs/2510.11567v1)

**作者:** Denis Zavadski, Damjan Kalšan, Tim Küchler, Haebom Lee, Stefan Roth, Carsten Rother

**发布时间:** 2025-10-13

### GPT解析

### 总结

研究提出了一种新框架，利用扩散模型和不完美伪标签将合成数据适应到目标域，生成高保真图像，解决了合成数据与真实数据之间的差距问题，实验证明其有效性。

### 背景

合成数据集被广泛用于训练城市场景识别模型，但即使高度逼真的渲染图像与真实图像之间仍然存在明显差距。当适应特定目标领域（如Cityscapes）时，这种差距尤为明显，因为建筑、植被、物体外观和相机特性的差异限制了下游性能。

### 目的

解决合成数据与真实数据之间的差距问题，避免使用昂贵的3D建模来缩小这一差距，因为这会违背低成本标记数据的目的。

### 方法

提出一种新框架，使用不完美的伪标签将现成的扩散模型适应到目标域。训练后的模型可以从任何合成数据集的语义图中生成高保真、目标对齐的图像。该方法过滤次优生成结果，修正图像-标签不对齐问题，并标准化不同数据集的语义，将弱合成数据转换为具有竞争力的真实域训练集。

### 主要发现

在五个合成数据集和两个真实目标数据集上的实验显示，与最先进的翻译方法相比，分割性能提升了高达+8.0% mIoU，使快速构建的合成数据集变得与需要大量手动设计的高投入、时间密集型合成数据集一样有效。

### 结论

这项工作展示了一个有价值的协作范式，其中快速语义原型与生成模型相结合，为城市场景理解实现了可扩展的高质量训练数据创建。

### 翻译

合成数据集被广泛用于训练城市场景识别模型，但即使高度逼真的渲染图像与真实图像之间仍然存在明显差距。当适应特定目标领域（如Cityscapes）时，这种差距尤为明显，因为建筑、植被、物体外观和相机特性的差异限制了下游性能。使用更详细的3D建模来缩小这一差距需要昂贵的资产和场景设计，违背了低成本标记数据的目的。为此，我们提出了一种新框架，仅使用不完美的伪标签将现成的扩散模型适应到目标域。一旦训练完成，它可以从任何合成数据集的语义图中生成高保真、目标对齐的图像，包括只需几小时而非数月创建的低投入来源。该方法过滤次优生成结果，修正图像-标签不对齐问题，并标准化不同数据集的语义，将弱合成数据转换为具有竞争力的真实域训练集。在五个合成数据集和两个真实目标数据集上的实验显示，与最先进的翻译方法相比，分割性能提升了高达+8.0% mIoU，使快速构建的合成数据集变得与需要大量手动设计的高投入、时间密集型合成数据集一样有效。这项工作展示了一个有价值的协作范式，其中快速语义原型与生成模型相结合，为城市场景理解实现了可扩展的高质量训练数据创建。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何高效生成高质量城市场景语义分割训练数据的问题，特别是如何将低成本的合成数据转换为真实世界场景的有效训练数据。这个问题重要是因为真实世界标注数据收集成本高、耗时长；即使最逼真的合成数据与真实图像间也存在明显差距，限制了下游模型性能；而传统3D建模方法需要昂贵资产和场景设计，违背了低成本标记数据的目的。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考过程是：认识到合成数据与真实数据差距问题→探索扩散模型替代昂贵光照建模→提出3D建模与生成建模协作范式→设计两阶段策略解决视觉风格学习和语义对齐两个竞争任务。借鉴了现有工作包括：使用预训练扩散模型(Stable Diffusion 2.1)作为基础；采用伪标签提供语义条件；使用分类器自由引导提高样本质量；结合深度图作为正则化输入；利用连通组件分析评估生成质量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过微调现成扩散模型适应特定目标域，使用不完美伪标签生成高保真目标对齐图像，采用对象中心过滤策略优化结果，并标准化不同数据集间的语义。整体流程：1)两阶段微调-第一阶段学习目标域视觉风格，第二阶段实现语义布局对齐；2)正则化技术-使用粗略伪标签、伪深度图和零条件引导提高鲁棒性；3)自动数据生成-为单个语义图生成多样样本，用MCOC评分筛选高质量样本。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)源无关框架，能从任何合成数据集生成目标域图像；2)两阶段训练策略，分别学习视觉风格和语义控制；3)多种正则化技术组合提高鲁棒性；4)对象中心样本选择机制提高数据质量。相比不同：1)与传统I2I方法相比，无需成对数据，能生成多样化样本，对源质量要求低，图像质量更高；2)与UDA方法相比，明确生成目标域图像，提供透明度，解耦数据生成与下游模型；3)与其他扩散模型方法相比，针对特定目标域微调，使用伪标签而非真实标签，结合多种正则化技术。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于扩散模型的框架，通过利用不完美伪标签和两阶段训练策略，将低努力合成的语义布局高效转换为高质量、目标域对齐的训练图像，显著提升了城市场景语义分割性能，并展示了快速语义场景原型与生成模型协作创建大规模高质量训练数据的范式转变。'}


### 论文摘要

Synthetic datasets are widely used for training urban scene recognition models, but even highly realistic renderings show a noticeable gap to real imagery. This gap is particularly pronounced when adapting to a specific target domain, such as Cityscapes, where differences in architecture, vegetation, object appearance, and camera characteristics limit downstream performance. Closing this gap with more detailed 3D modelling would require expensive asset and scene design, defeating the purpose of low-cost labelled data. To address this, we present a new framework that adapts an off-the-shelf diffusion model to a target domain using only imperfect pseudo-labels. Once trained, it generates high-fidelity, target-aligned images from semantic maps of any synthetic dataset, including low-effort sources created in hours rather than months. The method filters suboptimal generations, rectifies image-label misalignments, and standardises semantics across datasets, transforming weak synthetic data into competitive real-domain training sets. Experiments on five synthetic datasets and two real target datasets show segmentation gains of up to +8.0%pt. mIoU over state-of-the-art translation methods, making rapidly constructed synthetic datasets as effective as high-effort, time-intensive synthetic datasets requiring extensive manual design. This work highlights a valuable collaborative paradigm where fast semantic prototyping, combined with generative models, enables scalable, high-quality training data creation for urban scene understanding.

---

## 121. mmWalk: Towards Multi-modal Multi-view Walking Assistance

**论文链接:** [http://arxiv.org/abs/2510.11520v1](http://arxiv.org/abs/2510.11520v1)

**作者:** Kedi Ying, Ruiping Liu, Chongyan Chen, Mingzhe Tao, Hao Shi, Kailun Yang, Jiaming Zhang, Rainer Stiefelhagen

**发布时间:** 2025-10-13

**备注:** Accepted by NeurIPS 2025 Datasets and Benchmarks Track. Data and  Code: https://github.com/KediYing/mmWalk

### GPT解析

### 总结

本研究构建了mmWalk，一个针对盲人或低视力人群的多模态数据集，以及mmWalkVQA基准，用于户外安全导航辅助。研究评估了现有视觉语言模型的表现，并展示了所构建数据集的有效性。

### 背景

盲人或低视力人群在极端或复杂环境中的行走辅助仍然是一个重大挑战，主要是因为缺乏整体场景理解能力。

### 目的

为了满足盲人或低视力社区的实际需求，构建一个模拟的多模态数据集，用于户外安全导航。

### 方法

构建了mmWalk数据集，包含120条手动控制的、场景分类的行走轨迹和62k个同步帧；收集了超过559k张RGB、深度和语义模态的全景图像；每条轨迹都包含户外极端情况和可访问性特定地标；创建了包含69k个视觉问答三元组的mmWalkVQA基准；评估了最先进的视觉语言模型在零样本和少样本设置下的表现。

### 主要发现

最先进的视觉语言模型在风险评估和导航任务上表现不佳；在真实世界数据集上验证了mmWalk微调模型的有效性。

### 结论

所构建的数据集对于推进多模态行走辅助技术有效。

### 翻译

在极端或复杂环境中为盲人或低视力人群(BLV)提供行走辅助仍然是一个重大挑战，主要由于缺乏整体场景理解能力。受BLV社区的现实需求驱动，我们构建了mmWalk，这是一个模拟的多模态数据集，集成了多视图传感器和面向可访问性的特征，用于户外安全导航。我们的数据集包含120条手动控制、场景分类的行走轨迹，有62k个同步帧。它包含RGB、深度和语义模态下的超过559k张全景图像。此外，为了强调实际相关性，每条轨迹都包含户外极端情况和BLV用户特定的可访问性地标。此外，我们生成了mmWalkVQA，一个包含9个类别、超过69k个视觉问答三元组的VQA基准，专为安全和知情行走辅助而定制。我们使用零样本和少样本设置评估了最先进的视觉语言模型(VLMs)，发现它们在风险评估和导航任务上表现不佳。我们在真实世界数据集上验证了mmWalk微调模型，展示了我们的数据集在推进多模态行走辅助方面的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决盲人或低视力人士在极端或复杂环境中行走的安全辅助问题。这个问题很重要，因为全球有超过22亿人受盲症或低视力影响，超过63%的BLV人士在户外导航时受过伤害，7%的人每月至少摔倒一次，而现有导航系统未能充分识别危险状况、不平整表面和临时障碍物。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于BLV社区的真实需求设计方法，在Carla模拟器中手动收集多视角(步行者、导盲犬、无人机)和多模态(RGB、深度、语义分割)数据。借鉴了现有多视图辅助系统如OpenMPR和MSSP，以及视觉辅助数据集如VizWiz、GuideDog和SideGuide。同时参考了ATmaps统计中的地标信息，并定义了8种BLV相关的特殊情况。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个多模态、多视角的行走辅助数据集，特别关注BLV用户的安全需求，通过全景图像和多种传感器数据提供全面的场景理解。流程包括：在模拟器中收集120条轨迹和559k全景图像；标注轨迹元数据和特殊情况；使用GPT-4o生成69k视觉问答三元组；评估多种视觉语言模型；在真实世界数据集上验证微调模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首个结合多视角和可访问性特征的多模态数据集；包含RGB、深度和语义分割的全景图像；专门针对BLV的特殊情况和导航地标；包含69k视觉问答三元组的基准测试；全面评估现有模型局限性。相比之前工作，它特别关注BLV安全需求，提供多视角同步数据，结合全景图像和多种传感器模态，且VQA数据集规模更大、类别更丰富。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'mmWalk论文贡献了一个多模态、多视角的行走辅助数据集和基准测试，特别关注盲人和低视力人士的安全需求，为开发更安全、更全面的行走辅助系统提供了基础。'}


### 论文摘要

Walking assistance in extreme or complex environments remains a significant challenge for people with blindness or low vision (BLV), largely due to the lack of a holistic scene understanding. Motivated by the real-world needs of the BLV community, we build mmWalk, a simulated multi-modal dataset that integrates multi-view sensor and accessibility-oriented features for outdoor safe navigation. Our dataset comprises 120 manually controlled, scenario-categorized walking trajectories with 62k synchronized frames. It contains over 559k panoramic images across RGB, depth, and semantic modalities. Furthermore, to emphasize real-world relevance, each trajectory involves outdoor corner cases and accessibility-specific landmarks for BLV users. Additionally, we generate mmWalkVQA, a VQA benchmark with over 69k visual question-answer triplets across 9 categories tailored for safe and informed walking assistance. We evaluate state-of-the-art Vision-Language Models (VLMs) using zero- and few-shot settings and found they struggle with our risk assessment and navigational tasks. We validate our mmWalk-finetuned model on real-world datasets and show the effectiveness of our dataset for advancing multi-modal walking assistance.

---

## 122. REACT3D: Recovering Articulations for Interactive Physical 3D Scenes

**论文链接:** [http://arxiv.org/abs/2510.11340v1](http://arxiv.org/abs/2510.11340v1)

**作者:** Zhao Huang, Boyang Sun, Alexandros Delitzas, Jiaqi Chen, Marc Pollefeys

**发布时间:** 2025-10-13

**备注:** 8 pages

### GPT解析

### 总结

REACT3D是一个可扩展的零样本框架，能够将静态3D场景转换为模拟就绪的交互式副本，具有一致的几何形状，可直接用于各种下游任务。

### 背景

交互式3D场景对具身智能日益重要，但现有数据集有限，因为注释部分分割、运动类型和运动轨迹的过程是劳动密集型的。

### 目的

开发一个可扩展的零样本框架，解决静态3D场景转换为交互式副本的难题，降低关节场景理解的大规模研究门槛。

### 方法

包括四个主要贡献：可打开物体检测和分割；关节估计推断关节类型和运动参数；隐藏几何形状补全与交互式物体组装；交互式场景集成以确保与标准模拟平台的兼容性。

### 主要发现

在多样化的室内场景中，REACT3D在检测/分割和关节度量方面取得了最先进的性能，证明了框架的有效性。

### 结论

REACT3D为可扩展的交互式场景生成提供了实际基础，为大规模关节场景理解研究创造了条件。

### 翻译

交互式3D场景对于具身智能越来越重要，然而现有的数据集仍然有限，因为注释部分分割、运动类型和运动轨迹的过程是劳动密集型的。我们提出了REACT3D，一个可扩展的零样本框架，将静态3D场景转换为模拟就绪的交互式副本，具有一致的几何形状，能够直接用于各种下游任务。我们的贡献包括：(i)可打开物体检测和分割，从静态场景中提取候选可移动部分；(ii)关节估计，推断关节类型和运动参数；(iii)隐藏几何形状补全，然后进行交互式物体组装；(iv)交互式场景集成，以广泛支持的格式确保与标准模拟平台的兼容性。我们在多样化的室内场景中，在检测/分割和关节度量方面取得了最先进的性能，证明了我们框架的有效性，并为可扩展的交互式场景生成提供了实际基础，从而降低了关节场景理解的大规模研究门槛。我们的项目页面是react3d.github.io。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何将静态3D场景转换为具有交互功能的物理3D场景，特别是识别和恢复场景中可动关节（如门、抽屉等可开合物体）的问题。这个问题在现实中很重要，因为交互式3D场景对虚拟现实、游戏、电影制作以及机器人系统开发至关重要，而现有数据集由于标注过程劳动密集而受限，自动化生成此类场景能有效扩展研究规模并降低使用门槛。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者设计REACT3D时借鉴了多项现有工作：使用RAM++和LLaVA进行语义识别，Grounded SAM进行分割，OPDMulti进行关节估计，并改进了DRAWER的多视图融合方法。作者认识到现有方法在处理开放词汇物体、关节估计精度和隐藏几何生成方面的不足，因此设计了结合视觉基础模型和视觉语言模型的零样本框架，通过开放词汇检测、关节细化、隐藏几何生成和场景集成四个步骤实现静态到交互场景的转换。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视觉基础模型和视觉语言模型从静态3D场景中恢复物体关节，生成物理启用的交互式数字孪生。整体流程分为四个主要步骤：1) 开放物体检测和分割，识别可动物体并提取可动部分；2) 关节估计，推断关节类型和运动参数并进行细化；3) 隐藏几何生成，完成物体内部腔体结构；4) 交互场景集成，将可动物体与静态背景结合，生成纹理并导出为兼容多种模拟平台的格式。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 开放词汇检测方法减轻了标签偏差，提高了对长尾可动物体的覆盖；2) 基于定向边界框的关节细化提高了参数准确性；3) 隐藏几何生成解决了物体内部结构缺失问题；4) 多平台兼容的导出格式确保了广泛适用性。相比前人工作，REACT3D是零样本方法，无需针对特定类别训练；改进了多视图融合策略；引入了几何驱动的关节细化；并考虑了隐藏几何生成，而不仅仅是表面几何。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'REACT3D提供了一个创新的零样本框架，能够将静态3D场景转换为具有物理交互功能的数字孪生体，通过开放词汇检测、关节细化、隐藏几何生成和无缝集成，实现了在多种平台上就绪的交互式场景生成。'}


### 论文摘要

Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks. Our contributions include: (i) openable-object detection and segmentation to extract candidate movable parts from static scenes, (ii) articulation estimation that infers joint types and motion parameters, (iii) hidden-geometry completion followed by interactive object assembly, and (iv) interactive scene integration in widely supported formats to ensure compatibility with standard simulation platforms. We achieve state-of-the-art performance on detection/segmentation and articulation metrics across diverse indoor scenes, demonstrating the effectiveness of our framework and providing a practical foundation for scalable interactive scene generation, thereby lowering the barrier to large-scale research on articulated scene understanding. Our project page is \textit{\hypersetup{urlcolor=black}\href{https://react3d.github.io/}{react3d.github.io}}.

---

## 123. Real2USD: Scene Representations in Universal Scene Description Language

**论文链接:** [http://arxiv.org/abs/2510.10778v1](http://arxiv.org/abs/2510.10778v1)

**作者:** Christopher D. Hsu, Pratik Chaudhari

**发布时间:** 2025-10-12

**备注:** 8 pages, 10 figures, 1 table

### GPT解析

### 总结

本文提出使用通用场景描述（USD）语言作为大型语言模型（LLMs）机器人任务中环境表示的有效通用方法，并展示了'Real to USD'系统的实际应用。

### 背景

现有的大语言模型机器人方法都是针对特定任务的，如用于导航的视觉语言模型、用于映射的语言引导神经辐射场等，缺乏通用性。

### 目的

论证USD语言是LLM机器人任务中环境几何、光度学和语义信息的有效且通用的表示方法。

### 方法

开发'Real to USD'系统，使用搭载LiDAR和RGB相机的Unitree Go2四足机器人构建室内环境USD表示，并用谷歌Gemini解析USD实现场景理解、推理和规划；同时在模拟仓库和医院环境中测试系统。

### 主要发现

USD是基于XML的场景图，可被LLM和人类阅读，足够丰富以支持几乎所有任务，皮克斯开发此语言用于存储资产、场景甚至电影。

### 结论

USD是LLM机器人任务中环境表示的有效通用方法，能够处理多样化物体和具有挑战性的环境。

### 翻译

大型语言模型（LLMs）可以帮助机器人对抽象任务规范进行推理。这需要在机器人使用的经典环境表示基础上增加基于自然语言的先验知识。现有方法都是针对特定任务的，例如用于导航的视觉语言模型、用于映射的语言引导神经辐射场等。本文提出通用场景描述（USD）语言作为LLM机器人任务中环境几何、光度学和语义信息的有效且通用的表示方法。我们的论点很简单：USD是一种基于XML的场景图，可被LLM和人类阅读，足够丰富以支持几乎所有任务——皮克斯开发这种语言是为了存储资产、场景甚至电影。我们使用搭载LiDAR和RGB相机的Unitree Go2四足机器人展示了'Real to USD'系统，该系统能够（i）构建包含多样物体和大量玻璃等挑战性室内环境的显式USD表示，以及（ii）使用谷歌的Gemini解析USD以展示场景理解、复杂推理和规划能力。我们还使用Nvidia的Issac Sim在模拟仓库和医院环境中研究了该系统的不同方面。代码可在https://github.com/grasp-lyrl/Real2USD获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何为机器人创建一个通用的场景表示方法，该方法能够结合几何、光学和语义信息，并且能被大型语言模型读取和理解。这个问题很重要，因为当前机器人系统主要使用度量表示进行环境建模，缺乏人类使用的丰富语义信息，而结合语义和度量信息对机器人构建有效的空间表示至关重要。现有方法通常针对特定任务设计，需要一种通用且有效的场景表示方法来支持基于自然语言的机器人任务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先认识到机器人需要结合语义和度量信息，然后提出使用USD（Universal Scene Description）语言作为解决方案，因为它是基于XML的场景图，可被LLMs和人类读取，且足够丰富支持各种任务。作者借鉴了3D度量-语义场景图方法（如SLAM、神经辐射场）和'Real to Sim to Real'研究思路，以及使用深度学习进行语义分割和基础模型进行大型数据库搜索的方法。作者设计了'Real2USD'系统，使用四足机器人携带传感器构建室内环境的USD表示。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用USD语言作为通用场景表示，将真实世界场景转换为USD格式，使其能被大型语言模型读取和理解，从而支持基于自然语言的任务规划。整体流程包括：1)资产识别和检索（使用YOLOE检测对象，CLIP和FAISS检索资产）；2)资产定位（使用模拟器生成资产点云，通过ICP算法配准）；3)资产协调（使用非极大值抑制和评分系统选择最佳资产，物理模拟确保合理性）；4)场景理解与任务规划（使用LLM解析USD，生成路径点和导航计划）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用USD作为通用场景表示；2)开发了完整的Real2USD系统；3)提出模拟循环配准方法提高准确性；4)设计物理协调机制确保场景合理性；5)实现与LLM的集成支持复杂语义任务。相比之前工作，本文方法更具通用性（而非针对特定任务），使用结构化资产表示（而非非结构化网格），可直接被LLM解析（无需额外注释），并能处理更大更复杂的场景（而非单个图像或文本提示生成的小场景）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出并验证了一种使用USD语言作为通用场景表示的方法，使机器人能够将真实世界环境转换为可被大型语言模型读取和理解的格式，从而支持复杂的语义任务规划。'}


### 论文摘要

Large Language Models (LLMs) can help robots reason about abstract task specifications. This requires augmenting classical representations of the environment used by robots with natural language-based priors. There are a number of existing approaches to doing so, but they are tailored to specific tasks, e.g., visual-language models for navigation, language-guided neural radiance fields for mapping, etc. This paper argues that the Universal Scene Description (USD) language is an effective and general representation of geometric, photometric and semantic information in the environment for LLM-based robotics tasks. Our argument is simple: a USD is an XML-based scene graph, readable by LLMs and humans alike, and rich enough to support essentially any task -- Pixar developed this language to store assets, scenes and even movies. We demonstrate a ``Real to USD'' system using a Unitree Go2 quadruped robot carrying LiDAR and a RGB camera that (i) builds an explicit USD representation of indoor environments with diverse objects and challenging settings with lots of glass, and (ii) parses the USD using Google's Gemini to demonstrate scene understanding, complex inferences, and planning. We also study different aspects of this system in simulated warehouse and hospital settings using Nvidia's Issac Sim. Code is available at https://github.com/grasp-lyrl/Real2USD .

---

## 124. B2N3D: Progressive Learning from Binary to N-ary Relationships for 3D Object Grounding

**论文链接:** [http://arxiv.org/abs/2510.10194v1](http://arxiv.org/abs/2510.10194v1)

**作者:** Feng Xiao, Hongbin Xu, Hai Ci, Wenxiong Kang

**发布时间:** 2025-10-11

### GPT解析

### 背景

使用自然语言定位3D物体对机器人场景理解至关重要。然而，描述中通常涉及多个空间关系来区分相似物体，这使得3D-语言对齐变得困难。当前方法仅对成对物体建模关系，忽略了多模态关系理解中n元组合的全局感知显著性。

### 目的

解决现有方法在3D物体定位中只考虑成对关系而忽视n元组合全局感知的问题，提出一种新的渐进式关系学习框架。

### 方法

将关系学习从二元扩展到n元，识别与参照描述全局匹配的视觉关系；设计分组监督损失促进n元关系学习（因训练数据中缺乏参照对象的特定标注）；在n元关系创建的场景图中，使用具有混合注意力机制的多模态网络进一步定位n元组合中的目标。

### 主要发现

在ReferIt3D和ScanRefer基准上的实验和消融研究表明，该方法优于现有最先进技术，证明了n元关系感知在3D定位中的优势。

### 结论

n元关系感知对3D物体定位至关重要，提出的渐进式关系学习框架有效解决了现有方法的局限性，提高了3D物体定位的准确性。

### 翻译

使用自然语言定位3D物体对机器人场景理解至关重要。描述通常涉及多个空间关系来区分相似物体，这使得3D-语言对齐变得困难。当前方法仅对成对物体建模关系，忽略了多模态关系理解中n元组合的全局感知显著性。为解决这一问题，我们提出了一种用于3D物体定位的新型渐进式关系学习框架。我们将关系学习从二元扩展到n元，以识别与参照描述全局匹配的视觉关系。鉴于训练数据中缺乏参照对象的特定标注，我们设计了一种分组监督损失来促进n元关系学习。在使用n元关系创建的场景图中，我们使用具有混合注意力机制的多模态网络进一步定位n元组合中的目标。在ReferIt3D和ScanRefer基准上的实验和消融研究表明，我们的方法优于最先进技术，证明了n元关系感知在3D定位中的优势。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D物体定位问题，即在3D场景中根据自然语言描述来准确定位特定物体。这个问题在机器人场景理解中至关重要，因为现实世界任务通常需要自然语言指令来指导行动。当场景中有多个相似物体时，人们必须通过多个空间关系来描述目标位置，这需要模型能够同时理解多个关系，而当前方法仅能处理成对物体关系，难以应对复杂场景中的全局关系理解。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性进行思考：当前方法仅建模成对物体关系，忽略了多关系描述的全局感知。作者借鉴了Transformer框架、场景图构建、图神经网络和大型语言模型等现有技术，但创新性地提出了从二元到n元的渐进式关系学习框架。作者特别利用LLM提取实体关系作为训练监督，并设计了分组监督损失来处理指代物体不确定性，从而实现了更全面的关系理解。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过渐进式关系学习从简单二元关系扩展到复杂n元关系，实现全局关系感知。整体流程包括：1) 编码文本和3D物体特征；2) 使用B2N-PRL模块进行二元关系建模，再基于此进行n元关系建模；3) 选择top K2个n元组合构建场景图；4) 通过混合注意力机制(自注意力、图注意力、交叉注意力)更新节点特征；5) 使用MLP输出目标置信度并定位。整个过程通过分组监督损失进行训练，优化关系学习和目标定位。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 从二元到n元的渐进式关系学习框架，实现全局关系感知；2) 分组监督损失设计，处理训练数据中指代物体不确定性；3) 基于n元关系而非所有相邻实体构建场景图，减少噪声；4) 混合注意力机制的多模态网络增强空间感知。相比之前工作，本文突破了仅处理成对关系的局限，通过全局n元关系理解显著提升了复杂多关系描述下的定位准确性，特别是在有相似物体干扰的场景中表现更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'B2N3D通过从二元到n元的渐进式关系学习和注意力驱动的图学习，显著提升了复杂多关系描述下的3D物体定位性能，实现了全局关系感知和更准确的目标定位。'}


### 论文摘要

Localizing 3D objects using natural language is essential for robotic scene understanding. The descriptions often involve multiple spatial relationships to distinguish similar objects, making 3D-language alignment difficult. Current methods only model relationships for pairwise objects, ignoring the global perceptual significance of n-ary combinations in multi-modal relational understanding. To address this, we propose a novel progressive relational learning framework for 3D object grounding. We extend relational learning from binary to n-ary to identify visual relations that match the referential description globally. Given the absence of specific annotations for referred objects in the training data, we design a grouped supervision loss to facilitate n-ary relational learning. In the scene graph created with n-ary relationships, we use a multi-modal network with hybrid attention mechanisms to further localize the target within the n-ary combinations. Experiments and ablation studies on the ReferIt3D and ScanRefer benchmarks demonstrate that our method outperforms the state-of-the-art, and proves the advantages of the n-ary relational perception in 3D localization.

---

## 125. CapGeo: A Caption-Assisted Approach to Geometric Reasoning

**论文链接:** [http://arxiv.org/abs/2510.09302v1](http://arxiv.org/abs/2510.09302v1)

**作者:** Yuying Li, Siyi Qian, Hao Liang, Leqi Zheng, Ruichuan An, Yongzhen Guo, Wentao Zhang

**发布时间:** 2025-10-10

**备注:** preprint, under review

### GPT解析

### 总结

该论文提出CapGeo框架，通过将几何图形转换为文本描述来提升多模态大语言模型的几何推理能力，并创建了CapGeo-Bench评估基准数据集。

### 背景

几何推理是多模态大语言模型的核心挑战，即使是最先进的系统如GPT-O3和Gemini-2.5-Pro在解决几何问题时仍不可靠，尽管它们在文本推理任务上表现优异，表明瓶颈在于理解几何图形而非推理能力。

### 目的

开发一种将视觉内容转换为文本描述的方法，以提升MLLMs的几何推理能力，并创建相应的评估基准。

### 方法

引入CapGeo标题辅助推理框架连接视觉和文本模态，提出CapGeo-Bench数据集包含4,641个精选图形-标题对，并开发基于关键点的评估指标。

### 主要发现

配备标题后模型性能显著提升：Qwen2.5-VL-72B从8.6%提升到59.0%，Claude-Opus-4从44.8%提升到73.0%。

### 结论

CapGeo框架和CapGeo-Bench基准为提升多模态大语言模型中的几何推理能力提供了一条新途径。

### 翻译

几何推理仍然是对多模态大语言模型的核心挑战。即使是最先进的闭源系统，如GPT-O3和Gemini-2.5-Pro，在解决几何问题时仍然不可靠，尽管它们在国际数学奥林匹克竞赛等任务上表现出强大的文本推理能力。这一差距表明，瓶颈在于理解几何图形而非推理本身。由于几何图形通常可以用简洁的文本形式忠实描述，将视觉内容转换为标题是一个有前景的方向。受此启发，我们引入了CapGeo，一种标题辅助推理框架，连接视觉和文本模态。实验表明，当模型配备标题时，性能有显著提升：Qwen2.5-VL-72B从仅使用视觉的8.6%提升到59.0%，而Claude-Opus-4从44.8%提升到73.0%。为了系统评估和识别高质量的几何标题生成模型，我们进一步提出了CapGeo-Bench，一个包含4,641个精选图形-标题对的数据集。重要的是，CapGeo-Bench包含一个基于关键点的评估指标，该指标与下游CapGeo性能高度相关，能够可靠评估几何标题生成能力。我们的框架和基准共同为提升多模态大语言模型中的几何推理能力指明了一条新途径。


### 论文摘要

Geometric reasoning remains a core challenge for Multimodal Large Language Models (MLLMs). Even the most advanced closed-source systems, such as GPT-O3 and Gemini-2.5-Pro, still struggle to solve geometry problems reliably, despite exhibiting strong textual reasoning abilities on tasks like the International Mathematical Olympiad (IMO). This gap suggests that the bottleneck lies in understanding geometric diagrams rather than reasoning itself. Since geometric figures can often be faithfully described in concise textual form, converting visual content into captions offers a promising direction. Motivated by this insight, we introduce CapGeo, a caption-assisted reasoning framework that bridges visual and textual modalities. Experiments show substantial improvements when models are equipped with captions: Qwen2.5-VL-72B improves from 8.6% (vision-only) to 59.0%, while Claude-Opus-4 rises from 44.8% to 73.0%. To systematically evaluate and identify high-quality geometric captioning models, we further propose CapGeo-Bench, a dataset of 4,641 curated figure-caption pairs. Crucially, CapGeo-Bench incorporates a keypoint-based evaluation metric that correlates strongly with downstream CapGeo performance, enabling reliable assessment of geometric captioning ability. Together, our framework and benchmark highlight a new pathway toward advancing geometric reasoning in MLLMs.

---

## 126. CFVBench: A Comprehensive Video Benchmark for Fine-grained Multimodal Retrieval-Augmented Generation

**论文链接:** [http://arxiv.org/abs/2510.09266v1](http://arxiv.org/abs/2510.09266v1)

**作者:** Kaiwen Wei, Xiao Liu, Jie Zhang, Zijian Wang, Ruida Liu, Yuming Yang, Xin Xiao, Xiao Sun, Haoyang Zeng, Changzai Pan, Yidan Zhang, Jiang Zhong, Peijin Wang, Yingchao Feng

**发布时间:** 2025-10-10

### GPT解析

### 总结

本文提出了CFVBench基准和自适应视觉细化(AVR)框架，解决了多模态检索增强生成模型在捕捉细粒度多模态细节方面的瓶颈问题。

### 背景

多模态检索增强生成(MRAG)使多模态大语言模型能够利用外部多模态证据生成响应，但现有基准在模态覆盖和格式多样性方面有限，常专注于单模态任务或粗粒度场景理解。

### 目的

解决现有基准在模态覆盖和格式多样性方面的局限性，引入一个大规模、人工验证的基准来评估模型在检索和生成阶段的能力。

### 方法

构建CFVBench基准，包含599个公开视频产生5,360个开放式问答对，涵盖图表密集报告、新闻广播和软件教程等多种格式和领域；评估7种检索方法和14种MLLMs；提出自适应视觉细化(AVR)框架，自适应增加帧采样密度并选择性调用外部工具。

### 主要发现

当前模型(即使是GPT-4或Gemini)难以捕捉短暂但重要的细粒度多模态细节；AVR框架能够增强细粒度多模态理解，提高所有评估的MLLMs的性能。

### 结论

AVR是一种简单而有效的框架，可以解决当前模型在捕捉细粒度多模态细节方面的瓶颈，提升多模态检索增强生成模型的性能。

### 翻译

多模态检索增强生成(MRAG)使多模态大语言模型能够利用外部多模态证据生成响应，许多基于视频的MRAG基准已被提出以评估模型在检索和生成阶段的能力。然而，现有基准在模态覆盖和格式多样性方面仍然有限，常专注于单模态任务或粗粒度场景理解。为解决这些差距，我们引入了CFVBench，这是一个从599个公开视频中构建的大规模、人工验证的基准，产生5,360个开放式问答对。CFVBench涵盖图表密集报告、新闻广播和软件教程等高密度格式和领域，要求模型检索和推理长时间视频跨度，同时保持细粒度多模态信息。使用CFVBench，我们系统评估了7种检索方法和14种广泛使用的MLLMs，揭示了一个关键瓶颈：当前模型(即使是GPT-4或Gemini)难以捕捉短暂但重要的细粒度多模态细节。为缓解这一问题，我们提出了自适应视觉细化(AVR)，这是一个简单而有效的框架，自适应增加帧采样密度并在必要时选择性调用外部工具。实验表明，AVR一致地增强细粒度多模态理解，并提高所有评估的MLLMs的性能。


### 论文摘要

Multimodal Retrieval-Augmented Generation (MRAG) enables Multimodal Large Language Models (MLLMs) to generate responses with external multimodal evidence, and numerous video-based MRAG benchmarks have been proposed to evaluate model capabilities across retrieval and generation stages. However, existing benchmarks remain limited in modality coverage and format diversity, often focusing on single- or limited-modality tasks, or coarse-grained scene understanding. To address these gaps, we introduce CFVBench, a large-scale, manually verified benchmark constructed from 599 publicly available videos, yielding 5,360 open-ended QA pairs. CFVBench spans high-density formats and domains such as chart-heavy reports, news broadcasts, and software tutorials, requiring models to retrieve and reason over long temporal video spans while maintaining fine-grained multimodal information. Using CFVBench, we systematically evaluate 7 retrieval methods and 14 widely-used MLLMs, revealing a critical bottleneck: current models (even GPT5 or Gemini) struggle to capture transient yet essential fine-grained multimodal details. To mitigate this, we propose Adaptive Visual Refinement (AVR), a simple yet effective framework that adaptively increases frame sampling density and selectively invokes external tools when necessary. Experiments show that AVR consistently enhances fine-grained multimodal comprehension and improves performance across all evaluated MLLMs

---

## 127. BEAR: Benchmarking and Enhancing Multimodal Language Models for Atomic Embodied Capabilities

**论文链接:** [http://arxiv.org/abs/2510.08759v1](http://arxiv.org/abs/2510.08759v1)

**作者:** Yu Qi, Haibo Zhao, Ziyu Guo, Siyuan Ma, Ziyan Chen, Yaokun Han, Renrui Zhang, Zitiantao Lin, Shiji Xin, Yijian Huang, Kai Cheng, Peiheng Wang, Jiazheng Liu, Jiayi Zhang, Yizhe Zhu, Wenqing Wang, Yiran Qin, Xupeng Zhu, Haojie Huang, Lawson L. S. Wong

**发布时间:** 2025-10-09

### GPT解析

### 总结

该研究引入了BEAR基准测试，用于全面评估多模态大语言模型(MLLMs)的具身能力，并提出了BEAR-Agent模型来提升这些能力。研究显示现有MLLMs在具身能力方面存在局限，而新模型能显著改善性能。

### 背景

具身能力是指代理感知、理解和与物理世界交互的基本能力。尽管MLLMs作为具身代理有潜力，但现有基准测试主要关注特定领域，缺乏对MLLMs具身能力的全面系统性评估。

### 目的

弥补对MLLMs具身能力评估的空白，引入一个全面且细致的基准测试来评估MLLMs在基本具身能力方面的表现。

### 方法

提出BEAR基准测试，包含6个类别14个领域的4469个交错图像-视频-文本条目，涵盖从低级指向、轨迹理解到高级规划的任务；评估20个代表性MLLMs；提出BEAR-Agent，一个整合预训练视觉模型的多模态可对话代理，以增强MLLMs的感知、3D理解和规划能力。

### 主要发现

20个代表性MLLMs在所有具身能力领域都存在持续限制；BEAR-Agent显著提高了MLLMs在BEAR上的表现，在GPT-5上实现了9.12%的绝对增益和17.5%的相对改进；提高MLLMs的具身能力有利于模拟环境中的具身任务。

### 结论

BEAR基准测试填补了对MLLMs具身能力系统评估的空白；BEAR-Agent模型能有效提升MLLMs在具身能力方面的表现，为具身AI研究提供了新方向。

### 翻译

具身能力是指代理用于感知、理解和与物理世界交互的一系列基本能力。虽然多模态大语言模型(MLLMs)作为具身代理显示出潜力，但对它们具身能力的全面系统性评估仍然不足，因为现有基准测试主要关注规划或空间理解等特定领域。为了弥补这一差距，我们引入了BEAR，这是一个全面且细致的基准，用于评估MLLMs在基本具身能力方面的表现。BEAR包含6个类别14个领域中的4469个交错图像-视频-文本条目，包括从低级指向、轨迹理解、空间推理到高级规划的任务。对20个代表性MLLMs的广泛评估结果显示，它们在所有具身能力领域都存在持续的限制。为了解决这一不足，我们提出了BEAR-Agent，一个整合预训练视觉模型的多模态可对话代理，以增强MLLMs的感知、3D理解和规划能力。它在BEAR上显著提高了MLLMs在各种具身能力方面的表现，在GPT-5上实现了9.12%的绝对增益和17.5%的相对改进。此外，我们的实验表明，提高MLLMs的具身能力有利于模拟环境中的具身任务。项目网站：https://bear-official66.github.io/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多模态大语言模型(MLLMs)缺乏系统性的具身能力评估问题。这个问题很重要，因为具身能力是AI系统感知、理解和与物理世界交互的基础能力，对开发能在开放环境中有效工作的AI代理至关重要。现有评估基准主要关注特定领域，无法全面评估MLLMs的具身能力，限制了我们对它们潜力的理解和开发方向的指导。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有工作的局限性来设计BEAR基准。他们发现现有工作要么专注于特定领域(如指向、空间理解)，要么关注能力导向任务但未分解为逐步技能。作者从大规模具身家庭活动数据集(如BEHAVIOR-1K和ALFRED)中归纳分类，从人类认知过程中获取灵感，设计了将具身能力组织为6个类别和14个原子技能的基准。他们确实借鉴了现有工作，包括使用多种视觉模型和工具来增强BEAR-Agent，并参考了现有数据集和评估方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建全面基准(BEAR)系统评估MLLMs的具身能力，并基于评估结果开发增强型代理(BEAR-Agent)提升这些能力。BEAR基准包含4,469个交错图像-视频-文本条目，组织为6个类别和14个原子技能。BEAR-Agent是一个多模态可对话代理，通过对话与MLLM交互，提供工具增强视觉和空间能力，为不同类别提供特定模块(如对象检测、深度估计)，通过提供额外线索帮助模型生成更准确答案，最终提升MLLMs在具身任务中的表现。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) BEAR基准首次系统评估MLLMs具身能力，将能力组织为6个类别和14个原子技能；2) 包含4,469个交错图像-视频-文本条目，首次将具身任务分解为面向技能的步骤；3) BEAR-Agent多模态可对话代理显著提升性能，GPT-5提升9.12%；4) 在模拟环境中也表现优异，提升超过20.17%。相比之前工作，BEAR首次提供全面细粒度评估，BEAR-Agent不仅提升离线评估能力，还改进实际任务执行，并提供详细失败分析揭示MLLMs瓶颈。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了BEAR基准首次系统评估多模态大语言模型的具身能力，并基于评估结果开发了BEAR-Agent，显著提升了这些能力在评估和实际任务中的表现。'}


### 论文摘要

Embodied capabilities refer to a suite of fundamental abilities for an agent to perceive, comprehend, and interact with the physical world. While multimodal large language models (MLLMs) show promise as embodied agents, a thorough and systematic evaluation of their embodied capabilities remains underexplored, as existing benchmarks primarily focus on specific domains such as planning or spatial understanding. To bridge this gap, we introduce BEAR, a comprehensive and fine-grained benchmark that evaluates MLLMs on atomic embodied capabilities. BEAR comprises 4,469 interleaved image-video-text entries across 14 domains in 6 categories, including tasks from low-level pointing, trajectory understanding, spatial reasoning, to high-level planning. Extensive evaluation results of 20 representative MLLMs reveal their persistent limitations across all domains of embodied capabilities. To tackle the shortfall, we propose BEAR-Agent, a multimodal conversable agent that integrates pretrained vision models to strengthen MLLM perception, 3D understanding, and planning capabilities. It substantially enhances MLLM performance across diverse embodied capabilities on BEAR, yielding a 9.12% absolute gain and a relative improvement of 17.5% on GPT-5. Furthermore, our experiments indicate that improving MLLM embodied capabilities can benefit embodied tasks in simulated environments. Project website: https://bear-official66.github.io/

---

## 128. USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots

**论文链接:** [http://arxiv.org/abs/2510.07869v2](http://arxiv.org/abs/2510.07869v2)

**作者:** Junwen Gu, Zhiheng wu, Pengxuan Si, Shuang Qiu, Yukai Feng, Luoyang Sun, Laien Luo, Lianyi Yu, Jian Wang, Zhengxing Wu

**发布时间:** 2025-10-09

**备注:** Project Page: https://vincentgu2000.github.io/u0project/

### GPT解析

### 总结

本文提出了USIM数据集和U0模型，用于解决水下机器人面临的挑战，通过多任务视觉-语言-动作框架实现水下机器人的自主操作。

### 背景

水下环境对机器人操作提出了独特挑战，包括复杂流体动力学、能见度有限和通信受限。虽然数据驱动方法已推动陆地机器人发展，但开发能自主执行多项任务的水下智能仍然困难，因为大规模高质量水下数据集稀缺。

### 目的

为了解决这些限制，作者引入了USIM，这是一个基于模拟的水下机器人多任务视觉-语言-动作数据集。

### 方法

USIM包含来自1,852个轨迹的超过561K帧，总计约15.6小时的BlueROV2交互，涵盖9种不同场景中的20项任务。基于此数据集，作者提出了U0模型，通过多模态融合整合双目视觉和其他传感器模态，并采用基于卷积-注意力的感知焦点增强模块提高空间理解和移动操作能力。

### 主要发现

在检查、避障、扫描和动态跟踪等任务中，该框架实现了80%的成功率；在具有挑战性的移动操作任务中，与基线方法相比，将到目标的距离减少了21.2%。

### 结论

USIM和U0表明VLA模型可以有效地应用于水下机器人应用，为可扩展数据集构建、提高任务自主性和实现智能通用水下机器人的实际应用奠定了基础。

### 翻译

水下环境为机器人操作带来了独特的挑战，包括复杂的流体动力学、有限的可见性和受限的通信。尽管数据驱动方法已经推动了陆地机器人的具身智能发展，并使特定任务的水下自主机器人成为可能，但开发能够自主执行多项任务的水下智能仍然极具挑战性，因为大规模、高质量的水下数据集仍然稀缺。为了解决这些限制，我们引入了USIM，这是一个基于模拟的水下机器人多任务视觉-语言-动作数据集。USIM包含来自1,852个轨迹的超过561K帧，总计约15.6小时的BlueROV2交互，涵盖9种不同场景中的20项任务，范围从视觉导航到移动操作。基于此数据集，我们提出了U0，这是一个面向通用水下机器人的VLA模型，该模型通过多模态融合整合双目视觉和其他传感器模态，并进一步采用基于卷积-注意力的感知焦点增强模块（CAP）来提高空间理解和移动操作能力。在检查、避障、扫描和动态跟踪等任务中，该框架实现了80%的成功率，而在具有挑战性的移动操作任务中，与基线方法相比，它将到目标的距离减少了21.2%，证明了其有效性。USIM和U0表明VLA模型可以有效地应用于水下机器人应用，为可扩展数据集构建、提高任务自主性和实现智能通用水下机器人的实际应用奠定了基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决水下机器人领域缺乏大规模高质量数据集的问题，导致难以开发能够自主执行多任务的通用水下智能。这个问题很重要，因为水下环境对人类操作极具挑战性，而海洋覆盖地球表面71%，开发自主水下机器人能极大扩展人类探索海洋的能力；同时，水下任务目前仍严重依赖人工远程操作，成本高且效率低，而真实水下环境收集数据既昂贵又有风险。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到水下环境的独特挑战（流体动力学、能见度限制、通信约束）和现有水下数据集的局限性，因此决定采用模拟环境来收集大规模数据。他们借鉴了室内具身智能领域的进展，如DROID、Open X-Embodiment等数据集和RT-2、GR00T N1.5等VLA模型，同时参考了Stonefish等水下模拟器。基于这些现有工作，作者设计了USIM数据集和U0模型，专门针对水下环境的特性进行优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过构建大规模模拟水下数据集和专门设计的视觉-语言-动作模型来解决水下机器人数据稀缺问题。整体流程包括：1) 使用Stonefish模拟器构建9种多样化水下场景；2) 在这些场景中收集20个任务的561K帧数据，形成USIM数据集；3) 基于Isaac-GR00T N1.5构建U0模型，整合多模态传感器数据融合和卷积-注意力感知焦点增强模块(CAP)；4) 通过离线评估和在线测试验证模型效果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) USIM数据集：首个大规模多任务水下VLA数据集，覆盖9个场景中的20个任务；2) U0模型：专为水下机器人设计的VLA模型，整合多模态传感器融合和CAP模块；3) 可扩展的数据到任务框架。相比之前工作，USIM解决了现有水下数据集任务单一、多样性不足的问题；U0则针对水下环境的独特挑战进行了优化，特别是在空间理解和移动操作方面表现出色；整体框架实现了80%的任务成功率，比基线方法提升显著。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过构建首个大规模水下多任务视觉-语言-动作数据集USIM并提出专门针对水下环境的U0模型，为开发具有多任务自主能力的通用水下机器人奠定了基础。'}


### 论文摘要

Underwater environments present unique challenges for robotic operation, including complex hydrodynamics, limited visibility, and constrained communication. Although data-driven approaches have advanced embodied intelligence in terrestrial robots and enabled task-specific autonomous underwater robots, developing underwater intelligence capable of autonomously performing multiple tasks remains highly challenging, as large-scale, high-quality underwater datasets are still scarce. To address these limitations, we introduce USIM, a simulation-based multi-task Vision-Language-Action (VLA) dataset for underwater robots. USIM comprises over 561K frames from 1,852 trajectories, totaling approximately 15.6 hours of BlueROV2 interactions across 20 tasks in 9 diverse scenarios, ranging from visual navigation to mobile manipulation. Building upon this dataset, we propose U0, a VLA model for general underwater robots, which integrates binocular vision and other sensor modalities through multimodal fusion, and further incorporates a convolution-attention-based perception focus enhancement module (CAP) to improve spatial understanding and mobile manipulation. Across tasks such as inspection, obstacle avoidance, scanning, and dynamic tracking, the framework achieves a success rate of 80%, while in challenging mobile manipulation tasks, it reduces the distance to the target by 21.2% compared with baseline methods, demonstrating its effectiveness. USIM and U0 show that VLA models can be effectively applied to underwater robotic applications, providing a foundation for scalable dataset construction, improved task autonomy, and the practical realization of intelligent general underwater robots.

---

## 129. Out-of-Distribution Detection in LiDAR Semantic Segmentation Using Epistemic Uncertainty from Hierarchical GMMs

**论文链接:** [http://arxiv.org/abs/2510.08631v1](http://arxiv.org/abs/2510.08631v1)

**作者:** Hanieh Shojaei Miandashti, Claus Brenner

**发布时间:** 2025-10-08

### GPT解析

### 总结

本研究提出了一种无监督的分布外(OOD)对象检测方法，通过深度神经网络特征空间中高斯混合模型参数的层次贝叶斯建模来提取认知不确定性，无需辅助数据或额外训练阶段即可实现显著性能提升。

### 背景

除了通过LiDAR点云的精确语义分割实现准确场景理解外，检测分布外(OOD)对象（即训练过程中未遇到的实例）对于防止将未知对象错误分配到已知类别至关重要。

### 目的

解决现有无监督OOD检测方法中认知不确定性和偶然不确定性混淆的问题，避免将分布中的模糊区域错误分类为OOD。

### 方法

提出一种基于深度神经网络特征空间中高斯混合模型参数层次贝叶斯建模的无监督OOD检测方法，专门提取认知不确定性而非依赖预测熵。

### 主要发现

在SemanticKITTI数据集上，该方法优于现有基于不确定性的方法，AUROC提高18%，AUPRC增加22%，FPR95降低36%（从76%降至40%）。

### 结论

所提出的方法无需辅助数据或额外训练阶段，即可有效区分已知和未知对象，显著提升了无监督OOD检测的性能表现。

### 翻译

除了通过LiDAR点云的精确语义分割实现准确场景理解外，检测分布外(OOD)对象（即在训练过程中未遇到的实例）对于防止将未知对象错误分配到已知类别至关重要。虽然监督式OOD检测方法依赖于辅助的OOD数据集，但无监督方法避免了这一需求，通常仅依赖于预测熵，即通过对集成或多个后验权重样本的平均获得的预测分布的熵。然而，这些方法经常将认知（模型）不确定性和偶然（数据）不确定性混淆，将分布中的模糊区域错误地分类为OOD。为解决这一问题，我们提出了一种无监督OOD检测方法，该方法采用基于深度神经网络特征空间中高斯混合模型参数层次贝叶斯建模的认知不确定性。无需辅助数据或额外训练阶段，我们的方法在SemanticKITTI数据集上优于现有的基于不确定性的方法，与先前工作中使用的预测熵方法相比，AUROC提高了18%，AUPRC增加了22%，FPR95降低了36%（从76%降至40%）。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR语义分割中检测分布外(Out-of-Distribution, OOD)对象的问题，即那些在训练过程中未遇到的实例。这个问题在自动驾驶等安全关键应用中非常重要，因为现实世界环境中经常包含训练数据中未见过的对象，而深度模型往往对这些OOD对象做出过度自信的错误预测。有效的OOD检测可以防止系统将未知对象错误分类为已知类别，提高自主系统的安全性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：监督方法需要辅助OOD数据集，而非监督方法通常依赖预测熵，但这种方法会将认识不确定性(模型不确定性)和偶然不确定性(数据不确定性)混为一谈，导致误判。作者借鉴了GMMSeg使用高斯混合模型(GMM)在特征空间中建模语义类的思想，以及分层贝叶斯不确定性建模方法。通过结合这些现有工作的优势，作者设计了一种基于认识不确定性的无监督OOD检测方法，能够在不需要额外数据或训练的情况下有效区分OOD样本。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用从GMM参数的分层贝叶斯建模中获得的认识不确定性来进行OOD检测，避免传统预测熵方法将认识不确定性和偶然不确定性混合的问题。整体流程包括：1)使用深度神经网络提取LiDAR点云特征；2)在特征空间中使用类条件高斯混合模型建模每个语义类；3)通过分层贝叶斯方法对GMM参数进行建模；4)从参数后验分布中采样并计算认识不确定性；5)使用熵值作为不确定性的度量；6)将高不确定性像素识别为OOD样本。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出基于认识不确定性的无监督OOD检测方法；2)有效区分认识不确定性和偶然不确定性；3)不需要辅助数据或额外训练阶段；4)在SemanticKITTI数据集上实现了显著性能提升。相比之前的工作，与监督方法不同，它不需要辅助OOD数据；与传统预测熵方法不同，它能更好地区分两种不确定性；与MC Dropout和深度集合方法不同，它不需要重新训练网络；与原始GMMSeg相比，它增加了分层贝叶斯不确定性估计，提供了更强的检测能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于特征空间中高斯混合模型参数认识不确定性的无监督方法，有效解决了LiDAR语义分割中分布外对象检测问题，显著提高了检测准确性和系统安全性。'}


### 论文摘要

In addition to accurate scene understanding through precise semantic segmentation of LiDAR point clouds, detecting out-of-distribution (OOD) objects, instances not encountered during training, is essential to prevent the incorrect assignment of unknown objects to known classes. While supervised OOD detection methods depend on auxiliary OOD datasets, unsupervised methods avoid this requirement but typically rely on predictive entropy, the entropy of the predictive distribution obtained by averaging over an ensemble or multiple posterior weight samples. However, these methods often conflate epistemic (model) and aleatoric (data) uncertainties, misclassifying ambiguous in distribution regions as OOD. To address this issue, we present an unsupervised OOD detection approach that employs epistemic uncertainty derived from hierarchical Bayesian modeling of Gaussian Mixture Model (GMM) parameters in the feature space of a deep neural network. Without requiring auxiliary data or additional training stages, our approach outperforms existing uncertainty-based methods on the SemanticKITTI dataset, achieving an 18\% improvement in AUROC, 22\% increase in AUPRC, and 36\% reduction in FPR95 (from 76\% to 40\%), compared to the predictive entropy approach used in prior works.

---

## 130. Beyond CNNs: Efficient Fine-Tuning of Multi-Modal LLMs for Object Detection on Low-Data Regimes

**论文链接:** [http://arxiv.org/abs/2510.08589v1](http://arxiv.org/abs/2510.08589v1)

**作者:** Nirmal Elamon, Rouzbeh Davoudi

**发布时间:** 2025-10-03

### GPT解析

### 总结

本研究对比了传统CNN模型、零样本预训练的多模态大型语言模型(LLMs)和微调后的多模态LLMs在图像中人工文本叠加检测任务上的表现，发现LLMs在少量数据(少于1,000张图像)微调后可显著提升性能，达到36%的准确率提升，匹配或超过需要大量数据的CNN基线方法。

### 背景

物体检测和理解领域受传统CNN模型(如ResNet和YOLO)和新兴多模态大型语言模型(LLMs)的共同推动发展。CNN模型在图像任务中仍然有效，而基于transformer的LLMs引入了动态上下文推理、语言引导提示和整体场景理解等新能力，但即用型LLMs在专业视觉任务中表现往往不佳。

### 目的

通过对比微调的传统CNN、零样本预训练的多模态LLMs和微调的多模态LLMs，研究如何有效利用LLMs进行图像中人工文本叠加检测这一具有挑战性的任务，探索语言引导模型在最小监督下适应精确视觉理解的方法。

### 方法

进行了全面的比较研究，探索了在少量数据(少于1,000张图像)上微调LLMs的方法，研究如何将语言引导模型适应精确视觉理解，并评估了这些方法在人工文本叠加检测任务上的表现。

### 主要发现

LLMs可以在非常有限的数据(少于1,000张图像)上进行有效微调，实现高达36%的准确率提升，匹配或超过基于CNN的基线方法，而这些方法通常需要数量级更多的数据。这表明基于LLM的方法在真实世界物体检测任务中具有高适应性和数据效率。

### 结论

研究结果表明，通过少量数据微调，LLMs可以在专业视觉任务上取得优异表现，为在低资源视觉环境中应用多模态transformer提供了可行的指导。研究团队已将微调模型的代码公开在GitHub上，以支持该领域的持续进步。

### 翻译

物体检测和理解领域正迅速发展，这既得益于传统基于CNN模型的进步，也得益于新兴的多模态大型语言模型(LLMs)的发展。虽然ResNet和YOLO等CNN模型在基于图像的任务中仍然非常有效，但最近的基于transformer的LLMs引入了动态上下文推理、语言引导提示和整体场景理解等新能力。然而，当即用型LLMs用于专业视觉任务时，其全部潜力仍未被充分开发，往往导致次优性能。在本工作中，我们在图像中人工文本叠加检测这一具有挑战性的任务上，对微调的传统CNN、零样本预训练的多模态LLMs和微调的多模态LLMs进行了全面比较。我们研究的一个关键贡献是证明了LLMs可以在非常有限的数据(少于1,000张图像)上进行有效微调，实现高达36%的准确率提升，匹配或超过通常需要数量级更多数据的基于CNN的基线方法。通过探索语言引导模型如何在最小监督下适应精确视觉理解，我们的研究为弥合视觉和语言的更广泛努力做出了贡献，为高效的跨模态学习策略提供了新的见解。这些研究结果突显了基于LLM的方法在真实世界物体检测任务中的适应性和数据效率，并提供了在低资源视觉环境中应用多模态transformer的可行指导。为了支持该领域的持续进步，我们已在GitHub上公开了用于微调模型的代码，以便未来改进和相关应用中的重用。


### 论文摘要

The field of object detection and understanding is rapidly evolving, driven by advances in both traditional CNN-based models and emerging multi-modal large language models (LLMs). While CNNs like ResNet and YOLO remain highly effective for image-based tasks, recent transformer-based LLMs introduce new capabilities such as dynamic context reasoning, language-guided prompts, and holistic scene understanding. However, when used out-of-the-box, the full potential of LLMs remains underexploited, often resulting in suboptimal performance on specialized visual tasks. In this work, we conduct a comprehensive comparison of fine-tuned traditional CNNs, zero-shot pre-trained multi-modal LLMs, and fine-tuned multi-modal LLMs on the challenging task of artificial text overlay detection in images. A key contribution of our study is demonstrating that LLMs can be effectively fine-tuned on very limited data (fewer than 1,000 images) to achieve up to 36% accuracy improvement, matching or surpassing CNN-based baselines that typically require orders of magnitude more data. By exploring how language-guided models can be adapted for precise visual understanding with minimal supervision, our work contributes to the broader effort of bridging vision and language, offering novel insights into efficient cross-modal learning strategies. These findings highlight the adaptability and data efficiency of LLM-based approaches for real-world object detection tasks and provide actionable guidance for applying multi-modal transformers in low-resource visual environments. To support continued progress in this area, we have made the code used to fine-tune the models available in our GitHub, enabling future improvements and reuse in related applications.

---

## 131. Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer

**论文链接:** [http://arxiv.org/abs/2510.03342v2](http://arxiv.org/abs/2510.03342v2)

**作者:** Gemini Robotics Team, Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, Michael Bloesch, Konstantinos Bousmalis, Philemon Brakel, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Christine Chan, Oscar Chang, London Chappellet-Volpini, Jose Enrique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, Adrian Collister, David B. D'Ambrosio, Sudeep Dasari, Todor Davchev, Meet Kirankumar Dave, Coline Devin, Norman Di Palo, Tianli Ding, Carl Doersch, Adil Dostmohamed, Yilun Du, Debidatta Dwibedi, Sathish Thoppay Egambaram, Michael Elabd, Tom Erez, Xiaolin Fang, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Ruiqi Gao, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Oliver Groth, Agrim Gupta, Roland Hafner, Steven Hansen, Leonard Hasenclever, Sam Haves, Nicolas Heess, Brandon Hernaez, Alex Hofer, Jasmine Hsu, Lu Huang, Sandy H. Huang, Atil Iscen, Mithun George Jacob, Deepali Jain, Sally Jesmonth, Abhishek Jindal, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Matija Kecman, J. Chase Kew, Donnie Kim, Frank Kim, Junkyung Kim, Thomas Kipf, Sean Kirmani, Ksenia Konyushkova, Li Yang Ku, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Tuan Anh Le, Isabel Leal, Alex X. Lee, Tsang-Wei Edward Lee, Guy Lever, Jacky Liang, Li-Heng Lin, Fangchen Liu, Shangbang Long, Caden Lu, Sharath Maddineni, Anirudha Majumdar, Kevis-Kokitsi Maninis, Andrew Marmon, Sergio Martinez, Assaf Hurwitz Michaely, Niko Milonopoulos, Joss Moore, Robert Moreno, Michael Neunert, Francesco Nori, Joy Ortiz, Kenneth Oslund, Carolina Parada, Emilio Parisotto, Amaris Paryag, Acorn Pooley, Thomas Power, Alessio Quaglino, Haroon Qureshi, Rajkumar Vasudeva Raju, Helen Ran, Dushyant Rao, Kanishka Rao, Isaac Reid, David Rendleman, Krista Reymann, Miguel Rivas, Francesco Romano, Yulia Rubanova, Peter Pastor Sampedro, Pannag R Sanketi, Dhruv Shah, Mohit Sharma, Kathryn Shea, Mohit Shridhar, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Rachel Sterneck, Ian Storz, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Saran Tunyasuvunakool, Jake Varley, Grace Vesom, Giulia Vezzani, Maria Bauza Villalonga, Oriol Vinyals, René Wagner, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Chengda Wu, Markus Wulfmeier, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Jimmy Yan, Sherry Yang, Skye Yang, Yuxiang Yang, Hiu Hong Yu, Wenhao Yu, Wentao Yuan, Yuan Yuan, Jingwei Zhang, Tingnan Zhang, Zhiyuan Zhang, Allan Zhou, Guangyao Zhou, Yuxiang Zhou

**发布时间:** 2025-10-02

### GPT解析

### 总结

该研究介绍了Gemini Robotics模型家族的最新版本，包括Gemini Robotics 1.5和Gemini Robotics-ER1.5，通过三项创新提高了机器人的通用推理和任务执行能力。

### 背景

通用机器人需要深入理解物理世界、高级推理能力和通用灵巧的控制能力。

### 目的

介绍Gemini Robotics模型家族的最新版本，提高机器人解决复杂多步骤任务的能力。

### 方法

三项主要创新：1) 采用新颖架构和运动转移机制，从异构多形态机器人数据中学习；2) 在自然语言中将动作与多级内部推理过程交错进行，实现'行动前思考'；3) 建立新的具身推理最先进水平，提升视觉空间理解、任务规划和进度估计能力。

### 主要发现

Gemini Robotics 1.5能够更好地分解和执行复杂多步骤任务，行为更具可解释性；Gemini Robotics-ER1.5在具身推理方面达到新水平。

### 结论

这一系列模型使机器人能够感知、思考然后行动，朝着解决复杂多步骤任务的物理代理时代迈进。

### 翻译

通用机器人需要深入理解物理世界、高级推理能力和通用灵巧的控制。本报告介绍了Gemini Robotics模型家族的最新一代：Gemini Robotics 1.5，一个多形态视觉-语言-动作模型，以及Gemini Robotics-ER1.5，一个最先进的具身推理模型。我们带来了三大创新。首先，Gemini Robotics 1.5采用新颖架构和运动转移机制，使其能够从异构的多形态机器人数据中学习，使VLA更加通用。其次，Gemini Robotics 1.5在自然语言中将动作与多级内部推理过程交错进行，使机器人能够在行动前'思考'，显著提高其分解和执行复杂多步骤任务的能力，并使机器人的行为对用户更具可解释性。第三，Gemini Robotics-ER 1.5为具身推理建立了新的最先进水平，即对机器人至关重要的推理能力，如视觉和空间理解、任务规划和进度估计。总之，这一系列模型使我们迈向物理代理的新时代，使机器人能够感知、思考然后行动，从而解决复杂的多步骤任务。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何开发通用的机器人系统，使机器人能够深入理解物理世界、进行高级推理并执行灵活控制。这个问题非常重要，因为当前机器人大多只能执行特定任务，缺乏适应不同环境和任务的能力，而通用机器人可以大大扩展应用范围，从工业制造到家庭服务等多个领域，解决劳动力短缺和提高生活质量等问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到通用机器人需要三个核心能力：物理世界理解、高级推理和灵活控制。他们设计了一个包含两个主要模型的系统：Gemini Robotics 1.5(VLA模型)和Gemini Robotics-ER 1.5(ER模型)。作者借鉴了现有的Gemini模型基础架构、VLA模型范式和具身推理概念，同时引入了新的Motion Transfer机制和Thinking机制，使机器人能够在执行动作前进行思考，并实现不同机器人形态间的技能迁移。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将机器人系统分为高级推理规划模块和低级执行模块，引入'Thinking'机制使机器人在执行前进行思考，并使用Motion Transfer技术实现跨形态技能迁移。整体流程是：用户输入任务→高级推理模块理解需求、制定计划并调用外部工具→低级执行模块接收指令、分解步骤、执行前思考并转化为具体动作→两个模块协同工作，形成完整智能体系统处理复杂多步骤任务。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)Thinking VLA机制，提高复杂任务处理能力；2)Motion Transfer机制，实现跨形态机器人技能零样本迁移；3)高级具身推理能力，在多种推理任务上达到最先进性能；4)多形态通用机器人系统，单一模型控制多种不同机器人。相比之前工作，1.5版本引入思考机制提升复杂任务执行能力，能够处理多种机器人形态，在多步骤任务上有显著提升，并在保持通用能力的同时在具身推理任务上达到最先进水平。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Gemini Robotics 1.5通过引入思考机制、跨形态技能迁移和高级具身推理能力，显著提升了通用机器人在复杂物理世界中的感知、思考和行动能力，为实现真正的通用机器人系统提供了重要进展。'}


### 论文摘要

General-purpose robots need a deep understanding of the physical world, advanced reasoning, and general and dexterous control. This report introduces the latest generation of the Gemini Robotics model family: Gemini Robotics 1.5, a multi-embodiment Vision-Language-Action (VLA) model, and Gemini Robotics-ER 1.5, a state-of-the-art Embodied Reasoning (ER) model. We are bringing together three major innovations. First, Gemini Robotics 1.5 features a novel architecture and a Motion Transfer (MT) mechanism, which enables it to learn from heterogeneous, multi-embodiment robot data and makes the VLA more general. Second, Gemini Robotics 1.5 interleaves actions with a multi-level internal reasoning process in natural language. This enables the robot to "think before acting" and notably improves its ability to decompose and execute complex, multi-step tasks, and also makes the robot's behavior more interpretable to the user. Third, Gemini Robotics-ER 1.5 establishes a new state-of-the-art for embodied reasoning, i.e., for reasoning capabilities that are critical for robots, such as visual and spatial understanding, task planning, and progress estimation. Together, this family of models takes us a step towards an era of physical agents-enabling robots to perceive, think and then act so they can solve complex multi-step tasks.

---

## 132. MoMaps: Semantics-Aware Scene Motion Generation with Motion Maps

**论文链接:** [http://arxiv.org/abs/2510.11107v1](http://arxiv.org/abs/2510.11107v1)

**作者:** Jiahui Lei, Kyle Genova, George Kopanas, Noah Snavely, Leonidas Guibas

**发布时间:** 2025-10-13

**备注:** Accepted at ICCV 2025, project page:  https://jiahuilei.com/projects/momap/

### GPT解析

### 总结

本文提出了一种新颖的像素对齐运动图(MoMap)表示方法，用于从真实世界视频中学习语义和功能上有意义的3D运动先验，实现从单图像预测未来3D场景运动。

### 背景

从真实世界视频中学习语义和功能上有意义的3D运动先验是一项挑战。

### 目的

能够从单个输入图像预测未来的3D场景运动。

### 方法

提出像素对齐的运动图(MoMap)表示方法，从超过50,000个真实视频中创建大规模MoMap数据库，并训练扩散模型；运动生成流程包括先生成MoMap，然后扭曲图像并完成扭曲的点基渲染。

### 主要发现

实验结果表明，该方法能够生成合理且语义一致的3D场景运动。

### 结论

通过MoMap表示和扩散模型训练，可以有效地从真实视频中学习3D运动先验，并实现从单图像预测未来3D场景运动。

### 翻译

本文解决了从真实世界视频中学习语义和功能上有意义的3D运动先验的挑战，目的是能够从单个输入图像预测未来的3D场景运动。我们提出了一种新颖的像素对齐的运动图(MoMap)表示方法用于3D场景运动，可以从现有的生成图像模型生成，以促进高效和有效的运动预测。为了学习有意义的运动分布，我们从超过50,000个真实视频中创建了大规模的MoMap数据库，并基于这些表示训练了一个扩散模型。我们的运动生成不仅在3D中合成轨迹，还为2D视频合成提出了新流程：先生成一个MoMap，然后相应地扭曲图像，并完成扭曲的点基渲染。实验结果表明，我们的方法能够生成合理且语义一致的3D场景运动。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从真实世界视频中学习语义上和功能上有意义的3D运动先验知识，以便从单个输入图像预测未来3D场景运动。这个问题在计算机视觉领域非常重要，因为理解、重建和预测物体在3D空间中的运动对于增强现实、自动驾驶和机器人等与物理环境交互的应用至关重要。现有方法要么局限于2D视频生成，要么只能处理小规模的短轨迹，缺乏大规模学习3D生成运动先验的有效方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到需要一种适合神经网络处理的3D场景运动表示方式，避免2D轨迹中常见的遮挡问题。他们受到近期重用图像扩散模型进行其他任务（如深度预测）的启发，提出像素对齐的Motion Map表示。设计过程中借鉴了多个现有工作：利用4D重建技术（MoSca）处理真实视频，采用视频深度估计（DepthCrafter）和3D点跟踪（SpaTracker）技术，并应用视频对象分割（DEVA）获取语义信息。核心创新在于巧妙地将强大的预训练图像扩散模型（如Stable Diffusion）重用于3D运动预测任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将3D场景运动表示为像素对齐的Motion Map（MoMap），这种'轨迹图像'保留了图像结构但编码了3D位置信息，使能利用大型预训练图像扩散模型进行运动预测。整体流程分为四步：1)数据准备：从真实视频中提取深度、跟踪3D点、优化轨迹并生成MoMap；2)MoMap压缩：将高维运动数据编码为紧凑的潜在特征；3)MoMap扩散：修改预训练U-Net生成未来运动；4)应用：通过渲染和图像完成生成视频，或使用视觉语言模型进行精细控制。这种方法解耦了相机和物体运动，减少了问题复杂度。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出MoMap这一像素对齐的3D运动表示；2)构建了从5万+真实视频提取的大规模MoMap数据库；3)重用预训练图像扩散模型进行3D运动预测；4)提出先生成MoMap再完成视频的新范式；5)引入DSL语言实现VLM对运动的精细控制。相比之前工作，不同之处在于：专注于长期密集的3D运动而非短轨迹；直接学习3D轨迹而非通过像素变化隐式学习运动；使用真实世界大规模数据而非合成数据；利用预训练模型而非从头训练；实现了相机与物体运动的解耦。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种名为MoMap的像素对齐3D运动表示方法，利用大型预训练图像扩散模型从真实世界视频中学习语义上有意义的3D运动先验，实现了从单张输入图像预测未来3D场景运动，并展示了其在视频生成等应用中的潜力。'}


### 论文摘要

This paper addresses the challenge of learning semantically and functionally meaningful 3D motion priors from real-world videos, in order to enable prediction of future 3D scene motion from a single input image. We propose a novel pixel-aligned Motion Map (MoMap) representation for 3D scene motion, which can be generated from existing generative image models to facilitate efficient and effective motion prediction. To learn meaningful distributions over motion, we create a large-scale database of MoMaps from over 50,000 real videos and train a diffusion model on these representations. Our motion generation not only synthesizes trajectories in 3D but also suggests a new pipeline for 2D video synthesis: first generate a MoMap, then warp an image accordingly and complete the warped point-based renderings. Experimental results demonstrate that our approach generates plausible and semantically consistent 3D scene motion.

---

## 133. Seeing My Future: Predicting Situated Interaction Behavior in Virtual Reality

**论文链接:** [http://arxiv.org/abs/2510.10742v1](http://arxiv.org/abs/2510.10742v1)

**作者:** Yuan Xu, Zimu Zhang, Xiaoxuan Ma, Wentao Zhu, Yu Qiao, Yizhou Wang

**发布时间:** 2025-10-12

**备注:** Project Page: https://xy02-05.github.io/Seeing_My_Future/

### GPT解析

### 总结

该研究提出了一种分层的、意图感知框架，用于在虚拟和增强现实系统中预测人类行为，通过理解人类意图和利用认知机制，实现了对未来情境行为的准确预测，并在实验中取得了优越性能。

### 背景

虚拟和增强现实系统需要智能适应用户行为以增强交互体验。准确理解人类意图并预测未来的情境行为（如视线方向和物体交互）对创建响应式VR/AR环境和应用至关重要。

### 目的

开发一个能够对人类意图建模并预测详细情境行为的框架，以实现VR/AR系统对用户行为的智能适应，从而创建更响应式的交互环境。

### 方法

引入了一个分层的、意图感知框架，利用认知机制对人类意图建模并预测情境行为。该框架基于历史人类动态和场景上下文观察，识别潜在交互目标并预测细粒度未来行为。采用动态图卷积网络(GCN)来捕捉人-环境关系。

### 主要发现

在具有挑战性的真实世界基准测试和实时VR环境上的实验表明，该方法在所有指标上都取得了优越性能，能够实现主动VR系统的实际应用，这些系统能够预测用户行为并相应调整虚拟环境。

### 结论

所提出的框架有效解决了VR/AR系统中智能适应的关键挑战，通过理解人类意图和预测未来行为，使系统能够主动调整以提供更好的用户体验，为未来VR/AR应用的发展提供了新的可能性。

### 翻译

虚拟和增强现实系统日益需要智能适应用户行为以增强交互体验。实现这一点需要准确理解人类意图并预测未来的情境行为——如视线方向和物体交互——这对于创建响应式的VR/AR环境和个性化助手等应用至关重要。然而，准确的行为预测需要对驱动人-环境交互的潜在认知过程进行建模。在本工作中，我们引入了一个分层的、意图感知框架，通过利用认知机制对人类意图建模并预测详细的情境行为。给定历史人类动态和场景上下文观察，我们的框架首先识别潜在的交互目标并预测细粒度的未来行为。我们提出了一种动态图卷积网络(GCN)来有效捕捉人-环境关系。在具有挑战性的真实世界基准测试和实时VR环境上的大量实验证明了我们方法的有效性，在所有指标上均取得了优越性能，并实现了主动VR系统的实际应用，这些系统能够预测用户行为并相应地调整虚拟环境。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在虚拟现实(VR)环境中准确预测用户的情境化交互行为问题，包括用户视线方向、移动轨迹和物体交互。这个问题很重要，因为VR/AR系统需要智能适应用户行为以增强交互体验，准确预测用户行为能创建响应式环境，支持个性化助手、游戏环境调整和人机协作等应用，使虚拟环境能主动适应而非被动响应人类行为。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从认知科学研究中获取灵感，注意到人类通常会先形成交互意图再执行具体动作，视线在交互意图形成中起关键作用。方法设计借鉴了现有工作：1)视线-身体相关性研究，利用视线信息提高预测准确性；2)视线预测技术，同时预测视线、轨迹和物体交互；3)物体交互预测方法，但创新性地采用符合人类认知的分层框架，先预测潜在目标再预测详细行为。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出一个分层、意图感知框架，模拟人类认知过程，先预测潜在交互目标再预测详细行为，使用动态图卷积网络捕获人-环境关系。整体流程：1)观察编码模块：将历史人类状态和场景上下文编码为时空特征；2)分层意图感知解码模块：先预测潜在交互目标的交互概率，再解码人类和物体的下一个状态；3)动态GCN模块：使用自适应权重矩阵建模视线、人体特征与物体间关系；4)多任务训练：使用多个损失函数监督所有预测输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)分层意图感知框架，首次模拟人类认知过程预测交互行为；2)动态GCN设计，通过自适应权重矩阵有效捕获人-环境关系；3)多任务预测，同时预测视线、轨迹和物体交互。相比之前工作的不同：现有方法如Pose2Gaze缺乏环境上下文，SIF3D等虽利用环境信息但缺乏人-环境关系建模，本文方法在所有指标上表现优越，特别是在识别下一个活跃物体方面有显著优势，且在真实VR环境中验证了实用性和鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种受认知科学启发的分层意图感知框架，通过动态图卷积网络建模人-环境关系，实现了在VR环境中对用户视线、移动轨迹和物体交互的准确预测，为主动式VR系统提供了新的技术基础。'}


### 论文摘要

Virtual and augmented reality systems increasingly demand intelligent adaptation to user behaviors for enhanced interaction experiences. Achieving this requires accurately understanding human intentions and predicting future situated behaviors - such as gaze direction and object interactions - which is vital for creating responsive VR/AR environments and applications like personalized assistants. However, accurate behavioral prediction demands modeling the underlying cognitive processes that drive human-environment interactions. In this work, we introduce a hierarchical, intention-aware framework that models human intentions and predicts detailed situated behaviors by leveraging cognitive mechanisms. Given historical human dynamics and the observation of scene contexts, our framework first identifies potential interaction targets and forecasts fine-grained future behaviors. We propose a dynamic Graph Convolutional Network (GCN) to effectively capture human-environment relationships. Extensive experiments on challenging real-world benchmarks and live VR environment demonstrate the effectiveness of our approach, achieving superior performance across all metrics and enabling practical applications for proactive VR systems that anticipate user behaviors and adapt virtual environments accordingly.

---

## 134. Data-driven simulator of multi-animal behavior with unknown dynamics via offline and online reinforcement learning

**论文链接:** [http://arxiv.org/abs/2510.10451v1](http://arxiv.org/abs/2510.10451v1)

**作者:** Keisuke Fujii, Kazushi Tsutsui, Yu Teshima, Makoto Itoh, Naoya Takeishi, Nozomi Nishiumi, Ryoya Tanaka, Shunsuke Shigaki, Yoshinobu Kawahara

**发布时间:** 2025-10-12

**备注:** 21 pages, 7 figures

### GPT解析

### 总结

本研究提出了一种基于深度强化学习和反事实模拟的数据驱动多动物行为模拟器，解决了在生物学中实现真实多动物模拟的关键挑战，能够高保真地复现物种特异性行为，支持反事实行为预测和多个体建模。

### 背景

动物运动模拟器在行为研究中扮演重要角色。模仿学习在机器人学中的进步为复制人类和动物运动提供了新的可能性。然而，在生物学中实现真实的多动物模拟面临关键挑战：弥合未知现实世界转换模型与其模拟对应物之间的差距。由于运动动态很少是已知的，仅依靠数学模型是不够的。

### 目的

构建一个既能复现真实轨迹又支持奖励驱动优化的模拟器，解决高自由度运动引起的病态问题，实现物种特异性行为的准确复现，支持反事实行为预测和多个体建模。

### 方法

研究团队基于深度强化学习和反事实模拟构建了数据驱动模拟器。他们通过在强化学习框架中将不完整转换模型的运动变量估计为动作，来解决运动中高自由度引起的病态问题。此外，他们使用基于距离的伪奖励来对齐和比较赛博空间与物理空间之间的状态。

### 主要发现

该方法在人工代理、苍蝇、蝾螈和家蚕上得到了验证，与标准模仿和强化学习方法相比，实现了更高的物种特异性行为可再现性和改进的奖励获取。此外，它支持在新实验环境中的反事实行为预测，并支持多个个体的建模以实现灵活的假设轨迹生成。

### 结论

该数据驱动的多动物行为模拟器能够有效模拟和阐明复杂的多动物行为，为生物学研究提供了强大的工具，具有广泛的应用潜力。

### 翻译

动物运动模拟器在行为研究中扮演着重要角色。模仿学习在机器人学中的进步为复制人类和动物运动提供了新的可能性。在生物学中实现真实多动物模拟的一个关键挑战是弥合未知现实世界转换模型与其模拟对应物之间的差距。由于运动动态很少是已知的，仅依靠数学模型是不够的；构建一个既能复现真实轨迹又支持奖励驱动优化的模拟器仍然是一个开放问题。我们介绍了一种基于深度强化学习和反事实模拟的多动物行为数据驱动模拟器。我们通过在强化学习框架中将不完整转换模型的运动变量估计为动作，解决了运动中高自由度引起的病态问题。我们还使用基于距离的伪奖励来对齐和比较赛博空间与物理空间之间的状态。在人工代理、苍蝇、蝾螈和家蚕上的验证表明，与标准模仿和强化学习方法相比，我们的方法实现了更高的物种特异性行为可再现性和改进的奖励获取。此外，它支持在新实验环境中的反事实行为预测，并支持多个个体的建模以实现灵活的假设轨迹生成，表明其模拟和阐明复杂多动物行为的潜力。


### 论文摘要

Simulators of animal movements play a valuable role in studying behavior. Advances in imitation learning for robotics have expanded possibilities for reproducing human and animal movements. A key challenge for realistic multi-animal simulation in biology is bridging the gap between unknown real-world transition models and their simulated counterparts. Because locomotion dynamics are seldom known, relying solely on mathematical models is insufficient; constructing a simulator that both reproduces real trajectories and supports reward-driven optimization remains an open problem. We introduce a data-driven simulator for multi-animal behavior based on deep reinforcement learning and counterfactual simulation. We address the ill-posed nature of the problem caused by high degrees of freedom in locomotion by estimating movement variables of an incomplete transition model as actions within an RL framework. We also employ a distance-based pseudo-reward to align and compare states between cyber and physical spaces. Validated on artificial agents, flies, newts, and silkmoth, our approach achieves higher reproducibility of species-specific behaviors and improved reward acquisition compared with standard imitation and RL methods. Moreover, it enables counterfactual behavior prediction in novel experimental settings and supports multi-individual modeling for flexible what-if trajectory generation, suggesting its potential to simulate and elucidate complex multi-animal behaviors.

---

## 135. Are Video Models Emerging as Zero-Shot Learners and Reasoners in Medical Imaging?

**论文链接:** [http://arxiv.org/abs/2510.10254v1](http://arxiv.org/abs/2510.10254v1)

**作者:** Yuxiang Lai, Jike Zhong, Ming Li, Yuheng Li, Xiaofeng Yang

**发布时间:** 2025-10-11

### GPT解析

### 总结

研究探讨了大型视觉模型（LVM）在医学影像任务中的零样本学习能力，发现即使没有医学数据训练，该模型也能在器官分割、去噪、超分辨率和运动预测等任务上实现具有竞争力的性能，特别是在放射治疗运动预测中表现优异。

### 背景

最近大型生成模型的发展表明，适当扩展的自回归公式可以在不同领域表现出强大的零样本泛化能力。这一趋势启发研究者探索自回归视频建模原则在医学影像领域的应用潜力。

### 目的

研究旨在验证自回归视频建模原则是否可以直接应用于医学影像任务，评估大型视觉模型在从未接触过医学数据的情况下，在多种医学影像任务上的零样本性能。

### 方法

研究评估了一个大型视觉模型（LVM）在四个代表性医学影像任务上的零样本性能：器官分割、去噪、超分辨率和运动预测。研究使用了来自122名患者的4D CT数据，总计超过1,820个3D CT体积进行评估。

### 主要发现

1. 即使没有领域特定的微调，LVM也能在CT扫描中勾勒出解剖结构；2. LVM在分割、去噪和超分辨率任务上实现了具有竞争力的性能；3. 在放射治疗运动预测中，LVM能够直接从前4D CT扫描的前期阶段预测未来的3D CT阶段；4. 预测结果解剖上一致，能够以真实的时间连贯性捕捉患者特定的呼吸动力学；5. LVM在运动预测任务上超越了专门的DVF-based和生成式基线，达到了最先进的空间精度。

### 结论

这些发现揭示了医学视频建模中零样本能力的出现，突显了通用视频模型作为统一学习者和推理者的潜力，为建立在视频模型上的未来医学基础模型奠定了基础。

### 翻译

最近大型生成模型的进展表明，适当地扩展简单的自回归公式可以在不同领域表现出强大的零样本泛化能力。受这一趋势启发，我们研究是否可以将自回归视频建模原则直接应用于医学影像任务，尽管该模型从未在医学数据上训练过。具体来说，我们在四个代表性任务上评估了一个大型视觉模型（LVM）的零样本性能：器官分割、去噪、超分辨率和运动预测。值得注意的是，即使没有领域特定的微调，LVM也能在CT扫描中勾勒出解剖结构，并在分割、去噪和超分辨率任务上实现具有竞争力的性能。最显著的是，在放射治疗运动预测中，该模型直接从前4D CT扫描的前期阶段预测未来的3D CT阶段，产生解剖上一致的预测，能够以真实的时间连贯性捕捉患者特定的呼吸动力学。我们在122名患者的4D CT数据上评估了LVM，总计超过1,820个3D CT体积。尽管没有预先接触医学数据，该模型在所有任务上都实现了强大的性能，并在运动预测方面超越了专门的DVF-based和生成式基线，达到了最先进的空间精度。这些发现揭示了医学视频建模中零样本能力的出现，并突显了通用视频模型作为统一学习者和推理者的潜力，为建立在视频模型上的未来医学基础模型奠定了基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文探讨大型视频模型能否在未经医疗数据训练的情况下，直接应用于医疗影像任务并表现出色。这个问题很重要，因为传统医疗AI系统需要针对特定任务（如分割、去噪）专门训练，成本高昂且系统碎片化。如果通用视频模型能零样本应用于医疗领域，将大大降低医疗AI开发成本，提高系统通用性和适应性，使医疗AI更可扩展和实用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受大型语言模型展现的跨领域零样本泛化能力启发，思考自回归视频建模原则是否可直接应用于医疗影像。他们选择了大型视觉模型（LVM）作为基础，该模型在大规模自然图像和视频上训练，能通过提示适应不同视觉任务。作者借鉴了LLMs和VLMs的统一框架思想、自回归视频建模方法（如VQGAN和Transformer架构）、医疗分割框架（如nnUNet）和变形矢量场方法，创新性地将这些技术应用于医疗领域，特别是放疗运动预测任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是大型视频模型通过自回归学习时空表示，即使未经医疗数据训练，也能通过零样本学习在医疗影像任务上取得有竞争力表现。整体流程：1) 预处理：用nnUNet分割4D CT序列获取器官掩码；2) CT分词化：用VQGAN将CT图像编码为离散令牌序列；3) 序列建模：使用单向Transformer对CT阶段序列进行自回归建模，预测未来阶段；4) 评估：在分割、去噪、超分辨率和运动预测任务上评估性能，使用IoU、DSC等指标和定性可视化验证结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 首次证明大型视频模型可在无医疗训练数据下直接应用于多种医疗任务；2) 展示单一模型可处理分割、去噪、超分辨率和运动预测等多种任务，无需任务特定重训练；3) 在放疗运动预测上超越专门方法，能准确预测未来CT阶段；4) 整合CT图像和分割掩码提高运动建模准确性。相比之前工作：传统医疗AI需针对每任务设计专门模型，而本文使用通用模型处理多任务；现有医疗基础模型多局限于静态影像，本文专注于动态医疗数据；传统方法依赖预计算的DVF，本文直接从图像序列学习运动模式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次证明大型视频模型可以在未经医疗数据训练的情况下，通过零样本学习直接应用于多种医疗影像任务，特别是在放疗运动预测上超越了专门方法，为构建统一的医疗影像基础模型铺平了道路。'}


### 论文摘要

Recent advances in large generative models have shown that simple autoregressive formulations, when scaled appropriately, can exhibit strong zero-shot generalization across domains. Motivated by this trend, we investigate whether autoregressive video modeling principles can be directly applied to medical imaging tasks, despite the model never being trained on medical data. Specifically, we evaluate a large vision model (LVM) in a zero-shot setting across four representative tasks: organ segmentation, denoising, super-resolution, and motion prediction. Remarkably, even without domain-specific fine-tuning, the LVM can delineate anatomical structures in CT scans and achieve competitive performance on segmentation, denoising, and super-resolution. Most notably, in radiotherapy motion prediction, the model forecasts future 3D CT phases directly from prior phases of a 4D CT scan, producing anatomically consistent predictions that capture patient-specific respiratory dynamics with realistic temporal coherence. We evaluate the LVM on 4D CT data from 122 patients, totaling over 1,820 3D CT volumes. Despite no prior exposure to medical data, the model achieves strong performance across all tasks and surpasses specialized DVF-based and generative baselines in motion prediction, achieving state-of-the-art spatial accuracy. These findings reveal the emergence of zero-shot capabilities in medical video modeling and highlight the potential of general-purpose video models to serve as unified learners and reasoners laying the groundwork for future medical foundation models built on video models.

---

## 136. ExpVid: A Benchmark for Experiment Video Understanding & Reasoning

**论文链接:** [http://arxiv.org/abs/2510.11606v1](http://arxiv.org/abs/2510.11606v1)

**作者:** Yicheng Xu, Yue Wu, Jiashuo Yu, Ziang Yan, Tianxiang Jiang, Yinan He, Qingsong Zhao, Kai Chen, Yu Qiao, Limin Wang, Manabu Okumura, Yi Wang

**发布时间:** 2025-10-13

**备注:** Data & Code: https://github.com/OpenGVLab/ExpVid

### GPT解析

### 总结

研究团队开发了ExpVid基准测试，用于系统评估多模态大语言模型在科学实验视频上的能力，发现现有模型在细粒度感知、状态变化跟踪和科学推理方面存在明显不足。

### 背景

多模态大语言模型有望加速科学发现，但现有基准测试忽视了真实实验室工作的细粒度和长期特性，特别是在湿实验室环境中，导致对其真实能力的理解不足。

### 目的

弥补这一差距，引入ExpVid基准测试，系统性评估MLLMs在科学实验视频上的能力。

### 方法

ExpVid从同行评审的视频出版物中策划，包含三级任务层次结构：细粒度感知、程序理解和科学推理。采用视觉为中心的注释流程，结合自动生成和多学科专家验证，确保任务需要视觉基础。

### 主要发现

评估19个领先MLLMs后发现，它们擅长粗粒度识别，但在区分细粒度细节、跟踪状态变化和将实验程序与科学结果联系方面存在困难。专有模型和开源模型间存在明显性能差距，特别是在高阶推理方面。

### 结论

ExpVid不仅提供了诊断工具，还为开发能够成为科学实验可信伙伴的MLLMs绘制了路线图。

### 翻译

多模态大语言模型（MLLMs）有望通过解释复杂的实验流程来加速科学发现。然而，它们真实的能力未被充分理解，因为现有的基准测试忽视了真实实验室工作的细粒度和长期特性，特别是在湿实验室环境中。为了弥补这一差距，我们引入了ExpVid，这是第一个旨在系统性评估MLLMs在科学实验视频上的基准测试。从同行评审的视频出版物中策划，ExpVid具有一个新的三级任务层次结构，反映科学过程：（1）对工具、材料和动作的细粒度感知；（2）对步骤顺序和完整性的程序理解；（3）将整个实验与其已发表结论联系起来的科学推理。我们的视觉为中心的注释流程，结合自动生成和多学科专家验证，确保任务需要视觉基础。我们在ExpVid上评估了19个领先的MLLMs，发现虽然它们擅长粗粒度识别，但在区分细粒度细节、跟踪随时间变化的状态以及将实验程序与科学结果联系起来方面存在困难。我们的结果揭示了专有模型和开源模型之间的显著性能差距，特别是在高阶推理方面。ExpVid不仅提供了诊断工具，还为开发能够成为科学实验可信伙伴的MLLMs绘制了路线图。


### 论文摘要

Multimodal Large Language Models (MLLMs) hold promise for accelerating scientific discovery by interpreting complex experimental procedures. However, their true capabilities are poorly understood, as existing benchmarks neglect the fine-grained and long-horizon nature of authentic laboratory work, especially in wet-lab settings. To bridge this gap, we introduce ExpVid, the first benchmark designed to systematically evaluate MLLMs on scientific experiment videos. Curated from peer-reviewed video publications, ExpVid features a new three-level task hierarchy that mirrors the scientific process: (1) Fine-grained Perception of tools, materials, and actions; (2) Procedural Understanding of step order and completeness; and (3) Scientific Reasoning that connects the full experiment to its published conclusions. Our vision-centric annotation pipeline, combining automated generation with multi-disciplinary expert validation, ensures that tasks require visual grounding. We evaluate 19 leading MLLMs on ExpVid and find that while they excel at coarse-grained recognition, they struggle with disambiguating fine details, tracking state changes over time, and linking experimental procedures to scientific outcomes. Our results reveal a notable performance gap between proprietary and open-source models, particularly in high-order reasoning. ExpVid not only provides a diagnostic tool but also charts a roadmap for developing MLLMs capable of becoming trustworthy partners in scientific experimentation.

---

## 137. ODI-Bench: Can MLLMs Understand Immersive Omnidirectional Environments?

**论文链接:** [http://arxiv.org/abs/2510.11549v1](http://arxiv.org/abs/2510.11549v1)

**作者:** Liu Yang, Huiyu Duan, Ran Tao, Juntao Cheng, Sijing Wu, Yunhao Li, Jing Liu, Xiongkuo Min, Guangtao Zhai

**发布时间:** 2025-10-13

### GPT解析

### 总结

该研究提出了ODI-Bench基准测试和Omni-CoT方法，用于提升多模态大语言模型对全景图像的理解能力。

### 背景

全景图像提供360度全方位视角，广泛应用于VR、AR和具身智能，但多模态大语言模型对这类沉浸式环境的理解能力尚未充分探索。

### 目的

填补多模态大语言模型在全景图像理解方面的研究空白，提供专门的基准测试和改进方法。

### 方法

创建包含2000张全景图像和4000多个问答对的ODI-Bench基准，测试20个代表性MLLMs；提出Omni-CoT方法，通过思维链推理增强模型理解能力。

### 主要发现

当前MLLMs难以捕捉全景图像提供的沉浸式上下文信息。

### 结论

研究将发布ODI-Bench基准测试和Omni-CoT代码，促进全景图像理解领域的发展。

### 翻译

全景图像(ODIs)提供360x180度的全方位视角，广泛应用于VR、AR和具身智能应用。虽然多模态大语言模型(MLLMs)在传统2D图像和视频理解基准测试上表现出色，但它们对ODIs捕捉的沉浸式环境的理解能力仍 largely未被探索。为解决这一差距，我们首先提出了ODI-Bench，这是一个专为全景图像理解设计的新型综合基准测试。ODI-Bench包含2000张高质量全景图像和4000多个人工标注的问答对，涵盖10个细粒度任务，包括一般层面和空间层面的全景图像理解。我们在封闭式和开放式两种设置下对20个代表性MLLMs进行了广泛测试，包括专有和开源模型。实验结果表明，当前MLLMs仍然难以捕捉全景图像提供的沉浸式上下文。为此，我们进一步引入了Omni-CoT，这是一种无需训练的方法，通过在文本信息和视觉线索之间进行思维链推理，显著增强MLLMs在全景环境中的理解能力。基准测试和代码将在发表后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决评估多模态大语言模型(MLLMs)对全方向图像(ODI)的理解能力问题。ODIs提供360°全景视野，广泛应用于VR、AR和具身智能等领域，但MLLMs对这类沉浸式环境的理解能力尚未被充分探索。这个问题很重要，因为ODIs包含比传统2D图像更丰富的空间信息，需要更高级的空间推理能力，对推进具身智能和交互式多模态系统发展至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有ODI基准的局限性(分辨率低、场景多样性有限、问题领域受限、视角限制)来设计方法。他们结合了自动化管道和人工标注两种QA构建方式，借鉴了传统2D图像理解任务设计了5个通用级任务，同时参考空间理解任务设计了5个空间级任务。作者还借鉴了链式思维(Chain-of-Thought)推理方法，设计了Omni-CoT框架来增强模型性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个全面基准(ODI-Bench)评估MLLMs对全方向图像的理解，并通过Omni-CoT框架提升其理解能力。ODI-Bench包含2000张高质量ODIs和4000+问答对，涵盖10个细粒度任务。Omni-CoT框架包含三步：1)视角引导回答(将ODI投影为六个视角并生成描述)；2)裁剪线索的定位和细化(识别相关图像区域并过滤)；3)回答细化(结合视觉线索重新思考答案)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)全面的ODI-Bench基准，同时评估通用级和空间级理解；2)高分辨率(>8K)和多样化场景(室内+室外)的图像集；3)细粒度任务设计(10个任务涵盖属性识别、计数、方向判断等)；4)Omni-CoT训练增强框架显著提升模型性能。相比之前工作，ODI-Bench分辨率更高、场景更多样、任务更全面、标注质量更高，且采用封闭式和开放式双格式评估。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了ODI-Bench全面基准和Omni-CoT增强框架，显著提升了多模态大语言模型对沉浸式全方向环境的理解能力，为评估和改进此类模型提供了新标准。'}


### 论文摘要

Omnidirectional images (ODIs) provide full 360x180 view which are widely adopted in VR, AR and embodied intelligence applications. While multi-modal large language models (MLLMs) have demonstrated remarkable performance on conventional 2D image and video understanding benchmarks, their ability to comprehend the immersive environments captured by ODIs remains largely unexplored. To address this gap, we first present ODI-Bench, a novel comprehensive benchmark specifically designed for omnidirectional image understanding. ODI-Bench contains 2,000 high-quality omnidirectional images and over 4,000 manually annotated question-answering (QA) pairs across 10 fine-grained tasks, covering both general-level and spatial-level ODI understanding. Extensive experiments are conducted to benchmark 20 representative MLLMs, including proprietary and open-source models, under both close-ended and open-ended settings. Experimental results reveal that current MLLMs still struggle to capture the immersive context provided by ODIs. To this end, we further introduce Omni-CoT, a training-free method which significantly enhances MLLMs' comprehension ability in the omnidirectional environment through chain-of-thought reasoning across both textual information and visual cues. Both the benchmark and the code will be released upon the publication.

---

## 138. video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory

**论文链接:** [http://arxiv.org/abs/2510.11129v1](http://arxiv.org/abs/2510.11129v1)

**作者:** Guangzhi Sun, Yixuan Li, Xiaodong Wu, Yudong Yang, Wei Li, Zejun Ma, Chao Zhang

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了video-SALMONN S，一种流式视听LLM，能够在固定内存预算下以1 FPS和360p分辨率处理3小时长视频，通过测试时训练内存模块和提示依赖内存读取器实现高效处理，并在多个长视频基准测试上超越离线和流式基线。

### 背景

连续、高帧率、高分辨率处理长视频流对未来AI代理至关重要，但当前视频理解LLM难以扩展。

### 目的

开发一种能够处理长时间视频流的模型，解决现有方法中离线方法需要适应帧率、流式方法因合并或丢弃令牌导致信息丢失的问题。

### 方法

提出video-SALMONN S，包含(i)测试时训练(TTT)内存模块，持续更新令牌表示以捕获长程依赖；(ii)提示依赖内存读取器，从固定大小内存中选择性检索上下文相关内容；使用无Hessian共轭梯度过程(TTT_HF)优化TTT模块。

### 主要发现

在长视频基准测试(Video-MME, LVBench, VideoEvalPro)上，video-SALMONN S能够在包含10k帧和1M令牌的多小时视频上保持高质量理解。

### 结论

80亿参数的video-SALMONN S模型在Video-MME长分集上达到74.2%总体得分和67.8%的得分，优于离线和流式基线，证明了处理长视频流的有效性。

### 翻译

连续、高帧率、高分辨率处理长视频流对未来AI代理至关重要，但当前视频理解LLM难以扩展。离线时，固定帧数方法需要流长度适应帧率；流式方法通过合并或丢弃令牌来限制内存，导致信息丢失。我们提出了video-SALMONN S，一种流式视听LLM，据我们所知是首个在固定内存预算下以1 FPS和360p分辨率处理3小时视频的模型。我们的模型引入了(i)测试时训练(TTT)内存模块，通过替换令牌合并持续更新令牌表示以捕获长程依赖；(ii)提示依赖内存读取器，从固定大小内存中选择性检索上下文相关内容。TTT模块使用无Hessian共轭梯度过程(TTT_HF)进行优化，实现高效适应。在长视频基准测试(Video-MME, LVBench, VideoEvalPro)上，video-SALMONN S在包含10k帧和1M令牌的多小时视频上保持高质量理解。我们的80亿参数模型在Video-MME长分集上达到74.2%总体得分和67.8%的得分，优于离线和流式基线。


### 论文摘要

Continuous, high-frame-rate, high-resolution processing of long video streams is critical for future AI agents, yet current video-understanding LLMs struggle to scale. Offline, fixed-frame-number methods require the stream length to adapt frame rates; streaming methods constrain memory by merging or discarding tokens, losing information. We propose video-SALMONN S, a streaming audio-visual LLM that, to our knowledge, is the first to process 3-hour videos at 1 FPS and 360p resolution under a fixed memory budget. Our model introduces (i) a test-time-training (TTT) memory module that continually updates token representations to capture long-range dependencies by replacing token merging, and (ii) a prompt-dependent memory reader that selectively retrieves context-relevant content from fixed-size memory. The TTT module is optimised with a Hessian-free conjugate-gradient procedure (TTT_HF) for efficient adaptation. On long-video benchmarks (Video-MME, LVBench, VideoEvalPro), video-SALMONN S sustains high-quality understanding on multi-hour videos with 10k frames and 1M tokens. Our 8B-parameter model achieves 74.2% overall and 67.8% on the Video-MME long split, outperforming both offline and streaming baselines.

---

## 139. Robust Photoplethysmography Signal Denoising via Mamba Networks

**论文链接:** [http://arxiv.org/abs/2510.11058v1](http://arxiv.org/abs/2510.11058v1)

**作者:** I Chiu, Yu-Tung Liu, Kuan-Chen Wang, Hung-Yu Wei, Yu Tsao

**发布时间:** 2025-10-13

**备注:** 5 pages, 2 figures

### GPT解析

### 总结

该论文提出了一种基于深度学习的PPG去噪框架，通过DPNet网络和创新的损失函数设计，有效解决了PPG信号中的噪声问题，同时保留了重要的生理信息。

### 背景

光电容积描记法(PPG)被广泛应用于可穿戴健康监测，但其可靠性常因噪声和运动伪影而降低，限制了心率(HR)估计等下游应用。

### 目的

提出一种深度学习框架用于PPG去噪，重点是保留生理信息。

### 方法

提出DPNet，一种基于Mamba的去噪主干网络，专为有效的时间建模而设计；采用尺度不变信号失真比(SI-SDR)损失函数提高波形保真度；引入辅助心率预测器(HRP)通过基于心率的监督提供生理一致性。

### 主要发现

在BIDMC数据集上的实验表明，该方法对合成噪声和真实世界运动伪影都具有很强的鲁棒性，优于传统滤波和现有神经模型；能有效恢复PPG信号同时保持心率准确性；证明了SI-SDR损失和心率引导监督的互补作用。

### 结论

该方法在可穿戴健康系统实际部署中具有潜力。

### 翻译

光电容积描记法(PPG)被广泛应用于可穿戴健康监测，但其可靠性常因噪声和运动伪影而降低，限制了心率(HR)估计等下游应用。本文提出了一种深度学习框架用于PPG去噪，重点是保留生理信息。在该框架中，我们提出了DPNet，一种基于Mamba的去噪主干网络，专为有效的时间建模而设计。为进一步增强去噪性能，该框架还采用了尺度不变信号失真比(SI-SDR)损失函数来提高波形保真度，以及一个辅助心率预测器(HRP)，通过基于心率的监督提供生理一致性。在BIDMC数据集上的实验表明，我们的方法对合成噪声和真实世界运动伪影都具有很强的鲁棒性，优于传统滤波和现有神经模型。我们的方法能有效恢复PPG信号同时保持心率准确性，突显了SI-SDR损失和心率引导监督的互补作用。这些结果证明了我们的方法在可穿戴健康系统实际部署中的潜力。


### 论文摘要

Photoplethysmography (PPG) is widely used in wearable health monitoring, but its reliability is often degraded by noise and motion artifacts, limiting downstream applications such as heart rate (HR) estimation. This paper presents a deep learning framework for PPG denoising with an emphasis on preserving physiological information. In this framework, we propose DPNet, a Mamba-based denoising backbone designed for effective temporal modeling. To further enhance denoising performance, the framework also incorporates a scale-invariant signal-to-distortion ratio (SI-SDR) loss to promote waveform fidelity and an auxiliary HR predictor (HRP) that provides physiological consistency through HR-based supervision. Experiments on the BIDMC dataset show that our method achieves strong robustness against both synthetic noise and real-world motion artifacts, outperforming conventional filtering and existing neural models. Our method can effectively restore PPG signals while maintaining HR accuracy, highlighting the complementary roles of SI-SDR loss and HR-guided supervision. These results demonstrate the potential of our approach for practical deployment in wearable healthcare systems.

---

## 140. Mixup Helps Understanding Multimodal Video Better

**论文链接:** [http://arxiv.org/abs/2510.10986v1](http://arxiv.org/abs/2510.10986v1)

**作者:** Xiaoyu Ma, Ding Ding, Hao Chen

**发布时间:** 2025-10-13

### GPT解析

### 总结

该研究提出了一种改进的多模态视频理解方法，通过动态调整模态混合比例来解决强模态过拟合问题，提高模型的泛化能力和多模态鲁棒性。

### 背景

多模态视频理解在动作识别和情感分类等任务中至关重要，通过结合不同模态的信息。然而，多模态模型容易对强模态过拟合，导致强模态主导学习并抑制弱模态的贡献。

### 目的

解决多模态模型中强模态过拟合问题，提高模型的泛化能力和多模态鲁棒性，同时考虑模态不平衡问题。

### 方法

首先提出Multimodal Mixup (MM)，在聚合的多模态特征级别应用Mixup策略生成虚拟特征-标签对以减轻过拟合；然后进一步提出Balanced Multimodal Mixup (B-MM)，根据各模态对学习目标的相对贡献动态调整每个模态的混合比例。

### 主要发现

在多个数据集上的广泛实验表明，所提出的方法能有效提高模型的泛化能力和多模态鲁棒性，解决了强模态过拟合和模态不平衡问题。

### 结论

通过动态调整模态混合比例，B-MM方法能够有效平衡不同模态的贡献，减轻过拟合，提高多模态视频理解模型的性能和鲁棒性。

### 翻译

多模态视频理解通过结合不同模态的信息，在动作识别和情感分类等任务中发挥着关键作用。然而，多模态模型容易对强模态过拟合，这些强模态会主导学习并抑制弱模态的贡献。为了应对这一挑战，我们首先提出多模态混合(MM)，在聚合的多模态特征级别应用混合策略，通过生成虚拟特征-标签对来减轻过拟合。虽然MM有效提高了泛化能力，但它对所有模态一视同仁，没有考虑训练过程中的模态不平衡问题。基于MM，我们进一步引入平衡多模态混合(B-MM)，根据各模态对学习目标的相对贡献动态调整每个模态的混合比例。在多个数据集上的广泛实验证明了我们的方法在提高泛化能力和多模态鲁棒性方面的有效性。


### 论文摘要

Multimodal video understanding plays a crucial role in tasks such as action recognition and emotion classification by combining information from different modalities. However, multimodal models are prone to overfitting strong modalities, which can dominate learning and suppress the contributions of weaker ones. To address this challenge, we first propose Multimodal Mixup (MM), which applies the Mixup strategy at the aggregated multimodal feature level to mitigate overfitting by generating virtual feature-label pairs. While MM effectively improves generalization, it treats all modalities uniformly and does not account for modality imbalance during training. Building on MM, we further introduce Balanced Multimodal Mixup (B-MM), which dynamically adjusts the mixing ratios for each modality based on their relative contributions to the learning objective. Extensive experiments on several datasets demonstrate the effectiveness of our methods in improving generalization and multimodal robustness.

---

## 141. Video-STR: Reinforcing MLLMs in Video Spatio-Temporal Reasoning with Relation Graph

**论文链接:** [http://arxiv.org/abs/2510.10976v1](http://arxiv.org/abs/2510.10976v1)

**作者:** Wentao Wang, Heqing Zou, Tianze Luo, Rui Huang, Yutian Zhao, Zhuochen Wang, Hansheng Zhang, Chengwei Qin, Yan Wang, Lin Zhao, Huaijian Zhang

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出Video-STR，一种基于图的强化学习方法，用于解决多模态大语言模型在精确时空理解方面的不足，通过引入GRPO推理机制和构建STV-205k数据集，在各种基准测试上取得最先进结果，比基础模型提高13%。

### 背景

多模态大语言模型在语义理解方面表现出色，但在精确时空理解方面存在困难。现有时空方法主要关注视频本身，忽略了视频中的物理信息（如多物体布局和运动），限制了MLLM在具身智能和VR等需要高精度的下游应用中的使用。

### 目的

解决多模态大语言模型在精确时空理解方面的不足，开发一种能够进行精确视频时空推理的方法。

### 方法

基于可验证奖励的强化学习(RLVR)提高模型能力，引入基于图的组相对策略优化(GRPO)推理机制，指导模型在思考过程中推断场景的潜在时空拓扑结构，并构建包含205k个问答对的STV-205k数据集，涵盖室内和室外环境中的动态多物体场景。

### 主要发现

Video-STR在各种基准测试上取得了最先进的结果，在STI-Bench上比基础模型提高了13%，证明了该方法和数据集的有效性。

### 结论

Video-STR成功解决了多模态大语言模型在精确时空理解方面的不足，代码、模型和数据将公开发布。

### 翻译

多模态大语言模型(MLLMs)的最新进展展示了强大的语义理解能力，但在执行精确时空理解方面存在困难。现有的时空方法主要关注视频本身，而忽略了视频中的物理信息，如多物体布局和运动。这些限制限制了MLLM在需要高精度的下游应用中的使用，包括具身智能和VR。为解决这个问题，我们提出了Video-STR，一种基于图的强化学习方法，用于精确的视频时空推理。基于可验证奖励的强化学习(RLVR)提高模型能力的能力，我们引入了一种使用基于图的组相对策略优化(GRPO)方法的推理机制，指导模型在思考过程中推断场景的潜在时空拓扑结构。为解决时空训练数据的缺乏，我们构建了包含205k个问答对的STV-205k数据集，涵盖室内和室外环境中的动态多物体场景，以支持模型训练。实验表明，Video-STR在各种基准测试上取得了最先进的结果，在STI-Bench上比基础模型提高了13%，证明了我们方法和数据集的有效性。代码、模型和数据将发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态大语言模型（MLLMs）在精确时空理解方面的不足。现有模型虽然语义理解能力强，但在理解视频中的物体位置、布局、运动轨迹等物理信息方面表现不佳。这个问题很重要，因为它限制了MLLMs在需要高精度的下游应用（如具身智能和VR）中的使用，而这些应用对物体间的空间关系和时间动态有严格要求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有时空方法的局限性，发现它们主要关注视频本身而忽视物理信息，或者使用像素级定位和2D认知图等方法，这些方法无法准确推断物体在物理空间中的布局和分布。基于这些分析，作者设计了一个基于图的表示方法，将物体建模为节点，物体间关系建模为边，这种方法具有旋转不变性且更鲁棒。作者借鉴了强化学习与可验证奖励（RLVR）框架和Group Relative Policy Optimization（GRPO）方法，但扩展了它们以适应视频时空推理任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用基于图的表示来建模多物体场景的拓扑结构，并通过强化学习框架训练模型理解时空关系。整体流程包括：1）从TAO、ScanNet和KITTI收集数据；2）构建STV-205k数据集（205k问答对）；3）设计多种可验证奖励函数（格式、多选、数值、IoU奖励）；4）引入图推理机制帮助模型理解空间拓扑；5）使用扩展的GRPO算法进行训练；6）在多个基准测试上评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）构建STV-205k数据集，解决视频时空训练数据稀缺问题；2）首次使用物体间关系图表征多物体场景，扩展GRPO引入图推理机制；3）设计特定奖励函数和图推理机制监督模型理解时空信息。相比之前工作，本文强调视频嵌入的物理信息而非仅关注视频本身；使用图结构而非像素定位或2D认知图表示场景，具有旋转不变性；通过特定奖励函数更有效监督模型对时空信息的理解。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Video-STR通过引入基于图的时空推理机制和STV-205k数据集，显著提升了多模态大语言模型在视频时空推理任务上的性能，实现了现有方法的最佳效果。'}


### 论文摘要

Recent progress in Multimodal Large Language Models (MLLMs) has demonstrated strong semantic understanding capabilities, but struggles to perform precise spatio-temporal understanding. Existing spatio-temporal methods primarily focus on the video itself, while overlooking the physical information within the video, such as multi-object layouts and motion. Such limitations restrict the use of MLLMs in downstream applications that demand high precision, including embodied intelligence and VR. To address this issue, we present Video-STR, a novel graph-based reinforcement method for precise Video Spatio-Temporal Reasoning. Building upon the capacity of Reinforcement Learning with Verifiable Reward (RLVR) to improve model abilities, we introduce a reasoning mechanism using graph-based Group Relative Policy Optimization (GRPO) method to guide the model in inferring the underlying spatio-temporal topology of scenarios during the thinking process. To resolve the lack of spatio-temporal training data, we construct the STV-205k dataset with 205k question-answering pairs, covering dynamic multi-object scenes in both indoor and outdoor environments, to support the model training. Experiments show that Video-STR achieves state-of-the-art results on various benchmarks, outperforming the base model by 13% on STI-Bench, and demonstrating the effectiveness of our approach and dataset. Code, model, and data will be released.

---

## 142. OmniVideoBench: Towards Audio-Visual Understanding Evaluation for Omni MLLMs

**论文链接:** [http://arxiv.org/abs/2510.10689v1](http://arxiv.org/abs/2510.10689v1)

**作者:** Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Jiafu Tang, Zhenghao Song, Dingling Zhang, Ying He, Haoxiang Liu, Yuxuan Wang, Qiufeng Wang, Zhenhe Wu, Jiehui Luo, Zhiyu Pan, Weihao Xie, Chenchen Zhang, Zhaohui Wang, Jiayi Tian, Yanghai Wang, Zhe Cao, Minxin Dai, Ke Wang, Runzhe Wen, Yinghao Ma, Yaning Pan, Sungkyun Chang, Termeh Taheri, Haiwen Xia, Christos Plachouras, Emmanouil Benetos, Yizhi Li, Ge Zhang, Jian Yang, Tianhao Peng, Zili Wang, Minghao Liu, Junran Peng, Zhaoxiang Zhang, Jiaheng Liu

**发布时间:** 2025-10-12

### GPT解析

### 总结

该论文介绍了OmniVideoBench，一个专门用于评估多模态大语言模型协同视听理解能力的大规模基准测试。该基准测试包含1000个高质量问答对，覆盖13种问题类型，评估结果显示当前模型与人类推理能力存在明显差距，开源模型表现尤其不佳。

### 背景

多模态大语言模型在视频理解方面显示出巨大潜力，但现有基准测试无法全面评估跨音频和视觉模态的协同推理能力，往往忽略其中一个模态或以逻辑不一致的方式整合它们。

### 目的

弥补现有基准测试的不足，引入一个专门用于评估协同视听理解能力的大规模且精心设计的基准测试，强调模态互补性和逻辑一致性。

### 方法

构建OmniVideoBench基准测试，包含1000个高质量问答对，每个标注有逐步推理轨迹；数据来源于628个多样化视频，时长从几秒到30分钟不等；包含13种精心设计的问题类型，涵盖时间推理、空间定位、计数、因果推断和总结等；所有数据经过人工验证确保正确性和唯一性。

### 主要发现

在OmniVideoBench上对多个多模态大语言模型的评估显示，模型性能与人类推理之间存在明显差距；开源模型显著落后于闭源模型，这突显了真实视听推理的内在难度。

### 结论

将发布OmniVideoBench基准测试以促进具有更强和更可泛化推理能力的多模态大语言模型的发展。

### 翻译

最近多模态大语言模型的进展在视频理解方面展示了巨大潜力。然而，现有基准测试无法全面评估跨音频和视觉模态的协同推理能力，常常忽略其中一个模态或以逻辑不一致的方式整合它们。为了弥补这一差距，我们引入了OmniVideoBench，一个大规模且精心设计的基准测试，专门用于评估协同视听理解，特别强调模态互补性和逻辑一致性。具体来说，OmniVideoBench包含1000个高质量的问答对，每个都标注了逐步推理轨迹，来源于628个从几秒到30分钟不等的多样化视频，并经过人工验证以确保完全正确性和唯一性。此外，OmniVideoBench包含13种精心设计的问题类型，涵盖时间推理、空间定位、计数、因果推断、总结等，从而捕捉视频理解的基本挑战。在OmniVideoBench上对多个多模态大语言模型的评估揭示了模型性能与人类推理之间的明显差距，其中开源模型显著落后于闭源模型，这突显了真实视听推理的内在难度。我们将发布OmniVideoBench以促进具有更强和更可泛化推理能力的多模态大语言模型的发展。


### 论文摘要

Recent advances in multimodal large language models (MLLMs) have demonstrated substantial potential in video understanding. However, existing benchmarks fail to comprehensively evaluate synergistic reasoning capabilities across audio and visual modalities, often neglecting either one of the modalities or integrating them in a logically inconsistent manner. To bridge this gap, we introduce OmniVideoBench, a large-scale and rigorously designed benchmark dedicated to assessing synergistic audio-visual understanding, with a strong emphasis on modality complementarity and logical consistency. Specifically, OmniVideoBench comprises 1000 high-quality question-answer(QA) pairs, each annotated with step-by-step reasoning traces, derived from 628 diverse videos ranging from several seconds to 30 minutes, and manually verified to guarantee complete correctness and uniqueness. Moreover, OmniVideoBench encompasses 13 carefully designed question types, covering temporal reasoning, spatial localization, counting, causal inference, summarization, and beyond, thereby capturing the essential challenges of video understanding. Evaluation of multiple MLLMs on OmniVideoBench reveals a pronounced gap between model performance and human reasoning, with open-source models lagging significantly behind their closed-source counterparts, underscoring the inherent difficulty of genuine audio-visual reasoning. We will release OmniVideoBench to foster the development of MLLMs with stronger and more generalizable reasoning capabilities.

---

## 143. Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction

**论文链接:** [http://arxiv.org/abs/2510.10454v1](http://arxiv.org/abs/2510.10454v1)

**作者:** Sihang Zeng, Yujuan Fu, Sitong Zhou, Zixuan Yu, Lucas Jing Liu, Jun Wen, Matthew Thompson, Ruth Etzioni, Meliha Yetisgen

**发布时间:** 2025-10-12

**备注:** Accepted by NeurIPS 2025 GenAI4Health Workshop

### GPT解析

### 总结

Traj-CoA是一个多智能体系统，通过智能体链处理电子健康记录数据，减少噪声并保留完整时间线，在患者轨迹建模任务中表现优异。

### 背景

大型语言模型为患者轨迹建模提供了通用方法，但电子健康记录数据的时间推理存在数据冗长和嘈杂的问题。

### 目的

解决EHR数据在时间推理中的长序列和噪声问题，提高患者轨迹建模的准确性。

### 方法

提出Traj-CoA多智能体系统，使用工作智能体顺序处理EHR数据，将关键事件提炼到EHRMem记忆模块中，最后由管理智能体进行综合预测。

### 主要发现

在基于五年EHR数据的一年肺癌风险预测的零样本任务中，Traj-CoA优于四类基线方法，展现出与临床实践一致的时间推理能力。

### 结论

Traj-CoA是一种有前途的、鲁棒且通用的方法，用于建模复杂患者轨迹。

### 翻译

大型语言模型为患者轨迹建模提供了一种通用方法，但受电子健康记录数据在时间推理中冗长和嘈杂的特性所困扰。为应对这些挑战，我们引入了Traj-CoA，一个涉及智能体链的多智能体系统，用于患者轨迹建模。Traj-CoA采用一系列工作智能体顺序处理可管理的EHR数据块，将关键事件提炼到共享的长期记忆模块EHRMem中，以减少噪声并保留完整时间线。最终管理智能体综合工作智能体的摘要和EHRMem中提取的时间线进行预测。在基于五年EHR数据的一年肺癌风险预测的零样本任务中，Traj-CoA优于四类基线方法。分析表明，Traj-CoA展现出与临床实践一致的时间推理能力，使其成为建模复杂患者轨迹的一种前景广阔的鲁棒且通用的方法。


### 论文摘要

Large language models (LLMs) offer a generalizable approach for modeling patient trajectories, but suffer from the long and noisy nature of electronic health records (EHR) data in temporal reasoning. To address these challenges, we introduce Traj-CoA, a multi-agent system involving chain-of-agents for patient trajectory modeling. Traj-CoA employs a chain of worker agents to process EHR data in manageable chunks sequentially, distilling critical events into a shared long-term memory module, EHRMem, to reduce noise and preserve a comprehensive timeline. A final manager agent synthesizes the worker agents' summary and the extracted timeline in EHRMem to make predictions. In a zero-shot one-year lung cancer risk prediction task based on five-year EHR data, Traj-CoA outperforms baselines of four categories. Analysis reveals that Traj-CoA exhibits clinically aligned temporal reasoning, establishing it as a promisingly robust and generalizable approach for modeling complex patient trajectories.

---

## 144. AVoCaDO: An Audiovisual Video Captioner Driven by Temporal Orchestration

**论文链接:** [http://arxiv.org/abs/2510.10395v1](http://arxiv.org/abs/2510.10395v1)

**作者:** Xinlong Chen, Yue Ding, Weihong Lin, Jingyun Hua, Linli Yao, Yang Shi, Bozhou Li, Yuanxing Zhang, Qiang Liu, Pengfei Wan, Liang Wang, Tieniu Tan

**发布时间:** 2025-10-12

**备注:** Project webpage: https://avocado-captioner.github.io/

### GPT解析

### 总结

本文提出了AVoCaDO，一个由音频和视觉模态时间编排驱动的强大音视频视频字幕生成器，通过两阶段微调管道显著提升了字幕生成质量。

### 背景

音视频视频字幕生成旨在生成语义丰富的描述，并使视觉和听觉事件之间保持时间对齐，从而有助于视频理解和生成。

### 目的

开发一个强大的音视频视频字幕生成器，通过音频和视觉模态之间的时间编排来提高字幕生成的质量。

### 方法

提出了一个两阶段的微调管道：(1) AVoCaDO SFT，在一个新整理的包含107K高质量、时间对齐的音视频字幕的数据集上微调模型；(2) AVoCaDO GRPO，利用定制化的奖励函数来进一步增强时间连贯性和对话准确性，同时规范字幕长度并减少崩溃。

### 主要发现

1. AVoCaDO在四个音视频视频字幕生成基准测试中显著优于现有的开源模型。2. AVoCaDO在仅视觉设置下的VDC和DREAM-1K基准测试中实现了具有竞争力的性能。

### 结论

AVoCaDO通过两阶段微调管道有效地提高了音视频视频字幕生成的质量，不仅在音视频设置下表现出色，而且在仅视觉设置下也能保持竞争力。

### 翻译

音视频视频字幕生成旨在生成语义丰富的描述，并使视觉和听觉事件之间保持时间对齐，从而有助于视频理解和生成。在本文中，我们提出了AVoCaDO，一个由音频和视觉模态之间时间编排驱动的强大音视频视频字幕生成器。我们提出了一个两阶段的微调管道：(1) AVoCaDO SFT，在一个新整理的包含107K高质量、时间对齐的音视频字幕的数据集上微调模型；(2) AVoCaDO GRPO，利用定制化的奖励函数来进一步增强时间连贯性和对话准确性，同时规范字幕长度并减少崩溃。实验结果表明，AVoCaDO在四个音视频视频字幕生成基准测试中显著优于现有的开源模型，并且在仅视觉设置下的VDC和DREAM-1K基准测试中也实现了具有竞争力的性能。


### 论文摘要

Audiovisual video captioning aims to generate semantically rich descriptions with temporal alignment between visual and auditory events, thereby benefiting both video understanding and generation. In this paper, we present AVoCaDO, a powerful audiovisual video captioner driven by the temporal orchestration between audio and visual modalities. We propose a two-stage post-training pipeline: (1) AVoCaDO SFT, which fine-tunes the model on a newly curated dataset of 107K high-quality, temporally-aligned audiovisual captions; and (2) AVoCaDO GRPO, which leverages tailored reward functions to further enhance temporal coherence and dialogue accuracy while regularizing caption length and reducing collapse. Experimental results demonstrate that AVoCaDO significantly outperforms existing open-source models across four audiovisual video captioning benchmarks, and also achieves competitive performance on the VDC and DREAM-1K benchmark under visual-only settings.

---

## 145. MomentSeg: Moment-Centric Sampling for Enhanced Video Pixel Understanding

**论文链接:** [http://arxiv.org/abs/2510.09274v1](http://arxiv.org/abs/2510.09274v1)

**作者:** Ming Dai, Sen Yang, Boqiang Duan, Wankou Yang, Jingdong Wang

**发布时间:** 2025-10-10

### GPT解析

### 总结

该研究提出了一种统一框架，用于联合优化时间句子定位和视频对象分割，解决了现有方法中忽视时间线索或增加系统复杂性的问题。

### 背景

Referring Video Object Segmentation (RefVOS) 需要根据自然语言描述在视频中分割目标对象，这需要时间推理和细粒度的视觉理解能力。现有的基于LLM的采样策略通常依赖手工设计的启发式方法或外部关键帧模型，前者忽视重要时间线索，后者增加系统复杂性。

### 目的

提出一个统一框架，联合优化时间句子定位（TSG）和RefVOS，自然融合关键时刻定位能力。

### 方法

1) 训练阶段：引入新的TSG范式，使用[FIND]标记通过时间标记相似度匹配识别关键时刻，避免外部时间戳编码；2) 推理阶段：设计以时刻为中心的采样(MCS)策略，密集采样信息丰富时刻，稀疏采样非必要帧；3) 开发双向锚点更新传播(BAP)，利用最相关时刻初始化高质量掩码，动态更新减轻累积误差。

### 主要发现

通过联合优化TSG和RefVOS，以及创新的采样策略和传播方法，能够有效保留运动细节和全局上下文，同时提高跟踪稳定性。

### 结论

该框架解决了现有采样策略的局限性，通过自然集成关键时刻定位能力，实现了更高效的视频对象分割。

### 翻译

该论文提出了一种引用视频对象分割方法，通过自然语言描述引导视频中的目标对象分割，同时需要时间推理和细粒度视觉理解能力。基于LLM的现有采样策略通常依赖手工设计启发式方法或外部关键帧模型。前者常常忽视重要时间线索，后者增加系统复杂性。为此，我们提出统一框架，联合优化时间句子定位和引用视频对象分割，自然融入关键时刻定位能力。训练阶段，我们引入新型TSG范式，使用专用[FIND]标记通过时间标记相似度匹配识别关键时刻，避免需要外部时间戳编码。推理阶段，我们设计以时刻为中心的采样策略，密集采样信息丰富时刻，同时稀疏采样非必要帧，保留运动细节和全局上下文。为进一步增强跟踪稳定性，我们开发双向锚点更新传播，利用最相关时刻作为高质量掩码初始化起点，并在采样点动态更新以减轻累积误差。代码和模型将发布于：https://github.com/Dmmm1997/MomentSeg


### 论文摘要

Referring Video Object Segmentation (RefVOS) seeks to segment target objects in videos guided by natural language descriptions, demanding both temporal reasoning and fine-grained visual comprehension. Existing sampling strategies for LLM-based approaches typically rely on either handcrafted heuristics or external keyframe models. The former often overlooks essential temporal cues, while the latter increases system complexity. To address this, we propose a unified framework that jointly optimizes Temporal Sentence Grounding (TSG) and RefVOS, naturally incorporating key moment grounding capability. During training, we introduce a novel TSG paradigm that employs a dedicated \texttt{[FIND]} token for key moment identification through temporal token similarity matching, thereby avoiding the need for external timestamp encodings. For inference, we design a Moment-Centric Sampling (MCS) strategy that densely samples informative moments while sparsely sampling non-essential frames, preserving both motion details and global context. To further enhance tracking stability, we develop Bidirectional Anchor-updated Propagation (BAP), which leverages the most relevant moment as start point for high-quality mask initialization and dynamically updates at sampled points to mitigate accumulated errors. Code and model will be available at: https://github.com/Dmmm1997/MomentSeg

---

## 146. Diagnosing Shoulder Disorders Using Multimodal Large Language Models and Consumer-Grade Cameras

**论文链接:** [http://arxiv.org/abs/2510.09230v1](http://arxiv.org/abs/2510.09230v1)

**作者:** Jindong Hong, Wencheng Zhang, Shiqin Qiao, Jianhai Chen, Jianing Qiu, Chuanyang Zheng, Qian Xu, Yun Ji, Qianyue Wen, Weiwei Sun, Hao Li, Huizhen Li, Huichao Wang, Kai Wu, Meng Li, Yijun He, Lingjie Luo, Jiankai Sun

**发布时间:** 2025-10-10

### GPT解析

### 总结

本研究提出了一种基于消费级设备视频的肩部疾病诊断框架HMVDx，利用多模态大语言模型分离动作理解和疾病诊断任务，显著提高了诊断准确率。

### 背景

肩部疾病是全球常见疾病，在老年人和从事重复肩部任务的工作者中发病率高。在医疗资源稀缺地区，早期准确诊断面临挑战，需要低成本且易于扩展的辅助诊断方案。

### 目的

引入消费级设备拍摄的视频作为诊断基础降低成本，研究多模态大语言模型在肩部疾病初步诊断中的应用，提出HMVDx框架。

### 方法

HMVDx框架将动作理解和疾病诊断两个任务分开，由两个多模态大语言模型分别完成。提出'可用性指数'新型指标，基于医疗决策逻辑过程评估多模态大语言模型在医疗领域的有效性。

### 主要发现

HMVDx在诊断肩关节损伤方面的准确率比直接视频诊断提高了79.6%，显示低成本多模态大语言模型在医疗应用中的潜在价值。

### 结论

HMVDx对多模态大语言模型在医学领域视频理解应用的研究具有重大技术贡献。

### 翻译

肩部疾病，如冻结肩（又名粘连性关节囊炎），是影响全球人民健康的常见疾病，在老年人和从事重复肩部任务的工作者中发病率高。在医疗资源稀缺的地区，实现早期准确诊断面临重大挑战，迫切需要低成本且易于扩展的辅助诊断解决方案。本研究引入消费级设备拍摄的视频作为诊断基础，降低用户成本。我们专注于多模态大语言模型在肩部疾病初步诊断中的创新应用，并提出混合运动视频诊断框架。该框架将动作理解和疾病诊断两个任务分开，分别由两个多模态大语言模型完成。除了传统评估指标外，本研究还提出了一种名为'可用性指数'的新型指标，基于医疗决策的逻辑过程（动作识别、运动诊断和最终诊断）。该指数从整个医疗诊断路径的角度评估多模态大语言模型在医疗领域的有效性，揭示了低成本多模态大语言模型在医疗应用中对医疗从业者的潜在价值。在实验比较中，HMVDx在诊断肩关节损伤方面的准确率比直接视频诊断提高了79.6%，这是多模态大语言模型在医学领域视频理解应用研究中的重大技术贡献。


### 论文摘要

Shoulder disorders, such as frozen shoulder (a.k.a., adhesive capsulitis), are common conditions affecting the health of people worldwide, and have a high incidence rate among the elderly and workers engaged in repetitive shoulder tasks. In regions with scarce medical resources, achieving early and accurate diagnosis poses significant challenges, and there is an urgent need for low-cost and easily scalable auxiliary diagnostic solutions. This research introduces videos captured by consumer-grade devices as the basis for diagnosis, reducing the cost for users. We focus on the innovative application of Multimodal Large Language Models (MLLMs) in the preliminary diagnosis of shoulder disorders and propose a Hybrid Motion Video Diagnosis framework (HMVDx). This framework divides the two tasks of action understanding and disease diagnosis, which are respectively completed by two MLLMs. In addition to traditional evaluation indicators, this work proposes a novel metric called Usability Index by the logical process of medical decision-making (action recognition, movement diagnosis, and final diagnosis). This index evaluates the effectiveness of MLLMs in the medical field from the perspective of the entire medical diagnostic pathway, revealing the potential value of low-cost MLLMs in medical applications for medical practitioners. In experimental comparisons, the accuracy of HMVDx in diagnosing shoulder joint injuries has increased by 79.6\% compared with direct video diagnosis, a significant technical contribution to future research on the application of MLLMs for video understanding in the medical field.

---

## 147. Spatio-Temporal Graph Convolutional Networks for EV Charging Demand Forecasting Using Real-World Multi-Modal Data Integration

**论文链接:** [http://arxiv.org/abs/2510.09048v1](http://arxiv.org/abs/2510.09048v1)

**作者:** Jose Tupayachi, Mustafa C. Camur, Kevin Heaslip, Xueping Li

**发布时间:** 2025-10-10

### GPT解析

### 总结

本研究提出TW-GCN框架，结合图卷积网络和时间架构预测电动汽车充电需求，解决了充电设施分布不均和利用不规律对电网稳定性和投资规划的挑战

### 背景

交通运输是温室气体的主要来源，向电动汽车等可持续替代品转型非常紧迫，但充电设施的空间分布不均和利用不规律对电网稳定性和投资规划构成挑战

### 目的

开发TW-GCN框架，结合图卷积网络和时间架构，预测美国田纳西州的电动汽车充电需求

### 方法

利用真实世界的交通流量、天气条件和美国最大的电动汽车基础设施公司提供的专有数据，捕捉空间依赖性和时间动态

### 主要发现

中期（3小时）预测在响应性和稳定性之间取得最佳平衡；1DCNN在时间模型中表现持续优于其他模型；东、中、西田纳西州的预测准确性存在差异，反映了站点密度、人口和当地需求变异性对模型性能的影响

### 结论

TW-GCN框架推动了数据驱动智能与电动汽车基础设施规划的整合，支持可持续交通转型和弹性电网管理

### 翻译

交通仍然是温室气体的主要来源，这凸显了向电动汽车等可持续替代品过渡的紧迫性。然而，充电设施的空间分布不均和使用不规则对电网稳定性和投资规划构成了挑战。本研究引入了TW-GCN，一个结合图卷积网络和时间架构的时空预测框架，用于预测美国田纳西州的电动汽车充电需求。我们利用真实的交通流量、天气条件以及美国最大的电动汽车基础设施公司提供的专有数据，来捕捉空间依赖性和时间动态。在不同滞后时间、聚类策略和序列长度上的广泛实验表明，中期（3小时）预测在响应性和稳定性之间取得了最佳平衡，1DCNN持续优于其他时间模型。区域分析显示东、中、西田纳西州的预测准确性存在差异，反映了站点密度、人口和当地需求变异性如何影响模型性能。所提出的TW-GCN框架推动了数据驱动智能与电动汽车基础设施规划的整合，支持可持续交通转型和弹性电网管理。


### 论文摘要

Transportation remains a major contributor to greenhouse gas emissions, highlighting the urgency of transitioning toward sustainable alternatives such as electric vehicles (EVs). Yet, uneven spatial distribution and irregular utilization of charging infrastructure create challenges for both power grid stability and investment planning. This study introduces TW-GCN, a spatio-temporal forecasting framework that combines Graph Convolutional Networks with temporal architectures to predict EV charging demand in Tennessee, United States (U.S.). We utilize real-world traffic flows, weather conditions, and proprietary data provided by one of the largest EV infrastructure company in the U.S. to capture both spatial dependencies and temporal dynamics. Extensive experiments across varying lag horizons, clustering strategies, and sequence lengths reveal that mid-horizon (3-hour) forecasts achieve the best balance between responsiveness and stability, with 1DCNN consistently outperforming other temporal models. Regional analysis shows disparities in predictive accuracy across East, Middle, and West Tennessee, reflecting how station density, population, and local demand variability shape model performance. The proposed TW-GCN framework advances the integration of data-driven intelligence into EV infrastructure planning, supporting both sustainable mobility transitions and resilient grid management.

---

## 148. RO-Bench: Large-scale robustness evaluation of MLLMs with text-driven counterfactual videos

**论文链接:** [http://arxiv.org/abs/2510.08936v1](http://arxiv.org/abs/2510.08936v1)

**作者:** Zixi Yang, Jiapeng Li, Muxi Diao, Yinuo Jing, Kongming Liang

**发布时间:** 2025-10-10

### GPT解析

### 总结

本研究引入了Ro-Bench，首个用于评估多模态大语言模型在动态分布外反事实视频测试集上的基准，发现当前模型在面对被操纵的视频内容时鲁棒性不足，但通过反事实数据微调可显著提升性能。

### 背景

多模态大语言模型在各种视频理解任务中表现出显著性能，但当面对被操纵的视频内容时，它们的鲁棒性在很大程度上尚未被探索。

### 目的

引入Ro-Bench，首个用于评估多模态大语言模型在动态分布外反事实视频测试集上的基准。

### 方法

Ro-Bench通过编辑风格、物体、背景及其组合，整合高质量、多样化的时间相关视频数据；评估了八个最近的视频多模态大语言模型；并通过反事实数据微调多模态大语言模型以增强鲁棒性。

### 主要发现

当前模型在面对反事实视频内容时，在Ro-Bench上表现出显著的性能下降；使用反事实数据微调多模态大语言模型可以增强鲁棒性，在Ro-Bench上实现了21.73%的性能提升，在MVBench数据集的20个任务上实现了12.78%的改进。

### 结论

反事实数据在增强多模态大语言模型的视频理解能力方面具有显著有效性。

### 翻译

最近，多模态大语言模型在各种视频理解任务中展示了显著的性能。然而，它们的鲁棒性，特别是在面对被操纵的视频内容时，在很大程度上仍未被探索。在本文中，我们引入了Ro-Bench，这是首个用于评估多模态大语言模型在动态分布外反事实视频测试集上的基准。Ro-Bench通过编辑风格、物体、背景及其组合，整合了高质量、多样化且时间相关的视频数据。我们评估了八个最近的视频多模态大语言模型，发现当面对反事实视频内容时，当前模型在Ro-Bench上表现出显著的性能下降。此外，我们证明使用反事实数据微调多模态大语言模型可以增强鲁棒性，在Ro-Bench上实现了21.73%的性能提升，在MVBench数据集的20个任务上实现了12.78%的改进。这些发现强调了反事实数据在增强多模态大语言模型视频理解能力方面的有效性。代码和数据将很快发布。


### 论文摘要

Recently, Multi-modal Large Language Models (MLLMs) have demonstrated significant performance across various video understanding tasks. However, their robustness, particularly when faced with manipulated video content, remains largely unexplored. In this paper, we introduce Ro-Bench, the first benchmark for evaluating MLLMs on dynamic out-of-distribution (OOD) counterfactual video test sets. Ro-Bench incorporates high-quality, diverse and temporally relevant video data, by editing Style, Object, Background and their compositions. We evaluated eight recent video MLLMs and found that current models exhibit substantial performance degradation on Ro-Bench when exposed to counterfactual video content. Furthermore, we demonstrate that fine-tuning MLLMs with counterfactual data enhances robustness, achieving a 21.73% performance increase on Ro-Bench and a 12.78% improvement across 20 tasks in the MVBench dataset. These findings underscore the effectiveness of counterfactual data in enhancing the video understanding ability of MLLMs. The code and data will be released shortly.

---

## 149. D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition

**论文链接:** [http://arxiv.org/abs/2510.08818v1](http://arxiv.org/abs/2510.08818v1)

**作者:** Yiyang Huang, Yizhou Wang, Yun Fu

**发布时间:** 2025-10-09

**备注:** This paper has been accepted to EMNLP 2025

### GPT解析

### 总结

论文提出了D-CoDe，一个无需训练的适应框架，用于解决将图像预训练的视觉语言模型扩展到视频领域时面临的感知瓶颈和令牌过载问题。

### 背景

视频大语言模型在多样化的视频语言任务中表现出色，可以通过适应图像预训练的视觉语言模型来有效构建。然而，这种适应具有挑战性，因为需要处理密集且时间上延展的视觉输入，这超出了基于图像模型的处理能力。

### 目的

解决将基于图像的视觉语言模型扩展到视频领域时面临的感知瓶颈和令牌过载这两个关键挑战。

### 方法

提出D-CoDe，一个无需训练的适应框架，结合了动态压缩和问题分解两种技术。动态压缩通过自适应选择代表性帧和空间令牌的内容感知聚合来减轻感知瓶颈；问题分解通过将原始查询重新表述为子问题来缓解令牌过载。

### 主要发现

实验证明D-CoDe在各种基准测试中有效提高了视频理解能力，特别是在具有挑战性的长视频基准测试上表现出色，突显了其处理复杂视频语言任务的潜力。

### 结论

D-CoDe框架能够有效解决视频理解中的感知瓶颈和令牌过载问题，为视频大语言模型的构建提供了新的思路。

### 翻译

视频大语言模型在多样化的视频语言任务中表现出色，可以通过适应图像预训练的视觉语言模型来有效构建。然而，这种适应仍然具有挑战性，因为它需要处理密集且时间上延展的视觉输入，这超出了基于图像模型的处理能力。本文确定了感知瓶颈和令牌过载是将基于图像的视觉语言模型扩展到视频领域时的关键挑战。为了解决这些问题，我们提出了D-CoDe，一个无需训练的适应框架，结合了动态压缩和问题分解。具体而言，动态压缩通过自适应选择代表性帧和空间令牌的内容感知聚合来减轻感知瓶颈，从而减少冗余同时保留信息内容。同时，问题分解通过将原始查询重新表述为子问题来缓解令牌过载，指导模型关注视频的不同方面，实现更全面的理解。实验证明D-CoDe在各种基准测试中有效提高了视频理解能力。此外，在具有挑战性的长视频基准测试上的良好表现突显了D-CoDe处理复杂视频语言任务的潜力。代码可在https://github.com/hukcc/D-CoDe获取。


### 论文摘要

Video large language models (Vid-LLMs), which excel in diverse video-language tasks, can be effectively constructed by adapting image-pretrained vision-language models (VLMs). However, this adaptation remains challenging, as it requires processing dense and temporally extended visual inputs that exceed the capacity of image-based models. This paper identifies the perception bottleneck and token overload as key challenges in extending image-based VLMs to the video domain. To address these issues, we propose D-CoDe, a training-free adaptation framework that incorporates dynamic compression and question decomposition. Specifically, dynamic compression alleviates the perception bottleneck through adaptive selection of representative frames and content-aware aggregation of spatial tokens, thereby reducing redundancy while preserving informative content. In parallel, question decomposition mitigates token overload by reformulating the original query into sub-questions, guiding the model to focus on distinct aspects of the video and enabling more comprehensive understanding. Experiments demonstrate that D-CoDe effectively improves video understanding across various benchmarks. Furthermore, strong performance on the challenging long-video benchmark highlights the potential of D-CoDe in handling complex video-language tasks. Code is available at https://github.com/hukcc/D-CoDe.

---

## 150. Edu-EmotionNet: Cross-Modality Attention Alignment with Temporal Feedback Loops

**论文链接:** [http://arxiv.org/abs/2510.08802v1](http://arxiv.org/abs/2510.08802v1)

**作者:** S M Rafiuddin

**发布时间:** 2025-10-09

**备注:** 6 Pages, 6 Figures, 3 Tables, Accepted as a Regular Research paper at  ICMLA 2025

### GPT解析

### 总结

本研究提出了Edu-EmotionNet框架，用于在线教育中的学习者情绪识别，通过联合建模时间情绪演变和模态可靠性，实现了稳健的情感识别。

### 背景

在线教育中理解学习者情绪对提高参与度和个性化教学至关重要。现有情绪识别方法通常采用静态融合策略，并假设模态输入始终可靠，这在真实学习环境中很少成立。

### 目的

提出一个新框架，联合建模时间情绪演变和模态可靠性，以实现稳健的情感识别，特别适用于在线教育环境。

### 方法

该模型包含三个关键组件：跨模态注意力对齐模块用于动态跨模态上下文共享；模态重要性估计器为每个模态在每一步分配基于置信度的权重；时间反馈循环利用先前的预测来强制时间一致性。

### 主要发现

Edu-EmotionNet在IEMOCAP和MOSEI的教育子集上取得了最先进的性能，并显示出对缺失或有噪声模态的强大鲁棒性。可视化证实了其捕捉情绪转变和自适应优先考虑可靠信号的能力。

### 结论

该模型适合部署在实时学习系统中，能够有效识别学习者的情绪状态，为个性化教学提供支持。

### 翻译

理解在线教育中的学习者情绪对于提高参与度和个性化教学至关重要。虽然先前在情绪识别方面的工作探索了多模态融合和时间建模，但现有方法通常依赖静态融合策略，并假设模态输入始终可靠，这在真实学习环境中很少成立。我们引入了Edu-EmotionNet，一个新颖的框架，联合建模时间情绪演变和模态可靠性，以实现稳健的情感识别。我们的模型包含三个关键组件：用于动态跨模态上下文共享的跨模态注意力对齐模块，为每个模态在每一步分配基于置信度的权重的模态重要性估计器，以及利用先前预测强制时间一致性的时间反馈循环。在为困惑、好奇、无聊和沮丧重新注释的教育子集IEMOCAP和MOSEI上评估，Edu-EmotionNet取得了最先进的性能，并显示出对缺失或有噪声模态的强大鲁棒性。可视化证实了其捕捉情绪转变和自适应优先考虑可靠信号的能力，使其非常适合部署在实时学习系统中。


### 论文摘要

Understanding learner emotions in online education is critical for improving engagement and personalized instruction. While prior work in emotion recognition has explored multimodal fusion and temporal modeling, existing methods often rely on static fusion strategies and assume that modality inputs are consistently reliable, which is rarely the case in real-world learning environments. We introduce Edu-EmotionNet, a novel framework that jointly models temporal emotion evolution and modality reliability for robust affect recognition. Our model incorporates three key components: a Cross-Modality Attention Alignment (CMAA) module for dynamic cross-modal context sharing, a Modality Importance Estimator (MIE) that assigns confidence-based weights to each modality at every time step, and a Temporal Feedback Loop (TFL) that leverages previous predictions to enforce temporal consistency. Evaluated on educational subsets of IEMOCAP and MOSEI, re-annotated for confusion, curiosity, boredom, and frustration, Edu-EmotionNet achieves state-of-the-art performance and demonstrates strong robustness to missing or noisy modalities. Visualizations confirm its ability to capture emotional transitions and adaptively prioritize reliable signals, making it well suited for deployment in real-time learning systems

---

## 151. GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.07791v2](http://arxiv.org/abs/2510.07791v2)

**作者:** Qinghongbing Xie, Zhaoyuan Xia, Feng Zhu, Lijun Gong, Ziyue Li, Rui Zhao, Long Zeng

**发布时间:** 2025-10-09

**备注:** 20 pages, 13 figures

### GPT解析

### 总结

本文提出了Geo-Temporal Reasoning基准测试(GTR-Bench)，用于评估视觉-语言模型在结合图像/视频和图形上下文时的地理时空智能能力。评估显示当前最佳模型表现显著落后于人类，并揭示了三个主要缺陷。

### 背景

视觉-语言模型的时空智能在自动驾驶、具身AI和通用人工智能领域受到关注。现有基准测试主要关注自我视角推理或地理视角推理，无法评估结合图像/视频和图形上下文时的地理时空智能，这对交通管理和应急响应等领域很重要。

### 目的

解决现有基准测试的不足，引入Geo-Temporal Reasoning基准测试(GTR-Bench)，用于评估大规模摄像头网络中移动目标的地理时间推理能力。

### 方法

创建GTR-Bench基准测试，要求模型在地图和视频之间进行多视角切换，对多个具有非重叠视野的视频进行联合推理，并对任何视频上下文都未观察到的时空区域进行推理。评估了10多种流行的视觉-语言模型。

### 主要发现

即使最佳专有模型Gemini-2.5-Pro(34.9%)在地理时间推理上也显著落后于人类表现(78.61%)。当前模型存在三个主要缺陷：(1)时空上下文利用不平衡；(2)时间预测能力较弱，时间强调任务表现差；(3)缺乏理解或对齐地图数据与多视角视频输入的能力。

### 结论

GTR-Bench为时空智能的研究和应用提供了有价值的见解和新的机会。基准测试和代码将在https://github.com/X-Luffy/GTR-Bench上发布。

### 翻译

最近，视觉-语言模型的时空智能因其对自动驾驶、具身AI和通用人工智能的重要性而受到广泛关注。现有的时空基准测试主要关注基于图像/视频上下文的自我视角推理，或基于图形上下文(如地图)的地理视角推理，因此无法评估VLMs在结合图像/视频和图形上下文时的地理时空智能，这对交通管理和应急响应等领域很重要。为解决这些差距，我们引入了Geo-Temporal Reasoning基准测试(GTR-Bench)，这是一个在大规模摄像头网络中对移动目标进行地理时间推理的新挑战。GTR-Bench更具挑战性，因为它需要在地图和视频之间进行多视角切换，对多个具有非重叠视野的视频进行联合推理，以及对任何视频上下文都未观察到的时空区域进行推理。对GTR-Bench上10多种流行VLMs的评估表明，即使是最佳专有模型Gemini-2.5-Pro(34.9%)在地理时间推理上也显著落后于人类表现(78.61%)。此外，我们对GTR-Bench的全面分析揭示了当前模型在地理时间推理方面的三个主要缺陷。(1)VLMs的推理受到时空上下文不平衡利用的影响。(2)VLMs在时间预测方面能力较弱，导致在时间强调任务上的表现比空间强调任务差。(3)VLMs缺乏理解或对齐地图数据与多视角视频输入的能力。我们相信GTR-Bench为时空智能的研究和应用提供了有价值的见解和新的机会。基准测试和代码将在https://github.com/X-Luffy/GTR-Bench上发布。


### 论文摘要

Recently spatial-temporal intelligence of Visual-Language Models (VLMs) has attracted much attention due to its importance for Autonomous Driving, Embodied AI and General Artificial Intelligence. Existing spatial-temporal benchmarks mainly focus on egocentric perspective reasoning with images/video context, or geographic perspective reasoning with graphics context (eg. a map), thus fail to assess VLMs' geographic spatial-temporal intelligence with both images/video and graphics context, which is important for areas like traffic management and emergency response. To address the gaps, we introduce Geo-Temporal Reasoning benchmark (GTR-Bench), a novel challenge for geographic temporal reasoning of moving targets in a large-scale camera network. GTR-Bench is more challenging as it requires multiple perspective switches between maps and videos, joint reasoning across multiple videos with non-overlapping fields of view, and inference over spatial-temporal regions that are unobserved by any video context. Evaluations of more than 10 popular VLMs on GTR-Bench demonstrate that even the best proprietary model, Gemini-2.5-Pro (34.9%), significantly lags behind human performance (78.61%) on geo-temporal reasoning. Moreover, our comprehensive analysis on GTR-Bench reveals three primary deficiencies of current models for geo-temporal reasoning. (1) VLMs' reasoning is impaired by an imbalanced utilization of spatial-temporal context. (2) VLMs are weak in temporal forecasting, which leads to worse performance on temporal-emphasized tasks than on spatial-emphasized tasks. (3) VLMs lack the proficiency to comprehend or align the map data with multi-view video inputs. We believe GTR-Bench offers valuable insights and opens up new opportunities for research and applications in spatial-temporal intelligence. Benchmark and code will be released at https://github.com/X-Luffy/GTR-Bench.

---

## 152. From Captions to Keyframes: KeyScore for Multimodal Frame Scoring and Video-Language Understanding

**论文链接:** [http://arxiv.org/abs/2510.06509v2](http://arxiv.org/abs/2510.06509v2)

**作者:** Shih-Yao Lin, Sibendu Paul, Caren Chen

**发布时间:** 2025-10-07

**备注:** 10 pages, 4 figures

### GPT解析

### 总结

本文提出了KeyScore和STACFP两种方法，用于选择信息量大的关键帧，提高视频理解效率和准确性。KeyScore是一种基于字幕感知的帧评分方法，STACFP是一种时空自适应聚类方法，两者结合可减少无用帧，保留关键内容，实现更快更准确的推理。

### 背景

选择信息量大的关键帧对于高效的视频理解至关重要，但现有方法往往依赖于启发式方法，忽略语义，或者产生冗余帧。

### 目的

提出一种能够结合语义相似性、时间代表性和上下文下降影响三种互补信号的帧评分方法，生成帧级别的重要性分数，用于训练关键帧提取器或指导视频语言模型。

### 方法

1. KeyScore：字幕感知的帧评分方法，结合三种互补信号：与字幕的语义相似性、时间代表性和上下文下降影响。2. STACFP：时空自适应聚类方法，在长视频中生成多样且紧凑的帧提案。

### 主要发现

在三个标准视频语言基准（MSRVTT、MSVD、DiDeMo）上的实验表明，结合STACFP和KeyScore与全帧处理相比可实现高达99%的帧减少，同时在视频文本检索、关键帧提取和动作识别任务中优于均匀8帧编码器。

### 结论

通过专注于语义相关的帧，该方法提高了效率和性能，实现了可扩展的、基于字幕的视频理解。

### 翻译

选择信息量大的关键帧对于高效的视频理解至关重要，但现有方法往往依赖于启发式方法，忽略语义，或者产生冗余帧。我们提出了KeyScore，一种字幕感知的帧评分方法，结合了三种互补信号：与字幕的语义相似性、时间代表性和上下文下降影响。应用于大规模视频字幕数据集，KeyScore生成帧级别的重要性分数，使能够训练关键帧提取器或指导视频语言模型。为此，我们还提出了STACFP，一种时空自适应聚类方法，能够在长视频中生成多样且紧凑的帧提案。KeyScore和STACFP共同减少无用帧，同时保留关键内容，实现更快更准确的推理。我们在三个标准视频语言基准（MSRVTT、MSVD、DiDeMo）上的实验表明，与全帧处理相比，结合STACFP和KeyScore可以实现高达99%的帧减少，同时在视频文本检索、关键帧提取和动作识别任务中优于均匀8帧编码器。通过专注于语义相关的帧，我们的方法提高了效率和性能，实现了可扩展的、基于字幕的视频理解。


### 论文摘要

Selecting informative keyframes is critical for efficient video understanding, yet existing approaches often rely on heuristics, ignore semantics, or produce redundant frames. We propose KeyScore, a caption-aware frame scoring method that combines three complementary signals: semantic similarity to captions, temporal representativeness, and contextual drop impact. Applied to large-scale video-caption datasets, KeyScore generates frame-level importance scores that enable training keyframe extractors or guiding video-language models. To support this, we also propose STACFP, a Spatio-Temporal Adaptive Clustering method that generates diverse and compact frame proposals across long videos. Together, KeyScore and STACFP reduce uninformative frames while preserving critical content, resulting in faster and more accurate inference. Our experiments on three standard video-language benchmarks (MSRVTT, MSVD, DiDeMo) show that combining STACFP and KeyScore enables up to 99% frame reduction compared to full-frame processing, while outperforming uniform 8-frame encoders in video-text retrieval, keyframe extraction, and action recognition tasks. By focusing on semantically relevant frames, our method enhances both efficiency and performance, enabling scalable and caption-grounded video understanding.

---

## 153. InfiniHuman: Infinite 3D Human Creation with Precise Control

**论文链接:** [http://arxiv.org/abs/2510.11650v1](http://arxiv.org/abs/2510.11650v1)

**作者:** Yuxuan Xue, Xianghui Xie, Margaret Kostyrko, Gerard Pons-Moll

**发布时间:** 2025-10-13

**DOI:** 10.1145/3757377.3763815

**备注:** Accepted to ACM SIGGRAPH Asia 2025. Project website:  https://yuxuan-xue.com/infini-human

### GPT解析

### 总结

论文提出了InfiniHuman框架，通过利用现有基础模型生成丰富注释的3D人体数据，解决了大规模人体数据集收集和标注成本高的问题。该框架包括InfiniHumanData（自动数据生成管道）和InfiniHumanGen（基于扩散的生成管道），能够生成高质量、可精确控制的3D人体化身。

### 背景

生成真实且可控的3D人体化身是一项长期挑战，尤其是在覆盖广泛属性范围（如种族、年龄、服装风格和详细身体形状）时。捕获和注释用于训练生成模型的大规模人体数据集成本极高，且在规模和多样性上受到限制。

### 目的

研究核心问题是：现有基础模型是否可以被提炼，以生成理论上无限的、丰富注释的3D人体数据？论文旨在提出一种方法，以最低成本和理论上无限的扩展性生成丰富注释的人体数据。

### 方法

提出了InfiniHuman框架协同提炼现有模型；InfiniHumanData作为全自动管道利用视觉语言和图像生成模型创建大规模多模态数据集；InfiniHumanGen作为基于扩散的生成管道以文本、身体形状和服装资源为条件。数据集包含111K个身份，每个身份都有多粒度文本描述、多视图RGB图像、详细服装图像和SMPL身体形状参数。

### 主要发现

用户研究表明自动生成的身份与扫描渲染无法区分；InfiniHumanData包含111K个身份，覆盖前所未有的多样性；实验证明在视觉质量、生成速度和可控性方面显著优于最先进方法；通过实用且经济实惠的解决方案实现了在有效无限规模下的高质量化身生成和细粒度控制。

### 结论

InfiniHuman框架提供了一种实用且经济实惠的解决方案，能够以有效无限的规模生成高质量、细粒度控制的3D人体化身。研究团队将公开自动数据生成管道、全面的InfiniHumanData数据集和InfiniHumanGen模型。

### 翻译

生成真实且可控的3D人体化身是一项长期存在的挑战，尤其是在覆盖广泛的属性范围（如种族、年龄、服装风格和详细的身体形状）时。捕获和注释用于训练生成模型的大规模人体数据集成本极高，且在规模和多样性上受到限制。我们在本文中要解决的核心问题是：现有的基础模型是否可以被提炼，以生成理论上无限的、丰富注释的3D人体数据？我们介绍了InfiniHuman，一个协同提炼这些模型以最低成本和理论上无限的扩展性生成丰富注释的人体数据的框架。我们提出了InfiniHumanData，一个全自动管道，利用视觉语言和图像生成模型创建大规模多模态数据集。用户研究表明，我们自动生成的身份与扫描渲染无法区分。InfiniHumanData包含111K个身份，覆盖前所未有的多样性。每个身份都注释有多粒度文本描述、多视图RGB图像、详细服装图像和SMPL身体形状参数。基于此数据集，我们提出了InfiniHumanGen，一个基于扩散的生成管道，以文本、身体形状和服装资源为条件。InfiniHumanGen能够实现快速、真实且精确可控的化身生成。大量实验表明，在视觉质量、生成速度和可控性方面显著优于最先进的方法。我们的方法通过实用且经济实惠的解决方案，实现了在有效无限规模下的高质量化身生成和细粒度控制。我们将在https://yuxuan-xue.com/infini-human公开自动数据生成管道、全面的InfiniHumanData数据集和InfiniHumanGen模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决生成真实且可控制的3D人类化身（avatar）的挑战，特别是在覆盖广泛属性如种族、年龄、服装风格和身体形状时。这个问题在虚拟现实、数字时尚、游戏和社会远程呈现等领域至关重要，因为这些领域对逼真且可定制的个性化化身需求日益增长，而现有方法要么依赖成本高昂的真实扫描数据，要么生成质量有限或控制性不足。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过'蒸馏'现有基础模型来生成大规模3D人类数据，设计了一个完全自动化的框架。他们利用了多种现有技术：使用GPT-4o生成文本描述，微调FLUX模型生成正交视图的'扫描式'图像，借鉴虚拟试衣技术提取服装图像，使用SMPL模型表示人体形状和姿势，应用扩散模型生成一致的多视图图像，并基于OminiControl2进行微调实现高分辨率生成。这种方法整合了多个领域的前沿技术，形成了一个协同工作的系统。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过整合和重新利用现有基础模型创建一个完全自动化的框架，生成大规模、丰富注释的3D人类数据，并基于这些数据训练生成模型以实现精确控制。整体流程分为两部分：1) 数据生成（InfiniHumanData）：包括多粒度文本描述生成、正交文本到图像转换、虚拟试衣提取服装图像、单目身体拟合获取SMPL参数、正交多视图扩散生成高分辨率多视图图像；2) 生成模型（InfiniHumanGen）：包括Gen-Schnell（快速生成3D高斯点云）和Gen-HRes（高分辨率纹理网格生成），两者都支持基于文本、身体形状和服装图像的条件化生成。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) InfiniHuman框架，实现完全自动化的无限3D人类数据生成；2) InfiniHumanData数据集，包含111K个多样化身份，具有前所未有的种族、年龄、服装风格等多样性；3) InfiniHumanGen生成模型，提供Gen-Schnell和Gen-HRes两种互补模型，支持精确控制。相比之前工作，该方法解决了视觉质量、生成速度和属性可控性的局限性，提供了对服装的精确控制，生成的身份在视觉上与真实扫描无法区分，数据集规模和多样性远超现有公开数据集，生成速度比现有高分辨率方法快8倍以上。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'InfiniHuman通过整合现有基础模型创建了一个完全自动化的框架，能够生成大规模、多样化的3D人类数据集，并基于此实现了高质量、高速度且具有精确控制的3D人类化身生成，从而降低了高质量化身创建的门槛并实现了无限规模的扩展。'}


### 论文摘要

Generating realistic and controllable 3D human avatars is a long-standing challenge, particularly when covering broad attribute ranges such as ethnicity, age, clothing styles, and detailed body shapes. Capturing and annotating large-scale human datasets for training generative models is prohibitively expensive and limited in scale and diversity. The central question we address in this paper is: Can existing foundation models be distilled to generate theoretically unbounded, richly annotated 3D human data? We introduce InfiniHuman, a framework that synergistically distills these models to produce richly annotated human data at minimal cost and with theoretically unlimited scalability. We propose InfiniHumanData, a fully automatic pipeline that leverages vision-language and image generation models to create a large-scale multi-modal dataset. User study shows our automatically generated identities are undistinguishable from scan renderings. InfiniHumanData contains 111K identities spanning unprecedented diversity. Each identity is annotated with multi-granularity text descriptions, multi-view RGB images, detailed clothing images, and SMPL body-shape parameters. Building on this dataset, we propose InfiniHumanGen, a diffusion-based generative pipeline conditioned on text, body shape, and clothing assets. InfiniHumanGen enables fast, realistic, and precisely controllable avatar generation. Extensive experiments demonstrate significant improvements over state-of-the-art methods in visual quality, generation speed, and controllability. Our approach enables high-quality avatar generation with fine-grained control at effectively unbounded scale through a practical and affordable solution. We will publicly release the automatic data generation pipeline, the comprehensive InfiniHumanData dataset, and the InfiniHumanGen models at https://yuxuan-xue.com/infini-human.

---

## 154. Benchmarking foundation models for hyperspectral image classification: Application to cereal crop type mapping

**论文链接:** [http://arxiv.org/abs/2510.11576v1](http://arxiv.org/abs/2510.11576v1)

**作者:** Walid Elbarz, Mohamed Bourriz, Hicham Hajji, Hamd Ait Abdelali, François Bourzeix

**发布时间:** 2025-10-13

**备注:** Being reviewed for WHISPERS conference ( Workshop on Hyperspectral  Image and Signal Processing: Evolution in Remote Sensing )

### GPT解析

### 总结

该研究评估了三种基础模型在超光谱作物制图中的性能，发现SpectralEarth预训练的Vision Transformer表现最佳，达到93.5%的总体准确率，强调了模型架构对跨区域泛化能力的重要性。

### 背景

基础模型正在改变地球观测领域，但它们在超光谱作物制图中的潜力尚未得到充分探索。

### 目的

基准测试三种基础模型用于超光谱作物制图，评估它们在不同地理区域和传感器平台上的泛化能力。

### 方法

测试了三种基础模型（HyperSigma、DOFA和在SpectralEarth数据集上预训练的Vision Transformer），在手动标记的训练区域数据上进行微调，并在独立的测试区域进行评估。性能测量包括总体准确率(OA)、平均准确率(AA)和F1分数。

### 主要发现

HyperSigma达到34.5%的OA（±1.8%），DOFA达到62.6%的OA（±3.5%），SpectralEarth模型达到93.5%的OA（±0.8%）。从头开始训练的紧凑型SpectralEarth变体达到了91%的性能，突显了模型架构对于在地理区域和传感器平台之间强泛化能力的重要性。

### 结论

这些结果为操作超光谱作物制图的基础模型提供了系统评估，并概述了未来模型开发的方向。

### 翻译

基础模型正在改变地球观测，但它们在超光谱作物制图中的潜力仍然未被充分探索。本研究使用超光谱图像对三种基础模型进行了基准测试，用于谷物作物制图：HyperSigma、DOFA以及在SpectralEarth数据集上预训练的Vision Transformer（一个大的多时相超光谱档案）。模型在手动标记的训练区域数据上进行微调，并在独立的测试区域进行评估。性能通过总体准确率(OA)、平均准确率(AA)和F1分数来衡量。HyperSigma实现了34.5%的OA（±1.8%），DOFA达到62.6%（±3.5%），而SpectralEarth模型实现了93.5%的OA（±0.8%）。从头开始训练的紧凑型SpectralEarth变体达到了91%，突显了模型架构对于在地理区域和传感器平台之间强泛化能力的重要性。这些结果为操作超光谱作物制图的基础模型提供了系统评估，并概述了未来模型开发的方向。


### 论文摘要

Foundation models are transforming Earth observation, but their potential for hyperspectral crop mapping remains underexplored. This study benchmarks three foundation models for cereal crop mapping using hyperspectral imagery: HyperSigma, DOFA, and Vision Transformers pre-trained on the SpectralEarth dataset (a large multitemporal hyperspectral archive). Models were fine-tuned on manually labeled data from a training region and evaluated on an independent test region. Performance was measured with overall accuracy (OA), average accuracy (AA), and F1-score. HyperSigma achieved an OA of 34.5% (+/- 1.8%), DOFA reached 62.6% (+/- 3.5%), and the SpectralEarth model achieved an OA of 93.5% (+/- 0.8%). A compact SpectralEarth variant trained from scratch achieved 91%, highlighting the importance of model architecture for strong generalization across geographic regions and sensor platforms. These results provide a systematic evaluation of foundation models for operational hyperspectral crop mapping and outline directions for future model development.

---

## 155. How many samples to label for an application given a foundation model? Chest X-ray classification study

**论文链接:** [http://arxiv.org/abs/2510.11553v1](http://arxiv.org/abs/2510.11553v1)

**作者:** Nikolay Nechaev, Evgenia Przhezdzetskaya, Viktor Gombolevskiy, Dmitry Umerenkov, Dmitry Dylov

**发布时间:** 2025-10-13

**备注:** 8 pages, 5 figures

### GPT解析

### 总结

该研究探讨了胸部X光分类中基础模型对标注数据的需求量，发现XrayCLIP和XraySigLIP模型在显著少于传统ResNet-50基线模型的情况下仍能实现高性能，且仅需50个标注样本即可准确预测最终性能表现。

### 背景

胸部X光分类对医学诊断至关重要，但传统方法需要大量标注数据，资源消耗大。基础模型虽可减少对标注数据的依赖，但具体需要多少标注样本尚不明确。

### 目的

系统性评估使用幂律拟合方法预测达到特定ROC-AUC阈值所需训练样本量的可行性，旨在确定胸部X光分类任务中基础模型的最优标注样本数量。

### 方法

研究测试了多种病理学和基础模型，应用幂律拟合来预测达到特定性能阈值所需的训练样本量，并验证了从少量标注样本(仅50个)的学习曲线斜率预测最终性能的准确性。

### 主要发现

XrayCLIP和XraySigLIP模型相比ResNet-50基线模型，在显著更少的标注样本下实现了同等或更好的性能；仅使用50个标注病例的学习曲线就能准确预测模型最终的性能平台期。

### 结论

研究结果使医学影像从业者能够通过仅标注针对目标性能所必需的样本，有效降低数据标注成本，优化资源分配。

### 翻译

胸部X光分类至关重要但资源密集，通常需要大量标注数据才能实现准确诊断。基础模型可以减少这种依赖，但需要多少标注样本尚不清楚。我们系统性评估了使用幂律拟合来预测达到特定ROC-AUC阈值所需训练样本量的方法。通过测试多种病理学和基础模型，我们发现XrayCLIP和XraySigLIP在显著少于ResNet-50基线的标注样本下实现了强大性能。重要的是，仅从50个标注病例的学习曲线斜率就能准确预测最终性能平台期。我们的结果使从业者能够通过仅标注针对目标性能所必需的样本来最小化标注成本。


### 论文摘要

Chest X-ray classification is vital yet resource-intensive, typically demanding extensive annotated data for accurate diagnosis. Foundation models mitigate this reliance, but how many labeled samples are required remains unclear. We systematically evaluate the use of power-law fits to predict the training size necessary for specific ROC-AUC thresholds. Testing multiple pathologies and foundation models, we find XrayCLIP and XraySigLIP achieve strong performance with significantly fewer labeled examples than a ResNet-50 baseline. Importantly, learning curve slopes from just 50 labeled cases accurately forecast final performance plateaus. Our results enable practitioners to minimize annotation costs by labeling only the essential samples for targeted performance.

---

## 156. An Encoder-Integrated PhoBERT with Graph Attention for Vietnamese Token-Level Classification

**论文链接:** [http://arxiv.org/abs/2510.11537v1](http://arxiv.org/abs/2510.11537v1)

**作者:** Ba-Quang Nguyen

**发布时间:** 2025-10-13

**备注:** 11 pages, 1 figure. Submitted to VLSP 2025 and reviewed

### GPT解析

### 总结

本文提出了一种名为TextGraphFuseGAT的新型神经网络架构，结合了预训练的transformer编码器和图注意力网络，用于标记级分类任务，在多个越南语数据集上取得了优异的性能。

### 背景

现有的标记级分类模型在处理标记间复杂关系时存在局限性，特别是在特定领域如医疗、COVID-19等需要精确识别实体类型的任务中。

### 目的

开发一种能够有效捕获标记间依赖关系并结合预训练语义特征的模型，以提高标记级分类任务在多个领域的性能。

### 方法

提出TextGraphFuseGAT模型，在PhoBERT生成的标记嵌入上构建全连接图，使用GAT层捕获标记间依赖关系，并应用Transformer风格的自注意力层增强上下文化，最后通过分类头进行序列标注。

### 主要发现

在三个越南语基准数据集（PhoNER-COVID19、PhoDisfluency和VietMed-NER）上的实验表明，该方法始终优于强基线模型，包括仅使用transformer的模型和混合神经网络模型。

### 结论

将预训练语义特征与基于图的建模相结合是提高跨领域标记分类性能的有效方法，特别是在处理具有专业词汇和领域特定表达的数据时。

### 翻译

我们提出了一种名为TextGraphFuseGAT的新型神经架构，该架构集成了预训练的transformer编码器（PhoBERT）和图注意力网络，用于标记级分类任务。所提出的模型在PhoBERT生成的标记嵌入上构建了一个全连接图，使GAT层能够捕获丰富的标记间依赖关系，而不仅仅是顺序上下文建模。为了进一步增强上下文化，在图增强的嵌入上应用了Transformer风格的自注意力层。最终的标记表示通过分类头进行序列标注。我们在三个越南语基准数据集上评估了我们的方法：PhoNER-COVID19用于COVID-19领域的命名实体识别，PhoDisfluency用于言语不流畅性检测，以及VietMed-NER用于医疗领域NER。VietMed-NER是第一个越南语医疗口语NER数据集，包含18种从真实医疗语音转录中收集的实体类型，并使用BIO标记方案进行标注。其专业词汇和领域特定表达使其成为标记级分类模型的一个具有挑战性的基准。实验结果表明，我们的方法始终优于强基线模型，包括仅使用transformer的模型和混合神经网络模型（如BiLSTM + CNN + CRF），证实了结合预训练语义特征和基于图的关系建模对提高跨领域标记分类的有效性。


### 论文摘要

We propose a novel neural architecture named TextGraphFuseGAT, which integrates a pretrained transformer encoder (PhoBERT) with Graph Attention Networks for token-level classification tasks. The proposed model constructs a fully connected graph over the token embeddings produced by PhoBERT, enabling the GAT layer to capture rich inter-token dependencies beyond those modeled by sequential context alone. To further enhance contextualization, a Transformer-style self-attention layer is applied on top of the graph-enhanced embeddings. The final token representations are passed through a classification head to perform sequence labeling. We evaluate our approach on three Vietnamese benchmark datasets: PhoNER-COVID19 for named entity recognition in the COVID-19 domain, PhoDisfluency for speech disfluency detection, and VietMed-NER for medical-domain NER. VietMed-NER is the first Vietnamese medical spoken NER dataset, featuring 18 entity types collected from real-world medical speech transcripts and annotated with the BIO tagging scheme. Its specialized vocabulary and domain-specific expressions make it a challenging benchmark for token-level classification models. Experimental results show that our method consistently outperforms strong baselines, including transformer-only and hybrid neural models such as BiLSTM + CNN + CRF, confirming the effectiveness of combining pretrained semantic features with graph-based relational modeling for improved token classification across multiple domains.

---

## 157. Uncertainty-Aware ControlNet: Bridging Domain Gaps with Synthetic Image Generation

**论文链接:** [http://arxiv.org/abs/2510.11346v1](http://arxiv.org/abs/2510.11346v1)

**作者:** Joshua Niemeijer, Jan Ehrhardt, Heinz Handels, Hristina Uzunova

**发布时间:** 2025-10-13

**备注:** Accepted for presentation at ICCV Workshops 2025, "The 4th Workshop  on What is Next in Multimodal Foundation Models?" (MMFM)

### GPT解析

### 总结

本文提出了一种利用未标记领域数据训练ControlNet的方法，通过引入不确定性概念到控制机制中，使模型能够创建具有高不确定性的目标领域标注数据，显著提高了分割结果，无需额外监督。

### 背景

生成模型是创建高质量图像数据的有价值工具。ControlNet等受控扩散模型已允许创建标记分布，可用于增强原始训练分布。然而，ControlNet倾向于重现原始训练分布，限制了增强效果。以视网膜OCT为例，高质量的Spectralis图像有真实分割可用于训练，但Home-OCT设备产生的图像质量较低且存在较大域偏移，使得现成的分割网络无法应用。

### 目的

提出一种方法利用未标记领域数据训练ControlNet，通过引入不确定性概念，创建具有高不确定性的目标领域标注数据，解决域偏移问题，提高分割性能。

### 方法

将不确定性概念引入控制机制中，不确定性表示给定图像不属于下游任务（如分割）的训练分布。最终网络结合两种控制：来自未标记数据集的不确定性控制和来自标记数据集的语义控制。这种方法允许创建来自目标领域的具有高不确定性的标注数据，即来自未标记分布的带标签合成数据。

### 主要发现

所提出的ControlNet能够创建来自Home-OCT域的带注释图像，显著提高了分割结果，无需额外监督。与风格迁移相比，不确定性引导能够实现任意的域偏移，而无需严格学习图像风格，这一点在交通场景实验中也得到了验证。

### 结论

通过引入不确定性概念到ControlNet的控制机制中，可以利用未标记领域数据训练模型，创建高质量的标注数据，有效解决域偏移问题，显著提高分割性能，且无需额外监督。

### 翻译

生成模型是用于高质量图像数据受控创建的有价值工具。像ControlNet这样的受控扩散模型已允许创建标记分布。当训练判别模型（如语义分割）时，这类合成数据可以增强原始训练分布。然而，这种增强效果有限，因为ControlNet倾向于重现原始训练分布。这项工作引入了一种利用未标记领域数据训练ControlNet的方法，通过将不确定性概念引入控制机制中。不确定性表示给定图像不属于下游任务（如分割）的训练分布。因此，最终网络涉及两种控制：来自未标记数据集的不确定性控制和来自标记数据集的语义控制。所得到的ControlNet允许我们创建来自目标领域的高不确定性标注数据，即来自未标记分布的带标签合成数据。在我们的场景中，我们考虑视网膜OCT，通常高质量的Spectralis图像有给定的真实分割，可用于训练分割网络。然而，Home-OCT设备的最新发展产生了质量较低且存在较大域偏移的视网膜OCT，使得现成的分割网络无法应用于此类数据。使用所提出的方法合成来自Home-OCT域的带注释图像弥合了这一差距，并在不添加任何额外监督的情况下显著提高了分割结果。与风格迁移相比，不确定性引导的优势很明显：它能够实现任意的域偏移，而无需严格学习图像风格。这一点在交通场景实验中也得到了验证。


### 论文摘要

Generative Models are a valuable tool for the controlled creation of high-quality image data. Controlled diffusion models like the ControlNet have allowed the creation of labeled distributions. Such synthetic datasets can augment the original training distribution when discriminative models, like semantic segmentation, are trained. However, this augmentation effect is limited since ControlNets tend to reproduce the original training distribution.   This work introduces a method to utilize data from unlabeled domains to train ControlNets by introducing the concept of uncertainty into the control mechanism. The uncertainty indicates that a given image was not part of the training distribution of a downstream task, e.g., segmentation. Thus, two types of control are engaged in the final network: an uncertainty control from an unlabeled dataset and a semantic control from the labeled dataset. The resulting ControlNet allows us to create annotated data with high uncertainty from the target domain, i.e., synthetic data from the unlabeled distribution with labels. In our scenario, we consider retinal OCTs, where typically high-quality Spectralis images are available with given ground truth segmentations, enabling the training of segmentation networks. The recent development in Home-OCT devices, however, yields retinal OCTs with lower quality and a large domain shift, such that out-of-the-pocket segmentation networks cannot be applied for this type of data. Synthesizing annotated images from the Home-OCT domain using the proposed approach closes this gap and leads to significantly improved segmentation results without adding any further supervision. The advantage of uncertainty-guidance becomes obvious when compared to style transfer: it enables arbitrary domain shifts without any strict learning of an image style. This is also demonstrated in a traffic scene experiment.

---

## 158. Protein as a Second Language for LLMs

**论文链接:** [http://arxiv.org/abs/2510.11188v1](http://arxiv.org/abs/2510.11188v1)

**作者:** Xinhui Chen, Zuchao Li, Mengqi Gao, Yufeng Zhang, Chak Tou Leong, Haoyang Li, Jiaqi Chen

**发布时间:** 2025-10-13

**备注:** Main paper: 9 pages, 6 figures. With references and appendix: 18  pages, 9 figures total. Submitted to ICLR 2026 (under review)

### GPT解析

### 总结

研究提出'蛋白质作为第二语言'框架，将氨基酸序列重新表述为大型语言模型可解释的符号语言，通过自适应构建序列-问题-答案三元组在零样本设置中揭示蛋白质功能线索，无需额外训练。

### 背景

解析未知蛋白质序列的功能是具有广泛科学影响的基本挑战，但现有方法大多依赖于特定任务适配器或大规模监督微调。

### 目的

开发一种不依赖特定任务适配器或大规模监督微调的蛋白质功能解析方法。

### 方法

引入'蛋白质作为第二语言'框架，将氨基酸序列重新表述为新型符号语言中的句子，大型语言模型可通过上下文示例解释。该方法自适应构建序列-问题-答案三元组，在零样本设置中揭示功能线索。为此，整理了包含79,926个蛋白质-QA实例的双语语料库，涵盖属性预测、描述性理解和扩展推理。

### 主要发现

该方法在各种开源大型语言模型和GPT-4上取得了一致的改进，ROUGE-L最高提升17.2%（平均+7%），甚至超过了微调的蛋白质特定语言模型。

### 结论

当用蛋白质作为语言的线索引导时，通用大型语言模型可以超越领域专用模型，为基础模型中的蛋白质理解提供了可扩展的途径。

### 翻译

解析未知蛋白质序列的功能是一个具有广泛科学影响的基本挑战，但大多数现有方法依赖于特定任务的适配器或大规模监督微调。我们引入了'蛋白质作为第二语言'框架，将氨基酸序列重新表述为一种新型符号语言中的句子，大型语言模型可以通过上下文示例来解释。我们的方法自适应地构建序列-问题-答案三元组，在零样本设置中揭示功能线索，无需任何进一步训练。为此，我们整理了一个包含79,926个蛋白质-QA实例的双语语料库，涵盖属性预测、描述性理解和扩展推理。实验上，我们的方法在各种开源大型语言模型和GPT-4上取得了一致的改进，ROUGE-L最高提升17.2%（平均+7%），甚至超过了微调的蛋白质特定语言模型。这些结果表明，当用蛋白质作为语言的线索引导时，通用大型语言模型可以超越领域专用模型，为基础模型中的蛋白质理解提供了可扩展的途径。


### 论文摘要

Deciphering the function of unseen protein sequences is a fundamental challenge with broad scientific impact, yet most existing methods depend on task-specific adapters or large-scale supervised fine-tuning. We introduce the "Protein-as-Second-Language" framework, which reformulates amino-acid sequences as sentences in a novel symbolic language that large language models can interpret through contextual exemplars. Our approach adaptively constructs sequence-question-answer triples that reveal functional cues in a zero-shot setting, without any further training. To support this process, we curate a bilingual corpus of 79,926 protein-QA instances spanning attribute prediction, descriptive understanding, and extended reasoning. Empirically, our method delivers consistent gains across diverse open-source LLMs and GPT-4, achieving up to 17.2% ROUGE-L improvement (average +7%) and even surpassing fine-tuned protein-specific language models. These results highlight that generic LLMs, when guided with protein-as-language cues, can outperform domain-specialized models, offering a scalable pathway for protein understanding in foundation models.

---

## 159. G2L:From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2510.11176v1](http://arxiv.org/abs/2510.11176v1)

**作者:** Yesung Cho, Sungmin Lee, Geongyu Lee, Minkyung Lee, Jongbae Park, Dongmyung Shin

**发布时间:** 2025-10-13

### GPT解析

### 总结

本研究提出了一种名为G2L框架的新策略，通过知识蒸馏技术使大规模病理学模型（仅占giga-scale模型15%参数）在癌症特定任务上达到与giga-scale模型相当的性能，同时显著降低计算成本。

### 背景

近期研究表明，扩大训练数据规模、增加癌症类型多样性和增大模型尺寸可提升病理学基础模型性能。然而，giga-scale基础模型（训练于数十万张玻片，覆盖数十种癌症类型，含数十亿参数）因开发和部署中的巨大计算成本，实际应用面临重大挑战。

### 目的

提出一种新策略(G2L框架)，使大规模基础模型（仅占giga-scale模型15%的参数）在癌症特定任务上达到与giga-scale模型相当的性能水平。

### 方法

应用知识蒸馏技术，将giga-scale模型的能力转移到大规模模型，仅使用目标癌症（如乳腺癌、前列腺癌等）的1K张病理玻片进行知识蒸馏。

### 主要发现

蒸馏后的模型在多个基准测试中优于同规模的最先进模型，甚至在某些测试中超过了giga-scale教师模型和huge-scale模型；蒸馏模型表现出更高的鲁棒性指数，对来自多个机构的图像变化具有更强的适应能力。

### 结论

所提出的蒸馏方法是一种数据和参数高效的方式，可以在癌症特定应用中达到giga-scale级别的性能，同时避免过高的计算负担。

### 翻译

近期病理学基础模型研究表明，扩展训练数据、增加癌症类型多样性和提升模型尺寸能持续改善模型性能。然而，giga-scale基础模型（训练于覆盖数十种癌症类型的数十万张玻片，包含数十亿参数）因开发和部署中的巨大计算成本，给实际应用带来重大挑战。本研究提出了一种名为G2L框架的新策略，使大规模基础模型（仅占giga-scale模型15%的参数）在癌症特定任务上达到与giga-scale模型相当的性能水平。我们的方法应用知识蒸馏技术，将giga-scale模型的能力转移到大规模模型，仅使用目标癌症（如乳腺癌、前列腺癌等）的1K张病理玻片。所得蒸馏模型不仅在多个基准测试中优于同规模的最先进模型，而且有趣的是，在某些基准测试中甚至超过了giga-scale教师模型和huge-scale模型。此外，蒸馏模型表现出更高的鲁棒性指数，表明对来自多个机构的图像变化具有更强的适应能力。这些发现表明，针对大规模模型提出的蒸馏方法是实现giga-scale级别性能的数据和参数高效途径，且不会带来过高的计算负担。


### 论文摘要

Recent studies in pathology foundation models have shown that scaling training data, diversifying cancer types, and increasing model size consistently improve their performance. However, giga-scale foundation models, which are trained on hundreds of thousands of slides covering tens of cancer types and contain billions of parameters, pose significant challenges for practical use due to their tremendous computational costs in both development and deployment. In this work, we present a novel strategy, named the G2L framework, to increase the performance of large-scale foundation models, which consist of only $15\%$ of the parameters of giga-scale models, to a comparable performance level of giga-scale models in cancer-specific tasks. Our approach applies knowledge distillation, transferring the capabilities of a giga-scale model to a large-scale model, using just 1K pathology slides of a target cancer (e.g., breast, prostate, etc.). The resulting distilled model not only outperformed state-of-the-art models of the same size (i.e., large-scale) across several benchmarks but also, interestingly, surpassed the giga-scale teacher and huge-scale models in some benchmarks. In addition, the distilled model exhibited a higher robustness index, indicating improved resilience to image variations originating from multiple institutions. These findings suggest that the proposed distillation approach for a large-scale model is a data- and parameter-efficient way to achieve giga-scale-level performance for cancer-specific applications without prohibitive computational burden.

---

## 160. What Slows Down FMware Development? An Empirical Study of Developer Challenges and Resolution Times

**论文链接:** [http://arxiv.org/abs/2510.11138v1](http://arxiv.org/abs/2510.11138v1)

**作者:** Zitao Wang, Zhimin Zhao, Michael W. Godfrey

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文对Foundation Models (FMs)驱动的FMware生态系统进行了首次大规模分析，研究其应用领域、开发挑战和问题解决需求，为改进FMware工具、工作流程和社区支持提供指导。

### 背景

Foundation Models如OpenAI的GPT正在改变软件工程实践，催生了FMware（围绕这些模型构建的应用和基础设施）。FMware支持代码生成、自然语言交互、知识集成和多模态内容创建，但其设计、实现和演化在云平台和本地部署环境中带来了新的挑战。

### 目的

研究FMware在云平台和开源仓库中的开发情况，通过三个重点领域进行实证分析：(1)最常见的应用领域，(2)开发者遇到的关键挑战，(3)需要最大努力解决的问题类型。

### 方法

从GitHub仓库和领先的FMware平台（包括HuggingFace、GPTStore、Ora和Poe）收集数据进行实证调查分析。

### 主要发现

FMware强烈关注教育、内容创建和商业战略；在内存管理、依赖处理和tokenizer配置方面存在持久技术挑战；GitHub上最常报告的是错误报告和核心功能问题；最耗时的解决方案是代码审查、相似性搜索和提示模板设计。

### 结论

通过揭示开发者的实践和痛点，研究指出了改进FMware工具、工作流程和社区支持的机会，提供了指导FMware开发未来的可行见解。

### 翻译

基础模型（FMs），如OpenAI的GPT，正在从根本上改变软件工程的实践，使能够开发围绕这些模型的FMware——应用和基础设施。FMware系统现在支持代码生成、自然语言交互、知识集成和多模态内容创建等任务，凸显了它们对当前软件工程工作流程的颠覆性影响。然而，FMware的设计、实现和演化带来了显著的新挑战，特别是在云平台和本地部署平台上，这些平台的目标、流程和工具往往与传统软件开发不同。据我们所知，这是首次对云平台和开源仓库中的FMware开发进行大规模分析。我们通过三个重点领域对FMware生态系统进行了实证研究：(1)FMware最常见的应用领域，(2)开发者遇到的关键挑战，(3)需要最大努力解决的问题类型。我们的分析借鉴了GitHub仓库以及领先的FMware平台（包括HuggingFace、GPTStore、Ora和Poe）的数据。我们的研究结果显示，FMware强烈关注教育、内容创建和商业战略，同时在内存管理、依赖处理和tokenizer配置方面存在持久的技术挑战。在GitHub上，错误报告和核心功能问题是最常报告的问题，而代码审查、相似性搜索和提示模板设计是最耗时的解决方案。通过揭示开发者的实践和痛点，这项研究指出了改进FMware工具、工作流程和社区支持的机会，并提供了可行的见解，帮助指导FMware开发的未来。


### 论文摘要

Foundation Models (FMs), such as OpenAI's GPT, are fundamentally transforming the practice of software engineering by enabling the development of \emph{FMware} -- applications and infrastructures built around these models. FMware systems now support tasks such as code generation, natural-language interaction, knowledge integration, and multi-modal content creation, underscoring their disruptive impact on current software engineering workflows. However, the design, implementation, and evolution of FMware present significant new challenges, particularly across cloud-based and on-premise platforms where goals, processes, and tools often diverge from those of traditional software development.   To our knowledge, this is the first large-scale analysis of FMware development across both cloud-based platforms and open-source repositories. We empirically investigate the FMware ecosystem through three focus areas: (1) the most common application domains of FMware, (2) the key challenges developers encounter, and (3) the types of issues that demand the greatest effort to resolve. Our analysis draws on data from GitHub repositories and from leading FMware platforms, including HuggingFace, GPTStore, Ora, and Poe. Our findings reveal a strong focus on education, content creation, and business strategy, alongside persistent technical challenges in memory management, dependency handling, and tokenizer configuration. On GitHub, bug reports and core functionality issues are the most frequently reported problems, while code review, similarity search, and prompt template design are the most time-consuming to resolve.   By uncovering developer practices and pain points, this study points to opportunities to improve FMware tools, workflows, and community support, and provides actionable insights to help guide the future of FMware development.

---

## 161. Improving AI Efficiency in Data Centres by Power Dynamic Response

**论文链接:** [http://arxiv.org/abs/2510.11119v1](http://arxiv.org/abs/2510.11119v1)

**作者:** Andrea Marinoni, Sai Shivareddy, Pietro Lio', Weisi Lin, Erik Cambria, Clare Grey

**发布时间:** 2025-10-13

### GPT解析

### 总结

本研究探讨了基于创新方法的AI数据中心电力管理解决方案的能力和局限性，提出将部分输入电力动态化以提高可持续性。

### 背景

人工智能近年来在大型语言模型和基础模型等复杂模型的推动下加速发展，但AI数据中心对电力需求极大，其电力管理问题对环境和可持续发展造成影响。

### 目的

研究基于创新方法的AI数据中心电力管理解决方案的能力和局限性，即让部分输入电力与数据计算功能使用的电力一样动态化。

### 方法

通过分析全球多个数据平台的电力趋势，量化比较被动设备和主动设备在计算增益、能源效率、资本支出减少和管理成本方面的性能。

### 主要发现

这种动态电力管理策略代表了AI数据中心电力管理的一种范式转变，有潜力显著提高AI超算的可持续性。

### 结论

该策略可以增强AI在环境、财务和社会领域的影响，促进AI的可持续发展。

### 翻译

人工智能的稳定增长近年来有所加速，这得益于大型语言模型和基础模型等复杂模型的发展。确保强大可靠的基础设施对于充分发挥AI的潜力至关重要。然而，AI数据中心极其耗电，使其电力管理问题成为焦点，特别是它们对环境和可持续发展的影响。在这项工作中，我们研究了基于创新方法的AI数据中心电力管理解决方案的能力和局限性，即使部分输入电源与用于数据计算功能的电源一样动态化。通过分析来自全球多个数据平台的电力趋势，我们以计算增益、能源效率、资本支出减少和管理成本为量化指标，比较了被动设备和主动设备的性能。这种策略代表了AI数据中心电力管理的一种范式转变，有潜力显著提高AI超算的可持续性，增强其在环境、财务和社会领域的影响力。


### 论文摘要

The steady growth of artificial intelligence (AI) has accelerated in the recent years, facilitated by the development of sophisticated models such as large language models and foundation models. Ensuring robust and reliable power infrastructures is fundamental to take advantage of the full potential of AI. However, AI data centres are extremely hungry for power, putting the problem of their power management in the spotlight, especially with respect to their impact on environment and sustainable development. In this work, we investigate the capacity and limits of solutions based on an innovative approach for the power management of AI data centres, i.e., making part of the input power as dynamic as the power used for data-computing functions. The performance of passive and active devices are quantified and compared in terms of computational gain, energy efficiency, reduction of capital expenditure, and management costs by analysing power trends from multiple data platforms worldwide. This strategy, which identifies a paradigm shift in the AI data centre power management, has the potential to strongly improve the sustainability of AI hyperscalers, enhancing their footprint on environmental, financial, and societal fields.

---

## 162. Enhancing Zero-Shot Anomaly Detection: CLIP-SAM Collaboration with Cascaded Prompts

**论文链接:** [http://arxiv.org/abs/2510.11028v1](http://arxiv.org/abs/2510.11028v1)

**作者:** Yanning Hou, Ke Xu, Junfa Li, Yanran Ruan, Jianfeng Qiu

**发布时间:** 2025-10-13

**备注:** Accepted by PRCV

### GPT解析

### 总结

本文提出了一种新的两阶段框架，用于工业异常检测中的零样本异常分割任务，有效结合了CLIP的异常定位能力和SAM的边界感知能力，解决了基础模型在下游任务中的引导问题。

### 背景

基础模型展现出强大的泛化能力，为零样本异常分割任务带来了新解决方案，但正确引导这些模型解决下游任务仍面临挑战。

### 目的

开发一个能够有效利用CLIP和SAM优势的两阶段框架，提高零样本异常分割的准确性和精确性。

### 方法

1) 提出Co-Feature Point Prompt Generation (PPG)模块，协同使用CLIP和SAM生成正负点提示，引导SAM专注于分割异常区域而非整个对象；2) 引入Cascaded Prompts for SAM (CPS)模块，采用混合提示与SAM轻量级解码器级联，优化分割结果并减少边界粗糙和孤立噪声。

### 主要发现

在多个数据集上验证了该方法的有效性，取得了最先进的零样本异常分割结果，特别是在Visa数据集上，F1-max和AP指标分别比最先进方法高出10.3%和7.7%。

### 结论

该两阶段框架成功解决了SAM在异常分割中的局限性，通过协同利用CLIP和SAM的优势，实现了异常区域的精确分割，为工业异常检测提供了有效的零样本解决方案。

### 翻译

最近，基础模型展现出的强大泛化能力为零样本异常分割任务带来了新的解决方案。然而，正确引导这些基础模型解决下游任务仍然是一个挑战。本文提出了一种用于工业异常检测中零样本异常分割任务的新型两阶段框架。该框架出色地利用了CLIP的强大异常定位能力和SAM的边界感知能力。1) 为缓解SAM倾向于目标分割的问题，我们提出了Co-Feature Point Prompt Generation (PPG)模块。该模块协同使用CLIP和SAM生成正负点提示，引导SAM专注于分割异常区域而非整个对象。2) 为进一步优化SAM的分割结果并缓解边界粗糙和孤立噪声，我们引入了Cascaded Prompts for SAM (CPS)模块。该模块采用与SAM轻量级解码器级联的混合提示，实现异常区域的精确分割。在多个数据集上的一致实验验证表明，我们的方法取得了最先进的零样本异常分割结果。特别值得注意的是，我们在Visa数据集上的表现，在F1-max和AP指标上分别比最先进方法高出10.3%和7.7%。


### 论文摘要

Recently, the powerful generalization ability exhibited by foundation models has brought forth new solutions for zero-shot anomaly segmentation tasks. However, guiding these foundation models correctly to address downstream tasks remains a challenge. This paper proposes a novel two-stage framework, for zero-shot anomaly segmentation tasks in industrial anomaly detection. This framework excellently leverages the powerful anomaly localization capability of CLIP and the boundary perception ability of SAM.(1) To mitigate SAM's inclination towards object segmentation, we propose the Co-Feature Point Prompt Generation (PPG) module. This module collaboratively utilizes CLIP and SAM to generate positive and negative point prompts, guiding SAM to focus on segmenting anomalous regions rather than the entire object. (2) To further optimize SAM's segmentation results and mitigate rough boundaries and isolated noise, we introduce the Cascaded Prompts for SAM (CPS) module. This module employs hybrid prompts cascaded with a lightweight decoder of SAM, achieving precise segmentation of anomalous regions. Across multiple datasets, consistent experimental validation demonstrates that our approach achieves state-of-the-art zero-shot anomaly segmentation results. Particularly noteworthy is our performance on the Visa dataset, where we outperform the state-of-the-art methods by 10.3\% and 7.7\% in terms of {$F_1$-max} and AP metrics, respectively.

---

## 163. Frequency Domain Unlocks New Perspectives for Abdominal Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2510.11005v1](http://arxiv.org/abs/2510.11005v1)

**作者:** Kai Han, Siqi Ma, Chengxuan Qian, Jun Chen, Chongwen Lyu, Yuqing Song, Zhe Liu

**发布时间:** 2025-10-13

### GPT解析

### 总结

本文提出了一种前景感知频谱分割(FASS)框架，用于解决医学图像中肿瘤和相邻正常组织分割的挑战，特别是在复杂、低对比度背景下。

### 背景

在医学图像中准确分割肿瘤和相邻正常组织对手术规划和肿瘤分期至关重要。基础模型在分割任务中表现良好，但在复杂、低对比度背景下往往难以聚焦前景区域，因为某些恶性肿瘤与正常器官相似，难以区分。

### 目的

开发一种能够在复杂条件下提高分割鲁棒性和精细结构识别能力的分割框架。

### 方法

FASS框架包含三个主要模块：1) 前景感知模块，增强背景与整个体积空间之间的区别；2) 基于小波变换的特征级频率增强模块，提取判别性高频特征以增强边界识别；3) 边缘约束模块，保持分割边界的几何连续性。

### 主要发现

在多个医学数据集上的实验表明，该框架在所有指标上表现优越，特别是在复杂条件下的鲁棒性和精细结构识别方面效果显著。

### 结论

该框架显著提高了低对比度图像的分割效果，为更多样化和复杂的医学成像场景中的应用奠定了基础。

### 翻译

医学图像中肿瘤和相邻正常组织的准确分割对于手术规划和肿瘤分期至关重要。尽管基础模型在分割任务中通常表现良好，但它们往往难以在复杂、低对比度背景下聚焦前景区域，因为某些恶性肿瘤与正常器官相似，使上下文区分复杂化。为解决这些挑战，我们提出了前景感知频谱分割(FASS)框架。首先，我们引入了前景感知模块来增强背景与整个体积空间之间的区别，使模型能够更有效地集中注意力在目标区域。其次，基于小波变换的特征级频率增强模块提取判别性高频特征，以增强边界识别和细节感知。最后，我们引入了边缘约束模块来保持分割边界的几何连续性。在多个医学数据集上的大量实验证明，所有指标上均表现出优越性能，验证了我们框架的有效性，特别是在复杂条件下的鲁棒性和精细结构识别方面。我们的框架显著提高了低对比度图像的分割效果，为更多样化和复杂的医学成像场景中的应用铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决低对比度腹部医学图像中肿瘤和正常组织的准确分割问题。这个问题在现实中非常重要，因为准确的肿瘤分割对手术计划和肿瘤分期至关重要，直接影响治疗效果和患者生存率。同时，手动标注非常耗时费力，需要大量临床专业知识，特别是在处理复杂解剖结构或模糊边界时。低对比度图像中肿瘤与周围组织灰度值相似，边界模糊，使得现有分割方法难以准确区分肿瘤和正常组织，导致分割结果不完整或边界断裂。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了低对比度医学图像分割的关键挑战，识别出现有方法在处理复杂背景和相似组织时的局限性。他们从频率域角度思考解决方案，利用频率增强来放大高频成分，使模型能更好地捕捉细微特征。设计过程中，作者借鉴了深度学习在医学图像分割中的应用经验，参考了两阶段策略和单阶段方法的优缺点，并利用了小波变换在图像处理中的成功经验。同时，他们整合了注意力机制和对抗训练等现有技术，但创新性地将它们结合起来，形成了前景感知模块、特征级频率增强模块和边缘约束模块三个互补组件，共同解决低对比度分割的挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用频率域信息增强来改善低对比度医学图像的分割效果，通过对抗训练使模型更专注于前景区域，并结合边缘约束确保分割结果的几何连续性。整体实现流程包括：1) 前景感知模块：通过对抗训练学习前景和背景特征的异质性，使模型能专注于目标区域；2) 特征级频率增强模块：利用小波变换将特征分解为不同频率成分，选择性增强判别性高频特征，提高边界识别和细节感知能力；3) 边缘约束模块：通过边界关键点集约束确保分割边界的几何连续性；4) 整体框架将这三个模块有机结合，通过端到端训练实现高质量的分割结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 前景感知模块：创新性地使用对抗训练策略最大化背景和输入图像特征间的分布差异，使模型能有效抵抗复杂背景干扰；2) 特征级频率增强模块：首次将小波变换与交叉注意力机制结合，选择性增强和利用判别性高频信息，减少噪声干扰；3) 边缘约束模块：将物理模型先验知识整合到深度学习框架中，确保分割边界的几何连续性；4) 整体框架设计：首次将频率域分析与前景感知、边缘约束相结合。相比之前的工作，这种方法能更有效地处理低对比度图像中的细微差异，选择性利用判别性特征而非盲目增强所有高频成分，并通过边缘约束保持边界完整性，整体性能显著优于现有方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于频率域分析的前景感知谱分割框架，通过结合前景感知、特征级频率增强和边缘约束三个创新模块，显著提高了低对比度腹部医学图像中肿瘤和器官的分割精度和边界连续性。'}


### 论文摘要

Accurate segmentation of tumors and adjacent normal tissues in medical images is essential for surgical planning and tumor staging. Although foundation models generally perform well in segmentation tasks, they often struggle to focus on foreground areas in complex, low-contrast backgrounds, where some malignant tumors closely resemble normal organs, complicating contextual differentiation. To address these challenges, we propose the Foreground-Aware Spectrum Segmentation (FASS) framework. First, we introduce a foreground-aware module to amplify the distinction between background and the entire volume space, allowing the model to concentrate more effectively on target areas. Next, a feature-level frequency enhancement module, based on wavelet transform, extracts discriminative high-frequency features to enhance boundary recognition and detail perception. Eventually, we introduce an edge constraint module to preserve geometric continuity in segmentation boundaries. Extensive experiments on multiple medical datasets demonstrate superior performance across all metrics, validating the effectiveness of our framework, particularly in robustness under complex conditions and fine structure recognition. Our framework significantly enhances segmentation of low-contrast images, paving the way for applications in more diverse and complex medical imaging scenarios.

---

## 164. Deep Learning in Astrophysics

**论文链接:** [http://arxiv.org/abs/2510.10713v1](http://arxiv.org/abs/2510.10713v1)

**作者:** Yuan-Sen Ting

**发布时间:** 2025-10-12

**备注:** Manuscript submitted to Annual Review of Astronomy and Astrophysics  for Volume 64. This is the authors' version. Revisions and the final version  will be available at https://www.annualreviews.org/content/journals/astro

### GPT解析

### 总结

深度学习为天文学提供了多样化视角，通过将物理对称性、守恒定律和微分方程编码到网络架构中，扩展了数据分析工具包。尽管面临未标记数据庞大而确认样本稀少的挑战，深度学习仍通过架构设计整合领域知识，为天文学提供了新方法。

### 背景

深度学习在天文学中产生了多样化视角，支持者和怀疑者之间的持续讨论促使了这篇综述。天文学提供了独特机会，可以通过编码物理对称性、守恒定律和微分方程直接到架构中，创建能推广到训练数据之外的模型。

### 目的

检查神经网络如何补充经典统计学，扩展现代调查的数据分析工具包；评估深度学习方法提供真正进步的领域和需要仔细审查的主张；展示如何通过架构设计将领域知识整合到深度学习中。

### 方法

通过将物理对称性、守恒定律和微分方程直接编码到网络架构中，创建能推广到训练数据之外的模型；通过架构设计将领域知识整合到模型中，使模型朝着物理上有意义的解决方案发展；评估深度学习在天文学不同应用领域的效果。

### 主要发现

神经架构通过将物理对称性和守恒定律编码到网络结构中，克服了可扩展性、表达能力和数据效率之间的权衡；基于模拟的推理和异常检测能够从复杂、非高斯分布中提取信息，使宇宙学场级分析和罕见现象的系统性发现成为可能；多尺度神经建模弥合了天文模拟中的分辨率差距，从高保真运行中学习有效的次网格物理；强化学习用于望远镜操作，基础模型从最少示例中学习，大型语言模型代理用于研究自动化等新兴范式显示出潜力。

### 结论

深度学习通过将领域知识整合到架构设计中，为天文学提供了新的数据分析工具，能够处理大规模数据并从有限标记数据中学习。尽管面临数据挑战，但深度学习方法在多个天文应用领域显示出实质性进步，特别是在处理复杂分布、弥合分辨率差距和研究自动化方面。

### 翻译

深度学习在天文学中产生了多样化视角，支持者和怀疑者之间的持续讨论促使了这篇综述。我们检查了神经网络如何补充经典统计学，扩展了现代调查的数据分析工具包。天文学提供了独特机会，可以通过编码物理对称性、守恒定律和微分方程直接到架构中，创建能推广到训练数据之外的模型。然而挑战仍然存在，因为未标记观测数据数量达数十亿，而具有已知属性的确认样本仍然稀少且昂贵。这篇综述展示了深度学习如何通过架构设计整合领域知识，内置假设引导模型朝向物理上有意义的解决方案。我们评估了这些方法在哪些方面提供了真正进步，以及哪些主张需要仔细审查。神经架构通过将物理对称性和守恒定律编码到网络结构中，克服了可扩展性、表达能力和数据效率之间的权衡，能够从有限的标记数据中学习。基于模拟的推理和异常检测从复杂、非高斯分布中提取信息，在这些分布中分析似然失败，使宇宙学场级分析和罕见现象的系统性发现成为可能。多尺度神经建模弥合了天文模拟中的分辨率差距，从昂贵的高保真运行中学习有效的次网格物理，以增强直接计算仍然不可行的大体积计算。新兴范式——用于望远镜操作的强化学习，从最少示例中学习的基础模型，以及用于研究自动化的大型语言模型代理——显示出潜力，尽管在天文学应用中仍在发展中。


### 论文摘要

Deep learning has generated diverse perspectives in astronomy, with ongoing discussions between proponents and skeptics motivating this review. We examine how neural networks complement classical statistics, extending our data analytical toolkit for modern surveys. Astronomy offers unique opportunities through encoding physical symmetries, conservation laws, and differential equations directly into architectures, creating models that generalize beyond training data. Yet challenges persist as unlabeled observations number in billions while confirmed examples with known properties remain scarce and expensive. This review demonstrates how deep learning incorporates domain knowledge through architectural design, with built-in assumptions guiding models toward physically meaningful solutions. We evaluate where these methods offer genuine advances versus claims requiring careful scrutiny. - Neural architectures overcome trade-offs between scalability, expressivity, and data efficiency by encoding physical symmetries and conservation laws into network structure, enabling learning from limited labeled data. - Simulation-based inference and anomaly detection extract information from complex, non-Gaussian distributions where analytical likelihoods fail, enabling field-level cosmological analysis and systematic discovery of rare phenomena. - Multi-scale neural modeling bridges resolution gaps in astronomical simulations, learning effective subgrid physics from expensive high-fidelity runs to enhance large-volume calculations where direct computation remains prohibitive. - Emerging paradigms-reinforcement learning for telescope operations, foundation models learning from minimal examples, and large language model agents for research automation-show promise though are still developing in astronomical applications.

---

## 165. Scalable Face Security Vision Foundation Model for Deepfake, Diffusion, and Spoofing Detection

**论文链接:** [http://arxiv.org/abs/2510.10663v1](http://arxiv.org/abs/2510.10663v1)

**作者:** Gaojian Wang, Feng Lin, Tong Wu, Zhisheng Yan, Kui Ren

**发布时间:** 2025-10-12

**备注:** 18 pages, 9 figures, project page:  https://fsfm-3c.github.io/fsvfm.html

### GPT解析

### 总结

本研究提出FS-VFM，一个可扩展的自监督预训练框架，用于学习真实人脸图像的基本表示，通过结合掩码图像建模和实例判别，提高人脸安全任务的泛化能力。

### 背景

如何利用大量未标记的真实人脸图像学习鲁棒且可迁移的人脸表示，以提升各种人脸安全任务的泛化能力。

### 目的

提出一个可扩展的自监督预训练框架，学习真实人脸图像的基本表示，并在多种人脸安全任务上实现更好的泛化性能。

### 方法

引入3C学习目标，结合掩码图像建模(MIM)和实例判别(ID)；设计CRFR-P掩码策略；提出可靠的自蒸馏机制建立局部到全局的对应关系；使用普通视觉变压器(ViTs)作为下游任务的通用视觉基础模型；提出FS-Adapter轻量级即插即用瓶颈。

### 主要发现

在11个公共基准测试上，FS-VFM在各种视觉基础模型中泛化能力更好，包括自然和人脸领域；在不同监督范式和ViT规模上都表现出色；甚至优于最先进的任务特定方法；FS-Adapter提供了出色的效率-性能权衡。

### 结论

FS-VFM框架能够有效学习人脸表示，并在各种人脸安全任务上实现更好的泛化性能，代码和模型已公开可用。

### 翻译

利用大量未标记的真实人脸，我们如何学习鲁棒且可迁移的人脸表示来提高各种人脸安全任务的泛化能力？我们首次尝试并提出FS-VFM，一个可扩展的自监督预训练框架，用于学习真实人脸图像的基本表示。我们引入三个学习目标，即3C，协同结合掩码图像建模(MIM)和实例判别(ID)，使FS-VFM能够编码真实人脸的局部模式和全局语义。具体而言，我们为MIM制定了各种面部掩码策略，并设计了一种简单而有效的CRFR-P掩码，明确提示模型追求有意义的区域内一致性和挑战性的区域间连贯性。我们提出了一个可靠的自蒸馏机制，将MIM与ID无缝耦合，建立底层局部到全局的对应关系。预训练后，普通视觉变压器(ViTs)作为下游人脸安全任务的通用视觉基础模型：跨数据集深度伪造检测、跨域人脸防欺骗和未见扩散人脸取证。为了高效迁移预训练的FS-VFM，我们进一步提出FS-Adapter，一种新颖的真实锚点对比目标的轻量级即插即用瓶颈，位于冻结骨干网络之上。在11个公共基准上的广泛实验表明，我们的FS-VFM比各种视觉基础模型泛化能力更好，包括自然和人脸领域，完全、弱和自监督范式，小型、基础和大型的ViT规模，甚至优于最先进的任务特定方法，而FS-Adapter提供了出色的效率-性能权衡。代码和模型可在https://fsfm-3c.github.io/fsvfm.html获取。


### 论文摘要

With abundant, unlabeled real faces, how can we learn robust and transferable facial representations to boost generalization across various face security tasks? We make the first attempt and propose FS-VFM, a scalable self-supervised pre-training framework, to learn fundamental representations of real face images. We introduce three learning objectives, namely 3C, that synergize masked image modeling (MIM) and instance discrimination (ID), empowering FS-VFM to encode both local patterns and global semantics of real faces. Specifically, we formulate various facial masking strategies for MIM and devise a simple yet effective CRFR-P masking, which explicitly prompts the model to pursue meaningful intra-region Consistency and challenging inter-region Coherency. We present a reliable self-distillation mechanism that seamlessly couples MIM with ID to establish underlying local-to-global Correspondence. After pre-training, vanilla vision transformers (ViTs) serve as universal Vision Foundation Models for downstream Face Security tasks: cross-dataset deepfake detection, cross-domain face anti-spoofing, and unseen diffusion facial forensics. To efficiently transfer the pre-trained FS-VFM, we further propose FS-Adapter, a lightweight plug-and-play bottleneck atop the frozen backbone with a novel real-anchor contrastive objective. Extensive experiments on 11 public benchmarks demonstrate that our FS-VFM consistently generalizes better than diverse VFMs, spanning natural and facial domains, fully, weakly, and self-supervised paradigms, small, base, and large ViT scales, and even outperforms SOTA task-specific methods, while FS-Adapter offers an excellent efficiency-performance trade-off. The code and models are available on https://fsfm-3c.github.io/fsvfm.html.

---

## 166. Equipping Vision Foundation Model with Mixture of Experts for Out-of-Distribution Detection

**论文链接:** [http://arxiv.org/abs/2510.10584v1](http://arxiv.org/abs/2510.10584v1)

**作者:** Shizhen Zhao, Jiahui Liu, Xin Wen, Haoru Tan, Xiaojuan Qi

**发布时间:** 2025-10-12

### GPT解析

### 总结

本研究系统性地探索了预训练视觉基础模型在分布外检测任务中的应用，发现DINOv2模型无需微调即可提供高度判别性的特征空间，并提出了MoFE模块和Dynamic-β Mixup策略来解决语义空间较大场景中的性能问题。

### 背景

预训练视觉基础模型已改变众多计算机视觉任务，它们在学习判别性和可泛化特征方面能力强大，这些特征对分布外检测至关重要，但它们对这一任务的影响尚未被充分探索。

### 目的

系统性地研究代表性的视觉基础模型在分布外检测任务中的应用和性能。

### 方法

研究预训练DINOv2模型在分布外检测中的表现；探索在领域内数据上微调基础模型如何增强分布外检测；提出Mixture of Feature Experts (MoFE)模块将特征划分为子空间；引入Dynamic-β Mixup策略从动态beta分布中采样插值权重。

### 主要发现

预训练的DINOv2模型无需在领域内数据上微调就能提供高度判别性的特征空间，性能媲美现有最先进方法；在语义空间较大的场景中，视觉基础模型的性能仍然不令人满意，这是因为类别数量增加导致决策边界复杂化，使优化过程变得复杂。

### 结论

MoFE模块和Dynamic-β Mixup策略能有效捕获复杂数据分布并细化决策边界，大量实验证明该方法显著优于基线方法。

### 翻译

预训练视觉基础模型已改变许多计算机视觉任务。尽管它们在学习对分布外检测至关重要的判别性和可泛化特征方面能力强大，它们对该任务的影响仍未被充分探索。受此差距启发，我们系统性地研究了代表性的视觉基础模型用于分布外检测。我们的发现表明，预训练的DINOv2模型即使在领域内数据上未进行微调，也能为分布外检测提供高度判别性的特征空间，实现与现有最先进方法相当的性能，而无需复杂设计。除此之外，我们探索了在领域内数据上微调基础模型如何增强分布外检测。然而，我们观察到在语义空间较大的场景中，视觉基础模型的性能仍然不令人满意。这是因为随着类别数量增加，决策边界的复杂度提高，使优化过程变得复杂。为缓解这一问题，我们提出了特征专家混合（MoFE）模块，该模块将特征划分为子空间，有效捕获复杂数据分布并细化决策边界。此外，我们引入了动态-β混合策略，从动态beta分布中采样插值权重。这使模型能够适应不同类别间的不同学习难度，提升更具挑战性类别的特征学习。大量实验证明了我们方法的有效性，显著优于基线方法。


### 论文摘要

Pre-trained vision foundation models have transformed many computer vision tasks. Despite their strong ability to learn discriminative and generalizable features crucial for out-of-distribution (OOD) detection, their impact on this task remains underexplored. Motivated by this gap, we systematically investigate representative vision foundation models for OOD detection. Our findings reveal that a pre-trained DINOv2 model, even without fine-tuning on in-domain (ID) data, naturally provides a highly discriminative feature space for OOD detection, achieving performance comparable to existing state-of-the-art methods without requiring complex designs. Beyond this, we explore how fine-tuning foundation models on in-domain (ID) data can enhance OOD detection. However, we observe that the performance of vision foundation models remains unsatisfactory in scenarios with a large semantic space. This is due to the increased complexity of decision boundaries as the number of categories grows, which complicates the optimization process. To mitigate this, we propose the Mixture of Feature Experts (MoFE) module, which partitions features into subspaces, effectively capturing complex data distributions and refining decision boundaries. Further, we introduce a Dynamic-$\beta$ Mixup strategy, which samples interpolation weights from a dynamic beta distribution. This adapts to varying levels of learning difficulty across categories, improving feature learning for more challenging categories. Extensive experiments demonstrate the effectiveness of our approach, significantly outperforming baseline methods.

---

## 167. Post-TIPS Prediction via Multimodal Interaction: A Multi-Center Dataset and Framework for Survival, Complication, and Portal Pressure Assessment

**论文链接:** [http://arxiv.org/abs/2510.10464v1](http://arxiv.org/abs/2510.10464v1)

**作者:** Junhao Dong, Dejia Liu, Ruiqi Ding, Zongxing Chen, Yingjie Huang, Zhu Meng, Jianbo Zhao, Zhicheng Zhao, Fei Su

**发布时间:** 2025-10-12

**备注:** 81 pages, 13 figures

### GPT解析

### 总结

本研究提出了首个用于TIPS预后的公开多中心数据集MultiTIPS，并基于此开发了一种新的多模态预后框架，包含双选项分割、多模态交互和多任务预测三个核心模块，解决了当前研究面临的ROI标注量大、单模态方法可靠性差和单终点评估不完整等挑战。

### 背景

TIPS是治疗门静脉高压的既定方法，但存在生存结果差异大和频繁出现明显肝性脑病(OHE)的问题，需要准确的术前预后模型。

### 目的

开发一种准确、可靠的TIPS预后模型，解决当前研究中存在的ROI标注量大、单模态方法可靠性差和单终点评估不完整等挑战，并提供公开数据集促进该领域研究。

### 方法

提出MultiTIPS数据集和一种新的多模态预后框架，该框架包含三个核心模块：(1)双选项分割：结合半监督和基础模型实现有限标注下的鲁棒ROI分割；(2)多模态交互：引入MGRA、POD和CGPE技术实现跨模态特征交互；(3)多任务预测：使用分阶段训练策略同时优化生存、PPG和OHE预测。

### 主要发现

在MultiTIPS上的大量实验表明，所提出的方法优于最先进的方法，具有强大的跨域泛化性和可解释性，显示出临床应用的潜力。

### 结论

MultiTIPS数据集和所提出的多模态预后框架为TIPS预后评估提供了有效解决方案，有望在临床实践中应用。

### 翻译

经颈静脉肝内门体分流术(TIPS)是治疗门静脉高压的既定方法，但提供不同的生存结果和频繁明显的肝性脑病(OHE)，表明需要准确的术前预后建模。当前研究通常从术前CT图像或临床特征构建机器学习模型，但面临三个关键挑战：(1)劳动密集型的感兴趣区域(ROI)标注，(2)单模态方法的可靠性和泛化能力差，(3)单终点预测评估不完整。此外，缺乏公开可访问的数据集限制了该领域的研究。因此，我们提出了MultiTIPS，这是首个用于TIPS预后的公共多中心数据集，并基于它提出了一个新的多模态预后框架。该框架包含三个核心模块：(1)双选项分割，结合半监督和基于基础模型的流程，实现有限标注下的鲁棒ROI分割并促进后续特征提取；(2)多模态交互，引入多粒度放射组学注意力(MGRA)、渐进正交解耦(POD)和临床引导预后增强(CGPE)技术，实现跨模态特征交互和互补表示集成，从而提高模型准确性和鲁棒性；(3)多任务预测，使用分阶段训练策略对生存、门静脉压力梯度(PPG)和OHE预测进行稳定优化，实现全面预后评估。在MultiTIPS上的大量实验证明了所提出方法优于最先进的方法，同时具有强大的跨域泛化性和可解释性，表明其临床应用前景。该数据集和代码是公开可用的。


### 论文摘要

Transjugular intrahepatic portosystemic shunt (TIPS) is an established procedure for portal hypertension, but provides variable survival outcomes and frequent overt hepatic encephalopathy (OHE), indicating the necessity of accurate preoperative prognostic modeling. Current studies typically build machine learning models from preoperative CT images or clinical characteristics, but face three key challenges: (1) labor-intensive region-of-interest (ROI) annotation, (2) poor reliability and generalizability of unimodal methods, and (3) incomplete assessment from single-endpoint prediction. Moreover, the lack of publicly accessible datasets constrains research in this field. Therefore, we present MultiTIPS, the first public multi-center dataset for TIPS prognosis, and propose a novel multimodal prognostic framework based on it. The framework comprises three core modules: (1) dual-option segmentation, which integrates semi-supervised and foundation model-based pipelines to achieve robust ROI segmentation with limited annotations and facilitate subsequent feature extraction; (2) multimodal interaction, where three techniques, multi-grained radiomics attention (MGRA), progressive orthogonal disentanglement (POD), and clinically guided prognostic enhancement (CGPE), are introduced to enable cross-modal feature interaction and complementary representation integration, thus improving model accuracy and robustness; and (3) multi-task prediction, where a staged training strategy is used to perform stable optimization of survival, portal pressure gradient (PPG), and OHE prediction for comprehensive prognostic assessment. Extensive experiments on MultiTIPS demonstrate the superiority of the proposed method over state-of-the-art approaches, along with strong cross-domain generalization and interpretability, indicating its promise for clinical application. The dataset and code are available.

---

## 168. Vision4PPG: Emergent PPG Analysis Capability of Vision Foundation Models for Vital Signs like Blood Pressure

**论文链接:** [http://arxiv.org/abs/2510.10366v1](http://arxiv.org/abs/2510.10366v1)

**作者:** Saurabh Kataria, Ayca Ermis, Lovely Yeswanth Panchumarthi, Minxiao Wang, Xiao Hu

**发布时间:** 2025-10-11

**备注:** BHI abstract extended

### GPT解析

### 总结

该研究探索了使用视觉基础模型(VFM)处理光电容积脉搏波描记术(PPG)信号的可能性，发现将一维PPG信号转换为二维图像表示后，视觉模型在血压估计等多种生理任务上达到最先进性能，并展示了良好的泛化能力。

### 背景

光电容积脉搏波描记术(PPG)传感器在可穿戴和临床设备中能够以非侵入式和实时方式提供有价值的生理信息。目前通常使用专门的基础模型或重新利用的时间序列基础模型来基准化生理任务。

### 目的

研究视觉基础模型(VFM)在PPG信号处理中的应用潜力，评估其在各种生理任务中的性能，并与现有时间序列基础模型进行比较。

### 方法

将一维PPG信号转换为二维图像表示，如短时傅里叶变换(STFT)，然后使用最新的视觉基础模型(如DINOv3和SIGLIP-2)进行微调。采用参数高效微调(PEFT)技术提高计算效率。

### 主要发现

视觉基础模型(VFM)在PPG信号处理中表现优异，特别是在血压估计任务上达到了最先进(SOTA)的性能。该方法在其他生命体征和血液实验室测量任务中也取得了有希望的结果，并且可以推广到STFT相位和递归图等其他2D输入表示。

### 结论

提出的Vision4PPG方法解锁了一类新的基础模型用于PPG处理，能够实现最先进性能并具有良好的泛化能力。这些工具为临床科学家提供了计算效率高的新选择，通过参数高效微调技术实现。

### 翻译

光电容积脉搏波描记术(PPG)传感器在可穿戴和临床设备中以非侵入式和实时方式提供有价值的生理洞见。专门的基础模型(FM)或重新利用的时间序列FM被用于基准化生理任务。我们对微调FM的实验表明，视觉FM(VFM)也可用于此目的，事实上在许多任务上(特别是血压估计)出乎意料地达到了最先进(SOTA)性能。我们通过简单地将一维PPG信号转换为类图像的二维表示(如短时傅里叶变换STFT)来利用VFMs。使用最新的VFMs(如DINOv3和SIGLIP-2)，我们在其他生命体征和血液实验室测量任务中也取得了有希望的性能。我们的提案Vision4PPG解锁了一类新的FM，实现了SOTA性能，并能显著推广到其他2D输入表示，包括STFT相位和递归图。我们的工作通过进行全面研究、将其与最先进的时间序列FM进行比较，并在六个额外任务上报告结果，改进了先前关于视觉模型用于PPG的研究。因此，我们为临床科学家提供了一套新的强大工具，由于参数高效微调(PEFT)技术，这些工具也具有计算效率高的特点。


### 论文摘要

Photoplethysmography (PPG) sensor in wearable and clinical devices provides valuable physiological insights in a non-invasive and real-time fashion. Specialized Foundation Models (FM) or repurposed time-series FMs are used to benchmark physiological tasks. Our experiments with fine-tuning FMs reveal that Vision FM (VFM) can also be utilized for this purpose and, in fact, surprisingly leads to state-of-the-art (SOTA) performance on many tasks, notably blood pressure estimation. We leverage VFMs by simply transforming one-dimensional PPG signals into image-like two-dimensional representations, such as the Short-Time Fourier transform (STFT). Using the latest VFMs, such as DINOv3 and SIGLIP-2, we achieve promising performance on other vital signs and blood lab measurement tasks as well. Our proposal, Vision4PPG, unlocks a new class of FMs to achieve SOTA performance with notable generalization to other 2D input representations, including STFT phase and recurrence plots. Our work improves upon prior investigations of vision models for PPG by conducting a comprehensive study, comparing them to state-of-the-art time-series FMs, and demonstrating the general PPG processing ability by reporting results on six additional tasks. Thus, we provide clinician-scientists with a new set of powerful tools that is also computationally efficient, thanks to Parameter-Efficient Fine-Tuning (PEFT) techniques.

---

## 169. End-to-end Automatic Speech Recognition and Speech Translation: Integration of Speech Foundational Models and LLMs

**论文链接:** [http://arxiv.org/abs/2510.10329v1](http://arxiv.org/abs/2510.10329v1)

**作者:** Nam Luu, Ondřej Bojar

**发布时间:** 2025-10-11

### GPT解析

### 总结

这项研究探索了结合预训练语音编码器和大型语言模型的端到端架构，用于同时进行语音识别和语音翻译，在英语到德语任务上取得了显著成果。

### 背景

语音翻译是将一种语言的语音信号转换为另一种语言对应文本的机器翻译任务，存在传统级联方法和端到端方法两种不同途径。

### 目的

探索一种结合预训练语音编码器和大型语言模型的端到端架构，实现同时进行自动语音识别和语音翻译。

### 方法

使用预训练的语音编码器和大型语言模型构建端到端架构，并在英语到德语的语言对上进行实验。

### 主要发现

最佳模型不仅比SeamlessM4T大型基础端到端多模态翻译模型取得更好的翻译结果，还能匹配使用Whisper和NLLB的级联系统性能，在COMET-DA22指标上获得高达8%的分数提升。

### 结论

结合预训练语音编码器和大型语言模型的端到端架构在语音翻译任务上表现出色，超越了现有模型的性能。

### 翻译

语音翻译是一种机器翻译任务，涉及将一种语言的语音信号转换为另一种语言的对应文本；该任务有两种不同的方法，即传统的级联方法和最近的端到端方法。本文探索了结合预训练语音编码器和大型语言模型的端到端架构，用于同时执行自动语音识别和语音翻译。英语到德语语言对的实验表明，我们的最佳模型不仅能够比SeamlessM4T（大型基础端到端多模态翻译模型）获得更好的翻译结果，还能匹配使用Whisper和NLLB的级联系统性能，在COMET-DA22指标上获得高达8%的分数提升。


### 论文摘要

Speech Translation (ST) is a machine translation task that involves converting speech signals from one language to the corresponding text in another language; this task has two different approaches, namely the traditional cascade and the more recent end-to-end. This paper explores a combined end-to-end architecture of pre-trained speech encoders and Large Language Models (LLMs) for performing both Automatic Speech Recognition (ASR) and ST simultaneously. Experiments with the English-to-German language pair show that our best model not only can achieve better translation results than SeamlessM4T, a large foundational end-to-end, multi-modal translation model, but can also match the performance of a cascaded system with Whisper and NLLB, with up to a score gain of 8% in $\text{COMET}^{\text{DA}}_{22}$ metric.

---

## 170. From Generic to Specialized: A Subspecialty Diagnostic System Powered by Self-Supervised Learning for Cervical Histopathology

**论文链接:** [http://arxiv.org/abs/2510.10196v1](http://arxiv.org/abs/2510.10196v1)

**作者:** Yizhi Wang, Li Chen, Qiang Huang, Tian Guan, Xi Deng, Zhiyuan Shen, Jiawen Li, Xinrui Chen, Bin Hu, Xitong Ling, Taojie Zhu, Zirui Huang, Deshui Yu, Yan Liu, Jiurun Chen, Lianghui Zhu, Qiming He, Yiqing Liu, Diwei Shi, Hanzhong Liu, Junbo Hu, Hongyi Gao, Zhen Song, Xilong Zhao, Chao He, Ming Zhao, Yonghong He

**发布时间:** 2025-10-11

**备注:** 32 pages, 6 figures

### GPT解析

### 总结

研究团队开发了名为CerS-Path的宫颈癌亚专科病理诊断系统，通过两个预训练阶段构建，支持八种诊断功能，在前瞻性测试中达到99.38%的筛查敏感性，展示了优秀的泛化能力和临床应用潜力。

### 背景

宫颈癌是一种主要恶性肿瘤，需要广泛复杂的组织病理学评估。现有深度学习模型缺乏准确性和泛化能力，通用基础模型在捕获亚专科特定特征和任务适应性方面存在局限。

### 目的

开发一个专门针对宫颈癌病理的诊断系统，提高宫颈癌病理诊断的准确性和泛化能力，解决现有模型的局限性。

### 方法

开发CerS-Path诊断系统，通过两个协同预训练阶段：1)自监督学习使用约1.9亿个组织块构建宫颈特异性特征提取器；2)多模态增强使用250万对图像-文本对进行增强，然后整合多个下游诊断功能。

### 主要发现

CerS-Path在范围和临床适用性方面超越了以前的基础模型，全面评估显示在宫颈癌病理方面取得显著进展，在五个中心对3173例病例的前瞻性测试中保持99.38%的筛查敏感性和优秀泛化能力。

### 结论

CerS-Path在亚专科诊断转化和宫颈癌筛查方面具有潜力，代表了宫颈癌病理诊断的重要进步。

### 翻译

宫颈癌仍然是一种主要恶性肿瘤，需要广泛而复杂的组织病理学评估和全面的支持工具。尽管深度学习显示出前景，但这些模型仍然缺乏准确性和泛化能力。通用基础模型提供了更广泛的覆盖范围，但在捕获亚专科特定特征和任务适应性方面仍然存在局限。我们引入了宫颈亚专科病理学(CerS-Path)诊断系统，通过两个协同的预训练阶段开发：自监督学习来自约14万张幻灯片的1.9亿个组织块以构建宫颈特异性特征提取器，以及使用250万对图像-文本对进行多模态增强，随后整合多个下游诊断功能。支持包括罕见癌症分类和多模态问答在内的八种诊断功能，CerS-Path在范围和临床适用性方面超越了以前的基础模型。全面评估显示在宫颈癌病理方面取得了显著进展，在五个中心对3173例病例的前瞻性测试中保持了99.38%的筛查敏感性和优秀的泛化能力，突显了其在亚专科诊断转化和宫颈癌筛查方面的潜力。


### 论文摘要

Cervical cancer remains a major malignancy, necessitating extensive and complex histopathological assessments and comprehensive support tools. Although deep learning shows promise, these models still lack accuracy and generalizability. General foundation models offer a broader reach but remain limited in capturing subspecialty-specific features and task adaptability. We introduce the Cervical Subspecialty Pathology (CerS-Path) diagnostic system, developed through two synergistic pretraining stages: self-supervised learning on approximately 190 million tissue patches from 140,000 slides to build a cervical-specific feature extractor, and multimodal enhancement with 2.5 million image-text pairs, followed by integration with multiple downstream diagnostic functions. Supporting eight diagnostic functions, including rare cancer classification and multimodal Q&A, CerS-Path surpasses prior foundation models in scope and clinical applicability. Comprehensive evaluations demonstrate a significant advance in cervical pathology, with prospective testing on 3,173 cases across five centers maintaining 99.38% screening sensitivity and excellent generalizability, highlighting its potential for subspecialty diagnostic translation and cervical cancer screening.

---

## 171. SparseUWSeg: Active Sparse Point-Label Augmentation for Underwater Semantic Segmentation

**论文链接:** [http://arxiv.org/abs/2510.10163v1](http://arxiv.org/abs/2510.10163v1)

**作者:** César Borja, Carlos Plou, Rubén Martinez-Cantín, Ana C. Murillo

**发布时间:** 2025-10-11

### GPT解析

### 总结

SparseUWSeg是一个新型框架，通过主动采样策略和混合方法解决水下图像语义分割中稀疏点标注的挑战，实现了比现有方法更高的分割精度。

### 背景

语义分割对自动化水下图像分析和生态监测至关重要，但细粒度的水下场景分析仍是开放问题，获取密集专家标注标签成本高昂。

### 目的

解决水下图像语义分割中稀疏点标注的选择和传播问题，提高分割模型的性能。

### 方法

SparseUWSeg采用主动采样策略指导标注者选择有价值的点，并结合SAM2和基于超像素方法的混合技术传播稀疏标签。

### 主要发现

在两个不同水下数据集上，SparseUWSeg相比最先进方法实现了最高5%的mIoU提升。

### 结论

SparseUWSeg框架和其集成的交互式标注工具使生态研究人员能够高效利用基础模型和计算机视觉技术生成高质量分割掩模。

### 翻译

语义分割对自动化水下图像分析和生态监测至关重要。不幸的是，即使对于最先进的分割模型，细粒度的水下场景分析仍然是一个开放问题。获取密集的、专家标注的分割标签成本很高，阻碍了该领域模型的监督学习。虽然稀疏点标签更容易获取，但它们在标注哪些点和如何传播稀疏信息方面带来了挑战。我们提出了SparseUWSeg，一个解决这两个问题的新型框架。SparseUWSeg采用主动采样策略指导标注者，最大化其点标签的价值。然后，它使用结合SAM2和基于超像素方法优点的混合方法传播这些稀疏标签。在两个不同水下数据集上的实验证明了SparseUWSeg相比最先进方法的优势，相比D+NN方法实现了最高5%的mIoU提升。我们的主要贡献是设计并发布了一个简单但有效的交互式标注工具，整合了我们的算法。它使生态研究人员能够利用基础模型和计算机视觉高效生成高质量的分割掩模来处理他们的数据。


### 论文摘要

Semantic segmentation is essential to automate underwater imagery analysis with ecology monitoring purposes. Unfortunately, fine grained underwater scene analysis is still an open problem even for top performing segmentation models. The high cost of obtaining dense, expert-annotated, segmentation labels hinders the supervision of models in this domain. While sparse point-labels are easier to obtain, they introduce challenges regarding which points to annotate and how to propagate the sparse information. We present SparseUWSeg, a novel framework that addresses both issues. SparseUWSeg employs an active sampling strategy to guide annotators, maximizing the value of their point labels. Then, it propagates these sparse labels with a hybrid approach leverages both the best of SAM2 and superpixel-based methods. Experiments on two diverse underwater datasets demonstrate the benefits of SparseUWSeg over state-of-the-art approaches, achieving up to +5\% mIoU over D+NN. Our main contribution is the design and release of a simple but effective interactive annotation tool, integrating our algorithms. It enables ecology researchers to leverage foundation models and computer vision to efficiently generate high-quality segmentation masks to process their data.

---

## 172. Tracking the Spatiotemporal Evolution of Landslide Scars Using a Vision Foundation Model: A Novel and Universal Framework

**论文链接:** [http://arxiv.org/abs/2510.10084v1](http://arxiv.org/abs/2510.10084v1)

**作者:** Meijun Zhou, Gang Mei, Zhengjing Ma, Nengxiong Xu, Jianbing Peng

**发布时间:** 2025-10-11

### GPT解析

### 总结

该研究提出了一种基于视觉基础模型的新框架，能够通过将离散遥感图像转换为连续视频序列，实现大规模滑坡疤痕时空演化的连续追踪，为滑坡早期预警和灾害评估提供了有效工具。

### 背景

追踪大规模滑坡疤痕的时空演化对于理解演化机制和破坏前兆、实现有效预警至关重要。然而，现有研究多关注单阶段或破坏前后的双阶段滑坡识别，难以追踪滑坡疤痕的时空演化过程。

### 目的

解决现有方法难以追踪滑坡疤痕时空演化的问题，提出一个新框架用于追踪大规模滑坡疤痕的时空演化。

### 方法

使用视觉基础模型，将离散的光学遥感图像重建为连续的视频序列，使专为视频分割开发的视觉基础模型可用于追踪滑坡疤痕演化。该框架在知识引导、自动传播和交互式精炼的范式中运行，确保滑坡疤痕的连续和准确识别。

### 主要发现

该框架通过白格滑坡和色拉滑坡(2017-2025)两个案例得到验证，能够连续追踪滑坡疤痕，捕捉对预警至关重要的破坏前兆，以及对评估次生灾害和长期稳定性至关重要的破坏后演化。

### 结论

所提出的框架为滑坡疤痕的时空演化追踪提供了有效方法，有助于早期预警和灾害评估。

### 翻译

追踪大规模滑坡疤痕的时空演化对于理解演化机制和破坏前兆、实现有效预警至关重要。然而，大多数现有研究只关注单阶段或破坏前后的双阶段滑坡识别。虽然这些方法能够确定破坏后的滑坡边界，但难以追踪滑坡疤痕的时空演化。为解决这一问题，本研究提出了一种新的通用框架，使用视觉基础模型追踪大规模滑坡疤痕的时空演化。该框架的关键思路是将离散的光学遥感图像重建为连续的视频序列。这种转换使得专为视频分割开发的视觉基础模型可以用于追踪滑坡疤痕的演化。该框架在知识引导、自动传播和交互式精炼的范式中运行，以确保滑坡疤痕的连续和准确识别。该框架已通过两个代表性案例的应用得到验证：破坏后的白格滑坡和活跃的色拉滑坡(2017-2025)。结果表明，所提出的框架能够连续追踪滑坡疤痕，捕捉对预警至关重要的破坏前兆，以及对评估次生灾害和长期稳定性至关重要的破坏后演化。


### 论文摘要

Tracking the spatiotemporal evolution of large-scale landslide scars is critical for understanding the evolution mechanisms and failure precursors, enabling effective early-warning. However, most existing studies have focused on single-phase or pre- and post-failure dual-phase landslide identification. Although these approaches delineate post-failure landslide boundaries, it is challenging to track the spatiotemporal evolution of landslide scars. To address this problem, this study proposes a novel and universal framework for tracking the spatiotemporal evolution of large-scale landslide scars using a vision foundation model. The key idea behind the proposed framework is to reconstruct discrete optical remote sensing images into a continuous video sequence. This transformation enables a vision foundation model, which is developed for video segmentation, to be used for tracking the evolution of landslide scars. The proposed framework operates within a knowledge-guided, auto-propagation, and interactive refinement paradigm to ensure the continuous and accurate identification of landslide scars. The proposed framework was validated through application to two representative cases: the post-failure Baige landslide and the active Sela landslide (2017-2025). Results indicate that the proposed framework enables continuous tracking of landslide scars, capturing both failure precursors critical for early warning and post-failure evolution essential for assessing secondary hazards and long-term stability.

---

