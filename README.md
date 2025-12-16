# 今日论文推荐 - 2025-12-16

共 112 篇论文

---

## 1. LitePT: Lighter Yet Stronger Point Transformer

**论文链接:** [http://arxiv.org/abs/2512.13689v1](http://arxiv.org/abs/2512.13689v1)

**作者:** Yuanwen Yue, Damien Robert, Jianyuan Wang, Sunghwan Hong, Jan Dirk Wegner, Christian Rupprecht, Konrad Schindler

**发布时间:** 2025-12-15

**备注:** Project page: https://litept.github.io/

### GPT解析

### 总结

本文分析了3D点云处理网络中卷积层和注意力模块的最佳组合方式，提出了一种新的骨干网络架构，在早期使用卷积，深层使用注意力，并引入PointROPE位置编码，实现了更高效且性能相当的LitePT模型。

### 背景

现代3D点云处理神经网络架构同时包含卷积层和注意力模块，但最佳组合方式尚不清楚。

### 目的

分析不同计算模块在3D点云网络中的作用，并提出改进的3D点云骨干网络架构。

### 方法

基于卷积适合早期高分辨率层提取低级几何特征，注意力适合后期低分辨率层捕获高级语义和上下文信息的发现，提出新的3D点云骨干网络，在早期阶段使用卷积，在深层转向注意力机制，并引入无需训练的3D位置编码PointROPE以避免丢失空间布局信息。

### 主要发现

卷积适合在早期高分辨率层提取低级几何特征，而注意力更适合在后期低分辨率层捕获高级语义和上下文信息。

### 结论

提出的LitePT模型比最先进的Point Transformer V3参数少3.6倍，速度快2倍，内存使用少2倍，但在各种任务和数据集上表现相当或更好。

### 翻译

现代用于3D点云处理的神经网络架构包含卷积层和注意力模块，但最佳的组合方式尚不清楚。我们分析了不同计算模块在3D点云网络中的作用，并发现了一种直观的行为：在早期高分辨率的层中，卷积足以提取低级几何特征，而注意力机制代价高昂且没有带来任何益处；在低分辨率的深层中，注意力机制能够更有效地捕获高级语义和上下文信息。受这一设计原则的指导，我们提出了一种新的改进的3D点云骨干网络，在早期阶段采用卷积，在深层转向注意力机制。为了避免在丢弃冗余卷积层时丢失空间布局信息，我们引入了一种新颖的无需训练的3D位置编码，PointROPE。由此产生的LitePT模型比最先进的Point Transformer V3参数少3.6倍，运行速度快2倍，内存使用少2倍，但在各种任务和数据集上仍能匹配甚至超越其性能。代码和模型可在以下网址获取：https://github.com/prs-eth/LitePT。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D点云处理模型中计算效率和性能之间的权衡问题。现有的高性能模型（如Point Transformer V3）虽然效果好，但参数量大、计算成本高、内存占用大，限制了它们在资源受限设备上的应用。这个问题很重要，因为3D点云处理在机器人、自动驾驶、定位、制图和环境监测等现实应用中至关重要，需要高效且准确的模型。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了当前最先进的Point Transformer V3架构，发现它同时使用大量卷积层和注意力块，但两种操作在不同层次的作用和效率不同。通过实验发现，卷积在早期高分辨率层提取局部几何特征更有效，而注意力在后期低分辨率层捕获高级语义更有效。作者借鉴了Transformer的注意力机制、U-Net的层次结构、稀疏卷积在点云处理中的应用，以及旋转位置编码(RoPE)的思想，并将其适应到3D点云处理中，提出了PointROPE。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'卷积用于低级几何，注意力用于高级关系'。整体实现流程是：1)采用U-Net架构，编码器分为5个阶段；2)前3个阶段使用稀疏卷积块提取局部几何特征；3)后2个阶段使用PointROPE增强的注意力块捕获高级语义；4)PointROPE将特征分成三个子空间，分别应用1D RoPE嵌入；5)解码器有两种设计：简化版(LitePT-S)或对称混合版(LitePT-S*)。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出根据网络层次需求使用不同操作的设计原则；2)发明PointROPE参数免费位置编码；3)设计高效LitePT架构；4)验证分层使用卷积和注意力的最优性。相比之前工作：1)比PTv3参数少3.6倍，速度快2倍，内存少2倍，但性能相当或更好；2)与其他混合模型不同，后者在所有层次使用相同混合块，而LitePT根据层次使用不同纯块；3)PointROPE比传统卷积位置编码更高效且无参数。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LitePT通过在点云网络不同层次使用最适合的操作（早期卷积、后期注意力）并引入高效的PointROPE位置编码，实现了在大幅减少计算资源需求的同时保持或提升性能的3D点云处理架构。'}


### 论文摘要

Modern neural architectures for 3D point cloud processing contain both convolutional layers and attention blocks, but the best way to assemble them remains unclear. We analyse the role of different computational blocks in 3D point cloud networks and find an intuitive behaviour: convolution is adequate to extract low-level geometry at high-resolution in early layers, where attention is expensive without bringing any benefits; attention captures high-level semantics and context in low-resolution, deep layers more efficiently. Guided by this design principle, we propose a new, improved 3D point cloud backbone that employs convolutions in early stages and switches to attention for deeper layers. To avoid the loss of spatial layout information when discarding redundant convolution layers, we introduce a novel, training-free 3D positional encoding, PointROPE. The resulting LitePT model has $3.6\times$ fewer parameters, runs $2\times$ faster, and uses $2\times$ less memory than the state-of-the-art Point Transformer V3, but nonetheless matches or even outperforms it on a range of tasks and datasets. Code and models are available at: https://github.com/prs-eth/LitePT.

---

## 2. MMDrive: Interactive Scene Understanding Beyond Vision with Multi-representational Fusion

**论文链接:** [http://arxiv.org/abs/2512.13177v1](http://arxiv.org/abs/2512.13177v1)

**作者:** Minghui Hou, Wei-Hsing Huang, Shaofeng Liang, Daizong Liu, Tai-Hao Wen, Gang Wang, Runwei Guan, Weiping Ding

**发布时间:** 2025-12-15

### GPT解析

### 总结

MMDrive是一个创新的多模态视觉-语言模型框架，将传统2D图像理解扩展到3D场景理解，通过整合占用地图、激光雷达点云和文本描述三种模态，显著提升了自动驾驶场景中的理解和推理能力。

### 背景

视觉-语言模型通过多源信息融合实现复杂交通场景的理解和推理，是自动驾驶的核心技术。然而，现有模型受限于2D平面图像理解范式，难以有效感知3D空间信息和执行深层语义融合，导致在复杂自动驾驶环境中表现不佳。

### 目的

提出MMDrive框架，将传统的图像理解扩展到广义的3D场景理解框架，以克服现有视觉-语言模型的局限性，提升自动驾驶场景中的理解能力。

### 方法

MMDrive整合三种互补模态：占用地图、激光雷达点云和文本场景描述。引入两个关键组件：1) 面向文本的多模态调制器，根据问题语义线索动态加权各模态贡献；2) 跨模态抽象器，使用可学习抽象令牌生成突出关键区域的跨模态摘要。

### 主要发现

在DriveLM和NuScenes-QA基准测试中，MMDrive显著超越现有模型：DriveLM上BLEU-4得分为54.56，METEOR得分为41.78；NuScenes-QA上准确率达62.7%。该模型有效突破了传统仅图像理解的限制，实现复杂驾驶环境中的强大多模态推理。

### 结论

MMDrive通过整合多模态信息和引入创新组件，成功打破了传统图像理解的障碍，为自动驾驶场景的理解提供了新的基础，并增强了系统的可解释性。

### 翻译

视觉-语言模型通过多源信息融合实现复杂交通场景的理解和推理，使其成为自动驾驶的核心技术。然而，现有的视觉-语言模型受限于2D平面图像理解范式，限制了它们感知3D空间信息和执行深度语义融合的能力，导致在复杂自动驾驶环境中表现不佳。本研究提出了MMDrive，一个多模态视觉-语言模型框架，将传统的图像理解扩展到广义的3D场景理解框架。MMDrive集成了占用地图、激光雷达点云和文本场景描述三种互补模态。为此，它引入了两个用于自适应跨模态融合和关键信息提取的新组件。具体而言，面向文本的多模态调制器根据问题中的语义线索动态加权每种模态的贡献，引导上下文感知的特征集成。跨模态抽象器使用可学习的抽象令牌生成突出关键区域和基本语义的紧凑跨模态摘要。在DriveLM和NuScenes-QA基准上的全面评估表明，MMDrive相比现有的自动驾驶视觉-语言模型取得了显著的性能提升，在DriveLM上BLEU-4得分为54.56，METEOR得分为41.78，在NuScenes-QA上准确得分为62.7%。MMDrive有效打破了传统仅图像理解的障碍，能够在复杂的驾驶环境中实现强大的多模态推理，为可解释的自动驾驶场景理解提供了新的基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有自动驾驶视觉语言模型受限于2D图像理解范式，无法有效感知3D空间信息和执行深度语义融合的问题。这个问题在现实中很重要，因为自动驾驶场景本质上是动态和复杂的，需要全面的环境理解和精确的空间感知，仅依赖2D图像的方法在复杂场景（如遮挡、恶劣天气）中表现不佳，无法提供可靠的环境感知，限制了自动驾驶系统的安全性和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先认识到现有方法遵循传统的'图像理解'范式不足以应对自动驾驶场景的复杂性，因此需要扩展到更全面的'场景理解'范式。作者发现不同文本查询关注不同的模态，需要动态调整不同模态的贡献权重。在动态复杂环境中，模型需要有效优先处理大量多模态空间中的信息。作者借鉴了多种现有自动驾驶VLM方法（如DriveLM-Agent、EM-VLM4AD等）的优点，同时使用了现有的编码器架构（如UniRepLKNet、T5）和预训练模型，但创新性地整合了这些技术并设计了新的模块来解决特定问题。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将传统的'图像理解'范式扩展为'场景理解'范式，通过融合占用图、LiDAR点云和场景描述三种互补模态信息，并引入两个关键组件：Text-oriented Multimodal Modulator (TMM)和Cross-Modal Abstractor (CMA)。整体流程包括：1)多模态信息编码（图像、文本、占用图、LiDAR和场景描述）；2)使用TMM动态调整模态权重并执行跨模态注意力融合；3)使用CMA生成跨模态摘要；4)将处理后的多模态输入送入大语言模型生成答案。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)从'图像理解'到'场景理解'的范式转变，整合占用图、LiDAR和场景描述三种互补模态；2)Text-oriented Multimodal Modulator (TMM)，根据文本查询语义动态调整模态权重；3)Cross-Modal Abstractor (CMA)，使用可学习抽象令牌生成紧凑跨模态摘要；4)两阶段场景描述生成策略。相比之前的工作，MMDrive不仅融合了更多样化的模态信息，还通过动态权重调整和注意力机制实现了自适应融合，并通过抽象机制提高了信息处理效率，解决了传统方法难以处理不同查询对模态差异化需求的问题。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MMDrive通过融合多模态信息并引入创新的自适应融合与关键信息提取组件，将自动驾驶视觉语言模型从图像理解提升到全面的场景理解，显著提升了复杂驾驶环境中的多模态推理能力。'}


### 论文摘要

Vision-language models enable the understanding and reasoning of complex traffic scenarios through multi-source information fusion, establishing it as a core technology for autonomous driving. However, existing vision-language models are constrained by the image understanding paradigm in 2D plane, which restricts their capability to perceive 3D spatial information and perform deep semantic fusion, resulting in suboptimal performance in complex autonomous driving environments. This study proposes MMDrive, an multimodal vision-language model framework that extends traditional image understanding to a generalized 3D scene understanding framework. MMDrive incorporates three complementary modalities, including occupancy maps, LiDAR point clouds, and textual scene descriptions. To this end, it introduces two novel components for adaptive cross-modal fusion and key information extraction. Specifically, the Text-oriented Multimodal Modulator dynamically weights the contributions of each modality based on the semantic cues in the question, guiding context-aware feature integration. The Cross-Modal Abstractor employs learnable abstract tokens to generate compact, cross-modal summaries that highlight key regions and essential semantics. Comprehensive evaluations on the DriveLM and NuScenes-QA benchmarks demonstrate that MMDrive achieves significant performance gains over existing vision-language models for autonomous driving, with a BLEU-4 score of 54.56 and METEOR of 41.78 on DriveLM, and an accuracy score of 62.7% on NuScenes-QA. MMDrive effectively breaks the traditional image-only understanding barrier, enabling robust multimodal reasoning in complex driving environments and providing a new foundation for interpretable autonomous driving scene understanding.

---

## 3. Less Is More: Sparse and Cooperative Perturbation for Point Cloud Attacks

**论文链接:** [http://arxiv.org/abs/2512.13119v1](http://arxiv.org/abs/2512.13119v1)

**作者:** Keke Tang, Tianyu Hao, Xiaofei Wang, Weilong Peng, Denghui Zhang, Peican Zhu, Zhihong Tian

**发布时间:** 2025-12-15

**备注:** Accepted by AAAI'2026 (Oral)

### GPT解析

### 总结

本文提出了一种名为SCP的稀疏协作扰动框架，通过选择和利用点的紧凑子集，实现高效的点云对抗攻击，仅需修改少量点即可达到100%攻击成功率。

### 背景

大多数针对点云的对抗攻击会扰动大量点，导致广泛的几何变化，限制了在实际场景中的应用。而现有的稀疏攻击方法由于单个扰动的影响有限，往往难以保持有效性。

### 目的

开发一种稀疏且高效的对抗攻击方法，通过点的联合扰动产生放大的对抗效果，减少对点云几何结构的破坏。

### 方法

SCP框架识别那些相对于其联合扰动使分类错误损失局部凸的点子集，通过检查相应Hessian块的正定性来确定，然后优化该子集以生成高影响对抗示例。

### 主要发现

SCP在实验中实现了100%的攻击成功率，超越了最先进的稀疏攻击方法，并且比密集攻击具有更好的不可感知性，同时修改的点数显著减少。

### 结论

SCP是一种高效的点云对抗攻击方法，通过稀疏协作扰动实现了高攻击成功率和良好的不可感知性，在实际应用中具有更大潜力。

### 翻译

大多数针对点云的对抗攻击会扰动大量点，导致广泛的几何变化，限制了在实际场景中的应用。虽然最近的工作探索了稀疏攻击，仅修改少数点，但这类方法往往由于单个扰动的有限影响而难以保持有效性。在本文中，我们提出了SCP，一种稀疏和协作扰动框架，它选择并利用点的紧凑子集，其联合扰动能够产生放大的对抗效果。具体而言，SCP识别子集，使得相对于它们的联合扰动，分类错误损失是局部凸的，这是通过检查相应Hessian块的正定性来确定的。然后优化选定的子集，以生成具有最小修改的高影响对抗示例。大量实验表明，SCP实现了100%的攻击成功率，超越了最先进的稀疏攻击，并且与密集攻击相比，具有更好的不可感知性，且修改的点数更少。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云对抗攻击中的效率问题，即如何在修改尽可能少的点的情况下实现有效的对抗攻击。这个问题很重要，因为随着3D传感器和深度学习技术在自动驾驶、机器人操作等关键领域的广泛应用，这些系统容易受到对抗攻击的威胁，而现有方法要么需要修改大量点导致明显几何变化（不实用），要么稀疏攻击方法因忽略点间相互作用而效果有限。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到点云网络中点之间存在强烈非线性相互作用，假设联合扰动某些点能产生比单独扰动更强的效果。他们借鉴了现有稀疏攻击和几何感知优化的思想，但指出这些方法通常假设点对攻击的贡献是独立的。因此，作者提出需要超越孤立点选择，识别能产生协同效应的点子集，并基于Hessian矩阵的正定性来量化这种合作效应，从而设计了SCP框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是识别点云中具有协同效应的紧凑点子集，这些点的联合扰动能产生放大的对抗效果。实现流程分为三步：1)基于梯度分析选择有影响力的初始点集；2)通过Schur补条件检查进行凸性保持扩展，确保选定的点子集位于损失函数的局部凸区域；3)对选定的合作子集进行联合扰动优化，生成高效且不可感知的对抗样本。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)识别了稀疏点云攻击中被忽视的合作相互作用，并建立了基于Hessian的准则来表征这种协同；2)开发了SCP框架，通过梯度筛选和Schur补引导的扩展选择紧凑合作子集；3)实现了100%攻击成功率，同时修改点数极少。相比之前工作，SCP不仅考虑点的独立影响力，还建模点间二阶相互作用，实现了比现有稀疏攻击更高的成功率、更好的不可感知性，且比密集攻击修改点数少两个数量级。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SCP通过识别和利用点云中具有协同效应的紧凑点子集，实现了以极少数点修改达到100%攻击成功率的稀疏且高效的点云对抗攻击，同时保持了优于现有方法的不可感知性。'}


### 论文摘要

Most adversarial attacks on point clouds perturb a large number of points, causing widespread geometric changes and limiting applicability in real-world scenarios. While recent works explore sparse attacks by modifying only a few points, such approaches often struggle to maintain effectiveness due to the limited influence of individual perturbations. In this paper, we propose SCP, a sparse and cooperative perturbation framework that selects and leverages a compact subset of points whose joint perturbations produce amplified adversarial effects. Specifically, SCP identifies the subset where the misclassification loss is locally convex with respect to their joint perturbations, determined by checking the positivedefiniteness of the corresponding Hessian block. The selected subset is then optimized to generate high-impact adversarial examples with minimal modifications. Extensive experiments show that SCP achieves 100% attack success rates, surpassing state-of-the-art sparse attacks, and delivers superior imperceptibility to dense attacks with far fewer modifications.

---

## 4. Diffusion-Based Restoration for Multi-Modal 3D Object Detection in Adverse Weather

**论文链接:** [http://arxiv.org/abs/2512.13107v1](http://arxiv.org/abs/2512.13107v1)

**作者:** Zhijian He, Feifei Liu, Yuwei Li, Zhanpeng Liu, Jintao Cheng, Xieyuanli Chen, Xiaoyu Tang

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出DiffFusion框架，通过扩散模型和自适应跨模态融合增强多模态3D目标检测在恶劣天气条件下的鲁棒性

### 背景

多模态3D目标检测对机器人和自动驾驶的可靠感知至关重要，但在恶劣天气条件下效果受限，主要受天气导致的失真和不同数据模态间不匹配的影响

### 目的

开发一个能够增强在挑战性天气条件下鲁棒性的框架，解决多模态3D目标检测在恶劣环境中的性能下降问题

### 方法

利用扩散模型去噪和生成能力适应各种天气条件；Diffusion-IR恢复受天气影响的图像；点云恢复(PCR)使用图像对象线索补偿损坏的LiDAR数据；双向自适应融合和对齐模块(BAFAM)实现动态多模态融合和双向鸟瞰图对齐

### 主要发现

在三个公共数据集上的实验表明DiffFusion在恶劣天气下实现了最先进的鲁棒性，同时保持了强大的清洁数据性能；真实世界DENSE数据集上的零样本结果验证了其泛化能力

### 结论

DiffFusion框架通过扩散模型和自适应融合有效解决了恶劣天气条件下的多模态3D目标检测问题，将以开源形式发布

### 翻译

多模态3D目标检测对机器人和自动驾驶中的可靠感知至关重要。然而，由于天气引起的失真和不同数据模态之间的不匹配，其在恶劣天气条件下的有效性仍然有限。在这项工作中，我们提出了DiffFusion，一个基于扩散恢复和自适应跨模态融合设计的新框架，旨在增强在挑战性天气下的鲁棒性。我们的关键见解是，扩散模型具有强大的去噪和生成数据能力，可以适应各种天气条件。基于此，DiffFusion引入了Diffusion-IR恢复受天气影响退化的图像，以及点云恢复(PCR)使用图像对象线索补偿损坏的LiDAR数据。为了解决两种模态之间的不匹配问题，我们开发了双向自适应融合和对齐模块(BAFAM)。它实现了动态多模态融合和双向鸟瞰图(BEV)对齐，以保持一致的空间对应关系。在三个公共数据集上的广泛实验表明，DiffFusion在恶劣天气下实现了最先进的鲁棒性，同时保持了强大的清洁数据性能。在真实世界DENSE数据集上的零样本结果进一步验证了其泛化能力。我们的DiffFusion实现将以开源形式发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在恶劣天气条件下（如雨、雾、强光等）多模态3D目标检测性能显著下降的问题。这个问题在现实中非常重要，因为自动驾驶和机器人技术需要在各种天气条件下可靠运行，而现有的3D目标检测方法在恶劣天气下性能大幅下降，限制了这些技术在真实世界环境中的安全部署。提高恶劣天气条件下的感知能力对于自动驾驶技术的实用化和普及至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了恶劣天气下多模态3D目标检测面临的两个主要挑战：传感器数据退化和模态间空间失配。核心洞察是扩散模型具有强大的去噪和生成能力，可以适应各种天气条件。基于此，作者设计了DiffFusion框架，包含基于扩散的恢复模块和双向自适应融合模块。作者借鉴了现有工作如DDPM/DDIM扩散模型、BEV-based多模态融合方法、点云上采样技术和CenterNet等2D检测方法，但创新性地将这些技术整合到一个专门解决恶劣天气问题的统一框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用扩散模型的去噪和生成能力恢复恶劣天气下退化的传感器数据，并通过自适应的多模态融合和对齐机制建立不同模态间的可靠对应关系。整体流程分为三步：1) 基于扩散的恢复模块，包括Diffusion-IR恢复图像和PCR利用图像信息恢复点云；2) 双向自适应融合和对齐模块(BAFAM)，包括交叉注意力自适应融合和双向BEV对齐；3) 将融合后的特征输入3D检测头生成最终检测结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的扩散模型多模态恢复框架，同时处理图像和点云退化；2) 双向自适应融合和对齐模块(BAFAM)，解决模态间空间失配问题；3) 端到端的恶劣天气处理，在保持干净数据性能的同时提高恶劣天气鲁棒性。相比之前工作，DiffFusion不同于传统多模态方法（主要优化于干净数据），也不同于其他恶劣天气处理方法（如TripleMixer只处理单模态，SAMFusion计算开销大），它系统性地解决了天气引起的传感器退化和模态失配问题，并实现了强大的跨域泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DiffFusion通过结合基于扩散的传感器数据恢复和自适应多模态融合，显著提高了自动驾驶系统在恶劣天气条件下的3D目标检测鲁棒性和泛化能力。'}


### 论文摘要

Multi-modal 3D object detection is important for reliable perception in robotics and autonomous driving. However, its effectiveness remains limited under adverse weather conditions due to weather-induced distortions and misalignment between different data modalities. In this work, we propose DiffFusion, a novel framework designed to enhance robustness in challenging weather through diffusion-based restoration and adaptive cross-modal fusion. Our key insight is that diffusion models possess strong capabilities for denoising and generating data that can adapt to various weather conditions. Building on this, DiffFusion introduces Diffusion-IR restoring images degraded by weather effects and Point Cloud Restoration (PCR) compensating for corrupted LiDAR data using image object cues. To tackle misalignments between two modalities, we develop Bidirectional Adaptive Fusion and Alignment Module (BAFAM). It enables dynamic multi-modal fusion and bidirectional bird's-eye view (BEV) alignment to maintain consistent spatial correspondence. Extensive experiments on three public datasets show that DiffFusion achieves state-of-the-art robustness under adverse weather while preserving strong clean-data performance. Zero-shot results on the real-world DENSE dataset further validate its generalization. The implementation of our DiffFusion will be released as open-source.

---

## 5. VoroLight: Learning Quality Volumetric Voronoi Meshes from General Inputs

**论文链接:** [http://arxiv.org/abs/2512.12984v1](http://arxiv.org/abs/2512.12984v1)

**作者:** Jiayin Lu, Ying Jiang, Yin Yang, Chenfanfu Jiang

**发布时间:** 2025-12-15

### GPT解析

### 总结

VoroLight是一个基于Voronoi网格化的可微分3D形状重建框架，能够从多种输入生成平滑、水密且拓扑一致的体积网格。

### 背景

3D形状重建领域需要能够处理多种输入格式并生成高质量表面和体积网格的方法。

### 目的

开发一个能够从图像、隐式形状level-set场、点云和网格等多种输入直接生成高质量3D形状的可微分框架。

### 方法

VoroLight采用三阶段方法：1)使用可微分Voronoi公式初始化表面；2)通过多边形面球训练阶段改进表面质量；3)重新使用可微分Voronoi公式进行体积优化，添加内部生成点。

### 主要发现

基于Voronoi网格化的可微分框架能够生成平滑、水密且拓扑一致的体积网格，适用于多种输入格式。

### 结论

VoroLight提供了一个灵活且有效的3D形状重建方法，能够处理多种输入并生成高质量的表面和体积表示。

### 翻译

我们提出了VoroLight，一个基于Voronoi网格化的可微分3D形状重建框架。我们的方法能够直接从多种输入（包括图像、隐式形状level-set场、点云和网格）生成平滑、水密且拓扑一致的体积网格。VoroLight分为三个阶段：首先使用可微分Voronoi公式初始化表面，然后通过多边形面球训练阶段改进表面质量，最后重新使用可微分Voronoi公式进行体积优化，添加额外的内部生成点。项目页面：https://jiayinlu19960224.github.io/vorolight/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从多种输入（图像、隐式形状水平集场、点云和网格）生成高质量、防水且拓扑一致的体积Voronoi网格的问题。这个问题在现实中很重要，因为现有的3D重建方法要么只关注表面而忽略内部结构，要么依赖于干净输入或固定离散化，限制了几何灵活性。高质量的体积Voronoi网格对模拟、分析和制造等应用至关重要，因为它们具有凸的、防水的和拓扑一致的特性，适合物理应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了多个现有工作：VoroMesh的可微分Voronoi表面优化、VoroCrust的基于球体的边界一致Voronoi网格构建，以及Differentiable Voronoi Diagrams的闭式可微分公式。作者设计了一个三阶段框架：首先使用可微分Voronoi公式初始化表面，然后通过多边形面球体训练细化表面质量，最后重用可微分Voronoi公式进行体积优化。作者特别扩展了VoroCrust的球体构建方法，使其成为可微分、可学习的公式，并比较了三角形面与多边形面约束的效果。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过优化可微分Voronoi结构，从多种输入直接生成高质量的体积Voronoi网格，利用Voronoi图自然产生凸的、防水的和拓扑一致的单元的特性。整体流程分为三阶段：1) 初始多边形表面初始化：通过边界反射放置生成器，使用可微分Voronoi公式优化；2) 表面球体细化：为每个顶点定义可训练球体，优化球体参数使相交于同一面的球体通过面的两个交点；3) 体积网格生成：固定表面生成器，引入内部生成器并在质心Voronoi剖分损失下优化，产生一致的体积结构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) VoroLight三阶段框架，支持多种输入类型；2) 多边形面球体训练公式，实现自正则化表面细化；3) 发现三角形面约束比多边形面约束产生更高质量表面。相比之前工作，VoroLight不同于VoroMesh（仅限表面和点云输入），不同于VoroCrust（确定性方法且仅限三角形网格输入），不同于TetSphere（生成局部四面体簇而非全局一致网格），也不同于传统体积网格化方法（不依赖干净输入或固定离散化）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VoroLight提出了一种可微分三阶段框架，通过优化Voronoi结构从多种输入直接生成高质量的防水体积Voronoi网格，结合基于球体的表面细化和质心Voronoi剖分的体积优化，实现了从图像、点云、隐式场和网格输入的一致拓扑重建。'}


### 论文摘要

We present VoroLight, a differentiable framework for 3D shape reconstruction based on Voronoi meshing. Our approach generates smooth, watertight surfaces and topologically consistent volumetric meshes directly from diverse inputs, including images, implicit shape level-set fields, point clouds and meshes. VoroLight operates in three stages: it first initializes a surface using a differentiable Voronoi formulation, then refines surface quality through a polygon-face sphere training stage, and finally reuses the differentiable Voronoi formulation for volumetric optimization with additional interior generator points. Project page: https://jiayinlu19960224.github.io/vorolight/

---

## 6. Lemon: A Unified and Scalable 3D Multimodal Model for Universal Spatial Understanding

**论文链接:** [http://arxiv.org/abs/2512.12822v1](http://arxiv.org/abs/2512.12822v1)

**作者:** Yongyuan Liang, Xiyao Wang, Yuanchen Ju, Jianwei Yang, Furong Huang

**发布时间:** 2025-12-14

### GPT解析

### 总结

该论文提出了Lemon，一种统一的transformer架构，用于解决大规模多模态模型扩展到3D理解时的挑战。通过联合处理3D点云补丁和语言令牌，实现了早期的空间-语言融合，提高了参数效率，并支持更有效的模型扩展。

### 背景

将大规模多模态模型扩展到3D理解面临独特挑战：点云数据稀疏且不规则；现有模型依赖具有模态特定编码器的碎片化架构；训练流程通常不稳定且可扩展性差。

### 目的

开发一个统一的transformer架构来解决3D理解中的挑战，实现更有效的模型扩展，并在3D理解和推理任务上建立新的最先进性能。

### 方法

提出了Lemon统一transformer架构，联合处理3D点云补丁和语言令牌作为单一序列；开发了保留空间上下文的结构化补丁化和令牌化方案；设计了三阶段训练课程，从对象级识别逐步构建到场景级空间推理能力。

### 主要发现

Lemon在全面的3D理解和推理任务上建立了新的最先进性能，包括对象识别、描述到3D场景中的空间推理；随着模型大小和训练数据的增加，Lemon展示了强大的扩展特性。

### 结论

该工作为推进现实世界应用中的3D空间智能提供了统一的基础。

### 翻译

将大规模多模态模型扩展到3D理解面临独特挑战：点云数据稀疏且不规则，现有模型依赖具有模态特定编码器的碎片化架构，训练流程通常不稳定且可扩展性差。我们介绍了Lemon，一种统一transformer架构，通过将3D点云补丁和语言令牌作为单一序列联合处理来解决这些挑战。与依赖模态特定编码器和跨模态对齐模块的先前工作不同，这种设计实现了早期的空间-语言融合，消除了冗余编码器，提高了参数效率，并支持更有效的模型扩展。为了处理3D数据的复杂性，我们开发了一种保留空间上下文的结构化补丁化和令牌化方案，以及一个三阶段训练课程，逐步从对象级识别构建到场景级空间推理能力。Lemon在全面的3D理解和推理任务上建立了新的最先进性能，从对象识别和描述到3D场景中的空间推理，同时展示了随着模型大小和训练数据增加的强大扩展特性。我们的工作为推进现实世界应用中的3D空间智能提供了统一的基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文旨在解决将大型多模态模型扩展到3D理解领域的挑战，特别是点云数据稀疏不规则、现有模型依赖碎片化架构以及训练流程不稳定和可扩展性差的问题。这个问题非常重要，因为3D理解对于机器人技术、AR/VR系统和空间AI应用至关重要，能够使AI系统更好地理解和交互物理世界，但目前缺乏像2D视觉语言模型那样强大的通用3D理解能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有3D多模态模型的局限性，包括3D编码器适应性有限、3D数据规模不足以及架构不平衡等问题。他们设计了一个统一的transformer架构，将3D点云块和语言标记作为单一序列进行联合处理，消除了对模态特定编码器的需求。作者借鉴了2D视觉语言模型（如VisualBERT、Fuyu-8B）的统一架构设计思想，以及现有的3D基础模型（如Point-BERT、ULIP）的处理方法，并在训练策略上参考了多模态模型训练框架如LLaMA-Factory。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的transformer架构，将3D点云块和语言标记作为单一序列进行联合处理，直接将3D数据嵌入到语言模型框架中。整体实现流程包括：1)数据预处理：使用FPS采样对点云采样，通过递归3D空间方案分区点云为块，使用线性投影器将3D块映射到语言嵌入空间；2)模型架构：使用专门标记编码3D模态，连接3D块嵌入与文本标记创建统一序列；3)三阶段训练：从对象级识别到对象级标题生成再到场景级空间问答；4)推理应用：处理各种3D多模态任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的transformer架构，首次在单个序列中处理点云块和语言标记；2)动态3D分块和标记化方案，将不规则点云转换为结构化标记序列；3)三阶段渐进式训练课程，逐步构建能力；4)首次系统分析3D LMM的扩展定律。相比之前工作，LEMON的不同之处在于：使用统一架构而非模态特定编码器；直接处理3D数据而非依赖预训练编码器；采用渐进式训练而非单一阶段训练；在多个任务上实现更优性能并展示更好的可扩展性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LEMON通过统一的transformer架构和渐进式训练课程，首次实现了高效可扩展的3D多模态理解，为空间智能和具身AI提供了新的基础。'}


### 论文摘要

Scaling large multimodal models (LMMs) to 3D understanding poses unique challenges: point cloud data is sparse and irregular, existing models rely on fragmented architectures with modality-specific encoders, and training pipelines often suffer from instability and poor scalability. We introduce Lemon, a unified transformer architecture that addresses these challenges by jointly processing 3D point cloud patches and language tokens as a single sequence. Unlike prior work that relies on modality-specific encoders and cross-modal alignment modules, this design enables early spatial-linguistic fusion, eliminates redundant encoders, improves parameter efficiency, and supports more effective model scaling. To handle the complexity of 3D data, we develop a structured patchification and tokenization scheme that preserves spatial context, and a three-stage training curriculum that progressively builds capabilities from object-level recognition to scene-level spatial reasoning. Lemon establishes new state-of-the-art performance across comprehensive 3D understanding and reasoning tasks, from object recognition and captioning to spatial reasoning in 3D scenes, while demonstrating robust scaling properties as model size and training data increase. Our work provides a unified foundation for advancing 3D spatial intelligence in real-world applications.

---

## 7. From Small to Large: Generalization Bounds for Transformers on Variable-Size Inputs

**论文链接:** [http://arxiv.org/abs/2512.12805v1](http://arxiv.org/abs/2512.12805v1)

**作者:** Anastasiia Alokhina, Pan Li

**发布时间:** 2025-12-14

### GPT解析

### 总结

Transformers表现出显著的大小泛化能力，能够从小型token集合外推到更长的token集合，这种现象在点云、图和自然语言等多种应用中已被观察到，但缺乏严格的理论解释。

### 背景

尽管Transformers在大小泛化方面取得了经验性成功，但这种能力仍缺乏严谨的理论表征，需要建立理论框架来解释这一现象。

### 目的

开发一个理论框架来分析几何数据的大小泛化现象，这些几何数据被表示为来自连续源的离散样本（如来自流形的点云，来自图论的图）。

### 方法

提出一个关于离散样本的Transformer输出与其连续域等效输出之间误差的界限，证明对于具有稳定位置编码的Transformers，这个界限由采样密度和数据流形的内在维度决定。

### 主要发现

Transformers的输出误差界限取决于采样密度和数据流形的内在维度，这一发现为理解Transformers的大小泛化能力提供了理论基础。

### 结论

在各种尺寸的图和点云上的实验证实了所提出的理论界限的紧致性，验证了理论框架的有效性。

### 翻译

Transformers表现出显著的'大小泛化'特性，能够从小型token集合外推到显著更长的token集合。这种行为在点云、图和自然语言等多种应用中都有记录。尽管 empirically 成功，但这种能力仍缺乏一些严谨的理论表征。在本文中，我们开发了一个理论框架来分析几何数据中的这种现象，我们将几何数据表示为来自连续源的离散样本（例如，来自流形的点云，来自图论的图）。我们的核心贡献是提出了一个关于离散样本的Transformer输出与其连续域等效输出之间误差的界限。我们证明，对于具有稳定位置编码的Transformers，这个界限由采样密度和数据流形的内在维度决定。在各种尺寸的图和点云上的实验证实了我们的理论界限的紧致性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决Transformer模型在不同规模输入数据上的泛化能力缺乏理论解释的问题。具体来说，虽然Transformer在实践中表现出能够从小规模训练数据泛化到显著更大的测试数据（如从小分子到大分子，或从固定到更长的上下文窗口），但这种能力缺乏严格的数学理论支持。这个问题在研究中很重要，因为它填补了理论与实践之间的鸿沟，帮助我们理解为什么Transformer能够处理规模变化的数据，这对于开发更强大、更可靠的模型至关重要，特别是在处理点云、图和自然语言等规模变化的数据时。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者借鉴了消息传递神经网络(MPNNs)在图数据上的理论框架，特别是基于图论(graphon)的理论。作者注意到，虽然MPNNs的大小泛化性质已经被广泛研究，但Transformer由于使用全局注意力和位置编码，其泛化性质尚未得到充分理解。作者设计的方法包括：1)将数据样本建模为'tokenset'（token集合），这些token是从连续域中独立采样的；2)定义了一个适用于连续和离散数据的Transformer类；3)推导了离散tokenset与连续极限之间输出误差的界限；4)证明了对于具有稳定位置编码的Transformer，这个界限由采样密度和数据流形的内在维度决定。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：将Transformer的泛化能力与数据采样密度和位置编码的稳定性联系起来。具体来说，作者证明了Transformer在离散tokenset上的输出与连续极限上的输出之间的误差界限，这个界限随着采样密度的增加而减小，并且位置编码的稳定性对泛化性能至关重要。整体实现流程如下：1)定义连续tokenset和离散tokenset的概念；2)定义适用于连续和离散数据的Transformer类；3)推导单层Transformer的离散化误差和扰动误差的界限；4)将单层误差界限扩展到多层Transformer；5)将输出误差界限转化为学习任务的泛化界限；6)通过实验验证理论界限的紧致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次为Transformer在几何数据（如图和点云）上的大小泛化提供了严格的理论保证；2)提出了位置编码稳定性(ρ-stability)的概念，并证明这是Transformer泛化能力的关键因素；3)推导了泛化误差的界限，该界限由离散化误差和位置编码误差组成；4)证明了随机游走位置编码是稳定的，而最短路径距离位置编码是不稳定的。相比之前的工作，这篇论文的主要不同在于：之前的工作主要集中在自然语言处理领域，而本文专注于几何数据；之前的工作大多关注MPNNs，而本文将理论框架扩展到Transformer架构；本文首次明确了位置编码稳定性对Transformer泛化能力的重要性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文为Transformer在不同规模输入数据上的泛化能力提供了第一个严格的理论保证，证明了这种泛化能力取决于数据采样密度和位置编码的稳定性，并通过实验验证了理论结果。'}


### 论文摘要

Transformers exhibit a notable property of \emph{size generalization}, demonstrating an ability to extrapolate from smaller token sets to significantly longer ones. This behavior has been documented across diverse applications, including point clouds, graphs, and natural language. Despite its empirical success, this capability still lacks some rigorous theoretical characterizations. In this paper, we develop a theoretical framework to analyze this phenomenon for geometric data, which we represent as discrete samples from a continuous source (e.g., point clouds from manifolds, graphs from graphons). Our core contribution is a bound on the error between the Transformer's output for a discrete sample and its continuous-domain equivalent. We prove that for Transformers with stable positional encodings, this bound is determined by the sampling density and the intrinsic dimensionality of the data manifold. Experiments on graphs and point clouds of various sizes confirm the tightness of our theoretical bound.

---

## 8. DrivePI: Spatial-aware 4D MLLM for Unified Autonomous Driving Understanding, Perception, Prediction and Planning

**论文链接:** [http://arxiv.org/abs/2512.12799v1](http://arxiv.org/abs/2512.12799v1)

**作者:** Zhe Liu, Runhui Huang, Rui Yang, Siming Yan, Zining Wang, Lu Hou, Di Lin, Xiang Bai, Hengshuang Zhao

**发布时间:** 2025-12-14

### GPT解析

### 总结

DrivePI是一种新颖的空间感知4D多模态大语言模型，作为统一的视觉-语言-行动框架，在自动驾驶的3D感知、预测和规划任务中表现出色，即使使用较小的模型也能超越现有专门模型。

### 背景

多模态大语言模型(MLLMs)在多个领域表现出强大能力，但在自动驾驶中生成细粒度3D感知和预测输出方面的应用尚未充分探索。

### 目的

提出DrivePI，一种新颖的具有空间感知能力的4D MLLM，作为统一的视觉-语言-行动(VLA)框架，同时兼容视觉-行动(VA)模型。

### 方法

通过端到端优化并行执行空间理解、3D感知、预测和规划；在统一的MLLM架构中整合点云、多视图图像和语言指令；开发数据引擎生成文本-占用和文本流问答对用于4D空间理解。

### 主要发现

仅使用0.5B的Qwen2.5模型作为骨干，DrivePI作为单一统一模型就能匹配或超越现有VLA和VA模型；在nuScenes-QA上的平均准确率比OpenDriveVLA-7B高2.5%；在nuScenes上的碰撞率比ORION降低70%；在OpenOcc上的3D占用比FB-OCC高10.3 RayIoU；在OpenOcc上的占用流将mAVE从0.591降至0.509；在nuScenes上的规划比VAD低32%的L2误差。

### 结论

DrivePI是一个统一的多模态模型，在自动驾驶的3D感知、预测和规划任务中表现出色，即使使用较小的模型也能超越现有专门模型。

### 翻译

尽管多模态大语言模型(MLLMs)已在多个领域展现出强大能力，但它们在自动驾驶中生成细粒度3D感知和预测输出方面的应用仍探索不足。在本文中，我们提出了DrivePI，一种新颖的空间感知4D MLLM，作为统一的视觉-语言-行动(VLA)框架，同时也兼容视觉-行动(VA)模型。我们的方法通过端到端优化并行执行空间理解、3D感知(即3D占用)、预测(即占用流)和规划(即行动输出)。为了获得精确的几何信息和丰富的视觉外观，我们的方法在统一的MLLM架构中整合了点云、多视图图像和语言指令。我们进一步开发了一个数据引擎，用于生成文本-占用和文本流问答对，以实现4D空间理解。值得注意的是，仅使用0.5B的Qwen2.5模型作为MLLM骨干，DrivePI作为单一统一模型就能匹配或超越现有的VLA模型和专门的VA模型。具体而言，与VLA模型相比，DrivePI在nuScenes-QA上的平均准确率比OpenDriveVLA-7B高2.5%，在nuScenes上的碰撞率比ORION降低70%(从0.37%降至0.11%)。与专门的VA模型相比，DrivePI在OpenOcc上的3D占用比FB-OCC高10.3 RayIoU，在OpenOcc上的占用流将mAVE从0.591降至0.509，在nuScenes上的规划比VAD低32%的L2误差(从0.72m降至0.49m)。代码将在https://github.com/happinesslz/DrivePI上提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶系统中如何结合视觉-动作(VA)模型的精确空间感知能力和视觉-语言-动作(VLA)框架的自然语言交互能力的问题。这个问题很重要，因为现有VA模型虽然空间感知精确但缺乏语言交互，而VLA框架虽有良好交互能力却缺乏可靠的细粒度3D感知输出，导致自动驾驶系统要么无法与人类自然交互，要么缺乏可解释性和安全保证。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了VA模型(如UniAD、VAD)和VLA框架(如OpenDriveVLA、ORION)各自的优缺点，然后思考能否创建一个统一框架结合两者的优势。他们借鉴了VA模型的多模态输入处理方式和模块化设计，以及VLA框架的交互能力和语言理解，同时采用了现有的多模态视觉编码器和BEV表示方法，在此基础上创新性地引入LiDAR作为补充传感模态，并设计了专门的空间投影机制和数据引擎。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的视觉-语言-动作框架，结合粗粒度语言空间理解和细粒度3D感知能力，引入LiDAR提供精确3D几何信息，同时生成中间细粒度的3D感知和预测表示。整体流程包括：1)使用多模态视觉编码器处理图像和LiDAR数据获取BEV特征；2)通过空间投影将BEV特征转换为视觉令牌；3)将视觉令牌和文本令牌输入MLLM；4)通过四个专门头部(文本、3D占用、占用流、动作扩散)生成输出；5)所有任务通过端到端方式联合优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出首个统一的空间感知4D MLLM框架；2)引入LiDAR作为补充传感模态提供精确3D几何信息；3)实现准确的3D感知和预测以增强可解释性和安全性；4)开发空间理解基准评估语言空间推理能力；5)仅用0.5B参数模型就超越现有专用模型。相比之前工作，DrivePI同时结合了VA模型的精确空间感知和VLA框架的交互能力，不仅依赖相机图像还引入LiDAR，生成中间细粒度3D表示，并将3D数据集成到自然语言描述中。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DrivePI通过统一的空间感知4D多模态大语言模型框架，成功结合了自动驾驶系统中精确的空间感知能力与自然语言交互能力，仅用0.5B参数模型就实现了超越现有专用模型的性能。'}


### 论文摘要

Although multi-modal large language models (MLLMs) have shown strong capabilities across diverse domains, their application in generating fine-grained 3D perception and prediction outputs in autonomous driving remains underexplored. In this paper, we propose DrivePI, a novel spatial-aware 4D MLLM that serves as a unified Vision-Language-Action (VLA) framework that is also compatible with vision-action (VA) models. Our method jointly performs spatial understanding, 3D perception (i.e., 3D occupancy), prediction (i.e., occupancy flow), and planning (i.e., action outputs) in parallel through end-to-end optimization. To obtain both precise geometric information and rich visual appearance, our approach integrates point clouds, multi-view images, and language instructions within a unified MLLM architecture. We further develop a data engine to generate text-occupancy and text-flow QA pairs for 4D spatial understanding. Remarkably, with only a 0.5B Qwen2.5 model as MLLM backbone, DrivePI as a single unified model matches or exceeds both existing VLA models and specialized VA models. Specifically, compared to VLA models, DrivePI outperforms OpenDriveVLA-7B by 2.5% mean accuracy on nuScenes-QA and reduces collision rate by 70% over ORION (from 0.37% to 0.11%) on nuScenes. Against specialized VA models, DrivePI surpasses FB-OCC by 10.3 RayIoU for 3D occupancy on OpenOcc, reduces the mAVE from 0.591 to 0.509 for occupancy flow on OpenOcc, and achieves 32% lower L2 error than VAD (from 0.72m to 0.49m) for planning on nuScenes. Code will be available at https://github.com/happinesslz/DrivePI

---

## 9. A Graph Attention Network-Based Framework for Reconstructing Missing LiDAR Beams

**论文链接:** [http://arxiv.org/abs/2512.12410v1](http://arxiv.org/abs/2512.12410v1)

**作者:** Khalfalla Awedat, Mohamed Abidalrekab, Mohammad El-Yabroudi

**发布时间:** 2025-12-13

### GPT解析

### 总结

这篇论文提出了一种基于图注意力网络（GAT）的框架，用于重建旋转式激光雷达传感器中因硬件老化、灰尘、雪花、雾或明亮反射等原因丢失的垂直光束，仅使用当前激光雷达帧而不需要额外信息，能有效恢复点云中的缺失垂直切片。

### 背景

垂直光束丢失在旋转式激光雷达传感器中是一个严重问题，可能由硬件老化、灰尘、雪花、雾或明亮反射等因素触发，会导致点云中缺失整个垂直切片，严重影响自动驾驶汽车中的3D感知能力。

### 目的

开发一种仅使用当前激光雷达帧就能重建缺失垂直通道的方法，无需依赖相机图像或时间信息，以改善自动驾驶汽车中的3D感知能力。

### 方法

将每个激光雷达扫描表示为非结构化空间图（点作为节点，边连接附近点并保持原始光束索引顺序），使用多层图注意力网络学习局部几何邻域的自适应注意力权重，并直接回归丢失位置的高程（z）值。

### 主要发现

在1,065个具有模拟通道丢失的KITTI序列上测试，平均高度均方根误差为11.67厘米；87.98%的重建点落在10厘米误差阈值内；单GPU每帧推理时间为14.65秒；重建质量对不同邻域大小k保持稳定。

### 结论

纯图注意力模型仅基于原始点云几何就能有效恢复现实传感器退化情况下的丢失垂直光束，为自动驾驶系统提供了一种可靠的点云重建方法。

### 翻译

旋转式激光雷达传感器中由硬件老化、灰尘、雪花、雾或明亮反射引起的垂直光束丢失会从点云中移除整个垂直切片，严重降低自动驾驶汽车中的3D感知能力。本文提出了一种基于图注意力网络（GAT）的框架，仅使用当前激光雷达帧就能重建这些缺失的垂直通道，不需要相机图像或时间信息。每个激光雷达扫描被表示为一个非结构化空间图：点是节点，边连接附近的点同时保持原始光束索引顺序。多层GAT学习局部几何邻域的自适应注意力权重，并直接回归丢失位置的高程（z）值。在1,065个具有模拟通道丢失的原始KITTI序列上训练和评估，该方法实现了平均高度均方根误差为11.67厘米，87.98%的重建点落在10厘米误差阈值内。在单个GPU上，每帧推理时间为14.65秒，重建质量对于不同的邻域大小k保持稳定。这些结果表明，仅基于原始点云几何操作的纯图注意力模型能有效恢复现实传感器退化情况下的丢失垂直光束。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决旋转式激光雷达传感器中垂直光束丢失的重建问题。在现实应用中，由于硬件老化、灰尘、雪、雾或强反射等原因，LiDAR传感器可能会丢失整个垂直光束切片，导致点云中出现垂直不连续性。这个问题在自动驾驶领域尤为重要，因为垂直光束包含关键的高度信息(z坐标)，对于区分可行驶表面和障碍物、保持几何一致性以及准确识别车辆周围物体至关重要。缺失这些光束会严重降低3D感知系统的可靠性，影响物体检测、深度估计和自由空间判断等关键任务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了LiDAR垂直光束丢失的原因和影响，认识到现有基于体素CNN或插值方法难以捕捉真实扫描的不规则结构。他们转向图神经网络(GNNs)领域，特别是图注意力网络(GAT)，因为GAT能够自适应地加权邻近点的贡献，强调最具信息量的几何线索。作者借鉴了GNN在点云处理中的应用经验，以及先前关于LiDAR垂直结构重要性的研究，但针对垂直光束重建这一特定问题进行了创新设计。他们构建了一个多层GAT框架，将LiDAR扫描表示为空间图，通过注意力机制学习局部几何关系，仅使用单帧点云信息重建缺失光束，而不依赖相机图像或多帧时间信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将LiDAR点云表示为一个非结构化空间图，其中点是节点，边连接空间邻近的点同时保留原始光束索引顺序，然后使用图注意力网络(GAT)学习节点间的自适应权重关系，通过多层GAT结构捕获局部和全局几何上下文信息，最终回归缺失的高度(z)值。整体实现流程包括：1)将每个LiDAR扫描构建为k近邻图结构；2)通过多层GAT层进行特征学习和注意力权重计算；3)每层GAT执行线性投影、注意力分数计算、权重归一化和加权聚合；4)最终通过回归头预测缺失的垂直坐标；5)在KITTI数据集上训练和评估，模拟垂直光束丢失场景并重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)纯LiDAR-based解决方案，不依赖相机图像或多帧时间信息；2)首次将图注意力网络直接应用于LiDAR垂直光束重建任务；3)非结构化空间图表示，保留原始光束索引同时连接空间邻近点；4)多层GAT架构捕获更广泛上下文信息。相比之前工作，本文方法与传统插值方法相比能更好捕捉不规则结构；与基于体素CNN的方法相比避免了数据转换；与其他图方法相比充分利用了注意力机制优势；与多模态方法相比在纯LiDAR场景下更适用；与GLiDR等模型相比专注于光束级重建而非仅全局形状一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于图注意力网络的纯LiDAR点云重建框架，能够仅利用当前帧的点云几何信息有效恢复因传感器退化或环境因素导致的垂直光束缺失，显著提高了自动驾驶系统在点云数据不完整情况下的3D感知能力。'}


### 论文摘要

Vertical beam dropout in spinning LiDAR sensors triggered by hardware aging, dust, snow, fog, or bright reflections removes entire vertical slices from the point cloud and severely degrades 3D perception in autonomous vehicles. This paper proposes a Graph Attention Network (GAT)-based framework that reconstructs these missing vertical channels using only the current LiDAR frame, with no camera images or temporal information required. Each LiDAR sweep is represented as an unstructured spatial graph: points are nodes and edges connect nearby points while preserving the original beam-index ordering. A multi-layer GAT learns adaptive attention weights over local geometric neighborhoods and directly regresses the missing elevation (z) values at dropout locations. Trained and evaluated on 1,065 raw KITTI sequences with simulated channel dropout, the method achieves an average height RMSE of 11.67 cm, with 87.98% of reconstructed points falling within a 10 cm error threshold. Inference takes 14.65 seconds per frame on a single GPU, and reconstruction quality remains stable for different neighborhood sizes k. These results show that a pure graph attention model operating solely on raw point-cloud geometry can effectively recover dropped vertical beams under realistic sensor degradation.

---

## 10. M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.12378v1](http://arxiv.org/abs/2512.12378v1)

**作者:** Junqiao Fan, Yunjiao Zhou, Yizhuo Yang, Xinyuan Cui, Jiarui Zhang, Lihua Xie, Jianfei Yang, Chris Xiaoxuan Lu, Fangqiang Ding

**发布时间:** 2025-12-13

### GPT解析

### 总结

本文介绍了M4Human，目前最大规模的多模态人体网格重建基准数据集，包含高分辨率毫米波雷达、RGB和深度数据，旨在解决现有视觉传感方法的局限性。

### 背景

现有人体网格重建数据集主要依赖RGB输入，但视觉传感受限于遮挡、光照变化和隐私问题。虽然毫米波雷达被探索用于隐私保护的室内人体感知，但当前雷达数据集存在骨架标签稀疏、规模有限和动作简单等限制。

### 目的

为了推进人体网格重建研究，作者提出了M4Human数据集，旨在克服现有数据集的局限性，为雷达-based人体建模提供更全面的数据支持。

### 方法

作者构建了M4Human数据集，包含661K帧数据（是之前最大数据集的9倍），提供原始雷达张量(RT)和处理后的雷达点云(RPC)两种数据形式，涵盖20个受试者和50种多样化动作，包括原地、坐姿原地和自由空间运动或康复动作。

### 主要发现

通过在RT和RPC模态以及RGB-D多模态融合上建立的基准实验，突显了M4Human对基于雷达的人体建模的重要性，同时揭示了在快速、不受限制运动下仍存在持续挑战。

### 结论

M4Human作为目前最大规模的多模态人体网格重建基准数据集，为雷达-based人体建模研究提供了重要资源，有助于推动隐私保护的室内人体感知技术的发展。

### 翻译

人体网格重建(HMR)直接提供身体与环境交互的洞察，使各种沉浸式应用成为可能。虽然现有的大规模HMR数据集主要依赖视距RGB输入，但基于视觉的传感受限于遮挡、光照变化和隐私问题。为了克服这些限制，最近的研究探索了射频毫米波雷达用于隐私保护的室内人体感知。然而，当前雷达数据集受到稀疏骨架标签、有限规模和简单原地动作的限制。为了推进HMR研究社区，我们介绍了M4Human，这是目前最大规模（661K帧，是之前最大规模的9倍）的多模态基准，具有高分辨率毫米波雷达、RGB和深度数据。M4Human提供原始雷达张量(RT)和处理后的雷达点云(RPC)，以支持不同级别的射频信号粒度的研究。M4Human包含高质量的运动捕捉(MoCap)注释，具有3D网格和全局轨迹，涵盖20个受试者和50种多样化动作，包括原地、坐姿原地和自由空间运动或康复动作。我们在RT和RPC模态以及RGB-D模态的多模态融合上建立了基准。大量结果突显了M4Human对基于雷达的人体建模的重要性，同时揭示了在快速、不受限制运动下的持续挑战。数据集和代码将在论文发表后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于毫米波雷达的高保真人体网格重建(HMR)数据集不足的问题。这个问题在现实中很重要，因为现有视觉传感方法存在隐私泄露风险、易受光照和遮挡影响，而毫米波雷达可以在保护隐私的同时适应各种环境条件，支持VR游戏、老人护理、康复训练等多种应用场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有HMR数据集的局限性，认识到毫米波雷达作为替代方案的优势。他们设计了一个整合高分辨率毫米波雷达、RGB-D相机和Vicon MoCap系统的统一平台，提供原始雷达张量(RT)和处理后的雷达点云(RPC)两种表示。作者借鉴了现有RGB-based数据集的数据收集方法，但在标注方面采用基于标记的MoCap系统而非依赖RGB相机，提供更高质量的标注。他们还参考了现有毫米波雷达数据集的采集技术，但进行了改进以提高数据质量和多样性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个大规模、多模态的毫米波雷达基准数据集，支持高保真的人体网格重建。整体实现流程包括：1) 系统设置：集成毫米波雷达、RGB-D相机和MoCap系统；2) 数据采集：在多样化场景和距离下采集多模态数据；3) 高质量标注：使用标记式MoCap系统提供精确的3D网格标注；4) 数据集构建：包含661K帧、20名受试者和50个动作类别的多样化数据集，同时提供原始和处理后的雷达数据。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 规模最大的毫米波雷达HMR数据集(661K帧，是之前最大数据集的9倍)；2) 提供四种同步模态(RGB、深度、RT和RPC)；3) 使用标记式MoCap提供高质量3D网格标注；4) 包含50个多样化动作类别(日常、康复和体育)；5) 提出RT-Mesh，首个直接从原始雷达张量进行HMR的方法。相比之前工作，M4Human规模更大、数据质量更高、动作更复杂多样，同时提供原始雷达张量这一新表示，首次支持了基于RT的HMR研究。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'M4Human是一个大规模、多模态的毫米波雷达基准数据集，通过提供高质量标注和多样化动作，首次支持了基于原始雷达张量的人体网格重建，推动了隐私保护、环境适应性强的3D人体感知技术的发展。'}


### 论文摘要

Human mesh reconstruction (HMR) provides direct insights into body-environment interaction, which enables various immersive applications. While existing large-scale HMR datasets rely heavily on line-of-sight RGB input, vision-based sensing is limited by occlusion, lighting variation, and privacy concerns. To overcome these limitations, recent efforts have explored radio-frequency (RF) mmWave radar for privacy-preserving indoor human sensing. However, current radar datasets are constrained by sparse skeleton labels, limited scale, and simple in-place actions. To advance the HMR research community, we introduce M4Human, the current largest-scale (661K-frame) ($9\times$ prior largest) multimodal benchmark, featuring high-resolution mmWave radar, RGB, and depth data. M4Human provides both raw radar tensors (RT) and processed radar point clouds (RPC) to enable research across different levels of RF signal granularity. M4Human includes high-quality motion capture (MoCap) annotations with 3D meshes and global trajectories, and spans 20 subjects and 50 diverse actions, including in-place, sit-in-place, and free-space sports or rehabilitation movements. We establish benchmarks on both RT and RPC modalities, as well as multimodal fusion with RGB-D modalities. Extensive results highlight the significance of M4Human for radar-based human modeling while revealing persistent challenges under fast, unconstrained motion. The dataset and code will be released after the paper publication.

---

## 11. INDOOR-LiDAR: Bridging Simulation and Reality for Robot-Centric 360 degree Indoor LiDAR Perception -- A Robot-Centric Hybrid Dataset

**论文链接:** [http://arxiv.org/abs/2512.12377v1](http://arxiv.org/abs/2512.12377v1)

**作者:** Haichuan Li, Changda Tian, Panos Trahanias, Tomi Westerlund

**发布时间:** 2025-12-13

### GPT解析

### 总结

INDOOR-LIDAR是一个全面的室内3D激光雷达点云混合数据集，旨在推进机器人感知研究

### 背景

现有的室内激光雷达数据集通常存在规模有限、注释格式不一致以及数据收集中人为引起的变异性等问题

### 目的

解决现有数据集的局限性，提供一致且可复现的基准，用于推进复杂室内环境中的机器人感知研究

### 方法

通过整合模拟环境和使用自主地面机器人获取的真实世界扫描创建混合数据集，每个样本包含密集点云数据和KITTI风格注释

### 主要发现

该数据集支持多种应用，包括3D目标检测、鸟瞰图感知、SLAM、语义场景理解和模拟与真实室内领域之间的域适应

### 结论

通过弥合合成和真实世界数据之间的差距，INDOOR-LIDAR建立了可扩展、真实且可复现的基准，用于推进复杂室内环境中的机器人感知

### 翻译

我们提出了INDOOR-LIDAR，这是一个全面的室内3D激光雷达点云混合数据集，旨在推进机器人感知研究。现有的室内激光雷达数据集通常存在规模有限、注释格式不一致以及数据收集中人为引起的变异性等问题。INDOOR-LIDAR通过整合模拟环境和使用自主地面机器人获取的真实世界扫描来解决这些局限性，提供了一致的覆盖范围和受控变化条件下的真实传感器行为。每个样本由密集的点云数据和强度测量值以及KITTI风格的注释组成。注释模式涵盖了各种场景中的常见室内物体类别。模拟子集允许灵活配置布局、点密度和遮挡，而真实世界子集捕获了真实室内环境特有的真实传感器噪声、杂乱和特定领域伪影。INDOOR-LIDAR支持广泛的应用，包括3D目标检测、鸟瞰图感知、SLAM、语义场景理解以及模拟和真实室内领域之间的域适应。通过弥合合成和真实世界数据之间的差距，INDOOR-LIDAR为推进复杂室内环境中的机器人感知建立了可扩展、真实且可复现的基准。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有室内LiDAR数据集的局限性，包括规模有限、标注格式不一致以及手持采集方法导致的视野盲区问题。这个问题很重要，因为室内机器人需要完整360度的环境信息进行安全导航和感知，而现有数据集的局限性使训练出的模型难以直接应用于实际系统，限制了室内机器人感知算法的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有数据集的缺陷，认识到LiDAR在几何一致性方面的优势，决定结合模拟环境和真实扫描创建混合数据集。设计过程中借鉴了Unity和MuJoCo的环境建模、Taichi的高性能光线投射、SUSTechPOINTS标注工具以及KITTI数据格式标准，但创新性地从机器人视角采集数据，确保完整的360度视野覆盖。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个结合模拟与真实数据的混合数据集，从机器人视角采集确保完整视野，提供标准化标注促进算法发展。流程包括：1)使用Unity/MuJoCo和Taichi生成模拟数据；2)用自主地面机器人采集真实数据；3)对模拟数据进行完美标注，真实数据采用半自动标注；4)组织为.bin点云文件和KITTI格式标注；5)支持3D检测、鸟瞰图感知、SLAM等多种应用。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：混合真实与模拟环境、机器人中心360度数据采集、全面的检测基准。相比之前工作，INDOOR-LiDAR避免了手持采集的视野盲区，结合了模拟与真实数据而非单一来源，提供了更全面的标注和基线测试，且包含了强度信息、点级标注、360度视野和3D模拟模型等特性，大多数现有数据集缺乏这些特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'INDOOR-LiDAR通过结合模拟与真实世界的机器人中心360度LiDAR数据，提供了一个规模大、标注一致、无遮挡的混合数据集，有效弥合了室内机器人感知中模拟到现实的差距，为开发和评估鲁棒的室内感知算法建立了新的基准。'}


### 论文摘要

We present INDOOR-LIDAR, a comprehensive hybrid dataset of indoor 3D LiDAR point clouds designed to advance research in robot perception. Existing indoor LiDAR datasets often suffer from limited scale, inconsistent annotation formats, and human-induced variability during data collection. INDOOR-LIDAR addresses these limitations by integrating simulated environments with real-world scans acquired using autonomous ground robots, providing consistent coverage and realistic sensor behavior under controlled variations. Each sample consists of dense point cloud data enriched with intensity measurements and KITTI-style annotations. The annotation schema encompasses common indoor object categories within various scenes. The simulated subset enables flexible configuration of layouts, point densities, and occlusions, while the real-world subset captures authentic sensor noise, clutter, and domain-specific artifacts characteristic of real indoor settings. INDOOR-LIDAR supports a wide range of applications including 3D object detection, bird's-eye-view (BEV) perception, SLAM, semantic scene understanding, and domain adaptation between simulated and real indoor domains. By bridging the gap between synthetic and real-world data, INDOOR-LIDAR establishes a scalable, realistic, and reproducible benchmark for advancing robotic perception in complex indoor environments.

---

## 12. A Framework for Scalable Digital Twin Deployment in Smart Campus Building Facility Management

**论文链接:** [http://arxiv.org/abs/2512.12149v1](http://arxiv.org/abs/2512.12149v1)

**作者:** Thyda Siv

**发布时间:** 2025-12-13

### GPT解析

### 总结

该研究提出了一个全面的数字孪生框架，用于智能校园建筑的可扩展部署，通过整合3D激光扫描、BIM建模和物联网数据可视化来支持设施运营和维护。案例研究表明该框架可实现集中式资产文档、改进系统可见性以及增强维护工作流程。

### 背景

数字孪生为校园设施管理提供了重大机遇，但现有研究往往只关注孤立领域，如点云几何或能源分析，缺乏将建筑几何、设备元数据和运营数据整合到统一平台的可扩展和互操作工作流程。

### 目的

开发一个全面的框架，用于智能校园建筑中可扩展的数字孪生部署，通过整合3D激光扫描、BIM建模和物联网支持的数据可视化，来支持设施运营和维护。

### 方法

包括三个主要部分：(1)使用地面激光扫描和结构化点云处理进行现实捕捉；(2)开发包含建筑、机械、电气、管道、运输和传感器系统的丰富BIM模型；(3)创建数字孪生环境，将设备元数据、维护策略和模拟物联网数据链接到数字孪生管理平台。

### 主要发现

在佐治亚理工学院Price Gilbert大楼的案例研究中，共建模了509个设备项目并嵌入OmniClass分类，开发了10个交互式仪表板可视化系统性能。结果表明该框架实现了集中式资产文档、改进系统可见性和增强维护工作流程。

### 结论

尽管由于现有传感器基础设施有限，大多数物联网数据是模拟的，但该原型验证了可扩展数字孪生用于设施管理的可行性，并为实时监控、分析集成和未来自主建筑运营建立了参考模型。

### 翻译

数字孪生(DT)为校园环境中的设施管理(FM)提供了重大机遇。然而，现有研究往往只专注于孤立领域，如点云几何或能源分析，没有提供可扩展和互操作的工作流程，将建筑几何、设备元数据和运营数据整合到统一的FM平台中。本研究提出了一个全面的框架，用于智能校园建筑中可扩展的数字孪生部署，通过整合3D激光扫描、BIM建模和物联网支持的数据可视化来支持设施运营和维护。该方法包括：(1)使用地面激光扫描和结构化点云处理进行现实捕捉；(2)开发包含建筑、机械、电气、管道、运输和传感器系统的丰富BIM模型；(3)创建数字孪生环境，将设备元数据、维护策略和模拟的物联网数据链接到数字孪生管理平台。佐治亚理工学院Price Gilbert大楼的案例研究展示了此工作流程的实施。共建模了509个设备项目，并将其嵌入到带有OmniClass分类的数字孪生中。开发了10个交互式仪表板来可视化系统性能。结果表明，提出的框架实现了集中式资产文档编制、改进的系统可见性以及增强的预防和反应性维护工作流程。尽管由于现有传感器基础设施有限，大多数物联网数据是模拟的，但原型验证了可扩展数字孪生用于设施管理的可行性，并为实时监控、分析集成和未来自主建筑运营建立了参考模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决数字孪生技术在校园建筑设施管理中缺乏可扩展、可互工作流程的问题，现有研究通常只关注单一领域(如点云几何或能源分析)，没有将建筑几何、设备元数据和运营数据整合到统一的设施管理平台。这个问题很重要，因为传统设施管理依赖人工检查和纸质文档，导致效率低下、维护成本高、设备故障难以及时发现，而数字孪生技术可以提供实时监测、预测维护和数据驱动的决策支持，从而提高校园建筑的运营效率、可持续性和用户满意度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了校园设施管理的挑战，包括缺乏实时可见性、被动维护、资产信息碎片化等问题，然后研究了数字孪生技术的潜在应用。通过文献 review 发现现有研究的不足，特别是缺乏建筑规模的参考模型和AI在核心建筑系统中的应用细节。作者借鉴了多项现有工作，包括Lu等人在建筑和城市规模数字孪生的应用、点云驱动的工作流程、基于工具的数字孪生架构、结合IoT数据与BIM的能源研究，以及数字孪生与移动IoT传感技术的整合。基于这些基础，作者设计了一个整合3D激光扫描、BIM建模和IoT数据可视化的综合框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将物理校园建筑转化为动态的数字孪生，整合3D几何模型、设备元数据和实时运营数据，通过统一的数字孪生平台实现集中化的资产文档、提高建筑系统可见性、增强预防性和反应性维护工作流程。整体实现流程分为三个阶段：1)激光扫描阶段：使用地面激光扫描仪捕获建筑空间数据，进行数据采集、配准和处理，生成统一的点云文件；2)BIM模型开发阶段：使用点云数据在Revit中创建包含建筑、机械、电气、管道等系统的三维模型，并为每个元素添加元数据；3)数字孪生开发阶段：创建数字孪生环境，部署IoT传感器网络，使用数字孪生管理平台整合BIM模型和实时数据流，开发交互式仪表板可视化系统性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出了针对校园建筑设施管理的全面可扩展数字孪生部署框架；2)整合了3D激光扫描、BIM建模和IoT数据可视化；3)开发了详细的实施方法，包括点云处理、BIM模型增强和数字孪生环境创建；4)在Price Gilbert建筑案例中实现了该工作流程，建模了509个设备项目；5)开发了10个交互式仪表板可视化关键系统性能。相比之前工作，该框架更全面，不仅关注单一技术领域，还提供了可扩展、可互操作的工作流程；不仅关注技术实现，还整合了设施管理工作流程；不仅使用模拟数据，还结合了实际建筑案例验证可行性；特别关注核心建筑系统(如HVAC、电气系统)的数字孪生应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文贡献了一个整合3D激光扫描、BIM建模和IoT数据可视化的可扩展数字孪生框架，为校园建筑设施管理提供了集中化资产文档、提高系统可见性和增强维护工作流程的创新解决方案。'}


### 论文摘要

Digital twin (DT) offers significant opportunities for enhancing facility management (FM) in campus environments. However, existing research often focuses narrowly on isolated domains, such as point-cloud geometry or energy analytics, without providing a scalable and interoperable workflow that integrates building geometry, equipment metadata, and operational data into a unified FM platform. This study proposes a comprehensive framework for scalable digital-twin deployment in smart campus buildings by integrating 3D laser scanning, BIM modeling, and IoT-enabled data visualization to support facility operations and maintenance. The methodology includes: (1) reality capture using terrestrial laser scanning and structured point-cloud processing; (2) development of an enriched BIM model incorporating architectural, mechanical, electrical, plumbing, conveying, and sensor systems; and (3) creation of a digital-twin environment that links equipment metadata, maintenance policies, and simulated IoT data within a digital-twin management platform. A case study of the Price Gilbert Building at Georgia Tech demonstrates the implementation of this workflow. A total of 509 equipment items were modeled and embedded with OmniClass classifications into the digital twin. Ten interactive dashboards were developed to visualize system performance. Results show that the proposed framework enables centralized asset documentation, improved system visibility, and enhanced preventive and reactive maintenance workflows. Although most IoT data were simulated due to limited existing sensor infrastructure, the prototype validates the feasibility of a scalable digital twin for facility management and establishes a reference model for real-time monitoring, analytics integration, and future autonomous building operations.

---

## 13. Exploring Spatial-Temporal Representation via Star Graph for mmWave Radar-based Human Activity Recognition

**论文链接:** [http://arxiv.org/abs/2512.12013v1](http://arxiv.org/abs/2512.12013v1)

**作者:** Senhao Gao, Junqing Zhang, Luoyu Mei, Shuai Wang, Xuyu Wang

**发布时间:** 2025-12-12

**DOI:** 10.1109/TMC.2025.3634221

### GPT解析

### 总结

本文提出了一种基于毫米波雷达点云的人体活动识别系统，使用离散动态图神经网络(DDGNN)处理点云稀疏性和可变尺寸问题，实现了高精度的活动分类。

### 背景

人体活动识别需要提取准确的人体运动时空特征，但毫米波雷达点云系统由于信号物理特性存在稀疏性和可变尺寸问题，现有基于视觉的密集点云预处理方法可能不适用于毫米波雷达系统。

### 目的

设计一种适合毫米波雷达点云特性的方法，解决其稀疏性和可变尺寸问题，提高人体活动识别准确率。

### 方法

提出了一种图表示方法，使用离散动态图神经网络(DDGNN)探索人体运动特征的时空表示；设计了星形图描述静态中心点与动态毫米波雷达点间的高维相对关系；采用DDGNN学习可变尺寸星形图中的特征。

### 主要发现

实验结果表明该方法优于其他基线方法；系统总体分类准确率达94.27%，接近基于视觉骨骼数据的97.25%最优性能；在树莓派4上测试证明了其在资源受限平台上的有效性；系统优于三种最近的雷达特定方法，无需重采样或帧聚合器。

### 结论

所提出的基于图表示和DDGNN的方法能有效处理毫米波雷达点云的稀疏性和可变尺寸问题，实现高精度的人体活动识别，适用于资源受限平台。

### 翻译

人体活动识别需要提取准确的人体运动时空特征。毫米波雷达点云系统由于毫米波信号的物理特性，存在稀疏性和可变尺寸问题。现有工作通常借鉴基于视觉的密集点云系统的预处理算法，但这些方法可能不适用于毫米波雷达系统。在这项工作中，我们提出了一种使用离散动态图神经网络的图表示方法，用于探索人体运动相关特征的时空表示。具体而言，我们设计了一种星形图来描述手动添加的静态中心点与同一帧和连续帧中的动态毫米波雷达点之间的高维相对关系。然后，我们采用DDGNN来学习可变尺寸星形图中的特征。实验结果表明，我们的方法在使用真实世界人体活动识别数据集的其他基线方法中表现更优。我们的系统总体分类准确率达到94.27%，接近基于视觉骨骼数据的97.25%的最优性能。我们还在树莓派4上进行了推理测试，证明了其在资源受限平台上的有效性。我们提供了针对可变DDGNN结构的全面消融研究，以验证我们的模型设计。我们的系统还优于三种最近的雷达特定方法，无需重采样或帧聚合器。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于毫米波雷达的人体活动识别中的点云数据稀疏性和尺寸变化问题。这个问题很重要，因为毫米波雷达可以保护隐私（不同于摄像头），适合在浴室等敏感环境使用，同时能提供与视觉传感器互补的功能，在人机交互和机器人控制等领域有广泛应用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有视觉传感器HAR方法的局限性，指出它们不适合毫米波雷达的稀疏点云。作者借鉴了图神经网络在视觉HAR中的应用，特别是骨架图概念，但针对毫米波雷达特点进行了创新设计。作者观察到毫米波雷达点云虽稀疏但能捕捉人体动态部分，因此设计了星形图结构，将静态参考点与动态雷达点连接，并开发了DDGNN模型来处理可变大小的图结构，无需额外预处理算法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用星形图表示人体活动，其中包含一个静态中心点和动态雷达点，捕捉它们之间的相对关系；使用DDGNN模型处理可变大小的图结构。整体流程包括：1)点云预处理（噪声减少和DBSCAN聚类）；2)图生成（添加静态中心点并构建星形图）；3)DDGNN模型处理（空间特征提取器使用两层图卷积网络，时间特征提取器使用双向LSTM）；4)分类器生成预测结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)星形图表示，专注于静态中心点和动态雷达点的相对关系；2)DDGNN模型，能处理可变大小的图结构；3)无需重采样或帧聚合器；4)在资源受限平台上的高效实现。相比之前工作，不同于视觉骨架图方法依赖密集点云，也不同于其他基于图的方法（如MMPointGNN和Tesla-Rapture）的高计算复杂度或需要重采样，同时避免了体素化方法在稀疏点云中产生大量空体素的问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '该论文提出了一种基于星形图表示和离散动态图神经网络的毫米波雷达人体活动识别方法，有效解决了点云数据稀疏性和尺寸变化问题，实现了94.27%的分类准确率，接近基于视觉骨架数据的性能，同时在资源受限平台上展现出高效性。'}


### 论文摘要

Human activity recognition (HAR) requires extracting accurate spatial-temporal features with human movements. A mmWave radar point cloud-based HAR system suffers from sparsity and variable-size problems due to the physical features of the mmWave signal. Existing works usually borrow the preprocessing algorithms for the vision-based systems with dense point clouds, which may not be optimal for mmWave radar systems. In this work, we proposed a graph representation with a discrete dynamic graph neural network (DDGNN) to explore the spatial-temporal representation of human movement-related features. Specifically, we designed a star graph to describe the high-dimensional relative relationship between a manually added static center point and the dynamic mmWave radar points in the same and consecutive frames. We then adopted DDGNN to learn the features residing in the star graph with variable sizes. Experimental results demonstrated that our approach outperformed other baseline methods using real-world HAR datasets. Our system achieved an overall classification accuracy of 94.27\%, which gets the near-optimal performance with a vision-based skeleton data accuracy of 97.25\%. We also conducted an inference test on Raspberry Pi~4 to demonstrate its effectiveness on resource-constraint platforms. \sh{ We provided a comprehensive ablation study for variable DDGNN structures to validate our model design. Our system also outperformed three recent radar-specific methods without requiring resampling or frame aggregators.

---

## 14. TransBridge: Boost 3D Object Detection by Scene-Level Completion with Transformer Decoder

**论文链接:** [http://arxiv.org/abs/2512.11926v1](http://arxiv.org/abs/2512.11926v1)

**作者:** Qinghao Meng, Chenming Wu, Liangjun Zhang, Jianbing Shen

**发布时间:** 2025-12-12

**DOI:** 10.1109/TITS.2025.3617527

**备注:** 12 pages, 9 figures

### GPT解析

### 总结

这篇论文提出了一种名为TransBridge的新型transformer上采样块和DSRecon模块，通过联合完成和检测框架提高了自动驾驶中远距离稀疏点云区域的3D物体检测性能。

### 背景

3D物体检测在自动驾驶中至关重要，能提供关于移动物体和障碍物的关键信息。然而，在远距离区域只有少量LiDAR点的情况下进行检测仍然是一个挑战，许多策略已开发出来通过密集化来解决点云稀疏问题。

### 目的

提出一个联合完成和检测框架，在保持成本不变的同时提高稀疏区域的检测特征。

### 方法

提出了TransBridge，一种基于transformer的上采样块，融合了检测和完成网络的特征；设计了动态-静态重建(DSRecon)模块，为完成网络生成密集的LiDAR数据；使用transformer机制建立通道和空间关系之间的连接，生成用于完成的高分辨率特征图。

### 主要发现

在nuScenes和Waymo数据集上的实验证明了框架的有效性；该框架一致地改进了端到端的3D物体检测，平均精度(mAP)在多种方法中从0.7到1.5不等，表明其泛化能力；对于两阶段检测框架，mAP提升了高达5.78个百分点。

### 结论

所提出的联合完成和检测框架有效解决了远距离稀疏点云的3D物体检测问题，显著提升了检测性能。

### 翻译

三维物体检测在自动驾驶中至关重要，它提供了关于移动物体和障碍物的关键信息。在只有少量LiDAR点的远距离区域进行检测仍然是一个挑战，已经开发出许多策略通过密集化来解决点云稀疏问题。本文提出了一个联合完成和检测框架，在保持成本不变的同时提高了稀疏区域的检测特征。具体来说，我们提出了TransBridge，这是一种新颖的基于transformer的上采样块，融合了检测和完成网络的特征。检测网络可以从获取由完成网络派生的隐式完成特征中受益。此外，我们设计了动态-静态重建(DSRecon)模块，为完成网络生成密集的LiDAR数据，满足密集点云地面真实值的需求。此外，我们采用transformer机制建立通道和空间关系之间的连接，生成用于完成的高分辨率特征图。在nuScenes和Waymo数据集上的大量实验证明了所提框架的有效性。结果表明，我们的框架一致地改进了端到端的3D物体检测，在多种方法中的平均精度(mAP)从0.7到1.5不等，表明其泛化能力。对于两阶段检测框架，它也将mAP提升了高达5.78个百分点。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D物体检测中的点云稀疏性问题，特别是在自动驾驶中远处区域LiDAR点数据稀少导致的物体检测困难。这个问题在现实中非常重要，因为准确检测远处物体对自动驾驶系统的安全决策至关重要，稀疏点云会导致漏检和误检，影响系统可靠性和安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了LiDAR数据的固有稀疏性和非均匀性是3D物体检测的主要挑战，评估了现有方法如虚拟点云、深度信息增强和点云补全等技术的局限性。作者设计了一个联合检测和补全的框架，借鉴了CenterPoint等检测器作为基础，同时改进了点云补全方法。核心创新是设计了TransBridge模块，利用transformer机制连接检测和补全特征，并引入DSRecon模块生成高质量训练数据，借鉴了表面重建技术来处理点云数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过联合训练3D物体检测和场景级点云补全，改善稀疏区域的检测特征，使检测网络能够从补全网络获取隐式特征。整体流程包括：1)接收LiDAR点云并转换为体素；2)使用共享编码器生成多尺度检测特征图；3)通过检测头生成3D边界框；4)TransBridge模块处理特征，包含上采样桥(UB)和解释桥(IB)；5)使用稀疏控制模块(SCM)保持计算效率；6)利用DSRecon生成的密集点云作为监督信号；7)输出补全特征图和检测结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)联合检测和补全框架，共享编码器但有独立输出模块；2)TransBridge模块，包含上采样桥和解释桥，补偿信息损失并解释不同语义；3)动态-静态重建(DSRecon)模块，生成高质量密集点云训练数据；4)稀疏控制模块(SCM)，确保高效计算。相比之前工作，本文将检测和补全联合训练而非独立训练，保持原始推理速度而非增加计算需求，培养出能更好区分空体素的鲁棒特征提取器，并通过DSRecon减少了远处区域的噪声。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TransBridge通过联合训练3D物体检测和场景级点云补全，使用基于transformer的桥接模块和动态-静态重建技术，显著提高了自动驾驶中远处稀疏区域物体的检测精度，同时保持了原始推理效率。'}


### 论文摘要

3D object detection is essential in autonomous driving, providing vital information about moving objects and obstacles. Detecting objects in distant regions with only a few LiDAR points is still a challenge, and numerous strategies have been developed to address point cloud sparsity through densification.This paper presents a joint completion and detection framework that improves the detection feature in sparse areas while maintaining costs unchanged. Specifically, we propose TransBridge, a novel transformer-based up-sampling block that fuses the features from the detection and completion networks.The detection network can benefit from acquiring implicit completion features derived from the completion network. Additionally, we design the Dynamic-Static Reconstruction (DSRecon) module to produce dense LiDAR data for the completion network, meeting the requirement for dense point cloud ground truth.Furthermore, we employ the transformer mechanism to establish connections between channels and spatial relations, resulting in a high-resolution feature map used for completion purposes.Extensive experiments on the nuScenes and Waymo datasets demonstrate the effectiveness of the proposed framework.The results show that our framework consistently improves end-to-end 3D object detection, with the mean average precision (mAP) ranging from 0.7 to 1.5 across multiple methods, indicating its generalization ability. For the two-stage detection framework, it also boosts the mAP up to 5.78 points.

---

## 15. FloraForge: LLM-Assisted Procedural Generation of Editable and Analysis-Ready 3D Plant Geometric Models For Agricultural Applications

**论文链接:** [http://arxiv.org/abs/2512.11925v1](http://arxiv.org/abs/2512.11925v1)

**作者:** Mozhgan Hadadi, Talukder Z. Jubery, Patrick S. Schnable, Arti Singh, Bedrich Benes, Adarsh Krishnamurthy, Baskar Ganapathysubramanian

**发布时间:** 2025-12-11

### GPT解析

### 总结

FloraForge是一个LLM辅助框架，使植物科学领域专家能够通过自然语言交互生成生物准确的完全参数化3D植物模型，无需专业编程知识。

### 背景

准确的3D植物模型对计算表型和基于物理的模拟至关重要，但当前方法存在局限性：基于学习的方法需要大量特定物种训练数据且缺乏可编辑性；程序建模需要专业几何建模知识和复杂程序规则的深入理解。

### 目的

开发一个使植物科学领域专家能够轻松生成高质量3D植物模型的框架，无需专业编程或几何建模知识。

### 方法

利用LLM协同设计改进生成参数化植物几何的Python脚本，植物几何表示为具有植物约束的层次B样条曲面，包含控制点和参数变形函数。通过植物描述符(PD)文件进行手动细化，将模型拟合到经验点云数据。

### 主要发现

框架成功应用于玉米、大豆和绿豆的3D建模，能够生成双重输出：用于可视化的三角形网格和带有额外参数元数据用于定量分析的三角形网格。该方法结合了LLM辅助模板创建、数学连续表示和直接参数控制。

### 结论

FloraForge使植物科学领域能够民主化复杂的几何建模，同时保持数学严谨性，为植物科学研究提供了强大而易用的工具。

### 翻译

准确的3D植物模型对计算表型和基于物理的模拟至关重要；然而，当前方法面临显著限制。基于学习的方法需要大量特定物种的训练数据且缺乏可编辑性。程序建模提供参数控制，但需要几何建模的专业知识和对复杂程序规则的深入理解，使得领域科学家难以使用。我们提出了FloraForge，这是一个LLM辅助框架，使领域专家能够通过迭代自然语言植物细化(PR)生成生物准确的完全参数化3D植物模型，最小化编程专业知识需求。我们的框架利用LLM协同设计来改进生成参数化植物几何的Python脚本，这些几何作为具有植物约束的层次B样条曲面表示，包含显式控制点和参数变形函数。这种表示可以轻松细分为任意精度的多边形网格，确保与功能结构植物分析工作流程的兼容性，如光模拟、计算流体动力学和有限元分析。我们在玉米、大豆和绿豆上展示了该框架，通过手动细化植物描述符(PD)（人类可读文件）将程序模型拟合到经验点云数据。该流程生成双重输出：用于可视化的三角形网格和带有额外参数元数据用于定量分析的三角形网格。这种方法独特地结合了LLM辅助模板创建、同时支持表型和渲染的数学连续表示以及通过PD的直接参数控制。该框架使植物科学能够民主化复杂的几何建模，同时保持数学严谨性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决农业应用中生成准确、可编辑且分析就绪的3D植物几何模型的挑战。这个问题很重要，因为精确的植物模型对计算表型分析和基于物理的模拟至关重要，可用于植物育种、农学研究等场景，但现有方法要么需要大量训练数据且不可编辑，要么需要专业几何知识，限制了植物科学中先进3D建模的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有方法(学习重建、传统程序建模)的局限性，然后利用大型语言模型(LLM)在代码生成和多模态推理方面的能力。作者借鉴了多个领域的工作：程序植物建模(L-systems)、逆程序建模、学习植物重建、参数几何表示(NURBS/B样条)和LLM辅助3D建模。设计了一个两阶段流程：第一阶段通过自然语言对话让LLM生成程序植物生成器和植物描述符；第二阶段通过调整参数使模型与实际点云数据匹配，实现了无需专业编程知识即可生成植物模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用LLM辅助生成程序化的植物几何模型，使植物专家能通过自然语言描述和可编辑参数文件生成准确3D植物模型。整体流程分两阶段：1) LLM辅助模板设计：用户通过自然语言'植物精炼'与LLM交互，生成Python程序生成器和YAML参数文件，创建初始3D模板；2) 手动参数调整：专家调整参数文件使模型匹配实际点云数据。植物器官用B样条曲面表示，支持单子叶和双子叶不同植物架构，输出包括可视化网格和分析就绪的参数化模型。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) LLM辅助程序植物生成器，无需编程知识；2) 分析就绪的连续B样条曲面表示，兼具数学严谨性和可视化能力；3) 人类可读的参数控制界面，实现分层参数继承；4) 跨物种验证能力。相比之前工作：1) 学习重建方法需大量训练数据且不可编辑，而FloraForge无需训练数据且完全可编辑；2) 传统程序建模需专业几何知识，FloraForge通过LLM简化了模板创建；3) 逆程序方法依赖预定义模板，FloraForge可自动适应新物种；4) 现有LLM 3D系统缺乏植物领域知识，FloraForge专为植物形态学设计。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FloraForge通过利用大型语言模型自动化程序化植物模板创建，使植物科学家无需编程专业知识即可生成可编辑、分析就绪且生物准确的3D植物几何模型，解决了传统方法中可访问性与分析效用之间的根本矛盾。'}


### 论文摘要

Accurate 3D plant models are crucial for computational phenotyping and physics-based simulation; however, current approaches face significant limitations. Learning-based reconstruction methods require extensive species-specific training data and lack editability. Procedural modeling offers parametric control but demands specialized expertise in geometric modeling and an in-depth understanding of complex procedural rules, making it inaccessible to domain scientists. We present FloraForge, an LLM-assisted framework that enables domain experts to generate biologically accurate, fully parametric 3D plant models through iterative natural language Plant Refinements (PR), minimizing programming expertise. Our framework leverages LLM-enabled co-design to refine Python scripts that generate parameterized plant geometries as hierarchical B-spline surface representations with botanical constraints with explicit control points and parametric deformation functions. This representation can be easily tessellated into polygonal meshes with arbitrary precision, ensuring compatibility with functional structural plant analysis workflows such as light simulation, computational fluid dynamics, and finite element analysis. We demonstrate the framework on maize, soybean, and mung bean, fitting procedural models to empirical point cloud data through manual refinement of the Plant Descriptor (PD), human-readable files. The pipeline generates dual outputs: triangular meshes for visualization and triangular meshes with additional parametric metadata for quantitative analysis. This approach uniquely combines LLM-assisted template creation, mathematically continuous representations enabling both phenotyping and rendering, and direct parametric control through PD. The framework democratizes sophisticated geometric modeling for plant science while maintaining mathematical rigor.

---

## 16. LightTopoGAT: Enhancing Graph Attention Networks with Topological Features for Efficient Graph Classification

**论文链接:** [http://arxiv.org/abs/2512.13617v1](http://arxiv.org/abs/2512.13617v1)

**作者:** Ankit Sharma, Sayan Roy Gupta

**发布时间:** 2025-12-15

**备注:** 9 pages

### GPT解析

### 总结

LightTopoGAT是一种轻量级图注意力网络，通过拓扑增强提高节点特征表示，在图分类任务中实现了优越性能且不增加架构复杂性。

### 背景

图神经网络在图分类任务中取得显著成功，但通常需要大量计算资源且难以有效捕捉全局图特性。

### 目的

引入LightTopoGAT，一种轻量级图注意力网络，通过拓扑增强增强节点特征，改进图表示学习。

### 方法

通过整合节点度和局部聚类系数进行拓扑增强，利用简化的注意力机制保持参数效率，并整合局部消息传递方案通常忽略的结构信息。

### 主要发现

在MUTAG、ENZYMES和PROTEINS三个基准数据集上，相比GCN、GraphSAGE和标准GAT等基线，LightTopoGAT在MUTAG上准确率提高6.6%，在PROTEINS上提高2.2%；消融研究证实性能提升直接来自于拓扑特征的引入。

### 结论

LightTopoGAT展示了简单而有效的策略来增强图神经网络性能，无需增加架构复杂性。

### 翻译

图神经网络在图分类任务中已经展示了显著的成功，但它们通常需要大量的计算资源，并且难以有效捕捉全局图特性。我们引入了LightTopoGAT，一种轻量级图注意力网络，通过拓扑增强来增强节点特征，通过整合节点度和局部聚类系数来改进图表示学习。所提出的方法通过简化的注意力机制保持参数效率，同时整合了局部消息传递方案通常忽略的结构信息。通过在MUTAG、ENZYMES和PROTEINS三个基准数据集上的全面实验，我们表明LightTopoGAT相比包括GCN、GraphSAGE和标准GAT在内的既定基线实现了优越性能，在MUTAG上准确率提高了6.6%，在PROTEINS上提高了2.2%。消融研究进一步证实，这些性能提升直接来自于拓扑特征的包含，展示了一种简单而有效的策略来增强图神经网络性能，而无需增加架构复杂性。


### 论文摘要

Graph Neural Networks have demonstrated significant success in graph classification tasks, yet they often require substantial computational resources and struggle to capture global graph properties effectively. We introduce LightTopoGAT, a lightweight graph attention network that enhances node features through topological augmentation by incorporating node degree and local clustering coefficient to improve graph representation learning. The proposed approach maintains parameter efficiency through streamlined attention mechanisms while integrating structural information that is typically overlooked by local message passing schemes. Through comprehensive experiments on three benchmark datasets, MUTAG, ENZYMES, and PROTEINS, we show that LightTopoGAT achieves superior performance compared to established baselines including GCN, GraphSAGE, and standard GAT, with a 6.6 percent improvement in accuracy on MUTAG and a 2.2 percent improvement on PROTEINS. Ablation studies further confirm that these performance gains arise directly from the inclusion of topological features, demonstrating a simple yet effective strategy for enhancing graph neural network performance without increasing architectural complexity.

---

## 17. OPAL: Operator-Programmed Algorithms for Landscape-Aware Black-Box Optimization

**论文链接:** [http://arxiv.org/abs/2512.12809v1](http://arxiv.org/abs/2512.12809v1)

**作者:** Junbo Jacob Lian, Mingyang Yu, Kaichen Ouyang, Shengwei Fu, Rui Zhong, Yujun Zhang, Jun Zhang, Huiling Chen

**发布时间:** 2025-12-14

**备注:** Source code, experiment scripts, and results are publicly available at https://github.com/junbolian/OPAL. The real-world application part hasn't been done yet

### GPT解析

### 总结

本文提出了一种名为操作符编程算法(OPAL)的黑盒优化新框架，通过元学习为每个问题实例单独设计优化策略，在CEC 2017测试套件上取得了与最先进算法相当的性能。

### 背景

黑盒优化通常依赖进化和群体智能算法，但这些算法的性能高度依赖于具体问题，且大多数现有方法是基于启发式比喻的。

### 目的

开发一种景观感知的框架，能够针对每个问题实例单独学习优化策略，以提升连续黑盒优化性能。

### 方法

将优化器视为小型搜索操作符程序，使用标准差分进化基线探测问题景观，构建采样点的k最近邻图，通过图神经网络编码轨迹，最后由元学习器生成分阶段的探索、重启和局部搜索操作符调度。

### 主要发现

单个元训练的OPAL策略在CEC 2017测试套件上与最先进的自适应差分进化算法具有统计竞争力，且显著优于简单基线；消融研究验证了设计阶段、轨迹图和操作符编程表示的选择；元组件仅增加适度的计算开销。

### 结论

操作符编程结合景观感知的每实例设计是黑盒优化中超越传统启发式比喻算法的有效实用方法。

### 翻译

黑盒优化通常依赖于进化和群体智能算法，其性能高度依赖于问题本身。我们将优化器视为一个小型搜索操作符的程序，并为每个问题实例单独学习这个操作符程序。我们在操作符编程算法(OPAL)中实现了这一想法，这是一个用于连续黑盒优化的景观感知框架，它使用标准差分进化基线以小的设计预算探测景观，构建采样点的k最近邻图，并使用图神经网络编码这一轨迹。然后，元学习器将得到的表示映射到分阶段的探索、重启和局部搜索操作符调度。在CEC 2017测试套件上，单个元训练的OPAL策略与最先进的自适应差分进化变体具有统计竞争力，并在非参数测试下比简单基线方法取得显著改进。对CEC 2017的消融研究验证了设计阶段、轨迹图和操作符编程表示的选择，同时元组件仅增加了适度的时钟开销。总体而言，结果表明操作符编程、景观感知的每实例设计是黑盒优化中超越启发式比喻算法的实用前进方向。


### 论文摘要

Black-box optimization often relies on evolutionary and swarm algorithms whose performance is highly problem dependent. We view an optimizer as a short program over a small vocabulary of search operators and learn this operator program separately for each problem instance. We instantiate this idea in Operator-Programmed Algorithms (OPAL), a landscape-aware framework for continuous black-box optimization that uses a small design budget with a standard differential evolution baseline to probe the landscape, builds a $k$-nearest neighbor graph over sampled points, and encodes this trajectory with a graph neural network. A meta-learner then maps the resulting representation to a phase-wise schedule of exploration, restart, and local search operators. On the CEC~2017 test suite, a single meta-trained OPAL policy is statistically competitive with state-of-the-art adaptive differential evolution variants and achieves significant improvements over simpler baselines under nonparametric tests. Ablation studies on CEC~2017 justify the choices for the design phase, the trajectory graph, and the operator-program representation, while the meta-components add only modest wall-clock overhead. Overall, the results indicate that operator-programmed, landscape-aware per-instance design is a practical way forward beyond ad hoc metaphor-based algorithms in black-box optimization.

---

## 18. DynaGen: Unifying Temporal Knowledge Graph Reasoning with Dynamic Subgraphs and Generative Regularization

**论文链接:** [http://arxiv.org/abs/2512.12669v1](http://arxiv.org/abs/2512.12669v1)

**作者:** Jiawei Shen, Jia Zhu, Hanghui Guo, Weijie Shi, Guoqing Ma, Yidan Liang, Jingjiang Liu, Hao Chen, Shimin Di

**发布时间:** 2025-12-14

### GPT解析

### 总结

DynaGen是一种统一的时间知识图谱推理方法，通过动态构建实体中心子图和条件扩散过程，有效解决了内插和外推任务中的关键挑战，在多个基准数据集上实现了最先进的性能。

### 背景

时间知识图谱推理(TKGR)旨在完成时间线上的缺失事实元素。根据查询时间位置，任务分为内插和外推。现有内插方法通常将时间信息嵌入单个事实中完成历史知识，外推技术则利用图快照序列模型识别重复模式预测未来事件。这些方法面临内插上下文建模有限和外推认知泛化偏差两大挑战。

### 目的

提出一个统一的TKGR方法，解决现有方法在内插和外推方面的关键挑战，提高预测性能。

### 方法

DynaGen方法包含两部分：对于内插，动态构建以实体为中心的子图，使用协同双分支GNN编码器处理以捕获不断变化的上下文结构；对于外推，应用条件扩散过程，迫使模型学习底层进化原理而非表面模式，增强预测未见未来事件的能力。

### 主要发现

在六个基准数据集上的大量实验表明，DynaGen实现了最先进的性能。与第二好的模型相比，DynaGen将内插的倒数排名平均值(MRR)提高了2.61分，外推提高了1.45分。

### 结论

DynaGen通过统一的方法有效解决了时间知识图谱推理中的内插和外推挑战，实验证明其在多个数据集上显著优于现有方法。

### 翻译

时间知识图谱推理(TKGR)旨在完成时间线上的缺失事实元素。根据查询的时间位置，任务分为内插和外推。现有的内插方法通常将时间信息嵌入到单个事实中以完成缺失的历史知识，而外推技术则经常利用图快照上的序列模型来识别重复模式用于未来事件预测。这些方法面临两个关键挑战：内插中的上下文建模有限和外推中的认知泛化偏差。为解决这些问题，我们提出了一个统一的TKGR方法，命名为DynaGen。对于内插，DynaGen动态构建以实体为中心的子图，并使用协同双分支GNN编码器处理它们，以捕获不断变化的上下文结构。对于外推，它应用条件扩散过程，迫使模型学习底层进化原理而非表面模式，增强其预测未见未来事件的能力。在六个基准数据集上的大量实验表明，DynaGen实现了最先进的性能。平均而言，与第二好的模型相比，DynaGen将内插的倒数排名平均值(MRR)提高了2.61分，外推提高了1.45分。


### 论文摘要

Temporal Knowledge Graph Reasoning (TKGR) aims to complete missing factual elements along the timeline. Depending on the temporal position of the query, the task is categorized into interpolation and extrapolation. Existing interpolation methods typically embed temporal information into individual facts to complete missing historical knowledge, while extrapolation techniques often leverage sequence models over graph snapshots to identify recurring patterns for future event prediction. These methods face two critical challenges: limited contextual modeling in interpolation and cognitive generalization bias in extrapolation. To address these, we propose a unified method for TKGR, dubbed DynaGen. For interpolation, DynaGen dynamically constructs entity-centric subgraphs and processes them with a synergistic dual-branch GNN encoder to capture evolving structural context. For extrapolation, it applies a conditional diffusion process, which forces the model to learn underlying evolutionary principles rather than just superficial patterns, enhancing its ability to predict unseen future events. Extensive experiments on six benchmark datasets show DynaGen achieves state-of-the-art performance. On average, compared to the second-best models, DynaGen improves the Mean Reciprocal Rank (MRR) score by 2.61 points for interpolation and 1.45 points for extrapolation.

---

## 19. Modeling Authorial Style in Urdu Novels Using Character Interaction Graphs and Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.12654v1](http://arxiv.org/abs/2512.12654v1)

**作者:** Hassan Mujtaba, Hamza Naveed, Hanzlah Munir

**发布时间:** 2025-12-14

**备注:** 6 pages

### GPT解析

### 总结

本研究提出了一种基于图的框架，将乌尔都语小说建模为角色交互网络，以检验仅从叙事结构是否可以推断作者风格。

### 背景

作者分析传统上关注文本中的词汇和风格线索，而叙事结构等更高级别的特征研究不足，特别是在低资源语言如乌尔都语方面。

### 目的

探索仅通过叙事结构而非传统的词汇和风格特征来推断乌尔都语小说作者风格的可行性。

### 方法

将每部乌尔都语小说表示为图，节点对应角色，边表示他们在叙事中的共现；系统比较多种图表示方法，包括全局结构特征、节点级语义摘要、无监督图嵌入和监督图神经网络。

### 主要发现

在52部由7位作者撰写的乌尔都语小说数据集上，学习的图表示显著优于手工制作和无监督基线，在严格的作者感知评估协议下准确率达到0.857。

### 结论

通过角色交互网络建模叙事结构可以有效推断乌尔都语小说的作者风格，证明了叙事结构在作者分析中的重要性。

### 翻译

作者分析传统上关注文本中的词汇和风格线索，而更高层次的叙事结构研究不足，特别是对于乌尔都语这样的低资源语言。这项工作提出了一种基于图的框架，将乌尔都语小说建模为角色交互网络，以检验仅从叙事结构是否可以推断作者风格。每部小说被表示为一个图，其中节点对应角色，边表示他们在叙事邻近中的共现。我们系统地比较了多种图表示方法，包括全局结构特征、节点级语义摘要、无监督图嵌入和监督图神经网络。在包含52部由七位作者撰写的乌尔都语小说的数据集上的实验表明，学习的图表示显著优于手工制作和无监督基线，在严格的作者感知评估协议下达到0.857的准确率。


### 论文摘要

Authorship analysis has traditionally focused on lexical and stylistic cues within text, while higher-level narrative structure remains underexplored, particularly for low-resource languages such as Urdu. This work proposes a graph-based framework that models Urdu novels as character interaction networks to examine whether authorial style can be inferred from narrative structure alone. Each novel is represented as a graph where nodes correspond to characters and edges denote their co-occurrence within narrative proximity. We systematically compare multiple graph representations, including global structural features, node-level semantic summaries, unsupervised graph embeddings, and supervised graph neural networks. Experiments on a dataset of 52 Urdu novels written by seven authors show that learned graph representations substantially outperform hand-crafted and unsupervised baselines, achieving up to 0.857 accuracy under a strict author-aware evaluation protocol.

---

## 20. Torch Geometric Pool: the Pytorch library for pooling in Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.12642v1](http://arxiv.org/abs/2512.12642v1)

**作者:** Filippo Maria Bianchi, Carlo Abate, Ivan Marisca

**发布时间:** 2025-12-14

### GPT解析

### 总结

Torch Geometric Pool (tgp) 是一个用于图神经网络分层池化的库，基于Pytorch Geometric构建，提供多种池化算子，具有一致API和模块化设计。

### 背景

图神经网络需要分层池化操作，但目前可能缺乏统一且高效的解决方案。

### 目的

介绍Torch Geometric Pool库，展示其结构并进行全面基准测试，比较不同下游任务中实现的图池化方法的性能。

### 方法

构建了一个基于Pytorch Geometric的库，提供多种池化算子，采用一致API和模块化设计，包括预计算池化功能以加速训练。展示了库的结构并进行了广泛的基准测试。

### 主要发现

最优池化算子的选择取决于具体的任务和数据，支持需要能够快速原型的库。

### 结论

Torch Geometric Pool库提供了多样化和高效的池化操作，能够支持不同任务下的快速原型开发。

### 翻译

我们介绍了Torch Geometric Pool (tgp)，这是一个用于图神经网络分层池化的库。基于Pytorch Geometric构建，Torch Geometric Pool (tgp) 提供了多种池化算子，在一致的API和模块化设计下统一。该库强调可用性和可扩展性，包括预计算池化等功能，这些功能显著加速了一类算子的训练。在本文中，我们展示了tgp的结构并进行了广泛的基准测试。后者展示了库的功能，并系统比较了不同下游任务中实现的图池化方法的性能。结果表明，最优池化算子的选择取决于任务和手头的数据，这支持了需要能够快速原型的库的必要性。


### 论文摘要

We introduce Torch Geometric Pool (tgp), a library for hierarchical pooling in Graph Neural Networks. Built upon Pytorch Geometric, Torch Geometric Pool (tgp) provides a wide variety of pooling operators, unified under a consistent API and a modular design. The library emphasizes usability and extensibility, and includes features like precomputed pooling, which significantly accelerate training for a class of operators. In this paper, we present tgp's structure and present an extensive benchmark. The latter showcases the library's features and systematically compares the performance of the implemented graph-pooling methods in different downstream tasks. The results, showing that the choice of the optimal pooling operator depends on tasks and data at hand, support the need for a library that enables fast prototyping.

---

## 21. Empirical Mode Decomposition and Graph Transformation of the MSCI World Index: A Multiscale Topological Analysis for Graph Neural Network Modeling

**论文链接:** [http://arxiv.org/abs/2512.12526v1](http://arxiv.org/abs/2512.12526v1)

**作者:** Agustín M. de los Riscos, Julio E. Sandubete, Diego Carmona-Fernández, León Beleña

**发布时间:** 2025-12-14

**备注:** 19 pages, 3 figures, 6 tables

### GPT解析

### 总结

该研究将经验模态分解应用于MSCI世界指数，将得到的本征模态函数转换为图表示，以便使用图神经网络进行建模。研究通过拓扑分析发现了不同频率分量的图结构特性，为金融时间序列预测提供了指导。

### 背景

金融时间序列分析需要有效的方法来捕捉不同时间尺度的特征，图神经网络为时间序列建模提供了新的可能性。

### 目的

将经验模态分解得到的本征模态函数转换为图表示，以便使用图神经网络进行建模，并分析不同频率分量的图结构特性。

### 方法

使用CEEMDAN提取9个IMFs，涵盖从高频波动到长期趋势的各个时间尺度。每个IMF通过四种时间序列到图的方法转换为图：自然可见性图、水平可见性图、递归图和转移图。

### 主要发现

拓扑分析显示出明显的尺度依赖结构：高频IMFs产生密集的高度连接的小世界图，而低频IMFs产生具有更长特征路径长度的稀疏网络。基于可见性的方法对幅度变化更敏感，通常产生更高的聚类，而递归图更好地保留了时间依赖性。

### 结论

这些结果为设计针对分解分量的结构特性的GNN架构提供了指导，支持金融时间序列更有效的预测建模。

### 翻译

本研究将经验模态分解应用于MSCI世界指数，并将得到的本征模态函数转换为图表示，以便使用图神经网络进行建模。使用CEEMDAN，我们提取了九个本征模态函数，涵盖了从高频波动到长期趋势的各个时间尺度。每个本征模态函数通过四种时间序列到图的方法转换为图：自然可见性图、水平可见性图、递归图和转移图。拓扑分析显示出明显的尺度依赖结构：高频本征模态函数产生密集的高度连接的小世界图，而低频本征模态函数产生具有更长特征路径长度的稀疏网络。基于可见性的方法对幅度变化更敏感，通常产生更高的聚类，而递归图更好地保留了时间依赖性。这些结果为设计针对分解分量的结构特性的图神经网络架构提供了指导，支持金融时间序列更有效的预测建模。


### 论文摘要

This study applies Empirical Mode Decomposition (EMD) to the MSCI World index and converts the resulting intrinsic mode functions (IMFs) into graph representations to enable modeling with graph neural networks (GNNs). Using CEEMDAN, we extract nine IMFs spanning high-frequency fluctuations to long-term trends. Each IMF is transformed into a graph using four time-series-to-graph methods: natural visibility, horizontal visibility, recurrence, and transition graphs. Topological analysis shows clear scale-dependent structure: high-frequency IMFs yield dense, highly connected small-world graphs, whereas low-frequency IMFs produce sparser networks with longer characteristic path lengths. Visibility-based methods are more sensitive to amplitude variability and typically generate higher clustering, while recurrence graphs better preserve temporal dependencies. These results provide guidance for designing GNN architectures tailored to the structural properties of decomposed components, supporting more effective predictive modeling of financial time series.

---

## 22. GoMS: Graph of Molecule Substructure Network for Molecule Property Prediction

**论文链接:** [http://arxiv.org/abs/2512.12489v1](http://arxiv.org/abs/2512.12489v1)

**作者:** Shuhui Qu, Cheolwoo Park

**发布时间:** 2025-12-13

### GPT解析

### 总结

本文介绍了一种名为GoMS的新型图神经网络架构，用于分子性质预测，通过建模分子子结构之间的相互作用和空间排列，优于现有的ESAN等方法。

### 背景

现有的图神经网络方法如ESAN将分子视为独立子结构的集合，忽略了这些组件之间的重要关系。

### 目的

开发一种能够明确建模分子子结构之间相互作用和空间排列的新型架构，以提高分子性质预测的准确性。

### 方法

提出Graph of Molecule Substructures (GoMS)，构建一个图结构，其中节点代表子图，边捕获它们的结构关系，保留关于子结构如何在分子中连接和重叠的关键拓扑信息。

### 主要发现

GoMS在公共分子数据集上优于ESAN和其他基线方法，特别是对于大分子(超过100个原子)；随着分子尺寸增加，性能差距扩大；理论分析表明GoMS可以区分具有相同子图组成但不同空间排列的分子。

### 结论

GoMS通过捕获基于集合的方法中丢失的子结构关系，代表了面向真实世界应用的可扩展和可解释分子性质预测的重要进展。

### 翻译

尽管图神经网络在分子性质预测中显示出显著的成功，但当前的方法如等变子结构聚合网络(ESAN)将分子视为独立子结构的集合，忽略了这些组件之间的关键关系。我们提出了分子子结构图(GoMS)，一种新型架构，明确建模分子子结构之间的相互作用和空间排列。与ESAN的基于集合的表示不同，GoMS构建一个图，其中节点代表子图，边捕获它们的结构关系，保留了关于子结构如何在分子中连接和重叠的关键拓扑信息。通过在公共分子数据集上的广泛实验，我们证明GoMS优于ESAN和其他基线方法，特别是对于包含超过100个原子的大分子有显著改进。随着分子尺寸的增加，性能差距扩大，展示了GoMS在模拟工业规模分子方面的有效性。我们的理论分析表明，GoMS可以区分具有相同子图组成但不同空间排列的分子。我们的方法在涉及复杂分子的材料科学应用中显示出特别的前景，这些分子的性质来自多个功能单元之间的相互作用。通过捕获基于集合的方法中丢失的子结构关系，GoMS代表了面向真实世界应用的可扩展和可解释分子性质预测的重要进展。


### 论文摘要

While graph neural networks have shown remarkable success in molecular property prediction, current approaches like the Equivariant Subgraph Aggregation Networks (ESAN) treat molecules as bags of independent substructures, overlooking crucial relationships between these components. We present Graph of Molecule Substructures (GoMS), a novel architecture that explicitly models the interactions and spatial arrangements between molecular substructures. Unlike ESAN's bag-based representation, GoMS constructs a graph where nodes represent subgraphs and edges capture their structural relationships, preserving critical topological information about how substructures are connected and overlap within the molecule. Through extensive experiments on public molecular datasets, we demonstrate that GoMS outperforms ESAN and other baseline methods, with particularly improvements for large molecules containing more than 100 atoms. The performance gap widens as molecular size increases, demonstrating GoMS's effectiveness for modeling industrial-scale molecules. Our theoretical analysis demonstrates that GoMS can distinguish molecules with identical subgraph compositions but different spatial arrangements. Our approach shows particular promise for materials science applications involving complex molecules where properties emerge from the interplay between multiple functional units. By capturing substructure relationships that are lost in bag-based approaches, GoMS represents a significant advance toward scalable and interpretable molecular property prediction for real-world applications.

---

## 23. Can Graphs Improve Tabular Foundation Models?

**论文链接:** [http://arxiv.org/abs/2512.12405v1](http://arxiv.org/abs/2512.12405v1)

**作者:** Franck Le, Keith Grueneberg, Erich Nahum, Vadim Sheinin

**发布时间:** 2025-12-13

### GPT解析

### 总结

研究引入了BOLERO，一种结合简单图先验的轻量级方法，用于增强预训练的表格转换器。通过在大量数据集上的评估，证明了这种方法的有效性，并在分类和回归任务上都取得了最佳性能。

### 背景

表格数据是许多现实世界系统的核心。虽然最近的表格转换器和上下文学习器包含有限的行间推理能力，但大多数方法仍然缺乏对实例之间关系的明确建模机制，尽管相似样本通常共享相关结果。

### 目的

研究引入简单的图先验是否能增强预训练的表格转换器。

### 方法

提出了BOLERO，这是一个轻量级的静态二分图头，它增强了RoBERTa-Tab（一种使用掩码令牌预测进行预训练的RoBERTa风格表格主干）。每个实例连接到特征/值锚点；一个小型图神经网络优化行表示，同时主干保持冻结。在80个分类和64个回归数据集上进行了评估，并与多种强基线方法进行比较。

### 主要发现

BOLERO在分类和回归中都取得了最多的统计显著胜利，表明轻量级图先验显著改善了预训练的表格转换器。研究使用了Wilcoxon符号秩检验和效应大小分析来确保统计上可靠的结论。

### 结论

轻量级图先验可以有效地增强预训练的表格转换器，提高其在表格数据任务上的性能。

### 翻译

表格数据是许多现实世界系统的核心。虽然最近的表格转换器和上下文学习器（如SAINT、TP-BERTa、TabPFN、TabICL和MITRA）包含有限的行间推理能力，但大多数方法仍然缺乏对实例之间关系的明确建模机制，尽管相似样本通常共享相关结果。我们研究了引入简单图先验是否能增强预训练的表格转换器。具体来说，我们引入了BOLERO，这是一个轻量级的静态二分图头，它增强了RoBERTa-Tab（一种使用掩码令牌预测进行预训练的RoBERTa风格表格主干）。每个实例连接到特征/值锚点；一个小型图神经网络优化行表示，同时主干保持冻结。我们在TP-BERTa基准套件的80个分类和64个回归数据集上进行了评估，并与包括XGBoost、CatBoost、TabPFN-v2、MITRA、TabICL、TP-BERTa和RoBERTa-Tab在内的强基线进行比较。为确保统计上可靠的结论，我们遵循多数据集评估的最佳实践：使用每对数据集分数差异的Wilcoxon符号秩检验和效应大小（中位数改进及置信区间），而不是依赖于竞争者池的均值秩事后检验。BOLERO在分类和回归中都取得了最多的统计显著胜利，证明了轻量级图先验显著改善了预训练的表格转换器。


### 论文摘要

Tabular data are central to many real-world systems. While recent tabular transformers and in-context learners such as SAINT, TP-BERTa, TabPFN, TabICL, and MITRA incorporate limited inter-row reasoning, most approaches still lack an explicit mechanism to model relationships among instances, even though similar samples often share related outcomes. We investigate whether introducing \emph{simple graph priors} can enhance \emph{pretrained tabular transformers}. Concretely, we introduce {BOLERO}, a lightweight, static bipartite graph head that augments {RoBERTa-Tab} (a RoBERTa-style tabular backbone pretrained with masked-token prediction.) Each instance connects to feature/value anchors; a small GNN refines row representations, while the backbone remains frozen. We evaluate on 80 classification and 64 regression datasets from the TP-BERTa benchmark suites, comparing against strong baselines including XGBoost, CatBoost, TabPFN-v2, MITRA, TabICL, TP-BERTa, and RoBERTa-Tab. To ensure statistically sound conclusions, we follow best practices for multi-dataset evaluation: pairwise Wilcoxon signed-rank tests on per-dataset score differences and effect sizes (median improvement with confidence intervals), rather than mean-rank post-hoc tests that depend on the competitor pool. BOLERO achieves the highest number of statistically significant wins across both classification and regression, demonstrating that lightweight graph priors meaningfully improve pretrained tabular transformers.

---

## 24. High-Dimensional Tensor Discriminant Analysis: Low-Rank Discriminant Structure, Representation Synergy, and Theoretical Guarantees

**论文链接:** [http://arxiv.org/abs/2512.12122v1](http://arxiv.org/abs/2512.12122v1)

**作者:** Elynn Chen, Yuefeng Han, Jiayu Li

**发布时间:** 2025-12-13

### GPT解析

### 总结

本文提出了一种高维CP低秩张量判别分析方法（CP-TDA），针对高维张量值预测器的分类问题。该方法结合随机复合PCA初始化和迭代优化算法，在张量高斯混合模型下实现了全局收敛和最优误分类率。同时开发了半参数张量判别模型处理非正态张量数据，实验表明该方法在高维小样本情况下显著优于现有方法。

### 背景

高维张量值预测器在现代应用中日益普遍，特别是作为神经网络的表示学习结果。然而，现有的张量分类方法依赖于稀疏性或Tucker结构，通常缺乏理论保证。

### 目的

提出一种基于CP低秩结构的判别张量建模方法，解决现有张量分类方法的局限性，并提供理论保证，特别是在处理相关性和各向异性噪声、弱信号强度和不相干条件下。

### 方法

1) 在张量高斯混合模型下提出CP-TDA；2) 使用随机复合PCA(rc-PCA)初始化处理相关性和各向异性噪声；3) 开发迭代优化算法；4) 建立全局收敛和最优误分类率理论；5) 开发半参数张量判别模型；6) 通过生成模型将张量表示映射到适合CP-TDA的潜在空间；7) 将误分类风险分解为表示、近似和估计误差。

### 主要发现

1) CP低秩结构适用于判别张量建模；2) rc-PCA初始化对处理相关性和各向异性噪声至关重要；3) 在较弱信号强度下仍能有效工作；4) 建立了全局收敛和最优误分类率理论；5) 半参数模型能有效处理非正态张量数据；6) 在图分类任务中显著优于现有方法，特别是在高维小样本情况下。

### 结论

CP-TDA方法为高维张量分类提供了有效解决方案，结合CP低秩结构和优化算法不仅提供了理论保证，还在实验中表现出优越性能，特别是在处理神经网络学习得到的表示和高维小样本情况下。

### 翻译

高维张量值预测器在现代应用中日益普遍，特别是作为从神经网络学习得到的表示。现有的张量分类方法依赖于稀疏性或Tucker结构，通常缺乏理论保证。受判别信号集中在少数多线性分量上的经验证据启发，我们引入了判别张量的CP低秩结构，这是一种尚未被探索的建模视角。在张量高斯混合模型下，我们提出了具有随机复合PCA（rc-PCA）初始化的高维CP低秩张量判别分析（CP-TDA），这对于处理较弱信号强度和不相干条件下的相关性和各向异性噪声至关重要，随后是迭代优化算法。我们建立了全局收敛和最小最大最优误分类率。为了处理偏离张量正态性的张量数据，我们开发了首个半参数张量判别模型，其中学习到的张量表示通过深度生成模型映射到为CP-TDA量身定制的潜在空间。误分类风险分解为表示误差、近似误差和估计误差。在图分类上的数值研究和真实数据分析表明，该方法比现有的张量分类器和最先进的图神经网络有显著提升，特别是在高维小样本情况下。


### 论文摘要

High-dimensional tensor-valued predictors arise in modern applications, increasingly as learned representations from neural networks. Existing tensor classification methods rely on sparsity or Tucker structures and often lack theoretical guarantees. Motivated by empirical evidence that discriminative signals concentrate along a few multilinear components, we introduce CP low-rank structure for the discriminant tensor, a modeling perspective not previously explored. Under a Tensor Gaussian Mixture Model, we propose high-dimensional CP low-rank Tensor Discriminant Analysis (CP-TDA) with Randomized Composite PCA (\textsc{rc-PCA}) initialization, that is essential for handling dependent and anisotropic noise under weaker signal strength and incoherence conditions, followed by iterative refinement algorithm. We establish global convergence and minimax-optimal misclassification rates.   To handle tensor data deviating from tensor normality, we develop the first semiparametric tensor discriminant model, in which learned tensor representations are mapped via deep generative models into a latent space tailored for CP-TDA. Misclassification risk decomposes into representation, approximation, and estimation errors. Numerical studies and real data analysis on graph classification demonstrate substantial gains over existing tensor classifiers and state-of-the-art graph neural networks, particularly in high-dimensional, small-sample regimes.

---

## 25. Scalable IP Mimicry: End-to-End Deceptive IP Blending to Overcome Rectification and Scale Limitations of IP Camouflage

**论文链接:** [http://arxiv.org/abs/2512.12061v1](http://arxiv.org/abs/2512.12061v1)

**作者:** Junling Fan, George Rushevich, Giorgio Rusconi, Mengdi Zhu, Reiner Dizon-Paradis, Domenic Forte

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出两种新型端到端模型，解决现有IP伪装技术的局限性，包括高开销后处理、不易扩展以及逻辑表示与RE分析不匹配的问题。

### 背景

半导体知识产权盗窃造成巨大经济损失，估计每年损失2250亿至6000亿美元，尽管有CHIPS法案等举措，许多半导体设计仍然容易受到逆向工程攻击。

### 目的

解决现有IP Camouflage技术的高开销、不易扩展以及逻辑表示与RE分析不匹配的问题。

### 方法

提出两种新颖的端到端模型：图匹配算法解决表示问题，基于DNAS的NAND阵列模型实现可扩展性，并引入拟态感知分区方法实现大规模设计的分而治之。

### 主要发现

所提出的模型能够抵抗SAT和GNN-RE攻击，为端到端欺骗性IP设计提供了高效且可扩展的路径。

### 结论

这些新型模型克服了现有IP伪装技术的局限性，为半导体知识产权保护提供了更有效的解决方案。

### 翻译

半导体知识产权盗窃造成估计每年2250亿至6000亿美元的损失。尽管有CHIPS法案等举措，许多半导体设计仍然容易受到逆向工程攻击。IP Camouflage是一种突破性技术，通过'拟态欺骗'使整个模块伪装成不同的IP，超越了传统伪装的逻辑门隐藏方法。然而，它面临关键局限性：需要高开销的后生成校正步骤，不易扩展，且使用的AIG逻辑表示与标准RE分析流程不匹配。本文通过引入两种新颖的端到端模型解决了这些缺点。我们提出了图匹配算法来解决表示问题，以及基于DNAS的NAND阵列模型来实现可扩展性。为此，我们还引入了拟态感知分区方法，使大规模设计能够采用分而治之的方法。我们的结果表明，这些模型能够抵抗SAT和GNN-RE攻击，为端到端欺骗性IP设计提供了高效且可扩展的路径。


### 论文摘要

Semiconductor intellectual property (IP) theft incurs estimated annual losses ranging from $225 billion to $600 billion. Despite initiatives like the CHIPS Act, many semiconductor designs remain vulnerable to reverse engineering (RE). IP Camouflage is a recent breakthrough that expands beyond the logic gate hiding of traditional camouflage through "mimetic deception," where an entire module masquerades as a different IP. However, it faces key limitations: requires a high-overhead post-generation rectification step, is not easily scalable, and uses an AIG logic representation that is mismatched with standard RE analysis flows. This paper addresses these shortcommings by introducing two novel, end-to-end models. We propose a Graph-Matching algorithm to solve the representation problem and a DNAS-based NAND Array model to achieve scalability. To facilitate this, we also introduce a mimicry-aware partitioning method, enabling a divide-and-conquer approach for large-scale designs. Our results demonstrate that these models are resilient to SAT and GNN-RE attacks, providing efficient and scalable paths for end-to-end deceptive IP design.

---

## 26. Context-Aware Agentic Power Resources Optimisation in EV using Smart2ChargeApp

**论文链接:** [http://arxiv.org/abs/2512.12048v1](http://arxiv.org/abs/2512.12048v1)

**作者:** Muddsair Sharif, Huseyin Seker

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出了CAMAC-DRA框架，通过Smart2Charge应用优化智能电动汽车充电生态系统，协调250辆电动汽车和45个充电站网络中的自主充电智能体，通过上下文感知决策适应动态环境条件。

### 背景

随着电动汽车普及，充电基础设施管理和优化成为重要问题，需要能够协调多方利益并适应动态环境变化的解决方案。

### 目的

开发一个能够平衡电动汽车用户、电网运营商、充电站运营商、车队运营商和环境因素等多方利益的智能充电系统，提高能源效率、降低成本并减少电网压力。

### 方法

采用多智能体方法，结合协调的深度Q网络、图神经网络和注意力机制，处理20个上下文特征，通过加权协调机制和共识协议平衡各方需求。

### 主要发现

在441,077笔充电交易的真实数据集上验证，相比基线算法表现更优，达到92%协调成功率，15%能源效率提升，10%成本降低，20%电网压力减少，2.3倍更快收敛速度，88%训练稳定性和85%样本效率，商业可行性验证显示净现值为-122,962美元，通过可再生能源整合实现69%成本降低。

### 结论

CAMAC-DRA框架成功实现了上下文感知的多利益相关者协调，平衡竞争性目标并适应实时变量，是智能电动汽车协调充电和可持续交通电气化的突破性解决方案。

### 翻译

本文通过Smart2Charge应用提出了一个新颖的上下文敏感多智能体协调动态资源分配框架，用于优化智能电动汽车充电生态系统。该系统协调250辆电动汽车和45个充电站网络中的自主充电智能体，并通过上下文感知决策适应动态环境条件。我们的多智能体方法采用协调的深度Q网络与图神经网络和注意力机制相结合，处理包括天气模式、交通状况、电网负载波动和电价在内的20个上下文特征。该框架通过加权协调机制和共识协议平衡五个生态系统利益相关者的需求。使用包含441,077笔充电交易的真实世界数据集进行的全面验证表明，与基线算法相比性能更优。CAMAC-DRA框架实现92%的协调成功率，15%的能源效率提升，10%的成本降低，20%的电网压力减少，以及2.3倍的更快收敛速度，同时保持88%的训练稳定性和85%的样本效率。真实世界验证证实了其商业可行性，净现值为-122,962美元，通过可再生能源整合实现69%的成本降低。


### 论文摘要

This paper presents a novel context-sensitive multi\-agent coordination for dynamic resource allocation (CAMAC-DRA) framework for optimizing smart electric vehicle (EV) charging ecosystems through the Smart2Charge application. The proposed system coordinates autonomous charging agents across networks of 250 EVs and 45 charging stations while adapting to dynamic environmental conditions through context-aware decision-making. Our multi-agent approach employs coordinated Deep Q\-Networks integrated with Graph Neural Networks and attention mechanisms, processing 20 contextual features including weather patterns, traffic conditions, grid load fluctuations, and electricity pricing.The framework balances five ecosystem stakeholders i.e. EV users (25\%), grid operators (20\%), charging station operators (20\%), fleet operators (20%), and environmental factors (15\%) through weighted coordination mechanisms and consensus protocols. Comprehensive validation using real-world datasets containing 441,077 charging transactions demonstrates superior performance compared to baseline algorithms including DDPG, A3C, PPO, and GNN approaches. The CAMAC\-DRA framework achieves 92\% coordination success rate, 15\% energy efficiency improvement, 10\% cost reduction, 20% grid strain decrease, and \2.3x faster convergence while maintaining 88\% training stability and 85\% sample efficiency. Real-world validation confirms commercial viability with Net Present Cost of -\$122,962 and 69\% cost reduction through renewable energy integration. The framework's unique contribution lies in developing context-aware multi-stakeholder coordination that successfully balances competing objectives while adapting to real-time variables, positioning it as a breakthrough solution for intelligent EV charging coordination and sustainable transportation electrification.

---

## 27. Accelerating Sparse Matrix-Matrix Multiplication on GPUs with Processing Near HBMs

**论文链接:** [http://arxiv.org/abs/2512.12036v1](http://arxiv.org/abs/2512.12036v1)

**作者:** Shiju Li, Younghoon Min, Hane Yie, Hoshik Kim, Soohong Ahn, Joonseop Sim, Chul-Ho Lee, Jongryool Kim

**发布时间:** 2025-12-12

**备注:** 13 pages, 11 figures

### GPT解析

### 总结

该论文提出了一种基于哈希的多阶段稀疏矩阵乘法(HM-SpGEMM)和间接内存访问加速技术(AIA)，通过硬件-软件协同设计优化GPU高带宽内存上的稀疏矩阵乘法操作，显著提升了性能。

### 背景

稀疏矩阵乘法(SpGEMM)是众多科学计算和数据分析应用中的基本操作，常因不规则内存访问模式而成为性能瓶颈。

### 目的

开发一种创新的近内存处理方法，优化GPU HBM上的SpGEMM操作，解决内存访问效率问题。

### 方法

提出Hash based Multi-phase SpGEMM on GPU和AIA技术，构建硬件-软件协同设计框架，并在多种图工作负载上进行评估，包括图收缩、马尔可夫聚类和图神经网络。

### 主要发现

对于图分析应用，AIA比纯软件实现减少17.3%时间；与cuSPARSE相比，图收缩减少76.5%时间，马尔可夫聚类减少58.4%时间；对于GNN训练，混合方法平均加速1.43倍，比cuSPARSE加速1.95倍，在大规模数据集上最高达4.18倍。

### 结论

该框架在处理复杂、特定应用的工作负载时，比现有方法展现显著性能改进，具有实际应用价值。

### 翻译

稀疏矩阵乘法(SpGEMM)是众多科学计算和数据分析应用中的基本操作，常因不规则内存访问模式而成为瓶颈。本文提出基于哈希的多阶段GPU稀疏矩阵乘法和间接内存访问加速技术(AIA)，这是一种创新的近内存处理方法，用于优化GPU高带宽内存上的SpGEMM。我们的SpGEMM硬件-软件协同设计框架在处理复杂、特定应用的工作负载时，比最先进方法展现出显著的性能改进。我们在多种图工作负载上评估了该方法，包括图收缩、马尔可夫聚类和图神经网络(GNNs)，展示了其实际应用价值。对于图分析应用，AIA比纯软件实现减少17.3%的时间，与cuSPARSE相比，图收缩减少76.5%时间，马尔可夫聚类减少58.4%时间。对于具有结构化全局剪枝的GNN训练应用，我们的混合方法在六个基准数据集和三种架构(GCN、GIN、GraphSAGE)上平均比纯软件实现加速1.43倍，与cuSPARSE相比GNN工作负载加速1.95倍，在大规模数据集上最高可达4.18倍。


### 论文摘要

Sparse General Matrix-Matrix Multiplication (SpGEMM) is a fundamental operation in numerous scientific computing and data analytics applications, often bottlenecked by irregular memory access patterns. This paper presents Hash based Multi-phase SpGEMM on GPU and the Acceleration of Indirect Memory Access (AIA) technique, a novel custom near-memory processing approach to optimizing SpGEMM on GPU HBM. Our hardware-software co-designed framework for SpGEMM demonstrates significant performance improvements over state-of-the-art methods, particularly in handling complex, application-specific workloads. We evaluate our approach on various graph workloads, including graph contraction, Markov clustering, and Graph Neural Networks (GNNs), showcasing its practical applicability. For graph analytics applications, AIA demonstrates up to 17.3% time reduction from the software-only implementation, while achieving time reduction of 76.5% for Graph Contraction and 58.4% for Markov Clustering compared to cuSPARSE. For GNN training applications with structured global pruning, our hybrid approach delivers an average of 1.43x speedup over software-only implementation across six benchmark datasets and three architectures (GCN, GIN, GraphSAGE), and shows 1.95x speedup for GNN workloads when compared to cuSPARSE, with up to 4.18x gains on large-scale datasets.

---

## 28. 论文ID: 2512.11936v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.11936v1.json'

---

## 29. AGAPI-Agents: An Open-Access Agentic AI Platform for Accelerated Materials Design on AtomGPT.org

**论文链接:** [http://arxiv.org/abs/2512.11935v1](http://arxiv.org/abs/2512.11935v1)

**作者:** Jaehyung Lee, Justin Ely, Kent Zhang, Akshaya Ajith, Charles Rhys Campbell, Kamal Choudhary

**发布时间:** 2025-12-12

### GPT解析

### 总结

本研究介绍了AGAPI（AtomGPT.org API），一个开放访问的智能AI平台，整合了多个开源大型语言模型和材料科学API端点，通过统一的编排框架连接数据库、模拟工具和机器学习模型，为材料研究提供可扩展、透明的基础，促进可复现的AI加速材料发现。

### 背景

人工智能正在重塑科学发现，但其在材料研究中的应用受到碎片化计算生态系统、可复现性挑战以及对商业大型语言模型依赖的限制。

### 目的

开发一个开放访问的智能AI平台，整合多种开源LLMs和材料科学API端点，统一数据库、模拟工具和机器学习模型，通过自主构建和执行多步骤工作流程，实现AI加速的材料发现。

### 方法

AGAPI采用Agent-Planner-Executor-Summarizer架构，自主构建和执行跨越材料数据检索、图神经网络属性预测、机器学习力场优化、紧束缚计算、衍射分析和逆向设计的多步骤工作流程。通过端到端工作流程（包括异质结构构建、粉末X射线衍射分析和半导体缺陷工程）进行演示，并使用30多个示例提示作为测试案例评估AGAPI，将智能代理预测与实验数据进行比较。

### 主要发现

AGAPI能够执行多达十个连续操作的工作流程，包括异质结构构建、粉末X射线衍射分析和半导体缺陷工程。评估表明，AGAPI提供了可扩展和透明的基础，用于可复现、AI加速的材料发现。

### 结论

AGAPI作为一个拥有1,000多名活跃用户的平台，为材料研究提供了有效的AI解决方案，解决了计算生态系统碎片化、可复现性挑战和对商业模型依赖的问题。

### 翻译

人工智能正在重塑科学发现，但其在材料研究中的应用受到碎片化计算生态系统、可复现性挑战以及对商业大型语言模型依赖的限制。在此，我们介绍了AGAPI（AtomGPT.org API），一个开放访问的智能AI平台，整合了八个以上的开源大型语言模型和二十多个材料科学API端点，通过统一的编排框架连接数据库、模拟工具和机器学习模型。AGAPI采用Agent-Planner-Executor-Summarizer架构，自主构建和执行跨越材料数据检索、图神经网络属性预测、机器学习力场优化、紧束缚计算、衍射分析和逆向设计的多步骤工作流程。我们通过端到端工作流程展示了AGAPI的能力，包括异质结构构建、粉末X射线衍射分析和需要多达十个连续操作的半导体缺陷工程。此外，我们使用三十多个示例提示作为测试案例评估了AGAPI，并将智能代理预测与实验数据进行了比较。拥有1,000多名活跃用户的AGAPI，为可复现、AI加速的材料发现提供了可扩展和透明的基础。AGAPI-Agents代码库可在https://github.com/atomgptlab/agapi获取。


### 论文摘要

Artificial intelligence is reshaping scientific discovery, yet its use in materials research remains limited by fragmented computational ecosystems, reproducibility challenges, and dependence on commercial large language models (LLMs). Here we introduce AGAPI (AtomGPT.org API), an open-access agentic AI platform that integrates more than eight open-source LLMs with over twenty materials-science API endpoints, unifying databases, simulation tools, and machine-learning models through a common orchestration framework. AGAPI employs an Agent-Planner-Executor-Summarizer architecture that autonomously constructs and executes multi-step workflows spanning materials data retrieval, graph neural network property prediction, machine-learning force-field optimization, tight-binding calculations, diffraction analysis, and inverse design. We demonstrate AGAPI through end-to-end workflows, including heterostructure construction, powder X-ray diffraction analysis, and semiconductor defect engineering requiring up to ten sequential operations. In addition, we evaluate AGAPI using 30+ example prompts as test cases and compare agentic predictions with and without tool access against experimental data. With more than 1,000 active users, AGAPI provides a scalable and transparent foundation for reproducible, AI-accelerated materials discovery. AGAPI-Agents codebase is available at https://github.com/atomgptlab/agapi.

---

## 30. Enhancing Urban Visual Place Recognition for Crowdsourced Flood Imagery via LLM-Guided Attention

**论文链接:** [http://arxiv.org/abs/2512.11811v1](http://arxiv.org/abs/2512.11811v1)

**作者:** Fengyi Xu, Jun Ma, Waishan Qiu, Cui Guo

**发布时间:** 2025-11-25

### GPT解析

### 总结

VPR-AttLLM是一个模型无关的框架，整合大型语言模型的语义推理和地理空间知识到视觉位置识别流程中，通过注意力引导的描述符增强提高检索性能，无需重新训练模型或额外数据。

### 背景

社交媒体众包的街景图像为城市洪水等危机事件提供实时视觉证据，但缺乏可靠地理元数据；现有VPR模型在跨源场景中因视觉失真和域偏移导致性能显著下降。

### 目的

提出VPR-AttLLM框架，利用LLMs识别位置信息区域并抑制视觉噪声，提高VPR检索性能。

### 方法

利用LLMs识别城市背景下具有位置信息的区域并抑制瞬时视觉噪声；在SF-XL、合成洪水场景、Mapillary照片和HK-URBAN数据集上评估；将VPR-AttLLM与CosPlace、EigenPlaces和SALAD三种VPR模型集成。

### 主要发现

VPR-AttLL一致提高召回性能，相对增益通常为1-3%，在最具挑战性的真实洪水图像上达到8%；建立了LLM引导的多模态融合范式；桥接了类人空间推理与现代VPR架构。

### 结论

VPR-AttLLM的即插即用设计、跨源鲁棒性和可解释性使其在可扩展城市监测和众包危机图像快速地理定位方面具有潜力。

### 翻译

社交媒体众包的街景图像为城市洪水和其他危机事件提供了宝贵的实时视觉证据，但它通常缺乏可靠的地理元数据用于应急响应。现有的视觉位置识别模型在应用于此类图像时表现出显著的性能下降，这是由于跨源场景中固有的视觉失真和域偏移。本文提出了VPR-AttLLM，一个模型无关的框架，通过注意力引导的描述符增强，将大型语言模型的语义推理和地理空间知识整合到现有的VPR流程中。通过利用LLMs识别城市背景中具有位置信息的区域并抑制瞬时视觉噪声，VPR-AttLLM在不要求模型重新训练或额外数据的情况下提高了检索性能。在包含真实社交媒体洪水图像的SF-XL、在已建立的查询集和Mapillary照片上的合成洪水场景以及捕捉形态各异城市景观的新HK-URBAN数据集上进行了全面评估。将VPR-AttLLM与三种最先进的VPR模型-CosPlace、EigenPlaces和SALAD-集成，一致提高了召回性能，相对增益通常在1-3%之间，在最具挑战性的真实洪水图像上达到8%。除了在检索准确性方面可衡量的提升外，这项研究为视觉检索系统中LLM引导的多模态融合建立了可推广的范式。通过将城市感知理论的原理嵌入到注意力机制中，VPR-AttLLM桥接了类人的空间推理与现代VPR架构。它的即插即用设计、强大的跨源鲁棒性和可解释性突显了其在可扩展城市监测和众包危机图像快速地理定位方面的潜力。


### 论文摘要

Crowdsourced street-view imagery from social media provides valuable real-time visual evidence of urban flooding and other crisis events, yet it often lacks reliable geographic metadata for emergency response. Existing Visual Place Recognition (VPR) models exhibit substantial performance degradation when applied to such imagery due to visual distortions and domain shifts inherent in cross-source scenarios. This paper presents VPR-AttLLM, a model-agnostic framework that integrates the semantic reasoning and geospatial knowledge of Large Language Models (LLMs) into established VPR pipelines through attention-guided descriptor enhancement. By leveraging LLMs to identify location-informative regions within the city context and suppress transient visual noise, VPR-AttLLM improves retrieval performance without requiring model retraining or additional data. Comprehensive evaluations are conducted on extended benchmarks including SF-XL enriched with real social-media flood images, synthetic flooding scenarios over established query sets and Mapillary photos, and a new HK-URBAN dataset capturing morphologically distinct cityscapes. Integrating VPR-AttLLM with three state-of-the-art VPR models-CosPlace, EigenPlaces, and SALAD-consistently improves recall performance, yielding relative gains typically between 1-3% and reaching up to 8% on the most challenging real flood imagery. Beyond measurable gains in retrieval accuracy, this study establishes a generalizable paradigm for LLM-guided multimodal fusion in visual retrieval systems. By embedding principles from urban perception theory into attention mechanisms, VPR-AttLLM bridges human-like spatial reasoning with modern VPR architectures. Its plug-and-play design, strong cross-source robustness, and interpretability highlight its potential for scalable urban monitoring and rapid geo-localization of crowdsourced crisis imagery.

---

## 31. Recurrent Video Masked Autoencoders

**论文链接:** [http://arxiv.org/abs/2512.13684v1](http://arxiv.org/abs/2512.13684v1)

**作者:** Daniel Zoran, Nikhil Parthasarathy, Yi Yang, Drew A Hudson, Joao Carreira, Andrew Zisserman

**发布时间:** 2025-12-15

### GPT解析

### 总结

提出循环视频掩码自编码器（RVM），一种新型视频表示学习方法

### 背景

现有视频表示学习方法在效率和时空结构捕捉方面存在局限性

### 目的

开发一种能够有效捕捉视频时空结构的高效视频表示学习方法

### 方法

使用基于Transformer的循环神经网络聚合密集图像特征，通过非对称掩码预测任务和标准像素重建目标进行学习

### 主要发现

RVM在视频级任务上与最先进视频模型性能相当，在几何和空间理解任务上优于图像模型，小模型情况下表现出色且无需知识蒸馏，参数效率提高30倍，循环特性允许长时间稳定特征传播

### 结论

RVM能够学习场景语义、结构和运动的丰富表示

### 翻译

我们提出循环视频掩码自编码器（RVM）：一种新型视频表示学习方法，使用基于Transformer的循环神经网络聚合密集图像特征，有效捕捉自然视频数据的时空结构。RVM通过非对称掩码预测任务学习，仅需标准像素重建目标。这种设计产生了一个高效的通用编码器：RVM在视频级任务（如动作识别和点/对象跟踪）上实现了与最先进视频模型（如VideoMAE、V-JEPA）相当的性能，同时在测试几何和密集空间理解的任务上也优于图像模型（如DINOv2）。值得注意的是，RVM在小模型情况下表现出色，无需知识蒸馏，参数效率比竞争的视频掩码自编码器高30倍。此外，我们证明RVM的循环特性允许在长时间范围内稳定传播特征，计算成本为线性，克服了标准时空注意力架构的一些局限性。最后，我们使用定性可视化强调RVM学习了场景语义、结构和运动的丰富表示。


### 论文摘要

We present Recurrent Video Masked-Autoencoders (RVM): a novel video representation learning approach that uses a transformer-based recurrent neural network to aggregate dense image features over time, effectively capturing the spatio-temporal structure of natural video data. RVM learns via an asymmetric masked prediction task requiring only a standard pixel reconstruction objective. This design yields a highly efficient ``generalist'' encoder: RVM achieves competitive performance with state-of-the-art video models (e.g. VideoMAE, V-JEPA) on video-level tasks like action recognition and point/object tracking, while also performing favorably against image models (e.g. DINOv2) on tasks that test geometric and dense spatial understanding. Notably, RVM achieves strong performance in the small-model regime without requiring knowledge distillation, exhibiting up to 30x greater parameter efficiency than competing video masked autoencoders. Moreover, we demonstrate that RVM's recurrent nature allows for stable feature propagation over long temporal horizons with linear computational cost, overcoming some of the limitations of standard spatio-temporal attention-based architectures. Finally, we use qualitative visualizations to highlight that RVM learns rich representations of scene semantics, structure, and motion.

---

## 32. I-Scene: 3D Instance Models are Implicit Generalizable Spatial Learners

**论文链接:** [http://arxiv.org/abs/2512.13683v1](http://arxiv.org/abs/2512.13683v1)

**作者:** Lu Ling, Yunhao Ge, Yichen Sheng, Aniket Bera

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出了一种重新编程预训练3D实例生成器的方法，使其作为场景级别学习者，解决了3D场景生成中的泛化挑战。通过用模型为中心的空间监督取代数据集边界监督，该方法使生成器能够处理未见过的布局和新的物体组合。

### 背景

现有的基于学习的3D场景生成方法在空间理解方面依赖于有限的场景数据集，限制了模型在新布局上的泛化能力。

### 目的

解决3D场景生成中的泛化问题，使模型能够处理未见过的布局和新的物体组合。

### 方法

重新编程预训练的3D实例生成器，使其作为场景级别学习者，用模型为中心的空间监督取代数据集边界监督。用一种以视图为中心的场景空间公式取代广泛使用的规范空间，实现完全前馈、可泛化的场景生成器。

### 主要发现

即使训练场景是随机组合的物体，空间推理仍然会出现。生成器的可转移场景先验为从纯几何线索推断接近性、支撑性和对称性提供了丰富的学习信号。

### 结论

3D实例生成器是一个隐含的空间学习者和推理者，这为交互式3D场景理解和生成的基础模型指明了方向。

### 翻译

通用性仍然是交互式3D场景生成的核心挑战。现有的基于学习的方法将空间理解建立在有限的场景数据集上，限制了在新布局上的泛化能力。我们重新编程了一个预训练的3D实例生成器，使其作为场景级别的学习者，用模型为中心的空间监督取代了数据集边界监督。这种重新编程解锁了生成器的可转移空间知识，使其能够泛化到未见过的布局和新的物体组合。值得注意的是，即使训练场景是随机组合的物体，空间推理仍然会出现。这表明生成器的可转移场景先验为从纯几何线索推断接近性、支撑性和对称性提供了丰富的学习信号。我们用一种以视图为中心的场景空间公式取代了广泛使用的规范空间，产生了一个完全前馈、可泛化的场景生成器，直接从实例模型中学习空间关系。定量和定性结果表明，3D实例生成器是一个隐含的空间学习者和推理者，指向了交互式3D场景理解和生成的基础模型。项目页面：https://luling06.github.io/I-Scene-project/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决交互式3D场景生成中的泛化问题。现有方法受限于有限的场景数据集，无法生成多样化的布局，这对于虚拟内容创建、模拟和具身AI等应用至关重要，因为这些应用需要生成可编辑、有感知能力且空间连贯的物体排列。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到预训练的3D实例模型隐式编码了可转移的空间知识（如深度、遮挡、尺度等），因此重新编程这个先验作为场景级别的空间学习器。他们借鉴了现有的3D生成技术如TRELLIS骨干网络，但创新性地用视图中心场景空间取代规范物体空间，并引入场景上下文注意力机制，使实例生成能够参考全局场景布局。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将预训练的3D实例模型重新编程为空间学习器，利用视图中心场景空间保留几何关系，并从非语义合成场景中学习空间知识。实现流程包括：1) 双分支架构（空间指导分支和实例生成分支）；2) 场景上下文注意力机制；3) 视图中心场景空间表示；4) 在随机组合的非语义场景上训练；5) 使用条件修正流方法作为训练目标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 以模型为中心的监督，重新编程实例模型作为空间学习器；2) 视图中心场景空间，保留几何关系线索；3) 数据独立的布局学习，从非语义场景学习空间知识；4) 强大的泛化能力，能处理未见过的布局。相比之前工作，I-Scene不受限于特定数据集的偏见，能生成更复杂的空间关系（如小物体在大物体上），且保持实例级别的可编辑性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'I-Scene通过重新编程预训练的3D实例模型并利用视图中心场景空间和非语义合成数据，实现了在未见过的布局上生成连贯3D场景的强大泛化能力，同时保持实例级别的可编辑性。'}


### 论文摘要

Generalization remains the central challenge for interactive 3D scene generation. Existing learning-based approaches ground spatial understanding in limited scene dataset, restricting generalization to new layouts. We instead reprogram a pre-trained 3D instance generator to act as a scene level learner, replacing dataset-bounded supervision with model-centric spatial supervision. This reprogramming unlocks the generator transferable spatial knowledge, enabling generalization to unseen layouts and novel object compositions. Remarkably, spatial reasoning still emerges even when the training scenes are randomly composed objects. This demonstrates that the generator's transferable scene prior provides a rich learning signal for inferring proximity, support, and symmetry from purely geometric cues. Replacing widely used canonical space, we instantiate this insight with a view-centric formulation of the scene space, yielding a fully feed-forward, generalizable scene generator that learns spatial relations directly from the instance model. Quantitative and qualitative results show that a 3D instance generator is an implicit spatial learner and reasoner, pointing toward foundation models for interactive 3D scene understanding and generation. Project page: https://luling06.github.io/I-Scene-project/

---

## 33. RoboTracer: Mastering Spatial Trace with Reasoning in Vision-Language Models for Robotics

**论文链接:** [http://arxiv.org/abs/2512.13660v1](http://arxiv.org/abs/2512.13660v1)

**作者:** Enshen Zhou, Cheng Chi, Yibo Li, Jingkun An, Jiayuan Zhang, Shanyu Rong, Yi Han, Yuheng Ji, Mengzhen Liu, Pengwei Wang, Zhongyuan Wang, Lu Sheng, Shanghang Zhang

**发布时间:** 2025-12-15

**备注:** Project page: https://zhoues.github.io/RoboTracer

### GPT解析

### 总结

该论文提出了RoboTracer，一种创新的3D感知视觉语言模型，用于解决机器人空间追踪这一复杂任务。通过结合监督微调和强化微调方法，以及大规模数据集的支持，RoboTracer在空间理解、测量和指代方面表现出色，具有广泛的适用性，可在不同类型的机器人上执行复杂任务。

### 背景

空间追踪作为机器人的基本具身交互能力，本质上具有挑战性，因为它需要多步基于度量的推理，结合复杂的空间指代和真实世界度量测量。然而，现有方法难以处理这种组合任务。

### 目的

提出RoboTracer，一种3D感知的视觉语言模型，实现空间指代和测量功能；增强监督微调过程中的尺度感知能力；通过强化微调推进多步基于度量的推理；创建支持训练的大规模数据集；提出具有挑战性的基准测试来评估空间追踪能力。

### 方法

提出RoboTracer，使用通用空间编码器和回归监督解码器实现3D空间指代和测量；通过度量敏感的过程奖励进行强化微调，推进多步基于度量的推理；引入TraceSpatial，包含3000万个问答对的大规模数据集，涵盖室外/室内/桌面场景；提出TraceSpatial-Bench基准测试。

### 主要发现

RoboTracer在空间理解、测量和指代方面优于基线方法，平均成功率为79.1%；在TraceSpatial-Bench上取得最先进性能，比Gemini-2.5-Pro高出36%的准确率；可与各种控制策略集成，在混乱的真实世界场景中执行长期、动态任务；可在不同机器人(UR5、G1人形机器人)上使用。

### 结论

RoboTracer有效解决了机器人空间追踪的挑战；通过结合监督微调和强化微调，实现了高级空间推理能力；提出的数据集和基准测试为领域发展提供了支持。

### 翻译

空间追踪作为机器人的基本具身交互能力，本质上具有挑战性，因为它需要多步基于度量的推理，结合复杂的空间指代和真实世界度量测量。然而，现有方法难以处理这种组合任务。为此，我们提出了RoboTracer，一种3D感知的视觉语言模型，首次通过通用空间编码器和回归监督解码器实现3D空间指代和测量，在监督微调过程中增强尺度感知能力。此外，RoboTracer通过具有度量敏感过程奖励的强化微调，推进多步基于度量的推理，监督关键中间感知线索以准确生成空间轨迹。为支持SFT和RFT训练，我们引入了TraceSpatial，一个包含3000万个问答对的大规模数据集，涵盖室外/室内/桌面场景，并支持复杂的推理过程(最多9步)。我们还提出了TraceSpatial-Bench，一个具有挑战性的基准测试，填补了评估空间追踪能力的空白。实验结果表明，RoboTracer在空间理解、测量和指代方面优于基线方法，平均成功率为79.1%，并且在TraceSpatial-Bench上以较大优势取得了最先进的性能，准确率比Gemini-2.5-Pro高出36%。值得注意的是，RoboTracer可以与各种控制策略集成，在混乱的真实世界场景中执行长期、动态任务，适用于不同类型的机器人(UR5、G1人形机器人)。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决机器人空间追踪问题，即让机器人能够根据空间约束指令（如'从左到右浇花，喷壶在每个花上方1-5厘米处'）生成3D位置序列。这个问题很重要，因为空间追踪是机器人具身交互的基础能力，现有方法难以处理需要多步骤推理和真实世界度量的复杂任务，限制了机器人在复杂环境中的自主性和适应性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将空间追踪任务分解为3D空间指代和3D空间测量两个关键组件，借鉴了现有视觉-语言模型(VLMs)的基本架构，但针对3D空间理解和度量感知进行了改进。他们参考了强化微调(RFT)方法如GRPO，利用现有3D几何模型作为先验知识，并借鉴了2D视觉轨迹生成方法但扩展到3D空间。设计上采用了通用空间编码器和比例解码器，结合监督微调(SFT)和强化微调(RFT)两阶段训练策略。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个3D感知的VLM，通过通用空间编码器灵活整合几何信息，使用比例解码器增强真实世界尺度感知，并采用两阶段训练方法。整体流程：1)输入处理(接受RGB图像和任务指令，可选几何配置)；2)编码阶段(视觉编码器处理图像，空间编码器处理几何信息，比例解码器预测度量比例)；3)推理阶段(LLM处理指令和编码信息，生成多步骤推理和空间轨迹)；4)训练阶段(SFT学习基础能力，RFT使用度量敏感奖励改进推理)；5)输出3D空间轨迹和可选中间推理步骤。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)3D感知VLM架构，包含通用空间编码器和比例解码器；2)两阶段训练方法(SFT+RFT)；3)大规模TraceSpatial数据集(4.5M样本，30M问答对)；4)新的TraceSpatial-Bench评估基准；5)实际应用验证。相比之前工作，RoboTracer从2D扩展到3D，增强了组合推理能力，具有真正的度量感知，能处理任意几何输入，且数据规模和质量远超现有工作。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RoboTracer通过结合3D感知的视觉-语言模型架构、两阶段训练方法和大规模数据集，首次实现了机器人空间追踪任务中的多步骤、度量引导推理，显著提升了机器人在复杂3D环境中执行空间约束指令的能力。'}


### 论文摘要

Spatial tracing, as a fundamental embodied interaction ability for robots, is inherently challenging as it requires multi-step metric-grounded reasoning compounded with complex spatial referring and real-world metric measurement. However, existing methods struggle with this compositional task. To this end, we propose RoboTracer, a 3D-aware VLM that first achieves both 3D spatial referring and measuring via a universal spatial encoder and a regression-supervised decoder to enhance scale awareness during supervised fine-tuning (SFT). Moreover, RoboTracer advances multi-step metric-grounded reasoning via reinforcement fine-tuning (RFT) with metric-sensitive process rewards, supervising key intermediate perceptual cues to accurately generate spatial traces. To support SFT and RFT training, we introduce TraceSpatial, a large-scale dataset of 30M QA pairs, spanning outdoor/indoor/tabletop scenes and supporting complex reasoning processes (up to 9 steps). We further present TraceSpatial-Bench, a challenging benchmark filling the gap to evaluate spatial tracing. Experimental results show that RoboTracer surpasses baselines in spatial understanding, measuring, and referring, with an average success rate of 79.1%, and also achieves SOTA performance on TraceSpatial-Bench by a large margin, exceeding Gemini-2.5-Pro by 36% accuracy. Notably, RoboTracer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (UR5, G1 humanoid) in cluttered real-world scenes.

---

## 34. DePT3R: Joint Dense Point Tracking and 3D Reconstruction of Dynamic Scenes in a Single Forward Pass

**论文链接:** [http://arxiv.org/abs/2512.13122v1](http://arxiv.org/abs/2512.13122v1)

**作者:** Vivek Alumootil, Tuan-Anh Vu, M. Khalid Jawed

**发布时间:** 2025-12-15

**备注:** This is a work in progress

### GPT解析

### 总结

DePT3R是一个新颖的框架，能够在单一前向传递中同时执行密集点跟踪和动态场景的3D重建，无需已知相机姿态，显著提高了适应性和效率。

### 背景

当前动态场景密集3D点跟踪方法通常依赖于成对处理、已知相机姿态或假设输入帧有时间顺序，限制了其灵活性和适用性。最近进展已实现从大规模无姿态图像集合中进行高效3D重建，为动态场景理解的统一方法提供了机会。

### 目的

提出DePT3R框架，同时执行密集点跟踪和动态场景的3D重建，从多张图像在单一前向传递中完成，提高方法的适应性和效率。

### 方法

通过多任务学习实现，使用强大的骨干网络提取深时空特征，并用密集预测头回归像素级映射。DePT3R操作不需要相机姿态，特别适合快速变化的动态环境。

### 主要发现

在多个具有挑战性的动态场景基准测试中验证了DePT3R，展示了强大的性能，与现有最先进方法相比在内存效率方面有显著改进。数据和代码可通过开放存储库获取。

### 结论

DePT3R是一个有效的框架，能够在不依赖相机姿态的情况下同时进行密集点跟踪和3D重建，在动态场景中表现出色，特别是在内存效率方面有显著优势。

### 翻译

当前动态场景中密集3D点跟踪方法通常依赖于成对处理，需要已知的相机姿态，或假设输入帧有时间顺序，限制了它们的灵活性和适用性。此外，最近的进展已成功实现从大规模、无姿态图像集合中进行高效3D重建，凸显了动态场景理解统一方法的机会。受此启发，我们提出了DePT3R，一个新颖的框架，能够在单一前向传递中同时从多张图像执行密集点跟踪和动态场景的3D重建。这种多任务学习是通过使用强大的骨干网络提取深时空特征，并用密集预测头回归像素级映射来实现的。关键的是，DePT3R操作不需要相机姿态，显著提高了其适应性和效率——这对快速变化的动态环境尤其重要。我们在几个涉及动态场景的具有挑战性的基准测试中验证了DePT3R，展示了其强大的性能和与现有最先进方法相比在内存效率方面的显著改进。数据和代码可通过开放存储库获取：https://github.com/StructuresComp/DePT3R

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文旨在解决动态场景中密集3D点跟踪和3D重建的联合处理问题，特别是消除对已知相机姿态、成对处理或时间顺序假设的依赖。这个问题在自主导航、增强现实和机器人等领域至关重要，因为准确跟踪动态环境中的点和重建3D结构是智能系统有效感知和与现实世界交互的基础，而传统方法在处理复杂动态场景时往往表现不佳且效率低下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到近期DUSt3R等工作的启发，注意到这些方法虽为静态场景设计但也在动态内容上表现良好。作者观察到现有方法如VGGT在动态场景处理上存在局限性：大幅非刚性变形下表现不佳，且内存占用限制了密集跟踪效率。作者基于VGGT架构进行扩展，借鉴了St4RTrack的时间相关点图表示和DINOv2图像标记化技术，同时添加了专用的运动预测头和查询机制，以实现更高效的全局时间推理。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是采用'帧到查询'的公式，通过全局图像聚合在单次前向传递中同时实现点跟踪和3D重建，无需帧到帧的链接过程。整体流程为：1) 输入多帧图像和查询时间；2) 为每帧预测点位置和运动场；3) 使用扩展的VGGT骨干网络处理图像特征；4) 通过专用运动头预测点运动；5) 添加相机内参嵌入提高准确性；6) 使用多任务损失函数联合优化所有预测。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 帧到查询的运动场预测公式，实现长程变形推理；2) 全局时间注意力机制，提高计算效率和准确性；3) 相机内参嵌入，解决尺度模糊问题；4) 专用运动预测头，与重建任务共享特征；5) 两阶段训练策略，提高模型稳定性。相比之前工作，DePT3R不需要成对处理、已知相机姿态或时间顺序假设，内存效率更高，在处理大幅非刚性变形时表现更好，且能在单次前向传递中处理大量帧。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DePT3R提出了一种创新的前馈框架，通过全局图像聚合和帧到查询的公式，在单次前向传递中实现了动态场景的密集点跟踪和3D重建，无需相机姿态信息，同时显著提高了内存效率和跟踪精度。'}


### 论文摘要

Current methods for dense 3D point tracking in dynamic scenes typically rely on pairwise processing, require known camera poses, or assume a temporal ordering to input frames, constraining their flexibility and applicability. Additionally, recent advances have successfully enabled efficient 3D reconstruction from large-scale, unposed image collections, underscoring opportunities for unified approaches to dynamic scene understanding. Motivated by this, we propose DePT3R, a novel framework that simultaneously performs dense point tracking and 3D reconstruction of dynamic scenes from multiple images in a single forward pass. This multi-task learning is achieved by extracting deep spatio-temporal features with a powerful backbone and regressing pixel-wise maps with dense prediction heads. Crucially, DePT3R operates without requiring camera poses, substantially enhancing its adaptability and efficiency-especially important in dynamic environments with rapid changes. We validate DePT3R on several challenging benchmarks involving dynamic scenes, demonstrating strong performance and significant improvements in memory efficiency over existing state-of-the-art methods. Data and codes are available via the open repository: https://github.com/StructuresComp/DePT3R

---

## 35. Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos

**论文链接:** [http://arxiv.org/abs/2512.13080v1](http://arxiv.org/abs/2512.13080v1)

**作者:** Yicheng Feng, Wanpeng Zhang, Ye Wang, Hao Luo, Haoqi Yuan, Sipeng Zheng, Zongqing Lu

**发布时间:** 2025-12-15

### GPT解析

### 总结

该研究提出了一种空间感知的视觉-语言-动作预训练范式，通过在预训练过程中对齐视觉空间和物理空间，使模型在机器人策略学习前获得三维空间理解能力。研究者实现了VIPA-VLA模型，一种双编码器架构，通过三维视觉编码器增强语义视觉表示。当应用于下游机器人任务时，该模型显著改善了二维视觉与三维动作之间的对齐，产生更鲁棒和可泛化的机器人策略。

### 背景

视觉-语言-动作模型为机器人学习提供了一种有前景的范式，但大多数现有方法依赖二维视觉输入在三维物理环境中执行动作，导致感知与动作基础之间存在显著差距。

### 目的

为了弥合二维视觉感知与三维物理动作之间的差距，研究者提出了一种空间感知的VLA预训练范式，使模型在机器人策略学习前获得三维空间理解能力。

### 方法

研究者从预训练的视觉语言模型开始，利用大规模人类演示视频提取三维视觉和三维动作注释，形成新的监督源，将二维视觉观察与三维空间推理对齐。实现了VIPA-VLA双编码器架构，集成了三维视觉编码器来增强语义视觉表示。

### 主要发现

当适应下游机器人任务时，VIPA-VLA实现了二维视觉和三维动作之间显著改善的对齐，产生了更鲁棒和可泛化的机器人策略。

### 结论

通过空间感知的预训练范式和VIPA-VLA模型，成功弥合了二维视觉感知与三维物理动作之间的差距，为机器人学习提供了更有效的解决方案。

### 翻译

视觉-语言-动作模型通过整合视觉感知与语言引导的策略学习，为机器人学习提供了一种有前景的范式。然而，大多数现有方法依赖于二维视觉输入在三维物理环境中执行动作，这导致了感知与动作基础之间的重要差距。为了弥合这一差距，我们提出了一种空间感知的VLA预训练范式，在预训练过程中明确对齐视觉空间和物理空间，使模型在机器人策略学习前获得三维空间理解能力。从预训练的视觉语言模型开始，我们利用大规模人类演示视频提取三维视觉和三维动作注释，形成一种新的监督源，用于将二维视觉观察与三维空间推理对齐。我们通过VIPA-VLA实现了这一范式，它是一种双编码器架构，集成了三维视觉编码器，用于通过三维感知特征增强语义视觉表示。当适应下游机器人任务时，VIPA-VLA实现了二维视觉和三维动作之间显著改善的对齐，产生了更鲁棒和可泛化的机器人策略。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉-语言-行动（VLA）模型中2D视觉感知与3D物理行动之间的显著差距问题。这个问题很重要，因为它限制了机器人对物理世界的有效理解和交互，导致模型难以将2D视觉信息准确映射到3D空间中的行动，从而影响机器人在复杂环境中的表现和泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有VLA模型依赖2D视觉输入进行3D物理环境行动的局限性，然后提出利用大规模人类演示视频作为监督来源，因为人类视频中自然包含了2D视觉观察与3D物理行动之间的对应关系。他们借鉴了现有的视觉-语言模型作为基础，利用3D视觉技术进行点云估计，采用人类动作捕捉和手部姿态标注技术，并参考了现有VLA模型架构，但增加了3D空间感知能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过'视觉-物理对齐'建立2D视觉观察与3D物理空间之间的明确对应关系。整体流程包括：1)构建Hand3D数据集，从人类视频中提取3D视觉和动作标注；2)设计VIPA-VLA双编码器架构，结合语义视觉和3D空间特征；3)进行两阶段预训练，第一阶段对齐2D视觉与3D空间，第二阶段学习物理基础的动作先验；4)在机器人数据上进行微调，添加动作头生成可执行行动。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)空间感知VLA预训练范式，在预训练阶段显式对齐视觉与物理空间；2)Hand3D数据集，提供3D视觉和动作标注的监督信号；3)VIPA-VLA双编码器架构，有效整合语义和空间特征；4)两阶段预训练策略。相比之前工作，该方法明确建模2D视觉与3D物理空间的对齐关系，专注于建立3D感知与物理行动的对应，利用人类视频而非昂贵的机器人数据进行预训练，避免了embodiment mismatch问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '该论文通过从人类视频中提取3D视觉和动作标注，提出了一种空间感知的视觉-语言-行动预训练范式，有效弥合了2D视觉感知与3D物理行动之间的差距，显著提升了机器人在各种任务中的空间定位能力和泛化性能。'}


### 论文摘要

Vision-Language-Action (VLA) models provide a promising paradigm for robot learning by integrating visual perception with language-guided policy learning. However, most existing approaches rely on 2D visual inputs to perform actions in 3D physical environments, creating a significant gap between perception and action grounding. To bridge this gap, we propose a Spatial-Aware VLA Pretraining paradigm that performs explicit alignment between visual space and physical space during pretraining, enabling models to acquire 3D spatial understanding before robot policy learning. Starting from pretrained vision-language models, we leverage large-scale human demonstration videos to extract 3D visual and 3D action annotations, forming a new source of supervision that aligns 2D visual observations with 3D spatial reasoning. We instantiate this paradigm with VIPA-VLA, a dual-encoder architecture that incorporates a 3D visual encoder to augment semantic visual representations with 3D-aware features. When adapted to downstream robot tasks, VIPA-VLA achieves significantly improved grounding between 2D vision and 3D action, resulting in more robust and generalizable robotic policies.

---

## 36. SLIM-VDB: A Real-Time 3D Probabilistic Semantic Mapping Framework

**论文链接:** [http://arxiv.org/abs/2512.12945v1](http://arxiv.org/abs/2512.12945v1)

**作者:** Anja Sheppard, Parker Ewen, Joey Wilson, Advaith V. Sethuraman, Benard Adewole, Anran Li, Yuzhen Chen, Ram Vasudevan, Katherine A. Skinner

**发布时间:** 2025-12-15

**备注:** Accepted into R-AL

### GPT解析

### 总结

这篇论文介绍了SLIM-VDB，一种新的轻型语义映射系统，具有概率语义融合功能，支持封闭集或开放集字典。

### 背景

计算机图形学社区的数据结构进步（如OpenVDB）在体积场景表示中显示出计算和内存效率的显著提高。虽然OpenVDB已被用于机器人应用中的几何映射，但使用OpenVDB进行场景理解的语义映射仍未被探索。此外，现有的语义映射系统缺乏在单一框架内集成固定类别和开放语言标签预测的支持。

### 目的

提出一种新颖的3D语义映射系统，利用OpenVDB数据结构，并集成统一的贝叶斯更新框架，用于封闭集和开放集语义融合。

### 方法

利用OpenVDB数据结构，并集成统一的贝叶斯更新框架进行语义融合。

### 主要发现

与当前最先进的语义映射方法相比，SLIM-VDB在内存和集成时间方面实现了显著减少，同时保持了相当的映射精度。

### 结论

SLIM-VDB是一种高效的语义映射系统，在保持精度的同时显著提高了效率。

### 翻译

本文介绍了SLIM-VDB，一种新的轻型语义映射系统，具有用于封闭集或开放集字典的概率语义融合功能。计算机图形学社区的数据结构进步，如OpenVDB，在体积场景表示中显示出计算和内存效率的显著提高。虽然OpenVDB已被用于机器人应用中的几何映射，但使用OpenVDB进行场景理解的语义映射仍未被探索。此外，现有的语义映射系统缺乏在单一框架内集成固定类别和开放语言标签预测的支持。在本文中，我们提出了一种新颖的3D语义映射系统，利用OpenVDB数据结构，并集成了统一的贝叶斯更新框架，用于封闭集和开放集语义融合。与当前最先进的语义映射方法相比，我们提出的SLIM-VDB框架在内存和集成时间方面实现了显著减少，同时保持了相当的映射精度。带有Python接口的开源C++代码库可在https://github.com/umfieldrobotics/slim-vdb获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有语义地图系统内存占用大、难以实时运行的问题，以及缺乏对固定类别(封闭集)和开放语言标签(开放集)预测的统一支持。这个问题很重要，因为机器人需要准确的世界地图和场景理解来指导行动，而实时语义地图对移动机器人至关重要，同时能够处理两种语义类型的框架会更加灵活和强大，适应复杂环境需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有语义地图系统的局限性，然后借鉴计算机图形学领域的OpenVDB数据结构，该结构在体积场景表示方面表现出色。在语义融合方面，作者借鉴了贝叶斯推理方法，如ConvBKI和LatentBKI，分别用于封闭集和开放集语义融合。在几何映射方面，借鉴了VDBFusion的工作。作者将这些现有技术与创新的统一贝叶斯更新框架相结合，创造出SLIM-VDB系统，实现了高效、实时的语义地图构建。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用OpenVDB数据结构的高效体积表示能力，结合统一的贝叶斯概率框架，实现实时、内存高效的语义地图构建，同时支持封闭集和开放集语义。整体流程包括：1)感知阶段：输入RGB图像、深度图像和机器人位姿，生成语义预测；2)传感器处理：将语义分割结果投影到3D空间形成语义点云；3)主要集成循环：进行光线投射和TSDF更新进行几何更新，使用贝叶斯推理进行语义更新；4)可视化：使用NanoVDB进行GPU加速的地图渲染。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将OpenVDB数据结构应用于语义地图构建，提供O(1)的查找、插入和删除时间；2)统一的贝叶斯语义融合框架，同时支持封闭集和开放集语义；3)高效的内存和计算效率，显著减少内存占用和集成时间；4)开源实现，提供C++代码库和Python接口。相比之前工作，SLIM-VDB不同于VDBFusion(只支持几何映射)、ConvBKI和SEE-CSOM(只支持封闭集语义)、LatentBKI(无法实时运行且内存占用大)以及SNI-SLAM(需要专业级GPU)，实现了更高效、更灵活的语义地图构建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SLIM-VDB首次将OpenVDB数据结构与统一的贝叶斯概率框架相结合，实现了实时、内存高效且同时支持封闭集和开放集语义的3D语义地图构建系统。'}


### 论文摘要

This paper introduces SLIM-VDB, a new lightweight semantic mapping system with probabilistic semantic fusion for closed-set or open-set dictionaries. Advances in data structures from the computer graphics community, such as OpenVDB, have demonstrated significantly improved computational and memory efficiency in volumetric scene representation. Although OpenVDB has been used for geometric mapping in robotics applications, semantic mapping for scene understanding with OpenVDB remains unexplored. In addition, existing semantic mapping systems lack support for integrating both fixed-category and open-language label predictions within a single framework. In this paper, we propose a novel 3D semantic mapping system that leverages the OpenVDB data structure and integrates a unified Bayesian update framework for both closed- and open-set semantic fusion. Our proposed framework, SLIM-VDB, achieves significant reduction in both memory and integration times compared to current state-of-the-art semantic mapping approaches, while maintaining comparable mapping accuracy. An open-source C++ codebase with a Python interface is available at https://github.com/umfieldrobotics/slim-vdb.

---

## 37. MRD: Using Physically Based Differentiable Rendering to Probe Vision Models for 3D Scene Understanding

**论文链接:** [http://arxiv.org/abs/2512.12307v1](http://arxiv.org/abs/2512.12307v1)

**作者:** Benjamin Beilharz, Thomas S. A. Wallis

**发布时间:** 2025-12-13

**备注:** 18 pages, 6 figures. Supplementary material and code will be provided at the end of January

### GPT解析

### 总结

本文介绍了MRD（metamers rendered differentiably）方法，这是一种基于物理的可微分渲染技术，用于探测视觉模型对3D场景属性的隐式理解。

### 背景

深度学习在视觉任务中取得成功，但其模型表示和决策难以解释。视觉模型虽在2D数据上训练，但被认为能形成对底层3D场景的隐式理解。

### 目的

开发一种方法来探测视觉模型如何理解和表示3D场景的物理属性，以增进对模型决策过程的理解。

### 方法

MRD方法通过寻找物理不同但产生相同模型激活（模型元形）的3D场景参数来探测模型。与基于像素的方法不同，这些重建结果基于物理场景描述，可独立控制形状、材质和光照等参数。

### 主要发现

评估显示模型在目标场景和优化场景间激活高度相似，但视觉效果不同。这些重建能揭示模型对哪些物理属性敏感或不变。

### 结论

MRD方法通过分析物理参数如何驱动模型响应变化，有助于增进对计算机视觉和人类视觉的理解。

### 翻译

虽然深度学习方法在许多视觉基准测试中取得了令人印象深刻的成功，但理解和解释这些模型的表示和决策仍然很困难。虽然视觉模型通常在2D输入上训练，但人们通常假设它们能够对底层3D场景形成隐式表示（例如，对部分遮挡表现出容忍度，或能够推理相对深度）。在这里，我们介绍了MRD（metamers rendered differentiably），一种基于物理的可微分渲染方法，通过寻找物理不同但产生相同模型激活（即模型元形）的3D场景参数，来探测视觉模型对生成性3D场景属性的隐式理解。与以往评估模型表示的基于像素的方法不同，这些重建结果总是基于物理场景描述。这意味着，例如，我们可以在保持材质和光照不变的情况下探测模型对物体形状的敏感性。作为原理证明，我们评估了多个模型恢复几何形状（形状）和双向反射分布函数（材质）场景参数的能力。结果显示目标场景和优化场景之间的模型激活具有高度相似性，但视觉效果有所不同。从质量上讲，这些重建有助于研究模型对哪些物理场景属性敏感或不变。MRD通过使分析物理参数如何驱动模型响应的变化，有望促进我们对计算机视觉和人类视觉的理解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决的问题是：如何理解和解释深度视觉模型的内部表示和决策过程，特别是这些模型如何理解3D场景。这个问题很重要，因为随着深度学习在计算机视觉领域的广泛应用，理解这些模型如何'思考'变得至关重要，有助于提高模型的透明度和可信度，揭示模型可能存在的偏见或局限性，并推动计算机视觉和人类视觉的共同研究。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到深度学习模型虽然只在2D图像上训练，但似乎能够理解3D场景结构。他们想找到一种方法来探测模型对3D场景的隐式理解，而不仅仅是表面的2D特征。作者借鉴了基于物理的渲染和可微分渲染技术，以及人类视觉研究中使用的metamer概念（产生相同感知但物理上不同的刺激），将这些技术结合起来创建了MRD方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用基于物理的可微分渲染技术来找到与目标场景在视觉模型中产生相同激活但物理上不同的3D场景参数。整体流程是：从已知参数的场景开始，渲染一组真实图像；初始化一个具有不同参数的新场景；在采样相机位置渲染图像并计算与真实图像的损失；计算梯度并更新目标场景参数；重复优化过程直到找到产生相同模型激活的物理不同场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次将基于物理的可微分渲染与metamer概念结合；能够将模型激活与物理环境属性直接联系起来；允许分离物理原因，专门探测模型对特定场景属性的敏感性；提供在物理单位中解释视觉模型表示的方法。相比之前工作，MRD不同于传统的像素级方法，能够探测3D场景属性理解；不同于仅关注2D特征的解释方法；不同于基于神经网络的逆渲染方法，基于物理光传输；不同于简单的特征可视化，能找到物理上不同但在模型中等效的场景表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了MRD方法，通过基于物理的可微分渲染技术来探测视觉模型对3D场景的隐式理解，建立了模型激活与物理场景属性之间的桥梁，为理解和解释深度视觉模型提供了新视角。'}


### 论文摘要

While deep learning methods have achieved impressive success in many vision benchmarks, it remains difficult to understand and explain the representations and decisions of these models. Though vision models are typically trained on 2D inputs, they are often assumed to develop an implicit representation of the underlying 3D scene (for example, showing tolerance to partial occlusion, or the ability to reason about relative depth). Here, we introduce MRD (metamers rendered differentiably), an approach that uses physically based differentiable rendering to probe vision models' implicit understanding of generative 3D scene properties, by finding 3D scene parameters that are physically different but produce the same model activation (i.e. are model metamers). Unlike previous pixel-based methods for evaluating model representations, these reconstruction results are always grounded in physical scene descriptions. This means we can, for example, probe a model's sensitivity to object shape while holding material and lighting constant. As a proof-of-principle, we assess multiple models in their ability to recover scene parameters of geometry (shape) and bidirectional reflectance distribution function (material). The results show high similarity in model activation between target and optimized scenes, with varying visual results. Qualitatively, these reconstructions help investigate the physical scene attributes to which models are sensitive or invariant. MRD holds promise for advancing our understanding of both computer and human vision by enabling analysis of how physical scene parameters drive changes in model responses.

---

## 38. A Multi-Year Urban Streetlight Imagery Dataset for Visual Monitoring and Spatio-Temporal Drift Detection

**论文链接:** [http://arxiv.org/abs/2512.12205v1](http://arxiv.org/abs/2512.12205v1)

**作者:** Peizheng Li, Ioannis Mavromatis, Ajith Sahadevan, Tim Farnham, Adnan Aijaz, Aftab Khan

**发布时间:** 2025-12-13

**备注:** 10 pages, 7 figures. Submitted to Data in Brief (Elsevier)

### GPT解析

### 总结

该研究介绍了一个大规模、长期的城市路灯视觉数据集，由英国布里斯托尔部署的22个固定角度摄像头从2021年到2025年拍摄。数据集包含超过526,000张图像，每小时收集一次，涵盖不同光照、天气和季节条件，并附带丰富的元数据。研究还提供了基于卷积变分自编码器的自监督框架和两种漂移指标，为评估长期视觉系统稳定性提供了真实基准。

### 背景

随着智能城市部署的发展，视觉漂移、异常检测和MLOps策略的研究需要大规模、长期的真实世界数据支持，而现有数据集可能不足以满足这些研究需求。

### 目的

创建一个大规模、长期的城市路灯视觉数据集，用于详细研究智能城市部署中的视觉漂移、异常检测和MLOps策略，同时提供自监督框架和漂移指标以促进二次分析。

### 方法

部署22个固定角度摄像头在英国布里斯托尔从2021年到2025年收集数据；每小时收集图像记录不同条件；为每张图像提供元数据；基于卷积变分自编码器构建自监督框架；为每个摄像头节点和日/夜图像集分别训练模型；定义相对质心漂移和相对重建误差两种漂移指标。

### 主要发现

创建了包含超过526,000张图像的大规模数据集，覆盖多种环境条件；提供的自监督框架能有效处理长期视觉数据；定义的漂移指标能量化模型性能随时间的变化。

### 结论

该数据集为评估长期模型稳定性、漂移感知学习和可部署视觉系统提供了真实的、细粒度基准，数据集公开发布支持可重复性和多种下游应用。

### 翻译

我们提出了一个大规模、长期的城市路灯视觉数据集，由英国布里斯托尔部署的22个固定角度摄像头从2021年到2025年拍摄。该数据集包含超过526,000张图像，每小时收集一次，涵盖不同的光照、天气和季节条件。每张图像都附带丰富的元数据，包括时间戳、GPS坐标和设备标识符。这一独特的真实世界数据集 enables详细研究智能城市部署中的视觉漂移、异常检测和MLOps策略。为了促进二次分析，我们还提供了一个基于卷积变分自编码器(CNN-VAEs)的自监督框架。模型为每个摄像头节点和日/夜图像集分别训练。我们定义了两种每样本漂移指标：相对质心漂移，捕获与基线季度的潜在空间偏差；和相对重建误差，测量归一化图像域退化。该数据集为评估长期模型稳定性、漂移感知学习和可部署视觉系统提供了真实的、细粒度基准。图像和结构化元数据以JPEG和CSV格式公开发布，支持可重复性和下游应用，如路灯监测、天气推断和城市场景理解。该数据集可在https://doi.org/10.5281/zenodo.17781192和https://doi.org/10.5281/zenodo.17859120找到。


### 论文摘要

We present a large-scale, longitudinal visual dataset of urban streetlights captured by 22 fixed-angle cameras deployed across Bristol, U.K., from 2021 to 2025. The dataset contains over 526,000 images, collected hourly under diverse lighting, weather, and seasonal conditions. Each image is accompanied by rich metadata, including timestamps, GPS coordinates, and device identifiers. This unique real-world dataset enables detailed investigation of visual drift, anomaly detection, and MLOps strategies in smart city deployments. To promtoe seconardary analysis, we additionally provide a self-supervised framework based on convolutional variational autoencoders (CNN-VAEs). Models are trained separately for each camera node and for day/night image sets. We define two per-sample drift metrics: relative centroid drift, capturing latent space deviation from a baseline quarter, and relative reconstruction error, measuring normalized image-domain degradation. This dataset provides a realistic, fine-grained benchmark for evaluating long-term model stability, drift-aware learning, and deployment-ready vision systems. The images and structured metadata are publicly released in JPEG and CSV formats, supporting reproducibility and downstream applications such as streetlight monitoring, weather inference, and urban scene understanding. The dataset can be found at https://doi.org/10.5281/zenodo.17781192 and https://doi.org/10.5281/zenodo.17859120.

---

## 39. Audio-Visual Camera Pose Estimationn with Passive Scene Sounds and In-the-Wild Video

**论文链接:** [http://arxiv.org/abs/2512.12165v1](http://arxiv.org/abs/2512.12165v1)

**作者:** Daniel Adebi, Sagnik Majumder, Kristen Grauman

**发布时间:** 2025-12-13

### GPT解析

### 总结

该研究探索了利用被动场景声音作为视觉信息的补充，用于相对相机姿态估计，提出了一种简单但有效的视听框架，在视觉条件恶化的情况下表现出色。

### 背景

理解相机运动是具身感知和3D场景理解的基本问题。尽管视觉方法发展迅速，但在视觉条件恶化的情况下（如运动模糊或遮挡）往往表现不佳。

### 目的

展示被动场景声音可以为野外视频中的相对相机姿态估计提供补充线索，特别是在视觉信息不可靠的情况下。

### 方法

研究团队引入了一个简单但有效的视听框架，将到达方向(DOA)谱和双耳嵌入集成到最先进的纯视觉姿态估计模型中。

### 主要发现

在两个大型数据集上的结果显示，相比强大的视觉基线模型，该方法取得了持续的性能提升，并且在视觉信息被损坏时表现出鲁棒性。

### 结论

据作者所知，这是首次成功利用音频进行真实世界视频中相对相机姿态估计的研究，确立了偶然的日常音频作为解决经典空间挑战的一个意外但有前景的信号。

### 翻译

理解相机运动是具身感知和3D场景理解的基本问题。虽然视觉方法已经取得了快速发展，但它们在视觉条件恶化的情况下（如运动模糊或遮挡）往往表现不佳。在这项工作中，我们表明被动场景声音可以为野外视频中的相对相机姿态估计提供补充线索。我们介绍了一个简单但有效的视听框架，将到达方向(DOA)谱和双耳嵌入集成到最先进的纯视觉姿态估计模型中。我们在两个大型数据集上的结果显示，相比强大的视觉基线模型，该方法取得了持续的性能提升，并且在视觉信息被损坏时表现出鲁棒性。据我们所知，这代表了首次成功利用音频进行真实世界视频中相对相机姿态估计的工作，它确立了偶然的日常音频作为解决经典空间挑战的一个意外但有前景的信号。项目：http://vision.cs.utexas.edu/projects/av_camera_pose.

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是：在真实世界视频中进行相机姿态估计时，如何利用被动场景声音来辅助视觉信息，提高相机姿态估计的准确性，特别是在视觉信息受损的情况下（如运动模糊、光照不足、遮挡等）。这个问题在现实中很重要，因为许多实际应用（如增强现实、机器人导航、自动驾驶）都需要准确的相机姿态信息，而这些应用场景常常会遇到视觉条件差的情况。此外，这是首次成功利用音频进行真实世界视频中的相对相机姿态估计工作，填补了研究空白。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到在视觉条件差的场景（如黑暗拥挤的演唱会），尽管视觉信息严重退化，但周围的声音却能提供清晰的空间信息。他们意识到被动场景声音具有非侵入性、多样性和自然可用等优点。设计方法时，作者借鉴了现有的视觉相机姿态估计模型（Reloc3r），以及音频处理技术（如MUSIC++算法进行方向到达估计）和自监督学习方法（设计新颖视角声学合成NVAS任务）。作者设计了一个空间音频编码器，结合两种互补的音频特征，并将这些特征与视觉特征进行后期融合，增强现有的视觉模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用被动场景声音中的空间信息来补充视觉信息，特别是在视觉信息受损的情况下。通过结合两种互补的音频特征（显式的方向到达信息和隐式的双耳音频特征）提供丰富的空间线索。整体实现流程：1) 输入处理：视频帧通过Reloc3r提取视觉特征，音频片段通过STFT转换为频谱图；2) 空间音频编码：使用MUSIC++算法计算DoA频谱，训练NVAS模型学习双耳音频特征，然后将两种特征拼接；3) 音频-视觉融合：将空间音频嵌入与视觉特征后期融合；4) 姿态预测：融合后的特征输入到姿态预测头，估计相对相机姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 首次利用被动场景声音（非主动发出的声音）进行真实世界视频的相机姿态估计；2) 设计空间音频编码器，结合方向到达频谱和双耳音频嵌入两种互补特征；3) 使用多样化的真实世界视频数据训练模型，而非静态环境或模拟数据；4) 方法在视觉信息受损情况下仍保持良好性能，展示跨模态鲁棒性。相比之前工作的不同：之前工作主要使用主动声波定位或模拟音频，在静态环境中训练；本文使用自然发生的场景声音和真实世界数据，专注于相机姿态估计这一任务，并采用分析性与学习性特征相结合的方式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文首次证明了可以利用被动场景声音作为互补信号，通过结合方向到达信息和双耳音频特征，显著提升真实世界视频中的相机姿态估计准确性，特别是在视觉信息受损的情况下展现出更强的鲁棒性。'}


### 论文摘要

Understanding camera motion is a fundamental problem in embodied perception and 3D scene understanding. While visual methods have advanced rapidly, they often struggle under visually degraded conditions such as motion blur or occlusions. In this work, we show that passive scene sounds provide complementary cues for relative camera pose estimation for in-the-wild videos. We introduce a simple but effective audio-visual framework that integrates direction-ofarrival (DOA) spectra and binauralized embeddings into a state-of-the-art vision-only pose estimation model. Our results on two large datasets show consistent gains over strong visual baselines, plus robustness when the visual information is corrupted. To our knowledge, this represents the first work to successfully leverage audio for relative camera pose estimation in real-world videos, and it establishes incidental, everyday audio as an unexpected but promising signal for a classic spatial challenge. Project: http://vision.cs.utexas.edu/projects/av_camera_pose.

---

## 40. Aion: Towards Hierarchical 4D Scene Graphs with Temporal Flow Dynamics

**论文链接:** [http://arxiv.org/abs/2512.11903v1](http://arxiv.org/abs/2512.11903v1)

**作者:** Iacopo Catalano, Eduardo Montijano, Javier Civera, Julio A. Placed, Jorge Pena-Queralta

**发布时间:** 2025-12-10

### GPT解析

### 总结

本文提出了Aion框架，将时间流动态直接嵌入到分层3D场景图中，有效整合了时间维度，提高了自主导航在复杂动态环境中的规划和交互能力。

### 背景

自主导航在动态环境中需要能够捕捉语义结构和时间演化的空间表示。现有的3D场景图(3DSGs)虽然提供分层多分辨率抽象，编码几何和语义信息，但动态扩展主要关注单个对象或智能体。而动态地图(MoDs)模拟典型运动模式和时序规律，但通常基于网格离散化，缺乏语义感知，且不易扩展到大型环境。

### 目的

开发一个能够有效整合时间维度的框架，将时间流动态直接嵌入到分层3D场景图中，提高自主导航在复杂动态环境中的能力。

### 方法

Aion框架采用基于图的稀疏MoD表示来捕获任意时间间隔的运动流，并将其附加到场景图中的导航节点上，产生更可解释和可扩展的预测。

### 主要发现

Aion框架产生了更可解释和可扩展的预测，提高了在复杂动态环境中的规划和交互能力。

### 结论

Aion框架有效整合了时间和空间信息，通过将时间流动态嵌入到分层3D场景图中，解决了现有方法在语义感知和可扩展性方面的局限性，提高了自主导航在复杂动态环境中的性能。

### 翻译

在动态环境中的自主导航需要捕捉语义结构和时间演化的空间表示。3D场景图(3DSGs)提供分层多分辨率抽象，编码几何和语义信息，但现有的动态扩展主要关注单个对象或智能体。同时，动态地图(MoDs)模拟典型运动模式和时序规律，但通常基于网格离散化，缺乏语义感知，且不易扩展到大型环境。本文介绍了Aion框架，它将时间流动态直接嵌入到分层3D场景图中，有效整合了时间维度。Aion采用基于图的稀疏MoD表示来捕获任意时间间隔的运动流，并将其附加到场景图中的导航节点上，产生更可解释和可扩展的预测，从而改善复杂动态环境中的规划和交互。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文旨在解决3D场景图缺乏时间维度的问题，使其能够捕捉和预测动态环境中的运动模式。这个问题在自主导航中至关重要，因为机器人需要理解空间结构随时间的演变，才能安全高效地在人类环境中导航，预测人类移动并避免碰撞。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者结合了3D场景图(3DSGs)的分层语义结构和动态地图(MoDs)的时间建模能力，思考如何将两者优势互补。他们借鉴了3DSGs的几何语义表示和MoDs的运动模式捕捉，同时扩展了FreMEn模型用于时间周期性预测，并参考了Hydra系统实现实时集成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将时间流动动力学直接整合到分层的3D场景图中，创建4D时空表示。实现流程包括：1)构建分层的3D场景图结构；2)为导航节点维护方向直方图进行时空建模；3)使用稀疏空间哈希实现可扩展的时间建模；4)通过全局时间模型架构预测运动趋势；5)采用时间所有权转移机制确保长期一致性；6)通过异步处理实现实时集成。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于图的动态地图，在有意义的位置而非任意网格上进行时间建模；2)动态拓扑时间建模，通过位置不变索引保持一致性；3)3DSG集成和流动预测，为导航节点提供直接的时间信息。相比之前工作，Aion不仅捕捉个体对象运动，还关注集体活动模式，利用语义结构而非统一网格，并解决了动态图结构上的时间一致性问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Aion通过将时间流动动力学整合到分层3D场景图中，创建了首个语义感知的4D时空表示，使机器人能够更有效地预测和导航复杂动态环境中的活动模式。'}


### 论文摘要

Autonomous navigation in dynamic environments requires spatial representations that capture both semantic structure and temporal evolution. 3D Scene Graphs (3DSGs) provide hierarchical multi-resolution abstractions that encode geometry and semantics, but existing extensions toward dynamics largely focus on individual objects or agents. In parallel, Maps of Dynamics (MoDs) model typical motion patterns and temporal regularities, yet are usually tied to grid-based discretizations that lack semantic awareness and do not scale well to large environments. In this paper we introduce Aion, a framework that embeds temporal flow dynamics directly within a hierarchical 3DSG, effectively incorporating the temporal dimension. Aion employs a graph-based sparse MoD representation to capture motion flows over arbitrary time intervals and attaches them to navigational nodes in the scene graph, yielding more interpretable and scalable predictions that improve planning and interaction in complex dynamic environments.

---

## 41. Cross-Level Sensor Fusion with Object Lists via Transformer for 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.12884v1](http://arxiv.org/abs/2512.12884v1)

**作者:** Xiangzhong Liu, Jiajie Zhang, Hao Shen

**发布时间:** 2025-12-14

**DOI:** 10.1109/IV64158.2025.11097627

**备注:** 6 pages, 3 figures, accepted at IV2025

### GPT解析

### 总结

该研究提出了一种基于Transformer的端到端跨层融合方法，将智能传感器提供的对象列表信息与原始摄像头图像结合用于3D目标检测，显著提升了性能并展示了良好的泛化能力。

### 背景

在汽车传感器融合系统中，智能传感器和V2X模块被广泛使用，但这些系统提供的数据通常是处理后的对象列表，而非传统传感器的原始数据。传统方法需要单独处理其他原始数据然后在对象级别进行融合。

### 目的

提出一种端到端的跨层融合概念，将高度抽象的对象列表信息与原始摄像头图像结合用于3D目标检测，避免传统方法的复杂处理流程。

### 方法

使用Transformer架构，将对象列表作为去噪查询输入，与可学习查询一起通过特征聚合过程传播；从对象列表的位置和尺寸先验中导出可变形高斯掩码并集成到Transformer解码器中；提出通过模拟状态噪声和假阳性假阴性从真实边界框生成伪对象列表的方法。

### 主要发现

作为首个进行跨层融合的工作，在nuScenes数据集上相比基于视觉的基线方法显示出显著的性能提升；方法能够在不同噪声级别的模拟对象列表和真实检测器上展示其泛化能力。

### 结论

该跨层融合方法有效结合了抽象对象列表信息和原始图像数据，提高了3D目标检测的性能，为汽车传感器融合提供了新的解决方案。

### 翻译

在汽车传感器融合系统中，智能传感器和车联网(V2X)模块被广泛使用。这些系统的传感器数据通常仅以处理后的对象列表形式提供，而非传统传感器的原始传感器数据。我们提出了一种基于Transformer的端到端跨层融合概念，将高度抽象的对象列表信息与原始摄像头图像结合用于3D目标检测，而不是像传统方法那样单独处理其他原始数据然后在对象级别进行融合。对象列表作为去噪查询输入到Transformer中，并与可学习查询一起通过后续特征聚合过程传播。此外，从对象列表的位置和尺寸维度先验中导出的可变形高斯掩码被显式集成到Transformer解码器中，这引导注意力集中在感兴趣的目标区域并加速模型训练收敛。进一步地，由于没有包含对象列表作为独立模态的公共数据集，我们提出了一种通过模拟状态噪声和假阳性假阴性从真实边界框生成伪对象列表的方法。作为首个进行跨层融合的工作，我们的方法在nuScenes数据集上相比基于视觉的基线方法显示出显著的性能提升，并在不同噪声级别的模拟对象列表和真实检测器上展示了其泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决如何在3D目标检测中融合不同抽象级别的传感器数据，特别是将高抽象级别的'对象列表'与低抽象级别的原始相机图像数据进行融合。这个问题在现实中很重要，因为现代自动驾驶系统中的智能传感器和V2X模块通常只提供处理后的对象列表而非原始数据，有效的跨级别融合可以提高3D目标检测的准确性和鲁棒性，增强自动驾驶系统在动态场景中的感知能力。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了不同级别的传感器融合方法的局限性，认识到在自动驾驶场景中往往只能获得高级别的对象列表。他们选择基于Transformer的方法，特别是借鉴了DETR的查询机制和PETRv2框架作为基础。具体设计中，作者采用了查询去噪(QDN)机制受DN-DETR启发，显式注意力引导借鉴了SMCA方法，并开发了伪对象列表生成(POLG)模块来处理数据缺失问题。这些现有技术被创新性地组合和改进，以实现跨级别融合这一新概念。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过端到端的Transformer框架，将高抽象级别的对象列表信息与低抽象级别的原始相机图像数据进行融合，利用对象列表作为'去噪查询'并配合可变形高斯掩码来引导注意力机制。整体流程包括：1)处理多视角相机图像和对象列表输入；2)提取图像特征和生成3D位置感知特征；3)将对象列表转换为去噪查询并与检测查询一起输入Transformer；4)利用对象列表信息创建高斯掩码引导交叉注意力；5)通过检测头生成最终的目标边界框；6)使用伪对象列表进行训练和评估。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次提出并实现跨级别传感器融合概念；2)创新的查询去噪(QDN)机制融合对象列表；3)显式注意力引导通过高斯掩码关注目标区域；4)伪对象列表生成(POLG)模块解决数据缺失问题。相比之前的工作，本文实现了不同抽象级别的端到端融合，而非同一级别的融合；采用基于学习的端到端方法替代传统贝叶斯方法；专门针对稀疏对象列表设计了融合策略，而非处理点云等密集数据；利用查询架构更适合处理结构化的对象信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次提出了跨级别传感器融合的概念，通过创新的查询去噪和显式注意力引导机制，实现了高抽象级别对象列表与低级别原始相机图像的端到端融合，显著提升了3D目标检测的性能。'}


### 论文摘要

In automotive sensor fusion systems, smart sensors and Vehicle-to-Everything (V2X) modules are commonly utilized. Sensor data from these systems are typically available only as processed object lists rather than raw sensor data from traditional sensors. Instead of processing other raw data separately and then fusing them at the object level, we propose an end-to-end cross-level fusion concept with Transformer, which integrates highly abstract object list information with raw camera images for 3D object detection. Object lists are fed into a Transformer as denoising queries and propagated together with learnable queries through the latter feature aggregation process. Additionally, a deformable Gaussian mask, derived from the positional and size dimensional priors from the object lists, is explicitly integrated into the Transformer decoder. This directs attention toward the target area of interest and accelerates model training convergence. Furthermore, as there is no public dataset containing object lists as a standalone modality, we propose an approach to generate pseudo object lists from ground-truth bounding boxes by simulating state noise and false positives and negatives. As the first work to conduct cross-level fusion, our approach shows substantial performance improvements over the vision-based baseline on the nuScenes dataset. It demonstrates its generalization capability over diverse noise levels of simulated object lists and real detectors.

---

## 42. A Domain-Adapted Lightweight Ensemble for Resource-Efficient Few-Shot Plant Disease Classification

**论文链接:** [http://arxiv.org/abs/2512.13428v1](http://arxiv.org/abs/2512.13428v1)

**作者:** Anika Islam, Tasfia Tahsin, Zaarin Anjum, Md. Bakhtiar Hasan, Md. Hasanul Kabir

**发布时间:** 2025-12-15

### GPT解析

### 总结

提出了一种轻量级高效的植物叶片疾病识别框架，结合领域适应的MobileNetV2和MobileNetV3模型作为特征提取器，使用特征融合技术和带有注意力机制的双向LSTM分类器，在少样本学习场景下取得了接近SOTA的性能，同时保持模型轻量且适合移动设备。

### 背景

植物叶片疾病的准确及时识别对可持续农业至关重要，但大多数深度学习方法依赖于大型标注数据集和计算密集型模型，不适合数据稀缺和资源有限的环境。

### 目的

开发一种少样本学习方法，在轻量级高效的框架中解决植物叶片疾病识别的挑战，使其适合数据稀缺和资源有限的环境。

### 方法

结合领域适应的MobileNetV2和MobileNetV3模型作为特征提取器，使用特征融合技术生成鲁棒特征表示，并通过带有注意力机制的双向LSTM分类器捕获序列依赖性并关注最相关特征，实现复杂环境下的最优分类性能。

### 主要发现

在PlantVillage数据集上，15-shot时达到98.23%±0.33%的准确率，接近SOTA基准；在真实世界条件下(Dhan Shomadhan数据集)15-shot时达到69.28%±1.49%；在六种疾病的PlantVillage子集上以15-shot学习达到99.72%，超越之前96.0%的SOTA；模型大小约40MB，推理复杂度约1.12 GFLOPs。

### 结论

该工作为数据稀缺地区的精确植物疾病诊断建立了可扩展、移动就绪的基础，能够在资源有限的环境中实现高效的植物疾病识别。

### 翻译

准确及时的植物叶片疾病识别对弹性可持续农业至关重要，但大多数深度学习方法依赖于大型标注数据集和计算密集型模型，不适合数据稀缺和资源受限环境。为解决这些挑战，我们提出了一种少样本学习方法，在轻量级高效框架中结合领域适应的MobileNetV2和MobileNetV3模型作为特征提取器，以及特征融合技术来生成鲁棒特征表示。对于分类任务，融合特征通过带有注意力机制增强的双向LSTM分类器，以捕获序列依赖性并关注最相关特征，从而即使在具有噪声或杂乱背景的复杂真实世界环境中也能实现最佳分类性能。所提出的框架在多种实验设置中进行了评估，包括实验室控制和野外采集的数据集。在PlantVillage数据集的番茄叶片疾病上，它从1到15 shot场景中持续提高性能，在15 shot时达到98.23±0.33%，接近Transductive LSTM with attention实现的99.98% SOTA基准，同时保持轻量级和移动友好。在Dhan Shomadhan数据集的野外图像真实世界条件下，它保持鲁棒性能，在15-shot时达到69.28±1.49%，并展现出对复杂背景的强韧性。值得注意的是，它在PlantVillage的六种疾病上也超过了之前96.0%的SOTA准确率，仅用15-shot学习就达到了99.72%。凭借约40MB的紧凑模型大小和约1.12 GFLOPs的推理复杂度，这项工作为数据稀缺地区精确植物疾病诊断建立了可扩展、移动就绪的基础。


### 论文摘要

Accurate and timely identification of plant leaf diseases is essential for resilient and sustainable agriculture, yet most deep learning approaches rely on large annotated datasets and computationally intensive models that are unsuitable for data-scarce and resource-constrained environments. To address these challenges we present a few-shot learning approach within a lightweight yet efficient framework that combines domain-adapted MobileNetV2 and MobileNetV3 models as feature extractors, along with a feature fusion technique to generate robust feature representation. For the classification task, the fused features are passed through a Bi-LSTM classifier enhanced with attention mechanisms to capture sequential dependencies and focus on the most relevant features, thereby achieving optimal classification performance even in complex, real-world environments with noisy or cluttered backgrounds. The proposed framework was evaluated across multiple experimental setups, including both laboratory-controlled and field-captured datasets. On tomato leaf diseases from the PlantVillage dataset, it consistently improved performance across 1 to 15 shot scenarios, reaching 98.23+-0.33% at 15 shot, closely approaching the 99.98% SOTA benchmark achieved by a Transductive LSTM with attention, while remaining lightweight and mobile-friendly. Under real-world conditions using field images from the Dhan Shomadhan dataset, it maintained robust performance, reaching 69.28+-1.49% at 15-shot and demonstrating strong resilience to complex backgrounds. Notably, it also outperformed the previous SOTA accuracy of 96.0% on six diseases from PlantVillage, achieving 99.72% with only 15-shot learning. With a compact model size of approximately 40 MB and inference complexity of approximately 1.12 GFLOPs, this work establishes a scalable, mobile-ready foundation for precise plant disease diagnostics in data-scarce regions.

---

## 43. Adapting Multimodal Foundation Models for Few-Shot Learning: A Comprehensive Study on Contrastive Captioners

**论文链接:** [http://arxiv.org/abs/2512.12824v1](http://arxiv.org/abs/2512.12824v1)

**作者:** N. K. B. M. P. K. B. Narasinghe, Uthayasanker Thayasivam

**发布时间:** 2025-12-14

**备注:** 9 pages, 3 figures. Accepted to VISAPP 2026

### GPT解析

### 总结

该论文研究了CoCa模型在少样本图像分类任务上的适应能力，系统评估了从训练自由混合原型到LoRA深度参数适应的策略，发现了数据增强对不同方法的不同影响，以及监督对比损失的优越性。

### 背景

大规模多模态基础模型如CoCa通过统一对比对齐与生成字幕化取得了最先进结果，但其对极端数据稀缺情况的适应能力尚未被充分探索，现有研究主要关注双编码器架构如CLIP，而对CoCa独特潜在空间的研究存在空白。

### 目的

研究CoCa视觉主干在少样本图像分类任务上的适应方法，评估不同参数高效微调策略的性能，并提供针对数据稀缺的实证参考设置。

### 方法

进行系统实证研究，评估从无需训练的混合原型到通过低秩适应（LoRA）进行深度参数适应的层次化策略，研究数据增强的影响，以及结合监督对比（SupCon）损失的混合目标性能。

### 主要发现

1) 存在'增强发散'现象：强数据增强在低样本设置中降低线性探测性能但对稳定LoRA微调至关重要；2) 结合SupCon损失的混合目标在不同样本数量下都优于标准交叉熵；3) 表征了训练配置对数据稀缺的敏感性。

### 结论

提供了正则化、秩和采样策略的实证参考设置，有助于高效适应生成-对比基础模型。

### 翻译

大规模多模态基础模型，特别是对比字幕生成器（CoCa），通过统一对比对齐与生成字幕化实现了最先进的结果。虽然零样本迁移能力已有充分记录，但这些生成-对比混合模型对下游任务的适应，尤其是在极端数据稀缺（少样本学习）情况下的适应，仍未被充分探索。现有文献主要关注双编码器架构如CLIP，留下了关于CoCa独特潜在空间如何响应参数高效微调（PEFT）的研究空白。本文提出了对CoCa视觉主干进行少样本图像分类适应的综合实证研究。我们系统评估了从无需训练的混合原型到通过低秩适应（LoRA）进行深度参数适应的策略层次。首先，我们确定了'增强发散'：虽然强数据增强在低样本设置中降低了线性探测性能，但它对于稳定LoRA微调至关重要。我们还证明，在不同样本数量下，结合监督对比（SupCon）损失的混合目标比标准交叉熵始终带来性能提升。关键地，我们表征了训练配置对数据稀缺的敏感性，提供了正则化、秩和采样策略的实证参考设置，以促进生成-对比基础模型的高效适应。


### 论文摘要

Large-scale multimodal foundation models, particularly Contrastive Captioners (CoCa), have achieved state-of-the-art results by unifying contrastive alignment with generative captioning. While zero-shot transfer capabilities are well-documented, the adaptation of these generative-contrastive hybrids to downstream tasks with extreme data scarcity (few-shot learning) remains under-explored. Existing literature predominantly focuses on dual-encoder architectures like CLIP, leaving a gap in understanding how CoCa's distinct latent space responds to parameter-efficient fine-tuning (PEFT). This paper presents a comprehensive empirical study on adapting the CoCa visual backbone for few-shot image classification. We systematically evaluate a hierarchy of strategies, ranging from training-free hybrid prototyping to deep parameter adaptation via Low-Rank Adaptation (LoRA). First, we identify an "augmentation divergence": while strong data augmentation degrades the performance of linear probing in low-shot settings, it is essential for stabilizing LoRA fine-tuning. We also demonstrate that hybrid objectives incorporating Supervised Contrastive (SupCon) loss yield consistent performance improvements over standard Cross-Entropy across varying shot counts. Crucially, we characterize the sensitivity of training configurations to data scarcity, providing empirical reference settings for scaling regularization, rank, and sampling strategies to facilitate the efficient adaptation of generative-contrastive foundation models.

---

## 44. MetaTPT: Meta Test-time Prompt Tuning for Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.12268v1](http://arxiv.org/abs/2512.12268v1)

**作者:** Yuqing Lei, Yingjun Du, Yawen Huang, Xiantong Zhen, Ling Shao

**发布时间:** 2025-12-13

**备注:** NeurIPS 2025 Workshop

### GPT解析

### 总结

本文提出了Meta Test-Time Prompt Tuning (MetaTPT)，一种元学习框架，通过自监督辅助任务指导测试时提示调整，以改进视觉语言模型在域偏移下的测试时适应能力。

### 背景

Vision-language models (VLMs)如CLIP虽然表现出强大的零样本泛化能力，但对测试时的域偏移仍然敏感。现有的测试时提示调整(TPT)通过固定增强来缓解这个问题，但在更具挑战性的场景中可能表现不佳。

### 目的

提出一种元学习框架，学习自监督辅助任务来指导测试时提示调整，以解决域偏移问题。

### 方法

MetaTPT采用双循环优化范式：内循环学习自监督任务生成信息丰富的视图，外循环通过在这些视图间保持一致性执行提示调整。辅助任务动态学习每个样本的参数化增强，能够捕获目标域中本质特征的更具表现力的变换。

### 主要发现

通过将增强学习与提示调整相结合，MetaTPT改进了域偏移下的测试时适应能力，在领域泛化和跨数据集基准测试上达到最先进的性能。

### 结论

大量实验表明，MetaTPT在领域泛化和跨数据集基准测试上实现了最先进的性能，有效解决了视觉语言模型在域偏移下的测试时适应问题。

### 翻译

视觉语言模型（如CLIP）展现出强大的零样本泛化能力，但在测试时对域偏移仍然敏感。测试时提示调整（TPT）通过固定增强来缓解这一问题，但在更具挑战性的场景中可能表现不佳。在这项工作中，我们提出了元测试时提示调整（MetaTPT），这是一个元学习框架，通过学习自监督辅助任务来指导测试时提示调整。辅助任务为每个样本动态学习参数化增强，能够捕获目标域中本质特征的更具表现力的变换。MetaTPT采用双循环优化范式：内循环学习自监督任务以生成信息丰富的视图，而外循环通过在这些视图间保持一致性来执行提示调整。通过将增强学习与提示调整相结合，MetaTPT改进了域偏移下的测试时适应能力。大量实验表明，MetaTPT在领域泛化和跨数据集基准测试上实现了最先进的性能。


### 论文摘要

Vision-language models (VLMs) such as CLIP exhibit strong zero-shot generalization but remain sensitive to domain shifts at test time. Test-time prompt tuning (TPT) mitigates this issue by adapting prompts with fixed augmentations, which may falter in more challenging settings. In this work, we propose Meta Test-Time Prompt Tuning (MetaTPT), a meta-learning framework that learns a self-supervised auxiliary task to guide test-time prompt tuning. The auxiliary task dynamically learns parameterized augmentations for each sample, enabling more expressive transformations that capture essential features in target domains. MetaTPT adopts a dual-loop optimization paradigm: an inner loop learns a self-supervised task that generates informative views, while the outer loop performs prompt tuning by enforcing consistency across these views. By coupling augmentation learning with prompt tuning, MetaTPT improves test-time adaptation under domain shifts. Extensive experiments demonstrate that MetaTPT achieves state-of-the-art performance on domain generalization and cross-dataset benchmarks.

---

## 45. TA-KAND: Two-stage Attention Triple Enhancement and U-KAN based Diffusion For Few-shot Knowledge Graph Completion

**论文链接:** [http://arxiv.org/abs/2512.12182v1](http://arxiv.org/abs/2512.12182v1)

**作者:** Xinyu Gao

**发布时间:** 2025-12-13

### GPT解析

### 总结

本文提出了一种基于生成表示的少样本知识图谱补全框架，结合了两阶段注意力三元组增强器和U-KAN扩散模型，在公共数据集上取得了最先进结果。

### 背景

知识图谱因其简洁高效的三元组结构已被广泛应用于智能问答和推荐系统等领域。然而，现实世界数据的异构性和多面性导致关系分布呈现长尾特性，使得在有限样本下补全缺失事实变得至关重要。

### 目的

重新审视知识图谱补全问题，从生成表示角度出发，提出一种能有效利用有限样本进行知识图谱补全的方法。

### 方法

提出一种少样本知识图谱补全框架，整合了两阶段注意力三元组增强器和基于U-KAN的扩散模型。

### 主要发现

在两个公共数据集上的大量实验表明，该方法取得了新的最先进结果。

### 结论

通过结合两阶段注意力三元组增强器和U-KAN扩散模型，该方法有效解决了关系分布长尾情况下的有限样本知识图谱补全问题，性能优于现有方法。

### 翻译

知识图谱（KGs）凭借其简洁高效的三元组结构，已被广泛应用于智能问答、推荐系统等领域。然而，现实世界数据的异构性和多面性不可避免地导致关系分布呈现长尾特性，这使得在有限样本下补全缺失事实变得至关重要。以往研究主要基于度量匹配或元学习，但它们要么未能充分利用图中的邻域信息，要么忽视了对比信号的分布特征。本文我们从生成表示的角度重新审视这一问题，并提出了一种结合两阶段注意力三元组增强器和基于U-KAN的扩散模型的少样本知识图谱补全框架。在两个公共数据集上的大量实验表明，我们的方法取得了最新的最先进结果。


### 论文摘要

Knowledge Graphs (KGs), thanks to their concise and efficient triple-based structure, have been widely applied in intelligent question answering, recommender systems and other domains. However, the heterogeneous and multifaceted nature of real-world data inevitably renders the distribution of relations long-tailed, making it crucial to complete missing facts with limited samples. Previous studies mainly based on metric matching or meta learning, yet they either fail to fully exploit neighborhood information in graph or overlook the distributional characteristics of contrastive signals. In this paper, we re-examine the problem from a perspective of generative representation and propose a few-shot knowledge graph completion framework that integrates two-stage attention triple enhancer with U-KAN based diffusion model. Extensive experiments on two public datasets show that our method achieve new state-of-the-art results.

---

## 46. Floorplan2Guide: LLM-Guided Floorplan Parsing for BLV Indoor Navigation

**论文链接:** [http://arxiv.org/abs/2512.12177v1](http://arxiv.org/abs/2512.12177v1)

**作者:** Aydin Ayanzadeh, Tim Oates

**发布时间:** 2025-12-13

**备注:** Accepted for publication in the proceedings of the IEEE International Conference on Big Data (IEEE BigData 2025)

### GPT解析

### 总结

Floorplan2Guide是一种创新的室内导航解决方案，通过将平面图转换为知识图谱和使用大语言模型，减少了手动预处理需求，提高了视障人士在室内环境中的导航能力。

### 背景

室内导航对视障人士来说仍然是一个重大挑战。当前解决方案主要基于基础设施的系统，限制了在动态环境中安全导航的能力。

### 目的

提出一种新的导航方法，利用基础模型将平面图转换为可导航的知识图谱，并生成人类可读的导航指令。

### 方法

提出了Floorplan2Guide方法，该方法利用基础模型将平面图转换为可导航的知识图谱，并集成大语言模型（LLM）从建筑布局中提取空间信息，减少了早期平面图解析方法所需的手动预处理。系统采用基于图的空间结构和上下文学习来提高导航性能。

### 主要发现

少样本学习在模拟和真实世界评估中比零样本学习提高了导航准确性。Claude 3.7 Sonnet在评估的模型中达到了最高准确率，在5-shot提示下分别为：短路线92.31%，中路线76.92%，长路线61.54%。基于图的空间结构的成功率比直接视觉推理高15.4%。

### 结论

图形表示和上下文学习使解决方案对视障人士（BLV用户）的室内导航更加精确。

### 翻译

室内导航对视障人士来说仍然是一个重大挑战。目前的解决方案主要依赖于基于基础设施的系统，这限制了它们在动态环境中安全导航的能力。我们提出了一种新颖的导航方法，利用基础模型将平面图转换为可导航的知识图谱，并生成人类可读的导航指令。Floorplan2Guide集成了大语言模型（LLM）从建筑布局中提取空间信息，减少了早期平面图解析方法所需的手动预处理。实验结果表明，与在模拟和真实世界评估中的零样本学习相比，少样本学习提高了导航准确性。在评估的模型中，Claude 3.7 Sonnet在MP-1平面图的5-shot提示下分别达到了短路线92.31%、中路线76.92%和长路线61.54%的最高准确率。所有模型中，基于图的空间结构的成功率比直接视觉推理高15.4%，这证实了图形表示和上下文学习提高了导航性能，并使我们的解决方案对盲人和低视力（BLV）用户的室内导航更加精确。


### 论文摘要

Indoor navigation remains a critical challenge for people with visual impairments. The current solutions mainly rely on infrastructure-based systems, which limit their ability to navigate safely in dynamic environments. We propose a novel navigation approach that utilizes a foundation model to transform floor plans into navigable knowledge graphs and generate human-readable navigation instructions. Floorplan2Guide integrates a large language model (LLM) to extract spatial information from architectural layouts, reducing the manual preprocessing required by earlier floorplan parsing methods. Experimental results indicate that few-shot learning improves navigation accuracy in comparison to zero-shot learning on simulated and real-world evaluations. Claude 3.7 Sonnet achieves the highest accuracy among the evaluated models, with 92.31%, 76.92%, and 61.54% on the short, medium, and long routes, respectively, under 5-shot prompting of the MP-1 floor plan. The success rate of graph-based spatial structure is 15.4% higher than that of direct visual reasoning among all models, which confirms that graphical representation and in-context learning enhance navigation performance and make our solution more precise for indoor navigation of Blind and Low Vision (BLV) users.

---

## 47. MicroPhaseNO: Adapting an Earthquake-Trained Phase Neural Operator for Microseismic Phase Picking

**论文链接:** [http://arxiv.org/abs/2512.13197v1](http://arxiv.org/abs/2512.13197v1)

**作者:** Ayrat Abdullin, Umair bin Waheed, Leo Eisner, Naveed Iqbal

**发布时间:** 2025-12-15

**备注:** Submitted to Pure and Applied Geophysics

### GPT解析

### 总结

本研究展示了如何通过迁移学习将Phase Neural Operator (PhaseNO)网络范围地震相拾取器适应微地震监测，使用预训练模型并在少量微地震数据上微调，显著提高了拾取性能。

### 背景

地震相拾取是微地震监测和地下成像的关键技术，但传统手动处理不适用于实时应用或大型阵列。基于深度学习的拾取器虽可自动化，但通常针对高信噪比、长时间网络优化，难以处理微地震数据集的挑战。

### 目的

展示如何将PhaseNO网络范围地震相拾取器通过迁移学习适应微地震监测任务。

### 方法

从在57,000多个三分量地震和噪声记录上预训练的PhaseNO模型开始，使用仅200个来自水力压裂环境中诱发事件的标记地震图和噪声地震图对模型进行微调，保留从丰富地震数据中学到的时空表示，同时适应微地震相的特征和标记约定。

### 主要发现

与原始PhaseNO和传统工作流程相比，调整后的模型将F1分数和准确率提高了高达30%，显著减少了系统时间偏差和拾取不确定性。

### 结论

这种适应方法依赖于小型、特定于校准的数据集，可轻松转移到其他微地震任务，相关代码将在GitHub上公开。

### 翻译

地震相拾取经常用于微地震监测和地下成像。传统手动处理不适用于实时应用或大型阵列。基于深度学习的拾取器在大型地震目录上训练，提供了自动化的替代方案。然而，它们通常针对高信噪比、长时间网络进行优化，难以处理微地震数据集的挑战，这些数据集是为有限时间而构建的，且没有预先检测到的地震活动。在本研究中，我们展示了如何通过网络范围的地震相拾取器Phase Neural Operator (PhaseNO)使用迁移学习适应微地震监测。从在57,000多个三分量地震和噪声记录上预训练的PhaseNO模型开始，我们仅使用来自水力压裂环境中诱发事件的200个标记地震图和噪声地震图对模型进行微调。微调后的模型保留了从丰富地震数据中学到的丰富时空表示，同时适应了微地震相的特征和标记约定，这些相通常在峰值或谷值处拾取，而不是在起始点处。我们在三个具有不同网络几何形状和采集参数的真实世界微地震数据集上评估性能。与原始PhaseNO和传统工作流程相比，调整后的模型将F1分数和准确率提高了高达30%，并显著减少了系统时间偏差和拾取不确定性。由于适应依赖于小型、特定于活动校准的数据集，该方法可轻松转移到其他微地震任务，其中公共地震数据和预训练模型是可访问的。相关代码将在https://github.com/ayratabd/MicroPhaseNO上公开。


### 论文摘要

Seismic phase picking is very often used for microseismic monitoring and subsurface imaging. Traditional manual processing is not feasible for either real-time applications or large arrays. Deep learning-based pickers trained on large earthquake catalogs offer an automated alternative. However, they are typically optimized for high signal-to-noise, long-duration networks and struggle with the challenges presented by microseismic datasets, which are purpose-built for limited time without previously detected seismicity. In this study, we demonstrate how a network-wide earthquake phase picker, the Phase Neural Operator (PhaseNO), can be adapted to microseismic monitoring using transfer learning. Starting from a PhaseNO model pre-trained on more than 57,000 three-component earthquake and noise records, we fine-tune the model using only 200 labeled and noise seismograms from induced events in hydraulic-fracturing settings. The fine-tuned model thus preserves the rich spatio-temporal representation learned from abundant earthquake data, while adapting to the characteristics and labeling conventions of microseismic phases, which are often picked on peaks or troughs rather than onsets. We evaluate performance on three distinct real-world microseismic datasets with different network geometries and acquisition parameters. Compared to the original PhaseNO and a conventional workflow, the adapted model increases F1 score and accuracy by up to 30%, and strongly reduces systematic timing bias and pick uncertainty. Because the adaptation relies on a small, campaign-specific calibration set, the approach is readily transferable to other microseismic tasks where public earthquake data and pre-trained models are accessible. The associated code will be released openly at https://github.com/ayratabd/MicroPhaseNO.

---

## 48. Multi-fidelity aerodynamic data fusion by autoencoder transfer learning

**论文链接:** [http://arxiv.org/abs/2512.13069v1](http://arxiv.org/abs/2512.13069v1)

**作者:** Javier Nieto-Centenero, Esther Andrés, Rodrigo Castellanos

**发布时间:** 2025-12-15

**备注:** 29 pages, 13 figures

### GPT解析

### 总结

本文提出了一种多保真度深度学习框架，结合自编码器迁移学习和多分割保形预测策略，在数据极度稀缺情况下实现高精度空气动力学预测并量化不确定性。

### 背景

高精度空气动力学预测通常依赖高保真度模拟，但其计算成本极高，限制了在数据驱动建模中的应用。这促使研究人员开发多保真度策略，利用低成本的低保真度信息而不牺牲准确性。

### 目的

解决数据极度稀缺情况下的不确定性感知空气动力学数据融合问题，开发一种高效可靠的空气动力学回归解决方案。

### 方法

利用丰富的低保真度数据学习紧凑的潜在物理表示作为冻结知识库，然后使用稀缺的高保真度样本对解码器进行微调，并结合新开发的多分割保形预测策略进行不确定性量化。

### 主要发现

模型成功校正了低保真度偏差，使用极少的高保真度训练数据实现了高精度压力预测；MSCP框架产生了稳健、可操作的不确定性区间，点覆盖率超过95%；该方法在二维NACA翼型和三维跨音速机翼数据库测试中表现良好。

### 结论

通过结合极端数据效率和不确定性量化，为数据稀缺环境中的空气动力学回归提供了可扩展且可靠的解决方案。

### 翻译

准确的空气动力学预测通常依赖于高保真度模拟；然而，其高昂的计算成本严重限制了它们在数据驱动建模中的应用。这一局限性促使了多保真度策略的发展，该策略利用低成本的低保真度信息而不牺牲准确性。为应对这一挑战，本研究提出了一个多保真度深度学习框架，结合基于自编码器的迁移学习和新开发的多分割保形预测策略，在极端数据稀缺情况下实现不确定性感知的空气动力学数据融合。该方法利用丰富的低保真度数据学习紧凑的潜在物理表示，该表示作为冻结知识库，用于随后使用稀缺的高保真度样本进行微调的解码器。在NACA翼型二维和跨音速机翼三维数据库的表面压力分布测试中，模型成功校正了低保真度偏差，并使用极少的高保真度训练数据实现了高精度压力预测。此外，MSCP框架产生了稳健、可操作的不确定性区间，点覆盖率超过95%。通过结合极端数据效率与不确定性量化，本研究为数据稀缺环境中的空气动力学回归提供了可扩展且可靠的解决方案。


### 论文摘要

Accurate aerodynamic prediction often relies on high-fidelity simulations; however, their prohibitive computational costs severely limit their applicability in data-driven modeling. This limitation motivates the development of multi-fidelity strategies that leverage inexpensive low-fidelity information without compromising accuracy. Addressing this challenge, this work presents a multi-fidelity deep learning framework that combines autoencoder-based transfer learning with a newly developed Multi-Split Conformal Prediction (MSCP) strategy to achieve uncertainty-aware aerodynamic data fusion under extreme data scarcity. The methodology leverages abundant Low-Fidelity (LF) data to learn a compact latent physics representation, which acts as a frozen knowledge base for a decoder that is subsequently fine-tuned using scarce HF samples. Tested on surface-pressure distributions for NACA airfoils (2D) and a transonic wing (3D) databases, the model successfully corrects LF deviations and achieves high-accuracy pressure predictions using minimal HF training data. Furthermore, the MSCP framework produces robust, actionable uncertainty bands with pointwise coverage exceeding 95%. By combining extreme data efficiency with uncertainty quantification, this work offers a scalable and reliable solution for aerodynamic regression in data-scarce environments.

---

## 49. Comprehensive Deployment-Oriented Assessment for Cross-Environment Generalization in Deep Learning-Based mmWave Radar Sensing

**论文链接:** [http://arxiv.org/abs/2512.13018v1](http://arxiv.org/abs/2512.13018v1)

**作者:** Tomoya Tanaka, Tomonori Ikeda, Ryo Yonemoto

**发布时间:** 2025-12-15

**备注:** 8 pages, 6 figures. Comprehensive evaluation of preprocessing, data augmentation, and transfer learning for cross-environment generalization in deep learning-based mmWave radar sensing

### GPT解析

### 总结

本研究首次全面评估了空间泛化技术，这些技术对于深度学习射频感应的实际部署至关重要。研究聚焦于使用调频连续波多输入多输出雷达进行室内环境中的人数统计，系统研究了多种方法。

### 背景

空间泛化技术对于深度学习射频感应的实际部署至关重要，研究关注室内环境中使用调频连续波多输入多输出雷达进行人数统计。

### 目的

系统性评估空间泛化技术，探索不同方法在跨环境性能上的表现。

### 方法

研究包括幅度统计预处理（sigmoid加权和阈值归零）、频域滤波、基于自编码器的背景抑制、数据增强策略和迁移学习。实验在两种不同布局的环境中收集结果。

### 主要发现

基于sigmoid的幅度加权方法在跨环境性能上始终表现最佳，与基线方法相比，RMSE和MAE分别降低了50.1%和55.2%。数据增强提供了额外的适度益处，MAE最多提高8.8%。对于大的空间变化，迁移学习至关重要，使用540个目标域样本，RMSE和MAE分别降低了82.1%和91.3%。

### 结论

通过结合深度学习模型、基于幅度的预处理和高效的迁移学习，可以为雷达感应系统建立一个高度实用的方向，使其能够在空间变化下保持准确度。

### 翻译

本研究首次全面评估了空间泛化技术，这些技术对于基于深度学习的射频感应的实际部署至关重要。研究聚焦于使用调频连续波多输入多输出雷达在室内环境中进行人数统计，我们系统性地研究了广泛的方法集，包括基于幅度的统计预处理（sigmoid加权和阈值归零）、频域滤波、基于自编码器的背景抑制、数据增强策略和迁移学习。在两种不同布局的环境中收集的实验结果表明，基于sigmoid的幅度加权始终实现优异的跨环境性能，与基线方法相比，均方根误差和平均绝对误差分别降低了50.1%和55.2%。数据增强提供了额外的适度益处，平均绝对误差最多提高8.8%。相比之下，对于大的空间变化，迁移学习至关重要，使用540个目标域样本，均方根误差和平均绝对误差分别降低了82.1%和91.3%。综合来看，这些发现为开发雷达感应系统建立了一个高度实用的方向，通过将深度学习模型与基于幅度的预处理和高效的迁移学习相结合，使其能够在空间变化下保持准确度。


### 论文摘要

This study presents the first comprehensive evaluation of spatial generalization techniques, which are essential for the practical deployment of deep learning-based radio-frequency (RF) sensing. Focusing on people counting in indoor environments using frequency-modulated continuous-wave (FMCW) multiple-input multiple-output (MIMO) radar, we systematically investigate a broad set of approaches, including amplitude-based statistical preprocessing (sigmoid weighting and threshold zeroing), frequency-domain filtering, autoencoder-based background suppression, data augmentation strategies, and transfer learning. Experimental results collected across two environments with different layouts demonstrate that sigmoid-based amplitude weighting consistently achieves superior cross-environment performance, yielding 50.1% and 55.2% reductions in root-mean-square error (RMSE) and mean absolute error (MAE), respectively, compared with baseline methods. Data augmentation provides additional though modest benefits, with improvements up to 8.8% in MAE. By contrast, transfer learning proves indispensable for large spatial shifts, achieving 82.1% and 91.3% reductions in RMSE and MAE, respectively, with 540 target-domain samples. Taken together, these findings establish a highly practical direction for developing radar sensing systems capable of maintaining robust accuracy under spatial variations by integrating deep learning models with amplitude-based preprocessing and efficient transfer learning.

---

## 50. Unsupervised learning of multiscale switching dynamical system models from multimodal neural data

**论文链接:** [http://arxiv.org/abs/2512.12881v1](http://arxiv.org/abs/2512.12881v1)

**作者:** DongKyu Kim, Han-Lin Hsieh, Maryam M. Shanechi

**发布时间:** 2025-12-14

**备注:** 30 pages, 8 figures

### GPT解析

### 总结

该研究开发了一种新颖的无监督学习算法，用于从多尺度神经观测中学习切换多尺度动力学系统模型，解决了依赖性非平稳性建模的挑战，并展示了在行为解码和神经动力学建模方面的优越性能。

### 背景

神经群体活动常表现出依赖性非平稳性，表现为切换动力学。现有方法主要从单一神经模态学习模型，而实际研究经常记录多种神经模态。此外，训练数据中通常没有可用的依赖性标签，这给学习依赖性切换动力学模型带来了挑战。

### 目的

开发一种无监督学习算法，仅使用多尺度神经观测来学习多尺度切换动力学系统模型，以更准确地建模复杂神经动力学并提高行为解码性能。

### 方法

开发了一种新颖的无监督学习算法，使用多尺度神经观测来学习切换多尺度动力学系统模型的参数。通过模拟和两个不同的实验数据集（包含不同运动任务期间的多模态尖峰-LFP观测）验证该方法。

### 主要发现

切换多尺度动力学系统模型比切换单尺度动力学模型更准确地解码行为，证明了多尺度神经融合的成功。此外，这些模型优于平稳多尺度模型，说明了在多模态神经数据中跟踪依赖性非平稳性的重要性。

### 结论

开发的无监督学习框架通过利用多模态记录中的信息并纳入依赖性切换，能够更准确地建模复杂的多尺度神经动力学。这种方法有望提高脑机接口随时间的性能和鲁棒性，并增进我们对行为神经基础的理解。

### 翻译

神经群体活动通常表现为依赖性非平稳性，以切换动力学的形式存在。学习准确的切换动力学系统模型可以揭示行为如何在神经活动中被编码。现有的切换方法主要专注于从单一神经模态（连续高斯信号或离散泊松信号）学习模型。然而，为了测量大脑活动的不同时空尺度，经常同时记录多种神经模态，并且所有这些模态都可以编码行为。此外，训练数据中通常没有可用的依赖性标签，这给学习依赖性切换动力学模型带来了重大挑战。为了应对这些挑战，我们开发了一种新颖的无监督学习算法，仅使用多尺度神经观测来学习切换多尺度动力学系统模型的参数。我们使用模拟和两个不同的实验数据集（包含不同运动任务期间的多模态尖峰-LFP观测）展示了我们的方法。我们发现，我们的切换多尺度动力学系统模型比切换单尺度动力学模型更准确地解码行为，表明了多尺度神经融合的成功。此外，我们的模型优于平稳多尺度模型，说明了在多模态神经数据中跟踪依赖性非平稳性的重要性。开发的无监督学习框架通过利用多模态记录中的信息并纳入依赖性切换，能够更准确地建模复杂的多尺度神经动力学。这种方法有望提高脑机接口随时间的性能和鲁棒性，并增进我们对行为神经基础的理解。


### 论文摘要

Neural population activity often exhibits regime-dependent non-stationarity in the form of switching dynamics. Learning accurate switching dynamical system models can reveal how behavior is encoded in neural activity. Existing switching approaches have primarily focused on learning models from a single neural modality, either continuous Gaussian signals or discrete Poisson signals. However, multiple neural modalities are often recorded simultaneously to measure different spatiotemporal scales of brain activity, and all these modalities can encode behavior. Moreover, regime labels are typically unavailable in training data, posing a significant challenge for learning models of regime-dependent switching dynamics. To address these challenges, we develop a novel unsupervised learning algorithm that learns the parameters of switching multiscale dynamical system models using only multiscale neural observations. We demonstrate our method using both simulations and two distinct experimental datasets with multimodal spike-LFP observations during different motor tasks. We find that our switching multiscale dynamical system models more accurately decode behavior than switching single-scale dynamical models, showing the success of multiscale neural fusion. Further, our models outperform stationary multiscale models, illustrating the importance of tracking regime-dependent non-stationarity in multimodal neural data. The developed unsupervised learning framework enables more accurate modeling of complex multiscale neural dynamics by leveraging information in multimodal recordings while incorporating regime switches. This approach holds promise for improving the performance and robustness of brain-computer interfaces over time and for advancing our understanding of the neural basis of behavior.

---

## 51. TRACER: Transfer Learning based Real-time Adaptation for Clinical Evolving Risk

**论文链接:** [http://arxiv.org/abs/2512.12795v1](http://arxiv.org/abs/2512.12795v1)

**作者:** Mengying Yan, Ziye Tian, Siqi Li, Nan Liu, Benjamin A. Goldstein, Molei Liu, Chuan Hong

**发布时间:** 2025-12-14

### GPT解析

### 总结

TRACER是一种基于迁移学习的框架，用于应对临床决策支持工具中的性能漂移问题，特别是在人口时间变化和混合人群的情况下。

### 背景

基于电子健康记录的临床决策支持工具常常因人口时间变化而出现性能漂移，当临床环境变化最初只影响部分患者时，会导致转变为混合人群。这种病例组合变化通常发生在系统级运营更新或新疾病（如COVID-19）出现之后。

### 目的

开发一种能够识别就诊级别过渡成员身份并适应预测模型的框架，无需完全重新训练，以保持临床决策支持工具的性能。

### 方法

提出了TRACER（基于迁移学习的临床 evolving 风险实时适应）框架，该框架使用迁移学习技术来适应预测模型，而不需要完全重新训练。

### 主要发现

在模拟研究中，TRACER优于在历史或当代数据上训练的静态模型。在预测COVID-19转变后急诊科就诊后住院的真实世界应用中，TRACER提高了区分度和校准度。

### 结论

TRACER提供了一种可扩展的方法，可以在不断变化和异质的临床条件下保持强大的预测性能。

### 翻译

基于电子健康记录构建的临床决策支持工具经常因人口时间变化而出现性能漂移，特别是当临床环境变化最初只影响部分患者时，导致转变为混合人群。这种病例组合变化通常发生在系统级运营更新或新疾病（如COVID-19）出现之后。我们提出了TRACER（基于迁移学习的临床 evolving 风险实时适应）框架，该框架识别就诊级别的过渡成员身份，并使用迁移学习适应预测模型，无需完全重新训练。在模拟研究中，TRACER优于在历史或当代数据上训练的静态模型。在预测COVID-19转变后急诊科就诊后住院的真实世界应用中，TRACER提高了区分度和校准度。TRACER提供了一种可扩展的方法，可以在不断变化和异质的临床条件下保持强大的预测性能。


### 论文摘要

Clinical decision support tools built on electronic health records often experience performance drift due to temporal population shifts, particularly when changes in the clinical environment initially affect only a subset of patients, resulting in a transition to mixed populations. Such case-mix changes commonly arise following system-level operational updates or the emergence of new diseases, such as COVID-19. We propose TRACER (Transfer Learning-based Real-time Adaptation for Clinical Evolving Risk), a framework that identifies encounter-level transition membership and adapts predictive models using transfer learning without full retraining. In simulation studies, TRACER outperformed static models trained on historical or contemporary data. In a real-world application predicting hospital admission following emergency department visits across the COVID-19 transition, TRACER improved both discrimination and calibration. TRACER provides a scalable approach for maintaining robust predictive performance under evolving and heterogeneous clinical conditions.

---

## 52. Machine Learning Predictive Analytics for Social Media Enabled Women's Economic Empowerment in Pakistan

**论文链接:** [http://arxiv.org/abs/2512.12685v1](http://arxiv.org/abs/2512.12685v1)

**作者:** Maryam Arif, Soban Saeed

**发布时间:** 2025-12-14

### GPT解析

### 总结

本研究探讨了年轻女性赋权与巴基斯坦经济增长之间的关系，重点关注社交媒体如何增强她们的业务并推动经济发展。研究发现社交媒体使用与创业之间存在潜力，但也存在显著差距和多种障碍。

### 背景

研究背景为巴基斯坦父权制社会中年轻女性赋权与经济增长的关系，社交媒体在其中的作用尚未被充分研究。

### 目的

调查社交媒体使用如何增强年轻女性的业务并推动巴基斯坦经济发展，分析社交媒体参与模式与创业预测之间的关系。

### 方法

采用混合研究方法设计，结合线上和线下随机抽样调查51名受访者，并利用社交媒体使用数据(n=1000)和创业数据(n=1092)进行分析。使用无监督学习识别社交媒体参与模式，应用监督模型进行创业预测，逻辑回归表现最佳。

### 主要发现

39.4%的受访者认为社交媒体对经济有积极影响，但只有14%参与创业，存在显著差距。YouTube(66.7%)和WhatsApp(62.7%)是最常用的平台。主要障碍包括网络骚扰、数字素养有限和文化约束。52.9%的受访者不了解支持女性创业者的政府倡议。

### 结论

社交媒体在女性赋权和经济发展中具有潜力，但需要解决数字参与与商业采用之间的差距，以及文化、教育等多重障碍，同时加强政策覆盖和支持。

### 翻译

我们的研究调查了年轻女性赋权与巴基斯坦经济增长之间的相互作用，重点关注社交媒体使用如何增强她们的业务并推动经济发展。我们采用混合研究方法设计，结合线上和线下随机抽样，对51名受访者进行调查。我们还利用了包含社交媒体使用(n=1000)和创业(n=1092)的现有数据集。我们的分析通过无监督学习识别不同的社交媒体参与模式，并应用监督模型进行创业预测，逻辑回归在预测准确性和稳定性方面优于所有其他算法。在社交媒体使用方面，聚类分析显示在K=2时，用户形成紧密的、良好分离的参与群体。结果表明，39.4%的受访者认为社交媒体通过帮助企业增加收入对经济产生积极影响。然而，只有14%的受访者参与创业，突显了数字参与与商业采用之间的显著差距。分析表明，每日社交媒体使用普遍，YouTube(66.7%)和WhatsApp(62.7%)是最常使用的平台。确定的主要障碍是网络骚扰、数字素养有限以及在巴基斯坦这样的父权制社会中的文化约束。此外，52.9%的受访者不知道支持女性创业者的政府倡议，表明政策覆盖有限。


### 论文摘要

Our study investigates the interplay between young women's empowerment and Pakistan's economic growth, focusing on how social media use enhances their businesses and drives economic advancement. We utilize a mixed-methods research design, integrating both online and offline random sampling, for our survey of 51 respondents. We also utilized existing datasets consisting of both social media usage (n = 1000) and entrepreneurship (n = 1092). Our analysis identifies distinct social media engagement patterns via unsupervised learning and applies supervised models for entrepreneurship prediction, with logistic regression outperforming all other algorithms in terms of predictive accuracy and stability. In social media use, the cluster analysis reveals that at K=2, users form tightly packed, well-separated engagement groups. The results indicate that 39.4 percent of respondents believe social media positively impacts the economy by enabling businesses to generate increased revenue. However, only 14 percent of respondents participate in entrepreneurship, highlighting a substantial gap between digital engagement and business adoption. The analysis indicates that daily social media consumption is widespread with YouTube (66.7 percent) and WhatsApp (62.7 percent) being the most frequently used platforms. Key barriers identified are online harassment, limited digital literacy, and cultural constraints in a patriarchal society such as Pakistan. Additionally, 52.9 percent of respondents are unaware of government initiatives supporting women entrepreneurs, indicating limited policy outreach.

---

## 53. GrowTAS: Progressive Expansion from Small to Large Subnets for Efficient ViT Architecture Search

**论文链接:** [http://arxiv.org/abs/2512.12296v1](http://arxiv.org/abs/2512.12296v1)

**作者:** Hyunju Lee, Youngmin Oh, Jeimin Jeon, Donghyeon Baek, Bumsub Ham

**发布时间:** 2025-12-13

**备注:** Accepted to WACV 2026

### GPT解析

### 总结

本文提出了一种名为GrowTAS的渐进式训练框架，用于解决Transformer架构搜索(TAS)中子网络权重共享导致的干扰问题，并通过GrowTAS+进一步优化大型子网络性能，在多个基准测试上证明了方法的有效性。

### 背景

现有的TAS方法通常训练一个包含所有候选架构的超网络，但这些子网络共享相同权重，导致干扰，严重影响小型子网络性能。

### 目的

减少子网络间的干扰，稳定训练过程，提高Transformer架构搜索的效率，自动发现更优的视觉Transformer架构。

### 方法

提出GrowTAS渐进式训练框架，从训练小型子网络开始，逐步纳入更大的子网络；进一步提出GrowTAS+，仅微调部分权重以提高大型子网络性能。

### 主要发现

训练良好的小型子网络可以作为训练大型子网络的良好基础；渐进式训练可以减少子网络间的干扰并稳定训练过程。

### 结论

GrowTAS和GrowTAS+在ImageNet和多个迁移学习基准测试上均优于当前TAS方法，证明了该渐进式训练框架的有效性。

### 翻译

Transformer架构搜索(TAS)旨在自动发现高效的视觉Transformer(ViT)，减少手动设计的需求。现有的TAS方法通常训练一个包含所有候选架构（即子网络）的过参数化网络（即超网络）。然而，所有子网络共享相同的权重集，这导致干扰严重降低了小型子网络的性能。我们发现训练良好的小型子网络可以作为训练大型子网络的良好基础。受此启发，我们提出了一个名为GrowTAS的渐进式训练框架，从训练小型子网络开始，并逐步纳入更大的子网络。这减少了干扰并稳定了训练过程。我们还引入了GrowTAS+，它仅微调部分权重以进一步提高大型子网络的性能。在ImageNet和多个迁移学习基准测试（包括CIFAR-10/100、Flowers、CARS和INAT-19）上的大量实验证明了我们的方法优于当前的TAS方法。


### 论文摘要

Transformer architecture search (TAS) aims to automatically discover efficient vision transformers (ViTs), reducing the need for manual design. Existing TAS methods typically train an over-parameterized network (i.e., a supernet) that encompasses all candidate architectures (i.e., subnets). However, all subnets share the same set of weights, which leads to interference that degrades the smaller subnets severely. We have found that well-trained small subnets can serve as a good foundation for training larger ones. Motivated by this, we propose a progressive training framework, dubbed GrowTAS, that begins with training small subnets and incorporate larger ones gradually. This enables reducing the interference and stabilizing a training process. We also introduce GrowTAS+ that fine-tunes a subset of weights only to further enhance the performance of large subnets. Extensive experiments on ImageNet and several transfer learning benchmarks, including CIFAR-10/100, Flowers, CARS, and INAT-19, demonstrate the effectiveness of our approach over current TAS methods

---

## 54. A Lightweight Transfer Learning-Based State-of-Health Monitoring with Application to Lithium-ion Batteries in Autonomous Air Vehicles

**论文链接:** [http://arxiv.org/abs/2512.08512v2](http://arxiv.org/abs/2512.08512v2)

**作者:** Jiang Liu, Yan Qin, Wei Dai, Chau Yuen

**发布时间:** 2025-12-09

**DOI:** 10.1109/TII.2025.3631012

**备注:** in IEEE Transactions on Industrial Informatics,2025

### GPT解析

### 总结

本文提出了一种基于轻量级迁移学习的锂离子电池健康状态监测方法，称为结构化增量迁移学习(CITL)，解决了传统方法在便携式移动设备上计算资源消耗大的问题，通过半监督学习和迭代添加网络节点提高监测精度。

### 背景

准确快速的锂离子电池健康状态监测对电池供电的便携式移动设备至关重要，但传统迁移学习方法需要大量目标工作条件下的训练数据，且在便携式设备上会消耗大量计算资源，降低工作续航时间。

### 目的

开发一种轻量级的迁移学习方法，适用于便携式移动设备的SOH监测，解决传统方法计算资源消耗大的问题，并利用目标域中的未标记数据提高监测精度。

### 方法

提出基于结构化增量迁移学习(CITL)的轻量级SOH监测方法，利用目标域未标记数据实现半监督迁移学习，通过迭代添加网络节点最小化监测残差，并通过结构风险最小化、传输失配最小化和流形一致性最大化保证跨域学习能力。

### 主要发现

CITL方法在AAV电池数据集上表现优异，比SS-TCA、MMD-LSTM-DA、DDAN、BO-CNN-TL和AS$^3$LSTM分别提高83.73%、61.15%、28.24%、87.70%和57.34%的SOH估计精度，通过均方根误差指标评估。

### 结论

CITL方法有效解决了传统迁移学习方法在便携式移动设备上的计算资源消耗问题，通过半监督学习和结构化增量迁移学习显著提高SOH监测准确性，同时减少计算资源需求，适合在便携式移动设备上使用。

### 翻译

准确且快速的锂离子电池健康状态监测对电池供电的便携式移动设备指示能量信息起着重要作用。为了应对变化的工作条件，迁移学习成为一种有前景的技术，可以利用数据丰富的源工作条件中的知识，显著减少SOH监测所需的训练数据量。然而，当应用于便携式移动设备时，传统的基于迁移学习的SOH监测不可行，因为在迁移学习阶段会消耗大量计算资源并降低工作续航时间。为解决这些挑战，本文提出了一种基于轻量级迁移学习的SOH监测方法，采用结构化增量迁移学习。首先，利用目标域中的未标记数据，提出半监督迁移学习机制，通过迭代添加网络节点以建设性方式最小化监测残差。其次，通过结构风险最小化、传输失配最小化和流形一致性最大化，全面保证了CITL节点参数的跨域学习能力。此外，提供了CITL的收敛分析，理论上保证了迁移学习性能和网络紧凑性的有效性。最后，通过使用从数十次飞行任务收集的真实自主飞行器电池数据集进行的广泛实验验证了所提出的方法。具体而言，使用均方根误差指标评估时，CITL在SOH估计方面分别优于其他方法，提高幅度显著。


### 论文摘要

Accurate and rapid state-of-health (SOH) monitoring plays an important role in indicating energy information for lithium-ion battery-powered portable mobile devices. To confront their variable working conditions, transfer learning (TL) emerges as a promising technique for leveraging knowledge from data-rich source working conditions, significantly reducing the training data required for SOH monitoring from target working conditions. However, traditional TL-based SOH monitoring is infeasible when applied in portable mobile devices since substantial computational resources are consumed during the TL stage and unexpectedly reduce the working endurance. To address these challenges, this paper proposes a lightweight TL-based SOH monitoring approach with constructive incremental transfer learning (CITL). First, taking advantage of the unlabeled data in the target domain, a semi-supervised TL mechanism is proposed to minimize the monitoring residual in a constructive way, through iteratively adding network nodes in the CITL. Second, the cross-domain learning ability of node parameters for CITL is comprehensively guaranteed through structural risk minimization, transfer mismatching minimization, and manifold consistency maximization. Moreover, the convergence analysis of the CITL is given, theoretically guaranteeing the efficacy of TL performance and network compactness. Finally, the proposed approach is verified through extensive experiments with a realistic autonomous air vehicles (AAV) battery dataset collected from dozens of flight missions. Specifically, the CITL outperforms SS-TCA, MMD-LSTM-DA, DDAN, BO-CNN-TL, and AS$^3$LSTM, in SOH estimation by 83.73%, 61.15%, 28.24%, 87.70%, and 57.34%, respectively, as evaluated using the index root mean square error.

---

## 55. Self-Supervised Ultrasound Representation Learning for Renal Anomaly Prediction in Prenatal Imaging

**论文链接:** [http://arxiv.org/abs/2512.13434v1](http://arxiv.org/abs/2512.13434v1)

**作者:** Youssef Megahed, Inok Lee, Robin Ducharme, Kevin Dick, Adrian D. C. Chan, Steven Hawken, Mark C. Walker

**发布时间:** 2025-12-15

**备注:** 14 pages, 8 figures, 4 tables

### GPT解析

### 总结

本研究评估了一种自监督超声基础模型在自动胎儿肾脏畸形分类中的性能，结果表明该模型在所有评估指标上均优于传统基线模型，特别是在多类分类任务中表现出显著优势。

### 背景

产前超声是检测肾脏和泌尿道先天性畸形的基础，但诊断受到操作者依赖性和成像条件不佳的限制。

### 目的

评估一种自监督超声基础模型在自动胎儿肾脏畸形分类中的性能。

### 方法

使用包含969张二维超声图像的精选数据集，对预训练的超声自监督掩码自编码基础模型(USF-MAE)进行微调，用于二元分类和多类分类（正常肾脏、尿路扩张和囊性发育不良肾），并与DenseNet-169卷积基线模型进行比较，使用交叉验证和独立测试集评估模型。

### 主要发现

USF-MAE在所有评估指标上均优于基线模型；验证集上AUC提高约1.87%，F1分数提高7.8%；独立测试集上AUC提高2.32%，F1分数提高4.33%；多类分类设置中改进最大：AUC提高16.28%，F1分数提高46.15%；Score-CAM可视化表明模型预测基于已知的临床相关肾脏结构。

### 结论

超声特定的自监督学习可以生成有用的表示作为下游诊断任务的基础；提出的框架为产前检测肾脏畸形提供了一种稳健、可解释的方法；基础模型在产科影像学中显示出前景。

### 翻译

产前超声是检测肾脏和泌尿道先天性畸形的基础，但诊断受到操作者依赖性和成像条件不佳的限制。我们旨在评估一种自监督超声基础模型在自动胎儿肾脏畸形分类中的性能，使用包含969张二维超声图像的精选数据集。预训练的超声自监督掩码自编码基础模型(USF-MAE)被微调用于二元和多类分类，包括正常肾脏、尿路扩张和囊性发育不良肾。模型使用交叉验证和独立测试集与DenseNet-169卷积基线进行比较。USF-MAE在二元和多类设置的所有评估指标上一致优于基线。在验证集上，USF-MAE的AUC提高了约1.87%，F1分数提高了7.8%；在独立测试集上，AUC提高了2.32%，F1分数提高了4.33%。最大的改进出现在多类设置中，AUC提高了16.28%，F1分数提高了46.15%。为促进模型可解释性，将Score-CAM可视化适配到transformer架构，表明模型预测基于已知的临床相关肾脏结构，包括尿路扩张中的肾盂和囊性发育不良肾中的囊性区域。这些结果表明，超声特定的自监督学习可以生成有用的表示作为下游诊断任务的基础。所提出的框架为支持产前肾脏畸形检测提供了一种稳健、可解释的方法，并展示了基础模型在产科影像学中的前景。


### 论文摘要

Prenatal ultrasound is the cornerstone for detecting congenital anomalies of the kidneys and urinary tract, but diagnosis is limited by operator dependence and suboptimal imaging conditions. We sought to assess the performance of a self-supervised ultrasound foundation model for automated fetal renal anomaly classification using a curated dataset of 969 two-dimensional ultrasound images. A pretrained Ultrasound Self-Supervised Foundation Model with Masked Autoencoding (USF-MAE) was fine-tuned for binary and multi-class classification of normal kidneys, urinary tract dilation, and multicystic dysplastic kidney. Models were compared with a DenseNet-169 convolutional baseline using cross-validation and an independent test set. USF-MAE consistently improved upon the baseline across all evaluation metrics in both binary and multi-class settings. USF-MAE achieved an improvement of about 1.87% (AUC) and 7.8% (F1-score) on the validation set, 2.32% (AUC) and 4.33% (F1-score) on the independent holdout test set. The largest gains were observed in the multi-class setting, where the improvement in AUC was 16.28% and 46.15% in F1-score. To facilitate model interpretability, Score-CAM visualizations were adapted for a transformer architecture and show that model predictions were informed by known, clinically relevant renal structures, including the renal pelvis in urinary tract dilation and cystic regions in multicystic dysplastic kidney. These results show that ultrasound-specific self-supervised learning can generate a useful representation as a foundation for downstream diagnostic tasks. The proposed framework offers a robust, interpretable approach to support the prenatal detection of renal anomalies and demonstrates the promise of foundation models in obstetric imaging.

---

## 56. PvP: Data-Efficient Humanoid Robot Learning with Proprioceptive-Privileged Contrastive Representations

**论文链接:** [http://arxiv.org/abs/2512.13093v1](http://arxiv.org/abs/2512.13093v1)

**作者:** Mingqi Yuan, Tao Yu, Haolin Song, Bo Li, Xin Jin, Hua Chen, Wenjun Zeng

**发布时间:** 2025-12-15

**备注:** 13 pages, 12 figures

### GPT解析

### 总结

该研究提出了一种名为PVP的本体感觉-特权对比学习框架，解决了人形机器人全身控制中强化学习的样本效率问题，并通过开发的SRL4Humanoid框架进行了系统评估，实验证明该方法显著提高了样本效率和最终性能。

### 背景

人形机器人在动态环境中执行复杂任务需要高效且鲁棒的全身控制(WBC)。尽管强化学习(RL)在此领域取得了成功，但由于人形机器人的复杂动力学和部分可观察性，其样本效率低仍然是一个重大挑战。

### 目的

解决强化学习在人形机器人控制中的样本效率问题，提高学习速度和稳定性。

### 方法

提出了PVP(Proprioceptive-Privileged对比学习框架)，利用本体感觉状态和特权状态之间的内在互补性。PVP能够学习紧凑且与任务相关的潜在表示，不需要手工设计的数据增强。同时开发了SRL4Humanoid框架，这是第一个统一且模块化的框架，为人形机器人学习提供了高质量的状态表示学习(SRL)方法实现。

### 主要发现

在LimX Oli机器人上的广泛实验表明，与基线SRL方法相比，PVP显著提高了样本效率和最终性能，特别是在速度跟踪和运动模仿任务中。

### 结论

研究提供了将SRL与RL集成用于人形机器人全身控制的实用见解，为数据高效的人形机器人学习提供了有价值的指导。

### 翻译

实现高效且鲁棒的全身控制(WBC)对于使人形机器人在动态环境中执行复杂任务至关重要。尽管强化学习(RL)在此领域取得了成功，但由于人形机器人的复杂动力学和部分可观察性，其样本效率低仍然是一个重大挑战。为解决这一局限性，我们提出了PVP，一个本体感觉-特权对比学习框架，它利用本体感觉状态和特权状态之间的内在互补性。PVP学习紧凑且与任务相关的潜在表示，不需要手工设计的数据增强，从而实现更快且更稳定的策略学习。为支持系统评估，我们开发了SRL4Humanoid，这是第一个统一且模块化的框架，为人形机器人学习提供了高质量的状态表示学习(SRL)方法实现。在LimX Oli机器人上进行的广泛实验，包括速度跟踪和运动模仿任务，证明与基线SRL方法相比，PVP显著提高了样本效率和最终性能。我们的研究进一步提供了将SRL与RL集成用于人形机器人全身控制的实用见解，为数据高效的人形机器人学习提供了宝贵的指导。


### 论文摘要

Achieving efficient and robust whole-body control (WBC) is essential for enabling humanoid robots to perform complex tasks in dynamic environments. Despite the success of reinforcement learning (RL) in this domain, its sample inefficiency remains a significant challenge due to the intricate dynamics and partial observability of humanoid robots. To address this limitation, we propose PvP, a Proprioceptive-Privileged contrastive learning framework that leverages the intrinsic complementarity between proprioceptive and privileged states. PvP learns compact and task-relevant latent representations without requiring hand-crafted data augmentations, enabling faster and more stable policy learning. To support systematic evaluation, we develop SRL4Humanoid, the first unified and modular framework that provides high-quality implementations of representative state representation learning (SRL) methods for humanoid robot learning. Extensive experiments on the LimX Oli robot across velocity tracking and motion imitation tasks demonstrate that PvP significantly improves sample efficiency and final performance compared to baseline SRL methods. Our study further provides practical insights into integrating SRL with RL for humanoid WBC, offering valuable guidance for data-efficient humanoid robot learning.

---

## 57. Citation importance-aware document representation learning for large-scale science mapping

**论文链接:** [http://arxiv.org/abs/2512.13054v1](http://arxiv.org/abs/2512.13054v1)

**作者:** Zhentao Liang, Nees Jan van Eck, Xuehua Wu, Jin Mao, Gang Li

**发布时间:** 2025-12-15

**DOI:** 10.1016/j.ipm.2025.104557

### GPT解析

### 总结

本研究提出了一种考虑引文重要性的对比学习框架，通过测量引文重要性并整合到对比学习过程中，解决了科学映射中引文异质性问题，有效提高了文档表示质量和科学映射准确性，并在大规模数据集上成功应用。

### 背景

科学映射是科学计量学和情报研究中的重要任务，但面临引文复杂性和异质性的挑战。先前研究虽尝试通过整合引文和语义信息改进文档表示，但忽视了引文的异质性。

### 目的

解决引文异质性问题，提出一个考虑引文重要性的对比学习框架，以改进文档表示和科学映射的准确性。

### 方法

开发基于位置、频率和自引特征的引文重要性可扩展测量方法；将引文重要性整合到对比学习过程中，通过重要性感知采样策略选择低重要性引文作为困难样本；使用SciBERT模型进行微调；在SciDocs和PubMed基准数据集上评估；将模型应用于Web of Science的3300多万篇文档。

### 主要发现

在文档表示质量和科学映射准确性方面均取得一致改进；生成的科学地图准确可视化科学的全球和局部智力结构；揭示了跨学科研究前沿。

### 结论

通过将引文异质性转化为可扩展的计算框架，证明了区分引文重要性可有效改进文档表示和科学映射。

### 翻译

有效的科学映射依赖于高质量的科学文档表示。作为科学计量学和情报研究中的重要任务，科学映射常常面临引文复杂和异质性的挑战。虽然先前研究试图通过整合引文和语义信息来改进文档表示，但引文的异质性常被忽视。为解决这一问题，本研究提出了一个考虑引文重要性的对比学习框架，以优化监督信号。我们首先开发了一种基于位置、频率和自引特征的引文重要性可扩展测量方法。然后通过重要性感知采样策略将引文重要性整合到对比学习过程中，选择低重要性引文作为困难样本。这迫使模型学习能够区分重要性和形式性引文的细粒度表示。为验证所提框架的有效性，我们微调了SciBERT模型，并在SciDocs和PubMed基准数据集上进行了广泛评估。结果表明，在文档表示质量和科学映射准确性方面均取得了一致的改进。此外，我们将训练模型应用于Web of Science中的3300多万篇文档。生成的科学地图准确可视化了科学的全球和局部智力结构，并揭示了跨学科研究前沿。通过将引文异质性转化为可扩展的计算框架，本研究证明了如何通过区分引文重要性来有效改进文档表示和科学映射。


### 论文摘要

Effective science mapping relies on high-quality representations of scientific documents. As an important task in scientometrics and information studies, science mapping is often challenged by the complex and heterogeneous nature of citations. While previous studies have attempted to improve document representations by integrating citation and semantic information, the heterogeneity of citations is often overlooked. To address this problem, this study proposes a citation importance-aware contrastive learning framework that refines the supervisory signal. We first develop a scalable measurement of citation importance based on location, frequency, and self-citation characteristics. Citation importance is then integrated into the contrastive learning process through an importance-aware sampling strategy, which selects low-importance citations as hard negatives. This forces the model to learn finer-grained representations that distinguish between important and perfunctory citations. To validate the effectiveness of the proposed framework, we fine-tune a SciBERT model and perform extensive evaluations on SciDocs and PubMed benchmark datasets. Results show consistent improvements in both document representation quality and science mapping accuracy. Furthermore, we apply the trained model to over 33 million documents from Web of Science. The resulting map of science accurately visualizes the global and local intellectual structure of science and reveals interdisciplinary research fronts. By operationalizing citation heterogeneity into a scalable computational framework, this study demonstrates how differentiating citations by their importance can be effectively leveraged to improve document representation and science mapping.

---

## 58. BLADE: A Behavior-Level Data Augmentation Framework with Dual Fusion Modeling for Multi-Behavior Sequential Recommendation

**论文链接:** [http://arxiv.org/abs/2512.12964v1](http://arxiv.org/abs/2512.12964v1)

**作者:** Yupeng Li, Mingyue Cheng, Yucong Luo, Yitong Zhou, Qingyang Mao, Shijin Wang

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出了一种名为BLADE的多行为序列推荐框架，通过双重项目-行为融合架构和行为级数据增强方法解决用户行为异质性和数据稀疏性问题，在三个真实数据集上证明了其有效性。

### 背景

多行为序列推荐旨在通过建模用户随时间变化的多种类型交互来捕捉用户的动态兴趣，尽管已有研究探索此领域，但推荐性能仍不理想。

### 目的

解决多行为序列推荐中的两个基本挑战：用户行为的异质性和数据稀疏性，从而提升推荐性能。

### 方法

提出BLADE框架，包含双重项目-行为融合架构（在输入和中间层面融入行为信息）和行为级数据增强方法（直接在行为序列上操作，生成多样化增强视图并保持语义一致性），通过对比学习增强表示学习和泛化能力。

### 主要发现

在三个真实数据集上的实验证明了BLADE框架的有效性，能够提升多行为序列推荐性能。

### 结论

BLADE框架通过解决用户行为异质性和数据稀疏性问题，有效提升了多行为序列推荐的性能。

### 翻译

多行为序列推荐旨在通过建模用户随时间变化的多种类型交互来捕捉用户的动态兴趣。尽管已有一些研究探索了这个领域，但推荐性能仍然不理想，主要由于两个基本挑战：用户行为的异质性和数据稀疏性。为解决这些挑战，我们提出了BLADE框架，该框架增强多行为建模同时减轻数据稀疏性。具体而言，为处理行为异质性，我们引入了双重项目-行为融合架构，在输入和中间层面都融入行为信息，从而能够从多个角度进行偏好建模。为减轻数据稀疏性，我们设计了三种行为级数据增强方法，这些方法直接在行为序列上操作，而不是在核心项目序列上。这些方法生成多样化的增强视图，同时保持项目序列的语义一致性。这些增强视图通过对比学习进一步增强了表示学习和泛化能力。在三个真实数据集上的实验证明了我们方法的有效性。


### 论文摘要

Multi-behavior sequential recommendation aims to capture users' dynamic interests by modeling diverse types of user interactions over time. Although several studies have explored this setting, the recommendation performance remains suboptimal, mainly due to two fundamental challenges: the heterogeneity of user behaviors and data sparsity. To address these challenges, we propose BLADE, a framework that enhances multi-behavior modeling while mitigating data sparsity. Specifically, to handle behavior heterogeneity, we introduce a dual item-behavior fusion architecture that incorporates behavior information at both the input and intermediate levels, enabling preference modeling from multiple perspectives. To mitigate data sparsity, we design three behavior-level data augmentation methods that operate directly on behavior sequences rather than core item sequences. These methods generate diverse augmented views while preserving the semantic consistency of item sequences. These augmented views further enhance representation learning and generalization via contrastive learning. Experiments on three real-world datasets demonstrate the effectiveness of our approach.

---

## 59. Predictive Sample Assignment for Semantically Coherent Out-of-Distribution Detection

**论文链接:** [http://arxiv.org/abs/2512.12906v1](http://arxiv.org/abs/2512.12906v1)

**作者:** Zhimao Peng, Enguang Wang, Xialei Liu, Ming-Ming Cheng

**发布时间:** 2025-12-15

**备注:** Accepted by TCSVT2024

### GPT解析

### 总结

本文提出了一种基于预测样本分配(PSA)的语义连贯分布外检测框架，通过双阈值三元样本分配策略和概念对比表示学习损失，有效解决了现有方法中引入大量噪声样本的问题。

### 背景

语义连贯分布外检测(SCOOD)是一种新兴的检测设置，它使用标记的分布内数据和混合的分布内与分布外无标签数据作为训练数据，旨在让模型准确识别测试数据中的分布外样本。现有方法主要采用基于聚类的分布内样本过滤策略，但不可避免地引入大量噪声样本。

### 目的

解决现有SCOOD方法中引入大量噪声样本的问题，提高分布内和分布外样本集的纯度，并增强模型在表示空间中区分分布内和分布外样本的能力。

### 方法

提出基于预测样本分配(PSA)的SCOOD框架，包括：(1)基于预测能量分数的双阈值三元样本分配策略，将不确定的无标签数据分配到丢弃样本集；(2)概念对比表示学习损失，扩大表示空间中分布内和分布外样本的距离；(3)重训练策略，帮助模型充分拟合所选的辅助样本。

### 主要发现

在两个标准SCOOD基准上的实验表明，所提出的方法以显著优势优于最先进的方法。

### 结论

基于预测样本分配的SCOOD框架能有效解决现有方法中的噪声样本问题，提高样本纯度和模型区分能力，从而在分布外检测任务上取得更好的性能。

### 翻译

语义连贯的分布外检测(SCOOD)是一种最近提出的现实分布外检测设置：给定标记的分布内数据和混合的分布内与分布外无标签数据作为训练数据，SCOOD旨在使训练好的模型能够准确识别测试数据中的分布外样本。当前的SCOOD方法主要采用各种基于聚类的分布内样本过滤策略从无标签数据中选择干净的分布内样本，并将剩余样本视为辅助分布外数据，这不可避免地在训练中引入了大量噪声样本。为了解决上述问题，我们提出了一种基于预测样本分配(PSA)的简洁SCOOD框架。PSA包括一个基于预测能量分数的双阈值三元样本分配策略，通过将不确定的无标签数据分配到额外的丢弃样本集中，显著提高所选分布内和分布外样本集的纯度，以及一个概念对比表示学习损失，进一步扩大表示空间中分布内和分布外样本之间的距离，辅助分布内/分布外区分。此外，我们还引入了一种重训练策略，帮助模型充分拟合所选的辅助分布内/分布外样本。在两个标准SCOOD基准上的实验表明，我们的方法以显著优势优于最先进的方法。


### 论文摘要

Semantically coherent out-of-distribution detection (SCOOD) is a recently proposed realistic OOD detection setting: given labeled in-distribution (ID) data and mixed in-distribution and out-of-distribution unlabeled data as the training data, SCOOD aims to enable the trained model to accurately identify OOD samples in the testing data. Current SCOOD methods mainly adopt various clustering-based in-distribution sample filtering (IDF) strategies to select clean ID samples from unlabeled data, and take the remaining samples as auxiliary OOD data, which inevitably introduces a large number of noisy samples in training. To address the above issue, we propose a concise SCOOD framework based on predictive sample assignment (PSA). PSA includes a dual-threshold ternary sample assignment strategy based on the predictive energy score that can significantly improve the purity of the selected ID and OOD sample sets by assigning unconfident unlabeled data to an additional discard sample set, and a concept contrastive representation learning loss to further expand the distance between ID and OOD samples in the representation space to assist ID/OOD discrimination. In addition, we also introduce a retraining strategy to help the model fully fit the selected auxiliary ID/OOD samples. Experiments on two standard SCOOD benchmarks demonstrate that our approach outperforms the state-of-the-art methods by a significant margin.

---

## 60. Learning Common and Salient Generative Factors Between Two Image Datasets

**论文链接:** [http://arxiv.org/abs/2512.12800v1](http://arxiv.org/abs/2512.12800v1)

**作者:** Yunlong He, Gwilherm Lesné, Ziqian Liu, Michaël Soumm, Pietro Gori

**发布时间:** 2025-12-14

**备注:** This is the author's version of a work submitted to IEEE for possible publication. The final version may differ from this version

### GPT解析

### 总结

这篇论文提出了一种称为对比分析(CA)的新方法，用于分离两个图像数据集之间的共同生成因素和显著因素。该方法可适应GAN和扩散模型，通过新的学习策略和损失函数实现高质量的图像合成。

### 背景

图像合成领域的最新进展使高质量图像生成和操作成为可能。大多数研究工作集中在条件操作(基于给定属性修改图像)和解纠缠表示学习(每个潜在方向表示不同语义属性)这两种方法上。

### 目的

论文旨在解决一个不同且研究较少的问题：对比分析(CA)。给定两个图像数据集，目标是分离共同生成因素(两个数据集共享)和显著因素(仅属于一个数据集)。

### 方法

提出了一种新的CA框架，可适应GAN和扩散模型，学习共同和显著因素。通过定义新的、适应性强的学习策略和损失函数，确保共同和显著因素之间的相关分离，同时保持高质量生成。

### 主要发现

在多种数据集上(包括人脸、动物图像和医学扫描)的评估表明，该框架与先前方法相比，展示了更好的分离能力和图像质量合成。

### 结论

所提出的对比分析方法为图像合成领域提供了一种新的研究思路，能够在不依赖属性作为监督信号的情况下，仅使用数据集信号实现高质量的图像生成和因素分离。

### 翻译

图像合成的最新进展已经实现了高质量图像的生成和操作。大多数研究工作集中在：1)条件操作，即根据给定属性修改图像，或2)解纠缠表示学习，其中每个潜在方向应代表不同的语义属性。在本文中，我们关注一个不同且研究较少的问题，称为对比分析(CA)。给定两个图像数据集，我们希望将共同的生成因素(在两个数据集间共享)与显著因素(仅属于一个数据集)分离开来。与使用属性作为编辑监督信号(如眼镜、性别)的现有方法相比，所提出的方法较弱，因为它仅使用数据集信号。我们提出了一个用于CA的新框架，可以适应GAN和扩散模型，学习共同和显著因素。通过定义新的且适应性强的学习策略和损失函数，我们确保了共同和显著因素之间的相关分离，同时保持高质量生成。我们在多种数据集上评估了我们的方法，包括人脸、动物图像和医学扫描。与先前方法相比，我们的框架展示了更好的分离能力和图像质量合成。


### 论文摘要

Recent advancements in image synthesis have enabled high-quality image generation and manipulation. Most works focus on: 1) conditional manipulation, where an image is modified conditioned on a given attribute, or 2) disentangled representation learning, where each latent direction should represent a distinct semantic attribute. In this paper, we focus on a different and less studied research problem, called Contrastive Analysis (CA). Given two image datasets, we want to separate the common generative factors, shared across the two datasets, from the salient ones, specific to only one dataset. Compared to existing methods, which use attributes as supervised signals for editing (e.g., glasses, gender), the proposed method is weaker, since it only uses the dataset signal. We propose a novel framework for CA, that can be adapted to both GAN and Diffusion models, to learn both common and salient factors. By defining new and well-adapted learning strategies and losses, we ensure a relevant separation between common and salient factors, preserving a high-quality generation. We evaluate our approach on diverse datasets, covering human faces, animal images and medical scans. Our framework demonstrates superior separation ability and image quality synthesis compared to prior methods.

---

## 61. Anatomy-Guided Representation Learning Using a Transformer-Based Network for Thyroid Nodule Segmentation in Ultrasound Images

**论文链接:** [http://arxiv.org/abs/2512.12662v1](http://arxiv.org/abs/2512.12662v1)

**作者:** Muhammad Umar Farooq, Abd Ur Rehman, Azka Rehman, Muhammad Usman, Dong-Kyu Chae, Junaid Qadir

**发布时间:** 2025-12-14

### GPT解析

### 总结

SSMT-Net是一种创新的半监督多任务Transformer网络，解决了甲状腺结节分割中的关键挑战，通过结合未标记数据和联合优化多个任务，显著提高了分割性能。

### 背景

准确的甲状腺结节超声图像分割对诊断和治疗计划至关重要。然而，结节与周围组织之间的模糊边界、尺寸变化以及标注超声数据的稀缺性，给自动化分割带来了重大挑战。现有的深度学习模型难以整合甲状腺腺体的上下文信息，并有效泛化到多样化的案例中。

### 目的

开发一种能够有效利用未标注数据的方法，解决甲状腺结节分割面临的挑战，提高分割的准确性和鲁棒性。

### 方法

提出了SSMT-Net（半监督多任务Transformer网络），在初始无监督阶段利用未标记数据增强基于Transformer的编码器特征提取能力；在监督阶段，模型联合优化结节分割、腺体分割和结节大小估计，整合局部和全局上下文特征。

### 主要发现

在TN3K和DDTI数据集上的广泛评估表明，SSMT-Net优于最先进的方法，具有更高的准确性和鲁棒性。

### 结论

SSMT-Net显示出在真实临床应用中的潜力，能够有效解决甲状腺结节分割中的关键挑战。

### 翻译

准确的甲状腺结节超声图像分割对诊断和治疗计划至关重要。然而，结节与周围组织之间的模糊边界、尺寸变化以及标注超声数据的稀缺性，给自动化分割带来了重大挑战。现有的深度学习模型难以整合甲状腺腺体的上下文信息，并有效泛化到多样化的案例中。为应对这些挑战，我们提出了SSMT-Net，一种半监督多任务Transformer网络，它在初始无监督阶段利用未标记数据增强基于Transformer的编码器特征提取能力。在监督阶段，模型联合优化结节分割、腺体分割和结节大小估计，整合局部和全局上下文特征。在TN3K和DDTI数据集上的广泛评估表明，SSMT-Net优于最先进的方法，具有更高的准确性和鲁棒性，显示出其在真实临床应用中的潜力。


### 论文摘要

Accurate thyroid nodule segmentation in ultrasound images is critical for diagnosis and treatment planning. However, ambiguous boundaries between nodules and surrounding tissues, size variations, and the scarcity of annotated ultrasound data pose significant challenges for automated segmentation. Existing deep learning models struggle to incorporate contextual information from the thyroid gland and generalize effectively across diverse cases. To address these challenges, we propose SSMT-Net, a Semi-Supervised Multi-Task Transformer-based Network that leverages unlabeled data to enhance Transformer-centric encoder feature extraction capability in an initial unsupervised phase. In the supervised phase, the model jointly optimizes nodule segmentation, gland segmentation, and nodule size estimation, integrating both local and global contextual features. Extensive evaluations on the TN3K and DDTI datasets demonstrate that SSMT-Net outperforms state-of-the-art methods, with higher accuracy and robustness, indicating its potential for real-world clinical applications.

---

## 62. Supervised Contrastive Frame Aggregation for Video Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.12549v1](http://arxiv.org/abs/2512.12549v1)

**作者:** Shaif Chowdhury, Mushfika Rahman, Greg Hamerly

**发布时间:** 2025-12-14

**备注:** 12 pages

### GPT解析

### 总结

本研究提出了一种监督对比学习框架，用于视频表示学习，通过视频到图像聚合策略和对比学习目标，有效学习视频表示，在分类任务上取得了优异性能。

### 背景

视频表示学习是计算机视觉领域的重要任务，现有方法如视频Transformer模型计算复杂度高，需要更多计算资源。

### 目的

提出一种高效的视频表示学习方法，在保持高分类准确率的同时减少计算资源消耗。

### 方法

1. 提出监督对比学习框架，利用时间全局上下文；2. 设计视频到图像聚合策略，将多个视频帧空间排列成单个输入图像；3. 使用预训练的卷积神经网络骨干网络（如ResNet50）；4. 设计对比学习目标，直接比较模型生成的成对投影；5. 定义正样本为来自相同标签视频的投影，其他为负样本；6. 通过不同时间帧采样创建同一视频的多个自然视图。

### 主要发现

1. 在Penn Action数据集上达到76%的分类准确率，优于ViVIT的43%；2. 在HMDB51数据集上达到48%的准确率，优于ViVIT的37%；3. 该方法在计算资源消耗上少于现有方法；4. 帧级变化产生具有全局上下文的多样化正样本，减少过拟合。

### 结论

提出的监督对比帧聚合方法在监督和自监督设置中都能学习有效的视频表示，支持视频分类和字幕等任务，且在准确率和计算效率上均优于现有方法。

### 翻译

我们提出了一种用于视频表示学习的监督对比学习框架，它利用时间全局上下文。我们引入了一种视频到图像聚合策略，将每个视频的多个帧在空间上排列成单个输入图像。这种设计使得可以使用预训练的卷积神经网络骨干网络（如ResNet50），并避免了复杂视频Transformer模型的计算开销。然后，我们设计了一个对比学习目标，直接比较模型生成的成对投影。正样本定义为来自共享相同标签的视频的投影，而所有其他投影被视为负样本。使用同一底层视频的不同时间帧采样创建同一视频的多个自然视图。这些帧级变化产生具有全局上下文的多样化正样本，减少过拟合，而非依赖数据增强。在Penn Action和HMDB51数据集上的实验表明，所提出的方法在分类准确率上优于现有方法，同时需要更少的计算资源。提出的监督对比帧聚合方法在监督和自监督设置中都能学习有效的视频表示，并支持视频分类和字幕等任务。该方法在Penn Action上实现了76%的分类准确率，而ViVIT实现了43%；在HMDB51上实现了48%的准确率，而ViVIT实现了37%。


### 论文摘要

We propose a supervised contrastive learning framework for video representation learning that leverages temporally global context. We introduce a video to image aggregation strategy that spatially arranges multiple frames from each video into a single input image. This design enables the use of pre trained convolutional neural network backbones such as ResNet50 and avoids the computational overhead of complex video transformer models. We then design a contrastive learning objective that directly compares pairwise projections generated by the model. Positive pairs are defined as projections from videos sharing the same label while all other projections are treated as negatives. Multiple natural views of the same video are created using different temporal frame samplings from the same underlying video. Rather than relying on data augmentation these frame level variations produce diverse positive samples with global context and reduce overfitting. Experiments on the Penn Action and HMDB51 datasets demonstrate that the proposed method outperforms existing approaches in classification accuracy while requiring fewer computational resources. The proposed Supervised Contrastive Frame Aggregation method learns effective video representations in both supervised and self supervised settings and supports video based tasks such as classification and captioning. The method achieves seventy six percent classification accuracy on Penn Action compared to forty three percent achieved by ViVIT and forty eight percent accuracy on HMDB51 compared to thirty seven percent achieved by ViVIT.

---

## 63. DeepVekua: Geometric-Spectral Representation Learning for Physics-Informed Fields

**论文链接:** [http://arxiv.org/abs/2512.12402v1](http://arxiv.org/abs/2512.12402v1)

**作者:** Vladimer Khasia

**发布时间:** 2025-12-13

**DOI:** 10.5281/zenodo.17918695

### GPT解析

### 总结

DeepVekua是一种混合架构，统一了几何深度学习和谱分析，用于在稀疏数据条件下求解偏微分方程。

### 背景

在稀疏数据条件下求解偏微分方程是一个挑战性问题，传统方法难以有效处理。

### 目的

开发一种能够统一几何深度学习和谱分析的方法，以解决稀疏数据条件下的偏微分方程问题。

### 方法

学习一种微分同胚坐标变换，将复杂几何映射到潜在调和空间，将几何学习与物理学习分离，并以封闭形式求解最优谱权重。

### 主要发现

在平流-扩散系统中，该方法比最先进的隐式表示表现更好，比基线谱方法提高100倍性能。

### 结论

DeepVekua为解决稀疏数据条件下的偏微分方程问题提供了一种有效方法，克服了标准基于坐标网络的谱偏差问题。

### 翻译

我们提出了DeepVekua，这是一种混合架构，统一了几何深度学习和谱分析，用于在稀疏数据条件下求解偏微分方程。通过学习一种微分同胚坐标变换，将复杂几何映射到潜在调和空间，我们的方法在平流-扩散系统中优于最先进的隐式表示。与难以克服谱偏差的标准基于坐标的网络不同，DeepVekua将几何学习与物理学习分离，并以封闭形式求解最优谱权重。我们展示了比基线谱方法提高100倍的性能。代码可在https://github.com/VladimerKhasia/vekuanet获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从稀疏数据中重建连续物理场的问题，特别是那些由偏微分方程(PDEs)控制的物理场。这个问题在计算物理、医学成像和计算机图形学等多个领域都很重要，因为现有方法要么难以处理高频细节(标准神经网络存在频谱偏差)，要么在复杂几何形状上表现不佳(经典频谱方法受吉布斯现象影响)，要么需要密集数据(现有混合方法)。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者从分析现有方法的局限性入手：标准多层感知机存在频谱偏差问题，经典频谱方法在复杂几何形状上失败，混合方法要么需要密集数据要么牺牲物理可解释性。受I.N. Vekua的广义解析函数变形原理启发，作者提出'复杂物理场可能是扭曲的调和函数'这一观点。方法设计上借鉴了坐标神经网络、物理信息神经网络(PINNs)、傅里叶神经算子和多分辨率网格等现有工作，但通过学习微分同胚坐标变换将复杂几何映射到潜在调和空间，实现了创新性结合。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将近似分解为两个部分：可学习的微分同胚坐标变换(几何)和闭式频谱基投影(物理)。不同于直接近似函数，DeepVekua近似的是函数变为线性的坐标系。整体流程是：输入坐标通过多个残差块处理，每个块首先通过神经网络学习坐标变形，然后将坐标嵌入复平面，构造径向调制傅里叶基函数，最后通过可微最小二乘法解析求解最优频谱权重。各层预测结果累加得到最终输出，形成双层优化结构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)混合架构统一几何深度学习与频谱分析；2)学习微分同胚坐标变换将复杂几何映射到潜在调和空间；3)提出数值稳定的径向调制傅里叶基，结合振荡分量和增长趋势；4)在正向传播中以闭式形式求解最优基权重。相比SIREN等标准坐标网络，DeepVekua不直接近似函数而是近似坐标系，解决了频谱偏差问题；相比经典频谱方法，能处理复杂几何避免吉布斯现象；相比现有混合方法，不需要密集数据且保持物理可解释性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DeepVekua通过结合几何深度学习和频谱分析，提出了一种能够从稀疏数据中高效重建物理场的新方法，特别适用于平流-扩散方程系统，在保持物理可解释性的同时显著提高了重建精度。'}


### 论文摘要

We present DeepVekua, a hybrid architecture that unifies geometric deep learning with spectral analysis to solve partial differential equations (PDEs) in sparse data regimes. By learning a diffeomorphic coordinate transformation that maps complex geometries to a latent harmonic space, our method outperforms state-of-the-art implicit representations on advection-diffusion systems. Unlike standard coordinate-based networks which struggle with spectral bias, DeepVekua separates the learning of geometry from the learning of physics, solving for optimal spectral weights in closed form. We demonstrate a 100x improvement over spectral baselines. The code is available at https://github.com/VladimerKhasia/vekuanet.

---

## 64. Fine-Grained Zero-Shot Learning with Attribute-Centric Representations

**论文链接:** [http://arxiv.org/abs/2512.12219v1](http://arxiv.org/abs/2512.12219v1)

**作者:** Zhi Chen, Jingcai Guo, Taotao Cai, Yuxiang Cai

**发布时间:** 2025-12-13

**备注:** Preprint

### GPT解析

### 总结

本文提出了一种名为AttributeCentric Representations (ACR)的零样本学习框架，通过属性解纠缠解决细粒度类别识别中的视觉差异区分问题，在多个基准数据集上取得了最先进的结果。

### 背景

识别未见过的细粒度类别需要能够区分细微视觉差异的模型，通常通过将已见类别的视觉属性关系迁移到未见类别来实现。

### 目的

解决属性纠缠问题，即传统模型将不同属性（如颜色、形状和纹理）压缩为单个视觉嵌入导致的干扰问题。

### 方法

ACR框架包含两个混合专家组件：Mixture of Patch Experts (MoPE)使用双重路由机制将图像块分派给专门专家；Mixture of Attribute Experts (MoAE)将专家细化的特征投影到稀疏的、面向部分的属性图。

### 主要发现

传统方法的事后解决方案不足以处理已混合的表示，而ACR在表示学习过程中强制属性解纠缠，有效解决了属性纠缠问题。

### 结论

在零样本学习基准数据集CUB、AwA2和SUN上，ACR框架取得了持续的最先进结果，证明了其在细粒度零样本分类任务中的有效性。

### 翻译

识别未见过的细粒度类别需要一个能够区分细微视觉差异的模型。这通常通过将已见类别的视觉属性关系迁移到未见类别来实现。核心挑战是属性纠缠，传统模型将颜色、形状和纹理等不同属性压缩为单个视觉嵌入。这种干扰掩盖了这些关键区别。先前工作的事后解决方案不足，因为它们已在混合的表示上操作。我们提出一个零样本学习框架，学习以属性为中心的表示（ACR）来解决这个问题，在表示学习过程中强制属性解纠缠。ACR通过两个混合专家组件实现，包括Mixture of Patch Experts (MoPE)和Mixture of Attribute Experts (MoAE)。首先，MoPE使用双重路由机制插入到transformer中，有条件地将图像块分派给专门的专家。这确保了连贯的属性族由专门的专家处理。最后，MoAE头将这些专家细化的特征投影到稀疏的、面向部分的属性图，以实现鲁棒的零样本分类。在零样本学习基准数据集CUB、AwA2和SUN上，我们的ACR取得了持续的最先进结果。


### 论文摘要

Recognizing unseen fine-grained categories demands a model that can distinguish subtle visual differences. This is typically achieved by transferring visual-attribute relationships from seen classes to unseen classes. The core challenge is attribute entanglement, where conventional models collapse distinct attributes like color, shape, and texture into a single visual embedding. This causes interference that masks these critical distinctions. The post-hoc solutions of previous work are insufficient, as they operate on representations that are already mixed. We propose a zero-shot learning framework that learns AttributeCentric Representations (ACR) to tackle this problem by imposing attribute disentanglement during representation learning. ACR is achieved with two mixture-of-experts components, including Mixture of Patch Experts (MoPE) and Mixture of Attribute Experts (MoAE). First, MoPE is inserted into the transformer using a dual-level routing mechanism to conditionally dispatch image patches to specialized experts. This ensures coherent attribute families are processed by dedicated experts. Finally, the MoAE head projects these expert-refined features into sparse, partaware attribute maps for robust zero-shot classification. On zero-shot learning benchmark datasets CUB, AwA2, and SUN, our ACR achieves consistent state-of-the-art results.

---

## 65. Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring

**论文链接:** [http://arxiv.org/abs/2512.12069v1](http://arxiv.org/abs/2512.12069v1)

**作者:** Peichun Hua, Hao Li, Shanghao Shi, Zhiyuan Yu, Ning Zhang

**发布时间:** 2025-12-12

**备注:** 40 pages, 13 figures

### GPT解析

### 总结

这篇论文提出了一种名为表示对比评分（RCS）的新型越狱检测框架，通过分析LVLM内部表示的几何结构来区分良性输入和恶意输入，实现了良好的泛化能力和实用性。

### 背景

大型视觉-语言模型（LVLMs）容易受到多模态越狱攻击的威胁，现有防御方法要么针对特定攻击模式（限制了泛化能力），要么带来高计算开销。轻量级异常检测方法虽然前景广阔，但其常见的一类设计倾向于将新型良性输入与恶意输入混淆，导致不可靠的过度拒绝。

### 目的

开发一种能够有效检测越狱攻击的方法，同时保持良好的泛化能力和实用性，解决现有轻量级异常检测方法中良性输入与恶意输入混淆的问题。

### 方法

提出表示对比评分（RCS）框架，基于LVLM内部表示中最强安全信号存在的洞察。检查这些表示的内部几何结构，学习轻量级投影在安全关键层中分离良性输入和恶意输入，实现简单而强大的对比分数。具体实现包括MCD（马氏对比检测）和KCD（K最近邻对比检测）。

### 主要发现

通过对适当的内部表示应用简单、可解释的统计方法，可以实现有效的越狱检测。RCS框架能够区分真正的恶意攻击与新型良性输入，MCD和KCD在针对未见过的攻击类型的泛化能力上达到了最先进的性能。

### 结论

有效且实用的越狱检测可以通过对适当的内部表示应用简单、可解释的统计方法实现，为更安全的LVLM部署提供了实用路径。相关代码已在Github上公开。

### 翻译

大型视觉-语言模型（LVLMs）容易受到越来越多的多模态越狱攻击的威胁，需要既能推广到新型威胁又适合实际部署的防御方法。许多当前策略要么针对特定攻击模式（这限制了泛化能力），要么带来高计算开销。虽然轻量级异常检测方法提供了一个有前景的方向，但我们发现其常见的一类设计倾向于将新型良性输入与恶意输入混淆，导致不可靠的过度拒绝。为此，我们提出了表示对比评分（RCS），这是一个基于关键洞察构建的框架：最强的安全信号存在于LVLM自身的内部表示中。我们的方法检查这些表示的内部几何结构，学习一个轻量级投影，在安全关键层中最大程度地分离良性输入和恶意输入。这使得一个简单而强大的对比分数能够区分真正的恶意意图与仅仅是新颖性。我们的实现MCD（马氏对比检测）和KCD（K最近邻对比检测）在一个具有挑战性的评估协议上实现了最先进的性能，该协议旨在测试对未见攻击类型的泛化能力。这项工作证明，通过对适当的内部表示应用简单、可解释的统计方法，可以实现有效的越狱检测，为更安全的LVLM部署提供了实用路径。我们的代码可在Github上获取：https://github.com/sarendis56/Jailbreak_Detection_RCS。


### 论文摘要

Large Vision-Language Models (LVLMs) are vulnerable to a growing array of multimodal jailbreak attacks, necessitating defenses that are both generalizable to novel threats and efficient for practical deployment. Many current strategies fall short, either targeting specific attack patterns, which limits generalization, or imposing high computational overhead. While lightweight anomaly-detection methods offer a promising direction, we find that their common one-class design tends to confuse novel benign inputs with malicious ones, leading to unreliable over-rejection. To address this, we propose Representational Contrastive Scoring (RCS), a framework built on a key insight: the most potent safety signals reside within the LVLM's own internal representations. Our approach inspects the internal geometry of these representations, learning a lightweight projection to maximally separate benign and malicious inputs in safety-critical layers. This enables a simple yet powerful contrastive score that differentiates true malicious intent from mere novelty. Our instantiations, MCD (Mahalanobis Contrastive Detection) and KCD (K-nearest Contrastive Detection), achieve state-of-the-art performance on a challenging evaluation protocol designed to test generalization to unseen attack types. This work demonstrates that effective jailbreak detection can be achieved by applying simple, interpretable statistical methods to the appropriate internal representations, offering a practical path towards safer LVLM deployment. Our code is available on Github https://github.com/sarendis56/Jailbreak_Detection_RCS.

---

## 66. CLARGA: Multimodal Graph Representation Learning over Arbitrary Sets of Modalities

**论文链接:** [http://arxiv.org/abs/2512.11901v1](http://arxiv.org/abs/2512.11901v1)

**作者:** Santosh Patapati

**发布时间:** 2025-12-10

**备注:** WACV; Supplementary material is available on CVF proceedings

### GPT解析

### 总结

CLARGA是一种通用的多模态融合架构，用于多模态表示学习，可处理任意数量和类型的模态，无需改变底层框架。

### 背景

需要一种能够灵活处理多种模态数据的通用框架，以适应各种机器学习任务。

### 目的

开发一种能够自适应融合不同模态表示的架构，提高跨模态一致性和对噪声输入的鲁棒性。

### 方法

通过构建特征上的注意力加权图并使用多头图注意力网络传递消息，为每个样本学习模态间的相互影响；使用可学习掩码适应缺失模态；结合监督任务损失和对比InfoNCE损失进行训练。

### 主要发现

CLARGA在7个跨金融、人机交互、多媒体分类和情感计算的数据集上持续优于基线和最先进模型；展示了对缺失输入的鲁棒性和在特定任务上的出色能力。

### 结论

CLARGA可以轻松集成到各种机器学习模型中，有效且高效地学习跨任务的表示。

### 翻译

我们介绍了CLARGA，一种用于多模态表示学习的通用多模态融合架构，它可以在不改变底层框架的情况下处理任意数量和类型的模态。给定监督数据集，CLARGA可以应用于几乎任何机器学习任务，以融合不同的多模态表示供下游层处理。在样本基础上，CLARGA通过在特征上构建注意力加权图并使用多头图注意力网络沿此图传递消息，学习模态之间应如何相互影响。这不仅使CLARGA高度自适应（因为它为不同样本构建独特图），而且随着模态数量增长，它实现了具有亚二次复杂度的高效融合。通过可学习掩码，它还可以适应缺失的模态输入。该模型使用混合目标函数进行训练，结合了监督任务损失和对比InfoNCE损失，提高了跨模态一致性和对噪声输入的鲁棒性。我们在7个跨金融、人机交互、通用多媒体分类和情感计算的数据集上展示了CLARGA在多样化多模态表示学习任务中的有效性。它持续优于基线、最先进的模型和消融实验。额外的实验还证明了其对缺失输入的鲁棒性以及在特定任务上的出色能力。总体而言，CLARGA可以轻松插入机器学习模型中，以有效且高效地学习各种任务的表示。


### 论文摘要

We introduce CLARGA, a general-purpose multimodal fusion architecture for multimodal representation learning that works with any number and type of modalities without changing the underlying framework. Given a supervised dataset, CLARGA can be applied to virtually any machine learning task to fuse different multimodal representations for processing by downstream layers. On a sample-by-sample basis, CLARGA learns how modalities should inform one another by building an attention weighted graph over their features and passing messages along this graph with a multi-head Graph Attention Network. Not only does this make CLARGA highly adaptive, as it constructs unique graphs for different samples, it makes for efficient fusion with sub-quadratic complexity as the number of modalities grows. Through a learnable mask, it can also adapt to missing modality inputs. The model is trained with a hybrid objective that combines a supervised task loss with contrastive InfoNCE loss, improving cross-modal consistency and robustness to noisy inputs. We demonstrate CLARGA's effectiveness in diverse multimodal representation learning tasks across 7 datasets spanning finance, human-computer interaction, general multimedia classification, and affective computing. It consistently outperforms baselines, state-of-the-art models, and ablations. Additional experiments also demonstrate its robustness to missing inputs and ability to excel on niche tasks. Overall, CLARGA can be easily plugged into machine learning models for effective and efficient learning of representations across a wide variety of tasks.

---

## 67. Grab-3D: Detecting AI-Generated Videos from 3D Geometric Temporal Consistency

**论文链接:** [http://arxiv.org/abs/2512.13665v1](http://arxiv.org/abs/2512.13665v1)

**作者:** Wenhan Chen, Sezer Karaoglu, Theo Gevers

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出了一种名为Grab-3D的几何感知transformer框架，用于检测AI生成的视频，通过分析3D几何时间一致性来区分真实视频与AI生成视频。

### 背景

基于扩散模型的生成技术使AI能够创建高度逼真的视频，这使得可靠的检测机制变得更为重要，但现有检测方法对生成视频中存在的3D几何模式探索有限。

### 目的

开发一种基于3D几何时间一致性来检测AI生成视频的方法，揭示真实视频与AI生成视频在几何一致性方面的根本差异。

### 方法

使用消失点作为3D几何模式的显式表示；构建静态场景的AI生成视频数据集以实现可靠的3D几何特征提取；提出具有几何位置编码、时间-几何注意力和基于EMA的几何分类器头的几何感知transformer框架。

### 主要发现

真实视频与AI生成视频在几何一致性方面存在根本差异；Grab-3D在检测AI生成视频方面显著优于现有最先进的检测器。

### 结论

Grab-3D能够实现强大的跨领域泛化能力，对未见过的生成器也有效，为AI生成视频检测提供了新的解决方案。

### 翻译

最近的扩散生成技术进展使AI模型能够生成高度逼真的视频，增加了对可靠检测机制的需求。然而，现有的检测方法对生成视频中存在的3D几何模式仅提供了有限的探索。在本文中，我们使用消失点作为3D几何模式的显式表示，揭示了真实视频与AI生成视频在几何一致性方面的基本差异。我们引入了Grab-3D，一种基于3D几何时间一致性的几何感知transformer框架，用于检测AI生成的视频。为了实现可靠的评估，我们构建了一个静态场景的AI生成视频数据集，允许稳定的3D几何特征提取。我们提出了一种几何感知transformer，配备了几何位置编码、时间-几何注意力和基于EMA的几何分类器头，将3D几何感知显式注入到时间建模中。实验证明，Grab-3D显著优于最先进的检测器，实现了对未见过的生成器的强大跨领域泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何检测AI生成的视频问题。随着AI视频生成技术快速发展，生成的视频越来越逼真，难以与真实视频区分，这对媒体信息的真实性和安全性构成重大挑战。现有检测方法主要依赖纹理、伪影或RGB运动模式，但这些特征在不同生成器和内容变化下不够鲁棒，且缺乏对3D几何一致性的建模，而3D几何一致性是真实视频中的基本物理特性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到真实视频中3D几何结构随时间保持物理一致性，而AI生成视频则表现出这种一致性的偏差。他们选择消失点作为3D几何的显式表示，发现真实视频中消失点运动稳定，而AI生成视频则波动明显。为避免物体运动干扰，他们构建了仅含相机运动的静态场景数据集。方法借鉴了3D感知生成模型(如ViewDiff、CamCo)的几何一致性思想，以及现有视频检测方法(如DeMamba)的transformer架构，但创新性地将3D几何感知注入时间建模。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用3D几何时间一致性区分真实和AI生成视频。真实视频中场景几何结构随时间保持物理一致性，而AI生成视频无法保持这种一致性，特别是在消失点运动轨迹上表现不稳定。整体流程包括：1)提取消失点特征并构建3D几何表示；2)构建静态场景视频数据集；3)设计几何感知transformer框架(GRAB-3D)，包含几何位置编码、时间-几何注意力和几何分类器头；4)预训练几何头并优化分类器；5)在静态场景数据集上评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统研究AI生成视频中的3D几何时间一致性；2)构建首个静态场景AI生成视频数据集(3,322对真实-生成视频)；3)提出GRAB-3D框架，包含几何位置编码、时间-几何注意力和几何分类器头。相比之前工作，不同之处在于：专注于3D几何一致性而非纹理或运动特征；显式将3D几何感知注入时间建模；使用完全静态场景确保可靠特征提取；在跨域检测(未见生成器)上表现出色。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过利用3D几何时间一致性和创新的几何感知transformer框架，显著提升了AI生成视频检测的准确性和跨域泛化能力。'}


### 论文摘要

Recent advances in diffusion-based generation techniques enable AI models to produce highly realistic videos, heightening the need for reliable detection mechanisms. However, existing detection methods provide only limited exploration of the 3D geometric patterns present in generated videos. In this paper, we use vanishing points as an explicit representation of 3D geometry patterns, revealing fundamental discrepancies in geometric consistency between real and AI-generated videos. We introduce Grab-3D, a geometry-aware transformer framework for detecting AI-generated videos based on 3D geometric temporal consistency. To enable reliable evaluation, we construct an AI-generated video dataset of static scenes, allowing stable 3D geometric feature extraction. We propose a geometry-aware transformer equipped with geometric positional encoding, temporal-geometric attention, and an EMA-based geometric classifier head to explicitly inject 3D geometric awareness into temporal modeling. Experiments demonstrate that Grab-3D significantly outperforms state-of-the-art detectors, achieving robust cross-domain generalization to unseen generators.

---

## 68. TARA: Simple and Efficient Time Aware Retrieval Adaptation of MLLMs for Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.13511v1](http://arxiv.org/abs/2512.13511v1)

**作者:** Piyush Bagad, Andrew Zisserman

**发布时间:** 2025-12-15

**备注:** 18 Pages. Project page at http://bpiyush.github.io/tara-website

### GPT解析

### 总结

论文提出了TARA（时间感知检索适应）方法，构建了一个通用的、具有时间感知能力的视频-文本嵌入模型，用于检索任务，无需使用任何视频数据即可将多模态大语言模型适应为时间感知模型。

### 背景

现有视频-文本嵌入模型可能缺乏对时间信息的感知能力，需要一种方法来增强模型的时间感知特性。

### 目的

构建一个通用的、具有时间感知能力的视频-文本嵌入模型，用于检索任务，并且不需要使用任何视频数据。

### 方法

提出TARA方法，将多模态大语言模型（MLLMs）适应为时间感知的视频-文本嵌入模型；同时提出新的基准测试，使用时间相反（手性）动作作为困难负样本，并包含手性和非手性动作的精心划分。

### 主要发现

TARA在提出的手性基准上优于所有现有视频-文本模型；在标准基准上也取得良好结果；TARA还具有否定感知能力，在视频中的动词和副词理解方面取得最先进性能。

### 结论

TARA产生了一个强大、多功能、具有时间感知能力的视频-文本嵌入模型，具有最先进的零样本性能。

### 翻译

我们的目标是构建一个用于检索的通用时间感知视频-文本嵌入模型。为此，我们提出了一种简单高效的方案，称为TARA（时间感知检索适应），无需使用任何视频数据即可将多模态大语言模型适应为时间感知的视频-文本嵌入模型。为了评估检索中的时间感知能力，我们提出了一个新的基准，使用时间相反的手性动作作为困难负样本，并包含手性和非手性动作的精心划分。我们证明TARA在该手性基准上优于所有现有的视频-文本模型，同时在标准基准上也取得了良好结果。此外，我们发现了TARA在时间感知能力之外的额外优势：TARA嵌入具有否定感知能力，在评估视频检索中否定的NegBench基准中所示，TARA在视频中的动词和副词理解方面取得了最先进的性能。总体而言，TARA产生了一个强大、多功能、具有时间感知能力的视频-文本嵌入模型，具有最先进的零样本性能。


### 论文摘要

Our objective is to build a general time-aware video-text embedding model for retrieval. To that end, we propose a simple and efficient recipe, dubbed TARA (Time Aware Retrieval Adaptation), to adapt Multimodal LLMs (MLLMs) to a time-aware video-text embedding model without using any video data at all. For evaluating time-awareness in retrieval, we propose a new benchmark with temporally opposite (chiral) actions as hard negatives and curated splits for chiral and non-chiral actions. We show that TARA outperforms all existing video-text models on this chiral benchmark while also achieving strong results on standard benchmarks. Furthermore, we discover additional benefits of TARA beyond time-awareness: (i) TARA embeddings are negation-aware as shown in NegBench benchmark that evaluates negation in video retrieval, (ii) TARA achieves state of the art performance on verb and adverb understanding in videos. Overall, TARA yields a strong, versatile, time-aware video-text embedding model with state of the art zero-shot performance.

---

## 69. USTM: Unified Spatial and Temporal Modeling for Continuous Sign Language Recognition

**论文链接:** [http://arxiv.org/abs/2512.13415v1](http://arxiv.org/abs/2512.13415v1)

**作者:** Ahmed Abul Hasanaath, Hamzah Luqman

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出了一种统一时空建模(USTM)框架，用于连续手语识别，通过结合增强型Swin Transformer主干网络和轻量级时间适配器与位置编码(TAPE)，有效捕捉细粒度空间特征和短期长期时间上下文，在多个基准数据集上实现了最先进的性能。

### 背景

连续手语识别需要精确的时空建模来准确识别视频中的手势序列，而现有框架通常基于CNN的空间主干网络结合时间卷积或循环模块，这些技术在捕捉细粒度手部和面部线索以及建模长程时间依赖性方面存在不足。

### 目的

解决现有技术在捕捉细粒度特征和建模长程时间依赖性方面的局限性，提出一种能够有效建模复杂模式的统一时空建模框架。

### 方法

提出USTM框架，一种时空编码器，结合增强型Swin Transformer主干网络和轻量级时间适配器与位置编码(TAPE)，能够捕捉细粒度空间特征以及短期和长期时间上下文，无需依赖多流输入或辅助模态，从RGB视频中实现鲁棒的手语识别。

### 主要发现

在PHOENIX14、PHOENIX14T和CSL-Daily等基准数据集上的实验表明，USTM在基于RGB的方法以及多模态CSLR方法上达到了最先进的性能，同时与多流方法相比保持了具有竞争力的性能。

### 结论

USTM框架在连续手语识别方面展示了其优势和有效性，代码已在GitHub上公开：https://github.com/gufranSabri/USTM

### 翻译

连续手语识别(CSLR)需要精确的时空建模来准确识别视频中的手势序列。现有框架通常依赖于基于CNN的空间主干网络，结合时间卷积或循环模块。这些技术在捕捉细粒度的手部和面部线索以及建模长程时间依赖性方面存在不足。为解决这些限制，我们提出了统一时空建模(USTM)框架，一种时空编码器，通过结合增强型Swin Transformer主干网络和轻量级时间适配器与位置编码(TAPE)，有效建模复杂模式。我们的框架捕捉细粒度空间特征以及短期和长期时间上下文，使系统能够仅从RGB视频中实现鲁棒的手语识别，而无需依赖多流输入或辅助模态。在PHOENIX14、PHOENIX14T和CSL-Daily等基准数据集上的大量实验表明，USTM在基于RGB以及多模态CSLR方法上达到了最先进的性能，同时与多流方法相比保持了具有竞争力的性能。这些结果突显了USTM框架在CSLR方面的优势和有效性。代码可在https://github.com/gufranSabri/USTM获取。


### 论文摘要

Continuous sign language recognition (CSLR) requires precise spatio-temporal modeling to accurately recognize sequences of gestures in videos. Existing frameworks often rely on CNN-based spatial backbones combined with temporal convolution or recurrent modules. These techniques fail in capturing fine-grained hand and facial cues and modeling long-range temporal dependencies. To address these limitations, we propose the Unified Spatio-Temporal Modeling (USTM) framework, a spatio-temporal encoder that effectively models complex patterns using a combination of a Swin Transformer backbone enhanced with lightweight temporal adapter with positional embeddings (TAPE). Our framework captures fine-grained spatial features alongside short and long-term temporal context, enabling robust sign language recognition from RGB videos without relying on multi-stream inputs or auxiliary modalities. Extensive experiments on benchmarked datasets including PHOENIX14, PHOENIX14T, and CSL-Daily demonstrate that USTM achieves state-of-the-art performance against RGB-based as well as multi-modal CSLR approaches, while maintaining competitive performance against multi-stream approaches. These results highlight the strength and efficacy of the USTM framework for CSLR. The code is available at https://github.com/gufranSabri/USTM

---

## 70. What Happens Next? Next Scene Prediction with a Unified Video Model

**论文链接:** [http://arxiv.org/abs/2512.13015v1](http://arxiv.org/abs/2512.13015v1)

**作者:** Xinjie Li, Zhimin Chen, Rui Zhao, Florian Schiffers, Zhenyu Liao, Vimal Bhat

**发布时间:** 2025-12-15

### GPT解析

### 总结

该论文引入了Next Scene Prediction (NSP)任务，提出了一种结合Qwen-VL和LTX的统一框架，通过三阶段训练方法在NSP数据集上实现了最先进的性能，提高了多模态系统对未来场景的预测能力。

### 背景

最近的统一理解和生成模型显著提升了视觉生成能力，但这些模型主要关注传统任务如文本到视频生成，导致统一模型的时间推理潜力在很大程度上未被探索。

### 目的

为了填补这一空白，作者引入了NSP任务，推动统一视频模型进行时间和因果推理，要求模型从先前上下文预测合理的未来场景。

### 方法

作者提出了一个统一框架，结合Qwen-VL进行理解和LTX进行合成，通过潜查询嵌入和连接器模块桥接。模型在三个阶段训练：文本到视频预训练、监督微调和强化学习（通过GRPO）以及提出的因果一致性奖励。

### 主要发现

实验证明，该模型在基准测试上达到了最先进的性能，显著提高了通用多模态系统预测接下来会发生什么的能力。

### 结论

通过NSP任务和提出的统一框架，作者展示了统一模型在时间和因果推理方面的潜力，为多模态系统预测未来场景提供了新方向。

### 翻译

最近的统一理解和生成模型显著提升了视觉生成能力。然而，这些模型对传统任务（如文本到视频生成）的关注，使得统一模型的时间推理潜力在很大程度上未被探索。为了填补这一空白，我们引入了Next Scene Prediction (NSP)，这是一个新任务，推动统一视频模型进行时间和因果推理。与文本到视频生成不同，NSP要求从先前上下文预测合理的未来，需要更深层次的理解和推理。为了解决这一任务，我们提出了一个统一框架，结合Qwen-VL进行理解和LTX进行合成，通过潜查询嵌入和连接器模块桥接。该模型在我们新收集的大规模NSP数据集上分三个阶段训练：文本到视频预训练、监督微调和强化学习（通过GRPO）以及我们提出的因果一致性奖励。实验表明，我们的模型在基准测试上取得了最先进的性能，提高了通用多模态系统预测接下来会发生什么的能力。


### 论文摘要

Recent unified models for joint understanding and generation have significantly advanced visual generation capabilities. However, their focus on conventional tasks like text-to-video generation has left the temporal reasoning potential of unified models largely underexplored. To address this gap, we introduce Next Scene Prediction (NSP), a new task that pushes unified video models toward temporal and causal reasoning. Unlike text-to-video generation, NSP requires predicting plausible futures from preceding context, demanding deeper understanding and reasoning. To tackle this task, we propose a unified framework combining Qwen-VL for comprehension and LTX for synthesis, bridged by a latent query embedding and a connector module. This model is trained in three stages on our newly curated, large-scale NSP dataset: text-to-video pre-training, supervised fine-tuning, and reinforcement learning (via GRPO) with our proposed causal consistency reward. Experiments demonstrate our model achieves state-of-the-art performance on our benchmark, advancing the capability of generalist multimodal systems to anticipate what happens next.

---

## 71. Unified Interactive Multimodal Moment Retrieval via Cascaded Embedding-Reranking and Temporal-Aware Score Fusion

**论文链接:** [http://arxiv.org/abs/2512.12935v1](http://arxiv.org/abs/2512.12935v1)

**作者:** Toan Le Ngo Thanh, Phat Ha Huu, Tan Nguyen Dang Duy, Thong Nguyen Le Minh, Anh Nguyen Nhu Tinh

**发布时间:** 2025-12-15

**备注:** Accepted at AAAI Workshop 2026

### GPT解析

### 总结

本文提出了一种统一的多模态时刻检索系统，通过三个创新解决了现有方法的三大挑战：固定权重融合策略的局限性、时间建模难以捕捉连贯事件序列以及需要手动模态选择的问题。

### 背景

视频内容的指数级增长对高效的多模态时刻检索系统提出了迫切需求，现有方法难以满足这一需求。

### 目的

开发一个能够处理跨模态噪声、模糊查询，并能捕捉连贯事件序列且无需手动模态选择的多模态时刻检索系统。

### 方法

提出三个关键创新：1) 级联双重嵌入流水线（结合BEIT-3、SigLIP和BLIP-2）；2) 时间感知评分机制（通过束搜索应用指数衰减惩罚）；3) Agent引导的查询分解（使用GPT-4o自动解释和分解模糊查询）。

### 主要发现

定性分析表明，该系统能够有效处理模糊查询，检索时间连贯的序列，并动态适应融合策略，显著提升了交互式时刻搜索能力。

### 结论

该统一多模态时刻检索系统通过创新方法解决了现有技术的关键挑战，推进了交互式时刻搜索领域的发展。

### 翻译

视频内容的指数级增长对高效的多模态时刻检索系统提出了迫切需求。然而，现有方法面临三个关键挑战：(1)固定权重融合策略无法处理跨模态噪声和模糊查询，(2)时间建模难以捕捉连贯的事件序列，同时对不现实的间隔进行惩罚，(3)系统需要手动选择模态，降低了可用性。我们提出了一种统一的多模态时刻检索系统，具有三个关键创新。首先，级联双重嵌入流水线结合BEIT-3和SigLIP进行广泛检索，通过基于BLIP-2的重排序来平衡召回率和精确率。其次，时间感知评分机制通过束搜索对大的时间间隔应用指数衰减惩罚，构建连贯的事件序列而非孤立帧。第三，Agent引导的查询分解（GPT-4o）自动解释模糊查询，将其分解为特定模态的子查询（视觉/OCR/ASR），并进行自适应分数融合，消除了手动模态选择。定性分析表明，我们的系统能够有效处理模糊查询，检索时间连贯的序列，并动态适应融合策略，推进了交互式时刻搜索能力。


### 论文摘要

The exponential growth of video content has created an urgent need for efficient multimodal moment retrieval systems. However, existing approaches face three critical challenges: (1) fixed-weight fusion strategies fail across cross modal noise and ambiguous queries, (2) temporal modeling struggles to capture coherent event sequences while penalizing unrealistic gaps, and (3) systems require manual modality selection, reducing usability. We propose a unified multimodal moment retrieval system with three key innovations. First, a cascaded dual-embedding pipeline combines BEIT-3 and SigLIP for broad retrieval, refined by BLIP-2 based reranking to balance recall and precision. Second, a temporal-aware scoring mechanism applies exponential decay penalties to large temporal gaps via beam search, constructing coherent event sequences rather than isolated frames. Third, Agent-guided query decomposition (GPT-4o) automatically interprets ambiguous queries, decomposes them into modality specific sub-queries (visual/OCR/ASR), and performs adaptive score fusion eliminating manual modality selection. Qualitative analysis demonstrates that our system effectively handles ambiguous queries, retrieves temporally coherent sequences, and dynamically adapts fusion strategies, advancing interactive moment search capabilities.

---

## 72. MADTempo: An Interactive System for Multi-Event Temporal Video Retrieval with Query Augmentation

**论文链接:** [http://arxiv.org/abs/2512.12929v1](http://arxiv.org/abs/2512.12929v1)

**作者:** Huu-An Vu, Van-Khanh Mai, Trong-Tam Nguyen, Quang-Duc Dam, Tien-Huy Nguyen, Thanh-Huong Le

**发布时间:** 2025-12-15

### GPT解析

### 总结

MADTempo是一个统一了时间搜索和网络规模视觉基础的视频检索框架，通过事件级连续性建模和外部网络图像扩展，解决了现有方法在时间依赖建模和罕见概念处理方面的不足。

### 背景

在线平台上视频内容的快速扩展加速了对检索系统的需求，这些系统不仅要理解独立的视觉时刻，还要理解复杂事件的时间结构。

### 目的

解决现有方法在建模多个事件间时间依赖关系和处理引用未见或罕见视觉概念的查询方面的不足。

### 方法

1) 时间搜索机制：通过聚合连续视频片段的相似度分数捕捉事件级连续性，实现多事件查询的连贯检索；2) 基于Google图片搜索的回退模块：使用外部网络图像扩展查询表示，提高对分布外查询的鲁棒性。

### 主要发现

这两个组件共同提升了现代视频检索系统的时间推理和泛化能力。

### 结论

这种方法为大规模视频语料库中更语义感知和自适应的检索铺平了道路。

### 翻译

在线平台上视频内容的快速扩展加速了对检索系统的需求，这些系统不仅要理解独立的视觉时刻，还要理解复杂事件的时间结构。现有方法通常难以建模多个事件间的时间依赖关系，以及处理引用未见或罕见视觉概念的查询。为应对这些挑战，我们引入了MADTempo，一个由AIO_Trinh团队开发的视频检索框架，它统一了时间搜索与网络规模视觉基础。我们的时间搜索机制通过聚合连续视频片段的相似度分数来捕捉事件级连续性，实现多事件查询的连贯检索。互补地，基于Google图片搜索的回退模块使用外部网络图像扩展查询表示，有效弥合预训练视觉嵌入的差距，并提高对分布外查询的鲁棒性。这些组件共同提升了现代视频检索系统的时间推理和泛化能力，为大规模视频语料库中更语义感知和自适应的检索铺平了道路。


### 论文摘要

The rapid expansion of video content across online platforms has accelerated the need for retrieval systems capable of understanding not only isolated visual moments but also the temporal structure of complex events. Existing approaches often fall short in modeling temporal dependencies across multiple events and in handling queries that reference unseen or rare visual concepts. To address these challenges, we introduce MADTempo, a video retrieval framework developed by our team, AIO_Trinh, that unifies temporal search with web-scale visual grounding. Our temporal search mechanism captures event-level continuity by aggregating similarity scores across sequential video segments, enabling coherent retrieval of multi-event queries. Complementarily, a Google Image Search-based fallback module expands query representations with external web imagery, effectively bridging gaps in pretrained visual embeddings and improving robustness against out-of-distribution (OOD) queries. Together, these components advance the temporal reasoning and generalization capabilities of modern video retrieval systems, paving the way for more semantically aware and adaptive retrieval across large-scale video corpora.

---

## 73. StreamingAssistant: Efficient Visual Token Pruning for Accelerating Online Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.12560v1](http://arxiv.org/abs/2512.12560v1)

**作者:** Xinqi Jin, Hanxun Yu, Bohan Yu, Kebin Liu, Jian Liu, Keda Tao, Yixuan Pei, Huan Wang, Fan Dang, Jiangchuan Liu, Weiqiang Wang

**发布时间:** 2025-12-14

### GPT解析

### 总结

本文提出了一种基于token剪枝的方法来解决多模态大语言模型应用于在线视频理解时的计算效率问题，通过减少上下文长度同时保留关键信息，显著提高了准确性且剪枝延迟极低。

### 背景

在线视频理解对于公共监控和AI眼镜等应用至关重要，但将多模态大语言模型应用于此领域面临挑战。

### 目的

解决视频帧数量庞大导致的GPU内存使用量高和计算延迟问题，减少上下文长度同时保留关键信息。

### 方法

提出token剪枝方法，引入MSSAVT冗余度量指标，设计掩码剪枝策略确保只修剪互不相邻的标记，并整合现有时间冗余剪枝方法消除视频模态的时间冗余。

### 主要发现

在多个在线和离线视频理解基准测试中，该方法显著提高了准确性（最多提高4%）且剪枝延迟可忽略不计（小于1毫秒）。

### 结论

该方法有效解决了多模态大语言模型应用于视频理解领域的计算效率挑战，完整实现将公开可用。

### 翻译

在线视频理解对于公共监控和AI眼镜等应用至关重要。然而，将多模态大语言模型应用于此领域具有挑战性，因为视频帧数量庞大，导致GPU内存使用量高和计算延迟。为解决这些挑战，我们提出token剪枝作为减少上下文长度同时保留关键信息的方法。具体来说，我们引入了一种新的冗余度量指标——与空间相邻视频标记的最大相似性（MSSAVT），该指标考虑了标记相似性和空间位置。为了缓解剪枝与冗余之间的双向依赖关系，我们进一步设计了一种掩码剪枝策略，确保只修剪互不相邻的标记。我们还整合了一个现有的基于时间冗余的剪枝方法，以消除视频模态的时间冗余。在多个在线和离线视频理解基准测试上的实验结果表明，我们的方法显著提高了准确性（最多提高4%），同时产生可忽略不计的剪枝延迟（小于1毫秒）。我们的完整实现将公开提供。


### 论文摘要

Online video understanding is essential for applications like public surveillance and AI glasses. However, applying Multimodal Large Language Models (MLLMs) to this domain is challenging due to the large number of video frames, resulting in high GPU memory usage and computational latency. To address these challenges, we propose token pruning as a means to reduce context length while retaining critical information. Specifically, we introduce a novel redundancy metric, Maximum Similarity to Spatially Adjacent Video Tokens (MSSAVT), which accounts for both token similarity and spatial position. To mitigate the bidirectional dependency between pruning and redundancy, we further design a masked pruning strategy that ensures only mutually unadjacent tokens are pruned. We also integrate an existing temporal redundancy-based pruning method to eliminate temporal redundancy of the video modality. Experimental results on multiple online and offline video understanding benchmarks demonstrate that our method significantly improves the accuracy (i.e., by 4\% at most) while incurring a negligible pruning latency (i.e., less than 1ms). Our full implementation will be made publicly available.

---

## 74. VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.12360v1](http://arxiv.org/abs/2512.12360v1)

**作者:** Yufei Yin, Qianke Meng, Minghao Chen, Jiajun Ding, Zhenwei Shao, Zhou Yu

**发布时间:** 2025-12-13

### GPT解析

### 总结

本研究提出了VideoARM，一种用于长视频理解的基于智能体推理-分层记忆的范式。该系统通过自适应、实时的智能体推理和记忆构建，克服了现有方法依赖手工推理流程和消耗大量令牌预处理的问题。VideoARM通过观察-思考-行动-记忆的循环，结合分层多模态记忆，显著提高了长视频理解的效率和性能。

### 背景

长视频理解因其扩展的时间结构和密集的多模态线索而具有挑战性。尽管最近有所进展，但许多现有方法仍依赖手工制作的推理流程或消耗大量令牌的视频预处理来引导多模态大型语言模型(MLLMs)进行自主推理。

### 目的

克服现有方法的局限性，提出一种更高效的长视频理解方法，减少令牌消耗并提高性能。

### 方法

VideoARM执行自适应、实时的智能体推理和记忆构建，而非静态的详尽预处理。具体实现包括：(1)观察-思考-行动-记忆的自适应连续循环；(2)控制器自主调用工具以粗到细方式解释视频；(3)分层多模态记忆持续捕获和更新多级线索，为决策提供精确上下文。

### 主要发现

在主流基准测试上，VideoARM优于最先进的方法DVD，同时显著减少了长视频的令牌消耗，实现了更高的效率和性能。

### 结论

VideoARM通过自适应智能体推理和分层记忆构建，有效解决了长视频理解中的挑战，在性能和效率方面均优于现有方法，为长视频理解提供了新的有效范式。

### 翻译

长视频理解由于其扩展的时间结构和密集的多模态线索仍然具有挑战性。尽管最近有所进展，但许多现有方法仍然依赖手工制作的推理流程或采用消耗大量令牌的视频预处理来引导多模态大型语言模型进行自主推理。为了克服这些局限性，我们提出了VideoARM，一种用于长视频理解的基于智能体推理-分层记忆的范式。VideoARM执行自适应、实时的智能体推理和记忆构建，而不是静态的、详尽的预处理。具体来说，VideoARM执行观察、思考、行动和记忆的自适应连续循环，其中控制器自主调用工具以粗到细的方式解释视频，从而显著减少令牌消耗。同时，分层多模态记忆在整个智能体操作过程中持续捕获和更新多级线索，为控制器提供精确的上下文信息以支持决策。在主流基准测试上的实验表明，VideoARM优于最先进的方法DVD，同时显著减少了长视频的令牌消耗。


### 论文摘要

Long-form video understanding remains challenging due to the extended temporal structure and dense multimodal cues. Despite recent progress, many existing approaches still rely on hand-crafted reasoning pipelines or employ token-consuming video preprocessing to guide MLLMs in autonomous reasoning. To overcome these limitations, we introduce VideoARM, an Agentic Reasoning-over-hierarchical-Memory paradigm for long-form video understanding. Instead of static, exhaustive preprocessing, VideoARM performs adaptive, on-the-fly agentic reasoning and memory construction. Specifically, VideoARM performs an adaptive and continuous loop of observing, thinking, acting, and memorizing, where a controller autonomously invokes tools to interpret the video in a coarse-to-fine manner, thereby substantially reducing token consumption. In parallel, a hierarchical multimodal memory continuously captures and updates multi-level clues throughout the operation of the agent, providing precise contextual information to support the controller in decision-making. Experiments on prevalent benchmarks demonstrate that VideoARM outperforms the state-of-the-art method, DVD, while significantly reducing token consumption for long-form videos.

---

## 75. CORE: Contrastive Masked Feature Reconstruction on Graphs

**论文链接:** [http://arxiv.org/abs/2512.13235v1](http://arxiv.org/abs/2512.13235v1)

**作者:** Jianyuan Bo, Yuan Fang

**发布时间:** 2025-12-15

### GPT解析

### 总结

这篇论文研究了图自监督学习中生成式和对比式方法的关系，提出了一种新的CORE框架，整合了掩码特征重构和对比学习，在节点和图分类任务上取得了最先进的结果。

### 背景

图自监督学习领域中，生成式和对比式方法是两种主导方法。掩码特征重构(MFR)是一种生成式技术，模型通过自监督方式学习恢复被掩码节点的原始特征。

### 目的

基于MFR和图对比学习(GCL)都旨在最大化相似元素之间一致性的观察，探索这两种方法的整合，以增强图自监督学习性能。

### 方法

提出对比掩码特征重构(CORE)框架，将对比学习整合到MFR中。仅在掩码节点的原始特征和重构特征之间形成正样本对，鼓励编码器优先考虑上下文信息；同时利用掩码节点本身作为负样本，结合MFR的重构能力和GCL的判别能力。

### 主要发现

在特定条件下，MFR和节点级GCL的目标会收敛，表明这些方法互补而非根本不同；CORE框架在节点分类和图分类任务上显著优于MFR，达到最先进结果，超越GraphMAE和GraphMAE2分别高达2.80%和3.72%(节点分类)，以及3.82%和3.76%(图分类)。

### 结论

CORE框架通过整合对比学习和掩码特征重构，有效结合了两种方法的优点，在图自监督学习任务中取得了显著的性能提升。

### 翻译

在图自监督学习这一快速发展的领域中，生成式和对比式方法已成为两种主导方法。我们的研究专注于掩码特征重构(MFR)，这是一种生成式技术，模型通过自监督方式学习恢复被掩码节点的原始特征。我们观察到MFR和图对比学习(GCL)都旨在最大化相似元素之间的一致性。基于这一观察，我们揭示了一个新的理论见解：在特定条件下，尽管操作机制不同，MFR和节点级GCL的目标会收敛。这一理论连接表明这些方法互补而非根本不同，促使我们探索它们的整合以增强图自监督学习。我们的研究提出了对比掩码特征重构(CORE)，这是一种新的图自监督学习框架，将对比学习整合到MFR中。具体来说，我们仅在掩码节点的原始特征和重构特征之间形成正样本对，鼓励编码器优先考虑上下文信息而非节点自身的特征。此外，我们利用掩码节点本身作为负样本，结合MFR的重构能力和GCL的判别能力，以更好地捕获图的内在结构。从实验上看，我们提出的CORE框架在节点分类和图分类任务上显著优于MFR，展示了最先进的结果。


### 论文摘要

In the rapidly evolving field of self-supervised learning on graphs, generative and contrastive methodologies have emerged as two dominant approaches. Our study focuses on masked feature reconstruction (MFR), a generative technique where a model learns to restore the raw features of masked nodes in a self-supervised manner. We observe that both MFR and graph contrastive learning (GCL) aim to maximize agreement between similar elements. Building on this observation, we reveal a novel theoretical insight: under specific conditions, the objectives of MFR and node-level GCL converge, despite their distinct operational mechanisms. This theoretical connection suggests these approaches are complementary rather than fundamentally different, prompting us to explore their integration to enhance self-supervised learning on graphs. Our research presents Contrastive Masked Feature Reconstruction (CORE), a novel graph self-supervised learning framework that integrates contrastive learning into MFR. Specifically, we form positive pairs exclusively between the original and reconstructed features of masked nodes, encouraging the encoder to prioritize contextual information over the node's own features. Additionally, we leverage the masked nodes themselves as negative samples, combining MFR's reconstructive power with GCL's discriminative ability to better capture intrinsic graph structures. Empirically, our proposed framework CORE significantly outperforms MFR across node and graph classification tasks, demonstrating state-of-the-art results. In particular, CORE surpasses GraphMAE and GraphMAE2 by up to 2.80% and 3.72% on node classification tasks, and by up to 3.82% and 3.76% on graph classification tasks.

---

## 76. A Semantically Enhanced Generative Foundation Model Improves Pathological Image Synthesis

**论文链接:** [http://arxiv.org/abs/2512.13164v1](http://arxiv.org/abs/2512.13164v1)

**作者:** Xianchao Guan, Zhiyuan Fan, Yifeng Wang, Fuqiang Chen, Yanjiang Zhou, Zengyang Che, Hongxue Meng, Xin Li, Yaowei Wang, Hongpeng Wang, Min Zhang, Heng Tao Shen, Zheng Zhang, Yongbing Zhang

**发布时间:** 2025-12-15

**备注:** 67 pages, 9 figures, 16 tables

### GPT解析

### 总结

CRAFTS是首个用于病理学特定文本到图像合成的生成式基础模型，通过双阶段训练和新型对齐机制解决数据稀缺问题，提供多样化、高质量病理图像数据源。

### 背景

临床级病理学人工智能发展受限于多样化、高质量标注数据集稀缺；生成式模型虽有潜力但存在语义不稳定和形态幻觉问题，影响诊断可靠性。

### 目的

解决病理学AI发展中的数据稀缺和隐私问题，创建多样化、标注组织学数据的无限来源，开发罕见和复杂癌症表型的强大诊断工具。

### 方法

引入CRAFTS框架，利用约280万对图像-标题对进行双阶段训练，采用新型对齐机制抑制语义漂移确保生物学准确性，并与ControlNet结合实现对组织架构的精确控制。

### 主要发现

CRAFTS生成了涵盖30种癌症类型的多样化病理图像，质量通过客观指标和病理学家评估验证；CRAFTS增强的数据集提高了分类、跨模态检索、自监督学习和视觉问答等临床任务性能。

### 结论

CRAFTS克服了数据稀缺和隐私问题的关键障碍，为病理学AI发展提供了可靠的数据来源，有效解锁了罕见和复杂癌症表型的诊断工具创建。

### 翻译

病理学临床级人工智能的发展受到多样化、高质量标注数据集稀缺的限制。生成式模型提供了潜在的解决方案，但存在语义不稳定和形态幻觉问题，影响诊断可靠性。为了应对这一挑战，我们引入了组织合成的相关性调节对齐框架（CRAFTS），这是首个用于病理学特定文本到图像合成的生成式基础模型。通过利用约280万对图像-标题对进行双阶段训练，CRAFTS采用了一种新型对齐机制，抑制语义漂移以确保生物学准确性。该模型生成了涵盖30种癌症类型的多样化病理图像，其质量通过客观指标和病理学家评估得到严格验证。此外，CRAFTS增强的数据集提高了各种临床任务的性能，包括分类、跨模态检索、自监督学习和视觉问答。此外，将CRAFTS与ControlNet结合，可以根据核分割掩码和荧光图像等输入精确控制组织架构。通过克服数据稀缺和隐私问题的关键障碍，CRAFTS提供了多样化、标注组织学数据的无限来源，有效解锁了罕见和复杂癌症表型的强大诊断工具的创建。


### 论文摘要

The development of clinical-grade artificial intelligence in pathology is limited by the scarcity of diverse, high-quality annotated datasets. Generative models offer a potential solution but suffer from semantic instability and morphological hallucinations that compromise diagnostic reliability. To address this challenge, we introduce a Correlation-Regulated Alignment Framework for Tissue Synthesis (CRAFTS), the first generative foundation model for pathology-specific text-to-image synthesis. By leveraging a dual-stage training strategy on approximately 2.8 million image-caption pairs, CRAFTS incorporates a novel alignment mechanism that suppresses semantic drift to ensure biological accuracy. This model generates diverse pathological images spanning 30 cancer types, with quality rigorously validated by objective metrics and pathologist evaluations. Furthermore, CRAFTS-augmented datasets enhance the performance across various clinical tasks, including classification, cross-modal retrieval, self-supervised learning, and visual question answering. In addition, coupling CRAFTS with ControlNet enables precise control over tissue architecture from inputs such as nuclear segmentation masks and fluorescence images. By overcoming the critical barriers of data scarcity and privacy concerns, CRAFTS provides a limitless source of diverse, annotated histology data, effectively unlocking the creation of robust diagnostic tools for rare and complex cancer phenotypes.

---

## 77. BUT Systems for WildSpoof Challenge: SASV in the Wild

**论文链接:** [http://arxiv.org/abs/2512.12851v1](http://arxiv.org/abs/2512.12851v1)

**作者:** Junyi Peng, Jin Li, Johan Rohdin, Lin Zhang, Miroslav Hlaváček, Oldrich Plchot

**发布时间:** 2025-12-14

**备注:** 4 pages

### GPT解析

### 总结

本文介绍了BUT团队在WildSpoof挑战中的提交方案，专注于Spoofing-robust Automatic Speaker Verification (SASV)赛道。

### 背景

WildSpoof挑战中的SASV赛道，关注语音欺骗检测和自动说话人验证。

### 目的

设计一个能够抵抗欺骗的自动说话人验证系统，提高系统对未知神经声码器和录制环境变化的鲁棒性。

### 方法

提出一种SASV框架，集成多种自监督学习前端（从通用音频模型如Dasheng到语音特定编码器如WavLM），使用轻量级多头因子化注意力后端聚合表示，引入基于分布不确定性的特征域增强策略减轻域偏移，并将鲁棒的CM分数与先进ASV系统融合。

### 主要发现

所提出的方法在最小化a-DCFs和EERs方面取得了优越的性能。

### 结论

SASV框架和特征域增强策略有效提高了系统对欺骗攻击的鲁棒性。

### 翻译

本文介绍了BUT团队在WildSpoof挑战中的提交方案，专注于Spoofing-robust Automatic Speaker Verification (SASV)赛道。我们提出了一种SASV框架，旨在弥合通用音频理解和专业语音分析之间的差距。我们的子系统集成了多种自监督学习前端，从通用音频模型（如Dasheng）到语音特定编码器（如WavLM）。这些表示通过轻量级多头因子化注意力后端进行聚合，以完成相应的子任务。此外，我们引入了一种基于分布不确定性的特征域增强策略，以明确建模并减轻由未知神经声码器和录制环境引起的域偏移。通过将这些鲁棒的CM分数与最先进的ASV系统融合，我们的方法在最小化a-DCFs和EERs方面取得了优越的性能。


### 论文摘要

This paper presents the BUT submission to the WildSpoof Challenge, focusing on the Spoofing-robust Automatic Speaker Verification (SASV) track. We propose a SASV framework designed to bridge the gap between general audio understanding and specialized speech analysis. Our subsystem integrates diverse Self-Supervised Learning front-ends ranging from general audio models (e.g., Dasheng) to speech-specific encoders (e.g., WavLM). These representations are aggregated via a lightweight Multi-Head Factorized Attention back-end for corresponding subtasks. Furthermore, we introduce a feature domain augmentation strategy based on Distribution Uncertainty to explicitly model and mitigate the domain shift caused by unseen neural vocoders and recording environments. By fusing these robust CM scores with state-of-the-art ASV systems, our approach achieves superior minimization of the a-DCFs and EERs.

---

## 78. $β$-CLIP: Text-Conditioned Contrastive Learning for Multi-Granular Vision-Language Alignment

**论文链接:** [http://arxiv.org/abs/2512.12678v1](http://arxiv.org/abs/2512.12678v1)

**作者:** Fatimah Zohra, Chen Zhao, Hani Itani, Bernard Ghanem

**发布时间:** 2025-12-14

### GPT解析

### 总结

本文提出β-CLIP，一种多粒度文本条件对比学习框架，通过实现多层次文本粒度与对应视觉区域之间的层次对齐，显著改善了CLIP在细粒度任务上的表现，达到了最先进的密集对齐效果。

### 背景

CLIP模型通过全局视觉和文本表示的对比学习，在零样本图像-文本检索任务中表现出色，但在细粒度任务上表现不佳，即使使用长而详细的标题进行微调也是如此。

### 目的

设计一个多粒度文本条件对比学习框架，实现从完整标题到句子和短语的多层次文本粒度与其对应视觉区域之间的层次对齐。

### 方法

β-CLIP为每个粒度级别使用交叉注意力动态池化图像块，生成上下文化的视觉嵌入。为解决层次结构中固有的语义重叠问题，引入了β-上下文化对比对齐损失(β-CAL)，该目标参数化了特定查询的严格匹配与松弛的图像内上下文化之间的权衡，支持软交叉熵和硬二元交叉熵两种形式。

### 主要发现

通过大量实验，β-CLIP显著改善了密集对齐：在Urban1K上达到91.8%的T2I和92.3%的I2T@R1，在FG-OVD(Hard)上达到30.9%，在不使用硬负样本训练的方法中达到了最先进水平。

### 结论

β-CLIP为密集视觉-语言对应关系建立了强大且自适应的基线。代码和模型已在GitHub上发布。

### 翻译

CLIP通过对齐全局视觉和文本表示，在零样本图像-文本检索中取得了强大性能，然而即使在长详细标题上微调后，它在细粒度任务上仍然表现不佳。在这项工作中，我们提出β-CLIP，一种多粒度文本条件对比学习框架，旨在实现从完整标题到句子和短语的多层次文本粒度与其对应视觉区域之间的层次对齐。对于每个粒度级别，β-CLIP使用交叉注意力动态池化图像块，生成上下文化的视觉嵌入。为解决此层次结构中固有的语义重叠问题，我们引入了β-上下文化对比对齐损失(β-CAL)。该目标参数化了特定查询的严格匹配与松弛的图像内上下文化之间的权衡，支持软交叉熵和硬二元交叉熵两种形式。通过大量实验，我们证明β-CLIP显著改善了密集对齐：在Urban1K上达到91.8%的T2I和92.3%的I2T@R1，在FG-OVD(Hard)上达到30.9%，在不使用硬负样本训练的方法中建立了最先进的水平。β-CLIP为密集视觉-语言对应关系建立了强大且自适应的基线。代码和模型已在https://github.com/fzohra/B-CLIP发布。


### 论文摘要

CLIP achieves strong zero-shot image-text retrieval by aligning global vision and text representations, yet it falls behind on fine-grained tasks even when fine-tuned on long, detailed captions. In this work, we propose $β$-CLIP, a multi-granular text-conditioned contrastive learning framework designed to achieve hierarchical alignment between multiple textual granularities-from full captions to sentences and phrases-and their corresponding visual regions. For each level of granularity, $β$-CLIP utilizes cross-attention to dynamically pool image patches, producing contextualized visual embeddings. To address the semantic overlap inherent in this hierarchy, we introduce the $β$-Contextualized Contrastive Alignment Loss ($β$-CAL). This objective parameterizes the trade-off between strict query-specific matching and relaxed intra-image contextualization, supporting both soft Cross-Entropy and hard Binary Cross-Entropy formulations. Through extensive experiments, we demonstrate that $β$-CLIP significantly improves dense alignment: achieving 91.8% T2I 92.3% I2T at R@1 on Urban1K and 30.9% on FG-OVD (Hard), setting state-of-the-art among methods trained without hard negatives. $β$-CLIP establishes a robust, adaptive baseline for dense vision-language correspondence. The code and models are released at https://github.com/fzohra/B-CLIP.

---

## 79. Noise-robust Contrastive Learning for Critical Transition Detection in Dynamical Systems

**论文链接:** [http://arxiv.org/abs/2512.12523v1](http://arxiv.org/abs/2512.12523v1)

**作者:** Wenqi Fang, Ye Li

**发布时间:** 2025-12-14

**备注:** under revision

### GPT解析

### 总结

提出了一种基于奇异值分解和半正交约束训练的神经网络架构，用于在复杂嘈杂的时间序列数据中检测临界转换。

### 背景

在复杂、嘈杂的时间序列数据中检测临界转换是科学与工程领域的基础性挑战。临界转换可能通过低维序参量的出现来预测，但其特征通常被高幅度随机变异性所掩盖。

### 目的

解决传统对比学习方法参数过多、对噪声敏感的问题，提高临界点识别的准确性。

### 方法

提出一种使用奇异值分解技术构建的神经网络架构，结合严格的半正交约束训练算法，以增强传统对比学习的性能。

### 主要发现

所提出的方法在识别临界转换方面与传统对比学习技术性能相当，但明显更轻量级，并且对噪声的抗性显著更强。

### 结论

通过奇异值分解和半正交约束训练可以有效改进对比学习方法，使其在检测临界转换时更加高效和鲁棒。

### 翻译

在复杂、嘈杂的时间序列数据中检测临界转换是科学与工程领域的基础性挑战。此类临界转换可能通过低维序参量的出现来预测，但其特征通常被高幅度随机变异性所掩盖。基于深度神经网络的对比学习方法虽然对检测临界转换有前景，但通常参数过多，对无关噪声敏感，导致临界点识别不准确。为解决这些局限性，我们提出了一种使用奇异值分解技术构建的神经网络架构，结合严格的半正交约束训练算法，以增强传统对比学习的性能。大量实验表明，所提出的方法在识别临界转换方面与传统对比学习技术性能相当，但明显更轻量级，并且对噪声的抗性显著更强。


### 论文摘要

Detecting critical transitions in complex, noisy time-series data is a fundamental challenge across science and engineering. Such transitions may be anticipated by the emergence of a low-dimensional order parameter, whose signature is often masked by high-amplitude stochastic variability. Standard contrastive learning approaches based on deep neural networks, while promising for detecting critical transitions, are often overparameterized and sensitive to irrelevant noise, leading to inaccurate identification of critical points. To address these limitations, we propose a neural network architecture, constructed using singular value decomposition technique, together with a strictly semi-orthogonality-constrained training algorithm, to enhance the performance of traditional contrastive learning. Extensive experiments demonstrate that the proposed method matches the performance of traditional contrastive learning techniques in identifying critical transitions, yet is considerably more lightweight and markedly more resistant to noise.

---

## 80. MetaHGNIE: Meta-Path Induced Hypergraph Contrastive Learning in Heterogeneous Knowledge Graphs

**论文链接:** [http://arxiv.org/abs/2512.12477v1](http://arxiv.org/abs/2512.12477v1)

**作者:** Jiawen Chen, Yanyan He, Qi Shao, Mengli Wei, Duxin Chen, Wenwu Yu, Yanlong Zhao

**发布时间:** 2025-12-13

### GPT解析

### 总结

论文提出了MetaHGNIE，一个基于元路径诱导的超图对比学习框架，用于异构知识图中的节点重要性估计。该方法通过显式建模高阶交互和跨模态对齐，解决了现有方法忽视高阶依赖性和独立处理结构与语义信号的问题。

### 背景

异构知识图中的节点重要性估计(NIE)是一个关键且具有挑战性的任务，对于推荐、知识推理和问答等应用至关重要。现有方法通常依赖于成对连接，忽视了多个实体和关系之间的高阶依赖性，并且将结构和语义信号独立处理，阻碍了有效的跨模态集成。

### 目的

解决现有NIE方法中忽视高阶依赖性和独立处理结构与语义信号的问题，通过显式建模高阶交互和跨模态对齐来提高异构知识图中节点重要性估计的性能。

### 方法

MetaHGNIE是一个元路径诱导的超图对比学习框架，用于解耦和对齐结构信息与语义信息。该方法通过元路径序列构建更高阶的知识图，其中类型化的超边捕获多实体关系上下文。结构依赖性通过局部注意力进行聚合，而语义表示则通过配备稀疏分块的超图transformer进行编码以减少冗余。最后，一个多模态融合模块在对比学习和辅助监督下集成结构和语义嵌入，确保鲁棒的跨模态对齐。

### 主要发现

在基准NIE数据集上的大量实验表明，MetaHGNIE始终优于最先进的基线方法。这些结果突显了在异构知识图中显式建模高阶交互和跨模态对齐的有效性。

### 结论

通过显式建模高阶交互和跨模态对齐，MetaHGNIE在异构知识图节点重要性估计任务中取得了显著性能提升，为相关应用提供了更有效的解决方案。

### 翻译

异构知识图中的节点重要性估计(NIE)是一个关键且具有挑战性的任务，对于推荐、知识推理和问答等应用至关重要。现有方法通常依赖于成对连接，忽视了多个实体和关系之间的高阶依赖性，并且将结构和语义信号独立处理，阻碍了有效的跨模态集成。为了解决这些挑战，我们提出了MetaHGNIE，一个用于解耦和对齐结构信息与语义信息的元路径诱导超图对比学习框架。MetaHGNIE通过元路径序列构建更高阶的知识图，其中类型化的超边捕获多实体关系上下文。结构依赖性通过局部注意力进行聚合，而语义表示则通过配备稀疏分块的超图transformer进行编码以减少冗余。最后，一个多模态融合模块在对比学习和辅助监督下集成结构和语义嵌入，确保鲁棒的跨模态对齐。在基准NIE数据集上的大量实验表明，MetaHGNIE始终优于最先进的基线方法。这些结果突显了在异构知识图中显式建模高阶交互和跨模态对齐的有效性。我们的代码可在https://github.com/SEU-WENJIA/DualHNIE获取。


### 论文摘要

Node importance estimation (NIE) in heterogeneous knowledge graphs is a critical yet challenging task, essential for applications such as recommendation, knowledge reasoning, and question answering. Existing methods often rely on pairwise connections, neglecting high-order dependencies among multiple entities and relations, and they treat structural and semantic signals independently, hindering effective cross-modal integration. To address these challenges, we propose MetaHGNIE, a meta-path induced hypergraph contrastive learning framework for disentangling and aligning structural and semantic information. MetaHGNIE constructs a higher-order knowledge graph via meta-path sequences, where typed hyperedges capture multi-entity relational contexts. Structural dependencies are aggregated with local attention, while semantic representations are encoded through a hypergraph transformer equipped with sparse chunking to reduce redundancy. Finally, a multimodal fusion module integrates structural and semantic embeddings under contrastive learning with auxiliary supervision, ensuring robust cross-modal alignment. Extensive experiments on benchmark NIE datasets demonstrate that MetaHGNIE consistently outperforms state-of-the-art baselines. These results highlight the effectiveness of explicitly modeling higher-order interactions and cross-modal alignment in heterogeneous knowledge graphs. Our code is available at https://github.com/SEU-WENJIA/DualHNIE

---

## 81. Knowledge-Guided Masked Autoencoder with Linear Spectral Mixing and Spectral-Angle-Aware Reconstruction

**论文链接:** [http://arxiv.org/abs/2512.12445v1](http://arxiv.org/abs/2512.12445v1)

**作者:** Abdul Matin, Rupasree Dey, Tanjim Bin Faruk, Shrideep Pallickara, Sangmi Lee Pallickara

**发布时间:** 2025-12-13

### GPT解析

### 总结

本文提出了一种新颖的知识引导的ViT-based Masked Autoencoder，通过将科学领域知识嵌入自监督重建过程，结合线性光谱混合模型和光谱角映射器作为物理约束，提高了模型的可解释性、泛化能力和数据效率。

### 背景

将领域知识整合到深度学习中是提高模型可解释性、泛化能力和数据效率的有前景的方向。

### 目的

提出一种新颖的知识引导的ViT-based Masked Autoencoder，在自监督重建过程中嵌入科学领域知识，使学习到的表示符合观测信号与其潜在成分之间已知的结构关系。

### 方法

引入线性光谱混合模型作为物理约束，基于物理的光谱角映射器，确保学习到的表示符合观测信号与其潜在成分之间已知的结构关系。框架联合优化LSMM和SAM损失与传统的Huber损失目标，促进特征空间中的数值精度和几何一致性。

### 主要发现

这种知识引导的设计增强了重建保真度，在有限监督下稳定训练，并产生基于物理原理的可解释潜在表示。实验结果表明，所提出的模型显著提高了重建质量并改善了下游任务性能。

### 结论

在基于transformer的自监督学习中嵌入物理归纳偏见具有很大潜力，能够有效提升模型性能和可解释性。

### 翻译

将领域知识整合到深度学习中已成为提高模型可解释性、泛化能力和数据效率的一个有前景的方向。在这项工作中，我们提出了一种新颖的知识引导的ViT-based Masked Autoencoder，在自监督重建过程中嵌入科学领域知识。除了依赖数据驱动的优化外，我们提出的方法将线性光谱混合模型作为物理约束和基于物理的光谱角映射器纳入其中，确保学习到的表示符合观测信号与其潜在成分之间已知的结构关系。该框架联合优化LSMM和SAM损失与传统的Huber损失目标，促进特征空间中的数值精度和几何一致性。这种知识引导的设计增强了重建保真度，在有限监督下稳定训练，并产生基于物理原理的可解释潜在表示。实验结果表明，所提出的模型显著提高了重建质量并改善了下游任务性能，强调了在基于transformer的自监督学习中嵌入物理归纳偏见的潜力。


### 论文摘要

Integrating domain knowledge into deep learning has emerged as a promising direction for improving model interpretability, generalization, and data efficiency. In this work, we present a novel knowledge-guided ViT-based Masked Autoencoder that embeds scientific domain knowledge within the self-supervised reconstruction process. Instead of relying solely on data-driven optimization, our proposed approach incorporates the Linear Spectral Mixing Model (LSMM) as a physical constraint and physically-based Spectral Angle Mapper (SAM), ensuring that learned representations adhere to known structural relationships between observed signals and their latent components. The framework jointly optimizes LSMM and SAM loss with a conventional Huber loss objective, promoting both numerical accuracy and geometric consistency in the feature space. This knowledge-guided design enhances reconstruction fidelity, stabilizes training under limited supervision, and yields interpretable latent representations grounded in physical principles. The experimental findings indicate that the proposed model substantially enhances reconstruction quality and improves downstream task performance, highlighting the promise of embedding physics-informed inductive biases within transformer-based self-supervised learning.

---

## 82. CLOAK: Contrastive Guidance for Latent Diffusion-Based Data Obfuscation

**论文链接:** [http://arxiv.org/abs/2512.12086v1](http://arxiv.org/abs/2512.12086v1)

**作者:** Xin Yang, Omid Ardakanian

**发布时间:** 2025-12-12

### GPT解析

### 总结

Cloak是一种基于潜在扩散模型的新型数据混淆框架，通过对比学习提取解耦表示来平衡隐私保护和数据效用，在资源受限环境中表现出色。

### 背景

数据混淆是缓解半可信方对传感器时间序列数据进行属性推断攻击的有效技术，现有方法利用条件生成模型结合对抗训练或互信息正则化来平衡隐私与效用。

### 目的

开发一种无需修改下游任务、计算效率高且能灵活调整隐私-效用权衡的数据混淆框架，使其适合在资源受限的移动物联网设备上部署。

### 方法

提出Cloak框架，基于潜在扩散模型，采用对比学习提取解耦表示，指导扩散过程保留有用信息同时隐藏私人信息，使用户能根据隐私需求调整保护级别。

### 主要发现

在四个时间序列数据集和一个面部图像数据集上的实验表明，Cloak在隐私保护和数据效用方面均优于现有技术，且适合资源受限环境部署。

### 结论

Cloak为时间序列数据提供了高效实用的数据混淆解决方案，解决了现有方法在部署灵活性和计算效率方面的局限性。

### 翻译

数据混淆是一种有前途的技术，可通过半可信方访问的传感器发出的时间序列数据来缓解属性推断攻击。最近的进展利用条件生成模型结合对抗训练或基于互信息的正则化来平衡数据隐私和效用。然而，这些方法通常需要修改下游任务，难以实现令人满意的隐私-效用权衡，或者计算密集，使得它们难以在资源受限的移动物联网设备上部署。我们提出Cloak，一种基于潜在扩散模型的新型数据混淆框架。与先前工作不同，我们采用对比学习来提取解耦表示，这些表示指导潜在扩散过程保留有用信息同时隐藏私人信息。这种方法使具有不同隐私需求的用户能够以最少的再训练来导航隐私-效用权衡。在四个公共时间序列数据集（涵盖多种传感模态）和一个面部图像数据集上的广泛实验表明，Cloak始终优于最先进的混淆技术，并且适合在资源受限的环境中部署。


### 论文摘要

Data obfuscation is a promising technique for mitigating attribute inference attacks by semi-trusted parties with access to time-series data emitted by sensors. Recent advances leverage conditional generative models together with adversarial training or mutual information-based regularization to balance data privacy and utility. However, these methods often require modifying the downstream task, struggle to achieve a satisfactory privacy-utility trade-off, or are computationally intensive, making them impractical for deployment on resource-constrained mobile IoT devices. We propose Cloak, a novel data obfuscation framework based on latent diffusion models. In contrast to prior work, we employ contrastive learning to extract disentangled representations, which guide the latent diffusion process to retain useful information while concealing private information. This approach enables users with diverse privacy needs to navigate the privacy-utility trade-off with minimal retraining. Extensive experiments on four public time-series datasets, spanning multiple sensing modalities, and a dataset of facial images demonstrate that Cloak consistently outperforms state-of-the-art obfuscation techniques and is well-suited for deployment in resource-constrained settings.

---

## 83. Towards Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification

**论文链接:** [http://arxiv.org/abs/2512.12070v1](http://arxiv.org/abs/2512.12070v1)

**作者:** Jie Ma, Junqing Zhang, Guanxiong Shen, Linning Peng, Alan Marshall

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出了一种三阶段射频指纹识别方法，用于物联网设备认证，能够有效缓解信道和接收效应的影响，同时减少训练数据需求。

### 背景

射频指纹识别(RFFI)是一种新兴的物联网设备认证方法，利用设备固有的硬件缺陷进行分类。基于深度学习的RFFI表现良好，但仍面临公共训练数据集有限以及信道和接收效应影响等挑战。

### 目的

提出一种能够有效缓解信道和接收效应影响的三阶段RFFI方法，减少对大量训练数据的依赖。

### 方法

提出了一种三阶段RFFI方法：(1)对比学习增强的预训练；(2)基于Siamese网络的分类网络训练；(3)推理阶段。具体技术包括使用频谱图作为信号表示以分离发射机缺陷，提出无监督对比学习方法预训练信道鲁棒的RFF提取器，并通过数据增强和对比损失增强Siamese网络方案。

### 主要发现

所提出的方法能够有效同时缓解信道和接收效应的影响；预训练可以显著减少微调所需的数据量；在动态非视距场景下，每台设备仅有20个数据包时，准确率超过90%。

### 结论

提出的三阶段RFFI方法在减少数据需求的同时，能够有效应对信道和接收效应的挑战，在动态非视距场景下仍保持高准确率。

### 翻译

射频指纹识别(RFFI)是一种用于认证物联网设备的 emerging 方法。RFFI 利用固有的、独特的硬件缺陷来分类物联网设备。基于深度学习的 RFFI 已表现出优异的性能。然而，仍然存在一些研究挑战，如公共训练数据集有限以及信道和接收效应的影响。在本文中，我们提出了一种三阶段 RFFI 方法，包括对比学习增强的预训练、基于 Siamese 网络的分类网络训练和推理阶段。具体来说，我们采用频谱图作为信号表示，以将发射机缺陷与信道效应和接收机效应解耦。我们提出了一种无监督对比学习方法来预训练信道鲁棒的 RFF 提取器。此外，基于 Siamese 网络的方案通过数据增强和对比损失得到增强，能够共同缓解信道和接收机效应的影响。我们使用三个公共 LoRa 数据集和一个自收集的 LoRa 数据集进行了全面的实验评估。结果表明，我们的方法能够有效且同时缓解信道和接收机效应的影响。我们还表明，预训练可以显著减少微调所需的数据量。在每台设备仅有 20 个数据包的动态非视距(NLOS)场景下，我们提出的方法实现了超过 90% 的准确率。


### 论文摘要

Radio frequency fingerprint identification (RFFI) is an emerging method for authenticating Internet of Things (IoT) devices. RFFI exploits the intrinsic and unique hardware imperfections for classifying IoT devices. Deep learning-based RFFI has shown excellent performance. However, there are still remaining research challenges, such as limited public training datasets as well as impacts of channel and receive effects. In this paper, we proposed a three-stage RFFI approach involving contrastive learning-enhanced pretraining, Siamese network-based classification network training, and inference. Specifically, we employed spectrogram as signal representation to decouple the transmitter impairments from channel effects and receiver impairments. We proposed an unsupervised contrastive learning method to pretrain a channel-robust RFF extractor. In addition, the Siamese network-based scheme is enhanced by data augmentation and contrastive loss, which is capable of jointly mitigating the effects of channel and receiver impairments. We carried out a comprehensive experimental evaluation using three public LoRa datasets and one self-collected LoRa dataset. The results demonstrated that our approach can effectively and simultaneously mitigate the effects of channel and receiver impairments. We also showed that pretraining can significantly reduce the required amount of the fine-tuning data. Our proposed approach achieved an accuracy of over 90% in dynamic non-line-of-sight (NLOS) scenarios when there are only 20 packets per device.

---

## 84. SCR2-ST: Combine Single Cell with Spatial Transcriptomics for Efficient Active Sampling via Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.13635v1](http://arxiv.org/abs/2512.13635v1)

**作者:** Junchao Zhu, Ruining Deng, Junlin Guo, Tianyuan Yao, Chongyu Qu, Juming Xiong, Siqi Lu, Zhengyi Lu, Yanfan Zhu, Marilyn Lionts, Yuechen Yang, Yalin Zheng, Yu Wang, Shilin Zhao, Haichun Yang, Yuankai Huo

**发布时间:** 2025-12-15

### GPT解析

### 总结

SCR2-ST是一个利用单细胞先验知识指导高效数据获取和准确表达预测的统一框架，结合了单细胞引导的强化学习主动采样和混合回归-检索预测网络。

### 背景

空间转录组学(ST)是一种新兴技术，可研究组织形态背后的分子关系，但获取ST数据成本高昂，传统固定网格采样策略会导致对形态相似或生物学信息不丰富区域的冗余测量，从而限制了当前方法。

### 目的

开发一种框架，利用单细胞测序提供的丰富生物学数据作为辅助来源，解决ST数据获取成本高和效率低的问题。

### 方法

SCR2-ST整合了单细胞引导的强化学习(SCRL)主动采样和混合回归-检索预测网络SCR2Net。SCRL结合单细胞基础模型嵌入和空间密度信息构建奖励信号，选择性获取信息丰富的组织区域；SCR2Net结合回归建模和检索增强推理，使用多数细胞类型过滤机制抑制噪声匹配，检索到的表达谱作为辅助监督的软标签。

### 主要发现

在三个公开ST数据集上评估显示，SCR2-ST在采样效率和预测准确性方面都达到了最先进的性能，特别是在低预算场景下表现优异。

### 结论

SCR2-ST框架有效地利用单细胞先验知识解决了ST数据获取成本高和效率低的问题，为空间转录组学研究提供了新的解决方案。

### 翻译

空间转录组学(ST)是一项新兴技术，使研究人员能够研究组织形态背后的分子关系。然而，获取ST数据仍然成本高昂，传统的固定网格采样策略导致对形态相似或生物学信息不丰富的区域的冗余测量，从而限制了当前方法的数据稀缺问题。然而，成熟且完善的单细胞测序领域可以提供丰富的生物学数据作为有效的辅助来源来缓解这一限制。为了弥合这些差距，我们引入了SCR2-ST，这是一个利用单细胞先验知识指导高效数据获取和准确表达预测的统一框架。SCR2-ST整合了基于单细胞引导的强化学习(SCRL)主动采样和混合回归-检索预测网络SCR2Net。SCRL结合单细胞基础模型嵌入和空间密度信息构建生物学基础的奖励信号，在有限的测序预算下能够选择性获取信息丰富的组织区域。SCR2Net然后通过结合基于回归的建模和检索增强推理的混合架构，利用主动采样的数据，其中多数细胞类型过滤机制抑制噪声匹配，检索到的表达谱作为辅助监督的软标签。我们在三个公共ST数据集上评估了SCR2-ST，证明了其在采样效率和预测准确性方面都达到了最先进的性能，特别是在低预算场景下。代码已在GitHub上公开：https://github.com/hrlblab/SCR2ST


### 论文摘要

Spatial transcriptomics (ST) is an emerging technology that enables researchers to investigate the molecular relationships underlying tissue morphology. However, acquiring ST data remains prohibitively expensive, and traditional fixed-grid sampling strategies lead to redundant measurements of morphologically similar or biologically uninformative regions, thus resulting in scarce data that constrain current methods. The well-established single-cell sequencing field, however, could provide rich biological data as an effective auxiliary source to mitigate this limitation. To bridge these gaps, we introduce SCR2-ST, a unified framework that leverages single-cell prior knowledge to guide efficient data acquisition and accurate expression prediction. SCR2-ST integrates a single-cell guided reinforcement learning-based (SCRL) active sampling and a hybrid regression-retrieval prediction network SCR2Net. SCRL combines single-cell foundation model embeddings with spatial density information to construct biologically grounded reward signals, enabling selective acquisition of informative tissue regions under constrained sequencing budgets. SCR2Net then leverages the actively sampled data through a hybrid architecture combining regression-based modeling with retrieval-augmented inference, where a majority cell-type filtering mechanism suppresses noisy matches and retrieved expression profiles serve as soft labels for auxiliary supervision. We evaluated SCR2-ST on three public ST datasets, demonstrating SOTA performance in both sampling efficiency and prediction accuracy, particularly under low-budget scenarios. Code is publicly available at: https://github.com/hrlblab/SCR2ST

---

## 85. DBT-DINO: Towards Foundation model based analysis of Digital Breast Tomosynthesis

**论文链接:** [http://arxiv.org/abs/2512.13608v1](http://arxiv.org/abs/2512.13608v1)

**作者:** Felix J. Dorfner, Manon A. Dorster, Ryan Connolly, Oscar Gentilhomme, Edward Gibbs, Steven Graham, Seth Wander, Thomas Schultz, Manisha Bahl, Dania Daye, Albert E. Kim, Christopher P. Bridge

**发布时间:** 2025-12-15

### GPT解析

### 总结

研究开发了首个数字断层合成乳腺成像(DBT)基础模型DBT-DINO，并在乳腺密度分类、5年乳腺癌风险预测和病灶检测三个临床任务上进行了评估。DBT-DINO在前两项任务上表现优于基线模型，但在病灶检测任务上表现不一，需要进一步方法改进。

### 背景

基础模型在医学影像领域显示出潜力，但在三维成像模态中研究不足。目前尚无针对数字断层合成乳腺成像(DBT)的基础模型，尽管DBT已被用于乳腺癌筛查。

### 目的

开发并评估一个用于DBT的基础模型(DBT-DINO)在多个临床任务上的表现，并评估领域特定预训练的影响。

### 方法

使用DINOv2方法进行自监督预训练，数据包括超过2500万个来自27,990名患者的487,975个体积的2D切片。评估三个下游任务：乳腺密度分类(5,000次筛查检查)、5年乳腺癌发展风险预测(106,417次筛查检查)和病灶检测(393个注释体积)。

### 主要发现

乳腺密度分类：DBT-DINO准确率为0.79，优于MetaAI DINOv2基线(0.73)和DenseNet-121(0.74)。5年乳腺癌风险预测：DBT-DINO的AUROC为0.78，高于DINOv2的0.76。病灶检测：DINOv2的平均敏感性为0.67，高于DBT-DINO的0.62，但DBT-DINO在癌性病灶检测率上更高(78.8% vs 77.3%)。

### 结论

使用前所未有的数据集开发了DBT-DINO，这是首个DBT基础模型。DBT-DINO在乳腺密度分类和癌症风险预测方面表现出色，但领域特定预训练在检测任务上的益处有限，表明局部检测任务需要进一步的方法开发。

### 翻译

基础模型在医学影像领域显示出潜力，但在三维成像模态中研究不足。目前尚无针对数字断层合成乳腺成像(DBT)的基础模型，尽管DBT已被用于乳腺癌筛查。为开发并评估一个用于DBT的基础模型(DBT-DINO)在多个临床任务上的表现并评估领域特定预训练的影响，我们使用DINOv2方法进行了自监督预训练，数据包括超过2500万个来自27,990名患者的487,975个体积的2D切片。评估了三个下游任务：(1)使用5,000次筛查检查的乳腺密度分类；(2)使用106,417次筛查检查的5年乳腺癌发展风险预测；(3)使用393个注释体积的病灶检测。在乳腺密度分类方面，DBT-DINO的准确率为0.79，优于MetaAI DINOv2基线(0.73)和DenseNet-121(0.74)。在5年乳腺癌风险预测方面，DBT-DINO的AUROC为0.78，高于DINOv2的0.76。在病灶检测方面，DINOv2的平均敏感性为0.67，高于DBT-DINO的0.62，但DBT-DINO在癌性病灶检测率上更高(78.8% vs 77.3%)。使用前所未有的数据集，我们开发了DBT-DINO，这是首个DBT基础模型。DBT-DINO在乳腺密度分类和癌症风险预测方面表现出色。然而，领域特定预训练在检测任务上的益处不一，ImageNet基线在一般病灶检测方面优于DBT-DINO，表明局部检测任务需要进一步的方法开发。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决数字乳腺断层合成（DBT）缺乏专门基础模型的问题。DBT虽已用于乳腺癌筛查并比传统2D乳腺摄影效果更好，但其影像解读耗时更长，需要自动化工具提高效率。然而，开发这些工具需要大量带标注的数据，而基础模型可以通过在大量未标记数据上预训练，再在小数据集上微调，有效解决这一挑战，对提高乳腺癌筛查效率和准确性具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到DBT作为三维成像模态，其应用在基础模型领域尚不充分，且面临数据量大、图像各向异性、病灶信息局部化等挑战。他们借鉴了Meta AI的DINOv2自监督学习方法，从ImageNet预训练模型初始化，然后在大规模DBT数据集上继续预训练。设计上采用ViT作为骨干网络，针对不同下游任务（密度分类、风险预测、病灶检测）使用不同的特征聚合和任务特定头，同时探索了多种特征聚合策略以优化性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过在大量未标记的DBT图像上进行自监督预训练，学习通用的视觉特征表示，这些特征可迁移到多种临床任务中，减少对外部标注数据的依赖。整体流程包括：1）收集大规模DBT数据集（48万个体积，2.8万名患者）；2）使用DINOv2方法进行自监督预训练；3）针对三个下游任务（密度分类、风险预测、病灶检测）设计不同的特征聚合和预测头；4）评估模型性能并与ImageNet预训练基线比较。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）开发首个专门针对DBT的基础模型DBT-DINO；2）使用前所未有的大规模DBT数据集进行预训练；3）系统探索跨图像体积的特征聚合方法；4）在多个临床任务上全面评估模型；5）进行详细的偏差和公平性分析。相比之前工作，不同之处在于：专注于三维DBT而非2D乳腺摄影；采用自监督学习减少标注依赖；单一模型支持多任务应用；使用更大规模数据；更全面的评估包括公平性分析。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文开发了DBT-DINO，首个针对数字乳腺断层合成的基础模型，通过大规模自监督预训练实现了在乳腺密度分类和癌症风险预测等临床任务上的高性能表现，为DBT影像分析提供了高效、准确的自动化解决方案。'}


### 论文摘要

Foundation models have shown promise in medical imaging but remain underexplored for three-dimensional imaging modalities. No foundation model currently exists for Digital Breast Tomosynthesis (DBT), despite its use for breast cancer screening.   To develop and evaluate a foundation model for DBT (DBT-DINO) across multiple clinical tasks and assess the impact of domain-specific pre-training.   Self-supervised pre-training was performed using the DINOv2 methodology on over 25 million 2D slices from 487,975 DBT volumes from 27,990 patients. Three downstream tasks were evaluated: (1) breast density classification using 5,000 screening exams; (2) 5-year risk of developing breast cancer using 106,417 screening exams; and (3) lesion detection using 393 annotated volumes.   For breast density classification, DBT-DINO achieved an accuracy of 0.79 (95\% CI: 0.76--0.81), outperforming both the MetaAI DINOv2 baseline (0.73, 95\% CI: 0.70--0.76, p<.001) and DenseNet-121 (0.74, 95\% CI: 0.71--0.76, p<.001). For 5-year breast cancer risk prediction, DBT-DINO achieved an AUROC of 0.78 (95\% CI: 0.76--0.80) compared to DINOv2's 0.76 (95\% CI: 0.74--0.78, p=.57). For lesion detection, DINOv2 achieved a higher average sensitivity of 0.67 (95\% CI: 0.60--0.74) compared to DBT-DINO with 0.62 (95\% CI: 0.53--0.71, p=.60). DBT-DINO demonstrated better performance on cancerous lesions specifically with a detection rate of 78.8\% compared to Dinov2's 77.3\%.   Using a dataset of unprecedented size, we developed DBT-DINO, the first foundation model for DBT. DBT-DINO demonstrated strong performance on breast density classification and cancer risk prediction. However, domain-specific pre-training showed variable benefits on the detection task, with ImageNet baseline outperforming DBT-DINO on general lesion detection, indicating that localized detection tasks require further methodological development.

---

## 86. DA-SSL: self-supervised domain adaptor to leverage foundational models in turbt histopathology slides

**论文链接:** [http://arxiv.org/abs/2512.13600v1](http://arxiv.org/abs/2512.13600v1)

**作者:** Haoyue Zhang, Meera Chappidi, Erolcan Sayar, Helen Richards, Zhijun Chen, Lucas Liu, Roxanne Wadia, Peter A Humphrey, Fady Ghali, Alberto Contreras-Sanz, Peter Black, Jonathan Wright, Stephanie Harmon, Michael Haffner

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出了一种名为DA-SSL的域自适应自监督适配器，用于解决病理学基础模型在特定癌症类型上的局限性问题，特别是在膀胱肿瘤电切术样本中的应用。该方法能够在不修改基础模型的情况下重新对齐预训练特征，并在治疗反应预测中表现出色。

### 背景

深度学习框架，特别是多实例学习结合病理学基础模型在组织病理学中表现强大，但这些模型在某些癌症类型上存在局限性，这是由于域偏移造成的。膀胱肿瘤电切术样本含有组织碎片和电灼伪影，在公开可用的基础模型中未被广泛使用。

### 目的

开发一种方法使预训练的病理学基础模型特征能够适应膀胱肿瘤电切术域，而无需微调基础模型本身，并利用这些特征预测治疗反应，识别将从新辅助化疗中受益的患者。

### 方法

提出域自适应自监督适配器(DA-SSL)，重新对齐预训练的PFM特征到TURBT域。在多中心研究中使用五折交叉验证和外部测试评估性能，采用多数投票方法进行预测。

### 主要发现

DA-SSL在五折交叉验证中实现了0.77+/-0.04的AUC，外部测试准确率达到0.84，敏感性为0.71，特异性为0.91。轻量级域自适应与自监督能有效增强基于PFM的MIL流程。

### 结论

轻量级域自适应与自监督可有效增强基于病理学基础模型的多实例学习流程，用于解决具有临床挑战性的组织病理学任务，特别是在特定域适应方面。

### 翻译

最近的组织病理学深度学习框架，特别是多实例学习与病理学基础模型相结合，已展现出强大的性能。然而，PFMs在某些癌症类型或样本类型上存在局限性，这是由于域偏移造成的 - 这些癌症类型很少用于预训练，或者样本包含在预训练人群中很少见的基于组织的伪影。膀胱肿瘤电切术就是这种情况，它是诊断肌肉浸润性膀胱癌所必需的，但含有组织碎片和电灼伪影，并且在公开可用的PFMs中未被广泛使用。为了解决这个问题，我们提出了一种简单而有效的域自适应自监督适配器，它将预训练的PFM特征重新对齐到TURBT域，而不需要微调基础模型本身。我们在TURBT的治疗反应预测中试点了这个框架，其中组织形态学特征目前未被充分利用，并且识别将从新辅助化疗中受益的患者具有挑战性。在我们的多中心研究中，DA-SSL在五折交叉验证中实现了0.77+/-0.04的AUC，使用多数投票方法在外部测试中达到了0.84的准确率、0.71的敏感性和0.91的特异性。我们的结果表明，使用自监督的轻量级域自适应可以有效地增强基于PFM的MIL流程，用于具有临床挑战性的组织病理学任务。代码可在https://github.com/zhanghaoyue/DA_SSL_TURBT获取。


### 论文摘要

Recent deep learning frameworks in histopathology, particularly multiple instance learning (MIL) combined with pathology foundational models (PFMs), have shown strong performance. However, PFMs exhibit limitations on certain cancer or specimen types due to domain shifts - these cancer types were rarely used for pretraining or specimens contain tissue-based artifacts rarely seen within the pretraining population. Such is the case for transurethral resection of bladder tumor (TURBT), which are essential for diagnosing muscle-invasive bladder cancer (MIBC), but contain fragmented tissue chips and electrocautery artifacts and were not widely used in publicly available PFMs. To address this, we propose a simple yet effective domain-adaptive self-supervised adaptor (DA-SSL) that realigns pretrained PFM features to the TURBT domain without fine-tuning the foundational model itself. We pilot this framework for predicting treatment response in TURBT, where histomorphological features are currently underutilized and identifying patients who will benefit from neoadjuvant chemotherapy (NAC) is challenging. In our multi-center study, DA-SSL achieved an AUC of 0.77+/-0.04 in five-fold cross-validation and an external test accuracy of 0.84, sensitivity of 0.71, and specificity of 0.91 using majority voting. Our results demonstrate that lightweight domain adaptation with self-supervision can effectively enhance PFM-based MIL pipelines for clinically challenging histopathology tasks. Code is Available at https://github.com/zhanghaoyue/DA_SSL_TURBT.

---

## 87. Memory in the Age of AI Agents

**论文链接:** [http://arxiv.org/abs/2512.13564v1](http://arxiv.org/abs/2512.13564v1)

**作者:** Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng, Zhenrong Cheng, Xuanbo Fan, Jiaxin Guo, Xinlei Yu, Zhenhong Zhou, Zewen Hu, Jiahao Huo, Junhao Wang, Yuwei Niu, Yu Wang, Zhenfei Yin, Xiaobin Hu, Yue Liao, Qiankun Li, Kun Wang, Wangchunshu Zhou, Yixin Liu, Dawei Cheng, Qi Zhang, Tao Gui, Shirui Pan, Yan Zhang, Philip Torr, Zhicheng Dou, Ji-Rong Wen, Xuanjing Huang, Yu-Gang Jiang, Shuicheng Yan

**发布时间:** 2025-12-15

### GPT解析

### 总结

这是一篇关于智能体记忆研究的综述文章，旨在提供当前智能体记忆研究的全景图。文章通过形式、功能和动态三个视角对智能体记忆进行了分类和分析，并提供了记忆基准和开源框架的综合总结，同时探讨了未来的研究前沿。

### 背景

记忆已成为基于基础模型的智能体的核心能力，并将继续如此。智能体记忆研究迅速扩展并吸引了前所未有的关注，但该领域变得越来越碎片化。现有工作在动机、实现和评估协议上存在显著差异，而记忆术语的激增进一步模糊了概念清晰性。传统的长期/短期记忆分类方法不足以捕捉当代智能体记忆系统的多样性。

### 目的

提供当前智能体记忆研究的最新全景图，明确定义智能体记忆的范围并将其与相关概念区分开，为未来智能体智能设计中记忆作为第一类原语提供概念基础。

### 方法

通过形式、功能和动态的统一视角检查智能体记忆；从形式视角识别三种主要实现方式；从功能视角提出更精细的分类法；从动态视角分析记忆随时间的形成、演变和检索过程；编写记忆基准和开源框架的综合总结；探讨新兴研究前沿。

### 主要发现

智能体记忆可标记级、参数化和潜在三种形式实现；功能上可分为事实记忆、经验记忆和工作记忆；记忆随时间形成、演变和检索的动态过程；提供了记忆基准和开源框架的综合总结。

### 结论

这项调查不仅作为现有工作的参考，也为重新思考记忆作为未来智能体设计中第一类原语的概念基础。

### 翻译

记忆已经成为并将继续成为基于基础模型的智能体的核心能力。随着智能体记忆研究的迅速扩展并吸引了前所未有的关注，该领域也变得越来越碎片化。属于智能体记忆范畴的现有工作在动机、实现和评估协议上往往存在显著差异，而大量定义松散的记忆术语进一步模糊了概念清晰性。传统的长期/短期记忆等分类方法已被证明不足以捕捉当代智能体记忆系统的多样性。本文旨在提供当前智能体记忆研究的最新全景图。我们首先明确定义智能体记忆的范围，并将其与LLM记忆、检索增强生成(RAG)和上下文工程等相关概念区分开来。然后，我们通过形式、功能和动态的统一视角来审视智能体记忆。从形式视角，我们确定了智能体记忆的三种主要实现方式，即标记级记忆、参数化记忆和潜在记忆。从功能视角，我们提出了更精细的分类法，区分事实记忆、经验记忆和工作记忆。从动态视角，我们分析了记忆如何随时间形成、演变和检索。为支持实际开发，我们整理了记忆基准和开源框架的综合总结。除整合外，我们还阐述了关于新兴研究前沿的前瞻性观点，包括记忆自动化、强化学习集成、多模态记忆、多智能体记忆和可信度问题。我们希望这项调查不仅作为现有工作的参考，也为重新思考记忆作为未来智能体设计中第一类原语的概念基础。


### 论文摘要

Memory has emerged, and will continue to remain, a core capability of foundation model-based agents. As research on agent memory rapidly expands and attracts unprecedented attention, the field has also become increasingly fragmented. Existing works that fall under the umbrella of agent memory often differ substantially in their motivations, implementations, and evaluation protocols, while the proliferation of loosely defined memory terminologies has further obscured conceptual clarity. Traditional taxonomies such as long/short-term memory have proven insufficient to capture the diversity of contemporary agent memory systems. This work aims to provide an up-to-date landscape of current agent memory research. We begin by clearly delineating the scope of agent memory and distinguishing it from related concepts such as LLM memory, retrieval augmented generation (RAG), and context engineering. We then examine agent memory through the unified lenses of forms, functions, and dynamics. From the perspective of forms, we identify three dominant realizations of agent memory, namely token-level, parametric, and latent memory. From the perspective of functions, we propose a finer-grained taxonomy that distinguishes factual, experiential, and working memory. From the perspective of dynamics, we analyze how memory is formed, evolved, and retrieved over time. To support practical development, we compile a comprehensive summary of memory benchmarks and open-source frameworks. Beyond consolidation, we articulate a forward-looking perspective on emerging research frontiers, including memory automation, reinforcement learning integration, multimodal memory, multi-agent memory, and trustworthiness issues. We hope this survey serves not only as a reference for existing work, but also as a conceptual foundation for rethinking memory as a first-class primitive in the design of future agentic intelligence.

---

## 88. Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains

**论文链接:** [http://arxiv.org/abs/2512.13534v1](http://arxiv.org/abs/2512.13534v1)

**作者:** Marianne Rakic, Siyu Gai, Etienne Chollet, John V. Guttag, Adrian V. Dalca

**发布时间:** 2025-12-15

**备注:** Accepted at NeurIPS 2025. Code available at: https://github.com/mariannerakic/Pancakes

### GPT解析

### 总结

本文介绍了Pancakes框架，它能够自动为来自新领域的生物医学图像生成多种合理协议的多标签分割图，同时保持相关图像间的语义一致性，显著优于现有基础模型。

### 背景

生物医学图像可根据不同应用需求进行多种有意义分割，如大脑MRI可按组织类型、血管区域、解剖区域等分割。现有自动分割模型通常只支持单一训练协议或需要繁琐手动提示指定分割方式。

### 目的

开发一个框架，使给定来自新领域图像时，能自动生成多种合理协议的多标签分割图，同时保持相关图像间的语义一致性。

### 方法

引入名为Pancakes的框架，该框架采用一种新的问题表述方式，在七个保留数据集上进行了一系列实验来验证其性能。

### 主要发现

实验表明，Pancakes模型在生成多种合理的全图像分割方面显著优于现有基础模型，且这些分割在图像间保持语义连贯性。

### 结论

Pancakes框架解决了现有自动分割模型的局限性，能够自动生成多标签分割图并保持语义一致性，为生物医学图像分割提供了新思路。

### 翻译

单个生物医学图像可以根据所需应用以多种有意义的方式进行分割。例如，大脑MRI可以根据组织类型、血管区域、广泛的解剖区域、精细的解剖结构或病理等进行分割。现有的自动分割模型通常要么只支持单一协议（即它们训练时所用的协议），要么需要繁琐的手动提示来指定所需的分割方式。我们引入了Pancakes框架，该框架能够根据来自先前未见领域的新图像，自动为多种合理的协议生成多标签分割图，同时在相关图像之间保持语义一致性。Pancakes引入了一种新的问题表述，这是现有基础模型目前无法实现的。在七个保留数据集上进行的一系列实验中，我们证明与现有基础模型相比，我们的模型在生成几种合理的全图像分割方面表现显著更好，且这些分割在图像之间是语义连贯的。


### 论文摘要

A single biomedical image can be meaningfully segmented in multiple ways, depending on the desired application. For instance, a brain MRI can be segmented according to tissue types, vascular territories, broad anatomical regions, fine-grained anatomy, or pathology, etc. Existing automatic segmentation models typically either (1) support only a single protocol, the one they were trained on, or (2) require labor-intensive manual prompting to specify the desired segmentation. We introduce Pancakes, a framework that, given a new image from a previously unseen domain, automatically generates multi-label segmentation maps for multiple plausible protocols, while maintaining semantic consistency across related images. Pancakes introduces a new problem formulation that is not currently attainable by existing foundation models. In a series of experiments on seven held-out datasets, we demonstrate that our model can significantly outperform existing foundation models in producing several plausible whole-image segmentations, that are semantically coherent across images.

---

## 89. Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model

**论文链接:** [http://arxiv.org/abs/2512.13507v1](http://arxiv.org/abs/2512.13507v1)

**作者:** Siyan Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Xuyan Chi, Jian Cong, Qinpeng Cui, Qide Dong, Junliang Fan, Jing Fang, Zetao Fang, Chengjian Feng, Han Feng, Mingyuan Gao, Yu Gao, Qiushan Guo, Boyang Hao, Qingkai Hao, Bibo He, Qian He, Tuyen Hoang, Ruoqing Hu, Xi Hu, Weilin Huang, Zhaoyang Huang, Zhongyi Huang, Siqi Jiang, Wei Jiang, Yunpu Jiang, Zhuo Jiang, Ashley Kim, Jianan Kong, Zhichao Lai, Shanshan Lao, Ai Li, Feiya Li, Gen Li, Huixia Li, JiaShi Li, Liang Li, Ming Li, Tao Li, Xian Li, Xiaojie Li, Xiaoyang Li, Xingxing Li, Yameng Li, Yifu Li, Yiying Li, Chao Liang, Ying Liang, Zhiqiang Liang, Wang Liao, Yalin Liao, Heng Lin, Kengyu Lin, Shanchuan Lin, Xi Lin, Zhijie Lin, Feng Ling, Fangfang Liu, Gaohong Liu, Jiawei Liu, Jie Liu, Shouda Liu, Shu Liu, Sichao Liu, Songwei Liu, Xin Liu, Xue Liu, Yibo Liu, Zikun Liu, Zuxi Liu, Junlin Lyu, Lecheng Lyu, Qian Lyu, Han Mu, Xiaonan Nie, Jingzhe Ning, Xitong Pan, Yanghua Peng, Lianke Qin, Xueqiong Qu, Yuxi Ren, Yuchen Shen, Guang Shi, Lei Shi, Yan Song, Yinglong Song, Fan Sun, Li Sun, Renfei Sun, Zeyu Sun, Wenjing Tang, Zirui Tao, Feng Wang, Furui Wang, Jinran Wang, Junkai Wang, Ke Wang, Kexin Wang, Qingyi Wang, Rui Wang, Sen Wang, Shuai Wang, Tingru Wang, Weichen Wang, Xin Wang, Yanhui Wang, Yue Wang, Yuping Wang, Yuxuan Wang, Ziyu Wang, Guoqiang Wei, Wanru Wei, Di Wu, Guohong Wu, Hanjie Wu, Jian Wu, Jie Wu, Ruolan Wu, Xinglong Wu, Yonghui Wu, Ruiqi Xia, Liang Xiang, Fei Xiao, XueFeng Xiao, Pan Xie, Shuangyi Xie, Shuang Xu, Jinlan Xue, Bangbang Yang, Ceyuan Yang, Jiaqi Yang, Runkai Yang, Tao Yang, Yang Yang, Yihang Yang, ZhiXian Yang, Ziyan Yang, Yifan Yao, Zilyu Ye, Bowen Yu, Chujie Yuan, Linxiao Yuan, Sichun Zeng, Weihong Zeng, Xuejiao Zeng, Yan Zeng, Chuntao Zhang, Heng Zhang, Jingjie Zhang, Kuo Zhang, Liang Zhang, Liying Zhang, Manlin Zhang, Ting Zhang, Weida Zhang, Xiaohe Zhang, Xinyan Zhang, Yan Zhang, Yuan Zhang, Zixiang Zhang, Fengxuan Zhao, Huating Zhao, Yang Zhao, Hao Zheng, Jianbin Zheng, Xiaozheng Zheng, Yangyang Zheng, Yijie Zheng, Jiexin Zhou, Kuan Zhu, Shenhan Zhu, Wenjia Zhu, Benhui Zou, Feilong Zuo

**发布时间:** 2025-12-15

**备注:** Seedance 1.5 pro Technical Report

### GPT解析

### 总结

Seedance 1.5 pro是一个专为原生联合音频视频生成设计的基础模型，采用双分支Diffusion Transformer架构，实现了出色的视听同步性和卓越的生成质量，并提供了精确的多语言口型同步、动态摄像机控制和增强的叙事连贯性。

### 背景

视频生成领域的最新进展为统一的视听生成铺平了道路。

### 目的

介绍Seedance 1.5 pro，这是一个专门为原生、联合音频视频生成设计的基础模型。

### 方法

采用双分支Diffusion Transformer架构，集成了跨模态联合模块和专门的多阶段数据管道；实施后训练优化，包括监督微调(SFT)和人类反馈强化学习(RLHF)；引入加速框架，提高推理速度10倍以上。

### 主要发现

Seedance 1.5 pro具备精确的多语言和方言口型同步能力、动态电影摄像机控制功能和增强的叙事连贯性。

### 结论

Seedance 1.5 pro是专业级内容创作的强大引擎，现在可通过火山引擎访问。

### 翻译

视频生成领域的最新进展为统一的视听生成铺平了道路。在这项工作中，我们提出了Seedance 1.5 pro，这是一个专门为原生、联合音频视频生成设计的基础模型。利用双分支Diffusion Transformer架构，该模型集成了跨模态联合模块和专门的多阶段数据管道，实现了出色的视听同步性和卓越的生成质量。为确保实用性，我们实施了精细的后训练优化，包括在高质量数据集上进行监督微调(SFT)和使用多维奖励模型进行人类反馈强化学习(RLHF)。此外，我们还引入了一个加速框架，将推理速度提高了10倍以上。Seedance 1.5 pro以其精确的多语言和方言口型同步、动态电影摄像机控制和增强的叙事连贯性而脱颖而出，成为专业级内容创作的强大引擎。Seedance 1.5 pro现可通过火山引擎访问，网址为https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?type=GenVideo。


### 论文摘要

Recent strides in video generation have paved the way for unified audio-visual generation. In this work, we present Seedance 1.5 pro, a foundational model engineered specifically for native, joint audio-video generation. Leveraging a dual-branch Diffusion Transformer architecture, the model integrates a cross-modal joint module with a specialized multi-stage data pipeline, achieving exceptional audio-visual synchronization and superior generation quality. To ensure practical utility, we implement meticulous post-training optimizations, including Supervised Fine-Tuning (SFT) on high-quality datasets and Reinforcement Learning from Human Feedback (RLHF) with multi-dimensional reward models. Furthermore, we introduce an acceleration framework that boosts inference speed by over 10X. Seedance 1.5 pro distinguishes itself through precise multilingual and dialect lip-syncing, dynamic cinematic camera control, and enhanced narrative coherence, positioning it as a robust engine for professional-grade content creation. Seedance 1.5 pro is now accessible on Volcano Engine at https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?type=GenVideo.

---

## 90. From Zipf's Law to Neural Scaling through Heaps' Law and Hilberg's Hypothesis

**论文链接:** [http://arxiv.org/abs/2512.13491v1](http://arxiv.org/abs/2512.13491v1)

**作者:** Łukasz Dębowski

**发布时间:** 2025-12-15

**备注:** 32 pages, no figures

### GPT解析

### 总结

本文探讨了神经缩放定律与齐普夫定律之间的演绎关系，证明了在特定假设条件下，神经缩放定律是齐普夫定律的必然结果。

### 背景

神经缩放定律描述基础模型（如大型语言模型）的交叉熵率随训练标记数量、参数量和计算量的变化规律；齐普夫定律则指出标记分布呈现幂律尾部特征。

### 目的

展示神经缩放定律如何从齐普夫定律推导而来，揭示两者之间的理论基础联系。

### 方法

通过一系列推导步骤，从齐普夫定律到希普定律，再到希尔伯格假说，最后推导出神经缩放定律；并通过圣塔菲过程的玩具例子进行验证。

### 主要发现

在特定假设条件下，神经缩放定律是齐普夫定律的必然结果；展示了从齐普夫定律推导出神经缩放定律的完整链条。

### 结论

神经缩放定律与齐普夫定律之间存在理论基础联系，这为理解语言模型的缩放行为提供了新的视角。

### 翻译

我们检查了神经缩放定律与齐普夫定律之间的演绎关系——这两个在机器学习和定量语言学中讨论的陈述。神经缩放定律描述基础模型（如大型语言模型）的交叉熵率如何随训练标记数量、参数量和计算量的变化而变化。相比之下，齐普夫定律认为标记的分布呈现幂律尾部特征。虽然在更具体的情境中已有类似主张，但我们展示了在揭示的某些广泛假设条件下，神经缩放定律是齐普夫定律的必然结果。推导步骤如下：我们从齐普夫定律推导出关于词汇增长的希普定律，从希普定律推导出关于熵缩放的希尔伯格假说，再从希尔伯格假说推导出神经缩放定律。我们通过满足所有四种统计定律的圣塔菲过程的玩具例子来说明这些推理步骤。


### 论文摘要

We inspect the deductive connection between the neural scaling law and Zipf's law -- two statements discussed in machine learning and quantitative linguistics. The neural scaling law describes how the cross entropy rate of a foundation model -- such as a large language model -- changes with respect to the amount of training tokens, parameters, and compute. By contrast, Zipf's law posits that the distribution of tokens exhibits a power law tail. Whereas similar claims have been made in more specific settings, we show that the neural scaling law is a consequence of Zipf's law under certain broad assumptions that we reveal systematically. The derivation steps are as follows: We derive Heaps' law on the vocabulary growth from Zipf's law, Hilberg's hypothesis on the entropy scaling from Heaps' law, and the neural scaling from Hilberg's hypothesis. We illustrate these inference steps by a toy example of the Santa Fe process that satisfies all the four statistical laws.

---

## 91. Test-Time Modification: Inverse Domain Transformation for Robust Perception

**论文链接:** [http://arxiv.org/abs/2512.13454v1](http://arxiv.org/abs/2512.13454v1)

**作者:** Arpit Jadon, Joshua Niemeijer, Yuki M. Asano

**发布时间:** 2025-12-15

**备注:** Preprint

### GPT解析

### 总结

该研究提出了一种使用扩散模型在测试时将目标图像映射回源分布的新方法，用于解决领域泛化问题，在多个视觉任务上取得了显著性能提升。

### 背景

生成式基础模型包含广泛的视觉知识，能够产生多样化的图像变化，在推进领域泛化任务方面具有前景。然而，使用它们进行训练数据增强存在合成目标领域变化缓慢、昂贵且不完整的问题。

### 目的

提出一种替代传统数据增强的方法，在测试时利用扩散模型将目标图像映射回源分布，以提高领域泛化性能。

### 方法

该方法仅需源领域描述，保留原始任务模型，无需生成大规模合成数据。研究者在具有未知目标分布的真实到真实领域泛化场景中，针对分割、检测和分类任务进行了验证，并分析了多种生成和下游模型，包括用于增强鲁棒性的集成变体。

### 主要发现

该方法在多个挑战性环境变化场景中取得了持续改进：在BDD100K-Night上实现了137%的相对增益，在ImageNet-R上实现了68%的相对增益，在DarkZurich上实现了62%的相对增益。

### 结论

使用扩散模型在测试时将目标图像映射回源分布是一种有效的方法，能够克服传统生成式模型在领域泛化中的局限性，显著提升模型在未知目标分布上的性能。

### 翻译

生成式基础模型包含广泛的视觉知识，并能产生多样化的图像变化，使其在推进领域泛化任务方面特别有前景。虽然它们可用于训练数据增强，但合成全面的目标领域变化仍然缓慢、昂贵且不完整。我们提出了一种替代方案：在测试时使用扩散模型将目标图像映射回下游模型训练的源分布。这种方法仅需源领域描述，保留任务模型，并消除了大规模合成数据生成。我们在具有挑战性环境变化的真实到真实领域泛化场景中，针对分割、检测和分类任务展示了持续改进，这些场景具有未知的目标分布。我们的分析涵盖了多种生成和下游模型，包括用于增强鲁棒性的集成变体。该方法实现了显著的相对增益：在BDD100K-Night上为137%，在ImageNet-R上为68%，在DarkZurich上为62%。


### 论文摘要

Generative foundation models contain broad visual knowledge and can produce diverse image variations, making them particularly promising for advancing domain generalization tasks. While they can be used for training data augmentation, synthesizing comprehensive target-domain variations remains slow, expensive, and incomplete. We propose an alternative: using diffusion models at test time to map target images back to the source distribution where the downstream model was trained. This approach requires only a source domain description, preserves the task model, and eliminates large-scale synthetic data generation. We demonstrate consistent improvements across segmentation, detection, and classification tasks under challenging environmental shifts in real-to-real domain generalization scenarios with unknown target distributions. Our analysis spans multiple generative and downstream models, including an ensemble variant for enhanced robustness. The method achieves substantial relative gains: 137% on BDD100K-Night, 68% on ImageNet-R, and 62% on DarkZurich.

---

## 92. RecTok: Reconstruction Distillation along Rectified Flow

**论文链接:** [http://arxiv.org/abs/2512.13421v1](http://arxiv.org/abs/2512.13421v1)

**作者:** Qingyu Shi, Size Wu, Jinbin Bai, Kaidong Yu, Yujing Wang, Yunhai Tong, Xiangtai Li, Xuelong Li

**发布时间:** 2025-12-15

### GPT解析

### 总结

RecTok是一种新型视觉tokenizer，通过流语义蒸馏和重建-对齐蒸馏两种创新方法克服了高维视觉tokenizer的局限性，实现了优越的图像重建、生成质量和判别性能，在gFID-50K上取得了最先进结果。

### 背景

视觉tokenizer在扩散模型中起关键作用，潜在空间维度决定重建保真度和语义表达能力，但现有方法受限于低维潜在空间，因为维度和生成质量之间存在基本权衡。

### 目的

解决高维视觉tokenizer表现不如低维tokenizer的问题，通过创新方法提高高维tokenizer的性能。

### 方法

提出RecTok，包含两个关键创新：流语义蒸馏和重建-对齐蒸馏。使流匹配中的前向流语义丰富，作为扩散变换器的训练空间，将VFM中的语义信息蒸馏到流匹配中的前向流轨迹中，并通过引入掩码特征重建损失进一步增强语义。

### 主要发现

RecTok实现了优越的图像重建、生成质量和判别性能；在有和无分类器自由指导设置下，在gFID-50K上取得了最先进结果；保持了语义丰富的潜在空间结构；随着潜在维度增加，观察到持续改进。

### 结论

RecTok成功解决了高维视觉tokenizer的局限性，通过流语义蒸馏和重建-对齐蒸馏方法提高了性能，代码和模型已公开可用。

### 翻译

视觉标记器在扩散模型中起着关键作用。潜在空间的维度决定了重建保真度和潜在特征的语义表达能力。然而，维度和生成质量之间存在基本权衡，这限制了现有方法只能使用低维潜在空间。尽管最近的工作利用视觉基础模型来丰富视觉标记器的语义并加速收敛，但高维标记器的表现仍然不如低维标记器。在这项工作中，我们提出了RecTok，它通过两个关键创新克服了高维视觉标记器的局限性：流语义蒸馏和重建-对齐蒸馏。我们的关键洞见是使流匹配中的前向流具有丰富的语义，作为扩散变换器的训练空间，而不是像以前的工作那样专注于潜在空间。具体来说，我们的方法将VFM中的语义信息蒸馏到流匹配中的前向流轨迹中。我们通过引入掩码特征重建损失进一步增强了语义。我们的RecTok实现了卓越的图像重建、生成质量和判别性能。在有和无分类器自由指导设置下，它在gFID-50K上取得了最先进的结果，同时保持了语义丰富的潜在空间结构。此外，随着潜在维度的增加，我们观察到持续改进。代码和模型可在https://shi-qingyu.github.io/rectok.github.io获取。


### 论文摘要

Visual tokenizers play a crucial role in diffusion models. The dimensionality of latent space governs both reconstruction fidelity and the semantic expressiveness of the latent feature. However, a fundamental trade-off is inherent between dimensionality and generation quality, constraining existing methods to low-dimensional latent spaces. Although recent works have leveraged vision foundation models to enrich the semantics of visual tokenizers and accelerate convergence, high-dimensional tokenizers still underperform their low-dimensional counterparts. In this work, we propose RecTok, which overcomes the limitations of high-dimensional visual tokenizers through two key innovations: flow semantic distillation and reconstruction--alignment distillation. Our key insight is to make the forward flow in flow matching semantically rich, which serves as the training space of diffusion transformers, rather than focusing on the latent space as in previous works. Specifically, our method distills the semantic information in VFMs into the forward flow trajectories in flow matching. And we further enhance the semantics by introducing a masked feature reconstruction loss. Our RecTok achieves superior image reconstruction, generation quality, and discriminative performance. It achieves state-of-the-art results on the gFID-50K under both with and without classifier-free guidance settings, while maintaining a semantically rich latent space structure. Furthermore, as the latent dimensionality increases, we observe consistent improvements. Code and model are available at https://shi-qingyu.github.io/rectok.github.io.

---

## 93. Harmonizing Generalization and Specialization: Uncertainty-Informed Collaborative Learning for Semi-supervised Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2512.13101v1](http://arxiv.org/abs/2512.13101v1)

**作者:** Wenjing Lu, Yi Hong, Yang Yang

**发布时间:** 2025-12-15

**备注:** This work has been submitted to the IEEE TMI for possible publication

### GPT解析

### 总结

提出了一种名为Uncertainty-informed Collaborative Learning (UnCoL)的双重教师框架，用于解决半监督医学图像分割中的泛化和专业化平衡问题。

### 背景

视觉基础模型通过大规模、异构预训练在医学图像分割中展示了强大的泛化能力，但在有限标注或罕见病理变异条件下难以推广到专门临床任务，原因是通用先验与任务特定需求不匹配。

### 目的

开发一种方法协调半监督医学图像分割中的泛化和专业化，解决基础模型在特定临床任务中的局限性。

### 方法

UnCoL框架从冻结的基础模型中提取视觉和语义表示传递通用知识，同时保持渐进适应的教师捕获细粒度和任务特定表示；通过预测不确定性自适应调节伪标签学习，选择性抑制不可靠监督，稳定模糊区域学习。

### 主要发现

在多种2D和3D分割基准测试中，UnCoL持续优于最先进的半监督方法和基础模型基线；在显著减少标注需求的情况下，实现了接近全监督的性能。

### 结论

UnCoL有效解决了基础模型在专门临床任务中的泛化挑战，为半监督医学图像分割提供了新思路。

### 翻译

视觉基础模型通过利用大规模、异构预训练在医学图像分割中展示了强大的泛化能力。然而，由于通用先验与任务特定需求之间的不匹配，它们通常难以在有限的标注或罕见的病理变异条件下推广到专门的临床任务。为此，我们提出了不确定性感知协作学习(UnCoL)，一种双重教师框架，用于在半监督医学图像分割中协调泛化和专业化。具体来说，UnCoL从冻结的基础模型中提炼视觉和语义表示以转移通用知识，同时保持一个渐进适应的教师来捕获细粒度和任务特定的表示。为了平衡两个教师的指导，UnCoL中的伪标签学习通过预测不确定性进行自适应调节，选择性地抑制不可靠的监督，并在模糊区域稳定学习。在各种2D和3D分割基准上的实验表明，UnCoL持续优于最先进的半监督方法和基础模型基线。此外，我们的模型在显著减少标注需求的情况下，提供了接近全监督的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉基础模型在医学图像分割中难以泛化到特定临床任务的问题，特别是在有限标注或罕见病理条件下。这个问题很重要，因为医学图像分割是临床诊断、治疗计划和疾病监测的基础，而专家标注成本高昂且耗时；基础模型虽然泛化能力强但缺乏任务特定精度，传统半监督方法又无法充分利用外部知识先验。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析发现基础模型提供通用先验但缺乏任务特定精度，而传统半监督方法专注于单个任务但无法整合外部知识。他们借鉴了知识蒸馏思想、半监督学习中的不确定性估计以及教师-学生框架，但创新性地设计了双教师框架，结合冻结基础模型和自适应专业教师，并通过双重路径知识蒸馏和基于不确定性的伪标记策略实现协同学习。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过不确定性信息协调泛化和专业化，实现通用先验和任务特定知识的协同学习。整体流程分为两阶段：1)数据高效预训练阶段，学生模型从标注数据学习并通过双重路径知识蒸馏(视觉和语义)从冻结基础模型获取知识；2)半监督微调阶段，同时使用标注和无标注数据，构建EMA专业教师，使用基于不确定性的伪标记策略动态选择可靠教师的预测来监督学习。推理时仅使用学生模型，无需提示或教师指导。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)不确定性信息协作学习框架(UnCoL)，协调泛化和专业化；2)双重路径知识蒸馏，转移视觉和语义表示；3)基于不确定性的伪标记策略，动态选择可靠教师。相比之前工作，该方法不依赖手动提示实现全自动管道，在域转移和有限标注下表现更好；不假设基础模型能准确捕获任务特定语义；使用像素级不确定性加权而非全局一致性假设；结合视觉和语义双重路径蒸馏而非仅对齐概率输出。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种不确定性信息协作学习框架，通过双教师设计和双重路径知识蒸馏，有效结合了基础模型的通用先验与自适应教师的任务特定知识，在半监督医学图像分割中实现了接近全监督的性能，同时显著减少了对标注数据的需求。'}


### 论文摘要

Vision foundation models have demonstrated strong generalization in medical image segmentation by leveraging large-scale, heterogeneous pretraining. However, they often struggle to generalize to specialized clinical tasks under limited annotations or rare pathological variations, due to a mismatch between general priors and task-specific requirements. To address this, we propose Uncertainty-informed Collaborative Learning (UnCoL), a dual-teacher framework that harmonizes generalization and specialization in semi-supervised medical image segmentation. Specifically, UnCoL distills both visual and semantic representations from a frozen foundation model to transfer general knowledge, while concurrently maintaining a progressively adapting teacher to capture fine-grained and task-specific representations. To balance guidance from both teachers, pseudo-label learning in UnCoL is adaptively regulated by predictive uncertainty, which selectively suppresses unreliable supervision and stabilizes learning in ambiguous regions. Experiments on diverse 2D and 3D segmentation benchmarks show that UnCoL consistently outperforms state-of-the-art semi-supervised methods and foundation model baselines. Moreover, our model delivers near fully supervised performance with markedly reduced annotation requirements.

---

## 94. UniVCD: A New Method for Unsupervised Change Detection in the Open-Vocabulary Era

**论文链接:** [http://arxiv.org/abs/2512.13089v1](http://arxiv.org/abs/2512.13089v1)

**作者:** Ziqiang Zhu, Bowei Yang

**发布时间:** 2025-12-15

**备注:** 10 pages, 6 figures

### GPT解析

### 总结

本文提出了一种名为UniVCD的无监督、开放词汇表变化检测方法，基于冻结的SAM2和CLIP视觉基础模型构建，能够在无需标注数据的情况下检测多样化场景中的变化，实验证明其性能优于或匹敌现有方法。

### 背景

变化检测(CD)用于从多时相观测中识别场景变化，广泛应用于城市发展和环境监测。现有CD方法大多依赖监督学习，导致性能高度依赖数据集且标注成本高，同时通常只关注少数预定义类别，对多样化场景的泛化能力差。随着SAM2和CLIP等视觉基础模型的兴起，为解决这些问题提供了新机会。

### 目的

提出一种无需标注数据或配对变化图像的、无监督的、开放词汇表的变化检测方法，能够在多样化场景和成像几何条件下检测类别无关的变化。

### 方法

提出Unified Open-Vocabulary Change Detection (UniVCD)方法，基于冻结的SAM2和CLIP构建。引入轻量级特征对齐模块，连接SAM2的空间详细表示和CLIP的语义先验，实现高分辨率、语义感知的变化估计，同时保持可训练参数数量较少。此外，还引入了简化的后处理流程来抑制噪声和伪变化，改善具有明确定边界的物体的检测准确性。

### 主要发现

在多个公共BCD(二元变化检测)和SCD(语义变化检测)基准测试上，UniVCD取得了一致的强性能。在F1和IoU等关键指标上，匹配或超过了现有的开放词汇表CD方法。

### 结论

使用冻结的视觉基础模型和轻量级多模态对齐的无监督变化检测是开放词汇表CD的一种实用且有效的范式。代码和预训练模型将在https://github.com/Die-Xie/UniVCD发布。

### 翻译

变化检测(CD)从多时相观测中识别场景变化，广泛应用于城市发展和环境监测。大多数现有CD方法依赖监督学习，使性能高度依赖数据集并产生高标注成本；它们通常只关注少数预定义类别，对多样化场景泛化能力差。随着SAM2和CLIP等视觉基础模型的兴起，为缓解这些约束提供了新机会。我们提出了Unified Open-Vocabulary Change Detection (UniVCD)，一种基于冻结SAM2和CLIP构建的无监督、开放词汇表变化检测方法。UniVCD无需任何标注数据或配对变化图像，就能在多样化场景和成像几何条件下检测类别无关的变化。引入了轻量级特征对齐模块，以连接SAM2的空间详细表示和CLIP的语义先验，实现高分辨率、语义感知的变化估计，同时保持可训练参数数量较少。在此基础上，进一步引入了简化的后处理流程，以抑制噪声和伪变化，提高具有明确定边界的物体的检测准确性。在多个公共BCD(二元变化检测)和SCD(语义变化检测)基准上的实验表明，UniVCD在F1和IoU等关键指标上取得了一致的强性能，匹配或超过了现有的开放词汇表CD方法。结果表明，使用冻结的视觉基础模型和轻量级多模态对齐的无监督变化检测是开放词汇表CD的一种实用且有效的范式。代码和预训练模型将在https://github.com/Die-Xie/UniVCD发布。


### 论文摘要

Change detection (CD) identifies scene changes from multi-temporal observations and is widely used in urban development and environmental monitoring. Most existing CD methods rely on supervised learning, making performance strongly dataset-dependent and incurring high annotation costs; they typically focus on a few predefined categories and generalize poorly to diverse scenes. With the rise of vision foundation models such as SAM2 and CLIP, new opportunities have emerged to relax these constraints. We propose Unified Open-Vocabulary Change Detection (UniVCD), an unsupervised, open-vocabulary change detection method built on frozen SAM2 and CLIP. UniVCD detects category-agnostic changes across diverse scenes and imaging geometries without any labeled data or paired change images. A lightweight feature alignment module is introduced to bridge the spatially detailed representations from SAM2 and the semantic priors from CLIP, enabling high-resolution, semantically aware change estimation while keeping the number of trainable parameters small. On top of this, a streamlined post-processing pipeline is further introduced to suppress noise and pseudo-changes, improving the detection accuracy for objects with well-defined boundaries. Experiments on several public BCD (Binary Change Detection) and SCD (Semantic Change Detection) benchmarks show that UniVCD achieves consistently strong performance and matches or surpasses existing open-vocabulary CD methods in key metrics such as F1 and IoU. The results demonstrate that unsupervised change detection with frozen vision foundation models and lightweight multi-modal alignment is a practical and effective paradigm for open-vocabulary CD. Code and pretrained models will be released at https://github.com/Die-Xie/UniVCD.

---

## 95. Forging a Dynamic Memory: Retrieval-Guided Continual Learning for Generalist Medical Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.13072v1](http://arxiv.org/abs/2512.13072v1)

**作者:** Zizhi Chen, Yizhen Gao, Minghao Han, Yizhou Liu, Zhaoyu Chen, Dingkang Yang, Lihua Zhang

**发布时间:** 2025-12-15

### GPT解析

### 总结

该研究提出了一种多模态生物医学视觉语言模型的持续学习框架，结合检索增强生成和动态知识蒸馏技术，解决了模型在持续学习中保持细粒度模内特征同时跨越不同模态间领域差距的难题，并在医学通用任务增量学习基准测试中取得最先进性能。

### 背景

多模态生物医学视觉语言模型在持续学习领域展现出巨大潜力，但面临核心困境：如何在保持细粒度模内特征的同时，跨越不同模态间的显著领域差距。

### 目的

解决多模态生物医学视觉语言模型在持续学习中的核心困境，即保持细粒度模内特征同时跨越不同模态间的显著领域差距。

### 方法

1. 基于PubMed科学论文构建1800万多模态综合医学检索数据库；2. 首次将检索增强生成(RAG)集成到持续学习中；3. 采用多模态、多层RAG系统提供实时指导；4. 引入动态知识蒸馏框架，动态调整参数空间重要性、知识粒度和数据分布；5. 设计医学通用任务增量学习(MGTIL)基准测试。

### 主要发现

所提出的方法在所有指标上均取得最先进性能，证明了模型在适应重大领域转移、保留细微领域特征和实时学习新型复杂医学任务方面的卓越能力。

### 结论

通过整合检索增强生成和动态知识蒸馏技术，成功解决了多模态生物医学视觉语言模型在持续学习中的核心困境，显著提升了模型性能，为医学领域的持续学习提供了新解决方案。

### 翻译

多模态生物医学视觉语言模型在持续学习领域展现出巨大潜力。然而，它们面临一个核心困境：如何在保持细粒度模内特征的同时，跨越不同模态间的显著领域差距。为应对这一挑战，我们提出了一个全面框架。利用我们从PubMed科学论文中获取的1800万多模态综合医学检索数据库，我们开创性地将检索增强生成(RAG)集成到持续学习中。具体而言，我们采用多模态、多层RAG系统，通过动态、按需知识检索为模型微调提供实时指导。在此基础上，我们引入了动态知识蒸馏框架，该框架通过根据所需详细程度动态调整参数空间重要性、蒸馏知识粒度和参考数据集的数据分布，精确解决了上述核心困境。为彻底验证我们策略的临床价值，我们设计了一个更严格的医学通用任务增量学习(MGTIL)基准。该基准旨在同时评估模型适应重大领域转移、保留细微领域特征和实时学习新型复杂医学任务的能力。大量实验结果表明，我们提出的方法在所有指标上均取得了最先进的性能。代码见补充材料。


### 论文摘要

Multimodal biomedical Vision-Language Models (VLMs) exhibit immense potential in the field of Continual Learning (CL). However, they confront a core dilemma: how to preserve fine-grained intra-modality features while bridging the significant domain gap across different modalities. To address this challenge, we propose a comprehensive framework. Leveraging our 18-million multimodal and comprehensive medical retrieval database derived from PubMed scientific papers, we pioneer the integration of Retrieval-Augmented Generation (RAG) into CL. Specifically, we employ a multi-modal, multi-layer RAG system that provides real-time guidance for model fine-tuning through dynamic, on-demand knowledge retrieval. Building upon this, we introduce a dynamic knowledge distillation framework. This framework precisely resolves the aforementioned core dilemma by dynamically modulating the importance of the parameter space, the granularity of the distilled knowledge, and the data distribution of the reference dataset in accordance with the required level of detail. To thoroughly validate the clinical value of our strategy, we have designed a more rigorous \textbf{M}edical Generalist Task Incremental Learning (MGTIL) benchmark. This benchmark is engineered to simultaneously evaluate the model's capacity for adaptation to significant domain shifts, retention of subtle intra-domain features, and real-time learning of novel and complex medical tasks. Extensive experimental results demonstrate that our proposed method achieves state-of-the-art (SOTA) performance across all metrics. The code is provided in the supplementary materials.

---

## 96. Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing

**论文链接:** [http://arxiv.org/abs/2512.13055v1](http://arxiv.org/abs/2512.13055v1)

**作者:** Jaeyoon Kim, Yoonki Cho, Sung-Eui Yoon

**发布时间:** 2025-12-15

**备注:** AAAI 2026

### GPT解析

### 总结

本文提出了一种高效的非对称视觉位置识别(VPR)框架，通过结合高容量图库模型和轻量级查询网络，解决了高容量模型在资源受限设备上的部署问题。

### 背景

视觉位置识别(VPR)已通过高容量基础模型(如DINOv2)取得显著进展，但这些模型的计算成本很高，使其在资源受限设备上的部署不切实际。

### 目的

开发一个高效的非对称VPR框架，使其能够在资源受限环境中有效部署，同时保持高性能。

### 方法

提出了一种非对称VPR框架，包含用于离线特征提取的高容量图库模型和用于在线处理的轻量级查询网络；引入地理记忆库利用地理位置元数据组织图库特征，避免昂贵的k-NN计算；采用隐式嵌入增强技术提升查询网络建模特征变化的能力。

### 主要发现

该方法显著降低了计算成本，同时性能优于现有的非对称检索技术，为资源受限环境下的VPR建立了新的方向。

### 结论

该非对称VPR框架能够在减少计算成本的同时提高性能，为资源受限环境中的视觉位置识别提供了有效解决方案。

### 翻译

视觉位置识别(VPR)已通过高容量基础模型(如DINOv2)取得显著进展，实现了卓越的性能。然而，其巨大的计算成本使得在资源受限设备上的部署不切实际。在本文中，我们引入了一种高效的非对称VPR框架，该框架包含用于离线特征提取的高容量图库模型和用于在线处理的轻量级查询网络。在这种设置中的一个关键挑战是确保这些异构网络之间的兼容性，传统方法通过计算昂贵的基于k-NN的兼容性训练来解决这一问题。为了克服这一挑战，我们提出了一种地理记忆库，利用VPR数据库中固有的地理位置元数据来组织图库特征，消除了对穷举k-NN计算的需求。此外，我们引入了一种隐式嵌入增强技术，增强查询网络以建模特征变化，尽管其容量有限。大量实验证明，我们的方法不仅显著降低了计算成本，而且性能优于现有的非对称检索技术，为资源受限环境中的VPR建立了新的方向。代码可在https://github.com/jaeyoon1603/AsymVPR获取。


### 论文摘要

Visual Place Recognition (VPR) has advanced significantly with high-capacity foundation models like DINOv2, achieving remarkable performance. Nonetheless, their substantial computational cost makes deployment on resource-constrained devices impractical. In this paper, we introduce an efficient asymmetric VPR framework that incorporates a high-capacity gallery model for offline feature extraction with a lightweight query network for online processing. A key challenge in this setting is ensuring compatibility between these heterogeneous networks, which conventional approaches address through computationally expensive k-NN-based compatible training. To overcome this, we propose a geographical memory bank that structures gallery features using geolocation metadata inherent in VPR databases, eliminating the need for exhaustive k-NN computations. Additionally, we introduce an implicit embedding augmentation technique that enhances the query network to model feature variations despite its limited capacity. Extensive experiments demonstrate that our method not only significantly reduces computational costs but also outperforms existing asymmetric retrieval techniques, establishing a new aspect for VPR in resource-limited environments. The code is available at https://github.com/jaeyoon1603/AsymVPR

---

## 97. Light Field Based 6DoF Tracking of Previously Unobserved Objects

**论文链接:** [http://arxiv.org/abs/2512.13007v1](http://arxiv.org/abs/2512.13007v1)

**作者:** Nikolai Goncharov, James L. Gray, Donald G. Dansereau

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出了一种基于光场图像的目标跟踪方法，该方法不依赖预训练模型，对复杂视觉行为（如反射）具有鲁棒性。通过使用视觉基础模型提取语义和几何特征，并转换为视图相关的高斯斑点作为统一对象表示，支持可微分渲染和姿态优化。实验表明，该方法在困难情况下与最先进的基于模型的跟踪器具有竞争力。

### 背景

目标跟踪是机器人和自动驾驶流程中的重要步骤，需要泛化到前所未见和复杂的物体。现有的高性能方法通常依赖预捕获的目标视图来构建显式参考模型，这限制了它们只能处理固定集合的已知物体。然而，这类参考模型在处理视觉复杂外观时存在困难，降低了跟踪质量。

### 目的

开发一种不依赖预训练模型的目标跟踪方法，该方法能够处理具有复杂视觉行为（如反射）的物体，提高跟踪质量，并扩展到更广泛的物体类别。

### 方法

提出一种基于光场图像的目标跟踪方法，使用视觉基础模型从光场输入中提取语义和几何特征，并将它们转换为视图相关的高斯斑点。这些斑点作为统一的对象表示，支持可微分渲染和姿态优化。同时引入了一个包含具有精确真实姿态的有挑战性反射物体的光场目标跟踪数据集。

### 主要发现

实验表明，该方法在处理具有挑战性的反射物体时，与最先进的基于模型的跟踪器具有竞争力。该方法不依赖预训练模型，能够处理复杂的视觉行为，如反射。

### 结论

该方法为机器人系统中的通用目标跟踪铺平了道路，通过不依赖预训练模型和对复杂视觉行为的鲁棒性，扩展了目标跟踪的应用范围。

### 翻译

目标跟踪是机器人和自动驾驶流程中的重要步骤，必须泛化到前所未见和复杂的物体。现有的高性能方法通常依赖预捕获的目标视图来构建显式参考模型，这限制了它们只能处理固定集合的已知物体。然而，这类参考模型在处理视觉复杂外观时存在困难，降低了跟踪质量。在这项工作中，我们介绍了一种基于光场图像的目标跟踪方法，该方法不依赖预训练模型，同时对复杂的视觉行为（如反射）具有鲁棒性。我们使用视觉基础模型从光场输入中提取语义和几何特征，并将它们转换为视图相关的高斯斑点。这些斑点作为统一的对象表示，支持可微分渲染和姿态优化。我们进一步引入了一个包含具有精确真实姿态的有挑战性反射物体的光场目标跟踪数据集。实验表明，在这些困难情况下，我们的方法与最先进的基于模型的跟踪器具有竞争力，为机器人系统中的通用目标跟踪铺平了道路。代码/数据可在 https://github.com/nagonch/LiFT-6DoF 获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是物体跟踪方法对未知物体的泛化问题。现有方法通常需要预捕获的物体视图或CAD模型来构建参考模型，这限制了它们只能处理已知物体。在机器人技术和自动驾驶领域，系统经常需要处理以前未见过的复杂物体，特别是具有反射表面的物体，这对实现通用物体跟踪至关重要。解决这一问题对于提高机器人系统在现实世界中的适应性和实用性具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，包括对预定义模型的依赖和对复杂视觉外观处理不足的问题。他们借鉴了多个领域的最新进展：视觉基础模型用于提取语义和几何特征，神经场景表示方法用于建模物体外观，以及光场成像技术提供丰富的几何线索。核心设计思路是将光场图像本身作为物体模型，通过将视觉基础模型与在线构建的高斯斑点表示相结合，实现对未知物体的鲁棒跟踪。具体实现中，他们使用了Grounding DINO和SAM2进行分割，Depth Anything进行深度估计，以及高斯斑点作为物体表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将光场图像本身视为物体模型，同时捕获几何和外观信息，为在线姿态跟踪提供自然基础。通过将视觉基础模型提供的语义和几何理解与在线参考物体模型（由高斯斑点表示）的外观监督相结合，实现对复杂视觉行为的鲁棒性。整体实现流程包括：1) 光场处理：使用Grounding DINO和SAM2分割物体，融合光场视差和Depth AnyThing的深度估计，生成3D点云；2) 高斯斑点初始化和训练：通过球谐系数初始化高斯斑点，并进行在线优化；3) 高斯斑点配准：使用彩色ICP进行粗配准，然后通过光度优化进行精细姿态调整。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出基于光场图像的在线物体建模方法，无需预训练模型；2) 将光场图像转换为视图相关的高斯斑点作为跟踪参考；3) 基于高斯斑点的姿态优化模块，对镜面物体表现优异；4) 创建了包含挑战性镜面物体的光场跟踪数据集。相比之前的工作，不同之处在于：不需要预定义的CAD模型或预捕获的物体视图；能够处理以前未见过的物体；对反射表面等复杂视觉行为具有更强的鲁棒性；使用光场图像而非RGB-D提供更丰富的几何信息；采用高斯斑点而非网格表示更好地建模物体外观。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于光场图像的6自由度物体跟踪方法，无需预先训练的物体模型，即可实现对具有复杂反射表面的未知物体的鲁棒跟踪，为机器人系统中的通用物体跟踪开辟了新途径。'}


### 论文摘要

Object tracking is an important step in robotics and reautonomous driving pipelines, which has to generalize to previously unseen and complex objects. Existing high-performing methods often rely on pre-captured object views to build explicit reference models, which restricts them to a fixed set of known objects. However, such reference models can struggle with visually complex appearance, reducing the quality of tracking. In this work, we introduce an object tracking method based on light field images that does not depend on a pre-trained model, while being robust to complex visual behavior, such as reflections. We extract semantic and geometric features from light field inputs using vision foundation models and convert them into view-dependent Gaussian splats. These splats serve as a unified object representation, supporting differentiable rendering and pose optimization. We further introduce a light field object tracking dataset containing challenging reflective objects with precise ground truth poses. Experiments demonstrate that our method is competitive with state-of-the-art model-based trackers in these difficult cases, paving the way toward universal object tracking in robotic systems. Code/data available at https://github.com/nagonch/LiFT-6DoF.

---

## 98. Investigating Data Pruning for Pretraining Biological Foundation Models at Scale

**论文链接:** [http://arxiv.org/abs/2512.12932v1](http://arxiv.org/abs/2512.12932v1)

**作者:** Yifan Wu, Jiyue Jiang, Xichen Ye, Yiqi Wang, Chang Zhou, Yitao Xu, Jiayang Chen, He Hu, Weizhong Zhang, Cheng Jin, Jiao Yuan, Yu Li

**发布时间:** 2025-12-15

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种影响引导的数据修剪框架，用于降低生物基础模型（BioFMs）的训练成本，同时保持模型性能

### 背景

生物基础模型（BioFMs）在大规模生物序列预训练后，能为多种生物信息学任务提供有意义的表示，但其训练需要大量计算资源和参数，导致学术实验室难以复现和使用

### 目的

解决BioFMs训练计算成本高的问题，提高其可访问性和可持续性

### 方法

提出一种后验影响引导的数据修剪框架，引入基于子集的自影响公式来估计样本重要性，并开发了Top-k影响（Top I）和以覆盖为中心的影响（CCI）两种选择策略

### 主要发现

在RNA-FM和ESM-C两个代表性BioFMs上的实验表明，该方法在超过99%的极端修剪率下仍优于随机选择；在RNA和蛋白质任务中，其核心集表现优于十倍大小的随机子集，显示生物序列数据中存在大量冗余

### 结论

影响引导的数据修剪能显著降低BioFM预训练的计算成本，为更高效、可访问和可持续的生物AI研究开辟了新途径

### 翻译

生物基础模型是在大规模生物序列上预训练的模型，最近显示出为各种下游生物信息学任务提供有意义表示的强大潜力。然而，这类模型通常依赖数百万至数十亿的训练序列和数十亿的参数，导致计算成本高昂，并对可重复性和可访问性构成重大障碍，特别是在学术实验室中。为解决这些挑战，我们研究了数据修剪对BioFM预训练的可行性，并提出了一种专门针对生物领域的后验影响引导数据修剪框架。我们的方法引入了一种基于子集的自影响公式，能够以低计算成本高效估计样本重要性，并在此基础上构建了两种简单而有效的选择策略，即Top-k影响和以覆盖为中心的影响。我们在两个代表性的BioFMs上实证验证了我们的方法。对于RNA，在超过99%的极端修剪率下，我们的框架始终优于随机选择基线，证明了其有效性。此外，我们使用ESM-C在蛋白质相关任务上展示了我们框架的通用性。特别是在RNA和蛋白质设置中，我们的核心集甚至比十倍大的随机子集表现更好，揭示了生物序列数据集中存在大量冗余。这些发现强调了影响引导的数据修剪在显著降低BioFM预训练计算成本方面的潜力，为更高效、可访问和可持续的生物AI研究铺平了道路。


### 论文摘要

Biological foundation models (BioFMs), pretrained on large-scale biological sequences, have recently shown strong potential in providing meaningful representations for diverse downstream bioinformatics tasks. However, such models often rely on millions to billions of training sequences and billions of parameters, resulting in prohibitive computational costs and significant barriers to reproducibility and accessibility, particularly for academic labs. To address these challenges, we investigate the feasibility of data pruning for BioFM pretraining and propose a post-hoc influence-guided data pruning framework tailored to biological domains. Our approach introduces a subset-based self-influence formulation that enables efficient estimation of sample importance at low computational cost, and builds upon it two simple yet effective selection strategies, namely Top-k Influence (Top I) and Coverage-Centric Influence (CCI). We empirically validate our method on two representative BioFMs, RNA-FM and ESM-C. For RNA, our framework consistently outperforms random selection baselines under an extreme pruning rate of over 99 percent, demonstrating its effectiveness. Furthermore, we show the generalizability of our framework on protein-related tasks using ESM-C. In particular, our coreset even outperforms random subsets that are ten times larger in both RNA and protein settings, revealing substantial redundancy in biological sequence datasets. These findings underscore the potential of influence-guided data pruning to substantially reduce the computational cost of BioFM pretraining, paving the way for more efficient, accessible, and sustainable biological AI research.

---

## 99. Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification

**论文链接:** [http://arxiv.org/abs/2512.12887v1](http://arxiv.org/abs/2512.12887v1)

**作者:** Han Liu, Bogdan Georgescu, Yanbo Zhang, Youngjin Yoo, Michael Baumgartner, Riqiang Gao, Jianing Wang, Gengyan Zhao, Eli Gibson, Dorin Comaniciu, Sasa Grbic

**发布时间:** 2025-12-15

**备注:** 1st Place in VLM3D Challenge

### GPT解析

### 总结

本文提出AnyMC3D，一种从2D基础模型适配的可扩展3D分类器，解决了医学基础模型在3D医学图像分类中的三个关键缺陷：数据集偏差、次优适应和任务覆盖不足。

### 背景

3D医学图像分类对现代临床工作流程至关重要。医学基础模型(FMs)作为扩展到新任务的有前途方法，但当前研究存在三个关键缺陷：数据集偏差、次优适应和任务覆盖不足。

### 目的

解决医学基础模型在3D医学图像分类中的关键缺陷，并提出一种可扩展的3D分类器框架。

### 方法

提出AnyMC3D，通过在单一冻结主干上添加轻量级插件(每个任务约100万参数)来高效扩展到新任务。该框架支持多视图输入、辅助像素级监督和可解释热图生成。作者建立了包含12个任务的全面基准，涵盖多种病理、解剖结构和模态。

### 主要发现

分析揭示了三个关键见解：(1)有效的适应对于释放基础模型潜力至关重要；(2)如果适当适应，通用基础模型可以匹配医学专用基础模型；(3)基于2D的方法在3D分类中优于3D架构。

### 结论

首次证明了使用单一可扩展框架在各种应用中实现最先进性能的可行性(包括在VLM3D挑战赛中获得第一名)，消除了对特定任务模型的单独需求。

### 翻译

3D医学图像分类对现代临床工作流程至关重要。医学基础模型(FMs)已成为扩展到新任务的一种有前途的方法，然而当前研究存在三个关键缺陷：数据集偏差、次优适应和任务覆盖不足。在本文中，我们解决了这些缺陷，并引入了AnyMC3D，一种从2D FMs适配的可扩展3D分类器。我们的方法通过在单一冻结主干上仅添加轻量级插件(每个任务约100万参数)来高效扩展到新任务。这个多功能的框架还支持多视图输入、辅助像素级监督和可解释热图生成。我们建立了一个包含12个任务的全面基准，涵盖不同的病理、解剖结构和模态，并系统分析了最先进的3D分类技术。我们的分析揭示了关键见解：(1)有效的适应对于释放FM潜力至关重要，(2)如果适当适应，通用FMs可以匹配医学专用FMs，以及(3)对于3D分类，基于2D的方法优于3D架构。我们首次证明了使用单一可扩展框架在各种应用中实现最先进性能的可行性(包括在VLM3D挑战赛中获得第一名)，消除了对单独的特定任务模型的需求。


### 论文摘要

3D medical image classification is essential for modern clinical workflows. Medical foundation models (FMs) have emerged as a promising approach for scaling to new tasks, yet current research suffers from three critical pitfalls: data-regime bias, suboptimal adaptation, and insufficient task coverage. In this paper, we address these pitfalls and introduce AnyMC3D, a scalable 3D classifier adapted from 2D FMs. Our method scales efficiently to new tasks by adding only lightweight plugins (about 1M parameters per task) on top of a single frozen backbone. This versatile framework also supports multi-view inputs, auxiliary pixel-level supervision, and interpretable heatmap generation. We establish a comprehensive benchmark of 12 tasks covering diverse pathologies, anatomies, and modalities, and systematically analyze state-of-the-art 3D classification techniques. Our analysis reveals key insights: (1) effective adaptation is essential to unlock FM potential, (2) general-purpose FMs can match medical-specific FMs if properly adapted, and (3) 2D-based methods surpass 3D architectures for 3D classification. For the first time, we demonstrate the feasibility of achieving state-of-the-art performance across diverse applications using a single scalable framework (including 1st place in the VLM3D challenge), eliminating the need for separate task-specific models.

---

## 100. SAGA: Open-World Mobile Manipulation via Structured Affordance Grounding

**论文链接:** [http://arxiv.org/abs/2512.12842v1](http://arxiv.org/abs/2512.12842v1)

**作者:** Kuan Fang, Yuxin Chen, Xinghao Zhu, Farzad Niroui, Lingfeng Sun, Jiuguang Wang

**发布时间:** 2025-12-14

**备注:** 9 pages, 7 figures

### GPT解析

### 总结

SAGA是一个通用且自适应的视觉运动控制框架，能够跨不同环境、任务目标和用户规范进行泛化，通过解耦高层语义意图与低层视觉运动控制实现高效学习。

### 背景

机器人视觉运动控制面临泛化挑战，需要在各种环境和任务条件下有效工作，现有端到端和模块化方法在泛化能力上存在局限。

### 目的

开发一个能够跨多种环境、任务目标和用户规范进行泛化的视觉运动控制框架，实现高效学习和任务执行。

### 方法

采用基于效用的任务表示法，利用多模态基础模型将任务表示锚定到机器人的视觉观察中生成3D效用热图，基于锚定效用训练条件策略实现全身控制，构建统一框架支持多种任务指定形式。

### 主要发现

SAGA在十一个真实世界任务上的实验中显著优于端到端和模块化基线，结构化的效用锚定为通用移动操作提供了可扩展且有效的途径。

### 结论

通过解耦高层语义意图与低层视觉运动控制，并基于观察到的环境明确锚定任务目标，SAGA实现了跨多种环境和任务的高效泛化。

### 翻译

我们提出了SAGA，一个通用且自适应的视觉运动控制框架，可以跨各种环境、任务目标和用户规范进行泛化。为了高效学习这种能力，我们的核心思想是通过在观察到的环境中明确锚定任务目标，将高层语义意图与低层视觉运动控制解耦。使用基于效用的任务表示，我们以统一、结构化的形式表达多样化和复杂的行为。通过利用多模态基础模型，SAGA将提出的任务表示锚定到机器人的视觉观察中，作为3D效用热图，突出显示任务相关实体，同时抽象出会阻碍泛化的表面外观变化。这些锚定的效用使我们能够有效地在多任务演示数据上训练条件策略，实现全身控制。在一个统一框架中，SAGA可以解决以不同形式指定的任务，包括语言指令、选定点和示例演示，实现零样本执行和少样本适应。我们在四足操作器上实例化了SAGA，并在十一个真实世界任务上进行了广泛实验。SAGA始终以显著优势优于端到端和模块化基线。这些结果共同表明，结构化的效用锚定为通用移动操作提供了一种可扩展且有效的途径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何让机器人在开放世界中灵活执行移动操作任务，并能泛化到各种环境、任务目标和用户指令形式的问题。这个问题在现实中很重要，因为它关系到机器人能否在家庭、办公室等真实环境中执行广泛任务，提高实用性；在研究中，它是实现真正通用机器人的关键挑战，因为当前系统通常只能在受控环境中执行特定任务，缺乏适应新环境的能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：端到端模型需要大量数据训练，而模块化框架在非结构化环境中不够强大。他们提出将高级语义意图与低级视觉运动控制解耦，通过明确将任务目标与环境相关联来实现开放世界操作。作者借鉴了多模态基础模型用于视觉识别、热力图作为空间任务表示、扩散策略用于视觉运动控制、基于功能的机器人表示以及上下文学习等现有工作，但通过结构化功能接地提供了更有效的解决方案。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过结构化功能接地将高级语义意图与低级视觉运动控制解耦，实现开放世界移动操作。整体流程包括：1)任务表示：使用'功能-实体对'表示任务目标；2)功能接地：将任务表示与视觉观察关联为3D功能热力图；3)条件策略：基于热力图点云训练条件扩散策略预测动作；4)多样化接口：支持语言指令、选定点和示例演示三种用户输入方式；5)移动操作系统：在四足操作平台上实现全身控制。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)结构化功能任务表示，支持多种交互类型和复杂任务组合；2)3D功能热力图接地，明确将任务目标关联到环境；3)多样化用户接口，支持零样本执行和少样本适应；4)高效通用移动操作，使用更少数据实现更好泛化。相比端到端方法，SAGA不需要将所有处理压缩在黑盒模型中；相比模块化方法，不依赖手工设计的低级模块；相比仅关注抓取的方法，支持更广泛的交互类型；相比传统热力图方法，提供更丰富的语义表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SAGA通过结构化功能接地将高级语义意图与低级视觉运动控制解耦，实现了一种高效、通用的开放世界移动操作框架，仅需少量演示数据即可在各种环境和任务目标中表现出强大的泛化能力。'}


### 论文摘要

We present SAGA, a versatile and adaptive framework for visuomotor control that can generalize across various environments, task objectives, and user specifications. To efficiently learn such capability, our key idea is to disentangle high-level semantic intent from low-level visuomotor control by explicitly grounding task objectives in the observed environment. Using an affordance-based task representation, we express diverse and complex behaviors in a unified, structured form. By leveraging multimodal foundation models, SAGA grounds the proposed task representation to the robot's visual observation as 3D affordance heatmaps, highlighting task-relevant entities while abstracting away spurious appearance variations that would hinder generalization. These grounded affordances enable us to effectively train a conditional policy on multi-task demonstration data for whole-body control. In a unified framework, SAGA can solve tasks specified in different forms, including language instructions, selected points, and example demonstrations, enabling both zero-shot execution and few-shot adaptation. We instantiate SAGA on a quadrupedal manipulator and conduct extensive experiments across eleven real-world tasks. SAGA consistently outperforms end-to-end and modular baselines by substantial margins. Together, these results demonstrate that structured affordance grounding offers a scalable and effective pathway toward generalist mobile manipulation.

---

## 101. Generative Spatiotemporal Data Augmentation

**论文链接:** [http://arxiv.org/abs/2512.12508v1](http://arxiv.org/abs/2512.12508v1)

**作者:** Jinfan Zhou, Lixin Luo, Sungmin Eum, Heesung Kwon, Jeong Joon Park

**发布时间:** 2025-12-14

### GPT解析

### 总结

该研究探索使用视频基础模型进行时空数据增强，以多样化相机视角和场景动态，并在低数据设置中取得了性能提升。

### 背景

现有的数据增强方法主要基于简单的几何变换或外观扰动，难以生成多样化的3D空间和时间变化。

### 目的

利用现成的视频扩散模型从给定的图像数据集中生成真实的3D空间和时间变化，作为补充训练数据以提高模型性能。

### 方法

利用现成的视频扩散模型生成合成视频片段，将其作为补充训练数据，并提供实践指导包括选择适当的时空生成设置、将标注转移到合成帧以及处理遮挡区域。

### 主要发现

在COCO子集和无人机捕获数据集上的实验表明，合理应用时空增强可以沿着传统和先前生成方法代表性不足的轴扩展数据分布。

### 结论

时空数据增强为提高数据稀缺情况下的模型性能提供了有效的手段。

### 翻译

我们探索使用视频基础模型进行时空数据增强，以多样化相机视角和场景动态。与基于简单几何变换或外观扰动的现有方法不同，我们的方法利用现成的视频扩散模型从给定的图像数据集中生成真实的3D空间和时间变化。将这些合成的视频片段作为补充训练数据，在低数据设置（如标注稀缺的无人机捕获图像）中能带来一致的性能提升。除了实证改进，我们还提供了实践指导，包括选择适当的时空生成设置、将标注转移到合成帧以及处理遮挡区域（在生成的视图中新出现且未标记的区域）。在COCO子集和无人机捕获数据集上的实验表明，合理应用时空增强可以沿着传统和先前生成方法代表性不足的轴扩展数据分布，为提高数据稀缺情况下的模型性能提供了有效的手段。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决计算机视觉任务中数据稀缺和多样性不足的问题。在现实世界中，收集多视角、多场景动态的数据（特别是无人机、机器人等领域）需要昂贵设备和大量人力，手动标注成本高昂。传统数据增强方法仅能进行简单几何变换或外观扰动，无法有效增加数据的时空多样性，限制了模型在数据有限场景下的性能表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现代视频扩散模型编码了强大的隐式3D和动态先验知识，可以直接用于感知任务。他们思考如何利用这些现成的视频扩散模型作为数据增强引擎，从单张输入图像生成新的视角和合理的时间变化。方法设计借鉴了现有工作：利用DimensionX和LTX-Video-13B等视频扩散模型进行数据生成，使用SAM 2视频分割模型进行自动标注，以及采用TAPNext和All-Tracker等点跟踪技术处理遮挡问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用现成的视频扩散模型作为时空数据增强引擎，从单张图像生成具有不同视角和动态变化的视频序列，并通过自动化标注传递将原始图像的标注信息传递到生成的视频帧中。整体流程包括：1) 数据生成阶段，使用视频扩散模型生成空间视角变化或时间动态变化的视频；2) 标注传递阶段，使用视频分割模型跟踪物体并生成边界框标注；3) 遮挡处理阶段，通过点跟踪确定可见区域；4) 模型训练阶段，结合原始数据和生成数据进行训练，保持1:1采样比例。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次证明现成的视频扩散模型可以生成合成数据并显著提高物体检测性能；2) 提供完全自动化的标注传递流程，利用时间一致性为所有生成帧生成准确边界框；3) 通过大量实验提供关于如何有效使用时空增强的实用指南。相比之前工作，本文不仅关注视觉外观多样化，更专注于空间视角和时间动态的多样性；不局限于位置不变的增强或有给定边界框的物体生成，而是生成连续视频帧并使用基础模型进行自动标注，提供了更大的灵活性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于视频扩散模型的时空数据增强方法，通过从单张图像生成多视角和动态变化的视频序列，显著提高了数据稀缺环境下物体检测的性能，并提供了实用的自动化标注传递流程。'}


### 论文摘要

We explore spatiotemporal data augmentation using video foundation models to diversify both camera viewpoints and scene dynamics. Unlike existing approaches based on simple geometric transforms or appearance perturbations, our method leverages off-the-shelf video diffusion models to generate realistic 3D spatial and temporal variations from a given image dataset. Incorporating these synthesized video clips as supplemental training data yields consistent performance gains in low-data settings, such as UAV-captured imagery where annotations are scarce. Beyond empirical improvements, we provide practical guidelines for (i) choosing an appropriate spatiotemporal generative setup, (ii) transferring annotations to synthetic frames, and (iii) addressing disocclusion - regions newly revealed and unlabeled in generated views. Experiments on COCO subsets and UAV-captured datasets show that, when applied judiciously, spatiotemporal augmentation broadens the data distribution along axes underrepresented by traditional and prior generative methods, offering an effective lever for improving model performance in data-scarce regimes.

---

## 102. BokehDepth: Enhancing Monocular Depth Estimation through Bokeh Generation

**论文链接:** [http://arxiv.org/abs/2512.12425v1](http://arxiv.org/abs/2512.12425v1)

**作者:** Hangwei Zhang, Armando Teles Fortes, Tianyi Wei, Xingang Pan

**发布时间:** 2025-12-13

### GPT解析

### 总结

BokehDepth是一个两阶段框架，解耦了Bokeh合成与深度预测，将散焦作为辅助的无监督几何线索，在提高Bokeh渲染质量的同时增强了单目深度估计的准确性和鲁棒性。

### 背景

Bokeh和单目深度估计通过相同的镜头成像几何紧密耦合，但当前方法利用这种连接不完整。高质量Bokeh渲染依赖噪声深度图，放大误差为可见伪影；现代单目深度模型在弱纹理、远距离和几何模糊区域表现不佳，而这些区域恰恰是散焦线索最有信息量的地方。

### 目的

解决Bokeh合成与深度预测之间的耦合问题，将散焦作为一种辅助的、无需监督的几何线索，同时提高Bokeh渲染质量和深度估计准确性。

### 方法

提出BokehDepth两阶段框架：第一阶段使用物理引导的可控Bokeh生成器，基于预训练图像编辑主干网络从单张锐利图像生成无深度Bokeh堆栈；第二阶段将轻量级散焦感知聚合模块插入现有单目深度编码器，沿散焦维度融合特征，同时保持下游解码器不变。

### 主要发现

在具有挑战性的基准测试中，BokehDepth相比基于深度图的Bokeh基线方法提高了视觉保真度，并一致地增强了强大单目深度基础模型的度量准确性和鲁棒性。

### 结论

通过解耦Bokeh合成与深度预测，并利用散焦作为辅助几何线索，可以同时改善Bokeh渲染和深度估计的质量，解决当前方法中存在的局限性。

### 翻译

Bokeh和单目深度估计通过相同的镜头成像几何紧密耦合，然而当前方法以不完整的方式利用这种连接。高质量的Bokeh渲染流程通常依赖于噪声深度图，这会将估计误差放大为可见的伪影，而现代单目度量深度模型在弱纹理、远距离和几何模糊区域仍然表现不佳，这些区域是散焦线索最有信息量的地方。我们引入了BokehDepth，一个两阶段框架，它将Bokeh合成与深度预测解耦，并将散焦作为辅助的无监督几何线索。在第一阶段，基于强大的预训练图像编辑主干网络构建的物理引导的可控Bokeh生成器，从单个锐利输入生成无深度的Bokeh堆栈，并校准Bokeh强度。在第二阶段，一个轻量级的散焦感知聚合模块插入现有的单目深度编码器中，沿散焦维度融合特征，同时暴露稳定的深度敏感变化，且保持下游解码器不变。在具有挑战性的基准测试中，BokehDepth相比基于深度图的Bokeh基线方法提高了视觉保真度，并一致地增强了强大单目深度基础模型的度量准确性和鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决单目深度估计与散景生成之间的耦合关系问题。现有高质量散景渲染依赖噪声深度图，会放大深度估计误差；而现代单目深度模型在弱纹理、远距离和几何模糊区域表现不佳，这些区域恰恰是散景线索最有信息量的地方。这个问题很重要，因为深度估计是计算机视觉的基础任务，对3D场景理解、自动驾驶、增强现实等应用至关重要，而散景效果影响图像美观性和主体突出效果，两者间的物理联系尚未被充分利用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到散景和单目深度估计通过镜头成像几何紧密相连，但现有方法利用这种联系不完整。核心洞察是将散景合成与深度预测解耦，利用散焦作为无监督几何线索。方法采用两阶段框架：第一阶段基于FLUX-Kontext图像编辑模型添加散景交叉注意力适配器生成散景堆栈；第二阶段将Divided Space Focus Attention模块插入现有深度编码器融合散焦线索。借鉴了薄透镜模型、扩散模型和现有深度估计模型的工作，通过统一的散景强度参数K对齐真实和合成数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将散景生成与深度估计解耦，利用物理引导的散景生成提供无监督几何线索增强深度估计。整体流程分两阶段：第一阶段基于FLUX-Kontext，通过物理模型生成不同强度K的散景堆栈；第二阶段将原始图像和散景堆栈输入深度编码器，插入DSFA模块融合散焦特征，保留参考帧特征通过原解码器输出增强后的深度图。DSFA模块包含空间注意力(帧内)和焦点注意力(帧间)，利用散景堆栈中的变化作为深度线索。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)解耦的散景-深度框架，避免传统方法中深度误差放大问题；2)物理引导的无监督散景生成，基于薄透镜模型统一控制参数K；3)散焦感知的DSFA模块，可插入现有深度编码器增强特征融合；4)跨域泛化能力，能提升多种基础深度模型性能。相比之前工作：不依赖深度图的散景生成减少伪影；利用散景线索改善弱纹理区域深度估计；仅需单张图像而非多孔径对；物理模型确保散景与深度间一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BokehDepth通过解耦散景生成与深度估计，并利用物理引导的散景堆栈作为无监督几何线索，显著提升了单目深度估计的准确性和鲁棒性，特别是在传统方法表现不佳的弱纹理和远距离区域。'}


### 论文摘要

Bokeh and monocular depth estimation are tightly coupled through the same lens imaging geometry, yet current methods exploit this connection in incomplete ways. High-quality bokeh rendering pipelines typically depend on noisy depth maps, which amplify estimation errors into visible artifacts, while modern monocular metric depth models still struggle on weakly textured, distant and geometrically ambiguous regions where defocus cues are most informative. We introduce BokehDepth, a two-stage framework that decouples bokeh synthesis from depth prediction and treats defocus as an auxiliary supervision-free geometric cue. In Stage-1, a physically guided controllable bokeh generator, built on a powerful pretrained image editing backbone, produces depth-free bokeh stacks with calibrated bokeh strength from a single sharp input. In Stage-2, a lightweight defocus-aware aggregation module plugs into existing monocular depth encoders, fuses features along the defocus dimension, and exposes stable depth-sensitive variations while leaving downstream decoder unchanged. Across challenging benchmarks, BokehDepth improves visual fidelity over depth-map-based bokeh baselines and consistently boosts the metric accuracy and robustness of strong monocular depth foundation models.

---

## 103. The Data Efficiency Frontier of Financial Foundation Models: Scaling Laws from Continued Pretraining

**论文链接:** [http://arxiv.org/abs/2512.12384v1](http://arxiv.org/abs/2512.12384v1)

**作者:** Jesse Ponnock

**发布时间:** 2025-12-13

**备注:** 8 pages, 4 figures, 1 table

### GPT解析

### 总结

该研究分析了在金融领域继续预训练大型语言模型的效果，发现即使是中等规模的模型也能有效适应特定领域，且不会损失通用能力。

### 背景

领域自适应预训练(DAPT)是一种在不完全重新训练的情况下，使大型语言模型适应高价值领域的实用方法。

### 目的

研究在特定金融领域上继续预训练大型语言模型的效果，探索不同参数规模模型在不同训练量下的表现。

### 方法

使用400M token的金融语料库训练10亿和30亿参数的Llama-3.2模型，在50M、100M、200M和400M token处设置验证检查点，并进行幂律拟合分析。

### 主要发现

两个模型在SEC领域验证损失持续改善，最大收益出现在前200M token；幂律拟合显示金融语言高度规则且可有效学习；通用领域验证损失基本不变，表明没有灾难性遗忘；模型向专业化方向发展，几乎没有混合领域退化。

### 结论

有意义的领域适应可以通过相对适度的token预算实现；在预期的数据需求下，更大规模的模型(7B-70B)仍然可行。

### 翻译

领域自适应预训练(DAPT)为在不完全重新训练的情况下使大型语言模型专业化应用于高价值领域提供了实用路径。我们在美国SEC文件上进行了持续预训练的早期扩展定律分析，使用400M token的金融语料库训练了10亿和30亿参数的Llama-3.2模型，并在50M、100M、200M和400M token处设置了验证检查点。结果显示两个模型在SEC领域验证损失方面均有持续改善，最大收益出现在前200M token，之后收益递减。幂律拟合显示指数较浅，表明金融语言高度规则，在持续预训练下可有效学习。通用领域验证损失在所有token预算下基本保持不变，表明最小漂移和没有灾难性遗忘的迹象。数据效率前沿进一步显示，两个模型都向改善专业化方向发展，几乎没有混合领域退化。总之，这些发现为扩展金融基础模型提供了早期的经验指导，表明有意义的领域适应可以通过相对适度的token预算实现，且在预期的数据需求下更大规模的模型(7B-70B)仍然可行。


### 论文摘要

Domain-adaptive pretraining (DAPT) offers a practical path to specializing large language models for high-value domains without full retraining. We conduct an early-stage scaling-law analysis of continued pretraining on U.S. SEC filings, training 1B and 3B-parameter Llama-3.2 models on a 400M-token financial corpus with validation checkpoints at 50M, 100M, 200M, and 400M tokens. Results show consistent improvements in SEC-domain validation loss for both models, with the largest gains occurring within the first 200M tokens and diminishing returns thereafter. Power-law fits reveal shallow exponents, indicating that financial language is highly regular and efficiently learnable under continued pretraining. General-domain validation loss remains effectively unchanged across all token budgets, suggesting minimal drift and no signs of catastrophic forgetting. A data-efficiency frontier further shows that both models move toward improved specialization with negligible mixed-domain degradation. Together, these findings provide early empirical guidance for scaling financial foundation models, suggesting that meaningful domain adaptation can be achieved with comparatively modest token budgets and that larger model scales (7B-70B) remain tractable under projected data requirements.

---

## 104. EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training

**论文链接:** [http://arxiv.org/abs/2512.12210v1](http://arxiv.org/abs/2512.12210v1)

**作者:** Yuting Tang, Weibang Jiang, Shanglin Li, Yong Li, Chenyu Liu, Xinliang Zhou, Yi Ding, Cuntai Guan

**发布时间:** 2025-12-13

**备注:** Accepted by AAAI-2026

### GPT解析

### 总结

本研究提出EEG-DLite数据蒸馏框架，通过选择性去除大型EEG数据集中的噪声和冗余样本，实现更高效的预训练，仅使用5%的数据即可达到与完整数据集相当甚至更好的性能。

### 背景

大规模EEG基础模型在多种下游任务中表现出强大的泛化能力，但由于EEG数据的数量大、质量参差不齐，其训练仍然非常消耗资源。

### 目的

开发一种数据蒸馏框架，能够从大型EEG数据集中高效筛选出高质量、低冗余的数据子集，减少训练资源需求同时保持模型性能。

### 方法

EEG-DLite首先使用自监督自编码器将EEG段编码为紧凑的潜在表示，然后基于这些表示过滤异常值并最小化冗余，得到一个更小但信息丰富的子集，保留了有效基础模型训练所需的多样性。

### 主要发现

通过大量实验证明，仅使用EEG-DLite筛选的2500小时数据集中5%的数据进行训练，在多个下游任务上可以与使用完整数据集训练的性能相媲美，甚至在某些情况下更好。

### 结论

EEG-DLite为更有效和高效的生理基础建模提供了可扩展且实用的途径，这是首次对EEG基础模型预训练数据蒸馏的系统性研究。

### 翻译

大规模脑电图基础模型已在多种下游任务中展现出强大的泛化能力，但由于脑电图数据的数量和质量参差不齐，其训练仍然资源密集。在本工作中，我们引入了EEG-DLite，这是一种数据蒸馏框架，通过从大型脑电图数据集中选择性去除噪声和冗余样本，实现更高效的预训练。EEG-DLite首先使用自监督自编码器将脑电图段编码为紧凑的潜在表示，使样本选择能够高效进行并降低对噪声的敏感性。基于这些表示，EEG-DLite过滤异常值并最小化冗余，产生一个更小但信息丰富的子集，保留了有效基础模型训练所需的多样性。通过大量实验，我们证明仅使用EEG-DLite筛选的2500小时数据集中5%的数据进行训练，在多个下游任务上能够达到与使用完整数据集训练相当甚至更好的性能。据我们所知，这是首次在脑电图基础模型背景下对预训练数据蒸馏的系统性研究。EEG-DLite为更有效和高效的生理基础建模提供了可扩展且实用的途径。代码可在https://github.com/t170815518/EEG-DLite获取。


### 论文摘要

Large-scale EEG foundation models have shown strong generalization across a range of downstream tasks, but their training remains resource-intensive due to the volume and variable quality of EEG data. In this work, we introduce EEG-DLite, a data distillation framework that enables more efficient pre-training by selectively removing noisy and redundant samples from large EEG datasets. EEG-DLite begins by encoding EEG segments into compact latent representations using a self-supervised autoencoder, allowing sample selection to be performed efficiently and with reduced sensitivity to noise. Based on these representations, EEG-DLite filters out outliers and minimizes redundancy, resulting in a smaller yet informative subset that retains the diversity essential for effective foundation model training. Through extensive experiments, we demonstrate that training on only 5 percent of a 2,500-hour dataset curated with EEG-DLite yields performance comparable to, and in some cases better than, training on the full dataset across multiple downstream tasks. To our knowledge, this is the first systematic study of pre-training data distillation in the context of EEG foundation models. EEG-DLite provides a scalable and practical path toward more effective and efficient physiological foundation modeling. The code is available at https://github.com/t170815518/EEG-DLite.

---

## 105. EchoVLM: Measurement-Grounded Multimodal Learning for Echocardiography

**论文链接:** [http://arxiv.org/abs/2512.12107v1](http://arxiv.org/abs/2512.12107v1)

**作者:** Yuheng Li, Yue Zhang, Abdoul Aziz Amadou, Yuxiang Lai, Jike Zhong, Tiziano Passerini, Dorin Comaniciu, Puneet Sharma

**发布时间:** 2025-12-13

### GPT解析

### 总结

本研究提出了EchoVLM，一种基于新型数据集EchoGround-MIMIC的视觉语言模型，用于超声心动图解释。该模型结合了视图信息对比损失和否定感知对比损失，在多种临床任务上取得了最先进的性能。

### 背景

超声心动图是心脏病学中最广泛使用的成像方式，但其解释劳动密集且多模态。现有视觉语言模型在超声心动图领域的应用受到缺乏大规模临床数据集和基于测量的推理的限制。

### 目的

创建首个基于测量的多模态超声心动图数据集EchoGround-MIMIC，并开发EchoVLM视觉语言模型以实现端到端的超声心动图解释。

### 方法

构建包含19,065个图像-文本对的数据集，标准化视图、结构化测量和疾病标签。EchoVLM采用两种新型预训练目标：视图信息对比损失和否定感知对比损失。

### 主要发现

EchoVLM在36项任务上实现最先进性能，包括零样本疾病分类(AUC 86.5%)和视图分类(准确率95.1%)。临床基础的多模态预训练产生可转移的视觉表示。

### 结论

EchoVLM被确立为端到端超声心动图解释的基础模型，EchoGround-MIMIC和数据整理代码将被发布以促进可重复性和进一步研究。

### 翻译

超声心动图是心脏病学中最广泛使用的成像方式，但其解释仍然劳动密集且本质上多模态，需要视图识别、定量测量、定性评估和基于指南的推理。虽然最近的视觉语言模型在自然图像和某些医学领域取得了广泛成功，但它们在超声心动图方面的潜力受到缺乏大规模、临床基础的图像-文本数据集以及缺乏超声解释中基于测量的推理的限制。我们介绍了EchoGround-MIMIC，第一个基于测量的多模态超声心动图数据集，包含来自1,572名患者的19,065个图像-文本对，具有标准化视图、结构化测量、基于测量的标题和从指南中派生的疾病标签。基于此资源，我们提出了EchoVLM，一种视觉语言模型，它包含两个新的预训练目标：(i) 编码超声心动图成像视图相关结构的视图信息对比损失，以及(ii) 区分临床上关键的阴性发现和阳性发现的否定感知对比损失。在涵盖多模态疾病分类、图像-文本检索、视图分类、心室分割和landmark检测的五类临床应用和36项任务中，EchoVLM取得了最先进的性能（零样本疾病分类的AUC为86.5%，视图分类的准确率为95.1%）。我们证明了临床基础的多模态预训练产生可转移的视觉表示，并将EchoVLM确立为端到端超声心动图解释的基础模型。我们将发布EchoGround-MIMIC和数据整理代码，实现可重复性和多模态超声心动图解释的进一步研究。


### 论文摘要

Echocardiography is the most widely used imaging modality in cardiology, yet its interpretation remains labor-intensive and inherently multimodal, requiring view recognition, quantitative measurements, qualitative assessments, and guideline-based reasoning. While recent vision-language models (VLMs) have achieved broad success in natural images and certain medical domains, their potential in echocardiography has been limited by the lack of large-scale, clinically grounded image-text datasets and the absence of measurement-based reasoning central to echo interpretation. We introduce EchoGround-MIMIC, the first measurement-grounded multimodal echocardiography dataset, comprising 19,065 image-text pairs from 1,572 patients with standardized views, structured measurements, measurement-grounded captions, and guideline-derived disease labels. Building on this resource, we propose EchoVLM, a vision-language model that incorporates two novel pretraining objectives: (i) a view-informed contrastive loss that encodes the view-dependent structure of echocardiographic imaging, and (ii) a negation-aware contrastive loss that distinguishes clinically critical negative from positive findings. Across five types of clinical applications with 36 tasks spanning multimodal disease classification, image-text retrieval, view classification, chamber segmentation, and landmark detection, EchoVLM achieves state-of-the-art performance (86.5% AUC in zero-shot disease classification and 95.1% accuracy in view classification). We demonstrate that clinically grounded multimodal pretraining yields transferable visual representations and establish EchoVLM as a foundation model for end-to-end echocardiography interpretation. We will release EchoGround-MIMIC and the data curation code, enabling reproducibility and further research in multimodal echocardiography interpretation.

---

## 106. RePack: Representation Packing of Vision Foundation Model Features Enhances Diffusion Transformer

**论文链接:** [http://arxiv.org/abs/2512.12083v1](http://arxiv.org/abs/2512.12083v1)

**作者:** Guanfang Dong, Luke Schultz, Negar Hassanpour, Chao Gao

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出了一种名为RePack的表示打包框架，用于解决预训练视觉基础模型增强扩散模型时的高维度表示导致的信息过载问题。RePack通过将VFM表示投影到低维流形，实现更紧凑的表示，加速模型收敛并提高生成性能。

### 背景

预训练视觉基础模型具有优异的表示能力，已被用于增强潜在扩散模型。这些方法将高维VFM表示注入到LDMs的不同阶段，加速学习并提高生成性能。然而，VFM表示的高维度可能导致信息过载，特别是当VFM特征超过原始图像大小时进行解码。

### 目的

解决VFM表示的高维度导致的信息过载问题，同时保留VFM特征的效用，提高扩散模型特别是DiTs的性能和收敛速度。

### 方法

提出RePack框架，通过将VFM表示投影到低维流形，将其转换为更紧凑的、解码器友好的表示。这种方法能够过滤非语义噪声，同时保留高保真重建所需的核心结构信息。

### 主要发现

RePack可以有效过滤非语义噪声，保留核心结构信息，显著加速DiT收敛。实验表明，RePack性能优于直接将原始VFM特征注入解码器的方法。在DiT-XL/2上，仅用64个周期就实现了3.66的FID，比最先进方法快35%。

### 结论

RePack成功提取了VFM表示的核心语义，同时避免了其高维度的副作用，是一种简单有效的改进扩散模型的方法。

### 翻译

预训练视觉基础模型的优异表示能力已被用于增强潜在扩散模型。这些方法在不同阶段将高维VFM表示注入LDMs，从而加速学习并提高生成性能。然而，VFM表示的高维度也可能导致信息过载，特别是当VFM特征超过原始图像大小时进行解码。为解决这一问题同时保留VFM特征的效用，我们提出了RePack，一个简单有效的改进扩散Transformer的框架。RePack通过将VFM表示投影到低维流形，将其转换为更紧凑、解码器友好的表示。我们发现RePack可以有效过滤非语义噪声，同时保留高保真重建所需的核心结构信息。实验结果表明，RePack显著加速了DiT收敛，并优于最近将原始VFM特征直接注入解码器进行图像重建的方法。在DiT-XL/2上，RePack仅用64个周期就实现了3.66的FID，比最先进方法快35%。这证明了RePack成功提取了VFM表示的核心语义，同时避免了其高维度的副作用。


### 论文摘要

The superior representation capability of pre-trained vision foundation models (VFMs) has been harnessed for enhancing latent diffusion models (LDMs). These approaches inject the rich semantics from high-dimensional VFM representations (e.g., DINOv3) into LDMs at different phases, resulting in accelerated learning and better generation performance. However, the high-dimensionality of VFM representations may also lead to Information Overload, particularly when the VFM features exceed the size of the original image for decoding. To address this issue while preserving the utility of VFM features, we propose RePack (Representation Packing), a simple yet effective framework for improving Diffusion Transformers (DiTs). RePack transforms the VFM representation into a more compact, decoder-friendly representation by projecting onto low-dimensional manifolds. We find that RePack can effectively filter out non-semantic noise while preserving the core structural information needed for high-fidelity reconstruction. Experimental results show that RePack significantly accelerates DiT convergence and outperforms recent methods that directly inject raw VFM features into the decoder for image reconstruction. On DiT-XL/2, RePack achieves an FID of 3.66 in only 64 epochs, which is 35% faster than the state-of-the-art method. This demonstrates that RePack successfully extracts the core semantics of VFM representations while bypassing their high-dimensionality side effects.

---

## 107. CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction

**论文链接:** [http://arxiv.org/abs/2512.11988v1](http://arxiv.org/abs/2512.11988v1)

**作者:** Xianghui Xie, Bowen Wen, Yan Chang, Hesam Rabeti, Jiefeng Li, Ye Yuan, Gerard Pons-Moll, Stan Birchfield

**发布时间:** 2025-12-12

**备注:** 14 pages, 8 figures, 4 tables. Project page: https://nvlabs.github.io/CARI4D/

### GPT解析

### 总结

本文提出了CARI4D，首个类别无关的方法，可以从单目RGB视频中重建空间和时间一致的4D人-物交互，达到度量尺度。

### 背景

从常见的RGB传感器准确捕捉人-物交互对于人类理解、游戏和机器人学习应用非常重要。

### 目的

解决从单一RGB视图推断4D交互的挑战，包括未知物体和人类信息、深度歧义、遮挡和复杂运动等问题。

### 方法

提出姿态假设选择算法，稳健集成基础模型的个体预测；通过学习的渲染和比较范式联合优化预测，确保空间、时间和像素对齐；推理复杂接触点以进一步优化，满足物理约束。

### 主要发现

在重建误差方面，CARI4D在分布内数据集上比之前的方法好38%，在未见过的数据集上好36%；模型可以泛化到训练类别之外，可以零样本应用于野外网络视频。

### 结论

CARI4D是首个类别无关的4D人-物交互重建方法，能够从单目RGB视频中重建空间和时间一致的交互，具有很好的泛化能力。

### 翻译

从常见的RGB传感器准确捕捉人-物交互对于人类理解、游戏和机器人学习应用非常重要。然而，由于物体和人类信息未知、深度歧义、遮挡和复杂运动等因素，从单一RGB视图推断4D交互极具挑战性，这阻碍了一致的3D和时间重建。先前的方法通过假设真实物体模板或限制到有限物体类别集合来简化设置。我们提出了CARI4D，这是首个类别无关的方法，可以从单目RGB视频中重建空间和时间一致的4D人-物交互，达到度量尺度。为此，我们提出了一种姿态假设选择算法，稳健地集成基础模型的个体预测，通过学习的渲染和比较范式联合优化它们，确保空间、时间和像素对齐，最后推理复杂的接触点以进一步优化，满足物理约束。实验表明，在重建误差方面，我们的方法在分布内数据集上比之前的方法好38%，在未见过的数据集上好36%。我们的模型泛化能力超越了训练类别，因此可以零样本应用于野外网络视频。我们的代码和预训练模型将公开发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从单目RGB视频中重建四维(3D空间+时间)人类-物体交互的问题。这个问题在现实中很重要，因为传统的人类-物体交互捕获需要昂贵且繁琐的多视角相机设置，难以扩展；而单目RGB视频丰富易得，能快速经济地获取数据，对人类理解、游戏、机器人学习等应用至关重要。现有方法要么需要真实物体模板，要么限制在特定物体类别，缺乏泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出从单目RGB视频中重建人类-物体交互的挑战：未知物体和人类信息、深度歧义、遮挡和复杂运动。他们指出现有方法的局限性，然后设计了一个类别无关的方法。作者借鉴了多个基础模型：使用Hunyuan3D-2进行物体重建，UniDepth进行深度估计，FoundationPose进行物体姿态估计，NLF进行人体姿态估计。将这些模型的预测整合到一个统一框架中，并设计了特定算法处理不一致性和噪声。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一个框架整合基础模型预测获得鲁棒初始化，再通过类别无关的交互推理模块提高接触一致性，使用渲染和比较范式优化人类和物体姿态。整体流程包括：1)物体重建和度量尺度估计；2)人类和物体姿态初始化；3)接触推理和细化(CoCoNet)；4)基于接触的联合优化。这种方法能够在不依赖物体模板的情况下，从单目RGB视频中重建空间和时间一致的4D人类-物体交互。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个类别无关的4D人类-物体交互重建方法；2)姿态假设选择算法，能在严重遮挡下鲁棒跟踪物体姿态；3)CoCoNet，类别无关的接触推理模型；4)接触感知的联合优化框架。相比之前工作，CARI4D不需要真实物体模板或预定义类别，能处理各种物体类别，实现零样本泛化，在空间和时间维度上保持一致重建，并能处理复杂遮挡，在重建误差上比现有方法提高36-38%。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CARI4D首次实现了从单目RGB视频中类别无关地重建四维人类-物体交互，无需物体模板即可处理未见过的物体类别，并在重建精度和泛化能力上显著优于现有方法。'}


### 论文摘要

Accurate capture of human-object interaction from ubiquitous sensors like RGB cameras is important for applications in human understanding, gaming, and robot learning. However, inferring 4D interactions from a single RGB view is highly challenging due to the unknown object and human information, depth ambiguity, occlusion, and complex motion, which hinder consistent 3D and temporal reconstruction. Previous methods simplify the setup by assuming ground truth object template or constraining to a limited set of object categories. We present CARI4D, the first category-agnostic method that reconstructs spatially and temporarily consistent 4D human-object interaction at metric scale from monocular RGB videos. To this end, we propose a pose hypothesis selection algorithm that robustly integrates the individual predictions from foundation models, jointly refine them through a learned render-and-compare paradigm to ensure spatial, temporal and pixel alignment, and finally reasoning about intricate contacts for further refinement satisfying physical constraints. Experiments show that our method outperforms prior art by 38% on in-distribution dataset and 36% on unseen dataset in terms of reconstruction error. Our model generalizes beyond the training categories and thus can be applied zero-shot to in-the-wild internet videos. Our code and pretrained models will be publicly released.

---

## 108. HMPCC: Human-Aware Model Predictive Coverage Control

**论文链接:** [http://arxiv.org/abs/2512.12717v1](http://arxiv.org/abs/2512.12717v1)

**作者:** Mattia Catellani, Marta Gabbi, Lorenzo Sabattini

**发布时间:** 2025-12-14

### GPT解析

### 总结

该研究提出了一种基于模型预测控制的人机感知覆盖框架(HMPCC)，用于协调机器人团队在未知环境中安全高效地进行覆盖任务，同时避免与人类等非合作代理的碰撞。

### 背景

传统覆盖策略通常依赖简化假设，如已知或凸环境以及静态密度函数，难以适应真实世界场景，特别是当涉及人类时。在人类存在的情况下，机器人团队协调覆盖环境面临挑战。

### 目的

解决机器人团队协调覆盖未知环境的问题，确保安全操作并避免与非合作代理碰撞，使机器人能够适应动态条件，特别是与人类互动的情况。

### 方法

提出一种基于模型预测控制(MPC)的人机感知覆盖框架(HMPCC)，将人类运动预测集成到规划过程中；在MPC范围内预测人类轨迹，使机器人能够主动协调行动；环境建模为高斯混合模型(GMM)；团队成员以完全去中心化的方式运作，不依赖显式通信。

### 主要发现

人类轨迹预测使覆盖更加高效和自适应，改善了人类和机器人代理之间的协调。

### 结论

所提出的框架能够更好地适应动态环境，特别是与人类互动的情况；去中心化方法在通信受限或敌对场景中具有优势。

### 翻译

我们解决了协调机器人团队覆盖未知环境的问题，同时确保安全操作并避免与非合作代理的碰撞。传统的覆盖策略通常依赖简化假设，如已知或凸环境以及静态密度函数，并且难以适应真实世界场景，特别是当涉及人类时。在这项工作中，我们提出了一种基于模型预测控制(MPC)的人机感知覆盖框架，即HMPCC，其中人类运动预测被集成到规划过程中。通过在MPC范围内预测人类轨迹，机器人可以主动协调其行动，避免冗余探索，并适应动态条件。环境被建模为高斯混合模型(GMM)，表示感兴趣区域。团队成员以完全去中心化的方式运作，不依赖显式通信，这在敌对或通信受限场景中是 essential 特征。我们的结果表明，人类轨迹预测使覆盖更加高效和自适应，改善了人类和机器人代理之间的协调。


### 论文摘要

We address the problem of coordinating a team of robots to cover an unknown environment while ensuring safe operation and avoiding collisions with non-cooperative agents. Traditional coverage strategies often rely on simplified assumptions, such as known or convex environments and static density functions, and struggle to adapt to real-world scenarios, especially when humans are involved. In this work, we propose a human-aware coverage framework based on Model Predictive Control (MPC), namely HMPCC, where human motion predictions are integrated into the planning process. By anticipating human trajectories within the MPC horizon, robots can proactively coordinate their actions %avoid redundant exploration, and adapt to dynamic conditions. The environment is modeled as a Gaussian Mixture Model (GMM), representing regions of interest. Team members operate in a fully decentralized manner, without relying on explicit communication, an essential feature in hostile or communication-limited scenarios. Our results show that human trajectory forecasting enables more efficient and adaptive coverage, improving coordination between human and robotic agents.

---

## 109. WAM-Diff: A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.11872v1](http://arxiv.org/abs/2512.11872v1)

**作者:** Mingwang Xu, Jiahao Cui, Feipeng Cai, Hanlin Shang, Zhihao Zhu, Shan Luan, Yifang Xu, Neng Zhang, Yaoyi Li, Jia Cai, Siyu Zhu

**发布时间:** 2025-12-06

### GPT解析

### 总结

本文提出了WAM-Diff，一个基于掩码扩散的视觉-语言-动作框架，用于自动驾驶中的轨迹生成，通过三个创新实现了高性能：掩码扩散的系统性适应、稀疏MoE架构扩展模型容量，以及使用GSPO的在线强化学习。

### 背景

基于视觉-语言-动作模型的端到端自动驾驶系统整合多模态传感器输入和语言指令以生成规划和控制信号。虽然自回归大语言模型和连续扩散策略很普遍，但离散掩码扩散在轨迹生成方面的潜力尚未得到充分探索。

### 目的

探索离散掩码扩散在自动驾驶轨迹生成中的应用潜力，开发一个有效的VLA框架，能够处理多模态输入并生成高质量的轨迹规划。

### 方法

提出WAM-Diff框架，采用掩码扩散来迭代优化表示未来自车轨迹的离散序列。方法包括：掩码扩散的系统性适应以支持灵活的非因果解码顺序；通过稀疏MoE架构扩展模型容量，该架构在运动预测和面向驾驶的视觉问答上进行联合训练；使用组序列策略优化进行在线强化学习，以优化序列级别的驾驶奖励。

### 主要发现

在NAVSIM-v1上达到91.0 PDMS，在NAVSIM-v2上达到89.7 EPDMS，证明了掩码扩散在自动驾驶中的有效性。该方法是自回归和基于扩散的策略的有前途的替代方案，支持场景感知的解码策略用于轨迹生成。

### 结论

掩码扩散在自动驾驶轨迹生成中具有巨大潜力，WAM-Diff框架通过三个关键创新实现了高性能，为自动驾驶领域提供了新的方法。

### 翻译

基于视觉-语言-动作模型的端到端自动驾驶系统整合多模态传感器输入和语言指令以生成规划和控制信号。虽然自回归大语言模型和连续扩散策略很普遍，但离散掩码扩散在轨迹生成方面的潜力尚未得到充分探索。本文提出了WAM-Diff，一个采用掩码扩散迭代优化表示未来自车轨迹的离散序列的VLA框架。我们的方法有三个关键创新：系统性地适应自动驾驶的掩码扩散，支持灵活的非因果解码顺序；通过稀疏MoE架构扩展模型容量，该架构在运动预测和面向驾驶的视觉问答上进行联合训练；使用组序列策略优化进行在线强化学习，以优化序列级别的驾驶奖励。值得注意的是，我们的模型在NAVSIM-v1上达到91.0 PDMS，在NAVSIM-v2上达到89.7 EPDMS，证明了掩码扩散在自动驾驶中的有效性。该方法为自回归和基于扩散的策略提供了有前途的替代方案，支持场景感知的解码策略用于轨迹生成。本文的代码将在https://github.com/fudan-generative-vision/WAM-Diff公开发布。


### 论文摘要

End-to-end autonomous driving systems based on vision-language-action (VLA) models integrate multimodal sensor inputs and language instructions to generate planning and control signals. While autoregressive large language models and continuous diffusion policies are prevalent, the potential of discrete masked diffusion for trajectory generation remains largely unexplored. This paper presents WAM-Diff, a VLA framework that employs masked diffusion to iteratively refine a discrete sequence representing future ego-trajectories. Our approach features three key innovations: a systematic adaptation of masked diffusion for autonomous driving that supports flexible, non-causal decoding orders; scalable model capacity via a sparse MoE architecture trained jointly on motion prediction and driving-oriented visual question answering (VQA); and online reinforcement learning using Group Sequence Policy Optimization (GSPO) to optimize sequence-level driving rewards. Remarkably, our model achieves 91.0 PDMS on NAVSIM-v1 and 89.7 EPDMS on NAVSIM-v2, demonstrating the effectiveness of masked diffusion for autonomous driving. The approach provides a promising alternative to autoregressive and diffusion-based policies, supporting scenario-aware decoding strategies for trajectory generation. The code for this paper will be released publicly at: https://github.com/fudan-generative-vision/WAM-Diff

---

## 110. Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj

**论文链接:** [http://arxiv.org/abs/2511.18248v2](http://arxiv.org/abs/2511.18248v2)

**作者:** Wei Zhen Teoh

**发布时间:** 2025-11-23

**备注:** 9 pages, 3 figures, accepted to the AI4TS Workshop at AAAI 2026

### GPT解析

### 总结

论文提出了一种名为CausalTraj的时间因果似然模型，用于解决多交互智能体联合轨迹预测的挑战，该模型在保持个体准确度的同时，显著提升了联合预测能力，能够生成连贯、真实的比赛演变。

### 背景

多交互智能体的联合轨迹预测是体育分析等涉及复杂群体动态领域的核心挑战。现有模型主要使用每个智能体的独立准确度指标(minADE, minFDE)进行评估，这些指标忽略了模型是否学习到哪些预测轨迹可以共同形成合理的多智能体未来。

### 目的

提出CausalTraj模型，一个基于时间因果关系的似然模型，旨在生成共同可能的多智能体轨迹预测，并引入联合指标(minJADE, minJFDE)来更好地评估集体建模能力。

### 方法

CausalTraj是一个基于时间因果关系的似然模型，专门设计用于生成共同可能的多智能体轨迹预测。研究使用联合指标评估模型在最佳生成的场景样本中跨智能体的联合准确度。

### 主要发现

在NBA SportVU、Basketball-U和Football-U数据集上评估，CausalTraj实现了具有竞争力的每个智能体准确度，同时在联合指标上取得了最佳记录结果，并能产生质量上连贯且真实的比赛演变。

### 结论

CausalTraj模型有效解决了多智能体联合轨迹预测的挑战，不仅个体准确度有竞争力，还在联合预测方面表现优异，能够生成连贯、真实的比赛演变，适用于体育分析等领域。

### 翻译

多交互智能体的联合轨迹预测是体育分析和其他涉及复杂群体动态领域的核心挑战。准确的预测可以实现真实的模拟和对比赛演变的战略理解。大多数现有模型仅使用每个智能体的准确度指标(minADE, minFDE)进行评估，这些指标独立评估每个智能体在其k个预测中的最佳表现。然而，这些指标忽略了模型是否学习到哪些预测的轨迹可以共同形成合理的多智能体未来。许多最先进的模型主要基于这些指标进行设计和优化。因此，它们在联合预测方面可能表现不佳，并且在团队体育中无法生成连贯、可解释的多智能体场景。我们提出CausalTraj，一个基于时间因果关系的似然模型，旨在生成共同可能的多智能体轨迹预测。为了更好地评估集体建模能力，我们强调联合指标(minJADE, minJFDE)，这些指标衡量在最佳生成的场景样本中跨智能体的联合准确度。在NBA SportVU、Basketball-U和Football-U数据集上评估，CausalTraj实现了具有竞争力的每个智能体准确度和最佳的联合指标记录结果，同时产生了质量上连贯且真实的比赛演变。


### 论文摘要

Jointly forecasting trajectories of multiple interacting agents is a core challenge in sports analytics and other domains involving complex group dynamics. Accurate prediction enables realistic simulation and strategic understanding of gameplay evolution. Most existing models are evaluated solely on per-agent accuracy metrics (minADE, minFDE), which assess each agent independently on its best-of-k prediction. However these metrics overlook whether the model learns which predicted trajectories can jointly form a plausible multi-agent future. Many state-of-the-art models are designed and optimized primarily based on these metrics. As a result, they may underperform on joint predictions and also fail to generate coherent, interpretable multi-agent scenarios in team sports. We propose CausalTraj, a temporally causal, likelihood-based model that is built to generate jointly probable multi-agent trajectory forecasts. To better assess collective modeling capability, we emphasize joint metrics (minJADE, minJFDE) that measure joint accuracy across agents within the best generated scenario sample. Evaluated on the NBA SportVU, Basketball-U, and Football-U datasets, CausalTraj achieves competitive per-agent accuracy and the best recorded results on joint metrics, while yielding qualitatively coherent and realistic gameplay evolutions.

---

## 111. Using Socio-economic Indicators, Smart Transit Systems, and Urban Simulator to Accelerate ZEV Adoption and Reduce VMT

**论文链接:** [http://arxiv.org/abs/2512.11870v1](http://arxiv.org/abs/2512.11870v1)

**作者:** Mulham Fawkherji, Bruce Race, Driss Benhaddou

**发布时间:** 2025-12-05

### GPT解析

### 总结

该论文研究了休斯顿市如何减少道路交通排放以实现2050年净零排放目标，提出利用社会经济指标和智能交通系统加速零排放车辆采用和减少车辆行驶里程的策略。

### 背景

全球道路交通占温室气体排放15%，导致约385,000人因PM2.5过早死亡；城市贡献75%全球能源相关温室气体排放；休斯顿道路交通占其气候行动计划基线排放48%；休斯顿是低密度、依赖私家车的城市，89%道路排放来自汽车和小型卡车，公共交通使用有限；社会经济差异进一步限制零排放车辆采用。

### 目的

建立道路交通排放基线方法；评估利用社会经济指标和智能交通系统加速零排放车辆采用和减少车辆行驶里程的政策；分析政策选项并确定潜在行动。

### 方法

开发基于Unity 3D的模拟环境，支持城市移动动态建模和政策场景可视化；研究智能停车、公共交通激励、安全数据系统和零排放车辆车队管理等策略；这些策略旨在改善交通模式划分和系统可靠性。

### 主要发现

休斯顿气候行动计划设定以2014年为基准减少70%排放目标，通过30%可再生能源抵消；休斯顿低密度和私家车依赖性使目标具挑战性；策略重点是扩大零排放车辆获取渠道，通过改善公共交通和城市设计将车辆行驶里程减少20%。

### 结论

依赖私家车的城市若要实现2050年排放目标，可从论文讨论的指标、度量和技术中受益；论文提供了评估和实施政策以加速零排放车辆采用和减少车辆行驶里程的方法。

### 翻译

全球范围内，道路交通占温室气体排放的15%，并估计有385,000人因PM2.5而过早死亡。城市在实现IPCC目标中发挥着关键作用，产生了75%的全球能源相关温室气体排放。在德克萨斯州的休斯顿，道路交通占气候行动计划基线排放的48%。为了在2050年实现净零排放，气候行动计划设定了以2014年为基准减少70%排放的目标，并通过30%的可再生能源抵消。这一目标具有挑战性，因为休斯顿是低密度且依赖私家车的城市，89%的道路排放来自汽车和小型卡车，公共交通使用有限。社会经济差异进一步限制了零排放车辆的采用。策略重点是扩大ZEV的获取渠道，并通过改善公共交通和城市设计将车辆行驶里程减少20%。本文介绍了建立道路交通排放基线和评估利用社会经济指标和智能交通系统加速ZEV采用和减少VMT的政策的方法。智能停车、公共交通激励、安全数据系统和ZEV车队管理等支持改善模式划分和系统可靠性。分析了政策选项并确定了潜在行动。为了支持评估，在Unity 3D中开发了一个模拟环境，能够实现城市移动的动态建模和政策场景的可视化。旨在实现2050年排放目标的依赖私家车的城市可以从讨论的指标、度量和技术中受益。


### 论文摘要

Globally, on-road transportation accounts for 15% of greenhouse gas (GHG) emissions and an estimated 385,000 premature deaths from PM2.5. Cities play a critical role in meeting IPCC targets, generating 75% of global energy-related GHG emissions. In Houston, Texas, on-road transportation represents 48% of baseline emissions in the Climate Action Plan (CAP). To reach net-zero by 2050, the CAP targets a 70% emissions reduction from a 2014 baseline, offset by 30% renewable energy. This goal is challenging because Houston is low-density and auto-dependent, with 89% of on-road emissions from cars and small trucks and limited public transit usage. Socio-economic disparities further constrain Zero Emissions Vehicle (ZEV) adoption. Strategies focus on expanding ZEV access and reducing Vehicle Miles Traveled (VMT) by 20% through transit improvements and city design. This paper presents methods for establishing an on-road emissions baseline and evaluating policies that leverage socio-economic indicators and Intelligent Transportation Systems (ITS) to accelerate ZEV adoption and reduce VMT. Smart parking, transit incentives, secure data systems, and ZEV fleet management support improvements in modal split and system reliability. Policy options are analyzed and potential actions identified. To support evaluation, a simulation environment was developed in Unity 3D, enabling dynamic modeling of urban mobility and visualization of policy scenarios. Auto-dependent cities aiming for 2050 emission targets can benefit from the indicators, metrics, and technologies discussed.

---

## 112. Establishing Reality-Virtuality Interconnections in Urban Digital Twins for Superior Intelligent Road Inspection and Simulation

**论文链接:** [http://arxiv.org/abs/2412.17699v2](http://arxiv.org/abs/2412.17699v2)

**作者:** Yikang Zhang, Chuang-Wei Liu, Jiahang Li, Yingbing Chen, Jie Cheng, Rui Fan

**发布时间:** 2024-12-23

**DOI:** 10.1109/LRA.2025.3640982

### GPT解析

### 总结

研究提出了一种结合多模态传感器平台与城市数字孪生系统的智能道路检测方法，用于解决传统道路检测方法的局限性。

### 背景

道路检测对维护道路服务性和确保交通安全至关重要，但传统人工评估方法劳动密集、成本高且耗时。数据驱动方法面临真实世界道路缺陷数据稀缺和空间稀疏的挑战，现有模拟器缺乏道路缺陷模型，且涉及道路表面交互的高级驾驶任务尚未得到充分探索。

### 目的

解决传统道路检测方法的局限性，提出一种多模态传感器平台与城市数字孪生系统集成的智能道路检测方法。

### 方法

使用车载传感器收集真实世界驾驶数据构建分层道路模型；生成数字道路孪生创建模拟环境；将这些场景导入模拟器促进数据采集和物理模拟。

### 主要发现

高保真道路缺陷场景显著改善了包括感知和决策在内的驾驶任务性能。

### 结论

所提出的多模态传感器平台与城市数字孪生系统集成的智能道路检测方法能够生成高质量的道路缺陷场景，对道路检测任务有显著帮助。

### 翻译

道路检测对于维护道路服务性和确保交通安全至关重要，因为道路缺陷会逐渐发展并影响功能。传统的人工评估方法劳动密集、成本高且耗时。虽然数据驱动方法正在获得关注，但真实世界道路缺陷的稀缺性和空间稀疏性在获取高质量数据集方面带来了重大挑战。然而，现有设计用于生成详细合成驾驶场景的模拟器缺乏道路缺陷模型。此外，涉及与道路表面交互的高级驾驶任务，如缺陷区域的规划和控制，仍未得到充分探索。为解决这些局限性，我们提出了一种与城市数字孪生系统集成的多模态传感器平台，用于智能道路检测。首先，使用车载传感器收集的真实世界驾驶数据构建分层道路模型，生成道路缺陷结构和表面高度的高详细度表示。接下来，生成数字道路孪生，创建模拟环境，用于算法性能的综合分析和评估。然后将这些场景导入模拟器，促进数据采集和物理模拟。实验结果表明，包括感知和决策在内的驾驶任务从我们系统生成的高保真道路缺陷场景中受益显著。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决传统道路检测方法效率低下、成本高以及数据驱动方法面临真实世界道路缺陷数据稀缺的问题。这个问题在现实中非常重要，因为道路缺陷会逐渐发展并影响道路功能，不仅会引起车辆振动，还会加速车辆部件磨损，及时检测和修复这些缺陷对确保交通安全至关重要。传统人工检测方法既危险又对交通流造成干扰，难以大规模应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统道路检测方法的局限性，注意到数据驱动方法面临的挑战，并发现现有模拟器缺乏道路缺陷模型。基于这些观察，他们设计了一个结合多模态传感器平台和城市数字孪生(UDT)系统的智能道路检测方案。该方法借鉴了数字孪生技术、多模态传感器数据处理、深度学习在道路缺陷检测中的应用，以及现有的模拟环境(如CARLA)进行实验。作者的创新点在于将这些现有技术整合到一个专门针对道路检测的系统中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是建立现实-虚拟互联(RVI)的城市数字孪生系统，专门用于智能道路检测。该系统不仅创建道路实体的数字孪生，还构建包含道路缺陷的模拟环境，支持感知和决策算法的评估。整体实现流程包括：1)使用便携式多模态传感器设备收集真实世界的道路数据；2)通过分层道路模型创建器(包括粗粒度流和细粒度流)重建道路表面和缺陷；3)使用数字道路孪生生成器将缺陷模型与表面模型集成，创建高保真场景；4)将这些场景应用于感知任务(如语义分割、立体匹配)和决策任务(如路径规划、速度控制)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)开发便携式多模态传感器实验装置；2)提出分层道路模型创建器和数字道路孪生生成器；3)创建包含语义和实例级标注的综合数据集；4)设计将道路缺陷视为负障碍物的实验框架；5)实现现实-虚拟互联的城市数字孪生系统。与之前工作不同，该方法在场景级别集成了有缺陷的道路实体，而非单独处理；重建了3D道路结构而非使用2D平面表示；允许车辆安全绕过或滑过某些缺陷而非完全避免；采用轮胎级碰撞检测而非车身边界框检查，支持更灵活的障碍物避免策略。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于城市数字孪生的现实-虚拟互联系统，通过高保真道路缺陷场景生成和物理模拟，显著提升了智能道路检测和驾驶任务中感知与决策算法的性能。'}


### 论文摘要

Road inspection is crucial for maintaining road serviceability and ensuring traffic safety, as road defects gradually develop and compromise functionality. Traditional inspection methods, which rely on manual evaluations, are labor-intensive, costly, and time-consuming. While data-driven approaches are gaining traction, the scarcity and spatial sparsity of real-world road defects present significant challenges in acquiring high-quality datasets. Existing simulators designed to generate detailed synthetic driving scenes, however, lack models for road defects. Moreover, advanced driving tasks that involve interactions with road surfaces, such as planning and control in defective areas, remain underexplored. To address these limitations, we propose a multi-modal sensor platform integrated with an urban digital twin (UDT) system for intelligent road inspection. First, hierarchical road models are constructed from real-world driving data collected using vehicle-mounted sensors, resulting in highly detailed representations of road defect structures and surface elevations. Next, digital road twins are generated to create simulation environments for comprehensive analysis and evaluation of algorithm performance. These scenarios are then imported into a simulator to facilitate both data acquisition and physical simulation. Experimental results demonstrate that driving tasks, including perception and decision-making, benefit significantly from the high-fidelity road defect scenes generated by our system.

---

