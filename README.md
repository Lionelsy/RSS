# 今日论文推荐 - 2025-12-15

共 50 篇论文

---

## 1. Particulate: Feed-Forward 3D Object Articulation

**论文链接:** [http://arxiv.org/abs/2512.11798v1](http://arxiv.org/abs/2512.11798v1)

**作者:** Ruining Li, Yuxin Yao, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, Andrea Vedaldi

**发布时间:** 2025-12-12

**备注:** Project page: https://ruiningli.com/particulate

### GPT解析

### 总结

论文提出了Particulate，一种前馈方法，可以从单个静态3D网格直接推断底层关节结构的所有属性，包括3D部件、运动结构和运动约束，比现有方法快得多，且能处理AI生成的3D资产。

### 背景

3D对象关节结构的自动推断是具有挑战性的问题，现有方法通常需要针对每个对象进行优化，速度较慢，且可能无法很好地处理AI生成的3D资产。

### 目的

开发一种能够从单个静态3D网格直接推断所有关节属性的方法，提高推断速度，使其远快于现有方法，并能处理AI生成的3D资产。

### 方法

提出Particulate方法，核心是Part Articulation Transformer网络，使用灵活可扩展的架构处理输入网格的点云，原生支持多关节预测，进行端到端训练，并在推理时将预测结果提升到输入网格上，同时引入新的3D关节估计基准测试和重新设计评估协议。

### 主要发现

Particulate可以在几秒内生成完全关节化的3D模型，比之前需要针对每个对象优化的方法快得多，能准确推断AI生成的3D资产的关节结构，且定量和定性结果显示显著优于最先进的方法。

### 结论

Particulate是一种高效、准确的方法，可以从单个静态3D网格推断关节结构，其速度和准确性使其成为实际应用的理想选择，能够处理真实和AI生成的3D资产。

### 翻译

我们提出了Particulate，一种前馈方法，给定日常物体的单个静态3D网格，直接推断底层关节结构的所有属性，包括其3D部件、运动结构和运动约束。其核心是Transformer网络，Part Articulation Transformer，它使用灵活可扩展的架构处理输入网格的点云，以原生支持多关节的方式预测上述所有属性。我们在来自公共数据集的多样化3D资产集合上端到端训练网络。在推理过程中，Particulate将网络的前馈预测提升到输入网格上，在几秒内生成完全关节化的3D模型，比之前需要针对每个对象优化的方法快得多。Particulate还可以准确推断AI生成的3D资产的关节结构，结合现成的图像到3D生成器，可以从单个（真实或合成）图像中完全提取关节化的3D对象。我们进一步引入了一个从高质量公共3D资产策划的3D关节估计新基准测试，并重新设计了评估协议，使其更符合人类偏好。定量和定性结果表明，Particulate显著优于最先进的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从静态3D网格中直接推断底层铰接结构的问题，包括3D部分、运动学结构和运动约束。这个问题在现实中很重要，因为日常环境中许多物体的功能（如柜门、抽屉）来自其可动部分的运动，机器人需要操作这些物体，游戏和模拟需要创建交互式数字孪生。现有方法速度慢且难以扩展到各种物体类别。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者认为基于规则的方法难以扩展到现实世界长尾分布的物体，而基于学习的方法从大量3D资产中捕获铰接先验更有潜力。他们观察到现有方法要么只预测静态部分分割，要么假设关键属性已知，或依赖部分检索。他们借鉴了transformer架构，设计了'部分铰接变换器'，使用点云作为输入表示，并利用PartField提供的3D语义部分特征作为输入增强。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一个前馈神经网络，从单个静态3D网格直接推断完整铰接结构。流程包括：1)从网格采样点云并添加特征；2)使用transformer架构处理点云和部分查询；3)通过多个解码器头预测部分分割、运动学树和运动参数；4)使用过参数化方法提高旋转轴预测准确性；5)在公共数据集上端到端训练；6)推理时快速预测并细化结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)前馈方法实现快速推理(约10秒)；2)完整预测铰接结构所有属性；3)灵活可扩展的transformer架构；4)创新的旋转轴过参数化方法；5)泛化到AI生成物体；6)引入新基准数据集；7)更符合人类偏好的评估协议。相比之前工作，它比基于优化的方法快得多，比生成型方法不需要先验知识，比VLM方法能处理内部部分且支持多关节，比部分检索方法不依赖数据库。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PARTICULATE提出了一种快速准确的前馈神经网络方法，能从静态3D网格中推断完整铰接结构，显著优于现有方法并支持AI生成物体的铰接结构提取。'}


### 论文摘要

We present Particulate, a feed-forward approach that, given a single static 3D mesh of an everyday object, directly infers all attributes of the underlying articulated structure, including its 3D parts, kinematic structure, and motion constraints. At its core is a transformer network, Part Articulation Transformer, which processes a point cloud of the input mesh using a flexible and scalable architecture to predict all the aforementioned attributes with native multi-joint support. We train the network end-to-end on a diverse collection of articulated 3D assets from public datasets. During inference, Particulate lifts the network's feed-forward prediction to the input mesh, yielding a fully articulated 3D model in seconds, much faster than prior approaches that require per-object optimization. Particulate can also accurately infer the articulated structure of AI-generated 3D assets, enabling full-fledged extraction of articulated 3D objects from a single (real or synthetic) image when combined with an off-the-shelf image-to-3D generator. We further introduce a new challenging benchmark for 3D articulation estimation curated from high-quality public 3D assets, and redesign the evaluation protocol to be more consistent with human preferences. Quantitative and qualitative results show that Particulate significantly outperforms state-of-the-art approaches.

---

## 2. A General Algorithm for Detecting Higher-Order Interactions via Random Sequential Additions

**论文链接:** [http://arxiv.org/abs/2512.11793v1](http://arxiv.org/abs/2512.11793v1)

**作者:** Ahmad Shamail, Claire McWhite

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出了一种简单的几何方法来发现系统组件间的相互作用和冗余性。通过随机顺序添加元素并多次试验绘制贡献图，会出现特征性的L形模式，这些模式直接反映了相互作用结构。该方法量化了每个元素的贡献如何依赖于之前添加的元素，并使用L分数（范围从-1到+1）来区分协同、独立和冗余关系。该方法适用于任何可以在非重复元素序列上增量评估性能的领域。

### 背景

许多系统组件之间表现出复杂的相互作用，有些特征或行为会放大彼此的效果，有些提供冗余信息，有些则独立贡献。

### 目的

提出一种简单的几何方法来发现系统组件间的相互作用和冗余性。

### 方法

当元素以随机顺序添加并在多次试验中绘制它们的贡献时，会出现特征性的L形模式，这些模式直接反映了相互作用结构。该方法量化了每个元素的贡献如何依赖于之前添加的元素，揭示了在统一尺度上区分相互作用、独立性和冗余性的模式。

### 主要发现

1. 冗余对形成L形模式，只有先添加的元素有贡献；2. 协同对形成L形模式，只有元素一起才有贡献；3. 独立元素显示顺序不变的分布；4. L形臂的相对缩放揭示了特征优势；5. 虽然仅从成对测量计算，但三个或更多元素之间的高阶相互作用通过一致的交叉对关系自然出现。

### 结论

该方法与度量无关，适用于任何可以在非重复元素序列上增量评估性能的领域，提供了一种统一的几何方法来揭示相互作用结构。

### 翻译

许多系统组件之间表现出复杂的相互作用：一些特征或行为会放大彼此的效果，另一些提供冗余信息，还有一些独立贡献。我们提出了一种简单的几何方法来发现相互作用和冗余性：当元素以随机顺序添加并在多次试验中绘制它们的贡献时，会出现特征性的L形模式，这些模式直接反映了相互作用结构。该方法量化了每个元素的贡献如何依赖于之前添加的元素，揭示了在统一尺度上区分相互作用、独立性和冗余性的模式。当成对贡献可视化为二维点云时，冗余对形成L形模式，只有先添加的元素有贡献，而协同对形成L形模式，只有元素一起才有贡献。独立元素显示顺序不变的分布。我们用L分数形式化了这一点，这是一个连续度量，范围从-1（完美协同）到0（独立）到+1（完美冗余）。L形臂的相对缩放揭示了特征优势，即哪个元素持续提供更多信息。虽然仅从成对测量计算，但三个或更多元素之间的高阶相互作用通过一致的交叉对关系自然出现。该方法与度量无关，适用于任何可以在非重复元素序列上增量评估性能的领域，提供了一种统一的几何方法来揭示相互作用结构。


### 论文摘要

Many systems exhibit complex interactions between their components: some features or actions amplify each other's effects, others provide redundant information, and some contribute independently. We present a simple geometric method for discovering interactions and redundancies: when elements are added in random sequential orders and their contributions plotted over many trials, characteristic L-shaped patterns emerge that directly reflect interaction structure. The approach quantifies how the contribution of each element depends on those added before it, revealing patterns that distinguish interaction, independence, and redundancy on a unified scale. When pairwise contributions are visualized as two--dimensional point clouds, redundant pairs form L--shaped patterns where only the first-added element contributes, while synergistic pairs form L--shaped patterns where only elements contribute together. Independent elements show order--invariant distributions. We formalize this with the L--score, a continuous measure ranging from $-1$ (perfect synergy, e.g. $Y=X_1X_2$) to $0$ (independence) to $+1$ (perfect redundancy, $X_1 \approx X_2$). The relative scaling of the L--shaped arms reveals feature dominance in which element consistently provides more information. Although computed only from pairwise measurements, higher--order interactions among three or more elements emerge naturally through consistent cross--pair relationships (e.g. AB, AC, BC). The method is metric--agnostic and broadly applicable to any domain where performance can be evaluated incrementally over non-repeating element sequences, providing a unified geometric approach to uncovering interaction structure.

---

## 3. DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation

**论文链接:** [http://arxiv.org/abs/2512.11465v1](http://arxiv.org/abs/2512.11465v1)

**作者:** Mohamed Abdelsamad, Michael Ulrich, Bin Yang, Miao Zhang, Yakov Miron, Abhinav Valada

**发布时间:** 2025-12-12

**备注:** AAAI-26

### GPT解析

### 总结

本文提出了DOS(Distilling Observable Softmaps)框架，一种用于3D点云自监督学习的新方法，仅在可观察点处进行语义相关性软映射蒸馏，解决了不规则几何、捷径重构和不平衡语义分布等挑战。

### 背景

自监督学习在3D点云表示学习方面显示出无需人工标注的巨大潜力，但仍然面临不规则几何、容易产生捷径的重构和不平衡语义分布等关键挑战。

### 目的

开发一种新的自监督学习框架，解决3D点云表示学习中的关键挑战，提高语义分割和3D目标检测的性能。

### 方法

提出DOS框架，仅在未掩码的可观察点处自蒸馏语义相关性软映射，防止掩码区域信息泄露；引入Zipfian原型并使用改进的Sinkhorn-Knopp算法(Zipf-Sinkhorn)强制使用原型的幂律先验，调节目标软映射的锐度。

### 主要发现

DOS在nuScenes、Waymo、SemanticKITTI、ScanNet和ScanNet200等多个基准测试的语义分割和3D目标检测任务中超越了当前最先进方法，无需额外数据或标注。

### 结论

可观察点软映射蒸馏为学习鲁棒的3D表示提供了一种可扩展且有效的范式。

### 翻译

最近的自监督学习(SSL)进展显示出在无需人工标注的情况下学习3D点云表示的巨大潜力。然而，由于不规则几何、容易产生捷径的重构和不平衡的语义分布，3D点云的SSL仍然面临关键挑战。在这项工作中，我们提出了DOS(Distilling Observable Softmaps)，一种新颖的SSL框架，仅在可观察(未掩码)点处自蒸馏语义相关性软映射。这种策略防止了来自掩码区域的信息泄露，提供了比离散令牌到原型分配更丰富的监督。为了解决无监督设置中不平衡语义的挑战，我们引入了Zipfian原型，并使用改进的Sinkhorn-Knopp算法Zipf-Sinkhorn将其整合，该算法强制使用原型的幂律先验，并在训练期间调节目标软映射的锐度。DOS在nuScenes、Waymo、SemanticKITTI、ScanNet和ScanNet200等多个基准测试的语义分割和3D目标检测任务中超越了当前最先进的方法，而不依赖额外数据或标注。我们的结果表明，可观察点软映射蒸馏为学习鲁棒的3D表示提供了一种可扩展且有效的范式。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D点云自监督学习中的三个关键挑战：不规则几何结构、容易陷入捷径学习以及语义分布不平衡。这些问题在现实中非常重要，因为3D点云是自动驾驶、机器人和增强现实等领域的基础数据源，而人工标注这些数据成本高昂且耗时。解决这些问题可以提高3D表示学习的效率和应用范围，减少对大量标注数据的依赖。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有3D点云自监督学习方法的局限性进行思考，发现基于重建的方法依赖低级几何特征，基于2D视觉模型的方法难以充分利用3D结构，掩码自蒸馏方法存在位置信息泄露，而聚类方法假设均匀原型使用与真实语义长尾分布不符。DOS借鉴了掩码自蒸馏框架、Sinkhorn-Knopp算法、InfoNCE对比学习和Zipf定律等现有工作，但进行了针对性改进和组合，以解决3D点云自监督学习的特定挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'DOS的核心思想包括：1) 只对可见点进行自蒸馏避免信息泄露；2) 使用语义软图促进点间语义竞争；3) 应用Zipf-Sinkhorn算法处理长尾语义分布。整体流程是：创建点云的两个增强视图，对学生视图进行掩码处理，学生网络处理可见点，教师网络处理完整点云，计算与原型的相似度，构建软图，使用Zipf-Sinkhorn正则化教师软图，最后通过KL散度损失让学生学习匹配正则化后的教师软图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 可观察点自蒸馏，只监督可见点避免位置信息泄露；2) 语义软图，将原型激活编码为空间分布，鼓励点间语义竞争；3) Zipf-Sinkhorn正则化，使用Zipf分布代替均匀先验更好地反映真实语义分布；4) 强大的跨域泛化能力。相比之前的工作，DOS解决了信息泄露问题，提供了更丰富的监督信号，更好地处理了语义不平衡，并在多个室内外数据集上实现了最先进性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DOS通过可观察点自蒸馏、语义软图和Zipf-Sinkhorn正则化，解决了3D点云自监督学习中的信息泄露、语义竞争不足和长尾分布问题，显著提高了表示学习质量并在多个下游任务中实现了最先进性能。'}


### 论文摘要

Recent advances in self-supervised learning (SSL) have shown tremendous potential for learning 3D point cloud representations without human annotations. However, SSL for 3D point clouds still faces critical challenges due to irregular geometry, shortcut-prone reconstruction, and unbalanced semantics distribution. In this work, we propose DOS (Distilling Observable Softmaps), a novel SSL framework that self-distills semantic relevance softmaps only at observable (unmasked) points. This strategy prevents information leakage from masked regions and provides richer supervision than discrete token-to-prototype assignments. To address the challenge of unbalanced semantics in an unsupervised setting, we introduce Zipfian prototypes and incorporate them using a modified Sinkhorn-Knopp algorithm, Zipf-Sinkhorn, which enforces a power-law prior over prototype usage and modulates the sharpness of the target softmap during training. DOS outperforms current state-of-the-art methods on semantic segmentation and 3D object detection across multiple benchmarks, including nuScenes, Waymo, SemanticKITTI, ScanNet, and ScanNet200, without relying on extra data or annotations. Our results demonstrate that observable-point softmaps distillation offers a scalable and effective paradigm for learning robust 3D representations.

---

## 4. A Multi-Mode Structured Light 3D Imaging System with Multi-Source Information Fusion for Underwater Pipeline Detection

**论文链接:** [http://arxiv.org/abs/2512.11354v1](http://arxiv.org/abs/2512.11354v1)

**作者:** Qinghan Hu, Haijiang Zhu, Na Sun, Lei Chen, Zhengqiang Fan, Zhiqing Li

**发布时间:** 2025-12-12

### GPT解析

### 总结

这篇论文开发了一种多模式水下结构光3D成像系统（UW-SLD系统），通过多源信息融合和先进算法，实现了水下管道缺陷的高精度、鲁棒检测与重建。

### 背景

水下管道容易受到腐蚀，这会缩短使用寿命并带来安全风险。与人工检测相比，智能实时成像系统已成为更可靠和实用的解决方案。在各种水下成像技术中，结构光3D成像可以恢复足够的空间细节用于精确缺陷表征。

### 目的

开发一种基于多源信息融合的水下管道检测多模式结构光3D成像系统（UW-SLD系统）。

### 方法

采用快速畸变校正方法进行高效水下图像校正；提出基于因子图的参数优化方法解决水下传感器外标定挑战；引入多模式3D成像策略适应管道几何变化；设计多源信息融合策略和自适应扩展卡尔曼滤波确保稳定姿态估计；提出基于边缘检测的ICP算法实现缺陷结构的鲁棒高保真重建。

### 主要发现

在不同操作模式、速度和深度下进行的实验结果表明，该系统具有卓越的精度、适应性和鲁棒性。

### 结论

开发的UW-SLD系统为自主水下管道检测提供了坚实的基础。

### 翻译

水下管道极易受到腐蚀，这不仅缩短了它们的使用寿命，还带来了重大的安全风险。与人工检测相比，水下管道检测的智能实时成像系统已成为一种更可靠和实用的解决方案。在各种水下成像技术中，结构光3D成像可以恢复足够的空间细节用于精确的缺陷表征。因此，本文基于多源信息融合开发了一种用于管道检测的多模式水下结构光3D成像系统（UW-SLD系统）。首先，采用快速畸变校正方法进行高效的水下图像校正。为了克服水下传感器之间的外标定挑战，提出了一种基于因子图的参数优化方法来估计结构光和声学传感器之间的变换矩阵。此外，引入了多模式3D成像策略以适应水下管道的几何变化。鉴于水下环境中存在大量干扰，设计了多源信息融合策略和自适应扩展卡尔曼滤波器以确保稳定的姿态估计和高精度测量。特别是，提出了一种基于边缘检测的ICP算法。该算法将管道边缘检测网络与增强的点云配准相结合，即使在运动条件变化的情况下也能实现缺陷结构的鲁棒且高保真的重建。在不同的操作模式、速度和深度下进行了大量实验。结果表明，开发的系统实现了卓越的精度、适应性和鲁棒性，为自主水下管道检测奠定了坚实的基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决水下管道腐蚀检测的问题。水下管道作为海洋工程中的关键设施，在输送油气、海水淡化等方面发挥重要作用，但长期暴露在复杂海洋环境中容易受到腐蚀，不仅缩短使用寿命，还带来重大安全风险。传统手动检测方法消耗大量人力物力且存在安全风险，而现有智能检测模型难以准确捕捉腐蚀管道的高度非线性退化行为。因此，开发可靠的实时成像系统对保障管道安全运行和长期可靠性至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有水下管道检测方法的局限性：声学成像技术空间分辨率低，被动视觉成像受水下光散射影响大，而结构光3D成像虽适合但缺乏针对管道检测的系统级框架。基于此，作者设计了UW-SLD系统，整合结构光主动视觉、声学和惯性传感器。借鉴了Kalibr校准方法、因子图优化、分层多频融合和自适应卡尔曼滤波等技术，但进行了创新性改进，使其适应水下管道检测的特殊需求。系统采用模块化设计，支持多种成像模式，并通过智能边缘检测提高检测精度。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是开发一个多模式水下结构光3D成像系统，通过多源信息融合技术，结合结构光、声学和惯性传感器，实现高精度、自适应和鲁棒的水下管道检测。系统采用即插即用的背包形式设计，基于ROS2架构构建软件框架。整体流程包括：1)系统启动和校准阶段；2)水下成像阶段，包括激光旋转控制、多传感器信息同步、AEKF姿态估计和点云生成；3)多模式成像策略(平移、旋转、平移-旋转)；4)智能管道边缘检测和ED-ICP点云配准。最终生成具有时空一致性的3D点云用于管道检测和重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)完整的UW-SLD系统开发，突破结构光成像多项关键技术；2)水下快速畸变校正方法，计算效率提高46倍；3)基于因子图的外部参数校准方法，位置误差降低24.4%；4)分层多频融合策略和AEKF方法，显著提高姿态估计精度；5)多模式成像策略和智能边缘检测网络，实现变速运动下的鲁棒点云配准。相比之前工作，本文提供了完整的系统级解决方案而非单模块优化，支持多种成像模式适应不同管道几何形状，通过多源融合和自适应滤波增强抗干扰能力，并引入深度学习提高智能化水平。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文开发了一种基于多源信息融合的多模式水下结构光3D成像系统，通过创新的校准、融合和成像策略，实现了水下管道的高精度、自适应和鲁棒检测与3D重建。'}


### 论文摘要

Underwater pipelines are highly susceptible to corrosion, which not only shorten their service life but also pose significant safety risks. Compared with manual inspection, the intelligent real-time imaging system for underwater pipeline detection has become a more reliable and practical solution. Among various underwater imaging techniques, structured light 3D imaging can restore the sufficient spatial detail for precise defect characterization. Therefore, this paper develops a multi-mode underwater structured light 3D imaging system for pipeline detection (UW-SLD system) based on multi-source information fusion. First, a rapid distortion correction (FDC) method is employed for efficient underwater image rectification. To overcome the challenges of extrinsic calibration among underwater sensors, a factor graph-based parameter optimization method is proposed to estimate the transformation matrix between the structured light and acoustic sensors. Furthermore, a multi-mode 3D imaging strategy is introduced to adapt to the geometric variability of underwater pipelines. Given the presence of numerous disturbances in underwater environments, a multi-source information fusion strategy and an adaptive extended Kalman filter (AEKF) are designed to ensure stable pose estimation and high-accuracy measurements. In particular, an edge detection-based ICP (ED-ICP) algorithm is proposed. This algorithm integrates pipeline edge detection network with enhanced point cloud registration to achieve robust and high-fidelity reconstruction of defect structures even under variable motion conditions. Extensive experiments are conducted under different operation modes, velocities, and depths. The results demonstrate that the developed system achieves superior accuracy, adaptability and robustness, providing a solid foundation for autonomous underwater pipeline detection.

---

## 5. Evaluating Foundation Models' 3D Understanding Through Multi-View Correspondence Analysis

**论文链接:** [http://arxiv.org/abs/2512.11574v1](http://arxiv.org/abs/2512.11574v1)

**作者:** Valentina Lilova, Toyesh Chakravorty, Julian I. Bibo, Emma Boccaletti, Brandon Li, Lívia Baxová, Cees G. M. Snoek, Mohammadreza Salehi

**发布时间:** 2025-12-12

**备注:** NeurIPS 2025 UniReps workshop

### GPT解析

### 总结

本文介绍了一种新的基准测试方法，用于评估基础模型的3D空间理解能力，无需微调即可直接探测密集视觉特征的质量。

### 背景

现有的3D空间理解评估方法通常依赖于下游微调，使用线性头或特定任务解码器，难以隔离预训练编码器的内在3D推理能力。

### 目的

引入一个无需微调的上下文3D场景理解基准测试，直接评估预训练模型在3D场景理解方面的能力。

### 方法

基于Hummingbird框架扩展至3D多视角ImageNet数据集，通过给定特定角度的图像（键），评估模型分割新视图（查询）的性能，并根据视角对比度分为四个难度等级。

### 主要发现

对8个最先进的基础模型测试表明，基于DINO的编码器在大视角变化下保持竞争力，而像VGGT这样的3D感知模型需要专门的多视角调整。

### 结论

这种新基准测试方法为评估基础模型的3D空间理解能力提供了一种更直接、无需微调的方法。

### 翻译

对基础模型的3D空间理解进行基准测试对于机器人技术和自动驾驶等实际应用至关重要。现有的评估通常依赖于使用线性头或特定任务解码器的下游微调，这使得难以隔离预训练编码器的内在3D推理能力。在这项工作中，我们引入了一种新颖的上下文3D场景理解基准测试，它不需要微调，直接探测密集视觉特征的质量。基于评估上下文2D场景理解的Hummingbird框架，我们将设置扩展到3D多视角ImageNet（MVImgNet）数据集。给定来自特定角度（键）的一组图像，我们基准测试分割新视图（查询）的性能，并根据键-查询视图对比度在容易、中等、困难和极端四个类别中报告分数。我们对8个最先进的基础模型进行了基准测试，结果表明基于DINO的编码器在大的视角变化下保持竞争力，而像VGGT这样的3D感知模型需要专门的多视角调整。我们的代码已在https://github.com/ToyeshC/open-hummingbird-3d-eval公开可用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何准确评估基础模型的三维空间理解能力，特别是在不同视角下的鲁棒性问题。这个问题在现实世界中很重要，因为机器人技术和自动驾驶等应用需要模型在各种视角下都能稳定工作；在研究中也很重要，因为现有评估方法通常依赖下游微调，无法准确衡量模型本身的内在三维推理能力，限制了我们对模型真实三维理解水平的认识。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有评估方法无法准确衡量模型的三维理解能力，特别是跨视角的泛化能力。他们借鉴了Hummingbird框架（一个评估2D场景理解的方法）和MVImgNet数据集（提供多视角标注），将其扩展到3D场景理解。作者设计了一个非参数的检索框架，通过构建特定视角的记忆库，然后在未见视角上评估分割性能，从而在不进行微调的情况下直接测试模型的三维理解能力。他们还设计了4个难度级别来系统化测试不同视角变化下的模型表现。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过评估模型在未见视角下的分割能力来衡量其三维理解能力，利用多视角图像和动态记忆机制，在不进行微调的情况下测试预训练模型的质量。整体流程包括：1)使用MVImgNet数据集选择15个类别，将视角离散化为7个区间；2)构建特定视角的记忆库存储图像和分割标签；3)在查询时使用交叉注意力检索最接近特征并生成分割预测；4)在4个难度级别上评估8个基础模型；5)使用平均交并比(mIoU)作为评估指标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出不依赖下游微调的3D场景理解评估基准；2)扩展Hummingbird框架到3D场景并引入视角分箱协议；3)设计从简单到极端的4个难度级别全面测试模型；4)评估8个最先进模型并比较通用编码器和专门3D模型。相比之前工作，本文直接评估预训练编码器的三维能力而非通过下游任务，测试更广泛的视角变化(0°-90°)，并揭示了自监督模型在视角变化下的优势，这些都是之前研究较少涉及的。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '该论文提出了一个新颖的评估框架，通过多视角对应分析系统化地评估了基础模型的三维理解能力，揭示了自监督学习模型在视角变化下的鲁棒性优势，并为未来改进三维感知模型提供了重要见解。'}


### 论文摘要

Benchmarking 3D spatial understanding of foundation models is essential for real-world applications such as robotics and autonomous driving. Existing evaluations often rely on downstream finetuning with linear heads or task-specific decoders, making it difficult to isolate the intrinsic 3D reasoning ability of pretrained encoders. In this work, we introduce a novel benchmark for in-context 3D scene understanding that requires no finetuning and directly probes the quality of dense visual features. Building on the Hummingbird framework, which evaluates in-context 2D scene understanding, we extend the setup to the 3D Multi-View ImageNet (MVImgNet) dataset. Given a set of images from objects in specific angles (keys), we benchmark the performance of segmenting novel views (queries) and report the scores in 4 categories of easy, medium, hard, and extreme based on the key-query view contrast. We benchmark 8 state-of-the-art foundation models and show DINO-based encoders remain competitive across large viewpoint shifts, while 3D-aware models like VGGT require dedicated multi-view adjustments. Our code is publicly available at https://github.com/ToyeshC/open-hummingbird-3d-eval .

---

## 6. Reconstruction as a Bridge for Event-Based Visual Question Answering

**论文链接:** [http://arxiv.org/abs/2512.11510v1](http://arxiv.org/abs/2512.11510v1)

**作者:** Hanyue Lou, Jiayi Zhou, Yang Zhang, Boyu Li, Yi Wang, Guangnan Ye, Boxin Shi

**发布时间:** 2025-12-12

### GPT解析

### 总结

研究提出了一种将事件相机与多模态大语言模型结合的方法，通过重构作为桥梁，解决了保留事件数据优势与确保基于帧模型兼容性之间的权衡问题。

### 背景

将事件相机与多模态大语言模型结合可以在具有挑战性的视觉条件下实现通用场景理解，但需要在保留事件数据独特优势和确保与基于帧的模型兼容性之间取得平衡。

### 目的

解决如何在保留事件数据优势的同时确保与基于帧模型兼容性的挑战。

### 方法

提出基于帧的重构和标记化(FRT)方法，设计利用事件稀疏性的高效自适应重构和标记化(ART)方法，并引入EvQA作为首个基于事件的MLLMs的客观、真实世界基准，包含来自22个公共数据集的1000个事件问答对。

### 主要发现

实验表明，所提出的方法在EvQA基准上达到了最先进的性能。

### 结论

突显了多模态大语言模型在基于事件的视觉中的巨大潜力。

### 翻译

将事件相机与多模态大语言模型(MLLMs)结合有望在具有挑战性的视觉条件下实现通用场景理解，但需要在保留事件数据的独特优势和确保与基于帧模型的兼容性之间进行权衡。我们通过使用重构作为桥梁来解决这一挑战，提出了简单的基于帧的重构和标记化(FRT)方法，并设计了利用事件稀疏性的高效自适应重构和标记化(ART)方法。为了进行稳健评估，我们引入了EvQA，这是第一个针对基于事件的MLLMs的客观、真实世界基准，包含来自22个公共数据集的1000个事件问答对。我们的实验证明，我们的方法在EvQA上取得了最先进的性能，突显了MLLMs在基于事件的视觉中的巨大潜力。


### 论文摘要

Integrating event cameras with Multimodal Large Language Models (MLLMs) promises general scene understanding in challenging visual conditions, yet requires navigating a trade-off between preserving the unique advantages of event data and ensuring compatibility with frame-based models. We address this challenge by using reconstruction as a bridge, proposing a straightforward Frame-based Reconstruction and Tokenization (FRT) method and designing an efficient Adaptive Reconstruction and Tokenization (ART) method that leverages event sparsity. For robust evaluation, we introduce EvQA, the first objective, real-world benchmark for event-based MLLMs, comprising 1,000 event-Q&A pairs from 22 public datasets. Our experiments demonstrate that our methods achieve state-of-the-art performance on EvQA, highlighting the significant potential of MLLMs in event-based vision.

---

## 7. VLM2GeoVec: Toward Universal Multimodal Embeddings for Remote Sensing

**论文链接:** [http://arxiv.org/abs/2512.11490v1](http://arxiv.org/abs/2512.11490v1)

**作者:** Emanuel Sánchez Aimar, Gulnaz Zhambulova, Fahad Shahbaz Khan, Yonghao Xu, Michael Felsberg

**发布时间:** 2025-12-12

**备注:** 21 pages, 7 figures, under review

### GPT解析

### 总结

论文提出了VLM2GeoVec，一种遵循指令的单一编码器视觉语言模型，通过对比训练将交错输入嵌入统一向量空间，实现了遥感中的多模态分析。同时引入了RSMEB基准测试，涵盖多种遥感嵌入应用场景。

### 背景

卫星图像与自然图像有根本区别：其空中视角、超高分辨率、多样化的尺度变化以及大量小物体需求区域级空间推理和整体场景理解。当前遥感方法在双编码器检索模型和生成助手之间分裂，前者擅长大规模跨模态搜索但不能交错模态，后者支持区域级解释但缺乏可扩展的检索能力。

### 目的

提出一种能够统一可扩展检索与区域级空间推理的单一编码器视觉语言模型，实现遥感中的多模态分析。

### 方法

VLM2GeoVec是一种遵循指令的单一编码器视觉语言模型，通过对比训练将交错输入（图像、文本、边界框和地理坐标）嵌入统一向量空间。该模型将所有输入交错为一个联合嵌入，使用对比损失进行训练，消除了多阶段流水线和任务特定模块。为评估其多功能性，引入了RSMEB基准测试，涵盖场景分类、跨模态搜索、组合检索、视觉问答、视觉定位和区域级推理以及语义地理空间检索等应用。

### 主要发现

在RSMEB上，VLM2GeoVec实现了显著性能提升：区域-标题检索的P@1达到26.6%（比双编码器基线提高25个百分点），指代表达式检索的P@1达到32.5%（提高19个百分点），语义地理定位检索的P@1达到17.8%（是之前最佳结果的3倍以上）。在场景分类和跨模态检索等传统任务上匹配或超过专用基线。

### 结论

VLM2GeoVec统一了可扩展检索与区域级空间推理，实现了遥感中的连贯多模态分析。作者将在接受后公开代码、检查点和数据。

### 翻译

卫星图像与自然图像有根本区别：其空中视角、超高分辨率、多样化的尺度变化以及大量小物体需求区域级空间推理和整体场景理解。当前的遥感方法在双编码器检索模型和生成助手之间仍然分裂，前者擅长大规模跨模态搜索但不能交错模态，后者支持区域级解释但缺乏可扩展的检索能力。我们提出了VLM2GeoVec，一种遵循指令的单一编码器视觉语言模型，通过对比训练将交错输入（图像、文本、边界框和地理坐标）嵌入统一向量空间。我们的单一编码器将所有输入交错为一个联合嵌入，使用对比损失进行训练，消除了多阶段流水线和任务特定模块。为了评估其多功能性，我们引入了RSMEB，一个涵盖关键遥感嵌入应用的新基准测试：场景分类；跨模态搜索；组合检索；视觉问答；视觉定位和区域级推理；以及语义地理空间检索。在RSMEB上，它实现了区域-标题检索26.6%的P@1（比双编码器基线提高25个百分点），指代表达式检索32.5%的P@1（提高19个百分点），以及语义地理定位检索17.8%的P@1（是之前最佳结果的3倍以上），同时在场景分类和跨模态检索等传统任务上匹配或超过专用基线。VLM2GeoVec统一了可扩展检索与区域级空间推理，实现了遥感中的连贯多模态分析。我们将在接受后公开代码、检查点和数据。


### 论文摘要

Satellite imagery differs fundamentally from natural images: its aerial viewpoint, very high resolution, diverse scale variations, and abundance of small objects demand both region-level spatial reasoning and holistic scene understanding. Current remote-sensing approaches remain fragmented between dual-encoder retrieval models, which excel at large-scale cross-modal search but cannot interleave modalities, and generative assistants, which support region-level interpretation but lack scalable retrieval capabilities. We propose $\textbf{VLM2GeoVec}$, an instruction-following, single-encoder vision-language model trained contrastively to embed interleaved inputs (images, text, bounding boxes, and geographic coordinates) in a unified vector space. Our single encoder interleaves all inputs into one joint embedding trained with a contrastive loss, eliminating multi-stage pipelines and task-specific modules. To evaluate its versatility, we introduce $\textbf{RSMEB}$, a novel benchmark covering key remote-sensing embedding applications: scene classification; cross-modal search; compositional retrieval; visual-question answering; visual grounding and region-level reasoning; and semantic geospatial retrieval. On RSMEB, it achieves $\textbf{26.6%}$ P@1 on region-caption retrieval (+25 pp vs. dual-encoder baselines), $\textbf{32.5%}$ P@1 on referring-expression retrieval (+19 pp), and $\textbf{17.8%}$ P@1 on semantic geo-localization retrieval (over $3\times$ prior best), while matching or exceeding specialized baselines on conventional tasks such as scene classification and cross-modal retrieval. VLM2GeoVec unifies scalable retrieval with region-level spatial reasoning, enabling cohesive multimodal analysis in remote sensing. We will publicly release the code, checkpoints, and data upon acceptance.

---

## 8. Fully Inductive Node Representation Learning via Graph View Transformation

**论文链接:** [http://arxiv.org/abs/2512.11561v1](http://arxiv.org/abs/2512.11561v1)

**作者:** Dooho Lee, Myeong Kong, Minho Jeong, Jaemin Yoo

**发布时间:** 2025-12-12

### GPT解析

### 总结

这项研究通过引入视图空间和图视图变换，解决了图结构数据中跨数据集完全归纳推理的困难，提出的循环GVT模型在多个基准测试上表现优异。

### 背景

在图结构数据中，将预训练模型泛化到未见过的数据集而不重新训练是实现基础模型的关键步骤，但由于特征空间的维度和语义差异很大，实现跨数据集的完全归纳推理很困难。

### 目的

提出一种方法，能够在不重新训练的情况下将预训练模型泛化到未见过的图数据集。

### 方法

引入了'视图空间'，这是一种新的表示轴，可以以统一方式自然编码任意图；提出了图视图变换(GVT)，这是在视图空间中的节点和特征置换等变映射；GVT作为循环GVT的构建块，这是一种完全归纳的节点表示学习模型。

### 主要发现

在OGBN-Arxiv上预训练，并在27个节点分类基准上评估，循环GVT比之前完全归纳的图模型GraphAny表现好8.93%，超越了12个单独调优的GNN至少3.30%。

### 结论

视图空间为完全归纳的节点表示学习提供了原则性和有效的基础。

### 翻译

将预训练模型泛化到未见过的数据集而不重新训练是迈向基础模型的关键一步。然而，在图结构数据中，由于特征空间的维度和语义差异很大，实现这种跨数据集的完全归纳推理很困难。特征空间中的任何变换都可能违反对未见数据集的归纳适用性，严格限制了图模型的设计空间。在这项工作中，我们引入了视图空间，这是一种新的表示轴，可以以统一方式自然编码任意图。随后，我们提出了图视图变换(GVT)，这是视图空间中的一种节点和特征置换等变映射。GVT作为循环GVT的构建块，这是一种用于节点表示学习的完全归纳模型。在OGBN-Arxiv上预训练并在27个节点分类基准上评估，循环GVT比之前完全归纳的图模型GraphAny高出8.93%，并且至少超越12个单独调优的GNN 3.30%。这些结果确立了视图空间作为完全归纳节点表示学习的原则性和有效基础。


### 论文摘要

Generalizing a pretrained model to unseen datasets without retraining is an essential step toward a foundation model. However, achieving such cross-dataset, fully inductive inference is difficult in graph-structured data where feature spaces vary widely in both dimensionality and semantics. Any transformation in the feature space can easily violate the inductive applicability to unseen datasets, strictly limiting the design space of a graph model. In this work, we introduce the view space, a novel representational axis in which arbitrary graphs can be naturally encoded in a unified manner. We then propose Graph View Transformation (GVT), a node- and feature-permutation-equivariant mapping in the view space. GVT serves as the building block for Recurrent GVT, a fully inductive model for node representation learning. Pretrained on OGBN-Arxiv and evaluated on 27 node-classification benchmarks, Recurrent GVT outperforms GraphAny, the prior fully inductive graph model, by +8.93% and surpasses 12 individually tuned GNNs by at least +3.30%. These results establish the view space as a principled and effective ground for fully inductive node representation learning.

---

## 9. Hyperbolic Gaussian Blurring Mean Shift: A Statistical Mode-Seeking Framework for Clustering in Curved Spaces

**论文链接:** [http://arxiv.org/abs/2512.11448v1](http://arxiv.org/abs/2512.11448v1)

**作者:** Arghya Pratihar, Arnab Seal, Swagatam Das, Inesh Chattopadhyay

**发布时间:** 2025-12-12

### GPT解析

### 总结

本研究提出了一种名为HypeGBMS的新型聚类算法，将高斯模糊均值偏移(GBMS)扩展到双曲空间，有效处理具有层次或树状结构的数据集。

### 背景

聚类是一种基本的无监督学习任务，用于发现数据中的模式。传统的GBMS方法在欧几里得空间中能有效识别任意形状的聚类，但在处理具有层次或树状结构的数据集时表现不佳。

### 目的

开发一种能够有效处理层次或树状结构数据集的聚类方法，保留GBMS的密度特性，同时能够捕捉潜在层次结构。

### 方法

提出HypeGBMS，一种GBMS在双曲空间的新型扩展。该方法用双曲距离替代欧几里得计算，并使用Möbius加权均值确保所有更新与空间几何保持一致。

### 主要发现

提供了关于收敛性和计算复杂性的理论见解，并通过实证结果证明了在层次数据集中改进的聚类质量。在11个真实世界数据集上的实验表明，HypeGBMS在非欧几里得设置中显著优于传统均值偏移聚类方法。

### 结论

这项工作将经典的均值偏移聚类与双曲表示学习联系起来，为弯曲空间中的基于密度的聚类提供了有原则的方法，展现出鲁棒性和有效性。

### 翻译

聚类是一种基本的无监督学习任务，用于发现数据中的模式。虽然高斯模糊均值偏移(GBMS)已被证明能够识别欧几里得空间中任意形状的聚类，但它难以处理表现出层次或树状结构的数据集。在这项工作中，我们引入了HypeGBMS，这是GBMS在双曲空间的一种新型扩展。我们的方法用双曲距离替代欧几里得计算，并采用Möbius加权均值确保所有更新与空间几何保持一致。HypeGBMS在保留GBMS的密度特性同时有效捕捉潜在层次结构。我们提供了关于收敛性和计算复杂性的理论见解，以及实证结果，证明了在层次数据集中的改进聚类质量。这项工作将经典的均值偏移聚类与双曲表示学习联系起来，为弯曲空间中的基于密度的聚类提供了有原则的方法。在11个真实世界数据集上的广泛实验评估表明，HypeGBMS在非欧几里得设置中显著优于传统的均值偏移聚类方法，凸显了其鲁棒性和有效性。


### 论文摘要

Clustering is a fundamental unsupervised learning task for uncovering patterns in data. While Gaussian Blurring Mean Shift (GBMS) has proven effective for identifying arbitrarily shaped clusters in Euclidean space, it struggles with datasets exhibiting hierarchical or tree-like structures. In this work, we introduce HypeGBMS, a novel extension of GBMS to hyperbolic space. Our method replaces Euclidean computations with hyperbolic distances and employs Möbius-weighted means to ensure that all updates remain consistent with the geometry of the space. HypeGBMS effectively captures latent hierarchies while retaining the density-seeking behavior of GBMS. We provide theoretical insights into convergence and computational complexity, along with empirical results that demonstrate improved clustering quality in hierarchical datasets. This work bridges classical mean-shift clustering and hyperbolic representation learning, offering a principled approach to density-based clustering in curved spaces. Extensive experimental evaluations on $11$ real-world datasets demonstrate that HypeGBMS significantly outperforms conventional mean-shift clustering methods in non-Euclidean settings, underscoring its robustness and effectiveness.

---

## 10. AgentBalance: Backbone-then-Topology Design for Cost-Effective Multi-Agent Systems under Budget Constraints

**论文链接:** [http://arxiv.org/abs/2512.11426v1](http://arxiv.org/abs/2512.11426v1)

**作者:** Shuowei Cai, Yansong Ning, Hao Liu

**发布时间:** 2025-12-12

### GPT解析

### 总结

AgentBalance是一个在明确token成本和延迟预算下构建成本效益高的多智能体系统（MAS）的框架，通过主干优先-拓扑优先的设计方法，实现了显著的性能提升和良好的预算适应性。

### 背景

大型语言模型（LLM）驱动的多智能体系统已成为大规模应用如网络搜索、社交网络分析和在线客户支持的关键构建模块，成本效益成为大规模部署的主要约束。

### 目的

解决现有MAS系统在预算约束下成本效益不足的问题，提出一个在明确token成本和延迟预算下构建成本效益高的MAS的框架。

### 方法

AgentBalance采用两阶段设计：首先进行面向主干的智能体生成，通过LLM池构建、池选择和角色-主干匹配构建具有异构主干的智能体；然后执行自适应MAS拓扑生成，通过智能体表示学习、门控和延迟感知拓扑合成指导智能体间通信。

### 主要发现

在包含14个候选LLM主干的基准测试中，AgentBalance在匹配的token成本和延迟预算下分别实现了高达10%和22%的性能提升，在跨基准的性能-预算曲线上表现出良好的AUC，可作为现有MAS的插件提高性能，并对未见过的LLMs具有良好的泛化能力。

### 结论

AgentBalance是一个有效的框架，能够在明确的预算约束下构建成本效益高的MAS系统，适用于实际预算感知的部署场景。

### 翻译

基于大型语言模型（LLM）的多智能体系统（MAS）正成为网络搜索、社交网络分析和在线客户支持等网络规模应用中不可或缺的构建模块，其中成本效益日益成为大规模部署的主要约束。虽然近期工作通过塑造智能体间通信拓扑和选择智能体主干来提高MAS的成本效益，但很少在反映部署约束的明确token成本和延迟预算下进行建模和优化。这通常导致拓扑优先的设计，并在预算受限时产生次优的成本效益。我们提出了AgentBalance，一个通过主干优先-拓扑优先设计在明确token成本和延迟预算下构建成本效益高的MAS的框架。AgentBalance首先执行面向主干的智能体生成，通过LLM池构建、池选择和角色-主干匹配来构建具有异构主干的智能体。然后执行自适应MAS拓扑生成，通过智能体表示学习、门控和延迟感知拓扑合成来指导智能体间通信。在包含14个候选LLM主干的基准测试中，AgentBalance在匹配的token成本和延迟预算下分别实现了高达10%和22%的性能提升，并在跨基准的性能-预算曲线上表现出良好的AUC。AgentBalance还可作为现有MAS的插件，在相同的token成本和延迟约束下提高性能，并且对未见过的LLMs具有良好的泛化能力，适用于实际、预算感知的部署。代码：https://github.com/usail-hkust/AgentBalance


### 论文摘要

Large Language Model (LLM)-based multi-agent systems (MAS) are becoming indispensable building blocks for web-scale applications such as web search, social network analytics, and online customer support, where cost-effectiveness is increasingly the primary constraint for large-scale deployment. While recent work improves MAS cost-effectiveness by shaping inter-agent communication topologies and selecting agent backbones, it rarely models and optimizes under explicit token-cost and latency budgets that reflect deployment constraints. This often leads to topology-first designs and suboptimal cost-effectiveness when budgets are binding. We present AgentBalance, a framework for constructing cost-effective MAS under explicit token-cost and latency budgets via a backbone-then-topology design. AgentBalance first performs backbone-oriented agent generation, constructing agents with heterogeneous backbones through LLM pool construction, pool selection, and role-backbone matching. It then performs adaptive MAS topology generation, guiding inter-agent communication via agent representation learning, gating, and latency-aware topology synthesis. Experiments on benchmarks with 14 candidate LLM backbones show that AgentBalance achieves up to 10% and 22% performance gains under matched token-cost and latency budgets, respectively, and yields strong AUC on performance-versus-budget curves across benchmarks. AgentBalance also functions as a plug-in for existing MAS, improving performance under the same token-cost and latency constraints, and it generalizes well to unseen LLMs for practical, budget-aware deployment. Code: https://github.com/usail-hkust/AgentBalance

---

## 11. Bhargava Cube--Inspired Quadratic Regularization for Structured Neural Embeddings

**论文链接:** [http://arxiv.org/abs/2512.11392v1](http://arxiv.org/abs/2512.11392v1)

**作者:** S Sairam, Prateek P Kulkarni

**发布时间:** 2025-12-12

**备注:** 12 pages, 3 figures

### GPT解析

### 总结

提出了一种结合数论中Bhargava立方体代数约束的新型神经表示学习方法

### 背景

传统深度学习方法在无结构的潜在空间中学习表示，缺乏可解释性和数学一致性

### 目的

创建一个受Bhargava立方体启发的框架，产生可解释且数学上一致的表示

### 方法

将输入数据映射到受约束的三维潜在空间，嵌入被正则化以满足从Bhargava组合结构导出的二次关系，使用可微的辅助损失函数独立于分类目标操作

### 主要发现

在MNIST上达到99.46%的准确率，产生可解释的3D嵌入，按数字类别自然聚类并满足学习的二次约束

### 结论

这是数论结构首次应用于神经表示学习，为在神经网络中融入结构化数学先验奠定了基础

### 翻译

我们提出了一种新型神经表示学习方法，融入了受数论中Bhargava立方体启发的代数约束。传统深度学习方法在缺乏可解释性和数学一致性的无结构潜在空间中学习表示。我们的框架将输入数据映射到受约束的三维潜在空间，其中嵌入被正则化以满足从Bhargava组合结构导出的二次关系。该架构采用可微的辅助损失函数，独立于分类目标操作，引导模型朝向数学结构化的表示。我们在MNIST上进行了评估，达到99.46%的准确率，同时产生可解释的3D嵌入，这些嵌入按数字类别自然聚类并满足学习的二次约束。与需要显式几何监督的现有流形学习方法不同，我们的方法通过可微约束引入弱代数先验，确保与标准优化的兼容性。这是数论结构首次应用于神经表示学习，为在神经网络中融入结构化数学先验奠定了基础。


### 论文摘要

We present a novel approach to neural representation learning that incorporates algebraic constraints inspired by Bhargava cubes from number theory. Traditional deep learning methods learn representations in unstructured latent spaces lacking interpretability and mathematical consistency. Our framework maps input data to constrained 3-dimensional latent spaces where embeddings are regularized to satisfy learned quadratic relationships derived from Bhargava's combinatorial structures. The architecture employs a differentiable auxiliary loss function operating independently of classification objectives, guiding models toward mathematically structured representations. We evaluate on MNIST, achieving 99.46% accuracy while producing interpretable 3D embeddings that naturally cluster by digit class and satisfy learned quadratic constraints. Unlike existing manifold learning approaches requiring explicit geometric supervision, our method imposes weak algebraic priors through differentiable constraints, ensuring compatibility with standard optimization. This represents the first application of number-theoretic constructs to neural representation learning, establishing a foundation for incorporating structured mathematical priors in neural networks.

---

## 12. Neuronal Attention Circuit (NAC) for Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.10282v2](http://arxiv.org/abs/2512.10282v2)

**作者:** Waleed Razzaq, Izis Kanjaraway, Yun-Bo Zhao

**发布时间:** 2025-12-11

**备注:** Paper for ICML2026

### GPT解析

### 总结

本文提出了一种名为神经元注意力电路(NAC)的新型生物可行的连续时间(CT)注意力机制，解决了传统注意力机制的离散性限制，通过将注意力logits计算重新表述为具有非线性互连门的一阶线性ODE的解，实现了高效的适应性动态。

### 背景

注意力机制在表征学习方面优于循环神经网络(RNNs)，但其离散性质限制了连续时间(CT)建模的发展。

### 目的

引入一种新的、生物上可行的CT-Attention机制，克服传统注意力机制的离散性限制，实现更有效的连续时间建模。

### 方法

提出神经元注意力电路(NAC)，将注意力logits计算重新表述为具有非线性互连门的一阶线性ODE的解，这些门源于重新利用C. elegans神经元电路策略(NCPs)的连接机制。NAC用稀疏感觉门替换密集投影用于key-query投影，并用具有两个头的稀疏骨干网络计算内容目标门和可学习时间常数门。此外，NAC支持三种注意力logits计算模式：显式欧拉积分、精确闭式解和稳态近似，并实现了稀疏Top-K串联方案以提高内存效率。

### 主要发现

NAC在多个领域（包括不规则时间序列分类、自动驾驶车辆的车道保持和工业预测）中实现了与竞争基线相当或更好的准确性。在运行时间和内存效率方面，NAC位于几个CT基线的中间位置。研究还提供了严格的理论保证，包括状态稳定性、有界近似误差和通用逼近。

### 结论

NAC是一种有效的连续时间注意力机制，它结合了生物启发的计算方法，在保持良好性能的同时，实现了适中的计算和内存效率。

### 翻译

注意力机制改善了RNNs的表征学习，但其离散性质限制了连续时间(CT)建模。我们引入了神经元注意力电路(NAC)，这是一种新颖的、生物上可行的CT-Attention机制，它将注意力logits计算重新表述为具有从重新利用C. elegans神经元电路策略(NCPs)连接机制派生的非线性互连门的一阶线性ODE的解。NAC用稀疏感觉门替换密集投影用于key-query投影，并用具有两个头的稀疏骨干网络计算内容目标门和可学习时间常数门，实现了高效的适应性动态。NAC支持三种注意力logits计算模式：(i)显式欧拉积分，(ii)精确闭式解，和(iii)稳态近似。为了提高内存强度，我们实现了一个稀疏Top-K串联方案，选择性选择key-query交互。我们提供了严格的理论保证，包括状态稳定性、有界近似误差和通用逼近。实验上，我们在多个领域实现了NAC，包括不规则时间序列分类、自动驾驶车辆的车道保持和工业预测。我们观察到NAC在准确性方面匹配或优于竞争基线，并且在运行时间和内存效率方面位于几个CT基线的中间位置。


### 论文摘要

Attention improves representation learning over RNNs, but its discrete nature limits continuous-time (CT) modeling. We introduce Neuronal Attention Circuit (NAC), a novel, biologically plausible CT-Attention mechanism that reformulates attention logits computation as the solution to a linear first-order ODE with nonlinear interlinked gates derived from repurposing \textit{C. elegans} Neuronal Circuit Policies (NCPs) wiring mechanism. NAC replaces dense projections with sparse sensory gates for key-query projections and a sparse backbone network with two heads for computing \textit{content-target} and \textit{learnable time-constant} gates, enabling efficient adaptive dynamics. NAC supports three attention logit computation modes: (i) explicit Euler integration, (ii) exact closed-form solution, and (iii) steady-state approximation. To improve memory intensity, we implemented a sparse Top-\emph{K} pairwise concatenation scheme that selectively curates key-query interactions. We provide rigorous theoretical guarantees, including state stability, bounded approximation errors, and universal approximation. Empirically, we implemented NAC in diverse domains, including irregular time-series classification, lane-keeping for autonomous vehicles, and industrial prognostics. We observed that NAC matches or outperforms competing baselines in accuracy and occupies an intermediate position in runtime and memory efficiency compared with several CT baselines.

---

## 13. Transfer Learning (Il)liquidity

**论文链接:** [http://arxiv.org/abs/2512.11731v1](http://arxiv.org/abs/2512.11731v1)

**作者:** Andrea Conti, Giacomo Morelli

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出了一种深度对数和指数神经网络架构，用于估计不流动市场中的风险中性密度(RND)，即使在只有少量期权报价的情况下也能有效恢复RND。

### 背景

在不流动市场中估计隐含于期权价格中的风险中性密度(RND)具有挑战性。

### 目的

开发一种能够处理不规则和不流动行权价格的RND估计方法。

### 方法

引入深度对数和指数神经网络架构，结合深度学习和迁移学习技术，并通过蒙特卡洛模拟和实证测试(使用SPX数据)进行验证。

### 主要发现

迁移学习可以在严重不流动性条件下改善RND估计；所提出的框架在极度不流动条件下仅使用三个期权报价就能恢复RND。

### 结论

该框架在处理不流动市场的RND估计问题上表现优异，具有实际应用价值。

### 翻译

在期权价格中隐含的风险中性密度(RND)估计具有挑战性，特别是在不流动市场中。我们引入了深度对数和指数神经网络架构，该架构利用深度学习和迁移学习来解决不规则和不流动行权价格存在情况下的RND估计问题。我们证明了模型的关键统计特性和估计量的一致性。我们通过蒙特卡洛模拟展示了迁移学习在严重不流动性条件下改善RND估计的优势，并在SPX数据上对其进行了实证测试，与流行的估计方法进行了比较。总体而言，我们的框架在极度不流动的条件下仅使用三个期权报价就能恢复RND。


### 论文摘要

The estimation of the Risk Neutral Density (RND) implicit in option prices is challenging, especially in illiquid markets. We introduce the Deep Log-Sum-Exp Neural Network, an architecture that leverages Deep and Transfer learning to address RND estimation in the presence of irregular and illiquid strikes. We prove key statistical properties of the model and the consistency of the estimator. We illustrate the benefits of transfer learning to improve the estimation of the RND in severe illiquidity conditions through Monte Carlo simulations, and we test it empirically on SPX data, comparing it with popular estimation methods. Overall, our framework shows recovery of the RND in conditions of extreme illiquidity with as few as three option quotes.

---

## 14. Kinetic Mining in Context: Few-Shot Action Synthesis via Text-to-Motion Distillation

**论文链接:** [http://arxiv.org/abs/2512.11654v1](http://arxiv.org/abs/2512.11654v1)

**作者:** Luca Cazzola, Ahed Alboody

**发布时间:** 2025-12-12

### GPT解析

### 总结

KineMIC是一种迁移学习框架，通过适应文本到动作生成模型解决人类活动识别中数据获取瓶颈问题，实现了少样本动作合成，显著提升了分类准确率。

### 背景

大型标注动作数据集的高昂获取成本是骨骼基人类活动识别的关键瓶颈。文本到动作生成模型虽能提供可扩展的合成数据，但其训练目标与HAR需求存在根本差异。

### 目的

解决文本到动作生成模型与人类活动识别需求之间的领域差距，开发一种能生成适合HAR分类器的动作数据的少样本动作合成方法。

### 方法

提出KineMIC框架，利用CLIP文本嵌入建立稀疏HAR标签与T2M源数据之间的语义对应关系，为运动学蒸馏提供软监督，指导微调通用T2M模型，将其转换为专门的少样本动作到动作生成器。

### 主要发现

使用HumanML3D作为源数据集，NTU RGB+D 120子集作为目标领域，每类仅10个样本的情况下，KineMIC生成的动作更加连贯，作为数据增强源带来了23.1%的准确率提升。

### 结论

KineMIC有效弥合了文本到动作生成模型与人类活动识别之间的领域差距，为解决HAR数据获取瓶颈提供了有效方案。

### 翻译

大型标注动作数据集的获取成本仍然是骨骼基人类活动识别的关键瓶颈。虽然文本到动作生成模型提供了可扩展的合成数据源，但其训练目标强调通用艺术动作，数据集结构与HAR对运动学精确、类别区分性动作的要求存在根本差异。这种差异造成了显著的领域差距，使得通用T2M模型无法生成适合HAR分类器的动作。为应对这一挑战，我们提出KineMIC(Kinetic Mining In Context)，一种少样本动作合成的迁移学习框架。KineMIC通过假设文本编码空间中的语义对应关系可以为运动学蒸馏提供软监督，将T2M扩散模型适应到HAR领域。我们通过运动挖掘策略实现这一点，利用CLIP文本嵌入建立稀疏HAR标签与T2M源数据之间的对应关系。这一过程指导微调，将通用T2M主干转换为专门的少样本动作到动作生成器。我们使用HumanML3D作为源T2M数据集，NTU RGB+D 120的子集作为目标HAR领域，每个动作类别随机仅选择10个样本来验证KineMIC。我们的方法生成的动作更加连贯，提供了强大的数据增强源，实现了23.1%的准确率提升。动画说明和补充材料可在(https://lucazzola.github.io/publications/kinemic)获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何利用大规模文本到动作(T2M)生成模型为骨骼基人类活动识别(HAR)生成合成数据的问题，特别是在数据稀缺的少样本场景下。这个问题很重要，因为获取高质量标注的动作数据集成本高昂且劳动密集，而骨骼基HAR在体育分析、人机协作和智能监控等领域具有广泛应用，但受限于数据可用性，特别是少样本情况下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了T2M模型与HAR任务间的领域差距(语义差距和运动学差距)，然后设计了KineMIC教师-学生框架。他们借鉴了CLIP文本编码器建立语义对应、MDM扩散模型作为基础架构、教师-学生知识蒸馏、LoRA高效微调以及对比学习对齐特征表示等现有工作。设计过程包括通过CLIP建立语义对应、使用运动学挖掘提取相关子序列，并将通用模型转变为专门的动作生成器。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用语义对应作为软监督，通过运动学挖掘将通用T2M模型适应到特定HAR领域，在少样本情况下生成高质量动作数据。整体流程包括：1)教师-学生架构(教师冻结，学生可训练)；2)语义对齐(用CLIP建立目标标签与源文本的对应关系)；3)运动学对齐(通过MIC模块学习帧级对齐)；4)运动学挖掘(提取源数据中相关帧段)；5)多目标损失优化(结合重构损失、蒸馏损失和动态窗口加权)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)首个解决T2M到A2M适应问题的方法；2)KineMIC教师-学生框架；3)语义检索与运动学挖掘相结合；4)专门的MIC模块。相比之前工作，不同之处在于：不同于传统少样本HAR的元学习/度量学习方法和Fukushi等人的GAN+跨域正则化方法，KineMIC采用'语义优先'策略；不同于直接使用预训练T2M模型或简单微调，KineMIC通过运动学挖掘生成更适合HAR分类器的动作，避免过拟合。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'KineMIC通过结合语义检索和运动学挖掘策略，成功将通用文本到动作模型适应为少样本场景下高质量的动作到动作生成器，显著提升了人类活动识别的准确性。'}


### 论文摘要

The acquisition cost for large, annotated motion datasets remains a critical bottleneck for skeletal-based Human Activity Recognition (HAR). Although Text-to-Motion (T2M) generative models offer a compelling, scalable source of synthetic data, their training objectives, which emphasize general artistic motion, and dataset structures fundamentally differ from HAR's requirements for kinematically precise, class-discriminative actions. This disparity creates a significant domain gap, making generalist T2M models ill-equipped for generating motions suitable for HAR classifiers. To address this challenge, we propose KineMIC (Kinetic Mining In Context), a transfer learning framework for few-shot action synthesis. KineMIC adapts a T2M diffusion model to an HAR domain by hypothesizing that semantic correspondences in the text encoding space can provide soft supervision for kinematic distillation. We operationalize this via a kinetic mining strategy that leverages CLIP text embeddings to establish correspondences between sparse HAR labels and T2M source data. This process guides fine-tuning, transforming the generalist T2M backbone into a specialized few-shot Action-to-Motion generator. We validate KineMIC using HumanML3D as the source T2M dataset and a subset of NTU RGB+D 120 as the target HAR domain, randomly selecting just 10 samples per action class. Our approach generates significantly more coherent motions, providing a robust data augmentation source that delivers a +23.1% accuracy points improvement. Animated illustrations and supplementary materials are available at (https://lucazzola.github.io/publications/kinemic).

---

## 15. Transfer learning of GW-Bethe-Salpeter Equation excitation energies

**论文链接:** [http://arxiv.org/abs/2512.11596v1](http://arxiv.org/abs/2512.11596v1)

**作者:** Dario Baum, Arno Förster, Lucas Visscher

**发布时间:** 2025-12-12

### GPT解析

### 总结

该研究展示了如何通过迁移学习解决机器学习中电子结构计算的低保真数据丰富与高保真数据稀缺之间的不平衡问题。预训练的图神经网络在有限的高保真数据微调后能够准确预测准粒子和激发能量，提高了准确性并减少了对昂贵高保真数据的依赖。

### 背景

机器学习在电子结构计算中面临的一个持久性挑战是低保真度数据（如DFT或TDDFT结果）丰富，而高保真度数据（如多体微扰论标签）稀缺之间的不平衡。

### 目的

展示如何通过迁移学习来弥合低保真数据与高保真数据之间的差距，实现准确的准粒子和激发能量预测。

### 方法

使用在DFT和TDDFT属性上预训练的图神经网络，然后用有限的qsGW和qsGW-BSE数据进行微调，以实现准粒子和激发能量的准确预测。

### 主要发现

预训练提高了准确性，减少了对昂贵的qsGW数据的依赖，即使对于比微调过程中见到的分子更大或化学性质不同的分子，也能减少大的预测异常值。

### 结论

多保真度迁移学习可以显著扩展多体级别预测在整个化学空间中的适用范围。

### 翻译

在电子结构计算的机器学习中，一个持久的挑战是低保真度数据（如DFT或TDDFT结果）丰富与高保真度数据（如多体微扰论标签）稀缺之间的尖锐不平衡。我们表明，迁移学习为弥合这一差距提供了有效途径：在DFT和TDDFT属性上预训练的图神经网络可以用有限的qsGW和qsGW-BSE数据进行微调，从而产生准粒子和激发能量的准确预测。在化学多样性测试集上评估全模型和仅输出微调，我们发现预训练提高了准确性，减少了对昂贵的qsGW数据的依赖，并减轻了大的预测异常值，即使对于比微调过程中见到的分子更大或化学性质不同的分子也是如此。我们的结果表明，多保真度迁移学习可以显著扩展多体级别预测在整个化学空间中的范围。


### 论文摘要

A persistent challenge in machine learning for electronic-structure calculations is the sharp imbalance between abundant low-fidelity data like DFT or TDDFT results and the scarcity of high-fidelity data like many-body perturbation theory labels. We show that transfer learning provides an effective route to bridge this gap: graph neural networks pretrained on DFT and TDDFT properties can be finetuned with limited qs$GW$ and qs$GW$-BSE data to yield accurate predictions of quasiparticle and excitation energies. Assessing both full-model and readout-only finetuning across chemically diverse test sets, we find that pretraining improves accuracy, reduces reliance on costly qs$GW$ data, and mitigates large predictive outliers even for molecules larger or chemically distinct from those seen during finetuning. Our results demonstrate that multi-fidelity transfer learning can substantially extend the reach of many-body-level predictions across chemical space.

---

## 16. Type II and Type III Solar Radio Burst Classification Using Transfer Learning

**论文链接:** [http://arxiv.org/abs/2512.11487v1](http://arxiv.org/abs/2512.11487v1)

**作者:** Herman le Roux, Ruhann Steyn, Du Toit Strauss, Mark Daly, Peter T. Gallagher, Jeremiah Scully, Shane A. Maloney, Christian Monstein, Gunther Drevin

**发布时间:** 2025-12-12

**备注:** 18 pages, 4 figures, Solar Physics Springer

### GPT解析

### 总结

研究使用深度学习模型自动分类太阳射电爆发(SRBs)，创建包含II型和III型SRB的频谱图数据集，并通过微调预训练模型实现高准确率的自动化分类。

### 背景

太阳周期性发出强烈的射电爆发(SRBs)，这些爆发会干扰无线电通信，并可能预示着大型太阳事件，这些事件可能破坏地球和太空中的技术基础设施。

### 目的

开发自动化的SRB分类系统，提高事件检测和实时监测能力，推进空间天气和相关现象研究的技术。

### 方法

创建使用e-Callisto网络记录的数据集，包含三类图像：空白频谱图、包含II型SRB的频谱图和包含III型SRB的频谱图；使用这些图像微调VGGnet-19、MobileNet、ResNet-152、DenseNet-201和YOLOv8等预训练深度学习模型进行分类。

### 主要发现

测试集上各模型的F1分数在87%到92%之间，YOLOv8表现最佳，证明使用预训练模型进行事件分类的有效性。

### 结论

使用预训练模型进行事件分类可以为SRB分类提供自动化解决方案，这种方法为II型SRB可用数据样本有限的问题提供了实用解决方案。

### 翻译

太阳周期性地发出强烈的射电爆发，称为太阳射电爆发(SRBs)。这些爆发会干扰无线电通信，并可能预示着大型太阳事件，这些事件可能破坏地球和太空中的技术基础设施。这些事件带来的风险突显了自动SRB分类的必要性，提供了提高事件检测和实时监测的潜力。这将推进用于研究空间天气及相关现象的技术。使用e-Callisto网络记录的数据创建了一个包含射电光谱图像的数据集。该数据集包含三类：空白频谱图；包含II型SRB的频谱图；以及包含III型SRB的频谱图。这些图像被用于微调几个流行的预训练深度学习模型，以分类II型和III型SRB。评估的模型包括VGGnet-19、MobileNet、ResNet-152、DenseNet-201和YOLOv8。在测试集上测试模型产生的F1分数在87%到92%之间。YOLOv8成为表现最佳的模型，证明使用预训练模型进行事件分类可以为SRB分类提供自动化解决方案。这种方法为II型SRB可用数据样本有限的问题提供了实用解决方案。


### 论文摘要

The Sun periodically emits intense bursts of radio emission known as solar radio bursts (SRBs). These bursts can disrupt radio communications and be indicative of large solar events that can disrupt technological infrastructure on Earth and in space. The risks posed by these events highlight the need for automated SRB classification, providing the potential to improve event detection and real-time monitoring. This would advance the techniques used to study space weather and related phenomena. A dataset containing images of radio spectra was created using data recorded by the Compound Astronomical Low frequency Low cost Instrument for Spectroscopy and Transportable Observatory (e-Callisto) network. This dataset comprises three categories: empty spectrograms; spectrograms containing Type II SRBs; and spectrograms containing Type III SRBs. These images were used to fine-tune several popular pre-trained deep learning models for classifying Type II and Type III SRBs. The evaluated models included VGGnet-19, MobileNet, ResNet-152, DenseNet-201, and YOLOv8. Testing the models on the test set produced F1 scores ranging from 87\% to 92\%. YOLOv8 emerged as the best-performing model among them, demonstrating that using pre-trained models for event classification can provide an automated solution for SRB classification. This approach provides a practical solution to the limited number of data samples available for Type II SRBs.

---

## 17. Reliable Detection of Minute Targets in High-Resolution Aerial Imagery across Temporal Shifts

**论文链接:** [http://arxiv.org/abs/2512.11360v1](http://arxiv.org/abs/2512.11360v1)

**作者:** Mohammad Sadegh Gholizadeh, Amir Arsalan Rezapour, Hamidreza Shayegh, Ehsan Pazouki

**发布时间:** 2025-12-12

### GPT解析

### 总结

本研究通过迁移学习初始化的Faster R-CNN架构解决了稻田中水稻幼苗检测的挑战，利用UAV数据集训练模型并验证了其在不同时间条件下的鲁棒性。

### 背景

通过无人机进行高效的作物检测对精准农业的规模化至关重要，但由于目标规模小和环境变化大，这仍然具有挑战性。

### 目的

解决在稻田中检测水稻幼苗的问题。

### 方法

利用通过迁移学习初始化的Faster R-CNN架构，构建了一个重要的UAV数据集进行训练，严格评估了模型的泛化能力，并通过在不同时间间隔获取的三个不同的测试集验证性能，评估对变化成像条件的鲁棒性。

### 主要发现

迁移学习不仅促进了农业背景下目标检测模型的快速收敛，尽管图像获取存在域偏移，但仍能保持一致的性能。

### 结论

迁移学习在农业目标检测中有效，能够处理域偏移问题。

### 翻译

通过无人机进行高效的作物检测对精准农业的规模化至关重要，但由于目标规模小和环境变化大，这仍然具有挑战性。本文通过利用迁移学习初始化的Faster R-CNN架构来解决稻田中水稻幼苗的检测问题。为了克服在高分辨率航空影像中检测微小物体的特定困难，我们整理了一个重要的UAV数据集进行训练，并严格评估了模型的泛化能力。具体而言，我们在三个不同时间间隔获取的不同测试集上验证性能，从而评估对变化成像条件的鲁棒性。我们的实证结果表明，迁移学习不仅促进了农业背景下目标检测模型的快速收敛，尽管图像获取存在域偏移，但仍能产生一致的性能。


### 论文摘要

Efficient crop detection via Unmanned Aerial Vehicles is critical for scaling precision agriculture, yet it remains challenging due to the small scale of targets and environmental variability. This paper addresses the detection of rice seedlings in paddy fields by leveraging a Faster R-CNN architecture initialized via transfer learning. To overcome the specific difficulties of detecting minute objects in high-resolution aerial imagery, we curate a significant UAV dataset for training and rigorously evaluate the model's generalization capabilities. Specifically, we validate performance across three distinct test sets acquired at different temporal intervals, thereby assessing robustness against varying imaging conditions. Our empirical results demonstrate that transfer learning not only facilitates the rapid convergence of object detection models in agricultural contexts but also yields consistent performance despite domain shifts in image acquisition.

---

## 18. HFS: Holistic Query-Aware Frame Selection for Efficient Video Reasoning

**论文链接:** [http://arxiv.org/abs/2512.11534v1](http://arxiv.org/abs/2512.11534v1)

**作者:** Yiqing Yang, Kin-Man Lam

**发布时间:** 2025-12-12

**备注:** 18 pages, 8 figures

### GPT解析

### 总结

本文提出了一种端到端可训练、任务自适应的视频帧选择框架，通过思维链引导小语言模型生成任务特定的隐式查询向量，结合多模态特征实现动态帧评分，并使用连续集级目标函数和学生-教师互学习机制优化帧选择过程，在多个基准测试上显著优于现有方法。

### 背景

视频理解中的关键帧选择面临重大挑战，传统top-K选择方法独立评分帧，无法整体优化选择，导致选择时间上聚集和视觉上冗余的帧。

### 目的

解决传统关键帧选择方法的局限性，提出一个能够动态适应任务目标的端到端可训练框架，提高帧选择的整体质量和效率。

### 方法

提出一个包含思维链引导的小语言模型生成任务特定隐式查询向量的框架，结合多模态特征实现动态帧评分；定义包含相关性、覆盖度和冗余度的连续集级目标函数，使用Gumbel-Softmax进行可微分优化；采用学生-教师互学习机制，通过KL散列对齐帧重要性分布，结合交叉熵损失实现端到端优化。

### 主要发现

通过思维链方法和小语言模型生成的任务特定查询向量可以有效改善帧选择质量；连续集级目标函数和学生-教师互学习机制能够显著提升帧选择性能，消除对静态伪标签的依赖。

### 结论

所提出的方法在Video-MME、LongVideoBench、MLVU和NExT-QA等多个基准测试上显著优于现有方法，证明了其在视频关键帧选择任务中的有效性和优越性。

### 翻译

视频理解中的关键帧选择面临重大挑战。传统的top-K选择方法独立评分帧，通常无法整体优化选择。这种独立评分经常导致选择时间上聚集和视觉上冗余的帧。此外，使用由多模态大语言模型(MLLMs)离线生成的伪标签训练轻量级选择器，阻止了监督信号动态适应任务目标。为了解决这些局限性，我们提出了一个用于帧选择的端到端可训练、任务自适应框架。思维链方法引导小语言模型(SLM)生成任务特定的隐式查询向量，这些向量与多模态特征结合以实现动态帧评分。我们进一步定义了一个包含相关性、覆盖度和冗余度的连续集级目标函数，通过Gumbel-Softmax实现可微分优化，以在集级选择最优帧组合。最后，采用学生-教师互学习，学生选择器(SLM)和教师推理器(MLLM)通过KL散列对齐其帧重要性分布。结合交叉熵损失，这实现了端到端优化，消除对静态伪标签的依赖。在Video-MME、LongVideoBench、MLVU和NExT-QA等多个基准测试上的实验表明，我们的方法显著优于现有方法。


### 论文摘要

Key frame selection in video understanding presents significant challenges. Traditional top-K selection methods, which score frames independently, often fail to optimize the selection as a whole. This independent scoring frequently results in selecting frames that are temporally clustered and visually redundant. Additionally, training lightweight selectors using pseudo labels generated offline by Multimodal Large Language Models (MLLMs) prevents the supervisory signal from dynamically adapting to task objectives. To address these limitations, we propose an end-to-end trainable, task-adaptive framework for frame selection. A Chain-of-Thought approach guides a Small Language Model (SLM) to generate task-specific implicit query vectors, which are combined with multimodal features to enable dynamic frame scoring. We further define a continuous set-level objective function that incorporates relevance, coverage, and redundancy, enabling differentiable optimization via Gumbel-Softmax to select optimal frame combinations at the set level. Finally, student-teacher mutual learning is employed, where the student selector (SLM) and teacher reasoner (MLLM) are trained to align their frame importance distributions via KL divergence. Combined with cross-entropy loss, this enables end-to-end optimization, eliminating reliance on static pseudo labels. Experiments across various benchmarks, including Video-MME, LongVideoBench, MLVU, and NExT-QA, demonstrate that our method significantly outperforms existing approaches.

---

## 19. TSkel-Mamba: Temporal Dynamic Modeling via State Space Model for Human Skeleton-based Action Recognition

**论文链接:** [http://arxiv.org/abs/2512.11503v1](http://arxiv.org/abs/2512.11503v1)

**作者:** Yanan Liu, Jun Liu, Hao Zhang, Dan Xu, Hossein Rahmani, Mohammed Bennamoun, Qiuhong Ke

**发布时间:** 2025-12-12

### GPT解析

### 总结

这篇论文提出了TSkel-Mamba，一个混合Transformer-Mamba框架，用于基于骨架的动作识别。该框架结合了空间Transformer的空间特征学习和Mamba的时间建模能力，并通过引入时间动态建模（TDM）块和多尺度时间交互（MTI）模块解决了Mamba在建模通道间依赖关系方面的局限性。实验证明该方法在多个数据集上实现了最先进的性能，同时保持了较高的效率。

### 背景

基于骨架的动作识别在计算机视觉领域受到了广泛关注。选择性状态空间模型（SSM）Mamba在建模一维时间序列方面取得了成功，但其对各个通道使用独立的SSM块的方式限制了建模通道间依赖关系的能力，特别是在骨架数据的时间依赖关系建模方面存在不足。

### 目的

开发一种能够有效捕捉骨架数据中空间和时间动态的框架，增强Mamba对骨架数据的适应性，提高其建模时间依赖关系的能力，从而实现高效且准确的基于骨架的动作识别。

### 方法

提出了TSkel-Mamba混合框架，利用空间Transformer进行空间特征学习，同时使用Mamba进行时间建模。为了解决Mamba在建模通道间依赖关系方面的局限性，引入了时间动态建模（TDM）块，这是一个即插即用组件，集成了多尺度时间交互（MTI）模块。MTI模块使用多尺度循环算子来捕获跨通道的时间交互。

### 主要发现

在NTU-RGB+D 60、NTU-RGB+D 120、NW-UCLA和UAV-Human数据集上的实验表明，TSkel-Mamba实现了最先进的性能，同时保持了较低的推理时间，证明了该方法的高效性和有效性。

### 结论

TSkel-Mamba框架通过结合Transformer和Mamba的优势，并引入TDM和MTI模块，成功解决了基于骨架的动作识别中的关键挑战，实现了性能和效率的良好平衡，为该领域的研究提供了新的思路和方法。

### 翻译

基于骨架的动作识别在计算机视觉界引起了广泛关注。受选择性状态空间模型（SSM）Mamba在建模一维时间序列方面最近成功的启发，我们提出了TSkel-Mamba，这是一个混合的Transformer-Mamba框架，能够有效捕捉空间和时间动态。特别是，我们的方法利用空间Transformer进行空间特征学习，同时使用Mamba进行时间建模。然而，Mamba对各个通道使用独立的SSM块，这 inherently 限制了其建模通道间依赖关系的能力。为了更好地使Mamba适应骨架数据并增强Mamba建模时间依赖关系的能力，我们引入了时间动态建模（TDM）块，这是一个通用的即插即用组件，集成了新的多尺度时间交互（MTI）模块。MTI模块采用多尺度循环算子来捕获跨通道的时间交互，这是动作识别的关键因素。在NTU-RGB+D 60、NTU-RGB+D 120、NW-UCLA和UAV-Human数据集上的大量实验表明，TSkel-Mamba实现了最先进的性能，同时保持了较低的推理时间，使其既高效又非常有效。


### 论文摘要

Skeleton-based action recognition has garnered significant attention in the computer vision community. Inspired by the recent success of the selective state-space model (SSM) Mamba in modeling 1D temporal sequences, we propose TSkel-Mamba, a hybrid Transformer-Mamba framework that effectively captures both spatial and temporal dynamics. In particular, our approach leverages Spatial Transformer for spatial feature learning while utilizing Mamba for temporal modeling. Mamba, however, employs separate SSM blocks for individual channels, which inherently limits its ability to model inter-channel dependencies. To better adapt Mamba for skeleton data and enhance Mamba`s ability to model temporal dependencies, we introduce a Temporal Dynamic Modeling (TDM) block, which is a versatile plug-and-play component that integrates a novel Multi-scale Temporal Interaction (MTI) module. The MTI module employs multi-scale Cycle operators to capture cross-channel temporal interactions, a critical factor in action recognition. Extensive experiments on NTU-RGB+D 60, NTU-RGB+D 120, NW-UCLA and UAV-Human datasets demonstrate that TSkel-Mamba achieves state-of-the-art performance while maintaining low inference time, making it both efficient and highly effective.

---

## 20. UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.11336v1](http://arxiv.org/abs/2512.11336v1)

**作者:** Hewen Pan, Cong Wei, Dashuang Liang, Zepeng Huang, Pengfei Gao, Ziqi Zhou, Lulu Xue, Pengfei Yan, Xiaoming Wei, Minghui Li, Shengshan Hu

**发布时间:** 2025-12-12

**备注:** 22 pages, 13 figures, technical report

### GPT解析

### 总结

本研究提出了UFVideo，首个具有统一多粒度协作理解能力的视频大型语言模型，能够处理全局、像素和时间尺度的视频理解任务。

### 背景

随着多模态大型语言模型的发展，视频大型语言模型得到了进一步发展，但现有工作仅限于专门的视频理解任务，无法实现全面和多粒度的视频感知。

### 目的

弥合现有工作与全面视频理解之间的差距，开发能够实现多粒度视频理解的统一模型。

### 方法

设计统一的视觉语言引导对齐机制，灵活处理单个模型内不同尺度的视频理解；UFVideo动态编码不同任务的视觉和文本输入，生成文本响应、时间定位或掩码；构建包含三个不同尺度协作任务的UFVideo-Bench评估数据集；在9个公共基准测试上验证模型有效性。

### 主要发现

UFVideo在多粒度视频理解任务上展现出灵活性和优势，优于GPT-4o；模型在多种常见视频理解任务上表现出有效性。

### 结论

UFVideo为未来视频大型语言模型的发展提供了有价值的见解和方向。

### 翻译

随着多模态大型语言模型(LLMs)的进步，视频大型语言模型(Video LLMs)已得到进一步发展，能够执行全面和专门的视频理解。然而，现有工作仅限于专门的视频理解任务，无法实现全面和多粒度的视频感知。为了弥合这一差距，我们引入了UFVideo，这是第一个具有统一多粒度协作理解能力的视频LLM。具体而言，我们设计了统一的视觉语言引导对齐，以灵活处理单个模型内全局、像素和时间尺度的视频理解。UFVideo动态编码不同任务的视觉和文本输入，并生成文本响应、时间定位或掩码。此外，为了评估具有挑战性的多粒度视频理解任务，我们构建了UFVideo-Bench，其中包含三个不同尺度的协作任务，这展示了UFVideo相对于GPT-4o的灵活性和优势。此外，我们在9个涵盖各种常见视频理解任务的公共基准上验证了我们模型的有效性，为未来的视频LLMs提供了有价值的见解。


### 论文摘要

With the advancement of multi-modal Large Language Models (LLMs), Video LLMs have been further developed to perform on holistic and specialized video understanding. However, existing works are limited to specialized video understanding tasks, failing to achieve a comprehensive and multi-grained video perception. To bridge this gap, we introduce UFVideo, the first Video LLM with unified multi-grained cooperative understanding capabilities. Specifically, we design unified visual-language guided alignment to flexibly handle video understanding across global, pixel and temporal scales within a single model. UFVideo dynamically encodes the visual and text inputs of different tasks and generates the textual response, temporal localization, or grounded mask. Additionally, to evaluate challenging multi-grained video understanding tasks, we construct the UFVideo-Bench consisting of three distinct collaborative tasks within the scales, which demonstrates UFVideo's flexibility and advantages over GPT-4o. Furthermore, we validate the effectiveness of our model across 9 public benchmarks covering various common video understanding tasks, providing valuable insights for future Video LLMs.

---

## 21. Physics-Informed Video Flare Synthesis and Removal Leveraging Motion Independence between Flare and Scene

**论文链接:** [http://arxiv.org/abs/2512.11327v1](http://arxiv.org/abs/2512.11327v1)

**作者:** Junqiao Wang, Yuanfei Huang, Hua Huang

**发布时间:** 2025-12-12

### GPT解析

### 总结

该论文提出了一种基于物理信息的动态眩光合成方法和视频眩光去除网络，解决了视频处理中眩光去除的挑战，构建了首个视频眩光数据集，并在实验中证明了该方法的有效性。

### 背景

镜头眩光是强光源引起的退化现象。现有眩光去除研究主要集中在图像上，而视频眩光的时空特性尚未得到充分探索。

### 目的

解决视频眩光合成和去除中的挑战，特别是处理眩光、光源和场景内容之间的复杂且相互独立的运动问题，避免恢复过程中出现的闪烁和伪影。

### 方法

提出一个基于物理信息的动态眩光合成流水线，使用光流模拟光源运动并建模眩光时序行为；设计一个视频眩光去除网络，采用注意力模块抑制眩光区域，并融入基于Mamba的时序建模组件捕获长程时空依赖关系。

### 主要发现

运动独立的时空表示有效消除了多帧对齐需求，减轻了眩光和场景内容之间的时间混叠，提高了视频眩光去除性能。

### 结论

该方法在真实和合成视频上始终优于现有方法，能够有效去除动态眩光，同时保持光源完整性并维持场景的时空一致性。

### 翻译

镜头眩光是由强光源引起的退化现象。现有的眩光去除研究主要集中在图像上，而视频眩光的时空特性在很大程度上仍未被探索。视频眩光的合成和去除比图像中具有更大的挑战性，这是由于眩光、光源和场景内容之间复杂且相互独立的运动。这种运动独立性进一步影响恢复性能，常常导致闪烁和伪影。为解决这一问题，我们提出了一个基于物理信息的动态眩光合成流水线，使用光流模拟光源运动，并建模散射和反射眩光的时序行为。同时，我们设计了一个视频眩光去除网络，采用注意力模块在空间上抑制眩光区域，并融入基于Mamba的时序建模组件来捕获长程时空依赖关系。这种运动独立的时空表示有效消除了多帧对齐的需求，减轻了眩光和场景内容之间的时间混叠，从而提高了视频眩光去除性能。在此基础上，我们构建了第一个视频眩光数据集，以全面评估我们的方法，包括大量合成配对视频和从互联网收集的额外真实世界视频，用于评估泛化能力。大量实验表明，我们的方法在真实和合成视频上始终优于现有的基于视频的恢复方法和基于图像的眩光去除方法，能够有效去除动态眩光，同时保持光源完整性并维持场景的时空一致性。


### 论文摘要

Lens flare is a degradation phenomenon caused by strong light sources. Existing researches on flare removal have mainly focused on images, while the spatiotemporal characteristics of video flare remain largely unexplored. Video flare synthesis and removal pose significantly greater challenges than in image, owing to the complex and mutually independent motion of flare, light sources, and scene content. This motion independence further affects restoration performance, often resulting in flicker and artifacts. To address this issue, we propose a physics-informed dynamic flare synthesis pipeline, which simulates light source motion using optical flow and models the temporal behaviors of both scattering and reflective flares. Meanwhile, we design a video flare removal network that employs an attention module to spatially suppress flare regions and incorporates a Mamba-based temporal modeling component to capture long range spatio-temporal dependencies. This motion-independent spatiotemporal representation effectively eliminates the need for multi-frame alignment, alleviating temporal aliasing between flares and scene content and thereby improving video flare removal performance. Building upon this, we construct the first video flare dataset to comprehensively evaluate our method, which includes a large set of synthetic paired videos and additional real-world videos collected from the Internet to assess generalization capability. Extensive experiments demonstrate that our method consistently outperforms existing video-based restoration and image-based flare removal methods on both real and synthetic videos, effectively removing dynamic flares while preserving light source integrity and maintaining spatiotemporal consistency of scene.

---

## 22. Graph Embedding with Mel-spectrograms for Underwater Acoustic Target Recognition

**论文链接:** [http://arxiv.org/abs/2512.11545v1](http://arxiv.org/abs/2512.11545v1)

**作者:** Sheng Feng, Shuqing Ma, Xiaoqian Zhu

**发布时间:** 2025-12-12

**DOI:** 10.1109/JOE.2025.3619314

### GPT解析

### 总结

本文提出了一种名为UATR-GTransformer的非欧几里得深度学习模型，结合Transformer架构和图神经网络，用于水下声学目标识别，有效处理了水下声信号的复杂拓扑结构。

### 背景

水下声学目标识别面临船舶辐射噪声复杂性和海洋环境多变性的挑战。现有深度学习方法大多假设水下声学数据位于欧几里得空间，但这一假设不适合水下声信号固有的非平稳、非高斯和非线性的复杂拓扑结构。

### 目的

为了克服现有模型的局限性，本文提出了一种非欧几里得深度学习模型UATR-GTransformer，整合Transformer架构和图神经网络，更好地处理水下声信号的复杂拓扑特性。

### 方法

UATR-GTransformer模型包含三个关键组件：Mel分块块将Mel谱图分割成重叠块；GTransformer块使用Transformer编码器捕获块间相互信息生成Mel图嵌入；分类头通过图神经网络增强嵌入并建模局部邻域关系，最后用前馈网络进行特征变换。

### 主要发现

基于两个基准数据集的实验表明，UATR-GTransformer性能与最先进方法相当。可解释性分析显示该模型能有效提取丰富的频域信息。

### 结论

UATR-GTransformer模型能够有效处理水下声信号的复杂特性，在水下声学目标识别任务中表现优异，在海洋工程应用中具有潜力。

### 翻译

水下声学目标识别由于船舶辐射噪声的复杂性和海洋环境的多变性而极具挑战性。尽管深度学习方法已取得有希望的结果，但大多数现有模型隐含地假设水下声学数据位于欧几里得空间中。然而，这种假设不适合水下声信号固有的复杂拓扑结构，这些信号表现出非平稳、非高斯和非线性的特性。为了克服这一局限，本文提出了UATR-GTransformer，一种结合了Transformer架构和图神经网络的非欧几里得深度学习模型。该模型包含三个关键组件：Mel分块块、GTransformer块和分类头。Mel分块块将Mel谱图分割成重叠的块，而GTransformer块使用Transformer编码器捕获分割块之间的相互信息以生成Mel图嵌入。随后，图神经网络通过建模局部邻域关系增强这些嵌入，前馈网络进一步执行特征变换。基于两个广泛使用的基准数据集的实验结果表明，UATR-GTransformer实现了与最先进方法相竞争的性能。此外，可解释性分析揭示，所提出的模型有效提取了丰富的频域信息，突显了其在海洋工程应用中的潜力。


### 论文摘要

Underwater acoustic target recognition (UATR) is extremely challenging due to the complexity of ship-radiated noise and the variability of ocean environments. Although deep learning (DL) approaches have achieved promising results, most existing models implicitly assume that underwater acoustic data lie in a Euclidean space. This assumption, however, is unsuitable for the inherently complex topology of underwater acoustic signals, which exhibit non-stationary, non-Gaussian, and nonlinear characteristics. To overcome this limitation, this paper proposes the UATR-GTransformer, a non-Euclidean DL model that integrates Transformer architectures with graph neural networks (GNNs). The model comprises three key components: a Mel patchify block, a GTransformer block, and a classification head. The Mel patchify block partitions the Mel-spectrogram into overlapping patches, while the GTransformer block employs a Transformer Encoder to capture mutual information between split patches to generate Mel-graph embeddings. Subsequently, a GNN enhances these embeddings by modeling local neighborhood relationships, and a feed-forward network (FFN) further performs feature transformation. Experiments results based on two widely used benchmark datasets demonstrate that the UATR-GTransformer achieves performance competitive with state-of-the-art methods. In addition, interpretability analysis reveals that the proposed model effectively extracts rich frequency-domain information, highlighting its potential for applications in ocean engineering.

---

## 23. CAT: Can Trust be Predicted with Context-Awareness in Dynamic Heterogeneous Networks?

**论文链接:** [http://arxiv.org/abs/2512.11352v1](http://arxiv.org/abs/2512.11352v1)

**作者:** Jie Wang, Zheng Yan, Jiahe Lan, Xuyan Li, Elisa Bertino

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出CAT模型，首个支持信任动态性和上下文感知的GNN信任预测模型，由图构建层、嵌入层、异构注意力层和预测层组成，能够处理动态图、捕捉时间信息、建模图异构性并提取上下文特征。

### 背景

信任预测为决策制定、风险缓解和系统安全增强提供支持。图神经网络(GNNs)因能学习表达节点表示而成为信任预测的有前景方法，但现有模型存在三个局限：无法捕捉信任动态性、忽略网络异构性、不支持上下文感知。

### 目的

开发能够捕捉信任动态性、处理网络异构性并支持上下文感知的信任预测模型，解决现有GNN信任预测模型的局限性。

### 方法

CAT模型使用连续时间表示处理动态图，通过时间编码函数捕捉时间信息；采用双重注意力机制建模图异构性；引入元路径概念提取上下文特征；通过构建上下文嵌入和集成上下文感知聚合器预测上下文感知信任和整体信任。

### 主要发现

在三个真实世界数据集上的实验表明，CAT在信任预测方面优于五组基线模型，同时表现出对大规模图的强大可扩展性，以及对信任导向和GNN导向攻击的鲁棒性。

### 结论

CAT成功解决了现有GNN信任预测模型的三个主要局限性，通过整合动态图处理、异构建模和上下文感知机制，提供了更全面、更准确的信任预测解决方案，在各种场景下表现出优越的性能和鲁棒性。

### 翻译

信任预测为决策制定、风险缓解和系统安全增强提供了有价值的支持。近年来，图神经网络(GNNs)已成为信任预测的一种有前景的方法，因为它们能够学习表达节点表示，捕捉网络内复杂的信任关系。然而，当前基于GNN的信任预测模型面临几个限制：(i)大多数模型无法捕捉信任的动态性，导致推理可疑。(ii)它们很少考虑现实世界网络的异构性质，导致丰富语义的丢失。(iii)它们都不支持信任的基本属性——上下文感知性，使预测结果变得粗粒度。为此，我们提出了CAT，这是第一个支持信任动态性并准确表示现实世界异构性的上下文感知GNN信任预测模型。CAT由图构建层、嵌入层、异构注意力层和预测层组成。它使用连续时间表示处理动态图，并通过时间编码函数捕捉时间信息。为了建模图异构性和利用语义信息，CAT采用双重注意力机制，识别不同节点类型及其各自类型内节点的重要性。为了实现上下文感知，我们引入了元路径的新概念来提取上下文特征。通过构建上下文嵌入和集成上下文感知聚合器，CAT可以预测上下文感知信任和整体信任。在三个真实世界数据集上的大量实验表明，CAT在信任预测方面优于五组基线模型，同时表现出对大规模图的强大可扩展性，以及对信任导向和GNN导向攻击的鲁棒性。


### 论文摘要

Trust prediction provides valuable support for decision-making, risk mitigation, and system security enhancement. Recently, Graph Neural Networks (GNNs) have emerged as a promising approach for trust prediction, owing to their ability to learn expressive node representations that capture intricate trust relationships within a network. However, current GNN-based trust prediction models face several limitations: (i) Most of them fail to capture trust dynamicity, leading to questionable inferences. (ii) They rarely consider the heterogeneous nature of real-world networks, resulting in a loss of rich semantics. (iii) None of them support context-awareness, a basic property of trust, making prediction results coarse-grained.   To this end, we propose CAT, the first Context-Aware GNN-based Trust prediction model that supports trust dynamicity and accurately represents real-world heterogeneity. CAT consists of a graph construction layer, an embedding layer, a heterogeneous attention layer, and a prediction layer. It handles dynamic graphs using continuous-time representations and captures temporal information through a time encoding function. To model graph heterogeneity and leverage semantic information, CAT employs a dual attention mechanism that identifies the importance of different node types and nodes within each type. For context-awareness, we introduce a new notion of meta-paths to extract contextual features. By constructing context embeddings and integrating a context-aware aggregator, CAT can predict both context-aware trust and overall trust. Extensive experiments on three real-world datasets demonstrate that CAT outperforms five groups of baselines in trust prediction, while exhibiting strong scalability to large-scale graphs and robustness against both trust-oriented and GNN-oriented attacks.

---

## 24. Condensation-Concatenation Framework for Dynamic Graph Continual Learning

**论文链接:** [http://arxiv.org/abs/2512.11317v1](http://arxiv.org/abs/2512.11317v1)

**作者:** Tingxu Yan, Ye Yuan

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出了一种名为CCC（基于凝聚-连接的持续学习）的新型框架，用于解决动态图神经网络中的灾难性遗忘问题。

### 背景

动态图在实际场景中普遍存在，其连续的结构变化会导致图神经网络(GNNs)出现灾难性遗忘现象。

### 目的

解决现有动态图持续学习方法忽略拓扑变化对现有节点影响的问题。

### 方法

提出CCC框架，首先将历史图快照压缩为保留原始标签分布和拓扑特性的紧凑语义表示，然后有选择地将这些历史嵌入与当前图表示连接起来，同时改进遗忘度量(FM)以更好地适应动态图场景。

### 主要发现

CCC在四个真实数据集上的实验中表现出优于最先进基线方法的性能。

### 结论

CCC框架能有效处理动态图中的持续学习问题，减轻灾难性遗忘现象。

### 翻译

动态图在实际场景中很常见，其连续的结构变化会导致图神经网络(GNNs)出现灾难性遗忘。虽然持续学习已扩展到动态图，但现有方法忽略了拓扑变化对现有节点的影响。为此，我们提出了一个用于动态图持续学习的新型框架，名为基于凝聚-连接的持续学习(CCC)。具体而言，CCC首先将历史图快照压缩为紧凑的语义表示，同时保留原始标签分布和拓扑特性。然后有选择地将这些历史嵌入与当前图表示连接起来。此外，我们改进了遗忘度量(FM)，通过量化结构更新导致的现有节点预测性能退化，使其更好地适应动态图场景。在大量实验中，CCC在四个真实数据集上表现出优于最先进基线的性能。


### 论文摘要

Dynamic graphs are prevalent in real-world scenarios, where continuous structural changes induce catastrophic forgetting in graph neural networks (GNNs). While continual learning has been extended to dynamic graphs, existing methods overlook the effects of topological changes on existing nodes. To address it, we propose a novel framework for continual learning on dynamic graphs, named Condensation-Concatenation-based Continual Learning (CCC). Specifically, CCC first condenses historical graph snapshots into compact semantic representations while aiming to preserve the original label distribution and topological properties. Then it concatenates these historical embeddings with current graph representations selectively. Moreover, we refine the forgetting measure (FM) to better adapt to dynamic graph scenarios by quantifying the predictive performance degradation of existing nodes caused by structural updates. CCC demonstrates superior performance over state-of-the-art baselines across four real-world datasets in extensive experiments.

---

## 25. Personalized Pricing in Social Networks with Individual and Group Fairness Considerations

**论文链接:** [http://arxiv.org/abs/2512.11252v1](http://arxiv.org/abs/2512.11252v1)

**作者:** Zeyu Chen, Bintong Chen, Wei Qian, Jing Huang

**发布时间:** 2025-12-12

### GPT解析

### 总结

个性化定价是根据客户特征为同一产品设定不同价格以提高零售商收入的做法，但这种做法常引发个人和群体层面的公平性问题。本文提出了一种结合两个维度公平性的个性化定价新方法，称为FairPricing框架。

### 背景

个性化定价根据客户特征为同一产品设定不同价格以提高零售商收入，但这种做法常引发公平性问题。个人层面：客户如果发现自己比他人支付更高价格，会感到不公平；群体层面：价格差异可能导致对某些受保护群体的歧视（如基于性别或种族）。现有研究通常分别处理个人和群体公平性。

### 目的

提出一种新的个性化定价问题公式化方法，将个人和群体两个公平性维度整合到社交网络环境中，开发一种能够适应网络变化的个性化定价策略。

### 方法

提出FairPricing框架，基于图神经网络(GNN)，利用客户特征和网络拓扑结构学习个性化定价策略。通过对客户需求施加惩罚来捕捉个人感知不公平，使用对抗性去偏和价格正则化项减轻群体层面的歧视。与需要重新优化的现有方法不同，FairPricing能根据更新后的网络结构为客户分配个性化价格。

### 主要发现

FairPricing实现了高盈利能力，改善了个人公平性感知，满足了群体公平性要求。

### 结论

FairPricing框架能够有效平衡盈利能力与公平性，该方法能够适应网络结构变化，具有良好的泛化能力。

### 翻译

个性化定价根据客户特定特征为同一产品设定不同价格，以提高零售商收入。然而，这种做法常常在个人和群体层面引发公平性担忧。在个人层面，如果客户注意到自己比他人支付更高价格，可能会感到不公平对待。在群体层面，价格差异可能导致对某些受保护群体的歧视，如基于性别或种族定义的群体。现有关于公平定价的研究通常分别处理个人和群体公平性。本文通过引入一种新的个性化定价问题公式化方法弥合这一差距，该方法在社交网络环境中整合了两个公平性维度。为解决此问题，我们提出了FairPricing，一种基于图神经网络(GNN)的新颖框架，它利用客户特征和网络拓扑结构学习个性化定价策略。在FairPricing中，个人感知的不公平性通过对客户需求（进而对利润目标）施加惩罚来捕捉，而群体层面的歧视则通过对抗性去偏和价格正则化项来减轻。与现有的基于优化的个性化定价不同（每当网络更新时需要重新优化），FairPricing学习到的定价策略能够根据客户特征和新网络结构为更新网络中的所有客户分配个性化价格，从而能够泛化到网络变化。大量实验结果表明，FairPricing在实现高盈利能力的同时，改善了个人公平性感知，并满足群体公平性要求。


### 论文摘要

Personalized pricing assigns different prices to customers for the same product based on customer-specific features to improve retailer revenue. However, this practice often raises concerns about fairness at both the individual and group levels. At the individual level, a customer may perceive unfair treatment if he/she notices being charged a higher price than others. At the group level, pricing disparities can result in discrimination against certain protected groups, such as those defined by gender or race. Existing studies on fair pricing typically address individual and group fairness separately. This paper bridges the gap by introducing a new formulation of the personalized pricing problem that incorporates both dimensions of fairness in social network settings. To solve the problem, we propose FairPricing, a novel framework based on graph neural networks (GNNs) that learns a personalized pricing policy using customer features and network topology. In FairPricing, individual perceived unfairness is captured through a penalty on customer demand, and thus the profit objective, while group-level discrimination is mitigated using adversarial debiasing and a price regularization term. Unlike existing optimization-based personalized pricing, which requires re-optimization whenever the network updates, the pricing policy learned by FairPricing assigns personalized prices to all customers in an updated network based on their features and the new network structure, thereby generalizing to network changes. Extensive experimental results show that FairPricing achieves high profitability while improving individual fairness perceptions and satisfying group fairness requirements.

---

## 26. Refining Graphical Neural Network Predictions Using Flow Matching for Optimal Power Flow with Constraint-Satisfaction Guarantee

**论文链接:** [http://arxiv.org/abs/2512.11127v1](http://arxiv.org/abs/2512.11127v1)

**作者:** Kshitiz Khanal

**发布时间:** 2025-12-11

### GPT解析

### 总结

该研究提出了一种新颖的两阶段学习框架，结合物理信息图神经网络(GNNs)与连续流匹配(CFM)技术，用于解决直流最优潮流(DC-OPF)问题。该方法通过将电力系统基本物理原理嵌入训练目标，实现了快速且接近最优的解决方案，同时满足所有约束条件。

### 背景

直流最优潮流(DC-OPF)问题是电力系统运行的基础，需要快速求解以实现实时电网管理。传统优化求解器虽然能提供最优解，但对于需要频繁重新计算的大规模系统，其计算成本过高。机器学习方法虽然有望加速求解，但通常难以满足约束条件和成本最优性要求。

### 目的

开发一种能够快速求解DC-OPF问题的新方法，结合传统优化求解器的最优性和机器学习方法的快速性，同时满足电力系统的物理约束和运行要求。

### 方法

提出两阶段学习框架：第一阶段训练图神经网络(GNN)，通过学习嵌入物理原理的损失函数来产生可行初始解；第二阶段采用连续流匹配(CFM)技术，通过学习向量场回归来优化这些解。将经济调度最优性条件、基尔霍夫定律和KKT互补条件等基本物理原理直接嵌入训练目标。

### 主要发现

在IEEE 30节点系统上针对五种负荷场景(70%-130%标称负荷)进行评估：对于标称负荷，实现了成本差距低于0.1%的近最优解；对于极端条件，成本差距低于3%；同时保持了100%的可行性。

### 结论

该框架弥合了快速但近似的神经网络预测与最优但缓慢的数值求解器之间的差距，为具有高可再生能源渗透率且需要频繁调度更新的现代电力系统提供了实用的解决方案。

### 翻译

直流最优潮流(DC-OPF)问题是电力系统运行的基础，需要快速求解以实现实时电网管理。虽然传统优化求解器能提供最优解，但对于需要频繁重新计算的大规模系统，其计算成本变得过高。机器学习方法虽然有望加速求解，但通常难以满足约束条件和成本最优性要求。我们提出了一种新颖的两阶段学习框架，结合了物理信息图神经网络(GNNs)与连续流匹配(CFM)技术，用于解决DC-OPF问题。我们的方法将基本物理原理（包括经济调度最优性条件、基尔霍夫定律和Karush-Kuhn-Tucker(KKT)互补条件）直接嵌入训练目标。第一阶段训练GNN通过学习编码电力系统约束的物理信息损失函数来产生可行的初始解。第二阶段采用CFM，一种无需模拟的连续归一化流技术，通过学习向量场回归来优化这些解。在IEEE 30节点系统上针对从70%到130%标称负荷的五种负荷场景进行评估，我们的方法实现了成本差距低于0.1%的近最优解（标称负荷）和低于3%（极端条件），同时保持100%的可行性。我们的框架弥合了快速但近似的神经网络预测与最优但缓慢的数值求解器之间的差距，为具有高可再生能源渗透率且需要频繁调度更新的现代电力系统提供了实用的解决方案。


### 论文摘要

The DC Optimal Power Flow (DC-OPF) problem is fundamental to power system operations, requiring rapid solutions for real-time grid management. While traditional optimization solvers provide optimal solutions, their computational cost becomes prohibitive for large-scale systems requiring frequent recalculations. Machine learning approaches offer promise for acceleration but often struggle with constraint satisfaction and cost optimality. We present a novel two-stage learning framework that combines physics-informed Graph Neural Networks (GNNs) with Continuous Flow Matching (CFM) for solving DC-OPF problems. Our approach embeds fundamental physical principles--including economic dispatch optimality conditions, Kirchhoff's laws, and Karush-Kuhn-Tucker (KKT) complementarity conditions--directly into the training objectives. The first stage trains a GNN to produce feasible initial solutions by learning from physics-informed losses that encode power system constraints. The second stage employs CFM, a simulation-free continuous normalizing flow technique, to refine these solutions toward optimality through learned vector field regression. Evaluated on the IEEE 30-bus system across five load scenarios ranging from 70\% to 130\% nominal load, our method achieves near-optimal solutions with cost gaps below 0.1\% for nominal loads and below 3\% for extreme conditions, while maintaining 100\% feasibility. Our framework bridges the gap between fast but approximate neural network predictions and optimal but slow numerical solvers, offering a practical solution for modern power systems with high renewable penetration requiring frequent dispatch updates.

---

## 27. Text2Graph: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios

**论文链接:** [http://arxiv.org/abs/2512.10061v2](http://arxiv.org/abs/2512.10061v2)

**作者:** João Lucas Luz Lima Sarcinelli, Ricardo Marcondes Marcacini

**发布时间:** 2025-12-10

### GPT解析

### 总结

Text2Graph是一个开源Python包，通过结合大型语言模型的部分标注和图神经网络的标签传播，实现了在降低能源和环境成本的同时保持分类性能的可持续文本分类解决方案。

### 背景

大型语言模型已成为有效的零样本分类器，但它们的高计算需求和环境成本限制了在高性能计算环境中大规模标注的实用性。

### 目的

开发一个支持更可持续工作流的工具，提供现有文本到图分类方法的模块化实现，使研究人员能够以更低的资源消耗进行文本分类。

### 方法

Text2Graph框架允许用户灵活地组合基于LLM的部分标注与GNN标签传播，并可轻松交换特征提取器、边构建方法和采样策略等组件。

### 主要发现

在五个涵盖主题分类和情感分析任务的零样本设置上测试显示，基于图的传播方法能够以一小部分能源和环境成本实现与大型语言模型相当的分类性能。

### 结论

图神经网络标签传播是一种可持续的替代方案，可以在显著降低计算资源消耗和环境影响的同时，保持与大型语言模型相当的文本分类性能。

### 翻译

大型语言模型已成为有效的零样本分类器，但它们的高计算需求和环境成本限制了在高性能计算环境中大规模标注的实用性。为支持更可持续的工作流程，我们提出了Text2Graph，一个开源Python包，提供了现有文本到图分类方法的模块化实现。该框架使用户能够灵活地结合基于LLM的部分标注和图神经网络标签传播，轻松交换特征提取器、边构建方法和采样策略等组件。我们在五个涵盖主题分类和情感分析任务的零样本设置上对Text2Graph进行了基准测试，将多个变体与其他零样本文本分类方法进行比较。除了报告性能外，我们还提供了详细的能源消耗和碳排放估计，显示基于图的传播方法以一小部分的能源和环境成本实现了具有竞争力的结果。


### 论文摘要

Large Language Models (LLMs) have become effective zero-shot classifiers, but their high computational requirements and environmental costs limit their practicality for large-scale annotation in high-performance computing (HPC) environments. To support more sustainable workflows, we present Text2Graph, an open-source Python package that provides a modular implementation of existing text-to-graph classification approaches. The framework enables users to combine LLM-based partial annotation with Graph Neural Network (GNN) label propagation in a flexible manner, making it straightforward to swap components such as feature extractors, edge construction methods, and sampling strategies. We benchmark Text2Graph on a zero-shot setting using five datasets spanning topic classification and sentiment analysis tasks, comparing multiple variants against other zero-shot approaches for text classification. In addition to reporting performance, we provide detailed estimates of energy consumption and carbon emissions, showing that graph-based propagation achieves competitive results at a fraction of the energy and environmental cost.

---

## 28. BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.09385v2](http://arxiv.org/abs/2512.09385v2)

**作者:** Uisang Lee, Changhoon Chung, Junmo Lee, Soo-Mook Moon

**发布时间:** 2025-12-10

**备注:** This paper is accepted to AAAI 2026

### GPT解析

### 总结

BugSweeper是一个端到端的深度学习框架，直接从源代码检测智能合约漏洞，无需手动规则设计，显著优于现有检测方法。

### 背景

以太坊快速增长使智能合约漏洞检测变得重要，但现有基于机器学习的方法多依赖专家设计的规则预处理，这种方法会丢弃源代码关键上下文，可能导致漏洞被忽略且适应性差。

### 目的

开发一个无需手动工程的端到端深度学习框架，直接从源代码检测智能合约漏洞。

### 方法

BugSweeper将每个Solidity函数表示为函数级抽象语法图（FLAG），结合抽象语法树与增强的控制流和数据流语义；使用两阶段图神经网络分析，第一阶段过滤噪声，第二阶段进行高级推理检测漏洞。

### 主要发现

在真实合约上的广泛实验表明，BugSweeper显著优于所有最先进的检测方法。

### 结论

通过消除手工规则需求，BugSweeper提供了强大、自动化和可扩展的智能合约安全解决方案，无需依赖安全专家。

### 翻译

以太坊的快速增长使得快速准确地检测智能合约漏洞变得更为重要。虽然基于机器学习的方法已经显示出一些前景，但许多仍然依赖于领域专家设计的基于规则的预处理方法。基于规则的预处理方法通常会丢弃源代码中的关键上下文，可能导致某些漏洞被忽略，并且对新出现的威胁适应性有限。我们引入了BugSweeper，一个端到端的深度学习框架，直接从源代码检测漏洞而无需手动工程。BugSweeper将每个Solidity函数表示为函数级抽象语法图（FLAG），这是一种结合了抽象语法树（AST）和增强控制流与数据流语义的新颖图。然后，我们的两阶段图神经网络（GNN）分析这些图。第一阶段GNN过滤语法图中的噪声，而第二阶段GNN进行高级推理以检测各种漏洞。在真实合约上的广泛实验表明，BugSweeper显著优于所有最先进的检测方法。通过消除手工规则的需求，我们的方法为保护智能合约提供了强大、自动化和可扩展的解决方案，无需任何安全专家的参与。


### 论文摘要

The rapid growth of Ethereum has made it more important to quickly and accurately detect smart contract vulnerabilities. While machine-learning-based methods have shown some promise, many still rely on rule-based preprocessing designed by domain experts. Rule-based preprocessing methods often discard crucial context from the source code, potentially causing certain vulnerabilities to be overlooked and limiting adaptability to newly emerging threats. We introduce BugSweeper, an end-to-end deep learning framework that detects vulnerabilities directly from the source code without manual engineering. BugSweeper represents each Solidity function as a Function-Level Abstract Syntax Graph (FLAG), a novel graph that combines its Abstract Syntax Tree (AST) with enriched control-flow and data-flow semantics. Then, our two-stage Graph Neural Network (GNN) analyzes these graphs. The first-stage GNN filters noise from the syntax graphs, while the second-stage GNN conducts high-level reasoning to detect diverse vulnerabilities. Extensive experiments on real-world contracts show that BugSweeper significantly outperforms all state-of-the-art detection methods. By removing the need for handcrafted rules, our approach offers a robust, automated, and scalable solution for securing smart contracts without any dependence on security experts.

---

## 29. SVG-T2I: Scaling Up Text-to-Image Latent Diffusion Model Without Variational Autoencoder

**论文链接:** [http://arxiv.org/abs/2512.11749v1](http://arxiv.org/abs/2512.11749v1)

**作者:** Minglei Shi, Haolin Wang, Borui Zhang, Wenzhao Zheng, Bohan Zeng, Ziyang Yuan, Xiaoshi Wu, Yuanxing Zhang, Huan Yang, Xintao Wang, Pengfei Wan, Kun Gai, Jie Zhou, Jiwen Lu

**发布时间:** 2025-12-12

**备注:** Code Repository: https://github.com/KlingTeam/SVG-T2I; Model Weights: https://huggingface.co/KlingTeam/SVG-T2I

### GPT解析

### 总结

该研究提出了SVG-T2I框架，用于在视觉基础模型（VFM）特征域内直接进行高质量文本到图像合成，实现了具有竞争力的性能。

### 背景

视觉基础模型（VFM）表示为视觉理解、感知和生成提供了统一的整合途径，但在VFM表示空间内完全训练大规模文本到图像扩散模型在很大程度上尚未探索。

### 目的

填补这一研究空白，扩展SVG框架，提出SVG-T2I以支持在VFM特征域内直接进行高质量文本到图像合成。

### 方法

利用标准的文本到图像扩散管道，提出SVG-T2I框架，支持在VFM特征域内进行文本到图像合成。

### 主要发现

SVG-T2I实现了具有竞争力的性能，在GenEval上达到0.75，在DPG-Bench上达到85.78，这一性能验证了VFM对于生成任务的内在表示能力。

### 结论

完全开源项目，包括自编码器和生成模型，以及它们的训练、推理、评估管道和预训练权重，以促进表示驱动的视觉生成方面的进一步研究。

### 翻译

基于视觉基础模型（VFM）表示的视觉生成为整合视觉理解、感知和生成提供了一条极具前景的统一途径。尽管有这种潜力，但在VFM表示空间内完全训练大规模文本到图像扩散模型在很大程度上仍未被探索。为了填补这一空白，我们扩展了SVG（用于视觉生成的自监督表示）框架，提出了SVG-T2I，以支持在VFM特征域内直接进行高质量文本到图像合成。通过利用标准的文本到图像扩散管道，SVG-T2I实现了具有竞争力的性能，在GenEval上达到0.75，在DPG-Bench上达到85.78。这一性能验证了VFM对于生成任务的内在表示能力。我们完全开源了该项目，包括自编码器和生成模型，以及它们的训练、推理、评估管道和预训练权重，以促进表示驱动的视觉生成方面的进一步研究。


### 论文摘要

Visual generation grounded in Visual Foundation Model (VFM) representations offers a highly promising unified pathway for integrating visual understanding, perception, and generation. Despite this potential, training large-scale text-to-image diffusion models entirely within the VFM representation space remains largely unexplored. To bridge this gap, we scale the SVG (Self-supervised representations for Visual Generation) framework, proposing SVG-T2I to support high-quality text-to-image synthesis directly in the VFM feature domain. By leveraging a standard text-to-image diffusion pipeline, SVG-T2I achieves competitive performance, reaching 0.75 on GenEval and 85.78 on DPG-Bench. This performance validates the intrinsic representational power of VFMs for generative tasks. We fully open-source the project, including the autoencoder and generation model, together with their training, inference, evaluation pipelines, and pre-trained weights, to facilitate further research in representation-driven visual generation.

---

## 30. Architecting Large Action Models for Human-in-the-Loop Intelligent Robots

**论文链接:** [http://arxiv.org/abs/2512.11620v1](http://arxiv.org/abs/2512.11620v1)

**作者:** Kanisorn Sangchai, Methasit Boonpun, Withawin Kraipetchara, Paulo Garcia

**发布时间:** 2025-12-12

### GPT解析

### 总结

本研究提出了一种构建大行动模型的新方法，通过组合现成基础模型并添加符号包装器，实现了智能机器人的可验证神经符号解决方案，无需大规模端到端训练即可实现高效控制。

### 背景

智能机器人实现需要环境感知、推理和行动的集成。经典AI技术面临可扩展性瓶颈，而大语言模型虽能力强但缺乏控制性、可解释性和可靠性。大行动模型旨在扩展大语言模型以涵盖完整认知周期，但仍存在训练需求大和可靠性问题。

### 目的

构建有能力的大行动模型，提高其控制性、可解释性和可靠性，为智能机器人实现可验证的神经符号解决方案。

### 方法

通过组合现成基础模型构建大行动模型，集成符号包装器及其输出验证；在多模态机器人上实验，将高效感知模型与逻辑驱动核心结合；通过生成PDDL代码驱动行动执行，实现人机循环验证阶段。

### 主要发现

大行动模型能力可通过组合现成基础模型实现，无需大规模端到端训练；通过集成高效感知模型与逻辑驱动核心可实现智能；生成PDDL代码驱动行动执行可引入人机验证阶段，有效缓解行动幻觉。

### 结论

该方法支持从业者在全新行业中设计和开发机器人大行动模型，为确保领域安全必须解决的持续挑战提供了启示。

### 翻译

智能机器人的实现，即自主运行并与其他智能体（人类或人工智能）交互，需要环境感知、推理和行动的整合。为此目的的经典人工智能技术，专注于符号方法，早已在计算和内存成本方面遇到了可扩展性瓶颈。过去十年大语言模型的进展（神经方法）展示了前所未有的能力，但代价是失去了控制性、可解释性和可解释性。大行动模型旨在扩展大语言模型以涵盖完整的感知、推理和行动周期；然而，它们通常需要更全面的训练，并存在相同的可靠性缺陷。在此，我们展示了通过组合现成基础模型构建有能力的大行动模型是可能的，并且它们的控制性、可解释性和可解释性可以通过在其输出中集成符号包装器和相关验证来实现，为智能机器人实现可验证的神经符号解决方案。我们在多模态机器人上的实验表明，大行动模型智能不需要大规模端到端训练，但可以通过将高效的感知模型与逻辑驱动的核心集成来实现。我们发现，通过生成规划域定义语言代码驱动行动执行，可以实现人机循环验证阶段，有效缓解行动幻觉。这些结果可以支持从业者在全新行业中设计和开发机器人大行动模型，并为确保领域安全必须解决的持续挑战提供了启示。


### 论文摘要

The realization of intelligent robots, operating autonomously and interacting with other intelligent agents, human or artificial, requires the integration of environment perception, reasoning, and action. Classic Artificial Intelligence techniques for this purpose, focusing on symbolic approaches, have long-ago hit the scalability wall on compute and memory costs. Advances in Large Language Models in the past decade (neural approaches) have resulted in unprecedented displays of capability, at the cost of control, explainability, and interpretability. Large Action Models aim at extending Large Language Models to encompass the full perception, reasoning, and action cycle; however, they typically require substantially more comprehensive training and suffer from the same deficiencies in reliability. Here, we show it is possible to build competent Large Action Models by composing off-the-shelf foundation models, and that their control, interpretability, and explainability can be effected by incorporating symbolic wrappers and associated verification on their outputs, achieving verifiable neuro-symbolic solutions for intelligent robots. Our experiments on a multi-modal robot demonstrate that Large Action Model intelligence does not require massive end-to-end training, but can be achieved by integrating efficient perception models with a logic-driven core. We find that driving action execution through the generation of Planning Domain Definition Language (PDDL) code enables a human-in-the-loop verification stage that effectively mitigates action hallucinations. These results can support practitioners in the design and development of robotic Large Action Models across novel industries, and shed light on the ongoing challenges that must be addressed to ensure safety in the field.

---

## 31. Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model

**论文链接:** [http://arxiv.org/abs/2512.11582v1](http://arxiv.org/abs/2512.11582v1)

**作者:** Sam Gijsen, Marc-Andre Schulz, Kerstin Ritter

**发布时间:** 2025-12-12

**备注:** Code and pretrained models available at https://github.com/SamGijsen/Brain-Semantoks

### GPT解析

### 总结

Brain-Semantoks是一种专门为学习大脑动态抽象表示而设计的自监督框架，通过语义tokenizer和自蒸馏目标解决了当前fMRI基础模型对噪声敏感的问题，在各种下游任务上表现优异。

### 背景

基础模型在功能磁共振成像(fMRI)时间序列方面的发展对预测与疾病和认知相关的表型有重要意义。然而，当前模型通常使用小脑区域的mask-and-reconstruct目标进行训练，这种对低级别信息的关注导致表示对噪声和时间波动敏感，需要大量微调才能用于下游任务。

### 目的

引入Brain-Semantoks，一个专门为学习大脑动态抽象表示而设计的自监督框架，以解决当前模型对噪声敏感的问题。

### 方法

Brain-Semantoks架构基于两个核心创新：1) 语义tokenizer，将嘈杂的区域信号聚合成代表功能网络的鲁棒token；2) 自蒸馏目标，强制表示在时间上保持稳定。通过新颖的训练课程确保目标稳定，使模型能够从低信噪比时间序列中鲁棒地学习有意义特征。

### 主要发现

学习到的表示在各种下游任务上表现良好，即使只使用线性探针。全面的扩展分析表明，更多未标记的数据可靠地提高了分布外性能，无需领域适应。

### 结论

Brain-Semantoks框架能够学习大脑功能的鲁棒表示，适用于各种下游任务，且随着更多未标记数据的加入性能持续提升。

### 翻译

功能磁共振成像(fMRI)时间序列基础模型的发展对预测与疾病和认知相关的表型具有重要前景。然而，当前模型通常使用小脑区域的mask-and-reconstruct目标进行训练。这种对低级别信息的关注导致表示对噪声和时间波动敏感，需要大量微调才能用于下游任务。我们引入了Brain-Semantoks，一个专门为学习大脑动态抽象表示而设计的自监督框架。其架构建立在两个核心创新之上：一个语义tokenizer，将嘈杂的区域信号聚合成代表功能网络的鲁棒token；以及一个自蒸馏目标，强制表示在时间上保持稳定。我们展示这一目标通过新颖的训练课程得以稳定，确保模型能够从低信噪比时间序列中鲁棒地学习有意义特征。我们证明，学习到的表示在各种下游任务上表现出色，即使只使用线性探针。此外，我们提供了全面的扩展分析，表明更多未标记的数据可靠地提高了分布外性能，无需领域适应。


### 论文摘要

The development of foundation models for functional magnetic resonance imaging (fMRI) time series holds significant promise for predicting phenotypes related to disease and cognition. Current models, however, are often trained using a mask-and-reconstruct objective on small brain regions. This focus on low-level information leads to representations that are sensitive to noise and temporal fluctuations, necessitating extensive fine-tuning for downstream tasks. We introduce Brain-Semantoks, a self-supervised framework designed specifically to learn abstract representations of brain dynamics. Its architecture is built on two core innovations: a semantic tokenizer that aggregates noisy regional signals into robust tokens representing functional networks, and a self-distillation objective that enforces representational stability across time. We show that this objective is stabilized through a novel training curriculum, ensuring the model robustly learns meaningful features from low signal-to-noise time series. We demonstrate that learned representations enable strong performance on a variety of downstream tasks even when only using a linear probe. Furthermore, we provide comprehensive scaling analyses indicating more unlabeled data reliably results in out-of-distribution performance gains without domain adaptation.

---

## 32. 3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation

**论文链接:** [http://arxiv.org/abs/2512.11557v1](http://arxiv.org/abs/2512.11557v1)

**作者:** Zhiguo Lu, Jianwen Lou, Mingjun Ma, Hairong Jin, Youyi Zheng, Kun Zhou

**发布时间:** 2025-12-12

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了3DTeethSAM，一种基于SAM2的3D牙齿分割方法，通过引入三个轻量级可学习模块和可变形全局注意力插件，实现了高精度的3D牙齿分割，在3DTeethSeg基准测试上达到了91.90%的IoU，创造了该领域的新技术水平。

### 背景

3D牙齿分割是数字牙科中的一个关键而具有挑战性的任务，涉及在3D牙科模型中定位牙齿实例并进行语义分类。由于真实世界牙科结构的复杂性，这一任务面临诸多挑战。

### 目的

作者旨在将SAM2（一种用于图像和视频分割的预训练基础模型）适应到3D牙齿数据上，实现高精度的3D牙齿分割。

### 方法

1. 将3D牙齿模型从预定义视角渲染为2D图像；2. 应用SAM2进行2D分割；3. 使用2D-3D投影重建3D结果；4. 引入三个轻量级可学习模块：提示嵌入生成器、掩码细化器和掩码分类器；5. 在SAM2的图像编码器中整合可变形全局注意力插件(DGAP)。

### 主要发现

作者的方法在3DTeethSeg基准测试上取得了91.90%的IoU（高分辨率3D牙齿网格上），建立了该领域的新技术水平。

### 结论

3DTeethSAM通过有效利用SAM2并引入针对性的改进，显著提高了3D牙齿分割的性能，为数字牙科领域提供了先进的解决方案。

### 翻译

3D牙齿分割涉及在3D牙科模型中定位牙齿实例并进行语义分类，是数字牙科中一个关键但具有挑战性的任务，这是由于真实世界牙科结构的复杂性。在本文中，我们提出了3DTeethSAM，这是Segment Anything Model 2 (SAM2)的一种适应版本，专门用于3D牙齿分割。SAM2是一种用于图像和视频分割的预训练基础模型，在各种下游场景中展现出强大的骨干能力。为了将SAM2适应到3D牙齿数据，我们从预定义视角渲染3D牙齿模型的图像，应用SAM2进行2D分割，并使用2D-3D投影重建3D结果。由于SAM2的性能依赖于输入提示，且其初始输出通常存在不足，加之其类别不可知性质，我们引入了三个轻量级可学习模块：(1) 提示嵌入生成器，用于从图像嵌入中导出提示嵌入，以实现精确的掩码解码；(2) 掩码细化器，用于增强SAM2的初始分割结果；(3) 掩码分类器，用于对生成的掩码进行分类。此外，我们在SAM2的图像编码器中整合了可变形全局注意力插件(DGAP)。DGAP提高了分割精度和训练速度。我们的方法已在3DTeethSeg基准测试上得到验证，在高分辨率3D牙齿网格上实现了91.90%的IoU，在该领域建立了新的技术水平。


### 论文摘要

3D teeth segmentation, involving the localization of tooth instances and their semantic categorization in 3D dental models, is a critical yet challenging task in digital dentistry due to the complexity of real-world dentition. In this paper, we propose 3DTeethSAM, an adaptation of the Segment Anything Model 2 (SAM2) for 3D teeth segmentation. SAM2 is a pretrained foundation model for image and video segmentation, demonstrating a strong backbone in various downstream scenarios. To adapt SAM2 for 3D teeth data, we render images of 3D teeth models from predefined views, apply SAM2 for 2D segmentation, and reconstruct 3D results using 2D-3D projections. Since SAM2's performance depends on input prompts and its initial outputs often have deficiencies, and given its class-agnostic nature, we introduce three light-weight learnable modules: (1) a prompt embedding generator to derive prompt embeddings from image embeddings for accurate mask decoding, (2) a mask refiner to enhance SAM2's initial segmentation results, and (3) a mask classifier to categorize the generated masks. Additionally, we incorporate Deformable Global Attention Plugins (DGAP) into SAM2's image encoder. The DGAP enhances both the segmentation accuracy and the speed of the training process. Our method has been validated on the 3DTeethSeg benchmark, achieving an IoU of 91.90% on high-resolution 3D teeth meshes, establishing a new state-of-the-art in the field.

---

## 33. SSL-MedSAM2: A Semi-supervised Medical Image Segmentation Framework Powered by Few-shot Learning of SAM2

**论文链接:** [http://arxiv.org/abs/2512.11548v1](http://arxiv.org/abs/2512.11548v1)

**作者:** Zhendi Gong, Xin Chen

**发布时间:** 2025-12-12

**备注:** Accepted by MICCAI 2025 CARE Challenge, waiting for publication

### GPT解析

### 总结

本文提出了一种名为SSL-MedSAM2的新型半监督学习框架，用于医学图像分割，解决了医学图像标注耗时的问题。

### 背景

尽管基于深度学习的模型在医学图像分割方面取得了成功，但最先进的方法主要依赖全监督学习，需要大规模标注的训练数据集。然而，医学图像标注非常耗时，阻碍了其在临床应用中的推广。

### 目的

开发一种半监督学习策略，减少标注成本，同时保持分割性能。

### 方法

SSL-MedSAM2框架包含两个分支：一是基于预训练大基础模型Segment Anything Model 2 (SAM2)的无训练少样本学习分支TFFS-MedSAM2，用于生成伪标签；二是基于nnUNet的迭代全监督学习分支FSL-nnUNet，用于伪标签优化。

### 主要发现

在MICCAI2025挑战赛CARE-LiSeg（肝脏分割）上，SSL-MedSAM2表现优异。测试集上的平均dice分数：GED4为0.9710，T1 MRI为0.9648；Hausdorff距离分别为20.07和21.97。

### 结论

SSL-MedSAM2在与其他方法的比较中表现出色，有效减少了医学图像分割所需的标注成本。

### 翻译

尽管基于深度学习的模型在医学图像分割方面取得了成功，但大多数最先进的方法执行全监督学习，通常依赖于大规模标注的训练数据集。然而，医学图像标注非常耗时，阻碍了其临床应用。半监督学习(SSL)已成为一种在有有限标注数据进行训练的吸引人的策略，大大减少了标注成本。我们提出了一种新型SSL框架SSL-MedSAM2，它包含一个基于预训练大基础模型Segment Anything Model 2 (SAM2)的无训练少样本学习分支TFFS-MedSAM2，用于生成伪标签，以及一个基于nnUNet的迭代全监督学习分支FSL-nnUNet，用于伪标签优化。在MICCAI2025挑战赛CARE-LiSeg（肝脏分割）上的结果显示，SSL-MedSAM2在其他方法中表现出色。测试集上GED4和T1 MRI的平均dice分数分别为0.9710和0.9648，Hausdorff距离分别为20.07和21.97。代码可通过https://github.com/naisops/SSL-MedSAM2/tree/main获取。


### 论文摘要

Despite the success of deep learning based models in medical image segmentation, most state-of-the-art (SOTA) methods perform fully-supervised learning, which commonly rely on large scale annotated training datasets. However, medical image annotation is highly time-consuming, hindering its clinical applications. Semi-supervised learning (SSL) has been emerged as an appealing strategy in training with limited annotations, largely reducing the labelling cost. We propose a novel SSL framework SSL-MedSAM2, which contains a training-free few-shot learning branch TFFS-MedSAM2 based on the pretrained large foundation model Segment Anything Model 2 (SAM2) for pseudo label generation, and an iterative fully-supervised learning branch FSL-nnUNet based on nnUNet for pseudo label refinement. The results on MICCAI2025 challenge CARE-LiSeg (Liver Segmentation) demonstrate an outstanding performance of SSL-MedSAM2 among other methods. The average dice scores on the test set in GED4 and T1 MRI are 0.9710 and 0.9648 respectively, and the Hausdorff distances are 20.07 and 21.97 respectively. The code is available via https://github.com/naisops/SSL-MedSAM2/tree/main.

---

## 34. On Geometric Understanding and Learned Data Priors in VGGT

**论文链接:** [http://arxiv.org/abs/2512.11508v1](http://arxiv.org/abs/2512.11508v1)

**作者:** Jelena Bratulić, Sudhanshu Mittal, Thomas Brox, Christian Rupprecht

**发布时间:** 2025-12-12

### GPT解析

### 总结

本研究对视觉几何基础Transformer (VGGT)进行了系统分析，探究其内部机制是否隐式包含几何理解，以及它如何平衡几何概念与数据驱动先验。

### 背景

VGGT是一种3D基础模型，可以在单次前向传播中推断相机几何和场景结构，在大型数据集上以监督、单步方式训练。它引发了一个关键问题：它是基于传统多视图方法的几何概念，还是主要依赖学习到的基于外观的数据驱动先验。

### 目的

通过分析VGGT的内部机制，揭示其表示中是否出现了几何理解，探究其如何实现功能，并评估其对学习数据先验的依赖性。

### 方法

通过探测中间特征、分析注意力模式和进行干预，检查模型如何实现其功能；使用空间输入掩码和扰动实验评估其鲁棒性；将其与经典多阶段管道进行比较。

### 主要发现

VGGT在其全局注意力层内隐式执行对应匹配；尽管在没有明确几何约束的情况下进行训练，但它编码了极线几何；评估了其对遮挡、外观变化和相机配置的鲁棒性。

### 结论

VGGT在内部化几何结构的同时使用学习到的数据驱动先验，这为理解现代3D基础模型如何处理几何信息提供了见解。

### 翻译

视觉几何基础Transformer (VGGT)是一种3D基础模型，可以在单次前向传播中推断相机几何和场景结构。在大型数据集上以监督、单步方式训练的VGGT提出了一个关键问题：它是基于传统多视图方法的几何概念构建，还是主要依赖于学习到的基于外观的数据驱动先验？在本工作中，我们对VGGT的内部机制进行了系统分析，以揭示其表示中是否出现了几何理解。通过探测中间特征、分析注意力模式和进行干预，我们检查了模型如何实现其功能。我们的发现表明，VGGT在其全局注意力层内隐式执行对应匹配，并编码了极线几何，尽管它是在没有明确几何约束的情况下进行训练的。我们进一步研究了VGGT对其学习数据先验的依赖性。通过使用空间输入掩码和扰动实验，我们评估了其对遮挡、外观变化和相机配置的鲁棒性，并将其与经典的多阶段管道进行比较。这些见解共同强调了VGGT如何在内部化几何结构的同时使用学习到的数据驱动先验。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要研究VGGT模型是真正理解了几何概念还是仅仅依赖从数据中学到的外观先验。这个问题很重要，因为它关系到现代3D重建模型的可靠性：如果模型真正理解了几何原理，就能在分布外情况下保持鲁棒性；如果仅依赖数据模式，则可能在新场景下失败。这有助于改进模型设计，提高泛化能力和可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从两个视角设计研究：1) 几何视角，通过探测内部表示和干预评估极线几何的因果效应；2) 数据视角，测试模型对输入变化的鲁棒性。他们使用ShapeNet构建受控合成数据集，设计实验探测基本矩阵恢复、分析注意力模式、进行注意力消除干预，并评估模型鲁棒性。研究借鉴了传统多视图几何方法、现有可解释性技术（如探测分类器、网络解剖）和因果分析方法（如激活修补）。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是VGGT在无显式几何约束训练下，隐式地学会了在全局注意力层执行对应点匹配并编码极线几何。实现流程：1) 使用ShapeNet构建受控合成数据集，包括不同物体和相机配置；2) 在各层训练MLP探针预测基本矩阵；3) 分析注意力空间中的对应点匹配；4) 通过注意力消除干预建立因果关系；5) 评估模型对遮挡、光照变化和焦距变化的鲁棒性；6) 比较与传统方法和基于学习方法的性能差异。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点：1) 首次系统性分析VGGT内部几何理解机制；2) 发现模型隐式执行对应点匹配和编码极线几何的能力；3) 建立注意力模式与几何理解间的因果关系；4) 全面评估数据先验作用。相比之前工作，这篇论文专注于3D视觉模型中的几何理解而非判别任务；使用受控合成数据集消除歧义；不仅可视化注意力模式，还建立因果关系；揭示模型处理遮挡时的'幻觉'行为，这可能是不期望的特性。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统性分析揭示了VGGT模型在无显式几何约束训练下能够隐式执行对应点匹配并编码极线几何，同时证明了其数据驱动先验提供了对输入变化的强大鲁棒性，为理解现代3D重建模型的内部工作机制提供了重要见解。'}


### 论文摘要

The Visual Geometry Grounded Transformer (VGGT) is a 3D foundation model that infers camera geometry and scene structure in a single feed-forward pass. Trained in a supervised, single-step fashion on large datasets, VGGT raises a key question: does it build upon geometric concepts like traditional multi-view methods, or does it rely primarily on learned appearance-based data-driven priors? In this work, we conduct a systematic analysis of VGGT's internal mechanisms to uncover whether geometric understanding emerges within its representations. By probing intermediate features, analyzing attention patterns, and performing interventions, we examine how the model implements its functionality. Our findings reveal that VGGT implicitly performs correspondence matching within its global attention layers and encodes epipolar geometry, despite being trained without explicit geometric constraints. We further investigate VGGT's dependence on its learned data priors. Using spatial input masking and perturbation experiments, we assess its robustness to occlusions, appearance variations, and camera configurations, comparing it with classical multi-stage pipelines. Together, these insights highlight how VGGT internalizes geometric structure while using learned data-driven priors.

---

## 35. CADMorph: Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop

**论文链接:** [http://arxiv.org/abs/2512.11480v1](http://arxiv.org/abs/2512.11480v1)

**作者:** Weijian Ma, Shizhao Sun, Ruiyu Wang, Jiang Bian

**发布时间:** 2025-12-12

**备注:** NeurIPS 2025

### GPT解析

### 总结

CADMorph是一种迭代式计划-生成-验证框架，用于几何驱动的参数化CAD编辑，能够在保持原始序列结构、确保编辑语义有效性和维持形状保真度的同时，处理稀缺的编辑数据。

### 背景

CAD模型以参数化构建序列和可见几何形状两种耦合形式编码对象。在迭代设计过程中，对几何形状的调整需要同步编辑底层的参数序列，这被称为几何驱动的参数化CAD编辑。

### 目的

解决几何驱动的参数化CAD编辑中的三大挑战：保持原始序列结构、确保编辑语义有效性、在数据稀缺情况下维持与目标形状的高度相似性。

### 方法

CADMorph框架整合了两个预训练领域特定基础模型：参数到形状(P2S)潜在扩散模型和掩码参数预测(MPP)模型。工作流程包括计划阶段(使用P2S模型确定修改区域)、生成阶段(MPP模型填充语义有效编辑)和验证阶段(P2S模型选择最接近目标形状的候选序列)。

### 主要发现

CADMorph利用预训练先验中的几何意识和设计知识，分别处理结构保持、语义有效性和形状保真度问题。P2S和MPP模型在没有三元组数据的情况下进行训练，绕过了数据稀缺的瓶颈。

### 结论

CADMorph超越了GPT-4o和专门的CAD基线性能，并支持迭代编辑和逆向工程增强等下游应用，为CAD模型的参数化编辑提供了有效解决方案。

### 翻译

计算机辅助设计(CAD)模型以两种耦合形式编码对象：参数化构建序列和其产生的可见几何形状。在迭代设计过程中，对几何形状的调整不可避免地需要同步编辑底层参数序列，这被称为几何驱动的参数化CAD编辑。该任务要求1)保持原始序列的结构，2)确保每个编辑的语义有效性，3)在稀缺的编辑数据三元组情况下保持与目标形状的高度相似性。我们提出了CADMorph，一个迭代式计划-生成-验证框架，在推理过程中协调预训练的领域特定基础模型：参数到形状(P2S)潜在扩散模型和掩码参数预测(MPP)模型。在计划阶段，P2S模型的交叉注意力图确定了需要修改的段并提供编辑掩码。MPP模型然后在生成阶段用语义有效的编辑填充这些掩码。在验证过程中，P2S模型将每个候选序列嵌入形状潜在空间，测量其与目标形状的距离，并选择最接近的一个。这三个阶段利用了预训练先验中固有的几何意识和设计知识，从而分别处理了结构保持、语义有效性和形状保真度问题。此外，P2S和MPP模型都在没有三元组数据的情况下进行训练，绕过了数据稀缺的瓶颈。CADMorph超越了GPT-4o和专门的CAD基线，并支持迭代编辑和逆向工程增强等下游应用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决几何驱动的参数化CAD编辑问题，即根据给定的目标几何形状，修改原有的参数化CAD序列，使其生成的新序列能够渲染出目标几何形状。这个问题在现实中非常重要，因为CAD是连接初始概念和可制造产品的重要桥梁，工程师经常需要调整几何形状以满足模拟反馈、人体工程学或美学要求，而这些调整需要对底层参数序列进行精确修改，这一过程既繁琐又容易出错。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出几何驱动的参数化CAD编辑面临的四个核心挑战：结构保留、语义有效性、形状保真度和数据稀缺。为了解决这些问题，作者借鉴了多个现有工作：利用预训练基础模型中的几何和设计先验知识；采用文本到图像模型中的交叉注意力图分析技术；应用测试时扩展原则，在推理时投入额外计算以提高性能；结合两种互补模型（参数到形状P2S模型和掩码参数预测MPP模型）。这些模型都不依赖于稀缺的三元组数据，而是利用现有CAD数据进行训练。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "CADMorph的核心思想是一个迭代的'计划-生成-验证'框架，利用两种互补预训练模型（P2S和MPP）之间的相互作用，逐步将原始参数序列转换为能够重现目标几何形状的序列。整体流程包括：1）计划阶段：分析P2S模型的交叉注意力图定位需要修改的片段，用[mask]标记替换；2）生成阶段：MPP模型多次填充[mask]标记，产生候选序列；3）验证阶段：将候选序列投影到P2S模型的潜在空间，选择与目标形状最接近的序列作为下一次迭代的输入。循环此过程直到收敛或达到最大迭代次数。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1）正式定义了'几何驱动的参数化CAD编辑'任务；2）利用两种互补模型（P2S和MPP）绕过数据稀缺瓶颈；3）引入迭代计划-生成-验证框架，结合交叉注意力分析、掩码填充和潜在空间验证。相比之前工作，CADMorph同时尊重原始参数序列中的设计者意图（与逆向工程方法不同）并利用目标几何形状的视觉线索（与之前的编辑方法不同），解决了现有方法无法完全解决的问题。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CADMorph通过一个迭代的计划-生成-验证框架，结合两种互补的预训练模型，实现了从原始参数化CAD序列到能够精确生成目标几何形状的更新序列的高效转换，解决了几何驱动的参数化CAD编辑这一重要但研究不足的问题。'}


### 论文摘要

A Computer-Aided Design (CAD) model encodes an object in two coupled forms: a parametric construction sequence and its resulting visible geometric shape. During iterative design, adjustments to the geometric shape inevitably require synchronized edits to the underlying parametric sequence, called geometry-driven parametric CAD editing. The task calls for 1) preserving the original sequence's structure, 2) ensuring each edit's semantic validity, and 3) maintaining high shape fidelity to the target shape, all under scarce editing data triplets. We present CADMorph, an iterative plan-generate-verify framework that orchestrates pretrained domain-specific foundation models during inference: a parameter-to-shape (P2S) latent diffusion model and a masked-parameter-prediction (MPP) model. In the planning stage, cross-attention maps from the P2S model pinpoint the segments that need modification and offer editing masks. The MPP model then infills these masks with semantically valid edits in the generation stage. During verification, the P2S model embeds each candidate sequence in shape-latent space, measures its distance to the target shape, and selects the closest one. The three stages leverage the inherent geometric consciousness and design knowledge in pretrained priors, and thus tackle structure preservation, semantic validity, and shape fidelity respectively. Besides, both P2S and MPP models are trained without triplet data, bypassing the data-scarcity bottleneck. CADMorph surpasses GPT-4o and specialized CAD baselines, and supports downstream applications such as iterative editing and reverse-engineering enhancement.

---

## 36. Benchmarking the Generality of Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2512.11315v1](http://arxiv.org/abs/2512.11315v1)

**作者:** Pranav Guruprasad, Sudipta Chowdhury, Harsh Sikka, Mridul Sharma, Helen Lu, Sean Rivera, Aryan Khurana, Hangliang Ren, Yangyue Wang

**发布时间:** 2025-12-12

**备注:** 23 pages, 7 figures, and 1 table

### GPT解析

### 总结

论文提出了MultiNet v1.0基准测试，用于评估视觉语言模型和视觉语言动作模型的跨领域通用性，评估结果显示当前模型在跨领域任务上表现不佳。

### 背景

全能多模态智能体应统一感知、语言和控制能力并在多样化领域稳健运行，但当前评估实践分散在孤立的基准测试中，难以评估基础模型是否真正超越训练分布。

### 目的

引入MultiNet v1.0统一基准，衡量视觉语言模型(VLMs)和视觉语言动作模型(VLAs)在六个基础能力领域的跨领域通用性：视觉定位、空间推理、工具使用、物理常识、多智能体协调和连续机器人控制。

### 方法

评估了GPT-5、Pi0和Magma等模型在MultiNet v1.0基准测试中的表现，测试其在六个基础能力领域的跨领域能力。

### 主要发现

没有模型表现出一致的通用性；所有模型在未见领域、不熟悉模态或跨领域任务转移上性能显著下降；失败表现为模态错位、输出格式不稳定和领域转移下的灾难性知识退化。

### 结论

全能智能体的期望与当前基础模型的实际能力之间存在持续差距；MultiNet v1.0为诊断这些差距和指导未来全能智能体发展提供了标准化评估基础；代码、数据和排行榜已公开可用。

### 翻译

全能多模态智能体有望统一感知、语言和控制能力，在多样化的真实领域中稳健运行。然而，当前的评估实践仍然分散在孤立的基准测试中，难以评估当今的基础模型是否真正超越了它们的训练分布。我们引入了MultiNet v1.0，这是一个统一的基准，用于衡量视觉语言模型(VLMs)和视觉语言动作模型(VLAs)在六个基础能力领域的跨领域通用性：视觉定位、空间推理、工具使用、物理常识、多智能体协调和连续机器人控制。评估GPT-5、Pi0和Magma后，我们发现没有模型表现出一致的通用性。所有模型在未见过的领域、不熟悉的模态或跨领域任务转移上都表现出显著的性能下降，尽管它们在训练分布内表现良好。这些失败表现为模态错位、输出格式不稳定以及在领域转移下的灾难性知识退化。我们的研究揭示了全能智能体的期望与当前基础模型的实际能力之间存在的持续差距。MultiNet v1.0为诊断这些差距和指导未来全能智能体的发展提供了标准化的评估基础。代码、数据和排行榜已公开提供。


### 论文摘要

Generalist multimodal agents are expected to unify perception, language, and control - operating robustly across diverse real world domains. However, current evaluation practices remain fragmented across isolated benchmarks, making it difficult to assess whether today's foundation models truly generalize beyond their training distributions. We introduce MultiNet v1.0, a unified benchmark for measuring the cross domain generality of vision language models (VLMs) and vision language action models (VLAs) across six foundational capability regimes. Visual grounding, spatial reasoning, tool use, physical commonsense, multi agent coordination, and continuous robot control. Evaluating GPT 5, Pi0, and Magma, we find that no model demonstrates consistent generality. All exhibit substantial degradation on unseen domains, unfamiliar modalities, or cross domain task shifts despite strong performance within their training distributions.These failures manifest as modality misalignment, output format instability, and catastrophic knowledge degradation under domain transfer.Our findings reveal a persistent gap between the aspiration of generalist intelligence and the actual capabilities of current foundation models.MultiNet v1.0 provides a standardized evaluation substrate for diagnosing these gaps and guiding the development of future generalist agents.Code, data, and leaderboards are publicly available.

---

## 37. VFMF: World Modeling by Forecasting Vision Foundation Model Features

**论文链接:** [http://arxiv.org/abs/2512.11225v1](http://arxiv.org/abs/2512.11225v1)

**作者:** Gabrijel Boduljak, Yushi Lan, Christian Rupprecht, Andrea Vedaldi

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出了一种基于视觉基础模型(VFM)特征的生成预测方法，通过自回归流匹配在特征空间进行世界状态预测，解决了传统像素预测计算量大和确定性回归无法捕捉不确定性的问题。

### 背景

预测部分观察结果对世界建模至关重要。现有方法要么通过图像表示世界并简化为随机视频生成(计算量大且不直接实用)，要么使用VFM特征进行确定性回归(无法捕捉不确定性)。

### 目的

开发一种既能保持计算效率又能捕捉不确定性的预测方法，解决确定性回归平均多个可能未来而损害预测准确性的关键限制。

### 方法

在VFM特征空间执行自回归流匹配，将VFM特征编码为适合扩散的紧凑潜在空间，这种潜在表示比基于PCA的替代方案更有效地保留信息，并将潜在预测解码为多种输出模态。

### 主要发现

在匹配的架构和计算条件下，该方法在所有模态上产生比回归更清晰、更准确的预测，随机条件生成VFM特征为未来世界模型提供了有前景且可扩展的基础。

### 结论

基于VFM特征的随机条件生成方法结合了像素预测的高实用性和确定性回归的计算效率，同时解决了不确定性捕捉的问题，为世界建模提供了新方向。

### 翻译

从部分观察结果进行预测是世界建模的核心。许多最近的方法通过图像表示世界，并将预测简化为随机视频生成。尽管这类方法在真实感和视觉保真度方面表现出色，但预测像素计算量大，并且在许多应用中不直接有用，因为它需要将RGB转换为对决策有用的信号。另一种方法使用视觉基础模型(VFM)的特征作为世界表示，执行确定性回归来预测未来世界状态。这些特征可以直接转换为可操作的信号，如语义分割和深度，同时保持计算效率。然而，确定性回归对多个可能的未来进行平均，由于未能捕捉不确定性而损害了预测准确性。为了解决这一关键限制，我们引入了一个在VFM特征空间执行自回归流匹配的生成预测器。我们的关键见解是，在这个空间中进行生成建模需要将VFM特征编码为适合扩散的紧凑潜在空间。我们证明，与先前使用的基于PCA的替代方案相比，这种潜在空间在预测和其他应用(如图像生成)中能更有效地保留信息。我们的潜在预测可以轻松解码为多种有用且可解释的输出模态：语义分割、深度、表面法线，甚至是RGB。在匹配的架构和计算条件下，我们的方法在所有模态上产生比回归更清晰、更准确的预测。我们的结果表明，VFM特征的随机条件生成为未来的世界模型提供了有前景且可扩展的基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决世界建模中的未来状态预测问题，特别是在部分观察条件下如何准确预测场景的未来状态。这个问题在现实世界中至关重要，因为从自动驾驶到机器人控制，再到游戏和虚拟现实，各种AI系统都需要准确预测环境变化来做出合理决策。现有方法要么通过预测像素来生成视频（计算密集且不直接有用），要么使用视觉基础模型特征进行确定性回归（无法捕捉不确定性，导致模糊预测），都无法很好地满足实际应用需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有世界建模方法的优缺点：像素级视频生成方法虽然视觉保真度高但计算密集且难以直接用于决策，而基于VFM特征的方法更高效但确定性回归无法捕捉不确定性。关键洞察是需要在VFM特征空间中实现生成式预测。作者借鉴了多个现有工作：视觉基础模型（如DINO）用于特征提取，扩散模型和流动匹配技术用于生成式预测，VAE思想用于特征压缩，以及多模态解码技术。创新点在于设计了专门用于VFM特征压缩的VAE，并在压缩的潜在空间中使用自回归流动匹配进行预测，实现不确定性感知的预测。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在视觉基础模型(VFM)的特征空间中进行生成式世界建模，而不是像素空间或确定性特征空间。整体流程包括：1)使用预训练的视觉基础模型(如DINO)从输入图像中提取特征；2)使用专门设计的VAE将高维VFM特征压缩到低维潜在空间；3)在压缩的潜在空间中使用自回归流动匹配进行预测，根据上下文长度调整不确定性；4)将预测的特征解码为多种有用模态，包括语义分割、深度图、表面法线和RGB图像。这种方法既保留了VFM特征的语义和几何信息，又通过生成式方法捕捉了未来状态的不确定性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次在VFM特征空间中实现生成式(非确定性)预测，显式建模未来状态的不确定性；2)提出专门设计的VFM特征VAE，比传统PCA更好地保留信息；3)将流动匹配技术应用于VFM特征预测，处理不同长度上下文；4)单一模型可生成多种下游任务表示。相比之前的工作，与像素级视频生成方法相比，计算效率更高且输出更直接有用；与确定性VFM特征预测方法相比，能捕捉不确定性且在短上下文下表现更好；与PCA压缩方法相比，保留更多信息且生成质量更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种在视觉基础模型特征空间中进行生成式世界建模的新方法，通过专门设计的VAE压缩和自回归流动匹配，实现了对未来状态的不确定性感知预测，并能解码为多种有用的下游任务表示，显著提升了预测的准确性和实用性。'}


### 论文摘要

Forecasting from partial observations is central to world modeling. Many recent methods represent the world through images, and reduce forecasting to stochastic video generation. Although such methods excel at realism and visual fidelity, predicting pixels is computationally intensive and not directly useful in many applications, as it requires translating RGB into signals useful for decision making. An alternative approach uses features from vision foundation models (VFMs) as world representations, performing deterministic regression to predict future world states. These features can be directly translated into actionable signals such as semantic segmentation and depth, while remaining computationally efficient. However, deterministic regression averages over multiple plausible futures, undermining forecast accuracy by failing to capture uncertainty. To address this crucial limitation, we introduce a generative forecaster that performs autoregressive flow matching in VFM feature space. Our key insight is that generative modeling in this space requires encoding VFM features into a compact latent space suitable for diffusion. We show that this latent space preserves information more effectively than previously used PCA-based alternatives, both for forecasting and other applications, such as image generation. Our latent predictions can be easily decoded into multiple useful and interpretable output modalities: semantic segmentation, depth, surface normals, and even RGB. With matched architecture and compute, our method produces sharper and more accurate predictions than regression across all modalities. Our results suggest that stochastic conditional generation of VFM features offers a promising and scalable foundation for future world models.

---

## 38. Seeing to Act, Prompting to Specify: A Bayesian Factorization of Vision Language Action Policy

**论文链接:** [http://arxiv.org/abs/2512.11218v1](http://arxiv.org/abs/2512.11218v1)

**作者:** Kechun Xu, Zhenjie Zhu, Anzhe Chen, Shuqi Zhao, Qing Huang, Yifei Yang, Haojian Lu, Rong Xiong, Masayoshi Tomizuka, Yue Wang

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文提出了一种名为BayesVLA的贝叶斯分解方法，解决了视觉-语言-动作(VLA)模型在微调过程中灾难性遗忘的问题，提高了模型对未见过的指令、物体和环境的泛化能力。

### 背景

在视觉-语言-动作(VLA)模型中追求分布外泛化能力时，微调过程常导致视觉语言模型(VLM)主干网络发生灾难性遗忘。现有协同训练方法虽有帮助，但需要经验调整和额外数据开销。

### 目的

解决VLA数据集中的模态不平衡问题，即语言多样性远低于视觉和动作多样性，从而提高模型的分布外泛化能力。

### 方法

提出BayesVLA，一种贝叶斯分解方法，将策略分解为视觉-动作先验(支持'看到即行动')和语言条件似然(实现'提示即指定')，并融入接触前和接触后阶段以更好利用预训练基础模型。

### 主要发现

信息论分析验证了BayesVLA在减轻捷径学习方面的有效性，实验表明其在未见过的指令、物体和环境方面优于现有方法。

### 结论

BayesVLA通过内在的贝叶斯分解方法解决了模态不平衡问题，提高了泛化能力，无需依赖外部数据或复杂调整。

### 翻译

在视觉-语言-动作(VLA)模型中追求分布外泛化的过程中，微调时视觉语言模型(VLM)主干网络的灾难性遗忘常常成为阻碍。虽然使用外部推理数据进行协同训练有所帮助，但这需要经验调整和与数据相关的额外开销。除了这些外部依赖，我们在VLA数据集中识别出一个固有原因：模态不平衡，即语言多样性远低于视觉和动作多样性。这种不平衡导致模型倾向于视觉捷径和语言遗忘。为解决这一问题，我们引入了BayesVLA，一种贝叶斯分解方法，将策略分解为视觉-动作先验(支持'看到即行动')和语言条件似然(实现'提示即指定')。这内在地保留了泛化能力并促进了指令遵循。我们还进一步结合了接触前和接触后阶段，以更好地利用预训练的基础模型。信息论分析正式验证了我们在减轻捷径学习方面的有效性。大量实验表明，与现有方法相比，BayesVLA在未见过的指令、物体和环境方面具有更好的泛化能力。项目页面可在以下网址获取：https://xukechun.github.io/papers/BayesVLA。


### 论文摘要

The pursuit of out-of-distribution generalization in Vision-Language-Action (VLA) models is often hindered by catastrophic forgetting of the Vision-Language Model (VLM) backbone during fine-tuning. While co-training with external reasoning data helps, it requires experienced tuning and data-related overhead. Beyond such external dependencies, we identify an intrinsic cause within VLA datasets: modality imbalance, where language diversity is much lower than visual and action diversity. This imbalance biases the model toward visual shortcuts and language forgetting. To address this, we introduce BayesVLA, a Bayesian factorization that decomposes the policy into a visual-action prior, supporting seeing-to-act, and a language-conditioned likelihood, enabling prompt-to-specify. This inherently preserves generalization and promotes instruction following. We further incorporate pre- and post-contact phases to better leverage pre-trained foundation models. Information-theoretic analysis formally validates our effectiveness in mitigating shortcut learning. Extensive experiments show superior generalization to unseen instructions, objects, and environments compared to existing methods. Project page is available at: https://xukechun.github.io/papers/BayesVLA.

---

## 39. Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching

**论文链接:** [http://arxiv.org/abs/2512.11130v1](http://arxiv.org/abs/2512.11130v1)

**作者:** Bowen Wen, Shaurya Dewan, Stan Birchfield

**发布时间:** 2025-12-11

### GPT解析

### 总结

Fast-FoundationStereo是一种新型立体视觉架构，首次实现了强大零样本泛化能力和实时帧率的结合，运行速度比FoundationStereo快10倍以上，同时保持相近的零样本准确率。

### 背景

立体基础模型具有强大的零样本泛化能力但计算成本高，不适合实时应用；而高效立体架构为了速度牺牲了鲁棒性，且需要昂贵的领域特定微调。

### 目的

弥合高性能立体模型与实时应用之间的差距，开发一种既能保持强大零样本泛化能力又能实现实时帧率的架构。

### 方法

提出Fast-FoundationStereo架构，采用分而治之的加速策略：1)知识蒸馏压缩混合骨干网络；2)分块神经架构搜索优化代价滤波设计；3)结构化剪枝消除迭代细化冗余；并引入自动伪标记流程筛选140万野外立体图像对补充训练数据。

### 主要发现

所得模型运行速度比FoundationStereo快10倍以上，同时几乎匹配其零样本准确率。

### 结论

Fast-FoundationStereo在实时方法中建立了新的最先进水平。

### 翻译

立体基础模型实现了强大的零样本泛化能力，但计算成本高，不适合实时应用。另一方面，高效的立体架构为了速度牺牲了鲁棒性，并且需要昂贵的领域特定微调。为了弥合这一差距，我们提出了Fast-FoundationStereo，这是一个架构家族，首次实现了强大零样本泛化能力和实时帧率。我们采用分而治之的加速策略，包含三个组件：(1)知识蒸馏将混合骨干网络压缩为单个高效学生模型；(2)分块神经架构搜索在延迟预算下自动发现最优代价滤波设计，指数级降低搜索复杂度；(3)结构化剪枝消除迭代细化模块中的冗余。此外，我们引入了自动伪标记流程，用于筛选140万野外立体图像对，以补充合成训练数据并促进知识蒸馏。所得模型的运行速度比FoundationStereo快10倍以上，同时几乎匹配其零样本准确率，因此在实时方法中建立了新的最先进水平。项目页面：https://nvlabs.github.io/Fast-FoundationStereo/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决立体匹配领域中的一个关键矛盾：如何在保持强大的零样本泛化能力的同时实现实时性能。这个问题很重要，因为精确的3D重建对机器人技术和增强现实等应用至关重要，而实时性能对于自动驾驶、机器人导航等实际应用必不可少。目前研究分裂为追求零样本泛化和追求实时性能两个方向，缺乏兼顾两者的方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者基于FoundationStereo这个强大的基础模型，采用'分而治之'的加速策略。他们首先将FoundationStereo的三个主要组件（特征提取、成本过滤和视差细化）分开考虑，然后针对每个组件设计专门的加速策略：特征提取使用知识蒸馏，成本过滤使用分块神经架构搜索，视差细化使用结构化剪枝。作者确实借鉴了现有工作，包括知识蒸馏、神经架构搜索、结构化剪枝和伪标签技术，但针对立体匹配任务进行了专门改进和创新。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过'分而治之'的加速策略，在不牺牲零样本泛化能力的前提下，将计算密集型的FoundationStereo模型加速到实时性能。整体流程包括：1)特征提取加速：使用知识蒸馏压缩混合特征主干；2)成本过滤加速：将模块划分为块，独立训练候选设计，组合搜索找到最优组合；3)视差细化加速：构建依赖图，评估参数重要性，剪枝冗余参数并重训练；4)数据增强：开发自动伪标签管道处理真实世界数据；5)最终集成：将加速组件组合成端到端模型进行零样本推理。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个兼顾零样本泛化和实时性能的立体匹配架构；2)分而治之的加速策略；3)分块知识蒸馏方法显著降低搜索复杂度；4)自动伪标签管道利用大规模真实世界数据。相比之前的工作，Fast-FoundationStereo比现有基础模型快10倍以上同时保持相近准确性，比其他实时方法有更强的泛化能力且无需针对特定领域微调。方法上结合了多种技术并针对立体匹配任务专门设计，数据利用上更注重真实世界数据的增强作用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Fast-FoundationStereo通过分而治之的加速策略，首次实现了强大的零样本泛化能力和实时性能的立体匹配模型，比现有基础模型快10倍以上同时保持相近准确性，无需针对特定领域进行微调。'}


### 论文摘要

Stereo foundation models achieve strong zero-shot generalization but remain computationally prohibitive for real-time applications. Efficient stereo architectures, on the other hand, sacrifice robustness for speed and require costly per-domain fine-tuning. To bridge this gap, we present Fast-FoundationStereo, a family of architectures that achieve, for the first time, strong zero-shot generalization at real-time frame rate. We employ a divide-and-conquer acceleration strategy with three components: (1) knowledge distillation to compress the hybrid backbone into a single efficient student; (2) blockwise neural architecture search for automatically discovering optimal cost filtering designs under latency budgets, reducing search complexity exponentially; and (3) structured pruning for eliminating redundancy in the iterative refinement module. Furthermore, we introduce an automatic pseudo-labeling pipeline used to curate 1.4M in-the-wild stereo pairs to supplement synthetic training data and facilitate knowledge distillation. The resulting model can run over 10x faster than FoundationStereo while closely matching its zero-shot accuracy, thus establishing a new state-of-the-art among real-time methods. Project page: https://nvlabs.github.io/Fast-FoundationStereo/

---

## 40. Information-driven Fusion of Pathology Foundation Models for Enhanced Disease Characterization

**论文链接:** [http://arxiv.org/abs/2512.11104v1](http://arxiv.org/abs/2512.11104v1)

**作者:** Brennan Flannery, Thomas DeSilvio, Jane Nguyen, Satish E. Viswanath

**发布时间:** 2025-12-11

**备注:** 29 Pages, 10 figures

### GPT解析

### 总结

本研究提出了一种信息驱动的智能融合策略，用于整合多个病理学基础模型为统一表示，并在三种癌症的分级和分期中验证其有效性。智能融合方法优于单一模型和简单融合方案，能够提高预测性能和模型可解释性。

### 背景

基础模型在各种病理学任务中表现出强大性能，但人们对这些模型在嵌入空间中的互补性、冗余性或特征生物学解释的理解仍然有限。

### 目的

开发一种信息驱动的智能融合策略，整合多个病理学基础模型，并系统评估其在癌症分级和分期任务中的性能。

### 方法

使用肾脏癌(519张)、前列腺癌(490张)和直肠癌(200张)的H&E全幻灯片图像，考虑了片块级别模型(Conch v1.5, MUSK等)和幻灯片级别模型(TITAN, CHIEF等)。评估了三种融合方案：多数投票集成、简单特征连接和基于相关性引导修剪冗余特征的智能融合。

### 主要发现

智能融合在所有三种癌症中都带来了一致的分类性能提升；全局相似性指标显示模型嵌入空间存在大量对齐但局部一致性较低，表明模型间存在互补信息；智能融合使注意力更集中在肿瘤区域。

### 结论

智能的、相关性引导的基础模型融合可以产生紧凑的、任务定制化的表示，在下游计算病理学任务中提高预测性能和可解释性。

### 翻译

基础模型已展示出在各种病理学任务中的强大性能。尽管基础模型的预训练目标存在相似性，但人们对它们在嵌入空间中的互补性、冗余性或特征的生物学解释仍然有限。本研究提出了一种信息驱动的智能融合策略，用于将多个病理学基础模型整合为统一表示，并系统评估了其在三种不同疾病的癌症分级和分期中的性能。研究使用了来自三种癌症的诊断性H&E全幻灯片图像，考虑了多种片块级别和幻灯片级别的基础模型，并评估了三种融合方案。结果显示智能融合方法在性能上优于其他方法，能够产生更紧凑、任务定制化的表示，提高预测性能和可解释性。


### 论文摘要

Foundation models (FMs) have demonstrated strong performance across diverse pathology tasks. While there are similarities in the pre-training objectives of FMs, there is still limited understanding of their complementarity, redundancy in embedding spaces, or biological interpretation of features. In this study, we propose an information-driven, intelligent fusion strategy for integrating multiple pathology FMs into a unified representation and systematically evaluate its performance for cancer grading and staging across three distinct diseases. Diagnostic H&E whole-slide images from kidney (519 slides), prostate (490 slides), and rectal (200 slides) cancers were dichotomized into low versus high grade or stage. Both tile-level FMs (Conch v1.5, MUSK, Virchow2, H-Optimus1, Prov-Gigapath) and slide-level FMs (TITAN, CHIEF, MADELEINE) were considered to train downstream classifiers. We then evaluated three FM fusion schemes at both tile and slide levels: majority-vote ensembling, naive feature concatenation, and intelligent fusion based on correlation-guided pruning of redundant features. Under patient-stratified cross-validation with hold-out testing, intelligent fusion of tile-level embeddings yielded consistent gains in classification performance across all three cancers compared with the best single FMs and naive fusion. Global similarity metrics revealed substantial alignment of FM embedding spaces, contrasted by lower local neighborhood agreement, indicating complementary fine-grained information across FMs. Attention maps showed that intelligent fusion yielded concentrated attention on tumor regions while reducing spurious focus on benign regions. Our findings suggest that intelligent, correlation-guided fusion of pathology FMs can yield compact, task-tailored representations that enhance both predictive performance and interpretability in downstream computational pathology tasks.

---

## 41. Vision-Language Models for Infrared Industrial Sensing in Additive Manufacturing Scene Description

**论文链接:** [http://arxiv.org/abs/2512.11098v1](http://arxiv.org/abs/2512.11098v1)

**作者:** Nazanin Mahjourian, Vinh Nguyen

**发布时间:** 2025-12-11

### GPT解析

### 总结

本文介绍了VLM-IRIS，一个用于红外工业传感的视觉-语言模型零样本框架，通过预处理红外图像使其兼容基于CLIP的编码器，实现了在红外数据上的高精度零样本预测，无需重新训练模型。

### 背景

许多制造环境在低光照条件下或封闭机器内运行，传统视觉系统难以应对。红外相机在这些环境中提供互补优势，但监督式AI系统需要大量标记数据，而当前视觉-语言基础模型无法理解红外数据，因为它们是在RGB数据上训练的。

### 目的

引入VLM-IRIS框架，使视觉-语言模型能够适应红外数据，实现零样本学习在红外工业传感中的应用。

### 方法

通过预处理由FLIR Boson传感器捕获的红外图像，将其转换为适合基于CLIP的编码器的RGB兼容输入。在3D打印机床上进行工件存在的零样本检测，将红外图像转换为magma表示，并使用CLIP ViT-B/32编码器应用质心提示集成。

### 主要发现

在红外图像上实现了高精度，无需任何模型重新训练。所提出的对VLMs的改进可以有效地扩展到热成像应用中。

### 结论

所提出的改进可以有效地扩展到热成像应用中，用于无标签监控。

### 翻译

许多制造环境在低光照条件下或封闭机器内运行，传统视觉系统难以应对。红外相机在这些环境中提供互补优势。同时，监督式AI系统需要大量标记数据，这使得零样本学习框架对于包括红外相机在内的应用更实用。视觉-语言基础模型(VLMs)的最新发展为基于图像-文本表示的零样本预测提供了新途径。然而，当前的VLMs无法理解红外相机数据，因为它们是在RGB数据上训练的。本文介绍了VLM-IRIS(用于红外工业传感的视觉-语言模型)，这是一个零样本框架，通过预处理由FLIR Boson传感器捕获的红外图像，将其转换为适合基于CLIP的编码器的RGB兼容输入，从而使VLMs能够适应红外数据。我们在3D打印机床上演示了工件存在的零样本检测，其中构建板和工件之间的温度差异使该任务非常适合热成像。VLM-IRIS将红外图像转换为magma表示，并使用CLIP ViT-B/32编码器应用质心提示集成，在红外图像上实现了高精度，无需任何模型重新训练。这些发现表明，所提出的对VLMs的改进可以有效地扩展到热成像应用中，用于无标签监控。


### 论文摘要

Many manufacturing environments operate in low-light conditions or within enclosed machines where conventional vision systems struggle. Infrared cameras provide complementary advantages in such environments. Simultaneously, supervised AI systems require large labeled datasets, which makes zero-shot learning frameworks more practical for applications including infrared cameras. Recent advances in vision-language foundation models (VLMs) offer a new path in zero-shot predictions from paired image-text representations. However, current VLMs cannot understand infrared camera data since they are trained on RGB data. This work introduces VLM-IRIS (Vision-Language Models for InfraRed Industrial Sensing), a zero-shot framework that adapts VLMs to infrared data by preprocessing infrared images captured by a FLIR Boson sensor into RGB-compatible inputs suitable for CLIP-based encoders. We demonstrate zero-shot workpiece presence detection on a 3D printer bed where temperature differences between the build plate and workpieces make the task well-suited for thermal imaging. VLM-IRIS converts the infrared images to magma representation and applies centroid prompt ensembling with a CLIP ViT-B/32 encoder to achieve high accuracy on infrared images without any model retraining. These findings demonstrate that the proposed improvements to VLMs can be effectively extended to thermal applications for label-free monitoring.

---

## 42. A probabilistic foundation model for crystal structure denoising, phase classification, and order parameters

**论文链接:** [http://arxiv.org/abs/2512.11077v1](http://arxiv.org/abs/2512.11077v1)

**作者:** Hyuna Kwon, Babak Sadigh, Sebastien Hamel, Vincenzo Lordi, John Klepeis, Fei Zhou

**发布时间:** 2025-12-11

### GPT解析

### 总结

该研究提出了一种基于对数概率的基础模型，用于统一处理原子模拟数据中的去噪、相分类和序参数提取，解决了现有方法在通用性、鲁棒性和可解释性方面的局限性。

### 背景

原子模拟产生大量嘈杂的结构数据，但提取相标签、序参数和缺陷信息仍具挑战性，需要一种通用、鲁棒且可解释的方法。

### 目的

开发一种能够统一处理去噪、相分类和序参数提取的概率框架方法，克服现有工具的局限性。

### 方法

重用映射到AFLOW原型的晶体结构上的MACE-MP基础原子间势，训练模型预测每个原子、每个相的对数概率值，并聚合为全局对数密度，通过梯度上升实现去噪，通过argmax获取相标签，利用对数概率值作为连续且对缺陷敏感的序参数。

### 主要发现

该方法在数百种原型上表现出通用性，在强热无序和缺陷引起的无序条件下保持鲁棒性，并能准确处理冰多形体、冰-水界面和冲击压缩的钛等复杂系统。

### 结论

所提出的对数概率基础模型为原子模拟数据的结构分析提供了更强大、更灵活的工具，能够同时处理多种分析任务并提供概率性解释。

### 翻译

原子模拟产生大量嘈杂的结构数据，但以通用、鲁棒且可解释的方式提取相标签、序参数和缺陷信息仍然具有挑战性。现有的PTM和CNA等工具仅限于少数手工制作的晶格(如FCC/BCC/HCP)，在强热无序或缺陷条件下性能下降，并且产生基于模板的硬标签，没有提供原子级别的概率或置信度分数。我们在此引入一种对数概率基础模型，在单一概率框架内统一了去噪、相分类和序参数提取。我们在映射到AFLOW原型的晶体结构上重用MACE-MP基础原子间势，训练它预测每个原子、每个相的对数概率l，并将它们聚合为全局对数密度log P_θ(r)，其梯度定义了一个保守的得分场。去噪对应于在这个学习到的对数密度上的梯度上升，相标签来自于argmax_c l_ac，而l值则作为连续的、对缺陷敏感且可解释的序参数，量化了与理想相的欧几里得距离。我们在数百种原型上证明了其通用性，在强热无序和缺陷引起的无序条件下表现出鲁棒性，并能准确处理冰多形体、冰-水界面和冲击压缩的钛等复杂系统。


### 论文摘要

Atomistic simulations generate large volumes of noisy structural data, but extracting phase labels, order parameters (OPs), and defect information in a way that is universal, robust, and interpretable remains challenging. Existing tools such as PTM and CNA are restricted to a small set of hand-crafted lattices (e.g.\ FCC/BCC/HCP), degrade under strong thermal disorder or defects, and produce hard, template-based labels without per-atom probability or confidence scores. Here we introduce a log-probability foundation model that unifies denoising, phase classification, and OP extraction within a single probabilistic framework. We reuse the MACE-MP foundation interatomic potential on crystal structures mapped to AFLOW prototypes, training it to predict per-atom, per-phase logits $l$ and to aggregate them into a global log-density $\log \hat{P}_θ(\boldsymbol{r})$ whose gradient defines a conservative score field. Denoising corresponds to gradient ascent on this learned log-density, phase labels follow from $\arg\max_c l_{ac}$, and the $l$ values act as continuous, defect-sensitive and interpretable OPs quantifying the Euclidean distance to ideal phases. We demonstrate universality across hundreds of prototypes, robustness under strong thermal and defect-induced disorder, and accurate treatment of complex systems such as ice polymorphs, ice--water interfaces, and shock-compressed Ti.

---

## 43. KathDB: Explainable Multimodal Database Management System with Human-AI Collaboration

**论文链接:** [http://arxiv.org/abs/2512.11067v1](http://arxiv.org/abs/2512.11067v1)

**作者:** Guorui Xiao, Enhao Zhang, Nicole Sullivan, Will Hansen, Magdalena Balazinska

**发布时间:** 2025-12-11

### GPT解析

### 总结

KathDB系统结合关系语义与基础模型的多模态处理能力，通过人机交互渠道在查询解析、执行和结果解释过程中，使用户能够跨数据模态迭代获得可解释的答案。

### 背景

传统DBMS执行SQL查询具有强大语义保证和高级优化，但编写复杂SQL困难且仅处理结构化表格；当代多模态系统要么要求用户手动使用机器学习UDF，要么完全依赖黑盒LLM，牺牲了可用性或可解释性。

### 目的

提出KathDB系统，结合关系语义和基础模型在多模态数据上的推理能力，并提供人机交互渠道，实现跨数据模态的可解释答案获取。

### 方法

设计KathDB系统，结合关系语义与基础模型的多模态处理能力，并在查询解析、执行和结果解释过程中加入人机交互渠道。

### 主要发现

摘要中未明确提及具体研究发现，主要介绍系统设计理念。

### 结论

KathDB系统能够通过人机交互机制，在多模态数据环境中提供可解释的查询结果。

### 翻译

传统数据库管理系统在关系型数据上执行用户或应用程序提供的SQL查询，具有强大的语义保证和高级查询优化，但编写复杂的SQL很困难，且仅关注结构化表格。当代多模态系统(处理关系型数据以及文本、图像甚至视频)要么暴露低级控制，要求用户在SQL中手动使用(或创建)机器学习UDF，要么将执行完全卸载给黑盒LLM，牺牲了可用性或可解释性。我们提出KathDB，一个结合关系语义与基础模型在多模态数据上推理能力的新系统。此外，KathDB在查询解析、执行和结果解释过程中包含人机交互渠道，使用户能够跨数据模态迭代获得可解释的答案。


### 论文摘要

Traditional DBMSs execute user- or application-provided SQL queries over relational data with strong semantic guarantees and advanced query optimization, but writing complex SQL is hard and focuses only on structured tables. Contemporary multimodal systems (which operate over relations but also text, images, and even videos) either expose low-level controls that force users to use (and possibly create) machine learning UDFs manually within SQL or offload execution entirely to black-box LLMs, sacrificing usability or explainability. We propose KathDB, a new system that combines relational semantics with the reasoning power of foundation models over multimodal data. Furthermore, KathDB includes human-AI interaction channels during query parsing, execution, and result explanation, such that users can iteratively obtain explainable answers across data modalities.

---

## 44. SoccerMaster: A Vision Foundation Model for Soccer Understanding

**论文链接:** [http://arxiv.org/abs/2512.11016v1](http://arxiv.org/abs/2512.11016v1)

**作者:** Haolin Yang, Jiayuan Rao, Haoning Wu, Weidi Xie

**发布时间:** 2025-12-11

### GPT解析

### 总结

该研究提出了SoccerMaster，首个足球特定的视觉基础模型，通过监督多任务预训练统一处理多样化的足球视觉理解任务，并构建了SoccerFactory预训练数据资源。

### 背景

足球理解因领域特定的复杂性和独特挑战而受到越来越多的研究关注。

### 目的

提出一个统一模型来处理多样化的足球视觉理解任务，从细粒度感知（如运动员检测）到语义推理（如事件分类）。

### 方法

开发SoccerMaster足球视觉基础模型，通过监督多任务预训练统一多种任务；构建自动化数据整理流程生成空间标注，创建SoccerFactory预训练数据资源。

### 主要发现

SoccerMaster在多样化下游任务中始终优于任务特定专家模型，展示了其广度和优越性。

### 结论

研究数据、代码和模型将公开可用，促进足球视觉理解领域的发展。

### 翻译

足球理解最近因其领域特定的复杂性和独特挑战而获得了越来越多的研究兴趣。与以往通常依赖孤立、任务特定的专家模型的工作不同，这项工作旨在提出一个统一模型来处理多样化的足球视觉理解任务，从细粒度感知（如运动员检测）到语义推理（如事件分类）。具体而言，我们的贡献有三方面：（i）我们提出了SoccerMaster，这是第一个足球特定的视觉基础模型，通过监督多任务预训练在单一框架内统一多样化的理解任务；（ii）我们开发了一个自动化数据整理流程来生成可扩展的空间标注，并将它们与各种现有的足球视频数据集集成，构建SoccerFactory，一个全面的预训练数据资源；（iii）我们进行了广泛的评估，证明SoccerMaster在多样化的下游任务中始终优于任务特定的专家模型，突显了其广度和优越性。数据、代码和模型将公开可用。


### 论文摘要

Soccer understanding has recently garnered growing research interest due to its domain-specific complexity and unique challenges. Unlike prior works that typically rely on isolated, task-specific expert models, this work aims to propose a unified model to handle diverse soccer visual understanding tasks, ranging from fine-grained perception (e.g., athlete detection) to semantic reasoning (e.g., event classification). Specifically, our contributions are threefold: (i) we present SoccerMaster, the first soccer-specific vision foundation model that unifies diverse understanding tasks within a single framework via supervised multi-task pretraining; (ii) we develop an automated data curation pipeline to generate scalable spatial annotations, and integrate them with various existing soccer video datasets to construct SoccerFactory, a comprehensive pretraining data resource; and (iii) we conduct extensive evaluations demonstrating that SoccerMaster consistently outperforms task-specific expert models across diverse downstream tasks, highlighting its breadth and superiority. The data, code, and model will be publicly available.

---

## 45. mViSE: A Visual Search Engine for Analyzing Multiplex IHC Brain Tissue Images

**论文链接:** [http://arxiv.org/abs/2512.11745v1](http://arxiv.org/abs/2512.11745v1)

**作者:** Liqiang Huang, Rachel W. Mills, Saikiran Mandula, Lin Bai, Mahtab Jeyhani, John Redell, Hien Van Nguyen, Saurabh Prasad, Dragan Maric, Badrinath Roysam

**发布时间:** 2025-12-12

### GPT解析

### 总结

本文介绍了一种名为mViSE的多模态视觉搜索引擎，用于分析脑组织全载玻片多模态图像，无需编程即可实现查询驱动的图像检索和分析。

### 背景

全载玻片脑组织多模态成像会产生信息密集的图像，这些图像难以分析且需要定制软件，限制了研究效率。

### 目的

开发一种替代性的查询驱动、无需编程的策略，使用mViSE学习脑组织的化学结构、细胞结构和髓鞘结构。

### 方法

采用分而治之策略，将数据组织成相关分子标记面板，使用自监督学习训练多模态编码器，多个面板可组合处理视觉查询，使用信息论方法检索相似细胞群落或多细胞生态位。

### 主要发现

mViSE能够有效检索单个细胞、邻近细胞对、组织斑块，并能准确描绘皮层层次、脑区域和亚区域。

### 结论

mViSE作为开源QuPath插件提供，使研究人员无需编程即可进行脑组织探索、区域划分和比较分析。

### 翻译

全载玻片脑组织多模态成像会产生信息密集的图像，这些图像难以分析且需要定制软件。我们提出了一种替代性的查询驱动、无需编程的策略，使用多模态视觉搜索引擎(mViSE)，它能学习脑组织的多方面化学结构、细胞结构和髓鞘结构。我们的分而治之策略将数据组织成相关分子标记面板，并使用自监督学习为每个面板训练多模态编码器，并有明确的学习成功视觉确认。多个面板可以组合起来处理视觉查询，使用信息论方法检索相似的细胞群落或多细胞生态位。这些检索可用于多种目的，包括组织探索、描绘脑区域和皮层细胞层次、分析和比较脑区域，无需计算机编程。我们验证了mViSE检索单个细胞、邻近细胞对、组织斑块、描绘皮层层次、脑区域和亚区域的能力。mViSE作为开源QuPath插件提供。


### 论文摘要

Whole-slide multiplex imaging of brain tissue generates massive information-dense images that are challenging to analyze and require custom software. We present an alternative query-driven programming-free strategy using a multiplex visual search engine (mViSE) that learns the multifaceted brain tissue chemoarchitecture, cytoarchitecture, and myeloarchitecture. Our divide-and-conquer strategy organizes the data into panels of related molecular markers and uses self-supervised learning to train a multiplex encoder for each panel with explicit visual confirmation of successful learning. Multiple panels can be combined to process visual queries for retrieving similar communities of individual cells or multicellular niches using information-theoretic methods. The retrievals can be used for diverse purposes including tissue exploration, delineating brain regions and cortical cell layers, profiling and comparing brain regions without computer programming. We validated mViSE's ability to retrieve single cells, proximal cell pairs, tissue patches, delineate cortical layers, brain regions and sub-regions. mViSE is provided as an open-source QuPath plug-in.

---

## 46. ACCOR: Attention-Enhanced Complex-Valued Contrastive Learning for Occluded Object Classification Using mmWave Radar IQ Signals

**论文链接:** [http://arxiv.org/abs/2512.11556v1](http://arxiv.org/abs/2512.11556v1)

**作者:** Stefan Hägele, Adam Misik, Constantin Patsch, Eckehard Steinbach

**发布时间:** 2025-12-12

**备注:** 7 pages, 6 figures

### GPT解析

### 总结

论文提出了一种名为ACCOR的注意力增强型复值对比学习方法，用于毫米波雷达的遮挡物体分类，通过复值CNN骨干网络、多头注意力层和混合损失函数，实现了超过96%的分类准确率。

### 背景

毫米波雷达能够在恶劣环境条件下可靠运行，可穿透轻质材料如包装或薄墙，实现非视觉传感，并与光学传感器结合为机器人提供增强的环境感知能力。现有MIMO毫米波雷达模型在穿透纸板包装进行遮挡物体分类方面仍有改进空间。

### 目的

提出一种更有效的遮挡物体分类方法，解决现有模型在不同频率下的性能问题，提高毫米波雷达在遮挡物体分类任务中的准确性和鲁棒性。

### 方法

使用复值CNN骨干网络处理复值IQ雷达信号，添加多头注意力层，设计结合加权交叉熵和监督对比的混合损失函数，扩展现有数据集添加67GHz子集，并在两种中心频率下评估模型性能。

### 主要发现

ACCOR方法在64GHz频率下达到96.60%的分类准确率，在67GHz频率下达到93.59%的分类准确率，优于先前的雷达特定模型和适应输入的图像分类模型。

### 结论

复值深度学习与注意力和对比学习的结合能够显著提高毫米波雷达在遮挡物体分类任务中的性能，适用于工业和自动化环境。

### 翻译

毫米波雷达已成为多种领域的稳健传感模式，能够在恶劣环境条件下可靠运行。它能够穿透轻质材料（如包装或薄墙），使工业和自动化环境中实现非视觉传感，并与光学传感器结合使用时可为机器人平台提供增强的环境感知能力。最近的工作表明，MIMO毫米波雷达能够穿透纸板包装进行遮挡物体分类，但现有模型仍有改进空间，需要在不同传感频率下进行更彻底的评估。在本文中，我们提出了ACCOR，一种用于雷达的注意力增强型复值对比学习方法，实现稳健的遮挡物体分类。我们使用复值CNN骨干网络处理复值IQ雷达信号，然后是多头注意力层和混合损失函数。我们提出的损失函数将加权交叉熵项与监督对比项相结合。我们进一步扩展了现有的64GHz数据集，添加了67GHz的遮挡物体子集，并使用两种中心频率评估我们的模型。性能评估表明，我们的方法优于先前的雷达特定模型和适应输入的图像分类模型，在64GHz和67GHz频率下对十种不同物体分别实现了96.60%和93.59%的分类准确率。这些结果证明了复值深度学习与注意力和对比学习相结合对毫米波雷达遮挡物体分类在工业和自动化环境中的益处。


### 论文摘要

Millimeter-wave (mmWave) radar has emerged as a robust sensing modality for several areas, offering reliable operation under adverse environmental conditions. Its ability to penetrate lightweight materials such as packaging or thin walls enables non-visual sensing in industrial and automated environments and can provide robotic platforms with enhanced environmental perception when used alongside optical sensors. Recent work with MIMO mmWave radar has demonstrated its ability to penetrate cardboard packaging for occluded object classification. However, existing models leave room for improvement and warrant a more thorough evaluation across different sensing frequencies. In this paper, we propose ACCOR, an attention-enhanced complex-valued contrastive learning approach for radar, enabling robust occluded object classification. We process complex-valued IQ radar signals using a complex-valued CNN backbone, followed by a multi-head attention layer and a hybrid loss. Our proposed loss combines a weighted cross-entropy term with a supervised contrastive term. We further extend an existing 64 GHz dataset with a 67 GHz subset of the occluded objects and evaluate our model using both center frequencies. Performance evaluation demonstrates that our approach outperforms prior radar-specific models and image classification models with adapted input, achieving classification accuracies of 96.60% at 64 GHz and 93.59% at 67 GHz for ten different objects. These results demonstrate the benefits of complex-valued deep learning with attention and contrastive learning for mmWave radar-based occluded object classification in industrial and automated environments.

---

## 47. SSA3D: Text-Conditioned Assisted Self-Supervised Framework for Automatic Dental Abutment Design

**论文链接:** [http://arxiv.org/abs/2512.11507v1](http://arxiv.org/abs/2512.11507v1)

**作者:** Mianjie Zheng, Xinquan Yang, Along He, Xuguang Li, Feilie Zhong, Xuefen Liu, Kun Tang, Zhicheng Zhang, Linlin Shen

**发布时间:** 2025-12-12

### GPT解析

### 总结

本研究提出了一种名为SS$A^3$D的自监督辅助自动基台设计框架，采用双分支架构和文本条件提示模块，显著提高了牙科种植体基台设计的自动化程度、准确性和效率。

### 背景

牙科种植体基台设计是种植体修复的关键步骤，但手动设计过程繁琐。使用AI自动化的研究受限于缺乏大型标注数据集，而自监督学习方法虽能缓解数据稀缺问题，但需要预训练和微调，导致计算成本高和训练时间长。

### 目的

开发一种新的自监督辅助自动基台设计框架，解决现有方法在数据利用和计算效率方面的局限性，提高基台设计的自动化程度和准确性。

### 方法

提出SS$A^3$D框架，采用双分支架构：重建分支学习恢复被遮蔽的口腔扫描数据并转移结构信息；回归分支在监督学习下预测基台参数，消除单独的预训练和微调过程。同时设计文本条件提示模块，整合临床信息以引导网络关注相关区域并约束参数预测。

### 主要发现

实验表明SS$A^3$D节省了一半的训练时间，比传统SSL方法达到更高准确性，与其他方法相比达到最先进性能，显著提高了自动基台设计的准确性和效率。

### 结论

SS$A^3$D框架有效解决了牙科种植体基台设计自动化中的数据稀缺和计算效率问题，通过创新的双分支架构和文本条件提示模块，实现了更准确、更高效的基台设计。

### 翻译

基台设计是牙科种植体修复的关键步骤。然而，手动设计涉及繁琐的测量和适配，由于缺乏大型标注数据集，使用AI自动此过程的研究有限。尽管自监督学习可以缓解数据稀缺问题，但其对预训练和微调的需求导致计算成本高和训练时间长。在本文中，我们提出了一个自监督辅助的自动基台设计框架，采用双分支架构，包含重建分支和回归分支。重建分支学习恢复被遮蔽的口腔扫描数据，并将学习到的结构信息转移到回归分支。回归分支然后在监督学习下预测基台参数，消除了单独的预训练和微调过程。我们还设计了一个文本条件提示模块，将临床信息整合到框架中，引导网络关注相关区域并约束参数预测。在收集的数据集上进行的大量实验表明，该框架节省了一半的训练时间，比传统SSL方法达到更高准确性，与其他方法相比达到最先进性能，显著提高了自动基台设计的准确性和效率。


### 论文摘要

Abutment design is a critical step in dental implant restoration. However, manual design involves tedious measurement and fitting, and research on automating this process with AI is limited, due to the unavailability of large annotated datasets. Although self-supervised learning (SSL) can alleviate data scarcity, its need for pre-training and fine-tuning results in high computational costs and long training times. In this paper, we propose a Self-supervised assisted automatic abutment design framework (SS$A^3$D), which employs a dual-branch architecture with a reconstruction branch and a regression branch. The reconstruction branch learns to restore masked intraoral scan data and transfers the learned structural information to the regression branch. The regression branch then predicts the abutment parameters under supervised learning, which eliminates the separate pre-training and fine-tuning process. We also design a Text-Conditioned Prompt (TCP) module to incorporate clinical information (such as implant location, system, and series) into SS$A^3$D. This guides the network to focus on relevant regions and constrains the parameter predictions. Extensive experiments on a collected dataset show that SS$A^3$D saves half of the training time and achieves higher accuracy than traditional SSL methods. It also achieves state-of-the-art performance compared to other methods, significantly improving the accuracy and efficiency of automated abutment design.

---

## 48. DAPO: Design Structure-Aware Pass Ordering in High-Level Synthesis with Graph Contrastive and Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2512.11342v1](http://arxiv.org/abs/2512.11342v1)

**作者:** Jinming Ge, Linfeng Du, Likith Anaparty, Shangkun Li, Tingyuan Liang, Afzal Ahmad, Vivek Chaturvedi, Sharad Sinha, Zhiyao Xie, Jiang Xu, Wei Zhang

**发布时间:** 2025-12-12

**备注:** Accepted by DATE 2026

### GPT解析

### 总结

DAPO是一种面向设计结构的通道排序框架，通过结合程序语义提取、对比学习、硬件指标估计和强化学习，实现了比传统HLS工具更优的性能。

### 背景

高级综合(HLS)工具被广泛用于基于FPGA的领域特定加速器设计，但现有工具依赖于从软件编译继承的固定优化策略，限制了其有效性。

### 目的

针对特定设计定制优化策略，需要深入理解语义、准确估计硬件指标和高级搜索算法的能力，而当前方法缺乏这些能力。

### 方法

提出DAPO框架，从控制流图和数据流图中提取程序语义，采用对比学习生成丰富的嵌入，利用分析模型进行准确的硬件指标估计，这些组件共同引导强化学习代理发现设计特定的优化策略。

### 主要发现

在经典HLS设计上的评估表明，DAPO的端到端流程比Vitis HLS平均实现了2.36倍的加速。

### 结论

DAPO框架能够有效地发现设计特定的优化策略，显著提高HLS工具的性能。

### 翻译

高级综合(HLS)工具被广泛应用于基于FPGA的领域特定加速器设计。然而，现有工具依赖于从软件编译继承的固定优化策略，限制了其有效性。针对特定设计定制优化策略需要深入的语义理解、准确的硬件指标估计和高级搜索算法能力——而这些是当前方法所缺乏的。我们提出了DAPO，一种面向设计结构的通道排序框架，它从控制流图和数据流图中提取程序语义，采用对比学习生成丰富的嵌入，并利用分析模型进行准确的硬件指标估计。这些组件共同引导强化学习代理发现设计特定的优化策略。在经典HLS设计上的评估表明，我们的端到端流程比Vitis HLS平均实现了2.36倍的加速。


### 论文摘要

High-Level Synthesis (HLS) tools are widely adopted in FPGA-based domain-specific accelerator design. However, existing tools rely on fixed optimization strategies inherited from software compilations, limiting their effectiveness. Tailoring optimization strategies to specific designs requires deep semantic understanding, accurate hardware metric estimation, and advanced search algorithms -- capabilities that current approaches lack.   We propose DAPO, a design structure-aware pass ordering framework that extracts program semantics from control and data flow graphs, employs contrastive learning to generate rich embeddings, and leverages an analytical model for accurate hardware metric estimation. These components jointly guide a reinforcement learning agent to discover design-specific optimization strategies. Evaluations on classic HLS designs demonstrate that our end-to-end flow delivers a 2.36 speedup over Vitis HLS on average.

---

## 49. Few-Shot VLM-Based G-Code and HMI Verification in CNC Machining

**论文链接:** [http://arxiv.org/abs/2512.11296v1](http://arxiv.org/abs/2512.11296v1)

**作者:** Yasaman Hashem Pour, Nazanin Mahjourian, Vinh Nguyen

**发布时间:** 2025-12-12

### GPT解析

### 总结

这篇论文提出了一种基于少样本视觉语言模型的验证方法，用于同时检查G-code和人机界面的错误和安全状态，解决了传统大型语言模型无法访问视觉模态的问题。

### 背景

手动生成G-code对于学习CNC机床操作很重要。先前的G-code验证工作使用大型语言模型，主要检查编程错误。然而，CNC加工需要大量使用和了解人机界面，显示机器状态和错误。大型语言模型目前无法利用人机界面的知识，因为它们无法访问视觉模态。

### 目的

提出一种基于少样本视觉语言模型的验证方法，同时评估G-code和人机界面显示的错误和安全状态。

### 方法

使用来自15斜角PRO车床的配对G-code文本和相关人机界面截图作为输入数据集，包括正确和易出错的情况。为了实现少样本学习，向视觉语言模型提供基于先验启发式知识的结构化JSON模式。确定提示后，使用包含错误或无错误的G-code和人机界面实例作为少样本示例来指导视觉语言模型。与零样本视觉语言模型进行比较，通过多个不正确G-code和人机界面错误的场景进行评估，使用每槽精度作为指标。

### 主要发现

少样本提示使视觉语言模型能够整体提高检测人机界面错误以及与G-code不一致的能力，有助于更全面的调试。

### 结论

所提出的框架被证明适合验证在CNC培训中通常开发的手动生成G-code。

### 翻译

手动生成G-code对于学习CNC机床的操作很重要。先前在G-code验证方面的工作使用大型语言模型，主要检查书面编程中的错误。然而，CNC加工需要大量使用和了解人机界面，该界面显示机器状态和错误。大型语言模型目前由于无法访问视觉模态，而缺乏利用人机界面知识的能力。本文提出了一种基于少样本视觉语言模型的验证方法，同时评估G-code和人机界面显示的错误和安全状态。输入数据集包括来自15斜角PRO车床的配对G-code文本和相关人机界面截图，包含正确和易出错的情况。为了实现少样本学习，向视觉语言模型提供了基于先验启发式知识的结构化JSON模式。在确定提示后，使用包含错误或无错误的G-code和人机界面实例作为少样本示例来指导视觉语言模型。然后通过多个不正确G-code和人机界面错误的场景，在每槽精度的基础上与零样本视觉语言模型进行比较。视觉语言模型表明，少样本提示导致了整体上提高检测人机界面错误以及与G-code不一致的能力，从而实现更全面的调试。因此，所提出的框架被证明适合验证在CNC培训中通常开发的手动生成G-code。


### 论文摘要

Manual generation of G-code is important for learning the operation of CNC machines. Prior work in G-code verification uses Large-Language Models (LLMs), which primarily examine errors in the written programming. However, CNC machining requires extensive use and knowledge of the Human-Machine Interface (HMI), which displays machine status and errors. LLMs currently lack the capability to leverage knowledge of HMIs due to their inability to access the vision modality. This paper proposes a few-shot VLM-based verification approach that simultaneously evaluates the G-code and the HMI display for errors and safety status. The input dataset includes paired G-code text and associated HMI screenshots from a 15-slant-PRO lathe, including both correct and error-prone cases. To enable few-shot learning, the VLM is provided with a structured JSON schema based on prior heuristic knowledge. After determining the prompts, instances of G-code and HMI that either contain errors or are error free are used as few-shot examples to guide the VLM. The model was then evaluated in comparison to a zero-shot VLM through multiple scenarios of incorrect G-code and HMI errors with respect to per-slot accuracy. The VLM showed that few-shot prompting led to overall enhancement of detecting HMI errors and discrepancies with the G-code for more comprehensive debugging. Therefore, the proposed framework was demonstrated to be suitable for verification of manually generated G-code that is typically developed in CNC training.

---

## 50. Leveraging LLMs for Title and Abstract Screening for Systematic Review: A Cost-Effective Dynamic Few-Shot Learning Approach

**论文链接:** [http://arxiv.org/abs/2512.11261v1](http://arxiv.org/abs/2512.11261v1)

**作者:** Yun-Chung Liu, Rui Yang, Jonathan Chong Kai Liew, Ziran Yin, Henry Foote, Christopher J. Lindsell, Chuan Hong

**发布时间:** 2025-12-12

**备注:** 22 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种两阶段动态少样本学习(DFSL)方法，用于提高大语言模型在系统综述标题和摘要筛选任务中的效率和性能。

### 背景

系统综述是循证医学的关键组成部分，对综合现有研究证据和指导临床决策至关重要。然而，随着研究出版物快速增长，进行系统综述变得越来越繁重，其中标题和摘要筛选是最耗时和资源密集的步骤之一。

### 目的

减轻系统综述中的筛选负担，提高大语言模型在标题和摘要筛选任务中的效率和性能。

### 方法

设计了一个两阶段动态少样本学习(DFSL)方法：首先使用低成本LLM进行初步筛选，然后使用高性能LLM重新评估低置信度实例，从而在提高筛选性能的同时控制计算成本。该方法在10个系统综述中进行了评估。

### 主要发现

结果证明该方法具有良好的泛化能力和成本效益，有潜力减少手动筛选负担并加速实际应用中的系统综述过程。

### 结论

该两阶段动态少样本学习方法能有效提高系统综述中标题和摘要筛选的效率和性能，为解决系统综述中的筛选负担提供了可行的解决方案。

### 翻译

系统综述是循证医学的关键组成部分，在综合现有研究证据和指导临床决策方面发挥着关键作用。然而，随着研究出版物的快速增长，进行系统综述变得越来越繁重，其中标题和摘要筛选是最耗时和资源密集的步骤之一。为缓解这一问题，我们设计了一种两阶段动态少样本学习(DFSL)方法，旨在提高大语言模型(LLMs)在标题和摘要筛选任务中的效率和性能。具体而言，该方法首先使用低成本LLM进行初步筛选，然后使用高性能LLM重新评估低置信度实例，从而在提高筛选性能的同时控制计算成本。我们在10个系统综述中评估了这种方法，结果证明了其良好的泛化能力和成本效益，有潜力在实际应用中减少手动筛选负担并加速系统综述过程。


### 论文摘要

Systematic reviews are a key component of evidence-based medicine, playing a critical role in synthesizing existing research evidence and guiding clinical decisions. However, with the rapid growth of research publications, conducting systematic reviews has become increasingly burdensome, with title and abstract screening being one of the most time-consuming and resource-intensive steps. To mitigate this issue, we designed a two-stage dynamic few-shot learning (DFSL) approach aimed at improving the efficiency and performance of large language models (LLMs) in the title and abstract screening task. Specifically, this approach first uses a low-cost LLM for initial screening, then re-evaluates low-confidence instances using a high-performance LLM, thereby enhancing screening performance while controlling computational costs. We evaluated this approach across 10 systematic reviews, and the results demonstrate its strong generalizability and cost-effectiveness, with potential to reduce manual screening burden and accelerate the systematic review process in practical applications.

---

