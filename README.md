# 今日论文推荐 - 2025-12-17

共 61 篇论文

---

## 1. Transfer Learning-Based Surrogate Modeling for Nonlinear Time-History Response Analysis of High-Fidelity Structural Models

**论文链接:** [http://arxiv.org/abs/2512.14161v1](http://arxiv.org/abs/2512.14161v1)

**作者:** Keiichi Ishikawa, Yuma Matsumoto, Taro Yaoyama, Sangwon Lee, Tatsuya Itoi

**发布时间:** 2025-12-16

**备注:** 20 pages, 21 figures

### GPT解析

### 总结

本研究提出了一种使用迁移学习构建高保真度响应分析替代模型的框架，以解决基于性能的地震工程框架中非线性时程响应分析计算成本高的问题。该框架利用低保真度模型作为预训练模型，显著减少了高保真度模型的训练成本。案例研究表明，仅使用20个样本训练的模型能够准确预测20层钢框架结构的地震响应。

### 背景

在基于性能的地震工程框架中，需要进行大量地震动的非线性时程响应分析来评估建筑物或土木工程结构的地震风险。然而，这些数值模拟计算成本高，限制了框架的实际应用。先前的研究使用机器学习预测结构响应，但这些研究大多专注于计算成本较低的单自由度模型，而高保真度模型需要更多的训练数据，进一步增加了计算负担。

### 目的

开发一种能够用少量训练样本实现高保真度响应分析替代模型的方法，以减少计算成本，同时保持预测的准确性，从而促进基于性能的地震工程框架的实际应用。

### 方法

提出了一种使用迁移学习构建高保真度响应分析替代模型的框架。该方法使用低保真度响应分析的替代模型作为预训练模型，并将其知识迁移到高保真度响应分析替代模型的构建中，从而显著降低计算成本。

### 主要发现

作为案例研究，仅使用20个样本作为训练数据集，成功构建了预测20层钢框架结构响应的替代模型。该模型预测的地震动响应与特定场地的基于时间的危险性一致，证明了所提出框架的有效性。

### 结论

所提出的迁移学习框架能够有效地减少高保真度响应分析替代模型的训练成本，同时保持预测的准确性。这种方法为基于性能的地震工程框架的实际应用提供了新的可能性，特别是在需要高保真度模型但计算资源有限的情况下。

### 翻译

在基于性能的地震工程框架中，需要对大量地震动进行非线性时程响应分析，以评估建筑物或土木工程结构的地震风险。然而，这类数值模拟计算成本高，限制了框架的实际应用。为解决这一问题，先前的研究使用机器学习以较低的计算成本预测结构对地震动的响应。这些研究通常对几百个地震动进行非线性时程响应分析，并利用结果训练和验证替代模型。然而，大多数先前的研究专注于计算成本较低的响应分析模型，如单自由度系统。基于性能的地震工程中需要高保真度响应分析的替代模型，以丰富用于损伤评估的信息数量和多样性。值得注意的是，随着响应分析模型保真度的提高，创建训练和验证数据集的计算成本也会增加。因此，需要一种能够用少量训练样本实现高保真度响应分析替代模型的方法。本研究提出了一种使用迁移学习构建高保真度响应分析替代模型的框架。该框架使用低保真度响应分析的替代模型作为预训练模型，并将其知识迁移到高保真度响应分析替代模型的构建中，从而显著降低计算成本。作为案例研究，仅使用20个样本作为训练数据集，构建了预测20层钢框架结构响应的替代模型。该替代模型预测的地震动响应与特定场地的基于时间的危险性一致。


### 论文摘要

In a performance based earthquake engineering (PBEE) framework, nonlinear time-history response analysis (NLTHA) for numerous ground motions are required to assess the seismic risk of buildings or civil engineering structures. However, such numerical simulations are computationally expensive, limiting the real-world practical application of the framework. To address this issue, previous studies have used machine learning to predict the structural responses to ground motions with low computational costs. These studies typically conduct NLTHAs for a few hundreds ground motions and use the results to train and validate surrogate models. However, most of the previous studies focused on computationally-inexpensive response analysis models such as single degree of freedom. Surrogate models of high-fidelity response analysis are required to enrich the quantity and diversity of information used for damage assessment in PBEE. Notably, the computational cost of creating training and validation datasets increases if the fidelity of response analysis model becomes higher. Therefore, methods that enable surrogate modeling of high-fidelity response analysis without a large number of training samples are needed. This study proposes a framework that uses transfer learning to construct the surrogate model of a high-fidelity response analysis model. This framework uses a surrogate model of low-fidelity response analysis as the pretrained model and transfers its knowledge to construct surrogate models for high-fidelity response analysis with substantially reduced computational cost. As a case study, surrogate models that predict responses of a 20-story steel moment frame were constructed with only 20 samples as the training dataset. The responses to the ground motions predicted by constructed surrogate model were consistent with a site-specific time-based hazard.

---

## 2. Unreasonable effectiveness of unsupervised learning in identifying Majorana topology

**论文链接:** [http://arxiv.org/abs/2512.13825v1](http://arxiv.org/abs/2512.13825v1)

**作者:** Jacob Taylor, Haining Pan, Sankar Das Sarma

**发布时间:** 2025-12-15

**备注:** 7 pages, 4 figures

### GPT解析

### 总结

本研究结合无监督和监督学习，使用自编码器分析马约拉纳纳米线中的无标签数据，以识别拓扑特性并确定拓扑与平庸状态的交叉点。

### 背景

无监督学习在深度学习中使用无标签数据，迫使算法发现数据中的隐藏模式。这种方法可能成为识别拓扑序的有力工具，因为拓扑特性（如拓扑超导性）并不总是以明显的物理方式显现。

### 目的

结合无监督和监督学习方法，利用自编码器分析马约拉纳纳米线中的无标签数据，以区分拓扑状态和平庸状态，并确定它们在参数空间中的交叉点。

### 方法

结合无监督和监督学习，使用自编码器分析真实短无序纳米线中的马约拉纳分裂的无标签数据。

### 主要发现

无标签数据不仅能区分'拓扑'和'平庸'状态，还能确定它们在相关参数空间中的交叉点。

### 结论

这种结合无监督和监督学习的方法可能成为识别马约拉纳纳米线拓扑特性的有用工具。

### 翻译

在无监督学习中，深度学习的训练数据不带有任何标签，因此迫使算法发现数据中的隐藏模式以辨别有用信息。原则上，这可能是一个识别拓扑序的有力工具，因为拓扑并不总是以明显的物理方式（例如拓扑超导性）表现以供确认。然而，问题在于无监督学习是一个困难的挑战，需要巨大的计算资源，且不一定总是有效。在当前工作中，我们使用自编码器结合无监督和监督学习，证明在真实短无序纳米线中的马约拉纳分裂的无标签数据不仅可以区分'拓扑'和'平庸'，还可以确定它们在相关参数空间中的交叉位置。这可能是识别马约拉纳纳米线拓扑特性的有用工具。


### 论文摘要

In unsupervised learning, the training data for deep learning does not come with any labels, thus forcing the algorithm to discover hidden patterns in the data for discerning useful information. This, in principle, could be a powerful tool in identifying topological order since topology does not always manifest in obvious physical ways (e.g., topological superconductivity) for its decisive confirmation. The problem, however, is that unsupervised learning is a difficult challenge, necessitating huge computing resources, which may not always work. In the current work, we combine unsupervised and supervised learning using an autoencoder to establish that unlabeled data in the Majorana splitting in realistic short disordered nanowires may enable not only a distinction between `topological' and `trivial', but also where their crossover happens in the relevant parameter space. This may be a useful tool in identifying topology in Majorana nanowires.

---

## 3. Probabilistic Predictions of Process-Induced Deformation in Carbon/Epoxy Composites Using a Deep Operator Network

**论文链接:** [http://arxiv.org/abs/2512.13746v1](http://arxiv.org/abs/2512.13746v1)

**作者:** Elham Kiyani, Amit Makarand Deshpande, Madhura Limaye, Zhiwei Gao, Sai Aditya Pradeep, Srikanth Pilla, Gang Li, Zhen Li, George Em Karniadakis

**发布时间:** 2025-12-15

**备注:** 21 pages, 13 figures

### GPT解析

### 总结

本研究开发了一种结合物理模型和数据驱动方法的框架，用于预测和优化复合材料制造过程中的过程诱导变形(PID)。研究使用双机制模型模拟PID，并通过深度算子网络和迁移学习技术提高预测精度，同时利用集合卡尔曼反演量化不确定性。

### 背景

纤维增强体和聚合物基体在制造条件下的热膨胀系数不匹配以及热固性树脂固化过程中的基体收缩，会产生多尺度的残余应力，这些应力的部分释放会导致过程诱导变形(PID)，需要通过优化的非等温固化循环进行准确预测和缓解。

### 目的

开发一种能够准确预测复合材料制造过程中PID的框架，并通过优化固化循环减少PID，同时量化预测中的不确定性。

### 方法

研究使用双机制框架模拟PID，考虑热膨胀/收缩和固化收缩；基于物理模型开发深度算子网络(DeepONets)代理模型，结合高保真模拟和实验数据；扩展到特征线性调制(FiLM) DeepONet，能够预测固化程度、粘度和变形的时间历程；使用迁移学习处理有限实验数据；通过集合卡尔曼反演(EKI)量化不确定性并支持优化。

### 主要发现

双机制模型能够准确模拟PID；深度算子网络结合物理模型和数据驱动方法提高了PID预测的准确性；特征线性调制DeepONet能够预测多个参数的时间历程；迁移学习有效处理了有限实验数据；集合卡尔曼反演能够量化预测不确定性并支持优化。

### 结论

结合物理模型和数据驱动方法的框架能够有效预测和优化复合材料制造过程中的PID，为减少复合材料制造中的变形提供了有效工具。

### 翻译

纤维增强体和聚合物基体由于热膨胀系数不匹配以及热固性树脂固化过程中的基体收缩，对制造条件的响应不同。这些异质性在多个长度尺度上产生残余应力，其部分释放导致过程诱导变形(PID)，需要通过优化的非等温固化循环进行准确预测和缓解。本研究考虑单向AS4碳纤维/胺双官能团环氧预浸料，并使用考虑热膨胀/收缩和固化收缩的双机制框架模拟PID。模型通过制造试验验证以识别初始和边界条件，然后用于生成多种非等温固化循环的PID响应。基于此物理基础，我们开发了基于深度算子网络(DeepONets)的数据驱动代理模型。DeepONet在结合高保真模拟和PID目标实验测量的数据集上进行训练。我们进一步扩展到特征线性调制(FiLM) DeepONet，其中分支网络特征由包括初始固化程度的外部参数调制，能够预测固化程度、粘度和变形的时间历程。由于实验数据仅在有限时间点可用(例如最终变形)，我们使用迁移学习：模拟训练的主干和分支网络保持固定，仅使用测量的最终变形更新最后一层。最后，我们通过集合卡尔曼反演(EKI)增强框架，以量化实验条件下的不确定性并支持减少复合材料PID的固化时间表优化。


### 论文摘要

Fiber reinforcement and polymer matrix respond differently to manufacturing conditions due to mismatch in coefficient of thermal expansion and matrix shrinkage during curing of thermosets. These heterogeneities generate residual stresses over multiple length scales, whose partial release leads to process-induced deformation (PID), requiring accurate prediction and mitigation via optimized non-isothermal cure cycles. This study considers a unidirectional AS4 carbon fiber/amine bi-functional epoxy prepreg and models PID using a two-mechanism framework that accounts for thermal expansion/shrinkage and cure shrinkage. The model is validated against manufacturing trials to identify initial and boundary conditions, then used to generate PID responses for a diverse set of non-isothermal cure cycles (time-temperature profiles). Building on this physics-based foundation, we develop a data-driven surrogate based on Deep Operator Networks (DeepONets). A DeepONet is trained on a dataset combining high-fidelity simulations with targeted experimental measurements of PID. We extend this to a Feature-wise Linear Modulation (FiLM) DeepONet, where branch-network features are modulated by external parameters, including the initial degree of cure, enabling prediction of time histories of degree of cure, viscosity, and deformation. Because experimental data are available only at limited time instances (for example, final deformation), we use transfer learning: simulation-trained trunk and branch networks are fixed and only the final layer is updated using measured final deformation. Finally, we augment the framework with Ensemble Kalman Inversion (EKI) to quantify uncertainty under experimental conditions and to support optimization of cure schedules for reduced PID in composites.

---

## 4. Unsupervised Learning of Density Estimates with Topological Optimization

**论文链接:** [http://arxiv.org/abs/2512.08895v2](http://arxiv.org/abs/2512.08895v2)

**作者:** Sunia Tanweer, Firas A. Khasawneh

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了一种基于拓扑的无监督学习方法，用于自动选择最优的核带宽参数，并通过与经典技术比较展示了其在不同维度上的潜力。

### 背景

核密度估计是机器学习、贝叶斯推断、随机动力学和信号处理等多种算法的关键组成部分。然而，这种无监督密度估计技术需要调整关键超参数：核带宽。带宽选择至关重要，因为它通过过度或不足平滑拓扑特征来控制偏差-方差权衡。

### 目的

开发一种使用基于拓扑的损失函数的无监督学习方法，以自动选择最优带宽，并在不同维度上展示其有效性。

### 方法

提出一种基于拓扑的损失函数的无监督学习方法，用于自动和无监督地选择最优带宽，并与经典技术进行基准比较。

### 主要发现

基于拓扑的损失函数可以有效选择最优核带宽参数，这种方法在不同维度上都显示出良好性能。

### 结论

基于拓扑的无监督带宽选择方法有效，可以在不同维度上自动选择最优带宽，无需人工干预。

### 翻译

核密度估计是机器学习、贝叶斯推断、随机动力学和信号处理等多种算法的关键组成部分。然而，这种无监督密度估计技术需要调整一个关键的超参数：核带宽。带宽的选择至关重要，因为它通过过度或不足平滑拓扑特征来控制偏差-方差权衡。拓扑数据分析提供了数学量化拓扑特征的方法，如连通分量、环、空隙等，即使在无法可视化密度估计的高维空间中也是如此。在本文中，我们提出了一种使用基于拓扑的损失函数的无监督学习方法，用于自动和无监督地选择最优带宽，并与经典技术进行比较——展示了其在不同维度上的潜力。


### 论文摘要

Kernel density estimation is a key component of a wide variety of algorithms in machine learning, Bayesian inference, stochastic dynamics and signal processing. However, the unsupervised density estimation technique requires tuning a crucial hyperparameter: the kernel bandwidth. The choice of bandwidth is critical as it controls the bias-variance trade-off by over- or under-smoothing the topological features. Topological data analysis provides methods to mathematically quantify topological characteristics, such as connected components, loops, voids et cetera, even in high dimensions where visualization of density estimates is impossible. In this paper, we propose an unsupervised learning approach using a topology-based loss function for the automated and unsupervised selection of the optimal bandwidth and benchmark it against classical techniques -- demonstrating its potential across different dimensions.

---

## 5. Unified Semantic Transformer for 3D Scene Understanding

**论文链接:** [http://arxiv.org/abs/2512.14364v1](http://arxiv.org/abs/2512.14364v1)

**作者:** Sebastian Koch, Johanna Wald, Hide Matsuki, Pedro Hermosilla, Timo Ropinski, Federico Tombari

**发布时间:** 2025-12-16

**备注:** Project page: https://unite-page.github.io/

### GPT解析

### 总结

UNITE是一个统一的3D场景理解语义变换器，是一种新颖的前馈神经网络，能够将多种3D语义任务整合到单一模型中，仅从RGB图像直接预测多个语义属性。

### 背景

全面的3D场景理解涉及捕捉和解析非结构化的3D环境。由于现实世界的固有复杂性，现有模型主要被开发为特定任务的，并且仅限于特定任务。

### 目的

引入UNITE，一个统一的3D场景理解语义变换器，旨在将多种3D语义任务统一到一个单一模型中，实现端到端的3D场景理解。

### 方法

UNITE是一个新颖的前馈神经网络，能够在完全端到端的方式下对未见过的场景进行操作，仅需几秒钟即可推断完整的3D语义几何。该方法结合了2D蒸馏进行训练，严重依赖自监督，并利用新颖的多视图损失来确保3D视图一致性。

### 主要发现

UNITE在几种不同的语义任务上达到了最先进的性能，甚至在许多情况下超越了特定任务的模型，超过了那些在真实3D几何上运行的方法。

### 结论

UNITE是一个统一的3D场景理解模型，能够高效地处理多种语义任务，并且性能优于现有的特定任务模型。

### 翻译

全面的3D场景理解涉及捕捉和解析非结构化的3D环境。由于现实世界的固有复杂性，现有模型主要被开发为特定任务的，并且仅限于特定任务。我们介绍了UNITE，一个用于3D场景理解的统一语义变换器，这是一个新颖的前馈神经网络，将多种3D语义任务统一到一个单一模型中。我们的模型以完全端到端的方式对未见过的场景进行操作，仅需几秒钟即可推断完整的3D语义几何。我们的方法能够仅从RGB图像直接预测多个语义属性，包括3D场景分割、实例嵌入、开放词汇特征以及功能性和关节性。该方法结合了2D蒸馏进行训练，严重依赖自监督，并利用新颖的多视图损失来确保3D视图一致性。我们证明UNITE在几种不同的语义任务上达到了最先进的性能，甚至在许多情况下超越了特定任务的模型，超过了那些在真实3D几何上运行的方法。项目网站见 unite-page.github.io

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有3D场景理解模型大多是特定任务的问题，无法统一处理多种语义任务。这个问题在AR/VR和机器人应用中非常重要，因为3D场景理解是这些应用的基础，使系统能够感知周围环境并构建结合了几何和高级语义的丰富3D表示。统一的模型可以简化流程，提高效率，同时处理多种语义任务如语义分割、实例分割、开放词汇搜索和关节预测等。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D场景理解方法的局限性：辐射场方法依赖相机姿态和场景特定训练；蒸馏方法需要3D重建；基于提升的方法依赖手工设计且难以扩展。作者借鉴了VGGT作为几何基础，使用DPT学习语义特征，采用对比学习方法学习实例嵌入，并使用专门的DPT头预测关节。训练方法上，作者使用2D基础模型的蒸馏，主要依赖自监督，并设计了新的多视图一致性损失来确保不同视角的一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个单一的前馈transformer模型，能够同时处理多种3D语义任务，并在单一网络中联合学习几何和语义，避免手工设计的提升步骤。整体流程为：1)输入多视角RGB图像；2)使用VGGT backbone预测几何属性；3)通过DPT头预测开放词汇语义特征；4)学习实例嵌入；5)预测物体关节；6)通过多视图一致性损失确保不同视角的一致性；7)输出包含语义特征、实例特征和关节预测的3D点云重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次在单一前馈网络中整合多种3D语义任务；2)完全端到端训练，无需手工后处理；3)提出新的多视图一致性损失；4)联合几何-语义学习；5)实现开放词汇推理。与之前工作不同，UNITE不需要场景特定训练(对比辐射场方法)，不需要推理时的3D重建(对比蒸馏方法)，不依赖手工设计的视图选择(对比基于提升的方法)，且是真正统一的模型(对比其他需要多阶段的语义前馈模型)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UNITE是一个统一的语义转换器，它通过在单一前馈网络中联合学习几何和语义，实现了从RGB图像直接进行多任务3D场景理解，并在多个任务上达到了最先进的性能。'}


### 论文摘要

Holistic 3D scene understanding involves capturing and parsing unstructured 3D environments. Due to the inherent complexity of the real world, existing models have predominantly been developed and limited to be task-specific. We introduce UNITE, a Unified Semantic Transformer for 3D scene understanding, a novel feed-forward neural network that unifies a diverse set of 3D semantic tasks within a single model. Our model operates on unseen scenes in a fully end-to-end manner and only takes a few seconds to infer the full 3D semantic geometry. Our approach is capable of directly predicting multiple semantic attributes, including 3D scene segmentation, instance embeddings, open-vocabulary features, as well as affordance and articulations, solely from RGB images. The method is trained using a combination of 2D distillation, heavily relying on self-supervision and leverages novel multi-view losses designed to ensure 3D view consistency. We demonstrate that UNITE achieves state-of-the-art performance on several different semantic tasks and even outperforms task-specific models, in many cases, surpassing methods that operate on ground truth 3D geometry. See the project website at unite-page.github.io

---

## 6. Consistent Instance Field for Dynamic Scene Understanding

**论文链接:** [http://arxiv.org/abs/2512.14126v1](http://arxiv.org/abs/2512.14126v1)

**作者:** Junyi Wu, Van Nguyen Nguyen, Benjamin Planche, Jiachen Tao, Changchang Sun, Zhongpai Gao, Zhenghao Zhao, Anwesa Choudhuri, Gengyu Zhang, Meng Zheng, Feiran Wang, Terrence Chen, Yan Yan, Ziyan Wu

**发布时间:** 2025-12-16

### GPT解析

### 总结

这篇论文提出了一致实例场，一种用于动态场景理解的连续且概率性的时空表示方法，通过建模每个时空点的占用概率和条件实例分布，将可见性与持久对象身份分离。

### 背景

先前的方法依赖于离散跟踪或视图相关特征，存在局限性，需要新的表示方法来解决动态场景理解中的问题。

### 目的

开发一种能够处理动态场景理解的连续且概率性的时空表示方法，解决可见性与对象身份分离的问题。

### 方法

1) 引入基于可变形3D高斯的新实例嵌入表示；2) 联合编码辐射度和语义信息；3) 通过可微分光栅化直接从输入RGB图像和实例掩码中学习；4) 引入新机制校准每个高斯身份并重新采样高斯朝向语义活跃区域。

### 主要发现

在HyperNeRF和Neu3D数据集上的实验表明，该方法在新型视图全景分割和开放词汇4D查询任务上显著优于最先进的方法。

### 结论

一致实例场方法通过连续概率表示有效解决了动态场景理解中的可见性与对象身份分离问题，在多个任务上取得了优越性能。

### 翻译

我们提出了一致实例场，一种用于动态场景理解的连续且概率性的时空表示。与依赖于离散跟踪或视图相关特征的先前方法不同，我们的方法通过建模每个时空点的占用概率和条件实例分布，将可见性与持久对象身份分离。为此，我们引入了一种基于可变形3D高斯的新实例嵌入表示，该表示联合编码辐射度和语义信息，并通过可微分光栅化直接从输入RGB图像和实例掩码中学习。此外，我们引入了新的机制来校准每个高斯的身份并重新采样高斯朝向语义活跃区域，确保在空间和时间上保持一致的实例表示。在HyperNeRF和Neu3D数据集上的实验表明，我们的方法在新型视图全景分割和开放词汇4D查询任务上显著优于最先进的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决动态场景理解中的实例一致性建模问题。现有方法在处理动态场景时往往依赖于离散跟踪或视图相关特征，导致实例身份识别不稳定，特别是在物体变形或遮挡情况下。这个问题在现实中非常重要，因为动态场景理解是增强/虚拟现实、自动驾驶和机器人等广泛应用的基础，需要系统能够准确识别物体身份并保持时间上一致的语义理解。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：它们依赖于视图相关特征，通过RGB调制进行监督，导致语义在动态场景中不一致。然后提出新视角：将焦点从跟踪变化的外观转向建模物体在4D空间中的持久组成。作者借鉴了基于NeRF和基于高斯的可变形表示，但扩展了它们以编码实例语义信息。同时参考了视觉语言特征和2D掩码监督方法，但解决了它们的视图依赖性问题。最终引入了实例身份估计和实例引导重采样两个关键机制，以对齐离散高斯表示与底层连续场。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是引入一致的实例场(Consistent Instance Field)，一种连续的时空表示，联合编码物体的存在性和身份。将物体的存在性与持久身份解耦：占用概率建模物理存在的时空连续性，条件身份分布保持变形和运动中的一致归属。整体实现流程包括：1)实例嵌入高斯表示，将场景表示为一组编码几何、颜色、不透明度、占用概率和身份分布的高斯；2)实例身份估计，通过跨视图和时间的聚合推断高斯身份并减轻可见性偏差；3)实例引导重采样，根据实例场信号自适应重新分配高斯密度；4)场感知溅射，通过可微分渲染联合优化所有组件。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出了一致的实例场(CIF)，一种连续且概率化的时空表示，联合编码占用和实例身份；2)引入实例身份估计和实例引导重采样两个机制，将离散高斯表示与连续场对齐；3)实现了在4D场景中连贯的几何、外观和身份建模。相比之前的工作，CIF解决了视图依赖性问题，避免了将身份与辐射关联，防止了可见性偏差导致的漂移；提供了更高效、更可解释的显式表示；不仅编码几何和外观，还明确建模实例身份；通过物理存在的持久性建立语义与几何和外观的原则性连接。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一致的实例场框架，通过联合建模物体的存在性和持久身份，实现了动态场景中时空一致的实例理解，显著提升了新视角全景分割和开放词汇4D查询任务的性能。'}


### 论文摘要

We introduce Consistent Instance Field, a continuous and probabilistic spatio-temporal representation for dynamic scene understanding. Unlike prior methods that rely on discrete tracking or view-dependent features, our approach disentangles visibility from persistent object identity by modeling each space-time point with an occupancy probability and a conditional instance distribution. To realize this, we introduce a novel instance-embedded representation based on deformable 3D Gaussians, which jointly encode radiance and semantic information and are learned directly from input RGB images and instance masks through differentiable rasterization. Furthermore, we introduce new mechanisms to calibrate per-Gaussian identities and resample Gaussians toward semantically active regions, ensuring consistent instance representations across space and time. Experiments on HyperNeRF and Neu3D datasets demonstrate that our method significantly outperforms state-of-the-art methods on novel-view panoptic segmentation and open-vocabulary 4D querying tasks.

---

## 7. Deep Learning Perspective of Scene Understanding in Autonomous Robots

**论文链接:** [http://arxiv.org/abs/2512.14020v1](http://arxiv.org/abs/2512.14020v1)

**作者:** Afia Maham, Dur E Nayab Tashfa

**发布时间:** 2025-12-16

**备注:** 11 pages. Review Paper on Deep Learning Perspective of Scene Understanding in Autonomous Robots

### GPT解析

### 总结

本文综述了深度学习在自主机器人场景理解中的应用，包括目标检测、语义和实例分割、深度估计、3D重建和视觉SLAM等创新技术，强调了这些技术如何解决传统几何模型的局限性并提升机器人的环境感知能力。

### 背景

自主机器人在复杂环境中进行场景理解面临挑战，传统几何模型存在局限性，需要更先进的技术来提高感知能力。

### 目的

综述深度学习技术在自主机器人场景理解中的应用，分析其优势和局限性，并指出未来研究方向。

### 方法

对深度学习在自主机器人场景理解中的应用进行文献综述，重点关注目标检测、语义和实例分割、深度估计、3D重建和视觉SLAM等技术。

### 主要发现

深度学习技术能够解决传统几何模型的局限性，提高实时深度感知能力（即使在遮挡和无纹理表面情况下），增强语义推理能力，使机器人能更好地理解环境，并在集成到动态和非结构化环境中时提升决策、导航和交互效果。

### 结论

基于学习的场景理解对自主机器人至关重要，但仍存在一些问题需要进一步研究解决，未来研究应关注解决这些问题以推进该领域的发展。

### 翻译

本文综述了深度学习在自主机器人场景理解中的应用，包括目标检测、语义和实例分割、深度估计、3D重建以及视觉SLAM等方面的创新。论文强调了这些技术如何解决传统几何模型的局限性，提高对遮挡和无纹理表面的实时深度感知能力，并增强语义推理以更好地理解环境。当这些感知模块集成到动态和非结构化环境中时，它们在决策、导航和交互方面变得更加有效。最后，该综述概述了现有问题和研究方向，以推进基于学习的自主机器人场景理解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自主机器人在复杂、动态和非结构化环境中的场景理解问题。这个问题在现实中非常重要，因为随着自主机器人在医疗、物流和制造等领域的广泛应用，它们需要具备高级感知和认知能力来安全有效地与人类和环境互动。传统计算机视觉方法在处理遮挡、无纹理表面和动态环境时存在局限，而深度学习能够提供更强大的特征提取、上下文分析和多模态融合能力，使机器人能够实现更准确的环境理解，从而支持更复杂的决策、导航和交互任务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作为一篇综述论文，作者并非设计新方法，而是系统性地梳理和分析了现有深度学习技术在机器人场景理解中的应用。作者首先指出了传统计算机视觉方法的局限性，然后深入探讨了各种深度学习架构（CNN、RNN、GAN、Transformer）如何解决这些局限。作者确实大量借鉴了现有工作，论文引用了81篇参考文献，涵盖了从基础理论研究到实际应用的广泛内容。通过整合这些现有工作，作者构建了一个全面的框架，展示了深度学习如何提升机器人在目标检测、语义分割、深度估计、3D重建和视觉SLAM等关键任务中的性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '这篇论文的核心思想是利用深度学习技术，特别是卷积神经网络、循环神经网络、生成对抗网络和Transformer等架构，使机器人能够从原始传感器数据中提取丰富的层次化特征，实现对环境的全面理解。这种方法超越了传统的几何模型，能够处理遮挡、无纹理表面等挑战，并在动态和非结构化环境中提供更好的语义推理。整体实现流程包括：首先介绍各种深度学习基础架构及其特点；然后分析这些技术在具体任务（目标检测、语义分割、深度估计、3D重建和视觉SLAM）中的应用；接着探讨在动态环境中处理动态物体、运动预测和场景理解的挑战；最后指出现有方法的局限性并提出未来研究方向。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '作为综述论文，本文的关键创新点在于：1) 全面性：涵盖了深度学习在机器人场景理解中的多个方面，从基础架构到具体应用，再到未来挑战；2) 整合视角：将几何和语义信息整合到场景理解中，超越了传统方法仅关注几何的局限；3) 强调动态环境：特别关注了动态环境中的场景理解，包括动态物体的识别、运动预测等；4) 前沿技术整合：探讨了Transformer、NeRFs等最新技术在机器人场景理解中的应用；5) 实用导向：关注实际应用中的挑战，如实时性能、计算效率、鲁棒性等。相比之前的综述工作，本文更新了最新研究成果，更加强调动态环境处理，更全面地探讨实际应用挑战，并更注重几何和语义信息的整合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文系统性地综述了深度学习技术在自主机器人场景理解中的应用，从基础架构到具体任务，再到实际挑战，为理解和推进机器人环境感知能力提供了全面的视角和指导。'}


### 论文摘要

This paper provides a review of deep learning applications in scene understanding in autonomous robots, including innovations in object detection, semantic and instance segmentation, depth estimation, 3D reconstruction, and visual SLAM. It emphasizes how these techniques address limitations of traditional geometric models, improve depth perception in real time despite occlusions and textureless surfaces, and enhance semantic reasoning to understand the environment better. When these perception modules are integrated into dynamic and unstructured environments, they become more effective in decisionmaking, navigation and interaction. Lastly, the review outlines the existing problems and research directions to advance learning-based scene understanding of autonomous robots.

---

## 8. MMDrive: Interactive Scene Understanding Beyond Vision with Multi-representational Fusion

**论文链接:** [http://arxiv.org/abs/2512.13177v2](http://arxiv.org/abs/2512.13177v2)

**作者:** Minghui Hou, Wei-Hsing Huang, Shaofeng Liang, Daizong Liu, Tai-Hao Wen, Gang Wang, Runwei Guan, Weiping Ding

**发布时间:** 2025-12-15

### GPT解析

### 总结

MMDrive是一个多模态视觉-语言模型框架，通过整合占用地图、LiDAR点云和文本描述，实现了从2D图像理解到3D场景理解的转变，显著提升了自动驾驶场景理解和推理能力。

### 背景

视觉-语言模型通过多源信息融合能够理解和推理复杂的交通场景，是自动驾驶的核心技术。然而，现有模型受限于二维平面图像理解范式，无法有效感知3D空间信息和执行深度语义融合，导致在复杂自动驾驶环境中表现不佳。

### 目的

提出MMDrive框架，将传统的图像理解扩展到广义的3D场景理解框架，以解决现有视觉-语言模型的局限性。

### 方法

MMDrive集成了三种互补模态：占用地图、LiDAR点云和文本场景描述。引入两个新颖组件：1) 面向文本的多模态调制器，根据问题语义线索动态加权各模态贡献；2) 跨模态抽象器，使用可学习抽象令牌生成紧凑的跨模态摘要，突出关键区域和基本语义。

### 主要发现

在DriveLM和NuScenes-QA基准测试上，MMDrive显著超越现有模型：DriveLM上BLEU-4得分为54.56，METEOR得分为41.78；NuScenes-QA上准确得分为62.7%。

### 结论

MMDrive突破了传统仅图像理解的障碍，能够在复杂驾驶环境中实现强大的多模态推理，为可解释的自动驾驶场景理解提供了新基础。

### 翻译

视觉-语言模型通过多源信息融合使复杂交通场景的理解和推理成为可能，使其成为自动驾驶的核心技术。然而，现有的视觉-语言模型受限于二维平面图像理解范式，限制了它们感知3D空间信息和执行深度语义融合的能力，导致在复杂的自动驾驶环境中表现不佳。本研究提出了MMDrive，一个多模态视觉-语言模型框架，将传统的图像理解扩展到广义的3D场景理解框架。MMDrive集成了三种互补模态，包括占用地图、LiDAR点云和文本场景描述。为此，它引入了两个用于自适应跨模态融合和关键信息提取的新组件。具体来说，面向文本的多模态调制器根据问题中的语义线索动态加权每种模态的贡献，引导上下文感知的特征集成。跨模态抽象器使用可学习的抽象令牌生成紧凑的跨模态摘要，突出关键区域和基本语义。在DriveLM和NuScenes-QA基准测试上的综合评估表明，MMDrive在自动驾驶视觉-语言模型方面取得了显著的性能提升，在DriveLM上BLEU-4得分为54.56，METEOR得分为41.78，在NuScenes-QA上准确得分为62.7%。MMDrive有效地突破了传统的仅图像理解的障碍，能够在复杂的驾驶环境中实现强大的多模态推理，为可解释的自动驾驶场景理解提供了新的基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶场景中现有视觉-语言模型受限于2D图像理解范式，无法有效处理3D空间信息和进行深度语义融合的问题。这个问题在现实中非常重要，因为自动驾驶环境复杂动态，仅依靠2D视觉表示缺乏必要的3D空间信息和深度感知，无法满足自动驾驶场景理解的精确要求，特别是在复杂环境中的安全决策方面。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有自动驾驶视觉-语言模型的局限性，发现它们主要遵循传统的'图像理解'范式，无法充分利用3D信息和语义先验。作者借鉴了现有的多模态融合技术和视觉-语言模型架构，但针对自动驾驶场景的特殊需求进行了创新。具体来说，作者整合了占用图、LiDAR点云和场景描述三种互补模态，并设计了TMM和CMA两个新组件，实现了动态多模态融合和关键信息提取。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将传统的'图像理解'范式扩展到通用的'场景理解'范式，通过整合多种互补模态(占用图、LiDAR点云和场景描述)来增强场景理解能力。整体实现流程包括：1)多模态信息编码，使用不同编码器处理各类输入；2)面向文本的多模态调制(TMM)，根据问题语义动态调整各模态权重；3)跨模态抽象(CMA)，生成紧凑的跨模态摘要；4)大语言模型处理，将融合后的输入序列生成最终答案。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)面向文本的多模态调制器(TMM)，根据问题语义动态调整各模态权重；2)跨模态抽象器(CMA)，使用可学习抽象令牌生成紧凑摘要；3)整合占用图、LiDAR点云和场景描述三种互补模态；4)两阶段场景描述生成策略。相比之前的工作，MMDrive突破了仅依赖图像的理解限制，实现了更全面的多模态融合和更高效的信息提取，在复杂驾驶场景中表现出更强的鲁棒性和准确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MMDrive通过整合多种互补模态并引入自适应多模态融合和关键信息提取机制，实现了从传统图像理解到全面场景理解的范式转变，显著提升了自动驾驶场景中视觉-语言模型的性能和鲁棒性。'}


### 论文摘要

Vision-language models enable the understanding and reasoning of complex traffic scenarios through multi-source information fusion, establishing it as a core technology for autonomous driving. However, existing vision-language models are constrained by the image understanding paradigm in 2D plane, which restricts their capability to perceive 3D spatial information and perform deep semantic fusion, resulting in suboptimal performance in complex autonomous driving environments. This study proposes MMDrive, an multimodal vision-language model framework that extends traditional image understanding to a generalized 3D scene understanding framework. MMDrive incorporates three complementary modalities, including occupancy maps, LiDAR point clouds, and textual scene descriptions. To this end, it introduces two novel components for adaptive cross-modal fusion and key information extraction. Specifically, the Text-oriented Multimodal Modulator dynamically weights the contributions of each modality based on the semantic cues in the question, guiding context-aware feature integration. The Cross-Modal Abstractor employs learnable abstract tokens to generate compact, cross-modal summaries that highlight key regions and essential semantics. Comprehensive evaluations on the DriveLM and NuScenes-QA benchmarks demonstrate that MMDrive achieves significant performance gains over existing vision-language models for autonomous driving, with a BLEU-4 score of 54.56 and METEOR of 41.78 on DriveLM, and an accuracy score of 62.7% on NuScenes-QA. MMDrive effectively breaks the traditional image-only understanding barrier, enabling robust multimodal reasoning in complex driving environments and providing a new foundation for interpretable autonomous driving scene understanding.

---

## 9. Audio-Visual Camera Pose Estimation with Passive Scene Sounds and In-the-Wild Video

**论文链接:** [http://arxiv.org/abs/2512.12165v2](http://arxiv.org/abs/2512.12165v2)

**作者:** Daniel Adebi, Sagnik Majumder, Kristen Grauman

**发布时间:** 2025-12-13

### GPT解析

### 总结

本文展示了一种利用被动场景声音作为补充线索来估计相对相机姿态的方法，在视觉条件恶化的情况下表现优于纯视觉方法。

### 背景

理解相机运动是具身感知和3D场景理解的基本问题，现有视觉方法在视觉条件恶化时（如运动模糊或遮挡）表现不佳。

### 目的

探索被动场景声音作为野外视频相对相机姿态估计的补充线索的可能性。

### 方法

引入了一个简单但有效的视听框架，将到达方向(DOA)谱和双耳嵌入集成到最先进的纯视觉姿态估计模型中。

### 主要发现

在两个大型数据集上的实验结果显示，该方法相比强大的视觉基线有持续提升，并且在视觉信息被损坏时具有鲁棒性。

### 结论

据作者所知，这是首次成功利用音频进行真实世界视频中的相对相机姿态估计的工作，确立了日常音频作为解决经典空间挑战的有前景信号。

### 翻译

理解相机运动是具身感知和3D场景理解的一个基本问题。虽然视觉方法已经取得了快速发展，但它们在视觉条件恶化的情况下（如运动模糊或遮挡）往往表现不佳。在这项工作中，我们展示了被动场景声音可以为野外视频的相对相机姿态估计提供补充线索。我们引入了一个简单但有效的视听框架，将到达方向(DOA)谱和双耳嵌入集成到最先进的纯视觉姿态估计模型中。我们在两个大型数据集上的结果表明，相比强大的视觉基线有持续提升，并且在视觉信息被损坏时具有鲁棒性。据我们所知，这是首次成功利用音频进行真实世界视频中的相对相机姿态估计的工作，它将偶然的日常音频确立为解决经典空间挑战的一个意外但有前景的信号。项目：http://vision.cs.utexas.edu/projects/av_camera_pose。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何利用被动场景声音（如环境音、对话、音乐等）来增强相机姿态估计的准确性，特别是在真实世界视频和视觉条件差的情况下（如运动模糊、遮挡、低光照）。这个问题很重要，因为相机姿态估计是AR、VR、机器人和自动驾驶等领域的基础任务，而真实世界视频常常面临视觉退化问题，声音可以提供视觉失效时的替代信息，使系统更加鲁棒。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到在视觉条件差的场景（如昏暗的演唱会），声音仍然能提供丰富的空间信息（如音乐随距离变化），因此提出假设：结合视觉和空间音频可以提高姿态估计准确性。他们设计了一个音频-视觉框架，将方向到达谱和双耳化嵌入集成到Reloc3r视觉模型中。他们借鉴了现有工作，包括MUSIC++算法用于DoA估计、NVAS任务用于双耳特征提取，以及SOFA工具箱用于音频双耳化处理，但创新性地将这些技术组合用于相机姿态估计任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用被动场景声音中蕴含的空间信息作为视觉信息的补充，特别是在视觉信息受损时提供可靠的替代线索。整体流程包括：1)空间音频编码器处理多通道音频，提取DoA谱和双耳特征并融合；2)基于Reloc3r的音频-视觉姿态预测器，通过后期融合将音频特征与视觉特征结合；3)使用交叉注意力建立视觉特征对应关系；4)通过回归头预测相机相对姿态。模型训练包括双耳特征提取器训练（通过NVAS任务）和姿态预测器训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次利用被动场景声音进行真实世界视频中的相机姿态估计；2)设计了结合DoA谱和双耳特征的空间音频编码器；3)在真实世界数据集（Ego-Exo4D和HM3D-SS）上验证方法；4)证明了在视觉信息受损情况下的鲁棒性。相比之前工作，不同之处在于：不同于纯视觉方法，添加了音频模态提高鲁棒性；不同于主动回声定位方法，利用已有被动声音而非主动发射信号；不同于声音定位工作，专注于相机姿态估计而非声音源定位；不同于自监督音频-视觉学习，专注于几何空间任务而非语义任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文首次展示了如何利用真实世界视频中的被动场景声音作为补充线索，显著提高相机姿态估计的准确性和鲁棒性，特别是在视觉信息受损的情况下。'}


### 论文摘要

Understanding camera motion is a fundamental problem in embodied perception and 3D scene understanding. While visual methods have advanced rapidly, they often struggle under visually degraded conditions such as motion blur or occlusions. In this work, we show that passive scene sounds provide complementary cues for relative camera pose estimation for in-the-wild videos. We introduce a simple but effective audio-visual framework that integrates direction-ofarrival (DOA) spectra and binauralized embeddings into a state-of-the-art vision-only pose estimation model. Our results on two large datasets show consistent gains over strong visual baselines, plus robustness when the visual information is corrupted. To our knowledge, this represents the first work to successfully leverage audio for relative camera pose estimation in real-world videos, and it establishes incidental, everyday audio as an unexpected but promising signal for a classic spatial challenge. Project: http://vision.cs.utexas.edu/projects/av_camera_pose.

---

## 10. CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives

**论文链接:** [http://arxiv.org/abs/2512.14696v1](http://arxiv.org/abs/2512.14696v1)

**作者:** Zihan Wang, Jiashun Wang, Jeff Tan, Yiwen Zhao, Jessica Hodgins, Shubham Tulsiani, Deva Ramanan

**发布时间:** 2025-12-16

**备注:** Project page: https://crisp-real2sim.github.io/CRISP-Real2Sim/

### GPT解析

### 总结

CRISP是一种从单目视频中恢复可模拟人体运动和场景几何的方法，通过拟合平面基本元到场景点云重建中，恢复凸起、干净且可模拟的几何形状，并利用人体-场景接触建模和强化学习确保物理合理性。

### 背景

先前关于人体-场景联合重建的工作依赖于数据驱动的先验和无物理循环的联合优化，或者恢复带有噪声的几何形状，这些噪声会导致与场景交互的运动跟踪策略失败。

### 目的

开发一种能够从单目视频中恢复可模拟人体运动和场景几何的方法，以改进人体与场景交互的模拟效果。

### 方法

通过简单的聚类管道（基于深度、法线和流）将平面基本元拟合到场景点云重建中；利用人体-场景接触建模（例如使用人体姿势重建被遮挡的椅子座位）；使用恢复的人体和场景通过强化学习驱动人形控制器，确保重建在物理上是合理的。

### 主要发现

在以人为中心的视频基准测试(EMDB, PROX)上，将运动跟踪失败率从55.2%降低到6.9%；RL模拟吞吐量提高了43%；在casually-captured videos、Internet videos和Sora-generated videos等多种类型视频上验证了有效性。

### 结论

CRISP能够大规模生成物理有效的人体运动和交互环境，极大地推进了机器人和AR/VR领域的现实到模拟应用。

### 翻译

我们介绍了CRISP，一种从单目视频中恢复可模拟人体运动和场景几何的方法。先前关于人体-场景联合重建的工作依赖于数据驱动的先验和无物理循环的联合优化，或者恢复带有噪声的几何形状，这些噪声会导致与场景交互的运动跟踪策略失败。相比之下，我们的关键见解是通过将平面基本元拟合到场景的点云重建中，恢复凸起、干净且可模拟的几何形状，这是通过基于深度、法线和流的简单聚类管道实现的。为了重建可能在交互过程中被遮挡的场景几何，我们利用人体-场景接触建模（例如，我们使用人体姿势来重建被遮挡的椅子座位）。最后，我们通过使用恢复的人体和场景通过强化学习驱动人形控制器，确保人体和场景重建在物理上是合理的。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从单目视频中恢复可模拟的人体运动和场景几何的问题。这个问题在现实中很重要，因为它能帮助我们将普通拍摄的视频转换为可用于物理模拟的3D资产，从而推动机器人学习、物理上合理的角色动画和AR/VR应用的发展。现有方法在处理人体与场景交互时往往产生噪声和伪影，导致运动跟踪策略失败，而精确的物理模拟需要干净、凸起的几何形状。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法在处理人体-场景交互时的局限性，包括依赖数据驱动的先验但没有物理循环，以及重建的几何形状有噪声和伪影。他们的设计思路是使用平面基元拟合点云重建场景，利用人体-场景接触建模重建被遮挡的场景部分，并通过强化学习确保物理合理性。他们借鉴了多项现有工作，包括使用MegaSAM进行相机姿态估计、GVHMR进行人体姿态估计、InteractVLM进行接触预测，以及参考Peng等人(2018)和MaskedMimic的强化学习运动跟踪方法，但将这些技术以创新方式组合起来。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用平面基元表示场景几何，利用人体-场景接触信息重建被遮挡场景部分，并通过物理模拟验证重建结果。整体流程包括：1)初始化阶段使用MegaSAM估计相机参数和深度图，GVHMR估计人体姿态；2)平面基元拟合阶段通过法线计算、K-means聚类、DBSCAN空间聚类、跨帧关联和RANSAC平面拟合将场景分解为约50个凸平面基元；3)接触引导场景重建阶段使用InteractVLM预测接触并应用时间-运动学过滤；4)物理模拟运动跟踪阶段使用强化学习训练策略使模拟人体精确跟踪参考动作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用平面基元而非密集网格表示场景几何；2)利用人体-场景接触信息重建被遮挡场景部分；3)通过强化学习验证重建结果的物理合理性；4)端到端的从单目视频到可模拟资产的完整流程。相比之前的工作(如VideoMimic)，CRISP使用更高效且适合物理模拟的平面基元而非密集网格，明确利用接触信息重建场景，并通过强化学习验证物理合理性，实现了显著更好的性能(运动跟踪失败率从55.2%降至6.9%，RL模拟吞吐量提高43%)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CRISP通过平面基元表示和接触引导重建，实现了从单目视频中恢复物理上合理的人体运动和场景几何，大大提高了真实到模拟的转换质量和效率。'}


### 论文摘要

We introduce CRISP, a method that recovers simulatable human motion and scene geometry from monocular video. Prior work on joint human-scene reconstruction relies on data-driven priors and joint optimization with no physics in the loop, or recovers noisy geometry with artifacts that cause motion tracking policies with scene interactions to fail. In contrast, our key insight is to recover convex, clean, and simulation-ready geometry by fitting planar primitives to a point cloud reconstruction of the scene, via a simple clustering pipeline over depth, normals, and flow. To reconstruct scene geometry that might be occluded during interactions, we make use of human-scene contact modeling (e.g., we use human posture to reconstruct the occluded seat of a chair). Finally, we ensure that human and scene reconstructions are physically-plausible by using them to drive a humanoid controller via reinforcement learning. Our approach reduces motion tracking failure rates from 55.2\% to 6.9\% on human-centric video benchmarks (EMDB, PROX), while delivering a 43\% faster RL simulation throughput. We further validate it on in-the-wild videos including casually-captured videos, Internet videos, and even Sora-generated videos. This demonstrates CRISP's ability to generate physically-valid human motion and interaction environments at scale, greatly advancing real-to-sim applications for robotics and AR/VR.

---

## 11. TUN: Detecting Significant Points in Persistence Diagrams with Deep Learning

**论文链接:** [http://arxiv.org/abs/2512.14274v1](http://arxiv.org/abs/2512.14274v1)

**作者:** Yu Chen, Hongwei Lin

**发布时间:** 2025-12-16

### GPT解析

### 总结

本研究提出了一种名为拓扑理解网络(TUN)的多模态网络，用于自动检测一维持久性图中的重要点，解决了拓扑数据分析在实际应用中的关键挑战。

### 背景

持久性图(PDs)是理解点云底层形状拓扑的强大工具，但识别哪些点编码真实信号仍然具有挑战性，这阻碍了拓扑数据分析在许多需要自动化可靠解释的下游决策应用中的采用。

### 目的

研究一维持久性图的自动显著性检测，提供一种自动化解决方案来识别持久性图中的重要点。

### 方法

提出拓扑理解网络(TUN)，一种多模态网络，结合了增强的PD描述符、自注意力机制、类似PointNet的点云编码器、学习融合、每点分类、稳定的预处理和不平衡感知训练。

### 主要发现

实验表明TUN在检测持久性图中的重要点方面优于经典方法，在实际应用中展现出有效性。

### 结论

TUN为持久性图中重要点的识别提供了自动化且有效的解决方案，这些重要点对下游应用至关重要。

### 翻译

持久性图(PDs)为理解点云底层形状的拓扑提供了强大工具。然而，识别PDs中哪些点编码真实信号仍然具有挑战性。这一挑战直接阻碍了拓扑数据分析在许多应用中的实际采用，在这些应用中，持久性图的自动化和可靠解释对下游决策至关重要。在本文中，我们研究了一维持久性图的自动显著性检测。具体来说，我们提出了拓扑理解网络(TUN)，这是一种多模态网络，结合了增强的PD描述符、自注意力、类似PointNet的点云编码器、学习融合和每点分类，以及稳定的预处理和不平衡感知训练。它为识别PDs中的重要点提供了自动化且有效的解决方案，这些点对下游应用至关重要。实验表明，TUN在检测PDs中的重要点方面优于经典方法，展示了其在实际应用中的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在持久同调图（PDs）中准确识别哪些点代表了真正重要的拓扑特征（显著点）的问题。这个问题很重要，因为PDs是理解点云数据底层拓扑结构的强大工具，广泛应用于机器人、医学影像和计算生物学等领域。准确识别显著点对下游任务（如属性预测和控制）至关重要，而传统方法仅基于持久性判断，无法准确区分重要特征和噪声，阻碍了拓扑数据分析在实际应用中的有效使用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析传统方法（如基于持久性的阈值处理和聚类方法）的局限性，发现它们忽略了拓扑特征与形状拓扑不变量之间的联系。作者设计了一个多模态深度学习框架TUN，将拓扑理解转化为逐点分类问题。该方法借鉴了PointNet风格的点云编码器、自注意力机制以及现有的PD处理方法（如PersLay和Persformer）的思想，但创新性地将这些技术结合起来，同时利用PD的拓扑信息和原始点云的几何信息来解决显著点识别问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多模态深度学习框架，同时利用持久同调图的拓扑信息和原始点云的几何信息，准确识别PD中的显著点。整体流程包括：1) 输入特征提取和处理（增强PD特征、原始点云数据和14维辅助特征）；2) 模型架构（PD编码器、点云编码器、多模态融合模块和显著性分类器）；3) 使用加权Focal Loss处理类别不平衡问题。这种方法突破了传统方法的局限，不仅考虑持久性，还考虑了绝对时间、点间关系和原始几何特性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 构建了第一个用于监督学习1维PD中拓扑特征显著性标记的大规模数据集；2) 提出了TUN，第一个专门用于PD自动化拓扑理解的深度学习框架；3) 设计了增强的PD描述符和全面的辅助特征；4) 采用加权Focal Loss处理类别不平衡。相比传统方法（如2-means聚类和置信集方法），TUN不仅考虑持久性，还整合了几何上下文；相比其他PD处理方法（如PersLay和Persformer），TUN是首个专注于显著点检测的深度学习方法，同时考虑了拓扑和几何信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了拓扑理解网络（TUN），一种多模态深度学习框架，通过融合持久同调图的拓扑信息和原始点云的几何信息，实现了对持久同调图中显著点的自动化准确识别，解决了拓扑数据分析中的一个关键瓶颈问题。'}


### 论文摘要

Persistence diagrams (PDs) provide a powerful tool for understanding the topology of the underlying shape of a point cloud. However, identifying which points in PDs encode genuine signals remains challenging. This challenge directly hinders the practical adoption of topological data analysis in many applications, where automated and reliable interpretation of persistence diagrams is essential for downstream decision-making. In this paper, we study automatic significance detection for one-dimensional persistence diagrams. Specifically, we propose Topology Understanding Net (TUN), a multi-modal network that combines enhanced PD descriptors with self-attention, a PointNet-style point cloud encoder, learned fusion, and per-point classification, alongside stable preprocessing and imbalance-aware training. It provides an automated and effective solution for identifying significant points in PDs, which are critical for downstream applications. Experiments show that TUN outperforms classic methods in detecting significant points in PDs, illustrating its effectiveness in real-world applications.

---

## 12. 4D-RaDiff: Latent Diffusion for 4D Radar Point Cloud Generation

**论文链接:** [http://arxiv.org/abs/2512.14235v1](http://arxiv.org/abs/2512.14235v1)

**作者:** Jimmie Kwok, Holger Caesar, Andras Palffy

**发布时间:** 2025-12-16

### GPT解析

### 总结

本文提出了一种名为4D-RaDiff的新框架，用于生成4D雷达点云数据，解决了雷达标注数据有限的问题。该框架通过扩散模型处理潜在点云表示，能够在对象或场景级别进行条件控制，将未标注边界框转换为高质量雷达标注，并将LiDAR点云转换为逼真雷达场景。

### 背景

汽车雷达因其成本效益和在恶劣天气条件下的鲁棒性，在环境感知方面展现出良好发展前景。然而，标注雷达数据的有限性对基于雷达的感知系统发展构成了重大挑战。

### 目的

提出一个新框架来生成4D雷达点云，用于训练和评估目标检测器，以解决标注雷达数据不足的问题。

### 方法

4D-RaDiff方法不同于基于图像的扩散，它专门考虑雷达点云的稀疏性和独特特性，通过将扩散应用于潜在点云表示来生成数据。在潜在空间中，生成可通过对象或场景级别的条件控制。该方法能够将未标注的边界框转换为高质量雷达标注，并将现有LiDAR点云数据转换为逼真雷达场景。

### 主要发现

实验表明，将4D-RaDiff生成的合成雷达数据作为训练时的数据增强方法，相比仅使用真实数据训练，能持续提高目标检测性能。此外，在合成数据上进行预训练可将所需的标注雷达数据量减少高达90%，同时实现相当的目标检测性能。

### 结论

4D-RaDiff框架有效解决了雷达数据标注有限的问题，通过生成高质量合成数据增强目标检测器性能，同时大幅减少对真实标注数据的需求。

### 翻译

汽车雷达由于其成本效益和在恶劣天气条件下的鲁棒性，在环境感知方面展现出良好的发展前景。然而，标注雷达数据的有限性对推进基于雷达的感知系统构成了重大挑战。为解决这一限制，我们提出了一种新框架来生成4D雷达点云，用于训练和评估目标检测器。与基于图像的扩散不同，我们的方法通过将扩散应用于潜在点云表示，专门考虑了雷达点云的稀疏性和独特特性。在此潜在空间中，生成可通过对象或场景级别的条件控制。所提出的4D-RaDiff能将未标注的边界框转换为高质量的雷达标注，并将现有的LiDAR点云数据转换为逼真的雷达场景。实验证明，在训练中将4D-RaDiff的合成雷达数据作为数据增强方法，与仅使用真实数据训练相比，能持续提高目标检测性能。此外，在我们的合成数据上进行预训练，可在实现相当目标检测性能的同时，将所需的标注雷达数据量减少高达90%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决4D雷达点云数据稀缺的问题。在自动驾驶领域，雷达传感器在恶劣天气条件下比相机和LiDAR更可靠，但获取标注好的雷达数据非常困难、耗时且成本高昂。这种数据 scarcity 限制了基于雷达的感知系统的发展，因此生成高质量的合成雷达数据对推动自动驾驶技术进步至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了雷达数据的独特特性（稀疏性和不规则分布），然后借鉴了多个现有工作：1) 潜在扩散模型(LDMs)的概念，将扩散过程应用于低维潜在空间以提高效率；2) 点云处理中的变分自编码器(VAE)技术，用于将雷达点云映射到规则化的潜在空间；3) LayoutDiffusion中的边界框编码方法，并将其从2D扩展到3D；4) PointPillars用于将LiDAR点云转换为中间表示。作者结合这些技术，针对雷达数据的特殊性进行了创新设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在潜在点云空间而非原始点云空间或图像空间中应用扩散模型，同时分离前景和背景的生成过程。整体流程分为三步：1) 首先训练一个点基础的VAE将雷达点云编码到规则化的潜在点云空间；2) 然后在潜在表示上训练扩散模型，前景生成以3D边界框为条件，背景生成以LiDAR点云为条件；3) 最后将生成的背景和前景点合并，形成完整的合成4D雷达点云，包含Doppler和RCS特征。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出首个直接在点基础雷达表示上操作的潜在扩散框架；2) 引入前景-背景分离生成流程；3) 同时生成雷达特有的Doppler和RCS特征。相比之前工作，不同之处在于：与基于图像的扩散方法不同，考虑了雷达点云的稀疏特性；与直接在原始点云上扩散的方法不同，在潜在空间中操作解决了不规则分布问题；与仅生成空间特征的雷达模型不同，完整保留了雷达特性；与需要雷达数据作为条件的方法不同，能从头生成全新雷达数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '4D-RaDiff通过创新的潜在扩散模型生成了高质量的4D雷达点云，解决了数据稀缺问题，可将标注数据需求减少90%同时提升3D对象检测性能。'}


### 论文摘要

Automotive radar has shown promising developments in environment perception due to its cost-effectiveness and robustness in adverse weather conditions. However, the limited availability of annotated radar data poses a significant challenge for advancing radar-based perception systems. To address this limitation, we propose a novel framework to generate 4D radar point clouds for training and evaluating object detectors. Unlike image-based diffusion, our method is designed to consider the sparsity and unique characteristics of radar point clouds by applying diffusion to a latent point cloud representation. Within this latent space, generation is controlled via conditioning at either the object or scene level. The proposed 4D-RaDiff converts unlabeled bounding boxes into high-quality radar annotations and transforms existing LiDAR point cloud data into realistic radar scenes. Experiments demonstrate that incorporating synthetic radar data of 4D-RaDiff as data augmentation method during training consistently improves object detection performance compared to training on real data only. In addition, pre-training on our synthetic data reduces the amount of required annotated radar data by up to 90% while achieving comparable object detection performance.

---

## 13. CLAIM: Camera-LiDAR Alignment with Intensity and Monodepth

**论文链接:** [http://arxiv.org/abs/2512.14001v1](http://arxiv.org/abs/2512.14001v1)

**作者:** Zhuo Zhang, Yonghui Liu, Meijie Zhang, Feiyang Tan, Yikang Ding

**发布时间:** 2025-12-16

**备注:** Accepted by IROS 2025

### GPT解析

### 总结

本文提出了CLAIM，一种新颖的相机-激光雷达数据对齐方法，利用单深度模型潜力，通过从粗到细的搜索方法，结合基于补丁皮尔逊相关的结构损失和基于互信息的纹理损失，实现高效对齐，无需复杂的数据处理、特征提取或特征匹配步骤。

### 背景

相机-激光雷达校准是计算机视觉和自动驾驶领域的重要问题，现有方法通常需要复杂的数据处理、特征提取或特征匹配步骤，限制了其应用场景和适应性。

### 目的

释放单深度模型在相机-激光雷达校准中的潜力，提出一种简单、适应性强且高性能的对齐方法。

### 方法

CLAIM方法利用初始猜测和图像-激光雷达点云对，采用从粗到细的搜索策略，寻找最小化两种损失的最优变换：基于补丁皮尔逊相关的结构损失和基于互信息的纹理损失，这两种损失作为对齐结果的度量指标。

### 主要发现

CLAIM方法在KITTI、Waymo和MIAS-LCEC等公开数据集上表现出色，与最先进的方法相比具有更好的性能，且方法简单适应性强，适用于大多数场景。

### 结论

CLAIM是一种有效的相机-激光雷达数据对齐方法，通过创新的损失函数和搜索策略，实现了简单、适应性强且高性能的对齐效果。

### 翻译

在本文中，我们释放了强大的单深度模型在相机-激光雷达校准中的潜力，并提出了CLAIM，一种新颖的相机和激光雷达数据对齐方法。给定初始猜测和图像与激光雷达点云对，CLAIM利用从粗到细的搜索方法，寻找最小化基于补丁皮尔逊相关的结构损失和基于互信息的纹理损失的最优变换。这两种损失作为相机-激光雷达对齐结果的良好指标，不需要像大多数方法那样复杂的数据处理、特征提取或特征匹配步骤，使我们的方法简单且适应大多数场景。我们在公开的KITTI、Waymo和MIAS-LCEC数据集上验证了CLAIM，实验结果表明其性能优于最先进的方法。代码可在https://github.com/Tompson11/claim获取。


### 论文摘要

In this paper, we unleash the potential of the powerful monodepth model in camera-LiDAR calibration and propose CLAIM, a novel method of aligning data from the camera and LiDAR. Given the initial guess and pairs of images and LiDAR point clouds, CLAIM utilizes a coarse-to-fine searching method to find the optimal transformation minimizing a patched Pearson correlation-based structure loss and a mutual information-based texture loss. These two losses serve as good metrics for camera-LiDAR alignment results and require no complicated steps of data processing, feature extraction, or feature matching like most methods, rendering our method simple and adaptive to most scenes. We validate CLAIM on public KITTI, Waymo, and MIAS-LCEC datasets, and the experimental results demonstrate its superior performance compared with the state-of-the-art methods. The code is available at https://github.com/Tompson11/claim.

---

## 14. Repurposing 2D Diffusion Models for 3D Shape Completion

**论文链接:** [http://arxiv.org/abs/2512.13991v1](http://arxiv.org/abs/2512.13991v1)

**作者:** Yao He, Youngjoong Kwon, Tiange Xiang, Wenxiao Cai, Ehsan Adeli

**发布时间:** 2025-12-16

### GPT解析

### 总结

研究提出了一种框架，将二维扩散模型适应为从点云完成三维形状的任务，通过引入Shape Atlas这一紧凑的二维表示方法，解决了三维扩散模型面临的数据稀缺和模态差距问题。

### 背景

文本到图像的扩散模型在丰富的二维数据上取得了显著成功，但三维扩散模型由于高质量三维数据集的稀缺以及三维输入与二维潜在空间之间持续的模态差距而发展滞后。

### 目的

克服三维扩散模型面临的限制，使二维扩散模型能够有效地用于三维形状完成，并从有限的3D数据中学习，生成高质量、细节保留的形状完成。

### 方法

引入Shape Atlas，这是一种3D几何的紧凑2D表示，它能够充分利用预训练2D扩散模型的生成能力，并对齐条件输入和输出空间之间的模态，实现更有效的条件设置。这种统一的2D表述使得从有限的3D数据中学习成为可能。

### 主要发现

在PCN和ShapeNet-55数据集上验证了该方法的有效性，能够生成高质量、细节保留的形状完成。此外，展示了从完成的点云创建艺术家创作的网格模型的下游应用。

### 结论

该方法通过统一的2D表述解决了3D扩散模型面临的数据稀缺和模态差距问题，能够从有限的3D数据中学习并生成高质量的3D形状完成，具有实际应用价值。

### 翻译

我们提出了一种框架，该框架将二维扩散模型适应为从点云完成三维形状。虽然文本到图像的扩散模型在丰富的二维数据上取得了显著成功，但由于高质量三维数据集的稀缺以及三维输入与二维潜在空间之间持续的模态差距，三维扩散模型的发展相对滞后。为了克服这些限制，我们引入了Shape Atlas，这是一种三维几何的紧凑二维表示，它(1)能够充分利用预训练二维扩散模型的生成能力，以及(2)对齐条件输入和输出空间之间的模态，实现更有效的条件设置。这种统一的二维表述使得从有限的3D数据中学习成为可能，并产生高质量、细节保留的形状完成。我们在PCN和ShapeNet-55数据集上验证了我们结果的有效性。此外，我们展示了从完成的点云创建艺术家创作的网格模型的下游应用，进一步证明了我们方法的实用性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何利用2D扩散模型来完成3D形状补全的问题，即从不完整的点云恢复完整的3D几何形状。这个问题在现实中非常重要，因为传感器（如LiDAR或RGBD相机）捕获的数据常因遮挡、有限视角和噪声而不完整，而可靠的形状补全是自动驾驶、机器人和AR/VR等应用的关键技术。在研究中，这一挑战也很重要，因为3D扩散模型因高质量3D数据集稀缺和3D与2D空间间的模态差距而发展滞后，无法达到2D扩散模型的性能水平。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到2D扩散模型在图像生成上成功但3D扩散模型受限的问题。他们思考如何利用2D模型的强大能力来解决3D问题，决定将3D形状补全重新表述为2D生成问题。他们借鉴了现有的3D到2D转换方法（特别是球面偏移和平面偏移技术），并采用了预训练的2D扩散模型（如Stable Diffusion）和ControlNet风格的条件控制方法。创新之处在于设计了Shape Atlas作为3D几何的紧凑2D表示，优化了球面偏移算法效率，并引入了条件U-Net和去噪U-Net的双网络架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D形状补全转化为2D生成问题，使用Shape Atlas作为3D几何的紧凑2D表示，解决3D与2D间的模态差距，并利用预训练2D扩散模型的强大生成能力。整体流程包括：1) 将3D点云转换为2D Shape Atlas（通过球面偏移和平面偏移）；2) 训练条件扩散模型（包括条件U-Net和去噪U-Net）；3) 使用扩散模型从不完整Shape Atlas生成完整Shape Atlas；4) 将完整Shape Atlas转换回3D点云。训练过程中结合了扩散损失和多种3D重建损失（如Chamfer距离、InfoCD和网格损失）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出Shape Atlas作为3D几何的紧凑2D表示，保留拓扑信息；2) 优化球面偏移算法，显著降低计算复杂度；3) 设计条件扩散模型的双网络架构；4) 结合多种损失函数提高生成质量。相比之前的工作，这种方法解决了3D与2D间的模态不一致问题，无需额外2D-3D融合模块，更有效利用2D扩散模型的能力，并支持从部分观察的条件生成，而不仅仅是无条件生成。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过引入Shape Atlas作为3D几何的2D表示，成功地将预训练的2D扩散模型重新用于高质量的3D形状补全，解决了3D扩散模型因数据稀缺和模态差距而发展受限的问题。'}


### 论文摘要

We present a framework that adapts 2D diffusion models for 3D shape completion from incomplete point clouds. While text-to-image diffusion models have achieved remarkable success with abundant 2D data, 3D diffusion models lag due to the scarcity of high-quality 3D datasets and a persistent modality gap between 3D inputs and 2D latent spaces. To overcome these limitations, we introduce the Shape Atlas, a compact 2D representation of 3D geometry that (1) enables full utilization of the generative power of pretrained 2D diffusion models, and (2) aligns the modalities between the conditional input and output spaces, allowing more effective conditioning. This unified 2D formulation facilitates learning from limited 3D data and produces high-quality, detail-preserving shape completions. We validate the effectiveness of our results on the PCN and ShapeNet-55 datasets. Additionally, we show the downstream application of creating artist-created meshes from our completed point clouds, further demonstrating the practicality of our method.

---

## 15. TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

**论文链接:** [http://arxiv.org/abs/2512.14698v1](http://arxiv.org/abs/2512.14698v1)

**作者:** Jun Zhang, Teng Wang, Yuying Ge, Yixiao Ge, Xinhao Li, Ying Shan, Limin Wang

**发布时间:** 2025-12-16

**备注:** Project Page: https://timelens-arc-lab.github.io/

### GPT解析

### 总结

本文提出TimeLens，一个系统性研究多模态大语言模型(MLLMs)在视频时序定位(VTG)上表现的框架，从数据质量和算法设计两个维度进行优化，建立了高质量基准数据集和有效的训练方法，实现了超越专有模型的性能。

### 背景

视频时序定位(VTG)是视频理解的核心能力，虽然多模态大语言模型在多种视频理解任务上表现出色，但针对VTG的优化方法研究不足。现有VTG基准存在质量问题，导致评估不可靠。

### 目的

系统性地研究构建具有强大VTG能力的MLLMs，从数据质量和算法设计两个主要维度进行探索，建立可靠的评估标准和高效的训练方法。

### 方法

1) 揭示现有VTG基准质量问题，创建TimeLens-Bench（三个流行基准的严格质量标准重新注释版本）；2) 通过自动重新注释管道创建TimeLens-100K大规模高质量训练数据集；3) 探索算法设计原则，包括交错文本编码、可验证奖励的强化学习(RLVR)训练范式和精心设计的RLVR训练方法。

### 主要发现

与传统基准相比，模型排名发生显著变化，证实了先前评估标准的不可靠性；提出了一系列有意义的见解和有效但高效的实践方法，如交错文本编码和RLVR训练方法。

### 结论

开发的TimeLens模型系列在开源模型中实现了最先进的VTG性能，甚至超过了专有模型如GPT-5和Gemini-2.5-Flash；所有代码、数据和模型将公开以促进未来研究。

### 翻译

本文并未引入新方法，而是为视频时序定位(VTG)建立了一个简单、增量但重要的基线，这是视频理解的核心能力。虽然多模态大语言模型(MLLMs)在多种视频理解任务上表现出色，但针对VTG的优化方法研究不足。在本文中，我们提出TimeLens，一个系统性研究构建具有强大VTG能力的MLLMs的框架，沿着两个主要维度：数据质量和算法设计。我们首先揭示了现有VTG基准中的关键质量问题，并引入TimeLens-Bench，包含三个流行基准的严格质量标准的精心重新注释版本。我们的分析显示与传统基准相比模型排名发生显著变化，证实了先前评估标准的不可靠性。我们还通过自动重新注释管道解决了嘈杂的训练数据问题，创建了TimeLens-100K，这是一个大规模、高质量的训练数据集。基于我们的数据基础，我们对算法设计原则进行了深入探索，得出了一系列有意义的见解和有效但高效的实践方法。这些包括用于时间表示的交错文本编码，一种可验证奖励的强化学习(RLVR)方法作为训练范式，以及精心设计的RLVR训练方法。这些努力最终促成了TimeLens模型系列，这是一组在开源模型中具有最先进VTG性能的MLLMs，甚至超过了GPT-5和Gemini-2.5-Flash等专有模型。所有代码、数据和模型都将被发布以促进未来研究。


### 论文摘要

This paper does not introduce a novel method but instead establishes a straightforward, incremental, yet essential baseline for video temporal grounding (VTG), a core capability in video understanding. While multimodal large language models (MLLMs) excel at various video understanding tasks, the recipes for optimizing them for VTG remain under-explored. In this paper, we present TimeLens, a systematic investigation into building MLLMs with strong VTG ability, along two primary dimensions: data quality and algorithmic design. We first expose critical quality issues in existing VTG benchmarks and introduce TimeLens-Bench, comprising meticulously re-annotated versions of three popular benchmarks with strict quality criteria. Our analysis reveals dramatic model re-rankings compared to legacy benchmarks, confirming the unreliability of prior evaluation standards. We also address noisy training data through an automated re-annotation pipeline, yielding TimeLens-100K, a large-scale, high-quality training dataset. Building on our data foundation, we conduct in-depth explorations of algorithmic design principles, yielding a series of meaningful insights and effective yet efficient practices. These include interleaved textual encoding for time representation, a thinking-free reinforcement learning with verifiable rewards (RLVR) approach as the training paradigm, and carefully designed recipes for RLVR training. These efforts culminate in TimeLens models, a family of MLLMs with state-of-the-art VTG performance among open-source models and even surpass proprietary models such as GPT-5 and Gemini-2.5-Flash. All codes, data, and models will be released to facilitate future research.

---

## 16. Kinetic-Mamba: Mamba-Assisted Predictions of Stiff Chemical Kinetics

**论文链接:** [http://arxiv.org/abs/2512.14471v1](http://arxiv.org/abs/2512.14471v1)

**作者:** Additi Pandey, Liang Wei, Hessam Babaee, George Em Karniadakis

**发布时间:** 2025-12-16

### GPT解析

### 总结

研究团队引入了Kinetic-Mamba，一种基于Mamba的神经算子框架，用于化学动力学建模，该框架结合了神经算子的表达能力和Mamba架构的高效时间建模能力，能够准确预测复杂动力学行为。

### 背景

准确的化学动力学建模对燃烧模拟至关重要，因为它控制着复杂反应路径和热化学状态的演化。

### 目的

开发一种能够准确预测化学动力学行为的神经网络框架，仅使用状态变量的初始条件即可实现高保真度预测。

### 方法

该框架包含三个互补模型：(i)独立Mamba模型预测热化学状态变量的时间演化；(ii)约束Mamba模型强制质量守恒；(iii)基于区域的架构使用两个Mamba模型捕获温度相关区域的动态。还开发了潜在Kinetic-Mamba变体，在简化空间中演化动态并重建完整状态。通过时间分解和递归预测策略评估模型，并在不同分布外数据集上测试外推能力。

### 主要发现

在Syngas和GRI-Mech 3.0反应机制上的计算实验表明，Kinetic-Mamba框架仅使用状态变量的初始条件就能在预测复杂动力学行为方面实现高保真度。

### 结论

Kinetic-Mamba框架能够准确预测化学动力学行为，为燃烧模拟提供了有效的工具。

### 翻译

准确的化学动力学建模对燃烧模拟至关重要，因为它控制着复杂反应路径和热化学状态的演化。在这项工作中，我们引入了Kinetic-Mamba，一种基于Mamba的神经算子框架，它结合了神经算子的表达能力和Mamba架构的高效时间建模能力。该框架包含三个互补模型：(i)一个独立Mamba模型，根据给定初始条件预测热化学状态变量的时间演化；(ii)一个约束Mamba模型，在学习状态动态的同时强制执行质量守恒；以及(iii)一个基于区域的架构，采用两个独立Mamba模型来捕获温度相关区域的动态。我们还开发了一个潜在Kinetic-Mamba变体，在简化的潜在空间中演化动态，并在物理流形上重建完整状态。我们使用时间分解和递归预测策略评估了Kinetic-Mamba的准确性和鲁棒性。我们进一步评估了模型在不同分布外数据集上的外推能力。在Syngas和GRI-Mech 3.0反应机制上的计算实验表明，我们的框架仅使用状态变量的初始条件就能在预测复杂动力学行为方面实现高保真度。


### 论文摘要

Accurate chemical kinetics modeling is essential for combustion simulations, as it governs the evolution of complex reaction pathways and thermochemical states. In this work, we introduce Kinetic-Mamba, a Mamba-based neural operator framework that integrates the expressive power of neural operators with the efficient temporal modeling capabilities of Mamba architectures. The framework comprises three complementary models: (i) a standalone Mamba model that predicts the time evolution of thermochemical state variables from given initial conditions; (ii) a constrained Mamba model that enforces mass conservation while learning the state dynamics; and (iii) a regime-informed architecture employing two standalone Mamba models to capture dynamics across temperature-dependent regimes. We additionally develop a latent Kinetic-Mamba variant that evolves dynamics in a reduced latent space and reconstructs the full state on the physical manifold. We evaluate the accuracy and robustness of Kinetic-Mamba using both time-decomposition and recursive-prediction strategies. We further assess the extrapolation capabilities of the model on varied out-of-distribution datasets. Computational experiments on Syngas and GRI-Mech 3.0 reaction mechanisms demonstrate that our framework achieves high fidelity in predicting complex kinetic behavior using only the initial conditions of the state variables.

---

## 17. Zoom-Zero: Reinforced Coarse-to-Fine Video Understanding via Temporal Zoom-in

**论文链接:** [http://arxiv.org/abs/2512.14273v1](http://arxiv.org/abs/2512.14273v1)

**作者:** Xiaoqian Shen, Min-Hung Chen, Yu-Chiang Frank Wang, Mohamed Elhoseiny, Ryo Hachiuma

**发布时间:** 2025-12-16

**备注:** Project page: https://xiaoqian-shen.github.io/Zoom-Zero/

### GPT解析

### 总结

本文提出了Zoom-Zero框架，解决大型视频语言模型在时间感知方面的局限性，通过从粗到细的方法提高视频问答任务中的时间定位准确性并减少幻觉。

### 背景

Grounded video question answering (GVQA)旨在定位视频中相关时间段并生成准确答案，但大型视频语言模型(LVLMs)存在时间感知有限的问题。基于Group Relative Policy Optimization (GRPO)的方法虽试图改进时间定位，但仍难以忠实地将答案定位到相关视频证据，导致时间错位和幻觉。

### 目的

解决GRPO在GVQA任务中的局限性，提高时间定位准确性，减少幻觉现象，增强长视频理解能力。

### 方法

提出Zoom-Zero框架，采用从粗到细的方法：首先定位查询相关时间段，然后时间放大到最显著帧进行精细视觉验证。包含两个关键创新：(1)放大精度奖励，验证时间定位预测保真度；(2)令牌选择性信用分配，将奖励归因于负责时间定位或答案生成的令牌。

### 主要发现

Zoom-Zero在NExT-GQA上提高时间定位5.2%，在ReXTime上提高4.6%，平均答案准确率提高2.4%。在推理过程中的从粗到细放大方法在长视频基准测试上平均提高6.4%。

### 结论

Zoom-Zero方法显著改进了基于视频的问答任务，通过从粗到细的放大方法能够在保持全局上下文的同时保留关键视觉细节，有效提升了时间定位和答案生成的准确性。

### 翻译

Grounded video question answering (GVQA)旨在定位视频中相关的时间段并针对给定问题生成准确答案；然而，大型视频语言模型(LVLMs)表现出有限的时间感知能力。虽然基于Group Relative Policy Optimization (GRPO)的现有方法试图提高时间定位，它们仍然难以将答案忠实地定位到相关视频证据中，导致时间错位和幻觉。在这项工作中，我们提出了Zoom-Zero，一个从粗到细的框架，首先定位查询相关的时间段，然后时间放大到最显著的帧进行更精细的视觉验证。我们的方法通过两个关键创新解决了GRPO在GVQA任务中的局限性：(i)放大精度奖励，验证时间定位预测的保真度并促进对已定位帧的细粒度视觉验证；(ii)令牌选择性信用分配，将奖励归因于负责时间定位或答案生成的令牌，减轻GRPO处理多面奖励信号的问题。我们提出的方法推进了基于视频的问答任务，在NExT-GQA上提高时间定位5.2%，在ReXTime上提高4.6%，同时平均答案准确率提高2.4%。此外，推理过程中的从粗到细放大方法通过在不损害全局上下文的情况下保留关键视觉细节，进一步受益于长视频理解，在长视频基准测试上平均提高6.4%。


### 论文摘要

Grounded video question answering (GVQA) aims to localize relevant temporal segments in videos and generate accurate answers to a given question; however, large video-language models (LVLMs) exhibit limited temporal awareness. Although existing approaches based on Group Relative Policy Optimization (GRPO) attempt to improve temporal grounding, they still struggle to faithfully ground their answers in the relevant video evidence, leading to temporal mislocalization and hallucinations. In this work, we present Zoom-Zero, a coarse-to-fine framework that first localizes query-relevant segments and then temporally zooms into the most salient frames for finer-grained visual verification. Our method addresses the limits of GRPO for the GVQA task with two key innovations: (i) a zoom-in accuracy reward that validates the fidelity of temporal grounding prediction and facilitates fine-grained visual verification on grounded frames; (ii) token-selective credit assignment, which attributes rewards to the tokens responsible for temporal localization or answer generation, mitigating GRPO's issue in handling multi-faceted reward signals. Our proposed method advances grounded video question answering, improving temporal grounding by 5.2\% on NExT-GQA and 4.6\% on ReXTime, while also enhancing average answer accuracy by 2.4\%. Additionally, the coarse-to-fine zoom-in during inference further benefits long-form video understanding by preserving critical visual details without compromising global context, yielding an average improvement of 6.4\% on long-video benchmarks.

---

## 18. KFS-Bench: Comprehensive Evaluation of Key Frame Sampling in Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2512.14017v1](http://arxiv.org/abs/2512.14017v1)

**作者:** Zongyao Li, Kengo Ishida, Satoshi Yamazaki, Xiaotong Ji, Jianquan Liu

**发布时间:** 2025-12-16

**备注:** WACV2026

### GPT解析

### 总结

本文提出了KFS-Bench，首个针对长视频问答中关键帧采样的基准测试，具有多场景注释功能，能够直接评估采样策略。研究发现采样精度、场景覆盖和采样平衡都是影响问答性能的关键因素，并提出了自适应平衡采样方法。

### 背景

关键帧采样对高效的长视频理解至关重要。在长视频问答中，选择信息丰富的帧可以使多模态大语言模型提高准确性和效率。先前的工作仅通过问答准确率间接评估帧选择质量，存在局限性。

### 目的

创建一个基准测试来直接评估长视频问答中的关键帧采样策略，解决先前工作只能间接评估帧选择质量的问题。

### 方法

提供每个问题所需的多不重叠场景的真实注释；设计了一个新的采样质量指标，与问答准确率相关；开发了一种新的关键帧采样方法，利用问题-视频相关性来平衡采样多样性与问题-帧相似性，提高相关场景的覆盖范围。

### 主要发现

采样精度不是唯一影响因素；场景覆盖和采样平衡也是影响问答性能的关键因素；自适应平衡采样方法在关键帧采样和问答性能方面都取得了优异的效果。

### 结论

KFS-Bench基准测试为直接评估长视频问答中的关键帧采样策略提供了新工具，通过多场景注释实现了更直接和稳健的评估。研究强调了采样质量不仅仅是精度问题，还包括场景覆盖和平衡性。

### 翻译

我们提出了KFS-Bench，这是首个针对长视频问答中关键帧采样的基准测试，具有多场景注释功能，能够直接且稳健地评估采样策略。关键帧采样对高效的长视频理解至关重要。在长视频问答中，选择信息丰富的帧可以使多模态大语言模型提高准确性和效率。KFS-Bench解决了先前工作仅通过问答准确率间接评估帧选择质量的局限性。通过提供每个问题所需的多不重叠场景的真实注释，KFS-Bench使我们能够直接分析不同采样方法如何捕捉整个长视频中的基本内容。使用KFS-Bench，我们对关键帧采样方法进行了全面研究，并发现不仅采样精度，而且场景覆盖和采样平衡都是影响问答性能的关键因素。关于所有这些因素，我们设计了一个与问答准确率相关的新型采样质量指标。此外，我们开发了一种新的关键帧采样方法，利用问题-视频相关性来平衡采样多样性与问题-帧相似性，从而提高相关场景的覆盖范围。我们的自适应平衡采样方法在关键帧采样和问答性能方面都取得了优异的效果。该基准测试可在https://github.com/NEC-VID/KFS-Bench获取。


### 论文摘要

We propose KFS-Bench, the first benchmark for key frame sampling in long video question answering (QA), featuring multi-scene annotations to enable direct and robust evaluation of sampling strategies. Key frame sampling is crucial for efficient long-form video understanding. In long video QA, selecting informative frames enables multimodal large language models (MLLMs) to improve both accuracy and efficiency. KFS-Bench addresses the limitation of prior works that only indirectly assess frame selection quality via QA accuracy. By providing ground-truth annotations of multiple disjoint scenes required per question, KFS-Bench allows us to directly analyze how different sampling approaches capture essential content across an entire long video. Using KFS-Bench, we conduct a comprehensive study of key frame sampling methods and identify that not only sampling precision but also scene coverage and sampling balance are the key factors influencing QA performance. Regarding all the factors, we design a novel sampling quality metric that correlates with QA accuracy. Furthermore, we develop a novel key frame sampling method that leverages question-video relevance to balance sampling diversity against question-frame similarity, thereby improving coverage of relevant scenes. Our adaptively balanced sampling approach achieves superior performance in both key frame sampling and QA performance. The benchmark is available at https://github.com/NEC-VID/KFS-Bench.

---

## 19. FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos

**论文链接:** [http://arxiv.org/abs/2512.14601v1](http://arxiv.org/abs/2512.14601v1)

**作者:** Zhaolun Li, Jichang Li, Yinqi Cai, Junye Chen, Xiaonan Luo, Guanbin Li, Rushi Lan

**发布时间:** 2025-12-16

### GPT解析

### 总结

本文提出了FakeRadar，一种新型深度伪造视频检测框架，旨在解决现实场景中跨领域泛化的挑战。

### 背景

现有检测方法通常依赖特定篡改线索，在已知伪造类型上表现良好，但对新兴篡改技术表现出严重局限性，无法有效适应未见过的篡改模式。

### 目的

利用大规模预训练模型（如CLIP）主动探测特征空间，明确显示真实视频、已知伪造品和未见操纵之间的分布差距。

### 方法

FakeRadar引入了'伪造异常值探测'，采用动态子聚类建模和聚类条件异常值生成来合成估计子聚类边界附近的异常样本；设计了'异常值引导的三重训练'，使用异常值驱动的对比学习和异常值条件的交叉熵损失来优化检测器。

### 主要发现

实验表明，在各种深度伪造视频检测基准数据集上，FakeRadar都优于现有方法，特别是在跨领域评估中，通过处理各种新兴的操纵技术。

### 结论

FakeRadar通过主动探测特征空间和生成异常样本，有效解决了深度伪造视频检测中的跨领域泛化问题。

### 翻译

在本文中，我们提出了FakeRadar，一种新型深度伪造视频检测框架，旨在解决现实场景中跨领域泛化的挑战。现有的检测方法通常依赖于特定的篡改线索，在已知的伪造类型上表现良好，但对新兴的篡改技术表现出严重的局限性。这种泛化能力差的原因是它们无法有效适应未见过的篡改模式。为了克服这个问题，我们利用大规模预训练模型（如CLIP）来主动探测特征空间，明确显示真实视频、已知伪造品和未见操纵之间的分布差距。具体来说，FakeRadar引入了'伪造异常值探测'，它采用动态子聚类建模和聚类条件异常值生成来合成估计子聚类边界附近的异常样本，模拟已知操纵类型之外的新型伪造伪影。此外，我们设计了'异常值引导的三重训练'，它使用提出的异常值驱动的对比学习和异常值条件的交叉熵损失来优化检测器，以区分真实、伪造和异常样本。实验表明，在各种深度伪造视频检测的基准数据集上，FakeRadar都优于现有方法，特别是在跨领域评估中，通过处理各种新兴的操纵技术。


### 论文摘要

In this paper, we propose FakeRadar, a novel deepfake video detection framework designed to address the challenges of cross-domain generalization in real-world scenarios. Existing detection methods typically rely on manipulation-specific cues, performing well on known forgery types but exhibiting severe limitations against emerging manipulation techniques. This poor generalization stems from their inability to adapt effectively to unseen forgery patterns. To overcome this, we leverage large-scale pretrained models (e.g. CLIP) to proactively probe the feature space, explicitly highlighting distributional gaps between real videos, known forgeries, and unseen manipulations. Specifically, FakeRadar introduces Forgery Outlier Probing, which employs dynamic subcluster modeling and cluster-conditional outlier generation to synthesize outlier samples near boundaries of estimated subclusters, simulating novel forgery artifacts beyond known manipulation types. Additionally, we design Outlier-Guided Tri-Training, which optimizes the detector to distinguish real, fake, and outlier samples using proposed outlier-driven contrastive learning and outlier-conditioned cross-entropy losses. Experiments show that FakeRadar outperforms existing methods across various benchmark datasets for deepfake video detection, particularly in cross-domain evaluations, by handling the variety of emerging manipulation techniques.

---

## 20. SuperCLIP: CLIP with Simple Classification Supervision

**论文链接:** [http://arxiv.org/abs/2512.14480v1](http://arxiv.org/abs/2512.14480v1)

**作者:** Weiheng Zhao, Zilong Huang, Jiashi Feng, Xinggang Wang

**发布时间:** 2025-12-16

**备注:** Accepted by NeurIPS 2025. Code: https://github.com/hustvl/SuperCLIP

### GPT解析

### 总结

SuperCLIP通过在对比学习中添加基于分类的监督，解决了CLIP模型未能充分利用文本细粒度语义信号的问题，显著提升了视觉-文本对齐能力。

### 背景

CLIP模型在视觉语言任务中表现出强大的泛化能力，但研究表明CLIP类模型未能充分利用文本中的细粒度语义信号，尤其是在处理长而详细的描述时。

### 目的

解决CLIP模型在细粒度视觉-文本对齐方面的局限性，提高模型在视觉语言任务中的性能。

### 方法

提出SuperCLIP框架，在视觉编码器中添加轻量级线性层，利用标记级别线索增强视觉-文本对齐，仅需增加0.077%的总FLOPs，不需要额外标注数据。

### 主要发现

SuperCLIP在零样本分类、图像-文本检索和纯视觉任务中都有持续改进；这种改进在使用原始网络数据或丰富的重新描述数据进行训练时都成立；SuperCLIP减轻了CLIP在小批量训练中的性能下降问题。

### 结论

SuperCLIP是一个简单而有效的框架，能够增强CLIP模型在视觉语言任务中的性能，特别是在细粒度语义对齐方面。

### 翻译

对比语言-图像预训练（CLIP）通过在共享嵌入空间中对齐图像和文本，在视觉语言任务中实现了强大的泛化能力。然而，最近的研究表明，CLIP类模型仍然未能充分利用文本中的细粒度语义信号，当处理长而详细的描述时，这一问题变得更加明显。这源于CLIP的训练目标，它只优化全局图像-文本相似性，而忽略了标记级别的监督，限制了其实现细粒度视觉-文本对齐的能力。为解决这一问题，我们提出了SuperCLIP，一个简单而有效的框架，通过基于分类的监督增强对比学习。通过仅在视觉编码器中添加一个轻量级线性层，SuperCLIP利用标记级别线索来增强视觉-文本对齐，仅增加了0.077%的总FLOPs，且不需要额外的标注数据。实验表明，SuperCLIP在零样本分类、图像-文本检索和纯视觉任务中都有持续改进。这些改进在模型使用原始网络数据或丰富的重新描述数据进行训练时都成立，证明了SuperCLIP在两种情况下都能恢复文本监督的能力。此外，SuperCLIP通过基于分类的监督减轻了CLIP在小批量训练中的性能下降问题。代码和模型将开源。


### 论文摘要

Contrastive Language-Image Pretraining (CLIP) achieves strong generalization in vision-language tasks by aligning images and texts in a shared embedding space. However, recent findings show that CLIP-like models still underutilize fine-grained semantic signals in text, and this issue becomes even more pronounced when dealing with long and detailed captions. This stems from CLIP's training objective, which optimizes only global image-text similarity and overlooks token-level supervision - limiting its ability to achieve fine-grained visual-text alignment. To address this, we propose SuperCLIP, a simple yet effective framework that augments contrastive learning with classification-based supervision. By adding only a lightweight linear layer to the vision encoder, SuperCLIP leverages token-level cues to enhance visual-textual alignment - with just a 0.077% increase in total FLOPs, and no need for additional annotated data. Experiments show that SuperCLIP consistently improves zero-shot classification, image-text retrieval, and purely visual tasks. These gains hold regardless of whether the model is trained on original web data or rich re-captioned data, demonstrating SuperCLIP's ability to recover textual supervision in both cases. Furthermore, SuperCLIP alleviates CLIP's small-batch performance drop through classification-based supervision that avoids reliance on large batch sizes. Code and models will be made open source.

---

## 21. PSMamba: Progressive Self-supervised Vision Mamba for Plant Disease Recognition

**论文链接:** [http://arxiv.org/abs/2512.14309v1](http://arxiv.org/abs/2512.14309v1)

**作者:** Abdullah Al Mamun, Miaohua Zhang, David Ahmedt-Aristizabal, Zeeshan Hayder, Mohammad Awrangjeb

**发布时间:** 2025-12-16

### GPT解析

### 总结

PSMamba是一种渐进式自监督框架，结合Vision Mamba的高效序列建模和双学生层次蒸馏策略，有效捕捉植物病害图像的层次化、多尺度病变模式，在三个基准数据集上表现优于最先进的SSL方法。

### 背景

自监督学习已成为无标注表示学习的强大范式，但现有框架主要关注全局对齐，难以捕捉植物病害图像特有的层次化、多尺度病变模式。

### 目的

提出PSMamba框架，解决现有SSL方法在捕捉植物病害图像层次化、多尺度病变模式方面的不足，提高在领域迁移和细粒度场景中的准确性和鲁棒性。

### 方法

PSMamba采用共享的全局教师和两个专门的学生：一个处理中等尺度视图捕捉病变分布和静脉结构，另一个专注于局部视图捕捉纹理不规则性和早期病变等细粒度线索，通过多粒度监督和一致性损失实现跨尺度对齐。

### 主要发现

在三个基准数据集上的实验表明，PSMamba始终优于最先进的SSL方法，在领域迁移和细粒度场景中提供更高的准确性和鲁棒性。

### 结论

PSMamba通过整合Vision Mamba的高效序列建模和双学生层次蒸馏策略，有效解决了植物病害图像中层次化、多尺度病变模式捕捉的问题，提升了自监督学习在植物病害识别中的性能。

### 翻译

自监督学习已成为一种无需人工标注的强大表征学习范式。然而，大多数现有框架专注于全局对齐，难以捕捉植物病害图像特有的层次化、多尺度病变模式。为解决这一差距，我们提出了PSMamba，一种渐进式自监督框架，将Vision Mamba(VM)的高效序列建模与双学生层次蒸馏策略相结合。与传统单一教师-学生设计不同，PSMamba采用共享的全局教师和两个专门的学生：一个处理中等尺度视图以捕捉病变分布和静脉结构，而另一个专注于局部视图以捕捉纹理不规则性和早期病变等细粒度线索。这种多粒度监督促进了上下文和详细表示的联合学习，一致性损失确保了跨尺度对齐的连贯性。在三个基准数据集上的实验表明，PSMamba始终优于最先进的SSL方法，在领域迁移和细粒度场景中提供更高的准确性和鲁棒性。


### 论文摘要

Self-supervised Learning (SSL) has become a powerful paradigm for representation learning without manual annotations. However, most existing frameworks focus on global alignment and struggle to capture the hierarchical, multi-scale lesion patterns characteristic of plant disease imagery. To address this gap, we propose PSMamba, a progressive self-supervised framework that integrates the efficient sequence modelling of Vision Mamba (VM) with a dual-student hierarchical distillation strategy. Unlike conventional single teacher-student designs, PSMamba employs a shared global teacher and two specialised students: one processes mid-scale views to capture lesion distributions and vein structures, while the other focuses on local views to capture fine-grained cues such as texture irregularities and early-stage lesions. This multi-granular supervision facilitates the joint learning of contextual and detailed representations, with consistency losses ensuring coherent cross-scale alignment. Experiments on three benchmark datasets show that PSMamba consistently outperforms state-of-the-art SSL methods, delivering superior accuracy and robustness in both domain-shifted and fine-grained scenarios.

---

## 22. Understanding the Gain from Data Filtering in Multimodal Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2512.14230v1](http://arxiv.org/abs/2512.14230v1)

**作者:** Divyansh Pareek, Sewoong Oh, Simon S. Du

**发布时间:** 2025-12-16

**备注:** 40 pages, 8 figures, 1 table. This work is accepted to the Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025

### GPT解析

### 总结

现代多模态表示学习的成功依赖于互联网规模数据集，而数据筛选已成为训练流程中的关键步骤。基于教师模型的筛选方法利用预训练模型计算质量分数，已被证明是一种有效的解决方案。理论分析表明，数据筛选可提供可证明的好处，显著降低对比学习的误差。

### 背景

现代多模态表示学习的成功依赖于互联网规模的数据集。然而，由于大量原始网络数据质量低下，数据筛选已成为训练流程中的关键步骤。

### 目的

解释基于教师模型筛选方法的经验成功，并表征在标准双模态数据生成模型下经过筛选的对比学习性能。

### 方法

使用基于训练模型的筛选方法（即基于教师模型的筛选），利用预训练模型计算质量分数，并在线性对比学习设置下进行分析。

### 主要发现

未筛选时的误差上下界为1/(η√n)，其中η是正确匹配模态的数据比例，n是成对样本数量；使用基于教师模型的筛选后，在大η情况下误差上界为1/√(ηn)，在小η情况下误差上界为1/√n。

### 结论

数据筛选提供了可证明的好处，基于教师模型的筛选方法在不同数据质量情况下都能有效降低误差。

### 翻译

现代多模态表示学习的成功依赖于互联网规模的数据集。由于大量原始网络数据质量低下，数据筛选已成为训练流程中的关键步骤。利用训练好的模型进行筛选（即基于教师模型的筛选）已成为一种成功的解决方案，它利用预训练模型计算质量分数。为了解释基于教师模型筛选方法的经验成功，我们在标准双模态数据生成模型下表征了经过筛选的对比学习的性能。将η∈(0,1]表示为n个成对样本中正确匹配模态的数据比例，我们利用线性对比学习设置展示了数据筛选的可证明好处：(i)未筛选时的误差上下界为1/(η√n)，(ii)使用基于教师模型的筛选后，在大η情况下误差上界为1/√(ηn)，在小η情况下误差上界为1/√n。


### 论文摘要

The success of modern multimodal representation learning relies on internet-scale datasets. Due to the low quality of a large fraction of raw web data, data curation has become a critical step in the training pipeline. Filtering using a trained model (i.e., teacher-based filtering) has emerged as a successful solution, leveraging a pre-trained model to compute quality scores. To explain the empirical success of teacher-based filtering, we characterize the performance of filtered contrastive learning under the standard bimodal data generation model. Denoting $η\in(0,1]$ as the fraction of data with correctly matched modalities among $n$ paired samples, we utilize a linear contrastive learning setup to show a provable benefit of data filtering: $(i)$ the error without filtering is upper and lower bounded by $\frac{1}{η\sqrt{n}}$, and $(ii)$ the error with teacher-based filtering is upper bounded by $\frac{1}{\sqrt{ηn}}$ in the large $η$ regime, and by $\frac{1}{\sqrt{n}}$ in the small $η$ regime.

---

## 23. Joint Multimodal Contrastive Learning for Robust Spoken Term Detection and Keyword Spotting

**论文链接:** [http://arxiv.org/abs/2512.14115v1](http://arxiv.org/abs/2512.14115v1)

**作者:** Ramesh Gundluru, Shubham Gupta, Sri Rama Murty K

**发布时间:** 2025-12-16

### GPT解析

### 总结

该研究提出了一种联合多模态对比学习框架，用于改进声学词嵌入在语音检索任务中的性能，解决了现有方法的局限性

### 背景

声学词嵌入(AWEs)提高了语音检索任务(如口语词检测(STD)和关键词定位(KWS))的效率

### 目的

解决现有AWE方法的局限性，包括单模态监督、音频-音频和音频-文本对齐的分离优化，以及需要特定任务模型的问题

### 方法

提出了一种联合多模态对比学习框架，在共享嵌入空间中统一了声学和跨模态监督。该方法同时优化：(i)基于CLAP损失的音频-文本对比学习，对齐音频和文本表示；(ii)通过深度词辨别(DWD)损失的音频-音频对比学习，增强类内紧凑性和类间分离性

### 主要发现

所提出的方法在词辨别任务上优于现有的AWE基线，同时灵活支持STD和KWS任务

### 结论

据我们所知，这是首个此类综合方法

### 翻译

声学词嵌入(AWEs)提高了语音检索任务(如口语词检测(STD)和关键词定位(KWS))的效率。然而，现有方法存在局限性，包括单模态监督、音频-音频和音频-文本对齐的分离优化，以及需要特定任务模型。为解决这些不足，我们提出了一种联合多模态对比学习框架，在共享嵌入空间中统一了声学和跨模态监督。我们的方法同时优化：(i)受CLAP损失启发的音频-文本对比学习，对齐音频和文本表示；(ii)通过深度词辨别(DWD)损失的音频-音频对比学习，增强类内紧凑性和类间分离性。所提出的方法在词辨别任务上优于现有的AWE基线，同时灵活支持STD和KWS。据我们所知，这是首个此类综合方法


### 论文摘要

Acoustic Word Embeddings (AWEs) improve the efficiency of speech retrieval tasks such as Spoken Term Detection (STD) and Keyword Spotting (KWS). However, existing approaches suffer from limitations, including unimodal supervision, disjoint optimization of audio-audio and audio-text alignment, and the need for task-specific models. To address these shortcomings, we propose a joint multimodal contrastive learning framework that unifies both acoustic and cross-modal supervision in a shared embedding space. Our approach simultaneously optimizes: (i) audio-text contrastive learning, inspired by the CLAP loss, to align audio and text representations and (ii) audio-audio contrastive learning, via Deep Word Discrimination (DWD) loss, to enhance intra-class compactness and inter-class separation. The proposed method outperforms existing AWE baselines on word discrimination task while flexibly supporting both STD and KWS. To our knowledge, this is the first comprehensive approach of its kind.

---

## 24. AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation

**论文链接:** [http://arxiv.org/abs/2512.14047v1](http://arxiv.org/abs/2512.14047v1)

**作者:** Kaike Zhang, Qi Cao, Fei Sun, Xinran Liu

**发布时间:** 2025-12-16

### GPT解析

### 总结

论文提出了一种名为AsarRec的自适应增强框架，用于解决顺序推荐系统中的噪声问题，通过学习生成变换矩阵来优化推荐性能。

### 背景

顺序推荐系统在建模用户动态偏好和捕捉项目转换模式方面表现出色，但真实世界用户行为往往因人为错误、不确定性和行为模糊性而存在噪声，导致推荐性能下降。

### 目的

解决现有自监督学习方法中依赖静态增强策略的问题，提出一种能够自适应选择增强类型的框架，以提高顺序推荐系统的鲁棒性。

### 方法

将基本增强操作统一为结构化变换矩阵，通过将用户序列编码为概率转移矩阵并使用可微的半-Sinkhorn算法投影到硬半双随机矩阵来学习生成变换矩阵，同时联合优化多样性、语义不变性和信息量三个目标。

### 主要发现

现有方法依赖的静态增强策略存在两个关键挑战：最优增强类型在不同场景下可能有显著差异；不适当的增强甚至可能降低推荐性能。AsarRec在不同噪声水平下的三个基准数据集上验证了其有效性。

### 结论

AsarRec框架表现出卓越的鲁棒性和一致的改进，能够有效解决顺序推荐系统中的噪声问题。

### 翻译

顺序推荐系统在建模用户动态偏好和捕捉项目转换模式方面已展现出强大的能力。然而，由于人为错误、不确定性和行为模糊性等因素，真实世界的用户行为往往存在噪声，这可能导致推荐性能下降。为解决这个问题，最近的方法广泛采用自监督学习，特别是对比学习，通过生成用户交互序列的扰动视图并最大化它们之间的互信息来提高模型鲁棒性。然而，这些方法严重依赖其预定义的静态增强策略（一旦选定，增强类型保持不变）来构建增强视图，导致两个关键挑战：（1）最优增强类型在不同场景下可能有显著差异；（2）不适当的增强甚至可能降低推荐性能，限制了自监督学习的有效性。为了克服这些局限性，我们提出了一个自适应增强框架。我们首先通过结构化变换矩阵将现有的基本增强操作统一为一个统一的公式。在此基础上，我们引入了AsarRec（用于鲁棒顺序推荐的自适应顺序增强），它通过将用户序列编码为概率转移矩阵，并使用可微的半-Sinkhorn算法将其投影到硬半双随机矩阵来学习生成变换矩阵。为确保学习到的增强有利于下游性能，我们联合优化了三个目标：多样性、语义不变性和信息量。在三个不同噪声水平下的基准数据集上进行的大量实验验证了AsarRec的有效性，展示了其卓越的鲁棒性和一致的改进。


### 论文摘要

Sequential recommender systems have demonstrated strong capabilities in modeling users' dynamic preferences and capturing item transition patterns. However, real-world user behaviors are often noisy due to factors such as human errors, uncertainty, and behavioral ambiguity, which can lead to degraded recommendation performance. To address this issue, recent approaches widely adopt self-supervised learning (SSL), particularly contrastive learning, by generating perturbed views of user interaction sequences and maximizing their mutual information to improve model robustness. However, these methods heavily rely on their pre-defined static augmentation strategies~(where the augmentation type remains fixed once chosen) to construct augmented views, leading to two critical challenges: (1) the optimal augmentation type can vary significantly across different scenarios; (2) inappropriate augmentations may even degrade recommendation performance, limiting the effectiveness of SSL. To overcome these limitations, we propose an adaptive augmentation framework. We first unify existing basic augmentation operations into a unified formulation via structured transformation matrices. Building on this, we introduce AsarRec (Adaptive Sequential Augmentation for Robust Sequential Recommendation), which learns to generate transformation matrices by encoding user sequences into probabilistic transition matrices and projecting them into hard semi-doubly stochastic matrices via a differentiable Semi-Sinkhorn algorithm. To ensure that the learned augmentations benefit downstream performance, we jointly optimize three objectives: diversity, semantic invariance, and informativeness. Extensive experiments on three benchmark datasets under varying noise levels validate the effectiveness of AsarRec, demonstrating its superior robustness and consistent improvements.

---

## 25. Unleashing the Power of Image-Tabular Self-Supervised Learning via Breaking Cross-Tabular Barriers

**论文链接:** [http://arxiv.org/abs/2512.14026v1](http://arxiv.org/abs/2512.14026v1)

**作者:** Yibing Fu, Yunpeng Zhao, Zhitao Zeng, Cheng Chen, Yueming Jin

**发布时间:** 2025-12-16

### GPT解析

### 总结

本文提出了CITab，一种新型跨表格多模态自监督学习框架，通过语义感知的表格建模机制和原型引导的线性层混合模块，有效整合医学图像和表格数据，在阿尔茨海默病诊断任务上优于现有方法。

### 背景

多模态学习整合医学图像和表格数据已显著推进临床决策，但现有自监督学习方法常局限于特定数据队列，因其僵化的表格建模机制难以处理异构表格数据，形成跨表格障碍。

### 目的

开发一种能够学习强大跨表格多模态特征表示的自监督学习框架，促进可转移医学知识的学习，并提高利用多种数据源进行预训练的可扩展性。

### 方法

提出CITab框架，从语义感知角度设计表格建模机制，整合列标题作为语义线索；同时提出原型引导的线性层混合(P-MoLin)模块处理表格数据异质性并探索潜在医学概念。

### 主要发现

在包含4461名受试者的三个阿尔茨海默病诊断数据队列上的综合评估表明，CITab优于最先进的方法，为有效且可扩展的跨表格多模态学习铺平了道路。

### 结论

CITab通过创新的表格建模机制和特征专业化方法，成功克服了跨表格多模态学习中的障碍，为医学图像和表格数据的有效整合提供了新途径。

### 翻译

整合医学图像和表格数据的多模态学习近年来显著推进了临床决策。自监督学习(SSL)已成为在这些大规模未标记图像-表格数据上预训练这些模型的有力范式，旨在学习判别性表示。然而，现有的用于图像-表格表示学习的SSL方法通常局限于特定的数据队列，主要是由于其建模异构表格数据时僵化的表格建模机制。这种跨表格障碍阻碍了多模态SSL方法有效学习跨不同队列的可转移医学知识。在本文中，我们提出了一个新的SSL框架，即CITab，旨在以跨表格方式学习强大的多模态特征表示。我们从语义感知角度设计表格建模机制，通过整合列标题作为语义线索，这促进了可转移知识的学习以及利用多种数据源进行预训练的可扩展性。此外，我们提出了一个原型引导的线性层混合(P-MoLin)模块用于表格特征专业化，使模型能够有效处理表格数据的异质性并探索潜在的医学概念。我们在三个包含4461名受试者的公开可用数据队列上对阿尔茨海默病诊断任务进行了全面评估。实验结果表明，CITab优于最先进的方法，为有效且可扩展的跨表格多模态学习铺平了道路。


### 论文摘要

Multi-modal learning integrating medical images and tabular data has significantly advanced clinical decision-making in recent years. Self-Supervised Learning (SSL) has emerged as a powerful paradigm for pretraining these models on large-scale unlabeled image-tabular data, aiming to learn discriminative representations. However, existing SSL methods for image-tabular representation learning are often confined to specific data cohorts, mainly due to their rigid tabular modeling mechanisms when modeling heterogeneous tabular data. This inter-tabular barrier hinders the multi-modal SSL methods from effectively learning transferrable medical knowledge shared across diverse cohorts. In this paper, we propose a novel SSL framework, namely CITab, designed to learn powerful multi-modal feature representations in a cross-tabular manner. We design the tabular modeling mechanism from a semantic-awareness perspective by integrating column headers as semantic cues, which facilitates transferrable knowledge learning and the scalability in utilizing multiple data sources for pretraining. Additionally, we propose a prototype-guided mixture-of-linear layer (P-MoLin) module for tabular feature specialization, empowering the model to effectively handle the heterogeneity of tabular data and explore the underlying medical concepts. We conduct comprehensive evaluations on Alzheimer's disease diagnosis task across three publicly available data cohorts containing 4,461 subjects. Experimental results demonstrate that CITab outperforms state-of-the-art approaches, paving the way for effective and scalable cross-tabular multi-modal learning.

---

## 26. EXAONE Path 2.5: Pathology Foundation Model with Multi-Omics Alignment

**论文链接:** [http://arxiv.org/abs/2512.14019v1](http://arxiv.org/abs/2512.14019v1)

**作者:** Juseung Yun, Sunwoo Yu, Sumin Ha, Jonghyun Kim, Janghyeon Lee, Jongseong Jang, Soonyoung Lee

**发布时间:** 2025-12-16

### GPT解析

### 总结

EXAONE Path 2.5是一种病理学基础模型，通过联合建模组织学、基因组、表观基因组和转录组模态，捕捉癌症进展中多个生物层之间的相互作用，产生更全面的肿瘤生物学表征。

### 背景

癌症进展源于多个生物层之间的相互作用，特别是超越形态学层面，在分子层面上的相互作用，这些层面仅依靠图像模型无法捕捉。

### 目的

开发能够捕捉更广泛生物学景观的病理学基础模型，通过综合多种生物模态，产生反映肿瘤生物学更全面整合的患者表征。

### 方法

提出EXAONE Path 2.5病理学基础模型，包含三个关键组件：(1)多模态SigLIP损失，支持异构模态间的全对比学习；(2)片段感知的旋转位置编码模块，保留WSI中的空间结构和组织片段拓扑；(3)领域专业化的内部基础模型，用于WSI和RNA-seq，提供生物学基础的嵌入以实现稳健的多模态对齐。

### 主要发现

在内部真实世界临床数据集和Patho-Bench基准(涵盖80项任务)上评估，框架显示出高数据效率和参数效率，在Patho-Bench上与最先进的基础模型性能相当，在内部临床环境中表现出最高的适应性。

### 结论

生物学信息丰富的多模态设计具有重要价值，集成的基因型到表型建模对下一代精准肿瘤学具有潜力。

### 翻译

癌症进展源于多个生物层之间的相互作用，特别是超越形态学层面，在分子层面上的相互作用，这些层面仅依靠图像模型无法捕捉。为了捕捉这一更广泛的生物学景观，我们提出了EXAONE Path 2.5，一种病理学基础模型，它联合建模组织学、基因组、表观基因组和转录组模态，产生反映肿瘤生物学更全面整合的患者表征。我们的方法包含三个关键组件：(1)多模态SigLIP损失，支持异构模态间的全对比学习；(2)片段感知的旋转位置编码模块，保留WSI中的空间结构和组织片段拓扑；(3)领域专业化的内部基础模型，用于WSI和RNA-seq，提供生物学基础的嵌入以实现稳健的多模态对齐。我们在两个互补基准上评估了EXAONE Path 2.5：内部真实世界临床数据集和涵盖80项任务的Patho-Bench基准。我们的框架显示出高数据效率和参数效率，在Patho-Bench上与最先进的基础模型性能相当，同时在内部临床环境中表现出最高的适应性。这些结果强调了生物学信息丰富的多模态设计价值，并突出了集成的基因型到表型建模对下一代精准肿瘤学的潜力。


### 论文摘要

Cancer progression arises from interactions across multiple biological layers, especially beyond morphological and across molecular layers that remain invisible to image-only models. To capture this broader biological landscape, we present EXAONE Path 2.5, a pathology foundation model that jointly models histologic, genomic, epigenetic and transcriptomic modalities, producing an integrated patient representation that reflects tumor biology more comprehensively. Our approach incorporates three key components: (1) multimodal SigLIP loss enabling all-pairwise contrastive learning across heterogeneous modalities, (2) a fragment-aware rotary positional encoding (F-RoPE) module that preserves spatial structure and tissue-fragment topology in WSI, and (3) domain-specialized internal foundation models for both WSI and RNA-seq to provide biologically grounded embeddings for robust multimodal alignment. We evaluate EXAONE Path 2.5 against six leading pathology foundation models across two complementary benchmarks: an internal real-world clinical dataset and the Patho-Bench benchmark covering 80 tasks. Our framework demonstrates high data and parameter efficiency, achieving on-par performance with state-of-the-art foundation models on Patho-Bench while exhibiting the highest adaptability in the internal clinical setting. These results highlight the value of biologically informed multimodal design and underscore the potential of integrated genotype-to-phenotype modeling for next-generation precision oncology.

---

## 27. Enhancing Semi-Supervised Multi-View Graph Convolutional Networks via Supervised Contrastive Learning and Self-Training

**论文链接:** [http://arxiv.org/abs/2512.13770v1](http://arxiv.org/abs/2512.13770v1)

**作者:** Huaiyuan Xiao, Fadi Dornaika, Jingjun Bi

**发布时间:** 2025-12-15

### GPT解析

### 总结

MV-SupGCN是一种半监督图卷积网络模型，通过整合互补组件来有效利用多视图数据中的互补信息，提高特征表示和模型性能。

### 背景

图卷积网络(GCN)为基础的多视图学习为整合异构视图的结构信息提供了强大框架，但现有方法往往无法充分利用跨视图的互补信息，导致次优的特征表示和有限的性能。

### 目的

提出MV-SupGCN模型，通过整合互补组件来更好地利用跨视图的互补信息，改进特征表示，提高模型性能。

### 方法

MV-SupGCN包含三个主要组件：1)设计联合损失函数结合交叉熵损失和监督对比损失；2)结合基于KNN和半监督的图构建方法；3)整合对比学习和伪标记技术，强制多视图嵌入一致性并增强语义对齐。

### 主要发现

在多个基准测试中，MV-SupGCN持续超越最先进的方法，验证了整合方法的有效性。

### 结论

MV-SupGCN通过整合互补组件，有效解决了现有方法无法充分利用跨视图互补信息的问题，提高了特征表示质量和模型性能。

### 翻译

基于图卷积网络(GCN)的多视图学习的出现为整合异构视图的结构信息提供了强大框架，能够对复杂的多视图数据进行有效建模。然而，现有方法往往无法充分利用跨视图的互补信息，导致次优的特征表示和有限的性能。为了解决这个问题，我们提出了MV-SupGCN，这是一种半监督GCN模型，集成了几个具有明确动机和相互强化的互补组件。首先，为了更好地捕捉判别性特征并提高模型泛化能力，我们设计了一个联合损失函数，将交叉熵损失与监督对比损失相结合，鼓励模型在潜在空间中同时最小化类内方差和最大化类间可分性。其次，认识到单图构建方法的不稳定性和不完整性，我们在每个视图上结合了基于KNN和半监督的图构建方法，从而增强了数据结构表示的鲁棒性并减少了泛化误差。第三，为了有效利用丰富的未标记数据并增强多视图之间的语义对齐，我们提出了一个统一框架，整合对比学习以强制多视图嵌入之间的一致性并捕捉有意义的跨视图关系，同时结合伪标记，为交叉熵和对比损失函数提供额外的监督，以增强模型泛化能力。大量实验表明，MV-SupGCN在多个基准测试中持续超越最先进的方法，验证了我们整合方法的有效性。源代码可在https://github.com/HuaiyuanXiao/MVSupGCN获取。


### 论文摘要

The advent of graph convolutional network (GCN)-based multi-view learning provides a powerful framework for integrating structural information from heterogeneous views, enabling effective modeling of complex multi-view data. However, existing methods often fail to fully exploit the complementary information across views, leading to suboptimal feature representations and limited performance. To address this, we propose MV-SupGCN, a semi-supervised GCN model that integrates several complementary components with clear motivations and mutual reinforcement. First, to better capture discriminative features and improve model generalization, we design a joint loss function that combines Cross-Entropy loss with Supervised Contrastive loss, encouraging the model to simultaneously minimize intra-class variance and maximize inter-class separability in the latent space. Second, recognizing the instability and incompleteness of single graph construction methods, we combine both KNN-based and semi-supervised graph construction approaches on each view, thereby enhancing the robustness of the data structure representation and reducing generalization error. Third, to effectively utilize abundant unlabeled data and enhance semantic alignment across multiple views, we propose a unified framework that integrates contrastive learning in order to enforce consistency among multi-view embeddings and capture meaningful inter-view relationships, together with pseudo-labeling, which provides additional supervision applied to both the cross-entropy and contrastive loss functions to enhance model generalization. Extensive experiments demonstrate that MV-SupGCN consistently surpasses state-of-the-art methods across multiple benchmarks, validating the effectiveness of our integrated approach. The source code is available at https://github.com/HuaiyuanXiao/MVSupGCN

---

## 28. A Semantically Enhanced Generative Foundation Model Improves Pathological Image Synthesis

**论文链接:** [http://arxiv.org/abs/2512.13164v2](http://arxiv.org/abs/2512.13164v2)

**作者:** Xianchao Guan, Zhiyuan Fan, Yifeng Wang, Fuqiang Chen, Yanjiang Zhou, Zengyang Che, Hongxue Meng, Xin Li, Yaowei Wang, Hongpeng Wang, Min Zhang, Heng Tao Shen, Zheng Zhang, Yongbing Zhang

**发布时间:** 2025-12-15

**备注:** 68 pages, 9 figures, 16 tables

### GPT解析

### 总结

研究团队开发了CRAFTS框架，这是一种针对病理学的文本到图像生成基础模型，通过克服数据稀缺性和生成质量问题，为临床级病理学人工智能提供了新的解决方案。

### 背景

临床级病理学人工智能的发展受限于高质量注释数据集的稀缺性，而现有生成模型虽能提供潜在解决方案，但存在语义不稳定性和形态幻觉问题，影响诊断可靠性。

### 目的

开发一种能够生成高质量、多样化病理图像的生成模型，解决病理学中高质量注释数据集稀缺的问题，同时确保生成图像的生物学准确性和诊断可靠性。

### 方法

提出CRAFTS（Correlation-Regulated Alignment Framework for Tissue Synthesis）框架，采用双阶段训练策略，基于约280万图像-标题对进行训练，并引入一种新的对齐机制来抑制语义漂移，确保生物学准确性。同时探索了CRAFTS与ControlNet的结合，以实现对组织架构的精确控制。

### 主要发现

CRAFTS能够生成跨越30种癌症类型的多样化病理图像，其质量通过客观指标和病理学家评估得到验证。使用CRAFTS增强的数据集显著提高了分类、跨模态检索、自监督学习和视觉问答等多种临床任务的表现。结合ControlNet后，模型能够从核分割掩模和荧光图像等输入精确控制组织架构。

### 结论

CRAFTS克服了病理学人工智能发展中数据稀缺性和隐私关注的关键障碍，提供了无限的多样化、注释组织学数据源，有效解锁了罕见和复杂癌症表型的稳健诊断工具开发。

### 翻译

病理学临床级人工智能的发展受限于多样化、高质量注释数据集的稀缺性。生成模型提供了潜在解决方案，但存在语义不稳定性和形态幻觉，影响诊断可靠性。为应对这一挑战，我们引入了CRAFTS（组织合成的相关性调节对齐框架），这是首个病理学特定的文本到图像生成生成基础模型。通过利用约280万图像-标题对的双阶段训练策略，CRAFTS结合了一种新的对齐机制，抑制语义漂移以确保生物学准确性。该模型生成跨越30种癌症类型的多样化病理图像，其质量通过客观指标和病理学家评估严格验证。此外，CRAFTS增强的数据集提高了分类、跨模态检索、自监督学习和视觉问答等各种临床任务的表现。此外，将CRAFTS与ControlNet结合能够从核分割掩模和荧光图像等输入精确控制组织架构。通过克服数据稀缺性和隐私关注的关键障碍，CRAFTS提供了无限的多样化、注释组织学数据源，有效解锁了罕见和复杂癌症表型的稳健诊断工具的开发。


### 论文摘要

The development of clinical-grade artificial intelligence in pathology is limited by the scarcity of diverse, high-quality annotated datasets. Generative models offer a potential solution but suffer from semantic instability and morphological hallucinations that compromise diagnostic reliability. To address this challenge, we introduce a Correlation-Regulated Alignment Framework for Tissue Synthesis (CRAFTS), the first generative foundation model for pathology-specific text-to-image synthesis. By leveraging a dual-stage training strategy on approximately 2.8 million image-caption pairs, CRAFTS incorporates a novel alignment mechanism that suppresses semantic drift to ensure biological accuracy. This model generates diverse pathological images spanning 30 cancer types, with quality rigorously validated by objective metrics and pathologist evaluations. Furthermore, CRAFTS-augmented datasets enhance the performance across various clinical tasks, including classification, cross-modal retrieval, self-supervised learning, and visual question answering. In addition, coupling CRAFTS with ControlNet enables precise control over tissue architecture from inputs such as nuclear segmentation masks and fluorescence images. By overcoming the critical barriers of data scarcity and privacy concerns, CRAFTS provides a limitless source of diverse, annotated histology data, effectively unlocking the creation of robust diagnostic tools for rare and complex cancer phenotypes.

---

## 29. TF-MCL: Time-frequency Fusion and Multi-domain Cross-Loss for Self-supervised Depression Detection

**论文链接:** [http://arxiv.org/abs/2512.13736v1](http://arxiv.org/abs/2512.13736v1)

**作者:** Li-Xuan Zhao, Chen-Yang Xu, Wen-Qiang Li, Bo Wang, Rong-Xing Wei, Qing-Hao Menga

**发布时间:** 2025-12-14

### GPT解析

### 总结

本文提出了一种名为TF-MCL的时间-频率融合和多域交叉损失模型，用于提高基于脑电图信号的重度抑郁症检测性能。

### 背景

近年来基于脑电图信号的重度抑郁症监督检测方法使用增加，但标记过程具有挑战性。对比学习作为自监督学习方法可解决监督学习过度依赖标签的问题，但现有对比学习方法未专门设计来表征EEG信号的时间-频率分布，获取低语义数据表示的能力不足。

### 目的

解决现有对比学习方法在MDD检测中的局限性，提出一种能更有效表征EEG信号时间-频率分布的模型。

### 方法

TF-MCL模型通过融合映射头(FMH)生成时间-频率混合表示，将时间-频率域信息重新映射到融合域；同时通过优化多域交叉损失函数，重建时间-频率域和融合域中表示的分布，提高模型获取融合表示的能力。

### 主要发现

在公开数据集MODMA和PRED+CT上的评估显示，该模型准确率显著提高，分别比现有最先进方法提高了5.87%和9.96%。

### 结论

TF-MCL模型能有效提升基于EEG信号的重度抑郁症检测性能，为临床应用提供了新的可能性。

### 翻译

近年来，基于脑电图信号的重度抑郁症监督检测方法使用显著增加。然而，MDD的标记过程仍然具有挑战性。作为自监督学习方法，对比学习可以解决监督学习方法在MDD检测中过度依赖标签的缺点。然而，现有对比学习方法并非专门设计来表征EEG信号的时间-频率分布，它们获取低语义数据表示的能力对于MDD检测任务仍然不足。为了解决对比学习方法的问题，我们提出了一种用于MDD检测的时间-频率融合和多域交叉损失(TF-MCL)模型。TF-MCL通过融合映射头(FMH)生成时间-频率混合表示，有效地将时间-频率域信息重新映射到融合域，从而可以增强模型合成时间-频率信息的能力。此外，通过优化多域交叉损失函数，重建了时间-频率域和融合域中表示的分布，从而提高了模型获取融合表示的能力。我们在公开数据集MODMA和PRED+CT上评估了模型的性能，显示准确率显著提高，分别比现有最先进(SOTA)方法提高了5.87%和9.96%。


### 论文摘要

In recent years, there has been a notable increase in the use of supervised detection methods of major depressive disorder (MDD) based on electroencephalogram (EEG) signals. However, the process of labeling MDD remains challenging. As a self-supervised learning method, contrastive learning could address the shortcomings of supervised learning methods, which are unduly reliant on labels in the context of MDD detection. However, existing contrastive learning methods are not specifically designed to characterize the time-frequency distribution of EEG signals, and their capacity to acquire low-semantic data representations is still inadequate for MDD detection tasks. To address the problem of contrastive learning method, we propose a time-frequency fusion and multi-domain cross-loss (TF-MCL) model for MDD detection. TF-MCL generates time-frequency hybrid representations through the use of a fusion mapping head (FMH), which efficiently remaps time-frequency domain information to the fusion domain, and thus can effectively enhance the model's capacity to synthesize time-frequency information. Moreover, by optimizing the multi-domain cross-loss function, the distribution of the representations in the time-frequency domain and the fusion domain is reconstructed, thereby improving the model's capacity to acquire fusion representations. We evaluated the performance of our model on the publicly available datasets MODMA and PRED+CT and show a significant improvement in accuracy, outperforming the existing state-of-the-art (SOTA) method by 5.87% and 9.96%, respectively.

---

## 30. ParaFormer: A Generalized PageRank Graph Transformer for Graph Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.14619v1](http://arxiv.org/abs/2512.14619v1)

**作者:** Chaohao Yuan, Zhenjie Song, Ercan Engin Kuruoglu, Kangfei Zhao, Yang Liu, Deli Zhao, Hong Cheng, Yu Rong

**发布时间:** 2025-12-16

**备注:** Accepted by WSDM 2026

### GPT解析

### 总结

该论文提出了一种名为PageRank Transformer (ParaFormer)的新型图神经网络架构，解决了全局注意力机制中存在的严重过度平滑问题。

### 背景

Graph Transformers (GTs)作为一种有前景的图学习工具，利用全连接特性有效捕获全局信息。为解决深度GNN中的过度平滑问题，最初引入了全局注意力机制，消除了使用深度GNN的必要性。

### 目的

减轻全局注意力机制中存在的严重过度平滑问题，使节点表示保持可区分性。

### 方法

提出PageRank Transformer (ParaFormer)，其特点是包含一个PageRank增强的注意力模块，专门设计来模仿深度Transformers的行为。

### 主要发现

通过理论和实证分析，验证了ParaFormer通过作为自适应通滤波器来减轻过度平滑问题。在11个从几千到几百万节点的数据集上，ParaFormer在节点分类和图分类任务中实现了一致的性能提升。

### 结论

ParaFormer有效解决了全局注意力中的过度平滑问题，在多种图学习任务中展现出优越的性能和有效性。

### 翻译

图变换器(GTs)已成为一种有前景的图学习工具，利用其全连接特性有效捕获全局信息。为解决深度图神经网络(GNNs)中的过度平滑问题，最初引入了全局注意力，消除了使用深度GNN的必要性。然而，通过实证和理论分析，我们验证了引入的全局注意力表现出严重的过度平滑，由于其固有的低通滤波特性导致节点表示变得不可区分。这种效应甚至比在GNN中观察到的效应更强。为缓解这一问题，我们提出了PageRank Transformer (ParaFormer)，它具有一个PageRank增强的注意力模块，设计用来模仿深度Transformers的行为。我们通过理论和实证证明，ParaFormer通过作为自适应通滤波器来减轻过度平滑。实验显示，ParaFormer在11个从几千到几百万节点的数据集上，在节点分类和图分类任务中实现了性能的持续提升，验证了其有效性。补充材料，包括代码和附录，可在https://github.com/chaohaoyuan/ParaFormer找到。


### 论文摘要

Graph Transformers (GTs) have emerged as a promising graph learning tool, leveraging their all-pair connected property to effectively capture global information. To address the over-smoothing problem in deep GNNs, global attention was initially introduced, eliminating the necessity for using deep GNNs. However, through empirical and theoretical analysis, we verify that the introduced global attention exhibits severe over-smoothing, causing node representations to become indistinguishable due to its inherent low-pass filtering. This effect is even stronger than that observed in GNNs. To mitigate this, we propose PageRank Transformer (ParaFormer), which features a PageRank-enhanced attention module designed to mimic the behavior of deep Transformers. We theoretically and empirically demonstrate that ParaFormer mitigates over-smoothing by functioning as an adaptive-pass filter. Experiments show that ParaFormer achieves consistent performance improvements across both node classification and graph classification tasks on 11 datasets ranging from thousands to millions of nodes, validating its efficacy. The supplementary material, including code and appendix, can be found in https://github.com/chaohaoyuan/ParaFormer.

---

## 31. ARCADE: Adaptive Robot Control with Online Changepoint-Aware Bayesian Dynamics Learning

**论文链接:** [http://arxiv.org/abs/2512.14331v1](http://arxiv.org/abs/2512.14331v1)

**作者:** Rishabh Dev Yadav, Avirup Das, Hongyu Song, Samuel Kaski, Wei Pan

**发布时间:** 2025-12-16

### GPT解析

### 总结

该研究提出了一种能够实时更新以处理机器人系统非线性动态变化的框架，能够有效应对逐渐漂移、瞬时波动和突然转变等多种变化模式。

### 背景

现实世界中的机器人必须在不断变化的动态条件下运行，这些变化由变化的操作条件、外部干扰和未建模效应引起，表现为逐渐漂移、瞬时波动或突然转变。

### 目的

开发一个能够实时更新的非线性动态建模框架，实现对机器人系统动态变化的实时适应，既对短期变化具有鲁棒性，又能对持久变化做出响应。

### 方法

提出将表示学习与在线适应解耦的框架，使用离线学习的潜在表示支持在线闭式贝叶斯更新；引入变化点感知机制，通过从数据可能性推断的潜在变量来指示连续性或转变；根据连续性或转变情况分别采用积累证据或调整信息的策略。

### 主要发现

框架的自适应遗憾随时间仅呈对数增长，随转变次数呈线性增长，与知道转变时间的预言家相当；在cartpole模拟和真实四旋翼飞行器验证中，显示出比相关基线更好的预测准确性、更快的恢复能力和更精确的闭环跟踪。

### 结论

该框架能够维持校准的不确定性，支持对瞬时、逐渐或结构变化的概率推理，为处理现实世界中机器人面临的动态变化提供了有效解决方案。

### 翻译

现实世界中的机器人必须在不断变化的动态条件下运行，这些变化由变化的操作条件、外部干扰和未建模效应引起。这些可能表现为逐渐漂移、瞬时波动或突然转变，要求实时适应既对短期变化具有鲁棒性，又能对持久变化做出响应。我们提出一个用于建模机器人系统非线性动态的框架，该框架可以从流数据中实时更新。该方法将表示学习与在线适应解耦，使用离线学习的潜在表示来支持在线闭式贝叶斯更新。为了处理不断变化的情况，我们引入了一个具有变化点感知能力的机制，通过从数据可能性推断的潜在变量来指示连续性或转变。当连续性可能时，证据积累以完善预测；当检测到转变时，调整过去信息以实现快速重新学习。这维持了校准的不确定性，并支持对瞬时、逐渐或结构变化的概率推理。我们证明了该框架的自适应遗憾随时间仅呈对数增长，随转变次数呈线性增长，与知道转变时间的预言家相当。我们在cartpole模拟和带有摆动载荷和飞行中丢弃载荷的真实四旋翼飞行器上进行了验证，显示出比相关基线更好的预测准确性、更快的恢复能力和更精确的闭环跟踪。


### 论文摘要

Real-world robots must operate under evolving dynamics caused by changing operating conditions, external disturbances, and unmodeled effects. These may appear as gradual drifts, transient fluctuations, or abrupt shifts, demanding real-time adaptation that is robust to short-term variation yet responsive to lasting change. We propose a framework for modeling the nonlinear dynamics of robotic systems that can be updated in real time from streaming data. The method decouples representation learning from online adaptation, using latent representations learned offline to support online closed-form Bayesian updates. To handle evolving conditions, we introduce a changepoint-aware mechanism with a latent variable inferred from data likelihoods that indicates continuity or shift. When continuity is likely, evidence accumulates to refine predictions; when a shift is detected, past information is tempered to enable rapid re-learning. This maintains calibrated uncertainty and supports probabilistic reasoning about transient, gradual, or structural change. We prove that the adaptive regret of the framework grows only logarithmically in time and linearly with the number of shifts, competitive with an oracle that knows timings of shift. We validate on cartpole simulations and real quadrotor flights with swinging payloads and mid-flight drops, showing improved predictive accuracy, faster recovery, and more accurate closed-loop tracking than relevant baselines.

---

## 32. ProtoFlow: Interpretable and Robust Surgical Workflow Modeling with Learned Dynamic Scene Graph Prototypes

**论文链接:** [http://arxiv.org/abs/2512.14092v1](http://arxiv.org/abs/2512.14092v1)

**作者:** Felix Holm, Ghazal Ghazaei, Nassir Navab

**发布时间:** 2025-12-16

### GPT解析

### 总结

ProtoFlow是一种新型框架，通过学习动态场景图原型来建模复杂外科手术流程，结合了自监督预训练和基于原型的微调，在有限数据条件下表现出色，并能提供可解释的手术见解。

### 背景

详细的外科手术识别对于推进AI辅助手术至关重要，但进展受到高标注成本、数据稀缺和缺乏可解释模型的阻碍。场景图虽然提供了手术事件的结构化抽象，但其潜力尚未完全开发。

### 目的

引入ProtoFlow框架，学习动态场景图原型，以可解释和稳健的方式建模复杂的外科手术流程。

### 方法

ProtoFlow利用图神经网络编码器-解码器架构，结合自监督预训练进行丰富的表征学习，以及基于原型的微调阶段，发现和完善封装重复的、具有临床意义的手术交互模式的核心原型。

### 主要发现

在CAT-SG数据集上，ProtoFlow在总体准确性和少样本场景中均优于标准GNN基线，即使仅使用一个外科视频训练也能保持强大性能，并能识别不同的外科子技术，提供对工作流偏差和罕见并发症的可解释见解。

### 结论

ProtoFlow结合了稳健的表征学习和固有的可解释性，朝着开发更透明、可靠和数据高效的AI系统迈出了重要一步，加速了其在外科培训、实时决策支持和工作流优化方面的临床应用潜力。

### 翻译

目的：详细的外科手术识别对于推进AI辅助手术至关重要，但进展受到高标注成本、数据稀缺和缺乏可解释模型的阻碍。虽然场景图提供了手术事件的结构化抽象，但其全部潜力仍未被开发。在这项工作中，我们引入了ProtoFlow，一个新颖的框架，它学习动态场景图原型，以可解释和稳健的方式建模复杂的外科手术流程。方法：ProtoFlow利用图神经网络编码器-解码器架构，结合自监督预训练进行丰富的表征学习，以及基于原型的微调阶段。这一过程发现和完善了封装重复的、具有临床意义的手术交互模式的核心原型，为工作流分析形成了可解释的基础。结果：我们在细粒度的CAT-SG数据集上评估了我们的方法。ProtoFlow不仅在总体准确性上优于标准GNN基线，而且在有限数据、少样本场景中也表现出卓越的稳健性，即使在仅使用一个外科视频进行训练的情况下也能保持强大的性能。我们的定性分析进一步表明，学习到的原型成功识别了不同的外科子技术，并对工作流偏差和罕见并发症提供了清晰、可解释的见解。结论：通过结合稳健的表征学习和固有的可解释性，ProtoFlow朝着开发更透明、可靠和数据高效的AI系统迈出了重要一步，加速了它们在外科培训、实时决策支持和工作流优化方面的临床应用潜力。


### 论文摘要

Purpose: Detailed surgical recognition is critical for advancing AI-assisted surgery, yet progress is hampered by high annotation costs, data scarcity, and a lack of interpretable models. While scene graphs offer a structured abstraction of surgical events, their full potential remains untapped. In this work, we introduce ProtoFlow, a novel framework that learns dynamic scene graph prototypes to model complex surgical workflows in an interpretable and robust manner.   Methods: ProtoFlow leverages a graph neural network (GNN) encoder-decoder architecture that combines self-supervised pretraining for rich representation learning with a prototype-based fine-tuning stage. This process discovers and refines core prototypes that encapsulate recurring, clinically meaningful patterns of surgical interaction, forming an explainable foundation for workflow analysis.   Results: We evaluate our approach on the fine-grained CAT-SG dataset. ProtoFlow not only outperforms standard GNN baselines in overall accuracy but also demonstrates exceptional robustness in limited-data, few-shot scenarios, maintaining strong performance when trained on as few as one surgical video. Our qualitative analyses further show that the learned prototypes successfully identify distinct surgical sub-techniques and provide clear, interpretable insights into workflow deviations and rare complications.   Conclusion: By uniting robust representation learning with inherent explainability, ProtoFlow represents a significant step toward developing more transparent, reliable, and data-efficient AI systems, accelerating their potential for clinical adoption in surgical training, real-time decision support, and workflow optimization.

---

## 33. Topologically-Stabilized Graph Neural Networks: Empirical Robustness Across Domains

**论文链接:** [http://arxiv.org/abs/2512.13852v1](http://arxiv.org/abs/2512.13852v1)

**作者:** Jelena Losic

**发布时间:** 2025-12-15

### GPT解析

### 总结

该研究提出了一种新框架，通过整合持久同调特征和稳定性正则化来增强图神经网络对结构扰动的鲁棒性。

### 背景

图神经网络已成为图表示学习的标准方法，但对结构扰动仍然脆弱。

### 目的

提出一个新框架，通过整合持久同调特征和稳定性正则化来增强图神经网络的鲁棒性。

### 方法

基于持久同调的稳定性定理，将GIN架构与从持久图像中提取的多尺度拓扑特征相结合，并通过Hiraoka-Kusano启发的稳定性约束进行强制执行。

### 主要发现

在六个涵盖生化、社交和协作网络的多样化数据集上，该方法对边扰动表现出卓越的鲁棒性，同时保持有竞争力的准确性。在扰动下观察到最小的性能下降(大多数数据集上为0-4%)，显著优于基线稳定性。

### 结论

该工作提供了一种理论上合理且经验验证的鲁棒图学习方法，符合拓扑正则化的最新进展。

### 翻译

图神经网络已成为图表示学习的标准方法，但对结构扰动仍然脆弱。我们提出了一种新框架，整合持久同调特征和稳定性正则化以增强鲁棒性。基于持久同调的稳定性定理，我们的方法将GIN架构与从持久图像中提取的多尺度拓扑特征相结合，并通过受Hiraoka-Kusano启发的稳定性约束强制执行。在六个涵盖生化、社交和协作网络的多样化数据集上，我们的方法在保持竞争力的准确性的同时，对边扰动表现出卓越的鲁棒性。值得注意的是，在扰动下我们观察到最小的性能下降(大多数数据集上为0-4%)，显著优于基线稳定性。我们的工作提供了一种理论上合理且经验验证的鲁棒图学习方法，符合拓扑正则化的最新进展。


### 论文摘要

Graph Neural Networks (GNNs) have become the standard for graph representation learning but remain vulnerable to structural perturbations. We propose a novel framework that integrates persistent homology features with stability regularization to enhance robustness. Building on the stability theorems of persistent homology \cite{cohen2007stability}, our method combines GIN architectures with multi-scale topological features extracted from persistence images, enforced by Hiraoka-Kusano-inspired stability constraints. Across six diverse datasets spanning biochemical, social, and collaboration networks , our approach demonstrates exceptional robustness to edge perturbations while maintaining competitive accuracy. Notably, we observe minimal performance degradation (0-4\% on most datasets) under perturbation, significantly outperforming baseline stability. Our work provides both a theoretically-grounded and empirically-validated approach to robust graph learning that aligns with recent advances in topological regularization

---

## 34. Enhancing Geo-localization for Crowdsourced Flood Imagery via LLM-Guided Attention

**论文链接:** [http://arxiv.org/abs/2512.11811v2](http://arxiv.org/abs/2512.11811v2)

**作者:** Fengyi Xu, Jun Ma, Waishan Qiu, Cui Guo, Jack C. P. Cheng

**发布时间:** 2025-11-25

**备注:** Updated author list to include additional contributor. Revised title and improved methodology section based on collaborative feedback

### GPT解析

### 总结

本文提出VPR-AttLLM框架，通过整合大型语言模型的语义推理和地理知识到视觉位置识别流程中，改进了社交媒体众包街景图像的地理定位性能，特别是在城市洪水等危机事件场景下。

### 背景

社交媒体众包的街景图像提供了城市洪水和其他危机事件的实时视觉证据，但通常缺乏可靠的地理元数据用于应急响应。现有的图像地理定位方法在应用于此类图像时性能显著下降，因为跨场景情况下的视觉失真和域偏移。

### 目的

开发一个无需重新训练模型或添加额外数据就能提高检索性能的框架，解决社交媒体图像地理定位中的挑战。

### 方法

VPR-AttLLM是一个与模型无关的框架，通过注意力引导的描述符增强，将大型语言模型的语义推理和地理知识整合到现有的视觉位置识别流程中。利用大型语言模型识别城市背景中的信息性区域并抑制视觉噪声。

### 主要发现

将VPR-AttLLM与三种最先进的VPR模型（CosPlace、EigenPlaces和SALAD）结合，始终提高了召回性能，相对增益通常在1-3%之间，在最具挑战性的真实洪水图像上达到8%。

### 结论

VPR-AttLLM建立了视觉检索系统中大型语言模型引导的多模态融合的可推广范式。通过将城市感知理论原理嵌入到注意力机制中，桥接了类人的空间推理与现代视觉位置识别架构。其即插即用设计、强大的跨源鲁棒性和可解释性突显了其在可扩展城市监测和众包危机图像快速地理定位方面的潜力。

### 翻译

社交媒体众包的街景图像为城市洪水和其他危机事件提供了实时视觉证据，但它通常缺乏应急响应所需的可靠地理元数据。现有的图像地理定位方法，也称为视觉位置识别模型，在应用于此类图像时表现出显著的性能下降，这是由于跨源场景中的视觉失真和域偏移。本文提出了VPR-AttLLM，一个与模型无关的框架，通过注意力引导的描述符增强，将大型语言模型的语义推理和地理知识整合到既定的视觉位置识别流程中。通过利用大型语言模型识别城市背景中的信息性区域并抑制视觉噪声，VPR-AttLLM在不要求重新训练模型或添加额外数据的情况下提高了检索性能。在扩展基准上进行了全面评估，包括用真实社交媒体洪水图像增强的SF-XL、在已建立的查询集和Mapillary照片上的合成洪水场景，以及捕捉形态各异城市景观的新HK-URBAN数据集。将VPR-AttLLM与三种最先进的视觉位置识别模型（CosPlace、EigenPlaces和SALAD）集成，始终提高了召回性能，相对增益通常在1-3%之间，在最具挑战性的真实洪水图像上达到8%。除了在检索准确性方面可衡量的增益外，本研究还建立了视觉检索系统中大型语言模型引导的多模态融合的可推广范式。通过将城市感知理论的原理嵌入到注意力机制中，VPR-AttLLM桥接了类人的空间推理与现代视觉位置识别架构。其即插即用设计、强大的跨源鲁棒性和可解释性突显了其在可扩展城市监测和众包危机图像快速地理定位方面的潜力。


### 论文摘要

Crowdsourced street-view imagery from social media provides real-time visual evidence of urban flooding and other crisis events, yet it often lacks reliable geographic metadata for emergency response. Existing image geo-localization approaches, also known as Visual Place Recognition (VPR) models, exhibit substantial performance degradation when applied to such imagery due to visual distortions and domain shifts in cross-source scenarios. This paper presents VPR-AttLLM, a model-agnostic framework that integrates the semantic reasoning and geo-knowledge of Large Language Models (LLMs) into established VPR pipelines through attention-guided descriptor enhancement. By leveraging LLMs to identify location-informative regions within the city context and suppress visual noise, VPR-AttLLM improves retrieval performance without requiring model retraining or additional data. Comprehensive evaluations are conducted on extended benchmarks including SF-XL enriched with real social-media flood images, synthetic flooding scenarios over established query sets and Mapillary photos, and a new HK-URBAN dataset capturing morphologically distinct cityscapes. Integrating VPR-AttLLM with three state-of-the-art VPR models-CosPlace, EigenPlaces, and SALAD-consistently improves recall performance, yielding relative gains typically between 1-3% and reaching up to 8% on the most challenging real flood imagery. Beyond measurable gains in retrieval accuracy, this study establishes a generalizable paradigm for LLM-guided multimodal fusion in visual retrieval systems. By embedding principles from urban perception theory into attention mechanisms, VPR-AttLLM bridges human-like spatial reasoning with modern VPR architectures. Its plug-and-play design, strong cross-source robustness, and interpretability highlight its potential for scalable urban monitoring and rapid geo-localization of crowdsourced crisis imagery.

---

## 35. Using Socio-economic Indicators, Smart Transit Systems, and Urban Simulator to Accelerate ZEV Adoption and Reduce VMT

**论文链接:** [http://arxiv.org/abs/2512.11870v2](http://arxiv.org/abs/2512.11870v2)

**作者:** Mulham Fawakherji, Bruce Race, Driss Benhaddou

**发布时间:** 2025-12-05

### GPT解析

### 总结

该论文研究了休斯顿市如何实现2050年净零排放目标，重点关注道路交通减排策略，通过建立排放基准和评估政策来加速零排放车辆采用并减少车辆行驶里程。

### 背景

全球道路交通占温室气体排放的15%，导致约38.5万人因PM2.5过早死亡；城市占全球能源相关温室气体排放的75%；休斯顿道路交通占其气候行动计划基准排放的48%。

### 目的

建立道路排放基准并评估政策，利用社会经济指标和智能交通系统加速零排放车辆采用并减少车辆行驶里程，以帮助休斯顿实现2050年净零排放目标。

### 方法

采用智能停车、公共交通激励、安全数据系统和零排放车队管理等策略；在Unity 3D中开发模拟环境，支持城市动态建模和政策情景可视化。

### 主要发现

休斯顿作为低密度、汽车依赖型城市，89%的道路排放来自汽车和小型卡车，公共交通使用有限；社会经济差异制约了零排放车辆采用；可通过扩大ZEV获取渠道和将VMT减少20%的策略应对挑战。

### 结论

依赖汽车的城市为实现2050年排放目标可从论文讨论的指标、度量和技术中受益。

### 翻译

全球范围内，道路交通占温室气体排放的15%，并估计有385,000人因PM2.5而过早死亡。城市在实现IPCC目标中发挥关键作用，占全球能源相关温室气体排放的75%。在德克萨斯州的休斯顿，道路交通占气候行动计划中基准排放的48%。到2050年实现净零排放，气候行动计划目标是比2014年基准减少70%的排放，30%通过可再生能源抵消。这一目标具有挑战性，因为休斯顿是低密度、汽车依赖型城市，89%的道路排放来自汽车和小型卡车，公共交通使用有限。社会经济差异进一步制约了零排放车辆的采用。策略重点在于扩大ZEV获取渠道并通过公共交通改善和城市设计将车辆行驶里程减少20%。本文介绍了建立道路排放基准的方法和评估政策，这些政策利用社会经济指标和智能交通系统加速ZEV采用并减少VMT。智能停车、公共交通激励、安全数据系统和ZEV车队管理支持模式分割和系统可靠性的改善。分析了政策选项并确定了潜在行动。为支持评估，在Unity 3D中开发了模拟环境，实现了城市动态建模和政策情景可视化。依赖汽车的城市为实现2050年排放目标可从讨论的指标、度量和技术中受益。


### 论文摘要

Globally, on-road transportation accounts for 15% of greenhouse gas (GHG) emissions and an estimated 385,000 premature deaths from PM2.5. Cities play a critical role in meeting IPCC targets, generating 75% of global energy-related GHG emissions. In Houston, Texas, on-road transportation represents 48% of baseline emissions in the Climate Action Plan (CAP). To reach net-zero by 2050, the CAP targets a 70% emissions reduction from a 2014 baseline, offset by 30% renewable energy. This goal is challenging because Houston is low-density and auto-dependent, with 89% of on-road emissions from cars and small trucks and limited public transit usage. Socio-economic disparities further constrain Zero Emissions Vehicle (ZEV) adoption. Strategies focus on expanding ZEV access and reducing Vehicle Miles Traveled (VMT) by 20% through transit improvements and city design. This paper presents methods for establishing an on-road emissions baseline and evaluating policies that leverage socio-economic indicators and Intelligent Transportation Systems (ITS) to accelerate ZEV adoption and reduce VMT. Smart parking, transit incentives, secure data systems, and ZEV fleet management support improvements in modal split and system reliability. Policy options are analyzed and potential actions identified. To support evaluation, a simulation environment was developed in Unity 3D, enabling dynamic modeling of urban mobility and visualization of policy scenarios. Auto-dependent cities aiming for 2050 emission targets can benefit from the indicators, metrics, and technologies discussed.

---

## 36. PrediFlow: A Flow-Based Prediction-Refinement Framework for Real-Time Human Motion Prediction in Human-Robot Collaboration

**论文链接:** [http://arxiv.org/abs/2512.13903v1](http://arxiv.org/abs/2512.13903v1)

**作者:** Sibo Tian, Minghui Zheng, Xiao Liang

**发布时间:** 2025-12-15

### GPT解析

### 总结

该研究解决了人机协作中随机人类运动预测的局限性，通过整合人类和机器人运动信息提高预测质量，同时保持实时性。

### 背景

随机人类运动预测在工业再制造人机协作中至关重要，可捕捉人类运动的不确定性和多模态行为。早期方法产生不现实运动，近期方法关注准确性和实时性但仍有改进空间，且现有研究常孤立考虑人类运动，忽略机器人运动的影响。

### 目的

解决研究空白，实现实时、逼真且交互感知的人类运动预测。

### 方法

提出预测-精炼框架，整合人类和机器人观察运动精炼预训练预测器的初始预测，采用Flow Matching结构处理不确定性。

### 主要发现

在人机协作桌面拆卸数据集上实验表明，方法显著提高预测准确性，保留人类运动的不确定性和多模态性，总推理时间在预算内。

### 结论

该方法具有有效性和实用性，适合实际应用场景。

### 翻译

随机人类运动预测对于工业再制造中安全有效的人机协作至关重要，因为它捕捉了人类运动的不确定性和多模态行为，而确定性方法无法处理这一点。虽然早期工作强调高度多样化的预测，但它们常常产生不现实的人类运动。最近的方法关注准确性和实时性能，然而在不超过时间预算的情况下，仍有进一步提高预测质量的潜力。此外，人机协作中随机人类运动预测的当前研究通常孤立地考虑人类运动，忽略了机器人运动对人类行为的影响。为了解决这些研究空白并实现实时、逼真且交互感知的人类运动预测，我们提出了一种新颖的预测-精炼框架，该框架整合人类和机器人的观察运动来精炼预训练的最先进预测器产生的初始预测。精炼模块采用Flow Matching结构来考虑不确定性。在人机协作桌面拆卸数据集上的实验研究表明，我们的方法显著提高了预测准确性，同时保留了人类运动的不确定性和多模态性。此外，所提出框架的总推理时间保持在时间预算内，突显了我们方法的有效性和实用性。


### 论文摘要

Stochastic human motion prediction is critical for safe and effective human-robot collaboration (HRC) in industrial remanufacturing, as it captures human motion uncertainties and multi-modal behaviors that deterministic methods cannot handle. While earlier works emphasize highly diverse predictions, they often generate unrealistic human motions. More recent methods focus on accuracy and real-time performance, yet there remains potential to improve prediction quality further without exceeding time budgets. Additionally, current research on stochastic human motion prediction in HRC typically considers human motion in isolation, neglecting the influence of robot motion on human behavior. To address these research gaps and enable real-time, realistic, and interaction-aware human motion prediction, we propose a novel prediction-refinement framework that integrates both human and robot observed motion to refine the initial predictions produced by a pretrained state-of-the-art predictor. The refinement module employs a Flow Matching structure to account for uncertainty. Experimental studies on the HRC desktop disassembly dataset demonstrate that our method significantly improves prediction accuracy while preserving the uncertainties and multi-modalities of human motion. Moreover, the total inference time of the proposed framework remains within the time budget, highlighting the effectiveness and practicality of our approach.

---

## 37. MMGR: Multi-Modal Generative Reasoning

**论文链接:** [http://arxiv.org/abs/2512.14691v1](http://arxiv.org/abs/2512.14691v1)

**作者:** Zefan Cai, Haoyi Qiu, Tianyi Ma, Haozhe Zhao, Gengze Zhou, Kung-Hsiang Huang, Parisa Kordjamshidi, Minjia Zhang, Xiao Wen, Jiuxiang Gu, Nanyun Peng, Junjie Hu

**发布时间:** 2025-12-16

**备注:** work in progress

### GPT解析

### 总结

该论文介绍了MMGR（多模态生成推理评估与基准），一个基于五种推理能力的评估框架，用于评估生成模型在物理、逻辑和空间约束方面的表现。

### 背景

视频基础模型能生成视觉逼真和时间一致的内容，但作为世界模拟器的可靠性取决于它们能否捕捉物理、逻辑和空间约束。现有指标如Frechet Video Distance强调感知质量而忽略推理失败。

### 目的

引入MMGR评估框架，全面评估生成模型在多种推理任务上的表现，填补现有评估方法在推理能力评估上的空白。

### 方法

MMGR基于五种推理能力构建：物理、逻辑、3D空间、2D空间和时间。评估三个领域：抽象推理（ARC-AGI、数独）、具身导航（现实世界3D导航和定位）和物理常识（运动和组合交互）。应用细粒度指标要求视频和图像生成中整体正确性。

### 主要发现

对领先视频模型（Veo-3、Sora-2、Wan-2.2）和图像模型（Nano-banana、Nano-banana Pro、GPT-4o-image、Qwen-image）的基准测试显示各领域存在明显性能差距。模型在物理常识任务上表现中等，但在抽象推理上表现不佳（ARC-AGI准确率低于10%），在具身环境中长程空间规划方面存在困难。

### 结论

当前模型存在关键局限性：过度依赖感知数据、弱全局状态一致性，以及奖励视觉合理性而非因果正确性。MMGR提供统一诊断基准，为推理感知的生成世界模型发展指明方向。

### 翻译

视频基础模型生成视觉上逼真和时间上一致的内容，但它们作为世界模拟器的可靠性取决于它们是否能够捕捉物理、逻辑和空间约束。现有的指标如Frechet Video Distance强调感知质量而忽略了推理失败，包括因果性、物理学和全局一致性的违反。我们介绍了MMGR（多模态生成推理评估与基准），这是一个基于五种推理能力的评估框架：物理、逻辑、3D空间、2D空间和时间。MMGR评估三个领域的生成推理：抽象推理（ARC-AGI、数独）、具身导航（现实世界3D导航和定位）和物理常识（运动和组合交互）。MMGR应用细粒度指标，要求在视频和图像生成中整体正确性。我们对领先的视频模型（Veo-3、Sora-2、Wan-2.2）和图像模型（Nano-banana、Nano-banana Pro、GPT-4o-image、Qwen-image）进行了基准测试，发现在各领域存在明显的性能差距。模型在物理常识任务上表现出中等成功，但在抽象推理上表现不佳（ARC-AGI准确率低于10%），并且在具身环境中的长程空间规划方面存在困难。我们的分析揭示了当前模型的关键局限性，包括过度依赖感知数据、弱全局状态一致性，以及奖励视觉合理性而非因果正确性的目标。MMGR提供了一个统一的诊断基准，并为推理感知的生成世界模型发展指明了方向。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有视频生成模型虽然能生成视觉上引人注目的内容，但无法内化和遵守现实世界的物理、逻辑和空间约束的问题。这个问题很重要，因为随着生成AI在电影制作、科学可视化、机器人等领域的应用，如果模型不理解现实约束，会产生严重错误，限制其可靠性和实用性。同时，现有评估指标（如FVD）主要关注视觉保真度，无法检测这些推理失败，导致对模型能力的误判。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于人类认知理论，提出世界模拟需要五种核心推理能力（物理、逻辑、3D空间、2D空间和时间），并基于此设计了MMGR评估框架。他们借鉴了认知科学中的核心知识理论，参考了Matterport3D等导航数据集，利用了VideoPhy本体论评估物理常识，并改编了ARC-AGI等抽象推理基准用于生成模型评估。这些现有工作为MMGR提供了理论基础和任务设计灵感。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是基于五种推理能力框架开发一个全面的基准测试套件，评估生成模型在抽象推理、具身导航和物理常识三个领域的推理能力。整体流程包括：1)定义三个领域的具体任务；2)创建多样化的测试样本并控制难度；3)使用先进生成模型创建响应；4)通过VLM评估器和人工评估进行评估；5)分析模型表现并识别优势和局限性。评估强调整体正确性而非部分成功，使用细粒度领域特定指标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于五种推理能力的系统性评估框架；2)首个同时评估视频和图像生成模型推理能力的多模态基准；3)抽象推理、具身导航和物理常识三个互补评估领域；4)要求整体正确性的细粒度评估指标；5)通过分析识别训练数据不平衡、架构弱点等局限性。相比之前工作，MMGR超越了传统感知保真度评估，关注推理能力；从理解转向生成评估；强调长时间范围内的全局一致性；首次统一评估不同模态模型，揭示性能不对称性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MMGR通过全面的评估框架和基准测试，首次系统揭示了当前多模态生成模型在物理、逻辑和空间推理方面的关键局限性，为开发真正能理解和模拟现实世界约束的下一代生成模型指明了方向。'}


### 论文摘要

Video foundation models generate visually realistic and temporally coherent content, but their reliability as world simulators depends on whether they capture physical, logical, and spatial constraints. Existing metrics such as Frechet Video Distance (FVD) emphasize perceptual quality and overlook reasoning failures, including violations of causality, physics, and global consistency. We introduce MMGR (Multi-Modal Generative Reasoning Evaluation and Benchmark), a principled evaluation framework based on five reasoning abilities: Physical, Logical, 3D Spatial, 2D Spatial, and Temporal. MMGR evaluates generative reasoning across three domains: Abstract Reasoning (ARC-AGI, Sudoku), Embodied Navigation (real-world 3D navigation and localization), and Physical Commonsense (sports and compositional interactions). MMGR applies fine-grained metrics that require holistic correctness across both video and image generation. We benchmark leading video models (Veo-3, Sora-2, Wan-2.2) and image models (Nano-banana, Nano-banana Pro, GPT-4o-image, Qwen-image), revealing strong performance gaps across domains. Models show moderate success on Physical Commonsense tasks but perform poorly on Abstract Reasoning (below 10 percent accuracy on ARC-AGI) and struggle with long-horizon spatial planning in embodied settings. Our analysis highlights key limitations in current models, including overreliance on perceptual data, weak global state consistency, and objectives that reward visual plausibility over causal correctness. MMGR offers a unified diagnostic benchmark and a path toward reasoning-aware generative world models.

---

## 38. A Multicenter Benchmark of Multiple Instance Learning Models for Lymphoma Subtyping from HE-stained Whole Slide Images

**论文链接:** [http://arxiv.org/abs/2512.14640v1](http://arxiv.org/abs/2512.14640v1)

**作者:** Rao Muhammad Umer, Daniel Sens, Jonathan Noll, Christian Matek, Lukas Wolfseher, Rainer Spang, Ralf Huss, Johannes Raffler, Sarah Reinke, Wolfram Klapper, Katja Steiger, Kristina Schwamborn, Carsten Marr

**发布时间:** 2025-12-16

**备注:** 17 pages

### GPT解析

### 总结

本研究提出了首个多中心淋巴瘤基准测试数据集，评估了五种病理学基础模型与两种多实例学习聚合器在不同放大倍率下的性能，并探讨了模型的泛化能力。

### 背景

及时准确的淋巴瘤诊断对癌症治疗至关重要。标准诊断结合HE染色切片与多种测试确定淋巴瘤亚型，需要昂贵设备、专业人员并导致治疗延迟。深度学习方法可从常规HE染色切片提取诊断信息协助病理学家，但缺乏多中心数据上的淋巴瘤亚型分型基准测试。

### 目的

创建首个覆盖四种常见淋巴瘤亚型和健康对照组织的多中心淋巴瘤基准测试数据集，并系统评估不同模型在多种放大倍率下的性能。

### 方法

评估五种公开病理学基础模型(H-optimus-1, H0-mini, Virchow2, UNI2, Titan)与两种多实例学习聚合器(基于注意力的AB-MIL和基于transformer的TransMIL)在三种放大倍率(10x, 20x, 40x)下的性能。

### 主要发现

在分布内测试集上，模型在所有放大倍率下多类平衡准确率超过80%，所有基础模型表现相似，两种聚合方法结果相当；40x分辨率足够，更高分辨率无性能提升；但在分布外测试集上性能降至约60%，显示显著泛化挑战。

### 结论

为推进该领域，需要覆盖更多罕见淋巴瘤亚型的更大规模多中心研究，研究提供了自动化基准测试流程促进未来研究。

### 翻译

及时准确的淋巴瘤诊断对于指导癌症治疗至关重要。标准诊断实践结合了苏木精-伊红(HE)染色全切片图像与免疫组织化学、流式细胞术和分子遗传学测试来确定淋巴瘤亚型，这一过程需要昂贵的设备、熟练的人员，并导致治疗延迟。深度学习方法可以通过从常规可用的HE染色切片中提取诊断信息来协助病理学家，但在多中心数据上进行淋巴瘤亚型分型的全面基准测试仍然缺乏。在这项工作中，我们提出了首个覆盖四种常见淋巴瘤亚型和健康对照组织的多中心淋巴瘤基准测试数据集。我们系统地评估了五种公开的病理学基础模型(H-optimus-1, H0-mini, Virchow2, UNI2, Titan)与基于注意力的(AB-MIL)和基于transformer的(TransMIL)多实例学习聚合器在三种放大倍率(10x, 20x, 40x)下的性能。在分布内测试集上，模型在所有放大倍率下都实现了超过80%的多类平衡准确率，所有基础模型表现相似，两种聚合方法也显示出相当的结果。放大倍率研究表明，40x分辨率足够，更高分辨率或跨放大倍率聚合没有带来性能提升。然而，在分布外测试集上，性能显著下降到约60%，突显了显著的泛化挑战。为了推进该领域，需要覆盖更多罕见淋巴瘤亚型的更大规模多中心研究。我们提供了一个自动化基准测试流程，以促进未来的此类研究。


### 论文摘要

Timely and accurate lymphoma diagnosis is essential for guiding cancer treatment. Standard diagnostic practice combines hematoxylin and eosin (HE)-stained whole slide images with immunohistochemistry, flow cytometry, and molecular genetic tests to determine lymphoma subtypes, a process requiring costly equipment, skilled personnel, and causing treatment delays. Deep learning methods could assist pathologists by extracting diagnostic information from routinely available HE-stained slides, yet comprehensive benchmarks for lymphoma subtyping on multicenter data are lacking. In this work, we present the first multicenter lymphoma benchmarking dataset covering four common lymphoma subtypes and healthy control tissue. We systematically evaluate five publicly available pathology foundation models (H-optimus-1, H0-mini, Virchow2, UNI2, Titan) combined with attention-based (AB-MIL) and transformer-based (TransMIL) multiple instance learning aggregators across three magnifications (10x, 20x, 40x). On in-distribution test sets, models achieve multiclass balanced accuracies exceeding 80% across all magnifications, with all foundation models performing similarly and both aggregation methods showing comparable results. The magnification study reveals that 40x resolution is sufficient, with no performance gains from higher resolutions or cross-magnification aggregation. However, on out-of-distribution test sets, performance drops substantially to around 60%, highlighting significant generalization challenges. To advance the field, larger multicenter studies covering additional rare lymphoma subtypes are needed. We provide an automated benchmarking pipeline to facilitate such future research.

---

## 39. Native Intelligence Emerges from Large-Scale Clinical Practice: A Retinal Foundation Model with Deployment Efficiency

**论文链接:** [http://arxiv.org/abs/2512.14499v1](http://arxiv.org/abs/2512.14499v1)

**作者:** Jia Guo, Jiawei Du, Shengzhu Yang, Shuai Lu, Wenquan Cheng, Kaiwen Zhang, Yihua Sun, Chuhong Yang, Weihang Zhang, Fang Chen, Yilan Wu, Lie Ju, Guochen Ning, Longfei Ma, Huiping Yao, Jinyuan Wang, Peilun Shi, Yukun Zhou, Jie Xu, Pearse A. Keane, Hanruo Liu, Hongen Liao, Ningli Wang, Huiqi Li

**发布时间:** 2025-12-16

### GPT解析

### 总结

该研究提出了ReVision视网膜基础模型，通过真实医疗实践中的远程医疗数据直接学习临床原生智能，无需大量任务特定优化即可在资源有限环境中高效部署。

### 背景

当前视网膜基础模型受限于缺乏真实临床环境的研究数据集，需要针对每个应用进行大量特定任务优化，限制了在资源有限环境中的部署效率。

### 目的

克服现有模型的局限性，直接从真实医疗实践中构建临床原生智能，实现医疗AI系统在资源有限环境中的高效部署。

### 方法

利用大规模远程医疗项目作为学习资源库，构建ReVision视网膜基础模型，从485,980张彩色眼底照片和相应诊断报告的自然对齐中学习，这些数据通过跨越中国162家医疗机构的十年远程医疗项目积累。

### 主要发现

ReVision在27个眼科基准测试中实现高效部署；无需特定任务训练，在12个公共基准测试上实现零样本疾病检测，平均AUROC为0.946，在3个独立临床队列上为0.952；在最小适应情况下，与大量微调的替代方案相匹配，但需要的参数和标记示例少几个数量级；学习到的表示能有效迁移到新的临床环境和任务；在33名眼科医生的研究中，提高诊断准确率14.8%。

### 结论

临床原生智能可以直接从临床档案中提取，无需额外注释，构建适合各种资源有限环境的医疗AI系统。

### 翻译

当前视网膜基础模型仍受限于缺乏真实临床环境的研究数据集，需要针对每个应用进行大量特定任务优化，限制了它们在资源有限环境中的部署效率。在这里，我们表明这些障碍可以通过直接从真实医疗实践中构建临床原生智能来克服。我们的关键见解是，大型远程医疗项目（专家中心为分散设施提供远程咨询）代表了学习临床图像解释的自然资源库。我们提出了ReVision，一个视网膜基础模型，从485,980张彩色眼底照片及其相应诊断报告的自然对齐中学习，这些数据通过跨越中国162家医疗机构的十年远程医疗项目积累。通过对27个眼科基准测试的广泛评估，我们证明ReVision能够以最少的本地资源实现高效部署。无需任何特定任务训练，ReVision在12个公共基准测试上实现零样本疾病检测，平均AUROC为0.946，在3个独立临床队列上为0.952。当最小适应可行时，ReVision与大量微调的替代方案相匹配，但需要的可训练参数和标记示例少几个数量级。学习到的表示也有效迁移到新的临床站点、成像领域、成像方式和系统性健康预测任务。在33名眼科医生的前瞻性读者研究中，ReVision的零样本帮助提高了各经验水平医生的诊断准确率14.8%。这些结果表明，临床原生智能可以直接从临床档案中提取，无需任何进一步注释，来构建适合各种资源有限环境的医疗AI系统。


### 论文摘要

Current retinal foundation models remain constrained by curated research datasets that lack authentic clinical context, and require extensive task-specific optimization for each application, limiting their deployment efficiency in low-resource settings. Here, we show that these barriers can be overcome by building clinical native intelligence directly from real-world medical practice. Our key insight is that large-scale telemedicine programs, where expert centers provide remote consultations across distributed facilities, represent a natural reservoir for learning clinical image interpretation. We present ReVision, a retinal foundation model that learns from the natural alignment between 485,980 color fundus photographs and their corresponding diagnostic reports, accumulated through a decade-long telemedicine program spanning 162 medical institutions across China. Through extensive evaluation across 27 ophthalmic benchmarks, we demonstrate that ReVison enables deployment efficiency with minimal local resources. Without any task-specific training, ReVision achieves zero-shot disease detection with an average AUROC of 0.946 across 12 public benchmarks and 0.952 on 3 independent clinical cohorts. When minimal adaptation is feasible, ReVision matches extensively fine-tuned alternatives while requiring orders of magnitude fewer trainable parameters and labeled examples. The learned representations also transfer effectively to new clinical sites, imaging domains, imaging modalities, and systemic health prediction tasks. In a prospective reader study with 33 ophthalmologists, ReVision's zero-shot assistance improved diagnostic accuracy by 14.8% across all experience levels. These results demonstrate that clinical native intelligence can be directly extracted from clinical archives without any further annotation to build medical AI systems suited to various low-resource settings.

---

## 40. A4-Agent: An Agentic Framework for Zero-Shot Affordance Reasoning

**论文链接:** [http://arxiv.org/abs/2512.14442v1](http://arxiv.org/abs/2512.14442v1)

**作者:** Zixin Zhang, Kanghao Chen, Hanqing Wang, Hongfei Zhang, Harold Haodong Chen, Chenfei Liao, Litao Guo, Ying-Cong Chen

**发布时间:** 2025-12-16

### GPT解析

### 总结

本文提出了A4-Agent，一个无需训练的智能体框架，通过解耦可预测性预测为三个阶段，显著提升了在新物体和未见环境中的泛化能力。

### 背景

可预测性预测对具身AI至关重要，但当前端到端模型将高级推理和低级定位耦合在单一流水线中，依赖标注数据训练，导致泛化能力差。

### 目的

超越当前范式，提出一个无需训练的智能体框架，将可预测性预测解耦为三阶段流水线，提升模型泛化能力。

### 方法

A4-Agent框架在测试时协调三个专门的基础模型：Dreamer使用生成模型可视化交互过程；Thinker利用视觉-语言模型决定交互对象部位；Spotter协调视觉模型精确定位交互区域。整个框架无需任务特定微调。

### 主要发现

通过利用预训练模型的互补优势，该零样本框架在多个基准测试中显著优于最先进的监督方法，并展现出对真实世界设置的强大泛化能力。

### 结论

A4-Agent通过解耦预测过程并利用预训练模型，实现了更好的泛化能力，代表了可预测性预测的新范式。

### 翻译

可预测性预测是基于语言指令识别物体上交互区域的过程，对具身AI至关重要。当前主流的端到端模型将高级推理和低级定位耦合到单一流水线中，依赖标注数据训练，导致在新物体和未见环境中泛化能力差。本文提出了A4-Agent，一个无需训练的智能体框架，将可预测性预测解耦为三阶段流水线：Dreamer使用生成模型可视化交互如何呈现；Thinker利用大型视觉-语言模型决定与物体哪部分交互；Spotter协调视觉基础模型精确定位交互区域。通过利用预训练模型的互补优势，该零样本框架在多个基准测试中显著优于最先进的监督方法，并表现出对真实世界设置的强大泛化能力。


### 论文摘要

Affordance prediction, which identifies interaction regions on objects based on language instructions, is critical for embodied AI. Prevailing end-to-end models couple high-level reasoning and low-level grounding into a single monolithic pipeline and rely on training over annotated datasets, which leads to poor generalization on novel objects and unseen environments. In this paper, we move beyond this paradigm by proposing A4-Agent, a training-free agentic framework that decouples affordance prediction into a three-stage pipeline. Our framework coordinates specialized foundation models at test time: (1) a $\textbf{Dreamer}$ that employs generative models to visualize $\textit{how}$ an interaction would look; (2) a $\textbf{Thinker}$ that utilizes large vision-language models to decide $\textit{what}$ object part to interact with; and (3) a $\textbf{Spotter}$ that orchestrates vision foundation models to precisely locate $\textit{where}$ the interaction area is. By leveraging the complementary strengths of pre-trained models without any task-specific fine-tuning, our zero-shot framework significantly outperforms state-of-the-art supervised methods across multiple benchmarks and demonstrates robust generalization to real-world settings.

---

## 41. TiCard: Deployable EXPLAIN-only Residual Learning for Cardinality Estimation

**论文链接:** [http://arxiv.org/abs/2512.14358v1](http://arxiv.org/abs/2512.14358v1)

**作者:** Qizhi Wang

**发布时间:** 2025-12-16

**备注:** 16 pages(/wo references), 4 figures, 10 tables

### GPT解析

### 总结

这篇论文介绍了TiCard，一个低侵入性、基于修正的框架，用于增强数据库的原生基数估计器，解决了传统估计器无法捕捉相关性和学习型估计器需要侵入性集成的问题。

### 背景

基数估计是基于成本的查询优化的关键瓶颈，但可部署的改进仍然困难。传统的估计器无法捕捉相关性，而学习型估计器通常需要工作负载特定的训练流程，并且需要侵入性地集成到优化器中。

### 目的

提出TiCard框架，这是一个低侵入性、基于修正的框架，用于增强(而非替换)数据库的原生估计器，提高基数估计的准确性。

### 方法

TiCard仅使用EXPLAIN功能学习乘法残差修正，并仅使用EXPLAIN ANALYZE进行离线标签。研究两种实际实现：(i)梯度提升回归器，用于亚毫秒级推理；(ii)TabPFN，一种上下文表格基础模型，通过刷新小型参考集来适应，无需梯度重新训练。

### 主要发现

在TiDB上使用TPCH和连接顺序基准测试，在低跟踪设置下，TiCard显著提高了操作级别的尾部准确性：P90 Q-error从312.85降至13.69，P99从37,974.37降至3,416.50，而仅连接策略保留了接近完美的中位数行为。

### 结论

将TiCard定位为专注于可部署性的AI4DB构建块：明确范围、保守的集成策略，以及从离线修正到优化器内使用的集成路线图。

### 翻译

基数估计是基于成本的查询优化的关键瓶颈，但可部署的改进仍然困难：传统估计器无法捕捉相关性，而学习型估计器通常需要工作负载特定的训练流程和侵入性集成到优化器中。本文介绍了TiCard，一个低侵入性、基于修正的框架，它增强(而非替换)数据库的原生估计器。TiCard使用仅EXPLAIN功能学习乘法残差修正，并仅使用EXPLAIN ANALYZE进行离线标签。我们研究了两种实际实现：(i)用于亚毫秒级推理的梯度提升回归器，和(ii)TabPFN，一种通过刷新小型参考集来适应而无需梯度重新训练的上下文表格基础模型。在TiDB上使用TPCH和连接顺序基准测试，在低跟踪设置下(总共执行263次；157次用于学习)，TiCard显著提高了操作级别的尾部准确性：P90 Q-error从312.85(原生)降至13.69(TiCard-GBR)，P99从37,974.37降至3,416.50(TiCard-TabPFN)，而仅连接策略保留了接近完美的中位数行为。我们将TiCard定位为专注于可部署性的AI4DB构建块：明确范围、保守的集成策略，以及从离线修正到优化器内使用的集成路线图。


### 论文摘要

Cardinality estimation is a key bottleneck for cost-based query optimization, yet deployable improvements remain difficult: classical estimators miss correlations, while learned estimators often require workload-specific training pipelines and invasive integration into the optimizer. This paper presents TiCard, a low intrusion, correction-based framework that augments (rather than replaces) a database's native estimator. TiCard learns multiplicative residual corrections using EXPLAIN-only features, and uses EXPLAIN ANALYZE only for offline labels. We study two practical instantiations: (i) a Gradient Boosting Regressor for sub-millisecond inference, and (ii) TabPFN, an in-context tabular foundation model that adapts by refreshing a small reference set without gradient retraining. On TiDB with TPCH and the Join Order Benchmark, in a low-trace setting (263 executions total; 157 used for learning), TiCard improves operator-level tail accuracy substantially: P90 Q-error drops from 312.85 (native) to 13.69 (TiCard-GBR), and P99 drops from 37,974.37 to 3,416.50 (TiCard-TabPFN), while a join-only policy preserves near-perfect median behavior. We position TiCard as an AI4DB building block focused on deployability: explicit scope, conservative integration policies, and an integration roadmap from offline correction to in-optimizer use.

---

## 42. FLAME: Flow Enhanced Legendre Memory Models for General Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2512.14253v1](http://arxiv.org/abs/2512.14253v1)

**作者:** Xingjian Wu, Hanyin Cheng, Xiangfei Qiu, Zhengyu Li, Jilin Hu, Chenjuan Guo, Bin Yang

**发布时间:** 2025-12-16

### GPT解析

### 总结

FLAME是一个轻量级时间序列基础模型家族，支持确定性和概率性预测，通过Legendre Memory及其变体实现高效和准确的预测。

### 背景

时间序列预测需要能够处理长期依赖关系并保持计算效率的模型。

### 目的

开发一个轻量级但功能强大的时间序列基础模型，支持确定性和概率性预测，同时保持高效和稳健性。

### 方法

FLAME利用Legendre Memory及其变体（平移Legendre和缩放Legendre）在编码和解码阶段捕捉数据中的归纳偏置，并采用基于归一化流的预测头来建模预测范围内的复杂分布。

### 主要发现

FLAME在TSFM-Bench和ProbTS等基准测试上实现了最先进的零样本性能，在确定性和概率性预测任务上表现一致优异。

### 结论

FLAME是一个高效且稳健的时间序列基础模型，通过结合Legendre Memory和归一化流预测头，实现了在确定性和概率性预测任务上的最先进性能。

### 翻译

这项工作中，我们引入了FLAME，一个极其轻量且功能强大的时间序列基础模型家族，它通过生成概率建模支持确定性和概率性预测，从而确保了效率和稳健性。FLAME利用Legendre Memory获得强大的泛化能力。通过在编码和解码阶段调整Legendre Memory的变体，即平移Legendre (LegT)和缩放Legendre (LegS)，FLAME可以有效捕捉数据中的固有归纳偏置，并进行高效的长程推理。为了在保持高效的同时提高概率预测的准确性，FLAME采用了基于归一化流的预测头，可以以生成方式对预测范围内的任意复杂分布进行建模。在TSFM-Bench和ProbTS等公认基准上的全面实验表明，FLAME在确定性和概率性预测任务上 consistently达到了最先进的零样本性能。


### 论文摘要

In this work, we introduce FLAME, a family of extremely lightweight and capable Time Series Foundation Models, which support both deterministic and probabilistic forecasting via generative probabilistic modeling, thus ensuring both efficiency and robustness. FLAME utilizes the Legendre Memory for strong generalization capabilities. Through adapting variants of Legendre Memory, i.e., translated Legendre (LegT) and scaled Legendre (LegS), in the Encoding and Decoding phases, FLAME can effectively capture the inherent inductive bias within data and make efficient long-range inferences. To enhance the accuracy of probabilistic forecasting while keeping efficient, FLAME adopts a Normalization Flow based forecasting head, which can model the arbitrarily intricate distributions over the forecasting horizon in a generative manner. Comprehensive experiments on well-recognized benchmarks, including TSFM-Bench and ProbTS, demonstrate the consistent state-of-the-art zero-shot performance of FLAME on both deterministic and probabilistic forecasting tasks.

---

## 43. HydroGEM: A Self Supervised Zero Shot Hybrid TCN Transformer Foundation Model for Continental Scale Streamflow Quality Control

**论文链接:** [http://arxiv.org/abs/2512.14106v1](http://arxiv.org/abs/2512.14106v1)

**作者:** Ijaz Ul Haq, Byung Suk Lee, Julia N. Perdrial, David Baude

**发布时间:** 2025-12-16

**备注:** Supplementary materials, datasets, and implementation code will be made publicly available upon acceptance for publication in a peer-reviewed journal

### GPT解析

### 总结

这项研究介绍了HydroGEM，一个用于大陆尺度河流流量质量控制的基础模型，它通过两阶段训练和混合TCN-Transformer架构实现了高效的数据质量检测和重建。

### 背景

实时河流流量监测网络每年产生数百万观测数据，但在维护数千个远程传感器的数据质量方面仍然需要大量人工劳动。

### 目的

开发一个基础模型HydroGEM用于大陆尺度的河流流量质量控制，减少人工维护工作量，提高数据质量检测和重建的效率。

### 方法

HydroGEM采用两阶段训练：首先在6.03万个美国地质调查局站点的序列上进行自监督预训练学习水文表示，然后使用合成异常进行微调以进行检测和重建。模型采用混合TCN-Transformer架构（1420万参数），捕获局部时间模式和长程依赖关系，同时使用分层归一化处理六个数量级的流量变化。

### 主要发现

在包含799个站点和18种专家验证的异常类型的保留合成测试中，HydroGEM的检测F1得分为0.792，重建误差减少了68.7%，比现有方法提高了36.3%。在100个加拿大环境和气候变化站点的零样本迁移测试中，F1得分为0.586，超过了所有基线，展示了跨国家泛化能力。模型在不同校正幅度下保持一致的检测效果，并与操作季节性模式保持一致。

### 结论

HydroGME设计用于人机协同工作流程，输出的是需要专家审核的质量控制建议，而不是自主校正。

### 翻译

实时河流流量监测网络每年产生数百万观测数据，但在维护数千个远程传感器的数据质量方面仍然需要大量人工劳动。我们介绍了HydroGEM（水文监测通用编码器），这是一个用于大陆尺度河流流量质量控制的基础模型。HydroGEM使用两阶段训练：在3724个美国地质调查局站点的603万个序列上进行自监督预训练学习水文表示，然后使用合成异常进行微调以进行检测和重建。混合TCN-Transformer架构（1420万参数）捕获局部时间模式和长程依赖关系，而分层归一化处理六个数量级的流量变化。在包含799个站点和18种专家验证的异常类型的保留合成测试中，HydroGEM的检测F1得分为0.792，重建误差减少了68.7%，比现有方法提高了36.3%。在100个加拿大环境和气候变化站点的零样本迁移测试中，F1得分为0.586，超过了所有基线，展示了跨国家泛化能力。模型在不同校正幅度下保持一致的检测效果，并与操作季节性模式保持一致。HydroGME设计用于人机协同工作流程，输出的是需要专家审核的质量控制建议，而不是自主校正。


### 论文摘要

Real-time streamflow monitoring networks generate millions of observations annually, yet maintaining data quality across thousands of remote sensors remains labor-intensive. We introduce HydroGEM (Hydrological Generalizable Encoder for Monitoring), a foundation model for continental-scale streamflow quality control. HydroGEM uses two-stage training: self-supervised pretraining on 6.03 million sequences from 3,724 USGS stations learns hydrological representations, followed by fine-tuning with synthetic anomalies for detection and reconstruction. A hybrid TCN-Transformer architecture (14.2M parameters) captures local temporal patterns and long-range dependencies, while hierarchical normalization handles six orders of magnitude in discharge. On held-out synthetic tests comprising 799 stations with 18 expert-validated anomaly types, HydroGEM achieves F1 = 0.792 for detection and 68.7% reconstruction-error reduction, a 36.3% improvement over existing methods. Zero-shot transfer to 100 Environment and Climate Change Canada stations yields F1 = 0.586, exceeding all baselines and demonstrating cross-national generalization. The model maintains consistent detection across correction magnitudes and aligns with operational seasonal patterns. HydroGEM is designed for human-in-the-loop workflows - outputs are quality control suggestions requiring expert review, not autonomous corrections.

---

## 44. Neurosymbolic Inference On Foundation Models For Remote Sensing Text-to-image Retrieval With Complex Queries

**论文链接:** [http://arxiv.org/abs/2512.14102v1](http://arxiv.org/abs/2512.14102v1)

**作者:** Emanuele Mezzi, Gertjan Burghouts, Maarten Kruithof

**发布时间:** 2025-12-16

### GPT解析

### 总结

本文介绍了一种名为RUNE（基于神经符号实体的推理）的新方法，结合大语言模型和神经符号AI进行遥感文本到图像检索，通过显式推理提高性能和可解释性。

### 背景

随着针对航空和卫星图像的大视觉语言模型兴起，遥感文本到图像检索技术快速发展，但有限的可解释性和复杂空间关系处理能力仍是实际应用中的主要挑战。

### 目的

解决现有遥感大视觉语言模型的可解释性差和复杂空间关系处理能力不足的问题，提高遥感图像检索的性能、鲁棒性和可解释性。

### 方法

RUNE结合大语言模型和神经符号AI，将文本查询转换为一阶逻辑表达式，通过推理检测到的实体与逻辑表达式间的兼容性检索图像；提出逻辑分解策略在实体子集上操作，缩短执行时间；仅利用基础模型生成逻辑表达式，将推理任务委托给神经符号推理模块。

### 主要发现

RUNE在重新利用的DOTA数据集上表现出优越性能；引入查询复杂度检索鲁棒性(RRQC)和图像不确定性检索鲁棒性(RRIU)两个新指标；在复杂遥感检索任务中优于联合嵌入模型，在性能、鲁棒性和可解释性方面都有提升；通过洪水后卫星图像检索用例展示了现实应用潜力。

### 结论

RUNE通过结合大语言模型和神经符号AI，有效解决了遥感文本到图像检索中的可解释性和复杂空间关系处理问题，为现实世界遥感应用提供了更强大的工具。

### 翻译

遥感(RS)领域的文本到图像检索随着专门针对航空和卫星图像的大视觉语言模型(LVLMs)的兴起而快速发展，最终形成了遥感大视觉语言模型(RS-LVLMs)。然而，有限的可解释性和对复杂空间关系的处理能力不足仍然是实际应用中的主要挑战。为了解决这些问题，我们引入了RUNE（基于神经符号实体的推理），一种结合大语言模型(LLMs)和神经符号AI的方法，通过推理检测到的实体与从文本查询推导出的一阶逻辑(FOL)表达式之间的兼容性来检索图像。与依赖隐式联合嵌入的RS-LVLMs不同，RUNE执行明确的推理，提高了性能和可解释性。为了扩展性，我们提出了一种逻辑分解策略，在检测到的实体子集上操作，保证了比神经方法更短的执行时间。我们仅利用基础模型生成FOL表达式，将推理任务委托给神经符号推理模块。为了评估，我们重新利用了原本用于目标检测的DOTA数据集，通过添加比现有基准更复杂的查询来增强它。我们展示了LLM在文本到逻辑翻译方面的有效性，并与最先进的RS-LVLMs进行了比较，证明了RUNE的优越性能。我们引入了两个指标：查询复杂度检索鲁棒性(RRQC)和图像不确定性检索鲁棒性(RRIU)，它们评估了相对于查询复杂度和图像不确定性的性能。在复杂的遥感检索任务中，RUNE优于联合嵌入模型，在性能、鲁棒性和可解释性方面都有所提升。我们通过洪水后卫星图像检索的用例，展示了RUNE在现实世界遥感应用中的潜力。


### 论文摘要

Text-to-image retrieval in remote sensing (RS) has advanced rapidly with the rise of large vision-language models (LVLMs) tailored for aerial and satellite imagery, culminating in remote sensing large vision-language models (RS-LVLMS). However, limited explainability and poor handling of complex spatial relations remain key challenges for real-world use. To address these issues, we introduce RUNE (Reasoning Using Neurosymbolic Entities), an approach that combines Large Language Models (LLMs) with neurosymbolic AI to retrieve images by reasoning over the compatibility between detected entities and First-Order Logic (FOL) expressions derived from text queries. Unlike RS-LVLMs that rely on implicit joint embeddings, RUNE performs explicit reasoning, enhancing performance and interpretability. For scalability, we propose a logic decomposition strategy that operates on conditioned subsets of detected entities, guaranteeing shorter execution time compared to neural approaches. Rather than using foundation models for end-to-end retrieval, we leverage them only to generate FOL expressions, delegating reasoning to a neurosymbolic inference module. For evaluation we repurpose the DOTA dataset, originally designed for object detection, by augmenting it with more complex queries than in existing benchmarks. We show the LLM's effectiveness in text-to-logic translation and compare RUNE with state-of-the-art RS-LVLMs, demonstrating superior performance. We introduce two metrics, Retrieval Robustness to Query Complexity (RRQC) and Retrieval Robustness to Image Uncertainty (RRIU), which evaluate performance relative to query complexity and image uncertainty. RUNE outperforms joint-embedding models in complex RS retrieval tasks, offering gains in performance, robustness, and explainability. We show RUNE's potential for real-world RS applications through a use case on post-flood satellite image retrieval.

---

## 45. Scalable Frameworks for Real-World Audio-Visual Speech Recognition

**论文链接:** [http://arxiv.org/abs/2512.14083v1](http://arxiv.org/abs/2512.14083v1)

**作者:** Sungnyun Kim

**发布时间:** 2025-12-16

**备注:** PhD Dissertation

### GPT解析

### 总结

该研究针对音频-视觉语音识别系统在现实环境中因噪声和干扰导致的性能下降问题，提出了一种系统化的分层解决方案，在表示、架构和系统三个层面实现鲁棒可扩展性。

### 背景

音频-视觉语音识别系统在实际部署中面临严重性能下降问题，主要源于现实环境中不可预测的声学噪声和视觉干扰。

### 目的

通过系统化的分层方法克服这些挑战，在表示、架构和系统层面实现稳健的可扩展性。

### 方法

在表示层面：研究构建统一模型的方法，学习对多样化现实世界干扰具有内在鲁棒性的音频-视觉特征；在架构层面：探索如何有效扩展模型容量，同时确保多模态输入的自适应和可靠使用；在系统层面：通过与大尺度基础模型的模块化集成扩展系统功能，利用其强大的认知和生成能力。

### 主要发现

通过在这三个层面提供系统化的解决方案，可以构建下一代具有高可靠性的稳健可扩展AVSR系统。

### 结论

该研究旨在构建一个在现实应用中具有高可靠性的下一代、稳健且可扩展的AVSR系统。

### 翻译

音频-视觉语音识别(AVSR)系统的实际部署从根本上受到现实环境中显著性能下降的挑战，这些环境以不可预测的声学噪声和视觉干扰为特征。本论文认为，系统化的分层方法是克服这些挑战所必需的，在表示、架构和系统层面实现稳健的可扩展性。在表示层面，我们研究构建统一模型的方法，该模型学习对多样化现实世界干扰具有内在鲁棒性的音频-视觉特征，从而使模型能够推广到新环境而无需专门模块。为解决架构可扩展性，我们探索如何有效扩展模型容量，同时确保多模态输入的自适应和可靠使用，开发了一种基于输入特征智能分配计算资源的框架。最后，在系统层面，我们提出通过与大尺度基础模型的模块化集成扩展系统功能的方法，利用其强大的认知和生成能力最大化最终识别精度。通过在这三个层面中的每个层面提供系统化的解决方案，本论文旨在构建一个下一代、稳健且可扩展的AVSR系统，在现实应用中具有高可靠性。


### 论文摘要

The practical deployment of Audio-Visual Speech Recognition (AVSR) systems is fundamentally challenged by significant performance degradation in real-world environments, characterized by unpredictable acoustic noise and visual interference. This dissertation posits that a systematic, hierarchical approach is essential to overcome these challenges, achieving the robust scalability at the representation, architecture, and system levels. At the representation level, we investigate methods for building a unified model that learns audio-visual features inherently robust to diverse real-world corruptions, thereby enabling generalization to new environments without specialized modules. To address architectural scalability, we explore how to efficiently expand model capacity while ensuring the adaptive and reliable use of multimodal inputs, developing a framework that intelligently allocates computational resources based on the input characteristics. Finally, at the system level, we present methods to expand the system's functionality through modular integration with large-scale foundation models, leveraging their powerful cognitive and generative capabilities to maximize final recognition accuracy. By systematically providing solutions at each of these three levels, this dissertation aims to build a next-generation, robust, and scalable AVSR system with high reliability in real-world applications.

---

## 46. OpenDataArena: A Fair and Open Arena for Benchmarking Post-Training Dataset Value

**论文链接:** [http://arxiv.org/abs/2512.14051v1](http://arxiv.org/abs/2512.14051v1)

**作者:** Mengzhang Cai, Xin Gao, Yu Li, Honglin Lin, Zheng Liu, Zhuoshi Pan, Qizhi Pei, Xiaoran Shang, Mengyuan Sun, Zinan Tang, Xiaoyang Wang, Zhanping Zhong, Yun Zhu, Dahua Lin, Conghui He, Lijun Wu

**发布时间:** 2025-12-16

### GPT解析

### 总结

OpenDataArena (ODA)是一个全面的开放平台，用于基准测试训练后数据的内在价值，解决了大型语言模型训练数据不透明的问题。

### 背景

大型语言模型(LLMs)的快速发展依赖于训练后数据集的质量和多样性，但这些数据集仍是'黑盒'，其组成、来源和评估都缺乏透明度，阻碍了可重复性并模糊了数据特征与模型行为之间的因果关系。

### 目的

引入OpenDataArena (ODA)平台，建立训练后数据评估的全面生态系统，推动数据为中心AI的研究从试错方法转向原则性科学。

### 方法

ODA建立四个关键支柱：(i)统一的训练-评估管道确保跨模型和领域的公平比较；(ii)多维评分框架沿数十个轴评估数据质量；(iii)交互式数据血统探索器可视化数据集谱系；(iv)完全开源的工具包促进数据研究。

### 主要发现

分析揭示了数据复杂性与任务性能之间的固有权衡；通过血统追踪确定了流行基准测试中的冗余；绘制了数据集之间的谱系关系图。

### 结论

ODA平台和所有相关资源已公开发布，旨在促进高质量数据评估的民主化访问，推动数据为中心AI的科学化发展。

### 翻译

大型语言模型(LLMs)的快速发展依赖于训练后数据集的质量和多样性。然而，一个关键的两难问题仍然存在：虽然模型受到严格的基准测试，但支撑它们的数据仍然是黑盒——其特征是不透明的组成、不确定的来源和缺乏系统评估。这种不透明性阻碍了可重复性，并模糊了数据特征与模型行为之间的因果关系。为了弥合这一差距，我们引入了OpenDataArena (ODA)，一个全面的开放平台，用于基准测试训练后数据的内在价值。ODA建立了由四个关键支柱组成的全面生态系统：(i)统一的训练-评估管道，确保在不同模型(如Llama、Qwen)和领域间进行公平、开放的比较；(ii)多维评分框架，沿数十个不同轴描绘数据质量；(iii)交互式数据血统探索器，可视化数据集谱系并分析组成部分来源；(iv)完全开源的工具包，用于训练、评估和评分，促进数据研究。在ODA上进行的大量实验——涵盖多个领域120多个训练数据集、22个基准测试，由600多次训练运行和4000万个处理数据点验证——揭示了重要见解。我们的分析揭示了数据复杂性与任务性能之间的固有权衡，通过血统追踪确定了流行基准测试中的冗余，并绘制了数据集之间的谱系关系。我们发布了所有结果、工具和配置，以促进对高质量数据评估的民主化访问。ODA的目标不仅仅是扩展排行榜，而是推动数据策从试错方法转向数据为中心AI的原则性科学，为数据混合规律和基础模型的战略组合的严谨研究铺平道路。


### 论文摘要

The rapid evolution of Large Language Models (LLMs) is predicated on the quality and diversity of post-training datasets. However, a critical dichotomy persists: while models are rigorously benchmarked, the data fueling them remains a black box--characterized by opaque composition, uncertain provenance, and a lack of systematic evaluation. This opacity hinders reproducibility and obscures the causal link between data characteristics and model behaviors. To bridge this gap, we introduce OpenDataArena (ODA), a holistic and open platform designed to benchmark the intrinsic value of post-training data. ODA establishes a comprehensive ecosystem comprising four key pillars: (i) a unified training-evaluation pipeline that ensures fair, open comparisons across diverse models (e.g., Llama, Qwen) and domains; (ii) a multi-dimensional scoring framework that profiles data quality along tens of distinct axes; (iii) an interactive data lineage explorer to visualize dataset genealogy and dissect component sources; and (iv) a fully open-source toolkit for training, evaluation, and scoring to foster data research. Extensive experiments on ODA--covering over 120 training datasets across multiple domains on 22 benchmarks, validated by more than 600 training runs and 40 million processed data points--reveal non-trivial insights. Our analysis uncovers the inherent trade-offs between data complexity and task performance, identifies redundancy in popular benchmarks through lineage tracing, and maps the genealogical relationships across datasets. We release all results, tools, and configurations to democratize access to high-quality data evaluation. Rather than merely expanding a leaderboard, ODA envisions a shift from trial-and-error data curation to a principled science of Data-Centric AI, paving the way for rigorous studies on data mixing laws and the strategic composition of foundation models.

---

## 47. Sparsity-Controllable Dynamic Top-p MoE for Large Foundation Model Pre-training

**论文链接:** [http://arxiv.org/abs/2512.13996v1](http://arxiv.org/abs/2512.13996v1)

**作者:** Can Jin, Hongwu Peng, Mingcan Xiang, Qixin Zhang, Xiangchi Yuan, Amit Hasan, Ohiremen Dibua, Yifan Gong, Yan Kang, Dimitris N. Metaxas

**发布时间:** 2025-12-16

### GPT解析

### 总结

本文提出了一种名为DTop-p MoE的新型稀疏专家混合架构，通过动态调整概率阈值来控制稀疏性，解决了传统Top-k路由策略的局限性，并在大型语言模型和扩散Transformer实验中表现出优于基线方法的效果。

### 背景

稀疏专家混合(MoE)架构通过只激活每个输入令牌的专家子集来有效扩展模型容量。然而，标准的Top-k路由策略采用统一的稀疏模式，忽略了令牌难度的差异。虽然Top-p路由提供了灵活的替代方案，但现有实现通常依赖固定的全局概率阈值，导致计算成本不可控且对超参数选择敏感。

### 目的

开发一种能够控制稀疏性的动态Top-p路由机制，解决优化不可微分阈值的问题，并允许不同层学习不同的专家选择模式，同时利用全局概率阈值。

### 方法

1. 提出了DTop-p MoE，一种稀疏可控的动态Top-p路由机制；2. 使用比例-积分(PI)控制器动态调整概率阈值，使运行中的激活专家稀疏度与指定目标保持一致；3. 引入动态路由归一化机制，适应层间路由logits，允许不同层学习不同的专家选择模式。

### 主要发现

1. DTop-p在大型语言模型和扩散Transformer上的实验中，始终优于Top-k和固定阈值的Top-p基线方法；2. DTop-p能够保持对激活专家数量的精确控制，同时自适应地分配不同令牌和层的资源；3. DTop-p在专家粒度、专家容量、模型大小和数据集大小方面表现出良好的扩展性。

### 结论

DTop-p为大规模MoE预训练提供了一个强大的框架，能够有效平衡模型容量和计算效率。

### 翻译

稀疏专家混合(MoE)架构通过仅激活每个输入令牌的专家子集来有效扩展模型容量。然而，标准的Top-k路由策略施加了统一的稀疏模式，忽略了令牌难度的差异。虽然Top-p路由提供了灵活的替代方案，但现有实现通常依赖固定的全局概率阈值，导致计算成本不可控且对超参数选择敏感。在本文中，我们提出了DTop-p MoE，一种稀疏可控的动态Top-p路由机制。为了解决优化不可微分阈值的挑战，我们利用比例-积分(PI)控制器动态调整概率阈值，使运行中的激活专家稀疏度与指定目标保持一致。此外，我们引入了动态路由归一化机制，适应层间路由logits，允许不同层学习不同的专家选择模式，同时利用全局概率阈值。在大型语言模型和扩散Transformer上的大量实验表明，DTop-p始终优于Top-k和固定阈值Top-p基线。我们的分析确认，DTop-p在保持对激活专家数量的精确控制的同时，自适应地分配不同令牌和层的资源。此外，DTop-p在专家粒度、专家容量、模型大小和数据集大小方面表现出强大的扩展性，为大规模MoE预训练提供了强大的框架。


### 论文摘要

Sparse Mixture-of-Experts (MoE) architectures effectively scale model capacity by activating only a subset of experts for each input token. However, the standard Top-k routing strategy imposes a uniform sparsity pattern that ignores the varying difficulty of tokens. While Top-p routing offers a flexible alternative, existing implementations typically rely on a fixed global probability threshold, which results in uncontrolled computational costs and sensitivity to hyperparameter selection. In this paper, we propose DTop-p MoE, a sparsity-controllable dynamic Top-p routing mechanism. To resolve the challenge of optimizing a non-differentiable threshold, we utilize a Proportional-Integral (PI) Controller that dynamically adjusts the probability threshold to align the running activated-expert sparsity with a specified target. Furthermore, we introduce a dynamic routing normalization mechanism that adapts layer-wise routing logits, allowing different layers to learn distinct expert-selection patterns while utilizing a global probability threshold. Extensive experiments on Large Language Models and Diffusion Transformers demonstrate that DTop-p consistently outperforms both Top-k and fixed-threshold Top-p baselines. Our analysis confirms that DTop-p maintains precise control over the number of activated experts while adaptively allocating resources across different tokens and layers. Furthermore, DTop-p exhibits strong scaling properties with respect to expert granularity, expert capacity, model size, and dataset size, offering a robust framework for large-scale MoE pre-training.

---

## 48. Informing Acquisition Functions via Foundation Models for Molecular Discovery

**论文链接:** [http://arxiv.org/abs/2512.13935v1](http://arxiv.org/abs/2512.13935v1)

**作者:** Qi Chen, Fabio Ramos, Alán Aspuru-Guzik, Florian Shkurti

**发布时间:** 2025-12-15

### GPT解析

### 总结

提出一种无需似然性的贝叶斯优化方法，通过整合大型语言模型和化学基础模型的先验知识，显著提升分子发现过程中的可扩展性、鲁棒性和样本效率。

### 背景

贝叶斯优化是加速分子发现的关键方法，通过估计分子到其性质的映射来寻找最优候选分子。传统方法在数据不足、先验知识有限和候选空间巨大的情况下性能受限。大型语言模型和化学基础模型虽能提供丰富先验，但高维特征、上下文学习成本高和深度贝叶斯代理模型的计算负担阻碍了其充分利用。

### 目的

解决大型语言模型和化学基础模型在增强贝叶斯优化时面临的挑战，开发一种无需显式代理建模的方法，直接利用这些模型的先验知识来指导采集函数。

### 方法

提出一种无需似然性的贝叶斯优化方法：1)绕过显式代理建模，直接利用通用LLMs和化学特定基础模型的先验知识；2)学习分子搜索空间的树结构分区，使用局部采集函数；3)通过蒙特卡洛树搜索实现高效候选选择；4)集成基于LLMs的粗粒度聚类，将采集函数评估限制在高性质值的簇中，提高可扩展性。

### 主要发现

通过大量实验和消融研究表明，该方法显著提高了LLM引导的贝叶斯优化在分子发现中的可扩展性、鲁棒性和样本效率。

### 结论

该方法解决了传统贝叶斯优化在低数据条件和高维特征空间中的局限性，通过整合大型语言模型和化学基础模型的先验知识，实现了更高效的分子发现过程。

### 翻译

贝叶斯优化是一种通过估计分子与其性质之间的映射来加速分子发现并寻找最优候选分子的关键方法。通常，贝叶斯优化迭代更新这种映射的概率代理模型，并优化从模型中导出的采集函数来指导分子选择。然而，在数据不足、先验知识有限和候选空间巨大的情况下，其性能受到限制。大型语言模型和化学基础模型提供了丰富的先验知识以增强贝叶斯优化，但高维特征、昂贵的上下文学习和深度贝叶斯代理模型的计算负担阻碍了它们的充分利用。为解决这些挑战，我们提出了一种无需似然性的贝叶斯优化方法，绕过显式代理建模，直接利用通用大型语言模型和化学特定基础模型的先验知识来指导采集函数。我们的方法还学习了分子搜索空间的树结构分区，并使用局部采集函数，通过蒙特卡洛树搜索实现高效的候选选择。通过进一步整合基于大型语言模型的粗粒度聚类，它通过将采集函数评估限制在具有统计上更高性质值的簇中，显著提高了对大型候选集的可扩展性。通过大量实验和消融研究，我们证明该方法显著提高了大型语言模型引导的贝叶斯优化在分子发现中的可扩展性、鲁棒性和样本效率。


### 论文摘要

Bayesian Optimization (BO) is a key methodology for accelerating molecular discovery by estimating the mapping from molecules to their properties while seeking the optimal candidate. Typically, BO iteratively updates a probabilistic surrogate model of this mapping and optimizes acquisition functions derived from the model to guide molecule selection. However, its performance is limited in low-data regimes with insufficient prior knowledge and vast candidate spaces. Large language models (LLMs) and chemistry foundation models offer rich priors to enhance BO, but high-dimensional features, costly in-context learning, and the computational burden of deep Bayesian surrogates hinder their full utilization. To address these challenges, we propose a likelihood-free BO method that bypasses explicit surrogate modeling and directly leverages priors from general LLMs and chemistry-specific foundation models to inform acquisition functions. Our method also learns a tree-structured partition of the molecular search space with local acquisition functions, enabling efficient candidate selection via Monte Carlo Tree Search. By further incorporating coarse-grained LLM-based clustering, it substantially improves scalability to large candidate sets by restricting acquisition function evaluations to clusters with statistically higher property values. We show through extensive experiments and ablations that the proposed method substantially improves scalability, robustness, and sample efficiency in LLM-guided BO for molecular discovery.

---

## 49. Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model

**论文链接:** [http://arxiv.org/abs/2512.13507v2](http://arxiv.org/abs/2512.13507v2)

**作者:** Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Tianheng Cheng, Xinqi Cheng, Xuyan Chi, Jian Cong, Jing Cui, Qinpeng Cui, Qide Dong, Junliang Fan, Jing Fang, Zetao Fang, Chengjian Feng, Han Feng, Mingyuan Gao, Yu Gao, Dong Guo, Qiushan Guo, Boyang Hao, Qingkai Hao, Bibo He, Qian He, Tuyen Hoang, Ruoqing Hu, Xi Hu, Weilin Huang, Zhaoyang Huang, Zhongyi Huang, Donglei Ji, Siqi Jiang, Wei Jiang, Yunpu Jiang, Zhuo Jiang, Ashley Kim, Jianan Kong, Zhichao Lai, Shanshan Lao, Yichong Leng, Ai Li, Feiya Li, Gen Li, Huixia Li, JiaShi Li, Liang Li, Ming Li, Shanshan Li, Tao Li, Xian Li, Xiaojie Li, Xiaoyang Li, Xingxing Li, Yameng Li, Yifu Li, Yiying Li, Chao Liang, Han Liang, Jianzhong Liang, Ying Liang, Zhiqiang Liang, Wang Liao, Yalin Liao, Heng Lin, Kengyu Lin, Shanchuan Lin, Xi Lin, Zhijie Lin, Feng Ling, Fangfang Liu, Gaohong Liu, Jiawei Liu, Jie Liu, Jihao Liu, Shouda Liu, Shu Liu, Sichao Liu, Songwei Liu, Xin Liu, Xue Liu, Yibo Liu, Zikun Liu, Zuxi Liu, Junlin Lyu, Lecheng Lyu, Qian Lyu, Han Mu, Xiaonan Nie, Jingzhe Ning, Xitong Pan, Yanghua Peng, Lianke Qin, Xueqiong Qu, Yuxi Ren, Kai Shen, Guang Shi, Lei Shi, Yan Song, Yinglong Song, Fan Sun, Li Sun, Renfei Sun, Yan Sun, Zeyu Sun, Wenjing Tang, Yaxue Tang, Zirui Tao, Feng Wang, Furui Wang, Jinran Wang, Junkai Wang, Ke Wang, Kexin Wang, Qingyi Wang, Rui Wang, Sen Wang, Shuai Wang, Tingru Wang, Weichen Wang, Xin Wang, Yanhui Wang, Yue Wang, Yuping Wang, Yuxuan Wang, Ziyu Wang, Guoqiang Wei, Wanru Wei, Di Wu, Guohong Wu, Hanjie Wu, Jian Wu, Jie Wu, Ruolan Wu, Xinglong Wu, Yonghui Wu, Ruiqi Xia, Liang Xiang, Fei Xiao, XueFeng Xiao, Pan Xie, Shuangyi Xie, Shuang Xu, Jinlan Xue, Shen Yan, Bangbang Yang, Ceyuan Yang, Jiaqi Yang, Runkai Yang, Tao Yang, Yang Yang, Yihang Yang, ZhiXian Yang, Ziyan Yang, Songting Yao, Yifan Yao, Zilyu Ye, Bowen Yu, Jian Yu, Chujie Yuan, Linxiao Yuan, Sichun Zeng, Weihong Zeng, Xuejiao Zeng, Yan Zeng, Chuntao Zhang, Heng Zhang, Jingjie Zhang, Kuo Zhang, Liang Zhang, Liying Zhang, Manlin Zhang, Ting Zhang, Weida Zhang, Xiaohe Zhang, Xinyan Zhang, Yan Zhang, Yuan Zhang, Zixiang Zhang, Fengxuan Zhao, Huating Zhao, Yang Zhao, Hao Zheng, Jianbin Zheng, Xiaozheng Zheng, Yangyang Zheng, Yijie Zheng, Jiexin Zhou, Jiahui Zhu, Kuan Zhu, Shenhan Zhu, Wenjia Zhu, Benhui Zou, Feilong Zuo

**发布时间:** 2025-12-15

**备注:** Seedance 1.5 pro Technical Report

### GPT解析

### 总结

该论文介绍了Seedance 1.5 pro模型，这是一个专门用于原生音视频联合生成的基础模型，采用双分支Diffusion Transformer架构，实现了出色的音视频同步和高质量生成，并通过多种优化技术和加速框架提高了实用性。

### 背景

视频生成领域的最新进展为统一的音视频生成铺平了道路。

### 目的

开发一个专门用于原生、联合音视频生成的基础模型，实现高质量的音视频同步和生成。

### 方法

采用双分支Diffusion Transformer架构；集成交叉模态联合模块；使用专门的多阶段数据管道；实施监督微调(SFT)在高质量数据集上；应用基于人类反馈的强化学习(RLHF)和多维奖励模型；引入加速框架提高推理速度。

### 主要发现

实现了出色的音视频同步和卓越的生成质量；通过优化技术提高了实用性；推理速度提升超过10倍；实现了精确的多语言和方言口型同步；支持动态电影摄像机控制；增强了叙事连贯性。

### 结论

Seedance 1.5 pro是一个强大的专业级内容创作引擎，已在火山引擎上提供访问。

### 翻译

最近视频生成领域的进展为统一的音视频生成铺平了道路。在这项工作中，我们提出了Seedance 1.5 pro，这是一个专门为原生、联合音视频生成而设计的基础模型。利用双分支Diffusion Transformer架构，该模型集成了交叉模态联合模块和专门的多阶段数据管道，实现了卓越的音视频同步和卓越的生成质量。为确保实用性，我们实施了精细的后训练优化，包括在高质量数据集上进行监督微调(SFT)和使用多维奖励模型进行基于人类反馈的强化学习(RLHF)。此外，我们还引入了一个加速框架，将推理速度提高了10倍以上。Seedance 1.5 pro以其精确的多语言和方言口型同步、动态电影摄像机控制和增强的叙事连贯性而脱颖而出，定位为专业级内容创作的强大引擎。Seedance 1.5 pro现已在火山引擎上可通过https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?type=GenVideo访问。


### 论文摘要

Recent strides in video generation have paved the way for unified audio-visual generation. In this work, we present Seedance 1.5 pro, a foundational model engineered specifically for native, joint audio-video generation. Leveraging a dual-branch Diffusion Transformer architecture, the model integrates a cross-modal joint module with a specialized multi-stage data pipeline, achieving exceptional audio-visual synchronization and superior generation quality. To ensure practical utility, we implement meticulous post-training optimizations, including Supervised Fine-Tuning (SFT) on high-quality datasets and Reinforcement Learning from Human Feedback (RLHF) with multi-dimensional reward models. Furthermore, we introduce an acceleration framework that boosts inference speed by over 10X. Seedance 1.5 pro distinguishes itself through precise multilingual and dialect lip-syncing, dynamic cinematic camera control, and enhanced narrative coherence, positioning it as a robust engine for professional-grade content creation. Seedance 1.5 pro is now accessible on Volcano Engine at https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?type=GenVideo.

---

## 50. Robust Beamforming for Multiuser MIMO Systems with Unknown Channel Statistics: A Hybrid Offline-Online Framework

**论文链接:** [http://arxiv.org/abs/2512.14165v1](http://arxiv.org/abs/2512.14165v1)

**作者:** Wenzhuo Zou, Ming-Min Zhao, An Liu, Min-Jian Zhao

**发布时间:** 2025-12-16

**备注:** 13 pages, 8 figures

### GPT解析

### 总结

本文提出了一种混合离线-在线框架，用于解决多用户多输入多输出系统中信道状态信息不完美时的稳健波束形成问题，特别是在信道估计误差统计未知的情况下。

### 背景

在多用户多输入多输出系统中，当信道状态信息不完美且信道估计误差统计未知时，稳健波束形成是一个基本挑战。传统方法依赖误差协方差矩阵的先验知识，而深度学习方法对未见信道条件泛化能力差。

### 目的

解决传统方法的局限性，实现有效的离线学习和快速在线适应，提高系统在未知或非平稳信道条件下的性能。

### 方法

提出混合离线-在线框架：离线阶段使用用户共享的深度神经网络学习信道估计误差协方差，无需先验统计知识；采用稀疏增广低秩方法降低复杂度；在线阶段通过少量梯度步骤快速微调网络；引入多模型无关元学习策略维护多个元初始化并动态选择最佳初始化。

### 主要发现

模拟结果表明，该框架在各种信道条件下表现出强大的鲁棒性，性能显著优于现有最先进方法。

### 结论

混合离线-在线框架能有效解决信道状态信息不完美情况下的波束形成问题，特别是在信道估计误差统计未知的情况下具有优越性能。

### 翻译

在信道状态信息不完美的情况下进行稳健波束形成设计是多用户多输入多输出系统中的一个基本挑战，特别是在信道估计误差统计未知的情况下。传统的模型驱动方法通常依赖于误差协方差矩阵的先验知识，而数据驱动的深度学习方法对未见过的信道条件泛化能力差。为解决这些限制，本文提出了一种混合离线-在线框架，实现了有效的离线学习和快速的在线适应。在离线阶段，我们提出了一种用户共享的深度神经网络，能够从观测样本中学习信道估计误差协方差，从而无需先验统计知识即可实现稳健波束形成。同时，为了便于实时部署，我们提出了一种稀疏增广低秩方法，在保持可比性能的同时降低复杂度。在线阶段，我们证明提出的网络可以用最少的梯度步骤进行快速微调。此外，进一步提出了一种多模型无关元学习策略，通过维护多个元初始化并在在线动态选择最佳初始化，可以提高该框架在未见或非平稳信道条件下的适应和泛化能力。模拟结果表明，提出的离线-在线框架在各种信道条件下表现出强大的鲁棒性，能够显著优于最先进的基线方法。


### 论文摘要

Robust beamforming design under imperfect channel state information (CSI) is a fundamental challenge in multiuser multiple-input multiple-output (MU-MIMO) systems, particularly when the channel estimation error statistics are unknown. Conventional model-driven methods usually rely on prior knowledge of the error covariance matrix and data-driven deep learning approaches suffer from poor generalization capability to unseen channel conditions. To address these limitations, this paper proposes a hybrid offline-online framework that achieves effective offline learning and rapid online adaptation. In the offline phase, we propose a shared (among users) deep neural network (DNN) that is able to learn the channel estimation error covariance from observed samples, thus enabling robust beamforming without statistical priors. Meanwhile, to facilitate real-time deployment, we propose a sparse augmented low-rank (SALR) method to reduce complexity while maintaining comparable performance. In the online phase, we show that the proposed network can be rapidly fine-tuned with minimal gradient steps. Furthermore, a multiple basis model-agnostic meta-learning (MB-MAML) strategy is further proposed to maintain multiple meta-initializations and by dynamically selecting the best one online, we can improve the adaptation and generalization capability of the proposed framework under unseen or non-stationary channels. Simulation results demonstrate that the proposed offline-online framework exhibits strong robustness across diverse channel conditions and it is able to significantly outperform state-of-the-art (SOTA) baselines.

---

## 51. EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery

**论文链接:** [http://arxiv.org/abs/2512.13857v1](http://arxiv.org/abs/2512.13857v1)

**作者:** Kamer Ali Yuksel

**发布时间:** 2025-12-15

### GPT解析

### 总结

EvoLattice是一种新颖的框架，通过在有向无环图中存储多个替代方案，解决了传统LLM引导进化方法中丢弃有用变体、遭受破坏性编辑和探索脆弱搜索空间的问题。

### 背景

大语言模型(LLMs)越来越多地用于进化和程序及多智能体系统，但大多数现有方法基于覆盖式突变，一次只维护单个候选。

### 目的

开发一种能够保留多个候选变体、避免破坏性编辑、提供更稳定搜索空间的进化框架。

### 方法

EvoLattice框架将整个候选种群表示在一个有向无环图中，每个节点存储多个持久性替代方案，每条有效路径定义一个可执行候选。通过替代级别评估提供数据驱动的反馈信号，并使用确定性自修复机制保证结构正确性。

### 主要发现

EvoLattice在程序合成方面产生了更稳定的进化、更大的表达能力和更强的改进轨迹，其动态类似于质量-多样性优化。

### 结论

EvoLattice提供了一种更有效、更稳定的方法来使用LLMs进行程序和多智能体系统的进化，能够保留成功组件并提供密集的反馈信号。

### 翻译

大型语言模型(LLMs)越来越多地用于进化和程序及多智能体系统，然而大多数现有方法依赖于基于覆盖的突变，一次只维护单个候选。此类方法丢弃有用的变体，遭受破坏性编辑，并探索一个易受结构故障影响的脆弱搜索空间。我们引入了EvoLattice，一个在有向无环图中表示整个候选程序或智能体行为种群的框架。每个节点存储多个持久性替代方案，图中的每条有效路径定义一个不同的可执行候选，产生大的组合搜索空间而不重复结构。EvoLattice通过对每个替代方案出现在所有路径中进行评分，实现细粒度的替代级别评估，产生揭示局部设计选择如何影响全局性能的统计信息。这些统计信息为LLM指导的突变、重组和修剪提供密集、数据驱动的反馈信号，同时保留成功组件。结构正确性由确定性自修复机制保证，该机制强制执行无环性和依赖一致性，独立于LLM。EvoLattice通过将替代方案解释为提示片段或子智能体行为，自然地扩展到智能体进化。在程序合成(代理和优化器元学习)方面，EvoLattice比先前的LLM引导方法产生更稳定的进化、更大的表达能力和更强的改进轨迹。所得动态类似于质量-多样性优化，从EvoLattice的内部多替代表示中隐式出现，而不是来自显式的外部档案。


### 论文摘要

Large language models (LLMs) are increasingly used to evolve programs and multi-agent systems, yet most existing approaches rely on overwrite-based mutations that maintain only a single candidate at a time. Such methods discard useful variants, suffer from destructive edits, and explore a brittle search space prone to structural failure. We introduce EvoLattice, a framework that represents an entire population of candidate programs or agent behaviors within a single directed acyclic graph. Each node stores multiple persistent alternatives, and every valid path through the graph defines a distinct executable candidate, yielding a large combinatorial search space without duplicating structure. EvoLattice enables fine-grained alternative-level evaluation by scoring each alternative across all paths in which it appears, producing statistics that reveal how local design choices affect global performance. These statistics provide a dense, data-driven feedback signal for LLM-guided mutation, recombination, and pruning, while preserving successful components. Structural correctness is guaranteed by a deterministic self-repair mechanism that enforces acyclicity and dependency consistency independently of the LLM. EvoLattice naturally extends to agent evolution by interpreting alternatives as prompt fragments or sub-agent behaviors. Across program synthesis (proxy and optimizer meta-learning), EvoLattice yields more stable evolution, greater expressivity, and stronger improvement trajectories than prior LLM-guided methods. The resulting dynamics resemble quality-diversity optimization, emerging implicitly from EvoLattice's internal multi-alternative representation rather than an explicit external archive.

---

## 52. EEG-D3: A Solution to the Hidden Overfitting Problem of Deep Learning Models

**论文链接:** [http://arxiv.org/abs/2512.13806v1](http://arxiv.org/abs/2512.13806v1)

**作者:** Siegfried Ludwig, Stylianos Bakas, Konstantinos Barmpas, Georgios Zoumpourlis, Dimitrios A. Adamos, Nikolaos Laskaris, Yannis Panagakis, Stefanos Zafeiriou

**发布时间:** 2025-12-15

### GPT解析

### 总结

该研究提出了一种名为Disentangled Decoding Decomposition (D3)的弱监督方法，用于跨EEG数据集训练深度学习模型，通过分离脑活动的潜在成分来解决隐藏的过拟合问题，提高模型在实际应用中的泛化能力。

### 背景

深度学习在解码EEG信号方面已获得关注，许多研究声称达到最先进准确率，但在受控BCI基准测试上的表现与在实际环境中缺乏泛化能力之间的脱节表明存在隐藏的过拟合问题，成功转化为实际应用的情况有限。

### 目的

引入一种弱监督方法，用于跨EEG数据集训练深度学习模型，解决隐藏的过拟合问题，提高模型在实际应用中的泛化能力，并分离脑活动的潜在成分以提高模型可解释性。

### 方法

提出D3方法，通过预测输入窗口在试验序列中的采样位置来分离脑活动的潜在成分，类似于非线性ICA；使用具有完全独立子网络的新型模型架构以确保可解释性；提出特征解释范式对比不同数据集上的成分激活曲线并检查相关时间和空间滤波器。

### 主要发现

该方法在运动想象数据上可靠分离脑活动潜在成分；在适当成分子集上训练下游分类器可防止任务相关伪影引起的隐藏过拟合；利用线性可分离潜在空间实现有效少样本学习；能区分真实脑活动成分和虚假特征，避免隐藏过拟合问题。

### 结论

该方法能生成避免隐藏过拟合且能很好泛化到实际应用的模型，仅需最少标记数据；为神经科学研究人员提供分离单个脑过程的工具，可能发现未知动态。

### 翻译

深度学习用于解码脑电图信号已获得关注，许多研究声称达到了最先进的准确率。然而，尽管基准测试性能令人信服，但成功转化为实际应用的情况有限。在受控脑机接口基准测试上的性能与缺乏泛化到实际环境能力之间的频繁脱节表明存在隐藏的过拟合问题。我们引入了解耦解码分解方法，这是一种跨脑电图数据集训练深度学习模型的弱监督方法。通过预测输入窗口在相应试验序列中的采样位置，该方法分离了脑活动的潜在成分，类似于非线性独立成分分析。我们利用一种具有完全独立子网络的新型模型架构以确保严格的可解释性。我们概述了一种特征解释范式，用于对比不同数据集上的成分激活曲线并检查相关的时间和空间滤波器。该方法在运动想象数据上可靠地分离了脑活动的潜在成分。在这些成分的适当子集上训练下游分类器可以防止由任务相关伪影引起的隐藏过拟合，这种伪影严重影响端到端分类器。我们进一步利用线性可分离的潜在空间在睡眠阶段分类中实现有效的少样本学习。能够区分脑活动的真实成分和虚假特征，使得模型能够避免隐藏的过拟合问题并很好地泛化到实际应用，同时只需要最少的标记数据。对神经科学界而言，该方法为研究人员提供了一种分离单个脑过程的工具，甚至可能发现迄今为止未知的动态。


### 论文摘要

Deep learning for decoding EEG signals has gained traction, with many claims to state-of-the-art accuracy. However, despite the convincing benchmark performance, successful translation to real applications is limited. The frequent disconnect between performance on controlled BCI benchmarks and its lack of generalisation to practical settings indicates hidden overfitting problems. We introduce Disentangled Decoding Decomposition (D3), a weakly supervised method for training deep learning models across EEG datasets. By predicting the place in the respective trial sequence from which the input window was sampled, EEG-D3 separates latent components of brain activity, akin to non-linear ICA. We utilise a novel model architecture with fully independent sub-networks for strict interpretability. We outline a feature interpretation paradigm to contrast the component activation profiles on different datasets and inspect the associated temporal and spatial filters. The proposed method reliably separates latent components of brain activity on motor imagery data. Training downstream classifiers on an appropriate subset of these components prevents hidden overfitting caused by task-correlated artefacts, which severely affects end-to-end classifiers. We further exploit the linearly separable latent space for effective few-shot learning on sleep stage classification. The ability to distinguish genuine components of brain activity from spurious features results in models that avoid the hidden overfitting problem and generalise well to real-world applications, while requiring only minimal labelled data. With interest to the neuroscience community, the proposed method gives researchers a tool to separate individual brain processes and potentially even uncover heretofore unknown dynamics.

---

## 53. Comparative Evaluation of Embedding Representations for Financial News Sentiment Analysis

**论文链接:** [http://arxiv.org/abs/2512.13749v1](http://arxiv.org/abs/2512.13749v1)

**作者:** Joyjit Roy, Samaresh Kumar Singh

**发布时间:** 2025-12-15

**备注:** 6 pages, 2 figures. Submitted to IEEE IATMSI-2026 (Track: AI, IoT and Computer Vision Enabled Technologies)

### GPT解析

### 总结

本研究评估了资源受限环境下金融新闻情感分类的嵌入方法，发现预训练嵌入在数据稀缺时收益有限，并提出了替代解决方案。

### 背景

金融情感分析有助于市场理解，但标准自然语言处理方法在小数据集应用时面临显著挑战。

### 目的

对资源受限环境下金融新闻情感分类的嵌入方法进行评估比较，寻找适合小数据集的解决方案。

### 方法

评估Word2Vec、GloVe和句子变换器表示与梯度 boosting结合的效果，使用手动标注的头条新闻进行实验。

### 主要发现

验证集和测试性能存在显著差距；模型表现不如简单基线模型；预训练嵌入在数据充足度低于临界阈值时收益递减；小型验证集导致模型选择过程中的过拟合。

### 结论

嵌入质量本身无法解决情感分类中的基本数据稀缺问题；资源有限的从业者应考虑少样本学习、数据增强或基于词典的混合方法等替代方案。

### 翻译

金融情感分析增强了市场理解；然而，当应用于小型数据集时，标准自然语言处理方法会遇到显著挑战。本研究提供了资源受限环境下基于嵌入的金融新闻情感分类方法的比较评估。Word2Vec、GloVe和句子变换器表示与梯度 boosting相结合，在手动标注的头条新闻上进行了评估。实验结果确定了验证和测试性能之间存在显著差距，尽管验证指标强劲，但模型表现不如简单基线。分析表明，预训练嵌入在数据充足度低于关键阈值时收益递减，并且小型验证集在模型选择过程中导致过拟合。通过每周情感聚合和叙事总结用于市场监控工作流程，展示了实际应用。研究结果提供了经验证据，证明仅嵌入质量无法解决情感分类中的基本数据稀缺问题。对于资源有限的从业者，结果表明当标记样本稀缺时，需要考虑替代方法，如少样本学习、数据增强或基于词典的混合方法。


### 论文摘要

Financial sentiment analysis enhances market understanding; however, standard natural language processing approaches encounter significant challenges when applied to small datasets. This study provides a comparative evaluation of embedding-based methods for financial news sentiment classification in resource-constrained environments. Word2Vec, GloVe, and sentence transformer representations are evaluated in combination with gradient boosting on manually labeled headlines. Experimental results identify a substantial gap between validation and test performance, with models performing worse than trivial baselines despite strong validation metrics. The analysis demonstrates that pretrained embeddings yield diminishing returns below a critical data sufficiency threshold, and that small validation sets contribute to overfitting during model selection. Practical application is illustrated through weekly sentiment aggregation and narrative summarization for market monitoring workflows. The findings offer empirical evidence that embedding quality alone cannot address fundamental data scarcity in sentiment classification. For practitioners operating with limited resources, the results indicate the need to consider alternative approaches such as few-shot learning, data augmentation, or lexicon-enhanced hybrid methods when labeled samples are scarce.

---

## 54. Federated Few-Shot Learning for Epileptic Seizure Detection Under Privacy Constraints

**论文链接:** [http://arxiv.org/abs/2512.13717v1](http://arxiv.org/abs/2512.13717v1)

**作者:** Ekaterina Sysoykova, Bernhard Anzengruber-Tanase, Michael Haslgrubler, Philipp Seidl, Alois Ferscha

**发布时间:** 2025-12-09

**备注:** 12 pages

### GPT解析

### 总结

该研究提出了一种两阶段联邦少样本学习框架，用于解决EEG癫痫检测中数据稀缺和隐私限制的问题。

### 背景

许多深度学习方法依赖大型集中式标注数据集，但临床实践中EEG数据稀缺、患者特定且分布在不同机构，受隐私法规限制无法集中数据。

### 目的

解决在真实医疗环境中创建可用AI癫痫检测模型的挑战，提出一种两阶段联邦少样本学习框架用于个性化EEG癫痫发作检测。

### 方法

在包含六种EEG事件类别的TUH事件语料库上训练和评估。第一阶段使用联邦学习在非独立同分布模拟医院站点上微调预训练生物信号变换器；第二阶段使用联邦少样本个性化仅用五个标记EEG段将分类器适配到每个患者。

### 主要发现

联邦微调达到平衡准确率0.43，Cohen's kappa 0.42，加权F1 0.69；FFSL阶段客户端特定模型在四个站点上平均达到平衡准确率0.77，Cohen's kappa 0.62，加权F1 0.73。

### 结论

FFSL可以在现实数据可用性和隐私限制条件下支持有效的患者自适应癫痫发作检测。

### 翻译

许多深度学习方法已被开发用于基于脑电图(EEG)的癫痫发作检测；然而，大多数依赖于访问大型集中式标注数据集。在临床实践中，EEG数据通常稀缺，患者特定且分布在各机构，并受严格隐私法规限制禁止数据集中。因此，在真实医疗环境中创建可用的AI癫痫检测模型仍然具有挑战性。为解决这些限制，我们提出了一种用于个性化EEG癫痫发作检测的两阶段联邦少样本学习(FFSL)框架。该方法在包含六种EEG事件类别的TUH事件语料库上进行训练和评估。在第一阶段，使用联邦学习在非独立同分布模拟医院站点上微调预训练的生物信号变换器(BIOT)，实现共享表示学习而不集中EEG记录。在第二阶段，联邦少样本个性化使用仅五个标记EEG段将分类器适配到每个患者，同时保留癫痫特定信息并受益于跨站点知识。联邦微调达到平衡准确率0.43(集中式:0.52)，Cohen's kappa 0.42(0.49)，和加权F1 0.69(0.74)。在FFSL阶段，客户端特定模型在四个具有异构事件分布的站点上平均达到平衡准确率0.77，Cohen's kappa 0.62，和加权F1 0.73。这些结果表明，FFSL可以在现实数据可用性和隐私限制条件下支持有效的患者自适应癫痫发作检测。


### 论文摘要

Many deep learning approaches have been developed for EEG-based seizure detection; however, most rely on access to large centralized annotated datasets. In clinical practice, EEG data are often scarce, patient-specific distributed across institutions, and governed by strict privacy regulations that prohibit data pooling. As a result, creating usable AI-based seizure detection models remains challenging in real-world medical settings. To address these constraints, we propose a two-stage federated few-shot learning (FFSL) framework for personalized EEG-based seizure detection. The method is trained and evaluated on the TUH Event Corpus, which includes six EEG event classes. In Stage 1, a pretrained biosignal transformer (BIOT) is fine-tuned across non-IID simulated hospital sites using federated learning, enabling shared representation learning without centralizing EEG recordings. In Stage 2, federated few-shot personalization adapts the classifier to each patient using only five labeled EEG segments, retaining seizure-specific information while still benefiting from cross-site knowledge. Federated fine-tuning achieved a balanced accuracy of 0.43 (centralized: 0.52), Cohen's kappa of 0.42 (0.49), and weighted F1 of 0.69 (0.74). In the FFSL stage, client-specific models reached an average balanced accuracy of 0.77, Cohen's kappa of 0.62, and weighted F1 of 0.73 across four sites with heterogeneous event distributions. These results suggest that FFSL can support effective patient-adaptive seizure detection under realistic data-availability and privacy constraints.

---

## 55. Meta Hierarchical Reinforcement Learning for Scalable Resource Management in O-RAN

**论文链接:** [http://arxiv.org/abs/2512.13715v1](http://arxiv.org/abs/2512.13715v1)

**作者:** Fatemeh Lotfi, Fatemeh Afghah

**发布时间:** 2025-12-08

**备注:** This paper is submitted to IEEE Open Journal of the Communications Society

### GPT解析

### 总结

论文提出了一种自适应的Meta-Hierarchical Reinforcement Learning框架，用于优化O-RAN中的资源分配和网络切片，结合分层控制和元学习实现全局和本地适应，提高网络管理效率和适应性。

### 背景

现代应用复杂性要求无线网络具备实时适应性和高效资源管理能力。O-RAN架构及其RIC模块成为动态资源管理和网络切片的关键解决方案，但大多数AI驱动方法在不可预测和高度动态条件下难以保持性能。

### 目的

提出一种自适应的Meta-Hierarchical Reinforcement Learning框架，基于Model Agnostic Meta Learning灵感，联合优化O-RAN中的资源分配和网络切片。

### 方法

框架结合分层控制与元学习：高层控制器在各切片间分配资源，低级代理执行切片内调度。自适应元更新机制通过时间差分误差方差加权任务，提高稳定性并优先处理复杂网络场景。理论分析建立了两级学习过程的亚线性收敛性和遗憾保证。

### 主要发现

模拟显示该方法比基线方法提高19.8%的网络管理效率，具有更快适应性和更高的eMBB、URLLC和mMTC切片服务质量满意度。消融和可扩展性研究证实了方法的鲁棒性，实现高达40%的更快速适应，并在网络规模扩大时保持一致的公平性、延迟和吞吐量性能。

### 结论

自适应Meta-Hierarchical Reinforcement Learning框架在O-RAN环境中提供了有效的资源分配和网络切片解决方案，能够适应动态变化的网络条件并在各种场景下保持高性能。

### 翻译

现代应用日益复杂，要求无线网络具备实时适应性和高效资源管理能力。具有RAN智能控制器模块的开放无线接入网络架构已成为动态资源管理和网络切片的关键解决方案。尽管人工智能驱动的方法显示出潜力，但大多数方法在不可预测和高度动态的条件下难以保持性能。本文提出了一种自适应的Meta-Hierarchical Reinforcement Learning框架，其灵感来自Model Agnostic Meta Learning，用于联合优化O-RAN中的资源分配和网络切片。该框架将分层控制与元学习相结合，实现全局和本地适应：高层控制器在各切片间分配资源，而低级代理执行切片内调度。自适应元更新机制通过时间差分误差方差加权任务，提高稳定性并优先处理复杂的网络场景。理论分析建立了两级学习过程的亚线性收敛性和遗憾保证。模拟结果显示，与基线RL和meta-RL方法相比，该方法提高了19.8%的网络管理效率，同时具有更快的适应性和更高的eMBB、URLLC和mMTC切片的服务质量满意度。额外的消融和可扩展性研究证实了该方法的鲁棒性，实现了高达40%的更快速适应，并在网络规模扩大时保持一致的公平性、延迟和吞吐量性能。


### 论文摘要

The increasing complexity of modern applications demands wireless networks capable of real time adaptability and efficient resource management. The Open Radio Access Network (O-RAN) architecture, with its RAN Intelligent Controller (RIC) modules, has emerged as a pivotal solution for dynamic resource management and network slicing. While artificial intelligence (AI) driven methods have shown promise, most approaches struggle to maintain performance under unpredictable and highly dynamic conditions. This paper proposes an adaptive Meta Hierarchical Reinforcement Learning (Meta-HRL) framework, inspired by Model Agnostic Meta Learning (MAML), to jointly optimize resource allocation and network slicing in O-RAN. The framework integrates hierarchical control with meta learning to enable both global and local adaptation: the high-level controller allocates resources across slices, while low level agents perform intra slice scheduling. The adaptive meta-update mechanism weights tasks by temporal difference error variance, improving stability and prioritizing complex network scenarios. Theoretical analysis establishes sublinear convergence and regret guarantees for the two-level learning process. Simulation results demonstrate a 19.8% improvement in network management efficiency compared with baseline RL and meta-RL approaches, along with faster adaptation and higher QoS satisfaction across eMBB, URLLC, and mMTC slices. Additional ablation and scalability studies confirm the method's robustness, achieving up to 40% faster adaptation and consistent fairness, latency, and throughput performance as network scale increases.

---

## 56. A Graph-Based Forensic Framework for Inferring Hardware Noise of Cloud Quantum Backend

**论文链接:** [http://arxiv.org/abs/2512.14541v1](http://arxiv.org/abs/2512.14541v1)

**作者:** Subrata Das, Archisman Ghosh, Swaroop Ghosh

**发布时间:** 2025-12-16

**备注:** 11 pages, 5 figures, conference

### GPT解析

### 总结

本文介绍了一种基于图神经网络(GNN)的法医框架，用于预测未知量子处理器的单量子比特和量子比特链路错误率，仅使用拓扑信息和转译电路特征，无需访问校准数据。

### 背景

云量子平台让用户可访问多种不同量子比特技术、耦合布局和噪声级别的后端，但电路执行依赖于用户无法观察的内部分配和路由策略。提供商可能将作业重定向到错误率更高的区域，导致保真度下降但仍提供过时的校准数据，造成用户无法验证电路是否在计费硬件上执行的安全漏洞。

### 目的

开发一种法医方法，从用户可见的工件推断后端行为，使用户能够验证量子电路执行环境并评估实际错误率。

### 方法

构建基于图神经网络(GNN)的法医框架，从IBM 27量子比特设备收集数据，合并静态校准特征和动态转译特征，为单量子比特和双量子比特错误分别训练GNN回归器，并在推理时从用户可访问的特征重建完整错误图。

### 主要发现

模型在目标后端上准确恢复后端错误率，单量子比特错误平均不匹配度约22%，量子比特链路错误约18%；预测错误值排序与实际校准错误高度匹配(Spearman相关性高)；框架能持续识别弱链路和高噪声量子比特，在现实噪声漂移条件下保持鲁棒性。

### 结论

基于GNN的法医框架能从用户可访问信息准确预测量子处理器错误特性，为用户提供验证电路执行环境的方法，填补了云量子平台透明度方面的安全漏洞。

### 翻译

云量子平台让用户可以访问具有不同量子比特技术、耦合布局和噪声级别的多种后端。然而，电路的执行依赖于用户无法观察到的内部分配和路由策略。提供商可能会将作业重定向到错误率更高的区域以节省资源、平衡负载或其他不透明原因，从而导致保真度下降，同时仍然提供过时或平均的校准数据。这种缺乏透明性创造了一个安全漏洞：用户无法验证他们的电路是否在为其计费的硬件上执行。因此，从用户可见的工件推断后端行为的法医方法变得至关重要。在这项工作中，我们介绍了一种基于图神经网络(GNN)的法医框架，该框架仅使用拓扑信息和从转译电路中提取的聚合特征来预测未知后端的单量子比特和量子比特链路错误率。我们从几个IBM 27量子比特设备构建数据集，将静态校准特征与动态转译特征合并，并为单量子比特和双量子比特错误分别训练GNN回归器。在推理时，模型无需访问目标后端的校准数据，并从用户可用的特征重建完整的错误图。我们在目标后端上的结果显示，后端错误率的准确恢复，单量子比特错误的平均不匹配度约为22%，量子比特链路错误的平均不匹配度约为18%。该模型还表现出强排名一致性，预测错误值排序与实际校准错误排序高度匹配，如高Spearman相关性所示。该框架持续识别弱链路和高噪声量子比特，并在现实的噪声漂移条件下保持鲁棒性。


### 论文摘要

Cloud quantum platforms give users access to many backends with different qubit technologies, coupling layouts, and noise levels. The execution of a circuit, however, depends on internal allocation and routing policies that are not observable to the user. A provider may redirect jobs to more error-prone regions to conserve resources, balance load or for other opaque reasons, causing degradation in fidelity while still presenting stale or averaged calibration data. This lack of transparency creates a security gap: users cannot verify whether their circuits were executed on the hardware for which they were charged. Forensic methods that infer backend behavior from user-visible artifacts are therefore becoming essential. In this work, we introduce a Graph Neural Network (GNN)-based forensic framework that predicts per-qubit and per-qubit link error rates of an unseen backend using only topology information and aggregated features extracted from transpiled circuits. We construct a dataset from several IBM 27-qubit devices, merge static calibration features with dynamic transpilation features and train separate GNN regressors for one- and two-qubit errors. At inference time, the model operates without access to calibration data from the target backend and reconstructs a complete error map from the features available to the user. Our results on the target backend show accurate recovery of backend error rate, with an average mismatch of approximately 22% for single-qubit errors and 18% for qubit-link errors. The model also exhibits strong ranking agreement, with the ordering induced by predicted error values closely matching that of the actual calibration errors, as reflected by high Spearman correlation. The framework consistently identifies weak links and high-noise qubits and remains robust under realistic temporal noise drift.

---

## 57. Dual-Axis RCCL: Representation-Complete Convergent Learning for Organic Chemical Space

**论文链接:** [http://arxiv.org/abs/2512.14418v1](http://arxiv.org/abs/2512.14418v1)

**作者:** Dejun Hu, Zhiming Li, Jia-Rui Shen, Jia-Ning Tu, Zi-Hao Ye, Junliang Zhang

**发布时间:** 2025-12-16

**备注:** 33 pages, 10 figures

### GPT解析

### 总结

该研究提出了一种双轴表示-完全收敛学习(RCCL)策略，通过整合图卷积网络和无桥图编码实现分子表示的完整性，开发了FD25数据集，实现了有机分子的近乎完全组合覆盖，并展示了图神经网络的优异预测性能。

### 背景

机器学习正在深刻改变分子和材料建模领域，但由于化学空间的巨大规模（10的30次方到10的60次方），模型能否在这个空间中实现收敛学习仍然是一个开放的科学问题。

### 目的

解决机器学习模型在庞大化学空间中实现收敛学习的科学问题，建立分子表示、结构完整性和模型泛化之间的定量联系，为可解释、可迁移和数据高效的分子智能奠定基础。

### 方法

提出双轴表示-完全收敛学习策略，开发整合图卷积网络编码局部价键环境和无桥图编码环/笼状拓扑的分子表示方法，基于RCCL框架开发FD25数据集，覆盖13,302个局部价键单元和165,726个环/笼状拓扑，使用图神经网络进行训练。

### 主要发现

图神经网络在FD25数据集上训练后表现出表示完全收敛学习和强分布外泛化能力，在外部基准测试中整体预测误差约为1.0千卡/摩尔平均绝对误差，建立了分子表示、结构完整性和模型泛化之间的定量联系。

### 结论

RCCL框架和FD25数据集为分子建模提供了新方法，实现了近乎完全的组合覆盖，并展示了优异的预测性能，为可解释、可迁移和数据高效的分子智能奠定了基础。

### 翻译

机器学习正在深刻重塑分子和材料建模；然而，鉴于化学空间的巨大规模（10的30次方到10的60次方），模型能否在这个空间中实现收敛学习仍然是一个开放的科学问题。我们引入了一种双轴表示-完全收敛学习策略，通过一种分子表示方法实现该方法，该方法整合了基于现代价键理论的局部价键环境的图卷积网络编码，以及环/笼状拓扑的无桥图编码，提供了化学空间覆盖度的定量度量。该框架形式化了表示完整性，为构建支持大模型收敛学习的数据集建立了原则性基础。在该RCCL框架指导下，我们开发了FD25数据集，系统覆盖了13,302个局部价键单元和165,726个环/笼状拓扑，实现了含有H/C/N/O/F元素的有机分子的近乎完全组合覆盖。在FD25上训练的图神经网络表现出表示完全收敛学习和强分布外泛化能力，在外部基准测试中整体预测误差约为1.0千卡/摩尔平均绝对误差。我们的结果建立了分子表示、结构完整性和模型泛化之间的定量联系，为可解释、可迁移和数据高效的分子智能奠定了基础。


### 论文摘要

Machine learning is profoundly reshaping molecular and materials modeling; however, given the vast scale of chemical space (10^30-10^60), it remains an open scientific question whether models can achieve convergent learning across this space. We introduce a Dual-Axis Representation-Complete Convergent Learning (RCCL) strategy, enabled by a molecular representation that integrates graph convolutional network (GCN) encoding of local valence environments, grounded in modern valence bond theory, together with no-bridge graph (NBG) encoding of ring/cage topologies, providing a quantitative measure of chemical-space coverage. This framework formalizes representation completeness, establishing a principled basis for constructing datasets that support convergent learning for large models. Guided by this RCCL framework, we develop the FD25 dataset, systematically covering 13,302 local valence units and 165,726 ring/cage topologies, achieving near-complete combinatorial coverage of organic molecules with H/C/N/O/F elements. Graph neural networks trained on FD25 exhibit representation-complete convergent learning and strong out-of-distribution generalization, with an overall prediction error of approximately 1.0 kcal/mol MAE across external benchmarks. Our results establish a quantitative link between molecular representation, structural completeness, and model generalization, providing a foundation for interpretable, transferable, and data-efficient molecular intelligence.

---

## 58. Graph Signal Denoising Using Regularization by Denoising and Its Parameter Estimation

**论文链接:** [http://arxiv.org/abs/2512.14213v1](http://arxiv.org/abs/2512.14213v1)

**作者:** Hayate Kojima, Hiroshi Higashi, Yuichi Tanaka

**发布时间:** 2025-12-16

**备注:** Submitted to APSIPA Transactions on Signal and Information Processing

### GPT解析

### 总结

本文提出了一种基于去噪正则化(RED)的图信号可解释去噪方法，展示了RED技术在图信号处理中的应用价值，并提出了基于深度算法展开的参数估计方法，提高了算法在无监督设置下的适用性。

### 背景

RED是一种最初为图像恢复开发的技术，在优化问题的正则化项中使用高效的去噪器。通过RED，可以明确使用去噪器设计优化问题，并在温和条件下轻松计算正则化项的梯度。

### 目的

将RED技术扩展应用于图像处理之外的图信号去噪领域，提高图信号去噪的准确性和可解释性。

### 方法

适应RED用于图信号去噪；证明多种图信号去噪器（包括图神经网络）满足RED条件；从图滤波器角度研究RED的有效性；提出基于深度算法展开的监督和无监督参数估计方法。

### 主要发现

许多图信号去噪器理论上或实践上满足RED条件；从图滤波器角度验证了RED的有效性；提出的参数估计方法提高了算法适用性，特别是在无监督设置下。

### 结论

在合成和真实数据集的去噪实验中，所提方法在均方误差方面比现有图信号去噪方法表现更好，证明了RED技术在图信号去噪中的有效性。

### 翻译

在本文中，我们提出了一种基于去噪正则化(RED)的图信号可解释去噪方法。RED是一种为图像恢复开发的技术，在优化问题的正则化项中使用高效（有时是黑盒）的去噪器。通过使用RED，可以明确使用去噪器设计优化问题，并在温和条件下轻松计算正则化项的梯度。我们将RED适应用于图像处理之外的图信号去噪。我们证明许多图信号去噪器，包括图神经网络，理论上或实践上满足RED的条件。我们还从图滤波器的角度研究了RED的有效性。此外，我们提出了基于深度算法展开的监督和无监督参数估计方法。这些方法旨在提高算法的适用性，特别是在无监督设置下。合成和真实数据集的去噪实验表明，我们提出的方法在均方误差方面提高了信号去噪精度，比现有的图信号去噪方法更好。


### 论文摘要

In this paper, we propose an interpretable denoising method for graph signals using regularization by denoising (RED). RED is a technique developed for image restoration that uses an efficient (and sometimes black-box) denoiser in the regularization term of the optimization problem. By using RED, optimization problems can be designed with the explicit use of the denoiser, and the gradient of the regularization term can be easily computed under mild conditions. We adapt RED for denoising of graph signals beyond image processing. We show that many graph signal denoisers, including graph neural networks, theoretically or practically satisfy the conditions for RED. We also study the effectiveness of RED from a graph filter perspective. Furthermore, we propose supervised and unsupervised parameter estimation methods based on deep algorithm unrolling. These methods aim to enhance the algorithm applicability, particularly in the unsupervised setting. Denoising experiments for synthetic and real-world datasets show that our proposed method improves signal denoising accuracy in mean squared error compared to existing graph signal denoising methods.

---

## 59. Multivariate Time Series Forecasting with Hybrid Euclidean-SPD Manifold Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2512.14023v1](http://arxiv.org/abs/2512.14023v1)

**作者:** Yong Fang, Na Li, Hangguan Shan, Eryun Liu, Xinyu Li, Wei Ni, Er-Ping Li

**发布时间:** 2025-12-16

### GPT解析

### 总结

本文提出了一种名为HSMGNN的混合对称正定流形图神经网络，用于多元时间序列预测。该模型结合了欧几里得和黎曼几何框架，能够更有效地捕捉现实数据中的复杂几何结构和时空依赖关系。

### 背景

多元时间序列预测在交通管理和预测性维护等实际应用中具有重要作用。现有方法通常在欧几里得或黎曼空间中建模MTS数据，限制了它们捕捉现实世界中数据多样几何结构和复杂时空依赖的能力。

### 目的

克服现有方法的局限性，提出一种能够捕捉数据几何特性的混合几何表示框架，用于更准确和全面的多元时间序列预测。

### 方法

作者提出了HSMGNN模型，包括三个关键组件：(1)子流形-交叉段(SCS)嵌入，将输入MTS投影到欧几里得和黎曼空间；(2)自适应距离库(ADB)层，通过可训练记忆机制降低黎曼距离的计算成本；(3)融合图卷积网络(FGCN)，通过可学习融合算子整合双空间特征进行准确预测。

### 主要发现

实验结果表明，在三个基准数据集上，HSMGNN在预测准确率上比最先进的基线方法提高了最多13.8%，证明了混合几何表示在MTS预测中的有效性。

### 结论

HSMGNN是首个利用混合几何表示进行MTS预测的工作，通过结合欧几里得和黎曼框架，能够更全面地建模数据的几何特性，从而提高预测性能。

### 翻译

多元时间序列预测在交通管理和预测性维护等各种实际应用中发挥着至关重要的作用。现有方法通常在欧几里得或黎曼空间中对MTS数据进行建模，限制了它们捕捉现实数据中固有的多样几何结构和复杂时空依赖的能力。为了克服这一限制，我们提出了混合对称正定流形图神经网络(HSMGNN)，这是一种基于图神经网络的创新模型，能够在混合欧几里得-黎曼框架内捕捉数据几何特性。据我们所知，这是首个利用混合几何表示进行MTS预测的工作，能够表达和全面建模几何特性。具体而言，我们引入了子流形-交叉段(SCS)嵌入，将输入MTS投影到欧几里得和黎曼空间，从而在不同几何域中捕捉时空变化。为了减轻黎曼距离的高计算成本，我们进一步设计了一个具有可训练记忆机制的自适应距离库(ADB)层。最后，我们设计了一个融合图卷积网络(FGCN)，通过可学习融合算子整合双空间特征，以实现准确预测。在三个基准数据集上的实验表明，HSMGNN在预测准确率上比最先进的基线方法提高了最多13.8%。


### 论文摘要

Multivariate Time Series (MTS) forecasting plays a vital role in various real-world applications, such as traffic management and predictive maintenance. Existing approaches typically model MTS data in either Euclidean or Riemannian space, limiting their ability to capture the diverse geometric structures and complex spatio-temporal dependencies inherent in real-world data. To overcome this limitation, we propose the Hybrid Symmetric Positive-Definite Manifold Graph Neural Network (HSMGNN), a novel graph neural network-based model that captures data geometry within a hybrid Euclidean-Riemannian framework. To the best of our knowledge, this is the first work to leverage hybrid geometric representations for MTS forecasting, enabling expressive and comprehensive modeling of geometric properties. Specifically, we introduce a Submanifold-Cross-Segment (SCS) embedding to project input MTS into both Euclidean and Riemannian spaces, thereby capturing spatio-temporal variations across distinct geometric domains. To alleviate the high computational cost of Riemannian distance, we further design an Adaptive-Distance-Bank (ADB) layer with a trainable memory mechanism. Finally, a Fusion Graph Convolutional Network (FGCN) is devised to integrate features from the dual spaces via a learnable fusion operator for accurate prediction. Experiments on three benchmark datasets demonstrate that HSMGNN achieves up to a 13.8 percent improvement over state-of-the-art baselines in forecasting accuracy.

---

## 60. A Complete Guide to Spherical Equivariant Graph Transformers

**论文链接:** [http://arxiv.org/abs/2512.13927v1](http://arxiv.org/abs/2512.13927v1)

**作者:** Sophia Tang

**发布时间:** 2025-12-15

**备注:** This paper is a technical version of the article originally published in Alchemy Bio (99 pages, 46 figures)

### GPT解析

### 总结

球面等变图神经网络(EGNNs)为三维分子和生物分子系统学习提供了尊重物理旋转对称性的框架，通过将节点和边特征表示为球面张量，扩展了传统GNN和Transformer，并提供了完整的理论基础和实现指南。

### 背景

在三维分子和生物分子系统中，预测必须尊重物理学中固有的旋转对称性，这需要专门的神经网络架构来处理。

### 目的

开发一个完整的、直观的球面等变建模基础，并构建相应的架构，使研究人员能够理解和实现球面EGNNs应用于化学、分子性质预测等领域。

### 方法

将节点和边特征表示为在旋转群SO(3)不可约表示下变换的球面张量，构建SO(3)-等变核，并开发张量场网络和SE(3)-Transformer架构，实现等变消息传递和注意力操作。

### 主要发现

通过球面等变建模，可以确保神经网络在输入旋转时预测以物理上有意义的方式变化，这对于分子和生物分子系统学习至关重要。

### 结论

球面EGNNs为处理具有旋转对称性的三维数据提供了有效框架，该指南通过清晰的数学推导和代码示例，为研究人员提供了理解和实现这些模型的全面资源。

### 翻译

球面等变图神经网络(EGNNs)为学习三维分子和生物分子系统提供了原则性框架，其中预测必须尊重物理学中固有的旋转对称性。这些模型通过将节点和边特征表示为在旋转群SO(3)不可约表示下变换的球面张量，扩展了传统的消息传递GNN和Transformer，确保在输入旋转时预测以物理上有意义的方式变化。本指南从群表示和球谐函数开始，建立了完整的球面等变建模直观基础，包括张量积、Clebsch-Gordan分解和SO(3)-等变核的构建。基于这一基础，我们构建了张量场网络和SE(3)-Transformer架构，并解释了它们如何在几何图上进行等变消息传递和注意力操作。通过清晰的数学推导和带注释的代码片段，本指南为寻求理解或实现球面EGNNs应用于化学、分子性质预测、蛋白质结构建模和生成建模的研究人员和学习者提供了自包含的介绍。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是三维分子和生物分子系统学习中的旋转对称性问题。标准神经网络在处理三维分子数据时无法保持旋转对称性，导致模型在旋转输入后可能无法识别相同的分子结构，从而造成效率低下、泛化能力差或物理不一致的预测。这个问题在研究中非常重要，因为分子系统的许多特性不依赖于它们在空间中的绝对方向，粒子间的物理相互作用在旋转下可预测地变换。保持旋转对称性对于准确建模分子间相互作用、预测分子性质、蛋白质结构建模和生成建模等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到标准神经网络无法处理三维数据的旋转对称性，然后引入群表示理论，特别是SO(3)旋转群，作为处理旋转对称性的数学框架。他们使用球形张量表示节点和边特征，这些张量在SO(3)的不可约表示下变换，并构建等变核确保预测在输入旋转时以物理上有意义的方式变化。作者借鉴了多项现有工作，包括SE(3)-Transformers论文（Fuchs et al., 2020）、Deep Graph Library (DGL)框架，以及量子力学中的球谐函数、不可约表示和Clebsch-Gordan系数等概念。论文还基于现有的球形等变图神经网络（EGNNs）框架，如Fuchs et al., Geiger and Smidt, Thomas等人的工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用球形张量表示节点和边特征，这些张量在SO(3)旋转群的不可约表示下变换，构建等变核函数确保模型在输入旋转时保持等变性，并将消息传递和注意力机制设计为等变的以保持物理一致性。整体实现流程包括：1)将点云转换为几何图，节点代表原子或残基；2)将节点特征表示为球形张量列表；3)使用球谐函数将边的位移向量投影到球形张量；4)构建结合球谐函数、Clebsch-Gordan分解和可学习径向函数的等变核；5)实现等变消息传递机制（Tensor Field Network）；6)构建基于注意力的等变架构（SE(3)-Transformer）；7)应用到分子性质预测任务中。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提供了球形等变图神经网络的完整指南，从基础数学概念到实际实现；2)清晰解释了复杂的数学概念，如群表示、球谐函数、张量积和Clebsch-Gordan分解；3)将Tensor Field Network和SE(3)-Transformer架构统一在球形等变框架下；4)提供了详细的数学推导和带注释的代码实现；5)构建了完整的工具包，使研究人员能够实现球形等变神经网络。相比之前的工作，本指南提供了更全面和直观的基础理论解释，包含更多实现细节和代码示例，降低了入门门槛，更系统地解释了球形等变建模的各个方面，并将不同球形等变架构统一在一个框架下。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过提供球形等变图神经网络的完整理论基础和实用指南，使研究人员能够构建尊重物理旋转对称性的三维分子系统模型，从而提高了分子性质预测和生物分子建模的准确性和一致性。'}


### 论文摘要

Spherical equivariant graph neural networks (EGNNs) provide a principled framework for learning on three-dimensional molecular and biomolecular systems, where predictions must respect the rotational symmetries inherent in physics. These models extend traditional message-passing GNNs and Transformers by representing node and edge features as spherical tensors that transform under irreducible representations of the rotation group SO(3), ensuring that predictions change in physically meaningful ways under rotations of the input. This guide develops a complete, intuitive foundation for spherical equivariant modeling - from group representations and spherical harmonics, to tensor products, Clebsch-Gordan decomposition, and the construction of SO(3)-equivariant kernels. Building on this foundation, we construct the Tensor Field Network and SE(3)-Transformer architectures and explain how they perform equivariant message-passing and attention on geometric graphs. Through clear mathematical derivations and annotated code excerpts, this guide serves as a self-contained introduction for researchers and learners seeking to understand or implement spherical EGNNs for applications in chemistry, molecular property prediction, protein structure modeling, and generative modeling.

---

## 61. Network-Wide Traffic Volume Estimation from Speed Profiles using a Spatio-Temporal Graph Neural Network with Directed Spatial Attention

**论文链接:** [http://arxiv.org/abs/2512.13758v1](http://arxiv.org/abs/2512.13758v1)

**作者:** Léo Hein, Giovanni de Nunzio, Giovanni Chierchia, Aurélie Pirayre, Laurent Najman

**发布时间:** 2025-12-15

### GPT解析

### 总结

本文提出了混合定向注意力时空图神经网络(HDA-STGNN)，一种用于网络范围交通量估计的深度学习框架，利用速度曲线、静态道路属性和道路网络拓扑预测所有路段的每日交通量曲线，无需在推理时依赖交通量数据。

### 背景

现有交通量估计方法通常只处理配备传感器道路的交通量预测或使用附近传感器进行空间插补。预测模型不考虑未监测道路，而空间插补方法虽处理网络范围估计，但在推理时依赖交通量数据，限制了在传感器稀少城市中的应用。

### 目的

提出一种利用更广泛可用的探测车辆速度数据和静态道路属性，而非依赖交通量数据进行网络范围交通量估计的方法。

### 方法

开发混合定向注意力时空图神经网络(HDA-STGNN)框架，利用速度曲线、静态道路属性和道路网络拓扑来预测网络中所有路段的每日交通量曲线。

### 主要发现

通过消融研究证明模型能捕捉复杂时空依赖关系，并强调拓扑信息对在没有推理时交通量数据的情况下准确进行网络范围交通量估计的价值。

### 结论

HDA-STGNN提供了一种有效方法，可在不依赖推理时交通量数据的情况下，利用速度曲线、静态道路属性和道路网络拓扑进行网络范围交通量估计。

### 翻译

现有的交通量估计方法通常只处理预测配备传感器的道路上的交通量或使用附近传感器进行空间插补来估算缺失的交通量这两种情况之一。虽然预测模型在设计上通常不考虑未监测的道路，但空间插补方法明确处理网络范围的估计；然而，这种方法在推理时依赖于交通量数据，限制了其在传感器稀少城市中的适用性。与交通量数据不同，探测车辆速度和静态道路属性更广泛可得，并支持大多数城市网络中路段的完全覆盖。在这项工作中，我们提出了混合定向注意力时空图神经网络(HDA-STGNN)，这是一个归纳深度学习框架，旨在解决网络范围的交通量估计问题。我们的方法利用速度曲线、静态道路属性和道路网络拓扑来预测网络中所有路段的每日交通量曲线。为了评估我们方法的有效性，我们进行了大量的消融研究，证明了模型捕捉复杂时空依赖关系的能力，并强调了拓扑信息在没有推理时交通量数据的情况下进行准确网络范围交通量估计的价值。


### 论文摘要

Existing traffic volume estimation methods typically address either forecasting traffic on sensor-equipped roads or spatially imputing missing volumes using nearby sensors. While forecasting models generally disregard unmonitored roads by design, spatial imputation methods explicitly address network-wide estimation; yet this approach relies on volume data at inference time, limiting its applicability in sensor-scarce cities. Unlike traffic volume data, probe vehicle speeds and static road attributes are more broadly accessible and support full coverage of road segments in most urban networks. In this work, we present the Hybrid Directed-Attention Spatio-Temporal Graph Neural Network (HDA-STGNN), an inductive deep learning framework designed to tackle the network-wide volume estimation problem. Our approach leverages speed profiles, static road attributes, and road network topology to predict daily traffic volume profiles across all road segments in the network. To evaluate the effectiveness of our approach, we perform extensive ablation studies that demonstrate the model's capacity to capture complex spatio-temporal dependencies and highlight the value of topological information for accurate network-wide traffic volume estimation without relying on volume data at inference time.

---

