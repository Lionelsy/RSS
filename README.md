# 今日论文推荐 - 2025-10-08

共 64 篇论文

---

## 1. Cross-Embodiment Dexterous Hand Articulation Generation via Morphology-Aware Learning

**论文链接:** [http://arxiv.org/abs/2510.06068v1](http://arxiv.org/abs/2510.06068v1)

**作者:** Heng Zhang, Kevin Yuchen Ma, Mike Zheng Shou, Weisi Lin, Yan Wu

**发布时间:** 2025-10-07

### GPT解析

### 总结

该研究提出了一种基于特征抓取的端到端框架，用于跨实体形态的灵巧抓取生成，解决了多指手抓取中的高维关节运动和优化成本问题，实现了在不同形态手之间的有效泛化。

### 背景

灵巧的多指手抓取具有挑战性，主要由于高维关节运动和基于优化的管道的高成本。现有的端到端方法需要在特定手的大规模数据集上进行训练，限制了它们在不同实体形态之间泛化的能力。

### 目的

开发一种能够跨不同实体形态进行抓取生成的端到端框架，无需针对每种手型进行大规模数据集训练。

### 方法

从手的形态描述中推导形态嵌入和特征抓取集，结合物体点云和手腕姿态，通过振幅预测器在低维空间回归关节系数，再解码为完整关节运动。使用运动感知关节损失(KAL)进行监督学习，强调指尖相关运动并注入形态特定结构。

### 主要发现

在模拟环境中对三种灵巧手和未见物体测试，平均抓取成功率达91.9%，每次抓取推理时间少于0.4秒；通过少样本适应对未见手，在模拟中未见物体上达到85.6%成功率；真实世界实验中少样本泛化手达到87%成功率。

### 结论

所提出的特征抓取端到端框架能够有效处理不同形态手的灵巧抓取问题，具有高泛化能力和计算效率，无需针对每种手型进行大规模数据集训练。

### 翻译

灵巧的多指手抓取仍然具有挑战性，这是由于高维关节运动和基于优化的管道的高成本。现有的端到端方法需要在特定手的大规模数据集上进行训练，限制了它们在不同实体形态之间泛化的能力。我们提出了一种基于特征抓取的端到端框架，用于跨实体形态的抓取生成。从手的形态描述中，我们推导出形态嵌入和特征抓取集。基于这些，结合物体点云和手腕姿态，振幅预测器在低维空间回归关节系数，这些系数被解码为完整的关节运动。关节学习通过运动感知关节损失(KAL)进行监督，该损失强调与指尖相关的运动并注入形态特定的结构。在模拟中对三种灵巧手的未见物体进行测试，我们的模型平均抓取成功率达到91.9%，每次抓取推理时间少于0.4秒。通过对未见手的少样本适应，在模拟中未见物体上达到85.6%的成功率，而在这个少样本泛化手的真实世界实验中达到87%的成功率。代码和附加材料将在我们的项目网站上发布：https://connor-zh.github.io/cross_embodiment_dexterous_grasping。


### 论文摘要

Dexterous grasping with multi-fingered hands remains challenging due to high-dimensional articulations and the cost of optimization-based pipelines. Existing end-to-end methods require training on large-scale datasets for specific hands, limiting their ability to generalize across different embodiments. We propose an eigengrasp-based, end-to-end framework for cross-embodiment grasp generation. From a hand's morphology description, we derive a morphology embedding and an eigengrasp set. Conditioned on these, together with the object point cloud and wrist pose, an amplitude predictor regresses articulation coefficients in a low-dimensional space, which are decoded into full joint articulations. Articulation learning is supervised with a Kinematic-Aware Articulation Loss (KAL) that emphasizes fingertip-relevant motions and injects morphology-specific structure. In simulation on unseen objects across three dexterous hands, our model attains a 91.9% average grasp success rate with less than 0.4 seconds inference per grasp. With few-shot adaptation to an unseen hand, it achieves 85.6% success on unseen objects in simulation, and real-world experiments on this few-shot generalized hand achieve an 87% success rate. The code and additional materials will be made available upon publication on our project website https://connor-zh.github.io/cross_embodiment_dexterous_grasping.

---

## 2. Carré du champ flow matching: better quality-generalisation tradeoff in generative models

**论文链接:** [http://arxiv.org/abs/2510.05930v1](http://arxiv.org/abs/2510.05930v1)

**作者:** Jacob Bamberger, Iolo Jones, Dennis Duncan, Michael M. Bronstein, Pierre Vandergheynst, Adam Gosztolai

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种名为Carré du champ flow matching (CDC-FM)的新方法，通过几何感知噪声正则化概率路径，改善了深度生成模型中样本质量和泛化能力之间的权衡关系。

### 背景

深度生成模型通常面临基本权衡：高样本质量可能导致模型记忆化训练数据而非泛化到基础数据几何结构上。

### 目的

引入CDC-FM方法，作为flow matching(FM)的泛化，通过几何感知噪声正则化概率路径，改善质量-泛化权衡。

### 方法

用空间变化、各向异性高斯噪声替代FM中的同质、各向同性噪声，这种噪声的协方差能捕获潜在数据流形的局部几何结构；证明几何噪声可从数据中最优估计且可扩展到大数据；在多种数据集和神经网络架构上进行了实验评估。

### 主要发现

CDC-FM始终提供更好的质量-泛化权衡；在数据稀缺环境和高度非均匀采样数据集中相比标准FM有显著改进；这些情况在AI科学应用中经常遇到。

### 结论

该工作为研究生成模型中数据几何、泛化和记忆化之间的相互作用提供了数学框架；提供了一种稳健且可扩展的算法，可轻松集成到现有flow matching流程中。

### 翻译

深度生成模型通常面临一个基本权衡：高样本质量可能以记忆化为代价，即模型重现训练数据而非泛化到基础数据几何结构上。我们引入了Carré du champ flow matching (CDC-FM)，这是flow matching(FM)的一种泛化，通过用几何感知噪声正则化概率路径来改善质量-泛化权衡。我们的方法用空间变化、各向异性高斯噪声替代FM中的同质、各向同性噪声，其协方差能够捕获潜在数据流形的局部几何结构。我们证明这种几何噪声可以从数据中最优估计，并且可扩展到大数据集。此外，我们在多种数据集（合成流形、点云、单细胞基因组学、动物运动捕捉和图像）以及各种神经网络架构（MLP、CNN和transformers）上提供了广泛的实验评估。我们证明CDC-FM始终提供更好的质量-泛化权衡。我们在数据稀缺环境和高度非均匀采样数据集中观察到相比标准FM的显著改进，这些情况在AI科学应用中经常遇到。我们的工作为研究生成模型中数据几何、泛化和记忆化之间的相互作用提供了数学框架，以及一种可以轻松集成到现有flow matching流程中的稳健且可扩展的算法。


### 论文摘要

Deep generative models often face a fundamental tradeoff: high sample quality can come at the cost of memorisation, where the model reproduces training data rather than generalising across the underlying data geometry. We introduce Carr\'e du champ flow matching (CDC-FM), a generalisation of flow matching (FM), that improves the quality-generalisation tradeoff by regularising the probability path with a geometry-aware noise. Our method replaces the homogeneous, isotropic noise in FM with a spatially varying, anisotropic Gaussian noise whose covariance captures the local geometry of the latent data manifold. We prove that this geometric noise can be optimally estimated from the data and is scalable to large data. Further, we provide an extensive experimental evaluation on diverse datasets (synthetic manifolds, point clouds, single-cell genomics, animal motion capture, and images) as well as various neural network architectures (MLPs, CNNs, and transformers). We demonstrate that CDC-FM consistently offers a better quality-generalisation tradeoff. We observe significant improvements over standard FM in data-scarce regimes and in highly non-uniformly sampled datasets, which are often encountered in AI for science applications. Our work provides a mathematical framework for studying the interplay between data geometry, generalisation and memorisation in generative models, as well as a robust and scalable algorithm that can be readily integrated into existing flow matching pipelines.

---

## 3. ALISE: Annotation-Free LiDAR Instance Segmentation for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2510.05752v1](http://arxiv.org/abs/2510.05752v1)

**作者:** Yongxuan Lyu, Guangfeng Jiang, Hongsi Liu, Jun Liu

**发布时间:** 2025-10-07

### GPT解析

### 总结

该研究介绍了ALISE，一种无需任何标注即可进行激光雷达实例分割的新型框架，通过视觉基础模型生成初始伪标签，结合时空投票模块和语义监督策略，在无监督3D实例分割任务上建立了新的最先进水平。

### 背景

户外激光雷达点云的手动实例标注极其耗时且成本高昂。当前方法试图减轻这一负担，但仍依赖于某种形式的人工标注。

### 目的

完全消除对人工标注的依赖，开发一种能够在没有任何标注的情况下进行激光雷达实例分割的方法。

### 方法

1. 使用视觉基础模型(VFMs)在文本和图像指导下生成初始伪标签；2. 通过时空投票模块优化标签，结合2D和3D语义进行离线和在线优化；3. 引入基于2D先验的损失函数将视觉知识注入3D网络；4. 使用基于原型的对比损失利用3D语义一致性构建判别性特征空间。

### 主要发现

该综合设计带来了显著的性能提升，为无监督3D实例分割建立了新的最先进水平。ALISE甚至超过了使用真实2D边界框监督的MWSIS方法，在平均精度(mAP)上以2.53%的优势(50.95%对比48.42%)胜出。

### 结论

ALISE框架成功地实现了无需任何标注的激光雷达实例分割，通过创新的伪标签生成和优化方法，以及语义监督策略，在性能上超越了现有方法，甚至超过了有监督的方法。

### 翻译

户外激光雷达点云的手动实例标注极其耗时且成本高昂。当前方法试图减轻这一负担，但仍依赖于某种形式的人工标注。为了完全消除这种依赖，我们引入了ALISE，一种无需任何标注即可执行激光雷达实例分割的新型框架。核心挑战是以完全无监督的方式生成高质量的伪标签。我们的方法首先使用视觉基础模型(VFMs)，在文本和图像的指导下生成初始伪标签。然后，我们通过专门的时空投票模块优化这些标签，该模块结合2D和3D语义进行离线和在线优化。为了实现卓越的特征学习，我们进一步引入了两种形式的语义监督：一组基于2D先验的损失函数，将视觉知识注入3D网络；以及一种新颖的基于原型的对比损失，通过利用3D语义一致性构建判别性特征空间。这种全面的设计带来了显著的性能提升，为无监督3D实例分割建立了新的最先进水平。值得注意的是，我们的方法甚至超过了MWSIS(一种使用真实2D边界框监督的方法)，在平均精度(mAP)上以2.53%的优势(50.95%对比48.42%)胜出。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决激光雷达点云实例分割任务中依赖密集人工标注的问题。这个问题在现实中非常重要，因为3D点云标注成本极高、耗时极长，限制了自动驾驶感知技术的发展和应用范围。完全消除标注依赖可以大幅降低数据获取成本，加速自动驾驶技术的普及和应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有弱监督方法仍需某种形式人工标注的局限性，然后借鉴了视觉基础模型(VFMs)如GroundingDINO和SAM的能力，设计了ALISE框架。该方法融合了多阶段伪标签精炼(离线精炼OFR和在线精炼ONR)和多方面监督方案(VPD和PCL模块)，通过时序一致性和跨模态知识传递来提升性能。作者也参考了对比学习、知识蒸馏等现有技术，但创新性地将其组合应用于完全无标注的3D实例分割场景。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用视觉基础模型生成高质量的3D点云实例分割伪标签，并通过多阶段精炼和知识蒸馏机制，使3D分割网络能够从这些伪标签中学习，实现无需任何人工标注的实例分割。整体流程包括：1)无监督伪标签生成(UPG)利用VFMs生成初始标签；2)两阶段精炼策略，包括利用相邻帧信息的离线精炼(OFR)和利用网络自身预测的在线精炼(ONR)；3)多方面监督训练，包括VFMs先验知识蒸馏(VPD)和基于原型的对比学习(PCL)；4)结合多个损失项进行网络训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)完全无标注框架，消除任何形式的人工标注依赖；2)创新的伪标签生成与精炼管道，保留VFMs的语义分布而非直接生成one-hot标签；3)多阶段精炼策略，结合离线和在线优化；4)多方面监督方案，包括软标签蒸馏和动态原型对比学习。相比之前工作，ALISE不仅超越了多种弱监督方法，甚至在某些情况下超过了使用GT 2D边界框监督的方法，且仅需少量标注数据微调就能超过全监督基线。之前的无监督方法通常生成硬标签且缺乏时序一致性考虑，而ALISE通过软标签和时序投票解决了这些问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ALISE提出了一种完全无标注的激光雷达实例分割框架，通过多阶段伪标签精炼和知识蒸馏技术，显著提升了无监督3D实例分割性能，甚至在某些情况下超越了弱监督方法。'}


### 论文摘要

The manual annotation of outdoor LiDAR point clouds for instance segmentation is extremely costly and time-consuming. Current methods attempt to reduce this burden but still rely on some form of human labeling. To completely eliminate this dependency, we introduce ALISE, a novel framework that performs LiDAR instance segmentation without any annotations. The central challenge is to generate high-quality pseudo-labels in a fully unsupervised manner. Our approach starts by employing Vision Foundation Models (VFMs), guided by text and images, to produce initial pseudo-labels. We then refine these labels through a dedicated spatio-temporal voting module, which combines 2D and 3D semantics for both offline and online optimization. To achieve superior feature learning, we further introduce two forms of semantic supervision: a set of 2D prior-based losses that inject visual knowledge into the 3D network, and a novel prototype-based contrastive loss that builds a discriminative feature space by exploiting 3D semantic consistency. This comprehensive design results in significant performance gains, establishing a new state-of-the-art for unsupervised 3D instance segmentation. Remarkably, our approach even outperforms MWSIS, a method that operates with supervision from ground-truth (GT) 2D bounding boxes by a margin of 2.53% in mAP (50.95% vs. 48.42%).

---

## 4. PointNSP: Autoregressive 3D Point Cloud Generation with Next-Scale Level-of-Detail Prediction

**论文链接:** [http://arxiv.org/abs/2510.05613v1](http://arxiv.org/abs/2510.05613v1)

**作者:** Ziqiao Meng, Qichao Wang, Zhiyang Dou, Zixing Song, Zhipeng Zhou, Irwin King, Peilin Zhao

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种名为PointNSP的自回归点云生成框架，通过多尺度分解解决了传统自回归方法中的序列偏差问题，首次实现了在自回归范式中达到最先进的点云生成质量。

### 背景

自回归点云生成在质量上一直落后于基于扩散的方法，这是因为自回归模型对本质上无序的点集施加了人为排序，导致形状生成本质上是一系列局部预测的序列过程。

### 目的

解决自回归点云生成中的序列偏差问题，使模型能够捕获长程依赖关系并强制执行全局结构属性，如对称性、一致拓扑和大尺度几何规律性。

### 方法

受形状建模中细节层次（LOD）原则的启发，提出了PointNSP，一种从粗到细的生成框架。该框架在低分辨率保留全局形状结构，通过下一尺度预测范式逐步在高尺度细化精细几何，使自回归目标与点集的排列不变性质保持一致。

### 主要发现

在ShapeNet上的实验表明，PointNSP首次在自回归范式内建立了最先进的生成质量；在参数、训练和推理效率方面超过了基于扩散的基线方法；在8,192点的密集生成中，其优势更加明显，突显了可扩展性潜力。

### 结论

PointNSP通过多尺度分解成功解决了自回归点云生成中的序列偏差问题，实现了高质量、高效率的点云生成，为自回归方法在点云生成领域的应用开辟了新途径。

### 翻译

自回归点云生成在质量上一直落后于基于扩散的方法。这种性能差距源于自回归模型对本质上无序的点集施加了人为排序，迫使形状生成本质上作为一系列局部预测的序列过程。这种序列偏差强调短程连续性，但损害了模型捕获长程依赖关系的能力，阻碍了其强制执行全局结构属性（如对称性、一致拓扑和大尺度几何规律性）的能力。受形状建模中细节层次（LOD）原则的启发，我们提出了PointNSP，一种从粗到细的生成框架，在低分辨率保留全局形状结构，并通过下一尺度预测范式逐步在高尺度细化精细几何。这种多尺度分解使自回归目标与点集的排列不变性质保持一致，实现了丰富的尺度内交互，同时避免了脆弱的固定排序。ShapeNet上的实验表明，PointNSP首次在自回归范式内建立了最先进的生成质量。此外，它在参数、训练和推理效率方面超过了强大的基于扩散的基线方法。最后，在8,192点的密集生成中，PointNSP的优势变得更加明显，突显了其可扩展性潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D点云生成中自回归模型质量落后于扩散模型的问题。传统自回归方法对无序点集施加人为顺序，导致生成过程偏向局部预测而难以捕捉全局结构。这个问题很重要，因为点云是3D物体形状的基本表示，广泛应用于自动驾驶、机器人感知、计算机辅助设计等领域，高质量点云生成对于形状合成、重建等任务至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受形状建模中细节层次(LOD)原则启发，结合视觉自回归建模在2D图像生成中的成功经验，设计了从粗到细的生成框架。方法借鉴了VAR(视觉自回归模型)的多尺度预测思想，并采用Farthest Point Sampling(FPS)算法获取多尺度表示，同时创新性地设计了多尺度残差向量量化和位置感知的注意力机制，以适应3D点云的无序特性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过预测下一尺度的细节层次(LoD)来逐步生成3D点云，而非传统方法中的逐点预测。整体流程包括：1)使用FPS算法获取多尺度LoD序列；2)通过残差查询提取多尺度特征并使用RVQ量化为标记；3)训练自回归Transformer进行下一尺度预测，结合块对角因果掩码和位置感知软掩码建模尺度内和尺度间交互；4)从最粗尺度开始逐步生成，最终组合所有尺度重建完整点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)从粗到细的生成策略，预测整个尺度而非单个点；2)保持点集的排列不变性；3)多尺度因子化对齐自回归目标与排列不变性质；4)高效的多尺度特征表示；5)位置感知的注意力机制。相比传统自回归方法，PointNSP保留了全局结构；相比扩散方法，它避免了高计算成本的迭代去噪；相比VAR方法，它专门针对3D点云特性进行了优化，并在参数效率和生成速度上有显著优势。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PointNSP提出了一种从粗到细的自回归3D点云生成框架，通过预测下一尺度的细节层次，在保持排列不变性的同时实现了最先进的生成质量和显著的效率提升。'}


### 论文摘要

Autoregressive point cloud generation has long lagged behind diffusion-based approaches in quality. The performance gap stems from the fact that autoregressive models impose an artificial ordering on inherently unordered point sets, forcing shape generation to proceed as a sequence of local predictions. This sequential bias emphasizes short-range continuity but undermines the model's capacity to capture long-range dependencies, hindering its ability to enforce global structural properties such as symmetry, consistent topology, and large-scale geometric regularities. Inspired by the level-of-detail (LOD) principle in shape modeling, we propose PointNSP, a coarse-to-fine generative framework that preserves global shape structure at low resolutions and progressively refines fine-grained geometry at higher scales through a next-scale prediction paradigm. This multi-scale factorization aligns the autoregressive objective with the permutation-invariant nature of point sets, enabling rich intra-scale interactions while avoiding brittle fixed orderings. Experiments on ShapeNet show that PointNSP establishes state-of-the-art (SOTA) generation quality for the first time within the autoregressive paradigm. In addition, it surpasses strong diffusion-based baselines in parameter, training, and inference efficiency. Finally, in dense generation with 8,192 points, PointNSP's advantages become even more pronounced, underscoring its scalability potential.

---

## 5. Human Action Recognition from Point Clouds over Time

**论文链接:** [http://arxiv.org/abs/2510.05506v1](http://arxiv.org/abs/2510.05506v1)

**作者:** James Dickens

**发布时间:** 2025-10-07

### GPT解析

### 总结

这篇论文提出了一种新颖的3D视频动作识别方法，通过点云分割、跟踪和身体部位分割流程，结合基于点技术和稀疏卷积网络，实现了与现有骨骼动作识别算法相媲美的性能，并在集成设置中达到了89.3%的准确率。

### 背景

当前人类动作识别研究主要集中在骨骼动作识别和基于视频的方法。随着消费级深度传感器和激光雷达设备的普及，有机会利用密集的3D数据进行动作识别，开发第三种方法。

### 目的

开发一种利用3D视频进行动作识别的新方法，形成不同于骨骼动作识别和基于视频方法的第三种途径。

### 方法

提出了一种3D视频动作识别方法，引入了从场景背景分割人体点云、随时间跟踪个体、执行身体部位分割的流程。支持深度传感器和单目深度估计的点云，核心是一种结合基于点技术与稀疏卷积网络的新骨干网络。实验中结合了表面法线、颜色、红外强度和身体部位解析标签等辅助特征提高识别准确性。

### 主要发现

在NTU RGB-D 120数据集上的评估表明，该方法与现有骨骼动作识别算法具有竞争力。在集成设置中结合基于传感器和估计的深度输入，当训练和测试使用不同受试者时，达到89.3%的准确率，超过了之前的点云动作识别方法。

### 结论

该方法提供了一种利用密集3D数据进行动作识别的有效途径，结合不同来源的深度数据可以进一步提高性能。

### 翻译

最近的人类动作识别研究主要集中在骨骼动作识别和基于视频的方法上。随着消费级深度传感器和激光雷达设备的日益普及，利用密集的3D数据进行动作识别的机会正在增加，从而开发第三种方法。本文通过引入一个从场景背景分割人体点云、随时间跟踪个体并执行身体部位分割的流程，提出了一种从3D视频识别动作的新方法。该方法支持来自深度传感器和单目深度估计的点云。所提出的HAR框架核心是一种新的3D动作识别骨干网络，它将基于点技术与应用于体素映射点云序列的稀疏卷积网络相结合。实验包括表面法线、颜色、红外强度和身体部位解析标签等辅助点特征，以提高识别准确性。在NTU RGB-D 120数据集上的评估表明，该方法与现有的骨骼动作识别算法具有竞争力。此外，在集成设置中结合基于传感器和估计的深度输入，当考虑不同受试者进行训练和测试时，该方法达到89.3%的准确率，超过了之前的点云动作识别方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从3D点云序列中识别人类动作的问题。现有的人体动作识别研究主要基于视频或骨骼数据，而随着深度传感器和激光雷达设备的普及，利用密集3D数据进行动作识别的机会增加。这个问题在现实中非常重要，因为它可以应用于监控系统中实现异常和暴力检测的自动化，帮助识别老年人跌倒，实现视频自动标注（特别是在体育分析中），以及在自动驾驶中确保行人和驾驶员的安全。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有两种主要动作识别方法（视频方法和骨骼方法）的局限性，然后指出现有点云动作识别方法的不足，包括缺乏人体分割、身体部位分割和辅助特征使用。基于这些分析，作者设计了一个新流程，适用于有深度传感器和只有RGB视频两种场景。作者借鉴了现有的深度学习技术，如M2FP模型进行实例分割、ByteTrack算法进行人物跟踪、迭代最远点采样进行点采样、T-Net进行点云嵌入，以及稀疏卷积神经网络进行特征提取。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用点云数据中的3D几何信息进行动作识别，结合点处理技术和稀疏卷积网络，通过人体分割和身体部位分割去除背景噪声，并提取更精细的特征。整体流程分为两个阶段：1) 数据预处理阶段：对输入图像进行实例和身体部分分割，进行掩码去噪，将实例投影到3D空间，去除异常点，进行人物跟踪，采样点云并计算表面法线；2) 动作识别模型阶段：使用T-Net嵌入层获取全局信息，将点云映射到体素网格，使用稀疏CNN骨干网络提取特征，进行全局池化，最后通过全连接层进行动作分类。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 全新的点云动作识别流程，同时进行实例和身体部分分割；2) 新颖的骨干网络架构SP-HP-ConvoT，结合点处理技术和稀疏卷积网络；3) 利用表面法线、红外/RGB颜色和身体部位标签等多种辅助特征；4) 适用于深度传感器和单目深度估计两种场景。相比之前的工作，该方法不依赖于关键点估计（避免误差），保留了更丰富的3D几何信息，进行了人体分割减少背景噪声，且不需要深度传感器也能工作。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种新颖的从3D点云序列识别人类动作的方法，通过结合人体分割、身体部位分割和稀疏卷积网络，在NTU RGB-D 120数据集上达到了最先进的性能，并且适用于深度传感器和单目深度估计两种场景。'}


### 论文摘要

Recent research into human action recognition (HAR) has focused predominantly on skeletal action recognition and video-based methods. With the increasing availability of consumer-grade depth sensors and Lidar instruments, there is a growing opportunity to leverage dense 3D data for action recognition, to develop a third way. This paper presents a novel approach for recognizing actions from 3D videos by introducing a pipeline that segments human point clouds from the background of a scene, tracks individuals over time, and performs body part segmentation. The method supports point clouds from both depth sensors and monocular depth estimation. At the core of the proposed HAR framework is a novel backbone for 3D action recognition, which combines point-based techniques with sparse convolutional networks applied to voxel-mapped point cloud sequences. Experiments incorporate auxiliary point features including surface normals, color, infrared intensity, and body part parsing labels, to enhance recognition accuracy. Evaluation on the NTU RGB- D 120 dataset demonstrates that the method is competitive with existing skeletal action recognition algorithms. Moreover, combining both sensor-based and estimated depth inputs in an ensemble setup, this approach achieves 89.3% accuracy when different human subjects are considered for training and testing, outperforming previous point cloud action recognition methods.

---

## 6. Tensor-on-tensor Regression Neural Networks for Process Modeling with High-dimensional Data

**论文链接:** [http://arxiv.org/abs/2510.05329v1](http://arxiv.org/abs/2510.05329v1)

**作者:** Qian Wang, Mohammad N. Bisheh, Kamran Paynabar

**发布时间:** 2025-10-06

### GPT解析

### 总结

这篇论文介绍了一种新的张量对张量回归神经网络（TRNN），用于处理现代传感和计量系统产生的大规模异构高维数据。

### 背景

现代传感和计量系统现在正在传输TB级的异构、高维数据，这些数据的自然表示是多向张量。

### 目的

开发一种回归模型，既能保持张量几何结构，又能捕获主导工业和机械过程的显著非线性相互作用。

### 方法

提出了一种张量对张量回归神经网络（TRNN），统一了基于张量的回归器和传统神经网络的范式。

### 主要发现

现有的基于张量的回归器本质上是线性的，而传统神经网络在处理时会丢失空间结构并导致参数数量过多。

### 结论

TRNN能够统一保持张量几何结构和表达非线性这两个范式，为处理高维数据提供了新的解决方案。

### 翻译

现代传感和计量系统现在正在传输TB级的异构、高维数据，这些数据的自然表示是多向张量。理解这类数据需要保持张量几何结构的回归模型，同时要有足够的表达能力来捕获主导许多工业和机械过程的显著非线性相互作用。现有的基于张量的回归器满足第一个要求但本质上是线性的。相反，传统的神经网络只有在展平后才提供非线性，从而丢弃了空间结构并导致参数数量过多。本文引入了一个张量对张量回归神经网络（TRNN），它统一了这两个范式。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何在处理高维数据时同时保留数据的张量结构（多路结构）并捕获数据间的非线性关系。这个问题在现实中非常重要，因为现代工业环境中传感器和测量系统产生大量异构高维数据（如数据曲线、图像和点云），这些数据自然表示为多路张量。理解这些数据需要既能保留张量几何结构又能表达复杂非线性相互作用的回归模型，这对实现更深层次的过程理解、更早的故障检测和更严格的质量控制至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：基于张量的回归器保留结构但本质上是线性的，而传统神经网络在展平后才能提供非线性，导致丢失空间结构和参数过多。作者认识到需要统一这两种范式，借鉴了自编码器的编码器-解码器架构，但进行了张量化改进。同时，作者借鉴了张量分解技术（特别是Tucker分解）来减少参数数量，并利用ReLU激活函数引入非线性，还展示了他们的一个线性特例可以简化为偏最小二乘法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用张量神经网络同时保留多路数据结构和捕捉非线性关系，采用类似自编码器的架构但使用张量操作代替向量操作，并在中间层引入收缩算子允许不同阶数的张量映射。整体流程分为三部分：1）编码器通过收缩Tucker层逐步减少输入张量维度，每层后接ReLU激活；2）收缩层（瓶颈）使用爱因斯坦收缩积将压缩特征映射到潜在表示，可改变张量阶数；3）解码器通过扩展ReLU层和Tucker层重建输出张量结构。训练时使用均方误差作为损失函数，通过推导的前向和反向传播公式进行端到端训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）首个非线性张量-张量回归神经网络框架；2）张量化自编码器架构，使用可学习Tucker层保留多线性结构；3）瓶颈处引入收缩算子允许不同阶数张量映射；4）通过Tucker分解显著减少参数数量。相比之前工作，与线性张量回归方法相比能捕捉非线性关系，RMSE降低最多45%；与传统神经网络相比保留张量结构、减少参数、降低过拟合风险；与近期相关工作相比不强制输入输出张量同阶、不限制表达能力在特定阶段。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种创新的张量-张量回归神经网络框架，能够在保留高维数据多路结构的同时有效捕捉非线性关系，显著提高了工业过程建模的准确性和效率。'}


### 论文摘要

Modern sensing and metrology systems now stream terabytes of heterogeneous, high-dimensional (HD) data profiles, images, and dense point clouds, whose natural representation is multi-way tensors. Understanding such data requires regression models that preserve tensor geometry, yet remain expressive enough to capture the pronounced nonlinear interactions that dominate many industrial and mechanical processes. Existing tensor-based regressors meet the first requirement but remain essentially linear. Conversely, conventional neural networks offer nonlinearity only after flattening, thereby discarding spatial structure and incurring prohibitive parameter counts. This paper introduces a Tensor-on-Tensor Regression Neural Network (TRNN) that unifies these two paradigms.

---

## 7. Identification in source apportionment using geometry

**论文链接:** [http://arxiv.org/abs/2510.03616v2](http://arxiv.org/abs/2510.03616v2)

**作者:** Bora Jin, Abhirup Datta

**发布时间:** 2025-10-04

### GPT解析

### 总结

本研究解决了源解析分析中非负矩阵分解(NMF)的非唯一性和依赖不可验证假设的问题，建立了更弱条件下源归因百分比矩阵的可识别性，并提出了一种几何估计方法。

### 背景

源解析分析旨在量化多种空气污染物观测浓度对特定来源的归因，通常表述为非负矩阵分解(NMF)问题，但NMF是非唯一的，且依赖于不可验证的假设如稀疏性和不可解释的缩放比例。

### 目的

建立源归因百分比矩阵在更弱和更现实条件下的可识别性，开发一种不依赖稀疏性或参数分布假设的几何估计方法。

### 方法

引入源归因百分比矩阵的总体估计量，证明其尺度不变性和可识别性；将数据视为锥壳中的点云，开发几何估计量，无需稀疏性或参数分布假设，同时适应时空依赖性。

### 主要发现

源归因百分比矩阵的几何估计量在无需稀疏性或参数分布假设的情况下是一致的，能够适应时空依赖性，即使在NMF因子不可识别时仍然可识别。

### 结论

数值实验验证了所提出的理论和方法的有效性。

### 翻译

源解析分析旨在量化观测到的多种空气污染物浓度对特定来源的归因，可以表述为非负矩阵分解(NMF)问题。然而，NMF是非唯一的，并且通常依赖于不可验证的假设，如稀疏性和不可解释的缩放比例。在本手稿中，我们在更弱和更现实的条件下建立了源归因百分比矩阵的可识别性。我们引入了该矩阵的总体估计量，并证明即使NMF因子不可识别，它也是尺度不变的且可识别的。将数据视为锥壳中的一个点云，我们展示了源归因百分比矩阵的几何估计量是一致的，无需任何稀疏性或参数分布假设，同时能够适应时空依赖性。数值实验验证了这一理论。


### 论文摘要

Source apportionment analysis, which aims to quantify the attribution of observed concentrations of multiple air pollutants to specific sources, can be formulated as a non-negative matrix factorization (NMF) problem. However, NMF is non-unique and typically relies on unverifiable assumptions such as sparsity and uninterpretable scalings. In this manuscript, we establish identifiability of the source attribution percentage matrix under much weaker and more realistic conditions. We introduce the population-level estimand for this matrix, and show that it is scale-invariant and identifiable even when the NMF factors are not. Viewing the data as a point cloud in a conical hull, we show that a geometric estimator of the source attribution percentage matrix is consistent without any sparsity or parametric distributional assumptions, and while accommodating spatio-temporal dependence. Numerical experiments corroborate the theory.

---

## 8. Analyzing the Effect of Embedding Norms and Singular Values to Oversmoothing in Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.06066v1](http://arxiv.org/abs/2510.06066v1)

**作者:** Dimitrios Kelesis, Dimitris Fotakis, Georgios Paliouras

**发布时间:** 2025-10-07

### GPT解析

### 总结

本研究探讨了深度图神经网络中过平滑现象的影响因素，通过提出新的MASED指标量化过平滑程度，并基于理论分析提出了G-Reg正则化方案，有效提高了深度GNN的分类准确率。

### 背景

深度图神经网络在处理图结构数据时会出现过平滑问题，即随着网络层数增加，节点嵌入逐渐趋同，导致模型性能下降。这种现象限制了GNN的深度和表达能力。

### 目的

本研究旨在量化分析过平滑现象的影响因素，提出减轻过平滑的方法，并探索深度GNN的有效训练策略，以提高节点分类任务的性能。

### 方法

1. 提出了新的MASED指标来量化过平滑程度；2. 推导了MASED的逐层界限，聚合成全局上下界；3. 分析了节点嵌入范数和权重矩阵奇异值对过平滑的影响；4. 基于理论分析提出了G-Reg正则化方案；5. 通过实验验证了方法的有效性。

### 主要发现

1. 过平滑程度随可训练权重矩阵数量和邻接矩阵数量的增加而增加；2. 通过G-Reg正则化方案增加MASED界限可以提高节点分类准确率；3. 减少深度网络中的过平滑可以使模型在某些任务上获得比浅层网络更好的结果；4. 存在感受野大小与性能之间的权衡关系，通过合理分配邻接跳跃可以避免欠参数化或过参数化。

### 结论

通过理论分析和实验验证，本研究有效量化了图神经网络中的过平滑现象，并提出了减轻过平滑的正则化方案，使深度GNN能够在保持高性能的同时增加网络深度，为图神经网络的设计和训练提供了新的见解。

### 翻译

在本文中，我们研究了导致深度图神经网络过平滑效应的因素。具体而言，我们的分析基于一个新的指标（平均平方距离-MASED）来量化过平滑的程度。我们推导了MASED的逐层界限，这些界限聚合成全局上下距离界限。基于对过平滑的量化，我们进一步分析了模型两个不同属性的重要性；即生成节点嵌入的范数，以及权重矩阵的最大和最小奇异值。基于从理论分析中获得的见解，我们展示了过平滑随着可训练权重矩阵数量和邻接矩阵数量的增加而增加。我们还使用推导出的MASED逐层界限提出了将跳跃次数（即邻接深度）与权重矩阵数量解耦的方案。特别是，我们引入了G-Reg，一种增加界限的正则化方案，并通过大量实验证明，这样做可以提高节点分类准确率，在大深度下实现鲁棒性。我们进一步表明，通过减少深度网络中的过平滑，我们可以在某些任务上获得比使用浅层网络更好的结果。具体来说，我们在'冷启动'场景下进行了实验，即当未标记节点没有特征信息时。最后，我们使用MASED界限 empirically 证明了感受野大小（即权重矩阵数量）与性能之间的权衡。这是通过将邻接跳跃分布在少量可训练层中实现的，避免了GNN的欠参数化或过参数化的极端情况。


### 论文摘要

In this paper, we study the factors that contribute to the effect of oversmoothing in deep Graph Neural Networks (GNNs). Specifically, our analysis is based on a new metric (Mean Average Squared Distance - $MASED$) to quantify the extent of oversmoothing. We derive layer-wise bounds on $MASED$, which aggregate to yield global upper and lower distance bounds. Based on this quantification of oversmoothing, we further analyze the importance of two different properties of the model; namely the norms of the generated node embeddings, along with the largest and smallest singular values of the weight matrices. Building on the insights drawn from the theoretical analysis, we show that oversmoothing increases as the number of trainable weight matrices and the number of adjacency matrices increases. We also use the derived layer-wise bounds on $MASED$ to form a proposal for decoupling the number of hops (i.e., adjacency depth) from the number of weight matrices. In particular, we introduce G-Reg, a regularization scheme that increases the bounds, and demonstrate through extensive experiments that by doing so node classification accuracy increases, achieving robustness at large depths. We further show that by reducing oversmoothing in deep networks, we can achieve better results in some tasks than using shallow ones. Specifically, we experiment with a ``cold start" scenario, i.e., when there is no feature information for the unlabeled nodes. Finally, we show empirically the trade-off between receptive field size (i.e., number of weight matrices) and performance, using the $MASED$ bounds. This is achieved by distributing adjacency hops across a small number of trainable layers, avoiding the extremes of under- or over-parameterization of the GNN.

---

## 9. A comprehensive comparison of neural operators for 3D industry-scale engineering designs

**论文链接:** [http://arxiv.org/abs/2510.05995v1](http://arxiv.org/abs/2510.05995v1)

**作者:** Weiheng Zhong, Qibang Liu, Diab Abueidda, Seid Koric, Hadi Meidani

**发布时间:** 2025-10-07

### GPT解析

### 总结

该研究提出了六个代表性的3D工业规模工程设计数据集，并对四种类型的神经算子变体进行了系统比较，为神经算子在工程设计中的应用提供了基准和指导。

### 背景

神经算子已成为学习函数空间之间非线性映射的强大工具，能够实时预测科学和工程应用中的复杂动态。随着在工程设计评估中的广泛应用，已提出多种神经算子架构，但由于缺乏公平和全面的比较，模型选择仍然具有挑战性。

### 目的

提出并标准化六个代表性的3D工业规模工程设计数据集，涵盖热分析、线性弹性、弹塑性、时变塑性问题和计算流体动力学，使数据集可直接用于各种神经算子架构的训练。

### 方法

使用这些数据集，对四种类型的神经算子变体进行系统比较，包括受DeepONet启发的基于分支-主干神经算子、受图神经网络启发的基于图神经算子、受傅里叶神经算子启发的基于网格神经算子、以及受PointNet启发的基于点神经算子。研究引入实际增强以使这些模型适应不同的工程设置，并评估每个模型在预测性能、计算效率、内存使用和部署复杂性方面的优势和局限性。

### 主要发现

研究提供了神经算子在多种工程问题上的性能比较，分析了不同神经算子架构的优缺点，为模型选择提供了依据。

### 结论

研究结果为未来神经算子开发提供了可操作的见解，有助于推动神经算子在工程设计领域的应用和发展。

### 翻译

神经算子已成为学习函数空间之间非线性映射的强大工具，能够实时预测科学和工程应用中的复杂动态。随着在工程设计评估中的广泛采用，已为各种问题环境提出了多种神经算子架构。然而，由于缺乏公平和全面的比较，模型选择仍然具有挑战性。为解决这一问题，我们提出并标准化了六个代表性的3D工业规模工程设计数据集，涵盖热分析、线性弹性、弹塑性、时变塑性问题和计算流体动力学。所有数据集包含完全预处理的输入和输出，用于模型训练，使其可直接用于各种神经算子架构。使用这些数据集，我们对四种类型的神经算子变体进行了系统比较，包括受DeepONet启发的基于分支-主干神经算子、受图神经网络启发的基于图神经算子、受傅里叶神经算子启发的基于网格神经算子、以及受PointNet启发的基于点神经算子。我们进一步引入了实际增强以使这些模型适应不同的工程设置，提高了比较的公平性。我们的基准研究评估了每个模型在预测性能、计算效率、内存使用和部署复杂性方面的优势和局限性。研究结果为未来神经算子开发提供了可操作的见解。


### 论文摘要

Neural operators have emerged as powerful tools for learning nonlinear mappings between function spaces, enabling real-time prediction of complex dynamics in diverse scientific and engineering applications. With their growing adoption in engineering design evaluation, a wide range of neural operator architectures have been proposed for various problem settings. However, model selection remains challenging due to the absence of fair and comprehensive comparisons. To address this, we propose and standardize six representative 3D industry-scale engineering design datasets spanning thermal analysis, linear elasticity, elasto-plasticity, time-dependent plastic problems, and computational fluid dynamics. All datasets include fully preprocessed inputs and outputs for model training, making them directly usable across diverse neural operator architectures. Using these datasets, we conduct a systematic comparison of four types of neural operator variants, including Branch-Trunk-based Neural Operators inspired by DeepONet, Graph-based Neural Operators inspired by Graph Neural Networks, Grid-based Neural Operators inspired by Fourier Neural Operators, and Point-based Neural Operators inspired by PointNet. We further introduce practical enhancements to adapt these models to different engineering settings, improving the fairness of the comparison. Our benchmarking study evaluates each model strengths and limitations in terms of predictive performance, computational efficiency, memory usage, and deployment complexity. The findings provide actionable insights to guide future neural operator development.

---

## 10. RareAgent: Self-Evolving Reasoning for Drug Repurposing in Rare Diseases

**论文链接:** [http://arxiv.org/abs/2510.05764v1](http://arxiv.org/abs/2510.05764v1)

**作者:** Lang Qin, Zijian Gan, Xu Cao, Pengcheng Jiang, Yankai Jiang, Jiawei Han, Kaishun Wu, Jintai Chen

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种名为RareAgent的自进化多智能体系统，用于罕见疾病的药物重定位研究。该系统通过主动证据寻求推理而非被动模式识别，显著提高了药物与疾病关联的预测性能，并提供了透明的推理过程。

### 背景

计算药物重定位对于罕见疾病特别具有挑战性，尤其是当药物和目标疾病之间没有先前的关联存在时。在这种情况下，知识图谱补全和消息传递图神经网络难以学习和传播可靠信号，导致性能不佳。

### 目的

开发一种能够将罕见疾病药物重定位任务从被动模式识别转变为主动证据寻求推理的系统，以提高预测性能并提供可解释的推理过程。

### 方法

RareAgent组织特定任务的对立辩论，其中智能体从不同角度动态构建证据图，以支持、反驳或蕴含假设。系统通过自进化循环分析推理策略，产生文本反馈以完善智能体策略，并将成功的推理路径提炼为可转移的启发式方法，以加速未来研究。

### 主要发现

全面评估显示，RareAgent比推理基线提高指示AUPRC 18.1%，且提供的推理链与临床证据一致，具有良好的可解释性。

### 结论

RareAgent通过主动证据寻求推理有效改善了罕见疾病的药物重定位性能，不仅提高了预测准确率，还提供了透明的推理过程，有助于加速罕见疾病的治疗发现。

### 翻译

对于罕见疾病，当药物与目标疾病之间没有先前的关联时，计算药物重定位尤其具有挑战性。因此，知识图谱补全和消息传递图神经网络几乎没有可靠的信号可以学习和传播，导致性能不佳。我们提出了RareAgent，这是一个自进化的多智能体系统，它将这一任务从被动的模式识别重新定义为主动的证据寻求推理。RareAgent组织特定任务的对立辩论，其中智能体从不同角度动态构建证据图，以支持、反驳或蕴含假设。推理策略在自进化循环中被事后分析，产生文本反馈以完善智能体策略，同时成功的推理路径被提炼为可转移的启发式方法，以加速未来的研究。全面的评估显示，RareAgent比推理基线提高了18.1%的指示AUPRC，并提供与临床证据一致的透明推理链。


### 论文摘要

Computational drug repurposing for rare diseases is especially challenging when no prior associations exist between drugs and target diseases. Therefore, knowledge graph completion and message-passing GNNs have little reliable signal to learn and propagate, resulting in poor performance. We present RareAgent, a self-evolving multi-agent system that reframes this task from passive pattern recognition to active evidence-seeking reasoning. RareAgent organizes task-specific adversarial debates in which agents dynamically construct evidence graphs from diverse perspectives to support, refute, or entail hypotheses. The reasoning strategies are analyzed post hoc in a self-evolutionary loop, producing textual feedback that refines agent policies, while successful reasoning paths are distilled into transferable heuristics to accelerate future investigations. Comprehensive evaluations reveal that RareAgent improves the indication AUPRC by 18.1% over reasoning baselines and provides a transparent reasoning chain consistent with clinical evidence.

---

## 11. 论文ID: 2510.05751v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.05751v1.json'

---

## 12. Are Heterogeneous Graph Neural Networks Truly Effective? A Causal Perspective

**论文链接:** [http://arxiv.org/abs/2510.05750v1](http://arxiv.org/abs/2510.05750v1)

**作者:** Xiao Yang, Xuejiao Zhao, Zhiqi Shen

**发布时间:** 2025-10-07

### GPT解析

### 总结

这篇论文研究了异构图神经网络(HGNNs)的有效性，从模型架构和异构信息两个角度进行了系统分析。通过在21个数据集和20个基线上的系统复现，并结合全面的超参数调优，作者开发了一个因果效应估计框架来评估性能提升的来源。研究得出两个主要结论：模型架构和复杂性对性能没有因果影响，而异构信息通过增加同质性和局部-全局分布差异对性能产生正面因果效应，使节点类别更易区分。

### 背景

图神经网络(GNNs)在节点分类方面取得了显著成功。在此基础上，异构图神经网络(HGNNs)整合了关系类型和节点与边的语义，以利用异构信息。HGNNs的因果分析正在快速发展，旨在将真实的因果效应与虚假相关性分开。然而，HGNNs是否本质上有效尚未得到充分研究，大多数研究只是隐含地假设而非建立这种有效性。

### 目的

本研究旨在从两个角度(模型架构和异构信息)检查HGNNs的有效性，并确定性能提升的真正来源。

### 方法

作者在21个数据集和20个基线上进行了系统复现，并结合了全面的超参数调优。为了进一步分离性能提升的来源，他们开发了一个因果效应估计框架，通过事实分析和反事实分析在标准假设下构建和评估候选因素，并通过最小充分调整集、跨方法一致性检查和敏感性分析验证了结果的稳健性。

### 主要发现

研究发现模型架构和复杂性对性能没有因果影响；异构信息通过增加同质性和局部-全局分布差异对性能产生正面因果效应，使节点类别更易区分。

### 结论

HGNNs的性能提升主要来自于异构信息，而非模型架构或复杂性。异构信息通过增加同质性和局部-全局分布差异，使节点类别更易区分，从而提高了性能。

### 翻译

图神经网络(GNNs)在节点分类方面取得了显著成功。在此基础上，异构图神经网络(HGNNs)整合了关系类型和节点与边的语义，以利用异构信息。HGNNs的因果分析正在快速发展，旨在将真实的因果效应与虚假相关性分开。然而，HGNNs是否本质上有效尚未得到充分研究，大多数研究只是隐含地假设而非建立这种有效性。在本工作中，我们从两个角度检查HGNNs：模型架构和异构信息。我们在21个数据集和20个基线上进行了系统复现，并结合了全面的超参数调优。为了进一步分离性能提升的来源，我们开发了一个因果效应估计框架，通过事实分析和反事实分析在标准假设下构建和评估候选因素，并通过最小充分调整集、跨方法一致性检查和敏感性分析验证了结果的稳健性。我们的研究得出了两个结论。首先，模型架构和复杂性对性能没有因果影响。其次，异构信息通过增加同质性和局部-全局分布差异对性能产生正面因果效应，使节点类别更易区分。实现可在https://github.com/YXNTU/CausalHGNN公开获取。


### 论文摘要

Graph neural networks (GNNs) have achieved remarkable success in node classification. Building on this progress, heterogeneous graph neural networks (HGNNs) integrate relation types and node and edge semantics to leverage heterogeneous information. Causal analysis for HGNNs is advancing rapidly, aiming to separate genuine causal effects from spurious correlations. However, whether HGNNs are intrinsically effective remains underexamined, and most studies implicitly assume rather than establish this effectiveness. In this work, we examine HGNNs from two perspectives: model architecture and heterogeneous information. We conduct a systematic reproduction across 21 datasets and 20 baselines, complemented by comprehensive hyperparameter retuning. To further disentangle the source of performance gains, we develop a causal effect estimation framework that constructs and evaluates candidate factors under standard assumptions through factual and counterfactual analyses, with robustness validated via minimal sufficient adjustment sets, cross-method consistency checks, and sensitivity analyses. Our results lead to two conclusions. First, model architecture and complexity have no causal effect on performance. Second, heterogeneous information exerts a positive causal effect by increasing homophily and local-global distribution discrepancy, which makes node classes more distinguishable. The implementation is publicly available at https://github.com/YXNTU/CausalHGNN.

---

## 13. Convolution and Graph-based Deep Learning Approaches for Gamma/Hadron Separation in Imaging Atmospheric Cherenkov Telescopes

**论文链接:** [http://arxiv.org/abs/2510.05736v1](http://arxiv.org/abs/2510.05736v1)

**作者:** Abhay Mehta, Dan Parsons, Tim Lukas Holch, David Berge, Matthias Weidlich

**发布时间:** 2025-10-07

**DOI:** 10.22323/1.501.0752

**备注:** PoS(ICRC2025)752

### GPT解析

### 总结

本文提出并评估了三种基于深度学习的模型，用于在成像大气切伦科夫望远镜(IACTs)数据中从强子背景中识别γ射线，展示了结合卷积神经网络和图神经网络的方法在提高识别性能方面的潜力。

### 背景

在成像大气切伦科夫望远镜(IACTs)的地面探测中，从强子背景中识别γ射线是一个关键方面。当前方法在利用复杂数据相关性方面能力有限，难以有效处理复杂的观测数据。

### 目的

设计具有与任务相关归纳偏见的模型架构，以解决基于深度学习的模型在稳健性和适用性方面面临的挑战，提高从强子背景中识别γ射线的性能。

### 方法

提出、训练并评估了三种基于深度学习的模型：(1)结合图像和图数据的混合卷积和图神经网络模型(CNN-GNN)；(2)在图构建中纳入额外重建信息的增强版CNN-GNN变体；(3)使用图像矩作为基线的图神经网络(GNN)模型。所有模型均在模拟数据上进行训练和评估。

### 主要发现

新的结合卷积和基于图的方法显示出比传统方法更好的性能。在图构建中纳入重建信息进一步提高了模型在真实观测数据上的泛化能力。

### 结论

结合卷积神经网络和图神经网络的方法为γ射线识别提供了有效途径，而包含重建信息则有助于提高模型在实际应用中的泛化性能，为未来研究提供了有价值的方向。

### 翻译

使用成像大气切伦科夫望远镜(IACTs)进行地面探测时，从主要强子背景中识别γ射线是一个关键方面。虽然当前方法在利用复杂数据相关性方面能力有限，但基于深度学习的模型通过直接利用图像级信息提供了一种有前景的替代方案。然而，这类模型在稳健性和适用性方面仍面临几个挑战。设计具有与任务相关归纳偏见的模型架构可以帮助缓解这一问题。论文提出了三种基于深度学习的模型，在模拟数据上进行了训练和评估：(1)使用图像和图数据的混合卷积和图神经网络模型(CNN-GNN)；(2)在图构建中纳入额外重建信息的增强版CNN-GNN变体；(3)使用图像矩作为基线的图神经网络(GNN)模型。新的结合卷积和基于图的方法显示出比传统方法更好的性能，而包含重建信息则在真实观测数据的泛化能力方面提供了进一步的可能性。


### 论文摘要

The identification of $\gamma$-rays from the predominant hadronic-background is a key aspect in their ground-based detection using Imaging Atmospheric Cherenkov Telescopes (IACTs). While current methods are limited in their ability to exploit correlations in complex data, deep learning-based models offer a promising alternative by directly leveraging image-level information. However, several challenges involving the robustness and applicability of such models remain. Designing model architectures with inductive biases relevant for the task can help mitigate the problem. Three such deep learning-based models are proposed, trained, and evaluated on simulated data: (1) a hybrid convolutional and graph neural network model (CNN-GNN) using both image and graph data; (2) an enhanced CNN-GNN variant that incorporates additional reconstructed information within the graph construction; and (3) a graph neural network (GNN) model using image moments serving as a baseline. The new combined convolution and graph-based approach demonstrates improved performance over traditional methods, and the inclusion of reconstructed information offers further potential in generalization capabilities on real observational data.

---

## 14. QGraphLIME - Explaining Quantum Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.05683v1](http://arxiv.org/abs/2510.05683v1)

**作者:** Haribandhu Jena, Jyotirmaya Shivottam, Subhankar Mishra

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种名为QGraphLIME的模型无关、事后解释框架，用于解释量子图神经网络。该框架通过在保持图结构的扰动上拟合局部代理，并聚合代理属性及其离散度，提供不确定性感知的节点和边重要性排名。实证研究表明该方法能提供准确稳定的解释，且非线性代理建模具有明显优势。

### 背景

量子图神经网络在图结构数据学习中提供了强大范式，但由于测量诱导的随机性和图结构的组合性质，它们的可解释性变得复杂。

### 目的

开发一种模型无关的、事后的解释框架，为量子图模型提供不确定性感知的节点和边重要性排名。

### 方法

QGraphLIME将模型解释视为在保持图结构的扰动上拟合的局部代理的分布。通过聚合代理属性及其离散度，该框架为量子图模型提供不确定性感知的节点和边重要性排名。该框架还提供了关于代理集大小的无分布、有限样本保证：Dvoretzky-Kiefer-Wolfowitz边界确保在标准独立假设下，以目标精度和置信度均匀逼近二元类别概率的诱导分布。

### 主要发现

在对具有已知真实情况的受控合成图进行的实证研究中，展示了准确和稳定的解释，消融研究表明非线性代理建模的明显优势，并突出了对扰动设计的敏感性。

### 结论

这些结果共同建立了一种原则性的、不确定性感知的、对结构敏感的量子图神经网络解释方法，并为扩展到更广泛的架构和真实世界数据奠定了基础，随着量子资源的成熟。

### 翻译

量子图神经网络为图结构数据学习提供了强大的范式，然而它们的可解释性因测量诱导的随机性和图结构的组合性质而变得复杂。在本文中，我们引入了QuantumGraphLIME（QGraphLIME），一种模型无关的、事后的框架，该框架将模型解释视为在保持图结构的扰动上拟合的局部代理的分布。通过聚合代理属性及其离散度，QGraphLIME为量子图模型产生不确定性感知的节点和边重要性排名。该框架进一步提供了代理集大小的无分布、有限样本保证：Dvoretzky-Kiefer-Wolfowitz边界确保在标准独立假设下，以目标精度和置信度均匀逼近二元类别概率的诱导分布。在具有已知真实情况的受控合成图上的实证研究展示了准确和稳定的解释，消融研究表明非线性代理建模的明显优势，并突出了对扰动设计的敏感性。总的来说，这些结果建立了一种原则性的、不确定性感知的、对结构敏感的量子图神经网络解释方法，并为随着量子资源的成熟扩展到更广泛的架构和真实世界数据奠定了基础。代码可在https://github.com/smlab-niser/qglime获取。


### 论文摘要

Quantum graph neural networks offer a powerful paradigm for learning on graph-structured data, yet their explainability is complicated by measurement-induced stochasticity and the combinatorial nature of graph structure. In this paper, we introduce QuantumGraphLIME (QGraphLIME), a model-agnostic, post-hoc framework that treats model explanations as distributions over local surrogates fit on structure-preserving perturbations of a graph. By aggregating surrogate attributions together with their dispersion, QGraphLIME yields uncertainty-aware node and edge importance rankings for quantum graph models. The framework further provides a distribution-free, finite-sample guarantee on the size of the surrogate ensemble: a Dvoretzky-Kiefer-Wolfowitz bound ensures uniform approximation of the induced distribution of a binary class probability at target accuracy and confidence under standard independence assumptions. Empirical studies on controlled synthetic graphs with known ground truth demonstrate accurate and stable explanations, with ablations showing clear benefits of nonlinear surrogate modeling and highlighting sensitivity to perturbation design. Collectively, these results establish a principled, uncertainty-aware, and structure-sensitive approach to explaining quantum graph neural networks, and lay the groundwork for scaling to broader architectures and real-world datasets, as quantum resources mature. Code is available at https://github.com/smlab-niser/qglime.

---

## 15. Inductive inference of gradient-boosted decision trees on graphs for insurance fraud detection

**论文链接:** [http://arxiv.org/abs/2510.05676v1](http://arxiv.org/abs/2510.05676v1)

**作者:** Félix Vandervorst, Bruno Deprez, Wouter Verbeke, Tim Verdonck

**发布时间:** 2025-10-07

### GPT解析

### 总结

该论文提出了一种新颖的归纳图梯度提升机器（G-GBM），用于解决异质性和动态图上的监督学习问题，特别是在保险欺诈检测领域。

### 背景

基于图的方法在机器学习中越来越受欢迎，能够建模复杂的数据和关系。保险欺诈是一个典型用例，因为虚假索赔通常来自犯罪团伙策划事故或同一人在多个保单上提交错误索赔。然而，基于图的方法面临高类别不平衡和异质动态网络的挑战，导致表格数据上的梯度提升树方法仍占主导地位。

### 目的

提出一种新颖的归纳图梯度提升机器（G-GBM），用于异质性和动态图上的监督学习，特别是在保险欺诈检测场景中。

### 方法

开发了一种图梯度提升机器（G-GBM），在模拟随机图实验中与流行图神经网络方法竞争，并在开源和专有真实世界数据集上展示其在保险欺诈检测中的能力。应用成熟可解释性方法来理解模型预测。

### 主要发现

G-GBM能够与流行的图神经网络方法相竞争，在保险欺诈检测任务中显示出强大能力，同时通过可解释性方法提供了对模型决策的更好理解。

### 结论

G-GBM为处理异质性和动态图上的监督学习提供了一种有效方法，特别是在保险欺诈检测领域，结合了图表示学习与梯度提升的优势。

### 翻译

基于图的方法在机器学习中越来越受欢迎，因为它们能够建模复杂的数据和关系。保险欺诈是一个典型的用例，因为虚假索赔通常是犯罪团伙策划事故或同一人在多个保单上提交错误索赔的结果。一个挑战是，由于欺诈数据中存在高度类别不平衡，基于图的方法难以找到数据的有意义表示。另一个是，考虑到人员、公司和保单之间关系的变化，保险网络是异质和动态的。这就是为什么表格数据上的梯度提升树方法在该领域仍然占主导地位。因此，我们提出了一种新颖的归纳图梯度提升机器（G-GBM），用于异质和动态图上的监督学习。我们在使用各种模拟随机图的实验中表明，我们的估计器与流行的图神经网络方法相竞争。我们使用开源和真实世界的专有数据集展示了G-GBM在保险欺诈检测方面的能力。鉴于骨干模型是梯度提升森林，我们应用了成熟的可解释性方法，以更好地理解G-GBM所做的预测。


### 论文摘要

Graph-based methods are becoming increasingly popular in machine learning due to their ability to model complex data and relations. Insurance fraud is a prime use case, since false claims are often the result of organised criminals that stage accidents or the same persons filing erroneous claims on multiple policies. One challenge is that graph-based approaches struggle to find meaningful representations of the data because of the high class imbalance present in fraud data. Another is that insurance networks are heterogeneous and dynamic, given the changing relations among people, companies and policies. That is why gradient boosted tree approaches on tabular data still dominate the field. Therefore, we present a novel inductive graph gradient boosting machine (G-GBM) for supervised learning on heterogeneous and dynamic graphs. We show that our estimator competes with popular graph neural network approaches in an experiment using a variety of simulated random graphs. We demonstrate the power of G-GBM for insurance fraud detection using an open-source and a real-world, proprietary dataset. Given that the backbone model is a gradient boosting forest, we apply established explainability methods to gain better insights into the predictions made by G-GBM.

---

## 16. When Does Global Attention Help? A Unified Empirical Study on Atomistic Graph Learning

**论文链接:** [http://arxiv.org/abs/2510.05583v1](http://arxiv.org/abs/2510.05583v1)

**作者:** Arindam Chowdhury, Massimiliano Lupo Pasini

**发布时间:** 2025-10-07

**备注:** 40 pages, 8 figures, 18 tables

### GPT解析

### 总结

这篇论文介绍了一个统一的、可重现的基准测试框架HydraGNN，用于系统性地评估图神经网络中全局注意力机制的实际效益，并比较了不同架构在原子尺度化合物建模中的表现。

### 背景

图神经网络（GNNs）被广泛用于替代昂贵实验和第一性原理模拟，研究原子尺度化合物的行为。随着GNN架构复杂度的增加，大多数最新GNN结合了传统的消息传递神经网络（MPNNs）层和具有全局注意力机制的图变换器（GTs），但尚不清楚全局注意力机制相对于精心调优的MPNN层何时真正有益。

### 目的

引入第一个统一的、可重现的基准测试框架，用于系统性地隔离和评估消息传递、全局注意力和基于编码器的特征增强在原子图学习中的贡献，并量化全局注意力机制的准确性-计算权衡。

### 方法

构建了基于HydraGNN的统一基准测试框架，支持在四种受控模型类别之间无缝切换：MPNN、带有化学/拓扑编码器的MPNN、MPNN与全局注意力的GPS风格混合模型，以及带有编码器的完全融合的局部-全局模型。使用七个开源数据集进行回归和分类任务的基准测试，系统性地隔离不同组件的贡献。

### 主要发现

基于编码器增强的MPNNs形成了一个强大的基线，而融合的局部-全局模型在受长程相互作用效应影响的属性上表现出最明显的好处。研究还量化了注意力的准确性-计算权衡，报告了其在内存中的开销。

### 结论

这些结果建立了原子图学习中全局注意力的首次受控评估，并为未来模型开发提供了可重现的测试平台，有助于理解不同架构在原子尺度化合物建模中的优势和适用场景。

### 翻译

图神经网络（GNNs）被广泛用作昂贵实验和第一性原理模拟的替代品，以研究原子尺度化合物的行为，其架构复杂度不断增加，以实现对复杂物理的建模。虽然最近的GNN大多将传统的消息传递神经网络（MPNNs）层与具有全局注意力机制的更先进的图变换器（GTs）结合，以分别建模短程和长程相互作用，但由于实现、特征或超参数调优的不一致，全局注意力机制相对于精心调优的MPNN层何时真正提供益处仍不清楚。我们引入了第一个统一的、可重现的基准测试框架 - 基于HydraGNN构建 - 它能够在四种受控模型类别之间无缝切换：MPNN、带有化学/拓扑编码器的MPNN、MPNN与全局注意力的GPS风格混合模型，以及带有编码器的完全融合的局部-全局模型。使用七个开源数据集进行回归和分类任务的基准测试，我们系统性地隔离了消息传递、全局注意力和基于编码器的特征增强的贡献。我们的研究表明，基于编码器增强的MPNNs形成了一个强大的基线，而融合的局部-全局模型在受长程相互作用效应影响的属性上产生最明显的好处。我们进一步量化了注意力的准确性-计算权衡，报告了其在内存中的开销。总之，这些结果建立了原子图学习中全局注意力的首次受控评估，并为未来模型开发提供了可重现的测试平台。


### 论文摘要

Graph neural networks (GNNs) are widely used as surrogates for costly experiments and first-principles simulations to study the behavior of compounds at atomistic scale, and their architectural complexity is constantly increasing to enable the modeling of complex physics. While most recent GNNs combine more traditional message passing neural networks (MPNNs) layers to model short-range interactions with more advanced graph transformers (GTs) with global attention mechanisms to model long-range interactions, it is still unclear when global attention mechanisms provide real benefits over well-tuned MPNN layers due to inconsistent implementations, features, or hyperparameter tuning. We introduce the first unified, reproducible benchmarking framework - built on HydraGNN - that enables seamless switching among four controlled model classes: MPNN, MPNN with chemistry/topology encoders, GPS-style hybrids of MPNN with global attention, and fully fused local - global models with encoders. Using seven diverse open-source datasets for benchmarking across regression and classification tasks, we systematically isolate the contributions of message passing, global attention, and encoder-based feature augmentation. Our study shows that encoder-augmented MPNNs form a robust baseline, while fused local-global models yield the clearest benefits for properties governed by long-range interaction effects. We further quantify the accuracy - compute trade-offs of attention, reporting its overhead in memory. Together, these results establish the first controlled evaluation of global attention in atomistic graph learning and provide a reproducible testbed for future model development.

---

## 17. Generative Dynamic Graph Representation Learning for Conspiracy Spoofing Detection

**论文链接:** [http://arxiv.org/abs/2510.05562v1](http://arxiv.org/abs/2510.05562v1)

**作者:** Sheng Xiang, Yidong Jiang, Yunting Chen, Dawei Cheng, Guoping Zhao, Changjun Jiang

**发布时间:** 2025-10-07

**DOI:** 10.1145/3696410.3714518

**备注:** 10 pages, 5 figures, ACM the web conference 2025

### GPT解析

### 总结

该论文提出了一种名为生成动态图模型(GDGM)的新框架，用于金融交易中的欺骗检测，特别是针对复杂的共谋欺骗行为。该方法通过动态建模交易行为和节点间关系，有效捕捉了时间模式和不断变化的市场条件，并在实验和实际应用中表现出色。

### 背景

金融交易中的欺骗检测至关重要，特别是识别复杂的共谋欺骗行为。传统机器学习方法主要关注孤立节点特征，忽略了互联节点的更广泛背景。基于图的技术如GNNs虽有进展，但在面对现实世界中动态、不规则的交易行为模式时仍面临挑战，难以捕捉动态和多样化的节点间关系复杂性。

### 目的

提出一种名为生成动态图模型(GDGM)的新框架，用于建模动态交易行为和节点间关系，学习用于共谋欺骗检测的表示，以解决现有方法难以捕捉动态关系的问题。

### 方法

将原始交易数据转换为时间戳序列，使用神经常微分方程和门控循环单元建模交易行为，生成包含欺骗模式时间动态的表示，并采用伪标签生成和异构聚合技术收集相关信息以增强检测性能。

### 主要发现

在欺骗检测数据集上的实验表明，该方法在检测准确性上优于最先进的模型。该欺骗检测系统已成功部署在世界上最大的交易市场之一，验证了其实际应用价值。

### 结论

生成动态图模型(GDGM)能够有效捕捉动态交易行为和不断变化的节点间关系，在共谋欺骗检测任务中表现出色，具有良好的实际应用前景。

### 翻译

金融交易中的欺骗检测至关重要，特别是在识别复杂的共谋欺骗行为方面。传统的机器学习方法主要关注孤立节点的特征，往往忽略了互联节点的更广泛背景。基于图的技术，特别是图神经网络(GNNs)，通过有效利用关系信息推动了该领域的发展。然而，在现实世界的欺骗检测数据集中，交易行为表现出动态、不规则的模式。现有的欺骗检测方法虽然在某些场景中有效，但难以捕捉动态和多样化、不断发展的节点间关系的复杂性。为解决这些挑战，我们提出了一种名为生成动态图模型(GDGM)的新框架，该框架通过建模动态交易行为和节点间关系来学习用于共谋欺骗检测的表示。具体而言，我们的方法集成了生成动态潜在空间来捕捉时间模式和不断变化的市场条件。原始交易数据首先被转换为带时间戳的序列。然后我们使用神经常微分方程和门控循环单元对交易行为进行建模，以生成包含欺骗模式时间动态的表示。此外，还采用了伪标签生成和异构聚合技术来收集相关信息并增强对共谋欺骗行为的检测性能。在欺骗检测数据集上进行的实验表明，我们的方法在检测准确性上优于最先进的模型。此外，我们的欺骗检测系统已成功部署在世界上最大的交易市场之一，进一步验证了所提出方法的实际应用性和性能。


### 论文摘要

Spoofing detection in financial trading is crucial, especially for identifying complex behaviors such as conspiracy spoofing. Traditional machine-learning approaches primarily focus on isolated node features, often overlooking the broader context of interconnected nodes. Graph-based techniques, particularly Graph Neural Networks (GNNs), have advanced the field by leveraging relational information effectively. However, in real-world spoofing detection datasets, trading behaviors exhibit dynamic, irregular patterns. Existing spoofing detection methods, though effective in some scenarios, struggle to capture the complexity of dynamic and diverse, evolving inter-node relationships. To address these challenges, we propose a novel framework called the Generative Dynamic Graph Model (GDGM), which models dynamic trading behaviors and the relationships among nodes to learn representations for conspiracy spoofing detection. Specifically, our approach incorporates the generative dynamic latent space to capture the temporal patterns and evolving market conditions. Raw trading data is first converted into time-stamped sequences. Then we model trading behaviors using the neural ordinary differential equations and gated recurrent units, to generate the representation incorporating temporal dynamics of spoofing patterns. Furthermore, pseudo-label generation and heterogeneous aggregation techniques are employed to gather relevant information and enhance the detection performance for conspiratorial spoofing behaviors. Experiments conducted on spoofing detection datasets demonstrate that our approach outperforms state-of-the-art models in detection accuracy. Additionally, our spoofing detection system has been successfully deployed in one of the largest global trading markets, further validating the practical applicability and performance of the proposed method.

---

## 18. Fundamental Limits of Crystalline Equivariant Graph Neural Networks: A Circuit Complexity Perspective

**论文链接:** [http://arxiv.org/abs/2510.05494v1](http://arxiv.org/abs/2510.05494v1)

**作者:** Yang Cao, Zhao Song, Jiahao Zhang, Jiale Zhao

**发布时间:** 2025-10-07

### GPT解析

### 总结

本研究通过电路复杂度的透镜，表征了等变图神经网络在晶体结构预测中的内在计算和表达极限。研究表明，在特定条件下，这些模型可以被多项式大小的均匀阈值电路族模拟，为这类架构在现实资源约束下可解决的问题提供了具体上限。

### 背景

图神经网络已成为关系数据学习的核心范式。在材料科学中，等变图神经网络因其能够尊重欧几里得对称性和周期性边界条件，已成为晶体结构预测的有力骨干。尽管有强大的实证性能，它们在周期性、对称性约束环境下的表达能力仍然不为人知。

### 目的

通过电路复杂度的透镜，表征等变图神经网络在晶体结构预测中的内在计算和表达极限，理解其在周期性、对称性约束环境下的表达能力。

### 方法

分析等变图神经网络层在节点特征、原子坐标和晶格矩阵上进行的计算。在多项式精度下，证明对于特定数量的节点、层深度和宽度，这些模型可以被多项式大小的均匀阈值电路族模拟，并提供明确的常数深度界限。

### 主要发现

1) 对于特定数量的节点、层深度和宽度，嵌入宽度有特定要求；2) 这些等变图神经网络模型可以被多项式大小的均匀阈值电路族模拟；3) 将这类模型置于特定复杂度类别中为架构在现实资源约束下可解决的问题提供了具体上限；4) 明确了需要增加深度、更丰富的几何基元或更宽的层等架构修改来超越这一范围。

### 结论

该分析补充了Weisfeiler-Lehman风格的结果，这些结果不能直接转移到周期性晶体。研究为晶体系统上的对称感知图学习提供了复杂性理论基础，有助于理解这类神经网络的极限和改进方向。

### 翻译

图神经网络已成为关系数据学习的核心范式。在材料科学中，等变图神经网络因其能够尊重欧几里得对称性和周期性边界条件，已成为晶体结构预测的有力骨干。尽管有强大的实证性能，它们在周期性、对称性约束环境下的表达能力仍然不为人知。本研究通过电路复杂度的透镜，表征了等变图神经网络在晶体结构预测中的内在计算和表达极限。我们分析了等变图神经网络层在节点特征、原子坐标和晶格矩阵上进行的计算，并证明在多项式精度下，对于特定数量的节点、层深度和宽度，嵌入宽度有特定要求的模型可以被多项式大小的均匀阈值电路族模拟(具有明确的常数深度界限)。将等变图神经网络置于特定复杂度类别中为这类架构在现实资源约束下可解决的决策和预测问题提供了具体上限，并明确了需要哪些架构修改(例如增加深度、更丰富的几何基元或更宽的层)来超越这一范围。该分析补充了Weisfeiler-Lehman风格的结果，这些结果不能直接转移到周期性晶体，并为晶体系统上的对称感知图学习提供了复杂性理论基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是确定用于晶体结构预测的等变图神经网络（EGNNs）的基本计算和表达能力的限制。这个问题在研究中很重要，因为尽管EGNNs在材料科学领域表现出强大性能，能够尊重欧几里得对称性和周期边界条件，但它们在周期性、对称约束条件下的理论表达能力仍不清楚。了解这些限制有助于确定哪些架构修改是必要的以超越这些限制，并为对称感知的晶体系统图学习提供理论基础。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从电路复杂性角度分析EGNNs的表达能力，将模型置于具体的电路类别中，以确定它们在现实约束下能解决的问题类型。他们首先形式化了EGNNs的结构，然后分析了EGNN层在节点特征、原子坐标和晶格矩阵上的计算，并量化了使用均匀阈值电路模拟这些计算所需的资源。作者借鉴了电路复杂性理论（特别是TC0电路类别）、现有的浮点数计算和矩阵乘法的电路实现结果，但区别于传统的Weisfeiler-Lehman分析（这些分析专注于离散图同构，抽象掉了连续坐标和对称约束）和其他架构（如Transformer）的电路分析。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过电路复杂性理论确定EGNNs在晶体结构预测中的基本计算限制，证明在多项式精度、特定架构条件下，EGNN可以被TC0电路模拟，从而为其能解决的问题类型提供具体上限。实现流程包括：1)定义晶体结构的单元胞表示和分数坐标矩阵；2)形式化EGNN架构，包括傅里叶变换、成对消息传递和层操作；3)分析基本EGNN构建块的电路复杂性，证明傅里叶变换、MLP和消息传递等可在TC0电路中计算；4)在给定假设条件下证明整个EGNN模型可被均匀TC0电路族实现。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次对晶体EGNNs进行电路复杂性分析；形式化EGNNs结构为理论分析提供基础；确定TC0上限为EGNNs能解决的问题提供具体上限；提供超越限制的架构修改指导。相比之前工作的不同：区别于传统的Weisfeiler-Lehman分析（专注于离散图同构，抽象连续坐标和对称约束）；区别于其他架构（如Transformer）的电路分析；区别于现有几何深度学习研究（较少探索晶体结构中的基本局限性）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过电路复杂性理论确定了晶体结构预测中EGNNs的基本计算限制，证明其在特定条件下可被TC0电路模拟，为这类模型能解决的问题类型提供了理论上限并指出了超越这些限制的架构修改方向。'}


### 论文摘要

Graph neural networks (GNNs) have become a core paradigm for learning on relational data. In materials science, equivariant GNNs (EGNNs) have emerged as a compelling backbone for crystalline-structure prediction, owing to their ability to respect Euclidean symmetries and periodic boundary conditions. Despite strong empirical performance, their expressive power in periodic, symmetry-constrained settings remains poorly understood. This work characterizes the intrinsic computational and expressive limits of EGNNs for crystalline-structure prediction through a circuit-complexity lens. We analyze the computations carried out by EGNN layers acting on node features, atomic coordinates, and lattice matrices, and prove that, under polynomial precision, embedding width $d=O(n)$ for $n$ nodes, $O(1)$ layers, and $O(1)$-depth, $O(n)$-width MLP instantiations of the message/update/readout maps, these models admit a simulation by a uniform $\mathsf{TC}^0$ threshold-circuit family of polynomial size (with an explicit constant-depth bound). Situating EGNNs within $\mathsf{TC}^0$ provides a concrete ceiling on the decision and prediction problems solvable by such architectures under realistic resource constraints and clarifies which architectural modifications (e.g., increased depth, richer geometric primitives, or wider layers) are required to transcend this regime. The analysis complements Weisfeiler-Lehman style results that do not directly transfer to periodic crystals, and offers a complexity-theoretic foundation for symmetry-aware graph learning on crystalline systems.

---

## 19. AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative Multi-Agent Question Answering

**论文链接:** [http://arxiv.org/abs/2510.05445v1](http://arxiv.org/abs/2510.05445v1)

**作者:** Zheyuan Zhang, Kaiwen Shi, Zhengqing Yuan, Zehong Wang, Tianyi Ma, Keerthiram Murugesan, Vincent Galassi, Chuxu Zhang, Yanfang Ye

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出了一种名为tAgentRouter的框架，通过知识图谱引导的多代理路由机制来优化问答任务性能，解决了在多种模型和代理策略中选择最佳配置的问题。

### 背景

大型语言模型和基于代理的框架发展迅速，但实践者在选择下游任务最佳配置时面临不确定性。研究表明不同代理和骨干模型具有互补优势，且更大模型并不总是更优。现有代理路由方法通常只强调成本效率，忽视了QA任务中固有的细粒度上下文和关系结构。

### 目的

提出一种自适应路由机制，能够根据任务需求选择最适合的模型和代理配置，有效利用不同代理的互补优势来提高问答任务性能。

### 方法

将多代理QA表述为知识图谱引导的路由问题，由经验性能信号监督。将QA实例转换为联合编码查询、上下文实体和代理的知识图谱，训练异构图神经网络在节点类型间传播信息并生成任务感知的路由分布。利用软监督和代理输出的加权聚合学习协作方案。

### 主要发现

广泛的实验表明，该框架一致地优于单代理和集成基线，同时能够跨基准测试和LLM骨干模型进行有效泛化。

### 结论

图监督的多代理路由在问答任务中具有显著的有效性和鲁棒性，能够有效捕捉不同代理的互补优势。

### 翻译

大型语言模型和基于代理的框架发展迅速，支持多种应用。然而，随着模型和代理策略的激增，实践者在选择下游任务的最佳配置时面临重大不确定性。先前研究表明不同代理和骨干模型具有互补优势，且更大的模型并不总是更优，这凸显了对自适应路由机制的需求。然而，现有的代理路由方法通常只强调成本效率，而忽视了问答任务中固有的细粒度上下文和关系结构。在本文中，我们提出了tAgentRouter，一个将多代理QA表述为由经验性能信号监督的知识图谱引导的路由问题的框架。具体来说，我们将QA实例转换为联合编码查询、上下文实体和代理的知识图谱，然后训练异构图神经网络在节点类型间传播信息，并生成对代理的任务感知路由分布。通过利用软监督和代理输出的加权聚合，AgentRouter学习能够捕捉不同代理互补优势的有原则的协作方案。广泛的实验证明，我们的框架一致地优于单代理和集成基线，同时能够跨基准测试和LLM骨干模型进行泛化。这些结果突显了图监督的多代理路由在问答任务中的有效性和鲁棒性。


### 论文摘要

Large language models (LLMs) and agent-based frameworks have advanced rapidly, enabling diverse applications. Yet, with the proliferation of models and agentic strategies, practitioners face substantial uncertainty in selecting the best configuration for a downstream task. Prior studies show that different agents and backbones exhibit complementary strengths, and that larger models are not always superior, underscoring the need for adaptive routing mechanisms. Existing approaches to agent routing, however, often emphasize cost efficiency while overlooking the fine-grained contextual and relational structure inherent in QA tasks. In this paper, we propose tAgentRouter, a framework that formulates multi-agent QA as a knowledge-graph-guided routing problem supervised by empirical performance signals. Specifically, we convert QA instance into a knowledge graph that jointly encodes queries, contextual entities, and agents, and then train a heterogeneous graph neural network (GNN) to propagate information across node types and produce task-aware routing distributions over agents. By leveraging soft supervision and weighted aggregation of agent outputs, AgentRouter learns principled collaboration schemes that capture the complementary strengths of diverse agents. Extensive experiments demonstrate that our framework consistently outperforms single-agent and ensemble baselines, while generalizing across benchmarks and LLM backbones. These results highlight the effectiveness and robustness of graph-supervised multi-agent routing for question answering.

---

## 20. BioAutoML-NAS: An End-to-End AutoML Framework for Multimodal Insect Classification via Neural Architecture Search on Large-Scale Biodiversity Data

**论文链接:** [http://arxiv.org/abs/2510.05888v1](http://arxiv.org/abs/2510.05888v1)

**作者:** Arefin Ittesafun Abian, Debopom Sutradhar, Md Rafi Ur Rashid, Reem E. Mohamed, Md Rafiqul Islam, Asif Karim, Kheng Cher Yeo, Sami Azam

**发布时间:** 2025-10-07

### GPT解析

### 总结

本研究提出了BioAutoML-NAS模型，一种基于多模态数据和神经架构搜索的昆虫分类方法，解决了昆虫特征复杂、类别不平衡和大数据集等挑战，在多个数据集上实现了高精度分类。

### 背景

昆虫分类对农业管理和生态研究至关重要，直接影响作物健康和生产。然而，该任务面临昆虫特征复杂、类别不平衡及大规模数据集等挑战，难以有效进行。

### 目的

开发一种能够有效处理昆虫分类挑战的方法，通过结合多模态数据和自动架构搜索提高分类准确性和效率。

### 方法

提出BioAutoML-NAS模型，使用多模态数据（图像和元数据），应用神经架构搜索自动学习最佳网络结构，采用多模态融合模块结合视觉和生物信息，使用交替双层优化策略更新网络参数，并通过零操作移除不重要连接以产生高效架构。

### 主要发现

在BIOSCAN-5M数据集上达到96.81%准确率、97.46%精确率、96.81%召回率和97.05% F1分数，优于现有方法约8-16%；在Insects-1M数据集上获得93.25%准确率、93.71%精确率、92.74%召回率和93.22% F1分数。

### 结论

BioAutoML-NAS提供了准确可靠的昆虫分类方法，支持现代可持续农业实践。

### 翻译

昆虫分类对农业管理和生态研究很重要，因为它直接影响作物健康和生产。然而，由于昆虫的复杂特征、类别不平衡和大规模数据集，这项任务仍然具有挑战性。为了解决这些问题，我们提出了BioAutoML-NAS，这是第一个使用多模态数据（包括图像和元数据）的BioAutoML模型，它应用神经架构搜索（NAS）来为图像自动学习每个单元内每个连接的最佳操作。多个单元堆叠形成完整网络，每个单元提取详细的图像特征表示。多模态融合模块将图像嵌入与元数据结合，使模型能够利用视觉和分类生物信息来分类昆虫。交替双层优化训练策略联合更新网络权重和架构参数，同时零操作移除不太重要的连接，产生稀疏、高效且高性能的架构。在BIOSCAN-5M数据集上的广泛评估表明，BioAutoML-NAS实现了96.81%的准确率、97.46%的精确率、96.81%的召回率和97.05%的F1分数，比最先进的迁移学习、transformer、AutoML和NAS方法分别高出约16%、10%和8%。在Insects-1M数据集上的进一步验证获得了93.25%的准确率、93.71%的精确率、92.74%的召回率和93.22%的F1分数。这些结果表明，BioAutoML-NAS提供了准确、可靠的昆虫分类，支持现代可持续农业。


### 论文摘要

Insect classification is important for agricultural management and ecological research, as it directly affects crop health and production. However, this task remains challenging due to the complex characteristics of insects, class imbalance, and large-scale datasets. To address these issues, we propose BioAutoML-NAS, the first BioAutoML model using multimodal data, including images, and metadata, which applies neural architecture search (NAS) for images to automatically learn the best operations for each connection within each cell. Multiple cells are stacked to form the full network, each extracting detailed image feature representations. A multimodal fusion module combines image embeddings with metadata, allowing the model to use both visual and categorical biological information to classify insects. An alternating bi-level optimization training strategy jointly updates network weights and architecture parameters, while zero operations remove less important connections, producing sparse, efficient, and high-performing architectures. Extensive evaluation on the BIOSCAN-5M dataset demonstrates that BioAutoML-NAS achieves 96.81% accuracy, 97.46% precision, 96.81% recall, and a 97.05% F1 score, outperforming state-of-the-art transfer learning, transformer, AutoML, and NAS methods by approximately 16%, 10%, and 8% respectively. Further validation on the Insects-1M dataset obtains 93.25% accuracy, 93.71% precision, 92.74% recall, and a 93.22% F1 score. These results demonstrate that BioAutoML-NAS provides accurate, confident insect classification that supports modern sustainable farming.

---

## 21. Empirical Comparison of Membership Inference Attacks in Deep Transfer Learning

**论文链接:** [http://arxiv.org/abs/2510.05753v1](http://arxiv.org/abs/2510.05753v1)

**作者:** Yuxuan Bai, Gauri Pradhan, Marlon Tobaben, Antti Honkela

**发布时间:** 2025-10-07

**备注:** 30 pages, 13 figures, published in TMLR  https://openreview.net/forum?id=UligTUCgdt

### GPT解析

### 总结

研究表明，在迁移学习环境下评估隐私风险时，没有一种通用的成员推理攻击方法，实践者应根据具体应用场景和数据特点选择合适的攻击方法

### 背景

强大的大规模基础模型的出现正在改变训练范式，训练方式正从从头开始训练转向迁移学习，这种转变使得在敏感应用中可以使用小型、领域特定的数据集进行高效训练

### 目的

比较不同类型的成员推理攻击(MIAs)在迁移学习环境中的性能，帮助从业者识别用于隐私风险评估的最有效攻击方法

### 方法

对比多种成员推理攻击在迁移学习设置中的表现，特别关注针对通过迁移学习微调的模型的攻击评估

### 主要发现

基于分数的成员推理攻击的有效性随训练数据的增加而降低；没有一种单一的成员推理攻击能够捕捉迁移学习训练模型中的所有隐私风险；似然比攻击(LiRA)在大多数实验场景中表现优异；反向Hessian攻击(IHA)在针对PatchCamelyon数据集微调的高数据量模型上更有效

### 结论

需要综合考虑多种成员推理攻击方法来全面评估迁移学习模型的隐私风险，不同攻击方法在不同场景下可能表现不同，应根据具体情况选择

### 翻译

随着强大大规模基础模型的出现，训练范式正日益从从头开始训练转向迁移学习。这使得在敏感应用中，可以使用小型、领域特定的数据集进行高效训练。成员推理攻击(MIAs)为机器学习模型提供了隐私泄露的经验性估计。然而，先前针对通过迁移学习微调模型的成员推理攻击评估仅依赖于可能攻击的一小部分。我们通过比较迁移学习环境中不同成员推理攻击的性能来解决这一问题，以帮助从业者识别用于隐私风险评估的最有效攻击。我们发现，对于基于分数的成员推理攻击，攻击有效性随训练数据的增加而降低。我们发现，没有一种成员推理攻击能够捕捉迁移学习训练模型中的所有隐私风险。虽然似然比攻击(LiRA)在大多数实验场景中表现出优越性能，但反向Hessian攻击(IHA)被证明在针对PatchCamelyon数据集微调的高数据量模型上更有效


### 论文摘要

With the emergence of powerful large-scale foundation models, the training paradigm is increasingly shifting from from-scratch training to transfer learning. This enables high utility training with small, domain-specific datasets typical in sensitive applications.Membership inference attacks (MIAs) provide an empirical estimate of the privacy leakage by machine learning models. Yet, prior assessments of MIAs against models fine-tuned with transfer learning rely on a small subset of possible attacks. We address this by comparing performance of diverse MIAs in transfer learning settings to help practitioners identify the most efficient attacks for privacy risk evaluation. We find that attack efficacy decreases with the increase in training data for score-based MIAs. We find that there is no one MIA which captures all privacy risks in models trained with transfer learning. While the Likelihood Ratio Attack (LiRA) demonstrates superior performance across most experimental scenarios, the Inverse Hessian Attack (IHA) proves to be more effective against models fine-tuned on PatchCamelyon dataset in high data regime.

---

## 22. Transfer Learning on Edge Connecting Probability Estimation under Graphon Model

**论文链接:** [http://arxiv.org/abs/2510.05527v1](http://arxiv.org/abs/2510.05527v1)

**作者:** Yuyao Wang, Yu-Hung Cheng, Debarghya Mukherjee, Huimin Cheng

**发布时间:** 2025-10-07

### GPT解析

### 总结

该研究提出了一种名为GTRANS的迁移学习框架，用于在小规模目标图中提高图模型估计的准确性。该方法结合了邻域平滑和Gromov-Wasserstein最优传输，对齐和转移图之间的结构模式，并通过自适应去偏机制防止负迁移。

### 背景

Graphon模型为估计网络中的潜在连接概率提供了一个灵活的非参数框架，支持链接预测和数据增强等下游应用。然而，准确的图模型估计通常需要大型图，而实践中往往只能观察到小型网络。

### 目的

研究旨在通过迁移学习框架解决小型网络中图模型估计不准确的问题，利用大型相关源图的结构信息来改进小型目标图的估计。

### 方法

提出了GTRANS方法，这是一个集成邻域平滑和Gromov-Wasserstein最优传输的迁移学习框架，用于对齐和转移图之间的结构模式。GTRANS包含一个自适应去偏机制，通过残差平滑识别并校正目标特定的偏差。

### 主要发现

研究提供了对估计对齐矩阵稳定性的理论保证，并通过大量合成和真实数据实验证明了GTRANS在提高目标图估计准确性方面的有效性。这些改进直接转化为下游应用（如图分类任务和链接预测任务）的性能提升。

### 结论

GTRANS方法通过有效迁移结构信息并防止负迁移，显著提高了小规模网络中图模型估计的准确性，进而提升了下游应用性能。

### 翻译

图模型(Graphon models)为估计网络中的潜在连接概率提供了一个灵活的非参数框架，支持链接预测和数据增强等一系列下游应用。然而，准确的图模型估计通常需要大型图，而实践中往往只能观察到小型网络。解决这一问题的一种方法是采用迁移学习框架，旨在利用来自更大相关源图的结构信息来改进小型目标图中的估计。在本文中，我们提出了一种新方法，即GTRANS，这是一个迁移学习框架，集成了邻域平滑和Gromov-Wasserstein最优传输，以对齐和转移图之间的结构模式。为防止负迁移，GTRANS包含一个自适应去偏机制，通过残差平滑识别并校正目标特定的偏差。我们提供了对估计对齐矩阵稳定性的理论保证，并通过大量合成和真实数据实验证明了GTRANS在提高目标图估计准确性方面的有效性。这些改进直接转化为下游应用的性能提升，如图分类任务和链接预测任务。


### 论文摘要

Graphon models provide a flexible nonparametric framework for estimating latent connectivity probabilities in networks, enabling a range of downstream applications such as link prediction and data augmentation. However, accurate graphon estimation typically requires a large graph, whereas in practice, one often only observes a small-sized network. One approach to addressing this issue is to adopt a transfer learning framework, which aims to improve estimation in a small target graph by leveraging structural information from a larger, related source graph. In this paper, we propose a novel method, namely GTRANS, a transfer learning framework that integrates neighborhood smoothing and Gromov-Wasserstein optimal transport to align and transfer structural patterns between graphs. To prevent negative transfer, GTRANS includes an adaptive debiasing mechanism that identifies and corrects for target-specific deviations via residual smoothing. We provide theoretical guarantees on the stability of the estimated alignment matrix and demonstrate the effectiveness of GTRANS in improving the accuracy of target graph estimation through extensive synthetic and real data experiments. These improvements translate directly to enhanced performance in downstream applications, such as the graph classification task and the link prediction task.

---

## 23. Fusion-Based Neural Generalization for Predicting Temperature Fields in Industrial PET Preform Heating

**论文链接:** [http://arxiv.org/abs/2510.05394v1](http://arxiv.org/abs/2510.05394v1)

**作者:** Ahmad Alsheikh, Andreas Fischer

**发布时间:** 2025-10-06

**备注:** Workshop paper, AIP2025: Second Workshop on AI in Production (2025).  Licensed under CC BY 4.0

### GPT解析

### 总结

该研究提出了一种新型深度学习框架，用于PET预成型件在工业微波系统中预热过程的温度预测。通过迁移学习和模型融合技术，该方法能够在不同材料和设计条件下实现高效准确的温度预测，减少了重新训练的需求，并提高了泛化能力。

### 背景

PET预成型件在工业微波系统中的预热过程对吹塑成型至关重要。准确且高效的温度预测是优化这一过程的关键，传统方法需要针对每种材料或设计变更进行大量重新训练。

### 目的

提出一种用于通用温度预测的新型深度学习框架，解决传统模型需要大量重新训练的问题，实现跨不同材料和设计条件的高效温度预测。

### 方法

引入数据高效的神经网络架构，利用迁移学习和模型融合技术；预训练专门神经回归器处理不同条件（如回收PET热容量或变化的预成型件几何形状）；将表示集成到统一全局模型中；架构包含跳跃连接以增强稳定性和预测准确性；减少对大型模拟数据集的需求。

### 主要发现

相比从头训练的模型，该方法实现了更优的性能；在材料变性和几何多样性两个案例研究上的实验验证表明泛化能力有显著提高；建立了一种可扩展的基于机器学习的制造环境智能热控制解决方案。

### 结论

数据高效的泛化策略可以扩展到其他涉及有限数据复杂物理建模的工业应用，为制造环境中的智能热控制提供了有效解决方案。

### 翻译

准确且高效的温度预测对于优化PET预成型件在工业微波系统中吹塑成型前的预热过程至关重要。我们提出了一种用于通用温度预测的新型深度学习框架。与需要针对每种材料或设计变更进行大量重新训练的传统模型不同，我们的方法引入了一种数据高效的神经网络架构，利用迁移学习和模型融合来推广到未见过的场景。通过在不同条件下预训练专门的神经回归器，并将它们的表示集成到统一的全局模型中，我们创建了一个能够学习异构输入间共享热动力学的系统。该架构包含跳跃连接以增强稳定性和预测准确性。我们的方法减少了对大型模拟数据集的需求，同时实现了比从头训练的模型更优的性能。在两个案例研究上的实验验证表明泛化能力有显著提高，为制造环境中的智能热控制建立了可扩展的基于机器学习的解决方案。


### 论文摘要

Accurate and efficient temperature prediction is critical for optimizing the preheating process of PET preforms in industrial microwave systems prior to blow molding. We propose a novel deep learning framework for generalized temperature prediction. Unlike traditional models that require extensive retraining for each material or design variation, our method introduces a data-efficient neural architecture that leverages transfer learning and model fusion to generalize across unseen scenarios. By pretraining specialized neural regressor on distinct conditions such as recycled PET heat capacities or varying preform geometries and integrating their representations into a unified global model, we create a system capable of learning shared thermal dynamics across heterogeneous inputs. The architecture incorporates skip connections to enhance stability and prediction accuracy. Our approach reduces the need for large simulation datasets while achieving superior performance compared to models trained from scratch. Experimental validation on two case studies material variability and geometric diversity demonstrates significant improvements in generalization, establishing a scalable ML-based solution for intelligent thermal control in manufacturing environments. Moreover, the approach highlights how data-efficient generalization strategies can extend to other industrial applications involving complex physical modeling with limited data.

---

## 24. Fine-Tuned CNN-Based Approach for Multi-Class Mango Leaf Disease Detection

**论文链接:** [http://arxiv.org/abs/2510.05326v1](http://arxiv.org/abs/2510.05326v1)

**作者:** Jalal Ahmmed, Faruk Ahmed, Rashedul Hasan Shohan, Md. Mahabub Rana, Mahdi Hasan

**发布时间:** 2025-10-06

**备注:** Double column 6 pages, 10 figures, ieee conference style

### GPT解析

### 总结

该研究评估了五种预训练卷积神经网络模型在芒果叶疾病识别中的性能，发现DenseNet201表现最佳，为智能农业中的疾病检测提供了可靠解决方案。

### 背景

芒果是南亚重要的水果作物，但其种植经常受到叶部疾病的阻碍，这些疾病严重影响产量和质量。

### 目的

研究五种预训练卷积神经网络模型在芒果叶疾病多类识别中的性能表现。

### 方法

采用迁移学习策略和微调技术，测试DenseNet201、InceptionV3、ResNet152V2、SeResNet152和Xception五种模型，针对八类芒果叶疾病进行多类识别，并通过准确率、精确率、召回率、F1分数和混淆矩阵等标准评估指标进行评估。

### 主要发现

DenseNet201表现最佳，准确率达99.33%，在各类别指标上表现一致，特别是在识别切象鼻虫和细菌性溃疡方面表现出色；ResNet152V2和SeResNet152提供了良好结果；InceptionV3和Xception在视觉相似的类别（如煤污病和白粉病）中表现较差；高性能模型的训练和验证图显示了稳定收敛。

### 结论

微调的迁移学习模型能够为智能农业应用中的芒果叶疾病检测提供精确可靠的多类检测能力。

### 翻译

芒果是南亚重要的水果作物，但其种植经常受到叶部疾病的阻碍，这些疾病严重影响产量和质量。本研究考察了五种预训练卷积神经网络模型（DenseNet201、InceptionV3、ResNet152V2、SeResNet152和Xception）在迁移学习策略和微调技术下，针对八类芒果叶疾病进行多类识别的性能。这些模型通过准确率、精确率、召回率、F1分数和混淆矩阵等标准评估指标进行评估。在测试的架构中，DenseNet201取得了最佳结果，准确率达到99.33%，各类别指标表现一致，特别是在识别切象鼻虫和细菌性溃疡方面表现出色。此外，ResNet152V2和SeResNet152提供了良好的结果，而InceptionV3和Xception在视觉相似的类别（如煤污病和白粉病）中表现较差。训练和验证图显示了高性能模型的稳定收敛。微调的迁移学习模型具备在智能农业应用中进行精确可靠的芒果叶疾病多类检测的能力。


### 论文摘要

Mango is an important fruit crop in South Asia, but its cultivation is frequently hampered by leaf diseases that greatly impact yield and quality. This research examines the performance of five pre-trained convolutional neural networks, DenseNet201, InceptionV3, ResNet152V2, SeResNet152, and Xception, for multi-class identification of mango leaf diseases across eight classes using a transfer learning strategy with fine-tuning. The models were assessed through standard evaluation metrics, such as accuracy, precision, recall, F1-score, and confusion matrices. Among the architectures tested, DenseNet201 delivered the best results, achieving 99.33% accuracy with consistently strong metrics for individual classes, particularly excelling in identifying Cutting Weevil and Bacterial Canker. Moreover, ResNet152V2 and SeResNet152 provided strong outcomes, whereas InceptionV3 and Xception exhibited lower performance in visually similar categories like Sooty Mould and Powdery Mildew. The training and validation plots demonstrated stable convergence for the highest-performing models. The capability of fine-tuned transfer learning models, for precise and dependable multi-class mango leaf disease detection in intelligent agricultural applications.

---

## 25. A novel hallucination classification framework

**论文链接:** [http://arxiv.org/abs/2510.05189v1](http://arxiv.org/abs/2510.05189v1)

**作者:** Maksym Zavhorodnii, Dmytro Dehtiarov, Anna Konovalenko

**发布时间:** 2025-10-06

**备注:** 15 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种自动检测大型语言模型推理过程中产生幻觉的新方法，通过系统化分类学和提示工程实现幻觉类型的可控重现，并利用无监督学习技术分析幻觉数据。

### 背景

大型语言模型在推理过程中会产生幻觉现象，这些幻觉可能包含错误或误导性信息，影响模型输出的可靠性。

### 目的

开发一种能够自动检测和区分LLM生成幻觉与准确响应的有效方法，提高模型输出的可靠性。

### 方法

基于系统化分类学，通过提示工程控制性重现不同类型幻觉；使用嵌入模型将幻觉数据集映射到向量空间；在降维表示中应用无监督学习技术分析幻觉与真实响应；评估质心间距离的定量分析。

### 主要发现

幻觉中信息失真的严重程度与它们从正确输出簇的空间偏离之间存在一致的相关性；简单的分类算法可以在单个LLM中可靠地区分幻觉和准确响应。

### 结论

所提出的方法为提高模型可靠性提供了一种轻量级但有效的框架，即使使用简单算法也能可靠识别幻觉。

### 翻译

这项工作引入了一种用于自动检测大型语言模型推理过程中产生的幻觉的新方法。所提出的方法基于系统化的分类学，并通过提示工程控制性地重现不同类型的幻觉。随后，使用嵌入模型将专门的幻觉数据集映射到向量空间，并在降维表示中，使用无监督学习技术分析幻觉与真实响应。对质心间距离的定量评估显示，幻觉中信息失真的严重程度与它们从正确输出簇的空间偏离之间存在一致的相关性。这些发现提供了理论和实证证据，表明即使在单个LLM中，简单的分类算法也可以可靠地区分幻觉和准确响应，从而为提高模型可靠性提供了一种轻量级而有效的框架。


### 论文摘要

This work introduces a novel methodology for the automatic detection of hallucinations generated during large language model (LLM) inference. The proposed approach is based on a systematic taxonomy and controlled reproduction of diverse hallucination types through prompt engineering. A dedicated hallucination dataset is subsequently mapped into a vector space using an embedding model and analyzed with unsupervised learning techniques in a reduced-dimensional representation of hallucinations with veridical responses. Quantitative evaluation of inter-centroid distances reveals a consistent correlation between the severity of informational distortion in hallucinations and their spatial divergence from the cluster of correct outputs. These findings provide theoretical and empirical evidence that even simple classification algorithms can reliably distinguish hallucinations from accurate responses within a single LLM, thereby offering a lightweight yet effective framework for improving model reliability.

---

## 26. Adversarial Reinforcement Learning for Offensive and Defensive Agents in a Simulated Zero-Sum Network Environment

**论文链接:** [http://arxiv.org/abs/2510.05157v1](http://arxiv.org/abs/2510.05157v1)

**作者:** Abrar Shahid, Ibteeker Mahir Ishum, AKM Tahmidul Haque, M Sohel Rahman, A. B. M. Alim Al Islam

**发布时间:** 2025-10-03

**备注:** 8 pages, 5 tables, 5 figures. 12th International Conference on Next  Generation Computing, Communication, Systems and Security

### GPT解析

### 总结

本研究通过自定义OpenAI Gym环境，对网络安全的对抗性强化学习进行了受控研究，模拟了多端口服务上的暴力攻击和反应式防御策略。

### 背景

网络环境中存在真实的安全权衡，包括背景流量噪声、渐进式利用机制、基于IP的规避策略、蜜罐陷阱和多级速率限制防御等复杂因素。

### 目的

研究对抗性强化学习在网络安全中的应用，特别是在攻击者和防御者之间的零和奖励框架下，评估不同配置下防御者的有效性。

### 方法

创建自定义OpenAI Gym环境模拟网络攻击和防御场景；使用深度Q网络(DQN)训练攻击者和防御者代理；在零和奖励框架下进行训练，成功利用获得大额终端奖励，增量操作产生小成本；系统性评估多种配置（变化陷阱检测概率、利用难度阈值和训练计划）。

### 主要发现

防御者的可观测性和蜜罐的有效性对成功攻击构成重大障碍；奖励塑造和仔细的训练调度对于在这种对抗环境中学习稳定性至关重要；防御者在50,000+训练集中始终保持战略优势；当面对复杂的防御策略（包括自适应IP阻止和特定于端口的控制）时，性能增益会放大。

### 结论

提供了完整的实现细节、可复现的超参数配置和架构指南，以支持未来在网络安全对抗性RL方面的研究；零和公式和真实的操作约束使该环境适合研究自主防御系统、攻击者-防御者共同进化和向真实世界网络安全场景的迁移学习。

### 翻译

本文通过自定义OpenAI Gym环境对网络安全的对抗性强化学习进行了受控研究，该环境模拟了多端口服务上的暴力攻击和反应式防御。该环境捕捉了真实的安全权衡，包括背景流量噪声、渐进式利用机制、基于IP的规避策略、蜜罐陷阱和多级速率限制防御。使用深度Q网络(DQN)在零和奖励框架下训练竞争性的攻击者和防御者代理，成功的利用产生大的终端奖励，而增量操作产生小成本。通过系统评估多种配置（变化的陷阱检测概率、利用难度阈值和训练计划），结果表明防御者的可观测性和蜜罐的有效性对成功攻击构成重大障碍。实验揭示，奖励塑造和仔细的训练调度对于在这种对抗环境中的学习稳定性至关重要。防御者在50,000+训练集中始终保持战略优势，当面对复杂的防御策略（包括自适应IP阻止和特定于端口的控制）时，性能增益会放大。提供了完整的实现细节、可复现的超参数配置和架构指南，以支持未来在网络安全对抗性RL方面的研究。零和公式和真实的操作约束使该环境适合研究自主防御系统、攻击者-防御者共同进化和向真实世界网络安全场景的迁移学习。


### 论文摘要

This paper presents a controlled study of adversarial reinforcement learning in network security through a custom OpenAI Gym environment that models brute-force attacks and reactive defenses on multi-port services. The environment captures realistic security trade-offs including background traffic noise, progressive exploitation mechanics, IP-based evasion tactics, honeypot traps, and multi-level rate-limiting defenses. Competing attacker and defender agents are trained using Deep Q-Networks (DQN) within a zero-sum reward framework, where successful exploits yield large terminal rewards while incremental actions incur small costs. Through systematic evaluation across multiple configurations (varying trap detection probabilities, exploitation difficulty thresholds, and training regimens), the results demonstrate that defender observability and trap effectiveness create substantial barriers to successful attacks. The experiments reveal that reward shaping and careful training scheduling are critical for learning stability in this adversarial setting. The defender consistently maintains strategic advantage across 50,000+ training episodes, with performance gains amplifying when exposed to complex defensive strategies including adaptive IP blocking and port-specific controls. Complete implementation details, reproducible hyperparameter configurations, and architectural guidelines are provided to support future research in adversarial RL for cybersecurity. The zero-sum formulation and realistic operational constraints make this environment suitable for studying autonomous defense systems, attacker-defender co-evolution, and transfer learning to real-world network security scenarios.

---

## 27. PhishSSL: Self-Supervised Contrastive Learning for Phishing Website Detection

**论文链接:** [http://arxiv.org/abs/2510.05900v1](http://arxiv.org/abs/2510.05900v1)

**作者:** Wenhao Li, Selvakumar Manickam, Yung-Wey Chong, Shankar Karuppayah, Priyadarsi Nanda, Binyong Li

**发布时间:** 2025-10-07

**备注:** Accepted by the 26th International Conference on Web Information  Systems Engineering (WISE 2025)

### GPT解析

### 总结

PhishSSL是一种自监督对比学习框架，用于检测网络钓鱼网站，它消除了对标记钓鱼数据的依赖，通过混合表格增强和自适应特征注意力技术，在各种数据集上都优于无监督和自监督基线方法，显示出强大的泛化能力和可转移性，是应对动态网络环境中不断演变威胁的有效解决方案。

### 背景

网络钓鱼网站通过模仿合法网站来窃取敏感用户信息，持续构成网络安全威胁。现有的基于机器学习的检测方法通常依赖标记数据的监督学习，这不仅带来大量的标注成本，还限制了适应新型攻击模式的能力。

### 目的

解决现有监督学习方法面临的标注成本高和对新型攻击模式适应性差的问题，开发一种无需标记钓鱼数据训练的检测方法。

### 方法

提出PhishSSL，一种自监督对比学习框架，结合混合表格增强与自适应特征注意力，生成语义一致的视图并强调判别性属性，在三种具有不同特征组成的网络钓鱼数据集上进行了评估。

### 主要发现

在所有数据集上，PhishSSL持续优于无监督和自监督基线方法；消融研究确认了每个组件的贡献；尽管特征集多样化，PhishSSL仍保持强大的性能，突显了其强大的泛化能力和可转移性。

### 结论

PhishSSL为网络钓鱼网站检测提供了有前景的解决方案，特别有效于应对动态网络环境中不断演变的威胁。

### 翻译

网络钓鱼网站通过模仿合法网站来窃取敏感用户信息，持续构成网络安全威胁。现有的基于机器学习的检测方法通常依赖标记数据的监督学习，这不仅带来大量的标注成本，还限制了适应新型攻击模式的能力。为应对这些挑战，我们提出了PhishSSL，一种自监督对比学习框架，消除了训练期间对标记钓鱼数据的需求。PhishSSL结合混合表格增强与自适应特征注意力，生成语义一致的视图并强调判别性属性。我们在三种具有不同特征组成的网络钓鱼数据集上评估了PhishSSL。在所有数据集上，PhishSSL持续优于无监督和自监督基线方法，同时消融研究确认了每个组件的贡献。此外，尽管特征集多样化，PhishSSL仍保持强大的性能，突显了其强大的泛化能力和可转移性。这些结果表明，PhishSSL为网络钓鱼网站检测提供了有前景的解决方案，特别有效于应对动态网络环境中不断演变的威胁。


### 论文摘要

Phishing websites remain a persistent cybersecurity threat by mimicking legitimate sites to steal sensitive user information. Existing machine learning-based detection methods often rely on supervised learning with labeled data, which not only incurs substantial annotation costs but also limits adaptability to novel attack patterns. To address these challenges, we propose PhishSSL, a self-supervised contrastive learning framework that eliminates the need for labeled phishing data during training. PhishSSL combines hybrid tabular augmentation with adaptive feature attention to produce semantically consistent views and emphasize discriminative attributes. We evaluate PhishSSL on three phishing datasets with distinct feature compositions. Across all datasets, PhishSSL consistently outperforms unsupervised and self-supervised baselines, while ablation studies confirm the contribution of each component. Moreover, PhishSSL maintains robust performance despite the diversity of feature sets, highlighting its strong generalization and transferability. These results demonstrate that PhishSSL offers a promising solution for phishing website detection, particularly effective against evolving threats in dynamic Web environments.

---

## 28. Towards Robust and Realible Multimodal Fake News Detection with Incomplete Modality

**论文链接:** [http://arxiv.org/abs/2510.05839v1](http://arxiv.org/abs/2510.05839v1)

**作者:** Hengyang Zhou, Yiwei Wei, Jian Yang, Zhenyu Zhang

**发布时间:** 2025-10-07

### GPT解析

### 总结

论文提出了MMLNet（多专家模态不完整学习网络），一种简单有效的多模态融合策略，用于解决多模态假新闻检测中模态不完整的问题。

### 背景

随着社交媒体平台上大量多模态虚假内容的出现，多模态假新闻检测成为紧迫任务。现有研究主要关注复杂特征提取和融合，但忽略了实际应用中多媒体新闻在传播过程中可能丢失信息导致的模态不完整问题。

### 目的

提出一种通用且鲁棒的多模态融合策略，确保在模态信息缺失情况下仍能准确检测虚假新闻，有效遏制恶意虚假信息传播。

### 方法

MMLNet包含三个关键步骤：(1)多专家协作推理：通过多个专家动态利用互补信息补偿缺失模态；(2)不完整模态适配器：利用新特征分布补偿缺失信息；(3)模态缺失学习：通过标签感知的自适应加权策略和对比学习学习鲁棒表示。

### 主要发现

在两种语言的三个真实世界基准测试中，MMLNet与最先进方法相比表现优越，同时保持相对简单的结构，能有效处理模态不完整的假新闻检测场景。

### 结论

MMLNet通过解决多模态假新闻检测中的模态不完整问题，提高了检测的准确性和鲁棒性，有效遏制了恶意虚假信息的传播。代码已在GitHub公开。

### 翻译

多模态假新闻检测（MFND）已成为一项紧迫任务，随着社交媒体平台上大量多模态虚假内容的出现。先前的研究主要集中在复杂特征提取和融合上，从多模态内容中学习判别性信息。然而，在实际应用中，多媒体新闻在传播过程中可能会自然丢失一些信息，导致模态不完整，这对现有模型的泛化能力和鲁棒性不利。为此，我们提出了一种新颖的通用且鲁棒的多模态融合策略，称为多专家模态不完整学习网络（MMLNet），它简单而有效。它包含三个关键步骤：(1) 多专家协作推理，通过多个专家动态利用互补信息来补偿缺失的模态。(2) 不完整模态适配器通过利用新的特征分布来补偿缺失的信息。(3) 模态缺失学习利用标签感知的自适应加权策略，通过对比学习学习鲁棒表示。我们在两种语言的三个真实世界基准上评估了MMLNet，与最先进的方法相比表现出优越的性能，同时保持了相对简单的结构。通过确保在信息传播导致的不完整模态场景下假新闻检测的准确性，MMLNet有效地遏制了恶意虚假信息的传播。代码已在GitHub公开。


### 论文摘要

Multimodal fake news detection (MFND) has become an urgent task with the emergence of huge multimodal fake content on social media platforms. Previous studies mainly focus on complex feature extraction and fusion to learn discriminative information from multimodal content. However, in real-world applications, multimedia news may naturally lose some information during dissemination, resulting in modality incompleteness, which is detrimental to the generalization and robustness of existing models. To this end, we propose a novel generic and robust multimodal fusion strategy, termed Multi-expert Modality-incomplete Learning Network (MMLNet), which is simple yet effective. It consists of three key steps: (1) Multi-Expert Collaborative Reasoning to compensate for missing modalities by dynamically leveraging complementary information through multiple experts. (2) Incomplete Modality Adapters compensates for the missing information by leveraging the new feature distribution. (3) Modality Missing Learning leveraging an label-aware adaptive weighting strategy to learn a robust representation with contrastive learning. We evaluate MMLNet on three real-world benchmarks across two languages, demonstrating superior performance compared to state-of-the-art methods while maintaining relative simplicity. By ensuring the accuracy of fake news detection in incomplete modality scenarios caused by information propagation, MMLNet effectively curbs the spread of malicious misinformation. Code is publicly available at https://github.com/zhyhome/MMLNet.

---

## 29. Diversity Is All You Need for Contrastive Learning: Spectral Bounds on Gradient Magnitudes

**论文链接:** [http://arxiv.org/abs/2510.05767v1](http://arxiv.org/abs/2510.05767v1)

**作者:** Peter Ochieng

**发布时间:** 2025-10-07

### GPT解析

### 总结

本研究通过推导非渐近谱带来界定平方InfoNCE梯度范数，并设计了谱感知批量选择方法，有效提高了训练效率。

### 背景

研究关注InfoNCE梯度的谱特性以及批量选择对训练效率的影响。

### 目的

推导非渐近谱带用于界定平方InfoNCE梯度范数，并设计基于谱特性的高效批量选择方法。

### 方法

通过对齐、温度和批量谱推导非渐近谱带；使用有效秩作为各向异性代理设计谱感知批量选择；采用批量内白化促进各向同性。

### 主要发现

恢复了1/τ²定律；在合成数据和ImageNet上密切跟踪批量均值梯度；Greedy-64在ImageNet-100上将训练时间缩短15%(相比随机)和24%(相比Pool-P3)；CIFAR-10显示类似改进；批量内白化减少50步梯度方差1.37倍，与理论上界匹配。

### 结论

谱感知批量选择方法能有效提高训练效率；批量内白化能有效减少梯度方差。

### 翻译

我们推导了通过对比度、温度和批量谱来界定平方InfoNCE梯度范数的非渐近谱带，恢复了1/τ²定律，并在合成数据和ImageNet上密切跟踪批量均值梯度。使用有效秩作为各向异性代理，我们设计了谱感知批量选择，包括一个快速贪婪构建器。在ImageNet-100上，Greedy-64在达到相同准确率的情况下，相比随机方法将时间缩短15%(相比Pool-P3缩短24%)；CIFAR-10显示出类似的改进。批量内白化促进各向同性并将50步梯度方差减少1.37倍，与我们的理论上限一致。


### 论文摘要

We derive non-asymptotic spectral bands that bound the squared InfoNCE gradient norm via alignment, temperature, and batch spectrum, recovering the \(1/\tau^{2}\) law and closely tracking batch-mean gradients on synthetic data and ImageNet. Using effective rank \(R_{\mathrm{eff}}\) as an anisotropy proxy, we design spectrum-aware batch selection, including a fast greedy builder. On ImageNet-100, Greedy-64 cuts time-to-67.5\% top-1 by 15\% vs.\ random (24\% vs.\ Pool--P3) at equal accuracy; CIFAR-10 shows similar gains. In-batch whitening promotes isotropy and reduces 50-step gradient variance by \(1.37\times\), matching our theoretical upper bound.

---

## 30. Oracle-Guided Masked Contrastive Reinforcement Learning for Visuomotor Policies

**论文链接:** [http://arxiv.org/abs/2510.05692v1](http://arxiv.org/abs/2510.05692v1)

**作者:** Yuhang Zhang, Jiaping Xiao, Chao Yan, Mir Feroskhan

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种名为Oracle-Guided Masked Contrastive Reinforcement Learning (OMC-RL)的新框架，用于提高视觉运动政策学习的样本效率和渐进性能。

### 背景

学习视觉运动政策的普遍方法是使用强化学习将高维视觉观察直接映射到动作命令，但高维视觉输入与敏捷机动输出的组合导致样本效率低下和显著的模拟到现实差距问题。

### 目的

解决视觉运动政策学习中的样本效率和渐进性能问题，同时改善泛化能力。

### 方法

OMC-RL框架将学习过程分为两个阶段：1）上游表征学习阶段：使用带掩码的Transformer模块进行时间建模和对比学习，从序列视觉输入中提取时间感知和任务相关的表征；2）下游政策学习阶段：利用拥有全局状态信息的oracle教师政策在早期训练中提供指导，随训练进展逐渐减少指导以促进独立探索。

### 主要发现

在模拟和真实世界环境中的大量实验表明，OMC-RL实现了卓越的样本效率和渐进政策性能，同时提高了在多样性和感知复杂场景中的泛化能力。

### 结论

OMC-RL框架有效解决了视觉运动政策学习中的样本效率和模拟到现实差距问题，为强化学习在视觉控制任务中的应用提供了新思路。

### 翻译

学习视觉运动政策的一种普遍方法是使用强化学习将高维视觉观察直接映射到动作命令。然而，高维视觉输入和敏捷机动输出的组合导致长期挑战，包括样本效率低下和显著的模拟到现实差距。为解决这些问题，我们提出了Oracle-Guided Masked Contrastive Reinforcement Learning (OMC-RL)，这是一个新框架，旨在提高视觉运动政策学习的样本效率和渐进性能。OMC-RL明确将学习过程分为两个阶段：上游表征学习阶段和下游政策学习阶段。在上游阶段，带掩码的Transformer模块通过时间建模和对比学习进行训练，从序列视觉输入中提取时间感知和任务相关的表征。训练后，学习到的编码器被冻结用于从连续帧中提取视觉表征，而Transformer模块被丢弃。在下游阶段，拥有全局状态信息特权访问的oracle教师政策在早期训练中指导代理，提供信息性指导并加速早期政策学习。这种指导会逐渐减少，以允许随着训练进展进行独立探索。在模拟和真实世界环境中的大量实验表明，OMC-RL实现了卓越的样本效率和渐进政策性能，同时提高了在多样性和感知复杂场景中的泛化能力。


### 论文摘要

A prevailing approach for learning visuomotor policies is to employ reinforcement learning to map high-dimensional visual observations directly to action commands. However, the combination of high-dimensional visual inputs and agile maneuver outputs leads to long-standing challenges, including low sample efficiency and significant sim-to-real gaps. To address these issues, we propose Oracle-Guided Masked Contrastive Reinforcement Learning (OMC-RL), a novel framework designed to improve the sample efficiency and asymptotic performance of visuomotor policy learning. OMC-RL explicitly decouples the learning process into two stages: an upstream representation learning stage and a downstream policy learning stage. In the upstream stage, a masked Transformer module is trained with temporal modeling and contrastive learning to extract temporally-aware and task-relevant representations from sequential visual inputs. After training, the learned encoder is frozen and used to extract visual representations from consecutive frames, while the Transformer module is discarded. In the downstream stage, an oracle teacher policy with privileged access to global state information supervises the agent during early training to provide informative guidance and accelerate early policy learning. This guidance is gradually reduced to allow independent exploration as training progresses. Extensive experiments in simulated and real-world environments demonstrate that OMC-RL achieves superior sample efficiency and asymptotic policy performance, while also improving generalization across diverse and perceptually complex scenarios.

---

## 31. Contrastive Learning Using Graph Embeddings for Domain Adaptation of Language Models in the Process Industry

**论文链接:** [http://arxiv.org/abs/2510.04631v2](http://arxiv.org/abs/2510.04631v2)

**作者:** Anastasia Zhukova, Jonas Lührs, Christian E. Lobmüller, Bela Gipp

**发布时间:** 2025-10-06

**备注:** accepted to EMNLP 2025 (industry track)

### GPT解析

### 总结

本文探索了将针对科学出版物设计的图感知邻域对比学习方法SciNCL应用于过程工业领域，通过使用图嵌入三元组微调语言模型，在专有基准测试上显著优于最先进模型，同时模型参数量更少。

### 背景

最近NLP趋势利用知识图谱增强预训练语言模型，通过图结构中的额外知识学习领域特定术语或文档间关系。

### 目的

探索如何将SciNCL应用于过程工业领域，该领域的文本日志包含日常操作的关键信息，通常被构造成稀疏知识图谱。

### 方法

使用从图嵌入(GE)推导出的三元组对语言模型进行微调，在过程工业文本嵌入基准(PITEB)上进行评估。

### 主要发现

在PITEB上，使用图嵌入三元组微调的语言模型比最先进的mE5-large文本编码器高出9.8-14.3%(5.45-7.96p)，同时参数量减少3倍。

### 结论

图感知对比学习方法在过程工业的文本嵌入任务上表现优异，同时模型更轻量，具有实际应用价值。

### 翻译

最近的NLP趋势利用知识图谱(KGs)来增强预训练语言模型，通过从图结构中纳入额外知识来学习领域特定术语或文档间关系，这些关系可能被忽略。本文探讨了如何将SciNCL(一种最初为科学出版物设计的图感知邻域对比学习方法)应用于过程工业领域，该领域的文本日志包含关于日常操作的关键信息，通常被构造成稀疏知识图谱。我们的实验证明，使用从图嵌入(GE)推导出的三元组进行微调的语言模型在专有的过程工业文本嵌入基准(PITEB)上比最先进的mE5-large文本编码器高出9.8-14.3%(5.45-7.96p)，同时参数量减少3倍。


### 论文摘要

Recent trends in NLP utilize knowledge graphs (KGs) to enhance pretrained language models by incorporating additional knowledge from the graph structures to learn domain-specific terminology or relationships between documents that might otherwise be overlooked. This paper explores how SciNCL, a graph-aware neighborhood contrastive learning methodology originally designed for scientific publications, can be applied to the process industry domain, where text logs contain crucial information about daily operations and are often structured as sparse KGs. Our experiments demonstrate that language models fine-tuned with triplets derived from graph embeddings (GE) outperform a state-of-the-art mE5-large text encoder by 9.8-14.3% (5.45-7.96p) on the proprietary process industry text embedding benchmark (PITEB) while having 3 times fewer parameters.

---

## 32. Learning More with Less: A Generalizable, Self-Supervised Framework for Privacy-Preserving Capacity Estimation with EV Charging Data

**论文链接:** [http://arxiv.org/abs/2510.05172v1](http://arxiv.org/abs/2510.05172v1)

**作者:** Anushiya Arunan, Yan Qin, Xiaoli Li, U-Xuan Tan, H. Vincent Poor, Chau Yuen

**发布时间:** 2025-10-05

**DOI:** 10.1109/TII.2025.3613385

**备注:** Accepted in IEEE Transactions on Industrial Informatics

### GPT解析

### 总结

本研究提出了一种基于自监督预训练的电动汽车电池容量估计模型，利用大规模隐私友好型充电数据片段，通过片段相似性加权的掩码输入重建和对比学习技术，从特征较少且碎片化的数据中学习丰富、可泛化的表示，显著提高了容量估计的准确性。

### 背景

准确的电池容量估计对缓解消费者对电动车电池性能和可靠性的担忧至关重要。然而，严格的隐私法规和标记数据短缺限制了通用容量估计模型的发展。现有的自监督学习技术不能有效从具有挑战性的现场数据或隐私友好型数据（通常特征较少且噪声更大）中学习有效信息。

### 目的

开发一种基于自监督预训练的容量估计模型，利用大规模隐私友好型充电数据片段，从特征较少且碎片化的隐私友好型数据中学习丰富、可泛化的表示，提高电池容量估计的准确性。

### 方法

提出片段相似性加权的掩码输入重建预训练框架，利用对比学习捕获片段之间的高层次相似性，结合片段级对比学习和相似性加权掩码重建，学习单个片段内的细粒度充电模式和不同片段间的高层次关联关系。

### 主要发现

该模型在具有挑战性的领域偏移设置下表现一致优于最先进的基线，与性能最好的基准相比，测试误差降低了31.9%，能够处理由制造商和年龄引起的分布偏移。

### 结论

该研究首次提出基于自监督预训练的容量估计模型，能够从隐私友好型数据中学习丰富的表示，在实际应用中表现出色，特别是在处理具有挑战性的领域偏移情况时。

### 翻译

准确的电池容量估计是缓解消费者对电动汽车电池性能和可靠性担忧的关键。然而，严格的隐私法规限制和标记数据短缺阻碍了能够保持对现实世界数据分布偏移具有鲁棒性的通用容量估计模型的发展。虽然自监督学习可以利用未标记数据，但现有技术并非特别设计用于从具有挑战性的现场数据中有效学习——更不用说从隐私友好型数据中学习了，这些数据通常特征较少且噪声更大。在这项工作中，我们提出了首个基于自监督预训练的容量估计模型，该模型基于从现实世界电动汽车操作中收集的大规模隐私友好型充电数据片段集进行开发。我们的预训练框架——片段相似性加权的掩码输入重建——旨在从特征较少且碎片化的隐私友好型数据中学习丰富、可泛化的表示。我们的关键创新在于利用对比学习首先捕获碎片化片段之间的高层次相似性，否则这些片段缺乏有意义的上下文。通过我们的片段级对比学习和后续的相似性加权掩码重建，我们能够学习单个片段内细粒度充电模式和不同片段间高层次关联关系的丰富表示。凭借这种丰富的表示学习，我们的模型一致优于最先进的基线，即使在受到制造商和年龄引起的分布偏移影响的具有挑战性的领域偏移设置下，也比性能最好的基准实现了31.9%的更低测试误差。


### 论文摘要

Accurate battery capacity estimation is key to alleviating consumer concerns about battery performance and reliability of electric vehicles (EVs). However, practical data limitations imposed by stringent privacy regulations and labeled data shortages hamper the development of generalizable capacity estimation models that remain robust to real-world data distribution shifts. While self-supervised learning can leverage unlabeled data, existing techniques are not particularly designed to learn effectively from challenging field data -- let alone from privacy-friendly data, which are often less feature-rich and noisier. In this work, we propose a first-of-its-kind capacity estimation model based on self-supervised pre-training, developed on a large-scale dataset of privacy-friendly charging data snippets from real-world EV operations. Our pre-training framework, snippet similarity-weighted masked input reconstruction, is designed to learn rich, generalizable representations even from less feature-rich and fragmented privacy-friendly data. Our key innovation lies in harnessing contrastive learning to first capture high-level similarities among fragmented snippets that otherwise lack meaningful context. With our snippet-wise contrastive learning and subsequent similarity-weighted masked reconstruction, we are able to learn rich representations of both granular charging patterns within individual snippets and high-level associative relationships across different snippets. Bolstered by this rich representation learning, our model consistently outperforms state-of-the-art baselines, achieving 31.9% lower test error than the best-performing benchmark, even under challenging domain-shifted settings affected by both manufacturer and age-induced distribution shifts.

---

## 33. Parallel Tokenizers: Rethinking Vocabulary Design for Cross-Lingual Transfer

**论文链接:** [http://arxiv.org/abs/2510.06128v1](http://arxiv.org/abs/2510.06128v1)

**作者:** Muhammad Dehan Al Kautsar, Fajri Koto

**发布时间:** 2025-10-07

**备注:** 18 pages, 25 tables, 7 figures

### GPT解析

### 总结

本研究提出了并行标记器(parallel tokenizers)新框架，通过确保语义等价的词在不同语言中获得一致的表示，改进多语言语言模型中的标记化过程，提高跨语言迁移学习效果。

### 背景

现有多语言标记化方法无法支持有效的跨语言迁移，因为语义等价的词被分配到不同的嵌入表示中。例如，英语中的'I eat rice'和豪萨语中的'Ina cin shinkafa'通常被映射到不同的词汇索引，阻碍了共享表示和跨语言泛化。

### 目的

开发一种新的标记化框架，确保语义等价的词在不同语言中获得一致的表示，从而改进多语言表示学习，特别是在低资源语言环境中。

### 方法

提出并行标记器框架，首先单语训练标记器，然后使用双语词典或词到词翻译彻底对齐词汇表，确保语义等价的词具有一致的索引。在十三种低资源语言上预训练transformer编码器，并在情感分析、仇恨言论检测、情感分类和句子嵌入相似性任务上评估效果。

### 主要发现

在所有评估任务中，使用并行标记器训练的模型都优于传统的多语言基线模型，证实了重新思考标记化对于推进多语言表示学习的重要性。

### 结论

重新思考标记化方法对于推进多语言表示学习至关重要，特别是在低资源语言设置中。并行标记器框架确保语义等价的词在不同语言中获得一致的表示，提高了跨语言迁移学习的效果。

### 翻译

Tokenization: 标记化/分词；Multilingual language models: 多语言语言模型；Cross-lingual transfer: 跨语言迁移；Embeddings: 嵌入表示；Parallel tokenizers: 并行标记器；Bilingual dictionaries: 双语词典；Transformer encoder: Transformer编码器；Low-resource languages: 低资源语言；Sentiment analysis: 情感分析；Hate speech detection: 仇恨言论检测；Emotion classification: 情感分类；Sentence embedding similarity: 句子嵌入相似性


### 论文摘要

Tokenization defines the foundation of multilingual language models by determining how words are represented and shared across languages. However, existing methods often fail to support effective cross-lingual transfer because semantically equivalent words are assigned distinct embeddings. For example, "I eat rice" in English and "Ina cin shinkafa" in Hausa are typically mapped to different vocabulary indices, preventing shared representations and limiting cross-lingual generalization. We introduce parallel tokenizers. This new framework trains tokenizers monolingually and then aligns their vocabularies exhaustively using bilingual dictionaries or word-to-word translation, ensuring consistent indices for semantically equivalent words. This alignment enforces a shared semantic space across languages while naturally improving fertility balance. To assess their effectiveness, we pretrain a transformer encoder from scratch on thirteen low-resource languages and evaluate it on sentiment analysis, hate speech detection, emotion classification, and sentence embedding similarity. Across all tasks, models trained with parallel tokenizers outperform conventional multilingual baselines, confirming that rethinking tokenization is essential for advancing multilingual representation learning--especially in low-resource settings.

---

## 34. Multimodal Trajectory Representation Learning for Travel Time Estimation

**论文链接:** [http://arxiv.org/abs/2510.05840v1](http://arxiv.org/abs/2510.05840v1)

**作者:** Zhi Liu, Xuyuan Hu, Xiao Han, Zhehao Dai, Zhaolin Deng, Guojiang Shen, Xiangjie Kong

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种多模态动态轨迹集成（MDTI）框架，通过整合GPS序列、网格轨迹和道路网络约束来提高行程时间估计（TTE）的准确性。

### 背景

准确的行程时间估计在智能交通系统中至关重要，但由于异构数据源和复杂交通动态，这仍然具有挑战性。传统方法将轨迹转换为固定长度表示，忽略了现实世界轨迹的固有变异性，导致信息丢失或特征冗余。

### 目的

解决行程时间估计中的挑战，提高准确性，通过整合多种数据源和动态建模来克服传统方法的局限性。

### 方法

MDTI框架采用模态特定编码器和跨模态交互模块捕获互补的空间、时间和拓扑语义，并使用动态轨迹建模机制自适应调节不同长度轨迹的信息密度。同时，通过对比对齐和掩码语言建模两种自监督预训练目标来加强多模态一致性和上下文理解。

### 主要发现

在三个真实世界数据集上的大量实验表明，MDTI框架始终优于最先进的基线方法，证实了其鲁棒性和强大的泛化能力。

### 结论

MDTI框架有效解决了行程时间估计中的挑战，通过整合多种数据源和动态建模显著提高了准确性，代码已在GitHub上公开。

### 翻译

准确的行程时间估计在智能交通系统中起着关键作用。然而，由于异构数据源和复杂的交通动态，这仍然具有挑战性。此外，传统方法通常将轨迹转换为固定长度的表示，忽略了现实世界轨迹的固有变异性，这往往导致信息丢失或特征冗余。为了应对这些挑战，本文引入了多模态动态轨迹集成（MDTI）框架——一种新颖的多模态轨迹表示学习方法，它集成GPS序列、网格轨迹和道路网络约束以提高TTE准确性。MDTI采用模态特定编码器和跨模态交互模块来捕获互补的空间、时间和拓扑语义，同时动态轨迹建模机制自适应调节不同长度轨迹的信息密度。两种名为对比对齐和掩码语言建模的自监督预训练目标进一步加强了多模态一致性和上下文理解。在三个真实世界数据集上的大量实验表明，MDTI始终优于最先进的基线方法，证实了其鲁棒性和强大的泛化能力。代码已在https://github.com/freshhxy/MDTI/公开提供。


### 论文摘要

Accurate travel time estimation (TTE) plays a crucial role in intelligent transportation systems. However, it remains challenging due to heterogeneous data sources and complex traffic dynamics. Moreover, conventional approaches typically convert trajectories into fixed-length representations, neglecting the inherent variability of real-world trajectories, which often leads to information loss or feature redundancy. To address these challenges, this paper introduces the Multimodal Dynamic Trajectory Integration (MDTI) framework--a novel multimodal trajectory representation learning approach that integrates GPS sequences, grid trajectories, and road network constraints to enhance TTE accuracy. MDTI employs modality-specific encoders and a cross-modal interaction module to capture complementary spatial, temporal, and topological semantics, while a dynamic trajectory modeling mechanism adaptively regulates information density for trajectories of varying lengths. Two self-supervised pretraining objectives, named contrastive alignment and masked language modeling, further strengthen multimodal consistency and contextual understanding. Extensive experiments on three real-world datasets demonstrate that MDTI consistently outperforms state-of-the-art baselines, confirming its robustness and strong generalization abilities. The code is publicly available at: https://github.com/freshhxy/MDTI/

---

## 35. DiffSDA: Unsupervised Diffusion Sequential Disentanglement Across Modalities

**论文链接:** [http://arxiv.org/abs/2510.05717v1](http://arxiv.org/abs/2510.05717v1)

**作者:** Hedi Zisling, Ilan Naiman, Nimrod Berman, Supasorn Suwajanakorn, Omri Azencot

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种名为扩散序列解缠结自编码器(DiffSDA)的新框架，用于解决无监督表征学习中的序列解缠结问题，在多种真实世界数据模态上表现优于现有方法。

### 背景

无监督表征学习特别是序列解缠结旨在分离数据中的静态和动态变化因素而不依赖标签。现有基于变分自编码器和生成对抗网络的方法存在优化复杂、应用于真实世界数据效果不佳以及缺乏成熟评估协议等问题。

### 目的

开发一种新的框架来解决序列解缠结问题，使其能够有效应用于多种真实世界数据模态，并提供严格的评估协议。

### 方法

提出DiffSDA框架，这是一种新颖的、与模态无关的方法，利用新的概率建模、潜在扩散和高效采样器，同时引入具有挑战性的评估协议进行严格测试。

### 主要发现

在多样化的真实世界基准测试中，DiffSDA优于现有的最先进序列解缠结方法，证明了其在时间序列、视频和音频等多种数据模态上的有效性。

### 结论

DiffSDA为序列解缠结问题提供了有效的解决方案，填补了扩散模型在这一应用领域的理论空白，并建立了新的评估标准。

### 翻译

无监督表征学习，特别是序列解缠结，旨在不依赖标签的情况下分离数据中的静态和动态变化因素。这仍然是一个具有挑战性的问题，因为基于变分自编码器和生成对抗网络的现有方法通常依赖多个损失项，使优化过程复杂化。此外，序列解缠结方法应用于真实世界数据时面临挑战，目前还没有既定的评估协议来评估它们在这种情况下性能。最近，扩散模型已成为最先进的生成模型，但其在序列解缠结应用方面尚无理论形式化。在这项工作中，我们引入了扩散序列解缠结自编码器(DiffSDA)，这是一个新颖的、与模态无关的框架，在包括时间序列、视频和音频在内的各种真实世界数据模态中有效。DiffSDA利用新的概率建模、潜在扩散和高效采样器，同时纳入具有挑战性的评估协议进行严格测试。我们在各种真实世界基准上的实验证明，DiffSDA在序列解缠结方面优于最近的最先进方法。


### 论文摘要

Unsupervised representation learning, particularly sequential disentanglement, aims to separate static and dynamic factors of variation in data without relying on labels. This remains a challenging problem, as existing approaches based on variational autoencoders and generative adversarial networks often rely on multiple loss terms, complicating the optimization process. Furthermore, sequential disentanglement methods face challenges when applied to real-world data, and there is currently no established evaluation protocol for assessing their performance in such settings. Recently, diffusion models have emerged as state-of-the-art generative models, but no theoretical formalization exists for their application to sequential disentanglement. In this work, we introduce the Diffusion Sequential Disentanglement Autoencoder (DiffSDA), a novel, modal-agnostic framework effective across diverse real-world data modalities, including time series, video, and audio. DiffSDA leverages a new probabilistic modeling, latent diffusion, and efficient samplers, while incorporating a challenging evaluation protocol for rigorous testing. Our experiments on diverse real-world benchmarks demonstrate that DiffSDA outperforms recent state-of-the-art methods in sequential disentanglement.

---

## 36. Permutation-Invariant Representation Learning for Robust and Privacy-Preserving Feature Selection

**论文链接:** [http://arxiv.org/abs/2510.05535v1](http://arxiv.org/abs/2510.05535v1)

**作者:** Rui Liu, Tao Zhe, Yanjie Fu, Feng Xia, Ted Senator, Dongjie Wang

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种在联邦学习场景下进行特征选择的创新框架，解决了现有方法在特征交互捕捉、场景适应性、排列敏感性和凸性假设限制等方面的问题，同时处理了数据不平衡、异构性和隐私保护挑战。

### 背景

现有特征选择方法难以捕捉复杂特征交互且适应性差；近期采用生成智能的方法仍受排列敏感性和凸性假设限制；实际分布式环境中数据高度不平衡、异构且受隐私法规限制，无法直接共享。

### 目的

开发隐私保护的知识融合策略，在不共享敏感原始数据的情况下推导统一表示空间；引入样本感知的加权策略，解决异构本地客户端间的分布不平衡问题，提高特征选择在联邦学习中的效果。

### 方法

提出整合排列不变嵌入与策略引导搜索的新框架；在期刊扩展版本中增加两方面改进：1)隐私保护的知识融合策略；2)样本感知的加权策略，以处理数据不平衡和异构性问题。

### 主要发现

大量实验验证了框架的有效性、鲁棒性和效率；结果表明该方法在联邦学习场景中具有强大的泛化能力，能够有效集成客户端间的特征选择知识而不暴露敏感信息。

### 结论

所提出的框架成功解决了分布式场景中的数据不平衡、异构性和隐私保护问题，为联邦学习环境下的特征选择提供了有效解决方案。

### 翻译

特征选择消除特征间的冗余，以提高下游任务性能同时减少计算开销。现有方法通常难以捕捉复杂的特征交互并适应多样化的应用场景。最近的进展采用生成智能来缓解这些缺点。然而，这些方法仍受嵌入中的排列敏感性和基于梯度搜索对凸性假设的依赖限制。为解决这些局限性，我们初步工作引入了一种整合排列不变嵌入与策略引导搜索的新框架。尽管有效，它仍存在适应真实分布式场景的改进空间。实际上，本地客户端的数据高度不平衡、异构，且受严格隐私法规限制，限制了直接共享。这些挑战凸显了需要一种能够在不暴露敏感信息的情况下集成客户端间特征选择知识的框架。在本扩展期刊版本中，我们从两个角度推进框架：1)开发隐私保护的知识融合策略，在不共享敏感原始数据的情况下推导统一表示空间；2)引入样本感知的加权策略，解决异构本地客户端间的分布不平衡问题。大量实验验证了我们框架的有效性、鲁棒性和效率。结果进一步证明了其在联邦学习场景中的强大泛化能力。代码和数据公开可用：https://anonymous.4open.science/r/FedCAPS-08BF。


### 论文摘要

Feature selection eliminates redundancy among features to improve downstream task performance while reducing computational overhead. Existing methods often struggle to capture intricate feature interactions and adapt across diverse application scenarios. Recent advances employ generative intelligence to alleviate these drawbacks. However, these methods remain constrained by permutation sensitivity in embedding and reliance on convexity assumptions in gradient-based search. To address these limitations, our initial work introduces a novel framework that integrates permutation-invariant embedding with policy-guided search. Although effective, it still left opportunities to adapt to realistic distributed scenarios. In practice, data across local clients is highly imbalanced, heterogeneous and constrained by strict privacy regulations, limiting direct sharing. These challenges highlight the need for a framework that can integrate feature selection knowledge across clients without exposing sensitive information. In this extended journal version, we advance the framework from two perspectives: 1) developing a privacy-preserving knowledge fusion strategy to derive a unified representation space without sharing sensitive raw data. 2) incorporating a sample-aware weighting strategy to address distributional imbalance among heterogeneous local clients. Extensive experiments validate the effectiveness, robustness, and efficiency of our framework. The results further demonstrate its strong generalization ability in federated learning scenarios. The code and data are publicly available: https://anonymous.4open.science/r/FedCAPS-08BF.

---

## 37. AUREXA-SE: Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement

**论文链接:** [http://arxiv.org/abs/2510.05295v1](http://arxiv.org/abs/2510.05295v1)

**作者:** M. Sajid, Deepanshu Gupta, Yash Modi, Sanskriti Jain, Harshith Jai Surya Ganji, A. Rahaman, Harshvardhan Choudhary, Nasir Saleem, Amir Hussain, M. Tanveer

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出了AUREXA-SE（使用交叉注意力和Squeezeformer进行语音增强的视听统一表示交换架构），这是一个专门为视听语音增强设计的渐进式双模态框架。该模型通过结合音频和视觉信息，利用交叉注意力机制和Squeezeformer块，显著提高了嘈杂环境下的语音质量和可理解性。

### 背景

语音增强是语音处理领域的重要任务，特别是在嘈杂环境下提高语音质量和可理解性。视听语音增强（AVSE）利用视觉信息（如说话者的口型）来辅助语音增强，这在嘈杂环境或低信噪比情况下尤为重要。

### 目的

开发一个先进的视听语音增强框架，有效结合音频和视觉信息，提高嘈杂环境下的语音质量和可理解性。

### 方法

1. 使用基于U-Net的一维卷积编码器处理原始音频波形；2. 采用Swin Transformer V2进行视觉特征提取；3. 设计双向交叉注意力机制促进模态间深度上下文融合；4. 引入轻量级Squeezeformer块捕获融合嵌入中的时间依赖关系；5. 通过U-Net风格解码器进行波形重建。

### 主要发现

1. 实验证明AUREXA-SE在嘈杂基线上有显著性能提升；2. 具体性能指标：STOI为0.516，PESQ为1.323，SI-SDR为-4.322 dB；3. 模型能够产生感知一致且可理解的语音输出。

### 结论

AUREXA-SE是一个有效的视听语音增强框架，通过结合音频和视觉信息，利用交叉注意力机制和Squeezeformer块，显著提高了嘈杂环境下的语音质量和可理解性。该模型在多个评估指标上均表现出色，证明了其作为语音增强解决方案的有效性。

### 翻译

在本文中，我们提出了AUREXA-SE（使用交叉注意力和Squeezeformer进行语音增强的视听统一表示交换架构），这是一个专门为视听语音增强设计的渐进式双模态框架。AUREXA-SE通过使用基于U-Net的一维卷积编码器处理音频，以及使用Swin Transformer V2进行高效且表达性强的视觉特征提取，共同利用原始音频波形和视觉线索。该架构的核心是一个新颖的双向交叉注意力机制，它促进了模态间的深度上下文融合，实现了丰富且互补的表示学习。为了捕获融合嵌入中的时间依赖关系，引入了一系列轻量级Squeezeformer块，结合了卷积和注意力模块。增强的嵌入随后通过U-Net风格解码器进行解码，直接进行波形重建，确保感知一致且可理解的语音输出。实验评估证明了AUREXA-SE的有效性，与嘈杂基线相比实现了显著的性能提升，STOI为0.516，PESQ为1.323，SI-SDR为-4.322 dB。AUREXA-SE的源代码可在https://github.com/mtanveer1/AVSEC-4-Challenge-2025获取。


### 论文摘要

In this paper, we propose AUREXA-SE (Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement), a progressive bimodal framework tailored for audio-visual speech enhancement (AVSE). AUREXA-SE jointly leverages raw audio waveforms and visual cues by employing a U-Net-based 1D convolutional encoder for audio and a Swin Transformer V2 for efficient and expressive visual feature extraction. Central to the architecture is a novel bidirectional cross-attention mechanism, which facilitates deep contextual fusion between modalities, enabling rich and complementary representation learning. To capture temporal dependencies within the fused embeddings, a stack of lightweight Squeezeformer blocks combining convolutional and attention modules is introduced. The enhanced embeddings are then decoded via a U-Net-style decoder for direct waveform reconstruction, ensuring perceptually consistent and intelligible speech output. Experimental evaluations demonstrate the effectiveness of AUREXA-SE, achieving significant performance improvements over noisy baselines, with STOI of 0.516, PESQ of 1.323, and SI-SDR of -4.322 dB. The source code of AUREXA-SE is available at https://github.com/mtanveer1/AVSEC-4-Challenge-2025.

---

## 38. Provable Speech Attributes Conversion via Latent Independence

**论文链接:** [http://arxiv.org/abs/2510.05191v1](http://arxiv.org/abs/2510.05191v1)

**作者:** Jonathan Svirsky, Ofir Lindenbaum, Uri Shaham

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出了一种通用的语音属性转换框架，通过理论分析和保证解决了现有方法缺乏理论基础的问题

### 背景

信号转换和解缠表示学习在音频、图像和多模态生成等领域显示出潜力，但现有语音风格转换方法大多是经验性的，缺乏严格的理论基础来保证可靠和可解释的控制

### 目的

提出一个通用的语音属性转换框架，并在合理假设下提供理论分析和保证

### 方法

构建基于非概率自编码器架构的框架，在预测的潜变量和目标可控变量之间施加独立性约束，确保在观察到风格变量条件下的信号转换一致性，同时保留原始内容并修改所需属性

### 主要发现

通过在说话人身份和情感等语音风格上的评估，证明了该方法的通用性；定量评估证实了所提出方法的有效性和通用性

### 结论

该框架为语音属性转换提供了理论基础，确保了可靠和可解释的控制

### 翻译

尽管信号转换和解缠表示学习在音频、图像和多模态生成等跨领域的数据属性操作方面显示出前景，但现有方法，特别是语音风格转换方面，大多是经验性的，缺乏严格的理论基础来保证可靠和可解释的控制。在这项工作中，我们提出了一个语音属性转换的通用框架，并在合理假设下提供了理论分析和保证。我们的框架构建在一个非概率自编码器架构上，预测的潜变量与目标可控变量之间具有独立性约束。这种设计确保了在观察到风格变量条件下的信号转换一致性，同时保留原始内容并修改所需属性。我们通过在语音风格（包括说话人身份和情感）上的评估进一步证明了我们方法的通用性。定量评估证实了所提出方法的有效性和通用性。


### 论文摘要

While signal conversion and disentangled representation learning have shown promise for manipulating data attributes across domains such as audio, image, and multimodal generation, existing approaches, especially for speech style conversion, are largely empirical and lack rigorous theoretical foundations to guarantee reliable and interpretable control. In this work, we propose a general framework for speech attribute conversion, accompanied by theoretical analysis and guarantees under reasonable assumptions. Our framework builds on a non-probabilistic autoencoder architecture with an independence constraint between the predicted latent variable and the target controllable variable. This design ensures a consistent signal transformation, conditioned on an observed style variable, while preserving the original content and modifying the desired attribute. We further demonstrate the versatility of our method by evaluating it on speech styles, including speaker identity and emotion. Quantitative evaluations confirm the effectiveness and generality of the proposed approach.

---

## 39. When and How to Cut Classical Concerts? A Multimodal Automated Video Editing Approach

**论文链接:** [http://arxiv.org/abs/2510.05661v1](http://arxiv.org/abs/2510.05661v1)

**作者:** Daniel Gonzálbez-Biosca, Josep Cabacas-Maso, Carles Ventura, Ismael Benito-Altamirano

**发布时间:** 2025-10-07

**DOI:** 10.1145/3746278.3759387

### GPT解析

### 总结

本研究提出了一种多模态自动视频编辑方法，特别针对古典音乐会多摄像头录像的编辑问题，将任务分解为'何时剪切'和'如何剪切'两个子任务，并在检测剪切点和视觉镜头选择方面取得了优于先前基线的结果。

### 背景

自动视频编辑在计算机视觉和多媒体领域是一个探索不足的任务，与日益增长的视频生成和场景理解研究兴趣相比，视频编辑的研究相对较少。

### 目的

解决多摄像头古典音乐会录像编辑的具体挑战，通过分解问题为两个关键子任务：何时剪切和如何剪切，以推进多模态自动视频编辑的技术水平。

### 方法

为时间分割任务提出了一种新的多模态架构，整合音频信号的log-mel频谱图、可选图像嵌入和标量时间特征，通过轻量级卷积-Transformer管道处理；为空间选择任务使用基于CLIP的编码器替换旧的骨干网络，并将干扰项选择限制在同一音乐会的片段中；采用伪标记方法构建数据集，将原始视频数据自动聚类为连贯的镜头片段。

### 主要发现

模型在检测剪切点方面优于之前的基线方法，同时提供了具有竞争力的视觉镜头选择能力。

### 结论

该研究成功推进了多模态自动视频编辑领域的最先进技术，为自动视频编辑提供了新的解决方案。

### 翻译

自动视频编辑在计算机视觉和多媒体领域仍然是一个探索不足的任务，特别是与日益增长的视频生成和场景理解研究兴趣相比。在本研究中，我们通过将问题分解为两个关键子任务来解决古典音乐会多摄像头录像编辑的具体挑战：何时剪切和如何剪切。基于近期文献，我们为时间分割任务(何时剪切)提出了一种新的多模态架构，该架构通过轻量级卷积-Transformer管道整合了音频信号的log-mel频谱图、可选图像嵌入和标量时间特征。对于空间选择任务(如何剪切)，我们通过使用基于CLIP的编码器替换旧的骨干网络(如ResNet)并限制干扰项选择来自同一音乐会的片段来改进现有文献。我们的数据集采用伪标记方法构建，将原始视频数据自动聚类为连贯的镜头片段。我们证明，我们的模型在检测剪切点方面优于之前的基线，并提供了具有竞争力的视觉镜头选择，推进了多模态自动视频编辑的最先进技术。


### 论文摘要

Automated video editing remains an underexplored task in the computer vision and multimedia domains, especially when contrasted with the growing interest in video generation and scene understanding. In this work, we address the specific challenge of editing multicamera recordings of classical music concerts by decomposing the problem into two key sub-tasks: when to cut and how to cut. Building on recent literature, we propose a novel multimodal architecture for the temporal segmentation task (when to cut), which integrates log-mel spectrograms from the audio signals, plus an optional image embedding, and scalar temporal features through a lightweight convolutional-transformer pipeline. For the spatial selection task (how to cut), we improve the literature by updating from old backbones, e.g. ResNet, with a CLIP-based encoder and constraining distractor selection to segments from the same concert. Our dataset was constructed following a pseudo-labeling approach, in which raw video data was automatically clustered into coherent shot segments. We show that our models outperformed previous baselines in detecting cut points and provide competitive visual shot selection, advancing the state of the art in multimodal automated video editing.

---

## 40. HoloScene: Simulation-Ready Interactive 3D Worlds from a Single Video

**论文链接:** [http://arxiv.org/abs/2510.05560v1](http://arxiv.org/abs/2510.05560v1)

**作者:** Hongchi Xia, Chih-Hao Lin, Hao-Yu Hsu, Quentin Leboutet, Katelyn Gao, Michael Paulitsch, Benjamin Ummenhofer, Shenlong Wang

**发布时间:** 2025-10-07

**备注:** Project page: https://xiahongchi.github.io/HoloScene

### GPT解析

### 总结

HoloScene是一种新颖的交互式3D重建框架，解决了现有方法在几何完整性、对象交互性、物理合理性、照片级真实感渲染和动态模拟物理属性方面的局限性，实现了全面的数字孪生重建。

### 背景

将物理世界数字化为准确的模拟就绪虚拟环境在增强现实、虚拟现实、游戏和机器人技术等领域有重要应用机会，但当前3D重建和场景理解方法通常在一个或多个关键方面存在不足。

### 目的

开发一种能同时满足几何完整性、对象交互性、物理合理性、照片级真实感渲染和可靠动态模拟物理属性的交互式3D重建框架。

### 方法

HoloScene利用全面的交互式场景图表示法编码对象几何、外观和物理属性以及层次间和对象间关系；将重建表述为基于能量的优化问题，整合观测数据、物理约束和生成先验；通过结合基于采样的探索和基于梯度的细化的混合方法高效执行优化。

### 主要发现

生成的数字孪生表现出完整精确的几何、物理稳定性和新视角的真实感渲染；在多个基准数据集上评估显示优越性能；交互式游戏和实时数字孪生操作的实际用例证明了其广泛适用性和有效性。

### 结论

HoloScene成功解决了现有3D重建方法的局限性，实现了多种关键要求的统一，为物理世界的数字化提供了有效解决方案。

### 翻译

将物理世界数字化为准确的模拟就绪虚拟环境在增强现实、虚拟现实、游戏和机器人技术等多个领域提供了重要机会。然而，当前的3D重建和场景理解方法通常在一个或多个关键方面存在不足，如几何完整性、对象交互性、物理合理性、照片级真实感渲染或可靠动态模拟所需的物理属性。为解决这些局限性，我们引入了HoloScene，一种新颖的交互式3D重建框架，可同时满足这些要求。HoloScene利用全面的交互式场景图表示法，编码对象几何、外观和物理属性以及层次间和对象间关系。重建被表述为基于能量的优化问题，将观测数据、物理约束和生成先验整合到统一、连贯的目标中。优化通过结合基于采样的探索和基于梯度的细化的混合方法高效执行。生成的数字孪生表现出完整且精确的几何、物理稳定性和从新视角的真实感渲染。在多个基准数据集上进行的评估展示了优越性能，而交互式游戏和实时数字孪生操作的实际用例说明了HoloScene的广泛适用性和有效性。项目页面：https://xiahongchi.github.io/HoloScene。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从单个视频中重建出完整的、物理上合理的、可交互的3D数字孪生环境的问题。这个问题在现实中非常重要，因为数字化物理世界为增强现实、虚拟现实、游戏和机器人等领域提供了巨大机会，而现有方法无法同时满足几何完整性、物理合理性和交互性的需求，限制了数字孪生技术在机器人学习、自动驾驶和内容创作等领域的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有3D重建方法在几何完整性、物理合理性和交互性方面的不足，特别是在处理遮挡区域、推断对象间关系和确保物理稳定性方面存在挑战。他们分析了现有Real2Sim方法、Amodal重建和物理合理重建的局限性，然后设计了基于交互式场景图表示的方法，将场景图重建制定为基于能量的优化问题，整合观测数据、物理约束和生成先验。该方法借鉴了神经SDF表示几何形状、3D Gaussian Splatting技术进行渲染、图像到3D生成方法完成遮挡区域以及可微分物理和模拟工作确保稳定性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将场景表示为一个交互式3D场景图，编码对象几何、外观、物理属性以及层次化的对象间关系，通过基于能量的优化方法同时实现几何完整性、物理合理性和交互性。整体实现流程分为三个阶段：1) 基于梯度的优化初始化，优化对象几何和外观使其与观测数据匹配；2) 基于采样的优化，通过生成采样和树搜索完善形状和物理参数；3) 基于梯度的细化，精调高斯斑点确保真实外观。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：统一的交互式3D重建框架首次同时实现几何完整性、物理合理性和交互性；交互式场景图表示支持物理交互；基于能量的优化方法整合观测数据、物理约束和生成先验；混合优化策略结合采样和梯度细化；生成先验与物理约束的结合。相比之前工作，HoloScene执行多视图联合优化而非单图像生成；使用循环模拟器确保长期物理稳定性而非仅避免穿透；采用统一的基于能量公式而非多阶段管道；能够处理复杂室内场景中的对象间遮挡和交互。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HoloScene提出了一种创新的交互式3D重建框架，通过结合场景图表示、基于能量的优化和混合推理策略，首次实现了从单个视频中重建出几何完整、物理合理且可交互的3D数字孪生环境。'}


### 论文摘要

Digitizing the physical world into accurate simulation-ready virtual environments offers significant opportunities in a variety of fields such as augmented and virtual reality, gaming, and robotics. However, current 3D reconstruction and scene-understanding methods commonly fall short in one or more critical aspects, such as geometry completeness, object interactivity, physical plausibility, photorealistic rendering, or realistic physical properties for reliable dynamic simulation. To address these limitations, we introduce HoloScene, a novel interactive 3D reconstruction framework that simultaneously achieves these requirements. HoloScene leverages a comprehensive interactive scene-graph representation, encoding object geometry, appearance, and physical properties alongside hierarchical and inter-object relationships. Reconstruction is formulated as an energy-based optimization problem, integrating observational data, physical constraints, and generative priors into a unified, coherent objective. Optimization is efficiently performed via a hybrid approach combining sampling-based exploration with gradient-based refinement. The resulting digital twins exhibit complete and precise geometry, physical stability, and realistic rendering from novel viewpoints. Evaluations conducted on multiple benchmark datasets demonstrate superior performance, while practical use-cases in interactive gaming and real-time digital-twin manipulation illustrate HoloScene's broad applicability and effectiveness. Project page: https://xiahongchi.github.io/HoloScene.

---

## 41. Sci-Phi: A Large Language Model Spatial Audio Descriptor

**论文链接:** [http://arxiv.org/abs/2510.05542v1](http://arxiv.org/abs/2510.05542v1)

**作者:** Xilin Jiang, Hannes Gamper, Sebastian Braun

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文介绍了Sci-Phi，一个能够全面描述声景的空间音频大型语言模型，可以估计声源和环境的完整参数集。

### 背景

声景感知涉及描述声音的类型、时间、方向、距离、响度和混响。虽然音频语言模型在声音识别方面表现出色，但单通道输入限制了空间理解能力。

### 目的

开发一个能够进行全面空间场景描述的音频大型语言模型，克服单通道输入的空间理解限制。

### 方法

提出了Sci-Phi，一个具有双空间和频谱编码器的空间音频大型语言模型，从超过4,000小时的合成一阶Ambisonics录音中学习，能够估计所有声源和周围环境的完整参数集。

### 主要发现

Sci-Phi能够一次列举并描述多达四个方向性声源，以及非方向性背景声音和房间特性；在15个指标(涵盖内容、位置、时间、响度和混响)的评估中表现出色；能够推广到真实的房间脉冲响应，性能只有轻微下降。

### 结论

Sci-Phi是第一个能够进行全面空间场景描述的音频LLM，具有强大的实际部署潜力。

### 翻译

声景感知涉及描述声音的类型、时间、方向和距离，以及它们的响度和混响。虽然音频语言模型在声音识别方面表现出色，但单通道输入从根本上限制了空间理解。这项工作提出了Sci-Phi，一个具有双空间和频谱编码器的空间音频大型语言模型，可以估计所有声源和周围环境的完整参数集。从超过4,000小时的合成一阶Ambisonics录音(包括元数据)中学习，Sci-Phi一次列举并描述多达四个方向性声源，以及非方向性背景声音和房间特性。我们使用排列不变协议和15个指标(涵盖内容、位置、时间、响度和混响)评估了该模型，并分析了其在源数量、信噪比、混响水平以及声学、空间或时间上相似的源的挑战性混合物方面的鲁棒性。值得注意的是，Sci-Phi能够推广到真实的房间脉冲响应，性能只有轻微下降。总的来说，这项工作建立了第一个能够进行全面空间场景描述的音频LLM，具有强大的实际部署潜力。演示：https://sci-phi-audio.github.io/demo


### 论文摘要

Acoustic scene perception involves describing the type of sounds, their timing, their direction and distance, as well as their loudness and reverberation. While audio language models excel in sound recognition, single-channel input fundamentally limits spatial understanding. This work presents Sci-Phi, a spatial audio large language model with dual spatial and spectral encoders that estimates a complete parameter set for all sound sources and the surrounding environment. Learning from over 4,000 hours of synthetic first-order Ambisonics recordings including metadata, Sci-Phi enumerates and describes up to four directional sound sources in one pass, alongside non-directional background sounds and room characteristics. We evaluate the model with a permutation-invariant protocol and 15 metrics covering content, location, timing, loudness, and reverberation, and analyze its robustness across source counts, signal-to-noise ratios, reverberation levels, and challenging mixtures of acoustically, spatially, or temporally similar sources. Notably, Sci-Phi generalizes to real room impulse responses with only minor performance degradation. Overall, this work establishes the first audio LLM capable of full spatial-scene description, with strong potential for real-world deployment. Demo: https://sci-phi-audio.github.io/demo

---

## 42. Large Language Models Achieve Gold Medal Performance at the International Olympiad on Astronomy & Astrophysics (IOAA)

**论文链接:** [http://arxiv.org/abs/2510.05016v2](http://arxiv.org/abs/2510.05016v2)

**作者:** Lucas Carrit Delgado Pinheiro, Ziru Chen, Bruno Caixeta Piazza, Ness Shroff, Yingbin Liang, Yuan-Sen Ting, Huan Sun

**发布时间:** 2025-10-06

**备注:** 18 pages, 6 figures, to be submitted, comments are welcome.  Reproducibility details can be found at:  https://github.com/OSU-NLP-Group/LLM-IOAA

### GPT解析

### 总结

研究通过国际天文和天体物理学奥林匹克竞赛(IOAA)考试系统评估了五种最先进的大型语言模型，发现Gemini 2.5 Pro和GPT-5在理论考试中达到金牌水平，接近人类顶尖表现，但在数据分析考试中表现差异较大，所有模型在概念推理、几何推理和空间可视化方面存在明显弱点。

### 背景

当前基于特定任务演示的方法在将大型语言模型应用于自动化天文研究任务方面显示出早期成功，但仅提供了解决问题所需能力的不完整视图。现有的基准测试和评估主要集中在简单问答上，主要测试天文知识，未能评估现实研究中所需的复杂推理能力。

### 目的

解决现有基准测试的局限性，通过IOAA考试系统评估大型语言模型在深入概念理解、多步推导和多模式分析方面的能力，以更全面地理解LLM的优势和局限性。

### 方法

使用国际天文和天体物理学奥林匹克竞赛(IOAA)考试对五种最先进的大型语言模型进行系统基准测试，考试包括理论和数据分析部分，涵盖2022-2025年的四届比赛。

### 主要发现

Gemini 2.5 Pro和GPT-5分别获得85.6%和84.2%的理论考试平均分，达到金牌水平且排名参赛者前两名；在数据分析考试中，GPT-5平均得分为88.5%排名前10，其他模型性能下降至48-76%；所有模型在概念推理、几何推理和空间可视化方面准确率仅为52-79%，存在明显弱点。

### 结论

尽管大型语言模型在理论考试中接近顶尖人类表现，但在它们能够作为天文学自主研究代理之前，必须解决概念推理、几何推理和空间可视化等关键能力差距。

### 翻译

虽然特定任务的演示在将大型语言模型应用于自动化某些天文研究任务方面显示出早期成功，但它们仅提供了解决天文问题所需能力的片面视图，需要更全面地理解大型语言模型的优势和局限性。迄今为止，现有的基准测试和评估主要集中在简单的问答上，主要测试天文知识，未能评估该学科现实研究所需的复杂推理。在这里，我们通过在国际天文和天体物理学奥林匹克竞赛(IOAA)考试上系统性地对五种最先进的大型语言模型进行基准测试来解决这一差距，这些考试旨在检验深入的概念理解、多步推导和多模式分析能力。凭借85.6%和84.2%的平均分，Gemini 2.5 Pro和GPT-5（两个表现最佳的模型）不仅达到金牌水平，而且在所有四个IOAA理论考试（2022-2025年）中排名在前200-300名参赛者中的前两名。相比之下，数据分析考试的结果显示出更大的差异。GPT-5在这些考试中仍然表现出色，平均得分为88.5%，在最近四届IOAA中排名前10，而其他模型的性能下降到48-76%。此外，我们深入的错误分析强调概念推理、几何推理和空间可视化（52-79%的准确率）是所有大型语言模型的一致弱点。因此，尽管大型语言模型在理论考试中接近顶尖人类表现，但在它们能够作为天文学自主研究代理之前，必须解决关键差距。


### 论文摘要

While task-specific demonstrations show early success in applying large language models (LLMs) to automate some astronomical research tasks, they only provide incomplete views of all necessary capabilities in solving astronomy problems, calling for more thorough understanding of LLMs' strengths and limitations. So far, existing benchmarks and evaluations focus on simple question-answering that primarily tests astronomical knowledge and fails to evaluate the complex reasoning required for real-world research in the discipline. Here, we address this gap by systematically benchmarking five state-of-the-art LLMs on the International Olympiad on Astronomy and Astrophysics (IOAA) exams, which are designed to examine deep conceptual understanding, multi-step derivations, and multimodal analysis. With average scores of 85.6% and 84.2%, Gemini 2.5 Pro and GPT-5 (the two top-performing models) not only achieve gold medal level performance but also rank in the top two among ~200-300 participants in all four IOAA theory exams evaluated (2022-2025). In comparison, results on the data analysis exams show more divergence. GPT-5 still excels in the exams with an 88.5% average score, ranking top 10 among the participants in the four most recent IOAAs, while other models' performances drop to 48-76%. Furthermore, our in-depth error analysis underscores conceptual reasoning, geometric reasoning, and spatial visualization (52-79% accuracy) as consistent weaknesses among all LLMs. Hence, although LLMs approach peak human performance in theory exams, critical gaps must be addressed before they can serve as autonomous research agents in astronomy.

---

## 43. AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials

**论文链接:** [http://arxiv.org/abs/2510.04704v2](http://arxiv.org/abs/2510.04704v2)

**作者:** Taoyuze Lv, Alexander Chen, Fengyu Xie, Chu Wu, Jeffrey Meng, Dongzhan Zhou, Bram Hoex, Zhicheng Zhong, Tong Xie

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文介绍了AtomWorld基准测试，用于评估大型语言模型在处理晶体学信息文件(CIFs)相关任务时的能力，特别是在材料科学领域的原子结构理解和空间推理能力。

### 背景

大型语言模型在文本推理方面表现出色，并开始发展空间理解能力。在材料科学等领域，对3D原子结构的深入理解是基础，但缺乏标准化的基准来系统评估LLMs在多样化原子结构上的核心推理能力。

### 目的

引入AtomWorld基准测试，用于评估LLMs基于晶体学信息文件(CIFs)的任务表现，CIFs是一种标准的结构表示格式。

### 方法

开发了AtomWorld基准测试，包含基于CIFs的任务，如结构编辑、CIF感知和属性引导建模。

### 主要发现

当前模型虽然建立了有希望的基线，但在结构理解和空间推理方面存在明显局限。这些模型在结构修改任务中频繁出错，甚至在基本的CIF格式理解方面也存在问题，可能导致后续分析和材料见解中的累积错误。

### 结论

通过定义这些标准化任务，AtomWorld为推进LLMs向稳健的原子级建模发展奠定了基础，这对于加速材料研究和自动化科学工作流程至关重要。

### 翻译

大型语言模型在文本推理方面表现出色，并开始发展空间理解能力，这引发了一个问题：这些能力是否可以结合起来用于复杂、特定领域的任务。这个问题在材料科学等领域至关重要，因为对3D原子结构的深入理解是基础。虽然初步研究已成功将LLMs应用于涉及纯晶体生成或坐标理解的任务，但缺乏一个标准化的基准来系统评估它们在多样化原子结构上的核心推理能力。为解决这一差距，我们引入了AtomWorld基准测试，用于评估LLMs基于晶体学信息文件(CIFs)的任务表现，CIFs是一种标准的结构表示格式。这些任务包括结构编辑、CIF感知和属性引导建模，揭示了一个关键局限：当前模型尽管建立了有希望的基线，但在结构理解和空间推理方面持续失败。我们的实验显示，这些模型在结构修改任务中频繁出错，甚至在基本的CIF格式理解方面也存在问题，可能导致后续分析和材料见解中的累积错误。通过定义这些标准化任务，AtomWorld为推进LLMs向稳健的原子级建模发展奠定了基础，这对加速材料研究和自动化科学工作流程至关重要。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决缺乏标准化基准来系统评估大型语言模型在处理晶体材料结构时的空间推理能力问题。这个问题在材料科学领域至关重要，因为深入理解3D原子结构是该领域的基础。当前模型在结构修改任务中经常出错，甚至在基本的CIF格式理解方面也存在问题，可能导致后续分析和材料洞察中的累积错误，影响材料研究的效率和准确性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者将LLMs处理CIF文件的能力分为三个阶段：运动技能（几何操作）、感知技能（识别模式）和认知技能（推理和创造力）。AtomWorld基准专门评估'运动技能'，据称是第一个检查晶体学这一基本能力的基准。作者借鉴了现有的CIF格式标准、传统晶体学软件（如Ovito和ASE）的工作方式，以及LLM4Mat-Bench等问答基准的设计思路，但专注于结构操作而非问答，形成了新的评估框架。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是AtomWorld作为一个可扩展的数据生成器，通过定义标准化的晶体结构操作任务来评估和训练LLMs的空间推理能力。整体流程包括：1)数据生成，创建'之前'和'之后'的CIF文件及动作提示；2)支持多种结构修改动作；3)评估流程，通过随机采样、初始化、结构操作、提示生成和结果比较来完成对LLM的评估。使用StructureMatcher比较生成结构与目标结构，计算成功率和最大距离等指标。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专门评估LLMs晶体学运动技能的AtomWorld基准；2)设计互补基准(PointWorld、CIF-Gen等)全面评估不同能力维度；3)系统评估前沿模型在原子结构操作上的表现；4)提供可扩展数据生成器支持训练。相比之前工作，AtomWorld专注于实际的结构操作而非生成或问答，不仅评估性能还提供训练框架，并探索了工具增强LLM的潜力，填补了LLMs在材料科学空间推理评估方面的空白。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AtomWorld基准首次系统评估了大型语言模型在晶体材料空间推理和结构操作方面的能力，揭示了当前模型在理解和修改原子结构时的关键局限性，为未来开发更强大的原子级建模工具奠定了基础。'}


### 论文摘要

Large Language Models (LLMs) excel at textual reasoning and are beginning to develop spatial understanding, prompting the question of whether these abilities can be combined for complex, domain-specific tasks. This question is essential in fields like materials science, where deep understanding of 3D atomic structures is fundamental. While initial studies have successfully applied LLMs to tasks involving pure crystal generation or coordinate understandings, a standardized benchmark to systematically evaluate their core reasoning abilities across diverse atomic structures has been notably absent. To address this gap, we introduce the AtomWorld benchmark to evaluate LLMs on tasks based in Crystallographic Information Files (CIFs), a standard structure representation format. These tasks, including structural editing, CIF perception, and property-guided modeling, reveal a critical limitation: current models, despite establishing promising baselines, consistently fail in structural understanding and spatial reasoning. Our experiments show that these models make frequent errors on structure modification tasks, and even in the basic CIF format understandings, potentially leading to cumulative errors in subsequent analysis and materials insights. By defining these standardized tasks, AtomWorld lays the ground for advancing LLMs toward robust atomic-scale modeling, crucial for accelerating materials research and automating scientific workflows.

---

## 44. Deforming Videos to Masks: Flow Matching for Referring Video Segmentation

**论文链接:** [http://arxiv.org/abs/2510.06139v1](http://arxiv.org/abs/2510.06139v1)

**作者:** Zanyi Wang, Dengyang Jiang, Liuzhuozheng Li, Sizhe Dang, Chengzu Li, Harry Yang, Guang Dai, Mengmeng Wang, Jingdong Wang

**发布时间:** 2025-10-07

### GPT解析

### 总结

FlowRVS是一种将Referring Video Object Segmentation重新概念化为条件连续流问题的新框架，通过学习从视频整体表示到目标掩码的直接语言引导变形，实现了细粒度像素控制、文本-视频语义对齐和时间一致性，在所有主要RVOS基准测试中取得了新的最先进结果。

### 背景

Referring Video Object Segmentation需要根据自然语言描述在视频中分割特定对象，核心挑战是将抽象的语言概念锚定到特定像素并在视频动态中持续分割。先前方法采用'定位然后分割'的级联设计，简化语义为粗略几何提示，难以保持时间一致性。

### 目的

克服现有方法的根本局限性，提出一种新的框架重新概念化RVOS问题，实现更精确的分割和时间一致性。

### 方法

提出FlowRVS框架，将RVOS重新概念化为条件连续流问题，利用预训练T2V模型的固有优势，通过学习从视频整体表示到目标掩码的直接语言引导变形，采用单阶段生成方法而非传统从噪声生成掩码或直接预测掩码。

### 主要发现

在所有主要RVOS基准测试中取得新最先进结果，在MeViS上达到51.1的J&F分数（比之前SOTA高1.6），在零样本Ref-DAVIS17上达到73.3（高2.7）。

### 结论

证明了将视频理解任务建模为连续变形过程具有巨大潜力，FlowRVS框架有效克服了传统方法的局限性。

### 翻译

参考视频对象分割要求根据自然语言描述在视频中分割特定对象。RVOS的核心挑战是将抽象的语言概念锚定到特定的像素集合，并在视频的复杂动态中持续分割它们。面对这一困难，先前的工作通常将任务分解为一个实用的'定位然后分割'流水线。然而，这种级联设计通过将语义简化为粗略的几何提示（如点）创造了信息瓶颈，并且由于分割过程通常与初始语言解耦而难以保持时间一致性。为了克服这些根本限制，我们提出了FlowRVS，一个将RVOS重新概念化为条件连续流问题的新颖框架。这使我们能够利用预训练T2V模型的固有优势、细粒度像素控制、文本-视频语义对齐和时间一致性。我们不是通过从噪声生成掩码或直接预测掩码，而是通过学习从视频的整体表示到其目标掩码的直接语言引导变形来重新表述任务。我们单阶段的生成方法在所有主要的RVOS基准测试中取得了新的最先进结果。具体来说，在MeViS上达到51.1的J&F（比之前SOTA高1.6），在零样本Ref-DAVIS17上达到73.3（高2.7），证明了将视频理解任务建模为连续变形过程的巨大潜力。


### 论文摘要

Referring Video Object Segmentation (RVOS) requires segmenting specific objects in a video guided by a natural language description. The core challenge of RVOS is to anchor abstract linguistic concepts onto a specific set of pixels and continuously segment them through the complex dynamics of a video. Faced with this difficulty, prior work has often decomposed the task into a pragmatic `locate-then-segment' pipeline. However, this cascaded design creates an information bottleneck by simplifying semantics into coarse geometric prompts (e.g, point), and struggles to maintain temporal consistency as the segmenting process is often decoupled from the initial language grounding. To overcome these fundamental limitations, we propose FlowRVS, a novel framework that reconceptualizes RVOS as a conditional continuous flow problem. This allows us to harness the inherent strengths of pretrained T2V models, fine-grained pixel control, text-video semantic alignment, and temporal coherence. Instead of conventional generating from noise to mask or directly predicting mask, we reformulate the task by learning a direct, language-guided deformation from a video's holistic representation to its target mask. Our one-stage, generative approach achieves new state-of-the-art results across all major RVOS benchmarks. Specifically, achieving a $\mathcal{J}\&\mathcal{F}$ of 51.1 in MeViS (+1.6 over prior SOTA) and 73.3 in the zero shot Ref-DAVIS17 (+2.7), demonstrating the significant potential of modeling video understanding tasks as continuous deformation processes.

---

## 45. When Thinking Drifts: Evidential Grounding for Robust Video Reasoning

**论文链接:** [http://arxiv.org/abs/2510.06077v1](http://arxiv.org/abs/2510.06077v1)

**作者:** Mi Luo, Zihui Xue, Alex Dimakis, Kristen Grauman

**发布时间:** 2025-10-07

**备注:** Accepted by NeurIPS 2025, Project page:  https://vision.cs.utexas.edu/projects/video-ver/

### GPT解析

### 总结

该论文研究了视频推理中思维链(CoT)机制的应用问题，发现CoT在视频推理中会导致性能下降，产生'视觉思维漂移'现象，并提出了视觉证据奖励(VER)框架来解决这个问题。

### 背景

视频推理是让机器通过多步逻辑从动态视觉内容中推断信息的能力，对高级AI至关重要。虽然思维链(CoT)机制已增强基于文本任务的推理，但在视频理解中的应用仍探索不足。

### 目的

分析CoT在视频推理中的问题，并提出解决方案以改善视频推理性能。

### 方法

引入视觉证据奖励(VER)，一种新型强化学习框架，明确奖励可验证基于视觉证据的推理轨迹生成。

### 主要发现

CoT在视频推理中通常会降低性能，产生冗长但误导性的内部独白，导致产生幻觉的视觉细节和覆盖正确的直觉，这种现象称为'视觉思维漂移'。CoT轨迹经常与实际视觉证据偏离，反而放大内部偏见或语言先验。

### 结论

Video-VER在10个多样化的视频理解基准上取得了顶尖性能。这项工作揭示了以视频为中心推理的独特挑战，并鼓励开发能将其推断牢固建立在视觉证据上的AI。

### 翻译

视频推理是让机器通过多步逻辑从动态视觉内容中推断信息的任务，对高级AI至关重要。虽然思维链(CoT)机制已增强基于文本任务的推理，但在视频理解中的应用仍探索不足。本文进行了系统分析，揭示CoT在视频推理中通常会降低性能，产生冗长但误导性的内部独白，导致产生幻觉的视觉细节和覆盖正确的直觉 - 我们将这种现象称为'视觉思维漂移'。我们通过贝叶斯视角解释这种漂移，认为CoT轨迹经常与实际视觉证据偏离，反而放大内部偏见或语言先验，导致模型讲故事而非进行基于证据的推理。为解决这一问题，我们引入视觉证据奖励(VER)，一种新型强化学习框架，明确奖励可验证基于视觉证据的推理轨迹生成。在10个多样化的视频理解基准上的全面评估表明，我们的Video-VER始终取得顶尖性能。我们的工作揭示了以视频为中心推理的独特挑战，并鼓励开发能将其推断牢固建立在视觉证据上的AI - 对于不仅'在回答前思考'，而且'在思考时看见'的大型多模态模型。


### 论文摘要

Video reasoning, the task of enabling machines to infer from dynamic visual content through multi-step logic, is crucial for advanced AI. While the Chain-of-Thought (CoT) mechanism has enhanced reasoning in text-based tasks, its application to video understanding remains underexplored. This paper presents a systematic analysis revealing that CoT often degrades performance in video reasoning, generating verbose but misleading internal monologues, and leading to hallucinated visual details and overridden correct intuitions - a phenomenon we term "visual thinking drift". We explain this drift through a Bayesian lens, positing that CoT traces often diverge from actual visual evidence, instead amplifying internal biases or language priors, causing models to storytell rather than engage in grounded reasoning. To counteract this, we introduce Visual Evidence Reward (VER), a novel reinforcement learning framework that explicitly rewards the generation of reasoning traces that are verifiably grounded in visual evidence. Comprehensive evaluation across 10 diverse video understanding benchmarks demonstrates that our Video-VER consistently achieves top performance. Our work sheds light on the distinct challenges of video-centric reasoning and encourages the development of AI that robustly grounds its inferences in visual evidence - for large multimodal models that not only "think before answering", but also "see while thinking".

---

## 46. VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization

**论文链接:** [http://arxiv.org/abs/2510.06040v1](http://arxiv.org/abs/2510.06040v1)

**作者:** Xinye Cao, Hongcan Guo, Jiawen Qian, Guoshun Nan, Chao Wang, Yuqi Pan, Tianhao Hou, Xiaojuan Wang, Yutong Gao

**发布时间:** 2025-10-07

**备注:** Accepted by ICCV 2025

### GPT解析

### 总结

本文提出了一种名为VideoMiner的方法，用于解决长视频理解中的关键挑战，并引入了T-GRPO强化学习技术来精确定位关键帧。

### 背景

理解长时间视频对于以人为中心的AI应用很重要，但使用多模态大语言模型进行端到端视频理解时，均匀采样视频帧会导致模型被大量无关信息淹没。现有的分层关键帧提取方法虽提高了准确性，但仍面临两个关键挑战。

### 目的

解决两个关键挑战：1) 如何减轻长视频中大量冗余信息的干扰；2) 如何让模型动态适应复杂的分层结构，同时准确识别关键帧。

### 方法

提出VideoMiner方法，迭代地分割、标注和聚类长视频，形成分层树结构，从长视频到事件再到帧，同时保持时间连贯性。引入T-GRPO（基于树的组相对策略优化）强化学习方法，专为树结构设计，在事件级别整合时空信息，同时受问题引导。

### 主要发现

在所有长视频理解任务中取得了优越的性能；T-GRPO意外地激励模型自发生成推理链；设计的树生长素动态调整扩展深度，获得准确性和效率提升。

### 结论

VideoMiner和T-GRPO方法有效解决了长视频理解中的关键挑战，相关代码已在GitHub公开。

### 翻译

理解长时间视频与多模态大语言模型(MM-LLMs)丰富了以人为中心的AI应用前景。然而，对于使用LLM进行端到端视频理解，随着视频长度增加，均匀采样视频帧会导致LLM被大量无关信息淹没。现有的分层关键帧提取方法提高了视频理解准确性，但仍面临两个关键挑战：1) 如何减轻长视频中大量冗余信息的干扰？2) 如何让模型动态适应复杂的分层结构，同时准确识别关键帧？为解决这些问题，我们提出了VideoMiner，它迭代地分割、标注和聚类长视频，形成分层树结构。所提出的VideoMiner从长视频到事件再到帧，同时保持时间连贯性，有效解决了第一个挑战。为了精确定位关键帧，我们引入了T-GRPO，一种基于树的组相对策略优化强化学习方法，该方法专为树结构设计，在事件级别整合时空信息，同时受问题引导，从而解决了第二个挑战。我们在所有长视频理解任务中取得了优越的性能，并发现了一些有趣的见解。我们提出的T-GRPO意外地激励模型自发生成推理链。此外，设计的树生长生长素动态调整扩展深度，获得了准确性和效率提升。代码已在https://github.com/caoxinye/VideoMiner公开。


### 论文摘要

Understanding hour-long videos with multi-modal large language models (MM-LLMs) enriches the landscape of human-centered AI applications. However, for end-to-end video understanding with LLMs, uniformly sampling video frames results in LLMs being overwhelmed by a vast amount of irrelevant information as video length increases. Existing hierarchical key frame extraction methods improve the accuracy of video understanding but still face two critical challenges. 1) How can the interference of extensive redundant information in long videos be mitigated? 2) How can a model dynamically adapt to complex hierarchical structures while accurately identifying key frames? To address these issues, we propose VideoMiner, which iteratively segments, captions, and clusters long videos, forming a hierarchical tree structure. The proposed VideoMiner progresses from long videos to events to frames while preserving temporal coherence, effectively addressing the first challenge. To precisely locate key frames, we introduce T-GRPO, a tree-based group relative policy optimization in reinforcement learning method that guides the exploration of the VideoMiner. The proposed T-GRPO is specifically designed for tree structures, integrating spatiotemporal information at the event level while being guided by the question, thus solving the second challenge. We achieve superior performance in all long-video understanding tasks and uncover several interesting insights. Our proposed T-GRPO surprisingly incentivizes the model to spontaneously generate a reasoning chain. Additionally, the designed tree growth auxin dynamically adjusts the expansion depth, obtaining accuracy and efficiency gains. The code is publicly available at https://github.com/caoxinye/VideoMiner.

---

## 47. 论文ID: 2510.05836v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.05836v1.json'

---

## 48. StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars

**论文链接:** [http://arxiv.org/abs/2510.06200v1](http://arxiv.org/abs/2510.06200v1)

**作者:** Weijian Li, Hong-Yu Chen, Qinjie Lin, Nabeel Rehemtulla, Ved G. Shah, Dennis Wu, Adam A. Miller, Han Liu

**发布时间:** 2025-10-07

### GPT解析

### 总结

研究介绍了StarEmbed基准，用于评估时间序列基础模型在天文数据上的表现。结果表明，即使是在非天文数据上训练的模型，也能在天文任务上表现出色，这为天文学领域采用通用基础模型而非特定任务模型提供了依据。

### 背景

时间序列基础模型(TSFMs)正被广泛采用作为通用的时间序列表示学习器，但其训练语料库不包括天文时间序列数据。恒星观测产生拍字节时间序列，具有独特挑战，包括不规则采样和异方差性。

### 目的

介绍StarEmbed，这是第一个用于对恒星时间序列观测('光曲线')上的最先进TSFMs进行严格标准化评估的公共基准。

### 方法

在三个科学动机的下游任务上进行基准测试：无监督聚类、监督分类和分布外源检测。StarEmbed整合了专家验证的标签目录与Zwicky Transient Facility的多变量光曲线，产生约40k个手工标记的光曲线，分布在七个天体物理学类别中。评估三个TSFMs(MOIRAI、Chronos、Chronos-Bolt)和一个领域特定转换器(Astromer)的零样本表示能力，与天文学文献中长期存在的基线手工制作的特征提取进行比较。

### 主要发现

这些TSFMs，特别是Chronos模型(在完全不同于天文观测的数据上训练)，在某些任务上可以超越既有的天文学特定基线，并有效推广到全新数据。特别是，TSFMs在分布外源检测基准上提供了最先进的性能。

### 结论

通过第一个在天文时间序列数据上的TSFMs基准测试，研究测试了它们的泛化极限，并推动了时域天文学从使用任务特定的全监督管道转向采用通用基础模型表示来分析即将到来的天文台拍字节数据集的范式转变。

### 翻译

时间序列基础模型(TSFMs)正日益被采用作为高度通用的通用时间序列表示学习器。尽管它们的训练语料库庞大，但它们不包括天文时间序列数据。恒星观测产生拍字节时间序列，具有独特挑战，包括不规则采样和异方差性。我们介绍了StarEmbed，这是第一个用于对恒星时间序列观测('光曲线')上的最先进TSFMs进行严格标准化评估的公共基准。我们在三个科学动机的下游任务上进行基准测试：无监督聚类、监督分类和分布外源检测。StarEmbed整合了专家验证的标签目录与Zwicky Transient Facility的多变量光曲线，产生约40k个手工标记的光曲线，分布在七个天体物理学类别中。我们评估了三个TSFMs(MOIRAI、Chronos、Chronos-Bolt)和一个领域特定转换器(Astromer)的零样本表示能力，与天文学文献中长期存在的基线手工制作的特征提取进行比较。我们的结果表明，这些TSFMs，特别是Chronos模型(在完全不同于天文观测的数据上训练)，可以在某些任务上超越既有的天文学特定基线，并有效推广到全新数据。特别是，TSFMs在我们的分布外源检测基准上提供了最先进的性能。通过第一个在天文时间序列数据上的TSFMs基准测试，我们测试了它们的泛化极限，并推动了时域天文学从使用任务特定的全监督管道转向采用通用基础模型表示来分析即将到来的天文台拍字节数据集的范式转变。


### 论文摘要

Time series foundation models (TSFMs) are increasingly being adopted as highly-capable general-purpose time series representation learners. Although their training corpora are vast, they exclude astronomical time series data. Observations of stars produce peta-scale time series with unique challenges including irregular sampling and heteroskedasticity. We introduce StarEmbed, the first public benchmark for rigorous and standardized evaluation of state-of-the-art TSFMs on stellar time series observations (``light curves''). We benchmark on three scientifically-motivated downstream tasks: unsupervised clustering, supervised classification, and out-of-distribution source detection. StarEmbed integrates a catalog of expert-vetted labels with multi-variate light curves from the Zwicky Transient Facility, yielding ~40k hand-labeled light curves spread across seven astrophysical classes. We evaluate the zero-shot representation capabilities of three TSFMs (MOIRAI, Chronos, Chronos-Bolt) and a domain-specific transformer (Astromer) against handcrafted feature extraction, the long-standing baseline in the astrophysics literature. Our results demonstrate that these TSFMs, especially the Chronos models, which are trained on data completely unlike the astronomical observations, can outperform established astrophysics-specific baselines in some tasks and effectively generalize to entirely new data. In particular, TSFMs deliver state-of-the-art performance on our out-of-distribution source detection benchmark. With the first benchmark of TSFMs on astronomical time series data, we test the limits of their generalization and motivate a paradigm shift in time-domain astronomy from using task-specific, fully supervised pipelines toward adopting generic foundation model representations for the analysis of peta-scale datasets from forthcoming observatories.

---

## 49. TabPFN-Wide: Continued Pre-Training for Extreme Feature Counts

**论文链接:** [http://arxiv.org/abs/2510.06162v1](http://arxiv.org/abs/2510.06162v1)

**作者:** Christopher Kolberg, Katharina Eggensperger, Nico Pfeifer

**发布时间:** 2025-10-07

### GPT解析

### 总结

该研究提出了一种新的机器学习策略，通过在合成数据上持续预训练来扩展现有模型，以处理生物医学中的高维数据，同时保持可解释性。

### 背景

机器学习在生物医学中用于揭示分子测量与病理学关系的新见解，但该领域数据通常只有少量观测值却有数千个潜在噪声特征，对传统机器学习方法构成挑战。

### 目的

开发一种能够处理高维生物医学数据的大量特征的机器学习方法，同时保持特征重要性的可分析能力。

### 方法

通过在从定制先验采样的合成数据上进行持续预训练来扩展现有基础模型，创建名为TabPFN-Wide的新模型。

### 主要发现

TabPFN-Wide模型匹配或超过其基础模型的性能，提高对噪声的鲁棒性，可无缝扩展到50,000多个特征，同时保持固有的可解释性。

### 结论

先验信息适应适合增强基础模型处理高维数据的能力，在生物医学应用中具有重要价值。

### 翻译

从分子测量与病理学之间的关系揭示新的见解，仍然是机器学习在生物医学中非常有影响力的应用。该领域的数据通常只包含少量观测值但有数千个潜在噪声特征，对传统机器学习方法构成挑战。虽然先验数据拟合网络作为表格数据的基础模型出现，但它们目前不适合处理大量特征。虽然特征减少使它们能够应用，但它妨碍了特征重要性分析。我们提出了一种策略，通过在从定制先验采样的合成数据上进行持续预训练来扩展现有模型。 resulting model, TabPFN-Wide, matches or exceeds its base model's performance while exhibiting improved robustness to noise. It seamlessly scales beyond 50,000 features, regardless of noise levels, while maintaining inherent interpretability, which is critical for biomedical applications. Our results show that prior-informed adaptation is suitable to enhance the capability of foundation models for high-dimensional data. On real-world biomedical datasets many of the most relevant features identified by the model overlap with previous biological findings, while others propose potential starting points for future studies.


### 论文摘要

Revealing novel insights from the relationship between molecular measurements and pathology remains a very impactful application of machine learning in biomedicine. Data in this domain typically contain only a few observations but thousands of potentially noisy features, posing challenges for conventional machine learning approaches. While prior-data fitted networks emerge as foundation models for tabular data, they are currently not suited to handle large feature counts (>500). Although feature reduction enables their application, it hinders feature importance analysis. We propose a strategy that extends existing models through continued pre-training on synthetic data sampled from a customized prior. The resulting model, TabPFN-Wide, matches or exceeds its base model's performance while exhibiting improved robustness to noise. It seamlessly scales beyond 50,000 features, regardless of noise levels, while maintaining inherent interpretability, which is critical for biomedical applications. Our results show that prior-informed adaptation is suitable to enhance the capability of foundation models for high-dimensional data. On real-world biomedical datasets many of the most relevant features identified by the model overlap with previous biological findings, while others propose potential starting points for future studies.

---

## 50. Discrete Diffusion Models with MLLMs for Unified Medical Multimodal Generation

**论文链接:** [http://arxiv.org/abs/2510.06131v1](http://arxiv.org/abs/2510.06131v1)

**作者:** Jiawei Mao, Yuhan Wang, Lifeng Chen, Can Zhao, Yucheng Tang, Dong Yang, Liangqiong Qu, Daguang Xu, Yuyin Zhou

**发布时间:** 2025-10-07

**备注:** 16 pages,6 figures

### GPT解析

### 总结

MeDiM是一个创新的医学离散扩散模型，能够学习跨模态的共享分布，无需模态特定组件。该模型统一了多种生成任务，通过离散扩散框架连接视觉和语言表示，并利用多模态大语言模型作为扩散骨干。

### 背景

最近的生成式医学模型进展受到模态特定场景的限制，这些限制阻碍了从成像、病理学和临床笔记中整合互补证据。这种碎片化限制了它们发展为能够学习和推理整个生物医学数据谱系的基础模型。

### 目的

提出MeDiM，第一个医学离散扩散模型，学习跨模态的共享分布，不需要模态特定组件，实现多种医学数据模态的统一生成和处理。

### 方法

MeDiM基于离散扩散框架构建，通过共享概率空间连接视觉和语言表示。采用多模态大语言模型作为扩散骨干，利用其先验知识和跨模态推理能力。关键设计包括：移除因果注意力掩码以实现双向上下文，以及注入连续时间步嵌入以实现扩散感知。

### 主要发现

实验证明MeDiM能够生成高保真度的医学内容（在MIMIC-CXR上的FID为16.60，在PathGen上的FID为24.19）和准确的报告（METEOR分数为0.2650和0.2580）。联合生成的图像-报告对显著提高了下游性能（BLEU-1提高6.43%，BLEU-2提高18.57%，BLEU-3提高31.58%，METEOR提高4.80%）。

### 结论

MeDiM支持连贯且临床上有依据的多模态输出，代表了医学生成模型的重要进步，能够统一处理多种医学数据模态，实现跨模态的学习和推理。

### 翻译

最近的生成式医学模型进展受到模态特定场景的限制，这些限制阻碍了从成像、病理学和临床笔记中整合互补证据。这种碎片化限制了它们发展为能够学习和推理整个生物医学数据谱系的基础模型。我们提出了MeDiM，这是第一个医学离散扩散模型，它学习跨模态的共享分布，不需要模态特定组件。MeDiM统一了多种生成任务：图像和文本之间的转换，以及跨领域响应提示联合生成图像-报告对。基于离散扩散框架构建，MeDiM通过共享概率空间连接视觉和语言表示。为了实现统一和灵活的医学生成，我们采用多模态大语言模型作为扩散骨干，利用其先验知识和跨模态推理能力。引入了两个关键设计：(1)移除因果注意力掩码以实现双向上下文，(2)注入连续时间步嵌入以实现扩散感知。实验证明了高保真度的医学生成（在MIMIC-CXR上的FID为16.60，在PathGen上的FID为24.19）和准确的报告生成（METEOR分数为0.2650和0.2580）。联合生成的图像-报告对进一步提高了下游性能（BLEU-1提高6.43%，BLEU-2提高18.57%，BLEU-3提高31.58%，METEOR提高4.80%），表明MeDiM支持连贯且临床上有依据的多模态输出。


### 论文摘要

Recent advances in generative medical models are constrained by modality-specific scenarios that hinder the integration of complementary evidence from imaging, pathology, and clinical notes. This fragmentation limits their evolution into foundation models that can learn and reason across the full spectrum of biomedical data. We propose MeDiM, the first medical discrete diffusion model that learns shared distributions across modalities without modality-specific components. MeDiM unifies multiple generative tasks: translating between images and text, and jointly producing image-report pairs across domains in response to prompts. Built on a discrete diffusion framework, MeDiM bridges vision and language representations through a shared probabilistic space. To enable unified and flexible medical generation, we employ a multimodal large language model (MLLM) as the diffusion backbone, leveraging its prior knowledge and cross-modal reasoning. Two key designs are introduced: (1) removing the causal attention mask for bidirectional context, and (2) injecting continuous timestep embeddings for diffusion awareness. Experiments demonstrate high-fidelity medical generation (FID 16.60 on MIMIC-CXR and FID 24.19 on PathGen) and accurate report generation (METEOR 0.2650 and 0.2580). Jointly generated image-report pairs further enhance downstream performance (plus6.43 percent BLEU-1, plus18.57 percent BLEU-2, plus31.58 percent BLEU-3, plus4.80 percent METEOR), showing that MeDiM supports coherent and clinically grounded multimodal outputs.

---

## 51. Diffusion Models for Low-Light Image Enhancement: A Multi-Perspective Taxonomy and Performance Analysis

**论文链接:** [http://arxiv.org/abs/2510.05976v1](http://arxiv.org/abs/2510.05976v1)

**作者:** Eashan Adhikarla, Yixin Liu, Brian D. Davison

**发布时间:** 2025-10-07

### GPT解析

### 总结

该调查论文提供了对扩散模型用于低光图像增强(LLIE)的最新批判性分析，包括与传统方法的对比评估、实际部署挑战和新兴范式的前景展望。

### 背景

低光图像增强对安全关键应用如监控、自主导航和医疗成像至关重要，其中可见度下降会影响下游任务性能。扩散模型因其能够通过迭代去噪建模复杂图像分布而成为LLIE有前景的生成范式。

### 目的

提供对扩散模型用于LLIE的最新批判性分析，对比评估扩散模型与生成对抗网络和基于Transformer的最先进方法，审查实际部署挑战，并展望基础模型等新兴范式的作用。

### 方法

提出一个包含六类(内在分解、频谱与潜在、加速、引导、多模态和自主)的多角度分类法，基于模型机制和条件信号的混合视图。评估定性的失败模式、基准不一致性以及可解释性、泛化能力和推理效率之间的权衡，并讨论实际部署约束和伦理考量。

### 主要发现

扩散模型能够通过迭代去噪建模复杂图像分布；实际部署存在内存、能源使用等约束和伦理考量；基础模型具有潜力。

### 结论

该调查旨在通过突出趋势和揭示开放研究问题来指导下一代基于扩散模型的LLIE研究，包括新型条件设置、实时适应和基础模型的潜力。

### 翻译

低光图像增强(LLIE)对于监控、自主导航和医疗成像等安全关键应用至关重要，因为可见度下降会损害下游任务性能。最近，扩散模型已成为LLIE有前景的生成范式，因为它们能够通过迭代去噪建模复杂的图像分布。本调查提供了对扩散模型用于LLIE的最新批判性分析，特别强调了与生成对抗网络和基于Transformer的最先进方法的深入性能对比评估，对实际部署挑战的彻底检查，以及对基础模型等新兴范式作用的展望。我们提出了一个包含六类的多角度分类法：内在分解、频谱与潜在、加速、引导、多模态和自主；这些类别映射了跨越物理先验、条件方案和计算效率的增强方法。我们的分类法基于模型机制和条件信号的混合视图。我们评估了定性的失败模式、基准不一致性以及可解释性、泛化能力和推理效率之间的权衡。我们还讨论了实际部署约束(如内存、能源使用)和伦理考虑。本调查旨在通过突出趋势和揭示开放研究问题来指导下一代基于扩散模型的LLIE研究，包括新型条件设置、实时适应和基础模型的潜力。


### 论文摘要

Low-light image enhancement (LLIE) is vital for safety-critical applications such as surveillance, autonomous navigation, and medical imaging, where visibility degradation can impair downstream task performance. Recently, diffusion models have emerged as a promising generative paradigm for LLIE due to their capacity to model complex image distributions via iterative denoising. This survey provides an up-to-date critical analysis of diffusion models for LLIE, distinctively featuring an in-depth comparative performance evaluation against Generative Adversarial Network and Transformer-based state-of-the-art methods, a thorough examination of practical deployment challenges, and a forward-looking perspective on the role of emerging paradigms like foundation models. We propose a multi-perspective taxonomy encompassing six categories: Intrinsic Decomposition, Spectral & Latent, Accelerated, Guided, Multimodal, and Autonomous; that map enhancement methods across physical priors, conditioning schemes, and computational efficiency. Our taxonomy is grounded in a hybrid view of both the model mechanism and the conditioning signals. We evaluate qualitative failure modes, benchmark inconsistencies, and trade-offs between interpretability, generalization, and inference efficiency. We also discuss real-world deployment constraints (e.g., memory, energy use) and ethical considerations. This survey aims to guide the next generation of diffusion-based LLIE research by highlighting trends and surfacing open research questions, including novel conditioning, real-time adaptation, and the potential of foundation models.

---

## 52. StereoSync: Spatially-Aware Stereo Audio Generation from Video

**论文链接:** [http://arxiv.org/abs/2510.05828v1](http://arxiv.org/abs/2510.05828v1)

**作者:** Christian Marinoni, Riccardo Fosco Gramaccioni, Kazuki Shimada, Takashi Shibuya, Yuki Mitsufuji, Danilo Comminiello

**发布时间:** 2025-10-07

**备注:** Accepted at IJCNN 2025

### GPT解析

### 总结

StereoSync是一个新颖高效的模型，用于生成与参考视频时间同步且空间对齐的音频。它利用预训练基础模型实现效率，同时保持高质量的合成，通过提取空间线索并使用交叉注意力条件，能够生成动态适应视频场景的立体声音频。

### 背景

音频生成在近年来已被广泛研究，但与视频对齐的音频生成仍然是一个相对未被探索的前沿领域。

### 目的

解决视频对齐音频生成的研究空白，开发能够生成与视频时间同步且空间对齐的音频模型。

### 方法

StereoSync从深度图和边界框中提取空间线索，使用这些线索作为基于扩散的音频生成模型中的交叉注意力条件，实现立体声音频的生成，使其能够动态适应视频场景的空间结构和运动。

### 主要发现

StereoSync在Walking The Maps数据集上实现了时间和空间对齐，在视频到音频生成方面取得了最先进的结果，产生了更加沉浸式和真实的音频体验。

### 结论

StereoSync代表了视频对齐音频生成领域的重要进步，通过引入空间感知能力，超越了仅关注时间同步的现有方法。

### 翻译

尽管音频生成在近年来已被广泛研究，但与视频对齐的音频生成仍然是一个相对未被探索的前沿领域。为解决这一空白，我们引入了StereoSync，这是一个新颖且高效的模型，旨在生成与参考视频时间同步且与其视觉空间上下文对齐的音频。此外，StereoSync通过利用预训练基础模型实现效率，减少了大量训练的需求，同时保持高质量的合成。与主要关注时间同步的现有方法不同，StereoSync通过将空间感知纳入视频对齐音频生成，引入了重大进展。实际上，给定输入视频，我们的方法从深度图和边界框中提取空间线索，将它们用作基于扩散的音频生成模型中的交叉注意力条件。这种方法使StereoSync能够超越简单的同步，产生动态适应视频场景空间结构的立体声音频。我们在Walking The Maps数据集上评估了StereoSync，这是一个包含视频游戏中角色穿过多样化环境行走视频的精选数据集。实验结果证明了StereoSync实现时间和空间对齐的能力，推动了视频到音频生成的最先进水平，并产生了显著更加沉浸式和真实的音频体验。


### 论文摘要

Although audio generation has been widely studied over recent years, video-aligned audio generation still remains a relatively unexplored frontier. To address this gap, we introduce StereoSync, a novel and efficient model designed to generate audio that is both temporally synchronized with a reference video and spatially aligned with its visual context. Moreover, StereoSync also achieves efficiency by leveraging pretrained foundation models, reducing the need for extensive training while maintaining high-quality synthesis. Unlike existing methods that primarily focus on temporal synchronization, StereoSync introduces a significant advancement by incorporating spatial awareness into video-aligned audio generation. Indeed, given an input video, our approach extracts spatial cues from depth maps and bounding boxes, using them as cross-attention conditioning in a diffusion-based audio generation model. Such an approach allows StereoSync to go beyond simple synchronization, producing stereo audio that dynamically adapts to the spatial structure and movement of a video scene. We evaluate StereoSync on Walking The Maps, a curated dataset comprising videos from video games that feature animated characters walking through diverse environments. Experimental results demonstrate the ability of StereoSync to achieve both temporal and spatial alignment, advancing the state of the art in video-to-audio generation and resulting in a significantly more immersive and realistic audio experience.

---

## 53. VCoT-Grasp: Grasp Foundation Models with Visual Chain-of-Thought Reasoning for Language-driven Grasp Generation

**论文链接:** [http://arxiv.org/abs/2510.05827v1](http://arxiv.org/abs/2510.05827v1)

**作者:** Haoran Zhang, Shuanghao Bai, Wanqi Zhou, Yuedi Zhang, Qi Zhang, Pengxiang Ding, Cheng Chi, Donglin Wang, Badong Chen

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出了一种名为VCoT-Grasp的端到端抓取基础模型，通过视觉思维链推理增强视觉理解能力，解决了现有语言驱动抓取方法推理能力不足和泛化能力差的问题。

### 背景

机器人抓取是机器人操作中最基础的任务之一，而抓取检测/生成一直是广泛研究的主题。最近，语言驱动的抓取生成因其实用的交互能力成为一个有前景的研究方向。

### 目的

为了在复杂环境中保持强大的推理能力和泛化能力，作者提出了一种新的抓取基础模型，解决现有方法缺乏推理和泛化能力、依赖复杂模块化流程，以及当前抓取基础模型过度强调对话和对象语义导致性能低下且仅限于单目标抓取的问题。

### 方法

作者提出了VCoT-Grasp，一种端到端的抓取基础模型，结合视觉思维链推理来增强视觉理解。该模型采用多轮处理范式，动态关注视觉输入，同时提供可解释的推理轨迹。此外，作者还创建并引入了一个大规模数据集VCoT-GraspSet，包含167K合成图像和1.36M+抓取，以及400+真实世界图像和1.2K+抓取，并标注了中间边界框。

### 主要发现

在VCoT-GraspSet和真实机器人上的广泛实验表明，该方法显著提高了抓取成功率，并能有效泛化到未见过的对象、背景和干扰物。

### 结论

VCoT-Grasp通过视觉思维链推理和多轮处理范式，解决了现有语言驱动抓取方法的局限性，在复杂环境中实现了更强的推理能力和泛化能力。

### 翻译

机器人抓取是机器人操作中最基础的任务之一，而抓取检测/生成长期以来一直是广泛研究的主题。最近，语言驱动的抓取生成因其实用的交互能力成为一个有前景的研究方向。然而，大多数现有方法要么缺乏足够的推理和泛化能力，要么依赖于复杂的模块化流程。此外，当前的抓取基础模型往往过度强调对话和对象语义，导致性能低下且仅限于单目标抓取。为了在复杂环境中保持强大的推理能力和泛化能力，我们提出了VCoT-Grasp，一种结合视觉思维链推理来增强视觉理解的端到端抓取基础模型。VCoT-Grasp采用多轮处理范式，动态关注视觉输入，同时提供可解释的推理轨迹。在训练方面，我们改进并引入了一个大规模数据集VCoT-GraspSet，包含167K合成图像和超过1.36M抓取，以及400+真实世界图像和超过1.2K抓取，并标注了中间边界框。在VCoT-GraspSet和真实机器人上的广泛实验表明，我们的方法显著提高了抓取成功率，并能有效泛化到未见过的对象、背景和干扰物。更多详情请访问https://zhanghr2001.github.io/VCoT-Grasp.github.io。


### 论文摘要

Robotic grasping is one of the most fundamental tasks in robotic manipulation, and grasp detection/generation has long been the subject of extensive research. Recently, language-driven grasp generation has emerged as a promising direction due to its practical interaction capabilities. However, most existing approaches either lack sufficient reasoning and generalization capabilities or depend on complex modular pipelines. Moreover, current grasp foundation models tend to overemphasize dialog and object semantics, resulting in inferior performance and restriction to single-object grasping. To maintain strong reasoning ability and generalization in cluttered environments, we propose VCoT-Grasp, an end-to-end grasp foundation model that incorporates visual chain-of-thought reasoning to enhance visual understanding for grasp generation. VCoT-Grasp adopts a multi-turn processing paradigm that dynamically focuses on visual inputs while providing interpretable reasoning traces. For training, we refine and introduce a large-scale dataset, VCoT-GraspSet, comprising 167K synthetic images with over 1.36M grasps, as well as 400+ real-world images with more than 1.2K grasps, annotated with intermediate bounding boxes. Extensive experiments on both VCoT-GraspSet and real robot demonstrate that our method significantly improves grasp success rates and generalizes effectively to unseen objects, backgrounds, and distractors. More details can be found at https://zhanghr2001.github.io/VCoT-Grasp.github.io.

---

## 54. Transcribing Rhythmic Patterns of the Guitar Track in Polyphonic Music

**论文链接:** [http://arxiv.org/abs/2510.05756v1](http://arxiv.org/abs/2510.05756v1)

**作者:** Aleksandr Lukoianov, Anssi Klapuri

**发布时间:** 2025-10-07

**备注:** Accepted to WASPAA 2025

### GPT解析

### 总结

本研究提出了一种三步框架，用于转录多声道音乐中吉他音轨的节奏模式，实现了高精度的节奏模式识别和人类可读的表示。

### 背景

和弦转录在过去几十年受到广泛关注，但节奏模式的转录和编码研究相对较少。这一主题对节奏吉他等乐器尤为重要，因为这些乐器通常通过扫弦演奏重复和变化的节奏模式，且往往难以客观定义单一'正确'的节奏模式。

### 目的

创建一个具有明确定义真实标签的数据集，让专业音乐家转录410首流行歌曲中的节奏模式，并录制遵循这些转录的吉他音轨封面版本。

### 方法

提出一个三步框架：1)执行近似音源分离提取吉他部分；2)使用预训练的基础模型(MERT)检测分离的吉他音频中的单个扫弦；3)进行模式解码，将转录的吉他扫弦序列用专家策划的词汇中的模式表示。

### 主要发现

可以以相当高的精度转录多声道音乐中吉他音轨的节奏模式，产生的表示是人类可读的，包括自动检测的小节线和拍号标记。

### 结论

通过消融研究和错误分析，提出了一套评估指标来评估预测的节奏模式序列的准确性和可读性。

### 翻译

虽然和弦转录在过去几十年中受到了相当多的关注，但投入到转录和编码歌曲中出现的节奏模式的工作却少得多。这一主题对于节奏吉他等乐器尤为重要，因为这些乐器通常通过扫弦演奏重复和随时间变化的节奏模式。然而，在许多情况下，无法客观地为给定的歌曲段落定义单一的'正确'节奏模式。为了创建一个具有明确定义真实标签的数据集，我们要求专业音乐家转录410首流行歌曲中的节奏模式，并录制吉他音轨遵循这些转录的封面版本。为了转录扫弦及其相应的节奏模式，我们提出了一个三步框架。首先，我们执行近似音源分离，从多声道混合中提取吉他部分。其次，我们使用预训练的基础模型(MERT)作为骨干网络，检测分离的吉他音频中的单个扫弦。最后，我们进行模式解码过程，将转录的吉他扫弦序列用专家策划的词汇中的模式表示。我们表明，可以以相当高的精度转录多声道音乐中吉他音轨的节奏模式，产生一种人类可读的表示，包括自动检测的小节线和拍号标记。我们进行了消融研究和错误分析，并提出了一套评估指标来评估预测的节奏模式序列的准确性和可读性。


### 论文摘要

Whereas chord transcription has received considerable attention during the past couple of decades, far less work has been devoted to transcribing and encoding the rhythmic patterns that occur in a song. The topic is especially relevant for instruments such as the rhythm guitar, which is typically played by strumming rhythmic patterns that repeat and vary over time. However, in many cases one cannot objectively define a single "right" rhythmic pattern for a given song section. To create a dataset with well-defined ground-truth labels, we asked expert musicians to transcribe the rhythmic patterns in 410 popular songs and record cover versions where the guitar tracks followed those transcriptions. To transcribe the strums and their corresponding rhythmic patterns, we propose a three-step framework. Firstly, we perform approximate stem separation to extract the guitar part from the polyphonic mixture. Secondly, we detect individual strums within the separated guitar audio, using a pre-trained foundation model (MERT) as a backbone. Finally, we carry out a pattern-decoding process in which the transcribed sequence of guitar strums is represented by patterns drawn from an expert-curated vocabulary. We show that it is possible to transcribe the rhythmic patterns of the guitar track in polyphonic music with quite high accuracy, producing a representation that is human-readable and includes automatically detected bar lines and time signature markers. We perform ablation studies and error analysis and propose a set of evaluation metrics to assess the accuracy and readability of the predicted rhythmic pattern sequence.

---

## 55. ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems

**论文链接:** [http://arxiv.org/abs/2510.05746v1](http://arxiv.org/abs/2510.05746v1)

**作者:** Bohan Yao, Shiva Krishna Reddy Malay, Vikas Yadav

**发布时间:** 2025-10-07

**备注:** 29 pages, 2 figures

### GPT解析

### 总结

该研究提出了一种新的多智能体系统自动设计范式，通过优化思维链推理来显著提升系统性能，并开发出智能体推理模块(ARM)作为通用推理构建块，展现出优秀的泛化能力。

### 背景

大语言模型驱动的多智能体系统在复杂推理任务上取得先进成果，但现有自动化MAS设计技术表现不佳，常与简单基线相当或更差，且需要计算昂贵的架构重新发现和数据标注。

### 目的

提出一种新的MAS自动设计范式，将重点转移到优化思维链(CoT)推理上，因为简单的CoT推理通常与复杂系统具有竞争力。

### 方法

开发了智能体推理模块(ARM)，这是CoT的智能体泛化，通过在代码空间上的树搜索发现，从简单CoT模块开始，基于执行轨迹反思的突变进行演化，可作为递归循环或元编排器中的子程序使用。

### 主要发现

ARM方法显著优于手动设计的MAS和最先进的自动MAS设计方法，使用ARM构建的MAS在不同基础模型和任务域上保持高性能，无需进一步优化。

### 结论

通过专注于优化思维链推理，可以开发出更有效的多智能体系统，ARM作为通用推理构建块具有良好的泛化能力。

### 翻译

大语言模型(LLM)驱动的多智能体系统(MAS)已在各种复杂推理任务上取得了最先进的结果。最近的工作提出了自动化MAS设计的技术，消除了手动工程的需要。然而，这些技术表现不佳，通常只达到与简单基线相似或更差的表现。此外，它们需要为每个新的任务域进行计算昂贵的架构重新发现，并且在没有现有标记验证集的领域上进行昂贵的数据标注。一个关键的见解是，简单的思维链(CoT)推理通常与这些复杂系统具有竞争力，这表明MAS的基本推理单元CoT值得进一步研究。为此，我们提出了一种新的MAS自动设计范式，将重点转移到优化CoT推理上。我们引入了智能体推理模块(ARM)，这是CoT的一种智能体泛化，其中每个细粒度的推理步骤由专门的推理模块执行。该模块通过在代码空间上的树搜索被发现，从一个简单的CoT模块开始，并通过基于执行轨迹反思的突变进行演化。生成的ARM作为通用的推理构建块，可以直接用作递归循环或作为学习到的元编排器中的子程序。我们的方法显著优于手动设计的MAS和最先进的自动MAS设计方法。关键是，使用ARM构建的MAS表现出出色的泛化能力，在不同的基础模型和任务域上保持高性能，无需进一步优化。


### 论文摘要

Large Language Model (LLM)-powered Multi-agent systems (MAS) have achieved state-of-the-art results on various complex reasoning tasks. Recent works have proposed techniques to automate the design of MASes, eliminating the need for manual engineering. However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines. Furthermore, they require computationally expensive re-discovery of architectures for each new task domain and expensive data annotation on domains without existing labeled validation sets. A critical insight is that simple Chain of Thought (CoT) reasoning often performs competitively with these complex systems, suggesting that the fundamental reasoning unit of MASes, CoT, warrants further investigation. To this end, we present a new paradigm for automatic MAS design that pivots the focus to optimizing CoT reasoning. We introduce the Agentic Reasoning Module (ARM), an agentic generalization of CoT where each granular reasoning step is executed by a specialized reasoning module. This module is discovered through a tree search over the code space, starting from a simple CoT module and evolved using mutations informed by reflection on execution traces. The resulting ARM acts as a versatile reasoning building block which can be utilized as a direct recursive loop or as a subroutine in a learned meta-orchestrator. Our approach significantly outperforms both manually designed MASes and state-of-the-art automatic MAS design methods. Crucially, MASes built with ARM exhibit superb generalization, maintaining high performance across different foundation models and task domains without further optimization.

---

## 56. Redefining Generalization in Visual Domains: A Two-Axis Framework for Fake Image Detection with FusionDetect

**论文链接:** [http://arxiv.org/abs/2510.05740v1](http://arxiv.org/abs/2510.05740v1)

**作者:** Amirtaha Amanzadi, Zahra Dehghanian, Hamid Beigy, Hamid R. Rabiee

**发布时间:** 2025-10-07

**备注:** Project code: http://github.com/amir-aman/FusionDetect

### GPT解析

### 总结

本文提出了OmniGen基准测试和FusionDetect方法，用于解决生成图像检测中的跨生成器和跨视觉域泛化问题，实验表明新方法在多个基准测试上取得了最先进的结果。

### 背景

生成式模型的快速发展使得开发能够可靠检测合成图像的检测器变得日益重要。目前大多数工作集中在跨生成器泛化上，但这一观点过于局限。

### 目的

解决检测合成图像中的另一个重要挑战：跨视觉域泛化能力。为此，提出OmniGen基准测试和FusionDetect方法。

### 方法

介绍了FusionDetect方法，它利用两个冻结的基础模型(CLIP & Dinov2)的优势，通过从这两个互补模型中提取特征，开发了一个能够自然适应生成器内容和设计变化的一致特征空间。

### 主要发现

实验表明FusionDetect比最接近的竞争对手准确率高3.87%，在已建立基准上的平均精度高6.13%，在OmniGen上准确率提高了4.48%，同时对常见的图像扰动表现出卓越的鲁棒性。

### 结论

不仅引入了一个高性能的检测器，还引入了一个新的基准和框架，用于推进通用AI图像检测。

### 翻译

生成模型的快速发展使得开发能够可靠检测合成图像的检测器变得日益重要。尽管目前大多数工作集中在跨生成器泛化上，我们认为这一观点过于局限。检测合成图像涉及另一个同样重要的挑战：跨视觉域泛化。为弥合这一差距，我们提出了OmniGen基准测试。这个全面的评估数据集纳入了12种最先进的生成器，提供了一种更现实的方式来评估检测器在现实条件下的性能。此外，我们引入了一种新方法FusionDetect，旨在解决两个泛化向量。FusionDraw利用两个冻结的基础模型(CLIP & Dinov2)的优势。通过从这两个互补模型中提取特征，我们开发了一个能够自然适应生成器内容和设计变化的一致特征空间。我们广泛的实验表明，FusionDetect不仅达到了新的最先进水平，比最接近的竞争对手准确率高3.87%，在已建立基准上的平均精度高6.13%，而且在OmniGen上准确率提高了4.48%，同时对常见的图像扰动表现出卓越的鲁棒性。我们不仅引入了一个高性能的检测器，还引入了一个新的基准和框架，用于推进通用AI图像检测。代码和数据集可在http://github.com/amir-aman/FusionDetect获取。


### 论文摘要

The rapid development of generative models has made it increasingly crucial to develop detectors that can reliably detect synthetic images. Although most of the work has now focused on cross-generator generalization, we argue that this viewpoint is too limited. Detecting synthetic images involves another equally important challenge: generalization across visual domains. To bridge this gap,we present the OmniGen Benchmark. This comprehensive evaluation dataset incorporates 12 state-of-the-art generators, providing a more realistic way of evaluating detector performance under realistic conditions. In addition, we introduce a new method, FusionDetect, aimed at addressing both vectors of generalization. FusionDetect draws on the benefits of two frozen foundation models: CLIP & Dinov2. By deriving features from both complementary models,we develop a cohesive feature space that naturally adapts to changes in both thecontent and design of the generator. Our extensive experiments demonstrate that FusionDetect delivers not only a new state-of-the-art, which is 3.87% more accurate than its closest competitor and 6.13% more precise on average on established benchmarks, but also achieves a 4.48% increase in accuracy on OmniGen,along with exceptional robustness to common image perturbations. We introduce not only a top-performing detector, but also a new benchmark and framework for furthering universal AI image detection. The code and dataset are available at http://github.com/amir-aman/FusionDetect

---

## 57. InstaGeo: Compute-Efficient Geospatial Machine Learning from Data to Deployment

**论文链接:** [http://arxiv.org/abs/2510.05617v1](http://arxiv.org/abs/2510.05617v1)

**作者:** Ibrahim Salihu Yusuf, Iffanice Houndayi, Rym Oualha, Mohamed Aziz Cherif, Kobby Panford-Quainoo, Arnu Pretorius

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文介绍了InstaGeo，一个开源的端到端框架，用于解决地理空间基础模型(GFMs)部署中的两个主要限制：缺乏自动化的地理空间数据管道和微调模型体积过大。该框架整合了自动数据整理、任务特定模型蒸馏和交互式部署功能，将研究级GFMs转变为实用的低碳工具，用于实时、大规模的地球观测。

### 背景

来自Landsat 8-9和Sentinel-2等任务的开放多光谱图像推动了地理空间基础模型在人道主义和环境应用方面的发展。然而，这些模型的部署仍然受到两个主要限制：(i)缺乏自动化的地理空间数据管道，以及(ii)微调模型体积过大。现有的GFMs缺乏处理原始卫星图像的工作流程，而下游调整通常保留了原始编码器的全部复杂性。

### 目的

开发一个开源的端到端框架，解决地理空间基础模型部署中的自动化数据管道缺失和模型体积过大的问题，将研究级GFMs转变为实用的低碳工具，用于实时、大规模的地球观测。

### 方法

InstaGeo框架整合了三个主要组件：(1)自动数据整理，将原始图像转换为模型就绪的数据集；(2)任务特定模型蒸馏，生成紧凑、计算效率高的模型；(3)无缝部署为交互式网络地图应用。

### 主要发现

1) 使用InstaGeo重现了三项已发表研究的数据集，训练的模型在洪水测绘中mIoU差异为-0.73 pp，作物分割中为-0.20 pp，沙漠蝗虫预测中为+1.79 pp；2) 蒸馏后的模型比标准微调对应模型小8倍，减少了FLOPs和CO2排放，同时精度损失最小；3) 利用InstaGeo的简化数据管道，整理了一个更大的作物分割数据集，实现了最先进的60.65% mIoU，比先前基线提高了12个百分点；4) InstaGeo使用户能够在单个工作日内从原始数据到模型部署。

### 结论

通过统一数据准备、模型压缩和部署，InstaGeo将研究级GFMs转变为实用的、低碳的工具，用于实时、大规模的地球观测。这种方法将地理空间AI转向数据质量和应用驱动的创新。

### 翻译

来自Landsat 8-9和Sentinel-2等任务的开放多光谱图像推动了地理空间基础模型(GFMs)在人道主义和环境应用方面的发展。然而，它们的部署仍然受到两个方面的限制：(i)缺乏自动化的地理空间数据管道，以及(ii)微调模型体积过大。现有的GFMs缺乏处理原始卫星图像的工作流程，而下游调整通常保留了原始编码器的全部复杂性。我们提出了InstaGeo，这是一个开源的端到端框架，通过整合以下内容解决了这些挑战：(1)自动数据整理，将原始图像转换为模型就绪的数据集；(2)任务特定模型蒸馏，生成紧凑、计算效率高的模型；(3)无缝部署为交互式网络地图应用。使用InstaGeo，我们重现了三项已发表研究的数据集，训练的模型在洪水测绘中mIoU差异为-0.73 pp，在作物分割中为-0.20 pp，在沙漠蝗虫预测中为+1.79 pp。蒸馏后的模型比标准微调对应模型小8倍，减少了FLOPs和CO2排放，同时精度损失最小。利用InstaGeo的简化数据管道，我们还整理了一个更大的作物分割数据集，实现了最先进的60.65% mIoU，比先前基线提高了12个百分点。此外，InstaGeo使用户能够在单个工作日内从原始数据到模型部署。通过统一数据准备、模型压缩和部署，InstaGeo将研究级GFMs转变为实用的、低碳的工具，用于实时、大规模的地球观测。这种方法将地理空间AI转向数据质量和应用驱动的创新。源代码、数据集和模型检查点可在以下网址获取：https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git


### 论文摘要

Open-access multispectral imagery from missions like Landsat 8-9 and Sentinel-2 has fueled the development of geospatial foundation models (GFMs) for humanitarian and environmental applications. Yet, their deployment remains limited by (i) the absence of automated geospatial data pipelines and (ii) the large size of fine-tuned models. Existing GFMs lack workflows for processing raw satellite imagery, and downstream adaptations often retain the full complexity of the original encoder.   We present InstaGeo, an open-source, end-to-end framework that addresses these challenges by integrating: (1) automated data curation to transform raw imagery into model-ready datasets; (2) task-specific model distillation to derive compact, compute-efficient models; and (3) seamless deployment as interactive web-map applications. Using InstaGeo, we reproduced datasets from three published studies and trained models with marginal mIoU differences of -0.73 pp for flood mapping, -0.20 pp for crop segmentation, and +1.79 pp for desert locust prediction. The distilled models are up to 8x smaller than standard fine-tuned counterparts, reducing FLOPs and CO2 emissions with minimal accuracy loss.   Leveraging InstaGeo's streamlined data pipeline, we also curated a larger crop segmentation dataset, achieving a state-of-the-art mIoU of 60.65%, a 12 pp improvement over prior baselines. Moreover, InstaGeo enables users to progress from raw data to model deployment within a single working day.   By unifying data preparation, model compression, and deployment, InstaGeo transforms research-grade GFMs into practical, low-carbon tools for real-time, large-scale Earth observation. This approach shifts geospatial AI toward data quality and application-driven innovation. Source code, datasets, and model checkpoints are available at: https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git

---

## 58. Improving Chain-of-Thought Efficiency for Autoregressive Image Generation

**论文链接:** [http://arxiv.org/abs/2510.05593v1](http://arxiv.org/abs/2510.05593v1)

**作者:** Zeqi Gu, Markos Georgopoulos, Xiaoliang Dai, Marjan Ghazvininejad, Chu Wang, Felix Juefei-Xu, Kunpeng Li, Yujun Shi, Zecheng He, Zijian He, Jiawei Zhou, Abe Davis, Jialiang Wang

**发布时间:** 2025-10-07

### GPT解析

### 总结

本文提出ShortCoTI框架，通过优化思维链推理过程，减少图像生成中的冗余提示，提高计算效率同时保持图像质量。

### 背景

自回归多模态大语言模型在图像生成领域流行，新方法采用思维链推理增强对齐和细节，但这种方法可能引入不必要的冗余，增加计算成本并产生与原始提示矛盾的细节。

### 目的

探索如何生成更简洁的思维链序列，实现更高效的图像生成。

### 方法

介绍ShortCoTI轻量级优化框架，使用自适应函数奖励更简洁的提示，该函数根据任务估计难度调整，并将此奖励纳入强化学习范式。

### 主要发现

ShortCoTI将提示推理长度减少54%，在多个基准测试中保持或略微提高质量指标，消除冗长解释和重复改进，生成既简洁又语义丰富的推理提示。

### 结论

ShortCoTI提高计算效率，同时不损害生成图像的保真度或视觉吸引力。

### 翻译

自回归多模态大语言模型最近在图像生成领域流行起来，这得益于基础模型的进展。为了增强对齐和细节，新方法采用思维链推理，在图像合成前将用户输入扩展为详细提示。然而，这种策略可能引入不必要的冗余——我们称之为视觉过度思考——这增加了计算成本，并可能引入与原始提示相矛盾的细节。在这项工作中，我们探索如何生成更简洁的思维链序列以实现更高效的图像生成。我们介绍ShortCoTI，一个轻量级优化框架，它鼓励更简洁的思维链，同时保持输出图像质量。ShortCoTI使用自适应函数奖励更简洁的提示，该函数根据每个任务的估计难度进行调整。将此奖励纳入强化学习范式，可以在多个基准测试中将提示推理长度减少54%，同时保持或略微提高质量指标。定性分析表明，我们的方法消除了冗长的解释和重复的改进，产生既简洁又语义丰富的推理提示。因此，ShortCoTI提高了计算效率，同时不损害生成图像的保真度或视觉吸引力。


### 论文摘要

Autoregressive multimodal large language models have recently gained popularity for image generation, driven by advances in foundation models. To enhance alignment and detail, newer approaches employ chain-of-thought (CoT) reasoning, expanding user inputs into elaborated prompts prior to image synthesis. However, this strategy can introduce unnecessary redundancy -- a phenomenon we call visual overthinking -- which increases computational costs and can introduce details that contradict the original prompt. In this work, we explore how to generate more concise CoT sequences for more efficient image generation. We introduce ShortCoTI, a lightweight optimization framework that encourages more concise CoT while preserving output image quality. ShortCoTI rewards more concise prompts with an adaptive function that scales according to an estimated difficulty for each task. Incorporating this reward into a reinforcement learning paradigm reduces prompt reasoning length by 54% while maintaining or slightly improving quality metrics across multiple benchmarks (T2I-CompBench, GenEval). Qualitative analysis shows that our method eliminates verbose explanations and repetitive refinements, producing reasoning prompts that are both concise and semantically rich. As a result, ShortCoTI improves computational efficiency without compromising the fidelity or visual appeal of generated images.

---

## 59. ATOM: A Pretrained Neural Operator for Multitask Molecular Dynamics

**论文链接:** [http://arxiv.org/abs/2510.05482v1](http://arxiv.org/abs/2510.05482v1)

**作者:** Luke Thompson, Davy Guan, Dai Shi, Slade Matthews, Junbin Gao, Andi Han

**发布时间:** 2025-10-07

### GPT解析

### 总结

ATOM是一个预训练的Transformer神经算子，用于多任务分子动力学模拟，采用准等变设计和时间注意力机制，能够在无需明确分子图的情况下准确并行解码多个未来状态。

### 背景

分子动力学模拟是现代计算药物发现、材料科学和生物化学的基础。最近的机器学习模型能够提供高保真的分子动力学预测，而无需重复求解量子力学力，相比传统流程有显著加速。然而，许多此类方法通常强制执行严格的等变性并依赖顺序滚动，限制了其灵活性和模拟效率。它们也通常是单任务的，针对单个分子和固定时间帧进行训练，这限制了它们对未见化合物和扩展时间步长的泛化能力。

### 目的

解决现有分子动力学模拟方法的限制，提高模型的灵活性、效率和泛化能力，特别是对未见化合物和不同时间跨度的预测。

### 方法

提出ATOM（分子原子Transformer算子），一个预训练的Transformer神经算子，用于多任务分子动力学。ATOM采用准等变设计，不需要明确的分子图，并采用时间注意力机制，允许准确并行解码多个未来状态。为了支持跨化学品和时间尺度的算子预训练，作者创建了TG80数据集，这是一个包含80种化合物超过250万飞秒轨迹的大型、多样化且数值稳定的分子动力学数据集。

### 主要发现

ATOM在既定的单任务基准测试（如MD17、RMD17和MD22）上取得了最先进的性能。在TG80上进行多任务预训练后，ATOM对跨不同时间跨度的未见分子表现出卓越的零样本泛化能力。

### 结论

ATOM代表了向准确、高效和可转移的分子动力学模型迈出的重要一步。

### 翻译

分子动力学模拟支持现代计算药物发现、材料科学和生物化学。最近的机器学习模型提供高保真的分子动力学预测，无需重复求解量子力学力，相比传统流程实现了显著加速。然而，许多此类方法通常强制执行严格的等变性并依赖顺序滚动，从而限制了它们的灵活性和模拟效率。它们也通常是单任务的，针对单个分子和固定时间帧进行训练，这限制了它们对未见化合物和扩展时间步长的泛化能力。为解决这些问题，我们提出了ATOM（分子原子Transformer算子），这是一个用于多任务分子动力学的预训练Transformer神经算子。ATOM采用准等变设计，不需要明确的分子图，并采用时间注意力机制，允许准确并行解码多个未来状态。为了支持跨化学品和时间尺度的算子预训练，我们整理了TG80，这是一个包含80种化合物超过250万飞秒轨迹的大型、多样化且数值稳定的分子动力学数据集。ATOM在既定的单任务基准测试（如MD17、RMD17和MD22）上取得了最先进的性能。在TG80上进行多任务预训练后，ATOM对跨不同时间跨度的未见分子表现出卓越的零样本泛化能力。我们认为ATOM代表了向准确、高效和可转移的分子动力学模型迈出的重要一步。


### 论文摘要

Molecular dynamics (MD) simulations underpin modern computational drug dis- covery, materials science, and biochemistry. Recent machine learning models provide high-fidelity MD predictions without the need to repeatedly solve quantum mechanical forces, enabling significant speedups over conventional pipelines. Yet many such methods typically enforce strict equivariance and rely on sequential rollouts, thus limiting their flexibility and simulation efficiency. They are also com- monly single-task, trained on individual molecules and fixed timeframes, which restricts generalization to unseen compounds and extended timesteps. To address these issues, we propose Atomistic Transformer Operator for Molecules (ATOM), a pretrained transformer neural operator for multitask molecular dynamics. ATOM adopts a quasi-equivariant design that requires no explicit molecular graph and employs a temporal attention mechanism, allowing for the accurate parallel decod- ing of multiple future states. To support operator pretraining across chemicals and timescales, we curate TG80, a large, diverse, and numerically stable MD dataset with over 2.5 million femtoseconds of trajectories across 80 compounds. ATOM achieves state-of-the-art performance on established single-task benchmarks, such as MD17, RMD17 and MD22. After multitask pretraining on TG80, ATOM shows exceptional zero-shot generalization to unseen molecules across varying time hori- zons. We believe ATOM represents a significant step toward accurate, efficient, and transferable molecular dynamics models

---

## 60. Vul-R2: A Reasoning LLM for Automated Vulnerability Repair

**论文链接:** [http://arxiv.org/abs/2510.05480v1](http://arxiv.org/abs/2510.05480v1)

**作者:** Xin-Cheng Wen, Zirui Lin, Yijun Yang, Cuiyun Gao, Deheng Ye

**发布时间:** 2025-10-07

**备注:** 13 pages, 8 figures. This paper is accepted by ASE 2025

### GPT解析

### 总结

当前自动漏洞修复(AVR)方法将问题表述为序列生成问题并利用大型语言模型解决，但面临缺乏高质量漏洞相关推理数据和难以验证中间修复过程两大挑战。

### 背景

软件漏洞数量呈指数级增长，对自动漏洞修复解决方案有迫切需求。

### 目的

解决当前自动漏洞修复方法面临的挑战，提高漏洞修复效果。

### 方法

将自动漏洞修复表述为序列生成问题，利用大型语言模型生成漏洞修复方案。

### 主要发现

当前方法面临两大挑战：(1)缺乏高质量、漏洞相关的推理数据，导致难以捕捉多样化的漏洞修复模式；(2)难以在LLM训练过程中验证中间的漏洞修复过程，因为缺乏可验证的中间反馈。

### 结论

虽然当前自动漏洞修复方法展示了最先进的性能，但缺乏高质量漏洞相关推理数据和难以验证中间修复过程仍是主要挑战。

### 翻译

软件漏洞的指数级增长已经对自动漏洞修复(AVR)解决方案产生了迫切需求。最近的研究将AVR表述为序列生成问题，并利用大型语言模型(LLMs)来解决这一问题。通常，这些方法提示或微调LLMs以直接生成漏洞修复方案。尽管这些方法展示了最先进的性能，但它们面临以下挑战：(1)缺乏高质量的、与漏洞相关的推理数据。当前方法主要依赖主要编码通用编程知识的基础模型。没有漏洞相关的推理数据，它们往往无法捕捉多样化的漏洞修复模式。(2)难以在LLM训练过程中验证中间的漏洞修复过程。现有的强化学习方法通常利用来自环境的中间执行反馈(例如基于沙盒的执行结果)来指导强化学习训练。相比之下，漏洞修复过程通常缺乏这种可验证的中间反馈，这给模型训练带来了额外的挑战。


### 论文摘要

The exponential increase in software vulnerabilities has created an urgent need for automatic vulnerability repair (AVR) solutions. Recent research has formulated AVR as a sequence generation problem and has leveraged large language models (LLMs) to address this problem. Typically, these approaches prompt or fine-tune LLMs to generate repairs for vulnerabilities directly. Although these methods show state-of-the-art performance, they face the following challenges: (1) Lack of high-quality, vulnerability-related reasoning data. Current approaches primarily rely on foundation models that mainly encode general programming knowledge. Without vulnerability-related reasoning data, they tend to fail to capture the diverse vulnerability repair patterns. (2) Hard to verify the intermediate vulnerability repair process during LLM training. Existing reinforcement learning methods often leverage intermediate execution feedback from the environment (e.g., sandbox-based execution results) to guide reinforcement learning training. In contrast, the vulnerability repair process generally lacks such intermediate, verifiable feedback, which poses additional challenges for model training.

---

## 61. MHA-RAG: Improving Efficiency, Accuracy, and Consistency by Encoding Exemplars as Soft Prompts

**论文链接:** [http://arxiv.org/abs/2510.05363v1](http://arxiv.org/abs/2510.05363v1)

**作者:** Abhinav Jain, Xinyu Yao, Thomas Reps, Christopher Jermaine

**发布时间:** 2025-10-06

**备注:** 17 pages, 5 figures

### GPT解析

### 总结

研究提出了一种新的MHA-RAG框架，通过将示例表示为软提示并使用多头注意力机制，在有限数据下有效适应基础模型到新领域，同时提高性能并降低计算成本。

### 背景

基础模型在有限训练数据下适应新领域具有挑战性且计算成本高。先前研究表明使用领域特定示例作为上下文演示是有效的，但纯文本表示可能不是最优方案。

### 目的

调查将示例纯文本表示是否是最有效、高效和稳定的方法，并探索替代方案：将示例表示为软提示，并使用示例顺序不变的模型架构。

### 方法

引入多头注意力检索增强生成(MHA-RAG)框架，使用注意力头数量作为简单超参数控制软提示生成，架构对示例顺序具有不变性。

### 主要发现

在多个问答基准和模型规模上，MHA-RAG比标准RAG实现了20点的性能提升，同时将推理成本降低了10倍XGFLOPs，提供了更高的准确性和效率。

### 结论

MHA-RAG框架在准确性和效率方面都优于标准RAG，且对示例顺序具有不变性，是一种更有效的领域适应方法。

### 翻译

在有限训练数据下将基础模型适应到新领域具有挑战性且计算成本高昂。虽然先前工作已经证明了使用领域特定示例作为上下文演示的有效性，但我们研究将示例纯文本表示是否是最有效、高效和稳定的方法。我们探索了一种替代方案：将示例表示为软提示，并采用示例顺序不变的模型架构。为此，我们引入了多头注意力检索增强生成(MHA-RAG)，这是一个框架，其中注意力头的数量作为简单超参数，用于控制不同任务中的软提示生成。在多个问答基准和模型规模上，MHA-RAG比标准RAG实现了20点的性能提升，同时将推理成本降低了10倍XGFLOPs—提供了更高的准确性和效率，且对示例顺序不变。


### 论文摘要

Adapting Foundation Models to new domains with limited training data is challenging and computationally expensive. While prior work has demonstrated the effectiveness of using domain-specific exemplars as in-context demonstrations, we investigate whether representing exemplars purely as text is the most efficient, effective, and stable approach. We explore an alternative: representing exemplars as soft prompts with an exemplar order invariant model architecture. To this end, we introduce Multi-Head Attention Retrieval-Augmented Generation (MHA-RAG), a framework with the number of attention heads serving as a simple hyperparameter to control soft prompt-generation across different tasks. Across multiple question-answering benchmarks and model scales, MHA-RAG achieves a 20-point performance gain over standard RAG, while cutting inference costs by a factor of 10X GFLOPs-delivering both higher accuracy and greater efficiency, invariant to exemplar order.

---

## 62. VER: Vision Expert Transformer for Robot Learning via Foundation Distillation and Dynamic Routing

**论文链接:** [http://arxiv.org/abs/2510.05213v1](http://arxiv.org/abs/2510.05213v1)

**作者:** Yixiao Wang, Mingxiao Huo, Zhixuan Liang, Yushi Du, Lingfeng Sun, Haotian Lin, Jinghuan Shang, Chensheng Peng, Mohit Bansal, Mingyu Ding, Masayoshi Tomizuka

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出了VER(Vision Expert transformer for Robot learning)，一种用于机器人学习的视觉专家transformer模型，通过动态选择多个预训练视觉基础模型(VFMs)的专家，实现了跨任务的高性能机器人学习。

### 背景

预训练视觉基础模型(VFMs)通过丰富的视觉表示推进了机器人学习，但单个VFM通常只在特定领域表现出色，限制了跨任务的泛化能力。将多个VFM蒸馏为统一表示用于策略可以缓解这一问题，但通常导致僵化的任务特定特征选择，并需要昂贵的完全重新训练来融入机器人领域知识。

### 目的

开发一种能够灵活选择任务相关特征并高效整合机器人领域知识的视觉表示模型，以提升机器人在多样化任务中的学习能力和泛化性能。

### 方法

在预训练阶段将多个VFM蒸馏到视觉专家库中；仅微调一个轻量级路由网络(参数少于0.4%)，从预训练库中动态选择任务相关专家；引入基于课程Top-K退火的分块专家路由，提高动态专家选择的灵活性和精确度；支持参数高效的微调，实现可扩展的专家利用和自适应机器人领域知识整合。

### 主要发现

在17个多样化的机器人任务和多个策略头中，VER实现了最先进的性能；VER减少了任务不相关区域(如背景)中的大范数异常值，并专注于任务关键区域。

### 结论

VER有效解决了现有方法在机器人学习中的局限性，实现了更灵活的特征选择和更高效的知识整合，在多种任务上取得了优异的性能。

### 翻译

预训练视觉基础模型(VFMs)通过丰富的视觉表示推进机器人学习，然而单个VFM通常只在特定领域表现出色，限制了跨任务的泛化能力。将多个VMs蒸馏为统一表示用于策略可以缓解这一限制，但通常产生僵化的任务特定特征选择，并需要昂贵的完全重新训练来融入机器人领域知识。我们提出了VER，一种用于机器人学习的视觉专家transformer。在预训练期间，VER将多个VFM蒸馏到视觉专家库中。然后，它仅微调一个轻量级路由网络(参数少于0.4%)，从预训练库中动态选择任务相关专家用于下游机器人任务。我们进一步引入了基于课程Top-K退火的分块专家路由，以提高动态专家选择的灵活性和精确度。此外，VER支持参数高效的微调，以实现可扩展的专家利用和自适应机器人领域知识整合。在17个多样化的机器人任务和多个策略头中，VER实现了最先进的性能。我们发现VER减少了任务不相关区域(如背景)中的大范数异常值，并专注于任务关键区域。可视化和代码可在https://yixiaowang7.github.io/ver_page/找到。


### 论文摘要

Pretrained vision foundation models (VFMs) advance robotic learning via rich visual representations, yet individual VFMs typically excel only in specific domains, limiting generality across tasks. Distilling multiple VFMs into a unified representation for policy can mitigate this limitation but often yields inflexible task-specific feature selection and requires costly full re-training to incorporate robot-domain knowledge. We propose VER, a Vision Expert transformer for Robot learning. During pretraining, VER distills multiple VFMs into a vision expert library. It then fine-tunes only a lightweight routing network (fewer than 0.4% of parameters) to dynamically select task-relevant experts from the pretrained library for downstream robot tasks. We further introduce Patchwise Expert Routing with Curriculum Top-K Annealing to improve both flexibility and precision of dynamic expert selection. Moreover, VER supports parameter-efficient finetuning for scalable expert utilization and adaptive robot-domain knowledge integration. Across 17 diverse robotic tasks and multiple policy heads, VER achieves state-of-the-art performance. We find that VER reduces large-norm outliers in task-irrelevant regions (e.g., background) and concentrates on task-critical regions. Visualizations and codes can be found in https://yixiaowang7.github.io/ver_page/.

---

## 63. Representation Potentials of Foundation Models for Multimodal Alignment: A Survey

**论文链接:** [http://arxiv.org/abs/2510.05184v1](http://arxiv.org/abs/2510.05184v1)

**作者:** Jianglin Lu, Hailing Wang, Yi Xu, Yizhou Wang, Kuo Yang, Yun Fu

**发布时间:** 2025-10-05

### GPT解析

### 总结

本文综述了基础模型的表示潜力，即其学习到的表示在单一模态内捕获特定任务信息的能力，同时为跨模态对齐和统一提供可迁移的基础。

### 背景

基础模型通过大规模多样化数据预学习高度可迁移的表示，研究表明这些表示在不同架构和模态间表现出显著的相似性。

### 目的

调查基础模型的表示潜力，并分析其在跨模态迁移和对齐中的潜力。

### 方法

回顾代表性基础模型和使对齐可测量的关键指标，综合来自视觉、语言、语音、多模态和神经科学研究中表示潜力的实证证据。

### 主要发现

基础模型在其表示空间中通常表现出结构规律性和语义一致性，这使它们成为跨模态迁移和对齐的有力候选者。

### 结论

分析促进表示潜力的关键因素，讨论开放性问题，并指出潜在挑战。

### 翻译

基础模型通过大规模多样化数据预学习高度可迁移的表示。越来越多的研究表明这些表示在不同架构和模态间表现出显著的相似性。在本综述中，我们调查了基础模型的表示潜力，定义为它们学习到的表示在单一模态内捕获特定任务信息的能力，同时为跨模态对齐和统一提供可迁移的基础。我们从回顾代表性的基础模型和使对齐可测量的关键指标开始。然后我们综合了来自视觉、语言、语音、多模态和神经科学研究中表示潜力的实证证据。证据表明基础模型在其表示空间中通常表现出结构规律性和语义一致性，这使它们成为跨模态迁移和对齐的有力候选者。我们进一步分析了促进表示潜力的关键因素，讨论了开放性问题，并指出了潜在挑战。


### 论文摘要

Foundation models learn highly transferable representations through large-scale pretraining on diverse data. An increasing body of research indicates that these representations exhibit a remarkable degree of similarity across architectures and modalities. In this survey, we investigate the representation potentials of foundation models, defined as the latent capacity of their learned representations to capture task-specific information within a single modality while also providing a transferable basis for alignment and unification across modalities. We begin by reviewing representative foundation models and the key metrics that make alignment measurable. We then synthesize empirical evidence of representation potentials from studies in vision, language, speech, multimodality, and neuroscience. The evidence suggests that foundation models often exhibit structural regularities and semantic consistencies in their representation spaces, positioning them as strong candidates for cross-modal transfer and alignment. We further analyze the key factors that foster representation potentials, discuss open questions, and highlight potential challenges.

---

## 64. Learning to Crawl: Latent Model-Based Reinforcement Learning for Soft Robotic Adaptive Locomotion

**论文链接:** [http://arxiv.org/abs/2510.05957v1](http://arxiv.org/abs/2510.05957v1)

**作者:** Vaughn Gzenda, Robin Chhabra

**发布时间:** 2025-10-07

### GPT解析

### 总结

本研究提出了一种基于模型的强化学习框架，用于软体爬行机器人的自适应运动控制，通过从机载传感器推断的潜在动力学来指导运动策略优化。

### 背景

软体爬行机器人利用软体变形和顺应性通过表面接触实现移动，但设计其控制策略面临模型不准确、传感器噪声和需要发现运动步态等挑战。

### 目的

开发一种基于模型的强化学习框架，使软体机器人能够仅依靠嘈杂的传感器反馈实现自适应运动。

### 方法

提出一种基于模型的强化学习框架，其中从机载传感器推断的潜在动力学作为预测模型，指导actor-critic算法优化运动策略。在模拟中使用惯性测量单元和飞行时间传感器作为观察值，对最小爬行模型进行评估。

### 主要发现

学习的潜在动力学能够实现短时间范围内的运动预测，而actor-critic算法发现了有效的运动策略。

### 结论

基于潜在动力学的MB-RL方法在仅依靠嘈杂传感器反馈的基础上，为软体机器人的自适应运动提供了可能性。

### 翻译

软体爬行机器人是利用软体变形和顺应性通过表面接触实现移动的移动机器人。由于模型不准确、传感器噪声以及需要发现运动步态等因素，为这类系统设计控制策略具有挑战性。在这项工作中，我们提出了一种基于模型的强化学习框架，其中从机载传感器推断的潜在动力学作为预测模型，指导actor-critic算法优化运动策略。我们在模拟中使用惯性测量单元和飞行时间传感器作为观察值，对最小爬行模型评估了该框架。学习的潜在动力学实现了短时间范围内的运动预测，而actor-critic发现了有效的运动策略。这种方法强调了基于潜在动力学的MB-RL在仅依靠嘈杂传感器反馈的基础上实现软体机器人自适应运动的潜力。


### 论文摘要

Soft robotic crawlers are mobile robots that utilize soft body deformability and compliance to achieve locomotion through surface contact. Designing control strategies for such systems is challenging due to model inaccuracies, sensor noise, and the need to discover locomotor gaits. In this work, we present a model-based reinforcement learning (MB-RL) framework in which latent dynamics inferred from onboard sensors serve as a predictive model that guides an actor-critic algorithm to optimize locomotor policies. We evaluate the framework on a minimal crawler model in simulation using inertial measurement units and time-of-flight sensors as observations. The learned latent dynamics enable short-horizon motion prediction while the actor-critic discovers effective locomotor policies. This approach highlights the potential of latent-dynamics MB-RL for enabling embodied soft robotic adaptive locomotion based solely on noisy sensor feedback.

---

