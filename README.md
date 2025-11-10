# 今日论文推荐 - 2025-11-10

共 18 篇论文

---

## 1. 论文ID: 2511.05449v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.05449v1.json'

---

## 2. Rethinking Metrics and Diffusion Architecture for 3D Point Cloud Generation

**论文链接:** [http://arxiv.org/abs/2511.05308v1](http://arxiv.org/abs/2511.05308v1)

**作者:** Matteo Bastico, David Ryckelynck, Laurent Corté, Yannick Tillier, Etienne Decencière

**发布时间:** 2025-11-07

**备注:** This paper has been accepted at International Conference on 3D Vision  (3DV) 2026

### GPT解析

### 总结

本文研究了3D点云生成模型的评估指标和生成方法，提出了改进的评估指标和新的生成模型架构，在ShapeNet数据集上取得了最先进的性能。

### 背景

3D点云已成为现代技术的基石，对复杂生成模型和可靠评估指标的需求呈指数级增长。

### 目的

暴露常用评估指标的局限性，提出更可靠的评估方法，并开发新的生成模型架构。

### 方法

指出基于Chamfer Distance的指标缺乏鲁棒性；提出在对齐样本后再计算距离；引入Density-Aware Chamfer Distance；提出新的Surface Normal Concordance指标；开发Diffusion Point Transformer模型架构。

### 主要发现

传统CD指标对缺陷缺乏鲁棒性；样本对齐和DCD能提高评估的一致性和鲁棒性；SNC指标通过比较点法线近似表面相似性；结合传统指标提供更全面的评估。

### 结论

提出的模型在ShapeNet数据集上优于之前的解决方案；在生成点云质量方面取得了新的最先进水平；代码已公开在GitHub上。

### 翻译

随着3D点云成为现代技术的基石，对复杂生成模型和可靠评估指标的需求呈指数级增长。在这项工作中，我们首先指出一些常用的评估生成点云的指标，特别是基于Chamfer Distance的指标，对缺陷缺乏鲁棒性，并且作为质量指标时无法捕捉几何保真度和局部形状一致性。我们进一步表明，在距离计算前引入样本对齐先验，并用Density-Aware Chamfer Distance替代CD，是确保点云生成模型评估指标一致性和鲁棒性的简单而必要的步骤。虽然现有指标主要关注直接比较3D欧几里得坐标，但我们提出了一个名为Surface Normal Concordance的新指标，通过比较估计的点法线来近似表面相似性。这个新指标与传统指标结合，能对生成样本的质量提供更全面的评估。最后，利用基于Transformer的点云分析模型的最新进展，如序列化块注意力，我们提出了一种用于生成高保真3D结构的新架构——Diffusion Point Transformer。我们在ShapeNet数据集上进行了广泛的实验和比较，表明我们的模型优于之前的解决方案，特别是在生成点云质量方面，取得了新的最先进水平。代码可在https://github.com/matteo-bastico/DiffusionPointTransformer获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决两个问题：一是3D点云生成模型的评估指标缺乏鲁棒性问题，二是现有生成模型架构因依赖体素化或下采样而导致局部结构细节损失的问题。这些问题在现实中非常重要，因为3D点云是自动驾驶、机器人和医疗等现代技术的核心，可靠的评估方法和高质量的生成模型对于推动这些领域的技术进步至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有评估指标的局限性，如图1所示传统指标对噪声和移位的敏感性，然后借鉴了密度感知Chamfer Distance(DCD)概念并结合质心对齐来改进评估。对于生成模型，作者受到DiT-3D的扩散结构和Point Transformer的点云处理架构启发，采用空间填充曲线序列化点云，并引入序列化块注意力和增强条件位置编码等创新技术。论文确实大量借鉴了现有工作，包括扩散模型、Transformer架构和点云处理技术，但进行了创新性整合和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想包括：1)评估指标方面，通过质心对齐确保评估稳定性，使用密度感知Chamfer Distance考虑点密度，引入表面一致性指标(SNC)通过比较点法线评估表面质量；2)生成模型方面，直接在原始点云上进行点级扩散避免细节损失，使用Transformer架构捕捉点间关系。整体流程为：点云序列化→随机排序→嵌入编码→DiPT块处理(条件位置编码→序列化块注意力→线性变换→特征调制→缩放和残差连接)→重复N次→输出处理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)评估指标：质心对齐方法、密度感知Chamfer Distance(DCD)、表面一致性指标(SNC)；2)生成模型：Diffusion Point Transformer(DiPT)直接处理原始点云、序列化块注意力机制、增强条件位置编码(xCPE)。相比之前工作的不同：传统评估指标对噪声敏感而改进后指标更鲁棒；SNC关注表面法线而非仅欧氏距离；DiPT直接处理原始点云而非依赖体素化或下采样，使用序列化块注意力提高效率，保留原始分辨率避免细节损失。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过改进3D点云生成模型的评估指标（质心对齐、密度感知Chamfer Distance和表面一致性指标）并提出一种直接在原始点云上进行点级扩散的Transformer架构（Diffusion Point Transformer），显著提高了生成点云的质量和评估的可靠性。'}


### 论文摘要

As 3D point clouds become a cornerstone of modern technology, the need for sophisticated generative models and reliable evaluation metrics has grown exponentially. In this work, we first expose that some commonly used metrics for evaluating generated point clouds, particularly those based on Chamfer Distance (CD), lack robustness against defects and fail to capture geometric fidelity and local shape consistency when used as quality indicators. We further show that introducing samples alignment prior to distance calculation and replacing CD with Density-Aware Chamfer Distance (DCD) are simple yet essential steps to ensure the consistency and robustness of point cloud generative model evaluation metrics. While existing metrics primarily focus on directly comparing 3D Euclidean coordinates, we present a novel metric, named Surface Normal Concordance (SNC), which approximates surface similarity by comparing estimated point normals. This new metric, when combined with traditional ones, provides a more comprehensive evaluation of the quality of generated samples. Finally, leveraging recent advancements in transformer-based models for point cloud analysis, such as serialized patch attention , we propose a new architecture for generating high-fidelity 3D structures, the Diffusion Point Transformer. We perform extensive experiments and comparisons on the ShapeNet dataset, showing that our model outperforms previous solutions, particularly in terms of quality of generated point clouds, achieving new state-of-the-art. Code available at https://github.com/matteo-bastico/DiffusionPointTransformer.

---

## 3. Implicit reconstruction from point cloud: an adaptive level-set-based semi-Lagrangian method

**论文链接:** [http://arxiv.org/abs/2511.05145v1](http://arxiv.org/abs/2511.05145v1)

**作者:** Silvia Preda, Matteo Semplice

**发布时间:** 2025-11-07

### GPT解析

### 总结

提出了一种基于水平集的半拉格朗日方法，用于从点云数据重建表面，使用分级自适应笛卡尔网格和曲率约束，生成高质量的隐式表面表示。

### 背景

从点云数据重建表面是一个重要问题，需要获得能够作为偏微分方程模型计算域的高质量隐式表示。

### 目的

获取对真实形状的隐式、高质量表示，作为偏微分方程模型的计算域。

### 方法

使用变分数学公式结合曲率约束，将问题重新表述为平流-扩散方程，采用半拉格朗日方案和局部高阶插值器求解，并利用四叉树和八叉树数据结构在表面附近生成精细网格。

### 主要发现

通过二维和三维的大量数值测试验证了该方法的有效性。

### 结论

该方法能够有效地从点云数据重建高质量表面，适用于复杂和演化拓扑的情况。

### 翻译

我们提出了一种基于水平集的半拉格朗日方法，用于在分级自适应笛卡尔网格上解决从点云数据重建表面的问题。目标是获得对真实形状的隐式、高质量表示，随后可以作为偏微分方程模型的计算域。数学公式是变分的，结合了曲率约束，在最小化表面积的同时，考虑重建表面与输入点云的距离的权重。在水平集框架内，这个问题被重新表述为平流-扩散方程，我们使用半拉格朗日方案结合局部高阶插值器来求解。基于水平集和半拉格朗日方法的特性，我们使用四叉树和八叉树数据结构来表示网格，并在零水平集（即重建表面界面）附近生成具有最精细分辨率的网格。描述了完整的表面重建工作流程，包括定位和重新初始化技术，以及处理复杂和演化拓扑的策略。展示了二维和三维的大量数值测试，以评估该方法的有效性。


### 论文摘要

We propose a level-set-based semi-Lagrangian method on graded adaptive Cartesian grids to address the problem of surface reconstruction from point clouds. The goal is to obtain an implicit, high-quality representation of real shapes that can subsequently serve as computational domain for partial differential equation models. The mathematical formulation is variational, incorporating a curvature constraint that minimizes the surface area while being weighted by the distance of the reconstructed surface from the input point cloud. Within the level set framework, this problem is reformulated as an advection-diffusion equation, which we solve using a semi-Lagrangian scheme coupled with a local high-order interpolator. Building on the features of the level set and semi-Lagrangian method, we use quadtree and octree data structures to represent the grid and generate a mesh with the finest resolution near the zero level set, i.e., the reconstructed surface interface. The complete surface reconstruction workflow is described, including localization and reinitialization techniques, as well as strategies to handle complex and evolving topologies. A broad set of numerical tests in two and three dimensions is presented to assess the effectiveness of the method.

---

## 4. Efficient representation of 3D spatial data for defense-related applications

**论文链接:** [http://arxiv.org/abs/2511.05109v1](http://arxiv.org/abs/2511.05109v1)

**作者:** Benjamin Kahl, Marcus Hebel, Michael Arens

**发布时间:** 2025-11-07

**DOI:** 10.1117/12.3069693

### GPT解析

### 总结

本文比较了地理空间传感器数据的传统表示方法和现代神经表示技术，提出了一种混合系统架构，结合传统方法的几何准确性和现代方法的视觉保真度。

### 背景

地理空间传感器数据对现代国防和安全至关重要，提供态势感知所需的3D信息，这些数据来自激光雷达传感器和光学相机等来源。

### 目的

对传统表示方法（点云、体素网格、三角形网格）与现代神经和隐式技术（NeRFs、3DGS）进行比较分析，评估它们在地理空间数据处理中的优缺点。

### 方法

通过比较传统表示方法和现代神经表示技术在几何准确性和视觉保真度方面的表现，评估它们在不同应用场景中的适用性。

### 主要发现

传统模型提供强大的几何准确性，适合功能性任务如视线分析和物理模拟；现代方法在产生高保真、照片级真实感的视觉效果方面表现出色，但往往缺乏几何可靠性。

### 结论

混合方法是最有前途的发展路径，结合传统网格支架以保证几何完整性，以及神经表示以提供视觉细节，在分层场景结构中进行管理以确保可扩展性和性能。

### 翻译

地理空间传感器数据对现代国防和安全至关重要，为态势感知提供了不可或缺的3D信息。这些数据来自激光雷达传感器和光学相机等来源，能够为操作环境创建详细模型。在本文中，我们对传统表示方法（如点云、体素网格和三角形网格）与现代神经和隐式技术（如神经辐射场NeRFs和3D高斯溅射3DGS）进行了比较分析。我们的评估揭示了一个基本权衡：传统模型提供强大的几何准确性，适合视线分析和物理模拟等功能性任务，而现代方法在产生高保真、照片级真实的视觉效果方面表现出色，但往往缺乏几何可靠性。基于这些发现，我们得出结论，混合方法是最有前途的发展路径。我们提出了一种系统架构，结合传统网格支架以保证几何完整性，以及像3DGS这样的神经表示以提供视觉细节，在分层场景结构中进行管理，以确保可扩展性和性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何高效表示3D空间数据，特别是针对国防应用的问题。在现实中，这非常重要，因为现代国防需要快速准确地感知环境信息，3D表示比2D能提供更丰富的信息（如路线优化、视线评估），而实时整合处理来自不同传感器（激光雷达、相机等）的海量异构数据集是一个重大技术挑战，原始传感器数据通常过于庞大或无结构，无法直接用于战术决策。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析传统表示方法（点云、体素网格、网格）和现代神经表示方法（NeRF、3DGS）的优缺点，发现传统模型提供强大几何准确性但视觉保真度不足，而现代方法视觉效果好但几何可靠性差。基于这一权衡分析，作者设计了一个混合方法，借鉴了现有工作中的多种技术：神经辐射场(NeRF)、3D高斯溅射(3DGS)、可微分渲染技术（如软光栅化器）以及表面对齐的高斯溅射(SuGaR)等方法，将它们整合到一个统一的框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是采用混合方法结合传统3D表示的几何准确性和现代神经表示的视觉保真度。整体流程分为四个阶段：1)数据收集阶段，从各种传感器获取原始数据；2)数据融合阶段，将异构传感器数据统一为时空格式，进行对齐和后处理；3)聚合阶段，构建混合模型，使用实例分割创建初始实例，训练每个实例的几何网格支架和3DGS表示，通过组合损失函数同时优化几何准确性和视觉保真度；4)使用阶段，支持视线分析、可视化、路线规划等多种应用。整个系统使用层次加速结构(BVH)管理大型环境，确保可扩展性和性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)混合表示架构，结合传统网格支架和现代神经表示；2)双目标优化方法，同时优化几何准确性和视觉保真度；3)基于实例的处理流程，每个实例独立训练其表示；4)传感器数据融合策略，有效整合激光雷达和相机数据。相比之前的工作，这篇论文专注于国防应用领域，提供系统级解决方案而非单一算法，强调实用性和可扩展性，并提供了针对国防相关应用场景的全面评估框架。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种创新的混合3D数据表示方法，通过结合传统网格支架的几何准确性和现代神经表示的视觉保真度，为国防应用提供了既可靠又高效的战场空间可视化解决方案。'}


### 论文摘要

Geospatial sensor data is essential for modern defense and security, offering indispensable 3D information for situational awareness. This data, gathered from sources like lidar sensors and optical cameras, allows for the creation of detailed models of operational environments. In this paper, we provide a comparative analysis of traditional representation methods, such as point clouds, voxel grids, and triangle meshes, alongside modern neural and implicit techniques like Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS). Our evaluation reveals a fundamental trade-off: traditional models offer robust geometric accuracy ideal for functional tasks like line-of-sight analysis and physics simulations, while modern methods excel at producing high-fidelity, photorealistic visuals but often lack geometric reliability. Based on these findings, we conclude that a hybrid approach is the most promising path forward. We propose a system architecture that combines a traditional mesh scaffold for geometric integrity with a neural representation like 3DGS for visual detail, managed within a hierarchical scene structure to ensure scalability and performance.

---

## 5. J-SGFT: Joint Spatial and Graph Fourier Domain Learning for Point Cloud Attribute Deblocking

**论文链接:** [http://arxiv.org/abs/2511.05047v1](http://arxiv.org/abs/2511.05047v1)

**作者:** Muhammad Talha, Qi Yang, Zhu Li, Anique Akhtar, Geert Van Der Auwera

**发布时间:** 2025-11-07

**备注:** Accepted to ICIP 2025 Workshop on Generative AI for World Simulations  and Communications & Celebrating 40 Years of Excellence in Education:  Honoring Professor Aggelos Katsaggelos, Sept. 2025, Alaska

### GPT解析

### 总结

提出了一种多尺度后处理框架，有效去除重建点云中的块状伪影，显著提高视觉质量且开销最小

### 背景

点云对于AR/VR和自动驾驶至关重要，但其体积大、不规则采样和稀疏性给压缩方案带来挑战。MPEG的GPCC方法虽成功降低比特率，但会在重建点云中引入明显的块状伪影

### 目的

开发一种新的多尺度后处理框架，有效去除重建点云中的块状伪影

### 方法

融合图傅里叶潜在属性表示与稀疏卷积和通道注意力的多尺度后处理框架，用于高效去除重建点云中的块状伪影

### 主要发现

与GPCC TMC13v14基准相比，在8iVFBv2数据集上实现Y通道18.81%和联合YUV 18.14%的BD-rate降低，显著提高视觉保真度

### 结论

所提出方法能有效去除重建点云中的块状伪影，在保持最小开销的同时显著提高视觉质量

### 翻译

点云对于AR/VR和自动驾驶至关重要，但其体积大、不规则采样和稀疏性给压缩方案带来挑战。MPEG的基于几何的点云压缩方法成功降低了比特率；然而，它们在重建的点云中引入了明显的块状伪影。我们引入了一种新颖的多尺度后处理框架，该框架融合了图傅里叶潜在属性表示与稀疏卷积和通道注意力，以高效去除重建点云中的块状伪影。与GPCC TMC13v14基准相比，我们的方法在8iVFBv2数据集上实现了Y通道18.81%的BD-rate降低和联合YUV上18.14%的BD-rate降低，以最小的开销提供了显著改进的视觉保真度。


### 论文摘要

Point clouds (PC) are essential for AR/VR and autonomous driving but challenge compression schemes with their size, irregular sampling, and sparsity. MPEG's Geometry-based Point Cloud Compression (GPCC) methods successfully reduce bitrate; however, they introduce significant blocky artifacts in the reconstructed point cloud. We introduce a novel multi-scale postprocessing framework that fuses graph-Fourier latent attribute representations with sparse convolutions and channel-wise attention to efficiently deblock reconstructed point clouds. Against the GPCC TMC13v14 baseline, our approach achieves BD-rate reduction of 18.81\% in the Y channel and 18.14\% in the joint YUV on the 8iVFBv2 dataset, delivering markedly improved visual fidelity with minimal overhead.

---

## 6. 4D Imaging in ISAC Systems: A Framework Based on 5G NR Downlink Signals

**论文链接:** [http://arxiv.org/abs/2511.04913v1](http://arxiv.org/abs/2511.04913v1)

**作者:** Haoyang Weng, Haisu Wu, Hong Ren, Cunhua Pan, Jiangzhou Wang

**发布时间:** 2025-11-07

**备注:** TVT

### GPT解析

### 总结

该研究提出了一种完全符合5G新空口(NR)协议的4D成像框架，通过端到端处理链路和Zoom-OMP算法实现高精度环境重建，为6G网络中的实际ISAC应用奠定基础。

### 背景

集成感知与通信(ISAC)已成为第六代(6G)无线网络的关键使能技术，支持频谱共享和硬件集成。除了增强通信外，ISAC还能实现高精度的环境重建和成像，这对自动驾驶和数字孪生等应用至关重要。

### 目的

开发一种与蜂窝系统兼容的4D成像框架，实现高精度的环境重建和成像，为6G网络中的实际ISAC应用奠定基础。

### 方法

提出完全符合5G新空口(NR)协议的4D成像框架；开发覆盖波形生成、回波处理和多基站点云融合的端到端处理链路；引入Zoom-OMP算法，一种用于高分辨率角度估计的从粗到细的稀疏恢复算法，以在降低计算成本的同时实现高精度。

### 主要发现

提出的框架相比传统基准方法具有更优的空间精度和重建质量，实现了稳健的4D成像性能。

### 结论

该研究为6G网络中的实际ISAC环境重建铺平了道路，展示了ISAC技术在6G网络中的实际应用潜力。

### 翻译

集成感知与通信(ISAC)已成为第六代(6G)无线网络的关键使能技术，支持频谱共享和硬件集成。除了增强通信外，ISAC还能实现高精度的环境重建和成像，这对自动驾驶和数字孪生等应用至关重要。本文提出了一种完全符合5G新空口(NR)协议的4D成像框架，确保与蜂窝系统的兼容性。具体而言，我们开发了一个覆盖波形生成、回波处理和多基站点云融合的端到端处理链路。此外，我们引入了Zoom-OMP，一种用于高分辨率角度估计的从粗到细的稀疏恢复算法，以在降低计算成本的同时实现高精度。仿真结果表明，与传统的基准方法相比，提出的框架具有更优的空间精度和重建质量，实现了稳健的4D成像性能，为6G网络中的实际ISAC环境重建铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何利用5G NR下行信号实现高精度4D成像的问题。现有研究大多依赖理想化的OFDM信号模型，忽略了5G NR标准的特定物理层结构，阻碍了实际部署；同时，成像分辨率与计算复杂度之间的权衡是实现实时应用的主要瓶颈。这个问题很重要，因为ISAC被认为是6G网络的关键技术，能将通信网络转变为无处不在的感知平台，实现高精度的环境重建，对自动驾驶、数字孪生等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有研究的局限性，包括对理想化OFDM信号模型的依赖和成像分辨率与计算复杂度之间的权衡问题。作者借鉴了早期工作通过分析反射毫米波信号中子载波相位偏移估计ToF的方法，以及后续研究结合传统2D-FFT和超分辨率算法的混合框架。在此基础上，作者创新性地设计了一个完全符合5G NR标准的4D成像框架，开发了端到端处理链，并提出了名为Zoom-OMP的粗到细稀疏恢复算法，在减少计算成本的同时实现高精度角度估计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用标准5G NR下行信号作为感知源，通过多基站协作感知架构利用空间多样性增强检测覆盖范围，并采用粗到细的稀疏恢复策略实现高精度角度估计。整体实现流程包括：1) 联合距离-速度估计：从回波信号提取散射体参数，通过时频域转换获得距离-多普勒图，检测峰值并计算距离和速度；2) 高分辨率角度估计：使用Zoom-OMP算法，先进行粗搜索再进行精细搜索，精确估计方位角和仰角；3) 多基站结果融合：定义全局坐标系，通过坐标变换将所有局部点云注册到全局坐标系中，聚合获得最终全球点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 完全符合5G NR标准的4D成像框架，确保与蜂窝系统兼容；2) 提出Zoom-OMP算法，采用粗到细稀疏恢复策略，显著降低计算复杂度；3) 多基站协作感知架构，解决单视点感知的遮挡问题。相比之前工作，本文框架更注重实际应用，完全符合5G NR标准而非理想化模型；Zoom-OMP算法在保持高成像质量的同时，计算复杂度远低于传统OMP和2D-MUSIC算法；系统级设计不仅关注算法，还包括完整的端到端处理链和多基站协作机制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于5G NR下行信号的完全兼容的ISAC 4D成像框架，通过创新的Zoom-OMP算法和多基站协作感知，实现了高精度、高效率的环境重建，为6G网络中的实际应用铺平了道路。'}


### 论文摘要

Integrated sensing and communication (ISAC) has emerged as a key enabler for sixth-generation (6G) wireless networks, supporting spectrum sharing and hardware integration. Beyond communication enhancement, ISAC also enables high-accuracy environment reconstruction and imaging, which are crucial for applications such as autonomous driving and digital twins. This paper proposes a 4D imaging framework fully compliant with the 5G New Radio (NR) protocol, ensuring compatibility with cellular systems. Specifically, we develop an end-to-end processing chain that covers waveform generation, echo processing, and multi-BS point cloud fusion. Furthermore, we introduce Zoom-OMP, a coarse-to-fine sparse recovery algorithm for high-resolution angle estimation that achieves high accuracy with reduced computational cost. The simulation results demonstrate that the proposed framework achieves robust 4D imaging performance with superior spatial accuracy and reconstruction quality compared to conventional benchmarks, paving the way for practical ISAC-enabled environment reconstruction in 6G networks.

---

## 7. Self-Supervised Implicit Attention Priors for Point Cloud Reconstruction

**论文链接:** [http://arxiv.org/abs/2511.04864v1](http://arxiv.org/abs/2511.04864v1)

**作者:** Kyle Fogarty, Chenyue Cai, Jing Yang, Zhilin Guo, Cengiz Öztireli

**发布时间:** 2025-11-06

**备注:** Accepted at 3DV 2026

### GPT解析

### 总结

该论文提出了一种从不规则点云恢复高质量表面的方法，通过隐式自先验技术直接从输入数据中学习形状特定的先验，并将其与隐式神经表示结合。该方法仅使用自监督损失进行训练，不需要外部数据，并通过隐式移动最小二乘法整合学习到的先验，能够在保留精细几何细节的同时正则化稀疏区域。

### 背景

从不规则点云恢复高质量表面是一个不适定问题，除非有强大的几何先验可用。

### 目的

开发一种能够从点云中恢复高质量表面的方法，无需依赖外部训练数据或强几何先验，同时保留精细几何细节并对数据退化具有鲁棒性。

### 方法

引入隐式自先验方法，联合训练可学习嵌入字典和隐式距离场，通过交叉注意力机制捕获形状的重复结构和长程相关性；使用自监督点云重建损失进行训练；通过自动微分采样提取密集点和法线；最后整合到稳健的隐式移动最小二乘公式中。

### 主要发现

该方法能够有效捕获和重用形状中的重复结构和长程相关性；仅使用自监督损失即可训练出高质量模型；能够保留输入数据中的精细几何细节，同时利用学习到的先验正则化稀疏区域；对常见数据退化具有鲁棒性。

### 结论

所提出的方法在生成高保真表面方面优于经典方法和基于学习的方法，具有卓越的细节保持能力和对数据退化的鲁棒性。

### 翻译

从不规则点云恢复高质量表面是一个不适定问题，除非有强大的几何先验可用。我们引入了一种隐式自先验方法，直接从输入点云中提炼形状特定的先验，并将其嵌入到隐式神经表示中。这是通过联合训练一个小的可学习嵌入字典和一个隐式距离场来实现的；在每一个查询位置，该字段通过交叉注意力机制关注字典，使网络能够捕获和重用形状中固有的重复结构和长程相关性。仅使用自监督的点云重建损失进行优化，我们的方法不需要外部训练数据。为了在保持输入保真度的同时有效整合这种学习到的先验，通过自动微分对训练好的场进行采样，以提取密集分布的点和分析法线。我们将得到的密集点云和相应的法线集成到稳健的隐式移动最小二乘公式中。我们表明这种混合策略保留了输入数据中的精细几何细节，同时利用学习到的先验来正则化稀疏区域。实验表明，我们的方法在生成具有卓越细节保持能力和对常见数据退化具有鲁棒性的高保真表面方面，优于经典方法和基于学习的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从离散、不规则的点云中恢复高质量表面的问题。这个问题在现实中非常重要，因为3D扫描设备经常产生不完整、有噪声或稀疏的点云数据，而高质量表面重建对于文化遗产数字化、工业检测、自动驾驶、医学成像和增强现实等应用至关重要。由于点云数据的不完整性，这个问题在数学上是不适定的，需要引入几何先验知识来约束重建过程。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统基于全局平滑先验的方法难以保留尖锐特征，而基于变形的方法如Point2Mesh虽能学习自相似性但受限于固定拓扑结构。作者从信号处理中获取灵感，将复杂信号表示为共享原子的稀疏组合，并结合隐式神经表示和注意力机制，设计出通过交叉注意力机制利用可学习字典捕获点云中自相似模式的方法。这种方法借鉴了Point2Mesh的自先验概念、神经隐式表示的灵活性以及注意力机制的非局部建模能力，但创新地将它们结合成一种新的自监督方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过交叉注意力机制利用可学习字典捕获点云中的自相似模式和重复结构，从而学习特定形状的几何先验，无需外部训练数据。整体流程分为两个阶段：第一阶段，训练一个MLP近似神经SDF场，使用可学习字典编码自先验，通过交叉注意力机制为每个查询点生成特征表示并预测SDF值；第二阶段，离散化学习到的几何场，使用鲁棒隐式移动最小二乘法(RIMLS)定义最终形状，结合MLP的表达能力和RIMLS的特征保持属性，产生全局一致且细节丰富的重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)自监督隐式框架，通过交叉注意力机制从输入点云学习特定形状的几何先验；2)将学习到的场与鲁棒隐式MLS相结合，产生准确、灵活的表面重建；3)在自相似形状上实现了最先进的性能。相比之前的工作，这篇论文不依赖外部训练数据，完全自监督；结合了隐式神经表示的灵活性和注意力机制的非局部建模能力；能捕获和重用重复结构和长程相关性；通过两阶段流程结合了学习先验和传统重建方法的优势；相比Point2Mesh等方法，不依赖固定拓扑结构，能处理任意形状。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种自监督的隐式注意力先验方法，通过可学习字典和交叉注意力机制从点云本身学习特定形状的几何模式，实现了高质量、细节丰富的表面重建，无需外部训练数据。'}


### 论文摘要

Recovering high-quality surfaces from irregular point cloud is ill-posed unless strong geometric priors are available. We introduce an implicit self-prior approach that distills a shape-specific prior directly from the input point cloud itself and embeds it within an implicit neural representation. This is achieved by jointly training a small dictionary of learnable embeddings with an implicit distance field; at every query location, the field attends to the dictionary via cross-attention, enabling the network to capture and reuse repeating structures and long-range correlations inherent to the shape. Optimized solely with self-supervised point cloud reconstruction losses, our approach requires no external training data. To effectively integrate this learned prior while preserving input fidelity, the trained field is then sampled to extract densely distributed points and analytic normals via automatic differentiation. We integrate the resulting dense point cloud and corresponding normals into a robust implicit moving least squares (RIMLS) formulation. We show this hybrid strategy preserves fine geometric details in the input data, while leveraging the learned prior to regularize sparse regions. Experiments show that our method outperforms both classical and learning-based approaches in generating high-fidelity surfaces with superior detail preservation and robustness to common data degradations.

---

## 8. iFlyBot-VLM Technical Report

**论文链接:** [http://arxiv.org/abs/2511.04976v1](http://arxiv.org/abs/2511.04976v1)

**作者:** Xin Nie, Zhiyuan Cheng, Yuan Zhang, Chao Ji, Jiajia Wu, Yuhan Zhang, Jia Pan

**发布时间:** 2025-11-07

### GPT解析

### 总结

iFlyBot-VLM是一个通用的视觉语言模型，用于提升具身智能领域。它旨在弥合高维环境感知与低级机器人运动控制之间的跨模态语义鸿沟。

### 背景

具身智能领域需要处理复杂的视觉和空间信息，并将其转化为机器人可理解的语言。

### 目的

创建一个可扩展和可推广的基础模型，用于具身AI，促进从专门的面向任务的系统向通用的、具有认知能力的智能体发展。

### 方法

该模型通过将复杂的视觉和空间信息抽象为与身体无关且可转移的操作语言，实现跨不同机器人平台的感知-动作闭环协调。其架构被系统设计以实现四个关键功能能力：空间理解和度量推理；交互式目标定位；动作抽象和控制参数生成；任务规划和技能排序。

### 主要发现

在10个当前主流的具身智能相关VLM基准数据集（如Blink和Where2Place）上进行了评估，取得了最佳性能，同时保留了模型的通用能力。

### 结论

iFlyBot-VLM代表了具身智能领域的重要进展，研究团队将公开发布训练数据和模型权重，以促进该领域的进一步研究和开发。

### 翻译

我们介绍了iFlyBot-VLM，这是一个通用的视觉语言模型，用于提升具身智能领域。iFlyBot-VLM的核心目标是弥合高维环境感知与低级机器人运动控制之间的跨模态语义鸿沟。为此，该模型将复杂的视觉和空间信息抽象为与身体无关且可转移的操作语言，从而能够在多样化的机器人平台上实现无缝的感知-动作闭环协调。iFlyBot-VLM的架构经过系统设计，以实现具身智能所需的四个关键功能能力：1）空间理解和度量推理；2）交互式目标定位；3）动作抽象和控制参数生成；4）任务规划和技能排序。我们将iFlyBot-VLM视为具身AI的一个可扩展和可推广的基础模型，促进从专门的面向任务的系统向通用的、具有认知能力的智能体发展。我们在10个当前主流的具身智能相关VLM基准数据集（如Blink和Where2Place）上进行了评估，在保持模型通用能力的同时取得了最佳性能。我们将公开发布训练数据和模型权重，以促进具身智能领域的进一步研究和开发。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决高维环境感知与低级机器人运动控制之间的跨模态语义鸿沟问题，以及现有视觉语言模型在具身智能领域面临的'动作记忆'现象和泛化能力弱的问题。这个问题很重要，因为它使机器人能够真正理解物理世界并执行复杂任务，而不是仅限于数字世界中的感知任务，是实现通用机器人的关键挑战。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有VLM在具身智能应用的局限性，认识到需要抽象环境信息为机器人可理解的'操作语言'。模型架构借鉴了主流的'ViT-Projector-LLM'三阶段范式，特别是InternVL3的设计，同时在位置编码层进行了创新。数据构建整合了多个现有数据集，如ScanNet、AgiBotWorld等，并进行了针对性的处理和扩充。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将复杂视觉和空间信息抽象为独立于机器人本体且可转移的'操作语言'，实现感知-动作闭环协调。整体流程包括：采用三阶段'ViT-Projector-LLM'架构；创新性地使用维度扩展位置嵌入(DEPE)增强空间感知；构建约380万样本的多源数据集；通过监督微调和链式思维引导进行训练；在多个基准数据集上进行评估。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：提出'操作语言'概念；创新维度扩展位置嵌入(DEPE)方法；构建全面的多源数据集；设计四个核心能力模块；引入链式思维引导推理。相比之前工作，iFlyBot-VLM更专注于具身智能领域，解决了跨模态语义鸿沟，在多个基准测试中表现更优，如Where2Place上得分70.23，Refspatial-bench上得分51.5，显著优于现有模型。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': "iFlyBot-VLM通过创新的'操作语言'概念和全面的数据集构建，有效解决了高维环境感知与低级机器人运动控制之间的跨模态语义鸿沟，显著提升了机器人在具身智能任务中的空间理解、目标定位、动作抽象和任务规划能力。"}


### 论文摘要

We introduce iFlyBot-VLM, a general-purpose Vision-Language Model (VLM) used to improve the domain of Embodied Intelligence. The central objective of iFlyBot-VLM is to bridge the cross-modal semantic gap between high-dimensional environmental perception and low-level robotic motion control. To this end, the model abstracts complex visual and spatial information into a body-agnostic and transferable Operational Language, thereby enabling seamless perception-action closed-loop coordination across diverse robotic platforms. The architecture of iFlyBot-VLM is systematically designed to realize four key functional capabilities essential for embodied intelligence: 1) Spatial Understanding and Metric Reasoning; 2) Interactive Target Grounding; 3) Action Abstraction and Control Parameter Generation; 4) Task Planning and Skill Sequencing. We envision iFlyBot-VLM as a scalable and generalizable foundation model for embodied AI, facilitating the progression from specialized task-oriented systems toward generalist, cognitively capable agents. We conducted evaluations on 10 current mainstream embodied intelligence-related VLM benchmark datasets, such as Blink and Where2Place, and achieved optimal performance while preserving the model's general capabilities. We will publicly release both the training data and model weights to foster further research and development in the field of Embodied Intelligence.

---

## 9. SiamMM: A Mixture Model Perspective on Deep Unsupervised Learning

**论文链接:** [http://arxiv.org/abs/2511.05462v1](http://arxiv.org/abs/2511.05462v1)

**作者:** Xiaodong Wang, Jing Huang, Kevin J Liang

**发布时间:** 2025-11-07

### GPT解析

### 总结

该研究建立了无监督聚类方法与统计学混合模型之间的联系，开发了名为SiamMM的新模型，该方法在自监督学习基准测试中表现优异，学习到的聚类与真实标签高度相似，同时揭示了数据集中可能存在的错误标记问题。

### 背景

近期研究表明基于聚类的方法在自监督和无监督学习中是有效的，然而聚类的应用通常是启发式的，最佳方法尚不清楚。

### 目的

建立无监督聚类方法与统计学中的经典混合模型之间的联系，改进现有聚类方法。

### 方法

通过建立聚类方法与统计学混合模型的框架，显著改进现有聚类方法，开发名为SiamMM的新模型。

### 主要发现

SiamMM方法在各种自监督学习基准测试中达到了最先进的性能，学习到的聚类与未见过的真实标签高度相似。

### 结论

对学习到的聚类进行检查发现它们与真实标签高度相似，同时揭示了数据集中可能存在的错误标记实例。

### 翻译

最近的研究已经证明了基于聚类的方法在自监督和无监督学习中的有效性。然而，聚类的应用通常是启发式的，最佳方法尚不清楚。在这项工作中，我们建立了这些无监督聚类方法与统计学中的经典混合模型之间的联系。通过这个框架，我们展示了这些聚类方法的显著改进，开发了一个名为SiamMM的新模型。我们的方法在各种自监督学习基准测试中取得了最先进的性能。对学习到的聚类进行检查发现，它们与未见过的真实标签高度相似，揭示了可能存在的错误标记实例。


### 论文摘要

Recent studies have demonstrated the effectiveness of clustering-based approaches for self-supervised and unsupervised learning. However, the application of clustering is often heuristic, and the optimal methodology remains unclear. In this work, we establish connections between these unsupervised clustering methods and classical mixture models from statistics. Through this framework, we demonstrate significant enhancements to these clustering methods, leading to the development of a novel model named SiamMM. Our method attains state-of-the-art performance across various self-supervised learning benchmarks. Inspection of the learned clusters reveals a strong resemblance to unseen ground truth labels, uncovering potential instances of mislabeling.

---

## 10. Deep Progressive Training: scaling up depth capacity of zero/one-layer models

**论文链接:** [http://arxiv.org/abs/2511.04981v1](http://arxiv.org/abs/2511.04981v1)

**作者:** Zhiqi Bu

**发布时间:** 2025-11-07

### GPT解析

### 总结

本文研究了深度学习中模型深度的扩展问题，提出了一种零/一层渐进训练方法，以在计算效率和模型性能之间实现最佳权衡。

### 背景

深度学习中模型深度是一把双刃剑：更深层的模型能获得更高的准确性，但需要更高的计算成本。

### 目的

为了有效地大规模训练模型，研究通过渐进式训练来扩展大模型的深度，探索优化理论和特征学习视角下的模型扩展。

### 方法

提出了一种零/一层渐进训练方法，用于在计算和损失之间实现最佳权衡。研究内容包括新层的初始化、超参数传递、学习率调度和模型扩展时机的选择。

### 主要发现

在GPT2上应用零/一层渐进训练，与完全训练的60层、70亿参数模型相比，可以节省约80%的计算量，或等效地加速约5倍，同时实现几乎相同的损失。

### 结论

渐进式训练是一种有效的方法，可以在训练过程中扩展模型容量，显著减少计算量，同时几乎没有性能下降。

### 翻译

在深度学习中，模型深度是一把双刃剑：更深层的模型能获得更高的准确性，但需要更高的计算成本。为了有效地大规模训练模型，渐进式训练是一种有效的策略，它在训练过程中扩展模型容量，从而显著减少计算量，同时几乎没有性能下降。在本工作中，我们从优化理论和特征学习的角度研究了大型模型的深度扩展，提供了关于新层初始化、超参数传递、学习率调度和模型扩展时机的见解。具体而言，我们提出了零/一层渐进训练，以在计算和损失之间实现最佳权衡。例如，与完全训练的60层、70亿参数模型相比，在GPT2上应用零/一层渐进训练可以节省约80%的计算量，或等效地加速约5倍，同时实现几乎相同的损失。


### 论文摘要

Model depth is a double-edged sword in deep learning: deeper models achieve higher accuracy but require higher computational cost. To efficiently train models at scale, an effective strategy is the progressive training, which scales up model capacity during training, hence significantly reducing computation with little to none performance degradation. In this work, we study the depth expansion of large models through the lens of optimization theory and feature learning, offering insights on the initialization of new layers, hyperparameter transfer, learning rate schedule, and timing of model expansion. Specifically, we propose zero/one-layer progressive training for the optimal tradeoff between computation and loss. For example, zero/one-layer progressive training on GPT2 can save $\approx 80\%$ compute, or equivalently accelerate $\approx 5\times$ while achieving almost the same loss, compared to to a fully trained 60-layer model with 7B parameters.

---

## 11. ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining

**论文链接:** [http://arxiv.org/abs/2511.05245v1](http://arxiv.org/abs/2511.05245v1)

**作者:** Xincheng Yao, Yan Luo, Zefeng Qian, Chongyang Zhang

**发布时间:** 2025-11-07

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了一种专门针对异常检测任务的预训练表征学习框架，解决了当前基于ImageNet预训练的特征网络在异常检测任务中的局限性。

### 背景

当前主流和最先进的异常检测方法主要依赖于ImageNet预训练产生的特征网络，但ImageNet预训练过程与异常检测目标不匹配，且自然图像与工业图像数据存在分布偏移，导致ImageNet预训练特征对AD任务次优。

### 目的

为了进一步促进AD领域发展，提出专门针对AD任务的预训练表征学习框架，学习鲁棒性和判别性更强的预训练表征用于工业异常检测。

### 方法

提出角度和方向导向的对比损失，同时最大化正常和异常特征之间的角度大小和范数差异；在大型AD数据集RealIAD上进行预训练以避免分布偏移；基于类可泛化的残差特征学习预训练AD表征以缓解预训练数据和下游AD数据集之间的潜在偏移。

### 主要发现

在五个基于嵌入的AD方法和五个AD数据集、五个骨干网络上的广泛实验一致表明，作者提出的预训练特征具有优越性。

### 结论

专门为异常检测任务设计的预训练表征能有效提升异常检测性能，代码已公开在GitHub上。

### 翻译

当前主流和最先进的异常检测(AD)方法主要建立在ImageNet预训练产生的预训练特征网络基础上。然而，无论监督还是自监督预训练，ImageNet上的预训练过程都与异常检测的目标不匹配（即自然图像预训练不旨在区分正常和异常）。此外，自然图像和AD场景中的工业图像数据通常存在分布偏移。这两个问题可能导致ImageNet预训练特征对AD任务次优。为了进一步促进AD领域的发展，专门针对AD任务的预训练表征是迫切且有价值的。为此，我们提出了一种新颖的AD表征学习框架，专门设计用于学习工业异常检测的鲁棒性和判别性预训练表征。具体来说，紧密围绕异常检测的目标（即关注正常和异常之间的差异），我们提出了角度和方向导向的对比损失，同时最大化正常和异常特征之间的角度大小和范数差异。为了避免从自然图像到AD图像的分布偏移，我们的预训练在大型AD数据集RealIAD上进行。为了进一步缓解预训练数据和下游AD数据集之间的潜在偏移，我们基于类可泛化的表征、残差特征来学习预训练的AD表征。对于评估，基于五个基于嵌入的AD方法，我们简单地将它们的原始特征替换为我们的预训练表征。在五个AD数据集和五个骨干网络上的广泛实验一致表明我们预训练特征的优越性。代码可在https://github.com/xcyao00/ADPretrain获取。


### 论文摘要

The current mainstream and state-of-the-art anomaly detection (AD) methods are substantially established on pretrained feature networks yielded by ImageNet pretraining. However, regardless of supervised or self-supervised pretraining, the pretraining process on ImageNet does not match the goal of anomaly detection (i.e., pretraining in natural images doesn't aim to distinguish between normal and abnormal). Moreover, natural images and industrial image data in AD scenarios typically have the distribution shift. The two issues can cause ImageNet-pretrained features to be suboptimal for AD tasks. To further promote the development of the AD field, pretrained representations specially for AD tasks are eager and very valuable. To this end, we propose a novel AD representation learning framework specially designed for learning robust and discriminative pretrained representations for industrial anomaly detection. Specifically, closely surrounding the goal of anomaly detection (i.e., focus on discrepancies between normals and anomalies), we propose angle- and norm-oriented contrastive losses to maximize the angle size and norm difference between normal and abnormal features simultaneously. To avoid the distribution shift from natural images to AD images, our pretraining is performed on a large-scale AD dataset, RealIAD. To further alleviate the potential shift between pretraining data and downstream AD datasets, we learn the pretrained AD representations based on the class-generalizable representation, residual features. For evaluation, based on five embedding-based AD methods, we simply replace their original features with our pretrained representations. Extensive experiments on five AD datasets and five backbones consistently show the superiority of our pretrained features. The code is available at https://github.com/xcyao00/ADPretrain.

---

## 12. Causal Structure and Representation Learning with Biomedical Applications

**论文链接:** [http://arxiv.org/abs/2511.04790v1](http://arxiv.org/abs/2511.04790v1)

**作者:** Caroline Uhler, Jiaqi Zhang

**发布时间:** 2025-11-06

**备注:** This article has successfully completed peer review and will appear  in the Proceedings of the International Congress of Mathematicians 2026. Both  authors contributed equally to this work

### GPT解析

### 总结

该论文探讨了表示学习与因果推理的结合，特别是在多模态数据背景下，提出了一种统计和计算框架来解决因果结构和表示学习问题。

### 背景

大规模数据收集有望更好地理解复杂现象并做出更好的决策。表示学习已成为深度学习应用的关键驱动力，因为它允许学习捕获数据重要属性的潜在空间，而无需监督注释。

### 目的

开发一种统计和计算框架，用于因果结构和表示学习，解决基本的生物医学问题，包括如何有效利用观察性和扰动性数据进行因果发现、如何利用多模态视图学习因果变量，以及如何设计最优扰动。

### 方法

提出了一种结合表示学习和因果推理的统计和计算框架，利用日益增长的多模态数据（观察性和扰动性、基于成像和基于测序的、在不同生物水平上的数据）。

### 主要发现

表示学习在预测任务中非常成功，但在因果任务中可能失败，这表明需要将表示学习与因果推理相结合。多模态数据的可用性为这种结合提供了机会。

### 结论

表示学习与因果推理的结合，特别是在多模态数据背景下，为理解和预测复杂现象提供了新的机会，特别是在生物医学领域。

### 翻译

大规模数据收集有望更好地理解复杂现象，并最终做出更好的决策。表示学习已成为深度学习应用的关键驱动力，因为它允许学习捕获数据重要属性的潜在空间，而无需任何监督注释。尽管表示学习在预测任务中取得了巨大成功，但在因果任务（包括预测扰动/干预的效果）中可能会彻底失败。这需要将表示学习与因果推理相结合。在这方面，一个令人兴奋的机会来自于多模态数据（观察性和扰动性、基于成像和基于测序的、在单细胞水平、组织水平和生物体水平）的日益增长的可获得性。我们概述了一个统计和计算框架，用于因果结构和表示学习，其动机是基本的生物医学问题：如何有效地利用观察性和扰动性数据对观察到的因果变量进行因果发现；如何利用系统的多模态视图学习因果变量；以及如何设计最优扰动。


### 论文摘要

Massive data collection holds the promise of a better understanding of complex phenomena and, ultimately, better decisions. Representation learning has become a key driver of deep learning applications, as it allows learning latent spaces that capture important properties of the data without requiring any supervised annotations. Although representation learning has been hugely successful in predictive tasks, it can fail miserably in causal tasks including predicting the effect of a perturbation/intervention. This calls for a marriage between representation learning and causal inference. An exciting opportunity in this regard stems from the growing availability of multi-modal data (observational and perturbational, imaging-based and sequencing-based, at the single-cell level, tissue-level, and organism-level). We outline a statistical and computational framework for causal structure and representation learning motivated by fundamental biomedical questions: how to effectively use observational and perturbational data to perform causal discovery on observed causal variables; how to use multi-modal views of the system to learn causal variables; and how to design optimal perturbations.

---

## 13. DGTN: Graph-Enhanced Transformer with Diffusive Attention Gating Mechanism for Enzyme DDG Prediction

**论文链接:** [http://arxiv.org/abs/2511.05483v1](http://arxiv.org/abs/2511.05483v1)

**作者:** Abigail Lin

**发布时间:** 2025-11-07

### GPT解析

### 总结

本研究提出了DGTN(Diffused Graph-Transformer Network)架构，通过扩散机制共同学习图神经网络的结构先验和transformer注意力，有效捕捉了蛋白质局部结构几何与全局序列模式之间的复杂耦合关系，在酶热力学稳定性预测任务上取得了最先进性能。

### 背景

预测氨基酸突变对酶热力学稳定性(DDG)是蛋白质工程和药物设计的基础。虽然最近的深度学习方法显示出潜力，但它们通常独立处理序列和结构信息，无法捕捉局部结构几何与全局序列模式之间的复杂耦合关系。

### 目的

开发一种能够有效整合蛋白质序列和结构信息的方法，以更准确地预测氨基酸突变对酶热力学稳定性的影响。

### 方法

提出DGTN架构，通过扩散机制共同学习图神经网络(GNN)的结构先验和transformer注意力权重。关键创新是双向扩散过程：(1)GNN衍生的结构嵌入通过可学习的扩散核引导transformer注意力；(2)transformer表示通过注意力调节的图更新来改进GNN消息传递。提供了严格的数学分析证明该共同学习方案的优越性。

### 主要发现

在ProTherm和SKEMPI基准测试上，DGTN取得了最先进性能(Pearson Rho = 0.87, RMSE = 1.21 kcal/mol)，比最佳基线提高了6.2%。消融研究表明扩散机制对相关性的贡献为4.8个百分点。理论分析证明扩散的注意力收敛到最优结构-序列耦合，收敛速率为O(1/sqrt(T))，其中T是扩散步数。

### 结论

该工作建立了一个通过可学习扩散整合异构蛋白质表示的原则性框架，为蛋白质工程和药物设计提供了新的工具。

### 翻译

预测氨基酸突变对酶热力学稳定性(DDG)的影响是蛋白质工程和药物设计的基础。虽然最近的深度学习方法显示出前景，但它们通常独立处理序列和结构信息，无法捕捉局部结构几何与全局序列模式之间的复杂耦合关系。我们提出了DGTN(Diffused Graph-Transformer Network)，一种新颖的架构，通过扩散机制共同学习图神经网络(GNN)的结构先验和transformer注意力权重。我们的关键创新是双向扩散过程：(1)GNN衍生的结构嵌入通过可学习的扩散核引导transformer注意力；(2)transformer表示通过注意力调节的图更新来改进GNN消息传递。我们提供了严格的数学分析，证明这种共同学习方案比独立处理具有更好的近似界限。在ProTherm和SKEMPI基准测试上，DGTN取得了最先进性能(Pearson Rho = 0.87, RMSE = 1.21 kcal/mol)，比最佳基线提高了6.2%。消融研究表明扩散机制对相关性的贡献为4.8个百分点。我们的理论分析证明扩散的注意力收敛到最优结构-序列耦合，收敛速率为O(1/sqrt(T))，其中T是扩散步数。这项工作建立了一个通过可学习扩散整合异构蛋白质表示的原则性框架。


### 论文摘要

Predicting the effect of amino acid mutations on enzyme thermodynamic stability (DDG) is fundamental to protein engineering and drug design. While recent deep learning approaches have shown promise, they often process sequence and structure information independently, failing to capture the intricate coupling between local structural geometry and global sequential patterns. We present DGTN (Diffused Graph-Transformer Network), a novel architecture that co-learns graph neural network (GNN) weights for structural priors and transformer attention through a diffusion mechanism. Our key innovation is a bidirectional diffusion process where: (1) GNN-derived structural embeddings guide transformer attention via learnable diffusion kernels, and (2) transformer representations refine GNN message passing through attention-modulated graph updates. We provide rigorous mathematical analysis showing this co-learning scheme achieves provably better approximation bounds than independent processing. On ProTherm and SKEMPI benchmarks, DGTN achieves state-of-the-art performance (Pearson Rho = 0.87, RMSE = 1.21 kcal/mol), with 6.2% improvement over best baselines. Ablation studies confirm the diffusion mechanism contributes 4.8 points to correlation. Our theoretical analysis proves the diffused attention converges to optimal structure-sequence coupling, with convergence rate O(1/sqrt(T) ) where T is diffusion steps. This work establishes a principled framework for integrating heterogeneous protein representations through learnable diffusion.

---

## 14. Linking Warm Dark Matter to Merger Tree Histories via Deep Learning Networks

**论文链接:** [http://arxiv.org/abs/2511.05367v1](http://arxiv.org/abs/2511.05367v1)

**作者:** Ilem Leisher, Paul Torrey, Alex M. Garcia, Jonah C. Rose, Francisco Villaescusa-Navarro, Zachary Lubberts, Arya Farahi, Stephanie O'Neil, Xuejian Shen, Olivia Mostow, Nitya Kallivayalil, Dhruv Zimmerman, Desika Narayanan, Mark Vogelsberger

**发布时间:** 2025-11-07

**备注:** 20 pages, 9 figures, submitted to ApJ

### GPT解析

### 总结

本研究利用深度学习方法从暗物质晕的合并树结构中推断温暗物质粒子质量和反馈参数，证明合并树本身包含宇宙学参数信息。

### 背景

暗物质晕在宇宙中通过一系列合并事件形成层次结构，宇宙模拟可将这些合并表示为图状'树'结构。已知这些合并树对宇宙学模拟参数敏感，但作为暗物质结构，它们对暗物质模型的敏感性仍不清楚。

### 目的

研究使用深度学习方法训练合并树以推断温暗物质(WDM)粒子质量的可行性，并探索将相同方法应用于超新星和活动星系核反馈参数的推断。

### 方法

将1,024个放大模拟中的合并树组织成图结构，节点表示星系合并历史，边表示遗传链接。变化节点特征复杂性，训练图神经网络(GNN)使用合并树图表示作为输入来预测WDM质量和反馈参数。

### 主要发现

GNN能成功预测WDM粒子质量(R²从0.07到0.95)，预测效果取决于图的复杂性和节点特征。同样方法成功应用于推断超新星和活动星系核反馈参数。即使没有任何节点特征，GNN也能从合并树结构推断WDM质量，表明合并树结构本身继承了形成它们的模拟的宇宙学参数信息。

### 结论

合并树的结构包含关于宇宙学参数的信息，即使不包含节点特征。图神经网络可以有效地从合并树中推断暗物质模型参数和反馈参数。

### 翻译

暗物质晕在宇宙中通过一系列合并事件形成层次结构。宇宙模拟可以将这一系列合并表示为类似图的'树'结构。先前的工作表明这些合并树对宇宙学模拟参数敏感，但作为暗物质结构，它们对暗物质模型的敏感性仍然未知。在这项工作中，我们研究了使用深度学习方法训练合并树以从DREAMS模拟套件中推断温暗物质(WDM)粒子质量的可行性。我们将1,024个放大模拟中的合并树组织成图，节点代表星系的合并历史，边表示遗传链接。我们变化图中包含的节点特征的复杂性，从单个节点特征到多个星系属性(如晕质量、恒星形成率等)。我们训练图神经网络(GNN)使用合并树的图表示作为输入来预测WDM质量。我们发现GNN可以预测WDM粒子的质量(R²从0.07到0.95)，成功与否取决于图的复杂性和节点特征。我们将相同的方法扩展到超新星和活动星系核反馈参数A_SN1、A_SN2和A_AGN，成功推断出超新星参数。GNN甚至可以在没有任何节点特征的情况下从合并树历史推断WDM质量，表明合并树的结构本身就继承了它们形成的模拟的宇宙学参数信息。


### 论文摘要

Dark matter (DM) halos form hierarchically in the Universe through a series of merger events. Cosmological simulations can represent this series of mergers as a graph-like ``tree'' structure. Previous work has shown these merger trees are sensitive to cosmology simulation parameters, but as DM structures, the outstanding question of their sensitivity to DM models remains unanswered. In this work, we investigate the feasibility of deep learning methods trained on merger trees to infer Warm Dark Matter (WDM) particles masses from the DREAMS simulation suite. We organize the merger trees from 1,024 zoom-in simulations into graphs with nodes representing the merger history of galaxies and edges denoting hereditary links. We vary the complexity of the node features included in the graphs ranging from a single node feature up through an array of several galactic properties (e.g., halo mass, star formation rate, etc.). We train a Graph Neural Network (GNN) to predict the WDM mass using the graph representation of the merger tree as input. We find that the GNN can predict the mass of the WDM particle ($R^2$ from 0.07 to 0.95), with success depending on the graph complexity and node features. We extend the same methods to supernovae and active galactic nuclei feedback parameters $A_\text{SN1}$, $A_\text{SN2}$, and $A_\text{AGN}$, successfully inferring the supernovae parameters. The GNN can even infer the WDM mass from merger tree histories without any node features, indicating that the structure of merger trees alone inherits information about the cosmological parameters of the simulations from which they form.

---

## 15. No One-Model-Fits-All: Uncovering Spatio-Temporal Forecasting Trade-offs with Graph Neural Networks and Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.05179v1](http://arxiv.org/abs/2511.05179v1)

**作者:** Ragini Gupta, Naman Raina, Bo Chen, Li Chen, Claudiu Danilov, Josh Eckhardt, Keyshla Bernard, Klara Nahrstedt

**发布时间:** 2025-11-07

### GPT解析

### 总结

本研究系统分析了不同传感器部署密度和采样间隔条件下各类预测模型的性能表现，发现时空图神经网络(STGNNs)在稀疏部署和中等采样率下表现优异，而时序基础模型(TSFMs)在高频采样下更具竞争力，多变量TSF模型Moirai通过学习跨传感器依赖关系整体性能最佳。

### 背景

现代物联网环境感知系统产生大量时空数据用于预测任务，现有边缘数据优化技术忽视了采样频率和空间覆盖变化对模型性能的影响。采样频率、空间覆盖与预测模型架构间的相互作用尚未得到充分探索。

### 目的

研究不同传感器节点密度和采样间隔条件下各类预测模型的性能表现，为构建时空系统中的高效预测流水线提供实用见解。

### 方法

使用无线传感器网络中的真实温度数据，测试多种预测模型包括经典模型(VAR)、神经网络(GRU, Transformer)、时空图神经网络(STGNNs)和时序基础模型(TSFMs: Chronos, Moirai, TimesFM)，通过变化传感器节点密度和采样间隔评估模型性能。

### 主要发现

STGNNs在传感器部署稀疏且采样率适中时有效，利用空间相关性补偿有限覆盖；TSFMs在高频采样下表现好但空间覆盖减少时性能下降；多变量TSF模型Moirai通过原生学习跨传感器依赖关系，性能超越所有其他模型。

### 结论

不同预测模型在不同传感器部署和采样条件下各有优势，合理选择模型和采样策略可构建高效的时空预测系统。所有代码已开源以确保研究结果可复现。

### 翻译

现代物联网环境感知部署产生大量时空数据，以支持预测等下游任务，通常由机器学习模型驱动。虽然现有的过滤和战略部署技术优化了边缘收集的数据量，但它们忽略了采样频率和空间覆盖的变化如何影响下游模型性能。在许多预测模型中，通过提供更广泛的空间上下文，来自额外传感器的数据可以通过去噪预测。这种采样频率、空间覆盖与不同预测模型架构之间的相互作用仍未得到充分探索。这项工作提出了一项对预测模型的系统研究 - 经典模型(VAR)、神经网络(GRU, Transformer)、时空图神经网络(STGNNs)以及在不同空间传感器节点密度和采样间隔下的时序基础模型(TSFMs: Chronos, Moirai, TimesFM)，使用无线传感器网络中的真实温度数据。我们的结果表明，当传感器部署稀疏且采样率适中时，STGNNs有效，它们通过编码的图结构利用空间相关性来补偿有限的覆盖范围。相比之下，TSFMs在高频率下表现具有竞争力，但当来自相邻传感器的空间覆盖减少时性能下降。关键的是，多变量TSF模型Moirai通过原生学习跨传感器依赖关系，性能优于所有模型。这些发现为构建时空系统中的高效预测流水线提供了实用见解。所有模型配置、训练、数据集和日志代码均已开源以确保可复现性：https://github.com/UIUC-MONET-Projects/Benchmarking-Spatiotemporal-Forecast-Models


### 论文摘要

Modern IoT deployments for environmental sensing produce high volume spatiotemporal data to support downstream tasks such as forecasting, typically powered by machine learning models. While existing filtering and strategic deployment techniques optimize collected data volume at the edge, they overlook how variations in sampling frequencies and spatial coverage affect downstream model performance. In many forecasting models, incorporating data from additional sensors denoise predictions by providing broader spatial contexts. This interplay between sampling frequency, spatial coverage and different forecasting model architectures remain underexplored. This work presents a systematic study of forecasting models - classical models (VAR), neural networks (GRU, Transformer), spatio-temporal graph neural networks (STGNNs), and time series foundation models (TSFMs: Chronos Moirai, TimesFM) under varying spatial sensor nodes density and sampling intervals using real-world temperature data in a wireless sensor network. Our results show that STGNNs are effective when sensor deployments are sparse and sampling rate is moderate, leveraging spatial correlations via encoded graph structure to compensate for limited coverage. In contrast, TSFMs perform competitively at high frequencies but degrade when spatial coverage from neighboring sensors is reduced. Crucially, the multivariate TSFM Moirai outperforms all models by natively learning cross-sensor dependencies. These findings offer actionable insights for building efficient forecasting pipelines in spatio-temporal systems. All code for model configurations, training, dataset, and logs are open-sourced for reproducibility: https://github.com/UIUC-MONET-Projects/Benchmarking-Spatiotemporal-Forecast-Models

---

## 16. Peptide2Mol: A Diffusion Model for Generating Small Molecules as Peptide Mimics for Targeted Protein Binding

**论文链接:** [http://arxiv.org/abs/2511.04984v1](http://arxiv.org/abs/2511.04984v1)

**作者:** Xinheng He, Yijia Zhang, Haowei Lin, Xingang Peng, Xiangzhe Kong, Mingyu Li, Jianzhu Ma

**发布时间:** 2025-11-07

**备注:** Abstract 1 page, main text 9 pages, references 2 pages, 4 figures.  Submitted to RECOMB 2026

### GPT解析

### 总结

本文提出了Peptide2Mol，一种E(3)-等变图神经网络扩散模型，通过参考原始肽配体及其周围蛋白质口袋环境来生成小分子，解决了传统AI方法忽略内源性蛋白与肽相互作用的问题。

### 背景

基于结构的药物设计在整合人工智能方面取得了显著进展，特别是在生成命中化合物和先导化合物方面。然而，大多数AI方法忽略了内源性蛋白与肽相互作用的重要性，可能导致次优的分子设计。

### 目的

开发一种能够考虑原始肽配体及其周围蛋白质口袋环境的模型，用于生成和优化生物活性小分子。

### 方法

Peptide2Mol是一种E(3)-等变图神经网络扩散模型，在大数据集上训练，并利用复杂的建模技术。

### 主要发现

1. Peptide2Mol在非自回归生成任务中取得了最先进的性能；2. 产生的分子与原始肽配体具有相似性；3. 该模型允许通过部分扩散过程进行分子优化和肽模拟物设计。

### 结论

Peptide2Mol是一种有效的深度生成模型，可用于从蛋白质结合口袋生成和优化生物活性小分子。

### 翻译

基于结构的药物设计随着人工智能的整合取得了显著进展，特别是在生成命中化合物和先导化合物方面。然而，大多数AI驱动的方法忽略了内源性蛋白与肽相互作用的重要性，这可能导致次优的分子设计。在这项工作中，我们提出了Peptide2Mol，一种E(3)-等变图神经网络扩散模型，它通过参考原始肽配体及其周围的蛋白质口袋环境来生成小分子。在大数据集上训练并利用复杂的建模技术，Peptide2Mol不仅在非自回归生成任务中取得了最先进的性能，还产生了与原始肽配体相似的分子。此外，该模型允许通过部分扩散过程进行分子优化和肽模拟物设计。我们的研究结果表明，Peptide2Mol是一种有效的深度生成模型，可用于从蛋白质结合口袋生成和优化生物活性小分子。


### 论文摘要

Structure-based drug design has seen significant advancements with the integration of artificial intelligence (AI), particularly in the generation of hit and lead compounds. However, most AI-driven approaches neglect the importance of endogenous protein interactions with peptides, which may result in suboptimal molecule designs. In this work, we present Peptide2Mol, an E(3)-equivariant graph neural network diffusion model that generates small molecules by referencing both the original peptide binders and their surrounding protein pocket environments. Trained on large datasets and leveraging sophisticated modeling techniques, Peptide2Mol not only achieves state-of-the-art performance in non-autoregressive generative tasks, but also produces molecules with similarity to the original peptide binder. Additionally, the model allows for molecule optimization and peptidomimetic design through a partial diffusion process. Our results highlight Peptide2Mol as an effective deep generative model for generating and optimizing bioactive small molecules from protein binding pockets.

---

## 17. SPECTRA: Spectral Target-Aware Graph Augmentation for Imbalanced Molecular Property Regression

**论文链接:** [http://arxiv.org/abs/2511.04838v1](http://arxiv.org/abs/2511.04838v1)

**作者:** Brenda Nogueira, Meng Jiang, Nitesh V. Chawla, Nuno Moniz

**发布时间:** 2025-11-06

### GPT解析

### 总结

SPECTRA是一种光谱目标感知的图增强框架，用于在分子属性预测中生成真实的分子图，解决了标准GNN在处理稀有但重要分子时的不足，避免了现有过采样方法对分子拓扑的扭曲。

### 背景

在分子属性预测中，有价值的化合物（如高活性）通常占据目标空间的稀疏区域。标准图神经网络（GNNs）通常优化平均误差，在这些罕见但关键的情况下表现不佳，而现有的过采样方法通常会扭曲分子拓扑结构。

### 目的

引入SPECTRA框架，在光谱域生成真实的分子图，改善标准GNN在处理稀有分子时的性能，同时避免对分子拓扑结构的扭曲。

### 方法

SPECTRA框架包括：从SMILES重建多属性分子图；通过(Fused)Gromov-Wasserstein耦合对齐分子对；在稳定共享基中插值拉普拉斯特征值、特征向量和节点特征；重建边以合成物理合理的中间体。结合基于标签核密度估计的稀有感知预算方案和边缘感知切比雪夫卷积的光谱GNN。

### 主要发现

SPECTRA能够在保持全球平均绝对误差竞争力的同时，持续改善相关目标范围内的误差，并产生可解释的合成分子，其结构反映了底层的光谱几何。

### 结论

光谱、几何感知的增强是不平衡分子属性回归的有效且高效的策略。

### 翻译

在分子属性预测中，最有价值的化合物（例如高活性）通常占据目标空间的稀疏区域。标准图神经网络（GNNs）通常优化平均误差，在这些罕见但关键的情况下表现不佳，而现有的过采样方法往往会扭曲分子拓扑结构。在本文中，我们引入了SPECTRA，一种光谱目标感知的图增强框架，用于在光谱域生成真实的分子图。SPECTRA（i）从SMILES重建多属性分子图；（ii）通过(Fused)Gromov-Wasserstein耦合对齐分子对以获得节点对应关系；（iii）在稳定共享基中插值拉普拉斯特征值、特征向量和节点特征；（iv）重建边以合成具有插值目标的物理合理的中间体。一种从标签核密度估计导出的稀有感知预算方案，在数据稀缺的地方集中增强。结合使用边缘感知切比雪夫卷积的光谱GNN，SPECTRA在保持全球准确性的同时，增加了代表性不足的区域。在基准测试中，SPECTRA持续改善相关目标范围内的误差，同时保持竞争性的整体MAE，并产生可解释的合成分子，其结构反映了底层的光谱几何。我们的结果表明，光谱、几何感知的增强是不平衡分子属性回归的有效且高效的策略。


### 论文摘要

In molecular property prediction, the most valuable compounds (e.g., high potency) often occupy sparse regions of the target space. Standard Graph Neural Networks (GNNs) commonly optimize for the average error, underperforming on these uncommon but critical cases, with existing oversampling methods often distorting molecular topology. In this paper, we introduce SPECTRA, a Spectral Target-Aware graph augmentation framework that generates realistic molecular graphs in the spectral domain. SPECTRA (i) reconstructs multi-attribute molecular graphs from SMILES; (ii) aligns molecule pairs via (Fused) Gromov-Wasserstein couplings to obtain node correspondences; (iii) interpolates Laplacian eigenvalues, eigenvectors and node features in a stable share-basis; and (iv) reconstructs edges to synthesize physically plausible intermediates with interpolated targets. A rarity-aware budgeting scheme, derived from a kernel density estimation of labels, concentrates augmentation where data are scarce. Coupled with a spectral GNN using edge-aware Chebyshev convolutions, SPECTRA densifies underrepresented regions without degrading global accuracy. On benchmarks, SPECTRA consistently improves error in relevant target ranges while maintaining competitive overall MAE, and yields interpretable synthetic molecules whose structure reflects the underlying spectral geometry. Our results demonstrate that spectral, geometry-aware augmentation is an effective and efficient strategy for imbalanced molecular property regression.

---

## 18. Hardware-Accelerated GNN-based Hit Filtering for the Belle II Level-1 Trigger

**论文链接:** [http://arxiv.org/abs/2511.04731v1](http://arxiv.org/abs/2511.04731v1)

**作者:** Greta Heine, Fabio Mayer, Marc Neu, Jürgen Becker, Torben Ferber

**发布时间:** 2025-11-06

### GPT解析

### 总结

该论文提出了一种基于FPGA的硬件加速击中过滤系统，使用图神经网络(GNN)处理Belle II Level-1触发器中的数据。该系统能够实时处理数据，实现探测器级别的背景抑制，具有低延迟和高吞吐量的特点。

### 背景

Belle II实验需要实时处理高亮度对撞机条件下的海量数据，需要低延迟和高吞吐量的解决方案。传统的数据处理方法可能无法满足这些严格要求。

### 目的

开发一种基于GNN的硬件加速击中过滤系统，用于Belle II实验的Level-1触发器，实现实时数据处理和背景抑制，同时满足严格的延迟和吞吐量要求。

### 方法

使用图神经网络处理传感线击中数据；通过量化、剪枝和静态图构建优化GNN以实现高吞吐量硬件操作；采用扇区空间并行化扩展到全探测器覆盖；在AMD Ultrascale XVCU190 FPGA上实现原型系统。

### 主要发现

系统在31.804 MHz的持续吞吐量下实时处理传感线数据；实现探测器级别的背景抑制，测量延迟为632.4 ns；使用35.65%的查找表和29.75%的触发器，零数字信号处理使用；离线验证达到83%的背景击中拒绝率，同时保持95%的信号击中效率。

### 结论

该工作确立了在FPGA上基于GNN的击中级别过滤作为高亮度对撞机条件下实时数据缩减的可扩展低延迟解决方案。

### 翻译

我们提出了一种硬件加速的击中过滤系统，在Belle II Level-1触发器的现场可编程门阵列上采用图神经网络。该GNN利用传感线击中之间的空间和时间关系，并通过量化、剪枝和静态图构建优化为高吞吐量硬件操作。扇区空间并行化允许扩展到全探测器覆盖，满足严格的延迟和吞吐量要求。在31.804 MHz的持续吞吐量下，系统实时处理传感线数据，并实现探测器级别的背景抑制，测量延迟为632.4 ns，同时使用35.65%的查找表和29.75%的触发器，零数字信号处理使用，如在对单个扇区的AMD Ultrascale XVCU190上的原型实现所示。使用Belle II数据的离线验证产生83%的背景击中拒绝率，同时保持95%的信号击中效率。这项工作确立了在FPGA上基于GNN的击中级别过滤作为高亮度对撞机条件下实时数据缩减的可扩展低延迟解决方案。


### 论文摘要

We present a hardware-accelerated hit filtering system employing Graph Neural Networks (GNNs) on Field-Programmable Gate Arrays (FPGAs) for the Belle II Level-1 Trigger. The GNN exploits spatial and temporal relationships among sense wire hits and is optimized for high-throughput hardware operation via quantization, pruning, and static graph-building. Sector-wise spatial parallelization permits scaling to full-detector coverage, satisfying stringent latency and throughput requirements. At a sustained throughput of 31.804 MHz, the system processes sense wire data in real-time and achieves detector-level background suppression with a measured latency of 632.4 ns while utilizing 35.65% of Look-Up Tables (LUTs), and 29.75% of Flip-Flops, with zero Digital Signal Processing (DSP) usage, as demonstrated in a prototype implementation for a single sector on an AMD Ultrascale XVCU190. Offline validation using Belle II data yields a background hit rejection of 83% while maintaining 95% signal hit efficiency. This work establishes hit-level GNN-based filtering on FPGAs as a scalable low-latency solution for real-time data reduction in high-luminosity collider conditions.

---

