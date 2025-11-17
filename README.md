# 今日论文推荐 - 2025-11-17

共 337 篇论文

---

## 1. LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures

**论文链接:** [http://arxiv.org/abs/2511.10209v1](http://arxiv.org/abs/2511.10209v1)

**作者:** Wenzhe He, Xiaojun Chen, Ruiqi Wang, Ruihui Li, Huilong Pi, Jiapeng Zhang, Zhuo Tang, Kenli Li

**发布时间:** 2025-11-13

**备注:** 18 pages, 13 figures, Accepted to AAAI 2026

### GPT解析

### 总结

论文提出LiNeXt，一种轻量级、非扩散网络，用于快速准确的点云补全，解决了传统扩散模型多步迭代采样带来的计算开销问题。

### 背景

3D LiDAR场景补全是自动驾驶车辆感知系统的基本组成部分。先前方法主要采用扩散模型进行高保真重建，但多步迭代采样导致显著的计算开销，限制了实时应用能力。

### 目的

开发一种轻量级、非扩散网络，实现快速准确的点云补全，以克服传统扩散模型的计算效率限制。

### 方法

1. 提出LiNeXt网络架构；2. 应用Noise-to-Coarse (N2C)模块对输入噪声点云进行单次去噪；3. 使用Refine模块对粗略点云进行精确细化；4. 提出Distance-aware Selected Repeat策略，生成更均匀分布的噪声点云。

### 主要发现

在SemanticKITTI数据集上，LiNeXt实现了199.8倍的推理速度提升，将Chamfer距离减少了50.7%，仅使用LiDiff参数的6.1%。

### 结论

LiNeXt在实时场景补全方面表现出卓越的效率和效果，解决了传统扩散模型的计算效率问题。

### 翻译

从点云进行3D LiDAR场景补全是自动驾驶车辆感知系统的基本组成部分。先前方法主要采用扩散模型进行高保真重建。然而，它们的多步迭代采样带来了显著的计算开销，限制了其实时应用能力。为此，我们提出LiNeXt——一种为快速准确的点云补全而优化的轻量级、非扩散网络。具体而言，LiNeXt首先应用Noise-to-Coarse (N2C)模块对输入噪声点云进行单次去噪，从而避免了基于扩散方法的多步迭代采样。然后，Refine模块接收来自N2C模块的粗略点云及其中间特征，进行更精确的细化，进一步增强结构完整性。此外，我们观察到LiDAR点云表现出距离依赖的空间分布，在近距离处密集采样，在远距离处稀疏采样。相应地，我们提出Distance-aware Selected Repeat策略，生成更均匀分布的噪声点云。在SemanticKITTI数据集上，LiNeXt实现了199.8倍的推理速度提升，将Chamfer距离减少了50.7%，仅使用LiDiff参数的6.1%。这些结果表明LiNeXt在实时场景补全方面具有卓越的效率和效果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D LiDAR点云场景补全中的计算效率问题。现有方法主要使用扩散模型进行高保真重建，但多步迭代采样导致计算开销大，限制了实时应用能力。这个问题在自动驾驶领域至关重要，因为LiDAR传感器获取的点云通常稀疏且存在遮挡，导致未观测区域，影响物体检测、姿态估计和地图构建等关键下游任务的执行。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了扩散模型的局限性，认识到其计算开销不适合实时应用。观察到LiDAR点云具有距离依赖的空间分布特性（近距离密集，远距离稀疏），作者设计了距离感知的选择性重复策略。他们借鉴了点云表示方法（如PointNet）进行特征提取，多尺度特征提取思想，并引入注意力机制，但专门设计了跨点注意力（CPA）模块。作者保留了扩散模型中的种子点生成思想，但通过简化实现和直接重建策略避免了迭代过程，从而显著提高了效率。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用轻量级非扩散架构替代计算密集的扩散模型，利用距离感知的点云重复策略生成更均匀分布的噪声点云，通过两阶段处理实现高效且准确的场景补全。整体流程包括：1)距离感知的选择性重复：根据点云距离分组并差异化重复，添加噪声生成均匀分布；2)噪声到粗略模块(N2C)：使用多尺度稀疏卷积提取特征，通过层次化种子点生成和跨点注意力处理几何关系，生成粗略点云；3)精细模块：接收N2C输出，再次使用CPA机制和反卷积生成最终高质量点云；4)训练：使用Chamfer Distance作为损失函数进行两阶段独立训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)距离感知的选择性重复策略(DSR)：根据点云距离分布调整重复次数，解决近距离过采样和远距离欠采样问题；2)跨点注意力机制(CPA)：动态对齐特征并融合互补信息，通过序列段最大池化实现高效特征压缩；3)多尺度稀疏卷积(MSSC)：在多个体素分辨率上并行应用稀疏卷积，捕获多尺度几何信息；4)轻量级非扩散架构：直接重建场景，避免多步迭代采样。相比LiDiff等扩散模型方法，LiNeXt推理速度快199.8倍，参数减少93.9%，Chamfer Distance降低50.7%，且重建质量更高，无条纹伪影。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiNeXt通过创新的轻量级非扩散架构和距离感知的点云处理策略，实现了比现有扩散模型快199.8倍且参数减少93.9%的高效LiDAR场景补全，同时显著提高了重建质量。'}


### 论文摘要

3D LiDAR scene completion from point clouds is a fundamental component of perception systems in autonomous vehicles. Previous methods have predominantly employed diffusion models for high-fidelity reconstruction. However, their multi-step iterative sampling incurs significant computational overhead, limiting its real-time applicability. To address this, we propose LiNeXt-a lightweight, non-diffusion network optimized for rapid and accurate point cloud completion. Specifically, LiNeXt first applies the Noise-to-Coarse (N2C) Module to denoise the input noisy point cloud in a single pass, thereby obviating the multi-step iterative sampling of diffusion-based methods. The Refine Module then takes the coarse point cloud and its intermediate features from the N2C Module to perform more precise refinement, further enhancing structural completeness. Furthermore, we observe that LiDAR point clouds exhibit a distance-dependent spatial distribution, being densely sampled at proximal ranges and sparsely sampled at distal ranges. Accordingly, we propose the Distance-aware Selected Repeat strategy to generate a more uniformly distributed noisy point cloud. On the SemanticKITTI dataset, LiNeXt achieves a 199.8x speedup in inference, reduces Chamfer Distance by 50.7%, and uses only 6.1% of the parameters compared with LiDiff. These results demonstrate the superior efficiency and effectiveness of LiNeXt for real-time scene completion.

---

## 2. RWKV-PCSSC: Exploring RWKV Model for Point Cloud Semantic Scene Completion

**论文链接:** [http://arxiv.org/abs/2511.09878v1](http://arxiv.org/abs/2511.09878v1)

**作者:** Wenzhe He, Xiaojun Chen, Wentang Chen, Hongyu Wang, Ying Liu, Ruihui Li

**发布时间:** 2025-11-13

**DOI:** 10.1145/3746027.3754908

**备注:** 13 pages, 8 figures, published to ACM MM

### GPT解析

### 总结

本文提出了一种轻量级的点云语义场景补全网络RWKV-PCSSC，通过RWKV机制实现了高效的特征聚合和点云生成，在减少模型参数的同时保持了性能优势。

### 背景

现有语义场景补全方法通常采用密集网络架构，参数量大，导致模型复杂度高和资源需求大。

### 目的

解决现有语义场景补全方法参数量大、模型复杂度高的问题，设计一种轻量级但性能优越的网络架构。

### 方法

提出RWKV-PCSSC网络，包含RWKV种子生成器（RWKV-SG）模块从部分点云生成带粗略特征的粗略点云，以及多个RWKV点反卷积（RWKV-PD）模块逐步恢复点云的点特征。

### 主要发现

RWKV-PCSSC相比PointSSC方法，参数量减少4.18倍，内存效率提高1.37倍，同时在多个室内外场景数据集上达到最先进性能。

### 结论

通过紧凑高效的设计，RWKV-PCSSC实现了轻量级模型表示，同时保持了语义场景补全任务的高性能。

### 翻译

语义场景补全旨在从不完整的输入生成完整的语义场景。现有方法通常采用参数量大的密集网络架构，导致模型复杂度和资源需求增加。为解决这些限制，我们提出了RWKV-PCSSC，一种受接收加权键值机制启发的轻量级点云语义场景补全网络。具体而言，我们引入了RWKV种子生成器（RWKV-SG）模块，可以从部分点云聚合特征，生成带粗略特征的粗略点云。随后，通过多个RWKV点反卷积（RWKV-PD）模块逐步恢复点云的点特征。通过利用紧凑高效的设计，我们的方法实现了轻量级模型表示。实验结果表明，与最先进的方法PointSSC相比，RWKV-PCSSC将参数量减少了4.18倍，并将内存效率提高了1.37倍。此外，我们的网络在既有的室内（SSC-PC, NYUCAD-PC）和室外（PointSSC）场景数据集以及我们提出的数据集（NYUCAD-PC-V2, 3D-FRONT-PC）上达到了最先进的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云语义场景补全问题，即从不完整的点云输入生成完整且带有语义标签的3D场景。这个问题在自动驾驶和机器人导航等领域非常重要，因为传感器限制和环境复杂性常常导致不完整的输入数据，这对场景理解和决策构成挑战。现有方法通常存在模型参数量大、内存消耗高、计算效率低等问题，限制了实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有点云语义场景补全方法的局限性，包括高参数量、高内存需求和无法有效捕捉复杂特征等问题。他们借鉴了RWKV机制在自然语言处理中的高效性，以及PointNet、PointNet++等点云处理方法。同时参考了CasFusionNet和PointSSC等点云语义场景补全方法，以及Set Abstraction、Farthest Point Sampling等技术。基于这些现有工作，作者设计了轻量级的RWKV-PCSSC网络，包括RWKV种子生成器和RWKV点反卷积模块，采用'粗到精'的层次化方法逐步实现场景补全和语义分割。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是利用RWKV机制的高效性设计轻量级网络，通过'粗到精'的层次化方法实现场景补全。整体流程是：1) 输入不完整点云；2) 通过RWKV种子生成器生成带有粗略特征和语义标签的粗略点云；3) 通过多阶段RWKV点反卷积模块逐步精细化和分割点云；4) 输出完整点云和语义标签。每个阶段都使用特征提取器、RWKV注意力和重建头等组件，逐步提升点云质量和语义准确性。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 轻量级网络设计，参数量比PointSSC减少76.1%，内存减少27%；2) RWKV种子生成器模块，能有效预测缺失区域并捕获局部结构；3) RWKV点反卷积模块，通过多阶段迭代实现精细重建；4) RWKV注意力和点RWKV模块，结合全局和局部特征；5) 提出3D-FRONT-PC和NYUCAD-PC-V2两个新数据集。相比之前工作，RWKV-PCSSC避免了体素化的高计算成本，采用更轻量级的架构，更好地结合了场景补全和语义分割任务，在保持性能的同时显著降低了资源需求。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于RWKV机制的轻量级点云语义场景补全网络，通过创新的种子生成器和点反卷积模块实现了高效高质量的场景重建，同时大幅减少了模型参数量和内存使用。'}


### 论文摘要

Semantic Scene Completion (SSC) aims to generate a complete semantic scene from an incomplete input. Existing approaches often employ dense network architectures with a high parameter count, leading to increased model complexity and resource demands. To address these limitations, we propose RWKV-PCSSC, a lightweight point cloud semantic scene completion network inspired by the Receptance Weighted Key Value (RWKV) mechanism. Specifically, we introduce a RWKV Seed Generator (RWKV-SG) module that can aggregate features from a partial point cloud to produce a coarse point cloud with coarse features. Subsequently, the point-wise feature of the point cloud is progressively restored through multiple stages of the RWKV Point Deconvolution (RWKV-PD) modules. By leveraging a compact and efficient design, our method achieves a lightweight model representation. Experimental results demonstrate that RWKV-PCSSC reduces the parameter count by 4.18$\times$ and improves memory efficiency by 1.37$\times$ compared to state-of-the-art methods PointSSC. Furthermore, our network achieves state-of-the-art performance on established indoor (SSC-PC, NYUCAD-PC) and outdoor (PointSSC) scene dataset, as well as on our proposed datasets (NYUCAD-PC-V2, 3D-FRONT-PC).

---

## 3. DANCE: Density-agnostic and Class-aware Network for Point Cloud Completion

**论文链接:** [http://arxiv.org/abs/2511.07978v1](http://arxiv.org/abs/2511.07978v1)

**作者:** Da-Yeong Kim, Yeong-Jun Cho

**发布时间:** 2025-11-11

### GPT解析

### 总结

本文提出了一种名为DANCE（Density-agnostic and Class-aware Network）的新型点云补全框架，能够在不完整3D扫描中恢复缺失的几何结构，同时保留观察到的部分，且对输入密度变化和噪声具有鲁棒性。

### 背景

点云补全旨在从不完整的3D扫描中恢复缺失的几何结构，这些扫描通常因遮挡或传感器视角有限而不完整。现有方法通常假设固定的输入/输出密度或依赖基于图像的表示，不太适合具有可变稀疏性和有限监督的现实场景。

### 目的

开发一种密度无关且类别感知的点云补全框架，能够只补全缺失区域，同时保留观察到的几何结构，适用于现实世界中的可变稀疏性和有限监督场景。

### 方法

DANCE通过从多个视角进行基于射线的采样来生成候选点，使用transformer解码器精炼点的位置并预测不透明度分数以确定点的有效性，并在几何特征上训练轻量级分类头实现类别一致的补全，无需外部图像监督。

### 主要发现

在PCN和MVP基准上的大量实验表明，DANCE在准确性和结构一致性方面优于最先进的方法，同时对不同输入密度和噪声水平保持鲁棒性。

### 结论

DANCE是一种有效的点云补全方法，能够处理现实世界中的挑战性场景，包括可变的稀疏性和有限的监督，实现了比现有方法更好的性能。

### 翻译

点云补全旨在从不完整的3D扫描中恢复缺失的几何结构，这些扫描通常因遮挡或传感器视角有限而不完整。现有方法通常假设固定的输入/输出密度或依赖基于图像的表示，使得它们不太适合具有可变稀疏性和有限监督的现实场景。在本文中，我们引入了密度无关且类别感知网络（DANCE），这是一种新型框架，它只补全缺失区域，同时保留观察到的几何结构。DANCE通过从多个视角进行基于射线的采样来生成候选点。然后，transformer解码器精炼它们的位置并预测不透明度分数，该分数确定每个点是否有效，应包含在最终表面中。为了融入语义指导，直接在几何特征上训练了一个轻量级分类头，使得能够在没有外部图像监督的情况下实现类别一致的补全。在PCN和MVP基准上的大量实验表明，DANCE在准确性和结构一致性方面优于最先进的方法，同时保持对不同输入密度和噪声水平的鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云补全中的两个关键问题：一是现有方法假设固定输入/输出密度，不适合真实世界场景中密度变化的情况；二是许多方法依赖基于图像的表示获取语义信息，导致与原始3D输入几何不一致。这个问题在现实中非常重要，因为自动驾驶、机器人和3D重建等应用需要完整准确的3D表示，而真实世界的点云数据往往密度不一且存在遮挡，现有方法难以处理这些情况。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性进行设计，发现固定密度假设和图像依赖是主要瓶颈。他们借鉴了NeRF的射线采样策略，但将其应用于点云补全而非视图合成；同时采用了Transformer架构和编码器-解码器框架，但创新性地组合这些技术以解决密度无关性和类别感知性问题。关键创新在于设计了一个分类头直接从3D几何特征中学习语义信息，而不是依赖外部图像监督。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是密度无关设计和类别感知补全。整体流程包括：1)使用基于射线的采样策略从多个视角生成候选点；2)编码器提取不完全点云和候选点的3D特征；3)解码器包含面部Transformer(通过交叉注意力和自注意力增强特征交互)、分类头(预测物体类别)和融合网络(结合几何特征和类别信息预测点的偏移量和不透明度)；4)根据预测调整候选点位置并筛选有效点，生成最终补全结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：密度无关设计、类别感知模块、基于射线的候选点生成、面部Transformer和融合网络。相比之前的工作，DANCE不再假设固定输入/输出密度，更适合真实世界；不再依赖图像表示获取语义，直接从3D几何学习；只补全缺失区域而非重新生成整个点云，更好保留观察到的几何；能够产生与物体类别一致的补全结果，提高语义一致性。实验表明它在准确性和结构一致性上优于现有方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DANCE提出了一种密度无关且类别感知的点云补全框架，能够处理不同密度的输入，产生灵活密度的输出，并通过直接从3D几何中学习语义信息，实现了比现有方法更准确、结构一致性更高的点云补全。'}


### 论文摘要

Point cloud completion aims to recover missing geometric structures from incomplete 3D scans, which often suffer from occlusions or limited sensor viewpoints. Existing methods typically assume fixed input/output densities or rely on image-based representations, making them less suitable for real-world scenarios with variable sparsity and limited supervision. In this paper, we introduce Density-agnostic and Class-aware Network (DANCE), a novel framework that completes only the missing regions while preserving the observed geometry. DANCE generates candidate points via ray-based sampling from multiple viewpoints. A transformer decoder then refines their positions and predicts opacity scores, which determine the validity of each point for inclusion in the final surface. To incorporate semantic guidance, a lightweight classification head is trained directly on geometric features, enabling category-consistent completion without external image supervision. Extensive experiments on the PCN and MVP benchmarks show that DANCE outperforms state-of-the-art methods in accuracy and structural consistency, while remaining robust to varying input densities and noise levels.

---

## 4. Adaptive 3D Reconstruction via Diffusion Priors and Forward Curvature-Matching Likelihood Updates

**论文链接:** [http://arxiv.org/abs/2511.06310v1](http://arxiv.org/abs/2511.06310v1)

**作者:** Seunghyeok Shin, Dabin Kim, Hongki Lim

**发布时间:** 2025-11-09

### GPT解析

### 总结

本文提出了一种新的前向曲率匹配(FCM)更新方法与扩散采样相结合，用于从图像重建高质量点云。该方法动态确定最优步长，实现高保真重建，支持多种输入模态且无需重新训练。

### 背景

从图像重建高质量点云在计算机视觉中具有挑战性。现有的基于生成模型的方法，特别是直接学习后验的扩散模型方法，存在不灵活的问题，需要训练时的条件信号，只支持固定数量的输入视图，且需要针对不同测量重新训练。

### 目的

解决现有扩散模型方法中依赖启发式固定步长导致的收敛速度慢和重建质量不佳的问题，实现更高效、更灵活的点云重建方法。

### 方法

结合新的前向曲率匹配(FCM)更新方法与扩散采样，使用前向自动微分和有限差分曲率估计动态确定最优步长，实现似然更新的精确优化。

### 主要发现

该方法能够从单视图和多视图输入进行高保真重建，支持各种输入模态且无需重新训练。实验表明在匹配或更低NFE下实现了更好的重建质量，得到更高的F分数和更低的CD和EMD。

### 结论

FCM方法在点云重建任务中表现出更高的效率和适应性，适用于实际应用。

### 翻译

从图像重建高质量点云在计算机视觉中仍然具有挑战性。现有的基于生成模型的方法，特别是直接学习后验的扩散模型方法，可能存在不灵活的问题——它们需要训练时的条件信号，只支持固定数量的输入视图，并且需要针对不同的测量进行完全重新训练。最近的基于扩散的方法尝试通过结合先验模型和似然更新来解决这个问题，但它们依赖于启发式的固定步长进行似然更新，导致收敛速度慢和重建质量不佳。我们通过将新颖的前向曲率匹配(FCM)更新方法与扩散采样相结合，推进了这一方法。我们的方法仅使用前向自动微分和有限差分曲率估计来确定最优步长，实现了似然更新的精确优化。这种形式能够从单视图和多视图输入进行高保真重建，并通过简单的算子替换支持各种输入模态——所有这些都不需要重新训练。在ShapeNet和CO3D数据集上的实验表明，我们的方法在匹配或更低NFE下实现了更好的重建质量，得到更高的F分数和更低的CD和EMD，验证了其在实际应用中的效率和适应性。代码可在https://github.com/Seunghyeok0715/FCM获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从图像中重建高质量点云的问题，特别是现有扩散模型方法在处理不同输入视图数量和测量模态时缺乏灵活性，以及使用固定步长导致收敛缓慢和重建质量不佳的问题。这个问题在现实中很重要，因为3D重建在机器人、自动驾驶、增强现实和虚拟环境等多种应用中至关重要，而点云作为表示对象和场景的基本数据结构，其生成质量直接影响这些应用的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有扩散模型在3D重建中的局限性，特别是处理逆问题时渲染操作的复杂性和非线性导致的伴随计算不可行问题。他们借鉴了扩散后采样（DPS）框架，将后验分布分解为预训练先验和可适应似然项。为解决DPS中固定步长导致的收敛问题，作者设计了前向曲率匹配（FCM）方法，利用前向自动微分和有限差分曲率估计动态确定最优步长，避免了复杂渲染器的伴随计算。该方法还扩展到多视图和不同测量模态的应用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将后验分布分解为训练好的先验和可更新的似然，并使用FCM方法动态确定似然更新中的最优步长。整体流程包括：1）训练扩散先验模型；2）设计可微渲染器作为测量操作符；3）实现FCM引导的似然更新，利用自适应比例探针和Barzilai-Borwein规则计算步长；4）将DDIM采样与FCM结合，在每一步应用FCM优化；5）扩展到多视图重建，计算多视图平均损失和梯度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）前向曲率匹配（FCM）优化方法，利用前向自动微分确定最优步长；2）自适应步长确定机制，结合Barzilai-Borwein规则和Armijo条件；3）支持多种输入模态和视图数量而无需重新训练；4）提供理论保证证明收敛性和最优性。相比之前工作，PC2直接学习后验分布需训练时包含图像，BDM依赖特定后验得分函数，而传统DPS使用固定步长导致收敛慢。FCM方法避免了这些限制，实现了更高效、灵活的3D重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过提出前向曲率匹配优化方法，解决了扩散模型在3D重建中的步长选择问题，实现了从单视图或多视图图像的高质量点云重建，同时支持多种输入模态而无需重新训练模型。'}


### 论文摘要

Reconstructing high-quality point clouds from images remains challenging in computer vision. Existing generative-model-based approaches, particularly diffusion-model approaches that directly learn the posterior, may suffer from inflexibility -- they require conditioning signals during training, support only a fixed number of input views, and need complete retraining for different measurements. Recent diffusion-based methods have attempted to address this by combining prior models with likelihood updates, but they rely on heuristic fixed step sizes for the likelihood update that lead to slow convergence and suboptimal reconstruction quality. We advance this line of approach by integrating our novel Forward Curvature-Matching (FCM) update method with diffusion sampling. Our method dynamically determines optimal step sizes using only forward automatic differentiation and finite-difference curvature estimates, enabling precise optimization of the likelihood update. This formulation enables high-fidelity reconstruction from both single-view and multi-view inputs, and supports various input modalities through simple operator substitution -- all without retraining. Experiments on ShapeNet and CO3D datasets demonstrate that our method achieves superior reconstruction quality at matched or lower NFEs, yielding higher F-score and lower CD and EMD, validating its efficiency and adaptability for practical applications. Code is available at https://github.com/Seunghyeok0715/FCM

---

## 5. ALISE: Annotation-Free LiDAR Instance Segmentation for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2510.05752v2](http://arxiv.org/abs/2510.05752v2)

**作者:** Yongxuan Lyu, Guangfeng Jiang, Hongsi Liu, Jun Liu

**发布时间:** 2025-10-07

### GPT解析

### 总结

ALISE是一种无需任何标注即可进行激光雷达实例分割的新框架，通过视觉基础模型生成初始伪标签，并利用时空投票模块和两种形式的语义监督进行优化，显著提升了无监督3D实例分割的性能。

### 背景

手动标注室外激光雷达点云用于实例分割非常耗时且成本高昂，现有方法虽试图减少负担但仍依赖某种形式的人工标注。

### 目的

完全消除对人工标注的依赖，实现无需任何标注的激光雷达实例分割。

### 方法

使用视觉基础模型在文本和图像指导下生成初始伪标签，通过时空投票模块结合2D和3D语义进行优化，并引入基于2D先验的损失函数和基于原型的对比损失进行语义监督。

### 主要发现

该方法在无监督3D实例分割中建立了新的最先进水平，性能甚至超过了使用真实地面真值2D边界框进行监督的MWSIS方法，在mAP上高出2.53%（50.95%对48.42%）。

### 结论

ALISE框架通过全面的设计实现了显著的性能提升，为无监督3D实例分割提供了新的解决方案。

### 翻译

室外激光雷达点云的手动标注用于实例分割极其耗时且成本高昂。当前方法试图减轻这一负担但仍依赖某种形式的人工标注。为完全消除这种依赖，我们引入了ALISE，一种无需任何标注即可执行激光雷达实例分割的新框架。核心挑战是以完全无监督的方式生成高质量的伪标签。我们的方法首先采用由文本和图像引导的视觉基础模型(VFMs)来生成初始伪标签。然后，我们通过专门的时空投票模块优化这些标签，该模块结合2D和3D语义进行离线和在线优化。为实现卓越的特征学习，我们进一步引入了两种形式的语义监督：一组基于2D先验的损失函数，将视觉知识注入3D网络；以及一种新颖的基于原型的对比损失，通过利用3D语义一致性构建判别性特征空间。这种全面的设计带来了显著的性能提升，为无监督3D实例分割建立了新的最先进水平。值得注意的是，我们的方法甚至超过了使用地面真真实(GT)2D边界框进行监督的MWSIS方法，mAP高出2.53%（50.95%对48.42%）。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决LiDAR点云实例分割中人工标注极其昂贵和耗时的问题。这个问题在自动驾驶领域尤为重要，因为高质量标注需要大量人力物力，限制了模型开发速度和数据规模，阻碍了自动驾驶技术的普及和应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有弱监督和无监督方法主要关注语义分割而非实例分割，后者更具挑战性。他们观察到视觉基础模型(VFMs)有强大泛化能力，可利用图像信息生成3D伪标签。方法借鉴了MWSIS等弱监督方法使用2D边界框的思想，但完全消除了对标注的依赖；同时利用GroundingDINO和SAM等现有模型进行创新应用，通过对比学习等方法提升特征学习效果。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过视觉基础模型生成高质量3D伪标签，再通过时空投票和多模态监督优化这些标签。整体流程包括：1)无监督伪标签生成(UPG)：使用2D检测和分割生成3D伪标签；2)时空伪标签优化：包括离线优化(OFR)利用相邻帧信息和在线优化(ONR)使用网络自身预测；3)多模态监督训练：通过VFM先验知识蒸馏和基于原型的对比学习提升特征表示；4)组合各部分损失函数训练3D分割网络。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)完全无标注框架，消除任何人工标注依赖；2)UPG模块保留VFM语义分布而非简单生成one-hot标签；3)两阶段标签优化(OFR和ONR)提高伪标签质量；4)多模态监督方案(VPD和PCL)增强特征学习；5)跨视图实例合并处理多视图检测情况。相比之前工作，ALISE不仅完全无标注，还通过保留语义分布和时空优化显著提高了伪标签质量，超越了多种弱监督方法，甚至在使用少量标注微调后超越了全监督基线。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ALISE提出了一种完全无标注的LiDAR实例分割框架，通过视觉基础模型生成高质量伪标签，结合时空优化和多模态监督，显著提升了无监督3D实例分割的性能，甚至超越了依赖少量标注的弱监督方法。'}


### 论文摘要

The manual annotation of outdoor LiDAR point clouds for instance segmentation is extremely costly and time-consuming. Current methods attempt to reduce this burden but still rely on some form of human labeling. To completely eliminate this dependency, we introduce ALISE, a novel framework that performs LiDAR instance segmentation without any annotations. The central challenge is to generate high-quality pseudo-labels in a fully unsupervised manner. Our approach starts by employing Vision Foundation Models (VFMs), guided by text and images, to produce initial pseudo-labels. We then refine these labels through a dedicated spatio-temporal voting module, which combines 2D and 3D semantics for both offline and online optimization. To achieve superior feature learning, we further introduce two forms of semantic supervision: a set of 2D prior-based losses that inject visual knowledge into the 3D network, and a novel prototype-based contrastive loss that builds a discriminative feature space by exploiting 3D semantic consistency. This comprehensive design results in significant performance gains, establishing a new state-of-the-art for unsupervised 3D instance segmentation. Remarkably, our approach even outperforms MWSIS, a method that operates with supervision from ground-truth (GT) 2D bounding boxes by a margin of 2.53% in mAP (50.95% vs. 48.42%).

---

## 6. Geospatial Chain of Thought Reasoning for Enhanced Visual Question Answering on Satellite Imagery

**论文链接:** [http://arxiv.org/abs/2511.11198v1](http://arxiv.org/abs/2511.11198v1)

**作者:** Shambhavi Shanker, Manikandan Padmanaban, Jagabondhu Hazra

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究提出了一种结合链式思维推理(CoT)与直接偏好优化(DPO)的VQA框架，用于提升卫星图像视觉问答的性能，特别是在气候相关应用领域。

### 背景

地理空间链式思维推理对推进卫星图像视觉问答至关重要，尤其在气候相关应用如灾害监测、基础设施风险评估等方面。现有VQA模型虽能实现遥感数据的可扩展解释，但常缺乏复杂地理空间查询所需的结构化推理能力。

### 目的

开发一个整合CoT推理与DPO的VQA框架，提高模型的可解释性、鲁棒性和准确性，使其能够更好地处理复杂的地理空间查询任务。

### 方法

通过生成中间推理过程，使模型能够更好地处理涉及检测、分类、空间关系和比较分析的任务，并将CoT推理与直接偏好优化(DPO)相结合应用于VQA框架中。

### 主要发现

实验表明，CoT监督比直接基线提高了34.9%的准确性，而DPO在准确性和推理质量方面带来了额外提升，显著增强了模型在复杂地理空间任务中的表现。

### 结论

所提出的系统通过实现更丰富的地理空间推理和更有效的气候应用案例，成功推进了多光谱地球观测的VQA发展，为气候相关决策支持提供了可靠工具。

### 翻译

地理空间链式思维(CoT)推理对于推进卫星图像视觉问答(VQA)至关重要，特别是在气候相关应用如灾害监测、基础设施风险评估、城市弹性规划和政策支持等方面。现有的VQA模型能够实现遥感数据的可扩展解释，但通常缺乏复杂地理空间查询所需的结构化推理。我们提出了一种将CoT推理与直接偏好优化(DPO)相结合的VQA框架，以提高可解释性、鲁棒性和准确性。通过生成中间推理过程，模型能够更好地处理涉及检测、分类、空间关系和比较分析的任务，这些任务对于高风险气候领域中可靠的决策支持至关重要。实验表明，CoT监督比直接基线提高了34.9%的准确性，而DPO在准确性和推理质量方面带来了额外提升。所 resulting系统通过实现更丰富的地理空间推理和更有效的气候应用案例，推进了多光谱地球观测的VQA发展。


### 论文摘要

Geospatial chain of thought (CoT) reasoning is essential for advancing Visual Question Answering (VQA) on satellite imagery, particularly in climate related applications such as disaster monitoring, infrastructure risk assessment, urban resilience planning, and policy support. Existing VQA models enable scalable interpretation of remote sensing data but often lack the structured reasoning required for complex geospatial queries. We propose a VQA framework that integrates CoT reasoning with Direct Preference Optimization (DPO) to improve interpretability, robustness, and accuracy. By generating intermediate rationales, the model better handles tasks involving detection, classification, spatial relations, and comparative analysis, which are critical for reliable decision support in high stakes climate domains. Experiments show that CoT supervision improves accuracy by 34.9\% over direct baselines, while DPO yields additional gains in accuracy and reasoning quality. The resulting system advances VQA for multispectral Earth observation by enabling richer geospatial reasoning and more effective climate use cases.

---

## 7. Miniature Testbed for Validating Multi-Agent Cooperative Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.11022v1](http://arxiv.org/abs/2511.11022v1)

**作者:** Hyunchul Bae, Eunjae Lee, Jehyeop Han, Minhee Kang, Jaehyeon Kim, Junggeun Seo, Minkyun Noh, Heejin Ahn

**发布时间:** 2025-11-14

**备注:** 8 pages

### GPT解析

### 总结

该研究设计并实现了一个名为CIVAT的1:15比例小型测试平台，用于验证合作自动驾驶功能，该平台配备了具有感知、边缘计算和通信能力的智能基础设施。

### 背景

合作自动驾驶通过车辆与智能路侧基础设施的实时协作来扩展车辆自主性，这是一个具有挑战性但至关重要的问题。然而，现有测试平台均未配备具有感知、边缘计算和通信能力的智能基础设施。

### 目的

解决现有测试平台的不足，设计并实现一个配备智能基础设施的测试平台，以验证合作自动驾驶功能。

### 方法

设计并实现了一个1:15比例的小型测试平台CIVAT，包括缩放城市地图、配备车载传感器的自动驾驶车辆和智能基础设施。通过共享Wi-Fi和ROS2框架，采用发布-订阅模式集成V2V和V2I通信，实现车辆与基础设施之间的信息交换。

### 主要发现

通过基于基础设施的感知和交叉路口管理实验成功验证了系统功能。

### 结论

CIVAT测试平台成功实现了车辆与智能基础设施之间的协作，为合作自动驾驶研究提供了有效的验证工具。

### 翻译

合作自动驾驶通过实现车辆与智能路侧基础设施之间的实时协作来扩展车辆自主性，仍然是一个具有挑战性但至关重要的问题。然而，现有的测试平台都没有配备具有感知、边缘计算和通信能力的智能基础设施。为解决这一差距，我们设计并实现了一个1:15比例的小型测试平台CIVAT，用于验证合作自动驾驶，该平台由缩放城市地图、配备车载传感器的自动驾驶车辆和智能基础设施组成。所提出的测试平台通过共享Wi-Fi和ROS2框架，采用发布-订阅模式集成V2V和V2I通信，使车辆与基础设施之间能够交换信息，实现协作驾驶功能。作为案例研究，我们通过基于基础设施的感知和交叉路口管理实验验证了该系统。


### 论文摘要

Cooperative autonomous driving, which extends vehicle autonomy by enabling real-time collaboration between vehicles and smart roadside infrastructure, remains a challenging yet essential problem. However, none of the existing testbeds employ smart infrastructure equipped with sensing, edge computing, and communication capabilities. To address this gap, we design and implement a 1:15-scale miniature testbed, CIVAT, for validating cooperative autonomous driving, consisting of a scaled urban map, autonomous vehicles with onboard sensors, and smart infrastructure. The proposed testbed integrates V2V and V2I communication with the publish-subscribe pattern through a shared Wi-Fi and ROS2 framework, enabling information exchange between vehicles and infrastructure to realize cooperative driving functionality. As a case study, we validate the system through infrastructure-based perception and intersection management experiments.

---

## 8. CityVerse: A Unified Data Platform for Multi-Task Urban Computing with Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.10418v1](http://arxiv.org/abs/2511.10418v1)

**作者:** Yaqiao Zhu, Hongkai Wen, Mark Birkin, Man Luo

**发布时间:** 2025-11-13

### GPT解析

### 总结

CityVerse是首个统一平台，整合多源城市数据、基于能力的任务分类法和动态模拟功能，用于系统化评估城市环境中的大型语言模型(LLMs)。

### 背景

大型语言模型(LLMs)在城市计算领域展现出巨大潜力，但在评估LLMs时面临两个关键挑战：缺乏统一平台获取一致的多源数据，以及任务定义碎片化妨碍公平比较。

### 目的

开发CityVerse平台，解决LLMs在城市任务评估中的挑战，提供系统化评估框架。

### 方法

CityVerse平台提供三方面功能：1)基于坐标的数据API，整合10类城市数据(包括空间特征、时间动态、人口统计和多模态图像)，包含3800万条记录；2)任务API，将43个城市计算任务组织成四级认知层次(感知、空间理解、推理与预测、决策与交互)；3)交互式可视化前端，支持实时数据检索、多层显示和模拟回放。

### 主要发现

通过在主流LLMs和代表性任务上的评估验证了平台有效性，证明其支持可重现和系统化的评估能力。

### 结论

CityVerse为城市计算领域推进LLMs和多任务方法提供了可重用的基础平台。

### 翻译

大型语言模型(LLMs)在城市计算领域展现出巨大潜力，从空间推理到预测分析。然而，在多样化城市任务中评估LLMs面临两个关键挑战：缺乏统一平台获取一致的多源数据，以及任务定义碎片化妨碍公平比较。为解决这些挑战，我们提出CityVerse，这是首个统一平台，整合多源城市数据、基于能力的任务分类法和动态模拟功能，用于系统化评估城市环境中的LLMs。CityVerse提供：1)基于坐标的数据API，统一了10类城市数据(包括空间特征、时间动态、人口统计和多模态图像)，包含超过3800万条精选记录；2)任务API将43个城市计算任务组织成四级认知层次：感知、空间理解、推理与预测，以及决策与交互，实现跨能力级别的标准化评估；3)交互式可视化前端，支持实时数据检索、多层显示和模拟回放，便于直观探索和验证。我们通过在主流LLMs和代表性任务上的评估验证了平台的有效性，展示了其支持可重现和系统化评估的能力。CityVerse为城市计算领域推进LLMs和多任务方法提供了可重用的基础。


### 论文摘要

Large Language Models (LLMs) show remarkable potential for urban computing, from spatial reasoning to predictive analytics. However, evaluating LLMs across diverse urban tasks faces two critical challenges: lack of unified platforms for consistent multi-source data access and fragmented task definitions that hinder fair comparison. To address these challenges, we present CityVerse, the first unified platform integrating multi-source urban data, capability-based task taxonomy, and dynamic simulation for systematic LLM evaluation in urban contexts. CityVerse provides: 1) coordinate-based Data APIs unifying ten categories of urban data-including spatial features, temporal dynamics, demographics, and multi-modal imagery-with over 38 million curated records; 2) Task APIs organizing 43 urban computing tasks into a four-level cognitive hierarchy: Perception, Spatial Understanding, Reasoning and Prediction, and Decision and Interaction, enabling standardized evaluation across capability levels; 3) an interactive visualization frontend supporting real-time data retrieval, multi-layer display, and simulation replay for intuitive exploration and validation. We validate the platform's effectiveness through evaluations on mainstream LLMs across representative tasks, demonstrating its capability to support reproducible and systematic assessment. CityVerse provides a reusable foundation for advancing LLMs and multi-task approaches in the urban computing domain.

---

## 9. COMBUST: Gridded combustible mass estimates of the built environment in the conterminous United States (1975-2020)

**论文链接:** [http://arxiv.org/abs/2511.08893v1](http://arxiv.org/abs/2511.08893v1)

**作者:** Johannes H. Uhl, Maxwell C. Cook, Cibele Amaral, Stefan Leyk, Jennifer K. Balch, Alan Robock, Owen B. Toon

**发布时间:** 2025-11-12

### GPT解析

### 总结

该研究开发了美国本土城市可燃物的精细估算数据集COMBUST，量化了建筑材料、建筑内容和私家车的可燃质量，为灾害风险评估提供了重要资源。

### 背景

自然灾害频率增加、城市扩张和地缘政治不稳定导致火灾风险上升，而对可燃物及其时空分布的定量知识对风险评估至关重要。现有研究主要关注生物质燃料，对城市建筑环境可燃物的空间明确量化研究不足。

### 目的

开发美国本土城市燃料的精细估算，量化建筑材料、建筑内容和私家车的可燃质量，创建空间明确的数据集，支持灾害风险评估和规划决策。

### 方法

通过整合多种地理空间数据源（地球观测数据、房地产数据、统计估计和自愿地理信息），开发了名为COMBUST的数据集，并创建了配套的COMBUST PLUS数据集，便于建筑和人口的燃烧暴露建模。

### 主要发现

成功估算了250米空间分辨率的美国本土城市燃料，包含1975年至2020年的不同回溯情景，为生态和社会科学研究以及灾害风险管理提供了丰富的数据资源。

### 结论

COMBUST数据集及其配套数据为生态和社会科学应用、灾害风险管理和美国定居点规划相关的决策制定提供了重要资源，可在提供的DOI获取。

### 翻译

随着野火和干旱等自然灾害的发生频率增加，以及城市扩张和土地消耗，导致人口和人类聚居地的火灾风险水平不断上升。此外，世界许多地区日益增加的地缘政治不稳定性要求评估与军事行动可能造成的潜在危险相关的情景。关于可燃物及其在景观中时空分布的定量知识对风险评估和潜在损害评估至关重要。虽然基于遥感观测对生物质燃料的分布有良好理解，但建筑环境中的可燃质量很少被空间明确地量化。因此，我们为美国本土开发了城市燃料的精细估算，估算了建筑材料、建筑内容和私家车在250米空间分辨率下的可燃质量。 resulting数据集名为COMBUST（美国本土可燃建筑环境质量），包含1975年至2020年的不同回溯情景。COMBUST基于多种地理空间数据源的整合，如地球观测数据、房地产数据、统计估计和自愿地理信息。COMBUST配有COMBUST PLUS，一套一致的网格数据集，便于建筑和人口的燃烧暴露建模。这些数据集构成了生态和社会科学应用以及美国定居点灾害风险管理和规划相关决策制定的丰富资源。COMBUST可在https://doi.org/10.5281/zenodo.15611963获取。


### 论文摘要

The increasing occurrence of natural hazards such as wildfires and drought, along with urban expansion and land consumption, causes increasing levels of fire risk to populations and human settlements. Moreover, increasing geopolitical instability in many regions of the world requires evaluation of scenarios related to potential hazards caused by military operations. Quantitative knowledge on burnable fuels and their spatio-temporal distribution across landscapes is crucial for risk and potential damage assessments. While there is good understanding of the distributions of biomass fuels based on remote sensing observations, the combustible mass of the built environment has rarely been quantified in a spatially explicit manner. Therefore, we developed fine-grained estimates of urban fuels for the conterminous United States, estimating the combustible mass of building materials, building contents, and personal vehicles at 250 m spatial resolution. The resulting dataset is called COMBUST (Combustible mass of the built environment in the conterminous United States) and includes different backcasting scenarios from 1975 to 2020. COMBUST is based on the integration of a variety of geospatial data sources such as Earth-observation derived data, real estate data, statistical estimates and volunteered geographic information. COMBUST is accompanied by COMBUST PLUS, a set of consistently enumerated gridded datasets facilitating combustion exposure modelling of buildings and population. These datasets constitute a rich resource for ecological and social science applications, as well as for disaster risk management and planning-related decision making for U.S. settlements. COMBUST is available at https://doi.org/10.5281/zenodo.15611963.

---

## 10. DiffRegCD: Integrated Registration and Change Detection with Diffusion Features

**论文链接:** [http://arxiv.org/abs/2511.07935v2](http://arxiv.org/abs/2511.07935v2)

**作者:** Seyedehanita Madani, Rama Chellappa, Vishal M. Patel

**发布时间:** 2025-11-11

**备注:** 10 pages, 8 figures. Accepted to WACV 2026

### GPT解析

### 总结

DiffRegCD是一个统一的密集配准和变化检测框架，通过将对应关系估计作为高斯平滑分类任务，并利用预训练去噪扩散模型的特征，实现了在各种数据集上超越基线方法的性能，且对时间和几何变化具有鲁棒性。

### 背景

变化检测是计算机视觉和遥感的基础，支持环境监测、灾害响应和城市发展等应用。然而，实际图像常存在视差、视角变化和长时间间隔导致的严重错位问题，使传统方法难以处理。

### 目的

开发一个能够处理大位移、视差和长时间间隔变化的统一变化检测框架，解决传统两阶段方法和最近联合框架的局限性。

### 方法

提出DiffRegCD框架，将对应关系估计重新表述为高斯平滑分类任务，利用预训练去噪扩散模型的冻结多尺度特征，并通过受控仿射扰动提供监督，无需伪标签即可获得成对真实标签。

### 主要发现

在多个航空和地面级别数据集上的实验表明，DiffRegCD持续超越最近的基线方法，并在广泛的时间和几何变化下保持可靠性，证明了扩散特征和基于分类的对应关系作为统一变化检测基础的有效性。

### 结论

DiffRegCD通过统一密集配准和变化检测，解决了大位移和长时间间隔变化带来的挑战，为变化检测领域提供了一个强大而鲁棒的解决方案。

### 翻译

变化检测是计算机视觉和遥感的基础，支持环境监测、灾害响应和城市发展等应用。大多数变化检测模型假设输入图像已配准，但实际图像常存在视差、视角变化和长时间间隔导致的严重错位问题。传统两阶段方法和最近的联合框架在大位移情况下仍然表现不佳。我们提出了DiffRegCD，一个将密集配准和变化检测统一在单一模型中的集成框架。DiffRegCD将对应关系估计重新表述为高斯平滑分类任务，实现亚像素精度和稳定训练。它利用预训练去噪扩散模型的冻结多尺度特征，确保对光照和视角变化的鲁棒性。通过在标准变化检测数据集上应用受控的仿射扰动提供监督，无需伪标签即可获得光流和变化检测的成对真实标签。在多个航空和地面级别数据集上的大量实验表明，DiffRegCD持续超越最近的基线方法，并在广泛的时间和几何变化下保持可靠性，建立了扩散特征和基于分类的对应关系作为统一变化检测的强大基础。


### 论文摘要

Change detection (CD) is fundamental to computer vision and remote sensing, supporting applications in environmental monitoring, disaster response, and urban development. Most CD models assume co-registered inputs, yet real-world imagery often exhibits parallax, viewpoint shifts, and long temporal gaps that cause severe misalignment. Traditional two stage methods that first register and then detect, as well as recent joint frameworks (e.g., BiFA, ChangeRD), still struggle under large displacements, relying on regression only flow, global homographies, or synthetic perturbations. We present DiffRegCD, an integrated framework that unifies dense registration and change detection in a single model. DiffRegCD reformulates correspondence estimation as a Gaussian smoothed classification task, achieving sub-pixel accuracy and stable training. It leverages frozen multi-scale features from a pretrained denoising diffusion model, ensuring robustness to illumination and viewpoint variation. Supervision is provided through controlled affine perturbations applied to standard CD datasets, yielding paired ground truth for both flow and change detection without pseudo labels. Extensive experiments on aerial (LEVIR-CD, DSIFN-CD, WHU-CD, SYSU-CD) and ground level (VL-CMU-CD) datasets show that DiffRegCD consistently surpasses recent baselines and remains reliable under wide temporal and geometric variation, establishing diffusion features and classification based correspondence as a strong foundation for unified change detection.

---

## 11. Occlusion-Aware Ground Target Search by a UAV in an Urban Environment

**论文链接:** [http://arxiv.org/abs/2511.07822v1](http://arxiv.org/abs/2511.07822v1)

**作者:** Collin Hague, Artur Wolek

**发布时间:** 2025-11-11

**备注:** 18 pages, 18 figures, 5 tables

### GPT解析

### 总结

本文研究使用无人机在城市道路网络中搜索移动兴趣点的问题，提出了一种基于概率可见体积的路径规划方法。

### 背景

无人机在城市环境中搜索移动目标时，面临视线可能被遮挡的挑战，且无人机传感器存在虚警概率。

### 目的

开发一种有效的搜索策略，使无人机能够高效地在城市道路网络中找到移动的兴趣点。

### 方法

将无人机建模为变速度Dubins车辆，利用概率可见体积表示传感约束，结合迭代加深A*算法进行路径规划，并通过最大池化创建可变时间步规划器。

### 主要发现

通过蒙特卡洛模拟验证，所提出的方法在复杂环境和传感器高虚警概率情况下优于基线方法。

### 结论

基于概率可见体积的路径规划方法能有效平衡长期和短期规划，提高无人机在城市环境中搜索移动目标的效率。

### 翻译

本文考虑使用无人机在城市道路网络中搜索沿道路移动的兴趣点的问题。无人机被建模为具有视线传感器的变速度Dubins车辆，在城市环境中可能存在视线遮挡问题。论文提出了一种利用概率可见体积的搜索策略，结合迭代加深A*算法来规划未来运动。概率可见体积是针对POI状态特定分布的时间变化三维表示，用于表示传感约束。规划器使用启发式方法来乐观估计在时间范围内看到POI的概率，以找到最有可能看到POI的路径。通过最大池化概率可见体积创建可变时间步规划器，以减少搜索空间并平衡长期和短期规划。通过蒙特卡洛模拟将提出的路径规划方法与先前工作进行了比较，结果表明在高虚警概率环境下，该方法优于基线方法。


### 论文摘要

This paper considers the problem of searching for a point of interest (POI) moving along an urban road network with an uncrewed aerial vehicle (UAV). The UAV is modeled as a variable-speed Dubins vehicle with a line-of-sight sensor in an urban environment that may occlude the sensor's view of the POI. A search strategy is proposed that exploits a probabilistic visibility volume (VV) to plan its future motion with iterative deepening $A^\ast$. The probabilistic VV is a time-varying three-dimensional representation of the sensing constraints for a particular distribution of the POI's state. To find the path most likely to view the POI, the planner uses a heuristic to optimistically estimate the probability of viewing the POI over a time horizon. The probabilistic VV is max-pooled to create a variable-timestep planner that reduces the search space and balances long-term and short-term planning. The proposed path planning method is compared to prior work with a Monte-Carlo simulation and is shown to outperform the baseline methods in cluttered environments when the UAV's sensor has a higher false alarm probability.

---

## 12. DIAL-GS: Dynamic Instance Aware Reconstruction for Label-free Street Scenes with 4D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2511.06632v1](http://arxiv.org/abs/2511.06632v1)

**作者:** Chenpeng Su, Wenhua Wu, Chensheng Peng, Tianchen Deng, Zhe Liu, Hesheng Wang

**发布时间:** 2025-11-10

### GPT解析

### 总结

DIAL-GS是一种创新的动态感知重建方法，用于无标签街道场景，通过4D高斯溅射技术实现高质量的城市场景重建和实例级编辑。

### 背景

城市场景重建对自动驾驶至关重要，能够为数据合成和闭环测试提供结构化3D表示。现有方法存在局限性：监督方法依赖昂贵的人工标注且缺乏可扩展性，而当前自监督方法常混淆静态和动态元素，无法区分单个动态对象，限制了精细编辑能力。

### 目的

开发一种能够准确识别动态实例、实现动态自适应和感知感知重建的城市场景重建方法，同时增强重建的完整性和一致性。

### 方法

提出DIAL-GS，一种基于4D高斯溅射的无标签街道场景动态感知重建方法。首先，通过利用渲染失真与实际观测之间的外观-位置不一致性准确识别动态实例。然后在实例级动态感知的指导下，使用实例感知的4D高斯作为统一体积表示，实现动态自适应和感知感知重建。此外，引入双向机制，使身份和动态相互强化，增强完整性和一致性。

### 主要发现

在驾驶场景实验中，DIAL-GS在重建质量和实例级编辑方面优于现有自监督基线方法。

### 结论

DIAL-GS为城市场景建模提供了一种简洁而强大的解决方案，能够有效处理动态对象并实现高质量重建。

### 翻译

城市场景重建对自动驾驶至关重要，能够为数据合成和闭环测试提供结构化的3D表示。监督方法依赖昂贵的人工标注且缺乏可扩展性，而当前自监督方法常混淆静态和动态元素，无法区分单个动态对象，限制了精细编辑。我们提出DIAL-GS，一种基于4D高斯溅射的无标签街道场景动态感知重建方法。我们首先通过利用渲染失真与实际观测之间的不一致性准确识别动态实例。在实例级动态感知的指导下，我们使用实例感知的4D高斯作为统一体积表示，实现动态自适应和感知感知重建。此外，我们引入双向机制，使身份和动态相互强化，增强完整性和一致性。在驾驶场景实验中，DIAL-GS在重建质量和实例级编辑方面优于现有自监督基线，为城市场景建模提供了简洁而强大的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶场景中城市街景的动态实例感知重建问题，特别是在没有人工标注的情况下准确区分静态背景和动态对象。这个问题对自动驾驶至关重要，因为准确重建和区分静态与动态元素能为自动驾驶系统提供可靠的环境理解，支持数据合成、闭环测试和安全决策，同时不依赖昂贵的人工标注大大降低了成本并提高了技术的可扩展性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性进行思考：监督方法依赖昂贵人工标注且难以扩展，自监督方法则存在动态-静态混淆且缺乏实例感知能力。作者借鉴了3D高斯溅射的高效渲染特性、PVG的4D高斯表示、BoT-SORT等2D追踪器以及高斯分组的ID嵌入思想，创新性地利用运动导致的外观-位置不一致性来识别动态对象，并设计了相互促进的身份-动态训练机制，使身份嵌入和动态属性相互增强，解决了现有方法的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用运动导致的不一致性来识别动态对象，使用实例感知的4D高斯作为统一表示，并通过相互促进的身份-动态训练机制增强重建质量。整体采用两阶段流程：阶段1通过比较渲染帧与真实帧的外观和位置不一致性计算动态分数，识别动态实例并生成ID标签和动态掩码；阶段2使用包含ID嵌入和动态属性的实例感知4D高斯进行重建，应用身份损失、动态属性正则化、3D身份损失和动态一致性损失等函数，最终实现支持实例级编辑的高质量街景重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 基于外观-位置不一致性的实例级动态感知算法，解决动态-静态混淆问题；2) 实例感知的4D高斯表示，通过ID嵌入实现统一建模；3) 相互促进的身份-动态训练策略，使身份和动态属性相互增强。相比之前工作，DIAL-GS无需人工标注，能准确区分静态和动态对象，支持细粒度的实例级编辑而非粗略分解，通过相互训练机制提升了ID嵌入的完整性和动态建模的一致性，实现了现有自监督方法不具备的能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DIAL-GS通过利用运动引起的不一致性和相互促进的身份-动态训练机制，实现了无需标签的街景动态实例感知重建，支持细粒度的实例级编辑，显著提升了自动驾驶场景重建的质量和实用性。'}


### 论文摘要

Urban scene reconstruction is critical for autonomous driving, enabling structured 3D representations for data synthesis and closed-loop testing. Supervised approaches rely on costly human annotations and lack scalability, while current self-supervised methods often confuse static and dynamic elements and fail to distinguish individual dynamic objects, limiting fine-grained editing. We propose DIAL-GS, a novel dynamic instance-aware reconstruction method for label-free street scenes with 4D Gaussian Splatting. We first accurately identify dynamic instances by exploiting appearance-position inconsistency between warped rendering and actual observation. Guided by instance-level dynamic perception, we employ instance-aware 4D Gaussians as the unified volumetric representation, realizing dynamic-adaptive and instance-aware reconstruction. Furthermore, we introduce a reciprocal mechanism through which identity and dynamics reinforce each other, enhancing both integrity and consistency. Experiments on urban driving scenarios show that DIAL-GS surpasses existing self-supervised baselines in reconstruction quality and instance-level editing, offering a concise yet powerful solution for urban scene modeling.

---

## 13. VDNeRF: Vision-only Dynamic Neural Radiance Field for Urban Scenes

**论文链接:** [http://arxiv.org/abs/2511.06408v1](http://arxiv.org/abs/2511.06408v1)

**作者:** Zhengyu Zou, Jingfeng Li, Hao Li, Xiaolei Hou, Jinwen Hu, Jingkun Chen, Lechao Cheng, Dingwen Zhang

**发布时间:** 2025-11-09

### GPT解析

### 总结

本文提出了一种名为VDNeRF的新方法，能够在不依赖额外相机位姿信息或昂贵传感器数据的情况下，准确恢复相机轨迹并学习动态城市场景的时空表示，实现逼真的新视图渲染。

### 背景

现有的NeRF方法在自动驾驶和机器人感知等应用中面临挑战，主要因为难以获取准确的相机位姿以及处理大规模动态环境的能力有限。

### 目的

解决现有NeRF方法在相机位姿获取和动态环境处理方面的局限性，开发一种不需要额外传感器数据的方法来准确恢复相机轨迹并处理动态场景。

### 方法

VDNeRF采用两个独立的NeRF模型联合重建场景：静态NeRF模型优化相机位姿和静态背景，动态NeRF模型集成3D场景流以确保动态对象的准确一致重建。同时设计了一个有效的训练框架来解决相机运动和独立物体运动之间的模糊性问题，实现鲁棒的相机位姿估计和场景中静态与动态元素的自监督分解。

### 主要发现

在主流城市驾驶数据集上的大量评估表明，VDNeRF在相机位姿估计和动态新视图合成方面都超越了最先进的基于NeRF的无位姿方法。

### 结论

VDNeRF为解决NeRF在动态环境中的应用挑战提供了一种有效的方法，特别是在需要精确相机轨迹和动态场景表示的自动驾驶和机器人感知领域具有应用价值。

### 翻译

神经辐射场使用具有已知相机位姿的一组图像隐式地建模连续的三维场景，能够渲染出逼真的新视图。然而，现有的基于NeRF的方法在自动驾驶和机器人感知等应用中遇到挑战，主要由于难以获取准确的相机位姿以及处理大规模动态环境的局限性。为解决这些问题，我们提出了仅使用视觉的动态NeRF（VDNeRF），该方法能够在不依赖额外相机位姿信息或昂贵传感器数据的情况下准确恢复相机轨迹并学习动态城市场景的时空表示。VDNeRF采用两个独立的NeRF模型联合重建场景，静态NeRF模型优化相机位姿和静态背景，而动态NeRF模型集成3D场景流以确保动态对象的准确一致重建。为了解决相机运动和独立物体运动之间的模糊性问题，我们设计了一个有效的训练框架来实现鲁棒的相机位姿估计和场景中静态与动态元素的自监督分解。在主流城市驾驶数据集上的大量评估表明，VDNeRF在相机位姿估计和动态新视图合成方面都超越了最先进的基于NeRF的无位姿方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有NeRF方法在动态城市场景中面临的两个关键挑战：一是难以获取准确的相机姿态信息，二是难以处理大规模动态环境。这个问题在现实中非常重要，因为自动驾驶、机器人感知和AR/VR等应用需要准确的环境表示，而传统方法依赖于结构从运动(SfM)等技术获取相机姿态，这些方法在包含动态元素的城市环境中往往不可靠，导致场景重建质量下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性来设计VDNeRF。他们发现RoDynRF依赖于精确的2D运动掩码且在大型城市场景中表现不佳，而LocalRF和NoPe-NeRF等静态场景方法无法处理动态元素。作者借鉴了多个现有工作：使用两个分离的NeRF模型表示静态和动态部分（类似D2NeRF和EmerNeRF），采用渐进式优化方法（类似LocalRF），利用3D场景流捕捉动态对象运动（类似NSFF和DynamicNeRF），以及引入阴影权重融合静态和动态表示（类似D2NeRF）。通过整合这些思想并设计特定训练框架，作者解决了相机姿态估计和静态-动态分解之间的模糊性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用两个分离的NeRF模型（静态NeRF和动态NeRF）分别表示场景的静态背景和动态前景，通过精心设计的训练框架实现鲁棒的相机姿态估计和自监督的静态-动态分解，并利用3D场景流准确捕捉动态对象的运动。整体流程包括：1)将城市场景划分为多个重叠的子场景；2)对每个子场景进行三个训练阶段：渐进式优化静态NeRF和相机姿态（使用运动掩码减少干扰）、固定相机姿态并激活动态NeRF重建动态对象、创建新子场景保持一致性；3)使用多种损失函数（颜色、深度、光流、循环一致性、动态和阴影损失）优化模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出仅使用视觉信息就能在动态城市场景中准确恢复相机轨迹的方法；2)设计两个分离的NeRF模型处理静态背景和动态对象；3)引入3D场景流提高动态对象重建质量；4)设计特定训练框架处理相机运动和独立物体运动间的模糊性；5)实现自监督的静态-动态分解，无需精确运动掩码。相比之前工作，VDNeRF不依赖RoDynRF的2D运动掩码，能处理LocalRF和NoPe-NeRF无法处理的动态场景，且不需要EmerNeRF的真实相机姿态输入，在相机姿态估计和动态新视图合成方面表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VDNeRF提出了一种仅使用图像序列就能在动态城市场景中准确恢复相机轨迹、实现时空场景表示和高质量动态新视图合成的创新方法，通过两个分离的NeRF模型和精心设计的训练框架，解决了相机姿态估计和静态-动态分解的关键挑战。'}


### 论文摘要

Neural Radiance Fields (NeRFs) implicitly model continuous three-dimensional scenes using a set of images with known camera poses, enabling the rendering of photorealistic novel views. However, existing NeRF-based methods encounter challenges in applications such as autonomous driving and robotic perception, primarily due to the difficulty of capturing accurate camera poses and limitations in handling large-scale dynamic environments. To address these issues, we propose Vision-only Dynamic NeRF (VDNeRF), a method that accurately recovers camera trajectories and learns spatiotemporal representations for dynamic urban scenes without requiring additional camera pose information or expensive sensor data. VDNeRF employs two separate NeRF models to jointly reconstruct the scene. The static NeRF model optimizes camera poses and static background, while the dynamic NeRF model incorporates the 3D scene flow to ensure accurate and consistent reconstruction of dynamic objects. To address the ambiguity between camera motion and independent object motion, we design an effective and powerful training framework to achieve robust camera pose estimation and self-supervised decomposition of static and dynamic elements in a scene. Extensive evaluations on mainstream urban driving datasets demonstrate that VDNeRF surpasses state-of-the-art NeRF-based pose-free methods in both camera pose estimation and dynamic novel view synthesis.

---

## 14. Environment-Aware MIMO Channel Estimation in Pilot-Constrained Upper Mid-Band Systems

**论文链接:** [http://arxiv.org/abs/2511.05771v1](http://arxiv.org/abs/2511.05771v1)

**作者:** Seyed Alireza Javid, Nuria González-Prelcic

**发布时间:** 2025-11-08

**备注:** Submitted to ICASSP 2026

### GPT解析

### 总结

本文提出了一种新颖的物理信息神经网络框架，结合基于模型的信道估计与深度网络，在导频受限场景下实现了优越性能，在复杂环境中仅需少量导频就能获得显著性能提升。

### 背景

准确的MIMO信道估计对下一代无线系统至关重要，能够提高通信和感知性能。然而，传统的基于模型的信道估计方法在复杂环境和有限导频数量下性能下降，而纯数据驱动方法缺乏物理解释性，需要大量数据收集，且通常具有站点特异性。

### 目的

开发一种结合基于模型的信道估计与深度网络的方法，利用传播环境的先验信息，在导频受限场景下实现优越性能。

### 方法

提出了一种物理信息神经网络框架，采用增强的U-Net架构和交叉注意力机制，将初始信道估计与接收信号强度地图融合，提供精细化的信道估计。

### 主要发现

使用来自城市环境的真实射线追踪数据进行全面评估，结果表明与最先进的方法相比，归一化均方误差获得了超过5分贝的提升，在导频受限场景下表现特别出色，且在不同频率和环境下具有鲁棒性，仅需最微调。

### 结论

所提出的框架保持了实际的计算复杂度，使其在上中频段的大规模MIMO系统中具有可行性。

### 翻译

准确的MIMO信道估计对下一代无线系统至关重要，能够提高通信和感知性能。然而，传统的基于模型的信道估计方法在复杂环境和有限导频数量下性能下降，而纯数据驱动方法缺乏物理解释性，需要大量数据收集，且通常具有站点特异性。本文提出了一种新颖的物理信息神经网络框架，结合了基于模型的信道估计与深度网络，利用传播环境的先验信息，在导频受限场景下实现优越性能。所提出的方法采用增强的U-Net架构和交叉注意力机制，将初始信道估计与接收信号强度地图融合，提供精细化的信道估计。使用来自城市环境的真实射线追踪数据进行全面评估，结果表明与最先进的方法相比，归一化均方误差获得了超过5分贝的提升，在导频受限场景下表现特别出色，且在不同频率和环境下具有鲁棒性，仅需最微调。所提出的框架保持了实际的计算复杂度，使其在上中频段的大规模MIMO系统中具有可行性。


### 论文摘要

Accurate multiple-input multiple-output (MIMO) channel estimation is critical for next-generation wireless systems, enabling enhanced communication and sensing performance. Traditional model-based channel estimation methods suffer, however, from performance degradation in complex environments with a limited number of pilots, while purely data-driven approaches lack physical interpretability, require extensive data collection, and are usually site-specific. This paper presents a novel physics-informed neural network (PINN) framework that combines model-based channel estimation with a deep network to exploit prior information about the propagation environment and achieve superior performance under pilot-constrained scenarios. The proposed approach employs an enhanced U-Net architecture with cross-attention mechanisms to fuse initial channel estimates with received signal strength (RSS) maps to provide refined channel estimates. Comprehensive evaluation using realistic ray-tracing data from urban environments demonstrates significant performance improvements, achieving over 5 dB gain in normalized mean squared error (NMSE) compared to state-of-the-art methods, with particularly strong performance in pilot-limited scenarios and robustness across different frequencies and environments with only minimal fine-tuning. The proposed framework maintains practical computational complexity, making it viable for massive MIMO systems in upper mid-band frequencies.

---

## 15. Reasoning Is All You Need for Urban Planning AI

**论文链接:** [http://arxiv.org/abs/2511.05375v1](http://arxiv.org/abs/2511.05375v1)

**作者:** Sijie Yang, Jiatong Li, Filip Biljecki

**发布时间:** 2025-11-07

**备注:** Submitted to AAAI 2026 Workshop AI4UP

### GPT解析

### 总结

该论文提出了一个代理式城市规划AI框架，整合了三个认知层和六个逻辑组件，通过多代理协作实现AI辅助的城市规划决策，强调AI应增强而非取代人类规划者的判断能力。

### 背景

AI在城市规划分析中已证明成功，能够从数据学习模式预测未来状况。下一个前沿是AI辅助决策，包括推荐地点、分配资源、评估权衡，同时透明推理约束和利益相关者价值观。

### 目的

开发一个具备推理能力的规划代理框架，使AI能够系统探索解决方案空间、验证法规合规性、透明地权衡利弊，从而增强人类规划者的决策能力。

### 方法

提出Agentic Urban Planning AI Framework，整合三个认知层（感知、基础、推理）和六个逻辑组件（分析、生成、验证、评估、协作、决策），通过多代理协作框架实现。

### 主要发现

规划决策需要明确的推理能力，包括基于价值观（应用规范性原则）、基于规则（保证约束满足）和可解释性（生成透明理由）的能力，这些是统计学习无法单独满足的。

### 结论

该框架展示了AI代理如何通过计算推理能力增强人类规划者的判断，而非取代人类判断，使AI成为城市规划的有效辅助工具。

### 翻译

AI已被证明在城市规划分析中高度成功——从数据中学习模式以预测未来状况。下一个前沿是AI辅助决策：能够推荐地点、分配资源、评估权衡，同时透明地推理约束和利益相关者价值观的代理。推理AI的最新突破——思维链提示、ReAct和多代理协作框架——现在使这一愿景成为可能。本文提出了代理式城市规划AI框架，用于具备推理能力的规划代理，该框架通过多代理协作框架将三个认知层（感知、基础、推理）与六个逻辑组件（分析、生成、验证、评估、协作、决策）相结合。我们证明了为什么规划决策需要明确的推理能力，包括基于价值观（应用规范性原则）、基于规则（保证约束满足）和可解释性（生成透明理由）的要求——这些是统计学习无法单独满足的。我们将推理代理与统计学习进行了比较，提出了包含基准评估指标的全面架构，并概述了关键研究挑战。该框架展示了AI代理如何通过系统探索解决方案空间、验证法规合规性、透明地权衡利弊来增强人类规划者——不是用计算推理能力取代人类判断，而是增强它。


### 论文摘要

AI has proven highly successful at urban planning analysis -- learning patterns from data to predict future conditions. The next frontier is AI-assisted decision-making: agents that recommend sites, allocate resources, and evaluate trade-offs while reasoning transparently about constraints and stakeholder values. Recent breakthroughs in reasoning AI -- CoT prompting, ReAct, and multi-agent collaboration frameworks -- now make this vision achievable.   This position paper presents the Agentic Urban Planning AI Framework for reasoning-capable planning agents that integrates three cognitive layers (Perception, Foundation, Reasoning) with six logic components (Analysis, Generation, Verification, Evaluation, Collaboration, Decision) through a multi-agents collaboration framework. We demonstrate why planning decisions require explicit reasoning capabilities that are value-based (applying normative principles), rule-grounded (guaranteeing constraint satisfaction), and explainable (generating transparent justifications) -- requirements that statistical learning alone cannot fulfill. We compare reasoning agents with statistical learning, present a comprehensive architecture with benchmark evaluation metrics, and outline critical research challenges. This framework shows how AI agents can augment human planners by systematically exploring solution spaces, verifying regulatory compliance, and deliberating over trade-offs transparently -- not replacing human judgment but amplifying it with computational reasoning capabilities.

---

## 16. Optimizing Sensor Placement in Urban Storm Sewers: A Data-Driven Sparse Sensing Approach

**论文链接:** [http://arxiv.org/abs/2511.04556v1](http://arxiv.org/abs/2511.04556v1)

**作者:** Zihang Ding, Kun Zhang

**发布时间:** 2025-11-06

**备注:** 32 pages (including supplementary information), 11 figures (and 7 figures in supplementary). Submitted to Nature Water. Partially presented at HydroML 2025 Symposium, Minnesota Water Resources Conference 2025, and will be presented at AGU Fall Meeting 2025

### GPT解析

### 总结

该研究提出了一种数据驱动的稀疏传感（DSS）框架，与EPA-SWMM模型集成，用于优化城市排水系统中的传感器布点并重建峰值流量。通过在明尼苏达州德卢斯市的Woodland Avenue流域进行案例研究，证明仅需3个优化放置的传感器即可实现高精度的流量重建（NSE值0.92-0.95），为资源有限条件下的城市洪水监测提供了有效解决方案。

### 背景

城市地表水淹问题日益频繁和广泛，主要由高强度降雨超出排水系统容量引起。尽管高时空分辨率的洪水预测和监测需求迫切，但时间、预算和技术限制阻碍了其全面实施。如何在资源有限的情况下监测城市排水网络并预测水流状况成为重大挑战。

### 目的

研究在资源约束条件下监测城市排水网络并预测水流状况的方法，优化传感器布点并重建暴雨系统中的峰值流量。

### 方法

提出数据驱动的稀疏传感（DSS）框架，与EPA-SWMM模型集成。使用SWMM模型生成训练数据集，应用奇异值分解进行降维和QR分解进行传感器分配，以确定最优监测节点。通过比较DSS重建的峰值流量与SWMM结果验证代表性。以明尼苏达州德卢斯市的Woodland Avenue流域为案例研究。

### 主要发现

在77个节点中使用3个优化放置的传感器实现了令人满意的重建性能，纳什-萨克利夫效率（NSE）值在0.92-0.95之间（第25至75百分位）。模型对测量不确定性具有良好的鲁棒性，对传感器故障的鲁棒性取决于位置并随部署传感器数量增加而提高。

### 结论

该框架在计算效率和物理可解释性之间取得平衡，实现了用最少的传感器进行高精度流量重建。该DSS框架可进一步与预测模型集成，在有限的传感和监测资源条件下实现洪水预警和实时控制。

### 翻译

城市地表水淹，由高强度降雨超出排水系统容量而引发，正变得越来越频繁和广泛。尽管期望高时空分辨率的洪水预测和监测，但时间、预算和技术方面的实际限制阻碍了其全面实施。如何在资源有限的情况下监测城市排水网络并预测水流状况是一个主要挑战。本研究提出了一种数据驱动的稀疏传感（DSS）框架，与EPA-SWMM集成，用于优化传感器布点并重建雨水系统中的峰值流量，以明尼苏达州德卢斯市的Woodland Avenue流域为案例研究。我们使用SWMM模型生成整个雨水网络峰值流量剖面的训练数据集。此外，我们应用DSS框架——利用奇异值分解进行降维和QR分解进行传感器分配——基于模拟训练数据集确定最佳监测节点。然后，我们通过比较DSS重建的峰值流量剖面与SWMM获得的剖面来验证这些识别的监测节点的代表性。在77个节点中，三个优化放置的传感器实现了令人满意的重建性能，纳什-萨克利夫效率（NSE）值在0.92-0.95之间（第25至75百分位）。此外，该模型对测量不确定性表现出良好的鲁棒性。其对传感器故障的鲁棒性取决于位置，并随部署传感器数量的增加而提高。该框架在计算效率和物理可解释性之间取得平衡，实现了用最少的传感器进行高精度流量重建。该DSS框架可以进一步与预测模型集成，在有限的传感和监测资源条件下实现洪水预警和实时控制。


### 论文摘要

Urban surface water flooding, triggered by intense rainfall overwhelming drainage systems, is increasingly frequent and widespread. While flood prediction and monitoring in high spatial-temporal resolution are desired, practical constraints in time, budget, and technology hinder its full implementation. How to monitor urban drainage networks and predict flow conditions under constrained resource is a major challenge. This study presents a data-driven sparse sensing (DSS) framework, integrated with EPA-SWMM, to optimize sensor placement and reconstruct peak flowrates in a stormwater system, using the Woodland Avenue catchment in Duluth, Minnesota, as a case study. We utilized a SWMM model to generate a training dataset of peak flowrate profiles across the stormwater network. Furthermore, we applied DSS - leveraging singular value decomposition for dimensionality reduction and QR factorization for sensor allocation - to identify the optimal monitoring nodes based on the simulated training dataset. We then validated the representativeness of these identified monitoring nodes by comparing the DSS-reconstructed peak flowrate profiles with those obtained from SWMM. Three optimally placed sensors among 77 nodes achieved satisfactory reconstruction performance with Nash-Sutcliffe Efficiency (NSE) values of 0.92-0.95 (25th to 75th percentiles). In addition, the model showed good robustness to uncertainty in measurements. Its robustness to sensor failures is location-dependent and improves with the number of sensors deployed. The framework balances computational efficiency and physical interpretability, enabling high-accuracy flow reconstruction with minimal sensors. This DSS framework can be further integrated with predictive models to realize flood early warning and real-time control under limited sensing and monitoring resource.

---

## 17. Correlation and Temporal Consistency Analysis of Mono-static and Bi-static ISAC Channels

**论文链接:** [http://arxiv.org/abs/2511.03837v1](http://arxiv.org/abs/2511.03837v1)

**作者:** Saúl Fenollosa, Narcis Cardona, Wenfei Yang, Jian Li

**发布时间:** 2025-11-05

**备注:** 6 pages, 7 figures, 2 tables. Accepted for publication at the 2025 IEEE Global Communications Conference (GLOBECOM), WS-26: 4th Workshop on Propagation Channel Models and Evaluation Methodologies for 6G

### GPT解析

### 总结

该研究通过实证测量揭示了单基地和双基地ISAC信道之间的关系，发现它们虽瞬时相关性低但具有统一的时序一致性，为ISAC系统设计和标准化提供了重要见解。

### 背景

集成感知与通信(ISAC)对于未来无线网络(如6G)中高效利用频谱和硬件至关重要。

### 目的

填补现有信道模型对ISAC特定特性表征的不足，特别是单基地和双基地感知配置之间的关系。

### 方法

在动态城市微小区(UMi)环境中使用79GHz FMCW信道探测器进行实证测量，并在七个真实场景中验证结果。

### 主要发现

单基地和双基地信道由于不同的传播几何结构表现出持续的低瞬时相关性；尽管如此，两种信道都具有统一的时序一致性，在环境动力学下可预测地演变。

### 结论

这些发现为稳健的ISAC系统设计和未来标准化提供了重要依据。

### 翻译

集成感知与通信(ISAC)对于未来无线网络(如6G)中高效利用频谱和硬件至关重要。然而，现有信道模型缺乏对ISAC特定特性的全面表征，特别是单基地(收发器同置)和双基地(收发器分离)感知配置之间的关系。在动态城市微小区(UMi)环境中使用79GHz FMCW信道探测器进行的实证测量有助于填补这一空白。研究证明了两个关键发现：(1)单基地和双基地信道由于不同的传播几何结构表现出持续的低瞬时相关性；(2)尽管瞬时相关性低，但两种信道都具有统一的时序一致性，在环境动力学下可预测地演变。这些见解在七个真实场景(带有移动目标/收发器)中得到验证，为稳健的ISAC系统设计和未来标准化提供了指导。


### 论文摘要

Integrated Sensing and Communication (ISAC) is critical for efficient spectrum and hardware utilization in future wireless networks like 6G. However, existing channel models lack comprehensive characterization of ISAC-specific dynamics, particularly the relationship between mono-static (co-located Tx/Rx) and bi-static (separated Tx/Rx) sensing configurations. Empirical measurements in dynamic urban microcell (UMi) environments using a 79-GHz FMCW channel sounder help bridge this gap. Two key findings are demonstrated: (1) mono-static and bi-static channels exhibit consistently low instantaneous correlation due to divergent propagation geometries; (2) despite low instantaneous correlation, both channels share unified temporal consistency, evolving predictably under environmental kinematics. These insights, validated across seven real-world scenarios with moving targets/transceivers, inform robust ISAC system design and future standardization.

---

## 18. Security-Aware Joint Sensing, Communication, and Computing Optimization in Low Altitude Wireless Networks

**论文链接:** [http://arxiv.org/abs/2511.01451v1](http://arxiv.org/abs/2511.01451v1)

**作者:** Jiacheng Wang, Changyuan Zhao, Jialing He, Geng Sun, Weijie Yuan, Dusit Niyato, Liehuang Zhu, Tao Xiang

**发布时间:** 2025-11-03

**备注:** 14 pages, 10 figures

### GPT解析

### 总结

本文针对低空无线网络中集成感知、通信和计算(ISCC)的联合性能优化问题，提出了一种基于深度Q网络的多目标进化算法，有效平衡了感知精度、通信保密性和信息新鲜性。

### 背景

随着地面资源日益饱和，研究注意力转向低空空域，催生了城市空中出租车和空中检测等新兴应用。低空无线网络(LAWNs)是这些应用的基础，而ISCC是LAWNs的核心部分之一。

### 目的

研究在考虑通信保密性的情况下，对ISCC进行联合性能优化，以应对低空空域开放性带来的安全威胁。

### 方法

推导波束图误差、保密率和信息年龄作为性能指标，构建多目标优化问题，提出基于深度Q网络的多目标进化算法，根据演化目标自适应选择进化算子。

### 主要发现

所提出的方法在感知精度、通信保密性和信息新鲜性之间实现了比基线算法更好的平衡，有效保护了ISCC性能和LAWN支持的低空应用。

### 结论

通过联合优化ISCC性能并考虑通信保密性，所提出的方法能够有效保障低空无线网络支持的各类应用的可靠性。

### 翻译

随着地面资源日益饱和，研究注意力正转向低空空域，出现了城市空中出租车和空中检测等新兴应用。低空无线网络(LAWNs)是这些应用的基础，而集成感知、通信和计算(ISCC)是LAWNs的核心部分之一。然而，低空空域的开放性使通信面临安全威胁，降低了ISCC性能，最终损害了LAWN支持的应用的可靠性。为应对这些挑战，本文研究了在考虑通信保密性的情况下对ISCC进行联合性能优化。具体而言，我们推导了波束图误差、保密率和信息年龄(AoI)作为感知、保密通信和计算的性能指标。基于这些指标，我们制定了一个多目标优化问题，该问题平衡了感知和计算性能，同时保持通信被检测到的概率低于所需阈值。然后，我们提出了一种基于深度Q网络(DQN)的多目标进化算法，该算法根据演化的优化目标自适应选择进化算子，从而获得更有效的解决方案。广泛的模拟表明，与基线算法相比，所提出的方法在感知精度、通信保密性和信息新鲜性之间实现了更好的平衡，从而保护了ISCC性能和LAWN支持的低空应用。


### 论文摘要

As terrestrial resources become increasingly saturated, the research attention is shifting to the low-altitude airspace, with many emerging applications such as urban air taxis and aerial inspection. Low-Altitude Wireless Networks (LAWNs) are the foundation for these applications, with integrated sensing, communications, and computing (ISCC) being one of the core parts of LAWNs. However, the openness of low-altitude airspace exposes communications to security threats, degrading ISCC performance and ultimately compromising the reliability of applications supported by LAWNs. To address these challenges, this paper studies joint performance optimization of ISCC while considering secrecyness of the communications. Specifically, we derive beampattern error, secrecy rate, and age of information (AoI) as performance metrics for sensing, secrecy communication, and computing. Building on these metrics, we formulate a multi-objective optimization problem that balances sensing and computation performance while keeping the probability of communication being detected below a required threshold. We then propose a deep Q-network (DQN)-based multi-objective evolutionary algorithm, which adaptively selects evolutionary operators according to the evolving optimization objectives, thereby leading to more effective solutions. Extensive simulations show that the proposed method achieves a superior balance among sensing accuracy, communication secrecyness, and information freshness compared with baseline algorithms, thereby safeguarding ISCC performance and LAWN-supported low-altitude applications.

---

## 19. Optimizing Energy and Latency in 6G Smart Cities with Edge CyberTwins

**论文链接:** [http://arxiv.org/abs/2511.00955v3](http://arxiv.org/abs/2511.00955v3)

**作者:** Amine Abouaomar, Badr Ben Elallid, Nabil Benamar

**发布时间:** 2025-11-02

### GPT解析

### 总结

论文提出了一种边缘感知的CyberTwin框架，结合混合联邦学习，用于智慧城市中6G网络切片的能量-延迟协同优化，解决了大规模物联网设备部署中的挑战。

### 背景

智慧城市中物联网设备的激增给6G网络带来了挑战，特别是在不同切片之间存在相互冲突的能量-延迟需求，现有方法难以处理大规模部署（超过50,000台设备/平方公里）的能量-延迟权衡问题。

### 目的

开发一种边缘感知的CyberTwin框架，用于6G网络切片中的能量-延迟协同优化，以支持大规模物联网设备部署。

### 方法

提出结合混合联邦学习的CyberTwin框架，为延迟敏感切片采用集中式AI调度，为非关键切片采用分布式联邦学习，并通过基于压缩感知的数字孪生和可再生能源感知的资源分配进行增强；使用三层架构混合调度器和基于物理不可克隆函数(PUF)的安全认证。

### 主要发现

全面模拟显示，与非实时切片相比，比扩散-强化学习基线减少52%的能量消耗，同时为URLLC应用保持0.9ms延迟，99.1% SLA合规性；框架可扩展至50,000台设备/平方公里，CPU开销低于25%。

### 结论

所提出的CyberTwin框架有效解决了智慧城市大规模物联网设备部署中的能量-延迟权衡问题，通过混合联邦学习架构实现了高效的网络切片优化。

### 翻译

智慧城市中物联网设备的激增给6G网络带来了挑战，不同切片之间存在相互冲突的能量-延迟需求。现有方法难以处理能量-延迟权衡，特别是在大规模部署超过50,000台设备/平方公里的情况下。本文提出了一种边缘感知的CyberTwin框架，整合混合联邦学习，用于6G网络切片中的能量-延迟协同优化。我们的方法将延迟敏感切片的集中式人工智能调度与非关键切片的分布式联邦学习相结合，并通过基于压缩感知的数字孪生和可再生能源感知的资源分配进行增强。混合调度器利用三层架构，基于物理不可克隆函数(PUF)的安全认证达到99.7%的攻击检测准确率。全面的模拟表明，与非实时切片相比，比扩散-强化学习基线减少52%的能源消耗，同时为URLLC应用保持0.9ms的延迟，99.1%的SLA合规性。该框架可扩展至50,000台设备/平方公里，CPU开销低于25%，通过NS-3混合模拟在真实智慧城市场景中得到验证。


### 论文摘要

The proliferation of IoT devices in smart cities challenges 6G networks with conflicting energy-latency requirements across heterogeneous slices. Existing approaches struggle with the energy-latency trade-off, particularly for massive scale deployments exceeding 50,000 devices km. This paper proposes an edge-aware CyberTwin framework integrating hybrid federated learning for energy-latency co-optimization in 6G network slicing. Our approach combines centralized Artificial Intelligence scheduling for latency-sensitive slices with distributed federated learning for non-critical slices, enhanced by compressive sensing-based digital twins and renewable energy-aware resource allocation. The hybrid scheduler leverages a three-tier architecture with Physical Unclonable Function (PUF) based security attestation achieving 99.7% attack detection accuracy. Comprehensive simulations demonstrate 52% energy reduction for non-real-time slices compared to Diffusion-Reinforcement Learning baselines while maintaining 0.9ms latency for URLLC applications with 99.1% SLA compliance. The framework scales to 50,000 devices km with CPU overhead below 25%, validated through NS-3 hybrid simulations across realistic smart city scenarios.

---

## 20. Direct measurement of atomic number density using Single Pass Absorption Spectroscopy (SPAS)

**论文链接:** [http://arxiv.org/abs/2511.00526v1](http://arxiv.org/abs/2511.00526v1)

**作者:** Sumit Achar, Shivam Sinha, Ezhilarasan M, Chandankumar R, Arijit Sharma

**发布时间:** 2025-11-01

**备注:** Sumit Achar and Shivam Sinha contributed equally to this work

### GPT解析

### 总结

研究人员展示了使用单程吸收光谱法（SPAS）直接测量碱金属原子数密度的方法，基于铷蒸气的绝对吸收光谱建模，考虑了光泵浦和渡越时间展宽效应，与实验数据有极好的一致性，适用于广泛的温度、激光功率和电池长度范围。

### 背景

现有技术可能需要高温、基于饱和的方法进行基线校正，在测量稀释原子蒸气中的原子数密度时存在局限性。

### 目的

开发一种基于模型的方法，通过测量绝对吸收光谱来准确推断原子数密度，特别是在弱吸收区域。

### 方法

结合Lindblad形式和密度矩阵方法建模铷蒸气的绝对吸收光谱，考虑激光束功率、激光束直径和电池温度等可实验测量的参数，明确包含光泵浦和渡越时间展宽效应，并通过测量和减去光电探测器的暗电流来确保准确量化绝对吸收测量。

### 主要发现

模型与实验数据有极好的一致性（>99%），适用于广泛的温度范围（293-343 K）、激光功率（~0.2I_sat - ~2I_sat）和电池长度（2-100 mm），即使在弱吸收区域也能准确确定原子数密度。

### 结论

该方法为高温、基于饱和的基线校正方法提供了合适的替代方案，能够精确确定稀释原子蒸气中的原子数密度，适用于量子技术应用，并可扩展用于测定环境中有害气体和污染物的浓度。

### 翻译

我们展示了使用单程吸收光谱法（SPAS）直接测量碱金属原子数密度的方法。我们基于温铷（Rb）蒸气的绝对吸收光谱建模开发了该方法，并从SPAS测量中推断原子数密度。该模型结合了Lindblad形式和密度矩阵方法，并纳入了可实验测量的参数，如激光束功率、激光束直径和电池温度。该框架明确包含了光泵浦和渡越时间展宽效应，并且在广泛的温度（293-343 K）、激光功率（~0.2I_sat - ~2I_sat）和电池长度（2-100 mm）范围内，与铷蒸气电池的实验数据有极好的一致性（>99%）。为确保通过稀释原子蒸气准确量化绝对吸收测量，我们测量并减去了光电探测器的暗电流。该暗电流在没有光入射到光电探测器的情况下记录，以获得准确的基线校正。这种方法即使在弱吸收区域也能确保原子数密度的准确确定。它为高温、基于饱和的基线校正方法提供了合适的替代方案，并能够使用微型原子蒸气电池在通信、传感和计量学的量子技术应用中精确确定稀释原子蒸气中的原子数密度。此外，该方法可扩展用于测定城市、农村以及工业环境中有害气体和气体污染物的浓度。


### 论文摘要

We demonstrate a direct measurement of the atomic number density of alkali atoms using single-pass absorption spectroscopy (SPAS). We developed our methodology based on modeling the absolute absorption spectra of warm rubidium (Rb) vapor and infer the atomic number density from SPAS measurements. The model combines the Lindblad formalism with a density matrix approach and incorporates experimentally measurable parameters such as laser beam power, laser beam diameter, and cell temperature. The framework explicitly incorporates optical pumping and transit-time broadening effects and shows excellent agreement ($> 99\%$) with experimental data using rubidium vapor cells across a wide range of temperature ($293$-$343$ K), laser powers ($\sim 0.2~I_{sat}$ -$~ 2~I_{sat}$), and cell lengths ($2$-$100$ mm). To ensure accurate quantification of absolute absorption measurements through the dilute atomic vapor, we measure and subtract the dark current of the photodetectors. This dark current is recorded in the absence of any light incident on the photodetector, to obtain accurate baseline corrections. This approach ensures an accurate determination of the atomic number density, even in the weak absorption regime. It provides a suitable alternative to the high-temperature, saturation-based method for baseline correction and enables the precise determination of the atomic number density in dilute atomic vapor for quantum technology applications in communication, sensing, and metrology using miniature atomic vapor cells. Furthermore, the methodology can be extended to determine the concentration of harmful gases and gaseous pollutants in urban, rural, as well as industrial environments.

---

## 21. IoT- and AI-informed urban air quality models for vehicle pollution monitoring

**论文链接:** [http://arxiv.org/abs/2511.00187v1](http://arxiv.org/abs/2511.00187v1)

**作者:** Jan M. Armengol, Vicente Masip, Ada Barrantes, Gabriel M. Beltrami, Sergi Albiach, Daniel Rodriguez-Rey, Marc Guevara, Albert Soret, Eduardo Quiñones, Elli Kartsakli

**发布时间:** 2025-10-31

### GPT解析

### 总结

该研究通过整合低成本传感器、AI视频交通分析和高分辨率城市空气质量模型，弥合了IoT空气质量传感与物理建模之间的差距，在巴塞罗那进行了试点部署，实现了对交通相关污染物的高时间粒度预测。

### 背景

随着智能物联网系统在城市环境中的兴起，为实时环境监测带来了新机遇，但大多数研究要么专注于IoT空气质量传感，要么专注于物理建模，两者是分开进行的。

### 目的

整合低成本传感器、AI视频交通分析和高分辨率城市空气质量模型，提高城市空气质量预测的时间粒度。

### 方法

在巴塞罗那Eixample区的道路交叉口进行试点部署，系统捕获动态交通条件和环境变量，在边缘处理数据，将实时数据输入高性能计算仿真管道，并与官方NO2测量值进行验证。

### 主要发现

与传统依赖静态排放清单的模型相比，IoT辅助方法提高了交通相关污染物的城市空气质量预测的时间粒度。

### 结论

该研究展示了利用IoT-边缘-云-HPC架构的可扩展、自适应且注重隐私的城市污染监测解决方案，为下一代物联网驱动的环境智能奠定了基础。

### 翻译

随着城市环境中智能物联网(IoT)系统的兴起，为实时环境监测带来了新的机遇。虽然大多数研究要么专注于基于IoT的空气质量传感，要么专注于基于物理的建模，但这项工作通过整合低成本传感器和AI驱动的基于视频的交通分析以及高分辨率城市空气质量模型来弥合这一差距。我们在巴塞罗那Eixample区的一个道路交叉口进行了真实世界的试点部署，系统捕获动态交通条件和环境变量，在边缘处理它们，并将实时数据输入高性能计算(HPC)仿真管道。结果与官方的二氧化氮(NO2)空气质量测量值进行了验证。与传统依赖静态排放清单的模型相比，IoT辅助方法提高了交通相关污染物的城市空气质量预测的时间粒度。利用IoT-边缘-云-HPC架构的全部能力，这项工作展示了一个可扩展、自适应且注重隐私的城市污染监测解决方案，并为下一代物联网驱动的环境智能奠定了基础。


### 论文摘要

With the rise of intelligent Internet of Things (IoT) systems in urban environments, new opportunities are emerging to enhance real-time environmental monitoring. While most studies focus either on IoT-based air quality sensing or physics-based modeling in isolation, this work bridges that gap by integrating low-cost sensors and AI-powered video-based traffic analysis with high-resolution urban air quality models. We present a real-world pilot deployment at a road intersection in Barcelona's Eixample district, where the system captures dynamic traffic conditions and environmental variables, processes them at the edge, and feeds real-time data into a high-performance computing (HPC) simulation pipeline. Results are validated against official air quality measurements of nitrogen dioxide (NO2). Compared to traditional models that rely on static emission inventories, the IoT-assisted approach enhances the temporal granularity of urban air quality predictions of traffic-related pollutants. Using the full capabilities of an IoT-edge-cloud-HPC architecture, this work demonstrates a scalable, adaptive, and privacy-conscious solution for urban pollution monitoring and establishes a foundation for next-generation IoT-driven environmental intelligence.

---

## 22. Mask-to-Height: A YOLOv11-Based Architecture for Joint Building Instance Segmentation and Height Classification from Satellite Imagery

**论文链接:** [http://arxiv.org/abs/2510.27224v1](http://arxiv.org/abs/2510.27224v1)

**作者:** Mahmoud El Hussieni, Bahadır K. Güntürk, Hasan F. Ateş, Oğuz Hanoğlu

**发布时间:** 2025-10-31

**DOI:** 10.1109/ASYU67174.2025.11208374

### GPT解析

### 总结

本文分析了YOLOv11模型在建筑实例分割和高度分类中的应用，展示了其在卫星图像处理中的优越性能。

### 背景

准确建筑实例分割和高度分类对城市规划、3D城市建模和基础设施监测至关重要。研究使用包含12个城市超过125,000个标注建筑的DFC2023 Track 2数据集。

### 目的

分析YOLOv11模型在卫星图像中联合建筑提取和离散高度分类的应用能力。

### 方法

使用YOLOv11模型进行建筑实例分割和高度分类，该模型引入了更高效的架构，更好地结合不同尺度的特征，提高目标定位精度。使用精度、召回率、F1分数和mAP等指标评估性能。

### 主要发现

YOLOv11达到60.4% mAP@50和38.3% mAP@50-95的实例分割性能，在五个高度层保持稳健分类准确性。模型在处理遮挡、复杂建筑形状和类别不平衡方面表现出色，特别是对高层结构。在检测准确性和推理速度上优于早期多任务框架。

### 结论

YOLOv11适合实时、大规模城市映射，通过简化的分类高度建模推进语义城市重建，为遥感和地理空间智能发展提供可操作见解。

### 翻译

准确的建筑实例分割和高度分类对城市规划、三维城市建模和基础设施监测至关重要。本文详细分析了YOLOv11（YOLO系列深度学习模型的最新进展），重点关注其在卫星图像中联合建筑提取和离散高度分类的应用。YOLOv11通过引入更高效的架构，更好地结合不同尺度的特征，改进了目标定位精度，并在复杂城市场景中提高了性能，从而建立在早期YOLO模型的优势之上。使用包含12个城市超过125,000个标注建筑的DFC2023 Track 2数据集，我们使用精度、召回率、F1分数和平均精度均值(mAP)等指标评估了YOLOv11的性能。我们的研究结果表明，YOLOv11在实例分割方面表现出色，达到60.4% mAP@50和38.3% mAP@50-95，同时在五个预定义高度层中保持了稳健的分类准确性。该模型在处理遮挡、复杂建筑形状和类别不平衡方面表现出色，特别是对于罕见的高层结构。比较分析证实，YOLOv11在检测准确性和推理速度方面都优于早期的多任务框架，使其非常适合实时、大规模的城市映射。这项研究强调了YOLOv11通过简化的分类高度建模推进语义城市重建的潜力，为遥感和地理空间智能的未来发展提供了可操作的见解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从卫星图像中同时进行建筑实例分割和高度分类的问题。这个问题对城市规划、3D城市建模、基础设施监测和灾害响应等应用至关重要，因为准确的建筑高度信息有助于理解城市结构、评估建筑密度和规划发展。传统方法通常将高度估计作为回归问题处理，但这种方法对噪声数据敏感且结果难以解释，而离散高度分类更符合实际规划需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将传统的高度回归问题重构为分类问题，将建筑物分为5个预定义的高度类别。他们选择YOLOv11作为基础架构，因为它在实时目标检测和实例分割方面表现优异。作者设计了结构化的预处理流程将DSM数据转换为YOLOv11兼容的标注，并使用focal loss和自适应类别权重处理数据中的类别不平衡问题。他们借鉴了YOLO系列模型的发展，特别是YOLOv8和YOLOv10的架构，并参考了DFC2023 Track 2数据集的使用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将建筑物高度估计从回归问题转变为分类问题，使用统一的YOLOv11架构同时处理建筑实例分割和高度分类，通过离散的高度类别提供更符合实际规划需求的可解释输出。整体流程包括：1)数据准备阶段，处理DSM数据计算每个建筑物掩码内的平均高度并映射到5个预定义类别；2)模型架构阶段，使用增强的CSPDarknet作为backbone，改进的PANet++作为neck，解耦头设计处理多任务；3)训练配置阶段，使用COCO预训练权重初始化，在DFC2023数据集上微调并应用focal loss；4)评估测试阶段，使用标准分割指标评估模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)将高度估计从回归转变为分类，提高对噪声数据的鲁棒性；2)使用YOLOv11的统一架构同时处理实例分割和高度分类，避免复杂多分支设计；3)设计针对卫星图像优化的DSM预处理流程；4)使用focal loss和自适应类别权重处理类别不平衡。相比之前工作，本研究避免了LIGHT的复杂交互模块和HGDNet的教师网络依赖，同时解决了Huo等人方法无法区分单个建筑实例的问题，在实例分割任务上达到84.2% mAP@50，在罕见高层建筑物类别上也表现优异。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本研究提出了一种基于YOLOv11的统一框架，通过将建筑物高度估计从回归问题转变为分类问题，实现了从卫星图像中同时进行高精度建筑实例分割和可解释高度分类的创新方法，为城市规划和3D重建提供了更高效、更实用的解决方案。'}


### 论文摘要

Accurate building instance segmentation and height classification are critical for urban planning, 3D city modeling, and infrastructure monitoring. This paper presents a detailed analysis of YOLOv11, the recent advancement in the YOLO series of deep learning models, focusing on its application to joint building extraction and discrete height classification from satellite imagery. YOLOv11 builds on the strengths of earlier YOLO models by introducing a more efficient architecture that better combines features at different scales, improves object localization accuracy, and enhances performance in complex urban scenes. Using the DFC2023 Track 2 dataset -- which includes over 125,000 annotated buildings across 12 cities -- we evaluate YOLOv11's performance using metrics such as precision, recall, F1 score, and mean average precision (mAP). Our findings demonstrate that YOLOv11 achieves strong instance segmentation performance with 60.4\% mAP@50 and 38.3\% mAP@50--95 while maintaining robust classification accuracy across five predefined height tiers. The model excels in handling occlusions, complex building shapes, and class imbalance, particularly for rare high-rise structures. Comparative analysis confirms that YOLOv11 outperforms earlier multitask frameworks in both detection accuracy and inference speed, making it well-suited for real-time, large-scale urban mapping. This research highlights YOLOv11's potential to advance semantic urban reconstruction through streamlined categorical height modeling, offering actionable insights for future developments in remote sensing and geospatial intelligence.

---

## 23. Predicting Household Water Consumption Using Satellite and Street View Images in Two Indian Cities

**论文链接:** [http://arxiv.org/abs/2510.26957v1](http://arxiv.org/abs/2510.26957v1)

**作者:** Qiao Wang, Joseph George

**发布时间:** 2025-10-30

### GPT解析

### 总结

本研究探讨了利用公开可用的图像和地理空间数据预测家庭用水量的可行性，结果显示这种方法接近传统调查方法的准确性。

### 背景

在快速城市化地区，家庭用水监测受到高成本、耗时的列举方法和调查的阻碍。

### 目的

研究是否可以利用公开可用的图像（卫星图像、谷歌街景分割）和简单的地理空间协变量（夜间灯光强度、人口密度）来预测印度Hubballi-Dharwad地区的家庭用水量。

### 方法

比较四种方法：调查特征（基准）、CNN嵌入（卫星、谷歌街景、组合）以及带有辅助数据的谷歌街景语义地图。在有序分类框架下进行评估。

### 主要发现

谷歌街景分割加上遥感协变量实现了0.55的用水量准确率，接近基于调查的模型（0.59准确率）。错误分析显示在家庭用水量分布的极端情况下精度高，但中产阶级之间存在混淆，原因是视觉代理重叠。研究还比较了家庭用水消耗估计与家庭主观收入的估计。

### 结论

开放获取的图像结合最少量的地理空间数据，为在分析学中获取可靠的家庭用水消耗估计提供了一种有希望的替代方案，无需依赖调查。

### 翻译

在快速城市化地区，家庭用水监测受到高成本、耗时的列举方法和调查的阻碍。我们研究了是否可以利用公开可用的图像（卫星图像、谷歌街景分割）和简单的地理空间协变量（夜间灯光强度、人口密度）来预测印度Hubballi-Dharwad地区的家庭用水量。我们比较了四种方法：调查特征（基准）、CNN嵌入（卫星、谷歌街景、组合）以及带有辅助数据的谷歌街景语义地图。在有序分类框架下，谷歌街景分割加上遥感协变量实现了0.55的用水量准确率，接近基于调查的模型（0.59准确率）。错误分析显示在家庭用水量分布的极端情况下精度高，但中产阶级之间存在混淆，原因是视觉代理重叠。我们还比较了家庭用水消耗估计与家庭主观收入的估计。我们的研究表明，开放获取的图像结合最少量的地理空间数据，为在分析学中获取可靠的家庭用水消耗估计提供了一种有希望的替代方案，无需依赖调查。


### 论文摘要

Monitoring household water use in rapidly urbanizing regions is hampered by costly, time-intensive enumeration methods and surveys. We investigate whether publicly available imagery-satellite tiles, Google Street View (GSV) segmentation-and simple geospatial covariates (nightlight intensity, population density) can be utilized to predict household water consumption in Hubballi-Dharwad, India. We compare four approaches: survey features (benchmark), CNN embeddings (satellite, GSV, combined), and GSV semantic maps with auxiliary data. Under an ordinal classification framework, GSV segmentation plus remote-sensing covariates achieves 0.55 accuracy for water use, approaching survey-based models (0.59 accuracy). Error analysis shows high precision at extremes of the household water consumption distribution, but confusion among middle classes is due to overlapping visual proxies. We also compare and contrast our estimates for household water consumption to that of household subjective income. Our findings demonstrate that open-access imagery, coupled with minimal geospatial data, offers a promising alternative to obtaining reliable household water consumption estimates using surveys in urban analytics.

---

## 24. Citizen science dataset on residents' urban heat perception in outdoor public spaces of climate-vulnerable neighborhoods

**论文链接:** [http://arxiv.org/abs/2510.25645v1](http://arxiv.org/abs/2510.25645v1)

**作者:** Ferran Larroya, Isabelle Bonhoure, Femke Min, Josep Perelló

**发布时间:** 2025-10-29

### GPT解析

### 总结

研究巴塞罗那都市区五个社区的城市热和热感知，创建了一个结合客观环境测量与主观热感知的多维度数据集

### 背景

巴塞罗那都市区存在城市热岛效应和气候脆弱性问题，需要研究城市热环境与居民感知的关系

### 目的

创建一个数据集来研究城市热岛效应、城市健康和气候韧性，支持热、热不平等和气候脆弱性经验维度的研究

### 方法

与14个非学术合作伙伴合作，组织439名居民作为共同研究者，确定210个公共户外场所，进行48次热步行活动，使用便携式传感器收集环境数据，并通过标准化调查收集主观热感知数据

### 主要发现

收集了296,286个微气候数据点和5,169个自我报告条目，整合了客观环境测量与主观热感知，实现了城市结构中热体验的点对点分析

### 结论

该数据集为城市规划、公共卫生和气候适应提供了多维度资源，可支持基于证据的决策制定

### 翻译

我们提出了一个数据集，用于研究巴塞罗那都市区五个社区的城市热和热感知。我们与14个非学术合作伙伴组织合作，开展了一系列公民科学活动，涉及439名居民作为共同研究者，他们参与了研究过程的所有阶段。参与者是居住在被归类为高度或非常高度气候脆弱地区的居民，确定了与他们日常生活相关的210个公共户外场所。这些地点随后使用一系列与城市热岛效应、城市健康和气候韧性相关的空间和环境指标进行了特征描述。在48次热步行过程中，参与者携带便携式低成本传感器，连续记录空气温度、相对湿度和地理位置，产生了296,286个处理过的微气候数据点。在预定义的站点，参与者完成了标准化调查，报告了他们的热感觉投票和热舒适度投票，产生了5,169个自我报告条目。还收集了社会经济人口数据，以进一步参与者的反应提供背景。产生的数据集集成了客观环境测量与主观热感知，实现了城市结构中热体验的点对点分析。它提供了一个新颖的多维度资源，支持热、热不平等和气候脆弱性经验维度的研究，并旨在为城市规划、公共卫生和气候适应提供基于证据的决策依据。


### 论文摘要

We present a dataset generated to investigate urban heat and thermal perception across five neighborhoods in the Barcelona metropolitan area. In collaboration with 14 non-academic partner organizations, we conducted a series of citizen science campaigns involving 439 residents as co-researchers engaged throughout all stages of the research process. Participants, residents of areas classified as highly or very highly climate-vulnerable, identified 210 public outdoor sites relevant to their daily lives. These locations were subsequently characterized using a range of spatial and environmental indicators pertinent to urban heat island effects, urban health, and climate resilience. Over the course of 48 thermal walks, participants carried portable, low-cost sensors that continuously recorded air temperature, relative humidity, and geolocation, resulting in 296,286 processed microclimatic data points. At pre-defined sites, individuals completed standardized surveys to report their Thermal Sensation Votes and Thermal Comfort Votes, yielding 5,169 self-reported entries. Sociodemographic data were also collected to further contextualize participants' responses. The resulting dataset integrates objective environmental measurements with subjective perceptions of heat, enabling point-by-point analysis of thermal experience within the urban fabric. It offers a novel, multi-dimensional resource to support research on heat, thermal inequality, and the experiential dimensions of climate vulnerability, and is intended to inform evidence-based decision-making in urban planning, public health, and climate adaptation.

---

## 25. Quantum-Resilient Threat Modelling for Secure RIS-Assisted ISAC in 6G UAV Corridors

**论文链接:** [http://arxiv.org/abs/2510.25411v1](http://arxiv.org/abs/2510.25411v1)

**作者:** Sana Hafeez, Ghulam E Mustafa Abro, Hifza Mustafa

**发布时间:** 2025-10-29

**备注:** 6 Pages, 5figures

### GPT解析

### 总结

本文提出了一种量子弹性威胁建模(QRTM)框架，用于解决6G网络中无人机走廊RIS辅助ISAC系统的安全问题，以应对量子计算带来的新型安全威胁。

### 背景

6G网络中无人机走廊快速部署需要安全、智能驱动的集成感知与通信系统。RIS技术虽能提高频谱效率、定位精度和态势感知能力，但也引入了新的安全漏洞。量子计算的兴起增加了'现时解密后攻击'和量子增强欺骗的风险。

### 目的

开发针对RIS辅助ISAC在无人机走廊中的量子弹性威胁建模框架，解决量子计算带来的安全挑战，确保系统安全性。

### 方法

QRTM框架整合经典、量子就绪和量子辅助对手，采用后量子密码学原语(ML-KEM和Falcon)进行防御，并将其嵌入RIS控制信号和无人机协调中。引入RIS编码场景水印通过广义似然比检验验证，安全ISAC效用联合优化保密率、欺骗检测和吞吐量，调度器计算复杂度为二次方级别。

### 主要发现

基于3GPP第19版中频段城市峡谷模型(7-15 GHz)的蒙特卡洛评估显示：在虚警率为千分之一时，欺骗检测概率接近百分之九十九；对量子能力对手的保密率保留超过百分之九十；与基线相比，信号干扰利用率提高约百分之二十五。

### 结论

研究结果为智能城市和非陆地网络中的无人机走廊提供可靠、量子弹性的ISAC系统标准兼容路径。

### 翻译

第六代网络中无人机走廊的快速部署需要安全、智能驱动的集成感知与通信。可重构智能表面提高频谱效率、定位精度和态势感知能力，同时引入新的脆弱性。量子计算的兴起增加了'现时解密后攻击'策略和量子增强欺骗的风险。我们提出了一种针对RIS辅助ISAC在无人机走廊中的量子弹性威胁建模框架，以应对这些挑战。QRTM整合经典、量子就绪和量子辅助对手，使用后量子密码学原语进行防御：ML-KEM用于密钥建立，Falcon用于认证，两者都嵌入RIS控制信号和无人机协调中。为加强安全感知，该框架引入RIS编码场景水印，通过广义似然比检验验证，其检测概率通过Marcum Q函数表征。此外，安全ISAC效用联合优化保密率、欺骗检测和吞吐量，在RIS约束下，由计算复杂度为O(n^2)的调度器实现。使用3GPP第19版中频段城市峡谷模型(7-15 GHz)进行的蒙特卡洛评估表明，在虚警率为1e-3时，欺骗检测概率接近0.99，对量子能力对手的保密率保留超过90%，与基线相比信号干扰利用率提高约25%。这些结果表明，为智能城市和非陆地网络中的无人机走廊提供可靠、量子弹性的ISAC具有标准兼容路径。


### 论文摘要

The rapid deployment of unmanned aerial vehicle (UAV) corridors in sixth-generation (6G) networks requires safe, intelligence-driven integrated sensing and communications (ISAC). Reconfigurable intelligent surfaces (RIS) enhance spectrum efficiency, localisation accuracy, and situational awareness, while introducing new vulnerabilities. The rise of quantum computing increases the risks associated with harvest-now-decrypt-later strategies and quantum-enhanced spoofing. We propose a Quantum-Resilient Threat Modelling (QRTM) framework for RIS-assisted ISAC in UAV corridors to address these challenges. QRTM integrates classical, quantum-ready, and quantum-aided adversaries, countered using post-quantum cryptographic (PQC) primitives: ML-KEM for key establishment and Falcon for authentication, both embedded within RIS control signalling and UAV coordination. To strengthen security sensing, the framework introduces RIS-coded scene watermarking validated through a generalised likelihood ratio test (GLRT), with its detection probability characterised by the Marcum Q function. Furthermore, a Secure ISAC Utility (SIU) jointly optimises secrecy rate, spoofing detection, and throughput under RIS constraints, enabled by a scheduler with computational complexity of O(n^2). Monte Carlo evaluations using 3GPP Release 19 mid-band urban-canyon models (7-15 GHz) demonstrate a spoof-detection probability approaching 0.99 at a false-alarm rate of 1e-3, secrecy-rate retention exceeding 90 percent against quantum-capable adversaries, and signal-interference utilisation improvements of about 25 percent compared with baselines. These results show a standards-compliant path towards reliable, quantum-resilient ISAC for UAV corridors in smart cities and non-terrestrial networks.

---

## 26. HAPS-ISAC for 6G: Architecture, Design Trade-offs, and a Practical Roadmap

**论文链接:** [http://arxiv.org/abs/2510.23147v1](http://arxiv.org/abs/2510.23147v1)

**作者:** Parisa Kanani, Mohammad Javad Omidi, Mahmoud Modarres-Hashemi, Halim Yanikomeroglu

**发布时间:** 2025-10-27

### GPT解析

### 总结

提出了一种基于高空平台站(HAPS)的集成通信和感知(ISAC)架构，结合无人机形成智能3D网络，提高网络性能、感知精度和服务公平性。

### 背景

为了满足下一代6G网络的宏伟目标，包括超高数据速率和无处不在的覆盖。

### 目的

开发一种新型的高空平台站架构，实现通信和感知功能一体化。

### 方法

设计在平流层运行的高空平台站，作为通信中心和环境传感器，结合协作无人机形成可扩展的智能3D网络。

### 主要发现

该架构显著提高了网络性能，改善了感知精度，确保了用户间更公平的服务分布，优于传统仅使用无人机的方案。

### 结论

该技术有潜力应用于智能城市和其他大规模环境，并提供了相应的部署路线图。

### 翻译

为了满足下一代6G网络的宏伟目标，包括超高数据速率和无处不在的覆盖，我们提出了一种新型的高空平台站(HAPS)架构，用于集成通信和感知(ISAC)。HAPS在平流层运行，既作为强大的通信中心，又作为先进的环境传感器。结合一组协作的无人机(UAV)，这种双用途系统形成一个可扩展的智能3D网络。模拟结果表明，这种方法显著提高了网络性能，改善了感知精度，并确保了用户之间更公平的服务分布，优于传统的仅使用无人机的基线。我们最后概述了这项技术在智能城市和其他大规模环境中的潜在应用和部署路线图。


### 论文摘要

To meet the ambitious goals of next-generation 6G networks, including ultra-high data rates and ubiquitous coverage, we propose a novel high-altitude platform station (HAPS)-based integrated sensing and communication (ISAC) architecture. Operating in the stratosphere, the HAPS functions as both a powerful communication hub and an advanced environmental sensor. Combined with a fleet of cooperative uncrewed aerial vehicles (UAVs), this dual-purpose system forms a scalable and intelligent 3D network. Simulation results indicate that this approach significantly boosts network performance, improves sensing accuracy, and ensures a fairer service distribution across users, outperforming conventional UAV-only baselines. We conclude by outlining the prospective applications and a deployment roadmap for this technology for smart cities and other large-scale environments.

---

## 27. Planning Oriented Integrated Sensing and Communication

**论文链接:** [http://arxiv.org/abs/2510.23021v1](http://arxiv.org/abs/2510.23021v1)

**作者:** Xibin Jin, Guoliang Li, Shuai Wang, Fan Liu, Miaowen Wen, Huseyin Arslan, Derrick Wing Kwan Ng, Chengzhong Xu

**发布时间:** 2025-10-27

### GPT解析

### 总结

本文提出了一种面向规划的集成感知与通信(PISAC)框架，通过减少规划瓶颈障碍物的感知不确定性，扩展安全可导航路径，从而连接物理层优化和运动层规划，提高自动驾驶车辆的安全性和效率。

### 背景

集成感知与通信(ISAC)技术使连接自动驾驶车辆能够同时实现定位、环境感知和数据交换。然而，现有的ISAC设计通常优先考虑感知精度和通信吞吐量，对所有目标一视同仁，忽略了关键障碍物对运动效率的影响。

### 目的

克服现有ISAC设计的局限性，提出一种能够减少规划瓶颈障碍物感知不确定性并扩展安全可导航路径的PISAC框架，填补物理层优化与运动层规划之间的差距。

### 方法

推导出一个封闭形式的安全边界，明确将ISAC发射功率与感知不确定性联系起来，基于Cramér-Rao Bound和占用膨胀原理。构建双层功率分配和运动规划(PAMP)问题，内层优化ISAC波束功率分布，外层在不确定性感知的安全约束下计算无碰撞轨迹。

### 主要发现

在高保真度城市驾驶环境中的全面模拟显示，PISAC比现有的基于ISAC和面向通信的基准实现了高达40%更高的成功率和超过5%的更短遍历时间。

### 结论

PISAC框架通过有针对性地减少规划瓶颈障碍物的感知不确定性，成功连接了物理层优化和运动层规划，显著提高了自动驾驶车辆的安全性和效率。

### 翻译

集成感知与通信(ISAC)使连接自动驾驶车辆能够同时实现定位、环境感知和数据交换。然而，大多数现有的ISAC设计优先考虑感知精度和通信吞吐量，对所有目标一视同仁，忽略了关键障碍物对运动效率的影响。为了克服这一限制，我们提出了一种面向规划的ISAC(PISAC)框架，减少规划瓶颈障碍物的感知不确定性，扩展自车的安全可导航路径，从而填补物理层优化与运动层规划之间的差距。PISAC的核心在于推导出一个封闭形式的安全边界，明确将ISAC发射功率与感知不确定性联系起来，基于Cramér-Rao Bound和占用膨胀原理。利用此模型，我们构建了一个双层功率分配和运动规划(PAMP)问题，其中内层优化ISAC波束功率分布，外层在不确定性感知的安全约束下计算无碰撞轨迹。在高保真度城市驾驶环境中的全面模拟表明，PISAC比现有的基于ISAC和面向通信的基准实现了高达40%更高的成功率和超过5%的更短遍历时间，验证了其在提高安全性和效率方面的有效性。


### 论文摘要

Integrated sensing and communication (ISAC) enables simultaneous localization, environment perception, and data exchange for connected autonomous vehicles. However, most existing ISAC designs prioritize sensing accuracy and communication throughput, treating all targets uniformly and overlooking the impact of critical obstacles on motion efficiency. To overcome this limitation, we propose a planning-oriented ISAC (PISAC) framework that reduces the sensing uncertainty of planning-bottleneck obstacles and expands the safe navigable path for the ego-vehicle, thereby bridging the gap between physical-layer optimization and motion-level planning. The core of PISAC lies in deriving a closed-form safety bound that explicitly links ISAC transmit power to sensing uncertainty, based on the Cramér-Rao Bound and occupancy inflation principles. Using this model, we formulate a bilevel power allocation and motion planning (PAMP) problem, where the inner layer optimizes the ISAC beam power distribution and the outer layer computes a collision-free trajectory under uncertainty-aware safety constraints. Comprehensive simulations in high-fidelity urban driving environments demonstrate that PISAC achieves up to 40% higher success rates and over 5% shorter traversal times than existing ISAC-based and communication-oriented benchmarks, validating its effectiveness in enhancing both safety and efficiency.

---

## 28. Designing Knowledge Tools: How Students Transition from Using to Creating Generative AI in STEAM classroom

**论文链接:** [http://arxiv.org/abs/2510.19405v1](http://arxiv.org/abs/2510.19405v1)

**作者:** Qian Huang, Nachamma Sockalingam, Thijs Willems, King Wang Poon

**发布时间:** 2025-10-22

**备注:** to be published in IEEE TALE 2025

### GPT解析

### 总结

这项研究探讨了城市规划项目的研究生如何从生成式AI的被动使用者转变为基于GPT的自定义知识工具的主动创造者。

### 背景

研究设置在一个为期两个学期的课程中，学生首先使用教师创建的GPT支持定性研究任务，然后重新设计这些工具创建自己的自定义应用，包括'访谈伙伴GPT'。

### 目的

基于自我决定理论，研究调查了设计AI工具如何影响学生的学习体验、身份形成和知识参与，以及这种转变如何满足学生的自主性、胜任力和关联性需求。

### 方法

研究采用定性主题分析方法，分析学生的幻灯片演示和焦点小组访谈数据。

### 主要发现

学生在角色和心态上发生了显著转变：选择工具功能、设计和目的时感到更加自主；通过获取AI相关技能(如提示工程和迭代测试)感到更加胜任；通过团队协作和共同目标感与同伴建立更强的联系。

### 结论

当学习者被邀请共同设计他们使用的技术时，学生的能动性可以被有力地激活。从AI工具使用者到AI工具设计者的转变重新配置了学生与技术和知识的关系，使他们从消费者转变为教育环境中的共同创造者。

### 翻译

这项研究探讨了城市规划项目的研究生如何从生成式人工智能的被动使用者转变为基于GPT的自定义知识工具的主动创造者。基于强调自主性、胜任力和关联性作为内在动机基础的自我决定理论，该研究调查了设计人工智能工具如何影响学生的学习体验、身份形成和知识参与。研究设置在一个为期两个学期的课程中，学生首先使用教师创建的GPT支持定性研究任务，然后重新设计这些工具创建自己的自定义应用，包括'访谈伙伴GPT'。通过对学生的幻灯片演示和焦点小组访谈进行定性主题分析，研究结果突显了学生在角色和心态上的显著转变。学生报告说，他们在选择工具功能、设计和目的时感到更加自主；通过获取人工智能相关技能(如提示工程和迭代测试)感到更加胜任；通过团队协作和共同目标感与同伴建立更强的联系。该研究增加了越来越多的证据，表明当学习者被邀请共同设计他们使用的技术时，学生的能动性可以被有力地激活。从人工智能工具使用者到人工智能工具设计者的转变重新配置了学生与技术和知识的关系，使他们从消费者转变为在不断发展的教育环境中的共同创造者。


### 论文摘要

This study explores how graduate students in an urban planning program transitioned from passive users of generative AI to active creators of custom GPT-based knowledge tools. Drawing on Self-Determination Theory (SDT), which emphasizes the psychological needs of autonomy, competence, and relatedness as foundations for intrinsic motivation, the research investigates how the act of designing AI tools influences students' learning experiences, identity formation, and engagement with knowledge. The study is situated within a two-term curriculum, where students first used instructor-created GPTs to support qualitative research tasks and later redesigned these tools to create their own custom applications, including the Interview Companion GPT. Using qualitative thematic analysis of student slide presentations and focus group interviews, the findings highlight a marked transformation in students' roles and mindsets. Students reported feeling more autonomous as they chose the functionality, design, and purpose of their tools, more competent through the acquisition of AI-related skills such as prompt engineering and iterative testing, and more connected to peers through team collaboration and a shared sense of purpose. The study contributes to a growing body of evidence that student agency can be powerfully activated when learners are invited to co-design the very technologies they use. The shift from AI tool users to AI tool designers reconfigures students' relationships with technology and knowledge, transforming them from consumers into co-creators in an evolving educational landscape.

---

## 29. New ASKAP radio-continuum surveys of the Small Magellanic Cloud

**论文链接:** [http://arxiv.org/abs/2511.09954v1](http://arxiv.org/abs/2511.09954v1)

**作者:** O. K. Khattab, M. D. Filipovic', Z. J. Smeaton, R. Z. E. Alsaberi, E. J. Crawford, D. Leahy, S. Dai, N. Rajabpour

**发布时间:** 2025-11-13

**备注:** 22 pages, 12 Figures and 2 Tables

### GPT解析

### 总结

本文介绍了来自ASKAP POSSUM survey对小麦哲伦云方向的两个新的射电连续谱图像，包含两个频率的射电源目录，并与MeerKAT目录进行了交叉匹配，提高了对该区域的理解。

### 背景

ASKAP POSSUM survey正在进行中，小麦哲伦云是重要的天文观测目标，已有MeerKAT目录的先前数据。

### 目的

生成小麦哲伦云区域的射电源目录，提高对该区域的理解，展示现代射电望远镜如ASKAP研究不同银河系源种群的能力。

### 方法

使用ASKAP POSSUM survey获取944 MHz和1367 MHz两个频率的射电连续谱图像，使用Aegean软件包生成点源目录，与先前发布的MeerKAT目录交叉匹配，使用2角秒的匹配半径识别共同点源，并估计匹配射电点源的光谱指数。

### 主要发现

在944 MHz频率检测到36,571个射电连续谱源，波束大小约为14.5×12.2角秒；在1367 MHz频率检测到15,227个源，波束大小约为8.7×8.2角秒；通过交叉匹配，在944 MHz和1367 MHz分别识别出21,442和12,654个共同点源。

### 结论

这些新目录提高了对小麦哲伦云的理解，并展示了ASKAP等现代射电望远镜研究不同银河系源种群的能力。

### 翻译

我们展示了来自ASKAP POSSUM survey对小麦哲伦云方向的两个新的射电连续谱图像。这两个新的源列表包含在944 MHz检测到的36,571个射电连续谱源和在1367 MHz检测到的15,227个源，波束大小分别约为14.5×12.2角秒和8.7×8.2角秒。我们使用Aegean软件包生成了这些点源目录，与先前发布的MeerKAT目录一起，我们估计了完整匹配射电点源集合的光谱指数。通过将我们的ASKAP目录与MeerKAT数据进行交叉匹配，我们使用2角秒的匹配半径分别在944 MHz和1367 MHz识别出21,442和12,654个共同点源。这些新目录提高了我们对小麦哲伦云的理解，并展示了ASKAP等当前射电望远镜研究不同银河系源种群的能力。


### 论文摘要

We present two new radio continuum images from the ASKAP POSSUM survey in the direction of the Small Magellanic Cloud. The two new source lists contain 36,571 radio continuum sources detected at 944 MHz and 15,227 sources at 1367 MHz, with beam sizes of approximately 14.5 by 12.2 arcsec and 8.7 by 8.2 arcsec, respectively. We used the Aegean software package to generate these point source catalogues, and together with the previously published MeerKAT catalogue, we estimated spectral indices for the full set of matched radio point sources. By cross-matching our ASKAP catalogues with the MeerKAT data, we identified 21,442 and 12,654 common point sources at 944 MHz and 1367 MHz, respectively, using a 2 arcsec matching radius. These new catalogues improve our understanding of the Small Magellanic Cloud and demonstrate the capability of current radio telescopes such as ASKAP to investigate diverse galactic source populations

---

## 30. IPCD: Intrinsic Point-Cloud Decomposition

**论文链接:** [http://arxiv.org/abs/2511.09866v1](http://arxiv.org/abs/2511.09866v1)

**作者:** Shogo Sato, Takuhiro Kaneko, Shoichiro Takeda, Tomoyasu Shimada, Kazuhiko Murasaki, Taiga Yoshida, Ryuichi Tanida, Akisato Kimura

**发布时间:** 2025-11-13

**备注:** Accepted in WACV2026

### GPT解析

### 总结

本研究提出了一种名为Intrinsic Point-Cloud Decomposition (IPCD)的新方法，用于将彩色点云直接分解为反照率和阴影，解决了点云非网格结构和全局光照考虑不足的挑战，并在增强现实和机器人等领域有广泛应用。

### 背景

点云广泛应用于增强现实(AR)和机器人等领域，在这些领域中，重新打光和纹理编辑对真实可视化至关重要。实现这些任务需要准确分离反照率(albedo)和阴影(shade)。

### 目的

解决点云分解面临的两个关键挑战：(1)点云的非网格结构使传统图像分解模型无效；(2)现有点云模型未明确考虑全局光照方向，导致阴影不准确。

### 方法

提出Intrinsic Point-Cloud Decomposition (IPCD)框架，包括：(1)IPCD-Net，通过点特征聚合扩展基于图像的模型以处理非网格数据；(2)基于投影的亮度分布(PLD)，通过分层特征细化和多视图投影捕获全局光照线索。

### 主要发现

实验结果表明IPCD-Net减少了反照率中的投射阴影，增强了阴影中的颜色准确性。该方法在纹理编辑、不同光照条件下的重新打光和点云配准中展示了有效应用。

### 结论

IPCD成功解决了点云分解的两个主要挑战，并在实际应用中验证了其有效性和适用性，为增强现实和机器人等领域提供了新的解决方案。

### 翻译

点云广泛应用于增强现实(AR)和机器人等多个领域，在这些领域中，重新打光和纹理编辑对真实可视化至关重要。实现这些任务需要准确分离反照率和阴影。然而，在点云上进行这种分离面临两个关键挑战：(1)点云的非网格结构使传统的基于图像的分解模型无效；(2)为其他任务设计的点云模型没有明确考虑全局光照方向，导致阴影不准确。在本文中，我们引入了内在点云分解(IPCD)，它将图像分解扩展到直接将彩色点云分解为反照率和阴影。为解决挑战(1)，我们提出了IPCD-Net，通过点特征聚合扩展基于图像的模型以处理非网格数据。为解决挑战(2)，我们引入了基于投影的亮度分布(PLD)，通过分层特征细化，利用多视图投影捕获全局光照线索。为全面评估，我们创建了一个合成户外场景数据集。实验结果表明，IPCD-Net减少了反照率中的投射阴影，并增强了阴影中的颜色准确性。此外，我们展示了其在纹理编辑、不同光照条件下的重新打光和点云配准中的应用。最后，我们验证了IPCD-Net的实际适用性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是将彩色点云分解为反照率(albedo)和阴影(shade)的问题。这个问题在现实中非常重要，因为增强现实、机器人等领域需要重新打光和纹理编辑来实现真实感可视化，但这些应用需要准确分离物体的固有颜色(反照率)和光照效果(阴影)。传统图像分解方法无法直接应用于点云，因为点云的非网格结构使基于图像的模型失效，而其他点云模型又没有考虑全局光照方向，导致阴影估计不准确。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先借鉴了二维图像领域的内在图像分解(IID)方法，特别是PoInt-Net框架。针对点云的非网格结构，他们扩展了PoInt-Net，创建IPCD-NetBase，使其能够在点云空间直接进行分解。为了解决全局光照方向缺失的问题，他们创新地设计了基于投影的亮度分布(PLD)，通过多视角投影隐式捕获光照信息。此外，他们还借鉴了点云处理领域的技术，如Point Transformer v2，并引入了分层特征细化来有效集成PLD特征。作者通过共享编码器和同时训练策略进一步改进了模型性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将二维图像分解扩展到三维点云领域，直接在点云空间进行反照率和阴影的分离，而不需要将点云渲染为图像。整体实现流程包括：1) 使用共享编码器处理输入点云，生成预反照率和预阴影表示；2) 计算PLD特征，通过将点云从324个不同方向投影到半球形表面并计算平均亮度；3) 使用SphereNet处理球面PLD数据；4) 通过分层特征细化将PLD特征与预反照率和预阴影表示连接；5) 最后通过反照率和阴影头部估计最终结果。模型采用同时训练策略，在合成的户外场景数据集上进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次将内在图像分解扩展到点云领域，提出IPCD任务；2) 设计IPCD-NetBase，在点云空间直接进行分解，避免渲染和投影过程；3) 引入PLD通过多视角投影隐式捕获全局光照线索；4) 采用分层特征细化逐步估计反照率和阴影；5) 创建合成户外场景数据集专门评估IPCD性能。相比之前的工作，IPCD直接在点云空间操作，避免了渲染方法因遮挡和几何信息缺失导致的质量问题；同时通过PLD隐式捕获光照信息，而不需要显式提供光照方向，使其更适用于真实场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了内在点云分解(IPCD)方法，通过点级特征聚合和基于投影的亮度分布(PLD)，实现了直接从彩色点云中准确分离反照率和阴影，为点云的纹理编辑、重新打光和配准等应用提供了基础。'}


### 论文摘要

Point clouds are widely used in various fields, including augmented reality (AR) and robotics, where relighting and texture editing are crucial for realistic visualization. Achieving these tasks requires accurately separating albedo from shade. However, performing this separation on point clouds presents two key challenges: (1) the non-grid structure of point clouds makes conventional image-based decomposition models ineffective, and (2) point-cloud models designed for other tasks do not explicitly consider global-light direction, resulting in inaccurate shade. In this paper, we introduce \textbf{Intrinsic Point-Cloud Decomposition (IPCD)}, which extends image decomposition to the direct decomposition of colored point clouds into albedo and shade. To overcome challenge (1), we propose \textbf{IPCD-Net} that extends image-based model with point-wise feature aggregation for non-grid data processing. For challenge (2), we introduce \textbf{Projection-based Luminance Distribution (PLD)} with a hierarchical feature refinement, capturing global-light ques via multi-view projection. For comprehensive evaluation, we create a synthetic outdoor-scene dataset. Experimental results demonstrate that IPCD-Net reduces cast shadows in albedo and enhances color accuracy in shade. Furthermore, we showcase its applications in texture editing, relighting, and point-cloud registration under varying illumination. Finally, we verify the real-world applicability of IPCD-Net.

---

## 31. PALMS+: Modular Image-Based Floor Plan Localization Leveraging Depth Foundation Model

**论文链接:** [http://arxiv.org/abs/2511.09724v1](http://arxiv.org/abs/2511.09724v1)

**作者:** Yunqian Cheng, Benjamin Princen, Roberto Manduchi

**发布时间:** 2025-11-12

**备注:** Accepted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026, Application Track. Main paper: 8 pages, 5 figures. Supplementary material included

### GPT解析

### 总结

本文提出了PALMS+系统，一种模块化、基于图像的室内定位方法，解决了传统PALMS方法在智能手机LiDAR短距离和室内布局模糊性方面的局限性，实现了无需基础设施的高精度室内定位。

### 背景

在GPS受限的室内环境中进行定位对于应急响应和辅助导航等应用至关重要。基于视觉的方法如PALMS可以在不需要基础设施的情况下，仅使用楼层平面图和静态扫描来实现定位，但这些方法受限于智能手机LiDAR的短距离和室内布局的模糊性。

### 目的

提出PALMS+系统，解决传统PALMS方法面临的挑战，创建一个模块化、基于图像的系统，能够在不需要基础设施的情况下实现更精确的室内定位。

### 方法

PALMS+通过使用基础单目深度估计模型（Depth Pro）对带有姿态的RGB图像进行重建，生成尺度对齐的3D点云，然后通过与楼层平面图的卷积进行几何布局匹配，输出位置和方向的后验概率，可用于直接或顺序定位。

### 主要发现

在Structured3D和一个包含80次观测的自定义校园数据集上评估，PALMS+在静态定位精度上优于PALMS和F3Loc，且不需要任何训练。当与粒子滤波器集成用于33条真实世界轨迹的顺序定位时，PALMS+实现了比其他方法更低的定位误差，证明了其在无相机跟踪方面的鲁棒性。

### 结论

PALMS+是一种有效的方法，可以在GPS受限的室内环境中实现高精度定位，该系统不需要基础设施，具有实际应用潜力，代码和数据可在提供的GitHub链接获取。

### 翻译

在GPS受限的室内环境中的定位对于应急响应和辅助导航等应用至关重要。基于视觉的方法如PALMS使得无需基础设施的定位成为可能，仅需使用楼层平面图和静态扫描，但受限于智能手机LiDAR的短距离和室内布局的模糊性。我们提出了PALMS+，一个模块化、基于图像的系统，通过使用基础单目深度估计模型（Depth Pro）对带有姿态的RGB图像进行重建，生成尺度对齐的3D点云，然后通过与楼层平面图的卷积进行几何布局匹配，解决了这些挑战。PALMS+输出位置和方向的后验概率，可用于直接或顺序定位。在Structured3D和一个包含80次观测、涵盖四个大型校园建筑的自定义校园数据集上评估，PALMS+在静态定位精度上优于PALMS和F3Loc，且不需要任何训练。此外，当与粒子滤波器集成，用于33条真实世界轨迹的顺序定位时，PALMS+实现了比其他方法更低的定位误差，证明了其在无相机跟踪方面的鲁棒性及其对无基础设施应用的潜力。代码和数据可在https://github.com/Head-inthe-Cloud/PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones获取。


### 论文摘要

Indoor localization in GPS-denied environments is crucial for applications like emergency response and assistive navigation. Vision-based methods such as PALMS enable infrastructure-free localization using only a floor plan and a stationary scan, but are limited by the short range of smartphone LiDAR and ambiguity in indoor layouts. We propose PALMS$+$, a modular, image-based system that addresses these challenges by reconstructing scale-aligned 3D point clouds from posed RGB images using a foundation monocular depth estimation model (Depth Pro), followed by geometric layout matching via convolution with the floor plan. PALMS$+$ outputs a posterior over the location and orientation, usable for direct or sequential localization. Evaluated on the Structured3D and a custom campus dataset consisting of 80 observations across four large campus buildings, PALMS$+$ outperforms PALMS and F3Loc in stationary localization accuracy -- without requiring any training. Furthermore, when integrated with a particle filter for sequential localization on 33 real-world trajectories, PALMS$+$ achieved lower localization errors compared to other methods, demonstrating robustness for camera-free tracking and its potential for infrastructure-free applications. Code and data are available at https://github.com/Head-inthe-Cloud/PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones

---

## 32. HOTFLoc++: End-to-End Hierarchical LiDAR Place Recognition, Re-Ranking, and 6-DoF Metric Localisation in Forests

**论文链接:** [http://arxiv.org/abs/2511.09170v1](http://arxiv.org/abs/2511.09170v1)

**作者:** Ethan Griffiths, Maryam Haghighat, Simon Denman, Clinton Fookes, Milad Ramezani

**发布时间:** 2025-11-12

**备注:** 9 pages, 2 figures. Submitted to RA-L

### GPT解析

### 总结

本文提出了HOTFLoc++，一个用于森林环境中激光雷达地点识别、重排序和6自由度度量定位的端到端框架。

### 背景

在森林等复杂环境中，激光雷达定位面临杂乱、自相似性和视角变化等挑战，包括地面到地面和地面到空中场景。

### 目的

开发一个能够有效处理森林环境中激光雷达定位问题的框架，提高地点识别的准确性和定位精度。

### 方法

使用基于八叉树的Transformer提取多粒度层次化局部描述符增强鲁棒性；提出可学习的多尺度几何验证模块减少重排序失败；采用粗到细的配准方法，相比RANSAC在密集点云上的运行时间提高了两个数量级。

### 主要发现

在CS-Wild-Places数据集上达到平均Recall@1为90.7%，比基线提高29.6个百分点；在Wild-Places和MulRan数据集上分别达到91.7%和96.0%的平均Recall@1；97.2%的6自由度配准尝试实现了2米和5度以下的误差；多尺度重排序模块将定位误差平均减少约2倍。

### 结论

HOTFLoc++框架在森林环境中的激光雷达地点识别和定位任务中表现出色，在保持高精度的同时显著提高了计算效率。

### 翻译

本文提出了HOTFLoc++，这是一个用于森林环境中的激光雷达地点识别、重排序和6自由度度量定位的端到端框架。利用基于八叉树的Transformer，我们的方法在多种粒度下提取层次化局部描述符，以提高对杂乱、自相似性和视角变化的鲁棒性，包括森林和城市环境中的地面到地面和地面到空中等具有挑战性的场景。我们提出了一个可学习的多尺度几何验证模块，以减少在单尺度对应关系退化情况下的重排序失败。我们的粗到细配准方法实现了与基线相当或更低的定位误差，在密集点云上的运行时间比RANSAC提高了两个数量级。公共数据集上的实验结果表明，我们的方法优于最先进的方法，在CS-Wild-Places上达到平均Recall@1为90.7%，比基线提高了29.6个百分点，同时在单源基准测试中保持高性能，在Wild-Places和MulRan上分别达到91.7%和96.0%的平均Recall@1。我们的方法在97.2%的6自由度配准尝试中实现了2米和5度以下的误差，我们的多尺度重排序模块将定位误差平均减少了约2倍。代码将在接受后提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在森林等自然环境中进行激光雷达地点识别、重新排序和6自由度定位的挑战。这个问题很重要，因为大多数现有方法针对结构化城市环境设计，而森林环境缺乏独特地标、具有强自相似性和季节变化，导致现有方法失效。此外，在GPS受限环境中，长期移动机器人的自主导航依赖于可靠的地点识别和定位技术，这对森林探索、野生动物监测和林业管理等实际应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：关键点检测在杂乱环境中难以产生可重复点，单尺度几何验证无法处理复杂场景的层次结构。作者借鉴了HOTFormerLoc的八叉树变换器提取层次特征，参考了SpectralGV的几何验证思想，并采用了GeoTransformer的几何变换器设计。在此基础上，作者创新性地设计了多尺度几何验证模块和无关键点的粗到精注册方法，并通过端到端联合训练三个相关任务，利用它们之间的互补约束提高整体性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用分层特征表示处理森林环境的复杂性，通过多尺度验证提高鲁棒性，采用无关键点方法避免关键点提取困难，并联合训练多个相关任务。整体流程包括：1)使用八叉树变换器提取多尺度特征和全局描述符进行地点识别；2)通过MSGV模块分析多尺度几何一致性重新排序候选位置；3)利用粗到精注册方法，从粗对应关系扩展到补丁级对应，通过最优传输和局部到全局注册估计6-DoF变换；4)采用两阶段端到端训练，同时优化地点识别、重新排序和重定位任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)HOTFLoc++统一框架，端到端解决地点识别、重新排序和6-DoF定位；2)多尺度几何验证(MSGV)模块，提高对单尺度特征退化的鲁棒性；3)无关键点的粗到精注册方法，避免在杂乱环境中提取可重复关键点的困难，且比RANSAC快两个数量级；4)联合训练方法，利用任务间的互补约束提升性能。相比之前的工作，该方法在森林环境中表现更好，运行速度更快，在跨源设置中更鲁棒，且通过联合训练实现了更好的整体性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HOTFLoc++通过引入多尺度几何验证和无关键点的粗到精注册方法，实现了在森林等复杂自然环境中高效且鲁棒的端到端激光雷达地点识别、重新排序和6-DoF定位。'}


### 论文摘要

This article presents HOTFLoc++, an end-to-end framework for LiDAR place recognition, re-ranking, and 6-DoF metric localisation in forests. Leveraging an octree-based transformer, our approach extracts hierarchical local descriptors at multiple granularities to increase robustness to clutter, self-similarity, and viewpoint changes in challenging scenarios, including ground-to-ground and ground-to-aerial in forest and urban environments. We propose a learnable multi-scale geometric verification module to reduce re-ranking failures in the presence of degraded single-scale correspondences. Our coarse-to-fine registration approach achieves comparable or lower localisation errors to baselines, with runtime improvements of two orders of magnitude over RANSAC for dense point clouds. Experimental results on public datasets show the superiority of our approach compared to state-of-the-art methods, achieving an average Recall@1 of 90.7% on CS-Wild-Places: an improvement of 29.6 percentage points over baselines, while maintaining high performance on single-source benchmarks with an average Recall@1 of 91.7% and 96.0% on Wild-Places and MulRan, respectively. Our method achieves under 2 m and 5 degrees error for 97.2% of 6-DoF registration attempts, with our multi-scale re-ranking module reducing localisation errors by ~2$\times$ on average. The code will be available upon acceptance.

---

## 33. Galactification: painting galaxies onto dark matter only simulations using a transformer-based model

**论文链接:** [http://arxiv.org/abs/2511.08438v1](http://arxiv.org/abs/2511.08438v1)

**作者:** Shivam Pandey, Christopher C. Lovell, Chirag Modi, Benjamin D. Wandelt

**发布时间:** 2025-11-11

**备注:** 8 pages, 4 figures. , accepted at Machine Learning and the Physical Sciences Workshop at NeurIPS 2025

### GPT解析

### 总结

研究开发了一种基于Transformer的加速前向模型，能够从暗物质模拟快速生成模拟星系目录，完整保留星系属性及其空间分布和条件依赖关系。

### 背景

将星系形成演化与大尺度结构联系起来对于解释宇宙学观测至关重要，但流体动力学模拟在大体积上计算成本过高。

### 目的

开发一个框架，基于廉价的暗物质-only模拟快速生成模拟星系目录，解决计算成本问题。

### 方法

提出多模态Transformer模型，以三维暗物质密度和速度场为输入，输出具有物理属性的星系点云。

### 主要发现

训练后的模型能忠实再现各种星系统计量，正确捕捉星系属性随底层宇宙学和天体物理学参数变化的规律。

### 结论

该模型是首个加速前向模型，能捕捉所有相关星系属性、完整空间分布及流体动力学模拟中的条件依赖关系。

### 翻译

将星系的形成和演化与大尺度结构联系起来对于解释宇宙学观测至关重要。虽然流体动力学模拟能准确模拟星系的关联属性，但在与现代调查相匹配的大体积上运行计算成本过高。我们通过开发一个框架来解决这个问题，该框架基于廉价的暗物质-only模拟快速生成模拟星系目录。我们提出了一种多模态的基于Transformer的模型，以三维暗物质密度和速度场作为输入，输出具有物理属性的相应星系点云。我们证明训练后的模型能忠实再现各种星系统计量，并正确捕捉星系属性随底层宇宙学和天体物理学参数变化的规律，使其成为首个捕捉所有相关星系属性、完整空间分布及其在流体动力学模拟中条件依赖关系的加速前向模型。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何快速生成模拟星系目录的问题。传统流体动力学模拟虽然能准确模拟星系形成，但计算成本极高（单次模拟约需2亿CPU小时），无法匹配现代天文调查的大规模数据需求。而暗物质模拟虽然快100倍以上，但无法直接模拟星系形成过程。这个问题的重要性在于，理解宇宙结构和演化需要大量星系样本，而高效生成模拟星系目录对于分析天文数据、测试宇宙学模型和约束宇宙参数至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到星系形成本质上是随机过程，需要采用生成式方法而非确定性映射。他们选择transformer架构是因为它能高效处理多模态输入和输出，捕捉复杂的条件概率分布。作者借鉴了Pandey等人(2024)的多模态transformer模型架构，并参考了Bourdin等人(2024)的扩散模型和Cuesta-Lazaro & Mishra-Sharma (2024)的点云扩散模型方法，但进行了关键改进以处理可变数量的星系和变化的宇宙学参数。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用基于transformer的多模态模型学习暗物质模拟与星系分布之间的复杂映射关系，将星系形成视为条件概率分布问题。整体流程包括：1)使用成对的N-body和流体动力学模拟数据；2)从N-body模拟提取暗物质密度和速度场作为输入；3)将星系表示为包含位置和性质的标记序列；4)使用编码器(CBAM+Vision Transformer)提取特征；5)通过解码器(transformer层)生成星系序列；6)将解码序列转换为星系目录。整个过程在GPU上仅需30秒，而等效流体动力学模拟需要6000 CPU小时。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)同时模拟星系空间分布和多种物理属性；2)处理可变数量的星系而非固定数量物体；3)结合多尺度环境信息；4)根据宇宙学参数条件化生成；5)计算效率提升100倍。相比之前工作，不同于早期确定性映射算法，采用生成式方法；比Bourdin等人的扩散模型能模拟更多星系属性；比Cuesta-Lazaro的点云扩散模型处理可变数量星系；比Pandey等人的工作扩展到处理变化的宇宙学参数并达到更小尺度。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于transformer的高效模型，能够利用快速暗物质模拟生成与高成本流体动力学模拟相当的完整星系目录，大幅降低了宇宙学模拟的计算成本。'}


### 论文摘要

Connecting the formation and evolution of galaxies to the large-scale structure is crucial for interpreting cosmological observations. While hydrodynamical simulations accurately model the correlated properties of galaxies, they are computationally prohibitive to run over volumes that match modern surveys. We address this by developing a framework to rapidly generate mock galaxy catalogs conditioned on inexpensive dark-matter-only simulations. We present a multi-modal, transformer-based model that takes 3D dark matter density and velocity fields as input, and outputs a corresponding point cloud of galaxies with their physical properties. We demonstrate that our trained model faithfully reproduces a variety of galaxy summary statistics and correctly captures their variation with changes in the underlying cosmological and astrophysical parameters, making it the first accelerated forward model to capture all the relevant galaxy properties, their full spatial distribution, and their conditional dependencies in hydrosimulations.

---

## 34. Registration-Free Monitoring of Unstructured Point Cloud Data via Intrinsic Geometrical Properties

**论文链接:** [http://arxiv.org/abs/2511.05623v1](http://arxiv.org/abs/2511.05623v1)

**作者:** Mariafrancesca Patalano, Giovanna Capizzi, Kamran Paynabar

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了一种无需配准的复杂形状点云监测新方法，消除了传统预处理步骤的需要。

### 背景

现代传感技术能够收集不同大小的非结构点云数据(PCD)，被广泛应用于先进制造过程（增材、减材和混合制造）中监测3D物体的几何精度。传统方法通常需要配准和网格重建等预处理步骤，但这些步骤容易出错、耗时且可能引入伪影。

### 目的

开发一种无需配准和网格重建的点云监测方法，避免传统预处理步骤的缺点，确保分析一致性和避免误报。

### 方法

提出两种替代性的特征学习方法和一个通用的监测方案。特征学习方法利用形状的内蕴几何特性，通过拉普拉斯和测地距离捕获。监测方案中，使用阈值技术进一步选择最能指示潜在失控状态的内蕴特征。

### 主要发现

数值实验和案例研究表明，所提出的方法能有效识别不同类型的缺陷。

### 结论

这种无需配准的点云监测方法在复杂形状监测中表现出色，避免了传统预处理步骤的问题，提高了监测的准确性和效率。

### 翻译

现代传感技术使得能够收集不同大小的非结构点云数据(PCD)，用于监测3D物体的几何精度。PCD广泛应用于先进制造过程，包括增材、减材和混合制造。为确保分析一致性并避免误报，通常在监测前进行配准和网格重建等预处理步骤。然而，这些步骤容易出错、耗时且可能引入伪影，可能影响监测结果。本文提出了一种无需配准的复杂形状点云监测新方法，消除了配准和网格重建的需要。我们的提案包括两种替代性特征学习方法和一个通用监测方案。特征学习方法利用通过拉普拉斯和测地距离捕获的形状内蕴几何特性。在监测方案中，使用阈值技术进一步选择最能指示潜在失控状态的内蕴特征。数值实验和案例研究强调了所提出方法在识别不同类型缺陷方面的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何对无结构点云数据进行质量监控，而无需依赖点云配准和网格重建这两个预处理步骤。这个问题在先进制造领域非常重要，因为传统方法需要耗时的预处理步骤，这些步骤容易出错并可能引入伪影，影响监控准确性。特别是对于复杂形状的3D打印件，这些预处理更具挑战性，可能导致监控结果不可靠。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统监控方法对无结构点云数据的局限性，然后分析了现有工作：表面数据监控方法对全3D形状监控计算成本高；基于偏差的方法依赖配准或丢失空间信息；基于内在属性的方法仅适用于封闭网格。作者借鉴了拉普拉斯-贝尔特拉米算子在几何处理中的应用和测地距离计算的热方法，设计了两种替代方法：一种基于稳健拉普拉斯的频谱，另一种基于测地距离。整体思路是利用对象的内在几何属性，通过谱降维技术捕获形状特征，并应用阈值技术选择最具指示性的特征进行监控。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用对象的内在几何属性来监控点云数据，这些属性不受对象位置、方向或等形变的影响。整体流程包括：1) 特征学习阶段：RFM-RL方法计算稳健拉普拉斯的频谱并选择低频部分作为特征；RFM-HM方法使用热方法计算测地距离矩阵并导出相似矩阵，然后计算特征分解。2) 监控方案阶段：从训练集和调优集计算特征向量，假设正常状态下的特征向量来自某个分布，在监控新点云时使用三种控制统计量之一检测变化。3) 控制图设计阶段：选择训练集大小和特征数量，使用调优集估计控制界限。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 完全避免配准和网格重建步骤；2) 提出两种替代的特征学习方法；3) 直接处理原始点云数据；4) 适用于开放网格（如空心对象）；5) 灵活的监控方案。相比之前的工作，不同之处在于：传统基于偏差的方法依赖配准或丢失空间信息；之前基于拉普拉斯-贝尔特拉米算子的方法仅适用于封闭网格；流形学习方法仅近似测地距离；而本文方法保留了空间信息，无需配准，可处理开放网格，且直接处理点云数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种无需配准和网格重建的创新方法，通过利用点云数据的内在几何属性，实现了对复杂形状3D打印件的精确质量监控。'}


### 论文摘要

Modern sensing technologies have enabled the collection of unstructured point cloud data (PCD) of varying sizes, which are used to monitor the geometric accuracy of 3D objects. PCD are widely applied in advanced manufacturing processes, including additive, subtractive, and hybrid manufacturing. To ensure the consistency of analysis and avoid false alarms, preprocessing steps such as registration and mesh reconstruction are commonly applied prior to monitoring. However, these steps are error-prone, time-consuming and may introduce artifacts, potentially affecting monitoring outcomes. In this paper, we present a novel registration-free approach for monitoring PCD of complex shapes, eliminating the need for both registration and mesh reconstruction. Our proposal consists of two alternative feature learning methods and a common monitoring scheme. Feature learning methods leverage intrinsic geometric properties of the shape, captured via the Laplacian and geodesic distances. In the monitoring scheme, thresholding techniques are used to further select intrinsic features most indicative of potential out-of-control conditions. Numerical experiments and case studies highlight the effectiveness of the proposed approach in identifying different types of defects.

---

## 35. Faint galaxies in the Zone of Avoidance revealed by JWST/NIRCam

**论文链接:** [http://arxiv.org/abs/2510.12488v1](http://arxiv.org/abs/2510.12488v1)

**作者:** J. L. Nilo-Castellón, M. V. Alonso, L. Baravalle, C. Villalon, C. N. A. Willmer, C. Valotto, M. Soto, D. Minniti, M. A. Sgró, I. V. Daza-Perilla, H. Cuevas Larenas, A. Ramirez, J. Alonso-García, P. Marchant Cortés, F. Milla Castro

**发布时间:** 2025-10-14

**备注:** Accepted for publication in Astronomy & Astrophysics (Section 4: Extragalactic Astronomy). Minor editorial corrections pending (reference updates and JWST typesetting)

### GPT解析

### 总结

本研究评估了詹姆斯·韦伯太空望远镜(JWST)的近红外相机(NIRCam)在银河系遮挡区(ZoA)探测河外星系的能力。通过分析NGC 3324区域的JWST图像，研究团队成功识别出102个星系，展示了JWST/NIRCam在高度遮挡的银河区域探测河外源的潜力，为绘制大尺度结构图开辟了新途径。

### 背景

银河系遮挡区(ZoA)是构建宇宙三维综合地图的最后前沿之一。由于银河系消光、恒星密集和噪声混淆，历史上在这些区域探测背景星系受到限制，这对大尺度结构和宇宙学测量产生重要影响。

### 目的

评估詹姆斯·韦伯太空望远镜(JWST)近红外相机(NIRCam)在银河系高度污染区域探测河外源的能力，探索其在银河系遮挡区(ZoA)绘制大尺度结构图的潜力。

### 方法

分析NGC 3324区域的JWST/NIRCam宽滤光片图像，使用定制的SExtractor v2.28软件进行源检测。在F444W波段检测源，并与F090W和F200W波段交叉匹配，然后与DAOPHOT点扩散函数光度测量进行验证。通过半高全宽(FWHM)与信噪比(SNR)标准以及人工检查获得精炼样本。

### 主要发现

1. 在JWST/NIRCam视场内识别出102个星系；2. 星等分布呈双峰模式，约10%的星系亮于15星等，约60%在17-19星等范围内；3. 典型大小约6.5角秒，从紧凑到扩展系统；4. 形态学上从紧凑型到螺旋系和透镜状系统，包括一个紧凑星系群；5. 还探测到在分子云最不透明区域仍可见的'星系云'。

### 结论

这些结果证明了JWST/NIRCam能够高度遮挡的银河区域探测河外源，为跨越ZoA绘制大尺度结构图开辟了新途径，对宇宙学研究和三维宇宙地图构建具有重要意义。

### 翻译

银河系遮挡区(ZoA)仍然是构建宇宙三维综合地图的最后前沿之一。银河系消光、恒星密集和噪声混淆历史上限制了这些区域背景星系的探测，这对大尺度结构和宇宙学测量产生影响。我们评估了詹姆斯·韦伯太空望远镜(JWST)近红外相机(NIRCam)在银河系高度污染区域探测河外源的能力。我们分析了NGC 3324的JWST/NIRCam宽滤光片图像，使用定制的SExtractor v2.28实现。在F444W波段检测源，与F090W和F200W交叉匹配，并与最近的DAOPHOT点扩散函数光度测量进行验证。通过半高全宽与信噪比标准和人工检查获得精炼样本。我们在JWST/NIRCam视场内识别出102个星系。星等分布呈双峰模式，约10%的星系亮于15星等，约60%的星系在17-19星等范围内。典型大小约6.5角秒，从紧凑到扩展系统，等面积区域最大达约2000像素。形态学上从紧凑型到螺旋系和透镜状系统，包括在视场东部边缘的一个紧凑星系群。我们还报告了'星系云'的探测，这些星系在分子云最不透明的区域仍然可见。这些结果证明了JWST/NIRCam能够高度遮挡的银河区域探测河外源，为跨越ZoA绘制大尺度结构图开辟了新途径。


### 论文摘要

The Zone of Avoidance (ZoA) remains one of the last frontiers in constructing a comprehensive three-dimensional map of the Universe. Galactic extinction, stellar crowding, and confusion noise have historically limited the detection of background galaxies in these regions, with implications for large-scale structure and cosmological measurements. We assess the capability of the James Webb Space Telescope (JWST) Near Infrared Camera (NIRCam) to detect extragalactic sources in a heavily contaminated region of the Milky Way. We analyzed JWST/NIRCam wide-filter images of NGC 3324 with a customized implementation of SExtractor v2.28. Sources were detected in the F444W band, cross-matched with F090W and F200W, and validated against recent DAOPHOT point spread function (PSF) photometry. A refined sample was obtained through full width at half maximum (FWHM) - signal-to-noise ratio (SNR) criteria and visual inspection. We identified 102 galaxies across the JWST/NIRCam field of view. The magnitude (F444W) distribution is bimodal, with about 10% brighter than m_F444W < 15 mag and about 60% in the range 17 < m_F444W < 19 mag. Typical sizes are FWHM ~6.5 arcsec, from compact to extended systems with isophotal areas up to ~2000 pixels (~7.9 arcsec^2). Morphologies span from compact to spiral and lenticular systems, including a compact group at the eastern edge of the field. We also report the detection of "transnebular galaxies", visible through the most opaque regions of the molecular cloud. These results demonstrate the potential of JWST/NIRCam to probe extragalactic sources through highly obscured Galactic regions, opening new avenues for mapping large-scale structures across the ZoA.

---

## 36. Gesplat: Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2510.10097v2](http://arxiv.org/abs/2510.10097v2)

**作者:** Jiahui Lu, Haihong Xiao, Xueyan Zhao, Wenxiong Kang

**发布时间:** 2025-10-11

### GPT解析

### 总结

Gesplat是一种基于3DGS的框架，可以从未配准的稀疏图像实现鲁棒的新视角合成和几何一致的重建，克服了NeRF和3DGS对准确相机姿态和密集视角覆盖的依赖限制。

### 背景

NeRF和3DGS在3D重建和新视角合成方面取得了进展，但它们严重依赖准确的相机姿态和密集的视角覆盖，限制了它们在稀疏视角设置中的应用。

### 目的

为了克服NeRF和3DGS在稀疏视角设置中的局限性，提出一种可以从未配准的稀疏图像实现鲁棒新视角合成和几何一致重建的框架。

### 方法

Gesplat利用VGGT基础模型获得更可靠的初始姿态和密集点云，并集成了混合高斯表示（双位置-形状优化和跨视图匹配一致性）、基于图的属性细化模块和基于流的深度正则化等创新技术。

### 主要发现

全面的定量和定性实验表明，Gesplat与前向和大规模复杂数据集上的其他无姿态方法相比，实现了更鲁棒的性能。

### 结论

Gesplat通过创新的混合高斯表示、图引导属性细化模块和基于流的深度正则化，成功实现了从稀疏未配准图像的鲁棒3D重建和新视角合成。

### 翻译

神经辐射场(NeRF)和3D高斯泼溅(3DGS)推动了3D重建和新视角合成的发展，但仍然严重依赖准确的相机姿态和密集的视角覆盖。这些要求限制了它们在稀疏视角设置中的应用，其中姿态估计变得不可靠且监督不足。为了克服这些挑战，我们引入了Gesplat，一个基于3DGS的框架，可以从未配准的稀疏图像实现鲁棒的新视角合成和几何一致的重建。与依赖COLMAP进行稀疏点云初始化的先前工作不同，我们利用VGGT基础模型获得更可靠的初始姿态和密集点云。我们的方法集成了几个关键创新：1)具有双位置-形状优化和跨视图匹配一致性增强的混合高斯表示；2)基于图的属性细化模块以增强场景细节；以及3)基于流的深度正则化，提高深度估计准确性以实现更有效的监督。全面的定量和定性实验证明，与其他无姿态方法相比，我们的方法在前向和大规模复杂数据集上实现了更鲁棒的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从稀疏视角且未提供相机姿态的图像中进行鲁棒的3D重建和新视角合成问题。这个问题在现实中很重要，因为获取密集、覆盖良好的图像集往往不切实际且成本高昂，而传统方法如NeRF和3DGS严重依赖准确的相机姿态和密集视角覆盖，这在实际应用场景（如自动驾驶、VR/AR和机器人技术）中难以满足。稀疏视角设置下的监督不足会导致训练过程中的伪影和有缺陷的重建，限制了这些方法在真实世界场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法在稀疏视角和无姿态设置下的局限性，发现传统方法如COLMAP在稀疏视角下无法产生可靠的相机参数。他们意识到需要更强大的几何先验来约束场景结构。设计方法时借鉴了VGGT基础模型来获得更可靠的初始姿态和密集点云，替代传统的COLMAP。基于InstantSplat框架进行开发，但引入了混合高斯表示结合匹配先验、图引导优化模块和基于流的深度正则化等创新。他们还利用了图神经网络和光学流等技术来提高重建质量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用混合高斯表示结合多视图匹配先验来增强几何一致性，通过图引导优化细化场景细节，利用基于流的深度正则化提高深度估计准确性，并联合优化高斯参数和相机姿态。整体流程包括：1)使用VGGT模型生成初始密集点云和相机姿态；2)提取匹配先验信息；3)初始化混合高斯表示；4)优化高斯位置和形状；5)使用图神经网络优化高斯属性；6)应用基于流的深度正则化；7)联合优化高斯参数和相机姿态，最终实现高质量的3D重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)混合高斯表示与双位置-形状优化，基于匹配先验增强多视图一致性；2)图引导的属性细化模块，使用图神经网络优化高斯属性；3)基于流的深度正则化，利用光学流提高深度估计准确性；4)使用VGGT替代传统COLMAP获得更可靠的初始姿态和密集点云。相比之前的工作，Gesplat不需要密集视角覆盖，在稀疏视角下表现更好；使用3DGS而非MLP（如NoPe-NeRF）具有更快的训练和渲染速度；引入了混合高斯表示和深度正则化（相比CF-3DGS）更好地保留了细节；使用VGGT而非DUSt3R进行初始化（相比InstantSplat），提高了几何一致性和细节质量。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Gesplat通过引入混合高斯表示、图引导优化和基于流的深度正则化，实现了从稀疏无姿态图像中进行鲁棒、高质量的3D重建和新视角合成，显著优于现有方法。'}


### 论文摘要

Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have advanced 3D reconstruction and novel view synthesis, but remain heavily dependent on accurate camera poses and dense viewpoint coverage. These requirements limit their applicability in sparse-view settings, where pose estimation becomes unreliable and supervision is insufficient. To overcome these challenges, we introduce Gesplat, a 3DGS-based framework that enables robust novel view synthesis and geometrically consistent reconstruction from unposed sparse images. Unlike prior works that rely on COLMAP for sparse point cloud initialization, we leverage the VGGT foundation model to obtain more reliable initial poses and dense point clouds. Our approach integrates several key innovations: 1) a hybrid Gaussian representation with dual position-shape optimization enhanced by inter-view matching consistency; 2) a graph-guided attribute refinement module to enhance scene details; and 3) flow-based depth regularization that improves depth estimation accuracy for more effective supervision. Comprehensive quantitative and qualitative experiments demonstrate that our approach achieves more robust performance on both forward-facing and large-scale complex datasets compared to other pose-free methods.

---

## 37. Photonic-integrated quantum sensor array for microscale magnetic localisation

**论文链接:** [http://arxiv.org/abs/2511.11496v1](http://arxiv.org/abs/2511.11496v1)

**作者:** Hao-Cheng Weng, John G. Rarity, Krishna C. Balram, Joe A. Smith

**发布时间:** 2025-11-14

**备注:** 13 pages, 5 figures

### GPT解析

### 总结

研究人员通过将氮空位中心与硅氮化物光子集成电路集成，成功实现了八个局部化NV传感器的阵列化操作，能够同时独立读取各个传感器。利用这些传感器和机器学习方法，他们实现了30微米针尖的微观磁定位，并展示了其在跟踪磁性微机器人方面的潜力。

### 背景

氮空位中心(NVs)是有前途的固态纳米级量子传感器，应用范围从材料科学到生物技术。使用多个传感器同步操作可探测波动场的时空相关性或点缺陷的动力学。

### 目的

实现多个氮空位中心传感器的同步操作，用于精确探测和定位微观物体，特别是生物医学应用中的磁性微机器人。

### 方法

将氮空位中心与硅氮化物光子集成电路集成，构建八个局部化NV传感器阵列，实现同时独立读取。使用八个NV传感器和机器学习方法进行多点磁场重建。

### 主要发现

成功实现了30微米针尖的微观磁定位，定位误差低于针尖尺寸，并能高保真度动态跟踪。模拟验证了该平台可用于监测生物和临床用途的磁性微机器人的位置和方向。

### 结论

这种光子集成多传感器平台无需复杂的光体光学，为在实验室外条件下实现真实生物医学应用提供了可能。

### 翻译

氮空位中心(NVs)是有前途的固态纳米级量子传感器，应用范围从材料科学到生物技术。使用多个传感器同步操作可探测波动场的时空相关性或点缺陷的动力学。在本工作中，通过将NVs与硅氮化物光子集成电路集成，我们实现了八个局部化NV传感器阵列的可扩展操作，并能同时、独立地读取各个传感器。使用这八个NV传感器和机器学习方法进行多点磁场重建，我们展示了30微米大小的针尖的微观磁定位。实验中，针尖可以被定位在其尺寸以下的误差范围内，并可以高保真度地进行动态跟踪。我们进一步模拟了该平台用于监测生物和临床用途的磁性微机器人的位置和方向的可行性。无需复杂的光体光学，这种光子集成多传感器平台为在实验室外条件下实现真实生物医学应用迈出了一步。


### 论文摘要

Nitrogen-vacancy centres (NVs) are promising solid-state nanoscale quantum sensors for applications ranging from material science to biotechnology. Using multiple sensors simultaneously offers advantages for probing spatiotemporal correlations of fluctuating fields or the dynamics of point defects. In this work, by integrating NVs with foundry silicon-nitride photonic integrated circuits, we realise the scalable operation of eight localised NV sensors in an array, with simultaneous, distinct readout of the individual sensors. Using the eight NV sensors and machine-learning methods for multi-point magnetic field reconstruction, we demonstrate microscale magnetic localisation of a 30 $μ$m-sized needle tip. Experimentally, the needle tip can be localised with an error below its dimension and tracked dynamically with high fidelity. We further simulate the feasibility of our platform for monitoring the position and orientation of magnetic microrobots designed for biological and clinical purposes. Without the complexity of bulk optics, our photonic-integrated multi-sensor platform presents a step towards real-life biomedical applications under out-of-the-lab conditions.

---

## 38. MoCap2Radar: A Spatiotemporal Transformer for Synthesizing Micro-Doppler Radar Signatures from Motion Capture

**论文链接:** [http://arxiv.org/abs/2511.11462v1](http://arxiv.org/abs/2511.11462v1)

**作者:** Kevin Chen, Kenneth W. Parker, Anish Arora

**发布时间:** 2025-11-14

### GPT解析

### 总结

提出了一种纯机器学习方法，用于从动作捕捉数据合成雷达频谱图。

### 背景

需要从动作捕捉数据生成雷达频谱图，但传统方法可能存在计算复杂或数据稀缺的问题。

### 目的

开发一种基于机器学习的框架，将动作捕捉数据转换为多普勒雷达频谱图。

### 方法

使用基于Transformer的模型，将MoCap到频谱图的转换视为窗口序列到序列的任务，该模型同时捕捉标记间的空间关系和跨帧的时间动态。

### 主要发现

实际实验显示该方法生成的频谱图在视觉和定量上均合理且泛化性好；消融实验证明模型能将多部分动作转换为多普勒特征并理解人体部位间的空间关系。

### 结论

该方法展示了Transformer在时间序列信号处理中的应用潜力；特别适用于边缘计算和物联网雷达；可用丰富的MoCap数据增强稀缺雷达数据集；计算量远小于基于物理的方法。

### 翻译

我们提出了一种纯机器学习过程，用于从动作捕捉数据合成雷达频谱图。我们将MoCap到频谱图的转换视为基于窗口的序列到序列任务，使用基于Transformer的模型，该模型能够同时捕捉MoCap标记之间的空间关系和跨帧的时间动态。实际实验表明，所提出的方法能够生成视觉和定量上都合理的多普勒雷达频谱图，并具有良好的泛化能力。消融实验表明，学习到的模型既可以将多部分动作转换为多普勒特征，又理解人体不同部分之间的空间关系。这是一个使用Transformer进行时间序列信号处理的有趣例子，特别适用于边缘计算和物联网雷达。它还表明可以使用更丰富的MoCap数据来增强稀缺的雷达数据集，用于训练更高级的应用。最后，与基于物理的生成雷达数据方法相比，它的计算需求要小得多。


### 论文摘要

We present a pure machine learning process for synthesizing radar spectrograms from Motion-Capture (MoCap) data. We formulate MoCap-to-spectrogram translation as a windowed sequence-to-sequence task using a transformer-based model that jointly captures spatial relations among MoCap markers and temporal dynamics across frames. Real-world experiments show that the proposed approach produces visually and quantitatively plausible doppler radar spectrograms and achieves good generalizability. Ablation experiments show that the learned model includes both the ability to convert multi-part motion into doppler signatures and an understanding of the spatial relations between different parts of the human body.   The result is an interesting example of using transformers for time-series signal processing. It is especially applicable to edge computing and Internet of Things (IoT) radars. It also suggests the ability to augment scarce radar datasets using more abundant MoCap data for training higher-level applications. Finally, it requires far less computation than physics-based methods for generating radar data.

---

## 39. Computationally-efficient deep learning models for nowcasting of precipitation: A solution for the Weather4cast 2025 challenge

**论文链接:** [http://arxiv.org/abs/2511.11197v1](http://arxiv.org/abs/2511.11197v1)

**作者:** Anushree Bhuskute, Kaushik Gopalan, Jeet Shah

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究提出了一种基于卷门控循环单元的迁移学习框架，用于Weather4Cast 2025比赛中的短期降雨预测，在累积降雨任务中获得第二名。

### 背景

研究基于Weather4Cast 2025比赛，使用SEVIRI红外通道（10.8微米波长）作为输入数据，该数据包含一小时内四次观测值。

### 目的

开发一种能够预测未来四小时降雨情况的模型，并在事件预测任务中也应用该模型。

### 方法

采用两阶段训练策略：第一阶段训练ConvGRU预测SEVIRI亮温，第二阶段使用经验推导的非线性变换将预测场转换为OPERA兼容的降雨率；事件预测任务中使用3D事件检测和时空特征提取处理转换后的降雨预报。

### 主要发现

在累积降雨任务中获得第二名；同一模型未经修改用于事件预测任务，与基线模型获得相似分数。

### 结论

基于ConvGRU的迁移学习框架在短期降雨预测中表现良好，且模型具有较好的通用性，可用于不同任务。

### 翻译

本研究提出了一个基于卷门控循环单元的迁移学习框架，用于Weather4Cast 2025比赛中的短期降雨预测。使用单个SEVIRI红外通道（10.8微米波长）作为输入，包含一小时内四次观测。采用两阶段训练策略生成长达四小时的降雨估计。第一阶段训练ConvGRU预测SEVIRI的亮温，使模型能够捕捉相关的时空模式。第二阶段使用经验推导的非线性变换将预测场转换为OPERA兼容的降雨率。对于事件预测任务，转换后的降雨预报经过3D事件检测和时空特征提取，以识别和表征降水事件。我们的提交在累积降雨任务中获得第二名。此外，同一模型未经修改用于事件预测任务，结果与竞赛基线模型相似。


### 论文摘要

This study presents a transfer-learning framework based on Convolutional Gated Recurrent Units (ConvGRU) for short-term rainfall prediction in the Weather4Cast 2025 competition. A single SEVIRI infrared channel (10.8 μm wavelength) is used as input, which consists of four observations over a one-hour period. A two-stage training strategy is applied to generate rainfall estimates up to four hours ahead. In the first stage, ConvGRU is trained to forecast the brightness temperatures from SEVIRI, enabling the model to capture relevant spatiotemporal patterns. In the second stage, an empirically derived nonlinear transformation maps the predicted fields to OPERA-compatible rainfall rates.   For the event-prediction task, the transformed rainfall forecasts are processed using 3D event detection followed by spatiotemporal feature extraction to identify and characterize precipitation events. Our submission achieved 2nd place in the cumulative rainfall task. Further, the same model was used out-of-the-box for the event prediction task, and resulted in similar scores as the baseline model to the competition.

---

## 40. CATS-V2V: A Real-World Vehicle-to-Vehicle Cooperative Perception Dataset with Complex Adverse Traffic Scenarios

**论文链接:** [http://arxiv.org/abs/2511.11168v1](http://arxiv.org/abs/2511.11168v1)

**作者:** Hangyu Li, Bofeng Cao, Zhaohui Liang, Wuzhen Li, Juyoung Oh, Yuxuan Chen, Shixiao Liang, Hang Zhou, Chengyuan Ma, Jiaxi Liu, Zheng Li, Peng Zhang, KeKe Long, Maolin Liu, Jackson Jiang, Chunlei Yu, Shengxiang Liu, Hongkai Yu, Xiaopeng Li

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文介绍了CATS-V2V数据集，这是首个针对复杂不良交通场景下的V2V协作感知的真实世界数据集，包含多种传感器数据和多场景标注，旨在促进自动驾驶相关研究。

### 背景

V2V协作感知有潜力通过克服复杂不良交通场景中的感知限制来增强自动驾驶性能。然而，现有数据集主要关注普通交通场景，限制了协作感知的优势。

### 目的

引入CATS-V2V数据集，填补V2V协作感知在复杂不良交通场景下的数据空白。

### 方法

数据集由两个硬件时间同步的车辆收集，覆盖10个不同地点的10种天气和光照条件，包含60K帧激光雷达点云、1.26M多视角相机图像、750K高精度GNSS和IMU记录，并提供时间一致的3D边界框标注和基于目标的时间对齐方法。

### 主要发现

CATS-V2V是目前同类数据集中最大规模、支持最全面、质量最高的数据集。

### 结论

CATS-V2V将有益于自动驾驶社区在相关任务中的研究和发展。

### 翻译

车对车(V2V)协作感知通过克服复杂不良交通场景(CATS)中的感知限制，在增强自动驾驶性能方面具有巨大潜力。同时，数据是现代自动驾驶AI的基础设施。然而，由于严格的数据收集要求，现有数据集主要关注普通交通场景，限制了协作感知的优势。为应对这一挑战，我们引入了CATS-V2V，这是首个针对复杂不良交通场景下的V2V协作感知的真实世界数据集。该数据集由两个硬件时间同步的车辆收集，覆盖10个不同地点的10种天气和光照条件。这100个剪辑的数据集包括60K帧的10Hz激光雷达点云和1.26M多视角30Hz相机图像，以及750K匿名但高精度的RTK固定GNSS和IMU记录。相应地，我们为对象提供了时间一致的3D边界框标注，以及静态场景以构建4D BEV表示。在此基础上，我们提出了基于目标的时间对齐方法，确保所有对象在所有传感器模态中精确对齐。我们希望CATS-V2V，作为迄今为止同类数据集中最大规模、支持最全面、质量最高的数据集，将有益于自动驾驶社区在相关任务中的研究。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文解决的是现有车辆间协同感知(V2V)数据集缺乏复杂不利交通场景(CATS)覆盖的问题。这些场景包括恶劣天气(雨、雪、雾)、特殊光照条件(强光、低光、夜间)和不规则工作区域等，它们虽然罕见但对自动驾驶安全至关重要。现有数据集主要关注普通交通场景，无法有效测试和验证协同感知技术在极端条件下的性能，限制了自动驾驶系统在真实世界中的鲁棒性和泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有V2X数据集的局限性，特别是CATS覆盖不足的问题。他们选择了真实世界数据收集而非模拟，以确保数据真实性；采用两车协同而非车-基础设施方案，以提高灵活性。他们借鉴了现有V2X数据集(如DAIR-V2X、V2X-Seq)和单车辆数据集(如KITTI、Waymo)的经验，但在时间同步精度、传感器配置和时序对齐方法上进行了创新。作者设计了专门的传感器配置(如双返回模式激光雷达应对雨雪)和高精度时间同步系统(1毫秒精度)，并提出了基于目标的时序对齐方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个覆盖复杂不利交通场景的真实世界V2V协同感知数据集，通过高精度时间同步和多模态传感器配置确保数据质量，并提出创新的时序对齐方法提高不同传感器间的一致性。整体流程包括：1)配置两辆实验车，每车配备128束激光雷达、7个摄像头和高精度INS；2)在10个不同地点的10种天气和光照条件下收集数据；3)对激光雷达数据进行运动补偿，将两车点云注册到统一坐标系；4)提供精确的3D边界框标注和全局唯一对象ID；5)提出基于目标的时序对齐方法，计算每个对象点的平均时间戳并与最近摄像头帧关联。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专门针对复杂不利交通场景的真实世界V2V数据集；2)提供高帧率(10Hz激光雷达、30Hz摄像头)和高精度传感器数据；3)实现1毫秒的硬件级时间同步精度；4)提出基于目标的时序对齐方法。相比之前工作，CATS-V2V专注于V2V而非V2I协同感知，提供更丰富的场景覆盖，更高的数据质量和同步精度，以及创新的时序对齐方法。它还包含双返回模式激光雷达等特殊配置，提高了在不利条件下的鲁棒性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CATS-V2V是首个覆盖复杂不利交通场景的真实世界车辆间协同感知数据集，通过高精度时间同步、多模态传感器配置和创新的基于目标的时序对齐方法，为自动驾驶社区提供了迄今为止规模最大、支持最全面、质量最高的V2V协同感知数据资源。'}


### 论文摘要

Vehicle-to-Vehicle (V2V) cooperative perception has great potential to enhance autonomous driving performance by overcoming perception limitations in complex adverse traffic scenarios (CATS). Meanwhile, data serves as the fundamental infrastructure for modern autonomous driving AI. However, due to stringent data collection requirements, existing datasets focus primarily on ordinary traffic scenarios, constraining the benefits of cooperative perception. To address this challenge, we introduce CATS-V2V, the first-of-its-kind real-world dataset for V2V cooperative perception under complex adverse traffic scenarios. The dataset was collected by two hardware time-synchronized vehicles, covering 10 weather and lighting conditions across 10 diverse locations. The 100-clip dataset includes 60K frames of 10 Hz LiDAR point clouds and 1.26M multi-view 30 Hz camera images, along with 750K anonymized yet high-precision RTK-fixed GNSS and IMU records. Correspondingly, we provide time-consistent 3D bounding box annotations for objects, as well as static scenes to construct a 4D BEV representation. On this basis, we propose a target-based temporal alignment method, ensuring that all objects are precisely aligned across all sensor modalities. We hope that CATS-V2V, the largest-scale, most supportive, and highest-quality dataset of its kind to date, will benefit the autonomous driving community in related tasks.

---

## 41. PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI

**论文链接:** [http://arxiv.org/abs/2511.11048v1](http://arxiv.org/abs/2511.11048v1)

**作者:** Sun Jo, Seok Young Hong, JinHyun Kim, Seungmin Kang, Ahjin Choi, Don-Gwan An, Simon Song, Je Hyeong Hong

**发布时间:** 2025-11-14

**备注:** Accepted at AAAI 2026. Supplementary material included after references. 27 pages, 21 figures, 11 tables

### GPT解析

### 总结

本文提出PINGS-X，一种用于4D flow MRI超分辨率的新框架，使用轴对齐的时空高斯表示建模高分辨率流速。该方法通过三个创新显著减少了训练时间，同时实现了优越的超分辨率准确性。

### 背景

4D flow MRI是一种可靠、非侵入性的血流速度估计方法，对心血管诊断至关重要。与传统MRI不同，它需要高时空分辨率来及早发现关键状况，但高分辨率通常导致扫描时间延长，在采集速度和预测准确性间产生权衡。现有物理信息神经网络(PINNs)方法因训练过程极慢且需为每个患者单独进行而适用性有限。

### 目的

克服现有PINNs方法在MRI数据超分辨率中的局限性，开发一种能够快速训练且保持高准确性的超分辨率方法，解决训练速度与准确性之间的权衡问题。

### 方法

提出PINGS-X框架，受3D高斯飞溅(3DGS)在新型视图合成中的有效性启发，包含三个主要创新：(i)具有正式收敛保证的归一化高斯飞溅；(ii)轴对齐高斯，简化高维数据训练同时保持准确性和收敛保证；(iii)高斯合并过程，防止退化解并提高计算效率。

### 主要发现

在计算流体动力学(CFD)和真实4D flow MRI数据集上的实验结果表明，PINGS-X显著减少了训练时间，同时实现了优越的超分辨率准确性，解决了现有方法的局限性。

### 结论

PINGS-X是一种有效的4D flow MRI超分辨率方法，通过创新的轴对齐高斯表示和优化的训练过程，实现了快速训练和高准确性，为心血管诊断提供了更实用的工具。

### 翻译

4D flow magnetic resonance imaging (MRI)是一种可靠、非侵入性的血流速度估计方法，对心血管诊断至关重要。与传统专注于解剖结构的MRI不同，4D flow MRI需要高时空分辨率来及早发现狭窄或动脉瘤等关键状况。然而，实现这样的分辨率通常会导致扫描时间延长，在采集速度和预测准确性之间产生权衡。最近的研究利用物理信息神经网络(PINNs)对MRI数据进行超分辨率，但它们的实际适用性有限，因为必须为每个患者执行极慢的训练过程。为克服这一限制，我们提出了PINGS-X，一种使用轴对齐的时空高斯表示建模高分辨率流速的新框架。受3D高斯飞溅(3DGS)在新型视图合成中的有效性的启发，PINGS-X通过几个重要的创新扩展了这一概念：(i)具有正式收敛保证的归一化高斯飞溅；(ii)轴对齐高斯，简化高维数据的训练同时保持准确性和收敛保证；(iii)高斯合并过程，防止退化解并提高计算效率。在计算流体动力学(CFD)和真实4D flow MRI数据集上的实验结果表明，PINGS-X显著减少了训练时间，同时实现了优越的超分辨率准确性。我们的代码和数据集可在https://github.com/SpatialAILab/PINGS-X获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决4D流磁共振成像(4D flow MRI)的超分辨率问题。这个问题很重要，因为高分辨率的4D flow MRI对早期检测心血管疾病(如狭窄或动脉瘤)至关重要，但传统高分辨率扫描时间过长，对患者不友好且易产生运动伪影；而现有的加速技术存在噪声放大和结构模糊等问题；物理信息神经网络(PINNs)虽然不需要大型数据集，但训练速度极慢，限制了临床可行性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了PINNs在4D flow MRI超分辨率中的计算瓶颈，类似于新视图合成中NeRF面临的挑战。从3D高斯溅射(3DGS)在新视图合成中相比NeRF的显著速度优势获得灵感，将3DGS的原理适应到4D速度场的建模中。针对物理信息超分辨率的特殊需求，提出了归一化高斯溅射、轴对齐高斯表示和高斯合并等创新。该方法借鉴了3DGS的表示效率和训练速度优势，利用了PINNs的物理约束思想，并参考了Nadaraya-Watson估计器等非参数回归方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用显式的4D时空高斯表示来建模高分辨率流速场，结合物理信息约束和归一化高斯溅射技术，实现高效且准确的超分辨率。整体流程包括：1)初始化高斯集合；2)使用归一化高斯溅射预测值；3)计算数据保真项和物理信息正则化项的组合损失；4)优化高斯参数并定期应用自适应密度控制(分裂/克隆高斯和合并相似高斯)；5)使用训练好的高斯参数进行推理生成高分辨率输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)归一化高斯溅射(NGS)与形式收敛保证，解决了传统非归一化加权的预测在远离高斯中心区域崩溃的问题；2)轴对齐高斯用于高效训练，简化高维数据计算，将协方差参数从10个减少到4个；3)高斯合并用于稳定性和可扩展性，防止退化解并提高计算效率。相比之前的工作，PINGS-X相比PIGS提供了理论收敛保证和更高精度；相比PIG避免了神经网络计算开销，训练速度更快；相比传统PINNs显著缩短了训练时间；相比数据驱动方法减少了对大型数据集的依赖。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PINGS-X通过引入归一化高斯溅射、轴对齐高斯表示和高斯合并技术，将3D高斯溅射的效率优势带入物理信息超分辨率领域，显著提高了4D flow MRI超分辨率的训练速度和预测准确性。'}


### 论文摘要

4D flow magnetic resonance imaging (MRI) is a reliable, non-invasive approach for estimating blood flow velocities, vital for cardiovascular diagnostics. Unlike conventional MRI focused on anatomical structures, 4D flow MRI requires high spatiotemporal resolution for early detection of critical conditions such as stenosis or aneurysms. However, achieving such resolution typically results in prolonged scan times, creating a trade-off between acquisition speed and prediction accuracy. Recent studies have leveraged physics-informed neural networks (PINNs) for super-resolution of MRI data, but their practical applicability is limited as the prohibitively slow training process must be performed for each patient. To overcome this limitation, we propose PINGS-X, a novel framework modeling high-resolution flow velocities using axes-aligned spatiotemporal Gaussian representations. Inspired by the effectiveness of 3D Gaussian splatting (3DGS) in novel view synthesis, PINGS-X extends this concept through several non-trivial novel innovations: (i) normalized Gaussian splatting with a formal convergence guarantee, (ii) axes-aligned Gaussians that simplify training for high-dimensional data while preserving accuracy and the convergence guarantee, and (iii) a Gaussian merging procedure to prevent degenerate solutions and boost computational efficiency. Experimental results on computational fluid dynamics (CFD) and real 4D flow MRI datasets demonstrate that PINGS-X substantially reduces training time while achieving superior super-resolution accuracy. Our code and datasets are available at https://github.com/SpatialAILab/PINGS-X.

---

## 42. LLM enhanced graph inference for long-term disease progression modelling

**论文链接:** [http://arxiv.org/abs/2511.10890v1](http://arxiv.org/abs/2511.10890v1)

**作者:** Tiantian He, An Zhao, Elinor Thompson, Anna Schroder, Ahmed Abdulaal, Frederik Barkhof, Daniel C. Alexander

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究提出了一种利用大语言模型作为专家指南的新框架，用于理解神经退行性疾病期间脑区生物标志物之间的相互作用，从而更准确地预测疾病进展路径。

### 背景

当前神经退行性疾病研究中的病理生理模型通常描述有毒蛋白等变量如何在基于脑连接的动态系统中时空交互，但现有方法过于简化脑连接的复杂关系，导致对病理传播的预测不准确；同时，纯数据驱动方式学习图形面临可识别性问题。

### 目的

开发一种新方法，能够从不规则采样的纵向患者数据中学习疾病进展，同时构建具有生物约束的脑区间相互作用图结构，提高疾病传播预测的准确性和可解释性。

### 方法

提出一种利用大语言模型(LLMs)作为区域变量相互作用的专家指南的框架，利用LLMs综合多模态关系和纳入多样化疾病驱动机制的能力，同时优化长期疾病轨迹构建和生物约束图结构学习两个目标，并通过阿尔茨海默病队列的tau-PET成像数据进行验证。

### 主要发现

新框架相比传统方法展示出优越的预测准确性和可解释性，同时揭示了超越传统连接度量的额外疾病驱动因素，能够更准确地估计病理传播。

### 结论

将大语言模型整合到神经退行性疾病进展研究中，能够有效解决传统方法在脑连接建模和疾病预测方面的局限性，提供更准确、可解释的疾病进展模型，有助于理解疾病机制并指导临床干预。

### 翻译

理解神经退行性疾病期间脑区生物标志物之间的相互作用对于阐明疾病进展的机制至关重要。例如，阿尔茨海默病的病理生理模型通常描述了有毒蛋白等变量如何在基于脑连接的动态系统中时空交互，通常基于脑连接作为底层生物底物驱动的系统。然而，当前方法通过假设单一模态脑连接组作为疾病传播底物，极大地简化了脑连接之间的复杂关系，这导致对病理传播的不准确预测，特别是在长期进展期间。同时，以纯数据驱动方式学习此类图形的方法由于缺乏适当约束而面临可识别性问题。因此，我们提出了一种新框架，利用大语言模型作为区域变量相互作用的专家指南，增强从不规则采样的纵向患者数据中学习疾病进展。通过利用LLMs综合多模态关系和纳入多样化疾病驱动机制的能力，我们的方法同时优化了1)从个体水平观察构建长期疾病轨迹和2)具有更好可识别性的生物约束图结构，该结构捕捉脑区间的相互作用。我们通过使用阿尔茨海默病队列的tau-PET成像数据估计病理传播来演示这一新方法。与传统方法相比，新框架展示了优越的预测准确性和可解释性，同时揭示了超越传统连接度量的额外疾病驱动因素。


### 论文摘要

Understanding the interactions between biomarkers among brain regions during neurodegenerative disease is essential for unravelling the mechanisms underlying disease progression. For example, pathophysiological models of Alzheimer's Disease (AD) typically describe how variables, such as regional levels of toxic proteins, interact spatiotemporally within a dynamical system driven by an underlying biological substrate, often based on brain connectivity. However, current methods grossly oversimplify the complex relationship between brain connectivity by assuming a single-modality brain connectome as the disease-spreading substrate. This leads to inaccurate predictions of pathology spread, especially during the long-term progression period. Meanhwile, other methods of learning such a graph in a purely data-driven way face the identifiability issue due to lack of proper constraint. We thus present a novel framework that uses Large Language Models (LLMs) as expert guides on the interaction of regional variables to enhance learning of disease progression from irregularly sampled longitudinal patient data. By leveraging LLMs' ability to synthesize multi-modal relationships and incorporate diverse disease-driving mechanisms, our method simultaneously optimizes 1) the construction of long-term disease trajectories from individual-level observations and 2) the biologically-constrained graph structure that captures interactions among brain regions with better identifiability. We demonstrate the new approach by estimating the pathology propagation using tau-PET imaging data from an Alzheimer's disease cohort. The new framework demonstrates superior prediction accuracy and interpretability compared to traditional approaches while revealing additional disease-driving factors beyond conventional connectivity measures.

---

## 43. Scalable data-driven modeling of microstructure evolution by learning local dependency and spatiotemporal translation invariance rules in phase field simulation

**论文链接:** [http://arxiv.org/abs/2511.10171v1](http://arxiv.org/abs/2511.10171v1)

**作者:** Zishuo Lan, Qionghuan Zeng, Weilong Ma, Xiangju Liang, Yue Li, Yu Chen, Yiming Chen, Xiaobing Hu, Junjie Li, Lei Wang, Jing Zhang, Zhijun Wang, Jincheng Wang

**发布时间:** 2025-11-13

### GPT解析

### 总结

该研究展示了一种基于简约卷积神经网络的相场模拟加速方法，仅需小规模训练数据即可实现长期预测和系统扩展，有效解决了传统PF模拟计算成本高和数据驱动方法需要大量训练数据的问题。

### 背景

相场模拟是预测微观结构演变的强大框架，但存在计算成本过高的问题，严重限制了实际应用中的时空尺度。现有数据驱动方法虽然能加速模拟，但需要大量训练数据且存在黑盒性质，引发长期预测可靠性的担忧。

### 目的

展示一个简约的卷积神经网络能在极小数据集上训练，实现从单个小尺度模拟中学习并扩展到更大系统，同时实现可靠的长时期预测，超出训练数据的时间范围。

### 方法

使用简约的卷积神经网络，通过晶粒生长和调幅分解的例子进行验证，采用极小数据集进行训练，并进行有效的感受野分析，验证模型在训练过程中捕获了局部性和时空平移不变性等基本特性。

### 主要发现

CNN模型的成功源于其归纳偏置与相场模拟物理先验的一致性；微观结构演化代表了有限数量局部环境的连续重新分布；当模型在早期训练数据中已经遇到几乎所有可能的局部环境时，它可以可靠地推广到更长的演化时间尺度，无论全球微观结构形态发生多大变化。

### 结论

从还原论角度看，代理模型建立了时空不变的回归映射，连接网格点的局部环境及其后续状态；当模型捕获了几乎所有可能的局部环境时，可以可靠预测长期演化，解决了传统计算成本问题。

### 翻译

相场模拟为预测微观结构演变提供了强大框架，但高昂的计算成本严重限制了实际应用中的时空尺度。虽然数据驱动方法已成为加速相场模拟的有前景途径，但现有方法需要大量演化轨迹的训练数据，且其固有的黑盒性质引发了长期预测可靠性的担忧。本文通过晶粒生长和调幅分解的示例证明，即使使用极小数据集训练，甚至仅基于单个小尺度模拟，一个简约的卷积神经网络也能无缝扩展到更大系统，并提供超出训练数据时间范围的可靠长期预测。本工作的关键洞察在于揭示了基于CNN模型的成功源于其归纳偏置与相场模拟物理先验的一致性，特别是局部性和时空平移不变性。通过有效的感受野分析，我们验证了模型在训练过程中捕获了这些基本特性。因此，从还原论角度看，代理模型本质上建立了网格点局部环境与其后续状态之间的时空不变回归映射。对模型特征空间的进一步分析表明，微观结构演化实际上代表了有限数量局部环境的连续重新分布。当模型在早期训练数据中已经遇到几乎所有可能的局部环境时，它可以可靠地推广到更长的演化时间尺度，无论全球微观结构形态发生何种剧烈变化。


### 论文摘要

Phase-field (PF) simulation provides a powerful framework for predicting microstructural evolution but suffers from prohibitive computational costs that severely limit accessible spatiotemporal scales in practical applications. While data-driven methods have emerged as promising approaches for accelerating PF simulations, existing methods require extensive training data from numerous evolution trajectories, and their inherent black-box nature raises concerns about long-term prediction reliability. This work demonstrates, through examples of grain growth and spinodal decomposition, that a minimalist Convolutional Neural Network (CNN) trained with a remarkably small dataset even from a single small-scale simulation can achieve seamless scalability to larger systems and reliable long-term predictions far beyond the temporal range of the training data. The key insight of this work lies in revealing that the success of CNN-based models stems from the alignment between their inductive biases and the physical priors of phase-field simulations specifically, locality and spatiotemporal translation invariance. Through effective receptive field analysis, we verify that the model captures these essential properties during training. Therefore, from a reductionist perspective, the surrogate model essentially establishes a spatiotemporally invariant regression mapping between a grid point's local environment and its subsequent state. Further analysis of the model's feature space demonstrates that microstructural evolution effectively represents a continuous redistribution of a finite set of local environments. When the model has already encountered nearly all possible local environments in the early-stage training data, it can reliably generalize to much longer evolution timescales, regardless of the dramatic changes in global microstructural morphology.

---

## 44. Multivariate Gaussian Representation Learning for Medical Action Evaluation

**论文链接:** [http://arxiv.org/abs/2511.10060v1](http://arxiv.org/abs/2511.10060v1)

**作者:** Luming Yang, Haoxian Liu, Siqing Li, Alper Yilmaz

**发布时间:** 2025-11-13

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

该研究针对医学视觉中细粒度动作评估的挑战，提出了一个多元高斯编码框架GaussMedAct，并创建了包含6372个专家标注视频的CPREval-6k基准数据集。该方法通过自适应时空表征学习实现了92.1%的Top-1准确率，比基线提高5.9%，同时计算效率更高。

### 背景

医学视觉中的细粒度动作评估面临独特挑战，包括缺乏全面数据集、严格的精度要求以及对极快速动作的时空动态建模不足。

### 目的

支持医学动作评估领域的开发和评估工作，通过引入新的基准数据集和先进方法解决现有挑战。

### 方法

提出GaussMedAct多元高斯编码框架，通过多元高斯表示将联合运动投影到时标多维空间，将动作分解为自适应三维高斯作为标记，并采用混合空间编码策略有效利用骨骼信息。

### 主要发现

在CPREval-6k基准上实现92.1%的Top-1准确率，实时推理，相比ST-GCN基线准确率提高5.9%，而计算量仅为10%。跨数据集实验证实了方法的鲁棒性优势。

### 结论

所提出的方法在医学动作评估中表现出色，为医学视觉领域提供了新的有效解决方案。

### 翻译

医学视觉中的细粒度动作评估面临着独特挑战，这是由于缺乏全面数据集、严格的精度要求以及对极快速动作的时空动态建模不足。为了支持开发和评估，我们引入了CPREval-6k，这是一个多视角、多标签的医学动作基准，包含6372个专家标注的视频和22个临床标签。利用这个数据集，我们提出了GaussMedAct，一个多元高斯编码框架，通过自适应时空表征学习来推进医学运动分析。多元高斯表示将联合运动投影到时标多维空间，并将动作分解为自适应的三维高斯，这些高斯作为标记。这些标记通过各向异性协方差建模保留运动语义，同时保持对时空噪声的鲁棒性。混合空间编码采用笛卡尔和向量双流策略，有效利用关节和骨骼特征的骨骼信息。所提出的方法在基准上实现了92.1%的Top-1准确率，实时推理，相比ST-GCN基线提高了5.9%的准确率，而FLOPs仅为10%。跨数据集实验证实了我们的方法在鲁棒性方面的优越性。


### 论文摘要

Fine-grained action evaluation in medical vision faces unique challenges due to the unavailability of comprehensive datasets, stringent precision requirements, and insufficient spatiotemporal dynamic modeling of very rapid actions. To support development and evaluation, we introduce CPREval-6k, a multi-view, multi-label medical action benchmark containing 6,372 expert-annotated videos with 22 clinical labels. Using this dataset, we present GaussMedAct, a multivariate Gaussian encoding framework, to advance medical motion analysis through adaptive spatiotemporal representation learning. Multivariate Gaussian Representation projects the joint motions to a temporally scaled multi-dimensional space, and decomposes actions into adaptive 3D Gaussians that serve as tokens. These tokens preserve motion semantics through anisotropic covariance modeling while maintaining robustness to spatiotemporal noise. Hybrid Spatial Encoding, employing a Cartesian and Vector dual-stream strategy, effectively utilizes skeletal information in the form of joint and bone features. The proposed method achieves 92.1% Top-1 accuracy with real-time inference on the benchmark, outperforming the ST-GCN baseline by +5.9% accuracy with only 10% FLOPs. Cross-dataset experiments confirm the superiority of our method in robustness.

---

## 45. EEGAgent: A Unified Framework for Automated EEG Analysis Using Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.09947v1](http://arxiv.org/abs/2511.09947v1)

**作者:** Sha Zhao, Mingyi Peng, Haiteng Jiang, Tao Li, Shijian Li, Gang Pan

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文介绍了EEGAgent，一个利用大语言模型调度多种工具以自动完成脑电图相关任务的全能框架。

### 背景

可扩展和可推广的脑活动分析对临床诊断和认知研究至关重要。脑电图作为一种具有高时间分辨率的非侵入性技术已被广泛应用，但现有模型通常针对特定任务定制，限制了其在多任务和连续推理场景中的实用性。

### 目的

开发一个通用框架，能够处理多种EEG任务并支持连续推理，以克服现有EEG模型的局限性。

### 方法

设计EEGAgent框架，包含EEG基本信息感知、时空探索、事件检测、用户交互和报告生成功能。构建由预处理、特征提取、事件检测等工具组成的工具箱，并在公共数据集上进行评估。

### 主要发现

EEGAgent能够支持灵活且可解释的EEG分析，展现了在现实世界临床应用中的潜力。

### 结论

EEGAgent作为一个通用EEG分析框架，能够有效处理多种任务并支持连续推理，为脑电图分析提供了新的可能性。

### 翻译

可扩展和可推广的脑活动分析对于推进临床诊断和认知研究都至关重要。脑电图(EEG)是一种具有高时间分辨率的非侵入性技术，已被广泛用于脑状态分析。然而，大多数现有的EEG模型通常针对特定任务定制，限制了它们在涉及多任务和连续推理的现实场景中的实用性。在这项工作中，我们引入了EEGAgent，一个利用大语言模型(LLMs)来调度和规划多个工具以自动完成EEG相关任务的全能框架。EEGAgent能够执行关键功能：EEG基本信息感知、时空EEG探索、EEG事件检测、与用户交互、EEG报告生成。为实现这些能力，我们设计了一个由不同工具组成的工具箱，包括EEG预处理、特征提取、事件检测等。这些能力在公共数据集上进行了评估，我们的EEGAgent能够支持灵活且可解释的EEG分析，突显了其在现实世界临床应用中的潜力。


### 论文摘要

Scalable and generalizable analysis of brain activity is essential for advancing both clinical diagnostics and cognitive research. Electroencephalography (EEG), a non-invasive modality with high temporal resolution, has been widely used for brain states analysis. However, most existing EEG models are usually tailored for individual specific tasks, limiting their utility in realistic scenarios where EEG analysis often involves multi-task and continuous reasoning. In this work, we introduce EEGAgent, a general-purpose framework that leverages large language models (LLMs) to schedule and plan multiple tools to automatically complete EEG-related tasks. EEGAgent is capable of performing the key functions: EEG basic information perception, spatiotemporal EEG exploration, EEG event detection, interaction with users, and EEG report generation. To realize these capabilities, we design a toolbox composed of different tools for EEG preprocessing, feature extraction, event detection, etc. These capabilities were evaluated on public datasets, and our EEGAgent can support flexible and interpretable EEG analysis, highlighting its potential for real-world clinical applications.

---

## 46. FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching

**论文链接:** [http://arxiv.org/abs/2511.09731v1](http://arxiv.org/abs/2511.09731v1)

**作者:** Bernardo Perrone Ribeiro, Jana Faganeli Pucer

**发布时间:** 2025-11-12

**备注:** Under Review

### GPT解析

### 总结

该研究引入了FlowCast，首个应用条件流匹配(CFM)于降水临近预报的模型，解决了扩散模型计算效率低的问题，实现了快速、高保真的降水预测，并建立了新的预测准确性最先进水平。

### 背景

基于雷达的降水临近预报对洪水风险管理和决策至关重要。尽管深度学习已推动该领域发展，但仍面临两个基本挑战：大气动力学的不确定性以及高维数据的有效建模。

### 目的

开发一种计算效率高且准确的降水临近预报方法，以克服现有扩散模型迭代采样过程计算成本过高的问题，满足时间关键应用的需求。

### 方法

研究团队引入了FlowCast，这是一种应用条件流匹配(CFM)于降水临近预报的模型。与扩散模型不同，CFM学习直接从噪声到数据的映射，实现快速、高保真样本生成，且函数评估次数大幅减少。

### 主要发现

FlowCast在预测准确性上建立了新的最先进水平；直接比较表明，CFM目标比相同架构上的扩散目标更准确且更高效；CFM能用显著更少的采样步骤保持高性能。

### 结论

该研究将CFM定位为高维时空预测的有力且实用的替代方案，为降水临近预报领域提供了新的高效解决方案。

### 翻译

基于雷达的降水临近预报，即从先前雷达图像预报短期降水场的任务，是洪水风险管理和决策中的关键问题。虽然深度学习已显著推动了该领域的发展，但两个基本挑战仍然存在：大气动力学的不确定性以及高维数据的有效建模。扩散模型已显示出强大潜力，能够产生清晰可靠的预测，但其迭代采样过程对时间关键应用来说计算成本过高。我们引入了FlowCast，这是首个将条件流匹配(CFM)应用于降水临近预报的模型。与扩散模型不同，CFM学习直接从噪声到数据的映射，能够以少得多的函数评估实现快速、高保真的样本生成。我们的实验证明，FlowCast在预测准确性上建立了新的最先进水平。直接比较进一步揭示，在相同架构上，CFM目标比扩散目标更准确且效率更高，能用显著更少的采样步骤保持高性能。这项研究将CFM定位为高维时空预测的有力且实用的替代方案。


### 论文摘要

Radar-based precipitation nowcasting, the task of forecasting short-term precipitation fields from previous radar images, is a critical problem for flood risk management and decision-making. While deep learning has substantially advanced this field, two challenges remain fundamental: the uncertainty of atmospheric dynamics and the efficient modeling of high-dimensional data. Diffusion models have shown strong promise by producing sharp, reliable forecasts, but their iterative sampling process is computationally prohibitive for time-critical applications. We introduce FlowCast, the first model to apply Conditional Flow Matching (CFM) to precipitation nowcasting. Unlike diffusion, CFM learns a direct noise-to-data mapping, enabling rapid, high-fidelity sample generation with drastically fewer function evaluations. Our experiments demonstrate that FlowCast establishes a new state-of-the-art in predictive accuracy. A direct comparison further reveals the CFM objective is both more accurate and significantly more efficient than a diffusion objective on the same architecture, maintaining high performance with significantly fewer sampling steps. This work positions CFM as a powerful and practical alternative for high-dimensional spatiotemporal forecasting.

---

## 47. Spatio-Temporal Data Enhanced Vision-Language Model for Traffic Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.08978v1](http://arxiv.org/abs/2511.08978v1)

**作者:** Jingtian Ma, Jingyuan Wang, Wayne Xin Zhao, Guoping Liu, Xiang Wen

**发布时间:** 2025-11-12

### GPT解析

### 总结

本研究提出了一种新型时空增强模型(ST-CLIP)，用于交通场景理解(TSU)任务，整合了时空信息与视觉文本数据，解决了传统方法忽略时空信息和交通场景不同方面相互关系的问题。

### 背景

导航和拼车应用已收集大量带有时空数据的图像，分析这类图像的核心技术是交通场景理解(TSU)，旨在提供对交通场景的全面描述。

### 目的

解决当前TSU研究忽略时空信息和交通场景不同方面相互关系的问题，提出一种能整合时空信息的模型。

### 方法

基于经典视觉语言模型CLIP设计ST-CLIP模型，采用时空上下文感知的多方面提示(SCAMP)学习方法，包含动态时空上下文表示模块和双层ST感知多方面提示学习模块，将时空信息整合到TSU中并利用交通场景不同方面的交互关系。

### 主要发现

首次尝试将时空信息整合到视觉语言模型中以促进TSU任务，实验证明在少样本学习策略下，该模型在复杂场景理解场景中表现出优越性能。

### 结论

ST-CLIP模型有效解决了传统TSU方法忽略时空信息和场景不同方面相互关系的问题，在真实世界数据集上验证了其有效性。

### 翻译

如今，导航和拼车应用已收集了大量带有时空数据的图像。分析这类与时空信息相关图像的核心技术是交通场景理解(TSU)，旨在提供对交通场景的全面描述。与传统时空数据分析任务不同，TSU任务对时空数据和视觉文本数据的双重依赖带来了独特的挑战。然而，最近的研究常将TSU视为普通图像理解任务，忽略了时空信息和交通场景不同方面的相互关系。为解决这些问题，我们提出了一种基于CILP的新型时空增强模型(ST-CLIP)用于TSU。我们的模型使用经典视觉语言模型CLIP作为骨干网络，并设计了时空上下文感知的多方面提示(SCAMP)学习方法，将时空信息整合到TSU中。提示学习方法包含两个组件：动态时空上下文表示模块，为每个交通场景图像提取时空数据的表示向量；以及双层ST感知多方面提示学习模块，将ST上下文表示向量整合到CLIP模型的提示词嵌入中。第二个模块还提取低级视觉特征和图像级别的高级语义特征，以利用交通场景不同方面之间的交互关系。据我们所知，这是首次尝试将时空信息整合到视觉语言模型中以促进TSU任务。在两个真实世界数据集上的实验表明，在少样本学习策略下，该模型在复杂场景理解场景中表现出优越性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决交通场景理解(TSU)问题，具体是现有方法忽略了时空信息的重要性且过于关注低级视觉特征而缺乏高级语义特征建模。这个问题在现实中非常重要，因为导航和共享出行应用已收集大量带有时空数据的图像，而准确的交通场景理解对交通流预测、自动驾驶和路线推荐等智能交通应用至关重要。现有方法无法充分利用时空信息和高级语义特征，限制了理解复杂交通场景的能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了交通场景理解的特殊性和挑战性，评估了早期基于深度学习的视觉方法和近年来预训练的视觉-语言模型的优缺点。基于分析，作者决定利用CLIP作为基础模型，通过提示学习整合时空信息，并设计双层注意力机制解决高级语义建模不足的问题。该方法借鉴了CLIP的跨模态能力、提示学习思想、图神经网络和循环神经网络处理序列数据的技术，以及注意力机制在特征提取和关系建模中的应用，但针对交通场景理解特点进行了专门设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过时空上下文感知的多方面提示学习(SCAMP)，将时空信息整合到预训练的视觉-语言模型中，实现全面准确的多方面交通场景理解。整体流程包括：1)动态时空上下文表示，将GPS轨迹转换为基于路段的轨迹并编码静态和时间变化属性；2)可学习的ST感知多方面提示，为每个方面定义可学习嵌入矩阵并基于时空上下文生成自适应提示；3)双层多方面提示注意力，包括补丁级跨模态注意力和图像级跨方面注意力；4)模型训练和描述生成，冻结CLIP参数微调SCAMP模块，使用模板生成最终描述。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)首次将时空信息整合到预训练多模态模型中；2)动态时空上下文表示方法；3)双层多方面提示注意力机制；4)'预训练模型+时空数据'框架。相比之前工作，本文的不同在于：明确整合了时空信息而非仅关注静态空间线索；设计了双层注意力机制联合建模低级视觉特征和高级语义特征；针对交通场景特点专门设计了时空感知的多方面提示；在少样本条件下也能实现有效理解；能生成全面的交通场景文本描述而非仅限于分类或检测任务。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于时空数据增强的视觉-语言模型，通过时空上下文感知的多方面提示学习，实现了对复杂交通场景的全面理解和准确描述，显著提升了在少样本学习条件下的性能。'}


### 论文摘要

Nowadays, navigation and ride-sharing apps have collected numerous images with spatio-temporal data. A core technology for analyzing such images, associated with spatiotemporal information, is Traffic Scene Understanding (TSU), which aims to provide a comprehensive description of the traffic scene. Unlike traditional spatio-temporal data analysis tasks, the dependence on both spatio-temporal and visual-textual data introduces distinct challenges to TSU task. However, recent research often treats TSU as a common image understanding task, ignoring the spatio-temporal information and overlooking the interrelations between different aspects of the traffic scene. To address these issues, we propose a novel SpatioTemporal Enhanced Model based on CILP (ST-CLIP) for TSU. Our model uses the classic vision-language model, CLIP, as the backbone, and designs a Spatio-temporal Context Aware Multiaspect Prompt (SCAMP) learning method to incorporate spatiotemporal information into TSU. The prompt learning method consists of two components: A dynamic spatio-temporal context representation module that extracts representation vectors of spatio-temporal data for each traffic scene image, and a bi-level ST-aware multi-aspect prompt learning module that integrates the ST-context representation vectors into word embeddings of prompts for the CLIP model. The second module also extracts low-level visual features and image-wise high-level semantic features to exploit interactive relations among different aspects of traffic scenes. To the best of our knowledge, this is the first attempt to integrate spatio-temporal information into visionlanguage models to facilitate TSU task. Experiments on two realworld datasets demonstrate superior performance in the complex scene understanding scenarios with a few-shot learning strategy.

---

## 48. Weaver: Kronecker Product Approximations of Spatiotemporal Attention for Traffic Network Forecasting

**论文链接:** [http://arxiv.org/abs/2511.08888v1](http://arxiv.org/abs/2511.08888v1)

**作者:** Christopher Cheong, Gary Davis, Seongjin Choi

**发布时间:** 2025-11-12

### GPT解析

### 总结

Weaver是一种新的基于注意力的时空预测模型，通过Kronecker乘积近似和价态注意力技术，实现了高效且准确的交通网络预测。

### 背景

交通网络上的时空预测需要理解交通节点在动态演变系统中的相互作用，这对现代交通和商业至关重要。现有Transformer架构虽提高预测性能但计算开销大且可解释性低。

### 目的

开发一种既准确又可解释、高效且稳健的交通网络预测模型，解决现有方法的高计算开销和可解释性问题。

### 方法

1. 应用Kronecker乘积近似分解时空注意力；2. 实现并行Kronecker矩阵向量积进行高效时空消息传递；3. 引入价态注意力处理负边在交通行为建模中的重要性；4. 使用交通相位字典进行自调节。

### 主要发现

Weaver在PEMS-BAY和METR-LA数据集上实现了与现有模型相当的性能，同时训练效率更高。

### 结论

Weaver通过创新的注意力机制和计算优化，成功平衡了预测准确性、计算效率和模型可解释性，为交通网络预测提供了有效解决方案。

### 翻译

在交通网络上的时空预测是一项复杂任务，需要理解交通节点在由交通流动态和社会行为模式决定的动态演变系统中如何相互作用。交通网络和智能交通系统对现代交通和商业的重要性，需要预测模型不仅准确，还要可解释、高效和稳健。最近的方法，特别是基于Transformer的架构，提高了预测性能，但往往以高计算开销和降低架构可解释性为代价。在这项工作中，我们引入了Weaver，一种新的基于注意力的模型，应用Kronecker乘积近似将时空注意力的复杂度分解为局部时间和空间注意图。这种Kronecker注意图使得我们能够实现并行矩阵向量积，以实现高效的时空消息传递。为了捕捉真实的交通动态，我们通过引入价态注意力来处理负边在建模交通行为中的重要性，这提供了精确的潜在图生成和训练稳定性所需的特性。为了充分利用模型的学习能力，我们引入了交通相位字典进行自调节。在PEMS-BAY和METR-LA上的评估显示，Weaver在各类模型中实现了具有竞争力的性能，同时训练效率更高。


### 论文摘要

Spatiotemporal forecasting on transportation networks is a complex task that requires understanding how traffic nodes interact within a dynamic, evolving system dictated by traffic flow dynamics and social behavioral patterns. The importance of transportation networks and ITS for modern mobility and commerce necessitates forecasting models that are not only accurate but also interpretable, efficient, and robust under structural or temporal perturbations. Recent approaches, particularly Transformer-based architectures, have improved predictive performance but often at the cost of high computational overhead and diminished architectural interpretability. In this work, we introduce Weaver, a novel attention-based model that applies Kronecker product approximations (KPA) to decompose the PN X PN spatiotemporal attention of O(P^2N^2) complexity into local P X P temporal and N X N spatial attention maps. This Kronecker attention map enables our Parallel-Kronecker Matrix-Vector product (P2-KMV) for efficient spatiotemporal message passing with O(P^2N + N^2P) complexity. To capture real-world traffic dynamics, we address the importance of negative edges in modeling traffic behavior by introducing Valence Attention using the continuous Tanimoto coefficient (CTC), which provides properties conducive to precise latent graph generation and training stability. To fully utilize the model's learning capacity, we introduce the Traffic Phase Dictionary for self-conditioning. Evaluations on PEMS-BAY and METR-LA show that Weaver achieves competitive performance across model categories while training more efficiently.

---

## 49. ForeSWE: Forecasting Snow-Water Equivalent with an Uncertainty-Aware Attention Model

**论文链接:** [http://arxiv.org/abs/2511.08856v1](http://arxiv.org/abs/2511.08856v1)

**作者:** Krishu K Thapa, Supriya Savalkar, Bhupinderjeet Singh, Trong Nghia Hoang, Kirti Rajagopalan, Ananth Kalyanaraman

**发布时间:** 2025-11-12

**备注:** Accepted for publication at the 2026 AAAI conference

### GPT解析

### 总结

本文提出了一种名为ForeSWE的新型概率时空预测模型，用于雪当量(SWE)的预测，该模型结合了深度学习和经典概率技术，提高了预测精度并提供了不确定性估计。

### 背景

在以降雪为主的流域中，水资源管理决策依赖于雪当量(SWE)这一关键指标。然而，SWE预测具有挑战性，因为它受地形和环境条件等多种因素影响，表现出时空变异性，且传统方法未能充分利用时空相关性，也不提供不确定性估计。

### 目的

开发一种能够提供不确定性估计的SWE预测模型，提高预测精度和预测区间质量。

### 方法

提出ForeSWE模型，结合了深度学习和经典概率技术，包含注意力机制以整合时空特征和相互作用，以及高斯过程模块提供预测不确定性的原则性量化。在美国西部512个雪情遥测(SNOTEL)站点的数据上进行了评估。

### 主要发现

与现有方法相比，ForeSWE在预测精度和预测区间方面都有显著改进，并验证了不同方法间不确定性估计的有效性。

### 结论

这些发现为水资源管理社区的部署和反馈提供了平台。

### 翻译

在各种以降雪为主的流域中，水资源管理决策依赖于雪当量(SWE)的知识——这是一种广泛用于估算积雪层水分含量的关键指标。然而，预测SWE具有挑战性，因为它受多种因素影响，包括地形和各种环境条件，因此被观察到具有时空变异性。传统的SWE预测方法未能充分利用这些时空相关性，也不提供不确定性估计——这对决策者可能具有重要价值。在本文中，我们提出了ForeSWE，一种新的概率时空预测模型，它结合了深度学习和经典概率技术。该模型结合了注意力机制以整合时空特征和相互作用，以及提供预测不确定性原则性量化高斯过程模块。我们在美国西部512个雪情遥测(SNOTEL)站点的数据上评估了该模型。结果显示，与现有方法相比，预测精度和预测区间都有显著改进。结果还突显了不同方法间不确定性估计的有效性。总体而言，这些发现为水资源管理社区的部署和反馈提供了平台。


### 论文摘要

Various complex water management decisions are made in snow-dominant watersheds with the knowledge of Snow-Water Equivalent (SWE) -- a key measure widely used to estimate the water content of a snowpack. However, forecasting SWE is challenging because SWE is influenced by various factors including topography and an array of environmental conditions, and has therefore been observed to be spatio-temporally variable. Classical approaches to SWE forecasting have not adequately utilized these spatial/temporal correlations, nor do they provide uncertainty estimates -- which can be of significant value to the decision maker. In this paper, we present ForeSWE, a new probabilistic spatio-temporal forecasting model that integrates deep learning and classical probabilistic techniques. The resulting model features a combination of an attention mechanism to integrate spatiotemporal features and interactions, alongside a Gaussian process module that provides principled quantification of prediction uncertainty. We evaluate the model on data from 512 Snow Telemetry (SNOTEL) stations in the Western US. The results show significant improvements in both forecasting accuracy and prediction interval compared to existing approaches. The results also serve to highlight the efficacy in uncertainty estimates between different approaches. Collectively, these findings have provided a platform for deployment and feedback by the water management community.

---

## 50. Data-driven spatiotemporal modeling reveals personalized trajectories of cortical atrophy in Alzheimer's disease

**论文链接:** [http://arxiv.org/abs/2511.08847v1](http://arxiv.org/abs/2511.08847v1)

**作者:** Chunyan Li, Yutong Mao, Xiao Liu, Wenrui Hao

**发布时间:** 2025-11-12

**备注:** 23 pages, 7 figures and 3 tables

### GPT解析

### 总结

该研究提出了一种个性化基于图的动态模型，能够从纵向MRI和PET数据中捕捉皮质萎缩的时空演变，准确预测AD生物标志物和认知功能下降，揭示疾病进展亚型，并确定疾病扩散的关键区域。

### 背景

阿尔茨海默病的特点是病理学在脑网络中的渐进性扩散，但在个体层面预测这种级联反应仍然具有挑战性。

### 目的

开发一个个性化的基于图的动态模型，用于预测阿尔茨海默病的进展轨迹。

### 方法

构建个性化脑网络图，从纵向MRI和PET数据中学习驱动区域神经变性的动力学，应用于1,891名ADNI参与者，并进行敏感性分析确定疾病扩散的区域驱动因素。

### 主要发现

模型准确预测关键的AD生物标志物，性能优于临床和神经影像学基准；患者特异性参数揭示了不同的进展亚型；比标准生物标志物更有效地预测未来的认知能力下降；敏感性分析重现了已知的颞叶边缘和额叶易感性模式。

### 结论

这种基于网络的数字孪生框架为AD轨迹预测提供了量化的、个性化范式，对患者分层、临床试验设计和针对性治疗开发具有重要意义。

### 翻译

阿尔茨海默病的特点是病理学在脑网络中的渐进性扩散，但在个体层面预测这种级联反应仍然具有挑战性。我们提出了一种个性化的基于图的动态模型，该模型能够从纵向MRI和PET数据中捕捉皮质萎缩的时空演变。该方法构建个性化的脑网络图并学习驱动区域神经变性的动力学。将其应用于阿尔茨海默病神经影像计划的1,891名参与者，该模型准确预测了关键的AD生物标志物——包括淀粉样蛋白-β、tau蛋白、神经变性和认知功能——性能优于临床和神经影像学基准。患者特异性参数揭示了不同的进展亚型，并且比标准生物标志物更有效地预测未来的认知能力下降。敏感性分析突出了疾病扩散的区域驱动因素，重现了已知的颞叶边缘和额叶易感性模式。这种基于网络的数字孪生框架为AD轨迹预测提供了量化的、个性化范式，对患者分层、临床试验设计和针对性治疗开发具有重要意义。


### 论文摘要

Alzheimer's disease (AD) is characterized by the progressive spread of pathology across brain networks, yet forecasting this cascade at the individual level remains challenging. We present a personalized graph-based dynamical model that captures the spatiotemporal evolution of cortical atrophy from longitudinal MRI and PET data. The approach constructs individualized brain graphs and learns the dynamics driving regional neurodegeneration. Applied to 1,891 participants from the Alzheimer's Disease Neuroimaging Initiative, the model accurately predicts key AD biomarkers -- including amyloid-beta, tau, neurodegeneration, and cognition -- outperforming clinical and neuroimaging benchmarks. Patient-specific parameters reveal distinct progression subtypes and anticipate future cognitive decline more effectively than standard biomarkers. Sensitivity analysis highlights regional drivers of disease spread, reproducing known temporolimbic and frontal vulnerability patterns. This network-based digital twin framework offers a quantitative, personalized paradigm for AD trajectory prediction, with implications for patient stratification, clinical trial design, and targeted therapeutic development.

---

## 51. SRNN: Spatiotemporal Relational Neural Network for Intuitive Physics Understanding

**论文链接:** [http://arxiv.org/abs/2511.06761v1](http://arxiv.org/abs/2511.06761v1)

**作者:** Fei Yang

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文提出时空关系神经网络(SRNN)，通过大脑启发的计算原理来缩小人类与机器在直觉物理理解方面的差距。

### 背景

人类在直觉物理方面的能力仍然无法被机器匹敌。

### 目的

为了缩小这一差距，作者提出向大脑启发的计算原理转变，开发一种能够模拟人类直觉物理理解能力的模型。

### 方法

引入时空关系神经网络(SRNN)，建立对象属性、关系和时间线的统一神经表示，通过专门的'What'和'How'通路上的赫布式'一起激活，一起连接'机制控制计算，直接生成视觉场景的结构化语言描述，采用'预定义然后微调'而非主流的'预训练然后微调'范式。

### 主要发现

在CLEVRER基准测试上，SRNN取得了具有竞争力的性能；分析揭示了基准偏差，提出了更全面评估的路径，展示了SRNN在精确错误诊断方面的白盒效用。

### 结论

将生物智能转化为工程系统以理解直觉物理是可行的。

### 翻译

人类在直觉物理方面的能力仍然无法被机器匹敌。为了缩小这一差距，我们主张向大脑启发的计算原理进行根本性转变。本文介绍了时空关系神经网络(SRNN)，该模型建立了对象属性、关系和时间线的统一神经表示，计算由专门的'What'和'How'通路上的赫布式'一起激活，一起连接'机制控制。这种统一表示直接用于生成视觉场景的结构化语言描述，在共享神经基质中连接感知和语言。此外，与主流的'预训练然后微调'范式不同，SRNN采用'预定义然后微调'的方法。在CLEVRER基准测试上，SRNN取得了具有竞争力的性能。我们的分析进一步揭示了基准偏差，概述了更全面评估的路径，并展示了SRNN的白盒效用，可用于精确错误诊断。我们的研究证实了将生物智能转化为工程系统以理解直觉物理的可行性。


### 论文摘要

Human prowess in intuitive physics remains unmatched by machines. To bridge this gap, we argue for a fundamental shift towards brain-inspired computational principles. This paper introduces the Spatiotemporal Relational Neural Network (SRNN), a model that establishes a unified neural representation for object attributes, relations, and timeline, with computations governed by a Hebbian ``Fire Together, Wire Together'' mechanism across dedicated \textit{What} and \textit{How} pathways. This unified representation is directly used to generate structured linguistic descriptions of the visual scene, bridging perception and language within a shared neural substrate. Moreover, unlike the prevalent ``pretrain-then-finetune'' paradigm, SRNN adopts a ``predefine-then-finetune'' approach. On the CLEVRER benchmark, SRNN achieves competitive performance. Our analysis further reveals a benchmark bias, outlines a path for a more holistic evaluation, and demonstrates SRNN's white-box utility for precise error diagnosis. Our work confirms the viability of translating biological intelligence into engineered systems for intuitive physics understanding.

---

## 52. On the Potential of Digital Twins for Distribution System State Estimation with Randomly Missing Data in Heterogeneous Measurements

**论文链接:** [http://arxiv.org/abs/2511.06583v1](http://arxiv.org/abs/2511.06583v1)

**作者:** Ying Zhang, Yihao Wang, Yuanshuo Zhang, Eric Larson, Di Shi, Fanping Sui

**发布时间:** 2025-11-10

**备注:** Accepted by the 2025 IEEE Power and Energy Society General Meeting

### GPT解析

### 总结

这篇论文提出了一种基于交互注意力的DSSE模型，整合物理实体、虚拟建模和数据融合三个核心组件，通过物理信息增强、数据传输、时空特征学习和交叉交互特征融合等方法，实现了在数据缺失情况下的稳健电网监测。

### 背景

传统的基于统计优化的状态估计(DSSE)算法依赖于详细的电网参数和所有可能不确定性的数学假设。由于通信故障、拥塞和网络攻击导致的数据随机缺失，使得这些方法难以实施。

### 目的

受到数字孪生(DT)最新进展的启发，本文提出了一种基于交互注意力的DSSE模型，用于通过整合三个核心组件（物理实体、虚拟建模和数据融合）来实现稳健的电网监测。

### 方法

为了使方法能够抵抗异构测量中的各种数据缺失，首先提出了物理信息增强和数据传输。此外，提出了一种最先进的基于注意力的时空特征学习，随后是一种新颖的交叉交互特征融合方法，用于稳健的电压估计。

### 主要发现

在一个真实的84节点不平衡配电系统案例研究中，使用原始数据验证了所提出的DT模型在估计电压状态方面的准确性和鲁棒性，能够处理随机位置、任意比例（高达总测量的40%）的数据缺失。

### 结论

该数字孪生模型能够有效处理数据缺失情况，提供准确的电压状态估计，具有很好的鲁棒性。

### 翻译

传统的基于统计优化的状态估计(DSSE)算法依赖于详细的电网参数和所有可能不确定性的数学假设。此外，由于通信故障、拥塞和网络攻击导致的数据随机缺失，使得这些方法难以实施。受到数字孪生(DT)最新进展的启发，本文提出了一种基于交互注意力的DSSE模型，用于通过整合三个核心组件（物理实体、虚拟建模和数据融合）来实现稳健的电网监测。为了使方法能够抵抗异构测量中的各种数据缺失，首先提出了物理信息增强和数据传输。此外，提出了一种最先进的基于注意力的时空特征学习，随后是一种新颖的交叉交互特征融合方法，用于稳健的电压估计。在一个真实的84节点不平衡配电系统案例研究中，使用原始数据验证了所提出的DT模型在估计电压状态方面的准确性和鲁棒性，能够处理随机位置、任意比例（高达总测量的40%）的数据缺失。


### 论文摘要

Traditional statistical optimization-based state estimation (DSSE) algorithms rely on detailed grid parameters and mathematical assumptions of all possible uncertainties. Furthermore, random data missing due to communication failures, congestion, and cyberattacks, makes these methods easily infeasible. Inspired by recent advances in digital twins (DTs), this paper proposes an interactive attention-based DSSE model for robust grid monitoring by integrating three core components: physical entities, virtual modeling, and data fusion. To enable robustness against various data missing in heterogeneous measurements, we first propose physics-informed data augmentation and transfer. Moreover, a state-of-the-art attention-based spatiotemporal feature learning is proposed, followed by a novel cross-interaction feature fusion for robust voltage estimation. A case study in a real-world unbalanced 84-bus distribution system with raw data validates the accuracy and robustness of the proposed DT model in estimating voltage states, with random locational, arbitrary ratios (up to 40% of total measurements) of data missing.

---

## 53. Modulo Video Recovery via Selective Spatiotemporal Vision Transformer

**论文链接:** [http://arxiv.org/abs/2511.07479v1](http://arxiv.org/abs/2511.07479v1)

**作者:** Tianyu Geng, Feng Ji, Wee Peng Tay

**发布时间:** 2025-11-09

### GPT解析

### 总结

本文提出了一种名为选择性时空视觉Transformer(SSViT)的深度学习框架，用于模态视频重建，解决了传统图像传感器在高动态范围场景中的饱和问题。

### 背景

传统图像传感器动态范围有限，在高动态范围场景中会导致饱和。模态相机通过折叠入射辐射度到有限范围内解决这个问题，但需要专门的解包算法。与HDR恢复不同，模态恢复是从折叠样本中恢复实际值，且进展缓慢，特别是在现代深度学习应用方面。

### 目的

开发适用于模态视频恢复的深度学习框架，解决模态图像恢复进展缓慢的问题。

### 方法

研究发现标准HDR方法不适合模态恢复，而Transformer能捕捉全局依赖和时空关系。提出SSViT框架，采用令牌选择策略提高效率并专注于关键区域。

### 主要发现

SSViT能够从8位折叠视频中产生高质量重建，并在模态视频恢复方面取得了最先进的性能。

### 结论

SSViT是模态视频恢复的有效解决方案，深度学习技术特别是Transformer架构在模态恢复中具有潜力。

### 翻译

传统图像传感器的动态范围有限，导致在高动态范围(HDR)场景中出现饱和。模态相机通过将入射辐射度折叠到有限范围内来解决这个问题，但需要专门的解包算法来重建底层信号。与从常规采样扩展动态范围的HDR恢复不同，模态恢复是从折叠样本中恢复实际值。尽管模态图像恢复在十多年前就已提出，但其进展缓慢，特别是在使用现代深度学习技术方面。在本工作中，我们证明了标准HDR方法不适合模态恢复。然而，Transformer可以捕捉解决折叠视频帧所必需的全局依赖和时空关系。尽管如此，将现有的Transformer架构适应模态恢复需要新技术。为此，我们提出了选择性时空视觉Transformer(SSViT)，这是第一个用于模态视频重建的深度学习框架。SSViT采用令牌选择策略来提高效率并专注于最关键区域。实验证实，SSViT能够从8位折叠视频中产生高质量重建，并在模态视频恢复方面取得了最先进的性能。


### 论文摘要

Conventional image sensors have limited dynamic range, causing saturation in high-dynamic-range (HDR) scenes. Modulo cameras address this by folding incident irradiance into a bounded range, yet require specialized unwrapping algorithms to reconstruct the underlying signal. Unlike HDR recovery, which extends dynamic range from conventional sampling, modulo recovery restores actual values from folded samples. Despite being introduced over a decade ago, progress in modulo image recovery has been slow, especially in the use of modern deep learning techniques. In this work, we demonstrate that standard HDR methods are unsuitable for modulo recovery. Transformers, however, can capture global dependencies and spatial-temporal relationships crucial for resolving folded video frames. Still, adapting existing Transformer architectures for modulo recovery demands novel techniques. To this end, we present Selective Spatiotemporal Vision Transformer (SSViT), the first deep learning framework for modulo video reconstruction. SSViT employs a token selection strategy to improve efficiency and concentrate on the most critical regions. Experiments confirm that SSViT produces high-quality reconstructions from 8-bit folded videos and achieves state-of-the-art performance in modulo video recovery.

---

## 54. MiVID: Multi-Strategic Self-Supervision for Video Frame Interpolation using Diffusion Model

**论文链接:** [http://arxiv.org/abs/2511.06019v1](http://arxiv.org/abs/2511.06019v1)

**作者:** Priyansh Srivastava, Romit Chatterjee, Abir Sen, Aradhana Behura, Ratnakar Dash

**发布时间:** 2025-11-08

### GPT解析

### 总结

MiVID是一种轻量级、自监督的基于扩散框架的视频帧插值方法，通过结合3D U-Net和transformer风格的时间注意力，在无需显式运动估计的情况下实现了高效的视频插值。

### 背景

视频帧插值(VFI)是视频增强的基础技术，用于慢动作渲染、帧率转换和视频恢复等任务。传统光流方法和需要密集真实标注的基于学习的方法都面临遮挡、域变化和模糊运动等挑战。

### 目的

开发一种无需高帧率监督的视频插值框架，解决现有方法在处理遮挡和不确定运动时的问题。

### 方法

结合3D U-Net骨干网络与transformer风格的时间注意力，在混合掩码机制下训练，模拟遮挡和运动不确定性；使用基于余弦的渐进掩码和自适应损失调度；在UCF101-7和DAVIS-7数据集上评估；仅使用CPU和9帧视频段进行训练。

### 主要发现

MiVID在仅50个epoch的训练后取得了与多个监督基线相竞争的结果；自监督扩散先验在时间一致帧合成方面表现出强大能力；该框架是低资源但高效的解决方案。

### 结论

自监督扩散先验在时间一致帧合成方面具有强大能力，为开发可访问和通用化的视频帧插值系统提供了可扩展的路径。

### 翻译

视频帧插值(VFI)仍然是视频增强的基石，能够为慢动作渲染、帧率转换和视频恢复等任务提供时间上采样。虽然传统方法依赖于光流，而基于学习的方法假设可以获得密集的真实标注数据，但两者都面临着遮挡、域变化和模糊运动等问题。本文介绍了MiVID，一个轻量级、自监督的基于扩散框架的视频插值方法。我们的模型通过结合3D U-Net骨干网络和transformer风格的时间注意力，消除了对显式运动估计的需求，在模拟遮挡和运动不确定性的混合掩码机制下进行训练。基于余弦的渐进掩码和自适应损失调度的使用，使我们的网络能够在没有任何高帧率监督的情况下学习鲁棒的空间时间表示。我们的框架在UCF101-7和DAVIS-7数据集上进行了评估。MiVID完全使用CPU和这些数据集以及9帧视频段进行训练，使其成为一个低资源但高效的流程。尽管存在这些限制，我们的模型在仅50个epoch时就取得了最佳结果，与几个监督基线具有竞争力。这项工作证明了自监督扩散先验在时间一致帧合成方面的强大能力，并为通用的VFI系统提供了可扩展的发展路径。


### 论文摘要

Video Frame Interpolation (VFI) remains a cornerstone in video enhancement, enabling temporal upscaling for tasks like slow-motion rendering, frame rate conversion, and video restoration. While classical methods rely on optical flow and learning-based models assume access to dense ground-truth, both struggle with occlusions, domain shifts, and ambiguous motion. This article introduces MiVID, a lightweight, self-supervised, diffusion-based framework for video interpolation. Our model eliminates the need for explicit motion estimation by combining a 3D U-Net backbone with transformer-style temporal attention, trained under a hybrid masking regime that simulates occlusions and motion uncertainty. The use of cosine-based progressive masking and adaptive loss scheduling allows our network to learn robust spatiotemporal representations without any high-frame-rate supervision. Our framework is evaluated on UCF101-7 and DAVIS-7 datasets. MiVID is trained entirely on CPU using the datasets and 9-frame video segments, making it a low-resource yet highly effective pipeline. Despite these constraints, our model achieves optimal results at just 50 epochs, competitive with several supervised baselines.This work demonstrates the power of self-supervised diffusion priors for temporally coherent frame synthesis and provides a scalable path toward accessible and generalizable VFI systems.

---

## 55. SSTODE: Ocean-Atmosphere Physics-Informed Neural ODEs for Sea Surface Temperature Prediction

**论文链接:** [http://arxiv.org/abs/2511.05629v1](http://arxiv.org/abs/2511.05629v1)

**作者:** Zheng Jiang, Wei Wang, Gaowei Zhang, Yi Wang

**发布时间:** 2025-11-07

**备注:** To be published in the Proceedings of AAAI-AISI 2026

### GPT解析

### 总结

SSTODE是一种物理信息神经常微分方程框架，用于海面温度预测，解决了现有物理信息神经网络在表征海水运动和整合外部SST驱动因素方面的不足，实现了高精度且具有物理一致性的SST预测。

### 背景

海面温度对于理解上层海洋热动力学和海洋-大气相互作用至关重要，这些过程对经济和社会有深远影响。数据驱动模型虽然显示出预测潜力，但其黑盒性质限制了可解释性。物理信息神经网络在处理复杂海洋-大气动力学时面临两大挑战：对海水运动(如上升流)的特征描述不足，以及外部SST驱动因素(如湍流热通量)的整合不足。

### 目的

开发SSTODE框架，解决现有物理信息神经网络在海水运动表征和外部驱动因素整合方面的局限性，实现更准确且具有物理一致性的海面温度预测。

### 方法

SSTODE框架从流体传输原理推导ODE，结合平流和扩散模型来表征海洋时空动力学。通过变分优化恢复潜在速度场，明确控制SST时间动态。此外，引入受海洋热量收支方程启发的能量交换积分器(EEI)，整合外部强迫因素，使这些因素分量的变化为SST动力学提供更深入见解。

### 主要发现

SSTODE在全球和区域SST预测基准测试中取得了最先进的性能。该模型直观地揭示了平流动力学、热扩散模式和日加热-冷却循环对SST演化的影响，证明了其解释性和物理一致性。

### 结论

SSTODE框架成功解决了现有物理信息神经网络在表征海水运动和整合外部驱动因素方面的挑战，实现了高精度且具有物理一致性的SST预测，同时提供了对SST动力学的深入见解。

### 翻译

海面温度(SST)对于理解上层海洋热动力学和海洋-大气相互作用至关重要，这些过程对经济和社会有深远影响。虽然数据驱动模型在SST预测中显示出潜力，但其黑盒性质往往限制了可解释性并忽略了关键物理过程。最近，物理信息神经网络正在兴起，但由于两个主要挑战而难以处理复杂的海洋-大气动力学：1)对海水运动(如上升流)的特征描述不足；2)外部SST驱动因素(如湍流热通量)的整合不足。为应对这些挑战，我们提出了SSTODE，这是一种用于SST预测的物理信息神经常微分方程(Neural ODEs)框架。首先，我们从流体传输原理推导ODE，结合平流和扩散来建模海洋时空动力学。通过变分优化，我们恢复了一个潜在速度场，明确控制SST的时间动态。在ODE基础上，我们引入了受海洋热量收支方程启发的能量交换积分器(EEI)，以考虑外部强迫因素。因此，这些因素分量的变化为SST动力学提供了更深入的见解。大量实验证明，SSTODE在全球和区域SST预测基准测试中取得了最先进的性能。此外，SSTODE直观地揭示了平流动力学、热扩散模式和日加热-冷却循环对SST演化的影响。这些发现证明了模型的解释性和物理一致性。


### 论文摘要

Sea Surface Temperature (SST) is crucial for understanding upper-ocean thermal dynamics and ocean-atmosphere interactions, which have profound economic and social impacts. While data-driven models show promise in SST prediction, their black-box nature often limits interpretability and overlooks key physical processes. Recently, physics-informed neural networks have been gaining momentum but struggle with complex ocean-atmosphere dynamics due to 1) inadequate characterization of seawater movement (e.g., coastal upwelling) and 2) insufficient integration of external SST drivers (e.g., turbulent heat fluxes). To address these challenges, we propose SSTODE, a physics-informed Neural Ordinary Differential Equations (Neural ODEs) framework for SST prediction. First, we derive ODEs from fluid transport principles, incorporating both advection and diffusion to model ocean spatiotemporal dynamics. Through variational optimization, we recover a latent velocity field that explicitly governs the temporal dynamics of SST. Building upon ODE, we introduce an Energy Exchanges Integrator (EEI)-inspired by ocean heat budget equations-to account for external forcing factors. Thus, the variations in the components of these factors provide deeper insights into SST dynamics. Extensive experiments demonstrate that SSTODE achieves state-of-the-art performances in global and regional SST forecasting benchmarks. Furthermore, SSTODE visually reveals the impact of advection dynamics, thermal diffusion patterns, and diurnal heating-cooling cycles on SST evolution. These findings demonstrate the model's interpretability and physical consistency.

---

## 56. Nowcast3D: Reliable precipitation nowcasting via gray-box learning

**论文链接:** [http://arxiv.org/abs/2511.04659v2](http://arxiv.org/abs/2511.04659v2)

**作者:** Huaguan Chen, Wei Han, Haofei Sun, Ning Lin, Xingtao Song, Yunfan Yang, Jie Tian, Yang Liu, Ji-Rong Wen, Xiaoye Zhang, Xueshun Shen, Hao Sun

**发布时间:** 2025-11-06

### GPT解析

### 总结

本文提出了一种灰盒式全三维临近预报框架，通过结合物理约束的神经算子与数据驱动学习，直接处理体积雷达反射率数据，实现了极端降水的高效可靠预报。

### 背景

极端降水临近预报需要高时空保真度和延长预报时效，但现有方法存在局限性：数值天气预报及其深度学习模拟对快速发展的对流过程太慢且分辨率低；外推方法和纯数据驱动模型存在误差累积和过度平滑问题；混合二维雷达方法丢弃了关键的垂直信息。

### 目的

开发一种能够准确处理极端降水预报的三维框架，解决现有方法的局限性，提高预报的准确性和可靠性。

### 方法

引入灰盒式全三维临近预报框架，直接处理体积雷达反射率数据，结合物理约束的神经算子与数据驱动学习；学习垂直变化的3D平流场；参数化空间变化的扩散；引入布朗运动启发的随机项表示未解析运动；使用残差分支捕获小尺度对流和微物理变异性；通过扩散-based随机模块估计不确定性。

### 主要发现

该框架在长达3小时的预报时效内实现了更准确的预报；在不同降水条件下表现良好；在160名气象学家进行的盲测评估中，57%的案例排名第一。

### 结论

通过恢复具有物理一致性的完整3D动力学，该框架为极端降水的高效可靠的临近预报提供了一种可扩展且稳健的途径。

### 翻译

极端降水临近预报需要高时空保真度和延长预报时效，然而现有方法仍然有限。数值天气预报及其深度学习模拟对于快速发展的对流过程来说太慢且分辨率太低，而外推和纯数据驱动模型则存在误差累积和过度平滑问题。混合二维雷达方法丢弃了关键的垂直信息，阻碍了准确重建高度依赖的动力学。我们引入了一种灰盒式、完全三维的临近预报框架，直接处理体积雷达反射率数据，并将物理约束的神经算子与数据驱动学习相结合。该模型在保守平流算子下学习垂直变化的3D平流场，参数化空间变化的扩散，并引入受布朗运动启发的随机项来表示未解析的运动。残差分支捕获小尺度对流 initiation 和微物理变异性，而基于扩散的随机模块估计不确定性。该框架在长达3小时的预报时效内实现了更准确的预报，并在160名气象学家进行的盲测评估中，57%的案例排名第一。通过恢复具有物理一致性的完整3D动力学，它为极端降水的高效可靠的临近预报提供了一种可扩展且稳健的途径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决极端降水临近预报的可靠性问题，特别是现有方法在时空分辨率和预报准确性方面的局限性。这个问题在现实中极其重要，因为极端降水事件如洪水和山体滑坡会对生命和基础设施造成严重威胁，如论文开头提到的广东梅州极端降雨事件导致55人死亡或失踪，超过16万人受灾。准确的短期降水预报对于灾害预防至关重要，但现有方法要么太慢（如数值天气预报），要么误差累积严重（如外推方法），要么忽略了关键的垂直信息（如二维雷达方法）。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：数值天气预报太慢太粗糙，外推和纯数据驱动模型有误差累积和过度平滑问题，而混合2D方法忽略了关键的垂直信息。然后作者提出需要充分利用三维雷达数据中包含的丰富信息，将物理约束与数据驱动学习相结合。作者借鉴了多项现有工作：NowcastNet等物理约束神经网络方法将演化算子嵌入网络架构；条件扩散模型用于生成预报集合；Helmholtz分解用于重建速度场；以及半拉格朗日平流方法处理流体变形。作者将这些方法创新性地整合到一个统一的框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过一个'灰盒'学习框架，直接处理三维体积雷达反射率，将物理约束的神经算子与数据驱动学习相结合，将复杂大气过程分解为物理可解释的组成部分。整体实现流程分为两个阶段：1)物理驱动预测模块：使用神经网络从历史雷达数据推断三维风场、扩散张量和微物理源项，通过求解物理方程生成确定性预报；2)概率生成模块：使用双分支条件扩散模型，基于确定性预报生成结构和强度组件，通过添加不同噪声样本创建集合预报，提供不确定性估计。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)直接处理三维体积雷达数据而非二维合成图；2)将反射率演变分解为平流、扩散和微物理三个物理过程；3)通过Helmholtz分解重建完整三维速度场；4)引入布朗运动随机项建模未解析运动；5)使用残差分支捕获小尺度对流变化；6)双分支条件扩散模型生成集合预报。相比之前工作，Nowcast3D克服了传统NWP的慢速低分辨率问题，减少了外推方法的误差累积，避免了纯数据驱动模型的过度平滑，并解决了混合2D方法忽略垂直信息的局限，实现了更准确、物理一致的极端降水预报。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Nowcast3D通过整合物理约束的三维动力学与深度生成模型，实现了对极端降水更准确、可靠和物理一致的临近预报，显著提高了预报的时空保真度并延长了有效预警时间。'}


### 论文摘要

Extreme precipitation nowcasting demands high spatiotemporal fidelity and extended lead times, yet existing approaches remain limited. Numerical Weather Prediction (NWP) and its deep-learning emulations are too slow and coarse for rapidly evolving convection, while extrapolation and purely data-driven models suffer from error accumulation and excessive smoothing. Hybrid 2D radar-based methods discard crucial vertical information, preventing accurate reconstruction of height-dependent dynamics. We introduce a gray-box, fully three-dimensional nowcasting framework that directly processes volumetric radar reflectivity and couples physically constrained neural operators with datadriven learning. The model learns vertically varying 3D advection fields under a conservative advection operator, parameterizes spatially varying diffusion, and introduces a Brownian-motion--inspired stochastic term to represent unresolved motions. A residual branch captures small-scale convective initiation and microphysical variability, while a diffusion-based stochastic module estimates uncertainty. The framework achieves more accurate forecasts up to three-hour lead time across precipitation regimes and ranked first in 57\% of cases in a blind evaluation by 160 meteorologists. By restoring full 3D dynamics with physical consistency, it offers a scalable and robust pathway for skillful and reliable nowcasting of extreme precipitation.

---

## 57. Deep Learning-Driven Downscaling for Climate Risk Assessment of Projected Temperature Extremes in the Nordic Region

**论文链接:** [http://arxiv.org/abs/2511.03770v1](http://arxiv.org/abs/2511.03770v1)

**作者:** Parthiban Loganathan, Elias Zea, Ricardo Vinuesa, Evelyn Otero

**发布时间:** 2025-11-05

### GPT解析

### 总结

本研究开发了一个整合降尺度框架，结合ViT、ConvLSTM和GeoStaNet模型，用于北欧不同气候区的高分辨率温度预测。通过DL-TOPSIS多标准决策系统评估，框架能够生成可信的降尺度预测，显示亚寒带和暖夏大陆性气候区到2100年将显著变暖，日温范围扩大，时间出现信号最早出现在亚寒带冬季，强调了适应措施的紧迫性。

### 背景

北欧不同Koppen-Geiger区域经历快速变化和增加的气候变异性，产生了显著的适应需求。区域规划需要高分辨率的温度预测。

### 目的

开发一个整合降尺度框架，用于生成高分辨率的温度预测，支持区域适应规划。

### 方法

提出整合降尺度框架，结合Vision Transformer (ViT)、卷积长短期记忆(ConvLSTM)和具有注意力和不平衡感知网络的地理空间时空Transformer(GeoStaNet)模型。使用多标准决策系统Deep Learning-TOPSIS (DL-TOPSIS)进行评估，选择十个战略性地选择的气象站，涵盖温带海洋性(Cfb)、亚极地海洋性(Cfc)、暖夏大陆性(Dfb)和亚寒带(Dfc)气候区域。使用Norwegian Earth System Model (NorESM2-LM) CMIP6输出，在1951-2014年期间进行偏差校正，并与早期观测数据进行验证。

### 主要发现

ViT显示出改进的性能(均方根误差：1.01摄氏度；R平方：0.92)，能够产生可信的降尺度预测。在SSP5-8.5情景下，到2100年，Dfc和Dfb气候区预计将分别变暖4.8摄氏度和3.9摄氏度，日温范围扩大超过1.5摄氏度。时间出现信号首先出现在亚寒带冬季季节(Dfc：约2032年)，表明需要紧急采取适应措施。

### 结论

提出的框架提供了基于站点的高分辨率估计不确定性和极端值，可直接用于快速环境变化的高纬度地区的适应政策。

### 翻译

广泛变化的北欧Koppen-Geiger区域的快速变化和增加的气候变异性产生了显著的适应需求。区域规划需要高分辨率的预测温度。这项工作提出了一个整合降尺度框架，结合了Vision Transformer (ViT)、卷积长短期记忆(ConvLSTM)和具有注意力和不平衡感知网络的地理空间时空Transformer(GeoStaNet)模型。该框架使用多标准决策系统Deep Learning-TOPSIS (DL-TOPSIS)进行评估，选择了十个战略性地选择的气象站，涵盖温带海洋性(Cfb)、亚极地海洋性(Cfc)、暖夏大陆性(Dfb)和亚寒带(Dfc)气候区域。Norwegian Earth System Model (NorESM2-LM) Coupled Model Intercomparison Project Phase 6 (CMIP6)输出在1951-2014年期间进行了偏差校正，并与早期的日温度指标和日温范围统计数据进行了验证。ViT显示出改进的性能(均方根误差：1.01摄氏度；R平方：0.92)，能够产生可信的降尺度预测。在SSP5-8.5情景下，到2100年，Dfc和Dfb气候区预计将分别变暖4.8摄氏度和3.9摄氏度，日温范围扩大超过1.5摄氏度。时间出现信号首先出现在亚寒带冬季季节(Dfc：约2032年)，表明需要紧急采取适应措施。提出的框架提供了基于站点的高分辨率估计不确定性和极端值，可直接用于快速环境变化的高纬度地区的适应政策。


### 论文摘要

Rapid changes and increasing climatic variability across the widely varied Koppen-Geiger regions of northern Europe generate significant needs for adaptation. Regional planning needs high-resolution projected temperatures. This work presents an integrative downscaling framework that incorporates Vision Transformer (ViT), Convolutional Long Short-Term Memory (ConvLSTM), and Geospatial Spatiotemporal Transformer with Attention and Imbalance-Aware Network (GeoStaNet) models. The framework is evaluated with a multicriteria decision system, Deep Learning-TOPSIS (DL-TOPSIS), for ten strategically chosen meteorological stations encompassing the temperate oceanic (Cfb), subpolar oceanic (Cfc), warm-summer continental (Dfb), and subarctic (Dfc) climate regions. Norwegian Earth System Model (NorESM2-LM) Coupled Model Intercomparison Project Phase 6 (CMIP6) outputs were bias-corrected during the 1951-2014 period and subsequently validated against earlier observations of day-to-day temperature metrics and diurnal range statistics. The ViT showed improved performance (Root Mean Squared Error (RMSE): 1.01 degrees C; R^2: 0.92), allowing for production of credible downscaled projections. Under the SSP5-8.5 scenario, the Dfc and Dfb climate zones are projected to warm by 4.8 degrees C and 3.9 degrees C, respectively, by 2100, with expansion in the diurnal temperature range by more than 1.5 degrees C. The Time of Emergence signal first appears in subarctic winter seasons (Dfc: approximately 2032), signifying an urgent need for adaptation measures. The presented framework offers station-based, high-resolution estimates of uncertainties and extremes, with direct uses for adaptation policy over high-latitude regions with fast environmental change.

---

## 58. MoE-GraphSAGE-Based Integrated Evaluation of Transient Rotor Angle and Voltage Stability in Power Systems

**论文链接:** [http://arxiv.org/abs/2511.08610v1](http://arxiv.org/abs/2511.08610v1)

**作者:** Kunyu Zhang, Guang Yang, Fashun Shi, Shaoying He, Yuchi Zhang

**发布时间:** 2025-11-05

### GPT解析

### 总结

本文提出了一种基于专家混合(MoE)的图神经网络框架MoE-GraphSAGE，用于统一暂态角度稳定性和暂态电压稳定性评估，解决了传统方法在准确性和计算效率方面的问题。

### 背景

大规模可再生能源和电力电子设备的集成增加了电力系统稳定性复杂性，使暂态稳定性评估更加困难，而传统方法在准确性和计算效率方面存在局限性。

### 目的

开发一种能够统一评估暂态角度稳定性和暂态电压稳定性的高效方法，提高评估的准确性和计算效率。

### 方法

提出MoE-GraphSAGE框架，利用GraphSAGE捕获电力系统的时空拓扑特征，并采用具有门控机制的多专家网络共同建模不同的不稳定模式。

### 主要发现

在IEEE 39节点系统上的实验结果表明，MoE-GraphSAGE实现了优越的准确性和效率。

### 结论

MoE-GraphSAGE为复杂电力系统在线多任务暂态稳定性评估提供了有效的解决方案。

### 翻译

可再生能源和电力电子设备的大规模集成增加了电力系统稳定性复杂性，使暂态稳定性评估更具挑战性。传统方法在准确性和计算效率方面都有限制。为应对这些挑战，本文提出了MoE-GraphSAGE，一种基于专家混合(MoE)的图神经网络框架，用于统一的TAS和TVS评估。该框架利用GraphSAGE捕获电力系统的时空拓扑特征，并采用具有门控机制的多专家网络共同建模不同的不稳定模式。IEEE 39节点系统上的实验结果表明，MoE-GraphSAGE实现了优越的准确性和效率，为复杂电力系统在线多任务暂态稳定性评估提供了有效的解决方案。


### 论文摘要

The large-scale integration of renewable energy and power electronic devices has increased the complexity of power system stability, making transient stability assessment more challenging. Conventional methods are limited in both accuracy and computational efficiency. To address these challenges, this paper proposes MoE-GraphSAGE, a graph neural network framework based on the MoE for unified TAS and TVS assessment. The framework leverages GraphSAGE to capture the power grid's spatiotemporal topological features and employs multi-expert networks with a gating mechanism to model distinct instability modes jointly. Experimental results on the IEEE 39-bus system demonstrate that MoE-GraphSAGE achieves superior accuracy and efficiency, offering an effective solution for online multi-task transient stability assessment in complex power systems.

---

## 59. Drone Swarm Energy Management

**论文链接:** [http://arxiv.org/abs/2511.11557v1](http://arxiv.org/abs/2511.11557v1)

**作者:** Michael Z. Zgurovsky, Pavlo O. Kasyanov, Liliia S. Paliichuk

**发布时间:** 2025-11-14

**备注:** 14 pages, 4 Tables, 2 Figures

### GPT解析

### 总结

本研究提出了一种基于部分可观察马尔可夫决策过程与深度确定性策略梯度强化学习相结合的分析框架，用于在不确定性环境下运行的无人机蜂群系统的决策制定。该框架实现了无人机在认知人工智能平台中的自适应控制和协作行为，使每个智能体能从动态环境状态中学习最优能源管理和导航策略。

### 背景

无人机蜂群系统在不确定性环境下的决策制定面临挑战，需要智能化的决策方法来处理部分可观察环境和动态变化。

### 目的

开发一种能够处理部分可观察环境的无人机蜂群决策框架，提高任务成功率和能源效率，支持分布式学习和多智能体决策协调。

### 方法

将部分可观察马尔可夫决策过程与深度确定性策略梯度强化学习相结合，扩展标准DDPG架构，加入贝叶斯滤波推导的信念状态表示，并在高斯情况下通过数值比较评估性能。

### 主要发现

POMDP-DDPG蜂群控制模型相比基线方法显著提高了任务成功率和能源效率，能够支持多智能体间的分布式学习和决策协调。

### 结论

该框架为可扩展的认知蜂群自主性提供了基础，有助于推进智能多agent系统的节能控制算法发展，可应用于安全、环境监测和基础设施检查等场景。

### 翻译

本文提出了一种用于在不确定性环境下运行的无人机蜂群系统决策制定的分析框架，该框架基于部分可观察马尔可夫决策过程与深度确定性策略梯度强化学习的集成。所提出的方法使无人机在认知人工智能平台中能够实现自适应控制和协作行为，其中每个智能体从动态环境状态中学习最优能源管理和导航策略。我们通过贝叶斯滤波推导的信念状态表示扩展了标准DDPG架构，使部分可观察环境中的决策制定更加稳健。在本文中，针对高斯情况，我们通过数值比较了从DDPG派生的策略与原始连续问题离散化版本的最优策略的性能。模拟结果表明，基于POMDP-DDPG的蜂群控制模型相比基线方法显著提高了任务成功率和能源效率。所开发的框架支持多智能体之间的分布式学习和决策协调，为可扩展的认知蜂群自主性提供了基础。本研究成果有助于推进智能多agent系统能源感知控制算法的发展，可应用于安全、环境监测和基础设施检查场景。


### 论文摘要

This note presents an analytical framework for decision-making in drone swarm systems operating under uncertainty, based on the integration of Partially Observable Markov Decision Processes (POMDP) with Deep Deterministic Policy Gradient (DDPG) reinforcement learning. The proposed approach enables adaptive control and cooperative behavior of unmanned aerial vehicles (UAVs) within a cognitive AI platform, where each agent learns optimal energy management and navigation policies from dynamic environmental states. We extend the standard DDPG architecture with a belief-state representation derived from Bayesian filtering, allowing for robust decision-making in partially observable environments. In this paper, for the Gaussian case, we numerically compare the performance of policies derived from DDPG to optimal policies for discretized versions of the original continuous problem. Simulation results demonstrate that the POMDP-DDPG-based swarm control model significantly improves mission success rates and energy efficiency compared to baseline methods. The developed framework supports distributed learning and decision coordination across multiple agents, providing a foundation for scalable cognitive swarm autonomy. The outcomes of this research contribute to the advancement of energy-aware control algorithms for intelligent multi-agent systems and can be applied in security, environmental monitoring, and infrastructure inspection scenarios.

---

## 60. OpenUS: A Fully Open-Source Foundation Model for Ultrasound Image Analysis via Self-Adaptive Masked Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2511.11510v1](http://arxiv.org/abs/2511.11510v1)

**作者:** Xiaoyu Zheng, Xu Chen, Awais Rauf, Qifan Fu, Benedetta Monosi, Felice Rivellese, Myles J. Lewis, Shaogang Gong, Gregory Slabaugh

**发布时间:** 2025-11-14

### GPT解析

### 总结

OpenUS是首个基于大量公共数据构建的可复现、开源超声基础模型，采用创新的自适应掩码框架和动态学习计划，可轻松适应各种下游任务，实现标签高效微调。

### 背景

超声(US)是一种广泛使用的医学成像方式，具有低成本、便携、实时反馈和无电离辐射的优点。然而，超声图像解释高度依赖操作者，因解剖区域、采集协议和设备类型而有很大差异。散斑、低对比度和有限的标准化注释等挑战阻碍了通用化、标签高效的超声AI模型的发展。

### 目的

提出OpenUS，构建第一个可复现的、开源的超声基础模型，建立在大量公共数据之上，以解决超声AI模型面临的挑战。

### 方法

OpenUS采用视觉Mamba主干网络捕捉图像中的局部和全局长程依赖关系；引入自适应掩码框架结合对比学习和掩码图像建模；应用动态学习计划逐步调整预训练难度；构建了包含308K张图像来自42个公共数据集的最大公共超声数据集。

### 主要发现

OpenUS预训练模型可以轻松适应特定的下游任务，作为标签高效微调的主干网络，有效解决了超声图像解释的挑战。

### 结论

OpenUS是一个创新的、可复现的、开源的超声基础模型，通过其独特的方法和数据集规模，为超声AI的发展提供了新的可能性，代码已在GitHub上公开。

### 翻译

超声(US)是最广泛使用的医学成像方式之一，得益于其低成本、便携性、实时反馈和无电离辐射。然而，超声图像解释仍然高度依赖操作者，并且在不同的解剖区域、采集协议和设备类型之间差异显著。这些变化以及散斑、低对比度和有限的标准化注释等独特挑战，阻碍了通用化、标签高效的超声AI模型的发展。在本文中，我们提出了OpenUS，这是建立在大量公共数据上的第一个可复现的、开源的超声基础模型。OpenUS采用视觉Mamba主干网络，捕捉图像中局部和全局的长程依赖关系。为了在预训练期间提取丰富的特征，我们引入了一种新颖的自适应掩码框架，将对比学习与掩码图像建模相结合。该策略将教师的注意力图与学生的重建损失相结合，自适应地完善临床相关的掩码，以提高预训练效果。OpenUS还应用动态学习计划，逐步调整预训练过程的难度。为了开发这个基础模型，我们编译了迄今为止最大的公共超声数据集，包含来自42个可公开获取数据集的超过308K张图像，涵盖了多样的解剖区域、机构、成像设备和疾病类型。我们预训练的OpenUS模型可以通过作为标签高效微调的主干网络，轻松适应特定的下游任务。代码可在https://github.com/XZheng0427/OpenUS获取。


### 论文摘要

Ultrasound (US) is one of the most widely used medical imaging modalities, thanks to its low cost, portability, real-time feedback, and absence of ionizing radiation. However, US image interpretation remains highly operator-dependent and varies significantly across anatomical regions, acquisition protocols, and device types. These variations, along with unique challenges such as speckle, low contrast, and limited standardized annotations, hinder the development of generalizable, label-efficient ultrasound AI models. In this paper, we propose OpenUS, the first reproducible, open-source ultrasound foundation model built on a large collection of public data. OpenUS employs a vision Mamba backbone, capturing both local and global long-range dependencies across the image. To extract rich features during pre-training, we introduce a novel self-adaptive masking framework that combines contrastive learning with masked image modeling. This strategy integrates the teacher's attention map with student reconstruction loss, adaptively refining clinically-relevant masking to enhance pre-training effectiveness. OpenUS also applies a dynamic learning schedule to progressively adjust the difficulty of the pre-training process. To develop the foundation model, we compile the largest to-date public ultrasound dataset comprising over 308K images from 42 publicly available datasets, covering diverse anatomical regions, institutions, imaging devices, and disease types. Our pre-trained OpenUS model can be easily adapted to specific downstream tasks by serving as a backbone for label-efficient fine-tuning. Code is available at https://github.com/XZheng0427/OpenUS.

---

## 61. PAS : Prelim Attention Score for Detecting Object Hallucinations in Large Vision--Language Models

**论文链接:** [http://arxiv.org/abs/2511.11502v1](http://arxiv.org/abs/2511.11502v1)

**作者:** Nhat Hoang-Xuan, Minh Vu, My T. Thai, Manish Bhattarai

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文研究了大型视觉-语言模型中的目标幻觉问题，发现模型常忽略图像而依赖先前生成的输出，并提出了一种轻量级的幻觉检测方法。

### 背景

大型视觉-语言模型功能强大但存在目标幻觉问题，即模型生成与图像内容不符的对象。

### 目的

探究LVLM产生目标幻觉的原因，并开发一种有效的幻觉检测方法。

### 方法

引入Prelim Attention Score (PAS)，一种基于先前生成tokens上注意力权重的轻量级、无需训练的信号，可在推理过程中实时计算。

### 主要发现

在许多幻觉性预测中，模型实际上忽略了图像，而是依赖于先前生成的输出tokens来推断新对象；图像依赖性弱与幻觉有很强的相关性。

### 结论

PAS方法在多个模型和数据集上实现了最先进的目标幻觉检测，能够实现实时过滤和干预，无需额外计算资源。

### 翻译

大型视觉-语言模型功能强大，但由于目标幻觉问题而仍然不可靠。在本工作中，我们表明在许多幻觉性预测中，大型视觉-语言模型实际上忽略了图像，而是依赖于先前生成的输出tokens来推断新对象。我们通过计算图像与预测对象之间以先前生成为条件的互信息来量化这种行为，证明图像依赖性弱与幻觉有很强的相关性。基于这一发现，我们引入了Prelim Attention Score，这是一种基于先前生成tokens上注意力权重的轻量级、无需训练的信号。PAS不需要额外的前向传播计算，可以在推理过程中实时计算。利用这个先前被忽视的信号，PAS在多个模型和数据集上实现了最先进的目标幻觉检测，能够实现实时过滤和干预。


### 论文摘要

Large vision-language models (LVLMs) are powerful, yet they remain unreliable due to object hallucinations. In this work, we show that in many hallucinatory predictions the LVLM effectively ignores the image and instead relies on previously generated output (prelim) tokens to infer new objects. We quantify this behavior via the mutual information between the image and the predicted object conditioned on the prelim, demonstrating that weak image dependence strongly correlates with hallucination. Building on this finding, we introduce the Prelim Attention Score (PAS), a lightweight, training-free signal computed from attention weights over prelim tokens. PAS requires no additional forward passes and can be computed on the fly during inference. Exploiting this previously overlooked signal, PAS achieves state-of-the-art object-hallucination detection across multiple models and datasets, enabling real-time filtering and intervention.

---

## 62. Sat2RealCity: Geometry-Aware and Appearance-Controllable 3D Urban Generation from Satellite Imagery

**论文链接:** [http://arxiv.org/abs/2511.11470v1](http://arxiv.org/abs/2511.11470v1)

**作者:** Yijie Kang, Xinliang Wang, Zhenyu Wu, Yifeng Shi, Hailong Zhu

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了Sat2RealCity，一种基于真实世界卫星图像的几何感知和外观可控的3D城市生成框架，解决了现有方法需要大规模3D城市资产和缺乏真实世界外观连接的问题。

### 背景

生成模型在3D城市生成方面取得了显著进展，应用于数字孪生、虚拟城市和大规模模拟。然而现有方法面临两个关键挑战：需要大规模3D城市资产进行监督训练，这些资产难以且昂贵获取；依赖语义图或高度图，仅用于生成虚拟世界中的建筑，与真实世界外观缺乏连接，限制了真实性和泛化能力。

### 目的

提出Sat2RealCity框架，解决现有方法的局限性，实现基于真实世界卫星图像的3D城市生成，具有几何感知和外观控制能力。

### 方法

Sat2RealCity基于单个建筑实体构建生成，而非城市级别，利用丰富的先验知识和预训练知识，同时减少对大规模3D城市资产的依赖。具体包括：引入基于OSM的空间先验策略，实现可解释的几何生成；设计外观引导的可控建模机制，实现细粒度的外观真实性和风格控制；构建基于MLLM的语义引导生成流程，桥接语义解释和几何重建。

### 主要发现

大量的定量和定性实验表明，Sat2RealCity在结构一致性和外观真实性方面显著超越现有基线方法。

### 结论

Sat2RealCity为与真实世界对齐的3D城市内容创建奠定了坚实基础，代码即将发布。

### 翻译

生成模型的最新进展显著增强了3D城市生成能力，使其能够应用于数字孪生、虚拟城市和大规模模拟。然而，现有方法面临两个关键挑战：(1)需要大规模3D城市资产进行监督训练，这些资产难以且昂贵获取；(2)依赖语义图或高度图，这些方法仅用于生成虚拟世界中的建筑，与真实世界外观缺乏连接，限制了生成城市的真实性和泛化能力。为解决这些局限性，我们提出了Sat2RealCity，一种基于真实世界卫星图像的几何感知和外观可控的3D城市生成框架。与之前的城市级生成方法不同，Sat2RealCity基于单个建筑实体构建生成，能够利用3D对象生成中的丰富先验和预训练知识，同时大幅减少对大规模3D城市资产的依赖。具体来说，(1)我们引入了基于OSM的空间先验策略，从空间拓扑到建筑实例实现可解释的几何生成；(2)我们设计了外观引导的可控建模机制，实现细粒度的外观真实性和风格控制；(3)我们构建了基于MLLM的语义引导生成流程，桥接语义解释和几何重建。大量的定量和定性实验证明，Sat2RealCity在结构一致性和外观真实性方面显著超越现有基线方法，为与真实世界对齐的3D城市内容创建奠定了坚实基础。代码即将发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决两个关键问题：一是现有3D城市生成方法需要大规模3D城市资产进行监督训练，而这些数据难以获取且成本高昂；二是现有方法依赖语义图或高度图，这些输入主要用于生成虚拟世界中的建筑，缺乏与真实世界外观的联系，限制了生成城市的真实性和泛化能力。这些问题在数字孪生、虚拟城市和大规模模拟等应用中非常重要，因为这些应用需要高质量、真实感强的3D城市模型，而现有方法生成的城市往往缺乏真实世界的细节和一致性，难以在实际应用中部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认为现有城市级生成方法将整个城市块视为单一场景存在局限性，因此提出基于建筑实体的生成方法，可以整合3D对象生成的丰富先验知识和预训练模型，同时减少对大规模3D城市资产的依赖。作者借鉴了多项现有工作：基于TRELLIS的两阶段生成结构-外观生成范式、利用OpenStreetMap (OSM)数据作为几何先验、使用多模态大语言模型(MLLM)进行语义理解，以及采用扩散模型和Transformer架构。通过这些借鉴和改进，作者设计了一个完整的从卫星图像到3D城市的生成流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是基于建筑实体的生成方法，而非整体城市块，利用OSM数据提供几何先验，实现几何感知和地理空间对齐，通过外观引导的可控建模机制实现细粒度外观真实性和风格控制，并使用MLLM支持的语义引导生成管道连接语义解释和几何重建。整体流程包括：1)使用OSM数据提取建筑轮廓和高度信息作为几何先验；2)通过卫星图像提取建筑顶部视图特征；3)使用MLLM分析卫星图像，生成建筑正面外观描述；4)根据描述生成建筑正面外观图像；5)结合几何先验、顶部视图和正面视图，使用TRELLIS框架生成3D建筑；6)将生成的建筑按OSM坐标组装成城市场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于OSM的空间先验策略，实现从空间拓扑到建筑实例的可解释几何生成；2)外观引导的可控建模机制，实现细粒度外观真实性和风格控制；3)MLLM支持的语义引导生成管道，连接语义解释和几何重建；4)构建了专门的3D建筑数据集，包含11,579个高质量AIGC生成的建筑模型。相比之前的工作，Sat2RealCity不再依赖大规模3D城市资产进行训练，直接从真实世界卫星图像生成，而非使用语义图或高度图；采用基于建筑实体的生成，而非整个城市块；实现了几何感知和外观可控，同时保持区域风格一致性；结合了多模态理解能力，提高了生成结果的真实性和一致性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Sat2RealCity提出了一种基于卫星图像的几何感知和外观可控的3D城市生成框架，通过建筑实体级生成、OSM几何先验、外观引导建模和MLLM语义理解，实现了高保真、真实对齐的城市场景生成，显著提升了结构一致性和外观真实感。'}


### 论文摘要

Recent advances in generative modeling have substantially enhanced 3D urban generation, enabling applications in digital twins, virtual cities, and large-scale simulations. However, existing methods face two key challenges: (1) the need for large-scale 3D city assets for supervised training, which are difficult and costly to obtain, and (2) reliance on semantic or height maps, which are used exclusively for generating buildings in virtual worlds and lack connection to real-world appearance, limiting the realism and generalizability of generated cities. To address these limitations, we propose Sat2RealCity, a geometry-aware and appearance-controllable framework for 3D urban generation from real-world satellite imagery. Unlike previous city-level generation methods, Sat2RealCity builds generation upon individual building entities, enabling the use of rich priors and pretrained knowledge from 3D object generation while substantially reducing dependence on large-scale 3D city assets. Specifically, (1) we introduce the OSM-based spatial priors strategy to achieve interpretable geometric generation from spatial topology to building instances; (2) we design an appearance-guided controllable modeling mechanism for fine-grained appearance realism and style control; and (3) we construct an MLLM-powered semantic-guided generation pipeline, bridging semantic interpretation and geometric reconstruction. Extensive quantitative and qualitative experiments demonstrate that Sat2RealCity significantly surpasses existing baselines in structural consistency and appearance realism, establishing a strong foundation for real-world aligned 3D urban content creation. The code will be released soon.

---

## 63. Benchmarking Visual LLMs Resilience to Unanswerable Questions on Visually Rich Documents

**论文链接:** [http://arxiv.org/abs/2511.11468v1](http://arxiv.org/abs/2511.11468v1)

**作者:** Davide Napolitano, Luca Cagliero, Fabrizio Battiloro

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究提出了VRD-UQA基准，用于评估视觉大语言模型(VLLMs)对视觉丰富文档(VRDs)中看似合理但实际无法回答问题的检测能力。

### 背景

视觉大语言模型(VLLMs)已革新视觉丰富文档(VRDs)的自动理解，但在检测无法回答的问题方面仍存在挑战。

### 目的

探究VLLMs对看似合理但实际无法回答的问题的鲁棒性，并开发评估框架来衡量这种能力。

### 方法

创建VRD-UQA基准，通过替换原始自然语言实体生成损坏问题，使用VLLM-as-a-judge方法验证不可回答性，并在12个模型上测试不同类型的损坏(实体、文档元素、布局)和知识注入策略。

### 主要发现

VLLMs在检测无法回答问题方面存在局限性，不同类型的损坏和知识注入策略对模型性能有显著影响。

### 结论

VRD-UQA可作为开发鲁棒文档视觉问答系统的评估框架，帮助改进模型对不可回答问题的检测能力。

### 翻译

视觉大语言模型(VLLMs)的发展革新了视觉丰富文档(VRDs)的自动理解，这些文档包含文本和视觉元素。尽管VLLMs在多页VRDs的视觉问答(VQA)方面表现出色，但它们检测无法回答问题的能力仍然是一个开放的研究问题。我们的研究探讨了VLLMs对看似合理但实际无法回答问题的鲁棒性，即由于相关概念交换或合理问题表述产生的细微损坏而无法回答的问题。损坏是通过将原始自然语言实体替换为同一类型但属于不同文档元素、不同布局位置或不同页面的其他实体来生成的。为此，我们提出了VRD-UQA(视觉丰富文档不可回答问题回答)基准，用于评估VLLMs在多个维度上对看似合理但实际无法回答问题的恢复能力。它自动修改现有VQA数据集中的问题，使用VLLM-as-a-judge方法验证其不可回答性，然后全面评估VLLMs的性能。在12个模型上进行的实验分析了：(1) VLLMs在页面和文档级别检测不可回答问题的准确性；(2) 不同类型损坏(NLP实体、文档元素、布局)的影响；(3) 基于上下文学习的不同知识注入策略(OCR、多页选择或不可回答的可能性)的有效性。我们的研究结果揭示了VLLMs的局限性，并表明VRD-UQA可以作为开发鲁棒文档VQA系统的评估框架。


### 论文摘要

The evolution of Visual Large Language Models (VLLMs) has revolutionized the automatic understanding of Visually Rich Documents (VRDs), which contain both textual and visual elements. Although VLLMs excel in Visual Question Answering (VQA) on multi-page VRDs, their ability to detect unanswerable questions is still an open research question. Our research delves into the robustness of the VLLMs to plausible yet unanswerable questions, i.e., questions that appear valid but cannot be answered due to subtle corruptions caused by swaps between related concepts or plausible question formulations. Corruptions are generated by replacing the original natural language entities with other ones of the same type, belonging to different document elements, and in different layout positions or pages of the related document. To this end, we present VRD-UQA (VISUALLY RICH DOCUMENT UNANSWERABLE QUESTION ANSWERING), a benchmark for evaluating VLLMs' resilience to plausible yet unanswerable questions across multiple dimensions. It automatically alters the questions of existing VQA datasets consisting of multi-page VRDs, verifies their unanswerability using a VLLM-as-a-judge approach, and then thoroughly evaluates VLLMs' performance. Experiments, run on 12 models, analyze: (1) The VLLMs' accuracy in detecting unanswerable questions at both page and document levels; (2) The effect of different types of corruption (NLP entity, document element, layout); (3) The effectiveness of different knowledge injection strategies based on in-context learning (OCR, multi-page selection, or the possibility of unanswerability). Our findings reveal VLLMs' limitations and demonstrate that VRD-UQA can serve as an evaluation framework for developing resilient document VQA systems.

---

## 64. Totally mixed conditional independence equilibria of generic games

**论文链接:** [http://arxiv.org/abs/2511.11467v1](http://arxiv.org/abs/2511.11467v1)

**作者:** Matthieu Bouyer, Irem Portakal, Javier Sendra-Arranz

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文进一步发展了条件独立(CI)均衡的代数-几何基础，这是依赖均衡的一种精炼，将图形模型中的条件独立关系整合到战略推理中，从而包含纳什均衡。作者分析了任意格式的一般博弈中相关的Spohn CI varieties的结构，引入了聚类图形模型的Nash CI varieties类，并证明了它们的不可约性。

### 背景

条件独立(CI)均衡是依赖均衡的一种精炼，它将图形模型中的条件独立关系整合到战略推理中，从而包含纳什均衡。之前已有关于二元博弈的研究，但需要扩展到更一般的博弈形式。

### 目的

进一步发展条件独立均衡的代数-几何基础，分析一般博弈中相关的Spohn CI varieties结构，并为聚类图形模型引入Nash CI varieties类，研究其性质和完全混合CI均衡的存在条件。

### 方法

采用代数几何方法，分析Spohn CI varieties的结构，研究其维数和性质。对于聚类图形模型，引入Nash CI varieties类，证明其不可约性，描述其定义方程、次数以及完全混合CI均衡存在的条件。

### 主要发现

对于一般博弈，Spohn CI variety要么为空，其余维数等于玩家策略维度之和减去参数化无向图形模型中的玩家数量；当非空时，完全混合CI均衡的集合形成光滑流形；对于聚类图形模型，Nash CI varieties是不可约的，并可以描述其定义方程、次数和完全混合CI均衡存在的条件。

### 结论

条件独立均衡为战略推理提供了更丰富的代数-几何基础，特别是在结合图形模型的条件独立关系方面。这些结果为理解和分析博弈中的均衡结构提供了新的视角。

### 翻译

本文进一步发展了条件独立(CI)均衡的代数-几何基础，这是依赖均衡的一种精炼，它将图形模型中的条件独立关系整合到战略推理中，从而包含纳什均衡。扩展了关于二元博弈的早期工作，我们分析了任意格式的一般博弈中相关的Spohn CI varieties的结构。我们证明，对于一般博弈，Spohn CI variety要么为空，其余维数等于玩家策略维度之和减去参数化无向图形模型中的玩家数量。当非空时，完全混合CI均衡的集合对于一般博弈形成一个光滑流形。对于聚类图形模型，我们引入了Nash CI varieties类，证明了它们的不可约性，并描述了它们的定义方程、次数以及一般博弈中完全混合CI均衡存在的条件。


### 论文摘要

This paper further develops the algebraic--geometric foundations of conditional independence (CI) equilibria, a refinement of dependency equilibria that integrates conditional independence relations from graphical models into strategic reasoning and thereby subsumes Nash equilibria. Extending earlier work on binary games, we analyze the structure of the associated Spohn CI varieties for generic games of arbitrary format. We show that for generic games the Spohn CI variety is either empty or has codimension equal to the sum of the players' strategy dimensions minus the number of players in the parametrized undirected graphical model. When non-empty, the set of totally mixed CI equilibria forms a smooth manifold for generic games. For cluster graphical models, we introduce the class of Nash CI varieties, prove their irreducibility, and describe their defining equations, degrees, and conditions for the existence of totally mixed CI equilibria for generic games.

---

## 65. VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.11438v1](http://arxiv.org/abs/2511.11438v1)

**作者:** Mingjie Xu, Jinpeng Chen, Yuzhi Zhao, Jason Chun Lok Li, Yue Qiu, Zekang Du, Mengyang Wu, Pingping Zhang, Kun Li, Hongzheng Yang, Wenao Ma, Jiaheng Wei, Qinbin Li, Kangcheng Liu, Wenqiang Lei

**发布时间:** 2025-11-14

**备注:** This is the extended version of the paper accepted at AAAI 2026, which includes all technical appendices and additional experimental details

### GPT解析

### 总结

该研究提出了VP-Bench，一个用于评估多模态大语言模型(MLLMs)视觉提示(VPs)感知和利用能力的基准。VP-Bench采用两阶段评估框架，评估了28个MLLMs，并分析了影响VP理解的因素。

### 背景

多模态大语言模型已实现多种视觉语言应用，人类用户使用视觉提示(如边界框)来指定图像中的特定区域。然而，缺乏系统评估MLLMs解释此类VPs能力的基准。

### 目的

解决现有基准无法评估MLLMs视觉提示能力的局限性，引入VP-Bench基准。

### 方法

VP-Bench采用两阶段评估框架：第一阶段评估模型在自然场景中感知VPs的能力，使用30k可视化提示(8种形状和355种属性组合)；第二阶段研究VPs对下游任务的影响。评估了28个MLLMs，包括专有系统和开源模型，分析影响VP理解的因素。

### 主要发现

评估了28个MLLMs(如GPT-4o、InternVL3和Qwen2.5-VL)的VP理解能力，分析了VP属性、问题排列和模型规模等因素的影响。

### 结论

VP-Bench为研究MLLMs如何理解和解决基于上下文的指代问题建立了新的参考框架。

### 翻译

多模态大语言模型(MLLMs)已经实现了多种先进的视觉语言应用，包括细粒度物体识别和上下文理解。当查询图像中的特定区域或物体时，人类用户自然使用'视觉提示'(VPs)，如边界框，来提供参考。然而，目前没有现有的基准系统能够评估MLLMs解释此类VPs的能力。这一局限性使得当前MLLMs是否能够有效识别VPs(人类直观的提示方法)并利用它们解决问题尚不明确。为解决这一问题，我们引入了VP-Bench，一个用于评估MLLMs在VP感知和利用能力的基准。VP-Bench采用两阶段评估框架：第一阶段检查模型在自然场景中感知VPs的能力，使用跨越8种形状和355种属性组合的30k可视化提示；第二阶段研究VPs对下游任务的影响，衡量它们在现实问题解决场景中的有效性。使用VP-Bench，我们评估了28个MLLMs，包括专有系统(如GPT-4o)和开源模型(如InternVL3和Qwen2.5-VL)，并提供影响VP理解的因素的综合分析，如VP属性变化、问题排列和模型规模。VP-Bench为研究MLLMs如何理解和解决基于上下文的指代问题建立了新的参考框架。


### 论文摘要

Multimodal large language models (MLLMs) have enabled a wide range of advanced vision-language applications, including fine-grained object recognition and contextual understanding. When querying specific regions or objects in an image, human users naturally use "visual prompts" (VPs), such as bounding boxes, to provide reference. However, no existing benchmark systematically evaluates the ability of MLLMs to interpret such VPs. This gap leaves it unclear whether current MLLMs can effectively recognize VPs, an intuitive prompting method for humans, and use them to solve problems. To address this limitation, we introduce VP-Bench, a benchmark for assessing MLLMs' capability in VP perception and utilization. VP-Bench employs a two-stage evaluation framework: Stage 1 examines models' ability to perceive VPs in natural scenes, using 30k visualized prompts spanning eight shapes and 355 attribute combinations. Stage 2 investigates the impact of VPs on downstream tasks, measuring their effectiveness in real-world problem-solving scenarios. Using VP-Bench, we evaluate 28 MLLMs, including proprietary systems (e.g., GPT-4o) and open-source models (e.g., InternVL3 and Qwen2.5-VL), and provide a comprehensive analysis of factors that affect VP understanding, such as variations in VP attributes, question arrangement, and model scale. VP-Bench establishes a new reference framework for studying how MLLMs comprehend and resolve grounded referring questions.

---

## 66. WEAVE: Unleashing and Benchmarking the In-context Interleaved Comprehension and Generation

**论文链接:** [http://arxiv.org/abs/2511.11434v1](http://arxiv.org/abs/2511.11434v1)

**作者:** Wei Chow, Jiachun Pan, Yongyuan Liang, Mingze Zhou, Xue Song, Liyu Jia, Saining Zhang, Siliang Tang, Juncheng Li, Fengda Zhang, Weijia Wu, Hanwang Zhang, Tat-Seng Chua

**发布时间:** 2025-11-14

### GPT解析

### 总结

WEAVE是首个用于上下文交错跨模态理解和生成的套件，包含WEAVE-100k大规模数据集和WEAVEBench基准测试，解决了现有多模态模型在多轮、上下文依赖性图像生成和编辑方面的局限性。

### 背景

统一多模态模型(UMMs)在视觉理解和生成方面取得了显著进展，但现有数据集和基准测试主要关注单轮交互，无法捕捉现实世界中图像创建和编辑的多轮、上下文依赖特性。

### 目的

解决现有数据集和基准测试的局限性，提出一个能够评估多模态模型在多轮、上下文感知环境中表现的新框架。

### 方法

WEAVE套件包含两个互补部分：WEAVE-100k是包含10万交错样本、跨越37万对话轮次和50万图像的大规模数据集；WEAVEBench是基于480张图像的100个人工标注任务基准，采用混合VLM评估框架。

### 主要发现

在WEAVE-100k上训练使模型具备视觉理解、图像编辑和理解-生成协作能力，有助于发展涌现的视觉记忆能力；在WEAVEBench上的评估暴露了当前方法在多轮、上下文感知图像生成和编辑方面的持续局限。

### 结论

WEAVE为多模态社区研究上下文交错理解和生成提供了视角和基础。

### 翻译

统一多模态模型(UMMs)的最新进展已在视觉理解和生成方面取得了令人印象深刻的进展。然而，现有数据集和基准测试主要关注单轮交互，无法捕捉现实世界中图像创建和编辑的多轮、上下文依赖特性。为解决这一差距，我们提出了WEAVE，这是首个用于上下文交错跨模态理解和生成的套件。我们的套件包含两个互补部分。WEAVE-100k是一个包含10万交错样本的大规模数据集，跨越超过37万对话轮次和50万图像，涵盖需要基于历史上下文进行推理的理解、编辑和生成任务。WEAVEBench是一个基于480张图像的100个任务的人工标注基准，采用基于参考图像以及原始图像与编辑指令组合的混合VLM评估框架，评估模型在多轮生成、视觉记忆和跨领域知识推理方面的能力。实验证明，在WEAVE-100k上训练使模型具备视觉理解、图像编辑和理解-生成协作能力。此外，它有助于UMMs发展涌现的视觉记忆能力，同时在WEAVEBench上的广泛评估暴露了当前方法在多轮、上下文感知图像生成和编辑方面的持续局限和挑战。我们相信WEAVE为多模态社区研究上下文交错理解和生成提供了视角和基础。


### 论文摘要

Recent advances in unified multimodal models (UMMs) have enabled impressive progress in visual comprehension and generation. However, existing datasets and benchmarks focus primarily on single-turn interactions, failing to capture the multi-turn, context-dependent nature of real-world image creation and editing. To address this gap, we present WEAVE, the first suite for in-context interleaved cross-modality comprehension and generation. Our suite consists of two complementary parts. WEAVE-100k is a large-scale dataset of 100K interleaved samples spanning over 370K dialogue turns and 500K images, covering comprehension, editing, and generation tasks that require reasoning over historical context. WEAVEBench is a human-annotated benchmark with 100 tasks based on 480 images, featuring a hybrid VLM judger evaluation framework based on both the reference image and the combination of the original image with editing instructions that assesses models' abilities in multi-turn generation, visual memory, and world-knowledge reasoning across diverse domains. Experiments demonstrate that training on WEAVE-100k enables vision comprehension, image editing, and comprehension-generation collaboration capabilities. Furthermore, it facilitates UMMs to develop emergent visual-memory capabilities, while extensive evaluations on WEAVEBench expose the persistent limitations and challenges of current approaches in multi-turn, context-aware image generation and editing. We believe WEAVE provides a view and foundation for studying in-context interleaved comprehension and generation for multi-modal community.

---

## 67. Q-Doc: Benchmarking Document Image Quality Assessment Capabilities in Multi-modal Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.11410v1](http://arxiv.org/abs/2511.11410v1)

**作者:** Jiaxi Huang, Dongxu Wu, Hanwei Zhu, Lingyu Zhu, Jun Xing, Xu Wang, Baoliang Chen

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了Q-Doc框架，用于系统性地评估多模态大语言模型在文档图像质量评估方面的能力，在三个粒度层次上进行了探索，并发现模型虽有初步能力但存在明显局限性，而思维链提示可显著提升性能。

### 背景

多模态大语言模型(MLLMs)已快速发展并扩展到高级视觉任务之外，但其在文档图像质量评估(DIQA)方面的潜力尚未被充分探索。

### 目的

提出Q-Doc框架，用于系统性地评估MLLMs在DIQA方面的能力，在粗粒度、中等粒度和细粒度三个层次上进行探索。

### 方法

a)粗粒度层面：指导MLLMs为文档图像分配质量分数并分析与质量注释的相关性；b)中等粒度层面：设计失真类型识别任务，包括单选和多选测试；c)细粒度层面：引入失真严重程度评估，让MLLMs根据人工注释参考分类失真强度。

### 主要发现

MLLMs具有初步的DIQA能力，但存在关键局限性：评分不一致、失真误识别和严重程度误判；思维链(CoT)提示在所有层面上都显著提高了性能。

### 结论

该工作为MLLMs的DIQA能力提供了基准，揭示了模型在质量感知方面的明显缺陷，指出了有前景的改进途径，相关基准和代码已在GitHub上公开。

### 翻译

多模态大语言模型(MLLMs)的快速发展已将其能力扩展到高级视觉任务之外。然而，它们在文档图像质量评估(DIQA)方面的潜力仍未被充分探索。为了填补这一空白，我们提出了Q-Doc，一个三层评估框架，用于系统性地探索MLLMs在粗粒度、中等粒度和细粒度水平上的DIQA能力。a)在粗粒度层面，我们指导MLLMs为文档图像分配质量分数，并分析其与质量注释的相关性。b)在中等粒度层面，我们设计了失真类型识别任务，包括多失真场景下的单选和多选测试。c)在细粒度层面，我们引入了失真严重程度评估，让MLLMs根据人工注释的参考来分类失真强度。我们的评估表明，虽然MLLMs具有初步的DIQA能力，但它们表现出关键局限性：评分不一致、失真误识别和严重程度误判。显著的是，我们显示思维链(CoT)提示在所有层面上都显著提高了性能。我们的工作为MLLMs的DIQA能力提供了基准，揭示了它们在质量感知方面的明显缺陷和有前景的改进途径。该基准和代码可在以下网址公开获取：https://github.com/cydxf/Q-Doc。


### 论文摘要

The rapid advancement of Multi-modal Large Language Models (MLLMs) has expanded their capabilities beyond high-level vision tasks. Nevertheless, their potential for Document Image Quality Assessment (DIQA) remains underexplored. To bridge this gap, we propose Q-Doc, a three-tiered evaluation framework for systematically probing DIQA capabilities of MLLMs at coarse, middle, and fine granularity levels. a) At the coarse level, we instruct MLLMs to assign quality scores to document images and analyze their correlation with Quality Annotations. b) At the middle level, we design distortion-type identification tasks, including single-choice and multi-choice tests for multi-distortion scenarios. c) At the fine level, we introduce distortion-severity assessment where MLLMs classify distortion intensity against human-annotated references. Our evaluation demonstrates that while MLLMs possess nascent DIQA abilities, they exhibit critical limitations: inconsistent scoring, distortion misidentification, and severity misjudgment. Significantly, we show that Chain-of-Thought (CoT) prompting substantially enhances performance across all levels. Our work provides a benchmark for DIQA capabilities in MLLMs, revealing pronounced deficiencies in their quality perception and promising pathways for enhancement. The benchmark and code are publicly available at:   https://github.com/cydxf/Q-Doc.

---

## 68. MicroVQA++: High-Quality Microscopy Reasoning Dataset with Weakly Supervised Graphs for Multimodal Large Language Model

**论文链接:** [http://arxiv.org/abs/2511.11407v1](http://arxiv.org/abs/2511.11407v1)

**作者:** Manyu Li, Ruian He, Chenxi Ma, Weimin Tan, Bo Yan

**发布时间:** 2025-11-14

**备注:** 11 pages, 4 figures

### GPT解析

### 总结

研究团队提出了MicroVQA++，一个三阶段构建的大规模高质量显微镜视觉问答语料库，解决了多模态大型语言模型在生物医学显微镜推理中面临的数据稀缺问题。

### 背景

多模态大型语言模型越来越多地应用于生物医学成像，但在显微镜科学推理方面仍然受限于大规模、高质量训练数据的缺乏。

### 目的

创建一个名为MicroVQA++的大规模、高质量显微镜视觉问答语料库，以提升显微镜领域的科学推理能力。

### 方法

采用三阶段构建方法：第一阶段从同行评议文章中专家验证的图表对中引导监督；第二阶段应用HiCQA-Graph异构图融合多种技术识别和过滤不一致样本；第三阶段使用多模态大型语言模型生成多项选择题并进行人工筛选。

### 主要发现

发布的数据集包含大型训练分割和人工检查的测试分割，其布鲁姆级别困难样本分布超过了MicroVQA基准。精心构建的数据使40亿规模的MLLM能够达到具有竞争力的显微镜推理性能，并在开源MLLM中实现最先进的性能。

### 结论

工作提供了质量可控的数据集，将专家文献与基于图的过滤和人工改进相结合；提出了HiCQA-Graph，这是第一个联合建模图像、标题和问答用于跨模态一致性过滤的图；证明了数据构建的精心设计 enables 40B-scale MLLMs达到竞争性的显微镜推理性能。

### 翻译

多模态大型语言模型越来越多地应用于生物医学成像，但在显微镜科学推理方面仍然受限于大规模、高质量训练数据的缺乏。我们引入了MicroVQA++，这是一个从BIOMEDICA档案库中衍生的三阶段、大规模、高质量的显微镜视觉问答语料库。第一阶段从同行评议文章中专家验证的图表对中引导监督。第二阶段应用HiCQA-Graph，这是一种关于图像、标题和问答的新型异构图，它融合了基于NLI的文本蕴含、基于CLIP的视觉语言对齐和代理信号，以识别和过滤不一致样本。第三阶段使用多模态大型语言模型代理生成多项选择题，然后进行人工筛选。发布的结果包含大型训练分割和人工检查的测试分割，其布鲁姆级别困难样本分布超过了MicroVQA基准。我们的工作提供了(i)一个质量可控的数据集，将专家文献与基于图的过滤和人工改进相结合；(ii)HiCQA-Graph，这是第一个联合建模（图像、标题、问答）用于跨模态一致性过滤的图；(iii)证据表明，精心构建的数据使40亿规模的MLLM能够达到具有竞争力的显微镜推理性能（如GPT-5），并在开源MLLM中实现最先进的性能。代码和数据集将在评审过程结束后发布。


### 论文摘要

Multimodal Large Language Models are increasingly applied to biomedical imaging, yet scientific reasoning for microscopy remains limited by the scarcity of large-scale, high-quality training data. We introduce MicroVQA++, a three-stage, large-scale and high-quality microscopy VQA corpus derived from the BIOMEDICA archive. Stage one bootstraps supervision from expert-validated figure-caption pairs sourced from peer-reviewed articles. Stage two applies HiCQA-Graph, a novel heterogeneous graph over images, captions, and QAs that fuses NLI-based textual entailment, CLIP-based vision-language alignment, and agent signals to identify and filter inconsistent samples. Stage three uses a MultiModal Large Language Model (MLLM) agent to generate multiple-choice questions (MCQ) followed by human screening. The resulting release comprises a large training split and a human-checked test split whose Bloom's level hard-sample distribution exceeds the MicroVQA benchmark. Our work delivers (i) a quality-controlled dataset that couples expert literature with graph-based filtering and human refinement; (ii) HiCQA-Graph, the first graph that jointly models (image, caption, QA) for cross-modal consistency filtering; (iii) evidence that careful data construction enables 4B-scale MLLMs to reach competitive microscopy reasoning performance (e.g., GPT-5) and achieve state-of-the-art performance among open-source MLLMs. Code and dataset will be released after the review process concludes.

---

## 69. Multi-Phase Spacecraft Trajectory Optimization via Transformer-Based Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.11402v1](http://arxiv.org/abs/2511.11402v1)

**作者:** Amit Jain, Victor Rodriguez-Fernandez, Richard Linares

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一种基于Transformer的强化学习框架，用于统一多阶段航天器轨迹优化，通过单一策略架构替代传统多阶段控制器，提高适应性和降低操作复杂性。

### 背景

自主航天器控制任务（如发射、上升、级间分离和轨道插入）面临重大挑战，需要能在动态不同区域泛化的自适应策略。现有强化学习方法通常需要为不同任务阶段使用单独的策略，限制了适应性并增加了操作复杂性。

### 目的

引入一个基于Transformer的强化学习框架，通过单一策略架构统一多阶段轨迹优化，利用Transformer固有的建模扩展时间上下文的能力，消除手动阶段转换同时保持控制决策稳定性。

### 方法

在近端策略优化（PPO）基础上构建框架，用Transformer编码器-解码器结构替代传统循环网络，集成门控Transformer-XL（GTrXL）架构，使代理能够在关键操作过程中保持跨任务阶段的一致性记忆。

### 主要发现

Transformer框架在单阶段基准测试上展示接近最优性能，扩展到多阶段路径导航，并成功处理复杂多阶段火箭上升问题。结果表明该框架能在简单情况下匹配解析解，并在动态不同区域学习一致的控制策略。

### 结论

该研究建立了可扩展的自主任务规划基础，减少对特定阶段控制器的依赖，同时保持与安全关键验证协议的兼容性。

### 翻译

自主航天器控制任务如发射、上升、级间分离和轨道插入仍然是一个重大挑战，因为需要能在动态不同区域泛化的自适应策略。虽然强化学习在单个天体动力学任务中显示出前景，但现有方法通常需要为不同任务阶段使用单独的策略，限制了适应性并增加了操作复杂性。这项工作引入了一个基于Transformer的强化学习框架，通过单一策略架构统一多阶段轨迹优化，利用Transformer固有的建模扩展时间上下文的能力。基于近端策略优化（PPO），我们的框架用Transformer编码器-解码器结构替代传统的循环网络，使代理能够在关键操作过程中（持续数秒至数分钟）保持跨任务阶段的一致性记忆。通过集成门控Transformer-XL（GTrXL）架构，该框架消除了手动阶段转换，同时保持控制决策的稳定性。我们逐步验证了我们的方法：首先在单阶段基准测试（双积分器和范德波尔振荡器）上展示接近最优性能，然后扩展到多阶段路径导航变体，最后处理包括大气飞行、级间分离和真空操作的复杂多阶段火箭上升问题。结果表明，Transformer框架不仅能在简单情况下匹配解析解，还能有效地在动态不同的区域学习一致的控制策略，为减少对特定阶段控制器的依赖同时保持与安全关键验证协议的兼容性，建立了可扩展的自主任务规划基础。


### 论文摘要

Autonomous spacecraft control for mission phases such as launch, ascent, stage separation, and orbit insertion remains a critical challenge due to the need for adaptive policies that generalize across dynamically distinct regimes. While reinforcement learning (RL) has shown promise in individual astrodynamics tasks, existing approaches often require separate policies for distinct mission phases, limiting adaptability and increasing operational complexity. This work introduces a transformer-based RL framework that unifies multi-phase trajectory optimization through a single policy architecture, leveraging the transformer's inherent capacity to model extended temporal contexts. Building on proximal policy optimization (PPO), our framework replaces conventional recurrent networks with a transformer encoder-decoder structure, enabling the agent to maintain coherent memory across mission phases spanning seconds to minutes during critical operations. By integrating a Gated Transformer-XL (GTrXL) architecture, the framework eliminates manual phase transitions while maintaining stability in control decisions. We validate our approach progressively: first demonstrating near-optimal performance on single-phase benchmarks (double integrator and Van der Pol oscillator), then extending to multiphase waypoint navigation variants, and finally tackling a complex multiphase rocket ascent problem that includes atmospheric flight, stage separation, and vacuum operations. Results demonstrate that the transformer-based framework not only matches analytical solutions in simple cases but also effectively learns coherent control policies across dynamically distinct regimes, establishing a foundation for scalable autonomous mission planning that reduces reliance on phase-specific controllers while maintaining compatibility with safety-critical verification protocols.

---

## 70. Unsupervised Segmentation of Micro-CT Scans of Polyurethane Structures By Combining Hidden-Markov-Random Fields and a U-Net

**论文链接:** [http://arxiv.org/abs/2511.11378v1](http://arxiv.org/abs/2511.11378v1)

**作者:** Julian Grolig, Lars Griem, Michael Selzer, Hans-Ulrich Kauczor, Simon M. F. Triphan, Britta Nestler, Arnd Koeppe

**发布时间:** 2025-11-14

### GPT解析

### 总结

论文提出了一种结合隐马尔可夫随机场(HMRF)理论和CNN分割的方法，实现了无监督学习和快速分割时间的优势，在聚氨酯泡沫结构的显微CT图像数据集上实现了高分割精度。

### 背景

从图像中提取数字材料表征是材料属性定量分析的必要前提。过去的分割方法通常缺乏准确性或速度。监督卷积神经网络(CNN)虽然取得了最先进性能，但需要大量标记数据。无监督方法不需要真实数据，但分割时间长且精度较差。

### 目的

开发一种结合HMRF理论和CNN分割的方法，利用无监督学习和快速分割时间的优势，研究不同邻域项对无监督HMRF损失的贡献，并减少训练分割模型所需的真实数据量。

### 方法

提出一种整合HMRF理论和CNN分割的方法，研究不同邻域项和组件对无监督HMRF损失的贡献，使用HMRF-UNet模型，并设计一种预训练策略。

### 主要发现

HMRF-UNet能够在聚氨酯泡沫结构的显微CT图像数据集上实现高分割精度，无需真实数据；提出的预训练策略显著减少了训练分割模型所需的真实数据量。

### 结论

结合HMRF理论和CNN分割的方法能够实现无监督学习和快速分割时间的优势，在材料表征分割任务中表现出色。

### 翻译

从图像中提取数字材料表征是材料属性定量分析的必要前提。过去已经广泛研究了不同的分割方法来实现这一任务，但通常缺乏准确性或速度。随着机器学习的出现，监督卷积神经网络(CNN)在不同分割任务上取得了最先进的性能。然而，这些模型通常以监督方式训练，需要大量标记数据集。无监督方法不需要真实数据进行学习，但分割时间长且分割精度通常较差。隐马尔可夫随机场(HMRF)是一种无监督分割方法，结合了邻域和类别分布的概念。我们提出了一种整合HMRF理论和CNN分割的方法，利用两个领域的优势：无监督学习和快速分割时间。我们研究了不同邻域项和组件对无监督HMRF损失的贡献。我们证明HMRF-UNet能够在聚氨酯泡沫结构的显微CT图像数据集上实现高分割精度，无需真实数据。最后，我们提出并演示了一种预训练策略，显著减少了训练分割模型所需的真实数据量。


### 论文摘要

Extracting digital material representations from images is a necessary prerequisite for a quantitative analysis of material properties. Different segmentation approaches have been extensively studied in the past to achieve this task, but were often lacking accuracy or speed. With the advent of machine learning, supervised convolutional neural networks (CNNs) have achieved state-of-the-art performance for different segmentation tasks. However, these models are often trained in a supervised manner, which requires large labeled datasets. Unsupervised approaches do not require ground-truth data for learning, but suffer from long segmentation times and often worse segmentation accuracy. Hidden Markov Random Fields (HMRF) are an unsupervised segmentation approach that incorporates concepts of neighborhood and class distributions. We present a method that integrates HMRF theory and CNN segmentation, leveraging the advantages of both areas: unsupervised learning and fast segmentation times. We investigate the contribution of different neighborhood terms and components for the unsupervised HMRF loss. We demonstrate that the HMRF-UNet enables high segmentation accuracy without ground truth on a Micro-Computed Tomography ($μ$CT) image dataset of Polyurethane (PU) foam structures. Finally, we propose and demonstrate a pre-training strategy that considerably reduces the required amount of ground-truth data when training a segmentation model.

---

## 71. LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.11334v1](http://arxiv.org/abs/2511.11334v1)

**作者:** Jian Gao, Richeng Xuan, Zhaolu Kang, Dingshi Liao, Wenxin Huang, Zongmou Huang, Yangdi Xu, Bowen Qin, Zheqi He, Xi Yang, Changjin Li

**发布时间:** 2025-11-14

### GPT解析

### 总结

研究团队推出了LaoBench，这是首个针对老挝语的大规模、高质量、多维度基准数据集，用于评估大语言模型在老挝语方面的综合语言理解和推理能力。

### 背景

大语言模型(LLMs)快速发展，但在低资源语言(特别是东南亚语言如老挝语)方面的评估不匹配，存在明显的研究空白。

### 目的

填补大语言模型在老挝语等低资源语言评估方面的空白，提供专业的评估基准以促进相关研究和开发。

### 方法

构建包含超过17,000个精心挑选样本的数据集，涵盖知识应用、K12基础教育和老挝语-中文-英语双语翻译三个核心维度；数据集分为开源和闭源子集；采用专家人工筛选与自动化代理辅助验证相结合的数据构建流程。

### 主要发现

通过对多个最先进的大语言模型进行基准测试，发现当前模型在掌握老挝语和多样化任务方面仍面临重大挑战。

### 结论

希望LaoBench能促进对代表性不足的东南亚语言的AI技术研究和开发的进一步发展。

### 翻译

研究团队推出了LaoBench，这是首个针对老挝语的大规模、高质量、多维度基准数据集，用于评估大语言模型在老挝语方面的综合语言理解和推理能力。


### 论文摘要

The rapid advancement of large language models (LLMs) has not been matched by their evaluation in low-resource languages, especially Southeast Asian languages like Lao. To fill this gap, we introduce LaoBench, the first large-scale, high-quality, and multidimensional benchmark dataset dedicated to assessing LLMs' comprehensive language understanding and reasoning abilities in Lao. LaoBench comprises over 17,000 carefully curated samples spanning three core dimensions: knowledge application, K12 foundational education, and bilingual translation among Lao, Chinese, and English. The dataset is divided into open-source and closed-source subsets, with the closed-source portion enabling black-box evaluation on an official platform to ensure fairness and data security. Our data construction pipeline integrates expert human curation with automated agent-assisted verification, ensuring linguistic accuracy, cultural relevance, and educational value. Benchmarking multiple state-of-the-art LLMs on LaoBench reveals that current models still face significant challenges in mastering Lao across diverse tasks. We hope LaoBench will catalyze further research and development of AI technologies for underrepresented Southeast Asian languages.

---

## 72. DocSLM: A Small Vision-Language Model for Long Multimodal Document Understanding

**论文链接:** [http://arxiv.org/abs/2511.11313v1](http://arxiv.org/abs/2511.11313v1)

**作者:** Tanveer Hannan, Dimitrios Mallios, Parth Pathak, Faegheh Sardari, Thomas Seidl, Gedas Bertasius, Mohsen Fayyaz, Sunando Sengupta

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了DocSLM，一种高效的微型视觉语言模型，专为在内存资源受限条件下的长文档理解而设计。该模型通过分层多模态压缩器和流式拒绝机制，显著减少了内存消耗和计算负担，同时保持了高性能。

### 背景

大型视觉语言模型(LVLMs)在处理长而复杂的文档时表现出强大的多模态推理能力，但它们的高内存占用使得它们在资源受限的边缘设备上部署不切实际。

### 目的

提出DocSLM，一个高效的微型视觉语言模型，专为在内存资源受限条件下的长文档理解而设计。

### 方法

DocSLM包含一个分层多模态压缩器，将每页的视觉、文本和布局信息共同编码为固定长度的序列，大大减少了内存消耗，同时保留了局部和全局语义。为了实现对任意长输入的可扩展处理，引入了一种流式拒绝机制，该机制按顺序处理文档段，并使用基于熵的不确定性校准器过滤低置信度响应。

### 主要发现

在多个长多模态文档基准测试中，DocSLM匹配或超越了最先进的方法，同时使用82%更少的视觉令牌，75%更少的参数，以及71%更低的延迟。

### 结论

DocSLM能够在轻量级边缘设备上提供可靠的多模态文档理解。

### 翻译

大型视觉语言模型(LVLMs)在处理长而复杂的文档时已展现出强大的多模态推理能力。然而，它们的高内存占用使得在资源受限的边缘设备上部署变得不切实际。我们提出了DocSLM，一种高效的微型视觉语言模型，专为在内存资源受限条件下的长文档理解而设计。DocSLM包含一个分层多模态压缩器，将每页的视觉、文本和布局信息共同编码为固定长度的序列，大大减少了内存消耗，同时保留了局部和全局语义。为了实现对任意长输入的可扩展处理，我们引入了一种流式拒绝机制，该机制按顺序处理文档段，并使用基于熵的不确定性校准器过滤低置信度响应。在多个长多模态文档基准测试中，DocSLM匹配或超越了最先进的方法，同时使用82%更少的视觉令牌，75%更少的参数，以及71%更低的延迟，在轻量级边缘设备上提供了可靠的多模态文档理解。代码可在补充材料中获取。


### 论文摘要

Large Vision-Language Models (LVLMs) have demonstrated strong multimodal reasoning capabilities on long and complex documents. However, their high memory footprint makes them impractical for deployment on resource-constrained edge devices. We present DocSLM, an efficient Small Vision-Language Model designed for long-document understanding under constrained memory resources. DocSLM incorporates a Hierarchical Multimodal Compressor that jointly encodes visual, textual, and layout information from each page into a fixed-length sequence, greatly reducing memory consumption while preserving both local and global semantics. To enable scalable processing over arbitrarily long inputs, we introduce a Streaming Abstention mechanism that operates on document segments sequentially and filters low-confidence responses using an entropy-based uncertainty calibrator. Across multiple long multimodal document benchmarks, DocSLM matches or surpasses state-of-the-art methods while using 82\% fewer visual tokens, 75\% fewer parameters, and 71\% lower latency, delivering reliable multimodal document understanding on lightweight edge devices. Code is available in the supplementary material.

---

## 73. Large-scale modality-invariant foundation models for brain MRI analysis: Application to lesion segmentation

**论文链接:** [http://arxiv.org/abs/2511.11311v1](http://arxiv.org/abs/2511.11311v1)

**作者:** Petros Koutsouvelis, Matej Gazda, Leroy Volmer, Sina Amirrajab, Kamil Barbierik, Branislav Setlak, Jakub Gazda, Peter Drotar

**发布时间:** 2025-11-14

**备注:** Submitted to IEEE ISBI 2026

### GPT解析

### 总结

该研究提出了一种模态不变的表示学习方法，用于脑部MRI数据的大规模预训练，并在卒中与癫痫病变分割任务中进行了验证。

### 背景

计算机视觉领域正转向通过自监督学习进行大规模基础模型预训练，利用大量未标记的脑部MRI数据可学习解剖学先验知识，提升神经影像任务的少样本性能，但现有SSL框架主要针对自然图像设计，对多模态MRI信息的适应性探索不足。

### 目的

提出一种模态不变的表示学习设置，并评估其在卒中与癫痫病变分割任务中的有效性，该设置基于大规模预训练。

### 方法

设计模态不变的表示学习框架，在大规模预训练的基础上，应用于卒中与癫痫病变分割任务中进行评估。

### 主要发现

尽管模型成功实现了跨模态对齐，但病变分割任务主要受益于保留细粒度的模态特定特征，而非仅依赖模态不变表示。

### 结论

该研究证明了在医学影像处理中保留模态特定特征的重要性，同时公开了模型检查点和代码以促进研究共享。

### 翻译

计算机视觉领域正通过自监督学习向大规模基础模型预训练转变。利用大量未标记的脑部MRI数据，此类模型可学习解剖学先验知识，提高各种神经影像任务中的少样本性能。然而，大多数SSL框架针对自然图像设计，其对捕捉多模态MRI信息的适应性探索仍不充分。本研究提出了一种模态不变的表示学习设置，并在大规模预训练后评估了其在卒中与癫痫病变分割中的有效性。实验结果表明，尽管成功实现了跨模态对齐，但病变分割主要受益于保留细粒度的模态特定特征。模型检查点和代码已公开提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何利用大规模无标签的脑部MRI数据进行预训练，学习模态不变的表示以提高下游病变分割任务性能。这个问题重要是因为脑部MRI分析受限于临床标注数据稀缺，多模态MRI数据的有效整合仍不充分，而自监督学习在医学图像分析中的应用还处于早期阶段。模态不变表示可能减少对特定模态的依赖，提高模型在缺失模态情况下的鲁棒性，并通过学习通用解剖先验知识提高有限数据下的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到计算机视觉正转向大规模基础模型预训练，而医学图像分析面临标注数据稀缺和多模态整合挑战。他们设计方法基于模态不变表示可能减少模态依赖的假设，同时认识到保留模态特定细节对病变分割的重要性。作者借鉴了MoCo v2对比学习框架、掩码图像建模(MIM)目标以及混合Swin编码器等现有工作，但针对医学MRI数据特点进行了专门调整，如定义跨模态正样本对和使用特定预处理流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过自监督学习同时学习模态不变的表示和保留模态特定特征，平衡全局解剖结构和局部病理细节。整体流程包括：1)使用FOMO60k大规模数据集进行标准化预处理；2)实施模态不变对比学习(MCL)，对齐不同模态表示；3)集成掩码图像建模(MIM)保留模态特定特征；4)使用混合Swin编码器进行200k步预训练和5k步微调；5)在三个下游数据集上评估病变分割性能。关键创新在于同时探索模态不变性与模态特定特征的权衡。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)在大规模脑部MRI数据集上实现模态不变预训练；2)专门设计的模态不变对比学习(MCL)框架处理多模态MRI；3)结合MIM目标以保留模态特定特征；4)公开预训练模型促进研究。相比之前工作，不同在于：使用更大规模数据而非小规模实验；同时探索模态不变性和模态特定特征而非仅关注一方面；专注于病变分割任务；不仅评估表示质量还评估对下游任务的实际影响。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合模态不变对比学习和掩码图像建模的大规模自监督预训练框架，证明了虽然跨模态表示对齐可行，但病变分割主要受益于保留模态特定的精细特征而非模态不变性。'}


### 论文摘要

The field of computer vision is undergoing a paradigm shift toward large-scale foundation model pre-training via self-supervised learning (SSL). Leveraging large volumes of unlabeled brain MRI data, such models can learn anatomical priors that improve few-shot performance in diverse neuroimaging tasks. However, most SSL frameworks are tailored to natural images, and their adaptation to capture multi-modal MRI information remains underexplored. This work proposes a modality-invariant representation learning setup and evaluates its effectiveness in stroke and epilepsy lesion segmentation, following large-scale pre-training. Experimental results suggest that despite successful cross-modality alignment, lesion segmentation primarily benefits from preserving fine-grained modality-specific features. Model checkpoints and code are made publicly available.

---

## 74. EcoAlign: An Economically Rational Framework for Efficient LVLM Alignment

**论文链接:** [http://arxiv.org/abs/2511.11301v1](http://arxiv.org/abs/2511.11301v1)

**作者:** Ruoxi Cheng, Haoxuan Ma, Teng Ma, Hongyi Zhang

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一种名为EcoAlign的推理时框架，用于解决大型视觉-语言模型(LVLMs)的对齐问题，通过经济理性搜索方法在安全性、实用性和计算成本之间取得平衡。

### 背景

大型视觉-语言模型(LVLMs)展现出强大的推理能力，但存在复杂的越狱漏洞。当前对齐方法难以在安全性、实用性和运营成本之间取得平衡，且仅关注最终输出会浪费计算预算在危险的推理上。

### 目的

开发一种能够有效平衡安全性、实用性和计算成本的LVLMs对齐方法，防止有害推理以良性借口伪装并规避安全检测。

### 方法

提出EcoAlign框架，将LVLM视为有限理性代理，通过对齐作为经济理性搜索。该方法增量扩展思维图，使用前瞻性函数(类似于净现值)对行动进行评分，动态权衡预期安全性、实用性和成本与剩余预算，并通过最弱链原则强制执行路径安全以防止欺骗。

### 主要发现

在3个闭源和2个开源模型上的6个数据集进行的广泛实验表明，EcoAlign在较低计算成本下匹配或超越了最先进的安全性和实用性。

### 结论

EcoAlign为稳健的LVLM对齐提供了原则性、经济性的途径，解决了传统对齐方法在安全性、实用性和成本之间的权衡问题。

### 翻译

大型视觉-语言模型(LVLMs)展现出强大的推理能力，但存在复杂的越狱漏洞。从根本上说，对齐LVLMs不仅是安全挑战，也是经济效率问题。当前对齐方法难以在安全性、实用性和运营成本之间取得平衡。关键的是，仅关注最终输出(过程盲目)会浪费大量计算预算在危险的推理上。这一缺陷允许有害推理以良性借口伪装，从而规避简单的加性安全评分。为解决这一问题，我们提出了EcoAlign，一种推理时框架，通过将LVLM视为有限理性代理，将对齐重新构建为经济理性搜索。EcoAlign增量扩展思维图，并使用前瞻性函数(类似于净现值)对行动进行评分，动态权衡预期安全性、实用性和成本与剩余预算。为防止欺骗，通过最弱链原则强制执行路径安全。在3个闭源和2个开源模型上的6个数据集进行的广泛实验表明，EcoAlign在较低计算成本下匹配或超越了最先进的安全性和实用性，从而为稳健的LVLM对齐提供了原则性、经济性的途径。


### 论文摘要

Large Vision-Language Models (LVLMs) exhibit powerful reasoning capabilities but suffer sophisticated jailbreak vulnerabilities. Fundamentally, aligning LVLMs is not just a safety challenge but a problem of economic efficiency. Current alignment methods struggle with the trade-off between safety, utility, and operational costs. Critically, a focus solely on final outputs (process-blindness) wastes significant computational budget on unsafe deliberation. This flaw allows harmful reasoning to be disguised with benign justifications, thereby circumventing simple additive safety scores. To address this, we propose EcoAlign, an inference-time framework that reframes alignment as an economically rational search by treating the LVLM as a boundedly rational agent. EcoAlign incrementally expands a thought graph and scores actions using a forward-looking function (analogous to net present value) that dynamically weighs expected safety, utility, and cost against the remaining budget. To prevent deception, path safety is enforced via the weakest-link principle. Extensive experiments across 3 closed-source and 2 open-source models on 6 datasets show that EcoAlign matches or surpasses state-of-the-art safety and utility at a lower computational cost, thereby offering a principled, economical pathway to robust LVLM alignment.

---

## 75. AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.11299v1](http://arxiv.org/abs/2511.11299v1)

**作者:** Haokun Chen, Jianing Li, Yao Zhang, Jinhe Bi, Yan Xia, Jindong Gu, Volker Tresp

**发布时间:** 2025-11-14

**备注:** AAAI 2026. Code: https://github.com/HaokunChen245/AUVIC

### GPT解析

### 总结

本文介绍了一种名为AUVIC的新型视觉概念遗忘框架，用于多模态大语言模型(MLLMs)，通过对抗性扰动实现精确遗忘，有效隔离目标概念同时避免对相似实体的意外影响。

### 背景

多模态大语言模型在大量数据优化后表现优异，但这些数据常含敏感或版权内容，引发隐私问题。'被遗忘权'的监管需求推动了机器遗忘技术，但MLLMs中的视觉概念遗忘研究仍不足。

### 目的

开发一种能精确移除MLLMs中目标视觉概念而不影响相关实体模型性能的方法，解决视觉概念遗忘领域的研究空白。

### 方法

提出AUVIC框架，通过应用对抗性扰动实现精确遗忘。为评估该方法，构建了VCUBench基准测试，这是首个针对群体环境中视觉概念遗忘评估的基准。

### 主要发现

实验结果表明，AUVIC在目标遗忘率方面达到最先进水平，同时对非目标概念的性能影响最小。

### 结论

AUVIC框架为MLLMs中的视觉概念遗忘提供了有效解决方案，能在精确移除目标概念的同时保持模型在相关实体上的性能。

### 翻译

多模态大语言模型(MLLMs)一旦在大量数据上优化就能取得令人印象深刻的性能。这类数据通常包含敏感或受版权保护的内容，引发重大数据隐私问题。要求'被遗忘权'的监管框架推动了对机器遗忘的需求。这项技术允许在不消耗资源重新训练的情况下移除目标数据。然而，虽然文本遗忘已被充分研究，MLLMs中的视觉概念遗忘仍探索不足。主要挑战是精确移除目标视觉概念而不干扰模型对相关实体的性能。为此，我们引入AUVIC，一种新颖的MLLM视觉概念遗忘框架。AUVIC应用对抗性扰动实现精确遗忘，有效隔离目标概念同时避免对相似实体的意外影响。为评估我们的方法，我们构建了VCUBench，它是首个旨在评估群体环境中视觉概念遗忘的基准测试。实验结果表明，AUVIC在目标遗忘率方面达到最先进水平，同时对非目标概念性能影响最小。


### 论文摘要

Multimodal Large Language Models (MLLMs) achieve impressive performance once optimized on massive datasets. Such datasets often contain sensitive or copyrighted content, raising significant data privacy concerns. Regulatory frameworks mandating the 'right to be forgotten' drive the need for machine unlearning. This technique allows for the removal of target data without resource-consuming retraining. However, while well-studied for text, visual concept unlearning in MLLMs remains underexplored. A primary challenge is precisely removing a target visual concept without disrupting model performance on related entities. To address this, we introduce AUVIC, a novel visual concept unlearning framework for MLLMs. AUVIC applies adversarial perturbations to enable precise forgetting. This approach effectively isolates the target concept while avoiding unintended effects on similar entities. To evaluate our method, we construct VCUBench. It is the first benchmark designed to assess visual concept unlearning in group contexts. Experimental results demonstrate that AUVIC achieves state-of-the-art target forgetting rates while incurs minimal performance degradation on non-target concepts.

---

## 76. Experiences from Benchmarking Vision-Language-Action Models for Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2511.11298v1](http://arxiv.org/abs/2511.11298v1)

**作者:** Yihao Zhang, Yuankai Qi, Xi Zheng

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究对四种代表性视觉-语言-行动模型在机器人操作任务中的性能进行了系统评估，建立了标准化评估框架，发现了不同模型在适应性、稳定性和计算需求等方面的差异，为实际应用提供了重要参考。

### 背景

基础模型在机器人领域的应用，特别是视觉-语言-行动模型在通用操作方面有很大潜力，但系统性的真实世界评估和跨模型比较仍然缺乏。

### 目的

报告对四种代表性VLA模型（ACT、OpenVLA-OFT、RDT-1B和π₀）在模拟和ALOHA Mobile平台上进行的四个操作任务的基准测试经验。

### 方法

建立一个标准化的评估框架，从三个关键维度衡量性能：准确性和效率（成功率和达到成功所需时间）、适应性（包括分布内、空间分布外和实例加空间分布外设置）、语言指令遵循准确性。

### 主要发现

π₀在分布外场景中表现出卓越的适应性，ACT在分布内提供了最高的稳定性；分析揭示了计算需求、数据扩展行为和重复出现的失败模式（如接近抓取、过早释放和长时程状态漂移）的差异。

### 结论

这些发现揭示了VLA模型架构在平衡精度、泛化能力和部署成本方面的实际权衡，为在真实世界机器人操作任务中选择和部署VLA提供了可行的见解。

### 翻译

应用于机器人的基础模型，特别是视觉-语言-行动模型，在实现通用操作方面展现出巨大潜力。然而，系统性的真实世界评估和跨模型比较仍然稀缺。本文报告了我们在模拟和ALOHA Mobile平台上对四个代表性VLA模型（ACT、OpenVLA-OFT、RDT-1B和π₀）进行四个操作任务基准测试的经验。我们建立了一个标准化的评估框架，从三个关键维度衡量性能：（1）准确性和效率（成功率和达到成功所需时间），（2）在分布内、空间分布外和实例加空间分布外设置下的适应性，（3）语言指令遵循准确性。通过这一过程，我们观察到π₀在分布外场景中表现出卓越的适应性，而ACT在分布内提供了最高的稳定性。进一步的分析突显了计算需求、数据扩展行为和重复出现的失败模式（如接近抓取、过早释放和长时程状态漂移）之间的差异。这些发现揭示了VLA模型架构在平衡精度、泛化能力和部署成本方面的实际权衡，为在真实世界机器人操作任务中选择和部署VLA提供了可行的见解。


### 论文摘要

Foundation models applied in robotics, particularly \textbf{Vision--Language--Action (VLA)} models, hold great promise for achieving general-purpose manipulation. Yet, systematic real-world evaluations and cross-model comparisons remain scarce. This paper reports our \textbf{empirical experiences} from benchmarking four representative VLAs -- \textbf{ACT}, \textbf{OpenVLA--OFT}, \textbf{RDT-1B}, and \boldmath{$π_0$} -- across four manipulation tasks conducted in both simulation and on the \textbf{ALOHA Mobile} platform. We establish a \textbf{standardized evaluation framework} that measures performance along three key dimensions: (1) \textit{accuracy and efficiency} (success rate and time-to-success), (2) \textit{adaptability} across in-distribution, spatial out-of-distribution, and instance-plus-spatial out-of-distribution settings, and (3) \textit{language instruction-following accuracy}. Through this process, we observe that \boldmath{$π_0$} demonstrates superior adaptability in out-of-distribution scenarios, while \textbf{ACT} provides the highest stability in-distribution. Further analysis highlights differences in computational demands, data-scaling behavior, and recurring failure modes such as near-miss grasps, premature releases, and long-horizon state drift. These findings reveal practical trade-offs among VLA model architectures in balancing precision, generalization, and deployment cost, offering actionable insights for selecting and deploying VLAs in real-world robotic manipulation tasks.

---

## 77. Toward Scalable Early Cancer Detection: Evaluating EHR-Based Predictive Models Against Traditional Screening Criteria

**论文链接:** [http://arxiv.org/abs/2511.11293v1](http://arxiv.org/abs/2511.11293v1)

**作者:** Jiheum Park, Chao Pang, Tristan Y. Lee, Jeong Yun Yang, Jacob Berkowitz, Alexander Z. Wei, Nicholas Tatonetti

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究评估了基于电子健康记录的预测模型与传统风险因素在识别癌症高风险个体方面的临床效用，发现EHR模型显著提高了癌症病例的识别率。

### 背景

当前癌症筛查指南仅涵盖少数癌症类型，依赖年龄或单一风险因素等狭义标准。基于EHR的预测模型可捕获大规模纵向健康信息，检测癌症预诊断信号，但与传统方法相比的有效性证据有限。

### 目的

系统评估基于EHR的预测模型与传统风险因素相比，在识别八种主要癌症高风险个体方面的临床效用。

### 方法

使用All of Us研究项目数据，该项目整合了超过865,000名参与者的电子健康记录、基因组和调查数据，比较EHR模型与传统风险因素的表现。

### 主要发现

即使使用基础建模方法，EHR模型也比传统风险因素单独使用时，在识别高风险个体中真实癌症病例的富集率提高了3至6倍。最先进的EHR基础模型进一步提高了26种癌症类型的预测性能。

### 结论

基于EHR的预测建模具有临床潜力，可支持更精确和可扩展的癌症早期检测策略。

### 翻译

当前的癌症筛查指南仅涵盖少数几种癌症类型，并依赖于狭义定义的标准（如年龄或单一风险因素如吸烟史）来识别高风险个体。使用电子健康记录的预测模型可以捕获大规模纵向患者级别的健康信息，可能通过检测癌症的细微预诊断信号，为识别高风险群体提供更有效的工具。大型语言模型和基础模型的最新进展进一步扩展了这一潜力，但关于EHR模型与筛查指南中使用的传统风险因素相比的有用性的证据仍然有限。我们使用All of Us研究项目的数据（该项目整合了超过865,000名参与者的电子健康记录、基因组和调查数据），系统评估了基于EHR的预测模型与传统风险因素（包括基因突变和癌症家族史）相比在识别八种主要癌症（乳腺癌、肺癌、结直肠癌、前列腺癌、卵巢癌、肝癌、胰腺癌和胃癌）高风险个体方面的临床效用。即使使用基础建模方法，基于EHR的模型也比仅使用传统风险因素时，在识别为高风险的个体中真实癌症病例的富集率提高了3至6倍，无论是作为独立工具还是补充工具。基于EHR的基础模型（一种在全面患者轨迹上训练的最先进方法）进一步提高了26种癌症类型的预测性能，证明了基于EHR的预测建模支持更精确和可扩展的早期检测策略的临床潜力。


### 论文摘要

Current cancer screening guidelines cover only a few cancer types and rely on narrowly defined criteria such as age or a single risk factor like smoking history, to identify high-risk individuals. Predictive models using electronic health records (EHRs), which capture large-scale longitudinal patient-level health information, may provide a more effective tool for identifying high-risk groups by detecting subtle prediagnostic signals of cancer. Recent advances in large language and foundation models have further expanded this potential, yet evidence remains limited on how useful HER-based models are compared with traditional risk factors currently used in screening guidelines. We systematically evaluated the clinical utility of EHR-based predictive models against traditional risk factors, including gene mutations and family history of cancer, for identifying high-risk individuals across eight major cancers (breast, lung, colorectal, prostate, ovarian, liver, pancreatic, and stomach), using data from the All of Us Research Program, which integrates EHR, genomic, and survey data from over 865,000 participants. Even with a baseline modeling approach, EHR-based models achieved a 3- to 6-fold higher enrichment of true cancer cases among individuals identified as high risk compared with traditional risk factors alone, whether used as a standalone or complementary tool. The EHR foundation model, a state-of-the-art approach trained on comprehensive patient trajectories, further improved predictive performance across 26 cancer types, demonstrating the clinical potential of EHR-based predictive modeling to support more precise and scalable early detection strategies.

---

## 78. Φeat: Physically-Grounded Feature Representation

**论文链接:** [http://arxiv.org/abs/2511.11270v1](http://arxiv.org/abs/2511.11270v1)

**作者:** Giuseppe Vecchio, Adrien Kaiser, Rouffet Romain, Rosalie Martin, Elena Garces, Tamy Boubekeur

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为Φeat的新型物理驱动的视觉骨干模型，通过自监督学习方式分离高级语义与低级物理因素，使特征表示对材料身份敏感，包括反射线索和几何亚结构。

### 背景

基础模型已成为许多视觉任务的有效骨干，但当前自监督特征将高级语义与低级物理因素（如几何和光照）纠缠在一起，阻碍了它们在需要明确物理推理的任务中的使用。

### 目的

引入Φeat，一种新型物理驱动的视觉骨干，鼓励对材料身份敏感的表示，包括反射线索和几何亚结构。

### 方法

采用预训练策略，对比同一材料在不同形状和光照条件下的空间裁剪和物理增强，使用纯自监督训练策略，无需明确标签，提供对外部物理因素具有鲁棒性的特征先验。

### 主要发现

通过特征相似性分析和材料选择评估表明，Φeat捕获了超越语义分组的物理驱动结构，证明了无监督物理特征学习作为视觉和图形中物理感知基础的前景。

### 结论

无监督物理特征学习在视觉和图形中物理感知方面具有广阔前景，可作为物理感知感知的基础。

### 翻译

基础模型已成为许多视觉任务的有效骨干。然而，当前自监督特征将高级语义与低级物理因素（如几何和光照）纠缠在一起，阻碍了它们在需要明确物理推理的任务中的使用。在本文中，我们引入Φeat，一种新型物理驱动的视觉骨干，鼓励对材料身份敏感的表示，包括反射线索和几何亚结构。我们的关键想法是采用一种预训练策略，对比同一材料在不同形状和光照条件下的空间裁剪和物理增强。虽然类似数据已被用于高端监督任务，如内在分解或材料估计，但我们证明纯自监督训练策略，无需明确标签，已经为需要对外部物理因素具有鲁棒性的特征的任务提供了强大的先验。我们通过特征相似性分析和材料选择评估学习的表示，表明Φeat捕获了超越语义分组的物理驱动结构。这些发现突显了无监督物理特征学习作为视觉和图形中物理感知基础的前景。


### 论文摘要

Foundation models have emerged as effective backbones for many vision tasks. However, current self-supervised features entangle high-level semantics with low-level physical factors, such as geometry and illumination, hindering their use in tasks requiring explicit physical reasoning. In this paper, we introduce $Φ$eat, a novel physically-grounded visual backbone that encourages a representation sensitive to material identity, including reflectance cues and geometric mesostructure. Our key idea is to employ a pretraining strategy that contrasts spatial crops and physical augmentations of the same material under varying shapes and lighting conditions. While similar data have been used in high-end supervised tasks such as intrinsic decomposition or material estimation, we demonstrate that a pure self-supervised training strategy, without explicit labels, already provides a strong prior for tasks requiring robust features invariant to external physical factors. We evaluate the learned representations through feature similarity analysis and material selection, showing that $Φ$eat captures physically-grounded structure beyond semantic grouping. These findings highlight the promise of unsupervised physical feature learning as a foundation for physics-aware perception in vision and graphics. These findings highlight the promise of unsupervised physical feature learning as a foundation for physics-aware perception in vision and graphics.

---

## 79. Boundary Compactified Imaginary Liouville Theory

**论文链接:** [http://arxiv.org/abs/2511.11269v1](http://arxiv.org/abs/2511.11269v1)

**作者:** Yang Xiao, Yuxiao Xie

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究将紧致化虚李亚普诺夫理论（CILT）从闭曲面推广到有边界的曲面，通过从具有诺伊曼边界条件的紧致化高斯自由场出发，添加曲率项和指数势能进行扰动，并证明了所得概率路径积分满足共形场论公理。

### 背景

紧致化虚李亚普诺夫理论是一种定义在闭曲面上的非单位对数共形场论，在物理上被认为描述了Potts模型和O(n)模型等回路模型的标度极限。

### 目的

将CILT构造推广到有边界的曲面，为边界CILT的研究奠定基础，并帮助理解CILT理论。

### 方法

从具有诺伊曼边界条件的紧致化高斯自由场出发，在体区域和边界上添加曲率项和指数势能进行扰动，使用虚高斯乘法混沌定义势能项，并证明满足Segal的拼接公理。

### 主要发现

成功将CILT构造扩展到有边界曲面，证明了理论满足共形场论的基本公理，建立了与物理回路模型之间的联系。

### 结论

该工作为边界CILT的进一步研究提供了理论基础，有助于更深入地理解CILT理论。

### 翻译

我们将紧致化虚李亚普诺夫理论（CILT）的构造推广到有边界的曲面，这是一种定义在闭曲面上的非单位对数共形场论。我们从具有诺伊曼边界条件的紧致化高斯自由场出发，通过在体区域和边界上添加曲率项和指数势能进行扰动。在物理上，该理论被认为描述了Potts模型和O(n)模型等回路模型的标度极限。为了数学定义这一理论，曲率项需要详细分析拓扑结构，势能项则使用虚高斯乘法混沌（GMC）定义。我们证明了所得的概率路径积分满足共形场论的公理，包括Segal的拼接公理。这项工作为边界CILT的后续研究奠定了基础，也将有助于理解CILT理论。


### 论文摘要

We generalize the construction of Compactified Imaginary Liouville Theory (CILT), a non-unitary logarithmic Conformal Field Theory (CFT) defined on closed surfaces, to surfaces with boundary. Starting from a compactified Gaussian Free Field (GFF) with Neumann boundary condition, we perturb it by adding in curvature terms and exponential potentials on both the bulk and the boundary. In physics, this theory is conjectured to describe the scaling limit of loop models such as the Potts and $O(n)$ models. To define it mathematically, the curvature terms require a detailed analysis of the topology, and the potential terms are defined using the imaginary Guassian Multiplicative Chaos (GMC). We prove that the resulting probabilistic path integral satisfies the axioms of CFT, including Segal's gluing axioms. This work provides the foundation for future studies of boundary CILT and will also help with the understanding of CILT.

---

## 80. GraphPilot: Grounded Scene Graph Conditioning for Language-Based Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.11266v1](http://arxiv.org/abs/2511.11266v1)

**作者:** Fabian Schmidt, Markus Enzweiler, Abhinav Valada

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为GraphPilot的新方法，通过交通场景图的结构化关系上下文增强基于语言的驾驶模型，显著提高了自动驾驶性能。

### 背景

视觉-语言模型最近成为自动驾驶的有前途的规划器，其成功依赖于对多模态输入的空间结构和动态交互进行拓扑感知推理。然而，现有模型通常在没有明确编码这些关系依赖的监督下进行训练，限制了它们从原始传感器数据中推断代理和其他交通实体如何相互影响的能力。

### 目的

开发一种与模型无关的方法，将基于语言的驾驶模型条件化为交通场景图形式的结构化关系上下文，以弥合现有模型与明确关系监督需求之间的差距。

### 方法

将交通场景图以各种抽象级别和格式序列化，并通过结构化提示模板将其整合到模型中，从而能够系统分析关系监督何时以及如何最有益。

### 主要发现

在公共LangAuto基准上的评估表明，场景图调节使最先进方法的驾驶性能得到显著且持久的改进，LMDrive的驾驶分数提高了高达15.6%，BEVDriver提高了17.5%。这表明模型可以通过场景图调节训练更好地内化和关系先验，即使在测试时不需要场景图输入。

### 结论

通过场景图调节训练，模型能够更好地理解和利用交通场景中的关系信息，从而显著提高自动驾驶性能，即使在实际应用中不需要显式提供场景图。

### 翻译

视觉-语言模型最近已成为自动驾驶的有前途的规划器，其成功依赖于对多模态输入的空间结构和动态交互进行拓扑感知推理。然而，现有模型通常在没有明确编码这些关系依赖的监督下进行训练，限制了它们从原始传感器数据中推断代理和其他交通实体如何相互影响的能力。在这项工作中，我们通过一种新颖的与模型无关的方法弥合这一差距，该方法将基于语言的驾驶模型条件化为交通场景图形式的结构化关系上下文。我们以各种抽象级别和格式序列化场景图，并通过结构化提示模板将它们整合到模型中，从而能够系统分析关系监督何时以及如何最有益。在公共LangAuto基准上的广泛评估表明，最先进方法的场景图调节在驾驶性能上产生了巨大且持久的改进。值得注意的是，我们观察到LMDrive的驾驶分数提高了高达15.6%，BEVDriver提高了17.5%，这表明模型可以通过场景图调节训练更好地内化和关系先验，即使在测试时不需要场景图输入。代码、微调模型和我们的场景图数据集可在https://github.com/iis-esslingen/GraphPilot公开获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的问题是现有基于语言的自动驾驶模型缺乏明确的拓扑感知推理能力，无法从原始传感器数据中推断出交通实体之间的关系和动态交互。这个问题在现实中非常重要，因为自动驾驶系统需要准确理解交通环境中各种实体间的互动关系（如哪辆车在哪个车道、交通灯控制方向等）才能做出安全有效的驾驶决策，避免交通事故。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有模型在复杂环境中表现有限，原因是它们依赖的表示没有显式编码关系结构。他们借鉴了场景图在室内导航中的应用，将其扩展到自动驾驶领域。设计上选择序列化场景图并通过提示模板整合到模型中，而非修改架构或添加专门组件，保持了模型无关性。作者借鉴了现有语言模型在自动驾驶中的应用（如LMDrive、BEVDriver）和场景图的构建方法，但首次将其用于语言驱动的驾驶规划。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将交通场景图作为显式结构化关系上下文注入语言模型，使模型能更好地理解和推理交通场景中的空间结构、规则和参与者交互。实现流程包括：1)构建场景图（检测节点和关系，提供三种抽象级别）；2)序列化场景图为文本、JSON或YAML格式；3)设计提示模板将场景图与导航指令结合；4)通过提示级别条件化使模型基于结构化关系信息进行决策。整个过程无需修改模型架构，仅在输入层面操作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次将交通场景图作为显式上下文注入语言驱动驾驶模型；提出模型无关的提示级别条件化方法；系统评估不同抽象级别和格式；发现模型可在训练期间内化关系知识无需测试时场景图；提出训练时用场景图监督而测试时无需的新范式。相比之前工作，不同之处在于：现有方法使用密集视觉特征而非显式关系结构；现有场景图应用主要用于风险估计而非语言规划；GraphPilot证明训练期间的关系监督比仅测试时注入更有效且持久。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GraphPilot提出了一种模型无关的方法，通过将交通场景图作为显式结构化上下文注入语言驱动的自动驾驶模型中，显著提升了模型在复杂交通场景中的推理和决策能力，同时证明了模型可以在训练期间内化关系知识，无需在测试时提供场景图即可实现高性能驾驶。'}


### 论文摘要

Vision-language models have recently emerged as promising planners for autonomous driving, where success hinges on topology-aware reasoning over spatial structure and dynamic interactions from multimodal input. However, existing models are typically trained without supervision that explicitly encodes these relational dependencies, limiting their ability to infer how agents and other traffic entities influence one another from raw sensor data. In this work, we bridge this gap with a novel model-agnostic method that conditions language-based driving models on structured relational context in the form of traffic scene graphs. We serialize scene graphs at various abstraction levels and formats, and incorporate them into the models via structured prompt templates, enabling a systematic analysis of when and how relational supervision is most beneficial. Extensive evaluations on the public LangAuto benchmark show that scene graph conditioning of state-of-the-art approaches yields large and persistent improvement in driving performance. Notably, we observe up to a 15.6\% increase in driving score for LMDrive and 17.5\% for BEVDriver, indicating that models can better internalize and ground relational priors through scene graph-conditioned training, even without requiring scene graph input at test-time. Code, fine-tuned models, and our scene graph dataset are publicly available at https://github.com/iis-esslingen/GraphPilot.

---

## 81. AIonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery

**论文链接:** [http://arxiv.org/abs/2511.11257v1](http://arxiv.org/abs/2511.11257v1)

**作者:** Yuqi Yin, Yibo Fu, Siyuan Wang, Peng Sun, Hongyu Wang, Xiaohui Wang, Lei Zheng, Zhiyong Li, Zhirong Liu, Jianji Wang, Zhaoxi Sun

**发布时间:** 2025-11-14

### GPT解析

### 总结

研究团队开发了AIonopedia，这是首个用于离子液体发现的LLM智能体，利用大型语言模型和增强的多模态领域基础模型进行准确的属性预测和分子设计，通过实际验证展现出强大的泛化能力。

### 背景

新型离子液体的发现面临关键挑战，包括数据有限、模型准确性差和工作流程分散等问题，阻碍了新型离子液体的发现进程。

### 目的

开发一个基于大型语言模型的智能系统，克服离子液体发现中的预测挑战，加速离子液体的实际发现过程。

### 方法

引入AIonopedia智能体，采用LLM增强的多模态领域基础模型，结合分层搜索架构进行分子筛选和设计，在综合数据集上训练评估，并通过湿实验室验证实际效果。

### 主要发现

1) 模型在综合数据集上表现出优越性能；2) 智能体能有效进行离子液体修饰；3) 在具有挑战性的分布外任务上展现出卓越的泛化能力。

### 结论

AIonopeda作为首个用于离子液体发现的LLM智能体，能够加速实际的离子液体发现过程，并在复杂任务上展现出强大的泛化能力，具有实际应用价值。

### 翻译

新型离子液体(ILs)的发现受到属性预测中的关键挑战阻碍，包括数据有限、模型准确性差和工作流程分散等问题。利用大型语言模型(LLMs)的力量，我们引入了AIonopedia，据我们所知，这是首个用于IL发现的LLM智能体。由一个增强的多模态领域基础模型驱动，AIonopedia能够进行准确的属性预测，并包含用于分子筛选和设计的分层搜索架构。在一个新整理的综合IL数据集上训练和评估后，我们的模型表现出优越的性能。补充这些结果，在文献报道的系统上的评估表明，该智能体能够有效地进行IL修饰。超越离线测试，实际功效通过实际湿实验室验证得到进一步确认，在该验证中，智能体在具有挑战性的分布外任务上表现出卓越的泛化能力，强调了其加速实际IL发现的能力。


### 论文摘要

The discovery of novel Ionic Liquids (ILs) is hindered by critical challenges in property prediction, including limited data, poor model accuracy, and fragmented workflows. Leveraging the power of Large Language Models (LLMs), we introduce AIonopedia, to the best of our knowledge, the first LLM agent for IL discovery. Powered by an LLM-augmented multimodal domain foundation model for ILs, AIonopedia enables accurate property predictions and incorporates a hierarchical search architecture for molecular screening and design. Trained and evaluated on a newly curated and comprehensive IL dataset, our model delivers superior performance. Complementing these results, evaluations on literature-reported systems indicate that the agent can perform effective IL modification. Moving beyond offline tests, the practical efficacy was further confirmed through real-world wet-lab validation, in which the agent demonstrated exceptional generalization capabilities on challenging out-of-distribution tasks, underscoring its ability to accelerate real-world IL discovery.

---

## 82. CountSteer: Steering Attention for Object Counting in Diffusion Models

**论文链接:** [http://arxiv.org/abs/2511.11253v1](http://arxiv.org/abs/2511.11253v1)

**作者:** Hyemin Boo, Hyoryung Kim, Myungjin Lee, Seunghyeon Lee, Jiyoung Lee, Jang-Hwan Choi, Hyunsoo Cho

**发布时间:** 2025-11-14

**备注:** Accepted to AAAI 2026 Workshop on Shaping Responsible Synthetic Data in the Era of Foundation Models (RSD)

### GPT解析

### 总结

研究人员发现文本到图像扩散模型虽然能生成真实图像，但在遵循数字指令方面存在不足。通过引导模型的交叉注意力状态，他们提出了一种无需训练的方法CountSteer，可提高对象计数准确性而不损害视觉质量。

### 背景

文本到图像扩散模型能够生成真实且连贯的图像，但通常无法遵循文本中的数字指令，表明语言和视觉表示之间存在差距。

### 目的

探索如何利用模型内部对数值正确性的隐式理解，提高文本到图像生成中对对象计数的控制能力。

### 方法

提出CountSteer，一种无需训练的方法，通过在推理过程中引导模型的交叉注意力隐藏状态来改进指定对象计数的生成。

### 主要发现

扩散模型并非完全对数字不敏感，它们隐式了解自己的计数准确性，内部信号会根据输出是否符合指定计数而一致变化，表明模型已编码了潜在的数值正确性概念。

### 结论

CountSteer在不损害视觉质量的情况下将对象计数准确性提高了约4%，是朝着更可控和语义上可靠的文本到图像生成迈出的简单而有效的一步。

### 翻译

文本到图像扩散模型能够生成真实且连贯的图像，但通常无法遵循文本中的数字指令，这揭示了语言和视觉表示之间的差距。有趣的是，我们发现这些模型并非完全对数字不敏感 - 它们隐式地了解自己的计数准确性，因为它们的内部信号会根据输出是否符合指定的计数而以一致的方式变化。这一观察表明，模型已经编码了一种潜在的数值正确性概念，可以利用它来更精确地指导生成过程。基于这一直觉，我们引入了CountSteer，一种无需训练的方法，通过在推理过程中引导模型的交叉注意力隐藏状态来改进指定对象计数的生成。在我们的实验中，CountSteer在不损害视觉质量的情况下，将对象计数准确性提高了约4%，这表明CountSteer是朝着更可控和语义上可靠的文本到图像生成迈出的简单而有效的一步。


### 论文摘要

Text-to-image diffusion models generate realistic and coherent images but often fail to follow numerical instructions in text, revealing a gap between language and visual representation. Interestingly, we found that these models are not entirely blind to numbers-they are implicitly aware of their own counting accuracy, as their internal signals shift in consistent ways depending on whether the output meets the specified count. This observation suggests that the model already encodes a latent notion of numerical correctness, which can be harnessed to guide generation more precisely. Building on this intuition, we introduce CountSteer, a training-free method that improves generation of specified object counts by steering the model's cross-attention hidden states during inference. In our experiments, CountSteer improved object-count accuracy by about 4% without compromising visual quality, demonstrating a simple yet effective step toward more controllable and semantically reliable text-to-image generation.

---

## 83. UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios

**论文链接:** [http://arxiv.org/abs/2511.11252v1](http://arxiv.org/abs/2511.11252v1)

**作者:** Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah

**发布时间:** 2025-11-14

**备注:** 18 pages, 5 Figures

### GPT解析

### 总结

该研究引入了UAVBench，一个包含50,000个验证过的UAV飞行场景的开放基准数据集，以及UAVBench_MCQ，一个包含50,000个多项选择题的推理导向扩展，用于评估大型语言模型在自主空中系统中的推理能力。

### 背景

自主空中系统越来越多地依赖大型语言模型进行任务规划、感知和决策，但缺乏标准化和物理基础的基准限制了对其推理能力的系统性评估。

### 目的

解决缺乏标准化基准的问题，提供一个可评估LLMs在自主空中系统中推理能力的框架，促进开放科学和可重现性。

### 方法

通过分类指导的LLM提示和多阶段安全验证生成50,000个UAV飞行场景；每个场景使用结构化JSON模式编码；构建包含十种认知和伦理推理风格的50,000个多项选择题；评估32个最先进的LLMs。

### 主要发现

在感知和策略推理方面表现出色，但在道德感知和资源受限决策方面仍存在持续挑战。

### 结论

UAVBench为自主空中系统中智能AI的基准测试提供了可重现和物理基础，推进了下一代UAV推理智能的发展。

### 翻译

自主空中系统越来越多地依赖大型语言模型进行任务规划、感知和决策，然而缺乏标准化和物理基础的基准限制了对其推理能力的系统性评估。为解决这一差距，我们引入了UAVBench，一个包含50,000个验证过的UAV飞行场景的开放基准数据集，通过分类指导的LLM提示和多阶段安全验证生成。每个场景以包含任务目标、车辆配置、环境条件和定量风险标签的结构化JSON模式编码，提供了跨不同领域UAV操作的统一表示。在此基础上，我们提出了UAVBench_MCQ，一个包含50,000个多项选择题的推理导向扩展，涵盖从空气动力学和导航到多智能体协调和集成推理的十种认知和伦理推理风格。该框架能够在真实的操作背景下对UAV特定认知进行可解释和机器可检查的评估。我们评估了32个最先进的LLMs，包括GPT-5、ChatGPT-4o、Gemini 2.5 Flash、DeepSeek V3、Qwen3 235B和ERNIE 4.5 300B，发现它们在感知和策略推理方面表现出色，但在道德感知和资源受限决策方面仍存在持续挑战。UAVBench为自主空中系统中智能AI的基准测试和推进下一代UAV推理智能的发展建立了可重现和物理基础。为支持开放科学和可重现性，我们在GitHub上发布了UAVBench数据集、UAVBench_MCQ基准、评估脚本和所有相关材料。


### 论文摘要

Autonomous aerial systems increasingly rely on large language models (LLMs) for mission planning, perception, and decision-making, yet the lack of standardized and physically grounded benchmarks limits systematic evaluation of their reasoning capabilities. To address this gap, we introduce UAVBench, an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation. Each scenario is encoded in a structured JSON schema that includes mission objectives, vehicle configuration, environmental conditions, and quantitative risk labels, providing a unified representation of UAV operations across diverse domains. Building on this foundation, we present UAVBench_MCQ, a reasoning-oriented extension containing 50,000 multiple-choice questions spanning ten cognitive and ethical reasoning styles, ranging from aerodynamics and navigation to multi-agent coordination and integrated reasoning. This framework enables interpretable and machine-checkable assessment of UAV-specific cognition under realistic operational contexts. We evaluate 32 state-of-the-art LLMs, including GPT-5, ChatGPT-4o, Gemini 2.5 Flash, DeepSeek V3, Qwen3 235B, and ERNIE 4.5 300B, and find strong performance in perception and policy reasoning but persistent challenges in ethics-aware and resource-constrained decision-making. UAVBench establishes a reproducible and physically grounded foundation for benchmarking agentic AI in autonomous aerial systems and advancing next-generation UAV reasoning intelligence. To support open science and reproducibility, we release the UAVBench dataset, the UAVBench_MCQ benchmark, evaluation scripts, and all related materials on GitHub at https://github.com/maferrag/UAVBench

---

## 84. Toward Gaze Target Detection of Young Autistic Children

**论文链接:** [http://arxiv.org/abs/2511.11244v1](http://arxiv.org/abs/2511.11244v1)

**作者:** Shijian Deng, Erin E. Kosloski, Siva Sai Nagender Vasireddy, Jia Li, Randi Sierra Sherwood, Feroz Mohamed Hatha, Siddhi Patel, Pamela R Rollins, Yapeng Tian

**发布时间:** 2025-11-14

**备注:** AAAI 2026 Artificial Intelligence for Social Impact Track

### GPT解析

### 总结

该研究提出了一种利用人工智能技术检测自闭症儿童注视目标的新方法，通过收集首个自闭症注视目标数据集并开发社交意识粗到精框架，有效解决了自闭症数据集中的类别不平衡问题，显著提升了注视目标检测性能。

### 背景

自闭症儿童往往缺乏足够的专业人员来改善生活质量，通过人工智能检测他们的注视目标具有重要意义。共同注意力是自闭症谱系障碍的核心挑战，而注视目标检测是构建能够测量共同注意力的自动化系统的基础。

### 目的

介绍一个新的真实世界AI应用，用于自闭症儿童的注视目标检测，从活动图像中预测儿童的注视点，为构建测量共同注意力的自动化系统奠定基础。

### 方法

收集了首个自闭症注视目标(AGT)数据集；提出了一种新颖的社交意识粗到精(SACF)注视检测框架，该框架利用场景的社会上下文克服自闭症数据集中的类别不平衡问题；采用双路径架构，包含专门用于社交注视和非社交注视的专家模型，并通过上下文感知门控模块进行指导。

### 主要发现

该框架在自闭症人群的注视目标检测任务上达到了最新的最先进性能，显著优于现有方法，特别是在面向人脸的关键少数类注视上表现尤为突出。

### 结论

该AI应用能够有效检测自闭症儿童的注视目标，有助于构建自动化系统来测量共同注意力，为改善自闭症儿童的生活质量提供了技术支持。

### 翻译

通过人工智能自动检测自闭症儿童的注视目标可能具有重大影响，特别是对于那些缺乏足够专业人员来改善生活质量的儿童。本文介绍了一种新的、真实世界的AI应用，用于自闭症儿童的注视目标检测，它可以从活动图像中预测儿童的注视点。这项任务是构建能够测量共同注意力的自动化系统的基础，而共同注意力是自闭症谱系障碍(ASD)的核心挑战。为了促进这一具有挑战性应用的研究，我们收集了首个自闭症注视目标(AGT)数据集。我们进一步提出了一种新颖的社交意识粗到精(SACF)注视检测框架，该框架明确利用场景的社会上下文来克服自闭症数据集中常见的类别不平衡问题——这是自闭症儿童倾向于减少对人脸注视的结果。它采用双路径架构，包含专门用于社交注视和非社交注视的专家模型，由上下文感知门控模块指导。我们全面实验的结果表明，我们的框架在该人群的注视目标检测任务上取得了最新的最先进性能，显著优于现有方法，特别是在面向人脸的关键少数类注视上。


### 论文摘要

The automatic detection of gaze targets in autistic children through artificial intelligence can be impactful, especially for those who lack access to a sufficient number of professionals to improve their quality of life. This paper introduces a new, real-world AI application for gaze target detection in autistic children, which predicts a child's point of gaze from an activity image. This task is foundational for building automated systems that can measure joint attention-a core challenge in Autism Spectrum Disorder (ASD). To facilitate the study of this challenging application, we collected the first-ever Autism Gaze Target (AGT) dataset. We further propose a novel Socially Aware Coarse-to-Fine (SACF) gaze detection framework that explicitly leverages the social context of a scene to overcome the class imbalance common in autism datasets-a consequence of autistic children's tendency to show reduced gaze to faces. It utilizes a two-pathway architecture with expert models specialized in social and non-social gaze, guided by a context-awareness gate module. The results of our comprehensive experiments demonstrate that our framework achieves new state-of-the-art performance for gaze target detection in this population, significantly outperforming existing methods, especially on the critical minority class of face-directed gaze.

---

## 85. Arcee: Differentiable Recurrent State Chain for Generative Vision Modeling with Mamba SSMs

**论文链接:** [http://arxiv.org/abs/2511.11243v1](http://arxiv.org/abs/2511.11243v1)

**作者:** Jitesh Chavan, Rohit Lal, Anand Kamat, Mengjia Xu

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为Arcee的跨块循环状态链方法，通过重用每个块的终端状态空间表示作为下一块的初始条件，显著提高了基于Mamba的视觉模型在长序列建模中的性能。

### 背景

状态空间模型（SSMs），特别是Mamba，越来越多地被用于长序列建模，通过输入依赖的因果选择性扫描操作提供线性时间聚合。然而，传统Mamba的选择性扫描操作在每个块重新初始化状态空间动力学，丢弃前一块的终端状态空间表示，导致跨块记忆丢失。

### 目的

解决传统Mamba模型在处理非序列信号（如图像）时丢失跨块记忆的问题，提高模型在长序列建模中的性能。

### 方法

提出Arcee方法，一种跨块循环状态链，重用每个块的终端状态空间表示作为下一块的初始条件。跨块交接被构建为可微分的边界映射，其雅可比矩阵使终端边界之间的端到端梯度流动成为可能。Arcee兼容所有先前的'vision-mamba'变体，无需额外参数，且成本恒定且可忽略。

### 主要发现

将终端SSR视为由输入上的因果传递引起的温和方向先验，而非非序列信号本身的估计器。在CelebA-HQ（256×256）上使用Flow Matching进行无条件生成，Arcee将FID从82.81降低到15.33（比基线低5.4倍）。

### 结论

Arcee是一种有效的方法，可以显著提高基于Mamba的视觉模型在长序列建模中的性能，同时保持兼容性和低计算成本。作者将发布高效的CUDA内核和训练代码，以支持严谨和可重复的研究。

### 翻译

状态空间模型（SSMs），特别是Mamba，越来越多地被用于长序列建模，通过输入依赖的、因果的选择性扫描操作提供线性时间聚合。沿着这一方向，近期的'Mamba-for-vision'变体主要探索多种扫描顺序，以对非序列信号（如图像）放宽严格的因果性。传统Mamba的选择性扫描操作在每个块重新初始化状态空间动力学，从零开始，丢弃前一块的终端状态空间表示（SSR）。Arcee是一种跨块循环状态链，重用每个块的终端状态空间表示作为下一块的初始条件。跨块交接被构建为可微分的边界映射，其雅可比矩阵使终端边界之间的端到端梯度流动成为可能。Arcee的关键实用性在于，它兼容所有先前的'vision-mamba'变体，无需参数，且成本恒定且可忽略。作为建模视角，我们将终端SSR视为由输入上的因果传递引起的温和方向先验，而非非序列信号本身的估计器。为了量化影响，在CelebA-HQ（256×256）上使用Flow Matching进行无条件生成时，Arcee将FID从82.81降低到15.33（比单扫描顺序Zigzag Mamba基线低5.4倍）。将发布高效的CUDA内核和训练代码，以支持严谨和可重复的研究。


### 论文摘要

State-space models (SSMs), Mamba in particular, are increasingly adopted for long-context sequence modeling, providing linear-time aggregation via an input-dependent, causal selective-scan operation. Along this line, recent "Mamba-for-vision" variants largely explore multiple scan orders to relax strict causality for non-sequential signals (e.g., images). Rather than preserving cross-block memory, the conventional formulation of the selective-scan operation in Mamba reinitializes each block's state-space dynamics from zero, discarding the terminal state-space representation (SSR) from the previous block. Arcee, a cross-block recurrent state chain, reuses each block's terminal state-space representation as the initial condition for the next block. Handoff across blocks is constructed as a differentiable boundary map whose Jacobian enables end-to-end gradient flow across terminal boundaries. Key to practicality, Arcee is compatible with all prior "vision-mamba" variants, parameter-free, and incurs constant, negligible cost. As a modeling perspective, we view terminal SSR as a mild directional prior induced by a causal pass over the input, rather than an estimator of the non-sequential signal itself. To quantify the impact, for unconditional generation on CelebA-HQ (256$\times$256) with Flow Matching, Arcee reduces FID$\downarrow$ from $82.81$ to $15.33$ ($5.4\times$ lower) on a single scan-order Zigzag Mamba baseline. Efficient CUDA kernels and training code will be released to support rigorous and reproducible research.

---

## 86. DoReMi: A Domain-Representation Mixture Framework for Generalizable 3D Understanding

**论文链接:** [http://arxiv.org/abs/2511.11232v1](http://arxiv.org/abs/2511.11232v1)

**作者:** Mingwei Xing, Xinliang Wang, Yifeng Shi

**发布时间:** 2025-11-14

### GPT解析

### 总结

DoReMi是一种专家混合框架，通过联合建模领域感知专家分支和统一表示分支，解决了3D深度学习在多域泛化中因数据集规模有限和多源点云高度异质性导致的负迁移问题。

### 背景

3D深度学习在多域泛化受限于现有数据集规模小和多源点云高度异质性。不同传感器收集的点云在密度和噪声分布上存在显著差异，导致多域融合中出现负迁移。现有方法只专注于领域感知特征或领域通用特征，忽视了两者间的协同作用。

### 目的

解决现有方法只关注领域感知特征或领域通用特征的问题，探索这两者之间的协同作用潜力，提高3D深度学习在多域场景下的泛化能力。

### 方法

提出DoReMi（Domain-Representation Mixture）框架，一种专家混合（MoE）架构，通过领域引导空间路由（DSR）动态激活领域感知专家分支，实现上下文感知的专家选择；采用熵控制动态分配（EDA）确保稳定高效的专家利用，适应性地建模多样化的域分布；结合通过鲁棒多属性自监督学习预训练的冻结统一表示分支，保留跨域几何和结构先验，同时保持全局一致性。

### 主要发现

在多个3D理解基准上评估显示，DoReMi在ScanNet Val上达到80.1% mIoU，在S3DIS上达到77.2% mIoU，与现有方法相比具有竞争性或更优的性能，展示了作为未来3D理解研究基础框架的巨大潜力。

### 结论

DoReMi有效解决了多域3D深度学习中的挑战，通过协同学习专业知识和可泛化知识，提高了模型在多样化域分布上的表现，代码即将发布。

### 翻译

摘要原文翻译：3D深度学习在多域的泛化仍受限于现有数据集的规模有限以及多源点云的高度异质性。从不同传感器（如LiDAR扫描和网格派生的点云）收集的点云在密度和噪声分布方面存在显著差异，导致多域融合过程中的负迁移。大多数现有方法只专注于领域感知特征或领域通用特征之一，忽视了它们之间的潜在协同作用。为此，我们提出了DoReMi（Domain-Representation Mixture），一种专家混合（MoE）框架，通过联合建模领域感知专家分支和统一的表示分支，实现专业知识和可泛化知识之间的协同学习。DoReMi通过领域引导空间路由（DSR）动态激活领域感知专家分支，实现上下文感知的专家选择，并采用熵控制动态分配（EDA）确保稳定高效的专家利用，从而自适应地建模多样化的域分布。结合通过鲁棒多属性自监督学习预训练的冻结统一表示分支，DoReMi保留了跨域几何和结构先验，同时保持全局一致性。我们在多个3D理解基准上评估了DoReMi。值得注意的是，DoReMi在ScanNet Val上达到80.1% mIoU，在S3DIS上达到77.2% mIoU，与现有方法相比具有竞争性或更优的性能，并显示出作为未来3D理解研究基础框架的巨大潜力。代码即将发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D深度学习在多个领域之间泛化能力有限的问题。具体表现为现有数据集规模小以及不同来源点云之间存在高度异质性。这个问题在现实中非常重要，因为3D场景理解是自动驾驶、机器人、增强现实等领域的基础，而现有模型通常只能处理特定类型的数据，无法适应多样化的实际应用场景。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的两个主要方向：领域感知特征学习和统一表示学习，发现它们通常被单独研究而忽略了互补性。基于此，作者设计了一个混合专家框架同时建模这两种特征。作者借鉴了自监督学习（如DINOv2和Sonata）、混合专家模型（MoE）范式、多属性自监督学习以及领域适应等现有工作的思想，并将其创新性地结合在一起。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过混合专家框架同时建模领域感知特征和统一表示特征，实现领域适应性和跨领域泛化能力的互补学习。整体流程分为三步：1)预训练阶段，使用多属性自监督学习学习通用特征；2)主训练阶段，包含两个分支 - 领域感知分支使用DSR进行专家选择和EDA进行动态分配，统一表示分支保持冻结；3)下游任务应用，可直接部署或进一步微调。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)集成的DoReMi框架，同时建模领域感知和通用特征；2)领域感知和统一表示的联合学习，通过DSR、EDA和预训练Re分支实现；3)稳定和可解释的专家选择机制。相比之前工作，DoReMi同时考虑领域感知和统一表示特征，基于领域和空间上下文进行专家选择，并能动态调整激活专家数量，而非固定数量。实验证明其在多个3D理解基准上取得了最先进的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DoReMi提出了一种创新的混合专家框架，通过联合建模领域感知特征和统一表示特征，有效解决了3D深度学习在多领域泛化中的挑战，显著提升了模型在不同场景和数据分布下的表现。'}


### 论文摘要

The generalization of 3D deep learning across multiple domains remains limited by the limited scale of existing datasets and the high heterogeneity of multi-source point clouds. Point clouds collected from different sensors (e.g., LiDAR scans and mesh-derived point clouds) exhibit substantial discrepancies in density and noise distribution, resulting in negative transfer during multi-domain fusion. Most existing approaches focus exclusively on either domain-aware or domain-general features, overlooking the potential synergy between them. To address this, we propose DoReMi (Domain-Representation Mixture), a Mixture-of-Experts (MoE) framework that jointly models Domain-aware Experts branch and a unified Representation branch to enable cooperative learning between specialized and generalizable knowledge. DoReMi dynamically activates domain-aware expert branch via Domain-Guided Spatial Routing (DSR) for context-aware expert selection and employs Entropy-Controlled Dynamic Allocation (EDA) for stable and efficient expert utilization, thereby adaptively modeling diverse domain distributions. Complemented by a frozen unified representation branch pretrained through robust multi-attribute self-supervised learning, DoReMi preserves cross-domain geometric and structural priors while maintaining global consistency. We evaluate DoReMi across multiple 3D understanding benchmarks. Notably, DoReMi achieves 80.1% mIoU on ScanNet Val and 77.2% mIoU on S3DIS, demonstrating competitive or superior performance compared to existing approaches, and showing strong potential as a foundation framework for future 3D understanding research. The code will be released soon.

---

## 87. MAFM^3: Modular Adaptation of Foundation Models for Multi-Modal Medical AI

**论文链接:** [http://arxiv.org/abs/2511.11212v1](http://arxiv.org/abs/2511.11212v1)

**作者:** Mohammad Areeb Qazi, Munachiso S Nwadike, Ibrahim Almakky, Mohammad Yaqub, Numan Saeed

**发布时间:** 2025-11-14

**备注:** 2 figures, 3 tables

### GPT解析

### 总结

MAFM^3框架使单一基础模型能够通过轻量级模块组件扩展到不同医学影像领域、任务和模态，解决了医学影像数据稀缺导致的预训练挑战。

### 背景

基础模型通常需要大量数据训练以捕捉领域普遍趋势，但医学影像领域的数据稀缺使得为每个领域、模态或任务进行预训练变得困难。

### 目的

提出MAFM^3框架，使单一基础模型能够通过模块化组件扩展到多样化领域、任务和模态，实现高效的多任务和多模态适应。

### 方法

使用模块化组件作为专业技能集，系统可根据输入类型或临床目标灵活激活适当能力；通过将胸部CT基础模型从分类扩展到预后和分割模块进行验证；整合PET扫描并与基线比较。

### 主要发现

模块化组件使基础模型在预后和分割任务上性能得到改善；整合PET扫描后，Dice分数比基线提高了5%；基础模型可发展为医学影像的多任务、多模态系统。

### 结论

基础模型配备模块化组件后不受初始训练范围的限制，能够扩展为医学影像的多任务、多模态系统。

### 翻译

基础模型在大量数据集上训练以捕捉领域的普遍趋势。然而，在医学影像中，数据稀缺使得为每个领域、模态或任务进行预训练具有挑战性。我们不是构建单独的模型，而是提出了MAFM^3（基础模型模块化适应用于多模态医学人工智能），这是一个框架，使单一基础模型能够通过轻量级模块组件扩展到不同领域、任务和模态。这些组件作为专业技能集，允许系统根据输入类型或临床目标在推理时灵活激活适当能力。与将每个新任务或模态孤立处理的常规适应方法不同，MAFM^3为高效的多任务和多模态适应提供了统一且可扩展的框架。通过实证，我们将最初为分类训练的胸部CT基础模型适应为预后和分割模块来验证我们的方法。结果显示两项任务性能均得到改善。此外，通过整合PET扫描，MAFM^3实现了比相应基线高5%的Dice分数。这些发现表明，配备模块化组件的基础模型并不受其初始训练范围的固有限制，可以发展为医学影像的多任务、多模态系统。本工作的代码实现可在https://github.com/Areeb2735/CTscan_prognosis_VLM找到。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决医学影像领域基础模型适应性和扩展性的挑战。医学影像数据稀缺，使得为每个领域、模态或任务预训练模型变得困难，而传统方法通常需要为每个任务构建单独模型，这在资源有限的医疗环境中不切实际。这个问题很重要，因为医学影像在现代医疗保健中扮演关键角色，但收集和整理医学影像数据资源密集，且不同模态(如CT、PET)捕捉人体不同方面的信息，导致数据分布差异大，难以用单一模型处理所有情况。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到医学影像数据稀缺和模态多样性带来的挑战，认为基础模型不应被视为静态系统，而应视为灵活的骨干，当配备轻量级模块组件时，可作为多任务、多模态医学AI系统的基础。他们借鉴了多个现有工作：医学基础模型如BiomedParse和CHIEF利用大规模数据学习可转移表示；持续学习方法如DynaMMo展示了增量扩展模型的能力；模块化适应策略如自扩展适配器专注于为模型配备可选择性激活的组件；参数高效微调方法如LoRA允许在不重新训练整个网络的情况下适应模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是MAFM³框架，使单个基础模型通过轻量级模块组件扩展到不同领域、任务和模态，这些模块作为专门技能集，允许系统根据输入类型或临床目标灵活激活适当能力。整体实现流程包括：1)使用预训练的CT-CLIP模型作为冻结骨干；2)通过三种机制实现模块化适应-模型内适应(WMA)使用LoRA微调、模型后适应(PMA)使用轻量级MLP层、分辨率适应处理不同扫描分辨率；3)渐进式扩展能力，从分类开始，逐步添加预后预测和分割任务，并整合多模态数据如PET扫描。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：统一的模块化框架使基础模型能高效扩展到新任务和模态；分辨率适应方法允许模型处理不同分辨率的输入；选择性模块激活实现高效多任务工作流程。相比之前工作，MAFM³提供了统一框架而非孤立处理每个任务；强调模块化和编排而非仅顺序扩展；通过仅微调小部分参数显著降低计算成本；通过解耦冻结骨干与轻量级模块避免灾难性遗忘，支持增量增长。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MAFM³提出了一种模块化适应框架，使单一医学基础模型能够通过轻量级组件扩展到多种任务和模态，同时保持原始性能并提高计算效率。'}


### 论文摘要

Foundational models are trained on extensive datasets to capture the general trends of a domain. However, in medical imaging, the scarcity of data makes pre-training for every domain, modality, or task challenging. Instead of building separate models, we propose MAFM^3 (Modular Adaptation of Foundation Models for Multi-Modal Medical AI), a framework that enables a single foundation model to expand into diverse domains, tasks, and modalities through lightweight modular components. These components serve as specialized skill sets that allow the system to flexibly activate the appropriate capability at the inference time, depending on the input type or clinical objective. Unlike conventional adaptation methods that treat each new task or modality in isolation, MAFM^3 provides a unified and expandable framework for efficient multitask and multimodality adaptation. Empirically, we validate our approach by adapting a chest CT foundation model initially trained for classification into prognosis and segmentation modules. Our results show improved performance on both tasks. Furthermore, by incorporating PET scans, MAFM^3 achieved an improvement in the Dice score 5% compared to the respective baselines. These findings establish that foundation models, when equipped with modular components, are not inherently constrained to their initial training scope but can evolve into multitask, multimodality systems for medical imaging. The code implementation of this work can be found at https://github.com/Areeb2735/CTscan_prognosis_VLM

---

## 88. One-to-N Backdoor Attack in 3D Point Cloud via Spherical Trigger

**论文链接:** [http://arxiv.org/abs/2511.11210v1](http://arxiv.org/abs/2511.11210v1)

**作者:** Dongmei Shan, Wei Lian, Chongxia Wang

**发布时间:** 2025-11-14

**备注:** 15 pages, 4 figures

### GPT解析

### 总结

本研究提出了一种创新的一对多后门攻击框架，基于可配置球形触发器，使单个触发器能够影响多个目标类别，为3D视觉系统建立了重要的安全基准。

### 背景

后门攻击对深度学习系统构成严重威胁，特别是在自动驾驶和机器人等安全敏感的3D领域。然而，现有的针对3D点云的后门攻击仅限于刚性的一对一模式。

### 目的

提出首个面向3D视觉的一对多后门框架，基于一种新颖的可配置球形触发器，以突破现有攻击模式的限制。

### 方法

利用球体的空间特性作为参数空间，使单个触发器设计能够编码多个目标类别；为3D中的一对多后门攻击建立理论基础，证明被污染的模型可以将不同触发器配置映射到不同目标标签。

### 主要发现

在多个数据集和模型架构上系统性地验证了该方法，实现了高达100%的攻击成功率，同时保持了对干净数据的高准确性。

### 结论

这项工作为3D视觉中的多目标威胁建立了重要基准，并为保护未来的3D驱动智能系统提供了基础理解。

### 翻译

后门攻击对深度学习系统构成了严重威胁，特别是在自动驾驶和机器人等安全敏感的3D领域。然而，现有的针对3D点云的后门攻击仅限于刚性的一对一模式。为此，我们提出了首个面向3D视觉的一对多后门框架，基于一种新颖的可配置球形触发器。我们的关键见解是利用球体的空间特性作为参数空间，使单个触发器设计能够编码多个目标类别。我们为3D中的一对多后门攻击建立了理论基础，证明被污染的模型可以将不同的触发器配置映射到不同的目标标签。实验结果在多个数据集和模型架构上系统性地验证了这一结论，实现了高达100%的攻击成功率，同时保持了对干净数据的准确性。这项工作为3D视觉中的多目标威胁建立了重要基准，并为保护未来的3D驱动智能系统提供了基础理解。


### 论文摘要

Backdoor attacks represent a critical threat to deep learning systems, particularly in safety-sensitive 3D domains such as autonomous driving and robotics. However, existing backdoor attacks for 3D point clouds have been limited to a rigid one-to-one paradigm. To address this, we present the first one-to-N backdoor framework for 3D vision, based on a novel, configurable spherical trigger. Our key insight is to leverage the spatial properties of spheres as a parameter space, allowing a single trigger design to encode multiple target classes. We establish a theoretical foundation for one-to-N backdoor attacks in 3D, demonstrating that poisoned models can map distinct trigger configurations to different target labels. Experimental results systematically validate this conclusion across multiple datasets and model architectures, achieving high attack success rates (up to 100\%) while maintaining accuracy on clean data. This work establishes a crucial benchmark for multi-target threats in 3D vision and provides the foundational understanding needed to secure future 3D-driven intelligent systems.

---

## 89. SynthSoM-Twin: A Multi-Modal Sensing-Communication Digital-Twin Dataset for Sim2Real Transfer via Synesthesia of Machines

**论文链接:** [http://arxiv.org/abs/2511.11503v1](http://arxiv.org/abs/2511.11503v1)

**作者:** Junlong Chen, Ziwei Huang, Xuesong Cai, Xiang Cheng, Liuqing Yang

**发布时间:** 2025-11-14

### GPT解析

### 总结

这篇论文提出了一个名为SynthSoM-Twin的新型多模态感知-通信数字孪生数据集，用于通过机器通感(SoM)进行Sim2Real迁移。

### 背景

现有真实世界多模态感知-通信数据集在数量和模态方面存在局限性，需要扩展数据集规模和补充缺失模态。

### 目的

构建一个在时空上与真实世界一致的多模态感知-通信数字孪生数据集，以支持Sim2Real迁移研究。

### 方法

提出了一种新框架扩展现有数据集，利用多模态感知辅助的目标检测和跟踪算法确保时空一致性，将构建的场景导入AirSim、WaveFarer和Sionna RT三个高保真模拟器，生成了包含RGB图像、深度图、激光雷达点云、毫米波雷达点云以及信道衰落数据的数据集。

### 主要发现

在SynthSoM-Twin数据集上训练的模型实现了良好的实际性能，真实世界数据的注入进一步促进了Sim2Real迁移能力，注入少于15%的真实世界数据可达到与全部真实世界数据训练相似甚至更好的效果。

### 结论

SynthSoM-Twin数据集有效支持了Sim2Real迁移，特别是在使用少量真实世界数据注入的情况下，能够平衡真实世界数据使用与模型实际性能。

### 翻译

本文构建了一个名为SynthSoM-Tin的新型多模态感知-通信数字孪生数据集，该数据集在时空上与真实世界一致，通过机器通感(SoM)实现Sim2Real迁移。为构建SynthSoM-Twin数据集，我们提出了一种新框架，可以扩展现有真实世界多模态感知-通信数据集的数量和缺失模态。具体而言，我们利用多模态感知辅助的目标检测和跟踪算法，确保静态和动态对象在真实世界与模拟环境之间的时空一致性。构建的场景被导入三个高保真模拟器，即AirSim、WaveFarer和Sionna RT。SynthSoM-Twin数据集包含与真实世界时空一致的数据，包括66,868张合成RGB图像、深度图、光检测和测距(LiDAR)点云、毫米波(mmWave)雷达点云以及大规模和小规模信道衰落数据。为验证SynthSoM-Twin数据集的实用性，我们通过跨模态生成模型(CMGMs)实施两个跨模态下游任务，即跨模态信道生成模型和多模态感知辅助的波束生成模型，进行了Sim2Real迁移研究。基于下游任务，我们探索了可以实现真实世界数据使用与模型实际性能之间良好平衡的真实世界数据注入阈值。实验结果表明，在SynthSoM-Twin数据集上训练的模型实现了良好的实际性能，真实世界数据的注入进一步促进了Sim2Real迁移能力。基于SynthSoM-Twin数据集，注入少于15%的真实世界数据可以实现与仅使用全部真实世界数据训练相似甚至更好的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何构建一个时空一致的多模态感知-通信数字孪生数据集，以减少现实世界数据收集成本同时保持模型实际性能的问题。这个问题很重要，因为真实世界多模态数据收集成本高、覆盖范围有限，而纯合成数据保真度不足，现有数字孪生数据集又忽略了动态对象的影响，无法实现时空一致性，限制了第六代通信中多模态感知-通信技术的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有数据集的局限性，指出纯合成数据保真度低、真实数据收集成本高、现有数字孪生数据集忽略动态对象等问题。他们借鉴了DeepSense 6G数据集的真实世界数据，并使用三个高保真模拟器(AirSim、WaveFarer和Sionna RT)来生成不同模态的数据。方法设计上采用了YOLOv12x进行目标检测、BoT-SORT进行跟踪，以及扭矩聚类算法处理点云数据，通过多模态感知辅助算法确保时空一致性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个与真实世界时空一致的多模态感知-通信数字孪生数据集，结合合成数据和少量真实世界数据，减少数据收集成本同时保持模型性能。整体流程包括：1)静态场景构建，使用真实世界数据重建建筑物和道路；2)动态场景构建，通过多模态感知算法跟踪车辆和行人；3)在AirSim中生成非RF感知数据；4)在WaveFarer中生成RF感知数据；5)在Sionna RT中生成通信数据；6)通过Python脚本协调各模拟器确保时间同步；7)验证合成数据与真实数据的时空一致性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出首个同时考虑静态和动态对象时空一致性的多模态感知-通信数字孪生数据集；2)设计新框架扩展现有数据集的数量和模态；3)包含66,868个多模态数据快照；4)通过两个跨模态下游任务验证数据集效用；5)发现仅需15%真实世界数据即可达到与全部真实数据相当的性能。相比之前工作，不同之处在于同时处理静态和动态对象、结合多种模拟器生成多模态数据、通过Sim2Real迁移验证数据集效用，以及大幅减少数据收集成本(节省85%以上)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了SynthSoM-Twin数据集，通过数字孪生技术结合合成数据和少量真实世界数据，在减少85%以上数据收集成本的同时，保持了模型在实际应用中的高性能。'}


### 论文摘要

This paper constructs a novel multi-modal sensing-communication digital-twin dataset, named SynthSoM-Twin, which is spatio-temporally consistent with the real world, for Sim2Real transfer via Synesthesia of Machines (SoM). To construct the SynthSoM-Twin dataset, we propose a new framework that can extend the quantity and missing modality of existing real-world multi-modal sensing-communication dataset. Specifically, we exploit multi-modal sensing-assisted object detection and tracking algorithms to ensure spatio-temporal consistency of static objects and dynamic objects across real world and simulation environments. The constructed scenario is imported into three high-fidelity simulators, i.e., AirSim, WaveFarer, and Sionna RT. The SynthSoM-Twin dataset contains spatio-temporally consistent data with the real world, including 66,868 snapshots of synthetic RGB images, depth maps, light detection and ranging (LiDAR) point clouds, millimeter wave (mmWave) radar point clouds, and large-scale and small-scale channel fading data. To validate the utility of SynthSoM-Twin dataset, we conduct Sim2Real transfer investigation by implementing two cross-modal downstream tasks via cross-modal generative models (CMGMs), i.e., cross-modal channel generation model and multi-modal sensing-assisted beam generation model. Based on the downstream tasks, we explore the threshold of real-world data injection that can achieve a decent trade-off between real-world data usage and models' practical performance. Experimental results show that the model training on the SynthSoM-Twin dataset achieves a decent practical performance, and the injection of real-world data further facilitates Sim2Real transferability. Based on the SynthSoM-Twin dataset, injecting less than 15% of real-world data can achieve similar and even better performance compared to that trained with all the real-world data only.

---

## 90. DGFusion: Dual-guided Fusion for Robust Multi-Modal 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.10035v1](http://arxiv.org/abs/2511.10035v1)

**作者:** Feiyang Jia, Caiyan Jia, Ailin Liu, Shaoqing Xu, Qiming Xia, Lin Liu, Lei Yang, Yan Gong, Ziying Song

**发布时间:** 2025-11-13

**DOI:** 10.1109/TCSVT.2025.3628019

### GPT解析

### 总结

本文提出了一种基于双引导范式的DGFusion方法，用于解决自动驾驶中的3D物体检测挑战，特别是针对远处、小型或被遮挡的困难实例。该方法通过困难感知实例匹配器和双引导模块，有效实现了多模态特征融合，在nuScenes数据集上取得了优于基线方法的性能。

### 背景

3D物体检测是自动驾驶感知系统中的关键任务，用于识别和跟踪车辆、行人等关键物体。然而，检测远处、小型或被遮挡的物体（困难实例）仍然是一个挑战，直接影响自动驾驶系统的安全性。

### 目的

解决现有多模态3D物体检测方法中单一引导范式的局限性，提高对困难实例的检测能力，从而增强自动驾驶系统的安全性。

### 方法

提出DGFusion，基于双引导范式，结合点引导图像范式和图像引导点范式的优势。核心是困难感知实例匹配器（DIPM），基于困难度执行实例级特征匹配，生成简单和困难实例对，双引导模块利用这些实例对的优势实现有效多模态特征融合。

### 主要发现

实验结果表明，DGFusion在nuScenes数据集上优于基线方法，分别提高了+1.0% mAP、+0.8% NDS和+1.3%平均召回率。在自车距离、大小、可见性和小规模训练场景下，困难实例检测具有一致的鲁棒性提升。

### 结论

DGFusion通过双引导范式有效解决了多模态3D物体检测中的困难实例检测问题，提高了自动驾驶系统的安全性和鲁棒性。

### 翻译

作为自动驾驶感知系统中的关键任务，3D物体检测用于识别和跟踪车辆和行人等关键物体。然而，检测远处、小型或被遮挡的物体（困难实例）仍然是一个挑战，这直接影响了自动驾驶系统的安全性。我们观察到现有的多模态3D物体检测方法通常遵循单一引导范式，未能考虑不同模态中困难实例的信息密度差异。在这项工作中，我们提出了基于双引导范式的DGFusion，它充分继承了点引导图像范式的优势，并集成了图像引导点范式，以解决单一范式的局限性。DGFusion的核心是困难感知实例匹配器（DIPM），它基于困难度执行实例级特征匹配，生成简单和困难实例对，而双引导模块则利用这两种实例对的优势实现有效的多模态特征融合。实验结果表明，我们的DGFusion优于基线方法，在nuScenes上分别提高了+1.0% mAP、+0.8% NDS和+1.3%平均召回率。大量实验表明，在自车距离、大小、可见性和小规模训练场景下，困难实例检测具有一致的鲁棒性提升。


### 论文摘要

As a critical task in autonomous driving perception systems, 3D object detection is used to identify and track key objects, such as vehicles and pedestrians. However, detecting distant, small, or occluded objects (hard instances) remains a challenge, which directly compromises the safety of autonomous driving systems. We observe that existing multi-modal 3D object detection methods often follow a single-guided paradigm, failing to account for the differences in information density of hard instances between modalities. In this work, we propose DGFusion, based on the Dual-guided paradigm, which fully inherits the advantages of the Point-guide-Image paradigm and integrates the Image-guide-Point paradigm to address the limitations of the single paradigms. The core of DGFusion, the Difficulty-aware Instance Pair Matcher (DIPM), performs instance-level feature matching based on difficulty to generate easy and hard instance pairs, while the Dual-guided Modules exploit the advantages of both pair types to enable effective multi-modal feature fusion. Experimental results demonstrate that our DGFusion outperforms the baseline methods, with respective improvements of +1.0\% mAP, +0.8\% NDS, and +1.3\% average recall on nuScenes. Extensive experiments demonstrate consistent robustness gains for hard instance detection across ego-distance, size, visibility, and small-scale training scenarios.

---

## 91. MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection Systems

**论文链接:** [http://arxiv.org/abs/2511.09999v1](http://arxiv.org/abs/2511.09999v1)

**作者:** Saket S. Chaturvedi, Gaurav Bagwe, Lan Zhang, Pan He, Xiaoyong Yuan

**发布时间:** 2025-11-13

**备注:** Accepted at AAAI 2026 Conference

### GPT解析

### 总结

本文提出了一种名为材料导向后门攻击(MOBA)的新框架，通过明确建模现实触发器材料属性来弥合数字-物理差距，解决了物理后门设计中材料鲁棒性和数字-物理行为对齐的挑战，实现了93.50%的攻击成功率。

### 背景

LiDAR 3D目标检测广泛应用于安全关键系统，但这些系统容易受到后门攻击。现有后门攻击的主要局限性是缺乏物理可实现性，数字触发器因忽略材料依赖的LiDAR反射特性而在现实环境中失败，物理构建的触发器又常因未经优化而效果低或易被检测。

### 目的

开发MOBA框架解决物理后门设计中的两个关键挑战：1) 触发器材料在不同环境条件下的鲁棒性；2) 物理触发器行为与其数字模拟之间的对齐。

### 方法

1) 提出系统方法选择鲁棒触发器材料，确定二氧化钛(TiO_2)具有高漫反射率和环境适应性；2) 开发新模拟管道确保数字触发器准确模拟物理行为，包括Oren-Nayar BRDF模型的角度无关近似和感知距离的缩放机制。

### 主要发现

在最先进的基于LiDAR和相机-LiDAR融合模型上实验表明，MOBA实现了93.50%的攻击成功率，比之前的方法提高41%以上。

### 结论

研究揭示了一类新的物理可实现威胁，强调了迫切需要开发考虑现实环境中材料特性的防御措施。

### 翻译

基于LiDAR的3D目标检测广泛应用于安全关键系统。然而，这些系统仍然容易受到后门攻击，这些攻击在训练过程中嵌入隐藏的恶意行为。现有后门攻击的一个关键局限性是缺乏物理可实现性，主要由于数字到物理领域的差距。数字触发器在现实环境中往往失败，因为它们忽略了依赖于材料的LiDAR反射特性。另一方面，物理构建的触发器通常未经优化，导致效果低或容易被检测。本文引入了材料导向后门攻击(MOBA)，一个通过明确建模现实触发器材料属性来弥合数字-物理差距的新框架。MOBA解决了物理后门设计中的两个关键挑战：1) 触发器材料在不同环境条件下的鲁棒性；2) 物理触发器行为与其数字模拟之间的对齐。首先，我们提出了一种系统方法来选择鲁棒的触发器材料，确定二氧化钛(TiO_2)具有高漫反射率和环境适应性。其次，为确保数字触发器准确模拟基于材料的物理触发器行为，我们开发了一个新的模拟管道，特点包括：(1) 使用Oren-Nayar BRDF模型的角度无关近似来生成真实的LiDAR强度；(2) 使用感知距离的缩放机制来保持不同深度上的空间一致性。我们在最先进的基于LiDAR和相机-LiDAR融合模型上进行了大量实验，表明MOBA实现了93.50%的攻击成功率，比之前的方法提高41%以上。我们的工作揭示了一类新的物理可实现威胁，并强调了迫切需要开发考虑现实环境中材料特性的防御措施。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR-based 3D object detection系统面临的后门攻击问题，特别是现有攻击方法缺乏物理可实现性的关键缺陷。这个问题在现实中非常重要，因为这些系统广泛应用于自动驾驶等安全关键领域，后门攻击可能导致严重的安全事故；在研究中，这个问题填补了数字攻击与物理实现之间的空白，为评估真实世界威胁提供了更准确的模型。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从材料科学和光学物理角度出发，认识到材料特性对LiDAR反射的关键影响，这是现有研究忽视的方面。他们借鉴了现有的光学物理模型（如Fresnel方程和Oren-Nayar BRDF模型），但进行了改进以适应LiDAR后门攻击的特殊需求。作者还借鉴了后门攻击理论和点云处理技术，但创新性地将它们与材料科学相结合，创造了一种物理可行的后门攻击方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过考虑材料特性来设计物理可行的后门触发器，弥合数字后门攻击与物理实现之间的差距。整体实现流程包括：1)材料建模阶段，系统评估多种材料的光学特性并选择二氧化钛作为最优触发材料；2)LiDAR强度模拟阶段，开发角度无关的BRDF模型近似和距离感知的缩放机制；3)触发器构造与注入，创建物理和数字触发器并放置在车辆后窗区域；4)模型训练，将含有数字触发器的点云注入训练数据；5)评估测试，在真实环境中测试触发器的效果和鲁棒性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)材料导向的后门攻击框架；2)系统化的材料选择策略；3)角度无关的BRDF模型近似；4)距离感知的触发器缩放机制；5)物理可行的触发器设计。相比之前的工作，MOBA首次系统性地考虑了材料特性对LiDAR反射的影响，解决了物理可实现性问题，在各种环境条件下保持高攻击成功率，并具有更好的隐蔽性和通用性，适用于LiDAR-only和Camera-LiDAR融合模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MOBA通过结合材料科学和光学物理原理，首次实现了高度物理可行的LiDAR后门攻击，成功弥合了数字攻击与物理实现之间的差距，显著提高了攻击在真实世界环境中的有效性和隐蔽性。'}


### 论文摘要

LiDAR-based 3D object detection is widely used in safety-critical systems. However, these systems remain vulnerable to backdoor attacks that embed hidden malicious behaviors during training. A key limitation of existing backdoor attacks is their lack of physical realizability, primarily due to the digital-to-physical domain gap. Digital triggers often fail in real-world settings because they overlook material-dependent LiDAR reflection properties. On the other hand, physically constructed triggers are often unoptimized, leading to low effectiveness or easy detectability.This paper introduces Material-Oriented Backdoor Attack (MOBA), a novel framework that bridges the digital-physical gap by explicitly modeling the material properties of real-world triggers. MOBA tackles two key challenges in physical backdoor design: 1) robustness of the trigger material under diverse environmental conditions, 2) alignment between the physical trigger's behavior and its digital simulation. First, we propose a systematic approach to selecting robust trigger materials, identifying titanium dioxide (TiO_2) for its high diffuse reflectivity and environmental resilience. Second, to ensure the digital trigger accurately mimics the physical behavior of the material-based trigger, we develop a novel simulation pipeline that features: (1) an angle-independent approximation of the Oren-Nayar BRDF model to generate realistic LiDAR intensities, and (2) a distance-aware scaling mechanism to maintain spatial consistency across varying depths. We conduct extensive experiments on state-of-the-art LiDAR-based and Camera-LiDAR fusion models, showing that MOBA achieves a 93.50% attack success rate, outperforming prior methods by over 41%. Our work reveals a new class of physically realizable threats and underscores the urgent need for defenses that account for material-level properties in real-world environments.

---

## 92. STORM: Segment, Track, and Object Re-Localization from a Single 3D Model

**论文链接:** [http://arxiv.org/abs/2511.09771v1](http://arxiv.org/abs/2511.09771v1)

**作者:** Yu Deng, Teng Cao, Hikaru Shindo, Jiahong Xue, Quentin Delfosse, Kristian Kersting

**发布时间:** 2025-11-12

### GPT解析

### 总结

STORM是一种开源的鲁棒实时6D姿态估计系统，无需手动标注，通过三阶段流水线结合视觉语言理解和自监督特征匹配实现目标定位、跟踪和重定位，并具有自动重新注册机制以应对遮挡和快速运动。

### 背景

准确的6D姿态估计和跟踪是物理AI系统的基本能力，但现有方法依赖于第一帧目标的手动标注分割掩码，既耗时又容易在遇到遮挡或快速移动时性能下降。

### 目的

解决现有方法的局限性，提出一种无需手动标注的6D姿态估计系统。

### 方法

提出STORM系统，采用三阶段流水线结合视觉语言理解和自监督特征匹配：上下文对象描述引导定位，自交叉注意力机制识别候选区域，分割模型生成精确掩码；并开发自动重新注册机制通过特征相似性监控检测跟踪失败并从遮挡或快速运动中恢复。

### 主要发现

STORM在具有多目标遮挡、高速运动和变化光照的挑战性工业数据集上实现了最先进的准确性，同时以实时速度运行，无需额外训练。

### 结论

这种无需标注的方法显著降低了部署开销，为柔性制造和智能质量控制等现代应用提供了实用的解决方案。

### 翻译

准确的6D姿态估计和跟踪是物理AI系统（如机器人）的基本能力。然而，现有方法通常依赖于第一帧目标的手动标注分割掩码，这既耗时又容易在遇到遮挡或快速移动时性能下降。为解决这些局限性，我们提出了STORM（从单个3D模型进行分割、跟踪和对象重定位），这是一个开源的鲁棒实时6D姿态估计系统，无需手动标注。STORM采用新颖的三阶段流水线，结合视觉语言理解和自监督特征匹配：上下文对象描述引导定位，自交叉注意力机制识别候选区域，分割模型生成精确掩码以进行准确的姿态估计。另一项关键创新是我们的自动重新注册机制，它通过特征相似性监控检测跟踪失败，并从严重遮挡或快速运动中恢复。STORM在具有多目标遮挡、高速运动和变化光照的挑战性工业数据集上实现了最先进的准确性，同时以实时速度运行，无需额外训练。这种无需标注的方法显著降低了部署开销，为现代应用（如柔性制造和智能质量控制）提供了实用的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决物理AI系统（如机器人）中6D姿态估计和跟踪的挑战。现有方法通常需要人工标注第一帧的目标对象分割掩码，这种方法劳动密集，且在遇到遮挡或快速移动时性能下降。这个问题很重要，因为6D姿态估计是机器人操作和增强现实的基础能力，手动标注增加了部署成本，而真实场景中对象经常面临遮挡和快速移动的情况，限制了系统处理多样化对象的能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性来设计STORM：现有方法在实例级别操作无法推广到新对象，依赖先验信息如对象掩码，在杂乱场景和遮挡情况下性能差，动态场景中跟踪容易失败。作者借鉴了现有工作，如基于参考的方法（CNOS和PerSAM）通过渲染3D模型生成掩码，使用预训练视觉模型（如DINOv2）提取特征，结合基础分割模型（如FastSAM和SAM2），但设计了新的层次空间融合注意力机制（HSFA）和跟踪损失分类器来解决现有方法的不足。', '这个方法的核心思想是什么？整体实现流程是怎样的？': 'STORM的核心思想是使用单个3D模型实现完全自动化的分割、跟踪和对象重新定位，无需手动标注。该方法结合视觉语言理解和自监督特征匹配，包含自动重新注册机制处理遮挡或快速移动。整体流程分为两个模块：1）分割对象模块（SOM）：将3D模型渲染成多视图参考图，用LLM生成语义描述，提取特征并通过HSFA融合，生成精确掩码；2）跟踪对象模块（TOM）：监控跟踪质量，检测失败时自动重新初始化，使用SOM生成新掩码并重新估计姿态，通过轻量级网络验证跟踪有效性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）统一的6D姿态估计框架，仅用单个3D模型运行；2）层次空间融合注意力（HSFA）机制，融合空间和语义特征产生高质量掩码；3）跟踪损失分类器，实时监控跟踪质量并触发快速恢复；4）自动重新注册机制，从遮挡或快速运动中恢复；5）完全自监督方法，无需手动标注。相比之前工作，STORM不需要初始对象掩码，具有内置跟踪失败检测，在复杂场景实现最先进性能且实时运行，提供了实用的解决方案如灵活制造和智能质量控制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'STORM提出了一种创新的自上而下方法，仅使用单个3D模型即可实现无需手动标注的鲁棒实时6D姿态估计和对象跟踪，通过结合视觉语言理解、自监督特征匹配和自动重新注册机制，有效解决了现有方法在遮挡和快速移动场景下的局限性。'}


### 论文摘要

Accurate 6D pose estimation and tracking are fundamental capabilities for physical AI systems such as robots. However, existing approaches typically rely on a manually annotated segmentation mask of the target in the first frame, which is labor-intensive and leads to reduced performance when faced with occlusions or rapid movement. To address these limi- tations, we propose STORM (Segment, Track, and Object Re-localization from a single 3D Model), an open-source robust real-time 6D pose estimation system that requires no manual annotation. STORM employs a novel three-stage pipeline combining vision-language understanding with self-supervised feature matching: contextual object descriptions guide localization, self-cross-attention mechanisms identify candidate regions, and a segmentation model produces precise masks for accurate pose estimation. Another key innovation is our automatic re-registration mechanism that detects tracking failures through feature similarity monitoring and recovers from severe occlusions or rapid motion. STORM achieves state-of-the-art accuracy on challenging industrial datasets featuring multi-object occlusions, high-speed motion, and varying illumination, while operating at real-time speeds without additional training. This annotation-free approach significantly reduces deployment overhead, providing a practical solution for modern applications, such as flexible manufacturing and intelligent quality control.

---

## 93. FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.09347v2](http://arxiv.org/abs/2511.09347v2)

**作者:** Jiangyong Yu, Changyong Shu, Sifan Zhou, Zichen Yu, Xing Hu, Yan Chen, Dawei Yang

**发布时间:** 2025-11-12

**备注:** I made an operational error. I intended to update the paper with Identifier arXiv:2502.15488, not submit a new paper with a different identifier. Therefore, I would like to withdraw the current submission and resubmit an updated version for Identifier arXiv:2502.15488

### GPT解析

### 总结

本文提出了一种名为FQ-PETR的完全量化框架，用于解决PETR模型在量化过程中面临的严重精度下降问题，实现了高精度和低延迟的平衡。

### 背景

基于摄像头的多视角3D检测对自动驾驶至关重要。PETR及其变体在基准测试中表现出色，但由于高计算成本和内存占用面临部署挑战。量化是一种有效的神经网络压缩技术，但直接应用于PETR会导致严重精度下降。

### 目的

开发一种完全量化的PETR框架，解决现有量化方法直接应用于PETR时导致的严重精度下降问题，同时保持模型性能并降低计算资源需求。

### 方法

提出三个关键创新：1) 量化友好的激光雷达射线位置嵌入(QFPE)，用激光雷达先验引导的单点采样替代多点采样；2) 双查找表(DULUT)，使用两个级联的线性LUT近似复杂非线性函数；3) 数值稳定后量化(QANS)，在softmax数值稳定后执行量化以减轻注意力失真。

### 主要发现

直接应用现有量化方法到PETR会导致严重精度下降，主要源于两个关键挑战：多模态特征(图像特征和相机射线位置嵌入)之间显著的幅度差异，以及非线性量化的低效率和近似误差。

### 结论

FQ-PETR在W8A8量化配置下实现了接近浮点精度的性能(仅1%的下降)，同时将延迟减少高达75%，显著优于现有的PTQ和QAT基线方法。

### 翻译

基于摄像头的多视角3D检测对自动驾驶至关重要。PETR及其变体在基准测试中表现出色，但由于高计算成本和内存占用面临部署挑战。量化是一种通过减少权重和激活比特宽度来压缩深度神经网络的有效技术。然而，将现有量化方法直接应用于PETR会导致严重的精度下降。这个问题主要源于两个关键挑战：(1)多模态特征(特别是图像特征和相机射线位置嵌入)之间的显著幅度差异，(2)非线性算子的量化效率低下和近似误差，这些算子通常依赖于硬件不友好的计算。在本文中，我们提出了FQ-PETR，一个用于PETR的完全量化框架，具有三个关键创新：(1)量化友好的激光射线位置嵌入(QFPE)：用激光先验引导的单点采样替代多点采样，基于锚点的嵌入消除了有问题的非线性(例如反sigmoid)操作，并使PE规模与图像特征保持一致，从而保留精度。(2)双查找表(DULUT)：该算法使用两个级联的线性LUT近似复杂的非线性函数，以最小条目实现高保真度且无需专用硬件。(3)数值稳定后量化(QANS)：在softmax数值稳定后执行量化，减轻大输入导致的注意力失真。在PETR(如PETR、StreamPETR、PETRv2、MV2d)上，FQ-PETR在W8A8下实现了接近浮点精度的性能(1%的下降)，同时将延迟减少高达75%，显著优于现有的PTQ和QAT基线。


### 论文摘要

Camera-based multi-view 3D detection is crucial for autonomous driving. PETR and its variants (PETRs) excel in benchmarks but face deployment challenges due to high computational cost and memory footprint. Quantization is an effective technique for compressing deep neural networks by reducing the bit width of weights and activations. However, directly applying existing quantization methods to PETRs leads to severe accuracy degradation. This issue primarily arises from two key challenges: (1) significant magnitude disparity between multi-modal features-specifically, image features and camera-ray positional embeddings (PE), and (2) the inefficiency and approximation error of quantizing non-linear operators, which commonly rely on hardware-unfriendly computations. In this paper, we propose FQ-PETR, a fully quantized framework for PETRs, featuring three key innovations: (1) Quantization-Friendly LiDAR-ray Position Embedding (QFPE): Replacing multi-point sampling with LiDAR-prior-guided single-point sampling and anchor-based embedding eliminates problematic non-linearities (e.g., inverse-sigmoid) and aligns PE scale with image features, preserving accuracy. (2) Dual-Lookup Table (DULUT): This algorithm approximates complex non-linear functions using two cascaded linear LUTs, achieving high fidelity with minimal entries and no specialized hardware. (3) Quantization After Numerical Stabilization (QANS): Performing quantization after softmax numerical stabilization mitigates attention distortion from large inputs. On PETRs (e.g. PETR, StreamPETR, PETRv2, MV2d), FQ-PETR under W8A8 achieves near-floating-point accuracy (1% degradation) while reducing latency by up to 75%, significantly outperforming existing PTQ and QAT baselines.

---

## 94. Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.08015v2](http://arxiv.org/abs/2511.08015v2)

**作者:** Jian Wang, Lijun He, Yixing Yong, Haixia Bi, Fan Li

**发布时间:** 2025-11-11

**备注:** Accepted by the AAAI 2026 (Main Track)

### GPT解析

### 总结

该研究提出了一种名为AdvRoad的方法，用于生成自然外观的道路风格对抗海报，能够在自动驾驶系统中实现隐蔽的3D物体检测对抗攻击。

### 背景

现代自动驾驶系统使用3D物体检测感知环境，基于RGB相机的视觉3D检测比LiDAR更经济，但当前深度神经网络模型易受对抗样本影响，存在安全隐患。

### 目的

调查自动驾驶场景中的真实对抗攻击，解决先前对抗海报不自然外观和固定内容的问题，实现隐蔽有效的攻击。

### 方法

提出AdvRoad生成多样化道路风格对抗海报，采用两阶段方法：道路风格对手生成和场景相关适应，在最大化攻击有效性的同时确保海报自然外观。

### 主要发现

AdvRoad在不同检测器、场景和欺骗位置上具有良好泛化能力，物理攻击证明了真实环境中的实际威胁。

### 结论

AdvRoad能够生成类似道路表面的自然对抗海报，在不引起人类注意的情况下秘密执行攻击，对自动驾驶系统构成实际安全威胁。

### 翻译

现代自动驾驶系统利用3D物体检测来感知3D环境中的前景物体，用于后续的预测和规划。基于RGB相机的视觉3D检测相比LiDAR范式提供了一种更具成本效益的解决方案。虽然取得了有前景的检测精度，但当前基于深度神经网络的模型仍然高度易受对抗样本的影响。潜在的安全问题促使我们研究自动驾驶场景中的真实对抗攻击。先前的工作证明了在路面上放置对抗海报以在检测器中引起幻觉的可行性。然而，海报的不自然外观使其容易被人类注意到，并且其固定内容很容易被针对和防御。为解决这些限制，我们提出了AdvRoad来生成多样化的道路风格对抗海报。这些对手具有类似于路面的自然外观，同时使检测器在攻击位置感知到不存在的物体。我们采用两阶段方法，称为道路风格对手生成和场景相关适应，以最大化对输入场景的攻击效果，同时确保海报的自然外观，使攻击能够秘密进行而不引起人类注意。大量实验表明，AdvRoad在不同检测器、场景和欺骗位置上具有良好的泛化能力。此外，物理攻击进一步证明了真实环境中的实际威胁。


### 论文摘要

Modern autonomous driving (AD) systems leverage 3D object detection to perceive foreground objects in 3D environments for subsequent prediction and planning. Visual 3D detection based on RGB cameras provides a cost-effective solution compared to the LiDAR paradigm. While achieving promising detection accuracy, current deep neural network-based models remain highly susceptible to adversarial examples. The underlying safety concerns motivate us to investigate realistic adversarial attacks in AD scenarios. Previous work has demonstrated the feasibility of placing adversarial posters on the road surface to induce hallucinations in the detector. However, the unnatural appearance of the posters makes them easily noticeable by humans, and their fixed content can be readily targeted and defended. To address these limitations, we propose the AdvRoad to generate diverse road-style adversarial posters. The adversaries have naturalistic appearances resembling the road surface while compromising the detector to perceive non-existent objects at the attack locations. We employ a two-stage approach, termed Road-Style Adversary Generation and Scenario-Associated Adaptation, to maximize the attack effectiveness on the input scene while ensuring the natural appearance of the poster, allowing the attack to be carried out stealthily without drawing human attention. Extensive experiments show that AdvRoad generalizes well to different detectors, scenes, and spoofing locations. Moreover, physical attacks further demonstrate the practical threats in real-world environments.

---

## 95. Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.07966v1](http://arxiv.org/abs/2511.07966v1)

**作者:** Shenao Zhao, Pengpeng Liang, Zhoufan Yang

**发布时间:** 2025-11-11

**备注:** Accepted to AAAI-26

### GPT解析

### 总结

本文提出了一种名为MMAssist的多模态辅助方法，用于提高基于激光雷达的3D目标检测中的无监督域适应性能。该方法利用图像和文本特征作为桥梁，对齐源域和目标域之间的3D特征，并通过特征融合提升检测效果。

### 背景

基于教师-学生架构和伪标签的无监督域适应在3D目标检测中已取得显著进展，但很少关注图像数据在3D UDA训练中的有用性。尽管同时收集点云和图像数据很常见，但图像数据的价值尚未被充分利用。

### 目的

提出一种方法，通过多模态辅助提高3D无监督域适应的性能，充分利用图像和文本信息来增强3D目标检测能力。

### 方法

MMAssist方法将3D标签投影到图像获取2D边界框，提取图像特征和文本特征作为桥梁对齐3D特征。在训练过程中，将预测的3D特征与对应的图像和文本特征融合，并使用学习权重进行最终预测。同时对齐目标域中学生和教师分支特征，并利用2D检测器和点云增强伪标签质量。

### 主要发现

在三个流行的3D目标检测数据集上的三个域适应任务中，MMAssist方法与最先进的方法相比取得了有希望的性能，证明了多模态辅助在3D UDA中的有效性。

### 结论

通过有效利用图像和文本特征作为桥梁对齐3D特征，MMAssist方法显著提升了3D无监督域适应的性能，为多模态数据在3D目标检测中的应用提供了新思路。

### 翻译

基于教师-学生架构和伪标签的无监督域适应(3D UDA)在基于激光雷达的3D目标检测中近年来取得了显著进展。尽管同时收集点云和图像数据相当普遍，但在训练模型时，很少关注图像数据在3D UDA中的有用性。在本文中，我们提出了一种名为MMAssist的方法，通过多模态辅助提高3D UDA的性能。设计了一种方法，利用图像和文本特征作为桥梁，对齐源域和目标域之间的3D特征。更具体地说，我们将真实标签或伪标签投影到图像上以获取一组2D边界框。对于每个2D框，我们从预训练的视觉主干网络中提取其图像特征。采用大型视觉语言模型(LVLM)提取框的文本描述，并使用预训练的文本编码器获取其文本特征。在源域模型和目标域学生模型的训练过程中，我们将预测框的3D特征与其相应的图像和文本特征对齐，并将3D特征和对齐后的特征以学习到的权重融合进行最终预测。目标域中学生分支和教师分支之间的特征也进行了对齐。为了增强伪标签，我们使用现成的2D目标检测器从图像生成2D边界框，借助点云估计其对应的3D框，并将这些3D框与教师模型生成的伪标签相结合。实验结果表明，与三个流行3D目标检测数据集上三个域适应任务中的最先进方法相比，我们的方法取得了有希望的性能。代码可在https://github.com/liangp/MMAssist获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的是基于激光雷达点云的3D目标检测在无监督域适应中的性能问题。在自动驾驶领域，不同环境（如不同地区、天气条件）会导致域差异，使模型在源域训练后难以直接应用到目标域。无监督域适应允许模型在没有标注数据的目标域上适应，降低了数据标注成本，而利用多模态信息（图像和文本）可以更好地处理域差异，提高模型在不同环境下的泛化能力，这对自动驾驶系统的实际部署至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到虽然自动驾驶中通常同时收集点云和图像数据，但大多数现有方法仅依赖LiDAR数据，没有充分利用图像信息。他们注意到图像特征的域差异比点云特征小，因为视觉模型在大规模数据上训练后具有强大的泛化能力；同时，大型视觉-语言模型(LVLM)能为不同域中的相似对象生成相似的文本描述。作者借鉴了教师-学生自训练方法DTS，并在此基础上扩展，设计了使用图像和文本特征作为桥梁来对齐源域和目标域3D特征的方法，并利用2D检测器生成的3D边界框来增强伪标签质量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用图像和文本特征作为桥梁来对齐不同域之间的3D特征，利用图像特征的较小域差异和文本描述的语义一致性来增强3D特征的域不变性，并结合从图像生成的3D边界框来增强伪标签质量。整体流程分为预训练阶段和自训练阶段：预训练阶段在源域数据上训练模型，对齐3D特征与对应的图像和文本特征并融合进行预测；自训练阶段使用教师模型生成目标域伪标签，结合从图像生成的3D边界框作为增强伪标签，训练学生模型并更新教师模型；测试阶段仅使用点云作为输入。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）多模态辅助：首次系统利用图像和文本信息辅助基于激光雷达的3D无监督域适应；2）特征对齐桥梁：使用图像和文本特征作为中间桥梁对齐不同域的3D特征；3）伪标签增强：结合2D检测器生成的3D边界框和教师模型生成的伪标签。相比之前工作，本文不仅利用点云数据，还充分利用了同时收集的图像和文本数据；不直接对齐不同域特征，而是使用图像和文本作为中间桥梁；在伪标签生成上特别关注远距离目标，并在多个数据集和检测器上取得了优于现有方法的性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种多模态辅助方法MMAssist，通过利用图像和文本特征作为桥梁来对齐不同域之间的3D特征，并结合从图像生成的3D边界框增强伪标签质量，显著提高了点云3D目标检测的无监督域适应性能。'}


### 论文摘要

Unsupervised domain adaptation for LiDAR-based 3D object detection (3D UDA) based on the teacher-student architecture with pseudo labels has achieved notable improvements in recent years. Although it is quite popular to collect point clouds and images simultaneously, little attention has been paid to the usefulness of image data in 3D UDA when training the models. In this paper, we propose an approach named MMAssist that improves the performance of 3D UDA with multi-modal assistance. A method is designed to align 3D features between the source domain and the target domain by using image and text features as bridges. More specifically, we project the ground truth labels or pseudo labels to the images to get a set of 2D bounding boxes. For each 2D box, we extract its image feature from a pre-trained vision backbone. A large vision-language model (LVLM) is adopted to extract the box's text description, and a pre-trained text encoder is used to obtain its text feature. During the training of the model in the source domain and the student model in the target domain, we align the 3D features of the predicted boxes with their corresponding image and text features, and the 3D features and the aligned features are fused with learned weights for the final prediction. The features between the student branch and the teacher branch in the target domain are aligned as well. To enhance the pseudo labels, we use an off-the-shelf 2D object detector to generate 2D bounding boxes from images and estimate their corresponding 3D boxes with the aid of point cloud, and these 3D boxes are combined with the pseudo labels generated by the teacher model. Experimental results show that our approach achieves promising performance compared with state-of-the-art methods in three domain adaptation tasks on three popular 3D object detection datasets. The code is available at https://github.com/liangp/MMAssist.

---

## 96. MonoCLUE : Object-Aware Clustering Enhances Monocular 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.07862v1](http://arxiv.org/abs/2511.07862v1)

**作者:** Sunghun Yang, Minhyeok Lee, Jungho Lee, Sangyoun Lee

**发布时间:** 2025-11-11

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

MonoCLUE通过结合局部聚类和全局场景记忆策略，解决了单目3D检测中的几何线索不足问题，能够有效处理遮挡场景，提高检测准确性。

### 背景

单目3D目标检测为自动驾驶提供经济有效的解决方案，但存在深度不适定和视野有限的问题，导致缺乏几何线索，在遮挡或截断场景中准确性降低。最近的方法虽然加入额外深度信息来解决几何歧义，但忽略了稳健识别所需的重要视觉线索。

### 目的

提出一种方法来增强单目3D检测能力，解决在遮挡和有限可见性情况下的稳健检测问题。

### 方法

提出MonoCLUE框架，利用局部聚类和视觉特征的全局场景记忆。对视觉特征进行K-means聚类捕获不同级别的物体外观部分，聚类特征在区域间传播捕获相似外观物体，构建通用场景记忆提供一致的表示，并将局部聚类和全局场景记忆整合到物体查询中引导注意力。

### 主要发现

局部聚类有助于检测部分可见物体；通用场景记忆提供一致的表示，能够泛化到不同场景；物体级别特征一致性提高，使检测能够在不同环境中保持稳定。

### 结论

MonoCLUE实现了遮挡和有限可见性情况下的稳健单目3D检测，在KITTI基准测试上达到了最先进的性能。

### 翻译

单目3D目标检测为自动驾驶提供经济有效的解决方案，但存在深度不适定和视野有限的问题。这些限制导致缺乏几何线索，在遮挡或截断场景中准确性降低。虽然最近的方法结合额外的深度信息来解决几何歧义，但它们忽略了稳健识别所必需的视觉线索。我们提出MonoCLUE，通过利用视觉特征的局部聚类和通用场景记忆来增强单目3D检测。首先，我们对视觉特征进行K-means聚类，以捕获不同级别的物体外观部分（如引擎盖、车顶），改进部分可见物体的检测。聚类特征在区域间传播，以捕获外观相似的物体。其次，我们通过跨图像聚合聚类特征构建通用场景记忆，提供能够泛化到不同场景的一致表示。这提高了物体级别特征的一致性，使检测能够在不同环境中保持稳定。最后，我们将局部聚类特征和通用场景记忆整合到物体查询中，引导注意力指向信息丰富的区域。利用统一的局部聚类和通用场景记忆策略，MonoCLUE实现了遮挡和有限可见性情况下的稳健单目3D检测，在KITTI基准测试上取得了最先进的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决单目3D目标检测中的两个核心问题：一是'病态深度问题'(ill-posed depth)，即由于缺乏视点差异导致难以准确推断物体深度；二是有限视野问题，即单张图像难以覆盖整个场景，特别是处理被遮挡或截断的物体。这些问题在自动驾驶领域至关重要，因为单目摄像头比多摄像头或LiDAR等传感器更经济实惠，但在遮挡、截断或复杂场景中现有方法性能会显著下降。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者注意到现有方法如MonoDETR和MonoDGP主要依赖深度信息解决几何歧义，但忽略了视觉线索对鲁棒识别的重要性。作者认为在单目设置中，物体中心、空间位置和方向等关键因素必须仅从外观推断，尤其在遮挡场景下仅依赖深度不足。该方法借鉴了多项现有工作：使用K-means聚类提取视觉特征、采用DETR架构进行目标检测、利用SAM生成的分割掩码指导聚类，并在MonoDGP基础上进行了改进，增加了局部聚类和场景记忆机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过局部聚类和广义场景记忆来增强单目3D检测，捕获多样化的视觉模式。整体流程包括：1)使用骨干网络提取多尺度特征；2)在物体区域应用K-means聚类，分离不同的物体级外观部分；3)通过相似度重新定位，将聚类特征传播到整个图像；4)构建广义场景记忆，聚合跨图像的聚类特征；5)将局部聚类特征和场景记忆初始化到对象查询中；6)使用2D和3D解码头进行最终检测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)局部聚类：在物体区域内应用K-means聚类，捕获不同外观部分；2)广义场景记忆：构建跨图像的共享表示，提供常见视觉模式；3)相似度重新定位：利用聚类特征识别被遮挡物体；4)查询初始化：将聚类特征集成到对象查询中。相比之前的工作，MonoCLUE更注重视觉线索而非仅依赖深度信息，使用物体形状掩码而非矩形掩码减少背景噪声，并结合局部和全局特征提供更全面的物体理解，在处理困难样本时表现更佳。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MonoCLUE通过局部聚类和广义场景记忆增强了单目3D目标检测，有效解决了遮挡和有限视野条件下的检测难题，在KITTI基准测试上达到了最先进的性能。'}


### 论文摘要

Monocular 3D object detection offers a cost-effective solution for autonomous driving but suffers from ill-posed depth and limited field of view. These constraints cause a lack of geometric cues and reduced accuracy in occluded or truncated scenes. While recent approaches incorporate additional depth information to address geometric ambiguity, they overlook the visual cues crucial for robust recognition. We propose MonoCLUE, which enhances monocular 3D detection by leveraging both local clustering and generalized scene memory of visual features. First, we perform K-means clustering on visual features to capture distinct object-level appearance parts (e.g., bonnet, car roof), improving detection of partially visible objects. The clustered features are propagated across regions to capture objects with similar appearances. Second, we construct a generalized scene memory by aggregating clustered features across images, providing consistent representations that generalize across scenes. This improves object-level feature consistency, enabling stable detection across varying environments. Lastly, we integrate both local cluster features and generalized scene memory into object queries, guiding attention toward informative regions. Exploiting a unified local clustering and generalized scene memory strategy, MonoCLUE enables robust monocular 3D detection under occlusion and limited visibility, achieving state-of-the-art performance on the KITTI benchmark.

---

## 97. HENet++: Hybrid Encoding and Multi-task Learning for 3D Perception and End-to-end Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.07106v1](http://arxiv.org/abs/2511.07106v1)

**作者:** Zhongyu Xia, Zhiwei Lin, Yongtao Wang, Ming-Hsuan Yang

**发布时间:** 2025-11-10

**备注:** Preliminary version, 19 pages

### GPT解析

### 总结

本文提出了HENet和HENet++框架，用于解决多任务三维感知和端到端自动驾驶中的计算资源限制和特征表示差异问题。

### 背景

三维特征提取是自动驾驶系统的关键组成部分，包括3D目标检测、鸟瞰图语义分割和占用预测等感知任务。大型图像编码器、高分辨率图像和长期时间输入可提高特征质量，但受计算资源限制难以兼容。

### 目的

开发一个能够处理多任务三维感知的端到端框架，解决不同任务特征表示差异和计算资源限制问题。

### 方法

提出混合图像编码网络，对短期帧使用大型图像编码器，对长期帧使用小型编码器；同时提取密集和稀疏特征，为不同任务提供更合适的表示；保持与现有三维特征提取方法的兼容性，支持多模态输入。

### 主要发现

HENet++在nuScenes基准测试上实现了最先进的端到端多任务三维感知结果，同时在nuScenes端到端自动驾驶基准测试上达到了最低碰撞率。

### 结论

HENet框架有效解决了多任务三维感知中的计算资源限制和特征表示差异问题，实现了高性能的端到端自动驾驶。

### 翻译

三维特征提取是自动驾驶系统的关键组成部分，其中3D目标检测、鸟瞰图(BEV)语义分割和占用预测等感知任务对三维特征形成重要约束。虽然大型图像编码器、高分辨率图像和长期时间输入可以显著提高特征质量并带来显著的性能提升，但由于计算资源限制，这些技术在训练和推理阶段往往难以兼容。此外，不同任务倾向于不同的特征表示，使得单一模型难以在多任务端到端推理中保持与单任务模型相当的准确性。为解决这些问题，我们提出了用于多任务三维感知和端到端自动驾驶的HENet和HENet++框架。具体而言，我们提出了一种混合图像编码网络，对短期帧使用大型图像编码器，对长期帧使用小型编码器。此外，我们的框架同时提取密集和稀疏特征，为不同任务提供更合适的表示，减少累积误差，并为规划模块提供更全面的信息。所提出的架构与各种现有三维特征提取方法保持兼容，并支持多模态输入。HENet++在nuScenes基准测试上实现了最先进的端到端多任务三维感知结果，同时在nuScenes端到端自动驾驶基准测试上达到了最低碰撞率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶系统中三个核心挑战：1) 高分辨率图像、大编码网络和长时序输入之间的计算资源冲突；2) 不同3D感知任务(如3D物体检测、BEV语义分割、占用预测)偏爱不同特征表示，导致多任务学习困难；3) 如何充分利用多模态信息构建统一的端到端自动驾驶框架。这些问题在现实中非常重要，因为3D感知是自动驾驶的基础，直接影响车辆与环境的交互能力；计算资源限制限制了高性能感知系统的部署；多任务学习能提高系统效率；而端到端自动驾驶是未来发展方向，但面临多任务整合和性能平衡的挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性进行设计：BEV方法在物体检测上不如稀疏查询方法，密集特征方法不如稀疏特征方法，不同任务需要不同特征表示。他们借鉴了BEVFormer、BEVDet等BEV方法构建特征，参考SparseBEV进行稀疏特征提取，将自然语言处理中的模型合并技术应用于多任务预训练，并吸收了UniAD、VAD等端到端自动驾驶方法的设计思想。在此基础上，作者创新性地设计了混合编码策略，同时提取稀疏和密集特征，并基于模型合并提出预训练策略，最终构建了支持多模态输入的端到端自动驾驶框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1) 混合编码-用大网络处理少量高分辨率帧，小网络处理长序列低分辨率帧；2) 稀疏-密集协同-同时提取稀疏前景特征和密集背景特征；3) 多任务学习-通过独立编码器和模型合并预训练提高性能；4) 端到端自动驾驶-利用提取特征进行预测和规划。整体流程：HENet框架首先通过混合图像编码网络处理短帧和长帧，然后通过时间特征集成模块融合多帧BEV特征，再通过独立BEV特征编码为不同任务分配特征，最后由任务特定解码器输出结果。HENet++在此基础上增加了稀疏特征提取和模型合并预训练。端到端自动驾驶框架则利用这些特征设计基于注意力的轨迹规划器，支持雷达和相机多模态输入，实现预测和规划。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 混合编码策略，兼容密集和稀疏特征提取；2) 稀疏-密集协同框架，同时为不同任务提供适合特征；3) 基于模型合并的预训练策略，提高多任务性能；4) 首个支持雷达和相机多模态输入的端到端自动驾驶方法。相比之前的工作，不同之处在于：同时处理三个3D感知任务而非单任务；使用稀疏和密集双特征表示而非单一特征；通过混合编码在保持高性能同时降低计算成本；首次实现多模态端到端自动驾驶；创新的预训练方法提升多任务性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了HENet++框架，通过混合编码和稀疏-密集协同特征提取，实现了高效的多任务3D感知和端到端自动驾驶，在保持高性能的同时显著降低了计算成本，并首次实现了雷达和相机多模态输入的端到端自动驾驶。'}


### 论文摘要

Three-dimensional feature extraction is a critical component of autonomous driving systems, where perception tasks such as 3D object detection, bird's-eye-view (BEV) semantic segmentation, and occupancy prediction serve as important constraints on 3D features. While large image encoders, high-resolution images, and long-term temporal inputs can significantly enhance feature quality and deliver remarkable performance gains, these techniques are often incompatible in both training and inference due to computational resource constraints. Moreover, different tasks favor distinct feature representations, making it difficult for a single model to perform end-to-end inference across multiple tasks while maintaining accuracy comparable to that of single-task models. To alleviate these issues, we present the HENet and HENet++ framework for multi-task 3D perception and end-to-end autonomous driving. Specifically, we propose a hybrid image encoding network that uses a large image encoder for short-term frames and a small one for long-term frames. Furthermore, our framework simultaneously extracts both dense and sparse features, providing more suitable representations for different tasks, reducing cumulative errors, and delivering more comprehensive information to the planning module. The proposed architecture maintains compatibility with various existing 3D feature extraction methods and supports multimodal inputs. HENet++ achieves state-of-the-art end-to-end multi-task 3D perception results on the nuScenes benchmark, while also attaining the lowest collision rate on the nuScenes end-to-end autonomous driving benchmark.

---

## 98. Relative Energy Learning for LiDAR Out-of-Distribution Detection

**论文链接:** [http://arxiv.org/abs/2511.06720v2](http://arxiv.org/abs/2511.06720v2)

**作者:** Zizhao Li, Zhengkang Xiang, Jiayang Ao, Joseph West, Kourosh Khoshelham

**发布时间:** 2025-11-10

**备注:** The code and checkpoints will be released after paper acceptance

### GPT解析

### 总结

本文提出了一种名为相对能量学习(REL)的框架，用于激光雷达点云的分布外(OOD)检测，通过利用正负logits之间的能量差距作为相对评分函数，并采用轻量级数据合成策略Point Raise生成辅助异常，在基准测试中显著优于现有方法。

### 背景

OOD检测是可靠自动驾驶的关键需求，当前激光雷达OOD方法难以区分罕见异常和常见类别，导致高误报率和过度自信错误。虽然2D图像上的OOD检测研究丰富，但直接应用于3D激光雷达点云效果不佳。

### 目的

开发一种有效的激光雷达点云OOD检测框架，提高自动驾驶系统在识别训练分布外物体时的可靠性。

### 方法

提出相对能量学习(REL)框架，利用正(分布内)和负logits之间的能量差距作为相对评分函数，缓解原始能量值校准问题；同时提出Point Raise数据合成策略，通过扰动现有点云生成辅助异常而不改变内点语义。

### 主要发现

在SemanticKITTI和STU基准测试上，REL框架以较大优势超越现有方法；建模相对能量与简单合成异常相结合，为开放世界自动驾驶中的可靠OOD检测提供了有效解决方案。

### 结论

相对能量学习框架结合简单合成异常策略，为开放世界自动驾驶中的可靠OOD检测提供了原则性且可扩展的解决方案。

### 翻译

分布外(OOD)检测是可靠自动驾驶的关键需求，其中安全性依赖于识别训练分布外的道路障碍物和意外物体。尽管在2D图像上有大量关于OOD检测的研究，但直接转移到3D激光雷达点云已被证明无效。当前的激光雷达OOD方法难以区分罕见异常与常见类别，导致在安全关键场景中出现高误报率和过度自信的错误。我们提出了相对能量学习(REL)，这是一种用于激光雷达点云OOD检测的简单而有效的框架。REL利用正(分布内)和负logits之间的能量差距作为相对评分函数，缓解原始能量值中的校准问题，并提高各种场景的鲁棒性。为解决训练期间缺少OOD样本的问题，我们提出了一种名为Point Raise的轻量级数据合成策略，它通过扰动现有点云来生成辅助异常，而不改变内点语义。在SemanticKITTI和意外发现(STU)基准上的评估表明，REL始终以较大优势优于现有方法。我们的研究结果表明，建模相对能量与简单的合成异常相结合，为开放世界自动驾驶中的可靠OOD检测提供了原则性和可扩展的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶系统中LiDAR点云数据的分布外（OOD）检测问题。这个问题在现实中至关重要，因为自动驾驶系统的安全性依赖于识别训练数据分布之外的障碍物和意外对象（如道路碎片、施工设备等）。这些对象在训练数据中稀缺但对安全构成严重威胁，如果被错误分类，车辆可能无法适当反应，导致事故。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有LiDAR OOD检测方法的局限性（如难以区分罕见异常与常见类别，高误报率）进行思考。他们借鉴了能量基础模型（EBMs）的思想，但发现传统方法存在超参数依赖和类别不平衡问题。作者设计了两步策略：一是尊重LiDAR几何特性的数据合成（Point Raise），二是基于相对能量的学习框架（REL）。这些方法基于Mask4Former-3D架构构建，并融入了对比学习和不平衡感知的损失函数设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过正（内部分布）和负（外部分布）对数之间的能量间隙作为相对评分函数，创建更可靠的OOD检测边界。整体流程包括：1）构建基于Mask4Former-3D的模型，添加轻量级OOD分支；2）使用Point Raise算法从道路点生成伪OOD样本；3）训练时使用REL目标函数，优化ID与OOD样本间的能量边界；4）推理时通过相对能量值判断每个点是否为OOD。Point Raise通过径向收缩和高度扰动将道路点转换为逼真的异常结构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）相对能量学习（REL）框架，利用正负对数间的能量间隙作为评分，消除超参数调优需求；2）Point Raise算法，轻量级合成逼真伪OOD数据，不依赖外部数据集；3）不平衡感知的损失函数，处理LiDAR数据中严重的类别不平衡。相比之前工作，REL显著降低了误报率（FPR@95降低36%），解决了传统方法对罕见异常检测能力不足的问题，且在保持封闭集分割性能的同时实现了更可靠的OOD检测。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出的相对能量学习（REL）和点提升（Point Raise）方法有效解决了自动驾驶中LiDAR点云的分布外检测问题，显著提高了异常检测准确性并大幅降低了误报率，为开放世界自动驾驶场景提供了更可靠的感知解决方案。'}


### 论文摘要

Out-of-distribution (OOD) detection is a critical requirement for reliable autonomous driving, where safety depends on recognizing road obstacles and unexpected objects beyond the training distribution. Despite extensive research on OOD detection in 2D images, direct transfer to 3D LiDAR point clouds has been proven ineffective. Current LiDAR OOD methods struggle to distinguish rare anomalies from common classes, leading to high false-positive rates and overconfident errors in safety-critical settings. We propose Relative Energy Learning (REL), a simple yet effective framework for OOD detection in LiDAR point clouds. REL leverages the energy gap between positive (in-distribution) and negative logits as a relative scoring function, mitigating calibration issues in raw energy values and improving robustness across various scenes. To address the absence of OOD samples during training, we propose a lightweight data synthesis strategy called Point Raise, which perturbs existing point clouds to generate auxiliary anomalies without altering the inlier semantics. Evaluated on SemanticKITTI and the Spotting the Unexpected (STU) benchmark, REL consistently outperforms existing methods by a large margin. Our results highlight that modeling relative energy, combined with simple synthetic outliers, provides a principled and scalable solution for reliable OOD detection in open-world autonomous driving.

---

## 99. SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2511.06702v1](http://arxiv.org/abs/2511.06702v1)

**作者:** Yifan Wang, Yian Zhao, Fanqi Pu, Xiaochen Yang, Yang Tang, Xi Chen, Wenming Yang

**发布时间:** 2025-11-10

### GPT解析

### 总结

SPAN是一种新的单目3D检测方法，通过空间点对齐和3D-2D投影对齐两个关键组件，解决了传统解耦预测范式中的几何一致性问题，并引入分层任务学习策略确保训练稳定性。

### 背景

现有的单目3D检测器通常通过解耦预测范式来处理3D边界框的显著非线性回归，该范式使用多个分支分别估计几何中心、深度、尺寸和旋转角度。这种解耦策略虽然简化了学习过程，但忽略了不同属性之间的几何协同约束，导致缺乏几何一致性先验，造成次优性能。

### 目的

解决现有单目3D检测器中解耦预测范式忽略几何协同约束的问题，提高检测性能。

### 方法

提出空间投影对齐(Span)方法，包含两个关键组件：(1)空间点对齐：强制预测和真实3D边界框之间的显式全局空间约束，纠正空间漂移；(2)3D-2D投影对齐：确保3D框在图像平面上的投影与2D检测边界框紧密对齐。同时引入分层任务学习策略，随着3D属性预测的完善逐步纳入空间投影对齐，防止早期误差传播。

### 主要发现

大量实验证明，SPAN方法可以轻松集成到任何既定的单目3D检测器中，并带来显著的性能提升。

### 结论

通过解决解耦预测中的几何一致性问题，SPAN方法有效提高了单目3D检测的性能。

### 翻译

现有的单目3D检测器通常通过解耦预测范式来处理3D边界框的显著非线性回归，该范式使用多个分支分别估计几何中心、深度、尺寸和旋转角度。尽管这种解耦策略简化了学习过程，但它本质上忽略了不同属性之间的几何协同约束，导致缺乏几何一致性先验，从而造成次优性能。为解决此问题，我们提出了具有两个关键组件的新型空间投影对齐(Span)方法：(i)空间点对齐强制预测和真实3D边界框之间的显式全局空间约束，从而纠正解耦属性回归引起的空间漂移；(ii)3D-2D投影对齐确保3D框在图像平面上的投影与其对应的2D检测边界框紧密对齐，减轻先前工作中被忽视的投影不对齐问题。为确保训练稳定性，我们进一步引入了分层任务学习策略，随着3D属性预测的逐步完善，逐步纳入空间投影对齐，防止早期阶段跨属性的误差传播。大量实验证明，提出的方法可以轻松集成到任何既定的单目3D检测器中，并带来显著的性能提升。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决单目3D目标检测中解耦预测范式忽略几何协作约束的问题。现有方法通过多个分支独立估计3D边界框的几何中心、深度、尺寸和旋转角度，虽然简化了学习过程，但忽略了不同属性间的几何关系，导致预测结果缺乏几何一致性，影响定位精度。这个问题很重要，因为3D目标检测是自动驾驶和机器人感知的基础，而单目3D检测相比激光雷达或双目相机具有成本优势和部署灵活性，提高其精度对实际应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有单目3D检测方法的局限性，发现解耦预测范式忽略了不同属性间的几何协作约束。他们借鉴了GUPNet的分层任务学习策略，但进行了改进；参考了ROI-10D和MonoDIS等回归角点作为辅助任务的方法，但创新性地将损失函数应用于从主要7自由度属性导出的八个角点坐标；同时受到Deep3DBox等分析求解器的启发，但开发了更稳健的可微方法。作者设计了两阶段解决方案：空间点对齐强制预测3D框与真实3D框在空间上对齐，3D-2D投影对齐确保3D框投影与2D检测框一致，并引入分层任务学习策略确保训练稳定性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是对3D边界框的不同属性施加几何协作约束，联合优化以增强空间和投影一致性。整体流程：1)检测器预测2D和3D属性；2)基于预测属性计算3D边界框的八个角点；3)空间点对齐阶段使用MGIoU损失约束预测角点与真实角点在3D空间中对齐；4)3D-2D投影对齐阶段将角点投影到2D平面，使用2D GIoU损失确保投影区域的最小包围矩形与2D检测框对齐；5)通过分层任务学习策略动态调整各损失权重，确保训练稳定性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)空间点对齐：将损失函数应用于从主要7自由度属性导出的角点坐标，直接正则化主要框参数；2)3D-2D投影对齐：提供可微的、基于梯度的学习信号，而非对噪声敏感的分析求解器；3)分层任务学习策略：动态调整损失权重，确保训练稳定性。相比之前工作，本文首次明确集成了空间投影对齐到端到端框架中，解决了几何协作约束不足的问题；方法可作为即插即用模块集成到现有检测器中，无需额外架构修改；通过动态权重调整克服了早期训练不稳定性问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了空间投影对齐（SPAN）方法，通过空间点对齐和3D-2D投影对齐两个关键组件，结合分层任务学习策略，解决了单目3D目标检测中解耦预测范式忽略几何协作约束的问题，显著提高了检测精度和几何一致性。'}


### 论文摘要

Existing monocular 3D detectors typically tame the pronounced nonlinear regression of 3D bounding box through decoupled prediction paradigm, which employs multiple branches to estimate geometric center, depth, dimensions, and rotation angle separately. Although this decoupling strategy simplifies the learning process, it inherently ignores the geometric collaborative constraints between different attributes, resulting in the lack of geometric consistency prior, thereby leading to suboptimal performance. To address this issue, we propose novel Spatial-Projection Alignment (SPAN) with two pivotal components: (i). Spatial Point Alignment enforces an explicit global spatial constraint between the predicted and ground-truth 3D bounding boxes, thereby rectifying spatial drift caused by decoupled attribute regression. (ii). 3D-2D Projection Alignment ensures that the projected 3D box is aligned tightly within its corresponding 2D detection bounding box on the image plane, mitigating projection misalignment overlooked in previous works. To ensure training stability, we further introduce a Hierarchical Task Learning strategy that progressively incorporates spatial-projection alignment as 3D attribute predictions refine, preventing early stage error propagation across attributes. Extensive experiments demonstrate that the proposed method can be easily integrated into any established monocular 3D detector and delivers significant performance improvements.

---

## 100. On Accurate and Robust Estimation of 3D and 2D Circular Center: Method and Application to Camera-Lidar Calibration

**论文链接:** [http://arxiv.org/abs/2511.06611v1](http://arxiv.org/abs/2511.06611v1)

**作者:** Jiajun Jiang, Xiao Hu, Wancheng Liu, Wei Jiang

**发布时间:** 2025-11-10

### GPT解析

### 总结

本文提出了一种基于几何原理的框架，用于解决LiDAR-相机外参校准中圆形目标的3D-2D圆形中心对应问题，包含两个创新方法，并在合成和真实数据集上验证了其优越性。

### 背景

圆形目标因具有几何一致性和易于检测的特点，被广泛应用于LiDAR-相机外参校准中。

### 目的

解决实现准确3D-2D圆形中心对应的挑战，克服现有方法因解耦3D拟合和错误2D椭圆中心估计而失败的问题。

### 方法

提出一个基于几何原理的框架，包含两个创新：(i)基于共形几何代数和RANSAC的鲁棒3D圆心估计器；(ii)弦长方差最小化方法，通过单应性验证或准RANSAC回退解决2D投影中心的双最小值模糊性。

### 主要发现

该框架在合成和真实数据集上显著优于最先进的方法，降低了外参估计误差，实现了跨不同传感器和目标类型（包括自然圆形物体）的鲁棒校准。

### 结论

所提出的框架有效解决了LiDAR-相机外参校准中圆形目标的3D-2D圆形中心对应问题，提高了校准的准确性和鲁棒性。

### 翻译

圆形目标因其在LiDAR-相机外参校准中的几何一致性和易于检测的特点而被广泛使用。然而，实现准确的3D-2D圆形中心对应仍然具有挑战性。现有方法常常因解耦的3D拟合和错误的2D椭圆中心估计而失败。为此，我们提出一个基于几何原理的框架，具有两个创新：(i)基于共形几何代数和RANSAC的鲁棒3D圆心估计器；(ii)弦长方差最小化方法，用于恢复真实的2D投影中心，通过单应性验证或准RANSAC回退解决其双最小值模糊性。在合成和真实数据集上的评估表明，我们的框架显著优于最先进的方法。它降低了外参估计误差，实现了跨不同传感器和目标类型（包括自然圆形物体）的鲁棒校准。我们的代码将公开发布以供复现。


### 论文摘要

Circular targets are widely used in LiDAR-camera extrinsic calibration due to their geometric consistency and ease of detection. However, achieving accurate 3D-2D circular center correspondence remains challenging. Existing methods often fail due to decoupled 3D fitting and erroneous 2D ellipse-center estimation. To address this, we propose a geometrically principled framework featuring two innovations: (i) a robust 3D circle center estimator based on conformal geometric algebra and RANSAC; and (ii) a chord-length variance minimization method to recover the true 2D projected center, resolving its dual-minima ambiguity via homography validation or a quasi-RANSAC fallback. Evaluated on synthetic and real-world datasets, our framework significantly outperforms state-of-the-art approaches. It reduces extrinsic estimation error and enables robust calibration across diverse sensors and target types, including natural circular objects. Our code will be publicly released for reproducibility.

---

## 101. 论文ID: 2511.05791v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.05791v1.json'

---

## 102. X-Ray Observations of Old Nearby Supernovae: Constraints on Compact Object Populations and Late Interaction

**论文链接:** [http://arxiv.org/abs/2511.03539v1](http://arxiv.org/abs/2511.03539v1)

**作者:** Julia Ahlvind, Josefin Larsson, Dennis Alp

**发布时间:** 2025-11-05

**备注:** 34 pages, 13 Figures, 7 Tables. Accepted for publication in ApJ

### GPT解析

### 总结

研究通过分析242个附近超新星的晚期X射线观测数据，探索了核心坍缩超新星形成的致密天体群体的性质，检测到12个超新星的X射线辐射，并发现平均初始自旋周期较长的脉冲星群体更受观测支持。

### 背景

核心坍缩超新星形成的致密天体群体的性质不确定，而爆炸后数年至数十年后的X射线观测可以提供线索，因为随着喷流吸收减少，来自中心区域的硬X射线辐射会出现。

### 目的

分析附近超新星的晚期X射线辐射，通过吸收模型研究致密天体的性质，特别是脉冲星及其与周围介质的相互作用。

### 方法

分析242个附近超新星的607次X射线观测数据，使用来自Chandra、XMM-Newton、Swift和NuSTAR的观测，并应用基于3D模拟的吸收模型来考虑非对称喷流对致密天体辐射的吸收。

### 主要发现

检测到12个超新星的X射线辐射(包括4个首次检测到的)；这些超新星的X射线谱大多与周围介质相互作用一致，SN 1979C显示额外硬成分可能来自脉冲星风云；群体合成表明平均初始自旋周期较长的脉冲星群体更受青睐；高光度超新星表明与致密壳层的相互作用。

### 结论

晚期X射线观测是研究核心坍缩超新星形成的致密天体群体的有效方法；脉冲星的自旋周期特性受到观测限制；CSM相互作用是超新星晚期辐射的重要机制。

### 翻译

核心坍缩超新星形成的致密天体群体的性质不确定。爆炸后数年至数十年后的X射线观测可以提供线索，因为随着喷流吸收减少，来自中心区域的硬X射线辐射会出现。我们使用Chandra、XMM-Newton、Swift和NuSTAR的607次观测数据，分析和限制了242个附近超新星的晚期X射线辐射上限。我们使用基于3D模拟的吸收模型来考虑非对称喷流对致密天体辐射的吸收。我们检测到12个超新星的X射线辐射，包括4个首次检测到的(SN 1982R, SN 1984J, SN 1992bu, 和 SN 2003gk)，以及其他几个在比以往更晚时间点的超新星。这些超新星的X射线谱与周围介质相互作用一致，除了SN 1979C，它显示额外的硬成分，如早期研究中所述。这种辐射可能来自脉冲星风云。使用整个样本的上限，我们还进行了群体合成以限制产生脉冲星的超新星比例和脉冲星本身的性质。我们发现平均初始自旋周期较长的脉冲星群体更受青睐。最后，我们注意到几个具有CSM相互作用的超新星的高光度意味着与致密壳层的相互作用。


### 论文摘要

The properties of the population of compact objects created in core-collapse supernovae (SNe) are uncertain. X-ray observations years to decades after the explosions offer a way to gain insight into this, as hard X-ray emission from the central regions will emerge as the ejecta absorption decreases. Here we analyze and place upper limits on late-time X-ray emission in 242 nearby SNe, using 607 observations from Chandra, XMM-Newton, Swift, and NuSTAR. We use absorption models based on 3D simulations of neutrino-driven explosions to account for absorption of emission from the compact objects by the asymmetric ejecta. We detect X-ray emission from 12 SNe, including four for the first time (SN 1982R, SN 1984J, SN 1992bu, and SN 2003gk), and several of the others at later epochs than before. The X-ray spectra of these SNe are consistent with interaction with the circumstellar medium (CSM), with the possible exception of SN 1979C, which shows an additional hard component, as also noted in previous studies at earlier epochs. This emission may be due to a pulsar wind nebula. Using the upper limits in the full sample, we also perform a population synthesis to constrain the fraction of SNe that produce pulsars and the properties of the pulsars themselves. We find that pulsar populations with mean initial spin periods $\gtrsim100\rm~ ms$ are favored. Finally, we note that the high luminosities of several of the SNe with CSM interaction imply interactions with dense shells.

---

## 103. VCode: a Multimodal Coding Benchmark with SVG as Symbolic Visual Representation

**论文链接:** [http://arxiv.org/abs/2511.02778v1](http://arxiv.org/abs/2511.02778v1)

**作者:** Kevin Qinghong Lin, Yuhao Zheng, Hangyu Ran, Dantong Zhu, Dongxing Mao, Linjie Li, Philip Torr, Alex Jinpeng Wang

**发布时间:** 2025-11-04

**备注:** Project page: https://csu-jpg.github.io/VCode Github: https://github.com/CSU-JPG/VCode

### GPT解析

### 总结

该研究引入了VCode基准测试，评估视觉中心编码能力，并提出了VCoder框架来增强视觉语言模型在SVG代码生成方面的表现。

### 背景

代码在智能体时代已成为精确且可执行的推理和行动媒介，但现有研究主要集中于语言中心任务，如程序合成和调试，而视觉中心编码研究不足。

### 目的

受人类通过草图推理的启发，提出SVG代码作为紧凑、可解释且可执行的可视化表示，并建立基准测试评估模型从图像生成保留符号意义的SVG的能力。

### 方法

VCode覆盖三个领域：通用常识(MM-Vet)、专业学科(MMMU)和视觉中心感知(CV-Bench)；提出CodeVQA评估协议；引入VCoder框架，通过'修订思考'和'视觉工具行动'两个维度增强VLM能力。

### 主要发现

最先进的VLM在生成忠实SVG方面表现不佳，揭示了语言中心编码和视觉中心编码之间的差距；VCoder比表现最好的Claude-4-Opus提高了12.3分；人类和VLM在渲染的SVG上表现较差但具有一致性。

### 结论

视觉中心编码是一个重要但被忽视的研究领域，VCoder框架能有效提升模型性能，VCode基准测试和相关代码已公开发布。

### 翻译

代码已成为智能体时代精确且可执行的推理和行动媒介。然而，研究进展主要集中在语言中心任务上，如程序合成和调试，而视觉中心编码研究不足。受人类如何通过草图进行推理的启发，我们提倡SVG代码作为紧凑、可解释且可执行的可视化表示。我们引入了VCode基准测试，将多模态理解重新定义为代码生成：给定图像，模型必须生成保留符号意义以供下游推理的SVG。VCode涵盖三个领域：通用常识(MM-Vet)、专业学科(MMMU)和视觉中心感知(CV-Bench)。为评估符号保真度，我们提出了CodeVQA评估协议，其中策略模型对渲染的SVG进行问答；正确答案表示忠实的符号保留。实验表明，最先进的视觉语言模型难以生成忠实的SVG，揭示了语言中心编码和视觉中心编码之间的持续差距。为缩小这一差距，我们引入了VCoder框架，沿两个维度增强VLM：(i) 通过修订进行思考，迭代分析差异并改进SVG代码；(ii) 通过视觉工具行动，检测器和解析器提供超出模型内在能力的结构化线索，如对象、形状和文本。在基准测试中，具有强大推理能力的前沿VLM整体表现良好，但在专业知识和3D推理方面仍然有限。VCoder比表现最好的Claude-4-Opus整体提高了12.3分。人类研究表明，人类和VLM在渲染的SVG上表现较差，它们的一致性揭示了符号视觉表示的潜力。基准测试和代码可在https://github.com/CSU-JPG/VCode获取。


### 论文摘要

Code has emerged as a precise and executable medium for reasoning and action in the agent era. Yet, progress has largely focused on language-centric tasks such as program synthesis and debugging, leaving visual-centric coding underexplored. Inspired by how humans reason over sketches, we advocate SVG code as a compact, interpretable, and executable visual representation. We introduce VCode, a benchmark that reframes multimodal understanding as code generation: given an image, a model must produce SVG that preserves symbolic meaning for downstream reasoning. VCode covers three domains - general commonsense (MM-Vet), professional disciplines (MMMU), and visual-centric perception (CV-Bench). To assess symbolic fidelity, we propose CodeVQA, a novel evaluation protocol in which a policy model answers questions over rendered SVGs; correct answers indicate faithful symbolic preservation. Empirically, frontier VLMs struggle to generate faithful SVGs, revealing a persistent gap between language-centric and visual-centric coding. To close this gap, we introduce VCoder, an agentic framework that augments VLMs along two axes: (i) Thinking with Revision, which iteratively analyzes discrepancies and refines SVG code; and (ii) Acting with Visual Tools, where detectors and parsers supply structured cues such as objects, shapes, and text beyond the model's intrinsic capacity. Across benchmarks, frontier VLMs with strong reasoning capabilities score well overall yet remain limited in professional knowledge and 3D reasoning. VCoder delivers a 12.3-point overall gain over the top-performing Claude-4-Opus. Human studies show that both humans and VLMs perform worse on rendered SVGs, their consistency reveals the promise of symbolic visual representation. The benchmark and code are available at https://github.com/CSU-JPG/VCode.

---

## 104. RIS-Assisted 3D Spherical Splatting for Object Composition Visualization using Detection Transformers

**论文链接:** [http://arxiv.org/abs/2511.02573v2](http://arxiv.org/abs/2511.02573v2)

**作者:** Anastasios T. Sotiropoulos, Stavros Tsimpoukis, Dimitrios Tyrovolas, Sotiris Ioannidis, Panagiotis D. Diamantoulakis, George K. Karagiannidis, Christos K. Liaskos

**发布时间:** 2025-11-04

**备注:** Submitted to IEEE ICC 2026

### GPT解析

### 总结

本研究提出了一种基于可编程无线环境(PWE)的射频(RF)框架，用于三维物体重建，使用材料感知的球基元结合RIS enabled场合成与检测变换器(DETR)。

### 背景

传统光学方法在遮挡或低光照条件下表现不佳，而RF感知可以穿透材料并编码几何和成分信息，但不受控制的多径传播限制了重建精度。

### 目的

利用可编程无线环境(PWE)和可重构智能表面(RIS)来克服多径传播问题，提高RF感知的准确性，实现三维物体的精确重建和材料分类。

### 方法

提出了一种基于PWE驱动的RF框架，使用材料感知的球基元进行三维物体重建，结合RIS enabled场合成与检测变换器(DETR)，直接从提取的RF特征推断空间和材料参数。

### 主要发现

该框架能够近似物体几何形状并分类材料成分，总体准确率达到79.35%。

### 结论

这是迈向可编程和物理基础的RF-based 3D物体组成可视化的初步步骤，展示了RF感知在多媒体体验中的潜力。

### 翻译

对沉浸式和结构感知的多媒体体验的追求，增强了对超越可见光限制重建物体的感知模态的兴趣。传统光学管道在遮挡或低照度条件下会退化，促使使用射频(RF)感知，其电磁波可以穿透材料并编码几何和成分信息。然而，不受控制的多径传播限制了重建精度。可编程无线环境(PWEs)的最新进展通过通过可重构智能表面(RISs)实现传播的软件定义操作来缓解这一限制，从而提供可控的照明多样性。基于这一能力，这项工作引入了一种基于PWE驱动的RF框架，用于使用材料感知的球基元进行三维物体重建。所提出的方法结合了RIS enabled场合成与检测变换器(DETR)，该变换器直接从提取的RF特征推断空间和材料参数。模拟结果证实了该框架近似物体几何形状和分类材料组成的能力，总体准确率为79.35%，标志着向可编程和物理基础的RF-based 3D物体组成可视化的初步迈进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何通过无线电频率(RF)传感来重建三维物体的几何结构和材料成分问题。这个问题很重要，因为传统光学方法在物体被遮挡或光线不足时效果不佳，而RF信号能穿透材料获取光学方法无法获取的信息。同时，多路径传播问题限制了RF感知的准确性，且现有方法大多只关注几何结构重建，很少同时推断材料特性，因此需要一种能同时处理几何和材料信息的统一方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统光学方法的局限性和RF传感的优势，然后借鉴了多个领域的工作：1) 借鉴了3D高斯溅射(3DGS)概念，将其从光学领域扩展到RF域；2) 利用了可编程无线环境(PWE)和可重构智能表面(RIS)技术来控制多路径传播；3) 采用了检测变换器(DETR)架构进行目标检测；4) 参考了材料电磁特性研究为不同材料定义参数。作者将物体表示为球形基元集合，通过RIS控制电磁波传播，提取RF特征，最后用DETR模型推断基元参数。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将三维物体表示为由球形基元组成的集合，每个基元包含几何属性(位置、半径)和材料属性，利用可编程环境和RIS技术控制电磁波传播产生丰富信号，从接收的RF波前中提取物理特征，再用深度学习模型推断基元参数。整体流程：1) 设置包含RIS墙壁、发射器和接收器的PWE环境；2) 在多种RIS配置下接收RF波前并提取多种特征(极化、角度到达、延迟统计等)；3) 使用DETR模型训练，学习从RF特征推断球形基元参数；4) 应用置信度阈值过滤结果，形成最终三维重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 首次将3D高斯溅射概念扩展到RF域，使用球形基元同时重建几何结构和材料成分；2) 利用PWE和RIS技术通过软件控制产生多样化电磁波前；3) 提出材料感知的RF特征提取方法；4) 使用DETR进行端到端重建。相比之前工作：不依赖先验光学信息，直接从RF恢复表面；同时推断材料成分而非仅几何结构；使用球形基元而非体素更高效；采用端到端学习框架结合物理约束确保重建合理性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种创新的基于可编程无线环境和可重构智能表面的RF传感框架，首次将3D高斯溅射概念扩展到无线电频率域，通过球形基元表示同时重建三维物体的几何结构和材料成分，实现了从RF波前中直接提取物理可解释的物体表示。'}


### 论文摘要

The pursuit of immersive and structurally aware multimedia experiences has intensified interest in sensing modalities that reconstruct objects beyond the limits of visible light. Conventional optical pipelines degrade under occlusion or low illumination, motivating the use of radio-frequency (RF) sensing, whose electromagnetic waves penetrate materials and encode both geometric and compositional information. Yet, uncontrolled multipath propagation restricts reconstruction accuracy. Recent advances in Programmable Wireless Environments (PWEs) mitigate this limitation by enabling software-defined manipulation of propagation through Reconfigurable Intelligent Surfaces (RISs), thereby providing controllable illumination diversity. Building on this capability, this work introduces a PWE-driven RF framework for three-dimensional object reconstruction using material-aware spherical primitives. The proposed approach combines RIS-enabled field synthesis with a Detection Transformer (DETR) that infers spatial and material parameters directly from extracted RF features. Simulation results confirm the framework's ability to approximate object geometries and classify material composition with an overall accuracy of 79.35%, marking an initial step toward programmable and physically grounded RF-based 3D object composition visualization.

---

## 105. All You Need for Object Detection: From Pixels, Points, and Prompts to Next-Gen Fusion and Multimodal LLMs/VLMs in Autonomous Vehicles

**论文链接:** [http://arxiv.org/abs/2510.26641v1](http://arxiv.org/abs/2510.26641v1)

**作者:** Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Hazim Alzorgan, Ahmad Sarlak, Mahlagha Fazeli, Abolfazl Razi

**发布时间:** 2025-10-30

### GPT解析

### 总结

这是一篇关于自动驾驶车辆物体检测的前沿综述，重点关注新兴技术范式如视觉语言模型、大语言模型和生成式AI，而非过时技术。文章系统回顾了传感器技术、数据集分类和最新检测方法，为领域提供明确发展路线图。

### 背景

自动驾驶车辆通过智能感知、决策和控制系统的进步正在改变未来交通。然而，其成功依赖于在复杂和多模态环境中可靠地检测物体的核心能力。尽管计算机视觉和人工智能的最新突破推动了显著进展，但知识在多模态感知、上下文推理和协作智能方面仍然分散，面临关键挑战。

### 目的

弥合自动驾驶车辆物体检测领域中知识分散的差距，提供前瞻性分析，强调新兴范式如视觉语言模型、大语言模型和生成式AI，而非过时技术。

### 方法

系统性地回顾自动驾驶车辆传感器（摄像头、超声波、LiDAR和雷达）及其融合策略；引入AV数据集的结构化分类，包括自车数据、基础设施数据和协作数据；分析从2D和3D管道到混合传感器融合的最先进检测方法，特别关注由视觉变换器、大语言模型和小语言模型驱动的变换器方法。

### 主要发现

各种传感器在动态驾驶环境中的能力和局限性；传感器融合与最新大语言模型/视觉语言模型感知框架的集成潜力；AV数据集的结构化分类和数据特征分析；基于变换器的检测方法等前沿技术的优势。

### 结论

通过综合多模态感知、传感器融合和先进AI方法，该综述为自动驾驶车辆物体检测领域提供了当前能力、开放挑战和未来机会的明确路线图，有助于推动技术发展。

### 翻译

自动驾驶车辆通过智能感知、决策和控制系统的进步正在改变未来交通。然而，它们的成功与一项核心能力紧密相关，即在复杂和多模态环境中可靠地检测物体。尽管计算机视觉和人工智能的最新突破已取得显著进展，但该领域仍面临一个关键挑战，因为知识在多模态感知、上下文推理和协作智能方面仍然分散。本综述通过提供自动驾驶车辆物体检测的前瞻性分析弥合了这一差距，强调视觉语言模型、大语言模型和生成式AI等新兴范式，而非重新审视过时技术。我们首先系统性地回顾了自动驾驶车辆传感器的基本范围（摄像头、超声波、LiDAR和雷达）及其融合策略，不仅突出了它们在动态驾驶环境中的能力和局限性，还强调了它们与最新大语言模型/视觉语言模型驱动的感知框架集成的潜力。接下来，我们引入了自动驾驶车辆数据集的结构化分类，超越了简单收集，将自车数据、基础设施数据和协作数据（如V2V、V2I、V2X、I2I）进行分类，随后对数据结构和特征进行交叉分析。最终，我们分析了从2D和3D管道到混合传感器融合的最先进检测方法，特别关注由视觉变换器、大语言模型和小语言模型以及视觉语言模型驱动的新兴变换器方法。通过综合这些观点，我们的综述为当前能力、开放挑战和未来机会提供了明确的路线图。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶车辆中物体检测知识碎片化的问题，弥合多模态感知、上下文推理和协作智能之间的知识鸿沟。这个问题很重要，因为自动驾驶车辆的成功依赖于在复杂环境中可靠地进行物体检测，而现有研究大多关注过时技术，忽视了新兴的视觉语言模型、大语言模型和生成式AI等前沿方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者采用系统性综述方法，检查了超过700篇研究论文，设计了一个三核心组件结构：AV传感器技术、AV数据集和物体检测方法。作者借鉴了2018-2025年间的重要综述论文，但指出了它们的局限性，如缺乏对LLMs/VLMs的讨论和数据集分类不系统等，从而构建了一个更全面的综述框架来填补这些空白。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提供一个全面、前瞻性的自动驾驶车辆物体检测综述，整合传统传感器技术与新兴的AI模型。整体流程包括：系统性文献收集与分析；结构化综述框架（传感器技术、数据集、检测方法）；比较分析各类传感器、数据集和方法；讨论开放问题和未来方向，为研究人员提供清晰的指南。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：全面的综述范围涵盖传统和新兴技术；结构化的传感器分析比较不同类型传感器的特性；创新的数据集分类基于自车、基础设施和通信范式；前瞻性的检测方法分析包括transformer驱动的模型。相比之前工作，这篇论文更新性更强（涵盖最新技术）、更全面（关注整体生态系统）、更有结构性（清晰的组织框架）且更具前瞻性（指明未来方向）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过整合传统传感器技术与新兴的多模态AI模型，为自动驾驶车辆中的物体检测提供了一个全面、前瞻性的综述框架，填补了现有研究在基础模型集成和协作感知方面的空白，并指明了未来研究方向。'}


### 论文摘要

Autonomous Vehicles (AVs) are transforming the future of transportation through advances in intelligent perception, decision-making, and control systems. However, their success is tied to one core capability, reliable object detection in complex and multimodal environments. While recent breakthroughs in Computer Vision (CV) and Artificial Intelligence (AI) have driven remarkable progress, the field still faces a critical challenge as knowledge remains fragmented across multimodal perception, contextual reasoning, and cooperative intelligence. This survey bridges that gap by delivering a forward-looking analysis of object detection in AVs, emphasizing emerging paradigms such as Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative AI rather than re-examining outdated techniques. We begin by systematically reviewing the fundamental spectrum of AV sensors (camera, ultrasonic, LiDAR, and Radar) and their fusion strategies, highlighting not only their capabilities and limitations in dynamic driving environments but also their potential to integrate with recent advances in LLM/VLM-driven perception frameworks. Next, we introduce a structured categorization of AV datasets that moves beyond simple collections, positioning ego-vehicle, infrastructure-based, and cooperative datasets (e.g., V2V, V2I, V2X, I2I), followed by a cross-analysis of data structures and characteristics. Ultimately, we analyze cutting-edge detection methodologies, ranging from 2D and 3D pipelines to hybrid sensor fusion, with particular attention to emerging transformer-driven approaches powered by Vision Transformers (ViTs), Large and Small Language Models (SLMs), and VLMs. By synthesizing these perspectives, our survey delivers a clear roadmap of current capabilities, open challenges, and future opportunities.

---

## 106. Fluorescence intensity correlations enable 3D imaging without sample rotations

**论文链接:** [http://arxiv.org/abs/2510.24386v2](http://arxiv.org/abs/2510.24386v2)

**作者:** Robert G. Radloff, Felix F. Zimmermann, Siqi Li, Stephan Kuschel, Anatoli Ulmer, Yanwen Sun, Takahiro Sato, Peihao Sun, Johann Haber, Diling Zhu, Miklós Tegze, Gyula Faigel, Matthew R. Ware, Jordan T. O'Neal, Jumpei Yamada, Taito Osaka, Robert Zierold, Carina Hedrich, Dimitrios Kazazis, Yasin Ekinci, Makina Yabashi, Ichiro Inoue, Andrew Aquila, Meng Liang, Agostino Marinelli, Tais Gorkhover

**发布时间:** 2025-10-28

### GPT解析

### 总结

本文介绍了一种利用X射线荧光强度相关性和超短X射线脉冲进行无透镜三维成像的新方法，能够在不旋转样品的情况下对非周期性物体进行纳米级成像。

### 背景

传统光和电子显微镜无法对厚样品提供纳米级元素特异性洞察；相干衍射成像方法虽能恢复三维纳米结构但需要样品旋转增加实验复杂性；X射线弹性散射具有高度方向性提供有限三维信息；X射线荧光主要各向同性发射，但一阶空间相干性传统上限制了纳米级荧光成像仅适用于单晶样品。

### 目的

开发一种不需要样品旋转的三维纳米成像方法，克服一阶空间相干性对非周期性样品成像的限制。

### 方法

利用超短X射线脉冲激发的X射线荧光强度相关性；使用钒箔作为样品，在亚200纳米X射线激光束焦点内照射；不改变样品方向，使用覆盖不同光子入射角的探测器区域记录16个不同的样品投影；通过沿像散平移荧光体积，系统观察投影变化。

### 主要发现

FEL诱导的荧光反映了实空间结构变化；荧光强度相关性包含非周期性物体的三维结构信息；可以在不旋转样品的情况下获取多个投影。

### 结论

建立了一种使用荧光强度相关性的无透镜三维成像新方法，适用于非周期性样品的三维成像，对材料科学、化学和纳米技术有广泛影响。

### 翻译

无透镜X射线成像为厚样品提供了超越传统光和电子显微镜范围的元素特异性纳米级洞察。相干衍射成像方法，如ptychographic tomography，可以恢复三维纳米结构，但需要大量样品旋转，增加了实验复杂性。来自单一样品方向的X射线弹性散射图案具有高度方向性，提供有限的结构三维信息。与X射线弹性散射不同，X射线荧光主要各向同性发射。然而，一阶空间相干性传统上限制了纳米级荧光成像仅适用于单晶样品。在这里，我们证明由超短X射线脉冲激发的X射线荧光强度相关性包含非周期性、静止物体的三维结构信息。在我们的实验中，我们在亚200纳米X射线激光束焦点内照射钒箔。在不改变样品方向的情况下，我们使用覆盖相对于X射线自由电子激光束不同光子入射角的探测器区域记录了16个不同的样品投影。当荧光体积沿像散平移时，投影系统性地变化，证实FEL诱导的荧光反映了实空间结构变化。我们的结果建立了一种使用荧光强度相关性的非周期性样品无透镜三维成像新方法，对材料科学、化学和纳米技术有广泛影响。


### 论文摘要

Lensless X-ray imaging provides element-specific nanoscale insights into thick samples beyond the reach of conventional light and electron microscopy. Coherent diffraction imaging (CDI) methods, such as ptychographic tomography, can recover three-dimensional (3D) nanoscale structures but require extensive sample rotation, adding complexity to experiments. X-ray elastic-scattering patterns from a single sample orientation are highly directional and provide limited 3D information about the structure. In contrast to X-ray elastic scattering, X-ray fluorescence is emitted mostly isotropically. However, first-order spatial coherence has traditionally limited nanoscale fluorescence imaging to single-crystalline samples. Here, we demonstrate that intensity correlations of X-ray fluorescence excited by ultrashort X-ray pulses contain 3D structural information of non-periodic, stationary objects. In our experiment, we illuminated a vanadium foil within a sub-200 nm X-ray laser beam focus. Without changing the sample orientation, we recorded 16 distinct specimen projections using detector regions covering different photon incidence angles relative to the X-ray free-electron laser (FEL) beam. The projections varied systematically as the fluorescing volume was translated along an astigmatism, confirming that FEL-induced fluorescence reflects real-space structural changes. Our results establish a new approach for lensless 3D imaging of non-periodic specimens using fluorescence intensity correlations, with broad implications for materials science, chemistry, and nanotechnology.

---

## 107. Generalizing Fair Clustering to Multiple Groups: Algorithms and Applications

**论文链接:** [http://arxiv.org/abs/2511.11539v1](http://arxiv.org/abs/2511.11539v1)

**作者:** Diptarka Chakraborty, Kushagra Chatterjee, Debarati Das, Tien-Long Nguyen

**发布时间:** 2025-11-14

**备注:** Accepted in AAAI 2026 for Oral Representation

### GPT解析

### 总结

本研究将'最近公平聚类'问题从两个组别的情况推广到任意数量(多于两个)组别的设置中，证明了问题的NP难性质，并提出了高效近似算法，同时改进了'公平相关聚类'的近似保证，并首次为'公平共识聚类'问题提供了近似算法。

### 背景

聚类是机器学习和数据分析中的基本任务，但传统聚类算法经常无法为多个受保护属性定义的各种边缘化社区提供公平表示，这通常是由训练数据中的偏差引起的。虽然Chakraborty等人提出了'最近公平聚类'问题，但他们的研究限制在只有两个组别的情况下，而实际数据通常涉及更多组别。

### 目的

将'最近公平聚类'问题的研究推广到任意数量(多于两个)组别的设置中，解决该问题的计算复杂性，并提出高效的近似算法，同时改进相关聚类问题的近似保证。

### 方法

研究采用理论分析的方法，证明问题的NP难性质，并提出近线性时间的近似算法来处理任意大小的多个组别。这些算法被进一步应用于改进'公平相关聚类'问题的近似保证，并为'公平共识聚类'问题提供首次近似算法。

### 主要发现

1. '最近公平聚类'问题在多个(多于两个)组别的情况下是NP难的，即使所有组别大小相等；2. 提出了能够高效处理任意大小多个组别的近线性时间近似算法；3. 利用这些算法改进了'公平相关聚类'问题的近似保证；4. 首次为涉及多个(多于两个)组别的'公平共识聚类'问题提供了近似算法。

### 结论

本研究解决了Chakraborty等人提出的一个开放性问题，将'最近公平聚类'问题从两个组别推广到任意数量组别，并提供了高效的近似算法。这些成果不仅推进了公平聚类理论的发展，也为实际应用提供了可行的解决方案。

### 翻译

聚类是机器学习和数据分析中的基本任务，但它经常无法为多个受保护属性定义的各种边缘化社区提供公平的表示——这一缺陷通常是由训练数据中的偏差引起的。因此，人们越来越需要提高聚类结果的公平性，理想情况下通过最小的修改，可能作为传统聚类后的后处理步骤。最近，Chakraborty等人[COLT'25]开始了对'最近公平聚类'的研究，尽管在受限场景下，数据点仅属于两个组别。然而，在实践中，数据点通常由许多组别来表征，反映了年龄、种族、性别等多样化的受保护属性。在这项工作中，我们将'最近公平聚类'问题的研究推广到具有任意数量(多于两个)组别的设置中。我们首先证明，即使所有组别大小相等，该问题也是NP难的——这与两个组别的情况形成鲜明对比，后者存在精确算法。接下来，我们提出了近线性时间的近似算法，能够高效处理任意大小的多个组别，从而回答了Chakraborty等人[COLT'25]提出的一个开放性问题。利用我们的最近公平聚类算法，我们进一步实现了'公平相关聚类'问题的改进近似保证，推进了Ahmadian等人[AISTATS'20]和Ahmadi等人[2020]建立的最先进结果。此外，我们首次为涉及多个(多于两个)组别的'公平共识聚类'问题提供了近似算法，从而解决了Chakraborty等人[COLT'25]指出的另一个开放方向。


### 论文摘要

Clustering is a fundamental task in machine learning and data analysis, but it frequently fails to provide fair representation for various marginalized communities defined by multiple protected attributes -- a shortcoming often caused by biases in the training data. As a result, there is a growing need to enhance the fairness of clustering outcomes, ideally by making minimal modifications, possibly as a post-processing step after conventional clustering. Recently, Chakraborty et al. [COLT'25] initiated the study of \emph{closest fair clustering}, though in a restricted scenario where data points belong to only two groups. In practice, however, data points are typically characterized by many groups, reflecting diverse protected attributes such as age, ethnicity, gender, etc.   In this work, we generalize the study of the \emph{closest fair clustering} problem to settings with an arbitrary number (more than two) of groups. We begin by showing that the problem is NP-hard even when all groups are of equal size -- a stark contrast with the two-group case, for which an exact algorithm exists. Next, we propose near-linear time approximation algorithms that efficiently handle arbitrary-sized multiple groups, thereby answering an open question posed by Chakraborty et al. [COLT'25].   Leveraging our closest fair clustering algorithms, we further achieve improved approximation guarantees for the \emph{fair correlation clustering} problem, advancing the state-of-the-art results established by Ahmadian et al. [AISTATS'20] and Ahmadi et al. [2020]. Additionally, we are the first to provide approximation algorithms for the \emph{fair consensus clustering} problem involving multiple (more than two) groups, thus addressing another open direction highlighted by Chakraborty et al. [COLT'25].

---

## 108. Learning and Testing Convex Functions

**论文链接:** [http://arxiv.org/abs/2511.11498v1](http://arxiv.org/abs/2511.11498v1)

**作者:** Renato Ferreira Pinto, Cassandra Marcussen, Elchanan Mossel, Shivam Nadimpalli

**发布时间:** 2025-11-14

**备注:** 43 pages

### GPT解析

### 总结

该研究探讨了高斯空间中实值凸函数的学习和测试问题，提出了在标准高斯测度下，假设函数样本可访问且满足Lipschitz光滑性条件下的算法和测试方法，并提供了相应的样本复杂度分析。

### 背景

尽管凸性在数学、统计学和计算机科学中得到了广泛研究，但其可学习性和可测试性大多仅在离散或受限环境中被研究，通常针对汉明距离，而汉明距离不适合实值函数。此外，即使在一维情况下，没有光滑性假设，也无法从有限样本中推断凸性。

### 目的

研究高维空间中实值凸函数的学习和测试问题，特别是在标准高斯测度下，假设函数样本可访问且满足Lipschitz光滑性条件。

### 方法

作者假设函数样本可访问且满足Lipschitz光滑性条件，并提出了一个针对Lipschitz凸函数的agnostic proper学习算法，一个容错的凸性测试算法，以及一个单向的凸性测试算法。

### 主要发现

1. 学习凸函数：一个能够在n^(O(1/ε²))样本内达到误差ε的agnostic proper学习算法，以及在相关统计查询(CSQ)模型中需要n^poly(1/ε)样本的下界。2. 测试凸函数：一个与学习算法具有相同样本复杂度的容错凸性测试算法，以及一个使用O((√n/ε)^n)样本的单向测试算法。

### 结论

该研究填补了实值凸函数在高斯空间中学习和测试的理论空白，提供了具有理论保证的算法和样本复杂度分析，为后续研究奠定了基础。

### 翻译

我们考虑高斯空间中实值凸函数的学习和测试问题。尽管凸性在数学、统计学和计算机科学中得到了广泛研究，但其可学习性和可测试性大多仅在离散或受限环境中被研究——通常针对汉明距离，而汉明距离不适合实值函数。相比之下，我们在高维空间中研究这些问题，采用标准高斯测度，假设函数样本可访问且满足一个温和的光滑性条件，即Lipschitz条件。光滑性假设是自然的，事实上，即使在一维情况下也是必要的：没有它，无法从有限样本中推断凸性。作为我们的主要结果，我们提供了：学习凸函数：一个针对Lipschitz凸函数的agnostic proper学习算法，使用n^(O(1/ε²))样本达到误差ε，以及在相关统计查询(CSQ)模型中需要n^poly(1/ε)样本的互补下界。测试凸函数：一个针对Lipschitz函数凸性的容错（双向）测试器，具有相同的样本复杂度（作为我们学习结果的一个推论），以及一个使用O((√n/ε)^n)样本的单向测试器（从不拒绝凸函数）。


### 论文摘要

We consider the problems of \emph{learning} and \emph{testing} real-valued convex functions over Gaussian space. Despite the extensive study of function convexity across mathematics, statistics, and computer science, its learnability and testability have largely been examined only in discrete or restricted settings -- typically with respect to the Hamming distance, which is ill-suited for real-valued functions.   In contrast, we study these problems in high dimensions under the standard Gaussian measure, assuming sample access to the function and a mild smoothness condition, namely Lipschitzness. A smoothness assumption is natural and, in fact, necessary even in one dimension: without it, convexity cannot be inferred from finitely many samples. As our main results, we give:   - Learning Convex Functions: An agnostic proper learning algorithm for Lipschitz convex functions that achieves error $\varepsilon$ using $n^{O(1/\varepsilon^2)}$ samples, together with a complementary lower bound of $n^{\mathrm{poly}(1/\varepsilon)}$ samples in the \emph{correlational statistical query (CSQ)} model.   - Testing Convex Functions: A tolerant (two-sided) tester for convexity of Lipschitz functions with the same sample complexity (as a corollary of our learning result), and a one-sided tester (which never rejects convex functions) using $O(\sqrt{n}/\varepsilon)^n$ samples.

---

## 109. Intrinsic Dimension Estimation for Radio Galaxy Zoo using Diffusion Models

**论文链接:** [http://arxiv.org/abs/2511.11490v1](http://arxiv.org/abs/2511.11490v1)

**作者:** Joan Font-Quer Roset, Devina Mohan, Anna Scaife

**发布时间:** 2025-11-14

**备注:** 9 pages, 5 figures, 2 tables, submitted to NeurIPS 2025 ML4PS Workshop

### GPT解析

### 总结

本研究使用基于分数的扩散模型估计了射电星系动物园(RGZ)数据集的内在维度(iD)，并分析了iD估计如何随贝叶斯神经网络(BNN)能量分数变化，该分数衡量了射电源与RGZ数据集的MiraBest子集的相似性。

### 背景

射电星系动物园(RGZ)数据集包含大量射电源图像，需要了解其内在维度以更好地理解数据特性和自监督学习算法的表现。

### 目的

估计RGZ数据集的内在维度，并研究其与能量分数、形态类别和信噪比之间的关系。

### 方法

采用基于分数的扩散模型，利用贝叶斯神经网络能量分数来评估射电源与RGZ数据集MiraBest子集的相似性，进而估计内在维度。

### 主要发现

分布外的源表现出更高的内在维度值；RGZ的整体内在维度超过自然图像数据集通常报告的值；内在维度在不同Fanaroff-Riley形态类别间无明显差异；内在维度较低时信噪比有轻微升高的趋势。

### 结论

RGZ数据集中内在维度与能量分数之间的关系可被用于定量研究和改进各种自监督学习算法所学习的表示。

### 翻译

在这项工作中，我们使用基于分数的扩散模型估计了射电星系动物园(RGZ)数据集的内在维度(iD)。我们检查了iD估计如何作为贝叶斯神经网络(BNN)能量分数的函数而变化，这些能量分数衡量了射电源与RGZ数据集的MiraBest子集的相似程度。我们发现分布外的源表现出更高的iD值，并且RGZ的整体iD超过自然图像数据集通常报告的值。此外，我们分析了iD如何随Fanaroff-Riley(FR)形态类别和信噪比(SNR)变化。虽然未发现FR I和FR II类别之间存在关系，但存在iD较低时SNR轻微升高的趋势。未来使用RGZ数据集的工作可以利用iD与能量分数之间的关系来定量研究和改进各种自监督学习算法所学习的表示。


### 论文摘要

In this work, we estimate the intrinsic dimension (iD) of the Radio Galaxy Zoo (RGZ) dataset using a score-based diffusion model. We examine how the iD estimates vary as a function of Bayesian neural network (BNN) energy scores, which measure how similar the radio sources are to the MiraBest subset of the RGZ dataset. We find that out-of-distribution sources exhibit higher iD values, and that the overall iD for RGZ exceeds those typically reported for natural image datasets. Furthermore, we analyse how iD varies across Fanaroff-Riley (FR) morphological classes and as a function of the signal-to-noise ratio (SNR). While no relationship is found between FR I and FR II classes, a weak trend toward higher SNR at lower iD. Future work using the RGZ dataset could make use of the relationship between iD and energy scores to quantitatively study and improve the representations learned by various self-supervised learning algorithms.

---

## 110. AI as a component in the action research tradition of learning-by-doing

**论文链接:** [http://arxiv.org/abs/2511.11445v1](http://arxiv.org/abs/2511.11445v1)

**作者:** Ian Benson, Alexei Semenov

**发布时间:** 2025-11-14

**备注:** 14 pages, 2 figures

### GPT解析

### 总结

本研究提出了一种基于行动研究、黑客精神和做中学的数学教育新模式，旨在取代传统的指令-执行工业模式。该模式强调自我意识、结构化绘图和形式图表，将数学教育建立在数学建模和程序设计专业数学家的活动基础上，并通过书面-口头对话和技术增强来促进学习。

### 背景

传统的19世纪工业模式数学教育强调指令和执行，而现代教育需要更有效的方法。练习和实践存在弱点，大型语言模型的统计预测也有陷阱。同时，数字技术的发展为教育提供了新的可能性。

### 目的

开发一种更有效的数学教育方法，通过基于专业数学家活动的学习模型，结合自我意识和结构化工具，解决传统教育方法的局限性，并利用数字技术增强人类学习过程。

### 方法

采用语言/行动方法，教师设计数学情境为学习者搭建脚手架，处理各种问题。强调学习者与教师之间的书面-口头对话（通常一对一），并由更有知识的高年级学生作为对话者辅助（每5-7名学生配备1名）。同时，利用交互式开发环境或AI等技术增强人类智慧。

### 主要发现

基于自我意识、类型、功能和结构化图表的学习模型能够克服传统练习和实践的弱点。通过专业数学家活动的学习方式，结合对话和技术增强，可以更有效地进行数学教育。学习者有生物和数字两部分，两者通过不同的方式参与学习过程。

### 结论

将数学/信息学教育建立在专业数学家活动基础上的新方法，结合对话、做中学和技术增强，能够提供比传统工业模式更有效的数学教育。这种方法强调学习过程中的互动和实际操作，而非简单的指令执行。

### 翻译

我们考虑通过行动研究、黑客精神、发现、探究和做中学来学习数学，这与19世纪的指令-执行工业模式形成对比。一种基于自我意识、类型、功能、结构化绘图和形式图表的学习模型解决了练习和实践的弱点以及大型语言模型统计预测的陷阱。换句话说，我们将数学/信息学教育建立在数学建模和设计程序的专业数学家的活动基础上。这一传统强调对话和做数学的作用。在语言/行动方法中，教师设计数学情境，为学习者搭建脚手架，处理已遇到或未知如何解决的问题，同时教师和教师/对话者监督这一过程。关键特征是学习者与教师之间的书面-口头对话。通常这是一对一的沟通。教师/对话者的角色，即更有知识的人，通常由高年级学生担任，每5-7名学生配备1名。受Doug Engelbart启发，我们提出通过交互式开发环境或AI等数字技术增强人类智慧的隐喻。每个人都有生物和数字部分。学习者的生物部分通过内心对话对工作做出反应；数字部分提出问题、解释代码并提出不一定可靠的想法。


### 论文摘要

We consider learning mathematics through action research, hacking, discovery, inquiry, learning-by-doing as opposed to the instruct and perform, industrial model of the 19th century. A learning model based on self-awareness, types, functions, structured drawing and formal diagrams addresses the weaknesses of drill and practice and the pitfalls of statistical prediction with Large Language Models.   In other words, we build mathematics/informatics education on the activity of a professional mathematician in mathematical modelling and designing programs. This tradition emphasises the role of dialogue and doing mathematics. In the Language/Action approach the teacher designs mathematising situations that scaffold previously encountered, or not-known-how-to-solve problems for the learner while teachers and teacher/interlocutors supervise the process.   A critical feature is the written-oral dialogue between the learner and the teacher. As a rule, this is 1 to 1 communication. The role of the teacher/interlocutor, a more knowledgeable other, is mostly performed by a more senior student, 1 per 5 to 7 pupils. After Doug Engelbart we propose the metaphor of human intellect augmented by digital technologies such as interactive development environments or AI. Every human has their bio and digital parts. The bio part of the learner reacts to their work through dialogue in the mind. The digital part poses questions, interprets code and proposes not necessarily sound ideas.

---

## 111. Robust inverse material design with physical guarantees using the Voigt-Reuss Net

**论文链接:** [http://arxiv.org/abs/2511.11388v1](http://arxiv.org/abs/2511.11388v1)

**作者:** Sanath Keshav, Felix Fritzen

**发布时间:** 2025-11-14

### GPT解析

### 总结

提出了一种用于正向和逆向力学均匀化的谱归一化替代方法，具有严格的物理保证，在多种实验中表现出高精度和强泛化能力。

### 背景

传统力学均匀化方法需要处理复杂的物理约束，而现有方法在保证物理一致性的同时难以实现高精度预测和逆向设计。

### 目的

开发一种能够学习无量纲、对称半正定表示的方法，其特征值在[0,1]区间内，并实现物理上可接受的张量预测和逆向设计。

### 方法

利用Voigt-Reuss界并通过Cholesky类算子分解差异；在3D线弹性上使用全连接Voigt-Reuss网络，训练超过75万个基于FFT的标签，结合236个各向同性不变描述符和三个对比参数；在2D平面应变上将谱归一化与可微渲染器和CNN结合。

### 主要发现

3D实验中恢复各向同性投影的保真度接近完美（R²≥0.998），张量级相对Frobenius误差中位数约1.7%，平均约3.4%；2D实验中所有分量R²>0.99，归一化损失小于百分之一，能准确跟踪渗透引起的特征值跳跃；批量一阶优化可在百分之几误差内匹配目标张量并返回多样化近最优设计。

### 结论

Voigt-Reuss网络成功统一了准确、物理上可接受的正向预测与大批量、约束一致的逆向设计，该方法适用于椭圆算子和耦合物理设置。

### 翻译

我们提出了一种具有严格物理保证的正向和逆向力学均匀化的谱归一化替代方法。利用Voigt-Reuss界，我们通过Cholesky类算子分解它们的差异，并学习一个特征值在[0,1]区间内的无量纲、对称半正定表示；逆映射返回在Löwner意义上位于边界之间的对称正定预测。在开放数据集上的3D线弹性随机双相微观结构中，使用236个各向同性不变描述符和三个对比参数训练的全连接Voigt-Reuss网络，在超过75万个基于FFT的标签上恢复各向同性投影的保真度接近完美（各向同性相关条目：R²≥0.998），而各向异性揭示的耦合无法从SO(3)不变输入中识别。跨分割的张量级相对Frobenius误差中位数约为1.7%，平均值约为3.4%。对于2D平面应变上的阈值三角微观结构，将谱归一化与可微渲染器和CNN结合，在所有分量上获得R²>0.99，归一化损失小于百分之一，准确跟踪渗透引起的特征值跳跃，并对分布外图像具有稳健泛化能力。将参数化微观结构视为设计变量，使用单个替代模型的批量一阶优化可在百分之几的误差内匹配目标张量，并返回多样化的近最优设计。总体而言，Voigt-Reuss网络将准确、物理上可接受的正向预测与大批量、约束一致的逆向设计统一起来，并且适用于椭圆算子和耦合物理设置。


### 论文摘要

We propose a spectrally normalized surrogate for forward and inverse mechanical homogenization with hard physical guarantees. Leveraging the Voigt-Reuss bounds, we factor their difference via a Cholesky-like operator and learn a dimensionless, symmetric positive semi-definite representation with eigenvalues in $[0,1]$; the inverse map returns symmetric positive-definite predictions that lie between the bounds in the Löwner sense. In 3D linear elasticity on an open dataset of stochastic biphasic microstructures, a fully connected Voigt-Reuss net trained on $>\!7.5\times 10^{5}$ FFT-based labels with 236 isotropy-invariant descriptors and three contrast parameters recovers the isotropic projection with near-perfect fidelity (isotropy-related entries: $R^2 \ge 0.998$), while anisotropy-revealing couplings are unidentifiable from $SO(3)$-invariant inputs. Tensor-level relative Frobenius errors have median $\approx 1.7\%$ and mean $\approx 3.4\%$ across splits. For 2D plane strain on thresholded trigonometric microstructures, coupling spectral normalization with a differentiable renderer and a CNN yields $R^2>0.99$ on all components, subpercent normalized losses, accurate tracking of percolation-induced eigenvalue jumps, and robust generalization to out-of-distribution images. Treating the parametric microstructure as design variables, batched first-order optimization with a single surrogate matches target tensors within a few percent and returns diverse near-optimal designs. Overall, the Voigt-Reuss net unifies accurate, physically admissible forward prediction with large-batch, constraint-consistent inverse design, and is generic to elliptic operators and coupled-physics settings.

---

## 112. Coordinative Learning with Ordinal and Relational Priors for Volumetric Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2511.11276v1](http://arxiv.org/abs/2511.11276v1)

**作者:** Haoyi Wang

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了协调序数关系解剖学习（CORAL）方法，用于解决三维医学图像分割中的挑战。CORAL通过对比排序目标和序数目标分别捕获连续的解剖相似性和全局方向一致性，从而产生解剖学上有意义的表示，在有限标注设置下实现了最先进的分割性能。

### 背景

三维医学图像分割面临固有解剖结构和有限标注可用性的挑战。现有方法通过对比切片间空间关系取得进展，但依赖硬二进制阈值定义正负样本，丢弃了解剖相似性的连续信息，且忽略了解剖进展的全局方向一致性，导致特征空间扭曲。

### 目的

开发一种能够同时捕获三维医学图像中局部和全局结构的方法，解决现有方法在利用解剖相似性信息和保持全局方向一致性方面的局限性，从而提高在有限标注条件下的分割性能。

### 方法

提出协调序数关系解剖学习（CORAL）框架，包含两个主要组件：1）对比排序目标，利用连续的解剖相似性，确保切片间的关系特征距离与其解剖位置差异成正比；2）序数目标，强制执行全局方向一致性，使学习到的特征分布与患者间的典型解剖进展保持一致。通过学习这些切片间关系产生解剖学上有意义的表示。

### 主要发现

CORAL通过协调学习局部和全局解剖结构，在有限标注设置下的基准数据集上实现了最先进的分割性能。学习到的表示具有有意义的解剖结构，能够有效捕捉患者间共享的典型解剖流形。

### 结论

CORAL框架成功解决了三维医学图像分割中的关键挑战，通过同时利用连续的解剖相似信息和保持全局方向一致性，显著提高了有限标注条件下的分割性能。该方法不仅实现了技术上的突破，还为医学图像分析领域提供了新的研究方向。

### 翻译

三维医学图像分割由于固有的解剖结构和有限的标注可用性而呈现独特的挑战。尽管最近的方法通过对比切片间的空间关系显示出前景，但它们依赖于硬二进制阈值来定义正负样本，从而丢弃了关于解剖相似性的宝贵连续信息。此外，这些方法忽略了解剖进展的全局方向一致性，导致特征空间扭曲，无法捕捉患者间共享的典型解剖流形。为解决这些局限性，我们提出了协调序数关系解剖学习（CORAL）来捕获三维图像中的局部和全局结构。首先，CORAL采用对比排序目标来利用连续的解剖相似性，确保切片间的关系特征距离与其解剖位置差异成正比。此外，CORAL纳入了序数目标来强制执行全局方向一致性，使学习到的特征分布与患者间的典型解剖进展保持一致。学习这些切片间的关系产生了解剖学上有意义的表示，有利于下游分割任务。通过这种协调学习框架，CORAL在有限的标注设置下在基准数据集上实现了最先进的性能，同时学习到了具有有意义解剖结构的表示。代码可在https://github.com/haoyiwang25/CORAL获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文解决的是三维医学图像分割中的挑战，现有方法依赖硬性二值阈值定义样本关系，丢弃了有价值的解剖相似性连续信息，同时忽略了解剖进展的全局方向一致性。这个问题很重要，因为医学图像分割是诊断和治疗规划的核心工具，但获取标注数据困难，且三维医学图像具有固有的解剖结构，相邻切片相似而远距离切片不同，正确捕捉这些关系能提高分割准确性，特别是在标注数据有限的情况下。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法如PCL和SAL的局限性，指出它们依赖硬性阈值和忽略全局方向一致性的问题。设计方法时借鉴了对比学习的基本思想，特别是PCL利用体积扫描空间组织的方法，以及Rank-N-Contrast的排序框架。作者的创新在于将排序机制应用于基于解剖位置差异的切片排序，并引入序数目标函数强制全局方向一致性，最终将这些思想组合成CORAL框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合关系解剖学习和序数解剖学习，同时捕捉局部相似关系和全局方向一致性。关系学习确保切片间的特征距离与解剖位置差异成正比，序数学习强制跨患者的一致解剖轨迹。整体采用两阶段流程：首先在未标记数据上预训练编码器，同时优化关系损失和序数损失；然后固定编码器，仅微调解码器进行分割任务。预训练中通过归一化位置编码使不同体积的切片可比较，学习解剖上有意义的表示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出CORAL框架结合关系学习和序数学习；用连续排序机制替代硬性二值阈值，消除手动调参需求；引入全局方向一致性约束，捕捉典型解剖进展。相比PCL，CORAL避免了硬性阈值并增加了全局一致性考虑；相比SAL，它超越了基于阈值的设计范式；相比一般对比学习，它专门针对医学体积图像的解剖结构优化。CORAL在有限标注条件下实现了最先进性能，同时学习到具有有意义解剖结构的表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CORAL通过结合连续关系学习和全局序数约束，在有限标注条件下实现了三维医学图像分割的最先进性能，同时学习到了具有有意义解剖结构的表示。'}


### 论文摘要

Volumetric medical image segmentation presents unique challenges due to the inherent anatomical structure and limited availability of annotations. While recent methods have shown promise by contrasting spatial relationships between slices, they rely on hard binary thresholds to define positive and negative samples, thereby discarding valuable continuous information about anatomical similarity. Moreover, these methods overlook the global directional consistency of anatomical progression, resulting in distorted feature spaces that fail to capture the canonical anatomical manifold shared across patients. To address these limitations, we propose Coordinative Ordinal-Relational Anatomical Learning (CORAL) to capture both local and global structure in volumetric images. First, CORAL employs a contrastive ranking objective to leverage continuous anatomical similarity, ensuring relational feature distances between slices are proportional to their anatomical position differences. In addition, CORAL incorporates an ordinal objective to enforce global directional consistency, aligning the learned feature distribution with the canonical anatomical progression across patients. Learning these inter-slice relationships produces anatomically informed representations that benefit the downstream segmentation task. Through this coordinative learning framework, CORAL achieves state-of-the-art performance on benchmark datasets under limited-annotation settings while learning representations with meaningful anatomical structure. Code is available at https://github.com/haoyiwang25/CORAL.

---

## 113. LANE: Lexical Adversarial Negative Examples for Word Sense Disambiguation

**论文链接:** [http://arxiv.org/abs/2511.11234v1](http://arxiv.org/abs/2511.11234v1)

**作者:** Jader Martins Camboim de Sá, Jooyoung Lee, Cédric Pruski, Marcos Da Silveira

**发布时间:** 2025-11-14

### GPT解析

### 总结

提出了一种名为LANE的新型对抗训练策略，通过选择性标记替代词生成挑战性负样本，使神经语言模型能够更好地捕捉局部语义细节，提高词义解析能力。

### 背景

神经语言模型在细粒度词义解析方面面临挑战，往往过度拟合全局句子表示，无法捕捉局部语义细节。

### 目的

通过将模型的学习重点转移到目标词上，解决神经语言模型无法捕捉局部语义细节的问题。

### 方法

提出名为LANE的对抗训练策略，通过在训练集中选择性标记替代词生成挑战性负训练样本，强制模型在具有不同标记词的相同句子间创建更大可分离性。

### 主要发现

在词汇语义变化检测和词义消歧基准上的实验表明，该方法能产生更具区分性的词表示，性能优于标准对比学习基线，且能更好地捕捉细微意义差异。

### 结论

LANE方法与模型无关，可集成到现有表示学习框架中，有效提升了神经语言模型的词义解析能力。

### 翻译

细粒度词义解析仍然是神经语言模型的一个关键挑战，因为它们常常过度拟合全局句子表示，无法捕捉局部语义细节。我们提出了一种名为LANE的新型对抗训练策略，通过将模型的学习重点故意转移到目标词上，来解决这一局限性。该方法通过在训练集中选择性标记替代词来生成具有挑战性的负训练样本。目的是强制模型在具有不同标记词的相同句子之间创建更大的可分离性。在词汇语义变化检测和词义消歧基准上的实验结果表明，我们的方法能产生更具区分性的词表示，提高了性能，超过了标准对比学习基线。我们进一步提供了定性分析，表明所提出的负样本能生成更好的表示，即使在具有挑战性的环境中也能捕捉细微的意义差异。我们的方法与模型无关，可以集成到现有的表示学习框架中。


### 论文摘要

Fine-grained word meaning resolution remains a critical challenge for neural language models (NLMs) as they often overfit to global sentence representations, failing to capture local semantic details. We propose a novel adversarial training strategy, called LANE, to address this limitation by deliberately shifting the model's learning focus to the target word. This method generates challenging negative training examples through the selective marking of alternate words in the training set. The goal is to force the model to create a greater separability between same sentences with different marked words. Experimental results on lexical semantic change detection and word sense disambiguation benchmarks demonstrate that our approach yields more discriminative word representations, improving performance over standard contrastive learning baselines. We further provide qualitative analyses showing that the proposed negatives lead to representations that better capture subtle meaning differences even in challenging environments. Our method is model-agnostic and can be integrated into existing representation learning frameworks.

---

## 114. Dynamic Deep Graph Learning for Incomplete Multi-View Clustering with Masked Graph Reconstruction Loss

**论文链接:** [http://arxiv.org/abs/2511.11181v1](http://arxiv.org/abs/2511.11181v1)

**作者:** Zhenghao Zhang, Jun Xie, Xingchen Chen, Tao Yu, Hongzhu Yi, Kaixin Xu, Yuanxiang Wang, Tianyu Zong, Xinming Wang, Jiahuan Chen, Guoqing Chao, Feng Chen, Zhepeng Wang, Jungang Xu

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一种名为DGIMVCM的新方法，用于解决不完整多视图聚类问题。该方法通过动态深度图学习和掩码图重建损失，有效克服了现有方法在构建静态图和优化过程中面临的挑战。

### 背景

现实世界多视图数据普遍存在，使不完整多视图聚类成为重要研究领域。图神经网络的快速发展已使其成为多视图聚类的主流方法之一。

### 目的

解决现有基于GNN的不完整多视图聚类方法中依赖KNN算法构建静态图引入噪声，以及使用MSE损失导致优化过程中产生大量梯度噪声的问题。

### 方法

提出DGIMVCM方法，包括：构建对缺失具有鲁棒性的全局图；设计图卷积嵌入层提取主要特征和动态视图特定图结构；通过图结构对比学习识别视图一致性；引入图自注意力编码器提取高级表示；使用掩码图重建损失减轻梯度噪声；构建聚类模块并通过伪标签自监督训练机制优化。

### 主要发现

在多个数据集上的大量实验验证了DGIMVCM的有效性和优越性。

### 结论

DGIMVCM是一种有效解决不完整多视图聚类问题的新方法，通过动态图学习和掩码重建损失克服了现有方法的局限性。

### 翻译

现实世界多视图数据的普遍存在使不完整多视图聚类成为一项重要研究。图神经网络的快速发展已使其成为多视图聚类的主流方法之一。尽管基于GNN的不完整多视图聚类取得了显著进展，但仍存在一些挑战：(1)大多数方法依赖K近邻算法从原始数据构建静态图，这引入了噪声并降低了图拓扑的鲁棒性。(2)现有方法通常直接使用重建图与稀疏邻接图之间的均方误差损失作为图重建损失，导致优化过程中产生大量梯度噪声。为解决这些问题，我们提出了一种新颖的具有掩码图重建损失的不完整多视图聚类的动态深度图学习方法(DGIMVCM)。首先，我们从原始数据构建一个对缺失具有鲁棒性的全局图。然后设计一个图卷积嵌入层来提取主要特征和精细的动态视图特定图结构，利用全局图进行缺失视图的插补。这一过程通过图结构对比学习得到补充，该学习能够识别视图特定图结构之间的一致性。其次，引入图自注意力编码器基于插补的主要特征和视图特定图提取高级表示，并使用掩码图重建损失进行优化，以减轻优化过程中的梯度噪声。最后，构建一个聚类模块并通过伪标签自监督训练机制进行优化。在多个数据集上的大量实验验证了DGIMVCM的有效性和优越性。


### 论文摘要

The prevalence of real-world multi-view data makes incomplete multi-view clustering (IMVC) a crucial research. The rapid development of Graph Neural Networks (GNNs) has established them as one of the mainstream approaches for multi-view clustering. Despite significant progress in GNNs-based IMVC, some challenges remain: (1) Most methods rely on the K-Nearest Neighbors (KNN) algorithm to construct static graphs from raw data, which introduces noise and diminishes the robustness of the graph topology. (2) Existing methods typically utilize the Mean Squared Error (MSE) loss between the reconstructed graph and the sparse adjacency graph directly as the graph reconstruction loss, leading to substantial gradient noise during optimization. To address these issues, we propose a novel \textbf{D}ynamic Deep \textbf{G}raph Learning for \textbf{I}ncomplete \textbf{M}ulti-\textbf{V}iew \textbf{C}lustering with \textbf{M}asked Graph Reconstruction Loss (DGIMVCM). Firstly, we construct a missing-robust global graph from the raw data. A graph convolutional embedding layer is then designed to extract primary features and refined dynamic view-specific graph structures, leveraging the global graph for imputation of missing views. This process is complemented by graph structure contrastive learning, which identifies consistency among view-specific graph structures. Secondly, a graph self-attention encoder is introduced to extract high-level representations based on the imputed primary features and view-specific graphs, and is optimized with a masked graph reconstruction loss to mitigate gradient noise during optimization. Finally, a clustering module is constructed and optimized through a pseudo-label self-supervised training mechanism. Extensive experiments on multiple datasets validate the effectiveness and superiority of DGIMVCM.

---

## 115. PRSM: A Measure to Evaluate CLIP's Robustness Against Paraphrases

**论文链接:** [http://arxiv.org/abs/2511.11141v1](http://arxiv.org/abs/2511.11141v1)

**作者:** Udo Schlegel, Franziska Weeber, Jian Lan, Thomas Seidl

**发布时间:** 2025-11-14

**备注:** 8 pages, accpeted as short paper at MMM 2026

### GPT解析

### 总结

本研究探讨了对比语言-图像预训练模型（CLIP）对释义变化的鲁棒性，提出了一种新的度量方法来评估CLIP对释义查询的敏感性。

### 背景

CLIP是一个广泛使用的多模态模型，通过大规模训练对齐文本和图像表示，在零样本和少样本任务上表现良好，但其对语言变化的鲁棒性，特别是释义的鲁棒性，尚未得到充分探索。

### 目的

引入释义排序稳定性指标（PRSM），量化CLIP对释义查询的敏感性，并评估CLIP在释义变化下的稳定性，特别是与性别相关的差异。

### 方法

提出释义排序稳定性指标（PRSM）作为新的度量方法，使用社会反事实数据集（Social Counterfactuals dataset）作为基准来评估CLIP的释义稳定性，并检查释义鲁棒性与性别之间的相互作用。

### 主要发现

CLIP的鲁棒性因释义策略而异，在男性相关查询和女性相关查询之间观察到微妙但一致的差异。

### 结论

释义鲁棒性对于可靠部署至关重要，特别是在社会敏感的背景下，不一致的表示可能会放大人口统计偏见，这对多模态系统的公平和公平部署有影响。

### 翻译

对比语言-图像预训练（CLIP）是一种广泛使用的多模态模型，通过大规模训练对齐文本和图像表示。虽然它在零样本和少样本任务上表现强劲，但它对语言变化的鲁棒性，特别是释义，仍然探索不足。释义鲁棒性对于可靠部署至关重要，特别是在社会敏感的背景下，不一致的表示可能会放大人口统计偏见。在本文中，我们引入了释义排序稳定性指标（PRSM），这是一种量化CLIP对释义查询敏感性的新度量。使用社会反事实数据集（一个旨在揭示社会和人口统计偏见的基准），我们经验性地评估了CLIP在释义变化下的稳定性，检查了释义鲁棒性与性别之间的相互作用，并讨论了多模态系统公平和公平部署的影响。我们的分析表明，鲁棒性因释义策略而异，在男性相关查询和女性相关查询之间观察到微妙但一致的差异。


### 论文摘要

Contrastive Language-Image Pre-training (CLIP) is a widely used multimodal model that aligns text and image representations through large-scale training. While it performs strongly on zero-shot and few-shot tasks, its robustness to linguistic variation, particularly paraphrasing, remains underexplored. Paraphrase robustness is essential for reliable deployment, especially in socially sensitive contexts where inconsistent representations can amplify demographic biases. In this paper, we introduce the Paraphrase Ranking Stability Metric (PRSM), a novel measure for quantifying CLIP's sensitivity to paraphrased queries. Using the Social Counterfactuals dataset, a benchmark designed to reveal social and demographic biases, we empirically assess CLIP's stability under paraphrastic variation, examine the interaction between paraphrase robustness and gender, and discuss implications for fairness and equitable deployment of multimodal systems. Our analysis reveals that robustness varies across paraphrasing strategies, with subtle yet consistent differences observed between male- and female-associated queries.

---

## 116. Hindsight Distillation Reasoning with Knowledge Encouragement Preference for Knowledge-based Visual Question Answering

**论文链接:** [http://arxiv.org/abs/2511.11132v1](http://arxiv.org/abs/2511.11132v1)

**作者:** Yu Zhao, Ying Zhang, Xuhui Sui, Baohang Zhou, Li Shen, Dacheng Tao

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为Hindsight Distilled Reasoning (HinD)的框架，配合Knowledge Encouragement Preference Optimization (KEPO)，用于激发和利用多模态大语言模型内部的知识推理能力，解决了基于知识的视觉问答中推理过程不明确的问题。

### 背景

现有的KBVQA方法要么通过上下文学习利用MLLMs中的隐式知识，要么通过检索增强生成利用显式知识，但它们的推理过程仍然是隐式的，没有明确的多步推理轨迹。

### 目的

设计一个框架，使MLLMs能够产生明确的多步推理轨迹，解决KBVQA中推理过程不明确的问题。

### 方法

1) 构建Hindsight-Zero训练数据，通过提示冻结的7B大小MLLM完成问题和答案间的推理过程；2) 将Hindsight-Zero自我蒸馏到CoT Generator和Knowledge Generator中；3) 使用KEPO优化Knowledge Generator，解决知识正确性和置信度不匹配问题；4) 利用生成的CoT和采样知识进行答案预测。

### 主要发现

在OK-VQA和A-OKVQA上的实验表明，仅从7B大小的MLLM中激发推理能力的HinD无需商业模型API或外部知识就能实现优越性能。

### 结论

HinD框架成功地解决了KBVQA中推理过程不明确的问题，通过自我蒸馏和知识鼓励偏好优化，有效地利用了MLLMs的内部知识推理能力。

### 翻译

基于知识的视觉问答需要整合超越跨模态理解的外部知识。现有的KBVQA方法要么通过上下文学习利用多模态大语言模型中的隐式知识，要么通过检索增强生成利用显式知识。然而，它们的推理过程仍然是隐式的，没有来自MLLMs的明确多步轨迹。为解决这一差距，我们提出了一个名为Hindsight Distilled Reasoning (HinD)的框架，配合Knowledge Encouragement Preference Optimization (KEPO)，旨在激发和利用MLLMs内部的知识推理能力。首先，为解决推理监督问题，我们通过提示一个冻结的7B大小的MLLM来完成问题和其真实答案之间的推理过程，构建Hindsight-Zero训练数据，强调MLLM的后见之明智慧。然后我们将Hindsight-Zero自我蒸馏到Chain-of-Thought (CoT) Generator和Knowledge Generator中，使它们能够生成连续步骤和离散事实。其次，为解决知识正确性和置信度之间的不匹配问题，我们使用KEPO优化Knowledge Generator，倾向于选择低置信度但有帮助的知识，而非高置信度但无帮助的知识。生成的CoT和采样知识随后用于答案预测。在OK-VQA和A-OKVQA上的实验验证了HinD的有效性，显示仅从7B大小的MLLM中激发推理能力的HinD无需商业模型API或外部知识就能实现优越性能。


### 论文摘要

Knowledge-based Visual Question Answering (KBVQA) necessitates external knowledge incorporation beyond cross-modal understanding. Existing KBVQA methods either utilize implicit knowledge in multimodal large language models (MLLMs) via in-context learning or explicit knowledge via retrieval augmented generation. However, their reasoning processes remain implicit, without explicit multi-step trajectories from MLLMs. To address this gap, we provide a Hindsight Distilled Reasoning (HinD) framework with Knowledge Encouragement Preference Optimization (KEPO), designed to elicit and harness internal knowledge reasoning ability in MLLMs. First, to tackle the reasoning supervision problem, we propose to emphasize the hindsight wisdom of MLLM by prompting a frozen 7B-size MLLM to complete the reasoning process between the question and its ground truth answer, constructing Hindsight-Zero training data. Then we self-distill Hindsight-Zero into Chain-of-Thought (CoT) Generator and Knowledge Generator, enabling the generation of sequential steps and discrete facts. Secondly, to tackle the misalignment between knowledge correctness and confidence, we optimize the Knowledge Generator with KEPO, preferring under-confident but helpful knowledge over the over-confident but unhelpful one. The generated CoT and sampled knowledge are then exploited for answer prediction. Experiments on OK-VQA and A-OKVQA validate the effectiveness of HinD, showing that HinD with elicited reasoning from 7B-size MLLM achieves superior performance without commercial model APIs or outside knowledge.

---

## 117. Detection of Bark Beetle Attacks using Hyperspectral PRISMA Data and Few-Shot Learning

**论文链接:** [http://arxiv.org/abs/2511.11096v1](http://arxiv.org/abs/2511.11096v1)

**作者:** Mattia Ferrari, Giancarlo Papitto, Giorgio Deligios, Lorenzo Bruzzone

**发布时间:** 2025-11-14

**备注:** 5 pages, 3 figures, accepted at IGARSS conference 3-8 August 2025 Brisbane, Australia

### GPT解析

### 总结

本文提出了一种基于对比学习的小样本学习方法，利用卫星PRISMA高光谱数据检测树皮甲虫侵袭，并在多洛米蒂地区实验中证明该方法优于传统方法。

### 背景

树皮甲虫侵袭是对保持针叶林健康的一个严重挑战

### 目的

提出一种基于对比学习的小样本学习方法，利用卫星PRISMA高光谱数据检测树皮甲虫侵袭

### 方法

基于对比学习框架预训练一维CNN编码器提取高光谱数据的鲁棒特征，将这些特征输入到每个类别对应的支持向量回归估计器中，在少量标记样本上训练，以估计每个像素中健康、受树皮甲虫侵袭和死亡树木的比例

### 主要发现

在多洛米蒂地区的研究实验表明，该方法优于使用原始PRISMA光谱波段和Sentinel-2数据

### 结论

PRISMA高光谱数据与小样本学习相结合为森林健康监测提供了显著优势

### 翻译

树皮甲虫侵袭是对保持针叶林健康的一个严重挑战。本文提出了一种基于对比学习的小样本学习方法，利用卫星PRISMA高光谱数据检测树皮甲虫侵袭。该方法基于对比学习框架预训练一维CNN编码器，能够从高光谱数据中提取鲁棒的特征表示。这些提取的特征随后被用作输入，输入到支持向量回归估计器中，每个类别一个，在少量标记样本上训练，以估计每个像素中健康、受树皮甲虫侵袭和死亡树木的比例。在多洛米蒂地区的研究实验表明，我们的方法优于使用原始PRISMA光谱波段和Sentinel-2数据。结果表明，PRISMA高光谱数据与小样本学习相结合为森林健康监测提供了显著优势。


### 论文摘要

Bark beetle infestations represent a serious challenge for maintaining the health of coniferous forests. This paper proposes a few-shot learning approach leveraging contrastive learning to detect bark beetle infestations using satellite PRISMA hyperspectral data. The methodology is based on a contrastive learning framework to pre-train a one-dimensional CNN encoder, enabling the extraction of robust feature representations from hyperspectral data. These extracted features are subsequently utilized as input to support vector regression estimators, one for each class, trained on few labeled samples to estimate the proportions of healthy, attacked by bark beetle, and dead trees for each pixel. Experiments on the area of study in the Dolomites show that our method outperforms the use of original PRISMA spectral bands and of Sentinel-2 data. The results indicate that PRISMA hyperspectral data combined with few-shot learning offers significant advantages for forest health monitoring.

---

## 118. Machine-Learning Based Detection of Coronary Artery Calcification Using Synthetic Chest X-Rays

**论文链接:** [http://arxiv.org/abs/2511.11093v1](http://arxiv.org/abs/2511.11093v1)

**作者:** Dylan Saeed, Ramtin Gharleghi, Susann Bier, Sonit Singh

**发布时间:** 2025-11-14

**备注:** 10 pages, 5 figures. Under review for MIDL 2026

### GPT解析

### 总结

本研究首次系统评估了数字重建放射照片（DRRs）作为冠状动脉钙化（CAC）检测替代训练领域的可行性，通过生成合成DRRs并评估多种训练策略，发现轻量级CNN结合超分辨率和对比度增强效果最佳，实现了0.754的平均AUC值。

### 背景

冠状动脉钙化是心血管事件的强预测指标，CT扫描是临床金标准但成本高昂不适合大规模筛查，而胸部X光片便宜却缺乏可靠真值标签限制了深度学习发展。

### 目的

评估DRRs作为CAC检测替代训练领域的有效性，建立CAC检测的可扩展标签丰富基础，并为未来向真实CXR的迁移学习和领域适应奠定基础。

### 方法

使用COCA数据集的667个CT扫描生成合成DRRs，评估模型能力、超分辨率保真度增强、预处理和训练策略，比较轻量级CNN与大型预训练网络性能，测试超分辨率与对比度增强结合效果，以及课程学习在弱监督下的训练稳定性。

### 主要发现

轻量级CNN从头训练优于大型预训练网络；超分辨率与对比度增强结合带来显著性能提升；课程学习在弱监督下稳定训练过程；最佳配置实现0.754平均AUC，达到或超过先前基于CXR的研究结果。

### 结论

DRRs可作为CAC检测的可扩展、标签丰富的基础，为未来向真实CXR的迁移学习和领域适应研究奠定基础。

### 翻译

冠状动脉钙化（CAC）是心血管事件的强预测指标，基于CT的Agatston评分被广泛认为是临床金标准。然而，CT对于大规模筛查来说成本高昂且不切实际，而胸部X光片（CXRs）虽然便宜，但缺乏可靠的真值标签，限制了深度学习的发展。数字重建放射照片（DRRs）通过将CT体积投影成类似CXR的图像同时继承精确标签，提供了可扩展的替代方案。在这项工作中，我们首次提供了对DRRs作为CAC检测替代训练领域的系统性评估。使用来自COCA数据集的667个CT扫描，我们生成合成DRRs并评估了模型能力、超分辨率保真度增强、预处理和训练策略。从头开始训练的轻量级CNN优于大型预训练网络；超分辨率与对比度增强相结合带来显著提升；课程学习在弱监督下稳定了训练。我们的最佳配置实现了平均AUC为0.754，与或超过了先前基于CXR的研究结果。这些结果确立了DRRs作为CAC检测的可扩展、标签丰富的基础，同时为未来向真实CXR的迁移学习和领域适应奠定了基础。


### 论文摘要

Coronary artery calcification (CAC) is a strong predictor of cardiovascular events, with CT-based Agatston scoring widely regarded as the clinical gold standard. However, CT is costly and impractical for large-scale screening, while chest X-rays (CXRs) are inexpensive but lack reliable ground truth labels, constraining deep learning development. Digitally reconstructed radiographs (DRRs) offer a scalable alternative by projecting CT volumes into CXR-like images while inheriting precise labels. In this work, we provide the first systematic evaluation of DRRs as a surrogate training domain for CAC detection. Using 667 CT scans from the COCA dataset, we generate synthetic DRRs and assess model capacity, super-resolution fidelity enhancement, preprocessing, and training strategies. Lightweight CNNs trained from scratch outperform large pretrained networks; pairing super-resolution with contrast enhancement yields significant gains; and curriculum learning stabilises training under weak supervision. Our best configuration achieves a mean AUC of 0.754, comparable to or exceeding prior CXR-based studies. These results establish DRRs as a scalable, label-rich foundation for CAC detection, while laying the foundation for future transfer learning and domain adaptation to real CXRs.

---

## 119. From Retinal Pixels to Patients: Evolution of Deep Learning Research in Diabetic Retinopathy Screening

**论文链接:** [http://arxiv.org/abs/2511.11065v1](http://arxiv.org/abs/2511.11065v1)

**作者:** Muskaan Chopra, Lorenz Sparrenberg, Armin Berger, Sarthak Khanna, Jan H. Terheyden, Rafet Sifa

**发布时间:** 2025-11-14

**备注:** Accepted in IEEE BigData 2025

### GPT解析

### 总结

这是一篇关于2016-2025年糖尿病视网膜病变(DR)深度学习研究的系统性综述，涵盖了50多项研究和20多个数据集的结果，评估了方法学进展和临床应用挑战。

### 背景

糖尿病视网膜病变(DR)是可预防性失明的主要原因，早期检测对减少全球视力损失至关重要。

### 目的

提供DR研究的首次系统性综合，批判性检查方法学进展，评估协议和可重复性挑战，并制定可重复、隐私保护且可在临床上部署的DR人工智能的实用议程。

### 方法

系统性综述2016-2025年DR研究，整理50多项研究和20多个数据集的结果，批判性检查自监督和半监督学习、领域泛化、联邦训练和混合神经符号模型等方法学进展，以及评估协议、报告标准和可重复性挑战。

### 主要发现

过去十年深度学习彻底改变了DR筛查，从早期在私有数据集上训练的卷积神经网络发展到解决类别不平衡、标签稀缺、领域偏移和可解释性问题的先进流程；多中心验证和临床信任仍存在开放差距；许多调查的创新可广泛扩展到大规模医学影像。

### 结论

通过将技术进步与翻译障碍联系起来，这项工作概述了可重复、隐私保护且可在临床上部署的DR人工智能的实用议程，其创新方法也适用于其他医学影像领域。

### 翻译

糖尿病视网膜病变(DR)仍然是可预防性失明的主要原因，早期检测对减少全球视力损失至关重要。过去十年，深度学习彻底改变了DR筛查，从早期在私有数据集上训练的卷积神经网络发展到解决类别不平衡、标签稀缺、领域偏移和可解释性问题的先进流程。本综述提供了2016-2025年DR研究的首次系统性综合，整理了50多项研究和20多个数据集的结果。我们批判性检查了方法学进展，包括自监督和半监督学习、领域泛化、联邦训练和混合神经符号模型，以及评估协议、报告标准和可重复性挑战。基准表格展示了跨数据集的性能背景，讨论强调了多中心验证和临床信任方面的开放差距。通过将技术进步与翻译障碍联系起来，这项工作概述了可重复、隐私保护且可在临床上部署的DR人工智能的实用议程。除了DR，许多调查的创新也广泛扩展到大规模医学影像。


### 论文摘要

Diabetic Retinopathy (DR) remains a leading cause of preventable blindness, with early detection critical for reducing vision loss worldwide. Over the past decade, deep learning has transformed DR screening, progressing from early convolutional neural networks trained on private datasets to advanced pipelines addressing class imbalance, label scarcity, domain shift, and interpretability. This survey provides the first systematic synthesis of DR research spanning 2016-2025, consolidating results from 50+ studies and over 20 datasets. We critically examine methodological advances, including self- and semi-supervised learning, domain generalization, federated training, and hybrid neuro-symbolic models, alongside evaluation protocols, reporting standards, and reproducibility challenges. Benchmark tables contextualize performance across datasets, while discussion highlights open gaps in multi-center validation and clinical trust. By linking technical progress with translational barriers, this work outlines a practical agenda for reproducible, privacy-preserving, and clinically deployable DR AI. Beyond DR, many of the surveyed innovations extend broadly to medical imaging at scale.

---

## 120. SemanticNN: Compressive and Error-Resilient Semantic Offloading for Extremely Weak Devices

**论文链接:** [http://arxiv.org/abs/2511.11038v1](http://arxiv.org/abs/2511.11038v1)

**作者:** Jiaming Huang, Yi Gao, Fuchang Pan, Renjie Li, Wei Dong

**发布时间:** 2025-11-14

### GPT解析

### 总结

这篇论文提出了SemanticNN，一种语义编解码器，用于在资源受限的物联网设备上实现容错的设备-边缘协作推理卸载。该方法追求语义级别的正确性而非比特级别的正确性，能够在动态信道条件下实现高效且鲁棒的协作推理。

### 背景

随着物联网的快速增长，在极弱嵌入式设备上集成人工智能引起了广泛关注，可以改善实时性能并增强数据隐私。然而，这类设备的资源限制和不可靠的网络条件需要容错设备-边缘协作系统。传统方法专注于比特级传输正确性，在动态信道条件下可能效率低下。

### 目的

提出一种能够容忍比特级错误、追求语义级正确性的语义编解码器，在严格的计算和通信约束下实现压缩且鲁棒的协作推理卸载。

### 方法

1. 提出了SemanticNN，包含一个比特错误率感知的解码器和一个基于软量化的编码器；2. 引入了特征增强学习，一种提高卸载效率的新型训练策略；3. 提出了基于XAI的不对称补偿，解决编码器-解码器能力不匹配问题，提高解码语义保真度。

### 主要发现

在STM32上使用三种模型和六个数据集进行的图像分类和目标检测任务实验表明，在不同传输错误率下，SemanticNN将特征传输量减少了56.82-344.83倍，同时保持了卓越的推理准确性。

### 结论

SemanticNN能够有效解决资源受限设备和不可靠网络条件下的协作推理问题，通过追求语义级正确性而非比特级正确性，显著减少了特征传输量同时保持高推理准确率。

### 翻译

随着物联网(IoT)的快速增长，在极弱嵌入式设备上集成人工智能(AI)已引起广泛关注，能够提高实时性能并增强数据隐私。然而，这类设备的资源限制和不可靠的网络条件需要容错的设备-边缘协作系统。传统方法专注于比特级传输正确性，在动态信道条件下可能效率低下。相比之下，我们提出了SemanticNN，一种语义编解码器，它容忍比特级错误以追求语义级正确性，能够在严格的计算和通信约束下实现压缩且鲁棒的协作推理卸载。它包含一个比特错误率(BER)感知的解码器，能够适应动态信道条件，以及一个基于软量化(SQ)的编码器，用于学习紧凑的表示。基于此架构，我们引入了特征增强学习(Feature-augmentation Learning)，一种提高卸载效率的新型训练策略。为了解决由不对称资源引起的编码器-解码器能力不匹配问题，我们提出了基于XAI的不对称补偿(XAI-based Asymmetry Compensation)来增强解码语义保真度。我们在STM32上使用三种模型和六个数据集进行了广泛的实验，涵盖图像分类和目标检测任务。实验结果表明，在不同传输错误率下，SemanticNN显著减少了56.82-344.83倍的特征传输量，同时保持了卓越的推理准确性。


### 论文摘要

With the rapid growth of the Internet of Things (IoT), integrating artificial intelligence (AI) on extremely weak embedded devices has garnered significant attention, enabling improved real-time performance and enhanced data privacy. However, the resource limitations of such devices and unreliable network conditions necessitate error-resilient device-edge collaboration systems. Traditional approaches focus on bit-level transmission correctness, which can be inefficient under dynamic channel conditions. In contrast, we propose SemanticNN, a semantic codec that tolerates bit-level errors in pursuit of semantic-level correctness, enabling compressive and resilient collaborative inference offloading under strict computational and communication constraints. It incorporates a Bit Error Rate (BER)-aware decoder that adapts to dynamic channel conditions and a Soft Quantization (SQ)-based encoder to learn compact representations. Building on this architecture, we introduce Feature-augmentation Learning, a novel training strategy that enhances offloading efficiency. To address encoder-decoder capability mismatches from asymmetric resources, we propose XAI-based Asymmetry Compensation to enhance decoding semantic fidelity. We conduct extensive experiments on STM32 using three models and six datasets across image classification and object detection tasks. Experimental results demonstrate that, under varying transmission error rates, SemanticNN significantly reduces feature transmission volume by 56.82-344.83x while maintaining superior inference accuracy.

---

## 121. PROMISE: Prompt-Attentive Hierarchical Contrastive Learning for Robust Cross-Modal Representation with Missing Modalities

**论文链接:** [http://arxiv.org/abs/2511.10997v1](http://arxiv.org/abs/2511.10997v1)

**作者:** Jiajun Chen, Sai Cheng, Yutao Yuan, Yirui Zhang, Haitao Yuan, Peng Peng, Yi Zhong

**发布时间:** 2025-11-14

**备注:** Accepted by AAAI'2026 Main Conference

### GPT解析

### 总结

本文提出了一种名为PROMISE的新型多模态框架，用于在模态缺失条件下进行稳健的跨模态表示学习。该框架结合了多模态提示学习和分层对比学习，并通过特别设计的提示注意力机制，有效解决了现有方法在处理模态缺失时跨模态一致性不足的问题。

### 背景

多模态模型整合自然语言和视觉信息已显著提升了表示模型的泛化能力。然而，在现实场景中当某些模态缺失或不可用时，这些模型的有效性会大幅下降。这种退化主要源于完整多模态数据与不完整模态场景之间的表示学习不一致。

### 目的

开发一种能够在模态缺失条件下保持稳健性能的多模态框架，解决现有方法在处理缺失模态时无法充分保留跨模态一致性的问题。

### 方法

提出PROMISE框架（PROMpting-Attentive HIerarchical ContraStive LEarning），创新性地将多模态提示学习整合到分层对比学习框架中，并配备专门设计的提示注意力机制。该机制能够为特定模态缺失的场景动态生成稳健且一致的表示。

### 主要发现

通过在基准数据集上进行的大量实验和全面的消融研究，证实PROMISE相比当前最先进的多模态方法具有优越的性能。提示注意力机制有效弥合了完整和不完整数据之间的表示差距。

### 结论

PROMISE框架通过创新的提示注意力机制和分层对比学习设计，成功解决了多模态模型在模态缺失情况下的性能退化问题，为现实应用中的跨模态表示学习提供了更稳健的解决方案。

### 翻译

整合自然语言和视觉信息的多模态模型已显著提升了表示模型的泛化能力。然而，在某些模态缺失或不可用的现实情况下，其有效性会大幅下降。这种退化主要源于完整多模态数据与不完整模态场景之间的表示学习不一致。现有方法通常通过相对简单的生成方法处理缺失模态，但这些方法未能充分保留跨模态一致性，导致性能不佳。为克服这一局限，我们提出了一种名为PROMISE的新型多模态框架，这是一种专门为模态缺失条件下稳健跨模态表示而设计的提示式注意力分层对比学习方法。具体而言，PROMISE创新性地将多模态提示学习整合到分层对比学习框架中，配备了特别设计的提示注意力机制。该机制能够为特定模态缺失的场景动态生成稳健且一致的表示，从而有效弥合完整数据与不完整数据之间的表示差距。在基准数据集上进行的大量实验及全面的消融研究明确表明，与当前最先进的多模态方法相比，PROMISE具有优越的性能。


### 论文摘要

Multimodal models integrating natural language and visual information have substantially improved generalization of representation models. However, their effectiveness significantly declines in real-world situations where certain modalities are missing or unavailable. This degradation primarily stems from inconsistent representation learning between complete multimodal data and incomplete modality scenarios. Existing approaches typically address missing modalities through relatively simplistic generation methods, yet these approaches fail to adequately preserve cross-modal consistency, leading to suboptimal performance. To overcome this limitation, we propose a novel multimodal framework named PROMISE, a PROMpting-Attentive HIerarchical ContraStive LEarning approach designed explicitly for robust cross-modal representation under conditions of missing modalities. Specifically, PROMISE innovatively incorporates multimodal prompt learning into a hierarchical contrastive learning framework, equipped with a specially designed prompt-attention mechanism. This mechanism dynamically generates robust and consistent representations for scenarios where particular modalities are absent, thereby effectively bridging the representational gap between complete and incomplete data. Extensive experiments conducted on benchmark datasets, along with comprehensive ablation studies, clearly demonstrate the superior performance of PROMISE compared to current state-of-the-art multimodal methods.

---

## 122. CardioEmbed: Domain-Specialized Text Embeddings for Clinical Cardiology

**论文链接:** [http://arxiv.org/abs/2511.10930v1](http://arxiv.org/abs/2511.10930v1)

**作者:** Richard J. Young, Alice M. Matthews

**发布时间:** 2025-11-14

**备注:** 14 pages, 6 figures

### GPT解析

### 总结

本研究开发了CardioEmbed，一个基于Qwen3-Embedding-8B的心脏病学领域专业嵌入模型，通过在心脏病学教材语料库上进行对比学习训练，实现了99.60%的心脏特异性语义检索准确率，比现有最佳模型MedTE提高了15.94个百分点。

### 背景

生物医学文本嵌入模型主要基于PubMed的研究文献开发，而临床心脏病学实践依赖于专业教材中的程序化知识和专业术语，这种研究实践差距限制了现有嵌入模型在临床心脏病学应用中的效果。

### 目的

开发一个专门针对心脏病学领域的嵌入模型，以提高临床应用的效果。

### 方法

基于Qwen3-Embedding-8B开发了CardioEmbed模型，使用对比学习在包含7本综合心脏病学教材（约150,000个句子）的语料库上进行训练，采用带批量内负样本的InfoNCE损失函数。

### 主要发现

在心脏特异性语义检索任务上达到99.60%的检索准确率，比MedTE提高15.94个百分点；在MTEB医疗基准测试中，BIOSSES得分为0.77（Spearman），SciFact得分为0.61（NDCG@10），在相关生物医学领域表现出有竞争力的性能。

### 结论

在综合临床教材上进行领域专业化训练可以实现接近完美的心脏病学检索效果，显著提高了临床应用的性能。

### 翻译

生物医学文本嵌入模型主要使用PubMed的研究文献开发，但临床心脏病学实践主要依赖于综合性教材中的程序化知识和专业术语，而非研究摘要。这种研究实践差距限制了现有嵌入模型在临床心脏病学应用中的有效性。本研究使用去重后约150,000个句子的七本综合心脏病学教材精心策划的语料库，通过对比学习训练了CardioEmbed，这是一个基于Qwen3-Embedding-8B的领域专业嵌入模型。该模型采用带批量内负样本的InfoNCE损失函数，在心脏特异性语义检索任务上实现了99.60%的检索准确率，比当前最先进的医疗嵌入模型MedTE提高了15.94个百分点。在MTEB医疗基准测试中，该模型获得了BIOSSES 0.77 Spearman和SciFact 0.61 NDCG@10的成绩，表明在相关生物医学领域具有竞争力的性能。在综合临床教材上进行领域专业化训练可实现接近完美的心脏病学检索（99.60% Acc@1），比MedTE提高了15.94个百分点。


### 论文摘要

Biomedical text embeddings have primarily been developed using research literature from PubMed, yet clinical cardiology practice relies heavily on procedural knowledge and specialized terminology found in comprehensive textbooks rather than research abstracts. This research practice gap limits the effectiveness of existing embedding models for clinical applications incardiology. This study trained CardioEmbed, a domain-specialized embedding model based on Qwen3-Embedding-8B, using contrastive learning on a curated corpus of seven comprehensive cardiology textbooks totaling approximately 150,000 sentences after deduplication. The model employs InfoNCE loss with in-batch negatives and achieves 99.60% retrieval accuracy on cardiac-specific semantic retrieval tasks, a +15.94 percentage point improvement over MedTE, the current state-of-the-art medical embedding model. On MTEB medical benchmarks, the model obtained BIOSSES 0.77 Spearman and SciFact 0.61 NDCG@10, indicating competitive performance on related biomedical domains. Domain-specialized training on comprehensive clinical textbooks yields near-perfect cardiology retrieval (99.60% Acc@1), improving over MedTE by +15.94 percentage points.

---

## 123. Automated Analysis of Learning Outcomes and Exam Questions Based on Bloom's Taxonomy

**论文链接:** [http://arxiv.org/abs/2511.10903v1](http://arxiv.org/abs/2511.10903v1)

**作者:** Ramya Kumar, Dhruv Gulwani, Sonit Singh

**发布时间:** 2025-11-14

**备注:** 7 Pages

### GPT解析

### 总结

这篇论文探讨了基于布鲁姆分类法对考试问题和学习成果进行自动分类的研究，比较了多种机器学习方法在小数据集上的表现。

### 背景

研究使用了一个包含600个句子的小型数据集，这些句子被标记为六个认知类别：知识、理解、应用、分析、综合和评价。

### 目的

比较不同机器学习模型在布鲁姆分类法分类任务上的表现，特别是在有限数据集情况下评估各模型的性能和过拟合问题。

### 方法

研究使用了传统机器学习模型（朴素贝叶斯、逻辑回归、支持向量机）、循环神经网络架构（LSTM、BiLSTM、GRU、BiGRU）、基于transformer的模型（BERT和RoBERTa）以及大型语言模型（OpenAI、Gemini、Ollama、Anthropic），并评估了不同的预处理和增强策略对模型性能的影响。

### 主要发现

传统机器学习方法中，使用数据增强的支持向量机表现最佳，准确率、召回率和F1分数均达到94%，且过拟合最小；RNN模型和BERT存在严重过拟合；RoBERTa初期克服过拟合但随后出现；大型语言模型中OpenAI和Gemini表现最佳，准确率约为0.72-0.73。

### 结论

在有限数据上训练复杂深度模型存在挑战，数据增强和简单算法（如增强SVM）在布鲁姆分类法分类中具有重要价值。

### 翻译

本文探讨了根据布鲁姆分类法对考试问题和学习成果进行自动分类。处理了一个包含600个句子的小型数据集，这些句子被标记为六个认知类别：知识、理解、应用、分析、综合和评价。研究使用了传统机器学习模型（朴素贝叶斯、逻辑回归、支持向量机）、循环神经网络架构（LSTM、BiLSTM、GRU、BiGRU）、基于transformer的模型（BERT和RoBERTa）以及大型语言模型（OpenAI、Gemini、Ollama、Anthropic）。每个模型在不同的预处理和增强策略（例如同义词替换、词嵌入等）下进行了评估。在传统机器学习方法中，使用数据增强的支持向量机取得了最佳整体性能，准确率、召回率和F1分数均达到94%，且过拟合最小。相比之下，RNN模型和BERT存在严重的过拟合问题，而RoBERTa最初克服了这一问题，但随着训练进行开始出现过拟合迹象。最后，大型语言模型的零样本评估表明，在测试的LLM中，OpenAI和Gemini表现最佳，准确率约为0.72-0.73，F1分数相当。这些发现强调了在有限数据上训练复杂深度模型的挑战，并强调了数据增强和简单算法（如增强SVM）在布鲁姆分类法分类中的价值。


### 论文摘要

This paper explores the automatic classification of exam questions and learning outcomes according to Bloom's Taxonomy. A small dataset of 600 sentences labeled with six cognitive categories - Knowledge, Comprehension, Application, Analysis, Synthesis, and Evaluation - was processed using traditional machine learning (ML) models (Naive Bayes, Logistic Regression, Support Vector Machines), recurrent neural network architectures (LSTM, BiLSTM, GRU, BiGRU), transformer-based models (BERT and RoBERTa), and large language models (OpenAI, Gemini, Ollama, Anthropic). Each model was evaluated under different preprocessing and augmentation strategies (for example, synonym replacement, word embeddings, etc.). Among traditional ML approaches, Support Vector Machines (SVM) with data augmentation achieved the best overall performance, reaching 94 percent accuracy, recall, and F1 scores with minimal overfitting. In contrast, the RNN models and BERT suffered from severe overfitting, while RoBERTa initially overcame it but began to show signs as training progressed. Finally, zero-shot evaluations of large language models (LLMs) indicated that OpenAI and Gemini performed best among the tested LLMs, achieving approximately 0.72-0.73 accuracy and comparable F1 scores. These findings highlight the challenges of training complex deep models on limited data and underscore the value of careful data augmentation and simpler algorithms (such as augmented SVM) for Bloom's Taxonomy classification.

---

## 124. MCN-CL: Multimodal Cross-Attention Network and Contrastive Learning for Multimodal Emotion Recognition

**论文链接:** [http://arxiv.org/abs/2511.10892v1](http://arxiv.org/abs/2511.10892v1)

**作者:** Feng Li, Ke Wu, Yongwei Li

**发布时间:** 2025-11-14

**备注:** Accepted by 32nd International Conference on MultiMedia Modeling (MMM 2026)

### GPT解析

### 总结

该论文提出了一种多模态交叉注意力网络和对比学习方法（MCN-CL）用于多模态情感识别，通过三重查询机制和困难负样本挖掘策略解决模态异质性和类别不平衡问题，实验证明该方法优于现有技术。

### 背景

多模态情感识别在心理健康监测、教育交互和人机交互等领域发挥关键作用。随着社交媒体场景中多模态数据的爆炸式增长，构建高效的跨模态融合框架进行情感识别的需求日益迫切。

### 目的

构建一个高效的多模态情感识别跨模态融合框架，解决现有方法面临的类别分布不均衡、动态面部动作单元时间建模复杂以及模态异质性导致的特征融合困难等问题。

### 方法

提出多模态交叉注意力网络和对比学习（MCN-CL）方法，使用三重查询机制和困难负样本挖掘策略去除特征冗余，同时保留重要的情感线索，有效解决模态异质性和类别不平衡问题。

### 主要发现

在IEMOCAP和MELD数据集上的实验结果表明，所提出的方法优于最先进的方法，加权F1分数分别提高了3.42%和5.73%。

### 结论

MCN-CL方法通过有效处理模态异质性和类别不平衡问题，显著提升了多模态情感识别的性能，为实际应用提供了更有效的解决方案。

### 翻译

多模态情感识别在许多领域（包括心理健康监测、教育交互和人机交互）中发挥着关键作用。然而，现有方法通常面临三大挑战：类别分布不均衡、动态面部动作单元时间建模的复杂性，以及由于模态异质性导致的特征融合困难。随着社交媒体场景中多模态数据的爆炸式增长，构建用于情感识别的高效跨模态融合框架的需求变得越来越迫切。为此，本文提出了用于多模态情感识别的多模态交叉注意力网络和对比学习（MCN-CL）。它使用三重查询机制和困难负样本挖掘策略来去除特征冗余，同时保留重要的情感线索，有效解决了模态异质性和类别不平衡问题。在IEMOCAP和MELD数据集上的实验结果表明，我们提出的方法优于最先进的方法，加权F1分数分别提高了3.42%和5.73%。


### 论文摘要

Multimodal emotion recognition plays a key role in many domains, including mental health monitoring, educational interaction, and human-computer interaction. However, existing methods often face three major challenges: unbalanced category distribution, the complexity of dynamic facial action unit time modeling, and the difficulty of feature fusion due to modal heterogeneity. With the explosive growth of multimodal data in social media scenarios, the need for building an efficient cross-modal fusion framework for emotion recognition is becoming increasingly urgent. To this end, this paper proposes Multimodal Cross-Attention Network and Contrastive Learning (MCN-CL) for multimodal emotion recognition. It uses a triple query mechanism and hard negative mining strategy to remove feature redundancy while preserving important emotional cues, effectively addressing the issues of modal heterogeneity and category imbalance. Experiment results on the IEMOCAP and MELD datasets show that our proposed method outperforms state-of-the-art approaches, with Weighted F1 scores improving by 3.42% and 5.73%, respectively.

---

## 125. From Efficiency to Adaptivity: A Deeper Look at Adaptive Reasoning in Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.10788v1](http://arxiv.org/abs/2511.10788v1)

**作者:** Chao Wu, Baoheng Li, Mingchen Gao, Zhenyi Wang

**发布时间:** 2025-11-13

### GPT解析

### 总结

这篇综述重新框定了大型语言模型中的推理概念，强调适应性推理的重要性，即根据输入特征如难度和不确定性来分配推理能力。作者形式化了不同类型的推理，提出了自适应推理的优化框架，并系统分类了现有方法。

### 背景

大型语言模型的最新进展使推理成为评估智能的中心基准。先前调查关注效率，通过缩短推理链或减少计算，但忽视了当前LLMs对任务复杂度应用统一推理策略的基本挑战。

### 目的

通过适应性视角重新框定推理概念，研究如何根据输入特征分配推理能力，并系统化现有方法以促进比较和理解。

### 方法

将演绎、归纳和溯因推理在LLM上下文中形式化；将自适应推理形式化为控制增强的策略优化问题；提出系统分类法，将方法分为基于训练的方法（通过强化学习、监督微调等内化适应性）和无需训练的方法（通过提示条件、反馈驱动等实现适应性）。

### 主要发现

当前LLMs对简单问题生成长推理链而对困难任务无法扩展推理；自适应推理需要平衡任务性能与计算成本；不同机制通过训练或无需训练的方式实现推理适应性。

### 结论

识别了自我评估、元推理和人类对齐推理控制方面的开放性挑战，为未来研究方向提供指导。

### 翻译

大型语言模型的最新进展使推理成为评估智能的中心基准。虽然先前的调查通过研究如何缩短推理链或减少计算来关注效率，但这种观点忽视了一个基本挑战：当前LLMs无论任务复杂度如何都应用统一的推理策略，为琐碎问题生成长推理链，同时无法为困难任务扩展推理。这篇综述通过适应性视角重新框定推理：根据输入特征（如难度和不确定性）分配推理能力的能力。我们做出三项贡献：首先，我们在LLM上下文中形式化演绎、归纳和溯因推理，将这些经典认知范式与其算法实现联系起来。其次，我们将自适应推理形式化为控制增强的策略优化问题，平衡任务性能与计算成本，区分学习策略与推理时控制机制。第三，我们提出系统分类法，将现有方法分为基于训练的方法（通过强化学习、监督微调和学习控制器将适应性内化）和无需训练的方法（通过提示条件、反馈驱动的停止和模块组合实现适应性）。该框架阐明了不同机制在实践中如何实现自适应推理，并能够系统化比较不同策略。我们通过识别自我评估、元推理和人类对齐推理控制方面的开放性挑战来结束本文。


### 论文摘要

Recent advances in large language models (LLMs) have made reasoning a central benchmark for evaluating intelligence. While prior surveys focus on efficiency by examining how to shorten reasoning chains or reduce computation, this view overlooks a fundamental challenge: current LLMs apply uniform reasoning strategies regardless of task complexity, generating long traces for trivial problems while failing to extend reasoning for difficult tasks. This survey reframes reasoning through the lens of {adaptivity}: the capability to allocate reasoning effort based on input characteristics such as difficulty and uncertainty. We make three contributions. First, we formalize deductive, inductive, and abductive reasoning within the LLM context, connecting these classical cognitive paradigms with their algorithmic realizations. Second, we formalize adaptive reasoning as a control-augmented policy optimization problem balancing task performance with computational cost, distinguishing learned policies from inference-time control mechanisms. Third, we propose a systematic taxonomy organizing existing methods into training-based approaches that internalize adaptivity through reinforcement learning, supervised fine-tuning, and learned controllers, and training-free approaches that achieve adaptivity through prompt conditioning, feedback-driven halting, and modular composition. This framework clarifies how different mechanisms realize adaptive reasoning in practice and enables systematic comparison across diverse strategies. We conclude by identifying open challenges in self-evaluation, meta-reasoning, and human-aligned reasoning control.

---

## 126. Practical Author Name Disambiguation under Metadata Constraints: A Contrastive Learning Approach for Astronomy Literature

**论文链接:** [http://arxiv.org/abs/2511.10722v1](http://arxiv.org/abs/2511.10722v1)

**作者:** Vicente Amado Olivo, Wolfgang Kerzendorf, Bangjing Lu, Joshua V. Shields, Andreas Flörs, Nutan Chen

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文介绍了一种名为'Neural Author Name Disambiguator'的神经作者姓名消歧方法，通过孪生神经网络仅依赖作者姓名、标题和摘要等基础元数据，在大型数字图书馆中实现高效准确的作者身份识别。

### 背景

准确识别和整理研究人员的出版物对确保学术认可、资金分配和招聘决策至关重要。然而，随着文献量增长，普遍存在的姓名歧义使准确链接研究人员作品变得困难。现有方法虽可扩展但大多依赖全面的元数据特征，而这些特征在大型数字图书馆中常不可用或不一致。

### 目的

开发一种在元数据有限的情况下，能够在大型数字图书馆中进行作者身份消歧的方法，解决现有方法依赖全面元数据但元数据常不可用的问题。

### 方法

将消歧任务表述为相似性学习问题，采用孪生神经网络区分不同出版物中的作者身份，仅依赖作者姓名、标题和摘要等广泛可用的元数据。构建了'Large-Scale Physics ORCiD Linked'数据集进行评估，通过交叉匹配NASA/ADS出版物与ORCiD，并利用基础模型将元数据嵌入为特征。

### 主要发现

该方法在成对消歧中达到高达94%的准确率，在将出版物聚类到研究人员身份的F1值超过95%。作者发布了测试数据集作为物理学和天文学的基准，为未来消歧方法提供真实评估条件。

### 结论

Neural Author Name Disambiguator算法在最小元数据的情况下表现出有效的消歧能力，为大型数字图书馆中的姓名歧义问题提供了可扩展的解决方案。

### 翻译

准确且恰当地整理单个研究人员的出版物对于确保适当的认可、指导研究资金的分配以及告知招聘决策至关重要。然而，由于文献量增长中普遍存在的姓名歧义问题，准确地将研究人员全部作品与个人身份进行分组和链接具有挑战性。算法作者姓名消歧提供了一种可扩展的消歧方法，但现有方法存在局限性。我们引入了神经作者姓名消歧器，一种在元数据可用性有限的情况下，在大型数字图书馆中进行作者身份消歧的方法。我们通过采用孪生神经网络，将消歧任务表述为相似性学习问题，仅依赖广泛可用的出版物元数据来区分不同出版物中的作者姓名。


### 论文摘要

The ability to distinctly and properly collate an individual researcher's publications is crucial for ensuring appropriate recognition, guiding the allocation of research funding and informing hiring decisions. However, accurately grouping and linking a researcher's entire body of work with their individual identity is challenging because of widespread name ambiguity across the growing literature. Algorithmic author name disambiguation provides a scalable approach to disambiguating author identities, yet existing methods have limitations. Many modern author name disambiguation methods rely on comprehensive metadata features such as venue or affiliation. Despite advancements in digitally indexing publications, metadata is often unavailable or inconsistent in large digital libraries(e.g. NASA/ADS). We introduce the Neural Author Name Disambiguator, a method that disambiguates author identities in large digital libraries despite limited metadata availability. We formulate the disambiguation task as a similarity learning problem by employing a Siamese neural network to disambiguate author names across publications relying solely on widely available publication metadata-author names, titles and abstracts. We construct the Large-Scale Physics ORCiD Linked dataset to evaluate the Neural Author Name Disambiguator by cross-matching NASA/ADS publications ORCiD. By leveraging foundation models to embed metadata into features, our model achieves up to 94% accuracy in pairwise disambiguation and over 95% F1 in clustering publications into their researcher identities. We release the testing dataset as a benchmark for physics and astronomy, providing realistic evaluation conditions for future disambiguation methods. The Neural Author Name Disambiguator algorithm demonstrates effective disambiguation with minimal metadata, offering a scalable solution for name ambiguity in large digital libraries.

---

## 127. Bytes of a Feather: Personality and Opinion Alignment Effects in Human-AI Interaction

**论文链接:** [http://arxiv.org/abs/2511.10544v1](http://arxiv.org/abs/2511.10544v1)

**作者:** Maximilian Eder, Clemens Lechner, Maurice Jakesch

**发布时间:** 2025-11-13

### GPT解析

### 总结

该研究探讨了AI助手个性化对用户交互结果和感知的影响，发现用户更倾向于与自己观点一致的AI模型，并认为这些模型更值得信赖、更有能力、更温暖和更具说服力。

### 背景

随着AI助手的交互越来越个性化，且这种个性化是由机器学习动态驱动的，人们对个性化如何影响交互结果和用户感知的理解有限。

### 目的

研究AI助手的人格特质和观点立场如何影响用户的交互体验和感知，特别是探索观点一致性和人格一致性对用户偏好的影响。

### 方法

进行了一项大规模对照实验，1000名参与者与具有特定人格特质和观点立场的AI助手进行交互。

### 主要发现

参与者普遍更喜欢与自己观点一致的模型；参与者认为观点一致的模型更值得信赖、更有能力、更温暖和更具说服力，这证实了AI-相似性-吸引力假设；相比之下，AI人格一致性几乎没有或只有很弱的影响，内向型参与者在评估内向型模型时认为其可信度和能力较低。

### 结论

这些发现突显了观点一致性作为AI个性化和用户偏好的核心维度，同时强调了需要对个性化AI的局限性和风险进行更深入的讨论。

### 翻译

与AI助手的交互越来越针对个别用户进行个性化。由于AI个性化是动态的且由机器学习驱动，我们对个性化如何影响交互结果和用户感知的理解有限。我们进行了一项大规模对照实验，1000名参与者与具有特定人格特质和观点立场的AI助手进行交互。我们的结果显示，参与者一致更喜欢与自己观点一致的模型。参与者还发现观点一致的模型更值得信赖、更有能力、更温暖和更具说服力，这证实了AI-相似性-吸引力假设。相比之下，我们观察到AI人格一致性几乎没有或只有很弱的影响，内向型参与者认为内向型模型的可信度和能力较低。这些发现突显了观点一致性作为AI个性化和用户偏好的核心维度，同时强调了需要对个性化AI的局限性和风险进行更深入的讨论。


### 论文摘要

Interactions with AI assistants are increasingly personalized to individual users. As AI personalization is dynamic and machine-learning-driven, we have limited understanding of how personalization affects interaction outcomes and user perceptions. We conducted a large-scale controlled experiment in which 1,000 participants interacted with AI assistants that took on certain personality traits and opinion stances. Our results show that participants consistently preferred to interact with models that shared their opinions. Participants also found opinion-aligned models more trustworthy, competent, warm, and persuasive, corroborating an AI-similarity-attraction hypothesis. In contrast, we observed no or only weak effects of AI personality alignment, with introvert models rated as less trustworthy and competent by introvert participants. These findings highlight opinion alignment as a central dimension of AI personalization and user preference, while underscoring the need for a more grounded discussion of the limits and risks of personalized AI.

---

## 128. Analogical Structure, Minimal Contextual Cues and Contrastive Distractors: Input Design for Sample-Efficient Linguistic Rule Induction

**论文链接:** [http://arxiv.org/abs/2511.10441v1](http://arxiv.org/abs/2511.10441v1)

**作者:** Chunyang Jiang, Paola Merlo

**发布时间:** 2025-11-13

### GPT解析

### 总结

研究通过类比范式组织方法，使轻量级模型在极少数据情况下达到高性能，仅需100个示例训练的模型F1值达0.95，优于零样本GPT-o3。

### 背景

大型语言模型通过在大量数据集上训练获得强大性能，但需要海量数据。

### 目的

探索类比范式组织是否能够使轻量级模型在最少数据的情况下达到类似性能。

### 方法

开发了一种计算方法，实现三种认知启发原则：类比结构、对比学习和最小上下文线索，使用结构化补全任务进行测试。

### 主要发现

轻量级模型仅用100个英语致使/非致使交替示例训练达到F1=0.95，优于零样本GPT-o3(F1=0.87)；类比组织和对比结构提高了性能；跨现象验证确认了方法鲁棒性。

### 结论

类比范式组织使传统方法需要少几个数量级的数据就能实现有竞争力的语言规则学习。

### 翻译

大型语言模型通过在大量数据集上训练获得强大性能。类比的范式组织是否能够使轻量级模型在最少数据的情况下达到这种性能？我们开发了一种实现三种认知启发原则的计算方法：类比结构、对比学习和最小上下文线索。我们使用结构化补全任务测试这种方法，模型需要从具有对比选项的类比模式中识别正确的句子补全。仅在100个英语致使/非致使交替的结构化示例上训练轻量级模型达到F1=0.95，优于零样本GPT-o3(F1=0.87)。消融研究确认类比组织和对比结构提高了性能，在所有架构上都一致超越了随机打乱的基线。使用未指定对象交替进行的跨现象验证复制了这些效率增益，确认了方法的鲁棒性。我们的结果表明，类比范式组织使得传统方法需要少几个数量级的数据就能实现有竞争力的语言规则学习。


### 论文摘要

Large language models achieve strong performance through training on vast datasets. Can analogical paradigm organization enable lightweight models to match this performance with minimal data? We develop a computational approach implementing three cognitive-inspired principles: analogical structure, contrastive learning, and minimal contextual cues. We test this approach with structured completion tasks where models identify correct sentence completions from analogical patterns with contrastive alternatives. Training lightweight models (BERT+CNN, $0.5M$ parameters) on only one hundred structured examples of English causative/inchoative alternations achieves $F1=0.95$, outperforming zero-shot \texttt{GPT-o3} ($F1=0.87$). Ablation studies confirm that analogical organization and contrastive structure improve performance, consistently surpassing randomly shuffled baselines across architectures. Cross-phenomenon validation using unspecified object alternations replicates these efficiency gains, confirming approach robustness. Our results show that analogical paradigm organization enables competitive linguistic rule learning with orders of magnitude less data than conventional approaches require.

---

## 129. Physics informed Transformer-VAE for biophysical parameter estimation: PROSAIL model inversion in Sentinel-2 imagery

**论文链接:** [http://arxiv.org/abs/2511.10387v2](http://arxiv.org/abs/2511.10387v2)

**作者:** Prince Mensah, Pelumi Victor Aderinto, Ibrahim Salihu Yusuf, Arnu Pretorius

**发布时间:** 2025-11-13

**备注:** My co-authors say some specific changes has to be made first

### GPT解析

### 总结

本文提出了一种物理信息的Transformer-VAE架构，用于从Sentinel-2卫星数据中反演PROSAIL辐射传输模型，以同时估计关键冠层参数。该方法仅使用模拟数据训练，但性能与使用真实图像的最先进方法相当，无需实地标签或真实图像校准，为全球植被监测提供了一种经济有效的自监督解决方案。

### 背景

从卫星图像中准确检索植被生物物理变量对生态系统监测和农业管理至关重要。传统的混合方法需要真实的卫星图像进行自监督训练，而本研究提出了一种新方法来解决这个问题。

### 目的

开发一种仅使用模拟数据训练就能达到与使用真实图像训练方法相当性能的模型，用于从Sentinel-2数据中同时估计叶面积指数(LAI)和冠层叶绿素含量(CCC)等关键冠层参数。

### 方法

提出了一种物理信息的Transformer-VAE架构，将PROSAIL模型作为可微分的物理解码器集成到模型中，确保推断的隐变量对应于物理上合理的叶和冠层属性。模型仅在模拟数据上进行训练，无需实地标签或真实图像校准。

### 主要发现

该方法在真实世界的数据集(FRM4Veg和BelSAR)上成功检索了叶面积指数(LAI)和冠层叶绿素含量(CCC)，检索精度与使用真实Sentinel-2数据训练的模型相当。仅使用模拟数据训练就能达到与使用真实图像训练的最先进方法相当的性能。

### 结论

将物理模型与先进的深度网络相结合可以改进辐射传输模型(RTMs)的反演，为大规模物理约束的植被特性遥感开辟了新的前景。这种方法提供了一种经济有效的自监督解决方案，用于全球植被监测。

### 翻译

从卫星图像中准确检索植被生物物理变量对生态系统监测和农业管理至关重要。在这项工作中，我们提出了一种物理信息的Transformer-VAE架构，用于反演PROSAIL辐射传输模型，以便从Sentinel-2数据中同时估计关键冠层参数。与需要真实卫星图像进行自监督训练的前混合方法不同，我们的模型完全在模拟数据上进行训练，却实现了与利用真实图像的最先进方法相当的性能。Transformer-VAE将PROSAIL模型作为可微分的物理解码器纳入，确保推断的隐变量对应于物理上合理的叶和冠层属性。我们在真实世界的野外数据集(FRM4Veg和BelSAR)上展示了叶面积指数(LAI)和冠层叶绿素含量(CCC)的检索，其精度与使用真实Sentinel-2数据训练的模型相当。我们的方法不需要实地标签或真实图像校准，为全球植被监测提供了一种经济有效的自监督解决方案。所提出的方法展示了如何将物理模型与先进的深度网络相结合以改进RTMs的反演，为大规模物理约束的植被特性遥感开辟了新的前景。


### 论文摘要

Accurate retrieval of vegetation biophysical variables from satellite imagery is crucial for ecosystem monitoring and agricultural management. In this work, we propose a physics-informed Transformer-VAE architecture to invert the PROSAIL radiative transfer model for simultaneous estimation of key canopy parameters from Sentinel-2 data. Unlike previous hybrid approaches that require real satellite images for self-supevised training. Our model is trained exclusively on simulated data, yet achieves performance on par with state-of-the-art methods that utilize real imagery. The Transformer-VAE incorporates the PROSAIL model as a differentiable physical decoder, ensuring that inferred latent variables correspond to physically plausible leaf and canopy properties. We demonstrate retrieval of leaf area index (LAI) and canopy chlorophyll content (CCC) on real-world field datasets (FRM4Veg and BelSAR) with accuracy comparable to models trained with real Sentinel-2 data. Our method requires no in-situ labels or calibration on real images, offering a cost-effective and self-supervised solution for global vegetation monitoring. The proposed approach illustrates how integrating physical models with advanced deep networks can improve the inversion of RTMs, opening new prospects for large-scale, physically-constrained remote sensing of vegetation traits.

---

## 130. Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment

**论文链接:** [http://arxiv.org/abs/2511.10334v1](http://arxiv.org/abs/2511.10334v1)

**作者:** Wenti Yin, Huaxin Zhang, Xiang Wang, Yuqing Lu, Yicheng Zhang, Bingquan Gong, Jialong Zuo, Li Yu, Changxin Gao, Nong Sang

**发布时间:** 2025-11-13

**备注:** Accepted to AAAI 2026. Code is available at https://github.com/lessiYin/DSANet

### GPT解析

### 总结

本文提出了一种解缠语义对齐网络（DSANet），用于弱监督视频异常检测。该方法通过粗粒度和细粒度两个层面分离正常和异常特征，提高可区分性。粗粒度层面使用自引导正常建模分支，细粒度层面采用解耦对比语义对齐机制。在XD-Violence和UCF-Crime两个基准测试上，DSANet超越了现有的最先进方法。

### 背景

弱监督视频异常检测的最新进展通过应用多模态基础模型（如CLIP）的多次实例学习范式来突出异常实例并分类类别，取得了显著性能。

### 目的

解决现有方法倾向于检测最显著响应片段、忽略挖掘多样化正常模式、以及因相似外观导致类别混淆的问题，从而提高细粒度分类结果。

### 方法

提出解缠语义对齐网络（DSANet），在粗粒度层面引入自引导正常建模分支，在学习的正常原型指导下重建输入视频特征；在细粒度层面提出解耦对比语义对齐机制，使用帧级异常分数将视频分解为中心事件和背景组件，然后应用视觉-语言对比学习增强类判别性表示。

### 主要发现

在XD-Violence和UCF-Crime两个标准基准上的综合实验表明，DSANet优于现有的最先进方法。

### 结论

DSANet通过从粗粒度和细粒度两个层面明确分离异常和正常特征，增强了可区分性，从而提高了视频异常检测的性能。

### 翻译

弱监督视频异常检测的最新进展通过应用基于CLIP等多模态基础模型的多次实例学习范式，突出异常实例并分类类别，取得了显著性能。然而，这些方法的目标可能倾向于检测最显著的响应片段，同时忽略挖掘与异常分离的多样化正常模式，并且由于相似外观容易导致类别混淆，导致细粒度分类结果不令人满意。因此，我们提出了一种新的解缠语义对齐网络（DSANet），明确从粗粒度和细粒度方面分离异常和正常特征，增强可区分性。具体而言，在粗粒度层面，我们引入了一个自引导的正常建模分支，在学习的正常原型指导下重建输入视频特征，鼓励模型利用视频固有的正常性线索，从而改善正常模式和异常事件的时间分离。在细粒度层面，我们提出了一个解耦对比语义对齐机制，首先使用帧级异常分数将每个视频在时间上分解为中心事件组件和背景中心组件，然后应用视觉-语言对比学习来增强类判别性表示。在XD-Violence和UCF-Crime两个标准基准上的综合实验表明，DSANet优于现有的最先进方法。


### 论文摘要

Recent advancements in weakly-supervised video anomaly detection have achieved remarkable performance by applying the multiple instance learning paradigm based on multimodal foundation models such as CLIP to highlight anomalous instances and classify categories. However, their objectives may tend to detect the most salient response segments, while neglecting to mine diverse normal patterns separated from anomalies, and are prone to category confusion due to similar appearance, leading to unsatisfactory fine-grained classification results. Therefore, we propose a novel Disentangled Semantic Alignment Network (DSANet) to explicitly separate abnormal and normal features from coarse-grained and fine-grained aspects, enhancing the distinguishability. Specifically, at the coarse-grained level, we introduce a self-guided normality modeling branch that reconstructs input video features under the guidance of learned normal prototypes, encouraging the model to exploit normality cues inherent in the video, thereby improving the temporal separation of normal patterns and anomalous events. At the fine-grained level, we present a decoupled contrastive semantic alignment mechanism, which first temporally decomposes each video into event-centric and background-centric components using frame-level anomaly scores and then applies visual-language contrastive learning to enhance class-discriminative representations. Comprehensive experiments on two standard benchmarks, namely XD-Violence and UCF-Crime, demonstrate that DSANet outperforms existing state-of-the-art methods.

---

## 131. Causal Model-Based Reinforcement Learning for Sample-Efficient IoT Channel Access

**论文链接:** [http://arxiv.org/abs/2511.10291v1](http://arxiv.org/abs/2511.10291v1)

**作者:** Aswin Arun, Christo Kurisummoottil Thomas, Rimalpudi Sarvendranath, Walid Saad

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出了一种基于因果模型的多智能体强化学习框架，用于解决无线网络中样本效率低的问题，同时提供可解释的决策能力。

### 背景

多智能体强化学习在无线网络用例如介质访问控制中具有优势，但在物联网中的实际部署受到样本效率低的阻碍。传统的基于模型的强化学习方法依赖于不可解释的黑盒模型，无法进行推理。

### 目的

开发一种能够提高样本效率并提供可解释性的因果模型多智能体强化学习框架，应用于资源受限的无线系统。

### 方法

利用因果学习工具，使用结构因果模型和基于注意力的推理网络表示网络变量间的因果依赖关系；开发可解释的因果模型捕获MAC控制消息对观察的影响、传输动作对结果的决定作用以及信道观察对奖励的影响；使用数据增强技术生成合成轨迹；通过近端策略优化进行策略优化。

### 主要发现

因果MBRL与黑盒方法相比具有指数级的样本复杂度优势；所提出的方法平均可减少58%的环境交互；与模型无关的基线相比收敛速度更快；通过基于注意力的因果归因提供可解释的调度决策。

### 结论

样本效率和可解释性的结合使因果MBRL成为资源受限无线系统的实用方法。

### 翻译

尽管多智能体强化学习在无线用例如介质访问控制中具有优势，但在物联网中的实际部署受到样本效率低的阻碍。为缓解这一挑战，可以利用基于模型的强化学习解决方案，然而传统的MBRL方法依赖于不可解释且无法推理的黑盒模型。相比之下，本文通过利用因果学习工具，开发了一种新颖的基于因果模型的MARL框架。特别是，所提出的模型可以使用结构因果模型和基于注意力的推理网络明确表示网络变量之间的因果依赖关系。然后开发可解释的因果模型来捕获MAC控制消息如何影响观察、传输动作如何决定结果以及信道观察如何影响奖励。接着使用数据增强技术利用学习的因果模型生成合成轨迹，并通过近端策略优化进行策略优化。分析结果表明，与黑盒方法相比，因果MBRL具有指数级的样本复杂度优势。广泛的模拟表明，平均而言，所提出的方法可以减少58%的环境交互，并与模型无关的基线相比实现更快的收敛。此外，所提出的方法还通过基于注意力的因果归因提供可解释的调度决策，揭示哪些网络条件驱动了策略。样本效率和可解释性的结合使因果MBRL成为资源受限无线系统的实用方法。


### 论文摘要

Despite the advantages of multi-agent reinforcement learning (MARL) for wireless use case such as medium access control (MAC), their real-world deployment in Internet of Things (IoT) is hindered by their sample inefficiency. To alleviate this challenge, one can leverage model-based reinforcement learning (MBRL) solutions, however, conventional MBRL approaches rely on black-box models that are not interpretable and cannot reason. In contrast, in this paper, a novel causal model-based MARL framework is developed by leveraging tools from causal learn- ing. In particular, the proposed model can explicitly represent causal dependencies between network variables using structural causal models (SCMs) and attention-based inference networks. Interpretable causal models are then developed to capture how MAC control messages influence observations, how transmission actions determine outcomes, and how channel observations affect rewards. Data augmentation techniques are then used to generate synthetic rollouts using the learned causal model for policy optimization via proximal policy optimization (PPO). Analytical results demonstrate exponential sample complexity gains of causal MBRL over black-box approaches. Extensive simulations demonstrate that, on average, the proposed approach can reduce environment interactions by 58%, and yield faster convergence compared to model-free baselines. The proposed approach inherently is also shown to provide interpretable scheduling decisions via attention-based causal attribution, revealing which network conditions drive the policy. The resulting combination of sample efficiency and interpretability establishes causal MBRL as a practical approach for resource-constrained wireless systems.

---

## 132. Inferring response times of perceptual decisions with Poisson variational autoencoders

**论文链接:** [http://arxiv.org/abs/2511.11480v1](http://arxiv.org/abs/2511.11480v1)

**作者:** Hayden R. Johnson, Anastasia N. Krouglova, Hadi Vafaii, Jacob L. Yates, Pedro J. Gonçalves

**发布时间:** 2025-11-14

**备注:** To appear at the NeurIPS 2025 Workshop on Data on the Mind and Brain

### GPT解析

### 总结

研究提出了一种感知决策的图像可计算模型，该模型通过高效的感官编码和神经放电活动的贝叶斯解码来生成选择和反应时间。

### 背景

深度神经网络可以很好地模拟感知决策的许多特性，但这些架构通常将决策视为即时读出，忽略了决策过程的动态时间特性。

### 目的

开发一个能够捕捉决策过程时间动态特性的感知决策模型，该模型能够生成选择和反应时间模式。

### 方法

使用泊松变分自编码器学习视觉刺激的无监督表征，将神经元建模为独立同质泊松过程；使用任务优化的解码器推断基于传入放电活动的动作近似后验；结合基于熵的停止规则构建完整的感知决策模型。

### 主要发现

该模型应用于MNIST数字分类时，能够重现感知决策的关键经验特征，包括随机变异性、右偏反应时间分布、反应时间随备选数量对数缩放（希克定律）以及速度-准确度权衡。

### 结论

该模型提供了一个原则性的、图像可计算的感知决策框架，能够捕捉决策过程的时间动态特性并生成与人类行为一致的选择和反应时间模式。

### 翻译

许多感知决策的特性可以通过深度神经网络很好地建模。然而，这类架构通常将决策视为即时读出，忽略了决策过程的动态时间特性。我们提出了一个感知决策的图像可计算模型，其中选择和反应时间来自高效的感官编码和神经放电活动的贝叶斯解码。我们使用泊松变分自编码器在速率编码神经元群体中学习视觉刺激的无监督表征，这些神经元被建模为独立同质泊松过程。然后使用任务优化的解码器持续推断基于传入放电活动的动作近似后验。将这些组件与基于熵的停止规则相结合，得到一个能够生成逐次试验选择和反应时间模式的感知决策模型。应用于MNIST数字分类时，该模型重现了感知决策的关键经验特征，包括随机变异性、右偏反应时间分布、反应时间随备选数量对数缩放（希克定律）以及速度-准确度权衡。


### 论文摘要

Many properties of perceptual decision making are well-modeled by deep neural networks. However, such architectures typically treat decisions as instantaneous readouts, overlooking the temporal dynamics of the decision process. We present an image-computable model of perceptual decision making in which choices and response times arise from efficient sensory encoding and Bayesian decoding of neural spiking activity. We use a Poisson variational autoencoder to learn unsupervised representations of visual stimuli in a population of rate-coded neurons, modeled as independent homogeneous Poisson processes. A task-optimized decoder then continually infers an approximate posterior over actions conditioned on incoming spiking activity. Combining these components with an entropy-based stopping rule yields a principled and image-computable model of perceptual decisions capable of generating trial-by-trial patterns of choices and response times. Applied to MNIST digit classification, the model reproduces key empirical signatures of perceptual decision making, including stochastic variability, right-skewed response time distributions, logarithmic scaling of response times with the number of alternatives (Hick's law), and speed-accuracy trade-offs.

---

## 133. Rethinking Efficient Mixture-of-Experts for Remote Sensing Modality-Missing Classification

**论文链接:** [http://arxiv.org/abs/2511.11460v1](http://arxiv.org/abs/2511.11460v1)

**作者:** Qinghao Gao, Jianhai Qu, Yunsong Li, Weiqiang Dong

**发布时间:** 2025-11-14

**备注:** 11 pages, 4 figures

### GPT解析

### 总结

本文提出了一种名为MaMOL的框架，用于解决多模态遥感分类中的模态缺失问题，通过双路由机制实现参数高效的适应，在各种缺失率下表现出优越的鲁棒性和泛化能力。

### 背景

多模态遥感分类常因环境干扰、传感器故障或大气效应导致模态缺失，严重影响分类性能。现有两阶段适应方法计算量大，且训练时假设完整的多模态数据，限制了其在实际不完整数据上的泛化能力。

### 目的

克服现有方法的局限性，提出一种能够处理模态缺失问题的框架，提高分类性能和泛化能力，同时保持计算效率。

### 方法

提出Missing-aware Mixture-of-Loras (MaMOL)框架，将模态缺失重新表述为多任务学习问题。MaMOL引入双路由机制：任务导向的动态路由器（针对不同缺失模式自适应激活专家）和模态特定的共享静态路由器（保持稳定的跨模态知识共享）。通过轻量级专家更新和共享专家重用实现参数高效适应。

### 主要发现

MaMOL在多个遥感基准测试中展现出优越的鲁棒性和泛化能力，即使在各种缺失率下也能保持高性能，且计算开销最小。在自然图像数据集上的迁移实验验证了其可扩展性和跨域适用性。

### 结论

MaMOL是一个通用且高效的解决方案，可用于处理不完整的多模态学习问题，具有实际应用价值。

### 翻译

多模态遥感分类常常因环境干扰、传感器故障或大气效应导致的模态缺失而受到影响，严重降低了分类性能。现有的两阶段适应方法计算量大，并且在训练时假设完整的多模态数据，限制了它们对现实世界不完整数据的泛化能力。为了克服这些问题，我们提出了一个称为缺失感知的LoRA混合模型(MaMOL)框架，该框架将模态缺失重新表述为多任务学习问题。MaMOL引入了双路由机制：一个面向任务的动态路由器，能够针对不同的缺失模式自适应地激活专家；以及一个模态特定的共享静态路由器，用于保持稳定的跨模态知识共享。与为每种缺失配置训练独立网络的前期方法不同，MaMOL通过轻量级专家更新和共享专家重用实现了参数高效的适应。在多个遥感基准测试中的实验表明，在各种缺失率下具有优越的鲁棒性和泛化能力，且计算开销最小。此外，在自然图像数据集上的迁移实验验证了其可扩展性和跨域适用性，突显了MaMOL作为不完整多模态学习的一个通用且高效的解决方案。


### 论文摘要

Multimodal classification in remote sensing often suffers from missing modalities caused by environmental interference, sensor failures, or atmospheric effects, which severely degrade classification performance. Existing two-stage adaptation methods are computationally expensive and assume complete multimodal data during training, limiting their generalization to real-world incompleteness. To overcome these issues, we propose a Missing-aware Mixture-of-Loras (MaMOL) framework that reformulates modality missing as a multi-task learning problem. MaMOL introduces a dual-routing mechanism: a task-oriented dynamic router that adaptively activates experts for different missing patterns, and a modality-specific-shared static router that maintains stable cross-modal knowledge sharing. Unlike prior methods that train separate networks for each missing configuration, MaMOL achieves parameter-efficient adaptation via lightweight expert updates and shared expert reuse. Experiments on multiple remote sensing benchmarks demonstrate superior robustness and generalization under varying missing rates, with minimal computational overhead. Moreover, transfer experiments on natural image datasets validate its scalability and cross-domain applicability, highlighting MaMOL as a general and efficient solution for incomplete multimodal learning.

---

## 134. VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2511.11450v1](http://arxiv.org/abs/2511.11450v1)

**作者:** Maximilian Rokuss, Moritz Langenberg, Yannick Kirchhoff, Fabian Isensee, Benjamin Hamm, Constantin Ulrich, Sebastian Regnery, Lukas Bauer, Efthimios Katsigiannopulos, Tobias Norajitra, Klaus Maier-Hein

**发布时间:** 2025-11-14

### GPT解析

### 总结

VoxTell是一个创新的视觉-语言模型，专门用于文本提示的体积医学图像分割，能够将自然语言描述转换为精确的3D分割掩码。

### 背景

医学图像分割通常需要专业知识来手动定义分割区域，传统方法可能难以处理复杂的医学概念或适应新的解剖结构。

### 目的

开发一个能够理解自然语言描述并将其映射到3D医学图像分割的模型，无需针对特定任务进行额外训练。

### 方法

VoxTell使用多阶段视觉-语言融合技术，在解码器层对齐文本和视觉特征，在超过62K的CT、MRI和PET体积上进行了训练，涵盖1000多种解剖和病理类别。

### 主要发现

模型在跨模态的未见数据集上实现了最先进的零样本性能，能够在熟悉概念上表现出色，同时推广到相关未见类别，并展示了强大的跨模态迁移能力和对临床语言的鲁棒性。

### 结论

VoxTell代表了医学图像分割领域的重要进展，通过自然语言处理实现了更直观、更灵活的分割方法，代码已在GitHub上公开可用。

### 翻译

我们介绍了VoxTell，一个用于文本提示的体积医学图像分割的视觉-语言模型。它将从单个单词到完整临床句子的自由形式描述映射到3D掩码。模型在超过62K的CT、MRI和PET体积上进行了训练，涵盖1000多种解剖和病理类别，使用跨解码器层的多阶段视觉-语言融合来对齐多尺度的文本和视觉特征。在跨模态的未见数据集上，它实现了最先进的零样本性能，在熟悉概念上表现出色，同时推广到相关的未见类别。大量实验进一步展示了强大的跨模态迁移能力，对语言变化和临床语言的鲁棒性，以及从真实世界文本中进行准确特定实例分割的能力。代码可在以下网址获取：https://www.github.com/MIC-DKFZ/VoxTell

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决医学图像分割方法局限于特定结构或模态、难以泛化到训练数据外分布的问题，以及现有文本引导分割模型对措辞变化敏感、无法处理复杂临床描述的局限性。这个问题很重要，因为准确的医学图像分割是现代诊断和治疗规划的基础，临床医生需要灵活描述解剖结构或利用现有放射学报告，而医学文本常描述特定空间关系而非固定类别，灵活的语言理解在临床场景中极具价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有文本引导医学分割方法的局限性，借鉴自然图像领域的开放词汇分割工作，设计多阶段视觉-语言融合策略。他们选择UNet架构而非Transformer，因为UNet在3D医学图像分割中表现更优，并引入深度监督促使模型早期整合文本指导。他们借鉴了MaskFormer架构但进行了关键修改，使用预训练文本编码器(Qwen3-Embedding-4B)来嵌入文本提示，并在整个解码器层次结构中重复跨模态交互，而非仅在晚期阶段融合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多阶段视觉-语言融合实现自由文本提示的3D医学图像分割，在整个解码器层次结构中重复跨模态交互，实现文本提示和体积特征之间的持续对齐。流程包括：1)输入处理，3D体积通过UNet编码器提取特征，文本通过预训练编码器嵌入；2)提示处理，transformer提示解码器生成多尺度文本指导；3)跨尺度融合，在每个尺度将文本特征与图像特征融合；4)深度监督，每个中间输出映射到预测并在多尺度使用辅助监督；5)训练，在大规模多模态数据集上训练，结合正负样本提示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)多阶段视觉-语言融合，在整个解码器层次结构中重复跨模态交互；2)深度监督，促使模型早期整合文本指导；3)大规模多模态数据集(62K+个体积，1K+类别)；4)全面词汇构建，统一同义词和扩展标签空间；5)面向实例的数据集处理细粒度空间提示。相比之前工作，VoxTell在融合策略上从单阶段晚期融合改为多阶段融合，增强了泛化能力到未见概念和跨模态场景，对多样化文本措辞更鲁棒，使用更大规模训练数据，并选择更适合3D医学图像的UNet架构而非Transformer。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VoxTell通过多阶段视觉-语言融合和深度监督，实现了从自由文本提示到3D掩码的映射，在已知和未知解剖结构以及跨模态场景中实现了最先进的性能，并展示了处理复杂临床语言描述的能力。'}


### 论文摘要

We introduce VoxTell, a vision-language model for text-prompted volumetric medical image segmentation. It maps free-form descriptions, from single words to full clinical sentences, to 3D masks. Trained on 62K+ CT, MRI, and PET volumes spanning over 1K anatomical and pathological classes, VoxTell uses multi-stage vision-language fusion across decoder layers to align textual and visual features at multiple scales. It achieves state-of-the-art zero-shot performance across modalities on unseen datasets, excelling on familiar concepts while generalizing to related unseen classes. Extensive experiments further demonstrate strong cross-modality transfer, robustness to linguistic variations and clinical language, as well as accurate instance-specific segmentation from real-world text. Code is available at: https://www.github.com/MIC-DKFZ/VoxTell

---

## 135. Retrofit: Continual Learning with Bounded Forgetting for Security Applications

**论文链接:** [http://arxiv.org/abs/2511.11439v1](http://arxiv.org/abs/2511.11439v1)

**作者:** Yiling He, Junchi Lei, Hongyu She, Shuo Shao, Xinran Zheng, Yiping Liu, Zhan Qin, Lorenzo Cavallaro

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为RETROFIT的无需数据回溯的持续学习方法，通过参数级合并和知识仲裁机制，有效解决了安全分析领域中深度学习模型在威胁环境演变时的性能退化问题。

### 背景

现代安全分析越来越多地依赖深度学习模型，但随着威胁环境演变和数据表示变化，模型性能会下降。虽然持续学习（CL）提供了解决方案，但现有方法通常需要完全重新训练或数据重放，这在数据敏感环境中不可行。此外，现有方法在安全关键场景中面临两个挑战：无旧数据情况下保留先验知识，以及最小干扰地整合新知识。

### 目的

开发一种无需历史数据的持续学习方法，实现有界的遗忘以进行有效的知识转移，解决安全分析领域中模型性能随时间退化的问题。

### 方法

RETROFIT方法通过参数级合并整合先前训练和新微调的模型，作为新旧知识的教学者，消除对历史数据的需求。为减少干扰，应用低秩和稀疏更新将参数变化限制在独立子空间，同时知识仲裁根据模型置信度动态平衡教师的贡献。

### 主要发现

在两个代表性应用中，RETROFIT持续减轻遗忘同时保持适应性。在时间漂移下的恶意软件检测中，保留分数从20.2%提高到38.6%，超过CL基线并超过oracle上限。在跨反编译级别的二进制摘要中，RETROFIT实现了先前工作中迁移学习的两倍左右的BLEU分数，并在跨表示泛化方面超过所有基线。

### 结论

RETROFIT是一种有效的持续学习方法，能够在数据敏感的安全场景中实现有界的遗忘和有效的知识转移，显著提升模型在动态威胁环境中的性能。

### 翻译

现代安全分析越来越多地由深度学习模型驱动，但随着威胁环境的演变和数据表示的变化，它们的性能往往会下降。虽然持续学习（CL）提供了一种保持模型有效性的有前景的范式，但许多方法依赖于完全重新训练或数据重放，这在数据敏感环境中是不可行的。此外，现有方法对于安全关键场景仍然不足，在知识转移方面面临两个相互关联的挑战：在没有旧数据的情况下保留先验知识，并以最小的干扰整合新知识。我们提出RETROFIT，一种无需数据回溯的持续学习方法，可以实现有界的遗忘以实现有效的知识转移。我们的核心思想是通过参数级合并来整合先前训练的新微调模型，作为新旧知识的教学者，从而消除对历史数据的需求。为了减少干扰，我们应用低秩和稀疏更新，将参数变化限制在独立的子空间中，同时知识仲裁根据模型置信度动态平衡教师的贡献。我们在两个代表性应用上的评估表明，RETROFIT持续减轻了遗忘，同时保持了适应性。在时间漂移下的恶意软件检测中，它显著提高了保留分数，从20.2%提高到38.6%，超过了CL基线，并在新数据上超过了oracle上限。在跨反编译级别的二进制摘要中，分析剥离的二进制文件尤其具有挑战性，RETROFIT实现了先前工作中使用的迁移学习的两倍左右的BLEU分数，并在跨表示泛化方面超过了所有基线。


### 论文摘要

Modern security analytics are increasingly powered by deep learning models, but their performance often degrades as threat landscapes evolve and data representations shift. While continual learning (CL) offers a promising paradigm to maintain model effectiveness, many approaches rely on full retraining or data replay, which are infeasible in data-sensitive environments. Moreover, existing methods remain inadequate for security-critical scenarios, facing two coupled challenges in knowledge transfer: preserving prior knowledge without old data and integrating new knowledge with minimal interference.   We propose RETROFIT, a data retrospective-free continual learning method that achieves bounded forgetting for effective knowledge transfer. Our key idea is to consolidate previously trained and newly fine-tuned models, serving as teachers of old and new knowledge, through parameter-level merging that eliminates the need for historical data. To mitigate interference, we apply low-rank and sparse updates that confine parameter changes to independent subspaces, while a knowledge arbitration dynamically balances the teacher contributions guided by model confidence. Our evaluation on two representative applications demonstrates that RETROFIT consistently mitigates forgetting while maintaining adaptability. In malware detection under temporal drift, it substantially improves the retention score, from 20.2% to 38.6% over CL baselines, and exceeds the oracle upper bound on new data. In binary summarization across decompilation levels, where analyzing stripped binaries is especially challenging, RETROFIT achieves around twice the BLEU score of transfer learning used in prior work and surpasses all baselines in cross-representation generalization.

---

## 136. BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning

**论文链接:** [http://arxiv.org/abs/2511.11421v1](http://arxiv.org/abs/2511.11421v1)

**作者:** Lan Li, Tao Hu, Da-Wei Zhou, Han-Jia Ye, De-Chuan Zhan

**发布时间:** 2025-11-14

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

BOFA是一种新颖的类增量学习框架，通过限制模型适应在CLIP的现有跨模态桥接层内，利用正交低秩融合机制防止知识遗忘，并结合跨模态混合原型提高分类性能，在实验中展现出优越的准确性和效率。

### 背景

类增量学习(CIL)旨在持续学习新类别而不忘记已获得的知识。视觉-语言模型如CLIP通过多模态监督提供强大的可迁移表示，但在应用于CIL时面临两个主要挑战：需要额外可学习模块增加复杂性和遗忘风险；以及未能充分整合视觉和文本模态的潜力。

### 目的

解决将CLIP应用于类增量学习时的两个主要挑战：(1)避免需要额外可学习模块导致的模型复杂化和遗忘问题；(2)充分实现视觉和文本模态的有效整合。

### 方法

提出BOFA(Bridge-layer Orthogonal Fusion for Adaptation)框架，将所有模型适应限制在CLIP的现有跨模态桥接层内，不增加额外参数或推理成本；利用正交低秩融合机制将参数更新限制在与过去任务特征正交的低秩'安全子空间'中，防止知识遗忘；采用跨模态混合原型，结合稳定的文本原型和从适应桥接层衍生的视觉原型提高分类性能。

### 主要发现

通过在标准基准上的大量实验，BOFA与现有方法相比实现了更高的准确性和效率。

### 结论

BOFA框架通过创新的桥接层适应机制和正交低秩融合方法，有效解决了CLIP在类增量学习中的应用挑战，实现了稳定的知识积累和优越的分类性能，无需数据回放。

### 翻译

类增量学习(CIL)旨在持续学习新类别而不忘记先前获得的知识。视觉-语言模型如CLIP通过多模态监督提供强大的可迁移表示，使其在CIL中具有广阔前景。然而，将CLIP应用于CIL存在两个主要挑战：(1)适应下游任务通常需要额外的可学习模块，增加了模型复杂性和遗忘风险；(2)尽管多模态表示提供了互补优势，但现有方法尚未充分实现有效整合视觉和文本模态的潜力。为解决这些问题，我们提出了BOFA(Bridge-layer Orthogonal Fusion for Adaptation)，一种用于CIL的新颖框架。BOFA将所有模型适应限制在CLIP现有的跨模态桥接层内，因此不添加额外参数或推理成本。为了防止在该层内发生遗忘，它利用正交低秩融合机制，将参数更新限制在数学上构建的与过去任务特征正交的低秩'安全子空间'中。这确保了稳定的知识积累，无需数据回放。此外，BOFA采用跨模态混合原型，将稳定的文本原型与从我们稳定适应的桥接层派生的视觉原型相结合，从而提高分类性能。在标准基准上的大量实验表明，与现有方法相比，BOFA实现了更高的准确性和效率。


### 论文摘要

Class-Incremental Learning (CIL) aims to continually learn new categories without forgetting previously acquired knowledge. Vision-language models such as CLIP offer strong transferable representations via multi-modal supervision, making them promising for CIL. However, applying CLIP to CIL poses two major challenges: (1) adapting to downstream tasks often requires additional learnable modules, increasing model complexity and susceptibility to forgetting; and (2) while multi-modal representations offer complementary strengths, existing methods have yet to fully realize their potential in effectively integrating visual and textual modalities. To address these issues, we propose BOFA (Bridge-layer Orthogonal Fusion for Adaptation), a novel framework for CIL. BOFA confines all model adaptation exclusively to CLIP's existing cross-modal bridge-layer, thereby adding no extra parameters or inference cost. To prevent forgetting within this layer, it leverages Orthogonal Low-Rank Fusion, a mechanism that constrains parameter updates to a low-rank ``safe subspace" mathematically constructed to be orthogonal to past task features. This ensures stable knowledge accumulation without data replay. Furthermore, BOFA employs a cross-modal hybrid prototype that synergizes stable textual prototypes with visual counterparts derived from our stably adapted bridge-layer, enhancing classification performance. Extensive experiments on standard benchmarks show that BOFA achieves superior accuracy and efficiency compared to existing methods.

---

## 137. One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs

**论文链接:** [http://arxiv.org/abs/2511.11137v1](http://arxiv.org/abs/2511.11137v1)

**作者:** Samuel Auroy, Pavlos Protopapas

**发布时间:** 2025-11-14

**备注:** Accepted at Machine Learning and the Physical Sciences Workshop, NeurIPS 2025

### GPT解析

### 总结

提出了一种结合摄动理论和一次性迁移学习的物理信息神经网络框架，用于高效求解非线性偏微分方程，能够在不重新训练的情况下快速适应新问题实例。

### 背景

非线性偏微分方程的求解是科学计算中的挑战性问题，传统方法通常需要针对每个新问题实例重新计算，效率较低。

### 目的

开发一种能够高效解决非线性偏微分方程的方法，实现一次训练后快速适应不同参数、边界条件或初始条件的新问题实例。

### 方法

将非线性PDE分解为一系列线性子问题，使用多头物理信息神经网络(Multi-Head PINN)学习线性算子的潜在表示，然后推导出闭式解来适应新问题实例。

### 主要发现

在KPP-Fisher方程和波动方程上验证，误差约为1e-3，适应新问题实例用时不到0.2秒，与经典求解器相当但迁移速度更快；敏感性分析显示误差增长可预测，明确了方法有效范围。

### 结论

成功将一次性迁移学习从非线性ODE扩展到PDE，推导出适应新PDE实例的闭式解，并在典型非线性PDE上展示了准确性和效率，为未来扩展到依赖导数的非线性和高维PDE提供了基础。

### 翻译

我们提出了一种结合摄动理论和一次性迁移学习的物理信息神经网络框架，用于求解非线性偏微分方程。包含多项式项的非线性PDE被分解为一系列线性子问题，使用多头物理信息神经网络高效求解。一旦学习到线性算子的潜在表示，就可以获得具有不同摄动、强迫项或边界/初始条件的新PDE实例的闭式解，无需重新训练。我们在KPP-Fisher方程和波动方程上验证了该方法，误差达到1e-3量级，同时能在0.2秒内适应新的问题实例；与经典求解器相当的准确性但迁移速度更快。敏感性分析显示误差随epsilon和多项式阶数的增长是可预测的，明确了该方法的有效范围。我们的贡献包括：(i)将一次性迁移学习从非线性ODE扩展到PDE，(ii)推导出适应新PDE实例的闭式解，(iii)在典型非线性PDE上展示了准确性和效率。最后我们概述了扩展到依赖导数的非线性和高维PDE的方案。


### 论文摘要

We propose a framework for solving nonlinear partial differential equations (PDEs) by combining perturbation theory with one-shot transfer learning in Physics-Informed Neural Networks (PINNs). Nonlinear PDEs with polynomial terms are decomposed into a sequence of linear subproblems, which are efficiently solved using a Multi-Head PINN. Once the latent representation of the linear operator is learned, solutions to new PDE instances with varying perturbations, forcing terms, or boundary/initial conditions can be obtained in closed form without retraining.   We validate the method on KPP-Fisher and wave equations, achieving errors on the order of 1e-3 while adapting to new problem instances in under 0.2 seconds; comparable accuracy to classical solvers but with faster transfer. Sensitivity analyses show predictable error growth with epsilon and polynomial degree, clarifying the method's effective regime.   Our contributions are: (i) extending one-shot transfer learning from nonlinear ODEs to PDEs, (ii) deriving a closed-form solution for adapting to new PDE instances, and (iii) demonstrating accuracy and efficiency on canonical nonlinear PDEs. We conclude by outlining extensions to derivative-dependent nonlinearities and higher-dimensional PDEs.

---

## 138. Unsupervised Robust Domain Adaptation: Paradigm, Theory and Algorithm

**论文链接:** [http://arxiv.org/abs/2511.11009v1](http://arxiv.org/abs/2511.11009v1)

**作者:** Fuxiang Huang, Xiaowei Fu, Shiyu Ye, Lina Ma, Wen Li, Xinbo Gao, David Zhang, Lei Zhang

**发布时间:** 2025-11-14

**备注:** To appear in IJCV

### GPT解析

### 总结

该研究解决了无监督领域适应(UDA)中对抗攻击鲁棒性的问题，提出了无监督鲁棒领域适应(URDA)范式和DART算法，有效提升了模型在对抗攻击下的鲁棒性同时保持领域适应能力。

### 背景

无监督领域适应旨在将有标签源领域知识转移到无标签目标领域，但大多数方法忽视了对抗攻击的鲁棒性。虽然普通对抗训练(VAT)能提高模型鲁棒性，但在UDA中效果不佳。

### 目的

回答三个关键问题：1)为什么VAT在UDA中效果不佳；2)攻击下的泛化边界理论及其与经典UDA理论的关系；3)如何实现无需复杂修改的鲁棒化训练。

### 方法

探索揭示了UDA+VAT范式的固有纠缠挑战，提出URDA范式和理论，并设计了DART算法，这是一种两步训练过程：首先预训练任意UDA模型，然后通过解缠蒸馏进行即时的鲁棒化后训练。

### 主要发现

揭示了通用UDA+VAT范式中的固有纠缠挑战，建立了URDA范式和理论，实验证明DART算法在四个基准数据集上有效增强了鲁棒性同时保持了领域适应性。

### 结论

首次建立了URDA范式和理论，DART算法简单有效，能够在不进行复杂修改的情况下提升模型对抗攻击的鲁棒性同时保持领域适应能力。

### 翻译

无监督领域适应(UDA)旨在通过解决领域偏移问题，将有标签源领域的知识转移到无标签目标领域。大多数UDA方法强调迁移能力，但常忽视对抗攻击的鲁棒性。虽然普通对抗训练(VAT)能提高深度神经网络的鲁棒性，但在UDA中效果有限。本文聚焦于回答三个关键问题：1)为什么在防御方面有效的VAT在UDA范式中失败？2)攻击下的泛化边界理论是什么，它如何从经典UDA理论演变而来？3)如何在不进行复杂修改的情况下实现鲁棒化训练？具体而言，作者探索并揭示了通用UDA+VAT范式中的固有纠缠挑战，并提出无监督鲁棒领域适应(URDA)范式。他们进一步推导了URDA范式的泛化边界理论，使其能够抵抗对抗噪声和领域偏移。据作者所知，这是首次建立URDA范式和理论。作者还引入了一种简单、新颖且有效的URDA算法，称为解缠对抗鲁棒训练(DART)，这是一种两步训练过程，确保了可迁移性和鲁棒性。在四个基准数据集上的实验（有/无攻击）表明，DART有效增强了鲁棒性同时保持了领域适应性，验证了URDA范式和理论。


### 论文摘要

Unsupervised domain adaptation (UDA) aims to transfer knowledge from a label-rich source domain to an unlabeled target domain by addressing domain shifts. Most UDA approaches emphasize transfer ability, but often overlook robustness against adversarial attacks. Although vanilla adversarial training (VAT) improves the robustness of deep neural networks, it has little effect on UDA. This paper focuses on answering three key questions: 1) Why does VAT, known for its defensive effectiveness, fail in the UDA paradigm? 2) What is the generalization bound theory under attacks and how does it evolve from classical UDA theory? 3) How can we implement a robustification training procedure without complex modifications? Specifically, we explore and reveal the inherent entanglement challenge in general UDA+VAT paradigm, and propose an unsupervised robust domain adaptation (URDA) paradigm. We further derive the generalization bound theory of the URDA paradigm so that it can resist adversarial noise and domain shift. To the best of our knowledge, this is the first time to establish the URDA paradigm and theory. We further introduce a simple, novel yet effective URDA algorithm called Disentangled Adversarial Robustness Training (DART), a two-step training procedure that ensures both transferability and robustness. DART first pre-trains an arbitrary UDA model, and then applies an instantaneous robustification post-training step via disentangled distillation.Experiments on four benchmark datasets with/without attacks show that DART effectively enhances robustness while maintaining domain adaptability, and validate the URDA paradigm and theory.

---

## 139. Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment

**论文链接:** [http://arxiv.org/abs/2511.10987v1](http://arxiv.org/abs/2511.10987v1)

**作者:** Wenbin Bai, Qiyu Chen, Xiangbo Lin, Jianwen Li, Quancheng Li, Hejiang Pan, Yi Sun

**发布时间:** 2025-11-14

**备注:** 13 pages, 15 figures. Accepted by AAAI 2026

### GPT解析

### 总结

提出了一种与手无关的操作转换系统，可将人类手操作视频转换为高质量的灵巧操作轨迹，解决了多指机器人手平台数据收集困难和数据稀缺问题。

### 背景

多指机器人手平台收集操作数据存在固有难度和可扩展性有限的问题，导致数据严重稀缺，阻碍了数据驱动的灵巧操作策略学习研究。

### 目的

开发一种能够高效转换人类手操作序列为灵巧操作轨迹的系统，无需大量训练数据，解决数据稀缺问题。

### 方法

设计了一个渐进式转换框架：首先基于运动匹配建立灵巧手的基本控制信号；然后使用动作空间重新缩放和拇指引导初始化训练残差策略，在统一奖励下动态优化接触交互；最后计算手腕控制轨迹以保持操作语义。系统仅使用人类手操作视频即可自动配置参数。

### 主要发现

该系统能自动生成平滑且语义正确的灵巧手操作，忠实地重现人类意图，平均转换成功率达到73%，具有高效性和强通用性。

### 结论

该框架为收集机器人灵巧操作数据提供了一种易于实施和可扩展的方法，有效解决了数据稀缺问题。

### 翻译

由于多指机器人手硬件平台收集操作数据存在固有难度和可扩展性有限的问题，导致严重的数据稀缺，阻碍了数据驱动的灵巧操作策略学习研究。为应对这一挑战，我们提出了一种与手无关的操作转换系统。它能够高效地将人类手操作序列从演示视频转换为高质量的灵巧操作轨迹，无需大量训练数据。为解决人类手与灵巧手之间的多维差异以及灵巧手高自由度协调控制的挑战，我们设计了一个渐进式转换框架：首先，基于运动匹配建立灵巧手的基本控制信号；随后，使用动作空间重新缩放和拇指引导初始化训练残差策略，在统一奖励下动态优化接触交互；最后，计算手腕控制轨迹，目标是保持操作语义。仅使用人类手操作视频，我们的系统即可为不同任务自动配置参数，平衡灵巧手、物体类别和任务之间的运动匹配和动态优化。大量实验结果表明，我们的框架可以自动生成平滑且语义正确的灵巧手操作，忠实地重现人类意图，平均转换成功率达到73%，为收集机器人灵巧操作数据提供了一种易于实施和可扩展的高效且通用性强的方法。


### 论文摘要

The inherent difficulty and limited scalability of collecting manipulation data using multi-fingered robot hand hardware platforms have resulted in severe data scarcity, impeding research on data-driven dexterous manipulation policy learning. To address this challenge, we present a hand-agnostic manipulation transfer system. It efficiently converts human hand manipulation sequences from demonstration videos into high-quality dexterous manipulation trajectories without requirements of massive training data. To tackle the multi-dimensional disparities between human hands and dexterous hands, as well as the challenges posed by high-degree-of-freedom coordinated control of dexterous hands, we design a progressive transfer framework: first, we establish primary control signals for dexterous hands based on kinematic matching; subsequently, we train residual policies with action space rescaling and thumb-guided initialization to dynamically optimize contact interactions under unified rewards; finally, we compute wrist control trajectories with the objective of preserving operational semantics. Using only human hand manipulation videos, our system automatically configures system parameters for different tasks, balancing kinematic matching and dynamic optimization across dexterous hands, object categories, and tasks. Extensive experimental results demonstrate that our framework can automatically generate smooth and semantically correct dexterous hand manipulation that faithfully reproduces human intentions, achieving high efficiency and strong generalizability with an average transfer success rate of 73%, providing an easily implementable and scalable method for collecting robot dexterous manipulation data.

---

## 140. Heterogeneous Multisource Transfer Learning via Model Averaging for Positive-Unlabeled Data

**论文链接:** [http://arxiv.org/abs/2511.10919v1](http://arxiv.org/abs/2511.10919v1)

**作者:** Jialei Liu, Jun Liao, Kuangnan Fang

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究提出了一种新颖的带模型平均的迁移学习框架，用于解决正样本-未标记(PU)学习中的数据稀缺和隐私限制问题。该方法整合来自异构数据源的信息，无需直接数据共享，并通过理论证明和实验验证了其有效性和优越性。

### 背景

正样本-未标记学习由于缺乏明确的负样本标记而带来独特挑战，特别是在欺诈检测和医疗诊断等高风险领域。这些领域通常面临数据稀缺和隐私保护的限制。

### 目的

为了解决数据稀缺性和隐私约束问题，开发一种能够整合异构数据源信息的方法，用于提升PU学习场景下的预测性能。

### 方法

提出了一种带模型平均的迁移学习框架，整合来自完全二元标记、半监督和PU数据集的信息。针对每种源域类型，采用定制化的逻辑回归模型，并通过模型平均将知识转移到PU目标域。通过最小化Kullback-Leibler散度的交叉验证标准确定组合源模型的最优权重。

### 主要发现

建立了权重最优性和收敛性的理论保证，涵盖了错误指定和正确指定的目标模型，并扩展到使用稀疏惩罚估计器的高维设置。实验表明该方法在预测准确性和稳健性方面优于其他比较方法。

### 结论

大量的模拟和真实世界的信用风险数据分析表明，该方法在标记数据有限和异构环境下表现优异，特别是在预测准确性和稳健性方面优于其他比较方法。

### 翻译

正样本-未标记(PU)学习由于缺乏明确的负样本标记而带来独特挑战，特别是在欺诈检测和医疗诊断等高风险领域。为了解决数据稀缺和隐私限制，我们提出了一种新颖的带模型平均的迁移学习框架，整合来自异构数据源的信息——包括完全二元标记、半监督和PU数据集——无需直接数据共享。对于每个源域类型，进行定制化的逻辑回归模型，并通过模型平均将知识转移到PU目标域。通过最小化Kullback-Leibler散度的交叉验证标准确定组合源模型的最优权重。我们建立了权重最优性和收敛性的理论保证，涵盖了错误指定和正确指定的目标模型，并进一步扩展到使用稀疏惩罚估计器的高维设置。大量的模拟和真实世界的信用风险数据分析表明，我们的方法在预测准确性和稳健性方面优于其他比较方法，特别是在标记数据有限和异构环境下。


### 论文摘要

Positive-Unlabeled (PU) learning presents unique challenges due to the lack of explicitly labeled negative samples, particularly in high-stakes domains such as fraud detection and medical diagnosis. To address data scarcity and privacy constraints, we propose a novel transfer learning with model averaging framework that integrates information from heterogeneous data sources - including fully binary labeled, semi-supervised, and PU data sets - without direct data sharing. For each source domain type, a tailored logistic regression model is conducted, and knowledge is transferred to the PU target domain through model averaging. Optimal weights for combining source models are determined via a cross-validation criterion that minimizes the Kullback-Leibler divergence. We establish theoretical guarantees for weight optimality and convergence, covering both misspecified and correctly specified target models, with further extensions to high-dimensional settings using sparsity-penalized estimators. Extensive simulations and real-world credit risk data analyses demonstrate that our method outperforms other comparative methods in terms of predictive accuracy and robustness, especially under limited labeled data and heterogeneous environments.

---

## 141. Graph Attention Network for Predicting Duration of Large-Scale Power Outages Induced by Natural Disasters

**论文链接:** [http://arxiv.org/abs/2511.10898v1](http://arxiv.org/abs/2511.10898v1)

**作者:** Chenghao Duan, Chuanyi Ji

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究提出了一种基于图注意力网络的创新方法，用于预测自然灾害导致的停电持续时间，模型表现出色且优于现有方法。

### 背景

自然灾害如飓风、野火和冬季风暴在美国引发了大规模停电，造成巨大的经济和社会影响。

### 目的

准确预测停电恢复和影响对于电网的韧性至关重要。

### 方法

研究人员提出了一种新方法，通过图注意力网络(GAT)估算严重天气导致的停电持续时间。该网络使用无监督预训练的简单结构，随后进行半监督学习。研究使用了影响美国东南部八个州501个县的四次主要飓风的数据。

### 主要发现

模型表现出色（准确率>93%），在整体性能和分类准确率上都比现有的XGBoost、随机森林、GCN和简单GAT方法高出2%-15%。

### 结论

机器学习框架可以从地理空间和天气数据中有效估算停电持续时间，图注意力网络是一个有效的方法。

### 翻译

自然灾害如飓风、野火和冬季风暴已在美国引发大规模停电，造成巨大的经济和社会影响。准确预测停电恢复和影响对电网韧性至关重要。机器学习的最新进展为从地理空间和天气数据估算停电持续时间提供了可行的框架。然而，在现实环境中，该任务存在三个主要挑战：数据的空间依赖性、影响的空间异质性和中等规模事件数据。我们提出了一种通过图注意力网络(GAT)估算严重天气导致的停电持续时间的新方法。我们的网络使用无监督预训练的简单结构，随后进行半监督学习。我们使用了影响美国东南部八个州501个县的四次主要飓风的数据。该模型表现出色（准确率>93%），在整体性能和分类准确率上都比现有的XGBoost、随机森林、GCN和简单GAT方法高出2%-15%。


### 论文摘要

Natural disasters such as hurricanes, wildfires, and winter storms have induced large-scale power outages in the U.S., resulting in tremendous economic and societal impacts. Accurately predicting power outage recovery and impact is key to resilience of power grid. Recent advances in machine learning offer viable frameworks for estimating power outage duration from geospatial and weather data. However, three major challenges are inherent to the task in a real world setting: spatial dependency of the data, spatial heterogeneity of the impact, and moderate event data. We propose a novel approach to estimate the duration of severe weather-induced power outages through Graph Attention Networks (GAT). Our network uses a simple structure from unsupervised pre-training, followed by semi-supervised learning. We use field data from four major hurricanes affecting $501$ counties in eight Southeastern U.S. states. The model exhibits an excellent performance ($>93\%$ accuracy) and outperforms the existing methods XGBoost, Random Forest, GCN and simple GAT by $2\% - 15\%$ in both the overall performance and class-wise accuracy.

---

## 142. CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening

**论文链接:** [http://arxiv.org/abs/2511.10896v1](http://arxiv.org/abs/2511.10896v1)

**作者:** Lihua Jian, Jiabo Liu, Shaowu Wu, Lihui Chen

**发布时间:** 2025-11-14

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

本文提出了一种名为CLIPPan的无监督全色锐化框架，利用CLIP视觉语言模型作为监督，解决了监督式全色锐化神经网络面临的分辨率域适应挑战。

### 背景

监督式全色锐化神经网络面临域适应挑战，因为模拟的低分辨率训练数据与真实世界全分辨率场景之间存在固有差异。

### 目的

提出一个无监督全色锐化框架CLIPPan，直接在全分辨率下进行模型训练，利用CLIP视觉语言模型作为监督。

### 方法

引入轻量级微调管道使CLIP适应全色锐化任务；提出结合语义语言约束的新损失函数，将图像级融合转换与协议对齐的文本提示保持一致，使CLIPPan能够使用语言作为监督信号无需真实值。

### 主要发现

CLIPPan在各种全色锐化骨干网络和真实世界数据集上持续提高了光谱和空间保真度，为无监督全分辨率全色锐化树立了新的最先进水平。

### 结论

CLIPPan通过利用CLIP视觉语言模型作为监督，有效解决了监督式方法面临的域适应挑战，实现了高质量的全色锐化。

### 翻译

尽管监督式全色锐化神经网络取得了显著进展，但这些方法由于模拟低分辨率训练数据与真实世界全分辨率场景之间的固有差异而面临分辨率域适应挑战。为了弥合这一差距，我们提出了一种名为CLIPPan的无监督全色锐化框架，它通过将CLIP（一种视觉语言模型）作为监督，直接在全分辨率下进行模型训练。然而，直接应用CLIP来监督全色锐化仍然具有挑战性，因为它对自然图像存在固有偏见且对全色锐化任务理解有限。因此，我们首先引入了一个轻量级微调管道，使CLIP能够识别低分辨率多光谱、全色图像和高分辨率多光谱图像，并理解全色锐化过程。然后，基于适应后的CLIP，我们提出了一种结合语义语言约束的新型损失函数，将图像级融合转换与协议对齐的文本提示（如Wald或Khan的描述）保持一致，从而使CLIPPan能够使用语言作为强大的监督信号，无需真实值指导融合学习。大量实验表明，CLIPPan在各种全色锐化骨干网络和真实世界数据集上持续提高了光谱和空间保真度，为无监督全分辨率全色锐化树立了新的最先进水平。


### 论文摘要

Despite remarkable advancements in supervised pansharpening neural networks, these methods face domain adaptation challenges of resolution due to the intrinsic disparity between simulated reduced-resolution training data and real-world full-resolution scenarios.To bridge this gap, we propose an unsupervised pansharpening framework, CLIPPan, that enables model training at full resolution directly by taking CLIP, a visual-language model, as a supervisor. However, directly applying CLIP to supervise pansharpening remains challenging due to its inherent bias toward natural images and limited understanding of pansharpening tasks. Therefore, we first introduce a lightweight fine-tuning pipeline that adapts CLIP to recognize low-resolution multispectral, panchromatic, and high-resolution multispectral images, as well as to understand the pansharpening process. Then, building on the adapted CLIP, we formulate a novel \textit{loss integrating semantic language constraints}, which aligns image-level fusion transitions with protocol-aligned textual prompts (e.g., Wald's or Khan's descriptions), thus enabling CLIPPan to use language as a powerful supervisory signal and guide fusion learning without ground truth. Extensive experiments demonstrate that CLIPPan consistently improves spectral and spatial fidelity across various pansharpening backbones on real-world datasets, setting a new state of the art for unsupervised full-resolution pansharpening.

---

## 143. Leveraging Parameter Space Symmetries for Reasoning Skill Transfer in LLMs

**论文链接:** [http://arxiv.org/abs/2511.10850v1](http://arxiv.org/abs/2511.10850v1)

**作者:** Stefan Horoi, Sangwoo Cho, Supriyo Chakraborty, Shi-Xiong Zhang, Sambit Sahu, Guy Wolf, Genta Indra Winata

**发布时间:** 2025-11-13

### GPT解析

### 总结

本研究提出了一种通过先对齐模型参数空间来解决任务算术中负干扰问题的方法，成功将高级推理技能转移到非推理模型中。

### 背景

任务算术是一种在大语言模型之间转移技能的强大技术，但当模型在训练过程中出现分歧时，它常常受到负干扰的影响。

### 目的

解决任务算术中的负干扰问题，使模型能够更有效地转移技能。

### 方法

首先对齐模型的参数空间，利用Transformer架构的固有排列、旋转和缩放对称性；将参数空间对齐适应现代分组查询注意力和SwiGLU层；探索基于权重和基于激活的方法；采用对齐优先策略。

### 主要发现

成功将高级推理技能转移到非推理模型；在具有挑战性的推理基准测试中，该方法始终优于标准任务算术。

### 结论

这项工作提供了一种有效的方法，用于在演进的LLM家族之间合并和转移专门技能，减少了冗余的微调，提高了模型适应性。

### 翻译

任务算术是一种在大语言模型之间转移技能的强大技术，但当模型在训练过程中出现分歧时，它常常受到负干扰的影响。我们通过首先对齐模型的参数空间来解决这一局限性，利用Transformer架构的固有排列、旋转和缩放对称性。我们将参数空间对齐适应现代分组查询注意力和SwiGLU层，探索基于权重和基于激活的方法。使用这种对齐优先策略，我们成功将高级推理技能转移到非推理模型。在具有挑战性的推理基准测试中，我们的方法始终优于标准任务算术。这项工作提供了一种有效的方法，用于在演进的LLM家族之间合并和转移专门技能，减少了冗余的微调，提高了模型适应性。


### 论文摘要

Task arithmetic is a powerful technique for transferring skills between Large Language Models (LLMs), but it often suffers from negative interference when models have diverged during training. We address this limitation by first aligning the models' parameter spaces, leveraging the inherent permutation, rotation, and scaling symmetries of Transformer architectures. We adapt parameter space alignment for modern Grouped-Query Attention (GQA) and SwiGLU layers, exploring both weight-based and activation-based approaches. Using this alignment-first strategy, we successfully transfer advanced reasoning skills to a non-reasoning model. Experiments on challenging reasoning benchmarks show that our method consistently outperforms standard task arithmetic. This work provides an effective approach for merging and transferring specialized skills across evolving LLM families, reducing redundant fine-tuning and enhancing model adaptability.

---

## 144. Towards Universal Neural Operators through Multiphysics Pretraining

**论文链接:** [http://arxiv.org/abs/2511.10829v1](http://arxiv.org/abs/2511.10829v1)

**作者:** Mikhail Masliaev, Dmitry Gusarov, Ilya Markov, Alexander Hvatov

**发布时间:** 2025-11-13

**备注:** 5 pages, 1 figure, accepted for Machine Learning and the Physical Sciences Workshop, NeurIPS 2025

### GPT解析

### 总结

本研究探讨了基于Transformer的神经算子在偏微分方程问题中的迁移学习能力，发现先进的神经算子架构能够有效地在不同PDE问题之间迁移知识。

### 背景

神经算子虽然在数据驱动的物理模拟中被广泛使用，但其训练计算成本高昂。最近的进展通过下游学习解决了这一问题，即在较简单问题上预训练的模型可以在更复杂问题上进行微调。

### 目的

调查基于Transformer的神经算子在更通用的迁移学习设置中的应用，这些算子之前仅被应用于特定问题。

### 方法

在多样化的偏微分方程问题上评估基于Transformer的神经算子性能，包括对未见参数的外推、新变量的纳入，以及从多方程数据集的迁移。

### 主要发现

先进的神经算子架构能够有效地在不同PDE问题之间迁移知识。

### 结论

基于Transformer的神经算子在通用迁移学习设置中表现良好，为降低神经算子训练的计算成本提供了新的途径。

### 翻译

虽然神经算子被广泛用于数据驱动的物理模拟，但其训练仍然计算成本高昂。最近的进展通过下游学习解决了这一问题，即在较简单问题上预训练的模型会在更复杂问题上进行微调。在本研究中，我们在更通用的迁移学习设置中调查了基于Transformer的神经算子，这些算子之前仅被应用于特定问题。我们在多样化的偏微分方程问题上评估了它们的性能，包括对未见参数的外推、新变量的纳入，以及从多方程数据集的迁移。我们的结果表明，先进的神经算子架构能够有效地在不同PDE问题之间迁移知识。


### 论文摘要

Although neural operators are widely used in data-driven physical simulations, their training remains computationally expensive. Recent advances address this issue via downstream learning, where a model pretrained on simpler problems is fine-tuned on more complex ones. In this research, we investigate transformer-based neural operators, which have previously been applied only to specific problems, in a more general transfer learning setting. We evaluate their performance across diverse PDE problems, including extrapolation to unseen parameters, incorporation of new variables, and transfer from multi-equation datasets. Our results demonstrate that advanced neural operator architectures can effectively transfer knowledge across PDE problems.

---

## 145. Excitonic Landscapes in Monolayer Lateral Heterostructures Revealed by Unsupervised Machine Learning

**论文链接:** [http://arxiv.org/abs/2511.10600v1](http://arxiv.org/abs/2511.10600v1)

**作者:** Maninder Kaur, Nicolas T. Sandino, Jason P. Terry, Mahdi Ghafariasl, Yohannes Abate

**发布时间:** 2025-11-13

### GPT解析

### 总结

该研究介绍了一种快速可扩展的无监督机器学习框架，用于分析二维异质结构的超光谱光致发光数据，提取与成分、应变和缺陷变化相关的定量和可解释信息。

### 背景

二维平面异质结构（包括成分梯度和具有定义界面的横向异质结构）展现出丰富的光电特性，为探索一维界面物理和多体相互作用效应提供了多功能平台。梯度Mo_xW_{1-x}S_2合金具有成分和应变的平滑空间变化，而MoS_2-WS_2横向异质结构包含支持一维激子现象的原子级锐利界面。超光谱成像技术能映射光学特征和局部变化，但手动解释大型数据集既缓慢又主观。

### 目的

引入一种快速且可扩展的无监督机器学习框架，从梯度Mo_xW_{1-x}S_2合金和MoS_2-WS_2异质结构的超光谱光致发光数据集中提取定量和可解释的信息。

### 方法

结合主成分分析（PCA）、t分布随机邻域嵌入（t-SNE）和基于密度的空间聚类（DBSCAN）方法，揭示与成分、应变和缺陷变化相关的光谱不同区域。

### 主要发现

1. 成分梯度的Mo_xW_{1-x}S_2合金显示出成分和应变的平滑空间变化，连续调节激子发射；2. MoS_2-WS_2横向异质结构包含支持一维激子现象的原子级锐利界面；3. 分解代表性光谱揭示了多种发射物种，包括带边激子和与缺陷相关的跃迁。

### 结论

机器学习驱动的分析为解释二维材料的丰富光学特性提供了一种稳健且自动化的途径，能够从大型超光谱数据集中提取定量和可解释的信息。

### 翻译

二维（2D）平面异质结构，包括成分梯度和具有定义界面的横向异质结构，展现出丰富的光电特性，并为探索一维界面物理和多体相互作用效应提供了多功能平台。梯度Mo_xW_{1-x}S_2合金显示出成分和应变的平滑空间变化，连续调节激子发射，而MoS_2-WS_2横向异质结构包含支持一维激子现象的原子级锐利界面。这些单层系统结合了可调的光学和电子特性，具有稳定、高性能光电器件的潜力。超光谱和纳米分辨光致发光成像能够映射光学特征以及成分、应变和缺陷的局部变化，但对这类大型数据集的手动解释既缓慢又主观。在此，我们引入了一种快速且可扩展的无监督机器学习框架，从梯度Mo_xW_{1-x}S_2合金和MoS_2-WS_2异质结构的超光谱光致发光数据集中提取定量和可解释的信息。结合主成分分析（PCA）、t分布随机邻域嵌入（t-SNE）和基于密度的空间聚类（DBSCAN），我们揭示了与成分、应变和缺陷变化相关的光谱不同区域。代表性光谱的分解揭示了多种发射物种，包括带边激子和与缺陷相关的跃迁，表明机器学习驱动的分析为解释二维材料的丰富光学特性提供了一种稳健且自动化的途径。


### 论文摘要

Two-dimensional (2D) in-plane heterostructures including compositionally graded alloys and lateral heterostructures with defined interfaces display rich optoelectronic properties and offer versatile platforms to explore one-dimensional interface physics and many-body interaction effects. Graded \(\mathrm{Mo}_x\mathrm{W}_{1-x}\mathrm{S}_2\) alloys show smooth spatial variations in composition and strain that continuously tune excitonic emission, while \(\mathrm{MoS}_2\)--\(\mathrm{WS}_2\) lateral heterostructures contain atomically sharp interfaces supporting one-dimensional excitonic phenomena. These single-layer systems combine tunable optical and electronic properties with potential for stable, high-performance optoelectronic devices. Hyperspectral and nano-resolved photoluminescence (PL) imaging enable spatial mapping of optical features along with local variations in composition, strain, and defects, but manual interpretation of such large datasets is slow and subjective. Here, we introduce a fast and scalable unsupervised machine-learning (ML) framework to extract quantitative and interpretable information from hyperspectral PL datasets of graded \(\mathrm{Mo}_x\mathrm{W}_{1-x}\mathrm{S}_2\) alloys and \(\mathrm{MoS}_2\)--\(\mathrm{WS}_2\) heterostructures. Combining principal-component analysis (PCA), t-distributed stochastic neighbor embedding (t-SNE), and density-based spatial clustering (DBSCAN), we uncover spectrally distinct domains associated with composition, strain, and defect variations. Decomposition of representative spectra reveals multiple emission species, including band-edge excitons and defect-related transitions, demonstrating that ML-driven analysis provides a robust and automated route to interpret rich optical properties of 2D materials.

---

## 146. From 2D to 3D Without Extra Baggage: Data-Efficient Cancer Detection in Digital Breast Tomosynthesis

**论文链接:** [http://arxiv.org/abs/2511.10597v1](http://arxiv.org/abs/2511.10597v1)

**作者:** Yen Nhi Truong Vu, Dan Guo, Sripad Joshi, Harshit Kumar, Jason Su, Thomas Paul Matthews

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出M&M-3D架构，解决了乳腺断层合成术(DBT)中深度学习模型因标注数据有限而发展受限的问题，实现了可学习的3D推理同时保持无参数特性，在乳腺癌检测任务中表现出色。

### 背景

DBT通过提供体积信息提高乳腺癌检测中病灶的可见性，减少重叠组织的影响，但有限标注数据限制了深度学习模型的发展。现有方法要么丢弃体积信息，要么引入需要更多DBT训练数据的复杂架构。

### 目的

设计一种能够实现可学习3D推理同时保持无参数特性的架构，解决DBT中数据稀缺问题，并有效利用FFDM模型的已有知识。

### 方法

M&M-3D构建恶性肿瘤引导的3D特征，通过将这些3D特征与切片级信息重复混合来学习3D推理。通过修改M&M中的操作实现，不增加参数，支持直接从FFDM转移权重。

### 主要发现

M&M-3D在定位方面比2D投影和3D切片方法高11-54%，分类方面高3-10%；在数据有限情况下，定位性能比复杂3D推理方法高20-47%，分类高2-10%；在BCS-DBT基准测试上，分类比顶级基线高4%，定位高10%。

### 结论

M&M-3D是一种有效架构，能够在不增加参数的情况下实现可学习的3D推理，直接从FFDM转移权重，在各种数据条件下均表现出色，为DBT中的乳腺癌检测提供了新思路。

### 翻译

数字乳腺断层合成术(DBT)通过提供体积信息提高乳腺癌检测中病灶的可见性，减少重叠组织的影响；然而，有限的标注数据限制了深度学习模型在DBT上的发展。为解决数据稀缺问题，现有方法试图通过将DBT体积扁平化或单独处理切片来重用2D全视野数字乳腺X线摄影(FFDM)模型，从而丢弃了体积信息。或者，3D推理方法引入了需要更多DBT训练数据的复杂架构。针对这些缺点，我们提出M&M-3D架构，使其能够实现可学习的3D推理，同时相对于其FFDM对应架构M&M保持无参数特性。M&M-3D构建恶性肿瘤引导的3D特征，并通过将这些3D特征与切片级信息重复混合来学习3D推理。这是通过修改M&M中的操作实现的，没有添加参数，从而能够直接从FFDM转移权重。大量实验表明，M&M-3D在定位方面比2D投影和3D切片方法高出11-54%，在分类方面高出3-10%。此外，在数据有限的情况下，M&M-3D在定位方面比复杂的3D推理变量高出20-47%，在分类方面高出2-10%，而在数据充足的情况下，其性能与这些变量相当。在流行的BCS-DBT基准测试上，M&M-3D在分类方面比之前的顶级基线高出4%，在定位方面高出10%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决数字乳腺断层合成（DBT）中的癌症检测面临的数据稀缺问题。DBT虽然通过提供体积信息提高了癌症检测可见性，但由于是较新技术，标注数据有限且标注成本高昂，这阻碍了高性能深度学习模型的发展。这个问题在现实中非常重要，因为乳腺癌是女性癌症相关死亡的主要原因，早期检测对患者预后至关重要，而DBT相比传统乳腺X线摄影能提供更好的检测效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：2D投影方法丢失深度信息，逐片处理方法丢弃体积信息，而复杂3D推理方法需要更多数据。设计目标是实现可学习的3D推理而不增加额外参数，以直接从FFDM预训练模型迁移权重。作者借鉴了Sparse R-CNN框架和M&M（高性能2D乳腺X线摄影检测器），利用了动态卷积和自注意力等组件。创新性地将2D提案重新解释为3D原始提案，引入切片级特征交互，并使用恶性驱动的加权平均进行特征融合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过恶性驱动的特征融合实现可学习的3D推理，同时不增加额外参数，以解决数据稀缺问题。整体流程包括：1)使用FFDM预训练的M&M模型初始化3D提案；2)通过6个级联头部处理，每个头部包含全局和多视图混合、切片级特征交互、切片到体积特征融合、框和分数细化以及z轴定位；3)使用多实例学习聚合图像和乳房分数；4)结合M&M损失函数和z轴定位损失进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)参数自由的3D推理，通过修改现有操作而非添加新模块实现；2)恶性驱动的3D特征构建，使用恶性分数作为权重进行特征融合；3)可学习的z轴定位，与放射科医生决策流程一致；4)特征交互机制，使3D特征与切片特征反复交互。相比2D投影方法，M&M-3D保留了3D信息，在R@0.25上提高0.20-0.39；相比逐片处理方法，避免了后处理复杂性，在R@0.25上提高约0.10；相比复杂3D推理方法，在低数据情况下表现更好，在10%数据时在R@0.25上提高0.10-0.20。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'M&M-3D提出了一种参数自由的3D推理框架，通过恶性驱动的特征融合实现数据高效的乳腺癌症检测，能够在不增加模型复杂度的情况下充分利用FFDM预训练知识，显著提升DBT癌症检测性能。'}


### 论文摘要

Digital Breast Tomosynthesis (DBT) enhances finding visibility for breast cancer detection by providing volumetric information that reduces the impact of overlapping tissues; however, limited annotated data has constrained the development of deep learning models for DBT. To address data scarcity, existing methods attempt to reuse 2D full-field digital mammography (FFDM) models by either flattening DBT volumes or processing slices individually, thus discarding volumetric information. Alternatively, 3D reasoning approaches introduce complex architectures that require more DBT training data. Tackling these drawbacks, we propose M&M-3D, an architecture that enables learnable 3D reasoning while remaining parameter-free relative to its FFDM counterpart, M&M. M&M-3D constructs malignancy-guided 3D features, and 3D reasoning is learned through repeatedly mixing these 3D features with slice-level information. This is achieved by modifying operations in M&M without adding parameters, thus enabling direct weight transfer from FFDM. Extensive experiments show that M&M-3D surpasses 2D projection and 3D slice-based methods by 11-54% for localization and 3-10% for classification. Additionally, M&M-3D outperforms complex 3D reasoning variants by 20-47% for localization and 2-10% for classification in the low-data regime, while matching their performance in high-data regime. On the popular BCS-DBT benchmark, M&M-3D outperforms previous top baseline by 4% for classification and 10% for localization.

---

## 147. Semi-Unified Sparse Dictionary Learning with Learnable Top-K LISTA and FISTA Encoders

**论文链接:** [http://arxiv.org/abs/2511.10575v1](http://arxiv.org/abs/2511.10575v1)

**作者:** Fengsheng Lin, Shengyi Yan, Trac Duy Tran

**发布时间:** 2025-11-13

### GPT解析

### 总结

提出了一种半统一稀疏字典学习框架，连接了经典稀疏模型和现代深度架构之间的差距。

### 背景

传统稀疏编码与现代深度学习架构之间存在差距，需要一种能够结合两者优势的方法。

### 目的

开发一种既保留传统稀疏编码可解释性，又能受益于高效可微训练的统一框架。

### 方法

将严格的Top-K LISTA及其基于FISTA的凸变体(LISTAConv)整合到判别性LC-KSVD2模型中，实现稀疏编码器和字典的共同演化，并为凸变体建立PALM风格的收敛分析。

### 主要发现

该方法在CIFAR-10上达到95.6%的准确率，在CIFAR-100上达到86.3%，在TinyImageNet上达到88.5%，具有更快的收敛速度和更低的内存成本（小于4GB GPU）。

### 结论

提出的LC-KSVD2 + LISTA/LISTAConv管道为现代深度架构提供了一种可解释且计算效率高的替代方案。

### 翻译

我们提出了一种半统一的稀疏字典学习框架，它弥合了经典稀疏模型与现代深度架构之间的差距。具体而言，该方法将严格的Top-K LISTA及其基于FISTA的凸变体(LISTAConv)整合到判别性LC-KSVD2模型中，使稀疏编码器和字典能够在监督或无监督模式下共同演化。这种统一设计保留了传统稀疏编码的可解释性，同时受益于高效、可微的训练。我们进一步为凸变体建立了PALM风格的收敛分析，确保在块交替下的理论稳定性。实验上，我们的方法在CIFAR-10上达到95.6%，在CIFAR-100上达到86.3%，在TinyImageNet上达到88.5%，具有更快的收敛速度和更低的内存成本（小于4GB GPU）。结果证实，所提出的LC-KSVD2 + LISTA/LISTAConv管道为现代深度架构提供了一种可解释且计算效率高的替代方案。


### 论文摘要

We present a semi-unified sparse dictionary learning framework that bridges the gap between classical sparse models and modern deep architectures. Specifically, the method integrates strict Top-$K$ LISTA and its convex FISTA-based variant (LISTAConv) into the discriminative LC-KSVD2 model, enabling co-evolution between the sparse encoder and the dictionary under supervised or unsupervised regimes. This unified design retains the interpretability of traditional sparse coding while benefiting from efficient, differentiable training.   We further establish a PALM-style convergence analysis for the convex variant, ensuring theoretical stability under block alternation. Experimentally, our method achieves 95.6\% on CIFAR-10, 86.3\% on CIFAR-100, and 88.5\% on TinyImageNet with faster convergence and lower memory cost ($<$4GB GPU). The results confirm that the proposed LC-KSVD2 + LISTA/LISTAConv pipeline offers an interpretable and computationally efficient alternative for modern deep architectures.

---

## 148. Don't Waste It: Guiding Generative Recommenders with Structured Human Priors via Multi-head Decoding

**论文链接:** [http://arxiv.org/abs/2511.10492v1](http://arxiv.org/abs/2511.10492v1)

**作者:** Yunkai Zhang, Qiang Zhang, Feng, Lin, Ruizhong Qiu, Hanchao Yu, Jason Liu, Yinglong Xia, Zhuoran Yu, Zeyu Zheng, Diji Yang

**发布时间:** 2025-11-13

### GPT解析

### 总结

论文提出了一种将人类先验知识直接集成到生成式推荐器端到端训练的框架，通过轻量级先验条件适配器头和分层组合策略，提高推荐系统的准确性和超越准确性的目标。

### 背景

工业界为推荐系统积累了大量结构化领域知识（人类先验知识），但这些知识通常通过后期调整应用，而非与核心模型学习结合。随着行业转向端到端生成式推荐基础模型，这种分离方法变得不理想。同时，许多针对超越准确性目标的方法需要特定架构修改并以无监督方式学习用户意图，丢弃了有价值的先验知识。

### 目的

开发一种框架，将人类先验知识直接集成到生成式推荐器的端到端训练中，提高准确性和超越准确性的目标（如多样性、新颖性和个性化）。

### 方法

提出一种与主干模型无关的框架，使用轻量级先验条件适配器头（受高效LLM解码策略启发）引导模型沿着人类可理解的轴解耦用户意图，并引入分层组合策略建模不同先验类型间的复杂交互。

### 主要发现

在三个大规模数据集上的实验表明，该方法显著提高了准确性和超越准确性的目标；人类先验知识使主干模型能够更有效地利用更长的上下文长度和更大的模型尺寸。

### 结论

通过直接集成人类先验知识到生成式推荐器的端到端训练中，可以改善推荐系统的性能，同时保留有价值的领域知识。

### 翻译

优化推荐系统以超越准确性为目标（如多样性、新颖性和个性化）对于长期用户满意度至关重要。为此，工业界实践者积累了大量结构化领域知识，我们称之为人类先验知识（例如，项目分类、时间模式）。这些知识通常通过在排序或后排序过程中的后期调整来应用。然而，这种方法与核心模型学习保持分离，随着行业转向端到端生成式推荐基础模型，这一点尤其不理想。另一方面，许多针对这些超越准确性目标的方法通常需要架构特定的修改，并通过完全无监督的方式学习用户意图来丢弃这些有价值的人类先验知识。我们没有丢弃多年实践中积累的人类先验知识，而是引入了一种与主干模型无关的框架，将这些人类先验知识无缝集成到生成式推荐器的端到端训练中。通过受高效LLM解码策略启发的轻量级先验条件适配器头，我们的方法引导模型沿着人类可理解的轴（如交互类型、长期与短期兴趣）来解耦用户意图。我们还引入了分层组合策略来建模不同先验类型之间的复杂交互。在三个大规模数据集上的广泛实验表明，我们的方法显著提高了准确性和超越准确性的目标。我们还表明，人类先验知识使主干模型能够更有效地利用更长的上下文长度和更大的模型尺寸。


### 论文摘要

Optimizing recommender systems for objectives beyond accuracy, such as diversity, novelty, and personalization, is crucial for long-term user satisfaction. To this end, industrial practitioners have accumulated vast amounts of structured domain knowledge, which we term human priors (e.g., item taxonomies, temporal patterns). This knowledge is typically applied through post-hoc adjustments during ranking or post-ranking. However, this approach remains decoupled from the core model learning, which is particularly undesirable as the industry shifts to end-to-end generative recommendation foundation models. On the other hand, many methods targeting these beyond-accuracy objectives often require architecture-specific modifications and discard these valuable human priors by learning user intent in a fully unsupervised manner.   Instead of discarding the human priors accumulated over years of practice, we introduce a backbone-agnostic framework that seamlessly integrates these human priors directly into the end-to-end training of generative recommenders. With lightweight, prior-conditioned adapter heads inspired by efficient LLM decoding strategies, our approach guides the model to disentangle user intent along human-understandable axes (e.g., interaction types, long- vs. short-term interests). We also introduce a hierarchical composition strategy for modeling complex interactions across different prior types. Extensive experiments on three large-scale datasets demonstrate that our method significantly enhances both accuracy and beyond-accuracy objectives. We also show that human priors allow the backbone model to more effectively leverage longer context lengths and larger model sizes.

---

## 149. destroR: Attacking Transfer Models with Obfuscous Examples to Discard Perplexity

**论文链接:** [http://arxiv.org/abs/2511.11309v1](http://arxiv.org/abs/2511.11309v1)

**作者:** Saadat Rafid Ahmed, Rubayet Shareen, Radoan Sharkar, Nazia Hossain, Mansur Mahi, Farig Yousuf Sadeque

**发布时间:** 2025-11-13

**备注:** 9 pages, 2 figures, 6 Table

### GPT解析

### 总结

该研究旨在分析和实验现有的最佳对抗攻击方法，并创建新的对抗攻击策略，特别是通过为当前最先进的机器学习模型生成模糊输入来开发新颖的对抗攻击方法。

### 背景

机器学习和神经网络近年来的进步导致了自然语言处理在各个领域的广泛应用，取得了显著成功。然而，研究表明机器学习模型可能存在多种脆弱性，使模型和它们所在的系统面临风险。

### 目的

分析和实验现有的最佳对抗攻击方法，并创建新的对抗攻击方法，特别是通过为当前最先进的机器学习模型生成模糊输入来开发新颖的对抗攻击策略。

### 方法

开发具有最大困惑度的对抗实例，利用机器学习和深度学习方法来欺骗模型。研究将分析多个数据集，专注于创建模糊的对抗示例，将模型置于困惑状态，并将孟加拉语纳入对抗攻击领域。

### 主要发现

通过产生模糊输入使模型困惑，可以有效地攻击当前最先进的机器学习模型。

### 结论

通过开发对抗实例和攻击策略，可以促进模型鲁棒性的未来发展，同时严格坚持效用使用减少和效率。

### 翻译

近年来机器学习和神经网络的进步导致了自然语言处理在各个领域的广泛实施，取得了显著成功，解决了各种复杂问题。然而，最近的研究表明，机器学习模型可能在多个方面存在脆弱性，使模型和它们所使用的系统面临风险。在本文中，我们打算分析和实验现有的最佳对抗攻击方法，并创建新的方法。我们专注于通过为当前最先进的机器学习模型产生模糊输入来开发一种新颖的对抗攻击策略，然后构建模型鲁棒性未来发展的路径。我们将利用机器学习和深度学习方法开发具有最大困惑度的对抗实例，以欺骗模型。在我们的攻击方法中，我们将分析多个数据集，并专注于创建模糊的对抗示例，使模型处于困惑状态，并将孟加拉语纳入对抗攻击领域。我们在整个工作中严格坚持效用使用减少和效率。


### 论文摘要

Advancements in Machine Learning & Neural Networks in recent years have led to widespread implementations of Natural Language Processing across a variety of fields with remarkable success, solving a wide range of complicated problems. However, recent research has shown that machine learning models may be vulnerable in a number of ways, putting both the models and the systems theyre used in at risk. In this paper, we intend to analyze and experiment with the best of existing adversarial attack recipes and create new ones. We concentrated on developing a novel adversarial attack strategy on current state-of-the-art machine learning models by producing ambiguous inputs for the models to confound them and then constructing the path to the future development of the robustness of the models. We will develop adversarial instances with maximum perplexity, utilizing machine learning and deep learning approaches in order to trick the models. In our attack recipe, we will analyze several datasets and focus on creating obfuscous adversary examples to put the models in a state of perplexity, and by including the Bangla Language in the field of adversarial attacks. We strictly uphold utility usage reduction and efficiency throughout our work.

---

## 150. Utilizing a Geospatial Foundation Model for Coastline Delineation in Small Sandy Islands

**论文链接:** [http://arxiv.org/abs/2511.10177v1](http://arxiv.org/abs/2511.10177v1)

**作者:** Tishya Chhabra, Manisha Bajpai, Walter Zesk, Skylar Tibbits

**发布时间:** 2025-11-13

**备注:** 8 pages, 7 figures

### GPT解析

### 总结

本研究评估了NASA和IBM的Prithvi-EO-2.0地理空间基础模型在利用卫星图像进行小沙岛海岸线划分方面的性能，展示了即使在极少训练数据的情况下也能取得高精度结果。

### 背景

海岸监测对于沿海地区管理至关重要，但在数据贫乏地区往往面临挑战。地理空间基础模型如Prithvi-EO-2.0可能为解决这一问题提供新途径。

### 目的

评估Prithvi-EO-2.0地理空间基础模型在小沙岛海岸线划分任务上的性能，探索其在数据贫乏地区海岸监测中的应用潜力。

### 方法

研究人员创建并标注了一个包含225张马尔代夫两个岛屿多光谱图像的数据集并公开发布。使用5到181张不等的训练子集对Prithvi-EO-2.0的3亿和6亿参数版本进行了微调，以测试模型在极少训练数据情况下的性能。

### 主要发现

实验表明，即使仅使用5张训练图像，模型也能实现高性能（F1分数0.94，IoU 0.79），证明了模型具有强大的迁移学习能力。

### 结论

Prithvi-EO-2.0地理空间基础模型展示了强大的迁移学习能力，此类模型有潜力支持数据贫乏地区的海岸监测工作。

### 翻译

我们展示了NASA和IBM的Prithvi-EO-2.0地理空间基础模型使用卫星图像对小沙岛海岸线进行划分的初步评估。我们整理并标注了一个包含两个马尔代夫岛屿的225张多光谱图像的数据集，我们公开发布此数据集，并且在5到181张图像不等的训练子集上对Prithvi的3亿和6亿参数版本进行了微调。我们的实验表明，即使只有5张训练图像，模型也能实现高性能（F1分数0.94，IoU 0.79）。我们的结果证明了Prithvi的强迁移学习能力，强调了此类模型在数据贫乏地区支持海岸监测的潜力。


### 论文摘要

We present an initial evaluation of NASA and IBM's Prithvi-EO-2.0 geospatial foundation model on shoreline delineation of small sandy islands using satellite images. We curated and labeled a dataset of 225 multispectral images of two Maldivian islands, which we publicly release, and fine-tuned both the 300M and 600M parameter versions of Prithvi on training subsets ranging from 5 to 181 images. Our experiments show that even with as few as 5 training images, the models achieve high performance (F1 of 0.94, IoU of 0.79). Our results demonstrate the strong transfer learning capability of Prithvi, underscoring the potential of such models to support coastal monitoring in data-poor regions.

---

## 151. Movement-Specific Analysis for FIM Score Classification Using Spatio-Temporal Deep Learning

**论文链接:** [http://arxiv.org/abs/2511.10713v1](http://arxiv.org/abs/2511.10713v1)

**作者:** Jun Masaki, Ariaki Higashi, Naoko Shinagawa, Kazuhiko Hirata, Yuichi Kurita, Akira Furui

**发布时间:** 2025-11-13

**备注:** 10 pages, 5 figures, 3tables, Accepted for the 2026 IEEE/SICE International Symposium on System Integration (SII 2026), January 11-14, 2026, Cancun, Mexico

### GPT解析

### 总结

研究提出了一种基于深度学习的自动化FIM分数估计方法，利用简单练习替代传统评估动作，减轻了评估负担，并在实际患者数据中显示出良好的区分能力和准确率。

### 背景

功能独立性测量（FIM）广泛用于评估患者在日常生活活动中的身体独立性，但传统FIM评估给患者和医疗专业人员带来了很大负担。

### 目的

提出一种自动化的FIM分数估计方法，利用不同于指定FIM评估动作的简单练习，以减轻传统评估方法的负担。

### 方法

使用深度神经网络架构，整合时空图卷积网络（ST-GCN）、双向长短期记忆（BiLSTM）和注意力机制来估计FIM运动项目分数，在277名康复患者的研究中进行了评估。

### 主要发现

该方法成功区分了完全独立的患者和需要帮助的患者，在不同FIM项目中实现了70.09-78.79%的平衡准确率，并揭示了可作为特定FIM评估项目可靠预测指标的特定运动模式。

### 结论

提出的自动化FIM分数估计方法有效，可以通过简单的练习和深度学习模型准确估计患者的FIM分数，减轻传统评估的负担。

### 翻译

功能独立性测量（FIM）广泛用于评估患者在日常生活活动中的身体独立性。然而，传统的FIM评估给患者和医疗专业人员带来了很大的负担。为了应对这一挑战，我们提出了一种自动化的FIM分数估计方法，利用不同于指定FIM评估动作的简单练习。我们的方法采用了一种深度神经网络架构，整合了时空图卷积网络（ST-GCN）、双向长短期记忆（BiLSTM）和注意力机制，以估计FIM运动项目分数。该模型有效捕捉长期时间依赖关系，并通过学习到的注意力权重识别关键身体关节的贡献。我们在一项针对277名康复患者的研究中评估了我们的方法，重点关注FIM转移和行走项目。我们的方法成功区分了完全独立的患者和需要帮助的患者，在不同的FIM项目中实现了70.09-78.79%的平衡准确率。此外，我们的分析揭示了特定运动模式，这些模式可作为特定FIM评估项目的可靠预测指标。


### 论文摘要

The functional independence measure (FIM) is widely used to evaluate patients' physical independence in activities of daily living. However, traditional FIM assessment imposes a significant burden on both patients and healthcare professionals. To address this challenge, we propose an automated FIM score estimation method that utilizes simple exercises different from the designated FIM assessment actions. Our approach employs a deep neural network architecture integrating a spatial-temporal graph convolutional network (ST-GCN), bidirectional long short-term memory (BiLSTM), and an attention mechanism to estimate FIM motor item scores. The model effectively captures long-term temporal dependencies and identifies key body-joint contributions through learned attention weights. We evaluated our method in a study of 277 rehabilitation patients, focusing on FIM transfer and locomotion items. Our approach successfully distinguishes between completely independent patients and those requiring assistance, achieving balanced accuracies of 70.09-78.79 % across different FIM items. Additionally, our analysis reveals specific movement patterns that serve as reliable predictors for particular FIM evaluation items.

---

## 152. Learning a Thousand Tasks in a Day

**论文链接:** [http://arxiv.org/abs/2511.10110v1](http://arxiv.org/abs/2511.10110v1)

**作者:** Kamil Dreczkowski, Pietro Vitiello, Vitalis Vosylius, Edward Johns

**发布时间:** 2025-11-13

**DOI:** 10.1126/scirobotics.adv7594

**备注:** This is the author's version of the work. It is posted here by permission of the AAAS for personal use, not for redistribution. The definitive version was published in Science Robotics on 12 November 2025, DOI: https://www.science.org/doi/10.1126/scirobotics.adv7594. Link to project website: https://www.robot-learning.uk/learning-1000-tasks

### GPT解析

### 总结

本研究提出了一种基于轨迹分解和检索的模仿学习方法MT3，实现了从极少数演示中学习大量日常任务，显著提高了学习效率。

### 背景

人类能够通过演示高效学习任务，但当前机器人操作的模仿学习方法通常需要大量演示数据（数百至数千次），限制了实际应用。

### 目的

研究如何通过操作轨迹分解和基于检索的泛化来提高模仿学习的效率，使机器人能够从少量演示中学习任务并泛化到新情境。

### 方法

通过3450次真实世界实验研究操作轨迹分解为对齐和交互阶段的效果；比较不同设计选择；开发多任务轨迹转移（MT3）方法，结合分解和检索技术；通过2200次额外实验评估方法性能。

### 主要发现

在每任务少于10次演示的情况下，分解方法比单阶段学习的数据效率提高一个数量级；检索方法在对齐和交互阶段都优于行为克隆；MT3可以从单次演示中学习日常任务并泛化到新对象实例；24小时内可教会机器人1000个不同任务。

### 结论

通过将操作轨迹分解为对齐和交互阶段，并采用基于检索的泛化方法，可以显著提高模仿学习的效率，使机器人能够从极少量演示中快速学习大量任务。

### 翻译

人类能够通过演示高效学习任务，但当今的机器人操作模仿学习方法通常需要每个任务数百或数千次演示。我们研究了提高学习效率的两个基本前提：将操作轨迹分解为顺序对齐和交互阶段，以及基于检索的泛化。通过3450次真实世界实验，我们系统研究了这种分解。我们比较了对齐和交互阶段的不同设计选择，并检查了相对于当前主流的单阶段整体行为克隆范式的泛化和扩展趋势。在每任务少量演示（<10次）的情况下，分解比单阶段学习的数据效率提高了一个数量级，检索在对齐和交互方面都优于行为克隆。基于这些见解，我们开发了多任务轨迹转移（MT3）方法，这是一种基于分解和检索的模仿学习方法。MT3可以从每个任务仅一次演示中学习日常操作任务，同时还能泛化到新的对象实例。这种效率使得我们能够在不到24小时的人类演示者时间内教会机器人1000个不同的日常任务。通过另外2200次真实世界实验，我们揭示了MT3在不同任务家族中的能力和局限性。实验视频可在https://www.robot-learning.uk/learning-1000-tasks查看。


### 论文摘要

Humans are remarkably efficient at learning tasks from demonstrations, but today's imitation learning methods for robot manipulation often require hundreds or thousands of demonstrations per task. We investigate two fundamental priors for improving learning efficiency: decomposing manipulation trajectories into sequential alignment and interaction phases, and retrieval-based generalisation. Through 3,450 real-world rollouts, we systematically study this decomposition. We compare different design choices for the alignment and interaction phases, and examine generalisation and scaling trends relative to today's dominant paradigm of behavioural cloning with a single-phase monolithic policy. In the few-demonstrations-per-task regime (<10 demonstrations), decomposition achieves an order of magnitude improvement in data efficiency over single-phase learning, with retrieval consistently outperforming behavioural cloning for both alignment and interaction. Building on these insights, we develop Multi-Task Trajectory Transfer (MT3), an imitation learning method based on decomposition and retrieval. MT3 learns everyday manipulation tasks from as little as a single demonstration each, whilst also generalising to novel object instances. This efficiency enables us to teach a robot 1,000 distinct everyday tasks in under 24 hours of human demonstrator time. Through 2,200 additional real-world rollouts, we reveal MT3's capabilities and limitations across different task families. Videos of our experiments can be found on at https://www.robot-learning.uk/learning-1000-tasks.

---

## 153. Limitations of Quantum Advantage in Unsupervised Machine Learning

**论文链接:** [http://arxiv.org/abs/2511.10709v1](http://arxiv.org/abs/2511.10709v1)

**作者:** Apoorva D. Patel

**发布时间:** 2025-11-13

**备注:** 4 pages,1 figure. Invited talk at the 2025 IEEE International Conference on Quantum Control, Computing and Learning (IEEE qCCL2025), Hong Kong, June 2025. Published in the proceedings, pp. 39-42

### GPT解析

### 总结

该论文探讨了机器学习模型在大数据分析中的应用，特别关注了量子模型相比经典模型可能具有的优势，以及这种优势的依赖条件和限制因素。

### 背景

机器学习模型用于大数据的模式识别分析，无需直接人工干预。无监督学习的任务是找到最能描述可用数据的概率分布，然后用于预测感兴趣的可观测值。经典模型通常将数据拟合到具有大量可调参数的哈密顿量玻尔兹曼分布，而量子扩展模型则用量子密度矩阵替代经典概率分布。

### 目的

研究量子模型相比经典模型可能具有的优势，探索何时以及为何量子模型可能提供优势，并分析这种优势的限制条件。

### 方法

将经典机器学习模型扩展到量子领域，使用量子密度矩阵替代经典概率分布，分析密度矩阵中经典概率分布所不具备的特征，并通过具体例子揭示限制量子优势的约束条件。

### 主要发现

只有当利用密度矩阵中经典概率分布所不具备的特征时，才能获得优势；这种优势情况取决于输入数据和目标可观测值；存在限制量子优势的约束条件；量子优势的程度与数据分析和传感应用有关。

### 结论

量子优势的程度是问题依赖的，这种依赖性对数据分析和传感应用有重要影响。

### 翻译

机器学习模型用于大数据的模式识别分析，无需直接人工干预。无监督学习的任务是找到最能描述可用数据的概率分布，然后使用它来预测感兴趣的可观测值。经典模型通常将数据拟合到具有大量可调参数的哈密顿量的玻尔兹曼分布。这些模型的量子扩展用量子密度矩阵替代经典概率分布。只有当利用密度矩阵中经典概率分布所不具备的特征时，才能获得优势。这种情况取决于输入数据以及目标可观测值。文中讨论了具体例子，揭示了限制可能量子优势的约束条件。量子优势的程度与问题有关，这对数据分析和传感应用都有影响。


### 论文摘要

Machine learning models are used for pattern recognition analysis of big data, without direct human intervention. The task of unsupervised learning is to find the probability distribution that would best describe the available data, and then use it to make predictions for observables of interest. Classical models generally fit the data to Boltzmann distribution of Hamiltonians with a large number of tunable parameters. Quantum extensions of these models replace classical probability distributions with quantum density matrices. An advantage can be obtained only when features of density matrices that are absent in classical probability distributions are exploited. Such situations depend on the input data as well as the targeted observables. Explicit examples are discussed that bring out the constraints limiting possible quantum advantage. The problem-dependent extent of quantum advantage has implications for both data analysis and sensing applications.

---

## 154. SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition

**论文链接:** [http://arxiv.org/abs/2511.10091v1](http://arxiv.org/abs/2511.10091v1)

**作者:** Qilang Ye, Yu Zhou, Lian He, Jie Zhang, Xuanming Guo, Jiayu Zhang, Mingkui Tan, Weicheng Xie, Yue Sun, Tao Tan, Xiaochen Yuan, Ghada Khoriba, Zitong Yu

**发布时间:** 2025-11-13

**备注:** Accepted by AAAI 2026 Main Track

### GPT解析

### 总结

本文提出了一种名为SUGAR的新方法，将大型语言模型(LLMs)与人骨骼数据结合用于动作分类和描述。该方法利用现有视频模型生成视觉和运动信息，监督骨骼学习获得离散表示，再通过LLM生成动作目标描述，并引入时间查询投影(TQP)模块处理长序列骨骼信号。

### 背景

大型语言模型(LLMs)拥有丰富的隐式知识和强大的可迁移性，但当将LLMs用作识别器时存在两个关键问题：1) LLMs如何理解骨骼数据？2) LLMs如何区分不同动作？

### 目的

探索将LLMs与人骨骼数据结合，实现动作分类和描述，解决LLMs理解骨骼数据和区分不同动作的问题。

### 方法

提出SUGAR(学习骨骼表示用于动作识别)新范式，流程包括：1)利用现成视频模型作为知识库生成视觉和运动信息；2)通过先验知识监督骨骼学习产生离散表示；3)使用未修改预训练权重的LLM理解这些表示并生成动作描述；4)提出时间查询投影(TQP)模块连续建模长序列骨骼信号。

### 主要发现

在多个基于骨骼的动作分类基准测试中，SUGAR方法表现出有效性；在零样本场景实验中，SUGAR比基于线性的方法更具通用性。

### 结论

SUGAR方法成功地将大型语言模型与人骨骼数据结合，解决了LLMs理解骨骼数据和区分不同动作的问题，并在动作分类和描述任务中表现出色，特别是在零样本场景下具有更好的通用性。

### 翻译

大型语言模型(LLMs)拥有丰富的隐式知识和强大的可迁移性。在本文中，我们探索将LLMs与人骨骼结合以执行动作分类和描述。然而，当将LLM用作识别器时，出现两个问题：1) LLMs如何理解骨骼？2) LLMs如何区分不同动作？为解决这些问题，我们引入了一种名为SUGAR(使用视觉运动知识学习骨骼表示用于动作识别)的新范式。在我们的流程中，我们首先利用现成的大规模视频模型作为知识库，生成与动作相关的视觉、运动信息。然后，我们提出通过这种先验知识监督骨骼学习，以产生离散表示。最后，我们使用未修改预训练权重的LLM来理解这些表示并生成期望的动作目标和描述。值得注意的是，我们提出了一个时间查询投影(TQP)模块，用于连续建模长序列的骨骼信号。在几个基于骨骼的动作分类基准上的实验证明了我们SUGAR的有效性。此外，在零样本场景的实验表明，SUGAR比基于线性的方法更具通用性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决两个问题：1)如何让大型语言模型(LLMs)理解骨骼数据；2)如何让LLMs区分相似动作(如喝水和吃零食)。这个问题很重要，因为骨骼数据作为人体行为的表示具有轻量化优势，适用于人机交互和智能监控，但日常生活中的许多相似动作(如喝水和吃零食)运动轨迹非常相似，导致传统识别器容易混淆它们，影响实际应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到LLMs具有丰富隐式知识和强大迁移能力，但面临理解骨骼和区分相似动作的挑战。他们设计了一个三步流程：1)利用预定义动作列表、GPT生成运动知识和GPT-4V生成视觉知识来构建文本；2)使用GCN作为骨骼编码器，通过对比学习使骨骼表示与文本对齐；3)使用TQP模块将骨骼表示映射到LLMs空间并微调LLMs。该方法借鉴了现有工作：利用现有视频模型作为知识库、借鉴CLIP预训练方法、使用LoRA微调LLMs、参考VQ-VAE学习离散token，以及利用Q-Former处理时间信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过引入视觉和运动知识学习更离散的骨骼表示，使识别器能够更好地区分相似动作，并利用LLMs的丰富先验知识作为识别器。整体流程分三步：1)文本构建：定义动作列表，用GPT生成运动知识(身体各部分动作描述)，用GPT-4V生成视觉知识(场景描述)；2)骨骼表示学习：使用GCN编码骨骼数据，CLIP编码文本，通过MIL对比学习使骨骼表示与文本对齐；3)动作识别：使用TQP模块将骨骼表示映射到LLMs空间，微调LLMs使其适应骨骼表示，最终实现动作分类和描述。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)引入视觉和运动知识学习更离散的骨骼表示，提升相似动作区分能力；2)提出SUGAR新范式，利用LLMs未预训练权重作为识别器，并设计TQP模块减少计算成本同时增强长期骨骼动态建模；3)在多个基准上实现最先进性能，尤其在零样本场景中表现出强泛化能力。相比之前工作：区别于传统仅基于骨骼的方法，引入了视觉和运动知识；相比现有LLMs方法，通过先验知识监督获得更好性能；相比线性方法，在零样本场景下具有更强泛化能力；TQP模块相比直接投影等方法能更好地处理时间信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SUGAR通过引入视觉和运动知识学习更离散的骨骼表示，并利用大型语言模型作为识别器，显著提升了基于骨骼的动作识别性能，特别是在区分相似动作和零样本场景下的泛化能力。'}


### 论文摘要

Large Language Models (LLMs) hold rich implicit knowledge and powerful transferability. In this paper, we explore the combination of LLMs with the human skeleton to perform action classification and description. However, when treating LLM as a recognizer, two questions arise: 1) How can LLMs understand skeleton? 2) How can LLMs distinguish among actions? To address these problems, we introduce a novel paradigm named learning Skeleton representation with visUal-motion knowledGe for Action Recognition (SUGAR). In our pipeline, we first utilize off-the-shelf large-scale video models as a knowledge base to generate visual, motion information related to actions. Then, we propose to supervise skeleton learning through this prior knowledge to yield discrete representations. Finally, we use the LLM with untouched pre-training weights to understand these representations and generate the desired action targets and descriptions. Notably, we present a Temporal Query Projection (TQP) module to continuously model the skeleton signals with long sequences. Experiments on several skeleton-based action classification benchmarks demonstrate the efficacy of our SUGAR. Moreover, experiments on zero-shot scenarios show that SUGAR is more versatile than linear-based methods.

---

## 155. 2.5D Transformer: An Efficient 3D Seismic Interpolation Method without Full 3D Training

**论文链接:** [http://arxiv.org/abs/2511.10033v1](http://arxiv.org/abs/2511.10033v1)

**作者:** Changxin Wei, Xintong Dong, Xinyang Wang

**发布时间:** 2025-11-13

### GPT解析

### 总结

该研究提出了一种2.5维Transformer网络(T-2.5D)，通过跨维度迁移学习策略，将二维Transformer编码器适应到三维地震数据插值任务中，显著降低了计算成本，同时保持了与全三维Transformer相当的性能。

### 背景

Transformer作为一种强大的深度学习技术，在二维地震数据插值中表现出色，得益于其全局建模能力。然而，其核心操作引入的二次方复杂度带来了沉重的计算负担，阻碍了其在更高维数据上的进一步应用。

### 目的

为了实现基于Transformer的三维地震数据插值，研究者提出了一个2.5维Transformer网络(T-2.5D)，采用跨维度迁移学习策略，使二维Transformer编码器能够适应三维地震数据。

### 方法

所提出的T-2.5D主要由二维Transformer编码器和三维地震维度适配器(SDAs)组成。每个三维SDA放置在Transformer编码器之前，用于学习地震线之间的空间相关性信息。提出的跨维度迁移学习策略包括两个阶段：二维预训练和三维微调。在第一阶段，使用大量二维数据块优化二维Transformer编码器；在第二阶段，冻结二维Transformer编码器，并使用有限的三维数据体积微调三维SDAs。

### 主要发现

在多个数据集上进行的大量实验评估了T-2.5D的有效性和效率。实验结果表明，所提出的方法以显著较低的成本实现了与全三维Transformer相当的性能。

### 结论

该研究提出的T-2.5D网络通过跨维度迁移学习策略，成功地将二维Transformer编码器适应到三维地震数据插值任务中，有效降低了计算复杂度，同时保持了良好的插值性能，为高维地震数据处理提供了高效解决方案。

### 翻译

Transformer已成为一种强大的二维地震数据插值深度学习技术，归功于其全局建模能力。然而，其核心操作由于二次方复杂度引入了沉重的计算负担，阻碍了其在更高维数据上的进一步应用。为了实现基于Transformer的三维地震数据插值，我们提出了一个2.5维Transformer网络(T-2.5D)，该网络采用跨维度迁移学习策略，使二维Transformer编码器能够适应三维地震数据。所提出的T-2.5D主要由二维Transformer编码器和三维地震维度适配器(SDAs)组成。每个三维SDA放置在Transformer编码器之前，用于学习地震线之间的空间相关性信息。提出的跨维度迁移学习策略包括两个阶段：二维预训练和三维微调。在第一阶段，我们使用大量二维数据块优化二维Transformer编码器。在第二阶段，我们冻结二维Transformer编码器，并使用有限的三维数据体积微调三维SDAs。在多个数据集上进行了大量实验，以评估T-2.5D的有效性和效率。实验结果表明，所提出的方法以显著较低的成本实现了与全三维Transformer相当的性能。


### 论文摘要

Transformer has emerged as a powerful deep-learning technique for two-dimensional (2D) seismic data interpolation, owing to its global modeling ability. However, its core operation introduces heavy computational burden due to the quadratic complexity, hindering its further application to higher-dimensional data. To achieve Transformer-based three-dimensional (3D) seismic interpolation, we propose a 2.5-dimensional Transformer network (T-2.5D) that adopts a cross-dimensional transfer learning (TL) strategy, so as to adapt the 2D Transformer encoders to 3D seismic data. The proposed T-2.5D is mainly composed of 2D Transformer encoders and 3D seismic dimension adapters (SDAs). Each 3D SDA is placed before a Transformer encoder to learn spatial correlation information across seismic lines. The proposed cross-dimensional TL strategy comprises two stages: 2D pre-training and 3D fine-tuning. In the first stage, we optimize the 2D Transformer encoders using a large amount of 2D data patches. In the second stage, we freeze the 2D Transformer encoders and fine-tune the 3D SDAs using limited 3D data volumes. Extensive experiments on multiple datasets are conducted to assess the effectiveness and efficiency of T-2.5D. Experimental results demonstrate that the proposed method achieves comparable performance to that of full 3D Transformer at a significantly low cost.

---

## 156. Probing the era of giant collisions: millimeter observations of the HD 166191 system

**论文链接:** [http://arxiv.org/abs/2511.11535v1](http://arxiv.org/abs/2511.11535v1)

**作者:** Kadin Worthen, Christine H. Chen, A. Meredith Hughes, Brandon C. Johnson, Isabel Rebollido, Diego E. Garcia, Jamar Kittling, Carey M. Lisse

**发布时间:** 2025-11-14

**备注:** Accepted for publication in ApJ

### GPT解析

### 总结

这项研究对HD 166191星盘进行了ALMA波段7和SMA的非同时观测，该星盘最近被认为在其类地行星区域发生了碰撞。研究人员检测了尘埃连续谱发射和12CO J=3-2谱线，但没有检测到SiO。毫米波长观测未发现变异性，CO和尘埃被限制在距离中心恒星20au范围内。星盘外区域可能富含气体，该星盘类似于已演化的原行星盘或过渡/混合盘。碰撞可能发生在星盘处于过渡阶段时，这使得HD 166191成为理解原行星盘和碎片盘之间过渡的重要天体。

### 背景

HD 166191星盘最近被认为在其类地行星区域发生了碰撞，其演化状态在文献中一直存在争议。该星盘可能处于从原行星盘向碎片盘过渡的阶段。

### 目的

研究HD 166191星盘的结构和演化状态，特别是探测其类地行星区域可能发生的碰撞事件，并了解该星盘在行星形成过程中的阶段。

### 方法

使用ALMA波段7和SMA对HD 166191星盘进行非同时观测，检测尘埃连续谱发射和12CO J=3-2谱线，对CO和连续谱可见度进行建模分析。

### 主要发现

检测到尘埃连续谱发射和12CO J=3-2谱线；未检测到SiO，但对总SiO质量设置了限制；与红外观测不同，毫米波长观测未发现变异性；CO和尘埃被限制在距离中心恒星20au范围内；星盘外区域可能富含气体；该星盘类似于已演化的原行星盘或过渡/混合盘；碰撞可能发生在星盘处于过渡阶段时，内部几au区域气体耗尽

### 结论

HD 166191是理解原行星盘和碎片盘之间过渡以及碰撞发生阶段的重要天体。其类地行星区域的碰撞可能发生在星盘处于过渡阶段时，这为研究行星形成和星盘演化提供了重要信息。

### 翻译

我们呈现了对HD 166191星盘的非同时ALMA波段7和SMA观测，该星盘最近被认为在其类地行星区域发生了一次碰撞。两次观测都检测到了尘埃连续谱发射，ALMA观测还探测到了环绕恒星的12CO J=3-2谱线。我们没有检测到SiO，这是巨大碰撞的一个潜在指标，但对系统中的总SiO质量设置了限制。与之前在红外波段观测到的不同，在比较2024年ALMA连续谱观测和2014年碰撞前的SMA观测时，我们没有发现毫米波长的变异性证据。我们对CO和连续谱可见度进行了建模，发现CO和尘埃都被略微空间分辨，并被限制在距离中心恒星20au范围内。对CO的建模表明，星盘的外部区域富含气体，尽管需要进一步观测来确认总气体质量。该系统的演化状态在文献中一直存在争议，我们的观测虽然不是决定性的，但总体上与该星盘类似于已演化的原行星盘或过渡/混合盘的观点一致。这可能表明，HD 166191在类地行星区域的碰撞发生在星盘处于过渡阶段时，此时内部几au区域气体耗尽。这使得HD 166191成为理解原行星盘和碎片盘之间过渡以及碰撞发生阶段的重要天体。


### 论文摘要

We present non-simultaneous ALMA band 7 and SMA observations of the HD 166191 disk, which was recently thought to have a collision in its terrestrial planet zone. Both observations detect dust continuum emission and the ALMA observations detect the 12CO J=3-2 line from the circumstellar disk. We do not detect SiO, a potential indicator of giant collisions, but place a limit on the total SiO mass in the system. Unlike previously observed in the infrared, we do not find evidence for variability at millimeter wavelengths when comparing the ALMA continuum observations from 2024 to the pre-collision SMA observations from 2014. We perform modeling of the CO and continuum visibilities and find that both the CO and dust are marginally spatially resolved and are contained to within 20 au from the central star. The modeling of the CO suggests that the outer regions of the disk are gas rich, although further observations are needed to confirm the total gas mass. The evolutionary state of this system has been debated in the literature, and our observations, while not definitive, are generally consistent with the idea that this disk is similar to an evolved protoplanetary or transition/hybrid disk. This could suggest that collisions in the terrestrial planet zone of HD 166191 are occurring while the disk is in a transitional phase, where the inner few au are depleted of gas. This makes HD 166191 an important object for understanding the transition between protoplanetary and debris disks and the stages at which collisions occur.

---

## 157. Bridging Hidden States in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.11526v1](http://arxiv.org/abs/2511.11526v1)

**作者:** Benjamin Fein-Ashley, Jacob Fein-Ashley

**发布时间:** 2025-11-14

### GPT解析

### 总结

BRIDGE是一种新型视觉-语言模型，通过在编码器顶部添加轻量级跨模态双向注意力层实现图像和文本的有效对齐，在多个基准测试中优于现有模型同时保持高效性。

### 背景

视觉-语言模型(VLMs)是一类对齐图像内容与自然语言的新型模型。现有方法通常采用早期融合(在编码器内混合特征)或晚期融合(比较池化嵌入)，且许多方法将融合与自回归解码器绑定。

### 目的

提出一种轻量级融合模块，直接对齐视觉和文本模态的隐藏状态，以匹配两种模态的内在结构。

### 方法

BRIDGE模型在编码器顶部附近添加几个仅跨模态、双向的注意力层。每层将视觉和文本编码器的隐藏状态序列投影到共享空间，进行跨模态关注，并通过门控残差更新返回结果，使用简单稳定器改善对齐。编码器保持非因果且强理解能力，同时通过可选解码器保持生成能力的清晰解耦。

### 主要发现

在标准检索、VQA和视觉推理基准测试中，BRIDGE性能优于可比的VLMs，同时保持了对比模型的双编码器效率。

### 结论

BRIDGE通过创新的轻量级融合架构实现了视觉和文本模态的有效对齐，在保持高效性的同时提升了模型性能。代码已公开在GitHub上。

### 翻译

视觉-语言模型(VLMs)是一类新型模型，用于对齐图像内容与自然语言。现有方法通常采用两种融合方式：(a)早期：在编码器内部混合token/特征，或(b)晚期：通过比较池化嵌入。许多方法还将融合与自回归解码器绑定。然而，视觉和文本模态的隐藏状态已经携带了丰富的模态特定结构(视觉中的空间布局；文本中的语法和语义)，因此直接对这些状态进行对齐是匹配两种模态'思考'方式的自然方式。我们提出了一种轻量级融合模块：在两个编码器顶部附近放置几个仅跨模态、双向的注意力层。每层将视觉和文本编码器的隐藏状态序列投影到共享空间，进行跨模态关注，并通过门控残差更新返回结果，使用简单的稳定器来改善对齐。编码器保持非因果且强理解能力，同时通过可选解码器保持生成能力的清晰解耦。在标准的检索、VQA和视觉推理基准测试中，BRIDGE性能优于可比的VLMs，同时保持了对比模型的双编码器效率。我们在https://github.com/jfeinashley/BRIDGE公开了代码。


### 论文摘要

Vision-Language Models (VLMs) are a new family of models that align image content with natural language. Existing approaches typically fuse either (a) early: by mixing tokens/features inside the encoders, or (b) late: by comparing pooled embeddings. Many methods also tie fusion to an autoregressive decoder. However, the hidden states of both modalities already carry rich, modality-specific structure (spatial layout in vision; syntax and semantics in text), so directly aligning these states is a natural way to match what the two modalities "think". We propose a lightweight fusion module: a few cross-only, bidirectional attention layers placed near the top of both encoders. Each layer projects the vision and text encoder hidden-state sequences into a shared space, attends across modalities, and sends gated residual updates back, with simple stabilizers to improve alignment. The encoders remain non-causal and strong for understanding, while generation stays cleanly decoupled via an optional decoder. Across standard retrieval, VQA, and visual reasoning benchmarks, BRIDGE outperforms comparable VLMs while preserving the bi-encoder efficiency of contrastive models. We make our code publicly available at https://github.com/jfeinashley/BRIDGE.

---

## 158. The Persistence of Cultural Memory: Investigating Multimodal Iconicity in Diffusion Models

**论文链接:** [http://arxiv.org/abs/2511.11435v1](http://arxiv.org/abs/2511.11435v1)

**作者:** Maria-Teresa De Rosa Palmini, Eva Cetinic

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究探讨了文本到图像扩散模型中泛化与记忆之间的模糊性，特别关注多模态标志性现象，即图像和文本唤起文化共享关联的情况。研究引入了一个评估框架，将识别文化引用与实现这些引用的方式分开，并通过评估五个扩散模型在767个文化引用上的表现，证明了该框架比现有方法更能有效区分复制与转变。研究发现模型即使在文本线索改变时也常再现标志性视觉结构，且文化对齐与训练数据频率、文本独特性、引用受欢迎程度和创建日期相关。研究认为扩散模型的价值在于它们如何转化和重新语境化文化知识。

### 背景

多模态标志性指的是图像和文本唤起文化共享关联的情况，例如标题唤起熟悉的艺术品或电影场景。先前关于记忆和遗忘的研究强调遗忘，而本研究关注被记住的内容及其方式，重点是识别文化引用与再现它们之间的平衡。

### 目的

引入一个评估框架，将识别（模型是否识别引用）与实现（如何通过复制或重新诠释来描绘）分开，并通过捕获这两个维度的措施进行量化，以更好地理解扩散模型在文化引用方面的表现。

### 方法

通过评估五个扩散模型在767个来自Wikidata的文化引用（涵盖静态和动态图像）上的表现，展示该框架比现有的基于相似性的方法更能有效区分复制与转变。为评估语言敏感性，使用同义词替换和字面图像描述进行提示扰动实验。

### 主要发现

模型即使在文本线索改变时也常常再现标志性的视觉结构；文化对齐不仅与训练数据频率相关，还与文本独特性、引用受欢迎程度和创建日期相关；扩散模型的价值不仅在于它们再现的内容，还在于它们如何转化和重新语境化文化知识。

### 结论

扩散模型的价值不仅在于它们再现的内容，还在于它们如何转化和重新语境化文化知识，将评估推进到超越简单文本图像匹配，向更丰富的语境理解发展。

### 翻译

我们的工作解决了文本到图像扩散模型中泛化与记忆之间的模糊性，特别关注我们称为多模态标志性的特定情况。这指的是图像和文本唤起文化共享关联的实例，例如当标题唤起熟悉的艺术品或电影场景时。虽然先前关于记忆和遗忘的研究强调遗忘，但我们研究被记住的内容及其方式，重点关注识别文化引用与再现它们之间的平衡。我们引入了一个评估框架，将识别（模型是否识别引用）与实现（如何通过复制或重新诠释来描绘）分开，通过捕获这两个维度的措施进行量化。通过评估五个扩散模型在767个来自Wikidata的文化引用上（涵盖静态和动态图像），我们表明我们的框架比现有的基于相似性的方法更能有效区分复制与转变。为评估语言敏感性，我们使用同义词替换和字面图像描述进行提示扰动实验，发现模型即使在文本线索改变时也常常再现标志性的视觉结构。最后，我们的分析显示文化对齐不仅与训练数据频率相关，还与文本独特性、引用受欢迎程度和创建日期相关。我们的工作揭示了扩散模型的价值不仅在于它们再现的内容，还在于它们如何转化和重新语境化文化知识，将评估推进到超越简单的文本图像匹配，向更丰富的语境理解发展。


### 论文摘要

Our work addresses the ambiguity between generalization and memorization in text-to-image diffusion models, focusing on a specific case we term multimodal iconicity. This refers to instances where images and texts evoke culturally shared associations, such as when a title recalls a familiar artwork or film scene. While prior research on memorization and unlearning emphasizes forgetting, we examine what is remembered and how, focusing on the balance between recognizing cultural references and reproducing them. We introduce an evaluation framework that separates recognition, whether a model identifies a reference, from realization, how it depicts it through replication or reinterpretation, quantified through measures capturing both dimensions. By evaluating five diffusion models across 767 Wikidata-derived cultural references spanning static and dynamic imagery, we show that our framework distinguishes replication from transformation more effectively than existing similarity-based methods. To assess linguistic sensitivity, we conduct prompt perturbation experiments using synonym substitutions and literal image descriptions, finding that models often reproduce iconic visual structures even when textual cues are altered. Finally, our analysis shows that cultural alignment correlates not only with training data frequency, but also textual uniqueness, reference popularity, and creation date. Our work reveals that the value of diffusion models lies not only in what they reproduce but in how they transform and recontextualize cultural knowledge, advancing evaluation beyond simple text-image matching toward richer contextual understanding.

---

## 159. When Genes Speak: A Semantic-Guided Framework for Spatially Resolved Transcriptomics Data Clustering

**论文链接:** [http://arxiv.org/abs/2511.11380v1](http://arxiv.org/abs/2511.11380v1)

**作者:** Jiangkai Long, Yanran Zhu, Chang Tang, Kun Sun, Yuanyuan Liu, Xuesong Yan

**发布时间:** 2025-11-14

**备注:** AAAI'2026 poster paper. 12 pages, 8 figures

### GPT解析

### 总结

SemST是一种语义引导的深度学习框架，用于空间转录组数据聚类，通过整合基因符号的生物学语义和空间信息，实现了最先进的聚类性能。

### 背景

空间转录组学能够在空间背景下进行基因表达谱分析，为组织微环境提供前所未有的见解。然而，大多数计算模型将基因视为孤立的数值特征，忽略了基因符号中编码的丰富生物学语义，这阻碍了对关键生物学特征的真正深入理解。

### 目的

克服现有模型的局限性，提出一种能够整合基因符号生物学语义和空间信息的聚类框架。

### 方法

SemST利用大型语言模型使基因能够通过其符号意义表达，将组织点中的基因集转换为具有生物学信息的嵌入；将这些嵌入与图神经网络捕获的空间邻域关系融合；引入细粒度语义调制模块学习点特定的仿射变换，使语义嵌入能够对空间特征执行逐元素校准，动态注入高阶生物学知识。

### 主要发现

在公共空间转录组数据集上的大量实验表明，SemST实现了最先进的聚类性能；FSM模块具有即插即用的多功能性，集成到其他基线方法时能持续提高性能。

### 结论

通过整合基因符号的生物学语义和空间信息，SemST框架有效改进了空间转录组数据的聚类分析，为理解组织微环境提供了新工具。

### 翻译

空间转录组学能够在空间背景下进行基因表达谱分析，为组织微环境提供前所未有的见解。然而，大多数计算模型将基因视为孤立的数值特征，忽略了基因符号中编码的丰富生物学语义。这阻碍了对关键生物学特征的真正深入理解。为克服这一局限，我们提出了SemST，一种用于空间转录组数据聚类的语义引导深度学习框架。SemST利用大型语言模型使基因能够通过其符号意义表达，将每个组织点中的基因集转换为具有生物学信息的嵌入。这些嵌入随后与图神经网络捕获的空间邻域关系融合，实现了生物功能和空间结构的协调整合。我们进一步引入了细粒度语义调制模块，以最佳方式利用这些生物学先验知识。该模块学习点特定的仿射变换，使语义嵌入能够对空间特征执行逐元素校准，从而将高阶生物学知识动态注入空间上下文。在公共空间转录组数据集上的大量实验表明，SemST实现了最先进的聚类性能。重要的是，FSM模块表现出即插即用的多功能性，集成到其他基线方法时能持续提高性能。


### 论文摘要

Spatial transcriptomics enables gene expression profiling with spatial context, offering unprecedented insights into the tissue microenvironment. However, most computational models treat genes as isolated numerical features, ignoring the rich biological semantics encoded in their symbols. This prevents a truly deep understanding of critical biological characteristics. To overcome this limitation, we present SemST, a semantic-guided deep learning framework for spatial transcriptomics data clustering. SemST leverages Large Language Models (LLMs) to enable genes to "speak" through their symbolic meanings, transforming gene sets within each tissue spot into biologically informed embeddings. These embeddings are then fused with the spatial neighborhood relationships captured by Graph Neural Networks (GNNs), achieving a coherent integration of biological function and spatial structure. We further introduce the Fine-grained Semantic Modulation (FSM) module to optimally exploit these biological priors. The FSM module learns spot-specific affine transformations that empower the semantic embeddings to perform an element-wise calibration of the spatial features, thus dynamically injecting high-order biological knowledge into the spatial context. Extensive experiments on public spatial transcriptomics datasets show that SemST achieves state-of-the-art clustering performance. Crucially, the FSM module exhibits plug-and-play versatility, consistently improving the performance when integrated into other baseline methods.

---

## 160. Discovering Meaningful Units with Visually Grounded Semantics from Image Captions

**论文链接:** [http://arxiv.org/abs/2511.11262v1](http://arxiv.org/abs/2511.11262v1)

**作者:** Melika Behjati, James Henderson

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一种将标题标记分组的模型架构，以捕获语言的细粒度表示，从而提高视觉语言模型对现实世界的细粒度理解能力。

### 背景

细粒度知识对视觉语言模型理解真实世界至关重要。现有工作主要关注将图像块与语言标记对齐，但图像块对人类无意义，单个标记也不一定包含图像中可验证的信息，实际上是标记组描述了场景的不同方面。

### 目的

提出一种能够将标题标记分组的模型，捕获语言的细粒度表示，使表示达到图像中存在的对象级别，并与发现对象的图像编码器输出对齐。

### 方法

设计一种模型架构，将标题标记分组作为其架构的一部分，学习将标记分组，使视觉语言模型对视觉和语言有更好的细粒度理解。

### 主要发现

通过学习将标记分组，视觉语言模型对视觉和语言有更好的细粒度理解。模型发现的标记组与文本中可验证的短语高度相似，无论在定性还是定量方面。

### 结论

通过将标记分组，可以更好地捕获语言的细粒度表示，提高模型对视觉和语言的细粒度理解能力，使模型能够更准确地理解和描述图像内容。

### 翻译

细粒度知识对于视觉语言模型获得对现实世界更好的理解至关重要。虽然已有工作试图在视觉和语言空间获取这类知识，但大多集中在将图像块与语言侧的标记对齐。然而，图像块对人类眼睛没有任何意义，单个标记也不一定包含图像中可验证的信息。实际上是标记组描述了场景的不同方面。在这项工作中，我们提出了一种模型，将标题标记分组作为其架构的一部分，以捕获语言的细粒度表示。我们期望我们的表示能够达到图像中存在的对象级别，因此将我们的表示与经过训练以发现对象的图像编码器的输出对齐。我们表明，通过学习将标记分组，视觉语言模型对视觉和语言有更好的细粒度理解。此外，我们模型发现的标记组与文本中可验证的短语高度相似，无论是在定性还是定量方面。


### 论文摘要

Fine-grained knowledge is crucial for vision-language models to obtain a better understanding of the real world. While there has been work trying to acquire this kind of knowledge in the space of vision and language, it has mostly focused on aligning the image patches with the tokens on the language side. However, image patches do not have any meaning to the human eye, and individual tokens do not necessarily carry groundable information in the image. It is groups of tokens which describe different aspects of the scene. In this work, we propose a model which groups the caption tokens as part of its architecture in order to capture a fine-grained representation of the language. We expect our representations to be at the level of objects present in the image, and therefore align our representations with the output of an image encoder trained to discover objects. We show that by learning to group the tokens, the vision-language model has a better fine-grained understanding of vision and language. In addition, the token groups that our model discovers are highly similar to groundable phrases in text, both qualitatively and quantitatively.

---

## 161. Beyond Flatlands: Unlocking Spatial Intelligence by Decoupling 3D Reasoning from Numerical Regression

**论文链接:** [http://arxiv.org/abs/2511.11239v1](http://arxiv.org/abs/2511.11239v1)

**作者:** Zhongbin Guo, Jiahe Liu, Yushan Li, Wenyu Gao, Zhen Yang, Chenzhi Li, Xinyue Zhang, Ping Jian

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究提出了一种名为GEODE的新型架构，用于解决现有视觉语言模型在3D空间智能理解方面的局限性。通过解耦3D推理和数值生成，GEODE引入了两个专门模块：解耦推理模块(DRM)和直接回归头(DRH)，使1.5B参数的模型能够达到媲美7B+模型的先进空间推理性能。

### 背景

现有的视觉语言模型(VLMs)在架构上根植于'平面'感知，难以理解现实世界的3D空间智能。这种失败源于双重瓶颈：输入阶段计算密集的几何感知编码器与仅使用2D特征的表面特征之间的冲突，以及输出阶段离散标记器无法产生精确连续数值的结构性不匹配。

### 目的

打破现有视觉语言模型在3D空间理解方面的双重瓶颈限制，开发一种能够有效处理3D空间智能的新型架构。

### 方法

作者提出了GEODE(几何输出和解耦输入引擎)架构，通过两个专门的可即插即用模块来解决双重瓶颈：1) 解耦推理模块(DRM)作为空间协处理器，通过交叉注意力将显式3D数据与2D视觉特征对齐，并将空间思维链(CoT)逻辑提炼为可注入的推理标记；2) 直接回归头(DRH)，采用'嵌入即值'范式，将专门的控制标记路由到轻量级MLP，用于精确连续地回归标量和3D边界框。

### 主要发现

GEODE架构使1.5B参数的模型能够作为高级语义调度器运行，实现了最先进的空间推理性能，可以媲美7B+参数模型的性能。

### 结论

通过GEODE架构的双重模块协同作用，成功解决了现有视觉语言模型在3D空间理解方面的双重瓶颈问题，使较小的模型也能实现较大的模型的性能，为3D空间智能理解提供了新的解决方案。

### 翻译

现有的视觉语言模型(VLMs)在架构上根植于'平面'感知，从根本上难以理解现实世界的3D空间智能。这种失败源于双重瓶颈：输入阶段计算密集的几何感知编码器与仅使用2D特征的表面特征之间的冲突，以及输出阶段离散标记器在结构上无法产生精确连续数值。为打破这一僵局，我们引入了GEODE(几何输出和解耦输入引擎)，一种通过解耦3D推理和数值生成来解决这一双重瓶颈的新架构。GEODE通过两个专门的可即插即用模块增强主VLM：解耦推理模块(DRM)作为空间协处理器，通过交叉注意力将显式3D数据与2D视觉特征对齐，并将空间思维链(CoT)逻辑提炼为可注入的推理标记；以及直接回归头(DRH)，一种'嵌入即值'范式，将专门的控制标记路由到轻量级MLP，用于精确连续地回归标量和3D边界框。这些模块的协同作用使我们的1.5B参数模型能够作为高级语义调度器运行，实现了与7B+模型相媲美的最先进空间推理性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文解决现有视觉语言模型（VLMs）无法理解和推理真实世界3D空间智能的根本缺陷。问题重要性在于：随着AI从2D图像扩展到3D物理世界，当前模型在深度估计、距离预测和路径规划等3D任务上表现不佳，限制了它们在自动驾驶、机器人等需要精确空间理解的应用场景中的部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者认识到现有VLMs的双重瓶颈：输入阶段高计算量3D编码器与浅层2D特征的冲突，以及输出阶段离散标记无法生成精确连续数值。他们设计了一种'解耦'思路，将3D推理与数值生成分离。借鉴了VGGT用于3D重建、Sonata作为点云编码器、空间思维链数据等现有工作，但创新性地整合为双重解耦架构，分别处理输入和输出瓶颈。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是'双重解耦'架构，将主VLM转变为参数高效的高层语义调度器。整体流程：1)DRM模块处理输入视频，生成3D点云并通过交叉注意力融合2D-3D特征，产出<Spatio>推理token；2)主VLM利用这些token进行空间推理，当需要数值输出时生成控制token；3)DRH模块拦截这些token，通过轻量MLP直接回归为连续数值；4)采用两阶段训练：先预训练DRM，再联合微调VLM和DRH。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点：1)双重解耦架构同时解决输入和输出瓶颈；2)DRM作为即插即用空间协处理器，通过交叉注意力实现2D-3D特征对齐；3)DRH的'嵌入即值'范式绕过离散标记化瓶颈。不同之处：相比纯2D VLMs，能处理显式3D几何；相比直接集成3D数据，避免高计算开销；相比现有3D VLMs，支持动态场景连续推理；相比传统数值处理方法，保持数值整体完整性。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GEODE通过双重解耦架构将3D推理与数值生成分离，使1.5B参数模型实现媲美7B+模型的先进3D空间推理能力，为需要精确空间理解的应用提供了高效解决方案。'}


### 论文摘要

Existing Vision Language Models (VLMs) architecturally rooted in "flatland" perception, fundamentally struggle to comprehend real-world 3D spatial intelligence. This failure stems from a dual-bottleneck: input-stage conflict between computationally exorbitant geometric-aware encoders and superficial 2D-only features, and output-stage misalignment where discrete tokenizers are structurally incapable of producing precise, continuous numerical values. To break this impasse, we introduce GEODE (Geometric-Output and Decoupled-Input Engine), a novel architecture that resolves this dual-bottleneck by decoupling 3D reasoning from numerical generation. GEODE augments main VLM with two specialized, plug-and-play modules: Decoupled Rationale Module (DRM) that acts as spatial co-processor, aligning explicit 3D data with 2D visual features via cross-attention and distilling spatial Chain-of-Thought (CoT) logic into injectable Rationale Tokens; and Direct Regression Head (DRH), an "Embedding-as-Value" paradigm which routes specialized control tokens to a lightweight MLP for precise, continuous regression of scalars and 3D bounding boxes. The synergy of these modules allows our 1.5B parameter model to function as a high-level semantic dispatcher, achieving state-of-the-art spatial reasoning performance that rivals 7B+ models.

---

## 162. GGBench: A Geometric Generative Reasoning Benchmark for Unified Multimodal Models

**论文链接:** [http://arxiv.org/abs/2511.11134v1](http://arxiv.org/abs/2511.11134v1)

**作者:** Jingxuan Wei, Caijun Jia, Xi Bai, Xinglong Xu, Siyuan Li, Linzhuang Sun, Bihui Yu, Conghui He, Lijun Wu, Cheng Tan

**发布时间:** 2025-11-14

**备注:** 35 pages, 22 figures

### GPT解析

### 总结

统一多模态模型(UMMs)代表了人工智能的范式转变，从被动感知转向主动的跨模态生成。尽管它们具有强大的信息合成能力，但在评估方面存在关键差距。现有基准测试无法衡量生成推理的集成认知过程。为此，研究者提出几何构建作为理想的测试平台，并引入GGBench基准测试来系统评估模型的几何生成推理能力。

### 背景

统一多模态模型(UMMs)的出现标志着人工智能领域的范式转变，使AI从被动感知转向主动的跨模态生成。这些模型具有前所未有的信息合成能力，但在评估方面存在明显不足。

### 目的

填补统一多模态模型评估中的关键差距，提出几何构建作为理想的测试平台，并开发专门的基准测试来评估模型的生成推理能力。

### 方法

引入GGBench，一个专门设计用于评估几何生成推理的基准测试。它提供了一个全面的框架，用于系统诊断模型的理解、推理和主动构建解决方案的能力。

### 主要发现

几何构建是测试语言理解和精确视觉生成融合的理想平台，而现有基准测试无法有效评估这种集成认知过程。

### 结论

GGBench为下一代智能系统设定了更严格的评估标准，能够全面评估模型的几何生成推理能力。

### 翻译

统一多模态模型(UMMs)的出现标志着人工智能的范式转变，从被动感知转向主动的跨模态生成。尽管它们具有前所未有的信息合成能力，但在评估方面仍存在一个关键差距：现有基准测试主要分别评估判别性理解或无约束图像生成，未能衡量生成推理的集成认知过程。为了填补这一差距，我们提出几何构建提供了理想的测试平台，因为它本质上需要语言理解和精确视觉生成的融合。我们引入了GGBench，一个专门设计用于评估几何生成推理的基准测试。它提供了一个全面的框架，用于系统诊断模型不仅理解和推理的能力，还包括主动构建解决方案的能力，从而为下一代智能系统设定了更严格的标准。项目网站：https://opendatalab-raiser.github.io/GGBench/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有评估基准无法全面衡量统一多模态模型的生成推理能力的问题。当前评估主要关注模型的判别式理解或无约束图像生成，无法测试模型在几何问题中从理解到生成的完整认知过程。这个问题很重要，因为几何问题解决需要结合语言理解、数学推理和精确视觉生成，反映了AI系统的高级认知能力，而缺乏这种评估限制了AI系统在需要精确空间推理的领域的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有评估基准的局限性，认识到需要一种能够同时评估理解、推理和生成能力的综合方法。他们借鉴了现有的数学推理基准（如GSM8K和MATH）和视觉数学推理基准（如MathVista）的发展思路，以及基于代码的评估方法（如MathCoder-VL）。在此基础上，作者设计了一个三模态数据结构，包含精确对齐的文本、代码和图像组件。设计过程包括问题收集、LLM辅助标记、提示设计、问题重构、解决方案生成、自动化筛选和专家验证等多个阶段，确保数据集质量和评估的全面性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：几何构造问题提供了评估集成理解-生成能力的理想测试平台，因为这类问题 inherently 要求模型解析抽象语言指令，制定基于几何原理的多步计划，并生成满足约束的精确图形。整体流程包括：1）收集和筛选几何问题；2）设计复合提示和重构问题；3）生成包含推理文本、可执行代码和渲染图像的解决方案；4）通过LLM质量控制和专家审核进行筛选，最终形成包含1411个高质量问题的数据集，每个问题包含3-7个图像和平均189.83个token的文本描述。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）首个专门评估几何生成推理能力的基准测试；2）三模态数据结构（文本、代码、图像精确对齐）；3）利用GeoGebra代码提供可验证的评估方法；4）系统性的数据构建管道；5）全面的四阶段评估协议（规划、中间过程、最终结果和总体评分）。相比之前工作，GGBench不仅关注判别式任务，而是要求模型构建满足约束的构造性解决方案；提供了100%的文本、代码和图像对齐，而之前基准通常缺乏代码监督；强制执行多步推理并将步骤与代码绑定，确保辅助对象和依赖关系是可操作的，而不仅仅是叙述性的。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了GGBench，首个专门设计用于评估统一多模态模型在几何生成推理中综合能力的基准测试，通过三模态对齐的数据结构和可验证的评估方法，填补了现有评估框架在衡量从理解到生成的完整推理过程方面的空白。'}


### 论文摘要

The advent of Unified Multimodal Models (UMMs) signals a paradigm shift in artificial intelligence, moving from passive perception to active, cross-modal generation. Despite their unprecedented ability to synthesize information, a critical gap persists in evaluation: existing benchmarks primarily assess discriminative understanding or unconstrained image generation separately, failing to measure the integrated cognitive process of generative reasoning. To bridge this gap, we propose that geometric construction provides an ideal testbed as it inherently demands a fusion of language comprehension and precise visual generation. We introduce GGBench, a benchmark designed specifically to evaluate geometric generative reasoning. It provides a comprehensive framework for systematically diagnosing a model's ability to not only understand and reason but to actively construct a solution, thereby setting a more rigorous standard for the next generation of intelligent systems. Project website: https://opendatalab-raiser.github.io/GGBench/.

---

## 163. A Space-Time Transformer for Precipitation Forecasting

**论文链接:** [http://arxiv.org/abs/2511.11090v1](http://arxiv.org/abs/2511.11090v1)

**作者:** Levi Harris, Tianlong Chen

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为SaTformer的视频transformer模型，基于完整时空注意力机制，能够从卫星辐射数据预测极端降水，并在NeurIPS Weather4Cast 2025累积降雨挑战中获得第一名。

### 背景

气象机构依赖实时洪水指导发布救命建议和警告。传统数值天气预报模型存在计算量大和临近预报性能下降等局限性，而AI天气预报方法虽然取得成功，但视频理解架构在天气预报中的应用仍不充分。

### 目的

解决传统天气预报方法的局限性，探索视频理解架构在天气预报中的应用，开发能够预测极端降水的新模型。

### 方法

提出SaTformer：一种基于完整时空注意力的视频transformer模型；将降水回归重新表述为分类问题；采用类别加权损失来解决降水数据中的标签不平衡问题。

### 主要发现

SaTformer模型在NeurIPS Weather4Cast 2025累积降雨挑战中获得第一名，证明了视频理解架构在天气预报中的有效性。

### 结论

AI方法特别是视频transformer架构在天气预报领域具有巨大潜力，能够有效处理极端降水预测问题，并解决数据不平衡挑战。

### 翻译

世界各地的气象机构依赖实时洪水指导来发布救命建议和警告。几十年来，传统数值天气预报(NWP)模型一直是降水预报的最先进技术。然而，物理参数化模型存在几个核心局限性：首先，求解偏微分方程来解决大气动力学计算量大；其次，这些方法在临近预报时间尺度（即0-4小时提前量）性能下降。受这些缺点的激励，最近的工作提出了AI天气预报(AI-WP)替代方案，用神经网络学习模拟分析数据。虽然这些数据驱动方法在不同时空分辨率上取得了巨大成功，但视频理解架构在天气预报中的应用仍然探索不足。为解决这些差距，我们提出了SaTformer：一种基于完整时空注意力的视频transformer，能够从卫星辐射数据巧妙预测极端降水。除了我们新颖的架构外，我们还引入了处理长尾降水数据集的技术。具体来说，我们将降水回归重新表述为分类问题，并采用类别加权损失来解决标签不平衡问题。我们的模型在NeurIPS Weather4Cast 2025累积降雨挑战中获得第一名。代码和模型权重可在以下网址获取：https://github.com/leharris3/satformer


### 论文摘要

Meteorological agencies around the world rely on real-time flood guidance to issue live-saving advisories and warnings. For decades traditional numerical weather prediction (NWP) models have been state-of-the-art for precipitation forecasting. However, physically-parameterized models suffer from a few core limitations: first, solving PDEs to resolve atmospheric dynamics is computationally demanding, and second, these methods degrade in performance at nowcasting timescales (i.e., 0-4 hour lead-times). Motivated by these shortcomings, recent work proposes AI-weather prediction (AI-WP) alternatives that learn to emulate analysis data with neural networks. While these data-driven approaches have enjoyed enormous success across diverse spatial and temporal resolutions, applications of video-understanding architectures for weather forecasting remain underexplored. To address these gaps, we propose SaTformer: a video transformer built on full space-time attention that skillfully forecasts extreme precipitation from satellite radiances. Along with our novel architecture, we introduce techniques to tame long-tailed precipitation datasets. Namely, we reformulate precipitation regression into a classification problem, and employ a class-weighted loss to address label imbalances. Our model scored first place on the NeurIPS Weather4Cast 2025 Cumulative Rainfall challenge. Code and model weights are available: https://github.com/leharris3/satformer

---

## 164. AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning

**论文链接:** [http://arxiv.org/abs/2511.11025v1](http://arxiv.org/abs/2511.11025v1)

**作者:** Jirong Zha, Yuxuan Fan, Tianyu Zhang, Geng Chen, Yingfeng Chen, Chen Gao, Xinlei Chen

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究引入了AirCopBench，第一个全面评估多模态大语言模型在具身空中协同感知中表现的基准，包含14.6k+个问题，覆盖四个任务维度和14种任务类型。

### 背景

多模态大语言模型在单智能体视觉任务中表现出色，但多智能体协同感知的评估基准稀缺。多无人机系统相比单传感器有优势，而现有基准无法评估MLLMs在复杂第一人称协同场景中的表现，特别是在感知条件下降的情况下。

### 目的

解决多智能体协同感知评估基准的缺乏问题，引入AirCopBench以评估MLLMs在具有挑战性感知条件下的具身空中协同感知能力。

### 方法

AirCopBench包含来自模拟器和真实世界数据的14.6k+个问题，涵盖场景理解、物体理解、感知评估和协同决策四个维度。使用具有挑战性的感知下降场景数据构建，通过模型、规则和人类方法在严格质量控制下生成问题，并在40个MLLMs上进行了评估。

### 主要发现

评估显示MLLMs在协同感知任务中存在显著性能差距，最佳模型平均落后人类24.38%，且在不同任务中表现不一致。微调实验证实了空中协同感知和推理中模拟到真实转移的可行性。

### 结论

AirCopBench为评估MLLMs在具身空中协同感知方面提供了全面基准，表明当前MLLMs在协同感知任务中有较大改进空间。

### 翻译

多模态大语言模型在单智能体视觉任务中显示出潜力，然而评估多智能体协同感知的基准仍然稀缺。这一差距很关键，因为与单传感器设置相比，多无人机系统提供更好的覆盖范围、鲁棒性和协作能力。现有的多图像基准主要针对使用高质量单智能体图像的基本感知任务，因此无法评估MLLMs在更复杂的第一人称协同场景中的表现，特别是在真实世界感知条件下降的情况下。为解决这些挑战，我们引入了AirCopBench，这是第一个全面评估MLLMs在具有挑战性感知条件下的具身空中协同感知的基准。AirCopBench包括来自模拟器和真实世界数据的14.6k+个问题，涵盖四个关键任务维度：场景理解、物体理解、感知评估和协同决策，共14种任务类型。我们使用具有挑战性感知下降场景的数据构建基准，标注协同事件，并在严格质量控制下通过基于模型、规则和人类的方法生成大规模问题。对40个MLLMs的评估显示协同感知任务中存在显著的性能差距，最佳模型平均落后人类24.38%，且在不同任务中表现出不一致的结果。微调实验进一步证实了在空中协同感知和推理中模拟到真实转移的可行性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决缺乏一个全面的基准测试来评估多无人机协作感知和推理能力的问题，特别是在具有挑战性感知退化条件下的表现。这个问题很重要，因为多无人机系统相比单传感器具有更强的覆盖范围、鲁棒性和协作能力，而现有基准测试未能评估MLLMs在复杂、以自我为中心的协作场景中的表现，特别是在真实世界感知退化条件下，这限制了多无人机系统在实际应用中的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有基准测试的局限性（过于简化的感知设置和缺乏具身推理能力）来设计方法。他们考虑了多无人机系统在实际应用中面临的挑战，如障碍物遮挡、传感器噪声等。作者借鉴了现有工作，包括使用Carla和AirSim等模拟器数据、MDMT真实世界数据，参考了Coperception-UAV和AeroCollab3D等现有协作感知基准，并利用了现有的标注方法和质量控制流程。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个全面的基准测试，评估多无人机在具有挑战性感知退化条件下的协作感知能力，包含多种感知退化类型和四个关键任务维度。整体实现流程包括：1)数据收集（模拟器数据、真实世界数据和衍生数据）；2)数据标注（事件级标注和物体级标注）；3)问题生成（基于模型、基于规则和基于人类的方法）；4)质量控制（标准检查、盲过滤和人工优化）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个全面的语义具身空中协作感知基准；2)包含14种任务类型跨越4个评估维度；3)包含多种感知退化类型；4)支持多种模态；5)覆盖多样化的目标类别；6)从第一视角的具身多无人机协作。相比之前的工作，AirCopBench考虑了更多真实世界感知退化类型，包含了协作决策任务，支持更复杂的协作推理，并且具有更多样化的数据源和标注方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AirCopBench首次提供了一个全面的基准测试，用于评估多模态大语言模型在具有挑战性感知退化条件下的多无人机协作感知和推理能力，揭示了当前模型在这一领域的不足，并为未来研究提供了方向。'}


### 论文摘要

Multimodal Large Language Models (MLLMs) have shown promise in single-agent vision tasks, yet benchmarks for evaluating multi-agent collaborative perception remain scarce. This gap is critical, as multi-drone systems provide enhanced coverage, robustness, and collaboration compared to single-sensor setups. Existing multi-image benchmarks mainly target basic perception tasks using high-quality single-agent images, thus failing to evaluate MLLMs in more complex, egocentric collaborative scenarios, especially under real-world degraded perception conditions.To address these challenges, we introduce AirCopBench, the first comprehensive benchmark designed to evaluate MLLMs in embodied aerial collaborative perception under challenging perceptual conditions. AirCopBench includes 14.6k+ questions derived from both simulator and real-world data, spanning four key task dimensions: Scene Understanding, Object Understanding, Perception Assessment, and Collaborative Decision, across 14 task types. We construct the benchmark using data from challenging degraded-perception scenarios with annotated collaborative events, generating large-scale questions through model-, rule-, and human-based methods under rigorous quality control. Evaluations on 40 MLLMs show significant performance gaps in collaborative perception tasks, with the best model trailing humans by 24.38% on average and exhibiting inconsistent results across tasks. Fine-tuning experiments further confirm the feasibility of sim-to-real transfer in aerial collaborative perception and reasoning.

---

## 165. EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation

**论文链接:** [http://arxiv.org/abs/2511.11002v1](http://arxiv.org/abs/2511.11002v1)

**作者:** Zongyang Qiu, Bingyuan Wang, Xingbei Chen, Yingqing He, Zeyu Wang

**发布时间:** 2025-11-14

**备注:** 15 pages, 12 figures. Accepted as an Oral presentation at AAAI 2026. For code and dataset, see https://zane-zyqiu.github.io/EmoVid

### GPT解析

### 总结

本研究介绍了EmoVid，首个专为创意媒体设计的多模态情感标注视频数据集，并开发了基于情感条件的视频生成技术，显著提升了视频生成质量。

### 背景

情感在视频表达中起关键作用，但现有视频生成系统主要关注低级视觉指标而忽视情感维度；视频社区缺乏连接情感理解与生成任务的专门资源，尤其针对风格化和非现实情境。

### 目的

解决视频生成系统中情感维度被忽视的问题，创建创意媒体专用情感标注数据集，并开发情感条件视频生成技术。

### 方法

1) 创建EmoVid数据集，包含卡通动画、电影片段和动画贴纸；2) 为每个视频标注情感标签、视觉属性和文本描述；3) 分析视觉特征与情感感知的时空模式；4) 基于分析结果微调Wan2.1模型开发情感条件视频生成技术。

### 主要发现

1) 发现了不同视频形式中视觉特征与情感感知的时空关联模式；2) 情感条件视频生成技术在定量指标和视觉质量上均有显著提升；3) EmoVid为情感视频计算建立了新基准。

### 结论

EmoVid为艺术风格视频中的视觉情感分析提供了见解，也为增强视频生成中的情感表达提供了实用方法，建立了情感视频计算的新基准。

### 翻译

情感在基于视频的表达中起着关键作用，但现有的视频生成系统主要关注低级视觉指标而忽视情感维度。尽管情感分析在视觉领域取得了进展，但视频社区缺乏专门资源来连接情感理解和生成任务，特别是针对风格化和非现实情境。为了解决这一差距，我们介绍了EmoVid，这是第一个专门为创意媒体设计的多模态、情感标注视频数据集，包括卡通动画、电影片段和动画贴纸。每个视频都标注了情感标签、视觉属性（亮度、色彩度、色调）和文本描述。通过系统分析，我们揭示了不同视频形式中视觉特征与情感感知之间的时空模式。基于这些见解，我们通过微调Wan2.1模型开发了一种情感条件的视频生成技术。结果表明，在文本到视频和图像到视频任务中，定量指标和生成视频的视觉质量都有显著提升。EmoVid为情感视频计算建立了新的基准。我们的工作不仅为艺术风格视频中的视觉情感分析提供了有价值的见解，也为增强视频生成中的情感表达提供了实用方法。


### 论文摘要

Emotion plays a pivotal role in video-based expression, but existing video generation systems predominantly focus on low-level visual metrics while neglecting affective dimensions. Although emotion analysis has made progress in the visual domain, the video community lacks dedicated resources to bridge emotion understanding with generative tasks, particularly for stylized and non-realistic contexts. To address this gap, we introduce EmoVid, the first multimodal, emotion-annotated video dataset specifically designed for creative media, which includes cartoon animations, movie clips, and animated stickers. Each video is annotated with emotion labels, visual attributes (brightness, colorfulness, hue), and text captions. Through systematic analysis, we uncover spatial and temporal patterns linking visual features to emotional perceptions across diverse video forms. Building on these insights, we develop an emotion-conditioned video generation technique by fine-tuning the Wan2.1 model. The results show a significant improvement in both quantitative metrics and the visual quality of generated videos for text-to-video and image-to-video tasks. EmoVid establishes a new benchmark for affective video computing. Our work not only offers valuable insights into visual emotion analysis in artistically styled videos, but also provides practical methods for enhancing emotional expression in video generation.

---

## 166. Influence of Prior Distributions on Gaussian Process Hyperparameter Inference

**论文链接:** [http://arxiv.org/abs/2511.10950v1](http://arxiv.org/abs/2511.10950v1)

**作者:** Ayumi Mutoh, Junoh Heo

**发布时间:** 2025-11-14

**备注:** 26 pages, 5 figures

### GPT解析

### 总结

本文研究了高斯过程模型中不同先验和提议分布对预测性能的影响，特别是在处理长度尺度参数时的效果。

### 背景

高斯过程被广泛用于近似昂贵的计算机仿真，特别是在工程设计和空间预测中。然而，当协方差参数估计不佳时，模型性能会显著下降。

### 目的

研究不同先验和提议分布对分层高斯模型中预测性能的影响，以更好地理解它们对预测精度和不确定性量化的影响。

### 方法

使用模拟和真实数据实验来评估不同类型的先验和提议分布对长度尺度参数θ的影响。

### 主要发现

摘要中未明确提及具体研究发现，仅说明了研究目的和方法。

### 结论

摘要中未明确提及结论，仅表明通过评估各种先验和提议分布，可以更好地理解它们对模型性能的影响。

### 翻译

高斯过程被广泛用作近似昂贵计算机仿真的元模型，特别是在工程设计和空间预测中。然而，当协方差参数估计不佳时，它们的性能会显著下降，突显了准确推断的重要性。最常见的方法是最大化边际似然，从而得到这些参数的点估计。然而，这种方法对初始化和优化设置非常敏感。另一种方法是采用完全贝叶斯分层框架，推断协方差参数的后验分布。这种方法提供了更稳健的不确定性量化，并减少了对参数选择的敏感性。然而，一个关键挑战在于仔细指定这些参数的先验分布。虽然许多可用的软件包提供默认先验，但它们对模型行为的影响往往未被充分探索。此外，提议分布的选择也会影响采样效率和收敛性。在本文中，我们研究了不同先验和提议分布对分层高斯模型中长度尺度参数θ预测性能的影响，使用模拟和真实数据实验。通过评估各种类型的先验和提议，我们旨在更好地了解它们对预测精度和不确定性量化的影响。


### 论文摘要

Gaussian processes (GPs) are widely used metamodels for approximating expensive computer simulations, particularly in engineering design and spatial prediction. However, their performance can deteriorate significantly when covariance parameters are poorly estimated, highlighting the importance of accurate inference. The most common approach involves maximizing the marginal likelihood, yielding point estimates of these parameters. However, this approach is highly sensitive to initialization and optimization settings. An alternative is to adopt a fully Bayesian hierarchical framework, where the posterior distribution over the covariance parameters is inferred. This approach provides more robust uncertainty quantification and reduces sensitivity to parameter selection. Yet, a key challenge lies in the careful specification of prior distributions for these parameters. While many available software packages provide default priors, their influence on model behavior is often underexplored. Additionally, the choice of proposal distributions can also influence sampling efficiency and convergence. In this paper, we examine how different prior and proposal distributions over the lengthscale parameters $θ$ affect predictive performance in a hierarchical GP model, using both simulated and real data experiments. By evaluating various types of priors and proposals, we aim to better understand their influence on predictive accuracy and uncertainty quantification.

---

## 167. Abstract 3D Perception for Spatial Intelligence in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.10946v1](http://arxiv.org/abs/2511.10946v1)

**作者:** Yifan Liu, Fangneng Zhan, Kaichen Zhou, Yilun Du, Paul Pu Liang, Hanspeter Pfister

**发布时间:** 2025-11-14

### GPT解析

### 总结

SandboxVLM框架通过抽象边界框编码几何结构和物理运动学，有效提升了视觉语言模型在3D相关任务中的表现，无需额外训练即可提高3D推理能力。

### 背景

视觉语言模型(VLMs)在处理空间认知和物理理解等3D相关任务时存在困难，这些任务对机器人和具身智能体等实际应用至关重要。

### 目的

弥合3D任务与VLM的2D训练之间的模态差距，提高视觉语言模型在3D任务中的表现。

### 方法

提出SandboxVLM框架，设计了一个包含四个阶段的3D沙盒重建和感知流程：生成具有抽象控制的多视图先验、代理高程、多视图投票和聚类、3D感知推理。

### 主要发现

在多个基准测试和VLM主干网络的零样本设置中评估，该方法在空间智能方面取得了一致改进，例如在SAT Real上比基线方法提高了8.3%。

### 结论

为VLM配备3D抽象可以显著提高其3D推理能力，无需额外训练，为通用具身智能提供了新的可能性。

### 翻译

视觉语言模型(VLMs)在处理空间认知和物理理解等3D相关任务时存在困难，这些任务对机器人和具身智能体等实际应用至关重要。我们认为这是由于3D任务与VLM的2D训练之间存在模态差距，导致从2D输入中检索3D信息效率低下。为了弥合这一差距，我们引入了SandboxVLM，一个简单而有效的框架，利用抽象边界框为VLM编码几何结构和物理运动学。具体来说，我们设计了一个包含四个阶段的3D沙盒重建和感知流程：生成具有抽象控制的多视图先验、代理高程、多视图投票和聚类、3D感知推理。在多个基准测试和VLM主干网络的零样本设置中评估，我们的方法在空间智能方面取得了一致改进，例如在SAT Real上比基线方法提高了8.3%。这些结果表明，为VLM配备3D抽象可以显著提高其3D推理能力，无需额外训练，为通用具身智能提供了新的可能性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视觉语言模型(VLMs)在3D相关任务上的困难，如空间认知和物理理解。这个问题很重要，因为这些能力对于机器人和具身智能等实际应用至关重要。当前VLMs本质上是二维的，将世界视为投影而非体积空间，缺乏对现实世界固有3D特性的理解，这限制了它们在需要真正空间理解的任务上的表现，如视点变化推理、相对位置估计或物体交互结果预测。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到人类如何感知和有效作用于3D空间的启发——人类不需要构建精确的度量模型，而是通过粗略的、关系性的理解来推理。他们观察到现有方法存在局限性：早期方法如3D-LLM依赖密集3D监督，而像MindJourney这样的模型仍基于2D表示。作者借鉴了视频扩散模型生成多视角先验、深度估计技术、2D分割模型和多视角投票聚类算法等现有工作，但创建了一个新颖的框架，通过抽象感知而非密集几何重建来解决3D推理问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是用紧凑的抽象3D边界框表示场景，编码空间排列和物理动力学，同时丢弃低级视觉细节，类似于人类的抽象感知方式。整体流程包括：1)多视角先验生成：使用视频扩散模型从单张2D图像生成多视角观察，并通过抽象控制聚焦相关视角；2)代理提升：识别任务相关对象，生成掩码，并使用深度估计将对象提升到3D空间；3)多视角投票和聚类：聚合来自多视角的3D代理点，通过跨视角一致性过滤不可靠点，并将剩余点聚类为定向3D边界框；4)3D感知推理：从信息丰富的视角渲染抽象3D框，将渲染图像与查询和原始输入图像组合，并反馈给VLM进行空间和物理推理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入抽象感知概念——一种受人类启发的3D推理方法，通过粗略结构线索而非精确几何重建；2)提出SandboxVLM，一个无需训练的框架，通过代理提升和边界框表示将符号3D结构注入现有VLM；3)证明这种抽象能显著提升跨基准和骨干模型的零样本空间推理能力。与之前工作的不同：不同于依赖密集3D监督的早期方法，SandboxVLM无需训练即可工作；不同于仍基于2D表示的模型，它提供真正的3D结构信息；不同于仅适用于开源模型的训练方法，它能利用专有VLM如GPT-5；它专注于抽象感知而非详细重建。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SandboxVLM通过一种轻量级的、无需训练的框架，利用符号3D边界框表示场景，在无需详细几何重建的情况下显著增强了视觉语言模型的3D空间推理能力。'}


### 论文摘要

Vision-language models (VLMs) struggle with 3D-related tasks such as spatial cognition and physical understanding, which are crucial for real-world applications like robotics and embodied agents. We attribute this to a modality gap between the 3D tasks and the 2D training of VLM, which led to inefficient retrieval of 3D information from 2D input. To bridge this gap, we introduce SandboxVLM, a simple yet effective framework that leverages abstract bounding boxes to encode geometric structure and physical kinematics for VLM. Specifically, we design a 3D Sandbox reconstruction and perception pipeline comprising four stages: generating multi-view priors with abstract control, proxy elevation, multi-view voting and clustering, and 3D-aware reasoning. Evaluated in zero-shot settings across multiple benchmarks and VLM backbones, our approach consistently improves spatial intelligence, achieving an 8.3\% gain on SAT Real compared with baseline methods for instance. These results demonstrate that equipping VLMs with a 3D abstraction substantially enhances their 3D reasoning ability without additional training, suggesting new possibilities for general-purpose embodied intelligence.

---

## 168. SURFACEBENCH: Can Self-Evolving LLMs Find the Equations of 3D Scientific Surfaces?

**论文链接:** [http://arxiv.org/abs/2511.10833v1](http://arxiv.org/abs/2511.10833v1)

**作者:** Sanchit Kabra, Shobhnik Kriplani, Parshin Shojaee, Chandan K. Reddy

**发布时间:** 2025-11-13

### GPT解析

### 总结

SurfaceBench是首个用于符号曲面发现的综合基准测试，包含183个任务，涵盖15个符号复杂性类别，旨在解决现有基准测试的局限性，并评估方程发现质量。

### 背景

从数据中发现方程式是机器学习在科学领域的核心挑战，需要恢复控制复杂物理和几何现象的简洁符号表达式。现有的大型语言模型方法在符号回归方面显示出潜力，但它们的成功往往依赖于记忆化的公式或过度简化的函数形式。现有的基准测试专注于标量函数，忽略了领域基础，并依赖于脆弱的基于字符串匹配的指标，无法捕捉科学等价性。

### 目的

介绍SurfaceBench，第一个用于符号曲面发现的综合基准测试，旨在解决现有基准测试的局限性，并评估方程发现质量。

### 方法

SurfaceBench包含183个任务，分布在15个符号复杂性类别中，涵盖显式、隐式和参数化方程表示形式。每个任务包括真实方程、变量语义和合成的三维数据采样。与之前的SR数据集不同，这些任务反映了曲面级结构，通过新颖的符号组合抵抗LLM记忆，并基于流体动力学、机器人学、电磁学和几何等科学领域。为了评估方程发现质量，将符号检查与几何感知指标（如Chamfer和Hausdorff距离）配对，同时捕捉代数保真度和空间重建准确性。

### 主要发现

实验表明，最先进的框架虽然在特定家族上偶尔成功，但在跨表示类型和曲面复杂性的泛化方面存在困难。

### 结论

SurfaceBench建立了一个具有挑战性和诊断性的测试平台，将符号推理与几何重建相结合，使对组合泛化、数据驱动的科学归纳以及与LLM的几何感知推理的进展进行有原则的基准测试成为可能。代码已发布在GitHub上。

### 翻译

从数据中发现方程式是机器学习在科学领域的核心挑战，需要恢复控制复杂物理和几何现象的简洁符号表达式。最近使用大型语言模型的方法在符号回归方面显示出潜力，但它们的成功往往依赖于记忆化的公式或过度简化的函数形式。现有的基准测试加剧了这一限制：它们专注于标量函数，忽略了领域基础，并依赖于脆弱的基于字符串匹配的指标，这些指标无法捕捉科学等价性。我们介绍了SurfaceBench，这是第一个用于符号曲面发现的综合基准测试。SurfaceBench包含183个任务，分布在15个符号复杂性类别中，涵盖了显式、隐式和参数化方程表示形式。每个任务包括真实方程、变量语义和合成的三维数据采样。与之前的SR数据集不同，我们的任务反映了曲面级结构，通过新颖的符号组合抵抗LLM记忆，并基于流体动力学、机器人学、电磁学和几何等科学领域。为了评估方程发现质量，我们将符号检查与几何感知指标（如Chamfer和Hausdorff距离）配对，同时捕捉代数保真度和空间重建准确性。我们的实验表明，最先进的框架虽然在特定家族上偶尔成功，但在跨表示类型和曲面复杂性的泛化方面存在困难。SurfaceBench因此建立了一个具有挑战性和诊断性的测试平台，将符号推理与几何重建相结合，使对组合泛化、数据驱动的科学归纳以及与LLM的几何感知推理的进展进行有原则的基准测试成为可能。我们在此发布代码：https://github.com/Sanchit-404/surfacebench

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从3D科学表面数据中发现其数学方程的问题。这个问题在科学发现中至关重要，因为它能帮助我们从实验数据中自动发现描述物理和几何现象的数学规律，减少对领域专家手动指定变量和函数形式的依赖，加速科学发现过程。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有符号回归基准的局限性，包括它们主要关注标量函数、容易受到LLM记忆影响、缺乏科学领域多样性等。他们借鉴了LLM-SRBench的符号增强理念，结合传统符号回归方法和几何处理技术，设计了一个包含183个任务、跨越15个科学类别的3D表面方程基准测试。通过引入几何感知评估指标，超越了传统的字符串匹配方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将符号回归从标量函数扩展到3D科学表面，挑战模型处理多变量耦合、潜在坐标系统和符号非唯一性的能力。整体流程包括：1)数据构建（选择科学领域、代表性方程、应用符号增强、验证方程、生成3D数据点）；2)模型评估（给模型提供3D表面数据，生成候选符号表达式，使用几何指标和符号准确性评估结果）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)SurfaceBench基准，首个全面的3D科学表面符号回归基准；2)几何感知评估方法，使用Chamfer和Hausdorff距离评估表面重建质量；3)多样化的挑战，通过符号增强防止模型记忆。相比之前工作，这个方法超越了标量函数限制，避免了字符串匹配的局限性，减少了LLM记忆的影响，并提供了更广泛的科学领域覆盖。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了SurfaceBench，首个全面的3D科学表面符号回归基准，通过几何感知评估方法揭示了当前方法在处理复杂表面方程时的局限性，为评估大语言模型在科学方程发现方面提供了新标准。'}


### 论文摘要

Equation discovery from data is a core challenge in machine learning for science, requiring the recovery of concise symbolic expressions that govern complex physical and geometric phenomena. Recent approaches with large language models (LLMs) show promise in symbolic regression, but their success often hinges on memorized formulas or overly simplified functional forms. Existing benchmarks exacerbate this limitation: they focus on scalar functions, ignore domain grounding, and rely on brittle string-matching based metrics that fail to capture scientific equivalence. We introduce SurfaceBench, first comprehensive benchmark for symbolic surface discovery. SurfaceBench comprises 183 tasks across 15 categories of symbolic complexity, spanning explicit, implicit, and parametric equation representation forms. Each task includes ground-truth equations, variable semantics, and synthetically sampled three dimensional data. Unlike prior SR datasets, our tasks reflect surface-level structure, resist LLM memorization through novel symbolic compositions, and are grounded in scientific domains such as fluid dynamics, robotics, electromagnetics, and geometry. To evaluate equation discovery quality, we pair symbolic checks with geometry-aware metrics such as Chamfer and Hausdorff distances, capturing both algebraic fidelity and spatial reconstruction accuracy. Our experiments reveal that state-of-the-art frameworks, while occasionally successful on specific families, struggle to generalize across representation types and surface complexities. SurfaceBench thus establishes a challenging and diagnostic testbed that bridges symbolic reasoning with geometric reconstruction, enabling principled benchmarking of progress in compositional generalization, data-driven scientific induction, and geometry-aware reasoning with LLMs. We release the code here: https://github.com/Sanchit-404/surfacebench

---

## 169. Optimal propagation distance for maximal biphoton entanglement through the weakly turbulent atmosphere

**论文链接:** [http://arxiv.org/abs/2511.10755v1](http://arxiv.org/abs/2511.10755v1)

**作者:** Luchang Niu, Saleem Iqbal, Yang Xu, Robert W. Boyd

**发布时间:** 2025-11-13

### GPT解析

### 总结

研究大气湍流对纠缠双光子态传播的影响，为自由空间量子通信提供理论支持

### 背景

理解大气湍流对纠缠双光子态传播的影响对自由空间量子通信协议至关重要

### 目的

推导通过湍流信道传播后的信号和闲置场联合密度算符的解析表达式，并研究纠缠特性在湍流中的变化

### 方法

使用扩展的惠更斯-菲涅尔原理和科尔莫戈罗夫湍流模型，分析通过独立湍流信道传播后的纠缠双光子态

### 主要发现

1) 尽管态纯度降低，信号和闲置子之间的空间相关性在湍流中仍然保持；2) 这些相关性从量子性质转变为经典性质；3) 存在一个有限的传播距离范围，在此范围内角度-轨道角动量纠缠最大化

### 结论

这些发现为设计通过湍流大气运行数公里的自由空间量子通信链路提供了有价值的见解

### 翻译

理解大气湍流对纠缠双光子态传播的影响对于自由空间量子通信协议至关重要。使用扩展的惠更斯-菲涅尔原理和科尔莫戈罗夫湍流模型，我们推导了通过独立湍流信道传播后，经由SPDC产生的信号和闲置场联合密度算符的解析表达式。通过将该密度算符表示在连续位置基中，我们展示了尽管失去了态的纯度，信号和闲置子之间的空间相关性如何在湍流中保持，以及它们如何从量子性质转变为经典性质。我们进一步确定了角度-轨道角动量纠缠最大化的有限传播距离范围，这为设计通过湍流大气运行数公里的自由空间量子通信链路提供了有价值的见解。


### 论文摘要

Understanding the influence of atmospheric turbulence on the propagation of entangled biphoton states is essential for free-space quantum communication protocols. Using the extended Huygens-Fresnel principle and the Kolmogorov turbulence model, we derive an analytical expression for the combined density operator of the signal and idler fields generated via SPDC, following propagation through separate turbulent channels. By expressing this density operator in the continuous position basis, we show how the spatial correlations between signal and idler persist through turbulence despite the loss of state purity, as they transition from being quantum to classical in nature. We further identify a finite range of propagation distances over which the angle-OAM entanglement is maximized, which provides valuable insights for designing free-space quantum communication links operating over several kilometers through the turbulent atmosphere.

---

## 170. Diffusion in the stochastic Klein-Gordon equation

**论文链接:** [http://arxiv.org/abs/2511.10738v1](http://arxiv.org/abs/2511.10738v1)

**作者:** Jonathan Oppenheim, Emanuele Panella

**发布时间:** 2025-11-13

**备注:** 26 pages + appendices

### GPT解析

### 总结

本文研究了随机克莱因-戈登方程，作为理解线性化经典-量子混合引力现象学的起点，并计算了标量场的非平衡两点函数。

### 背景

在度规本质上是经典的理论中，引力场存在随机波动，这属于经典-量子混合引力理论的研究范畴。

### 目的

研究随机克莱因-戈登方程作为理解线性化经典-量子混合引力现象学的起点，并探讨其对引力波背景的潜在影响。

### 方法

使用'模平方推迟'极点处方来计算标量场的非平衡两点函数，明确初始状态在调节发散中的作用。

### 主要发现

场的协方差只在光锥外非零，与时空点的空间距离成反比，随时间线性增长；能量存在类似量子情况中的接触发散。

### 结论

异常扩散对混合引力理论有潜在影响，特别是预测的引力波背景中的能量密度可以从标量协方差中推断出来。

### 翻译

在度规本质上是经典的理论中，引力场存在随机波动。本文研究随机克莱因-戈登方程作为理解线性化经典-量子混合引力现象学的起点。特别地，我们描述了如何计算标量场的非平衡两点函数，明确显示初始状态在调节发散中的作用。为此，我们使用'模平方推迟'极点处方，发现场的协方差只在光锥外非零，与时空点的空间距离成反比，并随时间线性增长。能量存在类似量子情况中的接触发散。最后，我们讨论了异常扩散对混合引力理论的潜在影响，特别是关注预测的引力波背景中的能量密度，这可以从标量协方差中推断出来。


### 论文摘要

Theories of gravity in which the metric is fundamentally classical predict stochastic fluctuations in the gravitational field. In this article, we study the stochastic Klein-Gordon equation as a starting point to understand the phenomenology of linearised classical-quantum hybrid gravity. In particular, we describe how to compute the non-equilibrium two point function of the scalar field, showing explicitly the role of the initial state in regulating divergences. To do so, we use a "mod-squared-retarded" pole-prescription and find that the covariance in the field is non-zero only outside the lightcone, scales inversely with the spatial distance of the spacetime points and grows linearly in time. The energy has a contact divergence similar to that found in the quantum case. We conclude by discussing possible implications of anomalous diffusion for hybrid theories of gravity, especially looking at the energy density in the predicted gravitational waves background, which can be inferred from the scalar covariances.

---

## 171. Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals

**论文链接:** [http://arxiv.org/abs/2511.10615v1](http://arxiv.org/abs/2511.10615v1)

**作者:** Shruti Singh Baghel, Yash Pratap Singh Rathore, Sushovan Jena, Anurag Pradhan, Amit Shukla, Arnav Bhavsar, Pawan Goyal

**发布时间:** 2025-11-13

**备注:** 8 pages

### GPT解析

### 总结

该研究评估了不同参数规模的SmolVLM2模型在视频描述生成中的表现，特别关注其对盲人和低视力(BLV)用户的可访问性影响，并开发了专门的评估框架来衡量模型在资源受限设备上的实际应用性能。

### 背景

大型视觉语言模型(VLMs)在理解和生成视频描述方面表现出色，但其高内存需求、计算复杂度和部署挑战限制了其实际应用，尤其对于依赖详细、上下文感知描述的盲人和低视力(BLV)用户。

### 目的

研究模型大小对以可访问性为重点的描述质量的影响，评估不同参数规模的模型在BLV用户辅助方面的表现，并探索在资源受限设备上的部署可能性。

### 方法

研究团队评估了500M和2.2B参数的SmolVLM2变体，在户外(AVCaps)和室内(Charades)两个多样化数据集上进行测试。研究引入了两种新的评估框架：多上下文BLV框架(评估空间方向、社交互动、动作事件和环境上下文)和导航辅助框架(专注于移动关键信息)。此外，还评估了四种不同的提示设计策略，并在智能手机上部署模型，测试FP32和INT8精度变体以评估实际性能限制。

### 主要发现

研究比较了不同参数规模模型在BLV可访问性描述方面的表现，评估了不同上下文信息对BLV用户的价值，并测试了模型在资源受限设备上的实际运行效果。

### 结论

研究为优化VLMs在BLV辅助应用中的性能提供了见解，证明了即使在资源受限的设备上，经过适当优化的模型也能为BLV用户提供有价值的视频描述服务。

### 翻译

大型视觉语言模型(VLMs)在理解和生成视频描述方面表现出色，但其高内存、计算和部署需求阻碍了实际应用，特别是对盲人和低视力(BLV)用户而言，他们依赖详细、上下文感知的描述。为研究模型大小对以可访问性为重点的描述质量的影响，我们在两个多样化数据集上评估了具有500M和2.2B参数的SmolVLM2变体：AVCaps(户外)和Charades(室内)。在这项工作中，我们引入了两种专门为BLV可访问性评估设计的新颖评估框架：多上下文BLV框架，评估空间方向、社交互动、动作事件和环境上下文；以及导航辅助框架，专注于移动关键信息。此外，我们还对四种不同的提示设计策略进行了系统性评估，并将两个模型部署在智能手机上，评估FP32和INT8精度变体，以评估资源有限的移动设备上的实际性能约束。


### 论文摘要

Large Vision-Language Models (VLMs) excel at understanding and generating video descriptions but their high memory, computation, and deployment demands hinder practical use particularly for blind and low-vision (BLV) users who depend on detailed, context-aware descriptions. To study the effect of model size on accessibility-focused description quality, we evaluate SmolVLM2 variants with 500M and 2.2B parameters across two diverse datasets: AVCaps (outdoor), and Charades (indoor). In this work, we introduce two novel evaluation frameworks specifically designed for BLV accessibility assessment: the Multi-Context BLV Framework evaluating spatial orientation, social interaction, action events, and ambience contexts; and the Navigational Assistance Framework focusing on mobility-critical information. Additionally, we conduct a systematic evaluation of four different prompt design strategies and deploy both models on a smartphone, evaluating FP32 and INT8 precision variants to assess real-world performance constraints on resource-limited mobile devices.

---

## 172. Stochastic Burgers Equation Driven by Hermite Sheet: Existence, Uniqueness, and Regularity Properties

**论文链接:** [http://arxiv.org/abs/2511.10463v1](http://arxiv.org/abs/2511.10463v1)

**作者:** Atef Lechiheb

**发布时间:** 2025-11-13

### GPT解析

### 总结

本研究分析了由Hermite sheet驱动的随机Burgers方程，建立了温和解的适定性理论，研究了解的正则性、自相似性等性质，并发展了相关的随机积分理论。

### 背景

随机偏微分方程在数学物理和工程中有广泛应用，特别是具有非高斯噪声的方程在实际问题中更为常见。Hermite sheet作为一种重要的非高斯随机过程，其驱动的SPDEs研究具有重要意义。

### 目的

建立由q阶Hermite sheet驱动的随机Burgers方程的温和解适定性理论，研究解的正则性、自相似性等性质，发展相关的随机积分理论，为分析具有非高斯噪声的非线性SPDEs提供完整框架。

### 方法

在适当的Banach空间中使用不动点理论，通过Picard迭代方案证明解的存在性和唯一性，结合矩估计和正则性分析研究解的性质。

### 主要发现

1. 在Hurst参数H的适当条件下，解存在且唯一；2. 解具有由Hurst参数决定的空间和时间Hölder正则性；3. 解继承了Hermite sheet的自相似性，具有明确的时空缩放指数；4. 导出了空间和时间上的一致矩估计。

### 结论

本研究发展了关于Hermite sheet的随机积分理论，建立了分析具有非高斯噪声的非线性SPDEs的完整框架，有助于理解具有长程依赖性和非高斯波动的随机系统。

### 翻译

我们分析了由q阶（q≥1）Hermite sheet驱动的随机Burgers方程，通过在适当的Banach空间中使用不动点理论建立了温和解的适定性。在Hurst参数H=(H₀,H₁,…,H_d)∈(1/2,1)^(d+1)的适当条件下，我们通过Picard迭代方案证明了解的存在性和唯一性。解表现出空间和时间Hölder正则性，其指数由驱动噪声的Hurst参数决定。此外，我们证明了解继承了Hermite sheet的自相似性，具有连接时间和空间维度的明确缩放指数。导出了空间和时间上的一致矩估计，为正则性分析提供了基础。本研究发展了关于Hermite sheet的随机积分理论，建立了分析具有非高斯噪声的非线性SPDEs的完整框架，为理解具有长程依赖性和非高斯波动的随机系统做出了贡献。


### 论文摘要

We analyze the stochastic Burgers' equation driven by a Hermite sheet of order \( q \geq 1 \), establishing well-posedness of mild solutions via fixed-point arguments in suitable Banach spaces. Under appropriate conditions on the Hurst parameters \( \mathbf{H} = (H_0, H_1, \dots, H_d) \in (1/2, 1)^{d+1} \), we prove existence and uniqueness of solutions through a Picard iteration scheme. The solution exhibits spatial and temporal Hölder regularity, with exponents determined by the Hurst parameters of the driving noise. Furthermore, we demonstrate that the solution inherits the self-similarity property from the Hermite sheet, with explicit scaling exponents connecting temporal and spatial dimensions. Moment estimates are derived uniformly in space and time, providing the foundation for the regularity analysis. The work develops stochastic integration theory with respect to Hermite sheets and establishes a complete framework for analyzing nonlinear SPDEs with non-Gaussian noise, contributing to the understanding of stochastic systems with long-range dependence and non-Gaussian fluctuations.

---

## 173. MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation

**论文链接:** [http://arxiv.org/abs/2511.10376v2](http://arxiv.org/abs/2511.10376v2)

**作者:** Xun Huang, Shijia Zhao, Yunxiang Wang, Xin Lu, Wanfa Zhang, Rongsheng Qu, Weixin Li, Yunhong Wang, Chenglu Wen

**发布时间:** 2025-11-13

**备注:** 10 pages

### GPT解析

### 总结

本文提出了一种多模态3D场景图（M3DSG）方法，用于解决具身导航中的零样本学习问题，保留了传统方法中丢失的视觉线索。

### 背景

具身导航是机器人代理的基本能力，实际部署需要开放词汇泛化和低训练开销，这促使使用零样本方法而非特定任务的强化学习训练。

### 目的

解决现有零样本方法构建显式3D场景图时，将丰富的视觉观测压缩为纯文本关系所带来的高构建成本、视觉证据的不可逆损失和有限词汇量问题。

### 方法

引入多模态3D场景图（Multi-modal 3D Scene Graph, M3DSG），通过替换文本关系来保留视觉线索。

### 主要发现

现有零样本方法构建的3D场景图存在高构建成本、视觉信息不可逆损失和词汇量受限等缺陷。

### 结论

M3DSG方法通过保留视觉线索，解决了现有方法的局限性，为具身导航提供了更好的解决方案。

### 翻译

具身导航是运行中的机器人代理的基本能力。实际部署需要开放词汇泛化和低训练开销，这促使使用零样本方法而非特定任务的强化学习训练。然而，构建显式3D场景图的现有零样本方法通常将丰富的视觉观测压缩为纯文本关系，导致高构建成本、视觉证据的不可逆损失和有限的词汇量。为解决这些限制，我们引入了多模态3D场景图（M3DSG），通过替换文本关系来保留视觉线索。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决零样本具身导航中的挑战，特别是传统3D场景图方法存在的三个问题：构建成本高、视觉信息不足和词汇受限。这个问题在现实中非常重要，因为具身导航是机器人在现实环境中执行任务的基本能力，真实世界部署需要开放词汇泛化和低训练开销，解决这些问题可以降低训练成本，提高对未见环境和目标的泛化能力，促进机器人在现实世界中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统3D场景图的局限性，指出其过度抽象为简单文本标签的问题。受3D-Mem工作的启发，强调原始图像对导航的价值，同时借鉴了SayPlan、OVSG等基于图的场景探索方法。作者设计了多模态3D场景图（M3DSG）用图像替换文本关系边，并构建了包含关键子图选择、自适应词汇更新、闭环推理和基于可见性的视点决策的MSGNav系统，以解决导航中的关键挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是用多模态3D场景图（M3DSG）替代传统纯文本场景图，保留视觉信息；通过基于可见性的视点决策解决'最后一英里'问题。整体流程包括：1) 增量构建多模态场景图，对象更新提取当前观测中的对象属性，边更新存储对象间的图像关系；2) 导航流程通过关键子图选择减少计算量，自适应词汇更新扩展词汇表，闭环推理提高准确性；3) 最后通过可见性评分选择最佳视点，确保成功接近目标。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 多模态3D场景图（M3DSG）用图像边替换文本边，保留视觉信息；2) 识别并解决'最后一英里'问题，提出基于可见性的视点决策模块；3) MSGNav系统包含关键子图选择、自适应词汇更新和闭环推理模块。相比之前的工作，M3DSG保留了丰富的视觉信息而非纯文本关系，消除了昂贵的MLLM查询需求；MSGNav不仅定位目标，还选择最佳视点；支持开放词汇泛化，不受预设词汇表限制；通过子图选择显著提高推理效率。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了MSGNav，一个基于多模态3D场景图的零样本具身导航系统，通过保留视觉信息、解决最后一英里问题和支持开放词汇泛化，显著提高了机器人在复杂环境中的导航能力。'}


### 论文摘要

Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relation

---

## 174. Probing the Liquid Solid Interfaces of 2D SnSe MXene Battery Anodes at the Nanoscale

**论文链接:** [http://arxiv.org/abs/2511.10278v1](http://arxiv.org/abs/2511.10278v1)

**作者:** Lukas Worch, Kavin Arunasalam, Neil Mulcahy, Syeda Ramin Jannat, James Douglas, Baptiste Gault, Valeria Nicolosi, Michele Shelly Conroy

**发布时间:** 2025-11-13

### GPT解析

### 总结

该研究揭示了锂离子电池中锡硒(SnSe)负极材料的降解机制，通过低温聚焦离子束和低温原子探针断层扫描技术发现了铜腐蚀和铜离子迁移现象，为设计更耐久的电池材料提供了关键见解。

### 背景

锂离子电池的降解过程对于提高长期性能和推进可持续能源技术至关重要。锡硒(SnSe)因其高理论容量而成为一种有前景的负极材料。

### 目的

理解锂离子电池中的降解过程，特别是锡硒负极材料在充放电过程中的行为和退化机制。

### 方法

将SnSe纳米颗粒嵌入Ti3C2Tx MXene框架中，并使用低温聚焦离子束(cryo FIB)切片和观察技术，以及低温原子探针断层扫描(cryo APT)对选定区域进行高空间和化学分辨率的分析。

### 主要发现

揭示了纳米级降解机制，包括相变、活性材料的部分溶解，以及首次直接证据表明铜腐蚀和铜离子从集流体迁移到电极中。铜的重新分布表明集流体降解直接导致复合电极中的化学污染和容量衰减。

### 结论

低温聚焦离子束和低温原子探针断层扫描共同为阐明反应性、束敏感系统中的电极降解提供了强大的工作流程，为设计更耐久、稳定的下一代电池材料提供了关键见解。

### 翻译

理解锂离子电池中的降解过程对于提高长期性能和推进可持续能源技术至关重要。锡硒(SnSe)因其锡的高理论容量而成为一种有前景的负极材料。与传统嵌入式电极不同，SnSe与锂发生转化和合金反应，形成Li4.4Sn、Sn和Li2Se，虽然能实现高锂存储，但会引起大的体积变化，导致机械不稳定性和容量衰减。将SnSe纳米颗粒嵌入Ti3C2Tx MXene框架中，通过提高导电性和结构韧性来缓解这些影响。研究表明，低温聚焦离子束切片和观察揭示了循环过程中的材料重新分布和形态变化，强调了位点特异性化学分析的必要性。低温原子探针断层扫描对选定区域的分析，在保持束敏感相的同时提供高空间和化学分辨率，揭示了纳米级降解机制。重要的是，首次发现了铜腐蚀和铜离子从集流体迁移到电极的直接证据，表明集流体降解直接导致复合电极中的化学污染和容量衰减。这些技术共同为阐明电极降解机制提供了强大的工作流程，为设计更耐久的电池材料提供了关键见解。


### 论文摘要

Understanding degradation processes in lithium ion batteries is essential for improving long term performance and advancing sustainable energy technologies. Tin selenide (SnSe) has emerged as a promising anode material due to the high theoretical capacity of tin. Unlike conventional intercalation based electrodes, SnSe undergoes conversion and alloying reactions with lithium to form Li4.4Sn, Sn, and Li2Se, enabling high lithium storage but inducing large volume changes that cause mechanical instability and capacity fading. Embedding SnSe nanoparticles within a Ti3C2Tx MXene framework offers a strategy to mitigate these effects by enhancing conductivity and structural resilience. Here, cryogenic focused ion beam (cryo FIB) slice and view revealed progressive material redistribution and morphological transformation during cycling, underscoring the need for site specific chemical analysis. Cryogenic atom probe tomography (cryo APT) of selected regions provided high spatial and chemical resolution while preserving beam sensitive phases, uncovering nanoscale degradation mechanisms including phase transformations, partial dissolution of active material, and, importantly, the first direct evidence of copper corrosion and copper ion migration from the current collector into the electrode. The observation of copper redistribution demonstrates that current collector degradation contributes directly to chemical contamination and capacity fading in composite electrodes. Together, cryo FIB and cryo APT provide a powerful workflow for elucidating electrode degradation in reactive, beam sensitive systems, offering critical insights for designing more durable and stable next generation battery materials.

---

## 175. TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding

**论文链接:** [http://arxiv.org/abs/2511.10241v1](http://arxiv.org/abs/2511.10241v1)

**作者:** Jinxuan Li, Yi Zhang, Jian-Fang Hu, Chaolei Tan, Tianming Liang, Beihao Xia

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出了TubeRMC框架，通过文本条件重建和相互约束机制解决时空视频定位任务中的目标识别和跟踪一致性问题。

### 背景

时空视频定位(STVG)旨在未修剪视频中定位与给定语言查询对应的时空管。这是一个涉及复杂视觉语言理解和时空推理的挑战性任务。现有弱监督方法通常采用简单晚期融合方式，生成的管独立于文本描述，导致识别和跟踪问题。

### 目的

解决现有方法中目标识别失败和目标跟踪不一致的问题，提高时空视频定位的准确性。

### 方法

提出TubeRMC框架，利用预训练视觉定位模型生成文本条件候选管，并通过时空约束进行优化。设计了时间、空间和时空三种重建策略，全面捕捉管-文本对应关系。每个策略配备管条件重建器，利用时空管作为条件重建查询关键线索。引入空间和时间提案间的相互约束提高重建质量。

### 主要发现

TubeRMC在VidSTG和HCSTVG两个公共基准数据集上优于现有方法。可视化显示该方法有效减轻了目标识别错误和不一致的跟踪问题。

### 结论

TubeRMC框架通过管条件重建和相互约束机制，显著提高了时空视频定位任务中目标识别和跟踪的一致性，特别是在弱监督设置下表现优异。

### 翻译

时空视频定位(STVG)旨在在未修剪视频中定位与给定语言查询对应的时空管。这是一个具有挑战性的任务，因为它涉及复杂的视觉语言理解和时空推理。最近的研究在STVG中探索了弱监督设置，以消除对边界框或时间戳等细粒度注释的依赖。然而，它们通常遵循简单的晚期融合方式，生成的管独立于文本描述，常常导致目标识别失败和目标跟踪不一致。为了解决这一限制，我们提出了一个管条件重建与相互约束(TubeRMC)框架，利用预训练的视觉定位模型生成文本条件候选管，并通过时空约束进一步优化它们。具体来说，我们从时间、空间和时空三个角度设计了三种重建策略，全面捕捉丰富的管-文本对应关系。每个策略都配备了一个管条件重建器，利用时空管作为条件来重建查询中的关键线索。我们还进一步引入了空间和时间提案之间的相互约束，以提高它们的质量。TubeRMC在两个公共基准VidSTG和HCSTVG上优于现有方法。进一步的可视化显示，TubeRMC有效减轻了目标识别错误和不一致的跟踪。


### 论文摘要

Spatio-Temporal Video Grounding (STVG) aims to localize a spatio-temporal tube that corresponds to a given language query in an untrimmed video. This is a challenging task since it involves complex vision-language understanding and spatiotemporal reasoning. Recent works have explored weakly-supervised setting in STVG to eliminate reliance on fine-grained annotations like bounding boxes or temporal stamps. However, they typically follow a simple late-fusion manner, which generates tubes independent of the text description, often resulting in failed target identification and inconsistent target tracking. To address this limitation, we propose a Tube-conditioned Reconstruction with Mutual Constraints (\textbf{TubeRMC}) framework that generates text-conditioned candidate tubes with pre-trained visual grounding models and further refine them via tube-conditioned reconstruction with spatio-temporal constraints. Specifically, we design three reconstruction strategies from temporal, spatial, and spatio-temporal perspectives to comprehensively capture rich tube-text correspondences. Each strategy is equipped with a Tube-conditioned Reconstructor, utilizing spatio-temporal tubes as condition to reconstruct the key clues in the query. We further introduce mutual constraints between spatial and temporal proposals to enhance their quality for reconstruction. TubeRMC outperforms existing methods on two public benchmarks VidSTG and HCSTVG. Further visualization shows that TubeRMC effectively mitigates both target identification errors and inconsistent tracking.

---

## 176. Diamond-based sensing of stray fields from the bulk of thin-film magnets via nano-indentation

**论文链接:** [http://arxiv.org/abs/2511.10176v1](http://arxiv.org/abs/2511.10176v1)

**作者:** Ming-Zhong Ai, Kang-Yuan Liu, Biao Zhang, Weng-Hang Leong, Yao Gao, Yue Cui, Guoli Zhu, Licong Peng, Yanglong Hou, Quan Li, Ren-Bao Liu

**发布时间:** 2025-11-13

**备注:** Comments are welcome

### GPT解析

### 总结

本研究开发了一种非破坏性方法，可直接测量薄膜磁体内部任意指定位置的 stray fields，具有纳米级空间分辨率。

### 背景

测量薄膜或二维材料内部的磁化强度对于理解其内在特性很重要，可以避免边缘或畴壁带来的复杂性。然而，材料内部的 stray fields 会消失或非常弱，这限制了直接测量方法的应用。

### 目的

开发一种非破坏性的方法来直接测量薄膜磁体内部任意指定位置的 stray fields，并具有纳米级空间分辨率。

### 方法

使用纳米压痕来诱导材料中 stray fields 的泄漏，并使用纳米金刚石磁强计来测量这些 stray fields。

### 主要发现

将该方法应用于铁薄膜，成功确定了材料内部的内在磁化强度。

### 结论

这项工作为直接获取薄膜和低维材料的内在磁性特性提供了途径，同时提供了一种研究纳米材料中机械效应对磁化影响的方法。

### 翻译

测量薄膜或二维材料内部的磁化强度对于理解其内在特性很重要，可以避免边缘或畴壁带来的复杂性。然而，材料内部的 stray fields 会消失或非常弱，这限制了直接测量方法的应用。在此，我们开发了一种非破坏性方法，可直接测量薄膜磁体内部任意指定位置的 stray fields，并具有纳米级空间分辨率。我们采用纳米压痕来诱导材料中 stray fields 的泄漏，并使用纳米金刚石磁强计进行测量。我们将该方法应用于铁薄膜，确定了材料内部的内在磁化强度。这项工作为直接获取薄膜和低维材料的内在磁性特性提供了途径，同时也提供了一种研究纳米材料中机械效应对磁化影响的方法。


### 论文摘要

Measurement of the magnetization in the bulk of thin-film or two-dimensional materials is important for understanding their intrinsic properties without the complications from edges or domain walls. However, the stray fields from the bulk vanish or are very weak, which limits the application of direct measurement methods. Here, we develop a non-destructive approach to directly measuring the stray fields from the bulk of thin-film magnets at arbitrarily designatable locations, with nanoscale spatial resolution. We employ nano-indentation to induce the leakage of stray fields from the materials and use nano-diamond magnetometers to measure them. We apply the method to iron thin films and determine the intrinsic magnetization in the bulk of the materials. This work provides direct access to the intrinsic magnetic properties of thin-film and low-dimensional materials, as well as a method to study the mechanical effects on magnetization in nanomaterials.

---

## 177. Impact of selection criteria on the structural parameters of the Galactic thin and thick discs

**论文链接:** [http://arxiv.org/abs/2511.10092v1](http://arxiv.org/abs/2511.10092v1)

**作者:** Simon Alinder, Thomas Bensby, Paul McMillan

**发布时间:** 2025-11-13

**备注:** Submitted to A&A. Abstract abridged

### GPT解析

### 总结

研究不同分类方法对银河系厚盘和薄盘结构特性确定的影响，发现基于化学或年龄特性的方法能提供更清晰的分离结果。

### 背景

银河系包含厚盘和薄盘，它们在化学、运动学、结构和空间特性上存在差异，尤其是在高金属丰度区域有显著重叠。区分这两个主要结构成分对于理解银河系的形成和演化至关重要。

### 目的

研究不同的恒星分类方法（将恒星分类为厚盘或薄盘恒星）如何影响两个盘状结构特性的确定。

### 方法

应用了五种不同的选择方法：两种使用化学元素平面切割的方法、一种使用动力学分离的方法、一种基于年龄切割的方法和一种运动学似然方法。使用APOGEE DR17的红巨星和astroNN的恒星年龄数据，推导出每个成分的相对密度分布并拟合到双指数盘模型中。

### 主要发现

基于丰度或年龄数据的方法产生最清晰的分离，运动学和动力学方法污染较高；薄盘标高随半径增加而明显增大，厚盘标高保持约1千秒差距；所有方法都发现薄盘的标长比厚盘长，化学选择方法中的差异最大；厚盘标长为2.0千秒差距时，薄盘标长在2.3至3.0千秒差距之间。

### 结论

不同分类方法会影响对银河系厚盘和薄盘结构特性的确定，基于化学或年龄特性的方法通常能提供更清晰的分离结果。

### 翻译

背景：银河系包含厚盘和薄盘，它们在化学、运动学、结构和空间特性上存在差异，特别是在高金属丰度区域有显著重叠。区分这些主要结构成分对于理解银河系的形成和演化至关重要。目的：我们研究将恒星分类为厚盘和薄盘种群的分类方法如何影响对两个盘状结构特性的确定。方法：我们应用了五种不同的选择方法。两种方法使用[α/Fe]-[Fe/H]和[Mg/Mn]-[Al/Fe]平面的切割；一种使用Jφ-JZ空间中的动力学分离；一种使用基于年龄的切割；最后一种使用运动学似然方法。对于每种方法，我们推导出每个成分相对于银道面高度和银心半径的相对密度分布，并将其拟合到简单的双指数盘模型中。我们使用来自APOGEE DR17的红巨星和来自astroNN的恒星年龄。结果：基于丰度或年龄数据的方法产生最清晰的分离，而运动学和动力学方法由于难以分离充分混合的种群而受到更高的污染。薄盘标高随半径增加而明显增大，而厚盘在所有方法中在大多数半径上都保持约1千秒差距的恒定值。所有方法都发现薄盘的标长比厚盘长，化学选择方法中的差异最大。厚盘的标长为2.0千秒差距时，薄盘的标长在2.3至3.0千秒差距之间。


### 论文摘要

Context: The Milky Way contains a thick and a thin disc that differ in chemical, kinematic, structural, and spatial properties. There is significant overlap in the distributions of these properties, especially so at higher metallicities. Distinguishing between these major structural components is crucial for understanding the formation and evolution of the Galaxy. Multiple selection methods exist to classify stars as thin or thick disc stars, each with its own advantages and limitations. Aims: We investigate how different classification methods for categorising stars into the thick and thin disc populations influence the determination of structural properties of the two discs. Methods: We apply five different selection methods. Two methods use cuts in the [$α$/Fe]-[Fe/H] and [Mg/Mn]-[Al/Fe] planes; one uses a dynamical separation in $J_φ$ -$J_Z$ space; one uses an age-based cut; and the last one uses a kinematic likelihood method. For each method, we derive relative density profiles of each component as functions of height above the Galactic plane and Galactocentric radius, and fit these to a simple two-exponential disc model. We use red giant stars from APOGEE DR17 and stellar ages from astroNN. Results: Methods based on abundance or age data produce the cleanest separations, while kinematic and dynamical methods suffer higher contamination due to difficulties in separating well-mixed populations. The thin disc scale heights show a clear flaring as they increase with radius, while the thick disc stays approximately constant at around 1 kpc over most radii for all methods. All methods find the thin disc to have a longer scale length than the thick disc, with the difference being greatest for the chemical selection methods. A scale length of the thick disc of 2.0 kpc leads to one of between 2.3 and 3.0 kpc for the thin disc.

---

## 178. Testbed Evaluation of AI-based Precoding in Distributed MIMO Systems

**论文链接:** [http://arxiv.org/abs/2511.11251v1](http://arxiv.org/abs/2511.11251v1)

**作者:** Tianzheng Miao, Thomas Feys, Gilles Callebaut, Jarne Van Mulders, Md Arifur Rahman, François Rottenberg

**发布时间:** 2025-11-14

**备注:** 6 pages, conference

### GPT解析

### 总结

本研究提出了一种在分布式MIMO测试平台上实现和验证基于AI的预编码器的框架，通过硬件互易校准和真实世界信道状态信息微调，实现了比预训练模型15.7%的性能提升，并在单用户场景中接近最大比传输性能。

### 背景

分布式MIMO是未来6G网络的关键架构，但现有研究多依赖理想化信道模型且缺乏硬件验证。AI驱动的预编码技术虽潜力巨大，但实际部署受限于数据收集和模型泛化挑战。

### 目的

开发一个框架，在具有硬件互易校准的D-MIMO测试平台上实现和验证基于AI的预编码器。

### 方法

使用预先训练的基于图神经网络的模型，利用从Techtile平台收集的真实世界信道状态信息进行微调，在插值和外推场景下评估，并进行端到端验证。

### 主要发现

多用户情况下微调后模型比预训练模型提升15.7%性能；单用户场景在未知位置实现接近最大比传输性能，每信道使用量下降小于0.7比特（总吞吐量5.19比特/信道使用）；真实世界测量数据效率高，训练样本增加带来持续增益；端到端验证确认了与最大比传输相当的相干功率聚焦能力。

### 结论

该工作成功实现了基于AI的预编码器在D-MIMO测试平台上的部署和验证，通过真实世界数据微调，模型在多用户和单用户场景下均表现出色，端到端验证确认了系统实际可行性。

### 翻译

分布式MIMO（D-MIMO）已成为未来第六代（6G）网络的关键架构，使接入点能够在空间上实现协作传输。然而，大多数现有研究依赖于理想化的信道模型，缺乏硬件验证，导致算法设计与实际部署之间存在差距。同时，人工智能驱动的预编码最新进展显示出学习非线性信道到预编码器映射的强大潜力，但由于数据收集和模型泛化的挑战，其现实世界的部署仍然有限。这项工作提出了一种框架，用于在具有硬件互易校准的D-MIMO测试平台上实现和验证基于AI的预编码器。使用从Techtile平台收集的真实世界信道状态信息（CSI）对预先训练的基于图神经网络（GNN）的模型进行微调，并在插值和外推场景下进行评估，然后进行端到端验证。实验结果表明，在多用户情况下，微调后比预训练模型实现了15.7%的性能增益，而在单用户场景中，模型在未知位置实现了接近最大比传输（MRT）的性能，每信道使用量下降小于0.7比特（总吞吐量为5.19比特/信道使用）。进一步分析确认了真实世界测量的数据效率，显示随着训练样本的增加，增益持续增加，端到端验证验证了与MRT相当的相干功率聚焦能力。


### 论文摘要

Distributed MIMO (D-MIMO) has emerged as a key architecture for future sixth-generation (6G) networks, enabling cooperative transmission across spatially distributed access points (APs). However, most existing studies rely on idealized channel models and lack hardware validation, leaving a gap between algorithmic design and practical deployment. Meanwhile, recent advances in artificial intelligence (AI)-driven precoding have shown strong potential for learning nonlinear channel-to-precoder mappings, but their real-world deployment remains limited due to challenges in data collection and model generalization. This work presents a framework for implementing and validating an AI-based precoder on a D-MIMO testbed with hardware reciprocity calibration. A pre-trained graph neural network (GNN)-based model is fine-tuned using real-world channel state information (CSI) collected from the Techtile platform and evaluated under both interpolation and extrapolation scenarios before end-to-end validation. Experimental results demonstrate a 15.7% performance gain over the pre-trained model in the multi-user case after fine-tuning, while in the single-user scenario the model achieves near-maximum ratio transmission (MRT) performance with less than 0.7 bits/channel use degradation out of a total throughput of 5.19 bits/channel use on unseen positions. Further analysis confirms the data efficiency of real-world measurements, showing consistent gains with increasing training samples, and end-to-end validation verifies coherent power focusing comparable to MRT.

---

## 179. Heterogeneous Attributed Graph Learning via Neighborhood-Aware Star Kernels

**论文链接:** [http://arxiv.org/abs/2511.11245v1](http://arxiv.org/abs/2511.11245v1)

**作者:** Hong Huang, Chengyu Yao, Haiming Chen, Hang Gao

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了邻域感知星核(NASK)，一种专为带属性图学习设计的新型图核方法，能够有效捕捉异构属性语义和邻域信息，在多个基准测试上超越现有方法。

### 背景

带属性图在社交网络、生物信息学和化学信息学等领域普遍存在，具有不规则拓扑结构和数值与分类属性的混合特性。现有图核方法难以同时捕捉异构属性语义和邻域信息。

### 目的

开发一种新型图核方法，能够同时捕捉带属性图中的异构属性语义和邻域信息，提升图相似度测量的效果。

### 方法

NASK利用Gower相似性系数的指数变换有效建模数值和分类特征，采用通过Weisfeiler-Leh曼迭代增强的星形子结构整合多尺度邻域结构信息，并理论证明了其正定性，确保与基于核的学习框架兼容。

### 主要发现

在十一个带属性图和四个大规模真实世界图基准上的实验表明，NASK在十六个最先进的基线方法(包括九个图核和七个图神经网络)上一致取得了优越的性能。

### 结论

NASK作为一种新型图核方法，能够有效解决现有方法在带属性图学习中的局限性，为图相似度测量提供了更强大的框架。

### 翻译

带属性图通常具有不规则拓扑结构和数值与分类属性的混合特性，在社交网络、生物信息学和化学信息学等不同领域中普遍存在。虽然图核为测量图相似性提供了原则性框架，但现有核方法往往难以同时捕捉带属性图中的异构属性语义和邻域信息。在这项工作中，我们提出了邻域感知星核(NASK)，一种专为带属性图学习设计的新型图核。NASK利用Gower相似性系数的指数变换来有效联合建模数值和分类特征，并采用通过Weisfeiler-Leh曼迭代增强的星形子结构来整合多尺度邻域结构信息。我们从理论上证明了NASK是正定的，确保其与基于核的学习框架(如SVM)兼容。在十一个带属性图和四个大规模真实世界图基准上进行了广泛实验。结果表明，NASK在包括九个图核和七个图神经网络在内的十六个最先进基线方法上一致取得了优越的性能。


### 论文摘要

Attributed graphs, typically characterized by irregular topologies and a mix of numerical and categorical attributes, are ubiquitous in diverse domains such as social networks, bioinformatics, and cheminformatics. While graph kernels provide a principled framework for measuring graph similarity, existing kernel methods often struggle to simultaneously capture heterogeneous attribute semantics and neighborhood information in attributed graphs. In this work, we propose the Neighborhood-Aware Star Kernel (NASK), a novel graph kernel designed for attributed graph learning. NASK leverages an exponential transformation of the Gower similarity coefficient to jointly model numerical and categorical features efficiently, and employs star substructures enhanced by Weisfeiler-Lehman iterations to integrate multi-scale neighborhood structural information. We theoretically prove that NASK is positive definite, ensuring compatibility with kernel-based learning frameworks such as SVMs. Extensive experiments are conducted on eleven attributed and four large-scale real-world graph benchmarks. The results demonstrate that NASK consistently achieves superior performance over sixteen state-of-the-art baselines, including nine graph kernels and seven Graph Neural Networks.

---

## 180. SMART: A Surrogate Model for Predicting Application Runtime in Dragonfly Systems

**论文链接:** [http://arxiv.org/abs/2511.11111v1](http://arxiv.org/abs/2511.11111v1)

**作者:** Xin Wang, Pietro Lodi Rizzini, Sourav Medya, Zhiling Lan

**发布时间:** 2025-11-14

**备注:** Accepted at AAAI 2026

### GPT解析

### 总结

该研究提出了一种结合图神经网络和大语言模型的代理模型\ourmodel，用于解决Dragonfly网络中的工作负载干扰问题，实现高效的应用程序运行时间预测和混合模拟。

### 背景

Dragonfly网络是一种高性能计算中的互连网络，具有高基数和低直径的结构。其主要挑战是共享网络链路上的工作负载干扰。高保真并行离散事件模拟(PDES)虽常用于分析工作负载干扰，但计算成本高，不适用于大规模或实时场景。

### 目的

开发一种混合模拟方法，结合数据驱动的代理模型，特别是用于预测应用程序运行时间，克服传统PDES方法的计算局限性。

### 方法

提出\ourmodel代理模型，结合图神经网络(GNNs)和大语言模型(LLMs)，从端口级别路由器数据中捕获空间和时间模式。

### 主要发现

\ourmodel优于现有的统计和机器学习基线方法，能够准确预测运行时间，并支持Dragonfly网络的高效混合模拟。

### 结论

结合图神经网络和大语言模型的代理模型是一种有前景的替代方法，可以有效解决Dragonfly网络中的工作负载干扰问题，为高性能计算提供更高效的性能评估工具。

### 翻译

Dragonfly网络凭借其高基数和低直径结构，成为高性能计算中的领先互连技术。一个主要挑战是共享网络链路上的工作负载干扰。并行离散事件模拟(PDES)常用于分析工作负载干扰，但高保真PDES计算成本高，使其不适用于大规模或实时场景。结合数据驱动代理模型的混合模拟提供了一种有前途的替代方案，特别是对于预测应用程序运行时间，而这一任务因网络流量的动态行为而变得复杂。我们提出\ourmodel，这是一种结合图神经网络(GNNs)和大语言模型(LLMs)的代理模型，可从端口级别路由器数据中捕获空间和时间模式。\ourmodel优于现有的统计和机器学习基线，能够准确预测运行时间，并支持Dragonfly网络的高效混合模拟。


### 论文摘要

The Dragonfly network, with its high-radix and low-diameter structure, is a leading interconnect in high-performance computing. A major challenge is workload interference on shared network links. Parallel discrete event simulation (PDES) is commonly used to analyze workload interference. However, high-fidelity PDES is computationally expensive, making it impractical for large-scale or real-time scenarios. Hybrid simulation that incorporates data-driven surrogate models offers a promising alternative, especially for forecasting application runtime, a task complicated by the dynamic behavior of network traffic. We present \ourmodel, a surrogate model that combines graph neural networks (GNNs) and large language models (LLMs) to capture both spatial and temporal patterns from port level router data. \ourmodel outperforms existing statistical and machine learning baselines, enabling accurate runtime prediction and supporting efficient hybrid simulation of Dragonfly networks.

---

## 181. Echoless Label-Based Pre-computation for Memory-Efficient Heterogeneous Graph Learning

**论文链接:** [http://arxiv.org/abs/2511.11081v1](http://arxiv.org/abs/2511.11081v1)

**作者:** Jun Hu, Shangheng Chen, Yufei He, Yuan Li, Bryan Hooi, Bingsheng He

**发布时间:** 2025-11-14

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种名为Echoless-LP的异构图神经网络方法，解决了预计算HGNNs中的训练标签泄露问题（回声效应），通过分区无回声传播(PFEP)技术，在不牺牲内存效率的情况下保持了与任何消息传递方法的兼容性。

### 背景

异构图神经网络(HGNNs)被广泛用于异构图的深度学习，但传统端到端HGNNs在训练过程中需要重复的消息传递，限制了在大规模真实图上的效率。预计算HGNNs虽通过预处理只进行一次消息传递来提高效率，但基于标签的预计算方法存在训练标签泄露问题，即节点自身的标签信息在多跳消息传递过程中传播回自身，形成回声效应。

### 目的

开发一种能够消除训练标签泄露（回声效应）的预计算方法，同时保持内存效率并兼容任何消息传递方法，以提高HGNNs在大规模图上的训练效率。

### 方法

提出Echoless-LP方法，核心是分区无回声传播(PFEP)，该方法将目标节点分区，每个分区中的节点仅从其他分区的邻居收集标签信息，从而避免回声效应。同时引入非对称分区方案(APS)和后调整(PostAdjust)机制来解决分区导致的信息损失和分布偏移问题。

### 主要发现

Echoless-LP在公共数据集上实现了优越的性能，同时与基线方法相比保持了内存效率，证明该方法有效解决了训练标签泄露问题且适用于大规模图。

### 结论

Echoless-LP成功消除了训练标签泄露问题，在大规模图上是内存高效的，并且与任何消息传递方法兼容，为异构图神经网络提供了一种高效且实用的解决方案。

### 翻译

异构图神经网络(HGNNs)被广泛用于异构图的深度学习。典型的端到端HGNNs在训练过程中需要重复的消息传递，限制了在大规模真实图上的效率。基于预计算的HGNNs通过在预处理过程中只执行一次消息传递来解决这个问题，将邻居信息收集成规则形状的张量，从而实现高效的mini-batch训练。基于标签的预计算方法收集邻居的标签信息，但存在训练标签泄露问题，即节点自身的标签信息在多跳消息传递过程中传播回自身——回声效应。现有的缓解策略在大图上内存效率低下，或与高级消息传递方法存在兼容性问题。我们提出了无回声标签预计算(Echoless-LP)，通过分区无回声传播(PFEP)消除训练标签泄露。PFEP将目标节点分区并执行无回声传播，每个分区中的节点仅从其他分区的邻居收集标签信息，避免回声效应的同时保持内存效率，并兼容任何消息传递方法。我们还引入了非对称分区方案(APS)和后调整(PostAdjust)机制，以解决分区导致的信息损失和分区间的分布偏移问题。公共数据集上的实验表明，与基线方法相比，Echoless-LP实现了优越的性能并保持了内存效率。


### 论文摘要

Heterogeneous Graph Neural Networks (HGNNs) are widely used for deep learning on heterogeneous graphs. Typical end-to-end HGNNs require repetitive message passing during training, limiting efficiency for large-scale real-world graphs. Pre-computation-based HGNNs address this by performing message passing only once during preprocessing, collecting neighbor information into regular-shaped tensors, which enables efficient mini-batch training. Label-based pre-computation methods collect neighbors' label information but suffer from training label leakage, where a node's own label information propagates back to itself during multi-hop message passing - the echo effect. Existing mitigation strategies are memory-inefficient on large graphs or suffer from compatibility issues with advanced message passing methods. We propose Echoless Label-based Pre-computation (Echoless-LP), which eliminates training label leakage with Partition-Focused Echoless Propagation (PFEP). PFEP partitions target nodes and performs echoless propagation, where nodes in each partition collect label information only from neighbors in other partitions, avoiding echo while remaining memory-efficient and compatible with any message passing method. We also introduce an Asymmetric Partitioning Scheme (APS) and a PostAdjust mechanism to address information loss from partitioning and distributional shifts across partitions. Experiments on public datasets demonstrate that Echoless-LP achieves superior performance and maintains memory efficiency compared to baselines.

---

## 182. Enhancing Graph Representations with Neighborhood-Contextualized Message-Passing

**论文链接:** [http://arxiv.org/abs/2511.11046v1](http://arxiv.org/abs/2511.11046v1)

**作者:** Brian Godwin Lim

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一种新的邻域上下文化消息传递框架，通过融入更广泛的局部邻域上下文信息，增强了传统GNN的表达能力，并开发了SINC-GCN作为具体实现。

### 背景

图神经网络已成为分析关系数据不可或缺的工具，传统GNN可分为三种变体：卷积型、注意力型和消息传递型。标准消息传递型虽然表达能力强，但其典型的成对消息仅单独考虑中心节点和每个邻居节点的特征，未能融入更广泛局部邻域中包含的丰富上下文信息。

### 目的

解决传统消息传递型GNN无法融入更广泛局部邻域中丰富上下文信息的问题，避免其学习整个邻居节点集合中复杂关系的能力受到阻碍。

### 方法

正式提出'邻域上下文化'概念，基于注意力型变体的关键属性，将消息传递型推广为邻域上下文化消息传递框架；提出了一种简单、实用且高效的方法来参数化和实现NCMP，开发了软同构邻域上下文化图卷积网络。

### 主要发现

在合成二元节点分类问题上的初步分析证明了所提出GNN架构的表达能力和效率。

### 结论

该研究为NCMP新框架奠定了基础，作为进一步增强传统GNN图表示能力的实用途径。

### 翻译

图神经网络已成为分析关系数据不可或缺的工具。在文献中，传统GNN可分为三种变体：卷积型、注意力型和消息传递型。虽然标准消息传递型表达能力强，但其典型的成对消息仅单独考虑中心节点和每个邻居节点的特征。这种设计未能融入更广泛局部邻域中包含的丰富上下文信息，可能阻碍其学习整个邻居节点集合中复杂关系的能力。为解决这一局限，本研究首先基于注意力型变体的关键属性，正式提出了邻域上下文化的概念。这随后成为将消息传递型推广为所提出的邻域上下文化消息传递框架的基础。为展示其效用，提出了一种简单、实用且高效的方法来参数化和实现NCMP，从而开发了所提出的软同构邻域上下文化图卷积网络。在合成二元节点分类问题上的初步分析随后强调了所提出GNN架构的表达能力和效率。总体而言，该论文为NCMP新框架奠定了基础，作为进一步增强传统GNN图表示能力的实用途径。


### 论文摘要

Graph neural networks (GNNs) have become an indispensable tool for analyzing relational data. In the literature, classical GNNs may be classified into three variants: convolutional, attentional, and message-passing. While the standard message-passing variant is highly expressive, its typical pair-wise messages nevertheless only consider the features of the center node and each neighboring node individually. This design fails to incorporate the rich contextual information contained within the broader local neighborhood, potentially hindering its ability to learn complex relationships within the entire set of neighboring nodes. To address this limitation, this work first formalizes the concept of neighborhood-contextualization, rooted in a key property of the attentional variant. This then serves as the foundation for generalizing the message-passing variant to the proposed neighborhood-contextualized message-passing (NCMP) framework. To demonstrate its utility, a simple, practical, and efficient method to parametrize and operationalize NCMP is presented, leading to the development of the proposed Soft-Isomorphic Neighborhood-Contextualized Graph Convolution Network (SINC-GCN). A preliminary analysis on a synthetic binary node classification problem then underscores both the expressivity and efficiency of the proposed GNN architecture. Overall, the paper lays the foundation for the novel NCMP framework as a practical path toward further enhancing the graph representational power of classical GNNs.

---

## 183. GraphToxin: Reconstructing Full Unlearned Graphs from Graph Unlearning

**论文链接:** [http://arxiv.org/abs/2511.10936v1](http://arxiv.org/abs/2511.10936v1)

**作者:** Ying Song, Balaji Palanisamy

**发布时间:** 2025-11-14

**备注:** Submitted to S&P 2026. Code will be available

### GPT解析

### 总结

本文提出了GraphToxin，首个针对图遗忘的图重构攻击方法，揭示了图遗忘在隐私保护方面的严重漏洞，并强调了开发更有效防御策略的迫切性。

### 背景

图遗忘作为一种有前景的解决方案，旨在遵守'被遗忘权'法规，通过在请求时删除敏感信息。然而，多方参与创造了新的攻击面，删除的数据仍可能在遗忘的图神经网络中残留痕迹，这些漏洞可能被攻击者利用，恢复被删除的样本。

### 目的

开发一种图重构攻击方法，评估图遗忘方法的脆弱性，并强调开发更有效防御策略的必要性。

### 方法

作者提出了GraphToxin攻击方法，引入了一种新颖的曲率匹配模块，为完整遗忘图恢复提供细粒度指导。该方法被扩展到多种节点删除场景，包括白盒和黑盒设置，并提出了一个全面的评估框架来系统评估攻击性能。

### 主要发现

GraphToxin能够成功破坏图遗忘预期的监管保证，可恢复被删除个体的信息、个人联系及其连接中的敏感内容；现有防御机制对此攻击大多无效，在某些情况下甚至可能增强其性能。

### 结论

GraphToxin对图遗忘方法构成了严重的隐私风险，强调了开发更有效和健壮的防御策略来应对此类攻击的迫切需求。

### 翻译

图遗忘已成为遵守'被遗忘权'法规的有前景解决方案，能够在请求时删除敏感信息。然而，这一解决方案并非万无一失。多方参与创造了新的攻击面，删除的数据仍可能在遗忘的图神经网络中残留痕迹。这些漏洞可能被攻击者利用，恢复被 supposedly 删除的样本，从而破坏图遗忘的固有功能。在这项工作中，我们提出了GraphToxin，这是第一个针对图遗忘的图重构攻击。具体来说，我们引入了一种新颖的曲率匹配模块，为完整遗忘图恢复提供细粒度指导。我们证明GraphToxin能够成功破坏图遗忘预期的监管保证 - 它不仅可以恢复被删除个体的信息和个人联系，还可以恢复其连接中的敏感内容，从而构成更严重的威胁。此外，我们将GraphToxin扩展到多种节点删除场景，包括白盒和黑盒设置。我们强调了最坏情况分析的必要性，并提出了一种全面的评估框架，用于系统评估随机和最坏情况节点删除下的攻击性能。这提供了对图遗忘方法对图重构攻击脆弱性的更健壮和现实的衡量。我们的广泛实验证明了GraphToxin的有效性和灵活性。值得注意的是，我们表明现有防御机制对此攻击大多无效，在某些情况下甚至可能增强其性能。鉴于GraphToxin带来的严重隐私风险，我们的工作强调了开发更有效和健壮的防御策略来应对此类攻击的迫切需求。


### 论文摘要

Graph unlearning has emerged as a promising solution for complying with "the right to be forgotten" regulations by enabling the removal of sensitive information upon request. However, this solution is not foolproof. The involvement of multiple parties creates new attack surfaces, and residual traces of deleted data can still remain in the unlearned graph neural networks. These vulnerabilities can be exploited by attackers to recover the supposedly erased samples, thereby undermining the inherent functionality of graph unlearning. In this work, we propose GraphToxin, the first graph reconstruction attack against graph unlearning. Specifically, we introduce a novel curvature matching module to provide a fine-grained guidance for full unlearned graph recovery. We demonstrate that GraphToxin can successfully subvert the regulatory guarantees expected from graph unlearning - it can recover not only a deleted individual's information and personal links but also sensitive content from their connections, thereby posing substantially more detrimental threats. Furthermore, we extend GraphToxin to multiple node removals under both white-box and black-box setting. We highlight the necessity of a worst-case analysis and propose a comprehensive evaluation framework to systematically assess the attack performance under both random and worst-case node removals. This provides a more robust and realistic measure of the vulnerability of graph unlearning methods to graph reconstruction attacks. Our extensive experiments demonstrate the effectiveness and flexibility of GraphToxin. Notably, we show that existing defense mechanisms are largely ineffective against this attack and, in some cases, can even amplify its performance. Given the severe privacy risks posed by GraphToxin, our work underscores the urgent need for the development of more effective and robust defense strategies against this attack.

---

## 184. Multi-View Polymer Representations for the Open Polymer Prediction

**论文链接:** [http://arxiv.org/abs/2511.10893v1](http://arxiv.org/abs/2511.10893v1)

**作者:** Wonjin Jung, Yongseok Choi

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究采用多视图设计方法进行聚合物性质预测，通过整合四种互补表示提高预测准确性，在NeurIPS 2025开放聚合物预测挑战中取得优异成绩。

### 背景

聚合物性质预测是材料科学中的重要挑战，需要有效的方法来整合不同类型的分子表示信息。

### 目的

开发一个能够准确预测聚合物性质的系统，通过多视图设计充分利用不同类型的分子表示信息。

### 方法

系统整合了四种表示方法：(i)表格形式的RDKit/Morgan描述符，(ii)图神经网络，(iii)3D信息表示，(iv)预训练的SMILES语言模型；通过均匀集成对每个性质预测进行平均；使用10折分割进行模型训练；采用SMILES测试时增强进行评估。

### 主要发现

在NeurIPS 2025开放聚合物预测挑战中，该方法在2241个参赛团队中排名第9；提交的集成模型在公共集上的平均绝对误差为0.057，在私有集上的平均绝对误差为0.082。

### 结论

多视图设计结合不同类型的分子表示可以有效提高聚合物性质预测的准确性，在实际应用中取得了优异的性能。

### 翻译

我们采用多视图设计解决聚合物性质预测问题，该方法利用互补表示。我们的系统整合了四类方法：(i)表格形式的RDKit/Morgan描述符，(ii)图神经网络，(iii)3D信息表示，(iv)预训练的SMILES语言模型，并通过均匀集成对每个性质预测进行平均。模型使用10折分割进行训练，并采用SMILES测试时增强进行评估。该方法在NeurIPS 2025开放聚合物预测挑战中2241个团队中排名第9。提交的集成模型在公共集上的平均绝对误差为0.057，在私有集上的平均绝对误差为0.082。


### 论文摘要

We address polymer property prediction with a multi-view design that exploits complementary representations. Our system integrates four families: (i) tabular RDKit/Morgan descriptors, (ii) graph neural networks, (iii) 3D-informed representations, and (iv) pretrained SMILES language models, and averages per-property predictions via a uniform ensemble. Models are trained with 10-fold splits and evaluated with SMILES test-time augmentation. The approach ranks 9th of 2241 teams in the Open Polymer Prediction Challenge at NeurIPS 2025. The submitted ensemble achieves a public MAE of 0.057 and a private MAE of 0.082.

---

## 185. DESS: DeBERTa Enhanced Syntactic-Semantic Aspect Sentiment Triplet Extraction

**论文链接:** [http://arxiv.org/abs/2511.10577v1](http://arxiv.org/abs/2511.10577v1)

**作者:** Vishal Thenuwara, Nisansa de Silva

**发布时间:** 2025-11-13

**DOI:** 10.1007/978-3-032-09318-9_13

**备注:** 15 pages, 2 figures. Published in Proceedings of the 17th International Conference on Computational Collective Intelligence (ICCCI 2025), Lecture Notes in Artificial Intelligence, Springer

### GPT解析

### 总结

该研究提出了一种名为DESS的新方法，用于解决方面情感三元组提取（ASTE）中的细粒度情感分析挑战。通过整合DeBERTa的增强注意力机制和双通道结构（DeBERTa和LSTM），有效提高了文本中方面、观点和情感极性关系的识别能力，在标准测试集上取得了显著的性能提升。

### 背景

细粒度情感分析在方面情感三元组提取（ASTE）中面临持续挑战，特别是在准确捕捉方面、观点和情感极性之间的关系方面。尽管研究人员使用BERT和图神经网络取得了进展，但高级语言模型在理解复杂语言模式方面的潜力仍未被充分探索。

### 目的

探索更先进的语言模型在理解复杂语言模式方面的潜力，特别是在细粒度情感分析和方面情感三元组提取任务中的应用，以提高方面-观点对识别和情感极性判断的准确性。

### 方法

引入DESS方法，整合DeBERTa的增强注意力机制以更好地理解文本中的上下文和关系。框架保持双通道结构，其中DeBERTa与LSTM通道协同工作，处理文本中的意义和语法模式。研究仔细调整了这些组件的协同工作方式，特别关注不同类型语言信息的交互方式。

### 主要发现

在标准数据集上测试时，DESS相比当前方法显示出有意义的改进，在识别方面-观点对和准确确定情感方面，F1分数分别提高了4.85、8.36和2.42。DeBERTa的复杂注意力系统有助于DESS更好地处理复杂的句子结构，特别是当重要词相距较远时。

### 结论

在经过深思熟虑的集成后，升级到更先进的语言模型可以显著提高文本情感分析的能力。DESS方法的实现已在GitHub上公开。

### 翻译

细粒度情感分析在方面情感三元组提取（ASTE）中面临持续挑战，特别是在准确捕捉方面、观点和情感极性之间的关系方面。尽管研究人员使用BERT和图神经网络取得了进展，但高级语言模型在理解复杂语言模式方面的潜力仍未被充分探索。我们引入了DESS，一种新方法，它基于先前工作，整合了DeBERTa的增强注意力机制，以更好地理解文本中的上下文和关系。我们的框架保持双通道结构，其中DeBERTa与LSTM通道协同工作，处理文本中的意义和语法模式。我们仔细调整了这些组件的协同工作方式，特别关注不同类型语言信息的交互方式。当我们在标准数据集上测试DESS时，它显示出比当前方法有意义的改进，在识别方面-观点对和准确确定情感方面，F1分数分别提高了4.85、8.36和2.42。更深入地分析结果，我们发现DeBERTa的复杂注意力系统有助于DESS更好地处理复杂的句子结构，特别是当重要词相距较远时。我们的研究结果表明，在经过深思熟虑的集成后，升级到更先进的语言模型可以真正提高我们分析文本情感的能力。我们方法的实现已在GitHub上公开：https://github.com/VishalRepos/DESS。


### 论文摘要

Fine-grained sentiment analysis faces ongoing challenges in Aspect Sentiment Triple Extraction (ASTE), particularly in accurately capturing the relationships between aspects, opinions, and sentiment polarities. While researchers have made progress using BERT and Graph Neural Networks, the full potential of advanced language models in understanding complex language patterns remains unexplored. We introduce DESS, a new approach that builds upon previous work by integrating DeBERTa's enhanced attention mechanism to better understand context and relationships in text. Our framework maintains a dual-channel structure, where DeBERTa works alongside an LSTM channel to process both meaning and grammatical patterns in text. We have carefully refined how these components work together, paying special attention to how different types of language information interact. When we tested DESS on standard datasets, it showed meaningful improvements over current methods, with F1-score increases of 4.85, 8.36, and 2.42 in identifying aspect opinion pairs and determining sentiment accurately. Looking deeper into the results, we found that DeBERTa's sophisticated attention system helps DESS handle complicated sentence structures better, especially when important words are far apart. Our findings suggest that upgrading to more advanced language models when thoughtfully integrated, can lead to real improvements in how well we can analyze sentiments in text. The implementation of our approach is publicly available at: https://github.com/VishalRepos/DESS.

---

## 186. GraphFaaS: Serverless GNN Inference for Burst-Resilient, Real-Time Intrusion Detection

**论文链接:** [http://arxiv.org/abs/2511.10554v1](http://arxiv.org/abs/2511.10554v1)

**作者:** Lingzhi Wang, Vinod Yegneswaran, Xinyi Shi, Ziyu Li, Ashish Gehani, Yan Chen

**发布时间:** 2025-11-13

**备注:** Accepted by ML For Systems workshop at Neural Information Processing Systems (NeurIPS 2025)

### GPT解析

### 总结

GraphFaaS是一种专为基于图神经网络(GNN)的入侵检测设计的无服务器架构，通过动态扩展GNN推理管道解决了传统架构在检测延迟和工作负载处理方面的挑战。

### 背景

基于来源的入侵检测是图形机器学习在网络安全中日益流行的应用，系统活动被建模为来源图来捕获潜在恶意行为之间的因果关系和相关性，而GNN在此领域已展现出强大的性能。

### 目的

为了解决传统静态配置GNN推理架构无法满足的持续低检测延迟和高度不规则突发工作负载处理两大关键需求。

### 方法

GraphFaaS利用无服务器计算的弹性和敏捷性动态扩展GNN推理管道，将GNN工作流并行化并适应无服务器环境，使系统能实时响应波动的工作负载，同时将计算资源与静态配置分离。

### 主要发现

通过动态扩展和无服务器架构，GraphFaaS能够提供稳定的推理延迟，这对于可靠的入侵检测和网络安全操作中的及时事件响应至关重要。

### 结论

初步评估显示，与基线相比，GraphFaaS将平均检测延迟减少了85%，变异系数(CV)减少了64%，显著提升了入侵检测系统的性能。

### 翻译

基于来源的入侵检测是图形机器学习在网络安全中日益流行的应用，系统活动被建模为来源图以捕获潜在恶意行为之间的因果关系和相关性。图神经网络(GNN)在此场景中已展现出强大的性能。然而，传统的静态配置GNN推理架构无法满足入侵检测的两个关键需求：(1)保持持续的低检测延迟，(2)处理高度不规则和突发的工作负载。为了全面解决这些挑战，我们提出了GraphFaaS，一种专为基于GNN的入侵检测设计的无服务器架构。GraphFaaS利用无服务器计算的弹性和敏捷性来动态扩展GNN推理管道。我们将GNN工作流并行化并适应无服务器环境，确保系统能够实时响应波动的工作负载。通过将计算资源与静态配置分离，GraphFaaS提供了稳定的推理延迟，这对于可靠的入侵检测和网络安全操作中的及时事件响应至关重要。初步评估显示，与基线相比，GraphFaaS将平均检测延迟减少了85%，变异系数(CV)减少了64%。


### 论文摘要

Provenance-based intrusion detection is an increasingly popular application of graphical machine learning in cybersecurity, where system activities are modeled as provenance graphs to capture causality and correlations among potentially malicious actions. Graph Neural Networks (GNNs) have demonstrated strong performance in this setting. However, traditional statically-provisioned GNN inference architectures fall short in meeting two crucial demands of intrusion detection: (1) maintaining consistently low detection latency, and (2) handling highly irregular and bursty workloads. To holistically address these challenges, we present GraphFaaS, a serverless architecture tailored for GNN-based intrusion detection. GraphFaaS leverages the elasticity and agility of serverless computing to dynamically scale the GNN inference pipeline. We parallelize and adapt GNN workflows to a serverless environment, ensuring that the system can respond in real time to fluctuating workloads. By decoupling compute resources from static provisioning, GraphFaaS delivers stable inference latency, which is critical for dependable intrusion detection and timely incident response in cybersecurity operations. Preliminary evaluation shows GraphFaaS reduces average detection latency by 85% and coefficient of variation (CV) by 64% compared to the baseline.

---

## 187. Strategic Opponent Modeling with Graph Neural Networks, Deep Reinforcement Learning and Probabilistic Topic Modeling

**论文链接:** [http://arxiv.org/abs/2511.10501v2](http://arxiv.org/abs/2511.10501v2)

**作者:** Georgios Chalkiadakis, Charilaos Akasiadis, Gerasimos Koresis, Stergios Plataniotis, Leonidas Bakopoulos

**发布时间:** 2025-11-13

**备注:** 26 pages

### GPT解析

### 总结

这篇论文对图神经网络、深度强化学习和概率主题建模进行了全面综述，探讨了它们在战略多智能体环境中的应用潜力，分析了处理不确定性和异质性的能力，并确定了几个开放性挑战。

### 背景

在战略多智能体环境中，传统的博弈论方法依赖于共同先验假设和自利假设等在现实场景中通常无效的假设。同时，现实应用中普遍存在不确定性和异质性问题，需要新的方法来处理。

### 目的

(i) 探索目前用于揭示未知模型结构的机器学习方法，这些方法可适应战略对手建模的任务；(ii) 将这些方法与博弈论概念相结合，避免依赖无效假设；(iii) 分析处理不确定性和异质性的能力，以及可扩展性。

### 方法

论文综述了三种主要方法：图神经网络（用于处理图结构数据和执行节点分类、链接预测等任务）、多智能体深度强化学习（MADRL）以及概率主题建模（用于估计未知潜在分布和解决异质性问题）。

### 主要发现

图神经网络被证明是有效建模多智能体环境中关系和交互的强大工具；概率主题建模在文档分析领域之外的潜力被低估；存在几个关键挑战需要解决，包括适应非平稳环境、平衡稳定性和适应性、解决不确定性和异质性，以及保证可扩展性。

### 结论

结合机器学习方法和博弈论概念可以为战略多智能体环境提供更有效的解决方案，特别是在处理现实世界中的不确定性和异质性方面。图神经网络、深度强化学习和概率主题建模的结合应用具有巨大潜力，但仍需解决若干开放性挑战。

### 翻译

本文主要对图神经网络、深度强化学习和概率主题建模方法进行了全面综述，重点关注它们在战略多智能体环境中的潜在应用。我们关注(i)目前用于揭示未知模型结构的机器学习方法，这些方法可适应战略对手建模任务；(ii)将这些方法与博弈论概念相结合，避免依赖在现实场景中通常无效的假设，如共同先验假设(CPA)和自利假设(SIH)。我们分析了处理不确定性和异质性的能力，这两种特征在现实应用案例中非常常见，以及可扩展性。作为有效建模多智能体环境中关系和交互的潜在解决方案，我们提倡使用图神经网络(GNN)。这类方法旨在处理图结构数据，已被证明是执行节点分类和链接预测等任务的强大工具。接下来，我们回顾了强化学习(RL)领域，特别是多智能体深度强化学习(MADRL)。随后，我们描述了现有的相关博弈论解决方案概念，并考虑了公平性和稳定性等属性。我们的综述还包括了在文档分析和分类领域之外使用PTM的文献。PTM估计未知潜在分布的能力有助于处理异质性和未知智能体信念。最后，我们确定了特定的开放性挑战，特别是需要(i)适应非平稳环境，(ii)平衡稳定性和适应性，(iii)解决不确定性和异质性，(iv)保证可扩展性和解决方案的可处理性。


### 论文摘要

This paper provides a comprehensive review of mainly Graph Neural Networks, Deep Reinforcement Learning, and Probabilistic Topic Modeling methods with a focus on their potential incorporation in strategic multiagent settings. We draw interest in (i) Machine Learning methods currently utilized for uncovering unknown model structures adaptable to the task of strategic opponent modeling, and (ii) the integration of these methods with Game Theoretic concepts that avoid relying on assumptions often invalid in real-world scenarios, such as the Common Prior Assumption (CPA) and the Self-Interest Hypothesis (SIH). We analyze the ability to handle uncertainty and heterogeneity, two characteristics that are very common in real-world application cases, as well as scalability. As a potential answer to effectively modeling relationships and interactions in multiagent settings, we champion the use of Graph Neural Networks (GNN). Such approaches are designed to operate upon graph-structured data, and have been shown to be a very powerful tool for performing tasks such as node classification and link prediction. Next, we review the domain of Reinforcement Learning (RL), and in particular that of Multiagent Deep Reinforcement Learning (MADRL). Following, we describe existing relevant game theoretic solution concepts and consider properties such as fairness and stability. Our review comes complete with a note on the literature that utilizes PTM in domains other than that of document analysis and classification. The capability of PTM to estimate unknown underlying distributions can help with tackling heterogeneity and unknown agent beliefs. Finally, we identify certain open challenges specifically, the need to (i) fit non-stationary environments, (ii) balance the degrees of stability and adaptation, (iii) tackle uncertainty and heterogeneity, (iv) guarantee scalability and solution tractability.

---

## 188. FastGraph: Optimized GPU-Enabled Algorithms for Fast Graph Building and Message Passing

**论文链接:** [http://arxiv.org/abs/2511.10442v1](http://arxiv.org/abs/2511.10442v1)

**作者:** Aarush Agarwal, Raymond He, Jan Kieseler, Matteo Cremonesi, Shah Rukh Qasim

**发布时间:** 2025-11-13

### GPT解析

### 总结

FastGraph是一种新的GPU优化的k近邻算法，专门设计用于加速低维空间中的图构建，这对高性能图神经网络至关重要。

### 背景

图神经网络需要高效的图构建方法，特别是在低维空间中。

### 目的

开发一种GPU优化的算法来加速低维空间中的图构建，以提高图神经网络的性能。

### 方法

FastGraph采用GPU驻留的分区方法，具有完整的梯度流支持和自适应参数调整，显著提高了计算和内存效率。

### 主要发现

基准测试表明，FastGraph在维度小于10的情况下，比最先进的库如FAISS、ANNOY和SCANN快20-40倍，且几乎没有内存开销。

### 结论

这些改进直接转化为基于GNN工作流的显著性能提升，特别有利于低维计算密集型应用，如高能物理中的粒子聚类、视觉目标跟踪和图聚类。

### 翻译

我们介绍了FastGraph，这是一种新型的GPU优化k近邻算法，专门设计用于加速低维空间中的图构建，这对高性能图神经网络至关重要。我们的方法采用GPU驻留的分区方法，具有完整的梯度流支持和自适应参数调整，显著提高了计算和内存效率。基准测试表明，FastGraph在维度小于10的情况下，比最先进的库如FAISS、ANNOY和SCANN快20-40倍，且几乎没有内存开销。这些改进直接转化为基于GNN工作流的显著性能提升，特别有利于低维计算密集型应用，如高能物理中的粒子聚类、视觉目标跟踪和图聚类。


### 论文摘要

We introduce FastGraph, a novel GPU-optimized k-nearest neighbor algorithm specifically designed to accelerate graph construction in low-dimensional spaces (2-10 dimensions), critical for high-performance graph neural networks. Our method employs a GPU-resident, bin-partitioned approach with full gradient-flow support and adaptive parameter tuning, significantly enhancing both computational and memory efficiency. Benchmarking demonstrates that FastGraph achieves a 20-40x speedup over state-of-the-art libraries such as FAISS, ANNOY, and SCANN in dimensions less than 10 with virtually no memory overhead. These improvements directly translate into substantial performance gains for GNN-based workflows, particularly benefiting computationally intensive applications in low dimensions such as particle clustering in high-energy physics, visual object tracking, and graph clustering.

---

## 189. SVD-NO: Learning PDE Solution Operators with SVD Integral Kernels

**论文链接:** [http://arxiv.org/abs/2511.10025v1](http://arxiv.org/abs/2511.10025v1)

**作者:** Noam Koren, Ralf J. J. Mackenbach, Ruud J. G. van Sloun, Kira Radinsky, Daniel Freedman

**发布时间:** 2025-11-13

**备注:** AAAI-26

### GPT解析

### 总结

该论文提出了一种名为SVD-NO的新型神经算子，用于从数据中直接学习偏微分方程的解算子。该算子通过奇异值分解显式参数化核函数，并在低秩基中执行积分，实现了高表达能力和合理的计算复杂度，在五个不同基准方程测试中达到新最先进水平。

### 背景

神经算子已成为从数据中直接学习偏微分方程解算子的有前景范式。然而，现有方法(如基于傅里叶或图技术的方法)对核积分算子的结构做出强假设，这可能限制了其表达能力。

### 目的

开发一种能够克服现有方法限制的新型神经算子，该算子应具有更高的表达能力，同时保持合理的计算复杂度，特别是在解具有高度空间变化的PDEs上表现更好。

### 方法

提出SVD-NO，通过奇异值分解显式参数化核的神经算子。使用两个轻量级网络学习左右奇异函数，一个对角参数矩阵学习奇异值，以及一个Gram矩阵正则化器 enforcing 正交性。SVD-NO通过近似完整核获得高度表达能力，同时保持合理的计算复杂度。

### 主要发现

在五个不同基准方程的广泛评估中，SVD-NO达到新最先进水平。特别地，对于解具有高度空间变化的PDEs，SVD-NO提供了更大的性能提升。

### 结论

SVD-NO是一种实用且高性能的神经算子，能有效学习偏微分方程的解算子，特别是在解具有高度空间变化的情况下表现优异。该研究的代码已在GitHub上公开。

### 翻译

神经算子已成为一种有前景的范式，可以直接从数据中学习偏微分方程(PDEs)的解算子。现有方法，如基于傅里叶或图技术的方法，对核积分算子的结构做出了强假设，这些假设可能限制了表达能力。我们提出了SVD-NO，一种神经算子，它通过奇异值分解(SVD)显式参数化核，然后在低秩基中直接执行积分。两个轻量级网络学习左右奇异函数，一个对角参数矩阵学习奇异值，一个Gram矩阵正则化器 enforcing 正交性。由于SVD-NO近似完整核，它获得了高度表达能力。此外，由于其低秩结构，应用该算子的计算复杂度仍然合理，形成了一个实用的系统。在五个不同基准方程的广泛评估中，SVD-NO达到了新的最先进水平。特别是，对于解具有高度空间变化的PDEs，SVD-NO提供了更大的性能提升。本工作的代码可在https://github.com/2noamk/SVDNO.git公开获取。


### 论文摘要

Neural operators have emerged as a promising paradigm for learning solution operators of partial differential equa- tions (PDEs) directly from data. Existing methods, such as those based on Fourier or graph techniques, make strong as- sumptions about the structure of the kernel integral opera- tor, assumptions which may limit expressivity. We present SVD-NO, a neural operator that explicitly parameterizes the kernel by its singular-value decomposition (SVD) and then carries out the integral directly in the low-rank basis. Two lightweight networks learn the left and right singular func- tions, a diagonal parameter matrix learns the singular values, and a Gram-matrix regularizer enforces orthonormality. As SVD-NO approximates the full kernel, it obtains a high de- gree of expressivity. Furthermore, due to its low-rank struc- ture the computational complexity of applying the operator remains reasonable, leading to a practical system. In exten- sive evaluations on five diverse benchmark equations, SVD- NO achieves a new state of the art. In particular, SVD-NO provides greater performance gains on PDEs whose solutions are highly spatially variable. The code of this work is publicly available at https://github.com/2noamk/SVDNO.git.

---

## 190. GraphSB: Boosting Imbalanced Node Classification on Graphs through Structural Balance

**论文链接:** [http://arxiv.org/abs/2511.10022v1](http://arxiv.org/abs/2511.10022v1)

**作者:** Chaofan Zhu, Xiaobing Rui, Zhixiao Wang

**发布时间:** 2025-11-13

### GPT解析

### 总结

论文提出了GraphSB框架，通过结构平衡策略解决图学习中不平衡节点分类的根本问题，在节点合成前优化图结构，显著提高了GNNs的学习效果。

### 背景

不平衡节点分类是图学习中的关键挑战，现有方法主要分为数据级(合成少数类节点)和算法级(优化学习过程突出少数类)，但都未解决内在的不平衡图结构问题。

### 目的

提出GraphSB框架，以结构平衡作为关键策略，在节点合成前解决潜在的不平衡图结构问题，使GNNs能够进行更有效的学习。

### 方法

GraphSB采用两阶段结构优化：结构增强(自适应构建基于相似性的边，加强少数类节点连接)和关系扩散(捕获高阶依赖关系，放大少数类信号)，在节点合成前平衡结构分布。

### 主要发现

理论分析证实了内在的不平衡图结构是导致多数类主导和少数类同化的根本因素，而GraphSB框架通过结构平衡策略有效解决了这一问题。

### 结论

GraphSB显著优于最先进的方法，且结构平衡可作为即插即用模块集成到其他方法中，平均提高准确性3.67%。

### 翻译

不平衡节点分类是图学习中的一个关键挑战，大多数现有方法通常利用图神经网络学习节点表示。这些方法大致可分为数据级和算法级。前者旨在合成少数类节点以减轻数量不平衡，而后者试图优化学习过程以突出少数类。然而，这两类方法都没有解决内在的不平衡图结构问题，这是导致图神经网络中多数类主导和少数类同化的基本因素。我们的理论分析进一步支持了这一关键见解。因此，我们提出了GraphSB，一个新颖的框架，它将结构平衡作为关键策略，在节点合成前解决潜在的不平衡图结构问题。结构平衡执行两阶段结构优化：结构增强自适应构建基于相似性的边以加强少数类节点的连接性，关系扩散捕获高阶依赖关系同时放大少数类信号。因此，GraphSB在节点合成前平衡结构分布，使图神经网络能够进行更有效的学习。大量实验证明GraphSB显著优于最先进的方法。更重要的是，所提出的结构平衡可以无缝集成到最先进的方法中作为简单的即插即用模块，平均提高它们的准确性3.67%。


### 论文摘要

Imbalanced node classification is a critical challenge in graph learning, where most existing methods typically utilize Graph Neural Networks (GNNs) to learn node representations. These methods can be broadly categorized into the data-level and the algorithm-level. The former aims to synthesize minority-class nodes to mitigate quantity imbalance, while the latter tries to optimize the learning process to highlight minority classes. However, neither category addresses the inherently imbalanced graph structure, which is a fundamental factor that incurs majority-class dominance and minority-class assimilation in GNNs. Our theoretical analysis further supports this critical insight. Therefore, we propose GraphSB (Graph Structural Balance), a novel framework that incorporates Structural Balance as a key strategy to address the underlying imbalanced graph structure before node synthesis. Structural Balance performs a two-stage structure optimization: Structure Enhancement that adaptively builds similarity-based edges to strengthen connectivity of minority-class nodes, and Relation Diffusion that captures higher-order dependencies while amplifying signals from minority classes. Thus, GraphSB balances structural distribution before node synthesis, enabling more effective learning in GNNs. Extensive experiments demonstrate that GraphSB significantly outperforms the state-of-the-art methods. More importantly, the proposed Structural Balance can be seamlessly integrated into state-of-the-art methods as a simple plug-and-play module, increasing their accuracy by an average of 3.67\%.

---

## 191. ASSENT: Learning-Based Association Optimization for Distributed Cell-Free ISAC

**论文链接:** [http://arxiv.org/abs/2511.09992v1](http://arxiv.org/abs/2511.09992v1)

**作者:** Mehdi Zafari, A. Lee Swindlehurst

**发布时间:** 2025-11-13

**备注:** Preprint. 6 pages, 2 figures, 2 tables. Under review. Code and datasets: https://github.com/LS-Wireless/ASSENT-CellFree-ISAC

### GPT解析

### 总结

本文提出了一种名为ASSENT的图神经网络框架，用于解决分布式无小区集成感知与通信(ISAC)系统中的接入点聚类、用户/目标调度和AP模式选择问题，实现了接近最优的性能并减少了决策延迟。

### 背景

ISAC是6G的关键新兴技术，但现有方法在分布式部署下缺乏可扩展的联合AP聚类和用户/目标调度方法，且主要依赖集中处理和完整信道状态信息，限制了系统的可扩展性。

### 目的

解决分布式无小区ISAC系统中接入点聚类、用户和目标调度以及AP模式选择问题，这些系统在前传容量受限的情况下运行。

### 方法

将问题表述为混合整数线性规划(MILP)联合捕获干扰耦合、RF链限制和感知需求；提出ASSENT框架，一种在MILP解决方案上训练的图神经网络，从轻量级链路统计中学习关联和模式选择策略。

### 主要发现

ASSENT实现接近最优的效用；准确学习底层关联；与基于优化的方法相比，其单次前向推理减少了决策延迟。

### 结论

提供了开源Python/PyTorch实现和完整数据集，以促进可复制和可扩展的无小区ISAC研究。

### 翻译

集成感知与通信(ISAC)是6G的关键新兴技术。尽管已有进展，ISAC在分布式部署和前传限制下仍缺乏可扩展的联合AP聚类和用户/目标调度方法。此外，现有ISAC解决方案主要依赖集中处理和完整信道状态信息，限制了可扩展性。本文解决了在前传容量受限情况下运行的分布式无小区ISAC系统中的接入点(AP)聚类、用户和目标调度以及AP模式选择问题。我们将问题表述为混合整数线性规划(MILP)，联合捕获干扰耦合、RF链限制和感知需求，提供最优但计算密集型解决方案。为实现实时和可扩展操作，我们提出ASSENT(关联和实体选择)，一种在MILP解决方案上训练的图神经网络(GNN)框架，从轻量级链路统计中高效学习关联和模式选择策略。仿真显示，ASSENT实现接近最优效用，同时准确学习底层关联。此外，与基于优化的方法相比，其单次前向推理减少了决策延迟。提供了包含完整数据集的开源Python/PyTorch实现，以促进可复制和可扩展的无小区ISAC研究。


### 论文摘要

Integrated Sensing and Communication (ISAC) is a key emerging 6G technology. Despite progress, ISAC still lacks scalable methods for joint AP clustering and user/target scheduling in distributed deployments under fronthaul limits. Moreover, existing ISAC solutions largely rely on centralized processing and full channel state information, limiting scalability. This paper addresses joint access point (AP) clustering, user and target scheduling, and AP mode selection in distributed cell-free ISAC systems operating with constrained fronthaul capacity. We formulate the problem as a mixed-integer linear program (MILP) that jointly captures interference coupling, RF-chain limits, and sensing requirements, providing optimal but computationally demanding solutions. To enable real-time and scalable operation, we propose ASSENT (ASSociation and ENTity selection), a graph neural network (GNN) framework trained on MILP solutions to efficiently learn association and mode-selection policies directly from lightweight link statistics. Simulations show that ASSENT achieves near-optimal utility while accurately learning the underlying associations. Additionally, its single forward pass inference reduces decision latency compared to optimization-based methods. An open-source Python/PyTorch implementation with full datasets is provided to facilitate reproducible and extensible research in cell-free ISAC.

---

## 192. AI-Integrated Decision Support System for Real-Time Market Growth Forecasting and Multi-Source Content Diffusion Analytics

**论文链接:** [http://arxiv.org/abs/2511.09962v1](http://arxiv.org/abs/2511.09962v1)

**作者:** Ziqing Yin, Xuanjing Chen, Xi Zhang

**发布时间:** 2025-11-13

### GPT解析

### 总结

本研究提出了一种AI驱动的决策支持系统，用于预测AI生成内容的传播轨迹和市场影响，通过整合多源数据和混合神经网络模型，显著提升了营销决策的准确性和效率。

### 背景

AI生成内容(AIGC)的快速增长改变了数字营销和在线消费者行为的动态。然而，由于数据异质性、非线性传播机制和不断发展的消费者交互，预测此类内容的传播轨迹和市场影响仍然具有挑战性。

### 目的

开发一个AI驱动的决策支持系统(DSS)，以有效预测AI生成内容的传播轨迹和市场影响，为营销决策提供支持。

### 方法

该系统整合了多源数据（社交媒体流、营销支出记录、消费者参与日志和情感动态），采用混合图神经网络(GNN)和时序Transformer框架。通过双通道架构联合学习内容扩散结构和时间影响演化，同时使用因果推理模块分离营销刺激对投资回报率和市场可见性的影响。

### 主要发现

在从Twitter、TikTok和YouTube广告等多个在线平台收集的大规模真实世界数据集上的实验表明，该系统在所有六个评估指标上都优于现有基线方法。

### 结论

提出的决策支持系统通过提供对AI生成内容传播和市场增长模式的可解释实时洞察，显著增强了营销决策的质量和效果。

### 翻译

AI生成内容(AIGC)的迅速普及重塑了数字营销和在线消费者行为的动态。然而，由于数据异质性、非线性传播机制和不断发展的消费者交互，预测此类内容的传播轨迹和市场影响仍然具有挑战性。本研究提出了一种AI驱动的决策支持系统(DSS)，该系统整合了多源数据，包括社交媒体流、营销支出记录、消费者参与日志和情感动态，使用混合图神经网络(GNN)和时序Transformer框架。该模型通过双通道架构联合学习内容扩散结构和时间影响演化，同时因果推理模块分离营销刺激对投资回报率(ROI)和市场可见性的影响。在从Twitter、TikTok和YouTube广告等多个在线平台收集的大规模真实世界数据集上的实验表明，我们的系统在所有六个指标上都优于现有基线。所提出的DSS通过提供对AI生成内容驱动传播和市场增长模式的可解释实时洞察，增强了营销决策。


### 论文摘要

The rapid proliferation of AI-generated content (AIGC) has reshaped the dynamics of digital marketing and online consumer behavior. However, predicting the diffusion trajectory and market impact of such content remains challenging due to data heterogeneity, non linear propagation mechanisms, and evolving consumer interactions. This study proposes an AI driven Decision Support System (DSS) that integrates multi source data including social media streams, marketing expenditure records, consumer engagement logs, and sentiment dynamics using a hybrid Graph Neural Network (GNN) and Temporal Transformer framework. The model jointly learns the content diffusion structure and temporal influence evolution through a dual channel architecture, while causal inference modules disentangle the effects of marketing stimuli on return on investment (ROI) and market visibility. Experiments on large scale real-world datasets collected from multiple online platforms such as Twitter, TikTok, and YouTube advertising show that our system outperforms existing baselines in all six metrics. The proposed DSS enhances marketing decisions by providing interpretable real-time insights into AIGC driven content dissemination and market growth patterns.

---

## 193. AdaptViG: Adaptive Vision GNN with Exponential Decay Gating

**论文链接:** [http://arxiv.org/abs/2511.09942v1](http://arxiv.org/abs/2511.09942v1)

**作者:** Mustafa Munir, Md Mostafijur Rahman, Radu Marculescu

**发布时间:** 2025-11-13

**备注:** Accepted in 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2026)

### GPT解析

### 总结

该研究提出了AdaptViG，一种高效且强大的混合视觉图神经网络，通过引入自适应图卷积机制解决了传统ViGs在图构建阶段的计算挑战，实现了准确性和效率之间更好的权衡。

### 背景

Vision Graph Neural Networks (ViGs)为视觉架构的发展提供了新方向，但它们在图构建阶段通常面临重大的计算挑战，这限制了它们的效率。

### 目的

解决ViGs在图构建阶段的计算效率问题，同时保持或提高模型性能，实现准确性和效率之间的更好权衡。

### 方法

1. 提出了AdaptViG，一种高效且强大的混合视觉GNN；2. 引入了自适应图卷积机制，包括高效的静态轴向支架和基于内容的动态门控策略(指数衰减门控)；3. 采用混合策略，在早期阶段使用高效门控机制，在最终阶段使用完整的全局注意力块以实现最大特征聚合。

### 主要发现

1. AdaptViG-M模型达到82.6%的top-1准确率，比ViG-B高0.3%，同时参数减少80%，GMACs减少84%；2. 在下游任务上，AdaptViG-M获得45.8 mIoU、44.8 APbox和41.1 APmask；3. 这些结果超过了规模大得多的EfficientFormer-L7，同时参数减少78%。

### 结论

AdaptViG在视觉GNNs中实现了准确性和效率之间新的最先进权衡，证明了其在各种视觉任务中的优越性能和效率。

### 翻译

视觉图神经网络(ViGs)为视觉架构的发展提供了新方向。虽然功能强大，但ViGs通常面临来自图构建阶段的重大计算挑战，这可能阻碍它们的效率。为了解决这个问题，我们提出了AdaptViG，一种高效且强大的混合视觉GNN，引入了一种称为自适应图卷积的新型图构建机制。该机制基于高效的静态轴向支架和一种称为指数衰减门控的动态、内容感知门控策略。这种门控机制根据特征相似性选择性地加权长距离连接。此外，AdaptViG采用混合策略，在早期阶段使用我们的高效门控机制，在最终阶段使用完整的全局注意力块以实现最大特征聚合。我们的方法在视觉GNNs的准确性和效率之间实现了新的最先进权衡。


### 论文摘要

Vision Graph Neural Networks (ViGs) offer a new direction for advancements in vision architectures. While powerful, ViGs often face substantial computational challenges stemming from their graph construction phase, which can hinder their efficiency. To address this issue we propose AdaptViG, an efficient and powerful hybrid Vision GNN that introduces a novel graph construction mechanism called Adaptive Graph Convolution. This mechanism builds upon a highly efficient static axial scaffold and a dynamic, content-aware gating strategy called Exponential Decay Gating. This gating mechanism selectively weighs long-range connections based on feature similarity. Furthermore, AdaptViG employs a hybrid strategy, utilizing our efficient gating mechanism in the early stages and a full Global Attention block in the final stage for maximum feature aggregation. Our method achieves a new state-of-the-art trade-off between accuracy and efficiency among Vision GNNs. For instance, our AdaptViG-M achieves 82.6% top-1 accuracy, outperforming ViG-B by 0.3% while using 80% fewer parameters and 84% fewer GMACs. On downstream tasks, AdaptViG-M obtains 45.8 mIoU, 44.8 APbox, and 41.1 APmask, surpassing the much larger EfficientFormer-L7 by 0.7 mIoU, 2.2 APbox, and 2.1 APmask, respectively, with 78% fewer parameters.

---

## 194. GEM+: Scalable State-of-the-Art Private Synthetic Data with Generator Networks

**论文链接:** [http://arxiv.org/abs/2511.09672v1](http://arxiv.org/abs/2511.09672v1)

**作者:** Samuel Maddock, Shripad Gade, Graham Cormode, Will Bullock

**发布时间:** 2025-11-12

### GPT解析

### 总结

本文提出了GEM+方法，结合了AIM的自适应测量框架和GEM的可扩展生成器网络，用于差分私有合成表格数据生成。

### 背景

当前最先进的差分私有合成表格数据方法如AIM使用自适应'选择-测量-生成'框架，但图形模型在高维数据中效率低下；GEM使用生成器神经网络提高可扩展性，但经验比较主要基于小型数据集。

### 目的

开发GEM+方法，结合AIM的自适应测量框架和GEM的可扩展生成器网络，以处理大型数据集（超过一百列），解决AIM在内存和计算效率上的限制。

### 方法

将AIM的自适应测量框架与GEM的可扩展生成器网络集成，创建GEM+方法，保留两种方法的优势。

### 主要发现

实验表明GEM+在效用和可扩展性上都优于AIM，能够高效处理超过一百列的数据集，而AIM由于内存和计算开销无法处理。

### 结论

GEM+代表了差分私有合成表格数据生成的新进展，能够有效处理高维数据，提供了最先进的结果。

### 翻译

最先进的差分私有合成表格数据已由自适应的'选择-测量-生成'框架定义，例如AIM等方法。这些方法迭代地测量低阶噪声边际并拟合图形模型以生成合成数据，使能够在隐私约束下系统优化数据质量。然而，图形模型对于高维数据效率低下，因为它们需要大量内存，并且每当图形结构变化时都必须从头重新训练，导致显著的计算开销。最近的方法如GEM使用生成器神经网络提高了可扩展性，克服了这些限制。然而，经验比较主要集中在小型数据集上，限制了实际应用。在本工作中，我们引入了GEM+，它将AIM的自适应测量框架与GEM的可扩展生成器网络相结合。我们的实验表明，GEM+在效用和可扩展性上都优于AIM，提供了最先进的结果，同时能够高效处理超过一百列的数据集，而AIM由于内存和计算开销无法处理。


### 论文摘要

State-of-the-art differentially private synthetic tabular data has been defined by adaptive 'select-measure-generate' frameworks, exemplified by methods like AIM. These approaches iteratively measure low-order noisy marginals and fit graphical models to produce synthetic data, enabling systematic optimisation of data quality under privacy constraints. Graphical models, however, are inefficient for high-dimensional data because they require substantial memory and must be retrained from scratch whenever the graph structure changes, leading to significant computational overhead. Recent methods, like GEM, overcome these limitations by using generator neural networks for improved scalability. However, empirical comparisons have mostly focused on small datasets, limiting real-world applicability. In this work, we introduce GEM+, which integrates AIM's adaptive measurement framework with GEM's scalable generator network. Our experiments show that GEM+ outperforms AIM in both utility and scalability, delivering state-of-the-art results while efficiently handling datasets with over a hundred columns, where AIM fails due to memory and computational overheads.

---

## 195. TomoGraphView: 3D Medical Image Classification with Omnidirectional Slice Representations and Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.09605v1](http://arxiv.org/abs/2511.09605v1)

**作者:** Johannes Kiechle, Stefan M. Fischer, Daniel M. Lang, Cosmin I. Bercea, Matthew J. Nyflot, Lina Felsner, Julia A. Schnabel, Jan C. Peeken

**发布时间:** 2025-11-12

**备注:** Preprint submitted to Medical Image Analysis (MedIA)

### GPT解析

### 总结

论文提出了一种名为TomoGraphView的新框架，用于解决3D医学图像分类中的挑战，通过整合全方位体积切片和基于球图的特征聚合来优化特征提取过程。

### 背景

随着医学断层扫描检查数量增加，需要自动化方法提取影像特征以辅助下游任务如肿瘤表征，同时帮助医生管理工作量。3D医学图像分类因体积数据中复杂的空间关系和长程依赖性而具有挑战性，且缺乏3D大规模多模态数据集限制了基础模型发展。

### 目的

解决现有2D视觉基础模型应用于3D医学图像时的局限性，特别是传统切片策略无法充分捕捉目标结构空间范围以及现有聚合方法导致空间连贯性丧失的问题。

### 方法

提出TomoGraphView框架，整合全方位体积切片与基于球图的特征聚合，克服传统基于标准平面切片的局限性，并保持体积结构的空间连贯性。

### 主要发现

现有通过基于切片分解将2D模型应用于3D体积的方法不够优化；传统体积切片策略依赖标准平面，当目标结构与标准观察平面不对齐时无法充分捕捉目标结构空间范围；现有逐片聚合策略很少考虑保持体积结构，导致切片间空间连贯性丧失。

### 结论

TomoGraphView框架通过全方位体积切片和基于球图的特征聚合，能够更有效地提取3D医学图像特征，为医学影像分析提供了新的解决方案。

### 翻译

随着医学断层扫描检查数量的增加，开发能够提取全面影像特征的自动化方法已成为必要，以促进肿瘤表征等下游任务，同时协助医生管理日益增长的工作量。然而，由于体积数据中固有的复杂空间关系和长程依赖性，3D医学图像分类仍然是一项具有挑战性的任务。从零开始训练模型面临数据量不足的问题，且缺乏3D大规模多模态数据集限制了3D医学影像基础模型的发展。然而，最近的研究强调了2D视觉基础模型（最初在自然图像上训练）作为医学图像分析强大特征提取器的潜力。尽管取得了这些进展，但现有通过基于切片的分解将2D模型应用于3D体积的方法仍然不够优化。依赖轴向、矢状或冠状等标准平面的传统体积切片策略，当目标结构与标准观察平面不对齐时，可能无法充分捕捉目标结构的空间范围。此外，现有的逐片聚合策略很少考虑保持体积结构，导致切片间的空间连贯性丧失。为了克服这些局限性，我们提出了TomoGraphView，一个将全方位体积切片与基于球图的特征聚合相结合的新框架。我们在http://github.com/compai-lab/2025-MedIA-kiechle公开了我们的代码库，并在https://pypi.org/project/OmniSlicer提供了一个用于全方位体积切片的用户友好库。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D医学图像分类中的挑战，特别是如何有效处理体积数据中的复杂空间关系和长距离依赖关系。这个问题很重要，因为随着医学断层扫描检查数量增加，需要自动化方法提取全面成像特征以支持肿瘤表征等下游任务，同时减轻医生的工作负担。此外，3D医学图像分析受限于数据量不足和计算资源需求，现有方法无法充分捕捉与标准 viewing 平面不对齐的结构信息，导致分类性能不佳。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统体积切片策略仅使用轴向、冠状、矢状等标准平面，无法捕捉与这些平面不对齐的结构；而现有切片聚合方法很少保留体积结构的空间连贯性。基于此，作者设计了一个结合全方位体积切片和球形图神经网络特征聚合的框架。该方法借鉴了2D视觉基础模型（如DINOv2）作为特征提取器，利用图神经网络建模切片间关系，并采用球面点分布优化（基于Thomson问题）和Delaunay三角剖分构建图拓扑，从而在保留3D空间结构的同时利用2D模型的强大特征提取能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过全方位体积切片和基于球形图的特征聚合来保留3D医学图像的空间结构，突破传统标准平面的限制，更全面地捕捉体积结构信息。整体流程包括：1) 将体积数据嵌入球体，在球面上均匀采样点（包括3个固定代表标准平面和N-3个优化点）；2) 从这些方向提取切片并使用DINOv2编码为特征向量；3) 通过Delaunay三角剖分构建球形图拓扑，并为边分配基于距离的权重；4) 使用图神经网络聚合节点特征，通过均值池化获得全局表示，最后进行分类预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 全方位体积切片策略，突破传统标准平面限制，从球面均匀分布点导出非标准视图；2) 基于球形图的特征聚合方法，利用球形图拓扑和图神经网络显式建模切片关系；3) 综合框架在多个肿瘤学数据集上验证有效性。相比之前工作，不同之处在于：传统切片方法仅使用标准平面，而本文方法能捕捉更全面结构信息（AUROC提升0.0453）；其他聚合方法（如MLP、LSTM、Transformer）未显式建模空间关系，本文方法通过图结构保留空间连贯性（AUROC再提升0.0174）；甚至在多个数据集上超越了3D大规模预训练模型，为体积分析提供了强大框架。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TomoGraphView通过全方位体积切片和基于球形图的神经网络特征聚合，有效解决了3D医学图像分类中传统方法无法充分捕捉体积结构和空间关系的问题，在多个肿瘤学数据集上超越了现有方法的性能。'}


### 论文摘要

The growing number of medical tomography examinations has necessitated the development of automated methods capable of extracting comprehensive imaging features to facilitate downstream tasks such as tumor characterization, while assisting physicians in managing their growing workload. However, 3D medical image classification remains a challenging task due to the complex spatial relationships and long-range dependencies inherent in volumetric data. Training models from scratch suffers from low data regimes, and the absence of 3D large-scale multimodal datasets has limited the development of 3D medical imaging foundation models. Recent studies, however, have highlighted the potential of 2D vision foundation models, originally trained on natural images, as powerful feature extractors for medical image analysis. Despite these advances, existing approaches that apply 2D models to 3D volumes via slice-based decomposition remain suboptimal. Conventional volume slicing strategies, which rely on canonical planes such as axial, sagittal, or coronal, may inadequately capture the spatial extent of target structures when these are misaligned with standardized viewing planes. Furthermore, existing slice-wise aggregation strategies rarely account for preserving the volumetric structure, resulting in a loss of spatial coherence across slices. To overcome these limitations, we propose TomoGraphView, a novel framework that integrates omnidirectional volume slicing with spherical graph-based feature aggregation. We publicly share our accessible code base at http://github.com/compai-lab/2025-MedIA-kiechle and provide a user-friendly library for omnidirectional volume slicing at https://pypi.org/project/OmniSlicer.

---

## 196. DynamicRTL: RTL Representation Learning for Dynamic Circuit Behavior

**论文链接:** [http://arxiv.org/abs/2511.09593v1](http://arxiv.org/abs/2511.09593v1)

**作者:** Ruiyang Ma, Yunhao Zhou, Yipeng Wang, Yi Liu, Zhengyuan Shi, Ziyang Zheng, Kexin Chen, Zhiqiang He, Lingwei Yan, Gang Chen, Qiang Xu, Guojie Luo

**发布时间:** 2025-11-12

**备注:** Accepted by AAAI'2026

### GPT解析

### 总结

本文提出了一种名为DR-GNN的新型方法，通过结合电路的静态结构和多周期执行行为来学习RTL电路表征，解决了现有GNN模型无法捕捉电路运行时行为的问题。

### 背景

目前使用图神经网络(GNNs)学习电路表征的研究主要关注电路的静态特性，无法捕捉电路的运行时行为，而这对于电路验证和优化等任务至关重要。

### 目的

引入DR-GNN(DynamicRTL-GNN)方法，通过结合静态结构和多周期执行行为来学习RTL电路表征，以捕捉电路的动态依赖和运行时执行行为。

### 方法

DR-GNN利用操作级别的控制数据流图(CDFG)来表示寄存器传输级(RTL)电路。作者构建了第一个全面的动态电路数据集，包含超过6,300个Verilog设计和63,000个仿真轨迹，用于训练和评估DR-GNN。

### 主要发现

DR-GNN在分支命中预测和翻转率预测方面优于现有模型。此外，其学习到的表征可以有效地转移到相关的动态电路任务中，在功率估计和断言预测方面取得了良好的性能。

### 结论

DR-GNN通过结合静态结构和动态行为，能够更好地学习和表示电路，特别是在需要考虑运行时行为的任务中表现优异。

### 翻译

目前有越来越多的研究使用图神经网络(GNNs)来学习电路表征，主要关注其静态特性。然而，这些模型无法捕捉电路的运行时行为，这对于电路验证和优化等任务至关重要。为了解决这一局限，我们引入了DR-GNN(DynamicRTL-GNN)，一种新颖的方法，通过结合静态结构和多周期执行行为来学习RTL电路表征。DR-GNN利用操作级别的控制数据流图(CDFG)来表示寄存器传输级(RTL)电路，使模型能够捕捉动态依赖和运行时执行。为了训练和评估DR-GNN，我们构建了第一个全面的动态电路数据集，包含超过6,300个Verilog设计和63,000个仿真轨迹。我们的结果表明，DR-GNN在分支命中预测和翻转率预测方面优于现有模型。此外，其学习到的表征可以有效地转移到相关的动态电路任务中，在功率估计和断言预测方面取得了良好的性能。


### 论文摘要

There is a growing body of work on using Graph Neural Networks (GNNs) to learn representations of circuits, focusing primarily on their static characteristics. However, these models fail to capture circuit runtime behavior, which is crucial for tasks like circuit verification and optimization. To address this limitation, we introduce DR-GNN (DynamicRTL-GNN), a novel approach that learns RTL circuit representations by incorporating both static structures and multi-cycle execution behaviors. DR-GNN leverages an operator-level Control Data Flow Graph (CDFG) to represent Register Transfer Level (RTL) circuits, enabling the model to capture dynamic dependencies and runtime execution. To train and evaluate DR-GNN, we build the first comprehensive dynamic circuit dataset, comprising over 6,300 Verilog designs and 63,000 simulation traces. Our results demonstrate that DR-GNN outperforms existing models in branch hit prediction and toggle rate prediction. Furthermore, its learned representations transfer effectively to related dynamic circuit tasks, achieving strong performance in power estimation and assertion prediction.

---

## 197. A Distributed Training Architecture For Combinatorial Optimization

**论文链接:** [http://arxiv.org/abs/2511.09261v1](http://arxiv.org/abs/2511.09261v1)

**作者:** Yuyao Long

**发布时间:** 2025-11-12

### GPT解析

### 总结

本文提出了一种分布式图神经网络训练框架，用于解决组合优化问题，提高了解决方案质量和计算效率，同时增强了模型的可扩展性。

### 背景

图神经网络已被广泛应用于解决组合优化问题，但现有方法在处理复杂图时准确性有限且可扩展性差，因为完整训练需要一次性加载整个邻接矩阵和所有嵌入，可能导致单机内存不足。

### 目的

开发一个分布式GNN训练框架，解决现有方法在处理复杂图时的准确性和可扩展性问题，使其适用于大规模场景。

### 方法

首先将大图划分为多个小子图；然后对各个子图进行完整训练，为高效局部优化奠定基础；最后使用强化学习根据GNN输出来采取行动，确保跨节点之间的约束可以被学习。

### 主要发现

在真实大规模社交网络数据集(如Facebook、Youtube)和合成的高复杂度图上进行的实验表明，该框架在解决方案质量和计算效率方面都优于最先进的方法，且在大图实例上验证了模型的可扩展性。

### 结论

分布式GNN训练框架能有效解决组合优化问题，特别是在处理大规模复杂图时，提供了更好的解决方案质量和计算效率。

### 翻译

近年来，图神经网络已被广泛应用于解决组合优化问题。然而，现有方法在处理复杂图时仍存在准确性有限的问题，并且可扩展性差，因为完整训练需要一次性加载整个邻接矩阵和所有嵌入，这可能导致单机内存不足。这一限制显著限制了它们在大规模场景中的适用性。为应对这些挑战，我们提出了一种用于组合优化的分布式GNN训练框架。具体而言，首先将大图划分为多个小子图。然后对各个子图进行完整训练，为高效局部优化奠定基础。最后，使用强化学习根据GNN输出来采取行动，确保跨节点之间的约束可以被学习。在真实大规模社交网络数据集(如Facebook、Youtube)和合成的高复杂度图上进行了广泛实验，结果表明我们的框架在解决方案质量和计算效率方面都优于最先进的方法。此外，在大图实例上的实验也验证了模型的可扩展性。


### 论文摘要

In recent years, graph neural networks (GNNs) have been widely applied in tackling combinatorial optimization problems. However, existing methods still suffer from limited accuracy when addressing that on complex graphs and exhibit poor scalability, since full training requires loading the whole adjacent matrix and all embeddings at a time, the it may results in out of memory of a single machine. This limitation significantly restricts their applicability to large-scale scenarios. To address these challenges, we propose a distributed GNN-based training framework for combinatorial optimization. In details, firstly, large graph is partition into several small subgraphs. Then the individual subgraphs are full trained, providing a foundation for efficient local optimization. Finally, reinforcement learning (RL) are employed to take actions according to GNN output, to make sure the restrictions between cross nodes can be learned. Extensive experiments are conducted on both real large-scale social network datasets (e.g., Facebook, Youtube) and synthetically generated high-complexity graphs, which demonstrate that our framework outperforms state-of-the-art approaches in both solution quality and computational efficiency. Moreover, the experiments on large graph instances also validate the scalability of the model.

---

## 198. CoCo-MILP: Inter-Variable Contrastive and Intra-Constraint Competitive MILP Solution Prediction

**论文链接:** [http://arxiv.org/abs/2511.09209v1](http://arxiv.org/abs/2511.09209v1)

**作者:** Tianle Pu, Jianing Li, Yingying Gao, Shixuan Liu, Zijie Geng, Haoyang Liu, Chao Chen, Changjun Fan

**发布时间:** 2025-11-12

### GPT解析

### 总结

本研究提出了一种名为CoCo-MILP的新方法，通过结合变量间对比学习和约束内竞争机制，改进了混合整数线性规划问题的求解效果，解决了现有方法与MILP问题内在结构不匹配的问题。

### 背景

混合整数线性规划(MILP)是组合优化的基石，但求解大规模实例仍面临显著计算挑战。图神经网络(GNNs)在加速MILP求解器方面显示出潜力，但现有方法在两个层面与MILP问题的内在结构不匹配：在目标层面，二元交叉熵(BCE)损失独立处理变量，忽略相对优先级；在架构层面，标准GNN消息传递平滑变量间表示，无法捕捉约束内的自然竞争关系。

### 目的

解决现有图神经网络方法在处理MILP问题时存在的目标函数和架构层面的不匹配问题，提出能更好捕捉MILP问题内在结构的新方法。

### 方法

提出CoCo-MILP方法，明确建模变量间对比和约束内竞争关系。在目标层面引入变量间对比损失(VCL)，最大化分配值为1和0的变量之间的嵌入边界；在架构层面设计约束内竞争GNN层，学习区分约束内竞争变量的表示，捕捉其排他性质。

### 主要发现

实验结果表明，CoCo-MILP显著优于现有基于学习的方法，与传统求解器相比最多减少68.12%的解差距。代码已在https://github.com/happypu326/CoCo-MILP公开。

### 结论

CoCo-MILP通过引入变量间对比学习和约束内竞争机制，有效解决了现有方法的不匹配问题，显著提高了混合整数线性规划问题的求解效率和质量。

### 翻译

混合整数线性规划(MILP)是组合优化的基石，但求解大规模实例仍是一个重大的计算挑战。最近，图神经网络(GNNs)在通过预测高质量解来加速MILP求解器方面显示出潜力。然而，我们发现在两个层面上，现有方法与MILP问题的内在结构不匹配。在学习目标层面，二元交叉熵(BCE)损失独立处理变量，忽略了它们的相对优先级，并产生合理的logits。在模型架构层面，标准GNN消息传递本质上平滑了变量间的表示，无法捕捉约束内的自然竞争关系。为了应对这些挑战，我们提出了CoCo-MILP，它明确建模变量间的对比和约束内的竞争关系，用于高级MILP解预测。在目标层面，CoCo-MILP引入了变量间对比损失(VCL)，明确最大化分配值为1和0的变量之间的嵌入边界。在架构层面，我们设计了一种约束内竞争GNN层，它不是同质化特征，而是学习区分约束内竞争变量的表示，捕捉它们的排他性质。在标准基准上的实验结果表明，CoCo-MILP显著优于现有的基于学习的方法，与传统求解器相比最多减少68.12%的解差距。我们的代码可在https://github.com/happypu326/CoCo-MILP获取。


### 论文摘要

Mixed-Integer Linear Programming (MILP) is a cornerstone of combinatorial optimization, yet solving large-scale instances remains a significant computational challenge. Recently, Graph Neural Networks (GNNs) have shown promise in accelerating MILP solvers by predicting high-quality solutions. However, we identify that existing methods misalign with the intrinsic structure of MILP problems at two levels. At the leaning objective level, the Binary Cross-Entropy (BCE) loss treats variables independently, neglecting their relative priority and yielding plausible logits. At the model architecture level, standard GNN message passing inherently smooths the representations across variables, missing the natural competitive relationships within constraints. To address these challenges, we propose CoCo-MILP, which explicitly models inter-variable Contrast and intra-constraint Competition for advanced MILP solution prediction. At the objective level, CoCo-MILP introduces the Inter-Variable Contrastive Loss (VCL), which explicitly maximizes the embedding margin between variables assigned one versus zero. At the architectural level, we design an Intra-Constraint Competitive GNN layer that, instead of homogenizing features, learns to differentiate representations of competing variables within a constraint, capturing their exclusionary nature. Experimental results on standard benchmarks demonstrate that CoCo-MILP significantly outperforms existing learning-based approaches, reducing the solution gap by up to 68.12% compared to traditional solvers. Our code is available at https://github.com/happypu326/CoCo-MILP.

---

## 199. Graph Neural Field with Spatial-Correlation Augmentation for HRTF Personalization

**论文链接:** [http://arxiv.org/abs/2511.10697v1](http://arxiv.org/abs/2511.10697v1)

**作者:** De Hu, Junsheng Hu, Cuicui Jiang

**发布时间:** 2025-11-12

### GPT解析

### 总结

本文提出了一种名为GraphNF-SCA的图神经网络方法，用于解决VR/AR设备中高质量HRTF的个性化问题，该方法通过建模空间相关性实现了对未见受试者的个体化HRTF生成。

### 背景

在VR/AR设备上实现沉浸式空间音频渲染需要高质量的HRTF，但HRTF通常因人而异且依赖于位置，其测量过程耗时且繁琐。

### 目的

开发一种方法，能够为未见过的受试者生成个体化的HRTF，避免传统测量的繁琐过程。

### 方法

提出Graph Neural Field with Spatial-Correlation Augmentation (GraphNF-SCA)，包含三个关键组件：1) HRTF个性化模块，使用编码器-解码器架构的图神经网络预测目标受试者的HRTF；2) HRTF上采样模块，使用另一个图神经网络建模HRTF之间的空间相关性；3) 微调阶段，利用HRTF-P模块的输出增强预测HRTF的空间一致性。

### 主要发现

GraphNF-SCA有效利用了HRTF之间的固有空间相关性，提高了HRTF个性化的性能，优于现有不建模空间相关性的方法。

### 结论

实验结果表明，GraphNF-SCA在HRTF个性化任务上达到了最先进的水平。

### 翻译

为了在VR/AR设备上实现沉浸式空间音频渲染，高质量的头相关传输函数（HRTF）是必不可少的。通常，HRTF因人而异且依赖于位置，其测量过程耗时且繁琐。为解决这一挑战，我们提出了带有空间相关性增强的图神经网络场（GraphNF-SCA）用于HRTF个性化，可用于为未见过的受试者生成个体化的HRTF。GraphNF-SCA包含三个关键组件：HRTF个性化（HRTF-P）模块、HRTF上采样（HRTF-U）模块和一个微调阶段。在HRTF-P模块中，我们通过具有编码器-解码器架构的图神经网络预测目标受试者的HRTF，其中编码器提取通用特征，解码器整合目标相关特征并生成个性化的HRTF。HRTF-U模块使用另一个图神经网络来建模HRTF之间的空间相关性。该模块使用HRTF-P模块的输出进行微调，从而增强预测HRTF的空间一致性。与现有方法不同，那些方法按位置估计个体HRTF而不对空间相关性进行建模，而GraphNF-SCA有效利用了HRTF之间的固有空间相关性，提高了HRTF个性化的性能。实验结果表明，GraphNF-SCA达到了最先进的结果。


### 论文摘要

To achieve immersive spatial audio rendering on VR/AR devices, high-quality Head-Related Transfer Functions (HRTFs) are essential. In general, HRTFs are subject-dependent and position-dependent, and their measurement is time-consuming and tedious. To address this challenge, we propose the Graph Neural Field with Spatial-Correlation Augmentation (GraphNF-SCA) for HRTF personalization, which can be used to generate individual HRTFs for unseen subjects. The GraphNF-SCA consists of three key components: an HRTF personalization (HRTF-P) module, an HRTF upsampling (HRTF-U) module, and a fine-tuning stage. In the HRTF-P module, we predict HRTFs of the target subject via the Graph Neural Network (GNN) with an encoder-decoder architecture, where the encoder extracts universal features and the decoder incorporates the target-relevant features and produces individualized HRTFs. The HRTF-U module employs another GNN to model spatial correlations across HRTFs. This module is fine-tuned using the output of the HRTF-P module, thereby enhancing the spatial consistency of the predicted HRTFs. Unlike existing methods that estimate individual HRTFs position-by-position without spatial correlation modeling, the GraphNF-SCA effectively leverages inherent spatial correlations across HRTFs to enhance the performance of HRTF personalization. Experimental results demonstrate that the GraphNF-SCA achieves state-of-the-art results.

---

## 200. FsimNNs: An Open-Source Graph Neural Network Platform for SEU Simulation-based Fault Injection

**论文链接:** [http://arxiv.org/abs/2511.09131v1](http://arxiv.org/abs/2511.09131v1)

**作者:** Li Lu, Jianan Wen, Milos Krstic

**发布时间:** 2025-11-12

### GPT解析

### 总结

本文提出了一种基于时空图神经网络(STGNNs)的开源平台，用于加速单粒子翻转(SEU)故障仿真，解决了传统基于仿真的故障注入方法计算成本高的问题。

### 背景

基于仿真的故障注入是评估电路对单粒子翻转(SEUs)脆弱性的广泛采用的方法，但其计算成本随着电路复杂性的增加而显著增长。

### 目的

开发一个利用时空图神经网络(STGNNs)加速SEU故障仿真的开源平台，以降低计算成本。

### 方法

构建了一个包含三种STGNN架构的开源平台，这些架构融入了空洞空间金字塔池化(ASPP)和注意力机制等先进组件；同时，从六个不同复杂度的开源电路构建了SEU故障仿真数据集；在这些数据集上分析和比较了STGNN模型的预测能力，并评估了其在多个测试用例上的泛化能力。

### 主要发现

STGNNs能够有效加速SEU故障仿真过程；先进组件(如ASPP和注意力机制)提高了时空特征提取能力；STGNNs在多个测试用例上表现出良好的泛化能力。

### 结论

开发的平台和数据集已作为开源发布，支持可重复性并促进进一步研究，可在https://github.com/luli2021/FsimNNs获取。

### 翻译

基于仿真的故障注入是评估电路对单粒子翻转(SEUs)脆弱性的广泛采用的方法；然而，其计算成本随着电路复杂性的增加而显著增长。为解决这一限制，本文引入了一个利用时空图神经网络(STGNNs)加速SEU故障仿真的开源平台。该平台包含三种STGNN架构，融入了空洞空间金字塔池化(ASPP)和注意力机制等先进组件，从而提高了时空特征提取能力。此外，从六个不同复杂度的开源电路构建了SEU故障仿真数据集，为性能评估提供了全面的基准。在这些数据集上分析和比较了STGNN模型的预测能力。此外，为进一步研究该方法的效率，我们在多个测试用例上评估了STGNNs的预测能力，并讨论了它们的泛化能力。开发的平台和数据集已作为开源发布，支持可重复性和进一步研究，网址为https://github.com/luli2021/FsimNNs。


### 论文摘要

Simulation-based fault injection is a widely adopted methodology for assessing circuit vulnerability to Single Event Upsets (SEUs); however, its computational cost grows significantly with circuit complexity. To address this limitation, this work introduces an open-source platform that exploits Spatio-Temporal Graph Neural Networks (STGNNs) to accelerate SEU fault simulation. The platform includes three STGNN architectures incorporating advanced components such as Atrous Spatial Pyramid Pooling (ASPP) and attention mechanisms, thereby improving spatio-temporal feature extraction. In addition, SEU fault simulation datasets are constructed from six open-source circuits with varying levels of complexity, providing a comprehensive benchmark for performance evaluation. The predictive capability of the STGNN models is analyzed and compared on these datasets. Moreover, to further investigate the efficiency of the approach, we evaluate the predictive capability of STGNNs across multiple test cases and discuss their generalization capability. The developed platform and datasets are released as open-source to support reproducibility and further research on https://github.com/luli2021/FsimNNs.

---

## 201. Towards a Generalisable Cyber Defence Agent for Real-World Computer Networks

**论文链接:** [http://arxiv.org/abs/2511.09114v1](http://arxiv.org/abs/2511.09114v1)

**作者:** Tim Dudman, Martyn Bull

**发布时间:** 2025-11-12

**备注:** CAMLIS 2025. To be published in the Proceedings of Machine Learning Research (PMLR)

### GPT解析

### 总结

本研究介绍了TERLA（强化学习代理的拓扑扩展），一种使自主网络防御代理能够在不同拓扑和大小的网络中实现泛化而无需重新训练的新方法。

### 背景

深度强化学习在自主网络防御领域的最新进展已产生能够成功防御模拟计算机网络免受攻击的代理，但这些代理在面对不同拓扑或大小时通常需要重新训练，不适合现实网络环境。

### 目的

引入TERLA为强化学习代理提供拓扑扩展，使其能够防御具有不同拓扑和大小的网络，而无需重新训练。

### 方法

使用异构图神经网络层生成固定大小的潜在嵌入表示网络状态，结合简化、固定大小且可解释的动作空间，将TERLA应用于PPO代理模型，并在具有真实网络特征的CAGE环境中测试。

### 主要发现

TERLA代理保留了原始PPO代理的防御性能，同时提高了动作效率；所有TERLA代理采用相同网络无关架构；单个TERLA代理可成功防御不同拓扑和大小的网络段。

### 结论

TERLA成功解决了现有网络防御代理在面对网络拓扑和大小时变化的泛化问题，无需重新训练即可提供有效的网络防御能力。

### 翻译

近期的深度强化学习在自主网络防御领域的进展已经产生了能够成功防御模拟计算机网络免受网络攻击的代理。然而，许多这样的代理需要重新训练才能防御具有不同拓扑或大小的网络，这使得它们不太适合拓扑和大小可能随时间变化的现实网络。在本研究中，我们引入了一组新的强化学习代理拓扑扩展(TERLA)，这些扩展为防御具有不同拓扑和大小的网络提供了泛化能力，而无需重新训练。我们的方法涉及使用异构图神经网络层生成表示观察网络状态的固定大小潜在嵌入。这个表示学习阶段与简化的、固定大小的、语义明确且可解释的动作空间相结合。我们将TERLA应用于标准的深度强化学习近端策略优化(PPO)代理模型，并为了减少模拟到现实的差距，我们在网络自主实验环境(CAGE)挑战4中进行了研究。这个网络操作研究健身房环境具有许多现实网络的特征，如真实的入侵检测系统(IDS)事件和多个防御不同拓扑和大小的网络段的代理。TERLA代理保留了原始PPO代理的防御性能，同时显示出改进的动作效率。通过展示所有TERLA代理具有相同的网络无关神经网络架构，并通过多次部署单个TERLA代理来防御具有不同拓扑和大小的网络段，展示了改进的防御性能和效率，从而证明了其泛化能力。


### 论文摘要

Recent advances in deep reinforcement learning for autonomous cyber defence have resulted in agents that can successfully defend simulated computer networks against cyber-attacks. However, many of these agents would need retraining to defend networks with differing topology or size, making them poorly suited to real-world networks where topology and size can vary over time. In this research we introduce a novel set of Topological Extensions for Reinforcement Learning Agents (TERLA) that provide generalisability for the defence of networks with differing topology and size, without the need for retraining. Our approach involves the use of heterogeneous graph neural network layers to produce a fixed-size latent embedding representing the observed network state. This representation learning stage is coupled with a reduced, fixed-size, semantically meaningful and interpretable action space. We apply TERLA to a standard deep reinforcement learning Proximal Policy Optimisation (PPO) agent model, and to reduce the sim-to-real gap, conduct our research using Cyber Autonomy Gym for Experimentation (CAGE) Challenge 4. This Cyber Operations Research Gym environment has many of the features of a real-world network, such as realistic Intrusion Detection System (IDS) events and multiple agents defending network segments of differing topology and size. TERLA agents retain the defensive performance of vanilla PPO agents whilst showing improved action efficiency. Generalisability has been demonstrated by showing that all TERLA agents have the same network-agnostic neural network architecture, and by deploying a single TERLA agent multiple times to defend network segments with differing topology and size, showing improved defensive performance and efficiency.

---

## 202. Efficient Distributed Exact Subgraph Matching via GNN-PE: Load Balancing, Cache Optimization, and Query Plan Ranking

**论文链接:** [http://arxiv.org/abs/2511.09052v1](http://arxiv.org/abs/2511.09052v1)

**作者:** Yu Wang, Hui Wang, Jiake Ge, Xin Wang

**发布时间:** 2025-11-12

**备注:** 10 pages

### GPT解析

### 总结

本文提出了一种针对分布式环境下大规模图精确子图匹配的改进方法，通过三个核心创新解决了现有GNN-PE框架在分布式系统中可扩展性和优化不足的问题。

### 背景

大规模图上的精确子图匹配因高计算复杂度和分布式系统限制而面临挑战，现有的基于GNN的路径嵌入框架虽在单机上高效，但在分布式环境中缺乏可扩展性和优化。

### 目的

扩展GNN-PE框架至分布式系统，解决其可扩展性和优化不足的问题，提高分布式环境下子图匹配的效率和稳定性。

### 方法

提出三个核心创新：(1)轻量级动态关联感知负载均衡和热迁移机制；(2)基于在线增量学习的多GPU协作动态缓存策略；(3)由优势嵌入剪枝驱动的查询计划排序方法。并通过METIS分区、并行离线预处理和轻量级元数据管理实现分布式优化。

### 主要发现

该方法在分布式场景(数十台机器)中实现了'最小边切割+负载均衡+不间断查询'，显著提高了分布式子图匹配的效率和稳定性。

### 结论

通过三个核心创新和技术实现，成功解决了GNN-PE框架在分布式环境下的可扩展性和优化问题，为大规模图精确子图匹配提供了高效稳定的解决方案。

### 翻译

Exact subgraph matching on large-scale graphs remains a challenging problem due to high computational complexity and distributed system constraints. Existing GNN-based path embedding frameworks achieve efficient exact matching on single machines but lack scalability and optimization for distributed environments. To address this gap, we propose three core innovations to extend GNN-PE to distributed systems: (1) a lightweight dynamic correlation-aware load balancing and hot migration mechanism that fuses multi-dimensional metrics (CPU, communication, memory) and guarantees index consistency; (2) an online incremental learning-based multi-GPU collaborative dynamic caching strategy with heterogeneous GPU adaptation and graph-structure-aware replacement; (3) a query plan ranking method driven by dominance embedding pruning potential (PE-score) that optimizes execution order. Through METIS partitioning, parallel offline preprocessing, and lightweight metadata management, our approach achieves 'minimum edge cut + load balancing + non-interruptible queries' in distributed scenarios (tens of machines), significantly improving the efficiency and stability of distributed subgraph matching.


### 论文摘要

Exact subgraph matching on large-scale graphs remains a challenging problem due to high computational complexity and distributed system constraints. Existing GNN-based path embedding (GNN-PE) frameworks achieve efficient exact matching on single machines but lack scalability and optimization for distributed environments. To address this gap, we propose three core innovations to extend GNN-PE to distributed systems: (1) a lightweight dynamic correlation-aware load balancing and hot migration mechanism that fuses multi-dimensional metrics (CPU, communication, memory) and guarantees index consistency; (2) an online incremental learning-based multi-GPU collaborative dynamic caching strategy with heterogeneous GPU adaptation and graph-structure-aware replacement; (3) a query plan ranking method driven by dominance embedding pruning potential (PE-score) that optimizes execution order. Through METIS partitioning, parallel offline preprocessing, and lightweight metadata management, our approach achieves "minimum edge cut + load balancing + non-interruptible queries" in distributed scenarios (tens of machines), significantly improving the efficiency and stability of distributed subgraph matching.

---

## 203. GeoGNN: Quantifying and Mitigating Semantic Drift in Text-Attributed Graphs

**论文链接:** [http://arxiv.org/abs/2511.09042v1](http://arxiv.org/abs/2511.09042v1)

**作者:** Liangwei Yang, Jing Ma, Jianguo Zhang, Zhiwei Liu, Jielin Qiu, Shirley Kokane, Shiyu Wang, Haolin Chen, Rithesh Murthy, Ming Zhu, Huan Wang, Weiran Yao, Caiming Xiong, Shelby Heinecke

**发布时间:** 2025-11-12

**备注:** 10 pages

### GPT解析

### 总结

本文研究了文本属性图上的图神经网络中的语义漂移问题，提出了一种基于流形感知的测地线聚合方法，开发了GeoGNN模型，有效减轻了语义漂移问题，并在多个基准数据集上表现优于基线模型。

### 背景

现有的文本属性图上的图神经网络通常使用预训练语言模型编码节点文本，并通过线性邻域聚合传播这些嵌入表示。然而，现代预训练语言模型的表示空间是非线性和几何结构化的，文本嵌入位于弯曲的语义流形上，而非平坦的欧几里得空间。在这样的流形上进行线性聚合会扭曲几何结构，导致语义漂移现象。

### 目的

定量研究语义漂移问题，分析不同聚合机制对流形结构的影响，并设计一种能够保持表示与语义流形一致性的聚合方法。

### 方法

1. 引入基于局部PCA的度量指标来量化语义漂移程度；2. 提出Geodesic Aggregation，一种流形感知的聚合机制，通过单位球上的对数-指数映射沿测地线聚合邻居信息；3. 开发GeoGNN，一个集成了球形注意力和流形插值的实用实例。

### 主要发现

1. 线性聚合在语义流形上会导致几何扭曲和语义漂移；2. 提出的测地线聚合方法能够保持表示与语义流形的一致性；3. GeoGNN显著减轻了语义漂移问题，并在多个基准数据集上表现优于强基线模型。

### 结论

流形感知的聚合在文本属性图学习中具有重要作用，GeoGNN通过保持表示与语义流形的一致性，有效解决了传统线性聚合方法中的语义漂移问题。

### 翻译

文本属性图上的图神经网络通常使用预训练语言模型编码节点文本，并通过线性邻域聚合传播这些嵌入表示。然而，现代预训练语言模型的表示空间是非线性和几何结构化的，其中文本嵌入位于弯曲的语义流形上，而非平坦的欧几里得空间。在这样的流形上进行线性聚合不可避免地会扭曲几何结构并导致语义漂移现象——聚合表示偏离内在流形，失去语义保真度和表达能力。为了定量研究这一问题，这项工作引入了一种基于局部PCA的度量指标，用于衡量语义漂移程度，并提供了首个定量框架来分析不同聚合机制对流形结构的影响。基于这些见解，我们提出了Geodesic Aggregation，一种流形感知的聚合机制，通过单位球上的对数-指数映射沿测地线聚合邻居信息，确保表示在消息传递过程中保持对语义流形的忠实。我们进一步开发了GeoGNN，这是一个集成了球形注意力和流形插值的实用实例。在四个基准数据集和多种文本编码器上的广泛实验表明，GeoGNN显著减轻了语义漂移问题，并持续优于强基线模型，确立了流形感知聚合在文本属性图学习中的重要性。


### 论文摘要

Graph neural networks (GNNs) on text--attributed graphs (TAGs) typically encode node texts using pretrained language models (PLMs) and propagate these embeddings through linear neighborhood aggregation. However, the representation spaces of modern PLMs are highly non--linear and geometrically structured, where textual embeddings reside on curved semantic manifolds rather than flat Euclidean spaces. Linear aggregation on such manifolds inevitably distorts geometry and causes semantic drift--a phenomenon where aggregated representations deviate from the intrinsic manifold, losing semantic fidelity and expressive power. To quantitatively investigate this problem, this work introduces a local PCA--based metric that measures the degree of semantic drift and provides the first quantitative framework to analyze how different aggregation mechanisms affect manifold structure. Building upon these insights, we propose Geodesic Aggregation, a manifold--aware mechanism that aggregates neighbor information along geodesics via log--exp mappings on the unit sphere, ensuring that representations remain faithful to the semantic manifold during message passing. We further develop GeoGNN, a practical instantiation that integrates spherical attention with manifold interpolation. Extensive experiments across four benchmark datasets and multiple text encoders show that GeoGNN substantially mitigates semantic drift and consistently outperforms strong baselines, establishing the importance of manifold--aware aggregation in text--attributed graph learning.

---

## 204. Heterogeneous Graph Neural Networks for Assumption-Based Argumentation

**论文链接:** [http://arxiv.org/abs/2511.08982v1](http://arxiv.org/abs/2511.08982v1)

**作者:** Preesha Gehlot, Anna Rapberger, Fabrizio Russo, Francesca Toni

**发布时间:** 2025-11-12

**备注:** Accepted to AAAI2026. Version with Appendix

### GPT解析

### 总结

本文提出了一种基于图神经网络的方法来近似假设基础论证(ABA)中的可信接受问题，解决了ABA框架下稳定语义扩展计算的不可行性。

### 背景

假设基础论证(ABA)是一种强大的结构化论证形式化方法，但在大框架下稳定语义的扩展的精确计算是不可行的。

### 目的

提出第一个基于图神经网络(GNN)的方法来近似ABA中的可信接受(credulous acceptance)。

### 方法

通过依赖图表示建模ABA框架，将假设、主张和规则编码为节点，使用异构边标签区分关系；提出ABAGCN和ABAGAT两种GNN架构，分别使用残差异构卷积层和注意力层学习节点嵌入；在ICCMA 2023基准测试上训练模型，使用合成的ABAF增强，通过贝叶斯搜索优化超参数。

### 主要发现

ABAGCN和ABAGAT都优于改编的来自抽象论证文献的最先进GNN基线，在ICCMA实例上实现了高达0.71的节点级F1分数；开发的扩展重建算法在小ABAF上稳定扩展的F1超过0.85，在大框架上保持约0.58的F1。

### 结论

这项工作为结构化论证中的可扩展近似推理开辟了新的途径。

### 翻译

假设基础论证(ABA)是一种强大的结构化论证形式化方法，但在大框架下稳定语义的扩展的精确计算是不可行的。我们提出了第一个图神经网络(GNN)方法来近似ABA中的可信接受。为了利用GNN，我们通过依赖图表示来建模ABA框架，将假设、主张和规则编码为节点，使用异构边标签区分支持、推导和攻击关系。我们提出了两种GNN架构——ABAGCN和ABAGAT，它们分别堆叠残差异构卷积层或注意力层来学习节点嵌入。我们的模型在ICCMA 2023基准测试上训练，并使用合成的ABAF进行增强，通过贝叶斯搜索优化超参数。实验上，ABAGCN和ABAGAT都优于我们从抽象论证文献改编的最先进GNN基线，在ICCMA实例上实现了高达0.71的节点级F1分数。最后，我们开发了一个由预测器驱动的多项式时间扩展重建算法：它在小ABAF上重建的稳定扩展F1超过0.85，在大框架上保持约0.58的F1。我们的工作为结构化论证中的可扩展近似推理开辟了新的途径。


### 论文摘要

Assumption-Based Argumentation (ABA) is a powerful structured argumentation formalism, but exact computation of extensions under stable semantics is intractable for large frameworks. We present the first Graph Neural Network (GNN) approach to approximate credulous acceptance in ABA. To leverage GNNs, we model ABA frameworks via a dependency graph representation encoding assumptions, claims and rules as nodes, with heterogeneous edge labels distinguishing support, derive and attack relations. We propose two GNN architectures - ABAGCN and ABAGAT - that stack residual heterogeneous convolution or attention layers, respectively, to learn node embeddings. Our models are trained on the ICCMA 2023 benchmark, augmented with synthetic ABAFs, with hyperparameters optimised via Bayesian search. Empirically, both ABAGCN and ABAGAT outperform a state-of-the-art GNN baseline that we adapt from the abstract argumentation literature, achieving a node-level F1 score of up to 0.71 on the ICCMA instances. Finally, we develop a sound polynomial time extension-reconstruction algorithm driven by our predictor: it reconstructs stable extensions with F1 above 0.85 on small ABAFs and maintains an F1 of about 0.58 on large frameworks. Our work opens new avenues for scalable approximate reasoning in structured argumentation.

---

## 205. Scalable Coverage Trajectory Synthesis on GPUs as Statistical Inference

**论文链接:** [http://arxiv.org/abs/2511.11514v1](http://arxiv.org/abs/2511.11514v1)

**作者:** Max M. Sun, Jueun Kwon, Todd Murphey

**发布时间:** 2025-11-14

**备注:** Presented at the "Workshop on Fast Motion Planning and Control in the Era of Parallelism" at Robotics: Science and Systems 2025. Workshop website: https://sites.google.com/rice.edu/parallelized-planning-control/

### GPT解析

### 总结

该研究提出了一种基于流动匹配的覆盖运动规划新方法，通过解耦轨迹生成与控制合成，实现了并行化计算，显著提高了计算效率。

### 背景

覆盖运动规划对广泛的机器人任务至关重要。与传统运动规划问题不同，覆盖运动规划需要考虑整个轨迹的空间分布，这使得标准运动规划方法在计算效率上有限，且不太适合现代并行化框架。

### 目的

从流动匹配的角度，将覆盖运动规划问题表述为统计推断问题，以解决传统方法的计算效率问题。

### 方法

将常用的统计差异度量（如Kullback-Leibler散度和Sinkhorn散度）与标准线性二次调节器问题统一，并将覆盖的轨迹梯度生成与非线性系统动力学下的控制合成解耦，从而能够通过现代计算架构（特别是GPU）的并行化实现显著加速。

### 主要发现

该公式在并行化方面的可扩展性具有优势，与基于航点跟踪的传统方法相比，突显了其计算优势。

### 结论

该论文重点介绍了这种公式在通过并行化实现可扩展性方面的优势，强调了与传统方法相比的计算优势。

### 翻译

覆盖运动规划对广泛的机器人任务至关重要。与传统运动规划问题不同，覆盖运动规划需要考虑整个轨迹的空间分布，这使得标准运动规划方法在计算效率上有限，且不太适合现代并行化框架。本文从流动匹配的角度，将覆盖运动规划问题表述为统计推断问题。提出的方法将常用的统计差异度量与标准线性二次调节器问题统一，并将覆盖的轨迹梯度生成与非线性系统动力学下的控制合成解耦，从而能够通过现代计算架构的并行化实现显著加速。该论文重点介绍了这种公式在通过并行化实现可扩展性方面的优势。


### 论文摘要

Coverage motion planning is essential to a wide range of robotic tasks. Unlike conventional motion planning problems, which reason over temporal sequences of states, coverage motion planning requires reasoning over the spatial distribution of entire trajectories, making standard motion planning methods limited in computational efficiency and less amenable to modern parallelization frameworks. In this work, we formulate the coverage motion planning problem as a statistical inference problem from the perspective of flow matching, a generative modeling technique that has gained significant attention in recent years. The proposed formulation unifies commonly used statistical discrepancy measures, such as Kullback-Leibler divergence and Sinkhorn divergence, with a standard linear quadratic regulator problem. More importantly, it decouples the generation of trajectory gradients for coverage from the synthesis of control under nonlinear system dynamics, enabling significant acceleration through parallelization on modern computational architectures, particularly Graphics Processing Units (GPUs). This paper focuses on the advantages of this formulation in terms of scalability through parallelization, highlighting its computational benefits compared to conventional methods based on waypoint tracking.

---

## 206. Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective

**论文链接:** [http://arxiv.org/abs/2511.11478v1](http://arxiv.org/abs/2511.11478v1)

**作者:** Nhat Chung, Taisei Hanyu, Toan Nguyen, Huy Le, Frederick Bumgarner, Duy Minh Ho Nguyen, Khoa Vo, Kashu Yamazaki, Chase Rainwater, Tung Kieu, Anh Nguyen, Ngan Le

**发布时间:** 2025-11-14

**备注:** Accepted at AAAI 2026

### GPT解析

### 总结

该研究解决了具身智能体在复杂环境中需要对象级长期记忆的问题，提出了LIBERO-Mem测试套件和Embodied-SlotSSM框架，通过槽状态空间建模和关系编码器实现了时间可扩展的动作预测，为非马尔可夫环境中的机器人操作提供了有效解决方案。

### 背景

具身智能体在日益复杂的环境中操作时，需要能够随时间感知、跟踪和推理单个对象实例的能力，特别是在需要与视觉相似对象进行序列交互的任务中。在非马尔可夫环境中，关键决策线索通常隐藏在对象特定的历史中，而非当前场景。缺乏先前交互的持久记忆会导致视觉运动策略失败、重复过去动作或忽略已完成动作。

### 目的

引入LIBERO-Mem，一个非马尔可夫任务套件，用于在对象级部分可观测条件下测试机器人操作的压力；提出Embodied-SlotSSM框架，解决视觉-语言-行动模型在长时间序列任务中的可扩展性问题。

### 方法

提出Embodied-SlotSSM，一个面向时间可扩展性的基于槽的视觉-语言-行动框架。该框架保持时空一致的槽身份，并通过两种机制实现：(1)槽状态空间建模，用于重建短期历史；(2)关系编码器，将输入标记与动作解码对齐。这些组件共同支持时间接地、上下文感知的动作预测。

### 主要发现

视觉-语言-行动模型在非马尔可夫环境中表现不佳，即使对于跨越几百帧的任务，令牌扩展也会迅速变得难以处理。实验表明Embodied-SlotSSM在LIBERO-Mem和通用任务上具有基线性能，为对象中心化机器人策略中的非马尔可夫推理提供了可扩展的解决方案。

### 结论

Embodied-SlotSSM为处理非马尔可夫环境中的对象级推理提供了一种可扩展的方法，特别适合需要长期记忆和对象跟踪的复杂机器人任务，能够有效解决具身智能体在复杂环境中的对象交互挑战。

### 翻译

随着具身智能体在日益复杂的环境中操作，随时间感知、跟踪和推理单个对象实例的能力变得至关重要，特别是在需要与视觉相似对象进行序列交互的任务中。在这些非马尔可夫环境中，关键决策线索通常隐藏在对象特定的历史中，而非当前场景。没有先前交互的持久记忆（与什么交互过、在哪里、如何变化），视觉运动策略可能会失败、重复过去的动作或忽略已完成的动作。为应对这一挑战，我们引入LIBERO-Mem，一个非马尔可夫任务套件，用于在对象级部分可观测条件下测试机器人操作的压力。它结合了短期和长期目标跟踪与时间序列子目标，需要超越当前帧的推理。然而，视觉-语言-行动模型在这样的环境中往往表现不佳，即使对于跨越几百帧的任务，令牌扩展也会迅速变得难以处理。我们提出Embodied-SlotSSM，一个面向时间可扩展性的基于槽的视觉-语言-行动框架。它保持时空一致的槽身份，并通过两种机制利用它们：(1)槽状态空间建模，用于重建短期历史；(2)关系编码器，将输入标记与动作解码对齐。这些组件共同支持时间接地、上下文感知的动作预测。实验表明Embodied-SlotSSM在LIBERO-Mem和通用任务上的基线性能，为对象中心化机器人策略中的非马尔可夫推理提供了可扩展的解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人在操作任务中缺乏物体级别记忆的问题，特别是在处理视觉相似物体和需要序列交互的任务中。这个问题很重要，因为现实世界中人类能轻松回忆与特定物体的过去交互（如上次放置物品的位置），而现有机器人系统通常只依赖当前观察做决策，导致在重复操作、相似物体识别和长期任务中容易出错或重复无用动作。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别到现有视觉运动策略的局限性，即它们主要基于马尔可夫假设，仅使用当前观察做决策。他们借鉴了认知科学中人类对离散物体的推理理论，以及近期在模块化、插槽式架构方面的进展。具体设计上，他们结合了插槽注意力机制和状态空间建模技术，创建了LIBERO-Mem基准测试来评估物体中心记忆能力，并提出了Embodied-SlotSSM框架来维持结构化的物体中心记忆。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过基于插槽的表示维持结构化的物体中心记忆，实现时空一致的物体跟踪。整体流程包括：1)视觉编码将输入转换为密集嵌入；2)插槽注意力将视觉特征绑定到离散对象表示；3)插槽状态空间建模预测物体短期动态；4)插槽融合结合当前和预测状态；5)关系编码器生成上下文感知的关系标记；6)最终基于这些信息进行动作解码，实现时间感知的决策。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)LIBERO-Mem基准测试，专门评估物体中心记忆在部分可观察环境中的能力；2)Embodied-SlotSSM框架，通过插槽状态空间建模实现时间可扩展的记忆；3)结合短期和长期物体跟踪与时间序列子目标的任务设计。相比之前工作，LIBERO-Mem强调物体身份模糊性和子目标评估，而Embodied-SlotSSM解决了现有VLA模型在非马尔可夫环境中标记扩展困难的问题，超越了仅处理短期任务的物体中心学习方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过LIBERO-Mem基准测试和Embodied-SlotSSM框架，解决了机器人在部分可观察环境中缺乏物体级别记忆的问题，实现了更可靠的时间感知动作预测。'}


### 论文摘要

As embodied agents operate in increasingly complex environments, the ability to perceive, track, and reason about individual object instances over time becomes essential, especially in tasks requiring sequenced interactions with visually similar objects. In these non-Markovian settings, key decision cues are often hidden in object-specific histories rather than the current scene. Without persistent memory of prior interactions (what has been interacted with, where it has been, or how it has changed) visuomotor policies may fail, repeat past actions, or overlook completed ones. To surface this challenge, we introduce LIBERO-Mem, a non-Markovian task suite for stress-testing robotic manipulation under object-level partial observability. It combines short- and long-horizon object tracking with temporally sequenced subgoals, requiring reasoning beyond the current frame. However, vision-language-action (VLA) models often struggle in such settings, with token scaling quickly becoming intractable even for tasks spanning just a few hundred frames. We propose Embodied-SlotSSM, a slot-centric VLA framework built for temporal scalability. It maintains spatio-temporally consistent slot identities and leverages them through two mechanisms: (1) slot-state-space modeling for reconstructing short-term history, and (2) a relational encoder to align the input tokens with action decoding. Together, these components enable temporally grounded, context-aware action prediction. Experiments show Embodied-SlotSSM's baseline performance on LIBERO-Mem and general tasks, offering a scalable solution for non-Markovian reasoning in object-centric robotic policies.

---

## 207. Lattice Resonances in Periodic Arrays of Time-Modulated Scatterers

**论文链接:** [http://arxiv.org/abs/2511.11454v1](http://arxiv.org/abs/2511.11454v1)

**作者:** María Blanco de Paz, Juan R. Deop-Ruano, Diego M. Solís, Alejandro Manjavacas

**发布时间:** 2025-11-14

**备注:** 41 pages, 7 figures

### GPT解析

### 总结

该研究建立了理解时变调制阵列中集体晶格共振的简单理论框架，实现了对这些模式的动态控制和放大。研究通过偶极近似和时Floquet理论框架，分析了时变调制散射体周期阵列中的晶格共振现象，发现集体共振可以在显著较低的调制强度下实现放大。

### 背景

晶格共振是由周期性排列的散射体支持的集体光学模式，源于散射体在底层周期性结构下相干相互作用产生的结果。这些共振产生的光学响应比单个散射体更强且光谱更窄。虽然这种现象在传统时不变系统中已被广泛研究，但最近时变光子学的进展为利用和增强这些集体模式的非凡特性提供了新机会。

### 目的

研究时变调制散射体周期阵列中的晶格共振现象，建立理解集体晶格共振的简单理论框架，实现对这些模式的动态控制和放大。

### 方法

使用基于偶极近似和时Floquet理论的简单框架，将每个散射体建模为具有周期性变化光学性质的谐振子。首先分析单个散射体的响应，确定定义其动力学的复本征频率；然后扩展到周期阵列，研究时变调制与晶格共振之间的相互作用。

### 主要发现

1. 适当的调制幅度和频率可以使一个本征频率的虚部消失，导致放大效应；2. 与孤立散射体不同，晶格共振的集体特性引入了放大区域的明显更复杂的光谱依赖性；3. 放大出现在显著较低的调制强度下，这由集体共振增强的光-物质相互作用和增加的寿命所促进。

### 结论

该研究建立了理解时变调制阵列中集体晶格共振的简单理论框架，实现了对这些模式的动态控制和放大，为利用时变光子学增强集体光学特性提供了新途径。

### 翻译

晶格共振是由周期性排列的散射体支持的集体光学模式，源于它们在底层周期性结构下相干相互作用产生的结果。由于这些共振的集体特性，它们产生的光学响应比单个散射体更强且光谱更窄。虽然这种现象在传统时不变系统中已被广泛研究，但最近时变光子学的进展为利用和增强这些集体模式的非凡特性提供了新机会。我们使用基于偶极近似和时Floquet理论的简单框架，研究了时变调制散射体周期阵列中的晶格共振，其中每个散射体被建模为具有周期性变化光学性质的谐振子。我们首先分析单个散射体的响应，利用我们的模型确定定义其动力学的复本征频率。我们表明，对于适当的调制幅度和频率，这些本征频率中的一个的虚部消失，导致放大。在此基础上，我们将分析扩展到周期阵列，研究时变调制与晶格共振之间的相互作用。与孤立散射体不同，晶格共振的集体特性引入了放大区域的明显更复杂的光谱依赖性。值得注意的是，这种放大出现在显著较低的调制强度下，由集体共振增强的光-物质相互作用和增加的寿命所促进。我们的工作建立了理解时变调制阵列中集体晶格共振的简单理论框架，实现了对这些模式的动态控制和放大。


### 论文摘要

Lattice resonances are collective optical modes supported by periodic arrays of scatterers, arising from their coherent interaction enabled by the underlying periodicity. Owing to their collective nature, these resonances produce optical responses that are both stronger and spectrally narrower than those of individual scatterers. While such phenomena have been extensively studied in conventional time-invariant systems, recent advances in time-varying photonics present new opportunities to exploit and enhance the extraordinary characteristics of these collective modes. Here, we investigate lattice resonances in periodic arrays of time-modulated scatterers using a simple framework based on the dipolar approximation and time-Floquet theory, where each scatterer is modeled as a harmonic oscillator with periodically varying optical properties. We begin by analyzing the response of an individual scatterer, leveraging our model to identify the complex eigenfrequencies that define its dynamics. We show that, for the appropriate modulation amplitude and frequency, the imaginary part of one of these eigenfrequencies vanishes, leading to amplification. Building on this, we extend our analysis to a periodic array to investigate the effect of the interplay between temporal modulation and lattice resonances. In contrast to isolated scatterers, the collective nature of lattice resonances introduces a markedly more intricate spectral dependence of the amplification regime. Notably, this amplification emerges at substantially lower modulation strengths, facilitated by the enhanced light-matter interaction and increased lifetime provided by these collective resonances. Our work establishes a simple theoretical framework for understanding collective lattice resonances in time-modulated arrays, enabling dynamic control and amplification of these modes.

---

## 208. CURENet: Combining Unified Representations for Efficient Chronic Disease Prediction

**论文链接:** [http://arxiv.org/abs/2511.11423v1](http://arxiv.org/abs/2511.11423v1)

**作者:** Cong-Tinh Dao, Nguyen Minh Thao Phan, Jun-En Ding, Chenwei Wu, David Restrepo, Dongsheng Luo, Fanyi Zhao, Chun-Chieh Liao, Wen-Chih Peng, Chi-Te Wang, Pei-Fu Chen, Ling Chen, Xinglong Ju, Feng Liu, Fang-Ming Hung

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了CURENet，一种多模态模型，用于整合电子健康记录中的不同数据类型，实现慢性疾病的准确预测。

### 背景

电子健康记录(EHR)旨在整合多种数据类型，包括非结构化临床笔记、结构化实验室测试和时间序列就诊数据。医生利用这些多模态和时间序列数据形成对患者健康的全面认识，这对于明智的治疗决策至关重要。然而，大多数预测模型未能充分捕捉多种数据模态之间的交互、冗余和时间模式，往往只关注单一数据类型或忽略这些复杂性。

### 目的

开发一个能够整合多种EHR数据类型并捕捉它们之间复杂交互关系的多模态模型，以提高慢性疾病预测的准确性。

### 方法

研究者提出了CURENet（Combining Unified Representations for Efficient chronic disease prediction），一种多模态模型，利用大型语言模型(LLMs)处理临床文本和实验室测试文本，使用transformer编码器处理纵向序列就诊数据，从而整合非结构化临床笔记、实验室测试和患者时间序列数据。

### 主要发现

CURENet能够捕捉不同形式临床数据之间的复杂交互，创建更可靠的慢性疾病预测模型。在公共MIMIC-III和私有FEMH数据集上评估时，CURENet在多标签框架下预测前10种慢性疾病的准确率超过94%。

### 结论

多模态EHR整合具有增强临床决策和改善患者结果的潜力。

### 翻译

电子健康记录(EHR)旨在整合多种数据类型，包括非结构化临床笔记、结构化实验室测试和时间序列就诊数据。医生利用这些多模态和时间序列EHR数据形成对患者健康的全面认识，这对于明智的治疗决策至关重要。然而，大多数预测模型未能充分捕捉多种数据模态之间的交互、冗余和时间模式，往往只关注单一数据类型或忽略这些复杂性。在本文中，我们提出了CURENet，一种多模态模型（Combining Unified Representations for Efficient chronic disease prediction），它利用大型语言模型(LLMs)处理临床文本和实验室测试文本，使用transformer编码器处理纵向序列就诊数据，从而整合非结构化临床笔记、实验室测试和患者时间序列数据。CURENet能够捕捉不同形式临床数据之间的复杂交互，创建更可靠的慢性疾病预测模型。我们使用公共MIMIC-III和私有FEMH数据集评估了CURENet，在多标签框架下预测前10种慢性疾病的准确率超过94%。我们的研究结果突显了多模态EHR整合在增强临床决策和改善患者结果方面的潜力。


### 论文摘要

Electronic health records (EHRs) are designed to synthesize diverse data types, including unstructured clinical notes, structured lab tests, and time-series visit data. Physicians draw on these multimodal and temporal sources of EHR data to form a comprehensive view of a patient's health, which is crucial for informed therapeutic decision-making. Yet, most predictive models fail to fully capture the interactions, redundancies, and temporal patterns across multiple data modalities, often focusing on a single data type or overlooking these complexities. In this paper, we present CURENet, a multimodal model (Combining Unified Representations for Efficient chronic disease prediction) that integrates unstructured clinical notes, lab tests, and patients' time-series data by utilizing large language models (LLMs) for clinical text processing and textual lab tests, as well as transformer encoders for longitudinal sequential visits. CURENet has been capable of capturing the intricate interaction between different forms of clinical data and creating a more reliable predictive model for chronic illnesses. We evaluated CURENet using the public MIMIC-III and private FEMH datasets, where it achieved over 94\% accuracy in predicting the top 10 chronic conditions in a multi-label framework. Our findings highlight the potential of multimodal EHR integration to enhance clinical decision-making and improve patient outcomes.

---

## 209. Disentangling Emotional Bases and Transient Fluctuations: A Low-Rank Sparse Decomposition Approach for Video Affective Analysis

**论文链接:** [http://arxiv.org/abs/2511.11406v1](http://arxiv.org/abs/2511.11406v1)

**作者:** Feng-Qi Cui, Jinyang Huang, Ziyu Jia, Xinyu Li, Xin Yan, Xiaokang Zhou, Meng Wang

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一种名为LSEF的低秩稀疏情感理解框架，旨在解决视频情感计算中的模型不稳定性和表示退化问题，通过分层机制分离情感基和瞬时波动，显著提升了情感分析的鲁棒性和动态辨别能力。

### 背景

视频情感计算(VAC)对情感分析和人机交互至关重要，但因复杂的情感动态导致模型不稳定性和表示退化。核心局限在于缺乏分层结构机制来分离不同的情感成分，即情感基（长期情感基调）和瞬时波动（短期情感波动）。

### 目的

解决VAC中因情感动态复杂性导致的模型不稳定性和表示退化问题，通过分层机制分离不同情感成分，提升情感分析的准确性和鲁棒性。

### 方法

提出低秩稀疏情感理解框架(LSEF)，基于低秩稀疏原理构建统一模型，将情感动态重新构架为分层低秩稀疏组合过程。包含三个即插即用模块：稳定性编码模块(SEM)捕获低秩情感基；动态解耦模块(DDM)隔离稀疏瞬时信号；一致性集成模块(CIM)重建多尺度稳定性和反应性一致性。采用秩感知优化(RAO)策略自适应平衡梯度平滑度和敏感性。

### 主要发现

在多个数据集上的广泛实验证实，LSEF显著增强了情感计算的鲁棒性和动态辨别能力，验证了分层低秩稀疏建模在理解情感动态方面的有效性和通用性。

### 结论

分层低秩稀疏建模是理解情感动态的有效方法，LSEF框架通过分离情感基和瞬时波动，成功解决了VAC中的模型不稳定性和表示退化问题。

### 翻译

基于视频的情感计算(VAC)对情感分析和人机交互至关重要，但由于复杂的情感动态，它遭受着模型不稳定性和表示退化的问题。由于不同情感波动在不同情感语境下的含义可能不同，核心局限在于缺乏分层结构机制来分离不同的情感成分，即情感基（长期情感基调）和瞬时波动（短期情感波动）。为解决这一问题，我们提出了低秩稀疏情感理解框架(LSEF)，这是一个基于低秩稀疏原理的统一模型，理论上将情感动态重新构架为分层的低秩稀疏组合过程。LSEF采用三个即插即用模块，即稳定性编码模块(SEM)捕获低秩情感基；动态解耦模块(DDM)隔离稀疏瞬时信号；一致性集成模块(CIM)重建多尺度稳定性和反应性一致性。该框架通过秩感知优化(RAO)策略进行优化，该策略自适应平衡梯度平滑度和敏感性。在多个数据集上的广泛实验证实，LSEF显著增强了鲁棒性和动态辨别能力，这进一步验证了分层低秩稀疏建模在理解情感动态方面的有效性和通用性。


### 论文摘要

Video-based Affective Computing (VAC), vital for emotion analysis and human-computer interaction, suffers from model instability and representational degradation due to complex emotional dynamics. Since the meaning of different emotional fluctuations may differ under different emotional contexts, the core limitation is the lack of a hierarchical structural mechanism to disentangle distinct affective components, i.e., emotional bases (the long-term emotional tone), and transient fluctuations (the short-term emotional fluctuations). To address this, we propose the Low-Rank Sparse Emotion Understanding Framework (LSEF), a unified model grounded in the Low-Rank Sparse Principle, which theoretically reframes affective dynamics as a hierarchical low-rank sparse compositional process. LSEF employs three plug-and-play modules, i.e., the Stability Encoding Module (SEM) captures low-rank emotional bases; the Dynamic Decoupling Module (DDM) isolates sparse transient signals; and the Consistency Integration Module (CIM) reconstructs multi-scale stability and reactivity coherence. This framework is optimized by a Rank Aware Optimization (RAO) strategy that adaptively balances gradient smoothness and sensitivity. Extensive experiments across multiple datasets confirm that LSEF significantly enhances robustness and dynamic discrimination, which further validates the effectiveness and generality of hierarchical low-rank sparse modeling for understanding affective dynamics.

---

## 210. Modeling the Multi-Wavelength Afterglow of Short Gamma-Ray Bursts with a Plateau Phase

**论文链接:** [http://arxiv.org/abs/2511.11396v1](http://arxiv.org/abs/2511.11396v1)

**作者:** Chen Deng, Yong-Feng Huang, Abdusattar Kurban, Jin-Jun Geng, Fan Xu, Xiao-Fei Dong, Hao-Xuan Gao, En-Wei Liang, Liang Li

**发布时间:** 2025-11-14

**备注:** 19 pages, 6 figures, 1 table

### GPT解析

### 总结

该研究通过结合多波段观测数据，对具有平台相特征的短伽马射线暴进行了磁星能量注入模型下的宽余辉建模，解决了仅依靠X射线数据时存在的参数简并性问题，并发现了短伽马射线暴在洛伦兹因子与能量平面上的分布特征。

### 背景

短伽马射线暴的平台相为其中央引擎的合并后活动提供了有价值的信息，尽管平台相的物理起源尚不确定，但磁星能量注入模型能较好地解释观测到的时序和光度特征。

### 目的

通过结合X射线、光学和射电观测数据，在磁星能量注入模型框架下对七个具有平台相特征的短伽马射线暴进行宽余辉建模，以更准确地确定磁星参数。

### 方法

使用马尔可夫链蒙特卡洛方法对七个具有平台相特征的短伽马射线暴进行宽余辉建模，结合X射线、光学和射电观测数据来推导关键模型参数。

### 主要发现

1) 能量注入在大多数事件中显著修改了余辉动力学；2) 与仅X射线分析相比，宽波段建模给出了较低的磁场强度和较短的旋转周期，对应较高的注入光度；3) 多波长数据结合有效缓解了磁星参数和X射线辐射效率之间的简并性；4) 短伽马射线暴在初始洛伦兹因子与伽马射线能量平面上的分布与长伽马射线暴明显不同，这种偏移与短伽马射线暴较硬的谱一致。

### 结论

随着更大样本的可用性，短伽马射线暴在洛伦兹因子与能量平面上的分布特征可能成为研究其前身天体的有用诊断工具。

### 翻译

具有平台相的短伽马射线暴为其中央引擎的合并后活动提供了宝贵的见解。尽管平台相的物理起源尚不确定，但磁星能量注入模型提供了一个有说服力的解释，能够重现观测到的时序和光度特征。然而，之前仅依靠X射线数据的研究在约束磁星参数时存在严重的参数简并性。我们在磁星能量注入模型框架下，通过结合X射线、光学和射电观测，对七个具有平台相特征的短伽马射线暴进行了宽余辉建模。使用马尔可夫链蒙特卡洛方法推导关键模型参数。研究发现，能量注入在大多数事件中显著修改了余辉动力学。与仅X射线分析相比，我们的宽波段建模系统地给出了较低的磁场强度和较短的旋转周期，对应较高的注入光度。研究明确表明，结合多波长数据有效缓解了磁星参数和X射线辐射效率之间的简并性。此外，当绘制在初始洛伦兹因子与伽马射线能量平面上时，我们的短伽马射线暴样本与长伽马射线暴的分布明显不同。这种偏移与短伽马射线暴观测到的较硬谱一致，随着更大样本的可用性，可能成为研究前身天体的有用诊断工具。


### 论文摘要

Short gamma-ray bursts (GRBs) exhibiting a plateau phase provide valuable insights into the post-merger activity of their central engines. Although the physical origin of the plateau remains uncertain, the magnetar energy injection model offers a compelling explanation that reproduces the observed temporal and luminosity features. However, previous studies relying solely on X-ray data have suffered from strong parameter degeneracies when constraining the magnetar parameters. Here we perform broadband afterglow modeling on seven short GRBs with plateau features by combining X-ray, optical, and radio observations within the framework of the magnetar energy injection model. Key model parameters are derived by using the Markov Chain Monte Carlo method. It is found that the energy injection substantially modifies the afterglow dynamics in most events. Compared with X-ray--only analyses, our broadband modeling yields systematically a lower magnetic field strength and a shorter spin period for the central magnetar, corresponding to a higher injection luminosity. The study clearly shows that incorporating multi-wavelength data effectively alleviates the degeneracy between the magnetar parameters and X-ray radiative efficiency. In addition, the distribution of our short GRBs differs markedly from long GRBs when they are plotted on the initial Lorentz factor versus gamma-ray energy plane. This offset, consistent with the observed harder spectrum of short GRBs, may serve as a useful diagnostic for investigating the progenitor as larger samples are available.

---

## 211. Universal Safety Controllers with Learned Prophecies

**论文链接:** [http://arxiv.org/abs/2511.11390v1](http://arxiv.org/abs/2511.11390v1)

**作者:** Bernd Finkbeiner, Niklas Metzger, Satya Prakash Nayak, Anne-Kathrin Schmuck

**发布时间:** 2025-11-14

**备注:** AAAI 2026

### GPT解析

### 总结

本文提出了一种基于学习的通用安全控制器(USCs)合成近似算法，通过计算CTL公式作为预言表示，解决了精确计算预言的计算挑战，提高了效率和可解释性。

### 背景

通用安全控制器(USCs)是一种逻辑控制框架，能保证在应用于任何可实现工厂模型时满足给定的时间安全规范。相比传统方法，USC合成构建通用控制器，其输出由工厂行为（预言）调节，具有强大泛化和可扩展性优势。

### 目的

解决精确计算和验证预言的计算挑战，提高USC合成的效率和可解释性。

### 方法

引入基于学习的USC合成近似算法，不计算精确预言（通过自动机推理树集），而是从示例工厂计算下近似和上近似，推断计算树逻辑(CTL)公式作为预言表示，并通过验证步骤使生成的USC泛化到未见过的工厂。

### 主要发现

实验结果表明，学习的预言保持泛化能力，比精确树自动机表示更紧凑和可解释，同时提高了效率和可解释性。

### 结论

通过学习近似算法和CTL公式表示预言，可以克服精确预言的计算挑战，同时保持USCs的泛化能力，并提高效率和可解释性。

### 翻译

通用安全控制器(USCs)是一种有前景的逻辑控制框架，当应用于任何可实现的工厂模型时能保证满足给定的时间安全规范。与传统方法在给定详细工厂模型上合成一个逻辑控制器不同，USC合成构建一个通用控制器，其输出由工厂行为（称为预言）调节。因此，USCs相比经典逻辑控制器具有强大的泛化和可扩展性优势。然而，精确计算和验证预言仍然具有计算挑战性。本文引入了一种基于学习的USC合成近似算法，通过计算下近似和上近似而非精确预言（通过自动机推理树集），推断计算树逻辑(CTL)公式作为预言表示。生成的USC通过验证步骤泛化到未见过的工厂，通过小型简洁的CTL预言提供改进的效率和可解释性，这些预言保持人类可读和可解释。实验结果表明，学习的预言保持泛化能力，比精确树自动机表示更紧凑和可解释。


### 论文摘要

\emph{Universal Safety Controllers (USCs)} are a promising logical control framework that guarantees the satisfaction of a given temporal safety specification when applied to any realizable plant model. Unlike traditional methods, which synthesize one logical controller over a given detailed plant model, USC synthesis constructs a \emph{generic controller} whose outputs are conditioned by plant behavior, called \emph{prophecies}. Thereby, USCs offer strong generalization and scalability benefits over classical logical controllers. However, the exact computation and verification of prophecies remain computationally challenging. In this paper, we introduce an approximation algorithm for USC synthesis that addresses these limitations via learning. Instead of computing exact prophecies, which reason about sets of trees via automata, we only compute under- and over-approximations from (small) example plants and infer computation tree logic (CTL) formulas as representations of prophecies. The resulting USC generalizes to unseen plants via a verification step and offers improved efficiency and explainability through small and concise CTL prophecies, which remain human-readable and interpretable. Experimental results demonstrate that our learned prophecies remain generalizable, yet are significantly more compact and interpretable than their exact tree automata representations.

---

## 212. Free3D: 3D Human Motion Emerges from Single-View 2D Supervision

**论文链接:** [http://arxiv.org/abs/2511.11368v1](http://arxiv.org/abs/2511.11368v1)

**作者:** Sheng Liu, Yuanzhi Liang, Sidan Du

**发布时间:** 2025-11-14

### GPT解析

### 总结

Free3D是一个创新的3D人体运动生成框架，无需3D运动注释即可生成高质量的3D运动，通过2D数据训练和特定正则化技术实现性能超越完全3D监督模型。

### 背景

当前3D人体运动生成模型重建精度高但泛化能力有限，部分原因是使用精确3D监督使模型拟合固定坐标模式而非学习本质3D结构和运动语义。

### 目的

克服现有3D人体运动生成模型的泛化局限性，提出不依赖3D运动注释的框架来合成真实3D运动。

### 方法

提出Free3D框架，引入Motion-Lifting Residual Quantized VAE (ML-RQ)将2D运动序列映射到3D一致潜在空间，设计视图一致性、方向一致性和物理合理性等3D自由正则化目标，完全在2D数据上训练。

### 主要发现

Free3D能生成多样化、时间连贯且语义对齐的3D运动，性能可与甚至超越完全3D监督模型，放松显式3D监督可促进更强结构推理和泛化能力。

### 结论

通过不使用显式3D监督，Free3D提供了一种可扩展且数据高效的3D运动生成范式，模型能学习更本质的3D结构和运动语义。

### 翻译

最近的3D人体运动生成模型展示了显著的重建精度，但在训练分布之外泛化能力有限。这种局限性部分源于使用精确的3D监督，这会鼓励模型拟合固定的坐标模式，而不是学习强大的泛化所需的必要3D结构和运动语义线索。为了克服这一局限性，我们提出了Free3D，一个无需任何3D运动注释即可合成真实3D运动的框架。Free3D引入了Motion-Lifting Residual Quantized VAE (ML-RQ)，将2D运动序列映射到3D一致的潜在空间，以及一套3D自由正则化目标，强制执行视图一致性、方向一致性和物理合理性。完全在2D运动数据上训练的Free3D能够生成多样化、时间连贯且语义对齐的3D运动，其性能可与甚至超越完全3D监督的对应模型相媲美。这些结果表明，放松显式3D监督可以促进更强的结构推理和泛化能力，为3D运动生成提供了一种可扩展且数据高效的范式。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D人体运动生成模型在训练分布外泛化能力有限的问题。现有方法虽然重建精度高，但依赖精确的3D监督，导致模型拟合固定坐标模式而非学习本质结构和语义。这个问题很重要，因为3D运动数据收集成本高、种类有限，且现有方法生成的动作多样性不足，难以泛化到新场景或新动作。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法的局限性，认为精确的3D监督会约束模型泛化。他们提出从2D线索学习3D运动的想法，利用2D数据的丰富性和投影模糊性鼓励模型推断结构而非记忆坐标。方法借鉴了MoMask的架构，但修改为使用2D监督，设计了Free3D运动表示、Motion Lifting RQ-VAE和3D-free正则化三个关键组件，构建了完整的2D到3D运动生成框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是仅从2D运动监督学习3D运动，通过2D投影的模糊性鼓励模型推断深度、身体结构和物理合理性，而非直接复制3D坐标。整体流程包括：1)将运动分解为全局轨迹和相对肢体向量；2)使用ML-RQ编码器将2D运动编码到潜在空间并量化；3)解码器生成3D运动；4)应用3D-free正则化确保视图一致性、方向合理性和特征一致性；5)分阶段训练ML-RQ、Masked Transformer和Residual Transformer。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个完全基于单视图2D监督的3D运动生成框架；2)Free3D运动表示建立2D-3D结构对应；3)Motion Lifting RQ-VAE实现无3D监督的3D一致性学习；4)一套3D-free正则化目标。与传统方法不同，Free3D不依赖3D数据，使用2D监督鼓励结构推理而非坐标拟合，实验显示其在多样性和泛化方面表现更优，能处理3D数据难以捕捉的场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Free3D提出了一种仅从单视图2D监督学习3D人体运动的新范式，通过放松精确的3D监督约束，实现了与甚至超越完全3D监督方法的性能，同时增强了模型的泛化能力和生成多样性。'}


### 论文摘要

Recent 3D human motion generation models demonstrate remarkable reconstruction accuracy yet struggle to generalize beyond training distributions. This limitation arises partly from the use of precise 3D supervision, which encourages models to fit fixed coordinate patterns instead of learning the essential 3D structure and motion semantic cues required for robust generalization.To overcome this limitation, we propose Free3D, a framework that synthesizes realistic 3D motions without any 3D motion annotations. Free3D introduces a Motion-Lifting Residual Quantized VAE (ML-RQ) that maps 2D motion sequences into 3D-consistent latent spaces, and a suite of 3D-free regularization objectives enforcing view consistency, orientation coherence, and physical plausibility. Trained entirely on 2D motion data, Free3D generates diverse, temporally coherent, and semantically aligned 3D motions, achieving performance comparable to or even surpassing fully 3D-supervised counterparts. These results suggest that relaxing explicit 3D supervision encourages stronger structural reasoning and generalization, offering a scalable and data-efficient paradigm for 3D motion generation.

---

## 213. Extreme-PLS with missing data under weak dependence

**论文链接:** [http://arxiv.org/abs/2511.11338v1](http://arxiv.org/abs/2511.11338v1)

**作者:** Stéphane Girard, Cambyse Pakzad

**发布时间:** 2025-11-14

**备注:** 45 pages, 14 figures

### GPT解析

### 总结

本文开发了一种理论框架，用于在存在缺失数据和弱时间依赖的情况下进行极端偏最小二乘（EPLS）降维。

### 背景

基于最近开发的EPLS方法（用于建模响应变量与高维协变量之间的极端依赖关系），但需要扩展到更现实的数据环境中，这些环境存在序列相关性和缺失数据。

### 目的

扩展EPLS方法以处理重尾条件下的单指标逆回归模型，并考虑协变量上的随机缺失（MAR）机制，其概率取决于响应变量的极端性。

### 方法

在alpha混合框架内建立所提出估计量的渐近行为，并进行大量蒙特卡洛实验（涵盖十一种依赖方案），包括ARMA、GARCH和非线性ESTAR过程。

### 主要发现

该方法在广泛的重尾和依赖场景中表现稳健，即使大部分数据缺失；对环境数据的实际应用证实了该方法能够恢复有意义的尾部方向。

### 结论

所提出的EPLS框架在处理存在缺失数据和弱时间依赖的重尾数据时表现出色，能够有效地进行降维并保留极端依赖结构。

### 翻译

本文开发了一种理论框架，用于在存在缺失数据和弱时间依赖的情况下进行极端偏最小二乘（EPLS）降维。基于最近开发的EPLS方法（用于建模响应变量与高维协变量之间的极端依赖关系），我们将这种方法扩展到更现实的数据环境中，这些环境存在序列相关性和缺失数据。具体来说，我们在重尾条件下考虑了一个单指标逆回归模型，并引入了一种随机缺失（MAR）机制作用于协变量，其概率取决于响应变量的极端性。在alpha混合框架内建立了所提出估计量的渐近行为，导致在规则变化尾部条件下的一致性结果。涵盖十一种依赖方案（包括ARMA、GARCH和非线性ESTAR过程）的大量蒙特卡洛实验表明，该方法在广泛的重尾和依赖场景中表现稳健，即使大部分数据缺失。对环境数据的实际应用进一步证实了该方法恢复有意义的尾部方向的能力。


### 论文摘要

This paper develops a theoretical framework for Extreme Partial Least Squares (EPLS) dimension reduction in the presence of missing data and weak temporal dependence. Building upon the recent EPLS methodology for modeling extremal dependence between a response variable and high-dimensional covariates, we extend the approach to more realistic data settings where both serial correlation and missing-ness occur. Specifically, we consider a single-index inverse regression model under heavy-tailed conditions and introduce a Missing-at-Random (MAR) mechanism acting on the covariates, whose probability depends on the extremeness of the response. The asymptotic behavior of the proposed estimator is established within an alpha-mixing framework, leading to consistency results under regularly varying tails. Extensive Monte-Carlo experiments covering eleven dependence schemes (including ARMA, GARCH, and nonlinear ESTAR processes) demonstrate that the method performs robustly across a wide range of heavy-tailed and dependent scenarios, even when substantial portions of data are missing. A real-world application to environmental data further confirms the method's capacity to recover meaningful tail directions.

---

## 214. Adverbs Revisited: Enhancing WordNet Coverage of Adverbs with a Supersense Taxonomy

**论文链接:** [http://arxiv.org/abs/2511.11214v1](http://arxiv.org/abs/2511.11214v1)

**作者:** Jooyoung Lee, Jader Martins Camboim de Sá

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一个基于语言学的副词上义类型学，通过注释研究进行了经验验证，扩展了WordNet的副词分类体系，并促进了多种自然语言处理应用。

### 背景

WordNet为名词和动词提供了丰富的上义层次体系，但副词发展不足，缺乏系统性的语义分类。

### 目的

引入一个基于语言学原理的副词上义类型学，并通过注释进行经验验证。

### 方法

提出一个基于语言学的副词上义类型学，通过注释研究进行验证，该类型学捕捉了方式、时间、频率、程度、领域、说话者导向和主体导向等主要语义领域。

### 主要发现

初步注释研究结果表明这些类别能够覆盖自然文本中的大多数副词，且人类注释者可以可靠地分配这些类别。

### 结论

将此类型学纳入WordNet可以扩展其覆盖范围，使其更接近语言学理论，并促进词义消歧、事件提取、情感分析和话语建模等下游自然语言处理应用。

### 翻译

WordNet为名词和动词提供了丰富的上义层次体系，然而副词发展不足，缺乏系统性的语义分类。我们引入了一个基于语言学的副词上义类型学，通过注释进行了经验验证，捕捉了包括方式、时间、频率、程度、领域、说话者导向和主体导向功能在内的主要语义领域。初步注释研究结果表明这些类别能够自然文本中的副词提供广泛覆盖，且人类注释者可以可靠地分配这些类别。将此类型学纳入WordNet可以扩展其覆盖范围，使其更接近语言学理论，并促进词义消歧、事件提取、情感分析和话语建模等下游自然语言处理应用。我们提出了建议的上义类别、注释结果以及未来工作的方向。


### 论文摘要

WordNet offers rich supersense hierarchies for nouns and verbs, yet adverbs remain underdeveloped, lacking a systematic semantic classification. We introduce a linguistically grounded supersense typology for adverbs, empirically validated through annotation, that captures major semantic domains including manner, temporal, frequency, degree, domain, speaker-oriented, and subject-oriented functions. Results from a pilot annotation study demonstrate that these categories provide broad coverage of adverbs in natural text and can be reliably assigned by human annotators. Incorporating this typology extends WordNet's coverage, aligns it more closely with linguistic theory, and facilitates downstream NLP applications such as word sense disambiguation, event extraction, sentiment analysis, and discourse modeling. We present the proposed supersense categories, annotation outcomes, and directions for future work.

---

## 215. Reverberation: Learning the Latencies Before Forecasting Trajectories

**论文链接:** [http://arxiv.org/abs/2511.11164v1](http://arxiv.org/abs/2511.11164v1)

**作者:** Conghao Wong, Ziqian Zou, Beihao Xia, Xinge You

**发布时间:** 2025-11-14

### GPT解析

### 总结

该论文提出了一种新的混响变换和Rev轨迹预测模型，用于模拟和预测智能体的延迟偏好及其随机性，通过两个明确且可学习的混响核实现可控的轨迹预测。

### 背景

轨迹预测任务的核心是连接过去与未来，在空间和时间上连接智能体。尽管付出了很大努力，明确学习和预测延迟仍然具有挑战性，不同的智能体可能表现出不同的延迟偏好。

### 目的

提出一种能够模拟和预测每个智能体不同延迟偏好及其随机性的方法，以改善轨迹预测的准确性和可解释性。

### 方法

受声学中混响曲线的启发，提出了一种新的混响变换和相应的Rev轨迹预测模型，使用两个明确且可学习的混响核来模拟和预测延迟。

### 主要发现

在多个数据集上的实验表明，Rev实现了具有竞争力的准确性，同时揭示了智能体和场景之间可解释的延迟动态。定性分析验证了所提出的混响变换的属性。

### 结论

Rev具有作为通用延迟建模方法的潜力，能够改善轨迹预测系统的因果连续性，避免产生不合理或意外的轨迹。

### 翻译

连接过去与未来，在空间和时间上连接智能体，是轨迹预测任务的核心。尽管付出了巨大努力，明确学习和预测延迟（智能体对不同的轨迹变化事件做出反应并调整未来路径的时间延迟，无论是自主的还是交互式的）仍然具有挑战性。不同的智能体可能对特定的轨迹变化事件表现出不同的延迟偏好。缺乏对这种延迟的考虑可能会破坏预测系统的因果连续性，并导致不合理或意外的轨迹。受声学中混响曲线的启发，我们提出了一种新的混响变换和相应的Rev轨迹预测模型，通过使用两个明确且可学习的混响核来模拟和预测每个智能体的不同延迟偏好及其随机性，允许基于这些预测的延迟进行可控的轨迹预测。在多个数据集上的实验，无论是行人还是车辆，都表明Rev实现了具有竞争力的准确性，同时揭示了智能体和场景之间可解释的延迟动态。定性分析进一步验证了所提出的混响变换的属性，突显了其作为通用延迟建模方法的潜力。


### 论文摘要

Bridging the past to the future, connecting agents both spatially and temporally, lies at the core of the trajectory prediction task. Despite great efforts, it remains challenging to explicitly learn and predict latencies, the temporal delays with which agents respond to different trajectory-changing events and adjust their future paths, whether on their own or interactively. Different agents may exhibit distinct latency preferences for noticing, processing, and reacting to any specific trajectory-changing event. The lack of consideration of such latencies may undermine the causal continuity of the forecasting system and also lead to implausible or unintended trajectories. Inspired by the reverberation curves in acoustics, we propose a new reverberation transform and the corresponding Reverberation (short for Rev) trajectory prediction model, which simulates and predicts different latency preferences of each agent as well as their stochasticity by using two explicit and learnable reverberation kernels, allowing for the controllable trajectory prediction based on these forecasted latencies. Experiments on multiple datasets, whether pedestrians or vehicles, demonstrate that Rev achieves competitive accuracy while revealing interpretable latency dynamics across agents and scenarios. Qualitative analyses further verify the properties of the proposed reverberation transform, highlighting its potential as a general latency modeling approach.

---

## 216. Deep Learning for Short-Term Precipitation Prediction in Four Major Indian Cities: A ConvLSTM Approach with Explainable AI

**论文链接:** [http://arxiv.org/abs/2511.11152v1](http://arxiv.org/abs/2511.11152v1)

**作者:** Tanmay Ghosh, Shaurabh Anand, Rakesh Gomaji Nannewar, Nithin Nagaraj

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文开发了一种可解释的深度学习框架，用于印度四个主要城市的短期降水预测，通过混合CNN-ConvLSTM架构和多种可解释性分析方法，在保持准确性的同时提高了模型透明度。

### 背景

深度学习模型用于降水预测通常被视为黑箱，限制了其在实际天气预测中的应用。

### 目的

提高模型透明度同时保持准确性，开发一个可解释的深度学习框架用于印度四个主要城市的短期降水预测。

### 方法

开发了混合的Time-Distributed CNN-ConvLSTM架构，使用数十年的ERA5再分析数据进行训练，并为每个城市优化了架构：Bengaluru(32滤波器)，Mumbai和Delhi(64滤波器)，Kolkata(128滤波器)。使用排列重要性、梯度加权类激活映射、时间遮挡和反事实扰动进行可解释性分析。

### 主要发现

模型在四个城市取得了不同的RMSE值：Bengaluru(0.21 mm/day)，Mumbai(0.52 mm/day)，Delhi(0.48 mm/day)，Kolkata(1.80 mm/day)。模型依赖于城市特定的变量，预测范围从Bengaluru的一天到Kolkata的五天不等。

### 结论

该研究表明可解释人工智能(xAI)能够提供准确的预测，并对不同城市环境中的降水模式提供透明见解。

### 翻译

用于降水预测的深度学习模型通常作为黑箱运行，限制了它们在实际天气预测中的采用。为了在保持准确性的同时提高透明度，我们为印度四个主要城市（班加罗尔、孟买、德里和加尔各答）开发了可解释的深度学习框架，这些城市跨越了不同的气候区。我们在数十年的ERA5再分析数据上实施并训练了混合的Time-Distributed CNN-ConvLSTM（卷积神经网络-长短期记忆）架构。该架构针对每个城市进行了优化，使用不同数量的卷积滤波器：班加罗尔（32）、孟买和德里（64）、加尔各答（128）。模型在各城市取得的均方根误差（RMSE）值分别为：班加罗尔（0.21 mm/day）、孟买（0.52 mm/day）、德里（0.48 mm/day）和加尔各答（1.80 mm/day）。通过使用排列重要性、梯度加权类激活映射（Grad-CAM）、时间遮挡和反事实扰动进行可解释性分析，我们识别出模型行为的不同模式。模型依赖于城市特定的变量，预测范围从班加罗尔的一天到加尔各答的五天不等。本研究展示了可解释人工智能（xAI）如何能够在多样化的城市环境中提供准确的预测和对降水模式的透明见解。


### 论文摘要

Deep learning models for precipitation forecasting often function as black boxes, limiting their adoption in real-world weather prediction. To enhance transparency while maintaining accuracy, we developed an interpretable deep learning framework for short-term precipitation prediction in four major Indian cities: Bengaluru, Mumbai, Delhi, and Kolkata, spanning diverse climate zones. We implemented a hybrid Time-Distributed CNN-ConvLSTM (Convolutional Neural Network-Long Short-Term Memory) architecture, trained on multi-decadal ERA5 reanalysis data. The architecture was optimized for each city with a different number of convolutional filters: Bengaluru (32), Mumbai and Delhi (64), and Kolkata (128). The models achieved root mean square error (RMSE) values of 0.21 mm/day (Bengaluru), 0.52 mm/day (Mumbai), 0.48 mm/day (Delhi), and 1.80 mm/day (Kolkata). Through interpretability analysis using permutation importance, Gradient-weighted Class Activation Mapping (Grad-CAM), temporal occlusion, and counterfactual perturbation, we identified distinct patterns in the model's behavior. The model relied on city-specific variables, with prediction horizons ranging from one day for Bengaluru to five days for Kolkata. This study demonstrates how explainable AI (xAI) can provide accurate forecasts and transparent insights into precipitation patterns in diverse urban environments.

---

## 217. Nonequilibrium Thermodynamics of Associative Memory Continuous-Time Recurrent Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.11150v1](http://arxiv.org/abs/2511.11150v1)

**作者:** Miguel Aguilera, Daniele De Martino, Ivan Garashchuk, Dmitry Sinelshchikov

**发布时间:** 2025-11-14

**备注:** 7 pages, 4 figures

### GPT解析

### 总结

本文提出了一类基于类似Hopfield联想记忆的非对称耦合连续时间循环神经网络(CTRNNs)，该模型结合了联想记忆的表现力与可处理的数学形式，能够直接计算宏观可观测量的演化以及系统的熵和熵耗散，为复杂序列编码网络提供了更可解释的模型。

### 背景

连续时间循环神经网络(CTRNNs)被广泛用于建模复杂的时间行为，但其内部动力学往往难以解释。

### 目的

提出一类新的CTRNNs，基于类似Hopfield联想记忆的非对称耦合，以提高模型的解释性并建立动力学系统描述与统计力学之间的桥梁。

### 方法

开发一类基于类似Hopfield联想记忆的非对称耦合CTRNNs，结合联想记忆的表现力与可处理的数学形式，以描述非平衡动力学的波动。

### 主要发现

该数学描述可以直接计算宏观可观测量（编码特征）的演化，以及系统的瞬时熵和熵耗散，为低维可观测量的动力学系统描述与大非平衡网络的统计力学之间架起桥梁。

### 结论

非平衡联想CTRNNs可以作为更可解释的复杂序列编码网络模型。

### 翻译

连续时间循环神经网络(CTRNNs)因其能够建模复杂时间行为的能力而被广泛使用。然而，它们的内部动力学往往仍然难以解释。在本文中，我们提出了一类基于具有非对称耦合的类似Hopfield联想记忆的新型CTRNNs。该模型将联想记忆的表现力与可处理的数学形式相结合，以描述非平衡动力学的波动。我们表明，这种数学描述允许我们直接计算其宏观可观测量（编码特征）的演化，以及系统的瞬时熵和熵耗散，从而为低维可观测量的动力学系统描述与大非平衡网络的统计力学之间架起桥梁。我们的研究结果表明，这些非平衡联想CTRNNs可以作为复杂序列编码网络的可解释性更强的模型。


### 论文摘要

Continuous-Time Recurrent Neural Networks (CTRNNs) have been widely used for their capacity to model complex temporal behaviour. However, their internal dynamics often remain difficult to interpret. In this paper, we propose a new class of CTRNNs based on Hopfield-like associative memories with asymmetric couplings. This model combines the expressive power of associative memories with a tractable mathematical formalism to characterize fluctuations in nonequilibrium dynamics. We show that this mathematical description allows us to directly compute the evolution of its macroscopic observables (the encoded features), as well as the instantaneous entropy and entropy dissipation of the system, thereby offering a bridge between dynamical systems descriptions of low-dimensional observables and the statistical mechanics of large nonequilibrium networks. Our results suggest that these nonequilibrium associative CTRNNs can serve as more interpretable models for complex sequence-encoding networks.

---

## 218. Impact of Brain Anisotropy on Transcranial Temporal Interference Stimulation: Numerical Analysis Toward Reliable Montage Optimization

**论文链接:** [http://arxiv.org/abs/2511.11129v1](http://arxiv.org/abs/2511.11129v1)

**作者:** Kanata Yatsuda, Masaki Fukunaga, Wenwei Yu, Jose Gomez-Tames

**发布时间:** 2025-11-14

**备注:** 20 pages, 6 figures

### GPT解析

### 总结

本研究探讨了各向异性对经颅时间干扰刺激(TIS)的影响，发现各向异性导率显著改变干扰电场强度，影响TIS电极配置优化，但在实际容差范围内电极配置选择趋于一致。

### 背景

经颅时间干扰刺激(TIS)是一种新型经颅电刺激模式，能够实现对深部脑结构的聚焦靶向。当靶向深部区域时，电流通路会穿过高度各向异性的白质，使得各向异性成为一个关键因素。

### 目的

阐明各向异性如何影响深部脑区的干扰电流，并首次评估其对TIS电极配置优化的影响。

### 方法

比较具有各向异性和各向同性导率的人头导体模型，评估各向异性在颅内干扰电流分布和电极配置优化中的作用。使用扩散加权成像数据推导灰质、白质和深部脑结构的电导率张量，并基于帕累托前沿优化进行电极配置优化。

### 主要发现

在文献报道的TIS电极配置中，各向异性导率显著改变了干扰电场强度，白质中的差异高达18%，而深部脑结构的差异低于12%。对于TIS优化的电极配置，这些导率模型变化仅导致50%的电极配置不一致，但当将聚焦性和靶场强度的差异限制在10%以内时，一致性提高到近90%。

### 结论

纳入各向异性导率对于确定个体化的聚焦性和靶场特征很重要。各向异性可以显著影响最佳电极配置的选择，但在实际的聚焦性和靶场容差范围内，电极配置选择基本趋于一致。

### 翻译

Background & Aim: Transcranial temporal interference stimulation (TIS) is a novel transcranial electrical stimulation modality that enables focused targeting of deep brain structures. When targeting deep regions, current pathways traverse the highly anisotropic white matter, making anisotropy a potentially critical factor. This study aimed to clarify how anisotropy influences interferential currents in deep brain regions and to assess its impact on TIS montage optimization for the first time. Methods: Anatomical head conductor models with anisotropic and isotropic conductivities were compared to evaluate the role of anisotropy in the intracranial interferential currents distributions and montage optimization. For the anisotropic conductivity, conductivity tensors were derived from diffusion-weighted imaging data for gray matter, white matter, and deep brain structures. Montage optimization was conducted based on Pareto front optimization. Results: In literature-reported TIS montages, anisotropic conductivity significantly altered the interferential electric field intensity, with differences of up to 18% in the white matter, whereas discrepancies in deep brain structures remained below 12%. For TIS-optimized montages, these variations across conductivity models yielded only 50% montage disagreement. However, when constraining differences in focality and target field strength to within 10%, the agreement improved to nearly 90%. Conclusions: Incorporating anisotropic conductivities is important to determine individualized focality and target-field characteristics. Moreover, anisotropy can substantially affect the selection of the optimal montage, but under practical focality and target-field tolerances, montage choices largely converge.


### 论文摘要

Background & Aim: Transcranial temporal interference stimulation (TIS) is a novel transcranial electrical stimulation modality that enables focused targeting of deep brain structures. When targeting deep regions, current pathways traverse the highly anisotropic white matter, making anisotropy a potentially critical factor. This study aimed to clarify how anisotropy influences interferential currents in deep brain regions and to assess its impact on TIS montage optimization for the first time. Methods: Anatomical head conductor models with anisotropic and isotropic conductivities were compared to evaluate the role of anisotropy in the intracranial interferential currents distributions and montage optimization. For the anisotropic conductivity, conductivity tensors were derived from diffusion-weighted imaging data for gray matter, white matter, and deep brain structures. Montage optimization was conducted based on Pareto front optimization. Results: In literature-reported TIS montages, anisotropic conductivity significantly altered the interferential electric field intensity, with differences of up to 18% in the white matter, whereas discrepancies in deep brain structures remained below 12%. For TIS-optimized montages, these variations across conductivity models yielded only 50% montage disagreement. However, when constraining differences in focality and target field strength to within 10%, the agreement improved to nearly 90%. Conclusions: Incorporating anisotropic conductivities is important to determine individualized focality and target-field characteristics. Moreover, anisotropy can substantially affect the selection of the optimal montage, but under practical focality and target-field tolerances, montage choices largely converge.

---

## 219. Toward Generalized Detection of Synthetic Media: Limitations, Challenges, and the Path to Multimodal Solutions

**论文链接:** [http://arxiv.org/abs/2511.11116v1](http://arxiv.org/abs/2511.11116v1)

**作者:** Redwan Hussain, Mizanur Rahman, Prithwiraj Bhattacharjee

**发布时间:** 2025-11-14

**备注:** 10 Pages, 4 figures, 1 table, 7th International Conference on Trends in Computational and Cognitive Engineering(TCCE-2025)

### GPT解析

### 总结

该研究回顾了二十四个关于AI生成媒体检测的最新工作，分析了当前方法的局限和挑战，并提出了基于多模态深度学习模型的未来研究方向。

### 背景

人工智能在媒体领域迅速发展，GANs和扩散模型等技术提高了生成内容的质量，使得真实与合成内容难以区分。深度伪造技术被滥用传播错误信息，因此开发了多种检测模型，但这些模型在泛化能力和处理多模态数据方面存在局限。

### 目的

回顾AI生成媒体检测的最新研究，识别贡献和弱点，总结共同局限和关键挑战，并提出未来研究方向。

### 方法

对二十四个关于AI生成媒体检测的研究进行单独检查和分析，总结共同问题和挑战，并提出基于多模态深度学习模型的解决方案。

### 主要发现

当前检测模型难以在未见过的数据上泛化，难以处理来自不同模型的内容，并且在处理多模态数据和高度修改的内容方面效果不佳。

### 结论

多模态深度学习模型有潜力提供更强大和泛化的检测能力，为构建针对有害合成媒体的更强防御提供了明确的研究起点。

### 翻译

人工智能在媒体领域的应用在过去十年中迅速发展。生成对抗网络的引入提高了照片级真实感图像生成的质量。扩散模型后来带来了生成媒体的新时代。这些进步使得真实和合成内容难以区分。深度伪造的兴起展示了这些工具如何被滥用以传播错误信息、政治阴谋、隐私侵犯和欺诈。因此，许多检测模型被开发出来。它们通常使用卷积神经网络和视觉变换器等深度学习方法。这些模型搜索视觉、空间或时间异常。然而，这些方法通常难以在未见过的数据上泛化，并且难以处理来自不同模型的内容。此外，现有方法在处理多模态数据和高度修改的内容方面效果不佳。本研究回顾了二十四篇关于AI生成媒体检测的最新研究。每项研究都经过单独检查，以确定其贡献和弱点。该综述随后总结了当前方法的共同局限和关键挑战。基于此分析，提出了一个研究方向，重点关注多模态深度学习模型。这类模型有潜力提供更强大和泛化的检测能力，为未来研究人员构建针对有害合成媒体的更强防御提供了明确的起点。


### 论文摘要

Artificial intelligence (AI) in media has advanced rapidly over the last decade. The introduction of Generative Adversarial Networks (GANs) improved the quality of photorealistic image generation. Diffusion models later brought a new era of generative media. These advances made it difficult to separate real and synthetic content. The rise of deepfakes demonstrated how these tools could be misused to spread misinformation, political conspiracies, privacy violations, and fraud. For this reason, many detection models have been developed. They often use deep learning methods such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). These models search for visual, spatial, or temporal anomalies. However, such approaches often fail to generalize across unseen data and struggle with content from different models. In addition, existing approaches are ineffective in multimodal data and highly modified content. This study reviews twenty-four recent works on AI-generated media detection. Each study was examined individually to identify its contributions and weaknesses, respectively. The review then summarizes the common limitations and key challenges faced by current approaches. Based on this analysis, a research direction is suggested with a focus on multimodal deep learning models. Such models have the potential to provide more robust and generalized detection. It offers future researchers a clear starting point for building stronger defenses against harmful synthetic media.

---

## 220. VIDEOP2R: Video Understanding from Perception to Reasoning

**论文链接:** [http://arxiv.org/abs/2511.11113v1](http://arxiv.org/abs/2511.11113v1)

**作者:** Yifan Jiang, Yueying Wang, Rui Zhao, Toufiq Parag, Zhimin Chen, Zhenyu Liao, Jayakrishnan Unnikrishnan

**发布时间:** 2025-11-14

### GPT解析

### 总结

VideoP2R是一种新颖的感知视频强化微调框架，通过将感知和推理建模为不同过程来增强视频语言模型的推理能力，在七个基准测试中的六个上达到了最先进性能。

### 背景

强化微调(RFT)是一种包含监督微调(SFT)和强化学习(RL)的两阶段框架，在提高大语言模型的推理能力方面显示出有希望的结果，然而将其扩展到大视频语言模型仍具挑战性。

### 目的

提出VideoP2R框架，通过建模感知和推理作为不同过程来增强视频语言模型的视频推理能力。

### 方法

在SFT阶段，开发三步管道生成VideoP2R-CoT-162K高质量感知思考链数据集；在RL阶段，引入感知感知组相对策略优化(PA-GRPO)算法，为感知和推理提供单独奖励。

### 主要发现

大量实验表明VideoP2R在七个视频推理和理解基准测试中的六个上达到最先进性能；消融研究证实了感知感知建模和PA-GRPO的有效性，并表明模型的感知输出对下游推理信息充分。

### 结论

VideoP2R框架成功通过将感知和推理建模为不同过程，提高了大视频语言模型的视频推理能力。

### 翻译

强化微调(RFT)是一种包含监督微调(SFT)和强化学习(RL)的两阶段框架，在提高大语言模型(LLMs)的推理能力方面显示出有希望的结果。然而，将RFT扩展到大视频语言模型(LVLMs)仍然具有挑战性。我们提出VideoP2R，一种新颖的感知视频RFT框架，通过将感知和推理建模为不同过程来增强视频推理能力。在SFT阶段，我们开发了一个三步管道来生成VideoP2R-CoT-162K，这是一个高质量的、感知感知的思考链(CoT)数据集，用于感知和推理。在RL阶段，我们引入了一种新颖的感知感知组相对策略优化(PA-GRPO)算法，为感知和推理提供单独的奖励。大量实验表明，VideoP2R在七个视频推理和理解基准测试中的六个上达到了最先进的性能。消融研究进一步证实了我们的感知感知建模和PA-GRPO的有效性，并表明模型的感知输出对于下游推理是信息充分的。


### 论文摘要

Reinforcement fine-tuning (RFT), a two-stage framework consisting of supervised fine-tuning (SFT) and reinforcement learning (RL) has shown promising results on improving reasoning ability of large language models (LLMs). Yet extending RFT to large video language models (LVLMs) remains challenging. We propose VideoP2R, a novel process-aware video RFT framework that enhances video reasoning by modeling perception and reasoning as distinct processes. In the SFT stage, we develop a three-step pipeline to generate VideoP2R-CoT-162K, a high-quality, process-aware chain-of-thought (CoT) dataset for perception and reasoning. In the RL stage, we introduce a novel process-aware group relative policy optimization (PA-GRPO) algorithm that supplies separate rewards for perception and reasoning. Extensive experiments show that VideoP2R achieves state-of-the-art (SotA) performance on six out of seven video reasoning and understanding benchmarks. Ablation studies further confirm the effectiveness of our process-aware modeling and PA-GRPO and demonstrate that model's perception output is information-sufficient for downstream reasoning.

---

## 221. AccKV: Towards Efficient Audio-Video LLMs Inference via Adaptive-Focusing and Cross-Calibration KV Cache Optimization

**论文链接:** [http://arxiv.org/abs/2511.11106v1](http://arxiv.org/abs/2511.11106v1)

**作者:** Zhonghua Jiang, Kui Chen, Kunxi Li, Keting Yin, Yiyun Zhou, Zhaode Wang, Chengfei Lv, Shengyu Zhang

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为AccKV的自适应聚焦和交叉校准KV缓存优化框架，用于提高视听大语言模型(AV-LLMs)的推理效率。研究发现，AV-LLMs在高层对不同模态的关注并不严格依赖于任务，且直接集成音频和视频的KV缓存可能导致信息混淆和性能下降。AccKV通过层自适应聚焦技术和交叉校准技术解决了这些问题，实验证明该方法能在保持准确性的同时显著提高计算效率。

### 背景

视听大语言模型(AV-LLMs)在视听问答和多模态对话系统等任务中能力不断增强。视频和音频引入了扩展的时间维度，导致比静态图像嵌入更大的键值缓存。现有的优化策略是选择性关注和保留基于任务的音频或视频KV缓存。

### 目的

解决AV-LLMs在处理音频和视频模态时面临的KV缓存效率问题，提高推理效率同时保持模型准确性。

### 方法

提出AccKV框架，包括：(1)基于层自适应聚焦技术，根据不同层的特性选择性关注关键模态；(2)通过注意力重分配增强对高频标记的识别；(3)交叉校准技术，整合音频和视频模态内的低效KV缓存，对齐低优先级模态与高优先级模态，选择性驱逐低优先级模态的KV缓存。

### 主要发现

1) AV-LLMs在高层对不同模态的关注并不严格依赖于任务；2) 在较高层，AV-LLMs的关注更多地转向视频模态；3) 直接集成音频的时间KV和视频的时空KV可能导致信息混淆和性能显著下降；4) 不加区分地处理音频和视频可能导致某一模态的过度压缩或保留，破坏模态之间的对齐。

### 结论

实验结果表明，AccKV可以在保持准确性的同时显著提高AV-LLMs的计算效率，解决了AV-LLMs在处理多模态数据时的KV缓存优化问题。

### 翻译

近期视听大语言模型(AV-LLMs)的进展增强了它们在视听问答和多模态对话系统等任务中的能力。视频和音频引入了扩展的时间维度，导致与静态图像嵌入相比更大的键值缓存。一个简单的优化策略是基于任务选择性地关注和保留音频或视频的KV缓存。然而，在实验中，我们观察到AV-LLMs在高层对不同模态的关注并不严格依赖于任务。在较高层，AV-LLMs的关注更多地转向视频模态。此外，我们还发现直接集成音频的时间KV和视频的时空KV可能导致信息混淆和AV-LLMs性能的显著下降。如果音频和视频不加区分地处理，也可能导致某一模态的过度压缩或保留，从而破坏模态之间的对齐。为应对这些挑战，我们提出了AccKV，一个专为高效AV-LLMs推理设计的自适应聚焦和交叉校准KV缓存优化框架。我们的方法基于层自适应聚焦技术，根据不同层的特性选择性关注关键模态，并通过注意力重分配增强对高频标记的识别。此外，我们提出了交叉校准技术，首先整合音频和视频模态内的低效KV缓存，然后将低优先级模态与高优先级模态对齐，选择性驱逐低优先级模态的KV缓存。实验结果表明，AccKV可以在保持准确性的同时显著提高AV-LLMs的计算效率。


### 论文摘要

Recent advancements in Audio-Video Large Language Models (AV-LLMs) have enhanced their capabilities in tasks like audio-visual question answering and multimodal dialog systems. Video and audio introduce an extended temporal dimension, resulting in a larger key-value (KV) cache compared to static image embedding. A naive optimization strategy is to selectively focus on and retain KV caches of audio or video based on task. However, in the experiment, we observed that the attention of AV-LLMs to various modalities in the high layers is not strictly dependent on the task. In higher layers, the attention of AV-LLMs shifts more towards the video modality. In addition, we also found that directly integrating temporal KV of audio and spatial-temporal KV of video may lead to information confusion and significant performance degradation of AV-LLMs. If audio and video are processed indiscriminately, it may also lead to excessive compression or reservation of a certain modality, thereby disrupting the alignment between modalities. To address these challenges, we propose AccKV, an Adaptive-Focusing and Cross-Calibration KV cache optimization framework designed specifically for efficient AV-LLMs inference. Our method is based on layer adaptive focusing technology, selectively focusing on key modalities according to the characteristics of different layers, and enhances the recognition of heavy hitter tokens through attention redistribution. In addition, we propose a Cross-Calibration technique that first integrates inefficient KV caches within the audio and video modalities, and then aligns low-priority modalities with high-priority modalities to selectively evict KV cache of low-priority modalities. The experimental results show that AccKV can significantly improve the computational efficiency of AV-LLMs while maintaining accuracy.

---

## 222. ARCTraj: A Dataset and Benchmark of Human Reasoning Trajectories for Abstract Problem Solving

**论文链接:** [http://arxiv.org/abs/2511.11079v1](http://arxiv.org/abs/2511.11079v1)

**作者:** Sejin Kim, Hayan Choi, Seokki Lee, Sundong Kim

**发布时间:** 2025-11-14

### GPT解析

### 总结

ARCTraj是一个数据集和方法论框架，用于通过ARC中的复杂视觉任务建模人类推理，记录了人类如何将输入迭代转换为输出的时间顺序行动，揭示了传统数据集忽略的中间推理步骤。

### 背景

ARC已经激发了关于抽象推理的大量研究，但大多数现有方法依赖于静态的输入-输出监督，限制了人们对推理如何随时间展开的理解。

### 目的

解决现有方法对推理过程动态性理解不足的问题，通过记录对象级别的行动来捕捉人类推理的中间步骤。

### 方法

定义了一个统一的推理流程，包括数据收集、动作抽象、马尔可夫决策过程公式化和下游学习，使其能够与多种机器学习方法（如PPO、World Models、GFlowNets等）集成。

### 主要发现

对空间选择、颜色归属和战略收敛的分析突显了人类推理的结构和多样性，证明了记录推理过程的价值。

### 结论

ARCTraj为研究类人推理提供了结构化和可解释的基础，有助于推动可解释性、对齐和通用智能的发展。

### 翻译

我们提出了ARCTraj，这是一个数据集和方法论框架，用于通过Abstraction and Reasoning Corpus (ARC)中的复杂视觉任务来建模人类推理。虽然ARC已经激发了关于抽象推理的大量研究，但大多数现有方法依赖于静态的输入-输出监督，这限制了人们对推理如何随时间展开的理解。ARCTraj通过记录按时间排序的、对象级别的行动解决了这一差距，这些行动捕捉了人类如何将输入迭代地转换为输出，从而揭示了传统数据集忽略的中间推理步骤。通过O2ARC网页界面收集，它包含约10,000个轨迹，这些轨迹跨越ARC-AGI-1基准测试中的400个训练任务，并带有任务标识符、时间戳和成功标签。它进一步定义了一个统一的推理流程，包括数据收集、动作抽象、马尔可夫决策过程公式化和下游学习，使其能够与强化学习、生成建模和序列建模方法（如PPO、World Models、GFlowNets、Diffusion agents和Decision Transformers）集成。对空间选择、颜色归属和战略收敛的分析突显了人类推理的结构和多样性。这些贡献共同将ARCTraj定位为研究类人推理的结构化和可解释基础，推动了可解释性、对齐和通用智能的发展。


### 论文摘要

We present ARCTraj, a dataset and methodological framework for modeling human reasoning through complex visual tasks in the Abstraction and Reasoning Corpus (ARC). While ARC has inspired extensive research on abstract reasoning, most existing approaches rely on static input--output supervision, which limits insight into how reasoning unfolds over time. ARCTraj addresses this gap by recording temporally ordered, object-level actions that capture how humans iteratively transform inputs into outputs, revealing intermediate reasoning steps that conventional datasets overlook. Collected via the O2ARC web interface, it contains around 10,000 trajectories annotated with task identifiers, timestamps, and success labels across 400 training tasks from the ARC-AGI-1 benchmark. It further defines a unified reasoning pipeline encompassing data collection, action abstraction, Markov decision process (MDP) formulation, and downstream learning, enabling integration with reinforcement learning, generative modeling, and sequence modeling methods such as PPO, World Models, GFlowNets, Diffusion agents, and Decision Transformers. Analyses of spatial selection, color attribution, and strategic convergence highlight the structure and diversity of human reasoning. Together, these contributions position ARCTraj as a structured and interpretable foundation for studying human-like reasoning, advancing explainability, alignment, and generalizable intelligence.

---

## 223. LiteAttention: A Temporal Sparse Attention for Diffusion Transformers

**论文链接:** [http://arxiv.org/abs/2511.11062v1](http://arxiv.org/abs/2511.11062v1)

**作者:** Dor Shmilovich, Tony Wu, Aviad Dahan, Yuval Domb

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出LiteAttention方法，通过利用扩散注意力的时间相干性，实现视频生成扩散模型中动态跳过不必要的注意力计算，显著提高计算效率同时保持生成质量。

### 背景

扩散变换器在视频生成中能实现卓越质量，但存在二次注意力复杂度导致的过高延迟问题。现有加速方法面临基本权衡：动态估计稀疏注意力模式带来高计算开销和估计误差，而静态稀疏模式在整个去噪过程中固定且通常次优。

### 目的

解决扩散变换器在视频生成中的计算效率问题，通过减少注意力计算复杂度降低延迟，同时保持生成质量不受影响。

### 方法

提出LiteAttention方法，利用扩散注意力的时间相干性特性：在步骤t中被认为非必要的tile通常在步骤t+δ仍然如此。通过早期标记非必要的tile并将跳过决策向前传播，消除冗余的注意力计算，无需重复剖析开销，结合动态方法的适应性和静态方法的效率。

### 主要发现

扩散注意力的稀疏模式在去噪步骤间具有强时间相干性；早期标记非必要tile并传播跳过决策可显著减少计算量；Lite能在不降低质量的情况下实现实质性加速。

### 结论

LiteAttention方法通过利用扩散注意力的时间相干性，成功结合动态方法的适应性和静态方法的效率，在保持生成质量的同时显著提高了视频扩散模型的计算效率。

### 翻译

扩散变换器，特别是用于视频生成时，能实现卓越的质量，但受到二次注意力复杂度的困扰，导致难以接受的延迟。现有的加速方法面临一个基本的权衡：在每个去噪步骤动态估计稀疏注意力模式会带来高计算开销和估计误差，而静态稀疏模式在整个去噪过程中保持固定且通常次优。我们确定了扩散注意力的一个关键结构特性，即其稀疏模式在去噪步骤之间表现出强时间相干性。在步骤t中被认为非必要的tile通常在步骤t+δ仍然如此。利用这一观察，我们引入了LiteAttention，一种利用时间相干性在去噪序列中实现进化计算跳过的方法。通过早期标记非必要的tile并将跳过决策向前传播，LiteAttention消除了冗余的注意力计算，无需重复的剖析开销，结合了动态方法的适应性和静态方法的效率。我们在FlashAttention基础上实现了高度优化的LiteAttention内核，并在生产视频扩散模型上展示了显著的加速效果，同时没有质量下降。代码和实现细节将公开发布。


### 论文摘要

Diffusion Transformers, particularly for video generation, achieve remarkable quality but suffer from quadratic attention complexity, leading to prohibitive latency. Existing acceleration methods face a fundamental trade-off: dynamically estimating sparse attention patterns at each denoising step incurs high computational overhead and estimation errors, while static sparsity patterns remain fixed and often suboptimal throughout denoising. We identify a key structural property of diffusion attention, namely, its sparsity patterns exhibit strong temporal coherence across denoising steps. Tiles deemed non-essential at step $t$ typically remain so at step $t+δ$. Leveraging this observation, we introduce LiteAttention, a method that exploits temporal coherence to enable evolutionary computation skips across the denoising sequence. By marking non-essential tiles early and propagating skip decisions forward, LiteAttention eliminates redundant attention computations without repeated profiling overheads, combining the adaptivity of dynamic methods with the efficiency of static ones. We implement a highly optimized LiteAttention kernel on top of FlashAttention and demonstrate substantial speedups on production video diffusion models, with no degradation in quality. The code and implementation details will be publicly released.

---

## 224. TimeAudio: Bridging Temporal Gaps in Large Audio-Language Models

**论文链接:** [http://arxiv.org/abs/2511.11039v1](http://arxiv.org/abs/2511.11039v1)

**作者:** Hualei Wang, Yiming Li, Shuo Ma, Hong Liu, Xiangdong Wang

**发布时间:** 2025-11-14

**备注:** Accepted by The Fortieth AAAI Conference on Artificial Intelligence (AAAI 2026)

### GPT解析

### 总结

TimeAudio是一种新型方法，通过引入时间标记和绝对时间感知编码，解决了大型音频语言模型在时间定位和长音频理解方面的局限性，并通过分段级令牌合并提高了效率。

### 背景

大型音频语言模型在对话式问答任务中表现出理解音频内容的能力，但在时间戳理解和长音频感知方面存在局限，导致细粒度任务能力受限。

### 目的

解决LALMs在时间定位和长音频理解方面的局限性，使模型能够将音频内容理解与精确时间感知联系起来。

### 方法

引入独特的时间标记提高时间敏感推理；应用绝对时间感知编码将声学特征与绝对时间信息关联；引入分段级令牌合并模块减少音频令牌冗余并提高信息提取效率；创建专注于时间任务的新数据集和评估指标。

### 主要发现

TimeAudio在密集标注、时间定位和时间线语音摘要等多种细粒度任务上表现强劲，证明了其强大的时间定位和推理能力。

### 结论

TimeAudio有效解决了LALMs在时间定位和长音频理解方面的局限性，提升了模型在细粒度音频任务中的性能。

### 翻译

最近的大型音频语言模型在对话式问答任务中表现出理解音频内容的能力。然而，这些模型难以准确理解时间戳进行时间定位（如时间音频定位），并且仅限于短音频感知，导致在细粒度任务上能力受限。我们确定了限制其时间定位和长音频理解的三个关键方面：（i）时间戳表示，（ii）架构，和（iii）数据。为此，我们引入了TimeAudio，一种新型方法，使LALMs能够将其对音频内容的理解与精确的时间感知联系起来。具体而言，我们引入独特的时间标记来提高时间敏感推理，并应用绝对时间感知编码，将声学特征与绝对时间信息明确关联。此外，为实现端到端的长音频理解，我们引入了分段级令牌合并模块，大幅减少音频令牌冗余并提高信息提取效率。由于缺乏合适的数据集和评估指标，我们将现有的音频数据集整合为一个专注于时间任务的新数据集，并建立了一系列评估细粒度性能的指标。评估显示在多种细粒度任务（如密集标注、时间定位和时间线语音摘要）上表现强劲，证明了TimeAudio强大的时间定位和推理能力。


### 论文摘要

Recent Large Audio-Language Models (LALMs) exhibit impressive capabilities in understanding audio content for conversational QA tasks. However, these models struggle to accurately understand timestamps for temporal localization (e.g., Temporal Audio Grounding) and are restricted to short audio perception, leading to constrained capabilities on fine-grained tasks. We identify three key aspects that limit their temporal localization and long audio understanding: (i) timestamp representation, (ii) architecture, and (iii) data. To address this, we introduce TimeAudio, a novel method that empowers LALMs to connect their understanding of audio content with precise temporal perception. Specifically, we incorporate unique temporal markers to improve time-sensitive reasoning and apply an absolute time-aware encoding that explicitly grounds the acoustic features with absolute time information. Moreover, to achieve end-to-end long audio understanding, we introduce a segment-level token merging module to substantially reduce audio token redundancy and enhance the efficiency of information extraction. Due to the lack of suitable datasets and evaluation metrics, we consolidate existing audio datasets into a new dataset focused on temporal tasks and establish a series of metrics to evaluate the fine-grained performance. Evaluations show strong performance across a variety of fine-grained tasks, such as dense captioning, temporal grounding, and timeline speech summarization, demonstrating TimeAudio's robust temporal localization and reasoning capabilities.

---

## 225. PAS: A Training-Free Stabilizer for Temporal Encoding in Video LLMs

**论文链接:** [http://arxiv.org/abs/2511.10979v1](http://arxiv.org/abs/2511.10979v1)

**作者:** Bowen Sun, Yujun Cai, Ming-Hsuan Yang, Hang Wu, Yiwei Wang

**发布时间:** 2025-11-14

**备注:** 13 pages, 5 figures

### GPT解析

### 总结

Video LLMs存在时间不一致性问题，作者提出相位聚合平滑(PAS)方法解决了这一问题，通过在多头间应用相反相位偏移并聚合输出来平滑时间核，提高模型对时间变化的鲁棒性。

### 背景

Video LLMs存在时间不一致性问题，帧时间的小变化会翻转注意力并抑制相关帧，影响模型性能。

### 目的

解决Video LLMs中的时间不一致性问题，提高模型对时间变化的鲁棒性。

### 方法

提出相位聚合平滑(PAS)机制，在多头间应用小的相反相位偏移，然后聚合它们的输出，保留每头频谱幅度的同时平滑时间核。

### 主要发现

RoPE旋转的logit可近似为内容点积乘以时间核；平滑该核可获得对小的时间移位的Lipschitz稳定性；多相位平均衰减高频涟漪同时保留每头频谱。

### 结论

PAS为Video LLMs中的鲁棒时间编码提供了即插即用的升级，在多个视频理解基准测试中表现出一致改进且计算开销可忽略。

### 翻译

视频大语言模型(LLMs)遭受时间不一致性问题：帧时间的小变化会翻转注意力并抑制相关帧。我们将这种不稳定性追踪到通过多模态RoPE将旋转位置编码扩展到视频的常见做法。诱导的反傅里叶时间核表现出帧尺度的涟漪，以不同因子乘以相邻帧，干扰了应由原始查询键内积支配的注意力。我们提出相位聚合平滑(PAS)，一种简单、无需训练的机制，在头部间应用小的相反相位偏移，然后聚合它们的输出。PAS保留了每头频谱幅度，而聚合有效地平滑了时间核并减少了相位敏感性，同时不改变位置编码结构。我们的分析表明，RoPE旋转的logit可以近似为内容点积乘以时间核；平滑该核可获得对小的时间移位的Lipschitz稳定性；多相位平均衰减高频涟漪，同时在奈奎斯特有效采样下保留每头频谱。在多个视频理解基准测试中，在匹配的标记预算下，实验显示出一致的改进，且计算开销可忽略。PAS为Video LLMs中的鲁棒时间编码提供了即插即用的升级。


### 论文摘要

Video LLMs suffer from temporal inconsistency: small shifts in frame timing can flip attention and suppress relevant frames. We trace this instability to the common extension of Rotary Position Embeddings to video through multimodal RoPE. The induced inverse Fourier time kernel exhibits frame-scale ripples that multiply adjacent frames by different factors, which perturbs attention that should otherwise be governed by the raw query key inner product. We present Phase Aggregated Smoothing (PAS), a simple, training-free mechanism that applies small opposed phase offsets across heads and then aggregates their outputs. PAS preserves the per-head spectrum magnitude, while the aggregation effectively smooths the temporal kernel and reduces phase sensitivity without changing the positional encoding structure. Our analysis shows that the RoPE rotated logit can be approximated as a content dot product scaled by a time kernel; smoothing this kernel yields Lipschitz stability of attention to small temporal shifts; multi phase averaging attenuates high frequency ripples while preserving per-head spectra under Nyquist-valid sampling. Experiments on multiple video understanding benchmarks under matched token budgets show consistent improvements with negligible computational overhead. PAS provides a plug and play upgrade for robust temporal encoding in Video LLMs.

---

## 226. Text-guided Weakly Supervised Framework for Dynamic Facial Expression Recognition

**论文链接:** [http://arxiv.org/abs/2511.10958v1](http://arxiv.org/abs/2511.10958v1)

**作者:** Gunho Jung, Heejo Kong, Seong-Whan Lee

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为TG-DFER的文本引导弱监督框架，通过语义引导和连贯的时间建模来增强基于多示例学习的动态面部表情识别方法，解决了视觉多样性和时序动态复杂性问题。

### 背景

动态面部表情识别(DFER)旨在通过建模视频序列中面部运动的时序变化来识别情绪状态。面临的主要挑战是'多对一'标签问题，即由许多帧组成的视频被分配单一的情绪标签。

### 目的

解决DFER中基于多示例学习(MIL)的方法因情感表达视觉多样性和时序动态复杂性而面临的固有挑战。

### 方法

1. 提出TG-DFER文本引导弱监督框架；2. 集成视觉语言预训练模型提供语义引导；3. 引入视觉提示对齐文本情绪标签与视觉特征；4. 设计多粒度时间网络联合捕获短期面部动态和长期情感流。

### 主要发现

TG-DFER在弱监督下实现了更好的泛化能力、可解释性和时间敏感性。

### 结论

通过整合语义引导和连贯时间建模，TG-DFER有效解决了DFER中的多对一标签问题和视觉多样性、时序动态复杂性等挑战。

### 翻译

动态面部表情识别(DFER)旨在通过建模视频序列中面部运动的时序变化来识别情绪状态。DFER中的一个关键挑战是'多对一'标签问题，其中由许多帧组成的视频被分配单一的情绪标签。缓解此问题的常见策略是将DFER表述为多示例学习(MIL)问题。然而，基于MIL的方法固有地受到情感表达的视觉多样性和时序动态复杂性的困扰。为应对这一挑战，我们提出了TG-DFER，一种文本引导的弱监督框架，通过整合语义引导和连贯的时间建模来增强基于MIL的DFER。我们集成了一个视觉语言预训练(VLP)模型，通过情感上下文的细粒度文本描述提供语义引导。此外，我们引入了视觉提示，使丰富的文本情绪标签与视觉实例特征对齐，实现细粒度推理和帧级相关性估计。此外，还设计了一个多粒度时间网络，联合捕获短期面部动态和长期情感流，确保时间上的连贯情感理解。大量结果表明，TG-DFER在弱监督下实现了更好的泛化能力、可解释性和时间敏感性。


### 论文摘要

Dynamic facial expression recognition (DFER) aims to identify emotional states by modeling the temporal changes in facial movements across video sequences. A key challenge in DFER is the many-to-one labeling problem, where a video composed of numerous frames is assigned a single emotion label. A common strategy to mitigate this issue is to formulate DFER as a Multiple Instance Learning (MIL) problem. However, MIL-based approaches inherently suffer from the visual diversity of emotional expressions and the complexity of temporal dynamics. To address this challenge, we propose TG-DFER, a text-guided weakly supervised framework that enhances MIL-based DFER by incorporating semantic guidance and coherent temporal modeling. We incorporate a vision-language pre-trained (VLP) model is integrated to provide semantic guidance through fine-grained textual descriptions of emotional context. Furthermore, we introduce visual prompts, which align enriched textual emotion labels with visual instance features, enabling fine-grained reasoning and frame-level relevance estimation. In addition, a multi-grained temporal network is designed to jointly capture short-term facial dynamics and long-range emotional flow, ensuring coherent affective understanding across time. Extensive results demonstrate that TG-DFER achieves improved generalization, interpretability, and temporal sensitivity under weak supervision.

---

## 227. GFT: Graph Feature Tuning for Efficient Point Cloud Analysis

**论文链接:** [http://arxiv.org/abs/2511.10799v1](http://arxiv.org/abs/2511.10799v1)

**作者:** Manish Dhakal, Venkat R. Dasari, Raj Sunderraman, Yi Ding

**发布时间:** 2025-11-13

**备注:** WACV 2026

### GPT解析

### 总结

研究提出了一种名为Graph Features Tuning (GFT)的点云专用参数高效微调方法，通过学习动态图并使用轻量级图卷积网络，减少了可训练参数数量，同时在目标分类和分割任务上与现有方法相当。

### 背景

参数高效微调(PEFT)通过只更新模型的一小部分参数，显著降低了计算和内存成本，使模型能够更快地适应新任务且性能损失最小。先前的研究已经引入了针对点云数据的PEFT方法，因为通用方法在这些特定数据上表现不佳。

### 目的

进一步减少可训练参数的数量，开发一种专门针对点云数据的参数高效微调方法。

### 方法

提出了一种名为Graph Features Tuning (GFT)的点云专用PEFT方法，使用轻量级图卷积网络从初始标记化的transformer输入中学习动态图，并通过跳跃连接和高效的交叉注意力模块将这些图特征传递到更深的层。

### 主要发现

在目标分类和分割任务上的大量实验表明，GFT在相同领域内运行，能够与现有方法相媲美，同时减少了可训练参数的数量。

### 结论

GFT是一种有效的点云数据处理方法，能够在保持性能的同时显著减少参数数量，对于资源受限环境下的点云模型微调具有实际应用价值。

### 翻译

参数高效微调(PEFT)通过仅更新模型参数的一小部分，显著降低了计算和内存成本，使模型能够以最小的性能损失更快地适应新任务。先前的研究已经引入了针对点云数据的PEFT，因为通用方法在这些数据上表现不佳。为了进一步减少可训练参数的数量，我们提出了一种点云专用的PEFT，称为图特征微调(GFT)，该方法使用轻量级图卷积网络从transformer的初始标记化输入中学习动态图，并通过跳跃连接和高效的交叉注意力模块将这些图特征传递到更深的层。在目标分类和分割任务上的大量实验表明，GFT在相同领域内运行，与现有方法相媲美，同时减少了可训练参数。代码位于https://github.com/manishdhakal/GFT。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云分析中的参数高效微调问题。传统全参数微调计算和存储成本高，而现有通用PEFT方法在点云任务上表现不佳。这个问题很重要，因为点云分析广泛应用于自动驾驶、机器人等领域，减少可训练参数能显著降低计算和内存需求，使模型能更快适应新任务同时保持良好性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析点云数据的特殊性（无序性、局部几何结构重要性），指出现有PEFT方法的不足。他们借鉴了图神经网络（特别是EdgeConv）提取局部几何特征，利用Transformer中自注意力机制可解释为动态图的特性，以及视觉和语言领域的提示调优方法。通过结合这些思想，作者设计了从Transformer初始输入学习动态图，并通过轻量级图卷积网络和交叉注意力模块将图特征注入到更深层次的方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过学习点云数据的动态图结构并将图特征注入Transformer实现高效参数微调。整体流程包括：1)添加任务特定可学习提示到Transformer嵌入空间增加图自由度；2)使用EdgeConv模块从初始嵌入中提取图特征，通过KNN构建图结构并多层卷积；3)通过交叉注意力模块将图特征稀疏地注入到特定Transformer层（如第1、4、7、10层），实现轻量级高效微调。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专门针对点云数据的PEFT方法；2)从Transformer初始输入学习动态图并利用局部几何结构；3)轻量级设计仅更新0.73M参数（占原始3.26%）；4)结合提示、EdgeConv和交叉注意力三个模块。相比IDPT，GFT从早期标记提取特征并选择性注入，参数更少且性能更稳定；相比DAPT，GFT专注于图特征学习而非动态尺度；相比通用PEFT，GFT考虑了点云数据的特殊性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GFT通过结合任务特定提示、EdgeConv图特征提取和交叉注意力交互，实现了点云分析中的高效参数微调，显著减少了可训练参数数量，同时保持了与现有方法相当的性能。'}


### 论文摘要

Parameter-efficient fine-tuning (PEFT) significantly reduces computational and memory costs by updating only a small subset of the model's parameters, enabling faster adaptation to new tasks with minimal loss in performance. Previous studies have introduced PEFTs tailored for point cloud data, as general approaches are suboptimal. To further reduce the number of trainable parameters, we propose a point-cloud-specific PEFT, termed Graph Features Tuning (GFT), which learns a dynamic graph from initial tokenized inputs of the transformer using a lightweight graph convolution network and passes these graph features to deeper layers via skip connections and efficient cross-attention modules. Extensive experiments on object classification and segmentation tasks show that GFT operates in the same domain, rivalling existing methods, while reducing the trainable parameters. Code is at https://github.com/manishdhakal/GFT.

---

## 228. IFG: Internet-Scale Guidance for Functional Grasping Generation

**论文链接:** [http://arxiv.org/abs/2511.09558v1](http://arxiv.org/abs/2511.09558v1)

**作者:** Ray Muxin Liu, Mingxuan Li, Kenneth Shaw, Deepak Pathak

**发布时间:** 2025-11-12

**备注:** Website at https://ifgrasping.github.io/

### GPT解析

### 总结

本研究提出了一种结合大型视觉模型与基于模拟的力封闭抓取生成管道的方法，实现了高性能的语义抓取，无需手动收集训练数据。

### 背景

大型视觉模型在互联网规模数据训练下，能够在杂乱场景中分割和语义理解对象部分，但缺乏精确控制机械手进行3D抓取所需的几何理解能力。

### 目的

解决大型视觉模型在机器人抓取应用中缺乏几何理解的问题，实现精确的3D抓取控制。

### 方法

利用模拟与力封闭抓取生成管道理解场景中手部和物体的局部几何形状，将此管道产生的数据提炼为在相机点云上实时运行的扩散模型，结合大型视觉模型的全局语义理解和基于模拟的局部感知力封闭的几何精度。

### 主要发现

通过结合全局语义理解和局部几何精度，实现了高性能的语义抓取，无需任何手动收集的训练数据。

### 结论

所提出的方法成功解决了大型视觉模型在机器人抓取应用中的几何理解不足问题，实现了无需手动训练数据的高性能语义抓取。

### 翻译

在互联网规模数据上训练的大型视觉模型在分割和语义理解对象部分方面表现出色，即使在杂乱拥挤的场景中也是如此。然而，虽然这些模型可以将机器人引导到物体的一般区域，但它们缺乏精确控制灵巧机械手进行3D抓取所需的几何理解。为了克服这一点，我们的关键见解是利用模拟和力封闭抓取生成管道，该管道理解场景中手部和物体的局部几何形状。由于此管道速度慢且需要真实观测数据，因此将结果数据提炼为在相机点云上实时运行的扩散模型。通过结合互联网规模模型的全局语义理解和基于模拟的局部感知力封闭的几何精度，我们的方法实现了高性能的语义抓取，无需任何手动收集的训练数据。有关可视化，请访问我们的网站 https://ifgrasping.github.io/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何让机器人进行语义化的功能性抓取问题。虽然大型视觉语言模型能识别物体和部分，但缺乏精确控制灵巧机械手进行3D抓取的几何能力。这个问题很重要，因为机器人不仅需要识别物体，还需知道如何与物体交互；现有抓取方法往往不自然或不符合功能需求；在杂乱场景中，机器人需要能识别并抓取特定物体；手动收集抓取数据成本高昂且难以规模化；机器人理解任务语义并生成相应抓取姿态对日常生活复杂任务至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到VLMs在语义理解方面优势明显，但缺乏几何精确性，因此需要将两者结合。他们发现现有合成抓取方法缺乏语义指导，导致抓取不自然或不符合功能需求，在杂乱场景中表现不佳。他们借鉴了VLMs的语义理解能力(特别是SAM和VLPart模型)、基于力封闭的抓取优化方法(DexGraspNet)、扩散模型架构(DexDiffuser)以及点云处理技术(BPS)。作者设计了两阶段流程：先用VLM识别任务相关区域，再用基于力封闭的优化方法在这些区域生成抓取，最后通过模拟评估和扩散模型提炼实现实际应用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将互联网规模训练的大型视觉语言模型的语义理解能力与基于力封闭的抓取生成方法的几何精确性结合，生成既符合语义理解又具几何稳定性的功能性抓取，避免手动收集数据。整体流程分为四步：1)语义区域识别：多视角渲染物体图像，使用VLM识别任务相关区域并投影回3D空间；2)几何抓取合成：基于识别区域构建分割凸包，初始化机械手位置，使用能量函数优化抓取姿态；3)模拟评估：在模拟环境中测试抓取，通过扰动生成变体，计算成功率并过滤低质量抓取；4)扩散模型提炼：将抓取数据训练成条件扩散模型，实现从点云输入实时生成抓取姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将VLMs语义理解与抓取生成结合；2)设计两阶段流程(先识别区域再优化抓取)；3)只在语义相关区域初始化抓取而非整个物体；4)使用力量抓取而非精确抓取提高稳定性；5)完全基于合成数据无需手动收集；6)能在杂乱场景中识别抓取特定物体。相比Get a Grip，IFG有语义指导、只在相关区域初始化、使用力量抓取；相比DexGraspNet2，IFG能通过语义提示控制抓取特定物体，在困难物体上表现更好；相比纯VLMs，IFG能生成具体抓取姿态并具几何精确性；相比传统方法，IFG能生成符合任务需求的抓取，在杂乱场景中表现更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'IFG通过结合视觉语言模型的语义理解能力和基于力封闭的抓取生成方法，首次实现了无需手动数据收集的语义化功能性抓取，能够在杂乱场景中生成既自然又稳定的抓取姿态。'}


### 论文摘要

Large Vision Models trained on internet-scale data have demonstrated strong capabilities in segmenting and semantically understanding object parts, even in cluttered, crowded scenes. However, while these models can direct a robot toward the general region of an object, they lack the geometric understanding required to precisely control dexterous robotic hands for 3D grasping. To overcome this, our key insight is to leverage simulation with a force-closure grasping generation pipeline that understands local geometries of the hand and object in the scene. Because this pipeline is slow and requires ground-truth observations, the resulting data is distilled into a diffusion model that operates in real-time on camera point clouds. By combining the global semantic understanding of internet-scale models with the geometric precision of a simulation-based locally-aware force-closure, \our achieves high-performance semantic grasping without any manually collected training data. For visualizations of this please visit our website at https://ifgrasping.github.io/

---

## 229. Enhancing Rotation-Invariant 3D Learning with Global Pose Awareness and Attention Mechanisms

**论文链接:** [http://arxiv.org/abs/2511.08833v1](http://arxiv.org/abs/2511.08833v1)

**作者:** Jiaxun Guo, Manar Amayri, Nizar Bouguila, Xin Liu, Wentao Fan

**发布时间:** 2025-11-11

**备注:** 14 pages, 6 gigures,AAAI 2026

### GPT解析

### 总结

本文提出了一种新的旋转不变性学习方法，通过Shadow-informed Pose Feature (SiPF)和Rotation-invariant Attention Convolution (RIAttnConv)解决了现有方法中全局姿态信息丢失的问题，显著提升了在3D点云处理任务中的性能。

### 背景

旋转不变性(RI)学习在3D点云处理中的最新进展通常使用手工制作的旋转不变性特征替代原始坐标，以确保在任意旋转下的鲁棒性。然而，这些方法往往丢失全局姿态信息，无法区分几何相似但空间结构不同的物体。

### 目的

克服现有RI方法因感受野受限导致的Wing-tip特征崩溃问题，使模型能够在保持旋转不变性的同时保留全局姿态意识，从而区分几何相似但空间结构不同的组件。

### 方法

1) 引入Shadow-informed Pose Feature (SiPF)，通过从学习到的共享旋转派生的全局一致参考点（'阴影'）增强局部RI描述符；2) 提出旋转不变性注意力卷积(RIAttnConv)，将SiPF集成到特征聚合过程中；3) 设计基于单位四元球面Bingham分布的任务自适应阴影定位模块，动态学习构建一致阴影的最佳全局旋转。

### 主要发现

现有RI方法因感受野受限导致无法区分对称组件（如飞机左右机翼），这种现象称为Wing-tip特征崩溃。通过SiPF和RIAttnConv，模型能够在保持旋转不变性的同时保留全局姿态信息，有效区分几何相似但空间结构不同的组件。

### 结论

所提出的方法显著优于现有的RI方法，特别是在需要任意旋转下细粒度空间判别的任务中。该方法通过阴影机制解决了全局姿态信息丢失的问题，为旋转不变的3D点云处理提供了新思路。

### 翻译

近期在3D点云旋转不变性(RI)学习方面的最新进展通常使用手工制作的RI特征替代原始坐标，以确保在任意旋转下的鲁棒性。然而，这些方法常常遭受全局姿态信息的丢失，使其无法区分几何相似但空间结构不同的物体。我们确定这一局限性源于现有RI方法中受限的感受野，导致Wing-tip特征崩溃——由于局部几何不可区分而无法区分对称组件（如左右机翼）。为克服这一挑战，我们引入了Shadow-informed Pose Feature (SiPF)，它通过从学习到的共享旋转派生的全局一致参考点（称为'阴影'）来增强局部RI描述符。这种机制使模型能够在保持旋转不变性的同时保留全局姿态意识。我们进一步提出了旋转不变性注意力卷积(RIAttnConv)，这是一种基于注意力的算子，将SiPF集成到特征聚合过程中，从而增强模型区分结构相似组件的能力。此外，我们设计了一个基于单位四元球面Bingham分布的任务自适应阴影定位模块，动态学习用于构建一致阴影的最佳全局旋转。在3D分类和部分分割基准上的大量实验表明，我们的方法显著优于现有的RI方法，特别是在需要任意旋转下细粒度空间判别的任务中。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决3D点云学习中的旋转不变性问题，特别是现有方法无法区分几何相似但空间结构不同的组件（如飞机的左右机翼）。这个问题在现实中非常重要，因为自动驾驶、机器人和增强现实等应用中的3D物体通常以任意方向出现，而现有方法在处理这类场景时会出现'翼尖特征崩溃'现象，导致无法准确识别和区分空间上不同的对称结构，限制了3D模型在非受控环境中的实际应用。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有旋转不变性方法的局限性，特别是其受限感受野导致的对称结构区分问题，提出了创新的解决方案。他们借鉴了经典点对特征(PPF)的思想，但扩展了它以包含全局姿态信息；利用了注意力机制来动态聚合特征；并采用了Bingham分布来处理方向数据的概率建模。作者将这些思想整合，设计了阴影感知姿态特征(SiPF)和旋转不变注意力卷积(RIAttnConv)，并将这些方法集成到现有的点云处理架构(如DGCNN和AdaptConv)中进行验证。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是为每个点引入一个'阴影'点作为全局一致参考，结合局部几何特征与全局姿态信息，同时保持旋转不变性。整体流程包括：1)任务自适应阴影定位模块，使用Bingham分布估计全局参考旋转；2)SiPF和输入特征提取，构建局部参考帧并计算结合局部几何与全局姿态差异的特征；3)RIAttnConv操作，利用注意力机制动态聚合空间感知特征；4)将上述组件集成到分类或分割网络架构中，如DGCNN或AdaptConv。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)SiPF特征，通过引入阴影参考点解决'翼尖特征崩溃'问题；2)RIAttnConv算子，结合注意力机制和卷积操作实现动态特征聚合；3)基于Bingham分布的任务自适应阴影定位模块，动态学习最佳全局旋转。相比之前的工作，本文方法巧妙结合了局部几何与全局姿态信息，提供了严格的数学分析证明方法的有效性，特别针对对称结构的区分问题，并在提高性能的同时保持了与现有方法相当的效率。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过引入阴影感知姿态特征和旋转不变注意力卷积，有效解决了3D点云学习中旋转不变性与全局姿态感知之间的权衡问题，显著提升了模型在任意旋转条件下区分几何相似但空间不同结构的能力。'}


### 论文摘要

Recent advances in rotation-invariant (RI) learning for 3D point clouds typically replace raw coordinates with handcrafted RI features to ensure robustness under arbitrary rotations. However, these approaches often suffer from the loss of global pose information, making them incapable of distinguishing geometrically similar but spatially distinct structures. We identify that this limitation stems from the restricted receptive field in existing RI methods, leading to Wing-tip feature collapse, a failure to differentiate symmetric components (e.g., left and right airplane wings) due to indistinguishable local geometries. To overcome this challenge, we introduce the Shadow-informed Pose Feature (SiPF), which augments local RI descriptors with a globally consistent reference point (referred to as the 'shadow') derived from a learned shared rotation. This mechanism enables the model to preserve global pose awareness while maintaining rotation invariance. We further propose Rotation-invariant Attention Convolution (RIAttnConv), an attention-based operator that integrates SiPFs into the feature aggregation process, thereby enhancing the model's capacity to distinguish structurally similar components. Additionally, we design a task-adaptive shadow locating module based on the Bingham distribution over unit quaternions, which dynamically learns the optimal global rotation for constructing consistent shadows. Extensive experiments on 3D classification and part segmentation benchmarks demonstrate that our approach substantially outperforms existing RI methods, particularly in tasks requiring fine-grained spatial discrimination under arbitrary rotations.

---

## 230. Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning

**论文链接:** [http://arxiv.org/abs/2511.08240v1](http://arxiv.org/abs/2511.08240v1)

**作者:** Chenyu Hu, Xiaotong Li, Hao Zhu, Biao Hou

**发布时间:** 2025-11-11

**备注:** Accepted to AAAI 2026. Code is available at: https://github.com/wxszreal0/DiPVNet

### GPT解析

### 总结

本文提出了一种方向感知向量网络(DiPVNet)，用于解决点云旋转带来的表示学习挑战。该网络通过原子点积运算符同时编码方向选择性和旋转不变性，并引入可学习局部点积算子和全局方向响应谱，有效捕获了点云的多尺度方向特性。

### 背景

点云处理已成为许多3D视觉任务的核心技术，但任意旋转引入的点云方向变化一直是表示学习的长期挑战。旋转扰动破坏了点云的固有方向特征，而现有方法往往未能充分利用点云的多尺度方向特性来增强特征表示。

### 目的

开发一种能够有效处理点云旋转问题的方法，充分利用点云的多尺度方向特性来增强特征表示，提高点云分类和分割任务的性能。

### 方法

提出方向感知向量网络(DiPVNet)，包含：1)原子点积运算符，同时编码方向选择性和旋转不变性；2)可学习局部点积(L2DP)算子，使中心点与邻居交互自适应捕获非均匀局部结构；3)基于广义谐波分析的全局方向响应谱，通过方向感知的球面傅里叶变换建模整体方向结构。严格证明了两种算子的旋转不变性。

### 主要发现

DiPVNet在噪声和大角度旋转等挑战性场景下实现了点云分类和分割任务的最先进性能，有效解决了点云旋转带来的表示学习挑战。

### 结论

DiPVNet通过结合方向感知和旋转不变性，显著提升了点云处理性能，为点云表示学习提供了新思路。

### 翻译

点云处理已成为许多3D视觉任务的核心技术。然而，任意旋转引入的点云方向变化，给有效的表示学习带来了长期挑战。这个问题的核心是旋转扰动破坏了点云的固有方向特征。最近的方法试图隐式建模旋转等变性和不变性，保留方向信息并将其传播到深层语义空间。然而，它们往往未能充分利用点云的多尺度方向特性来增强特征表示。为此，我们提出了方向感知向量网络(DiPVNet)。其核心是一个原子点积运算符，同时编码方向选择性和旋转不变性，使网络具有旋转对称建模和自适应方向感知能力。在局部层面，我们引入了可学习局部点积(L2DP)算子，使中心点与其邻居之间能够交互，自适应地捕获点云的非均匀局部结构。在全局层面，我们利用广义谐波分析证明点云与球形采样向量之间的点积等价于方向感知的球面傅里叶变换(DASFT)。这导致了用于建模整体方向结构的全局方向响应谱的构建。我们严格证明了两种算子的旋转不变性。在涉及噪声和大角度旋转的挑战性场景上的大量实验表明，DiPVNet在点云分类和分割任务上实现了最先进的性能。我们的代码可在https://github.com/wxszreal0/DiPVNet获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云处理中的旋转不变性问题。当点云在3D空间中发生任意旋转时，会导致点云方向的变化，破坏点云的内在方向特征，使得传统方法难以有效学习。这个问题在现实中非常重要，因为3D感知技术广泛应用于自动驾驶、机器人、增强现实等领域，这些应用中物体可能以任意方向出现，系统需要能够识别它们而不受方向影响。旋转不变性对于3D视觉任务的鲁棒性至关重要，直接影响实际应用的可靠性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：显式方向编码方法依赖于固定的方向分区或手工设计的统计方案，缺乏自适应学习能力；隐式旋转对称建模方法虽然保留了方向信息，但往往无法在特征表示层面有效利用这些信息。基于这些分析，作者提出了关键见解：点积操作同时具有方向选择性和旋转不变性的双重特性。作者借鉴了向量神经元网络(VNN)和图神经网络的思想，但改进了VNN仅使用单一全局方向向量的局限性，并利用球面傅里叶变换的理论基础提出了更高效的方向感知球面傅里叶变换(DASFT)。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用点积操作的双重特性（方向选择性和旋转不变性）设计原子点积算子，构建一个能够自适应感知多尺度方向特征同时保持旋转对称性的点云表示学习框架。整体实现流程：1)构建K近邻图将点云转换为图特征；2)在每一层中，使用VNN块建模旋转等变性，通过L2DP算子提取局部方向特征，通过DASFT模块构建全局方向响应谱；3)使用交叉注意力机制融合局部和全局特征；4)将等变特征投影到规范基上产生旋转鲁棒特征；5)将融合后的特征用于下游任务。L2DP算子通过点积操作和特征聚合学习局部方向特征，DASFT模块通过点云与球面采样向量的点积构建全局方向响应谱。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)原子点积算子，利用点积的双重特性设计方向感知和旋转不变的算子；2)可学习局部点积(L2DP)算子，通过可微分点积操作自适应学习局部方向特征，提供DLP和SAP两种聚合策略；3)方向感知球面傅里叶变换(DASFT)，构建全局方向响应谱实现旋转不变描述。相比之前工作，不同之处在于：不依赖固定方向分区，能自适应学习非均匀局部点分布；不仅保留方向信息，还在特征表示层面有效利用多尺度方向特性；通过原子算子统一建模局部和全局特征，避免复杂特征解耦；DASFT计算效率高于传统球面谐波方法，无需预计算且并行化能力更好。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了基于原子点积算子的方向感知向量网络(DiPVNet)，通过同时建模局部和全局多尺度方向特征，实现了对任意旋转具有鲁棒性的点云表示学习，在点云分类和分割任务上达到了最先进性能。'}


### 论文摘要

Point cloud processing has become a cornerstone technology in many 3D vision tasks. However, arbitrary rotations introduce variations in point cloud orientations, posing a long-standing challenge for effective representation learning. The core of this issue is the disruption of the point cloud's intrinsic directional characteristics caused by rotational perturbations. Recent methods attempt to implicitly model rotational equivariance and invariance, preserving directional information and propagating it into deep semantic spaces. Yet, they often fall short of fully exploiting the multiscale directional nature of point clouds to enhance feature representations. To address this, we propose the Direction-Perceptive Vector Network (DiPVNet). At its core is an atomic dot-product operator that simultaneously encodes directional selectivity and rotation invariance--endowing the network with both rotational symmetry modeling and adaptive directional perception. At the local level, we introduce a Learnable Local Dot-Product (L2DP) Operator, which enables interactions between a center point and its neighbors to adaptively capture the non-uniform local structures of point clouds. At the global level, we leverage generalized harmonic analysis to prove that the dot-product between point clouds and spherical sampling vectors is equivalent to a direction-aware spherical Fourier transform (DASFT). This leads to the construction of a global directional response spectrum for modeling holistic directional structures. We rigorously prove the rotation invariance of both operators. Extensive experiments on challenging scenarios involving noise and large-angle rotations demonstrate that DiPVNet achieves state-of-the-art performance on point cloud classification and segmentation tasks. Our code is available at https://github.com/wxszreal0/DiPVNet.

---

## 231. BuildingWorld: A Structured 3D Building Dataset for Urban Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.06337v1](http://arxiv.org/abs/2511.06337v1)

**作者:** Shangfeng Huang, Ruisheng Wang, Xin Wang

**发布时间:** 2025-11-09

### GPT解析

### 总结

这篇论文介绍了BuildingWorld，一个全面的3D建筑数据集，旨在解决现有学习模型在建筑风格多样性方面的局限性，为城市数字孪生提供更准确的3D建筑模型。

### 背景

随着数字孪生成为现代城市转型的核心，准确和结构化的3D建筑模型成为高保真、可更新城市表示的关键使能技术。然而，大多数基于学习的3D城市模型是在建筑风格多样性有限的数据集上训练的，这严重限制了它们在不同城市环境中的泛化能力。

### 目的

提出BuildingWorld数据集，以弥合建筑风格多样性的差距，为城市规模的基础模型和分析提供全球代表性的数据集。

### 方法

创建了BuildingWorld数据集，包含约500万个LOD2建筑模型，来自地理和建筑风格多样化的地区（包括北美、欧洲、亚洲、非洲和大洋洲），并配有真实和模拟的机载激光雷达点云。同时引入了Cyber City虚拟城市模型，以生成具有定制化和结构化多样化点云分布的无限制训练数据。此外，提供了针对建筑重建的标准评估指标。

### 主要发现

BuildingWorld数据集提供了全面的3D建筑重建、检测和分割研究的基础，能够支持大规模视觉模型和基础模型在结构化3D城市环境中的训练、评估和比较。

### 结论

BuildingWorld数据集通过提供全球代表性的多样化建筑数据，解决了现有模型在泛化能力方面的局限性，为3D城市建模和分析提供了新的可能性。

### 翻译

随着数字孪生成为现代城市转型的核心，准确和结构化的3D建筑模型成为高保真、可更新城市表示的关键使能技术。这些模型支持多种应用，包括能源建模、城市规划、自主导航和实时推理。尽管3D城市建模最近有所进展，但大多数基于学习的模型是在建筑风格多样性有限的数据集上训练的，这严重限制了它们在不同城市环境中的泛化能力。为了解决这一局限性，我们提出了BuildingWorld，一个全面的、结构化的3D建筑数据集，旨在弥合风格多样性的差距。它包含了来自地理和建筑风格多样化地区的建筑——包括北美、欧洲、亚洲、非洲和大洋洲——为城市规模的基础建模和分析提供了一个具有全球代表性的数据集。具体来说，BuildingWorld提供了约500万个来自不同来源的LOD2建筑模型，并配有真实和模拟的机载激光雷达点云。这 enables全面的3D建筑重建、检测和分割研究。Cyber City是一个虚拟城市模型，被引入以生成具有定制化和结构化多样化点云分布的无限制训练数据。此外，我们提供了针对建筑重建的标准评估指标，旨在促进大规模视觉模型和基础模型在结构化3D城市环境中的训练、评估和比较。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有3D建筑数据集在建筑风格和地理多样性方面的不足，导致模型难以泛化到不同城市环境。这个问题很重要，因为随着数字孪生城市的发展，准确和结构化的3D建筑模型是支持能源建模、城市规划、自主导航等关键应用的基础，而数据多样性不足会限制这些模型在真实世界中的实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了大型语言模型和视觉模型的成功经验，认识到大规模高质量数据集对模型泛化能力的重要性。他们分析了现有建筑数据集的局限性，如地理范围有限、建筑风格单一，并参考了点云生成技术和生成式模型的设计思路。通过整合全球建筑模型、使用LiDAR模拟器和创建虚拟城市生成器，作者设计了一个综合解决方案，既克服了真实数据收集的困难，又提高了数据的多样性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过构建大规模、多样化的3D建筑数据集，提升模型在不同地理区域和建筑风格下的泛化能力。整体流程包括：1)收集全球44个城市约500万个LoD2建筑模型；2)使用Helios++模拟器生成逼真的航空点云；3)创建Cyber City虚拟城市生成器，可生成无限多样的城市配置；4)提供标准化评估指标和基准测试；5)结合真实和模拟数据验证模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个覆盖五大洲的大规模3D建筑数据集(约500万模型)；2)Cyber City虚拟城市生成器，可创建无限多样的城市场景；3)结合真实和模拟的航空LiDAR点云，考虑遮挡、激光入射角等真实因素；4)全面的评估指标体系。相比之前工作，BuildingWorld规模更大、地理覆盖更广、数据更多样，不仅支持建筑重建，还支持语义分割等多种任务，并通过程序化生成突破了数据收集的物理限制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BuildingWorld通过构建首个包含全球500万多样化建筑模型和配套点云数据的大规模结构化数据集，结合虚拟城市生成技术和先进LiDAR模拟，显著提升了3D城市基础模型在不同地理区域和建筑风格下的泛化能力和重建精度。'}


### 论文摘要

As digital twins become central to the transformation of modern cities, accurate and structured 3D building models emerge as a key enabler of high-fidelity, updatable urban representations. These models underpin diverse applications including energy modeling, urban planning, autonomous navigation, and real-time reasoning. Despite recent advances in 3D urban modeling, most learning-based models are trained on building datasets with limited architectural diversity, which significantly undermines their generalizability across heterogeneous urban environments. To address this limitation, we present BuildingWorld, a comprehensive and structured 3D building dataset designed to bridge the gap in stylistic diversity. It encompasses buildings from geographically and architecturally diverse regions -- including North America, Europe, Asia, Africa, and Oceania -- offering a globally representative dataset for urban-scale foundation modeling and analysis. Specifically, BuildingWorld provides about five million LOD2 building models collected from diverse sources, accompanied by real and simulated airborne LiDAR point clouds. This enables comprehensive research on 3D building reconstruction, detection and segmentation. Cyber City, a virtual city model, is introduced to enable the generation of unlimited training data with customized and structurally diverse point cloud distributions. Furthermore, we provide standardized evaluation metrics tailored for building reconstruction, aiming to facilitate the training, evaluation, and comparison of large-scale vision models and foundation models in structured 3D urban environments.

---

## 232. Memory- and Latency-Constrained Inference of Large Language Models via Adaptive Split Computing

**论文链接:** [http://arxiv.org/abs/2511.04002v1](http://arxiv.org/abs/2511.04002v1)

**作者:** Mingyu Sung, Vikas Palakonda, Suhwan Im, Sunghwan Moon, Il-Min Kim, Sangseok Yun, Jae-Mo Kang

**发布时间:** 2025-11-06

### GPT解析

### 总结

本研究提出了一种专门为在边缘设备上部署大型语言模型设计的自回归感知分拆计算框架，通过创新的压缩和优化技术解决了资源受限环境下的部署挑战。

### 背景

大型语言模型在各种推理任务上已接近人类水平，但由于其庞大的参数量和内存密集型的自回归解码，在资源受限的物联网设备上部署仍然不切实际。虽然分拆计算提供了一种解决方案，但现有方法未能解决自回归推理的独特挑战。

### 目的

开发第一个专门为在边缘设备上部署LLM设计的自回归感知分拆计算框架，以解决内存不足和通信开销问题。

### 方法

提出三种关键贡献：1)一点分割压缩(OPSC)，一种混合精度量化方案，将模型划分为不同精度级别的前端和后端段；2)两阶段中间压缩流水线，结合阈值分割和逐令牌自适应比特量化；3)统一的优化框架，联合选择最佳分割点、量化设置和序列长度。

### 主要发现

在各种LLM和硬件平台上的评估表明，该框架比最先进的量化方法(包括SmoothQuant、OmniQuant和Atom)性能更优，实现了1.49倍的推理加速，显著减少通信开销，同时保持或提高了模型准确性。

### 结论

该框架成功解决了在资源受限的物联网设备上部署大型语言模型的挑战，特别是在处理自回归推理时的内存和通信效率问题，为边缘计算环境中的LLM部署提供了实用解决方案。

### 翻译

大型语言模型已在多种推理任务上实现接近人类的性能，但由于其庞大的参数量和内存密集型的自回归解码，在资源受限的物联网设备上的部署仍然不切实际。虽然分拆计算通过在边缘设备和云服务器之间划分模型执行提供了一种有前途的解决方案，但现有方法未能解决自回归推理的独特挑战，特别是迭代令牌生成过程和不断增长的关键值缓存需求。这项工作引入了第一个专门为在边缘设备上部署LLM设计的自回归感知分拆计算框架。我们的方法有三个关键贡献：首先，我们开发了一点分割压缩(OPSC)，一种混合精度量化方案，通过将模型战略性地划分为具有不同精度级别的前端和后端段来防止内存不足故障；其次，我们提出了一种两阶段中间压缩流水线，结合阈值分割(TS)和逐令牌自适应比特量化(TAB-Q)，在显著减少通信开销的同时保留精度关键的激活；第三，我们制定了一个统一的优化框架，联合选择最佳分割点、量化设置和序列长度，以满足严格的内存和延迟约束。在各种LLM和硬件平台上进行的广泛评估表明，与最先进的量化方法相比，性能优越。该框架实现了1.49倍的推理加速，同时显著减少了通信开销，并保持或提高了模型准确性。


### 论文摘要

Large language models (LLMs) have achieved near-human performance across diverse reasoning tasks, yet their deployment on resource-constrained Internet-of-Things (IoT) devices remains impractical due to massive parameter footprints and memory-intensive autoregressive decoding. While split computing offers a promising solution by partitioning model execution between edge devices and cloud servers, existing approaches fail to address the unique challenges of autoregressive inference, particularly the iterative token generation process and expanding key-value (KV) cache requirements. This work introduces the first autoregressive-aware split computing framework designed explicitly for LLM deployment on edge devices. Our approach makes three key contributions. First, we develop one-point split compression (OPSC), a mixed-precision quantization scheme that prevents out-of-memory failures by strategically partitioning models into front-end and back-end segments with different precision levels. Second, we propose a two-stage intermediate compression pipeline that combines threshold splitting (TS) and token-wise adaptive bit quantization (TAB-Q) to preserve accuracy-critical activations while dramatically reducing communication overhead. Third, we formulate a unified optimization framework that jointly selects optimal split points, quantization settings, and sequence lengths to satisfy strict memory and latency constraints. Extensive evaluations across diverse LLMs and hardware platforms demonstrate superior performance compared to state-of-the-art quantization methods, including SmoothQuant, OmniQuant, and Atom. The framework achieves a 1.49 inference speedup and significant communication overhead reduction while maintaining or improving model accuracy.

---

## 233. Hierarchical Strategic Decision-Making in Layered Mobility Systems

**论文链接:** [http://arxiv.org/abs/2511.08734v1](http://arxiv.org/abs/2511.08734v1)

**作者:** Mingjia He, Zhiyu He, Jan Ghadamian, Florian Dörfler, Emilio Frazzoli, Gioele Zardini

**发布时间:** 2025-11-11

### GPT解析

### 总结

该研究提出了一种将层次博弈建模与在线反馈优化相结合的方法，将城市交通系统视为三层Stackelberg博弈（旅行者、运营商、市政府），并通过反馈闭环实现有效控制。

### 背景

城市交通系统是复杂的社会技术环境，受多方利益相关者影响，具有层次化的相互依赖决策，这使得有效控制和政策设计具有内在挑战性。

### 目的

通过层次博弈建模与在线反馈优化的结合，将城市交通视为一个三层Stackelberg博弈（旅行者、运营商、市政府），并将其置于反馈闭环中，以实现更有效的交通政策设计。

### 方法

市政府使用投影两点（无梯度）方案迭代更新税收、补贴和运营约束，而较低层级通过均衡计算（旅行者均衡使用Frank-Wolfe算法；运营商的最佳响应）进行响应。这种无模型的管道强制执行约束，适应异构用户和模式，并扩展到更高维度的政策向量。

### 主要发现

在瑞士苏黎世的真实多模式网络中，该方法比贝叶斯优化和遗传算法更好地实现市政目标，并确定了能增加多模式使用同时改善运营商目标的整合激励。

### 结论

基于反馈的监管可以将竞争引向合作结果，并在复杂、数据丰富的移动生态系统中带来切实的福利收益。

### 翻译

交通系统是复杂的社会技术环境，受多方利益相关者影响，具有层次化的相互依赖决策，这使得有效控制和政策设计本质上具有挑战性。我们将层次博弈建模与在线反馈优化相结合，将城市交通视为一个三层Stackelberg博弈（旅行者、运营商、市政府），并将其置于反馈闭环中。市政府使用投影两点（无梯度）方案迭代更新税收、补贴和运营约束，而较低层级通过均衡计算（旅行者均衡使用Frank-Wolfe算法；运营商的最佳响应）进行响应。这种无模型的管道强制执行约束，适应异构用户和模式，并扩展到更高维度的政策向量而无需通过均衡映射进行微分。在瑞士苏黎世的真实多模式网络上，我们的方法比贝叶斯优化和遗传算法显著更好地实现市政目标，并确定了能增加多模式使用同时改善运营商目标的整合激励。结果表明，基于反馈的监管可以将竞争引向合作结果，并在复杂、数据丰富的移动生态系统中带来切实的福利收益。


### 论文摘要

Mobility systems are complex socio-technical environments influenced by multiple stakeholders with hierarchically interdependent decisions, rendering effective control and policy design inherently challenging. We bridge hierarchical game-theoretic modeling with online feedback optimization by casting urban mobility as a tri-level Stackelberg game (travelers, operators, municipality) closed in a feedback loop. The municipality iteratively updates taxes, subsidies, and operational constraints using a projected two-point (gradient-free) scheme, while lower levels respond through equilibrium computations (Frank-Wolfe for traveler equilibrium; operator best responses). This model-free pipeline enforces constraints, accommodates heterogeneous users and modes, and scales to higher-dimensional policy vectors without differentiating through equilibrium maps.   On a real multimodal network for Zurich, Switzerland, our method attains substantially better municipal objectives than Bayesian optimization and Genetic algorithms, and identifies integration incentives that increase multimodal usage while improving both operator objectives. The results show that feedback-based regulation can steer competition toward cooperative outcomes and deliver tangible welfare gains in complex, data-rich mobility ecosystems.

---

## 234. Enhancing Heavy Rain Nowcasting with Multimodal Data: Integrating Radar and Satellite Observations

**论文链接:** [http://arxiv.org/abs/2511.00716v1](http://arxiv.org/abs/2511.00716v1)

**作者:** Rama Kassoumeh, David Rügamer, Henning Oppel

**发布时间:** 2025-11-01

**备注:** accepted to ICMLA 2025

### GPT解析

### 总结

研究开发了一种融合卫星和雷达数据的多模态临近预报模型，用于预测城市强降雨事件，显著提高了预测准确性，特别是在5分钟提前量下强雨预测的临界成功指数提高4%，暴雨提高3%，为城市洪水预警提供了更可靠的工具。

### 背景

强降雨事件频率增加导致城市洪水问题日益严重，但传统监测系统存在局限：德国2001-2018年间仅有17.3%的小时强降雨事件被雨量计记录；雷达数据虽能追踪正在发生的降水，但仅用雷达预报强降雨发展仍具挑战，因为此类事件短暂且不可预测。

### 目的

评估融合卫星和雷达数据用于临近预报的有效性，以提高城市地区强降雨预测的准确性。

### 方法

开发了一种多模态临近预报模型，结合雷达和卫星图像，用于预测5分钟、15分钟和30分钟提前量的降水情况。

### 主要发现

多模态策略明显优于仅使用雷达的方法；集成卫星数据提高了预测准确性，特别是对强降水；在5分钟提前量下，模型使强雨的临界成功指数提高4%，暴雨提高3%；在更长提前量下，模型保持更高预测技能，而仅雷达的性能下降；2021年德国北莱茵-威斯特法伦州严重洪水事件的案例分析表明，多模态模型能提供更详细准确的强降雨区域预报。

### 结论

融合卫星和雷达的多模态模型提高了强降雨预测的精度，能够提供及时、可靠、挽救生命的预警，有助于减轻城市洪水带来的危害。

### 翻译

强降雨事件频率增加，是城市洪水的主要原因，这凸显了精确降水预报的迫切需求——尤其是在城市地区，局部事件经常被地面传感器漏检。在德国，2001年至2018年间只有17.3%的小时强降雨事件被雨量计记录，突显了传统监测系统的局限性。雷达数据是另一种有效追踪正在发生的降水的来源；然而，仅使用雷达预报强降雨的发展仍然具有挑战性，因为此类事件具有短暂和不可预测的性质。我们的重点是评估融合卫星和雷达数据用于临近预报的有效性。我们开发了一种多模态临近预报模型，结合雷达和卫星图像，用于预测5分钟、15分钟和30分钟提前量的降水。我们证明这种多模态策略明显优于仅使用雷达的方法。实验结果表明，集成卫星数据提高了预测准确性，特别是对强降水。在5分钟提前量下，所提出的模型使强雨的临界成功指数提高4%，暴雨提高3%。此外，在雷达性能下降的更长提前量下，它保持了更高的预测技能。对2021年德国北莱茵-威斯特法伦州严重洪水事件的定性分析进一步说明了多模态模型的优越性能。与捕捉一般降水模式的仅雷达模型不同，多模态模型对受强降雨影响的区域产生更详细和准确的预报。这种改进的精度能够及时、可靠、挽救生命的预警。实施地址：https://github.com/RamaKassoumeh/Multimodal_heavy_rain


### 论文摘要

The increasing frequency of heavy rainfall events, which are a major cause of urban flooding, underscores the urgent need for accurate precipitation forecasting - particularly in urban areas where localized events often go undetected by ground-based sensors. In Germany, only 17.3% of hourly heavy rain events between 2001 and 2018 were recorded by rain gauges, highlighting the limitations of traditional monitoring systems. Radar data are another source that effectively tracks ongoing precipitation; however, forecasting the development of heavy rain using radar alone remains challenging due to the brief and unpredictable nature of such events. Our focus is on evaluating the effectiveness of fusing satellite and radar data for nowcasting. We develop a multimodal nowcasting model that combines both radar and satellite imagery for predicting precipitation at lead times of 5, 15, and 30 minutes. We demonstrate that this multimodal strategy significantly outperforms radar-only approaches. Experimental results show that integrating satellite data improves prediction accuracy, particularly for intense precipitation. The proposed model increases the Critical Success Index for heavy rain by 4% and for violent rain by 3% at a 5-minute lead time. Moreover, it maintains higher predictive skill at longer lead times, where radar-only performance declines. A qualitative analysis of the severe flooding event in the state of North Rhine-Westphalia, Germany in 2021 further illustrates the superior performance of the multimodal model. Unlike the radar-only model, which captures general precipitation patterns, the multimodal model yields more detailed and accurate forecasts for regions affected by heavy rain. This improved precision enables timely, reliable, life-saving warnings. Implementation available at https://github.com/RamaKassoumeh/Multimodal_heavy_rain

---

## 235. RoadSens-4M: A Multimodal Smartphone & Camera Dataset for Holistic Road-way Analysis

**论文链接:** [http://arxiv.org/abs/2510.25211v1](http://arxiv.org/abs/2510.25211v1)

**作者:** Amith Khandakar, David Michelson, Shaikh Golam Rabbani, Fariya Bintay Shafi, Md. Faysal Ahamed, Khondokar Radwanur Rahman, Md Abidur Rahman, Md. Fahmidun Nabi, Mohamed Arselene Ayari, Khaled Khan, Ponnuthurai Nagaratnam Suganthan

**发布时间:** 2025-10-29

### GPT解析

### 总结

这篇论文介绍了一个新的道路质量数据集，通过移动应用程序收集多种传感器数据，并结合GIS、天气和视频信息，为道路状况评估提供了全面解决方案。

### 背景

监测道路问题如颠簸和坑洼对提高安全性和改善道路状况至关重要。智能手机内置传感器提供了成本效益高且简便的道路质量评估方法，但由于缺乏高质量、标准化的数据集，该领域进展缓慢。

### 目的

创建一个综合数据集，为交通管理、基础设施发展、道路安全和城市规划提供资金支持，并促进智能交通系统的进一步研究和创新。

### 方法

开发移动应用程序收集来自GPS、加速度计、陀螺仪、磁力计、重力传感器和方向传感器的数据，并整合GIS数据、天气信息和道路状况视频。

### 主要发现

该数据集是少数几个将GIS数据与天气信息和道路状况视频相结合的数据集之一，提供了具有地理背景的道路问题的全面理解，允许对道路状况进行更清晰的分析。

### 结论

该数据集将支持交通管理、基础设施发展、道路安全和城市规划，并公开以促进智能交通系统的进一步研究和创新。

### 翻译

监测道路问题如颠簸和坑洼对提高安全性和改善道路状况非常重要。智能手机配备了各种内置传感器，提供了一种成本效益高且简便的方法来评估道路质量。然而，由于缺乏高质量、标准化的数据集，该领域的进展一直缓慢。本文讨论了一个由移动应用程序创建的新数据集，该应用程序收集来自GPS、加速度计、陀螺仪、磁力计、重力传感器和方向传感器的传感器数据。该数据集是少数几个将地理信息系统(GIS)数据与天气信息和道路状况视频相结合的数据集之一，提供了具有地理背景的道路问题的全面理解。通过汇编车辆速度、加速度、旋转速率和磁场强度等基本数据，以及GIS、天气和视频数据提供的视觉和空间背景，该数据集允许对道路状况进行更清晰的分析。其目标是为提高交通管理、基础设施发展、道路安全和城市规划的举措提供资金支持。此外，该数据集将公开以促进智能交通系统的进一步研究和创新。


### 论文摘要

It's important to monitor road issues such as bumps and potholes to enhance safety and improve road conditions. Smartphones are equipped with various built-in sensors that offer a cost-effective and straightforward way to assess road quality. However, progress in this area has been slow due to the lack of high-quality, standardized datasets. This paper discusses a new dataset created by a mobile app that collects sensor data from devices like GPS, accelerometers, gyroscopes, magnetometers, gravity sensors, and orientation sensors. This dataset is one of the few that integrates Geographic Information System (GIS) data with weather information and video footage of road conditions, providing a comprehensive understanding of road issues with geographic context. The dataset allows for a clearer analysis of road conditions by compiling essential data, including vehicle speed, acceleration, rotation rates, and magnetic field intensity, along with the visual and spatial context provided by GIS, weather, and video data. Its goal is to provide funding for initiatives that enhance traffic management, infrastructure development, road safety, and urban planning. Additionally, the dataset will be publicly accessible to promote further research and innovation in smart transportation systems.

---

## 236. Modeling realistic human behavior using generative agents in a multimodal transport system: Software architecture and Application to Toulouse

**论文链接:** [http://arxiv.org/abs/2510.19497v1](http://arxiv.org/abs/2510.19497v1)

**作者:** Trung-Dung Vu, Benoit Gaudou, Kamaldeep Singh Oberoi

**发布时间:** 2025-10-22

### GPT解析

### 总结

本文介绍了一种在复杂多模式交通系统中模拟真实人类移动行为的架构，通过法国图卢兹的案例研究进行了演示。该研究将大型语言模型应用于基于代理的模拟中，以捕捉真实城市环境中的决策制定过程。

### 背景

模拟真实的人类行为以理解人们的出行方式选择，从而提出个性化的出行解决方案，仍然具有挑战性。

### 目的

构建一种架构用于模拟复杂多模式交通系统中的真实人类移动行为，并评估大型语言模型与基于代理的模拟结合在智能交通系统和个性化出行解决方案方面的潜力。

### 方法

将大型语言模型应用于基于代理的模拟；集成GAMA模拟平台与基于LLM的生成式代理；使用通用交通馈送规范数据获取公共交通信息；使用OpenTripPlanner进行多模式路径规划；GAMA平台模拟交互式交通环境并提供可视化功能。

### 主要发现

在为期一个月的模拟中，代理能够做出具有情境感知能力的交通决策，并随时间形成习惯。

### 结论

结合大型语言模型与基于代理的模拟为推进智能交通系统和个性化多模式出行解决方案提供了有前景的方向。同时，研究也讨论了该方法的局限性，包括需要扩展到更大区域、整合实时数据和改进记忆模型。

### 翻译

模拟真实的人类行为以理解人们的出行方式选择，从而提出个性化的出行解决方案仍然具有挑战性。本文介绍了一种用于在复杂多模式交通系统中模拟真实人类移动行为的架构，通过法国图卢兹的案例研究进行了演示。我们在基于代理的模拟中应用大型语言模型，以捕捉真实城市环境中的决策制定。该框架集成了GAMA模拟平台与基于LLM的生成式代理，以及用于公共交通的通用交通馈送规范数据和用于多模式路径规划的OpenTripPlanner。GAMA平台模拟交互式交通环境，提供可视化和动态代理交互，同时无需从头开始构建模拟环境。这种设计使能够更专注于开发生成式代理，并评估其在交通决策制定过程中的性能。在为期一个月的模拟中，结果表明代理不仅能做出具有情境感知能力的交通决策，还能随时间形成习惯。我们得出结论，将LLMs与基于代理的模拟相结合，为推进智能交通系统和个性化多模式出行解决方案提供了有前景的方向。我们还讨论了该方法的一些局限性，并概述了未来工作，包括扩展到更大区域、整合实时数据和改进记忆模型。


### 论文摘要

Modeling realistic human behaviour to understand people's mode choices in order to propose personalised mobility solutions remains challenging. This paper presents an architecture for modeling realistic human mobility behavior in complex multimodal transport systems, demonstrated through a case study in Toulouse, France. We apply Large Language Models (LLMs) within an agent-based simulation to capture decision-making in a real urban setting. The framework integrates the GAMA simulation platform with an LLM-based generative agent, along with General Transit Feed Specification (GTFS) data for public transport, and OpenTripPlanner for multimodal routing. GAMA platform models the interactive transport environment, providing visualization and dynamic agent interactions while eliminating the need to construct the simulation environment from scratch. This design enables a stronger focus on developing generative agents and evaluating their performance in transport decision-making processes. Over a simulated month, results show that agents not only make context-aware transport decisions but also form habits over time. We conclude that combining LLMs with agent-based simulation offers a promising direction for advancing intelligent transportation systems and personalised multimodal mobility solutions. We also discuss some limitations of this approach and outline future work on scaling to larger regions, integrating real-time data, and refining memory models.

---

## 237. FlexBSS: A flexible multi-objective framework for bike-sharing station optimization

**论文链接:** [http://arxiv.org/abs/2510.16615v1](http://arxiv.org/abs/2510.16615v1)

**作者:** Jordi Grau-Escolano, David Duran-Rodas, Julian Vicens

**发布时间:** 2025-10-18

**备注:** 56 pages, 24 figures, 11 tables

### GPT解析

### 总结

该研究提出了一种灵活的、数据驱动的共享单车系统站点优化框架，通过多标准决策模型、网络级指标、坡度调整距离和遗传算法四个组件，解决了现有方法在适应不同规划目标和考虑城市复杂环境方面的局限性。

### 背景

共享单车系统是城市交通的关键组成部分，促进积极出行方式并补充公共交通。现有方法通常只关注单一规划目标或固定的两三个目标组合，限制了它们对不断变化的优先事项和不同环境的适应性。此外，这些方法通常将候选站点聚合为粗糙的空间单元，忽略了地形或有向骑行网络等特征。

### 目的

开发一个灵活的框架，用于优化共享单车系统站点布局，使其能够适应多样化的规划目标和复杂的城市环境，并考虑更多空间因素如地形和有向骑行网络。

### 方法

该框架包含四个组件：1)多标准决策模型，使用可配置的空间因素和可调整的权重评估候选站点；2)网络级指标，在有向自行车网络上平衡站点邻近性和系统范围的可达性；3)坡度调整距离，考虑上坡努力和下坡便利性；4)遗传算法，在满足最小距离约束的同时生成可行的站点集合。

### 主要发现

巴塞罗那的案例研究表明，网络级指标在保持相同规划目标的同时重塑了站点的空间布局；坡度调整距离提高了丘陵地区的站点可用性；在Eixample区的扩展场景中产生了与现有站点互补的连贯布局。

### 结论

通过整合空间因素和网络表示，该框架为复杂城市环境中的共享单车系统规划提供了多功能的决策支持工具。

### 翻译

共享单车系统(BSS)是城市交通的关键组成部分，促进积极出行并补充公共交通。本文提出了一种灵活的、数据驱动的BSS站点优化框架。现有方法通常只关注单一规划目标，如最大化需求，或固定的两三个目标组合，如结合需求和社会公平。这限制了它们对不断变化的优先事项和不同环境的适应性。此外，它们通常将候选站点聚合为粗糙的空间单元，忽略了地形或有向骑行网络等特征。该框架通过四个组件解决了这些局限性：(i)多标准决策模型，使用可配置的空间因素和可调整的权重评估候选站点，使规划者能够表示任何规划目标组合，而不仅限于固定集合；(ii)网络级指标，在有向自行车网络上明确平衡站点邻近性和系统范围的可达性；(iii)坡度调整距离，考虑上坡努力和下坡便利性；(iv)遗传算法，在满足最小距离约束的同时生成可行的站点集合。巴塞罗那的案例研究展示了其在专注于需求、多模式整合和社会公平的场景中的应用，使用开放数据。结果表明，网络级指标在保持相同规划目标的同时重塑了站点的空间布局，坡度调整距离提高了丘陵地区的站点可用性，而Eixample区的扩展场景产生了与现有站点互补的连贯布局。通过整合空间因素和网络表示，该框架为复杂城市环境中的BSS规划提供了多功能的决策支持工具。


### 论文摘要

Bike-sharing systems (BSS) are key components of urban mobility, promoting active travel and complementing public transport. This paper presents a flexible, data-driven framework for optimizing BSS station placement. Existing methods usually focus on a single planning objective, such as maximizing demand, or on a fixed set of two or three objectives, such as combining demand and social equity. This restricts their adaptability to evolving priorities and diverse contexts. Moreover, they often aggregate candidate sites into coarse spatial units and neglect features such as topography or directed cycling networks. The framework addresses these limitations through four components: (i) a multi-criteria decision-making model that evaluates candidate sites using configurable spatial factors and adjustable weights, thereby enabling planners to represent any combination of planning objectives rather than being limited to a fixed set; (ii) network-level metrics that explicitly balance station proximity and system-wide accessibility on directed bicycle networks; (iii) slope-adjusted distances that account for uphill effort and downhill facilitation; and (iv) a genetic algorithm that generates feasible station sets while respecting minimum-distance constraints. A case study in Barcelona demonstrates its application using open data for scenarios focused on demand, multimodal integration, and social equity. Results show that network-level metrics reshape the spatial arrangement of stations while keeping the same planning objectives, slope-adjusted distances improve station availability in hilly districts, and an expansion scenario in lEixample district produces coherent layouts complementing existing stations. By integrating spatial factors and network representation, the framework provides a versatile decision-support tool for BSS planning in complex urban environments.

---

## 238. Scaling Traffic Insights with AI and Language Model-Powered Camera Systems for Data-Driven Transportation Decision Making

**论文链接:** [http://arxiv.org/abs/2510.09981v1](http://arxiv.org/abs/2510.09981v1)

**作者:** Fan Zuo, Donglin Zhou, Jingqin Gao, Kaan Ozbay

**发布时间:** 2025-10-11

### GPT解析

### 总结

该研究提出了一种基于AI的端到端框架，利用现有交通摄像头基础设施进行大规模高分辨率交通监控，有效解决了传统传感器部署成本高和视频分析视角不一致的问题。

### 背景

准确、可扩展的交通监控对实时和长期交通管理至关重要，特别是在自然灾害、大型建设项目或重大政策变化期间。然而，传感器广泛部署受限于高成本，而现有视频分析难以处理动态摄像头视角和海量数据。

### 目的

开发一个基于AI的端到端框架，利用现有交通摄像头基础设施进行大规模高分辨率、纵向交通分析。

### 方法

使用在本地城市场景微调的YOLOv11模型实时提取交通数据；引入基于图的视角归一化方法解决非固定摄像头视角问题；集成领域特定大型语言模型处理24/7视频流生成自动摘要；使用纽约市约1000个交通摄像头的900多万张图像进行验证。

### 主要发现

纽约市拥堵缓解区内工作日乘客车辆密度下降9%；卡车数量早期减少但有反弹迹象；行人和骑行活动在走廊和区域范围持续增加；基于示例的提示提高了LLM的数值准确性并减少了幻觉。

### 结论

该框架可作为大规模、政策相关交通监控的实用解决方案，只需极少人工干预，具有基础设施就绪的潜力。

### 翻译

准确、可扩展的交通监控对实时和长期交通管理至关重要，特别是在自然灾害、大型建设项目或重大政策变化等中断期间，如纽约市首个全国性拥堵定价计划。然而，由于安装、维护和数据管理成本高，传感器的广泛部署仍然受限。虽然交通摄像头是成本效益高的替代方案，但现有的视频分析难以处理动态摄像头视角和大型摄像头网络的海量数据。本研究提出了一种基于AI的端到端框架，利用现有交通摄像头基础设施进行大规模高分辨率、纵向分析。在本地城市场景上微调的YOLOv11模型实时提取多模态交通密度和分类指标。为解决非固定平移变焦相机的不一致性问题，我们引入了一种新颖的基于图的视角归一化方法。还集成了领域特定的大型语言模型处理24/7视频流的海量数据，生成频繁、自动的交通模式演变摘要，这远超人工能力。我们在2025年纽约市拥堵定价计划早期部署期间使用来自约1000个交通摄像头的900多万张图像验证了该系统。结果显示，拥堵缓解区内工作日乘客车辆密度下降9%，卡车数量早期减少但有反弹迹象，行人和骑行活动在走廊和区域范围内持续增加。实验表明，基于示例的提示提高了LLM的数值准确性并减少了幻觉。这些发现证明了该框架作为大规模、政策相关交通监控的实用、基础设施就绪解决方案的潜力，只需极少人工干预。


### 论文摘要

Accurate, scalable traffic monitoring is critical for real-time and long-term transportation management, particularly during disruptions such as natural disasters, large construction projects, or major policy changes like New York City's first-in-the-nation congestion pricing program. However, widespread sensor deployment remains limited due to high installation, maintenance, and data management costs. While traffic cameras offer a cost-effective alternative, existing video analytics struggle with dynamic camera viewpoints and massive data volumes from large camera networks. This study presents an end-to-end AI-based framework leveraging existing traffic camera infrastructure for high-resolution, longitudinal analysis at scale. A fine-tuned YOLOv11 model, trained on localized urban scenes, extracts multimodal traffic density and classification metrics in real time. To address inconsistencies from non-stationary pan-tilt-zoom cameras, we introduce a novel graph-based viewpoint normalization method. A domain-specific large language model was also integrated to process massive data from a 24/7 video stream to generate frequent, automated summaries of evolving traffic patterns, a task far exceeding manual capabilities. We validated the system using over 9 million images from roughly 1,000 traffic cameras during the early rollout of NYC congestion pricing in 2025. Results show a 9% decline in weekday passenger vehicle density within the Congestion Relief Zone, early truck volume reductions with signs of rebound, and consistent increases in pedestrian and cyclist activity at corridor and zonal scales. Experiments showed that example-based prompts improved LLM's numerical accuracy and reduced hallucinations. These findings demonstrate the framework's potential as a practical, infrastructure-ready solution for large-scale, policy-relevant traffic monitoring with minimal human intervention.

---

## 239. Multimodal Large Language Model Framework for Safe and Interpretable Grid-Integrated EVs

**论文链接:** [http://arxiv.org/abs/2510.02592v1](http://arxiv.org/abs/2510.02592v1)

**作者:** Jean Douglas Carvalho, Hugo Kenji, Ahmad Mohammad Saber, Glaucia Melo, Max Mauro Dias Santos, Deepa Kundur

**发布时间:** 2025-10-02

**备注:** This paper has been presented at the 2025 IEEE PES Conference on Innovative Smart Grid Technologies (ISGT 2025)

### GPT解析

### 总结

本文提出了一种基于多模态大语言模型的框架，用于处理多模态传感器数据并生成自然语言警报，以提高电动汽车与智能电网集成环境下的驾驶安全性

### 背景

电动汽车与智能电网的集成为交通系统和能源网络提供了独特机会，但确保驾驶员、车辆和周围环境之间的安全且可解释的互动仍是一大挑战

### 目的

开发一个能够处理多模态传感器数据并生成自然语言警报的框架，以增强驾驶员对周围环境的理解并提高驾驶安全性

### 方法

结合视觉感知（YOLOv8）、地理编码定位和CAN总线遥测技术，处理来自真实世界城市道路驾驶的仪器车辆收集的多模态传感器数据

### 主要发现

框架能够有效生成针对行人、自行车手和其他车辆等关键情境的感知警报，通过真实数据案例研究验证了其在实际场景中的有效性

### 结论

大语言模型作为电动出行辅助工具有巨大潜力，通过实现可扩展的车队协调、电动汽车负荷预测和交通感知的能源规划，同时造福交通系统和电网

### 翻译

将电动汽车整合到智能电网中为交通系统和能源网络提供了独特的机会。然而，确保驾驶员、车辆和周围环境之间的安全且可解释的互动仍然是一个关键挑战。本文提出了一种基于多模态大语言模型的框架，处理多模态传感器数据——如目标检测、语义分割和车辆遥测——并为驾驶员生成自然语言警报。该框架使用从城市道路上行驶的仪器车辆收集的真实世界数据进行验证，确保其在实际场景中的适用性。通过结合视觉感知（YOLOv8）、地理编码定位和CAN总线遥测，该框架连接了原始传感器数据和驾驶员理解，使城市驾驶场景能够做出更安全、更明智的决策。使用真实数据的案例研究证明了该框架在生成关键情境感知警报方面的有效性，如靠近行人、自行车手和其他车辆的情况。本文强调了LLM作为电动出行辅助工具的潜力，通过实现可扩展的车队协调、EV负荷预测和交通感知的能源规划，使交通系统和电网双双受益。


### 论文摘要

The integration of electric vehicles (EVs) into smart grids presents unique opportunities to enhance both transportation systems and energy networks. However, ensuring safe and interpretable interactions between drivers, vehicles, and the surrounding environment remains a critical challenge. This paper presents a multi-modal large language model (LLM)-based framework to process multimodal sensor data - such as object detection, semantic segmentation, and vehicular telemetry - and generate natural-language alerts for drivers. The framework is validated using real-world data collected from instrumented vehicles driving on urban roads, ensuring its applicability to real-world scenarios. By combining visual perception (YOLOv8), geocoded positioning, and CAN bus telemetry, the framework bridges raw sensor data and driver comprehension, enabling safer and more informed decision-making in urban driving scenarios. Case studies using real data demonstrate the framework's effectiveness in generating context-aware alerts for critical situations, such as proximity to pedestrians, cyclists, and other vehicles. This paper highlights the potential of LLMs as assistive tools in e-mobility, benefiting both transportation systems and electric networks by enabling scalable fleet coordination, EV load forecasting, and traffic-aware energy planning.   Index Terms - Electric vehicles, visual perception, large language models, YOLOv8, semantic segmentation, CAN bus, prompt engineering, smart grid.

---

## 240. Identifying the Multimodal Hierarchy of Public Transit Systems Using Itinerary Data

**论文链接:** [http://arxiv.org/abs/2509.24220v2](http://arxiv.org/abs/2509.24220v2)

**作者:** Junhee Lee, Seungmo Kang, Jinwoo Lee

**发布时间:** 2025-09-29

### GPT解析

### 总结

这篇论文提出了一种'宏观多模式层次结构'的概念，用于理解城市中不同交通模式之间的互补和竞争关系，以及它们如何影响用户的多模式出行选择。

### 背景

随着城市交通整合传统和新兴模式，公共交通系统变得越来越复杂。不同交通模式之间存在互补和竞争关系，影响用户的多模式出行路线。

### 目的

提供对城市交通模式间交互的清晰、高层次理解，通过引入'宏观多模式层次结构'概念来分析用户的多模式出行行为。

### 方法

提出一种使用多模式智能卡行程数据识别城市多模式层次结构的方法，并在韩国首尔及周边都市圈的实际数据上进行了应用验证。

### 主要发现

出行遵循'先上升后下降'的顺序，开始和结束于具有高可达性的较低层次模式（如步行），同时使用较高层次的模式（如地铁）来提高效率。

### 结论

通过'宏观多模式层次结构'的概念和方法，可以更好地理解城市中不同交通模式之间的关系，为城市交通规划和管理提供新的视角。

### 翻译

随着城市交通融合传统和新兴模式，公共交通系统正变得越来越复杂。一些模式相互补充，而另一些则相互竞争，影响着用户的多模式出行路线。为了提供对这些交互的清晰、高层次理解，我们引入了'宏观多模式层次结构'的概念。在这个框架中，出行遵循'先上升后下降'的顺序，开始和结束于具有高可达性的较低层次模式（如步行），同时使用较高层次的模式（如地铁）来提高效率。我们提出了一种使用多模式智能卡行程数据识别城市多模式层次结构的方法，并展示了其在韩国首尔及周边都市圈收集的实际数据中的应用。


### 论文摘要

As urban mobility integrates traditional and emerging modes, public transit systems are becoming increasingly complex. Some modes complement each other, while others compete, influencing users' multimodal itineraries. To provide a clear, high-level understanding of these interactions, we introduce the concept of a macroscopic multimodal hierarchy. In this framework, trips follow an "ascending-descending" order, starting and ending with lower hierarchical modes (e.g., walking) that offer high accessibility, while utilizing higher modes (e.g., subways) for greater efficiency. We propose a methodology to identify the multimodal hierarchy of a city using multimodal smart card itinerary data and demonstrate its application with actual data collected from Seoul and the surrounding metropolitan area in South Korea.

---

## 241. Semantic Edge-Cloud Communication for Real-Time Urban Traffic Surveillance with ViT and LLMs over Mobile Networks

**论文链接:** [http://arxiv.org/abs/2509.21259v1](http://arxiv.org/abs/2509.21259v1)

**作者:** Murat Arda Onsu, Poonam Lohan, Burak Kantarci, Aisha Syed, Matthew Andrews, Sean Kennedy

**发布时间:** 2025-09-25

**备注:** 17 pages, 12 figures

### GPT解析

### 总结

该研究提出了一种语义通信框架，通过检测感兴趣区域、提取相关图像段并转换为紧凑嵌入向量，显著减少数据传输量，同时保持较高的多模态大语言模型响应准确性。

### 背景

实时城市交通监控对智能交通系统至关重要，通常在城市环境中部署边缘摄像头。然而，多模态大语言模型因计算需求高无法在边缘设备部署，导致需要在云端推理，而边缘到云端的数据传输受带宽限制，可能影响实时性能。

### 目的

解决边缘设备到云端的数据传输问题，减少传输开销，同时保持实时交通监控的性能。

### 方法

使用YOLOv11检测感兴趣区域并裁剪相关图像段，通过视觉变换器(ViT)将图像转换为紧凑嵌入向量传输到云端，云端使用图像解码器重建图像，再由多模态大语言模型生成交通状况描述。

### 主要发现

该方法实现了99.9%的数据传输量减少，对于重建的裁剪图像，大语言模型响应准确率达到89%，而原始裁剪图像的准确率为93%。

### 结论

ViT和大语言模型辅助的边缘-云端语义通信对于实时交通监控是高效且实用的。

### 翻译

实时城市交通监控对智能交通系统至关重要，可确保道路安全、优化交通流、跟踪车辆轨迹并在智慧城市中防止碰撞。在城市环境中部署边缘摄像头是监控道路状况的标准做法。然而，将这些与智能模型集成需要对动态交通场景有深入理解，并提供用户交互的响应式界面。尽管多模态大语言模型可以解释交通图像并生成信息丰富的响应，但由于其高计算需求，在边缘设备上部署是不可行的。因此，大语言模型推理必须在云端进行，这需要将视觉数据从边缘传输到云端，而这一过程因带宽有限而受阻，可能导致延迟，从而影响实时性能。为解决这一挑战，我们提出了一种语义通信框架，显著减少了传输开销。


### 论文摘要

Real-time urban traffic surveillance is vital for Intelligent Transportation Systems (ITS) to ensure road safety, optimize traffic flow, track vehicle trajectories, and prevent collisions in smart cities. Deploying edge cameras across urban environments is a standard practice for monitoring road conditions. However, integrating these with intelligent models requires a robust understanding of dynamic traffic scenarios and a responsive interface for user interaction. Although multimodal Large Language Models (LLMs) can interpret traffic images and generate informative responses, their deployment on edge devices is infeasible due to high computational demands. Therefore, LLM inference must occur on the cloud, necessitating visual data transmission from edge to cloud, a process hindered by limited bandwidth, leading to potential delays that compromise real-time performance. To address this challenge, we propose a semantic communication framework that significantly reduces transmission overhead. Our method involves detecting Regions of Interest (RoIs) using YOLOv11, cropping relevant image segments, and converting them into compact embedding vectors using a Vision Transformer (ViT). These embeddings are then transmitted to the cloud, where an image decoder reconstructs the cropped images. The reconstructed images are processed by a multimodal LLM to generate traffic condition descriptions. This approach achieves a 99.9% reduction in data transmission size while maintaining an LLM response accuracy of 89% for reconstructed cropped images, compared to 93% accuracy with original cropped images. Our results demonstrate the efficiency and practicality of ViT and LLM-assisted edge-cloud semantic communication for real-time traffic surveillance.

---

## 242. Boosting LiDAR-Based Localization with Semantic Insight: Camera Projection versus Direct LiDAR Segmentation

**论文链接:** [http://arxiv.org/abs/2509.20486v1](http://arxiv.org/abs/2509.20486v1)

**作者:** Sven Ochs, Philip Schörner, Marc René Zofka, J. Marius Zöllner

**发布时间:** 2025-09-24

### GPT解析

### 总结

本文提出了一种结合语义相机数据与激光雷达分割的方法，通过将激光雷达点投影到相机语义分割空间，提高自主移动系统中激光雷达定位的精度和可靠性。

### 背景

激光雷达数据的语义分割面临较大挑战，特别是在处理不同类型传感器和配置时。融入语义信息可增强基于激光雷达的定位技术。

### 目的

开发一种融合语义相机数据与激光雷达分割的方法，解决激光雷达数据语义分割挑战，提高定位精度和可靠性。

### 方法

1) 将激光雷达点投影到相机语义分割空间；2) 使用CoCar NextGen平台验证，提供多样化传感器配置；3) 采用Depth-Anything网络进行相机图像分割；4) 使用自适应分割网络进行激光雷达分割；5) 利用RTK-GNSS建立可靠基准；6) 在德国卡尔斯鲁厄进行55公里多环境测试。

### 主要发现

结合语义相机数据与激光雷达分割可有效提高激光雷达定位精度和可靠性，特别是在复杂真实环境中。

### 结论

这种多模态方法为更可靠精确的自主导航系统开辟了道路，尤其适用于复杂真实世界环境。

### 翻译

激光雷达数据的语义分割存在相当大的挑战，尤其是在处理不同类型的传感器和配置时。然而，融入语义信息可以显著增强自主移动系统中基于激光雷达的定位技术的准确性和鲁棒性。我们提出了一种将语义相机数据与激光雷达分割相结合的方法来应对这一挑战。通过将激光雷达点投影到相机的语义分割空间，我们的方法提高了激光雷达定位流水线的精度和可靠性。为了验证，我们利用了FZI信息技术研究中心的CoCar NextGen平台，该平台提供多样化的传感器模态和配置。CoCar NextGen的传感器设置能够对不同类型的传感器进行全面分析。我们的评估利用了最先进的Depth-Anything网络进行相机图像分割，以及用于激光雷达分割的自适应分割网络。为了建立可靠的激光雷达定位基准，我们使用了配备实时动态(RTK)修正的全球导航卫星系统(GNSS)解决方案。此外，我们在德国卡尔斯鲁厄市进行了55公里的广泛道路测试，涵盖了城市区域、多车道道路和农村高速公路等多种环境。这种多模态方法为更可靠和精确的自主导航系统铺平了道路，特别是在复杂的真实世界环境中。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何提高基于激光雷达（LiDAR）的定位系统准确性和鲁棒性的问题。这个问题在现实和研究中非常重要，因为准确的定位是自动驾驶系统的基础需求，特别是在复杂城市环境中。纯LiDAR定位方法在特征稀疏区域、动态场景或恶劣天气条件下表现不佳，而整合语义信息可以显著提高定位的可靠性和精确度，这对安全导航至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到LiDAR语义分割存在挑战，特别是处理不同传感器类型时，同时知道语义信息能增强定位性能。他们设计了一种将相机语义数据与LiDAR结合的方法，通过投影技术将LiDAR点映射到相机语义空间。该方法借鉴了现有工作：参考了Haidar等人的LiDAR分割评估、SalsaNext和TransRVNet等分割模型、Segformer等图像分割方法，以及Reichardt的ImageTo360和Gu的CLFT等融合方法。作者在CoCar NextGen平台上进行了55公里的实际道路测试，使用GNSS/RTK作为基准进行验证。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将相机的语义信息投影到LiDAR点云上，而不是直接对LiDAR点云进行语义分割。这种方法利用相机在语义分割上的优势，同时保留LiDAR在深度感知上的优势。整体流程包括：1）对相机图像进行语义分割（使用Depth-Anything网络）；2）同步相机和LiDAR的时间数据；3）将LiDAR点投影到分割后的图像平面上；4）将带有语义信息的点云用于定位和地图构建。这种方法的结果与直接对LiDAR进行语义分割的方法进行对比，两种结果都用于相同的映射过程。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出新的语义投影方法，将相机语义信息整合到LiDAR定位中；2）全面比较相机投影与直接LiDAR分割两种方法；3）在多样化真实环境中进行广泛实验验证。相比之前工作，不同之处在于：不直接依赖LiDAR点云语义分割，避免了LiDAR分割面临的传感器稀疏性和计算复杂性问题；采用简单投影方法而非复杂网络融合；在推理阶段不需要图像数据；提供了更全面的性能评估，特别是在GNSS受限环境下的表现。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种通过将相机语义信息投影到LiDAR点云上来增强LiDAR定位的新方法，并通过广泛的实验验证了其在提高定位精度和鲁棒性方面的优势，特别是在GNSS受限的环境中。'}


### 论文摘要

Semantic segmentation of LiDAR data presents considerable challenges, particularly when dealing with diverse sensor types and configurations. However, incorporating semantic information can significantly enhance the accuracy and robustness of LiDAR-based localization techniques for autonomous mobile systems. We propose an approach that integrates semantic camera data with LiDAR segmentation to address this challenge. By projecting LiDAR points into the semantic segmentation space of the camera, our method enhances the precision and reliability of the LiDAR-based localization pipeline.   For validation, we utilize the CoCar NextGen platform from the FZI Research Center for Information Technology, which offers diverse sensor modalities and configurations. The sensor setup of CoCar NextGen enables a thorough analysis of different sensor types. Our evaluation leverages the state-of-the-art Depth-Anything network for camera image segmentation and an adaptive segmentation network for LiDAR segmentation. To establish a reliable ground truth for LiDAR-based localization, we make us of a Global Navigation Satellite System (GNSS) solution with Real-Time Kinematic corrections (RTK). Additionally, we conduct an extensive 55 km drive through the city of Karlsruhe, Germany, covering a variety of environments, including urban areas, multi-lane roads, and rural highways. This multimodal approach paves the way for more reliable and precise autonomous navigation systems, particularly in complex real-world environments.

---

## 243. Large Language Models for Pedestrian Safety: An Application to Predicting Driver Yielding Behavior at Unsignalized Intersections

**论文链接:** [http://arxiv.org/abs/2509.19657v1](http://arxiv.org/abs/2509.19657v1)

**作者:** Yicheng Yang, Zixian Li, Jean Paul Bizimana, Niaz Zafri, Yongfeng Dong, Tianyi Li

**发布时间:** 2025-09-24

### GPT解析

### 总结

本研究探讨了行人安全在城市交通中的重要性，以及驾驶员与行人互动对安全的影响，提出利用多模态大型语言模型(LLMs)通过创新的提示设计来建模驾驶员让行行为，实验表明GPT-4o在准确率和召回率方面表现最佳，Deepseek-V3在精确度方面表现突出，为行人安全系统部署提供了实践指导。

### 背景

行人安全是城市交通的关键组成部分，受到行人决策与驾驶员在斑马线处让行行为之间互动的强烈影响。传统机器学习模型由于依赖固定特征表示和有限的可解释性，难以捕捉这些多因素互动中细微且依赖于上下文的推理过程。

### 目的

准确建模驾驶员与行人在交叉路口的互动，捕捉这些行为的复杂性，并利用大型语言模型从异构交通数据中提取模式，实现可解释且上下文感知的驾驶员让行行为推理。

### 方法

利用多模态大型语言模型，通过创新的提示设计，融入领域特定知识、结构化推理和少样本提示，以驾驶员让行行为建模为例，展示行人-驾驶员互动建模的应用。

### 主要发现

最先进的大型语言模型与传统分类器的基准测试表明，GPT-4o在准确率和召回率方面持续达到最高水平，而Deepseek-V3在精确度方面表现出色。

### 结论

这些发现突显了模型性能与计算效率之间的关键权衡，为在现实世界行人安全系统中部署大型语言模型提供了实践指导。

### 翻译

行人安全是城市交通的重要组成部分，并受到行人在斑马线处决策与驾驶员让行行为之间互动的强烈影响。在交叉路口建模驾驶员-行人互动需要准确捕捉这些行为的复杂性。传统机器学习模型通常难以捕捉这些多因素互动所需的细微且依赖于上下文的推理，这是因为它们依赖于固定的特征表示和有限的可解释性。相比之下，大型语言模型(LLMs)适合从异构交通数据中提取模式，能够准确建模驾驶员-行人互动。因此，本文通过创新的提示设计利用多模态大型语言模型，该设计融入了领域特定知识、结构化推理和少样本提示，实现了可解释且上下文感知的驾驶员让行行为推理，作为建模行人-驾驶员互动的示例应用。我们将最先进的大型语言模型与传统分类器进行了基准测试，发现GPT-4o在准确率和召回率方面持续达到最高水平，而Deepseek-V3在精确度方面表现出色。这些发现强调了模型性能与计算效率之间的关键权衡，为在现实世界行人安全系统中部署大型语言模型提供了实践指导。


### 论文摘要

Pedestrian safety is a critical component of urban mobility and is strongly influenced by the interactions between pedestrian decision-making and driver yielding behavior at crosswalks. Modeling driver--pedestrian interactions at intersections requires accurately capturing the complexity of these behaviors. Traditional machine learning models often struggle to capture the nuanced and context-dependent reasoning required for these multifactorial interactions, due to their reliance on fixed feature representations and limited interpretability. In contrast, large language models (LLMs) are suited for extracting patterns from heterogeneous traffic data, enabling accurate modeling of driver-pedestrian interactions. Therefore, this paper leverages multimodal LLMs through a novel prompt design that incorporates domain-specific knowledge, structured reasoning, and few-shot prompting, enabling interpretable and context-aware inference of driver yielding behavior, as an example application of modeling pedestrian--driver interaction. We benchmarked state-of-the-art LLMs against traditional classifiers, finding that GPT-4o consistently achieves the highest accuracy and recall, while Deepseek-V3 excels in precision. These findings highlight the critical trade-offs between model performance and computational efficiency, offering practical guidance for deploying LLMs in real-world pedestrian safety systems.

---

## 244. Zero-Shot Multi-Spectral Learning: Reimagining a Generalist Multimodal Gemini 2.5 Model for Remote Sensing Applications

**论文链接:** [http://arxiv.org/abs/2509.19087v1](http://arxiv.org/abs/2509.19087v1)

**作者:** Ganesh Mallya, Yotam Gigi, Dahun Kim, Maxim Neumann, Genady Beryozkin, Tomer Shekel, Anelia Angelova

**发布时间:** 2025-09-23

### GPT解析

### 总结

本文提出了一种无需训练的方法，将多光谱数据以零样本模式输入到通用多模态模型中，解决了专业多光谱数据无法与通用大型多模态模型兼容的问题。

### 背景

多光谱图像在遥感应用中扮演重要角色，包括土地利用分类、环境监测和城市规划。多光谱图像因其额外的光谱波段与地面物理材料（如冰、水和植被）有强相关性而被广泛采用。目前，这类数据的自动分析主要通过专门为多光谱输入训练的机器学习模型进行，这些模型训练和支持成本高。强大的通用大型多模态模型虽然能解决许多视觉问题，但无法理解专业的多光谱信号。

### 目的

解决专业多光谱数据无法与通用大型多模态模型兼容的问题，使地理空间专业人员能够利用强大的多模态模型（如Gemini2.5）来加速工作。

### 方法

提出一种无需训练的方法，将多光谱数据以零样本模式输入到仅基于RGB输入训练的通用多模态模型中。利用多模态模型对视觉空间的理解，将输入调整到该空间，并将领域特定信息作为指令注入到模型中。

### 主要发现

在流行的遥感基准测试中，对于土地覆盖和土地利用分类，观察到了显著的零样本性能提升。展示了Gemini2.5模型对新输入的易适应性。结果强调了地理空间专业人员可以利用强大的多模态模型，受益于其丰富的推理和上下文能力。

### 结论

所提出的方法使地理空间专业人员能够轻松利用强大的多模态模型，加速他们的工作。这些模型能够基于专业的传感器数据提供丰富的推理和上下文能力。

### 翻译

多光谱图像在包括土地利用分类、环境监测和城市规划在内的各种遥感应用中发挥着关键作用。这些图像被广泛采用，因为它们额外的光谱波段与地面上的物理材料（如冰、水和植被）有很强的相关性。这允许更准确的识别，并且从Sentinel-2和Landsat等任务中公开获得的可用性增加了它们的价值。目前，此类数据的自动分析主要通过专门为多光谱输入训练的机器学习模型管理，这些模型的训练和支持成本高昂。此外，尽管为遥感提供了很多实用功能，但这些额外的输入不能与强大的通用大型多模态模型一起使用，这些模型能够解决许多视觉问题，但无法理解专业的多光谱信号。为解决此问题，我们提出了一种无需训练的方法，将新的多光谱数据仅以零样本模式引入，作为通用多模态模型的输入，这些模型仅基于RGB输入进行训练。我们的方法利用了多模态模型对视觉空间的理解，并建议将输入调整到该空间，并将领域特定信息作为指令注入到模型中。我们使用Gemini2.5模型举例说明了这一想法，并观察到该方法在流行的遥感基准测试中对于土地覆盖和土地利用分类的零样本性能显著提升，并展示了Gemini2.5对新输入的易适应性。这些结果强调了地理空间专业人员在使用非标准专业输入时，能够轻松利用强大的多模态模型（如Gemini2.5）来加速他们的工作，受益于这些模型基于专业传感器数据的丰富推理和上下文能力。


### 论文摘要

Multi-spectral imagery plays a crucial role in diverse Remote Sensing applications including land-use classification, environmental monitoring and urban planning. These images are widely adopted because their additional spectral bands correlate strongly with physical materials on the ground, such as ice, water, and vegetation. This allows for more accurate identification, and their public availability from missions, such as Sentinel-2 and Landsat, only adds to their value. Currently, the automatic analysis of such data is predominantly managed through machine learning models specifically trained for multi-spectral input, which are costly to train and support. Furthermore, although providing a lot of utility for Remote Sensing, such additional inputs cannot be used with powerful generalist large multimodal models, which are capable of solving many visual problems, but are not able to understand specialized multi-spectral signals.   To address this, we propose a training-free approach which introduces new multi-spectral data in a Zero-Shot-only mode, as inputs to generalist multimodal models, trained on RGB-only inputs. Our approach leverages the multimodal models' understanding of the visual space, and proposes to adapt to inputs to that space, and to inject domain-specific information as instructions into the model. We exemplify this idea with the Gemini2.5 model and observe strong Zero-Shot performance gains of the approach on popular Remote Sensing benchmarks for land cover and land use classification and demonstrate the easy adaptability of Gemini2.5 to new inputs. These results highlight the potential for geospatial professionals, working with non-standard specialized inputs, to easily leverage powerful multimodal models, such as Gemini2.5, to accelerate their work, benefiting from their rich reasoning and contextual capabilities, grounded in the specialized sensor data.

---

## 245. Remote Sensing-Oriented World Model

**论文链接:** [http://arxiv.org/abs/2509.17808v2](http://arxiv.org/abs/2509.17808v2)

**作者:** Yuxi Lu, Biao Wu, Zhidong Li, Kunqi Li, Chenya Huang, Huacan Wang, Qizhen Lan, Ronghao Chen, Ling Chen, Bin Liang

**发布时间:** 2025-09-22

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

本文首次提出遥感领域世界建模框架，通过方向条件化的空间外推方法生成语义一致的相邻图像块，并开发了RSWISE基准进行评估，同时提出了RemoteBAGEL模型，实验证明其性能优于现有基线方法。

### 背景

现有世界建模方法主要在合成环境或受限场景中评估，限制了它们在真实世界中的应用；而遥感应用（如灾害响应和城市规划）迫切需要空间推理能力。

### 目的

填补现有世界建模方法在遥感领域的空白，开发适用于真实世界场景的空间推理能力。

### 方法

将遥感世界建模定义为方向条件化的空间外推，开发RSWISE基准（包含1600个评估任务，涵盖四个场景：一般、洪水、城市和农村），并提出了RemoteBAGEL模型（在遥感数据上微调的统一多模态模型）。

### 主要发现

RemoteBAGEL模型在RSWISE基准上始终优于最先进的基线方法，证明了其在遥感世界建模任务中的有效性。

### 结论

遥感世界建模框架为空间推理在遥感领域的应用提供了新思路，RemoteBAGEL模型展示了良好的性能，为灾害响应和城市规划等应用提供了支持。

### 翻译

世界模型在人工智能中显示出通过预测和推理超出直接观察的世界状态的潜力。然而，现有方法主要在合成环境或受限场景设置中进行评估，限制了它们在具有广泛空间覆盖和复杂语义的真实世界背景中的验证。同时，遥感应用迫切需要空间推理能力用于灾害响应和城市规划。本文通过引入遥感领域首个世界建模框架弥合了这一差距。我们将遥感世界建模定义为方向条件化的空间外推，即模型根据中心观测和方向指令生成语义一致的相邻图像块。为了进行严格评估，我们开发了RSWISE（遥感世界图像空间评估），一个包含1600个评估任务的基准，涵盖四个场景：一般、洪水、城市和农村。RSWISE结合了使用GPT-4o作为语义判断器的视觉保真度评估和指令合规性评估，确保模型真正执行空间推理而非简单复制。随后，我们提出了RemoteBAGEL，一个在遥感数据上微调的统一多模态模型，用于空间外推任务。大量实验证明，RemoteBAGEL在RSWISE上始终优于最先进的基线方法。


### 论文摘要

World models have shown potential in artificial intelligence by predicting and reasoning about world states beyond direct observations. However, existing approaches are predominantly evaluated in synthetic environments or constrained scene settings, limiting their validation in real-world contexts with broad spatial coverage and complex semantics. Meanwhile, remote sensing applications urgently require spatial reasoning capabilities for disaster response and urban planning. This paper bridges these gaps by introducing the first framework for world modeling in remote sensing. We formulate remote sensing world modeling as direction-conditioned spatial extrapolation, where models generate semantically consistent adjacent image tiles given a central observation and directional instruction. To enable rigorous evaluation, we develop RSWISE (Remote Sensing World-Image Spatial Evaluation), a benchmark containing 1,600 evaluation tasks across four scenarios: general, flood, urban, and rural. RSWISE combines visual fidelity assessment with instruction compliance evaluation using GPT-4o as a semantic judge, ensuring models genuinely perform spatial reasoning rather than simple replication. Afterwards, we present RemoteBAGEL, a unified multimodal model fine-tuned on remote sensing data for spatial extrapolation tasks. Extensive experiments demonstrate that RemoteBAGEL consistently outperforms state-of-the-art baselines on RSWISE.

---

## 246. V-SenseDrive: A Privacy-Preserving Road Video and In-Vehicle Sensor Fusion Framework for Road Safety & Driver Behaviour Modelling

**论文链接:** [http://arxiv.org/abs/2509.18187v1](http://arxiv.org/abs/2509.18187v1)

**作者:** Muhammad Naveed, Nazia Perwaiz, Sidra Sultana, Mohaira Ahmad, Muhammad Moazam Fraz

**发布时间:** 2025-09-18

### GPT解析

### 总结

V-SenseDrive是首个在巴基斯坦驾驶环境中收集的隐私保护型多模态驾驶员行为数据集，结合了智能手机传感器数据和道路视频，记录了多种道路条件下的驾驶行为。

### 背景

道路交通事故是重大公共卫生挑战，尤其在巴基斯坦等道路条件复杂国家。现有数据集多来自发达国家，缺乏新兴经济体行为多样性，且驾驶员面部录像侵犯隐私。

### 目的

创建首个完全在巴基斯坦驾驶环境中收集的隐私保护型多模态驾驶员行为数据集，填补全球驾驶员行为数据集的空白。

### 方法

结合智能手机惯性传感器和GPS数据与同步道路视频；记录正常、激进和危险三种驾驶行为；在城市主干道、次级道路和高速公路收集数据；使用定制Android应用获取高频传感器数据和视频；所有数据源精确时间对齐。

### 主要发现

V-SenseDrive代表巴基斯坦真实世界驾驶情况；数据集分为原始层、处理层和语义层，确保未来研究适应性。

### 结论

V-SenseDrive填补全球驾驶员行为数据集空白，为驾驶员行为分类、交通安全分析和ADAS开发研究奠定基础。

### 翻译

道路交通事故仍然是重大公共卫生挑战，特别是在巴基斯坦等具有异质道路条件、混合交通流和多变驾驶纪律的国家。可靠检测不安全驾驶行为是改善道路安全、促进先进驾驶辅助系统(ADAS)发展以及支持保险和车队管理数据驱动决策的前提。大多数现有数据集来自发达国家，对新兴经济体的行为多样性代表性有限，且驾驶员面部录像违反隐私保护原则。我们提出V-SenseDrive，这是首个完全在巴基斯坦驾驶环境中收集的隐私保护型多模态驾驶员行为数据集。V-SenseDrive结合了基于智能手机的惯性和GPS传感器数据与同步的道路面向视频，记录了城市主干道、次级道路和高速公路上的三种目标驾驶行为（正常、激进和危险）。数据使用定制的Android应用程序收集，捕获高频加速度计、陀螺仪和GPS流以及连续视频，所有数据源精确时间对齐以实现多模态分析。本工作重点在于数据采集过程，包括参与者选择、驾驶场景、环境考虑和传感器视频同步技术。数据集分为原始层、处理层和语义层，确保对驾驶员行为分类、交通安全分析和ADAS开发的未来研究具有适应性。通过代表巴基斯坦的真实世界驾驶情况，V-SenseDrive填补了全球驾驶员行为数据集的关键空白，为上下文感知的智能交通解决方案奠定了基础。


### 论文摘要

Road traffic accidents remain a major public health challenge, particularly in countries with heterogeneous road conditions, mixed traffic flow, and variable driving discipline, such as Pakistan. Reliable detection of unsafe driving behaviours is a prerequisite for improving road safety, enabling advanced driver assistance systems (ADAS), and supporting data driven decisions in insurance and fleet management. Most of existing datasets originate from the developed countries with limited representation of the behavioural diversity observed in emerging economies and the driver's face recording voilates the privacy preservation. We present V-SenseDrive, the first privacy-preserving multimodal driver behaviour dataset collected entirely within the Pakistani driving environment. V-SenseDrive combines smartphone based inertial and GPS sensor data with synchronized road facing video to record three target driving behaviours (normal, aggressive, and risky) on multiple types of roads, including urban arterials, secondary roads, and motorways. Data was gathered using a custom Android application designed to capture high frequency accelerometer, gyroscope, and GPS streams alongside continuous video, with all sources precisely time aligned to enable multimodal analysis. The focus of this work is on the data acquisition process, covering participant selection, driving scenarios, environmental considerations, and sensor video synchronization techniques. The dataset is structured into raw, processed, and semantic layers, ensuring adaptability for future research in driver behaviour classification, traffic safety analysis, and ADAS development. By representing real world driving in Pakistan, V-SenseDrive fills a critical gap in the global landscape of driver behaviour datasets and lays the groundwork for context aware intelligent transportation solutions.

---

## 247. Exploring multimodal implicit behavior learning for vehicle navigation in simulated cities

**论文链接:** [http://arxiv.org/abs/2509.15400v1](http://arxiv.org/abs/2509.15400v1)

**作者:** Eric Aislan Antonelo, Gustavo Claudio Karl Couto, Christian Möller

**发布时间:** 2025-09-18

**备注:** ENIAC conference

### GPT解析

### 总结

该论文提出了一种名为数据增强的隐式行为克隆(DA-IBC)方法，用于解决标准行为克隆无法处理多模态驾驶决策的问题。通过基于能量模型的隐式行为克隆，结合数据增强技术和更好的初始化方法，该方法在CARLA模拟器的城市驾驶任务中表现优于标准IBC，能够有效表示多模态动作分布。

### 背景

标准行为克隆(BC)在学习多模态驾驶决策时存在局限，无法处理同一场景下存在多个有效动作的情况。

### 目的

探索使用基于能量模型(EBMs)的隐式行为克隆(IBC)来更好地捕捉驾驶决策中的多模态性。

### 方法

提出数据增强的IBC(DA-IBC)方法，通过扰动专家动作形成IBC训练的反例，并使用更好的初始化方法进行无导数推理。

### 主要发现

在CARLA模拟器中使用鸟瞰图输入的实验表明，DA-IBC在专为评估多模态行为学习而设计的城市驾驶任务中优于标准IBC；学习的能量景观能够表示多模态动作分布，而BC无法实现这一点。

### 结论

DA-IBC方法在处理多模态驾驶决策方面比标准BC和IBC更有效，能够更好地捕捉复杂驾驶场景中的多模态行为。

### 翻译

标准行为克隆(BC)无法学习多模态驾驶决策，即在同一场景下存在多个有效动作的情况。我们探索了使用基于能量模型(EBMs)的隐式行为克隆(IBC)来更好地捕捉这种多模态性。我们提出了数据增强的IBC(DA-IBC)，通过扰动专家动作形成IBC训练的反例，并使用更好的初始化方法进行无导数推理。在CARLA模拟器中使用鸟瞰图输入的实验表明，在专为评估多模态行为学习而设计的城市驾驶任务中，DA-IBC优于标准IBC。学习的能量景观能够表示多模态动作分布，而BC无法实现这一点。


### 论文摘要

Standard Behavior Cloning (BC) fails to learn multimodal driving decisions, where multiple valid actions exist for the same scenario. We explore Implicit Behavioral Cloning (IBC) with Energy-Based Models (EBMs) to better capture this multimodality. We propose Data-Augmented IBC (DA-IBC), which improves learning by perturbing expert actions to form the counterexamples of IBC training and using better initialization for derivative-free inference. Experiments in the CARLA simulator with Bird's-Eye View inputs demonstrate that DA-IBC outperforms standard IBC in urban driving tasks designed to evaluate multimodal behavior learning in a test environment. The learned energy landscapes are able to represent multimodal action distributions, which BC fails to achieve.

---

## 248. A Modular and Multimodal Generative AI Framework for Urban Building Energy Data: Generating Synthetic Homes

**论文链接:** [http://arxiv.org/abs/2509.09794v1](http://arxiv.org/abs/2509.09794v1)

**作者:** Jackson Eshbaugh, Chetan Tiwari, Jorge Silveyra

**发布时间:** 2025-09-11

**备注:** 44 pages; 2 appendices; 9 figures; 1 table. Code available at https://github.com/Lafayette-EshbaughSilveyra-Group/synthetic-homes

### GPT解析

### 总结

该研究介绍了一个基于生成式人工智能的模块化多模态框架，用于从公开住宅信息和图像中生成能源建模所需数据，减少对昂贵或受限数据源的依赖，促进更易获取和可重复的研究。

### 背景

计算模型已成为能源建模研究的强大工具，具有可扩展性和定量结果的优点，但这些模型需要大量数据，其中一些数据难以获取、成本高昂或涉及隐私问题。

### 目的

开发一个框架，从公开可获取的住宅信息和图像中生成能源建模所需的数据，减少对昂贵或受限数据源的依赖。

### 方法

引入一个模块化多模态框架，利用生成式人工智能从公开住宅信息和图像中生成数据，提供并评估了展示该框架的流程及其生成式AI组件。

### 主要发现

实验表明，该框架使用AI避免了生成模型常见的问题，能够产生真实且带有标签的数据。

### 结论

通过减少对昂贵或受限数据源的依赖，该框架为更易获取和可重复的研究铺平了道路。

### 翻译

计算模型已成为能源建模研究的强大工具，以其可扩展性和定量结果而著称。然而，这些模型需要大量数据，其中一些数据难以获取、成本高昂或引发隐私问题。我们引入了一个模块化多模态框架，利用生成式人工智能从公开可获取的住宅信息和图像中生成这些数据。此外，我们提供了展示该框架的流程，并评估了其生成式AI组件。我们的实验表明，我们的框架使用AI避免了生成模型的常见问题。我们的框架生成真实且带有标签的数据。通过减少对昂贵或受限数据源的依赖，我们为更易获取和可重复的研究铺平了道路。


### 论文摘要

Computational models have emerged as powerful tools for energy modeling research, touting scalability and quantitative results. However, these models require a plethora of data, some of which is inaccessible, expensive, or raises privacy concerns. We introduce a modular multimodal framework to produce this data from publicly accessible residential information and images using generative artificial intelligence (AI). Additionally, we provide a pipeline demonstrating this framework, and we evaluate its generative AI components. Our experiments show that our framework's use of AI avoids common issues with generative models. Our framework produces realistic, labeled data. By reducing dependence on costly or restricted data sources, we pave a path towards more accessible and reproducible research.

---

## 249. Distributed Optimization of Pairwise Polynomial Graph Spectral Functions via Subgraph Optimization

**论文链接:** [http://arxiv.org/abs/2511.11517v1](http://arxiv.org/abs/2511.11517v1)

**作者:** Jitian Liu, Nicolas Kozachuk, Subhrajit Bhattacharya

**发布时间:** 2025-11-14

**备注:** 22 pages, 8 figures

### GPT解析

### 总结

研究固定拓扑和全局权重预算下有限度多项式Laplacian谱目标的分布式优化，针对整个频谱的集体行为而非少数极值特征值。

### 背景

分布式优化在大型几何图中的应用面临挑战，需要考虑整个频谱的集体行为而非仅关注少数极值特征值。

### 目的

开发一种分布式优化方法，能够在保持约束条件（正性和预算）的同时，针对整个频谱的集体行为进行权重调整。

### 方法

通过将全局成本重新表述为双线性形式，推导局部子图问题，使用基于SVD的ZC矩阵测试使梯度与全局下降方向对齐；提出迭代-嵌入方案在1跳邻域上操作；使用随机闲聊估计全局平均度进行热启动；引入基于学习的设计器预测边更新。

### 主要发现

对于依赖于成对特征值差异的目标，可获得度向量的二次上界；热启动方法实现了与集中优化相比约95%的性能；基于学习的设计器能实现目标值的立即减少。

### 结论

所提出的组件形成了一个实用的、模块化的频谱感知权重调整管道，保持约束并适用于更广泛的整个频谱成本类别。

### 翻译

我们研究固定拓扑和全局权重预算下有限度多项式Laplacian谱目标的分布式优化，针对整个频谱的集体行为而非少数极值特征值。通过将全局成本重新表述为双线性形式，我们推导出局部子图问题，通过基于SVD的ZC矩阵测试使梯度与全局下降方向大致对齐。这导致在1跳邻域上的迭代-嵌入方案，通过构造保持可行性（正性和预算），并能扩展到大型几何图。对于依赖于成对特征值差异h(λ_i-λ_j)的目标，我们获得了度向量的二次上界，这促使通过度正则化进行'热启动'。热启动使用随机闲聊估计全局平均度，加速后续局部下降，同时保持去中心化，实现了与集中优化相比约95%的性能。我们进一步引入基于学习的设计器，预测在最大1跳嵌入上的单次边更新，实现目标值的立即减少。这些组件共同形成了一个实用的、模块化的频谱感知权重调整管道，保持约束并适用于更广泛的整个频谱成本类别。


### 论文摘要

We study distributed optimization of finite-degree polynomial Laplacian spectral objectives under fixed topology and a global weight budget, targeting the collective behavior of the entire spectrum rather than a few extremal eigenvalues. By re-formulating the global cost in a bilinear form, we derive local subgraph problems whose gradients approximately align with the global descent direction via an SVD-based test on the $ZC$ matrix. This leads to an iterate-and-embed scheme over disjoint 1-hop neighborhoods that preserves feasibility by construction (positivity and budget) and scales to large geometric graphs. For objectives that depend on pairwise eigenvalue differences $h(λ_i-λ_j)$, we obtain a quadratic upper bound in the degree vector, which motivates a ``warm-start'' by degree-regularization. The warm start uses randomized gossip to estimate global average degree, accelerating subsequent local descent while maintaining decentralization, and realizing $\sim95\%{}$ of the performance with respect to centralized optimization. We further introduce a learning-based proposer that predicts one-shot edge updates on maximal 1-hop embeddings, yielding immediate objective reductions. Together, these components form a practical, modular pipeline for spectrum-aware weight tuning that preserves constraints and applies across a broader class of whole-spectrum costs.

---

## 250. Parameter-Efficient MoE LoRA for Few-Shot Multi-Style Editing

**论文链接:** [http://arxiv.org/abs/2511.11236v1](http://arxiv.org/abs/2511.11236v1)

**作者:** Cong Cao, Yujie Xu, Xiaodong Xu

**发布时间:** 2025-11-14

### GPT解析

### 总结

该论文提出了一种新颖的小样本风格编辑框架，通过参数高效的多风格专家混合低秩适应方法（MoE LoRA），解决了通用图像编辑模型在新风格上效果不佳的问题。

### 背景

图像编辑近年来受到越来越多的关注，但通用图像编辑模型在面对新风格时往往无法产生令人满意的结果。

### 目的

如何仅使用有限的成对数据有效地微调通用图像编辑模型以适应新风格。

### 方法

构建了一个包含五种不同风格的基准数据集；提出MoE LoRA方法，包含特定于风格和共享风格的路由机制；通过度量引导方法自动确定每层最优秩；探索在DiT模型中插入LoRA的最佳位置；集成对抗学习和流匹配指导扩散训练。

### 主要发现

实验结果表明，提出的方法以显著更少的LoRA参数优于现有的最先进方法。

### 结论

该方法能够有效解决通用图像编辑模型在新风格上的适应问题，实现了更好的编辑效果。

### 翻译

近年来，图像编辑引起了越来越多的关注。然而，当面对新风格时，通用图像编辑模型往往无法产生令人满意的结果。挑战在于如何仅使用有限的成对数据有效地将通用图像编辑模型微调到新风格。为解决这个问题，本文提出了一种新颖的小样本风格编辑框架。为此，我们构建了一个包含五种不同风格的基准数据集。相应地，我们提出了一种参数高效的多风格专家混合低秩适应方法（MoE LoRA），包含特定于风格和共享风格的路由机制，用于联合微调多种风格。风格特定路由确保不同风格之间不会相互干扰，而风格共享路由自适应地将共享MoE LoRAs分配以学习共同模式。我们的MoE LoRA可以通过一种新的度量引导方法自动确定每层的最优秩，该方法估计每个单秩组件的重要性分数。此外，我们探索了在Transformer扩散模型（DiT）中插入LoRA的最佳位置，并集成对抗学习和流匹配来指导扩散训练过程。实验结果表明，我们提出的方法以显著更少的LoRA参数优于现有的最先进方法。


### 论文摘要

In recent years, image editing has garnered growing attention. However, general image editing models often fail to produce satisfactory results when confronted with new styles. The challenge lies in how to effectively fine-tune general image editing models to new styles using only a limited amount of paired data. To address this issue, this paper proposes a novel few-shot style editing framework. For this task, we construct a benchmark dataset that encompasses five distinct styles. Correspondingly, we propose a parameter-efficient multi-style Mixture-of-Experts Low-Rank Adaptation (MoE LoRA) with style-specific and style-shared routing mechanisms for jointly fine-tuning multiple styles. The style-specific routing ensures that different styles do not interfere with one another, while the style-shared routing adaptively allocates shared MoE LoRAs to learn common patterns. Our MoE LoRA can automatically determine the optimal ranks for each layer through a novel metric-guided approach that estimates the importance score of each single-rank component. Additionally, we explore the optimal location to insert LoRA within the Diffusion in Transformer (DiT) model and integrate adversarial learning and flow matching to guide the diffusion training process. Experimental results demonstrate that our proposed method outperforms existing state-of-the-art approaches with significantly fewer LoRA parameters.

---

## 251. Scalable Population Training for Zero-Shot Coordination

**论文链接:** [http://arxiv.org/abs/2511.11083v1](http://arxiv.org/abs/2511.11083v1)

**作者:** Bingyu Hui, Lebin Yu, Quanming Yao, Yunpeng Qu, Xudong Zhang, Jian Wang

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为Scalable Population Training (ScaPT)的高效训练框架，用于解决零样本协调问题中的计算资源限制问题。

### 背景

零样本协调(ZSC)是强化学习研究的热点话题，关注智能体能与未见过的协作者良好协调而无需微调的能力。现有基于群体的训练方法虽能提供良好的零样本协调性能，但受限于计算资源，主要关注小规模群体的多样性优化，忽略了扩大群体规模可能带来的性能提升。

### 目的

解决现有零样本协调方法受计算资源限制的问题，探索扩大群体规模带来的潜在性能提升。

### 方法

提出Scalable Population Training (ScaPT)框架，包含两个关键组件：一个通过选择性共享参数来高效实现群体的元智能体；一个保证群体多样性的互信息正则化器。

### 主要发现

在Hanabi游戏中对ScaPT与代表性框架进行的评估证实了ScaPT的优越性。

### 结论

ScaPT是一种有效的训练框架，能够解决零样本协调中的计算资源限制问题，通过扩大群体规模提高性能。

### 翻译

零样本协调(ZSC)最近已成为强化学习研究的热点话题。它关注智能体的泛化能力，要求它们能够与未见过的协作者良好协调而无需任何微调。基于群体的训练已被证明能提供良好的零样本协调性能；然而，现有方法受限于计算资源，主要关注优化小规模群体的多样性，而忽略了扩大群体规模可能带来的潜在性能提升。为解决这一问题，本文提出了可扩展群体训练(ScaPT)，这是一种高效训练框架，包含两个关键组件：一个通过选择性共享参数来高效实现群体的元智能体；一个保证群体多样性的互信息正则化器。为经验性地验证ScaPT的有效性，本文在Hanabi游戏中对其与代表性框架进行了评估，并证实了其优越性。


### 论文摘要

Zero-shot coordination(ZSC) has become a hot topic in reinforcement learning research recently. It focuses on the generalization ability of agents, requiring them to coordinate well with collaborators that are not seen before without any fine-tuning. Population-based training has been proven to provide good zero-shot coordination performance; nevertheless, existing methods are limited by computational resources, mainly focusing on optimizing diversity in small populations while neglecting the potential performance gains from scaling population size. To address this issue, this paper proposes the Scalable Population Training (ScaPT), an efficient training framework comprising two key components: a meta-agent that efficiently realizes a population by selectively sharing parameters across agents, and a mutual information regularizer that guarantees population diversity. To empirically validate the effectiveness of ScaPT, this paper evaluates it along with representational frameworks in Hanabi and confirms its superiority.

---

## 252. Towards Federated Clustering: A Client-wise Private Graph Aggregation Framework

**论文链接:** [http://arxiv.org/abs/2511.10915v1](http://arxiv.org/abs/2511.10915v1)

**作者:** Guanxiong He, Jie Wang, Liaoyuan Tang, Zheng Wang, Rong Wang, Feiping Nie

**发布时间:** 2025-11-14

### GPT解析

### 总结

这篇论文提出了结构隐私保护联邦图聚类（SPP-FGC）算法，通过利用局部结构图作为隐私保护知识共享的主要媒介，解决了联邦聚类中性能与隐私之间的两难困境。该算法提供单次和迭代两种模式，实验证明在保持隐私的同时，聚类准确性提高了高达10%。

### 背景

联邦聚类面临从分散、未标记数据中提取模式的挑战。当前方法在性能和隐私之间需要妥协：传输嵌入表示有敏感数据泄露风险，而仅共享抽象聚类原型会导致模型准确性降低。

### 目的

解决联邦聚类中性能与隐私之间的两难困境，提出一种新的算法，能够在保护隐私的同时保持高性能。

### 方法

提出结构隐私保护联邦图聚类（SPP-FGC）算法，利用局部结构图作为隐私保护知识共享的主要媒介。采用客户端-服务器逻辑：客户端构建捕获数据内在关系的私有结构图，服务器安全聚合和排列这些图形成综合全局图并导出统一聚类结构。提供两种模式：SPP-FGC是单次方法，适合快速分析；SPP-FGC+是迭代过程，适用于图像等复杂非结构化数据。

### 主要发现

实验表明该框架达到了最先进的性能，与联邦基线相比聚类准确性提高了高达10%（NMI），同时保持了可证明的隐私保证。

### 结论

SPP-FGC算法成功解决了联邦聚类中性能与隐私之间的两难困境，通过利用结构图进行知识共享，超越了传统技术的限制。

### 翻译

联邦聚类解决了从分散、未标记数据中提取模式的关键挑战。然而，当前方法被迫在性能和隐私之间做出妥协：传输嵌入表示存在敏感数据泄露风险，而仅共享抽象聚类原型则导致模型准确性下降。为解决这一困境，我们提出了结构隐私保护联邦图聚类（SPP-FGC），一种新颖算法，创新性地利用局部结构图作为隐私保护知识共享的主要媒介，从而超越了传统技术的局限。我们的框架在清晰的客户端-服务器逻辑上运行；在客户端，每个参与者构建捕获数据内在关系的私有结构图，然后服务器安全聚合并排列这些图，形成从中导出统一聚类结构的综合全局图。该框架提供两种不同模式以适应不同需求。SPP-FGC设计为高效的单次方法，在单次通信轮次中完成任务，适合快速分析。对于图像等更复杂的非结构化数据，SPP-FGC+采用迭代过程，客户端和服务器协作优化特征表示以实现卓越的下游性能。大量实验证明，我们的框架达到了最先进的性能，比联邦基线提高聚类准确性高达10%（NMI），同时保持可证明的隐私保证。


### 论文摘要

Federated clustering addresses the critical challenge of extracting patterns from decentralized, unlabeled data. However, it is hampered by the flaw that current approaches are forced to accept a compromise between performance and privacy: \textit{transmitting embedding representations risks sensitive data leakage, while sharing only abstract cluster prototypes leads to diminished model accuracy}. To resolve this dilemma, we propose Structural Privacy-Preserving Federated Graph Clustering (SPP-FGC), a novel algorithm that innovatively leverages local structural graphs as the primary medium for privacy-preserving knowledge sharing, thus moving beyond the limitations of conventional techniques. Our framework operates on a clear client-server logic; on the client-side, each participant constructs a private structural graph that captures intrinsic data relationships, which the server then securely aggregates and aligns to form a comprehensive global graph from which a unified clustering structure is derived. The framework offers two distinct modes to suit different needs. SPP-FGC is designed as an efficient one-shot method that completes its task in a single communication round, ideal for rapid analysis. For more complex, unstructured data like images, SPP-FGC+ employs an iterative process where clients and the server collaboratively refine feature representations to achieve superior downstream performance. Extensive experiments demonstrate that our framework achieves state-of-the-art performance, improving clustering accuracy by up to 10\% (NMI) over federated baselines while maintaining provable privacy guarantees.

---

## 253. Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback

**论文链接:** [http://arxiv.org/abs/2511.10572v2](http://arxiv.org/abs/2511.10572v2)

**作者:** Mohammadsina Almasi, Hadis Anahideh

**发布时间:** 2025-11-13

**备注:** Accepted at AAAI-26 (AISI Track). Final version to appear in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-26), 2026

### GPT解析

### 总结

论文提出了一种新颖的双层上下文带框架，用于在延迟反馈下的个性化资源分配，能够在动态人群中高效运作，平衡短期效用与长期影响，同时确保公平性。

### 背景

在高风险领域（如教育、就业和医疗保健）公平分配有限资源需平衡短期效用与长期影响，考虑延迟结果、隐藏异质性和伦理约束，而现有学习框架多假设即时反馈或忽略个体特征与干预动态的复杂互动。

### 目的

开发一种在延迟反馈下运作的个性化资源分配系统，能够在具有动态人群、容量约束和时间敏感影响的现实环境中有效运行。

### 方法

采用双层框架：元级别优化子组级预算分配以满足公平性和运营约束；基础级别使用神经网络识别最敏感个体，尊重冷却窗口并通过资源特定延迟内核建模延迟治疗效果；算法通过明确建模时间动态和反馈延迟持续优化决策。

### 主要发现

在教育和劳动力发展的真实数据集验证中，该方法实现了更高累积结果，更好地适应延迟结构，确保子组间公平分配。

### 结论

延迟感知、数据驱动的决策系统具有改善机构政策和社会福利的潜力。

### 翻译

在高风险领域（如教育、就业和医疗保健）公平分配有限资源需要平衡短期效用与长期影响，同时考虑延迟结果、隐藏异质性和伦理约束。然而，大多数基于学习的分配框架要么假设即时反馈，要么忽略个体特征与干预动态之间的复杂相互作用。我们提出了一种新颖的双层上下文带框架，用于在延迟反馈下的个性化资源分配，设计用于在具有动态人群、容量约束和时间敏感影响的现实世界环境中运行。在元级别，模型优化子组级别的预算分配以满足公平性和运营约束。在基础级别，它使用在观察数据上训练的神经网络识别每个组内最敏感的个体，同时尊重冷却时间窗口并通过资源特定的延迟内核建模延迟治疗效果。通过明确建模时间动态和反馈延迟，算法随着新数据的到来不断完善其策略，实现更具响应性和适应性的决策。我们在教育和劳动力发展的两个真实世界数据集上验证了我们的方法，结果表明它实现了更高的累积结果，更好地适应了延迟结构，并确保了子组之间的公平分配。我们的研究结果强调了延迟感知、数据驱动的决策系统在改善机构政策和社会福利方面的潜力。


### 论文摘要

Equitably allocating limited resources in high-stakes domains-such as education, employment, and healthcare-requires balancing short-term utility with long-term impact, while accounting for delayed outcomes, hidden heterogeneity, and ethical constraints. However, most learning-based allocation frameworks either assume immediate feedback or ignore the complex interplay between individual characteristics and intervention dynamics. We propose a novel bi-level contextual bandit framework for individualized resource allocation under delayed feedback, designed to operate in real-world settings with dynamic populations, capacity constraints, and time-sensitive impact. At the meta level, the model optimizes subgroup-level budget allocations to satisfy fairness and operational constraints. At the base level, it identifies the most responsive individuals within each group using a neural network trained on observational data, while respecting cooldown windows and delayed treatment effects modeled via resource-specific delay kernels. By explicitly modeling temporal dynamics and feedback delays, the algorithm continually refines its policy as new data arrive, enabling more responsive and adaptive decision-making. We validate our approach on two real-world datasets from education and workforce development, showing that it achieves higher cumulative outcomes, better adapts to delay structures, and ensures equitable distribution across subgroups. Our results highlight the potential of delay-aware, data-driven decision-making systems to improve institutional policy and social welfare.

---

## 254. Measuring dissimilarity between convex cones by means of max-min angles

**论文链接:** [http://arxiv.org/abs/2511.10483v1](http://arxiv.org/abs/2511.10483v1)

**作者:** Welington de Oliveira, Valentina Sessa, David Sossa

**发布时间:** 2025-11-13

### GPT解析

### 总结

该研究提出了一种基于最大最小角度的两个凸锥之间新的差异度量方法，并将其应用于图像集分类任务，特别是在有限数据条件下的应用。

### 背景

在比较几何形状和集合时，需要有效的差异度量方法。凸锥作为数学中的重要结构，在多个领域有应用，但缺乏有效的差异度量方法。

### 目的

开发一种新的凸锥之间的差异度量方法，并将其应用于小样本学习中的图像集分类任务。

### 方法

基于最大最小角度提出新的差异度量，研究其与Pompeiu-Hausdorff距离的关系，对于多面锥使用非凸切割平面方法进行计算，采用定制版的Kelley切割平面算法。

### 主要发现

所提出的度量与Pompeiu-Hausdorff距离密切相关，在某些锥构型下具有简化或解析形式，对于多面锥可以通过算法近似计算，并且满足必要的优化条件。

### 结论

所提出的差异度量方法有效应用于图像集分类任务，特别是在小样本学习场景下，通过将同一类别的图像集建模为多面锥，可以判断两个图像集是否属于同一类别。

### 翻译

这项工作介绍了一种基于两个凸锥之间最大最小角度的新差异度量。我们证明了这种度量与Pompeiu-Hausdorff距离（一种比较紧致集的成熟度量）密切相关。此外，我们研究了该度量具有简化或解析形式的锥构型。对于多面锥的具体情况，采用非凸切割平面方法来至少近似计算它们之间的度量。我们的方法基于定制版的Kelley切割平面算法，涉及每次迭代解决一个具有挑战性的主程序。当主程序局部解决时，我们的方法产生一个满足底层非凸优化问题某些必要最优性条件的角度。作为所提出的数学和算法框架的应用，我们解决了有限数据条件下的图像集分类任务，属于小样本学习范畴。在此背景下，同一类别的图像集被建模为多面锥，我们的差异度量对于判断两个图像集是否属于同一类别很有用。


### 论文摘要

This work introduces a novel dissimilarity measure between two convex cones, based on the max-min angle between them. We demonstrate that this measure is closely related to the Pompeiu-Hausdorff distance, a well-established metric for comparing compact sets. Furthermore, we examine cone configurations where the measure admits simplified or analytic forms. For the specific case of polyhedral cones, a nonconvex cutting-plane method is deployed to compute, at least approximately, the measure between them. Our approach builds on a tailored version of Kelley's cutting-plane algorithm, which involves solving a challenging master program per iteration. When this master program is solved locally, our method yields an angle that satisfies certain necessary optimality conditions of the underlying nonconvex optimization problem yielding the dissimilarity measure between the cones. As an application of the proposed mathematical and algorithmic framework, we address the image-set classification task under limited data conditions, a task that falls within the scope of the \emph{Few-Shot Learning} paradigm. In this context, image sets belonging to the same class are modeled as polyhedral cones, and our dissimilarity measure proves useful for understanding whether two image sets belong to the same class.

---

## 255. Multi-agent In-context Coordination via Decentralized Memory Retrieval

**论文链接:** [http://arxiv.org/abs/2511.10030v1](http://arxiv.org/abs/2511.10030v1)

**作者:** Tao Jiang, Zichuan Lin, Lihe Li, Yi-Chen Li, Cong Guan, Lei Yuan, Zongzhang Zhang, Yang Yu, Deheng Ye

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出了一种名为MAICC的多智能体去中心化记忆检索方法，通过结合中心化嵌入模型和去中心化模型，在合作式多智能体强化学习中实现更高效的协调和更快的新任务适应。

### 背景

大型transformer模型在多样化数据集上训练后，无需参数更新即可在未见过的任务上表现出良好的少样本学习能力。在强化学习领域，代理通过与环境交互获取上下文并最大化累积奖励，展现出强大的适应性。然而，在合作式多智能体强化学习中，去中心化策略部署可能导致任务对齐和奖励分配不匹配，限制策略适应效率。

### 目的

解决合作式多智能体强化学习中因去中心化部署导致的协调问题，提高代理之间的协调能力并加快对新任务的适应速度。

### 方法

MAICC方法包括：1)训练中心化嵌入模型捕获细粒度轨迹表示；2)使用去中心化模型近似中心化模型获取团队级任务信息；3)检索相关轨迹作为上下文，结合代理当前子轨迹进行决策；4)引入记忆机制平衡测试时在线数据与离线记忆；5)提出混合效用分数结合个人和团队级回报，确保跨代理信用分配。

### 主要发现

在基于层级的觅食(LBF)和SMAC(v1/v2)等合作式多智能体强化学习基准测试中，MAICC能够比现有方法更快地适应未见过的任务。

### 结论

MAICC通过去中心化记忆检索实现多智能体上下文协调，有效解决了合作式多智能体强化学习中的协调挑战，提高了适应新任务的能力。

### 翻译

Large transformer models, trained on diverse datasets, have demonstrated impressive few-shot performance on previously unseen tasks without requiring parameter updates. This capability has also been explored in Reinforcement Learning (RL), where agents interact with the environment to retrieve context and maximize cumulative rewards, showcasing strong adaptability in complex settings. However, in cooperative Multi-Agent Reinforcement Learning (MARL), where agents must coordinate toward a shared goal, decentralized policy deployment can lead to mismatches in task alignment and reward assignment, limiting the efficiency of policy adaptation. To address this challenge, we introduce Multi-agent In-context Coordination via Decentralized Memory Retrieval (MAICC), a novel approach designed to enhance coordination by fast adaptation. Our method involves training a centralized embedding model to capture fine-grained trajectory representations, followed by decentralized models that approximate the centralized one to obtain team-level task information. Based on the learned embeddings, relevant trajectories are retrieved as context, which, combined with the agents' current sub-trajectories, inform decision-making. During decentralized execution, we introduce a novel memory mechanism that effectively balances test-time online data with offline memory. Based on the constructed memory, we propose a hybrid utility score that incorporates both individual- and team-level returns, ensuring credit assignment across agents. Extensive experiments on cooperative MARL benchmarks, including Level-Based Foraging (LBF) and SMAC (v1/v2), show that MAICC enables faster adaptation to unseen tasks compared to existing methods.


### 论文摘要

Large transformer models, trained on diverse datasets, have demonstrated impressive few-shot performance on previously unseen tasks without requiring parameter updates. This capability has also been explored in Reinforcement Learning (RL), where agents interact with the environment to retrieve context and maximize cumulative rewards, showcasing strong adaptability in complex settings. However, in cooperative Multi-Agent Reinforcement Learning (MARL), where agents must coordinate toward a shared goal, decentralized policy deployment can lead to mismatches in task alignment and reward assignment, limiting the efficiency of policy adaptation. To address this challenge, we introduce Multi-agent In-context Coordination via Decentralized Memory Retrieval (MAICC), a novel approach designed to enhance coordination by fast adaptation. Our method involves training a centralized embedding model to capture fine-grained trajectory representations, followed by decentralized models that approximate the centralized one to obtain team-level task information. Based on the learned embeddings, relevant trajectories are retrieved as context, which, combined with the agents' current sub-trajectories, inform decision-making. During decentralized execution, we introduce a novel memory mechanism that effectively balances test-time online data with offline memory. Based on the constructed memory, we propose a hybrid utility score that incorporates both individual- and team-level returns, ensuring credit assignment across agents. Extensive experiments on cooperative MARL benchmarks, including Level-Based Foraging (LBF) and SMAC (v1/v2), show that MAICC enables faster adaptation to unseen tasks compared to existing methods. Code is available at https://github.com/LAMDA-RL/MAICC.

---

## 256. PustakAI: Curriculum-Aligned and Interactive Textbooks Using Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.10002v2](http://arxiv.org/abs/2511.10002v2)

**作者:** Shivam Sharma, Riya Naik, Tejas Gawas, Heramb Patil, Kunal Korgaonkar

**发布时间:** 2025-11-13

### GPT解析

### 总结

本研究提出了'PustakAI'框架，用于设计和评估与印度NCERT课程大纲对齐的'NCERT-QA'问答数据集，涵盖6-8年级英语和科学科目，并评估了不同提示技术和各类LLMs在教育应用中的效果。

### 背景

大型语言模型(LLMs)在理解和生成类人内容方面表现出显著能力，彻底改变了多个领域。在教育领域，LLMs为个性化互动学习提供了潜力，特别是在教学资源有限的地区。然而，将LLMs有效适应特定课程内容(如NCERT课程大纲)在准确性、对齐性和教学相关性方面存在挑战。

### 目的

设计并评估一个与NCERT课程大纲对齐的问答数据集，评估不同提示技术在该数据集上的效果，并分析开源和高性能LLMs作为教育工具的优势和局限性。

### 方法

提出'PustakAI'框架，构建'NCERT-QA'数据集，将问答对分为事实型、推理型和其他类型。使用元提示、少样本提示和思维链风格提示等技术进行评估，采用多样化指标分析哪种提示方法更符合课程要求。

### 主要发现

研究确定了不同提示技术在NCERT课程问答数据集上的效果差异，识别了适合特定问题类型的最佳提示方法，并分析了开源和高性能LLMs作为教育工具的优势和局限性。

### 结论

'PustakAI'框架和'NCERT-QA'数据集为将LLMs应用于特定课程内容提供了有效方法，适当提示技术和模型选择可使LLMs成为正式教育系统的有效辅助工具，但也需考虑准确性和课程对齐性。

### 翻译

大型语言模型(LLMs)在理解和生成类人内容方面表现出显著能力。这彻底改变了医疗保健、软件开发和教育等多个领域。在教育领域，LLMs为个性化互动学习体验提供了潜力，特别是在教学资源有限的地区。然而，将这些模型有效地适应特定课程内容，如印度的国家教育研究与培训委员会(NCERT)课程大纲，在准确性、对齐性和教学相关性方面存在独特挑战。在本文中，我们提出了'PustakAI'框架(注：Pustak在许多印度语言中意为'书')，用于设计和评估一个名为'NCERT-QA'的新型问答数据集，该数据集与NCERT课程大纲对齐，涵盖6-8年级的英语和科学科目。我们将整理的问答对分为事实型、推理型和其他(评估型和推理型)。我们使用多种提示技术(如元提示、少样本提示和思维链风格提示)对数据集进行评估，并采用多样化的评估指标，以了解哪种方法更有效地与课程的结构和要求对齐。除了数据集的可用性外，我们还分析了当前开源LLMs(Gemma3:1b、Llama3.2:3b和Nemotron-mini:4b)和高性能LLMs(Llama-4-Scout-17B和Deepseek-r1-70B)作为正式教育系统中基于AI的学习工具的优势和局限性。


### 论文摘要

Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating human-like content. This has revolutionized various sectors such as healthcare, software development, and education. In education, LLMs offer potential for personalized and interactive learning experiences, especially in regions with limited teaching resources. However, adapting these models effectively to curriculum-specific content, such as the National Council of Educational Research and Training (NCERT) syllabus in India, presents unique challenges in terms of accuracy, alignment, and pedagogical relevance. In this paper, we present the framework "PustakAI"\footnote{Pustak means `book' in many Indian languages.} for the design and evaluation of a novel question-answering dataset "NCERT-QA" aligned with the NCERT curriculum for English and Science subjects of grades 6 to 8. We classify the curated QA pairs as Factoid, Inferential, and Others (evaluative and reasoning). We evaluate the dataset with various prompting techniques, such as meta-prompt, few-shot, and CoT-style prompting, using diverse evaluation metrics to understand which approach aligns more efficiently with the structure and demands of the curriculum. Along with the usability of the dataset, we analyze the strengths and limitations of current open-source LLMs (Gemma3:1b, Llama3.2:3b, and Nemotron-mini:4b) and high-end LLMs (Llama-4-Scout-17B and Deepseek-r1-70B) as AI-based learning tools in formal education systems.

---

## 257. Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.09933v1](http://arxiv.org/abs/2511.09933v1)

**作者:** Yuhang Zhou, Yanxiang Zhao, Zhongyun Hua, Zhipu Liu, Zhaoquan Gu, Qing Liao, Leo Yu Zhang

**发布时间:** 2025-11-13

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种针对行人重识别(ReID)任务中的对抗攻击防御方法，通过去偏双不变防御框架解决模型偏差和复合泛化要求两大挑战。

### 背景

行人重识别是许多实际应用如行人轨迹跟踪的基础任务，但先进的深度学习ReID模型容易受到对抗攻击，微小扰动会导致错误预测。分类任务的防御策略扩展到度量学习任务如ReID的研究较少，且现有防御未能解决对抗性鲁棒ReID的固有挑战。

### 目的

系统识别行人ReID中对抗防御的两个关键挑战：模型偏差和复合泛化要求，并提出相应解决方案。

### 方法

提出去偏双不变防御框架，包含两个阶段：1)数据平衡阶段，使用基于扩散模型的数据重采样策略缓解模型偏差；2)双对抗自元防御阶段，引入度量对抗训练方法和最远负样本扩展软化技术，以及对抗增强的自元机制实现双重泛化。

### 主要发现

实验证明，所提出的方法显著优于现有的最先进防御方法。

### 结论

该框架有效解决了行人ReID中的对抗防御挑战，提高了模型对对抗攻击的鲁棒性。

### 翻译

行人重识别(ReID)是许多实际应用如行人轨迹跟踪中的基础任务。然而，先进的基于深度学习的ReID模型极易受到对抗攻击，对行人图像的微小扰动会导致完全错误的预测，构成重大安全威胁。尽管已提出许多针对分类任务的对抗防御策略，但扩展到度量学习任务如行人ReID的研究相对较少。此外，现有的行人ReID防御未能解决对抗性鲁棒ReID的固有挑战。本文将行人ReID中对抗防御的挑战系统识别为两个关键问题：模型偏差和复合泛化要求。为解决这些问题，我们提出了一种由两个主要阶段组成的去偏双不变防御框架。在数据平衡阶段，我们使用基于扩散模型的数据重采样策略缓解模型偏差，促进训练数据的公平性和多样性。在双对抗自元防御阶段，我们引入了一种新颖的度量对抗训练方法，结合最远负样本扩展软化技术，克服因缺少分类器导致的鲁棒性退化。此外，我们还引入了对抗增强的自元机制，实现对未见身份和未见攻击类型的双重泛化。实验证明，我们的方法显著优于现有的最先进防御方法。


### 论文摘要

Person re-identification (ReID) is a fundamental task in many real-world applications such as pedestrian trajectory tracking. However, advanced deep learning-based ReID models are highly susceptible to adversarial attacks, where imperceptible perturbations to pedestrian images can cause entirely incorrect predictions, posing significant security threats. Although numerous adversarial defense strategies have been proposed for classification tasks, their extension to metric learning tasks such as person ReID remains relatively unexplored. Moreover, the several existing defenses for person ReID fail to address the inherent unique challenges of adversarially robust ReID. In this paper, we systematically identify the challenges of adversarial defense in person ReID into two key issues: model bias and composite generalization requirements. To address them, we propose a debiased dual-invariant defense framework composed of two main phases. In the data balancing phase, we mitigate model bias using a diffusion-model-based data resampling strategy that promotes fairness and diversity in training data. In the bi-adversarial self-meta defense phase, we introduce a novel metric adversarial training approach incorporating farthest negative extension softening to overcome the robustness degradation caused by the absence of classifier. Additionally, we introduce an adversarially-enhanced self-meta mechanism to achieve dual-generalization for both unseen identities and unseen attack types. Experiments demonstrate that our method significantly outperforms existing state-of-the-art defenses.

---

## 258. vMFCoOp: Towards Equilibrium on a Unified Hyperspherical Manifold for Prompting Biomedical VLMs

**论文链接:** [http://arxiv.org/abs/2511.09540v2](http://arxiv.org/abs/2511.09540v2)

**作者:** Minye Shao, Sihan Guo, Xinrun Li, Xingyu Miao, Haoran Duan, Yang Long

**发布时间:** 2025-11-12

**备注:** Accepted as an Oral Presentation at AAAI 2026 Main Technical Track (this version is not peer-reviewed; it is the extended version)

### GPT解析

### 总结

本研究提出了vMFCoOp框架，通过在超球流形上估计von Mises-Fisher分布，并利用统一语义锚点对齐不同模型间的语义偏差，解决了生物医学视觉语言模型中的语义不对齐和模态差距问题，实现了更强大的少样本分类性能。

### 背景

大型语言模型引导的上下文优化(CoOp)为生物医学CLIP模型的适应提供了可扩展方案，但面临LLMs与CLIP变体间语义不对齐、缺乏跨基础模型家族的可扩展性，以及传统欧几里得空间优化无法建模统一表示和应用局部几何约束等问题，导致复杂生物医学成像中的模态差距被放大，少样本适应不稳定。

### 目的

开发一个框架，解决语义不对齐问题，实现强大的生物医学提示和卓越的少样本分类，提高模型在多种医疗数据集、成像模态和解剖区域上的性能。

### 方法

提出vMFCoOp框架，在共享的超球流形上反向估计von Mises-Fisher分布，通过统一语义锚点对齐任意大型语言模型和CLIP主干之间的语义偏差，基于三个互补约束条件进行优化。

### 主要发现

vMFCoOp在14个医疗数据集、12种医学成像模态和13个解剖区域上展现出一致的改进，在准确性、泛化性和临床适用性方面超越了现有最先进方法。

### 结论

该研究为生物医学视觉语言模型的提示学习提供了新思路，未来将持续扩展以涵盖更多下游应用，相关资源已在GitHub平台分享。

### 翻译

近期在大型语言模型蒸馏的医学语义先验引导下的上下文优化进展，为适应基于生物医学CLIP的视觉语言模型提供了一种可扩展的替代方案，避免了手动提示工程和完全微调。然而，由于训练语料库和模型架构的差异，此背景下的提示学习面临LLMs与CLIP变体之间的语义不对齐挑战；它还缺乏在持续演进的基础模型家族中的可扩展性。更关键的是，通过传统欧几里得空间优化的成对多模态对齐无法建模统一表示或应用局部几何约束，这往往会放大复杂生物医学成像中的模态差距，并使少样本适应不稳定。在这项工作中，我们提出了vMFCoOp，一个在共享超球流形上反向估计von Mises-Fisher分布的框架，通过统一语义锚点对齐任意LLMs和CLIP主干之间的语义偏差，以实现强大的生物医学提示和卓越的少样本分类。基于三个互补约束，vMFCoOp在14个医疗数据集、12种医学成像模态和13个解剖区域上展现出一致的改进，在准确性、泛化性和临床适用性方面优于最先进方法。这项工作旨在持续扩展以涵盖更多下游应用，相关资源将通过https://github.com/VinyehShaw/UniEqui分享。


### 论文摘要

Recent advances in context optimization (CoOp) guided by large language model (LLM)-distilled medical semantic priors offer a scalable alternative to manual prompt engineering and full fine-tuning for adapting biomedical CLIP-based vision-language models (VLMs). However, prompt learning in this context is challenged by semantic misalignment between LLMs and CLIP variants due to divergent training corpora and model architectures; it further lacks scalability across continuously evolving families of foundation models. More critically, pairwise multimodal alignment via conventional Euclidean-space optimization lacks the capacity to model unified representations or apply localized geometric constraints, which tends to amplify modality gaps in complex biomedical imaging and destabilize few-shot adaptation. In this work, we propose vMFCoOp, a framework that inversely estimates von Mises-Fisher (vMF) distributions on a shared Hyperspherical Manifold, aligning semantic biases between arbitrary LLMs and CLIP backbones via Unified Semantic Anchors to achieve robust biomedical prompting and superior few-shot classification. Grounded in three complementary constraints, vMFCoOp demonstrates consistent improvements across 14 medical datasets, 12 medical imaging modalities, and 13 anatomical regions, outperforming state-of-the-art methods in accuracy, generalization, and clinical applicability. This work aims to continuously expand to encompass more downstream applications, and the corresponding resources are intended to be shared through https://github.com/VinyehShaw/UniEqui.

---

## 259. How Can We Effectively Use LLMs for Phishing Detection?: Evaluating the Effectiveness of Large Language Model-based Phishing Detection Models

**论文链接:** [http://arxiv.org/abs/2511.09606v1](http://arxiv.org/abs/2511.09606v1)

**作者:** Fujiao Ji, Doowon Kim

**发布时间:** 2025-11-12

### GPT解析

### 总结

该研究探讨了大型语言模型(LLMs)在钓鱼网站检测中的应用效果，通过评估不同输入模态、温度设置和提示工程策略，发现商业LLMs表现优于开源模型，截图输入对品牌识别效果最佳，建议使用截图输入和零温度设置以提高检测准确率。

### 背景

传统深度学习钓鱼检测器存在对未见网站泛化能力差和缺乏可解释性的局限性，而大型语言模型(LLMs)作为一种新兴的钓鱼检测机制，其效果尚未得到充分探索。

### 目的

研究如何有效利用LLMs进行钓鱼检测（包括目标品牌识别），并考察输入模态（截图、标志、HTML和URL）、温度设置和提示工程策略对检测效果的影响。

### 方法

使用包含19,131个真实世界钓鱼网站和243个良性网站的数据集，评估7个LLMs（两个商业模型GPT 4.1和Gemini 2.0 flash，以及五个开源模型Qwen、Llama、Janus、DeepSeek-VL2和R1），并与两个深度学习基线模型（PhishIntention和Phishpedia）进行比较。

### 主要发现

商业LLMs在钓鱼检测中通常优于开源模型，而DL模型在良性样本上表现更好；截图输入对品牌识别效果最佳，商业LLMs达到93-95%的准确率，开源模型特别是Qwen达到92%；同时使用多种输入模态或一次性提示并不一致提高性能，可能降低结果；较高的温度值会降低性能。

### 结论

建议使用截图输入和零温度设置来最大化基于LLMs的检测器的准确率，当截图信息不足时，HTML可作为辅助上下文。

### 翻译

大型语言模型(LLMs)已成为一种有前景的钓鱼检测机制，解决了传统深度学习检测器的局限性，包括对未见网站的泛化能力差和缺乏可解释性。然而，LLMs在钓鱼检测中的有效性尚未得到探索。本研究通过考察输入模态（截图、标志、HTML和URL）、温度设置和提示工程策略的影响，研究如何有效利用LLMs进行钓鱼检测（包括目标品牌识别）。使用包含19,131个真实世界钓鱼网站和243个良性网站的数据集，我们评估了七个LLMs——两个商业模型（GPT 4.1和Gemini 2.0 flash）和五个开源模型（Qwen、Llama、Janus、DeepSeek-VL2和R1），以及两个深度学习(DL)基线模型（PhishIntention和Phishpedia）。我们的研究结果表明，商业LLMs在钓鱼检测中通常优于开源模型，而DL模型在良性样本上表现更好。对于品牌识别，截图输入实现最佳结果，商业LLMs达到93-95%的准确率，开源模型特别是Qwen达到92%。然而，同时使用多种输入模态或应用一次性提示并不一致地提高性能，可能降低结果。此外，较高的温度值会降低性能。基于这些结果，我们建议使用截图输入和零温度来最大化基于LLMs的检测器的准确率，当截图信息不足时，HTML可作为辅助上下文。


### 论文摘要

Large language models (LLMs) have emerged as a promising phishing detection mechanism, addressing the limitations of traditional deep learning-based detectors, including poor generalization to previously unseen websites and a lack of interpretability. However, LLMs' effectiveness for phishing detection remains unexplored. This study investigates how to effectively leverage LLMs for phishing detection (including target brand identification) by examining the impact of input modalities (screenshots, logos, HTML, and URLs), temperature settings, and prompt engineering strategies. Using a dataset of 19,131 real-world phishing websites and 243 benign sites, we evaluate seven LLMs -- two commercial models (GPT 4.1 and Gemini 2.0 flash) and five open-source models (Qwen, Llama, Janus, DeepSeek-VL2, and R1) -- alongside two deep learning (DL)-based baselines (PhishIntention and Phishpedia).   Our findings reveal that commercial LLMs generally outperform open-source models in phishing detection, while DL models demonstrate better performance on benign samples. For brand identification, screenshot inputs achieve optimal results, with commercial LLMs reaching 93-95% accuracy and open-source models, particularly Qwen, achieving up to 92%. However, incorporating multiple input modalities simultaneously or applying one-shot prompts does not consistently enhance performance and may degrade results. Furthermore, higher temperature values reduce performance. Based on these results, we recommend using screenshot inputs with zero temperature to maximize accuracy for LLM-based detectors with HTML serving as auxiliary context when screenshot information is insufficient.

---

## 260. AutoSynth: Automated Workflow Optimization for High-Quality Synthetic Dataset Generation via Monte Carlo Tree Search

**论文链接:** [http://arxiv.org/abs/2511.09488v1](http://arxiv.org/abs/2511.09488v1)

**作者:** Shuzhen Bi, Chang Song, Siyu Song, Jinze Lv, Jian Chen, Xinyun Wang, Aimin Zhou, Hao Hao

**发布时间:** 2025-11-12

### GPT解析

### 总结

本文介绍了AutoSynth框架，用于自动化大型语言模型(LLMs)的监督微调工作流发现和优化，无需参考数据集。该框架通过蒙特卡洛树搜索和一种新颖的无数据集混合奖励机制来解决冷启动问题，特别适用于主观性、开放性任务。

### 背景

大型语言模型针对特定任务的监督微调需要高质量数据集，但人工整理成本过高。合成数据生成具有可扩展性，但依赖于复杂的多阶段工作流，包括提示工程和模型编排。现有的自动化工作流方法面临冷启动问题：需要标记数据集进行奖励建模，这对于没有客观事实依据的主观性、开放性任务尤其成问题。

### 目的

开发一个无需参考数据集即可自动发现和优化工作流的框架，解决数据中心AI中的冷启动问题，为主观性LLM任务提供可扩展、经济高效的方法。

### 方法

AutoSynth框架通过将问题重新定义为蒙特卡洛树搜索来自动化工作流发现和优化，由一种新颖的无数据集混合奖励引导。这种奖励通过两个'LLM作为评判者'组件实现元学习：一个组件使用动态生成的任务特定指标评估样本质量，另一个组件评估工作流代码和提示质量。

### 主要发现

在主观性教育任务上的实验表明，虽然专家设计的工作流在人类偏好率上更高(96-99%的胜率，而AutoSynth为40-51%)，但在AutoSynth生成数据上训练的模型显著优于基线(40-51%对比2-5%)，并在某些指标上匹配或超过专家设计的工作流，表明发现了超越人类直觉的质量维度。这些成果同时将人类努力从5-7小时减少到仅30分钟(减少90%以上)。

### 结论

AutoSynth解决了数据中心AI中的冷启动问题，为主观性LLM任务提供了一种可扩展、经济高效的方法。

### 翻译

大型语言模型针对特定任务的监督微调需要高质量数据集，但人工整理成本过高。合成数据生成具有可扩展性，但其有效性依赖于复杂的多阶段工作流，包括提示工程和模型编排。现有的自动化工作流方法面临冷启动问题：它们需要标记数据集进行奖励建模，这对于没有客观事实依据的主观性、开放性任务尤其成问题。我们引入了AutoSynth，这是一个无需参考数据集即可自动发现和优化工作流的框架，通过将问题重新定义为由一种新颖的无数据集混合奖励引导的蒙特卡洛树搜索来实现。这种奖励通过两个'LLM作为评判者'组件实现元学习：一个组件使用动态生成的任务特定指标评估样本质量，另一个组件评估工作流代码和提示质量。在主观性教育任务上的实验表明，虽然专家设计的工作流在人类偏好率上更高(96-99%的胜率对比AutoSynth的40-51%)，但在AutoSynth生成数据上训练的模型显著优于基线(40-51%对比2-5%)，并在某些指标上匹配或超过专家设计的工作流，表明发现了超越人类直觉的质量维度。这些成果同时将人类努力从5-7小时减少到仅30分钟(减少90%以上)。AutoSynth解决了数据中心AI中的冷启动问题，为主观性LLM任务提供了一种可扩展、经济高效的方法。代码：https://github.com/bisz9918-maker/AutoSynth。


### 论文摘要

Supervised fine-tuning (SFT) of large language models (LLMs) for specialized tasks requires high-quality datasets, but manual curation is prohibitively expensive. Synthetic data generation offers scalability, but its effectiveness relies on complex, multi-stage workflows, integrating prompt engineering and model orchestration. Existing automated workflow methods face a cold start problem: they require labeled datasets for reward modeling, which is especially problematic for subjective, open-ended tasks with no objective ground truth. We introduce AutoSynth, a framework that automates workflow discovery and optimization without reference datasets by reframing the problem as a Monte Carlo Tree Search guided by a novel dataset-free hybrid reward. This reward enables meta-learning through two LLM-as-judge components: one evaluates sample quality using dynamically generated task-specific metrics, and another assesses workflow code and prompt quality. Experiments on subjective educational tasks show that while expert-designed workflows achieve higher human preference rates (96-99% win rates vs. AutoSynth's 40-51%), models trained on AutoSynth-generated data dramatically outperform baselines (40-51% vs. 2-5%) and match or surpass expert workflows on certain metrics, suggesting discovery of quality dimensions beyond human intuition. These results are achieved while reducing human effort from 5-7 hours to just 30 minutes (>90% reduction). AutoSynth tackles the cold start issue in data-centric AI, offering a scalable, cost-effective method for subjective LLM tasks. Code: https://github.com/bisz9918-maker/AutoSynth.

---

## 261. LLM-Guided Dynamic-UMAP for Personalized Federated Graph Learning

**论文链接:** [http://arxiv.org/abs/2511.09438v1](http://arxiv.org/abs/2511.09438v1)

**作者:** Sai Puppala, Ismail Hossain, Md Jahangir Alam, Tanzim Ahad, Sajedul Talukder

**发布时间:** 2025-11-12

### GPT解析

### 总结

提出了一种在个性化与隐私约束下使用大型语言模型辅助图机器学习的方法，结合数据增强、提示调整和上下文学习，支持低资源环境下的节点分类和链接预测。

### 背景

图机器学习在个性化与隐私约束下面临挑战，需要新的方法来有效利用大型语言模型的能力。

### 目的

开发一种方法，在保护隐私的同时，利用大型语言模型提升图机器学习性能，特别是在低资源设置下的节点分类和链接预测任务。

### 方法

结合稀疏图数据增强、提示和指令调整、上下文学习提供少量图推理信号，参数化客户特定图嵌入的动态UMAP流形，在贝叶斯变分目标内实现个性化联邦学习，并通过跨模态正则化对齐语言模型潜在表示与图结构。

### 主要发现

提出了变体聚合过程的收敛论证，基于矩量会计的差分隐私威胁模型，并展示了在知识图谱补全、推荐式链接预测和引用及产品图等应用中的效果。

### 结论

讨论了评估LLM辅助图机器学习的基准测试考虑事项，为未来研究提供了方向。

### 翻译

我们提出了一种在个性化与隐私约束下使用大型语言模型辅助图机器学习的方法。该方法结合了稀疏图的数据增强、提示和指令调整以使基础模型适应图任务，以及上下文学习以提供少量图推理信号。这些信号参数化了客户特定图嵌入的动态UMAP流形，位于个性化联邦学习的贝叶斯变分目标内。该方法支持低资源设置下的节点分类和链接预测，并通过跨模态正则化使语言模型潜在表示与图结构对齐。我们概述了变体聚合过程的收敛论证，描述了基于矩量会计的差分隐私威胁模型，并展示了在知识图谱补全、推荐式链接预测以及引用和产品图中的应用。我们还讨论了用于基准测试LLM辅助图机器学习的评估考虑事项。


### 论文摘要

We propose a method that uses large language models to assist graph machine learning under personalization and privacy constraints. The approach combines data augmentation for sparse graphs, prompt and instruction tuning to adapt foundation models to graph tasks, and in-context learning to supply few-shot graph reasoning signals. These signals parameterize a Dynamic UMAP manifold of client-specific graph embeddings inside a Bayesian variational objective for personalized federated learning. The method supports node classification and link prediction in low-resource settings and aligns language model latent representations with graph structure via a cross-modal regularizer. We outline a convergence argument for the variational aggregation procedure, describe a differential privacy threat model based on a moments accountant, and present applications to knowledge graph completion, recommendation-style link prediction, and citation and product graphs. We also discuss evaluation considerations for benchmarking LLM-assisted graph machine learning.

---

## 262. A cross-modal pre-training framework with video data for improving performance and generalization of distributed acoustic sensing

**论文链接:** [http://arxiv.org/abs/2511.09342v1](http://arxiv.org/abs/2511.09342v1)

**作者:** Junyi Duan, Jiageng Chen, Zuyuan He

**发布时间:** 2025-11-12

### GPT解析

### 总结

本文提出了一种增强的DAS信号处理框架，结合短时傅里叶变换和视频到DAS跨模态预训练，解决了现有DAS-MAE模型在频率分析和训练数据限制方面的不足，显著提升了模型性能和泛化能力。

### 背景

光纤分布式声学传感(DAS)已成为关键的物联网传感技术，有广泛的工业应用。然而，DAS信号的二维时空形态分析具有挑战性，传统方法表现不佳，而深度学习方法更为适合。

### 目的

克服DAS-MAE模型在时序主导的DAS数据频率分析中的局限性，解决有效训练数据不足的问题，提升模型性能和泛化能力。

### 方法

提出一个增强框架，结合短时傅里叶变换(STFT)进行显式的时频特征提取，并采用视频到DAS的跨模态预训练来缓解数据限制。通过无标签重建任务学习高级表征，如事件分类。

### 主要发现

实验结果表明，新方法在少样本分类中达到0.1%的错误率，比DAS-MAE相对提高90.9%；在外部损伤预防应用中识别错误率为4.7%，比从头训练提高75.4%。

### 结论

该研究是首个视频到DAS跨模态预训练工作，通过连接计算机视觉和分布式传感领域扩展了可用训练资源，增强了DAS性能和泛化能力，促进了DAS在多样化工业场景的部署，推动了工业物联网传感的跨模态表征学习。

### 翻译

光纤分布式声学传感(DAS)已成为关键的物联网传感技术，具有广泛的工业应用。然而，DAS信号的二维时空形态呈现分析挑战，传统方法表现次优，而深度学习方法更为适合。尽管我们之前的工作DAS掩码自编码器(DAS-MAE)建立了无标签的先进性能和泛化能力，但在时序主导的DAS数据频率分析中并不令人满意。此外，有效训练数据的限制无法满足DAS-MAE中Transformer架构的大量数据需求。为克服这些局限，我们提出了一种增强框架，结合短时傅里叶变换(STFT)进行显式的时频特征提取，并开创性地采用视频到DAS的跨模态预训练来缓解数据限制。该方法通过无标签重建任务学习高级表征(如事件分类)。实验结果表明了变革性的改进：少样本分类错误率为0.1%(比DAS-MAE相对提高90.9%)，在外部损伤预防应用中识别错误率为4.7%(比从头训练提高75.4%)。作为首个开创视频到DAS跨模态预训练的工作，通过连接计算机视觉和分布式传感领域扩展了可用训练资源。增强的性能和泛化能力促进了DAS在多样化工业场景的部署，同时推动了工业物联网传感的跨模态表征学习。


### 论文摘要

Fiber-optic distributed acoustic sensing (DAS) has emerged as a critical Internet-of-Things (IoT) sensing technology with broad industrial applications. However, the two-dimensional spatial-temporal morphology of DAS signals presents analytical challenges where conventional methods prove suboptimal, while being well-suited for deep learning approaches. Although our previous work, DAS Masked Autoencoder (DAS-MAE), established state-of-the-art performance and generalization without labels, it is not satisfactory in frequency analysis in temporal-dominated DAS data. Moreover, the limitation of effective training data fails to address the substantial data requirements inherent to Transformer architectures in DAS-MAE. To overcome these limitations, we present an enhanced framework incorporating short-time Fourier transform (STFT) for explicit temporal-frequency feature extraction and pioneering video-to-DAS cross-modal pre-training to mitigate data constraints. This approach learns high-level representations (e.g., event classification) through label-free reconstruction tasks. Experimental results demonstrate transformative improvements: 0.1% error rate in few-shot classification (90.9% relative improvement over DAS-MAE) and 4.7% recognition error in external damage prevention applications (75.4% improvement over from-scratch training). As the first work to pioneer video-to-DAS cross-modal pre-training, available training resources are expanded by bridging computer vision and distributed sensing areas. The enhanced performance and generalization facilitate DAS deployment across diverse industrial scenarios while advancing cross-modal representation learning for industrial IoT sensing.

---

## 263. Iterated Population Based Training with Task-Agnostic Restarts

**论文链接:** [http://arxiv.org/abs/2511.09190v1](http://arxiv.org/abs/2511.09190v1)

**作者:** Alexander Chebykin, Tanja Alderliesten, Peter A. N. Bosman

**发布时间:** 2025-11-12

### GPT解析

### 总结

该论文提出了一种名为迭代基于种群的训练(IPBT)的新方法，作为PBT的变体，能够自动调整超参数更新间隔，通过重启动机制和时变贝叶斯优化来重新初始化超参数，在多种任务上表现出色。

### 背景

超参数优化可以减轻神经网络超参数调优的负担。基于种群的训练算法通过在权重优化过程中动态调整超参数而高效。研究表明，所有PBT变体中超参数更新之间的步数是一个重要的元超参数，但缺乏有效设置该值的方法或直觉。

### 目的

开发一种能够自动调整超参数更新间隔这一元超参数的方法，解决PBT算法中这一关键参数难以设置的问题。

### 方法

提出迭代基于种群的训练(IPBT)，一种新颖的PBT变体，通过任务无关的方式重用权重信息的重启动机制，并利用时变贝叶斯优化来重新初始化超参数。

### 主要发现

在8个图像分类和强化学习任务上的评估表明，IPBT平均匹配或优于5种先前的PBT变体和其他超参数优化算法（包括随机搜索、ASHA、SMAC3），无需增加预算或更改任何超参数。

### 结论

IPBT是一种有效的超参数优化方法，能够自动调整超参数更新间隔，在各种任务上表现出色，且不需要额外的计算资源或参数调整。

### 翻译

超参数优化可以减轻神经网络超参数调优的负担。来自基于种群的训练家族的HPO算法通过在权重优化的每几步动态调整超参数而高效。最近的研究结果表明，所有PBT变体中超参数更新之间的步数是一个重要的元超参数，可以显著影响它们的性能。然而，目前还没有有效设置该值的方法或直觉。我们引入了迭代基于种群的训练，这是一种新颖的PBT变体，它通过任务无关的方式重用权重信息的重启动机制，并利用时变贝叶斯优化来重新初始化超参数，从而自动调整这一超参数。在8个图像分类和强化学习任务上的评估表明，平均而言，我们的算法匹配或优于5种先前的PBT变体和其他HPO算法，无需增加预算或对其超参数进行任何更改。源代码可在指定网址获取。


### 论文摘要

Hyperparameter Optimization (HPO) can lift the burden of tuning hyperparameters (HPs) of neural networks. HPO algorithms from the Population Based Training (PBT) family are efficient thanks to dynamically adjusting HPs every few steps of the weight optimization. Recent results indicate that the number of steps between HP updates is an important meta-HP of all PBT variants that can substantially affect their performance. Yet, no method or intuition is available for efficiently setting its value. We introduce Iterated Population Based Training (IPBT), a novel PBT variant that automatically adjusts this HP via restarts that reuse weight information in a task-agnostic way and leverage time-varying Bayesian optimization to reinitialize HPs. Evaluation on 8 image classification and reinforcement learning tasks shows that, on average, our algorithm matches or outperforms 5 previous PBT variants and other HPO algorithms (random search, ASHA, SMAC3), without requiring a budget increase or any changes to its HPs. The source code is available at https://github.com/AwesomeLemon/IPBT.

---

## 264. Zero-Order Sharpness-Aware Minimization

**论文链接:** [http://arxiv.org/abs/2511.09156v1](http://arxiv.org/abs/2511.09156v1)

**作者:** Yao Fu, Yihang Jin, Chunxia Zhang, Junmin Liu, Haishan Ye

**发布时间:** 2025-11-12

### GPT解析

### 总结

ZOSA是一种创新的优化框架，结合零阶优化和锐度感知最小化，有效解决了传统提示调优方法计算密集的问题，在少样本学习任务中表现出色。

### 背景

提示学习已成为在有限数据条件下使大型语言模型适应特定任务的关键方法，但传统基于梯度的提示调优方法计算密集，效率面临挑战。

### 目的

引入ZOSA优化框架，提高提示调优的效率和效果，为资源有限环境中的基于提示的学习提供实用解决方案。

### 方法

ZOSA结合零阶优化和锐度感知最小化，使用Rademacher扰动向量估计梯度而不需要反向传播，通过融入锐度感知原则针对损失景观中的平坦最小值提高泛化能力，并采用由损失变化引导的自适应学习率确保稳定收敛。

### 主要发现

在少样本学习任务（如文本分类和自然语言推理）上的实验表明，ZOSA显著优于现有方法。

### 结论

ZOSA凭借其理论基础和计算效率，为资源有限环境中的基于提示的学习提供了实用的解决方案。

### 翻译

提示学习已成为一种关键方法，用于在有限数据条件下使大型语言模型适应特定任务。然而，用于调整提示的传统基于梯度的优化方法计算密集，对效率提出了挑战。我们引入了ZOSA（零阶锐度感知最小化），这是一种新的优化框架，将零阶优化与锐度感知最小化相结合，以增强提示调优。ZOSA使用Rademacher扰动向量来估计梯度，无需反向传播。通过融入锐度感知原则，它针对损失景观中的平坦最小值，提高泛化能力。一个由损失变化引导的自适应学习率进一步确保了稳定的收敛。在少样本学习任务（如文本分类和自然语言推理）上的实验表明，ZOSA显著优于现有方法。凭借其理论基础和计算效率，ZOSA为资源有限环境中的基于提示的学习提供了实用的解决方案。


### 论文摘要

Prompt learning has become a key method for adapting large language models to specific tasks with limited data. However, traditional gradient-based optimization methods for tuning prompts are computationally intensive, posing challenges for efficiency. We introduce ZOSA (Zero-Order Sharpness-Aware Minimization), a novel optimization framework that integrates zero-order optimization with sharpness-aware minimization to enhance prompt tuning. ZOSA employs Rademacher perturbation vectors to estimate gradients without requiring backpropagation. By incorporating sharpness-aware principles, it targets flat minima in the loss landscape, improving generalization. An adaptive learning rate, guided by loss variability, further ensures stable convergence. Experiments on few-shot learning tasks, such as text classification and natural language inference, show that ZOSA significantly outperforms existing methods. With its theoretical foundation and computational efficiency, ZOSA offers a practical solution for prompt-based learning in resource-limited settings.

---

## 265. Fairness-Aware Few-Shot Learning for Audio-Visual Stress Detection

**论文链接:** [http://arxiv.org/abs/2511.09039v1](http://arxiv.org/abs/2511.09039v1)

**作者:** Anushka Sanjay Shelke, Aditya Sneh, Arya Adyasha, Haroon R. Lone

**发布时间:** 2025-11-12

### GPT解析

### 总结

该研究提出了FairM2S框架，用于解决AI压力检测中的性别偏见问题，同时发布了SAVSD数据集，为心理健康AI提供了公平且可扩展的小样本压力检测方案。

### 背景

AI驱动的压力检测在心理健康护理的公平性方面至关重要，然而现有模型经常表现出性别偏见，特别是在数据稀缺的情况下。

### 目的

开发一个公平感知的压力检测框架，有效减轻性别偏见，特别是在数据稀缺的场景中实现公平的压力检测。

### 方法

提出了FairM2S，一个基于视听数据的公平感知元学习框架，在元训练和适应阶段整合均衡机会约束，采用对抗性梯度掩码和公平约束的元更新来减轻偏见。

### 主要发现

FairM2S实现了78.1%的准确率，将平等机会降低到0.06，显著提升了公平性；同时发布了带有性别标注的SAVSD智能手机数据集，支持低资源环境下的公平性研究。

### 结论

FairM2S成为心理健康AI中公平和可扩展的小样本压力检测的最先进方法，研究团队已公开数据和代码以促进该领域发展。

### 翻译

AI驱动的压力检测中的公平性对公平的心理健康护理至关重要，然而现有模型经常表现出性别偏见，特别是在数据稀缺的情况下。为此，我们提出了FairM2S，一个用于压力检测的公平感知元学习框架，利用视听数据。FairM2S在元训练和适应阶段都整合了均衡机会约束，采用对抗性梯度掩码和公平约束的元更新来有效减轻偏见。与五个最先进的基线相比，FairM2S实现了78.1%的准确率，同时将平等机会降低到0.06，显示出显著的公平性提升。我们还发布了SAVSD数据集，这是一个带有性别标注的智能手机采集的数据集，旨在支持低资源、真实世界环境中的公平性研究。这些贡献共同使FairM2S成为心理健康AI中公平和可扩展的小样本压力检测的最先进方法。我们随本文公开发布了我们的数据集和FairM2S。


### 论文摘要

Fairness in AI-driven stress detection is critical for equitable mental healthcare, yet existing models frequently exhibit gender bias, particularly in data-scarce scenarios. To address this, we propose FairM2S, a fairness-aware meta-learning framework for stress detection leveraging audio-visual data. FairM2S integrates Equalized Odds constraints during both meta-training and adaptation phases, employing adversarial gradient masking and fairness-constrained meta-updates to effectively mitigate bias. Evaluated against five state-of-the-art baselines, FairM2S achieves 78.1% accuracy while reducing the Equal Opportunity to 0.06, demonstrating substantial fairness gains. We also release SAVSD, a smartphone-captured dataset with gender annotations, designed to support fairness research in low-resource, real-world contexts. Together, these contributions position FairM2S as a state-of-the-art approach for equitable and scalable few-shot stress detection in mental health AI. We release our dataset and FairM2S publicly with this paper.

---

## 266. Boosting Adversarial Transferability via Ensemble Non-Attention

**论文链接:** [http://arxiv.org/abs/2511.08937v2](http://arxiv.org/abs/2511.08937v2)

**作者:** Yipeng Zou, Qin Liu, Jie Wu, Yu Peng, Guo Chen, Hui Zhou, Guanghui Ye

**发布时间:** 2025-11-12

**备注:** 16 pages, 11 figures, accepted by AAAI 2026

### GPT解析

### 总结

本研究提出了一种名为NAMEA的新型集成攻击方法，通过整合集成模型中非注意区域的梯度，显著提升了跨架构对抗样本的可迁移性，在ImageNet数据集上超越了现有最先进方法。

### 背景

集成攻击通过结合不同架构代理模型的输出来提高对抗样本的可迁移性，但以往研究在跨异构模型架构迁移时表现不佳。主要原因是异构代理模型的梯度更新方向差异大，难以在充分利用单个模型的同时减少集成模型的梯度方差。

### 目的

设计一种新型集成攻击方法，解决异构模型间梯度差异大的问题，提高跨架构对抗样本的可迁移性。

### 方法

提出NAMEA方法，首次将集成模型非注意区域的梯度整合到迭代梯度优化过程中。该方法基于异构模型注意区域差异大而非注意区域可能互补的观察，通过元学习将来自注意区域和非注意区域的梯度进行融合。

### 主要发现

在ImageNet数据集上的实验表明，NAMEA比最先进的集成攻击AdaEA和SMER分别平均高出15.0%和9.6%的性能。研究首次证明了集成非注意区域在提升跨架构可迁移性方面的潜力。

### 结论

NAMEA为发起集成攻击提供了新思路，通过整合非注意区域的梯度信息，有效解决了异构模型间梯度差异大的问题，显著提高了对抗样本的可迁移性。

### 翻译

集成攻击结合了不同架构代理模型的输出，可以与各种基于梯度的攻击结合，提高对抗样本的可迁移性。然而，先前的研究表明，在跨异构模型架构迁移时，攻击性能不理想。主要原因是异构代理模型的梯度更新方向差异很大，难以在充分利用单个模型的同时减少集成模型的梯度方差。为了应对这一挑战，我们设计了一种名为NAMEA的新型集成攻击，首次将集成模型非注意区域的梯度整合到迭代梯度优化过程中。我们的设计受到了异构模型的注意区域差异很大的观察启发，因此ViTs的非注意区域很可能是CNNs的关注点，反之亦然。因此，我们融合了来自集成模型注意区域和非注意区域的梯度，以融合CNNs和ViTs的迁移信息。具体来说，我们开创了一种将非注意区域的梯度与注意区域的梯度解耦的新方法，并通过元学习来合并梯度。在ImageNet数据集上的经验评估表明，NAMEA比最先进的集成攻击AdaEA和SMER分别平均高出15.0%和9.6%。这项工作是首次探索集成非注意区域在提升跨架构可迁移性方面的潜力，为发起集成攻击提供了新见解。


### 论文摘要

Ensemble attacks integrate the outputs of surrogate models with diverse architectures, which can be combined with various gradient-based attacks to improve adversarial transferability. However, previous work shows unsatisfactory attack performance when transferring across heterogeneous model architectures. The main reason is that the gradient update directions of heterogeneous surrogate models differ widely, making it hard to reduce the gradient variance of ensemble models while making the best of individual model. To tackle this challenge, we design a novel ensemble attack, NAMEA, which for the first time integrates the gradients from the non-attention areas of ensemble models into the iterative gradient optimization process. Our design is inspired by the observation that the attention areas of heterogeneous models vary sharply, thus the non-attention areas of ViTs are likely to be the focus of CNNs and vice versa. Therefore, we merge the gradients respectively from the attention and non-attention areas of ensemble models so as to fuse the transfer information of CNNs and ViTs. Specifically, we pioneer a new way of decoupling the gradients of non-attention areas from those of attention areas, while merging gradients by meta-learning. Empirical evaluations on ImageNet dataset indicate that NAMEA outperforms AdaEA and SMER, the state-of-the-art ensemble attacks by an average of 15.0% and 9.6%, respectively. This work is the first attempt to explore the power of ensemble non-attention in boosting cross-architecture transferability, providing new insights into launching ensemble attacks.

---

## 267. Vector Symbolic Algebras for the Abstraction and Reasoning Corpus

**论文链接:** [http://arxiv.org/abs/2511.08747v1](http://arxiv.org/abs/2511.08747v1)

**作者:** Isaac Joffe, Chris Eliasmith

**发布时间:** 2025-11-11

### GPT解析

### 总结

该论文提出了一种基于向量符号代数(VSAs)的认知上合理的ARC-AGI求解器，结合系统1直觉与系统2推理，通过面向对象的程序合成方法解决抽象推理问题。

### 背景

ARC-AGI是一个生成性的、少样本的流体智能基准测试，人类能够轻松解决，但即使是先进的人工智能系统也极难解决。

### 目的

开发一种认知上合理的ARC-AGI求解器，受神经科学到心理学的人类智能建模方法启发。

### 方法

使用基于向量符号代数(VSAs)的神经符号方法，将系统1直觉与系统2推理整合起来。求解器通过面向对象的程序合成工作，利用VSAs表示抽象对象、引导解决方案搜索和实现样本高效的神经学习。

### 主要发现

初步结果显示，求解器在ARC-AGI-1-Train上得分为10.8%，在ARC-AGI-1-Eval上得分为3.0%。在更简单的基准测试上表现良好，在Sort-of-ARC上得分为94.5%，在1D-ARC上得分为83.1%，后者以极小的计算成本优于GPT-4。

### 结论

作者认为他们的方法是独特的，是第一个将VSAs应用于ARC-AGI，并开发了迄今为止认知上最合理的ARC-AGI求解器。代码已公开。

### 翻译

ARC-AGI(用于通用人工智能的抽象与推理语料库)是一个生成性的、少样本的流体智能基准测试。虽然人类能够轻松解决ARC-AGI，但它对即使是先进的人工智能系统来说仍然极其困难。受从神经科学到心理学的人类智能建模方法启发，我们提出了一种认知上合理的ARC-AGI求解器。我们的求解器使用基于向量符号代数(VSAs)的神经符号方法，将系统1直觉与系统2推理整合到一个高效且可解释的过程中。我们的求解器通过面向对象的程序合成工作，利用VSAs表示抽象对象、引导解决方案搜索，并实现样本高效的神经学习。初步结果表明成功，我们的求解器在ARC-AGI-1-Train上得分为10.8%，在ARC-AGI-1-Eval上得分为3.0%。此外，我们的求解器在更简单的基准测试上表现良好，在Sort-of-ARC上得分为94.5%，在1D-ARC上得分为83.1%——后者以极小的计算成本优于GPT-4。重要的是，我们的方法是独特的；我们相信我们是第一个将VSAs应用于ARC-AGI，并开发了迄今为止认知上最合理的ARC-AGI求解器。我们的代码可在https://github.com/ijoffe/ARC-VSA-2025获取。


### 论文摘要

The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) is a generative, few-shot fluid intelligence benchmark. Although humans effortlessly solve ARC-AGI, it remains extremely difficult for even the most advanced artificial intelligence systems. Inspired by methods for modelling human intelligence spanning neuroscience to psychology, we propose a cognitively plausible ARC-AGI solver. Our solver integrates System 1 intuitions with System 2 reasoning in an efficient and interpretable process using neurosymbolic methods based on Vector Symbolic Algebras (VSAs). Our solver works by object-centric program synthesis, leveraging VSAs to represent abstract objects, guide solution search, and enable sample-efficient neural learning. Preliminary results indicate success, with our solver scoring 10.8% on ARC-AGI-1-Train and 3.0% on ARC-AGI-1-Eval. Additionally, our solver performs well on simpler benchmarks, scoring 94.5% on Sort-of-ARC and 83.1% on 1D-ARC -- the latter outperforming GPT-4 at a tiny fraction of the computational cost. Importantly, our approach is unique; we believe we are the first to apply VSAs to ARC-AGI and have developed the most cognitively plausible ARC-AGI solver yet. Our code is available at: https://github.com/ijoffe/ARC-VSA-2025.

---

## 268. SRE-Llama -- Fine-Tuned Meta's Llama LLM, Federated Learning, Blockchain and NFT Enabled Site Reliability Engineering(SRE) Platform for Communication and Networking Software Services

**论文链接:** [http://arxiv.org/abs/2511.08282v1](http://arxiv.org/abs/2511.08282v1)

**作者:** Eranga Bandara, Safdar H. Bouk, Sachin Shetty, Ravi Mukkamala, Abdul Rahman, Peter Foytik, Ross Gore, Xueping Liang, Ng Wee Keong, Kasun De Zoysa

**发布时间:** 2025-11-11

### GPT解析

### 总结

本文提出了一种名为SRE-Llama的新型站点可靠性工程平台，该平台结合生成式AI、联邦学习、区块链和非同质化代币技术，旨在解决云原生环境中SLI/SLO定义的挑战，实现监控、SLI/SLO生成和警报管理的自动化与简化。

### 背景

软件服务对于可靠通信和网络至关重要，站点可靠性工程(SRE)对于确保系统在云原生环境中保持可靠和良好性能非常重要。SRE使用Prometheus和Grafana等工具来监控系统指标，定义关键的SLI和SLO以维持高服务标准。然而，许多开发人员往往缺乏对这些工具以及定义适当SLI和SLO复杂性的深入理解。

### 目的

为了弥补开发人员对SRE工具和SLI/SLO定义理解不足的差距，提出一种自动化和简化监控、SLI/SLO生成和警报管理过程的平台，为开发人员提供便捷高效的访问。

### 方法

系统通过捕获云原生服务的指标并将其存储在时间序列数据库中工作。利用存储的数据，平台采用联邦学习模型识别不同服务和SLO的最相关和最有影响力的SLI指标，解决数据隐私问题。随后，使用微调的Meta的Llama-3大语言模型基于这些识别的SLI指标智能生成SLI、SLO、错误预算和相关警报机制。平台将生成的SLI和SLO编码为NFT对象并存储在区块链上，提供不可变的记录保存和便于验证审计。平台的自动化由区块链智能合约管理。

### 主要发现

SRE-Llama平台原型已经通过一个定制的Open5GS 5G核心的用例实现，验证了平台的有效性和实用性。

### 结论

结合生成式AI、联邦学习、区块链和非同质化代币技术的SRE-Llama平台能够有效解决云原生环境中开发人员面临的SRE实施挑战，特别是SLI/SLO定义方面的困难，提供自动化、高效且安全的SRE解决方案。

### 翻译

软件服务对于可靠通信和网络至关重要；因此，站点可靠性工程(SRE)对于确保这些系统在云原生环境中保持可靠和良好性能非常重要。SRE利用Prometheus和Grafana等工具来监控系统指标，定义关键的SLI和SLO以维持高服务标准。然而，一个重大挑战是许多开发人员通常缺乏对这些工具以及定义适当SLI和SLO复杂性的深入理解。为了弥补这一差距，我们提出了一种名为SRE-Llama的新型SRE平台，通过生成式AI、联邦学习、区块链和非同质化代币(NFT)增强。该平台旨在自动化和简化监控、SLI/SLO生成和警报管理的过程，为开发人员提供便捷高效的访问。系统通过捕获云原生服务的指标并将其存储在时间序列数据库(如Prometheus和Mimir)中工作。利用这些存储的数据，我们的平台采用联邦学习模型来识别不同服务和SLO的最相关和最有影响力的SLI指标，解决数据隐私问题。随后，采用微调的Meta的Llama-3大语言模型，基于这些识别的SLI指标智能地生成SLI、SLO、错误预算和相关警报机制。我们平台的一个独特方面是将生成的SLI和SLO编码为NFT对象，然后存储在区块链上。此功能提供了不可变的记录保存，并便于轻松验证和审计SRE指标和目标。所提出平台的自动化由区块链智能合约管理。SRE-Llama平台原型已经通过一个定制的Open5GS 5G核心的用例实现。


### 论文摘要

Software services are crucial for reliable communication and networking; therefore, Site Reliability Engineering (SRE) is important to ensure these systems stay reliable and perform well in cloud-native environments. SRE leverages tools like Prometheus and Grafana to monitor system metrics, defining critical Service Level Indicators (SLIs) and Service Level Objectives (SLOs) for maintaining high service standards. However, a significant challenge arises as many developers often lack in-depth understanding of these tools and the intricacies involved in defining appropriate SLIs and SLOs. To bridge this gap, we propose a novel SRE platform, called SRE-Llama, enhanced by Generative-AI, Federated Learning, Blockchain, and Non-Fungible Tokens (NFTs). This platform aims to automate and simplify the process of monitoring, SLI/SLO generation, and alert management, offering ease in accessibility and efficy for developers. The system operates by capturing metrics from cloud-native services and storing them in a time-series database, like Prometheus and Mimir. Utilizing this stored data, our platform employs Federated Learning models to identify the most relevant and impactful SLI metrics for different services and SLOs, addressing concerns around data privacy. Subsequently, fine-tuned Meta's Llama-3 LLM is adopted to intelligently generate SLIs, SLOs, error budgets, and associated alerting mechanisms based on these identified SLI metrics. A unique aspect of our platform is the encoding of generated SLIs and SLOs as NFT objects, which are then stored on a Blockchain. This feature provides immutable record-keeping and facilitates easy verification and auditing of the SRE metrics and objectives. The automation of the proposed platform is governed by the blockchain smart contracts. The proposed SRE-Llama platform prototype has been implemented with a use case featuring a customized Open5GS 5G Core.

---

## 269. Where and What Matters: Sensitivity-Aware Task Vectors for Many-Shot Multimodal In-Context Learning

**论文链接:** [http://arxiv.org/abs/2511.08246v1](http://arxiv.org/abs/2511.08246v1)

**作者:** Ziyu Ma, Chenhui Gou, Yiming Hu, Yong Wang, Xiangxiang Chu, Bohan Zhuang, Jianfei Cai

**发布时间:** 2025-11-11

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种感知敏感性的任务向量插入框架(STV)，解决了大型多模态模型在多样本设置下面临的上下文长度有限和推理成本高的问题。通过识别激活变化的敏感位置并应用强化学习选择最合适的任务向量插入，STV在各种多模态模型和任务上都显示出优于先前方法的性能。

### 背景

大型多模态模型(LMMs)在上下文学习方面显示出潜力，但扩展到多样本设置仍然困难，原因是上下文长度有限和推理成本高。

### 目的

解决基于任务向量的方法中忽略插入位置重要性或难以确定各位置合适值的问题，提高多样本设置下的模型性能。

### 方法

提出STV框架，利用查询-上下文对间激活变化的一致结构模式确定插入位置，构建预聚类激活库，并通过强化学习选择最合适的任务向量插入。

### 主要发现

激活变化在查询-上下文对间表现出一致的结构模式，为确定插入位置提供了可靠线索；STV在各种多模态模型和任务上都有效且泛化能力强。

### 结论

STV框架能够有效确定在哪里以及插入什么任务向量，显著提高了大型多模态模型在多样本设置下的性能，相比之前的任务向量方法有持续改进。

### 翻译

大型多模态模型(LMMs)在上下文学习(ICL)方面显示出前景，但由于上下文长度有限和高推理成本，扩展到多样本设置仍然困难。为应对这些挑战，已探索了基于任务向量的方法，通过将多样本上下文演示的紧凑表示插入模型激活中。然而，现有的基于任务向量的方法要么忽略了在哪里插入任务向量的重要性，要么难以确定每个位置的合适值。为此，我们提出了一种新颖的感知敏感性的任务向量插入框架(STV)，以确定在哪里以及插入什么。我们的核心洞察是，查询-上下文对之间的激活变化表现出一致的结构模式，为插入提供了可靠的线索。基于识别出的敏感感知位置，我们通过聚类激活值为每个位置构建预聚类激活库，然后应用强化学习选择最合适的插入。我们在多种多模态模型(如Qwen-VL, Idefics-2)和任务(如VizWiz, OK-VQA)上评估了STV，证明了其有效性，并显示出相比之前的基于任务向量的方法有持续改进，具有强大的泛化能力。


### 论文摘要

Large Multimodal Models (LMMs) have shown promising in-context learning (ICL) capabilities, but scaling to many-shot settings remains difficult due to limited context length and high inference cost. To address these challenges, task-vector-based methods have been explored by inserting compact representations of many-shot in-context demonstrations into model activations. However, existing task-vector-based methods either overlook the importance of where to insert task vectors or struggle to determine suitable values for each location. To this end, we propose a novel Sensitivity-aware Task Vector insertion framework (STV) to figure out where and what to insert. Our key insight is that activation deltas across query-context pairs exhibit consistent structural patterns, providing a reliable cue for insertion. Based on the identified sensitive-aware locations, we construct a pre-clustered activation bank for each location by clustering the activation values, and then apply reinforcement learning to choose the most suitable one to insert. We evaluate STV across a range of multimodal models (e.g., Qwen-VL, Idefics-2) and tasks (e.g., VizWiz, OK-VQA), demonstrating its effectiveness and showing consistent improvements over previous task-vector-based methods with strong generalization.

---

## 270. Generalizable Insights for Graph Transformers in Theory and Practice

**论文链接:** [http://arxiv.org/abs/2511.08028v1](http://arxiv.org/abs/2511.08028v1)

**作者:** Timo Stoll, Luis Müller, Christopher Morris

**发布时间:** 2025-11-11

**备注:** Accepted at NeurIPS 2025 as spotlight

### GPT解析

### 总结

该研究提出了通用距离Transformer (GDT)架构，通过大量实验分析了图Transformer在注意力机制和位置编码方面的表示能力，并识别出跨应用、任务和模型规模的一致性设计选择。

### 背景

Graph Transformers (GTs)虽展现出强大的实证性能，但现有架构在注意力机制、位置编码和表达能力方面差异显著，且现有表达能力结果常与特定设计选择绑定，缺乏大规模数据的全面实证验证，导致理论与实践之间存在差距。

### 目的

提出通用距离Transformer (GDT)架构，开发对GDT在注意力和位置编码方面表示能力的细粒度理解，识别在各种应用、任务和模型规模中表现一致的设计选择。

### 方法

提出使用标准注意力机制的GDT架构，融合近年来GT领域的多项进展；进行大规模实验，评估覆盖超过八百万个图、约2.7亿个标记，涉及图像目标检测、分子属性预测、代码摘要和分布外算法推理等多个领域。

### 主要发现

识别出跨应用、任务和模型规模表现一致的设计选择；在无需微调的少样本迁移设置中展现出强大性能；理论和实践发现可提炼为关于有效GT设计、训练和推理的通用见解。

### 结论

通过综合理论分析和大规模实证验证，该研究提供了超越特定应用领域的通用GT设计见解，弥合了理论与实践之间的差距。

### 翻译

图变换器(GTs)已展现出强大的实证性能，但当前架构在注意力机制、位置编码(PEs)和表达能力方面差异很大。现有的表达能力结果通常与特定的设计选择相关联，且缺乏在大规模数据上的全面实证验证。这导致理论与实践之间存在差距，难以获得超越特定应用领域的通用见解。在此，我们提出了通用距离变换器(GDT)，一种使用标准注意力的GT架构，融合了近年来GT领域的多项进展，并从注意力和PEs角度开发对GDT表示能力的细粒度理解。通过大量实验，我们识别出在各种应用、任务和模型规模中表现一致的设计选择，展示了在无需微调的少样本迁移设置中的强大性能。我们的评估覆盖了八个多领域的多百万个图，约2.7亿个标记，包括基于图像的目标检测、分子属性预测、代码摘要和分布外算法推理。我们将理论和实践发现提炼为关于有效GT设计、训练和推理的几个通用见解。


### 论文摘要

Graph Transformers (GTs) have shown strong empirical performance, yet current architectures vary widely in their use of attention mechanisms, positional embeddings (PEs), and expressivity. Existing expressivity results are often tied to specific design choices and lack comprehensive empirical validation on large-scale data. This leaves a gap between theory and practice, preventing generalizable insights that exceed particular application domains. Here, we propose the Generalized-Distance Transformer (GDT), a GT architecture using standard attention that incorporates many advancements for GTs from recent years, and develop a fine-grained understanding of the GDT's representation power in terms of attention and PEs. Through extensive experiments, we identify design choices that consistently perform well across various applications, tasks, and model scales, demonstrating strong performance in a few-shot transfer setting without fine-tuning. Our evaluation covers over eight million graphs with roughly 270M tokens across diverse domains, including image-based object detection, molecular property prediction, code summarization, and out-of-distribution algorithmic reasoning. We distill our theoretical and practical findings into several generalizable insights about effective GT design, training, and inference.

---

## 271. From Noise to Latent: Generating Gaussian Latents for INR-Based Image Compression

**论文链接:** [http://arxiv.org/abs/2511.08009v1](http://arxiv.org/abs/2511.08009v1)

**作者:** Chaoyi Lin, Yaojun Wu, Yue Li, Junru Li, Kai Zhang, Li Zhang

**发布时间:** 2025-11-11

### GPT解析

### 总结

本文提出了一种基于高斯潜在生成的新型图像压缩方法，通过从多尺度高斯噪声张量重建图像特定潜在变量，消除了传输潜在代码的需要，同时保留了基于潜在变量的优势，在Kodak和CLIC数据集上实现了具有竞争力的率失真性能。

### 背景

基于隐式神经表示(INR)的图像压缩方法通过过度拟合图像特定的潜在代码展示了有竞争力的性能，但由于缺乏表现力强的潜在表示，仍然不如端到端(E2E)压缩方法。而E2E方法依赖于传输潜在代码并需要复杂的熵模型，导致了解码复杂度的增加。

### 目的

探索从高斯噪声直接生成潜在变量的可能性，以结合INR和E2E压缩方法的优势，避免各自的缺点。

### 方法

提出一种新的图像压缩范式，使用共享随机种子确定性生成多尺度高斯噪声张量，通过高斯参数预测(GPP)模块估计分布参数，利用重参数化技巧实现一次性潜在生成，最后通过合成网络重建图像。

### 主要发现

该方法消除了传输潜在代码的需要，同时保留了基于潜在变量的优势，在Kodak和CLIC数据集上实现了具有竞争力的率失真性能。

### 结论

据作者所知，这是第一个探索用于学习图像压缩的高斯潜在生成的工作，为图像压缩领域提供了新的思路。

### 翻译

最近的基于隐式神经表示(INR)的图像压缩方法通过过度拟合图像特定的潜在代码展示了有竞争力的性能。然而，由于缺乏表现力强的潜在表示，它们仍然不如端到端(E2E)压缩方法。另一方面，E2E方法依赖于传输潜在代码并需要复杂的熵模型，导致了解码复杂度的增加。受E2E编解码器中标准化策略的启发，其中潜在变量被转换为高斯噪声以展示空间冗余的去除，我们探索相反的方向：直接从高斯噪声生成潜在变量。在本文中，我们提出了一种新的图像压缩范式，从多尺度高斯噪声张量重建图像特定的潜在变量，使用共享随机种子确定性生成。高斯参数预测(GPP)模块估计分布参数，通过重参数化技巧实现一次性潜在生成。预测的潜在变量随后通过合成网络传递以重建图像。我们的方法消除了传输潜在代码的需要，同时保留了基于潜在变量的优势，在Kodak和CLIC数据集上实现了具有竞争力的率失真性能。据我们所知，这是第一个探索用于学习图像压缩的高斯潜在生成的工作。


### 论文摘要

Recent implicit neural representation (INR)-based image compression methods have shown competitive performance by overfitting image-specific latent codes. However, they remain inferior to end-to-end (E2E) compression approaches due to the absence of expressive latent representations. On the other hand, E2E methods rely on transmitting latent codes and requiring complex entropy models, leading to increased decoding complexity. Inspired by the normalization strategy in E2E codecs where latents are transformed into Gaussian noise to demonstrate the removal of spatial redundancy, we explore the inverse direction: generating latents directly from Gaussian noise. In this paper, we propose a novel image compression paradigm that reconstructs image-specific latents from a multi-scale Gaussian noise tensor, deterministically generated using a shared random seed. A Gaussian Parameter Prediction (GPP) module estimates the distribution parameters, enabling one-shot latent generation via reparameterization trick. The predicted latent is then passed through a synthesis network to reconstruct the image. Our method eliminates the need to transmit latent codes while preserving latent-based benefits, achieving competitive rate-distortion performance on Kodak and CLIC dataset. To the best of our knowledge, this is the first work to explore Gaussian latent generation for learned image compression.

---

## 272. Terrain Costmap Generation via Scaled Preference Conditioning

**论文链接:** [http://arxiv.org/abs/2511.11529v1](http://arxiv.org/abs/2511.11529v1)

**作者:** Luisa Mao, Garret Warnell, Peter Stone, Joydeep Biswas

**发布时间:** 2025-11-14

### GPT解析

### 总结

SPACER是一种新的地形成本地图生成方法，能够同时实现对新地形的泛化和快速调整相对成本的能力，解决了现有方法的局限性。

### 背景

自主机器人在非道路环境中导航需要高质量的地形成本地图，现有方法要么能快速测试时间调整相对成本（如语义分割方法），要么能泛化到新地形类型（如表示学习方法），但不能同时做到这两点。

### 目的

开发一种既能广泛适应不同地形，又能快速调整相对成本以满足特定任务需求的地形成本地图生成方法。

### 方法

提出SPACER（scaled preference conditioned all-terrain costmap generation）方法，利用合成数据训练实现泛化到新地形，并通过基于用户指定的缩放偏好上下文实现快速测试时间调整相对成本。

### 主要发现

使用大规模 aerial 地图的经验证据表明，SPACER在生成地形导航的成本地图方面优于其他方法，在七个环境中的五个环境中，全局路径规划的遗憾值最低。

### 结论

SPACER成功解决了现有方法在生成地形成本地图时面临的权衡问题，同时实现了对新地形的泛化和快速调整相对成本的能力。

### 翻译

在非道路领域实现成功的自主机器人导航需要能够生成高质量的地形成本地图，这些成本地图既能广泛适应各种地形，又能在测试时快速调整相对成本以满足特定任务需求。现有的成本地图生成方法要么允许快速测试时间调整相对成本（例如语义分割方法），要么能泛化到新的地形类型（例如表示学习方法），但不能同时实现这两种能力。在这项工作中，我们提出了缩放偏好条件全地形成本地图生成（SPACER），这是一种生成地形成本地图的新方法，它利用训练期间的合成数据以很好地泛化到新地形，并通过基于用户指定的缩放偏好上下文来快速测试时间调整相对成本。使用大规模 aerial 地图，我们提供了经验证据，表明 SPACER 在生成地形导航的成本地图方面优于其他方法，在七个环境中的五个环境中，全局路径规划的遗憾值最低。


### 论文摘要

Successful autonomous robot navigation in off-road domains requires the ability to generate high-quality terrain costmaps that are able to both generalize well over a wide variety of terrains and rapidly adapt relative costs at test time to meet mission-specific needs. Existing approaches for costmap generation allow for either rapid test-time adaptation of relative costs (e.g., semantic segmentation methods) or generalization to new terrain types (e.g., representation learning methods), but not both. In this work, we present scaled preference conditioned all-terrain costmap generation (SPACER), a novel approach for generating terrain costmaps that leverages synthetic data during training in order to generalize well to new terrains, and allows for rapid test-time adaptation of relative costs by conditioning on a user-specified scaled preference context. Using large-scale aerial maps, we provide empirical evidence that SPACER outperforms other approaches at generating costmaps for terrain navigation, with the lowest measured regret across varied preferences in five of seven environments for global path planning.

---

## 273. Collaborative Representation Learning for Alignment of Tactile, Language, and Vision Modalities

**论文链接:** [http://arxiv.org/abs/2511.11512v1](http://arxiv.org/abs/2511.11512v1)

**作者:** Yiyun Zhou, Mingjing Xu, Jingwei Shi, Quanjiang Li, Jingyuan Chen

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出TLV-CoRe方法，一种基于CLIP的触觉-语言-视觉协同表征学习方法，以及RSS评估框架，旨在解决触觉传感器标准化不足和跨模态交互不充分的问题。

### 背景

触觉感知为机器人提供丰富且互补的信息，使其能够感知精细的物体属性。然而，现有触觉传感器缺乏标准化，导致冗余特征阻碍跨传感器泛化能力，且现有方法未能充分整合触觉、语言和视觉模态之间的中间交互。

### 目的

解决触觉传感器标准化不足和跨模态交互不充分的问题，提高跨传感器表征学习和跨模态对齐的能力。

### 方法

提出TLV-CoRe方法，引入传感器感知调制器统一不同传感器的触觉特征，采用触觉无关解耦学习分离无关触觉特征，并引入统一桥接适配器增强共享表征空间中的三模态交互；同时提出RSS评估框架，重点关注不同方法间的鲁棒性、协同性和稳定性。

### 主要发现

实验结果表明，TLV-CoRe显著提高了传感器无关的表征学习和跨模态对齐能力，为多模态触觉表征提供了新方向。

### 结论

TLV-CoRe方法有效解决了触觉传感器标准化不足和跨模态交互不充分的问题，通过创新的架构和评估框架，为多模态触觉感知领域提供了新的解决方案。

### 翻译

触觉感知为机器人和语言提供了丰富且互补的信息，使机器人能够感知精细的物体属性。然而，现有的触觉传感器缺乏标准化，导致冗余特征阻碍了跨传感器泛化。此外，现有方法未能充分整合触觉、语言和视觉模态之间的中间交互。为解决这一问题，我们提出了TLV-CoRe，一种基于CLIP的触觉-语言-视觉协同表征学习方法。TLV-CoRe引入了传感器感知调制器来统一不同传感器的触觉特征，并采用触觉无关解耦学习来分离无关的触觉特征。此外，还引入了统一桥接适配器来增强共享表征空间中的三模态交互。为了公平评估触觉模型的有效性，我们进一步提出了RSS评估框架，重点关注不同方法间的鲁棒性、协同性和稳定性。实验结果表明，TLV-CoRe显著提高了传感器无关的表征学习和跨模态对齐能力，为多模态触觉表征提供了新方向。


### 论文摘要

Tactile sensing offers rich and complementary information to vision and language, enabling robots to perceive fine-grained object properties. However, existing tactile sensors lack standardization, leading to redundant features that hinder cross-sensor generalization. Moreover, existing methods fail to fully integrate the intermediate communication among tactile, language, and vision modalities. To address this, we propose TLV-CoRe, a CLIP-based Tactile-Language-Vision Collaborative Representation learning method. TLV-CoRe introduces a Sensor-Aware Modulator to unify tactile features across different sensors and employs tactile-irrelevant decoupled learning to disentangle irrelevant tactile features. Additionally, a Unified Bridging Adapter is introduced to enhance tri-modal interaction within the shared representation space. To fairly evaluate the effectiveness of tactile models, we further propose the RSS evaluation framework, focusing on Robustness, Synergy, and Stability across different methods. Experimental results demonstrate that TLV-CoRe significantly improves sensor-agnostic representation learning and cross-modal alignment, offering a new direction for multimodal tactile representation.

---

## 274. SoK: Security Evaluation of Wi-Fi CSI Biometrics: Attacks, Metrics, and Systemic Weaknesses

**论文链接:** [http://arxiv.org/abs/2511.11381v1](http://arxiv.org/abs/2511.11381v1)

**作者:** Gioliano de Oliveira Braga, Pedro Henrique dos Santos Rocha, Rafael Pimenta de Mattos Paixão, Giovani Hoff da Costa, Gustavo Cavalcanti Morais, Lourenço Alves Pereira Júnior

**发布时间:** 2025-11-14

**备注:** An improved version will be submitted to Euro S&P 2026, and this paper will be updated in the near future

### GPT解析

### 总结

这项研究对Wi-Fi信道状态信息(CSI)生物识别认证进行了系统性的安全分析，揭示了当前研究中的方法论不一致性和安全评估缺陷，提出了统一的评估框架，并指出了未来研究方向。

### 背景

Wi-Fi CSI已被多次提出作为生物识别模态，有报告显示其具有高准确性和操作可行性，但该领域缺乏对其安全特性、对抗鲁棒性和方法一致性的综合理解。

### 目的

从安全角度审视基于CSI的生物识别认证，分析现有工作在不同方面的差异，构建统一评估框架，揭示当前研究中的安全问题和风险，并为严格评估、可重复实验和未来研究提供指导。

### 方法

通过系统知识综述(SoK)方法，分析现有工作在感知基础设施、信号表示、特征管道、学习模型和评估方法方面的差异，构建统一的评估框架进行实证分析，并考察安全相关指标如每类EER、FCS和基尼系数。

### 主要发现

研究发现当前研究存在系统性不一致：依赖总体准确度指标、有限报告FAR/FRR/EER、缺乏每用户风险分析、很少考虑威胁模型或对抗可行性；方法选择显著影响漏洞状况，包括重放、几何模仿和环境扰动等攻击面；安全相关指标能揭示传统报告中隐藏的风险集中。

### 结论

当前CSI生物识别具有明确的安全边界，需要采用更严格的评估方法、可重复的实验设计和考虑安全因素的指标；这项研究为安全社区提供了对Wi-Fi CSI生物识别作为认证原语适用性的结构化、证据驱动的重新评估。

### 翻译

Wi-Fi信道状态信息(CSI)已被反复提议作为生物识别模态，经常有报告显示其高准确性和操作可行性。然而，该领域缺乏对其安全特性、对抗鲁棒性和方法一致性的综合理解。这篇系统知识综述(SoK)从安全角度审视了基于CSI的生物识别认证，分析了现有工作在感知基础设施、信号表示、特征管道、学习模型和评估方法方面的差异。我们的综合分析揭示了系统性不一致：依赖总体准确度指标、有限报告FAR/FRR/EER、缺乏每用户风险分析、很少考虑威胁模型或对抗可行性。我们构建了一个统一的评估框架来实证暴露这些问题，并展示了如何通过安全相关指标（如每类EER、FCS和基尼系数）揭示传统报告实践中隐藏的风险集中。我们的分析强调了具体的攻击面，并展示了方法选择如何显著影响漏洞状况，包括重放、几何模仿和环境扰动。基于这些发现，我们阐述了当前CSI生物识别的安全边界，并为严格评估、可重复实验和未来研究方向提供指导。这篇SoK为安全社区提供了对Wi-Fi CSI生物识别及其作为认证原语适用性的结构化、证据驱动的重新评估。


### 论文摘要

Wi-Fi Channel State Information (CSI) has been repeatedly proposed as a biometric modality, often with reports of high accuracy and operational feasibility. However, the field lacks a consolidated understanding of its security properties, adversarial resilience, and methodological consistency. This Systematization of Knowledge (SoK) examines CSI-based biometric authentication through a security perspective, analyzing how existing work differs across sensing infrastructure, signal representations, feature pipelines, learning models, and evaluation methodologies. Our synthesis reveals systemic inconsistencies: reliance on aggregate accuracy metrics, limited reporting of FAR/FRR/EER, absence of per-user risk analysis, and scarce consideration of threat models or adversarial feasibility. We construct a unified evaluation framework to empirically expose these issues and demonstrate how security-relevant metrics, such as per-class EER, FCS, and the Gini Coefficient, uncover risk concentration that remains hidden under traditional reporting practices. Our analysis highlights concrete attack surfaces and shows how methodological choices materially influence vulnerability profiles, which include replay, geometric mimicry, and environmental perturbation. Based on these findings, we articulate the security boundaries of current CSI biometrics and provide guidelines for rigorous evaluation, reproducible experimentation, and future research directions. This SoK offers the security community a structured, evidence-driven reassessment of Wi-Fi CSI biometrics and their suitability as an authentication primitive.

---

## 275. 论文ID: 2511.11370v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.11370v1.json'

---

## 276. 论文ID: 2511.11305v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.11305v1.json'

---

## 277. SimuFreeMark: A Noise-Simulation-Free Robust Watermarking Against Image Editing

**论文链接:** [http://arxiv.org/abs/2511.11295v1](http://arxiv.org/abs/2511.11295v1)

**作者:** Yichao Tang, Mingyang Li, Di Miao, Sheng Li, Zhenxing Qian, Xinpeng Zhang

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为SimuFreeMark的新型图像水印框架，该框架无需噪声模拟训练，通过利用图像低频分量的稳定性，实现了对各种攻击的强大抵抗力，同时保持高质量的视觉表现。

### 背景

人工智能生成内容（AIGC）的发展迫切需要能够抵抗传统信号处理和新型语义编辑攻击的鲁棒图像水印技术。当前的深度学习方法依赖于手工制作的噪声模拟层进行训练，这限制了它们对不可预见失真的泛化能力。

### 目的

提出一种无需噪声模拟训练的水印框架，克服当前方法的局限性，实现对各种攻击的更强抵抗力。

### 方法

SimuFreeMark利用图像低频分量的固有稳定性，系统性地建立低频分量对广泛攻击具有显著鲁棒性的结论，然后将水印直接嵌入到低频分量的深度特征空间中，利用预训练的变分自编码器（VAE）将水印与结构稳定的图像表示绑定，完全消除了训练过程中对噪声模拟的需求。

### 主要发现

低频分量对广泛攻击具有显著鲁棒性；SimuFreeMark在各种传统和语义攻击上优于最先进的方法；SimuFreeMark保持了优越的视觉质量。

### 结论

SimuFreeMark是一种创新的噪声模拟免费水印框架，通过利用低频分量的稳定性，实现了对各种攻击的强大抵抗力，同时保持了高质量的视觉表现。

### 翻译

人工智能生成内容（AIGC）的进步迫切需要能够抵抗传统信号处理和新型语义编辑攻击的鲁棒图像水印技术。当前的基于深度学习的方法依赖于使用手工制作的噪声模拟层进行训练，这 inherently限制了它们对不可预见失真的泛化能力。在这项工作中，我们提出了SimuFreeMark，一种无噪声模拟的水印框架，它通过利用图像低频分量的固有稳定性来克服这一限制。我们首先系统性地建立了低频分量对广泛攻击具有显著鲁棒性的结论。基于这一基础，SimuFreeMark将水印直接嵌入到低频分量的深度特征空间中，利用预训练的变分自编码器（VAE）将水印与结构稳定的图像表示绑定。这种设计完全消除了训练过程中对噪声模拟的需求。大量实验表明，SimuFreeMark在各种传统和语义攻击上都优于最先进的方法，同时保持优越的视觉质量。


### 论文摘要

The advancement of artificial intelligence generated content (AIGC) has created a pressing need for robust image watermarking that can withstand both conventional signal processing and novel semantic editing attacks. Current deep learning-based methods rely on training with hand-crafted noise simulation layers, which inherently limit their generalization to unforeseen distortions. In this work, we propose $\textbf{SimuFreeMark}$, a noise-$\underline{\text{simu}}$lation-$\underline{\text{free}}$ water$\underline{\text{mark}}$ing framework that circumvents this limitation by exploiting the inherent stability of image low-frequency components. We first systematically establish that low-frequency components exhibit significant robustness against a wide range of attacks. Building on this foundation, SimuFreeMark embeds watermarks directly into the deep feature space of the low-frequency components, leveraging a pre-trained variational autoencoder (VAE) to bind the watermark with structurally stable image representations. This design completely eliminates the need for noise simulation during training. Extensive experiments demonstrate that SimuFreeMark outperforms state-of-the-art methods across a wide range of conventional and semantic attacks, while maintaining superior visual quality.

---

## 278. RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image

**论文链接:** [http://arxiv.org/abs/2511.11289v1](http://arxiv.org/abs/2511.11289v1)

**作者:** Hengfei Wang, Zhongqun Zhang, Yihua Cheng, Hyung Jin Chang

**发布时间:** 2025-11-14

**备注:** AAAI 2026

### GPT解析

### 总结

本研究提出了RTGaze，一种实时且高质量的视线重定向方法，能够生成具有可控眼部运动的逼真人脸图像，解决了现有方法在3D一致性、效率或质量方面的局限性。

### 背景

现有的视线重定向方法通常难以在3D一致性、效率或质量之间取得平衡，这限制了它们的实际应用。

### 目的

开发一种实时且高质量的视线重定向方法，能够生成具有可控眼部运动的逼真人脸图像，同时保持3D一致性。

### 方法

RTGaze通过从人脸图像和视线提示中学习视线可控的面部表示，然后通过神经渲染解码该表示来实现视线重定向。此外，还从预训练的3D人像生成器中提炼人脸几何先验，以提高生成质量。

### 主要发现

RTGaze在效率、重定向准确性和图像质量方面展示了最先进的性能，其系统通过前馈网络实现实时3D感知视线重定向，处理速度约为每张图像0.06秒，比之前最先进的3D感知方法快800倍。

### 结论

RTGaze成功解决了现有视线重定向方法在3D一致性、效率和质量方面的局限性，实现了实时、高质量的视线重定向，具有广泛的实际应用潜力。

### 翻译

视线重定向方法旨在生成具有可控眼部运动的逼真人脸图像。然而，现有方法通常在3D一致性、效率或质量方面存在困难，限制了它们的实际应用。在这项工作中，我们提出了RTGaze，一种实时且高质量的视线重定向方法。我们的方法从人脸图像和视线提示中学习视线可控的面部表示，然后通过神经渲染解码该表示以实现视线重定向。此外，我们从预训练的3D人像生成器中提炼人脸几何先验，以提高生成质量。我们对RTGaze进行了定性和定量评估，证明其在多个数据集的效率、重定向准确性和图像质量方面达到了最先进的性能。我们的系统通过前馈网络（约每张图像0.06秒）实现了实时、3D感知的视线重定向，比之前最先进的3D感知方法快800倍。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决实时3D感知的视线重定向问题。现有方法要么缺乏3D一致性(2D方法)，要么效率低下(3D方法)。这个问题在虚拟现实、数字人和CG电影制作等领域非常重要，因为视线是人类表达注意力和意图的关键面部特征，高质量的视线重定向能显著提升这些应用的用户体验。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有2D和3D方法的局限性，认识到需要平衡3D一致性和实时性能。他们借鉴了神经辐射场(NeRF)的3D结构学习，使用三平面表示(来自EG3D)，并采用DeepLabV3和Vision Transformer作为特征提取网络。同时，他们创新性地从预训练的3D肖像生成器中提炼面部几何先验，以提高生成质量。通过结合现有技术的优点并针对性地解决实时性问题，设计了RTGaze方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是学习一个视线可控的面部表示，并通过神经渲染进行解码，实现实时3D感知的视线重定向。整体流程：1)使用两个编码器分别提取面部图像的高频和低频特征；2)通过交叉注意力机制将视线提示注入高频特征；3)融合注入视线特征后的高频特征与低频特征；4)将融合后的表示输入三平面解码器生成3D面部表示；5)通过神经渲染生成最终的视线重定向图像。训练过程结合了重构损失、蒸馏损失和感知损失。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)实时且高质量的3D感知视线重定向模型；2)视线可控的面部表示学习模块，包含双编码器和视线注入机制；3)3D面部先验提炼方法。相比之前工作的不同：相比2D方法，RTGaze具有更好的3D一致性；相比3D方法(如GazeNeRF)，RTGaze实现实时性能(61ms/图像)，比之前方法快800倍，同时保持更高的图像质量和重定向精度；RTGaze避免了GAN反转过程，只需单张图像作为输入。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RTGaze提出了一种实时3D感知的视线重定向方法，通过视线可控的面部表示和3D面部先验提炼，实现了比现有方法更快、更高质量的视线重定向，每张图像处理时间仅需61毫秒。'}


### 论文摘要

Gaze redirection methods aim to generate realistic human face images with controllable eye movement. However, recent methods often struggle with 3D consistency, efficiency, or quality, limiting their practical applications. In this work, we propose RTGaze, a real-time and high-quality gaze redirection method. Our approach learns a gaze-controllable facial representation from face images and gaze prompts, then decodes this representation via neural rendering for gaze redirection. Additionally, we distill face geometric priors from a pretrained 3D portrait generator to enhance generation quality. We evaluate RTGaze both qualitatively and quantitatively, demonstrating state-of-the-art performance in efficiency, redirection accuracy, and image quality across multiple datasets. Our system achieves real-time, 3D-aware gaze redirection with a feedforward network (~0.06 sec/image), making it 800x faster than the previous state-of-the-art 3D-aware methods.

---

## 279. Sparse Methods for Vector Embeddings of TPC Data

**论文链接:** [http://arxiv.org/abs/2511.11221v1](http://arxiv.org/abs/2511.11221v1)

**作者:** Tyler Wheeler, Michelle P. Kuchera, Raghuram Ramanujan, Ryan Krupp, Chris Wrede, Saiprasad Ravishankar, Connor L. Cross, Hoi Yan Ian Heung, Andrew J. Jones, Benjamin Votaw

**发布时间:** 2025-11-14

**备注:** NeurIPS Machine Learning and the Physical Sciences Workshop 2025

### GPT解析

### 总结

研究探索了稀疏卷积网络在时间投影室(TPC)数据上的表示学习应用，发现稀疏ResNet架构能有效提供事件的结构化向量嵌入，即使在随机权重设置下也有用。通过在GADGET II TPC和AT-TPC数据上的测试，验证了该方法在不同探测器间的通用性和有效性。

### 背景

时间投影室(TPCs)是多功能探测器，可在电离介质中重建带电粒子轨迹，适用于广泛的核物理实验。研究使用了两种探测器：GADGET II TPC(针对低能β延迟粒子衰变测量优化)和AT-TPC(用于研究逆运动学核反应)。

### 目的

探索稀疏卷积网络在TPC数据上的表示学习应用，研究稀疏ResNet架构是否能提供有用的事件结构化向量嵌入，并验证该方法在不同探测器间的通用性。

### 方法

将GADGET II TPC的原始垫层信号表示为稀疏张量，使用Minkowski Engine ResNet模型进行训练，并通过简单的物理动机二元分类任务对架构进行预训练。然后将相同的编码器应用于AT-TPC数据，测试跨探测器性能。

### 主要发现

1) 即使权重随机设置的稀疏ResNet架构也能提供有用的事件结构化向量嵌入；2) 在简单物理动机二元分类任务上预训练可进一步提高嵌入质量；3) 未训练的稀疏ResNet模型能为AT-TPC数据提供有用嵌入；4) 在GADGET数据上训练的模型可改善AT-TPC数据的嵌入质量。

### 结论

稀疏卷积技术作为不同TPC实验中表示学习的通用工具有很大潜力，能够有效捕捉事件结构并跨探测器迁移学习。

### 翻译

时间投影室(TPCs)是多功能探测器，可在电离介质中重建带电粒子轨迹， enabling sensitive measurements across a wide range of nuclear physics experiments. We explore sparse convolutional networks for representation learning on TPC data, finding that a sparse ResNet architecture, even with randomly set weights, provides useful structured vector embeddings of events. Pre-training this architecture on a simple physics-motivated binary classification task further improves the embedding quality. Using data from the GAseous Detector with GErmanium Tagging (GADGET) II TPC, a detector optimized for measuring low-energy $β$-delayed particle decays, we represent raw pad-level signals as sparse tensors, train Minkowski Engine ResNet models, and probe the resulting event-level embeddings which reveal rich event structure. As a cross-detector test, we embed data from the Active-Target TPC (AT-TPC) -- a detector designed for nuclear reaction studies in inverse kinematics -- using the same encoder. We find that even an untrained sparse ResNet model provides useful embeddings of AT-TPC data, and we observe improvements when the model is trained on GADGET data. Together, these results highlight the potential of sparse convolutional techniques as a general tool for representation learning in diverse TPC experiments.

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决时间投影室(TPC)数据的高效处理和通用分析问题。TPC在核物理实验中广泛应用，能重建带电粒子轨迹，但产生大量高维稀疏数据，传统方法依赖特定探测器处理流程，限制了通用工具开发。这个问题很重要，因为随着实验数据量增长，需要更高效、通用的处理方法来促进科学发现，提高研究效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到稀疏卷积神经网络可高效处理TPC数据并提取有意义事件表示，特别关注模型在不同TPC系统间的迁移能力，以发展通用TPC基础模型。他们借鉴了Minkowski Engine框架实现稀疏卷积网络，采用ResNet14架构，参考了随机权重网络也能产生有效嵌入的研究，使用线性探针测试嵌入信息，并应用PCA可视化嵌入空间结构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用稀疏卷积神经网络处理TPC数据，即使随机初始化的权重也能产生有用的结构化事件嵌入，通过简单物理任务预训练可进一步提高嵌入质量，且能在不同TPC探测器间有效迁移。流程包括：1)将原始pad级信号表示为稀疏张量；2)使用稀疏ResNet14架构处理数据；3)在GADGET II数据上训练二元分类模型；4)从网络倒数第二层提取潜在向量；5)用线性探针和PCA评估嵌入质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次探索稀疏卷积网络在TPC数据表示学习中的应用；证明随机权重ResNet也能提供有用嵌入；通过简单物理任务预训练提高嵌入质量；展示模型在不同几何形状和实验目标的TPC系统间的有效迁移能力。相比之前工作，本文方法具有更好的通用性，展示了良好的迁移学习能力，比密集卷积网络更高效处理稀疏数据，且专注于物理领域特定应用和跨探测器迁移。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文证明了稀疏卷积网络方法能够学习可迁移的TPC数据表示，即使在随机初始化情况下也能提供有用嵌入，而通过简单物理任务的预训练可以显著提高嵌入质量，为开发通用的TPC基础模型奠定了基础。'}


### 论文摘要

Time Projection Chambers (TPCs) are versatile detectors that reconstruct charged-particle tracks in an ionizing medium, enabling sensitive measurements across a wide range of nuclear physics experiments. We explore sparse convolutional networks for representation learning on TPC data, finding that a sparse ResNet architecture, even with randomly set weights, provides useful structured vector embeddings of events. Pre-training this architecture on a simple physics-motivated binary classification task further improves the embedding quality. Using data from the GAseous Detector with GErmanium Tagging (GADGET) II TPC, a detector optimized for measuring low-energy $β$-delayed particle decays, we represent raw pad-level signals as sparse tensors, train Minkowski Engine ResNet models, and probe the resulting event-level embeddings which reveal rich event structure. As a cross-detector test, we embed data from the Active-Target TPC (AT-TPC) -- a detector designed for nuclear reaction studies in inverse kinematics -- using the same encoder. We find that even an untrained sparse ResNet model provides useful embeddings of AT-TPC data, and we observe improvements when the model is trained on GADGET data. Together, these results highlight the potential of sparse convolutional techniques as a general tool for representation learning in diverse TPC experiments.

---

## 280. LoRaCompass: Robust Reinforcement Learning to Efficiently Search for a LoRa Tag

**论文链接:** [http://arxiv.org/abs/2511.11190v1](http://arxiv.org/abs/2511.11190v1)

**作者:** Tianlang He, Zhongming Lin, Tianrui Jiang, S. -H. Gary Chan

**发布时间:** 2025-11-14

### GPT解析

### 总结

LoRaCompass是一种基于强化学习的模型，用于在未知环境中高效定位LoRa标签，解决了现有方法对域偏移和信号波动敏感的问题

### 背景

LoRa协议因其远距离和低功耗特性，被越来越多地用于精神障碍人士等走失风险人群的标签中，需要有效的定位方法

### 目的

研究移动传感器如何通过接收信号强度指示器引导，以最少的移动次数在一般未知环境中定位周期性广播的LoRa标签

### 方法

提出LoRaCompass，通过空间感知特征提取器和策略蒸馏损失函数从接收信号强度学习稳健的空间表示，并引入基于上置信界的探索函数

### 主要发现

LoRaCompass在超过80平方公里的不同未见环境中验证，成功率达到90%以上，比现有方法提高40%，搜索路径长度与初始距离成线性比例

### 结论

LoRaCompass能够实现稳健高效的LoRa标签搜索，解决了域偏移和信号波动导致的定位不准确问题

### 翻译

LoRa协议以其远距离和低功耗而闻名，已越来越多地被精神障碍人士和其他有走失风险的人士佩戴的标签所采用。我们研究了移动传感器在一般未知环境中通过接收信号强度指示器引导，以最少的移动次数定位周期性广播的LoRa标签的顺序决策过程。虽然现有方法利用强化学习进行搜索，但它们仍然容易受到域偏移和信号波动的影响，导致级联决策错误，最终造成显著的定位不准确。为了弥合这一差距，我们提出了LoRaCompass，一种为稳健高效搜索LoRa标签而设计的强化学习模型。为了在域偏移和信号波动下进行利用，LoRaCompass通过空间感知特征提取器和策略蒸馏损失函数从接收信号强度学习稳健的空间表示，以最大化向标签移动的概率。它进一步引入了受上置信界启发的探索函数，引导传感器以增加的信心朝向标签移动。我们在覆盖超过80平方公里的不同未见环境中的地面和无人机辅助场景中验证了LoRaCompass。它在100米范围内定位标签的成功率很高，比现有方法提高40%，并且搜索路径长度与初始距离成线性比例，效率很高。


### 论文摘要

The Long-Range (LoRa) protocol, known for its extensive range and low power, has increasingly been adopted in tags worn by mentally incapacitated persons (MIPs) and others at risk of going missing. We study the sequential decision-making process for a mobile sensor to locate a periodically broadcasting LoRa tag with the fewest moves (hops) in general, unknown environments, guided by the received signal strength indicator (RSSI). While existing methods leverage reinforcement learning for search, they remain vulnerable to domain shift and signal fluctuation, resulting in cascading decision errors that culminate in substantial localization inaccuracies. To bridge this gap, we propose LoRaCompass, a reinforcement learning model designed to achieve robust and efficient search for a LoRa tag. For exploitation under domain shift and signal fluctuation, LoRaCompass learns a robust spatial representation from RSSI to maximize the probability of moving closer to a tag, via a spatially-aware feature extractor and a policy distillation loss function. It further introduces an exploration function inspired by the upper confidence bound (UCB) that guides the sensor toward the tag with increasing confidence. We have validated LoRaCompass in ground-based and drone-assisted scenarios within diverse unseen environments covering an area of over 80km^2. It has demonstrated high success rate (>90%) in locating the tag within 100m proximity (a 40% improvement over existing methods) and high efficiency with a search path length (in hops) that scales linearly with the initial distance.

---

## 281. Improving Continual Learning of Knowledge Graph Embeddings via Informed Initialization

**论文链接:** [http://arxiv.org/abs/2511.11118v1](http://arxiv.org/abs/2511.11118v1)

**作者:** Gerard Pons, Besim Bilalli, Anna Queralt

**发布时间:** 2025-11-14

### GPT解析

### 总结

论文提出了一种新颖的嵌入初始化策略，用于知识图谱嵌入的持续学习，该方法利用知识图谱模式和先前学习的嵌入为新实体提供初始表示，提高了预测性能，增强了知识保留，并加速了知识获取。

### 背景

许多知识图谱经常更新，迫使知识图谱嵌入适应这些变化。持续学习技术需要为新实体嵌入进行初始化，同时更新旧的嵌入，这对最终嵌入的准确性和训练时间有重要影响，特别是对于相对较小且频繁的更新。

### 目的

解决知识图谱嵌入在持续学习过程中的初始化问题，提高新知识获取效率，减少灾难性遗忘，并加速学习过程。

### 方法

提出一种新颖的 informed embedding initialization 策略，利用知识图谱模式和先前学习的嵌入，基于实体所属的类别为新实体获取初始表示，该方法可以无缝集成到现有的KGE持续学习方法中。

### 主要发现

实验分析表明，所提出的初始化策略提高了结果KGE的预测性能，同时增强了知识保留。此外，该方法加速了知识获取，减少了增量学习新嵌入所需的周期数和时间。

### 结论

该方法在各种类型的KGE学习模型中都有益处，是一种有效的知识图谱嵌入持续学习策略。

### 翻译

许多知识图谱经常更新，迫使它们的知识图谱嵌入适应这些变化。为了解决这个问题，KGE的持续学习技术为新实体嵌入进行初始化，同时更新旧的嵌入。这些方法中必要的一步是嵌入的初始化，作为KGE学习过程的输入，这对最终嵌入的准确性以及训练所需的时间可能有重要影响。这对于相对较小且频繁的更新尤其重要。我们提出了一种新颖的 informed embedding initialization 策略，可以无缝集成到现有的KGE持续学习方法中，增强新知识的获取同时减少灾难性遗忘。具体来说，利用KG模式和先前学习的嵌入，基于实体所属的类别为新实体获取初始表示。我们大量的实验分析表明，所提出的初始化策略提高了结果KGE的预测性能，同时增强了知识保留。此外，我们的方法加速了知识获取，减少了增量学习新嵌入所需的周期数和时间。最后，证明了该方法在各种类型的KGE学习模型中的益处。


### 论文摘要

Many Knowledege Graphs (KGs) are frequently updated, forcing their Knowledge Graph Embeddings (KGEs) to adapt to these changes. To address this problem, continual learning techniques for KGEs incorporate embeddings for new entities while updating the old ones. One necessary step in these methods is the initialization of the embeddings, as an input to the KGE learning process, which can have an important impact in the accuracy of the final embeddings, as well as in the time required to train them. This is especially relevant for relatively small and frequent updates. We propose a novel informed embedding initialization strategy, which can be seamlessly integrated into existing continual learning methods for KGE, that enhances the acquisition of new knowledge while reducing catastrophic forgetting. Specifically, the KG schema and the previously learned embeddings are utilized to obtain initial representations for the new entities, based on the classes the entities belong to. Our extensive experimental analysis shows that the proposed initialization strategy improves the predictive performance of the resulting KGEs, while also enhancing knowledge retention. Furthermore, our approach accelerates knowledge acquisition, reducing the number of epochs, and therefore time, required to incrementally learn new embeddings. Finally, its benefits across various types of KGE learning models are demonstrated.

---

## 282. NP-LoRA: Null Space Projection Unifies Subject and Style in LoRA Fusion

**论文链接:** [http://arxiv.org/abs/2511.11051v1](http://arxiv.org/abs/2511.11051v1)

**作者:** Chuheng Chen, Xiaofei Zhou, Geyuan Zhang, Yong Huang

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为Null Space Projection LoRA (NP-LoRA)的新型LoRA融合框架，通过子空间分离来防止主方向间的结构干扰，从而提高融合质量。

### 背景

Low-Rank Adaptation (LoRA)融合是重用和组合已学习的主题和风格表示的关键技术，用于可控生成而无需昂贵重新训练。然而，现有基于权重合并的方法常导致一个LoRA主导另一个，造成干扰和保真度下降，这种干扰源于单独训练的LoRA占据低秩高维子空间，形成非正交和重叠的表示。

### 目的

分析LoRA内部结构，找到防止LoRA融合过程中结构干扰的方法，提高融合质量和保真度。

### 方法

提出NP-LoRA框架，通过奇异值分解(SVD)提取主要风格方向，将主题LoRA投影到其正交零空间中，并引入软投影机制实现对主题保真度和风格一致性之间权衡的平滑控制。

### 主要发现

LoRA的生成行为主要由低秩子空间中的几个主要方向主导，这些方向在融合过程中应保持免受干扰才能获得最佳效果。

### 结论

实验表明，NP-LoRA在基于DINO和CLIP的指标以及人类和LLM偏好评分等评估方法上一致地优于强基线方法，且可广泛适用于不同骨干网络和LoRA对，无需重新训练。

### 翻译

低秩适应(LoRA)融合已成为一种关键技术，用于重用和组合已学习的主题和风格表示，实现可控生成而无需昂贵的重新训练。然而，现有方法依赖于基于权重的合并，其中一个LoRA通常主导另一个，导致干扰和保真度下降。这种干扰是结构性的：单独训练的LoRA占据低秩高维子空间，导致非正交和重叠的表示。在本工作中，我们分析了LoRA的内部结构，发现它们的生成行为由低秩子空间中的几个主要方向主导，这些方向在融合过程中应保持免受干扰。为此，我们提出了零空间投影LoRA (NP-LoRA)，这是一种基于投影的LoRA融合框架，强制执行子空间分离以防止主方向之间的结构干扰。具体而言，我们首先通过奇异值分解(SVD)提取主要风格方向，然后将主题LoRA投影到其正交零空间中。此外，我们引入了一种软投影机制，能够对主题保真度和风格一致性之间的权衡进行平滑控制。实验表明，NP-LoRA在强基线方法(例如基于DINO和CLIP的指标，以及人类和LLM偏好评分)上一致地提高了融合质量，并且可以广泛适用于各种骨干网络和LoRA对，无需重新训练。


### 论文摘要

Low-Rank Adaptation (LoRA) fusion has emerged as a key technique for reusing and composing learned subject and style representations for controllable generation without costly retraining. However, existing methods rely on weight-based merging, where one LoRA often dominates the other, leading to interference and degraded fidelity. This interference is structural: separately trained LoRAs occupy low-rank high-dimensional subspaces, leading to non-orthogonal and overlapping representations. In this work, we analyze the internal structure of LoRAs and find their generative behavior is dominated by a few principal directions in the low-rank subspace, which should remain free from interference during fusion. To achieve this, we propose Null Space Projection LoRA (NP-LoRA), a projection-based framework for LoRA fusion that enforces subspace separation to prevent structural interference among principal directions. Specifically, we first extract principal style directions via singular value decomposition (SVD) and then project the subject LoRA into its orthogonal null space. Furthermore, we introduce a soft projection mechanism that enables smooth control over the trade-off between subject fidelity and style consistency. Experiments show NP-LoRA consistently improves fusion quality over strong baselines (e.g., DINO and CLIP-based metrics, with human and LLM preference scores), and applies broadly across backbones and LoRA pairs without retraining.

---

## 283. Who Moved My Distribution? Conformal Prediction for Interactive Multi-Agent Systems

**论文链接:** [http://arxiv.org/abs/2511.11567v1](http://arxiv.org/abs/2511.11567v1)

**作者:** Allen Emmanuel Binny, Anushri Dixit

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一种迭代共形预测框架，用于解决不确定性感知预测中的内生分布偏移问题，使代理能够适应周围代理的反应性行为变化，提供概率安全保证，并在仿真实验中表现出色。

### 背景

不确定性感知预测对安全运动规划至关重要，特别是在使用学习模型预测周围代理行为时。共形预测是常用的产生不确定性感知预测区域的统计工具，但现有框架通常假设周围代理是非交互式的。

### 目的

解决内生分布偏移带来的挑战，使不确定性感知的自我代理控制器能够适应周围代理的反应性行为变化。

### 方法

提出迭代共形预测框架，系统性地调整不确定性感知的自我代理控制器以适应内生分布偏移；建立内生分布偏移模型；提供迭代共形预测流程在该分布偏移下收敛的条件。

### 主要发现

在2个和3个代理交互场景的仿真中，该方法实现了碰撞避免而不过度保守，与其他基于共形预测的基线相比，成功率提高最高达9.6%。

### 结论

提出的迭代共形预测框架能够适应反应性非自我代理的演变行为，提供概率安全保证，在实际应用中表现良好。

### 翻译

不确定性感知预测对安全运动规划至关重要，特别是在使用学习模型预测周围代理行为时。共形预测是一种常用于为机器学习模型产生不确定性感知预测区域的统计工具。大多数利用基于共形预测的不确定性预测的现有框架假设周围代理是非交互式的。这是因为闭环系统中，不确定性感知的代理会根据预测不确定性改变行为，而周围代理会对这种变化做出反应，导致我们称之为内生分布偏移的分布偏移。为应对这一挑战，我们引入了迭代共形预测框架，使不确定性感知的自我代理控制器能够系统地适应内生分布偏移。所提出的方法在适应反应性非自我代理的演变行为的同时提供概率安全保证。我们建立了内生分布偏移的模型，并提供了在该分布偏移下迭代共形预测流程收敛的条件。我们在2个和3个代理交互场景的仿真中验证了我们的框架，展示了碰撞避免而不会导致过度保守的行为，与其它基于共形预测的基线相比，总体成功率提高了高达9.6%。


### 论文摘要

Uncertainty-aware prediction is essential for safe motion planning, especially when using learned models to forecast the behavior of surrounding agents. Conformal prediction is a statistical tool often used to produce uncertainty-aware prediction regions for machine learning models. Most existing frameworks utilizing conformal prediction-based uncertainty predictions assume that the surrounding agents are non-interactive. This is because in closed-loop, as uncertainty-aware agents change their behavior to account for prediction uncertainty, the surrounding agents respond to this change, leading to a distribution shift which we call endogenous distribution shift. To address this challenge, we introduce an iterative conformal prediction framework that systematically adapts the uncertainty-aware ego-agent controller to the endogenous distribution shift. The proposed method provides probabilistic safety guarantees while adapting to the evolving behavior of reactive, non-ego agents. We establish a model for the endogenous distribution shift and provide the conditions for the iterative conformal prediction pipeline to converge under such a distribution shift. We validate our framework in simulation for 2- and 3- agent interaction scenarios, demonstrating collision avoidance without resulting in overly conservative behavior and an overall improvement in success rates of up to 9.6% compared to other conformal prediction-based baselines.

---

## 284. Scalable Policy Evaluation with Video World Models

**论文链接:** [http://arxiv.org/abs/2511.11520v1](http://arxiv.org/abs/2511.11520v1)

**作者:** Wei-Cheng Tseng, Jinwei Gu, Qinsheng Zhang, Hanzi Mao, Ming-Yu Liu, Florian Shkurti, Lin Yen-Chen

**发布时间:** 2025-11-14

### GPT解析

### 总结

该论文提出了一种使用动作条件视频生成模型作为可扩展方式来学习用于策略评估的世界模型的方法，通过利用互联网规模的野外在线视频进行预训练，避免了昂贵的配对视频-动作数据集收集，实验表明这种方法在各种指标上都提供了有前景的策略评估方法。

### 背景

机器人操作通用策略的训练已经显示出巨大前景，能够在多样化场景中实现基于语言条件的多任务行为。然而，评估这些策略仍然很困难，因为现实世界测试成本高、耗时且劳动密集，需要频繁环境重置，并存在安全风险。手动创建和填充机器人操作模拟环境需要大量工程工作，且通常存在显著的模拟到现实差距。

### 目的

探索使用动作条件视频生成模型作为可扩展的方式来学习用于策略评估的世界模型，研究如何将动作条件纳入现有的预训练视频生成模型中，并利用互联网规模的野外在线视频进行预训练，减轻对大型配对视频-动作数据集的需求。

### 方法

将动作条件整合到现有的预训练视频生成模型中，利用互联网规模的野外在线视频进行预训练阶段，避免为机器人操作收集昂贵的配对视频-动作数据集。研究数据集多样性、预训练权重和常见失败案例对所提出评估管道的影响。

### 主要发现

实验表明，在各种指标上（包括策略排名和实际策略值与预测策略值之间的相关性），这些模型提供了一种有前景的策略评估方法，可以在不要求现实世界交互的情况下评估策略。

### 结论

动作条件视频生成模型提供了一种有前途的方法，可以在不要求现实世界交互的情况下评估机器人操作策略，这种方法避免了现实世界测试的昂贵、耗时和安全风险问题。

### 翻译

为机器人操作训练通用策略已显示出巨大前景，因为它们能够在多样化场景中实现基于语言条件的多任务行为。然而，评估这些策略仍然很困难，因为现实世界测试成本高、耗时且劳动密集。它还需要频繁的环境重置，并在物理机器人上部署未经证实的策略时存在安全风险。手动创建和填充机器人操作的模拟环境及其资源并没有解决这些问题，主要是因为需要大量的工程工作，并且在物理和渲染方面通常存在显著的模拟到现实差距。在本文中，我们探索使用动作条件视频生成模型作为可扩展的方式来学习用于策略评估的世界模型。我们展示了如何将动作条件整合到现有的预训练视频生成模型中。这允许在预训练阶段利用互联网规模的野外在线视频，并减轻了为机器人操作收集大型配对视频-动作数据集的需求，这些数据集收集成本高昂。我们的论文研究了数据集多样性、预训练权重以及所提出的评估管道的常见失败案例。我们的实验表明，在各种指标上，包括策略排名和实际策略值与预测策略值之间的相关性，这些模型提供了一种有前景的方法，可以在不要求现实世界交互的情况下评估策略。


### 论文摘要

Training generalist policies for robotic manipulation has shown great promise, as they enable language-conditioned, multi-task behaviors across diverse scenarios. However, evaluating these policies remains difficult because real-world testing is expensive, time-consuming, and labor-intensive. It also requires frequent environment resets and carries safety risks when deploying unproven policies on physical robots. Manually creating and populating simulation environments with assets for robotic manipulation has not addressed these issues, primarily due to the significant engineering effort required and the often substantial sim-to-real gap, both in terms of physics and rendering. In this paper, we explore the use of action-conditional video generation models as a scalable way to learn world models for policy evaluation. We demonstrate how to incorporate action conditioning into existing pre-trained video generation models. This allows leveraging internet-scale in-the-wild online videos during the pre-training stage, and alleviates the need for a large dataset of paired video-action data, which is expensive to collect for robotic manipulation. Our paper examines the effect of dataset diversity, pre-trained weight and common failure cases for the proposed evaluation pipeline.Our experiments demonstrate that, across various metrics, including policy ranking and the correlation between actual policy values and predicted policy values, these models offer a promising approach for evaluating policies without requiring real-world interactions.

---

## 285. Higher-order QCD corrections to top-quark pair production in association with a jet

**论文链接:** [http://arxiv.org/abs/2511.11431v1](http://arxiv.org/abs/2511.11431v1)

**作者:** Simon Badger, Matteo Bechetti, Colomba Brancaccio, Michal Czakon, Heribertus Bayu Hartanto, Rene Poncelet, Simone Zoia

**发布时间:** 2025-11-14

**备注:** 7 pages, 1 Table, 4 Figures

### GPT解析

### 总结

本研究首次在量子色动力学的次次领头阶预测了顶夸克对与轻喷注联合产生的各种运动学可观测量，大幅减少了预测中的不确定性，对标准模型研究和新物理搜索具有重要意义。

### 背景

顶夸克对是已知的最重基本粒子，其与轻喷注的联合产生是研究粒子物理标准模型特性的关键过程。同时，它作为对顶夸克质量敏感的信号过程和新物理搜索的背景过程具有重要性。

### 目的

需要高精度预测微分截面，减少来自缺失高阶贡献的不确定性，以更准确地描述顶夸克对产生过程。

### 方法

在领头色近似下评估必要的双圈振幅，提供缺失贡献影响的估计，分析微扰行为。

### 主要发现

首次在量子色动力学的次次领头阶预测各种运动学可观测量，大幅减少了来自缺失高阶贡献的不确定性。

### 结论

通过高精度预测，提高了对顶夸克对与轻喷注联合产生过程的理解，为标准模型验证和新物理搜索提供了更可靠的基础。

### 翻译

顶夸克对（已知的最重基本粒子）与轻喷注的联合产生是研究粒子物理标准模型特性的关键过程。由于其作为对顶夸克质量有很高敏感性的信号过程以及新物理搜索的背景过程的重要性，高精度预测微分截面至关重要。在本文中，我们首次在量子色动力学的次次领头阶预测了各种运动学可观测量。分析了微扰行为，并大幅减少了来自缺失高阶贡献的不确定性。必要的双圈振幅已在领头色近似下评估，我们提供了缺失贡献影响的估计。


### 论文摘要

The production of a top-quark pair, the heaviest known elementary particle, in association with a light jet is a key process for studying the properties of the Standard Model of Particle Physics. Due to its significance as a signal process with considerable sensitivity to the top-quark mass and as a background process for new physics searches, it is crucial to predict differential cross sections with high precision. In this article, we present, for the first time, predictions for various kinematical observables at next-to-next-to-leading order in Quantum Chromodynamics. The perturbative behavior is analyzed, and uncertainties arising from missing higher-order contributions are substantially reduced. The necessary two-loop amplitudes have been evaluated in the leading-color approximation, and we provide estimates for the impact of the missing contributions.

---

## 286. SEAL: Subspace-Anchored Watermarks for LLM Ownership

**论文链接:** [http://arxiv.org/abs/2511.11356v1](http://arxiv.org/abs/2511.11356v1)

**作者:** Yanbo Dai, Zongjie Li, Zhenlan Ji, Shuai Wang

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为SEAL的子空间锚定水印框架，用于保护大型语言模型的知识产权。该框架将多位签名直接嵌入到模型的潜在表示空间中，支持白盒和黑盒验证，能有效抵抗各种攻击，同时保持模型的原始功能。

### 背景

大型语言模型在各种自然语言处理任务上取得了显著成功，但训练这些模型需要大量计算资源、精心策划的数据集和复杂的对齐程序，因此构成了高价值的知识产权资产，需要强有力的保护机制。

### 目的

解决现有IP保护方法的局限性，包括模型指纹技术无法建立特定模型实例所有权，以及传统后门水印方法容易被通过微调或知识蒸馏等后处理操作移除的问题。

### 方法

提出SEAL水印框架，利用模型编辑技术将选定锚样本的隐藏表示与预定义的正交比特向量对齐，将水印嵌入模型的潜在表示空间，同时保留模型的原始事实预测能力，使水印功能上无害且隐蔽。

### 主要发现

在多个基准数据集和六个 prominent LLMs上的实验表明，SEAL相比11种现有方法在有效性、保真度、效率和鲁棒性方面表现更优。即使在对手了解水印机制和嵌入签名的情况下，SEAL仍能保持强大的验证性能。

### 结论

SEAL是一种有效且鲁棒的LLM水印方法，能够为大型语言模型提供可靠的知识产权保护，对抗包括知识型攻击在内的各种威胁。

### 翻译

大型语言模型在各种自然语言处理任务上取得了显著成功，展示了在文本生成、推理和问答方面达到人类水平的表现。然而，训练此类模型需要大量计算资源、精心策划的数据集和复杂的对齐程序。因此，它们构成了高价值的知识产权资产，需要强有力的保护机制。现有的IP保护方法存在严重局限性。模型指纹技术可以识别模型架构，但无法建立特定模型实例的所有权。相比之下，传统的基于后门的水印方法嵌入行为异常，可以通过微调或知识蒸馏等常见后处理操作轻松移除。我们提出了SEAL，一种基于子空间锚定的水印框架，将多位签名直接嵌入到模型的潜在表示空间中，支持白盒和黑盒验证场景。我们的方法利用模型编辑技术，将选定锚样本的隐藏表示与预定义的正交比特向量对齐。这种对齐方式嵌入水印，同时保留模型的原始事实预测，使水印在功能上无害且隐蔽。我们在多个基准数据集和六个 prominent LLMs上进行了全面实验，将SEAL与11种现有的指纹和水印方法进行比较，证明了其优越的有效性、保真度、效率和鲁棒性。此外，我们在潜在的知识型攻击下评估了SEAL，表明即使对手了解水印机制和嵌入的签名，它仍能保持强大的验证性能。


### 论文摘要

Large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks, demonstrating human-level performance in text generation, reasoning, and question answering. However, training such models requires substantial computational resources, large curated datasets, and sophisticated alignment procedures. As a result, they constitute highly valuable intellectual property (IP) assets that warrant robust protection mechanisms. Existing IP protection approaches suffer from critical limitations. Model fingerprinting techniques can identify model architectures but fail to establish ownership of specific model instances. In contrast, traditional backdoor-based watermarking methods embed behavioral anomalies that can be easily removed through common post-processing operations such as fine-tuning or knowledge distillation.   We propose SEAL, a subspace-anchored watermarking framework that embeds multi-bit signatures directly into the model's latent representational space, supporting both white-box and black-box verification scenarios. Our approach leverages model editing techniques to align the hidden representations of selected anchor samples with predefined orthogonal bit vectors. This alignment embeds the watermark while preserving the model's original factual predictions, rendering the watermark functionally harmless and stealthy. We conduct comprehensive experiments on multiple benchmark datasets and six prominent LLMs, comparing SEAL with 11 existing fingerprinting and watermarking methods to demonstrate its superior effectiveness, fidelity, efficiency, and robustness. Furthermore, we evaluate SEAL under potential knowledgeable attacks and show that it maintains strong verification performance even when adversaries possess knowledge of the watermarking mechanism and the embedded signatures.

---

## 287. Close-in compact super-Earth systems emerging from resonant chains: slow destabilization by unseen remnants of formation

**论文链接:** [http://arxiv.org/abs/2511.11329v1](http://arxiv.org/abs/2511.11329v1)

**作者:** Max Goldberg, Antoine C. Petit

**发布时间:** 2025-11-14

**备注:** Submitted to A&A

### GPT解析

### 总结

本研究解决了行星形成模拟预测与观测结果之间的差异，特别是共振系统随恒星年龄减少的现象。通过构建包含行星迁移和生长的简化模型，作者发现系统最初处于共振配置，但随时间逐渐失稳，这种失稳过程可以解释观测到的共振分数随年龄下降的趋势。

### 背景

行星形成模拟预测紧凑的、处于平均运动共振链中的小型行星系统，但凌日行星调查显示大多数系统是非共振且动态激发的。先前研究表明几乎所有原始共振链经历动态不稳定性和碰撞的场景与观测样本特征匹配，但现有模型尚未解释新观测发现的共振分数随恒星年龄在约1亿年尺度上急剧下降的现象。

### 目的

构建一个简化模型结合I型迁移、胚胎生长和N体积分，生成合成行星种群，解释观测到的共振分数随年龄下降的现象，并解决模拟预测与观测结果之间的差异。

### 方法

构建包含I型迁移、从胚胎生长和N体积分（持续到5亿年）的简化模型，生成合成行星种群，分析不稳定性的统计特性，使用威布尔分布将种群外推到十亿年的年龄，并与观测样本进行比较。

### 主要发现

几乎所有系统在盘阶段结束时都处于共振配置但随后缓慢扩散；动态不稳定性可能在数十或数亿年尺度上出现，尤其在具有会聚迁移陷阱的盘中形成的系统中；较小行星的次级链断裂会导致内部共振链失稳；不稳定性统计可用威布尔分布建模。

### 结论

模型系统与观测样本的高度匹配表明，这类模型预测的高共振比例实际上与数据一致，先前报告的共振系统过剩是将早期演化模拟与成熟系统比较的结果。由盘消散或其他早期机制触发的不稳定性不太可能与观测到的年轻系统一致。

### 翻译

行星形成模拟一致预测由行星-盘相互作用形成的紧凑的、处于平均运动共振链中的众多小型行星系统，但凌日行星调查发现大多数系统是非共振且有一定动态激发的。此前，一个几乎所有原始共振链都经历动态不稳定性和碰撞的场景被发现与观测到的行星样本的许多特征非常接近。然而，现有模型尚未针对新观测进行测试，这些观测显示共振分数随恒星年龄在约1亿年的时间尺度上急剧下降。我们构建了一个包含I型迁移、从胚胎生长和N体积分（持续到5亿年）的简化模型，并使用它生成合成行星种群。几乎所有系统在盘阶段结束时都处于共振配置，但开始缓慢地从共振中心扩散。动态不稳定性可能在数十或数亿年的时间尺度上出现，特别是在具有会聚迁移陷阱的盘中形成的系统中。在这种情况下，保持在其出生位置的较小行星的次级链最终会断裂，使内部共振链失稳。我们还表明，不稳定性统计可以用威布尔分布很好地建模，并使用该分布将我们的种群外推到十亿年的年龄。我们的模型系统与观测样本的密切匹配意味着这类模型预测的高共振比例实际上与数据一致，先前报告的共振系统过剩是将早期演化的模拟与成熟的十亿年老的系统进行比较的结果。这一结果还表明，由盘消散或其他非常早期的机制触发的不稳定性不太可能与观测到的年轻系统一致。


### 论文摘要

Planet formation simulations consistently predict compact systems of numerous small planets in chains of mean motion resonances formed by planet-disk interaction, but transiting planet surveys have found most systems to be non-resonant and somewhat dynamically excited. A scenario in which nearly all of the primordial resonant chains undergo dynamical instabilities and collisions has previously been found to closely match many features of the observed planet sample. However, existing models have not been tested against new observations that show a steep decline in the resonant fraction as a function of stellar age on a timescale of ~100 Myr. We construct a simplified model incorporating Type I migration, growth from embryos, and N-body integrations continued to 500 Myr and use it to generate a synthetic planet population. Nearly all systems exit the disk phase in a resonant configuration but begin slowly diffusing away from the center of the resonance. Dynamical instabilities can arise on timescales of tens or hundreds of Myr, especially when systems formed in disks with a convergent migration trap. In this case, a secondary chain of smaller planets that remained at their birth location eventually breaks, destabilizing the inner resonant chain. We also show that the instability statistics are well modeled by a Weibull distribution, and use this to extrapolate our population to Gyr ages. The close match of our modeled systems to the observed population implies that the high resonance fraction predicted by this class of models is in fact consistent with the data, and the previously-reported overabundance of resonant systems was a consequence of comparing simulations of early evolution to mature Gyr-old systems. This result also suggests that instabilities triggered by disk dissipation or other very early mechanisms are unlikely to be consistent with observed young systems.

---

## 288. Evidence for the Keplerian orbit of a close companion around a giant star

**论文链接:** [http://arxiv.org/abs/2511.11247v1](http://arxiv.org/abs/2511.11247v1)

**作者:** Mats Esseldeurs, Leen Decin, Joris De Ridder, Yoshiya Mori, Amanda I. Karakas, Jolien Malfait, Taïssa Danilovich, Stéphane Mathis, Anita M. S. Richards, Raghvendra Saha, Jeremy Yates, Marie Van de Sande, Maarten Baes, Alain Baudry, Jan Bolte, Thomas Ceulemans, Frederik De Ceuster, Ileyk El Mellah, Sandra Etoka, Carl Gottlieb, Fabrice Herpin, Pierre Kervella, Camille Landri, Louise Marinho, Iain McDonald, Karl Menten, Tom Millar, Zara Osborn, Bannawit Pimpanuwat, John Plane, Daniel J. Price, Lionel Siess, Owen Vermeulen, Ka Tat Wong

**发布时间:** 2025-11-14

**DOI:** 10.1038/s41550-025-02697-2

**备注:** published in Nature Astronomy: https://www.nature.com/articles/s41550-025-02697-2

### GPT解析

### 总结

研究发现渐近支巨星（AGB）恒星pi1 Gruis周围存在一个近距离伴星，该伴星遵循开普勒运动，质量略大于AGB恒星，可能是一颗主序星，且轨道呈圆形，这与模型预测不符。

### 背景

恒星的双星系统通过潮汐相互作用、质量转移和质量损失效应影响恒星演化。虽然年轻恒星、主序星、红巨星和致密天体周围都发现了这类伴星，但关于AGB恒星周围存在近距离伴星的直接观测证据一直难以获得。

### 目的

探索AGB恒星周围是否存在近距离伴星，并研究双星系统的演化机制。

### 方法

使用（亚）毫米波长时间域成像光谱技术观测AGB恒星pi1 Gruis，分析其伴星的运动特征。

### 主要发现

1) pi1 Gruis周围存在一个近距离伴星；2) 该伴星遵循开普勒运动；3) 伴星质量略大于AGB恒星，可能是一颗主序星；4) 与其他类似演化阶段的恒星相比，pi1 Gruis的伴星轨道呈圆形；5) 这表明偏心率生成机制可能在AGB阶段后期或之后才起作用；6) 模型预测的圆形化速率可能被低估了。

### 结论

多历元（亚）毫米波干涉测量技术在探测巨恒星附近伴星的开普勒运动方面具有潜力，为理解潮汐相互作用物理和双星演化开辟了新途径。

### 翻译

紧密的伴星通过潮汐相互作用、质量转移和质量损失效应影响恒星演化。虽然这样的伴星在年轻恒星天体、主序星、红巨星和致密天体周围都有被发现，但在渐近支（AGB）恒星周围存在紧密伴星的直接观测证据仍然难以捉摸。在此，我们呈现（亚）毫米波长时间域成像光谱，揭示了AGB恒星pi1 Gruis周围一个紧密伴星的开普勒运动。该伴星质量略大于AGB恒星，很可能是一颗主序星。与具有相似距离伴星的更演化恒星不同，pi1 Gruis的伴星遵循圆形轨道，这表明偏心率生成机制可能在AGB阶段后期或之后才出现。我们的分析表明，模型预测的圆形化速率可能被低估了。我们的结果突出了多历元（亚）毫米波干涉测量在探测巨恒星紧密伴星开普勒运动方面的潜力，为我们理解潮汐相互作用物理和双星演化开辟了新途径。


### 论文摘要

Close companions influence stellar evolution through tidal interactions, mass transfer, and mass loss effects. While such companions are detected around young stellar objects, main-sequence stars, red giants, and compact objects, direct observational evidence of close-in companions around asymptotic giant branch (AGB) stars has remained elusive. Here, we present (sub)millimeter time-domain imaging spectroscopy revealing the Keplerian motion of a close-in companion around the AGB star pi1 Gruis. The companion, slightly more massive than the AGB star, is likely a main-sequence star. Unlike more evolved stars with companions at comparable distances, pi1 Gru's companion follows a circular orbit, suggesting an eccentricity-generating mechanism late- or post-AGB. Our analysis suggests that model-predicted circularization rates may be underestimated. Our results highlight the potential of multi-epoch (sub)millimeter interferometry in detecting the Keplerian motion of close companions to giant stars and open avenues for our understanding of tidal interaction physics and binary evolution.

---

## 289. Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2511.11218v1](http://arxiv.org/abs/2511.11218v1)

**作者:** Chenhao Liu, Leyun Jiang, Yibo Wang, Kairan Yao, Jinchen Fu, Xiaoyu Ren

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究提出了一种基于强化学习的训练流程，使人形机器人能够进行高度动态且精确的羽毛球击球，无需运动先验或专家演示。训练采用三阶段课程学习，结合扩展卡尔曼滤波器进行轨迹预测，并在模拟和真实世界环境中进行了验证。

### 背景

人形机器人在确定性场景中表现出强大的交互能力，包括运动、操作和操作-运动任务。然而，真实世界是动态的，准静态交互不足以应对各种环境条件。

### 目的

朝着更动态的交互场景迈进，开发一种统一的全身控制器，使人形机器人能够协调地进行下肢步法和上身击球，无需任何运动先验或专家演示。

### 方法

采用三阶段课程学习训练：首先获取步法，然后生成精确引导的球拍摆动，最后进行任务聚焦的精炼。部署时结合扩展卡尔曼滤波器估计和预测羽毛球轨迹，并引入了一种无需预测的变体。

### 主要发现

在模拟中，两个机器人能够维持21次连续击球的回合；无需预测的变体实现了与目标已知策略可比的成功击球性能；真实世界测试中，预测和控制器模块表现出高精度，场上击球速度高达10米/秒，平均回球落点距离为3.5米。

### 结论

人形机器人可以在羽毛球中实现高度动态且精确的目标击球，该方法可适应更多动态关键领域。

### 翻译

人形机器人已经表现出在运动、操作以及更具挑战性的操作-运动任务中与确定性场景交互的强大能力。然而，真实世界是动态的，准静态交互不足以应对各种环境条件。作为迈向更动态交互场景的一步，我们提出了一个基于强化学习的训练流程，该流程为人形羽毛球生成统一的全身控制器，能够协调下肢步法和上身击球，无需任何运动先验或专家演示。训练遵循三阶段课程：首先获取步法，然后生成精确引导的球拍摆动，最后进行任务聚焦的精炼，产生下肢和上身共同服务于击球目标动作。对于部署，我们结合扩展卡尔曼滤波器来估计和预测羽毛球轨迹以实现目标击球。我们还引入了一种无需预测的变体，省去了EKF和显式轨迹预测。为验证该框架，我们在模拟和真实世界进行了五组实验。在模拟中，两个机器人能够维持21次连续击球的回合。此外，无需预测的变体相对于目标已知策略实现了可比的成功击球性能。在真实世界测试中，预测和控制器模块都表现出高精度，场上击球实现了高达10米/秒的出球速度，平均回球落点距离为3.5米。这些实验结果表明，我们的人形机器人可以在羽毛球中实现高度动态且精确的目标击球，并可适应更多动态关键领域。


### 论文摘要

Humanoid robots have demonstrated strong capability for interacting with deterministic scenes across locomotion, manipulation, and more challenging loco-manipulation tasks. Yet the real world is dynamic, quasi-static interactions are insufficient to cope with the various environmental conditions. As a step toward more dynamic interaction scenario, we present a reinforcement-learning-based training pipeline that produces a unified whole-body controller for humanoid badminton, enabling coordinated lower-body footwork and upper-body striking without any motion priors or expert demonstrations. Training follows a three-stage curriculum: first footwork acquisition, then precision-guided racket swing generation, and finally task-focused refinement, yielding motions in which both legs and arms serve the hitting objective. For deployment, we incorporate an Extended Kalman Filter (EKF) to estimate and predict shuttlecock trajectories for target striking. We also introduce a prediction-free variant that dispenses with EKF and explicit trajectory prediction. To validate the framework, we conduct five sets of experiment in both simulation and the real world. In simulation, two robots sustain a rally of 21 consecutive hits. Moreover, the prediction-free variant achieves successful hits with comparable performance relative to the target-known policy. In real-world tests, both the prediction and controller module exhibit high accuracy, and on-court hitting achieves an outgoing shuttle speed up to 10 m/s with a mean return landing distance of 3.5 m. These experiment results show that our humanoid robot can deliver highly dynamic while precise goal striking in badminton, and can be adapted to more dynamism critical domains.

---

## 290. RealisticDreamer: Guidance Score Distillation for Few-shot Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2511.11213v1](http://arxiv.org/abs/2511.11213v1)

**作者:** Ruocheng Wu, Haolan He, Yufei Wang, Zhihao Li, Bihan Wen

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种名为引导分数蒸馏(GSD)的框架，通过从预训练的视频扩散模型中提取多视图一致性先验，解决了3D高斯溅射在稀疏训练视图下的过拟合问题。

### 背景

3D高斯溅射(3DGS)因其高质量的实时渲染能力在3D场景表示中受到关注，但当输入包含稀疏训练视图时，3DGS容易出现过拟合，主要由于缺乏中间视图监督。

### 目的

解决3D高斯溅射在稀疏训练视图下的过拟合问题，通过引入视频扩散模型的多视图一致性先验来改进3D场景表示。

### 方法

提出引导分数蒸馏(GSD)框架，从预训练的视频扩散模型中提取多视图一致性先验；基于分数蒸馏采样的见解，监督来自多个相邻视图的渲染图像；引入统一的引导形式纠正VDM的噪声预测结果；结合基于真实深度图的深度变形引导和基于语义图像特征的引导，确保分数更新方向与正确的相机姿态和准确几何一致。

### 主要发现

通过引入深度变形引导和语义图像特征引导，解决了生成方向中涉及的物体运动和随机相机轨迹问题，使VDM的分数更新方向与正确的相机姿态和准确几何保持一致。

### 结论

实验结果表明，该方法在多个数据集上优于现有方法，有效解决了3D高斯溅射在稀疏训练视图下的过拟合问题。

### 翻译

3D高斯溅射(3DGS)最近因其高质量的实时渲染能力在3D场景表示中获得了广泛关注。然而，当输入包含稀疏训练视图时，3DGS容易出现过拟合，主要由于缺乏中间视图监督。受近期视频扩散模型(VDM)成功的启发，我们提出了一种名为引导分数蒸馏(GSD)的框架，从预训练的VDM中提取丰富的多视图一致性先验。基于分数蒸馏采样(SDS)的见解，GSD监督来自多个相邻视图的渲染图像，引导高斯溅射表示朝向VDM的生成方向。然而，生成方向通常涉及物体运动和随机相机轨迹，这使得在优化过程中进行直接监督具有挑战性。为解决这一问题，我们引入了一种统一的引导形式来纠正VDM的噪声预测结果。具体而言，我们结合了基于真实深度图的深度变形引导和基于语义图像特征的引导，确保来自VDM的分数更新方向与正确的相机姿态和准确几何保持一致。实验结果表明，我们的方法在多个数据集上优于现有方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D高斯溅射（3DGS）在稀疏训练视图下容易过拟合的问题。这个问题在现实中很重要，因为许多应用场景（如VR、3D游戏、自动驾驶等）难以获取大量训练视图，而现有方法在处理稀疏视图时往往会产生不准确的3D结构、过于平滑的纹理和视图不一致的问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到3DGS在稀疏视图下过拟合源于缺乏中间视图监督，然后受到视频扩散模型（VDM）能生成多视图一致视频的启发，结合现有的分数蒸馏技术（SDS）经验，提出利用预训练VDM提取多视图一致性先验。由于直接应用VDM存在挑战，作者设计统一的引导形式来校正VDM的噪声预测结果。该方法借鉴了SDS技术、DDIM逆变换、深度估计和图像变形技术，以及DINO特征作为语义特征引导的基础。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出引导分数蒸馏（GSD）框架，利用预训练视频扩散模型作为生成先验监督少样本3D高斯溅射，并通过深度变形引导和语义特征引导校正VDM的噪声预测方向。整体流程：1)从训练视图中选择两个视图作为相机轨迹起点和中间帧；2)渲染图像并应用DDIM逆变换获得噪声图像；3)使用视频扩散模型预测噪声；4)应用深度变形和语义特征引导校正噪声；5)计算校正结果的差异用于监督重建过程；6)结合多种损失函数进行优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出GSD框架首次将预训练VDM应用于少样本3DGS监督；2)设计深度变形引导基于真实深度图；3)提出语义特征引导基于DINO特征；4)引入统一引导形式校正VDM噪声预测方向。相比之前工作，GSD使用视频扩散模型而非2D扩散模型保持多视图一致性；性能不随训练视图增加而下降；通过两种引导机制更好解决几何对齐问题；是一种无需微调整个扩散模型的训练免费方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了引导分数蒸馏框架，通过结合深度变形和语义特征引导，有效利用视频扩散模型的多视图一致性先验，显著提升了少样本3D高斯溅射的重建质量和视图一致性。'}


### 论文摘要

3D Gaussian Splatting (3DGS) has recently gained great attention in the 3D scene representation for its high-quality real-time rendering capabilities. However, when the input comprises sparse training views, 3DGS is prone to overfitting, primarily due to the lack of intermediate-view supervision. Inspired by the recent success of Video Diffusion Models (VDM), we propose a framework called Guidance Score Distillation (GSD) to extract the rich multi-view consistency priors from pretrained VDMs. Building on the insights from Score Distillation Sampling (SDS), GSD supervises rendered images from multiple neighboring views, guiding the Gaussian splatting representation towards the generative direction of VDM. However, the generative direction often involves object motion and random camera trajectories, making it challenging for direct supervision in the optimization process. To address this problem, we introduce an unified guidance form to correct the noise prediction result of VDM. Specifically, we incorporate both a depth warp guidance based on real depth maps and a guidance based on semantic image features, ensuring that the score update direction from VDM aligns with the correct camera pose and accurate geometry. Experimental results show that our method outperforms existing approaches across multiple datasets.

---

## 291. Stroke Modeling Enables Vectorized Character Generation with Large Vectorized Glyph Model

**论文链接:** [http://arxiv.org/abs/2511.11119v1](http://arxiv.org/abs/2511.11119v1)

**作者:** Xinyue Zhang, Haolong Li, Jiawei Ma, Chen Ye

**发布时间:** 2025-11-14

### GPT解析

### 总结

本文提出了一种新型的大规模矢量字形模型（LVGM），通过预测下一个笔画来生成矢量中文字形，并发布了一个包含907,267个样本的大规模中文SVG数据集。

### 背景

矢量字形因其可扩展性和灵活性广泛应用于海报设计、网络动画和艺术展示等领域。在排版学中，矢量字形被视为有序笔画组成的特殊序列，这一概念可扩展到大型语言模型的标记序列预测能力。

### 目的

开发一种能够通过预测下一个笔画来生成矢量中文字形的新型模型。

### 方法

将笔画编码为离散潜变量（笔画嵌入），通过预测下一个笔画嵌入微调DeepSeek LLM来训练LVGM，并基于笔画创建了一个大规模中文SVG数据集。

### 主要发现

在给定有限笔画的情况下，模型能生成完整的字符、语义优美的单词和未见过的诗句；实验表明模型在数据规模上具有可扩展行为；生成的矢量字形已获专家验证。

### 结论

LVGM模型能有效生成矢量中文字形，具有良好的可扩展性和实用性。

### 翻译

矢量字形因其可扩展性和灵活性广泛应用于海报设计、网络动画、艺术展示等多个领域。在排版学中，它们通常被视为由有序笔画组成的特殊序列。这一概念扩展到了大型语言模型（LLMs）的标记序列预测能力，通过笔画建模实现矢量字符生成。本文提出了一种新型的大规模矢量字形模型（LVGM），用于通过预测下一个笔画来生成矢量中文字形。首先，我们将笔画编码为称为笔画嵌入的离散潜变量。随后，通过预测下一个笔画嵌入，微调DeepSeek LLM来训练我们的LVGM。在给定有限笔画的情况下，它能生成完整的字符、语义优美的单词，甚至是未见过的诗句，并以矢量形式呈现。此外，我们发布了一个新的基于笔画的大规模中文SVG数据集，包含907,267个样本，用于动态矢量字形生成。实验结果表明，我们的模型在数据规模上具有可扩展行为。我们生成的矢量字形已得到专家和相关人员的验证。


### 论文摘要

Vectorized glyphs are widely used in poster design, network animation, art display, and various other fields due to their scalability and flexibility. In typography, they are often seen as special sequences composed of ordered strokes. This concept extends to the token sequence prediction abilities of large language models (LLMs), enabling vectorized character generation through stroke modeling. In this paper, we propose a novel Large Vectorized Glyph Model (LVGM) designed to generate vectorized Chinese glyphs by predicting the next stroke. Initially, we encode strokes into discrete latent variables called stroke embeddings. Subsequently, we train our LVGM via fine-tuning DeepSeek LLM by predicting the next stroke embedding. With limited strokes given, it can generate complete characters, semantically elegant words, and even unseen verses in vectorized form. Moreover, we release a new large-scale Chinese SVG dataset containing 907,267 samples based on strokes for dynamically vectorized glyph generation. Experimental results show that our model has scaling behaviors on data scales. Our generated vectorized glyphs have been validated by experts and relevant individuals.

---

## 292. AdaptPNP: Integrating Prehensile and Non-Prehensile Skills for Adaptive Robotic Manipulation

**论文链接:** [http://arxiv.org/abs/2511.11052v1](http://arxiv.org/abs/2511.11052v1)

**作者:** Jinxuan Zhu, Chenrui Tie, Xinyi Cao, Yuran Wang, Jingxiang Guo, Zixuan Chen, Haonan Chen, Junting Chen, Yangyu Xiao, Ruihai Wu, Lin Shao

**发布时间:** 2025-11-14

### GPT解析

### 总结

这篇论文提出了ApaptPNP框架，一种结合视觉语言模型(VLM)的任务和运动规划系统，能够系统性地选择和组合抓取(P)和非抓取(NP)技能，以完成多样化的机器人操作任务，并在模拟和真实环境中进行了验证。

### 背景

非抓取(NP)操作（如推、戳、滑）使机器人在无法或不足以抓取的情况下能够改变物体状态，显著扩展了机器人操作能力。然而，开发一个能跨不同任务、物体和环境统一推广，并无缝集成非抓取和抓取(P)动作的框架仍然具有挑战性。机器人需要确定何时调用NP技能，为每个上下文选择合适的原语，并将P和NP策略组合成稳健的多步计划。

### 目的

开发一个统一的框架，使机器人能够智能地选择和结合抓取和非抓取技能，以完成多样化的操作任务，克服当前机器人操作中的局限性。

### 方法

ApaptPNP框架包含三个主要部分：1) 视觉语言模型(VLM)用于解释视觉场景观察和文本任务描述，生成规定P和NP动作序列和协调的高层计划骨架；2) 基于数字孪生的物体中心中间层预测期望的物体姿态，使操作序列能够进行主动心理预演；3) 控制模块合成底层机器人命令，通过连续执行反馈实现在线任务计划调整和通过VLM的自适应重新规划。

### 主要发现

研究在模拟和真实环境中的代表性P&NP混合操作任务上评估了ApaptPNP框架，验证了其有效性和潜力。

### 结论

混合P&NP操作作为实现通用类人机器人操作能力的关键步骤具有巨大潜力，ApaptPNP框架为此提供了有效的解决方案。

### 翻译

非抓取(NP)操作是指机器人在不形成稳定抓取的情况下改变物体状态（例如推、戳或滑），当抓取不可行或不足时，这显著扩展了机器人操作能力。然而，开发一个能够跨不同任务、物体和环境统一推广，并无缝集成非抓取和抓取(P)动作的统一框架仍然具有挑战性：机器人必须确定何时调用NP技能，为每个上下文选择合适的原语，并将P和NP策略组合成稳健的多步计划。我们介绍了ApaptPNP，一种视觉语言模型(VLM)赋能的任务和运动规划框架，它系统性地选择和组合P和NP技能来完成多样化的操作目标。我们的方法利用VLM解释视觉场景观察和文本任务描述，生成规定P和NP动作序列和协调的高层计划骨架。基于数字孪生的物体中心中间层预测期望的物体姿态，使操作序列能够进行主动心理预演。最后，控制模块合成底层机器人命令，通过连续执行反馈实现通过VLM的在线任务计划调整和自适应重新规划。我们在模拟和真实环境中的代表性P&NP混合操作任务上评估了ApaptPNP。这些结果强调了混合P&NP操作作为实现通用、类人机器人操作能力的关键步骤的潜力。项目网站：https://sites.google.com/view/adaptpnp/home


### 论文摘要

Non-prehensile (NP) manipulation, in which robots alter object states without forming stable grasps (for example, pushing, poking, or sliding), significantly broadens robotic manipulation capabilities when grasping is infeasible or insufficient. However, enabling a unified framework that generalizes across different tasks, objects, and environments while seamlessly integrating non-prehensile and prehensile (P) actions remains challenging: robots must determine when to invoke NP skills, select the appropriate primitive for each context, and compose P and NP strategies into robust, multi-step plans. We introduce ApaptPNP, a vision-language model (VLM)-empowered task and motion planning framework that systematically selects and combines P and NP skills to accomplish diverse manipulation objectives. Our approach leverages a VLM to interpret visual scene observations and textual task descriptions, generating a high-level plan skeleton that prescribes the sequence and coordination of P and NP actions. A digital-twin based object-centric intermediate layer predicts desired object poses, enabling proactive mental rehearsal of manipulation sequences. Finally, a control module synthesizes low-level robot commands, with continuous execution feedback enabling online task plan refinement and adaptive replanning through the VLM. We evaluate ApaptPNP across representative P&NP hybrid manipulation tasks in both simulation and real-world environments. These results underscore the potential of hybrid P&NP manipulation as a crucial step toward general-purpose, human-level robotic manipulation capabilities. Project Website: https://sites.google.com/view/adaptpnp/home

---

## 293. Bifurcations in Interior Transmission Eigenvalues: Theory and Computation

**论文链接:** [http://arxiv.org/abs/2511.11016v1](http://arxiv.org/abs/2511.11016v1)

**作者:** Davide Pradovera, Alessandro Borghi, Lukas Pieronek, Andreas Kleefeld

**发布时间:** 2025-11-14

### GPT解析

### 总结

本研究针对内部传输特征值问题中的非光滑谱行为进行了理论分析和数值验证，开发了识别非光滑行为充分条件的框架，并应用于径向对称几何形状。

### 背景

内部传输特征值问题在逆散射理论和非均匀介质谱分析中起核心作用，尽管在PDE层面具有平滑依赖性，但从材料参数到特征对的谱映射可能表现出非光滑或分支行为。

### 目的

开发识别一般域上ITP中非光滑谱行为充分条件的理论框架，并将分析特别应用于径向对称几何形状，以更精确地表征谱分支。

### 方法

将ITP表述为参数化、离散、非线性特征问题，使用基于匹配的自适应等值线特征求解器，在参数变化下准确有效地跟踪特征值轨迹。

### 主要发现

理论框架能够识别非光滑谱行为的条件；在径向对称几何形状上更精确地表征了谱分支；数值实验证实了理论预测并揭示了新的非光滑谱效应。

### 结论

研究提供了对ITP中非光滑谱行为的理论理解，通过数值方法验证了理论预测，并发现了新的非光滑谱效应。

### 翻译

内部传输特征值问题在逆散射理论和非均匀介质的谱分析中起着核心作用。尽管在偏微分方程层面，其对折射率的依赖是平滑的，但相应的从材料参数到特征对的谱映射可能表现出非平滑或分支行为。在这项工作中，我们开发了一个理论框架，识别在一般域上ITP中此类非平滑谱行为的充分条件。我们将分析进一步专门应用于某些径向对称几何形状，从而能够更精确地表征谱中的分支。在计算上，我们将ITP表述为参数化、离散、非线性特征问题，并使用基于匹配的自适应等值线特征求解器，在参数变化下准确有效地跟踪特征值轨迹。数值实验证实了理论预测，并揭示了新的非平滑谱效应。


### 论文摘要

The interior transmission eigenvalue problem (ITP) plays a central role in inverse scattering theory and in the spectral analysis of inhomogeneous media. Despite its smooth dependence on the refractive index at the PDE level, the corresponding spectral map from material parameters to eigenpairs may exhibit non-smooth or bifurcating behavior. In this work, we develop a theoretical framework identifying sufficient conditions for such non-smooth spectral behavior in the ITP on general domains. We further specialize our analysis to some radially symmetric geometries, enabling a more precise characterization of bifurcations in the spectrum. Computationally, we formulate the ITP as a parametric, discrete, nonlinear eigenproblem and use a match-based adaptive contour eigensolver to accurately and efficiently track eigenvalue trajectories under parameter variation. Numerical experiments confirm the theoretical predictions and reveal novel non-smooth spectral effects.

---

## 294. Studies of laser stimulated photodetachment from nanoparticles for particle charge measurements

**论文链接:** [http://arxiv.org/abs/2511.10956v1](http://arxiv.org/abs/2511.10956v1)

**作者:** Y. A. Ussenov, M. N. Shneider, S. Yatom, Y. Raitses

**发布时间:** 2025-11-14

### GPT解析

### 总结

该研究使用激光诱导光解吸(LSPD)技术成功测定了纳米颗粒的平均电荷，发现在平均直径约154.3纳米的颗粒中，每个颗粒的电荷约为16个基本电荷单位，低于轨道运动极限理论预测值，这种偏差归因于纳米尘埃等离子体中的显著电子耗尽。

### 背景

纳米颗粒的电荷测定比微颗粒更具挑战性，原因包括生长导致的大小变化、等离子体特性显著变化以及难以可视化单个颗粒，这使得传统的微颗粒电荷诊断方法在尘埃等离子体中无效。

### 目的

使用激光诱导光解吸(LSPD)技术推断纳米颗粒的平均电荷。

### 方法

在Ar和C2H2混合物中使用电容耦合RF放电生长纳米颗粒，通过圆柱形朗缪尔探针监测LSPD诱导的电子电流变化，在不同尘埃生长阶段获取并分析LSPD信号，结合激光消光法获得的颗粒密度估算每个颗粒的电荷。

### 主要发现

电子电流脉冲的缓慢衰减归因于残留负离子的存在，这些离子被有效静电捕获，LSPD后可能重新形成；对于平均直径约154.3纳米的颗粒，每个颗粒的电荷约为16个基本电荷单位，低于轨道运动极限理论预测值，这种偏差在纳米尘埃等离子体中常见，是由于显著的电子耗尽造成的。

### 结论

LSPD方法在Ar和C2H2纳米尘埃等离子体中的应用证实了该方法适用于估算单个纳米颗粒的电荷，但需注意背景残留负离子的电子解吸会影响解吸电流衰减，必须仔细考虑。

### 翻译

确定纳米颗粒电荷比微颗粒更具挑战性，这是由于生长导致的大小变化、等离子体特性显著变化以及难以可视化单个颗粒，这使得传统的微颗粒电荷诊断方法在尘埃等离子体中无效。在本工作中，我们利用激光诱导光解吸(LSPD)来推断纳米颗粒的平均电荷。纳米颗粒在Ar和C2H2混合物中使用电容耦合RF放电生长，并通过圆柱形朗缪尔探针监测LSPD诱导的电子电流变化。在不同的尘埃生长阶段获取并分析了LSPD信号。电子电流脉冲的缓慢衰减归因于残留负离子的存在，这些离子被有效静电捕获，并且在LSPD后可能重新形成新的负离子。使用通过激光消光法获得的已知颗粒密度值估算每个颗粒的电荷。对于平均直径Dp约154.3纳米的颗粒，发现每个颗粒的电荷约为16个基本电荷单位。这些电荷值低于轨道运动极限(OML)理论预测的值。这种偏差在纳米尘埃等离子体中很常见，是由于显著的电子耗尽造成的。Ar和C2H2纳米尘埃等离子体中的LSPD结果证实了这些方法适用于估算单个纳米颗粒的电荷。然而，也证明了背景残留负离子的电子解吸会影响解吸电流衰减，必须仔细考虑。


### 论文摘要

Determining nanoparticle charge is more challenging than that for microparticles due to growth-induced size changes, substantial plasma property variations, and difficulties in visualizing individual particles, rendering conventional microparticle charge diagnostics ineffective in dusty plasma. In this work, we utilized laser-stimulated photodetachment (LSPD) to deduce the mean charge of nanoparticles. Nanoparticles were grown in an Ara and C2H2 mixture using a capacitively coupled RF discharge and the LSPD induced changes in the electron current monitored by a cylindrical Langmuir probe. LSPD signals were obtained and analyzed across different dust growth phases. The prolonged decay of electron current pulses was attributed to the presence of residual negative ions, caused by the effective electrostatic trapping of these ions and the potential post - LSPD re-formation of new ones. The charge per particle was estimated using known values of particle density obtained by laser-light extinction method. For particles with mean diameter Dp ~ 154.3 nm, the charge found to be of Qd~16 elementary charge units. The charge values appear to be lower than the values predicted by orbital motion limited (OML) theory. This deviation is commonly observed in nanodusty plasmas due to significant electron depletion. LSPD results in Ar and C2H2 nano-dusty plasma confirm the applicability of these method for estimating individual nanoparticle charges. However, it has also been demonstrated that electron detachment from residual background negative ions can influence the detachment current decay and must be carefully considered

---

## 295. Multi-Joint Physics-Informed Deep Learning Framework for Time-Efficient Inverse Dynamics

**论文链接:** [http://arxiv.org/abs/2511.10878v1](http://arxiv.org/abs/2511.10878v1)

**作者:** Shuhao Ma, Zeyi Huang, Yu Cao, Wesley Doorsamy, Chaoyang Shi, Jun Li, Zhi-Qiang Zhang

**发布时间:** 2025-11-14

**备注:** 11 pages

### GPT解析

### 总结

本研究提出了一种物理信息深度学习框架，用于从运动学数据直接估计多关节系统的肌肉激活和力量，无需标记数据且计算效率高。

### 背景

在临床评估和辅助设备控制中，高效估计多关节系统的肌肉激活和力量至关重要。然而，传统方法计算成本高，且缺乏多关节应用的高质量标记数据集。

### 目的

解决传统方法计算昂贵和缺乏高质量标记数据集的问题，开发一种能够从运动学数据直接估计肌肉激活和力量的方法。

### 方法

提出了一种物理信息深度学习框架，包含一种新颖的多关节交叉注意力（MJCA）模块和双向门控循环单元（BiGRU）层，用于捕捉关节间协调。通过将多关节动力学、关节间耦合和外部力相互作用嵌入损失函数，创建了物理信息MJCA-BiGRU（PI-MJCA-BiGRU）模型。

### 主要发现

在两个数据集上的实验验证表明，PI-MJCA-BiGRU在不依赖真实标签的情况下，实现了与传统监督方法相当的性能。与其它基线架构相比，MJCA模块显著增强了关节间协调建模。

### 结论

所提出的PI-MJCA-BiGRU框架能够在不依赖标记数据的情况下，提供生理上一致的预测，同时实现高效的推理。

### 翻译

多关节系统肌肉激活和力量的高效估计对临床评估和辅助设备控制至关重要。然而，传统方法计算成本高，且缺乏多关节应用的高质量标记数据集。为解决这些挑战，我们提出了一种物理信息深度学习框架，直接从运动学数据估计肌肉激活和力量。该框架采用了一种新颖的多关节交叉注意力（MJCA）模块和双向门控循环单元（BiGRU）层，以捕捉关节间协调，使每个关节能够自适应地整合来自其他关节的运动信息。通过将多关节动力学、关节间耦合和外部力相互作用嵌入损失函数，我们的物理信息MJCA-BiGRU（PI-MJCA-BiGRU）无需标记数据即可提供生理上一致的预测，同时实现高效的推理。在两个数据集上的实验验证表明，PI-MJCA-BiGRU实现了与传统监督方法相当的性能，而不需要真实标签，同时MJCA模块与其他基线架构相比显著增强了关节间协调建模。


### 论文摘要

Time-efficient estimation of muscle activations and forces across multi-joint systems is critical for clinical assessment and assistive device control. However, conventional approaches are computationally expensive and lack a high-quality labeled dataset for multi-joint applications. To address these challenges, we propose a physics-informed deep learning framework that estimates muscle activations and forces directly from kinematics. The framework employs a novel Multi-Joint Cross-Attention (MJCA) module with Bidirectional Gated Recurrent Unit (BiGRU) layers to capture inter-joint coordination, enabling each joint to adaptively integrate motion information from others. By embedding multi-joint dynamics, inter-joint coupling, and external force interactions into the loss function, our Physics-Informed MJCA-BiGRU (PI-MJCA-BiGRU) delivers physiologically consistent predictions without labeled data while enabling time-efficient inference. Experimental validation on two datasets demonstrates that PI-MJCA-BiGRU achieves performance comparable to conventional supervised methods without requiring ground-truth labels, while the MJCA module significantly enhances inter-joint coordination modeling compared to other baseline architectures.

---

## 296. Adaptive Digital Twin of Sheet Metal Forming via Proper Orthogonal Decomposition-Based Koopman Operator with Model Predictive Control

**论文链接:** [http://arxiv.org/abs/2511.10852v1](http://arxiv.org/abs/2511.10852v1)

**作者:** Yi-Ping Chen, Derick Suarez, Ying-Kuan Tsai, Vispi Karkaria, Guanzhong Hu, Zihan Chen, Ping Guo, Jian Cao, Wei Chen

**发布时间:** 2025-11-13

### GPT解析

### 总结

本研究提出了一种自适应数字孪生框架，整合POD降维和Koopman算子，通过模型预测控制实现金属成形过程的实时决策与控制，并引入RLS算法实现模型实时更新，在机器人English wheel板材成形系统中验证了其有效性。

### 背景

数字孪生技术正在改变制造业，但在基于变形的金属成形应用中面临挑战，因为时空行为强耦合且工具路径与材料响应存在非线性关系。以English wheel为代表的板材成形工艺仍缺乏能够自主规划和调整成形策略的数字对应物。

### 目的

开发一个自适应数字孪生框架，解决金属成形过程中的实时决策和控制挑战，特别是实现板材成形过程的自主控制和优化。

### 方法

框架整合了POD进行物理感知的降维，以及Koopman算子在提升空间中表示非线性系统，通过模型预测控制进行实时决策。引入在线递归最小二乘算法实时更新算子系数，使模型能够随着新数据不断适应。

### 主要发现

在机器人English wheel板材成形系统上的实验表明，该自适应数字孪生能够通过有效捕捉非平稳工艺行为来控制成形过程，精确达到给定的目标形状。

### 结论

该框架为非线性制造系统的可解释、自适应和计算高效的数字孪生建立了通用方法，将降阶物理表示与数据驱动适应性相结合，支持自主工艺控制和优化。

### 翻译

数字孪生技术正在通过实现复杂过程的实时预测、监测和控制来改变制造业。然而，由于时空行为的强耦合以及工具路径与材料响应之间的非线性关系，将数字孪生应用于基于变形的金属成形仍然具有挑战性。例如，由English wheel(一种高度灵活但依赖人工的工艺)进行的板材成形仍然缺乏能够自主规划和调整成形策略的数字对应物。本研究提出了一种自适应数字孪生框架，整合了正交分解进行物理感知的降维和Koopman算子在提升空间中表示非线性系统，通过模型预测控制进行实时决策。为适应不断变化的工艺条件或材料状态，引入了在线递归最小二乘算法来实时更新算子系数，使数字孪生模型能够随着新的变形数据不断适应。该框架在机器人English wheel板材成形系统上进行了实验验证，在不同工具路径下测量和建模变形场。结果表明，自适应数字孪生能够通过有效捕捉非平稳工艺行为来控制成形过程，以达到给定的目标形状。除了这个案例研究，该框架为非线性制造系统的可解释、自适应和计算高效的数字孪生建立了通用方法，将降阶物理表示与数据驱动适应性相结合，以支持自主工艺控制和优化。


### 论文摘要

Digital Twin (DT) technologies are transforming manufacturing by enabling real-time prediction, monitoring, and control of complex processes. Yet, applying DT to deformation-based metal forming remains challenging because of the strongly coupled spatial-temporal behavior and the nonlinear relationship between toolpath and material response. For instance, sheet-metal forming by the English wheel, a highly flexible but artisan-dependent process, still lacks digital counterparts that can autonomously plan and adapt forming strategies. This study presents an adaptive DT framework that integrates Proper Orthogonal Decomposition (POD) for physics-aware dimensionality reduction with a Koopman operator for representing nonlinear system in a linear lifted space for the real-time decision-making via model predictive control (MPC). To accommodate evolving process conditions or material states, an online Recursive Least Squares (RLS) algorithm is introduced to update the operator coefficients in real time, enabling continuous adaptation of the DT model as new deformation data become available. The proposed framework is experimentally demonstrated on a robotic English Wheel sheet metal forming system, where deformation fields are measured and modeled under varying toolpaths. Results show that the adaptive DT is capable of controlling the forming process to achieve the given target shape by effectively capturing non-stationary process behaviors. Beyond this case study, the proposed framework establishes a generalizable approach for interpretable, adaptive, and computationally-efficient DT of nonlinear manufacturing systems, bridging reduced-order physics representations with data-driven adaptability to support autonomous process control and optimization.

---

## 297. A validated lumped-element model for bioinspired acoustic flow sensing toward the performance limit

**论文链接:** [http://arxiv.org/abs/2511.10830v1](http://arxiv.org/abs/2511.10830v1)

**作者:** Wei Sun, Wanyin Zheng, Xiangyu Wei, David A. Czaplewski, Ronald N. Miles, Jian Zhou

**发布时间:** 2025-11-13

### GPT解析

### 总结

研究开发了一种集总元件模型，用于预测浸没在流体中的细长微悬臂梁的宽带运动，该模型结合了分析简单性和定量准确性，并通过实验验证。

### 背景

流动传感对生物生存和技术创新都很重要。受生物机械感受器启发，人工流动传感器使用细长、粘性驱动的结构来检测微妙的流体运动。声学流动传感器模仿自然界对速度敏感的耳朵，有可能变革矢量声音检测，但对设计参数如何决定最终传感器性能的理解仍然有限。

### 目的

开发并实验验证一个集总元件模型，以捕捉浸没在流体中的细长微悬臂梁的宽带运动，有效指导流动传感器设计。

### 方法

开发并实验验证了一个集总元件模型，该模型结合了分析简单性和定量准确性，用于捕捉浸没在流体中的细长微悬臂梁的宽带运动。

### 主要发现

模型预测了流动引起的运动、热机械噪声和最小可检测信号水平，在100 Hz到10,000 Hz的宽频率范围内，模型预测与实验测量结果在空气中表现出高度一致性。

### 结论

验证后的模型为设计和高性能微/纳米机械传感器提供了直接的理论框架，这些传感器用于流动和矢量声音检测。

### 翻译

流动传感对生物生存和技术创新都至关重要。受生物机械感受器启发，人工流动传感器使用细长、粘性驱动的结构来检测微妙的流体运动。在这些传感器中，模仿自然界对速度敏感耳朵的声学流动传感器有可能变革矢量声音检测。然而，尽管它们有潜力，但对设计参数如何决定最终传感器性能的理解仍然有限。为了有效指导流动传感器设计，我们开发并实验验证了一个集总元件模型，该模型捕捉了浸没在流体中的细长微悬臂梁的宽带运动，结合了分析简单性与定量准确性。该模型预测了流动引起的运动、热机械噪声和最小可检测信号水平，在100 Hz到10,000 Hz的宽频率范围内，与空气中的实验测量结果表现出高度一致性。这一验证模型为设计和用于流动及矢量声音检测的高性能微/纳米机械传感器提供了直接的理论框架。


### 论文摘要

Flow sensing is fundamental to both biological survival and technological innovation. Inspired by biological mechanoreceptors, artificial flow sensors detect subtle fluid motion using slender, viscous-driven structures. Among these, acoustic flow sensors that mimic nature's velocity-sensitive ears have the potential to transform vector sound detection. Yet, despite their potential, understanding of how design parameters determine ultimate sensor performance remains limited. To effectively guide flow sensor design, we develop and experimentally validate a lumped-element model that captures the broadband motion of slender microcantilevers immersed in fluid, combining analytical simplicity with quantitative accuracy. The model predicts flow-induced motion, thermomechanical noise, and the minimum detectable signal level, showing strong agreement with experimental measurements in air over a broad frequency range from 100 Hz to 10,000 Hz. This validated model provides a straightforward theoretical framework for designing high-performance micro- and nanomechanical sensors for flow and vector sound detection.

---

## 298. Habit learning is associated with efficiently controlled network dynamics in naive macaque monkeys

**论文链接:** [http://arxiv.org/abs/2511.10757v1](http://arxiv.org/abs/2511.10757v1)

**作者:** Julia K. Brynildsen, Panagiotis Fotiadis, Karol P. Szymula, Jason Z. Kim, Fabio Pasqualetti, Ann M. Graybiel, Theresa M. Desrochers, Dani S. Bassett

**发布时间:** 2025-11-13

**备注:** This manuscript updates and substantially extends analyses originally presented in arXiv:2006.14565v1. Supplementary Material is provided as an ancillary file

### GPT解析

### 总结

研究提出了一个网络能量学理论，解释大脑状态如何影响序列行为，并通过猕猴实验验证了习惯形成与较低控制能量相关的假设，虚拟损伤实验证明了观察关系的稳健性。

### 背景

灵长类动物在不确定环境中使用分布式神经回路来学习习惯，但潜在的机制仍不清楚。

### 目的

提出一个关于网络能量学的正式理论，解释大脑状态如何影响序列行为，并在猕猴执行运动习惯任务时测试该理论。

### 方法

在猕猴的尾状核和皮层区域进行多单元记录，理论预测了通过有效连接传播活动所需的能量，用于在由试验特异性放电率代表的通道之间转换大脑状态，并通过虚拟损伤实验检验了关系的稳健性。

### 主要发现

习惯形成与较低的控制能量相关；观察到相似扫视模式之间的转换以及中等复杂度的模式所需能量较小；利用较少模式的会话所需能量也较小；模拟排除了神经元方向调谐的混杂因素。

### 结论

这项工作为研究行为如何从分布式回路中变化的活动产生铺平了道路。

### 翻译

灵长类动物在不确定环境中使用分布式神经回路来学习习惯，但潜在的机制仍不清楚。我们提出了一个关于网络能量学的正式理论，解释大脑状态如何影响序列行为。我们在执行运动习惯任务的猕猴的尾状核和皮层区域的多单元记录上测试了我们的理论。该理论预测了通过有效连接传播的活动所需的能量，用于在由通道间试验特异性放电率代表的大脑状态之间转换。我们假设习惯形成将与较低的控制能量相关。与此一致，我们观察到相似扫视模式之间以及中等复杂度模式之间的转换所需能量较小，且利用较少模式的会话所需能量也较小。模拟排除了神经元方向调谐的混杂因素。最后，虚拟损伤实验显示了观察到的控制能量与行为之间关系的稳健性。这项工作为研究行为如何从分布式回路中变化的活动产生铺平了道路。


### 论文摘要

Primates utilize distributed neural circuits to learn habits in uncertain environments, but the underlying mechanisms remain poorly understood. We propose a formal theory of network energetics explaining how brain states influence sequential behavior. We test our theory on multi-unit recordings from the caudate nucleus and cortical regions of macaques performing a motor habit task. The theory predicts the energy required to transition between brain states represented by trial-specific firing rates across channels, assuming activity spreads through effective connections. We hypothesized that habit formation would correlate with lower control energy. Consistent with this, we observed smaller energy requirements for transitions between similar saccade patterns and those of intermediate complexity, and sessions exploiting fewer patterns. Simulations ruled out confounds from neurons' directional tuning. Finally, virtual lesioning demonstrated robustness of observed relationships between control energy and behavior. This work paves the way for examining how behavior arises from changing activity in distributed circuitry.

---

## 299. Asymptotic Simplicity and Scattering in General Relativity from Quantum Field Theory

**论文链接:** [http://arxiv.org/abs/2511.10637v1](http://arxiv.org/abs/2511.10637v1)

**作者:** Stefano De Angelis, Aidan Herderschee, Radu Roiban, Fei Teng

**发布时间:** 2025-11-13

**备注:** 41 pages + references

### GPT解析

### 总结

该研究探讨了致密物体散射背景下渐近简单性的命运，通过计算双体系统应力张量作为源的时空度规，分析了引力场中Newman-Penrose标量的衰减行为，发现Sachs剥离性质在广义相对论中被显著违反。

### 背景

在广义相对论和引力散射理论中，渐近简单性是一个重要概念，特别是在致密物体相互作用的背景下，理解引力场的行为对于完善引力理论至关重要。

### 目的

研究旨在确定致密物体散射过程中时空度规的行为，特别是Newman-Penrose标量的衰减特性，以及Sachs剥离性质在广义相对论中的有效性。

### 方法

研究采用双体系统的应力张量作为源，在有限观测者距离处计算广义相对论中的时空度规，通过渐近展开方法，并将度规与动量空间中的末态引力子单点函数相关联，使用微扰量子场论技术进行计算。

### 主要发现

研究发现外部引力子的虚性中的简单极点和红外相关的对数分支割都做出了非平凡贡献；Newman-Penrose标量的衰减行为证实了Sachs剥离性质在后Minkowski展开的领头阶被违反；在后Minkowski展开的高阶分析中，发现了比先前认识到的更显著的剥离性质破坏。

### 结论

研究结果表明，局部化源与周围引力场之间的非线性长程相互作用导致Sachs剥离性质在广义相对论中被显著违反，这挑战了传统引力理论中对渐近行为的理解，对引力散射理论和致密物体相互作用的研究具有重要意义。

### 翻译

我们研究了致密物体散射物理相关背景下渐近简单性的命运。使用双体系统的应力张量作为源，我们在有限观测者距离处计算了广义相对论中的时空度规，采用渐近展开方法。为此，我们将度规与动量空间中的末态引力子单点函数相关联，使用微扰量子场论技术进行计算。外部引力子虚性中的简单极点和红外相关的对数分支割都做出了非平凡贡献。我们专注于确定Newman-Penrose标量的衰减行为，证实了先前关于Sachs剥离性质在后Minkowski展开的领头阶被违反的预测。我们在后Minkowski展开更高阶的分析揭示了比先前认识到的更显著的剥离性质破坏，这是由局部化源与周围引力场之间的非线性长程相互作用导致的。


### 论文摘要

We investigate the fate of asymptotic simplicity in physically relevant settings of compact-object scattering. Using the stress tensor of a two-body system as a source, we compute the spacetime metric in General Relativity at finite observer distance in an asymptotic expansion. To do so, we relate the metric to the final-state graviton one-point function in momentum space, which is computed using perturbative QFT techniques. Both the simple pole and the infrared-related logarithmic branch cut in the virtuality of the external graviton contribute nontrivially. We focus on determining the fall-off behavior of the Newman-Penrose scalars, confirming previous predictions that Sachs's peeling property is violated at leading order in the post-Minkowski expansion. Our analysis at higher orders in the post-Minkowskian expansion reveals a significantly stronger breakdown of the peeling property than previously recognized, which is the result of nonlinear, long-range interactions between localized sources and the surrounding gravitational field.

---

## 300. Safe Planning in Interactive Environments via Iterative Policy Updates and Adversarially Robust Conformal Prediction

**论文链接:** [http://arxiv.org/abs/2511.10586v1](http://arxiv.org/abs/2511.10586v1)

**作者:** Omid Mirzaeedodangeh, Eliot Shekhtman, Nikolai Matni, Lars Lindemann

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出了一种迭代框架，用于在交互式环境中自主代理的安全规划，解决了现有方法中安全保证失效的问题。

### 背景

自主代理在交互式环境（如自动驾驶车辆在行人和人类控制车辆之间）中的安全规划面临重大挑战，因为环境行为未知且对代理行为做出反应，导致交互驱动的分布偏移。

### 目的

提出一个迭代框架，在策略更新过程中稳健地保持安全保证，量化计划策略更新对环境行为的潜在影响。

### 方法

通过对抗稳健的保形预测(CP)实现，在每个回合中使用当前策略下观察到的数据进行常规CP步骤，然后基于策略到轨迹的敏感性分析调整CP结果以考虑分布偏移，从而在策略更新之间传递安全保证，实现安全、回合式的开环规划器。

### 主要发现

通过系统收缩分析提供了CP结果和策略更新保证收敛的条件，并在二维车辆-行人案例研究中实证演示了这些安全和收敛保证。

### 结论

据作者所知，这些是此类交互式环境中提供有效安全保证的首个结果。

### 翻译

交互式环境中自主代理的安全规划（如自动驾驶车辆在行人和人类控制车辆之间的控制）是一个重大挑战，因为环境行为未知且对自主代理行为做出反应。这种耦合导致交互驱动的分布偏移，其中自主代理的控制策略可能改变环境的行为，从而使现有工作中的安全保证失效。最近的研究使用保形预测(CP)来利用环境观察数据生成无分布的安全保证。然而，由于存在循环依赖（控制策略更新改变环境行为，反之亦然），CP的数据交换性假设在交互式设置中被违反。为了解决这一差距，我们提出了一个迭代框架，通过量化计划策略更新对环境行为的潜在影响，在策略更新过程中稳健地保持安全保证。我们通过对抗稳健的CP实现这一点，在每个回合中使用当前策略下观察到的数据进行常规CP步骤，然后通过分析调整CP结果以考虑分布偏移，从而在策略更新之间传递安全保证。这种调整基于策略到轨迹的敏感性分析，产生一个安全的、回合式的开环规划器。我们进一步对系统进行了收缩分析，提供了CP结果和策略更新保证收敛的条件。我们在一个二维车辆-行人案例研究中实证演示了这些安全和收敛保证。据我们所知，这些是此类交互式环境中提供有效安全保证的首个结果。


### 论文摘要

Safe planning of an autonomous agent in interactive environments -- such as the control of a self-driving vehicle among pedestrians and human-controlled vehicles -- poses a major challenge as the behavior of the environment is unknown and reactive to the behavior of the autonomous agent. This coupling gives rise to interaction-driven distribution shifts where the autonomous agent's control policy may change the environment's behavior, thereby invalidating safety guarantees in existing work. Indeed, recent works have used conformal prediction (CP) to generate distribution-free safety guarantees using observed data of the environment. However, CP's assumption on data exchangeability is violated in interactive settings due to a circular dependency where a control policy update changes the environment's behavior, and vice versa. To address this gap, we propose an iterative framework that robustly maintains safety guarantees across policy updates by quantifying the potential impact of a planned policy update on the environment's behavior. We realize this via adversarially robust CP where we perform a regular CP step in each episode using observed data under the current policy, but then transfer safety guarantees across policy updates by analytically adjusting the CP result to account for distribution shifts. This adjustment is performed based on a policy-to-trajectory sensitivity analysis, resulting in a safe, episodic open-loop planner. We further conduct a contraction analysis of the system providing conditions under which both the CP results and the policy updates are guaranteed to converge. We empirically demonstrate these safety and convergence guarantees on a two-dimensional car-pedestrian case study. To the best of our knowledge, these are the first results that provide valid safety guarantees in such interactive settings.

---

## 301. Preview, Accept or Discard? A Predictive Low-Motion Interaction Paradigm

**论文链接:** [http://arxiv.org/abs/2511.10532v1](http://arxiv.org/abs/2511.10532v1)

**作者:** Jose Berengueres

**发布时间:** 2025-11-13

### GPT解析

### 总结

该研究探讨了AI辅助预测输入如何减少电脑用户的手部动作，以解决重复性劳损(RSI)问题。

### 背景

重复性劳损(RSI)影响了约五分之一的电脑用户，尽管几十年来对鼠标进行了人体工程学重新设计，但问题仍未得到有效解决。所有现有设备仍需精细动作操作。

### 目的

研究预测性AI辅助输入是否可通过用排名屏幕建议替代物理指向来减少手部动作。

### 方法

引入'预览接受拒绝'(PAD)零点击交互范式，允许用户预览预测的GUI目标，浏览排名靠前的替代选项，并通过按键释放时间接受或拒绝。在基于浏览器的电子邮件客户端和ISO 9241-9键盘预测任务中评估PAD。

### 主要发现

与触控板使用相比，PAD显著减少了手部动作；当准确率与最佳拼写检查器相似时，任务时间与仅使用触控板相当。

### 结论

AI辅助预测输入可能有效减少电脑用户手部动作，从而减轻重复性劳损问题。

### 翻译

重复性劳损(RSI)影响了大约五分之一的电脑用户，尽管几十年来对鼠标进行了人体工程学重新设计，但这个问题仍未得到很好解决。所有这些设备都有一个基本限制：它们仍然需要精细动作来操作。这项研究探讨了预测性的AI辅助输入是否可以通过用屏幕上的排名建议替代物理指向来减少这种动作。为了保持用户的自主性，我们引入了'预览接受拒绝'(PAD)，一个零点击交互范式，让用户可以预览预测的GUI目标，循环浏览少量排名靠前的替代选项，并通过按键释放时间接受或拒绝它们。我们在两种设置中评估了PAD：一个基于浏览器的电子邮件客户端和一个在不同前3准确率下的ISO 9241-9键盘预测任务。在两项研究中，与使用触控板相比，PAD显著减少了手部动作，并且当准确率与最佳拼写检查器相似时，与仅使用触控板保持了可比的任务时间。


### 论文摘要

Repetitive strain injury (RSI) affects roughly one in five computer users and remains largely unresolved despite decades of ergonomic mouse redesign. All such devices share a fundamental limitation: they still require fine-motor motion to operate. This work investigates whether predictive, AI-assisted input can reduce that motion by replacing physical pointing with ranked on-screen suggestions. To preserve user agency, we introduce Preview Accept Discard (PAD), a zero-click interaction paradigm that lets users preview predicted GUI targets, cycle through a small set of ranked alternatives, and accept or discard them via key-release timing. We evaluate PAD in two settings: a browser-based email client and a ISO 9241-9 keyboard-prediction task under varying top-3 accuracies. Across both studies, PAD substantially reduces hand motion relative to trackpad use while maintaining comparable task times with the trackpad only when accuracies are similar to those of the best spell-checkers.

---

## 302. Entanglement Structure of Nonlocal Field Theories

**论文链接:** [http://arxiv.org/abs/2511.10505v1](http://arxiv.org/abs/2511.10505v1)

**作者:** Reza Pirmoradian, M. Hossein Bek-Khoshnevis, Sadaf Ebadi, M. Reza Tanhayi

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文研究了非局部相互作用对量子相关性的精细结构的影响，探索了玻色子非局部场论中的互信息和三部分信息等关联度量，发现非局部性尺度不仅决定体积律行为，还导致极长程互信息和特殊单配性结构，并通过全息对偶验证了场论与全息模型在关联性质上的显著差异。

### 背景

非局部相互作用已知可以产生体积律纠缠熵，但它们对量子相关性的精细结构的更深层次影响仍然是一个关键开放问题。

### 目的

探索非局部场论中超越纠缠熵的关联度量（互信息和三部分信息），研究非局部性尺度对量子相关性的影响。

### 方法

使用数值格点模拟研究玻色子非局部场论，并通过全息对偶验证结果。

### 主要发现

非局部性尺度决定体积律行为并导致极长程互信息和特殊单配性结构；增加大区域间分离可增强多部分纠缠；Ryu-Takayanagi公式正确捕捉熵的体积律标度；场论显示丰富空间相关性，而全息模型预测体积律相中互信息和三部分信息被完全抑制；全息描述中的非单配性行为与场论中的单配性结构化纠缠形成鲜明对比。

### 结论

非局部性产生了传统时空几何模型无法捕捉的复杂量子态，需要超越几何的新框架来全面理解这些关联的性质。

### 翻译

非局部相互作用已知可以产生体积律纠缠熵。然而，它们对量子相关性的精细结构的更深层次影响仍然是一个关键开放问题。在这项工作中，我们探索了一个玻色子非局部场论，考察了超越纠缠熵的关联度量，即互信息和三部分信息。使用数值格点模拟，我们表明非局部性尺度不仅决定了体积律行为的出现，还导致了显著的特征：特别是，极长程的互信息和异常的单配性结构。在这个区域中，增加大区域之间的分离可以悖论性地增强它们的多部分纠缠。通过全息对偶，我们验证了Ryu-Takayanagi公式正确捕捉了熵的体积律标度。然而，出现了显著的张力：虽然场论揭示了丰富的空间相关性，但全息模型预测在体积律相中互信息和三部分信息被完全抑制。全息描述中的这种非单配性行为与场论中观察到的单配性和高度结构化的纠缠形成鲜明对比。我们的结果表明，非局部性产生了如此复杂的量子态，以至于传统的时空几何模型无法捕捉。这表明需要超越几何的新框架来全面捕捉这些关联的本质。


### 论文摘要

Nonlocal interactions are known to generate volume-law entanglement entropy. However, their deeper impact on the fine structure of quantum correlations remains a key open question. In this work, we explore a bosonic nonlocal field theory, examining correlation measures beyond entanglement entropy, namely, mutual information and tripartite information. Using numerical lattice simulations, we show that the nonlocality scale, \(A\), not only determines the onset of volume-law behavior but also leads to striking features: notably, extremely long-range mutual information and an unusual monogamy structure. In this regime, increasing the separation between large regions can paradoxically enhance their multipartite entanglement. Through holographic duality, we verify that the Ryu-Takayanagi formula correctly captures the volume-law scaling of entropy. Yet, a significant tension emerges: while the field theory reveals rich spatial correlations, the holographic model predicts a complete suppression of both mutual and tripartite information in the volume-law phase. This non-monogamous behavior in the holographic description stands in sharp contrast to the monogamous and highly structured entanglement observed in the field theory. Our results demonstrate that nonlocality gives rise to quantum states of such complexity that conventional geometric models of spacetime fall short. This points to the need for a new framework that goes beyond geometry to fully capture the nature of these correlations.

---

## 303. Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration

**论文链接:** [http://arxiv.org/abs/2511.10439v1](http://arxiv.org/abs/2511.10439v1)

**作者:** Thomas Decker, Volker Tresp, Florian Buettner

**发布时间:** 2025-11-13

**备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

### GPT解析

### 总结

本文研究了基于扰动的解释方法与不确定性校准之间的关系，提出了一种名为ReCalX的新方法来重新校准模型，以提高解释的可靠性同时保持原始预测能力。

### 背景

基于扰动的解释方法被广泛用于提高机器学习模型的透明度，但它们的可靠性常常受到特定扰动下未知模型行为的影响。

### 目的

研究不确定性校准（模型置信度与实际准确性的一致性）与基于扰动的解释之间的关系，并解决模型在可解释性特定扰动下产生不可靠概率估计的问题。

### 方法

引入ReCalX，这是一种新颖的模型重新校准方法，旨在改进解释同时保留模型的原始预测能力。

### 主要发现

模型在受到可解释性特定扰动时，系统性地产生不可靠的概率估计，这直接损害了全局和局部解释的质量。ReCalX能够有效减少扰动特定的不校准问题，同时增强解释的鲁棒性和对全局重要输入特征的识别能力。

### 结论

通过在多种模型和数据集上的实证评估，ReCalX一致地最有效地减少了扰动特定的不校准问题，同时提高了解释的鲁棒性。

### 翻译

基于扰动的解释方法在实践中被广泛用于提高机器学习模型的透明度。然而，它们的可靠性常常受到所用特定扰动下未知模型行为的损害。本文研究了不确定性校准（模型置信度与实际准确性的一致性）与基于扰动的解释之间的关系。我们表明，当模型受到可解释性特定的扰动时，会系统性地产生不可靠的概率估计，并且从理论上证明这直接损害了全局和局部解释的质量。为了解决这个问题，我们引入了ReCalX，这是一种新颖的重新校准模型的方法，用于改进解释同时保留其原始预测。跨多种模型和数据集的实证评估表明，ReCalX一致地最有效地减少了扰动特定的不校准问题，同时增强了解释的鲁棒性和对全局重要输入特征的识别。


### 论文摘要

Perturbation-based explanations are widely utilized to enhance the transparency of machine-learning models in practice. However, their reliability is often compromised by the unknown model behavior under the specific perturbations used. This paper investigates the relationship between uncertainty calibration - the alignment of model confidence with actual accuracy - and perturbation-based explanations. We show that models systematically produce unreliable probability estimates when subjected to explainability-specific perturbations and theoretically prove that this directly undermines global and local explanation quality. To address this, we introduce ReCalX, a novel approach to recalibrate models for improved explanations while preserving their original predictions. Empirical evaluations across diverse models and datasets demonstrate that ReCalX consistently reduces perturbation-specific miscalibration most effectively while enhancing explanation robustness and the identification of globally important input features.

---

## 304. LongComp: Long-Tail Compositional Zero-Shot Generalization for Robust Trajectory Prediction

**论文链接:** [http://arxiv.org/abs/2511.10411v1](http://arxiv.org/abs/2511.10411v1)

**作者:** Benjamin Stoler, Jonathan Francis, Jean Oh

**发布时间:** 2025-11-13

**备注:** 8 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种新的长尾评估设置和安全感知的场景分解框架，用于评估自动驾驶轨迹预测模型在罕见安全关键场景下的鲁棒性，并通过任务模块化门控网络和难度预测头改进了模型的泛化能力。

### 背景

自动驾驶中的轨迹预测方法必须处理罕见但安全关键的场景，这使得仅依靠真实世界数据收集变得不可行。

### 目的

评估在罕见安全关键场景下的鲁棒性，并提出新的长尾评估设置。

### 方法

提出了一种安全感知的场景分解框架，将场景分解为离散的自我和社会上下文；受计算机视觉中组合零样本图像标注的启发，保留新的上下文组合来构建具有挑战性的封闭世界和开放世界设置；扩展任务模块化门控网络在轨迹预测模型中操作；开发一个辅助的难度预测头来改进内部表示。

### 主要发现

在封闭世界和开放世界设置中，相对于最先进基线的分布内性能，OOD性能差距分别为5.0%和14.7%；提出的策略将这两种设置中的OOD性能差距分别减少到2.8%和11.5%，同时仍然提高了分布内性能。

### 结论

提出的方法能够提高轨迹预测模型在罕见安全关键场景下的鲁棒性；新的长尾评估设置有助于更好地评估模型的泛化能力。

### 翻译

自动驾驶中的轨迹预测方法必须处理罕见但安全关键的场景，这使得仅依靠真实世界数据收集变得不可行。为了评估在这些条件下的鲁棒性，我们提出了新的长尾评估设置，这些设置重新划分数据集以创建具有挑战性的分布外测试集。我们首先介绍了一种安全感知的场景分解框架，该框架将场景分解为离散的自我上下文和社会上下文。基于与计算机视觉中组合零样本图像标注的类比，我们然后保留新的上下文组合来构建具有挑战性的封闭世界和开放世界设置。这一过程导致最先进基线的未来运动预测在封闭世界和开放世界设置中分别产生5.0%和14.7%的性能差距，相对于分布内性能。为了提高泛化能力，我们将任务模块化门控网络扩展到在轨迹预测模型中操作，并开发了一个辅助的难度预测头来改进内部表示。我们的策略将这两种设置中的性能差距分别减少到2.8%和11.5%，同时仍然提高了分布内性能。


### 论文摘要

Methods for trajectory prediction in Autonomous Driving must contend with rare, safety-critical scenarios that make reliance on real-world data collection alone infeasible. To assess robustness under such conditions, we propose new long-tail evaluation settings that repartition datasets to create challenging out-of-distribution (OOD) test sets. We first introduce a safety-informed scenario factorization framework, which disentangles scenarios into discrete ego and social contexts. Building on analogies to compositional zero-shot image-labeling in Computer Vision, we then hold out novel context combinations to construct challenging closed-world and open-world settings. This process induces OOD performance gaps in future motion prediction of 5.0% and 14.7% in closed-world and open-world settings, respectively, relative to in-distribution performance for a state-of-the-art baseline. To improve generalization, we extend task-modular gating networks to operate within trajectory prediction models, and develop an auxiliary, difficulty-prediction head to refine internal representations. Our strategies jointly reduce the OOD performance gaps to 2.8% and 11.5% in the two settings, respectively, while still improving in-distribution performance.

---

## 305. Locally Linear Convergence for Nonsmooth Convex Optimization via Coupled Smoothing and Momentum

**论文链接:** [http://arxiv.org/abs/2511.10239v2](http://arxiv.org/abs/2511.10239v2)

**作者:** Reza Rahimi Baghbadorani, Sergio Grammatico, Peyman Mohajerin Esfahani

**发布时间:** 2025-11-13

### GPT解析

### 总结

提出了一种针对非光滑凸优化问题的自适应加速平滑技术，该技术将平滑更新规则与动量参数耦合，并在多种问题上验证了其性能。研究提供了全局次线性和局部线性收敛保证，并观察到实际应用中的两阶段收敛行为。

### 背景

非光滑凸优化是一类重要的优化问题，在实际应用中广泛存在，如Lasso问题、MaxCut问题、核范数最小化和模型预测控制等。传统的优化方法在处理这类问题时可能面临收敛速度慢或参数选择困难等挑战。

### 目的

设计一种自适应加速平滑技术，提高非光滑凸优化问题的求解效率；将方法扩展到目标函数是两个非光滑函数之和的情况；提供严格的收敛性保证；并在多种实际应用中验证方法的有效性。

### 方法

提出一种自适应加速平滑技术，其中平滑更新规则与动量参数耦合；将方法扩展到目标函数是两个非光滑函数之和的情况；提供全局O(1/k)的次线性收敛保证和满足局部强凸性条件时的局部线性收敛保证；在多个实际问题上验证算法性能。

### 主要发现

所提方法提供了全局最优的O(1/k)次线性收敛保证；当非光滑项满足局部强凸性条件时，方法表现出局部线性收敛；在实际应用中观察到瞬态收敛速率为O(1/k^2)，随后是渐近线性收敛，这种两阶段行为可以通过所提出的平滑规则解释；在多个问题上验证了方法的有效性。

### 结论

所提出的自适应加速平滑技术通过将平滑更新规则与动量参数耦合，有效提高了非光滑凸优化问题的求解效率；方法在理论上提供了最优的收敛保证，在实际应用中表现出优于理论预期的性能；该技术可广泛应用于各类非光滑优化问题，具有重要的理论和实践意义。

### 翻译

我们提出了一种针对非光滑凸优化问题的自适应加速平滑技术，其中平滑更新规则与动量参数耦合。我们还扩展了设置到目标函数是两个非光滑函数之和的情况。关于收敛速率，我们提供了全局（最优）次线性收敛保证O(1/k)，这对于所研究的函数类被证明是最优的，同时如果非光滑项满足所谓的局部强凸性条件，则提供局部线性收敛。我们在多个问题上验证了我们的算法性能，包括带l1范数的回归（Lasso问题）、稀疏半定规划（MaxCut问题）、核范数最小化及其在无模型故障诊断中的应用，以及l1正则化模型预测控制，展示了耦合的好处。一个有趣的观察是，尽管我们的全局收敛结果保证O(1/k)收敛，但我们一致观察到实际瞬态收敛速率为O(1/k^2)，随后是理论预期的渐近线性收敛。这种两阶段行为也可以从所提出的平滑规则角度解释。


### 论文摘要

We propose an adaptive accelerated smoothing technique for a nonsmooth convex optimization problem where the smoothing update rule is coupled with the momentum parameter. We also extend the setting to the case where the objective function is the sum of two nonsmooth functions. With regard to convergence rate, we provide the global (optimal) sublinear convergence guarantees of O(1/k), which is known to be provably optimal for the studied class of functions, along with a local linear rate if the nonsmooth term fulfills a so-call locally strong convexity condition. We validate the performance of our algorithm on several problem classes, including regression with the l1-norm (the Lasso problem), sparse semidefinite programming (the MaxCut problem), Nuclear norm minimization with application in model free fault diagnosis, and l_1-regularized model predictive control to showcase the benefits of the coupling. An interesting observation is that although our global convergence result guarantees O(1/k) convergence, we consistently observe a practical transient convergence rate of O(1/k^2), followed by asymptotic linear convergence as anticipated by the theoretical result. This two-phase behavior can also be explained in view of the proposed smoothing rule.

---

## 306. OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer

**论文链接:** [http://arxiv.org/abs/2511.10560v2](http://arxiv.org/abs/2511.10560v2)

**作者:** Haosong Peng, Hao Li, Yalun Dai, Yushi Lan, Yihang Luo, Tianyu Qi, Zhengshen Zhang, Yufeng Zhan, Junfei Zhang, Wenchao Xu, Ziwei Liu

**发布时间:** 2025-11-13

**备注:** Project Page: https://livioni.github.io/OmniVGGT-official/

### GPT解析

### 总结

本文提出了OmniVGGT框架，能够有效利用任意数量的辅助几何模态，在多种视觉任务上实现性能提升。

### 背景

3D基础模型开始统一各种视觉任务，但大多数模型仅假设RGB输入，忽略了可用的几何线索（如相机内参、姿态和深度图）。

### 目的

解决大多数3D基础模型忽略几何线索的问题，提出一个能够有效利用任意数量辅助几何模态的框架。

### 方法

设计了OmniVGGT框架，包含GeoAdapter编码器将几何信息注入基础模型，采用零初始化卷积渐进式注入信息，并引入随机多模态融合方案在训练时随机采样模态子集。

### 主要发现

OmniVGGT在带有辅助输入的任务上优于先前方法，即使仅使用RGB输入也能达到最先进结果；集成到视觉-语言-动作模型中后，在主流基准测试和机器人任务上均表现出色。

### 结论

OmniVGGT是一个能够有效利用多种几何模态的框架，保持与VGGT相当的推理速度，在各种视觉任务上具有实用价值。

### 翻译

通用3D基础模型已经开始引领统一各种视觉任务的趋势，但大多数模型仅假设RGB输入并忽略了可用的几何线索（例如相机内参、姿态和深度图）。为解决这个问题，我们引入了OmniVGGT，一个新颖的框架，能够在训练和推理过程中有效利用任意数量的辅助几何模态。在我们的框架中，提出了GeoAdapter将深度和相机内参/外参编码到空间基础模型中。它采用零初始化卷积来渐进式注入几何信息而不破坏基础模型的表示空间。这种设计确保了稳定的优化且开销可忽略，即使在有多个额外输入的情况下也能保持与VGGT相当的推理速度。此外，还提出了随机多模态融合方案，在训练时对每个实例随机采样模量子集。这使得测试期间可以使用任意数量的模态输入，促进学习鲁棒的空间表示而非过度拟合辅助线索。在单目/多目深度估计、多视图立体和相机姿态估计上的全面实验表明，OmniVGGT优于先前方法，即使仅使用RGB输入也能达到最先进的结果。为进一步突出其实用性，我们将OmniVGGT集成到视觉-语言-动作模型中。由OmniVGGT增强的VLA模型不仅在主流基准测试上优于普通基于点云的基线模型，还能有效利用可用的辅助输入在机器人任务上实现一致的性能提升。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有3D基础模型（如VGGT）仅支持RGB图像输入，忽视深度图、相机参数等辅助几何模态信息的问题。这个问题很重要，因为在现实世界的3D视觉应用中（如VR/AR、自动驾驶、机器人操作），这些辅助信息通常可获取且能显著提升3D感知和重建质量，而现有模型无法充分利用这些信息限制了其实用性和性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3D基础模型的局限性，特别是它们对RGB-only输入的限制，以及Pow3R等方法在多模态处理上的不足（最多只能处理两种输入）。基于此，作者设计了OmniVGGT框架，核心是引入GeoAdapter来编码深度和相机参数到空间基础模型中，并采用随机多模态融合策略。该方法借鉴了VGGT的基础架构和Pow3R的多模态处理思路，但通过创新的零初始化卷积处理相机姿态信息，以及更灵活的模态融合策略，解决了现有方法的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是设计一个能够灵活处理任意数量辅助几何模态输入的3D基础模型，通过GeoAdapter模块有效融合多模态信息，并采用随机多模态融合策略使模型能适应测试时不同的输入组合。整体流程包括：1)接受图像和任意数量的辅助输入；2)通过GeoAdapter处理相机参数（归一化编码、零卷积处理）和深度图（归一化编码）；3)通过交替注意力块处理标记；4)使用三个预测头输出深度图、相机姿态和3D点图；5)训练时采用随机多模态融合策略模拟不同测试场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)轻量级GeoAdapter模块，能有效融合深度和相机参数信息；2)随机多模态融合策略，使模型能处理测试时任意数量的模态输入；3)零初始化卷积处理相机姿态，确保训练稳定性；4)端到端多任务学习框架。相比之前的工作：与VGGT相比扩展了输入模态；与Pow3R相比能处理任意数量的模态输入且推理速度更快（约30倍）；与其他3D重建方法相比实现了统一框架处理多种3D任务并灵活利用辅助信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OmniVGGT通过创新的GeoAdapter和随机多模态融合策略，实现了首个能够灵活处理任意数量辅助几何模态输入的3D基础模型，显著提升了3D重建和机器人任务的性能。'}


### 论文摘要

General 3D foundation models have started to lead the trend of unifying diverse vision tasks, yet most assume RGB-only inputs and ignore readily available geometric cues (e.g., camera intrinsics, poses, and depth maps). To address this issue, we introduce OmniVGGT, a novel framework that can effectively benefit from an arbitrary number of auxiliary geometric modalities during both training and inference. In our framework, a GeoAdapter is proposed to encode depth and camera intrinsics/extrinsics into a spatial foundation model. It employs zero-initialized convolutions to progressively inject geometric information without disrupting the foundation model's representation space. This design ensures stable optimization with negligible overhead, maintaining inference speed comparable to VGGT even with multiple additional inputs. Additionally, a stochastic multimodal fusion regimen is proposed, which randomly samples modality subsets per instance during training. This enables an arbitrary number of modality inputs during testing and promotes learning robust spatial representations instead of overfitting to auxiliary cues. Comprehensive experiments on monocular/multi-view depth estimation, multi-view stereo, and camera pose estimation demonstrate that OmniVGGT outperforms prior methods with auxiliary inputs and achieves state-of-the-art results even with RGB-only input. To further highlight its practical utility, we integrated OmniVGGT into vision-language-action (VLA) models. The enhanced VLA model by OmniVGGT not only outperforms the vanilla point-cloud-based baseline on mainstream benchmarks, but also effectively leverages accessible auxiliary inputs to achieve consistent gains on robotic tasks.

---

## 307. LoG3D: Ultra-High-Resolution 3D Shape Modeling via Local-to-Global Partitioning

**论文链接:** [http://arxiv.org/abs/2511.10040v1](http://arxiv.org/abs/2511.10040v1)

**作者:** Xinran Yang, Shuichang Lai, Jiangjing Lyu, Hongjie Li, Bowen Pan, Yuanqi Li, Jie Guo, Zhou Zhengkang, Yanwen Guo

**发布时间:** 2025-11-13

**备注:** 11 pages, 6 figures

### GPT解析

### 总结

本文提出了一种基于无符号距离场(UDFs)的新型3D变分自编码器(VAE)框架，通过局部到全局(LoG)架构和Pad-Average策略，解决了高保真3D内容生成中的拓扑表示和几何细节保持问题，实现了超高分辨率(2048^3)的3D内容生成，并在重建精度和生成质量方面达到最先进水平。

### 背景

生成高保真3D内容面临表示任意拓扑结构（如开放表面和复杂内部结构）同时保持几何细节的挑战。现有基于符号距离场(SDFs)的方法受限于昂贵的水密预处理且难以处理非流形几何，而点云表示则常出现采样伪影和表面不连续问题。

### 目的

克服现有3D内容生成方法的局限性，提出一种更高效、更稳健的3D表示方法和生成框架。

### 方法

构建基于无符号距离场(UDFs)的3D变分自编码器(VAE)框架，采用局部到全局(LoG)架构将UDF分割为均匀子体积(UBlocks)，结合3D卷积捕获局部细节和稀疏变换器确保全局一致性，并应用Pad-Average策略保证子体积边界处的平滑过渡，实现无缝扩展到超高分辨率(2048^3)。

### 主要发现

实验证明该方法在重建精度和生成质量方面达到最先进水平，能够产生更优的表面平滑度和几何灵活性，成功解决了现有方法在处理复杂和不完整形状时的局限性。

### 结论

通过创新的UDF表示方法和LoG架构，本文提出的3D VAE框架克服了现有方法的局限性，实现了高质量、高分辨率的3D内容生成，为3D内容生成领域提供了新的解决方案。

### 翻译

生成高保真3D内容仍然是一个基本挑战，因为需要表示任意拓扑结构（如开放表面和复杂内部结构）同时保持几何细节。基于符号距离场(SDFs)的现有方法受到昂贵的水密预处理阻碍，难以处理非流形几何，而点云表示通常存在采样伪影和表面不连续问题。为了克服这些限制，我们提出了一种构建于无符号距离场(UDFs)的新型3D变分自编码器(VAE)框架——一种更稳健且计算效率更高的表示方法，自然处理复杂和不完整的形状。我们的核心创新是局部到全局(LoG)架构，通过将UDF分割为均匀子体积（称为UBlocks）来处理UDF。该架构结合3D卷积捕获局部细节和稀疏变换器强制全局一致性。Pad-Average策略进一步确保重建时子体积边界处的平滑过渡。这种模块化设计能够无缝扩展到超高分辨率（最高2048^3）——这是3D VAEs以前无法实现的领域。实验证明该方法在重建精度和生成质量方面达到最先进水平，产生更优的表面平滑度和几何灵活性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决高保真度3D内容生成的问题，特别是如何在保持几何细节的同时处理任意拓扑结构（如开放表面和复杂内部结构）。这个问题在现实中很重要，因为3D内容生成是娱乐、设计和机器人等领域的基础挑战，而现有方法要么需要昂贵的水密预处理（SDF方法），要么存在表面不连续问题（点云方法），限制了复杂3D模型的创建和应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了现有SDF和点云表示的局限性，然后选择无符号距离场(UDF)作为更高效、更鲁棒的表示方法。他们发现现有VAE架构的全局瓶颈设计会丢失高频几何细节，因此提出结合局部3D卷积和全局稀疏transformer的混合架构。该方法借鉴了TRELLIS中的稀疏体素表示、transformer的注意力机制以及Marching Cubes网格提取算法，但在架构设计和表示方法上有显著创新。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将高分辨率UDF划分为均匀的子体积（UBlock），通过局部3D卷积捕获细节，利用稀疏transformer确保全局一致性，并使用Pad-Average策略平滑边界。整体流程为：1)将网格转换为UDF；2)创建稀疏表示；3)划分为UBlock；4)使用混合编码器编码；5)使用对称解码器解码；6)通过Pad-Average策略重组；7)用Marching Cubes提取最终网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)UBlock表示，解耦分辨率与复杂度；2)LoG-VAE架构，结合局部卷积和全局transformer；3)Pad-Average策略，确保边界平滑。相比之前工作，该方法使用UDF而非SDF避免水密预处理要求，可扩展到20483超高分辨率，在保持几何细节的同时支持任意拓扑结构，实现了更平滑的表面和更灵活的几何建模。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LoG3D通过基于无符号距离场的局部到全局变分自编码器架构，实现了超高分辨率3D形状建模，能够在保持几何细节的同时支持任意拓扑结构，突破了现有方法在分辨率和拓扑灵活性上的限制。'}


### 论文摘要

Generating high-fidelity 3D contents remains a fundamental challenge due to the complexity of representing arbitrary topologies-such as open surfaces and intricate internal structures-while preserving geometric details. Prevailing methods based on signed distance fields (SDFs) are hampered by costly watertight preprocessing and struggle with non-manifold geometries, while point-cloud representations often suffer from sampling artifacts and surface discontinuities. To overcome these limitations, we propose a novel 3D variational autoencoder (VAE) framework built upon unsigned distance fields (UDFs)-a more robust and computationally efficient representation that naturally handles complex and incomplete shapes. Our core innovation is a local-to-global (LoG) architecture that processes the UDF by partitioning it into uniform subvolumes, termed UBlocks. This architecture couples 3D convolutions for capturing local detail with sparse transformers for enforcing global coherence. A Pad-Average strategy further ensures smooth transitions at subvolume boundaries during reconstruction. This modular design enables seamless scaling to ultra-high resolutions up to 2048^3-a regime previously unattainable for 3D VAEs. Experiments demonstrate state-of-the-art performance in both reconstruction accuracy and generative quality, yielding superior surface smoothness and geometric flexibility.

---

## 308. AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.10017v1](http://arxiv.org/abs/2511.10017v1)

**作者:** Xinyi Wang, Xun Yang, Yanlong Xu, Yuchen Wu, Zhen Li, Na Zhao

**发布时间:** 2025-11-13

**备注:** NeurIPS 2025

### GPT解析

### 总结

该研究引入了细粒度3D具身推理任务，并提出AffordBot框架，通过结合多模态大语言模型与思维链推理，实现了高效的人与智能体在物理环境中的协作。

### 背景

现有的人与智能体在物理环境中的协作方法通常在物体级别操作，或者不连贯地处理细粒度的功能推理，缺乏连贯的、指令驱动的推理和基础。

### 目的

引入细粒度3D具身推理任务，要求智能体根据任务指令，预测3D场景中每个被引用的功能元素的结构化三元组，包括空间位置、运动类型和运动轴。

### 方法

提出AffordBot框架，将多模态大语言模型与定制的思维链推理范式相结合。通过渲染场景的环绕视图并将3D元素候选投影到这些视图中，弥合3D输入与2D兼容的多模态大语言模型之间的差距。思维链管道从主动感知阶段开始，提示模型选择信息量最大的视点，然后逐步推理以定位功能元素并推断可能的交互运动。

### 主要发现

在SceneFun3D数据集上评估，AffordBot实现了最先进的性能，仅使用3D点云输入和多模态大语言模型就展示了强大的泛化能力和物理基础推理。

### 结论

AffordBot框架在细粒度3D具身推理任务上表现出色，证明了结合多模态大语言模型与思维链推理的有效性。

### 翻译

在物理环境中实现有效的人与智能体协作不仅需要理解要作用于什么，还需要理解可操作元素的位置以及如何与之交互。现有方法通常在物体级别操作或不连贯地处理细粒度的功能推理，缺乏连贯的、指令驱动的推理和基础。在这项工作中，我们引入了一个新任务：细粒度3D具身推理，要求智能体根据任务指令，预测3D场景中每个被引用的功能元素的结构化三元组，包括其空间位置、运动类型和运动轴。为了解决这个任务，我们提出了AffordBot，一个将多模态大语言模型与定制的思维链推理范式相结合的新颖框架。为了弥合3D输入与2D兼容的多模态大语言模型之间的差距，我们渲染场景的环绕视图并将3D元素候选投影到这些视图中，形成与场景几何对齐的丰富视觉表示。我们的思维链管道从主动感知阶段开始，提示多模态大语言模型根据指令选择信息量最大的视点，然后继续进行逐步推理以定位功能元素并推断可能的交互运动。在SceneFun3D数据集上评估，AffordBot实现了最先进的性能，证明了仅使用3D点云输入和多模态大语言模型就具有强大的泛化能力和物理基础推理。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决智能体在3D物理环境中理解可操作元素及其交互方式的问题。现有方法通常停留在物体级别识别或孤立地处理细粒度推理，缺乏连贯的指令驱动的定位和推理能力。这个问题在现实中很重要，因为它关系到智能体能否有效执行复杂任务，比如'拔出圣诞树插头'或'打开带花瓶的木柜底层抽屉'，需要智能体不仅识别物体，还要理解具体哪个部分可操作以及如何操作。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先引入了细粒度3D具身推理的新任务，要求智能体基于任务指令预测每个引用的可供性元素的结构化三元组。他们借鉴了多模态大语言模型(MLLMs)的推理能力和思维链(CoT)推理范式，但进行了定制化设计。为了弥合3D输入和2D兼容的MLLM之间的差距，他们设计了环绕视图渲染和3D元素投影技术。作者还参考了3D场景理解、物体识别和可供性理解领域的现有工作，如SceneFun3D数据集和相关方法，并在其基础上进行了改进和创新。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过整合3D几何信息与MLLM的推理能力，实现基于自然语言指令的3D可供性定位和运动估计的统一框架。整体流程分为两部分：1) 构建整体多模态表示，包括渲染3D点云的环绕视图图像、提取几何-语义描述符、建立3D到2D的关联；2) 定制思维链推理过程，包括主动感知阶段(选择信息量最大的视角)、可供性定位(在场景中定位目标部分)和交互推理(预测运动类型和轴方向)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 引入细粒度3D具身推理的新任务表述；2) 提出AffordBot框架，通过整体多模态表示和定制思维链过程整合3D感知和MLLM推理；3) 在SceneFun3D数据集上实现最先进结果。相比之前的工作，AffordBot不依赖视频输入而是直接在3D点云上操作，避免了视频处理的高计算开销；构建了环绕视图和3D元素投影提供密集视觉上下文；设计了主动感知阶段使MLLM能自主选择最佳视角；将可供性定位和交互推理统一在连贯的推理管道中，而之前方法孤立处理这些子任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AffordBot通过整合3D场景理解与多模态大语言模型的推理能力，实现了基于自然语言指令的细粒度3D可供性元素定位和交互运动预测，显著提升了智能体在复杂物理环境中执行任务的能力。'}


### 论文摘要

Effective human-agent collaboration in physical environments requires understanding not only what to act upon, but also where the actionable elements are and how to interact with them. Existing approaches often operate at the object level or disjointedly handle fine-grained affordance reasoning, lacking coherent, instruction-driven grounding and reasoning. In this work, we introduce a new task: Fine-grained 3D Embodied Reasoning, which requires an agent to predict, for each referenced affordance element in a 3D scene, a structured triplet comprising its spatial location, motion type, and motion axis, based on a task instruction. To solve this task, we propose AffordBot, a novel framework that integrates Multimodal Large Language Models (MLLMs) with a tailored chain-of-thought (CoT) reasoning paradigm. To bridge the gap between 3D input and 2D-compatible MLLMs, we render surround-view images of the scene and project 3D element candidates into these views, forming a rich visual representation aligned with the scene geometry. Our CoT pipeline begins with an active perception stage, prompting the MLLM to select the most informative viewpoint based on the instruction, before proceeding with step-by-step reasoning to localize affordance elements and infer plausible interaction motions. Evaluated on the SceneFun3D dataset, AffordBot achieves state-of-the-art performance, demonstrating strong generalization and physically grounded reasoning with only 3D point cloud input and MLLMs.

---

## 309. HCC-3D: Hierarchical Compensatory Compression for 98% 3D Token Reduction in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.09883v1](http://arxiv.org/abs/2511.09883v1)

**作者:** Liheng Zhang, Jin Wang, Hui Li, Bingfeng Zhang, Weifeng Liu

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出了一种名为HCC-3D的分层补偿压缩方法，用于解决3D视觉-语言模型中3D tokens计算成本高的问题。通过全局结构压缩和自适应细节挖掘两个模块，该方法能够在保持关键信息的同时实现高达98%的压缩率，并达到新的最先进性能。

### 背景

3D理解最近受到广泛关注，利用视觉-语言模型实现点云和文本数据之间的多模态推理。当前3D-VLMs直接将3D点云嵌入到3D tokens中，遵循大型2D-VLMs的框架，但这种框架计算成本高，限制了其应用。

### 目的

减少3D tokens带来的计算开销，同时保持其基本信息完整性，从而提高3D视觉-语言模型的效率。

### 方法

提出分层补偿压缩(HCC-3D)方法，包括两个主要模块：1)全局结构压缩(GSC)：设计全局查询将所有3D tokens压缩为几个关键tokens，同时保持整体结构信息；2)自适应细节挖掘(ADM)：通过互补评分选择性地重新压缩显著但未被充分关注的功能，以补偿GSC中的信息损失。

### 主要发现

HCC-3D实现了极高的压缩率(约98%)，相比之前的3D-VLMs有显著提升；同时达到了新的最先进性能，在效率和性能上都显示出显著改进。

### 结论

HCC-3D有效地解决了3D视觉-语言模型中3D tokens计算成本高的问题，在保持性能的同时大幅提高了效率，为3D理解应用提供了更实用的解决方案。

### 翻译

3D理解最近引起了广泛关注，利用视觉-语言模型(VLMs)来实现点云和文本数据之间的多模态推理。当前的3D-VLMs直接将3D点云嵌入到3D tokens中，遵循具有强大推理能力的大型2D-VLMs。然而，这种框架有很大的计算成本，限制了其应用，我们确定瓶颈在于处理大型语言模型(LLM)部分中的所有3D tokens。这引发了一个问题：我们如何减少3D tokens引入的计算开销，同时保持其基本信息的完整性？为解决这个问题，我们引入了分层补偿压缩(HCC-3D)，以在保持关键细节保留的同时高效压缩3D tokens。具体来说，我们首先提出了全局结构压缩(GSC)，在其中我们设计了全局查询，将所有3D tokens压缩为几个关键tokens，同时保持整体结构信息。然后，为了补偿GSC中的信息损失，我们进一步提出了自适应细节挖掘(ADM)模块，通过互补评分选择性地重新压缩显著但未被充分关注的功能。大量实验证明，与之前的3D-VLMs相比，HCC-3D不仅实现了极高的压缩率(约98%)，而且达到了新的最先进性能，显示出在效率和性能方面的巨大改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D视觉语言模型(3D-VLMs)中处理3D点云数据时的计算效率问题。当前方法将3D点云直接嵌入为大量3D tokens送入大型语言模型，导致巨大计算开销(占推理时间90%以上)，限制了3D-VLMs的实际应用。这个问题很重要，因为3D理解在机器人、自动驾驶、AR/VR等领域有广泛应用，但高昂的计算成本使得现有方法难以在资源受限场景中部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先确定计算瓶颈在于LLM处理大量3D tokens，然后考虑点云数据的不规则分布特性导致信息密度不均匀。他们设计了分层压缩策略，认识到简单全局压缩会丢失重要细节，因此提出互补机制。该方法借鉴了视觉特征压缩领域的空间注意力机制、特征金字塔等技术，以及Transformer中的多头注意力机制，但针对3D点云特性进行了专门改进。同时利用了预训练的3D编码器如Point-BERT来提取点云特征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是分层补偿压缩(HCC-3D)，通过两个互补模块实现高效压缩：全局结构压缩(GSC)使用可学习的3D空间查询将所有3D tokens压缩为少量关键tokens保留整体结构；自适应细节挖掘(ADM)通过互补评分机制选择性重新压缩被全局压缩忽视的重要特征。整体流程：1)点云通过3D编码器生成特征；2)特征通过MLP投影；3)GSC模块使用全局查询和多头注意力压缩特征；4)ADM模块计算注意力覆盖度和特征重要性，选择重要特征并用细节查询压缩；5)全局和细节特征拼接形成最终表示；6)送入LLM处理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)实现98%的3D token压缩(从513减到12)同时保持甚至提高性能；2)设计GSC和ADM互补模块分别关注全局结构和局部细节；3)通过注意力引导选择机制动态识别被忽视的重要区域；4)高效训练(单GPU仅需11.9小时)。相比之前工作不同：现有方法要么简单采样导致性能下降，要么保留大量tokens导致计算开销大；HCC-3D通过互补机制确保关键信息不丢失，同时将推理速度从0.45秒/任务提高到0.36秒/任务，并在多个基准测试上达到新的state-of-the-art结果。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HCC-3D通过分层互补压缩架构实现了98%的3D token减少，在大幅提升计算效率的同时，在多个3D理解任务上达到了新的state-of-the-art性能，解决了3D视觉语言模型中计算效率与性能的关键权衡问题。'}


### 论文摘要

3D understanding has drawn significant attention recently, leveraging Vision-Language Models (VLMs) to enable multi-modal reasoning between point cloud and text data. Current 3D-VLMs directly embed the 3D point clouds into 3D tokens, following large 2D-VLMs with powerful reasoning capabilities. However, this framework has a great computational cost limiting its application, where we identify that the bottleneck lies in processing all 3D tokens in the Large Language Model (LLM) part. This raises the question: how can we reduce the computational overhead introduced by 3D tokens while preserving the integrity of their essential information? To address this question, we introduce Hierarchical Compensatory Compression (HCC-3D) to efficiently compress 3D tokens while maintaining critical detail retention. Specifically, we first propose a global structure compression (GSC), in which we design global queries to compress all 3D tokens into a few key tokens while keeping overall structural information. Then, to compensate for the information loss in GSC, we further propose an adaptive detail mining (ADM) module that selectively recompresses salient but under-attended features through complementary scoring. Extensive experiments demonstrate that HCC-3D not only achieves extreme compression ratios (approximately 98%) compared to previous 3D-VLMs, but also achieves new state-of-the-art performance, showing the great improvements on both efficiency and performance.

---

## 310. AHA! Animating Human Avatars in Diverse Scenes with Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2511.09827v1](http://arxiv.org/abs/2511.09827v1)

**作者:** Aymen Mir, Jian Wang, Riza Alp Guler, Chuan Guo, Gerard Pons-Moll, Bing Zhou

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出了一种使用3D高斯飞溅技术进行人体动画的新颖框架，实现了与3D场景交互的人体的几何一致自由视角渲染。

### 背景

3D高斯飞溅是一种神经场景表示方法，在新型视图合成方面取得了先进结果，但在人体-场景动画和交互方面的应用研究不足。

### 目的

开发一个使用3DGS技术在3D场景中实现人体动画的新框架。

### 方法

将人和场景表示为高斯分布，使渲染与运动合成解耦，使用高斯对齐的运动模块和人体-场景高斯细化优化来确保自然交互。

### 主要发现

该方法在多种场景和人体捕获数据上有效，并支持编辑单目RGB视频添加新动画人体的几何一致自由视角渲染。

### 结论

3DGS技术为人体在3D场景中的动画和交互提供了新可能性，特别是在几何一致的自由视角渲染方面具有优势。

### 翻译

我们提出了一种使用3D高斯飞溅技术对3D场景中的人体进行动画的新颖框架，3D高斯飞溅是一种神经场景表示方法，最近在新型视图合成方面取得了最先进的光照真实感结果，但在人体-场景动画和交互方面的应用研究还不够。与使用网格或点云作为底层3D表示的现有动画流程不同，我们的方法将3DGS作为3D表示引入到场景中的人体动画问题中。通过将人和场景表示为高斯分布，我们的方法允许对与3D场景交互的人体进行几何一致的自由视角渲染。我们的关键见解是渲染可以从运动合成中解耦，每个子问题可以独立解决，无需配对的人体-场景数据。我们方法的核心是一个高斯对齐的运动模块，它使用基于不透明度的线索和投影的高斯结构来指导人体放置和姿势对齐，而无需显式场景几何。为了确保自然的交互，我们进一步提出了一个人体-场景高斯细化优化，以强制实现真实的接触和导航。我们在Scannet++和SuperSplat库的场景以及从稀疏和密集多视角人体捕获重建的头像上评估了我们的方法。最后，我们证明了我们的框架允许新的应用，例如具有新动画人体的编辑单目RGB视频的几何一致自由视角渲染，展示了3DGS在基于单目视频的人体动画方面的独特优势。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何使用3D高斯散射技术在3D场景中实现逼真的人体动画问题。这个问题在现实中非常重要，因为游戏、影视、虚拟现实等领域需要真实感强的人物与场景交互效果。在研究中也很重要，因为传统方法使用网格或点云表示难以实现照片级真实感渲染，而且从单目视频重建场景并添加人物动画仍然是一个挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于两个关键洞察设计方法：1) 渲染与运动合成可以解耦，可以独立重建人体和场景；2) 即使3DGS不提供封闭表面，其不透明度场和投影的高斯结构也足以指导人体放置。作者借鉴了现有3DGS重建技术、SMPL人体模型、基于潜在变量的运动合成框架和强化学习方法，将这些技术创新性地整合应用到人体-场景动画问题中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用3D高斯散射作为统一的3D表示方法同时表示人体和场景，实现照片级真实感的人体-场景动画。整体流程分为三阶段：1) 高斯重建 - 用标准3DGS重建场景，学习可动画的人体高斯表示；2) 高斯对齐的运动合成 - 使用强化学习导航场景，确定性优化精细动作；3) 可微分接触优化 - 检测接触帧并优化人体高斯位置，确保自然接触和减少穿透。最后组合人体与场景高斯，从任意视角渲染生成真实交互视频。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 首将3DGS引入人体动画领域；2) 实现渲染与运动合成的解耦；3) 开发高斯对齐的运动模块；4) 提出人体-场景高斯优化方法；5) 实现单目视频编辑应用。相比传统方法，本文使用3D高斯散射而非网格/点云表示，无需成对人体-场景数据，支持单目视频重建并添加人物，能生成照片级真实感渲染效果。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次将3D高斯散射技术应用于人体-场景动画，实现了照片级真实感的人物与复杂环境交互，并支持从单目视频重建场景并添加新动画人物的创新应用。'}


### 论文摘要

We present a novel framework for animating humans in 3D scenes using 3D Gaussian Splatting (3DGS), a neural scene representation that has recently achieved state-of-the-art photorealistic results for novel-view synthesis but remains under-explored for human-scene animation and interaction. Unlike existing animation pipelines that use meshes or point clouds as the underlying 3D representation, our approach introduces the use of 3DGS as the 3D representation to the problem of animating humans in scenes. By representing humans and scenes as Gaussians, our approach allows for geometry-consistent free-viewpoint rendering of humans interacting with 3D scenes. Our key insight is that the rendering can be decoupled from the motion synthesis and each sub-problem can be addressed independently, without the need for paired human-scene data. Central to our method is a Gaussian-aligned motion module that synthesizes motion without explicit scene geometry, using opacity-based cues and projected Gaussian structures to guide human placement and pose alignment. To ensure natural interactions, we further propose a human-scene Gaussian refinement optimization that enforces realistic contact and navigation. We evaluate our approach on scenes from Scannet++ and the SuperSplat library, and on avatars reconstructed from sparse and dense multi-view human capture. Finally, we demonstrate that our framework allows for novel applications such as geometry-consistent free-viewpoint rendering of edited monocular RGB videos with new animated humans, showcasing the unique advantage of 3DGS for monocular video-based human animation.

---

## 311. TIME Commissioning Observations: I. Mapping Dust and Molecular Gas in the Sgr A Molecular Cloud Complex at the Galactic Center

**论文链接:** [http://arxiv.org/abs/2511.09473v1](http://arxiv.org/abs/2511.09473v1)

**作者:** Selina F. Yang, Sophie M. McAtee, Benjamin J. Vaughan, Abigail T. Crites, Victoria L. Butler, Dongwoo T. Chung, Ryan P. Keenan, Dang Pham, James J. Bock, Charles M. Bradford, Tzu-Ching Chang, Yun-Ting Cheng, Audrey Dunn, Nicholas Emerson, Clifford Frez, Jonathon Hunacek, Chao-Te Li, Ian N. Lowe, King Lau, Daniel P. Marrone, Evan C. Mayer, Guochao Sun, Isaac Trumper, Anthony D. Turner, Ta-Shun Wei, Michael Zemcov

**发布时间:** 2025-11-12

**备注:** 17 pages + 5 pages of bibliography and appendices. 14 figures, 3 tables. To be submitted to ApJ

### GPT解析

### 总结

研究团队使用TIME望远镜对Sagittarius A进行了观测，验证了其在复杂银河系场中恢复连续谱和谱线信号的能力，为未来的线强度映射和银河外调查做准备。

### 背景

TIME望远镜在2021-2022年进行调试运行，旨在验证其超光谱成像能力，为未来的线强度映射做准备。

### 目的

验证TIME仪器在复杂银河系场中恢复连续谱和谱线信号的能力，确认其进行银河外CO和[C II]调查的准备工作就绪。

### 方法

使用木星观测校准探测器增益和指向偏移，通过专门构建的管道处理Sgr A观测，采用相关加权的共同模式减法去除相关噪声，并使用地图域主成分分析识别系统误差。

### 主要发现

获得了频率分辨地图，检测到强烈的12CO(2-1)和13CO(2-1)发射，以及可以区分核盘自由-自由发射和分子云热尘埃发射的连续谱成分；与BGPS调查结果一致性约为5%；估计分子氢质量在54万到57万太阳质量之间，与先前研究一致。

### 结论

TIME能够在复杂银河系场中恢复连续谱和谱线信号，验证了其即将进行的银河外CO和[C II]调查的准备工作就绪。

### 翻译

我们展示了使用Tomographic Ionized-carbon Mapping Experiment (TIME)对Sagittarius A (Sgr A)进行的观测处理，这是2021-2022年调试运行的一部分，用于验证TIME的超光谱成像能力，为未来的线强度映射做准备。使用木星观测来校准探测器增益和指向偏移，我们在专门构建的管道中处理Sgr A观测，通过相关加权的共同模式减法去除相关噪声，并使用地图域主成分分析来识别进一步的系统误差。 resulting频率分辨地图恢复了强烈的12CO(2-1)和13CO(2-1)发射，以及一个连续谱成分，其谱指数可以区分核盘(CND)中的自由-自由发射和20公里每秒和50公里每秒分子云中的热尘埃发射。与Bolocam银河平面调查(BGPS)的宽带连续通量比较显示，在Sgr A区域的高信噪比分子云中一致性约为5%。从CO线检测中，我们估计分子氢质量在54万到57万太阳质量之间，与先前研究一致。这些结果表明TIME能够在复杂银河系场中恢复连续谱和谱线信号，验证了其即将进行的银河外CO和[C II]调查的准备工作就绪。


### 论文摘要

We present the processing of an observation of Sagittarius A (Sgr A) with the Tomographic Ionized-carbon Mapping Experiment (TIME), part of the 2021-2022 commissioning run to verify TIME's hyperspectral imaging capabilities for future line-intensity mapping. Using an observation of Jupiter to calibrate detector gains and pointing offsets, we process the Sgr A observation in a purpose-built pipeline that removes correlated noise through common-mode subtraction with correlation-weighted scaling, and uses map-domain principal component analysis to identify further systematic errors. The resulting frequency-resolved maps recover strong 12CO(2-1) and 13CO(2-1) emission, and a continuum component whose spectral index discriminates free-free emission in the circumnuclear disk (CND) versus thermal dust emission in the 20 km s$^{-1}$ and 50 km s$^{-1}$ molecular clouds. Broadband continuum flux comparisons with the Bolocam Galactic Plane Survey (BGPS) show agreement to within $\sim$5% in high-SNR molecular clouds in the Sgr A region. From the CO line detections, we estimate a molecular hydrogen mass of between $5.4 \times 10^5 M_\odot$ and $5.7 \times 10^5 M_\odot$, consistent with prior studies. These results demonstrate TIME's ability to recover both continuum and spectral-line signals in complex Galactic fields, validating its readiness for upcoming extragalactic CO and [C II] surveys.

---

## 312. UMIGen: A Unified Framework for Egocentric Point Cloud Generation and Cross-Embodiment Robotic Imitation Learning

**论文链接:** [http://arxiv.org/abs/2511.09302v1](http://arxiv.org/abs/2511.09302v1)

**作者:** Yan Huang, Shoujie Li, Xingting Li, Wenbo Ding

**发布时间:** 2025-11-12

### GPT解析

### 总结

UMIGen是一种统一框架，通过Cloud-UMI设备和可见性感知优化机制，解决了数据驱动机器人学习中高质量数据收集的挑战，实现了高效的跨机器人平台数据生成和泛化。

### 背景

数据驱动的机器人学习面临困境：稳健策略需要大规模高质量演示数据，但收集此类数据成本高、依赖专用硬件且当前方法空间泛化能力有限。

### 目的

开发一种能够高效生成高质量数据并支持跨机器人平台泛化的框架，解决数据收集难题。

### 方法

提出UMIGen框架，包含两个关键组件：(1)Cloud-UMI手持设备，无需视觉SLAM同时记录点云观察-动作对；(2)可见性感知优化机制，仅生成相机视野内的点，扩展到以自我为中心的3D观察。

### 主要发现

UMIGen能够生成与真实以自我为中心观察对齐的数据，可直接转移到不同机器人平台无需后处理；在模拟和真实环境中支持强大的跨embodiment泛化，加速了各种操作任务的数据收集。

### 结论

UMIGen通过创新的数据收集方法和优化机制，有效解决了机器人学习中数据收集的挑战，提高了跨平台泛化能力。

### 翻译

数据驱动的机器人学习面临明显困境：稳健的策略需要大规模、高质量的演示数据，但收集此类数据仍然是一个重大挑战，这是由于高操作成本、对专用硬件的依赖以及当前方法的有限空间泛化能力。通用操作接口(UMI)放宽了数据收集的严格硬件要求，但它仅限于捕捉场景的RGB图像，省略了许多任务所依赖的3D几何信息。受DemoGen启发，我们提出了UMIGen，一个统一框架，包含两个关键组件：(1)Cloud-UMI，一种手持式数据收集设备，不需要视觉SLAM，同时记录点云观察-动作对；(2)一种可见性感知的优化机制，通过仅生成相机视野内的点，将DemoGen管道扩展到以自我为中心的3D观察。这两个组件能够实现与真实以自我为中心观察对齐的高效数据生成，并且可以直接转移到不同的机器人embodiment，无需任何后处理。在模拟和真实世界环境中的实验表明，UMIGen支持强大的跨embodiment泛化，并加速了各种操作任务中的数据收集。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人学习中的数据收集挑战：需要大规模高质量演示数据来训练稳健策略，但收集这类数据成本高昂、依赖专业硬件，且当前方法的泛化能力有限。这个问题重要是因为数据收集成本限制了机器人学习的广泛应用，缺少3D几何信息限制了机器人在复杂任务中的表现，现有方法难以跨平台部署和泛化到不同机器人形态。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受DemoGen启发，注意到其全场景和静态视点假设在腕式自我中心设置中不适用。借鉴了UMI和FastUMI等便携式管道的设计理念，但增加了点云收集能力。设计了Cloud-UMI手持设备集成深度传感器和跟踪相机，并扩展了DemoGen的演示生成管道，引入可见性感知优化(VAO)机制来处理腕式相机视场限制问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建低成本手持数据收集设备(Cloud-UMI)记录点云观察-动作对，并通过可见性感知优化(VAO)机制只生成相机视场内的点，使生成的数据与真实自我中心观察一致，可直接跨不同机器人形态转移。流程包括：1)使用Cloud-UMI收集RGB-D图像并重建点云；2)通过变换链将点云从相机坐标系转换到机器人基座坐标系；3)将源演示分割为技能段和运动段；4)对技能段应用空间变换，对运动段重新规划；5)使用VAO过滤视场外点，生成符合实际观察的合成数据。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)Cloud-UMI手持设备无需机器人硬件或视觉SLAM即可记录配对点云观察和动作；2)可见性感知优化(VAO)机制处理自我中心观察中的遮挡问题，生成符合实际观察的合成数据；3)实现了跨形态泛化能力，可直接在不同机器人间迁移而无需后处理。相比之前工作的不同：UMI和FastUMI只收集RGB图像而UMIGen收集3D点云；DemoGen假设全场景可见而UMIGen处理腕式设置中的部分可见性；需要固定平台的3D收集系统而UMIGen提供便携式解决方案。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'UMIGen通过低成本手持设备和可见性感知优化机制，实现了高效的自我中心点云数据收集和跨形态机器人模仿学习，解决了机器人学习中的数据瓶颈问题。'}


### 论文摘要

Data-driven robotic learning faces an obvious dilemma: robust policies demand large-scale, high-quality demonstration data, yet collecting such data remains a major challenge owing to high operational costs, dependence on specialized hardware, and the limited spatial generalization capability of current methods. The Universal Manipulation Interface (UMI) relaxes the strict hardware requirements for data collection, but it is restricted to capturing only RGB images of a scene and omits the 3D geometric information on which many tasks rely. Inspired by DemoGen, we propose UMIGen, a unified framework that consists of two key components: (1) Cloud-UMI, a handheld data collection device that requires no visual SLAM and simultaneously records point cloud observation-action pairs; and (2) a visibility-aware optimization mechanism that extends the DemoGen pipeline to egocentric 3D observations by generating only points within the camera's field of view. These two components enable efficient data generation that aligns with real egocentric observations and can be directly transferred across different robot embodiments without any post-processing. Experiments in both simulated and real-world settings demonstrate that UMIGen supports strong cross-embodiment generalization and accelerates data collection in diverse manipulation tasks.

---

## 313. OG-PCL: Efficient Sparse Point Cloud Processing for Human Activity Recognition

**论文链接:** [http://arxiv.org/abs/2511.08910v1](http://arxiv.org/abs/2511.08910v1)

**作者:** Jiuqi Yan, Chendong Xu, Dongyu Liu

**发布时间:** 2025-11-12

### GPT解析

### 总结

本文提出了一种基于毫米波雷达的人类活动识别方法，通过OG-PCL网络处理稀疏3D雷达点云，实现了高精度且轻量级的活动识别。

### 背景

传统人类活动识别(HAR)主要依赖摄像头和可穿戴设备，但这些方法存在隐私问题。毫米波雷达提供了一种保护隐私且稳健的替代方案。

### 目的

开发一种能够处理毫米波雷达产生的稀疏3D点云的轻量级网络，用于高效准确的人类活动识别。

### 方法

提出占用门控并行CNN双向长短期记忆网络(OG-PCL)，参数大小仅0.83M，并引入占用门控卷积(OGConv)块，采用三视图并行结构保留三维空间信息。

### 主要发现

在RadHAR数据集上达到91.75%的准确率，优于2D CNN、PointNet和3D CNN等基线方法；消融研究验证了三视图并行结构在保留空间信息方面的优势；证明了占用补偿机制对处理稀疏点云的必要性。

### 结论

OG-PCL为轻量级平台上的实时雷达HAR提供了一个紧凑而准确的框架。

### 翻译

基于毫米波雷达的人类活动识别(HAR)提供了一种保护隐私且稳健的替代方案，优于基于摄像头和可穿戴设备的方法。在本工作中，我们提出了占用门控并行CNN双向长短期记忆(OG-PCL)网络来处理毫米波传感产生的稀疏3D雷达点云。针对轻量级部署设计，所提出的OG-PCL参数大小仅为0.83M，在RadHAR数据集上达到91.75%的准确率，优于现有的2D CNN、PointNet和3D CNN等基线方法。通过消融研究，我们验证了三视图并行结构在保留三维空间信息方面的优势，同时保持效率。我们进一步引入了占用门控卷积(OGConv)模块，并证明了其占用补偿机制对于处理稀疏点云的必要性。因此，所提出的OG-PCL为轻量级平台上的实时雷达HAR提供了一个紧凑而准确的框架。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何高效处理毫米波雷达产生的稀疏3D点云数据用于人类活动识别（HAR）的问题。这个问题在现实中很重要，因为毫米波雷达提供了一种隐私保护且环境鲁棒的替代方案，不受光照和遮挡影响，特别适合隐私敏感场景和轻量级部署。然而，现有方法往往依赖密集表示或大型网络，难以在资源受限设备上实时运行，且对稀疏点云的研究不足。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，包括对密集数据的依赖、计算复杂度高、无法有效处理稀疏数据等。他们借鉴了PointNet和3D CNN等点云处理方法，Bi-LSTM用于时序建模，以及部分卷积（partial convolutions）的概念。针对稀疏数据处理，作者设计了OGConv模块；为保留空间信息同时降低计算复杂度，采用了三视图投影策略；为轻量级部署，优化了网络结构，最终形成了OG-PCL框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过三视图投影将3D稀疏点云转换为2表示，使用专门设计的OGConv模块处理稀疏性，并通过并行CNN和Bi-LSTM进行特征提取和时序建模。整体流程：1) 将雷达点云体素化为10×32×32网格；2) 沿三个正交轴投影成2D视图；3) 每个视图通过OGConv-based CNN处理；4) 融合三个视图特征；5) 将特征序列输入Bi-LSTM进行时序建模；6) 使用LSTM输出进行分类。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 三视图投影策略，保留三维空间信息同时降低计算成本；2) 占用门控卷积（OGConv）模块，通过掩码机制和补偿因子处理稀疏数据；3) 轻量级设计（仅0.83M参数）。相比之前工作的不同：优于2D CNN方法（保留3D信息）；比PointNet更适合雷达稀疏数据；比3D CNN计算效率更高；比Transformer-based方法参数量显著减少；比TD-CNN + LSTM准确率高5%以上且保留了更多空间信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OG-PCL通过创新的三视图投影和占用门控卷积机制，实现了在轻量级网络架构下高效处理稀疏雷达点云并进行高精度人类活动识别，为隐私保护和非接触式传感提供了实用解决方案。'}


### 论文摘要

Human activity recognition (HAR) with millimeter-wave (mmWave) radar offers a privacy-preserving and robust alternative to camera- and wearable-based approaches. In this work, we propose the Occupancy-Gated Parallel-CNN Bi-LSTM (OG-PCL) network to process sparse 3D radar point clouds produced by mmWave sensing. Designed for lightweight deployment, the parameter size of the proposed OG-PCL is only 0.83M and achieves 91.75 accuracy on the RadHAR dataset, outperforming those existing baselines such as 2D CNN, PointNet, and 3D CNN methods. We validate the advantages of the tri-view parallel structure in preserving spatial information across three dimensions while maintaining efficiency through ablation studies. We further introduce the Occupancy-Gated Convolution (OGConv) block and demonstrate the necessity of its occupancy compensation mechanism for handling sparse point clouds. The proposed OG-PCL thus offers a compact yet accurate framework for real-time radar-based HAR on lightweight platforms.

---

## 314. Model of super-Alfvénic MHD turbulence and structure functions of polarization

**论文链接:** [http://arxiv.org/abs/2511.08800v1](http://arxiv.org/abs/2511.08800v1)

**作者:** A. Lazarian, D. Pogosyan, Y. Hu

**发布时间:** 2025-11-11

**备注:** 24 pages, 10 figures, submitted to the Astrophysical Journal

### GPT解析

### 总结

该研究提供了超阿尔文速度MHD湍流中偏振角结构函数谱斜率变化的解析框架，并通过数值模拟验证了预测，发现该结构函数随阿尔文马赫数增加而变浅，成为研究超阿尔文速度湍流的有力工具。

### 背景

超阿尔文速度的MHD湍流广泛存在于天体物理环境中，包括星系团和分子云。对于这类湍流的统计研究，偏振角结构函数D^φ(R)=⟨sin²(φ₁-φ₂)⟩被用于测量天空中投影距离R处的两点之间的偏振角。

### 目的

提供一个解析框架来解释超阿尔文速度湍流中偏振角结构函数D^φ(R)谱斜率的修改，并通过数值模拟验证预测。

### 方法

通过数值模拟验证理论预测，探索偏振度的结构函数和偏振方向的谱（D^φ的傅里叶变换）。

### 主要发现

对于超阿尔文速度湍流，结构函数D^φ(R)随着阿尔文马赫数(MA)的增加而变浅。这使得D^φ(R)成为超阿尔文速度湍流的一个有价值的诊断工具，并开辟了从观测中获得MA的途径。

### 结论

研究结果对星系团和星际介质中的湍流和磁场研究具有重要意义，为理解这些环境中的磁场特性提供了新方法。

### 翻译

超过阿尔文速度驱动的MHD湍流，即超阿尔文速度湍流，广泛存在于天体物理环境中，包括星系团和分子云。对于此类湍流的统计研究，我们探索了偏振角结构函数D^φ(R)=⟨sin²(φ₁-φ₂)⟩的实用性，其中φ表示在天空中投影距离R处测量的两点之间的偏振角。Lazarian、Yuen和Pogosyan(2022)表明，在超阿尔文速度湍流的情况下，D^φ(R)的谱斜率与底层磁波动的谱斜率不同，限制了其在已知技术中用于估计磁场强度的适用性。在本工作中，我们提供了一个解析框架来解释超阿尔文速度湍流中D^φ(R)谱斜率的修改，并通过数值模拟验证了我们的预测。我们证明，对于超阿尔文速度湍流，结构函数D^φ(R)随着MA的增加而变浅。我们的研究使D^φ(R)成为超阿尔文速度湍流的一个有价值的诊断工具，并开辟了从观测中获得MA的途径。我们还通过数值模拟探索了偏振度的结构函数和偏振方向的谱（后者是D^φ的傅里叶变换）。我们讨论了我们的发现对星系团和星际介质中湍流和磁场研究的意义。


### 论文摘要

MHD turbulence driven at velocities higher than the Alfvén velocity, i.e., super-Alfvénic turbulence, is widely spread in astrophysical environments, including galaxy clusters and molecular clouds. For statistical studies of such turbulence, we explore the utility of the polarization angle structure functions $D^φ(R)= \left\langle\sin^2(φ_1-φ_2) \right\rangle$, where $φ$ denotes the polarization angle measured at points separated by a projected distance $\mathbf{R}$ on the plane of the sky. Lazarian, Yuen and Pogosyan, 2022, showed that in the case of super-Alfvénic turbulence, the spectral slope of $D^φ(\mathbf{R})$ differs from that of the underlying magnetic fluctuations, limiting its applicability for field strength estimation with known techniques. In this work, we provide an analytical framework that explains the modification of the $D^φ(R)$ spectral slope in super-Alfvénic turbulence and validate our predictions with numerical simulations. We demonstrate that for super-Alfvénic turbulence, the structure function $D^φ(R)$ gets shallower with the increase of $M_A$. Our study makes $D^φ(R)$ a valuable diagnostic of super-Alfvénic turbulence and opens a way to obtain $M_A$ from observations. We also explore numerically the structure function of the polarization degree and the spectrum of the polarization directions, the latter being the Fourier transform of $D^φ$. We discuss the implications of our findings for turbulence and magnetic field studies in the intracluster and interstellar media.

---

## 315. Accurate and Efficient Surface Reconstruction from Point Clouds via Geometry-Aware Local Adaptation

**论文链接:** [http://arxiv.org/abs/2511.08233v1](http://arxiv.org/abs/2511.08233v1)

**作者:** Eito Ogawa, Taiga Hayami, Hiroshi Watanabe

**发布时间:** 2025-11-11

**备注:** 4 pages

### GPT解析

### 总结

本研究提出了一种基于点云曲率自适应调整局部区域间距和大小的点云表面重建方法，以提高重建的准确性和效率。

### 背景

随着深度学习的进步，点云表面重建的准确性得到了提高，使其能够应用于基础设施检查等领域。最近从局部区域重建而非整个点云的方法因其强大的泛化能力而受到关注。

### 目的

提高点云表面重建的准确性和效率，增强方法对不同几何复杂度的适应性。

### 方法

提出一种根据输入点云曲率自适应调制局部区域间距和大小的重建方法。

### 主要发现

通过自适应调整局部区域的间距和大小，可以改善重建的准确性和效率，增强对几何复杂度变化的适应性。

### 结论

自适应调制局部区域的间距和大小是一种有效的点云表面重建策略，能够提高重建质量和效率。

### 翻译

点云表面重建随着深度学习的进步提高了准确性，使基础设施检查等应用成为可能。最近从小的局部区域而非整个点云进行重建的方法因其强大的泛化能力而受到关注。然而，先前的工作通常均匀放置局部区域并保持其大小固定，限制了其对几何复杂度变化的适应性。在本研究中，我们提出了一种方法，通过根据输入点云的曲率自适应调整局部区域的间距和大小，提高重建的准确性和效率。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云表面重建中局部区域固定大小和均匀分布导致的适应性问题。在高曲率区域，固定大小的局部区域会导致多个表面混合和点密度不足；在低曲率区域则存在不必要的计算开销。这个问题很重要，因为随着激光雷达技术普及，从点云生成3D模型的需求增长，准确的表面重建对基础设施检查等应用至关重要，且改进方法能同时提高精度和效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有方法LoSF-UDF的局限性，虽然其通过局部区域重建获得优秀质量，但固定大小的局部区域在高曲率区域表现不佳。作者借鉴了UDF(无符号距离场)概念，该方法不需要内外部分类，能处理非水密表面。同时保留了LoSF-UDF的局部区域处理思想，但改进了其固定区域大小的做法。设计时从三方面考虑自适应调整：局部区域半径、查询点放置策略和重采样策略，均基于曲率条件进行优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是根据输入点云的曲率自适应调整局部区域的间距和大小。高曲率区域使用较小局部区域和更高查询点密度避免特征混合；低曲率区域使用较大局部区域和较低查询点密度确保点密度同时减少计算。流程包括：1)初始放置128³查询点网格；2)计算每个区域曲率；3)根据曲率调整局部区域半径；4)高曲率区域局部细化到256³并添加查询点；5)对未评估点进行插值；6)根据曲率条件化重采样；7)估计UDF值；8)使用DCUDF提取网格。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)曲率感知的局部区域半径调整，根据曲率动态调整半径大小；2)两阶段查询点放置策略，初始使用128³网格，仅在高曲率区域细化到256³；3)曲率条件化的重采样策略，平滑区域添加质心样本，高曲率区域复制现有样本。相比LoSF-UDF的不同：1)LoSF-UDF使用固定大小局部区域，本文方法自适应调整；2)LoSF-UDF使用均匀256³网格，本文使用两阶段策略减少计算；3)本文根据曲率使用不同重采样方法；4)本文通过优化流程提高了计算效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于曲率自适应调整局部区域大小和查询点密度的点云表面重建方法，在提高重建精度的同时显著减少了计算成本。'}


### 论文摘要

Point cloud surface reconstruction has improved in accuracy with advances in deep learning, enabling applications such as infrastructure inspection. Recent approaches that reconstruct from small local regions rather than entire point clouds have attracted attention for their strong generalization capability. However, prior work typically places local regions uniformly and keeps their size fixed, limiting adaptability to variations in geometric complexity. In this study, we propose a method that improves reconstruction accuracy and efficiency by adaptively modulating the spacing and size of local regions based on the curvature of the input point cloud.

---

## 316. A Small Leak Sinks All: Exploring the Transferable Vulnerability of Source Code Models

**论文链接:** [http://arxiv.org/abs/2511.08127v1](http://arxiv.org/abs/2511.08127v1)

**作者:** Weiye Li, Wenyi Tang

**发布时间:** 2025-11-11

### GPT解析

### 总结

该研究解决了源代码模型(SCMs)中可转移漏洞这一被忽视的问题，特别是针对大型语言模型用于代码(LLM4Code)的情况。研究团队提出了HABITAT方法，能够在不访问下游分类器的情况下生成有效的对抗样本，并揭示了传统SCM和LLM4Code之间的潜在漏洞相关性。

### 背景

源代码模型能够从源代码中学习适当的嵌入表示，在软件工程和安全任务中取得显著成功。大型语言模型的爆炸性发展扩展了SCM家族，彻底改变了开发工作流程。然而，研究不同类型的SCM漏洞是AI驱动软件生态系统安全和可靠性的基石，但基本问题——可转移漏洞——仍然未被充分探索。现有研究既没有提供产生有效对抗样本的实际方法，也没有关注在现代软件开发平台中广泛使用的LLM4Code。

### 目的

系统研究传统SCM和LLM4Code的内在漏洞可转移性，并提出一种不依赖于特定受害者的方法来生成实用的对抗样本。

### 方法

设计了HABITAT系统，包括定制的扰动插入机制和分层强化学习框架，能够自适应选择最佳扰动而无需访问SCM的下游分类器。同时进行了SCM漏洞的内在可转移性分析，揭示传统SCM和LLM4Code之间的潜在漏洞相关性，并确定控制不依赖于特定受害者的转移攻击成功率的基本因素。

### 主要发现

传统SCM和LLM4Code之间存在潜在漏洞相关性；确定了控制不依赖于特定受害者的转移攻击成功率的基本因素；基于传统SCM构建的对抗样本对LLM4Code的攻击成功率高达64%，超过了最先进方法15%以上。

### 结论

SCM漏洞的研究结果强调了未来开发强大防御措施的关键焦点，为构建更安全的AI驱动软件生态系统提供了重要指导。

### 翻译

源代码模型从源代码中学习适当的嵌入表示，在各种软件工程或安全任务中显示出显著的成功。最近大型语言模型的爆炸性发展扩展了SCM家族，带来了用于代码的大型语言模型，彻底改变了开发工作流程。研究不同类型的SCM漏洞是AI驱动软件生态系统安全和可靠性的基石，然而，基本的可转移漏洞仍然未被充分探索。现有研究既没有提供产生有效对抗样本的实际方法，即需要访问SCM的下游分类器，也没有关注在现代软件开发平台和基于云的集成开发环境中广泛使用的LLM4Code。因此，这项工作系统地研究了传统SCM和LLM4Code的内在漏洞可转移性，并提出了一种不依赖于特定受害者的方法来生成实用的对抗样本。我们设计了HABITAT，包括定制的扰动插入机制和分层强化学习框架，能够自适应选择最佳扰动而无需访问SCM的下游分类器。此外，进行了SCM漏洞的内在可转移性分析，揭示了传统SCM和LLM4Code之间的潜在漏洞相关性，以及控制不依赖于特定受害者的转移攻击成功率的基本因素。这些关于SCM漏洞的研究结果强调了未来开发强大防御措施的关键焦点。实验评估表明，我们基于传统SCM构建的对抗样本对LLM4Code的攻击成功率高达64%，超过了最先进方法15%以上。


### 论文摘要

Source Code Model learn the proper embeddings from source codes, demonstrating significant success in various software engineering or security tasks. The recent explosive development of LLM extends the family of SCMs,bringing LLMs for code that revolutionize development workflows. Investigating different kinds of SCM vulnerability is the cornerstone for the security and trustworthiness of AI-powered software ecosystems, however, the fundamental one, transferable vulnerability, remains critically underexplored. Existing studies neither offer practical ways, i.e. require access to the downstream classifier of SCMs, to produce effective adversarial samples for adversarial defense, nor give heed to the widely used LLM4Code in modern software development platforms and cloud-based integrated development environments. Therefore, this work systematically studies the intrinsic vulnerability transferability of both traditional SCMs and LLM4Code, and proposes a victim-agnostic approach to generate practical adversarial samples. We design HABITAT, consisting of a tailored perturbation-inserting mechanism and a hierarchical Reinforcement Learning framework that adaptively selects optimal perturbations without requiring any access to the downstream classifier of SCMs. Furthermore, an intrinsic transferability analysis of SCM vulnerabilities is conducted, revealing the potential vulnerability correlation between traditional SCMs and LLM4Code, together with fundamental factors that govern the success rate of victim-agnostic transfer attacks. These findings of SCM vulnerabilities underscore the critical focal points for developing robust defenses in the future. Experimental evaluation demonstrates that our constructed adversarial examples crafted based on traditional SCMs achieve up to 64% success rates against LLM4Code, surpassing the state-of-the-art by over 15%.

---

## 317. Clean up your Mesh! Part 1: Plane and simplex

**论文链接:** [http://arxiv.org/abs/2511.08058v2](http://arxiv.org/abs/2511.08058v2)

**作者:** Steven De Keninck, Martin Roelfs, Leo Dorst, David Eelbode

**发布时间:** 2025-11-11

**备注:** 22 pages, 10 figures

### GPT解析

### 总结

本文重新审视了网格表示的几何基础，通过基于平面的几何代数(PGA)视角，探讨了其在离散几何中的效率和表达能力，并提出了统一的坐标无关公式来表示几何属性。

### 背景

网格表示是计算机图形学和几何处理的基础，而基于平面的几何代数(PGA)提供了一种新的几何表示方法，但其效率和表达能力需要重新评估。

### 目的

重新评估基于平面的几何代数(PGA)在离散几何中的效率和表达能力，探索如何用简洁的数学表示来描述几何对象及其属性。

### 方法

利用基于平面的几何代数(PGA)的欧几里得范数和理想范数，推导k-单形和k-复形的统一表示公式，并将其扩展到经典几何问题的坐标无关解。

### 主要发现

k-单形（如顶点、边、面）可以简洁地表示为顶点的连接，k-复形（如点云、网格）可以表示为这些连接的和；k-大小（量、长度、面积等）可以通过PGA的范数自然推导；体积、质心和惯性矩等经典几何问题可以表示为统一的无坐标公式。

### 结论

基于平面的几何代数为离散几何提供了强大而统一的表示框架，能够简洁地表达各种几何对象及其属性，并在实际应用中展现出实用价值。

### 翻译

我们通过基于平面的几何代数(PGA)的视角重新审视了网格表示的几何基础，质疑了其在离散几何中的效率和表达能力。我们发现k-单形（顶点、边、面等）和k-复形（点云、线复形、网格等）可以简洁地写为顶点的连接及其和，分别。我们展示了如何通过PGA的欧几里得范数和理想范数自然地推导出它们的k-大小（量、长度、面积等）的单一公式。这个想法随后被扩展，为单形和任意维度的复形产生经典结果的统一的无坐标公式，如体积、质心和惯性矩。最后，我们在一些实际例子上展示了这些想法的实际应用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决网格表示的多样性和效率问题。在3D网格处理中，存在多种表示方法（如半边结构、邻接矩阵、三角形列表等），每种方法针对特定应用场景，这增加了算法设计的复杂性。论文提出使用平面几何代数(PGA)作为统一框架，提供更简洁、更通用的网格表示方法。这个问题在计算机图形学、CAD/CAM和工程仿真等领域非常重要，因为一个更高效的网格表示方法可以简化算法设计，提高计算效率，并促进不同应用间的技术迁移。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过重新审视网格表示的几何基础，特别是从平面几何代数(PGA)的视角出发，发现k-单形（顶点、边、面等）和k-复形（点云、线复形、网格等）可以简洁地表示为顶点的连接及其和。他们利用PGA的欧几里得和理想范数自然地推导出统一的几何量计算公式。作者借鉴了W.K. Clifford关于k-单形的工作，以及使用外微积分和几何代数的先前方法，但创新点在于包含了理想元素（位于无穷远处），这为PGA提供了欧几里得和理想范数，使方法更加完整和高效。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用平面几何代数(PGA)作为统一框架表示和处理网格几何元素。通过将几何元素（点、线、面等）表示为PGA中的特定代数对象，利用连接操作构建k-单形和复形，并应用欧几里得范数和理想范数计算几何量。整体流程包括：1)将网格顶点嵌入PGA表示；2)使用连接操作构建边、面等几何元素的PGA表示；3)应用范数计算几何大小；4)通过边界计算几何量而不需要显式重建几何元素；5)扩展到计算质心、惯性矩等高级几何量；6)通过实际示例展示应用。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用平面几何代数作为统一框架；2)提供k-单形和复形的简洁表示；3)给出统一的、无坐标的体积、质心和惯性矩计算公式；4)展示如何从边界计算几何大小；5)提供计算网格惯性张量和主轴的方法。相比之前的工作，本文创新地包含了理想元素，提供了欧几里得和理想范数，方法更简洁通用，可处理任意维度，并能从边界直接计算几何量而不需要显式重建几何元素。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过平面几何代数提供了一个统一、高效且坐标无关的框架，用于表示和处理网格及其几何属性，包括大小、质心和惯性矩等。'}


### 论文摘要

We revisit the geometric foundations of mesh representation through the lens of Plane-based Geometric Algebra (PGA), questioning its efficiency and expressiveness for discrete geometry. We find how $k$-simplices (vertices, edges, faces, ...) and $k$-complexes (point clouds, line complexes, meshes, ...) can be written compactly as joins of vertices and their sums, respectively. We show how a single formula for their $k$-magnitudes (amount, length, area, ...) follows naturally from PGA's Euclidean and Ideal norms. This idea is then extended to produce unified coordinate-free formulas for classical results such as volume, centre of mass, and moments of inertia for simplices and complexes of arbitrary dimensionality. Finally we demonstrate the practical use of these ideas on some real-world examples.

---

## 318. USV Obstacles Detection and Tracking in Marine Environments

**论文链接:** [http://arxiv.org/abs/2511.07950v1](http://arxiv.org/abs/2511.07950v1)

**作者:** Yara AlaaEldin, Enrico Simetti, Francesca Odone

**发布时间:** 2025-11-11

### GPT解析

### 总结

这篇论文提出了一种用于无人水面艇(USV)的障碍物检测和跟踪系统，通过结合相机和LiDAR传感器数据，并最终提出了一种混合方法来构建环境障碍物地图。

### 背景

为无人水面艇开发稳健有效的障碍物检测和跟踪系统在海洋环境中具有挑战性。热那亚大学的GRAAL实验室已经在这个领域进行了多年研究，提出了基于图像平面和3D LiDAR点云的障碍物检测和跟踪方法。

### 目的

继续开发现有的障碍物检测和跟踪系统，评估其在最新海洋数据集上的性能，将系统模块集成到ROS平台，并提出一种结合相机和LiDAR优势的混合方法。

### 方法

1) 评估现有系统在最近发布的海洋数据集上的性能；2) 将系统不同模块集成到ROS平台；3) 使用同步的LiDAR和相机数据进行实时测试；4) 比较两种方法：相机和LiDAR传感器融合方法与仅使用LiDAR点云方法；5) 提出一种混合方法，结合两种方法的优势。

### 主要发现

通过实验分析，比较了使用传感器融合和仅使用LiDAR点云两种方法的性能，并验证了系统在各种海洋条件下的有效性。

### 结论

提出了一种混合方法，结合了相机和LiDAR传感器融合与仅使用LiDAR点云两种方法的优势，为USV构建了周围环境的信息障碍物地图。

### 翻译

为无人水面艇(USV)在海洋环境中开发稳健有效的障碍物检测和跟踪系统是一项具有挑战性的任务。过去几年中，热那亚大学的GRAAL实验室在这个领域进行了研究，提出了一种在图像平面检测和跟踪障碍物，然后在3D LiDAR点云中定位它们的方法。在这项工作中，我们继续开发这个系统，首先评估了它在最近发布的海洋数据集上的性能。然后，我们将系统的不同模块集成到ROS平台，可以在同步的LiDAR和相机数据上进行实时测试，这些数据来自MIT海洋数据集中各种海洋条件下的收集。我们使用两种方法获得了结果的详细实验分析；一种使用相机和LiDAR之间的传感器融合来检测和跟踪障碍物，另一种仅使用LiDAR点云进行检测和跟踪。最后，我们提出了一种混合方法，结合了两种方法的优势，为USV构建了周围环境的信息障碍物地图。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在海洋环境中为无人水面艇（USV）开发鲁棒且有效的障碍物检测与跟踪系统的问题。这个问题在现实中非常重要，因为USV在科学探索、搜救行动和国防安全等领域的应用日益增多，可以替代人类执行海上繁琐和危险的任务。研究中，海洋环境面临独特挑战，如海雾、雨、风、波浪和洋流等环境干扰会影响视觉感知，各种光照条件和水面反射会影响检测准确性，而LiDAR点云稀疏问题也会影响障碍物识别，同时系统还需要满足实时性要求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析海洋环境中USV导航面临的独特挑战，评估了热那亚大学GRAAL实验室之前开发的系统，并对现有研究进行了全面综述，包括语义分割、聚类、边界框检测器和传感器融合等方法。基于这些分析，作者设计了改进的系统，使用YOLOv3进行障碍物检测，实现基于匈牙利算法的数据关联，使用卡尔曼滤波进行状态预测，并将不同模块整合到ROS平台。作者确实借鉴了现有工作，特别是基于热那亚大学GRAAL实验室之前的研究成果，并参考了多篇相关论文中的方法和思路。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '该方法的核心思想是通过融合相机和LiDAR两种传感器的数据，克服单一传感器的局限性，实现对海洋环境中障碍物的准确检测和跟踪。整体实现流程包括：1) 使用YOLOv3处理视频帧检测障碍物及其边界框；2) 使用匈牙利算法进行数据关联，结合位置和外观属性匹配检测到的障碍物与现有轨迹；3) 使用卡尔曼滤波器预测缺失检测的障碍物位置；4) 将跟踪结果投影到3D参考框架，提取对应的点云集群并用3D边界框包围；5) 结合相机和LiDAR的数据生成包含障碍物位置和方向的环境地图；6) 在ROS平台上整合所有模块实现实时处理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '论文的关键创新点包括：1) 将不同模块整合到ROS平台实现实时处理；2) 在多种海洋条件下进行全面评估和基准测试；3) 提出结合仅使用LiDAR和传感器融合两种方法的混合方法；4) 调整YOLOv3检测阈值以利用低置信度的正确检测；5) 优化轨迹添加和删除策略。相比之前的工作，这篇论文更注重全面性（涵盖检测、跟踪和3D定位）、实用性（强调实时性能和实际部署）、评估严格性（在各种真实海洋条件下验证）、方法创新性（提出混合方法）和开放性（使用公开数据集提高结果可复现性）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过整合深度学习与传统跟踪方法，在ROS平台上实现了一个融合相机和LiDAR数据的实时障碍物检测与跟踪系统，并在多种海洋条件下验证了其鲁棒性和有效性。'}


### 论文摘要

Developing a robust and effective obstacle detection and tracking system for Unmanned Surface Vehicle (USV) at marine environments is a challenging task. Research efforts have been made in this area during the past years by GRAAL lab at the university of Genova that resulted in a methodology for detecting and tracking obstacles on the image plane and, then, locating them in the 3D LiDAR point cloud. In this work, we continue on the developed system by, firstly, evaluating its performance on recently published marine datasets. Then, we integrate the different blocks of the system on ROS platform where we could test it in real-time on synchronized LiDAR and camera data collected in various marine conditions available in the MIT marine datasets. We present a thorough experimental analysis of the results obtained using two approaches; one that uses sensor fusion between the camera and LiDAR to detect and track the obstacles and the other uses only the LiDAR point cloud for the detection and tracking. In the end, we propose a hybrid approach that merges the advantages of both approaches to build an informative obstacles map of the surrounding environment to the USV.

---

## 319. Millimeter-Wave UAV Channel Model with Height-Dependent Path Loss and Shadowing in Urban Scenarios

**论文链接:** [http://arxiv.org/abs/2511.10763v1](http://arxiv.org/abs/2511.10763v1)

**作者:** Abdul Saboor, Evgenii Vinogradov

**发布时间:** 2025-11-13

**备注:** Submitted to the International Journal of Microwave and Wireless Technologies

### GPT解析

### 总结

该研究提出了一个基于无人机空中基站高度的毫米波信道模型，探讨了城市几何结构对视距概率和大尺度衰落的影响，并通过射线仿真验证了模型的有效性。

### 背景

无人机作为空中基站有望扩展6G毫米波在城市区域的覆盖并提高链路可靠性，但基于无人机的空对地信道高度依赖于高度和城市几何结构。

### 目的

提出一个基于ABS高度的毫米波信道模型，研究除标准建成区参数外，城市几何结构是否显著影响视距概率和大尺度衰落。

### 方法

使用MAT射线追踪在26 GHz频率进行仿真，为四种具有相同建成区参数但不同空间组织的城市布局模拟约10K个城市实现，使用sigmoid模型提取基于高度的视距概率，并通过指数拟合推导高度相关的路径损耗指数和阴影衰落趋势。

### 主要发现

非视距的路径损耗指数在高空处降至2.5-3，视距的路径损耗指数保持在2附近，阴影衰落随高度降低，即使建成区参数固定，几何布局也会导致路径损耗指数发生适度但一致的变化(±0.2)。

### 结论

提出的统一模型与射线追踪统计数据高度吻合，提供了一个实用的、基于高度的LSF模型，适合在复杂城市场景中进行ABS规划。

### 翻译

无人机作为空中基站(UAVs作为ABSs)有望扩展6G毫米波(mmWave)在城市区域的覆盖并提高链路可靠性。然而，基于无人机的空对地(UAV-based A2G)信道高度依赖于高度和城市几何结构。本文提出了一个基于ABS高度的毫米波信道模型，并研究了除标准建成区参数外，城市几何结构是否显著影响视距概率(PLoS)和大尺度衰落(LSF)。使用MATLAB射线追踪在26 GHz，我们为四种具有相同建成区参数但空间组织不同的城市布局模拟了约10K个城市实现。我们使用sigmoid模型提取基于高度的视距概率，并通过指数拟合推导高度相关的路径损耗指数(PLE)和阴影衰落趋势。结果表明，非视距(NLoS)的路径损耗指数在高空处降至2.5-3，视距(LoS)的路径损耗指数保持在2附近，阴影衰落随高度降低。我们还发现，即使建成区参数固定，几何布局也会导致路径损耗指数发生适度但一致的变化(±0.2)。提出的统一模型与射线追踪统计数据高度吻合，提供了一个实用的、基于高度的LSF模型，适合在复杂城市场景中进行ABS规划。


### 论文摘要

Uncrewed Aerial Vehicles (UAVs) serving as Aerial Base Stations (ABSs) are expected to extend 6G millimeter-Wave (mmWave) coverage and improve link reliability in urban areas. However, UAV-based Air-to-Ground (A2G) channels are highly dependent on height and urban geometry. This paper proposes an ABS height-dependent mmWave channel model and investigates whether urban geometry, beyond the standard built-up parameters, significantly affects LoS probability (PLoS) and Large-Scale Fading (LSF). Using MATLAB ray tracing at 26 GHz, we simulate approximately 10K city realizations for four urban layouts that share identical built-up parameters but differ in their spatial organization. We extract elevation-based PLoS using a sigmoid model and derive height-dependent Path-Loss Exponents (PLEs) and shadow-fading trends using exponential fits. Results show that PLE for Non-Line-of-Sight (NLoS) decreases toward 2.5-3 at high altitudes, Line-of-Sight (LoS) PLE remains near 2, and shadow fading reduces with height. We also find that geometric layout introduces a modest but consistent change in PLE (+/- 0.2), even when built-up parameters are fixed. The proposed unified model aligns well with ray-tracing statistics and offers a practical, height-dependent LSF model suitable for ABS planning in complex urban scenarios.

---

## 320. Population-mobility coevolution drives spatial heterogeneity of cities

**论文链接:** [http://arxiv.org/abs/2511.10198v1](http://arxiv.org/abs/2511.10198v1)

**作者:** Hao Huang, Yuming Lin, Jiazhen Liu

**发布时间:** 2025-11-13

### GPT解析

### 总结

本文提出一个描述城市内部人口-流动性共同演化的耦合动力学模型，解释空间异质性作为快速变化流动性与缓慢适应人口之间相互反馈的涌现结果，该模型在八个全球城市的3.88亿条记录上得到验证。

### 背景

城市空间异质性（人口和活动的不均匀分布）是城市动态的基础，与基础设施过载、住房可负担性和社会不平等等关键问题相关。尽管城市在人口和流动性方面具有相似的标度律，但它们表现出截然不同的空间模式，现有定性或描述性研究未能解释这一矛盾现象的基本机制。

### 目的

提出一个机制性模型解释城市空间异质性的涌现，验证该模型在真实城市数据上的有效性，为政策设计提供机械性、高分辨率的见解。

### 方法

开发了一个耦合动力学模型来描述城市内部人口-流动性的共同演化，并在八个不同全球城市的3.88亿条记录上进行了验证。

### 主要发现

1. 真实的异质性作为一个独特的稳定状态出现，介于无序同质性和不可持续的超级枢纽主导之间；2. 人口稠密地区主要由共同演化强度塑造，距离衰减增加导致城市经历同质性-异质性-同质性的三阶段转变；3. 区域间的功能性吸引力持续增强有序的异质结构；4. 综合人口-流动性政策比单一干预措施更具成本效益。

### 结论

该模型能够提供机械性、高分辨率的见解，为政策设计提供严格指导，表明综合政策干预比单一措施更有效。

### 翻译

城市空间异质性——人口和活动的不均匀分布——是城市动态的基础，与基础设施过载、住房可负担性和社会不平等等关键问题相关。尽管城市在人口和流动性方面具有相似的标度律，但它们表现出截然不同的空间模式。这一矛盾需要对空间异质性的涌现提供机制性解释，而现有的定性或描述性研究未能捕捉到基本机制。在这里，我们提出了一个描述城市内部人口-流动性共同演化的耦合动力学模型，将空间异质性解释为快速变化的流动性和缓慢适应的人口之间相互反馈的涌现结果。我们的模型在八个不同全球城市的3.88亿条记录上得到验证，成功重现了统计规律和真实的空间模式。我们发现真实的异质性作为一个独特的稳定状态出现，介于无序同质性和不可持续的超级枢纽主导之间。此外，我们从理论和实证上表明，人口稠密地区主要由共同演化强度塑造，而距离衰减的增加导致城市经历同质性-异质性-同质性的三阶段转变。此外，区域间的功能性吸引力持续增强有序的异质结构。对真实世界规划场景的模拟——包括危机引发的封锁、计划区域扩张和从拥挤中心疏散——表明，综合人口-流动性政策比单一干预措施更具成本效益。我们的模型可以为政策设计提供机械性、高分辨率的见解。


### 论文摘要

The spatial heterogeneity of cities -- the uneven distribution of population and activities -- is fundamental to urban dynamics and related to critical issues such as infrastructure overload, housing affordability, and social inequality. Despite sharing similar scaling laws of population and mobility, cities exhibit vastly different spatial patterns. This paradox call for a mechanistic explanation for the emergence of spatial heterogeneity, while existing qualitative or descriptive studies fail to capture the underlying mechanisms. Here, we propose a coupled dynamical model that describe the intra-city population-mobility coevolution, explaining spatial heterogeneity as an emergent outcome of mutual feedback between the fast-changing mobility and the slow-adapting population. Our model is validated on over 388 million records from eight diverse global cities, successfully reproduces both the statistical laws and realistic spatial patterns. We find out realistic heterogeneity emerges as a distinct stable state, intermediate between disordered homogeneity and unsustainable super-hub dominance. Moreover, we theoretically and empirically show that populated areas are predominantly shaped by coevolution strength, while the increasing distance decay leads cities through a three-phase transition of homogeneity-heterogeneity-homogeneity. Besides, functional attractiveness between areas consistently enhances the ordered heterogeneous structure. Simulations of real-world planning scenarios -- including crisis-induced lockdown, planned zone expansions, and dispersal from congested centers -- indicate that integrated population-mobility policies are more cost-effective than single interventions. Our model can provides mechanistic, high-resolution insights to rigorously inform policy design.

---

## 321. D-AWSIM: Distributed Autonomous Driving Simulator for Dynamic Map Generation Framework

**论文链接:** [http://arxiv.org/abs/2511.09080v1](http://arxiv.org/abs/2511.09080v1)

**作者:** Shunsuke Ito, Chaoran Zhao, Ryo Okamura, Takuya Azumi

**发布时间:** 2025-11-12

**备注:** 9 pages. This version includes minor lstlisting configuration adjustments for successful compilation. No changes to content or layout. Originally published at Euromicro DSD 2025

### GPT解析

### 总结

本文提出了一种名为D-AWSIM的分布式模拟器，用于支持大规模传感器部署和密集交通环境的自动驾驶系统研究，解决了传统单主机模拟器无法处理大规模城市交通场景的问题。

### 背景

自动驾驶系统已取得显著进展，在特定操作设计域内完全自动驾驶接近实际部署。然而，扩展这些领域需要解决多样化条件下的安全保障问题。通过车对车和车对基础设施的信息共享，基于车辆和路边传感器数据构建的动态地图平台提供了有前景的解决方案，但真实世界的实验面临高成本和监管挑战。

### 目的

开发一种能够模拟大规模传感器部署和密集交通环境的工具，使研究人员能够探索信息共享策略，而无需依赖物理测试平台。

### 方法

提出D-AWSIM分布式模拟器，通过将工作负载分配到多台机器来支持大规模场景模拟，并在D-AWSIM上构建动态地图生成框架。

### 主要发现

与单机设置相比，D-AWSIM显著提高了车辆数量和激光雷达传感器处理的吞吐量。与Autoware的集成证明了其在自动驾驶研究中的适用性。

### 结论

D-AWSIM为研究人员探索信息共享策略提供了有效工具，无需依赖物理测试平台，有助于解决自动驾驶系统在扩展操作设计域时面临的安全保障挑战。

### 翻译

自动驾驶系统已取得显著进展，在特定操作设计域内完全自动驾驶接近实际部署。扩展这些领域需要解决多样化条件下的安全保障问题。通过车对车和车对基础设施通信的信息共享，基于车辆和路边传感器数据构建的动态地图平台提供了有前景的解决方案。真实世界的实验涉及大量基础设施传感器，面临高成本和监管挑战。传统的单主机模拟器缺乏处理大规模城市交通场景的能力。本文提出了D-AWSIM，一种分布式模拟器，通过将工作负载分配到多台机器来支持大规模传感器部署和密集交通环境的模拟。D-AWSIM上的动态地图生成框架使研究人员无需依赖物理测试平台即可探索信息共享策略。评估显示，与单机设置相比，D-AWSIM显著提高了车辆数量和激光雷达传感器处理的吞吐量。与Autoware的集成证明了其在自动驾驶研究中的适用性。


### 论文摘要

Autonomous driving systems have achieved significant advances, and full autonomy within defined operational design domains near practical deployment. Expanding these domains requires addressing safety assurance under diverse conditions. Information sharing through vehicle-to-vehicle and vehicle-to-infrastructure communication, enabled by a Dynamic Map platform built from vehicle and roadside sensor data, offers a promising solution. Real-world experiments with numerous infrastructure sensors incur high costs and regulatory challenges. Conventional single-host simulators lack the capacity for large-scale urban traffic scenarios. This paper proposes D-AWSIM, a distributed simulator that partitions its workload across multiple machines to support the simulation of extensive sensor deployment and dense traffic environments. A Dynamic Map generation framework on D-AWSIM enables researchers to explore information-sharing strategies without relying on physical testbeds. The evaluation shows that D-AWSIM increases throughput for vehicle count and LiDAR sensor processing substantially compared to a single-machine setup. Integration with Autoware demonstrates applicability for autonomous driving research.

---

## 322. Multi-level Latent Variable Models for Coheritability Analysis in Electronic Health Records

**论文链接:** [http://arxiv.org/abs/2511.08532v1](http://arxiv.org/abs/2511.08532v1)

**作者:** Yinjun Zhao, Nicholas Tatonetti, Yuanjia Wang

**发布时间:** 2025-11-11

**备注:** 21 pages, 5 figures

### GPT解析

### 总结

该研究提出了一种稳健灵活的统计框架，用于在电子健康记录(EHR)的家族研究中联合估计连续和二元表型的遗传率和遗传相关性，解决了现有方法在处理家族相关结构复杂性、表型异质性和计算可扩展性方面的不足。

### 背景

电子健康记录与家族关系数据的关联为大规模研究复杂表型的遗传结构提供了独特机会，但现有的遗传率和共遗传率估计方法往往未能充分考虑家族相关结构的复杂性、表型类型的异质性和计算可扩展性。

### 目的

开发一个稳健且灵活的统计框架，用于在基于EHR的家族研究中联合估计连续和二元表型的遗传率和遗传相关性。

### 方法

基于多水平潜变量模型构建，将表型协方差分解为可解释的遗传和环境成分，同时纳入家族内和家族间的变异；基于广义方程估计(GEE)推导迭代算法进行参数估计。

### 主要发现

模拟研究表明，估计量具有一致性，并在多种现实情况下产生有效的推断；应用该方法于真实EHR数据，发现心理健康状况与内分泌/代谢表型之间存在显著的遗传相关性，支持共享病因的假设。

### 结论

该研究为高维EHR数据中的共遗传率分析提供了一个可扩展且严谨的框架，有助于促进复杂疾病网络中共享遗传影响的识别。

### 翻译

电子健康记录(EHRs)与家族关系数据的关联为大规模研究复杂表型的遗传结构提供了独特机会。然而，现有的遗传率和共遗传率估计方法通常未能充分考虑家族相关结构的复杂性、表型类型的异质性和计算可扩展性。我们提出了一种稳健且灵活的统计框架，用于在基于EHR的家族研究中联合估计连续和二元表型的遗传率和遗传相关性。我们的方法基于多水平潜变量模型构建，将表型协方差分解为可解释的遗传和环境成分，同时纳入家族内和家族间的变异。我们基于广义方程估计(GEE)推导了迭代算法进行估计。在各种参数配置下的模拟研究表明，我们的估计量具有一致性，并在多种现实情况下产生有效的推断。将我们的方法应用于来自大型城市健康系统的真实EHR数据，我们发现心理健康状况与内分泌/代谢表型之间存在显著的遗传相关性，这支持了共享病因的假设。这项工作为高维EHR数据中的共遗传率分析提供了一个可扩展且严谨的框架，有助于促进复杂疾病网络中共享遗传影响的识别。


### 论文摘要

Electronic health records (EHRs) linked with familial relationship data offer a unique opportunity to investigate the genetic architecture of complex phenotypes at scale. However, existing heritability and coheritability estimation methods often fail to account for the intricacies of familial correlation structures, heterogeneity across phenotype types, and computational scalability. We propose a robust and flexible statistical framework for jointly estimating heritability and genetic correlation among continuous and binary phenotypes in EHR-based family studies. Our approach builds on multi-level latent variable models to decompose phenotypic covariance into interpretable genetic and environmental components, incorporating both within- and between-family variations. We derive iteration algorithms based on generalized equation estimations (GEE) for estimation. Simulation studies under various parameter configurations demonstrate that our estimators are consistent and yield valid inference across a range of realistic settings. Applying our methods to real-world EHR data from a large, urban health system, we identify significant genetic correlations between mental health conditions and endocrine/metabolic phenotypes, supporting hypotheses of shared etiology. This work provides a scalable and rigorous framework for coheritability analysis in high-dimensional EHR data and facilitates the identification of shared genetic influences in complex disease networks.

---

## 323. VectorSynth: Fine-Grained Satellite Image Synthesis with Structured Semantics

**论文链接:** [http://arxiv.org/abs/2511.07744v1](http://arxiv.org/abs/2511.07744v1)

**作者:** Daniel Cher, Brian Wei, Srikumar Sastry, Nathan Jacobs

**发布时间:** 2025-11-11

### GPT解析

### 总结

VectorSynth是一个基于扩散的框架，用于根据带有语义属性的地理多边形注释进行像素级精确的卫星图像合成。

### 背景

与之前的文本或布局条件模型不同，VectorSynth学习密集的跨模态对应关系，对齐图像和语义向量几何。

### 目的

实现细粒度、空间感知的卫星图像编辑和生成，支持交互式工作流程。

### 方法

使用视觉语言对齐模块从多边形语义生成像素级嵌入，这些嵌入引导条件图像生成框架，使其同时尊重空间范围和语义线索。

### 主要发现

在语义保真度和结构真实感方面比先前方法有显著改进，训练的视觉语言模型显示出细粒度的空间定位能力。

### 结论

VectorSynth支持混合语言提示和几何感知条件的交互式工作流程，允许快速的场景模拟、空间编辑和地图感知的内容生成。

### 翻译

VectorSynth是一种基于扩散的框架，用于根据带有语义属性的地理多边形注释进行像素级精确的卫星图像合成。与之前的文本或布局条件模型不同，VectorSynth学习密集的跨模态对应关系，对齐图像和语义向量几何，实现细粒度、空间感知的编辑。视觉语言对齐模块从多边形语义生成像素级嵌入；这些嵌入引导条件图像生成框架，同时尊重空间范围和语义线索。VectorSynth支持交互式工作流程，混合语言提示和几何感知条件，允许快速的场景模拟、空间编辑和地图感知的内容生成。为训练和评估，我们收集了卫星场景与像素注册多边形注释的配对数据集，涵盖包含人造和自然特征的多样化城市场景。我们观察到在语义保真度和结构真实感方面比先前方法有显著改进，并且训练的视觉语言模型显示出细粒度的空间定位能力。代码和数据可在https://github.com/mvrl/VectorSynth获取。


### 论文摘要

We introduce VectorSynth, a diffusion-based framework for pixel-accurate satellite image synthesis conditioned on polygonal geographic annotations with semantic attributes. Unlike prior text- or layout-conditioned models, VectorSynth learns dense cross-modal correspondences that align imagery and semantic vector geometry, enabling fine-grained, spatially grounded edits. A vision language alignment module produces pixel-level embeddings from polygon semantics; these embeddings guide a conditional image generation framework to respect both spatial extents and semantic cues. VectorSynth supports interactive workflows that mix language prompts with geometry-aware conditioning, allowing rapid what-if simulations, spatial edits, and map-informed content generation. For training and evaluation, we assemble a collection of satellite scenes paired with pixel-registered polygon annotations spanning diverse urban scenes with both built and natural features. We observe strong improvements over prior methods in semantic fidelity and structural realism, and show that our trained vision language model demonstrates fine-grained spatial grounding. The code and data are available at https://github.com/mvrl/VectorSynth.

---

## 324. Fair and Efficient allocation of Mobility-on-Demand resources through a Karma Economy

**论文链接:** [http://arxiv.org/abs/2511.07225v1](http://arxiv.org/abs/2511.07225v1)

**作者:** Matteo Cederle, Saverio Bolognani, Gian Antonio Susto

**发布时间:** 2025-11-10

**备注:** 6 pages, 3 figures. Under review at the 2026 European Control Conference (ECC)

### GPT解析

### 总结

本文提出了一种基于Karma的非货币机制，用于解决按需出行系统中的社会经济不平等问题，同时考虑了用户紧急程度的时空变化性。

### 背景

按需出行系统如网约车改变了城市交通，但也加剧了社会经济不平等，部分原因是动态定价策略。现有的智能出行公平性框架忽视了用户紧急程度的时空变化性。

### 目的

引入一种基于Karma的非货币机制，模拟内生紧急程度，使用户时间敏感度能够根据系统条件和外部因素而动态演变。

### 方法

开发了一个理论框架，保持传统Karma经济体的效率和公平保证，同时适应真实的用户行为建模。

### 主要发现

在模拟的按需出行场景中应用，该框架能够实现高水平的系统效率，同时保证用户资源的公平分配。

### 结论

该框架能够在考虑用户紧急程度动态变化的情况下，平衡系统效率和资源分配的公平性。

### 翻译

像网约车这样的按需出行系统已经改变了城市交通，但它们也加剧了获取这些服务的社会经济不平等，部分原因也是由于动态定价策略。尽管在智能出行领域已经提出了几种公平感知框架，但它们常常忽视了塑造现实世界交通需求的用户紧急程度的时空变化性。本文引入了一种非货币的基于Karma的机制，该机制模拟内生紧急程度，允许用户的时间敏感度根据系统条件和外部因素而演变。我们开发了一个理论框架，保持了传统Karma经济体的效率和公平保证，同时适应了这种真实的用户行为建模。应用于模拟的按需出行场景，我们表明我们的框架能够实现高水平的系统效率，同时保证用户资源的公平分配。


### 论文摘要

Mobility-on-demand systems like ride-hailing have transformed urban transportation, but they have also exacerbated socio-economic inequalities in access to these services, also due to surge pricing strategies. Although several fairness-aware frameworks have been proposed in smart mobility, they often overlook the temporal and situational variability of user urgency that shapes real-world transportation demands. This paper introduces a non-monetary, Karma-based mechanism that models endogenous urgency, allowing user time-sensitivity to evolve in response to system conditions as well as external factors. We develop a theoretical framework maintaining the efficiency and fairness guarantees of classical Karma economies, while accommodating this realistic user behavior modeling. Applied to a simulated mobility-on-demand scenario we show that our framework is able to achieve high levels of system efficiency, guaranteeing at the same time equitable resource allocation for the users.

---

## 325. Beyond Gaussian Assumptions: A General Fractional HJB Control Framework for Lévy-Driven Heavy-Tailed Channels in 6G

**论文链接:** [http://arxiv.org/abs/2511.07167v1](http://arxiv.org/abs/2511.07167v1)

**作者:** Mengqi Li, Lixin Li, Wensheng Lin, Zhu Han, Tamer Başar

**发布时间:** 2025-11-10

**DOI:** 10.1109/TWC.2025.3631903

### GPT解析

### 总结

该论文提出了一种基于对称α稳定莱维过程的无线信道模型和广义最优控制框架，用于解决6G无线系统在具有挑战性环境中的性能下降问题，通过分数阶哈密顿-雅可比-贝尔曼方程优化传输功率。

### 背景

6G无线系统在高速列车穿越密集城市走廊和无人机在山区地形上的链路等具有挑战性的环境中会遭受严重的性能下降。这些场景表现为非高斯、非平稳的信道，具有重尾衰落和突然的信号波动特性。

### 目的

解决6G无线系统在具有挑战性环境中的性能下降问题，特别是处理非高斯、非平稳信道中的重尾衰落和信号波动。

### 方法

提出了一种基于对称α稳定莱维过程的无线信道模型，能够对长期和短期衰落进行连续时间状态空间表征。基于此模型，开发了包含黎兹分数阶算子的分数阶哈密顿-雅可比-贝尔曼方程形式的广义最优控制框架，以捕获非局部空间效应和依赖于记忆的动力学。

### 主要发现

严格建立了分数阶哈密顿-雅可比-贝尔曼方程粘性解的存在性和唯一性，确保了所提出控制公式的理论有效性。在多小区、多用户下行链路设置中的数值模拟证明了基于分数阶哈密顿-雅可比-贝尔曼的策略在重尾同信道和多用户干扰下优化传输功率的有效性。

### 结论

基于对称α稳定莱维过程的信道模型和分数阶哈密顿-雅可比-贝尔曼方程的控制框架能够有效解决6G无线系统在具有挑战性环境中的性能下降问题，特别是在处理非高斯、非平稳信道中的重尾衰落和信号波动方面具有优势。

### 翻译

新兴的6G无线系统在高速列车穿越密集城市走廊和无人机在山区地形上的链路等具有挑战性的环境中会遭受严重的性能下降。这些场景表现出非高斯、非平稳的信道，具有重尾衰落和突然的信号波动。为了应对这些挑战，本文提出了一种基于对称α稳定莱维过程的新型无线信道模型，从而能够对长期和短期衰落进行连续时间状态空间表征。基于此模型，通过包含黎兹分数阶算子的分数阶哈密顿-雅可比-贝尔曼方程开发了广义最优控制框架，以捕获非局部空间效应和依赖于记忆的动力学。论文严格建立了分数阶哈密顿-雅可比-贝尔曼方程粘性解的存在性和唯一性，从而确保了所提出控制公式的理论有效性。在多小区、多用户下行链路设置中进行的数值模拟证明了基于分数阶哈密顿-雅可比-贝尔曼的策略在重尾同信道和多用户干扰下优化传输功率的有效性。


### 论文摘要

Emerging 6G wireless systems suffer severe performance degradation in challenging environments like high-speed trains traversing dense urban corridors and Unmanned Aerial Vehicles (UAVs) links over mountainous terrain. These scenarios exhibit non-Gaussian, non-stationary channels with heavy-tailed fading and abrupt signal fluctuations. To address these challenges, this paper proposes a novel wireless channel model based on symmetric $α$-stable Lévy processes, thereby enabling continuous-time state-space characterization of both long-term and short-term fading. Building on this model, a generalized optimal control framework is developed via a fractional Hamilton-Jacobi-Bellman (HJB) equation that incorporates the Riesz fractional operator to capture non-local spatial effects and memory-dependent dynamics. The existence and uniqueness of viscosity solutions to the fractional HJB equation are rigorously established, thus ensuring the theoretical validity of the proposed control formulation. Numerical simulations conducted in a multi-cell, multi-user downlink setting demonstrate the effectiveness of the fractional HJB-based strategy in optimizing transmission power under heavy-tailed co-channel and multi-user interference.

---

## 326. AgentSUMO: An Agentic Framework for Interactive Simulation Scenario Generation in SUMO via Large Language Models

**论文链接:** [http://arxiv.org/abs/2511.06804v1](http://arxiv.org/abs/2511.06804v1)

**作者:** Minwoo Jeong, Jeeyun Chang, Yoonjin Yoon

**发布时间:** 2025-11-10

**备注:** Submitted to Transportation Research Part C (under review)

### GPT解析

### 总结

该研究提出了AgentSUMO，一个基于大型语言模型的智能体框架，用于交互式生成交通模拟场景，使非专业用户能够将抽象政策目标转化为可执行的模拟方案。

### 背景

城市交通系统日益复杂，使交通模拟成为必要工具，但现有平台如SUMO主要限于专家使用。创建真实模拟场景需要专业知识，对非专业用户构成障碍，且用户需求常不完整且抽象，与现有命令式工作流程不匹配。

### 目的

解决非专业用户使用交通模拟工具的困难，创建能够将抽象政策目标转化为可执行模拟场景的智能体框架。

### 方法

AgentSUMO引入自适应推理层，解释用户意图、评估任务复杂度、推断缺失参数和制定可执行计划。框架包含交互式规划协议（管理推理和用户交互）和模型上下文协议（管理模拟工具间通信）。

### 主要发现

在首尔和曼哈顿的实验表明，AgentSUMO在交通流量指标上取得显著改进，同时保持非专业用户的可访问性，成功弥合政策目标与可执行模拟工作流程间的差距。

### 结论

AgentSUMO有效解决了非专业用户使用交通模拟工具的困难，使抽象政策目标能够转化为可执行的交通模拟场景。

### 翻译

日益复杂的城市交通系统使交通模拟成为基于证据的交通规划和政策评估不可或缺的工具。然而，尽管有SUMO等平台的分析能力，其应用仍主要局限于领域专家。开发真实的模拟场景需要网络构建、起点-终点建模和政策实验参数配置的专业知识，为政策制定者、城市规划者和城市官员等非专业用户设置了巨大障碍。此外，这些用户表达的需求往往不完整且抽象—通常表述为高层次目标，与现有基于语言模型的模拟框架中使用的命令式、顺序工作流程不匹配。为应对这些挑战，本研究提出了AgentSUMO，一个通过大型语言模型进行交互式模拟场景生成的智能体框架。AgentSUMO通过引入自适应推理层，从命令式、命令驱动的执行中脱离出来，该层解释用户意图、评估任务复杂度、推断缺失参数和制定可执行的模拟计划。该框架围绕两个互补组件构建：交互式规划协议，管理推理和用户交互；以及模型上下文协议，管理模拟工具间的标准化通信和编排。通过这种设计，AgentSUMO将抽象政策目标转化为可执行的模拟场景。在首尔和曼哈顿城市网络上的实验表明，智能体工作流程在交通流量指标方面取得了显著改进，同时保持了非专业用户的可访问性，成功弥合了政策目标和可执行模拟工作流程之间的差距。


### 论文摘要

The growing complexity of urban mobility systems has made traffic simulation indispensable for evidence-based transportation planning and policy evaluation. However, despite the analytical capabilities of platforms such as the Simulation of Urban MObility (SUMO), their application remains largely confined to domain experts. Developing realistic simulation scenarios requires expertise in network construction, origin-destination modeling, and parameter configuration for policy experimentation, creating substantial barriers for non-expert users such as policymakers, urban planners, and city officials. Moreover, the requests expressed by these users are often incomplete and abstract-typically articulated as high-level objectives, which are not well aligned with the imperative, sequential workflows employed in existing language-model-based simulation frameworks. To address these challenges, this study proposes AgentSUMO, an agentic framework for interactive simulation scenario generation via large language models. AgentSUMO departs from imperative, command-driven execution by introducing an adaptive reasoning layer that interprets user intents, assesses task complexity, infers missing parameters, and formulates executable simulation plans. The framework is structured around two complementary components, the Interactive Planning Protocol, which governs reasoning and user interaction, and the Model Context Protocol, which manages standardized communication and orchestration among simulation tools. Through this design, AgentSUMO converts abstract policy objectives into executable simulation scenarios. Experiments on urban networks in Seoul and Manhattan demonstrate that the agentic workflow achieves substantial improvements in traffic flow metrics while maintaining accessibility for non-expert users, successfully bridging the gap between policy goals and executable simulation workflows.

---

## 327. Public Transport Under Epidemic Conditions: Nonlinear Trade-Offs Between Risk and Accessibility

**论文链接:** [http://arxiv.org/abs/2511.06377v1](http://arxiv.org/abs/2511.06377v1)

**作者:** Gerhard Hiermann, Joana Ji, Ana Moreno, Rolf Moeckel, Maximilian Schiffer

**发布时间:** 2025-11-09

### GPT解析

### 总结

该研究探讨了流行病期间公共健康与城市基本流动性之间的矛盾，特别是公共交通系统面临的困境，提出了一个整合流行病模拟与交通流量优化的建模框架，并以慕尼黑为案例研究，分析了不同干预措施对流行病结果和可及性的影响。

### 背景

流行病暴露了保护公共健康与维持基本城市流动性之间的关键矛盾。公共交通系统面临这一困境最为尖锐：它们使人们能够获得就业、教育和服务的通道，同时也促进了旅行者之间的密切接触。

### 目的

开发一个集成建模框架，将基于代理的流行病模拟与基于优化的公共交通流量模型相结合，分析设施关闭和交通限制的组合如何影响流行病结果和可及性。

### 方法

使用慕尼黑作为案例研究，开发了一个耦合基于代理的流行病模拟（EpiSim）与基于优化的公共交通流量模型的集成框架，考虑容量限制，分析不同干预措施的效果。

### 主要发现

研究结果揭示了三个关键见解：1）流行病干预措施重新分配而非简单减少感染风险，将传播转移到家庭；2）流行病和交通政策非线性相互作用，适度的需求抑制可以抵消大量的容量削减；3）流行病压力加剧了时间和空间不平等，不成比例地影响了外围和高峰时段的旅行者。

### 结论

全面限制既低效又不公平，需要采取有针对性的、时间和空间差异化的措施，建立具有流行病韧性和社会公平的交通系统。

### 翻译

流行病暴露了保护公共健康与维持基本城市流动性之间的关键矛盾。公共交通系统面临这一困境最为尖锐：它们使人们能够获得就业、教育和服务的通道，同时也促进了旅行者之间的密切接触。我们开发了一个集成建模框架，将基于代理的流行病模拟（EpiSim）与基于优化的公共交通流量模型相结合，考虑容量限制。以慕尼黑为案例研究，我们分析了设施关闭和交通限制的组合如何影响流行病结果和可及性。研究结果揭示了三个关键见解。首先，流行病干预措施重新分配而非简单减少感染风险，将传播转移到家庭。其次，流行病和交通政策非线性相互作用 - 适度的需求抑制可以抵消大量的容量削减。第三，流行病压力加剧了时间和空间不平等，不成比例地影响了外围和高峰时段的旅行者。这些发现表明，全面限制既低效又不公平，需要采取有针对性的、时间和空间差异化的措施，建立具有流行病韧性和社会公平的交通系统。


### 论文摘要

Epidemics expose critical tensions between protecting public health and maintaining essential urban mobility. Public transport systems face this dilemma most acutely: they enable access to jobs, education, and services, yet also facilitate close contact among travelers. We develop an integrated modeling framework that couples agent-based epidemic simulation (EpiSim) with an optimization-based public transport flow model under capacity constraints. Using Munich as a case study, we analyze how combinations of facility closures and transport restrictions shape epidemic outcomes and accessibility. The results reveal three key insights. First, epidemic interventions redistribute rather than simply reduce infection risks, shifting transmission to households. Second, epidemic and transport policies interact nonlinearly - moderate demand suppression can offset large capacity cuts. Third, epidemic pressures amplify temporal and spatial inequalities, disproportionately affecting peripheral and peak-hour travelers. These findings highlight that blanket restrictions are both inefficient and inequitable, calling for targeted, time- and space-differentiated measures to build epidemic-resilient and socially fair transport systems.

---

## 328. Assessing On-Demand Mobility Services and Policy Impacts: A Case Study from Chengdu, China

**论文链接:** [http://arxiv.org/abs/2511.06074v1](http://arxiv.org/abs/2511.06074v1)

**作者:** Youkai Wu, Zhaoxia Guo, Qi Liu abd Stein W. Wallace

**发布时间:** 2025-11-08

### GPT解析

### 总结

本研究通过仿真框架比较了网约车与传统扬招出租车的性能，并评估了三种政策对网约车服务的影响。结果显示，在相同车队规模下，网约车服务显著减少了乘客等待时间、空驶里程和能耗，尤其在低需求时段和偏远地区。扩大车队规模效益递减，地理围栏会恶化整体性能，而针对特定区域的需求管理可有效减少乘客等待时间。

### 背景

网约车服务的迅速扩张已重塑城市按需出行模式，但网约车与传统扬招出租车的相对表现以及相关政策干预的有效性尚不清楚。

### 目的

评估网约车与传统扬招出租车两种出行服务模式的性能，并检验时空特性和三种政策（车队规模管理、地理围栏和需求管理）对网约车服务性能的影响。

### 方法

提出一个整合基于图论的行程-车辆匹配机制和真实巡游出租车运营数据的仿真框架，模拟中国成都的网约车服务，并使用三个关键绩效指标进行评估：平均乘客等待时间、平均空驶里程和平均空驶能耗。

### 主要发现

在相同车队规模和行程需求下，不进行巡游的网约车服务使APWT、ADM和ADEC分别减少81%、75%和72.1%；这些改进在午夜低需求时段和偏远地区最为显著；扩大车队规模会产生边际效益递减；地理围栏会恶化整体性能但改善市中心服务；针对高吸引力、低需求地区的需求管理可有效减少乘客等待时间而不增加空驶成本。

### 结论

网约车服务相比传统扬招出租车有显著性能优势，但不同政策效果各异，需要根据具体情况制定合适的干预措施。

### 翻译

网约车服务的迅速扩张已显著重塑城市按需出行模式，但网约车与传统扬招出租车相比的表现如何，以及相关政策干预的有效性仍不清楚。本研究提出了一个整合基于图论的行程-车辆匹配机制与真实巡游出租车运营数据的仿真框架，以模拟中国成都的网约车服务。从三个关键绩效指标评估了两种按需出行服务模式（即网约车和扬招出租车）的性能：平均乘客等待时间、平均空驶里程和平均空驶能耗。我们进一步检验了时空特性和三种类型政策（车队规模管理、地理围栏和需求管理）对网约车服务性能的影响。结果显示，在相同的车队规模和行程需求下，不进行巡游的网约车服务实现了显著改进，分别使APWT、ADM和ADEC减少81%、75%和72.1%。这些改进在午夜低需求时段和机场等偏远地区最为明显。分析还表明，对于网约车服务：(1)扩大车队规模会产生边际效益递减；(2)地理围栏会恶化整体性能，但能改善对市中心所有行程的服务性能；(3)针对高吸引力、低需求地区行程的需求侧管理可以有效减少乘客等待时间，而不会增加空驶成本。


### 论文摘要

The rapid expansion of ride-hailing services has significantly reshaped urban on-demand mobility patterns, but it still remains unclear how they perform relative to traditional street-hailing services and how effective are related policy interventions. This study presents a simulation framework integrating a graph theory-based trip-vehicle matching mechanism with real cruising taxi operations data to simulate ride-hailing services in Chengdu, China. The performances of the two on-demand mobility service modes (i.e., ride-hailing and street-hailing) are evaluated in terms of three key performance indicators: average passenger waiting time (APWT), average deadheading miles (ADM), and average deadheading energy consumption (ADEC). We further examine the impacts of spatiotemporal characteristics and three types of policies: fleet size management, geofencing, and demand management, on the performance of ride-hailing services. Results show that under the same fleet size and trip demand as street-hailing taxis, ride-hailing services without cruising achieve substantial improvements, reducing APWT, ADM, and ADEC by 81\%, 75\%, and 72.1\%, respectively. These improvements are most pronounced during midnight low-demand hours and in remote areas such as airports. Our analysis also reveals that for ride-hailing service, (1) expanding fleet size yields diminishing marginal benefits; (2) geofencing worsens overall performance while it improves the performance of serving all trips within the city center; and (3) demand-side management targeting trips to high-attraction and low-demand areas can effectively reduce passenger waiting time without increasing deadheading costs.

---

## 329. RadioSim Agent: Combining Large Language Models and Deterministic EM Simulators for Interactive Radio Map Analysis

**论文链接:** [http://arxiv.org/abs/2511.05912v1](http://arxiv.org/abs/2511.05912v1)

**作者:** Sajjad Hussain, Conor Brennan

**发布时间:** 2025-11-08

**备注:** Submitted to EuCAP 2026

### GPT解析

### 总结

本文介绍了RadioSim Agent框架，该框架结合大型语言模型与基于物理的电磁求解器，实现了交互式和可解释的无线电地图生成，为下一代无线系统设计中的智能电磁模拟助手开辟了新途径。

### 背景

确定性电磁模拟器虽能提供准确的无线电传播建模，但通常需要专家配置且缺乏交互灵活性。

### 目的

开发一个能够支持交互式和可解释性无线电地图生成的智能框架。

### 方法

将大型语言模型与基于物理的电磁求解器及视觉推理能力相结合，将射线追踪模型封装为可调用的模拟工具，由能够解释自然语言目标、管理模拟工作流程和可视化分析结果无线电图的LLM进行编排。

### 主要发现

在城市无人机通信场景中演示表明，该代理能够自主选择适当的传播机制，执行确定性模拟，并提供路径损耗行为的语义和视觉摘要。

### 结论

RadioSim Agent提供了多模态可解释性和直观的用户交互，为下一代无线系统设计中的智能电磁模拟助手铺平了道路。

### 翻译

确定性电磁模拟器能提供准确的无线电传播建模，但通常需要专家配置且缺乏交互灵活性。我们提出了RadioSim Agent，这是一个智能框架，将大型语言模型与基于物理的电磁求解器和视觉推理能力相结合，实现了交互式和可解释的无线电地图生成。该框架将射线追踪模型封装为可调用的模拟工具，由能够解释自然语言目标、管理模拟工作流程和可视化分析结果无线电图的LLL进行编排。在城市无人机通信场景中的演示表明，该代理能够自主选择适当的传播机制，执行确定性模拟，并提供路径损耗行为的语义和视觉摘要。结果表明，RadioSim Agent提供了多模态可解释性和直观的用户交互，为下一代无线系统设计中的智能电磁模拟助手铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决确定性电磁模拟器需要专家配置且缺乏交互灵活性的问题。这个问题很重要，因为准确的无线电传播建模是6G无线通信系统设计的基础，让非专家用户通过自然语言与复杂模拟器交互能大大提高无线系统设计的效率和可及性，特别对无人机通信等新兴应用尤为关键。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者先识别现有AI代理框架主要在网络管理层面运行而不涉及物理层建模的局限性，然后结合大型语言模型与确定性电磁模拟器两种技术。设计过程中借鉴了多项现有工作，包括多代理架构、WirelessAgent系统、LLM4WM框架和GeoNR-PSW方法等，但在此基础上增加了物理驱动的路径损耗计算模块作为代理架构中的可调用工具。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将大型语言模型与确定性电磁模拟器结合，创建能理解自然语言指令并自主执行电磁模拟的智能代理，通过'推理-行动-观察'循环工作，并引入视觉推理能力。整体流程是：用户提供自然语言指令→规划器解析意图并提取参数→调用模拟工具库执行电磁计算→执行和输出模块管理工作流程→视觉推理工具分析结果→代理提供多模态摘要。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：设计基于自然语言指令的电磁模拟框架、引入多模态推理能力、提供语义可解释性和交互式实验、开源整个框架。相比之前工作，不同之处在于：RadioSim Agent在物理层建模而非仅网络管理层面工作；支持自主EM模拟而非依赖预计算数据集；直接访问底层EM计算提高透明度；结合物理驱动模块作为可调用工具；能对视觉数据进行自然语言查询。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RadioSim Agent通过将大型语言模型与确定性电磁模拟器相结合，创建了一个能理解自然语言指令、自主执行电磁模拟并提供多模态解释的智能代理框架，使非专家用户能交互式地分析和解释无线电传播环境。'}


### 论文摘要

Deterministic electromagnetic (EM) simulators provide accurate radio propagation modeling but often require expert configuration and lack interactive flexibility. We present RadioSim Agent, an agentic framework that integrates large language models (LLMs) with physics-based EM solvers and vision-enabled reasoning to enable interactive and explainable radio map generation. The framework encapsulates ray-tracing models as callable simulation tools, orchestrated by an LLM capable of interpreting natural language objectives, managing simulation workflows, and visually analyzing resulting radio maps. Demonstrations in urban UAV communication scenarios show that the agent autonomously selects appropriate propagation mechanisms, executes deterministic simulations, and provides semantic and visual summaries of pathloss behavior. The results indicate that RadioSim Agent provides multimodal interpretability and intuitive user interaction, paving the way for intelligent EM simulation assistants in next-generation wireless system design.

---

## 330. Neural Beamforming with Doppler-Aware Sparse Attention for High Mobility Environments

**论文链接:** [http://arxiv.org/abs/2511.03632v1](http://arxiv.org/abs/2511.03632v1)

**作者:** Cemil Vahapoglu, Timothy J. O'Shea, Wan Liu, Sennur Ulukus

**发布时间:** 2025-11-05

### GPT解析

### 总结

本文提出了一种多普勒感知的稀疏神经网络波束成形模型，通过信道自适应稀疏注意力机制在高移动性场景中显著提升波束成形性能。

### 背景

波束成形对提高多天线无线系统的频谱效率和减轻干扰具有重要意义，特别是在密集和高移动性场景中促进空间复用和分集。传统波束成形技术在不良信道条件下性能下降，而基于深度学习的波束成形提供了非线性映射的替代方案。

### 目的

提出一个多普勒感知的稀疏神经网络波束成形模型，在多用户单输入多输出设置中采用信道自适应稀疏注意力机制，解决传统Transformer模型二次注意力复杂度限制的问题。

### 方法

提出一种可根据信道动态在二维时间-频率轴上配置的稀疏结构，理论上确保在p跳内具有完全连通性（p为注意力头数量），并应用于多用户单输入多输出系统。

### 主要发现

在城市宏信道条件下的仿真结果表明，所提出的方法在高移动性场景中显著优于固定模式基线和传统波束成形技术，同时保持结构化稀疏性并控制每个查询的键的注意数量。

### 结论

通过考虑信道动态的稀疏注意力机制，所提出的Doppler-aware Sparse NNBF有效提高了高移动性场景中的波束成形性能，为无线通信中的波束成形提供了新的解决方案。

### 翻译

波束成形对提高多天线无线系统的频谱效率和减轻干扰具有重要意义，能够在密集和高移动性场景中促进空间复用和分集。传统的波束成形技术，如迫零波束成形和最小均方误差波束成形，在不良信道条件下会经历性能下降。基于深度学习的波束成形提供了一种替代方案，通过从信道状态信息到波束成形权重的非线性映射，提高对动态信道环境的鲁棒性。基于Transformer的模型因其能够建模时间和频率上的长程依赖关系而特别有效。然而，它们的二次注意力复杂度限制了在大型OFDM网格中的可扩展性。最近的研究通过稀疏注意力机制解决了这个问题，该机制降低了复杂度同时保持了表达能力，但通常采用的模式忽视了信道动态，因为它们并非专门为无线通信场景设计。在这项工作中，我们提出了一种多普勒感知的稀疏神经网络波束成形模型，该模型在多用户单输入多输出设置中集成了信道自适应稀疏注意力机制。所提出的稀疏结构可以根据信道动态在二维时间-频率轴上配置，理论上可以确保在p跳内具有完全连通性，其中p是注意力头的数量。在城市宏信道条件下的仿真结果表明，Doppler-aware Sparse NNBF在高移动性场景中显著优于固定模式基线（称为标准稀疏NNBF）和传统波束成形技术ZFBF和MMSE波束成形，同时保持结构化稀疏性并控制每个查询的键的注意数量。


### 论文摘要

Beamforming has significance for enhancing spectral efficiency and mitigating interference in multi-antenna wireless systems, facilitating spatial multiplexing and diversity in dense and high mobility scenarios. Traditional beamforming techniques such as zero-forcing beamforming (ZFBF) and minimum mean square error (MMSE) beamforming experience performance deterioration under adverse channel conditions. Deep learning-based beamforming offers an alternative with nonlinear mappings from channel state information (CSI) to beamforming weights by improving robustness against dynamic channel environments. Transformer-based models are particularly effective due to their ability to model long-range dependencies across time and frequency. However, their quadratic attention complexity limits scalability in large OFDM grids. Recent studies address this issue through sparse attention mechanisms that reduce complexity while maintaining expressiveness, yet often employ patterns that disregard channel dynamics, as they are not specifically designed for wireless communication scenarios. In this work, we propose a Doppler-aware Sparse Neural Network Beamforming (Doppler-aware Sparse NNBF) model that incorporates a channel-adaptive sparse attention mechanism in a multi-user single-input multiple-output (MU-SIMO) setting. The proposed sparsity structure is configurable along 2D time-frequency axes based on channel dynamics and is theoretically proven to ensure full connectivity within p hops, where p is the number of attention heads. Simulation results under urban macro (UMa) channel conditions show that Doppler-aware Sparse NNBF significantly outperforms both a fixed-pattern baseline, referred to as Standard Sparse NNBF, and conventional beamforming techniques ZFBF and MMSE beamforming in high mobility scenarios, while maintaining structured sparsity with a controlled number of attended keys per query.

---

## 331. A Conditional Diffusion Model for Building Energy Modeling Workflows

**论文链接:** [http://arxiv.org/abs/2511.02930v1](http://arxiv.org/abs/2511.02930v1)

**作者:** Saumya Sinha, Alexandre Cortiella, Rawad El Kontar, Andrew Glaws, Ryan King, Patrick Emami

**发布时间:** 2025-11-04

### GPT解析

### 总结

该研究提出使用生成建模方法填补建筑能源模型中的数据空白，通过基于表格扩散的框架处理异构建筑特征，并开发条件扩散能力根据已知属性推断缺失特征。研究通过大规模住宅建筑数据训练模型，并通过案例验证了方法的实用性。

### 背景

理解社区当前能源消费行为对未来能源决策至关重要，但城市能源模型需要详细建筑特征数据，而这些数据通常未知、获取成本高或不可用。

### 目的

使用生成建模方法生成真实建筑属性填补数据空白，为能源模型提供完整输入特征。

### 方法

采用基于表格扩散的框架处理异构建筑特征；通过训练包含220万栋住宅建筑的大规模模型学习复杂模式；开发条件扩散能力根据已知属性推断缺失特征；通过比较条件分布和案例研究验证方法。

### 主要发现

生成的条件分布与基础数据分布相符；巴尔的摩住宅区域案例研究展示了方法在实际应用中的效用。

### 结论

生成建模有潜力加速建筑能源建模工作流程。

### 翻译

了解社区当前的能源消费行为对于为未来的能源使用决策提供信息和实现高效能源管理至关重要。用于模拟这些能源使用模式的城市能源模型需要大量具有详细建筑特征的数据才能获得准确结果。然而，此类单个建筑级别的详细特征通常未知且获取成本高，或者不可用。通过这项工作，我们提出使用生成建模方法来生成真实的建筑属性，以填补数据空白，最终提供完整特征作为能源模型的输入。我们的模型通过训练包含220万栋住宅建筑的大规模住宅建筑库存模型，学习复杂的建筑级别模式。我们采用了一种基于表格扩散的框架，专为处理表格建筑数据中的异构（离散和连续）特征而设计，如占用率、楼层面积、供暖、制冷和其他设备详情。我们开发了条件扩散能力，能够根据已知属性推断缺失的建筑特征。我们对条件扩散模型进行了全面的验证，首先通过比较生成的条件分布与基础数据分布，其次通过巴尔的摩住宅区域的案例研究，展示了我们方法的实用性。我们的工作首次展示了生成建模在加速建筑能源建模工作流程方面的潜力。


### 论文摘要

Understanding current energy consumption behavior in communities is critical for informing future energy use decisions and enabling efficient energy management. Urban energy models, which are used to simulate these energy use patterns, require large datasets with detailed building characteristics for accurate outcomes. However, such detailed characteristics at the individual building level are often unknown and costly to acquire, or unavailable. Through this work, we propose using a generative modeling approach to generate realistic building attributes to fill in the data gaps and finally provide complete characteristics as inputs to energy models. Our model learns complex, building-level patterns from training on a large-scale residential building stock model containing 2.2 million buildings. We employ a tabular diffusion-based framework that is designed to handle heterogeneous (discrete and continuous) features in tabular building data, such as occupancy, floor area, heating, cooling, and other equipment details. We develop a capability for conditional diffusion, enabling the imputation of missing building characteristics conditioned on known attributes. We conduct a comprehensive validation of our conditional diffusion model, firstly by comparing the generated conditional distributions against the underlying data distribution, and secondly, by performing a case study for a Baltimore residential region, showing the practical utility of our approach. Our work is one of the first to demonstrate the potential of generative modeling to accelerate building energy modeling workflows.

---

## 332. On Systematic Performance of 3-D Holographic MIMO: Clarke, Kronecker, and 3GPP Models

**论文链接:** [http://arxiv.org/abs/2511.01780v1](http://arxiv.org/abs/2511.01780v1)

**作者:** Quan Gao, Shuai S. A. Yuan, Zhanwen Wang, Wanchen Yang, Chongwen Huang, Xiaoming Chen, Wei E. I. Sha

**发布时间:** 2025-11-03

**备注:** 11 pages, 17 figures, submitted to Electromagnetic Science

### GPT解析

### 总结

本文研究了三维全息MIMO技术在6G网络中的应用，分析了其如何克服传统平面实现的局限性，包括空间相关性和亚波长间距下的互耦问题，从而提高有效自由度和信道容量。

### 背景

全息MIMO是6G网络的关键使能技术，但传统平面实现存在亚波长间距下的空间相关性和互耦问题，这限制了有效自由度(EDOF)和信道容量。

### 目的

系统性地评估三维全息MIMO阵列，同时考虑电磁特性（如互耦和辐射效率），并在Clarke、Kronecker和3GPP信道模型下进行分析。

### 方法

使用解析推导和全波模拟方法，对三维阵列进行电磁特性分析和性能评估，并与传统二维阵列进行比较。

### 主要发现

三维架构比平面基准实现更高的EDOF、更窄的波束宽度和显著的容量提升；在3GPP城市宏信道中，当水平元件间距为0.3波长时，三维配置比传统二维阵列提供约20%的容量提升；三维设计在实际条件下具有鲁棒性和可扩展性。

### 结论

这些发现弥合了理论可行性与实际部署之间的差距，为下一代6G基站阵列设计提供了指导。

### 翻译

全息多输入多输出（MIMO）已成为6G网络的关键使能技术，然而传统的平面实现存在亚波长间距下的空间相关性和互耦问题，这从根本上限制了有效自由度（EDOF）和信道容量。三维（3-D）全息MIMO通过利用扩大有效孔径和释放额外空间模式的体积阵列配置，提供了一条克服这些限制的途径。这项工作首次系统性地评估，将电磁（EM）特性（如互耦和辐射效率）纳入到Clarke、Kronecker和第三代合作伙伴计划（3GPP）信道模型下的三维阵列分析中。解析推导和全波模拟表明，与平面基准相比，三维架构实现了更高的EDOF、更窄的波束宽度和显著的容量提升。在水平元件间距为0.3波长的3GPP城市宏信道中，三维配置比传统二维阵列提供约20%的容量提升，证实了体积设计在实际条件下的鲁棒性和可扩展性。这些发现弥合了理论可行性与实际部署之间的差距，为下一代6G基站阵列设计提供了指导。


### 论文摘要

Holographic multiple-input multiple-output (MIMO) has emerged as a key enabler for 6G networks, yet conventional planar implementations suffer from spatial correlation and mutual coupling at sub-wavelength spacing, which fundamentally limit the effective degrees of freedom (EDOF) and channel capacity. Three-dimensional (3-D) holographic MIMO offers a pathway to overcome these constraints by exploiting volumetric array configurations that enlarge the effective aperture and unlock additional spatial modes. This work presents the first systematic evaluation that jointly incorporates electromagnetic (EM) characteristics, such as mutual coupling and radiation efficiency, into the analysis of 3-D arrays under Clarke, Kronecker, and standardized 3rd Generation Partnership Project (3GPP) channel models. Analytical derivations and full-wave simulations demonstrate that 3-D architectures achieve higher EDOF, narrower beamwidths, and notable capacity improvements compared with planar baselines. In 3GPP urban macro channels with horizontal element spacing of 0.3 lambda, 3-D configurations yield approximately 20% capacity improvement over conventional 2-D arrays, confirming the robustness and scalability of volumetric designs under realistic conditions. These findings bridge the gap between theoretical feasibility and practical deployment, offering design guidance for next-generation 6G base station arrays.

---

## 333. Dynamic Population Distribution Aware Human Trajectory Generation with Diffusion Model

**论文链接:** [http://arxiv.org/abs/2511.01929v1](http://arxiv.org/abs/2511.01929v1)

**作者:** Qingyue Long, Can Rong, Tong Li, Yong Li

**发布时间:** 2025-11-02

### GPT解析

### 总结

本文提出了一种基于扩散模型的新型轨迹生成框架，通过整合动态人口分布约束来提高人类轨迹生成的质量，解决了直接使用真实轨迹数据面临的隐私、成本和质量问题。

### 背景

人类轨迹数据在城市规划、交通工程和公共卫生中至关重要，但直接使用真实轨迹数据面临隐私问题、数据获取成本和数据质量等挑战。

### 目的

开发一种轨迹生成方法来模拟人类移动行为，并解决现有方法忽视人口分布对轨迹生成影响的问题。

### 方法

提出基于扩散模型的轨迹生成框架，构建空间图增强轨迹空间相关性，设计动态人口分布感知的降噪网络捕捉人类移动行为的时空依赖性及人口分布影响。

### 主要发现

实验表明模型生成的轨迹在关键统计指标上与真实轨迹相似，性能优于最先进算法54%以上。

### 结论

该方法通过考虑人口分布因素，有效解决了直接使用真实轨迹数据面临的挑战，提高了轨迹生成质量。

### 翻译

人类轨迹数据在城市规划、交通工程和公共卫生中至关重要。然而，直接使用真实轨迹数据通常面临隐私问题、数据获取成本和数据质量等挑战。解决这些挑战的一个实用方法是轨迹生成，这是一种模拟人类移动行为的方法。现有的轨迹生成方法主要侧重于捕捉个体移动模式，但常常忽视了人口分布对轨迹生成的影响。实际上，动态人口分布反映了不同区域人口密度的变化，显著影响个体移动行为。因此，我们提出了一种基于扩散模型的新型轨迹生成框架，集成了动态人口分布约束来指导高质量的生成结果。具体而言，我们构建了一个空间图来增强轨迹的空间相关性。然后，我们设计了一个动态人口分布感知的降噪网络，以捕捉人类移动行为的时空依赖性以及降噪过程中人口分布的影响。大量实验表明，我们模型生成的轨迹在某些关键统计指标上能够与真实轨迹相似，性能优于最先进算法54%以上。


### 论文摘要

Human trajectory data is crucial in urban planning, traffic engineering, and public health. However, directly using real-world trajectory data often faces challenges such as privacy concerns, data acquisition costs, and data quality. A practical solution to these challenges is trajectory generation, a method developed to simulate human mobility behaviors. Existing trajectory generation methods mainly focus on capturing individual movement patterns but often overlook the influence of population distribution on trajectory generation. In reality, dynamic population distribution reflects changes in population density across different regions, significantly impacting individual mobility behavior. Thus, we propose a novel trajectory generation framework based on a diffusion model, which integrates the dynamic population distribution constraints to guide high-fidelity generation outcomes. Specifically, we construct a spatial graph to enhance the spatial correlation of trajectories. Then, we design a dynamic population distribution aware denoising network to capture the spatiotemporal dependencies of human mobility behavior as well as the impact of population distribution in the denoising process. Extensive experiments show that the trajectories generated by our model can resemble real-world trajectories in terms of some critical statistical metrics, outperforming state-of-the-art algorithms by over 54%.

---

## 334. Single-agent Reinforcement Learning Model for Regional Adaptive Traffic Signal Control

**论文链接:** [http://arxiv.org/abs/2511.00551v1](http://arxiv.org/abs/2511.00551v1)

**作者:** Qiang Li, Ningjing Zeng, Lina Yu

**发布时间:** 2025-11-01

### GPT解析

### 总结

该研究提出了一种基于强化学习的单智能体区域自适应交通信号控制模型，与探测车辆技术兼容，能够有效缓解大规模区域拥堵。

### 背景

多项研究已采用强化学习解决区域自适应交通信号控制的挑战，现有研究主要采用多智能体框架，但多智能体框架在可扩展性方面存在挑战。交通信号控制问题本质上是需要单智能体框架的，因为它依赖于单一控制中心的集中管理，该中心可以监控研究区域内所有道路的交通状况并协调所有交叉口的控制。

### 目的

提出一种与探测车辆技术兼容的单智能体基于强化学习的区域自适应交通信号控制模型。

### 方法

设计基于强化学习的单智能体区域ATSC模型，其关键组件包括状态、动作和奖励函数定义。状态和奖励函数基于队列长度定义以促进学习和管理拥堵，动作设计用于调节队列动态。使用的队列长度定义与常规定义略有不同，但与拥堵状态密切相关，且允许使用探测车辆的路段行程时间数据进行可靠估计。在SUMO仿真平台上对方法进行了全面评估。

### 主要发现

通过协调多交叉口控制，所提出的模型能够有效缓解大规模区域拥堵。使用探测车辆数据可以增强所提方法的广泛部署潜力，因为探测车辆数据已覆盖大多数城市道路。

### 结论

单智能体框架比多智能体框架更适合交通信号控制问题。基于队列长度设计的强化学习状态和奖励函数可以有效管理交通拥堵，与探测车辆技术兼容的方法具有实际部署的潜力。

### 翻译

多项研究已采用强化学习来解决区域自适应交通信号控制(ATSC)的挑战，并取得了有希望的结果。在该领域，现有研究主要采用多智能体框架。然而，多智能体框架的采用对可扩展性构成了挑战。相反，交通信号控制(TSC)问题需要单智能体框架。TSC本质上依赖于单一控制中心的集中管理，该中心可以监控研究区域内所有道路的交通状况并协调所有交叉口的控制。本文提出了一种与探测车辆技术兼容的单智能体基于强化学习的区域ATSC模型。强化学习设计的关键组件包括状态、动作和奖励函数的定义。为了促进学习和管理拥堵，状态和奖励函数都基于队列长度定义，动作设计用于调节队列动态。本研究中使用的队列长度定义与常规定义略有不同，但与拥堵状态密切相关。更重要的是，它允许使用探测车辆的路段行程时间数据进行可靠估计。由于探测车辆数据已覆盖大多数城市道路，这一特点增强了所提方法的广泛部署潜力。使用SUMO仿真平台对方法进行了全面评估。实验结果表明，所提出的模型通过协调多交叉口控制，有效缓解了大规模区域拥堵水平。


### 论文摘要

Several studies have employed reinforcement learning (RL) to address the challenges of regional adaptive traffic signal control (ATSC) and achieved promising results. In this field, existing research predominantly adopts multi-agent frameworks. However, the adoption of multi-agent frameworks presents challenges for scalability. Instead, the Traffic signal control (TSC) problem necessitates a single-agent framework. TSC inherently relies on centralized management by a single control center, which can monitor traffic conditions across all roads in the study area and coordinate the control of all intersections. This work proposes a single-agent RL-based regional ATSC model compatible with probe vehicle technology. Key components of the RL design include state, action, and reward function definitions. To facilitate learning and manage congestion, both state and reward functions are defined based on queue length, with action designed to regulate queue dynamics. The queue length definition used in this study differs slightly from conventional definitions but is closely correlated with congestion states. More importantly, it allows for reliable estimation using link travel time data from probe vehicles. With probe vehicle data already covering most urban roads, this feature enhances the proposed method's potential for widespread deployment. The method was comprehensively evaluated using the SUMO simulation platform. Experimental results demonstrate that the proposed model effectively mitigates large-scale regional congestion levels via coordinated multi-intersection control.

---

## 335. Robust Single-Agent Reinforcement Learning for Regional Traffic Signal Control Under Demand Fluctuations

**论文链接:** [http://arxiv.org/abs/2511.00549v1](http://arxiv.org/abs/2511.00549v1)

**作者:** Qiang Li, Jin Niu, Lina Yu

**发布时间:** 2025-11-01

### GPT解析

### 总结

本文提出了一种基于单智能体强化学习的区域自适应交通信号控制框架，通过集中式决策避免了多智能体系统的协调复杂性，显著减少了交通拥堵。

### 背景

交通拥堵主要由交叉路口排队引起，严重影响城市生活质量、安全、环境质量和经济效率，而传统交通信号控制系统的优化模型往往无法捕捉现实世界交通的复杂性和动态性。

### 目的

开发一种新型单智能体强化学习框架，用于区域自适应交通信号控制，避免多智能体系统的协调问题，提高交通控制效率。

### 方法

使用邻接矩阵编码道路网络拓扑和实时排队状态，集成当前信号定时参数，利用DreamerV3世界模型的学习能力，通过顺序选择交叉口并调整信号相位分割来调节交通流，奖励设计优先考虑排队消散。

### 主要发现

在SUMO模拟实验中，该框架在多级别(10%、20%、30%)起点-终点需求波动场景下表现出强大的抗波动能力，显著减少了队列长度。

### 结论

该研究为与探测车辆技术兼容的智能交通控制建立了新范式，未来研究将专注于提高实际适用性和探索应急事件的区域优化机制。

### 翻译

交通拥堵主要由交叉路口排队引起，严重影响城市生活质量、安全、环境质量和经济效率。虽然交通信号控制系统(TSC)有缓解拥堵的潜力，但传统优化模型往往无法捕捉现实世界交通的复杂性和动态性。本研究引入了一种新型的单智能体强化学习(RL)框架，用于区域自适应TSC，通过集中式决策范式避免了多智能体系统中的协调复杂性。该模型采用邻接矩阵统一编码道路网络拓扑、从探测车辆数据获取的实时排队状态和当前信号定时参数。利用DreamerV3世界模型的高效学习能力，智能体学习控制策略，通过顺序选择交叉口并调整其信号相位分割来调节交通流入/流出，类似于反馈控制系统。奖励设计优先考虑排队消散，直接将拥堵指标(队列长度)与控制动作关联。在SUMO进行的模拟实验证明了模型的有效性：在具有多级别(10%、20%、30%)起点-终点(OD)需求波动的推理场景下，该框架表现出强大的抗波动能力并显著减少了队列长度。这项工作为与探测车辆技术兼容的智能交通控制建立了新范式。未来研究将专注于通过在训练中纳入随机OD需求波动来提高实际适用性，并探索应急事件的区域优化机制。


### 论文摘要

Traffic congestion, primarily driven by intersection queuing, significantly impacts urban living standards, safety, environmental quality, and economic efficiency. While Traffic Signal Control (TSC) systems hold potential for congestion mitigation, traditional optimization models often fail to capture real-world traffic complexity and dynamics. This study introduces a novel single-agent reinforcement learning (RL) framework for regional adaptive TSC, circumventing the coordination complexities inherent in multi-agent systems through a centralized decision-making paradigm. The model employs an adjacency matrix to unify the encoding of road network topology, real-time queue states derived from probe vehicle data, and current signal timing parameters. Leveraging the efficient learning capabilities of the DreamerV3 world model, the agent learns control policies where actions sequentially select intersections and adjust their signal phase splits to regulate traffic inflow/outflow, analogous to a feedback control system. Reward design prioritizes queue dissipation, directly linking congestion metrics (queue length) to control actions. Simulation experiments conducted in SUMO demonstrate the model's effectiveness: under inference scenarios with multi-level (10%, 20%, 30%) Origin-Destination (OD) demand fluctuations, the framework exhibits robust anti-fluctuation capability and significantly reduces queue lengths. This work establishes a new paradigm for intelligent traffic control compatible with probe vehicle technology. Future research will focus on enhancing practical applicability by incorporating stochastic OD demand fluctuations during training and exploring regional optimization mechanisms for contingency events.

---

## 336. Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features

**论文链接:** [http://arxiv.org/abs/2511.00126v1](http://arxiv.org/abs/2511.00126v1)

**作者:** Lu Bowen

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种动态多专家门控框架，能够根据每个样本自适应选择最可靠的轨迹预测器，在复杂驾驶场景中提高了轨迹预测的可靠性。

### 背景

现有的深度轨迹预测器在复杂的长尾驾驶场景中仍然不可靠，主流的'单一模型适用于所有场景'范式存在局限性，在安全关键的城市场景中，简单的基于物理的模型有时可以优于高级网络。

### 目的

弥补现有轨迹预测方法的不足，提出一个动态多专家门控框架，能够根据每个样本自适应选择最可靠的轨迹预测器。

### 方法

提出动态多专家门控框架，包含三个专家模型：基于物理的LSTM、Transformer和微调后的GameFormer。利用内部模型信号（元特征）如稳定性和不确定性来选择最佳模型，将轨迹专家选择公式化为基于内部模型信号的成对排序问题，直接优化决策质量，无需后校准。

### 主要发现

内部模型信号比几何场景描述符更具信息量；在nuPlan-mini数据集上，LLM增强的三专家门控实现了2.567米的最终位移误差，比GameFormer减少了9.5%的误差；达到了57.8%的oracle性能边界；在开环模拟中，左转场景的FDE减少了约10%。

### 结论

自适应混合系统增强了安全关键自动驾驶中的轨迹可靠性，为静态单一模型范式提供了实用的超越途径。

### 翻译

最近的深度轨迹预测器（如Jiang等人，2023；Zhou等人，2022）已经取得了很强的平均准确性，但在复杂的长尾驾驶场景中仍然不可靠。这些局限性揭示了主流的'单一模型适用于所有场景'范式的弱点，特别是在安全关键的城市场景中，简单的基于物理的模型偶尔可以优于高级网络（Kalman，1960）。为了弥补这一差距，我们提出了一个动态多专家门控框架，能够根据每个样本自适应地在基于物理的LSTM、Transformer和微调后的GameFormer之间选择最可靠的轨迹预测器。我们的方法利用内部模型信号（元特征），如稳定性和不确定性（Gal和Ghahramani，2016），我们证明这些信号比几何场景描述符更具信息量。据我们所知，这是第一个将轨迹专家选择公式化为基于内部模型信号的成对排序问题（Burges等人，2005）的工作，直接优化决策质量而无需后校准。在nuPlan-mini数据集（Caesar等人，2021）上对1,287个样本进行评估，我们的LLM增强的三专家门控实现了2.567米的最终位移误差（FDE），比GameFormer（2.835米）减少了9.5%，并实现了57.8%的oracle性能边界。在开环模拟中，经过轨迹视界对齐后，相同配置在左转场景中将FDE降低了约10%，展示了在离线验证和开环评估中的一致改进。这些结果表明，自适应混合系统增强了安全关键自动驾驶中的轨迹可靠性，为静态单一模型范式提供了实用的超越途径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶中轨迹预测模型在复杂、长尾驾驶场景中表现不可靠的问题。这个问题在现实中非常重要，因为自动驾驶系统需要在各种场景下都能可靠工作，特别是在安全关键的长尾场景中。单一深度学习模型无法同时处理结构化的低不确定性动态和复杂的多智能体交互，而简单的物理模型有时在特定场景下表现更好，因此需要一种动态选择最适合模型的方法来提高整体系统的可靠性和安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到单一模型无法应对所有驾驶场景，特别是复杂的长尾场景。然后提出假设：通过动态选择最适合的预测模型可以显著提高性能。作者分析了现有混合专家(MoE)架构、不确定性量化和学习排序等方法，发现现有门控机制要么依赖固定的启发式路由，要么使用手工制作的几何特征，这些与实际误差相关性弱。因此，作者设计了一个动态多专家门控框架，提取模型内部元特征，将选择任务形式化为成对排序，并引入LLM监督器。该方法借鉴了混合专家架构、不确定性量化、学习排序和LLM在自动驾驶中的应用等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过动态选择最适合特定场景的预测模型来提高轨迹预测的可靠性，而不是试图创建一个完美的单一模型。整体实现流程包括：1)构建互补的专家集合（物理模型、交互模型和长尾专家）；2)从专家内部提取元特征（不确定性、稳定性和物理违规率）；3)使用排序门控基于元特征选择最佳专家；4)在低置信度或高风险场景中，由LLM监督器提供语义覆盖和决策建议。整个系统在训练时最小化排序损失，在评估时使用FDE和ORR等指标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)元特征提取，从模型内部状态提取预测性能信号；2)基于排序的门控，将模型选择形式化为成对排序任务，避免校准问题；3)LLM监督器，为低置信度场景提供语义理解；4)系统评估方法，首次对10+种门控策略进行严格比较。相比之前工作，本文不依赖几何特征而是使用模型元特征；采用排序而非分类或回归范式；引入LLM提供语义解释；提供全面的门控策略评估和消融研究。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于元特征和排序的动态门控框架，结合大型语言模型监督，实现了自动驾驶轨迹预测模型的自适应选择，显著提高了系统在复杂长尾场景中的可靠性和安全性。'}


### 论文摘要

Recent deep trajectory predictors (e.g., Jiang et al., 2023; Zhou et al., 2022) have achieved strong average accuracy but remain unreliable in complex long-tail driving scenarios. These limitations reveal the weakness of the prevailing "one-model-fits-all" paradigm, particularly in safety-critical urban contexts where simpler physics-based models can occasionally outperform advanced networks (Kalman, 1960). To bridge this gap, we propose a dynamic multi-expert gating framework that adaptively selects the most reliable trajectory predictor among a physics-informed LSTM, a Transformer, and a fine-tuned GameFormer on a per-sample basis.   Our method leverages internal model signals (meta-features) such as stability and uncertainty (Gal and Ghahramani, 2016), which we demonstrate to be substantially more informative than geometric scene descriptors. To the best of our knowledge, this is the first work to formulate trajectory expert selection as a pairwise-ranking problem over internal model signals (Burges et al., 2005), directly optimizing decision quality without requiring post-hoc calibration.   Evaluated on the nuPlan-mini dataset (Caesar et al., 2021) with 1,287 samples, our LLM-enhanced tri-expert gate achieves a Final Displacement Error (FDE) of 2.567 m, representing a 9.5 percent reduction over GameFormer (2.835 m), and realizes 57.8 percent of the oracle performance bound. In open-loop simulations, after trajectory horizon alignment, the same configuration reduces FDE on left-turn scenarios by approximately 10 percent, demonstrating consistent improvements across both offline validation and open-loop evaluation. These results indicate that adaptive hybrid systems enhance trajectory reliability in safety-critical autonomous driving, providing a practical pathway beyond static single-model paradigms.

---

## 337. A Hierarchical Deep Learning Model for Predicting Pedestrian-Level Urban Winds

**论文链接:** [http://arxiv.org/abs/2510.27101v1](http://arxiv.org/abs/2510.27101v1)

**作者:** Reda Snaiki, Jiachen Lu, Shaopeng Li, Negin Nazarian

**发布时间:** 2025-10-31

### GPT解析

### 总结

本文提出了一种基于深度学习的分层方法，用于准确预测行人层面的城市风场。该方法采用两阶段预测器-精炼器框架，结合U-Net和条件生成对抗网络(cGAN)，有效捕捉传统方法缺失的高频细节，如梯度变化和峰值风速。

### 背景

传统基于深度学习的代理模型虽为高保真计算流体动力学(CFD)模拟提供了计算高效的替代方案，但通常只能生成低频预测（邻近像素的平均值），忽略了关键的高频细节，如梯度变化和峰值风速。

### 目的

开发一种能够准确预测行人层面城市风流的分层方法，解决传统方法无法捕捉高频细节的问题。

### 方法

采用两阶段预测器-精炼器框架：第一阶段使用U-Net架构从城市几何形状生成基线预测；第二阶段使用条件生成对抗网络(cGAN)通过恢复缺失的高频内容来精炼基线预测。cGAN的生成器采用多尺度架构和逐步核大小，能够同时学习全局流动结构和细粒度的局部特征。在UrbanTALES数据集上进行了训练和验证。

### 主要发现

提出的分层框架显著优于基线预测器。在定性方面：有效解决了高速风射流和复杂湍流尾流的分辨率问题，改进了风统计；在定量方面：预测精度显著提高（例如，训练集的RMSE降低了76%，验证集的RMSE降低了60%）。

### 结论

这项工作为城市规划、行人舒适度评估和风安全分析中的代理模型预测保真度提供了一种有效且稳健的改进方法。所提出的模型将集成到一个名为Feilian Version 2的交互式网络平台中。

### 翻译

基于深度学习的代理模型为高保真计算流体动力学(CFD)模拟提供了一种计算高效的替代方案，用于预测城市风流。然而，传统方法通常只能产生低频预测（本质上是邻近像素的平均值），忽略了关键的高频细节，如梯度变化和峰值风速。本研究提出了一种用于准确预测行人层面城市风场的分层方法，采用两阶段预测器-精炼器框架。在第一阶段，U-Net架构从城市几何形状生成基线预测。在第二阶段，条件生成对抗网络(cGAN)通过恢复缺失的高频内容来精炼此基线预测。cGAN的生成器采用多尺度架构和逐步核大小，能够同时学习全局流动结构和细粒度的局部特征。在具有全面城市配置的UrbanTALES数据集上进行了训练和验证，所提出的分层框架显著优于基线预测器。在解决高速风射流和复杂湍流尾流的分辨率以及风统计方面有明显的定性改进，结果在预测精度方面有定量提升（例如，训练集的RMSE降低了76%，验证集的RMSE降低了60%）。这项工作为城市规划、行人舒适度评估和风安全分析中的代理模型预测保真度提供了一种有效且稳健的方法。所提出的模型将集成到一个名为Feilian Version 2的交互式网络平台中。


### 论文摘要

Deep learning-based surrogate models offer a computationally efficient alternative to high-fidelity computational fluid dynamics (CFD) simulations for predicting urban wind flow. However, conventional approaches usually only yield low-frequency predictions (essentially averaging values from proximate pixels), missing critical high-frequency details such as sharp gradients and peak wind speeds. This study proposes a hierarchical approach for accurately predicting pedestrian-level urban winds, which adopts a two-stage predictor-refiner framework. In the first stage, a U-Net architecture generates a baseline prediction from urban geometry. In the second stage, a conditional Generative Adversarial Network (cGAN) refines this baseline by restoring the missing high-frequency content. The cGAN's generator incorporates a multi-scale architecture with stepwise kernel sizes, enabling simultaneous learning of global flow structures and fine-grained local features. Trained and validated on the UrbanTALES dataset with comprehensive urban configurations, the proposed hierarchical framework significantly outperforms the baseline predictor. With a marked qualitative improvement in resolving high-speed wind jets and complex turbulent wakes as well as wind statistics, the results yield quantitative enhancement in prediction accuracy (e.g., RMSE reduced by 76% for the training set and 60% for the validation set). This work presents an effective and robust methodology for enhancing the prediction fidelity of surrogate models in urban planning, pedestrian comfort assessment, and wind safety analysis. The proposed model will be integrated into an interactive web platform as Feilian Version 2.

---

