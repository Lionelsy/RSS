# 今日论文推荐 - 2025-10-16

共 58 篇论文

---

## 1. LiFMCR: Dataset and Benchmark for Light Field Multi-Camera Registration

**论文链接:** [http://arxiv.org/abs/2510.13729v1](http://arxiv.org/abs/2510.13729v1)

**作者:** Aymeric Fleith, Julian Zirbel, Daniel Cremers, Niclas Zeller

**发布时间:** 2025-10-15

**备注:** Accepted at the International Symposium on Visual Computing (ISVC)  2025

### GPT解析

### 总结

本文提出了LiFMCR，一个用于多微透镜阵列光场相机配准的新数据集，提供同步图像序列和高精度姿态数据，用于严格评估多相机光场配准方法。

### 背景

现有光场数据集仅限于单相机设置，通常缺乏外部真实值，限制了多相机光场配准方法的评估。

### 目的

创建一个独特的多相机光场数据集，结合高分辨率光场相机图像和精确的6自由度姿态数据，以实现多相机光场配准方法的严格评估。

### 方法

提供两种互补的配准方法：1)基于RANSAC的鲁棒3D变换估计，使用跨视点点云；2)从单个光场图像估计外源性6-DoF姿态的光场PnP算法。两种方法都明确集成了光场相机模型。

### 主要发现

实验表明，所提出的方法与真实值显示出良好的对齐，支持可靠的多视点光场处理。

### 结论

LiFMCR数据集及其配套方法为多相机光场配准提供了基准，能够准确且可扩展地进行多相机配准。

### 翻译

本文提出了LiFMCR，一个用于多微透镜阵列光场相机配准的新颖数据集。虽然现有的光场数据集仅限于单相机设置且通常缺乏外部真实值，但LiFMCR提供了来自两个高分辨率Raytrix R32光场相机的同步图像序列，以及由Vicon动作捕捉系统记录的高精度6自由度姿态。这种独特组合能够严格评估多相机光场配准方法。作为基准，我们提供了两种互补的配准方法：一种基于RANSAC的鲁棒3D变换估计，使用跨视点点云；以及一种从单个光场图像估计外源性6-DoF姿态的光场PnP算法。两种方法都明确集成了光场相机模型，实现准确且可扩展的多相机配准。实验显示与真实值有良好的对齐，支持可靠的多视点光场处理。项目页面：https://lifmcr.github.io/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决多相机光场相机（plenoptic cameras）注册缺乏标准数据集和基准测试的问题。这个问题很重要，因为准确的3D重建对自主系统和机器人应用至关重要，而结合多个光场相机可以通过立体视觉优势扩展深度范围和精度，提高深度感知能力和场景理解能力。现有的光场数据集通常局限于单相机设置且缺乏外部真实值，限制了多相机注册方法的评估和改进。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有光场数据集的局限性，特别是缺乏多相机设置和精确真实值的问题。他们借鉴了现有工作：使用LiFCal进行内参校准，采用SIFT特征提取和匹配，以及基于RANSAC的3D变换估计方法。作者设计了两种互补的注册方法：一种基于3D点云对齐的RANSAC方法，另一种是首次应用于光场数据的PnP算法。两种方法都明确集成了光场相机模型，以准确处理光场相机的特殊光学和几何特性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提供一个包含同步多视角光场数据和精确6-DoF姿态真实值的数据集，并设计两种互补的多相机注册方法，一种基于3D点云对齐，另一种基于光场PnP算法，都考虑光场相机的特殊光学特性。3D RANSAC方法流程：内参校准→点云生成→SIFT特征提取与匹配→3D RANSAC对齐→计算相机间相对变换。光场PnP方法流程：仅参考相机点云→镜头畸变校正→光场模型透视投影→特征匹配→鲁棒基础矩阵估计→RANSAC PnP→Levenberg-Marquardt优化细化姿态。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出LiFMCR数据集，首次提供同步多视角光场序列和精确6-DoF真实值；2)提出两种互补注册方法，包括基于RANSAC的3D变换估计和首个光场PnP算法；3)两种方法都明确集成光场相机模型；4)提供完整内参和外参校准流程；5)使用Vicon系统提供亚毫米级精度真实值。相比之前工作：现有数据集多为单相机且缺乏真实值，而LiFMCR提供多相机同步数据和精确姿态；现有方法不专门针对光场多相机注册，而本文方法考虑了光场相机特性；首次将PnP算法应用于光场数据。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了LiFMCR数据集和两种互补的光场多相机注册方法，填补了多视角光场数据与精确姿态真实值结合的空白，为光场相机在3D重建、SLAM等应用中的可靠使用提供了基础。'}


### 论文摘要

We present LiFMCR, a novel dataset for the registration of multiple micro lens array (MLA)-based light field cameras. While existing light field datasets are limited to single-camera setups and typically lack external ground truth, LiFMCR provides synchronized image sequences from two high-resolution Raytrix R32 plenoptic cameras, together with high-precision 6-degrees of freedom (DoF) poses recorded by a Vicon motion capture system. This unique combination enables rigorous evaluation of multi-camera light field registration methods.   As a baseline, we provide two complementary registration approaches: a robust 3D transformation estimation via a RANSAC-based method using cross-view point clouds, and a plenoptic PnP algorithm estimating extrinsic 6-DoF poses from single light field images. Both explicitly integrate the plenoptic camera model, enabling accurate and scalable multi-camera registration. Experiments show strong alignment with the ground truth, supporting reliable multi-view light field processing.   Project page: https://lifmcr.github.io/

---

## 2. Characterizing Lidar Point-Cloud Adversities Using a Vector Field Visualization

**论文链接:** [http://arxiv.org/abs/2510.13619v1](http://arxiv.org/abs/2510.13619v1)

**作者:** Daniel Choate, Jason Rife

**发布时间:** 2025-10-15

**DOI:** 10.33012/2024.19864

**备注:** This is the preprint version of the paper published in: Proceedings  of the 37th International Technical Meeting of the Satellite Division of The  Institute of Navigation (ION GNSS+ 2024), September 2024 The final version is  available at https://doi.org/10.33012/2024.19864

### GPT解析

### 总结

本文提出了一种可视化方法，用于辅助分析师分类影响激光雷达扫描匹配的逆境模式，通过生成矢量场图揭示点云数据中的差异模式。

### 背景

激光雷达扫描匹配过程中存在多种逆境模式影响数据质量，分析师需要有效方法来识别和分类这些模式。

### 目的

开发一种离线可视化分析方法，帮助分析师识别和理解影响激光雷达扫描匹配的逆境机制。

### 方法

提出一种生成矢量场图的可视化方法，该图能描述一对已配准点云之间的局部差异，揭示难以从原始数据中提取的模式。

### 主要发现

通过模拟研究和现场实验验证，该方法能够帮助分析师识别和迭代移除逆境机制，逐步聚焦于更细微的数据差异。

### 结论

所提出的可视化方法有效辅助了分析师对激光雷达扫描匹配中逆境模式的分类和分析，提高了数据处理的效率和准确性。

### 翻译

在本文中，我们介绍了一种可视化方法，用于帮助人类分析师分类影响激光雷达扫描匹配的逆境模式。我们的方法适用于离线分析而非实时分析。该方法生成一个矢量场图，用于描述一对已配准点云之间的局部差异。矢量场图能够揭示分析师难以从原始点云数据中提取的模式。在介绍我们的方法后，我们将该过程应用于两个概念验证示例：一个是模拟研究，另一个是现场实验。对于这两个数据集，人类分析师能够推理一系列逆境机制，并从原始数据中迭代地移除这些机制，以帮助将注意力集中在逐渐变小的差异上。


### 论文摘要

In this paper we introduce a visualization methodology to aid a human analyst in classifying adversity modes that impact lidar scan matching. Our methodology is intended for offline rather than real-time analysis. The method generates a vector-field plot that characterizes local discrepancies between a pair of registered point clouds. The vector field plot reveals patterns that would be difficult for the analyst to extract from raw point-cloud data. After introducing our methodology, we apply the process to two proof-of-concept examples: one a simulation study and the other a field experiment. For both data sets, a human analyst was able to reason about a series of adversity mechanisms and iteratively remove those mechanisms from the raw data, to help focus attention on progressively smaller discrepancies.

---

## 3. Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning

**论文链接:** [http://arxiv.org/abs/2510.13307v1](http://arxiv.org/abs/2510.13307v1)

**作者:** Yang Li, Aming Wu, Zihao Zhang, Yahong Han

**发布时间:** 2025-10-15

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了一种基于结构因果模型（SCM）的点云分割新类别发现方法，通过因果表示与推理的联合学习，解决仅使用已标记类别监督信息对新类别进行分割的问题。

### 背景

点云分割中的新类别发现（3D-NCD）是一个挑战性问题，需要仅使用已标记（基础）3D类别的监督信息来学习能够分割未标记（新）3D类别的模型。

### 目的

学习一个模型，仅使用已标记（基础）3D类别的监督信息，能够对未标记（新）3D类别进行分割。

### 方法

引入结构因果模型（SCM）重新形式化3D-NCD问题，提出因果表示与推理的联合学习方法。通过SCM分析基础类别表示中的隐藏混杂因素以及基础类别和新类别之间的因果关系；设计消除混杂因素的因果表示原型；使用图结构建模基础类别因果表示原型与新类别原型之间的因果关系，实现从基础到新类别的因果推理。

### 主要发现

粗略或统计相关性学习可能导致新类别推理的混淆；通过引入因果约束可以准确发现与类别对应的点云表示；所提出的方法在3D和2D NCD语义分割任务上表现出优越性。

### 结论

基于结构因果模型的方法能够有效解决点云分割中的新类别发现问题，通过因果表示与推理的联合学习，实现仅使用已标记类别监督信息对新类别的准确分割。

### 翻译

在本文中，我们专注于点云分割的新类别发现（3D-NCD），旨在学习一个模型，仅使用已标记（基础）3D类别的监督信息，能够对未标记（新）3D类别进行分割。这项任务的关键在于建立点表示与基础类别标签之间的准确相关性，以及基础类别和新类别点之间的表示相关性。粗略或统计相关性学习可能导致新类别推理的混淆。如果在学习过程中施加因果关系作为强相关约束，应该能够准确发现与类别对应的本质点云表示。为此，我们引入结构因果模型（SCM）重新形式化3D-NCD问题，并提出一种新方法，即因果表示与推理的联合学习。具体而言，我们首先通过SCM分析基础类别表示中的隐藏混杂因素以及基础类别和新类别之间的因果关系。我们设计了一个消除混杂因素的因果表示原型，以捕获基础类别的因果表示。然后使用图结构建模基础类别因果表示原型与新类别原型之间的因果关系，实现从基础到新类别的因果推理。在3D和2D NCD语义分割任务上的大量实验和可视化结果证明了我们方法的优越性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文解决的是点云分割中的新类别发现问题，即如何仅使用已知类别的监督信息来训练模型，使其能够分割场景中未标记的新类别物体。这个问题在自动驾驶、机器人感知等真实场景中非常重要，因为这些环境中可能出现各种未预先定义的物体类别，传统'封闭世界'假设的方法无法应对这种开放世界环境，而新类别发现能够减少人工标注负担，使模型能够适应动态变化的环境。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统方法的局限性，指出它们倾向于学习捷径特征而非本质特征，且忽视了基类与新类别之间的因果关系。基于此，作者引入结构因果模型重新形式化问题，并借鉴了因果表示学习、结构因果模型、图卷积网络和生成对抗网络等现有工作。通过这些借鉴，作者设计了因果表示原型学习来消除混杂因素，并使用图结构建模基类与新类别间的因果关系，实现了从未知类别中学习更准确的分割。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过因果表示学习消除点云数据中的非因果特征，捕获基类的本质表示，并建立基类和新类别之间的因果关系模型，实现从已知到未知的知识迁移。整体流程分为三部分：1)因果表示原型学习，使用对抗训练消除混杂因素，生成基类的因果表示原型；2)因果推理图构建，创建包含基类和新类原型的图结构，设计因果自适应邻接矩阵和约束优化图结构；3)基于GCN的伪标签生成，利用图卷积网络处理优化后的图，通过多层传播和邻居聚合为新类别生成高质量伪标签。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将因果学习引入3D NCD领域，专注于学习因果关系而非统计相关性；2)提出因果表示原型学习，通过对抗机制消除混杂因素；3)提出基于图的因果推理方法，显式建模基类到新类别的因果路径。相比之前的工作，本文方法能够处理点云数据中的复杂因果关系，而非仅依赖表面特征相似性，通过因果推理更好地处理新类别的语义关系，减少错误分类，提高了在开放世界环境中的适应性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过引入因果表示学习和因果推理，首次解决了点云分割中新类别发现问题中的因果机制建模，实现了从未知类别中学习更准确、更鲁棒的语义分割。'}


### 论文摘要

In this paper, we focus on Novel Class Discovery for Point Cloud Segmentation (3D-NCD), aiming to learn a model that can segment unlabeled (novel) 3D classes using only the supervision from labeled (base) 3D classes. The key to this task is to setup the exact correlations between the point representations and their base class labels, as well as the representation correlations between the points from base and novel classes. A coarse or statistical correlation learning may lead to the confusion in novel class inference. lf we impose a causal relationship as a strong correlated constraint upon the learning process, the essential point cloud representations that accurately correspond to the classes should be uncovered. To this end, we introduce a structural causal model (SCM) to re-formalize the 3D-NCD problem and propose a new method, i.e., Joint Learning of Causal Representation and Reasoning. Specifically, we first analyze hidden confounders in the base class representations and the causal relationships between the base and novel classes through SCM. We devise a causal representation prototype that eliminates confounders to capture the causal representations of base classes. A graph structure is then used to model the causal relationships between the base classes' causal representation prototypes and the novel class prototypes, enabling causal reasoning from base to novel classes. Extensive experiments and visualization results on 3D and 2D NCD semantic segmentation demonstrate the superiorities of our method.

---

## 4. DAMM-LOAM: Degeneracy Aware Multi-Metric LiDAR Odometry and Mapping

**论文链接:** [http://arxiv.org/abs/2510.13287v1](http://arxiv.org/abs/2510.13287v1)

**作者:** Nishant Chandna, Akshat Kaushal

**发布时间:** 2025-10-15

**备注:** Accepted at IROS Active Perception Workshop

### GPT解析

### 总结

本文提出了一种名为DAMM-LOAM的新型LiDAR SLAM系统，通过点云分类和退化感知算法解决了特征稀疏环境下的定位建图问题，显著提高了室内环境中的导航精度。

### 背景

LiDAR SLAM系统对精确导航和环境重建至关重要。当前点对平面ICP算法在结构化、特征丰富的环境中表现良好，但在特征稀疏、重复几何结构和高频运动场景下表现不佳，导致6自由度姿态估计退化。虽然最先进算法通过添加额外传感模态应对挑战，但纯LiDAR解决方案在这种条件下仍有限制。

### 目的

解决特征稀疏、重复几何结构和高频运动场景下的SLAM退化问题，提出一种新颖的退化感知多度量LiDAR里程计与建图(DAMM-LOAM)模块。

### 方法

通过基于表面法线和邻域分析的点云分类提高建图精度，将点分类为地面、墙壁、屋顶、边缘和非平面点以实现准确对应；应用基于退化的加权最小二乘ICP算法进行精确里程计估计；实现基于ScanContext的后端以支持稳健的回环闭合。

### 主要发现

DAMM-LOAM在里程计准确性方面有显著改进，特别是在长走廊等室内环境中表现突出。

### 结论

DAMM-LOAM系统有效解决了传统LiDAR SLAM在特定场景下的退化问题，通过创新的点云分类和退化感知算法，提高了室内环境中的导航精度。

### 翻译

激光雷达同步定位与建图系统对于在各种应用中实现精确导航和环境重建至关重要。尽管当前点对平面ICP算法在结构化、特征丰富的环境中能有效工作，但在特征稀疏、重复几何结构和高频运动场景下表现不佳。这会导致6自由度姿态估计的退化。大多数最先进算法通过结合额外的传感模态来解决这些挑战，但纯激光雷达解决方案在这种条件下仍然面临限制。为解决这些问题，我们提出了一种新颖的退化感知多度量激光雷达里程计与建图模块。我们的系统通过基于表面法线和邻域分析的点云分类提高了建图精度。点被分类为地面、墙壁、屋顶、边缘和非平面点，从而实现准确的对应关系。然后应用基于退化的加权最小二乘ICP算法进行精确的里程计估计。此外，实现了基于扫描上下文的后端以支持稳健的回环闭合。DAMM-LOAM在里程计准确性方面表现出显著改进，特别是在长走廊等室内环境中。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR SLAM系统在特征稀疏、重复几何结构和高频运动等场景下的退化问题，导致6自由度位姿估计不准确。这个问题很重要，因为许多实际应用场景（如走廊、隧道）都存在特征稀疏问题，而机器人导航、自动驾驶等领域需要在复杂环境中进行精确定位和地图构建，当前系统在这些挑战性场景中表现不佳，限制了技术的广泛应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析传统点对点或点对平面ICP算法的局限性，认识到在变化几何结构和特征稀疏环境下表现不佳的问题。设计方法时借鉴了现有工作：利用NV-LIOM的球形投影法线提取方法，但进一步进行几何分类；借鉴了条件数和特征值分析来检测退化，但设计了新的点级加权方案；并整合了现有的Scan Context算法作为后端。作者不是完全重新发明方法，而是在现有基础上进行了改进和整合创新。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合几何特征分类和退化感知的自适应加权来提高LiDAR SLAM系统在特征稀疏环境中的鲁棒性和准确性。整体流程包括：1)几何特征提取：将点云投影到球形范围图像，估计表面法线，并进行五类几何分类（地面、墙壁、屋顶、边缘、非平面点）；2)点云处理：自适应下采样并建立类别对应的点对；3)退化感知的位姿估计：分析Hessian矩阵特征值，为点分配权重，结合点对点和点对平面残差进行优化；4)后端处理：使用Scan Context进行回环检测和全局优化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)基于法线图的语义特征提取：将点云分为五类而非传统的平面/非平面二分类；2)退化感知的逐点自适应加权：基于Hessian特征值分析为每个点分配权重，而非仅基于点类型数量；3)多度量残差整合：结合点对点和点对平面残差并动态调整权重；4)完整的端到端框架：整合几何特征提取、自适应加权优化和回环检测。相比之前工作，该方法提供了更细致的语义信息、更精确的退化处理和更全面的解决方案。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DAMM-LOAM通过基于法线图的语义特征分类和退化感知的自适应加权，显著提高了LiDAR SLAM系统在特征稀疏环境（如长走廊）中的定位精度和鲁棒性。'}


### 论文摘要

LiDAR Simultaneous Localization and Mapping (SLAM) systems are essential for enabling precise navigation and environmental reconstruction across various applications. Although current point-to-plane ICP algorithms perform effec- tively in structured, feature-rich environments, they struggle in scenarios with sparse features, repetitive geometric structures, and high-frequency motion. This leads to degeneracy in 6- DOF pose estimation. Most state-of-the-art algorithms address these challenges by incorporating additional sensing modalities, but LiDAR-only solutions continue to face limitations under such conditions. To address these issues, we propose a novel Degeneracy-Aware Multi-Metric LiDAR Odometry and Map- ping (DAMM-LOAM) module. Our system improves mapping accuracy through point cloud classification based on surface normals and neighborhood analysis. Points are classified into ground, walls, roof, edges, and non-planar points, enabling accurate correspondences. A Degeneracy-based weighted least squares-based ICP algorithm is then applied for accurate odom- etry estimation. Additionally, a Scan Context based back-end is implemented to support robust loop closures. DAMM-LOAM demonstrates significant improvements in odometry accuracy, especially in indoor environments such as long corridors

---

## 5. Prompt-based Adaptation in Large-scale Vision Models: A Survey

**论文链接:** [http://arxiv.org/abs/2510.13219v1](http://arxiv.org/abs/2510.13219v1)

**作者:** Xi Xiao, Yunbei Zhang, Lin Zhao, Yiyang Liu, Xiaoying Liao, Zheda Mai, Xingjian Li, Xiao Wang, Hao Xu, Jihun Hamm, Xue Lin, Min Xu, Qifan Wang, Tianyang Wang, Cheng Han

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文是一篇关于视觉提示(Visual Prompting, VP)和视觉提示调优(Visual Prompt Tuning, VPT)的综合调查，提出了一种称为基于提示的适应(Prompt-based Adaptation, PA)的统一框架，对现有方法进行了分类，并探讨了PA在不同领域的应用、挑战和未来方向。

### 背景

在计算机视觉领域，VP和VPT作为大规模视觉模型适应的轻量级有效替代方法，在'预训练后微调'范式中迅速发展。然而，当前研究中VP和VPT经常被互换使用，缺乏系统区分这些技术及其各自应用的明确界限。

### 目的

重新审视VP和VPT的设计，将它们概念化为一个统一的PA框架，提供清晰的方法分类，并探索PA在不同领域的应用、挑战和未来方向，为研究人员和实践者提供明确的路线图。

### 方法

提供了一种分类法，将现有方法分为可学习提示、生成提示和非可学习提示，并按注入粒度（像素级和令牌级）进一步组织。同时检查了PA在医学成像、3D点云和视觉语言任务等领域的整合，以及其在测试时适应和可信AI中的作用。

### 主要发现

PA在医学成像、3D点云和视觉语言任务等不同领域有广泛应用，并且在测试时适应和可信AI中发挥重要作用。作者总结了当前基准，并确定了关键挑战和未来方向。

### 结论

据作者所知，这是第一篇专门针对PA的方法和应用的综合调查，旨在为研究人员和实践者提供清晰的路线图，以理解和探索PA相关研究的不断发展的格局。

### 翻译

在计算机视觉中，视觉提示(Visual Prompting, VP)和视觉提示调优(Visual Prompt Tuning, VPT)最近已经出现作为轻量级且有效的替代方法，用于在'预训练后微调'范式中适应大规模视觉模型。然而，尽管进展迅速，它们的概念边界仍然模糊，因为VP和VPT在当前研究中经常被互换使用，反映了这些技术及其各自应用之间缺乏系统区分。在本调查中，我们从基本原理重新审视VP和VPT的设计，并将它们概念化为一个称为基于提示的适应(Prompt-based Adaptation, PA)的统一框架。我们提供了一个分类法，将现有方法分为可学习提示、生成提示和非可学习提示，并按注入粒度（像素级和令牌级）进一步组织。除了核心方法外，我们检查了PA在医学成像、3D点云和视觉语言任务等不同领域的整合，以及其在测试时适应和可信AI中的作用。我们还总结了当前基准，并确定了关键挑战和未来方向。据我们所知，我们是第一个专门针对PA的方法和应用的全面调查，考虑其独特特征。我们的调查旨在为所有领域的研究人员和实践者提供清晰的路线图，以理解和探索PA相关研究的不断发展的格局。


### 论文摘要

In computer vision, Visual Prompting (VP) and Visual Prompt Tuning (VPT) have recently emerged as lightweight and effective alternatives to full fine-tuning for adapting large-scale vision models within the ``pretrain-then-finetune'' paradigm. However, despite rapid progress, their conceptual boundaries remain blurred, as VP and VPT are frequently used interchangeably in current research, reflecting a lack of systematic distinction between these techniques and their respective applications. In this survey, we revisit the designs of VP and VPT from first principles, and conceptualize them within a unified framework termed Prompt-based Adaptation (PA). We provide a taxonomy that categorizes existing methods into learnable, generative, and non-learnable prompts, and further organizes them by injection granularity -- pixel-level and token-level. Beyond the core methodologies, we examine PA's integrations across diverse domains, including medical imaging, 3D point clouds, and vision-language tasks, as well as its role in test-time adaptation and trustworthy AI. We also summarize current benchmarks and identify key challenges and future directions. To the best of our knowledge, we are the first comprehensive survey dedicated to PA's methodologies and applications in light of their distinct characteristics. Our survey aims to provide a clear roadmap for researchers and practitioners in all area to understand and explore the evolving landscape of PA-related research.

---

## 6. ADPerf: Investigating and Testing Performance in Autonomous Driving Systems

**论文链接:** [http://arxiv.org/abs/2510.13078v1](http://arxiv.org/abs/2510.13078v1)

**作者:** Tri Minh-Triet Pham, Diego Elias Costa, Weiyi Shang, Jinqiu Yang

**发布时间:** 2025-10-15

**备注:** 13 pages, accepted by ASE 2025

### GPT解析

### 总结

该论文研究了自动驾驶系统中障碍物检测模块的性能和延迟问题，开发了一个名为ADPerf的工具用于测试和暴露检测延迟，并评估了其对系统整体可靠性的影响。

### 背景

障碍物检测对自动驾驶系统运行至关重要，系统依赖多种传感器结合深度学习模型进行时间敏感决策。然而，障碍物检测模块的延迟及其对LiDAR点云数据变化的适应性尚未被充分了解。

### 目的

首次全面测量和建模Apollo和Autoware两个行业级自动驾驶系统中障碍物检测模块的性能，开发ADPerf工具生成测试用例以暴露检测延迟增加，并评估其对后续模块的影响。

### 方法

对Apollo和Autoware系统中的障碍物检测模块进行性能测量和建模，开发ADPerf工具生成真实点云数据测试用例，对3D障碍物检测模块进行压力测试，并评估这些测试对轨迹预测模块的传播影响。

### 主要发现

障碍物检测组件（特别是3D障碍物检测）的性能测试非常必要，障碍物检测可能成为自动驾驶系统延迟增加的主要瓶颈，延迟增加的不利影响会传播到其他模块，降低系统整体可靠性。

### 结论

需要对障碍物检测组件进行性能测试，特别是3D障碍物检测，因为它们是自动驾驶系统延迟增加的主要瓶颈，会进一步影响其他模块，降低整体系统可靠性。

### 翻译

障碍物检测对自动驾驶系统的运行至关重要，这些系统依赖多种传感器（如摄像头和LiDAR）结合代码逻辑和深度学习模型来检测障碍物，以便进行时间敏感的决策。因此，障碍物检测延迟对自动驾驶系统的安全性和有效性至关重要。然而，障碍物检测模块的延迟及其对LiDAR点云数据各种变化的适应性尚未被充分了解。在这项工作中，我们首次对两个行业级自动驾驶系统（即Apollo和Autoware）中的障碍物检测模块性能进行了全面的测量和建模研究。从这项研究中，我们引入了ADPerf，这是一个旨在生成真实点云数据测试用例的工具，这些测试用例可以暴露检测延迟的增加。延迟降低会减少检测到障碍物的可用性，并对自动驾驶系统中后续模块的能力造成压力，即这些模块可能受到障碍物检测延迟增加的负面影响。我们将ADPerf应用于压力测试自动驾驶系统中广泛使用的3D障碍物检测模块的性能，以及此类测试对轨迹预测模块的传播影响。我们的评估强调了需要对障碍物检测组件进行性能测试，特别是3D障碍物检测，因为它们可能成为自动驾驶系统延迟增加的主要瓶颈。这种不利结果还会进一步传播到其他模块，降低自动驾驶系统的整体可靠性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶系统中障碍物检测模块的性能问题，特别是延迟(latency)问题。这个问题在现实中非常重要，因为障碍物检测延迟直接影响自动驾驶系统的安全性和有效性；延迟过大会导致系统无法及时做出决策，就像未检测到障碍物一样危险。同时，现有研究大多关注检测器的准确性和鲁棒性，而对其性能的研究相对不足，这导致自动驾驶系统在实际部署中可能存在未被发现性能瓶颈的风险。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了自动驾驶系统架构，识别出感知模块(特别是3D障碍物检测)是性能瓶颈。他们通过排队网络和排队Petri网对Apollo和Autoware系统进行性能建模，确认了3D障碍物检测的延迟问题。基于这些发现，他们设计了ADPerf工具，通过三种简单方法修改点云数据来增加检测延迟：添加障碍物边界外的噪声、添加新障碍物、移动现有障碍物。作者借鉴了现有的性能测试技术和障碍物检测鲁棒性测试方法，但专注于性能而非准确性，与现有的对抗攻击方法(如SlowLidar)相比，ADPerf采用更简单、更现实的修改方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过生成能增加3D障碍物检测延迟的测试场景，评估自动驾驶系统在性能压力下的行为及其对后续模块的影响。整体实现流程包括：1)数据准备，从真实世界驾驶场景数据集中提取点云表示和障碍物历史数据；2)测试场景生成，通过添加噪声、添加障碍物或移动障碍物来修改点云；3)模型执行与延迟测量，在修改和未修改的点云上运行检测模型并测量延迟；4)帧可用性估计，基于检测延迟估计哪些帧会被丢弃；5)轨迹预测评估，分析检测延迟对轨迹预测模块的级联影响。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次对自动驾驶系统障碍物检测模块性能进行综合研究；2)提出ADPerf工具，专门用于生成性能测试用例；3)研究性能问题的级联影响，关注检测延迟对整个系统的影响；4)采用更现实的测试方法。相比之前的工作，本文的关注点从准确性转向性能，方法上从复杂的对抗攻击转向简单的点云修改，评估范围从单个模块扩展到整个系统，且生成的测试场景更接近真实世界，具有更高的实用价值。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了ADPerf工具，通过生成增加障碍物检测延迟的测试场景，首次系统性地研究了自动驾驶系统中障碍物检测模块的性能瓶颈及其对整体系统可靠性的影响。'}


### 论文摘要

Obstacle detection is crucial to the operation of autonomous driving systems, which rely on multiple sensors, such as cameras and LiDARs, combined with code logic and deep learning models to detect obstacles for time-sensitive decisions. Consequently, obstacle detection latency is critical to the safety and effectiveness of autonomous driving systems. However, the latency of the obstacle detection module and its resilience to various changes in the LiDAR point cloud data are not yet fully understood. In this work, we present the first comprehensive investigation on measuring and modeling the performance of the obstacle detection modules in two industry-grade autonomous driving systems, i.e., Apollo and Autoware. Learning from this investigation, we introduce ADPerf, a tool that aims to generate realistic point cloud data test cases that can expose increased detection latency. Increasing latency decreases the availability of the detected obstacles and stresses the capabilities of subsequent modules in autonomous driving systems, i.e., the modules may be negatively impacted by the increased latency in obstacle detection.   We applied ADPerf to stress-test the performance of widely used 3D obstacle detection modules in autonomous driving systems, as well as the propagation of such tests on trajectory prediction modules. Our evaluation highlights the need to conduct performance testing of obstacle detection components, especially 3D obstacle detection, as they can be a major bottleneck to increased latency of the autonomous driving system. Such an adverse outcome will also further propagate to other modules, reducing the overall reliability of autonomous driving systems.

---

## 7. UrbanFusion: Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations

**论文链接:** [http://arxiv.org/abs/2510.13774v1](http://arxiv.org/abs/2510.13774v1)

**作者:** Dominik J. Mühlematter, Lin Che, Ye Hong, Martin Raubal, Nina Wiedemann

**发布时间:** 2025-10-15

### GPT解析

### 总结

UrbanFusion是一个地理基础模型(GeoFM)，采用随机多模态融合(SMF)技术，能够有效整合多种地理空间数据，在预测城市现象方面表现优异。

### 背景

预测城市现象如房价和公共健康指标需要有效整合各种地理空间数据。当前方法主要使用任务特定模型，而最近的用于空间表示的基础模型通常只支持有限模态，缺乏多模态融合能力。

### 目的

为了克服现有方法的局限性，开发一个能够处理多种地理数据模态并具有强大泛化能力的地理基础模型。

### 方法

UrbanFusion采用模态特定编码器处理街景图像、遥感数据、地图和兴趣点(POIs)数据，并通过基于Transformer的融合模块整合这些多模态输入，学习统一的表示。

### 主要发现

在全球56个城市41个任务的评估中，UrbanFusion表现出强大的泛化能力和预测性能：1)在位置编码方面优于之前的基础模型；2)在推理过程中允许多模态输入；3)对训练中未见过的区域泛化良好。

### 结论

UrbanFusion可在预训练和推理过程中灵活利用任何可用模态的子集，使模型在不同数据可用性场景下具有广泛的适用性，所有源代码已在GitHub开源。

### 翻译

预测城市现象如房价和公共健康指标需要有效整合各种地理空间数据。当前方法主要使用任务特定的模型，而最近的用于空间表示的基础模型通常只支持有限的模态，且缺乏多模态融合能力。为了克服这些挑战，我们提出了UrbanFusion，这是一个具有随机多模态融合(SMF)的地理基础模型(GeoFM)。该框架采用模态特定编码器处理不同类型的输入，包括街景图像、遥感数据、地图和兴趣点(POIs)数据。这些多模态输入通过基于Transformer的融合模块进行整合，学习统一的表示。在全球56个城市41个任务的广泛评估中，UrbanFusion与最先进的GeoAI模型相比表现出强大的泛化能力和预测性能。具体来说，它1)在位置编码方面优于之前的基础模型；2)在推理过程中允许多模态输入；3)对训练中未见过的区域泛化良好。UrbanFusion可以在预训练和推理过程中灵活利用任何可用模态的子集，使模型在不同数据可用性场景下具有广泛的适用性。所有源代码均可通过https://github.com/DominikM198/UrbanFusion获取。


### 论文摘要

Forecasting urban phenomena such as housing prices and public health indicators requires the effective integration of various geospatial data. Current methods primarily utilize task-specific models, while recent foundation models for spatial representations often support only limited modalities and lack multimodal fusion capabilities. To overcome these challenges, we present UrbanFusion, a Geo-Foundation Model (GeoFM) that features Stochastic Multimodal Fusion (SMF). The framework employs modality-specific encoders to process different types of inputs, including street view imagery, remote sensing data, cartographic maps, and points of interest (POIs) data. These multimodal inputs are integrated via a Transformer-based fusion module that learns unified representations. An extensive evaluation across 41 tasks in 56 cities worldwide demonstrates UrbanFusion's strong generalization and predictive performance compared to state-of-the-art GeoAI models. Specifically, it 1) outperforms prior foundation models on location-encoding, 2) allows multimodal input during inference, and 3) generalizes well to regions unseen during training. UrbanFusion can flexibly utilize any subset of available modalities for a given location during both pretraining and inference, enabling broad applicability across diverse data availability scenarios. All source code is available at https://github.com/DominikM198/UrbanFusion.

---

## 8. Scaling Vision Transformers for Functional MRI with Flat Maps

**论文链接:** [http://arxiv.org/abs/2510.13768v1](http://arxiv.org/abs/2510.13768v1)

**作者:** Connor Lane, Daniel Z. Kaplan, Tanishq Mathew Abraham, Paul S. Scotti

**发布时间:** 2025-10-15

**备注:** NeurIPS 2025 Workshop, Foundation Models for the Brain and Body;  Code: https://github.com/MedARC-AI/fmri-fm; Discord:  https://discord.gg/tVR4TWnRM9

### GPT解析

### 总结

本研究探索了将现代深度学习架构适应功能磁共振成像(fMRI)的方法，通过将4D体积fMRI数据转换为2D平面图视频，并使用时空掩码自编码器框架训练视觉Transformer模型。

### 背景

现代深度学习架构如何适应功能磁共振成像(fMRI)是一个关键问题，需要解决fMRI与自然图像之间的模态差距。

### 目的

研究如何将fMRI数据表示为模型输入，构建fMRI数据的基础模型。

### 方法

将4D体积fMRI数据转换为2D fMRI活动平面图视频，使用时空掩码自编码器框架在人类连接体项目的2.3K小时fMRI平面图视频上训练视觉Transformer，并进行掩码建模和下游分类基准测试。

### 主要发现

掩码fMRI建模性能随数据集大小严格遵循幂律缩放规律而提高；模型能够学习丰富的表示，支持跨受试者的精细状态解码和跨脑状态变化的受试者特异性特征解码。

### 结论

这是构建fMRI数据基础模型的开放科学项目的一部分，代码和数据已公开共享。

### 翻译

将现代深度学习架构适应功能磁共振成像(fMRI)的一个关键问题是如何为模型输入表示数据。为弥合fMRI与自然图像之间的模态差距，我们将4D体积fMRI数据转换为2D fMRI活动平面图视频。我们在人类连接体项目的2.3K小时fMRI平面图视频上使用时空掩码自编码器框架训练视觉Transformer。我们观察到，根据严格的幂律缩放规律，掩码fMRI建模性能随数据集大小增加而提高。下游分类基准测试表明，我们的模型学习了丰富的表示，既支持跨受试者的精细状态解码，也支持跨脑状态变化的受试者特异性特征解码。这项工作是构建fMRI数据基础模型的开放科学项目的一部分。我们的代码和数据可在https://github.com/MedARC-AI/fmri-fm获取。


### 论文摘要

A key question for adapting modern deep learning architectures to functional MRI (fMRI) is how to represent the data for model input. To bridge the modality gap between fMRI and natural images, we transform the 4D volumetric fMRI data into videos of 2D fMRI activity flat maps. We train Vision Transformers on 2.3K hours of fMRI flat map videos from the Human Connectome Project using the spatiotemporal masked autoencoder (MAE) framework. We observe that masked fMRI modeling performance improves with dataset size according to a strict power scaling law. Downstream classification benchmarks show that our model learns rich representations supporting both fine-grained state decoding across subjects, as well as subject-specific trait decoding across changes in brain state. This work is part of an ongoing open science project to build foundation models for fMRI data. Our code and datasets are available at https://github.com/MedARC-AI/fmri-fm.

---

## 9. NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching

**论文链接:** [http://arxiv.org/abs/2510.13721v1](http://arxiv.org/abs/2510.13721v1)

**作者:** Run Luo, Xiaobo Xia, Lu Wang, Longze Chen, Renke Shan, Jing Luo, Min Yang, Tat-Seng Chua

**发布时间:** 2025-10-15

### GPT解析

### 总结

NExT-OMNI是一个开源的全模态基础模型，通过离散流范式实现统一建模，支持任意到任意的跨模态理解与生成，在多模态生成、理解、多轮交互和跨模态检索方面表现优异。

### 背景

下一代多模态基础模型将成为人工通用智能系统的核心组件，但现有多模态模型受限于自回归架构，难以平衡理解与生成能力；混合和解耦策略虽有探索，但冗余设计限制了在广泛场景中的应用。

### 目的

开发一个支持任何到任何跨模态生成和多轮交互的多模态基础模型，克服现有模型的局限性，实现更高效、更广泛的应用。

### 方法

引入NExT-OMNI模型，利用度量诱导的概率路径和动力学最优速度，通过离散流范式实现统一建模，并在大规模交错文本、图像、视频和音频数据上进行训练。

### 主要发现

NExT-OMNI在多模态生成和理解基准测试中具有竞争力，在多轮多模态交互和跨模态检索方面优于之前的统一模型，突显了其作为下一代多模态基础模型的架构优势。

### 结论

NExT-OMNI通过简洁的统一表示而非任务解耦设计，实现了更广泛的应用场景，为促进进一步研究，已公开训练细节、数据协议，并开源了代码和模型检查点。

### 翻译

能够进行任意到任意跨模态生成和多轮交互的下一代多模态基础模型将成为人工通用智能系统的核心组件，在人机交互中发挥关键作用。然而，大多数现有的多模态模型仍受限于自回归架构，其固有局限性阻碍了理解与生成能力的平衡整合。尽管已经探索了混合和解耦策略来在统一框架内分别解决这些问题，但这些冗余、非集成的设计限制了它们在更广泛场景（如跨模态检索）中的适用性。在本工作中，我们引入了NExT-OMNI，一个开源的全模态基础模型，通过离散流范式实现统一建模。通过利用度量诱导的概率路径和动力学最优速度，NExT-OMNI原生支持任意到任意的理解和生成，同时通过简洁的统一表示而非任务解耦设计，实现更广泛的应用场景，并提高响应效率。在大型交错文本、图像、视频和音频数据上训练后，NExT-OMNI在多模态生成和理解基准测试中具有竞争力，同时在多轮多模态交互和跨模态检索方面优于之前的统一模型，突显了其作为下一代多模态基础模型的架构优势。为了促进进一步研究，我们发布了训练细节、数据协议，并开源了代码和模型检查点。


### 论文摘要

Next-generation multimodal foundation models capable of any-to-any cross-modal generation and multi-turn interaction will serve as core components of artificial general intelligence systems, playing a pivotal role in human-machine interaction. However, most existing multimodal models remain constrained by autoregressive architectures, whose inherent limitations prevent a balanced integration of understanding and generation capabilities. Although hybrid and decoupling strategies have been explored to address these tasks within unified frameworks separately, their redundant, non-integrated designs limit their applicability to broader scenarios, such as cross-modal retrieval.In this work, we introduce NExT-OMNI, an open-source omnimodal foundation model that achieves unified modeling through discrete flow paradigms. By leveraging metric-induced probability paths and kinetic optimal velocities, NExT-OMNI natively supports any-to-any understanding and generation with enhanced response efficiency, while enabling broader application scenarios through concise unified representations rather than task-decoupled designs. Trained on large-scale interleaved text, image, video, and audio data, NExT-OMNI delivers competitive performance on multimodal generation and understanding benchmarks, while outperforming prior unified models in multi-turn multimodal interaction and cross-modal retrieval, highlighting its architectural advantages as a next-generation multimodal foundation model. To advance further research, we release training details, data protocols, and open-source both the code and model checkpoints.

---

## 10. Axial Neural Networks for Dimension-Free Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.13665v1](http://arxiv.org/abs/2510.13665v1)

**作者:** Hyunsu Kim, Jonggeon Park, Joan Bruna, Hongseok Yang, Juho Lee

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种轴向神经网络(XNN)架构，解决了在物理数据上训练基础模型时面临的维度变化挑战，使模型能够有效处理不同维度的偏微分方程数据，同时保持计算效率和性能。

### 背景

基础模型在AI中的出现显著推进了通用学习，在零样本推理和上下文学习方面表现出色。然而，在物理数据（包括偏微分方程PDEs的解）上训练此类模型面临独特挑战，因为不同系统的维度各不相同。

### 目的

提出一种维度不可知的神经网络架构，解决传统方法在处理不同维度数据时效率低下的问题。

### 方法

提出轴向神经网络(XNN)，受Deep Sets和图神经网络等参数共享结构的启发。将现有的PDE基础模型转换为轴向神经网络，并在三种训练场景下评估性能：从头开始训练、在多个PDE上预训练以及在单个PDE上微调。

### 主要发现

实验表明，XNNs与原始模型表现相当，并且对未见维度表现出更好的泛化能力，突显了多维预训练对基础模型的重要性。

### 结论

XNN架构解决了在物理数据上训练基础模型的维度挑战，同时保持了性能和计算效率。

### 翻译

基础模型在AI中的出现显著推进了通用学习，使零样本推理和上下文学习能力显著提升。然而，在包括偏微分方程(PDEs)解在内的物理数据上训练此类模型，由于不同系统间维度的变化，带来了独特挑战。传统方法要么固定最大维度，要么为不同维度使用单独的编码器，导致效率低下。为此，我们提出了一种维度不可知的神经网络架构——轴向神经网络(XNN)，其灵感来自Deep Sets和图神经网络等参数共享结构。XNN能够在保持计算效率的同时，推广到变化的张量维度。我们将现有的PDE基础模型转换为轴向神经网络，并在三种训练场景下评估其性能：从头开始训练、在多个PDE上预训练以及在单个PDE上微调。实验表明，XNNs与原始模型表现相当，并且对未见维度表现出更好的泛化能力，突显了多维预训练对基础模型的重要性。


### 论文摘要

The advent of foundation models in AI has significantly advanced general-purpose learning, enabling remarkable capabilities in zero-shot inference and in-context learning. However, training such models on physics data, including solutions to partial differential equations (PDEs), poses a unique challenge due to varying dimensionalities across different systems. Traditional approaches either fix a maximum dimension or employ separate encoders for different dimensionalities, resulting in inefficiencies. To address this, we propose a dimension-agnostic neural network architecture, the Axial Neural Network (XNN), inspired by parameter-sharing structures such as Deep Sets and Graph Neural Networks. XNN generalizes across varying tensor dimensions while maintaining computational efficiency. We convert existing PDE foundation models into axial neural networks and evaluate their performance across three training scenarios: training from scratch, pretraining on multiple PDEs, and fine-tuning on a single PDE. Our experiments show that XNNs perform competitively with original models and exhibit superior generalization to unseen dimensions, highlighting the importance of multidimensional pretraining for foundation models.

---

## 11. Time Series Foundation Models: Benchmarking Challenges and Requirements

**论文链接:** [http://arxiv.org/abs/2510.13654v1](http://arxiv.org/abs/2510.13654v1)

**作者:** Marcel Meyer, Sascha Kaltenpoth, Kevin Zalipski, Oliver Müller

**发布时间:** 2025-10-15

### GPT解析

### 总结

时间序列基础模型(TSFMs)是一种新的时间序列预测范式，具有零样本预测能力，但其评估面临多个挑战，包括数据集代表性问题、缺乏时空评估、信息泄露风险和全局模式记忆问题。

### 背景

时间序列基础模型(TSFMs)代表了一种新的时间序列预测范式，提供无需领域特定预训练或微调的零样本预测能力。与大型语言模型(LLMs)类似，随着训练集的不断扩大，确保基准测试数据的完整性变得越来越困难。

### 目的

调查现有TSFM评估的挑战，并提出改进评估方法的建议，以保障TSFM评估的完整性。

### 方法

通过调查现有TSFM评估实践，分析数据分区问题，并提出新的评估方法建议，如在真正外来的未来数据上进行评估。

### 主要发现

现有TSFM评估存在多个挑战，包括基准数据集的代表性问题、缺乏时空评估、信息泄露风险和全局模式记忆问题。此外，关于数据分区的普遍混乱可能导致性能估计膨胀和全球知识错误地转移到局部时间序列。

### 结论

需要开发强大的评估方法来防止在LLM和经典时间序列基准测试中已经观察到的陷阱，并呼吁研究社区设计新的、有原则的评估方法，如在真正外来的未来数据上进行评估，以保障TSFM评估的完整性。

### 翻译

时间序列基础模型(TSFMs)代表了一种新的时间序列预测范式，提供无需领域特定预训练或微调的零样本预测能力。然而，与大型语言模型(LLMs)一样，评估TSFMs很棘手，因为随着训练集的不断扩展，确保基准测试数据的完整性变得越来越具有挑战性。我们对现有TSFM评估的调查揭示了多个挑战，从基准数据集的代表性、缺乏时空评估，到由于数据集重叠和不透明导致的信息泄露风险，以及由经济危机或疫情等外部冲击引起的全局模式记忆问题。我们的发现揭示了关于数据分区的普遍混乱，这可能导致性能估计膨胀和全球知识错误地转移到局部时间序列。我们呼吁开发强大的评估方法，以防止在LLM和经典时间序列基准测试中已经观察到的陷阱，并呼吁研究社区设计新的、有原则的方法，如在真正外来的未来数据上进行评估，以保障TSFM评估的完整性。


### 论文摘要

Time Series Foundation Models (TSFMs) represent a new paradigm for time series forecasting, offering zero-shot forecasting capabilities without the need for domain-specific pre-training or fine-tuning. However, as with Large Language Models (LLMs), evaluating TSFMs is tricky, as with ever more extensive training sets, it becomes more and more challenging to ensure the integrity of benchmarking data. Our investigation of existing TSFM evaluation highlights multiple challenges, ranging from the representativeness of the benchmark datasets, over the lack of spatiotemporal evaluation, to risks of information leakage due to overlapping and obscure datasets, and the memorization of global patterns caused by external shocks like economic crises or pandemics. Our findings reveal widespread confusion regarding data partitions, risking inflated performance estimates and incorrect transfer of global knowledge to local time series. We argue for the development of robust evaluation methodologies to prevent pitfalls already observed in LLM and classical time series benchmarking, and call upon the research community to design new, principled approaches, such as evaluations on truly out-of-sample future data, to safeguard the integrity of TSFM assessment.

---

## 12. Towards Adversarial Robustness and Uncertainty Quantification in DINOv2-based Few-Shot Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2510.13643v1](http://arxiv.org/abs/2510.13643v1)

**作者:** Akib Mohammed Khan, Bartosz Krawczyk

**发布时间:** 2025-10-15

**备注:** 10 pages, 5 figures, 3 tables

### GPT解析

### 总结

这篇论文研究了基于DINOv2等基础模型的小样本异常检测器的对抗性扰动敏感性和不确定性校准问题。作者通过在冻结的DINOv2特征上附加轻量级线性头创建对抗性攻击，评估了FGSM攻击的影响，并发现微小扰动可显著降低检测性能。同时，原始异常分数校准性较差，通过应用Platt缩放方法，作者提出了实用的攻击检测机制并降低了校准误差。

### 背景

基础模型如DINOv2在小样本异常检测中表现出强大性能，但两个关键问题尚未得到研究：(1)这些检测器对抗性扰动的敏感性如何；(2)它们的异常分数在多大程度上反映了校准的不确定性。

### 目的

研究DINOv2等基础模型在小样本异常检测中的对抗性鲁棒性和不确定性校准问题，并提出实用的攻击检测机制，以提高异常检测系统的可信度和安全性。

### 方法

基于AnomalyDINO（一种在DINOv2特征上的训练深度最近邻检测器），作者在冻结的DINOv2特征上附加轻量级线性头仅用于创建对抗性扰动。评估FGSM攻击在MVTec-AD和VisA数据集上的影响，并应用后验Platt缩放方法对异常分数进行不确定性估计。

### 主要发现

1) 微小对抗性扰动显著降低检测性能(F1、AUROC、AP和G-mean指标均下降)；2) 扰动可在特征空间翻转最近邻关系，导致有把握的错误分类；3) 原始异常分数校准性较差，置信度与正确性存在差距；4) Platt缩放得到的校验后验分布在对抗性扰动输入上产生更高预测熵；5) 该方法可用于实用攻击检测机制，同时降低校准误差(ECE)。

### 结论

DINOv2基础的小样本异常检测器存在具体脆弱性，对抗性鲁棒性和有原则的不确定性量化不是可选的附加功能，而是异常检测系统可信度和为真实世界部署做好准备所必需的基本能力。

### 翻译

基础模型如DINOv2在小样本异常检测中表现出强大的性能，但两个关键问题尚未得到研究：(i)这些检测器对抗性扰动的敏感性如何；(ii)它们的异常分数在多大程度上反映了校准的不确定性。基于AnomalyDINO（一种在DINOv2特征上的训练深度最近邻检测器），我们进行了此设置中对抗性攻击和不确定性估计的首次系统性研究之一。为了在保持测试时间行为的同时实现白盒梯度攻击，我们仅在创建扰动时为冻结的DINOv2特征附加了一个轻量级线性头。使用这种启发式方法，我们评估了FGSM在MVTec-AD和VisA数据集上的影响，并观察到F1、AUROC、AP和G-mean指标的一致下降，表明微小的扰动可以在特征空间中翻转最近邻关系，导致有把握的错误分类。除了鲁棒性外，我们还探测了可靠性，发现原始异常分数的校准性较差，揭示了置信度与正确性之间的差距，这限制了安全关键应用。作为迈向可信度的简单、强基线，我们对异常分数应用了后验Platt缩放进行不确定性估计。所得的校验后验分布在对抗性扰动输入上产生显著更高的预测熵，能够用于实用的攻击检测机制，同时降低校准误差（ECE）。我们的研究结果揭示了DINOv2基础小样本异常检测器的具体脆弱性，并为鲁棒、不确定性感知的异常检测建立了评估协议和基线。我们认为，对抗性鲁棒性和有原则的不确定性量化不是可选的附加功能，而是异常检测系统可信度和为真实世界部署做好准备所必需的基本能力。


### 论文摘要

Foundation models such as DINOv2 have shown strong performance in few-shot anomaly detection, yet two key questions remain unexamined: (i) how susceptible are these detectors to adversarial perturbations; and (ii) how well do their anomaly scores reflect calibrated uncertainty? Building on AnomalyDINO, a training-free deep nearest-neighbor detector over DINOv2 features, we present one of the first systematic studies of adversarial attacks and uncertainty estimation in this setting. To enable white-box gradient attacks while preserving test-time behavior, we attach a lightweight linear head to frozen DINOv2 features only for crafting perturbations. Using this heuristic, we evaluate the impact of FGSM across the MVTec-AD and VisA datasets and observe consistent drops in F1, AUROC, AP, and G-mean, indicating that imperceptible perturbations can flip nearest-neighbor relations in feature space to induce confident misclassification. Complementing robustness, we probe reliability and find that raw anomaly scores are poorly calibrated, revealing a gap between confidence and correctness that limits safety-critical use. As a simple, strong baseline toward trustworthiness, we apply post-hoc Platt scaling to the anomaly scores for uncertainty estimation. The resulting calibrated posteriors yield significantly higher predictive entropy on adversarially perturbed inputs than on clean ones, enabling a practical flagging mechanism for attack detection while reducing calibration error (ECE). Our findings surface concrete vulnerabilities in DINOv2-based few-shot anomaly detectors and establish an evaluation protocol and baseline for robust, uncertainty-aware anomaly detection. We argue that adversarial robustness and principled uncertainty quantification are not optional add-ons but essential capabilities if anomaly detection systems are to be trustworthy and ready for real-world deployment.

---

## 13. The Role of Computing Resources in Publishing Foundation Model Research

**论文链接:** [http://arxiv.org/abs/2510.13621v1](http://arxiv.org/abs/2510.13621v1)

**作者:** Yuexing Hao, Yue Huang, Haoran Zhang, Chenyang Zhao, Zhenwen Liang, Paul Pu Liang, Yue Zhao, Lichao Sun, Saleh Kalantari, Xiangliang Zhang, Marzyeh Ghassemi

**发布时间:** 2025-10-15

### GPT解析

### 总结

本研究评估了计算资源与基础模型科学进展之间的关系，发现增加计算资源与国家资金分配和引用量相关，但与研究环境、领域或研究方法无强相关性。

### 背景

前沿的人工智能研究需要大量资源，包括图形处理器(GPU)、数据和人力资源。

### 目的

评估这些资源与基础模型科学进展之间的关系。

### 方法

回顾了2022年至2024年间发表的6517篇基础模型论文，并调查了229位第一作者关于计算资源对科研产出影响的情况。

### 主要发现

增加的计算资源与国家资金分配和引用量相关，但未发现与研究环境(学术界或工业界)、领域或研究方法有强相关性。

### 结论

建议个人和机构专注于创建共享且负担得起的计算机会，以减少资源不足研究者的入门障碍，这些步骤可以帮助扩大基础模型研究的参与度，促进思想贡献者的多样性，并维持人工智能的创新和进步。

### 翻译

尖端的人工智能研究需要大量资源，包括图形处理器、数据和人力资源。在本文中，我们评估了这些资源与基础模型科学进展之间的关系。我们回顾了2022年至2024年间发表的6517篇基础模型论文，并对229位第一作者进行了调查，了解计算资源对科研产出的影响。我们发现，增加的计算资源与国家资金分配和引用量相关，但我们的研究结果未观察到与研究环境(学术界或工业界)、领域或研究方法有强相关性。我们建议个人和机构专注于创建共享且负担得起的计算机会，以降低资源不足研究者的入门门槛。这些步骤可以帮助扩大基础模型研究的参与度，促进思想贡献者的多样性，并维持人工智能的创新和进步。数据将在https://mit-calc.csail.mit.edu/提供。


### 论文摘要

Cutting-edge research in Artificial Intelligence (AI) requires considerable resources, including Graphics Processing Units (GPUs), data, and human resources. In this paper, we evaluate of the relationship between these resources and the scientific advancement of foundation models (FM). We reviewed 6517 FM papers published between 2022 to 2024, and surveyed 229 first-authors to the impact of computing resources on scientific output. We find that increased computing is correlated with national funding allocations and citations, but our findings don't observe the strong correlations with research environment (academic or industrial), domain, or study methodology. We advise that individuals and institutions focus on creating shared and affordable computing opportunities to lower the entry barrier for under-resourced researchers. These steps can help expand participation in FM research, foster diversity of ideas and contributors, and sustain innovation and progress in AI. The data will be available at: https://mit-calc.csail.mit.edu/

---

## 14. Generalizing WiFi Gesture Recognition via Large-Model-Aware Semantic Distillation and Alignment

**论文链接:** [http://arxiv.org/abs/2510.13390v1](http://arxiv.org/abs/2510.13390v1)

**作者:** Feng-Qi Cui, Yu-Tong Guo, Tianyue Zheng, Jinyang Huang

**发布时间:** 2025-10-15

**备注:** Accepted by IEEE ICPADS 2025

### GPT解析

### 总结

本文提出了一种名为GLSDA的新型泛化框架，利用预训练大型基础模型的语义先验来增强WiFi手势识别的泛化能力和语义表达能力，通过双路径CSI编码、多尺度语义编码、语义感知软监督和鲁棒双蒸馏策略，实现了在域内和跨域场景中的高性能手势识别。

### 背景

WiFi手势识别作为一种有前途的RF传感范式，可在AIoT环境中实现非接触式和隐私保护的人机交互。然而，现有方法因信道状态信息的域敏感特性和高级手势抽象的缺乏，面临泛化能力和语义表达能力有限的问题。

### 目的

解决现有WiFi手势识别方法泛化能力有限和语义表达能力不足的问题，提出一种能够增强域内和跨域场景中手势表示学习的新型框架。

### 方法

1) 设计双路径CSI编码管道，通过CSI-Ratio相位序列和多普勒频谱图捕获手势模式；2) 开发多尺度语义编码器，学习时序嵌入并通过跨模态注意力机制与手势语义对齐；3) 引入语义感知软监督方案，编码类间相关性并减少标签模糊性；4) 开发鲁棒双蒸馏策略，将对齐模型压缩为轻量级网络。

### 主要发现

在Widar3.0基准上的实验表明，GLSDA在域内和跨域手势识别任务中均优于现有最先进方法，同时显著减小了模型大小和推理延迟。

### 结论

GLSDA为现实世界AIoT应用中的通用RF手势界面提供了可扩展和可部署的解决方案。

### 翻译

基于WiFi的手势识别已成为一种有前途的RF传感范式，能够在AIoT环境中实现非接触式和隐私保护的人机交互。然而，由于信道状态信息的域敏感特性和高级手势抽象的缺乏，现有方法通常面临泛化能力和语义表达能力有限的问题。为解决这些挑战，我们提出了一种名为Large-Model-Aware Semantic Distillation and Alignment (GLSDA)的新型泛化框架，它利用预训练大型基础模型的语义先验来增强域内和跨域场景中的手势表示学习。具体而言，我们首先设计了一个双路径CSI编码管道，通过CSI-Ratio相位序列和多普勒频谱图捕获几何和动态手势模式。然后将这些表示输入多尺度语义编码器，学习鲁棒的时序嵌入，并通过跨模态注意力机制将其与手势语义对齐。为进一步增强类别区分度，我们引入了一种语义感知软监督方案，编码类间相关性并减少标签模糊性，特别是对于语义相似的手势。最后，我们开发了一种鲁棒双蒸馏策略，将对齐的模型压缩为轻量级学生网络，从教师模型联合蒸馏中间特征和语义感知软标签。在Widar3.0基准上的大量实验表明，GLSDA在域内和跨域手势识别任务中始终优于最先进的方法，同时显著减小了模型大小和推理延迟。我们的方法为现实世界AIoT应用中的通用RF手势界面提供了可扩展和可部署的解决方案。


### 论文摘要

WiFi-based gesture recognition has emerged as a promising RF sensing paradigm for enabling non-contact and privacy-preserving human-computer interaction in AIoT environments. However, existing methods often suffer from limited generalization and semantic expressiveness due to the domain-sensitive nature of Channel State Information and the lack of high-level gesture abstraction. To address these challenges, we propose a novel generalization framework, termed Large-Model-Aware Semantic Distillation and Alignment (GLSDA), which leverages the semantic prior of pre-trained large foundation models to enhance gesture representation learning in both in-domain and cross-domain scenarios. Specifically, we first design a dual-path CSI encoding pipeline that captures geometric and dynamic gesture patterns via CSI-Ratio phase sequences and Doppler spectrograms. These representations are then fed into a Multiscale Semantic Encoder, which learns robust temporal embeddings and aligns them with gesture semantics through cross-modal attention mechanisms. To further enhance category discrimination, we introduce a Semantic-Aware Soft Supervision scheme that encodes inter-class correlations and reduces label ambiguity, especially for semantically similar gestures. Finally, we develop a Robust Dual-Distillation strategy to compress the aligned model into a lightweight student network, jointly distilling intermediate features and semantic-informed soft labels from the teacher model. Extensive experiments on the Widar3.0 benchmark show that GLSDA consistently outperforms state-of-the-art methods in both in-domain and cross-domain gesture recognition tasks, while significantly reducing model size and inference latency. Our method offers a scalable and deployable solution for generalized RF-based gesture interfaces in real-world AIoT applications.

---

## 15. Document Intelligence in the Era of Large Language Models: A Survey

**论文链接:** [http://arxiv.org/abs/2510.13366v1](http://arxiv.org/abs/2510.13366v1)

**作者:** Weishi Wang, Hengchang Hu, Zhijie Zhang, Zhaochen Li, Hongxin Shao, Daniel Dahlmeier

**发布时间:** 2025-10-15

### GPT解析

### 总结

这篇论文综述了Document AI (DAI)领域在大语言模型(LLMs)影响下的发展，探讨了多模态、多语言和检索增强DAI的进展与挑战，并提出了未来研究方向。

### 背景

Document AI已成为重要应用领域，大语言模型的出现显著改变了这一领域，从早期的编码器-解码器架构发展为仅使用解码器的LLMs。

### 目的

提供DAI演变的全面概述，突出LLMs在该领域的当前研究和未来前景，为DAI的最先进技术提供结构化分析。

### 方法

通过综述形式，探索多模态、多语言和检索增强DAI的关键进展和挑战，并提出未来研究方向。

### 主要发现

解码器-only LLMs彻底改变了DAI，带来了理解和生成方面的显著进步；多模态、多语言和检索增强DAI面临关键进展和挑战；基于代理的方法和文档特定基础模型是有前景的未来研究方向。

### 结论

DAI在大语言模型的影响下正在快速发展，为学术和实践应用提供了新的可能性和挑战。

### 翻译

文档人工智能(DAI)已成为一个重要的应用领域，并因大型语言模型(LLMs)的出现而发生了显著变化。虽然早期方法依赖于编码器-解码器架构，但仅使用解码器的LLMs彻底改变了DAI，在理解和生成方面带来了显著进步。本综述提供了DAI演变的全面概述，突出了LLMs在该领域的当前研究和未来前景。我们探索了多模态、多语言和检索增强DAI的关键进展和挑战，同时提出了未来研究方向，包括基于代理的方法和文档特定基础模型。本文旨在为DAI的最先进技术提供结构化分析，及其对学术和实践应用的影响。


### 论文摘要

Document AI (DAI) has emerged as a vital application area, and is significantly transformed by the advent of large language models (LLMs). While earlier approaches relied on encoder-decoder architectures, decoder-only LLMs have revolutionized DAI, bringing remarkable advancements in understanding and generation. This survey provides a comprehensive overview of DAI's evolution, highlighting current research attempts and future prospects of LLMs in this field. We explore key advancements and challenges in multimodal, multilingual, and retrieval-augmented DAI, while also suggesting future research directions, including agent-based approaches and document-specific foundation models. This paper aims to provide a structured analysis of the state-of-the-art in DAI and its implications for both academic and practical applications.

---

## 16. Generative model for information metamaterial design

**论文链接:** [http://arxiv.org/abs/2510.13264v1](http://arxiv.org/abs/2510.13264v1)

**作者:** Jun Ming Hou, Long Chen, Xuan Zheng, Jia Wei Wu, Jian Wei You, Zi Xuan Cai, Jiahan Huang, Chen Xu Wu, Jian Lin Su, Lianlin Li, Jia Nan Zhang, Tie Jun Cui

**发布时间:** 2025-10-15

### GPT解析

### 总结

介绍了一种名为InfoMetaGen的通用生成模型，用于信息超材料设计，结合预训练基础模型和轻量级功能适配器，能够智能生成从元原子到任意空间编码模式的人工结构，相比传统方法具有更高的效率和泛化能力。

### 背景

生成式模型如AlphaFold和MatterGen可直接生成具有理想特性的新型材料结构，AlphaFold专注于蛋白质预测，MatterGen专注于预测周期性晶体结构，而超材料的通用设计更为复杂，需要设计元原子及其在空间中的任意非均匀分布。

### 目的

提出一个通用的生成式模型InfoMetaGen用于信息超材料设计，解决超材料设计中的复杂性问题，实现从元原子到任意空间编码模式的智能生成。

### 方法

InfoMetaGen结合预训练基础模型和轻量级功能适配器，通过微调轻量级适配器使单一通用生成模型能够切换不同功能，避免了传统方法需要为特定功能训练专用模型的局限。

### 主要发现

InfoMetaGen能够加速新型超材料的多样化发现，在超材料性能方面取得突破，填补了设计人工材料时通用生成框架的空白。

### 结论

该工作将生成模型的能力从微观自然材料的被动发现扩展到宏观人工材料的主动创造，为生成模型在材料设计领域开辟了前所未有的机会。

### 翻译

生成模型如AlphaFold和MatterGen可以直接生成具有理想特性的新型材料结构，加速新材料发现并将材料设计范式从传统的试错方法转变为智能按需生成。AlphaFold专注于具有特定非周期结构的蛋白质预测；而MatterGen专注于预测周期性和稳定的晶体结构。超材料的通用设计要复杂得多，因为它涉及设计元原子（类似于周期结构）及其在空间中的任意非均匀分布。在此，我们提出了InfoMetaGen，一种用于信息超材料设计的通用生成模型，它结合了预训练基础模型和轻量级功能适配器，智能生成从元原子到任意空间编码模式的人工结构。与需要为特定功能训练专用模型的传统智能超材料设计方法相比，InfoMetaGen使单个通用生成模型能够通过微调轻量级适配器切换不同功能，显著提高了效率和泛化能力。实验结果表明，InfoMetaGen不仅可以加速新型超材料的多样化发现，还能在超材料性能方面取得突破。这项工作填补了设计人工材料时通用生成框架的空白，并为将生成模型的能力从微观自然材料的被动发现扩展到宏观人工材料的主动创造开辟了前所未有的机会。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决信息超材料的智能设计问题。传统超材料设计依赖于试错法，周期长、效率低，且现有智能方法多局限于单一功能。超材料能实现自然材料中不存在的奇特物理特性，在无线通信、传感和超分辨率成像等领域有广泛应用，因此高效设计方法对推动这些领域发展至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了AlphaFold和MatterGen等生成模型的成功经验，认识到超材料设计比蛋白质或晶体设计更复杂，因为它涉及设计超原子和它们在空间中的任意非均匀分布。他们采用两阶段训练策略：首先预训练无条件扩散模型捕获不同设计空间中的功能编码模式，然后引入条件适配器模块微调模型引导生成过程。这种方法冻结基础模型参数，只微调轻量级适配器，使单个模型能处理多种功能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过一个预训练的基础模型结合轻量级功能适配器，智能生成从单个超原子到整个超阵列的结构配置，实现多功能的统一生成。流程包括：1)预训练阶段训练无条件扩散模型捕获不同功能的设计模式；2)微调阶段引入条件适配器，冻结基础模型参数只微调适配器；3)生成阶段根据特定功能条件将随机噪声转换为功能数字比特流；4)应用阶段实现超原子设计、波束成形、电磁聚焦和全息成像等多种功能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出InfoMetaGen通用生成框架填补设计空白；2)实现多尺度设计能力从超原子到超阵列；3)通过轻量级适配器实现高效多任务处理；4)创新比特表示方法解决离散编码与连续扩散模型兼容性；5)强大生成能力产生新颖超原子和1/3位非均匀超阵列。相比之前工作，它专注于宏观人工材料而非微观自然材料，使用单一通用模型替代专用模型，实现多功能的统一生成，并能突破训练数据集限制生成高性能宽带超原子。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'InfoMetaGen开创了信息超材料设计的通用生成范式，通过结合预训练基础模型与轻量级功能适配器，实现了从超原子到超阵列的多尺度、多功能智能生成，显著加速了新超材料的发现并突破了超材料性能的极限。'}


### 论文摘要

Generative models such as AlphaFold and MatterGen can directly generate novel material structures with desired properties, accelerating the new materials discovery and revolutionizing the material design paradigm from traditional trial-and-error approach to intelligent on-demand generation. AlphaFold is focused on protein prediction with specific aperiodic structures; while MatterGen is focused on predicting periodic and stable crystal structures. The universal design of metamaterials is much more complicated, since it involves to design meta-atoms (similar to the periodic structures) and their arbitrarily inhomogeneous distributions in space. Here, we propose InfoMetaGen, a universal generative model for information metamaterial design, which combines a pre-trained foundation model with lightweight functional adapters to intelligently generate artificial structures on-demand spanning from meta-atoms to arbitrary space coding patterns. In contrast to conventional intelligent metamaterial design methods that require training dedicated models for specific functionalities, InfoMetaGen enables a single universal generative model capable of switching across diverse functionalities by fine-tuning the lightweight adapters, significantly improving both efficiency and generalizability. Experimental results demonstrate that InfoMetaGen can not only accelerate the diverse discovery of new metamaterials, but also achieve breakthroughs in metamaterial performance. This work fills the gap of universal generative framework in designing artificial materials, and opens up unprecedented opportunities to expand the capability of generative models from the passive discovery of microscopic natural material to the active creation of macroscopic artificial materials.

---

## 17. NeuroRVQ: Multi-Scale EEG Tokenization for Generative Large Brainwave Models

**论文链接:** [http://arxiv.org/abs/2510.13068v1](http://arxiv.org/abs/2510.13068v1)

**作者:** Konstantinos Barmpas, Na Lee, Alexandros Koliousis, Yannis Panagakis, Dimitrios A. Adamos, Nikolaos Laskaris, Stefanos Zafeiriou

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种名为NeuroRVQ的新型脑电图基础模型，通过改进的信号令牌化方法解决了现有模型在高频动态处理和信号重建方面的局限性。

### 背景

脑电图(EEG)捕获了多个时间和频谱尺度的神经活动，产生的信号丰富但复杂，难以进行表示学习。现有的EEG基础模型在信号令牌化方面存在不足，无法保持高频动态，限制了信号重建的保真度。

### 目的

开发一种能够捕获完整频率神经频谱、支持高分辨率编码并实现高效训练的EEG信号令牌化器，以提高EEG信号的表示能力和重建质量。

### 方法

NeuroRVQ令牌化器包含三个关键组件：多尺度特征提取模块，捕获完整频率神经频谱；分层残差矢量量化(RVQ)码本，用于高分辨率编码；以及EEG信号相位和幅度感知损失函数，用于高效训练。

### 主要发现

NeuroRVQ设计支持所有频段的准确重建，同时实现高效的EEG压缩，实现了强大的生成掩码建模。实证结果表明，NeuroRVQ实现了更低的重建误差，并在各种下游任务上优于现有的大脑波模型。

### 结论

NeuroRVQ令牌化器为基于码本的通用脑波模型建立了强有力的先验，有望推动神经解码、生成建模和多模态生物信号集成等领域的发展。

### 翻译

脑电图(EEG)捕获了多个时间和频谱尺度的神经活动，产生的信号丰富但复杂，难以进行表示学习。最近，经过训练以预测掩码信号令牌的EEG基础模型在学习可泛化表示方面显示出前景。然而，它们的性能受到信号令牌化模块的限制。现有的神经令牌化器无法保持高频动态，限制了它们以高保真度重建EEG信号的能力。我们引入了NeuroRVQ，这是一个基于码本令牌化器的可扩展大脑波模型(LBM)。我们的令牌化器整合了：(i)捕获完整频率神经频谱的多尺度特征提取模块；(ii)用于高分辨率编码的分层残差矢量量化(RVQ)码本；以及(iii)用于高效训练的EEG信号相位和幅度感知损失函数。这种设计支持所有频段的准确重建，同时实现高效的EEG压缩，从而实现强大的生成掩码建模。我们的实证结果表明，NeuroRVQ实现了更低的重建误差，并在各种下游任务上优于现有的LBM。更广泛地说，NeuroRVQ令牌化器为基于码本的通用脑波模型建立了强有力的先验，推动了神经解码、生成建模和多模态生物信号集成的发展。


### 论文摘要

Electroencephalography (EEG) captures neural activity across multiple temporal and spectral scales, yielding signals that are rich but complex for representation learning. Recently, EEG foundation models trained to predict masked signal-tokens have shown promise for learning generalizable representations. However, their performance is hindered by their signal tokenization modules. Existing neural tokenizers fail to preserve high-frequency dynamics, limiting their ability to reconstruct EEG signals with high fidelity. We introduce NeuroRVQ, a scalable Large Brainwave Model (LBM) centered on a codebook-based tokenizer. Our tokenizer integrates: (i) multi-scale feature extraction modules that capture the full frequency neural spectrum; (ii) hierarchical residual vector quantization (RVQ) codebooks for high-resolution encoding; and, (iii) an EEG signal phase- and amplitude-aware loss function for efficient training. This design enables efficient EEG compression while supporting accurate reconstruction across all frequency bands, leading to robust generative masked modeling. Our empirical results demonstrate that NeuroRVQ achieves lower reconstruction error and outperforms existing LBMs on a variety of downstream tasks. More broadly, NeuroRVQ tokenizer establishes a strong prior for codebook-based general-purpose brainwave models, enabling advances in neural decoding, generative modeling and multimodal biosignal integration.

---

## 18. Epistemic-aware Vision-Language Foundation Model for Fetal Ultrasound Interpretation

**论文链接:** [http://arxiv.org/abs/2510.12953v1](http://arxiv.org/abs/2510.12953v1)

**作者:** Xiao He, Huangxuan Zhao, Guojia Wan, Wei Zhou, Yanxing Liu, Juhua Liu, Yongchao Xu, Yong Luo, Dacheng Tao, Bo Du

**发布时间:** 2025-10-14

### GPT解析

### 总结

该研究提出了FetalMind，一个专为胎儿超声检查设计的医学AI系统，用于报告生成和诊断。通过引入显著认识解耦（SED）方法和构建大规模数据集FetalSigma-1M，FetalMind在所有妊娠阶段都优于现有基线模型，平均提高14%的准确率，在关键条件下提高61.2%的准确率，同时保持高效、稳定和可扩展性。

### 背景

近期的医学视觉-语言模型在VQA、报告生成和异常检测等任务上表现出色，但大多数模型适应于结构化的成人影像，在胎儿超声检查方面表现不佳。胎儿超声面临多视图图像推理、疾病多样性和图像多样性的挑战。

### 目的

弥合现有医学视觉-语言模型在胎儿超声领域的应用差距，开发一个专门针对胎儿超声检查的AI系统，用于报告生成和诊断。

### 方法

在临床工作流程指导下提出显著认识解耦（SED）方法，将专家构建的二分图注入模型以解耦视图-疾病关联，并通过强化学习引导偏好选择。同时构建了FetalSigma-1M数据集，包含来自十二个医疗中心的20K份胎儿超声报告，解决了领域数据稀缺问题。

### 主要发现

FetalMind在所有妊娠阶段都优于开源和闭源基线模型，平均提高14%的准确率，在关键条件下提高61.2%的准确率，同时保持高效、稳定和可扩展性。

### 结论

FetalMind是一个有效的胎儿超声AI系统，通过SED方法和大规模数据集训练，成功解决了胎儿超声图像推理的挑战，在报告生成和诊断方面表现出色。

### 翻译

近期的医学视觉-语言模型在VQA、报告生成和异常检测等任务上显示出潜力。然而，大多数模型适应于结构化的成人影像，在胎儿超声检查方面表现不佳，这带来了多视图图像推理、疾病多样性和图像多样性的挑战。为了弥合这一差距，我们引入了FetalMind，一个专为胎儿超声设计的医学AI系统，用于报告生成和诊断。在临床工作流程的指导下，我们提出了显著认识解耦（SED）方法，将专家构建的二分图注入模型中，解耦视图-疾病关联，并通过强化学习引导偏好选择，遵循临床忠实步骤。这种设计减轻了疾病间的变异性和视图间的异质性，减少了学习瓶颈，同时使模型的推理与产科实践保持一致。为了大规模训练FetalMind，我们整理了FetalSigma-1M数据集，这是第一个大规模胎儿超声报告语料库，包含来自十二个医疗中心的20K份报告，解决了领域数据稀缺的问题。大量实验表明，FetalMind在所有妊娠阶段都优于开源和闭源基线模型，平均提高14%的准确率，在关键条件下提高61.2%的准确率，同时保持高效、稳定和可扩展性。项目页面：https://hexiao0275.github.io/FetalMind。


### 论文摘要

Recent medical vision-language models have shown promise on tasks such as VQA, report generation, and anomaly detection. However, most are adapted to structured adult imaging and underperform in fetal ultrasound, which poses challenges of multi-view image reasoning, numerous diseases, and image diversity. To bridge this gap, we introduce FetalMind, a medical AI system tailored to fetal ultrasound for both report generation and diagnosis. Guided by clinical workflow, we propose Salient Epistemic Disentanglement (SED), which injects an expert-curated bipartite graph into the model to decouple view-disease associations and to steer preference selection along clinically faithful steps via reinforcement learning. This design mitigates variability across diseases and heterogeneity across views, reducing learning bottlenecks while aligning the model's inference with obstetric practice. To train FetalMind at scale, we curate FetalSigma-1M dataset, the first large-scale fetal ultrasound report corpus, comprising 20K reports from twelve medical centers, addressing the scarcity of domain data. Extensive experiments show that FetalMind outperforms open- and closed-source baselines across all gestational stages, achieving +14% average gains and +61.2% higher accuracy on critical conditions while remaining efficient, stable, and scalable. Project Page: https://hexiao0275.github.io/FetalMind.

---

## 19. An Investigation of Memorization Risk in Healthcare Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.12950v1](http://arxiv.org/abs/2510.12950v1)

**作者:** Sana Tonekaboni, Lena Stempfle, Adibvafa Fallahpour, Walter Gerych, Marzyeh Ghassemi

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文介绍了一套用于评估在电子健康记录上训练的基础模型中隐私相关记忆风险的黑盒评估测试，包括在嵌入层和生成层探测记忆的方法，并发布了开源工具包促进医疗AI中的隐私评估。

### 背景

基础模型在大型去标识化的电子健康记录上训练有临床应用前景，但它们可能记住患者信息，引发隐私问题。

### 目的

引入一套黑盒评估测试来评估在结构化EHR数据上训练的基础模型中的隐私相关记忆风险。

### 方法

开发了一个框架，包括在嵌入层和生成层探测记忆的方法，旨在区分模型泛化和有害记忆，特别是在临床相关环境中。

### 主要发现

将记忆放在可能损害患者隐私的背景下，特别是对弱势亚群体。

### 结论

在公开可用的EHR基础模型上验证了这种方法，并发布了一个开源工具包，以促进医疗AI中可复现和协作的隐私评估。

### 翻译

在大型去标识化电子健康记录(EHRs)上训练的基础模型在临床应用方面具有潜力。然而，它们记忆患者信息的能力引发了重要的隐私问题。在这项工作中，我们引入了一套黑盒评估测试，用于评估在结构化EHR数据上训练的基础模型中的隐私相关记忆风险。我们的框架包括在嵌入层和生成层探测记忆的方法，旨在区分临床相关环境中的模型泛化和有害记忆。我们将记忆放在可能损害患者隐私的背景下，特别是对弱势亚群体。我们在公开可用的EHR基础模型上验证了我们的方法，并发布了一个开源工具包，以促进医疗AI中可复现和协作的隐私评估。


### 论文摘要

Foundation models trained on large-scale de-identified electronic health records (EHRs) hold promise for clinical applications. However, their capacity to memorize patient information raises important privacy concerns. In this work, we introduce a suite of black-box evaluation tests to assess privacy-related memorization risks in foundation models trained on structured EHR data. Our framework includes methods for probing memorization at both the embedding and generative levels, and aims to distinguish between model generalization and harmful memorization in clinically relevant settings. We contextualize memorization in terms of its potential to compromise patient privacy, particularly for vulnerable subgroups. We validate our approach on a publicly available EHR foundation model and release an open-source toolkit to facilitate reproducible and collaborative privacy assessments in healthcare AI.

---

## 20. SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Model

**论文链接:** [http://arxiv.org/abs/2510.12709v2](http://arxiv.org/abs/2510.12709v2)

**作者:** Lin Lin, Jiefeng Long, Zhihe Wan, Yuchi Wang, Dingkang Yang, Shuang Yang, Yueyang Yao, Xu Chen, Zirui Guo, Shengqiang Li, Weiran Li, Hanyu Li, Yaling Mou, Yan Qiu, Haiyang Yu, Xiao Liang, Hongsheng Li, Chao Feng

**发布时间:** 2025-10-14

**备注:** Technical Report

### GPT解析

### 总结

SAIL-Embedding是一个全模态嵌入基础模型，通过定制化训练策略和架构设计解决了现有多模态嵌入模型在现实应用中面临的挑战，包括模态支持有限、训练机制不稳定和工业领域差距等问题。

### 背景

多模态嵌入模型旨在产生信息丰富的统一表示以支持各种跨模态任务。尽管从基于CLIP的双塔架构到大型视觉语言模型的发展前景广阔，但现有工作在现实应用和业务场景中仍面临不可避免的挑战。

### 目的

介绍SAIL-Embedding，一个全模态嵌入基础模型，通过定制化训练策略和架构设计解决现有多模态嵌入模型面临的挑战。

### 方法

提出多阶段训练方案提高表示学习效果，包括内容感知渐进训练增强模型对多样化下游任务的适应性，协作感知推荐增强训练使多模态表示适应推荐场景，以及开发随机专业化和数据集驱动的模式匹配增强模型训练的灵活性和泛化能力。

### 主要发现

实验结果表明，SAIL-Embedding在不同检索任务中实现了SOTA性能。在线实验显示，Lifetime (LT)显著增加，在抖音精选场景中带来+0.5%的7天LT增益，为抖音feed排序模型带来+0.1%的AUC增益。

### 结论

SAIL-Embedding是一个有效的全模态嵌入基础模型，通过定制化训练策略和架构设计，在各种场景中表现出色，特别是在推荐系统中。

### 翻译

多模态嵌入模型旨在产生信息丰富的统一表示，支持各种跨模态任务。尽管从基于CLIP的双塔架构到大型视觉语言模型的发展前景广阔，但现有工作在现实应用和业务场景中仍面临不可避免的挑战，如模态支持有限、训练机制不稳定和工业领域差距等。在本工作中，我们介绍了SAIL-Embedding，一个全模态嵌入基础模型，通过定制化训练策略和架构设计解决这些问题。在优化过程中，我们提出多阶段训练方案以提高表示学习的多方面有效性。具体而言，内容感知渐进训练旨在增强模型对多样化下游任务的适应性，掌握丰富的跨模态能力。协作感知推荐增强训练通过从序列到项目和ID到项目嵌入中提炼知识，同时挖掘用户历史兴趣，使多模态表示适应推荐场景。同时，我们开发了随机专业化和数据集驱动的模式匹配，增强模型训练的灵活性和泛化能力。实验结果表明，SAIL-Embedding在不同检索任务中与其他方法相比实现了SOTA性能。在各种集成我们模型的现实场景的在线实验中，我们观察到作为推荐体验关键指标的Lifetime (LT)显著增加。例如，在抖音精选场景中，模型带来了+0.5%的7天LT增益。对于抖音feed排序模型，SAIL-Embedding产生的匹配特征带来了+0.1%的AUC增益。


### 论文摘要

Multimodal embedding models aim to yield informative unified representations that empower diverse cross-modal tasks. Despite promising developments in the evolution from CLIP-based dual-tower architectures to large vision-language models, prior works still face unavoidable challenges in real-world applications and business scenarios, such as the limited modality support, unstable training mechanisms, and industrial domain gaps. In this work, we introduce SAIL-Embedding, an omni-modal embedding foundation model that addresses these issues through tailored training strategies and architectural design. In the optimization procedure, we propose a multi-stage training scheme to boost the multifaceted effectiveness of representation learning. Specifically, the content-aware progressive training aims to enhance the model's adaptability to diverse downstream tasks and master enriched cross-modal proficiency. The collaboration-aware recommendation enhancement training further adapts multimodal representations for recommendation scenarios by distilling knowledge from sequence-to-item and ID-to-item embeddings while mining user historical interests. Concurrently, we develop the stochastic specialization and dataset-driven pattern matching to strengthen model training flexibility and generalizability. Experimental results show that SAIL-Embedding achieves SOTA performance compared to other methods in different retrieval tasks. In online experiments across various real-world scenarios integrated with our model, we observe a significant increase in Lifetime (LT), which is a crucial indicator for the recommendation experience. For instance, the model delivers the 7-day LT gain of +0.5% in the Douyin-Selected scenario. For the Douyin feed rank model, the match features produced by SAIL-Embedding yield a +0.1% AUC gain.

---

## 21. SWIR-LightFusion: Multi-spectral Semantic Fusion of Synthetic SWIR with {Thermal} IR {(LWIR/MWIR)} and RGB

**论文链接:** [http://arxiv.org/abs/2510.13404v1](http://arxiv.org/abs/2510.13404v1)

**作者:** Muhammad Ishfaq Hussain, Ma Van Linh, Zubia Naz, Unse Fatima, Yeongmin Ko, Moongu Jeon

**发布时间:** 2025-10-15

### GPT解析

### 总结

研究提出了一种基于合成短波红外（SWIR）的多模态融合框架，用于改善恶劣能见度条件下的场景理解，解决了传统成像模态在融合时难以提供全面场景信息的问题。

### 背景

在恶劣能见度条件下增强场景理解对监控和自主导航系统是一个关键挑战。传统的RGB和热红外成像模态在融合时难以在大气干扰或照明不足条件下提供全面场景信息。

### 目的

解决传统成像模态的局限性，克服SWIR成像发展和实施中因缺乏公开SWIR数据集而面临的障碍，提高融合图像质量。

### 方法

利用先进对比度增强技术从现有长波红外（LWIR）数据合成类似SWIR的结构/对比度线索图像，提出多模态融合框架集成合成SWIR、LWIR和RGB模态，采用优化的编码器-解码器神经网络架构和softmax门控融合头。

### 主要发现

在多个公共RGB-LWIR基准和私有真实RGB-MWIR-SWIR数据集上的实验表明，该框架提高了融合图像质量（对比度、边缘定义、结构保真度）同时保持实时性能，并优于其他三模态融合方法。

### 结论

合成SWIR增强的多模态融合框架在监控和自主系统中有实际应用的巨大潜力。

### 翻译

在恶劣能见度条件下增强场景理解对监控和自主导航系统仍然是一个关键挑战。传统的成像模态，如RGB和热红外（中波/长波），在融合时往往难以提供全面的场景信息，特别是在大气干扰或照明不足的条件下。为解决这些限制，短波红外（SWIR）成像因其能够穿透大气干扰并以更清晰的分辨率区分材料而成为一种有前景的模态。然而，基于SWIR系统的发展和广泛实施面临重大障碍，主要是由于缺乏公开可访问的SWIR数据集。为应对这一挑战，我们的研究提出了一种方法，利用先进的对比度增强技术从现有的LWIR数据中合成类似SWIR的结构/对比度线索图像（不声称光谱再现）。然后我们提出了一个多模态融合框架，集成合成SWIR、LWIR和RGB模态，采用具有模态特定编码器和softmax门控融合头的优化编码器-解码器神经网络架构。在公共RGB-LWIR基准（M3FD、TNO、CAMEL、MSRS、RoadScene）和额外的私有真实RGB-MWIR-SWIR数据集上的全面实验表明，我们的合成SWIR增强融合框架提高了融合图像质量（对比度、边缘定义、结构保真度）同时保持实时性能。我们还添加了公平的三模态基线（LP、LatLRR、GFF）和U2Fusion/SwinFusion的级联三模态变体，采用统一协议。结果突显了在监控和自主系统中实际应用的巨大潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在不良能见度条件下增强场景理解的问题，这对监控系统和自主导航系统至关重要。传统RGB和热红外成像融合在恶劣天气（如雾、烟、低光）下难以提供全面场景信息，而短波红外（SWIR）虽能穿透大气干扰并更好区分材料，但因公开SWIR数据集稀缺限制了其应用。这一问题在自动驾驶、安防监控等领域尤为重要，因为系统需要在各种环境条件下保持可靠的环境感知能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到RGB和热红外融合的局限性，注意到SWIR的优势但受限于数据集稀缺。他们借鉴了之前自己的LightFusion工作（使用灰度作为第三模态），但发现灰度无法充分体现SWIR优势。作者还参考了传统多尺度变换技术、深度学习方法（自编码器、CNN、GAN）以及红外-可见光图像融合（IVIF）领域的研究。基于这些思考，他们设计了一种使用对比度受限自适应直方图均衡化（CLAHE）从LWIR合成SWIR的方法，并开发了三模态融合框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过合成SWIR图像解决数据稀缺问题，并融合RGB、热红外和合成SWIR三种模态以提升场景理解能力。整体流程包括：1) 数据预处理（统一分辨率）；2) 使用CLAHE技术从LWIR生成合成SWIR图像；3) 使用三个模态特定编码器（RGB、MWIR和合成SWIR）独立提取特征，每个编码器采用轻量级梯度残块（Light-GRLB）；4) 通过softmax门控融合头整合多模态特征；5) 使用解码器重建高质量融合图像；6) 采用语义损失函数进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 使用CLAHE从LWIR合成SWIR图像解决数据稀缺问题；2) 设计三模态融合框架（RGB、LWIR/MWIR和合成SWIR）；3) 开发轻量级梯度残块（Light-GRLB）进行高效特征提取；4) 为每种模态使用独立编码器确保精确特征提取；5) 采用softmax门控融合头加权整合多模态特征。相比之前工作（如作者自己的LightFusion），不同之处在于使用合成SWIR替代灰度作为第三模态，采用专门设计的Light-GRLB而非标准卷积层，以及引入三重语义一致性损失函数。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种创新的多模态图像融合方法，通过合成生成短波红外图像并与RGB和热红外图像融合，显著提升了在不良能见度条件下的场景理解能力，同时保持了实时处理的效率。'}


### 论文摘要

Enhancing scene understanding in adverse visibility conditions remains a critical challenge for surveillance and autonomous navigation systems. Conventional imaging modalities, such as RGB and thermal infrared (MWIR / LWIR), when fused, often struggle to deliver comprehensive scene information, particularly under conditions of atmospheric interference or inadequate illumination. To address these limitations, Short-Wave Infrared (SWIR) imaging has emerged as a promising modality due to its ability to penetrate atmospheric disturbances and differentiate materials with improved clarity. However, the advancement and widespread implementation of SWIR-based systems face significant hurdles, primarily due to the scarcity of publicly accessible SWIR datasets. In response to this challenge, our research introduces an approach to synthetically generate SWIR-like structural/contrast cues (without claiming spectral reproduction) images from existing LWIR data using advanced contrast enhancement techniques. We then propose a multimodal fusion framework integrating synthetic SWIR, LWIR, and RGB modalities, employing an optimized encoder-decoder neural network architecture with modality-specific encoders and a softmax-gated fusion head. Comprehensive experiments on public {RGB-LWIR benchmarks (M3FD, TNO, CAMEL, MSRS, RoadScene) and an additional private real RGB-MWIR-SWIR dataset} demonstrate that our synthetic-SWIR-enhanced fusion framework improves fused-image quality (contrast, edge definition, structural fidelity) while maintaining real-time performance. We also add fair trimodal baselines (LP, LatLRR, GFF) and cascaded trimodal variants of U2Fusion/SwinFusion under a unified protocol. The outcomes highlight substantial potential for real-world applications in surveillance and autonomous systems.

---

## 22. DepthVLA: Enhancing Vision-Language-Action Models with Depth-Aware Spatial Reasoning

**论文链接:** [http://arxiv.org/abs/2510.13375v1](http://arxiv.org/abs/2510.13375v1)

**作者:** Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Zhuoguang Chen, Tao Jiang, Hang Zhao

**发布时间:** 2025-10-15

### GPT解析

### 总结

DepthVLA是一种简单而有效的VLA架构，通过预训练的深度预测模块明确整合空间感知能力，采用混合变压器设计统一VLM、深度变换器和动作专家，具有完全共享的注意力机制，形成具有增强空间推理能力的端到端模型。

### 背景

Vision-Language-Action (VLA) 模型最近展示了出色的泛化和语言引导的操作能力，但在需要精确空间推理的任务上表现不佳，这是由于从Vision-Language Models (VLMs) 继承的有限空间推理能力造成的。

### 目的

解决现有VLA模型在需要精确空间推理的任务上表现不佳的问题，提高模型的空间理解和推理能力。

### 方法

提出DepthVLA架构，通过预训练的深度预测模块明确整合空间感知能力，采用混合变压器设计统一VLM、深度变换器和动作专家，具有完全共享的注意力机制，形成端到端模型。

### 主要发现

在现实世界和模拟环境中的广泛评估表明，DepthVLA优于最先进的方法，在现实世界任务中取得了78.5%对比65.0%的进展，在LIBERO模拟器中为94.9%对比93.6%，在Simpler模拟器中为74.8%对比58.8%。

### 结论

DepthVLA通过明确整合空间感知能力，显著提高了VLA模型在需要精确空间推理任务上的表现，代码将公开可用。

### 翻译

Vision-Language-Action (VLA) 模型最近展示了出色的泛化和语言引导的操作能力。然而，在需要精确空间推理的任务上，它们的性能会下降，这是由于从Vision-Language Models (VLMs) 继承的有限空间推理能力。现有的VLA依赖于大量的动作数据预训练来将VLMs定位在3D空间中，这降低了训练效率，并且仍然不足以进行准确的空间理解。在这项工作中，我们提出了DepthVLA，一种简单而有效的VLA架构，通过预训练的深度预测模块明确地整合了空间感知能力。DepthVLA采用混合变压器设计，统一了VLM、深度变换器和动作专家，具有完全共享的注意力机制，形成具有增强空间推理能力的端到端模型。在现实世界和模拟环境中的广泛评估表明，DepthVLA优于最先进的方法，在现实世界任务中取得了78.5%对比65.0%的进展，在LIBERO模拟器中为94.9%对比93.6%，在Simpler模拟器中为74.8%对比58.8%。我们的代码将公开可用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决Vision-Language-Action(VLA)模型在需要精确空间推理的任务上性能下降的问题。这个问题很重要，因为机器人需要精确的空间感知能力来完成精细操作，如抓取小物体、执行精确操作或避免碰撞。现有的VLA模型依赖大量动作数据预训练来将VLMs嵌入3D空间，这降低了训练效率，且仍无法满足精确空间理解的需求，限制了机器人在实际应用中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有VLA模型在空间推理方面的局限性，以及现有方法（如大量动作数据预训练或生成世界模型）的不足来设计新方法。他们借鉴了π0的mixture-of-transformers(MoT)设计，并利用3D感知领域的最新进展，特别是Depth Anything V2作为深度专家的基础。作者提出通过预训练的深度预测模块显式整合空间感知能力，采用混合transformers架构统一VLM、深度专家和动作专家，并使用块状掩码保持预训练模块的学习能力，同时允许每个组件在不同数据集上分别预训练。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过预训练的深度预测模块显式整合空间推理能力到VLA模型中，利用混合transformers架构统一三个专家（VLM、深度专家和动作专家），在保持预训练知识的同时融合语义和空间线索以生成精确动作。整体实现流程包括：1)模型架构设计，包含VLM专家（编码图像和指令）、深度专家（处理图像推断几何信息）和动作专家（生成连续动作）；2)首先在多样化3D数据集上预训练深度专家；3)然后在具身动作数据上训练整个DepthVLA模型，使用模仿学习目标和流动匹配损失；4)在推理过程中，三个专家并行处理输入，共享注意力机制，动作专家基于融合的特征生成连续动作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)DepthVLA架构，首次将预训练的深度预测专家集成到VLA框架中；2)按专家预训练策略，允许每个专家在不同数据集上分别预训练；3)在所有中间层执行空间推理，提供更丰富的几何特征；4)端到端联合优化空间推理和动作生成。与之前工作的不同之处在于：相比现有VLA模型，不依赖大量动作数据预训练；相比SpatialVLA，深度专家是端到端优化的；相比生成世界模型，明确编码当前场景的3D知识且无高延迟；相比CoT推理方法，避免了自回归生成空间令牌的高延迟问题，推理时间仅增加20毫秒。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DepthVLA通过集成预训练的深度专家到混合transformers框架中，显著提升了机器人在需要精确空间推理任务上的性能，同时保持了高效的训练和推理速度。'}


### 论文摘要

Vision-Language-Action (VLA) models have recently shown impressive generalization and language-guided manipulation capabilities. However, their performance degrades on tasks requiring precise spatial reasoning due to limited spatial reasoning inherited from Vision-Language Models (VLMs). Existing VLAs rely on extensive action-data pretraining to ground VLMs in 3D space, which reduces training efficiency and is still insufficient for accurate spatial understanding. In this work, we present DepthVLA, a simple yet effective VLA architecture that explicitly incorporates spatial awareness through a pretrained depth prediction module. DepthVLA adopts a mixture-of-transformers design that unifies a VLM, a depth transformer, and an action expert with fully shared attentions, forming an end-to-end model with enhanced spatial reasoning. Extensive evaluations in both real-world and simulated environments show that DepthVLA outperforms state-of-the-art approaches, achieving 78.5% vs. 65.0% progress in real-world tasks, 94.9% vs. 93.6% in the LIBERO simulator, and 74.8% vs. 58.8% in the Simpler simulator. Our code will be made publicly available.

---

## 23. FlyAwareV2: A Multimodal Cross-Domain UAV Dataset for Urban Scene Understanding

**论文链接:** [http://arxiv.org/abs/2510.13243v1](http://arxiv.org/abs/2510.13243v1)

**作者:** Francesco Barbato, Matteo Caligiuri, Pietro Zanuttigh

**发布时间:** 2025-10-15

**备注:** 20 pages, 7 figures, 10 tables, data and code available

### GPT解析

### 总结

FlyAwareV2是一个新的多模态数据集，包含真实和合成的无人机图像，专为城市场景理解任务设计，解决了真实数据收集和标注的挑战。

### 背景

城市环境中无人机应用的计算机视觉算法开发严重依赖于大规模、准确标注的数据集，但收集和标注真实世界无人机数据极其困难且成本高昂。

### 目的

解决真实数据收集和标注的局限性，提供一个包含真实和合成无人机图像的多模态数据集，用于城市场景理解任务。

### 方法

基于SynDrone和FlyAware数据集开发FlyAwareV2，引入多模态数据(RGB、深度、语义标签)覆盖不同环境条件；通过最先进单目深度估计计算真实样本深度图；提供RGB和多模态语义分割基准；研究合成到真实域适应以评估模型泛化能力。

### 主要发现

FlyAwareV2具有丰富的标注和环境多样性，为基于无人机的3D城市场景理解研究提供了宝贵资源。

### 结论

FlyAwareV2通过其丰富的标注集和环境多样性，为基于无人机的3D城市场景理解研究提供了有价值的资源。

### 翻译

针对城市环境中无人机应用开发的计算机视觉算法严重依赖于具有准确标注的大规模数据集的可用性。然而，收集和标注真实世界的无人机数据极其具有挑战性和成本高昂。为了解决这一局限性，我们提出了FlyAwareV2，这是一个新颖的多模态数据集，包含专为城市场景理解任务定制的真实和合成无人机图像。基于最近引入的SynDrone和FlyAware数据集，FlyAwareV2引入了几个新的关键贡献：1)跨不同环境条件的多模态数据(RGB、深度、语义标签)，包括变化的天气和白天时间；2)通过最先进的单目深度估计计算的真实样本深度图；3)基于标准架构的RGB和多模态语义分割基准；4)关于合成到真实域适应的研究，以评估在合成数据上训练的模型的泛化能力。凭借其丰富的标注集和环境多样性，FlyAwareV2为基于无人机的3D城市场景理解研究提供了宝贵的资源。


### 论文摘要

The development of computer vision algorithms for Unmanned Aerial Vehicle (UAV) applications in urban environments heavily relies on the availability of large-scale datasets with accurate annotations. However, collecting and annotating real-world UAV data is extremely challenging and costly. To address this limitation, we present FlyAwareV2, a novel multimodal dataset encompassing both real and synthetic UAV imagery tailored for urban scene understanding tasks. Building upon the recently introduced SynDrone and FlyAware datasets, FlyAwareV2 introduces several new key contributions: 1) Multimodal data (RGB, depth, semantic labels) across diverse environmental conditions including varying weather and daytime; 2) Depth maps for real samples computed via state-of-the-art monocular depth estimation; 3) Benchmarks for RGB and multimodal semantic segmentation on standard architectures; 4) Studies on synthetic-to-real domain adaptation to assess the generalization capabilities of models trained on the synthetic data. With its rich set of annotations and environmental diversity, FlyAwareV2 provides a valuable resource for research on UAV-based 3D urban scene understanding.

---

## 24. True Self-Supervised Novel View Synthesis is Transferable

**论文链接:** [http://arxiv.org/abs/2510.13063v1](http://arxiv.org/abs/2510.13063v1)

**作者:** Thomas W. Mitchel, Hyunwoo Ryu, Vincent Sitzmann

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了XFactor，第一个无需几何信息且能实现真正新颖视图合成的自监督模型，其关键标准是姿态表示的可转移性。

### 背景

先前关于自监督新颖视图合成的工作分析表明，它们预测的姿态不具备可转移性——相同姿态在不同3D场景中会导致不同的相机轨迹。

### 目的

开发一个能够实现真正新颖视图合成的模型，其关键标准是姿态表示的可转移性。

### 方法

XFactor结合了成对姿态估计和简单的输入输出增强方案，能够将相机姿态与场景内容分离并促进几何推理，使用不受约束的潜在姿态变量，无需任何3D归纳偏置或多视图几何概念。

### 主要发现

XFactor实现了姿态表示的可转移性；引入了一种新的可转移性量化指标；大规模实验表明XFactor显著优于之前无需姿态的NVS变换器；探测实验显示潜在姿态与现实世界姿态高度相关。

### 结论

XFactor是第一个无需几何信息且能实现真正新颖视图合成的自监督模型，通过结合成对姿态估计和输入输出增强方案，成功实现了姿态表示的可转移性。

### 翻译

在这篇论文中，我们确定判断一个模型是否真正具备新颖视图合成(NVS)能力的关键标准是可转移性：即从一段视频序列中提取的任何姿态表示是否可用于在另一场景中重新渲染相同的相机轨迹。我们分析了之前关于自监督NVS的工作，发现它们预测的姿态不具备可转移性：相同的姿态集在不同3D场景中会导致不同的相机轨迹。在这里，我们提出了XFactor，这是第一个无需几何信息且能实现真正NVS的自监督模型。XFactor结合了成对姿态估计和简单的输入输出增强方案，能够将相机姿态与场景内容分离并促进几何推理。值得注意的是，我们证明XFactor使用不受约束的潜在姿态变量实现了可转移性，无需任何3D归纳偏置或多视图几何概念——例如将姿态显式参数化为SE(3)的元素。我们引入了一种新的量化可转移性的指标，并通过大规模实验证明XFactor显著优于之前无需姿态的NVS变换器，并通过探测实验显示潜在姿态与现实世界姿态高度相关。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自监督新视角合成(NVS)的可转移性问题，即能否从一个视频序列中提取的姿态表示用于重新渲染另一个视频序列中的相同相机轨迹。这个问题很重要，因为真正的NVS应该允许用户控制视角，相同的相机姿态应该总是渲染相同的视角。如果模型无法做到这一点，它就不是真正的NVS模型，而只是帧插值器，限制了用户在任意场景中定义想要渲染的视图的能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先确定可转移性是NVS的关键标准，然后分析现有自监督方法发现它们预测的姿态不能跨场景转移。作者提出两个关键见解：1)通过从必须外推的双视图模型开始训练来防止插值；2)将可转移性明确为训练目标，使用保持相机姿态但最小化像素内容重叠的增强方案。作者借鉴了RUST的无几何方法和CroCo的单目渲染思想，但通过创新的训练目标和架构设计解决了可转移性问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是XFactor模型，通过成对姿态估计和输入输出的简单增强方案相结合，解耦相机姿态和场景内容，实现无几何约束的可转移NVS。整体流程：1)训练立体-单目模型(只用一对图像)，消除插值路径；2)应用可转移性目标训练，确保一个序列的姿态能用于渲染另一序列；3)使用保持相机姿态的增强方案(如逆掩码)生成像素差异大的图像对；4)通过二次训练将立体模型扩展为多视图模型，支持更复杂的场景渲染。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出可转移性作为真正NVS的标准并引入TPS指标；2)识别现有方法实际是插值而非推理视角；3)提出促进可转移性的训练目标和增强策略；4)提出XFactor，首个完全自监督的可转移NVS模型；5)通过大规模实验验证有效性。相比RayZer(使用SE(3)参数化但降低可转移性)和RUST(仍受插值偏差影响)，XFactor不需要任何几何先验，实现了真正的跨场景姿态控制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'XFactor提出了第一个完全自监督且无几何的新视角合成模型，通过可转移性训练目标实现了真正的相机姿态控制，使相同的相机姿态能够在不同场景间产生一致的视角渲染。'}


### 论文摘要

In this paper, we identify that the key criterion for determining whether a model is truly capable of novel view synthesis (NVS) is transferability: Whether any pose representation extracted from one video sequence can be used to re-render the same camera trajectory in another. We analyze prior work on self-supervised NVS and find that their predicted poses do not transfer: The same set of poses lead to different camera trajectories in different 3D scenes. Here, we present XFactor, the first geometry-free self-supervised model capable of true NVS. XFactor combines pair-wise pose estimation with a simple augmentation scheme of the inputs and outputs that jointly enables disentangling camera pose from scene content and facilitates geometric reasoning. Remarkably, we show that XFactor achieves transferability with unconstrained latent pose variables, without any 3D inductive biases or concepts from multi-view geometry -- such as an explicit parameterization of poses as elements of SE(3). We introduce a new metric to quantify transferability, and through large-scale experiments, we demonstrate that XFactor significantly outperforms prior pose-free NVS transformers, and show that latent poses are highly correlated with real-world poses through probing experiments.

---

## 25. VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages

**论文链接:** [http://arxiv.org/abs/2510.12845v1](http://arxiv.org/abs/2510.12845v1)

**作者:** Jesse Atuhurra, Iqra Ali, Tomoya Iwakura, Hidetaka Kamigaito, Tatsuya Hiraoka

**发布时间:** 2025-10-14

### GPT解析

### 总结

本研究引入了一个新的多语言基准测试VLURes，用于评估视觉语言模型(VLMs)在细粒度视觉和语言理解能力方面的表现，特别是在长文本设置下。

### 背景

视觉语言模型(VLMs)对推进智能体的感知能力至关重要，但目前的评估主要局限于以英语为中心的基准测试，且图像-文本对仅包含短文本。

### 目的

评估VLMs在四种语言(英语、日语、斯瓦希里语和乌尔都语)的长文本设置下的细粒度能力，特别是物体识别、场景理解和关系理解等对智能体至关重要的任务。

### 方法

开发了包含八个视觉语言任务和一个不相关性任务的多语言基准测试VLURes；从目标语言网页资源收集包含十个不同图像类别和丰富文本上下文的数据集；通过提示VLMs生成回答和理由，并由自动系统和母语人士进行评估。

### 主要发现

揭示了VLMs在不同语言和任务上的表现差异；表现最好的模型GPT-4o总体准确率为90.8%，比人类表现低6.7%；开源模型与人类表现之间的差距更大。

### 结论

VLURes基准测试在开发能够处理多模态视觉推理的智能体方面发挥着关键作用，特别是在低资源语言环境中的应用。

### 翻译

视觉语言模型(VLMs)对推进智能体的感知能力至关重要。然而，对VLMs的评估仍然主要局限于以英语为中心的基准测试，其中图像-文本对仅包含短文本。为了评估VLMs在四种语言下的细粒度能力，特别是在长文本设置下，我们引入了一个新的多语言基准测试VLURes，包含八个视觉语言任务和一个开创性的不相关性任务，用于探测VLMs在英语、日语以及低资源语言斯瓦希里语和乌尔都语中的细粒度视觉和语言理解能力。我们的数据集从目标语言的网页资源中精心策划，包含十个不同的图像类别和丰富的文本上下文，为斯瓦希里语和乌尔都语引入了宝贵的视觉语言资源。通过提示VLMs生成回答和理由，并由自动系统和母语人士评估，我们揭示了VLMs在不同语言和任务上的表现差异，这些任务对智能体至关重要，如物体识别、场景理解和关系理解。我们使用VLURes评估了十个VLMs。表现最好的模型GPT-4o总体准确率达到90.8%，比人类表现低6.7%，尽管开源模型之间的差距更大。这一差距凸显了VLURes在开发能够处理多模态视觉推理的智能体方面的关键作用。


### 论文摘要

Vision Language Models (VLMs) are pivotal for advancing perception in intelligent agents. Yet, evaluation of VLMs remains limited to predominantly English-centric benchmarks in which the image-text pairs comprise short texts. To evaluate VLM fine-grained abilities, in four languages under long-text settings, we introduce a novel multilingual benchmark VLURes featuring eight vision-and-language tasks, and a pioneering unrelatedness task, to probe the fine-grained Visual and Linguistic Understanding capabilities of VLMs across English, Japanese, and low-resource languages, Swahili, and Urdu. Our datasets, curated from web resources in the target language, encompass ten diverse image categories and rich textual context, introducing valuable vision-language resources for Swahili and Urdu. By prompting VLMs to generate responses and rationales, evaluated automatically and by native speakers, we uncover performance disparities across languages and tasks critical to intelligent agents, such as object recognition, scene understanding, and relationship understanding. We conducted evaluations of ten VLMs with VLURes. The best performing model, GPT-4o, achieves an overall accuracy of 90.8% and lags human performance by 6.7%, though the gap is larger for open-source models. The gap highlights VLURes' critical role in developing intelligent agents to tackle multi-modal visual reasoning.

---

## 26. USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots

**论文链接:** [http://arxiv.org/abs/2510.07869v3](http://arxiv.org/abs/2510.07869v3)

**作者:** Junwen Gu, Zhiheng Wu, Pengxuan Si, Shuang Qiu, Yukai Feng, Luoyang Sun, Laien Luo, Lianyi Yu, Jian Wang, Zhengxing Wu

**发布时间:** 2025-10-09

**备注:** Project Page: https://vincentgu2000.github.io/u0project/

### GPT解析

### 总结

这篇论文介绍了一个名为USIM的水下机器人模拟数据集和一个名为U0的VLA模型，它们共同解决了水下环境中机器人操作面临的挑战，特别是在数据稀缺的情况下，通过提供大规模高质量数据集和有效的多任务学习方法。

### 背景

水下环境为机器人操作带来了独特的挑战，包括复杂的流体动力学、有限的视野和受限的通信。虽然数据驱动的方法已经在陆地机器人上推动了具身智能的发展，并使专用水下机器人能够自主工作，但开发能够自主执行多项任务的水下智能仍然非常困难，因为大规模、高质量的水下数据集仍然稀缺。

### 目的

为了解决这些限制，作者引入了USIM，这是一个基于模拟的多任务视觉-语言-动作数据集，用于水下机器人。

### 方法

USIM包含来自1,852个轨迹的超过561K帧，总计约15.6小时的BlueROV2交互，涵盖9个不同场景中的20项任务，范围从视觉导航到移动操作。基于这个数据集，作者提出了U0，这是一个用于通用水下机器人的VLA模型，它通过多模态融合整合双目视觉和其他传感器模态，并进一步采用基于卷积-注意力的感知增强模块(CAP)来提高空间理解和移动操作能力。

### 主要发现

在检查、避障、扫描和动态跟踪等任务中，该框架实现了80%的成功率，而在具有挑战性的移动操作任务中，与基线方法相比，它将到目标的距离减少了21.2%，证明了其有效性。

### 结论

USIM和U0表明VLA模型可以有效地应用于水下机器人应用，为可扩展的数据集构建、改进的任务自主性和智能通用水下机器人的实际实现提供了基础。

### 翻译

水下环境为机器人操作带来了独特的挑战，包括复杂的流体动力学、有限的视野和受限的通信。虽然数据驱动的方法已经在陆地机器人上推动了具身智能的发展，并使专用水下机器人能够自主工作，但开发能够自主执行多项任务的水下智能仍然非常困难，因为大规模、高质量的水下数据集仍然稀缺。为了解决这些限制，我们引入了USIM，这是一个基于模拟的多任务视觉-语言-动作数据集，用于水下机器人。USIM包含来自1,852个轨迹的超过561K帧，总计约15.6小时的BlueROV2交互，涵盖9个不同场景中的20项任务，范围从视觉导航到移动操作。基于这个数据集，我们提出了U0，这是一个用于通用水下机器人的VLA模型，它通过多模态融合整合双目视觉和其他传感器模态，并进一步采用基于卷积-注意力的感知增强模块(CAP)来提高空间理解和移动操作能力。在检查、避障、扫描和动态跟踪等任务中，该框架实现了80%的成功率，而在具有挑战性的移动操作任务中，与基线方法相比，它将到目标的距离减少了21.2%，证明了其有效性。USIM和U0表明VLA模型可以有效地应用于水下机器人应用，为可扩展的数据集构建、改进的任务自主性和智能通用水下机器人的实际实现提供了基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决水下机器人自主执行多任务的困难问题，原因是水下环境存在复杂流体动力学、有限可见性和受限通信等挑战，同时大规模高质量水下数据集稀缺。这个问题很重要，因为水下环境覆盖地球71%的面积，涉及海洋生态调查、资源开发、管道检查等多种应用，而水下操作对人类来说危险且困难，自主水下机器人能大幅提高效率和安全性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了水下机器人面临的特殊挑战，然后选择使用仿真环境解决真实数据收集成本高的问题。他们基于现有Isaac-GR00T N1.5模型进行改进，而非从头训练，并整合了双目视觉和其他传感器模态。借鉴了Stonefish模拟器构建环境、ROS框架进行数据收集、Vision-Language Model和Diffusion Transformer架构，以及PID控制器和MoveIt进行机械手控制等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过仿真环境构建大规模水下VLA数据集，开发适应水下环境的VLA模型，建立可扩展的数据到任务框架。流程包括：1)用Stonefish模拟器构建9种水下场景和BlueROV2机器人模型；2)收集20个任务的数据，共561K帧和15.6小时交互数据；3)基于Isaac-GR00T N1.5开发U0模型，整合多模态传感器数据和CAP感知增强模块；4)通过开环离线评估和闭环在线测试验证模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个大规模水下多任务VLA数据集USIM，覆盖20个任务和9种场景；2)首个专为水下机器人设计的VLA模型U0，整合多模态传感器数据和CAP模块；3)使用以机器人为中心的坐标系表示目标位置。相比之前工作，USIM是首个多任务VLA数据集，而现有数据集多为特定任务；U0是首个专门针对水下环境的VLA模型，考虑了水下视觉退化和特殊运动特性，能处理多种任务而非单一任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过构建首个大规模水下多任务Vision-Language-Action数据集USIM和开发专门的水下机器人通用模型U0，解决了水下机器人高质量数据稀缺和通用任务执行能力不足的问题，为构建可扩展的水下智能机器人框架奠定了基础。'}


### 论文摘要

Underwater environments present unique challenges for robotic operation, including complex hydrodynamics, limited visibility, and constrained communication. Although data-driven approaches have advanced embodied intelligence in terrestrial robots and enabled task-specific autonomous underwater robots, developing underwater intelligence capable of autonomously performing multiple tasks remains highly challenging, as large-scale, high-quality underwater datasets are still scarce. To address these limitations, we introduce USIM, a simulation-based multi-task Vision-Language-Action (VLA) dataset for underwater robots. USIM comprises over 561K frames from 1,852 trajectories, totaling approximately 15.6 hours of BlueROV2 interactions across 20 tasks in 9 diverse scenarios, ranging from visual navigation to mobile manipulation. Building upon this dataset, we propose U0, a VLA model for general underwater robots, which integrates binocular vision and other sensor modalities through multimodal fusion, and further incorporates a convolution-attention-based perception focus enhancement module (CAP) to improve spatial understanding and mobile manipulation. Across tasks such as inspection, obstacle avoidance, scanning, and dynamic tracking, the framework achieves a success rate of 80%, while in challenging mobile manipulation tasks, it reduces the distance to the target by 21.2% compared with baseline methods, demonstrating its effectiveness. USIM and U0 show that VLA models can be effectively applied to underwater robotic applications, providing a foundation for scalable dataset construction, improved task autonomy, and the practical realization of intelligent general underwater robots.

---

## 27. Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs

**论文链接:** [http://arxiv.org/abs/2510.13740v1](http://arxiv.org/abs/2510.13740v1)

**作者:** Mustafa Munir, Alex Zhang, Radu Marculescu

**发布时间:** 2025-10-15

**备注:** Published in the Proceedings of the Third Learning on Graphs  Conference (LoG 2024)

### GPT解析

### 总结

本研究提出了一种新的图构建方法Logarithmic Scalable Graph Construction (LSGC)和混合CNN-GNN模型LogViG，用于解决视觉图神经网络在大图像上计算成本高的问题，并通过引入高分辨率分支和多尺度特征融合提升了性能。

### 背景

Vision graph neural networks (ViG)作为传统卷积神经网络(CNN)和视觉Transformer(ViT)的竞争性替代方案在视觉任务中显示出潜力，但常见的图构建方法如k近邻(KNN)在大图像上计算成本高，而现有的Sparse Vision Graph Attention (SVGA)方法存在固定步长导致的过度压缩和错过多连接的问题。

### 目的

开发一种新的图构建方法，通过限制长距离链接的数量来提高视觉图神经网络的性能，同时降低计算复杂度，并构建一个结合CNN和GNN优势的混合模型。

### 方法

提出Logarithmic Scalable Graph Construction (LSGC)方法来增强性能，并设计了LogViG这一新型混合CNN-GNN模型；同时引入高分辨率分支，并在高分辨率和低分辨率分支之间融合特征，构建了多尺度高分辨率视觉GNN网络。

### 主要发现

LogViG在图像分类和语义分割任务上的准确率、GMACs和参数方面均优于现有的ViG、CNN和ViT架构；最小模型Ti-LogViG在ImageNet-1K上达到79.9%的平均top-1准确率，比Vision GNN高1.7%，参数减少24.3%，GMACs减少35.3%。

### 结论

通过提出的LSGC方法在ViG中利用长距离链接可以超过当前最先进ViG的性能，证明了该方法的有效性。

### 翻译

视觉图神经网络(ViG)作为传统卷积神经网络(CNN)和视觉Transformer(ViT)的竞争性替代方案，在视觉任务中显示出前景；然而，常见的图构建方法如k近邻(KNN)在大图像上可能计算成本高昂。虽然Sparse Vision Graph Attention (SVGA)等方法显示出前景，但SVGA的固定步长可能导致过度压缩和错过多个连接，无法从长距离链接中获取相同信息。基于这一观察，我们提出了一种新的图构建方法——对数可扩展图构建(LSGC)，通过限制长距离链接的数量来增强性能。为此，我们提出了LogViG，一种利用LSGC的新型混合CNN-GNN模型。此外，受多尺度和高分辨率架构成功的启发，我们引入并应用了一个高分辨率分支，并在高分辨率和低分辨率分支之间融合特征，构建了多尺度高分辨率视觉GNN网络。大量实验表明，LogViG在图像分类和语义分割任务上的准确率、GMACs和参数方面均优于现有的ViG、CNN和ViT架构。我们的最小模型Ti-LogViG在ImageNet-1K上达到79.9%的平均top-1准确率，标准差为0.2%，比Vision GNN高1.7%的平均准确率，参数减少24.3%，GMACs减少35.3%。我们的工作表明，通过我们提出的LSGC在ViG中利用长距离链接可以超过当前最先进ViG的性能。代码可在https://github.com/mmunir127/LogViG-Official获取。


### 论文摘要

Vision graph neural networks (ViG) have demonstrated promise in vision tasks as a competitive alternative to conventional convolutional neural nets (CNN) and transformers (ViTs); however, common graph construction methods, such as k-nearest neighbor (KNN), can be expensive on larger images. While methods such as Sparse Vision Graph Attention (SVGA) have shown promise, SVGA's fixed step scale can lead to over-squashing and missing multiple connections to gain the same information that could be gained from a long-range link. Through this observation, we propose a new graph construction method, Logarithmic Scalable Graph Construction (LSGC) to enhance performance by limiting the number of long-range links. To this end, we propose LogViG, a novel hybrid CNN-GNN model that utilizes LSGC. Furthermore, inspired by the successes of multi-scale and high-resolution architectures, we introduce and apply a high-resolution branch and fuse features between our high-resolution and low-resolution branches for a multi-scale high-resolution Vision GNN network. Extensive experiments show that LogViG beats existing ViG, CNN, and ViT architectures in terms of accuracy, GMACs, and parameters on image classification and semantic segmentation tasks. Our smallest model, Ti-LogViG, achieves an average top-1 accuracy on ImageNet-1K of 79.9% with a standard deviation of 0.2%, 1.7% higher average accuracy than Vision GNN with a 24.3% reduction in parameters and 35.3% reduction in GMACs. Our work shows that leveraging long-range links in graph construction for ViGs through our proposed LSGC can exceed the performance of current state-of-the-art ViGs. Code is available at https://github.com/mmunir127/LogViG-Official.

---

## 28. Message Passing on the Edge: Towards Scalable and Expressive GNNs

**论文链接:** [http://arxiv.org/abs/2510.13615v1](http://arxiv.org/abs/2510.13615v1)

**作者:** Pablo Barceló, Fabian Jogl, Alexander Kozachinskiy, Matthias Lanzinger, Stefan Neumann, Cristóbal Rojas

**发布时间:** 2025-10-15

### GPT解析

### 总结

本研究提出了EB-1WL和EB-GNN，一种基于边的颜色细化测试和相应的图神经网络架构。该架构受经典三角形计数算法启发，在消息传递过程中明确使用三角形信息。研究表明，EB-1WL比1-WL具有更强的表达能力，同时保持了接近线性的时间和内存复杂度。实验证明，EB-GNN是一种高效的通用架构，显著优于简单MPNN，且与任务专用GNN相比保持竞争力同时计算效率更高。

### 背景

图神经网络(GNN)领域存在对更具表达力且计算效率高的架构的需求，之前的提案在计算效率上存在问题。

### 目的

提出一种基于边的颜色细化测试(EB-1WL)和相应的GNN架构(EB-GNN)，以提高表达能力同时保持计算效率。

### 方法

提出EB-1WL（基于边的颜色细化测试）和EB-GNN架构，受Chiba和Nishizeki的经典三角形计数算法启发，在消息传递过程中明确使用三角形信息。

### 主要发现

EB-1WL比1-WL具有更强的表达能力；提供了基于一阶逻辑的EB-1WL完整逻辑表征和基于同态计数的匹配区分度结果；EB-1WL和EB-GNN在实际图学习任务中需要接近线性的时间和内存；EB-GNN显著优于简单MPNN，与任务专用GNN相比保持竞争力同时计算效率更高。

### 结论

EB-GNN是一种高效、通用的GNN架构，在表达能力与计算效率之间取得了良好的平衡。

### 翻译

我们提出了EB-1WL，一种基于边的颜色细化测试，以及相应的GNN架构EB-GNN。我们的架构受到Chiba和Nishizeki经典三角形计数算法的启发，并在消息传递过程中明确使用三角形。我们取得了以下结果：(1) EB-1WL比1-WL具有显著更强的表达能力。此外，我们基于一阶逻辑提供了EB-1WL的完整逻辑表征，并基于同态计数提供了匹配的区分度结果。(2) 与之前提出的更具表达力的GNN架构的重要区别在于，EB-1WL和EB-GNN在实际图学习任务中需要接近线性的时间和内存。(3) 从经验上看，我们表明EB-GNN是一种高效的通用架构：它显著优于简单的MPNN，并且在计算效率方面远高于任务专用的GNN的同时，与它们保持竞争力。


### 论文摘要

We propose EB-1WL, an edge-based color-refinement test, and a corresponding GNN architecture, EB-GNN. Our architecture is inspired by a classic triangle counting algorithm by Chiba and Nishizeki, and explicitly uses triangles during message passing. We achieve the following results: (1)~EB-1WL is significantly more expressive than 1-WL. Further, we provide a complete logical characterization of EB-1WL based on first-order logic, and matching distinguishability results based on homomorphism counting. (2)~In an important distinction from previous proposals for more expressive GNN architectures, EB-1WL and EB-GNN require near-linear time and memory on practical graph learning tasks. (3)~Empirically, we show that EB-GNN is a highly-efficient general-purpose architecture: It substantially outperforms simple MPNNs, and remains competitive with task-specialized GNNs while being significantly more computationally efficient.

---

## 29. F-BFQ: Flexible Block Floating-Point Quantization Accelerator for LLMs

**论文链接:** [http://arxiv.org/abs/2510.13401v1](http://arxiv.org/abs/2510.13401v1)

**作者:** Jude Haris, José Cano

**发布时间:** 2025-10-15

**备注:** Accepted to Workshop on New Approaches for Addressing the Computing  Requirements of LLMs and GNNs (LG-ARC) @ ISCA 2025

### GPT解析

### 总结

本文提出了一种名为F-BFQ的灵活块浮点量化加速器，用于提高BFP量化大语言模型在边缘设备上的推理效率，能够在两种BFP量化变体间动态切换而无需重新配置。

### 背景

大语言模型(LLMs)在日常任务中应用广泛，通过llama.cpp等推理框架的优化（如KV缓存和量化），使在边缘设备上部署LLMs变得更加可行。量化技术是使LLMs在资源受限的边缘设备上运行的关键，llama.cpp采用块浮点(BFP)量化来减小模型权重和输入张量的位宽、内存占用和计算需求。通常，LLMs在各层采用混合BFP量化以减少精度损失。

### 目的

为了高效加速BFP量化LLMs的各层计算，需要开发一种专门的加速器，使其能够支持不同的BFP量化变体而无需重新配置。

### 方法

作者提出了F-BFQ（Flexible Block Floating Point Quantization）加速器，该加速器可以动态切换两种BFP量化变体并执行矩阵乘法(MatMul)操作。

### 主要发现

在AMD Kria板上部署的初始F-BFQ加速器设计，在三种BFP量化LLMs上相比基于Arm NEON的CPU执行，平均减少了1.4倍的推理时间，实现了每秒5.2个token（约3.9个单词）的处理速度。

### 结论

F-BFQ加速器有效地提高了BFP量化LLMs在边缘设备上的推理效率，通过支持多种BFP量化变体的动态切换，无需重新配置即可加速模型各层的计算。

### 翻译

大语言模型(LLMs)在日常任务中日益突出，从改善语音转文本翻译到生成最新视频游戏的额外帧等。借助llama.cpp等支持KV缓存和量化等优化的LLM推理框架，现在在边缘设备上部署LLMs比以往任何时候都更容易。量化是使LLMs在资源受限的边缘设备上运行的基本技术，llama.cpp利用块浮点(BFP)量化大幅减小权重和输入张量的位宽、内存占用以及运行LLMs所需的计算能力。LLMs通常在模型各层采用混合BFP量化，以减少量化导致的模型精度损失。因此，为了高效加速BFP量化LLMs的各层，专门的加速器需要支持不同的BFP变体而无需重新配置。为解决这一问题，我们提出了F-BFQ（Flexible Block Floating Point Quantization）加速器，它可以在两种BFP量化变体间动态切换并执行矩阵乘法(MatMul)操作。我们在AMD Kria板上部署的初始F-BFQ加速器设计，在三种BFP量化LLMs上相比基于Arm NEON的CPU执行，平均减少了1.4倍的推理时间，同时实现了每秒5.2个token（约3.9个单词）的处理速度。


### 论文摘要

Large Language Models (LLMs) have become increasingly prominent for daily tasks, from improving sound-totext translation to generating additional frames for the latest video games. With the help of LLM inference frameworks, such as llama.cpp, which support optimizations such as KV-caching and quantization, it is now easier than ever to deploy LLMs on edge devices. Quantization is fundamental to enable LLMs on resource-constrained edge devices, and llama.cpp utilizes block floating point (BFP) quantization to drastically reduce the bit width of weights and input tensors, the memory footprint, and the computational power required to run LLMs. LLMs are typically quantized with mixed BFP quantization across the model layers to reduce the loss of model accuracy due to quantization. Therefore, to efficiently accelerate across the layers of BFP-quantized LLMs, specialized accelerators need to support different BFP variants without reconfiguration. To address this issue, we propose a Flexible Block FloatingPoint Quantization (F-BFQ) accelerator, which can dynamically switch between two BFP quantization variants and perform matrix multiplication (MatMul) operations. Our initial F-BFQ accelerator design, deployed on the AMD Kria board, reduces inference time by 1.4x on average over the Arm NEON-based CPU execution across three BFP quantized LLMs while achieving 5.2 tokens per second (~3.9 words per second).

---

## 30. Going with the Flow: Approximating Banzhaf Values via Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.13391v1](http://arxiv.org/abs/2510.13391v1)

**作者:** Benjamin Kempinski, Tal Kachman

**发布时间:** 2025-10-15

**备注:** 21 pages, 8 figures, 11-page appendix

### GPT解析

### 总结

这篇论文提出了一种使用图神经网络(GNN)来近似计算网络流游戏中Banzhaf值的方法，解决了传统方法在处理大规模系统时的计算效率问题。

### 背景

Banzhaf值用于量化多智能体系统中智能体的影响力，应用领域广泛，但精确计算对于超过约20个智能体的系统由于指数级复杂度而不可行；蒙特卡洛采样方法虽然可提供估计但存在样本复杂度高且无法跨网络配置泛化的问题。

### 目的

开发一种高效的方法来近似计算网络流游戏中的Banzhaf值，使其能够处理大规模和动态系统，并具备良好的泛化能力。

### 方法

使用图神经网络(GNN)将Banzhaf值计算问题框架化为图级预测任务，直接从网络拓扑和控制结构中学习智能体影响力的模式；比较了三种GNN架构(GAT、GINE和EdgeConv)在大型合成数据集上的性能。

### 主要发现

训练后的GNN模型实现了高保真的Banzhaf值近似，计算速度比传统方法快数量级；模型展示了强大的零样本泛化能力，能够在未见过的网络结构上准确预测Banzhaf值而无需重新训练。

### 结论

图神经网络可以作为复杂网络化系统可扩展合作博弈论分析的实用工具。

### 翻译

计算网络流游戏中的Banzhaf值对于量化多智能体系统中的智能体影响力至关重要，应用范围从网络安全到基础设施规划。然而，对于超过约20个智能体的系统，由于指数级复杂度，精确计算是不可行的。虽然蒙特卡洛采样方法可以提供统计估计，但它们存在样本复杂度高的问题，并且无法在不同网络配置之间转移知识，使其对于大规模或动态系统不切实际。我们提出了一种基于学习的新方法，使用图神经网络(GNN)来近似基数网络流游戏中的Banzhaf值。通过将问题框架化为图级预测任务，我们的方法直接从网络拓扑和控制结构中学习智能体影响力的可泛化模式。我们进行了全面的实证研究，比较了三种最先进的GNN架构-图注意力网络(GAT)、具有边特征的图同构网络(GINE)和EdgeConv-在每个配置200,000个图的大规模合成数据集上的性能，数据集在大小(20-100个节点)、智能体数量(5-20)和边概率(0.5-1.0)上有所不同。我们的结果表明，训练后的GNN模型实现了高保真的Banzhaf值近似，与精确和基于采样的方法相比，速度提高了数量级。最重要的是，我们展示了强大的零样本泛化能力：在特定大小和拓扑的图上训练的模型，能够准确预测具有完全不同结构特性的全新网络的Banzhaf值，而无需重新训练。这项工作确立了GNN作为复杂网络化系统可扩展合作博弈论分析的实用工具。


### 论文摘要

Computing the Banzhaf value in network flow games is fundamental for quantifying agent influence in multi-agent systems, with applications ranging from cybersecurity to infrastructure planning. However, exact computation is intractable for systems with more than $\sim20$ agents due to exponential complexity $\mathcal{O}(2^m)$. While Monte Carlo sampling methods provide statistical estimates, they suffer from high sample complexity and cannot transfer knowledge across different network configurations, making them impractical for large-scale or dynamic systems. We present a novel learning-based approach using Graph Neural Networks (GNNs) to approximate Banzhaf values in cardinal network flow games. By framing the problem as a graph-level prediction task, our method learns generalisable patterns of agent influence directly from network topology and control structure. We conduct a comprehensive empirical study comparing three state-of-the-art GNN architectures-Graph Attention Networks (GAT), Graph Isomorphism Networks with Edge features (GINE), and EdgeConv-on a large-scale synthetic dataset of 200,000 graphs per configuration, varying in size (20-100 nodes), agent count (5-20), and edge probability (0.5-1.0). Our results demonstrate that trained GNN models achieve high-fidelity Banzhaf value approximation with order-of-magnitude speedups compared to exact and sampling-based methods. Most significantly, we show strong zero-shot generalisation: models trained on graphs of a specific size and topology accurately predict Banzhaf values for entirely new networks with different structural properties, without requiring retraining. This work establishes GNNs as a practical tool for scalable cooperative game-theoretic analysis of complex networked systems.

---

## 31. Rethinking Graph Domain Adaptation: A Spectral Contrastive Perspective

**论文链接:** [http://arxiv.org/abs/2510.13254v1](http://arxiv.org/abs/2510.13254v1)

**作者:** Haoyu Zhang, Yuxuan Cheng, Wenqi Fan, Yulong Chen, Yifan Zhang

**发布时间:** 2025-10-15

**DOI:** 10.1007/978-3-032-06106-5_26

**备注:** This paper is accepted by ECML-PKDD 2025

### GPT解析

### 总结

FracNet是一种频率感知对比图网络，通过频谱分析将图分解为高频和低频成分，解决了图神经网络在领域适应中的挑战，通过对比学习框架改善了领域适应的模糊边界问题。

### 背景

图神经网络在各种领域取得了显著成功，但由于结构分布的显著变化和可转移模式探索不足，它们在领域适应方面常常遇到困难。传统方法没有区分处理全局和局部模式，导致多层GNN后图中的一些局部细节可能被破坏。

### 目的

提出一种新的方法来更好地理解和处理领域适应中的分布变化，特别是通过频谱分析来识别和利用不同频率成分中的模式，以提高图神经网络在领域适应中的性能。

### 方法

提出FracNet（频率感知对比图网络），包含两个协同模块，将原始图分解为高频和低频成分，并进行频率感知的领域适应。通过与对比学习框架集成，改善了领域适应的模糊边界问题。

### 主要发现

领域变化可以通过频谱分析更好地理解，其中低频成分通常编码领域不变的全局模式，高频成分捕获领域特定的局部细节。通过分解图的不同频率成分，可以更有效地进行领域适应。

### 结论

FracNet通过频谱分析和对比学习显著提高了领域适应的性能，实验证明其优于最先进的方法。研究不仅提供了实际应用价值，还提供了严格的理论证明来证明FracNet的优越性。

### 翻译

图神经网络在各种领域取得了显著成功，但由于结构分布的显著变化和可转移模式探索不足，它们在领域适应方面常常遇到困难。传统方法没有区分处理全局和局部模式，导致多层GNN后图中的一些局部细节可能被破坏。我们的关键见解是，领域变化可以通过频谱分析更好地理解，其中低频成分通常编码领域不变的全局模式，高频成分捕获领域特定的局部细节。因此，我们提出FracNet（频率感知对比图网络），包含两个协同模块，将原始图分解为高频和低频成分，并进行频率感知的领域适应。此外，通过与对比学习框架集成，改善了领域适应的模糊边界问题。除了实际应用意义外，我们还提供了严格的理论证明来证明FracNet的优越性。大量实验进一步证明了其优于最先进方法的显著改进。


### 论文摘要

Graph neural networks (GNNs) have achieved remarkable success in various domains, yet they often struggle with domain adaptation due to significant structural distribution shifts and insufficient exploration of transferable patterns. One of the main reasons behind this is that traditional approaches do not treat global and local patterns discriminatingly so that some local details in the graph may be violated after multi-layer GNN. Our key insight is that domain shifts can be better understood through spectral analysis, where low-frequency components often encode domain-invariant global patterns, and high-frequency components capture domain-specific local details. As such, we propose FracNet (\underline{\textbf{Fr}}equency \underline{\textbf{A}}ware \underline{\textbf{C}}ontrastive Graph \underline{\textbf{Net}}work) with two synergic modules to decompose the original graph into high-frequency and low-frequency components and perform frequency-aware domain adaption. Moreover, the blurring boundary problem of domain adaptation is improved by integrating with a contrastive learning framework. Besides the practical implication, we also provide rigorous theoretical proof to demonstrate the superiority of FracNet. Extensive experiments further demonstrate significant improvements over state-of-the-art approaches.

---

## 32. Universally Invariant Learning in Equivariant GNNs

**论文链接:** [http://arxiv.org/abs/2510.13169v1](http://arxiv.org/abs/2510.13169v1)

**作者:** Jiacheng Cen, Anyi Li, Ning Lin, Tingyang Xu, Yu Rong, Deli Zhao, Zihe Wang, Wenbing Huang

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种理论上健全的框架，用于构建高效且实用的完备等变图神经网络(GNNs)，通过两个关键组件实现：完备的标量函数和满秩的可转向基集。

### 背景

等变图神经网络在各种应用中显示出显著成功。为了实现完备性（即在等变函数空间上的通用逼近性质），网络必须有效捕捉不同节点之间复杂的多体相互作用。

### 目的

提出一个理论上健全的框架，用于构建高效且实用的完备等变GNN，解决现有方法计算成本高且没有多项式时间解决方案的问题。

### 方法

证明完备的等变GNN可以通过两个关键组件实现：1) 完备的标量函数，称为几何图的规范形式；2) 满秩的可转向基集。基于这一发现，提出了基于EGNN和TFN两种常见模型的高效算法。

### 主要发现

实证结果表明，该模型在仅有几层的情况下展现出优越的完备性和出色的性能，从而显著降低了计算开销，同时保持强大的实际效能。

### 结论

所提出的框架为构建高效且实用的完备等变GNN提供了理论基础，解决了现有方法的计算效率问题。

### 翻译

等变图神经网络在各种应用中已显示出显著成功。为了实现完备性——即在等变函数空间上的通用逼近性质——网络必须有效捕捉不同节点之间复杂的多体相互作用。现有方法通过更深层次的结构、增强的体阶数或增加可转向特征的维度来实现这一目标，通常计算成本高且没有多项式时间解决方案。在这项工作中，我们提出了一个理论上健全的框架，用于构建高效且实用的完备等变GNN。我们证明，完备的等变GNN可以通过两个关键组件实现：1) 完备的标量函数，称为几何图的规范形式；2) 满秩的可转向基集。利用这一发现，我们提出了基于两种常见模型(EGNN和TFN)的构建完备等变GNN的高效算法。实证结果表明，我们的模型在仅有几层的情况下展现出优越的完备性和出色的性能，从而显著降低了计算开销，同时保持强大的实际效能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决等变图神经网络(GNN)的完备性问题，即模型能否近似任何连续函数的能力。这个问题在科学计算和物理模拟中至关重要，因为它决定了模型能否准确捕捉复杂几何数据（如分子结构、蛋白质等）中的多体相互作用。现有方法需要通过增加网络深度、提高阶数或增加特征维度来获得更好的表达能力，但这会导致计算成本大幅增加，限制了等变GNN在实际应用中的使用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先重新审视了现有等变GNN模型（如EGNN、TFN、MACE等），将它们统一为基于多体高阶基函数扩展的形式，从而识别出当前方法的局限性。然后，作者从输出空间角度提出新的动态方法，借鉴了几何同构问题的研究成果，提出完全等变GNN需要两个关键组件：几何图的规范形式和满秩基集。作者还借鉴了FastEGNN等工作中关于虚拟节点学习的思想，并基于四点定位原理提出了多项式时间算法来构建规范形式。此外，作者借鉴了不对称图上的着色理论，证明了在非对称图上总能构建满秩基集。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过两个关键组件构建完全等变GNN：几何图的规范形式（完全标量函数）和满秩的可引导基集，而不需要通过增加网络深度、阶数或特征维度。实现流程包括：1) 构建几何图的规范形式，对于一般图使用四点定位原理（O(N^6)复杂度），对于非对称图使用E(3)等变函数生成参考点（O(N^2)复杂度）；2) 构建满秩基集，通过节点着色（⊕或⊗方法）使每个节点具有唯一特征；3) 实际模型实现，如EGNNcpl和TFNcpl，通过着色、构建虚拟节点、更新特征和全局操作来实现单层完备模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出新的完备性框架，将完全等变GNN构建转化为规范形式和满秩基集两个组件；2) 提出几何同构问题的多项式时间算法，为一般图提供O(N^6)算法，为非对称图提供O(N^2)算法；3) 证明在非对称几何图上总能构建任意度数的满秩基集；4) 提出EGNN/TFNcpl-global和EGNN/TFNcpl-local两种实际实现。相比之前的工作，传统方法需要通过增加阶数、层数或特征维度来实现完备性，计算成本高且无法保证在有限复杂度下实现完备性，而本文方法通过动态方法在保持低计算复杂度的同时实现了完备性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种名为Uni-EGNN的高效框架，通过几何图的规范形式和满秩基集两个关键组件，使等变图神经网络在保持低计算复杂度的同时实现了完备性，显著提升了模型在科学计算和物理模拟任务中的性能。'}


### 论文摘要

Equivariant Graph Neural Networks (GNNs) have demonstrated significant success across various applications. To achieve completeness -- that is, the universal approximation property over the space of equivariant functions -- the network must effectively capture the intricate multi-body interactions among different nodes. Prior methods attain this via deeper architectures, augmented body orders, or increased degrees of steerable features, often at high computational cost and without polynomial-time solutions. In this work, we present a theoretically grounded framework for constructing complete equivariant GNNs that is both efficient and practical. We prove that a complete equivariant GNN can be achieved through two key components: 1) a complete scalar function, referred to as the canonical form of the geometric graph; and 2) a full-rank steerable basis set. Leveraging this finding, we propose an efficient algorithm for constructing complete equivariant GNNs based on two common models: EGNN and TFN. Empirical results demonstrate that our model demonstrates superior completeness and excellent performance with only a few layers, thereby significantly reducing computational overhead while maintaining strong practical efficacy.

---

## 33. Post-hoc Popularity Bias Correction in GNN-based Collaborative Filtering

**论文链接:** [http://arxiv.org/abs/2510.12959v1](http://arxiv.org/abs/2510.12959v1)

**作者:** Md Aminul Islam, Elena Zheleva, Ren Wang

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了一种后验流行度去偏置(PPD)方法，用于纠正基于图神经网络的协同过滤中的流行度偏见问题。

### 背景

用户历史交互数据是协同过滤中学习用户偏好的主要信号，但训练数据通常呈现长尾分布，导致模型学习流行度偏见，降低推荐质量。图神经网络虽然有效，但其聚合过程会进一步传播和放大这种偏见。

### 目的

开发一种直接对抗GNN邻域聚合过程中传播的流行度偏见的方法，提高推荐的个性化程度和质量。

### 方法

提出PPD方法，该方法在预训练嵌入上操作，不需要重新训练。通过估计交互级别的流行度并使用流行度方向向量从节点表示中移除流行度组件，从而减少偏见同时保留用户偏好。

### 主要发现

现有方法通过修改训练目标解决偏见问题，但无法直接对抗GNN邻域聚合过程中的偏见传播；在聚合过程中对交互应用权重可能缓解问题，但由于训练早期节点表示不稳定，可能导致模型学习扭曲。

### 结论

实验结果表明，PPD方法在基于GNN的CF的流行度偏见纠正方面优于现有最先进的方法。

### 翻译

用户历史交互数据是协同过滤中学习用户偏好的主要信号。然而，训练数据通常呈现长尾分布，只有少数项目拥有大部分交互。直接在这种不平衡数据上训练的CF模型容易学习流行度偏见，降低个性化程度，导致次优的推荐质量。图神经网络(GNN)由于其消息传递机制对CF有效，但可以通过聚合过程进一步传播和放大流行度偏见。现有方法通常通过修改训练目标来解决流行度偏见，但未能直接对抗GNN在邻域聚合过程中传播的偏见。在聚合过程中对交互应用权重可以帮助缓解此问题，但由于训练早期阶段节点表示不稳定，可能会扭曲模型学习。在本文中，我们提出了一种后验流行度去偏置(PPD)方法，用于纠正基于GNN的CF中的流行度偏见，并直接在预训练嵌入上操作，无需重新训练。通过估计交互级别的流行度并使用流行度方向向量从节点表示中移除流行度组件，PPD减少了偏见同时保留了用户偏好。实验结果表明，我们的方法在基于GNN的CF的流行度偏见纠正方面优于最先进的方法。


### 论文摘要

User historical interaction data is the primary signal for learning user preferences in collaborative filtering (CF). However, the training data often exhibits a long-tailed distribution, where only a few items have the majority of interactions. CF models trained directly on such imbalanced data are prone to learning popularity bias, which reduces personalization and leads to suboptimal recommendation quality. Graph Neural Networks (GNNs), while effective for CF due to their message passing mechanism, can further propagate and amplify popularity bias through their aggregation process. Existing approaches typically address popularity bias by modifying training objectives but fail to directly counteract the bias propagated during GNN's neighborhood aggregation. Applying weights to interactions during aggregation can help alleviate this problem, yet it risks distorting model learning due to unstable node representations in the early stages of training. In this paper, we propose a Post-hoc Popularity Debiasing (PPD) method that corrects for popularity bias in GNN-based CF and operates directly on pre-trained embeddings without requiring retraining. By estimating interaction-level popularity and removing popularity components from node representations via a popularity direction vector, PPD reduces bias while preserving user preferences. Experimental results show that our method outperforms state-of-the-art approaches for popularity bias correction in GNN-based CF.

---

## 34. Leveraging Teleconnections with Physics-Informed Graph Attention Networks for Long-Range Extreme Rainfall Forecasting in Thailand

**论文链接:** [http://arxiv.org/abs/2510.12328v2](http://arxiv.org/abs/2510.12328v2)

**作者:** Kiattikun Chobtham, Kanoksri Sarinnapakorn, Kritanai Torsri, Prattana Deeprasertkul, Jirawan Kamma

**发布时间:** 2025-10-14

### GPT解析

### 总结

该研究提出了一种创新的物理信息图神经网络方法，结合极值分析技术，有效提高了泰国地区降雨预测的准确性，特别是对极端事件的预测能力。

### 背景

准确的降雨预报，尤其是极端事件的预报，在气候学和地球系统中仍然是一个重大挑战。泰国地区的站点降雨预测面临特殊挑战。

### 目的

开发结合物理信息的图神经网络和极值分析技术，改进泰国地区的站点降雨预测，特别是提高极端事件的预测准确性。

### 方法

使用图结构表示站点捕捉时空模式；预处理影响区域降雨的气候指标；提出Attention-LSTM模型，利用地形降水物理公式推导边特征；采用空间季节感知广义帕累托分布方法进行阈值超限映射，解决极值预测问题。

### 主要发现

该方法在大多数地区优于成熟的基线模型，包括易发生极端事件的区域；与最先进方法保持竞争力；相比SEAS5业务预报系统，显著改进了极端事件预测；提供高分辨率地图支持长期水资源管理决策。

### 结论

该方法在实际应用中提高了极端事件的预测能力，为长期水资源管理中的决策提供了实用增强。

### 翻译

准确的降雨预报，尤其是极端事件的预报，在气候学和地球系统中仍然是一个重大挑战。本文提出了新颖的物理信息图神经网络结合极值分析技术，以改进泰国地区的站点降雨预测。该模型利用站点图的图结构表示来捕捉复杂的时空模式，并通过遥相关提供可解释性。我们预处理了可能影响区域降雨的相关气候指标。提出的带有长短期记忆的图注意力网络使用简单地形降水物理公式推导的初始边特征应用注意力机制。嵌入随后由LSTM层处理。为解决极值问题，我们使用新颖的空间季节感知广义帕累托分布方法进行阈值超限映射，这克服了传统机器学习模型的局限性。实验表明，我们的方法在大多数地区都优于成熟的基线模型，包括易发生极端事件的区域，并且与最先进的方法保持强劲竞争力。与业务预报系统SEAS5相比，我们的实际应用改进了极端事件的预测，并提供了实用增强，以支持长期水资源管理中的决策。


### 论文摘要

Accurate rainfall forecasting, particularly for extreme events, remains a significant challenge in climatology and the Earth system. This paper presents novel physics-informed Graph Neural Networks (GNNs) combined with extreme-value analysis techniques to improve gauge-station rainfall predictions across Thailand. The model leverages a graph-structured representation of gauge stations to capture complex spatiotemporal patterns, and it offers explainability through teleconnections. We preprocess relevant climate indices that potentially influence regional rainfall. The proposed Graph Attention Network with Long Short-Term Memory (Attention-LSTM) applies the attention mechanism using initial edge features derived from simple orographic-precipitation physics formulation. The embeddings are subsequently processed by LSTM layers. To address extremes, we perform Peak-Over-Threshold (POT) mapping using the novel Spatial Season-aware Generalized Pareto Distribution (GPD) method, which overcomes limitations of traditional machine-learning models. Experiments demonstrate that our method outperforms well-established baselines across most regions, including areas prone to extremes, and remains strongly competitive with the state of the art. Compared with the operational forecasting system SEAS5, our real-world application improves extreme-event prediction and offers a practical enhancement to produce fine-resolution maps that support decision-making in long-term water management.

---

## 35. Cyclic Self-Supervised Diffusion for Ultra Low-field to High-field MRI Synthesis

**论文链接:** [http://arxiv.org/abs/2510.13735v1](http://arxiv.org/abs/2510.13735v1)

**作者:** Zhenxuan Zhang, Peiyuan Jing, Zi Wang, Ula Briski, Coraline Beitone, Yue Yang, Yinzhe Wu, Fanwen Wang, Liutao Yang, Jiahao Huang, Zhifan Gao, Zhaolin Chen, Kh Tohidul Islam, Guang Yang, Peter J. Lally

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种名为循环自监督扩散(CSS-Diff)的框架，用于从低场MRI合成高质量高场MRI图像，解决了现有方法中存在的临床保真度差距问题。

### 背景

低场MRI具有成本低、可及性高、安全性好等优点，但存在分辨率低和信噪比差的问题。从低场MRI合成高质量图像可以减少对昂贵采集的依赖并扩大数据可用性。

### 目的

解决从低场MRI合成高场MRI时存在的临床保真度差距，保留解剖保真度，增强细粒度结构细节，弥合图像对比度中的域差距。

### 方法

提出循环自监督扩散(CSS-Diff)框架，在循环一致性约束下重新制定基于扩散的合成过程，强制在整个生成过程中保持解剖结构。框架包含两个新过程：切片级差距感知网络通过对比学习对齐切片间不一致性；局部结构校正网络通过掩码和扰动块的自重建增强局部特征恢复。

### 主要发现

在跨场合成任务上取得了最先进的性能，包括PSNR、SSIM和LPIPS指标的提升。与原始低场MRI相比，保留了细粒度解剖结构，左脑白质误差从12.1%降至2.1%，皮层从4.2%降至3.7%。

### 结论

CSS-Diff可以合成既定量可靠又解剖一致的图像。

### 翻译

从低场MRI合成高质量图像具有巨大潜力。低场MRI更便宜、更易获取且更安全，但分辨率低且信噪比差。这种合成过程可以减少对昂贵采集的依赖并扩大数据可用性。然而，从低场MRI合成高场MRI仍存在临床保真度差距。需要保留解剖保真度，增强细粒度结构细节，并弥合图像对比度中的域差距。为解决这些问题，我们提出了一个用于从真实低场MRI数据合成高场MRI的循环自监督扩散(CSS-Diff)框架。我们的核心思想是在循环一致性约束下重新制定基于扩散的合成过程。它在整个生成过程中强制保持解剖结构，而不仅仅依赖成对的像素级监督。CSS-Diff框架还进一步整合了两个新过程：切片级差距感知网络通过对比学习对齐切片间不一致性；局部结构校正网络通过掩码和扰动块的自重建增强局部特征恢复。在跨场合成任务上的广泛实验证明了我们方法的有效性，取得了最先进的性能。除了像素级保真度外，与原始低场MRI相比，我们的方法还保留了细粒度解剖结构。总之，我们的CSS-Diff可以合成既定量可靠又解剖一致的图像。


### 论文摘要

Synthesizing high-quality images from low-field MRI holds significant potential. Low-field MRI is cheaper, more accessible, and safer, but suffers from low resolution and poor signal-to-noise ratio. This synthesis process can reduce reliance on costly acquisitions and expand data availability. However, synthesizing high-field MRI still suffers from a clinical fidelity gap. There is a need to preserve anatomical fidelity, enhance fine-grained structural details, and bridge domain gaps in image contrast. To address these issues, we propose a \emph{cyclic self-supervised diffusion (CSS-Diff)} framework for high-field MRI synthesis from real low-field MRI data. Our core idea is to reformulate diffusion-based synthesis under a cycle-consistent constraint. It enforces anatomical preservation throughout the generative process rather than just relying on paired pixel-level supervision. The CSS-Diff framework further incorporates two novel processes. The slice-wise gap perception network aligns inter-slice inconsistencies via contrastive learning. The local structure correction network enhances local feature restoration through self-reconstruction of masked and perturbed patches. Extensive experiments on cross-field synthesis tasks demonstrate the effectiveness of our method, achieving state-of-the-art performance (e.g., 31.80 $\pm$ 2.70 dB in PSNR, 0.943 $\pm$ 0.102 in SSIM, and 0.0864 $\pm$ 0.0689 in LPIPS). Beyond pixel-wise fidelity, our method also preserves fine-grained anatomical structures compared with the original low-field MRI (e.g., left cerebral white matter error drops from 12.1$\%$ to 2.1$\%$, cortex from 4.2$\%$ to 3.7$\%$). To conclude, our CSS-Diff can synthesize images that are both quantitatively reliable and anatomically consistent.

---

## 36. Seeing and Knowing in the Wild: Open-domain Visual Entity Recognition with Large-scale Knowledge Graphs via Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.13675v1](http://arxiv.org/abs/2510.13675v1)

**作者:** Hongkuan Zhou, Lavdim Halilaj, Sebastian Monka, Stefan Schmid, Yuqicheng Zhu, Jingcheng Wu, Nadeem Nazer, Steffen Staab

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种名为KnowCoL（Knowledge-guided Contrastive Learning）的框架，用于开放域视觉实体识别，通过结合图像和文本描述，利用维基数据的结构化信息，将视觉和文本输入抽象到概念层面，支持零样本实体识别。

### 背景

开放域视觉实体识别旨在识别和链接图像中描绘的实体，与维基数据等庞大且不断变化的真实世界概念集合相关联。与传统分类任务不同，它在开放集条件下运行，大多数目标实体在训练过程中未见，且呈现长尾分布，导致任务具有挑战性。

### 目的

解决开放域视觉实体识别任务中的挑战，包括有限的监督、高视觉歧义性和语义消歧的需要，特别是针对训练过程中未见到的实体。

### 方法

提出KnowCoL框架，将图像和文本描述结合到由维基数据结构化信息支持的共享语义空间中。通过将视觉和文本输入抽象到概念层面，模型利用实体描述、类型层次结构和关系上下文来支持零样本实体识别。

### 主要发现

在OVEN基准测试上评估显示，使用视觉、文本和结构化知识大大提高了准确性，特别是对于稀有和未见实体。最小的模型相比最先进的方法在未见实体上的准确性提高了10.5%，尽管模型尺寸缩小了35倍。

### 结论

KnowCoL框架有效地解决了开放域视觉实体识别中的挑战，特别是在处理稀有和未见实体方面表现出色，同时模型尺寸显著减小。

### 翻译

开放域视觉实体识别旨在识别和链接图像中描绘的实体，与维基数据等庞大且不断变化的真实世界概念集合相关联。与具有固定标签集的传统分类任务不同，它在开放集条件下运行，大多数目标实体在训练过程中未见，且呈现长尾分布。这使任务本身具有挑战性，因为监督有限、视觉歧义度高，且需要语义消歧。在这项工作中，我们提出了一个知识引导的对比学习框架，将图像和文本描述结合到由维基数据结构化信息支持的共享语义空间中。通过将视觉和文本输入抽象到概念层面，模型利用实体描述、类型层次结构和关系上下文来支持零样本实体识别。我们在OVEN基准上评估了我们的方法，OVEN是一个大规模开放域视觉识别数据集，以维基数据ID作为标签空间。我们的实验表明，使用视觉、文本和结构化知识大大提高了准确性，特别是对于稀有和未见实体。与最先进的方法相比，我们的最小模型在未见实体上的准确性提高了10.5%，尽管模型尺寸缩小了35倍。


### 论文摘要

Open-domain visual entity recognition aims to identify and link entities depicted in images to a vast and evolving set of real-world concepts, such as those found in Wikidata. Unlike conventional classification tasks with fixed label sets, it operates under open-set conditions, where most target entities are unseen during training and exhibit long-tail distributions. This makes the task inherently challenging due to limited supervision, high visual ambiguity, and the need for semantic disambiguation. In this work, we propose a Knowledge-guided Contrastive Learning (KnowCoL) framework that combines both images and text descriptions into a shared semantic space grounded by structured information from Wikidata. By abstracting visual and textual inputs to a conceptual level, the model leverages entity descriptions, type hierarchies, and relational context to support zero-shot entity recognition. We evaluate our approach on the OVEN benchmark, a large-scale open-domain visual recognition dataset with Wikidata IDs as the label space. Our experiments show that using visual, textual, and structured knowledge greatly improves accuracy, especially for rare and unseen entities. Our smallest model improves the accuracy on unseen entities by 10.5% compared to the state-of-the-art, despite being 35 times smaller.

---

## 37. Contrastive Learning-Based Dependency Modeling for Anomaly Detection in Cloud Services

**论文链接:** [http://arxiv.org/abs/2510.13368v1](http://arxiv.org/abs/2510.13368v1)

**作者:** Yue Xing, Yingnan Deng, Heyao Liu, Ming Wang, Yun Zi, Xiaoxuan Sun

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种结合对比学习的依赖建模和异常检测方法，解决了云服务环境中复杂依赖关系和多样化异常模式的挑战

### 背景

云服务环境中存在复杂依赖关系和多样化异常模式的挑战

### 目的

提出一种结合对比学习的依赖建模和异常检测方法，解决云服务环境中的异常检测问题

### 方法

将服务交互抽象为依赖图，通过嵌入函数提取时间和结构特征，使用图卷积机制聚合邻域信息实现上下文感知的服务表示，引入对比学习框架构建正负样本对增强正常和异常模式可分性，设计时间一致性约束保持表示稳定性，结合对比损失和时间一致性损失进行整体优化

### 主要发现

在公共数据集上从超参数、环境和数据敏感性角度系统评估了该方法，在精确率、召回率、F1分数和AUC等关键指标上显著优于现有方法，在稀疏标记、监控噪声和流量波动条件下保持鲁棒性

### 结论

验证了将依赖建模与对比学习结合的有效性，为云服务异常检测提供了完整的技术解决方案，在复杂环境中表现出强大的适应性和稳定性

### 翻译

本文通过提出一种结合对比学习的依赖建模和异常检测方法，解决了云服务环境中复杂依赖关系和多样化异常模式的挑战。该方法将服务交互抽象为依赖图，通过嵌入函数提取时间和结构特征，并采用图卷积机制聚合邻域信息以实现上下文感知的服务表示。随后引入对比学习框架，构建正负样本对以增强正常和异常模式在表示空间中的可分性。此外，设计了时间一致性约束以保持跨时间步的表示稳定性，减少短期波动和噪声的影响。整体优化结合了对比损失和时间一致性损失，确保在多维度特征下的稳定可靠检测。在公共数据集上从超参数、环境和数据敏感性角度对该方法进行了系统评估。结果表明，在精确率、召回率、F1分数和AUC等关键指标上，所提出的方法显著优于现有方法，同时在稀疏标记、监控噪声和流量波动条件下保持鲁棒性。本研究验证了结合依赖建模与对比学习的有效性，为云服务异常检测提供了完整的技术解决方案，并在复杂环境中表现出强大的适应性和稳定性。


### 论文摘要

This paper addresses the challenges of complex dependencies and diverse anomaly patterns in cloud service environments by proposing a dependency modeling and anomaly detection method that integrates contrastive learning. The method abstracts service interactions into a dependency graph, extracts temporal and structural features through embedding functions, and employs a graph convolution mechanism to aggregate neighborhood information for context-aware service representations. A contrastive learning framework is then introduced, constructing positive and negative sample pairs to enhance the separability of normal and abnormal patterns in the representation space. Furthermore, a temporal consistency constraint is designed to maintain representation stability across time steps and reduce the impact of short-term fluctuations and noise. The overall optimization combines contrastive loss and temporal consistency loss to ensure stable and reliable detection across multi-dimensional features. Experiments on public datasets systematically evaluate the method from hyperparameter, environmental, and data sensitivity perspectives. Results show that the proposed approach significantly outperforms existing methods on key metrics such as Precision, Recall, F1-Score, and AUC, while maintaining robustness under conditions of sparse labeling, monitoring noise, and traffic fluctuations. This study verifies the effectiveness of integrating dependency modeling with contrastive learning, provides a complete technical solution for cloud service anomaly detection, and demonstrates strong adaptability and stability in complex environments.

---

## 38. Universal Image Restoration Pre-training via Masked Degradation Classification

**论文链接:** [http://arxiv.org/abs/2510.13282v1](http://arxiv.org/abs/2510.13282v1)

**作者:** JiaKui Hu, Zhengjian Yao, Lujia Jin, Yinghao Chen, Yanye Lu

**发布时间:** 2025-10-15

### GPT解析

### 总结

本研究提出了一种掩码退化分类预训练方法（MaskDCPT），用于图像退化类型分类，从而实现全面的图像恢复预训练。该方法使用图像退化类型作为弱监督，同时利用图像重建增强性能和鲁棒性。MaskDCPT包含一个编码器和两个解码器，分别用于特征提取、退化类型分类和高质量图像重建。该方法结合了掩码图像建模和对比学习的优势，显著提升了CNN和Transformer在图像恢复任务中的性能。

### 背景

传统预训练方法在图像恢复任务中存在局限性，需要一种能够处理多种退化类型的通用图像恢复方法。

### 目的

开发一种能够分类图像退化类型的预训练方法，实现全面的图像恢复预训练，提高模型在通用图像恢复任务中的性能和鲁棒性。

### 方法

提出MaskDCPT方法，使用图像退化类型作为弱监督。构建包含一个编码器和两个解码器的架构：编码器从掩码的低质量输入图像中提取特征；分类解码器使用这些特征识别退化类型；重建解码器重建对应的高质量图像。利用掩码图像建模和对比学习的好处。构建UIR-2.5M数据集，包含250万对恢复样本，覆盖19种退化类型和200多种退化水平。

### 主要发现

MaskDCPT显著提高了CNN和Transformer的性能，在5D全一恢复任务中PSNR至少提高3.77分贝，在真实退化场景中PIQE减少34.8%。模型对未见过的退化类型和水平表现出强大的泛化能力。

### 结论

MaskDCPT是一种简单而强大的预训练方法，可用于通用图像恢复，能够处理多种退化类型并在各种场景中表现出色。发布的UIR-2.5M数据集、源代码和模型可供社区使用。

### 翻译

本研究介绍了一种掩码退化分类预训练方法（MaskDCPT），旨在促进输入图像中退化类型的分类，从而实现全面的图像恢复预训练。与传统预训练方法不同，MaskDCPT使用图像的退化类型作为极弱监督，同时利用图像重建来增强性能和鲁棒性。MaskDCPT包含一个编码器和两个解码器：编码器从掩码的低质量输入图像中提取特征；分类解码器使用这些特征识别退化类型，而重建解码器旨在重建相应的高质量图像。这种设计使预训练能够受益于掩码图像建模和对比学习，从而生成适合恢复任务的通用表示。得益于简单而强大的MaskDCPT，预训练的编码器可用于解决通用图像恢复并取得卓越性能。实施MaskDCPT显著提高了卷积神经网络（CNN）和Transformer的性能，在5D全一恢复任务中PSNR最小提高3.77分贝，在真实退化场景中与基线相比PIQE减少34.8%。它还对以前未见过的退化类型和水平表现出强大的泛化能力。此外，我们整理并发布了UIR-2.5M数据集，包含250万对恢复样本，涵盖19种退化类型和200多种退化水平，包括合成和真实世界数据。该数据集、源代码和模型可在https://github.com/MILab-PKU/MaskDCPT获取。


### 论文摘要

This study introduces a Masked Degradation Classification Pre-Training method (MaskDCPT), designed to facilitate the classification of degradation types in input images, leading to comprehensive image restoration pre-training. Unlike conventional pre-training methods, MaskDCPT uses the degradation type of the image as an extremely weak supervision, while simultaneously leveraging the image reconstruction to enhance performance and robustness. MaskDCPT includes an encoder and two decoders: the encoder extracts features from the masked low-quality input image. The classification decoder uses these features to identify the degradation type, whereas the reconstruction decoder aims to reconstruct a corresponding high-quality image. This design allows the pre-training to benefit from both masked image modeling and contrastive learning, resulting in a generalized representation suited for restoration tasks. Benefit from the straightforward yet potent MaskDCPT, the pre-trained encoder can be used to address universal image restoration and achieve outstanding performance. Implementing MaskDCPT significantly improves performance for both convolution neural networks (CNNs) and Transformers, with a minimum increase in PSNR of 3.77 dB in the 5D all-in-one restoration task and a 34.8% reduction in PIQE compared to baseline in real-world degradation scenarios. It also emergences strong generalization to previously unseen degradation types and levels. In addition, we curate and release the UIR-2.5M dataset, which includes 2.5 million paired restoration samples across 19 degradation types and over 200 degradation levels, incorporating both synthetic and real-world data. The dataset, source code, and models are available at https://github.com/MILab-PKU/MaskDCPT.

---

## 39. MotionBeat: Motion-Aligned Music Representation via Embodied Contrastive Learning and Bar-Equivariant Contact-Aware Encoding

**论文链接:** [http://arxiv.org/abs/2510.13244v1](http://arxiv.org/abs/2510.13244v1)

**作者:** Xuanchen Wang, Heng Wang, Weidong Cai

**发布时间:** 2025-10-15

**备注:** 5 pages, 1 figure. demo page: https://motionbeat2025.github.io/

### GPT解析

### 总结

MotionBeat是一个运动对齐的音乐表示学习框架，通过具身对比损失和结构节奏对齐损失，以及小节等变相旋转和接触引导注意力等创新架构，成功捕捉了音乐的具身维度，在音乐到舞蹈生成和多种音频处理任务中表现优异。

### 背景

音乐既是听觉现象也是具身现象，与人体运动密切相关，但现有音频表示忽略了这种具身维度，限制了捕捉驱动运动的节奏和结构线索的能力。

### 目的

提出MotionBeat框架，用于学习能够捕捉音乐运动特性的音乐表示。

### 方法

采用两个训练目标：具身对比损失(ECL)实现细粒度节奏区分，结构节奏对齐损失(SRAL)确保节奏一致性；架构上引入小节等变相旋转捕捉循环节奏模式，以及接触引导注意力强调与音乐重音同步的运动事件。

### 主要发现

MotionBeat在音乐到舞蹈生成方面优于最先进的音频编码器，并在节拍跟踪、音乐标记、流派和乐器分类、情感识别以及视听检索等任务中有效迁移。

### 结论

MotionBeat框架成功捕捉了音乐的具身维度，提高了音乐表示的质量，在多种音频处理任务中表现出色。

### 翻译

音乐既是听觉现象也是具身现象，与人体运动密切相关，并通过舞蹈自然表达。然而，大多数现有的音频表示忽略了这种具身维度，限制了它们捕捉驱动运动的节奏和结构线索的能力。我们提出了MotionBeat，一个用于运动对齐的音乐表示学习框架。MotionBeat通过两个新提出的目标进行训练：具身对比损失(ECL)，一种具有速度感知和节拍抖动负样本的增强型InfoNCE公式，用于实现细粒度节奏区分；以及结构节奏对齐损失(SRAL)，通过将音乐重音与相应运动事件对齐来确保节奏一致性。在架构上，MotionBeat引入了小节等变相旋转来捕捉循环节奏模式，以及接触引导注意力来强调与音乐重音同步的运动事件。实验表明，MotionBeat在音乐到舞蹈生成方面优于最先进的音频编码器，并有效迁移到节拍跟踪、音乐标记、流派和乐器分类、情感识别以及视听检索等任务。我们的项目演示页面：https://motionbeat2025.github.io/。


### 论文摘要

Music is both an auditory and an embodied phenomenon, closely linked to human motion and naturally expressed through dance. However, most existing audio representations neglect this embodied dimension, limiting their ability to capture rhythmic and structural cues that drive movement. We propose MotionBeat, a framework for motion-aligned music representation learning. MotionBeat is trained with two newly proposed objectives: the Embodied Contrastive Loss (ECL), an enhanced InfoNCE formulation with tempo-aware and beat-jitter negatives to achieve fine-grained rhythmic discrimination, and the Structural Rhythm Alignment Loss (SRAL), which ensures rhythm consistency by aligning music accents with corresponding motion events. Architecturally, MotionBeat introduces bar-equivariant phase rotations to capture cyclic rhythmic patterns and contact-guided attention to emphasize motion events synchronized with musical accents. Experiments show that MotionBeat outperforms state-of-the-art audio encoders in music-to-dance generation and transfers effectively to beat tracking, music tagging, genre and instrument classification, emotion recognition, and audio-visual retrieval. Our project demo page: https://motionbeat2025.github.io/.

---

## 40. GRACE: Globally-Seeded Representation-Aware Cluster-Specific Evolution for Compiler Auto-Tuning

**论文链接:** [http://arxiv.org/abs/2510.13176v1](http://arxiv.org/abs/2510.13176v1)

**作者:** Haolin Pan, Chao Zha, Jinyuan Dong, Mingjie Xing, Yanjun Wu

**发布时间:** 2025-10-15

### GPT解析

### 总结

GRACE是一个创新的编译器自动调优框架，通过利用通道协同性和加权评分方法缩小搜索空间，使用对比学习和相似感知聚类创建程序嵌入，并在聚类内进行进化搜索，生成针对未见程序具有强泛化能力的核心集通道序列。实验表明，GRACE在LLVM IR指令计数优化方面达到了最先进的性能，同时保持了高效的调优时间。

### 背景

编译器通道选择和阶段排序是实现最优程序性能的重大挑战，特别是对于代码大小缩减等目标。标准编译器启发式方法具有通用适用性，但由于其'一刀切'的特性，通常会产生次优的、程序特定的结果。虽然迭代编译可以找到量身定制的解决方案，但其高昂的搜索成本限制了实际应用。机器学习方法承诺更快的推理速度，但经常难以泛化到未见程序。

### 目的

开发一个高效的编译器自动调优框架，能够在保持快速调优时间的同时，为未见程序提供高质量的优化解决方案，特别是在LLVM IR指令计数优化方面。

### 方法

GRACE框架首先利用通道协同性和加权评分方法生成初始高质量候选序列和通道池，有效缩小搜索空间。然后采用对比学习方法，使用基于通道序列的数据增强技术创建程序嵌入，促进相似感知聚类。在这些聚类内进行进化搜索，生成k个专门设计的通道序列核心集，旨在对未见程序实现强泛化能力。在测试时，GRACE高效选择最佳核心集序列并使用轻量级技术进行优化。

### 主要发现

在七个不同的数据集上的实验结果表明，GRACE与opt -Oz相比，在LLVM 10.0.0上将LLVM IR指令计数平均减少了10.09%，在LLVM 18.1.6上平均减少了10.19%，同时每个程序的调优时间平均不到1秒，展示了其最先进的性能和实际有效性。

### 结论

GRACE框架成功解决了编译器自动调优中的搜索空间过大和泛化能力不足的问题，通过结合通道协同性、加权评分、对比学习和进化搜索等技术，实现了在保持高效调优时间的同时，为未见程序提供高质量优化的能力，在LLVM IR指令计数优化方面达到了最先进的性能。

### 翻译

编译器通道选择和阶段排序是实现最优程序性能的重大挑战，特别是对于代码大小缩减等目标。标准编译器启发式方法具有通用适用性，但由于其'一刀切'的特性，通常会产生次优的、程序特定的结果。虽然迭代编译可以找到量身定制的解决方案，但其高昂的搜索成本限制了实际应用。机器学习方法承诺更快的推理速度，但经常难以泛化到未见程序。本文介绍了GRACE，一个用于编译器自动调优的新颖框架，已在LLVM IR指令计数优化中得到验证。GRACE通过利用通道协同性和加权评分方法有效缩小搜索空间，生成初始高质量候选序列和通道池。然后采用对比学习方法，使用基于通道序列的数据增强技术创建程序嵌入，促进相似感知聚类。在这些聚类内进行进化搜索，生成k个专门设计的通道序列核心集，旨在对未见程序实现强泛化能力。在测试时，GRACE高效选择最佳核心集序列并使用轻量级技术进行优化。在七个不同数据集上的实验结果表明，GRACE与opt -Oz相比，在LLVM 10.0.0上将LLVM IR指令计数平均减少了10.09%，在LLVM 18.1.6上平均减少了10.19%，同时每个程序的调优时间平均不到1秒，展示了其最先进的性能和实际有效性。


### 论文摘要

Compiler pass selection and phase ordering present a significant challenge in achieving optimal program performance, particularly for objectives like code size reduction. Standard compiler heuristics offer general applicability but often yield suboptimal, program-specific results due to their one-size-fits-all nature. While iterative compilation can find tailored solutions, its prohibitive search cost limits practical use. Machine learning approaches promise faster inference but frequently struggle with generalization to unseen programs. This paper introduces GRACE, a novel framework for compiler auto-tuning, demonstrated for LLVM IR instruction count optimization. GRACE effectively curtails the search space by leveraging pass synergies and a weighted scoring method to generate initial high-quality candidate sequences and a pass pool. It then employs contrastive learning, using pass sequence-based data augmentation, to create program embeddings that facilitate similarity-aware clustering. Evolutionary search within these clusters yields a coreset of $k$ specialized pass sequences designed for robust generalization to unseen programs. At test time, GRACE efficiently selects the best coreset sequence and refines it using lightweight techniques. Experimental results on seven diverse datasets show that GRACE reduces LLVM IR instruction count by an average of 10.09% on LLVM 10.0.0 and 10.19% on LLVM 18.1.6 compared to opt -Oz, while incurring an average tuning time of less than 1s per program, demonstrating its state-of-the-art performance and practical effectiveness.

---

## 41. VCTR: A Transformer-Based Model for Non-parallel Voice Conversion

**论文链接:** [http://arxiv.org/abs/2510.12964v1](http://arxiv.org/abs/2510.12964v1)

**作者:** Maharnab Saikia

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了一种名为VCTR的高效非并行语音转换方法，结合了混合感知块和双剪枝自注意力机制，采用基于对比学习的对抗方法，解决了现有方法中存在的长距离依赖捕获不足的问题。

### 背景

非并行语音转换技术旨在无需配对训练数据的情况下将源语音域转换为目标语音域。现有的CycleGAN、VAE和CVC等方法在训练效果和语义捕获方面存在局限性。

### 目的

开发一种能够有效捕获语音中长距离依赖关系的高效非并行语音转换方法，以提升转换质量和全局语义表达能力。

### 方法

提出VCTR方法，结合了Hybrid Perception Block (HPB)和Dual Pruned Self-Attention (DPSA)技术，采用基于对比学习的对抗训练框架，能够更好地捕获语音中的长距离依赖关系。

### 主要发现

基于CNN的生成器虽然能捕获局部语义，但缺乏捕获全局语义所需的长距离依赖能力；所提出的VCTR方法通过结合HPB和DPSA有效解决了这一问题。

### 结论

VCTR是一种高效的非并行语音转换方法，通过创新的网络结构和训练策略，显著提升了语音转换的质量和全局语义表达能力。

### 翻译

非并行语音转换旨在无需配对训练数据的情况下将语音从源域转换到目标域。循环一致性生成对抗网络(CycleGAN)和变分自编码器(VAE)已被用于此任务，但这些模型面临训练困难和结果不理想的问题。后来，对比语音转换(Contrastive Voice Conversion, CVC)被提出，利用基于对比学习的方法解决这些问题。然而，这些方法使用基于CNN的生成器，虽然可以捕获局部语义，但缺乏捕获全局语义所需的长距离依赖能力。在本文中，我们提出了VCTR，一种用于非并行语音转换的高效方法，它结合了混合感知块(HPB)和双剪枝自注意力(DPSA)，以及基于对比学习的对抗方法。代码可在https://github.com/Maharnab-Saikia/VCTR找到。


### 论文摘要

Non-parallel voice conversion aims to convert voice from a source domain to a target domain without paired training data. Cycle-Consistent Generative Adversarial Networks (CycleGAN) and Variational Autoencoders (VAE) have been used for this task, but these models suffer from difficult training and unsatisfactory results. Later, Contrastive Voice Conversion (CVC) was introduced, utilizing a contrastive learning-based approach to address these issues. However, these methods use CNN-based generators, which can capture local semantics but lacks the ability to capture long-range dependencies necessary for global semantics. In this paper, we propose VCTR, an efficient method for non-parallel voice conversion that leverages the Hybrid Perception Block (HPB) and Dual Pruned Self-Attention (DPSA) along with a contrastive learning-based adversarial approach. The code can be found in https://github.com/Maharnab-Saikia/VCTR.

---

## 42. CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation

**论文链接:** [http://arxiv.org/abs/2510.13245v1](http://arxiv.org/abs/2510.13245v1)

**作者:** Li Liang, Bo Miao, Xinyu Wang, Naveed Akhtar, Jordan Vice, Ajmal Mian

**发布时间:** 2025-10-15

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出了SketchSem3D，第一个从抽象手绘草图和卫星图像伪标签注释生成3D户外语义场景的大规模基准数据集，以及Cylinder Mamba Diffusion (CymbaDiff)方法，显著增强了户外3D场景生成的空间连贯性。

### 背景

户外3D语义场景生成技术为城市模拟和自动驾驶等应用提供逼真且语义丰富的环境，但该领域发展受限于缺乏公开可用的、良好注释的数据集。

### 目的

引入SketchSem3D基准数据集，用于从抽象手绘草图和卫星图像的伪标签注释生成3D户外语义场景。

### 方法

SketchSem3D包含两个子集：基于Sketch的SemanticKITTI和基于Sketch的KITTI-360（包含LiDAR体素及其相应的草图和注释卫星图像）。提出Cylinder Mamba Diffusion (CymbaDiff)方法，施加结构化空间排序，捕获圆柱连续性和垂直层次结构，保持物理邻域关系和全局上下文。

### 主要发现

在SketchSem3D上的大量实验表明，CymbaDiff实现了卓越的语义一致性、空间真实性和跨数据集泛化能力。

### 结论

代码和数据集将在https://github.com/Lillian-research-hub/CymbaDiff上提供。

### 翻译

户外3D语义场景生成为城市模拟和自动驾驶等应用生成逼真且语义丰富的环境。然而，这一方向的进展受到缺乏公开可用、良好注释的数据集的限制。我们引入SketchSem3D，这是第一个从抽象手绘草图和卫星图像的伪标签注释生成3D户外语义场景的大规模基准。SketchSem3D包含两个子集：基于Sketch的SemanticKITTI和基于Sketch的KITTI-360（包含LiDAR体素及其相应的草图和注释卫星图像），以实现标准化、严格和多样化的评估。我们还提出了Cylinder Mamba Diffusion (CymbaDiff)，显著增强了户外3D场景生成的空间连贯性。CymbaDiff施加结构化空间排序，明确捕获圆柱连续性和垂直层次结构，并在生成的场景中保持物理邻域关系和全局上下文。在SketchSem3D上的大量实验表明，CymbaDiff实现了卓越的语义一致性、空间真实性和跨数据集泛化能力。代码和数据集将在https://github.com/Lillian-research-hub/CymbaDiff上提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决户外3D语义场景生成的问题，特别是从手绘草图和伪标记卫星图像注释生成3D城市场景。这个问题很重要，因为生成逼真且语义丰富的户外环境对城市模拟和自动驾驶等应用至关重要，但该领域缺乏公开、良好标注的数据集，且现有方法难以处理户外场景的高语义多样性、复杂空间结构和动态上下文依赖。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别到户外3D场景生成的重要性及现有方法的局限性，然后构建了SketchSem3D数据集作为基础。方法设计借鉴了状态空间模型(SSMs)在捕获长程依赖关系方面的优势，以及扩散模型在生成任务中的成功经验。作者创新性地结合了笛卡尔和圆柱坐标系统，设计了圆柱Mamba块(CylMa)来增强空间一致性，同时保留了三重Mamba模块以保持精确几何距离。整体架构包括场景结构估计网络、潜在映射网络和去噪网络，通过多尺度特征提取和维度分解残差块来提升性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结构化空间扩散结合圆柱和笛卡尔坐标系统的优势，增强户外3D场景生成的空间一致性。整体流程包括：1)数据预处理，生成草图和伪标记卫星图像注释；2)使用场景结构估计网络提取抽象结构信息；3)通过变分自编码器将输入条件压缩为潜在表示；4)在潜在空间中使用圆柱Mamba块进行去噪扩散；5)融合三重Mamba和圆柱Mamba的特征，结合径向和轴对齐的空间线索；6)生成最终的3D语义场景，每个体素被分配语义类标签。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)提出'基于草图的3D户外场景生成'新任务；2)构建SketchSem3D数据集，提供更高分辨率(256×256×32)、更多语义类别(20类)和更丰富的地理空间语义；3)设计圆柱Mamba扩散(CymbaDiff)模型；4)通过结构化空间排序捕获圆柱连续性和垂直层次结构；5)利用草图和伪标记卫星图像注释作为多模态条件输入。相比之前工作，CymbaDiff结合了圆柱和笛卡尔坐标系统，更好地表示户外场景结构；通过状态空间模型和扩散模型结合，更高效地捕获长程依赖；将草图生成从孤立对象和简单室内场景扩展到复杂户外场景。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了SketchSem3D数据集和CymbaDiff方法，首次实现了从手绘草图和伪标记卫星图像注释生成高质量、空间一致的3D户外语义场景，为城市模拟和自动驾驶等应用提供了新的解决方案。'}


### 论文摘要

Outdoor 3D semantic scene generation produces realistic and semantically rich environments for applications such as urban simulation and autonomous driving. However, advances in this direction are constrained by the absence of publicly available, well-annotated datasets. We introduce SketchSem3D, the first large-scale benchmark for generating 3D outdoor semantic scenes from abstract freehand sketches and pseudo-labeled annotations of satellite images. SketchSem3D includes two subsets, Sketch-based SemanticKITTI and Sketch-based KITTI-360 (containing LiDAR voxels along with their corresponding sketches and annotated satellite images), to enable standardized, rigorous, and diverse evaluations. We also propose Cylinder Mamba Diffusion (CymbaDiff) that significantly enhances spatial coherence in outdoor 3D scene generation. CymbaDiff imposes structured spatial ordering, explicitly captures cylindrical continuity and vertical hierarchy, and preserves both physical neighborhood relationships and global context within the generated scenes. Extensive experiments on SketchSem3D demonstrate that CymbaDiff achieves superior semantic consistency, spatial realism, and cross-dataset generalization. The code and dataset will be available at https://github.com/Lillian-research-hub/CymbaDiff

---

## 43. InteractiveOmni: A Unified Omni-modal Model for Audio-Visual Multi-turn Dialogue

**论文链接:** [http://arxiv.org/abs/2510.13747v1](http://arxiv.org/abs/2510.13747v1)

**作者:** Wenwen Tong, Hewei Guo, Dongchuan Ran, Jiangnan Chen, Jiefan Lu, Kaibin Wang, Keqiang Li, Xiaoxu Zhu, Jiakui Li, Kehan Li, Xueheng Li, Lumin Li, Chenxu Guo, Jiasheng Zhou, Jiandong Chen, Xianye Wu, Jiahao Wang, Silei Wu, Lei Chen, Hanming Deng, Yuxuan Song, Dinghao Zhou, Guiping Zhong, Ken Zheng, Shiyin Kang, Lewei Lu

**发布时间:** 2025-10-15

### GPT解析

### 总结

InteractiveOmni是一个统一的开源多模态大语言模型，参数规模从4B到8B，专注于音频视觉多轮交互，整合了视觉编码器、音频编码器、大语言模型和语音解码器，采用多阶段训练策略，具有强大的跨模态能力和类人长期对话能力。

### 背景

轻量级模型领域需要全面的多模态理解和语音生成能力，现有模型可能在这一方面存在不足。

### 目的

开发一个统一的、开源的多模态大语言模型，引领轻量级模型领域，提供全面的多模态理解和语音生成能力。

### 方法

将视觉编码器、音频编码器、大语言模型和语音解码器整合到统一模型中；设计多阶段训练策略：预训练用于多模态理解，然后进行语音对话和视听交互的后训练；精心策划多轮训练数据集，增强处理复杂多轮交互的能力；构建多模态多轮记忆基准和多轮语音交互基准，用于评估多轮记忆和语音交互能力。

### 主要发现

InteractiveOmni显著优于领先的开源模型；InteractiveOmni-4B在通用基准上可与更大的Qwen2.5-Omni-7B模型相媲美；InteractiveOmni-4B仅使用50%的模型大小就能保留InteractiveOmni-8B 97%的性能；在图像、音频、视频理解和语音生成任务上，与同等规模的模型相比取得了最先进的结果。

### 结论

InteractiveOmni是下一代智能交互系统的可访问开源基础模型，提供了更智能的多轮音频视觉体验，特别是在长期记忆能力方面表现出色。

### 翻译

我们介绍了InteractiveOmni，这是一个统一的开源多模态大语言模型，参数规模从4B到8B，专为音频视觉多轮交互设计，通过提供全面的多模态理解和语音生成能力引领轻量级模型领域。为此，我们将视觉编码器、音频编码器、大语言模型和语音解码器整合到一个统一模型中，用于理解和生成任务。我们设计了一个多阶段训练策略，以确保强大的跨模态能力，包括预训练用于多模态理解，随后进行语音对话和视听交互的后训练。为了实现类人长期对话能力，我们精心策划了一个多轮训练数据集，增强模型处理复杂多轮交互的能力。为了有效评估多轮记忆和语音交互能力，我们构建了多模态多轮记忆基准和多轮语音交互基准。实验证明，InteractiveOmni显著优于领先的开源模型，提供了更智能的多轮音频视觉体验，特别是在其长期记忆能力方面。值得注意的是，InteractiveOmni-4B在通用基准上可与大得多的Qwen2.5-Omni-7B模型相媲美，同时仅利用50%的模型大小就能保留InteractiveOmni-8B 97%的性能。在图像、音频、视频理解和语音生成任务上，与同等规模的模型相比取得了最先进的结果，InteractiveOmni是下一代智能交互系统的可访问开源基础。


### 论文摘要

We introduce InteractiveOmni, a unified and open-source omni-modal large language model for audio-visual multi-turn interaction, ranging from 4B to 8B parameters, designed to lead the field of lightweight models by offering comprehensive omni-modal understanding and speech generation capabilities. To achieve this, we integrate the vision encoder, audio encoder, large language model, and speech decoder into a unified model for understanding and generation tasks. We design a multi-stage training strategy to ensure robust cross-modal capabilities, including pre-training for omni-modal understanding, followed by post-training with speech conversation and audio-visual interaction. To enable human-like long-term conversational ability, we meticulously curate a multi-turn training dataset that enhances the model's ability to handle complex and multi-turn interactions. To effectively evaluate the multi-turn memory and speech interaction capabilities, we construct the multi-modal multi-turn memory benchmark and the multi-turn speech interaction benchmark. Experiments demonstrate that InteractiveOmni significantly outperforms leading open-source models and provides a more intelligent multi-turn audio-visual experience, particularly in its long-term memory capabilities. Notably, InteractiveOmni-4B is comparable to the much larger model like Qwen2.5-Omni-7B on general benchmarks, and it can retain 97% of the performance of the InteractiveOmni-8B while utilizing only 50% of the model size. Achieving state-of-the-art results against similarly sized models across image, audio, video understanding, and speech generation tasks, InteractiveOmni is an accessible, open-source foundation for next-generation intelligent interactive systems.

---

## 44. Hierarchical Bayesian Modeling of Dengue in Recife, Brazil (2015-2024): The Role of Spatial Granularity and Data Quality for Epidemiological Risk Mapping

**论文链接:** [http://arxiv.org/abs/2510.13672v1](http://arxiv.org/abs/2510.13672v1)

**作者:** Marcílio Ferreira dos Santos, Andreza dos Santos Rodrigues de Melo

**发布时间:** 2025-10-15

**备注:** 12 pages, 12 figures, 8 tables

### GPT解析

### 总结

该研究使用贝叶斯分层时空模型分析了巴西累西腓市2015-2024年的登革热病例，探讨了多种社会环境和气候因素对登革热风险的影响，并识别了高风险区域。

### 背景

登革热是巴西主要的流行病学挑战之一，表现为城市内部的不平等以及气候和社会环境因素的影响。

### 目的

分析2015-2024年累西腓市的登革热确诊病例，评估多种因素对登革热风险的影响。

### 方法

使用R-INLA实现的贝叶斯分层时空模型，结合BYM2空间结构和RW1时间成分。纳入的协变量包括人口密度、家庭规模、收入、排水渠道、滞后降水量和平均温度。

### 主要发现

人口密度和家庭规模增加登革热风险；收入和排水渠道具有保护作用；滞后降水量增加风险；高温显示反向关联，表明媒介活动的热阈值；模型拟合良好，收敛稳定；北部和西部存在持续高风险集群，与高密度和社会脆弱性区域重叠。

### 结论

贝叶斯模型支持概率预测和早期预警系统。与经典模型相比，INLA明确整合了不确定性和时空依赖性，为城市健康管理决策提供了可信区间推断。

### 翻译

登革热仍然是巴西主要的流行病学挑战之一，表现为城市内部的不平等以及气候和社会环境因素的影响。本研究使用R-INLA实现的贝叶斯分层时空模型分析了2015-2024年累西腓市的登革热确诊病例，结合了BYM2空间结构和RW1时间成分。协变量包括人口密度、家庭规模、收入、排水渠道、滞后降水量和平均温度。人口密度和家庭规模对登革热风险有正向影响，而收入和渠道存在具有保护作用。滞后降水量增加风险，较高温度显示反向关联，表明媒介活动的热阈值。模型拟合良好，收敛稳定，具有中等程度的残差空间自相关和2016-2019年间平滑的时间趋势。空间时间估计显示累西腓北部和西部持续存在高风险集群，与较高密度和社会脆弱性区域重叠。除了重现历史模式外，贝叶斯模型还支持概率预测和早期预警系统。与经典模型相比，INLA明确整合了不确定性和时空依赖性，为城市健康管理决策提供了可信区间推断。


### 论文摘要

Dengue remains one of Brazil's major epidemiological challenges, marked by strong intra-urban inequalities and the influence of climatic and socio-environmental factors. This study analyzed confirmed dengue cases in Recife from 2015 to 2024 using a Bayesian hierarchical spatio-temporal model implemented in R-INLA, combining a BYM2 spatial structure with an RW1 temporal component. Covariates included population density, household size, income, drainage channels, lagged precipitation, and mean temperature. Population density and household size had positive effects on dengue risk, while income and channel presence were protective. Lagged precipitation increased risk, and higher temperatures showed an inverse association, suggesting thermal thresholds for vector activity. The model achieved good fit (DIC=65817; WAIC=64506) and stable convergence, with moderate residual spatial autocorrelation (phi=0.06) and a smooth temporal trend between 2016 and 2019. Spatio-temporal estimates revealed persistent high-risk clusters in northern and western Recife, overlapping with areas of higher density and social vulnerability. Beyond reproducing historical patterns, the Bayesian model supports probabilistic forecasting and early warning systems. Compared with classical models (GLM, SAR, GWR, GTWR), INLA explicitly integrates uncertainty and spatial-temporal dependence, offering credible interval inference for decision-making in urban health management.

---

## 45. MemoTime: Memory-Augmented Temporal Knowledge Graph Enhanced Large Language Model Reasoning

**论文链接:** [http://arxiv.org/abs/2510.13614v1](http://arxiv.org/abs/2510.13614v1)

**作者:** Xingyu Tan, Xiaoyang Wang, Xiwei Xu, Xin Yuan, Liming Zhu, Wenjie Zhang

**发布时间:** 2025-10-15

### GPT解析

### 总结

MemoTime是一种记忆增强的时间知识图谱框架，通过结构化基础、递归推理和持续经验学习解决大型语言模型在时间理解方面的挑战，显著提升了模型在时间问答任务上的性能。

### 背景

大型语言模型已展现出强大的推理能力，但在处理涉及多个实体、复合运算符和演变事件序列的时间理解问题时存在困难。时间知识图谱虽提供了结构化的时间事实，但现有基于TKG的LLM推理方法仍面临四大挑战。

### 目的

解决现有TKG-based LLM推理方法面临的四大挑战：保持多跳推理的时间忠实性、实现多实体时间同步、适应不同时间运算符的检索、重用先验推理经验以提高稳定性和效率。

### 方法

提出MemoTime框架，将复杂时间问题分解为层次化的时间树，实现运算符感知推理；包含动态证据检索层自适应选择策略；以及自我演化的经验记忆存储已验证推理轨迹、工具包决策和子问题嵌入用于跨类型重用。

### 主要发现

在多个时间问答基准上，MemoTime取得了最先进的结果，比强基线模型高出24.0%；使较小模型(如Qwen3-4B)能达到与GPT-4-Turbo相当的推理性能。

### 结论

MemoTime有效解决了现有方法面临的四大挑战，显著提升了大型语言模型在时间理解方面的能力，并使较小模型也能达到高性能水平。

### 翻译

大型语言模型(LLMs)已经取得了令人印象深刻的推理能力，但在时间理解方面存在困难，特别是当问题涉及多个实体、复合运算符和不断演变的事件序列时。时间知识图谱(TKGs)以结构化格式捕获大量时间事实，为时间推理提供了可靠来源。然而，现有的基于TKG的LLM推理方法仍面临四大挑战：在多跳推理中保持时间忠实性、实现多实体时间同步、使检索适应不同的时间运算符、重用先前的推理经验以提高稳定性和效率。为解决这些问题，我们提出了MemoTime，这是一个记忆增强的时间知识图谱框架，通过结构化基础、递归推理和持续经验学习来增强LLM推理。MemoTime将复杂的时间问题分解为层次化的时间树，实现运算符感知推理，强制单调时间戳并在统一时间边界下共同约束多个实体。动态证据检索层自适应地选择特定运算符的检索策略，而自我演化的经验记忆存储已验证的推理轨迹、工具包决策和子问题嵌入用于跨类型重用。在多个时间问答基准上的综合实验显示，MemoTime取得了最先进的结果，比强大的基线模型高出24.0%。此外，MemoTime使较小的模型(如Qwen3-4B)能够实现与GPT-4-Turbo相当的推理性能。


### 论文摘要

Large Language Models (LLMs) have achieved impressive reasoning abilities, but struggle with temporal understanding, especially when questions involve multiple entities, compound operators, and evolving event sequences. Temporal Knowledge Graphs (TKGs), which capture vast amounts of temporal facts in a structured format, offer a reliable source for temporal reasoning. However, existing TKG-based LLM reasoning methods still struggle with four major challenges: maintaining temporal faithfulness in multi-hop reasoning, achieving multi-entity temporal synchronization, adapting retrieval to diverse temporal operators, and reusing prior reasoning experience for stability and efficiency. To address these issues, we propose MemoTime, a memory-augmented temporal knowledge graph framework that enhances LLM reasoning through structured grounding, recursive reasoning, and continual experience learning. MemoTime decomposes complex temporal questions into a hierarchical Tree of Time, enabling operator-aware reasoning that enforces monotonic timestamps and co-constrains multiple entities under unified temporal bounds. A dynamic evidence retrieval layer adaptively selects operator-specific retrieval strategies, while a self-evolving experience memory stores verified reasoning traces, toolkit decisions, and sub-question embeddings for cross-type reuse. Comprehensive experiments on multiple temporal QA benchmarks show that MemoTime achieves overall state-of-the-art results, outperforming the strong baseline by up to 24.0%. Furthermore, MemoTime enables smaller models (e.g., Qwen3-4B) to achieve reasoning performance comparable to that of GPT-4-Turbo.

---

## 46. Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs

**论文链接:** [http://arxiv.org/abs/2510.13251v1](http://arxiv.org/abs/2510.13251v1)

**作者:** Minji Kim, Taekyung Kim, Bohyung Han

**发布时间:** 2025-10-15

**备注:** 23 pages, 28 figures, 8 tables

### GPT解析

### 总结

这项研究探讨了视频大语言模型（VideoLLMs）在视频问答任务中的内部工作机制和信息流动模式。通过可解释性技术分析，研究者发现了VideoLLMs处理视频和文本信息的特定阶段和模式，并展示了如何通过选择有效信息通路来保持模型性能。

### 背景

视频大语言模型（VideoLLMs）将视觉-语言模型的能力扩展到时空输入，使视频问答（VideoQA）等任务成为可能。尽管近期VideoLLMs取得了进展，但它们在提取和传播视频与文本信息方面的内部机制仍较少被探索。

### 目的

研究旨在探究VideoLLMs的内部信息流动机制，特别是它们在视频问答任务中如何进行时序推理以及如何整合视频和文本信息。

### 方法

研究者使用可解释性技术来分析VideoLLMs的内部信息流动模式。

### 主要发现

1. 时序推理从早期到中间层开始，涉及帧间积极交互；2. 随后在中间层进行视频-语言逐步整合，这得益于视频表示与包含时间概念的词嵌入之间的对齐；3. 完成整合后，模型在中间到后期层准备生成正确答案；4. 通过选择有效信息通路并抑制大量注意力边缘（例如在LLaVA-NeXT-7B-Video-FT中为58%），VideoLLMs可以保持其视频问答性能。

### 结论

这些发现为理解VideoLLMs如何执行时序推理提供了蓝图，并为提高模型可解释性和下游泛化能力提供了实用见解。

### 翻译

视频大语言模型（VideoLLMs）将视觉-语言模型的能力扩展到时空输入，使视频问答（VideoQA）等任务成为可能。尽管近期VideoLLMs取得了进展，但它们在提取和传播视频与文本信息方面的内部机制仍较少被探索。在本研究中，我们使用可解释性技术研究了VideoLLMs的内部信息流动。我们的分析揭示了跨不同视频问答任务的一致模式：（1）VideoLLMs中的时序推理从早期到中间层的帧间积极交互开始，（2）随后在中间层进行视频-语言逐步整合。这得益于视频表示与包含时间概念的词嵌入之间的对齐。（3）完成此整合后，模型在中间到后期层准备生成正确答案。（4）基于我们的分析，我们表明VideoLLMs可以通过选择这些有效信息通路同时抑制大量注意力边缘来保持其视频问答性能，例如在LLaVA-NeXT-7B-Video-FT中为58%。这些发现为VideoLLMs如何执行时序推理提供了蓝图，并为提高模型可解释性和下游泛化能力提供了实用见解。我们的项目页面和源代码可在https://map-the-flow.github.io获取。


### 论文摘要

Video Large Language Models (VideoLLMs) extend the capabilities of vision-language models to spatiotemporal inputs, enabling tasks such as video question answering (VideoQA). Despite recent advances in VideoLLMs, their internal mechanisms on where and how they extract and propagate video and textual information remain less explored. In this study, we investigate the internal information flow of VideoLLMs using mechanistic interpretability techniques. Our analysis reveals consistent patterns across diverse VideoQA tasks: (1) temporal reasoning in VideoLLMs initiates with active cross-frame interactions in early-to-middle layers, (2) followed by progressive video-language integration in middle layers. This is facilitated by alignment between video representations and linguistic embeddings containing temporal concepts. (3) Upon completion of this integration, the model is ready to generate correct answers in middle-to-late layers. (4) Based on our analysis, we show that VideoLLMs can retain their VideoQA performance by selecting these effective information pathways while suppressing a substantial amount of attention edges, e.g., 58% in LLaVA-NeXT-7B-Video-FT. These findings provide a blueprint on how VideoLLMs perform temporal reasoning and offer practical insights for improving model interpretability and downstream generalization. Our project page with the source code is available at https://map-the-flow.github.io

---

## 47. Edit-Your-Interest: Efficient Video Editing via Feature Most-Similar Propagation

**论文链接:** [http://arxiv.org/abs/2510.13084v1](http://arxiv.org/abs/2510.13084v1)

**作者:** Yi Zuo, Zitao Wang, Lingling Li, Xu Liu, Fang Liu, Licheng Jiao

**发布时间:** 2025-10-15

**备注:** 32 pages, 11 figures

### GPT解析

### 总结

本文提出了一种名为Edit-Your-Interest的轻量级、文本驱动、零样本视频编辑方法，通过时空特征内存和特征传播技术解决了现有视频编辑方法计算开销大、内存消耗高和视觉保真度低的问题。

### 背景

现有文本到图像扩散模型在视频编辑方面取得了显著进展，但现有视频编辑方法受高计算开销和内存消耗的限制，且往往牺牲视觉保真度，导致时间不一致性和模糊、马赛克状伪影等问题。

### 目的

提出一种轻量级、文本驱动、零样本的视频编辑方法，以提高效率和视觉保真度。

### 方法

Edit-Your-Interest方法包含三个核心技术：1)时空特征内存(SFM)缓存来自先前帧的关键图像标记；2)特征最相似传播(FMP)方法将最相关标记从前一帧传播到后续帧；3)SFM更新算法持续刷新缓存特征。此外，还利用交叉注意图自动提取感兴趣实例的掩码，并将其集成到扩散去噪过程中实现细粒度控制。

### 主要发现

SFM显著减少了计算开销；FMP保留了时间一致性；SFM更新算法确保了特征的长期相关性和有效性；掩码集成方法实现了对目标对象的高度准确编辑，同时保持背景完整性。

### 结论

Edit-Your-Interest在效率和视觉保真度上都优于现有最先进方法，验证了其优越的有效性和实用性。

### 翻译

文本到图像(T2I)扩散模型最近在视频编辑方面展示了显著进展。然而，现有的视频编辑方法受到高计算开销和内存消耗的严重限制。此外，这些方法通常牺牲视觉保真度，导致不期望的时间不一致性和伪影，如模糊和明显的马赛克状图案。我们提出了Edit-Your-Interest，一种轻量级、文本驱动、零样本的视频编辑方法。Edit-Your-Interest引入了一个时空特征内存来缓存来自先前帧的特征，与完整序列时空建模方法相比显著减少了计算开销。具体来说，我们首先引入了一个时空特征内存库(SFM)，它被设计用来高效缓存和保留由空间注意力处理的关键图像标记。其次，我们提出了特征最相似传播(FMP)方法。FMP将最相关的标记从先前帧传播到后续帧，保持时间一致性。最后，我们引入了一个SFM更新算法，它不断刷新缓存的特征，确保它们在整个视频序列中的长期相关性和有效性。此外，我们利用交叉注意图自动提取感兴趣实例的掩码。这些掩码无缝集成到扩散去噪过程中，实现对目标对象的细粒度控制，并允许Edit-Your-Interest在稳健保持背景完整性的同时执行高度准确的编辑。大量实验明确证明，所提出的Edit-Your-Interest在效率和视觉保真度上都优于最先进的方法，验证了其优越的有效性和实用性。


### 论文摘要

Text-to-image (T2I) diffusion models have recently demonstrated significant progress in video editing.   However, existing video editing methods are severely limited by their high computational overhead and memory consumption.   Furthermore, these approaches often sacrifice visual fidelity, leading to undesirable temporal inconsistencies and artifacts such as blurring and pronounced mosaic-like patterns.   We propose Edit-Your-Interest, a lightweight, text-driven, zero-shot video editing method.   Edit-Your-Interest introduces a spatio-temporal feature memory to cache features from previous frames, significantly reducing computational overhead compared to full-sequence spatio-temporal modeling approaches.   Specifically, we first introduce a Spatio-Temporal Feature Memory bank (SFM), which is designed to efficiently cache and retain the crucial image tokens processed by spatial attention.   Second, we propose the Feature Most-Similar Propagation (FMP) method. FMP propagates the most relevant tokens from previous frames to subsequent ones, preserving temporal consistency.   Finally, we introduce an SFM update algorithm that continuously refreshes the cached features, ensuring their long-term relevance and effectiveness throughout the video sequence.   Furthermore, we leverage cross-attention maps to automatically extract masks for the instances of interest.   These masks are seamlessly integrated into the diffusion denoising process, enabling fine-grained control over target objects and allowing Edit-Your-Interest to perform highly accurate edits while robustly preserving the background integrity.   Extensive experiments decisively demonstrate that the proposed Edit-Your-Interest outperforms state-of-the-art methods in both efficiency and visual fidelity, validating its superior effectiveness and practicality.

---

## 48. SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding

**论文链接:** [http://arxiv.org/abs/2510.13016v1](http://arxiv.org/abs/2510.13016v1)

**作者:** Tanveer Hannan, Shuaicong Wu, Mark Weber, Suprosanna Shit, Jindong Gu, Rajat Koner, Aljoša Ošep, Laura Leal-Taixé, Thomas Seidl

**发布时间:** 2025-10-14

### GPT解析

### 总结

本研究引入了时空视频动作定位(SVAG)任务，旨在解决细粒度动作理解和对象时空定位的挑战，构建了大规模基准数据集SVAG-Bench，提出了基线框架SVAGFormer和评估工具包SVAGEval，发现现有模型在复杂场景中表现不佳，需要更高级的推理能力。

### 背景

细粒度动作理解和准确定位其对应的时间和空间位置是推进下一代AI系统的基础能力。然而，现有视频理解方法主要处理粗粒度动作识别或通用目标跟踪，忽略了根据动作联合检测和跟踪多个目标并在时间上定位它们的挑战。

### 目的

引入时空视频动作定位(SVAG)新任务，构建支持该任务的基准数据集，提出基线框架，并开发标准化评估工具包，以促进细粒度动作理解和对象时空定位的研究。

### 方法

构建SVAG-Bench基准测试，包含688个视频、19,590条标注记录和903个独特动词；提出SVAGFormer框架，适配最先进的视觉语言模型进行联合时空定位；开发SVAGEval标准化评估工具包以确保公平和可复现的基准测试。

### 主要发现

现有模型在SVAG任务上表现不佳，特别是在密集或复杂场景中，这突显了在长视频中针对细粒度对象-动作交互进行更高级推理的必要性。

### 结论

需要开发能够同时处理细粒度动作理解、对象跟踪和时空定位的AI系统，SVAG任务和基准测试为这一研究方向提供了重要基础。

### 翻译

理解细粒度动作并准确定位它们在空间和时间中对应的执行者是推进下一代AI系统的基础能力，包括具身智能体、自主平台和人机交互框架。尽管最近视频理解取得了进展，但现有方法主要处理粗粒度动作识别或通用目标跟踪，从而忽略了根据动作联合检测和跟踪多个目标并在时间上定位它们的挑战。为解决这一差距，我们引入时空视频动作定位(SVAG)，这是一个新任务，要求模型基于自然语言描述的动作同时检测、跟踪和时域定位视频中的所有指代对象。为支持此任务，我们构建了SVAG-Bench，这是一个大规模基准，包含688个视频、19,590条标注记录和903个独特动词，涵盖了多样化的对象、动作和现实世界场景。我们进一步提出了SVAGFormer，这是一个基线框架，适配最先进的视觉语言模型进行联合时空定位，并引入了SVAGEval，这是一个标准化评估工具包，用于公平和可复现的基准测试。实验结果表明，现有模型在SVAG上表现不佳，特别是在密集或复杂场景中，这突显了在长视频中针对细粒度对象-动作交互进行更高级推理的必要性。


### 论文摘要

Understanding fine-grained actions and accurately localizing their corresponding actors in space and time are fundamental capabilities for advancing next-generation AI systems, including embodied agents, autonomous platforms, and human-AI interaction frameworks. Despite recent progress in video understanding, existing methods predominantly address either coarse-grained action recognition or generic object tracking, thereby overlooking the challenge of jointly detecting and tracking multiple objects according to their actions while grounding them temporally. To address this gap, we introduce Spatio-temporal Video Action Grounding (SVAG), a novel task that requires models to simultaneously detect, track, and temporally localize all referent objects in videos based on natural language descriptions of their actions. To support this task, we construct SVAG-Bench, a large-scale benchmark comprising 688 videos, 19,590 annotated records, and 903 unique verbs, covering a diverse range of objects, actions, and real-world scenes. We further propose SVAGFormer, a baseline framework that adapts state of the art vision language models for joint spatial and temporal grounding, and introduce SVAGEval, a standardized evaluation toolkit for fair and reproducible benchmarking. Empirical results show that existing models perform poorly on SVAG, particularly in dense or complex scenes, underscoring the need for more advanced reasoning over fine-grained object-action interactions in long videos.

---

## 49. Real-Time Knee Angle Prediction Using EMG and Kinematic Data with an Attention-Based CNN-LSTM Network and Transfer Learning Across Multiple Datasets

**论文链接:** [http://arxiv.org/abs/2510.13443v1](http://arxiv.org/abs/2510.13443v1)

**作者:** Mojtaba Mollahossein, Gholamreza Vossoughi, Mohammad Hossein Rohban

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文提出了一种基于迁移学习的膝关节角度预测框架，使用轻量级注意力CNN-LSTM模型，仅需新受试者少量步态周期数据即可实现高精度预测。

### 背景

肌电信号(EMG)广泛用于通过机器学习和深度学习预测身体关节角度，但现有方法面临实时应用性有限、测试条件不具代表性以及需要大量数据集等挑战。

### 目的

开发一种仅需少量新受试者数据即可预测膝关节角度的迁移学习框架，解决现有方法的局限性。

### 方法

使用Georgia Tech、UCI和SMLE三个包含四个与膝关节运动相关EMG通道的数据集；开发轻量级基于注意力机制的CNN-LSTM模型，在Georgia Tech数据集上预训练后迁移到其他数据集；仅使用EMG输入，以及结合历史膝关节角度和多种传感器输入进行预测。

### 主要发现

仅使用EMG输入时，模型在异常受试者的一步和50步预测中NMAE分别为6.8%和13.7%；结合历史膝关节角度后，正常受试者NMAE降至3.1%和3.5%，异常受试者降至2.8%和7.5%；当使用EMG、运动学和相互作用力多种输入时，模型在一步和50步预测中NMAE分别达到1.09%和3.1%。

### 结论

该迁移学习框架在短期和长期康复场景中均表现出稳健的性能和强大的泛化能力，仅需少量新受试者数据即可实现高精度膝关节角度预测。

### 翻译

肌电(EMG)信号被广泛用于通过机器学习(ML)和深度学习(DL)方法预测身体关节角度。然而，这些方法通常面临实时应用性有限、测试条件不具代表性以及需要大量数据集才能实现最佳性能等挑战。本文提出了一个膝关节角度预测的迁移学习框架，只需要新受试者几个步态周期的数据。利用了三个数据集——Georgia Tech、加州大学欧文分校(UCI)和Sharif机械实验室外骨骼(SMLE)，这些数据集包含四个与膝关节运动相关的EMG通道。开发了一个轻量级的基于注意力机制的CNN-LSTM模型，在Georgia Tech数据集上进行预训练，然后转移到UCI和SMLE数据集。所提出的模型仅使用EMG输入，在异常受试者的一步和50步预测中实现了6.8%和13.7%的归一化平均绝对误差(NMAE)。结合历史膝关节角度将正常受试者的NMAE降低到3.1%和3.5%，异常受试者降低到2.8%和7.5%。当进一步适应SMLE外骨骼，使用EMG、运动学和相互作用力输入时，模型在一步和50步预测中分别实现了1.09%和3.1%的NMAE。这些结果表明模型在短期和长期康复场景中都具有稳健的性能和强大的泛化能力。


### 论文摘要

Electromyography (EMG) signals are widely used for predicting body joint angles through machine learning (ML) and deep learning (DL) methods. However, these approaches often face challenges such as limited real-time applicability, non-representative test conditions, and the need for large datasets to achieve optimal performance. This paper presents a transfer-learning framework for knee joint angle prediction that requires only a few gait cycles from new subjects. Three datasets - Georgia Tech, the University of California Irvine (UCI), and the Sharif Mechatronic Lab Exoskeleton (SMLE) - containing four EMG channels relevant to knee motion were utilized. A lightweight attention-based CNN-LSTM model was developed and pre-trained on the Georgia Tech dataset, then transferred to the UCI and SMLE datasets. The proposed model achieved Normalized Mean Absolute Errors (NMAE) of 6.8 percent and 13.7 percent for one-step and 50-step predictions on abnormal subjects using EMG inputs alone. Incorporating historical knee angles reduced the NMAE to 3.1 percent and 3.5 percent for normal subjects, and to 2.8 percent and 7.5 percent for abnormal subjects. When further adapted to the SMLE exoskeleton with EMG, kinematic, and interaction force inputs, the model achieved 1.09 percent and 3.1 percent NMAE for one- and 50-step predictions, respectively. These results demonstrate robust performance and strong generalization for both short- and long-term rehabilitation scenarios.

---

## 50. Machine Learning-Based Ultrasonic Weld Characterization Using Hierarchical Wave Modeling and Diffusion-Driven Distribution Alignment

**论文链接:** [http://arxiv.org/abs/2510.13023v1](http://arxiv.org/abs/2510.13023v1)

**作者:** Joshua R. Tempelman, Adam J. Wachtor, Eric B. Flynn

**发布时间:** 2025-10-14

**备注:** 26 pages, 6 page appendix

### GPT解析

### 总结

本文提出了一种端到端的机器学习工作流程，用于解决自动化超声波焊接检测中的数据有限和环境波动问题，通过结合降阶建模、扩散分布对齐和U-Net分割反演技术，实现了真实工业环境下的焊接缺陷检测。

### 背景

自动化超声波焊接检测在无损评估领域面临两大挑战：训练数据有限（由于实验标本整理或高保真模拟的复杂性）和工业环境的环境波动性（导致实时测量数据损坏）。

### 目的

开发一种在真实工业环境中进行声学焊接检测的端到端机器学习工作流程，克服数据整理和信号损坏问题。

### 方法

提出的工作流程包括：1)基于Lamb波理论的降阶Helmholtz模型生成综合数据集；2)使用相对廉价的低阶解为反演模型提供训练数据，并通过迁移学习优化；3)利用引导扩散处理分布外实验LDV扫描数据，生成分布内表示供反演模型处理。

### 主要发现

降阶模型能够生成全面的焊接异质性和裂纹缺陷数据集；迁移学习可有效利用有限的全3D弹性动力学模拟；扩散模型能够处理具有不可预测噪声分布的真实世界测量数据。

### 结论

该集成框架为真实数据上的自动化焊接检测提供了有效的端到端解决方案，克服了传统方法在数据有限和环境波动情况下的局限性。

### 翻译

自动化超声波焊接检测在无损评估领域仍是一个重大挑战，原因包括训练数据有限（由于整理实验标本或高保真模拟的复杂性）和许多工业环境的环境波动性（导致实时测量数据损坏）。因此，在真实（即工业）环境中进行声学焊接检测的端到端机器学习工作流程一直是一个难以实现的目标。本文通过提出包含降阶建模方案、基于扩散的分布对齐以及基于U-Net的分割和反演的工作流程，解决了数据整理和信号损坏的挑战。使用基于Lamb波理论的降阶Helmholtz模型，在变化的焊接异质性和裂纹缺陷上生成综合数据集。相对廉价的低阶解为反演模型提供了强大的训练数据集，这些模型通过使用有限的全3D弹性动力学模拟集的迁移学习阶段进行优化。为了处理具有变化且不可预测的噪声分布的分布外真实世界测量（即激光多普勒测振仪扫描），引导扩散生成OOD实验LDV扫描的分布内表示，随后由反演模型处理。这种集成框架为真实数据上的自动化焊接检测提供了端到端解决方案。


### 论文摘要

Automated ultrasonic weld inspection remains a significant challenge in the nondestructive evaluation (NDE) community to factors such as limited training data (due to the complexity of curating experimental specimens or high-fidelity simulations) and environmental volatility of many industrial settings (resulting in the corruption of on-the-fly measurements). Thus, an end-to-end machine learning (ML) workflow for acoustic weld inspection in realistic (i.e., industrial) settings has remained an elusive goal. This work addresses the challenges of data curation and signal corruption by proposing workflow consisting of a reduced-order modeling scheme, diffusion based distribution alignment, and U-Net-based segmentation and inversion. A reduced-order Helmholtz model based on Lamb wave theory is used to generate a comprehensive dataset over varying weld heterogeneity and crack defects. The relatively inexpensive low-order solutions provide a robust training dateset for inversion models which are refined through a transfer learning stage using a limited set of full 3D elastodynamic simulations. To handle out-of-distribution (OOD) real-world measurements with varying and unpredictable noise distributions, i.e., Laser Doppler Vibrometry scans, guided diffusion produces in-distribution representations of OOD experimental LDV scans which are subsequently processed by the inversion models. This integrated framework provides an end-to-end solution for automated weld inspection on real data.

---

## 51. PhysMaster: Mastering Physical Representation for Video Generation via Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.13809v1](http://arxiv.org/abs/2510.13809v1)

**作者:** Sihui Ji, Xi Chen, Xin Tao, Pengfei Wan, Hengshuang Zhao

**发布时间:** 2025-10-15

**备注:** Project Page: https://sihuiji.github.io/PhysMaster-Page/

### GPT解析

### 总结

PhysMaster是一个通过物理知识表示指导视频生成模型的框架，利用输入图像中的物理先验信息，通过强化学习和人类反馈优化物理表示，能够生成物理上更合理的视频。

### 背景

当前视频生成模型虽能生成视觉逼真的视频，但常常不遵守物理定律，限制了其生成物理合理视频的能力，使其无法成为有效的'世界模型'。

### 目的

提出PhysMaster模型，通过捕获物理知识作为表示来指导视频生成模型，增强其物理感知能力，使其能够生成物理上合理的视频。

### 方法

PhysMaster基于图像到视频任务，设计PhysEncoder从输入图像编码物理信息作为额外条件；采用强化学习与人类反馈相结合的方法，使用直接偏好优化(DPO)以端到端方式优化物理表示学习。

### 主要发现

PhysMaster为提高PhysEncoder的物理感知能力提供了可行解决方案，在简单代理任务上证明了其能力，并展现出广泛物理场景的泛化能力。

### 结论

PhysMaster通过在强化学习范式中通过表示学习统一解决各种物理过程，可作为物理感知视频生成的通用即插即用解决方案，具有更广泛的应用潜力。

### 翻译

当今的视频生成模型能够生成视觉上逼真的视频，但常常不遵守物理定律，限制了它们生成物理上合理的视频的能力，使其无法成为'世界模型'。为解决这一问题，我们提出了PhysMaster，它将物理知识捕获为一种表示，用于指导视频生成模型增强其物理感知能力。具体而言，PhysMaster基于图像到视频任务，模型需要从输入图像预测物理上合理的动态。由于输入图像提供了物理先验信息，如场景中物体的相对位置和潜在交互，我们设计了PhysEncoder从中编码物理信息作为额外条件，将物理知识注入视频生成过程。除了外观之外，模型物理性能缺乏适当的监督，这促使PhysEncoder应用强化学习与人类反馈相结合的方法进行物理表示学习，利用生成模型的反馈通过直接偏好优化(DPO)以端到端方式优化物理表示。PhysMaster为提高PhysEncoder的物理感知能力提供了可行解决方案，从而提高了视频生成的物理合理性，在简单代理任务上证明了其能力，并具有广泛物理场景的泛化能力。这意味着我们的PhysMaster通过在强化学习范式中通过表示学习统一解决各种物理过程，可以作为物理感知视频生成和更广泛应用的通用即插即用解决方案。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决当前视频生成模型不遵守物理规律的问题。这个问题很重要，因为物理真实性是视频生成模型能否作为'世界模型'的关键，限制了它们在模拟真实世界场景、预测物理交互等应用场景中的实用性，也阻碍了视频生成模型从内容创作者向世界模拟器的转变。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了视频生成模型在物理规律遵循方面的两个主要挑战：MSE损失关注外观拟合而非物理理解，以及生成模型难以从图像中提取物理知识。他们提出学习物理表示作为桥梁，借鉴了基于物理仿真和无仿真方法的思路，但避免了它们的局限性。同时采用了大型语言模型中的RLHF框架和DPO训练方法，设计了三阶段训练pipeline：SFT微调基础模型和PhysEncoder，然后两阶段DPO分别优化DiT模型和PhysEncoder，利用生成反馈改进物理表示。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是学习物理表示作为物理知识和视频生成之间的桥梁，通过PhysEncoder从输入图像提取物理特征作为额外条件指导视频生成。整体流程包括：1)基于DiT的扩散模型架构，结合3D VAE和T5编码器；2)PhysEncoder基于DINOv2编码器和物理头部设计；3)三阶段训练pipeline - SFT阶段同时训练DiT和PhysEncoder，DPO阶段先优化DiT模型再优化PhysEncoder；4)从'自由落体'代理任务开始，验证后扩展到一般开放世界场景；5)使用PisaBench和VIDEOPHY等评估方法验证效果。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)物理表示学习作为物理知识和视频生成的桥梁；2)自上而下的优化策略，基于最终视频的物理合理性优化物理编码器；3)三阶段训练pipeline结合SFT和DPO；4)插件式物理知识注入实现通用物理属性学习；5)从特定任务到开放世界场景的泛化能力。相比之前工作，PhysMaster不依赖特定物理仿真引擎，能处理更广泛物理现象；不依赖大规模物理数据集或昂贵人工注释；专注于优化物理编码器而非整个模型；效率更高，生成5秒视频仅需26秒。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PhysMaster通过学习物理表示并作为插件注入视频生成模型，利用强化学习优化物理编码器，显著提升了视频生成模型的物理合理性，使其能够从内容创作者转变为遵循物理规律的世界模拟器。'}


### 论文摘要

Video generation models nowadays are capable of generating visually realistic videos, but often fail to adhere to physical laws, limiting their ability to generate physically plausible videos and serve as ''world models''. To address this issue, we propose PhysMaster, which captures physical knowledge as a representation for guiding video generation models to enhance their physics-awareness. Specifically, PhysMaster is based on the image-to-video task where the model is expected to predict physically plausible dynamics from the input image. Since the input image provides physical priors like relative positions and potential interactions of objects in the scenario, we devise PhysEncoder to encode physical information from it as an extra condition to inject physical knowledge into the video generation process. The lack of proper supervision on the model's physical performance beyond mere appearance motivates PhysEncoder to apply reinforcement learning with human feedback to physical representation learning, which leverages feedback from generation models to optimize physical representations with Direct Preference Optimization (DPO) in an end-to-end manner. PhysMaster provides a feasible solution for improving physics-awareness of PhysEncoder and thus of video generation, proving its ability on a simple proxy task and generalizability to wide-ranging physical scenarios. This implies that our PhysMaster, which unifies solutions for various physical processes via representation learning in the reinforcement learning paradigm, can act as a generic and plug-in solution for physics-aware video generation and broader applications.

---

## 52. UniME-V2: MLLM-as-a-Judge for Universal Multimodal Embedding Learning

**论文链接:** [http://arxiv.org/abs/2510.13515v1](http://arxiv.org/abs/2510.13515v1)

**作者:** Tiancheng Gu, Kaicheng Yang, Kaichen Zhang, Xiang An, Ziyong Feng, Yueyi Zhang, Weidong Cai, Jiankang Deng, Lidong Bing

**发布时间:** 2025-10-15

**备注:** 12 pages, 6 figures, 11 tables

### GPT解析

### 总结

本研究提出了一种名为UniME-V2的新型通用多模态嵌入模型，通过利用大型多模态语言模型的先进理解能力来增强表示学习。该方法通过MLLM-as-a-Judge机制评估语义对齐，生成软语义匹配分数，用于高质量困难负样本挖掘和模型优化，显著提升了模型的判别能力，并在多个检索任务上实现了最先进的性能。

### 背景

现有的通用多模态嵌入模型通常采用批内负样本挖掘方法测量查询-候选对的相似性，但这些方法存在几个局限：难以捕捉候选者之间的细微语义差异，负样本缺乏多样性，以及在区分错误负样本和困难负样本方面的判别能力有限。

### 目的

本研究旨在解决现有多模态嵌入模型的局限性，提高其捕捉细微语义差异的能力，增加负样本的多样性，并增强模型对困难负样本的判别能力，从而提升通用多模态嵌入模型的整体性能。

### 方法

研究团队提出了一种名为UniME-V2的新型通用多模态嵌入模型，主要方法包括：1) 通过全局检索构建潜在困难负样本集；2) 引入MLLM-as-a-Judge机制，利用大型多模态语言模型评估查询-候选对的语义对齐并生成软语义匹配分数；3) 将这些分数作为困难负样本挖掘的基础，减轻错误负样本影响，识别多样化高质量困难负样本；4) 将语义匹配分数作为软标签，缓解一对一映射约束；5) 通过对齐相似度矩阵与软语义匹配分数矩阵，学习候选者间的语义区别；6) 提出UniME-V2-Reranker重排序模型，通过联合成对和列表级优化方法训练。

### 主要发现

在MMEB基准和多个检索任务上的全面实验表明，该方法在所有任务上平均达到了最先进的性能，证明了所提出方法的有效性和优越性。

### 结论

通过利用大型多模态语言模型的先进理解能力和创新的负样本挖掘方法，UniME-V2模型显著提高了通用多模态嵌入模型的性能，特别是在捕捉细微语义差异和区分困难负样本方面，为多模态表示学习领域提供了新的思路和解决方案。

### 翻译

通用多模态嵌入模型是各种任务的基础。现有方法通常采用批内负样本挖掘来测量查询-候选对的相似性。然而，这些方法往往难以捕捉候选者之间的细微语义差异，且负样本缺乏多样性。此外，嵌入模型在区分错误负样本和困难负样本方面的判别能力有限。在本文中，我们利用大型多模态语言模型的先进理解能力来增强表示学习，并提出了一种新颖的通用多模态嵌入模型。我们的方法首先通过全局检索构建潜在的困难负样本集。然后我们引入MLLM-as-a-Judge机制，利用大型多模态语言模型评估查询-候选对的语义对齐情况，并生成软语义匹配分数。这些分数作为困难负样本挖掘的基础，减轻了错误负样本的影响，并能够识别出多样化的高质量困难负样本。此外，语义匹配分数还被用作软标签，以缓解严格的一对一映射约束。通过将相似度矩阵与软语义匹配分数矩阵对齐，模型能够学习候选者之间的语义区别，显著提高其判别能力。为了进一步提高性能，我们提出了UniME-V2-Reranker，这是一个通过联合成对和列表级优化方法在我们挖掘的困难负样本上训练的重排序模型。我们在MMEB基准和多个检索任务上进行了全面实验，证明我们的方法在所有任务上平均达到了最先进的性能。


### 论文摘要

Universal multimodal embedding models are foundational to various tasks. Existing approaches typically employ in-batch negative mining by measuring the similarity of query-candidate pairs. However, these methods often struggle to capture subtle semantic differences among candidates and lack diversity in negative samples. Moreover, the embeddings exhibit limited discriminative ability in distinguishing false and hard negatives. In this paper, we leverage the advanced understanding capabilities of MLLMs to enhance representation learning and present a novel Universal Multimodal Embedding (UniME-V2) model. Our approach first constructs a potential hard negative set through global retrieval. We then introduce the MLLM-as-a-Judge mechanism, which utilizes MLLMs to assess the semantic alignment of query-candidate pairs and generate soft semantic matching scores. These scores serve as a foundation for hard negative mining, mitigating the impact of false negatives and enabling the identification of diverse, high-quality hard negatives. Furthermore, the semantic matching scores are used as soft labels to mitigate the rigid one-to-one mapping constraint. By aligning the similarity matrix with the soft semantic matching score matrix, the model learns semantic distinctions among candidates, significantly enhancing its discriminative capacity. To further improve performance, we propose UniME-V2-Reranker, a reranking model trained on our mined hard negatives through a joint pairwise and listwise optimization approach. We conduct comprehensive experiments on the MMEB benchmark and multiple retrieval tasks, demonstrating that our method achieves state-of-the-art performance on average across all tasks.

---

## 53. DistilCLIP-EEG: Enhancing Epileptic Seizure Detection Through Multi-modal Learning and Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2510.13497v1](http://arxiv.org/abs/2510.13497v1)

**作者:** Zexin Wang, Lin Shi, Haoyu Wu, Junru Luo, Xiangzeng Kong, Jun Qi

**发布时间:** 2025-10-15

**DOI:** 10.1109/JBHI.2025.3603022

**备注:** 16 pages, 9 figures, 5 tables

### GPT解析

### 总结

提出了一种基于CLIP框架的多模态模型DistilCLIP-EEG，整合脑电图信号和文本描述进行癫痫检测，并通过知识蒸馏方法创建轻量级学生模型，在多个数据集上实现了超过97%的准确率。

### 背景

癫痫是一种常见的神经系统疾病，特征是突然、短暂的大脑神经元过度活动，由异常放电引起。目前大多数癫痫检测的深度学习方法仅依赖单模态的脑电图信号，忽视了多模态信息的潜在优势。

### 目的

提出一种新颖的多模态模型DistilCLIP-EEG，基于CLIP框架，整合脑电图信号和文本描述，以捕捉癫痫发作的全面特征，并通过知识蒸馏方法提高效率和适应性。

### 方法

提出了一种基于CLIP框架的多模态模型DistilCLIP-EEG，整合脑电图信号和文本描述。该模型包含基于Conformer架构的脑电图编码器作为文本编码器，以及可学习BERT(BERT-LP)作为编码器内的提示学习。两者在共享的潜在空间中运行，实现有效的跨模态表示学习。同时引入知识蒸馏方法，训练好的DistilCLIP-EEG作为教师模型，指导一个更紧凑的学生模型。

### 主要发现

在TUSZ、AUBMC和CHB-MIT数据集上，教师模型和学生模型的准确率均超过97%。在所有数据集上，F1分数持续高于0.94，证明了所提出框架的鲁棒性和可靠性。学生模型的参数数量和模型大小约为教师模型的58.1%，显著降低了模型复杂性和存储需求，同时保持高性能。

### 结论

该模型突显了在基于脑电图的癫痫检测中的潜力，并为在资源受限环境中部署轻量级模型奠定了坚实基础。

### 翻译

癫痫是一种常见的神经系统疾病，特征是突然、短暂的大脑神经元过度活动 episodes，由异常放电引起，可能导致一些精神障碍。目前大多数用于癫痫检测的深度学习方法仅依赖单模态的脑电图(EEG)信号，忽视了多模态信息的潜在优势。为此，我们提出了一种新颖的多模态模型 DistilCLIP-EEG，基于CLIP框架，整合了脑电图信号和文本描述，以捕捉癫痫发作的全面特征。该模型包含基于Conformer架构的脑电图编码器作为文本编码器，以及我们提出的可学习BERT(BERT-LP)作为编码器内的提示学习。两者在共享的潜在空间中运行，实现有效的跨模态表示学习。为了提高效率和适应性，我们引入了一种知识蒸馏方法，其中训练好的DistilCLIP-EEG作为教师模型，指导一个更紧凑的学生模型，以降低训练复杂度和时间。在TUSZ、AUBMC和CHB-MIT数据集上，教师模型和学生模型的准确率均超过97%。在所有数据集上，F1分数持续高于0.94，证明了所提出框架的鲁棒性和可靠性。此外，学生模型的参数数量和模型大小约为教师模型的58.1%，显著降低了模型复杂性和存储需求，同时保持高性能。这些结果突显了我们提出的模型在基于脑电图的癫痫检测中的潜力，并为在资源受限环境中部署轻量级模型奠定了坚实基础。


### 论文摘要

Epilepsy is a prevalent neurological disorder marked by sudden, brief episodes of excessive neuronal activity caused by abnormal electrical discharges, which may lead to some mental disorders. Most existing deep learning methods for epilepsy detection rely solely on unimodal EEG signals, neglecting the potential benefits of multimodal information. To address this, we propose a novel multimodal model, DistilCLIP-EEG, based on the CLIP framework, which integrates both EEG signals and text descriptions to capture comprehensive features of epileptic seizures. The model involves an EEG encoder based on the Conformer architecture as a text encoder, the proposed Learnable BERT (BERT-LP) as prompt learning within the encoders. Both operate in a shared latent space for effective cross-modal representation learning. To enhance efficiency and adaptability, we introduce a knowledge distillation method where the trained DistilCLIP-EEG serves as a teacher to guide a more compact student model to reduce training complexity and time. On the TUSZ, AUBMC, and CHB-MIT datasets, both the teacher and student models achieved accuracy rates exceeding 97%. Across all datasets, the F1-scores were consistently above 0.94, demonstrating the robustness and reliability of the proposed framework. Moreover, the student model's parameter count and model size are approximately 58.1% of those of the teacher model, significantly reducing model complexity and storage requirements while maintaining high performance. These results highlight the potential of our proposed model for EEG-based epilepsy detection and establish a solid foundation for deploying lightweight models in resource-constrained settings.

---

## 54. End-to-End Multi-Modal Diffusion Mamba

**论文链接:** [http://arxiv.org/abs/2510.13253v1](http://arxiv.org/abs/2510.13253v1)

**作者:** Chunhao Lu, Qiang Lu, Meichen Dong, Jake Luo

**发布时间:** 2025-10-15

**备注:** Accepted by ICCV 2025

### GPT解析

### 总结

本文提出了一种名为MDM（多模态扩散Mamba）的新型架构，通过统一的变分自编码器实现多模态处理的统一，在多个任务上表现出色。

### 背景

当前端到端多模态模型使用不同的编码器和解码器处理输入和输出信息，这种分离阻碍了不同模态的联合表示学习。

### 目的

为了统一多模态处理，解决现有模型中不同模态处理分离的问题。

### 方法

MDM利用基于Mamba的多步选择扩散模型，通过统一的变分自编码器逐步生成和优化模态特定信息。

### 主要发现

在图像生成、图像描述、视觉问答、文本理解和推理任务等领域的评估表明，MDM显著优于现有的端到端模型（如MonoFormer、LlamaGen和Chameleon等），并能与GPT-4V、Gemini Pro和Mistral等最先进模型有效竞争。

### 结论

研究结果验证了MDM在统一多模态处理的同时保持计算效率方面的有效性，为端到端多模态架构建立了新方向。

### 翻译

当前端到端多模态模型使用不同的编码器和解码器来处理输入和输出信息。这种分离阻碍了不同模态的联合表示学习。为了统一多模态处理，我们提出了一种名为MDM（多模态扩散Mamba）的新型架构。MDM利用基于Mamba的多步选择扩散模型，通过统一的变分自编码器逐步生成和优化模态特定信息。这种创新方法使MDM在处理高维数据时能够实现卓越的性能，特别是在同时生成高分辨率图像和扩展文本序列方面。我们在图像生成、图像描述、视觉问答、文本理解和推理任务等领域的评估表明，MDM显著优于现有的端到端模型（MonoFormer、LlamaGen和Chameleon等），并能与GPT-4V、Gemini Pro和Mistral等最先进模型有效竞争。我们的结果验证了MDM在统一多模态处理的同时保持计算效率方面的有效性，为端到端多模态架构建立了新方向。


### 论文摘要

Current end-to-end multi-modal models utilize different encoders and decoders to process input and output information. This separation hinders the joint representation learning of various modalities. To unify multi-modal processing, we propose a novel architecture called MDM (Multi-modal Diffusion Mamba). MDM utilizes a Mamba-based multi-step selection diffusion model to progressively generate and refine modality-specific information through a unified variational autoencoder for both encoding and decoding. This innovative approach allows MDM to achieve superior performance when processing high-dimensional data, particularly in generating high-resolution images and extended text sequences simultaneously. Our evaluations in areas such as image generation, image captioning, visual question answering, text comprehension, and reasoning tasks demonstrate that MDM significantly outperforms existing end-to-end models (MonoFormer, LlamaGen, and Chameleon etc.) and competes effectively with SOTA models like GPT-4V, Gemini Pro, and Mistral. Our results validate MDM's effectiveness in unifying multi-modal processes while maintaining computational efficiency, establishing a new direction for end-to-end multi-modal architectures.

---

## 55. A Matter of Representation: Towards Graph-Based Abstract Code Generation

**论文链接:** [http://arxiv.org/abs/2510.13163v1](http://arxiv.org/abs/2510.13163v1)

**作者:** Nyx Iskandar, Hisham Bedri, Andy Tsen

**发布时间:** 2025-10-15

### GPT解析

### 总结

本文研究了基于图的抽象代码生成，提出并评估了JSON表示方法，使大型语言模型能够高精度地执行此类任务。

### 背景

大多数大型语言模型擅长生成原始顺序代码，但很少研究基于图的抽象代码生成，这种方法在可视化编程语言和原始源代码不可用的情况下很有价值。

### 目的

提出并评估JSON表示方法，以实现高精度的基于图的抽象代码生成，并研究不同表示方法对生成准确性的影响。

### 方法

使用ScratchTest（基于Scratch Python重新实现的迷你基准测试）评估不同的JSON图表示方法，测试LLM在代码图空间中的表现。

### 主要发现

大型语言模型可以在单次通过中执行基于图的抽象代码生成任务，无需依赖专门或复杂的管道，且不同表示方法会导致显著不同的准确性。

### 结论

这项工作为基于图的抽象代码生成的表示学习奠定了基础，突显了适当表示方法的重要性。

### 翻译

目前大多数大型语言模型擅长生成具有最小抽象和自定义结构的原始顺序代码。然而，很少有关于基于图的抽象代码生成的工作，其中重要逻辑被封装在预定义节点中，执行流程由边决定。这对于可视化编程语言，以及原始源代码对用户和LLM训练集不可用的情况相关。在这项工作中，我们提出并评估了用于图的JSON表示，以实现高精度的基于图的抽象代码生成。我们在ScratchTest上评估了这些表示，这是一个基于我们自定义的Scratch Python重新实现的迷你基准测试，用于测试LLM在代码图空间中的表现。我们的研究结果表明，LLM确实可以在单次通过中执行上述生成任务，而不依赖于专门的或复杂的管道，前提是使用正确的图表示。我们还表明，不同的表示会导致显著不同的准确性，突显了表示在此生成任务中的重要作用。总而言之，这项工作为基于图的抽象代码生成的表示学习建立了第一步。


### 论文摘要

Most large language models (LLMs) today excel at generating raw, sequential code with minimal abstractions and custom structures. However, there has been little work on graph-based abstract code generation, where significant logic is encapsulated in predefined nodes and execution flow is determined by edges. This is relevant for visual programming languages, and in cases where raw source code is inaccessible to users and LLM training sets. In this work, we propose and evaluate JSON representations for graphs to enable high accuracy graph-based abstract code generation. We evaluate these representations on ScratchTest, a mini-benchmark based on our custom Python re-implementation of Scratch, which tests the LLM in code graph space. Our findings demonstrate that LLMs can indeed perform the aforementioned generation task in a single pass without relying on specialized or complex pipelines, given the correct graph representations. We also show that different representations induce significantly different accuracies, highlighting the instrumental role of representations in this generation task. All in all, this work establishes the first steps towards representation learning for graph-based abstract code generation.

---

## 56. Information Shapes Koopman Representation

**论文链接:** [http://arxiv.org/abs/2510.13025v1](http://arxiv.org/abs/2510.13025v1)

**作者:** Xiaoyuan Cheng, Wenxuan Yuan, Yiming Yang, Yuanzhao Zhang, Sibo Cheng, Yi He, Zhuo Sun

**发布时间:** 2025-10-14

### GPT解析

### 总结

本研究通过信息论视角重新思考Koopman学习，提出一种平衡表示简单性和表达性的新方法，解决了Koopman算子在深度架构中面临的子空间选择挑战。

### 背景

Koopman算子为建模动力系统提供了强大框架，受到机器学习社区日益关注，但其无限维特性使得识别合适的有限维子空间具有挑战性，特别是在深度架构中。

### 目的

解决Koopman学习中次优表示学习的问题，平衡潜在变量在表达性和简单性之间的权衡，克服信息瓶颈困境。

### 方法

提出一种信息论拉格朗日公式化，明确平衡简单性和表达性的权衡；基于该公式开发新算法，促进潜在互信息（简单性）和冯·诺依曼熵（表达性）的共同优化。

### 主要发现

潜在互信息促进简单性但过度强调可能导致潜在空间崩溃；冯·诺依曼熵维持表达性并防止崩溃，鼓励模式多样性；所提方法产生稳定且可解释的Koopman表示。

### 结论

通过信息论视角重新审视Koopman学习，提出的新方法在多种动力系统上验证优于现有方法，实现了更好的性能和可解释性。

### 翻译

Koopman算子为建模动力系统提供了强大框架，并吸引了机器学习界的日益关注。然而，其无限维特性使得识别合适的有限维子空间具有挑战性，特别是对于深度架构。我们认为这些困难来自于次优的表示学习，其中潜在变量无法平衡表达性和简单性。这种张力与信息瓶颈(IB)困境密切相关：构建既紧凑又有预测能力的压缩表示。通过这一视角重新思考Koopman学习，我们证明潜在互信息促进简单性，但过度强调简单性可能导致潜在空间崩溃到少数主导模式。相比之下，表达性由冯·诺依曼熵维持，防止这种崩溃并鼓励模式多样性。这一见解促使我们提出一种明确平衡这种权衡的信息论拉格朗日公式化。此外，我们基于该公式提出新算法，鼓励简单性和表达性，产生稳定且可解释的Koopman表示。除了定量评估外，我们还可视化了在我们表示下学习到的流形，观察到与理论预测一致的实证结果。最后，我们在多种动力系统上验证了我们的方法，展示了与现有Koopman学习方法相比的改进性能。实现已在https://github.com/Wenxuan52/InformationKoopman公开可用。


### 论文摘要

The Koopman operator provides a powerful framework for modeling dynamical systems and has attracted growing interest from the machine learning community. However, its infinite-dimensional nature makes identifying suitable finite-dimensional subspaces challenging, especially for deep architectures. We argue that these difficulties come from suboptimal representation learning, where latent variables fail to balance expressivity and simplicity. This tension is closely related to the information bottleneck (IB) dilemma: constructing compressed representations that are both compact and predictive. Rethinking Koopman learning through this lens, we demonstrate that latent mutual information promotes simplicity, yet an overemphasis on simplicity may cause latent space to collapse onto a few dominant modes. In contrast, expressiveness is sustained by the von Neumann entropy, which prevents such collapse and encourages mode diversity. This insight leads us to propose an information-theoretic Lagrangian formulation that explicitly balances this tradeoff. Furthermore, we propose a new algorithm based on the Lagrangian formulation that encourages both simplicity and expressiveness, leading to a stable and interpretable Koopman representation. Beyond quantitative evaluations, we further visualize the learned manifolds under our representations, observing empirical results consistent with our theoretical predictions. Finally, we validate our approach across a diverse range of dynamical systems, demonstrating improved performance over existing Koopman learning methods. The implementation is publicly available at https://github.com/Wenxuan52/InformationKoopman.

---

## 57. A Multimodal XAI Framework for Trustworthy CNNs and Bias Detection in Deep Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.12957v1](http://arxiv.org/abs/2510.12957v1)

**作者:** Noor Islam S. Mohammad

**发布时间:** 2025-10-14

### GPT解析

### 总结

提出了一种新颖的多模态可解释AI框架，通过注意力增强特征融合、Grad-CAM++局部解释和Reveal-to-Revise反馈循环解决偏差检测和减轻问题，在多模态MNIST上实现了高准确率和解释保真度。

### 背景

标准基准数据集如MNIST无法揭示潜在的偏差和多模态特征复杂性，限制了深度神经网络在高风险应用中的可信度。

### 目的

开发一个多模态可解释AI框架，实现偏差检测和减轻，提高AI系统的透明度和可信度。

### 方法

统一了注意力增强的特征融合、基于Grad-CAM++的局部解释以及Reveal-to-Revise反馈循环，形成一个完整的偏差检测和减轻框架。

### 主要发现

在多模态扩展的MNIST上实现了93.2%的分类准确率、91.6%的F1分数和78.1%的解释保真度，优于单模态和不可解释的基线方法；消融研究表明可解释性与偏差感知学习的结合增强了模型的鲁棒性和人类对齐。

### 结论

该工作弥合了性能、透明度和公平性之间的差距，为敏感领域可信AI的实际应用提供了可行途径。

### 翻译

标准基准数据集如MNIST往往无法揭示潜在的偏差和多模态特征复杂性，限制了深度神经网络在高风险应用中的可信度。我们提出了一种新颖的多模态可解释AI(XAI)框架，统一了注意力增强的特征融合、基于Grad-CAM++的局部解释以及Reveal-to-Revise反馈循环，用于偏差检测和减轻。在多模态扩展的MNIST上评估，我们的方法实现了93.2%的分类准确率、91.6%的F1分数和78.1%的解释保真度，优于单模态和不可解释的基线。消融研究表明，将可解释性与偏差感知学习相结合可以增强鲁棒性和人类对齐。我们的工作弥合了性能、透明度和公平性之间的差距，突显了敏感领域可信AI的实际应用途径。


### 论文摘要

Standard benchmark datasets, such as MNIST, often fail to expose latent biases and multimodal feature complexities, limiting the trustworthiness of deep neural networks in high-stakes applications. We propose a novel multimodal Explainable AI (XAI) framework that unifies attention-augmented feature fusion, Grad-CAM++-based local explanations, and a Reveal-to-Revise feedback loop for bias detection and mitigation. Evaluated on multimodal extensions of MNIST, our approach achieves 93.2% classification accuracy, 91.6% F1-score, and 78.1% explanation fidelity (IoU-XAI), outperforming unimodal and non-explainable baselines. Ablation studies demonstrate that integrating interpretability with bias-aware learning enhances robustness and human alignment. Our work bridges the gap between performance, transparency, and fairness, highlighting a practical pathway for trustworthy AI in sensitive domains.

---

## 58. FedGTEA: Federated Class-Incremental Learning with Gaussian Task Embedding and Alignment

**论文链接:** [http://arxiv.org/abs/2510.12927v1](http://arxiv.org/abs/2510.12927v1)

**作者:** Haolin Li, Hoda Bidkhori

**发布时间:** 2025-10-14

### GPT解析

### 总结

本文提出了FedGTEA框架，用于联邦增量学习，通过高斯任务嵌入和实现对任务特定知识和模型不确定性的高效捕捉，具有可扩展性和通信效率优势。

### 背景

联邦增量学习领域需要有效捕捉任务特定知识和模型不确定性，同时确保可扩展性和通信效率。

### 目的

设计一个能够捕捉任务特定知识和模型不确定性，同时保持可扩展性和通信效率的联邦学习框架。

### 方法

客户端使用Cardinality-Agnostic Task Encoder (CATE)生成高斯分布的任务嵌入，编码任务知识并解决统计异构性；服务器端利用2-Wasserstein距离衡量任务间差距，通过Wasserstein损失强制任务间分离，同时保护任务级隐私。

### 主要发现

在多个流行数据集上的实证评估显示，FedGTEA实现了卓越的分类性能，显著减轻了遗忘问题，持续优于现有基线方法。

### 结论

FedGTEA框架在联邦增量学习任务中表现优异，能够有效处理任务特定知识、模型不确定性，同时保持可扩展性和通信效率。

### 翻译

我们提出了一种联邦增量学习的新框架，称为联邦高斯任务嵌入与对齐（FedGTEA）。FedGTEA旨在以可扩展且通信高效的方式捕捉任务特定知识和模型不确定性。在客户端，Cardinality-Agnostic Task Encoder (CATE)生成高斯分布的任务嵌入，这些嵌入编码任务知识，解决统计异构性问题，并量化数据不确定性。重要的是，CATE保持固定参数大小，无论任务数量如何，这确保了长任务序列的可扩展性。在服务器端，FedGTEA利用2-Wasserstein距离来衡量高斯嵌入之间的任务间差距。我们制定Wasserstein损失以强制实现任务间分离。这种概率性表述不仅增强了表示学习，还通过避免直接传输潜在嵌入来保护任务级隐私，符合联邦学习中的隐私约束。在流行数据集上的大量实证评估表明，FedGTEA实现了卓越的分类性能，显著减轻了遗忘问题，持续优于现有的强大基线。


### 论文摘要

We introduce a novel framework for Federated Class Incremental Learning, called Federated Gaussian Task Embedding and Alignment (FedGTEA). FedGTEA is designed to capture task-specific knowledge and model uncertainty in a scalable and communication-efficient manner. At the client side, the Cardinality-Agnostic Task Encoder (CATE) produces Gaussian-distributed task embeddings that encode task knowledge, address statistical heterogeneity, and quantify data uncertainty. Importantly, CATE maintains a fixed parameter size regardless of the number of tasks, which ensures scalability across long task sequences. On the server side, FedGTEA utilizes the 2-Wasserstein distance to measure inter-task gaps between Gaussian embeddings. We formulate the Wasserstein loss to enforce inter-task separation. This probabilistic formulation not only enhances representation learning but also preserves task-level privacy by avoiding the direct transmission of latent embeddings, aligning with the privacy constraints in federated learning. Extensive empirical evaluations on popular datasets demonstrate that FedGTEA achieves superior classification performance and significantly mitigates forgetting, consistently outperforming strong existing baselines.

---

