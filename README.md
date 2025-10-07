# 今日论文推荐 - 2025-10-07

共 98 篇论文

---

## 1. Do Vision-Language Models See Urban Scenes as People Do? An Urban Perception Benchmark

**论文链接:** [http://arxiv.org/abs/2509.14574v2](http://arxiv.org/abs/2509.14574v2)

**作者:** Rashid Mushkani

**发布时间:** 2025-09-18

### GPT解析

### 总结

该研究介绍了一个用于测试视觉语言模型(VLMs)在城市感知能力上的小基准测试，使用100张蒙特利尔街道图像，收集了来自7个社区群体的12名参与者在30个维度上的230份标注表。研究发现模型在客观属性理解上优于主观评价，最佳系统(claude-sonnet)在多标签项上达到了宏观0.31和平均Jaccard 0.48的分数。

### 背景

理解人们如何阅读城市场景可以为城市设计和规划提供重要信息，但目前缺乏针对视觉语言模型在城市感知能力上的标准化测试方法。

### 目的

创建一个小型基准测试，用于评估视觉语言模型在理解城市场景方面的能力，特别是在区分客观物理属性和主观印象方面。

### 方法

使用100张蒙特利尔街道图像（真实照片和逼真合成场景各半），收集7个社区群体12名参与者在30个维度上的标注表，将法语响应标准化为英语，在零样本设置下评估7个VLM模型，使用结构化提示和确定性解析器，对单选项目使用准确率，对多标签项目使用Jaccard重叠系数。

### 主要发现

1) 模型在可见的客观属性上比主观评价上有更好的对齐效果；2) 最佳系统(claude-sonnet)在多标签项上达到宏观0.31和平均Jaccard 0.48的分数；3) 人类一致性越高，模型得分越好；4) 合成图像略微降低了分数。

### 结论

该基准测试为评估视觉语言模型在城市感知方面的能力提供了工具，结果显示模型在客观属性理解上优于主观评价，研究还提供了可复现和不确定性感知的评估工具，可用于参与式城市分析。

### 翻译

理解人们如何阅读城市场景可以为设计和规划提供信息。我们引入了一个小型基准测试，用于使用100张蒙特利尔街道图像测试视觉语言模型(VLMs)在城市感知方面的能力，这些图像 evenly split between photographs和photorealistic synthetic scenes。来自7个社区群体的12名参与者在30个维度上提供了230份标注表，这些维度混合了物理属性和主观印象。法语响应被标准化为英语。我们在零样本设置下使用结构化提示和确定性解析器评估了7个VLM模型。对于单选项目我们使用准确率，对于多标签项目使用Jaccard重叠系数；人类一致性使用Krippendorff's alpha和成对Jaccard系数。结果表明，模型在可见的客观属性上比主观评价上有更好的对齐效果。最佳系统(claude-sonnet)在多标签项上达到了宏观0.31和平均Jaccard 0.48的分数。更高的人类一致性对应更好的模型得分。合成图像略微降低了分数。我们发布了基准测试、提示词和用于参与式城市分析的可复现、不确定性感知的评估工具。


### 论文摘要

Understanding how people read city scenes can inform design and planning. We introduce a small benchmark for testing vision-language models (VLMs) on urban perception using 100 Montreal street images, evenly split between photographs and photorealistic synthetic scenes. Twelve participants from seven community groups supplied 230 annotation forms across 30 dimensions mixing physical attributes and subjective impressions. French responses were normalized to English. We evaluated seven VLMs in a zero-shot setup with a structured prompt and deterministic parser. We use accuracy for single-choice items and Jaccard overlap for multi-label items; human agreement uses Krippendorff's alpha and pairwise Jaccard. Results suggest stronger model alignment on visible, objective properties than subjective appraisals. The top system (claude-sonnet) reaches macro 0.31 and mean Jaccard 0.48 on multi-label items. Higher human agreement coincides with better model scores. Synthetic images slightly lower scores. We release the benchmark, prompts, and harness for reproducible, uncertainty-aware evaluation in participatory urban analysis.

---

## 2. ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning

**论文链接:** [http://arxiv.org/abs/2510.05070v1](http://arxiv.org/abs/2510.05070v1)

**作者:** Siheng Zhao, Yanjie Ze, Yue Wang, C. Karen Liu, Pieter Abbeel, Guanya Shi, Rocky Duan

**发布时间:** 2025-10-06

**备注:** 9 pages, 8 figures

### GPT解析

### 总结

本研究提出了ResMimic，一个用于人形机器人全身运动操作的两阶段残差学习框架，通过结合通用运动跟踪和精确残差策略，实现了从人类运动数据中学习精确且富有表现力的机器人控制。

### 背景

人形机器人全身运动操作在日常生活服务和仓库任务中具有变革性潜力。尽管通用运动跟踪技术已使人形机器人能够复制人类动作，但这些策略缺乏运动操作所需的精确性和物体感知能力。

### 目的

开发一个能够实现精确且富有表现力的人形机器人控制系统，使机器人能够执行需要精确运动和物体交互的任务。

### 方法

提出ResMimic两阶段框架：首先使用大规模人类运动数据训练通用运动跟踪策略作为基础；然后学习高效但精确的残差策略来优化GMT输出，改善运动并融入物体交互。还设计了基于点云的物体跟踪奖励、接触奖励和基于课程的虚拟物体控制器来促进高效训练。

### 主要发现

在模拟和真实Unitree G1人形机器人上的评估表明，ResMimic与强大基线相比，在任务成功率、训练效率和鲁棒性方面都有显著提升。

### 结论

ResMimic框架有效解决了人形机器人全身运动操作中精确性和物体感知的挑战，为人形机器人在实际服务任务中的应用提供了新的可能性。

### 翻译

人形机器人全身运动操作有望为日常服务和仓库任务带来变革性能力。尽管最近在通用运动跟踪方面的进展使人形机器人能够复制各种人类动作，但这些策略缺乏运动操作所需的精确性和物体感知能力。为此，我们提出了ResMimic，一个从人类运动数据中实现精确且富有表现力的人形机器人控制的两阶段残差学习框架。首先，一个在仅基于人类大规模运动数据上训练的GMT策略，作为生成类人全身运动的与任务无关的基础。然后，学习一个高效但精确的残差策略来优化GMT输出，以改善运动并融入物体交互。为了进一步促进高效训练，我们设计了(i)基于点云的物体跟踪奖励，以实现更平滑的优化；(ii)接触奖励，鼓励精确的人形机器人-物体交互；(iii)基于课程的虚拟物体控制器，以稳定早期训练。我们在模拟和真实的Unitree G1人形机器人上评估了ResMimic。结果表明，与强大的基线相比，ResMimic在任务成功率、训练效率和鲁棒性方面都有显著提升。视频可在https://resmimic.github.io/ 查看。


### 论文摘要

Humanoid whole-body loco-manipulation promises transformative capabilities for daily service and warehouse tasks. While recent advances in general motion tracking (GMT) have enabled humanoids to reproduce diverse human motions, these policies lack the precision and object awareness required for loco-manipulation. To this end, we introduce ResMimic, a two-stage residual learning framework for precise and expressive humanoid control from human motion data. First, a GMT policy, trained on large-scale human-only motion, serves as a task-agnostic base for generating human-like whole-body movements. An efficient but precise residual policy is then learned to refine the GMT outputs to improve locomotion and incorporate object interaction. To further facilitate efficient training, we design (i) a point-cloud-based object tracking reward for smoother optimization, (ii) a contact reward that encourages accurate humanoid body-object interactions, and (iii) a curriculum-based virtual object controller to stabilize early training. We evaluate ResMimic in both simulation and on a real Unitree G1 humanoid. Results show substantial gains in task success, training efficiency, and robustness over strong baselines. Videos are available at https://resmimic.github.io/ .

---

## 3. Discrete scalar curvature as a weighted sum of Ollivier-Ricci curvatures

**论文链接:** [http://arxiv.org/abs/2510.04936v1](http://arxiv.org/abs/2510.04936v1)

**作者:** Abigail Hickok, Andrew J. Blumberg

**发布时间:** 2025-10-06

**备注:** 30 pages, 2 figures

### GPT解析

### 总结

该研究探讨了离散几何中Ricci曲率和标量曲率之间的关系，针对点云和图结构定义了Ollivier-Ricci曲率的标量版本，并证明了其在特定条件下的收敛性质。

### 背景

在离散几何中，传统的Ricci曲率和标量曲率需要被离散化以应用于点云和图结构。Ollivier-Ricci曲率是Ricci曲率的一种离散形式，而标量曲率在Riemannian流形中是Ricci曲率的迹。

### 目的

定义一种离散的标量曲率版本，研究其与连续标量曲率的关系，并探索Ollivier-Ricci曲率到Ricci曲率的收敛性。

### 方法

通过将标量曲率视为Ricci曲率的迹，定义了Ollivier-Ricci曲率的标量版本。研究最近邻图采样自流形时的收敛性，并证明Ollivier-Ricci曲率到Ricci曲率的收敛结果。

### 主要发现

所定义的标量Ollivier-Ricci曲率对于通过从流形采样获得的最近邻图会收敛到标量曲率。同时证明了关于Ollivier-Ricci曲率收敛到Ricci曲率的新结果。

### 结论

离散几何中可以合理地定义标量曲率概念，并且这些离散概念在适当的条件下会收敛到其连续对应物，为离散几何分析提供了理论基础。

### 翻译

我们研究了为点云和图定义的Ricci曲率和标量曲率的离散类比之间的关系。在离散设置中，Ricci曲率被Ollivier-Ricci曲率所替代。标量曲率可以作为Riemannian流形上Ricci曲率的迹来计算；这促使我们定义了一种新的Ollivier-Ricci曲率的标量版本。我们展示了我们的定义对于通过从流形采样获得的最近邻图会收敛到标量曲率。我们还证明了一些关于Ollivier-Ricci曲率收敛到Ricci曲率的新结果。


### 论文摘要

We study the relationship between discrete analogues of Ricci and scalar curvature that are defined for point clouds and graphs. In the discrete setting, Ricci curvature is replaced by Ollivier-Ricci curvature. Scalar curvature can be computed as the trace of Ricci curvature for a Riemannian manifold; this motivates a new definition of a scalar version of Ollivier-Ricci curvature. We show that our definition converges to scalar curvature for nearest neighbor graphs obtained by sampling from a manifold. We also prove some new results about the convergence of Ollivier-Ricci curvature to Ricci curvature.

---

## 4. Bridge Thinking and Acting: Unleashing Physical Potential of VLM with Generalizable Action Expert

**论文链接:** [http://arxiv.org/abs/2510.03896v1](http://arxiv.org/abs/2510.03896v1)

**作者:** Mingyu Liu, Zheng Huang, Xiaoyi Lin, Muzhi Zhu, Canyu Zhao, Zongze Du, Yating Wang, Haoyi Zhu, Hao Chen, Chunhua Shen

**发布时间:** 2025-10-04

### GPT解析

### 总结

本文提出了一种以可泛化动作专家为中心的新框架，通过稀疏3D轨迹作为中间表示，连接VLM的高级规划能力和低级物理动作模块，解决了传统VLA模型泛化能力差和双系统方法中语义歧义问题。

### 背景

Vision-Language Models (VLM)虽展现出强大的规划和推理能力，但在物理世界中应用面临挑战。传统Vision-Language-Action (VLA)模型受限于稀缺窄域数据导致泛化能力差；近期双系统方法虽解耦了思考与行动，但受限于动作模块的语义歧义，使得大规模跨任务训练不可行，且系统间合作机制不明确。

### 目的

解决传统VLA模型和双系统方法的局限性，提高系统在新环境中的适应性和泛化能力，并明确系统间的合作机制。

### 方法

引入以可泛化动作专家为中心的框架，使用稀疏3D轨迹作为中间表示；在规划阶段，VLM生成粗略3D路径点，由动作专家通过采样环境实时点云观测细化为密集可执行动作序列；提出'动作预训练，点云微调'新范式，结合VLM的广泛泛化能力和动作专家的细粒度动作级别泛化能力。

### 主要发现

新方法通过解耦高级规划和低级执行，有效解决了传统方法的语义歧义问题，提高了训练效率和鲁棒泛化能力。

### 结论

所提出的方法成功结合了VLM在视觉理解和规划方面的广泛泛化能力与动作专家的细粒度、动作级别的泛化能力，为物理世界中的智能行动提供了新思路。

### 翻译

尽管视觉语言模型(VLM)已展示了令人印象深刻的规划和推理能力，但这些能力在物理世界中的应用带来了重大挑战。整合推理和动作为一体的传统视觉语言动作(VLA)模型泛化能力差，受限于稀缺的窄域数据。虽然最近的双系统方法试图将'思考'与'行动'解耦，但它们常受限于动作模块内的语义歧义。这种歧义使得大规模、跨任务训练不可行。因此，这些系统通常在部署到新环境时需要对新收集的数据进行微调，且两个系统之间的合作机制仍然定义不明确。为解决这些局限性，我们首次引入了一个以可泛化动作专家为中心的框架。我们的方法使用稀疏3D轨迹作为中间表示，有效地连接了VLM的高级规划能力和低级物理动作模块。在规划阶段，VLM只需生成粗略的3D路径点。然后，这些路径点由我们的可泛化动作专家处理，通过采样环境的实时点云观测将其细化为密集的、可执行的动作序列。为提高训练效率和鲁棒泛化能力，我们引入了一种新颖的'动作预训练，点云微调'范式。我们的方法结合了VLM在视觉理解和规划方面的广泛泛化能力与动作专家的细粒度、动作级别泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决视觉语言模型（VLM）难以将强大的规划和推理能力转化为实际物理行动的问题。传统方法要么将推理和动作整合到单一架构中导致泛化能力差，要么试图分离'思考'和'行动'但面临语义歧义问题。这个问题在现实中非常重要，因为它限制了AI系统在机器人控制、自动驾驶等需要物理交互领域的应用，使VLM的能力主要局限于数字世界而无法影响物理环境。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性：传统VLA模型受限于稀缺数据导致泛化差，而双系统方法虽然分离了'思考'和'行动'，但动作模块仍需解释复杂语义信息，限制了大规模训练。作者借鉴了双系统框架和中间表示的概念，但创新性地使用稀疏3D轨迹作为明确接口，减轻了动作专家的语义负担。他们还参考了扩散模型架构，但提出了'动作预训练，点云微调'的新范式，使动作专家专注于几何精炼而非语义解释。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过稀疏3D轨迹作为高阶VLM规划和低阶动作专家之间的明确接口，实现真正的解耦。VLM只需生成简单的几何坐标（3D路径点），动作专家则专注于将这些路径点精炼为可执行动作序列。整体流程是：1) VLM接收语言指令和视觉输入，预测稀疏3D路径点；2) 将路径点从相机坐标系转换到机器人基座坐标系，用B样条插值形成连续轨迹；3) 动作专家接收引导姿态和实时点云观测，精炼为密集动作序列；4) 采用'动作预训练，点云微调'范式进行训练。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1) 引入围绕可泛化动作专家的框架，使用稀疏3D轨迹作为清晰接口；2) 提出'动作预训练，点云微调'训练策略；3) 在相机坐标系中直接预测路径点，保留VLM的视觉先验知识；4) 结合多种数据源并使用先进技术提高点云质量。相比之前的工作，本文方法完全解耦了规划和执行，解决了动作专家的语义负担问题，实现了真正的零样本部署，无需任务特定微调即可适应新环境。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于可泛化动作专家的新框架，通过稀疏3D轨迹作为接口解耦视觉语言模型的规划能力与低级物理动作，实现了无需任务特定微调的强泛化机器人控制。'}


### 论文摘要

Although Vision-Language Models (VLM) have demonstrated impressive planning and reasoning capabilities, translating these abilities into the physical world introduces significant challenges. Conventional Vision-Language-Action (VLA) models, which integrate reasoning and action into a monolithic architecture, generalize poorly because they are constrained by scarce, narrow-domain data. While recent dual-system approaches attempt to decouple "thinking" from "acting", they are often constrained by semantic ambiguities within the action module. This ambiguity makes large-scale, cross-task training infeasible. Consequently, these systems typically necessitate fine-tuning on newly collected data when deployed to novel environments, and the cooperation mechanism between the two systems remains ill-defined. To address these limitations, we introduce, for the first time, a framework centered around a generalizable action expert. Our approach utilizes sparse 3D trajectories as an intermediate representation, effectively bridging the high-level planning capabilities of the VLM with the low-level physical action module. During the planning phase, the VLM is only required to generate coarse 3D waypoints. These waypoints are then processed by our generalizable action expert, which refines them into dense, executable action sequences by sampling real-time point cloud observations of the environment. To promote training efficiency and robust generalization, we introduce a novel "Action Pre-training, Pointcloud Fine-tuning" paradigm. Our method combines the broad generalization capabilities of VLMs in visual understanding and planning with the fine-grained, action-level generalization of action expert.

---

## 5. Identification in source apportionment using geometry

**论文链接:** [http://arxiv.org/abs/2510.03616v1](http://arxiv.org/abs/2510.03616v1)

**作者:** Bora Jin, Abhirup Datta

**发布时间:** 2025-10-04

### GPT解析

### 总结

本研究解决了源解析分析中非负矩阵分解的非唯一性问题，提出了在更弱条件下可识别的源归因百分比矩阵估计方法，并通过几何估计量实现了一致的源解析结果。

### 背景

源解析分析旨在量化多种空气污染物观测浓度对特定来源的归因，传统方法通常使用非负矩阵分解(NMF)，但NMF具有非唯一性，依赖于不可验证的假设如稀疏性和难以解释的缩放。

### 目的

建立源归因百分比矩阵在更弱和更现实条件下的可识别性，提出无需稀疏性或参数分布假设的几何估计方法。

### 方法

将数据视为锥形锥中的一个点云，开发几何估计量来估计源归因百分比矩阵，该方法能够适应时空依赖性。

### 主要发现

源归因百分比矩阵的总体水平估计量是尺度不变的，即使NMF因子不可识别时也可识别；几何估计量是一致的，无需稀疏性或参数分布假设。

### 结论

所提出的几何估计方法在源解析分析中具有理论优势，数值实验验证了该方法的有效性。

### 翻译

源解析分析旨在量化观测到的多种空气污染物浓度对特定来源的归因，可以表述为非负矩阵分解问题。然而，NMF是非唯一的，通常依赖于不可验证的假设，如稀疏性和难以解释的缩放。在本研究中，我们在更弱和更现实的条件下建立了源归因百分比矩阵的可识别性。我们引入了这个矩阵的总体水平估计量，并证明它是尺度不变的，即使NMF因子不可识别时也是如此。将数据视为锥形锥中的一个点云，我们证明了源归因百分比矩阵的几何估计量是一致的，无需任何稀疏性或参数分布假设，同时能够适应时空依赖性。数值实验验证了这一理论。


### 论文摘要

Source apportionment analysis, which aims to quantify the attribution of observed concentrations of multiple air pollutants to specific sources, can be formulated as a non-negative matrix factorization (NMF) problem. However, NMF is non-unique and typically relies on unverifiable assumptions such as sparsity and uninterpretable scalings. In this manuscript, we establish identifiability of the source attribution percentage matrix under much weaker and more realistic conditions. We introduce the population-level estimand for this matrix, and show that it is scale-invariant and identifiable even when the NMF factors are not. Viewing the data as a point cloud in a conical hull, we show that a geometric estimator of the source attribution percentage matrix is consistent without any sparsity or parametric distributional assumptions, and while accommodating spatio-temporal dependence. Numerical experiments corroborate the theory.

---

## 6. Platonic Transformers: A Solid Choice For Equivariance

**论文链接:** [http://arxiv.org/abs/2510.03511v1](http://arxiv.org/abs/2510.03511v1)

**作者:** Mohammad Mohaiminul Islam, Rishabh Anand, David R. Wessels, Friso de Kruiff, Thijs P. Kuipers, Rex Ying, Clara I. Sánchez, Sharvaree Vadgama, Georg Bökman, Erik J. Bekkers

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文提出了一种名为Platonic Transformer的新型模型，解决了Transformer在处理几何对称性方面的局限性，同时保持了其高效性和灵活性。

### 背景

Transformer模型虽然广泛应用，但缺乏科学和计算机视觉中常见的几何对称性的归纳偏置。现有的等变方法通常通过复杂、计算密集型设计牺牲了Transformer的高效性和灵活性。

### 目的

解决等变性与效率/灵活性之间的权衡问题，开发一种能够处理几何对称性同时保持Transformer优势的模型。

### 方法

引入Platonic Transformer，通过定义相对于柏拉图立体对称群参考框架的注意力机制，诱导出有原则的权重共享方案，实现结合连续平移和柏拉图对称性的等变性，同时保留标准Transformer的精确架构和计算成本。

### 主要发现

该注意力机制在形式上等价于动态群卷积，使模型能够学习自适应几何滤波器，并实现高度可扩展的线性时间卷积变体。

### 结论

在计算机视觉(CIFAR-10)、3D点云(ScanObjectNN)和分子属性预测(QM9, OMol25)等多样化基准测试中，Platonic Transformer通过利用几何约束实现了具有竞争力的性能，且没有额外计算成本。

### 翻译

尽管Transformer应用广泛，但缺乏科学和计算机视觉中常见的几何对称性的归纳偏置。现有的等变方法通常通过复杂、计算密集型设计牺牲了Transformer的高效性和灵活性。我们引入Platonic Transformer来解决这种权衡。通过定义相对于柏拉图立体对称群参考框架的注意力，我们的方法诱导出一种有原则的权重共享方案。这实现了对连续平移和柏拉图对称性的组合等变性，同时保留了标准Transformer的精确架构和计算成本。此外，我们证明这种注意力在形式上等价于动态群卷积，这揭示了模型学习自适应几何滤波器的能力，并实现了一种高度可扩展的线性时间卷积变体。在计算机视觉(CIFAR-10)、3D点云(ScanObjectNN)和分子属性预测(QM9, OMol25)等多样化基准测试中，Platonic Transformer通过利用这些几何约束以无额外成本的方式实现了具有竞争力的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决的是如何在保持Transformer计算效率和灵活性的同时，为其添加几何对称性（等变性）的问题。这个问题在科学和计算机视觉领域非常重要，因为许多问题（如分子性质预测、3D点云处理）具有几何对称性，尊重这些对称性可以显著提高模型的性能、数据效率和鲁棒性，但现有的等变方法通常牺牲了Transformer的计算效率。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了Transformer在处理几何对称性方面的局限性，以及现有等变方法的不足。他们借鉴了Rotary Position Embeddings (RoPE)的工作，它已经为注意力机制提供了平移等变性；同时也借鉴了群等变卷积网络的思想。作者的核心思路是通过定义相对于柏拉图立体（Platonic solid）对称群参考帧的注意力，引入几何对称性，同时保持标准Transformer的计算图不变。他们通过权重共享和等变性线性层约束实现了这一目标。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过引入柏拉图立体的对称群作为参考帧，在保持标准Transformer架构的同时添加几何等变性。实现流程包括：1) 将输入特征提升为在群G上的函数；2) 修改RoPE操作使其依赖于参考帧；3) 在多个参考帧上并行计算注意力分数；4) 应用softmax得到注意力系数；5) 计算每个参考帧的输出；6) 确保所有线性层都是等变的；7) 选择柏拉图立体的对称群作为参考帧。整个过程保持了标准Transformer的计算图和计算成本。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次在不改变标准Transformer架构的情况下实现几何等变性；2) 使用柏拉图立体的对称群作为结构化参考帧；3) 设计了权重共享的RoPE注意力机制；4) 揭示了注意力与动态群卷积的等价性；5) 提出了线性时间复杂度的卷积变体。相比之前的工作，Platonic Transformer不需要修改底层注意力机制，避免了计算开销，同时实现了完全等变的注意力（而非不变性注意力），并以更低成本超过了帧平均方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Platonic Transformer通过引入柏拉图立体对称群的参考帧和权重共享机制，首次在不改变标准Transformer架构和计算成本的情况下，实现了对连续平移和离散旋转-反射的等变性，解决了等变性与计算效率之间的权衡问题。'}


### 论文摘要

While widespread, Transformers lack inductive biases for geometric symmetries common in science and computer vision. Existing equivariant methods often sacrifice the efficiency and flexibility that make Transformers so effective through complex, computationally intensive designs. We introduce the Platonic Transformer to resolve this trade-off. By defining attention relative to reference frames from the Platonic solid symmetry groups, our method induces a principled weight-sharing scheme. This enables combined equivariance to continuous translations and Platonic symmetries, while preserving the exact architecture and computational cost of a standard Transformer. Furthermore, we show that this attention is formally equivalent to a dynamic group convolution, which reveals that the model learns adaptive geometric filters and enables a highly scalable, linear-time convolutional variant. Across diverse benchmarks in computer vision (CIFAR-10), 3D point clouds (ScanObjectNN), and molecular property prediction (QM9, OMol25), the Platonic Transformer achieves competitive performance by leveraging these geometric constraints at no additional cost.

---

## 7. Warm-Starting Optimization-Based Motion Planning for Robotic Manipulators via Point Cloud-Conditioned Flow Matching

**论文链接:** [http://arxiv.org/abs/2510.03460v1](http://arxiv.org/abs/2510.03460v1)

**作者:** Sibo Tian, Minghui Zheng, Xiao Liang

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文提出了一种基于学习的机器人运动生成方法，利用流匹配模型以单视角点云为条件学习优化初始化，解决了传统方法在高维空间和复杂环境中的效率问题。

### 背景

快速机器人运动生成在人机协作系统中至关重要，机器人需要实时响应动态环境并重新规划运动以确保安全交互和高效任务执行。

### 目的

解决当前基于采样的运动规划器难以扩展到高维配置空间和基于优化的规划器初始化敏感的问题，提高机器人运动生成的效率和成功率。

### 方法

提出一种基于流匹配模型的学习方法，以单视角点云为条件学习优化初始化的次优解，直接从单视角深度相机输入生成可行轨迹，无需先验环境知识。

### 主要发现

在UR5e机械臂上的模拟研究表明，所提出的生成初始化器本身具有高成功率，显著提高了轨迹优化的成功率，需要更少的优化迭代，并对未见环境表现出强大的泛化能力。

### 结论

该方法有效地解决了机器人运动生成中的初始化问题，提高了轨迹优化的效率和成功率，具有良好的泛化能力。

### 翻译

在杂乱工作空间中对UR5e机械臂进行的模拟研究表明，所提出的生成初始化器本身具有高成功率，与传统和基于学习的基准初始化器相比显著提高了轨迹优化的成功率，需要更少的优化迭代，并对未见环境表现出强大的泛化能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人运动规划中的初始化问题。优化-based方法对初始化敏感，容易陷入局部最小值；而采样-based方法在高维空间扩展困难，生成的路径需要后处理。这个问题在人机协作系统中至关重要，因为机器人需要实时响应动态环境，持续观察并重新规划运动，确保安全互动和高效执行任务，特别是在共享工作空间高度动态的场景中。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统运动规划方法的局限性，然后回顾了现有的神经增强方法，包括偏置采样、模仿学习、生成模型和优化热启动等。他们借鉴了Flow Matching这一生成模型，它学习从简单分布到数据分布的最优直线转换，比扩散模型需要更少的生成步骤。结合PointNet++处理点云数据和SE-Transformer架构，设计了一个条件Flow Matching模型，以单视角点云为条件，无需障碍物位置的先验知识。同时使用了cuRobo优化器进行并行优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用Flow Matching模型学习生成近优的轨迹初始化种子，同时生成多个多样性的种子，提高优化成功率并减少迭代次数。方法以单视角点云为条件，无需障碍物先验知识。整体流程：输入单视角点云、起始和目标配置；使用PointNet++提取点云特征；通过SE-Transformer处理轨迹表示和条件信息；通过Flow Matching生成多个初始种子；使用cuRobo优化器并行优化；输出平滑、无碰撞的可行轨迹。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 基于Flow Matching的随机神经初始化器，可同时生成多个多样性的近优种子；2) 以单视角点云为条件，无需障碍物先验知识；3) 结合GPU加速优化器支持批量并行优化；4) 在复杂环境中展示强大泛化能力。相比之前的工作，它比传统初始化方法显著提高成功率和收敛速度；比确定性神经初始化器能生成多样种子避免局部最小值；比扩散模型推理速度更快；比需要环境先验知识的方法可直接从深度观测生成轨迹。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于点云条件Flow Matching的机器人运动规划初始化方法，通过生成多个多样性的近优种子显著提高了轨迹优化的成功率和收敛速度，使机器人能够在动态环境中快速、安全地重新规划运动。'}


### 论文摘要

Rapid robot motion generation is critical in Human-Robot Collaboration (HRC) systems, as robots need to respond to dynamic environments in real time by continuously observing their surroundings and replanning their motions to ensure both safe interactions and efficient task execution. Current sampling-based motion planners face challenges in scaling to high-dimensional configuration spaces and often require post-processing to interpolate and smooth the generated paths, resulting in time inefficiency in complex environments. Optimization-based planners, on the other hand, can incorporate multiple constraints and generate smooth trajectories directly, making them potentially more time-efficient. However, optimization-based planners are sensitive to initialization and may get stuck in local minima. In this work, we present a novel learning-based method that utilizes a Flow Matching model conditioned on a single-view point cloud to learn near-optimal solutions for optimization initialization. Our method does not require prior knowledge of the environment, such as obstacle locations and geometries, and can generate feasible trajectories directly from single-view depth camera input. Simulation studies on a UR5e robotic manipulator in cluttered workspaces demonstrate that the proposed generative initializer achieves a high success rate on its own, significantly improves the success rate of trajectory optimization compared with traditional and learning-based benchmark initializers, requires fewer optimization iterations, and exhibits strong generalization to unseen environments.

---

## 8. Unified Unsupervised Anomaly Detection via Matching Cost Filtering

**论文链接:** [http://arxiv.org/abs/2510.03363v1](http://arxiv.org/abs/2510.03363v1)

**作者:** Zhe Zhang, Mingxiu Cai, Gaochang Wu, Jing Zhang, Lingqiao Liu, Dacheng Tao, Tianyou Chai, Xiatian Zhu

**发布时间:** 2025-10-03

**备注:** 63 pages (main paper and supplementary material), 39 figures, 58  tables. Submitted to IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI)

### GPT解析

### 总结

本文提出了一种统一成本过滤(UCF)框架，用于增强无监督异常检测(UAD)方法，解决了匹配噪声问题，统一了单模态和多模态UAD场景。

### 背景

无监督异常检测(UAD)仅使用正常训练数据识别异常，广泛应用于工业检测和医疗分析等领域，但异常样本稀缺。现有方法基于重建或嵌入，进行图像或特征级匹配，但匹配噪声被忽视，限制了检测能力。从单模态RGB UAD发展到多模态场景(RGB-3D, RGB-Text)，但这些研究方向相互隔离。

### 目的

从匹配角度提出统一单模态和多模态UAD，开发一种通用的后处理优化框架来增强任何UAD模型的异常检测能力。

### 方法

提出统一成本过滤(UCF)框架，通过构建成本体积将测试样本与正常样本进行匹配，然后使用具有来自测试样本多层注意力指导的可学习过滤模块，减轻匹配噪声并突出细微异常。

### 主要发现

在22个多样化基准测试上进行的综合实验证明，UCF能够增强各种UAD方法，在单模态(RGB)和多模态(RGB-3D, RGB-Text)UAD场景中持续达到新的最先进结果。

### 结论

UCF框架有效解决了UAD中的匹配噪声问题，统一了单模态和多模态UAD方法，代码和模型将在https://github.com/ZHE-SAPI/CostFilter-AD发布。

### 翻译

无监督异常检测(UAD)旨在仅使用正常训练数据识别图像和像素级异常，广泛应用于工业检测和医疗分析等领域，在这些领域中由于隐私问题和冷启动限制，异常样本稀缺。现有方法，无论是基于重建(恢复正常对应物)还是基于嵌入(预训练表示)，本质上都进行图像或特征级匹配以生成异常图。尽管如此，匹配噪声在很大程度上被忽视，限制了它们的检测能力。超越早期对单模态RGB-based UAD的关注，最近的进展扩展到多模态场景，例如RGB-3D和RGB-Text，这得益于点云传感和视觉-语言模型。尽管存在共同挑战，但这些研究方向在很大程度上相互隔离，阻碍了全面理解和知识转移。在本文中，我们从匹配角度倡导统一单模态和多模态设置的UAD。在此见解下，我们提出了统一成本过滤(UCF)，这是一个通用的后处理优化框架，用于优化任何UAD模型的异常成本体积。成本体积是通过将测试样本与来自相同或不同模态的正常样本进行匹配而构建的，然后是一个可学习的过滤模块，具有来自测试样本的多层注意力指导，减轻匹配噪声并突出细微异常。在22个多样化基准测试上的综合实验证明了UCF在增强各种UAD方法方面的有效性，在单模态(RGB)和多模态(RGB-3D, RGB-Text)UAD场景中持续达到新的最先进结果。代码和模型将在https://github.com/ZHE-SAPI/CostFilter-AD发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决无监督异常检测（UAD）中的匹配噪声问题。这个问题在现实中非常重要，因为UAD广泛应用于工业检测和医疗分析等领域，这些领域由于隐私问题和冷启动约束，异常样本稀缺。匹配噪声会导致边界模糊、假阳性和假阴性，特别是对于细微异常、低对比度或接近正常区域的问题，严重影响检测准确性。同时，随着UAD从单模态扩展到多模态场景，这些挑战变得更加复杂，但不同模态的研究大多孤立进行，阻碍了全面理解和知识转移。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从匹配的视角重新概念化了单模态和多模态UAD，意识到匹配噪声是一个被忽视但关键的因素。他们受到立体匹配、深度估计、光流估计和光场渲染等领域中匹配成本过滤（成本体积过滤）概念的启发，将UAD重新构建为三步范式：特征提取、异常成本体积构建和异常成本体积过滤。他们借鉴了现有工作中的预训练模型（如CLIP、PointMAE）进行特征提取，多模态表示和特征匹配技术，以及3D U-Net架构和注意力机制，但创新性地将这些元素整合到一个统一的框架中专门解决匹配噪声问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将异常检测重新概念化为匹配成本过滤过程，通过多模板和多模态匹配构建异常成本体积，然后使用双流注意力引导的过滤网络来减少匹配噪声，同时保留边缘结构和细微异常的线索。整体实现流程分为四步：1) 特征提取：使用模态特定的预训练编码器从输入和参考模板中提取层次特征；2) 异常成本体积构建：通过执行补丁级内模态或跨模态匹配来构建多层成本体积；3) 异常成本体积过滤：引入一个过滤网络，以从测试样本中获得的多层注意力引导，逐步以粗到细的方式聚合多个模板的证据；4) 检测输出生成：通过沿匹配维度进行全局最小池化，然后通过卷积层和softmax生成正常-异常得分图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 重新概念化单模态和多模态UAD，明确解决内在匹配噪声问题；2) 提出统一成本过滤（UCF）框架，使用多层输入观察作为注意力查询来指导匹配去噪；3) 作为通用即插即用方法，可以灵活地构建和过滤来自RGB特征以及重建或基于嵌入的RGB、文本和点云表示的匹配成本体积；4) 提出类感知适配器，使用软分类logits动态调整分割损失，优先处理具有挑战性的样本并提高泛化能力。相比之前的工作，本文扩展到了多模态场景（RGB-3D和RGB-Text），而之前大多只关注单模态RGB UAD；本文明确解决了被忽略的匹配噪声问题；提供了一个统一的视角和方法，而之前的方法通常各自独立发展。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了统一成本过滤（UCF）框架，通过减少匹配噪声并利用多模态信息，显著提升了无监督异常检测在单模态和多模态场景下的性能，为异常检测提供了一个通用的即插即用解决方案。'}


### 论文摘要

Unsupervised anomaly detection (UAD) aims to identify image- and pixel-level anomalies using only normal training data, with wide applications such as industrial inspection and medical analysis, where anomalies are scarce due to privacy concerns and cold-start constraints. Existing methods, whether reconstruction-based (restoring normal counterparts) or embedding-based (pretrained representations), fundamentally conduct image- or feature-level matching to generate anomaly maps. Nonetheless, matching noise has been largely overlooked, limiting their detection ability. Beyond earlier focus on unimodal RGB-based UAD, recent advances expand to multimodal scenarios, e.g., RGB--3D and RGB--Text, enabled by point cloud sensing and vision--language models. Despite shared challenges, these lines remain largely isolated, hindering a comprehensive understanding and knowledge transfer. In this paper, we advocate unified UAD for both unimodal and multimodal settings in the matching perspective. Under this insight, we present Unified Cost Filtering (UCF), a generic post-hoc refinement framework for refining anomaly cost volume of any UAD model. The cost volume is constructed by matching a test sample against normal samples from the same or different modalities, followed by a learnable filtering module with multi-layer attention guidance from the test sample, mitigating matching noise and highlighting subtle anomalies. Comprehensive experiments on 22 diverse benchmarks demonstrate the efficacy of UCF in enhancing a variety of UAD methods, consistently achieving new state-of-the-art results in both unimodal (RGB) and multimodal (RGB--3D, RGB--Text) UAD scenarios. Code and models will be released at https://github.com/ZHE-SAPI/CostFilter-AD.

---

## 9. Diffusion^2: Turning 3D Environments into Radio Frequency Heatmaps

**论文链接:** [http://arxiv.org/abs/2510.02274v2](http://arxiv.org/abs/2510.02274v2)

**作者:** Kyoungjun Park, Yifan Yang, Changhan Ge, Lili Qiu, Shiqi Jiang

**发布时间:** 2025-10-02

### GPT解析

### 总结

Diffusion^2是一种基于扩散的射频信号传播建模方法，使用3D点云支持从Wi-Fi到毫米波的广泛频率范围，能够准确预测复杂环境中的RF信号行为。

### 背景

射频信号传播建模对于理解环境至关重要，能提供比RGB相机更有价值的洞察，RGB相机受限于可见光谱、镜头覆盖和遮挡。RF信号还支持无线诊断、部署和优化。

### 目的

开发一种能够准确预测复杂环境中射频信号传播的方法，解决信号与障碍物相互作用（如吸收和反射）带来的挑战。

### 方法

提出Diffusion^2方法，使用3D点云建模RF信号传播；开发RF-3D编码器从3D数据中捕获RF相关特征，封装3D几何复杂性和信号特定细节；通过多尺度嵌入模拟实际RF信号传播过程。

### 主要发现

Diffusion^2能够准确估计各种频率波段和环境条件下的RF信号行为，误差仅为1.9 dB，比现有方法快27倍，标志着该领域的重大进步。

### 结论

Diffusion^2为射频信号传播建模提供了有效解决方案，在准确性和计算效率方面均有显著提升。

### 翻译

射频信号传播建模对于理解环境至关重要，因为RF信号能提供超越RGB相机能力范围的宝贵见解，RGB相机受限于可见光谱、镜头覆盖和遮挡。RF信号传播建模也有助于支持无线诊断、部署和优化。然而，由于与障碍物的相互作用（如吸收和反射），在复杂环境中准确预测RF信号仍然是一个挑战。我们提出了Diffusion^2，一种基于扩散的方法，使用3D点云建模从Wi-Fi到毫米波广泛频率范围内的RF信号传播。为了从3D数据中有效捕获RF相关特征，我们提出了RF-3D编码器，它封装了3D几何的复杂性以及信号特定细节。这些特征经过多尺度嵌入，以模拟实际的RF信号传播过程。基于合成和真实世界测量的评估表明，Diffusion^2准确估计了各种频率波段和环境条件下的RF信号行为，误差仅为1.9 dB，比现有方法快27倍，标志着该领域的重大进步。更多信息请访问https://rfvision-project.github.io/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何准确预测复杂3D环境中的射频(RF)信号传播问题。这个问题很重要，因为RF信号能提供比RGB相机更丰富的环境洞察，对无线网络优化、智能环境部署和物联网应用等至关重要。现有方法要么需要大量预测量数据，要么计算效率低下，难以在实际应用中部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到生成式AI最新进展(如Sora)的启发，思考是否将生成式AI扩展到可见光谱外用于RF预测。他们选择扩散模型作为基础，因为其能处理环境不确定性和提供良好可控性。作者创新设计了RF-3D编码器和RF-3D配对块，借鉴了扩散模型在图像生成领域的成功应用和3D场景理解技术，结合了物理模型和机器学习方法的优势，针对RF信号预测问题进行了专门改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用扩散模型将3D环境模型转换为RF热图，通过将复杂信号传播问题分解为多个去噪步骤来学习RF传播模式。整体流程包括：1)用智能手机采集3D环境模型和少量RF预测量数据；2)通过RF-3D编码器提取3D几何、2D图像和RF信号特征；3)执行前向扩散过程(添加噪声)和反向扩散过程(条件引导去噪)；4)使用RF-3D配对块融合特征与预测；5)训练网络并生成RF热图；6)对动态场景扩展为视频扩散。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个基于扩散模型的RF信号传播估计方法；2)RF-3D编码器有效提取多模态特征；3)RF-3D配对块实现跨模态融合；4)支持视频扩散适应动态环境；5)高精度且仅需少量预测量数据。相比之前工作，Diffusion2数据需求大幅降低(只需15个测量点vs数千个)，计算效率提高27倍以上，支持多频率和动态场景，无需环境变化时重新训练，泛化能力更强。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Diffusion2首次将扩散模型应用于3D环境的射频信号传播预测，通过创新的RF-3D编码器和配对块设计，实现了仅需少量预测量数据的高精度、高效率RF热图生成，支持多频率和动态场景，为无线网络优化和智能环境部署提供了革命性工具。'}


### 论文摘要

Modeling radio frequency (RF) signal propagation is essential for understanding the environment, as RF signals offer valuable insights beyond the capabilities of RGB cameras, which are limited by the visible-light spectrum, lens coverage, and occlusions. It is also useful for supporting wireless diagnosis, deployment, and optimization. However, accurately predicting RF signals in complex environments remains a challenge due to interactions with obstacles such as absorption and reflection. We introduce Diffusion^2, a diffusion-based approach that uses 3D point clouds to model the propagation of RF signals across a wide range of frequencies, from Wi-Fi to millimeter waves. To effectively capture RF-related features from 3D data, we present the RF-3D Encoder, which encapsulates the complexities of 3D geometry along with signal-specific details. These features undergo multi-scale embedding to simulate the actual RF signal dissemination process. Our evaluation, based on synthetic and real-world measurements, demonstrates that Diffusion^2 accurately estimates the behavior of RF signals in various frequency bands and environmental conditions, with an error margin of just 1.9 dB and 27x faster than existing methods, marking a significant advancement in the field. Refer to https://rfvision-project.github.io/ for more information.

---

## 10. Matching the Optimal Denoiser in Point Cloud Diffusion with (Improved) Rotational Alignment

**论文链接:** [http://arxiv.org/abs/2510.03335v1](http://arxiv.org/abs/2510.03335v1)

**作者:** Ameya Daigavane, YuQing Xie, Bodhi P. Vani, Saeed Saremi, Joseph Kleinhenz, Tess Smidt

**发布时间:** 2025-10-02

**备注:** under review

### GPT解析

### 总结

本研究探讨了扩散模型在点云数据训练中的旋转对齐问题，分析了Kabsch-Umeyama算法对齐步骤的有效性，并提出了一种在噪声水平较低时对最优去噪器的更好近似方法。

### 背景

扩散模型是一类流行的生成模型，通过逆转从目标数据分布开始的加噪过程进行训练。当为分子和蛋白质等点云训练扩散模型时，通常无法分配规范的方向。为了捕获这种对称性，真实数据样本通常通过从SO(3)均匀采样的随机旋转进行增强。然后，在计算损失之前，去噪预测通常通过Kabsch-Umeyama算法旋转对齐到真实样本。

### 目的

理解旋转对齐步骤对扩散模型训练的影响，并探索在噪声水平较低时对最优去噪器的更好近似方法。

### 方法

研究表明最优去噪器可以用SO(3)上的矩阵Fisher分布表示。对齐对应于采样该分布的模式，并且对于小噪声水平而言是对齐的零阶近似。基于这一观点，研究人员推导出在噪声水平极限下对最优去噪器的更好近似方法。

### 主要发现

对齐步骤对应于矩阵Fisher分布的采样模式；对于小噪声水平，对齐是零阶近似，解释了其有效性；提出了在噪声水平较低时对最优去噪器的更好近似方法。

### 结论

对齐步骤对于扩散模型训练中最相关的噪声水平来说通常是一个'足够好'的近似。

### 翻译

扩散模型是一类流行的生成模型，经过训练可以逆转从目标数据分布开始的加噪过程。训练扩散模型包括学习如何在不同的噪声水平上去噪样本。当为分子和蛋白质等点云训练扩散模型时，通常无法分配规范的方向。为了捕获这种对称性，真实数据样本通常通过从SO(3)均匀采样的随机旋转进行增强。然后，在计算损失之前，去噪预测通常通过Kabsch-Umeyama算法旋转对齐到真实样本。然而，这种对齐步骤的影响尚未得到充分研究。在这里，我们表明最优去噪器可以用SO(3)上的矩阵Fisher分布表示。对齐对应于采样该分布的模式，并且对于小噪声水平而言是对齐的零阶近似，解释了其有效性。我们基于这一观点推导出在噪声水平极限下对最优去噪器的更好近似方法。我们的实验强调，对于扩散模型训练中最相关的噪声水平，对齐通常是一个'足够好'的近似。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决点云扩散模型训练中的旋转对齐问题。在处理分子、蛋白质等3D点云数据时，没有标准方向可分配，通常通过随机旋转增强数据，并在训练中使用Kabsch-Umeyama算法对齐去噪预测与真实样本。这个问题很重要，因为旋转对称性处理不当会影响模型性能，而现有方法缺乏对对齐步骤效果的理论分析，不清楚它是否引入偏差以及如何改进。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了旋转增强数据上训练扩散模型的挑战，推导了最优去噪器的数学表达式，证明其可表示为SO(3)上矩阵Fisher分布的期望。他们发现传统旋转对齐对应于该分布的众数，是小噪声水平下的零阶近似。基于此，他们使用Laplace方法推导了更高阶的校正项。作者借鉴了Karras等人的扩散模型理论、Kabsch-Umeyama算法、矩阵Fisher分布研究以及在分子结构上应用扩散模型的相关工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是最优去噪器可表示为旋转矩阵的期望，该旋转服从矩阵Fisher分布；传统对齐使用分布众数作为近似，而作者添加更高阶校正项以更好近似最优去噪器。流程包括：1)获取并旋转增强点云数据；2)在不同噪声水平下添加噪声；3)使用神经网络学习去噪器；4)计算最优旋转矩阵R*；5)应用零阶(直接对齐)、一阶(R*+σ²B1)或二阶(R*+σ²B1+σ⁴B2)校正；6)使用改进损失函数训练网络；7)训练好的模型用于生成新点云结构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次分析旋转对齐的理论基础，证明最优去噪器是矩阵Fisher分布的期望；2)提出高阶校正方法改进对齐估计；3)通过数值实验和实际应用验证方法。相比之前工作，本文提供了对齐步骤的理论解释，分析了其潜在偏差，提出了自适应的、基于噪声水平的对齐改进方法，而非使用固定策略，同时保持了计算效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过理论分析揭示了点云扩散模型中旋转对齐的本质，并提出了一种基于矩阵Fisher分布的高阶校正方法，在保持计算效率的同时改进了去噪性能。'}


### 论文摘要

Diffusion models are a popular class of generative models trained to reverse a noising process starting from a target data distribution. Training a diffusion model consists of learning how to denoise noisy samples at different noise levels. When training diffusion models for point clouds such as molecules and proteins, there is often no canonical orientation that can be assigned. To capture this symmetry, the true data samples are often augmented by transforming them with random rotations sampled uniformly over $SO(3)$. Then, the denoised predictions are often rotationally aligned via the Kabsch-Umeyama algorithm to the ground truth samples before computing the loss. However, the effect of this alignment step has not been well studied. Here, we show that the optimal denoiser can be expressed in terms of a matrix Fisher distribution over $SO(3)$. Alignment corresponds to sampling the mode of this distribution, and turns out to be the zeroth order approximation for small noise levels, explaining its effectiveness. We build on this perspective to derive better approximators to the optimal denoiser in the limit of small noise. Our experiments highlight that alignment is often a `good enough' approximation for the noise levels that matter most for training diffusion models.

---

## 11. Video-LMM Post-Training: A Deep Dive into Video Reasoning with Large Multimodal Models

**论文链接:** [http://arxiv.org/abs/2510.05034v1](http://arxiv.org/abs/2510.05034v1)

**作者:** Yunlong Tang, Jing Bi, Pinxin Liu, Zhenyu Pan, Zhangyun Tan, Qianxiang Shen, Jiani Liu, Hang Hua, Junjia Guo, Yunzhong Xiao, Chao Huang, Zhiyuan Wang, Susan Liang, Xinyi Liu, Yizhi Song, Yuhe Nie, Jia-Xing Zhong, Bozheng Li, Daiqing Qi, Ziyun Zeng, Ali Vosoughi, Luchuan Song, Zeliang Zhang, Daiki Shimada, Han Liu, Jiebo Luo, Chenliang Xu

**发布时间:** 2025-10-06

**备注:** The 1st version

### GPT解析

### 总结

这篇调查论文首次全面审视了Video-LMMs（视频大型多模态模型）的训练后方法，提供了结构化分类法和关键设计原则，旨在推进视频理解领域的研究与实践。

### 背景

视频理解是计算机视觉中最具挑战性的前沿领域，需要模型推理复杂的时空关系、长期依赖和多模态证据。Video-LMMs整合视觉编码器与基于解码器的强大语言模型，已在视频理解任务中展现出显著能力。

### 目的

提供对Video-LMMs训练后方法的首次全面检查，涵盖三个基本支柱：监督微调与思维链、可验证目标的强化学习以及通过增强推理计算进行的测试时扩展。

### 方法

提出结构化分类法，阐明这些技术的作用、相互联系和视频特定适应，并解决时间定位、时空接地、长视频效率和 multimodal 证据集成等独特挑战。系统分析代表性方法，综合关键设计原则、见解和评估协议。

### 主要发现

确定了在奖励设计、可扩展性和成本性能优化方面的关键开放挑战。整理了必要的基准、数据集和指标，以促进对训练后效果的严格评估。

### 结论

该调查为研究人员和从业者提供了一个统一的框架，以推进Video-LMM的能力。相关资源和更新可在GitHub仓库获取。

### 翻译

视频理解代表了计算机视觉中最具挑战性的前沿，需要模型能够推理复杂的时空关系、长期依赖和多模态证据。最近出现的Video-LMMs（视频大型多模态模型）将视觉编码器与强大的基于解码器的语言模型相结合，在视频理解任务中展示了非凡能力。然而，将这些模型从基本感知系统转变为复杂推理引擎的关键阶段（训练后）在文献中仍然分散。本调查首次对Video-LMMs的训练后方法进行了全面检查，涵盖三个基本支柱：思维链监督微调（SFT）、可验证目标的强化学习（RL）以及通过增强推理计算进行的测试时扩展（TTS）。我们提出了一个结构化分类法，阐明了这些技术的作用、相互联系和视频特定适应，解决了时间定位、时空接地、长视频效率和 multimodal 证据集成等独特挑战。通过对代表性方法的系统分析，我们综合了关键设计原则、见解和评估协议，同时确定了在奖励设计、可扩展性和成本性能优化方面的关键开放挑战。我们还整理了必要的基准、数据集和指标，以促进对训练后效果的严格评估。本调查旨在为研究人员和从业者提供一个统一的框架，以推进Video-LMM的能力。额外资源和更新保存在：https://github.com/yunlong10/Awesome-Video-LMM-Post-Training


### 论文摘要

Video understanding represents the most challenging frontier in computer vision, requiring models to reason about complex spatiotemporal relationships, long-term dependencies, and multimodal evidence. The recent emergence of Video-Large Multimodal Models (Video-LMMs), which integrate visual encoders with powerful decoder-based language models, has demonstrated remarkable capabilities in video understanding tasks. However, the critical phase that transforms these models from basic perception systems into sophisticated reasoning engines, post-training, remains fragmented across the literature. This survey provides the first comprehensive examination of post-training methodologies for Video-LMMs, encompassing three fundamental pillars: supervised fine-tuning (SFT) with chain-of-thought, reinforcement learning (RL) from verifiable objectives, and test-time scaling (TTS) through enhanced inference computation. We present a structured taxonomy that clarifies the roles, interconnections, and video-specific adaptations of these techniques, addressing unique challenges such as temporal localization, spatiotemporal grounding, long video efficiency, and multimodal evidence integration. Through systematic analysis of representative methods, we synthesize key design principles, insights, and evaluation protocols while identifying critical open challenges in reward design, scalability, and cost-performance optimization. We further curate essential benchmarks, datasets, and metrics to facilitate rigorous assessment of post-training effectiveness. This survey aims to provide researchers and practitioners with a unified framework for advancing Video-LMM capabilities. Additional resources and updates are maintained at: https://github.com/yunlong10/Awesome-Video-LMM-Post-Training

---

## 12. Your Vision-Language Model Can't Even Count to 20: Exposing the Failures of VLMs in Compositional Counting

**论文链接:** [http://arxiv.org/abs/2510.04401v1](http://arxiv.org/abs/2510.04401v1)

**作者:** Xuyang Guo, Zekai Huang, Zhenmei Shi, Zhao Song, Jiahao Zhang

**发布时间:** 2025-10-06

### GPT解析

### 总结

视觉语言模型在多种任务上表现出色，但在计数物体方面存在局限性，特别是在多种形状组合的情况下。

### 背景

视觉语言模型已成为AI社区的关注焦点，它们在从网络上获取的大规模视觉语言数据上训练后展现出了令人印象深刻的能力，在图像理解、视频理解、复杂视觉推理和具身AI等多种任务上表现出色。

### 目的

探究视觉语言模型是否能正确计数物体。

### 方法

引入了VLMCountBench基准测试，采用极简设置只包含基本几何形状及其组合，专注于计数任务；采用严格的独立变量控制，系统地研究颜色、大小和提示改进等简单属性的影响。

### 主要发现

当只有一种形状类型存在时，VLMs能够可靠计数；但当多种形状类型组合时(即组合计数)，它们表现出重大失败。

### 结论

这揭示了当前视觉语言模型的基本经验局限性，并为未来研究指明了重要方向。

### 翻译

视觉语言模型已成为当今AI社区的关注焦点，它们在从网络上获取的大规模视觉语言数据上训练后展现出了令人印象深刻的能力。这些模型在图像理解、视频理解、复杂视觉推理和具身AI等多种任务上表现出色。尽管取得了这些显著成功，一个基本问题仍然存在：视觉语言模型能否正确计数物体？在本文中，我们引入了一个简单而有效的基准测试VLMCountBench，在极简设置下设计，仅包含基本几何形状(如三角形、圆形)及其组合，专注于计数任务而不受其他因素干扰。我们采用严格的独立变量控制，并在消融研究中系统地研究了颜色、大小和提示改进等简单属性的影响。我们的实证结果表明，当只有一种形状类型存在时，VLMs能够可靠计数，但当多种形状类型组合时(即组合计数)，它们表现出重大失败。这突显了当前VLMs的基本经验局限性，并为未来研究指明了重要方向。


### 论文摘要

Vision-Language Models (VLMs) have become a central focus of today's AI community, owing to their impressive abilities gained from training on large-scale vision-language data from the Web. These models have demonstrated strong performance across diverse tasks, including image understanding, video understanding, complex visual reasoning, and embodied AI. Despite these noteworthy successes, a fundamental question remains: Can VLMs count objects correctly? In this paper, we introduce a simple yet effective benchmark, VLMCountBench, designed under a minimalist setting with only basic geometric shapes (e.g., triangles, circles) and their compositions, focusing exclusively on counting tasks without interference from other factors. We adopt strict independent variable control and systematically study the effects of simple properties such as color, size, and prompt refinement in a controlled ablation. Our empirical results reveal that while VLMs can count reliably when only one shape type is present, they exhibit substantial failures when multiple shape types are combined (i.e., compositional counting). This highlights a fundamental empirical limitation of current VLMs and motivates important directions for future research.

---

## 13. ChronoEdit: Towards Temporal Reasoning for Image Editing and World Simulation

**论文链接:** [http://arxiv.org/abs/2510.04290v1](http://arxiv.org/abs/2510.04290v1)

**作者:** Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M. Alvarez, Jun Gao, Sanja Fidler, Zian Wang, Huan Ling

**发布时间:** 2025-10-05

**备注:** Project Page: https://research.nvidia.com/labs/toronto-ai/chronoedit

### GPT解析

### 总结

本文提出了ChronoEdit框架，将图像编辑重新构建为视频生成问题，通过时间推理确保编辑后物体的物理一致性，在视觉保真度和物理可能性方面超越了现有方法。

### 背景

大型生成模型在图像编辑和上下文图像生成方面取得了显著进展，但在确保物理一致性方面存在关键差距，编辑后的物体必须保持连贯性，这对世界模拟相关任务尤为重要。

### 目的

开发一个能够确保编辑物体物理一致性的图像编辑框架，特别适用于需要物理一致性的场景。

### 方法

ChronoEdit框架将输入和编辑后的图像视为视频的第一帧和最后一帧，利用预训练视频生成模型捕捉物体的外观和隐含物理规律；引入时间推理阶段，通过推理令牌联合去噪目标帧，想象合理的编辑轨迹约束解决方案；几步后丢弃推理令牌以避免全视频渲染的高计算成本。

### 主要发现

作者引入了PBench-Edit基准测试，包含需要物理一致性的图像-提示对，实验证明ChronoEdit在视觉保真度和物理可能性方面均超越了最先进的基线方法。

### 结论

ChronoEdit通过将图像编辑视为视频生成问题，有效解决了图像编辑中的物理一致性问题，框架的14B和2B变体代码和模型将在项目页面发布。

### 翻译

最近大型生成模型的进展显著推动了图像编辑和上下文图像生成，但在确保物理一致性方面仍存在关键差距，编辑后的物体必须保持连贯。这种能力对世界模拟相关任务尤为重要。本文提出了ChronoEdit框架，将图像编辑重新构建为视频生成问题。首先，ChronoEdit将输入和编辑后的图像视为视频的第一帧和最后一帧，使其能够利用大型预训练视频生成模型，这些模型不仅捕捉物体外观，还通过学习的时间一致性捕获隐含的运动和交互物理规律。其次，ChronoEdit引入了时间推理阶段，在推理时显式执行编辑。在此设置下，目标帧与推理令牌联合去噪，想象合理的编辑轨迹，将解决方案空间约束为物理可行的变换。几步后丢弃推理令牌以避免全视频渲染的高计算成本。为验证ChronoEdit，我们引入了PBench-Edit，一个新的需要物理一致性的上下文图像-提示对基准，并证明ChronoEdit在视觉保真度和物理可能性方面均超越了最先进的基线方法。ChronoEdit的14B和2B变体的代码和模型将在项目页面发布：https://research.nvidia.com/labs/toronto-ai/chronoedit


### 论文摘要

Recent advances in large generative models have significantly advanced image editing and in-context image generation, yet a critical gap remains in ensuring physical consistency, where edited objects must remain coherent. This capability is especially vital for world simulation related tasks. In this paper, we present ChronoEdit, a framework that reframes image editing as a video generation problem. First, ChronoEdit treats the input and edited images as the first and last frames of a video, allowing it to leverage large pretrained video generative models that capture not only object appearance but also the implicit physics of motion and interaction through learned temporal consistency. Second, ChronoEdit introduces a temporal reasoning stage that explicitly performs editing at inference time. Under this setting, the target frame is jointly denoised with reasoning tokens to imagine a plausible editing trajectory that constrains the solution space to physically viable transformations. The reasoning tokens are then dropped after a few steps to avoid the high computational cost of rendering a full video. To validate ChronoEdit, we introduce PBench-Edit, a new benchmark of image-prompt pairs for contexts that require physical consistency, and demonstrate that ChronoEdit surpasses state-of-the-art baselines in both visual fidelity and physical plausibility. Code and models for both the 14B and 2B variants of ChronoEdit will be released on the project page: https://research.nvidia.com/labs/toronto-ai/chronoedit

---

## 14. Harnessing Synthetic Preference Data for Enhancing Temporal Understanding of Video-LLMs

**论文链接:** [http://arxiv.org/abs/2510.03955v1](http://arxiv.org/abs/2510.03955v1)

**作者:** Sameep Vani, Shreyas Jena, Maitreya Patel, Chitta Baral, Somak Aditya, Yezhou Yang

**发布时间:** 2025-10-04

**备注:** 17 pages, 9 figures, 6 tables. Presents TimeWarp, a synthetic  preference data framework to improve temporal understanding in Video-LLMs,  showing consistent gains across seven benchmarks. Includes supplementary  material in the Appendix

### GPT解析

### 总结

本文提出了TimeWarp方法，通过创建有针对性的合成时间数据集来微调视频大语言模型(Video-LLMs)，提高其在时间理解任务上的性能。

### 背景

视频大语言模型在一般视频理解基准测试中表现良好，但在需要细粒度时间理解的任务上表现不佳。这种局限性源于当前微调数据集中缺乏视觉复杂性和时间细微差别，导致模型过度依赖基于语言的推理。

### 目的

创建一个系统性的方法来生成针对性的合成时间数据集，微调Video-LLMs的响应，使其专注于给定的输入视频，提高模型对视频动态的时间理解能力。

### 方法

提出TimeWarp方法，创建了一个大规模的偏好数据集，捕捉通常被忽视的复杂时间动态，将模型响应基于视觉和时间信息进行训练。

### 主要发现

将TimeWarp方法应用于现有模型后，显著提高了时间理解基准测试的性能，在七个基准测试中实现了性能的绝对提升。

### 结论

TimeWarp创建的数据集在推进Video-LLMs的时间理解方面是有效的，能够帮助模型更好地关注视频内容而非过度依赖语言推理。

### 翻译

虽然视频大语言模型(Video-LLMs)在一般视频理解基准测试中表现出色，特别是在视频字幕和描述任务上，但在需要细粒度时间理解的任务上却表现不佳。这种局限性源于当前微调数据集中缺乏视觉复杂性和时间细微差别，导致这些模型过度依赖基于语言的推理，而不是真正理解视频动态。在这项工作中，我们提出了TimeWarp，这是一种系统性的方法，用于创建有针对性的合成时间数据集，以微调模型响应，鼓励模型专注于给定的输入视频。我们介绍了一个使用TimeWarp创建的大规模偏好数据集，捕捉了通常被忽视的复杂时间动态，将模型响应基于视觉和时间信息。我们证明，当我们的方法应用于现有模型时，它显著提高了时间理解基准测试的性能，突显了我们提出的数据集在推进Video-LLMs时间理解方面的有效性，在七个基准测试中实现了性能的绝对提升。代码可在https://github.com/sameepv21/timewarp获取。


### 论文摘要

While Video Large Language Models (Video-LLMs) have demonstrated remarkable performance across general video understanding benchmarks-particularly in video captioning and descriptive tasks-they consistently underperform on tasks that require fine-grained temporal understanding. This limitation arises due to the lack of visual complexity and temporal nuance in current fine-tuning datasets, leading these models to rely heavily on language-based reasoning rather than truly understanding video dynamics. In this work, we propose TimeWarp, a systematic method to create a targeted synthetic temporal dataset to fine-tune the model's responses to encourage it to focus on the given input video. We introduce a large-scale preference dataset, created using TimeWarp, that captures intricate temporal dynamics often overlooked, grounding the model's responses to visual and temporal information. We demonstrate that when our method is applied to existing models, it significantly improves performance on temporal understanding benchmarks, highlighting the effectiveness of our proposed datasets in advancing temporal understanding in Video-LLMs, resulting in an absolute improvement in performance across seven benchmarks. Code is available at https://github.com/sameepv21/timewarp.

---

## 15. 论文ID: 2510.03885v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.03885v1.json'

---

## 16. EmbodiSwap for Zero-Shot Robot Imitation Learning

**论文链接:** [http://arxiv.org/abs/2510.03706v1](http://arxiv.org/abs/2510.03706v1)

**作者:** Eadom Dessalene, Pavan Mantripragada, Michael Maynord, Yiannis Aloimonos

**发布时间:** 2025-10-04

**备注:** Video link:  https://drive.google.com/file/d/1UccngwgPqUwPMhBja7JrXfZoTquCx_Qe/view?usp=sharing

### GPT解析

### 总结

本研究介绍了一种名为EmbodiSwap的方法，用于在人类视频上生成逼真的合成机器人覆盖层，并利用这种方法进行零样本模仿学习，成功训练了闭环机器人操作策略，创新性地将V-JEPA视觉模型应用于机器人模仿学习领域。

### 背景

在机器人学习领域，存在野生自中心人类视频与目标机器人 embodiment 之间的差距，需要有效的方法来弥合这一差距。

### 目的

开发一种能够生成逼真合成机器人覆盖层的方法，并利用该方法进行零样本模仿学习训练，实现机器人操作策略的有效学习。

### 方法

提出EmbodiSwap方法生成合成机器人覆盖层，在生成的数据上训练闭环机器人操作策略，创新性地使用V-JEPA作为视觉主干，将其从视频理解领域重新应用于合成机器人视频的模仿学习。

### 主要发现

采用V-JEPA作为视觉主干比传统机器人领域使用的替代视觉主干表现更好；在真实世界测试中，零样本训练的V-JEPA模型达到82%的成功率，优于少样本训练的π₀网络以及在EmbodiSwap生成的数据上训练的π₀。

### 结论

EmbodiSwap方法结合V-JEPA视觉模型在机器人模仿学习领域取得了显著成果，通过发布代码、数据集和模型检查点，促进了该领域的可复现研究和广泛采用。

### 翻译

我们介绍了EmbodiSwap - 一种用于在人类视频上生成逼真合成机器人覆盖层的方法。我们将EmbodiSwap用于零样本模仿学习，弥合了野生自中心人类视频与目标机器人 embodiment 之间的差距。我们在EmbodiSwap生成的数据上训练闭环机器人操作策略。我们创新性地使用V-JEPA作为视觉主干，将V-JEPA从视频理解领域重新用于合成机器人视频的模仿学习。采用V-JEPA比机器人领域更传统使用的替代视觉主干表现更好。在真实世界测试中，我们的零样本训练的V-JEPA模型达到82%的成功率，优于少样本训练的π₀网络以及在EmbodiSwap生成的数据上训练的π₀。我们发布了(i)用于生成合成机器人覆盖层的代码，输入为人类视频和任意机器人URDF，生成机器人数据集，(ii)在EPIC-Kitchens、HOI4D和Ego4D上合成的机器人数据集，以及(iii)模型检查点和推理代码，以促进可复现研究和广泛采用。


### 论文摘要

We introduce EmbodiSwap - a method for producing photorealistic synthetic robot overlays over human video. We employ EmbodiSwap for zero-shot imitation learning, bridging the embodiment gap between in-the-wild ego-centric human video and a target robot embodiment. We train a closed-loop robot manipulation policy over the data produced by EmbodiSwap. We make novel use of V-JEPA as a visual backbone, repurposing V-JEPA from the domain of video understanding to imitation learning over synthetic robot videos. Adoption of V-JEPA outperforms alternative vision backbones more conventionally used within robotics. In real-world tests, our zero-shot trained V-JEPA model achieves an $82\%$ success rate, outperforming a few-shot trained $\pi_0$ network as well as $\pi_0$ trained over data produced by EmbodiSwap. We release (i) code for generating the synthetic robot overlays which takes as input human videos and an arbitrary robot URDF and generates a robot dataset, (ii) the robot dataset we synthesize over EPIC-Kitchens, HOI4D and Ego4D, and (iii) model checkpoints and inference code, to facilitate reproducible research and broader adoption.

---

## 17. FrameOracle: Learning What to See and How Much to See in Videos

**论文链接:** [http://arxiv.org/abs/2510.03584v1](http://arxiv.org/abs/2510.03584v1)

**作者:** Chaoyu Li, Tianzhi Li, Fei Tao, Zhenyu Zhao, Ziqian Wu, Maozheng Zhao, Juntong Song, Cheng Niu, Pooyan Fazli

**发布时间:** 2025-10-04

### GPT解析

### 总结

FrameOracle是一个轻量级即插即用模块，能够智能预测哪些帧与给定查询最相关以及需要多少帧，解决了视觉-语言模型在视频理解中受限于输入帧数的问题。

### 背景

视觉-语言模型(VLMs)在视频理解方面取得了进展，但其性能受限于能够处理的输入帧数。现有的帧采样策略如均匀选择或固定预算选择无法适应信息密度或任务复杂度的变化，导致效率低下和信息丢失。

### 目的

开发一个轻量级即插即用模块，用于预测(1)哪些帧与给定查询最相关，以及(2)需要多少帧，以提高视频理解的效率和准确性。

### 方法

FrameOracle使用四阶段课程进行训练，前三个阶段依赖弱代理信号如跨模态相似性，最后阶段利用新数据集FrameOracle-41K的更强监督，这是首个提供关键帧标注的大型VideoQA集合，指定了回答每个问题所需的最小帧集。

### 主要发现

在五个VLMs和六个基准测试上的实验表明，FrameOracle将16帧输入减少到平均10.4帧而没有任何精度损失；当从64帧候选开始时，将输入减少到平均13.9帧，同时提高精度1.4%，实现了可扩展视频理解的最先进效率-精度权衡。

### 结论

FrameOracle通过智能选择相关帧和确定所需帧数，解决了现有视频理解模型中帧采样策略的局限性，实现了更高的效率和更好的性能。

### 翻译

视觉-语言模型(VLMs)已推进视频理解，但其性能受限于它们可以处理的输入帧数。现有的帧采样策略，如均匀选择或固定预算选择，往往无法适应信息密度或任务复杂度的变化，导致效率低下和信息丢失。为解决这一问题，我们提出了FrameOracle，这是一个轻量级且即插即用的模块，可预测(1)哪些帧与给定查询最相关，以及(2)需要多少帧。FrameOracle使用四阶段课程进行训练，前三个阶段依赖弱代理信号如跨模态相似性。在最后阶段，它利用我们引入的新数据集FrameOracle-41K的更强监督，这是首个提供关键帧标注的大型VideoQA集合，指定了回答每个问题所需的最小帧集。在五个VLMs和六个基准测试上的广泛实验表明，FrameOracle将16帧输入减少到平均10.4帧而没有任何精度损失。当从64帧候选开始时，它将输入减少到平均13.9帧，同时提高精度1.4%，实现了可扩展视频理解的最先进效率-精度权衡。


### 论文摘要

Vision-language models (VLMs) have advanced video understanding, but their performance is limited by the number of input frames they can process. Existing frame sampling strategies, such as uniform or fixed-budget selection, often fail to adapt to variations in information density or task complexity, resulting in inefficiency and information loss. To address this, we present FrameOracle, a lightweight and plug-and-play module that predicts both (1) which frames are most relevant to a given query and (2) how many frames are needed. FrameOracle is trained using a four-stage curriculum, with the first three stages relying on weak proxy signals such as cross-modal similarity. In the final stage, it leverages stronger supervision from a new dataset we introduce, FrameOracle-41K, the first large-scale VideoQA collection to provide keyframe annotations specifying the minimal set of frames required to answer each question. Extensive experiments across five VLMs and six benchmarks demonstrate that FrameOracle reduces 16-frame inputs to an average of 10.4 frames without any loss in accuracy. When starting from 64-frame candidates, it reduces the input to an average of 13.9 frames while improving accuracy by 1.4%, achieving state-of-the-art efficiency-accuracy trade-offs for scalable video understanding.

---

## 18. Generalization of Graph Neural Network Models for Distribution Grid Fault Detection

**论文链接:** [http://arxiv.org/abs/2510.03571v1](http://arxiv.org/abs/2510.03571v1)

**作者:** Burak Karabulut, Carlo Manna, Chris Develder

**发布时间:** 2025-10-03

**备注:** This paper has been submitted and accepted for IEEE SmartGridComm  2025

### GPT解析

### 总结

该论文研究了电力配电网故障检测中不同图神经网络架构的性能比较，探索了GraphSAGE和图注意力网络在RNN+GNN管道中的应用，并评估了它们在不同拓扑设置中的泛化能力。

### 背景

电力配电网的故障检测对确保系统可靠性和防止昂贵停电至关重要。故障检测方法需要对不断变化的电网拓扑结构保持鲁棒性，这些变化可能由重构、设备故障和分布式能源资源集成等因素引起。

### 目的

系统地、一致地比较各种GNN架构在RNN+GNN管道模型中的性能，特别是探索GraphSAGE和图注意力网络在故障诊断中的应用，并评估这些模型在不同拓扑设置中的泛化潜力。

### 方法

在IEEE 123节点配网络上进行实验，使用RNN+GNN管道模型，比较了包括GraphSAGE、图注意力网络(GAT, GATv2)、RGCN和纯RNN模型(特别是GRU)在内的多种架构的性能。

### 主要发现

RGATv2模型展现出优异的泛化能力，在不同拓扑设置中F1分数仅下降约12%；相比之下，纯RNN模型F1分数下降高达约60%，其他RGNN变体F1分数下降约25%。

### 结论

Graph注意力网络，特别是RGATv2，在电力配电网故障检测任务中表现出色，具有良好的泛化能力，能够适应不同的电网拓扑结构变化。

### 翻译

电力配电网中的故障检测对确保系统可靠性和防止昂贵停电至关重要。此外，故障检测方法应对由重构、设备故障和分布式能源资源集成等因素引起的不断变化的电网拓扑结构保持鲁棒性。当前最先进的数据驱动方法使用循环神经网络进行时间建模，图神经网络进行空间学习，在RNN+GNN管道设置中。具体而言，对于电力系统故障诊断，已经采用了图卷积网络。然而，在电力系统领域之外，已经提出了和采用了各种更先进的GNN架构。在本文中，我们系统地、一致地比较了各种GNN架构在RNN+GNN管道模型中的性能。具体来说，据我们所知，我们首次提出在RGNN中使用GraphSAGE和图注意力进行故障诊断，以及与之前提出的RGNN解决方案和纯RNN模型进行全面比较，特别是探索它们在不同部署环境中的泛化潜力。我们在IEEE 123节点配电网上的实验结果表明，RGATv2具有优异的泛化能力，在不同拓扑设置中保持高性能，F1分数仅下降约12%。相比之下，纯RNN模型 largely 失败，F1分数下降高达约60%，而其他RGNN变体也表现出显著的性能下降，即F1分数低至约25%。


### 论文摘要

Fault detection in power distribution grids is critical for ensuring system reliability and preventing costly outages. Moreover, fault detection methodologies should remain robust to evolving grid topologies caused by factors such as reconfigurations, equipment failures, and Distributed Energy Resource (DER) integration. Current data-driven state-of-the-art methods use Recurrent Neural Networks (RNNs) for temporal modeling and Graph Neural Networks (GNNs) for spatial learning, in an RNN+GNN pipeline setting (RGNN in short). Specifically, for power system fault diagnosis, Graph Convolutional Networks (GCNs) have been adopted. Yet, various more advanced GNN architectures have been proposed and adopted in domains outside of power systems. In this paper, we set out to systematically and consistently benchmark various GNN architectures in an RNN+GNN pipeline model. Specifically, to the best of our knowledge, we are the first to (i) propose to use GraphSAGE and Graph Attention (GAT, GATv2) in an RGNN for fault diagnosis, and (ii) provide a comprehensive benchmark against earlier proposed RGNN solutions (RGCN) as well as pure RNN models (especially Gated Recurrent Unit (GRU)), particularly (iii) exploring their generalization potential for deployment in different settings than those used for training them. Our experimental results on the IEEE 123-node distribution network show that RGATv2 has superior generalization capabilities, maintaining high performance with an F1-score reduction of $\sim$12% across different topology settings. In contrast, pure RNN models largely fail, experiencing an F1-score reduction of up to $\sim$60%, while other RGNN variants also exhibit significant performance degradation, i.e., up to $\sim$25% lower F1-scores.

---

## 19. SegMASt3R: Geometry Grounded Segment Matching

**论文链接:** [http://arxiv.org/abs/2510.05051v1](http://arxiv.org/abs/2510.05051v1)

**作者:** Rohit Jayanti, Swayam Agrawal, Vansh Garg, Siddharth Tourani, Muhammad Haris Khan, Sourav Garg, Madhava Krishna

**发布时间:** 2025-10-06

**备注:** Accepted to The Thirty-Ninth Annual Conference on Neural Information  Processing Systems (NeurIPS 2025) as a Spotlight (top 3.5%)

### GPT解析

### 总结

本文提出了一种利用3D基础模型的空间理解能力来处理宽基线分割匹配的方法，能够在高达180度视点变化的图像对之间匹配分割区域。

### 背景

分割匹配是计算机视觉中的重要中间任务，它在图像之间建立语义或几何上一致区域的对应关系。与关键点匹配不同，分割匹配捕捉结构化区域，对遮挡、光照变化和视点变化具有更强的鲁棒性。

### 目的

解决宽基线分割匹配这一具有挑战性的任务，涉及极端视点变化下的图像分割区域匹配问题。

### 方法

提出一种利用3D基础模型归纳偏置的架构，能够在图像对之间匹配分割区域，支持高达180度的视点变化。

### 主要发现

在ScanNet++和Replica数据集上，与SAM2视频传播器和局部特征匹配方法等最先进方法相比，在AUPRC指标上提高了多达30%

### 结论

所提出的方法在宽基线分割匹配任务上优于现有方法，并在3D实例分割和图像目标导航等下游任务中展示了优势

### 翻译

分割匹配是计算机视觉中的一个重要中间任务，它在图像之间建立语义或几何上一致区域的对应关系。与关键点匹配不同，分割匹配捕捉结构化区域，对遮挡、光照变化和视点变化具有更强的鲁棒性。在本文中，我们利用3D基础模型的空间理解能力来处理宽基线分割匹配，这是一个涉及极端视点变化的挑战性任务。我们提出了一种架构，利用这些3D基础模型的归纳偏置来匹配图像对之间的分割区域，支持高达180度的视点变化。大量实验表明，在ScanNet++和Replica数据集上，与SAM2视频传播器和局部特征匹配方法等最先进方法相比，我们的方法在AUPRC指标上提高了多达30%。我们进一步展示了所提出模型在相关下游任务中的优势，包括3D实例分割和图像目标导航。项目页面：https://segmast3r.github.io/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决宽基线(wide-baseline)图像片段匹配问题，即在不同视角的图像之间建立语义或几何上一致区域的对应关系。这个问题在现实中很重要，因为它支撑着视频目标跟踪、场景图构建和机器人导航等关键应用；在研究中也很重要，因为现有方法在极端视角变化(如180度旋转)下性能急剧下降，而片段匹配比关键点匹配对遮挡、光照变化和视角变化具有更强的鲁棒性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法在宽基线条件下的局限性，认识到需要利用几何先验来处理视角变化。他们选择3D基础模型MASt3R作为基础，因为它已经学习了场景的深度、形状和姿态等几何属性。设计上，他们冻结了MASt3R的骨干网络，添加了一个轻量级的片段特征头将块级特征转换为片段级描述符，并借鉴了SuperGlue的对比损失函数和可微匹配层设计。此外，他们还利用了Segment Anything Model(SAM)来获取片段掩码，并参考了Sinkhorn算法用于软对应关系计算。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用3D基础模型MASt3R的几何理解能力来处理宽基线片段匹配，通过添加一个适配器将MASt3R的块级特征转换为片段级特征，并使用可微的最优传输层建立片段对应关系。整体流程：1)输入一对宽基线图像；2)使用冻结的MASt3R提取块级特征；3)通过片段特征头将块级特征转换为片段级描述符；4)使用可微的最优传输层匹配片段，包括计算相似度、添加dustbin机制、使用Sinkhorn算法获得软对应关系；5)通过行级argmax获得最终匹配结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将3D基础模型MASt3R重新用于片段匹配任务；2)设计片段特征头将像素级表示转换为片段级描述符；3)引入可微的最优传输层和dustbin机制处理不匹配情况；4)实现端到端训练。相比之前的工作，不同之处在于：相比局部特征匹配方法，它匹配结构化区域而非点，对遮挡和外观变化更鲁棒；相比SAM2，它在极端视角变化下显式强制执行几何一致性；相比基于2D监督的方法，它利用3D几何先验处理近180度的视角变化；在AUPRC指标上比最先进方法高出最多30%。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SegMASt3R通过利用3D基础模型的几何理解能力，实现了在极端视角变化下(最高180度旋转)的鲁棒图像片段匹配，显著提升了宽基线场景下的性能，并在3D实例映射和物体相对导航等下游任务中展现出实用价值。'}


### 论文摘要

Segment matching is an important intermediate task in computer vision that establishes correspondences between semantically or geometrically coherent regions across images. Unlike keypoint matching, which focuses on localized features, segment matching captures structured regions, offering greater robustness to occlusions, lighting variations, and viewpoint changes. In this paper, we leverage the spatial understanding of 3D foundation models to tackle wide-baseline segment matching, a challenging setting involving extreme viewpoint shifts. We propose an architecture that uses the inductive bias of these 3D foundation models to match segments across image pairs with up to 180 degree view-point change. Extensive experiments show that our approach outperforms state-of-the-art methods, including the SAM2 video propagator and local feature matching methods, by upto 30% on the AUPRC metric, on ScanNet++ and Replica datasets. We further demonstrate benefits of the proposed model on relevant downstream tasks, including 3D instance segmentation and image-goal navigation. Project Page: https://segmast3r.github.io/

---

## 20. Large Language Models Achieve Gold Medal Performance at International Astronomy & Astrophysics Olympiad

**论文链接:** [http://arxiv.org/abs/2510.05016v1](http://arxiv.org/abs/2510.05016v1)

**作者:** Lucas Carrit Delgado Pinheiro, Ziru Chen, Bruno Caixeta Piazza, Ness Shroff, Yingbin Liang, Yuan-Sen Ting, Huan Sun

**发布时间:** 2025-10-06

**备注:** 18 pages, 6 figures, to be submitted, comments are welcome

### GPT解析

### 总结

研究评估了五种先进大型语言模型在国际天文学和天体物理学奥林匹克竞赛中的表现，发现这些模型在理论考试中接近人类顶尖水平，但在数据分析考试中表现差异大，且在概念推理、几何推理和空间可视化方面存在共同弱点。

### 背景

当前针对特定任务的示范应用显示大型语言模型在自动化天文学研究任务方面有早期成功，但仅提供了解决天文学问题所需能力的不完整视角。现有基准测试主要关注简单问答，测试天文学知识，未能评估现实研究中的复杂推理能力。

### 目的

通过在国际天文学和天体物理学奥林匹克竞赛考试上系统评估五种先进大型语言模型，填补现有评估方法的空白，因为这些考试检验深入概念理解、多步推导和多模态分析能力。

### 方法

在国际天文学和天体物理学奥林匹克竞赛(IOAA)考试上对五种最先进的大型语言模型进行系统性基准测试，IOAA考试旨在检验深入概念理解、多步推导和多模态分析能力。

### 主要发现

Gemini 2.5 Pro和GPT-5平均得分分别为85.6%和84.2%，达到金牌水平，在理论考试中排名人类参赛者前两名；在数据分析考试中，GPT-5表现最佳(88.5%)，其他模型性能下降至48-76%；所有模型在概念推理、几何推理和空间可视化方面存在共同弱点(准确率52-79%)。

### 结论

尽管大型语言模型在理论考试中接近顶尖人类水平，但在它们能够作为天文学自主研究代理之前，必须解决关键差距。

### 翻译

虽然针对特定任务的示范应用显示将大型语言模型应用于自动化一些天文学研究任务方面显示出早期成功，但它们仅提供了解决天文学问题所需全部能力的不完整视角，这需要对大型语言模型的优势和局限性有更全面的理解。迄今为止，现有的基准测试和评估主要集中在简单的问答上，主要测试天文学知识，而未能评估该学科现实研究中所需的复杂推理能力。在这里，我们通过在国际天文学和天体物理学奥林匹克竞赛(IOAA)考试上系统性地对五种最先进的大型语言模型进行基准测试来解决这一差距，这些考试旨在检验深入的概念理解、多步推导和多模态分析能力。凭借85.6%和84.2%的平均分，Gemini 2.5 Pro和GPT-5（两种表现最佳的模型）不仅达到金牌水平，而且在所有四个评估的IOAA理论考试（2022-2025年）中排名在约200-300名参赛者的前两名。相比之下，数据分析考试的结果显示出更多差异。GPT-5在这些考试中仍然表现出色，平均得分为88.5%，在最近四届IOAA中排名前10，而其他模型的性能下降到48-76%。此外，我们深入的错误分析强调了概念推理、几何推理和空间可视化（52-79%的准确率）是所有大型语言模型中的一致性弱点。因此，尽管大型语言模型在理论考试中接近顶尖人类水平，但在它们能够作为天文学自主研究代理之前，必须解决关键差距。


### 论文摘要

While task-specific demonstrations show early success in applying large language models (LLMs) to automate some astronomical research tasks, they only provide incomplete views of all necessary capabilities in solving astronomy problems, calling for more thorough understanding of LLMs' strengths and limitations. So far, existing benchmarks and evaluations focus on simple question-answering that primarily tests astronomical knowledge and fails to evaluate the complex reasoning required for real-world research in the discipline. Here, we address this gap by systematically benchmarking five state-of-the-art LLMs on the International Olympiad on Astronomy and Astrophysics (IOAA) exams, which are designed to examine deep conceptual understanding, multi-step derivations, and multimodal analysis. With average scores of 85.6% and 84.2%, Gemini 2.5 Pro and GPT-5 (the two top-performing models) not only achieve gold medal level performance but also rank in the top two among ~200-300 participants in all four IOAA theory exams evaluated (2022-2025). In comparison, results on the data analysis exams show more divergence. GPT-5 still excels in the exams with an 88.5% average score, ranking top 10 among the participants in the four most recent IOAAs, while other models' performances drop to 48-76%. Furthermore, our in-depth error analysis underscores conceptual reasoning, geometric reasoning, and spatial visualization (52-79% accuracy) as consistent weaknesses among all LLMs. Hence, although LLMs approach peak human performance in theory exams, critical gaps must be addressed before they can serve as autonomous research agents in astronomy.

---

## 21. Progressive Gaussian Transformer with Anisotropy-aware Sampling for Open Vocabulary Occupancy Prediction

**论文链接:** [http://arxiv.org/abs/2510.04759v1](http://arxiv.org/abs/2510.04759v1)

**作者:** Chi Yan, Dan Xu

**发布时间:** 2025-10-06

**备注:** Project Page: https://yanchi-3dv.github.io/PG-Occ

### GPT解析

### 总结

PG-Occ是一种创新的渐进式高斯变换器框架，用于开放词汇的3D占用率预测，通过渐进式在线密集化和各向异性感知采样策略解决了稀疏和密集表示之间的权衡问题，实现了最先进的性能。

### 背景

3D占用率预测在基于视觉的自动驾驶系统中扮演重要角色。传统方法局限于固定语义类别，而近期方法转向预测文本对齐特征以支持开放词汇文本查询。然而，文本对齐场景建模存在权衡：稀疏高斯表示难以捕捉小物体，密集表示则带来显著计算开销。

### 目的

解决文本对齐场景建模中的权衡问题，实现开放词汇的3D占用率预测，同时能够捕捉细粒度的场景细节并有效控制计算开销。

### 方法

提出PG-Progressive Gaussian Transformer Framework，采用渐进式在线密集化策略逐步增强3D高斯表示，并引入各向异性感知采样策略结合时空融合，自适应地为不同尺度和阶段的高斯分配感受野，实现更有效的特征聚合和更丰富的场景信息捕获。

### 主要发现

通过迭代增强表示，框架实现了越来越精确和详细的场景理解。在广泛评估中，PG-Occ实现了最先进的性能，相比之前最佳方法相对提高了14.3%的mIoU。

### 结论

PG-Occ成功解决了3D占用率预测中开放词汇表示与计算效率之间的权衡问题，为自动驾驶系统提供了更精确的场景理解能力。

### 翻译

近年来，3D占用率预测任务取得了显著进展，在基于视觉的自动驾驶系统中发挥着关键作用。虽然传统方法仅限于固定的语义类别，但近期方法已转向预测文本对齐的特征，以支持在真实场景中进行开放词汇的文本查询。然而，在文本对齐的场景建模中存在权衡：稀疏高斯表示难以捕捉场景中的小物体，而密集表示则会带来显著的计算开销。为解决这些局限性，我们提出了PG-Occ，一种创新的渐进式高斯变换器框架，实现了开放词汇的3D占用率预测。我们的框架采用渐进式在线密集化，这是一种前馈策略，能够逐步增强3D高斯表示以捕捉细粒度的场景细节。通过迭代增强表示，该框架实现了越来越精确和详细的场景理解。另一个关键贡献是引入了具有时空融合的各向异性感知采样策略，该策略自适应地为不同尺度和阶段的高斯分配感受野，实现更有效的特征聚合和更丰富的场景信息捕获。通过广泛评估，我们证明PG-Occ实现了最先进的性能，比之前最佳方法相对提高了14.3%的mIoU。代码和预训练模型将在我们的项目页面上发布时公开：https://yanchi-3dv.github.io/PG-Occ

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D占用预测中的开放词汇语义识别问题。传统方法只能识别预定义的物体类别，无法应对现实世界中多样化的物体。在自动驾驶领域，系统需要能够识别和定位各种可能的物体，包括那些未预先定义的类别，这对安全决策至关重要。此外，现有方法在稀疏表示（计算效率高但细节不足）和密集表示（细节丰富但计算开销大）之间存在权衡，难以同时满足效率和精度的需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：固定语义类别的限制、稀疏与密集表示的权衡、高维文本特征的计算开销等。受3D高斯溅射技术的启发，作者考虑使用稀疏高斯表示来平衡效率和细节捕捉。他们提出渐进式思路，先使用粗略的基础高斯模型捕捉全局场景，再逐步细化。同时借鉴了GaussTR等使用稀疏高斯表示的工作，以及Metric3D V2和MaskCLIP等用于深度估计和文本对齐的技术。作者还创新性地设计了各向异性感知采样和非对称自注意力机制来解决特定问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用渐进式高斯表示，从粗到细逐步增强场景理解能力，同时利用各向异性感知采样和非对称自注意力机制来提高特征提取的准确性和训练的稳定性。整体流程包括：1)初始化阶段使用伪深度图和远点采样选择初始高斯位置；2)基础层处理捕获粗略场景几何；3)渐进式细化包括识别表示不足区域并添加新高斯、处理高斯间关系、根据高斯各向异性特性采样特征点；4)特征聚合与解码；5)最终将高斯表示转换为密集3D占用场，支持开放词汇查询。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)渐进式高斯变换器框架，通过在线渐进密集化策略逐步增强场景表示；2)各向异性感知采样策略，根据高斯空间分布自适应分配感受野；3)非对称自注意力机制，防止新添加高斯干扰已优化高斯。相比之前的工作，PG- adaptively扩展查询数量而非使用固定数量，考虑了高斯的各向异性特性而非简单视为点，采用非对称自注意力而非标准自注意力，使用纯前馈方式实现实时推理而非离线优化，且在不使用LiDAR数据训练的情况下仍能超过使用LiDAR数据的先前最佳方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PG-Occ通过渐进式高斯变换和各向异性感知采样，实现了高效且准确的开放词汇3D占用预测，在保持计算效率的同时显著提升了场景细节捕捉能力。'}


### 论文摘要

The 3D occupancy prediction task has witnessed remarkable progress in recent years, playing a crucial role in vision-based autonomous driving systems. While traditional methods are limited to fixed semantic categories, recent approaches have moved towards predicting text-aligned features to enable open-vocabulary text queries in real-world scenes. However, there exists a trade-off in text-aligned scene modeling: sparse Gaussian representation struggles to capture small objects in the scene, while dense representation incurs significant computational overhead. To address these limitations, we present PG-Occ, an innovative Progressive Gaussian Transformer Framework that enables open-vocabulary 3D occupancy prediction. Our framework employs progressive online densification, a feed-forward strategy that gradually enhances the 3D Gaussian representation to capture fine-grained scene details. By iteratively enhancing the representation, the framework achieves increasingly precise and detailed scene understanding. Another key contribution is the introduction of an anisotropy-aware sampling strategy with spatio-temporal fusion, which adaptively assigns receptive fields to Gaussians at different scales and stages, enabling more effective feature aggregation and richer scene information capture. Through extensive evaluations, we demonstrate that PG-Occ achieves state-of-the-art performance with a relative 14.3% mIoU improvement over the previous best performing method. Code and pretrained models will be released upon publication on our project page: https://yanchi-3dv.github.io/PG-Occ

---

## 22. Object-Centric Representation Learning for Enhanced 3D Scene Graph Prediction

**论文链接:** [http://arxiv.org/abs/2510.04714v1](http://arxiv.org/abs/2510.04714v1)

**作者:** KunHo Heo, GiHyun Kim, SuYeon Kim, MyeongAh Cho

**发布时间:** 2025-10-06

**备注:** Accepted by NeurIPS 2025. Code:  https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes

### GPT解析

### 总结

本文提出了一种改进的3D语义场景图预测方法，通过优化对象特征质量和结合几何与语义特征，显著提升了场景图预测的准确性。

### 背景

3D语义场景图预测是检测3D场景中对象及其语义关系的关键技术，对机器人和AR/VR应用至关重要。先前研究虽解决了数据集限制问题并探索了多种方法，但常未能充分优化对象和关系特征的表示能力，过度依赖图神经网络而判别能力不足。

### 目的

解决现有方法中对象特征质量不足的问题，提高场景图预测的整体准确性，并减少对图神经网络的过度依赖。

### 方法

设计高判别性的对象特征编码器，采用对比预训练策略将对象表示学习与场景图预测解耦，并有效结合几何和语义特征以实现更优的关系预测。

### 主要发现

对象特征质量对场景图整体准确性起关键作用；所提方法提高了对象分类准确性并改善了关系预测；将预训练编码器插入现有框架时，所有评估指标都观察到显著性能提升；结合几何和语义特征实现了优越的关系预测效果。

### 结论

在3DSSG数据集上的综合实验表明，该方法显著优于先前最先进的方法，相关代码已公开可用。

### 翻译

3D语义场景图预测旨在检测3D场景中的对象及其语义关系，已成为机器人和AR/VR应用的关键技术。虽然先前研究解决了数据集限制问题并探索了包括开放词汇设置在内的各种方法，但它们经常未能优化对象和关系特征的表示能力，尽管判别能力不足却过度依赖图神经网络。在本工作中，我们通过大量分析证明对象特征质量对整体场景图准确性起着关键作用。为应对这一挑战，我们设计了一个高判别性的对象特征编码器，并采用对比预训练策略将对象表示学习与场景图预测解耦。这种设计不仅提高了对象分类准确性，也直接改善了关系预测。值得注意的是，当将我们的预训练编码器插入现有框架时，我们在所有评估指标上都观察到显著的性能提升。此外，由于现有方法未充分利用关系信息的整合，我们有效地结合了几何和语义特征以实现优越的关系预测。在3DSSG数据集上的综合实验表明，我们的方法显著优于先前最先进的方法。我们的代码已在https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes公开可用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D语义场景图预测中对象分类错误导致关系预测不准确的问题。现有方法过度依赖图神经网络进行关系推理，而忽视了对象特征质量的提升。这个问题在现实中很重要，因为准确的3D场景图理解对于机器人导航、物体操作和AR/VR交互等关键应用至关重要，能提高机器人在复杂环境中的感知能力和人机交互的自然度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析发现对象分类错误与关系预测错误高度相关，且对象特征质量对整体场景图准确性起决定性作用。基于这一观察，作者设计了判别性对象特征编码器和新型关系特征编码器。作者借鉴了CLIP的跨模态对比学习、PointNet的点云处理和图神经网络等现有工作，但创新性地将这些技术组合并优化，特别强调了对象表示学习与场景图预测的解耦。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提高对象特征的判别力可以直接改善关系预测。整体流程分为三个阶段：1)对象特征学习阶段：使用对比预训练结合3D点云、2D图像和文本描述训练对象编码器；2)关系特征学习阶段：设计结合对象特征和几何信息的关系编码器，并引入局部空间增强模块；3)图神经网络阶段：应用双向边门控机制和全局空间增强进行场景图预测，捕获对象间的空间关系和不对称性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)判别性对象特征编码器，通过对比预训练提高对象特征质量；2)新型关系特征编码器，结合语义和几何特征；3)双向边门控机制，显式建模主-客体不对称性；4)全局空间增强，整合整体空间上下文；5)解耦对象表示学习与场景图预测的策略。相比之前工作，本文优先提升对象表示质量而非直接优化关系预测，且方法可集成到现有框架中提升整体性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过提出判别性对象特征编码器和新型关系特征编码器，显著提高了3D语义场景图预测的准确性，在对象分类和关系预测方面实现了最先进性能。'}


### 论文摘要

3D Semantic Scene Graph Prediction aims to detect objects and their semantic relationships in 3D scenes, and has emerged as a crucial technology for robotics and AR/VR applications. While previous research has addressed dataset limitations and explored various approaches including Open-Vocabulary settings, they frequently fail to optimize the representational capacity of object and relationship features, showing excessive reliance on Graph Neural Networks despite insufficient discriminative capability. In this work, we demonstrate through extensive analysis that the quality of object features plays a critical role in determining overall scene graph accuracy. To address this challenge, we design a highly discriminative object feature encoder and employ a contrastive pretraining strategy that decouples object representation learning from the scene graph prediction. This design not only enhances object classification accuracy but also yields direct improvements in relationship prediction. Notably, when plugging in our pretrained encoder into existing frameworks, we observe substantial performance improvements across all evaluation metrics. Additionally, whereas existing approaches have not fully exploited the integration of relationship information, we effectively combine both geometric and semantic features to achieve superior relationship prediction. Comprehensive experiments on the 3DSSG dataset demonstrate that our approach significantly outperforms previous state-of-the-art methods. Our code is publicly available at https://github.com/VisualScienceLab-KHU/OCRL-3DSSG-Codes.

---

## 23. AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials

**论文链接:** [http://arxiv.org/abs/2510.04704v1](http://arxiv.org/abs/2510.04704v1)

**作者:** Taoyuze Lv, Alexander Chen, Fengyu Xie, Chu Wu, Jeffrey Meng, Dongzhan Zhou, Bram Hoex, Zhicheng Zhong, Tong Xie

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文介绍了AtomWorld基准测试，用于评估大型语言模型在处理晶体学信息文件(CIFs)方面的能力，特别是在结构编辑、CIF感知和属性引导建模等任务上的表现。研究发现当前模型在结构理解和空间推理方面存在明显局限性。

### 背景

大型语言模型在文本推理方面表现出色，并开始发展空间理解能力。在材料科学等领域，对3D原子结构的深入理解是基础性的。虽然已有初步研究成功将LLMs应用于纯晶体生成或坐标理解任务，但缺乏一个标准化的基准来系统评估它们在多样化原子结构上的核心推理能力。

### 目的

引入AtomWorld基准测试，用于评估大型语言模型基于晶体学信息文件(CIFs)的任务表现，CIFs是一种标准结构表示格式。

### 方法

定义了一系列基于CIFs的任务，包括结构编辑、CIF感知和属性引导建模，用于系统评估LLMs的原子结构理解和空间推理能力。

### 主要发现

当前模型尽管建立了有希望的基准线，但在结构理解和空间推理方面持续失败。实验表明，这些模型在结构修改任务中经常出错，甚至在基本的CIF格式理解方面也存在问题，可能导致后续分析和材料见解中的累积误差。

### 结论

通过定义这些标准化任务，AtomWorld为推进大型语言模型向稳健的原子级建模奠定了基础，这对于加速材料研究和自动化科学工作流程至关重要。

### 翻译

大型语言模型(LLMs)在文本推理方面表现出色，并开始发展空间理解能力，这引发了一个问题：这些能力是否可以结合用于复杂、领域特定的任务。在材料科学等领域，这个问题尤为重要，因为对3D原子结构的深入理解是基础性的。虽然初步研究已成功将LLMs应用于涉及纯晶体生成或坐标理解的任务，但缺乏一个标准化的基准来系统评估它们在多样化原子结构上的核心推理能力。为解决这一差距，我们引入了AtomWorld基准测试，用于评估LLMs基于晶体学信息文件(CIFs)的任务表现，CIFs是一种标准结构表示格式。这些任务包括结构编辑、CIF感知和属性引导建模，揭示了一个关键局限：当前模型尽管建立了有希望的基准线，但在结构理解和空间推理方面持续失败。我们的实验表明，这些模型在结构修改任务中经常出错，甚至在基本的CIF格式理解方面也存在问题，可能导致后续分析和材料见解中的累积误差。通过定义这些标准化任务，AtomWorld为推进LLMs向稳健的原子级建模奠定了基础，这对于加速材料研究和自动化科学工作流程至关重要。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决缺乏标准化基准来系统评估大语言模型在处理晶体材料原子结构方面的空间推理能力问题。这个问题在材料科学等领域非常重要，因为深入理解3D原子结构是基础，而当前模型在结构理解和空间推理方面存在关键局限性，建立这一基准对于推动LLMs向稳健的原子级建模发展、加速材料研究和自动化科学工作流程至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将LLMs处理CIF文件的能力分为运动技能（几何操作）、感知技能（识别模式）和认知技能（推理与创造力）三个阶段，重点关注运动技能这一基础能力。他们设计了AtomWorld基准，基于晶体学信息文件（CIFs）创建多种原子结构操作任务。作者借鉴了Materials Project数据库中的CIF文件作为结构池，使用pymatgen库中的StructureMatcher进行结构比较，并参考了晶体学信息文件标准格式，但针对晶体材料领域进行了专门设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个标准化的基准测试框架，系统评估LLMs在处理晶体材料原子结构方面的空间推理能力。整体实现流程包括：1) AtomWorld生成器从结构池随机选择结构，初始化操作模板，应用操作获得目标结构，并生成自然语言描述；2) 将输入结构和动作提示提供给LLM，生成修改后的结构并与目标结构比较；3) 使用互补基准测试（如PointWorld、CIF读写测试等）全面评估不同能力维度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首个专门评估LLMs晶体学运动技能的基准；可扩展的数据生成器支持LLM训练；全面的任务设计涵盖多种原子结构操作；互补基准测试形成全面评估套件；系统化的多层级评估方法。相比之前工作，本文更专注于结构修改和空间推理能力而非仅晶体结构生成或问答；提供了标准化评估框架；不仅测试基本操作还评估复杂多步推理；直接针对材料科学研究中的实际操作需求设计任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'AtomWorld基准首次系统评估了大型语言模型在晶体材料原子结构修改和空间推理方面的能力，揭示了当前模型在这一关键领域的局限性，并为未来开发更强大的原子级建模工具奠定了基础。'}


### 论文摘要

Large Language Models (LLMs) excel at textual reasoning and are beginning to develop spatial understanding, prompting the question of whether these abilities can be combined for complex, domain-specific tasks. This question is essential in fields like materials science, where deep understanding of 3D atomic structures is fundamental. While initial studies have successfully applied LLMs to tasks involving pure crystal generation or coordinate understandings, a standardized benchmark to systematically evaluate their core reasoning abilities across diverse atomic structures has been notably absent. To address this gap, we introduce the AtomWorld benchmark to evaluate LLMs on tasks based in Crystallographic Information Files (CIFs), a standard structure representation format. These tasks, including structural editing, CIF perception, and property-guided modeling, reveal a critical limitation: current models, despite establishing promising baselines, consistently fail in structural understanding and spatial reasoning. Our experiments show that these models make frequent errors on structure modification tasks, and even in the basic CIF format understandings, potentially leading to cumulative errors in subsequent analysis and materials insights. By defining these standardized tasks, AtomWorld lays the ground for advancing LLMs toward robust atomic-scale modeling, crucial for accelerating materials research and automating scientific workflows.

---

## 24. Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation

**论文链接:** [http://arxiv.org/abs/2510.03863v1](http://arxiv.org/abs/2510.03863v1)

**作者:** Arina Kharlamova, Bowei He, Chen Ma, Xue Liu

**发布时间:** 2025-10-04

**备注:** Submitted to ICLR 2026

### GPT解析

### 总结

该研究提出了SpatialCAPTCHA，一种新型人机验证框架，利用人类与多模态大语言模型在空间推理方面的根本差异，设计需要几何推理、视角转换、遮挡处理和心理旋转等能力的验证问题，有效抵御现代AI攻击。

### 背景

在线服务依赖CAPTCHA作为抵御自动化滥用的第一道防线，但最近多模态大语言模型的进步已经削弱了专注于文本识别或2D图像理解的常规CAPTCHA设计的有效性。

### 目的

为了应对MLLMs对传统CAPTCHA的挑战，提出一种利用人类与AI在空间推理能力差异的新型验证框架，提高验证系统的安全性和有效性。

### 方法

采用程序生成管道，包含基于约束的难度控制、自动正确性验证和人工循环验证，在Spatial-CAPTCHA-Bench基准上评估性能，并与Google reCAPTCHA进行比较。

### 主要发现

人类在Spatial-CAPTCHA-Bench上的表现远超10种最先进的MLLMs，最佳模型的Pass@1准确率仅为31.0%，证实了该方法的有效性。

### 结论

Spatial CAPTCHA通过利用人类与AI在空间推理能力上的差异，提供了有效抵御现代AI攻击的验证方法，同时也可作为评估AI空间推理能力的诊断工具。

### 翻译

在线服务依赖CAPTCHA作为抵御自动化滥用的第一道防线，然而最近多模态大语言模型的进步已经削弱了专注于文本识别或2D图像理解的常规设计的有效性。为了应对这一挑战，我们提出了SpatialCAPTCHA，一种新型的人机验证框架，它利用人类与MLLMs在空间推理方面的根本差异。与依赖容易被现代AI攻破的低级感知任务的现有CAPTCHA不同，Spatial CAPTCHA生成需要几何推理、视角转换、遮挡处理和心理旋转的动态问题。这些技能对人类来说是直观的，但对最先进的AI系统来说却很困难。该系统采用程序生成管道，包含基于约束的难度控制、自动正确性验证和人工循环验证，以确保可扩展性、鲁棒性和适应性。在相应的基准Spatial-CAPTCHA-Bench上的评估表明，人类的表现远超10种最先进的MLLMs，其中最佳模型的Pass@1准确率仅为31.0%。此外，我们将Spatial CAPTCHA与Google reCAPTCHA进行了比较，证实了其作为安全机制和AI空间推理诊断工具的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是随着多模态大语言模型的发展，传统CAPTCHA系统越来越容易被AI破解，无法有效保护在线服务免受自动化滥用。这个问题非常重要，因为CAPTCHA是保护网络服务（如Google、Facebook等）的第一道防线，防止自动化攻击（如凭证填充、内容抓取和垃圾信息）导致的经济损失和用户信任下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到人类天生具有强大的3D空间推理能力，而AI在这方面存在局限。他们借鉴了人类认知科学研究中的空间能力分类，设计了七种类型的空间推理任务。作者参考了经典心理测量工具和CAPTCHA技术的历史发展，但创新性地将这些空间能力转化为区分人类和AI的有效方法。他们开发了一个自主生成系统，集成了难度控制、自动验证和人机循环验证等机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用人类与AI在空间推理能力上的根本差异，通过设计需要3D空间理解、几何推理、视角转换和心像旋转等能力的任务来区分人类和机器。整体流程包括：定义七种空间推理任务类型；开发程序生成管道（场景生成、干扰项合成、验证）；构建提示和答案；组装成完整CAPTCHA实例；使用基于约束的难度控制确保任务对人类简单但对AI具有挑战性；通过自动验证和人机循环验证确保任务质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：基于人类认知能力的理论基础设计；专注于空间推理的七种任务类型；可扩展的程序生成管道；基于约束的难度控制；自动正确性验证；人机循环验证机制。相比之前工作，不同之处在于：传统CAPTCHA依赖文本或2D图像识别，而Spatial CAPTCHA专注于3D空间推理；不仅是评估工具，也是实用的安全机制；利用人类与AI在空间能力上的根本差异，创造了更大的人机性能差距。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Spatial CAPTCHA通过利用人类与AI在空间推理能力上的根本差异，提出了一种新型人机验证框架，能有效区分人类和先进AI模型，为在线服务提供更可靠的安全保护。'}


### 论文摘要

Online services rely on CAPTCHAs as a first line of defense against automated abuse, yet recent advances in multi-modal large language models (MLLMs) have eroded the effectiveness of conventional designs that focus on text recognition or 2D image understanding. To address this challenge, we present Spatial CAPTCHA, a novel human-verification framework that leverages fundamental differences in spatial reasoning between humans and MLLMs. Unlike existing CAPTCHAs which rely on low-level perception tasks that are vulnerable to modern AI, Spatial CAPTCHA generates dynamic questions requiring geometric reasoning, perspective-taking, occlusion handling, and mental rotation. These skills are intuitive for humans but difficult for state-of-the-art (SOTA) AI systems. The system employs a procedural generation pipeline with constraint-based difficulty control, automated correctness verification, and human-in-the-loop validation to ensure scalability, robustness, and adaptability. Evaluation on a corresponding benchmark, Spatial-CAPTCHA-Bench, demonstrates that humans vastly outperform 10 state-of-the-art MLLMs, with the best model achieving only 31.0% Pass@1 accuracy. Furthermore, we compare Spatial CAPTCHA with Google reCAPTCHA, which confirms its effectiveness as both a security mechanism and a diagnostic tool for spatial reasoning in AI.

---

## 25. Spatial-ViLT: Enhancing Visual Spatial Reasoning through Multi-Task Learning

**论文链接:** [http://arxiv.org/abs/2510.03441v1](http://arxiv.org/abs/2510.03441v1)

**作者:** Chashi Mahiul Islam, Oteo Mamo, Samuel Jacob Chacko, Xiuwen Liu, Weikuan Yu

**发布时间:** 2025-10-03

**备注:** 12 pages, 5 figures

### GPT解析

### 总结

研究团队提出了SpatialViLT，一种增强的视觉语言模型，通过整合空间特征来解决3D场景和复杂物体配置中的空间推理挑战，并在视觉空间推理数据集上取得了最先进的准确性。

### 背景

视觉语言模型(VLMs)在多模态推理方面取得了进展，但在3D场景和复杂物体配置的空间推理方面仍面临挑战。

### 目的

开发一种增强的VLM，通过整合空间特征来提升模型对3D场景和复杂物体配置的空间推理能力。

### 方法

通过多任务学习框架整合深度图、3D坐标和边缘图等空间特征，提出SpatialViLT和MaskedSpatialViLT两种变体，分别关注完整和遮蔽的对象区域，并创建SpatialEnsemble结合两种方法。

### 主要发现

模型在方向关系、拓扑关系和邻近关系等空间推理类别方面表现出色，在具有挑战性的视觉空间推理(VSR)数据集上证明了其有效性。

### 结论

这项工作在增强AI系统的空间智能方面迈出了重要一步，这对高级多模态理解和实际应用至关重要。

### 翻译

视觉语言模型(VLMs)已推进多模态推理，但在3D场景和复杂物体配置的空间推理方面仍面临挑战。为此，我们引入了SpatialViLT，一种通过多任务学习框架整合深度图、3D坐标和边缘图等空间特征的增强型VLM。这种方法通过空间理解丰富了多模态嵌入。我们提出了两种变体：SpatialViLT和MaskedSpatialViLT，分别关注完整和遮蔽的对象区域。此外，SpatialEnsemble结合了这两种方法，实现了最先进的准确性。我们的模型在方向关系、拓扑关系和邻近关系等空间推理类别方面表现出色，这在具有挑战性的视觉空间推理(VSR)数据集上得到了证明。这项工作在增强AI系统的空间智能方面代表了重要一步，对高级多模态理解和实际应用至关重要。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言模型（VLMs）在3D场景和复杂物体配置中的空间推理能力有限的问题。这个问题很重要，因为空间推理是人类认知的核心方面，对AI系统理解真实世界中的空间配置和交互至关重要。缺乏这种能力限制了AI系统在高级多模态理解和实际应用中的表现，如自动驾驶、机器人导航和增强现实等领域。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了当前VLMs在VSR数据集上的表现（不超过70%准确率，而人类可达95.4%），认识到现有模型在空间推理方面的局限性。他们借鉴了SpatialVLM等工作的思路，但发现仅靠数据集和微调不足。作者选择了ViLT作为基础模型，因为它适合集成各种空间特征。通过分析VSR数据集中不同空间关系元类别的需求，作者设计了多任务学习框架，同时预测深度图、3D坐标和边缘图三种空间特征，从而增强模型的空间理解能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多任务学习框架将空间特征（深度图、3D坐标、边缘图）整合到视觉语言模型中，增强模型的空间先验知识和3D理解能力。整体流程包括：1)特征提取管道，使用CLIPSeg进行对象分割，MiDaS生成深度图，计算3D坐标，Canny算法生成边缘图；2)基于ViLT构建模型架构，包含CNN编码器处理空间特征，解码器重建特征，分类器做预测；3)多任务训练，同时优化分类任务和空间特征重建任务；4)SpatialEnsemble技术，结合多个模型的预测，根据性能加权投票提高整体准确率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出SpatialViLT和MaskedSpatialViLT两种模型变体，前者处理全局空间信息，后者专注于物体特定区域；2)设计多任务学习框架，同时预测深度图、3D坐标和边缘图三种空间特征；3)创建SpatialEnsemble集成方法，结合多个专家模型的优势；4)实现了在VSR数据集上的最先进性能。相比之前工作，本文不仅整合了更多种类的空间特征，还通过多任务学习框架同时优化多种空间表示，而非单一任务训练，并且引入了关系感知的集成方法，针对不同空间关系元类别调整模型权重。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SpatialViLT通过多任务学习框架整合空间特征并使用SpatialEnsemble技术，显著提升了视觉语言模型在3D场景中的空间推理能力，实现了在VSR数据集上的最先进性能。'}


### 论文摘要

Vision-language models (VLMs) have advanced multimodal reasoning but still face challenges in spatial reasoning for 3D scenes and complex object configurations. To address this, we introduce SpatialViLT, an enhanced VLM that integrates spatial features like depth maps, 3D coordinates, and edge maps through a multi-task learning framework. This approach enriches multimodal embeddings with spatial understanding. We propose two variants: SpatialViLT and MaskedSpatialViLT, focusing on full and masked object regions, respectively. Additionally, SpatialEnsemble combines both approaches, achieving state-of-the-art accuracy. Our models excel in spatial reasoning categories such as directional, topological, and proximity relations, as demonstrated on the challenging Visual Spatial Reasoning (VSR) dataset. This work represents a significant step in enhancing the spatial intelligence of AI systems, crucial for advanced multimodal understanding and real-world applications.

---

## 26. 论文ID: 2510.03342v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.03342v1.json'

---

## 27. The Geometry of Truth: Layer-wise Semantic Dynamics for Hallucination Detection in Large Language Models

**论文链接:** [http://arxiv.org/abs/2510.04933v1](http://arxiv.org/abs/2510.04933v1)

**作者:** Amir Hameed Mir

**发布时间:** 2025-10-06

**备注:** Comments: 14 pages, 14 figures, 5 tables. Code available at:  https://github.com/sirraya-tech/Sirraya_LSD_Code

### GPT解析

### 总结

本研究提出了层级语义动力学(LSD)框架，用于检测大型语言模型中的幻觉现象。该方法通过分析transformer层中隐藏状态的语义演变，实现了高效准确的幻觉检测，仅需单次前向传播即可达到高精度。

### 背景

大型语言模型(LLMs)经常产生流畅但事实不正确的陈述，这种现象称为'幻觉'，在高风险领域会带来严重风险。现有的幻觉检测方法通常依赖多次采样或外部验证源，效率较低。

### 目的

开发一种几何框架来检测大型语言模型中的幻觉现象，分析transformer层中隐藏状态语义的演变，提供一种高效、内在的检测方法。

### 方法

层级语义动力学(LSD)框架使用基于边距的对比学习，将隐藏激活与来自事实编码器的事实嵌入对齐。通过分析语义轨迹的分离情况来区分事实性响应和幻觉：事实性响应保持稳定对齐，而幻觉则在模型深度上表现出明显的语义漂移。

### 主要发现

在TruthfulQA和合成事实-幻觉数据集上评估，LSD实现了F1分数0.92、AUROC 0.96和聚类准确率0.89，性能优于SelfCheckGPT和语义熵基线方法。仅需单次前向传播，比基于采样的方法快5-20倍，同时保持了高精度和可解释性。

### 结论

LSD提供了一种可扩展的、与模型无关的机制，可用于实时幻觉监控。该研究为大型语言模型中事实一致性的几何结构提供了新的见解，有助于理解和减轻模型幻觉问题。

### 翻译

大型语言模型(LLMs)经常产生流畅但事实不正确的陈述—这种现象被称为'幻觉'—在高风险领域构成严重威胁。我们提出了层级语义动力学(LSD)，这是一种用于幻觉检测的几何框架，它分析transformer层中隐藏状态语义的演变。与依赖于多次采样或外部验证源的前沿方法不同，LSD在模型的表示空间内内在运行。使用基于边距的对比学习，LSD将隐藏激活与来自事实编码器的事实嵌入对齐，揭示出语义轨迹的明显分离：事实性响应保持稳定对齐，而幻觉则在深度上表现出明显的语义漂移。在TruthfulQA和合成事实-幻觉数据集上评估，LSD实现了F1分数0.92、AUROC 0.96和聚类准确率0.89，优于SelfCheckGPT和语义熵基线方法，同时仅需单次前向传播。这种效率比基于采样的方法快5-20倍，且不牺牲精度或可解释性。LSD为实时幻觉监控提供了一种可扩展的、与模型无关的机制，并为大型语言模型中事实一致性的几何结构提供了新的见解。


### 论文摘要

Large Language Models (LLMs) often produce fluent yet factually incorrect statements-a phenomenon known as hallucination-posing serious risks in high-stakes domains. We present Layer-wise Semantic Dynamics (LSD), a geometric framework for hallucination detection that analyzes the evolution of hidden-state semantics across transformer layers. Unlike prior methods that rely on multiple sampling passes or external verification sources, LSD operates intrinsically within the model's representational space. Using margin-based contrastive learning, LSD aligns hidden activations with ground-truth embeddings derived from a factual encoder, revealing a distinct separation in semantic trajectories: factual responses preserve stable alignment, while hallucinations exhibit pronounced semantic drift across depth. Evaluated on the TruthfulQA and synthetic factual-hallucination datasets, LSD achieves an F1-score of 0.92, AUROC of 0.96, and clustering accuracy of 0.89, outperforming SelfCheckGPT and Semantic Entropy baselines while requiring only a single forward pass. This efficiency yields a 5-20x speedup over sampling-based methods without sacrificing precision or interpretability. LSD offers a scalable, model-agnostic mechanism for real-time hallucination monitoring and provides new insights into the geometry of factual consistency within large language models.

---

## 28. A Comparative Study of Vision Transformers and CNNs for Few-Shot Rigid Transformation and Fundamental Matrix Estimation

**论文链接:** [http://arxiv.org/abs/2510.04794v1](http://arxiv.org/abs/2510.04794v1)

**作者:** Alon Kaya, Igal Bilik, Inna Stainvas

**发布时间:** 2025-10-06

### GPT解析

### 总结

本研究系统比较了Vision-transformers (ViTs)和大规模卷积神经网络(CNNs)在几何估计任务上的性能，特别是在低数据场景下的表现。研究聚焦于两个任务：估计图像对间的2D刚性变换和预测立体图像对的基本矩阵。

### 背景

Vision-transformers和大规模卷积神经网络通过预训练特征表示重塑了计算机视觉，实现了强大的迁移学习能力。然而，在涉及图像变形的低数据量几何估计任务中，它们作为骨干架构的效率仍是一个开放问题。

### 目的

比较大规模CNNs(ResNet, EfficientNet, CLIP-ResNet)与ViT基础模型(CLIP-ViT变体和DINO)在各种数据量设置下的性能，包括少样本场景，评估它们在几何估计任务上的适用性。

### 方法

系统比较预训练模型(针对分类或对比学习优化)在几何估计任务上的表现，这些模型通常专注于高级语义，但研究任务需要平衡局部和全局特征。在不同数据量设置下进行经验比较分析，包括少样本场景。

### 主要发现

在大型下游数据场景中，ViTs在精炼阶段优于CNNs；在小数据场景中，CNNs的归纳偏差和较小容量使其性能能够与ViT相匹配；ViTs在跨域评估中表现出更强的泛化能力。

### 结论

强调了为精炼仔细选择模型架构的重要性，促进了未来对混合架构的研究，这些架构能够平衡局部和全局表示。

### 翻译

Vision-transformers (ViTs) 和大规模卷积神经网络 (CNNs) 通过预训练特征表示重塑了计算机视觉，使迁移学习能够在各种任务上实现强大性能。然而，在涉及图像变形的低数据量几何估计任务中，它们作为骨干架构的效率仍是一个开放问题。本研究考虑了两个这样的任务：1) 估计图像对之间的2D刚性变换；2) 预测立体图像对的基本矩阵，这是自主移动、机器人和3D场景重建等应用中的重要问题。通过系统地比较大规模CNNs(ResNet, EfficientNet, CLIP-ResNet)与基于ViT的基础模型(CLIP-ViT变体和DINO)在各种数据量设置下的表现(包括少样本场景)，本研究解决了这个有趣的问题。这些预训练模型针对分类或对比学习进行了优化，鼓励它们主要关注高级语义。所考虑的任务需要不同程度地平衡局部和全局特征，挑战了将这些模型直接作为骨干架构的简单采用。经验比较分析表明，类似于从头开始训练，在大型下游数据场景中，ViTs在精炼阶段优于CNNs。然而，在小数据场景中，CNNs的归纳偏差和较小容量提高了它们的性能，使其能够与ViT相匹配。此外，在数据分布发生变化的跨域评估中，ViTs表现出更强的泛化能力。这些结果强调了为精炼仔细选择模型架构的重要性，促进了未来对混合架构的研究，这些架构能够平衡局部和全局表示。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在数据有限（特别是少样本）情况下，如何有效利用预训练的视觉Transformer（ViT）和卷积神经网络（CNN）进行几何估计任务的问题。具体研究两种任务：估计图像对间的2D刚性变换和预测立体图像对的基本矩阵。这个问题很重要，因为几何估计是3D场景重建、自动驾驶和机器人导航等应用的基础，而这些应用中往往标注数据有限，需要高效利用现有模型进行迁移学习。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先注意到ViT和CNN在图像分类等任务上的差异，但缺乏在几何估计任务上的系统比较。他们观察到几何估计需要平衡局部和全局特征，与传统的语义分类任务不同。因此设计了一个统一网络架构，包含特征提取模块和回归模块，使不同模型可以公平比较。他们借鉴了现有预训练模型（如CLIP-ViT、DINO-ViT、ResNet等）、基本矩阵估计的秩约束层、位置感知的最大池化技术，以及MSE和Huber损失函数等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是比较不同预训练模型在几何估计任务上的性能差异，探索数据量对性能的影响，以及不同预训练目标对下游任务的影响。整体流程包括：1) 特征提取模块：使用预训练ViT或CNN提取特征，通过卷积层和位置感知最大池化处理；2) 回归模块：针对不同任务设计特定结构（2D变换或基本矩阵估计）；3) 训练与评估：使用组合损失函数，在不同数据量条件下评估模型性能，包括少样本场景。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次系统比较ViT和CNN在几何估计任务上的性能，特别关注少样本场景；2) 揭示了数据量对模型选择的影响，发现小数据时CNN可媲美甚至优于ViT；3) 分析了不同预训练目标对下游任务的影响；4) 发现冻结ViT底层可减少小数据场景过拟合；5) 展示了ViT在跨域泛化方面的优势。相比之前工作，本文专注于迁移学习在几何任务中的应用，关注数据量影响，提供架构全面比较，并分析了模型选择的影响机制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统比较视觉Transformer和卷积神经网络在少样本几何估计任务上的性能，揭示了不同数据场景下的最优模型选择，为几何任务中的迁移学习提供了重要指导。'}


### 论文摘要

Vision-transformers (ViTs) and large-scale convolution-neural-networks (CNNs) have reshaped computer vision through pretrained feature representations that enable strong transfer learning for diverse tasks. However, their efficiency as backbone architectures for geometric estimation tasks involving image deformations in low-data regimes remains an open question. This work considers two such tasks: 1) estimating 2D rigid transformations between pairs of images and 2) predicting the fundamental matrix for stereo image pairs, an important problem in various applications, such as autonomous mobility, robotics, and 3D scene reconstruction. Addressing this intriguing question, this work systematically compares large-scale CNNs (ResNet, EfficientNet, CLIP-ResNet) with ViT-based foundation models (CLIP-ViT variants and DINO) in various data size settings, including few-shot scenarios. These pretrained models are optimized for classification or contrastive learning, encouraging them to focus mostly on high-level semantics. The considered tasks require balancing local and global features differently, challenging the straightforward adoption of these models as the backbone. Empirical comparative analysis shows that, similar to training from scratch, ViTs outperform CNNs during refinement in large downstream-data scenarios. However, in small data scenarios, the inductive bias and smaller capacity of CNNs improve their performance, allowing them to match that of a ViT. Moreover, ViTs exhibit stronger generalization in cross-domain evaluation where the data distribution changes. These results emphasize the importance of carefully selecting model architectures for refinement, motivating future research towards hybrid architectures that balance local and global representations.

---

## 29. Contrastive Learning Using Graph Embeddings for Domain Adaptation of Language Models in the Process Industry

**论文链接:** [http://arxiv.org/abs/2510.04631v1](http://arxiv.org/abs/2510.04631v1)

**作者:** Anastasia Zhukova, Jonas Lührs, Christian E. Matt, Bela Gipp

**发布时间:** 2025-10-06

**备注:** accepted to EMNLP 2025 (industry track)

### GPT解析

### 总结

本研究探索了如何将SciNCL图感知邻域对比学习方法应用于流程工业领域，通过使用从知识图谱中提取的三元组对语言模型进行微调，显著提高了文本嵌入性能并减小了模型大小。

### 背景

自然语言处理(NLP)的最新趋势是利用知识图谱(KGs)来增强预训练语言模型，通过从图结构中获取额外知识，学习特定领域的术语或文档间的关系。

### 目的

探索如何将SciNCL（一种针对科学出版物设计的图感知邻域对比学习方法）应用于流程工业领域，该领域的文本日志包含日常操作的关键信息，通常被构造成稀疏知识图谱。

### 方法

使用从知识图谱中提取的三元组对语言模型进行微调（Graph-enhanced, GE）

### 主要发现

使用GE三元组微调的语言模型在专有的流程工业文本嵌入基准(PITEB)上比最先进的mE5-large文本编码器高出9.8-14.3%（5.4-8.0个百分点），同时模型大小只有3-5倍小。

### 结论

图感知邻域对比学习方法在特定领域（流程工业）的文本嵌入任务中表现优异，不仅提高了性能，还减小了模型大小。

### 翻译

最近自然语言处理(NLP)的趋势是利用知识图谱(KGs)来增强预训练语言模型，通过从图结构中纳入额外知识来学习特定领域的术语或文档之间可能被忽视的关系。本文探讨了如何将SciNCL（一种最初为科学出版物设计的图感知邻域对比学习方法）应用于流程工业领域，该领域的文本日志包含关于日常操作的关键信息，通常被构造成稀疏知识图谱。我们的实验证明，使用从知识图谱中提取的三元组进行微调的语言模型在专有的流程工业文本嵌入基准(PITEB)上比最先进的mE5-large文本编码器高出9.8-14.3%（5.4-8.0个百分点），同时大小只有3-5倍小。


### 论文摘要

Recent trends in NLP utilize knowledge graphs (KGs) to enhance pretrained language models by incorporating additional knowledge from the graph structures to learn domain-specific terminology or relationships between documents that might otherwise be overlooked. This paper explores how SciNCL, a graph-aware neighborhood contrastive learning methodology originally designed for scientific publications, can be applied to the process industry domain, where text logs contain crucial information about daily operations and are often structured as sparse KGs. Our experiments demonstrate that language models fine-tuned with triplets derived from GE outperform a state-of-the-art mE5-large text encoder by 9.8-14.3% (5.4-8.0p) on the proprietary process industry text embedding benchmark (PITEB) while being 3-5 times smaller in size.

---

## 30. Detecting Semantic Clones of Unseen Functionality

**论文链接:** [http://arxiv.org/abs/2510.04143v1](http://arxiv.org/abs/2510.04143v1)

**作者:** Konstantinos Kitsios, Francesco Sovrano, Earl T. Barr, Alberto Bacchelli

**发布时间:** 2025-10-05

**备注:** 13 pages, 3 figures, accepted for publication (to appear) in the 40th  IEEE/ACM International Conference on Automated Software Engineering, ASE 2025

### GPT解析

### 总结

这篇论文探讨了语义代码克隆检测中模型对未见功能的泛化能力问题，提出了使用对比学习提高模型性能的方法。

### 背景

许多神经模型在语义代码克隆检测任务上表现优异，但主要擅长检测与训练数据相似的克隆，难以泛化到未见功能。

### 目的

评估现有模型在检测未见功能克隆方面的性能，并提出改进方法。

### 方法

重新评估六种最先进模型（任务特定模型和生成式LLMs），并应用对比学习技术（对比分类器和对比上下文学习）提高性能。

### 主要发现

任务特定模型在未见功能上的F1值平均下降31%，而LLMs平均仅下降3%；对比学习使任务特定模型F1值平均提高9%，LLMs平均提高3%。

### 结论

对比学习能有效提高模型在检测未见功能克隆方面的性能，特别是对于任务特定模型。

### 翻译

语义代码克隆检测是检测两个代码片段是否实现相同功能（如排序数组）的任务。最近，许多神经模型在这一任务上取得了近乎完美的性能。这些模型试图基于训练数据进行推理。因此，它们更擅长检测与训练过程中所见过的克隆相似的克隆，而可能难以检测那些未见过的克隆。当然，寻求克隆的开发者对两种类型的克隆都感兴趣。我们通过文献综述证实了这一观点，确定了三个实际的克隆检测任务，其中模型的目标是检测功能的克隆，即使它是在不同功能的克隆上训练的。基于这一发现，我们重新评估了六种最先进的模型，包括任务特定模型和生成式LLMs，在检测未见功能克隆方面的任务。我们的实验显示，任务特定模型的F1值最多下降48%（平均31%）。LLMs在没有专门针对克隆检测训练的情况下，与任务特定模型表现相当，但在未见功能上泛化能力更好，F1值最多下降5%（平均3%）。我们提出并评估了使用对比学习来提高现有模型在未见功能克隆上的性能。我们从计算机视觉和自然语言处理领域获取灵感，在这些领域中，对比学习擅长测量两个对象之间的相似性，即使它们来自训练中未见过的类别。我们用对比分类器替换了任务特定模型的最终分类器，而对于生成式LLMs，我们提出了对比上下文学习，引导LLMs专注于克隆与非克隆之间的差异。任务特定模型在未见功能克隆上的F1值最多提高了26%（平均9%），LLMs最多提高了5%（平均3%）。


### 论文摘要

Semantic code clone detection is the task of detecting whether two snippets of code implement the same functionality (e.g., Sort Array). Recently, many neural models achieved near-perfect performance on this task. These models seek to make inferences based on their training data. Consequently, they better detect clones similar to those they have seen during training and may struggle to detect those they have not. Developers seeking clones are, of course, interested in both types of clones. We confirm this claim through a literature review, identifying three practical clone detection tasks in which the model's goal is to detect clones of a functionality even if it was trained on clones of different functionalities. In light of this finding, we re-evaluate six state-of-the-art models, including both task-specific models and generative LLMs, on the task of detecting clones of unseen functionality. Our experiments reveal a drop in F1 of up to 48% (average 31%) for task-specific models. LLMs perform on par with task-specific models without explicit training for clone detection, but generalize better to unseen functionalities, where F1 drops up to 5% (average 3%) instead. We propose and evaluate the use of contrastive learning to improve the performance of existing models on clones of unseen functionality. We draw inspiration from the computer vision and natural language processing fields where contrastive learning excels at measuring similarity between two objects, even if they come from classes unseen during training. We replace the final classifier of the task-specific models with a contrastive classifier, while for the generative LLMs we propose contrastive in-context learning, guiding the LLMs to focus on the differences between clones and non-clones. The F1 on clones of unseen functionality is improved by up to 26% (average 9%) for task-specific models and up to 5% (average 3%) for LLMs.

---

## 31. Contrastive-SDE: Guiding Stochastic Differential Equations with Contrastive Learning for Unpaired Image-to-Image Translation

**论文链接:** [http://arxiv.org/abs/2510.03821v1](http://arxiv.org/abs/2510.03821v1)

**作者:** Venkata Narendra Kotyada, Revanth Eranki, Nagesh Bhattu Sristy

**发布时间:** 2025-10-04

**备注:** 9 pages, 3 figures

### GPT解析

### 总结

本文提出了一种结合对比学习和扩散模型的无配对图像到图像翻译方法，通过时间对比学习保留域不变特征并指导扩散模型进行翻译。

### 背景

无配对图像到图像翻译涉及在没有对齐或对应样本的情况下学习源域和目标域之间的映射关系。基于分数的扩散模型在生成任务中表现出色，而对比学习可以在没有配对数据的情况下学习语义相似性。

### 目的

提出一种结合对比学习和扩散模型的方法，提高无配对图像到图像翻译的效率和效果。

### 方法

提出了一种基于时间对比学习方法，使用SimCLR训练模型，将图像及其域不变特征视为正样本对，从而保留域不变特征并丢弃域特定特征。学习的对比模型指导预训练SDE进行I2I翻译任务。

### 主要发现

Contrastive-SDE在多个指标上达到了与最先进方法相当的结果，同时模型收敛速度显著加快，不需要标签监督或分类器训练。

### 结论

结合对比学习和扩散模型的方法为无配对图像到图像翻译提供了一种更高效的替代方案，能够在保持高质量翻译的同时提高训练效率。

### 翻译

无配对图像到图像翻译涉及在没有对齐或对应样本的情况下学习源域和目标域之间的映射关系。基于分数的扩散模型在生成任务中已展现出最先进的性能。它们通过随机微分方程近似复杂数据分布的能力，使其能够生成高质量和多样化的输出，特别适合无配对I2I设置。同时，对比学习提供了一种强大的框架，可以在没有明确监督或配对数据的情况下学习语义相似性。通过拉近语义相似样本的表示并推远不相似的样本，对比方法与无配对翻译的目标内在一致。在本文中，我们提出了一种基于时间对比学习方法，使用SimCLR训练模型，将图像及其域不变特征视为正样本对，从而保留域不变特征并丢弃域特定特征。学习到的对比模型随后指导预训练SDE进行I2I翻译任务。我们在三个常见的无配对I2I任务中，使用四个评估指标将Contrastive-SDE与几个基线方法进行了经验比较。Contrastive-SDE在多个指标上达到了与最先进方法相当的结果。此外，我们观察到模型收敛速度显著加快，不需要标签监督或分类器训练，使其成为该任务更高效的替代方案。


### 论文摘要

Unpaired image-to-image translation involves learning mappings between source domain and target domain in the absence of aligned or corresponding samples. Score based diffusion models have demonstrated state-of-the-art performance in generative tasks. Their ability to approximate complex data distributions through stochastic differential equations (SDEs) enables them to generate high-fidelity and diverse outputs, making them particularly well-suited for unpaired I2I settings. In parallel, contrastive learning provides a powerful framework for learning semantic similarities without the need for explicit supervision or paired data. By pulling together representations of semantically similar samples and pushing apart dissimilar ones, contrastive methods are inherently aligned with the objectives of unpaired translation. Its ability to selectively enforce semantic consistency at the feature level makes contrastive learning particularly effective for guiding generation in unpaired scenarios. In this work, we propose a time-dependent contrastive learning approach where a model is trained with SimCLR by considering an image and its domain invarient feature as a positive pair, enabling the preservation of domain-invariant features and the discarding of domain-specific ones. The learned contrastive model then guides the inference of a pretrained SDE for the I2I translation task. We empirically compare Contrastive-SDE with several baselines across three common unpaired I2I tasks, using four metrics for evaluation. Constrastive-SDE achieves comparable results to the state-of-the-art on several metrics. Furthermore, we observe that our model converges significantly faster and requires no label supervision or classifier training, making it a more efficient alternative for this task.

---

## 32. From Moments to Models: Graphon Mixture-Aware Mixup and Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.03690v1](http://arxiv.org/abs/2510.03690v1)

**作者:** Ali Azizpour, Reza Ramezanpour, Ashutosh Sabharwal, Santiago Segarra

**发布时间:** 2025-10-04

### GPT解析

### 总结

本文提出了一种统一框架，明确将图数据建模为多个潜在概率图生成模型（graphons）的混合。利用图矩（motif密度）对来自同一模型的图进行聚类，从而分离混合成分并识别不同的生成机制。该方法支持两种关键应用：图混合感知的Mixup（GMAM）数据增强技术和模型自适应的图对比学习（GCL），并改进了负采样策略。

### 背景

现实世界的图数据集通常包含多个不同群体的混合，这些图由多个不同的潜在分布生成。然而，现代的表示学习方法（如图对比学习和增强方法如Mixup）通常忽略了这种混合结构。

### 目的

提出一个统一框架，明确地将图数据建模为多个潜在概率图生成模型（以graphons表示）的混合，以解决现有方法忽略混合结构的问题。

### 方法

利用图矩（motif密度）来聚类来自同一模型的图，分离混合成分并识别不同的生成机制。实现图混合感知的Mixup（GMAM）数据增强技术和模型自适应的图对比学习（GCL），并通过引入新的模型感知目标改进负采样策略。

### 主要发现

1) 建立了新的理论保证：从具有小切割距离的graphons中采样的图将以高概率具有相似的motif密度；2) 在无监督学习中，MGCL在八个数据集上获得平均排名第一；3) 在监督学习中，GMAM在7个数据集中的6个上实现了新的最先进准确率。

### 结论

所提出的框架能够有效处理图数据中的混合结构，通过明确建模多个潜在概率图生成模型，显著提升了图表示学习的性能，在无监督和监督学习任务中均达到最先进水平。

### 翻译

现实世界的图数据集通常由多个群体的混合组成，其中图是从多个不同的潜在分布生成的。然而，现代的表示学习方法，如图对比学习（GCL）和增强方法如Mixup，通常忽略了这种混合结构。在这项工作中，我们提出了一个统一框架，明确地将数据建模为由graphons表示的多个潜在概率图生成模型的混合。为了表征这些graphons，我们利用图矩（motif密度）来聚类来自同一模型的图。这使我们能够分离混合成分并识别它们不同的生成机制。这种模型感知的分区受益于两个关键的图学习任务：1) 它实现了图混合感知的Mixup（GMAM），这是一种数据增强技术，在估计的graphons指导下在语义有效的空间中进行插值，而不是假设每个类只有一个graphon。2) 对于GCL，它实现了模型自适应和有原则的增强。此外，通过引入新的模型感知目标，我们提出的方法（称为MGCL）通过将负样本限制在其他模型的图中来改进负采样。我们建立了一个关键的理论保证：一个新的、更紧的界限表明，从具有小切割距离的graphons中采样的图将以高概率具有相似的motif密度。在基准数据集上的广泛实验展示了强大的实证性能。在无监督学习中，MGCL达到了最先进的结果，在八个数据集上获得了平均排名第一。在监督学习中，GMAM始终优于现有策略，在7个数据集中的6个上实现了新的最先进准确率。


### 论文摘要

Real-world graph datasets often consist of mixtures of populations, where graphs are generated from multiple distinct underlying distributions. However, modern representation learning approaches, such as graph contrastive learning (GCL) and augmentation methods like Mixup, typically overlook this mixture structure. In this work, we propose a unified framework that explicitly models data as a mixture of underlying probabilistic graph generative models represented by graphons. To characterize these graphons, we leverage graph moments (motif densities) to cluster graphs arising from the same model. This enables us to disentangle the mixture components and identify their distinct generative mechanisms. This model-aware partitioning benefits two key graph learning tasks: 1) It enables a graphon-mixture-aware mixup (GMAM), a data augmentation technique that interpolates in a semantically valid space guided by the estimated graphons, instead of assuming a single graphon per class. 2) For GCL, it enables model-adaptive and principled augmentations. Additionally, by introducing a new model-aware objective, our proposed approach (termed MGCL) improves negative sampling by restricting negatives to graphs from other models. We establish a key theoretical guarantee: a novel, tighter bound showing that graphs sampled from graphons with small cut distance will have similar motif densities with high probability. Extensive experiments on benchmark datasets demonstrate strong empirical performance. In unsupervised learning, MGCL achieves state-of-the-art results, obtaining the top average rank across eight datasets. In supervised learning, GMAM consistently outperforms existing strategies, achieving new state-of-the-art accuracy in 6 out of 7 datasets.

---

## 33. PEaRL: Pathway-Enhanced Representation Learning for Gene and Pathway Expression Prediction from Histology

**论文链接:** [http://arxiv.org/abs/2510.03455v1](http://arxiv.org/abs/2510.03455v1)

**作者:** Sejuti Majumder, Saarthak Kapse, Moinak Bhattacharya, Xuan Xu, Alisa Yurovsky, Prateek Prasanna

**发布时间:** 2025-10-03

### GPT解析

### 总结

PEaRL是一种多模态框架，通过通路激活分数表示转录组学，使用transformer编码生物通路信号，并通过对比学习与组织学特征对齐，在三种癌症空间转录组学数据集上表现出色，超越了现有最先进的方法。

### 背景

将组织病理学与空间转录组学结合为连接组织形态与分子功能提供了强大机会，但大多数现有的多模态方法依赖于少量高度可变的基因，这限制了预测范围，并忽视了塑造组织表型的协调生物程序。

### 目的

开发一种多模态框架，能够通过通路激活分数表示转录组学，减少维度，提高可解释性，并加强跨模态对应关系。

### 方法

PEaRL（Pathway Enhanced Representation Learning）是一种多模态框架，它使用ssGSEA计算通路激活分数来表示转录组学，使用transformer编码生物上连贯的通路信号，并通过对比学习将它们与组织学特征对齐。

### 主要发现

在三种癌症空间转录组学数据集（乳腺、皮肤和淋巴结）上，PEaRL持续超越最先进的方法，在基因和通路水平表达预测方面获得更高的准确性（与最先进方法相比，皮尔逊相关系数分别提高了高达58.9%和20.4%）。

### 结论

将转录组表示建立在通路上，可以产生更符合生物学原理且可解释的多模态模型，推动计算病理学超越基因级嵌入的发展。

### 翻译

将组织病理学与空间转录组学整合为连接组织形态与分子功能提供了强大机会。然而，大多数现有的多模态方法依赖于少量高度可变的基因，这限制了预测范围，并忽视了塑造组织表型的协调生物程序。我们提出了PEaRL（Pathway Enhanced Representation Learning），一种多模态框架，通过使用ssGSEA计算的通路激活分数来表示转录组学。通过使用transformer编码生物上连贯的通路信号，并通过对比学习将它们与组织学特征对齐，PEaRL减少了维度，提高了可解释性，并加强了跨模态对应关系。在三个癌症空间转录组学数据集（乳腺、皮肤和淋巴结）上，PEaRL持续超越最先进的方法，在基因和通路水平表达预测方面获得更高的准确性（与最先进方法相比，皮尔逊相关系数分别提高了高达58.9%和20.4%）。这些结果表明，将转录组表示建立在通路上可以产生更符合生物学原理且可解释的多模态模型，推动计算病理学超越基因级嵌入的发展。


### 论文摘要

Integrating histopathology with spatial transcriptomics (ST) provides a powerful opportunity to link tissue morphology with molecular function. Yet most existing multimodal approaches rely on a small set of highly variable genes, which limits predictive scope and overlooks the coordinated biological programs that shape tissue phenotypes. We present PEaRL (Pathway Enhanced Representation Learning), a multimodal framework that represents transcriptomics through pathway activation scores computed with ssGSEA. By encoding biologically coherent pathway signals with a transformer and aligning them with histology features via contrastive learning, PEaRL reduces dimensionality, improves interpretability, and strengthens cross-modal correspondence. Across three cancer ST datasets (breast, skin, and lymph node), PEaRL consistently outperforms SOTA methods, yielding higher accuracy for both gene- and pathway-level expression prediction (up to 58.9 percent and 20.4 percent increase in Pearson correlation coefficient compared to SOTA). These results demonstrate that grounding transcriptomic representation in pathways produces more biologically faithful and interpretable multimodal models, advancing computational pathology beyond gene-level embeddings.

---

## 34. Conditional Pseudo-Supervised Contrast for Data-Free Knowledge Distillation

**论文链接:** [http://arxiv.org/abs/2510.03375v1](http://arxiv.org/abs/2510.03375v1)

**作者:** Renrong Shao, Wei Zhang, Jun wang

**发布时间:** 2025-10-03

**DOI:** 10.1016/j.patcog.2023.109781

**备注:** 13 pages

### GPT解析

### 总结

本文提出了条件伪监督对比数据自由知识蒸馏(CPSC-DFKD)方法，解决了现有DFKD方法中的几个关键限制，提高了学生模型和生成器的性能。

### 背景

数据自由知识蒸馏(DFKD)是解决模型压缩和传输限制同时保留隐私保护的有效方法，近年来受到广泛关注。现有方法主要使用生成器合成图像来支持蒸馏过程，但仍存在诸多问题。

### 目的

探索DFKD中的伪监督范式；解决当前合成方法无法区分不同类别样本分布的问题；优化类别多样性的样本，帮助学生模型从多样样本中学习。

### 方法

提出了条件伪监督对比数据自由知识蒸馏(CPSC-DFKD)方法；引入条件生成对抗网络合成特定类别的多样图像用于伪监督学习；改进生成器模块以区分不同类别的分布；提出基于教师和学生视图的伪监督对比学习以增强多样性。

### 主要发现

在三个常用数据集上的综合实验验证了CPSC-DFKD带来的学生模型和生成器性能提升。

### 结论

CPSC-DFKD解决了现有DFKD方法的几个关键限制，包括无法区分不同类别样本分布和无法优化类别多样性样本的问题，通过创新的伪监督对比学习范式显著提高了模型性能。

### 翻译

无监督知识蒸馏(DFKD)是解决模型压缩和传输限制同时保留隐私保护的有效方式，近年来引起了广泛关注。目前，大多数现有方法利用生成器合成图像来支持蒸馏过程。尽管当前方法已取得巨大成功，但仍有许多问题有待探索。首先，深度学习中监督学习的卓越性能促使我们在DFKD上探索伪监督范式。其次，当前合成方法无法区分不同类别样本的分布，从而产生可能被教师模型错误评估的模糊样本。此外，当前方法无法优化类别多样性的样本，这将阻碍学生模型从多样样本中学习并取得更好性能。在本文中，为解决上述限制，我们提出了一个新的学习范式，即条件伪监督对比数据自由知识蒸馏(CPSC-DFKD)。CPSC-DFKD的主要创新包括：(1)引入条件生成对抗网络合成特定类别的多样图像用于伪监督学习，(2)改进生成器模块以区分不同类别的分布，(3)提出基于教师和学生视图的伪监督对比学习以增强多样性。在三个常用数据集上的综合实验验证了CPSC-DFKD带来的学生模型和生成器性能提升。代码可在https://github.com/RoryShao/CPSC-DFKD.git获取。


### 论文摘要

Data-free knowledge distillation~(DFKD) is an effective manner to solve model compression and transmission restrictions while retaining privacy protection, which has attracted extensive attention in recent years. Currently, the majority of existing methods utilize a generator to synthesize images to support the distillation. Although the current methods have achieved great success, there are still many issues to be explored. Firstly, the outstanding performance of supervised learning in deep learning drives us to explore a pseudo-supervised paradigm on DFKD. Secondly, current synthesized methods cannot distinguish the distributions of different categories of samples, thus producing ambiguous samples that may lead to an incorrect evaluation by the teacher. Besides, current methods cannot optimize the category-wise diversity samples, which will hinder the student model learning from diverse samples and further achieving better performance. In this paper, to address the above limitations, we propose a novel learning paradigm, i.e., conditional pseudo-supervised contrast for data-free knowledge distillation~(CPSC-DFKD). The primary innovations of CPSC-DFKD are: (1) introducing a conditional generative adversarial network to synthesize category-specific diverse images for pseudo-supervised learning, (2) improving the modules of the generator to distinguish the distributions of different categories, and (3) proposing pseudo-supervised contrastive learning based on teacher and student views to enhance diversity. Comprehensive experiments on three commonly-used datasets validate the performance lift of both the student and generator brought by CPSC-DFKD. The code is available at https://github.com/RoryShao/CPSC-DFKD.git

---

## 35. Latent Multi-view Learning for Robust Environmental Sound Representations

**论文链接:** [http://arxiv.org/abs/2510.02500v2](http://arxiv.org/abs/2510.02500v2)

**作者:** Sivan Ding, Julia Wilkins, Magdalena Fuentes, Juan Pablo Bello

**发布时间:** 2025-10-02

**备注:** Accepted to DCASE 2025 Workshop. 4+1 pages, 2 figures, 2 tables

### GPT解析

### 总结

论文提出了一种多视图学习框架，整合对比原则到生成管道中，用于捕获声音源和设备信息，通过两个自监督目标实现更好的环境声音表示学习。

### 背景

自监督学习方法（如对比和生成方法）已使用无标签数据推动了环境声音表示学习的进步，但这些方法如何在统一框架中相互补充仍较少研究。

### 目的

研究对比方法和生成方法如何在统一框架中相互补充，并开发能捕获声音源和设备信息的多视图学习框架。

### 方法

提出多视图学习框架，将压缩音频潜在变量编码为视图特定和视图共享的子空间，由对比学习（促进子空间间信息流）和重建（保存整体信息）两个自监督目标指导。

### 主要发现

在城市声音传感器网络数据集上的评估显示，该方法在声音源和传感器分类任务中优于传统自监督技术，且模型具有在结构化潜在空间中解离环境声音属性的潜力。

### 结论

所提出的多视图学习框架成功整合了对比和生成方法，在环境声音表示学习中实现了更好的性能，并能解离环境声音属性。

### 翻译

自监督学习方法（如对比和生成方法）已经使用无标签数据推动了环境声音表示学习的进步。然而，这些方法如何在统一框架中相互补充仍然相对未被探索。在这项工作中，我们提出了一种多视图学习框架，将对比原则整合到生成管道中，以捕获声音源和设备信息。我们的方法将压缩音频潜在变量编码为视图特定和视图共享的子空间，由两个自监督目标指导：子空间之间的目标信息流的对比学习，以及整体信息保存的重建。我们在城市声音传感器网络数据集上评估了我们的方法，用于声音源和传感器分类，展示了与传统自监督技术相比改进的下游性能。此外，我们还研究了模型在变化训练配置下在结构化潜在空间中解离环境声音属性的潜力。


### 论文摘要

Self-supervised learning (SSL) approaches, such as contrastive and generative methods, have advanced environmental sound representation learning using unlabeled data. However, how these approaches can complement each other within a unified framework remains relatively underexplored. In this work, we propose a multi-view learning framework that integrates contrastive principles into a generative pipeline to capture sound source and device information. Our method encodes compressed audio latents into view-specific and view-common subspaces, guided by two self-supervised objectives: contrastive learning for targeted information flow between subspaces, and reconstruction for overall information preservation. We evaluate our method on an urban sound sensor network dataset for sound source and sensor classification, demonstrating improved downstream performance over traditional SSL techniques. Additionally, we investigate the model's potential to disentangle environmental sound attributes within the structured latent space under varied training configurations.

---

## 36. Thin Bridges for Drug Text Alignment: Lightweight Contrastive Learning for Target Specific Drug Retrieval

**论文链接:** [http://arxiv.org/abs/2510.03309v1](http://arxiv.org/abs/2510.03309v1)

**作者:** Mallikarjuna Tupakula

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究提出了一种轻量级的多模态表示对齐方法，通过使用轻量级投影头对齐化学和文本表示，无需训练完整的多模态模型。

### 背景

多模态基础模型在药物发现和生物医学应用方面有潜力，但大多数现有方法依赖于大量预训练或大规模多模态语料库，计算成本较高。

### 目的

研究是否可以通过轻量级的对比桥梁对齐化学和文本表示，而不需要训练完整的多模态模型，以降低计算成本。

### 方法

使用ChEMBL中的配对机制，通过双重线性投影和对比目标训练，将ECFP4分子指纹与生物医学句子嵌入进行对齐。为处理共享相同治疗靶点的药物，集成了困难负样本加权和边缘损失。

### 主要发现

基于骨架分割的评估表明，该方法实现了非平凡的跨模态对齐，与冻结基线相比显著改善了靶点内辨别能力。

### 结论

轻量级桥梁是大规模多模态预训练的高效替代方案，能够在精准医学中实现骨架感知的药物文本对齐和靶点特异性检索。

### 翻译

多模态基础模型在药物发现和生物医学应用方面有前景，但大多数现有方法依赖于大量预训练或大规模多模态语料库。我们研究了轻量级对比桥梁（冻结单模态编码器上的轻量级投影头）是否能够在不训练完整多模态模型的情况下对齐化学和文本表示。使用来自ChEMBL的配对机制，我们通过对比目标训练的双重线性投影将ECFP4分子指纹与生物医学句子嵌入进行对齐。为了更好地处理共享相同治疗靶点的药物，我们集成了困难负样本加权和边缘损失。基于骨架分割的评估（需要在不同化学核心之间泛化）表明，我们的方法实现了非平凡的跨模态对齐，并且与冻结基线相比显著改善了靶点内辨别能力。这些结果表明轻量级桥梁是大规模多模态预训练的高效替代方案，能够在精准医学中实现骨架感知的药物文本对齐和靶点特异性检索。


### 论文摘要

Multimodal foundation models hold promise for drug discovery and biomedical applications, but most existing approaches rely on heavy pretraining or large scale multimodal corpora. We investigate whether thin contrastive bridges, lightweight projection heads over frozen unimodal encoders can align chemical and textual representations without training a full multimodal model. Using paired mechanisms from ChEMBL, we align ECFP4 molecular fingerprints with biomedical sentence embeddings through dual linear projections trained with a contrastive objective. To better handle drugs sharing the same therapeutic target, we incorporate hard negative weighting and a margin loss. Evaluation under scaffold based splits, which require generalization across disjoint chemical cores, demonstrates that our approach achieves non-trivial cross modal alignment and substantially improves within target discrimination compared to frozen baselines. These results suggest that thin bridges offer a compute efficient alternative to large scale multimodal pretraining, enabling scaffold aware drug text alignment and target specific retrieval in precision medicine.

---

## 37. StaMo: Unsupervised Learning of Generalizable Robot Motion from Compact State Representation

**论文链接:** [http://arxiv.org/abs/2510.05057v1](http://arxiv.org/abs/2510.05057v1)

**作者:** Mingyu Liu, Jiuhe Shu, Hui Chen, Zeju Li, Canyu Zhao, Jiange Yang, Shenyuan Gao, Hao Chen, Chunhua Shen

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出了一种名为StaMo的无监督方法，通过轻量级编码器和预训练的扩散Transformer解码器学习高度压缩的双令牌状态表示，用于高效的世界建模和决策制定。该方法在LIBERO任务上提高14.3%性能，在真实世界任务中提高30%成功率，且推理开销小。研究发现，令牌间的差异可作为有效潜在动作，无需显式监督即可捕捉结构化动态，并增强策略协同训练，比先前方法提高10.4%。

### 背景

具身智能面临的一个基本挑战是开发具有表现力和紧凑的状态表示，用于高效的世界建模和决策制定。然而，现有方法往往无法平衡这两点，产生的表示要么过于冗余，要么缺乏任务关键信息。

### 目的

提出一种无监督方法，学习高度压缩的双令牌状态表示，用于高效的世界建模和决策制定，解决现有方法在表示冗余性和信息完整性之间的平衡问题。

### 方法

使用轻量级编码器和预训练的扩散Transformer (DiT) 解码器，利用其强大的生成先验，创建高效、可解释的状态表示，并无缝集成到现有的基于VLA的模型中。

### 主要发现

1) 在LIBERO上提高14.3%性能，在真实世界任务中提高30%成功率，推理开销最小；2) 通过潜在插值获得的令牌间差异可作为有效潜在动作，解码为可执行机器人动作；3) 这种能力表明表示在无需显式监督的情况下捕捉了结构化动态；4) 潜在动作增强了策略协同训练，比先前方法提高10.4%，可解释性更好。

### 结论

StaMo方法能够从紧凑的状态表示学习可泛化的机器人运动，该状态表示从静态图像编码，挑战了学习潜在动作对复杂架构和视频数据的普遍依赖。该方法能有效扩展到各种数据源，包括真实机器人数据、模拟和人类第一人称视频。

### 翻译

具身智能的一个基本挑战是开发具有表现力和紧凑的状态表示，用于高效的世界建模和决策制定。然而，现有方法往往无法实现这种平衡，产生的表示要么过于冗余，要么缺乏任务关键信息。我们提出一种无监督方法，使用轻量级编码器和预训练的扩散Transformer (DiT) 解码器学习高度压缩的双令牌状态表示，利用其强大的生成先验。我们的表示是高效的、可解释的，并能无缝集成到现有的基于VLA的模型中，在LIBERO上提高14.3%的性能，在真实世界任务成功率上提高30%，且推理开销最小。更重要的是，我们发现通过潜在插值获得的这两个令牌之间的差异自然地作为非常有效的潜在动作，可以进一步解码为可执行的机器人动作。这种涌现能力表明，我们的表示在无需显式监督的情况下捕捉了结构化动态。我们将此方法命名为StaMo，因为它能够从紧凑的状态表示学习可泛化的机器人运动，该状态表示从静态图像编码，挑战了学习潜在动作对复杂架构和视频数据的普遍依赖。产生的潜在动作也增强了策略协同训练，比先前方法高出10.4%，且可解释性更好。此外，我们的方法能够有效地扩展到各种数据源，包括真实机器人数据、模拟和人类第一人称视频。


### 论文摘要

A fundamental challenge in embodied intelligence is developing expressive and compact state representations for efficient world modeling and decision making. However, existing methods often fail to achieve this balance, yielding representations that are either overly redundant or lacking in task-critical information. We propose an unsupervised approach that learns a highly compressed two-token state representation using a lightweight encoder and a pre-trained Diffusion Transformer (DiT) decoder, capitalizing on its strong generative prior. Our representation is efficient, interpretable, and integrates seamlessly into existing VLA-based models, improving performance by 14.3% on LIBERO and 30% in real-world task success with minimal inference overhead. More importantly, we find that the difference between these tokens, obtained via latent interpolation, naturally serves as a highly effective latent action, which can be further decoded into executable robot actions. This emergent capability reveals that our representation captures structured dynamics without explicit supervision. We name our method StaMo for its ability to learn generalizable robotic Motion from compact State representation, which is encoded from static images, challenging the prevalent dependence to learning latent action on complex architectures and video data. The resulting latent actions also enhance policy co-training, outperforming prior methods by 10.4% with improved interpretability. Moreover, our approach scales effectively across diverse data sources, including real-world robot data, simulation, and human egocentric video.

---

## 38. Exploring the Efficacy of Modified Transfer Learning in Identifying Parkinson's Disease Through Drawn Image Patterns

**论文链接:** [http://arxiv.org/abs/2510.05015v1](http://arxiv.org/abs/2510.05015v1)

**作者:** Nabil Daiyan, Md Rakibul Haque

**发布时间:** 2025-10-06

**备注:** 5 pages, 11 figures, published on 2024 2nd International Conference  on Information and Communication Technology (ICICT 2024)

### GPT解析

### 总结

该研究提出了一种基于机器学习的帕金森病诊断方法，利用手绘螺旋和波形图像作为生物标志物，通过卷积神经网络、迁移学习和注意力机制构建模型，实现了93.3%的总体准确率。

### 背景

帕金森病是一种进行性神经退行性疾病，由多巴胺能神经元死亡导致各种运动障碍症状。早期诊断对预防不良后果至关重要，但传统诊断方法通常繁琐且昂贵。

### 目的

开发一种基于机器学习的帕金森病检测方法，使用手绘螺旋和波形图像作为潜在生物标志物，提供一种非侵入性且经济有效的诊断解决方案。

### 方法

采用卷积神经网络、迁移学习和注意力机制提高模型性能和抗过拟合能力；通过数据增强增加图像数量提高多样性；架构分为三个阶段：使用预训练CNN、加入自定义卷积层、集成投票；采用硬投票聚合多个模型预测以提高性能。

### 主要发现

螺旋图像的加权平均精确率、召回率和F1分数为90%；波形图像的加权平均精确率、召回率和F1分数为96.67%；通过集成硬投票组合预测后，总体准确率为93.3%。

### 结论

机器学习在早期帕金森病诊断中具有潜力，提供了一种非侵入性且经济有效的解决方案，可以改善患者预后。

### 翻译

帕金森病是一种进行性神经退行性疾病，其特点是多巴胺能神经元死亡，导致各种运动障碍症状。早期诊断对预防不良后果至关重要，但传统诊断方法通常繁琐且昂贵。本研究提出了一种基于机器学习方法，使用手绘螺旋和波形图像作为帕金森病检测的潜在生物标志物。我们的方法利用卷积神经网络、迁移学习和注意力机制来提高模型性能和抗过拟合能力。为了增强螺旋和波形类别的多样性和丰富性，训练数据集经过增强以增加图像数量。所提出的架构包含三个阶段：利用预训练CNN、加入自定义卷积层和集成投票。采用硬投票通过聚合多个模型的预测进一步提高了性能。实验结果显示了有希望的准确率。对于螺旋图像，加权平均精确率、召回率和F1分数为90%，对于波形图像为96.67%。通过集成硬投票组合预测后，总体准确率为93.3%。这些发现强调了机器学习在早期帕金森病诊断中的潜力，提供了一种非侵入性且经济有效的解决方案来改善患者预后。


### 论文摘要

Parkinson's disease (PD) is a progressive neurodegenerative condition characterized by the death of dopaminergic neurons, leading to various movement disorder symptoms. Early diagnosis of PD is crucial to prevent adverse effects, yet traditional diagnostic methods are often cumbersome and costly. In this study, a machine learning-based approach is proposed using hand-drawn spiral and wave images as potential biomarkers for PD detection. Our methodology leverages convolutional neural networks (CNNs), transfer learning, and attention mechanisms to improve model performance and resilience against overfitting. To enhance the diversity and richness of both spiral and wave categories, the training dataset undergoes augmentation to increase the number of images. The proposed architecture comprises three phases: utilizing pre-trained CNNs, incorporating custom convolutional layers, and ensemble voting. Employing hard voting further enhances performance by aggregating predictions from multiple models. Experimental results show promising accuracy rates. For spiral images, weighted average precision, recall, and F1-score are 90%, and for wave images, they are 96.67%. After combining the predictions through ensemble hard voting, the overall accuracy is 93.3%. These findings underscore the potential of machine learning in early PD diagnosis, offering a non-invasive and cost-effective solution to improve patient outcomes.

---

## 39. Comparative Analysis of YOLOv5, Faster R-CNN, SSD, and RetinaNet for Motorbike Detection in Kigali Autonomous Driving Context

**论文链接:** [http://arxiv.org/abs/2510.04912v1](http://arxiv.org/abs/2510.04912v1)

**作者:** Ngeyen Yinkfu, Sunday Nwovu, Jonathan Kayizzi, Angelique Uwamahoro

**发布时间:** 2025-10-06

**备注:** 3 figures, 2 tables

### GPT解析

### 总结

本研究比较了四种目标检测模型在摩托车检测方面的性能，旨在为资源受限环境下的自动驾驶系统提供解决方案。

### 背景

在卢旺达基加利，摩托车出租车是主要交通方式，它们行驶路线不可预测且经常无视交通规则，这对自动驾驶系统构成了重大挑战。

### 目的

比较四种目标检测模型（YOLOv5、Faster R-CNN、SSD和RetinaNet）在摩托车检测方面的性能，以评估它们在资源受限环境下的实时导航适用性。

### 方法

使用在基加利收集的198张自定义数据集，在PyTorch框架下实现这些模型，并采用迁移学习。评估了模型的准确性、定位能力和推理速度。

### 主要发现

确定了实施挑战，包括数据集限制和模型复杂性。

### 结论

建议简化架构，以提高发展中国家如卢旺达的自动驾驶系统的可访问性。

### 翻译

在卢旺达基加利，摩托车出租车是主要交通方式，它们常常不可预测地行驶且无视交通规则，对自动驾驶系统构成重大挑战。本研究比较了四种目标检测模型——YOLOv5、Faster R-CNN、SSD和RetinaNet——使用在基加利收集的198张自定义数据集进行摩托车检测。这些模型在PyTorch中通过迁移学习实现，并评估了其准确性、定位能力和推理速度，以评估它们在资源受限环境下实时导航的适用性。我们确定了实施挑战，包括数据集限制和模型复杂性，并建议简化架构，以提高发展中国家如卢旺达的自动驾驶系统的可访问性。


### 论文摘要

In Kigali, Rwanda, motorcycle taxis are a primary mode of transportation, often navigating unpredictably and disregarding traffic rules, posing significant challenges for autonomous driving systems. This study compares four object detection models--YOLOv5, Faster R-CNN, SSD, and RetinaNet--for motorbike detection using a custom dataset of 198 images collected in Kigali. Implemented in PyTorch with transfer learning, the models were evaluated for accuracy, localization, and inference speed to assess their suitability for real-time navigation in resource-constrained settings. We identify implementation challenges, including dataset limitations and model complexities, and recommend simplified architectures for future work to enhance accessibility for autonomous systems in developing countries like Rwanda.

---

## 40. From Actions to Kinesics: Extracting Human Psychological States through Bodily Movements

**论文链接:** [http://arxiv.org/abs/2510.04844v1](http://arxiv.org/abs/2510.04844v1)

**作者:** Cheyu Lin, Katherine A. Flanigan

**发布时间:** 2025-10-06

**备注:** The 15th International Workshop on Structural Health Monitoring  (IWSHM)

### GPT解析

### 总结

研究提出了一种身势语识别框架，能够从3D骨骼关节数据中直接推断人类活动的交流功能，结合时空图卷积网络和卷积神经网络，利用迁移学习实现可扩展、准确且以人为本的行为建模。

### 背景

理解人类与建成环境之间的动态关系是环境心理学到强化学习等多个学科的关键挑战。建模这些互动的主要障碍是无法以既可泛化又保护隐私的方式捕捉人类心理状态。传统方法依赖理论模型或问卷，存在范围有限、静态和劳动密集等限制。

### 目的

开发一种能够直接从3D骨骼关节数据推断人类活动交流功能(身势语)的框架，实现可扩展、准确且以人为本的行为建模，为增强基于强化学习的人类-环境互动模拟提供新途径。

### 方法

提出一个身势语识别框架，结合时空图卷积网络(ST-GCN)和卷积神经网络(CNN)，利用迁移学习来绕过手动定义物理动作与心理类别之间映射的需求，保留用户匿名性的同时揭示反映认知和情绪状态的潜在身体运动结构。

### 主要发现

在Dyadic User EngagemenT (DUET)数据集上的结果表明，该方法能够实现可扩展、准确且以人为本的行为建模，为增强基于强化学习的人类-环境互动模拟提供了新的途径。

### 结论

该框架为理解人类与建成环境之间的动态关系提供了一种新方法，通过直接从身体动作数据推断心理状态，解决了传统方法的局限性，既保护了用户隐私，又能够捕捉人类心理状态。

### 翻译

理解人类与建成环境之间的动态关系是环境心理学到强化学习(RL)等多个学科的关键挑战。建模这些互动的主要障碍是无法以既可泛化又保护隐私的方式捕捉人类心理状态。传统方法依赖理论模型或问卷，存在范围有限、静态和劳动密集等限制。我们提出一个身势语识别框架，能够直接从3D骨骼关节数据推断人类活动的交流功能——即身势语。结合时空图卷积网络(ST-GCN)和卷积神经网络(CNN)，该框架利用迁移学习来绕过手动定义物理动作与心理类别之间映射的需求。该方法保留用户匿名性，同时揭示反映认知和情绪状态的潜在身体运动结构。我们在Dyadic User EngagemenT (DUET)数据集上的结果表明，该方法能够实现可扩展、准确且以人为本的行为建模，为增强基于RL的人类-环境互动模拟提供了新的途径。


### 论文摘要

Understanding the dynamic relationship between humans and the built environment is a key challenge in disciplines ranging from environmental psychology to reinforcement learning (RL). A central obstacle in modeling these interactions is the inability to capture human psychological states in a way that is both generalizable and privacy preserving. Traditional methods rely on theoretical models or questionnaires, which are limited in scope, static, and labor intensive. We present a kinesics recognition framework that infers the communicative functions of human activity -- known as kinesics -- directly from 3D skeleton joint data. Combining a spatial-temporal graph convolutional network (ST-GCN) with a convolutional neural network (CNN), the framework leverages transfer learning to bypass the need for manually defined mappings between physical actions and psychological categories. The approach preserves user anonymity while uncovering latent structures in bodily movements that reflect cognitive and emotional states. Our results on the Dyadic User EngagemenT (DUET) dataset demonstrate that this method enables scalable, accurate, and human-centered modeling of behavior, offering a new pathway for enhancing RL-driven simulations of human-environment interaction.

---

## 41. Beyond Appearance: Transformer-based Person Identification from Conversational Dynamics

**论文链接:** [http://arxiv.org/abs/2510.04753v1](http://arxiv.org/abs/2510.04753v1)

**作者:** Masoumeh Chapariniya, Teodora Vukovic, Sarah Ebling, Volker Dellwo

**发布时间:** 2025-10-06

### GPT解析

### 总结

该研究探究了基于Transformer的架构在自然面对面对话场景中人物识别的性能，通过实现并评估一个双流框架，分别对关键点的空间配置和时间运动模式进行建模。

### 背景

研究基于自然面对面对话场景下的人物识别，使用基于Transformer的架构。

### 目的

探究基于Transformer的架构在自然对话场景中人物识别的性能。

### 方法

实现并评估了一个双流框架，分别对133个COCO WholeBody关键点的空间配置和时间运动模式进行建模；从CANDOR对话语料库的子集中提取关键点；比较了预训练和从头开始的训练；研究了速度特征的使用；引入了多尺度时间Transformer进行分层运动建模。

### 主要发现

领域特定的训练显著优于迁移学习；空间配置比时间动态包含更多的判别信息；空间Transformer达到95.74%的准确率；多尺度时间Transformer达到93.90%的准确率；特征级融合将性能提升到98.03%，证实了姿态和动态信息是互补的。

### 结论

这些发现突出了Transformer架构在自然交互中人物识别的潜力，并为未来的多模态和跨文化研究提供了见解。

### 翻译

本文研究了基于Transformer的架构在自然面对面对话场景下的人物识别性能。我们实现并评估了一个双流框架，分别对从CANDOR对话语料库子集中提取的133个COCO WholeBody关键点的空间配置和时间运动模式进行建模。我们的实验比较了预训练和从头开始的训练，研究了速度特征的使用，并引入了多尺度时间Transformer进行分层运动建模。结果表明，领域特定的训练显著优于迁移学习，且空间配置比时间动态包含更多的判别信息。空间Transformer达到95.74%的准确率，而多尺度时间Transformer达到93.90%的准确率。特征级融合将性能提升到98.03%，证实了姿态和动态信息是互补的。这些发现突出了Transformer架构在自然交互中人物识别的潜力，并为未来的多模态和跨文化研究提供了见解。


### 论文摘要

This paper investigates the performance of transformer-based architectures for person identification in natural, face-to-face conversation scenario. We implement and evaluate a two-stream framework that separately models spatial configurations and temporal motion patterns of 133 COCO WholeBody keypoints, extracted from a subset of the CANDOR conversational corpus. Our experiments compare pre-trained and from-scratch training, investigate the use of velocity features, and introduce a multi-scale temporal transformer for hierarchical motion modeling. Results demonstrate that domain-specific training significantly outperforms transfer learning, and that spatial configurations carry more discriminative information than temporal dynamics. The spatial transformer achieves 95.74% accuracy, while the multi-scale temporal transformer achieves 93.90%. Feature-level fusion pushes performance to 98.03%, confirming that postural and dynamic information are complementary. These findings highlight the potential of transformer architectures for person identification in natural interactions and provide insights for future multimodal and cross-cultural studies.

---

## 42. Busemann Functions in the Wasserstein Space: Existence, Closed-Forms, and Applications to Slicing

**论文链接:** [http://arxiv.org/abs/2510.04579v1](http://arxiv.org/abs/2510.04579v1)

**作者:** Clément Bonet, Elsa Cazelles, Lucas Drumetz, Nicolas Courty

**发布时间:** 2025-10-06

### GPT解析

### 总结

本研究探讨了Wasserstein空间中Busemann函数的存在性和计算问题，并建立了两个重要情况下的闭式表达式，为概率分布设计了明确的投影方案，并定义了新的Sliced-Wasserstein距离。

### 背景

Busemann函数在几何机器学习问题中受到广泛关注，它自然定义了Riemannian流形上的测地射线投影并推广了超平面概念。许多数据源可建模为概率分布，而Wasserstein空间具有由最优传输指标诱导的丰富形式Riemannian结构。

### 目的

研究Wasserstein空间中Busemann函数的存在性和计算，该空间允许测地射线。

### 方法

建立一维分布和高斯测度两种重要情况下的闭式表达式，为实数上的概率分布设计明确的投影方案，并定义高斯混合和标记数据集上的Sliced-Wasserstein距离。

### 主要发现

在一维分布和高斯测度情况下获得了闭式表达式，设计了针对概率分布的明确投影方案，并定义了新的Sliced-Wasserstein距离。

### 结论

在合成数据集和迁移学习问题上展示了这些原始方案的有效性。

### 翻译

Busemann函数最近在多种几何机器学习问题中引起了广泛关注，因为它自然地定义了Riemannian流形上的测地射线投影，并推广了超平面的概念。由于多个数据源可以方便地建模为概率分布，因此在具有由最优传输指标诱导的丰富形式Riemannian结构的Wasserstein空间中研究该函数是很自然的。在本工作中，我们研究了Wasserstein空间中Busemann函数的存在性和计算，该空间允许测地射线。我们在两个重要情况下建立了闭式表达式：一维分布和高斯测度。这些结果使得能够为实数上的概率分布设计明确的投影方案，进而使我们能够定义高斯混合和标记数据集上的新型Sliced-Wasserstein距离。我们在合成数据集以及迁移学习问题上展示了这些原始方案的有效性。


### 论文摘要

The Busemann function has recently found much interest in a variety of geometric machine learning problems, as it naturally defines projections onto geodesic rays of Riemannian manifolds and generalizes the notion of hyperplanes. As several sources of data can be conveniently modeled as probability distributions, it is natural to study this function in the Wasserstein space, which carries a rich formal Riemannian structure induced by Optimal Transport metrics. In this work, we investigate the existence and computation of Busemann functions in Wasserstein space, which admits geodesic rays. We establish closed-form expressions in two important cases: one-dimensional distributions and Gaussian measures. These results enable explicit projection schemes for probability distributions on $\mathbb{R}$, which in turn allow us to define novel Sliced-Wasserstein distances over Gaussian mixtures and labeled datasets. We demonstrate the efficiency of those original schemes on synthetic datasets as well as transfer learning problems.

---

## 43. Categorical Invariants of Learning Dynamics

**论文链接:** [http://arxiv.org/abs/2510.04376v1](http://arxiv.org/abs/2510.04376v1)

**作者:** Abdulrahman Tamim

**发布时间:** 2025-10-05

### GPT解析

### 总结

本文提出了一种关于神经网络训练的新视角，将学习视为参数空间和已学习表征空间之间的结构保持变换，而非传统的梯度下降方法。

### 背景

传统观点认为神经网络训练是在损失面上进行梯度下降的过程。

### 目的

提出一种新的理论框架来理解神经网络学习和泛化的本质。

### 方法

使用范畴论中的函子概念描述学习过程，将学习定义为从参数空间到已学习表征空间的函子L，并通过持续同调和拉回构造等数学工具分析优化路径。

### 主要发现

1) 不同训练过程若产生相似测试性能，通常属于优化路径的同一同伦类；2) 通过同伦轨迹收敛的网络泛化性能相差在0.5%准确率内，而非同伦路径则相差超过3%；3) 持续同调可识别与泛化相关的稳定最小值；4) 拉回构造形式化了迁移学习；5) 2-分类结构解释了不同优化算法产生功能等效模型的条件。

### 结论

这种范畴论框架不仅提供了深度学习为何有效的理论见解，也为训练更鲁棒的网络提供了具体的算法原则。

### 翻译

神经网络训练通常被视为在损失面上的梯度下降。我们提出了一种根本不同的视角：学习是参数空间和已学习表征空间之间的结构保持变换。这种范畴框架揭示，产生相似测试性能的不同训练过程通常属于优化路径的同一同伦类。我们通过实验证明，通过同伦轨迹收敛的网络泛化性能彼此相差在0.5%的准确率内，而非同伦路径则相差超过3%。该理论提供了实用工具：持续同调识别可预测泛化的稳定最小值，拉回构造形式化了迁移学习，2-分类结构解释了不同优化算法何时产生功能等效的模型。这些分类不变量既提供了深度学习为何有效的理论见解，也为训练更鲁棒的网络提供了具体的算法原则。


### 论文摘要

Neural network training is typically viewed as gradient descent on a loss surface. We propose a fundamentally different perspective: learning is a structure-preserving transformation (a functor L) between the space of network parameters (Param) and the space of learned representations (Rep). This categorical framework reveals that different training runs producing similar test performance often belong to the same homotopy class (continuous deformation family) of optimization paths. We show experimentally that networks converging via homotopic trajectories generalize within 0.5% accuracy of each other, while non-homotopic paths differ by over 3%. The theory provides practical tools: persistent homology identifies stable minima predictive of generalization (R^2 = 0.82 correlation), pullback constructions formalize transfer learning, and 2-categorical structures explain when different optimization algorithms yield functionally equivalent models. These categorical invariants offer both theoretical insight into why deep learning works and concrete algorithmic principles for training more robust networks.

---

## 44. A Multilingual Framework for Dysarthria: Detection, Severity Classification, Speech-to-Text, and Clean Speech Generation

**论文链接:** [http://arxiv.org/abs/2510.03986v1](http://arxiv.org/abs/2510.03986v1)

**作者:** Ananya Raghu, Anisha Raghu, Nithika Vivek, Sofie Budman, Omar Mansour

**发布时间:** 2025-10-05

### GPT解析

### 总结

该研究提出一个基于AI的多语言统一框架，用于构音障碍的检测、分类、语音生成、语音转文本、情感检测和语音克隆，在英语、俄语和德语数据集上表现出色。

### 背景

构音障碍是一种运动性言语障碍，导致言语缓慢且难以理解，严重影响社交互动。它是帕金森病和ALS等神经系统疾病的特征，但现有工具缺乏跨语言和不同严重程度的泛化能力。

### 目的

开发一个解决构音障碍六个关键方面的AI多语言框架：二元构音障碍检测、严重程度分类、清晰语音生成、语音转文本转换、情感检测和语音克隆。

### 方法

分析英语、俄语和德语数据集，使用基于频谱图的可视化和声学特征提取来指导模型训练，开发六个关键组件的模型并评估其性能。

### 主要发现

二元检测模型在三种语言上达到97%准确率；严重程度分类模型也达到97%测试准确率；俄语翻译管道实现了低L1损失的可理解输出；通过跨语言迁移学习，英语模型微调后获得更好结果；语音转文本管道达到0.1367的词错误率。

### 结论

该研究的结果和产品可用于诊断构音障碍，并改善不同语言患者的沟通和理解。

### 翻译

研究训练了基于俄语构音障碍-清晰语音对的翻译管道，并探索了跨语言迁移学习，将俄语模型在英语数据上进行微调，实现了低资源设置下的有效应用。


### 论文摘要

Dysarthria is a motor speech disorder that results in slow and often incomprehensible speech. Speech intelligibility significantly impacts communication, leading to barriers in social interactions. Dysarthria is often a characteristic of neurological diseases including Parkinson's and ALS, yet current tools lack generalizability across languages and levels of severity. In this study, we present a unified AI-based multilingual framework that addresses six key components: (1) binary dysarthria detection, (2) severity classification, (3) clean speech generation, (4) speech-to-text conversion, (5) emotion detection, and (6) voice cloning. We analyze datasets in English, Russian, and German, using spectrogram-based visualizations and acoustic feature extraction to inform model training. Our binary detection model achieved 97% accuracy across all three languages, demonstrating strong generalization across languages. The severity classification model also reached 97% test accuracy, with interpretable results showing model attention focused on lower harmonics. Our translation pipeline, trained on paired Russian dysarthric and clean speech, reconstructed intelligible outputs with low training (0.03) and test (0.06) L1 losses. Given the limited availability of English dysarthric-clean pairs, we fine-tuned the Russian model on English data and achieved improved losses of 0.02 (train) and 0.03 (test), highlighting the promise of cross-lingual transfer learning for low-resource settings. Our speech-to-text pipeline achieved a Word Error Rate of 0.1367 after three epochs, indicating accurate transcription on dysarthric speech and enabling downstream emotion recognition and voice cloning from transcribed speech. Overall, the results and products of this study can be used to diagnose dysarthria and improve communication and understanding for patients across different languages.

---

## 45. Multi-Modal Oral Cancer Detection Using Weighted Ensemble Convolutional Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.03878v1](http://arxiv.org/abs/2510.03878v1)

**作者:** Ajo Babu George, Sreehari J R Ajo Babu George, Sreehari J R Ajo Babu George, Sreehari J R

**发布时间:** 2025-10-04

### GPT解析

### 总结

该研究开发了一种多模态深度学习框架，整合临床、放射学和病理组织学图像，通过DenseNet-121卷积神经网络的加权集成，提高了口腔鳞状细胞癌的早期检测能力，总体准确率达到84.58%。

### 背景

口腔鳞状细胞癌(OSCC)的晚期诊断导致其全球死亡率高，根据世界卫生组织统计，超过50%的病例在晚期阶段才被检测出来，5年生存率低于50%。

### 目的

开发一个多模态深度学习框架，整合临床、放射学和病理组织学图像，使用加权集成DenseNet-121卷积神经网络，以提高OSCC的早期检测能力。

### 方法

进行回顾性研究，使用公开可用的数据集代表三种不同的医学成像模态，通过迁移学习训练DenseNet-121 CNN，应用增强和模态特定预处理，使用验证加权集成策略融合预测，并使用准确率、精确率、召回率和F1分数进行评估。

### 主要发现

放射学(100%)和病理组织学(95.12%)模态的验证准确率很高，临床图像表现较低(63.10%)由于视觉异质性，集成模型在55个样本的多模态验证数据集上总体准确率达到84.58%，显示出改进的诊断鲁棒性。

### 结论

多模态集成框架通过提供非侵入性、AI辅助的分诊工具弥合了当前诊断工作流程中的差距，有助于增强高风险病变的早期识别，支持临床医生决策制定，符合全球肿瘤学指南，以减少诊断延迟并改善患者结局。

### 翻译

目的：口腔鳞状细胞癌(OSCC)的晚期诊断对其全球高死亡率有显著贡献，根据世界卫生组织统计，超过50%的病例在晚期阶段才被检测出来，5年生存率低于50%。本研究旨在通过开发一个多模态深度学习框架来提高OSCC的早期检测，该框架整合了临床、放射学和病理组织学图像，使用DenseNet-121卷积神经网络(CNN)的加权集成。材料与方法：使用代表三种不同医学成像模态的公开可用数据集进行回顾性研究。每种模态特定的数据集用于通过迁移训练训练DenseNet-121 CNN。应用增强和模态特定的预处理以提高鲁棒性。使用验证加权集成策略融合预测。使用准确率、精确率、召回率和F1分数进行评估。结果：放射学(100%)和病理组织学(95.12%)模态的验证准确率很高，临床图像表现较低(63.10%)由于视觉异质性。集成模型在55个样本的多模态验证数据集上总体准确率为84.58%，显示出改进的诊断鲁棒性。结论：多模态集成框架通过提供非侵入性、AI辅助的分诊工具弥合了当前诊断工作流程中的差距，增强了高风险病变的早期识别。它支持临床医生决策制定，符合全球肿瘤学指南，以减少诊断延迟并改善患者结局。


### 论文摘要

Aims Late diagnosis of Oral Squamous Cell Carcinoma (OSCC) contributes significantly to its high global mortality rate, with over 50\% of cases detected at advanced stages and a 5-year survival rate below 50\% according to WHO statistics. This study aims to improve early detection of OSCC by developing a multimodal deep learning framework that integrates clinical, radiological, and histopathological images using a weighted ensemble of DenseNet-121 convolutional neural networks (CNNs). Material and Methods A retrospective study was conducted using publicly available datasets representing three distinct medical imaging modalities. Each modality-specific dataset was used to train a DenseNet-121 CNN via transfer learning. Augmentation and modality-specific preprocessing were applied to increase robustness. Predictions were fused using a validation-weighted ensemble strategy. Evaluation was performed using accuracy, precision, recall, F1-score. Results High validation accuracy was achieved for radiological (100\%) and histopathological (95.12\%) modalities, with clinical images performing lower (63.10\%) due to visual heterogeneity. The ensemble model demonstrated improved diagnostic robustness with an overall accuracy of 84.58\% on a multimodal validation dataset of 55 samples. Conclusion The multimodal ensemble framework bridges gaps in the current diagnostic workflow by offering a non-invasive, AI-assisted triage tool that enhances early identification of high-risk lesions. It supports clinicians in decision-making, aligning with global oncology guidelines to reduce diagnostic delays and improve patient outcomes.

---

## 46. Searching for the Most Human-like Emergent Language

**论文链接:** [http://arxiv.org/abs/2510.03467v1](http://arxiv.org/abs/2510.03467v1)

**作者:** Brendon Boldt, David Mortensen

**发布时间:** 2025-10-03

**备注:** Accepted for publication at the 2025 Conference on Empirical Methods  in Natural Language Processing; 19 pages, 12 figures

### GPT解析

### 总结

本文设计了一个基于信号博弈的涌现通信环境，通过超参数优化生成与人类语言高度相似的涌现语言，并研究了熵对涌现语言迁移学习性能的预测能力。

### 背景

研究涌现通信系统的特性及其与人类语言的相似性，探索如何生成更接近人类语言的涌现语言。

### 目的

设计能够产生与人类语言相似的涌现通信系统，并研究影响涌现语言质量的因素。

### 方法

使用信号博弈构建涌现通信环境，采用XferBench作为目标函数进行超参数优化，XferBench通过测量深度迁移学习的适用性来量化涌现语言与人类语言的相似性。

### 主要发现

熵对涌现语言的迁移学习性能具有预测能力，涌现通信系统具有熵最小化特性，某些超参数能产生更真实的涌现语言（即更好地迁移到人类语言）。

### 结论

通过超参数优化的信号博弈环境可以生成与人类语言相似的涌现语言，熵是评估涌现语言质量的有效指标。

### 翻译

本文设计了一个基于信号博弈的涌现通信环境，以生成在相似性方面达到最先进水平的涌现语言。这是通过使用XferBench作为目标函数进行超参数优化完成的。XferBench通过测量涌现语言对人类语言深度迁移学习的适用性，来量化其与人类语言的统计相似性。此外，我们还证明了熵对涌现语言迁移学习性能的预测能力，并验证了先前关于涌现通信系统熵最小化特性的研究结果。最后，我们报告了哪些超参数能产生更真实的涌现语言，即那些能更好地迁移到人类语言的涌现语言。


### 论文摘要

In this paper, we design a signalling game-based emergent communication environment to generate state-of-the-art emergent languages in terms of similarity to human language. This is done with hyperparameter optimization, using XferBench as the objective function. XferBench quantifies the statistical similarity of emergent language to human language by measuring its suitability for deep transfer learning to human language. Additionally, we demonstrate the predictive power of entropy on the transfer learning performance of emergent language as well as corroborate previous results on the entropy-minimization properties of emergent communication systems. Finally, we report generalizations regarding what hyperparameters produce more realistic emergent languages, that is, ones which transfer better to human language.

---

## 47. Bayesian Transfer Learning for High-Dimensional Linear Regression via Adaptive Shrinkage

**论文链接:** [http://arxiv.org/abs/2510.03449v1](http://arxiv.org/abs/2510.03449v1)

**作者:** Parsa Jamshidian, Donatello Telesca

**发布时间:** 2025-10-03

### GPT解析

### 总结

BLAST是一种贝叶斯多源迁移学习框架，用于高维线性回归。它结合全局-局部收缩先验和贝叶斯源选择，平衡信息共享和正则化，通过贝叶斯模型平均同时处理源选择和稀疏回归，使用Gibbs采样实现高效后验模拟。

### 背景

在高维线性回归中，有效利用多源数据面临挑战，传统正则化方法可能无法充分利用多源信息，现有迁移学习方法在预测性能和不确定性量化方面存在不足。

### 目的

开发一种能够有效整合多源信息的高维线性回归框架，提高目标变量的后验推断准确性，同时保持良好的预测性能和不确定性量化能力。

### 方法

提出BLAST框架，结合全局-局部收缩先验和贝叶斯源选择，通过贝叶斯模型平均处理源选择和稀疏回归，使用Gibbs采样进行后验模拟，并在模拟研究和TCGA肿瘤突变负担估计案例中验证效果。

### 主要发现

BLAST能提取最有用数据源同时排除导致负面迁移的偏差信息；相比仅基于目标数据的正则化方法，提供更准确的目标后验推断；相比当前最先进迁移学习方法，具有竞争性预测性能和更优的不确定性量化能力。

### 结论

BLAST是计算上实用且推断上直接的方法，能有效处理高维线性回归中的多源迁移学习问题，在多个方面表现出优越性。

### 翻译

我们介绍了BLAST，即具有自适应收缩的贝叶斯线性回归用于迁移学习，这是一种用于高维线性回归的贝叶斯多源迁移学习框架。所提出的分析框架利用全局-局部收缩先验和贝叶斯源选择来平衡信息共享和正则化。我们展示了贝叶斯源选择如何允许提取最有用的数据源，同时排除可能导致负面迁移的偏差信息。在这个框架中，通过贝叶斯模型平均，源选择和稀疏回归在预测和推断中被同时考虑。我们的模型结构通过Gibbs采样算法允许高效的后验模拟，能够对目标回归系数进行完整后验推断，使BLAST在计算上实用且推断上直接。我们的方法为目标提供了比仅基于目标数据的正则化方法更准确的后验推断，同时与当前最先进的迁移学习方法相比提供了竞争性的预测性能和更优的不确定性量化。我们通过广泛的模拟研究验证了其有效性，并说明了当应用于从基因表达估计肿瘤突变负担的案例研究（使用癌症基因组图谱(TCGA)的数据）时的分析特性。


### 论文摘要

We introduce BLAST, Bayesian Linear regression with Adaptive Shrinkage for Transfer, a Bayesian multi-source transfer learning framework for high-dimensional linear regression. The proposed analytical framework leverages global-local shrinkage priors together with Bayesian source selection to balance information sharing and regularization. We show how Bayesian source selection allows for the extraction of the most useful data sources, while discounting biasing information that may lead to negative transfer. In this framework, both source selection and sparse regression are jointly accounted for in prediction and inference via Bayesian model averaging. The structure of our model admits efficient posterior simulation via a Gibbs sampling algorithm allowing full posterior inference for the target regression coefficients, making BLAST both computationally practical and inferentially straightforward. Our method achieves more accurate posterior inference for the target than regularization approaches based on target data alone, while offering competitive predictive performance and superior uncertainty quantification compared to current state-of-the-art transfer learning methods. We validate its effectiveness through extensive simulation studies and illustrate its analytical properties when applied to a case study on the estimation of tumor mutational burden from gene expression, using data from The Cancer Genome Atlas (TCGA).

---

## 48. High Cycle S-N curve prediction for Al 7075-T6 alloy using Recurrent Neural Networks (RNNs)

**论文链接:** [http://arxiv.org/abs/2510.03355v1](http://arxiv.org/abs/2510.03355v1)

**作者:** Aryan Patel

**发布时间:** 2025-10-02

### GPT解析

### 总结

该研究开发了一种基于迁移学习的框架，使用长短期记忆网络(LSTMs)来预测铝合金的高周扭转S-N曲线，从而减少获取材料疲劳特性的成本和时间。

### 背景

铝是一种广泛使用的合金，容易发生疲劳失效。表征材料的疲劳性能非常耗时且成本高昂，特别是对于高周疲劳数据。

### 目的

开发一种基于迁移学习的框架，以减少获取材料疲劳特性的成本和时间。

### 方法

使用长短期记忆网络(LSTMs)构建了一个基于迁移学习的框架。首先基于纯轴向疲劳数据训练源LSTM模型（针对7075-T6铝合金），然后将该模型迁移用于预测高周扭转S-N曲线。

### 主要发现

该框架能够准确预测更高周次范围的铝扭转S-N曲线。

### 结论

该框架将有助于显著减少收集不同材料疲劳特性的成本，并帮助在更好的成本和时间约束下优先进行测试。

### 翻译

铝是一种广泛使用的合金，容易发生疲劳失效。表征材料的疲劳性能非常耗时且成本高昂，特别是对于高周疲劳数据。为帮助缓解这一问题，研究人员开发了一种基于迁移学习的框架，使用长短期记忆网络(LSTMs)。在该框架中，基于7075-T6铝合金的纯轴向疲劳数据训练源LSTM模型，然后将其迁移用于预测高周扭转S-N曲线。该框架能够准确预测更高周次范围的铝扭转S-N曲线。研究人员相信，该框架将有助于显著减少收集不同材料疲劳特性的成本，并帮助在更好的成本和时间约束下优先进行测试。


### 论文摘要

Aluminum is a widely used alloy, which is susceptible to fatigue failure. Characterizing fatigue performance for materials is extremely time and cost demanding, especially for high cycle data. To help mitigate this, a transfer learning based framework has been developed using Long short-term memory networks (LSTMs) in which a source LSTM model is trained based on pure axial fatigue data for Aluminum 7075-T6 alloy which is then transferred to predict high cycle torsional S-N curves. The framework was able to accurately predict Al torsional S-N curves for a much higher cycle range. It is the belief that this framework will help to drastically mitigate the cost of gathering fatigue characteristics for different materials and help prioritize tests with better cost and time constraints.

---

## 49. Machine Learning Workflows in Climate Modeling: Design Patterns and Insights from Case Studies

**论文链接:** [http://arxiv.org/abs/2510.03305v1](http://arxiv.org/abs/2510.03305v1)

**作者:** Tian Zheng, Subashree Venkatasubramanian, Shuolin Li, Amy Braverman, Xinyi Ke, Zhewen Hou, Peter Jin, Samarth Sanjay Agrawal

**发布时间:** 2025-09-30

**备注:** Supplement

### GPT解析

### 总结

该论文分析了机器学习在气候建模中的应用案例，综合了不同项目中的工作流设计模式，旨在提供一个确保科学机器学习严谨性的框架，并促进跨学科合作。

### 背景

机器学习在气候建模中的应用日益增加，包括系统模拟加速、数据驱动参数推断、预测和知识发现等领域，同时面临物理一致性、多尺度耦合、数据稀疏性、鲁棒泛化和与科学工作流集成等挑战。

### 目的

分析机器学习在气候建模中的应用案例研究，重点关注设计选择和工作流结构，综合各种项目中的工作流设计模式，提供确保科学机器学习严谨性的框架，降低数据科学与气候建模跨学科合作的障碍。

### 方法

通过分析一系列应用机器学习研究的案例，关注工作流设计模式而非技术细节，研究代理建模、机器学习参数化、概率编程、基于模拟的推断和物理信息迁移学习等多种工作流，分析这些工作流如何基于物理知识、模拟数据设计并集成观测数据。

### 主要发现

机器学习方法可应用于气候建模的不同方面，工作流设计需考虑物理知识、模拟数据和观测数据的集成，透明模型开发、关键评估、信息适应和可重复性对确保科学机器学习的严谨性至关重要。

### 结论

通过提供透明模型开发、关键评估、信息适应和可重复性的框架，可以确保科学机器学习的严谨性，同时有助于降低数据科学和气候建模交叉学科合作的障碍。

### 翻译

机器学习在气候建模的系统模拟加速、数据驱动参数推断、预测和知识发现等方面应用日益广泛，解决了物理一致性、多尺度耦合、数据稀疏性、鲁棒泛化和与科学工作流集成等挑战。本文分析了机器学习在气候建模应用研究的一系列案例研究，重点关注设计选择和工作流结构。我们旨在综合机器学习赋能的气候建模中不同项目的工作流设计模式：从代理建模、机器学习参数化、概率编程，到基于模拟的推断和物理信息迁移学习。我们解析了这些工作流如何基于物理知识、由模拟数据提供信息并设计用于集成观测数据。我们旨在通过更透明的模型开发、关键评估、信息适应和可重复性提供确保科学机器学习严谨性的框架，并为降低数据科学与气候建模交叉学科合作的障碍做出贡献。


### 论文摘要

Machine learning has been increasingly applied in climate modeling on system emulation acceleration, data-driven parameter inference, forecasting, and knowledge discovery, addressing challenges such as physical consistency, multi-scale coupling, data sparsity, robust generalization, and integration with scientific workflows. This paper analyzes a series of case studies from applied machine learning research in climate modeling, with a focus on design choices and workflow structure. Rather than reviewing technical details, we aim to synthesize workflow design patterns across diverse projects in ML-enabled climate modeling: from surrogate modeling, ML parameterization, probabilistic programming, to simulation-based inference, and physics-informed transfer learning. We unpack how these workflows are grounded in physical knowledge, informed by simulation data, and designed to integrate observations. We aim to offer a framework for ensuring rigor in scientific machine learning through more transparent model development, critical evaluation, informed adaptation, and reproducibility, and to contribute to lowering the barrier for interdisciplinary collaboration at the interface of data science and climate modeling.

---

## 50. Transformer Classification of Breast Lesions: The BreastDCEDL_AMBL Benchmark Dataset and 0.92 AUC Baseline

**论文链接:** [http://arxiv.org/abs/2509.26440v2](http://arxiv.org/abs/2509.26440v2)

**作者:** Naomi Fridman, Anat Goldstein

**发布时间:** 2025-09-30

### GPT解析

### 总结

本研究引入了一种基于Transformer的框架，用于动态对比增强MRI中乳腺病变的自动分类，通过SegFormer架构实现了高准确率，并创建了标准化数据集，为临床应用提供支持。

### 背景

乳腺磁共振成像(MRI)是癌症检测和治疗规划的重要工具，但其临床应用受到特异性差的限制，导致假阳性率高和不必要的活检。

### 目的

引入一个基于Transformer的框架，用于动态对比增强MRI中乳腺病变的自动分类，解决区分良性和恶性病变的挑战。

### 方法

实现了SegFormer架构，通过语义分割量化恶性像素分布，创建了包含88名患者和133个注释病变的BreastDCEDL_AMBL数据集，并整合了超过1200名患者的训练数据。

### 主要发现

模型在病变级别分类中达到0.92的AUC，在患者水平上达到100%的敏感性和67%的特异性，有潜力在不遗漏恶性肿瘤的情况下消除三分之一的不必要活检。

### 结论

公开发布的数据集、模型和评估协议为DCE-MRI病变分类提供了第一个标准化基准，促进了临床部署的方法学进展。

### 翻译

乳腺磁共振成像是癌症检测和治疗规划的关键工具，但其临床效用受到特异性差的阻碍，导致高假阳性和不必要的活检。本研究引入了一种基于Transformer的框架，用于动态对比增强MRI中乳腺病变的自动分类，解决了区分良性和恶性发现的挑战。我们实现了SegFormer架构，在病变级别分类中达到0.92的AUC，在患者水平上达到100%的敏感性和67%的特异性 - 有可能在不遗漏恶性肿瘤的情况下消除三分之一的不必要活检。该模型通过语义分割量化恶性像素分布，产生可解释的空间预测，支持临床决策。为了建立可复现的基准，我们通过将癌症影像档案库的AMBL集合转换为标准化深度学习数据集，创建了BreastDCEDL_AMBL，包含88名患者和133个注释病变(89个良性，44个恶性)。这一资源解决了关键的基础设施差距，因为现有的公共数据集缺乏良性病变注释，限制了良恶性分类研究。训练通过与BreastDCEDL数据集整合，纳入了超过1200名患者的扩展队列，验证了仅使用原发性肿瘤注释的迁移学习方法。公开发布数据集、模型和评估协议为DCE-MRI病变分类提供了第一个标准化基准，推动了临床部署的方法学进步。


### 论文摘要

Breast magnetic resonance imaging is a critical tool for cancer detection and treatment planning, but its clinical utility is hindered by poor specificity, leading to high false-positive rates and unnecessary biopsies. This study introduces a transformer-based framework for automated classification of breast lesions in dynamic contrast-enhanced MRI, addressing the challenge of distinguishing benign from malignant findings. We implemented a SegFormer architecture that achieved an AUC of 0.92 for lesion-level classification, with 100% sensitivity and 67% specificity at the patient level - potentially eliminating one-third of unnecessary biopsies without missing malignancies. The model quantifies malignant pixel distribution via semantic segmentation, producing interpretable spatial predictions that support clinical decision-making. To establish reproducible benchmarks, we curated BreastDCEDL_AMBL by transforming The Cancer Imaging Archive's AMBL collection into a standardized deep learning dataset with 88 patients and 133 annotated lesions (89 benign, 44 malignant). This resource addresses a key infrastructure gap, as existing public datasets lack benign lesion annotations, limiting benign-malignant classification research. Training incorporated an expanded cohort of over 1,200 patients through integration with BreastDCEDL datasets, validating transfer learning approaches despite primary tumor-only annotations. Public release of the dataset, models, and evaluation protocols provides the first standardized benchmark for DCE-MRI lesion classification, enabling methodological advancement toward clinical deployment.

---

## 51. ResCP: Reservoir Conformal Prediction for Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2510.05060v1](http://arxiv.org/abs/2510.05060v1)

**作者:** Roberto Neglia, Andrea Cini, Michael M. Bronstein, Filippo Maria Bianchi

**发布时间:** 2025-10-06

### GPT解析

### 总结

ResCP是一种基于储层计算的新型共形预测方法，通过动态重新加权一致性分数来处理时间序列数据，无需复杂的模型训练，适用于小样本和分布变化的情况。

### 背景

共形预测提供了构建可交换数据分布无关预测区间的强大框架，但现有扩展到序列数据的方法依赖于拟合复杂模型捕获时间依赖性，这些方法在小样本时可能失效，且在分布变化时需要昂贵的重新训练。

### 目的

克服现有方法的局限性，提出一种无需训练的共形预测方法，专门用于时间序列数据。

### 方法

提出储层共形预测(ResCP)，利用储层计算效率和表示学习能力动态重新加权一致性分数，通过计算储层状态间相似性分数来自适应地重新加权每步的观测残差，从而在建模误差分布时考虑局部时间动态性。

### 主要发现

在合理假设下，ResCP实现了渐近条件覆盖，并在多样化的预测任务中实证证明了其有效性。

### 结论

ResCP能够有效处理时间序列数据，无需复杂训练，适用于小样本和分布变化的情况，为共形预测在序列数据应用中提供了新思路。

### 翻译

共形预测为构建可交换数据的无分布预测区间提供了强大框架。现有将共形预测扩展到序列数据的方法依赖于拟合相对复杂的模型来捕获时间依赖性。然而，这些方法在样本量小时可能会失败，并且在底层分布变化时通常需要昂贵的重新训练。为克服这些局限性，我们提出了储层共形预测(ResCP)，一种用于时间序列的新型无需训练的共形预测方法。我们的方法利用储层计算的效率和表示学习能力来动态重新加权一致性分数。特别是，我们计算储层状态之间的相似性分数，并使用它们来自适应地重新加权每一步的观测残差。通过这种方法，ResCP使我们在建模误差分布时能够考虑局部时间动态性，同时不损害计算可扩展性。我们证明，在合理假设下，ResCP实现了渐近条件覆盖，并在多样化的预测任务中实证证明了其有效性。


### 论文摘要

Conformal prediction offers a powerful framework for building distribution-free prediction intervals for exchangeable data. Existing methods that extend conformal prediction to sequential data rely on fitting a relatively complex model to capture temporal dependencies. However, these methods can fail if the sample size is small and often require expensive retraining when the underlying data distribution changes. To overcome these limitations, we propose Reservoir Conformal Prediction (ResCP), a novel training-free conformal prediction method for time series. Our approach leverages the efficiency and representation learning capabilities of reservoir computing to dynamically reweight conformity scores. In particular, we compute similarity scores among reservoir states and use them to adaptively reweight the observed residuals at each step. With this approach, ResCP enables us to account for local temporal dynamics when modeling the error distribution without compromising computational scalability. We prove that, under reasonable assumptions, ResCP achieves asymptotic conditional coverage, and we empirically demonstrate its effectiveness across diverse forecasting tasks.

---

## 52. Federated Self-Supervised Learning for Automatic Modulation Classification under Non-IID and Class-Imbalanced Data

**论文链接:** [http://arxiv.org/abs/2510.04927v1](http://arxiv.org/abs/2510.04927v1)

**作者:** Usman Akram, Yiyue Chen, Haris Vikalo

**发布时间:** 2025-10-06

### GPT解析

### 总结

这项研究提出了一种名为FedSSL-AMC的联邦自监督学习方法，用于自动调制分类(AMC)，解决了集中式训练的隐私问题和通信开销问题，同时提高了模型在信道变化和非独立同分布数据下的鲁棒性。

### 背景

在集中式数据上训练自动调制分类模型会引发隐私问题，产生通信开销，并且往往无法对信道变化保持鲁棒性。联邦学习虽然避免了集中聚合，通过在分布式客户端上训练，但仍对类别不平衡、非独立同分布客户端分布和有限的标记样本敏感。

### 目的

开发一种能够解决隐私问题、通信开销，同时对信道变化和非独立同分布数据具有鲁棒性的自动调制分类方法。

### 方法

提出FedSSL-AMC方法，在各个客户端的无标记I/Q序列上训练具有三元组损失自监督的因果时间膨胀CNN，然后在小型标记集上为每个客户端训练SVM。建立了联邦表示学习过程的收敛性，以及对下游分类器在特征噪声下的可分性保证。

### 主要发现

在合成数据和实际空中数据集上的实验表明，在异构信噪比、载波频率偏移和非独立同分布标签分区条件下，与监督联邦学习基线相比，该方法取得了持续的性能提升。

### 结论

FedSSL-AMC方法通过结合联邦学习和自监督学习，有效解决了AMC中的隐私、通信开销和鲁棒性问题，在各种信道条件下表现优于传统方法。

### 翻译

在集中聚合数据上训练自动调制分类模型会引发隐私问题，产生通信开销，并且往往无法对信道变化保持鲁棒性。联邦学习通过在分布式客户端上训练避免了集中聚合，但仍对类别不平衡、非独立同分布客户端分布和有限的标记样本敏感。我们提出了FedSSL-AMC，它在各个客户端的无标记I/Q序列上训练具有三元组损失自监督的因果时间膨胀CNN，然后在小型标记集上为每个客户端训练SVM。我们建立了联邦表示学习过程的收敛性，以及对下游分类器在特征噪声下的可分性保证。在合成数据和实际空中数据集上的实验表明，在异构信噪比、载波频率偏移和非独立同分布标签分区条件下，与监督联邦学习基线相比，该方法取得了持续的性能提升。


### 论文摘要

Training automatic modulation classification (AMC) models on centrally aggregated data raises privacy concerns, incurs communication overhead, and often fails to confer robustness to channel shifts. Federated learning (FL) avoids central aggregation by training on distributed clients but remains sensitive to class imbalance, non-IID client distributions, and limited labeled samples. We propose FedSSL-AMC, which trains a causal, time-dilated CNN with triplet-loss self-supervision on unlabeled I/Q sequences across clients, followed by per-client SVMs on small labeled sets. We establish convergence of the federated representation learning procedure and a separability guarantee for the downstream classifier under feature noise. Experiments on synthetic and over-the-air datasets show consistent gains over supervised FL baselines under heterogeneous SNR, carrier-frequency offsets, and non-IID label partitions.

---

## 53. BenthiCat: An opti-acoustic dataset for advancing benthic classification and habitat mapping

**论文链接:** [http://arxiv.org/abs/2510.04876v1](http://arxiv.org/abs/2510.04876v1)

**作者:** Hayat Rajani, Valerio Franchi, Borja Martinez-Clavel Valles, Raimon Ramos, Rafael Garcia, Nuno Gracias

**发布时间:** 2025-10-06

**备注:** Article under review by IJRR

### GPT解析

### 总结

该研究介绍了一个全面的多模态数据集，用于海洋底栖生境制图，包含约一百万个侧扫声纳瓦片及其相关数据，旨在为水下生境分类提供标准化基准。

### 背景

海洋底栖生境制图对于理解海洋生态系统、指导保护工作和支持可持续资源管理至关重要。然而，大型标注数据集的稀缺限制了该领域机器学习模型的发展和基准测试。

### 目的

创建一个全面的多模态数据集，建立水下生境制图的标准化基准，促进自主海底分类和多传感器集成技术的发展。

### 方法

收集了约一百万个来自加泰罗尼亚海岸的侧扫声纳瓦片，配以水深图和自主水下航行器获取的光学图像。约36,000个SSS瓦片已手动标注分割掩码。研究将光学图像与SSS瓦片在空间上关联，促进自监督、跨模态表示学习，并提供开源的预处理和标注工具。

### 主要发现

成功构建了一个大型多模态海洋底栖生境数据集，包含多种传感器数据，并提供了标注工具和数据关联方法，解决了多传感器数据融合的挑战。

### 结论

这一资源为水下生境制图提供了标准化基准，有望促进自主海底分类和多传感器集成技术的进步，同时通过提供开源工具和数据增强了研究可访问性。

### 翻译

海底生境制图对于理解海洋生态系统、指导保护工作和支持可持续资源管理至关重要。然而，大型标注数据集的稀缺限制了该领域机器学习模型的发展和基准测试。本文介绍了一个全面的多模态数据集，包含约一百万个沿加泰罗尼亚海岸收集的侧扫声纳瓦片，辅以水深图和一组使用自主水下航行器进行目标调查的配准光学图像。约36,000个SSS瓦片已手动标注分割掩码，用于监督分类模型的微调。所有原始传感器数据以及镶嵌图像也已发布，以支持进一步探索和算法开发。为解决AUV多传感器数据融合的挑战，我们在空间上将光学图像与相应的SSS瓦片关联，促进自监督、跨模态表示学习。提供了开源的预处理和标注工具，以提高可访问性和鼓励研究。这一资源旨在为水下生境制图建立标准化基准，促进自主海底分类和多传感器集成的发展。


### 论文摘要

Benthic habitat mapping is fundamental for understanding marine ecosystems, guiding conservation efforts, and supporting sustainable resource management. Yet, the scarcity of large, annotated datasets limits the development and benchmarking of machine learning models in this domain. This paper introduces a thorough multi-modal dataset, comprising about a million side-scan sonar (SSS) tiles collected along the coast of Catalonia (Spain), complemented by bathymetric maps and a set of co-registered optical images from targeted surveys using an autonomous underwater vehicle (AUV). Approximately \num{36000} of the SSS tiles have been manually annotated with segmentation masks to enable supervised fine-tuning of classification models. All the raw sensor data, together with mosaics, are also released to support further exploration and algorithm development. To address challenges in multi-sensor data fusion for AUVs, we spatially associate optical images with corresponding SSS tiles, facilitating self-supervised, cross-modal representation learning. Accompanying open-source preprocessing and annotation tools are provided to enhance accessibility and encourage research. This resource aims to establish a standardized benchmark for underwater habitat mapping, promoting advancements in autonomous seafloor classification and multi-sensor integration.

---

## 54. Compressed Concatenation of Small Embedding Models

**论文链接:** [http://arxiv.org/abs/2510.04626v1](http://arxiv.org/abs/2510.04626v1)

**作者:** Mohamed Ayoub Ben Ayad, Michael Dinzinger, Kanishka Ghosh Dastidar, Jelena Mitrovic, Michael Granitzer

**发布时间:** 2025-10-06

**DOI:** 10.1145/3746252.3760831

### GPT解析

### 总结

本文提出了一种通过连接多个小型嵌入模型并使用轻量级统一解码器来降低维度的方法，在保持高性能的同时实现模型压缩，适用于资源受限环境。

### 背景

嵌入模型在密集检索、语义搜索和推荐系统中起核心作用，但其体积大，难以部署在资源受限环境如浏览器或边缘设备。

### 目的

缩小小型嵌入模型与大型模型之间的性能差距，实现高效且高性能的模型部署。

### 方法

通过连接多个小型模型的嵌入向量，并引入一个使用套娃表示学习(MRL)损失训练的轻量级统一解码器，将高维联合表示映射到低维空间，无需微调基础模型。

### 主要发现

连接多个小型模型可超越单个大型基线模型；虽然连接更多模型收益递减，但解码器表示在压缩和量化下的鲁棒性提高；在MTEB检索任务上，连接-编码-量化流程以48倍压缩因子恢复了89%的原始性能。

### 结论

所提出的连接-编码-量化管道能够在资源受限环境中实现高效部署，同时保持高性能，为实际应用提供了可行的解决方案。

### 翻译

嵌入模型是密集检索、语义搜索和推荐系统的核心，但其体积通常使得在资源受限的环境中（如浏览器或边缘设备）部署变得不切实际。虽然较小的嵌入模型具有实际优势，但通常比大型模型性能较差。为了缩小这一差距，我们证明连接多个小型模型的原始嵌入向量可以在标准检索基准测试中超越单个大型基线模型。为了克服简单连接导致的高维度问题，我们引入了一个使用套娃表示学习(MRL)损失训练的轻量级统一解码器。这个解码器将高维联合表示映射到低维空间，在不需要微调基础模型的情况下保留了大部分原始性能。我们还表明，虽然连接更多基础模型会带来收益递减，但解码器表示在压缩和量化下的鲁棒性会提高。我们的实验显示，在MTEB检索任务的一个子集上，当将连接-编码-量化流程应用于四个小型嵌入模型时，该流程以48倍的压缩因子恢复了89%的原始性能。


### 论文摘要

Embedding models are central to dense retrieval, semantic search, and recommendation systems, but their size often makes them impractical to deploy in resource-constrained environments such as browsers or edge devices. While smaller embedding models offer practical advantages, they typically underperform compared to their larger counterparts. To bridge this gap, we demonstrate that concatenating the raw embedding vectors of multiple small models can outperform a single larger baseline on standard retrieval benchmarks. To overcome the resulting high dimensionality of naive concatenation, we introduce a lightweight unified decoder trained with a Matryoshka Representation Learning (MRL) loss. This decoder maps the high-dimensional joint representation to a low-dimensional space, preserving most of the original performance without fine-tuning the base models. We also show that while concatenating more base models yields diminishing gains, the robustness of the decoder's representation under compression and quantization improves. Our experiments show that, on a subset of MTEB retrieval tasks, our concat-encode-quantize pipeline recovers 89\% of the original performance with a 48x compression factor when the pipeline is applied to a concatenation of four small embedding models.

---

## 55. Conditional Representation Learning for Customized Tasks

**论文链接:** [http://arxiv.org/abs/2510.04564v1](http://arxiv.org/abs/2510.04564v1)

**作者:** Honglin Liu, Chao Sun, Peng Hu, Yunfan Li, Xi Peng

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出条件表示学习(CRL)方法，能够根据用户指定标准提取定制化表示，无需高成本监督微调。该方法通过大语言模型生成描述性文本构建语义基，再利用视觉-语言模型将图像表示投影到条件特征空间，从而更好地捕获特定语义。

### 背景

传统表示学习方法学习通用表示，主要捕获主导语义，可能与特定下游任务不完全一致。例如在动物栖息地分析中，研究人员关注场景相关特征，而通用嵌入强调类别语义，导致次优结果。

### 目的

提出条件表示学习(CRL)，旨在提取符合任意用户指定标准的定制化表示。

### 方法

基于空间语义由其基决定的原理，CRL首先使用大语言模型生成描述性文本来构建语义基，然后利用视觉-语言模型将图像表示投影到条件特征空间。

### 主要发现

条件表示能够更好地捕获特定标准的语义，可用于多种定制化任务。

### 结论

在分类和检索任务上的大量实验证明了CRL的优越性和通用性。代码已公开可用。

### 翻译

传统的表示学习方法学习的是通用表示，主要捕获主导语义，这并不总是与定制的下游任务一致。例如，在动物栖息地分析中，研究人员优先考虑场景相关特征，而通用嵌入强调类别语义，导致次优结果。作为解决方案，现有方法采用监督微调，但这会产生高计算和标注成本。在本文中，我们提出条件表示学习(CRL)，旨在提取符合任意用户指定标准的定制化表示。具体来说，我们揭示空间的语义由其基决定，从而使得一组描述性词语可以近似定制特征空间的基。基于这一见解，给定用户指定的标准，CRL首先使用大语言模型(LLM)生成描述性文本来构建语义基，然后利用视觉-语言模型(VLM)将图像表示投影到这个条件特征空间。条件表示能够更好地捕获特定标准的语义，可用于多种定制化任务。在分类和检索任务上的大量实验证明了所提出的CRL的优越性和通用性。代码可在 https://github.com/XLearning-SCU/2025-NeurIPS-CRL 获取。


### 论文摘要

Conventional representation learning methods learn a universal representation that primarily captures dominant semantics, which may not always align with customized downstream tasks. For instance, in animal habitat analysis, researchers prioritize scene-related features, whereas universal embeddings emphasize categorical semantics, leading to suboptimal results. As a solution, existing approaches resort to supervised fine-tuning, which however incurs high computational and annotation costs. In this paper, we propose Conditional Representation Learning (CRL), aiming to extract representations tailored to arbitrary user-specified criteria. Specifically, we reveal that the semantics of a space are determined by its basis, thereby enabling a set of descriptive words to approximate the basis for a customized feature space. Building upon this insight, given a user-specified criterion, CRL first employs a large language model (LLM) to generate descriptive texts to construct the semantic basis, then projects the image representation into this conditional feature space leveraging a vision-language model (VLM). The conditional representation better captures semantics for the specific criterion, which could be utilized for multiple customized tasks. Extensive experiments on classification and retrieval tasks demonstrate the superiority and generality of the proposed CRL. The code is available at https://github.com/XLearning-SCU/2025-NeurIPS-CRL.

---

## 56. GRACE: Generative Representation Learning via Contrastive Policy Optimization

**论文链接:** [http://arxiv.org/abs/2510.04506v1](http://arxiv.org/abs/2510.04506v1)

**作者:** Jiashuo Sun, Shixuan Liu, Zhaochen Su, Xianrui Zhong, Pengcheng Jiang, Bowen Jin, Peiran Li, Weijia Shi, Jiawei Han

**发布时间:** 2025-10-06

**备注:** 23 pages, 7 figures, 7 tables

### GPT解析

### 总结

GRACE是一种新型框架，通过对比策略优化实现生成表征学习，将大型语言模型转变为可解释的代理，生成推理理由并基于这些理由进行对比学习，在MTEB基准测试中表现优异，显著提升了监督和无监督设置下的性能。

### 背景

当前训练大型语言模型作为文本编码器的主流方法依赖于对比损失，这些方法将模型视为黑盒函数，放弃了其生成和推理能力，只使用静态嵌入。

### 目的

引入GRACE（通过对比策略优化的生成表征学习）框架，重新构想对比信号不是要最小化的损失，而是引导生成策略的奖励，以保留大型语言模型的生成和推理能力。

### 方法

在GRACE框架中，大型语言模型作为策略生成明确、可解释的推理理由，这些推理理由通过平均池编码为高质量嵌入，使用策略梯度优化，训练模型采用多组件奖励函数，该奖励函数最大化查询正对之间的相似性，同时最小化与负对的相似性。

### 主要发现

GRACE将大型语言模型从不透明的编码器转变为可解释的代理，其推理过程透明且可检查；在MTEB基准测试中，GRACE实现了广泛的跨类别提升，在监督设置下，四个骨干模型的平均总体得分比基线模型提高11.5%，无监督变体增加6.9%，同时保留了通用能力。

### 结论

将对比目标视为推理理由上的奖励，统一了表征学习和生成，产生了更强的嵌入和透明的推理理由。

### 翻译

目前训练大型语言模型作为文本编码器的方法依赖于对比损失，这些方法将模型视为黑盒函数，为了静态嵌入而放弃了其生成和推理能力。我们引入了GRACE（通过对比策略优化的生成表征学习），这是一个重新构想对比信号不是作为最小化的损失，而是作为引导生成策略的奖励的新型框架。在GRACE中，大型语言模型作为策略生成明确、人类可解释的推理理由——对其语义理解的结构化自然语言解释。然后这些推理理由通过平均池编码为高质量嵌入。使用策略梯度优化，我们训练模型采用多组件奖励函数，该函数最大化查询正对之间的相似性，同时最小化与负对的相似性。这使得大型语言模型从不透明的编码器转变为可解释的代理，其推理过程透明且可检查。在MTEB基准测试中，GRACE实现了广泛的跨类别提升：在四个骨干模型上平均，监督设置下比基线模型总体得分提高11.5%，无监督变体增加6.9%，同时保留了通用能力。这项工作将对比目标视为推理理由上的奖励，统一了表征学习和生成，以产生更强的嵌入和透明的推理理由。模型、数据和代码可在https://github.com/GasolSun36/GRACE获取。


### 论文摘要

Prevailing methods for training Large Language Models (LLMs) as text encoders rely on contrastive losses that treat the model as a black box function, discarding its generative and reasoning capabilities in favor of static embeddings. We introduce GRACE (Generative Representation Learning via Contrastive Policy Optimization), a novel framework that reimagines contrastive signals not as losses to be minimized, but as rewards that guide a generative policy. In GRACE, the LLM acts as a policy that produces explicit, human-interpretable rationales--structured natural language explanations of its semantic understanding. These rationales are then encoded into high-quality embeddings via mean pooling. Using policy gradient optimization, we train the model with a multi-component reward function that maximizes similarity between query positive pairs and minimizes similarity with negatives. This transforms the LLM from an opaque encoder into an interpretable agent whose reasoning process is transparent and inspectable. On MTEB benchmark, GRACE yields broad cross category gains: averaged over four backbones, the supervised setting improves overall score by 11.5% over base models, and the unsupervised variant adds 6.9%, while preserving general capabilities. This work treats contrastive objectives as rewards over rationales, unifying representation learning with generation to produce stronger embeddings and transparent rationales. The model, data and code are available at https://github.com/GasolSun36/GRACE.

---

## 57. Diffusion-Assisted Distillation for Self-Supervised Graph Representation Learning with MLPs

**论文链接:** [http://arxiv.org/abs/2510.04241v1](http://arxiv.org/abs/2510.04241v1)

**作者:** Seong Jin Ahn, Myoung-Ho Kim

**发布时间:** 2025-10-05

**DOI:** 10.1109/TAI.2025.3598791

### GPT解析

### 总结

本文提出了一种名为DAD-SGM的新方法，通过去噪扩散模型作为教师辅助，将自监督图神经网络的知识有效蒸馏到轻量级多层感知机中，提升了MLPs在图表示学习中的泛化能力和鲁棒性。

### 背景

在大规模应用中，人们倾向于用轻量级的多层感知机替代图神经网络，但在自监督图表示学习中，这种知识蒸馏更具挑战性，因为自监督学习的性能与模型的归纳偏置关系更密切。

### 目的

设计一种新的蒸馏方法，以弥合自监督图表示学习中图神经网络和多层感知机之间的巨大能力差距。

### 方法

提出DAD-SGM(Diffusion-Assisted Distillation for Self-supervised Graph representation learning with MLPs)方法，采用去噪扩散模型作为教师辅助，更好地将教师GNN的知识蒸馏到学生MLP中。

### 主要发现

大量实验表明，与最先进的GNN到MLP蒸馏方法相比，DAD-SGM能有效蒸馏自监督GNNs的知识，并显著增强MLPs在自监督图表示学习中的泛化能力和鲁棒性。

### 结论

DAD-SGM是一种有效的自监督GNN知识蒸馏方法，相关实现代码已在GitHub平台公开。

### 翻译

对于大规模应用，人们越来越有兴趣通过知识蒸馏用轻量级的多层感知机替代图神经网络。然而，在自监督图表示学习中将图神经网络蒸馏到多层感知机更具挑战性。这是因为自监督学习的性能比监督学习更依赖于模型的归纳偏置。这促使我们设计一种新的蒸馏方法，以弥合自监督图表示学习中图神经网络和多层感知机之间的巨大能力差距。在本文中，我们提出了DAD-SGM(用于多层感知机自监督图表示学习的扩散辅助蒸馏)。该方法采用去噪扩散模型作为教师辅助，以更好地将教师图神经网络的知识蒸馏到学生多层感知机中。这种方法增强了多层感知机在自监督图表示学习中的泛化能力和鲁棒性。大量实验表明，与最先进的图神经网络到多层感知机蒸馏方法相比，DAD-SGM能有效蒸馏自监督图神经网络的知识。我们的实现可在https://github.com/SeongJinAhn/DAD-SGM获取。


### 论文摘要

For large-scale applications, there is growing interest in replacing Graph Neural Networks (GNNs) with lightweight Multi-Layer Perceptrons (MLPs) via knowledge distillation. However, distilling GNNs for self-supervised graph representation learning into MLPs is more challenging. This is because the performance of self-supervised learning is more related to the model's inductive bias than supervised learning. This motivates us to design a new distillation method to bridge a huge capacity gap between GNNs and MLPs in self-supervised graph representation learning. In this paper, we propose \textbf{D}iffusion-\textbf{A}ssisted \textbf{D}istillation for \textbf{S}elf-supervised \textbf{G}raph representation learning with \textbf{M}LPs (DAD-SGM). The proposed method employs a denoising diffusion model as a teacher assistant to better distill the knowledge from the teacher GNN into the student MLP. This approach enhances the generalizability and robustness of MLPs in self-supervised graph representation learning. Extensive experiments demonstrate that DAD-SGM effectively distills the knowledge of self-supervised GNNs compared to state-of-the-art GNN-to-MLP distillation methods. Our implementation is available at https://github.com/SeongJinAhn/DAD-SGM.

---

## 58. MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition

**论文链接:** [http://arxiv.org/abs/2510.04136v1](http://arxiv.org/abs/2510.04136v1)

**作者:** Umberto Cappellazzo, Minsu Kim, Pingchuan Ma, Honglie Chen, Xubo Liu, Stavros Petridis, Maja Pantic

**发布时间:** 2025-10-05

**备注:** NeurIPS 2025

### GPT解析

### 总结

该研究提出了一种名为MoME（Mixture of Matryoshka Experts）的新型框架，将稀疏混合专家（MoE）集成到基于Matryoshka表示学习（MRL）的大语言模型中，用于音频-视觉语音识别。这种方法统一了MRL的适应性和MoE的效率，为资源感知的语音识别提供了可扩展且可解释的解决方案。

### 背景

大语言模型在音频-视觉语音识别方面显示出强大潜力，但其高计算需求和令牌粒度敏感性限制了在资源受限环境中的实用性。现有的令牌压缩方法需要在预先固定压缩率并产生固定长度的输出，无法灵活平衡信息密度和效率。Matryoshka表示学习虽能解决此问题，但当前方法在训练时独立处理每个尺度，限制了跨尺度泛化能力、高压缩下的鲁棒性和可解释性。

### 目的

克服现有基于MRL方法的局限性，提高跨尺度泛化能力、高压缩下的鲁棒性和可解释性，同时保持资源效率。

### 方法

提出MoME框架，将稀疏混合专家集成到基于MRL的LLMs中。MoME通过冻结LLM并添加top-k路由和共享专家，允许跨尺度和模态的动态容量分配。共享路由器促进跨粒度的一致专家激活，使压缩序列能够受益于在较低压缩率下学习到的表示。

### 主要发现

在LRS2和LRS3上的实验表明，MoME在AVSR、ASR和VSR任务上实现了最先进的性能，同时需要显著更少的参数，并在噪声条件下保持鲁棒性。

### 结论

MoME统一了MRL的适应性和MoE的效率，为资源感知的语音识别提供了可扩展且可解释的解决方案。

### 翻译

大语言模型最近在音频-视觉语音识别方面显示出强大的潜力，但其高计算需求和令牌粒度敏感性限制了它们在资源受限环境中的实用性。令牌压缩方法可以降低推理成本，但它们需要预先固定压缩率并产生单一固定长度的输出，无法在推理时灵活平衡信息密度和效率。Matryoshka表示学习通过使单个模型能够在多个令牌粒度上运行，允许动态调整压缩率来解决这个问题。然而，当前基于MRL的方法在训练时独立处理每个尺度，限制了跨尺度泛化能力、高压缩下的鲁棒性和可解释性。为了克服这些局限性，我们提出了MoME，一种将稀疏混合专家集成到基于MRL的LLMs中用于AVSR的新型框架。MoME通过冻结LLM并添加top-k路由和共享专家，允许跨尺度和模态的动态容量分配。共享路由器促进跨粒度的一致专家激活，使压缩序列能够受益于在较低压缩率下学习到的表示。在LRS2和LRS3上的实验表明，MoME在AVSR、ASR和VSR任务上实现了最先进的性能，同时需要显著更少的参数，并在噪声条件下保持鲁棒性。MoME统一了MRL的适应性和MoE的效率，为资源感知的语音识别提供了可扩展且可解释的解决方案。


### 论文摘要

Large language models (LLMs) have recently shown strong potential in audio-visual speech recognition (AVSR), but their high computational demands and sensitivity to token granularity limit their practicality in resource-constrained settings. Token compression methods can reduce inference cost, but they require fixing a compression rate in advance and produce a single fixed-length output, offering no flexibility to balance information density and efficiency at inference time. Matryoshka representation learning (MRL) addresses this by enabling a single model to operate across multiple token granularities, allowing compression rates to be adjusted dynamically. However, current MRL-based methods treat each scale independently during training, limiting cross-scale generalization, robustness at high compression, and interpretability. To overcome these limitations, we propose MoME (Mixture of Matryoshka Experts), a novel framework that integrates sparse Mixture-of-Experts (MoE) into MRL-based LLMs for AVSR. MoME augments a frozen LLM with top-k routed and shared experts, allowing dynamic capacity allocation across scales and modalities. A shared router promotes consistent expert activation across granularities, enabling compressed sequences to benefit from representations learned at lower compression. Experiments on LRS2 and LRS3 demonstrate that MoME achieves state-of-the-art performance across AVSR, ASR, and VSR tasks, while requiring significantly fewer parameters and maintaining robustness under noise. MoME unifies the adaptability of MRL with the efficiency of MoE, offering a scalable and interpretable solution for resource-aware speech recognition.

---

## 59. Attending on Multilevel Structure of Proteins enables Accurate Prediction of Cold-Start Drug-Target Interactions

**论文链接:** [http://arxiv.org/abs/2510.04126v1](http://arxiv.org/abs/2510.04126v1)

**作者:** Ziying Zhang, Yaqing Wang, Yuxuan Sun, Min Ye, Quanming Yao

**发布时间:** 2025-10-05

### GPT解析

### 总结

本文提出了一种名为ColdDTI的框架，用于冷启动药物-靶点相互作用(DTI)预测，该框架关注蛋白质的多层次结构，通过分层注意力机制挖掘蛋白质多层次结构与药物结构的相互作用，并融合不同级别的结构表示进行预测。

### 背景

冷启动药物-靶点相互作用预测关注新型药物和蛋白质之间的相互作用。现有方法通常学习药物和蛋白质结构之间的可转移模式，但蛋白质组学研究表明蛋白质具有多层次结构(从初级到四级)，这些结构都影响DTI。然而，现有工作通常仅用初级结构表示蛋白质，限制了它们捕获涉及更高级结构相互作用的能力。

### 目的

提出一个关注蛋白质多层次结构的框架用于冷启动DTI预测，以克服现有方法仅使用蛋白质初级结构的局限性。

### 方法

提出ColdDTI框架，采用分层注意力机制挖掘多层次蛋白质结构与药物结构在局部和全局粒度上的相互作用，然后利用这些相互作用融合不同级别的结构表示进行最终预测。

### 主要发现

该设计捕获了生物学上可转移的先验知识，避免了对表示学习的过度依赖导致的过拟合风险。在基准数据集上的实验表明，ColdDTI在冷启动设置中始终优于以前的方法。

### 结论

ColdDTI框架通过关注蛋白质多层次结构并采用分层注意力机制，有效提高了冷启动DTI预测的性能，为药物发现提供了新的思路。

### 翻译

冷启动药物-靶点相互作用(DTI)预测关注新型药物和蛋白质之间的相互作用。以前的方法通常学习药物和蛋白质结构之间的可转移相互作用模式来解决它。然而，蛋白质组学的见解表明蛋白质具有多层次结构，并且它们都影响DTI。现有工作通常仅用初级结构表示蛋白质，限制了它们捕获涉及更高级结构相互作用的能力。受这一见解的启发，我们提出了ColdDTI，一个关注蛋白质多层次结构用于冷启动DTI预测的框架。我们采用分层注意力机制挖掘多层次蛋白质结构(从初级到四级)与药物结构在局部和全局粒度上的相互作用。然后，我们利用挖掘到的相互作用融合不同级别的结构表示进行最终预测。我们的设计捕获了生物学上可转移的先验知识，避免了因过度依赖表示学习导致的过拟合风险。在基准数据集上的实验表明，ColdDTI在冷启动设置中始终优于以前的方法。


### 论文摘要

Cold-start drug-target interaction (DTI) prediction focuses on interaction between novel drugs and proteins. Previous methods typically learn transferable interaction patterns between structures of drug and proteins to tackle it. However, insight from proteomics suggest that protein have multi-level structures and they all influence the DTI. Existing works usually represent protein with only primary structures, limiting their ability to capture interactions involving higher-level structures. Inspired by this insight, we propose ColdDTI, a framework attending on protein multi-level structure for cold-start DTI prediction. We employ hierarchical attention mechanism to mine interaction between multi-level protein structures (from primary to quaternary) and drug structures at both local and global granularities. Then, we leverage mined interactions to fuse structure representations of different levels for final prediction. Our design captures biologically transferable priors, avoiding the risk of overfitting caused by excessive reliance on representation learning. Experiments on benchmark datasets demonstrate that ColdDTI consistently outperforms previous methods in cold-start settings.

---

## 60. A Hybrid Co-Finetuning Approach for Visual Bug Detection in Video Games

**论文链接:** [http://arxiv.org/abs/2510.03591v1](http://arxiv.org/abs/2510.03591v1)

**作者:** Faliu Yi, Sherif Abdelfattah, Wei Huang, Adrian Brown

**发布时间:** 2025-10-04

**备注:** Accepted at the 21st AAAI Conference on Artificial Intelligence and  Interactive Digital Entertainment (AIIDE 2025)

### GPT解析

### 总结

本文提出了一种混合的Co-FineTuning (CFT)方法，用于解决视频游戏视觉缺陷检测中标记数据稀缺的问题，通过整合标记和非标记数据提高检测效率。

### 背景

手动识别视频游戏视觉缺陷是一个资源密集且成本高昂的过程，需要专业知识。监督式视觉缺陷检测模型虽有前景，但依赖于大量标记数据，而这类缺陷很少出现，构成重大挑战。

### 目的

克服监督式视觉缺陷检测模型对大量标记数据的依赖，提出一种能够有效整合标记和非标记数据的方法，减少对特定目标游戏标记样本的依赖。

### 方法

提出混合Co-FineTuning (CFT)方法，整合来自目标游戏和不同领域游戏的标记样本，同时利用未标记数据增强特征表示学习，最大化所有可用数据的效用。

### 主要发现

开发的框架展示了增强的可扩展性和适应性，能在各种游戏标题中实现高效的视觉缺陷检测；实验结果表明该方法具有鲁棒性，与传统基线相比在多个游戏环境中表现优越；即使仅使用目标游戏50%的标记数据进行训练，CFT也能保持有竞争力的性能。

### 结论

CFT方法有效解决了视觉缺陷检测中标记数据稀缺的问题，具有良好的可扩展性和适应性，适用于多种游戏环境。

### 翻译

手动识别视频游戏中的视觉缺陷是一个资源密集且成本高昂的过程，通常需要专业的领域知识。虽然监督式视觉缺陷检测模型提供了有前景的解决方案，但它们对大量标记数据的依赖由于此类缺陷的罕见性而构成了重大挑战。为克服这一限制，我们提出了一种混合的Co-FineTuning (CFT)方法，能够有效整合标记和非标记数据。我们的方法利用来自目标游戏和不同领域游戏的标记样本，同时结合未标记数据以增强特征表示学习。这种策略最大化了所有可用数据的效用，显著减少了对特定目标游戏标记样本的依赖。开发的框架展示了增强的可扩展性和适应性，促进了各种游戏标题中高效的视觉缺陷检测。我们的实验结果表明，所提出的方法对游戏视觉缺陷检测具有鲁棒性，在多个游戏环境中表现出优于传统基线的性能。此外，即使仅使用目标游戏50%的标记数据进行训练，CFT仍能保持有竞争力的性能。


### 论文摘要

Manual identification of visual bugs in video games is a resource-intensive and costly process, often demanding specialized domain knowledge. While supervised visual bug detection models offer a promising solution, their reliance on extensive labeled datasets presents a significant challenge due to the infrequent occurrence of such bugs. To overcome this limitation, we propose a hybrid Co-FineTuning (CFT) method that effectively integrates both labeled and unlabeled data. Our approach leverages labeled samples from the target game and diverse co-domain games, additionally incorporating unlabeled data to enhance feature representation learning. This strategy maximizes the utility of all available data, substantially reducing the dependency on labeled examples from the specific target game. The developed framework demonstrates enhanced scalability and adaptability, facilitating efficient visual bug detection across various game titles. Our experimental results show the robustness of the proposed method for game visual bug detection, exhibiting superior performance compared to conventional baselines across multiple gaming environments. Furthermore, CFT maintains competitive performance even when trained with only 50% of the labeled data from the target game.

---

## 61. MINERVA: Mutual Information Neural Estimation for Supervised Feature Selection

**论文链接:** [http://arxiv.org/abs/2510.02610v2](http://arxiv.org/abs/2510.02610v2)

**作者:** Taurai Muvunza, Egor Kraev, Pere Planell-Morell, Alexander Y. Shestopaloff

**发布时间:** 2025-10-02

**备注:** 23 pages

### GPT解析

### 总结

MINERVA是一种创新的监督特征选择方法，利用神经网络估计特征和目标之间的互信息，通过两阶段过程解耦表示学习和特征选择，能够捕捉高阶特征交互，解决了传统方法在处理复杂特征-目标关系时的局限性。

### 背景

现有的特征过滤器依赖于统计成对依赖性指标来建模特征-目标关系，但当目标依赖于高阶特征交互而非单个特征贡献时，这种方法可能失效。

### 目的

提出一种新的监督特征选择方法，能够捕捉复杂特征-目标关系，特别是当目标依赖于高阶特征交互时。

### 方法

引入Mutual Information Neural Estimation Regularized Vetting Algorithm (MINERVA)，基于特征和目标之间互信息的神经估计进行监督特征选择；使用神经网络参数化互信息的近似，并使用精心设计的损失函数结合稀疏诱导正则化器进行特征选择；采用两阶段过程解耦表示学习与特征选择，通过评估特征子集集合来捕捉复杂特征-目标关系。

### 主要发现

展示了文献中很少捕捉到的普遍依赖结构；所提出的方法能够有效捕捉这些复杂的特征-目标关系；在合成和真实欺诈数据集上的实验结果证明了该方法的有效性。

### 结论

该方法能够执行精确解，具有更好的泛化能力，能够更准确地表达特征重要性。

### 翻译

现有的特征过滤器依赖于统计成对依赖性指标来建模特征-目标关系，但当目标依赖于高阶特征交互而非单个特征贡献时，这种方法可能失效。我们引入了互信息神经估计正则化审查算法（MINERVA），这是一种基于特征和目标之间互信息神经估计的监督特征选择新方法。我们使用神经网络参数化互信息的近似，并使用精心设计的损失函数结合稀疏诱导正则化器进行特征选择。我们的方法通过两阶段过程实现，将表示学习与特征选择解耦，确保更好的泛化能力和更准确的特征重要性表达。我们展示了文献中很少捕捉到的普遍依赖结构，并通过评估特征子集集合证明我们提出的方法能够有效捕捉这些复杂的特征-目标关系。在合成和真实欺诈数据集上的实验结果证明了我们方法的有效性及其执行精确解的能力。


### 论文摘要

Existing feature filters rely on statistical pair-wise dependence metrics to model feature-target relationships, but this approach may fail when the target depends on higher-order feature interactions rather than individual contributions. We introduce Mutual Information Neural Estimation Regularized Vetting Algorithm (MINERVA), a novel approach to supervised feature selection based on neural estimation of mutual information between features and targets. We paramaterize the approximation of mutual information with neural networks and perform feature selection using a carefully designed loss function augmented with sparsity-inducing regularizers. Our method is implemented in a two-stage process to decouple representation learning from feature selection, ensuring better generalization and a more accurate expression of feature importance. We present examples of ubiquitous dependency structures that are rarely captured in literature and show that our proposed method effectively captures these complex feature-target relationships by evaluating feature subsets as an ensemble. Experimental results on synthetic and real-life fraud datasets demonstrate the efficacy of our method and its ability to perform exact solutions.

---

## 62. Learning Representations Through Contrastive Neural Model Checking

**论文链接:** [http://arxiv.org/abs/2510.01853v2](http://arxiv.org/abs/2510.01853v2)

**作者:** Vladimir Krsmanovic, Matthias Cosler, Mohamed Ghanem, Bernd Finkbeiner

**发布时间:** 2025-10-02

### GPT解析

### 总结

本文提出对比神经网络模型检查(CNML)方法，将模型检查任务作为学习对齐表征的指导信号，通过自监督对比目标将逻辑规范和系统嵌入共享潜在空间，在工业启发检索任务上表现优异，且学习表征可有效迁移到下游任务并推广到复杂公式。

### 背景

模型检查是验证安全关键系统是否符合形式规范的关键技术，深度学习应用显示出前景，但表征学习在形式验证领域仍探索不足。

### 目的

探索表征学习在形式验证领域的应用，利用模型检查任务作为学习表征的指导信号，提高形式验证效果。

### 方法

提出对比神经网络模型检查(CNML)方法，通过自监督对比目标将逻辑规范和系统共同嵌入到共享的潜在空间中。

### 主要发现

CNML在工业启发的检索任务上明显优于算法和神经基线；学习到的表征可以有效地迁移到下游任务；学习到的表征可以推广到更复杂的公式。

### 结论

模型检查可以作为学习形式语言表征的目标函数。

### 翻译

模型检查是验证安全关键系统是否符合形式规范的关键技术，最近深度学习的应用显示出前景。然而，尽管表征学习在视觉和语言领域无处不在，但在形式验证领域仍然探索不足。我们引入对比神经网络模型检查(CNML)，一种利用模型检查任务作为学习对齐表征指导信号的新方法。CNML通过自监督对比目标将逻辑规范和系统共同嵌入到共享的潜在空间中。在工业启发的检索任务上，CNML在跨模态和模态内设置上都明显优于算法和神经基线。我们进一步表明，学习到的表征可以有效地迁移到下游任务并推广到更复杂的公式。这些发现表明，模型检查可以作为学习形式语言表征的目标函数。


### 论文摘要

Model checking is a key technique for verifying safety-critical systems against formal specifications, where recent applications of deep learning have shown promise. However, while ubiquitous for vision and language domains, representation learning remains underexplored in formal verification. We introduce Contrastive Neural Model Checking (CNML), a novel method that leverages the model checking task as a guiding signal for learning aligned representations. CNML jointly embeds logical specifications and systems into a shared latent space through a self-supervised contrastive objective. On industry-inspired retrieval tasks, CNML considerably outperforms both algorithmic and neural baselines in cross-modal and intra-modal settings. We further show that the learned representations effectively transfer to downstream tasks and generalize to more complex formulas. These findings demonstrate that model checking can serve as an objective for learning representations for formal languages.

---

## 63. Trajectory prediction for heterogeneous agents: A performance analysis on small and imbalanced datasets

**论文链接:** [http://arxiv.org/abs/2510.03776v1](http://arxiv.org/abs/2510.03776v1)

**作者:** Tiago Rodrigues de Almeida, Yufei Zhu, Andrey Rudenko, Tomasz P. Kucner, Johannes A. Stork, Martin Magnusson, Achim J. Lilienthal

**发布时间:** 2025-10-04

**DOI:** 10.1109/LRA.2024.3408510

**备注:** This paper has been accepted to the IEEE Robotics and Automation  Letters journal and presented at the 40th Anniversary of the IEEE  International Conference on Robotics and Automation, which was held in  Rotterdam, Netherlands on 23-26 September, 2024

### GPT解析

### 总结

本研究探讨了在复杂动态环境中导航的机器人如何通过预测周围智能体的未来行动和意图来高效到达目标并避免碰撞。研究提出并评估了基于类别的条件运动预测方法，特别是在数据有限或类别不平衡情况下的表现。

### 背景

机器人和其他智能系统在复杂动态环境中导航时，需要准确预测周围智能体的未来行为以实现高效导航和避免碰撞。这些智能体的动态行为与其任务、角色或可观察标签密切相关。基于类别的运动预测可以减少预测不确定性，提高对不同类型智能体预测的准确性，但这一领域在现有研究中很少被探索，特别是在移动机器人和有限数据应用方面。

### 目的

分析不同基于类别的轨迹预测方法在两个数据集上的表现，提出一套基于条件模式和高效深度学习的基线方法，并评估它们在机器人技术和户外数据集上的性能，特别关注在数据不平衡或新环境（数据不足）情况下的表现。

### 方法

研究者在两个数据集上分析了不同的基于类别的轨迹预测方法，提出了一套基于条件模式和高效深度学习的基线方法，并在机器人技术和户外数据集（THOR-MAGNI和斯坦福无人机数据集）上评估了它们的性能。

### 主要发现

实验表明，在大多数情况下，考虑类别标签可以提高所有方法的准确性。在从不平衡数据集学习或在新环境中存在显著差异。具体来说，深度学习方法在平衡数据集上表现更好，但在数据有限的应用中（如机器人在新环境中的冷启动或类别不平衡），基于模式的方法可能更可取。

### 结论

基于类别的条件运动预测可以提高预测准确性，但在不同数据分布和环境条件下，不同方法的表现有所差异。在数据有限或类别不平衡的情况下，基于模式的方法可能比深度学习方法更适用，这为实际应用中的方法选择提供了指导。

### 翻译

机器人和其他智能系统在复杂动态环境中导航时，应该预测周围智能体的未来行动和意图，以高效地达到目标并避免碰撞。这些智能体的动态行为很大程度上取决于它们的任务、角色或可观察标签。因此，基于类别的条件运动预测是一种减少预测不确定性并提高对异构智能体预测准确性的有吸引力的方法。然而，这在现有技术中很少被探索，特别是对于移动机器人和有限数据应用。在本文中，我们在两个数据集上分析了不同的基于类别的轨迹预测方法。我们提出了一套基于条件模式和高效深度学习的基线方法，并在机器人技术和户外数据集（THOR-MAGNI和斯坦福无人机数据集）上评估了它们的性能。我们的实验表明，在大多数情况下，当考虑类别标签时，所有方法都能提高准确性。更重要的是，我们观察到在从不平衡数据集学习或在新环境中（数据不足）存在显著差异。特别是，我们发现深度学习方法在平衡数据集上表现更好，但在数据有限的应用中，例如机器人在新环境中的冷启动或类别不平衡，基于模式的方法可能更可取。


### 论文摘要

Robots and other intelligent systems navigating in complex dynamic environments should predict future actions and intentions of surrounding agents to reach their goals efficiently and avoid collisions. The dynamics of those agents strongly depends on their tasks, roles, or observable labels. Class-conditioned motion prediction is thus an appealing way to reduce forecast uncertainty and get more accurate predictions for heterogeneous agents. However, this is hardly explored in the prior art, especially for mobile robots and in limited data applications. In this paper, we analyse different class-conditioned trajectory prediction methods on two datasets. We propose a set of conditional pattern-based and efficient deep learning-based baselines, and evaluate their performance on robotics and outdoors datasets (TH\"OR-MAGNI and Stanford Drone Dataset). Our experiments show that all methods improve accuracy in most of the settings when considering class labels. More importantly, we observe that there are significant differences when learning from imbalanced datasets, or in new environments where sufficient data is not available. In particular, we find that deep learning methods perform better on balanced datasets, but in applications with limited data, e.g., cold start of a robot in a new environment, or imbalanced classes, pattern-based methods may be preferable.

---

## 64. Modeling information acquisition via f-divergence and duality

**论文链接:** [http://arxiv.org/abs/2510.03482v1](http://arxiv.org/abs/2510.03482v1)

**作者:** Alex Bloedel, Tommaso Denti, Luciano Pomatto

**发布时间:** 2025-10-03

### GPT解析

### 总结

本文引入了一种基于多元统计差异理论的新实验成本函数f-information，推广了Sims的理性不注意模型和后验可分离成本函数类，并通过推导最优性条件来表征其行为预测。

### 背景

f-information基于多元统计差异理论，是对现有理性不注意模型和后验可分离成本函数类的推广。

### 目的

扩展Matejka和McKay (2015)以及Caplin、Dean和Leahy (2019)在互信息方面的工作，研究f-information在决策问题中的含义。

### 方法

通过推导最优性条件来表征f-information的行为预测，并在多个典型决策问题中应用这些工具。

### 主要发现

f-information框架可以使用微观经济学的熟悉方法进行分析，包括凸对偶和Arrow-Pratt预期效用方法。

### 结论

f-information是一种通用的实验成本函数，能够扩展现有理论并应用于多种决策问题分析。

### 翻译

我们引入了一种基于多元统计差异理论的新实验成本函数f-information，它推广了Sims的经典理性不注意模型以及后验可分离成本函数类。我们通过推导最优性条件来表征其行为预测，这些条件扩展了Matejka和McKay (2015)以及Caplin、Dean和Leahy (2019)在互信息方面的工作。利用这些工具，我们研究了f-information在多个典型决策问题中的含义。该框架的一个优势是它可以使用微观经济学的熟悉方法进行分析：凸对偶和Arrow-Pratt预期效用方法。


### 论文摘要

We introduce a new cost function over experiments, f-information, based on the theory of multivariate statistical divergences, that generalizes Sims's classic model of rational inattention as well as the class of posterior-separable cost functions. We characterize its behavioral predictions by deriving optimality conditions that extend those of Matejka and McKay (2015) and Caplin, Dean, and Leahy (2019) beyond mutual information. Using these tools, we study the implications of f-information in a number of canonical decision problems. A strength of the framework is that it can be analyzed using familiar methods of microeconomics: convex duality and the Arrow-Pratt approach to expected utility.

---

## 65. TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration

**论文链接:** [http://arxiv.org/abs/2510.05102v1](http://arxiv.org/abs/2510.05102v1)

**作者:** Cheng Xin, Fan Xu, Xin Ding, Jie Gao, Jiaxin Ding

**发布时间:** 2025-10-06

**备注:** submitted to ICML 2025

### GPT解析

### 总结

本文提出了一种名为TopInG的拓扑可解释图学习框架，用于解决图神经网络在关键决策应用中缺乏可解释性的问题。

### 背景

图神经网络在多个科学领域取得了显著成功，但在关键决策应用中常因缺乏可解释性而受限。现有的内在可解释GNN方法在处理复杂多样的基础理由子图时面临挑战。

### 目的

开发一种新的拓扑框架，能够识别持久的理由子图，以提高图神经网络的可解释性，同时保持预测性能。

### 方法

TopInG采用持久同调来识别持久的理由子图，使用理由过滤学习方法建模理由子图的自回归生成过程，并引入拓扑差异作为自调整拓扑约束，强制理由子图与无关部分之间的持久拓扑区别。

### 主要发现

在特定条件下，所提出的损失函数能被真实值唯一优化；广泛的实验表明，TopInG在处理各种理由子图、平衡预测性能与可解释性以及减轻虚假相关性方面有效；在预测准确性和解释质量上都优于最先进的方法。

### 结论

TopInG为图神经网络的可解释性提供了一种有效的解决方案，特别适用于处理复杂多样的理由子图，并能平衡预测性能与可解释性。

### 翻译

图神经网络在各个科学领域已展现出显著的成功，但它们在关键决策应用中的采用往往因缺乏可解释性而受到阻碍。最近，内在可解释的图神经网络已被研究用于通过识别图中的理由子结构来提供对模型预测的洞察。然而，当基础理由子图复杂多样时，现有方法面临挑战。在这项工作中，我们提出了TopInG：拓扑可解释图学习，这是一种利用持久同调来识别持久理由子图的新拓扑框架。TopInG采用理由过滤学习方法来建模理由子图的自回归生成过程，并引入了一种自调整拓扑约束，称为拓扑差异，以强制理由子图与无关对应物之间的持久拓扑区别。我们在特定条件下提供了理论保证，即我们的损失函数能被真实值唯一优化。广泛的实验证明了TopInG在处理关键挑战方面的有效性，如处理各种理由子图、平衡预测性能与可解释性以及减轻虚假相关性。结果表明，我们的方法在预测准确性和解释质量上都优于最先进的方法。


### 论文摘要

Graph Neural Networks (GNNs) have shown remarkable success across various scientific fields, yet their adoption in critical decision-making is often hindered by a lack of interpretability. Recently, intrinsically interpretable GNNs have been studied to provide insights into model predictions by identifying rationale substructures in graphs. However, existing methods face challenges when the underlying rationale subgraphs are complex and varied. In this work, we propose TopInG: Topologically Interpretable Graph Learning, a novel topological framework that leverages persistent homology to identify persistent rationale subgraphs. TopInG employs a rationale filtration learning approach to model an autoregressive generation process of rationale subgraphs, and introduces a self-adjusted topological constraint, termed topological discrepancy, to enforce a persistent topological distinction between rationale subgraphs and irrelevant counterparts. We provide theoretical guarantees that our loss function is uniquely optimized by the ground truth under specific conditions. Extensive experiments demonstrate TopInG's effectiveness in tackling key challenges, such as handling variform rationale subgraphs, balancing predictive performance with interpretability, and mitigating spurious correlations. Results show that our approach improves upon state-of-the-art methods on both predictive accuracy and interpretation quality.

---

## 66. On the sensitivity of different galaxy properties to warm dark matter

**论文链接:** [http://arxiv.org/abs/2510.05037v1](http://arxiv.org/abs/2510.05037v1)

**作者:** Belén Costanza, Bonny Y. Wang, Francisco Villaescusa-Navarro, Alex M. Garcia, Jonah C. Rose, Mark Vogelsberger, Paul Torrey, Arya Farahi, Xuejian Shen, Ilem Leisher

**发布时间:** 2025-10-06

**DOI:** 10.3847/1538-4357/ae0e6c

**备注:** Accepted for publication in The Astrophysical Journal

### GPT解析

### 总结

本研究使用DREAMS项目的1024个先进宇宙流体动力学模拟，探索温暗物质粒子质量对星系属性的影响。研究发现亚晕气体质量是约束WDM质量的最具信息量特征，确定系数高达0.9。通过比较MLP和GNN方法，表明预测能力主要来自全局描述符，晕级别信息提供有限增益。

### 背景

研究温暗物质粒子质量如何影响星系属性，基于DREAMS项目的1024个最先进的宇宙流体动力学模拟。

### 目的

探索温暗物质粒子质量对星系属性的影响，并开发有效的方法来约束WDM质量。

### 方法

使用多层感知器结合标准化流分析星系群体的全局统计描述符；采用符号回归提取简单关系；使用图神经网络结合标准化流分析单个暗物质晕的属性来推断WDM质量。

### 主要发现

亚晕气体质量是约束WDM质量的最具信息量特征，确定系数R平方等于0.9；基于全局特征的GNN方法仅比仅基于全局特征的MLP模型产生微小的改进。

### 结论

预测WDM质量的能力主要存在于星系群体的全局描述符中，晕级别信息提供的额外增益有限。

### 翻译

我们使用DREAMS项目的1024个最先进的宇宙流体动力学模拟研究温暗物质粒子质量对星系属性的影响。我们首先使用多层感知器结合标准化流来探索星系群体的全局统计描述符，如14种星系属性的平均值、标准差和直方图。我们发现亚晕气体质量是约束WDM质量的最具信息量的特征，确定系数R平方等于0.9。我们使用符号回归提取与WDM粒子质量的简单、可解释的关系。最后，我们采用更局部化的方法，选择单个暗物质晕，使用图神经网络结合标准化流来推断WDM质量，将亚晕属性作为节点特征，全局模拟统计作为图级特征。GNN方法仅比仅基于全局特征的MLP模型产生微小的改进，表明预测能力主要存在于全局描述符中，晕级别信息的增益有限。


### 论文摘要

We study the impact of warm dark matter (WDM) particle mass on galaxy properties using 1,024 state-of-the-art cosmological hydrodynamical simulations from the DREAMS project. We begin by using a Multilayer Perceptron (MLP) coupled with a normalizing flow to explore global statistical descriptors of galaxy populations, such as the mean, standard deviation, and histograms of 14 galaxy properties. We find that subhalo gas mass is the most informative feature for constraining the WDM mass, achieving a determination coefficient of R^2 = 0.9. We employ symbolic regression to extract simple, interpretable relations with the WDM particle mass. Finally, we adopt a more localized approach by selecting individual dark matter halos and using a Graph Neural Network (GNN) with a normalizing flow to infer the WDM mass, incorporating subhalo properties as node features and global simulation statistics as graph-level features. The GNN approach yields only a residual improvement over MLP models based solely on global features, indicating that most of the predictive power resides in the global descriptors, with only marginal gains from halo-level information.

---

## 67. NatGVD: Natural Adversarial Example Attack towards Graph-based Vulnerability Detection

**论文链接:** [http://arxiv.org/abs/2510.04987v1](http://arxiv.org/abs/2510.04987v1)

**作者:** Avilash Rath, Weiliang Qi, Youpeng Li, Xinda Wang

**发布时间:** 2025-10-06

**备注:** 10 pages, 2 figures (2 additional figures in Appendices)

### GPT解析

### 总结

本文提出了一种名为NatGVD的新型攻击方法，用于生成自然对抗性漏洞代码，绕过基于图神经网络和图感知transformer的漏洞检测器，并通过实验证明其有效性。

### 背景

基于图的模型在代码分析任务中表现出色，但这些模型在漏洞检测中对对抗样本攻击的鲁棒性仍是一个开放问题。

### 目的

提出一种名为NatGVD的新型攻击方法，用于生成自然对抗性漏洞代码，绕过基于GNN和图感知transformer的漏洞检测器。

### 方法

NatGVD采用一组代码转换方法，这些方法修改图结构同时保留代码语义。与之前的工作不同，NatGVD考虑了自然性要求：生成的样本不应被人类或程序分析工具轻易识别。

### 主要发现

在最新的漏洞检测系统上对NatGVD进行广泛评估，结果显示在基于GNN的检测器和基于图感知transformer的检测器中，逃避率高达53.04%。

### 结论

作者还探讨了潜在的防御策略，以增强这些系统对NatGVD的鲁棒性。

### 翻译

基于图的模型学习丰富的代码图结构信息，并在各种代码分析任务中表现出优越的性能。然而，在漏洞检测背景下，这些模型对抗对抗样本攻击的鲁棒性仍然是一个开放问题。本文提出了NatGVD，一种新颖的攻击方法，用于生成自然的对抗性漏洞代码，以规避基于GNN和图感知transformer的漏洞检测器。NatGVD采用一组代码转换，这些转换修改图结构同时保留代码语义。与之前注入死代码或不相关代码的工作不同，NatGVD考虑了自然性要求：生成的样本不应被人类或程序分析工具轻易识别。通过对最先进的漏洞检测系统进行广泛评估，结果显示在基于GNN的检测器和基于图感知transformer的检测器中，逃避率高达53.04%。我们还探讨了潜在的防御策略，以增强这些系统对NatGVD的鲁棒性。


### 论文摘要

Graph-based models learn rich code graph structural information and present superior performance on various code analysis tasks. However, the robustness of these models against adversarial example attacks in the context of vulnerability detection remains an open question. This paper proposes NatGVD, a novel attack methodology that generates natural adversarial vulnerable code to circumvent GNN-based and graph-aware transformer-based vulnerability detectors. NatGVD employs a set of code transformations that modify graph structure while preserving code semantics. Instead of injecting dead or unrelated code like previous works, NatGVD considers naturalness requirements: generated examples should not be easily recognized by humans or program analysis tools. With extensive evaluation of NatGVD on state-of-the-art vulnerability detection systems, the results reveal up to 53.04% evasion rate across GNN-based detectors and graph-aware transformer-based detectors. We also explore potential defense strategies to enhance the robustness of these systems against NatGVD.

---

## 68. Bond-Centered Molecular Fingerprint Derivatives: A BBBP Dataset Study

**论文链接:** [http://arxiv.org/abs/2510.04837v1](http://arxiv.org/abs/2510.04837v1)

**作者:** Guillaume Godin

**发布时间:** 2025-10-06

**备注:** 14 pages, 10 figures, 1 table

### GPT解析

### 总结

本研究提出了一种名为Bond Centered FingerPrint (BCFP)的分子指纹方法，作为Extended-Connectivity FingerPrints (ECFP)的补充。通过在BBBP分类任务上的评估，证明了结合使用这两种指纹可以显著提高预测性能，并提出了BCFP-Sort&Slice特征组合方案以进一步提升效果。

### 背景

Extended-Connectivity FingerPrints (ECFP)是分子描述的常用方法，但可能存在局限性。Bond Centered FingerPrint (BCFP)作为一种以化学键为中心的替代方法被提出，旨在补充ECFP的不足。

### 目的

开发一种轻量级、以键为中心的分子描述符，能够与以原子为中心的圆形指纹(如ECFP)互补，提高Blood-Brain Barrier Penetration (BBBP)预测性能。

### 方法

提出静态BCFP，模仿有向消息传递GNN(如ChemProp)使用的键卷积；使用快速随机森林模型在BBBP分类任务上评估；通过分层交叉验证比较ECFP与BCFP组合的效果；提出BCFP-Sort&Slice特征组合方案；将复合特征(键和原子特征)与MGTP预测方法进行比较。

### 主要发现

将ECFP与BCFP连接在一起比单独使用任何一种描述符都能提高AUROC和AUPRC；通过Turkey HSD多重比较分析确认了这一结果；半径r = 1表现最佳，r = 2未产生统计上可分离的增益；BCFP-Sort&Slice方案保留了ECFP中的OOV信息，同时实现BCFP变体的紧凑连接；使用复合新特征在BBBP评估中优于MGTP预测。

### 结论

轻量级的、以键为中心的描述符可以补充以原子为中心的圆形指纹，并为BBBP预测提供强大快速的基线。

### 翻译

键中心指纹(BCFP)是对扩展连接性指纹(ECFP)的一种互补的、以键为中心的替代方案。我们引入了一种静态BCFP，它模仿了ChemProp等有向消息传递GNN使用的键卷积，并使用快速随机森林模型在血脑屏障穿透(BBBP)分类任务上对其进行了评估。在分层交叉验证中，将ECFP与BCFP连接在一起，比单独使用任何一种描述符都能提高AUROC和AUPRC，这一点通过Turkey HSD多重比较分析得到了确认。在半径方面，r = 1表现最佳；在相同测试下，r = 2没有产生统计上可分离的增益。我们进一步提出了BCFP-Sort&Slice，这是一种简单的特征组合方案，它保留了ECFP计数向量中固有的词汇表外(OOV)计数信息，同时实现了BCFP变体的紧凑无哈希连接。我们还使用这种复合新特征(键和原子特征)在BBBP评估中优于MGTP预测。这些结果表明，轻量级的、以键为中心的描述符可以补充以原子为中心的圆形指纹，并为BBBP预测提供强大快速的基线。


### 论文摘要

Bond Centered FingerPrint (BCFP) are a complementary, bond-centric alternative to Extended-Connectivity Fingerprints (ECFP). We introduce a static BCFP that mirrors the bond-convolution used by directed message-passing GNNs like ChemProp, and evaluate it with a fast rapid Random Forest model on Brain-Blood Barrier Penetration (BBBP) classification task. Across stratified cross-validation, concatenating ECFP with BCFP consistently improves AUROC and AUPRC over either descriptor alone, as confirmed by Turkey HSD multiple-comparison analysis. Among radii, r = 1 performs best; r = 2 does not yield statistically separable gains under the same test. We further propose BCFP-Sort&Slice, a simple feature-combination scheme that preserves the out-of-vocabulary (OOV) count information native to ECFP count vectors while enabling compact unhashed concatenation of BCFP variants. We also outperform the MGTP prediction on our BBBP evaluation, using such composite new features bond and atom features. These results show that lightweight, bond-centered descriptors can complement atom-centered circular fingerprints and provide strong, fast baselines for BBBP prediction.

---

## 69. GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning

**论文链接:** [http://arxiv.org/abs/2510.04567v1](http://arxiv.org/abs/2510.04567v1)

**作者:** Weishuo Ma, Yanbo Wang, Xiyuan Wang, Lei Zou, Muhan Zhang

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出了GILT框架，一种无LLM和无调整的架构，用于处理图数据的极端异质性，实现了高效的上下文学习和少样本分类任务。

### 背景

图神经网络(GNNs)在处理关系数据时强大，但在推广到未见过的图时存在困难，导致图基础模型(GFMs)的发展。当前GFMs面临图数据极端异质性的挑战，每个图可能具有独特的特征空间、标签集和拓扑结构。

### 目的

解决现有GFMs面临的异构性问题，克服两种主要范式的局限性：基于大型语言模型的方法难以处理图中的数值特征，而基于结构的模型适应新任务需要昂贵的逐图调整阶段。

### 方法

提出GILT (Graph In-context Learning Transformer)框架，构建在无LLM和无调整的架构上，引入基于令牌的框架用于图中的上下文学习，将节点、边和图级别的分类任务重新框架化为统一框架，设计为处理通用数值特征，并能够从上下文中动态理解类语义。

### 主要发现

GILT在少样本学习中表现更强，与基于LLM或基于调整的基线相比，用时显著减少，验证了该方法的有效性。

### 结论

GILT成功解决了现有GFMs面临的异构性问题，通过无LLM和无调整的架构，实现了高效的图学习和适应。

### 翻译

图神经网络(GNNs)是处理关系数据的强大工具，但通常难以推广到未见过的图，这促进了图基础模型(GFMs)的发展。然而，当前GFMs面临图数据极端异质性的挑战，其中每个图可以具有独特的特征空间、标签集和拓扑结构。为此，出现了两种主要范式。第一种利用大型语言模型(LLMs)，但根本上依赖于文本，因此难以处理大量图中的数值特征。第二种预训练基于结构的模型，但适应新任务通常需要昂贵的逐图调整阶段，造成关键的效率瓶颈。在这项工作中，我们超越了这些限制，引入了GILT (Graph In-context Learning Transformer)，一个构建在无LLM和无调整架构上的框架。GILT引入了一种基于令牌的框架，用于图中的上下文学习(ICL)，将跨越节点、边和图级别的分类任务重新框架化为统一框架。这种机制是处理异质性的关键，因为它被设计为操作通用数值特征。此外，它从上下文动态理解类语义的能力实现了无调整适应。全面的实验表明，GILT比基于LLM或基于调整的基线实现更强的少样本性能，且用时显著减少，验证了我们方法的有效性。


### 论文摘要

Graph Neural Networks (GNNs) are powerful tools for precessing relational data but often struggle to generalize to unseen graphs, giving rise to the development of Graph Foundational Models (GFMs). However, current GFMs are challenged by the extreme heterogeneity of graph data, where each graph can possess a unique feature space, label set, and topology. To address this, two main paradigms have emerged. The first leverages Large Language Models (LLMs), but is fundamentally text-dependent, thus struggles to handle the numerical features in vast graphs. The second pre-trains a structure-based model, but the adaptation to new tasks typically requires a costly, per-graph tuning stage, creating a critical efficiency bottleneck. In this work, we move beyond these limitations and introduce \textbf{G}raph \textbf{I}n-context \textbf{L}earning \textbf{T}ransformer (GILT), a framework built on an LLM-free and tuning-free architecture. GILT introduces a novel token-based framework for in-context learning (ICL) on graphs, reframing classification tasks spanning node, edge and graph levels in a unified framework. This mechanism is the key to handling heterogeneity, as it is designed to operate on generic numerical features. Further, its ability to understand class semantics dynamically from the context enables tuning-free adaptation. Comprehensive experiments show that GILT achieves stronger few-shot performance with significantly less time than LLM-based or tuning-based baselines, validating the effectiveness of our approach.

---

## 70. Fractional Heat Kernel for Semi-Supervised Graph Learning with Small Training Sample Size

**论文链接:** [http://arxiv.org/abs/2510.04440v1](http://arxiv.org/abs/2510.04440v1)

**作者:** Farid Bozorgnia, Vyacheslav Kungurtsev, Shirali Kadyrov, Mohsen Yousefnezhad

**发布时间:** 2025-10-06

### GPT解析

### 总结

本研究引入了基于分数热核动力学的标签传播和自训练新算法，并将其集成到图神经网络架构中，通过切比雪夫多项式近似使大规模图计算可行。该方法在标记样本有限的情况下表现优越。

### 背景

研究基于信息论与抛物线演化方程物理学之间的经典对应关系，通过扩展经典扩散模型到拉普拉斯算子的分数幂，实现非局部交互和更全局的标签扩散。

### 目的

开发一种在标记样本有限情况下有效的标签传播和自训练方法，增强图神经网络的表示能力，同时保持大规模图计算的可行性。

### 方法

使用带源项的分数热核动力学进行标签传播和自训练，将分数热核集成到图卷积网络和图注意力网络等架构中，应用切比雪夫多项式近似处理大规模图计算。

### 主要发现

通过扩展经典扩散模型到拉普拉斯算子的分数幂，非局部交互能够提供更全局的标签扩散，在已知标签监督和图上扩散之间的特殊平衡使该方法在标记样本有限情况下特别有效。

### 结论

所提出的方法在标准数据集上证明了有效性，为半监督学习提供了一种新的解决方案，特别是在标记数据稀缺的场景下。

### 翻译

在这项工作中，我们引入了基于带源项的分数热核动力学的标签传播和自训练新算法。我们通过信息论与抛物线演化方程物理学之间的经典对应关系来论证该方法。我们将分数热核集成到图卷积网络和图注意力等图神经网络架构中，通过自适应的多跳扩散增强其表达能力。通过应用切比雪夫多项式近似，大规模图变得计算可行。变分公式的论证表明，通过将经典扩散模型扩展到拉普拉斯算子的分数幂，非局部交互提供了更全局的标签扩散。在已知标签监督和图上扩散之间的特殊平衡在只有少量标记训练样本的情况下特别有利。我们在标准数据集上证明了这种方法的有效性。


### 论文摘要

In this work, we introduce novel algorithms for label propagation and self-training using fractional heat kernel dynamics with a source term. We motivate the methodology through the classical correspondence of information theory with the physics of parabolic evolution equations. We integrate the fractional heat kernel into Graph Neural Network architectures such as Graph Convolutional Networks and Graph Attention, enhancing their expressiveness through adaptive, multi-hop diffusion. By applying Chebyshev polynomial approximations, large graphs become computationally feasible. Motivating variational formulations demonstrate that by extending the classical diffusion model to fractional powers of the Laplacian, nonlocal interactions deliver more globally diffusing labels. The particular balance between supervision of known labels and diffusion across the graph is particularly advantageous in the case where only a small number of labeled training examples are present. We demonstrate the effectiveness of this approach on standard datasets.

---

## 71. FoilDiff: A Hybrid Transformer Backbone for Diffusion-based Modelling of 2D Airfoil Flow Fields

**论文链接:** [http://arxiv.org/abs/2510.04325v1](http://arxiv.org/abs/2510.04325v1)

**作者:** Kenechukwu Ogbuagu, Sepehr Maleki, Giuseppe Bruni, Senthil Krishnababu

**发布时间:** 2025-10-05

### GPT解析

### 总结

该论文介绍了一种名为FoilDiff的基于扩散模型的替代模型，用于预测翼型周围的流场。该模型采用混合主干去噪网络，结合卷积特征提取和基于Transformer的全局注意力机制，实现了更准确和适应性更强的流场结构表示。

### 背景

翼型周围流场的准确预测对空气动力学设计和优化至关重要。计算流体动力学(CFD)模型虽然有效，但计算成本高昂，这促使了替代模型的发展以实现更快速的预测。

### 目的

开发一种基于扩散模型的替代模型，用于准确、高效地预测翼型周围的流场，同时提供更好的预测不确定性校准。

### 方法

提出FoilDiff，一种基于扩散的替代模型，具有混合主干去噪网络。该设计结合了卷积特征提取和基于Transformer的全局注意力，以生成更适应性和准确的流场结构表示。使用去噪扩散隐式模型(DDIM)采样优化采样过程，同时不损害模型泛化能力。使用雷诺数、攻角和翼型几何的编码表示定义输入空间，以在广泛的空气动力学条件下实现泛化。

### 主要发现

与最先进的模型相比，FoilDiff显示出显著的性能改进，在相同数据集上的平均预测误差减少了高达85%。结果表明，FoilDiff可以提供比现有基于扩散的模型更准确的预测和更好的预测不确定性校准。

### 结论

FoilDiff作为基于扩散的替代模型，能够有效预测复杂流场，提供比现有方法更准确的预测结果和更好的不确定性估计，为空气动力学设计和优化提供了有价值的工具。

### 翻译

翼型周围流场的准确预测对空气动力学设计和优化至关重要。计算流体动力学(CFD)模型虽然有效，但计算成本高昂，这促使了替代模型的发展以实现更快速的预测。这些替代模型可以基于深度学习架构，如卷积神经网络(CNNs)、图神经网络(GNNs)和扩散模型(DMs)。扩散模型在预测复杂流场方面显示出巨大潜力。在这项工作中，我们提出了FoilDiff，一种具有混合主干去噪网络的基于扩散的替代模型。这种混合设计结合了卷积特征提取和基于Transformer的全局注意力，以生成更适应性和准确的流场结构表示。FoilDiff利用去噪扩散隐式模型(DDIM)采样，在不损害模型泛化能力的情况下优化采样过程的效率。我们使用雷诺数、攻角和翼型几何的编码表示来定义输入空间，以在广泛的空气动力学条件下实现泛化。与最先进的模型相比，FoilDiff显示出显著的性能改进，在相同数据集上的平均预测误差减少了高达85%。结果表明，FoilDiff可以提供比现有基于扩散的模型更准确的预测和更好的预测不确定性校准。


### 论文摘要

The accurate prediction of flow fields around airfoils is crucial for aerodynamic design and optimisation. Computational Fluid Dynamics (CFD) models are effective but computationally expensive, thus inspiring the development of surrogate models to enable quicker predictions. These surrogate models can be based on deep learning architectures, such as Convolutional Neural Networks (CNNs), Graph Neural Networks (GNNs), and Diffusion Models (DMs). Diffusion models have shown significant promise in predicting complex flow fields. In this work, we propose FoilDiff, a diffusion-based surrogate model with a hybrid-backbone denoising network. This hybrid design combines the power of convolutional feature extraction and transformer-based global attention to generate more adaptable and accurate representations of flow structures. FoilDiff takes advantage of Denoising Diffusion Implicit Model (DDIM) sampling to optimise the efficiency of the sampling process at no additional cost to model generalisation. We used encoded representations of Reynolds number, angle of attack, and airfoil geometry to define the input space for generalisation across a wide range of aerodynamic conditions. When evaluated against state-of-the-art models, FoilDiff shows significant performance improvements, with mean prediction errors reducing by up to 85\% on the same datasets. The results have demonstrated that FoilDiff can provide both more accurate predictions and better-calibrated predictive uncertainty than existing diffusion-based models.

---

## 72. A Hybrid GNN-IZR Framework for Fast and Empirically Robust AC Power Flow Analysis in Radial Distribution Systems

**论文链接:** [http://arxiv.org/abs/2510.04264v1](http://arxiv.org/abs/2510.04264v1)

**作者:** Mohamed Shamseldein

**发布时间:** 2025-10-05

### GPT解析

### 总结

本文介绍了一种结合图神经网络(GNN)和隐式Z总线递归(IZR)方法的混合框架，用于解决交流潮流计算中的速度与可靠性权衡问题。该框架通过物理信息GNN进行快速预测，并使用两级触发机制识别压力情况，调用IZR求解器作为故障保护，在测试中实现了100%的成功率。

### 背景

交流潮流计算问题需要在数据驱动模型的速度和分析求解器的可靠性之间做出权衡。传统分析方法可靠但速度慢，而数据驱动模型速度快但可靠性不足。

### 目的

开发一种混合框架，结合数据驱动模型的速度和分析求解器的可靠性，实现近实时分析大量场景的能力。

### 方法

结合图神经网络和隐式Z总线递归方法，使用物理信息GNN进行快速初始预测，通过两级触发机制识别压力情况，将失败案例(定义为最大功率失配超过0.1 p.u.)委托给IZR求解器处理。

### 主要发现

在7,500个压力场景测试中，纯GNN模型失败率为13.11%，而混合框架实现了0.00%的失败率和100%的成功率；消融研究表明，物理信息训练和Z总线敏感特征对降低GNN失败率至关重要，将失败率从98.72%降至13.11%。

### 结论

混合方法为实现分析求解器的经验可靠性同时利用GNN速度提供了一条实用路径，能够显著增加近实时可分析的场景数量。

### 翻译

交流潮流计算问题需要在数据驱动模型的速度和分析求解器的可靠性之间做出权衡。本文引入了一种混合框架，将图神经网络与隐式Z总线递归方法相结合，这是一种用于辐射状配电网的稳健、非迭代求解器。该框架采用物理信息图神经网络进行快速初始预测，并通过两级触发机制调用IZR求解器作为压力情况的故障保护。失败被定义为任何最大功率失配超过0.1 p.u.的解，这是显著的运行偏差。在IEEE 33节点系统具有挑战性的7,500个压力场景测试集中，纯GNN模型在13.11%的案例中失败。相比之下，混合框架识别出所有潜在的失败案例，将它们委托给IZR求解器，实现了0.00%的失败率，在该特定测试集上 empirically 匹配了分析求解器的100%成功率。扩展的消融研究证实，物理信息训练和Z总线敏感特征都很关键，共同将GNN的失败率从98.72%(仅数据)降低到13.11%。混合方法为实现分析求解器的经验可靠性同时利用GNN速度提供了一条实用路径，能够显著增加近实时可分析的场景数量。


### 论文摘要

The Alternating Current Power Flow (ACPF) problem forces a trade-off between the speed of data-driven models and the reliability of analytical solvers. This paper introduces a hybrid framework that synergizes a Graph Neural Network (GNN) with the Implicit Z-Bus Recursive (IZR) method, a robust, non-iterative solver for radial distribution networks. The framework employs a physics-informed GNN for rapid initial predictions and invokes the IZR solver as a failsafe for stressed cases identified by a two-stage trigger. A failure is defined as any solution with a maximum power mismatch exceeding 0.1 p.u., a significant operational deviation. On a challenging test set of 7,500 stressed scenarios for the IEEE 33-bus system, the GNN-only model failed on 13.11 % of cases. In contrast, the hybrid framework identified all potential failures, delegating them to the IZR solver to achieve a 0.00 % failure rate, empirically matching the 100 % success rate of the analytical solver on this specific test set. An expanded ablation study confirms that both physics-informed training and Z-bus sensitivity features are critical, collaboratively reducing the GNN's failure rate from 98.72 % (data-only) to 13.11 %. The hybrid approach demonstrates a pragmatic path to achieving the empirical reliability of an analytical solver while leveraging GNN speed, enabling a significant increase in the number of scenarios analyzable in near real-time.

---

## 73. Physics-Inspired All-Pair Interaction Learning for 3D Dynamics Modeling

**论文链接:** [http://arxiv.org/abs/2510.04233v1](http://arxiv.org/abs/2510.04233v1)

**作者:** Kai Yang, Yuqi Huang, Junheng Tao, Wanyu Wang, Qitian Wu

**发布时间:** 2025-10-05

### GPT解析

### 总结

本文提出了一种名为PAINET的新方法，用于建模多体系统中的3D动力学，通过捕捉未观察到的相互作用来提高预测性能。

### 背景

建模3D动力学是多体系统在科学和工程领域的基础问题，在轨迹预测和模拟中具有重要实际意义。现有的基于GNN的方法虽通过强制几何对称性、编码高阶特征或结合神经-ODE力学取得了强性能，但通常依赖于明确观察到的结构。

### 目的

提出PAINET，一种用于学习多体系统中所有相互作用的SE(3)-等变神经架构，以捕捉现有方法无法获取的对复杂物理行为至关重要的未观察到的相互作用。

### 方法

PAINET包含：(1)一种新颖的受物理学启发的注意力网络，源自能量函数的最小化轨迹；(2)一个并行解码器，在保持等变性的同时实现高效推理。

### 主要发现

在人体动作捕捉、分子动力学和大规模蛋白质模拟等多样化真实世界基准测试中，PAINET始终优于最近提出的模型，在3D动力学预测中实现了4.7%到41.5%的错误减少，且在时间和内存计算成本方面具有可比性。

### 结论

PAINET有效捕捉了多体系统中的未观察到的相互作用，在各种应用中表现出色，同时保持高计算效率。

### 翻译

建模3D动力学是多体系统在科学和工程领域的基础问题，在轨迹预测和模拟中具有重要的实际意义。虽然最近的基于GNN的方法通过强制几何对称性、编码高阶特征或结合神经-ODE力学已经取得了强性能，但它们通常依赖于明确观察到的结构，并且本质上无法捕捉对复杂物理行为和动力学机制至关重要的未观察到的相互作用。在本文中，我们提出了PAINET，一种用于学习多体系统中所有相互作用的SE(3)-等变神经架构。该模型包括：(1)一种源自能量函数最小化轨迹的新颖的受物理学启发的注意力网络，和(2)一个保持等变性同时实现高效推理的并行解码器。在人体动作捕捉、分子动力学和大规模蛋白质模拟等多样化的真实世界基准测试中的经验结果表明，PAINET始终优于最近提出的模型，在3D动力学预测中实现了4.7%到41.5%的错误减少，同时在时间和内存计算成本方面具有可比性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是：在3D动力学建模中，现有的基于图神经网络的方法无法捕捉粒子之间未观察到的成对相互作用，而这些相互作用对于准确模拟复杂物理系统至关重要。这个问题在科学和工程领域非常重要，因为分子动力学、天体力学、物理模拟等多个领域都需要准确建模多体系统的3D动力学。忽略未观察到的相互作用会导致长期轨迹预测不准确，并掩盖长程相关性，在结晶、蛋白质折叠等情况下，基于固定观察结构的模型可能导致系统偏差和显著的误差积累。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从物理学中获取灵感，特别是能量最小化原理，将学习未观察相互作用的问题表述为最小化能量函数的问题。他们采用多项式势能形式（Landau-Ginzburg势能形式）推导出一个基于物理的前馈层。该方法借鉴了现有的图神经网络（如EGNN）和SE(3)-等变性概念，但创新点在于专注于捕捉未观察到的成对相互作用，而不仅仅是基于观察到的结构进行消息传递。作者设计了物理启发的注意力网络和并行等变解码器，在保持物理合理性的同时提高预测准确性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过能量最小化原理学习粒子之间的所有成对相互作用，包括未观察到的相互作用。整体实现流程包括：1) 初始化：输入初始粒子位置、速度和特征，通过MLP生成初始粒子嵌入；2) 编码阶段：递归应用物理启发的注意力网络更新粒子嵌入，使用自适应成对映射捕获特定于粒子类型的依赖关系；3) 解码阶段：使用并行等变解码器（基于EGNN）从初始状态生成多个时间步的预测位置，保持SE(3)-等变性；4) 训练和推理：以监督方式训练模型，最小化预测位置和真实位置之间的均方误差，推理时先计算粒子嵌入序列，再并行生成预测位置。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 能量基础的潜在结构学习公式，为将未观察到的相互作用纳入3D动力学建模提供原则性方法；2) 物理启发的注意力网络，具有自适应成对映射，能捕获超越观察结构的长程、特定于粒子类型的依赖关系；3) 并行等变解码器，在保持SE(3)-等变性的同时实现高效推理；4) 全局相互作用建模，能够建模所有粒子对之间的相互作用。相比之前的工作，PAINET不局限于局部邻域内的相互作用，而是显式地建模所有粒子对之间的未观察到的相互作用，结合了物理原理和深度学习的优势，在保持合理计算成本的同时实现了更高的准确性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PAINET通过物理启发的能量最小化原理和自适应注意力机制，首次在3D动力学建模中有效捕捉了未观察到的全粒子对相互作用，显著提高了多体系统轨迹预测的准确性，同时保持了计算效率和物理合理性。'}


### 论文摘要

Modeling 3D dynamics is a fundamental problem in multi-body systems across scientific and engineering domains and has important practical implications in trajectory prediction and simulation. While recent GNN-based approaches have achieved strong performance by enforcing geometric symmetries, encoding high-order features or incorporating neural-ODE mechanics, they typically depend on explicitly observed structures and inherently fail to capture the unobserved interactions that are crucial to complex physical behaviors and dynamics mechanism. In this paper, we propose PAINET, a principled SE(3)-equivariant neural architecture for learning all-pair interactions in multi-body systems. The model comprises: (1) a novel physics-inspired attention network derived from the minimization trajectory of an energy function, and (2) a parallel decoder that preserves equivariance while enabling efficient inference. Empirical results on diverse real-world benchmarks, including human motion capture, molecular dynamics, and large-scale protein simulations, show that PAINET consistently outperforms recently proposed models, yielding 4.7% to 41.5% error reductions in 3D dynamics prediction with comparable computation costs in terms of time and memory.

---

## 74. VBM-NET: Visual Base Pose Learning for Mobile Manipulation using Equivariant TransporterNet and GNNs

**论文链接:** [http://arxiv.org/abs/2510.04171v1](http://arxiv.org/abs/2510.04171v1)

**作者:** Lakshadeep Naik, Adam Fischer, Daniel Duberg, Danica Kragic

**发布时间:** 2025-10-05

### GPT解析

### 总结

本文提出了一种名为VBM-NET的学习方法，用于从场景的顶部正交投影中选择最优的移动基座姿态，实现高效物体抓取。该方法结合了等变性TransporterNet、图神经网络和强化学习，无需精确的物体姿态和环境模型信息。

### 背景

在移动操作领域，选择最优的移动基座姿态对成功物体抓取至关重要。现有方法依赖经典规划或基于状态的策略学习，但都需要可靠的状态信息，如精确的物体姿态和环境模型。

### 目的

研究直接从场景的顶部正交投影进行基座姿态规划，提供场景全局视角同时保留空间结构，并提出一种基于学习的方法VBM-NET用于基座姿态选择。

### 方法

提出VBM-NET方法，使用等变性TransporterNet利用空间对称性高效学习候选基座姿态，采用图神经网络表示可变数量的候选姿态，并通过强化学习确定最优基座姿态。

### 主要发现

VBM-NET在显著更少的计算时间内能产生与经典方法相当的结果；成功实现了从模拟到真实世界的策略迁移，验证了方法的有效性。

### 结论

VBM-NET提供了一种高效的基座姿态选择方法，直接从场景的顶部正交投影中学习，无需精确的状态信息，同时保持与经典方法相当的性能但计算效率更高。

### 翻译

在移动操作中，选择最优的移动基座姿态对于成功的物体抓取至关重要。先前的工作要么通过经典规划方法，要么通过学习基于状态的策略来解决这一问题。他们假设可以访问可靠的状态信息，如精确的物体姿态和环境模型。在这项工作中，我们直接从场景的顶部正交投影研究基座姿态规划，这种方法提供场景的全局视角，同时保留空间结构。我们提出了VBM-NET，一种使用这种顶部正交投影进行基座姿态选择的基于学习的方法。我们使用等变性TransporterNet利用空间对称性，并高效学习用于抓取的候选基座姿态。此外，我们使用图神经网络表示可变数量的候选基座姿态，并使用强化学习来确定它们之间的最优基座姿态。我们证明VBM-NET能够在显著更少的计算时间内产生与经典方法相当的结果。此外，我们通过将在模拟中训练的策略成功部署到真实世界的移动操作中，验证了模拟到真实的迁移学习。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决移动操作中机器人基座姿态选择问题，即在机器人需要移动到合适位置才能抓取物体时，如何确定最优的基座位置和方向。这个问题在现实中很重要，因为传统方法依赖精确的物体姿态估计和环境模型，而这些信息在实际应用中往往难以获取，尤其是在机器人距离物体较远时。有效的基座姿态规划能显著提高移动操作任务的效率和成功率，让机器人在没有精确状态信息的情况下也能完成抓取任务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统方法和现有学习方法都依赖于精确状态估计的局限性，然后思考如何仅使用视觉输入来解决这个问题。他们选择使用正交俯视投影作为视觉表示，因为它能提供场景全局概览且保持空间结构。作者借鉴了等变性神经网络（ENN）中的TransporterNet来利用空间对称性，使用图神经网络（GNN）处理可变数量的候选基座姿态，并采用强化学习来确定最优姿态。整体设计是一个两阶段方法：先学习可能的基座姿态，再选择最优的一个。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用正交俯视投影的场景表示来学习基座姿态选择策略，不依赖精确状态估计，利用等变性神经网络处理空间对称性，使用图神经网络处理可变数量的候选基座姿态。整体流程分为两阶段：第一阶段使用等变性TransporterNet从场景正交投影中学习可能的基座姿态，确保抓取可行性和无碰撞；第二阶段使用图神经网络和强化学习从候选姿态中选择最优的一个，考虑导航成本。最后机器人导航到选定基座姿态并进行抓取操作。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：首次从视觉表示中学习基座姿态规划而不依赖精确状态估计；提出VBM-NET两阶段学习方法；使用等变性TransporterNet提高样本效率；使用图神经网络处理可变数量的候选姿态；成功实现模拟到现实迁移。相比之前工作，不同之处在于：传统方法依赖精确状态估计，而VBM-NET仅使用视觉输入；现有基于视觉的方法通常使用机器人自带的相机缺乏全局视野，而VBM-NET使用正交俯视投影；现有方法通常不考虑导航成本，而VBM-NET明确考虑了这一点。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'VBM-NET首次提出了一种仅使用正交俯视投影而不依赖精确状态估计的学习方法，通过结合等变性TransporterNet和图神经网络，实现了移动操作中考虑导航成本的最优基座姿态规划，并在模拟到现实中成功验证了其有效性。'}


### 论文摘要

In Mobile Manipulation, selecting an optimal mobile base pose is essential for successful object grasping. Previous works have addressed this problem either through classical planning methods or by learning state-based policies. They assume access to reliable state information, such as the precise object poses and environment models. In this work, we study base pose planning directly from top-down orthographic projections of the scene, which provide a global overview of the scene while preserving spatial structure. We propose VBM-NET, a learning-based method for base pose selection using such top-down orthographic projections. We use equivariant TransporterNet to exploit spatial symmetries and efficiently learn candidate base poses for grasping. Further, we use graph neural networks to represent a varying number of candidate base poses and use Reinforcement Learning to determine the optimal base pose among them. We show that VBM-NET can produce comparable solutions to the classical methods in significantly less computation time. Furthermore, we validate sim-to-real transfer by successfully deploying a policy trained in simulation to real-world mobile manipulation.

---

## 75. ICEPool: Enhancing Graph Pooling Networks with Inter-cluster Connectivity

**论文链接:** [http://arxiv.org/abs/2510.03987v1](http://arxiv.org/abs/2510.03987v1)

**作者:** Michael Yang

**发布时间:** 2025-10-05

### GPT解析

### 总结

这篇论文提出了ICEPool，一种新的分层池化框架，用于增强模型对簇间连通性的理解，并保留原始图的结构完整性。ICEPool可与多种基于池化的图神经网络模型兼容，通过结合原有模型的优点和ICEPool强调簇间连通集成的能力，生成更全面和鲁棒的图级表示。

### 背景

分层池化模型在分类图结构数据方面表现出色。虽然已经提出了许多创新方法来设计簇分配和粗化策略，但簇之间的关系往往被忽视。

### 目的

引入ICEPool，一种新型分层池化框架，旨在增强模型对簇间连通性的理解，并保留原始图的结构完整性，解决传统模型忽视簇间关系的问题。

### 方法

提出ICEPool（Inter-cluster Connectivity Enhancement Pooling），一种分层池化框架，强调簇间连通性的集成，可与多种基于池化的GNN模型兼容。通过理论分析证明ICEPool的图重构能力，展示其在学习传统模型忽视的簇间关系方面的有效性。

### 主要发现

ICEPool能够与多种模型兼容，并有望提升现有图神经网络架构的性能。理论分析证明ICEPool在图重构方面的能力，展示了其在学习簇间关系方面的有效性。

### 结论

ICEPool作为对现有模型的增强，有效结合了原始模型的优点和ICEPool强调簇间连通集成的能力，生成更全面和鲁棒的图级表示。实验结果表明ICEPool具有广泛的模型兼容性，并能提升现有图神经网络架构的性能。

### 翻译

分层池化模型在分类图结构数据方面表现出色。虽然已经提出了许多创新方法来设计簇分配和粗化策略，但簇之间的关系往往被忽视。在本文中，我们介绍了ICEPool（Inter-cluster Connectivity Enhancement Pooling），一种新型分层池化框架，旨在增强模型对簇间连通性的理解以及保留原始图结构完整性的能力。ICEPool可与多种基于池化的GNN模型兼容。将ICEPool作为现有模型的增强，有效地结合了原始模型的优点和ICEPool强调簇间连通集成的能力，从而生成更全面和鲁棒的图级表示。此外，我们对ICEPool的图重构能力进行了理论分析，证明其在学习传统模型忽视的簇间关系方面的有效性。最后，实验结果表明了ICEPool与各种模型的兼容性及其提升现有图神经网络架构性能的潜力。


### 论文摘要

Hierarchical Pooling Models have demonstrated strong performance in classifying graph-structured data. While numerous innovative methods have been proposed to design cluster assignments and coarsening strategies, the relationships between clusters are often overlooked. In this paper, we introduce Inter-cluster Connectivity Enhancement Pooling (ICEPool), a novel hierarchical pooling framework designed to enhance model's understanding of inter-cluster connectivity and ability of preserving the structural integrity in the original graph. ICEPool is compatible with a wide range of pooling-based GNN models. The deployment of ICEPool as an enhancement to existing models effectively combines the strengths of the original model with ICEPool's capability to emphasize the integration of inter-cluster connectivity, resulting in a more comprehensive and robust graph-level representation. Moreover, we make theoretical analysis to ICEPool's ability of graph reconstruction to demonstrate its effectiveness in learning inter-cluster relationship that is overlooked by conventional models. Finally, the experimental results show the compatibility of ICEPool with wide varieties of models and its potential to boost the performance of existing graph neural network architectures.

---

## 76. On the Convergence and Size Transferability of Continuous-depth Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.03923v1](http://arxiv.org/abs/2510.03923v1)

**作者:** Mingsong Yan, Charles Kulick, Sui Tang

**发布时间:** 2025-10-04

### GPT解析

### 总结

本文研究了连续深度图神经网络(GNDEs)的收敛性分析，引入了Graphon神经微分方程(Graphon-NDEs)作为GNDEs的无限节点极限，证明了GNDE解到Graphon-NDE解的轨迹收敛性。研究推导了两种确定性图采样方案下的显式收敛率，并建立了大小转移界限，为GNDE模型在不同大小图之间的迁移提供了理论支持。

### 背景

连续深度图神经网络(GNDEs)结合了图神经网络(GNNs)的结构归纳偏置和神经ODEs的连续深度架构，为在图上建模动力学提供了可扩展且原则性的框架。

### 目的

对具有时变参数的GNDEs在无限节点极限情况下进行严格的收敛性分析，提供关于它们大小可转移性的理论见解。

### 方法

引入Graphon神经微分方程(Graphon-NDEs)作为GNDEs的无限节点极限并建立其适定性；利用图论理论和动力系统工具证明GNDE解到Graphon-NDE解的轨迹收敛；推导两种确定性图采样方案下的显式收敛率；建立大小转移界限。

### 主要发现

GNDEs在无限节点极限情况下收敛到Graphon-NDEs；在从平滑图子中采样的加权图和从{0,1}-值图子中采样的无权图两种方案下具有明确的收敛率；GNDE模型具有大小可转移性，可以将在中等规模图上训练的模型应用到更大图上而无需重新训练。

### 结论

数值实验使用合成和真实数据支持了理论发现，验证了GNDEs的收敛性和大小可转移性。

### 翻译

连续深度图神经网络，也称为图神经微分方程(GNDEs)，结合了图神经网络(GNNs)的结构归纳偏置和神经ODEs的连续深度架构，为在图上建模动力学提供了一个可扩展且原则性的框架。在本文中，我们提出了对具有时变参数的GNDEs在无限节点极限情况下的严格收敛性分析，为它们的大小可转移性提供了理论见解。为此，我们引入了图子神经微分方程(Graphon-NDEs)作为GNDEs的无限节点极限，并建立了它们的适定性。利用图子理论和动力系统工具，我们证明了GNDE解到Graphon-NDE解的轨迹收敛性。此外，我们在两种确定性图采样方案下推导了显式收敛率：(1)从平滑图子中采样的加权图，和(2)从{0,1}-值(不连续)图子中采样的无权图。我们进一步建立了大小转移界限，为将训练在中等规模图上的GNDE模型转移到更大、结构相似的图而无需重新训练的实际策略提供了理论依据。使用合成和真实数据的数值实验支持了我们的理论发现。


### 论文摘要

Continuous-depth graph neural networks, also known as Graph Neural Differential Equations (GNDEs), combine the structural inductive bias of Graph Neural Networks (GNNs) with the continuous-depth architecture of Neural ODEs, offering a scalable and principled framework for modeling dynamics on graphs. In this paper, we present a rigorous convergence analysis of GNDEs with time-varying parameters in the infinite-node limit, providing theoretical insights into their size transferability. To this end, we introduce Graphon Neural Differential Equations (Graphon-NDEs) as the infinite-node limit of GNDEs and establish their well-posedness. Leveraging tools from graphon theory and dynamical systems, we prove the trajectory-wise convergence of GNDE solutions to Graphon-NDE solutions. Moreover, we derive explicit convergence rates under two deterministic graph sampling regimes: (1) weighted graphs sampled from smooth graphons, and (2) unweighted graphs sampled from $\{0,1\}$-valued (discontinuous) graphons. We further establish size transferability bounds, providing theoretical justification for the practical strategy of transferring GNDE models trained on moderate-sized graphs to larger, structurally similar graphs without retraining. Numerical experiments using synthetic and real data support our theoretical findings.

---

## 77. Interpretable Neuropsychiatric Diagnosis via Concept-Guided Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.03351v1](http://arxiv.org/abs/2510.03351v1)

**作者:** Song Wang, Zhenyu Lei, Zhen Tan, Jundong Li, Javier Rasero, Aiying Zhang, Chirag Agarwal

**发布时间:** 2025-10-02

### GPT解析

### 总结

本文提出了一种名为CONCEPTNEURO的新型精神疾病诊断框架，通过结合大型语言模型和神经生物学领域知识，实现了高准确性和临床可解释性的精神疾病诊断。

### 背景

近五分之一的青少年被诊断患有精神或行为健康问题，如焦虑、抑郁或行为障碍，凸显了开发准确且可解释诊断工具的紧迫性。静息态功能磁共振成像(rs-fMRI)提供了一种研究大规模功能连接的有力工具，但现有的图神经网络(GNN)方法仍存在黑盒问题，限制了其可靠性和临床应用。

### 目的

开发一个基于概念的诊断框架，解决现有GNN方法作为黑盒模型的问题，提高精神疾病诊断的可靠性和临床转化能力，同时保持高预测性能。

### 方法

提出CONCEPTNEURO框架，利用大型语言模型和神经生物学领域知识自动生成、过滤和编码可解释的功能连接概念。每个概念被表示为连接特定脑区的结构化子图，然后通过概念分类器进行处理，使预测基于临床上有意义的连接模式。

### 主要发现

CONCEPTNEURO增强的GNN在多个精神疾病数据集上表现优于传统GNN，提高了诊断准确性，同时提供了透明且符合临床的解释。概念分析揭示了与专业知识一致且与特定疾病相关的连接模式，为未来研究提出了新假设。

### 结论

CONCEPTNEURO是一个可解释、领域知情的精神疾病诊断框架，成功结合了高性能预测和临床可解释性，有望推动精神疾病诊断工具的临床应用。

### 翻译

目前约五分之一的青少年被诊断患有精神或行为健康问题，如焦虑、抑郁或行为障碍，这凸显了开发准确且可解释诊断工具的紧迫性。静息态功能磁共振成像(rs-fMRI)为大规模功能连接提供了有力的视角，其中脑区被建模为节点，区域间同步性被建模为边，为精神障碍提供了临床相关的生物标志物。虽然先前的工作使用图神经网络(GNN)方法进行疾病预测，但它们仍然是复杂的黑盒，限制了其可靠性和临床转化。在这项工作中，我们提出了CONCEPTNEURO，一个基于概念的诊断框架，利用大型语言模型(LLMs)和神经生物学领域知识自动生成、过滤和编码可解释的功能连接概念。每个概念表示为连接特定脑区的结构化子图，然后通过概念分类器。我们的设计确保通过临床上有意义的连接模式进行预测，同时实现可解释性和强大的预测性能。在多个精神疾病数据集上的广泛实验表明，CONCEPTNEURO增强的GNN始终优于其传统版本，提高了准确性，同时提供透明、临床一致的解释。此外，概念分析突显了与专业知识一致的疾病特异性连接模式，并为未来研究提出了新假设，确立了CONCEPTNEURO作为精神疾病诊断的一种可解释、领域知情框架。


### 论文摘要

Nearly one in five adolescents currently live with a diagnosed mental or behavioral health condition, such as anxiety, depression, or conduct disorder, underscoring the urgency of developing accurate and interpretable diagnostic tools. Resting-state functional magnetic resonance imaging (rs-fMRI) provides a powerful lens into large-scale functional connectivity, where brain regions are modeled as nodes and inter-regional synchrony as edges, offering clinically relevant biomarkers for psychiatric disorders. While prior works use graph neural network (GNN) approaches for disorder prediction, they remain complex black-boxes, limiting their reliability and clinical translation. In this work, we propose CONCEPTNEURO, a concept-based diagnosis framework that leverages large language models (LLMs) and neurobiological domain knowledge to automatically generate, filter, and encode interpretable functional connectivity concepts. Each concept is represented as a structured subgraph linking specific brain regions, which are then passed through a concept classifier. Our design ensures predictions through clinically meaningful connectivity patterns, enabling both interpretability and strong predictive performance. Extensive experiments across multiple psychiatric disorder datasets demonstrate that CONCEPTNEURO-augmented GNNs consistently outperform their vanilla counterparts, improving accuracy while providing transparent, clinically aligned explanations. Furthermore, concept analyses highlight disorder-specific connectivity patterns that align with expert knowledge and suggest new hypotheses for future investigation, establishing CONCEPTNEURO as an interpretable, domain-informed framework for psychiatric disorder diagnosis.

---

## 78. Comparing fine-tuning strategies of MACE machine learning force field for modeling Li-ion diffusion in LiF for batteries

**论文链接:** [http://arxiv.org/abs/2510.05020v1](http://arxiv.org/abs/2510.05020v1)

**作者:** Nada Alghamdi, Paolo de Angelis, Pietro Asinari, Eliodoro Chiavazzo

**发布时间:** 2025-10-06

**备注:** 13 pages, 5 figures

### GPT解析

### 总结

该研究比较了MACE机器学习模型与DeePMD势能在预测LiF中锂扩散率方面的性能，发现MACE模型在少量或无需训练数据的情况下能达到与DeePMD相当的准确性。

### 背景

机器学习原子间势能(MLIPs)正在改变材料科学和工程，使研究电池操作等复杂现象成为可能。

### 目的

对MACE机器学习模型与训练良好的DeePMD势能进行基准测试，用于预测LiF中的间隙锂扩散率，LiF是锂离子电池固体电解质界面的关键组成部分。

### 方法

通过分子动力学模拟来预测关键扩散特性，比较MACE模型与DeePMD模型的预测结果。

### 主要发现

MACE-MPA-0基础模型预测活化能为0.22 eV，仅用300个数据点微调的模型预测为0.20 eV，两者都与DeePMD模型的参考值0.24 eV接近；微调方法只需少量数据即可达到与DeePMD相当的性能。

### 结论

微调方法只需要一小部分训练数据就能达到与超过40,000个主动学习数据训练的DeePMD势能相当的稳健性能。

### 翻译

机器学习原子间势能(MLIPs)正在通过使研究复杂现象（如电池操作中至关重要的现象）而改变材料科学和工程。在这项工作中，我们将MACE机器学习模型与训练良好的DeePMD势能进行基准测试，用于预测LiF中的间隙锂扩散率，LiF是锂离子电池中固体电解质界面的关键组成部分。我们的结果表明，MACE-MPA-0基础模型在基于分子动力学模拟预测关键扩散特性方面，与训练良好的DeePMD模型具有相当的准确性，同时需要最少或无需训练数据。例如，MACE-MPA-0预测活化能Ea为0.22 eV，仅用300个数据点微调的模型预测Ea = 0.20 eV，两者都与DeePMD模型的参考值Ea = 0.24 eV显示出良好的一致性。在这项工作中，我们提供了一个坚实的测试案例，其中微调方法-无论是使用为DeePMD生成的数据还是由基础MACE模型本身产生的数据-都能产生与DeePMD势能相当的稳健性能，后者需要超过40,000个主动学习的训练数据。


### 论文摘要

Machine learning interatomic potentials (MLIPs) are transforming materials science and engineering by enabling the study of complex phenomena, such as those critical to battery operation. In this work, we benchmark the MACE machine learning model against a well-trained DeePMD potential for predicting interstitial lithium diffusivity in LiF, a key component in the solid electrolyte interphase in Li ion batteries. Our results demonstrate that the MACE-MPA-0 foundational model achieves comparable accuracy to well-trained DeePMD, in predicting key diffusion properties based on molecular dynamics simulation, while requiring minimal or no training data. For instance, the MACE-MPA-0 predicts an activation energy Ea of 0.22 eV, the fine-tuned model with only 300 data points predicts Ea = 0.20 eV, both of which show good agreement with the DeePMD model reference value of Ea = 0.24 eV. In this work, we provide a solid test case where fine-tuning approaches - whether using data generated for DeePMD or data produced by the foundational MACE model itself - yield similar robust performance to the DeePMD potential trained with over 40,000 actively learned data, albeit requiring only a fraction of the training data.

---

## 79. ActiveMark: on watermarking of visual foundation models via massive activations

**论文链接:** [http://arxiv.org/abs/2510.04966v1](http://arxiv.org/abs/2510.04966v1)

**作者:** Anna Chistyakova, Mikhail Pautov

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出了一种视觉基础模型的所有权验证方法，通过微调模型的一小组表达性层和小型编码器-解码器网络，将数字水印嵌入到输入图像的内部表示中，使水印在模型的功能副本中仍可检测。

### 背景

视觉基础模型在大数据集上训练后，可通过微调适应各种下游任务，在计算机视觉应用中表现出色。由于数据收集和训练计算成本高，模型所有者通常分发模型并附带许可证以保护知识产权。

### 目的

开发可靠的所有权验证工具，以区分重新分发的模型副本和独立开发的模型。

### 方法

微调视觉基础模型的一小组表达性层，同时使用一个小型编码器-解码器网络，将数字水印嵌入到保留集输入图像的内部表示中。

### 主要发现

嵌入的水印在受保护模型的功能副本中仍然可检测，例如通过微调VFM用于特定下游任务时。

### 结论

理论和实验表明，所提出的方法对未加水印模型的错误检测概率低，对加水印模型的错误漏检概率也低。

### 翻译

在大规模数据集上训练的视觉基础模型可以通过微调适应各种下游任务，在各种计算机视觉应用中实现卓越的性能和效率。数据收集和训练的高计算成本促使一些视觉基础模型的所有者分发这些模型并附带许可证，以保护其知识产权。然而，受保护模型副本的不诚实用户可能会非法重新分发它，例如以获利为目的。因此，今天开发可靠的所有权验证工具非常重要，因为这些方法可用于区分受保护模型的重新分发副本和独立模型。在本文中，我们提出了一种通过微调视觉基础模型的一小组表达性层以及小型编码器-解码器网络来对所有权进行验证的方法，将数字水印嵌入到保留集输入图像的内部表示中。重要的是，嵌入的水印在受保护模型的功能副本中仍然可检测，例如通过将视觉基础模型微调用于特定下游任务时。我们在理论和实验上证明，所提出的方法对未加水印模型的错误检测概率低，对加水印模型的错误漏检概率也低。


### 论文摘要

Being trained on large and vast datasets, visual foundation models (VFMs) can be fine-tuned for diverse downstream tasks, achieving remarkable performance and efficiency in various computer vision applications. The high computation cost of data collection and training motivates the owners of some VFMs to distribute them alongside the license to protect their intellectual property rights. However, a dishonest user of the protected model's copy may illegally redistribute it, for example, to make a profit. As a consequence, the development of reliable ownership verification tools is of great importance today, since such methods can be used to differentiate between a redistributed copy of the protected model and an independent model. In this paper, we propose an approach to ownership verification of visual foundation models by fine-tuning a small set of expressive layers of a VFM along with a small encoder-decoder network to embed digital watermarks into an internal representation of a hold-out set of input images. Importantly, the watermarks embedded remain detectable in the functional copies of the protected model, obtained, for example, by fine-tuning the VFM for a particular downstream task. Theoretically and experimentally, we demonstrate that the proposed method yields a low probability of false detection of a non-watermarked model and a low probability of false misdetection of a watermarked model.

---

## 80. HyperVLA: Efficient Inference in Vision-Language-Action Models via Hypernetworks

**论文链接:** [http://arxiv.org/abs/2510.04898v1](http://arxiv.org/abs/2510.04898v1)

**作者:** Zheng Xiong, Kang Li, Zilin Wang, Matthew Jackson, Jakob Foerster, Shimon Whiteson

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文提出了一种名为HyperVLA的新型Vision-Language-Action模型，通过基于超网络的架构显著降低推理成本，同时保持或提高性能。

### 背景

Vision-Language-Action模型基于具有强大泛化能力的语言和视觉基础模型，在大规模机器人数据上训练，已成为学习通用机器人策略的有前景方法。但现有VLA的主要缺点是极高的推理成本。

### 目的

解决现有VLA模型推理成本过高的问题，设计一种能在推理阶段减少计算量同时保持高性能的模型架构。

### 方法

提出HyperVLA，采用基于超网络的架构，在推理阶段只激活小型特定任务策略，同时保留训练阶段的高模型容量。包含利用视觉基础模型先验知识、超网络归一化和动作生成策略等关键算法设计特征。

### 主要发现

与整体式VLA相比，HyperVLA在零样本泛化和少样本适应方面达到相似或更高成功率，同时显著降低推理成本。与OpenVLA相比，HyperVLA测试时激活参数数量减少90倍，推理速度提高120倍。

### 结论

HyperVLA通过创新的超网络架构成功解决了现有VLA模型推理成本过高的问题，在保持或提高性能的同时大幅降低了计算需求。

### 翻译

基于具有强大泛化能力的语言和视觉基础模型，并在大规模机器人数据上训练，Vision-Language-Action模型最近已成为学习通用机器人策略的有前景的方法。然而，现有VLA的一个主要缺点是极高的推理成本。在本文中，我们提出HyperVLA来解决这个问题。与在训练和推理过程中激活整个模型的整体式VLA不同，HyperVLA使用一种新颖的基于超网络的架构，在推理过程中只激活小型的特定任务策略，同时在训练阶段仍保留容纳多样化多任务行为所需的高模型容量。代码已在https://github.com/MasterXiong/HyperVLA公开。


### 论文摘要

Built upon language and vision foundation models with strong generalization ability and trained on large-scale robotic data, Vision-Language-Action (VLA) models have recently emerged as a promising approach to learning generalist robotic policies. However, a key drawback of existing VLAs is their extremely high inference costs. In this paper, we propose HyperVLA to address this problem. Unlike existing monolithic VLAs that activate the whole model during both training and inference, HyperVLA uses a novel hypernetwork (HN)-based architecture that activates only a small task-specific policy during inference, while still retaining the high model capacity needed to accommodate diverse multi-task behaviors during training. Successfully training an HN-based VLA is nontrivial so HyperVLA contains several key algorithm design features that improve its performance, including properly utilizing the prior knowledge from existing vision foundation models, HN normalization, and an action generation strategy. Compared to monolithic VLAs, HyperVLA achieves a similar or even higher success rate for both zero-shot generalization and few-shot adaptation, while significantly reducing inference costs. Compared to OpenVLA, a state-of-the-art VLA model, HyperVLA reduces the number of activated parameters at test time by $90\times$, and accelerates inference speed by $120\times$. Code is publicly available at https://github.com/MasterXiong/HyperVLA

---

## 81. A Clinical-grade Universal Foundation Model for Intraoperative Pathology

**论文链接:** [http://arxiv.org/abs/2510.04861v1](http://arxiv.org/abs/2510.04861v1)

**作者:** Zihan Zhao, Fengtao Zhou, Ronggang Li, Bing Chu, Xinke Zhang, Xueyi Zheng, Ke Zheng, Xiaobo Wen, Jiabo Ma, Yihui Wang, Jiewei Chen, Chengyou Zheng, Jiangyu Zhang, Yongqin Wen, Jiajia Meng, Ziqi Zeng, Xiaoqing Li, Jing Li, Dan Xie, Yaping Ye, Yu Wang, Hao Chen, Muyan Cai

**发布时间:** 2025-10-06

### GPT解析

### 总结

CRISP是一个临床级基础模型，基于八个医疗中心的超过10万张冰冻切片开发，旨在为术中病理学提供临床级稳健支持，在真实临床环境中表现出高诊断准确性并能辅助医生减少工作量。

### 背景

术中病理学对精准手术至关重要，但其临床应用受到诊断复杂性和高质量冰冻切片数据有限性的制约。尽管计算病理学取得显著进展，但缺乏大规模前瞻性验证阻碍了其在手术工作流程中的常规应用。

### 目的

开发一个名为CRISP的临床级基础模型，为病理学提供临床级稳健的术中支持，解决术中病理诊断的挑战。

### 方法

在八个医疗中心的超过10万张冰冻切片上开发CRISP模型，并在近100个回顾性诊断任务上对超过15,000张术中切片进行全面评估，包括良恶性鉴别、关键术中决策和泛癌检测等。随后在超过2,000名患者的前瞻性队列中进行验证。

### 主要发现

CRISP模型在不同机构、肿瘤类型和解剖部位（包括未见过的部位和罕见癌症）上表现出强大的泛化能力；在真实世界条件下保持高诊断准确性，在92.6%的病例中直接指导手术决策；人机协作将诊断工作量减少35%，避免105项辅助检查，并以87.5%的准确性提高微转移检测率。

### 结论

CRISP定位为AI驱动的术中病理学的临床级范式，弥合了计算进步与手术精确性之间的差距，加速了人工智能向常规临床实践的转化。

### 翻译

术中病理学对精准手术至关重要，但其临床影响受到诊断复杂性和高质量冰冻切片数据有限性的制约。虽然计算病理学已取得显著进展，但缺乏大规模的前瞻性验证阻碍了其在手术工作流程中的常规应用。在此，我们介绍CRISP，一个在八个医疗中心超过10万张冰冻切片上开发临床级基础模型，专门设计用于为病理学提供临床级稳健的术中支持。CRISP在近100个回顾性诊断任务上对超过15,000张术中切片进行了全面评估，包括良恶性鉴别、关键术中决策和泛癌检测等。该模型在不同机构、肿瘤类型和解剖部位（包括未见过的部位和罕见癌症）上表现出强大的泛化能力。在超过2,000名患者的前瞻性队列中，CRISP在真实世界条件下保持高诊断准确性，在92.6%的病例中直接指导手术决策。人机协作进一步将诊断工作量减少35%，避免了105项辅助检查，并以87.5%的准确性提高了微转移的检测率。总之，这些发现将CRISP定位为AI驱动的术中病理学的临床级范式，弥合了计算进步与手术精确性之间的差距，加速了人工智能向常规临床实践的转化。


### 论文摘要

Intraoperative pathology is pivotal to precision surgery, yet its clinical impact is constrained by diagnostic complexity and the limited availability of high-quality frozen-section data. While computational pathology has made significant strides, the lack of large-scale, prospective validation has impeded its routine adoption in surgical workflows. Here, we introduce CRISP, a clinical-grade foundation model developed on over 100,000 frozen sections from eight medical centers, specifically designed to provide Clinical-grade Robust Intraoperative Support for Pathology (CRISP). CRISP was comprehensively evaluated on more than 15,000 intraoperative slides across nearly 100 retrospective diagnostic tasks, including benign-malignant discrimination, key intraoperative decision-making, and pan-cancer detection, etc. The model demonstrated robust generalization across diverse institutions, tumor types, and anatomical sites-including previously unseen sites and rare cancers. In a prospective cohort of over 2,000 patients, CRISP sustained high diagnostic accuracy under real-world conditions, directly informing surgical decisions in 92.6% of cases. Human-AI collaboration further reduced diagnostic workload by 35%, avoided 105 ancillary tests and enhanced detection of micrometastases with 87.5% accuracy. Together, these findings position CRISP as a clinical-grade paradigm for AI-driven intraoperative pathology, bridging computational advances with surgical precision and accelerating the translation of artificial intelligence into routine clinical practice.

---

## 82. Federated Learning for Surgical Vision in Appendicitis Classification: Results of the FedSurg EndoVis 2024 Challenge

**论文链接:** [http://arxiv.org/abs/2510.04772v1](http://arxiv.org/abs/2510.04772v1)

**作者:** Max Kirchner, Hanna Hoffmann, Alexander C. Jenke, Oliver L. Saldanha, Kevin Pfeiffer, Weam Kanjo, Julia Alekseenko, Claas de Boer, Santhi Raj Kolamuri, Lorenzo Mazza, Nicolas Padoy, Sophia Bano, Annika Reinke, Lena Maier-Hein, Danail Stoyanov, Jakob N. Kather, Fiona R. Kolbinger, Sebastian Bodenstedt, Stefanie Speidel

**发布时间:** 2025-10-06

**备注:** A challenge report pre-print (31 pages), including 7 tables and 8  figures

### GPT解析

### 总结

FedSurg挑战是首个针对手术视频分类中联邦学习策略评估的基准测试，评估了模型在未见临床中心的泛化能力和本地微调适应能力，发现ViViT模型表现最佳，但存在泛化有限、类别不平衡敏感性高、超参数调优困难等问题，强调了架构选择、预处理和损失设计的重要性。

### 背景

联邦学习在医疗领域特别是手术视频分析中的应用面临独特挑战，需要在保护患者隐私的同时实现多中心协作模型开发。

### 目的

评估当前联邦学习方法在未见临床中心的泛化能力；评估通过本地微调进行中心特定适应的能力；在不共享患者数据的情况下促进协作模型开发；建立手术视频分类中联邦学习策略的评估基准。

### 方法

使用多中心Appendix300视频数据集的初步版本；参与者开发策略来分类阑尾炎的炎症阶段；评估两个任务：未见中心的泛化和微调后的中心特定适应；提交的方法包括基础模型与线性探测、带三元组损失的度量学习、各种FL聚合方案（FedAvg、FedMedian、FedSAM）；使用F1分数和预期成本评估性能，通过bootstrap和统计测试评估排名稳健性。

### 主要发现

在泛化任务中，跨中心的性能有限；在适应任务中，所有团队在微调后都有所改善，但排名稳定性低；ViViT-based提交实现了最强的整体性能；挑战突显了泛化限制、对类别不平衡的敏感性以及去中心化训练中超参数调优的困难；时空建模和上下文感知预处理作为有前途的策略出现。

### 结论

FedSurg挑战建立了评估手术视频分类中FL策略的首个基准；研究结果突显了本地个性化与全局稳健性之间的权衡；强调了架构选择、预处理和损失设计的重要性；该基准为未来在临床手术AI中开发不平衡感知、自适应和稳健的FL方法提供了参考点。

### 翻译

目的：FedSurg挑战旨在评估联邦学习在手术视频分类领域的最新技术水平。其目标是评估当前方法在未见过的临床中心的泛化能力，并通过本地微调进行适应，同时在不共享患者数据的情况下实现协作模型开发。方法：参与者使用多中心Appendix300视频数据集的初步版本开发策略，以分类阑尾炎的炎症阶段。挑战评估了两个任务：对未见中心的泛化和微调后的中心特定适应。提交的方法包括带有线性探测的基础模型、带三元组损失的度量学习以及各种FL聚合方案（FedAvg、FedMedian、FedSAM）。使用F1分数和预期成本评估性能，并通过bootstrap和统计测试评估排名稳健性。结果：在泛化任务中，跨中心的性能有限。在适应任务中，所有团队在微调后都有所改善，尽管排名稳定性低。基于ViViT的提交实现了最强的整体性能。该挑战突显了泛化限制、对类别不平衡的敏感性以及去中心化训练中超参数调优的困难，而时空建模和上下文感知预处理作为有前途的策略出现。结论：FedSurg挑战建立了评估手术视频分类中FL策略的首个基准。研究结果突显了本地个性化与全局稳健性之间的权衡，并强调了架构选择、预处理和损失设计的重要性。这一基准为未来在临床手术AI中开发不平衡感知、自适应和稳健的FL方法提供了参考点。


### 论文摘要

Purpose: The FedSurg challenge was designed to benchmark the state of the art in federated learning for surgical video classification. Its goal was to assess how well current methods generalize to unseen clinical centers and adapt through local fine-tuning while enabling collaborative model development without sharing patient data. Methods: Participants developed strategies to classify inflammation stages in appendicitis using a preliminary version of the multi-center Appendix300 video dataset. The challenge evaluated two tasks: generalization to an unseen center and center-specific adaptation after fine-tuning. Submitted approaches included foundation models with linear probing, metric learning with triplet loss, and various FL aggregation schemes (FedAvg, FedMedian, FedSAM). Performance was assessed using F1-score and Expected Cost, with ranking robustness evaluated via bootstrapping and statistical testing. Results: In the generalization task, performance across centers was limited. In the adaptation task, all teams improved after fine-tuning, though ranking stability was low. The ViViT-based submission achieved the strongest overall performance. The challenge highlighted limitations in generalization, sensitivity to class imbalance, and difficulties in hyperparameter tuning in decentralized training, while spatiotemporal modeling and context-aware preprocessing emerged as promising strategies. Conclusion: The FedSurg Challenge establishes the first benchmark for evaluating FL strategies in surgical video classification. Findings highlight the trade-off between local personalization and global robustness, and underscore the importance of architecture choice, preprocessing, and loss design. This benchmarking offers a reference point for future development of imbalance-aware, adaptive, and robust FL methods in clinical surgical AI.

---

## 83. ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion

**论文链接:** [http://arxiv.org/abs/2510.04706v1](http://arxiv.org/abs/2510.04706v1)

**作者:** Foivos Paraperas Papantoniou, Stefanos Zafeiriou

**发布时间:** 2025-10-06

**备注:** ICCVW 2025, Code: https://github.com/foivospar/Arc2Face

### GPT解析

### 总结

本研究提出了一种基于扩散的框架，能够在不损害身份一致性的情况下实现细粒度的面部表情控制，适用于AI驱动的故事讲述。

### 背景

面向AI驱动故事讲述的人本生成模型需要两个核心能力：身份一致性和对人类表演的精确控制。尽管基于扩散的方法在保持面部身份方面取得了进展，但在不损害身份的情况下实现细粒度表情控制仍然具有挑战性。

### 目的

开发一个能够忠实地在任意特定面部表情下重新构想任何主体的扩散框架。

### 方法

基于ID一致的面部基础模型，采用组合设计，包含由FLAME blendshape参数引导的表情交叉注意力模块用于显式控制。在多样化的图像和视频数据上进行训练，能够泛化到基本情感之外的微妙微表情和表情转换。此外，还提供了可插拔的参考适配器用于真实图像的表情编辑。

### 主要发现

大量的定量和定性评估表明，该模型在定制化和身份一致的表情生成方面优于现有方法。

### 结论

所提出的框架成功结合了身份一致性和细粒度表情控制，为AI驱动的故事讲述提供了有效工具。

### 翻译

面向AI驱动故事讲述的人本生成模型必须结合两个核心能力：身份一致性和对人类表演的精确控制。虽然最近的基于扩散的方法在保持面部身份方面取得了显著进展，但在不损害身份的情况下实现细粒度表情控制仍然具有挑战性。在这项工作中，我们提出了一个基于扩散的框架，能够在任何特定面部表情下忠实地重新构想任何主体。基于ID一致的面部基础模型，我们采用组合设计，包含由FLAME blendshape参数引导的表情交叉注意力模块，用于显式控制。在富含表情变化的多样化图像和视频数据上进行训练后，我们的适配器能够泛化到基本情感之外的微妙微表情和表情转换，这是先前工作所忽视的。此外，一个可插拔的参考适配器能够在合成过程中通过从参考帧转移外观来实现真实图像的表情编辑。大量的定量和定性评估表明，我们的模型在定制化和身份一致的表情生成方面优于现有方法。代码和模型可在https://github.com/foivospar/Arc2Face找到。


### 论文摘要

Human-centric generative models designed for AI-driven storytelling must bring together two core capabilities: identity consistency and precise control over human performance. While recent diffusion-based approaches have made significant progress in maintaining facial identity, achieving fine-grained expression control without compromising identity remains challenging. In this work, we present a diffusion-based framework that faithfully reimagines any subject under any particular facial expression. Building on an ID-consistent face foundation model, we adopt a compositional design featuring an expression cross-attention module guided by FLAME blendshape parameters for explicit control. Trained on a diverse mixture of image and video data rich in expressive variation, our adapter generalizes beyond basic emotions to subtle micro-expressions and expressive transitions, overlooked by prior works. In addition, a pluggable Reference Adapter enables expression editing in real images by transferring the appearance from a reference frame during synthesis. Extensive quantitative and qualitative evaluations show that our model outperforms existing methods in tailored and identity-consistent expression generation. Code and models can be found at https://github.com/foivospar/Arc2Face.

---

## 84. Label-Efficient Cross-Modality Generalization for Liver Segmentation in Multi-Phase MRI

**论文链接:** [http://arxiv.org/abs/2510.04705v1](http://arxiv.org/abs/2510.04705v1)

**作者:** Quang-Khai Bui-Tran, Minh-Toan Dinh, Thanh-Huy Nguyen, Ba-Thinh Lam, Mai-Anh Vu, Ulas Bagci

**发布时间:** 2025-10-06

**备注:** 11 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种标签高效的肝脏分割方法，通过结合基础模型适配和共同训练技术，解决了多相MRI中标记数据稀缺和分布不均的问题，实现了跨模态和跨供应商系统的稳健分割性能。

### 背景

在多相MRI中进行准确的肝脏分割对于肝纤维化评估至关重要，但标记数据通常稀缺，并且在成像模式和供应商系统之间分布不均。实际条件下，肝胆期注释有限，非对比序列未标记，且空间错位和缺失相位常见。

### 目的

提出一种标签高效的分割方法，促进在实际条件下的跨模态泛化能力，特别是在标记数据有限的情况下。

### 方法

集成一个基础规模的3D分割主干并通过微调进行适配；使用交叉伪监督进行共同训练以利用未标记的体积数据；采用标准化预处理流程；无需空间配准即可实现跨MRI相位和供应商的泛化。

### 主要发现

该方法在标记和未标记域中均表现出稳健的分割性能；所提出的标签基线对于多相、多供应商MRI中的肝脏分割有效。

### 结论

结合基础模型适配和共同训练对于实际临床成像任务具有潜力，能够有效解决多相MRI中肝脏分割面临的标记数据稀缺问题。

### 翻译

在多相MRI中进行准确的肝脏分割对于肝纤维化评估至关重要，但标记数据通常稀缺，并且在成像模式和供应商系统之间分布不均。我们提出了一种标签高效的分割方法，促进在实际条件下的跨模态泛化，其中肝胆期注释有限，非对比序列未标记，且空间错位和缺失相位常见。我们的方法集成了一个通过微调适配的基础规模3D分割主干，使用交叉伪监督进行共同训练以利用未标记体积，以及标准化预处理流程。无需空间配准，模型能够跨MRI相位和供应商泛化，在标记和未标记域中均表现出稳健的分割性能。我们的结果展示了所提出的标签基线对于多相、多供应商MRI中肝脏分割的有效性，并突显了结合基础模型适配和共同训练对于实际临床成像任务的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多相MRI中肝脏分割的标签效率问题和跨模态泛化问题。具体来说，就是如何在标记数据稀缺（只有少量GED4肝胆期有标签）且分布不均的情况下，实现不同MRI相位和供应商设备间的肝脏准确分割。这个问题很重要，因为肝脏是人体关键器官，其准确分割对肝纤维化等疾病的诊断至关重要，而临床环境中获取大量标记数据成本高、耗时长，不同医院和设备的差异也增加了分割难度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先明确定义了LiSeg任务面临的挑战，然后选择了STU-Net作为基础模型而非传统的nnU-Net，因为STU-Net具有更好的可扩展性和迁移能力。作者在TotalSegmentator数据集上预训练STU-Net，并在ATLAS肝脏分割数据集上进行微调以适应目标域。在半监督学习方面，作者采用了交叉伪监督(CPS)方法，结合了BCP和MiDSS等技术，让两个独立的网络互相提供伪标签。预处理方面借鉴了nnU-Net的标准化流程。整个方法综合了多种现有技术的优点，并针对医学图像特点进行了优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用大规模预训练模型作为基础，通过半监督学习同时利用有限的标记数据和大量的未标记数据，实现跨模态泛化。整体流程包括：1)使用LiQA多中心多供应商数据集；2)选择STU-Net作为基础模型并在ATLAS数据集上微调；3)采用双网络训练策略，两个网络互相提供伪标签；4)使用nnU-Net的预处理流程和增强技术；5)通过Dice相似系数和Hausdorff距离评估性能。这种方法不需要空间配准，就能处理不同MRI相位和供应商间的差异。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)标签效率与跨模态泛化的结合，特别适用于标记数据稀缺的多相MRI场景；2)使用具有更好迁移能力的STU-Net作为基础模型；3)采用交叉伪监督等多种半监督技术有效利用未标记数据；4)实现无需空间配准的跨模态泛化。相比之前工作，这种方法结合了多种技术的优点，不需要复杂的域适应技术，直接利用未标记数据进行端到端的跨模态泛化，在真实临床环境中表现更加鲁棒。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合基础模型微调和半监督学习的标签高效框架，能够在标记数据稀缺和跨模态差异的情况下实现多相MRI中肝脏的准确分割，展示了在真实临床环境中应用的潜力。'}


### 论文摘要

Accurate liver segmentation in multi-phase MRI is vital for liver fibrosis assessment, yet labeled data is often scarce and unevenly distributed across imaging modalities and vendor systems. We propose a label-efficient segmentation approach that promotes cross-modality generalization under real-world conditions, where GED4 hepatobiliary-phase annotations are limited, non-contrast sequences (T1WI, T2WI, DWI) are unlabeled, and spatial misalignment and missing phases are common. Our method integrates a foundation-scale 3D segmentation backbone adapted via fine-tuning, co-training with cross pseudo supervision to leverage unlabeled volumes, and a standardized preprocessing pipeline. Without requiring spatial registration, the model learns to generalize across MRI phases and vendors, demonstrating robust segmentation performance in both labeled and unlabeled domains. Our results exhibit the effectiveness of our proposed label-efficient baseline for liver segmentation in multi-phase, multi-vendor MRI and highlight the potential of combining foundation model adaptation with co-training for real-world clinical imaging tasks.

---

## 85. MedPAO: A Protocol-Driven Agent for Structuring Medical Reports

**论文链接:** [http://arxiv.org/abs/2510.04623v1](http://arxiv.org/abs/2510.04623v1)

**作者:** Shrish Shrinath Vaidya, Gowthamaan Palani, Sidharth Ramesh, Velmurugan Balasubramanian, Minmini Selvam, Gokulraja Srinivasaraja, Ganapathy Krishnamurthi

**发布时间:** 2025-10-06

**DOI:** 10.1007/978-3-032-06004-4_4

**备注:** Paper published at "Agentic AI for Medicine" Workshop, MICCAI 2025

### GPT解析

### 总结

该研究提出了MedPAO，一个用于结构化临床数据的新型智能框架，通过遵循临床协议确保准确性和可验证推理，解决了大型语言模型在临床数据结构化中的幻觉问题和无法遵循领域特定规则的问题。

### 背景

大型语言模型（LLMs）在结构化临床数据方面的应用受到其倾向于产生幻觉事实和无法遵循领域特定规则的严重阻碍。

### 目的

解决大型语言模型在临床数据结构化中的幻觉问题和无法遵循领域特定规则的问题，确保准确性和可验证推理。

### 方法

引入MedPAO，一个新颖的智能框架，它将报告结构化任务分解为由计划-行动-观察（PAO）循环和专门工具管理的透明过程，并基于已建立的临床协议（如CXR分析的ABCDEF协议）进行操作。

### 主要发现

MedPAO在概念分类的关键子任务上实现了0.96的F1分数。专家放射科医生和临床医生对最终结构化输出的平均评分为4.52分（满分5分），表明其可靠性超过了仅依赖基于LLM的基础模型的基线方法。

### 结论

MedPAO提供了一种可验证的替代方案，解决了传统单一模型在临床数据结构化中的问题，其可靠性和准确性超过了现有方法。

### 翻译

大型语言模型（LLMs）在结构化临床数据方面的部署受到其倾向于产生幻觉事实和无法遵循领域特定规则的严重阻碍。为解决这一问题，我们引入了MedPAO，一种新颖的智能框架，它通过基于已建立的临床协议（如CXR分析的ABCDEF协议）来确保操作准确性和可验证推理。MedPAO将报告结构化任务分解为由计划-行动-观察（PAO）循环和专门工具管理的透明过程。这种协议驱动的方法为不透明的单一模型提供了可验证的替代方案。我们通过严格的评估证明了我们方法的有效性：MedPAO在概念分类的关键子任务上实现了0.96的F1分数。值得注意的是，专家放射科医生和临床医生对最终结构化输出的平均评分为4.52分（满分5分），表明其可靠性超过了仅依赖基于LLM的基础模型的基线方法。代码可在以下网址获取：https://github.com/MiRL-IITM/medpao-agent


### 论文摘要

The deployment of Large Language Models (LLMs) for structuring clinical data is critically hindered by their tendency to hallucinate facts and their inability to follow domain-specific rules. To address this, we introduce MedPAO, a novel agentic framework that ensures accuracy and verifiable reasoning by grounding its operation in established clinical protocols such as the ABCDEF protocol for CXR analysis. MedPAO decomposes the report structuring task into a transparent process managed by a Plan-Act-Observe (PAO) loop and specialized tools. This protocol-driven method provides a verifiable alternative to opaque, monolithic models. The efficacy of our approach is demonstrated through rigorous evaluation: MedPAO achieves an F1-score of 0.96 on the critical sub-task of concept categorization. Notably, expert radiologists and clinicians rated the final structured outputs with an average score of 4.52 out of 5, indicating a level of reliability that surpasses baseline approaches relying solely on LLM-based foundation models. The code is available at: https://github.com/MiRL-IITM/medpao-agent

---

## 86. Pathology-CoT: Learning Visual Chain-of-Thought Agent from Expert Whole Slide Image Diagnosis Behavior

**论文链接:** [http://arxiv.org/abs/2510.04587v1](http://arxiv.org/abs/2510.04587v1)

**作者:** Sheng Wang, Ruiming Wu, Charles Herndon, Yihang Liu, Shunsuke Koga, Jeanne Shen, Zhi Huang

**发布时间:** 2025-10-06

### GPT解析

### 总结

本文介绍了一种名为AI会话记录器的系统，用于捕获病理学家观察全切片图像的行为模式，并基于这些数据构建了Pathologist-o3代理系统，在病理诊断中表现优异。

### 背景

诊断全切片图像是一个交互式多阶段过程，涉及放大倍数变化和视野间移动。尽管病理学基础模型强大，但缺乏能够决定下一个检查视野、调整放大倍数并提供可解释诊断的实用代理系统。主要障碍在于缺乏可扩展的、与临床一致的专家查看行为监督数据。

### 目的

开发一种将病理学家的日常查看日志转化为可扩展的、专家验证的监督数据的方法，构建实用的病理学代理系统，建立人类对齐的、可升级的临床AI路径。

### 方法

1) 引入AI会话记录器，记录常规导航并转换为标准化行为命令；2) 通过轻量级人工审核将AI起草理由转化为病理学思维链数据集；3) 基于行为数据构建两阶段代理系统Pathologist-o3，先提出感兴趣区域，再进行行为引导推理。

### 主要发现

1) AI会话记录器能有效捕获病理学家查看行为；2) 轻量级人工审核可将标记时间减少至传统方法的六分之一；3) Pathologist-o3在胃肠道淋巴结转移检测中达到84.5%精确率、100%召回率和75.4%准确率；4) 系统超越了OpenAI o3模型并能跨骨干架构泛化。

### 结论

将日常查看日志转化为可扩展的、专家验证的监督，该框架使代理病理学变得实用，为人类对齐的、可升级的临床AI铺平道路，是病理学领域首个基于行为的代理系统之一。

### 翻译

全切片图像诊断是一个涉及放大倍数变化和视野间移动的交互式多阶段过程。尽管最近的病理学基础模型表现强大，但仍然缺乏能够决定下一个检查视野、调整放大倍数并提供可解释诊断的实用代理系统。主要障碍在于数据：缺乏可扩展的、与临床一致的专家查看行为监督，这些行为是隐含的、基于经验的，未写在教科书或网络资源中，因此也缺失于大型语言模型的训练中。我们引入了AI会话记录器，它能够与标准WSI查看器配合使用，非侵入性地记录常规导航，并将查看器日志转换为标准化行为命令和边界框。轻量级人工循环审核将AI起草的理由转化为病理学思维链数据集，产生时间约为传统方法的六分之一。利用这些行为数据，我们构建了Pathologist-o3，在胃肠道淋巴结转移检测中表现优异，超越了最先进的OpenAI o3模型，并能跨骨干架构泛化。


### 论文摘要

Diagnosing a whole-slide image is an interactive, multi-stage process involving changes in magnification and movement between fields. Although recent pathology foundation models are strong, practical agentic systems that decide what field to examine next, adjust magnification, and deliver explainable diagnoses are still lacking. The blocker is data: scalable, clinically aligned supervision of expert viewing behavior that is tacit and experience-based, not written in textbooks or online, and therefore absent from large language model training. We introduce the AI Session Recorder, which works with standard WSI viewers to unobtrusively record routine navigation and convert the viewer logs into standardized behavioral commands (inspect or peek at discrete magnifications) and bounding boxes. A lightweight human-in-the-loop review turns AI-drafted rationales into the Pathology-CoT dataset, a form of paired "where to look" and "why it matters" supervision produced at roughly six times lower labeling time. Using this behavioral data, we build Pathologist-o3, a two-stage agent that first proposes regions of interest and then performs behavior-guided reasoning. On gastrointestinal lymph-node metastasis detection, it achieved 84.5% precision, 100.0% recall, and 75.4% accuracy, exceeding the state-of-the-art OpenAI o3 model and generalizing across backbones. To our knowledge, this constitutes one of the first behavior-grounded agentic systems in pathology. Turning everyday viewer logs into scalable, expert-validated supervision, our framework makes agentic pathology practical and establishes a path to human-aligned, upgradeable clinical AI.

---

## 87. Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators

**论文链接:** [http://arxiv.org/abs/2510.04354v1](http://arxiv.org/abs/2510.04354v1)

**作者:** Apurva Badithela, David Snyder, Lihan Zha, Joseph Mikhail, Matthew O'Kelly, Anushri Dixit, Anirudha Majumdar

**发布时间:** 2025-10-05

### GPT解析

### 总结

SureSim框架通过结合大规模仿真和小规模真实世界测试，为机器人操作策略的性能提供可靠统计推断，节省了20-25%的硬件评估工作量。

### 背景

机器人模仿学习、基础模型和大规模数据集的快速发展使得机器人操作策略能够泛化到各种任务和环境，但这些策略的严格评估仍然是一个挑战。

### 目的

提出SureSim框架，通过结合大规模仿真和相对小规模的真实世界测试，为策略在真实世界中的表现提供可靠的推断。

### 方法

将真实和仿真评估的组合问题形式化为预测驱动的推断问题，使用少量配对的真实和仿真评估修正仿真偏差，并利用非渐近均值估计算法为策略性能提供置信区间。

### 主要发现

在基于物理的仿真中评估扩散策略和多任务微调的π₀策略时，发现该方法可以节省超过20-25%的硬件评估工作量，同时获得类似的策略性能边界。

### 结论

SureSim框架能够有效结合仿真和真实世界测试，减少硬件评估需求，并提供策略性能的可靠统计推断。

### 翻译

模仿学习、基础模型和大规模数据集的快速发展使得机器人操作策略能够泛化到广泛多样的任务和环境。然而，对这些策略的严格评估仍然是一个挑战。通常在实践中，机器人策略往往只在少量硬件试验中进行评估，没有任何统计保证。我们提出了SureSim框架，通过将大规模仿真与相对小规模的真实世界测试相结合，为策略在真实世界中的表现提供可靠的推断。我们的核心思想是将结合真实和仿真评估的问题形式化为一个预测驱动的推断问题，其中使用少量配对的真实和仿真评估来修正大规模仿真中的偏差。然后，我们利用非渐近均值估计算法为策略性能均值提供置信区间。使用基于物理的仿真，我们在物体和初始条件的联合分布上评估了扩散策略和多任务微调的π₀策略，发现我们的方法可以节省超过20-25%的硬件评估工作量，同时获得类似的策略性能边界。


### 论文摘要

Rapid progress in imitation learning, foundation models, and large-scale datasets has led to robot manipulation policies that generalize to a wide-range of tasks and environments. However, rigorous evaluation of these policies remains a challenge. Typically in practice, robot policies are often evaluated on a small number of hardware trials without any statistical assurances. We present SureSim, a framework to augment large-scale simulation with relatively small-scale real-world testing to provide reliable inferences on the real-world performance of a policy. Our key idea is to formalize the problem of combining real and simulation evaluations as a prediction-powered inference problem, in which a small number of paired real and simulation evaluations are used to rectify bias in large-scale simulation. We then leverage non-asymptotic mean estimation algorithms to provide confidence intervals on mean policy performance. Using physics-based simulation, we evaluate both diffusion policy and multi-task fine-tuned \(\pi_0\) on a joint distribution of objects and initial conditions, and find that our approach saves over \(20-25\%\) of hardware evaluation effort to achieve similar bounds on policy performance.

---

## 88. DoRAN: Stabilizing Weight-Decomposed Low-Rank Adaptation via Noise Injection and Auxiliary Networks

**论文链接:** [http://arxiv.org/abs/2510.04331v1](http://arxiv.org/abs/2510.04331v1)

**作者:** Nghiem T. Diep, Hien Dang, Tuan Truong, Tan Dinh, Huy Nguyen, Nhat Ho

**发布时间:** 2025-10-05

**备注:** Nghiem T. Diep, Hien Dang, and Tuan Truong contributed equally to  this work

### GPT解析

### 总结

本文提出了DoRAN，一种基于DoRA改进的参数高效微调方法，通过噪声注入和动态网络生成两个关键技术，实现了训练稳定性和样本效率的双重提升。

### 背景

参数高效微调(PEFT)已成为适应大规模模型的标准范式。其中，权重分解低秩自适应(DoRA)通过将预训练权重分解为幅度和方向分量，提高了原始LoRA方法的学习能力和训练稳定性。

### 目的

设计一种新的DoRA变体，进一步稳定训练过程并提升DoRA的样本效率。

### 方法

包含两个关键阶段：(i)在DoRA权重分解的分母中注入噪声，作为自适应正则化器减轻不稳定性；(ii)用生成动态低秩矩阵的辅助网络替换静态低秩矩阵，实现层间参数耦合，提高样本效率。

### 主要发现

在视觉和语言基准测试上的全面实验表明，DoRAN持续优于LoRA、DoRA和其他PEFT基线方法。

### 结论

结合基于噪声的正则化与基于网络的参数生成，为基础模型的稳健高效微调提供了有前景的新方向。

### 翻译

参数高效微调(PEFT)方法已成为适应大规模模型的标准范式。在这些技术中，权重分解低秩自适应(DoRA)通过将预训练权重显式分解为幅度和方向分量，已被证明可以提高原始低秩自适应(LoRA)方法的学习能力和训练稳定性。在这项工作中，我们提出了DoRAN，一种新的DoRA变体，旨在进一步稳定训练并提高DoRA的样本效率。我们的方法包括两个关键阶段：(i)在DoRA权重分解的分母中注入噪声，作为自适应正则化器来减轻不稳定性；(ii)用生成它们的辅助网络动态替换静态低秩矩阵，实现层间参数耦合，在理论和实践中都获得更好的样本效率。在视觉和语言基准测试上的全面实验表明，DoRAN持续优于LoRA、DoRA和其他PEFT基线。这些结果强调了通过基于噪声的正则化与基于网络的参数生成相结合的有效性，为基础模型的稳健高效微调提供了有前景的方向。


### 论文摘要

Parameter-efficient fine-tuning (PEFT) methods have become the standard paradigm for adapting large-scale models. Among these techniques, Weight-Decomposed Low-Rank Adaptation (DoRA) has been shown to improve both the learning capacity and training stability of the vanilla Low-Rank Adaptation (LoRA) method by explicitly decomposing pre-trained weights into magnitude and directional components. In this work, we propose DoRAN, a new variant of DoRA designed to further stabilize training and boost the sample efficiency of DoRA. Our approach includes two key stages: (i) injecting noise into the denominator of DoRA's weight decomposition, which serves as an adaptive regularizer to mitigate instabilities; and (ii) replacing static low-rank matrices with auxiliary networks that generate them dynamically, enabling parameter coupling across layers and yielding better sample efficiency in both theory and practice. Comprehensive experiments on vision and language benchmarks show that DoRAN consistently outperforms LoRA, DoRA, and other PEFT baselines. These results underscore the effectiveness of combining stabilization through noise-based regularization with network-based parameter generation, offering a promising direction for robust and efficient fine-tuning of foundation models.

---

## 89. Why Cannot Neural Networks Master Extrapolation? Insights from Physical Laws

**论文链接:** [http://arxiv.org/abs/2510.04102v1](http://arxiv.org/abs/2510.04102v1)

**作者:** Ramzi Dakhmouche, Hossein Gorji

**发布时间:** 2025-10-05

### GPT解析

### 总结

本研究探讨了基础模型在时间序列预测中的应用，特别关注它们在外推和长期预测方面的局限性，以及与物理定律的对比。

### 背景

基础模型在语言建模中取得了显著成功，激发了其在时间序列预测领域的应用研究。这些模型对科学和工程具有变革性潜力，并在短期预测中表现出色。

### 目的

确定并形式化一个基本属性，该属性表征统计学习模型在训练域外进行更准确预测的能力，解释深度学习模型在外推设置中性能下降的原因，并为设计能够掌握外推能力的下一代预测模型提供方向。

### 方法

通过理论分析和实证研究，探讨神经网络结构与物理定律之间的根本差异，以及这种差异如何影响模型的外推能力。

### 主要发现

基础模型在短期预测中表现良好，但在外推或长期预测方面表现不佳，甚至无法超越简单基线方法。研究确定了一个基本属性，解释了深度学习模型在外推设置中性能下降的原因。

### 结论

研究结果阐明了外推差距的根本原因，并为设计能够掌握外推能力的下一代预测模型提供了方向。理解神经网络结构与物理定律之间的差异对于改进预测模型的外推能力至关重要。

### 翻译

受语言建模中基础模型的显著成功启发，人们对开发用于时间序列预测的基础模型越来越感兴趣，因为这些模型对科学和工程具有变革性潜力。这导致了基础模型在短期预测设置中取得显著成功。然而，对于外推或长期预测，基础模型仍然难以实现，甚至无法超越简单的基线方法。这与物理定律形成鲜明对比，因为物理定律具有很强的外推特性。这引发了关于神经网络结构与物理定律之间根本差异的问题。在这项工作中，作者确定并形式化了一个基本属性，该属性表征了统计学习模型在训练域外进行更准确预测的能力，从而解释了深度学习模型在外推设置中性能下降的原因。除了理论分析外，作者还展示了实证结果，说明了这一属性对当前深度学习架构的影响。研究结果不仅阐明了外推差距的根本原因，还为设计能够掌握外推能力的下一代预测模型提供了方向。


### 论文摘要

Motivated by the remarkable success of Foundation Models (FMs) in language modeling, there has been growing interest in developing FMs for time series prediction, given the transformative power such models hold for science and engineering. This culminated in significant success of FMs in short-range forecasting settings. However, extrapolation or long-range forecasting remains elusive for FMs, which struggle to outperform even simple baselines. This contrasts with physical laws which have strong extrapolation properties, and raises the question of the fundamental difference between the structure of neural networks and physical laws. In this work, we identify and formalize a fundamental property characterizing the ability of statistical learning models to predict more accurately outside of their training domain, hence explaining performance deterioration for deep learning models in extrapolation settings. In addition to a theoretical analysis, we present empirical results showcasing the implications of this property on current deep learning architectures. Our results not only clarify the root causes of the extrapolation gap but also suggest directions for designing next-generation forecasting models capable of mastering extrapolation.

---

## 90. Zephyrus: An Agentic Framework for Weather Science

**论文链接:** [http://arxiv.org/abs/2510.04017v1](http://arxiv.org/abs/2510.04017v1)

**作者:** Sumanth Varambally, Marshall Fisher, Jas Thakker, Yiwei Chen, Zhirui Xia, Yasaman Jafari, Ruijia Niu, Manas Jain, Veeramakali Vignesh Manivannan, Zachary Novack, Luyu Han, Srikar Eranky, Salva Rühling Cachay, Taylor Berg-Kirkpatrick, Duncan Watson-Parris, Yi-An Ma, Rose Yu

**发布时间:** 2025-10-05

### GPT解析

### 总结

研究构建了一个名为Zephyrus的气象科学智能体框架，结合了基础模型的数值数据处理能力和大型语言模型的文本理解能力，通过ZephyrusWorld环境与气象数据交互，并创建了ZephyrusBench基准测试评估性能。

### 背景

基础模型在气象科学领域通过大量结构化数值数据预训练，优于传统天气预报系统，但缺乏基于语言的推理能力；而大型语言模型擅长理解和生成文本，却无法推理高维气象数据集。

### 目的

弥合基础模型和大语言模型之间的差距，为气象科学构建一个新的智能体框架，结合两者的优势。

### 方法

构建ZephyrusWorld基于Python代码的交互环境，包含WeatherBench 2数据集接口、地理掩码查询、天气预报和气候模拟等功能；设计Zephyrus多回合LLM气象智能体，迭代分析数据并通过对话反馈循环改进；创建ZephyrusBench基准测试和数据生成管道，构建多样化气象任务问答对。

### 主要发现

Zephyrus智能体在正确性方面比纯文本基线高出多达35个百分点，但在更困难的任务上表现与纯文本基线相似，突显了基准测试的挑战性。

### 结论

该框架为气象科学提供了有前途的方向，但未来工作需解决更复杂任务的挑战，以进一步提升智能体性能。

### 翻译

气象科学的基础模型是在大量结构化数值数据上预训练的，并且优于传统的天气预报系统。然而，这些模型缺乏基于语言的推理能力，限制了它们在交互式科学工作流程中的实用性。大型语言模型(LLMs)擅长理解和生成文本，但无法推理高维气象数据集。我们通过为气象科学构建一个新的智能体框架来弥合这一差距。我们的框架包括一个基于Python代码的环境，供智能体(ZephyrusWorld)与气象数据交互，具有WeatherBench 2数据集接口、从自然语言查询地理掩码、天气预报和气候模拟等功能。我们设计了Zephyrus，这是一个多回合基于LLM的气象智能体，它迭代分析气象数据集，观察结果，并通过对话反馈循环改进其方法。我们为该智能体配备了新的基准测试ZephyrusBench，它具有可扩展的数据生成管道，能够构建跨气象相关任务的多样化问答对，从基本查询到高级预报、极端事件检测和反事实推理。在该基准测试上的实验表明，Zephyrus智能体在正确性方面比纯文本基线高出多达35个百分点。然而，在更困难的任务上，Zephyrus的表现与纯文本基线相似，突显了我们基准测试的挑战性，并为未来工作指出了有前途的方向。


### 论文摘要

Foundation models for weather science are pre-trained on vast amounts of structured numerical data and outperform traditional weather forecasting systems. However, these models lack language-based reasoning capabilities, limiting their utility in interactive scientific workflows. Large language models (LLMs) excel at understanding and generating text but cannot reason about high-dimensional meteorological datasets. We bridge this gap by building a novel agentic framework for weather science. Our framework includes a Python code-based environment for agents (ZephyrusWorld) to interact with weather data, featuring tools like an interface to WeatherBench 2 dataset, geoquerying for geographical masks from natural language, weather forecasting, and climate simulation capabilities. We design Zephyrus, a multi-turn LLM-based weather agent that iteratively analyzes weather datasets, observes results, and refines its approach through conversational feedback loops. We accompany the agent with a new benchmark, ZephyrusBench, with a scalable data generation pipeline that constructs diverse question-answer pairs across weather-related tasks, from basic lookups to advanced forecasting, extreme event detection, and counterfactual reasoning. Experiments on this benchmark demonstrate the strong performance of Zephyrus agents over text-only baselines, outperforming them by up to 35 percentage points in correctness. However, on harder tasks, Zephyrus performs similarly to text-only baselines, highlighting the challenging nature of our benchmark and suggesting promising directions for future work.

---

## 91. What Shapes a Creative Machine Mind? Comprehensively Benchmarking Creativity in Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.04009v1](http://arxiv.org/abs/2510.04009v1)

**作者:** Zicong He, Boxuan Zhang, Weihao Liu, Ruixiang Tang, Lu Cheng

**发布时间:** 2025-10-05

**备注:** 22 pages

### GPT解析

### 总结

本文提出了C^2-Eval，一个用于统一评估基础模型创造力的整体基准，区分收敛创造力和发散创造力，并使用实用性、原创性和惊喜度作为评估标准。

### 背景

基础模型的能力已扩展到传统任务之外，创造力作为人类智能的标志和创新驱动力，现在被认为是生成式基础模型时代机器智能的关键维度，补充了传统准确性衡量标准。

### 目的

解决现有创造力评估框架分散、缺乏理论基础的问题，引入一个统一的创造力评估基准。

### 方法

C^2-Eval区分收敛创造力（有约束解决方案的任务）和发散创造力（开放式任务），使用源自社会科学理论的细粒度标准评估这两个维度，重点关注实用性、原创性和惊喜度。

### 主要发现

通过对领先的专有和开源模型进行广泛实验，分析了它们在创造力能力方面的权衡，揭示了当前基础模型在追求创造性机器思维方面的优势和挑战。

### 结论

C^2-Eval是检查创意AI不断发展的格局的有效视角。

### 翻译

基础模型的兴起极大地扩展了其能力范围，远超传统任务。创造力长期以来被视为人类智能的标志和创新驱动力，现在在生成式基础模型时代越来越被认可为机器智能的关键维度，补充了传统准确性衡量标准。然而，现有的创造力评估框架仍然分散，依赖于临时性指标，这些指标没有牢固地建立在既定理论基础上。为了解决这一差距，我们引入了C^2-Eval，这是一个用于统一评估基础模型创造力的整体基准。C^2-Eval区分了两种互补的创造力形式：收敛创造力（任务允许有约束的解决方案，如代码生成）和发散创造力（任务是开放式的，如讲故事）。它使用源自社会科学理论的细粒度标准评估这两个维度，重点关注实用性、原创性和惊喜度。通过对领先的专有和开源模型进行广泛实验，我们分析了它们在创造力能力方面的权衡。我们的结果强调了当前基础模型在追求创造性机器思维方面的优势和挑战，表明C^2-Eval是检查创意AI不断发展的格局的有效视角。


### 论文摘要

The meteoric rise of foundation models (FMs) has expanded their capabilities far beyond conventional tasks. Creativity, long regarded as a hallmark of human intelligence and a driver of innovation, is now increasingly recognized as a critical dimension of machine intelligence in the era of generative FMs, complementing traditional measures of accuracy. However, existing evaluation frameworks for creativity remain fragmented, relying on ad hoc metrics not firmly grounded in established theories. To address this gap, we introduce C^2-Eval, a holistic benchmark for unified assessment of creativity in FMs. C^2-Eval distinguishes between two complementary forms of creativity: convergent creativity, where tasks admit constrained solutions (e.g., code generation), and divergent creativity, where tasks are open-ended (e.g., storytelling). It evaluates both dimensions using fine-grained criteria derived from social-science theory, focusing on Usefulness, Originality, and Surprise (U-O-S). Through extensive experiments on leading proprietary and open-source models, we analyze trade-offs in their creative capabilities. Our results highlight both the strengths and challenges of current FMs in pursuing a creative machine mind, showing that C^2-Eval is an effective lens for examining the evolving landscape of creative AI.

---

## 92. THEMIS: Unlocking Pretrained Knowledge with Foundation Model Embeddings for Anomaly Detection in Time Series

**论文链接:** [http://arxiv.org/abs/2510.03911v1](http://arxiv.org/abs/2510.03911v1)

**作者:** Yadav Mahesh Lorik, Kaushik Sarveswaran, Nagaraj Sundaramahalingam, Aravindakumar Venugopalan

**发布时间:** 2025-10-04

**备注:** Oral Presentation. AI4TS Workshop, IJCAI'25

### GPT解析

### 总结

本文提出了THEMIS，一个利用基础模型预训练知识的时间序列异常检测新框架，通过提取Chronos模型的嵌入并应用异常检测技术，在多个数据集上取得了最先进的结果。

### 背景

时间序列异常检测在多个领域至关重要但面临挑战，包括数据具有季节性、趋势、噪声和演变模式，异常类型多样且罕见导致数据不平衡，以及高维、实时检测阈值设置和结果可解释性等问题。

### 目的

开发强大、灵活且可解释的方法来应对时间序列异常检测的多方面挑战，提出THEMIS框架。

### 方法

THEMIS利用基础模型的预训练知识，从Chronos时间序列基础模型的编码器中提取嵌入，并在自相似矩阵上应用局部异常因子和谱分解等异常检测技术来识别异常。

### 主要发现

THEMIS在MSL数据集上取得最先进结果，在SMAP和SWAT数据上表现具有竞争力，超过了专门为异常检测训练的模型，具有超参数鲁棒性和默认可解释性。

### 结论

提倡使用基础模型的预训练表示来进行高效且适应性强的异常检测。

### 翻译

时间序列异常检测在多个领域中构成非常关键的区域，但带来了重大挑战。由于时间序列数据具有季节性、趋势、噪声和演变模式(概念漂移)，很难确定什么是正常行为的通用概念。异常本身可能是多样的，从单一异常值到上下文异常或集体异常，而且通常非常罕见，因此数据集严重不平衡。现代时间序列的高维问题、实时检测标准、设置适当的检测阈值以及获得可解释的结果等问题增加了额外的复杂性。为了应对这些多方面的挑战，需要非常强大、灵活且可解释的方法。本文提出了THEMIS，一个用于时间序列异常检测的新框架，它利用基础模型的预训练知识。THEMIS从Chronos时间序列基础模型的编码器中提取嵌入，并在自相似矩阵上应用局部异常因子和谱分解等异常检测技术，以发现数据中的异常。我们的实验表明，这种模块化方法在MSL数据集上取得了最先进的结果，并在SMAP和SWAT数据集上表现得相当有竞争力。值得注意的是，THEMIS超过了专门为异常检测训练的模型，具有超参数鲁棒性和默认的可解释性。本文提倡使用基础模型的预训练表示来对时间序列数据进行高效且适应性强的异常检测。


### 论文摘要

Time series anomaly detection forms a very crucial area in several domains but poses substantial challenges. Due to time series data possessing seasonality, trends, noise, and evolving patterns (concept drift), it becomes very difficult to set a general notion of what constitutes normal behavior. Anomalies themselves could be varied, ranging from a single outlier to contextual or collective anomalies, and are normally very rare; hence, the dataset is largely imbalanced. Additional layers of complexities arise due to the problems of increased dimensionality of modern time series, real-time detection criteria, setting up appropriate detection thresholds, and arriving at results that are interpretable. To embrace these multifaceted challenges, very strong, flexible, and interpretable approaches are required. This paper presents THEMIS, a new framework for time series anomaly detection that exploits pretrained knowledge from foundation models. THEMIS extracts embeddings from the encoder of the Chronos time series foundation model and applies outlier detection techniques like Local Outlier Factor and Spectral Decomposition on the self-similarity matrix, to spot anomalies in the data. Our experiments show that this modular method achieves SOTA results on the MSL dataset and performs quite competitively on the SMAP and SWAT$^*$ datasets. Notably, THEMIS exceeds models trained specifically for anomaly detection, presenting hyperparameter robustness and interpretability by default. This paper advocates for pretrained representations from foundation models for performing efficient and adaptable anomaly detection for time series data.

---

## 93. The Overlooked Value of Test-time Reference Sets in Visual Place Recognition

**论文链接:** [http://arxiv.org/abs/2510.03751v1](http://arxiv.org/abs/2510.03751v1)

**作者:** Mubariz Zaffar, Liangliang Nan, Sebastian Scherer, Julian F. P. Kooij

**发布时间:** 2025-10-04

**备注:** Accepted at ICCV 2025 Workshop CrocoDL

### GPT解析

### 总结

本研究提出了一种名为参考集微调(RSF)的新方法，通过利用测试时的参考集（地图）信息来微调视觉位置识别(VPR)模型，以提高模型在具有挑战性基准测试上的性能。

### 背景

视觉位置识别(VPR)的任务是根据查询图像从参考数据库中检索同一地点的图像，需要能够抵抗视角和外观变化。虽然使用视觉基础模型主干网络并在大规模VPR数据集上训练的方法已解决了一些基准测试，但当测试环境与训练数据集有显著差异时，这些方法仍面临挑战。

### 目的

探索一种互补的、未被探索的信息来源来弥合训练-测试域差距，以提高最先进VPR方法在具有挑战性基准测试上的性能。

### 方法

在测试时的参考集（地图）上进行简单的参考集微调(RSF)，因为参考集包含目标域的图像和姿态信息，且在某些VPR应用中必须在接收测试查询之前可用。

### 主要发现

在地图上微调VPR模型可以提高在具有挑战性数据集上的性能；微调后的模型平均Recall@1指标提升了约2.3%；微调后的模型保留了泛化能力；RSF方法在多样化的测试数据集上都有效。

### 结论

参考集微调(RSF)是一种有效的方法，可以利用测试时的参考集信息来提高VPR模型在具有挑战性基准测试上的性能，同时保持模型的泛化能力。

### 翻译

给定一个查询图像，视觉位置识别(VPR)是从参考数据库中检索同一地点图像的任务，需要能够抵抗视角和外观变化的影响。最近的研究表明，一些VPR基准测试可以通过使用视觉基础模型主干网络并在大规模、多样化的VPR专用数据集上进行训练来解决。然而，一些基准测试仍然具有挑战性，特别是当测试环境与通常的VPR训练数据集有显著差异时。我们提出了一种互补的、未被探索的信息来源来弥合训练-测试域差距，这可以进一步提高最先进(SOTA)的VPR方法在这些具有挑战性的基准测试上的性能。具体来说，我们识别出测试时的参考集，即'地图'，包含目标域的图像和姿态信息，并且在几种VPR应用中必须在接收测试时查询之前可用。因此，我们提出在地图上进行简单的参考集微调(RSF)的VPR模型，在这些具有挑战性的数据集上将SOTA性能提升了约2.3%（平均Recall@1）。微调后的模型保留了泛化能力，并且RSF在多样化的测试数据集上都有效。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决视觉地点识别（VPR）中的'训练-测试域差距'问题，即当测试环境与训练数据集差异显著时，即使是最先进的VPR方法表现也会下降。这个问题很重要，因为许多VPR应用（如地标检索、3D建模、图像搜索和基于地图的定位）要求测试时参考集在接收查询前就已可用，解决这个问题能显著提高VPR系统在真实世界不同环境中的性能和可靠性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到当前VPR方法在类似训练环境的数据集上表现良好，但在差异大的测试环境中表现不佳。他们注意到测试时参考集（地图）一直被忽视，但这个信息源在VPR应用中是可用的。作者设计了参考集微调（RSF）策略，利用测试前可用的参考图像来微调预训练模型。该方法借鉴了视觉基础模型（如DinoV2）、特征聚合方法（如BoQ）、域适应技术和数据增强方法，但将这些技术整合到一个新颖的应用场景中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用测试时已经可用的参考图像集来微调预训练的VPR模型，从而缩小训练-测试域差距，无需新的训练数据或更强的骨干网络。实现流程包括：1) 从测试参考集创建微调数据集，通过数据增强生成查询图像；2) 使用预训练模型对图像进行特征编码；3) 利用姿态信息进行三元组挖掘，找到正样本和困难负样本；4) 使用三元组损失函数进行模型微调；5) 在验证集和测试集上评估性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 发现并利用测试时参考集的价值来缩小训练-测试域差距；2) 提出简单有效的参考集微调（RSF）策略；3) 设计姿态感知的三元组挖掘方法；4) 创建自监督微调框架，无需额外标注数据。相比之前工作，这篇论文不仅关注查询-参考域差距（传统VPR焦点），还关注训练-测试域差距；利用测试时参考集而非需要额外标注数据；专注于可以离线使用参考集的应用场景；即使在没有姿态信息的情况下，RSF仍然有效。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种简单而有效的参考集微调（RSF）策略，利用测试时已经可用的参考图像集来缩小视觉地点识别中的训练-测试域差距，显著提高了模型在未见环境中的性能，同时保持了模型的泛化能力。'}


### 论文摘要

Given a query image, Visual Place Recognition (VPR) is the task of retrieving an image of the same place from a reference database with robustness to viewpoint and appearance changes. Recent works show that some VPR benchmarks are solved by methods using Vision-Foundation-Model backbones and trained on large-scale and diverse VPR-specific datasets. Several benchmarks remain challenging, particularly when the test environments differ significantly from the usual VPR training datasets. We propose a complementary, unexplored source of information to bridge the train-test domain gap, which can further improve the performance of State-of-the-Art (SOTA) VPR methods on such challenging benchmarks. Concretely, we identify that the test-time reference set, the "map", contains images and poses of the target domain, and must be available before the test-time query is received in several VPR applications. Therefore, we propose to perform simple Reference-Set-Finetuning (RSF) of VPR models on the map, boosting the SOTA (~2.3% increase on average for Recall@1) on these challenging datasets. Finetuned models retain generalization, and RSF works across diverse test datasets.

---

## 94. Bridging the Gap Between Multimodal Foundation Models and World Models

**论文链接:** [http://arxiv.org/abs/2510.03727v1](http://arxiv.org/abs/2510.03727v1)

**作者:** Xuehai He

**发布时间:** 2025-10-04

**备注:** PhD thesis

### GPT解析

### 总结

该研究探讨了如何弥合多模态基础模型与世界模型之间的差距，通过改进推理能力和生成能力，使模型能够更好地理解和模拟动态物理过程。

### 背景

人类通过整合多种感官模态理解世界，受此启发，多模态基础模型(MFMs)已成为多模态理解和生成的强大工具。然而，当前MFMs作为世界模型存在不足，缺乏关键能力。

### 目的

研究如何提升多模态基础模型的能力，使其更接近世界模型的功能，能够进行反事实推理、动态模拟、时空理解等高级认知任务。

### 方法

1) 通过区分性任务提高MFMs推理能力，赋予因果推断、反事实思维和时空推理等结构化推理技能；2) 探索图像和视频模态中的生成能力，引入结构化和可控生成框架；3) 利用场景图、多模态条件和多模态对齐策略指导生成过程；4) 将技术扩展到可控4D生成，实现时空交互式、可编辑和可变形对象合成。

### 主要发现

通过赋予MFMs结构化推理能力和可控生成能力，可以使其超越表面相关性，理解视觉和文本数据中的深层关系，并实现与高级语义和细粒度用户意图一致的生成结果。

### 结论

多模态基础模型通过增强推理和生成能力，可以逐步具备世界模型的功能，更好地模拟和理解动态物理过程。

### 翻译

人类通过整合多种感官模态来理解世界，使他们能够感知、推理和想象动态物理过程。受这一能力启发，多模态基础模型(MFMs)已成为多模态理解和生成的强大工具。然而，当今的MFMs作为有效的世界模型尚有不足。它们缺乏基本能力，如进行反事实推理、模拟动态、理解时空信息、控制生成的视觉结果和执行多方面推理。我们研究弥合多模态基础模型与世界模型之间差距所需的条件。我们首先通过区分性任务提高MFMs的推理能力，赋予其因果推断、反事实思维和时空推理等结构化推理技能，使它们能够超越表面相关性，理解视觉和文本数据中的深层关系。接下来，我们探索多模态基础模型在图像和视频模态中的生成能力，引入新的结构化和可控生成框架。我们的方法结合场景图、多模态条件和多模态对齐策略来指导生成过程，确保与高级语义和细粒度用户意图保持一致。我们进一步将这些技术扩展到可控4D生成，实现随时间和空间的交互式、可编辑和可变形对象合成。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态基础模型与世界模型之间的差距问题。当前的多模态模型虽然能处理多种任务，但缺乏核心的世界建模能力，如反事实推理、动态模拟、时空信息理解、视觉结果控制和多方面推理。这个问题很重要，因为人类通过整合多种感官模态理解世界，而缺乏这些能力的模型无法以类人方式模拟、规划和与世界互动，限制了人工智能系统向更高级认知能力的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出多模态基础模型与世界模型之间的差距，然后从两个主要方向进行改进：一是提升感知和推理能力，二是增强生成能力。在推理方面，作者借鉴了参数高效学习、提示学习范式、因果推理和图结构等方法；在生成方面，作者借鉴了扩散模型、多模态控制和场景图等技术。作者将现有方法与新的创新相结合，系统地解决了多模态模型的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过增强多模态模型的推理能力和生成能力来弥合与世界模型之间的差距。在推理方面，通过反事实思考、因果推理、图结构等方法使模型能够进行更深层次的推理；在生成方面，通过可控的文本到图像生成、视频生成的动态控制和4D场景生成等方法使模型能够模拟世界的变化。整体流程是：首先分析差距，然后分别从判别式和生成式两个方向进行改进，最后引入评估基准来验证进展。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 提出反事实提示学习范式增强模型推理能力；2) 设计多模态图变换器进行结构化推理；3) 开发判别性扩散模型将生成模型用于感知任务；4) 引入MMWorld和VLM4D基准全面评估世界模型能力；5) 开发FlexEControl、Mojito和Morpho4D框架实现可控的2D、3D和4D生成。相比之前工作，这些创新不仅关注表面理解，还强调深层推理；将生成模型与判别任务结合；从2D扩展到4D生成；强调可控性和交互性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过增强多模态模型的推理和生成能力并引入全面评估基准，显著缩小了多模态基础模型与世界模型之间的差距，朝着开发能像人类一样推理、模拟和与世界的智能系统迈出了重要一步。'}


### 论文摘要

Humans understand the world through the integration of multiple sensory modalities, enabling them to perceive, reason about, and imagine dynamic physical processes. Inspired by this capability, multimodal foundation models (MFMs) have emerged as powerful tools for multimodal understanding and generation. However, today's MFMs fall short of serving as effective world models. They lack the essential ability such as perform counterfactual reasoning, simulate dynamics, understand the spatiotemporal information, control generated visual outcomes, and perform multifaceted reasoning. We investigates what it takes to bridge the gap between multimodal foundation models and world models. We begin by improving the reasoning capabilities of MFMs through discriminative tasks and equipping MFMs with structured reasoning skills, such as causal inference, counterfactual thinking, and spatiotemporal reasoning, enabling them to go beyond surface correlations and understand deeper relationships within visual and textual data. Next, we explore generative capabilities of multimodal foundation models across both image and video modalities, introducing new frameworks for structured and controllable generation. Our approaches incorporate scene graphs, multimodal conditioning, and multimodal alignment strategies to guide the generation process, ensuring consistency with high-level semantics and fine-grained user intent. We further extend these techniques to controllable 4D generation, enabling interactive, editable, and morphable object synthesis over time and space.

---

## 95. GAS-MIL: Group-Aggregative Selection Multi-Instance Learning for Ensemble of Foundation Models in Digital Pathology Image Analysis

**论文链接:** [http://arxiv.org/abs/2510.03555v1](http://arxiv.org/abs/2510.03555v1)

**作者:** Peiran Quan, Zifan Gu, Zhuo Zhao, Qin Zhou, Donghan M. Yang, Ruichen Rong, Yang Xie, Guanghua Xiao

**发布时间:** 2025-10-03

### GPT解析

### 总结

本研究提出了一种名为Group-Aggregative Selection Multi-Instance Learning (GAS-MIL)的灵活集成框架，能够无缝整合多个Foundation models的特征，保留它们的互补优势，无需手动特征选择或大量任务特定的微调。该方法在三个癌症数据集的分类任务中表现优异，展示了其鲁棒性和泛化能力。

### 背景

Foundation models已经通过提供强大的通用特征提取器改变了计算病理学领域。然而，针对特定诊断任务调整和评估单个Foundation模型通常耗时且资源密集，特别是考虑到它们的规模和多样性。

### 目的

解决Foundation models在适应特定诊断任务和基准测试时面临的耗时和资源密集型挑战，提供一种无需大量任务特定微调的解决方案。

### 方法

引入Group-Aggregative Selection Multi-Instance Learning (GAS-MIL)，这是一种灵活的集成框架，能够无缝整合多个Foundation models的特征，保留它们的互补优势，而无需手动特征选择或大量任务特定的微调。

### 主要发现

在前列腺癌(PANDA)、卵巢癌(UBC-OCEAN)和乳腺癌(TCGA-BrCa)三个癌症数据集的分类任务中，GAS-MIL始终取得了优于或相当于单个Foundation models和既定MIL方法的性能，证明了其鲁棒性和泛化能力。

### 结论

通过高效整合异构的Foundation models，GAS-MIL简化了病理学模型部署，并为未来的多模态和精准肿瘤学应用提供了可扩展的基础。

### 翻译

Foundation models已经通过提供强大的通用特征提取器改变了计算病理学领域。然而，针对特定诊断任务调整和评估单个Foundation模型通常耗时且资源密集，特别是考虑到它们的规模和多样性。为应对这一挑战，我们引入了Group-Aggregative Selection Multi-Instance Learning (GAS-MIL)，这是一种灵活的集成框架，能够无缝整合多个Foundation models的特征，保留它们的互补优势，而无需手动特征选择或大量任务特定的微调。在前列腺癌(PANDA)、卵巢癌(UBC-OCEAN)和乳腺癌(TCGA-BrCa)三个癌症数据集的分类任务中，GAS-MIL始终取得了优于或相当于单个Foundation models和既定MIL方法的性能，证明了其鲁棒性和泛化能力。通过高效整合异构的Foundation models，GAS-MIL简化了病理学模型部署，并为未来的多模态和精准肿瘤学应用提供了可扩展的基础。


### 论文摘要

Foundation models (FMs) have transformed computational pathology by providing powerful, general-purpose feature extractors. However, adapting and benchmarking individual FMs for specific diagnostic tasks is often time-consuming and resource-intensive, especially given their scale and diversity. To address this challenge, we introduce Group-Aggregative Selection Multi-Instance Learning (GAS-MIL), a flexible ensemble framework that seamlessly integrates features from multiple FMs, preserving their complementary strengths without requiring manual feature selection or extensive task-specific fine-tuning. Across classification tasks in three cancer datasets-prostate (PANDA), ovarian (UBC-OCEAN), and breast (TCGA-BrCa)-GAS-MIL consistently achieves superior or on-par performance relative to individual FMs and established MIL methods, demonstrating its robustness and generalizability. By enabling efficient integration of heterogeneous FMs, GAS-MIL streamlines model deployment for pathology and provides a scalable foundation for future multimodal and precision oncology applications.

---

## 96. Domain Generalization for Semantic Segmentation: A Survey

**论文链接:** [http://arxiv.org/abs/2510.03540v1](http://arxiv.org/abs/2510.03540v1)

**作者:** Manuel Schwonberg, Hanno Gottschalk

**发布时间:** 2025-10-03

**备注:** Accepted to CVPR2025W

### GPT解析

### 总结

这篇综述全面概述了领域泛化语义分割的快速发展主题，对现有方法进行了分类和回顾，并展示了基础模型对领域泛化的重大影响。

### 背景

尽管深度神经网络近年来取得了巨大进步，但它们在未知领域的泛化能力仍然是一个重大挑战。领域泛化(DG)作为应对这一挑战的动态领域应运而生。与无监督领域适应不同，领域泛化无法访问或了解目标领域，而是旨在跨越多个不同的未见过的目标领域进行泛化。领域泛化对于语义分割任务特别重要，该任务在生物医学或自动驾驶等多个领域都有应用。

### 目的

这篇综述旨在推进领域泛化研究，并激励科学家探索新的研究方向。

### 方法

作者对现有方法进行了分类和回顾，确定了向基于基础模型的领域泛化的范式转变，并对所有方法进行了广泛的性能比较。

### 主要发现

研究确定了领域泛化语义分割领域向基于基础模型的范式转变，并通过广泛的性能比较突出了基础模型对领域泛化的重大影响。

### 结论

基础模型对领域泛化有显著影响，这一综述为领域泛化研究提供了全面概述，并鼓励科学家探索新的研究方向。

### 翻译

尽管深度神经网络近年来取得了巨大进步，但它们在未知领域的泛化能力仍然是一个重大挑战。因此，领域泛化(DG)这一动态领域应运而生。与无监督领域适应不同，领域泛化无法访问或了解目标领域，而是旨在跨越多个不同的未见过的目标领域进行泛化。领域泛化对于语义分割任务特别重要，该任务在生物医学或自动驾驶等多个领域都有应用。这篇综述对领域泛化语义分割这一快速发展的主题进行了全面概述。我们对现有方法进行了分类和回顾，并确定了向基于基础模型的领域泛化的范式转变。最后，我们对所有方法进行了广泛的性能比较，突出了基础模型对领域泛化的重大影响。这篇综述旨在推进领域泛化研究，并激励科学家探索新的研究方向。


### 论文摘要

The generalization of deep neural networks to unknown domains is a major challenge despite their tremendous progress in recent years. For this reason, the dynamic area of domain generalization (DG) has emerged. In contrast to unsupervised domain adaptation, there is no access to or knowledge about the target domains, and DG methods aim to generalize across multiple different unseen target domains. Domain generalization is particularly relevant for the task semantic segmentation which is used in several areas such as biomedicine or automated driving. This survey provides a comprehensive overview of the rapidly evolving topic of domain generalized semantic segmentation. We cluster and review existing approaches and identify the paradigm shift towards foundation-model-based domain generalization. Finally, we provide an extensive performance comparison of all approaches, which highlights the significant influence of foundation models on domain generalization. This survey seeks to advance domain generalization research and inspire scientists to explore new research directions.

---

## 97. TS-Reasoner: Aligning Time Series Foundation Models with LLM Reasoning

**论文链接:** [http://arxiv.org/abs/2510.03519v1](http://arxiv.org/abs/2510.03519v1)

**作者:** Fangxu Yu, Hongyu Zhao, Tianyi Zhou

**发布时间:** 2025-10-03

### GPT解析

### 总结

论文提出了一种新的方法TS-Reasoner，通过将时间序列基础模型的潜在表示与大型语言模型的文本输入对齐，解决了时间序列推理中的挑战，实现了高效且准确的时间序列理解和推理。

### 背景

时间序列推理在金融、能源使用、交通、天气和科学发现等多个领域的决策中至关重要。现有的时间序列基础模型(TSFMs)能够捕获低级动态模式并提供准确的预测，但通常需要额外的背景知识和复杂的推理能力，而这些是大多数TSFMs所缺乏的。大型语言模型(LLMs)可以实现这些推理能力，但如果没有昂贵的后训练，LLMs通常难以理解时间序列数据的数值方面。

### 目的

提出一种有效的方法来整合TSFMs和LLMs，使两种模态在推理任务上保持一致。开发TS-Reasoner，将TSFMs的潜在表示与LLMs的文本输入对齐，用于下游理解/推理任务。

### 方法

提出一种简单但有效的方法，为对齐训练整理多样化、合成的时间序列和文本标题对。开发一个两阶段训练方案，在对齐预训练后应用指令微调。与现有训练LLM接受时间序列作为输入的工作不同，他们利用预训练的TSFM并在训练期间冻结它。

### 主要发现

在多个基准测试上的大量实验表明，TS-Reasoner不仅优于各种现有的LLMs、视觉语言模型(VLMs)和时间序列LLMs。TS-Reasoner以显著的数据效率实现了这一优势，例如使用不到一半的训练数据。

### 结论

TS-Reasoner成功地将时间序列基础模型与大型语言模型结合起来，实现了高效的时间序列推理。

### 翻译

时间序列推理对金融、能源使用、交通、天气和科学发现等不同领域的决策至关重要。虽然现有的时间序列基础模型(TSFMs)可以捕获低级动态模式并提供准确的预测，但进一步分析通常需要额外的背景知识和复杂的推理能力，这些能力在大多数TSFMs中缺乏，但可以通过大型语言模型(LLMs)实现。另一方面，如果没有昂贵的后训练，LLMs通常难以理解时间序列数据的数值方面。虽然直观地整合这两种类型的模型是可行的，但开发有效的训练方法来使两种模态在推理任务上保持一致仍然是一个开放的挑战。为此，我们提出了TS-Reasoner，它将TSFMs的潜在表示与LLMs的文本输入对齐，用于下游理解/推理任务。具体来说，我们提出了一种简单而有效的方法，为对齐训练整理多样化、合成的时间序列和文本标题对。然后，我们开发了一个两阶段训练方案，在对齐预训练后应用指令微调。与现有训练LLM接受时间序列作为输入的工作不同，我们利用预训练的TSFM并在训练期间冻结它。在几个基准测试上的大量实验表明，TS-Reasoner不仅优于各种现有的LLMs、视觉语言模型(VLMs)和时间序列LLMs，而且以显著的数据效率实现了这一优势，例如使用不到一半的训练数据。


### 论文摘要

Time series reasoning is crucial to decision-making in diverse domains, including finance, energy usage, traffic, weather, and scientific discovery. While existing time series foundation models (TSFMs) can capture low-level dynamic patterns and provide accurate forecasting, further analysis usually requires additional background knowledge and sophisticated reasoning, which are lacking in most TSFMs but can be achieved through large language models (LLMs). On the other hand, without expensive post-training, LLMs often struggle with the numerical understanding of time series data. Although it is intuitive to integrate the two types of models, developing effective training recipes that align the two modalities for reasoning tasks is still an open challenge. To this end, we propose TS-Reasoner that aligns the latent representations of TSFMs with the textual inputs of LLMs for downstream understanding/reasoning tasks. Specifically, we propose a simple yet effective method to curate diverse, synthetic pairs of time series and textual captions for alignment training. We then develop a two-stage training recipe that applies instruction finetuning after the alignment pretraining. Unlike existing works that train an LLM to take time series as inputs, we leverage a pretrained TSFM and freeze it during training. Extensive experiments on several benchmarks demonstrate that TS-Reasoner not only outperforms a wide range of prevailing LLMs, Vision Language Models (VLMs), and Time Series LLMs, but also achieves this with remarkable data efficiency, e.g., using less than half the training data.

---

## 98. Foundation models for equation discovery in high energy physics

**论文链接:** [http://arxiv.org/abs/2510.03397v1](http://arxiv.org/abs/2510.03397v1)

**作者:** Manuel Morales-Alvarado

**发布时间:** 2025-10-03

**备注:** 6 pages

### GPT解析

### 总结

本研究展示了大型语言模型在高能物理符号回归任务中的应用潜力，特别是在发现已知没有封闭形式表达式的观测量的函数形式方面。

### 背景

基础模型是在广泛、多模态数据集上训练的大型机器学习模型，在科学应用中受到越来越多的关注。大型语言模型作为基础模型的突出实例，在文本和图像生成等任务上取得了显著成功。

### 目的

研究基础模型（特别是大型语言模型）在高能物理中方程发现的潜力，专注于符号回归方法。

### 方法

应用LLM-SR方法，研究轻子角分布方程恢复的基准问题，以及大型强子对撞机电弱玻色子产生中角系数函数形式的发现。

### 主要发现

LLM-SR能够在领域内和领域外的运动学区域中发现紧凑、准确且可解释的方程，能够有效融入嵌入的科学知识。

### 结论

为高能物理中的方程发现提供了有前景的新方法。

### 翻译

基础模型，即在广泛、多模态数据集上训练的大型机器学习模型，由于其能在多样化的下游任务上表现出色，在科学应用中正获得越来越多的关注。大型语言模型作为基础模型的突出实例，已在文本和图像生成等任务上取得了显著成功。在这项工作中，我们研究了它们在高能物理中方程发现的潜力，专注于符号回归。我们将LLM-SR方法应用于轻子角分布方程恢复的基准问题，以及大型强子对撞机电弱玻色子产生中角系数函数形式的发现，这些是具有高现象学相关性的可观测量，对于这些量，目前还不知道从第一性原理得出的封闭形式表达式。我们的结果表明，LLM-SR能够在领域内和领域外的运动学区域中发现紧凑、准确且可解释的方程，有效融入嵌入的科学知识，为高能物理中的方程发现提供了有前景的新方法。


### 论文摘要

Foundation models, large machine learning models trained on broad, multimodal datasets, have been gaining increasing attention in scientific applications due to their strong performance on diverse downstream tasks. Large Language Models (LLMs), a prominent instance of foundation models, have achieved remarkable success in tasks such as text and image generation. In this work, we investigate their potential for equation discovery in high energy physics, focusing on symbolic regression. We apply the LLM-SR methodology both to benchmark problems of equation recovery in lepton angular distributions and to the discovery of functional forms for angular coefficients in electroweak boson production at the Large Hadron Collider, observables of high phenomenological relevance for which no closed-form expressions are known from first principles. Our results demonstrate that LLM-SR can uncover compact, accurate, and interpretable equations across in-domain and out-of-domain kinematic regions, effectively incorporating embedded scientific knowledge and offering a promising new approach to equation discovery in high energy physics.

---

