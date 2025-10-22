# 今日论文推荐 - 2025-10-22

共 55 篇论文

---

## 1. GBlobs: Local LiDAR Geometry for Improved Sensor Placement Generalization

**论文链接:** [http://arxiv.org/abs/2510.18539v1](http://arxiv.org/abs/2510.18539v1)

**作者:** Dušan Malić, Christian Fruhwirth-Reisinger, Alexander Prutsch, Wei Lin, Samuel Schulter, Horst Possegger

**发布时间:** 2025-10-21

**备注:** 1st place at the IROS'25 RoboSense Challenge, Track #3: Cross-Sensor  Placement 3D Object Detection

### GPT解析

### 总结

这篇技术报告概述了RoboSense 2025:Track 3的顶级解决方案，在各种传感器配置下实现了3D目标检测的最先进性能。

### 背景

基于LiDAR的3D检测器在使用传统全局特征（即绝对笛卡尔坐标）进行训练时，常常受到'几何捷径'的影响，导致模型主要依赖物体绝对位置而非形状和外观特征，限制了在不同传感器配置下的泛化能力。

### 目的

开发一种方法来克服3D目标检测中的几何捷径问题，提高模型在不同传感器配置下的泛化能力。

### 方法

使用GBlobs（一种局部点云特征描述符）作为网络输入特征，有效绕过几何捷径，迫使网络学习强大的、以物体为中心的表示。

### 主要发现

通过GBlobs方法，模型能够学习更加鲁棒的物体中心表示，显著提高了模型泛化能力，在本挑战中展示了卓越性能。

### 结论

GBlobs方法成功解决了3D目标检测中的几何捷径问题，使模型能够更好地适应不同的传感器配置，在各种条件下实现优异的3D目标检测性能。

### 翻译

这篇技术报告概述了RoboSense 2025:Track 3的顶级解决方案，在各种传感器配置下实现了3D目标检测的最先进性能。我们的提交使用了GBlobs，这是一种局部点云特征描述符，专门设计用于增强模型在不同LiDAR配置上的泛化能力。当前的基于LiDAR的3D检测器在使用传统全局特征（即绝对笛卡尔坐标）进行训练时，常常受到'几何捷径'的影响。这引入了位置偏差，导致模型主要依赖物体的绝对位置，而不是区分形状和外观特征。虽然对领域内数据有效，但这种捷径在遇到不同点分布时（例如由不同传感器放置引起的）会严重限制泛化能力。通过使用GBlobs作为网络输入特征，我们有效地绕过了这种几何捷径，迫使网络学习强大的、以物体为中心的表示。这种方法显著提高了模型的泛化能力，从而在本挑战中展示了卓越的性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决激光雷达3D物体检测模型在不同传感器位置下泛化能力不足的问题。现有模型过度依赖物体的绝对位置信息，当传感器位置改变时，模型性能会大幅下降。这个问题在现实中很重要，因为自动驾驶车辆可能需要使用不同位置配置的激光雷达传感器，如果模型无法适应这种变化，每次更换传感器位置都需要重新训练，限制了系统的灵活性和实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者通过分析现有检测器的局限性（即'几何捷径'问题）来设计方法。他们发现使用全局坐标导致模型只关注物体位置而非形状特征，因此决定采用局部几何信息。作者借鉴了已有的GBlobs表示方法，将点云局部区域表示为高斯斑点，通过均值和协方差描述局部结构。同时，考虑到激光雷达数据在远距离的稀疏性，设计了双模型方法：一个基于GBlobs处理近距离，一个基于全局坐标处理远距离。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用局部几何特征替代全局坐标，迫使模型学习物体的形状和外观特征而非绝对位置，从而提高泛化能力。实现流程：1）将点云局部区域表示为GBlobs；2）分别训练基于GBlobs的主模型和基于全局坐标的辅助模型；3）推理时对输入应用测试时增强（随机旋转、翻转和缩放）；4）两个模型分别进行推理，然后反转增强并应用非极大值抑制；5）基于30米距离阈值融合结果：近距离用GBlobs模型，远距离用全局坐标模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1）将GBlobs作为输入特征解决几何捷径问题；2）设计双模型融合策略处理数据稀疏性；3）应用测试时增强技术提高鲁棒性；4）基于距离的预测融合方法。相比之前工作：传统检测器依赖全局坐标，而本文使用局部几何特征减轻位置偏见；之前工作未专门针对传感器位置变化优化；本文采用双模型而非单一模型适应不同距离的检测需求。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于局部几何特征GBlobs的3D物体检测方法，通过减轻几何捷径问题和采用双模型融合策略，显著提高了模型在不同传感器位置下的泛化能力。'}


### 论文摘要

This technical report outlines the top-ranking solution for RoboSense 2025: Track 3, achieving state-of-the-art performance on 3D object detection under various sensor placements. Our submission utilizes GBlobs, a local point cloud feature descriptor specifically designed to enhance model generalization across diverse LiDAR configurations. Current LiDAR-based 3D detectors often suffer from a \enquote{geometric shortcut} when trained on conventional global features (\ie, absolute Cartesian coordinates). This introduces a position bias that causes models to primarily rely on absolute object position rather than distinguishing shape and appearance characteristics. Although effective for in-domain data, this shortcut severely limits generalization when encountering different point distributions, such as those resulting from varying sensor placements. By using GBlobs as network input features, we effectively circumvent this geometric shortcut, compelling the network to learn robust, object-centric representations. This approach significantly enhances the model's ability to generalize, resulting in the exceptional performance demonstrated in this challenge.

---

## 2. ViSE: A Systematic Approach to Vision-Only Street-View Extrapolation

**论文链接:** [http://arxiv.org/abs/2510.18341v1](http://arxiv.org/abs/2510.18341v1)

**作者:** Kaiyuan Tan, Yingying Shen, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye

**发布时间:** 2025-10-21

### GPT解析

### 总结

该研究提出了一种在自动驾驶闭环仿真中进行真实视角外推的方法，在 ICCV 2025 RealADSim Workshop NVS 赛道中获得第一名。方法采用四阶段流水线，包括数据驱动初始化、几何先验注入、生成先验利用和数据驱动自适应，解决了街道视角外推的核心挑战，在 RealADSim-NVS 基准测试上获得 0.441 的最高分。

### 背景

自动驾驶闭环仿真中的真实视角外推是一个重要挑战，当前的 NovelView Synthesis (NVS) 方法在原始轨迹之外常常产生扭曲和不一致的图像。

### 目的

提出一个解决方案，解决街道视角外推的核心挑战，并在 RealADSim Workshop NVS 赛道中取得领先成绩。

### 方法

引入了一个全面的四阶段流水线：1) 使用数据驱动初始化策略生成稳健的伪激光雷达点云，避免局部最小值；2) 通过建模道路表面引入强几何先验，使用称为 2D-SDF 的新型降维 SDF；3) 利用生成先验为外推视点创建伪真实值，提供辅助监督；4) 使用数据驱动自适应网络移除时间特定伪影。

### 主要发现

在 RealADSim-NVS 基准测试上，该方法获得了 0.441 的最终得分，在所有参与者中排名第一。

### 结论

该方法成功解决了自动驾驶中视角外推的挑战，通过综合的四阶段流水线实现了高质量的视角外推，在基准测试中取得了最佳成绩。

### 翻译

真实的视角外推对自动驾驶的闭环仿真至关重要，然而对于当前的 NovelView Synthesis (NVS) 方法来说，这仍然是一个重大挑战，这些方法通常在原始轨迹之外产生扭曲和不一致的图像。本报告提出了我们的获奖解决方案，在 ICCV 2025 RealADSim Workshop NVS 赛道中荣获第一名。为解决街道视角外推的核心挑战，我们引入了一个全面的四阶段流水线。首先，我们采用数据驱动初始化策略生成稳健的伪激光雷达点云，避免局部最小值。其次，我们通过使用称为 2D-SDF 的新型降维 SDF 对道路表面建模，注入强几何先验。第三，我们利用生成先验为外推视点创建伪真实值，提供辅助监督。最后，数据驱动自适应网络移除时间特定伪影。在 RealADSim-NVS 基准测试上，我们的方法获得了 0.441 的最终得分，在所有参与者中排名第一。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶场景中的街景视角外推问题，即从原始轨迹之外生成新视角的图像。这个问题在现实中很重要，因为高保真的模拟是验证自动驾驶算法的关键技术，而传统模拟器存在领域差距，真实日志回放又无法支持交互式闭环评估。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过系统性分析街景外推的核心挑战，设计了一个四阶段流程。他们借鉴了3D高斯泼溅(3DGS)和神经辐射场(NeRF)等体积基元方法，但发现这些方法在外推视角下存在几何扭曲问题。同时参考了结构化运动(SfM)、视觉几何变换(VGGT)技术，并引入了生成模型如Difix3D+作为伪真实数据生成器，创新性地提出了2D-SDF来专门处理道路表面。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过系统化的四阶段流程实现鲁棒且几何一致的街景外推。整体流程包括：(1)使用VGGT生成视觉伪激光雷达点云进行3D场景初始化；(2)使用2D-SDF表示道路表面，结合3D高斯泼溅表示地面以上物体；(3)利用扩散模型为外推视角生成伪真实数据，提供对未观察区域的监督；(4)训练时不变自适应网络去除时间特定特征，确保在不同条件下保持一致的渲染结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：(1)鲁棒的无需激光雷达的初始化策略，避免局部最小值问题；(2)创新的2D-SDF表示，强制道路表面为局部平面先验；(3)迭代式伪真实数据框架，利用生成先验为未观察区域提供监督；(4)时不变自适应网络，去除时间特定特征。相比之前的工作，这个方法特别关注外推而非插值，同时结合了几何约束和生成模型的优势，既保证了几何一致性，又提供了视觉真实感。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种系统化的四阶段视觉街景外推方法，通过结合几何先验和生成模型，实现了在自动驾驶场景中从原始轨迹外推到新视角的鲁棒且几何一致的图像合成。'}


### 论文摘要

Realistic view extrapolation is critical for closed-loop simulation in autonomous driving, yet it remains a significant challenge for current Novel View Synthesis (NVS) methods, which often produce distorted and inconsistent images beyond the original trajectory. This report presents our winning solution which ctook first place in the RealADSim Workshop NVS track at ICCV 2025. To address the core challenges of street view extrapolation, we introduce a comprehensive four-stage pipeline. First, we employ a data-driven initialization strategy to generate a robust pseudo-LiDAR point cloud, avoiding local minima. Second, we inject strong geometric priors by modeling the road surface with a novel dimension-reduced SDF termed 2D-SDF. Third, we leverage a generative prior to create pseudo ground truth for extrapolated viewpoints, providing auxilary supervision. Finally, a data-driven adaptation network removes time-specific artifacts. On the RealADSim-NVS benchmark, our method achieves a final score of 0.441, ranking first among all participants.

---

## 3. BlendCLIP: Bridging Synthetic and Real Domains for Zero-Shot 3D Object Classification with Multimodal Pretraining

**论文链接:** [http://arxiv.org/abs/2510.18244v1](http://arxiv.org/abs/2510.18244v1)

**作者:** Ajinkya Khoche, Gergő László Nagy, Maciej Wozniak, Thomas Gustafsson, Patric Jensfelt

**发布时间:** 2025-10-21

**备注:** Under Review

### GPT解析

### 总结

本文提出了一种名为BlendCLIP的多模态预训练框架，用于解决零样本3D物体分类中合成数据与真实数据之间的领域差距问题，实现了在自动驾驶等实际应用中的高效3D物体识别。

### 背景

零样本3D物体分类对自动驾驶等实际应用至关重要，但面临合成训练数据与真实世界稀疏、嘈杂的LiDAR扫描数据之间的显著领域差距。仅使用合成数据训练的方法无法泛化到户外场景，而仅使用真实数据训练的方法缺乏语义多样性，难以识别罕见或未见过的物体。

### 目的

开发一种能够弥合合成数据与真实数据之间领域差距的方法，实现高效准确的零样本3D物体分类，特别是在自动驾驶等实际应用场景中。

### 方法

提出BlendCLIP多模态预训练框架，通过以下步骤实现：1) 从真实世界驾驶数据和人工标注的3D框中生成大规模物体级三元组数据集（点云、图像和文本描述）；2) 采用基于课程的数据混合策略，首先将模型建立在语义丰富的合成CAD数据基础上，然后逐步适应真实世界扫描的特性。

### 主要发现

实验表明该方法具有很高的标签效率：每批次仅引入1.5%的真实世界样本就能将nuScenes基准上的零样本准确率提高27%。最终模型在nuScenes和TruckScenes等户外数据集上实现了最先进的性能，比最佳先前方法提高了19.3%，同时在多样化合成基准上保持强大泛化能力。

### 结论

有效的领域适应而非大规模真实世界标注是解锁强大开放词汇3D感知的关键。该研究为解决3D物体识别中的领域差距提供了新思路，代码和数据集将在论文被接受后发布。

### 翻译

零样本3D物体分类对自动驾驶等实际应用至关重要，但常常受到用于训练的合成数据与真实世界中遇到的稀疏、嘈杂的LiDAR扫描之间的显著领域差距的阻碍。仅使用合成数据训练的方法无法泛化到户外场景，而仅使用真实数据训练的方法缺乏语义多样性，无法识别罕见或未见过的物体。我们引入了BlendCLIP，一个多模态预训练框架，通过战略性地结合两个领域的优势来弥合这一合成到真实的差距。我们首先提出了一种管道，从真实世界驾驶数据和人工标注的3D框中直接生成大规模物体级三元组数据集——包括点云、图像和文本描述。我们的核心贡献是基于课程的数据混合策略，首先将模型建立在语义丰富的合成CAD数据基础上，然后逐步使其适应真实世界扫描的特定特性。实验表明我们的方法具有很高的标签效率：每批次引入仅1.5%的真实世界样本就能将nuScenes基准上的零样本准确率提高27%。因此，我们的最终模型在nuScenes和TruckScenes等具有挑战性的户外数据集上实现了最先进的性能，在nuScenes上比最佳先前方法提高了19.3%，同时在多样化的合成基准上保持强大的泛化能力。我们的研究结果表明，有效的领域适应而非大规模真实世界标注是解锁强大开放词汇3D感知的关键。我们的代码和数据集将在论文被接受后发布在https://github.com/kesu1/BlendCLIP。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决零样本3D物体分类中的合成数据（如CAD模型）和真实世界数据（如LiDAR扫描）之间的域差距问题。这个问题在自动驾驶等现实应用中非常重要，因为系统需要识别各种可能遇到的物体，包括训练中未见过的类别。现有方法要么完全依赖合成数据但无法很好地泛化到真实世界的稀疏、嘈杂环境，要么完全依赖真实数据但缺乏语义多样性来识别罕见物体。同时，真实世界数据标注成本高昂，大规模标注不现实。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有方法的两条路径各有局限：纯合成数据无法泛化到真实世界，而纯真实数据缺乏语义多样性。因此他们思考如何结合两种数据的优势。设计方法时借鉴了ULIP-2的预训练策略，学习3D编码器并与CLIP嵌入对齐；采用了课程学习思想，从简单到复杂逐步引入数据；参考了多模态表示学习，使用点云-图像-文本三元组进行训练。核心思路是先让模型在语义丰富的合成数据上建立基础，然后逐步适应真实世界数据特性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用课程学习策略，先在语义丰富的合成CAD数据上建立基础，然后逐步适应真实世界LiDAR数据的特性，通过战略性地结合两个领域优势来弥合域差距。整体流程包括：1)数据准备，使用合成数据集和从真实数据集中构建的三元组；2)三元组生成，通过多扫描融合和运动补偿提取密集点云，投影边界框获取图像，用视觉-语言模型生成描述；3)课程训练，先仅用合成数据训练，再逐步引入真实世界数据；4)评估，在合成和真实数据集上评估零样本分类性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)新的课程学习数据混合策略；2)构建大规模户外三元组数据集；3)展示只需少量真实世界样本就能显著提高性能；4)在多个数据集上实现最先进的零样本分类。相比之前工作的不同：与纯合成数据方法相比能更好泛化到真实世界；与纯真实数据方法相比保留更好的语义多样性；相比简单数据混合策略避免了模型过度拟合到域标识符；相比需要大量标注的方法标注效率高，只需少量真实样本。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'BlendCLIP通过课程学习策略战略性地结合合成数据和真实世界数据，以最小的标注成本实现了强大的零样本3D物体分类，显著缩小了合成到真实世界的域差距。'}


### 论文摘要

Zero-shot 3D object classification is crucial for real-world applications like autonomous driving, however it is often hindered by a significant domain gap between the synthetic data used for training and the sparse, noisy LiDAR scans encountered in the real-world. Current methods trained solely on synthetic data fail to generalize to outdoor scenes, while those trained only on real data lack the semantic diversity to recognize rare or unseen objects.   We introduce BlendCLIP, a multimodal pretraining framework that bridges this synthetic-to-real gap by strategically combining the strengths of both domains. We first propose a pipeline to generate a large-scale dataset of object-level triplets -- consisting of a point cloud, image, and text description -- mined directly from real-world driving data and human annotated 3D boxes. Our core contribution is a curriculum-based data mixing strategy that first grounds the model in the semantically rich synthetic CAD data before progressively adapting it to the specific characteristics of real-world scans.   Our experiments show that our approach is highly label-efficient: introducing as few as 1.5\% real-world samples per batch into training boosts zero-shot accuracy on the nuScenes benchmark by 27\%. Consequently, our final model achieves state-of-the-art performance on challenging outdoor datasets like nuScenes and TruckScenes, improving over the best prior method by 19.3\% on nuScenes, while maintaining strong generalization on diverse synthetic benchmarks. Our findings demonstrate that effective domain adaptation, not full-scale real-world annotation, is the key to unlocking robust open-vocabulary 3D perception. Our code and dataset will be released upon acceptance on https://github.com/kesu1/BlendCLIP.

---

## 4. A Generalizable Light Transport 3D Embedding for Global Illumination

**论文链接:** [http://arxiv.org/abs/2510.18189v1](http://arxiv.org/abs/2510.18189v1)

**作者:** Bing Xu, Mukund Varma T, Cheng Wang, Tzumao Li, Lifan Wu, Bartlomiej Wronski, Ravi Ramamoorthi, Marco Salvi

**发布时间:** 2025-10-21

### GPT解析

### 总结

该研究提出了一种可泛化的3D光传输嵌入方法，直接从3D场景配置近似全局光照，无需使用光栅化或路径追踪的线索。

### 背景

全局光照对真实感渲染至关重要，但由于模拟间接光传输的复杂性，计算成本很高。现有神经方法主要依赖场景优化，跨场景泛化努力停留在2D屏幕空间，存在视图不一致性和有限空间理解问题。

### 目的

开发一个可泛化的3D光传输嵌入方法，直接从3D场景配置近似全局光照，不使用光栅化或路径追踪的线索。

### 方法

将场景表示为具有几何和材质特征点云；使用可扩展transformer模型对全局点与点交互建模，编码为神经基元；渲染时通过最近邻搜索检索附近基元，通过交叉注意力聚合潜在特征预测渲染量。

### 主要发现

在多样化室内场景中实现了漫反射全局光照预测；训练的嵌入可通过有限微调快速适应新渲染任务；展示了有光泽材质的空间-方向辐射场估计初步结果；归一化场可加速无偏路径引导。

### 结论

该方法展示了将学习先验整合到渲染管道中的路径，无需显式光线追踪光照线索。

### 翻译

全局光照(GI)对于真实感渲染至关重要，但由于模拟间接光传输的复杂性，计算成本仍然很高。最近的神经方法主要依赖于场景优化，有时扩展到处理相机或几何变化。跨场景泛化的努力主要停留在2D屏幕空间，如神经去噪或基于G缓冲区的GI预测，这些方法常常存在视图不一致性和有限的空间理解问题。我们提出了一种可泛化的3D光传输嵌入，直接从3D场景配置近似全局光照，不使用光栅化或路径追踪的线索。每个场景表示为具有几何和材质特征点云。可扩展的transformer模型对全局点与点之间的交互进行建模，将这些特征编码为神经基元。渲染时，每个查询点通过最近邻搜索检索附近基元，并通过交叉注意力聚合它们的潜在特征，以预测所需的渲染量。我们在不同布局、几何形状和材质的多样化室内场景中展示了漫反射全局光照预测的结果。为辐照度估计训练的嵌入可以通过有限的微调快速适应新的渲染任务。我们还展示了针对有光泽材质的空间-方向辐射场估计的初步结果，并展示了归一化场如何加速无偏路径引导。这种方法展示了将学习先验整合到渲染管道中的路径，而无需显式的光线追踪光照线索。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决全局照明的通用化问题。当前的全局照明方法计算成本高昂，而现有神经方法通常针对单个场景优化，缺乏跨场景泛化能力。这个问题在游戏开发、电影制作、虚拟现实和建筑设计等领域至关重要，因为这些领域需要高质量的全局照明效果，但又面临实时渲染和资源限制的挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到光传输算子和注意力机制之间的相似性，启发他们使用transformer架构来模拟光传输。他们意识到传统方法在处理复杂场景时的局限性，特别是视角一致性和泛化能力问题。他们借鉴了transformer架构在处理长距离依赖关系上的成功、PointTransformerV3作为点云编码的基础架构、传统的光传输方程和渲染方程，以及irradiance caching等传统技术的思想，但用神经网络方法进行了改进和创新。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用3D点云作为场景的中间表示，利用transformer架构编码场景点之间的长距离交互来模拟光传输，并设计可学习的局部查询解码器通过注意力机制聚合邻近点特征。整体流程包括：将3D场景转换为点云表示；使用transformer编码器处理场景点生成光传输嵌入；对于渲染时的查询点，通过k近邻搜索获取邻近场景点嵌入；使用基于交叉注意力的解码器聚合特征并预测渲染量；使用路径追踪的地面真实值进行端到端训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出通用的3D光传输嵌入方法直接从3D场景配置近似全局照明；使用点云作为中间表示避免屏幕空间限制；设计可扩展的transformer编码器捕获长距离交互；提出基于交叉注意力的局部查询解码器实现自适应特征聚合；展示如何重用预训练编码器适应不同渲染任务。与之前工作不同：相比单场景优化的神经方法，实现了跨场景泛化；相比2D屏幕空间方法，保持了多视角一致性；相比传统预计算方法，学习通用嵌入可跨场景重用；针对光传输模拟专门优化了点云处理方法；在3D空间直接预测入射辐射场，而非在2D渲染图像上训练。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于transformer的通用3D光传输嵌入方法，可以直接从场景几何、材质和光源配置中学习并泛化到复杂室内场景的全局照明效果，无需针对每个场景进行优化。'}


### 论文摘要

Global illumination (GI) is essential for realistic rendering but remains computationally expensive due to the complexity of simulating indirect light transport. Recent neural methods have mainly relied on per-scene optimization, sometimes extended to handle changes in camera or geometry. Efforts toward cross-scene generalization have largely stayed in 2D screen space, such as neural denoising or G-buffer based GI prediction, which often suffer from view inconsistency and limited spatial understanding. We propose a generalizable 3D light transport embedding that approximates global illumination directly from 3D scene configurations, without using rasterized or path-traced cues. Each scene is represented as a point cloud with geometric and material features. A scalable transformer models global point-to-point interactions to encode these features into neural primitives. At render time, each query point retrieves nearby primitives via nearest-neighbor search and aggregates their latent features through cross-attention to predict the desired rendering quantity. We demonstrate results on diffuse global illumination prediction across diverse indoor scenes with varying layouts, geometry, and materials. The embedding trained for irradiance estimation can be quickly adapted to new rendering tasks with limited fine-tuning. We also present preliminary results for spatial-directional radiance field estimation for glossy materials and show how the normalized field can accelerate unbiased path guiding. This approach highlights a path toward integrating learned priors into rendering pipelines without explicit ray-traced illumination cues.

---

## 5. HyperDiffusionFields (HyDiF): Diffusion-Guided Hypernetworks for Learning Implicit Molecular Neural Fields

**论文链接:** [http://arxiv.org/abs/2510.18122v1](http://arxiv.org/abs/2510.18122v1)

**作者:** Sudarshan Babu, Phillip Lo, Xiao Zhang, Aadi Srivastava, Ali Davariashtiyani, Jason Perera, Michael Maire, Aly A. Khan

**发布时间:** 2025-10-20

### GPT解析

### 总结

介绍了一种名为HyperDiffusionFields (HyDiF)的框架，将3D分子构象建模为连续场而非离散原子坐标或图，使用分子方向场和分子神经场表示，并通过超网络和去噪扩散模型实现生成能力。

### 背景

传统分子建模方法使用离散原子坐标或图表示，而本文提出了一种基于连续场的建模方法。

### 目的

开发一种能够将3D分子构象表示为连续场的框架，支持分子生成和性质预测任务。

### 方法

使用分子方向场(MDF)将空间中的任何点映射到特定类型最近原子的方向；通过分子特定的神经隐式场(MNF)表示MDF；采用超网络条件化为分子生成MNF权重；将超网络训练为去噪扩散模型以实现生成能力；扩展到掩码扩散机制支持结构条件生成任务。

### 主要发现

该方法能够支持空间细粒度特征提取，这是基于图或点云的方法难以实现的；该方法可扩展到更大的生物分子。

### 结论

基于场的分子建模是一个有前途的方向，能够有效处理分子生成和性质预测任务。

### 翻译

我们引入了HyperDiffusionFields (HyDiF)框架，将3D分子构象建模为连续场而非离散原子坐标或图。我们方法的核心是分子方向场(MDF)，它将空间中的任何点映射到特定类型最近原子的方向。我们使用分子特定的神经隐式场表示MDF，称为分子神经场(MNF)。为了实现跨分子学习和促进泛化，我们采用了一种方法，其中共享的超网络条件化为分子，生成给定分子MNF的权重。为了赋予模型生成能力，我们将超网络训练为去噪扩散模型，能够在分子场的函数空间中进行采样。我们的设计自然扩展到掩码扩散机制，通过选择性对场区域加噪来支持结构条件生成任务，如分子修复。除了生成任务外，MDF的局部和连续特性使得能够进行空间细粒度的特征提取用于分子性质预测，这是基于图或点云的方法难以实现的。此外，我们证明了我们的方法可以扩展到更大的生物分子，展示了基于场的分子建模的一个有前途的方向。


### 论文摘要

We introduce HyperDiffusionFields (HyDiF), a framework that models 3D molecular conformers as continuous fields rather than discrete atomic coordinates or graphs. At the core of our approach is the Molecular Directional Field (MDF), a vector field that maps any point in space to the direction of the nearest atom of a particular type. We represent MDFs using molecule-specific neural implicit fields, which we call Molecular Neural Fields (MNFs). To enable learning across molecules and facilitate generalization, we adopt an approach where a shared hypernetwork, conditioned on a molecule, generates the weights of the given molecule's MNF. To endow the model with generative capabilities, we train the hypernetwork as a denoising diffusion model, enabling sampling in the function space of molecular fields. Our design naturally extends to a masked diffusion mechanism to support structure-conditioned generation tasks, such as molecular inpainting, by selectively noising regions of the field. Beyond generation, the localized and continuous nature of MDFs enables spatially fine-grained feature extraction for molecular property prediction, something not easily achievable with graph or point cloud based methods. Furthermore, we demonstrate that our approach scales to larger biomolecules, illustrating a promising direction for field-based molecular modeling.

---

## 6. Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework

**论文链接:** [http://arxiv.org/abs/2510.18825v1](http://arxiv.org/abs/2510.18825v1)

**作者:** Yujie Xing, Xiao Wang, Bin Wu, Hai Huang, Chuan Shi

**发布时间:** 2025-10-21

**备注:** Accepted by NeurIPS 2025 (Poster)

### GPT解析

### 总结

本文提出了一种统一的分层掩码框架，揭示了模型架构和注意力掩码构建之间的等价性，并基于此设计了M3Dphormer模型，该模型结合了多级掩码和双注意力计算，在多个基准测试上实现了最先进的性能。

### 背景

图变换器(Graph Transformers)在图表示学习中表现出色，但现有方法往往依赖针对特定交互的复杂架构设计，缺乏灵活性。

### 目的

解决现有图变换器架构灵活性不足的问题，提出一个统一的框架来建模多样化的节点交互。

### 方法

提出统一的分层掩码框架，揭示模型架构与注意力掩码构建的等价性；设计M3Dphormer模型，包含三种理论支持的分层掩码和双层专家路由机制；引入双注意力计算方案，根据局部掩码稀疏性动态切换模式。

### 主要发现

正确分类的概率与感受野大小和标签一致性呈正相关；有效的注意力掩码应确保足够大的感受野和高水平的标签一致性；分层掩码在不同场景下具有互补优势。

### 结论

M3Dphormer通过统一框架和多级掩码设计有效解决了图变换器的灵活性限制，在多个基准测试上实现了最先进的性能，验证了所提方法的有效性。

### 翻译

图变换器(Graph Transformers)因其能够建模多样化的节点交互，已成为图表示学习的强大范式。然而，现有的图变换器通常依赖于针对特定交互的复杂架构设计，限制了其灵活性。为此，我们提出了一个统一的分层掩码框架，揭示了模型架构与注意力掩码构建之间的基本等价性。该框架通过精心设计的注意力掩码捕获多样化交互，实现了一致性的建模范式。在该框架下的理论分析表明，正确分类的概率与感受野大小和标签正相关，导致了一个基本设计原则：有效的注意力掩码应确保足够大的感受野和高水平的标签一致性。虽然没有单一现有的掩码能在所有场景下满足这一原则，但我们的分析显示分层掩码提供了互补优势，促使它们的有效集成。随后，我们引入了M3Dphormer，这是一种基于专家混合的图变换器，具有多级掩码和双注意力计算。M3Dphormer包含三种理论支持的分层掩码，并采用双层专家路由机制来自适应地集成多级交互信息。为确保可扩展性，我们进一步引入了双注意力计算方案，根据局部掩码稀疏性在密集和稀疏模式之间动态切换。在多个基准测试上的广泛实验证明，M3Dphormer达到了最先进的性能，验证了我们的统一框架和模型设计的有效性。


### 论文摘要

Graph Transformers (GTs) have emerged as a powerful paradigm for graph representation learning due to their ability to model diverse node interactions. However, existing GTs often rely on intricate architectural designs tailored to specific interactions, limiting their flexibility. To address this, we propose a unified hierarchical mask framework that reveals an underlying equivalence between model architecture and attention mask construction. This framework enables a consistent modeling paradigm by capturing diverse interactions through carefully designed attention masks. Theoretical analysis under this framework demonstrates that the probability of correct classification positively correlates with the receptive field size and label consistency, leading to a fundamental design principle: an effective attention mask should ensure both a sufficiently large receptive field and a high level of label consistency. While no single existing mask satisfies this principle across all scenarios, our analysis reveals that hierarchical masks offer complementary strengths, motivating their effective integration. Then, we introduce M3Dphormer, a Mixture-of-Experts-based Graph Transformer with Multi-Level Masking and Dual Attention Computation. M3Dphormer incorporates three theoretically grounded hierarchical masks and employs a bi-level expert routing mechanism to adaptively integrate multi-level interaction information. To ensure scalability, we further introduce a dual attention computation scheme that dynamically switches between dense and sparse modes based on local mask sparsity. Extensive experiments across multiple benchmarks demonstrate that M3Dphormer achieves state-of-the-art performance, validating the effectiveness of our unified framework and model design.

---

## 7. Exploring a Unified Vision-Centric Contrastive Alternatives on Multi-Modal Web Documents

**论文链接:** [http://arxiv.org/abs/2510.18703v1](http://arxiv.org/abs/2510.18703v1)

**作者:** Yiqi Lin, Alex Jinpeng Wang, Linjie Li, Zhengyuan Yang, Mike Zheng Shou

**发布时间:** 2025-10-21

**备注:** Project page: this https://linyq17.github.io/VC2L/

### GPT解析

### 总结

VC2L是一种统一的以视觉为中心的对比学习框架，通过在像素空间操作处理文本、图像及其组合，解决了传统对比视觉语言模型处理复杂网页文档的局限性，并在多个基准测试中表现出色。

### 背景

对比视觉语言模型如CLIP通过学习对齐的图像-文本对在各种多模态任务中表现出色，但它们处理复杂的真实世界网页文档的能力仍然有限，特别是在文本和图像交错、松散对齐或嵌入视觉形式的情况下。

### 目的

解决对比视觉语言模型处理复杂网页文档的局限性，提出一种统一的框架来有效处理文本、图像及其组合。

### 方法

提出以视觉为中心的对比学习（VC2L）框架，使用单一的视觉变换器建模文本、图像及其组合；完全在像素空间操作，将所有输入渲染为图像，消除OCR、文本分词或模态融合的需要；采用片段级对比学习目标，对齐连续的多模态片段，利用文档的固有连贯性。

### 主要发现

引入了三个检索基准：AnyCIR（跨模态检索）、SeqCIR（细粒度顺序理解）和CSR（泛化到未见数据）；实验结果表明，VC2L在提出的基准和已建立的datasets（如M-BEIR和MTEB）上与CLIP风格模型相比具有竞争性或更优的性能。

### 结论

多模态网页数据作为对比学习宝贵训练资源具有潜力；统一的、以视觉为中心的方法在多模态表示学习中具有良好的可扩展性。

### 翻译

对比视觉语言模型（如CLIP）通过学习对齐的图像-文本对在各种多模态任务中表现出色。然而，它们处理复杂的真实世界网页文档的能力仍然有限，特别是在文本和图像交错、松散对齐或嵌入视觉形式的情况下。为解决这些挑战，我们提出了以视觉为中心的对比学习（VC2L），这是一个统一框架，使用单一的视觉变换器对文本、图像及其组合进行建模。VC2L完全在像素空间操作，通过将所有输入（文本、视觉或组合）渲染为图像，从而消除了OCR、文本分词或模态融合策略的需要。为捕捉多模态网页文档中的复杂跨模态关系，VC2L采用片段级对比学习目标，对齐连续的多模态片段，利用文档的固有连贯性，不需要明确配对的图像-文本数据。为评估这种方法的有效性，我们引入了三个检索基准：AnyCIR、SeqCIR和CSR，分别用于评估跨模态检索、细粒度顺序理解和泛化到未见数据的能力。实验结果表明，与CLIP风格模型相比，VC2L在提出的基准和已建立的datasets（如M-BEIR和MTEB）上取得了竞争性或更优的性能。这些发现强调了多模态网页数据作为对比学习宝贵训练资源的潜力，并说明了统一的、以视觉为中心的方法在多模态表示学习中的可扩展性。代码和模型可在https://github.com/showlab/VC2L获取。


### 论文摘要

Contrastive vision-language models such as CLIP have demonstrated strong performance across a wide range of multimodal tasks by learning from aligned image-text pairs. However, their ability to handle complex, real-world web documents remains limited, particularly in scenarios where text and images are interleaved, loosely aligned, or embedded in visual form. To address these challenges, we propose Vision-Centric Contrastive Learning (VC2L), a unified framework that models text, images, and their combinations using a single vision transformer. VC2L operates entirely in pixel space by rendering all inputs, whether textual, visual, or combined, as images, thus eliminating the need for OCR, text tokenization, or modality fusion strategy. To capture complex cross-modal relationships in multimodal web documents, VC2L employs a snippet-level contrastive learning objective that aligns consecutive multimodal segments, leveraging the inherent coherence of documents without requiring explicitly paired image-text data. To assess the effectiveness of this approach, we introduce three retrieval benchmarks, AnyCIR, SeqCIR, and CSR, designed to evaluate cross-modal retrieval, fine-grained sequential understanding, and generalization to unseen data, respectively. Empirical results show that VC2L achieves competitive or superior performance compared to CLIP-style models on both the proposed benchmarks and established datasets such as M-BEIR and MTEB. These findings underscore the potential of multimodal web data as a valuable training resource for contrastive learning and illustrate the scalability of a unified, vision-centric approach for multimodal representation learning. Code and models are available at: https://github.com/showlab/VC2L.

---

## 8. A Stage-Wise Learning Strategy with Fixed Anchors for Robust Speaker Verification

**论文链接:** [http://arxiv.org/abs/2510.18530v1](http://arxiv.org/abs/2510.18530v1)

**作者:** Bin Gu, Lipeng Dai, Huipeng Du, Haitao Zhao, Jibo Wei

**发布时间:** 2025-10-21

### GPT解析

### 总结

本文提出了一种基于锚点的分阶段学习策略，用于在嘈杂条件下学习稳健的说话人表示。

### 背景

在嘈杂条件下学习稳健的说话人表示面临重大挑战，需要谨慎处理区分性和噪声不变性。

### 目的

开发一种能够在嘈杂环境中保持说话人识别准确性的方法。

### 方法

采用基于锚点的分阶段学习策略，首先训练基础模型建立区分性的说话人边界，然后从模型中提取锚嵌入作为稳定参考，最后在嘈杂输入上对基础模型的副本进行微调，通过强制接近固定的锚嵌入来保持失真情况下的说话人身份。

### 主要发现

这种策略相比传统的联合优化方法有优势，特别是在保持区分性的同时提高噪声鲁棒性。

### 结论

提出的方法在各种噪声条件下都表现出一致的改进，这可能是由于其能够分别处理边界稳定性和变化抑制。

### 翻译

在嘈杂条件下学习稳健的说话人表示面临重大挑战，这需要谨慎处理区分性和噪声不变性。在这项工作中，我们提出了一种基于锚点的分阶段学习策略用于稳健的说话人表示学习。具体来说，我们的方法首先训练一个基础模型来建立区分性的说话人边界，然后从该模型中提取锚嵌入作为稳定参考。最后，在嘈杂输入上对基础模型的副本进行微调，通过强制接近其对应的固定锚嵌入来保持失真情况下的说话人身份。实验结果表明，这种策略相比传统的联合优化方法具有优势，特别是在保持区分性的同时提高噪声鲁棒性。提出的方法在各种噪声条件下都表现出一致的改进，这可能是由于它能够分别处理边界稳定性和变化抑制。


### 论文摘要

Learning robust speaker representations under noisy conditions presents significant challenges, which requires careful handling of both discriminative and noise-invariant properties. In this work, we proposed an anchor-based stage-wise learning strategy for robust speaker representation learning. Specifically, our approach begins by training a base model to establish discriminative speaker boundaries, and then extract anchor embeddings from this model as stable references. Finally, a copy of the base model is fine-tuned on noisy inputs, regularized by enforcing proximity to their corresponding fixed anchor embeddings to preserve speaker identity under distortion. Experimental results suggest that this strategy offers advantages over conventional joint optimization, particularly in maintaining discrimination while improving noise robustness. The proposed method demonstrates consistent improvements across various noise conditions, potentially due to its ability to handle boundary stabilization and variation suppression separately.

---

## 9. Simple and Efficient Heterogeneous Temporal Graph Neural Network

**论文链接:** [http://arxiv.org/abs/2510.18467v1](http://arxiv.org/abs/2510.18467v1)

**作者:** Yili Wang, Tairan Huang, Changlong He, Qiutong Li, Jianliang Gao

**发布时间:** 2025-10-21

**备注:** Accepted by Neurips 2025

### GPT解析

### 总结

论文提出了一种简单高效的异质时序图神经网络(SE-HTGNN)，通过动态注意力机制将时间建模集成到空间学习中，并利用大型语言模型增强模型理解能力。

### 背景

异质时序图(HTGs)是现实世界中普遍存在的数据结构，现有基于注意力机制的神经网络方法采用解耦的时空学习范式，削弱了时空信息交互并导致模型复杂度高。

### 目的

开发一种新的HTGs学习范式，解决现有方法的时空信息交互弱和模型复杂度高的问题。

### 方法

通过创新的动态注意力机制将时间建模集成到空间学习中，保留历史快图的注意力信息指导后续计算；同时利用大型语言模型提示SE-HTGNN，捕获节点类型的隐含属性作为先验知识。

### 主要发现

SE-HTGNN比最先进和最新的基线方法快达10倍，同时保持最佳的预测准确性。

### 结论

SE-HTGNN在提高计算效率的同时保持了准确性，是一种有效的异质时序图表示学习方法。

### 翻译

异质时序图(HTGs)是现实世界中普遍存在的数据结构。最近，为了增强HTGs的表示学习，已提出许多基于注意力机制的神经网络。尽管取得了这些成功，现有方法依赖于解耦的时空学习范式，这削弱了时空信息的交互作用并导致模型复杂度高。为了弥合这一差距，我们提出了一种名为简单高效的异质时序图神经网络(SE-HTGNN)的新型HTGs学习范式。具体来说，我们通过创新的动态注意力机制将时间建模集成到空间学习中，该机制保留来自历史图快图的注意力信息以指导后续注意力计算，从而提高HTGs的整体判别性表示学习能力。此外，为了全面且自适应地理解HTGs，我们利用大型语言模型提示SE-HTGNN，使模型能够捕获节点类型的隐含属性作为先验知识。大量实验证明，SE-HTGNN比最先进和最新的基线方法快达10倍，同时保持最佳的预测准确性。


### 论文摘要

Heterogeneous temporal graphs (HTGs) are ubiquitous data structures in the real world. Recently, to enhance representation learning on HTGs, numerous attention-based neural networks have been proposed. Despite these successes, existing methods rely on a decoupled temporal and spatial learning paradigm, which weakens interactions of spatio-temporal information and leads to a high model complexity. To bridge this gap, we propose a novel learning paradigm for HTGs called Simple and Efficient Heterogeneous Temporal Graph N}eural Network (SE-HTGNN). Specifically, we innovatively integrate temporal modeling into spatial learning via a novel dynamic attention mechanism, which retains attention information from historical graph snapshots to guide subsequent attention computation, thereby improving the overall discriminative representations learning of HTGs. Additionally, to comprehensively and adaptively understand HTGs, we leverage large language models to prompt SE-HTGNN, enabling the model to capture the implicit properties of node types as prior knowledge. Extensive experiments demonstrate that SE-HTGNN achieves up to 10x speed-up over the state-of-the-art and latest baseline while maintaining the best forecasting accuracy.

---

## 10. ProLAP: Probabilistic Language-Audio Pre-Training

**论文链接:** [http://arxiv.org/abs/2510.18423v1](http://arxiv.org/abs/2510.18423v1)

**作者:** Toranosuke Manabe, Yuchi Ishikawa, Hokuto Munakata, Tatsuya Komatsu

**发布时间:** 2025-10-21

**备注:** Under review

### GPT解析

### 总结

本文提出了概率语言-音频预训练模型(ProLAP)，用于解决语言-音频关系中的多对多对应问题，通过概率分布扩散建模多样性，并引入层次包含损失和掩码排斥损失来提高学习效率。

### 背景

现有的语言-音频联合表征学习框架通常依赖确定性嵌入，假设音频和文本之间存在一一对应关系。然而在现实世界中，语言-音频关系本质上是多对多的：一个音频片段可以用多个字幕描述，反之亦然。

### 目的

解决语言-音频关系中的多对多对应问题，提出能够从小数据集中学习数据固有层次结构的模型。

### 方法

提出概率语言-音频预训练模型(ProLAP)，将多样性建模为联合语言-音频嵌入空间中概率分布的扩散。同时引入两个训练目标：层次包含损失促进对输入的语义层次理解，掩码排斥损失提高学习效率。

### 主要发现

ProLAP在音频-文本检索任务上优于现有的确定性方法。通过音频遍历任务实验，证明ProLAP能够捕捉合理的语义层次结构，即使从小数据集也能有效学习。

### 结论

ProLAP成功解决了语言-音频关系中的多对多对应问题，能够从小数据集中学习数据固有的层次结构，在检索任务和语义层次捕捉方面表现优异。

### 翻译

语言-音频联合表征学习框架通常依赖于确定性嵌入，假设音频和文本之间存在一一对应关系。然而在现实世界中，语言-音频关系本质上是多对多的：一个音频片段可以用多个字幕描述，反之亦然。为解决这一问题，我们提出了概率语言-音频预训练(ProLAP)，将多样性建模为联合语言-音频嵌入空间中概率分布的扩散。为有效训练模态内的层次关系，我们还引入了两个目标：(i)层次包含损失促进对输入的语义层次理解，(ii)掩码排斥损失优化层次包含损失时提高学习效率。通过这种训练策略，我们的模型能够从小数据集中学习数据固有的层次结构，这与依赖大规模数据集的先前概率方法形成对比。在我们的实验中，ProLAP在音频-文本检索任务上优于现有的确定性方法。此外，通过本文介绍的音频遍历任务实验，我们证明了ProLAP捕捉了合理的语义层次。


### 论文摘要

Language-audio joint representation learning frameworks typically depend on deterministic embeddings, assuming a one-to-one correspondence between audio and text. In real-world settings, however, the language-audio relationship is inherently many-to-many: one audio segment can be described by multiple captions and vice versa. To address this, we propose Probabilistic Language-Audio Pre-training (ProLAP), which models multiplicity as the spread of probability distributions in a joint language-audio embedding space. To train the intra-modal hierarchical relationship effectively, we also introduce two objectives: (i) hierarchical inclusion loss to promote semantic hierarchical understanding of inputs and (ii) mask repulsive loss to improve the efficiency of learning when optimizing the hierarchical inclusion loss. With this training strategy, our model can learn the hierarchical structure inherent in the data even from small datasets, in contrast to prior probabilistic approaches that rely on large-scale datasets. In our experiments, ProLAP outperforms existing deterministic approaches on audio-text retrieval tasks. Moreover, through experiments on the audio traversal task introduced in this paper, we demonstrate that ProLAP captures the plausible semantic hierarchy.

---

## 11. Towards Identifiability of Hierarchical Temporal Causal Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.18310v1](http://arxiv.org/abs/2510.18310v1)

**作者:** Zijian Li, Minghao Fu, Junxian Huang, Yifan Shen, Ruichu Cai, Yuewen Sun, Guangyi Chen, Kun Zhang

**发布时间:** 2025-10-21

### GPT解析

### 总结

本文提出了一种名为CHiLD的因果层次化潜在动态识别框架，能够有效捕捉时间序列数据中的多层次时间依赖关系，解决了现有方法无法从单时间步观测变量中恢复层次化潜在变量联合分布的问题。

### 背景

对时间序列数据中的层次化潜在动态进行建模对于捕捉现实世界任务中多层次的抽象时间依赖关系至关重要，但现有方法存在局限性。

### 目的

开发一种能够捕捉时间序列数据中层次化潜在动态的框架，解决现有时间序列因果表征学习方法的不足。

### 方法

提出CHiLD识别框架，首先利用时间上下文观测变量识别多层潜在变量的联合分布，然后利用层次结构的自然稀疏性识别每层内的潜在变量，并基于变分推断开发时间序列生成模型，包含上下文编码器和归一化流层次化先验网络。

### 主要发现

使用三个条件独立的观测可以唯一确定层次化潜在变量的联合分布，这一发现为构建新方法提供了理论基础。

### 结论

在合成和真实世界数据集上的经验评估验证了理论主张，证明了CHiLD在建模层次化潜在动态方面的有效性。

### 翻译

对时间序列数据背后的层次化潜在动态进行建模，对于捕捉现实世界任务中多层次抽象的时间依赖关系至关重要。然而，现有的时间序列因果表征学习方法无法捕捉此类动态，因为它们无法从单时间步观测变量中恢复层次化潜在变量的联合分布。有趣的是，我们发现使用三个条件独立的观测可以唯一确定层次化潜在变量的联合分布。基于这一见解，我们提出了一个因果层次化潜在动态（CHiLD）识别框架。我们的方法首先利用时间上下文观测变量来识别多层潜在变量的联合分布。随后，我们利用层次结构中潜在变量的自然稀疏性来识别每层内的潜在变量。在理论结果的指导下，我们开发了一个基于变分推断的时间序列生成模型。该模型包含一个上下文编码器来重建多层潜在变量，以及基于归一化流的层次化先验网络，以施加层次化潜在动态的独立噪声条件。在合成和真实世界数据集上的经验评估验证了我们的理论主张，并证明了CHiLD在建模层次化潜在动态方面的有效性。


### 论文摘要

Modeling hierarchical latent dynamics behind time series data is critical for capturing temporal dependencies across multiple levels of abstraction in real-world tasks. However, existing temporal causal representation learning methods fail to capture such dynamics, as they fail to recover the joint distribution of hierarchical latent variables from \textit{single-timestep observed variables}. Interestingly, we find that the joint distribution of hierarchical latent variables can be uniquely determined using three conditionally independent observations. Building on this insight, we propose a Causally Hierarchical Latent Dynamic (CHiLD) identification framework. Our approach first employs temporal contextual observed variables to identify the joint distribution of multi-layer latent variables. Sequentially, we exploit the natural sparsity of the hierarchical structure among latent variables to identify latent variables within each layer. Guided by the theoretical results, we develop a time series generative model grounded in variational inference. This model incorporates a contextual encoder to reconstruct multi-layer latent variables and normalize flow-based hierarchical prior networks to impose the independent noise condition of hierarchical latent dynamics. Empirical evaluations on both synthetic and real-world datasets validate our theoretical claims and demonstrate the effectiveness of CHiLD in modeling hierarchical latent dynamics.

---

## 12. Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.17959v1](http://arxiv.org/abs/2510.17959v1)

**作者:** Jeff Shen, Francois Lanusse, Liam Holden Parker, Ollie Liu, Tom Hehir, Leopoldo Sarra, Lucas Meyer, Micah Bowles, Sebastian Wagner-Carena, Sebastian Wagner-Carena, Helen Qu, Siavash Golkar, Alberto Bietti, Hatim Bourfoune, Nathan Cassereau, Pierre Cornette, Keiya Hirashima, Geraud Krawezik, Ruben Ohana, Nicholas Lourie, Michael McCabe, Rudy Morel, Payel Mukhopadhyay, Mariel Pettee, Bruno Régaldo-Saint Blancard, Kyunghyun Cho, Miles Cranmer, Shirley Ho

**发布时间:** 2025-10-20

**备注:** Accepted at NeurIPS 2025 Machine Learning and the Physical Sciences  Workshop

### GPT解析

### 总结

本文介绍了一种深度学习模型，能够以自监督方式学习异构天文光谱数据，生成统一且物理上有意义的表示，可作为天文领域基础模型的强大构建块，并可能扩展到其他科学领域。

### 背景

连续科学数据跨越多个分辨率和领域，将其统一为共同表示是开发科学基础模型的关键步骤。天文光谱体现了这一挑战：大规模调查已收集数百万个跨越广泛波长和分辨率的光谱，但分析仍分散在光谱域（如光学与红外）和天体类型（如恒星与星系）中，限制了跨数据集信息整合的能力。

### 目的

开发一个能够统一不同分辨率和领域光谱数据的模型，创建物理上有意义的表示，并适应各种下游任务。

### 方法

提出了一种深度学习模型，以自监督方式联合学习异构光谱数据。该通用光谱标记器直接在原生波长网格上处理多种天体类型和分辨率的光谱，产生内在对齐、同质且物理上有意义的表示。

### 主要发现

首次证明单个模型可以统一不同分辨率和领域的光谱数据。该模型能够高效适应，在各种下游任务中实现具有竞争力的性能。

### 结论

该模型可作为天文领域基础模型的强大构建块，并可能扩展到其他具有异构连续数据的科学领域，如气候和医疗保健。

### 翻译

连续科学数据跨越多个分辨率和领域，将其统一为共同表示是开发科学基础模型的关键步骤。天文光谱体现了这一挑战：大规模调查已收集了数百万个跨越广泛波长和分辨率的光谱，但分析仍分散在光谱域（如光学与红外）和天体类型（如恒星与星系）中，限制了跨数据集信息整合的能力。我们提出了一种深度学习模型，能够以自监督方式联合学习异构光谱数据。我们的通用光谱标记器直接在原生波长网格上处理多种天体类型和分辨率的光谱，产生内在对齐、同质且物理上有意义的表示，这些表示可以高效适应，在各种下游任务中实现具有竞争力的性能。我们首次证明，单个模型可以统一不同分辨率和领域的光谱数据，这表明我们的模型可以作为天文领域基础模型的强大构建块——并可能扩展到其他具有异构连续数据的科学领域，如气候和医疗保健。


### 论文摘要

Sequential scientific data span many resolutions and domains, and unifying them into a common representation is a key step toward developing foundation models for the sciences. Astronomical spectra exemplify this challenge: massive surveys have collected millions of spectra across a wide range of wavelengths and resolutions, yet analyses remain fragmented across spectral domains (e.g., optical vs. infrared) and object types (e.g., stars vs. galaxies), limiting the ability to pool information across datasets. We present a deep learning model that jointly learns from heterogeneous spectra in a self-supervised manner. Our universal spectral tokenizer processes spectra from a variety of object types and resolutions directly on their native wavelength grids, producing intrinsically aligned, homogeneous, and physically meaningful representations that can be efficiently adapted to achieve competitive performance across a range of downstream tasks. For the first time, we demonstrate that a single model can unify spectral data across resolutions and domains, suggesting that our model can serve as a powerful building block for foundation models in astronomy -- and potentially extend to other scientific domains with heterogeneous sequential data, such as climate and healthcare.

---

## 13. NeuCo-Bench: A Novel Benchmark Framework for Neural Embeddings in Earth Observation

**论文链接:** [http://arxiv.org/abs/2510.17914v1](http://arxiv.org/abs/2510.17914v1)

**作者:** Rikard Vinge, Isabelle Wittmann, Jannik Schneider, Michael Marszalek, Luis Gilch, Thomas Brunschwiler, Conrad M Albrecht

**发布时间:** 2025-10-19

### GPT解析

### 总结

本文介绍了NeuCo-Bench，一个用于评估地球观测背景下神经压缩和表示学习的新型基准框架。

### 背景

地球观测领域需要有效的神经压缩和表示学习方法，但目前缺乏标准化的评估框架。

### 目的

开发一个社区驱动、标准化的评估框架，用于评估地球观测领域的神经嵌入，平衡准确性和稳定性。

### 方法

基于固定大小的嵌入作为紧凑、任务无关的表示，包含三个核心组件：可重用嵌入的评估管道、隐藏任务排行榜的挑战模式、平衡准确性和稳定性的评分系统，并提供多光谱多时相的地球观测数据集。

### 主要发现

通过公开挑战和与最先进基础模型的消融研究，验证了NeuCo-Bench框架的有效性。

### 结论

NeuCo-Bench为地球观测及更广泛领域的神经嵌入的社区驱动、标准化评估提供了重要基础。

### 翻译

我们引入NeuCo-Bench，一个用于评估地球观测背景下（有损）神经压缩和表示学习的新型基准框架。我们的方法基于固定大小的嵌入，这些嵌入作为紧凑的、与任务无关的表示，适用于广泛的下游任务。NeuCo-Bench包含三个核心组件：(i)围绕可重用嵌入构建的评估管道，(ii)具有隐藏任务排行榜的新挑战模式，旨在减轻预训练偏差，(iii)平衡准确性和稳定性的评分系统。为了支持可重现性，我们发布了SSL4EO-S12-downstream，这是一个精心策划的多光谱、多时相的地球观测数据集。我们展示了在2025年CVPR EARTHVISION研讨会上公开挑战的初步结果，并使用最先进的基础模型进行了消融研究。NeuCo-Bench为地球观测及更广泛领域的神经嵌入的社区驱动、标准化评估迈出了第一步。


### 论文摘要

We introduce NeuCo-Bench, a novel benchmark framework for evaluating (lossy) neural compression and representation learning in the context of Earth Observation (EO). Our approach builds on fixed-size embeddings that act as compact, task-agnostic representations applicable to a broad range of downstream tasks. NeuCo-Bench comprises three core components: (i) an evaluation pipeline built around reusable embeddings, (ii) a new challenge mode with a hidden-task leaderboard designed to mitigate pretraining bias, and (iii) a scoring system that balances accuracy and stability. To support reproducibility, we release SSL4EO-S12-downstream, a curated multispectral, multitemporal EO dataset. We present initial results from a public challenge at the 2025 CVPR EARTHVISION workshop and conduct ablations with state-of-the-art foundation models. NeuCo-Bench provides a first step towards community-driven, standardized evaluation of neural embeddings for EO and beyond.

---

## 14. Diverse Influence Component Analysis: A Geometric Approach to Nonlinear Mixture Identifiability

**论文链接:** [http://arxiv.org/abs/2510.17040v2](http://arxiv.org/abs/2510.17040v2)

**作者:** Hoang-Son Nguyen, Xiao Fu

**发布时间:** 2025-10-19

**备注:** 30 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种名为多样化影响成分分析(DICA)的新框架，用于从未知非线性混合中识别潜在成分，无需依赖辅助信息、潜在成分独立性或雅可比矩阵稀疏性假设。

### 背景

从未知非线性混合中识别潜在成分是机器学习中的一个基础性挑战，应用于解缠表示学习和因果推断等任务。先前在线性独立成分分析方面的工作表明辅助信号可以支持潜在成分的可识别性，而较新方法则通过结构假设（如混合函数雅可比矩阵稀疏性）来放宽要求。

### 目的

开发一种能够从未知非线性混合中识别潜在成分的方法，同时减少对辅助信息、潜在成分独立性或雅可比矩阵稀疏性假设的依赖。

### 方法

研究提出了多样化影响成分分析(DICA)框架，利用混合函数雅可比矩阵的凸几何特性，并引入雅可比体积最大化(J-VolMax)准则，通过鼓励潜在成分对观测变量的影响多样化来实现潜在成分的识别。

### 主要发现

在合理条件下，所提出的方法无需依赖辅助信息、潜在成分独立性或雅可比矩阵稀疏性假设即可实现潜在成分的可识别性。

### 结论

研究结果扩展了可识别性分析的范围，为现有方法提供了互补的视角，为从非线性混合中识别潜在成分提供了新思路。

### 翻译

从未知非线性混合中识别潜在成分是机器学习中的一个基础性挑战，应用于解缠表示学习和因果推断等任务。先前在线性独立成分分析(nICA)方面的工作表明，辅助信号（如弱监督）可以支持条件独立潜在成分的可识别性。较新的方法则探索结构假设，例如混合函数的雅可比矩阵中的稀疏性，以放宽这些要求。在这项工作中，我们引入了多样化影响成分分析(DICA)框架，该框架利用混合函数雅可比矩阵的凸几何特性。我们提出了雅可比体积最大化(J-VolMax)准则，通过鼓励潜在成分对观测变量的影响多样化，实现潜在成分的识别。在合理条件下，这种方法无需依赖辅助信息、潜在成分独立性或雅可比矩阵稀疏性假设即可实现可识别性。这些结果扩展了可识别性分析的范围，并为现有方法提供了互补的视角。


### 论文摘要

Latent component identification from unknown nonlinear mixtures is a foundational challenge in machine learning, with applications in tasks such as disentangled representation learning and causal inference. Prior work in nonlinear independent component analysis (nICA) has shown that auxiliary signals -- such as weak supervision -- can support identifiability of conditionally independent latent components. More recent approaches explore structural assumptions, e.g., sparsity in the Jacobian of the mixing function, to relax such requirements. In this work, we introduce Diverse Influence Component Analysis (DICA), a framework that exploits the convex geometry of the mixing function's Jacobian. We propose a Jacobian Volume Maximization (J-VolMax) criterion, which enables latent component identification by encouraging diversity in their influence on the observed variables. Under reasonable conditions, this approach achieves identifiability without relying on auxiliary information, latent component independence, or Jacobian sparsity assumptions. These results extend the scope of identifiability analysis and offer a complementary perspective to existing methods.

---

## 15. Triangle Multiplication Is All You Need For Biomolecular Structure Representations

**论文链接:** [http://arxiv.org/abs/2510.18870v1](http://arxiv.org/abs/2510.18870v1)

**作者:** Jeffrey Ouyang-Zhang, Pranav Murugan, Daniel J. Diaz, Gianluca Scarpellini, Richard Strong Bowen, Nate Gruver, Adam Klivans, Philipp Krähenbühl, Aleksandra Faust, Maruan Al-Shedivat

**发布时间:** 2025-10-21

**备注:** Preprint

### GPT解析

### 总结

Pairmixer是一种替代AlphaFold3中Pairformer主干网络的高效解决方案，消除了计算密集型的三角形注意力机制，同时保留了高阶几何推理能力，显著提高了计算效率。

### 背景

AlphaFold已变革蛋白质结构预测，但虚拟配体筛选、全蛋白质组折叠和从头结合剂设计等新兴应用需要大规模预测，面临运行时间和内存成本过高的问题。

### 目的

开发一种替代方案，消除三角形注意力机制，同时保留对结构预测至关重要的高阶几何推理能力，提高计算效率。

### 方法

提出Pairmixer，一种简化的替代方案，消除三角形注意力机制，同时保留高阶几何推理能力。

### 主要发现

Pairmixer在折叠和对接基准测试中匹配最先进结构预测器性能，实现长序列上高达4倍的更快推理速度，训练成本降低34%；在BoltzDesign中提供超过2倍的更快采样速度，可扩展到比Pairformer内存限制长约30%的序列。

### 结论

Pairmixer解决了大规模蛋白质结构预测的计算瓶颈问题，为建模大型蛋白质复合物、高通量配体和结合剂筛选等下游应用提供了高效解决方案。

### 翻译

AlphaFold已经改变了蛋白质结构预测，但新兴应用如虚拟配体筛选、全蛋白质组折叠和从头结合剂设计需要大规模预测，此时运行时间和内存成本变得过高。主要瓶颈在于AlphaFold3类模型的Pairformer主干网络，它依赖于计算密集型的三角形原语——特别是三角形注意力机制——用于成对推理。我们引入Pairmixer，一种简化的替代方案，消除了三角形注意力同时保留了对结构预测至关重要的高阶几何推理能力。Pairmixer显著提高了计算效率，在折叠和对接基准测试中匹配最先进的结构预测器，在长序列上实现高达4倍的更快推理速度，同时将训练成本降低34%。其效率减轻了下游应用的计算负担，如建模大型蛋白质复合物、高通量配体和结合剂筛选以及基于幻觉的设计。例如，在BoltzDesign中，Pairmixer提供超过2倍的更快采样速度，并可扩展到比Pairformer内存限制长约30%的序列。


### 论文摘要

AlphaFold has transformed protein structure prediction, but emerging applications such as virtual ligand screening, proteome-wide folding, and de novo binder design demand predictions at a massive scale, where runtime and memory costs become prohibitive. A major bottleneck lies in the Pairformer backbone of AlphaFold3-style models, which relies on computationally expensive triangular primitives-especially triangle attention-for pairwise reasoning. We introduce Pairmixer, a streamlined alternative that eliminates triangle attention while preserving higher-order geometric reasoning capabilities that are critical for structure prediction. Pairmixer substantially improves computational efficiency, matching state-of-the-art structure predictors across folding and docking benchmarks, delivering up to 4x faster inference on long sequences while reducing training cost by 34%. Its efficiency alleviates the computational burden of downstream applications such as modeling large protein complexes, high-throughput ligand and binder screening, and hallucination-based design. Within BoltzDesign, for example, Pairmixer delivers over 2x faster sampling and scales to sequences ~30% longer than the memory limits of Pairformer.

---

## 16. Event-Grounding Graph: Unified Spatio-Temporal Scene Graph from Robotic Observations

**论文链接:** [http://arxiv.org/abs/2510.18697v1](http://arxiv.org/abs/2510.18697v1)

**作者:** Phuoc Nguyen, Francesco Verdoja, Ville Kyrki

**发布时间:** 2025-10-21

**备注:** Submitted to RA-L

### GPT解析

### 总结

本研究提出了事件基础图（EGG）框架，通过将事件交互与场景空间特征关联，增强了机器人对环境的理解和交互能力，使机器人能够感知、推理和响应复杂的时空查询。

### 背景

构建能够协助人类日常生活的智能自主机器人需要丰富的环境表示。虽然语义场景表示的进步已丰富了机器人的场景理解，但当前方法缺乏空间特征与动态事件之间的联系，例如无法将蓝色杯子与'洗杯子'事件联系起来。

### 目的

开发一种能够将事件交互与场景空间特征联系起来的框架，使机器人能够更好地理解和响应环境中的动态事件。

### 方法

引入事件基础图（EGG）框架，这是一种将事件交互与场景空间特征关联的环境表示方法，允许机器人感知、推理和响应复杂的时空查询。

### 主要发现

使用真实机器人数据的实验证明EGG能够检索相关信息并准确响应对环境和事件的询问，展示了其在实际应用中的有效性。

### 结论

EGG框架通过开源源代码和评估数据集（https://github.com/aalto-intelligent-robotics/EGG）为机器人环境理解和交互提供了新的解决方案，促进了该领域的发展。

### 翻译

构建能够协助人类日常生活的智能自主机器人的一个基本方面是构建丰富的环境表示。尽管语义场景表示的进步已经丰富了机器人的场景理解，但当前方法缺乏空间特征与动态事件之间的联系；例如，无法将蓝色杯子与'洗杯子'事件联系起来。在这项工作中，我们引入了事件基础图（EGG），这是一个将事件交互与场景空间特征关联的框架。这种表示允许机器人感知、推理和响应复杂的时空查询。使用真实机器人数据的实验证明了EGG能够检索相关信息并准确响应对环境和事件的询问。此外，EGG框架的源代码和评估数据集已在以下地址开源：https://github.com/aalto-intelligent-robotics/EGG。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决如何构建一个能够将空间特征与动态事件连接起来的统一场景表示方法。这个问题很重要，因为当前的场景表示方法要么只关注静态空间元素(如3D场景图)，要么只记录事件描述(如视频字幕)，但无法将两者有效连接。这种连接缺失使得机器人无法理解空间变化与导致这些变化的交互之间的关系，限制了它们回答复杂时空查询的能力，如'你最后一次看到我的杯子是什么时候？'或'找到一台咖啡机'。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者在设计方法时借鉴了现有工作的优点并试图解决其局限性。他们参考了3D场景图(3DSGs)的层次结构来表示空间元素，但增加了动态事件的表示；借鉴了记忆表示方法(如ReMEmbR和Embodied-RAG)记录事件交互的思想，但解决了它们缺乏空间特征连接的问题；还参考了利用大型语言模型进行信息检索的方法(如SayPlan和H-EMV)，但专注于改进基础表示。作者将这些方法的优势整合，创建了一个统一的图结构，通过边将事件与参与的空间元素连接起来。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过一个统一的图结构(EGG)同时表示场景中的空间元素和动态事件，并通过边将事件与参与事件的物体连接起来。整体流程包括：1) 构建EGG图，包含空间组件(类似3D场景图)和事件组件(描述观察到的活动)，以及连接事件与空间元素的事件边；2) 当接收到查询时，根据查询的四个维度(时间、位置、空间元素和事件)修剪图，提取相关子图；3) 将修剪后的子图序列化为JSON格式，输入到大型语言模型中生成回答。这种方法使机器人能够同时理解场景的空间布局和其中发生的事件。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 创建了统一的时空场景表示，同时包含空间组件和事件组件；2) 通过事件边将动态事件与参与的空间元素连接起来，解决了现有方法中空间特征与动态事件之间的连接缺失问题；3) 开发了基于查询的图修剪策略，减少计算负担并提高回答质量；4) 使用真实机器人数据进行了全面的实验验证。相比之前的工作，EGG的主要不同在于：与传统3D场景图相比，增加了动态事件的表示；与纯事件表示方法相比，保留了空间特征的精确表示；与现有记忆表示方法相比，通过事件边明确地连接事件与空间元素，解决了多视角一致性和冗余信息问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了EGG框架，通过将动态事件与空间特征连接起来，创建了一个统一的时空场景图，使机器人能够感知、推理和解释场景中发生的事件，并准确回答复杂的时空查询。'}


### 论文摘要

A fundamental aspect for building intelligent autonomous robots that can assist humans in their daily lives is the construction of rich environmental representations. While advances in semantic scene representations have enriched robotic scene understanding, current approaches lack a connection between spatial features and dynamic events; e.g., connecting the blue mug to the event washing a mug. In this work, we introduce the event-grounding graph (EGG), a framework grounding event interactions to spatial features of a scene. This representation allows robots to perceive, reason, and respond to complex spatio-temporal queries. Experiments using real robotic data demonstrate EGG's capability to retrieve relevant information and respond accurately to human inquiries concerning the environment and events within. Furthermore, the EGG framework's source code and evaluation dataset are released as open-source at: https://github.com/aalto-intelligent-robotics/EGG.

---

## 17. MoTVLA: A Vision-Language-Action Model with Unified Fast-Slow Reasoning

**论文链接:** [http://arxiv.org/abs/2510.18337v1](http://arxiv.org/abs/2510.18337v1)

**作者:** Wenhui Huang, Changhe Chen, Han Qi, Chen Lv, Yilun Du, Heng Yang

**发布时间:** 2025-10-21

### GPT解析

### 总结

研究提出MoTVLA模型，一种基于混合变换器的视觉语言动作模型，整合快速-慢速统一推理与行为策略学习，解决机器人学习中语言可控性和推理效率的平衡问题。

### 背景

将视觉语言指令整合到视觉运动策略中是增强机器人学习在开放世界中泛化能力的重要方向，但现有方法面临语言可控性有限或推理延迟显著的挑战。

### 目的

开发一种既能保持预训练视觉语言模型的通用智能，又能提高策略执行效率和语言可控性的机器人学习模型。

### 方法

提出MoTVLA模型，结合预训练视觉语言模型作为通用专家和领域特定专家，生成快速推理并基于分解的运动指令学习多样化行为。

### 主要发现

MoTVLA在保持通用智能的同时，通过领域专家生成快速推理显著提高了策略执行效率，并通过条件化动作专家提升了语言可控性。

### 结论

MoTVLA在自然语言处理基准、机器人仿真环境和真实世界实验中表现出优越的快速-慢速推理能力和操作任务性能。

### 翻译

将视觉语言指令整合到视觉运动策略中正在增强机器人学习在开放世界中的泛化能力。尽管取得了 promising 的进展，但现有方法面临两个挑战：在不使用生成推理作为条件时，语言可控性有限；当加入推理时，推理延迟显著。在这项工作中，我们引入了MoTVLA，一种基于混合变换器的视觉语言动作模型，整合了快速-慢速统一推理与行为策略学习。MoTVLA保留了预训练视觉语言模型的通用智能，用于感知、场景理解和语义规划等任务，同时引入领域专家（第二个与预训练VLM共享知识的Transformer）生成领域特定的快速推理（如机器人运动分解），从而提高策略执行效率。通过将动作专家基于分解的运动指令进行条件化，MoTVLA可以学习多样化行为并显著提高语言可控性。在自然语言处理基准、机器人仿真环境和真实世界实验中的广泛评估证实了MoTVLA在快速-慢速推理和操作任务性能方面的优越性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉-语言-行动(VLA)模型中的两个关键挑战：当不使用生成的推理作为条件时，语言可控性有限；当融入推理过程时，推理延迟显著。这个问题在机器人学习领域非常重要，因为机器人需要既能理解复杂语言指令，又能快速执行任务的能力。特别是在现实世界的应用中，机器人需要实时响应用户指令并准确执行任务，现有方法要么牺牲语言引导能力，要么牺牲推理速度，这限制了机器人在时间关键型应用中的实用性和泛化能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到现有方法要么缺乏语言可控性，要么推理速度慢，因此需要一种能够同时兼顾两者的方法。他们思考如何在一个统一架构中结合快速推理（用于高效执行）和慢速推理（用于复杂理解），同时保留预训练VLM的通用智能并添加领域特定知识以提高执行效率。作者借鉴了多种现有工作，包括：视觉-语言模型(VLMs)作为预训练基础，扩散策略(DPs)用于建模连续动作空间，混合变换器(MoT)架构用于整合不同功能组件，以及现有的VLA模型如RT-2、OpenVLA、π0.5等。在这些工作基础上，作者创新性地提出了统一的快速-慢速推理架构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "MoTVLA的核心思想是在单一架构中统一快速和慢速推理，通过在输入端分解模态，在输出端分解功能，同时在中间保持共享的全局知识库。模型包含三个关键组件：通才(generalist)负责视觉-文本理解和慢速推理；领域专家(domain expert)专注于机器人任务的快速推理；行动专家(action expert)负责多任务策略学习。整体实现流程：1)输入处理：将输入分解为语言提示、RGB图像和可学习查询；2)推理主干：遵循'分解-组合-分解'范式，先独立处理多模态输入，再通过全局自注意力机制整合，最后在输出端解耦执行不同类型推理；3)推理输出：慢速推理用下一个token预测，快速推理用token-wise预测；4)行动专家：用扩散Transformer(DiT)在动作扩散框架内学习策略；5)训练：先进行领域专家监督微调，再训练行动专家；6)推理：支持对话模式(慢速推理)和行动模式(快速推理)。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)统一的快速-慢速推理架构：首次在单一模型中统一两种推理，保留通用智能同时高效学习领域知识；2)基于分解运动的策略学习：通过快速推理生成的分解运动来条件化策略学习，加快任务执行速度；3)分解-组合-分解设计：输入端分解模态，中间通过共享注意力整合，输出端分解功能；4)双模式推理：支持对话模式和行动模式，确保响应与语言提示一致；5)知识共享机制：通才和领域专家通过全局自注意力共享知识。相比之前工作的不同：1)与现有VLA模型相比：同时实现高语言可控性和低推理延迟；2)与基于扩散的策略相比：显式生成推理来条件化策略，提高语言可控性；3)与π0.5等模型相比：通过token-wise预测实现快速推理，大幅提高推理效率；4)架构设计不同：采用创新的MoT架构，明确分离功能但允许知识共享；5)训练策略不同：采用两阶段课程学习，保留预训练VLM的通用智能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了MoTVLA，一种基于混合变换器架构的视觉-语言-行动模型，通过统一的快速-慢速推理机制，在保留预训练视觉语言模型通用智能的同时，显著提高了语言可控性和推理效率，为机器人学习中的开放世界语言指导任务提供了新的解决方案。'}


### 论文摘要

Integrating visual-language instructions into visuomotor policies is gaining momentum in robot learning for enhancing open-world generalization. Despite promising advances, existing approaches face two challenges: limited language steerability when no generated reasoning is used as a condition, or significant inference latency when reasoning is incorporated.In this work, we introduce MoTVLA, a mixture-of-transformers (MoT)-based vision-language-action (VLA) model that integrates fast-slow unified reasoning with behavior policy learning. MoTVLA preserves the general intelligence of pre-trained VLMs (serving as the generalist) for tasks such as perception, scene understanding, and semantic planning, while incorporating a domain expert, a second transformer that shares knowledge with the pretrained VLM, to generate domain-specific fast reasoning (e.g., robot motion decomposition), thereby improving policy execution efficiency. By conditioning the action expert on decomposed motion instructions, MoTVLA can learn diverse behaviors and substantially improve language steerability. Extensive evaluations across natural language processing benchmarks, robotic simulation environments, and real-world experiments confirm the superiority of MoTVLA in both fast-slow reasoning and manipulation task performance.

---

## 18. UWBench: A Comprehensive Vision-Language Benchmark for Underwater Understanding

**论文链接:** [http://arxiv.org/abs/2510.18262v1](http://arxiv.org/abs/2510.18262v1)

**作者:** Da Zhang, Chenggang Rong, Bingyu Li, Feiyu Wang, Zhiyuan Zhao, Junyu Gao, Xuelong Li

**发布时间:** 2025-10-21

**备注:** We have released V1, which only reports the test results. Our work is  still ongoing, and the next version will be coming soon

### GPT解析

### 总结

大型视觉语言模型在自然场景理解方面取得成功，但在水下环境应用尚未充分探索。研究人员开发了UWBench基准数据集，包含15003张高分辨率水下图像及丰富标注，建立了三个水下视觉语言理解基准任务，实验表明水下理解仍具挑战性，该基准为水下环境视觉语言研究提供重要资源。

### 背景

大型视觉语言模型在自然场景理解方面取得显著成功，但在水下环境中的应用仍被忽视。水下图像面临光衰减、色彩失真和悬浮颗粒散射等独特挑战，同时需要海洋生态系统和生物分类学专业知识。

### 目的

填补大型视觉语言模型在水下环境应用的空白，引入专为水下视觉语言理解设计的综合基准UWBench，为水下环境中的视觉语言研究提供必要资源，支持海洋科学、生态监测和自主水下探索等应用。

### 方法

构建UWBench基准，包含15003张不同水生环境的高分辨率水下图像，每张图像配有15281个对象指代表达式和124983个问答对。基于此建立三个综合基准：详细图像字幕生成、视觉定位和视觉问答，用于评估模型在水下环境中的表现。

### 主要发现

在最先进视觉语言模型上的实验表明，水下理解仍然具有挑战性，存在大量改进空间。数据集捕捉了可见度、光照条件和水浊度的丰富变化，为模型评估提供了现实的测试平台。

### 结论

UWBench基准为推进水下环境中的视觉语言研究提供了必要资源，支持海洋科学、生态监测和自主水下探索等应用。研究代码和基准将公开可用，促进相关领域发展。

### 翻译

大型视觉语言模型在自然场景理解方面取得了显著成功，但在水下环境中的应用仍然很大程度上未被探索。水下图像呈现独特的挑战，包括严重的光衰减、色彩失真和悬浮颗粒散射，同时需要海洋生态系统和生物分类学的专业知识。为了填补这一空白，我们引入UWBench，一个专门为水下视觉语言理解设计的综合基准。UWBench包含15003张在不同水生环境中捕获的高分辨率水下图像，涵盖海洋、珊瑚礁和深海栖息地。每张图像都经过人工验证的注释丰富，包括15281个精确描述海洋生物和水下结构的对象指代表达式，以及124983个涵盖从对象识别到生态关系理解的多样化推理能力的问答对。数据集捕捉到了可见度、光照条件和水浊度的丰富变化，为模型评估提供了现实的测试平台。基于UWBench，我们建立了三个综合基准：用于生成生态信息场景描述的详细图像字幕生成，用于精确定位海洋生物的视觉定位，以及用于水下环境多模态推理的视觉问答。在最先进的视觉语言模型上的广泛实验表明，水下理解仍然具有挑战性，有大量改进空间。我们的基准为推进水下环境中的视觉语言研究提供了必要的资源，支持海洋科学、生态监测和自主水下探索等应用。我们的代码和基准将公开可用。


### 论文摘要

Large vision-language models (VLMs) have achieved remarkable success in natural scene understanding, yet their application to underwater environments remains largely unexplored. Underwater imagery presents unique challenges including severe light attenuation, color distortion, and suspended particle scattering, while requiring specialized knowledge of marine ecosystems and organism taxonomy. To bridge this gap, we introduce UWBench, a comprehensive benchmark specifically designed for underwater vision-language understanding. UWBench comprises 15,003 high-resolution underwater images captured across diverse aquatic environments, encompassing oceans, coral reefs, and deep-sea habitats. Each image is enriched with human-verified annotations including 15,281 object referring expressions that precisely describe marine organisms and underwater structures, and 124,983 question-answer pairs covering diverse reasoning capabilities from object recognition to ecological relationship understanding. The dataset captures rich variations in visibility, lighting conditions, and water turbidity, providing a realistic testbed for model evaluation. Based on UWBench, we establish three comprehensive benchmarks: detailed image captioning for generating ecologically informed scene descriptions, visual grounding for precise localization of marine organisms, and visual question answering for multimodal reasoning about underwater environments. Extensive experiments on state-of-the-art VLMs demonstrate that underwater understanding remains challenging, with substantial room for improvement. Our benchmark provides essential resources for advancing vision-language research in underwater contexts and supporting applications in marine science, ecological monitoring, and autonomous underwater exploration. Our code and benchmark will be available.

---

## 19. OpenInsGaussian: Open-vocabulary Instance Gaussian Segmentation with Context-aware Cross-view Fusion

**论文链接:** [http://arxiv.org/abs/2510.18253v1](http://arxiv.org/abs/2510.18253v1)

**作者:** Tianyu Huang, Runnan Chen, Dongting Hu, Fengming Huang, Mingming Gong, Tongliang Liu

**发布时间:** 2025-10-21

### GPT解析

### 总结

OpenInsGaussian是一种开放词汇实例高斯分割框架，通过上下文感知特征提取和注意力驱动的特征聚合两个模块解决了现有3D场景理解方法中的两个主要局限：预处理过程中单个掩码的上下文线索不足，以及融合多视图特征时存在的不一致性和缺失细节。

### 背景

理解3D场景对自动驾驶、机器人和增强现实至关重要。现有的语义高斯飞溅方法利用大规模2D视觉模型将2D语义特征投影到3D场景上。

### 目的

解决现有语义高斯飞溅方法的两个主要局限：预处理过程中单个掩码的上下文线索不足，以及融合多视图特征时存在的不一致性和缺失细节。

### 方法

提出OpenInsGaussian框架，包含两个模块：1) 上下文感知特征提取，为每个掩码添加丰富的语义上下文；2) 注意力驱动的特征聚合，选择性融合多视图特征以减轻对齐错误和不完整性。

### 主要发现

在基准数据集上进行的大量实验表明，OpenInsGaussian在开放词汇3D高斯分割方面取得了最先进的结果，大幅优于现有基线。

### 结论

OpenInsGaussian方法具有鲁棒性和通用性，标志着3D场景理解及其在各种实际场景中实际部署的重要一步。

### 翻译

理解3D场景对自动驾驶、机器人和增强现实至关重要。最近的语义高斯飞溅方法利用大规模2D视觉模型将2D语义特征投影到3D场景上。然而，它们存在两个主要局限：(1) 预处理过程中单个掩码的上下文线索不足；(2) 融合这些2D模型的多视图特征时存在不一致性和缺失细节。在本文中，我们介绍了OpenInsGaussian，一个具有上下文感知跨视图融合的开放词汇实例高斯分割框架。我们的方法包含两个模块：上下文感知特征提取，为每个掩码添加丰富的语义上下文；以及注意力驱动的特征聚合，选择性融合多视图特征以减轻对齐错误和不完整性。在基准数据集上的大量实验表明，OpenInsGaussian在开放词汇3D高斯分割方面取得了最先进的结果，大幅优于现有基线。这些发现强调了所提出方法的鲁棒性和通用性，标志着3D场景理解及其在各种实际场景中实际部署的重要一步。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D场景理解中的两个关键问题：1) 在预处理阶段从掩码提取特征时丢失上下文信息，特别是对小型或部分遮挡对象的识别；2) 多视图特征融合时由于光照、遮挡和视角变化导致的不一致问题。这些问题在自动驾驶、机器人和增强现实等应用中至关重要，因为它们直接影响3D场景语义理解的准确性和可靠性，而现有的语义高斯飞溅方法在这两方面存在明显局限。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法的局限性，借鉴了3D高斯飞溅(3DGS)技术用于场景建模，利用CLIP等视觉语言模型提取语义特征，并参考了OpenGaussian的对比学习框架和SAM的分割能力。在此基础上，作者设计了两个关键模块：上下文感知特征提取模块，直接从CLIP中间特征图提取保留上下文的信息；以及注意力驱动的特征聚合策略，根据语义一致性选择性融合多视图特征。这种方法既利用了现有技术的优势，又针对性地解决了它们忽视的问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过保留上下文信息和智能融合多视图特征来提升3D场景语义理解的准确性。整体流程分为四个阶段：1) 预处理：使用SAM生成对象掩码，同时提取局部特征和上下文感知特征并融合；2) 实例特征学习：通过对比学习训练3D高斯的类无关实例特征；3) 离散化：使用分层聚类策略将3D高斯分割为类无关簇；4) 语言特征聚合：通过注意力机制将语言特征与分割实例关联，根据视图间语义一致性加权融合特征。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 上下文感知特征提取，从CLIP中间特征图而非裁剪图像提取特征，保留空间上下文；2) 注意力驱动的特征聚合，基于余弦相似度而非复杂自注意力机制，高效融合多视图特征；3) 几何集成策略融合局部和上下文特征。相比之前工作，OpenInsGaussian解决了LangSplat和LEGaussians的模糊特征问题，也克服了OpenGaussian在处理运动模糊图像时的局限性，同时计算效率更高，无需额外训练即可直接应用于CLIP特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'OpenInsGaussian通过创新的上下文感知特征提取和注意力驱动的多视图融合机制，显著提升了3D高斯飞溅在开放词汇场景理解中的性能，解决了现有方法中上下文丢失和多视图不一致的关键挑战。'}


### 论文摘要

Understanding 3D scenes is pivotal for autonomous driving, robotics, and augmented reality. Recent semantic Gaussian Splatting approaches leverage large-scale 2D vision models to project 2D semantic features onto 3D scenes. However, they suffer from two major limitations: (1) insufficient contextual cues for individual masks during preprocessing and (2) inconsistencies and missing details when fusing multi-view features from these 2D models. In this paper, we introduce \textbf{OpenInsGaussian}, an \textbf{Open}-vocabulary \textbf{Ins}tance \textbf{Gaussian} segmentation framework with Context-aware Cross-view Fusion. Our method consists of two modules: Context-Aware Feature Extraction, which augments each mask with rich semantic context, and Attention-Driven Feature Aggregation, which selectively fuses multi-view features to mitigate alignment errors and incompleteness. Through extensive experiments on benchmark datasets, OpenInsGaussian achieves state-of-the-art results in open-vocabulary 3D Gaussian segmentation, outperforming existing baselines by a large margin. These findings underscore the robustness and generality of our proposed approach, marking a significant step forward in 3D scene understanding and its practical deployment across diverse real-world scenarios.

---

## 20. HouseTour: A Virtual Real Estate A(I)gent

**论文链接:** [http://arxiv.org/abs/2510.18054v1](http://arxiv.org/abs/2510.18054v1)

**作者:** Ata Çelen, Marc Pollefeys, Daniel Barath, Iro Armeni

**发布时间:** 2025-10-20

**备注:** Published on ICCV 2025

### GPT解析

### 总结

本文介绍了HouseTour，一种从描绘现有3D空间的图像集合中生成具有空间感知的3D相机轨迹和自然语言摘要的方法。

### 背景

现有的视觉语言模型在几何推理方面存在困难，缺乏能够同时处理3D相机轨迹生成和文本描述的集成方法。

### 目的

开发一种能够生成平滑视频轨迹并提供3D基础描述的方法，实现自动化、专业质量的视频创作，应用于房地产和旅游领域，且不需要专业知识或设备。

### 方法

通过受已知相机位姿约束的扩散过程生成平滑视频轨迹；将这些信息整合到视觉语言模型中以生成基于3D的描述；使用3D高斯溅射技术合成最终视频，沿轨迹渲染新视图；提出包含1200多个房屋导览视频的HouseTour数据集，包括相机位姿、3D重建和房地产描述。

### 主要发现

将3D相机轨迹整合到文本生成过程中，比独立处理每个任务的方法性能更好；研究评估了单独和端到端的性能，并引入了一种新的联合度量标准。

### 结论

HouseTour方法实现了自动化、专业质量的视频创作，可应用于房地产和旅游领域，且不需要专业知识和设备。

### 翻译

我们介绍了HouseTour，一种从描绘现有3D空间的图像集合中生成具有空间感知的3D相机轨迹和自然语言摘要的方法。与现有的视觉语言模型不同，后者在几何推理方面存在困难，我们的方法通过受已知相机位姿约束的扩散过程生成平滑的视频轨迹，并将这些信息整合到视觉语言模型中以生成基于3D的描述。我们使用3D高斯溅射技术合成最终视频，沿轨迹渲染新视图。为了支持这项任务，我们提出了HouseTour数据集，其中包含超过1200个带有相机位姿、3D重建和房地产描述的房屋导览视频。实验表明，将3D相机轨迹整合到文本生成过程中，比独立处理每个任务的方法性能更好。我们评估了单独和端到端的性能，并引入了一种新的联合度量标准。我们的工作实现了自动化、专业质量的视频创作，可用于房地产和旅游应用，无需专业知识或设备。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从一组带有已知相机位置的图像中自动生成类似专业房地产视频游览的平滑3D相机轨迹和描述性文本摘要的问题。这个问题在现实中很重要，因为房地产视频游览在YouTube上有超过6.24亿个视频，是美国价值3.43万亿美元房地产市场的关键工具，但目前制作这类视频需要专业人员和高昂设备，劳动密集且成本高。在研究上，这个问题也很重要，因为现有的视觉语言模型在几何推理方面存在困难，生成基于3D空间的视频并用结构化语言描述空间特性仍然是一个挑战。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将问题分解为两个主要任务：3D相机轨迹生成和文本摘要生成。在轨迹生成方面，作者借鉴了Diffuser的扩散过程，但提出了残差扩散器(Residual Diffuser)，将轨迹规划表示为样条插值的残差，这种方法更适合不同房地产布局的交互空间。在文本生成方面，作者基于Qwen2-VL模型构建了Qwen2-VL-3D，使用LoRA微调方法，并将3D空间信息作为第三种模态整合到视觉语言模型中。作者还构建了HouseTour数据集来支持这一任务。整体流程是从输入图像生成轨迹，然后结合轨迹信息和图像生成文本摘要，最后使用3D高斯飞溅技术渲染视频。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过空间感知的3D相机轨迹生成和多模态文本生成来创建类似专业房地产视频游览的系统。整体流程是：1)输入一组带有已知相机位置的图像；2)使用残差扩散器生成平滑的3D相机轨迹；3)使用Qwen2-VL-3D模型结合轨迹信息和图像生成描述性文本摘要；4)使用3D高斯飞溅技术沿着生成的轨迹渲染新视图，合成最终视频。这种方法确保了文本与空间路径对齐，并能生成描述空间布局、功能性和建筑特征的文本，而不仅仅是列出物体。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)提出'空间感知的3D相机轨迹和文本摘要生成'这一新任务；2)开发残差扩散器，将轨迹规划表示为样条插值的残差；3)构建Qwen2-VL-3D模型，整合3D空间信息作为第三种模态；4)创建HouseTour数据集，包含专业房地产描述和真实世界视频轨迹。相比之前的工作，HouseTour的方法在轨迹生成上明确基于已知3D场景几何条件，整体性而非顺序地制定轨迹规划；在文本生成上能处理大量稀疏多图像数据，捕捉完整布局；在数据集上关注空间布局和建筑特征，而非仅关注家具和物体关系。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'HouseTour提出了一种结合扩散过程生成的3D相机轨迹和整合空间信息的视觉语言模型，能够自动生成专业质量的房地产视频游览，无需专业知识或设备，并为此任务构建了一个新的数据集。'}


### 论文摘要

We introduce HouseTour, a method for spatially-aware 3D camera trajectory and natural language summary generation from a collection of images depicting an existing 3D space. Unlike existing vision-language models (VLMs), which struggle with geometric reasoning, our approach generates smooth video trajectories via a diffusion process constrained by known camera poses and integrates this information into the VLM for 3D-grounded descriptions. We synthesize the final video using 3D Gaussian splatting to render novel views along the trajectory. To support this task, we present the HouseTour dataset, which includes over 1,200 house-tour videos with camera poses, 3D reconstructions, and real estate descriptions. Experiments demonstrate that incorporating 3D camera trajectories into the text generation process improves performance over methods handling each task independently. We evaluate both individual and end-to-end performance, introducing a new joint metric. Our work enables automated, professional-quality video creation for real estate and touristic applications without requiring specialized expertise or equipment.

---

## 21. SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes

**论文链接:** [http://arxiv.org/abs/2510.16714v2](http://arxiv.org/abs/2510.16714v2)

**作者:** Xiongkun Linghu, Jiangyong Huang, Ziyu Zhu, Baoxiong Jia, Siyuan Huang

**发布时间:** 2025-10-19

**备注:** Project page: https://scenecot.github.io/

### GPT解析

### 总结

本文提出了一种名为SCENECOT的新框架和SCENECOT-185K数据集，通过将复杂推理任务分解并结合多模态专家模块，实现了3D场景中的类人推理，在多个基准测试中表现出色。

### 背景

现有关于3D大型语言模型的研究仍然难以实现基于场景的问答，主要原因是人类场景-对象基础推理机制的探索不足。

### 目的

填补3D场景基础推理研究的空白，通过提出新颖框架实现类人场景-对象推理。

### 方法

引入SCENECOT方法将复杂推理任务分解为更简单的问题，基于多模态专家模块构建视觉线索，并开发包含185K个高质量实例的SCENECOT-185K数据集。

### 主要发现

在各种复杂的3D场景推理基准测试中，新框架在保持高基础问答连贯性的同时实现了强大的性能。

### 结论

首次成功将思维链推理应用于3D场景理解，实现了逐步类人推理，并显示出扩展到更广泛3D场景理解场景的潜力。

### 翻译

现有关于3D大型语言模型的研究仍然难以实现基于场景的问答，这主要是由于对类人场景-对象基础推理机制的探索不足。本文通过提出一个新颖框架来弥合这一差距。我们首先在3D场景中引入了一种基础的思维链推理方法，将复杂的推理任务解耦为更简单和可管理的问题，并基于多模态专家模块构建相应的视觉线索。为实现这种方法，我们开发了SCENECOT-185K，这是第一个大规模的基础CoT推理数据集，包含185K个高质量实例。在各种复杂的3D场景推理基准上进行的广泛实验表明，我们的新框架在保持高基础问答连贯性的同时实现了强大的性能。据我们所知，这是首次成功将CoT推理应用于3D场景理解，实现了逐步类人推理，并显示出扩展到更广泛的3D场景理解场景的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D大语言模型在实现基于场景的问答（grounded question-answering）方面的困难，特别是缺乏类人场景-对象推理机制的问题。这个问题在现实中很重要，因为3D场景理解是构建类人智能体的基础能力，而现有模型虽然能生成看似合理的答案，但无法将中间推理步骤与最终结果联系起来，导致grounding-QA一致性差，影响模型在复杂3D环境中的可靠性和实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了语言领域中的思维链（Chain-of-Thought, CoT）推理方法，观察到CoT通过将复杂问题分解为可管理的子问题，使语言模型在多种任务中表现出色。作者认为这种逐步推理的方式与人类认知过程相似，也符合3D场景中需要的多跳推理。然而，直接将CoT转移到3D场景具有挑战性，因为需要将基于语言的推理与多模态3D场景表示对齐。因此，作者设计了SCENECOT框架，将3D推理分解为四个阶段：任务识别、区域定位、实体定位和基于场景的推理。该方法借鉴了语言领域的CoT、2D视觉语言推理、多模态大型语言模型和专门的视觉定位模型等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将复杂的3D场景推理任务分解为逐步的、基于场景的思维链推理过程，确保每个答案都有明确的基于场景的步骤支持，从而增强grounding-QA一致性。整体实现流程分为四个阶段：1）任务识别和分析，确定回答问题所需的底层任务和初始分析；2）任务相关区域定位，通过方向线索或基于时钟的参考系缩小推理空间；3）实体定位，使用多模态专家模块定位与问题相关的目标对象；4）基于场景的推理，获取候选对象信息并整合为最终答案。技术实现上，SCENECOT建立在多模态大型语言模型基础上，融入专门的3D-VL和2D-VL模型以及符号引擎来支持这种逐步推理结构。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出SCENECOT框架，首次将思维链推理应用于3D场景理解；2）构建SCENECOT-185K数据集，第一个大规模3D场景思维链推理数据集；3）显著提高grounding-QA一致性，在Beacon3D基准上达到34.7的良好一致性。相比之前工作，SCENECOT采用显式的逐步推理机制而非端到端训练，通过显式强制推理前的定位实现准确答案和高grounding-QA一致性，将3D问答视为多阶段任务而非单步任务，并利用大规模标注的推理轨迹数据集进行训练，使决策过程更透明、更接近人类推理方式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SCENECOT通过引入首个大规模3D场景思维链推理数据集和框架，实现了类人、可解释且基于场景的3D推理，显著提高了复杂3D场景问答中的grounding-QA一致性。'}


### 论文摘要

Existing research on 3D Large Language Models (LLMs) still struggles to achieve grounded question-answering, primarily due to the under-exploration of the mechanism of human-like scene-object grounded reasoning. This paper bridges the gap by presenting a novel framework. We first introduce a grounded Chain-of-Thought reasoning method in 3D scenes (SCENECOT), decoupling a complex reasoning task into simpler and manageable problems, and building corresponding visual clues based on multimodal expert modules. To enable such a method, we develop SCENECOT-185K, the first large-scale grounded CoT reasoning dataset, consisting of 185K high-quality instances. Extensive experiments across various complex 3D scene reasoning benchmarks demonstrate that our new framework achieves strong performance with high grounding-QA coherence. To the best of our knowledge, this is the first successful application of CoT reasoning to 3D scene understanding, enabling step-by-step human-like reasoning and showing potential for extension to broader 3D scene understanding scenarios.

---

## 22. StreamingTOM: Streaming Token Compression for Efficient Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.18269v1](http://arxiv.org/abs/2510.18269v1)

**作者:** Xueyi Chen, Keda Tao, Kele Shao, Huan Wang

**发布时间:** 2025-10-21

### GPT解析

### 总结

StreamingTOM是一个解决流式视频视觉语言模型因果性和累积性约束的两阶段框架，通过因果时间减少和在线量化存储技术，实现了高效的流式视频理解。

### 背景

与离线处理不同，流式视频视觉语言模型面临两个基本约束：因果性（无法访问未来帧）和累积性（令牌无限增长）。现有方法仅调节后大语言模型的kv-cache，而未解决成本高昂的前大语言模型prefill问题。

### 目的

开发一个无需训练、即插即用的两阶段框架，解决流式视频视觉语言模型的前后大语言模型瓶颈，实现可预测延迟的高效处理。

### 方法

StreamingTOM采用两阶段框架：1）因果时间减少：为每帧设定固定预算，基于相邻帧变化和令牌显著性选择令牌，只处理紧凑的视觉子集；2）在线量化存储：以4位格式存储令牌，按需检索相关组并解量化，保持活跃kv-cache有界。

### 主要发现

实验表明，StreamingTOM实现了15.7倍的kv-cache压缩，1.2倍的更低峰值内存和2倍的更快TTFT。在无需训练的方法中保持最先进精度，在离线基准测试上平均达到63.8%，在RVS上达到55.8%/3.7。

### 结论

两阶段方法在具有有界增长的流式视频理解中具有实际优势，有效解决了因果性和累积性约束带来的效率瓶颈。

### 翻译

与离线处理不同，流式视频视觉语言模型面临两个基本约束：因果性和累积性。因果性阻止了对离线方法利用的未来帧的访问，而累积性导致令牌无限增长，造成效率瓶颈。然而，现有方法只调节后大语言模型的kv-cache，而保持成本高昂的前大语言模型prefill不变。我们引入了StreamingTOM，这是一个无需训练、即插即用的两阶段框架，通过可预测延迟解决了前大语言模型和后大语言模型的瓶颈。因果时间减少设定了固定的每帧预算，并根据相邻帧变化和令牌显著性选择令牌，通过每帧只处理紧凑的视觉令牌子集而非所有视觉令牌，显著降低了每帧prefill成本。在线量化存储以4位格式存储令牌，按需检索相关组并解量化，无论流长度如何保持活跃kv-cache有界。实验证明，我们的方法相比先前最先进技术实现了15.7倍的kv-cache压缩，1.2倍的更低峰值内存和2倍的更快TTFT。在无需训练的方法中，StreamingTOM在离线基准测试上平均保持63.8%的精度，在RVS上保持55.8%/3.7的精度。这些结果突显了我们的两阶段方法在具有有界增长的流式视频理解中的实际优势。


### 论文摘要

Unlike offline processing, streaming video vision-language models face two fundamental constraints: causality and accumulation. Causality prevents access to future frames that offline methods exploit, while accumulation causes tokens to grow unbounded, creating efficiency bottlenecks. However, existing approaches only regulate post-LLM kv-cache, leaving costly pre-LLM prefill unchanged. We introduce StreamingTOM, a training-free, plug-and-play two-stage framework that addresses both pre-LLM and post-LLM bottlenecks with predictable latency. Causal Temporal Reduction imposes a fixed per-frame budget and selects tokens based on adjacent-frame changes and token saliency, drastically reducing per-frame prefill cost by processing only a compact subset of visual tokens per frame instead of all visual tokens. Online Quantized Memory stores tokens in 4-bit format, retrieves relevant groups on demand, and dequantizes them, keeping the active kv-cache bounded regardless of stream length. Experiments demonstrate our method achieves $15.7\times$ kv-cache compression, $1.2\times$ lower peak memory and $2\times$ faster TTFT compared to prior SOTA. StreamingTOM maintains state-of-the-art accuracy among training-free methods with an average of $63.8\%$ on offline benchmarks and $55.8\%/3.7$ on RVS. These results highlight the practical benefits of our two-stage approach for efficient streaming video understanding with bounded growth.

---

## 23. ViBED-Net: Video Based Engagement Detection Network Using Face-Aware and Scene-Aware Spatiotemporal Cues

**论文链接:** [http://arxiv.org/abs/2510.18016v1](http://arxiv.org/abs/2510.18016v1)

**作者:** Prateek Gothwal, Deeptimaan Banerjee, Ashis Kumer Biswas

**发布时间:** 2025-10-20

**备注:** 10 pages, 4 figures, 2 tables

### GPT解析

### 总结

该研究提出了一种名为ViBED-Net的新型深度学习框架，用于从视频数据中检测学生在在线学习环境中的参与度，通过结合面部表情和场景上下文信息，实现了73.43%的准确率，优于现有方法。

### 背景

在线学习环境中的参与度检测对于提高学生成果和个性化教学至关重要，但现有的检测方法仍有改进空间。

### 目的

开发一种能够从视频数据中准确评估学生参与度的深度学习框架，通过结合面部表情和全场景上下文信息提高检测准确性。

### 方法

设计了ViBED-Net双流架构，使用EfficientNetV2进行空间特征提取，分别处理面部区域和整个视频帧；采用LSTM和Transformer编码器进行时间建模；在DAiSEE数据集上评估；应用有针对性的数据增强技术提高代表性不足类别的性能。

### 主要发现

ViBED-Net与LSTM结合的变体达到73.43%的准确率，优于现有最先进方法；结合面部感知和场景感知的时空线索显著提高了参与度检测准确性；模块化设计使其可灵活应用于教育、用户体验研究和内容个性化。

### 结论

ViBED-Net为视频情感计算提供了可扩展的高性能解决方案，可用于现实世界的参与度分析，推进了视频情感计算领域的发展。

### 翻译

在线学习环境中的参与度检测对于提高学生成果和个性化教学至关重要。我们提出了ViBED-Net（基于视频的参与度检测网络），这是一种新颖的深度学习框架，采用双流架构设计，用于从视频数据中评估学生参与度。ViBED-Net通过EfficientNetV2处理面部裁剪和整个视频帧来捕获面部表情和全场景上下文，进行空间特征提取。然后，使用两种时间建模策略分析这些特征：长短期记忆网络和Transformer编码器。我们的模型在DAiSEE数据集上进行了评估，这是一个大规模的电子学习情感状态识别基准。为了提高代表性不足的参与度类别的性能，我们应用了有针对性的数据增强技术。在测试的变体中，带有LSTM的ViBED-Net实现了73.43%的准确率，优于现有的最先进方法。ViBED-Net证明，结合面部感知和场景感知的时空线索显著提高了参与度检测的准确性。其模块化设计使其具有灵活性，可应用于教育、用户体验研究和内容个性化。这项工作通过为现实世界的参与度分析提供可扩展的高性能解决方案，推动了视频情感计算的发展。该项目的源代码可在https://github.com/prateek-gothwal/ViBED-Net获取。


### 论文摘要

Engagement detection in online learning environments is vital for improving student outcomes and personalizing instruction. We present ViBED-Net (Video-Based Engagement Detection Network), a novel deep learning framework designed to assess student engagement from video data using a dual-stream architecture. ViBED-Net captures both facial expressions and full-scene context by processing facial crops and entire video frames through EfficientNetV2 for spatial feature extraction. These features are then analyzed over time using two temporal modeling strategies: Long Short-Term Memory (LSTM) networks and Transformer encoders. Our model is evaluated on the DAiSEE dataset, a large-scale benchmark for affective state recognition in e-learning. To enhance performance on underrepresented engagement classes, we apply targeted data augmentation techniques. Among the tested variants, ViBED-Net with LSTM achieves 73.43\% accuracy, outperforming existing state-of-the-art approaches. ViBED-Net demonstrates that combining face-aware and scene-aware spatiotemporal cues significantly improves engagement detection accuracy. Its modular design allows flexibility for application across education, user experience research, and content personalization. This work advances video-based affective computing by offering a scalable, high-performing solution for real-world engagement analysis. The source code for this project is available on https://github.com/prateek-gothwal/ViBED-Net .

---

## 24. LongInsightBench: A Comprehensive Benchmark for Evaluating Omni-Modal Models on Human-Centric Long-Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.17305v2](http://arxiv.org/abs/2510.17305v2)

**作者:** ZhaoYang Han, Qihan Lin, Hao Liang, Bowen Chen, Zhou Liu, Wentao Zhang

**发布时间:** 2025-10-20

**备注:** Submitted to ARR Rolling Review

### GPT解析

### 总结

LongInsightBench是首个专门评估模型理解长视频能力的基准测试，整合视觉、音频和文本多模态信息，包含长时长信息密集视频、多样任务场景和严格质量保证流程，实验显示全模态模型在精确时间定位和长距离因果推断任务中仍有挑战。

### 背景

现有模型在理解长视频方面的能力缺乏系统评估基准，特别是对于包含丰富人类语言、视角、动作和其他上下文元素的长视频内容。

### 目的

开发一个专门的基准测试来评估模型在理解长视频时整合视觉、音频和文本多模态信息的能力，重点关注人类语言、视角、动作等上下文元素。

### 方法

1) 从开源数据集FineVideo精心筛选约1000个长时长、信息密集的视频，重点关注讲座、访谈和vlog等包含丰富语言元素的内容；2) 设计六种具有挑战性的任务场景，包括事件内任务和事件间任务；3) 开发三步半自动数据质量保证流程，确保合成问题和答案选项的难度与有效性。

### 主要发现

全模态模型(OLMs)在需要精确时间定位和长距离因果推断的任务中面临挑战；扩展实验揭示了OLMs多模态融合中存在信息损失和处理偏差问题。

### 结论

当前模型在长视频理解方面仍有明显局限性，特别是在需要精确时间定位和长距离因果推断的任务中，多模态融合过程中存在信息损失和处理偏差。

### 翻译

我们引入LongInsightBench，这是首个专门用于评估模型理解长视频能力的基准测试，重点关注人类语言、视角、动作和其他上下文元素，同时整合视觉、音频和文本多模态信息。我们的基准在三个关键方面表现出色：a) 长时长、信息密集的视频：我们基于时长限制和视觉与音频模态的信息密度，从开源数据集FineVideo中精心选择了约1000个视频，重点关注包含丰富语言元素的内容，如讲座、访谈和vlog；b) 多样且有挑战性的任务场景：我们设计了六种具有挑战性的任务场景，包括事件内任务和事件间任务；c) 严格且全面的质量保证流程：我们开发了一个三步半自动数据质量保证流程，以确保合成问题和答案选项的难度和有效性。基于LongInsightBench，我们设计了一系列实验。实验结果表明，全模态模型(OLMs)在需要精确时间定位(T-Loc)和长距离因果推断(CE-Caus)的任务中仍面临挑战。扩展实验揭示了OLMs多模态融合中的信息损失和处理偏差。我们的数据集和代码可在https://anonymous.4open.science/r/LongInsightBench-910F/获取。


### 论文摘要

We introduce \textbf{LongInsightBench}, the first benchmark designed to assess models' ability to understand long videos, with a focus on human language, viewpoints, actions, and other contextual elements, while integrating \textbf{visual, audio, and text} modalities. Our benchmark excels in three key areas: \textbf{a) Long-Duration, Information-Dense Videos:} We carefully select approximately 1,000 videos from open-source datasets FineVideo based on duration limit and the information density of both visual and audio modalities, focusing on content like lectures, interviews, and vlogs, which contain rich language elements. \textbf{b) Diverse and Challenging Task Scenarios:} We have designed six challenging task scenarios, including both Intra-Event and Inter-Event Tasks. \textbf{c) Rigorous and Comprehensive Quality Assurance Pipelines:} We have developed a three-step, semi-automated data quality assurance pipeline to ensure the difficulty and validity of the synthesized questions and answer options. Based on LongInsightBench, we designed a series of experiments. Experimental results shows that Omni-modal models(OLMs) still face challenge in tasks requiring precise temporal localization (T-Loc) and long-range causal inference (CE-Caus). Extended experiments reveal the information loss and processing bias in multi-modal fusion of OLMs. Our dataset and code is available at https://anonymous.4open.science/r/LongInsightBench-910F/.

---

## 25. ProCLIP: Progressive Vision-Language Alignment via LLM-based Embedder

**论文链接:** [http://arxiv.org/abs/2510.18795v1](http://arxiv.org/abs/2510.18795v1)

**作者:** Xiaoxing Hu, Kaicheng Yang, Ziyong Feng, Qi Ming, Zonghao Guo, Xiang An, Ziyong Feng, Junchi Yan, Xue Yang

**发布时间:** 2025-10-21

**备注:** 17 pages, 5 fiugres

### GPT解析

### 总结

ProCLIP是一种基于课程学习的渐进式视觉-语言对齐框架，旨在解决原始CLIP文本编码器在处理长文本、多语言理解和细粒度语义理解方面的局限性。

### 背景

原始CLIP文本编码器受限于77个token的最大输入长度，不支持多语言输入，限制了其在广泛任务中的应用。最近研究尝试用基于LLM的嵌入器替换CLIP文本编码器，但LLM和CLIP的表示空间独立预训练，直接对比学习会破坏CLIP图像编码器的视觉-语言对齐。

### 目的

提出ProCLIP框架，有效对齐CLIP图像编码器和基于LLM的嵌入器，解决CLIP文本编码器的局限性。

### 方法

ProCLIP首先从CLIP文本编码器向LLM嵌入器蒸馏知识，建立初始对齐；然后通过图像-文本对比调进一步对齐，使用自蒸馏正则化避免过拟合；采用实例语义对齐损失和嵌入结构对齐损失提高对齐效果。

### 主要发现

结合知识蒸馏和对比学习的渐进式对齐方法能够有效利用CLIP的预训练知识，同时保持视觉-语言对齐质量。

### 结论

ProCLIP框架能够解决CLIP文本编码器的局限性，实现更好的长文本处理、多语言理解和细粒度语义理解能力。

### 翻译

原始CLIP文本编码器受限于77个token的最大输入长度，这阻碍了它有效处理长文本和执行细粒度语义理解的能力。此外，CLIP文本编码器不支持多语言输入。所有这些限制显著限制了它在更广泛任务中的适用性。最近的研究试图用基于LLM的嵌入器替换CLIP文本编码器，以增强其在处理长文本、多语言理解和细粒度语义理解方面的能力。然而，由于LLM和CLIP的表示空间是独立预训练的，没有先验对齐，使用对比学习直接对齐会破坏CLIP图像编码器中的内在视觉-语言对齐，导致预训练期间获取的知识利用不足。为应对这一挑战，我们提出了ProCLIP，一种基于课程学习的渐进式视觉-语言对齐框架，有效对齐CLIP图像编码器和基于LLM的嵌入器。具体来说，ProCLIP首先从CLIP文本编码器向基于LLM的嵌入器蒸馏知识，利用CLIP丰富的预训练知识，同时建立LLM嵌入器和CLIP图像编码器之间的初始对齐。随后，ProCLIP通过图像-文本对比调进一步对齐CLIP图像编码器和基于LLM的嵌入器，采用自蒸馏正则化避免过拟合。为了实现更有效的对齐，在表示继承和对比调优过程中使用了实例语义对齐损失和嵌入结构对齐损失。代码可在https://github.com/VisionXLab/ProCLIP获取。


### 论文摘要

The original CLIP text encoder is limited by a maximum input length of 77 tokens, which hampers its ability to effectively process long texts and perform fine-grained semantic understanding. In addition, the CLIP text encoder lacks support for multilingual inputs. All these limitations significantly restrict its applicability across a broader range of tasks. Recent studies have attempted to replace the CLIP text encoder with an LLM-based embedder to enhance its ability in processing long texts, multilingual understanding, and fine-grained semantic comprehension. However, because the representation spaces of LLMs and the vision-language space of CLIP are pretrained independently without alignment priors, direct alignment using contrastive learning can disrupt the intrinsic vision-language alignment in the CLIP image encoder, leading to an underutilization of the knowledge acquired during pre-training. To address this challenge, we propose ProCLIP, a curriculum learning-based progressive vision-language alignment framework to effectively align the CLIP image encoder with an LLM-based embedder. Specifically, ProCLIP first distills knowledge from CLIP's text encoder into the LLM-based embedder to leverage CLIP's rich pretrained knowledge while establishing initial alignment between the LLM embedder and CLIP image encoder. Subsequently, ProCLIP further aligns the CLIP image encoder with the LLM-based embedder through image-text contrastive tuning, employing self-distillation regularization to avoid overfitting. To achieve a more effective alignment, instance semantic alignment loss and embedding structure alignment loss are employed during representation inheritance and contrastive tuning. The Code is available at https://github.com/VisionXLab/ProCLIP

---

## 26. SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery

**论文链接:** [http://arxiv.org/abs/2510.18740v1](http://arxiv.org/abs/2510.18740v1)

**作者:** Zhenqi He, Yuanpei Liu, Kai Han

**发布时间:** 2025-10-21

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

这篇论文提出了SEAL框架，通过利用自然层次结构和创新的对比学习方法解决了广义类别发现中的挑战，在多个基准测试上实现了最先进的性能，并展示了良好的泛化能力。

### 背景

广义类别发现在部分标记数据集上的目标是分类所有未标记图像，无论它们属于已知还是未知类别。现有方法通常依赖于单层语义或手动设计的抽象层次结构，限制了泛化能力和可扩展性。

### 目的

为了解决现有方法的局限性，引入一个由自然存在且易于访问的层次结构指导的语义感知层次学习框架(SEAL)。

### 方法

在SEAL框架中，提出层次语义引导的软对比学习方法，利用层次相似性生成信息丰富的软负样本；同时设计跨粒度一致性(CGC)模块，对齐不同粒度级别的预测。

### 主要发现

SEAL在细粒度基准测试上持续实现了最先进的性能，包括SSB基准、Oxford-Pet和Herbarium19数据集，并在粗粒度数据集上进一步展示了泛化能力。

### 结论

SEAL框架通过利用自然层次结构和创新的对比学习方法，有效解决了广义类别发现中的挑战，提高了模型的泛化能力和可扩展性。

### 翻译

这篇论文研究了广义类别发现(GCD)的问题。给定一个部分标记的数据集，GCD旨在对所有未标记图像进行分类，无论它们属于已知类别还是未知类别。现有方法通常依赖于单层语义或手动设计的抽象层次结构，这限制了它们的泛化能力和可扩展性。为了解决这些局限性，我们引入了一个由自然存在且易于访问的层次结构指导的语义感知层次学习框架(SEAL)。在SEAL中，我们提出了一种层次语义引导的软对比学习方法，利用层次相似性生成信息丰富的软负样本，解决了传统对比损失将所有负样本同等对待的局限性。此外，还设计了一个跨粒度一致性(CGC)模块，用于对齐不同粒度级别的预测。SEAL在细粒度基准测试上持续实现了最先进的性能，包括SSB基准、Oxford-Pet和Herbarium19数据集，并在粗粒度数据集上进一步展示了泛化能力。项目页面：https://visual-ai.github.io/seal/


### 论文摘要

This paper investigates the problem of Generalized Category Discovery (GCD). Given a partially labelled dataset, GCD aims to categorize all unlabelled images, regardless of whether they belong to known or unknown classes. Existing approaches typically depend on either single-level semantics or manually designed abstract hierarchies, which limit their generalizability and scalability. To address these limitations, we introduce a SEmantic-aware hierArchical Learning framework (SEAL), guided by naturally occurring and easily accessible hierarchical structures. Within SEAL, we propose a Hierarchical Semantic-Guided Soft Contrastive Learning approach that exploits hierarchical similarity to generate informative soft negatives, addressing the limitations of conventional contrastive losses that treat all negatives equally. Furthermore, a Cross-Granularity Consistency (CGC) module is designed to align the predictions from different levels of granularity. SEAL consistently achieves state-of-the-art performance on fine-grained benchmarks, including the SSB benchmark, Oxford-Pet, and the Herbarium19 dataset, and further demonstrates generalization on coarse-grained datasets. Project page: https://visual-ai.github.io/seal/

---

## 27. ε-Seg: Sparsely Supervised Semantic Segmentation of Microscopy Data

**论文链接:** [http://arxiv.org/abs/2510.18637v1](http://arxiv.org/abs/2510.18637v1)

**作者:** Sheida Rahnamai Kordasiabi, Damian Dalle Nogare, Florian Jug

**发布时间:** 2025-10-21

**备注:** 10 pages main text, 17 pages total

### GPT解析

### 总结

ε-Seg是一种创新的分层变分自编码器方法，结合了中心区域掩蔽、稀疏标签对比学习、高斯混合模型先验和无聚类标签预测等技术，用于解决电子显微镜图像的生物样本语义分割挑战，特别是在标签稀疏的情况下表现优异。

### 背景

电子显微镜(EM)图像的生物样本语义分割在生命科学中仍然是一个挑战。EM数据捕获生物结构的细节，有时复杂到即使是人类观察者也会感到难以处理。

### 目的

引入一种名为ε-Seg的方法，用于解决EM图像的语义分割问题，特别是在标签稀疏的情况下(总图像数据的0.05%或更少)。

### 方法

基于分层变分自编码器(HVAEs)的方法，采用中心区域掩蔽和修复损失来学习鲁棒和代表性的嵌入，使用对比学习和高斯混合模型先验来塑造潜在空间，并通过MLP语义分割头直接从潜在嵌入预测类标签，而不是聚类潜在嵌入。

### 主要发现

ε-Seg在两个密集的生物组织EM数据集上展示了经验结果，该方法也适用于荧光显微镜数据，能够在复杂的生物图像数据上实现具有竞争力的稀疏监督分割结果，即使只有有限的训练标签可用。

### 结论

ε-Seg是一种有效的方法，可以在标签稀疏的情况下进行生物图像的语义分割，为生命科学领域的图像分析提供了新的解决方案。

### 翻译

生物样本的电子显微镜(EM)图像语义分割在生命科学中仍然是一个挑战。EM数据捕获生物结构的细节，有时复杂到即使是人类观察者也会感到难以处理。我们引入ε-Seg，一种基于分层变分自编码器(HVAEs)的方法，采用中心区域掩蔽、稀疏标签对比学习(CL)、高斯混合模型(GMM)先验和无聚类标签预测。中心区域掩蔽和修复损失鼓励模型学习鲁棒和代表性的嵌入来区分所需的类别，即使训练标签稀疏(占总图像数据的0.05%或更少)。为了获得最佳性能，我们采用CL和GMM先验来塑造HVAE的潜在空间，使得编码的输入斑块倾向于关于我们希望区分的语义类别进行聚类。最后，我们不是对潜在嵌入进行聚类以进行语义分割，而是提出一个MLP语义分割头来直接从潜在嵌入预测类标签。我们在两个密集的生物组织EM数据集上展示了ε-Seg和基线方法的经验结果，并证明了我们的方法在荧光显微镜数据上的适用性。我们的结果表明，即使在只有有限训练标签可用的情况下，ε-Seg也能够在复杂的生物图像数据上实现具有竞争力的稀疏监督分割结果。


### 论文摘要

Semantic segmentation of electron microscopy (EM) images of biological samples remains a challenge in the life sciences. EM data captures details of biological structures, sometimes with such complexity that even human observers can find it overwhelming. We introduce {\epsilon}-Seg, a method based on hierarchical variational autoencoders (HVAEs), employing center-region masking, sparse label contrastive learning (CL), a Gaussian mixture model (GMM) prior, and clustering-free label prediction. Center-region masking and the inpainting loss encourage the model to learn robust and representative embeddings to distinguish the desired classes, even if training labels are sparse (0.05% of the total image data or less). For optimal performance, we employ CL and a GMM prior to shape the latent space of the HVAE such that encoded input patches tend to cluster wrt. the semantic classes we wish to distinguish. Finally, instead of clustering latent embeddings for semantic segmentation, we propose a MLP semantic segmentation head to directly predict class labels from latent embeddings. We show empirical results of {\epsilon}-Seg and baseline methods on 2 dense EM datasets of biological tissues and demonstrate the applicability of our method also on fluorescence microscopy data. Our results show that {\epsilon}-Seg is capable of achieving competitive sparsely-supervised segmentation results on complex biological image data, even if only limited amounts of training labels are available.

---

## 28. CovMatch: Cross-Covariance Guided Multimodal Dataset Distillation with Trainable Text Encoder

**论文链接:** [http://arxiv.org/abs/2510.18583v1](http://arxiv.org/abs/2510.18583v1)

**作者:** Yongmin Lee, Hye Won Chung

**发布时间:** 2025-10-21

**备注:** NeurIPS 2025

### GPT解析

### 总结

CovMatch是一种可扩展的多模态数据集蒸馏框架，通过联合优化两个编码器实现更强的跨模态对齐和改进的性能，在Flickr30K和COCO数据集上优于最先进方法，仅使用500个合成对即可实现高达6.8%的检索精度提升。

### 背景

多模态数据集蒸馏旨在合成少量图像-文本对以高效训练大规模视觉-语言模型。虽然数据集蒸馏在单模态任务中显示出前景，但扩展到多模态对比学习面临关键挑战：学习跨模态对齐和管理大型编码器的高计算成本。

### 目的

提出CovMatch框架，解决先前方法中语义对齐受限和性能扩展瓶颈的问题，实现更有效的多模态数据集蒸馏。

### 方法

CovMatch通过使真实和合成特征的跨协方差对齐，同时正则化每个模态内的特征分布。与先前方法不同，CovMatch能够联合优化两个编码器，而非仅更新图像编码器和文本投影层。

### 主要发现

先前方法通过冻结文本编码器来解决可扩展性问题，但这种方法严重限制了语义对齐，成为性能扩展的瓶颈。

### 结论

在Flickr30K和COCO上评估，CovMatch优于最先进的多模态蒸馏方法，仅使用500个合成对即可实现高达6.8%的检索精度绝对提升。

### 翻译

多模态数据集蒸馏旨在合成一组小的图像-文本对，以实现大规模视觉-语言模型的高效训练。虽然数据集蒸馏在单模态任务中显示出前景，但将其扩展到多模态对比学习存在关键挑战：学习跨模态对齐和管理大型编码器的高计算成本。先前的方法通过冻结文本编码器并仅更新图像编码器和文本投影层来解决可扩展性问题。然而，我们发现这严重限制了语义对齐，成为性能扩展的瓶颈。我们提出CovMatch，一种可扩展的数据集蒸馏框架，使真实和合成特征的跨协方差对齐，同时正则化每个模态内的特征分布。与先前的方法不同，CovMatch能够联合优化两个编码器，从而实现更强的跨模态对齐和改进的性能。在Flickr30K和COCO上评估，CovMatch优于最先进的多模态蒸馏方法，仅使用500个合成对即可实现高达6.8%的检索精度绝对提升。


### 论文摘要

Multimodal dataset distillation aims to synthesize a small set of image-text pairs that enables efficient training of large-scale vision-language models. While dataset distillation has shown promise in unimodal tasks, extending it to multimodal contrastive learning presents key challenges: learning cross-modal alignment and managing the high computational cost of large encoders. Prior approaches address scalability by freezing the text encoder and update only the image encoder and text projection layer. However, we find this severely limits semantic alignment and becomes a bottleneck for performance scaling. We propose CovMatch, a scalable dataset distillation framework that aligns the cross-covariance of real and synthetic features while regularizing feature distributions within each modality. Unlike prior approaches, CovMatch enables joint optimization of both encoders, leading to stronger cross-modal alignment and improved performance. Evaluated on Flickr30K and COCO, CovMatch outperforms state-of-the-art multimodal distillation methods and achieves up to 6.8% absolute gains in retrieval accuracy using only 500 synthetic pairs.

---

## 29. AWSPNet: Attention-based Dual-Tree Wavelet Scattering Prototypical Network for MIMO Radar Target Recognition and Jamming Suppression

**论文链接:** [http://arxiv.org/abs/2510.18422v1](http://arxiv.org/abs/2510.18422v1)

**作者:** Yizhen Jia, Siyao Xiao, Wenkai Jia, Hui Chen, Wen-Qin Wang

**发布时间:** 2025-10-21

**备注:** 13 pages, 10 figures, The code is available in  https://github.com/jiaxuanzhi/AwspNet

### GPT解析

### 总结

本文提出了一种名为AWSPNet的新型深度学习框架，用于雷达目标识别和干扰抑制。该方法结合了双树复小波变换、注意力机制、预训练网络和监督对比学习，在低信噪比环境下表现优异，具有良好的特征可分离性和泛化能力。通过与滑动窗口方法的集成，形成完整且实用的干扰识别与抑制系统。

### 背景

基于数字射频存储器的电子对抗措施日益增多，对雷达系统的生存能力和有效性构成重大威胁。这些干扰器能生成大量欺骗性虚假目标，淹没雷达的处理能力并掩盖真实目标。因此，稳健地区分真实目标和复杂干扰信号的能力，特别是在低信噪比环境下，显得尤为重要。

### 目的

开发一种能够稳健区分真实目标和复杂干扰信号的框架，特别关注低信噪比环境下的性能，实现雷达目标识别和干扰抑制的同时处理。

### 方法

提出了基于注意力的双树小波散射原型网络(AWSPNet)，利用双树复小波变换提取对噪声和信号平移具有内在鲁棒性的特征，通过注意力机制和预训练骨干网络进一步优化这些特征。采用监督对比学习策略解决标记数据有限的问题并增强泛化能力，使用原型网络进行分类。通过t-SNE可视化提供网络内部工作的物理解释，并将AWSPNet与时域滑动窗口方法集成形成完整算法。

### 主要发现

在-6 dB信噪比条件下，AWSPNet达到了90.45%的准确率。t-SNE可视化显示模型不同阶段的特征可分离性良好，集成算法不仅能识别还能有效抑制各种类型的干扰。

### 结论

AWSPNet在复杂电磁环境中具有实际应用潜力，能够有效处理低信噪比环境下的目标识别和干扰抑制问题。

### 翻译

基于数字射频存储器的电子对抗措施日益增多，对雷达系统的生存能力和有效性构成重大威胁。这些干扰器能生成大量欺骗性虚假目标，淹没雷达的处理能力并掩盖真实目标。因此，稳健地区分真实目标和复杂干扰信号的能力，特别是在低信噪比环境下，显得尤为重要。本文介绍了基于注意力的双树小波散射原型网络(AWSPNet)，一种为同时进行雷达目标识别和干扰抑制而设计的深度学习框架。AWSPNet的核心是一个编码器，它利用双树复小波变换提取对噪声和信号平移具有内在鲁棒性的特征。这些特征通过注意力机制和预训练骨干网络得到进一步优化。为了解决标记数据有限的问题并增强泛化能力，我们在训练阶段采用了监督对比学习策略。分类由原型网络执行，该网络在少样本学习场景中特别有效，能够快速适应新的信号类型。我们通过大量实验证明了该方法的有效性。结果显示，AWSPNet在-6 dB信噪比下达到90.45%的准确率。此外，我们通过t-SNE可视化提供了网络内部工作的物理解释，分析了模型不同阶段特征的可分离性。最后，通过将AWSPNet与时域滑动窗口方法集成，我们提出了一个不仅能识别还能有效抑制各种类型干扰的完整算法，从而验证了其在复杂电磁环境中实际应用的潜力。


### 论文摘要

The increasing of digital radio frequency memory based electronic countermeasures poses a significant threat to the survivability and effectiveness of radar systems. These jammers can generate a multitude of deceptive false targets, overwhelming the radar's processing capabilities and masking targets. Consequently, the ability to robustly discriminate between true targets and complex jamming signals, especially in low signal-to-noise ratio (SNR) environments, is of importance. This paper introduces the attention-based dual-tree wavelet scattering prototypical network (AWSPNet), a deep learning framework designed for simultaneous radar target recognition and jamming suppression. The core of AWSPNet is the encoder that leverages the dual-tree complex wavelet transform to extract features that are inherently robust to noise and signal translations. These features are further refined by an attention mechanism and a pre-trained backbone network. To address the challenge of limited labeled data and enhance generalization, we employ a supervised contrastive learning strategy during the training phase. The classification is performed by a prototypical network, which is particularly effective in few-shot learning scenarios, enabling rapid adaptation to new signal types. We demonstrate the efficacy of our approach through extensive experiments. The results show that AWSPNet achieves 90.45\% accuracy at -6 dB SNR. Furthermore, we provide a physical interpretation of the network's inner workings through t-SNE visualizations, which analyze the feature separability at different stages of the model. Finally, by integrating AWSPNet with a time-domain sliding window approach, we present a complete algorithm capable of not only identifying but also effectively suppressing various types of jamming, thereby validating its potential for practical application in complex electromagnetic environments.

---

## 30. Enhancing Few-Shot Classification of Benchmark and Disaster Imagery with ATTBHFA-Net

**论文链接:** [http://arxiv.org/abs/2510.18326v1](http://arxiv.org/abs/2510.18326v1)

**作者:** Gao Yu Lee, Tanmoy Dam, Md Meftahul Ferdaus, Daniel Puiu Poenar, Vu Duong

**发布时间:** 2025-10-21

**备注:** Submitted to a SN journal

### GPT解析

### 总结

本文提出了一种基于注意力的Bhattacharyya-Hellinger特征聚合网络(ATTBHFA-Net)，用于解决灾害图像分类中的少样本学习问题。该方法通过线性组合Bhattacharyya系数和Hellinger距离来比较和聚合特征概率分布，形成鲁棒的原型，并提出了基于Bhattacharyya-Hellinger距离的对比损失。实验表明，该方法在四个FSL基准和两个灾害图像数据集上表现出优越的有效性和泛化能力。

### 背景

自然和人为灾害的频率增加需要先进的视觉识别技术分析关键摄影数据。人工智能和弹性计算系统的进步使快速准确的灾害分类对有效救援行动变得至关重要。然而，灾害背景下的视觉识别面临数据有限且多样的挑战，难以收集和整理全面的高质量灾害图像。

### 目的

解决灾害图像分类中的数据稀缺问题，提高少样本学习在灾害图像分类中的性能，克服灾害图像高类内变异和类间相似性的挑战。

### 方法

引入了基于注意力的Bhattacharyya-Hellinger特征聚合网络(ATTBHFA-Net)，线性组合Bhattacharyya系数和Hellinger距离来比较和聚合特征概率分布形成鲁棒原型。Bhattacharyya系数作为对比边界增强类间可分性，Hellinger距离对同类对齐进行正则化。提出基于Bhattacharyya-Hellinger距离的对比损失作为余弦相似度损失的分布对应物，与分类交叉熵结合使用提高FSL性能。

### 主要发现

在四个FSL基准和两个灾害图像数据集上的实验表明，ATTBHFA-Net与现有方法相比具有优越的有效性和泛化能力。该方法能够有效处理灾害图像的高类内变异和类间相似性问题，提高少样本学习在灾害图像分类中的性能。

### 结论

ATTBHFA-Net为灾害图像分类中的少样本学习提供了有效解决方案。通过结合Bhattacharyya系数和Hellinger距离，该方法能够形成更鲁棒的原型，有效处理灾害图像的复杂特征分布，显著提高分类性能。

### 翻译

自然和人为灾害频率的增加需要能够分析关键摄影数据的先进视觉识别技术。随着人工智能和弹性计算系统的进步，快速准确的灾害分类对高效救援行动变得至关重要。然而，由于数据有限且多样，难以收集和整理全面的高质量灾害图像，灾害背景下的视觉识别面临重大挑战。少样本学习为数据稀缺问题提供了有前景的方法，但现有的FSL研究主要依赖于缺乏遥感灾害图像的通用基准数据集，限制了其实际有效性。此外，灾害图像表现出高的类内变异和类间相似性，阻碍了基于度量的传统FSL方法的性能。为解决这些问题，本文引入了基于注意力的Bhattacharyya-Hellinger特征聚合网络(ATTBHFA-Net)，该网络线性组合Bhattacharyya系数和Hellinger距离来比较和聚合特征概率分布，形成鲁棒的原型。Bhattacharyya系数作为对比边界增强类间可分性，而Hellinger距离对同类对齐进行正则化。该框架类似于对比学习，但在概率分布上运行，而不是嵌入特征点。此外，提出了基于Bhattacharyya-Hellinger距离的对比损失，作为余弦相似度损失的分布对应物，与分类交叉熵结合使用，显著提高FSL性能。在四个FSL基准和两个灾害图像数据集上的实验表明，与现有方法相比，ATTBHFA-Net具有优越的有效性和泛化能力。


### 论文摘要

The increasing frequency of natural and human-induced disasters necessitates advanced visual recognition techniques capable of analyzing critical photographic data. With progress in artificial intelligence and resilient computational systems, rapid and accurate disaster classification has become crucial for efficient rescue operations. However, visual recognition in disaster contexts faces significant challenges due to limited and diverse data from the difficulties in collecting and curating comprehensive, high-quality disaster imagery. Few-Shot Learning (FSL) provides a promising approach to data scarcity, yet current FSL research mainly relies on generic benchmark datasets lacking remote-sensing disaster imagery, limiting its practical effectiveness. Moreover, disaster images exhibit high intra-class variation and inter-class similarity, hindering the performance of conventional metric-based FSL methods. To address these issues, this paper introduces the Attention-based Bhattacharyya-Hellinger Feature Aggregation Network (ATTBHFA-Net), which linearly combines the Bhattacharyya coefficient and Hellinger distances to compare and aggregate feature probability distributions for robust prototype formation. The Bhattacharyya coefficient serves as a contrastive margin that enhances inter-class separability, while the Hellinger distance regularizes same-class alignment. This framework parallels contrastive learning but operates over probability distributions rather than embedded feature points. Furthermore, a Bhattacharyya-Hellinger distance-based contrastive loss is proposed as a distributional counterpart to cosine similarity loss, used jointly with categorical cross-entropy to significantly improve FSL performance. Experiments on four FSL benchmarks and two disaster image datasets demonstrate the superior effectiveness and generalization of ATTBHFA-Net compared to existing approaches.

---

## 31. SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection

**论文链接:** [http://arxiv.org/abs/2510.16219v2](http://arxiv.org/abs/2510.16219v2)

**作者:** Yang Feng, Xudong Pan

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了SentinelNet，一个用于多智能体系统中主动检测和缓解恶意行为的去中心化框架，通过基于信用的检测器和对比学习，实现了高精度的恶意智能体检测和系统准确性恢复。

### 背景

恶意智能体对基于大型语言模型的多智能体系统构成重大威胁，影响其可靠性和决策能力。现有防御措施通常采用被动设计或集中式架构，存在单点故障风险。

### 目的

解决现有防御措施的不足，提出一个去中心化的框架，用于主动检测和缓解多智能体协作中的恶意行为。

### 方法

提出SentinelNet框架，为每个智能体配备基于信用的检测器，通过对比学习在增强的对抗性辩论轨迹上训练，实现自主评估消息可信度，并通过bottom-k消除进行动态邻居排名，抑制恶意通信。同时生成模拟多样威胁的对抗性轨迹，以克服攻击数据稀缺问题。

### 主要发现

在多智能体系统基准测试中，SentinelNet实现了近乎完美的恶意智能体检测，两轮辩论内检测率接近100%，从被破坏的基线恢复了95%的系统准确性，并在不同领域和攻击模式中表现出强大的泛化能力。

### 结论

SentinelNet为保护协作式多智能体系统建立了一种新的范式，通过去中心化设计解决了单点故障问题，提高了系统的安全性和可靠性。

### 翻译

恶意智能体对基于大型语言模型的多智能体系统的可靠性和决策能力构成重大威胁。现有防御措施通常因被动设计或引入单点故障风险的集中式架构而不足。为应对这些挑战，我们提出了SentinelNet，这是第一个用于主动检测和缓解多智能体协作中恶意行为的去中心化框架。SentinelNet为每个智能体配备基于信用的检测器，通过在增强的对抗性辩论轨迹上进行对比学习训练，实现自主评估消息可信度，并通过bottom-k消除进行动态邻居排名以抑制恶意通信。为克服攻击数据稀缺问题，它生成模拟多样威胁的对抗性轨迹，确保训练的鲁棒性。在多智能体系统基准测试中，SentinelNet实现了近乎完美的恶意智能体检测，两轮辩论内接近100%，并从被破坏的基线恢复了95%的系统准确性。通过在不同领域和攻击模式中表现出强大的泛化能力，SentinelNet为保护协作式多智能体系统建立了一种新的范式。


### 论文摘要

Malicious agents pose significant threats to the reliability and decision-making capabilities of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs). Existing defenses often fall short due to reactive designs or centralized architectures which may introduce single points of failure. To address these challenges, we propose SentinelNet, the first decentralized framework for proactively detecting and mitigating malicious behaviors in multi-agent collaboration. SentinelNet equips each agent with a credit-based detector trained via contrastive learning on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking via bottom-k elimination to suppress malicious communications. To overcome the scarcity of attack data, it generates adversarial trajectories simulating diverse threats, ensuring robust training. Experiments on MAS benchmarks show SentinelNet achieves near-perfect detection of malicious agents, close to 100% within two debate rounds, and recovers 95% of system accuracy from compromised baselines. By exhibiting strong generalizability across domains and attack patterns, SentinelNet establishes a novel paradigm for safeguarding collaborative MAS.

---

## 32. Detection and Simulation of Urban Heat Islands Using a Fine-Tuned Geospatial Foundation Model for Microclimate Impact Prediction

**论文链接:** [http://arxiv.org/abs/2510.18773v1](http://arxiv.org/abs/2510.18773v1)

**作者:** Jannis Fleckenstein, David Kreismann, Tamara Rosemary Govindasamy, Thomas Brunschwiler, Etienne Vos, Mattia Rigotti

**发布时间:** 2025-10-21

**备注:** 10 pages, 9 figures. Accepted at the NeurIPS 2025 Workshop on  Tackling Climate Change with Machine Learning

### GPT解析

### 总结

这篇论文探讨了如何利用地理空间基础模型来预测城市热岛效应，并通过微调模型来评估缓解策略的有效性。

### 背景

随着城市化和气候变化的发展，城市热岛效应变得越来越频繁和严重。传统机器学习模型因数据有限，尤其是在服务不足的地区，往往产生不准确的预测。

### 目的

为了制定有效的城市热岛缓解计划，需要详细的气温数据，而地理空间基础模型提供了一种有希望的替代方案，表现出强大的泛化能力且只需少量微调。

### 方法

研究通过量化绿色空间的冷却效果建立城市热模式的实证真实基准，并将其与模型预测比较以评估模型准确性。随后对基础模型进行微调，预测未来气候情景下的地表温度，并通过模拟修复演示其实际应用价值。

### 主要发现

基础模型为评估数据稀缺地区的城市热岛缓解策略提供了一种强大的方式。

### 结论

基础模型可以帮助支持更具气候适应能力的城市建设。

### 翻译

随着城市化和气候变化的推进，城市热岛效应正变得更加频繁和严重。为了制定有效的缓解计划，城市需要详细的气温数据，然而传统的机器学习模型因数据有限，尤其是在服务不足的地区，往往产生不准确的预测。基于全球非结构化数据训练的地理空间基础模型提供了一种有希望的替代方案，表现出强大的泛化能力且只需少量微调。在本研究中，通过量化绿色空间的冷却效果并建立城市热模式的实证真实基准，将其与模型预测进行比较以评估模型准确性。随后对基础模型进行微调以预测未来气候情景下的地表温度，并通过模拟修复演示其实际价值。结果表明，基础模型为评估数据稀缺地区的城市热岛缓解策略提供了一种强大方式，以支持更具气候适应能力的城市建设。


### 论文摘要

As urbanization and climate change progress, urban heat island effects are becoming more frequent and severe. To formulate effective mitigation plans, cities require detailed air temperature data, yet conventional machine learning models with limited data often produce inaccurate predictions, particularly in underserved areas. Geospatial foundation models trained on global unstructured data offer a promising alternative by demonstrating strong generalization and requiring only minimal fine-tuning. In this study, an empirical ground truth of urban heat patterns is established by quantifying cooling effects from green spaces and benchmarking them against model predictions to evaluate the model's accuracy. The foundation model is subsequently fine-tuned to predict land surface temperatures under future climate scenarios, and its practical value is demonstrated through a simulated inpainting that highlights its role for mitigation support. The results indicate that foundation models offer a powerful way for evaluating urban heat island mitigation strategies in data-scarce regions to support more climate-resilient cities.

---

## 33. Adapting Language Balance in Code-Switching Speech

**论文链接:** [http://arxiv.org/abs/2510.18724v1](http://arxiv.org/abs/2510.18724v1)

**作者:** Enes Yavuz Ugan, Ngoc-Quan Pham, Alexander Waibel

**发布时间:** 2025-10-21

**备注:** Submitted to ICASSP 2026

### GPT解析

### 总结

大型基础模型在代码转换测试案例中表现不佳，原因可能是代码转换时刻不频繁且第二语言嵌入微妙。研究提出通过为训练过程提供标签并利用语言差异突出代码转换点，减轻生成过程中的上下文偏差，提高模型鲁棒性。

### 背景

大型基础模型在标准基准测试中取得了令人印象深刻的结果，但在代码转换测试案例中仍然表现不佳。

### 目的

提高大型基础模型在代码转换任务中的表现，解决模型在识别和预测代码转换点时的困难。

### 方法

利用嵌入语言和主要语言之间的差异来突出代码转换点，为训练过程提供标签，采用简单有效的可微分替代方法减轻生成过程中的上下文偏差。

### 主要发现

大型基础模型在代码转换测试案例中表现不佳的原因是代码转换时刻不频繁且第二语言嵌入微妙；通过提供标签和突出转换点，模型能够更正确地预测转换位置，替换错误减少。

### 结论

通过为训练过程提供标签并利用语言差异突出代码转换点，可以有效提高大型基础模型在代码转换任务中的鲁棒性，减轻生成过程中的上下文偏差这一核心挑战。

### 翻译

尽管在标准基准测试上取得了令人印象深刻的结果，大型基础模型仍然难以应对代码转换测试案例。当数据稀缺不能作为性能不佳的通常理由时，原因可能在于代码转换时刻的不频繁出现，其中第二语言的嵌入显得微妙。与其期望模型自己学习这种不频繁性，不如为训练过程提供标签。评估模型在代码转换数据上的性能需要仔细定位代码转换点，在这些点上识别错误最为关键，以便分析强调在这些时刻发生的错误。基于这一观察，我们利用嵌入语言和主要语言之间的差异来突出这些代码转换点，从而强调在这些位置的学习。这种简单而有效的可微分替代方法减轻了生成过程中的上下文偏差——代码转换中的核心挑战，从而提高了模型的鲁棒性。我们在阿拉伯语和中文-英语方面的实验表明，模型能够更正确地预测转换位置，表现为替换错误的减少。


### 论文摘要

Despite achieving impressive results on standard benchmarks, large foundational models still struggle against code-switching test cases. When data scarcity cannot be used as the usual justification for poor performance, the reason may lie in the infrequent occurrence of code-switched moments, where the embedding of the second language appears subtly. Instead of expecting the models to learn this infrequency on their own, it might be beneficial to provide the training process with labels. Evaluating model performance on code-switching data requires careful localization of code-switching points where recognition errors are most consequential, so that the analysis emphasizes mistakes occurring at those moments. Building on this observation, we leverage the difference between the embedded and the main language to highlight those code-switching points and thereby emphasize learning at those locations. This simple yet effective differentiable surrogate mitigates context bias during generation -- the central challenge in code-switching -- thereby improving the model's robustness. Our experiments with Arabic and Chinese-English showed that the models are able to predict the switching places more correctly, reflected by the reduced substitution error.

---

## 34. Bayesian Low-Rank Factorization for Robust Model Adaptation

**论文链接:** [http://arxiv.org/abs/2510.18723v1](http://arxiv.org/abs/2510.18723v1)

**作者:** Enes Yavuz Ugan, Ngoc-Quan Pham, Alexander Waibel

**发布时间:** 2025-10-21

**备注:** Submitted to ICASSP 2026

### GPT解析

### 总结

本研究探讨了使用贝叶斯因式分解适配器来适应语音基础模型，以处理代码切换场景，实现在特定领域适应的同时保留基础模型的通用性能。

### 背景

大型语音基础模型在许多领域表现出色，但需要适应本地需求如代码切换(说话者在同一话语中混合使用多种语言)。直接微调这些模型存在过拟合风险，可能会覆盖基础模型的广泛能力。

### 目的

解决语音基础模型在适应特定领域时保留通用性能的挑战，避免过拟合和灾难性遗忘问题。

### 方法

探索贝叶斯因式分解适配器用于语音基础模型，通过将先验设置接近零来实现更稀疏的适应矩阵，从而在适应特定领域的同时保留通用性能。将这种方法应用于Whisper模型，并在不同的多语言代码切换场景中进行评估。

### 主要发现

结果显示方法仅产生最小的适应损失，同时显著减少了基础模型的灾难性遗忘。与LoRA相比，该方法在新领域上仅下降4%，同时实现了54%的向后增益。

### 结论

贝叶斯适应方法在微调语音基础模型时非常有效，可以在不牺牲泛化能力的情况下实现特定领域的适应。

### 翻译

大型语音基础模型在许多领域实现了强大的性能，但它们通常需要适应处理本地需求，如代码切换，即说话者在同一话语中混合语言。直接微调这些模型存在对目标领域过拟合的风险，并可能覆盖基础模型的广泛能力。为解决这一挑战，我们探索了用于语音基础模型的贝叶斯因式分解适配器，它们将先验设置接近零，以实现更稀疏的适应矩阵，从而在适应特定领域的同时保留通用性能。我们将这种方法应用于Whisper模型，并在不同的多语言代码切换场景中进行评估。我们的结果显示，仅产生最小的适应损失，同时显著减少了基础模型的灾难性遗忘。与LoRA相比，我们的方法在新领域上仅下降4%，同时实现了54%的向后增益。这些发现强调了贝叶斯适应在微调语音基础模型而不牺牲泛化能力方面的有效性。


### 论文摘要

Large speech foundation models achieve strong performance across many domains, but they often require adaptation to handle local needs such as code-switching, where speakers mix languages within the same utterance. Direct fine-tuning of these models risks overfitting to the target domain and overwriting the broad capabilities of the base model. To address this challenge, we explore Bayesian factorized adapters for speech foundation models, which place priors near zero to achieve sparser adaptation matrices and thereby retain general performance while adapting to specific domains. We apply our approach to the Whisper model and evaluate on different multilingual code-switching scenarios. Our results show only minimal adaptation loss while significantly reducing catastrophic forgetting of the base model. Compared to LoRA, our method achieves a backward gain of 54% with only a 4% drop on the new domain. These findings highlight the effectiveness of Bayesian adaptation for fine-tuning speech foundation models without sacrificing generalization.

---

## 35. Think with 3D: Geometric Imagination Grounded Spatial Reasoning from Limited Views

**论文链接:** [http://arxiv.org/abs/2510.18632v1](http://arxiv.org/abs/2510.18632v1)

**作者:** Zhangquan Chen, Manyuan Zhang, Xinlei Yu, Xufang Luo, Mingze Sun, Zihao Pan, Yan Feng, Peng Pei, Xunliang Cai, Ruqi Huang

**发布时间:** 2025-10-21

**备注:** 12 pages, 4 figures

### GPT解析

### 总结

该论文提出了3DThinker框架，解决了视觉语言模型从有限视角理解3D空间关系的挑战，通过两阶段训练实现3D思维能力，无需3D先验输入或显式标记的3D数据，在多个基准测试中表现出色。

### 背景

视觉语言模型在多模态任务中取得显著进展，但从有限视角理解3D空间关系仍是重大挑战。先前方法依赖纯文本或2D视觉线索，其有限表示能力阻碍了需要3D空间想象力的任务性能。

### 目的

解决现有方法在3D空间关系理解上的局限性，开发一个能像人类一样在推理过程中利用图像中丰富几何信息的框架。

### 方法

提出3DThinker框架，首个在推理过程中实现3D思维而无需任何3D先验输入的框架。训练分两阶段：首先通过监督训练将VLM生成的3D潜变量与3D基础模型对齐；然后仅基于结果信号优化整个推理轨迹，优化底层3D思维能力。

### 主要发现

在多个基准测试上的广泛实验表明，3DThinker始终优于强大的基线模型，为将3D表示统一到多模态推理中提供了新视角。

### 结论

3DThinker框架成功解决了从有限视角理解3D空间关系的挑战，不依赖3D先验输入或显式标记的3D数据，通过两阶段训练实现了有效的3D思维能力。

### 翻译

尽管视觉语言模型的最新进展在广泛的多模态任务中取得了显著进步，但从有限视角理解3D空间关系仍然是一个重大挑战。先前的推理方法通常依赖纯文本（如拓扑认知图）或2D视觉线索。然而，它们有限的表示能力阻碍了需要3D空间想象力的特定任务性能。为解决这一限制，我们提出了3DThinker，一个能像人类一样在推理过程中有效利用图像中丰富几何信息的框架。我们的框架是首个在推理过程中实现3D思维而无需任何3D先验输入的框架，并且不依赖显式标记的3D数据进行训练。具体来说，我们的训练包括两个阶段。首先，我们进行监督训练，将VLM推理时生成的3D潜变量与3D基础模型（如VGGT）的3D潜变量对齐。然后，我们仅基于结果信号优化整个推理轨迹，从而优化底层的3D思维能力。在多个基准测试上的广泛实验表明，3DThinker始终优于强大的基线模型，并为将3D表示统一到多模态推理中提供了新视角。我们的代码将在https://github.com/zhangquanchen/3DThinker上提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视觉语言模型(VLMs)从有限视角理解3D空间关系的挑战。这个问题很重要，因为空间理解是机器与真实3D世界交互(如具身AI、自动驾驶)的关键能力，这些系统通常依赖于多视角观察，需要从有限视角想象完整场景并进行空间推理。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到人类认知机制的启发，特别是心理意象的认知机制。他们借鉴了Mirage框架利用图像嵌入进行监督训练的思想，以及现有的认知地图构建方法如MindCube和Ego3D。作者发现现有方法要么依赖纯文本或2D视觉线索，要么需要辅助模态或外部工具，因此设计3DThinker框架让VLMs能像人类一样在推理过程中进行3D空间想象。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是让VLMs在推理过程中生成紧凑的3D潜在嵌入作为3D令牌，模拟人类在空间推理中想象的3D场景。整体实现流程分为两阶段：1)监督训练阶段，将VLM生成的3D潜在与3D基础模型对齐，使用3D对齐损失和交叉熵损失优化；2)强化训练阶段，使用基于结果的信号优化整个采样轨迹，保持3D潜在对齐的同时优化推理过程中的3D视觉令牌。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：1)首次引入'3D思维'框架，无需密集标注数据；2)提出两阶段训练框架，从特征对齐到基于结果信号学习几何感知；3)增强模型可解释性，能从潜在空间恢复3D表示；4)在多个基准测试中表现优异。相比之前工作，3DThinker不依赖纯文本或2D视觉线索，不需要辅助模态或外部工具，不依赖真实图像监督，且在推理时无需外部3D先验。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '3DThinker首次让视觉语言模型能够在推理过程中进行3D空间想象，通过两阶段训练框架实现了从有限视角图像理解3D几何关系的能力，无需依赖外部3D先验或密集标注数据。'}


### 论文摘要

Though recent advances in vision-language models (VLMs) have achieved remarkable progress across a wide range of multimodal tasks, understanding 3D spatial relationships from limited views remains a significant challenge. Previous reasoning methods typically rely on pure text (e.g., topological cognitive maps) or on 2D visual cues. However, their limited representational capacity hinders performance in specific tasks that require 3D spatial imagination. To address this limitation, we propose 3DThinker, a framework that can effectively exploits the rich geometric information embedded within images while reasoning, like humans do. Our framework is the first to enable 3D mentaling during reasoning without any 3D prior input, and it does not rely on explicitly labeled 3D data for training. Specifically, our training consists of two stages. First, we perform supervised training to align the 3D latent generated by VLM while reasoning with that of a 3D foundation model (e.g., VGGT). Then, we optimize the entire reasoning trajectory solely based on outcome signals, thereby refining the underlying 3D mentaling. Extensive experiments across multiple benchmarks show that 3DThinker consistently outperforms strong baselines and offers a new perspective toward unifying 3D representations into multimodal reasoning. Our code will be available at https://github.com/zhangquanchen/3DThinker.

---

## 36. A Compositional Paradigm for Foundation Models: Towards Smarter Robotic Agents

**论文链接:** [http://arxiv.org/abs/2510.18608v1](http://arxiv.org/abs/2510.18608v1)

**作者:** Luigi Quarantiello, Elia Piccoli, Jack Bell, Malio Li, Giacomo Carfì, Eric Nuertey Coleman, Gerlando Gramaglia, Lanpei Li, Mauro Madeddu, Irene Testa, Vincenzo Lomonaco

**发布时间:** 2025-10-21

### GPT解析

### 总结

本文提出应用持续学习和组合性原则，以促进开发更灵活、高效和智能的AI解决方案，解决基础模型在适应动态现实世界场景时需要重新训练整个模型的问题。

### 背景

基础模型在语言、视觉、机器人控制等多种任务中带来了前所未有的结果。这些模型能够处理大量数据，提取和开发丰富的表示，这些表示可以跨不同领域和模态使用。

### 目的

提出应用持续学习和组合性原则，以促进开发更灵活、高效和智能的AI解决方案。

### 方法

应用持续学习和组合性原则。

### 主要发现

摘要中未明确提及具体发现。

### 结论

通过应用持续学习和组合性原则，可以开发更灵活、高效和智能的AI解决方案，解决基础模型在适应动态场景时需要重新训练的问题。

### 翻译

基础模型的诞生在从语言到视觉，再到机器人控制的广泛任务中带来了前所未有的结果。这些模型能够处理大量数据，并提取和开发丰富的表示，这些表示可以跨不同领域和模态使用。然而，它们在适应动态、现实世界场景方面仍然存在问题，需要从头开始重新训练整个模型。在这项工作中，我们提出应用持续学习和组合性原则，以促进开发更灵活、高效和智能的AI解决方案。


### 论文摘要

The birth of Foundation Models brought unprecedented results in a wide range of tasks, from language to vision, to robotic control. These models are able to process huge quantities of data, and can extract and develop rich representations, which can be employed across different domains and modalities. However, they still have issues in adapting to dynamic, real-world scenarios without retraining the entire model from scratch. In this work, we propose the application of Continual Learning and Compositionality principles to foster the development of more flexible, efficient and smart AI solutions.

---

## 37. Decoding Dynamic Visual Experience from Calcium Imaging via Cell-Pattern-Aware SSL

**论文链接:** [http://arxiv.org/abs/2510.18516v1](http://arxiv.org/abs/2510.18516v1)

**作者:** Sangyoon Bae, Mehdi Azabou, Jiook Cha, Blake Richards

**发布时间:** 2025-10-21

### GPT解析

### 总结

POYO-SSL是一种创新的神经科学自监督学习方法，通过专注于可预测神经元并利用数据异质性，实现了比传统方法更好的性能和可扩展性。

### 背景

自监督学习在神经科学领域有很大潜力，因为缺乏大规模、标签一致的神经数据集。大多数神经数据集包含异构群体，混合了稳定、可预测的细胞和高度随机、刺激依赖的细胞，这使得在自监督学习中识别一致的活动模式变得困难。

### 目的

提出一种新的自监督预训练方法，利用神经数据的异质性来改进预训练并实现规模效益。

### 方法

POYO-SSL仅在可预测的(统计规律性的)神经元上进行预训练，这些神经元通过简单的更高阶统计量(偏度和峰度)在预训练分割中识别，然后在不可预测的群体上进行微调，用于下游任务。

### 主要发现

在Allen Brain Observatory数据集上，POYO-SSL比从头开始训练获得了约12-13%的相对提升，显示出与模型大小相关的平滑、单调的扩展性。相比之下，现有最先进的基线在模型大小增加时趋于平稳或不稳定。

### 结论

通过将可预测性作为构建数据饮食的明确指标，POYO-SSL将异质性从负债转变为资产，为可扩展的神经解码提供了一种稳健的、基于生物学的方法，为神经动力学的基础模型铺平了道路。

### 翻译

自监督学习在神经科学应用中具有巨大潜力，这归因于缺乏大规模、标签一致的神经数据集。然而，大多数神经数据集包含异构群体，混合了稳定、可预测的细胞和高度随机、刺激依赖的细胞，这使得在自监督学习中识别一致的活动模式变得困难。因此，自监督预训练尚未在神经数据上显示出明显的规模效益。在这里，我们提出了一种自监督预训练的新方法POYO-SSL，它利用神经数据的异质性来改进预训练并实现规模效益。具体而言，在POYO-SSL中，我们仅在可预测的(统计规律性)神经元上进行预训练——这些神经元通过简单的更高阶统计量(偏度和峰度)在预训练分割中识别，然后在不可预测的群体上进行微调，用于下游任务。在Allen Brain Observatory数据集上，这种策略比从头开始训练获得了约12-13%的相对提升，并显示出与模型大小相关的平滑、单调的扩展性。相比之下，现有最先进的基线在模型大小增加时趋于平稳或不稳定。通过将可预测性作为构建数据饮食的明确指标，POYO-SSL将异质性从负债转变为资产，为可扩展的神经解码提供了一种稳健的、基于生物学的方法，并为神经动力学的基础模型铺平了道路。


### 论文摘要

Self-supervised learning (SSL) holds a great deal of promise for applications in neuroscience, due to the lack of large-scale, consistently labeled neural datasets. However, most neural datasets contain heterogeneous populations that mix stable, predictable cells with highly stochastic, stimulus-contingent ones, which has made it hard to identify consistent activity patterns during SSL. As a result, self-supervised pretraining has yet to show clear signs of benefits from scale on neural data. Here, we present a novel approach to self-supervised pretraining, POYO-SSL that exploits the heterogeneity of neural data to improve pre-training and achieve benefits of scale. Specifically, in POYO-SSL we pretrain only on predictable (statistically regular) neurons-identified on the pretraining split via simple higher-order statistics (skewness and kurtosis)-then we fine-tune on the unpredictable population for downstream tasks. On the Allen Brain Observatory dataset, this strategy yields approximately 12-13% relative gains over from-scratch training and exhibits smooth, monotonic scaling with model size. In contrast, existing state-of-the-art baselines plateau or destabilize as model size increases. By making predictability an explicit metric for crafting the data diet, POYO-SSL turns heterogeneity from a liability into an asset, providing a robust, biologically grounded recipe for scalable neural decoding and a path toward foundation models of neural dynamics.

---

## 38. Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models

**论文链接:** [http://arxiv.org/abs/2510.18457v1](http://arxiv.org/abs/2510.18457v1)

**作者:** Tianci Bi, Xiaoyi Zhang, Yan Lu, Nanning Zheng

**发布时间:** 2025-10-21

**备注:** Code and models available at: https://github.com/tianciB/VFM-VAE

### GPT解析

### 总结

本文提出了一种视觉基础模型变分自编码器(VFM-VAE)方法，解决了潜在扩散模型中视觉tokenizer与视觉基础模型对齐的鲁棒性问题，通过多尺度潜在融合和渐进分辨率重建技术，实现了高质量图像重建，并在较少训练epochs内取得了优异性能。

### 背景

潜在扩散模型(LDMs)的性能严重依赖于其视觉tokenizer的质量。近期研究尝试通过蒸馏方法整合视觉基础模型(VFMs)，但这种方法会削弱与原始VFM的鲁棒性对齐，导致潜在表示在分布偏移下语义偏离。

### 目的

开发一种直接整合视觉基础模型到潜在扩散模型中的方法，避免蒸馏方法带来的对齐鲁棒性问题，同时实现高质量图像重建和高效训练。

### 方法

提出视觉基础模型变分自编码器(VFM-VAE)，重新设计解码器结构，采用多尺度潜在融合和渐进分辨率重建模块，从空间上粗糙的VFM特征实现高质量重建。同时提供扩散训练期间表示动力学的综合分析，引入SE-CKNNA指标作为诊断工具，并开发联合tokenizer-扩散对齐策略。

### 主要发现

通过重新设计VFM-VAE解码器结构和联合对齐策略，系统在仅80个epoch内达到2.20的gFID(无CFG)，比之前的tokenizer快10倍；继续训练至640个epoch后，进一步达到1.62的gFID(无CFG)。直接VFM整合被证明是LDMs的优越范式。

### 结论

直接整合视觉基础模型到潜在扩散模型中的方法比传统的蒸馏方法更有效，能够保持与原始VFM的鲁棒性对齐，实现高质量图像重建，并显著加速训练过程。

### 翻译

潜在扩散模型(LDMs)的性能严重依赖于其视觉tokenizer的质量。虽然近期工作已经探索通过蒸馏整合视觉基础模型(VFMs)，但我们发现这种方法存在一个根本缺陷：它不可避免地会削弱与原始VFM的对齐鲁棒性，导致对齐的潜在表示在分布偏移下发生语义偏离。在本文中，我们通过提出一种更直接的方法来绕过蒸馏：视觉基础模型变分自编码器(VFM-VAE)。为了解决VFM的语义焦点与像素级保真度需求之间的内在张力，我们使用多尺度潜在融合和渐进分辨率重建块重新设计了VFM-VAE解码器，使得从空间上粗糙的VFM特征中实现高质量重建成为可能。此外，我们提供了扩散训练期间表示动力学的全面分析，引入了所提出的SE-CKNNA指标作为这一诊断的更精确工具。这一分析使我们能够开发联合tokenizer-扩散对齐策略，显著加速了收敛。我们在tokenizer设计和训练策略方面的创新带来了卓越的性能和效率：我们的系统在仅80个epoch内达到2.20的gFID(无CFG)(比之前的tokenizer快10倍)。继续训练至640个epoch后，它进一步获得了1.62的gFID(无CFG)，确立了直接VFM整合作为LDMs的优越范式。


### 论文摘要

The performance of Latent Diffusion Models (LDMs) is critically dependent on the quality of their visual tokenizer. While recent works have explored incorporating Vision Foundation Models (VFMs) via distillation, we identify a fundamental flaw in this approach: it inevitably weakens the robustness of alignment with the original VFM, causing the aligned latents to deviate semantically under distribution shifts. In this paper, we bypass distillation by proposing a more direct approach: Vision Foundation Model Variational Autoencoder (VFM-VAE). To resolve the inherent tension between the VFM's semantic focus and the need for pixel-level fidelity, we redesign the VFM-VAE decoder with Multi-Scale Latent Fusion and Progressive Resolution Reconstruction blocks, enabling high-quality reconstruction from spatially coarse VFM features. Furthermore, we provide a comprehensive analysis of representation dynamics during diffusion training, introducing the proposed SE-CKNNA metric as a more precise tool for this diagnosis. This analysis allows us to develop a joint tokenizer-diffusion alignment strategy that dramatically accelerates convergence. Our innovations in tokenizer design and training strategy lead to superior performance and efficiency: our system reaches a gFID (w/o CFG) of 2.20 in merely 80 epochs (a 10x speedup over prior tokenizers). With continued training to 640 epochs, it further attains a gFID (w/o CFG) of 1.62, establishing direct VFM integration as a superior paradigm for LDMs.

---

## 39. Automated urban waterlogging assessment and early warning through a mixture of foundation models

**论文链接:** [http://arxiv.org/abs/2510.18425v1](http://arxiv.org/abs/2510.18425v1)

**作者:** Chenxu Zhang, Fuxiang Huang, Lei Zhang

**发布时间:** 2025-10-21

**备注:** Submitted to Nature

### GPT解析

### 总结

本研究提出了UrbanWaterlogging Assessment (UWAssess)框架，利用基础模型自动识别监控图像中的内涝区域并生成结构化评估报告，通过半监督微调和思维链提示策略解决数据稀缺问题，显著提高了感知性能，并能够生成可靠的文本报告，支持城市管理和灾害应对。

### 背景

气候变化加剧，城市内涝对全球公共安全和基础设施构成越来越严重的威胁。

### 目的

开发一种能够自动识别内涝区域并生成结构化评估报告的框架，替代依赖人工报告的传统监测方法。

### 方法

设计UrbanWaterlogging Assessment (UWAssess)框架，采用半监督微调策略和思维链(CoT)提示策略，以解决标记数据稀缺问题，释放基础模型在数据稀缺下游任务中的潜力。

### 主要发现

在具有挑战性的视觉基准测试中评估显示感知性能有显著提高；基于GPT的评估确认了UWAssess能够生成可靠的文本报告，准确描述内涝范围、深度、风险和影响。

### 结论

UWAssess的双重能力使内涝监测从感知转变为生成，多个基础模型的协作框架为智能和可扩展系统奠定了基础，支持城市管理、灾害应对和气候韧性。

### 翻译

随着气候变化加剧，城市内涝对全球公共安全和基础设施构成越来越严重的威胁。然而，现有的监测方法主要依赖人工报告，无法提供及时和全面的评估。在本研究中，我们提出了UrbanWaterlogging Assessment (UWAssess)，这是一个基础模型驱动的框架，可以自动识别监控图像中的内涝区域并生成结构化评估报告。为解决标记数据稀缺的问题，我们设计了一种半监督微调策略和思维链(CoT)提示策略，以释放基础模型在数据稀缺下游任务中的潜力。在具有挑战性的视觉基准测试中的评估表明感知性能有显著提高。基于GPT的评估确认了UWAssess能够生成可靠的文本报告，准确描述内涝范围、深度、风险和影响。这种双重能力使内涝监测从感知转变为生成，而多个基础模型的协作框架为智能和可扩展系统奠定了基础，支持城市管理、灾害应对和气候韧性。


### 论文摘要

With climate change intensifying, urban waterlogging poses an increasingly severe threat to global public safety and infrastructure. However, existing monitoring approaches rely heavily on manual reporting and fail to provide timely and comprehensive assessments. In this study, we present Urban Waterlogging Assessment (UWAssess), a foundation model-driven framework that automatically identifies waterlogged areas in surveillance images and generates structured assessment reports. To address the scarcity of labeled data, we design a semi-supervised fine-tuning strategy and a chain-of-thought (CoT) prompting strategy to unleash the potential of the foundation model for data-scarce downstream tasks. Evaluations on challenging visual benchmarks demonstrate substantial improvements in perception performance. GPT-based evaluations confirm the ability of UWAssess to generate reliable textual reports that accurately describe waterlogging extent, depth, risk and impact. This dual capability enables a shift of waterlogging monitoring from perception to generation, while the collaborative framework of multiple foundation models lays the groundwork for intelligent and scalable systems, supporting urban management, disaster response and climate resilience.

---

## 40. Earth AI: Unlocking Geospatial Insights with Foundation Models and Cross-Modal Reasoning

**论文链接:** [http://arxiv.org/abs/2510.18318v1](http://arxiv.org/abs/2510.18318v1)

**作者:** Aaron Bell, Amit Aides, Amr Helmy, Arbaaz Muslim, Aviad Barzilai, Aviv Slobodkin, Bolous Jaber, David Schottlander, George Leifman, Joydeep Paul, Mimi Sun, Nadav Sherman, Natalie Williams, Per Bjornsson, Roy Lee, Ruth Alcantara, Thomas Turnbull, Tomer Shekel, Vered Silverman, Yotam Gigi, Adam Boulanger, Alex Ottenwess, Ali Ahmadalipour, Anna Carter, Charles Elliott, David Andre, Elad Aharoni, Gia Jung, Hassler Thurston, Jacob Bien, Jamie McPike, Juliet Rothenberg, Kartik Hegde, Kel Markert, Kim Philipp Jablonski, Luc Houriez, Monica Bharel, Phing VanLee, Reuven Sayag, Sebastian Pilarski, Shelley Cazares, Shlomi Pasternak, Siduo Jiang, Stone Jiang, Thomas Colthurst, Yang Chen, Yehonathan Refael, Yochai Blau, Yuval Carny, Yael Maguire, Avinatan Hassidim, James Manyika, Tim Thelin, Genady Beryozkin, Gautam Prasad, Luke Barrington, Yossi Matias, Niv Efron, Shravya Shetty

**发布时间:** 2025-10-21

### GPT解析

### 总结

Earth AI是一种创新的地理空间AI系统，结合了三个关键领域的基础模型和Gemini驱动的推理引擎，能够有效处理大量多样的地理空间数据，提供深入的见解和预测能力，特别是在危机情境中表现突出。

### 背景

地理空间数据为理解我们的星球提供了巨大潜力。然而，这些数据的巨大规模和多样性，以及不同的分辨率、时间尺度和稀疏性，给彻底分析和解释带来了显著挑战。

### 目的

介绍Earth AI，这是一种地理空间AI模型和智能代理推理系统，旨在显著提高我们从数据中解锁关于地球的新见解的能力。

### 方法

基于三个关键领域的基础模型——全球规模图像、人口和环境，以及一个由Gemini驱动的智能推理引擎。还开发了一个由Gemini驱动的代理，能够对多个基础模型以及大型地理空间数据源和工具进行联合推理，以处理复杂的多步骤查询。

### 主要发现

严格基准测试证明了基础模型的强大功能和新型能力；当这些模型一起使用时，它们为地理空间推理提供了互补价值，协同作用能够解锁卓越的预测能力；在真实世界危机场景基准测试中，代理展示了提供关键及时见解的能力，有效弥合了原始地理空间数据和可操作理解之间的差距。

### 结论

Earth AI通过结合基础模型和智能代理推理，能够克服地理空间数据分析中的挑战，提供更深入的见解，特别是在危机情境中，能够将原始数据转化为可操作的理解。

### 翻译

地理空间数据为理解我们的星球提供了巨大潜力。然而，这些数据的巨大规模和多样性，以及不同的分辨率、时间尺度和稀疏性，给彻底分析和解释带来了显著挑战。本文介绍了Earth AI，这是一种地理空间AI模型家族和智能代理推理系统，使我们在解锁关于地球的新颖而深刻见解的能力上取得了显著进步。该方法建立在三个关键领域的基础模型之上——全球规模图像、人口和环境，以及一个由Gemini驱动的智能推理引擎。我们展示了严格的基准测试，证明了我们基础模型的强大功能和新型能力，并验证了当一起使用时，它们为地理空间推理提供了互补价值，它们的协同作用能够解锁卓越的预测能力。为了处理复杂的多步骤查询，我们开发了一个由Gemini驱动的代理，能够对多个基础模型以及大型地理空间数据源和工具进行联合推理。在一个新的真实世界危机场景基准测试中，我们的代理展示了提供关键及时见解的能力，有效地弥合了原始地理空间数据和可操作理解之间的差距。


### 论文摘要

Geospatial data offers immense potential for understanding our planet. However, the sheer volume and diversity of this data along with its varied resolutions, timescales, and sparsity pose significant challenges for thorough analysis and interpretation. This paper introduces Earth AI, a family of geospatial AI models and agentic reasoning that enables significant advances in our ability to unlock novel and profound insights into our planet. This approach is built upon foundation models across three key domains--Planet-scale Imagery, Population, and Environment--and an intelligent Gemini-powered reasoning engine. We present rigorous benchmarks showcasing the power and novel capabilities of our foundation models and validate that when used together, they provide complementary value for geospatial inference and their synergies unlock superior predictive capabilities. To handle complex, multi-step queries, we developed a Gemini-powered agent that jointly reasons over our multiple foundation models along with large geospatial data sources and tools. On a new benchmark of real-world crisis scenarios, our agent demonstrates the ability to deliver critical and timely insights, effectively bridging the gap between raw geospatial data and actionable understanding.

---

## 41. From Agent Simulation to Social Simulator: A Comprehensive Review (Part 1)

**论文链接:** [http://arxiv.org/abs/2510.18271v1](http://arxiv.org/abs/2510.18271v1)

**作者:** Xiao Xue, Deyu Zhou, Ming Zhang, Fei-Yue Wang

**发布时间:** 2025-10-21

### GPT解析

### 总结

这篇论文是基于主体的建模(Agent-Based Modeling, ABM)综合评论的第一部分，重点介绍了ABM的历史发展和经典案例，包括基础模型和社会模拟案例的分类。

### 背景

传统物理模拟方法在社会领域面临重大挑战，这促使了ABM的发展。ABM有着自己的发展历史和设计原则。

### 目的

帮助读者理解传统物理模拟方法在社会领域面临的挑战，并介绍ABM的基础模型和经典案例。

### 方法

详细介绍了模拟社会系统的基础模型，包括个体模型、环境模型和基于规则的模型。

### 主要发现

社会模拟的经典案例可分为三类：思想实验、机制探索和平行优化。

### 结论

ABM作为一种模拟社会系统的方法，有着自己的历史发展、设计原则、基础模型和经典应用案例。

### 翻译

这是基于主体的建模(Agent-Based Modeling, ABM)综合评论的第一部分，重点介绍了ABM的历史发展和经典案例。它首先讨论了ABM的发展历史和设计原则，帮助读者理解传统物理模拟方法在社会领域面临的重大挑战。然后，它详细介绍了模拟社会系统的基础模型，包括个体模型、环境模型和基于规则的模型。最后，它介绍了社会模拟的经典案例，涵盖三种类型：思想实验、机制探索和平行优化。


### 论文摘要

This is the first part of the comprehensive review, focusing on the historical development of Agent-Based Modeling (ABM) and its classic cases. It begins by discussing the development history and design principles of Agent-Based Modeling (ABM), helping readers understand the significant challenges that traditional physical simulation methods face in the social domain. Then, it provides a detailed introduction to foundational models for simulating social systems, including individual models, environmental models, and rule-based models. Finally, it presents classic cases of social simulation, covering three types: thought experiments, mechanism exploration, and parallel optimization.

---

## 42. VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety

**论文链接:** [http://arxiv.org/abs/2510.18214v1](http://arxiv.org/abs/2510.18214v1)

**作者:** Shruti Palaskar, Leon Gatys, Mona Abdelrahman, Mar Jacobo, Larry Lindsey, Rutika Moharir, Gunnar Lund, Yang Xu, Navid Shiee, Jeffrey Bigham, Charles Maalouf, Joseph Yitan Cheng

**发布时间:** 2025-10-21

**备注:** 10 pages, 5 figures, 4 tables. Under review

### GPT解析

### 总结

本文提出Vision Language Safety Understanding (VLSU)框架，通过细化的严重性分类和组合分析系统评估多模态安全性，揭示了当前模型在联合图像-文本理解方面的弱点。

### 背景

当前多模态基础模型的安全评估通常将视觉和语言输入分开处理，忽略了良性内容组合后可能产生的风险；现有方法无法明确区分不安全内容和边缘情况，导致过度屏蔽或对真正有害内容拒绝不足。

### 目的

提出一个综合框架来系统评估多模态安全性，通过细化的严重性分类和组合分析识别模型在联合理解方面的缺陷。

### 方法

使用多阶段流程结合真实世界图像和人工注释，构建包含8,187个样本、涵盖15个危害类别的大规模基准数据集，评估11个最先进的模型。

### 主要发现

模型在清晰的单模态安全信号上准确率达90%以上，但在需要联合图像-文本推理时性能显著下降至20-55%；34%的联合分类错误发生在正确分类单个模态的情况下；模型难以平衡拒绝不安全内容与回应边缘情况；指令框架可减少过度屏蔽但代价是降低对不安全内容的拒绝率。

### 结论

该框架暴露了当前模型在联合图像-文本理解方面的弱点和对齐差距，为研究稳健的视觉-语言安全提供了关键的测试平台。

### 翻译

多模态基础模型的安全评估通常将视觉和语言输入分开处理，忽略了良性内容组合后可能产生的风险。现有方法也无法明确区分不安全内容和边缘情况，导致对真正有害内容过度屏蔽或拒绝不足。我们提出了Vision Language Safety Understanding (VLSU)，一个通过细化的严重性分类和17种不同安全模式的组合分析来系统评估多模态安全的综合框架。使用多阶段流程结合真实世界图像和人工注释，我们构建了一个包含8,187个样本、跨越15个危害类别的大规模基准。我们对11个最先进模型的评估揭示了系统性的联合理解失败：尽管模型在清晰的单模态安全信号上达到90%以上的准确率，但当需要联合图像-文本推理来确定安全标签时，性能显著下降到20-55%。最关键的是，34%的联合图像-文本安全分类错误发生在正确分类单个模态的情况下，进一步证明了缺乏组合推理能力。此外，我们发现模型难以平衡拒绝不安全内容同时回应值得关注的边缘情况。例如，我们发现指令框架可以将Gemini-1.5对边缘内容的过度屏蔽率从62.4%降低到10.4%，但代价是对不安全内容的拒绝率从90.8%下降到53.9%。总体而言，我们的框架暴露了当前模型在联合图像-文本理解方面的弱点和对齐差距，并为研究稳健的视觉-语言安全的下一个里程碑提供了关键的测试平台。


### 论文摘要

Safety evaluation of multimodal foundation models often treats vision and language inputs separately, missing risks from joint interpretation where benign content becomes harmful in combination. Existing approaches also fail to distinguish clearly unsafe content from borderline cases, leading to problematic over-blocking or under-refusal of genuinely harmful content. We present Vision Language Safety Understanding (VLSU), a comprehensive framework to systematically evaluate multimodal safety through fine-grained severity classification and combinatorial analysis across 17 distinct safety patterns. Using a multi-stage pipeline with real-world images and human annotation, we construct a large-scale benchmark of 8,187 samples spanning 15 harm categories. Our evaluation of eleven state-of-the-art models reveals systematic joint understanding failures: while models achieve 90%-plus accuracy on clear unimodal safety signals, performance degrades substantially to 20-55% when joint image-text reasoning is required to determine the safety label. Most critically, 34% of errors in joint image-text safety classification occur despite correct classification of the individual modalities, further demonstrating absent compositional reasoning capabilities. Additionally, we find that models struggle to balance refusing unsafe content while still responding to borderline cases that deserve engagement. For example, we find that instruction framing can reduce the over-blocking rate on borderline content from 62.4% to 10.4% in Gemini-1.5, but only at the cost of under-refusing on unsafe content with refusal rate dropping from 90.8% to 53.9%. Overall, our framework exposes weaknesses in joint image-text understanding and alignment gaps in current models, and provides a critical test bed to enable the next milestones in research on robust vision-language safety.

---

## 43. EMA-SAM: Exponential Moving-average for SAM-based PTMC Segmentation

**论文链接:** [http://arxiv.org/abs/2510.18213v1](http://arxiv.org/abs/2510.18213v1)

**作者:** Maryam Dialameh, Hossein Rajabzadeh, Jung Suk Sim, Hyock Ju Kwon

**发布时间:** 2025-10-21

### GPT解析

### 总结

本研究提出了一种名为EMA-SAM的新型模型，用于在甲状腺乳头状微小癌射频消融超声视频中实现更稳定的肿瘤分割。该模型通过在SAM-2基础上添加基于置信度的指数移动平均指针，解决了传统模型在介入超声应用中的不稳定预测和时间漂移问题。

### 背景

甲状腺乳头状微小癌(PTMC)越来越多地使用射频消融(RFA)进行治疗，但在超声视频中准确分割病灶面临挑战，主要由于低对比度、探头引起的运动和热相关伪影等问题。

### 目的

解决现有Segment Anything Model 2 (SAM-2)在介入超声视频中的不稳定预测和时间漂移问题，实现更稳定和准确的肿瘤分割。

### 方法

开发了EMA-SAM，这是SAM-2的轻量级扩展，在记忆库中融入了基于置信度的指数移动平均指针，提供跨帧的稳定肿瘤潜在原型，保持时间一致性的同时能快速适应新情况。

### 主要发现

在PTMC-RFA数据集上，EMA-SAM将maxDice从0.82提高到0.86，maxIoU从0.72提高到0.76，同时减少29%的假阳性；在外部基准测试中比SAM-2提高2-5个Dice点；计算开销增加不到0.1%，保持约30FPS的实时性能。

### 结论

EMA-SAM是一个稳健高效的框架，能够实现稳定的肿瘤跟踪，弥合了基础模型和介入超声严格需求之间的差距。

### 翻译

甲状腺乳头状微小癌(PTMC)越来越多地采用射频消融(RFA)治疗，然而由于低对比度、探头引起的运动和热相关伪影，在超声视频中准确分割病灶仍然困难。最近的Segment Anything Model 2 (SAM-2)在静态图像上表现良好，但其帧独立设计在介入超声中会导致不稳定预测和时间漂移。我们引入了EMA-SAM，这是SAM-2的一个轻量级扩展，在记忆库中融入了基于置信度的指数移动平均指针，提供跨帧的稳定肿瘤潜在原型。这种设计能够在探头压力和气泡遮挡时保持时间一致性，并在清晰证据重新出现时快速适应。在我们精心准备的PTMC-RFA数据集(124分钟，13名患者)上，EMA-SAM将maxDice从0.82(SAM-2)提高到0.86，maxIoU从0.72提高到0.76，同时减少29%的假阳性。在外部基准测试中，包括VTUS和结肠镜视频息肉数据集，EMA-SAM比SAM-2一致提高了2-5个Dice点。重要的是，EMA指针增加了不到0.1%的FLOPs，在单个A100 GPU上保持约30FPS的实时吞吐量。这些结果确立了EMA-SAM作为稳定肿瘤跟踪的稳健高效框架，弥合了基础模型和介入超声严格需求之间的差距。代码可在以下网址获取：https://github.com/mdialameh/EMA-SAM


### 论文摘要

Papillary thyroid microcarcinoma (PTMC) is increasingly managed with radio-frequency ablation (RFA), yet accurate lesion segmentation in ultrasound videos remains difficult due to low contrast, probe-induced motion, and heat-related artifacts. The recent Segment Anything Model 2 (SAM-2) generalizes well to static images, but its frame-independent design yields unstable predictions and temporal drift in interventional ultrasound. We introduce \textbf{EMA-SAM}, a lightweight extension of SAM-2 that incorporates a confidence-weighted exponential moving average pointer into the memory bank, providing a stable latent prototype of the tumour across frames. This design preserves temporal coherence through probe pressure and bubble occlusion while rapidly adapting once clear evidence reappears. On our curated PTMC-RFA dataset (124 minutes, 13 patients), EMA-SAM improves \emph{maxDice} from 0.82 (SAM-2) to 0.86 and \emph{maxIoU} from 0.72 to 0.76, while reducing false positives by 29\%. On external benchmarks, including VTUS and colonoscopy video polyp datasets, EMA-SAM achieves consistent gains of 2--5 Dice points over SAM-2. Importantly, the EMA pointer adds \textless0.1\% FLOPs, preserving real-time throughput of $\sim$30\,FPS on a single A100 GPU. These results establish EMA-SAM as a robust and efficient framework for stable tumour tracking, bridging the gap between foundation models and the stringent demands of interventional ultrasound. Codes are available here \hyperref[code {https://github.com/mdialameh/EMA-SAM}.

---

## 44. MACE Foundation Models for Lattice Dynamics: A Benchmark Study on Double Halide Perovskites

**论文链接:** [http://arxiv.org/abs/2510.18178v1](http://arxiv.org/abs/2510.18178v1)

**作者:** Jack Yang, Ziqi Yin, Lei Ao, Sean Li

**发布时间:** 2025-10-21

**备注:** 21 pages, 17 figures

### GPT解析

### 总结

材料信息学和人工智能的发展催生了MACE基础模型，为无机固体带来通用势能突破。研究使用DFT数据库对四种MACE变体进行基准测试，发现模型准确性随训练数据增加而提高，能更好地预测弱非谐性材料的动态稳定性，主要误差来源是原子力预测中的误差放大。

### 背景

材料信息学和人工智能的最新发展催生了材料化学的基础能量模型，以MACE系列基础模型为代表，为无机固体带来了通用势能的重大突破。

### 目的

对计算材料科学中的方法开发进行性能基准测试，理解模型局限性，促进模型改进，推动材料理论发展。

### 方法

使用作者发表的DFT数据库，包含约2000种立方卤化物双钙钛矿的室温动态稳定性和振动非谐性，对四种MACE基础模型变体进行基准测试。

### 主要发现

模型准确性随训练数据增加而提高；基础模型能更准确地再现弱非谐性材料的动态稳定性；预测动态稳定性的主要误差来自原子力预测误差的放大，而非构型空间采样的差异。

### 结论

希望研究结果能激励未来工作朝着更多物理启发的方向发展，以评估基础模型在原子建模中的准确性。

### 翻译

材料信息学和人工智能的最新发展催生了材料化学的基础能量模型，如MACE基础模型系列，为无机固体带来了通用势能的重大突破。对于计算材料科学中的所有方法开发，都需要针对特定应用与现有高级数据进行性能基准测试，以理解模型的局限性，从而促进模型开发过程的持续改进，有时会导致材料理论的重大概念飞跃。在此，我们使用自己发表的DFT（密度泛函理论）数据库，包含约2000种立方卤化物双钙钛矿的室温动态稳定性和振动非谐性，对四种不同变体的MACE基础模型进行了基准测试，评估其筛选无机固体动态稳定性的性能。我们的分析表明，正如预期，模型准确性随着更多训练数据的加入而提高。基础模型能更准确地再现弱非谐性材料的动态稳定性（由DFT预测），而非高度非谐性和动力学不稳定的材料。预测动态稳定性的主要误差来源在于预测谐波声子特性时通过计算Hessian矩阵放大原子力误差，而非DFT和基础模型在分子动力学中采样构型空间的差异。我们希望当前的研究发现将激励未来工作朝着更多物理启发的方向发展，以评估基础模型在原子建模中的准确性。


### 论文摘要

Recent developments in materials informatics and artificial intelligence has led to the emergence of foundational energy models for material chemistry, as represented by the suite of MACE-based foundation models, bringing a significant breakthrough in universal potentials for inorganic solids. As to all method developments in computational materials science, performance benchmarking against existing high-level data with focusing on specific applications, is critically needed to understand the limitations in the models, thus facilitating the ongoing improvements in the model development process, and occasionally, leading to significant conceptual leaps in materials theory. Here, using our own published DFT (Density Functional Theory) database of room-temperature dynamic stability and vibrational anharmonicity for $\sim2000$ cubic halide double perovskites, we benchmarked the performances of four different variants of the MACE foundation models for screening the dynamic stabilities of inorganic solids. Our analysis shows that, as anticipated, the model accuracy improves with more training data. The dynamic stabilities of weakly anharmonic materials (as predicted by DFT) are more accurately reproduced by the foundation model, than those highly anharmonic and dynamically unstable ones. The predominant source of error in predicting the dynamic stability arises predominantly from the amplification of errors in atomic forces when predicting the harmonic phonon properties through the computation of the Hessian matrix, less so is the contribution from possible differences in the range of the configurational spaces that are sampled by DFT and the foundation model in molecular dynamics. We hope that our present findings will stimulate future works towards more physics-inspired approaches in assessing the accuracy of foundation models for atomistic modelling.

---

## 45. MEG-GPT: A transformer-based foundation model for magnetoencephalography data

**论文链接:** [http://arxiv.org/abs/2510.18080v1](http://arxiv.org/abs/2510.18080v1)

**作者:** Rukuang Huang, Sungjun Cho, Chetan Gohil, Oiwi Parker Jones, Mark Woolrich

**发布时间:** 2025-10-20

### GPT解析

### 总结

该研究提出了MEG-GPT，一种基于transformer的基础模型，用于处理脑磁图数据。通过引入新颖的数据驱动标记化器和在大规模数据集上训练，该模型能生成具有真实脑电特性的数据，并在下游解码任务中表现出色，特别是在跨会话和跨受试者场景中。

### 背景

神经科学领域需要建模大规模脑动力学的复杂时空模式，但传统方法无法捕捉脑磁图等模态中的丰富结构。同时，深度学习通过大规模基础模型在语言和视觉等领域已取得显著进展。

### 目的

开发一个基于transformer的基础模型MEG-GPT，用于处理脑磁图数据，解决传统方法在处理脑电生理数据方面的局限性，并探索其在计算神经科学和神经解码中的应用潜力。

### 方法

提出MEG-GPT模型，使用时间注意力和下一时间点预测；引入数据驱动的标记化器处理连续MEG数据，保持高时间分辨率；在大规模MEG数据集上训练模型，包含612名闭眼休息状态的受试者数据；使用标记化的大脑区域时间序列进行训练。

### 主要发现

模型能生成具有真实时空频谱特性的数据，包括瞬态事件和群体变异性；在下游解码任务中表现良好，改善了监督预测任务；在跨会话准确率从0.54提高到0.59，跨受试者准确率从0.41提高到0.49；模型可在小标记数据集上高效微调，提升跨受试者解码性能。

### 结论

该研究为电生理数据建立了一个强大的基础模型，为计算神经科学和神经解码应用铺平了道路，展示了基础模型在神经科学领域的应用潜力。

### 翻译

建模大规模脑动力学的复杂时空模式对神经科学至关重要，但传统方法无法捕捉脑磁图等模态中的丰富结构。深度学习的最新进展通过大规模基础模型在语言和视觉等领域实现了显著进步。在此，我们介绍了MEG-GPT，一个基于transformer的基础模型，使用时间注意力和下一时间点预测。为此，我们还引入了一种新颖的数据驱动标记化器用于连续MEG数据，它保留了连续MEG信号的高时间分辨率而无需有损变换。我们在从大规模MEG数据集提取的标记化大脑区域时间序列上训练MEG-GPT，并表明学习到的模型能够生成具有真实时空频谱特性的数据，包括瞬态事件和群体变异性。关键的是，它在下游解码任务中表现良好，改善了下游监督预测任务，在跨会话和跨受试者方面显示出改进的零样本泛化能力。此外，我们表明模型可以在较小的标记数据集上高效微调，以提高跨受试者解码场景中的性能。这项工作为电生理数据建立了一个强大的基础模型，为计算神经科学和神经解码应用铺平了道路。


### 论文摘要

Modelling the complex spatiotemporal patterns of large-scale brain dynamics is crucial for neuroscience, but traditional methods fail to capture the rich structure in modalities such as magnetoencephalography (MEG). Recent advances in deep learning have enabled significant progress in other domains, such as language and vision, by using foundation models at scale. Here, we introduce MEG-GPT, a transformer based foundation model that uses time-attention and next time-point prediction. To facilitate this, we also introduce a novel data-driven tokeniser for continuous MEG data, which preserves the high temporal resolution of continuous MEG signals without lossy transformations. We trained MEG-GPT on tokenised brain region time-courses extracted from a large-scale MEG dataset (N=612, eyes-closed rest, Cam-CAN data), and show that the learnt model can generate data with realistic spatio-spectral properties, including transient events and population variability. Critically, it performs well in downstream decoding tasks, improving downstream supervised prediction task, showing improved zero-shot generalisation across sessions (improving accuracy from 0.54 to 0.59) and subjects (improving accuracy from 0.41 to 0.49) compared to a baseline methods. Furthermore, we show the model can be efficiently fine-tuned on a smaller labelled dataset to boost performance in cross-subject decoding scenarios. This work establishes a powerful foundation model for electrophysiological data, paving the way for applications in computational neuroscience and neural decoding.

---

## 46. Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity

**论文链接:** [http://arxiv.org/abs/2510.18037v1](http://arxiv.org/abs/2510.18037v1)

**作者:** Ziyu Lu, Anna J. Li, Alexander E. Ladd, Pascha Matveev, Aditya Deole, Eric Shea-Brown, J. Nathan Kutz, Nicholas A. Steinmetz

**发布时间:** 2025-10-20

**备注:** Accepted at the 39th Conference on Neural Information Processing  Systems (NeurIPS 2025) Workshop: Data on the Brain & Mind

### GPT解析

### 总结

该研究系统评估了深度学习模型在神经活动预测中的性能，发现深度学习模型优于传统方法，最佳模型可预测未来1.5秒的神经活动，为神经系统的理解和控制提供了新途径。

### 背景

神经活动预测对于理解神经系统和实现闭环控制至关重要。虽然深度学习最近在时间序列预测领域取得了最先进进展，但其在神经活动预测中的应用仍然有限。

### 目的

弥合深度学习在神经活动预测中应用的差距，系统评估多种深度学习模型在神经活动预测中的性能。

### 方法

系统评估了8种概率深度学习模型（包括2种基础模型），将其与4种经典统计模型和2种基线方法进行比较，使用宽场成像技术记录的小鼠皮层自发性神经活动作为数据，在不同的预测时间范围内进行评估。

### 主要发现

在各种预测时间范围内，几种深度学习模型持续优于经典方法，最佳模型能够提供未来1.5秒内有信息量的预测。

### 结论

研究结果指向未来的控制应用，为探索神经活动的内在时间结构开辟了新途径。

### 翻译

神经活动预测对于理解神经系统和实现闭环控制至关重要。虽然深度学习最近在时间序列预测文献中取得了最先进进展，但其在神经活动预测中的应用仍然有限。为了弥合这一差距，我们系统评估了八种概率深度学习模型（包括两种基础模型），这些模型在通用预测基准测试中表现出色。我们在通过宽场成像记录的小鼠皮层自发性神经活动上，将这些模型与四种经典统计模型和两种基线方法进行了比较。在各种预测时间范围内，几种深度学习模型持续优于经典方法，其中最佳模型能够提供未来1.5秒内有信息量的预测。我们的研究结果指向未来的控制应用，并为探索神经活动的内在时间结构开辟了新途径。


### 论文摘要

Neural activity forecasting is central to understanding neural systems and enabling closed-loop control. While deep learning has recently advanced the state-of-the-art in the time series forecasting literature, its application to neural activity forecasting remains limited. To bridge this gap, we systematically evaluated eight probabilistic deep learning models, including two foundation models, that have demonstrated strong performance on general forecasting benchmarks. We compared them against four classical statistical models and two baseline methods on spontaneous neural activity recorded from mouse cortex via widefield imaging. Across prediction horizons, several deep learning models consistently outperformed classical approaches, with the best model producing informative forecasts up to 1.5 seconds into the future. Our findings point toward future control applications and open new avenues for probing the intrinsic temporal structure of neural activity.

---

## 47. AION-1: Omnimodal Foundation Model for Astronomical Sciences

**论文链接:** [http://arxiv.org/abs/2510.17960v1](http://arxiv.org/abs/2510.17960v1)

**作者:** Liam Parker, Francois Lanusse, Jeff Shen, Ollie Liu, Tom Hehir, Leopoldo Sarra, Lucas Meyer, Micah Bowles, Sebastian Wagner-Carena, Helen Qu, Siavash Golkar, Alberto Bietti, Hatim Bourfoune, Nathan Casserau, Pierre Cornette, Keiya Hirashima, Geraud Krawezik, Ruben Ohana, Nicholas Lourie, Michael McCabe, Rudy Morel, Payel Mukhopadhyay, Mariel Pettee, Bruno Regaldo-Saint Blancard, Kyunghyun Cho, Miles Cranmer, Shirley Ho

**发布时间:** 2025-10-20

**备注:** Accepted at Neural Information Processing Systems (2025)

### GPT解析

### 总结

本文介绍了AION-1，一个天文领域的大规模多模态基础模型家族，能够整合异构的成像、光谱和标量数据，并在多种天文任务上表现出色。

### 背景

基础模型已在多个领域展现出潜力，但天文学仍缺乏一个统一框架来对其多样化的数据模态进行联合建模。

### 目的

开发一个天文领域的大规模多模态基础模型，能够处理天文学中各种不同的数据类型。

### 方法

采用两阶段架构：首先进行模态特定的标记化，然后使用基于transformer的跨模态标记序列掩码建模。模型在五个大规模调查数据集上进行预训练，包括Legacy Survey、HSC、SDSS、DESI和Gaia，覆盖超过2亿个天体观测。

### 主要发现

使用单个冻结编码器，AION-1在多种下游任务上取得优异表现，包括星系和恒星属性估计、星系形态分类、基于相似性的检索、星系图像分割和光谱超分辨率。

### 结论

AION-1提供了构建多模态科学基础模型的可扩展蓝图，能够无缝集成嘈杂的、仪器特定的观测数据。研究团队发布了参数量从3亿到31亿不等的模型变体，并开源了所有代码、标记器、预训练权重和评估套件。

### 翻译

尽管基础模型已在各种领域展现出前景，天文学仍然缺乏一个统一的框架来对其高度多样化的数据模态进行联合建模。在本文中，我们提出了AION-1，一个用于天文的大规模多模态基础模型家族。AION-1使用两阶段架构整合异构的成像、光谱和标量数据：模态特定的标记化，随后是基于transformer的跨模态标记序列掩码建模。该模型在五个大规模调查数据集上进行预训练：Legacy Survey、Hyper Suprime-Cam (HSC)、Sloan Digital Sky Survey (SDSS)、Dark Energy Spectroscopic Instrument (DESI)和Gaia。这些数据集涵盖了超过2亿颗恒星、星系和类星体的观测。使用单个冻结编码器，AION-1在广泛的下游任务上取得了强劲结果，包括星系和恒星属性估计、星系形态分类、基于相似性的检索、星系图像分割和光谱超分辨率。我们发布了参数量从3亿到31亿不等的AION-1模型变体。除天文学外，AION-1为多模态科学基础模型提供了可扩展的蓝图，能够无缝集成嘈杂的、仪器特定的观测。所有代码、标记器、预训练权重和轻量级评估套件均在开源许可下发布。


### 论文摘要

While foundation models have shown promise across a variety of fields, astronomy still lacks a unified framework for joint modeling across its highly diverse data modalities. In this paper, we present AION-1, a family of large-scale multimodal foundation models for astronomy. AION-1 integrates heterogeneous imaging, spectroscopic, and scalar data using a two-stage architecture: modality-specific tokenization followed by transformer-based masked modeling of cross-modal token sequences. The model is pretrained on five large-scale surveys: Legacy Survey, Hyper Suprime-Cam (HSC), Sloan Digital Sky Survey (SDSS), Dark Energy Spectroscopic Instrument (DESI), and Gaia. These span more than 200 million observations of stars, galaxies, and quasars. With a single frozen encoder, AION-1 achieves strong results on a broad suite of downstream tasks, including galaxy and stellar property estimation, galaxy morphology classification, similarity-based retrieval, galaxy image segmentation, and spectral super-resolution. We release AION-1 model variants ranging from 300 M to 3.1 B parameters. Beyond astronomy, AION-1 provides a scalable blueprint for multimodal scientific foundation models that can seamlessly integrate noisy, instrument-specific observations. All code, tokenizers, pretrained weights, and a lightweight evaluation suite are released under an open-source license.

---

## 48. Trust in foundation models and GenAI: A geographic perspective

**论文链接:** [http://arxiv.org/abs/2510.17942v1](http://arxiv.org/abs/2510.17942v1)

**作者:** Grant McKenzie, Krzysztof Janowicz, Carsten Kessler

**发布时间:** 2025-10-20

### GPT解析

### 总结

本文探讨了基础模型（特别是地理学领域）中的信任概念，将信任分为三类：对训练数据的认知信任、对模型功能的操作信任以及对模型开发者的人际信任。论文讨论了信任在地理应用中的含义、挑战、偏见问题、透明度和可解释性的重要性，以及地理信息科学家的独特视角。

### 背景

大型预训练机器学习模型已经改变了多个领域对人工智能的理解，包括地理学领域。随着这些模型被越来越多地依赖并用于关键决策，信任已成为讨论中的重要议题，但同时也变得复杂且多方面。

### 目的

论文旨在提供一个概念起点，帮助研究人员、从业者和政策制定者更好地理解（生成性）地理人工智能中的信任问题。

### 方法

作者将信任概念分为三个类型进行分析：认知信任、操作信任和人际信任，并探讨这些信任类型在地理应用中的独特含义。

### 主要发现

信任在基础模型中是一个多方面的概念；信任可分为三种类型：对训练数据的认知信任、对模型功能的操作信任以及对模型开发者的人际信任；文化背景、数据异质性和空间关系等主题对空间科学至关重要，并在发展信任中起重要作用；不同形式的偏见带来了挑战；透明度和可解释性很重要；模型开发中存在伦理责任；地理信息科学家提供了新的视角，呼吁进一步提高透明度、减少偏见并制定区域知情政策。

### 结论

随着对基础模型依赖的增加，信任已成为一个复杂但至关重要的概念。通过将信任分类并考虑地理学特有的因素，论文为理解和建立对地理人工智能的信任提供了概念框架。

### 翻译

大型预训练机器学习模型已经重塑了我们对多个领域人工智能的理解，包括我们自己的地理学领域。与任何新技术一样，信任在这一讨论中扮演着重要角色。在本章中，我们探讨了基础模型中信任的多方面概念，特别是在地理背景下。随着对这些模型的依赖增加并用于关键决策，信任虽然必不可少，但已成为一个分裂的概念。在这里，我们将信任分为三类：对训练数据的认知信任、对模型功能的操作信任以及对模型开发者的人际信任。每种信任类型都为地理应用带来了独特的含义。文化背景、数据异质性和空间关系等主题是空间科学的基础，并在发展信任中发挥重要作用。本章继续讨论了不同形式偏见带来的挑战、透明度和可解释性的重要性以及模型开发中的伦理责任。最后，强调了地理信息科学家的新颖视角，呼吁进一步提高透明度、减少偏见并制定区域知情政策。简而言之，本章旨在为研究人员、从业者和政策制定者提供一个概念起点，以更好地理解（生成性）地理人工智能中的信任。


### 论文摘要

Large-scale pre-trained machine learning models have reshaped our understanding of artificial intelligence across numerous domains, including our own field of geography. As with any new technology, trust has taken on an important role in this discussion. In this chapter, we examine the multifaceted concept of trust in foundation models, particularly within a geographic context. As reliance on these models increases and they become relied upon for critical decision-making, trust, while essential, has become a fractured concept. Here we categorize trust into three types: epistemic trust in the training data, operational trust in the model's functionality, and interpersonal trust in the model developers. Each type of trust brings with it unique implications for geographic applications. Topics such as cultural context, data heterogeneity, and spatial relationships are fundamental to the spatial sciences and play an important role in developing trust. The chapter continues with a discussion of the challenges posed by different forms of biases, the importance of transparency and explainability, and ethical responsibilities in model development. Finally, the novel perspective of geographic information scientists is emphasized with a call for further transparency, bias mitigation, and regionally-informed policies. Simply put, this chapter aims to provide a conceptual starting point for researchers, practitioners, and policy-makers to better understand trust in (generative) GeoAI.

---

## 49. Robustness Verification of Graph Neural Networks Via Lightweight Satisfiability Testing

**论文链接:** [http://arxiv.org/abs/2510.18591v1](http://arxiv.org/abs/2510.18591v1)

**作者:** Chia-Hsuan Lu, Tony Tan, Michael Benedikt

**发布时间:** 2025-10-21

### GPT解析

### 总结

图神经网络（GNNs）是学习图结构的主导架构，对抗攻击检测是一个重要问题。作者提出使用高效部分求解器替代传统强大求解器的方法，以提高结构鲁棒性，并在多种GNN变体和数据集上评估了其工具RobLight。

### 背景

图神经网络（GNNs）是学习图结构的主导架构。与任何机器学习模型一样，检测对抗性攻击（对手通过小幅扰动输入来改变输出）是一个重要问题。解决对抗鲁棒性问题（确定是否存在此类攻击）的技术最初是为图像分类开发的，但也有适用于其他机器学习架构的变体。

### 目的

提高图神经网络结构鲁棒性的最先进水平，通过替代传统方法中使用强大求解器的做法。

### 方法

用高效的部分求解器（运行时间为多项式时间但不一定完整）替换强大求解器的使用，开发工具RobLight，并在多种GNN变体和数据集上进行评估。

### 主要发现

可以通过使用高效的部分求解器替代强大求解器，来改进结构鲁棒性的最先进技术。

### 结论

作者的工具RobLight在多种GNN变体和数据集上进行了评估，表明该方法在对抗攻击检测方面具有潜力。

### 翻译

图神经网络（GNNs）是学习图结构的主导架构。与任何机器学习模型一样，一个重要问题是检测对抗性攻击，即对手可以通过对输入的小幅扰动来改变输出。解决对抗鲁棒性问题（确定是否存在此类攻击）的技术最初是为图像分类开发的，但也有适用于许多其他机器学习架构的变体。在图学习的情况下，攻击模型通常考虑对图结构的更改，而不仅仅是或代替输入的数值特征，该领域最先进的技术通过简化为约束求解来实现，基于强大的求解器（例如用于混合整数编程的求解器）。我们展示了可以通过用高效的部分求解器（运行时间为多项式时间但不一定完整）替换强大求解器的使用，来提高结构鲁棒性的最先进水平。我们在多种GNN变体和数据集上评估了我们的工具RobLight。


### 论文摘要

Graph neural networks (GNNs) are the predominant architecture for learning over graphs. As with any machine learning model, and important issue is the detection of adversarial attacks, where an adversary can change the output with a small perturbation of the input. Techniques for solving the adversarial robustness problem - determining whether such an attack exists - were originally developed for image classification, but there are variants for many other machine learning architectures. In the case of graph learning, the attack model usually considers changes to the graph structure in addition to or instead of the numerical features of the input, and the state of the art techniques in the area proceed via reduction to constraint solving, working on top of powerful solvers, e.g. for mixed integer programming. We show that it is possible to improve on the state of the art in structural robustness by replacing the use of powerful solvers by calls to efficient partial solvers, which run in polynomial time but may be incomplete. We evaluate our tool RobLight on a diverse set of GNN variants and datasets.

---

## 50. Benchmarking Fairness-aware Graph Neural Networks in Knowledge Graphs

**论文链接:** [http://arxiv.org/abs/2510.18473v1](http://arxiv.org/abs/2510.18473v1)

**作者:** Yuya Sasaki

**发布时间:** 2025-10-21

### GPT解析

### 总结

该研究引入了知识图谱上的公平感知图神经网络(GNNs)基准研究，从YAGO、DBpedia和Wikidata生成更大规模的数据集，评估不同GNN主干和早期停止条件下的预处理和内处理方法。

### 背景

图神经网络是学习图结构数据的强大工具，但通常对敏感属性产生有偏见的预测。尽管公平感知的GNNs已被研究用于减轻这种偏见，但之前的研究未在知识图谱这一重要应用领域进行评估。

### 目的

评估公平感知的GNNs在知识图谱上的表现，并建立相关基准。

### 方法

从三个知识图谱(YAGO、DBpedia和Wikidata)生成新的更大规模图数据集，在不同GNN主干和早期停止条件下对预处理和内处理方法进行基准测试。

### 主要发现

(i)知识图谱与现有数据集表现出不同趋势，在公平性与准确性间有更清晰的权衡；(ii)性能不仅受公平感知GNN方法影响，还受GNN主干和早期停止条件显著影响；(iii)预处理方法改善公平性指标，内处理方法提高预测准确性。

### 结论

知识图谱上的公平性研究需要综合考虑数据集特性、模型架构和训练方法的选择，这些因素共同影响公平性与准确性的权衡。

### 翻译

图神经网络(GNNs)是学习图结构数据的强大工具，但通常对敏感属性产生有偏见的预测。公平感知的GNNs已被积极研究用于减轻有偏见的预测。然而，之前的研究没有在知识图谱上评估公平感知的GNNs，而知识图谱是许多应用（如推荐系统）中最重要的图之一。因此，我们引入一个关于知识图谱的基准研究。我们从三个知识图谱（YAGO、DBpedia和Wikidata）生成新的图，这些图比公平性研究中使用的现有图数据集大得多。我们在不同的GNN主干和早期停止条件下对预处理和内处理方法进行基准测试。我们发现几个关键见解：(i)知识图谱显示出与现有数据集不同的趋势；在公平感知的GNNs中，与其他图相比，预测准确性和公平性指标之间有更清晰的权衡，(ii)性能不仅受到公平感知的GNN方法的影响，还受到GNN主干和早期停止条件的显著影响，以及(iii)预处理方法通常改善公平性指标，而内处理方法提高预测准确性。


### 论文摘要

Graph neural networks (GNNs) are powerful tools for learning from graph-structured data but often produce biased predictions with respect to sensitive attributes. Fairness-aware GNNs have been actively studied for mitigating biased predictions. However, no prior studies have evaluated fairness-aware GNNs on knowledge graphs, which are one of the most important graphs in many applications, such as recommender systems. Therefore, we introduce a benchmarking study on knowledge graphs. We generate new graphs from three knowledge graphs, YAGO, DBpedia, and Wikidata, that are significantly larger than the existing graph datasets used in fairness studies. We benchmark inprocessing and preprocessing methods in different GNN backbones and early stopping conditions. We find several key insights: (i) knowledge graphs show different trends from existing datasets; clearer trade-offs between prediction accuracy and fairness metrics than other graphs in fairness-aware GNNs, (ii) the performance is largely affected by not only fairness-aware GNN methods but also GNN backbones and early stopping conditions, and (iii) preprocessing methods often improve fairness metrics, while inprocessing methods improve prediction accuracy.

---

## 51. Training Diverse Graph Experts for Ensembles: A Systematic Empirical Study

**论文链接:** [http://arxiv.org/abs/2510.18370v1](http://arxiv.org/abs/2510.18370v1)

**作者:** Gangda Deng, Yuxin Yang, Ömer Faruk Akgül, Hanqing Zeng, Yinglong Xia, Rajgopal Kannan, Viktor Prasanna

**发布时间:** 2025-10-21

### GPT解析

### 总结

该研究首次对图神经网络集成的专家级多样化技术进行了系统性实证研究，评估了20种多样化策略在14个节点分类基准上的表现，构建并分析了200多个集成变体，为专家训练和图数据上有效混合专家框架的设计提供了指导。

### 背景

图神经网络已成为学习关系数据的重要工具，但单一图神经网络在处理现实世界图中存在的异质性时性能受限。近期混合专家框架的进展表明，组合多个具有明显不同泛化模式的多样化图神经网络可以显著提高性能。

### 目的

本研究旨在对图神经网络集成的专家级多样化技术进行首个系统性实证研究，评估不同多样化策略的效果，并提供训练最大化多样化专家的机制见解。

### 方法

研究评估了20种多样化策略，包括随机重新初始化、超参数调整、架构变化、方向性建模和训练数据分区等，在14个节点分类基准上构建并分析了200多个集成变体，从专家多样性、互补性和集成性能等方面全面评估了每种技术。

### 主要发现

研究揭示了训练最大化多样化专家的机制见解，发现不同多样化策略在产生专家多样性、互补性和提升集成性能方面有不同效果，为专家训练和有效混合专家框架设计提供了可操作的指导。

### 结论

该研究通过大规模实证分析，为图神经网络集成的专家级多样化技术提供了系统性理解和实用指导，有助于开发更高效的混合专家框架来处理图数据中的异质性。

### 翻译

图神经网络已成为学习关系数据的重要工具，然而单一图神经网络的性能往往受到现实世界图中存在的异质性的限制。混合专家框架的最新进展表明，组合多个具有明显不同泛化模式的多样化图神经网络可以显著提高性能。在这项工作中，我们首次对图神经网络集成的专家级多样化技术进行了系统性实证研究。通过在14个节点分类基准上评估20种多样化策略——包括随机重新初始化、超参数调整、架构变化、方向性建模和训练数据分区等，我们构建并分析了200多个集成变体。我们的全面评估从专家多样性、互补性和集成性能等方面检验了每种技术。我们还揭示了训练最大化多样化专家的机制见解。这些发现为图数据上的专家训练和有效混合专家框架的设计提供了可操作的指导。我们的代码可在指定链接获取。


### 论文摘要

Graph Neural Networks (GNNs) have become essential tools for learning on relational data, yet the performance of a single GNN is often limited by the heterogeneity present in real-world graphs. Recent advances in Mixture-of-Experts (MoE) frameworks demonstrate that assembling multiple, explicitly diverse GNNs with distinct generalization patterns can significantly improve performance. In this work, we present the first systematic empirical study of expert-level diversification techniques for GNN ensembles. Evaluating 20 diversification strategies -- including random re-initialization, hyperparameter tuning, architectural variation, directionality modeling, and training data partitioning -- across 14 node classification benchmarks, we construct and analyze over 200 ensemble variants. Our comprehensive evaluation examines each technique in terms of expert diversity, complementarity, and ensemble performance. We also uncovers mechanistic insights into training maximally diverse experts. These findings provide actionable guidance for expert training and the design of effective MoE frameworks on graph data. Our code is available at https://github.com/Hydrapse/bench-gnn-diversification.

---

## 52. Committors without Descriptors

**论文链接:** [http://arxiv.org/abs/2510.18018v1](http://arxiv.org/abs/2510.18018v1)

**作者:** Peilin Kang, Jintu Zhang, Enrico Trizio, TingJun Hou, Michele Parrinello

**发布时间:** 2025-10-20

### GPT解析

### 总结

本研究提出了一种结合图神经网络的改进版基于committor的方法，用于原子模拟中的稀有事件研究。

### 背景

稀有事件的研究是原子模拟中的主要挑战之一，已提出几种增强采样方法。最近建议使用committor来提供稀有事件的精确形式描述。

### 目的

提出一种基于committor的方法，促进系统亚稳态之间的频繁转换，并允许对过程过渡态集合进行广泛采样。

### 方法

利用变分标准迭代优化基于神经网络的committor参数化，使用一组物理描述符作为输入，方法具有自洽和半自动的优势。

### 主要发现

将之前的方法与图神经网络结合，可以直接处理原子坐标而不是描述符，进一步自动化该过程。

### 结论

结合图神经网络增强了方法的能力，特别是在处理原子坐标和描述溶剂分子的作用方面，如离子对解离或配体结合。

### 翻译

The study of rare events is one of the major challenges in atomistic simulations, and several enhanced sampling methods towards its solution have been proposed. Recently, it has been suggested that the use of the committor, which provides a precise formal description of rare events, could be of use in this context. We have recently followed up on this suggestion and proposed a committor-based method that promotes frequent transitions between the metastable states of the system and allows extensive sampling of the process transition state ensemble. One of the strengths of our approach is being self-consistent and semi-automatic, exploiting a variational criterion to iteratively optimize a neural-network-based parametrization of the committor, which uses a set of physical descriptors as input. Here, we further automate this procedure by combining our previous method with the expressive power of graph neural networks, which can directly process atomic coordinates rather than descriptors. Besides applications on benchmark systems, we highlight the advantages of a graph-based approach in describing the role of solvent molecules in systems, such as ion pair dissociation or ligand binding.


### 论文摘要

The study of rare events is one of the major challenges in atomistic simulations, and several enhanced sampling methods towards its solution have been proposed. Recently, it has been suggested that the use of the committor, which provides a precise formal description of rare events, could be of use in this context. We have recently followed up on this suggestion and proposed a committor-based method that promotes frequent transitions between the metastable states of the system and allows extensive sampling of the process transition state ensemble. One of the strengths of our approach is being self-consistent and semi-automatic, exploiting a variational criterion to iteratively optimize a neural-network-based parametrization of the committor, which uses a set of physical descriptors as input. Here, we further automate this procedure by combining our previous method with the expressive power of graph neural networks, which can directly process atomic coordinates rather than descriptors. Besides applications on benchmark systems, we highlight the advantages of a graph-based approach in describing the role of solvent molecules in systems, such as ion pair dissociation or ligand binding.

---

## 53. QINNs: Quantum-Informed Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.17984v1](http://arxiv.org/abs/2510.17984v1)

**作者:** Aritra Bal, Markus Klute, Benedikt Maier, Melik Oughton, Eric Pezone, Michael Spannowsky

**发布时间:** 2025-10-20

**备注:** 20 pages, 9 figures

### GPT解析

### 总结

本文提出量子信息神经网络(QINNs)框架，将量子信息概念引入经典模型，通过量子费舍尔信息矩阵(QFIM)作为粒子关联的紧凑表示，在喷注标记任务中提高模型性能，使粒子碰撞分析更加实用、可解释和可扩展。

### 背景

经典深度神经网络能够学习强子对撞机数据中的丰富多粒子关联，但其归纳偏差很少基于物理结构。

### 目的

开发一个通用框架，将量子信息概念和量子可观测量引入纯经典模型中，增强粒子碰撞分析的量子信息处理能力。

### 方法

研究QINNs的一个具体实现，将每个粒子编码为量子比特，使用量子费舍尔信息矩阵(QFIM)作为粒子关联的紧凑、基无关摘要，在图神经网络中将QFIM用作轻量级嵌入。

### 主要发现

QFIM能够区分QCD和强子顶喷注，显示出符合物理预期的不同模式，表明QINNs能够捕捉有意义的物理特征。

### 结论

QINNs为粒子碰撞的量子信息分析(断层扫描)提供了一种实用、可解释和可扩展的途径，特别是通过增强现有的深度学习方法。

### 翻译

经典深度神经网络可以学习强子对撞机数据中的丰富多粒子关联，但它们的归纳偏差很少锚定在物理结构上。我们提出了量子信息神经网络(QINNs)，这是一个通用框架，将量子信息概念和量子可观测量引入纯经典模型。虽然该框架很广泛，但在本文中，我们研究了一个具体实现，将每个粒子编码为量子比特，并使用量子费舍尔信息矩阵(QFIM)作为粒子关联的紧凑、基无关摘要。以喷注标记为案例研究，QFIM在图神经网络中充当轻量级嵌入，提高了模型的表型和可塑性。QFIM揭示了QCD和强子顶喷注的不同模式，这些模式符合物理预期。因此，QINNs为粒子碰撞的量子信息分析(即断层扫描)提供了一条实用、可解释和可扩展的途径，特别是通过增强成熟的深度学习方法。


### 论文摘要

Classical deep neural networks can learn rich multi-particle correlations in collider data, but their inductive biases are rarely anchored in physics structure. We propose quantum-informed neural networks (QINNs), a general framework that brings quantum information concepts and quantum observables into purely classical models. While the framework is broad, in this paper, we study one concrete realisation that encodes each particle as a qubit and uses the Quantum Fisher Information Matrix (QFIM) as a compact, basis-independent summary of particle correlations. Using jet tagging as a case study, QFIMs act as lightweight embeddings in graph neural networks, increasing model expressivity and plasticity. The QFIM reveals distinct patterns for QCD and hadronic top jets that align with physical expectations. Thus, QINNs offer a practical, interpretable, and scalable route to quantum-informed analyses, that is, tomography, of particle collisions, particularly by enhancing well-established deep learning approaches.

---

## 54. SemiAdapt and SemiLoRA: Efficient Domain Adaptation for Transformer-based Low-Resource Language Translation with a Case Study on Irish

**论文链接:** [http://arxiv.org/abs/2510.18725v1](http://arxiv.org/abs/2510.18725v1)

**作者:** Josh McGiff, Nikola S. Nikolov

**发布时间:** 2025-10-21

**备注:** 8 pages

### GPT解析

### 总结

本文介绍了SemiAdapt和SemiLoRA两种半监督推理高效方法，用于加强神经机器翻译中的领域适应，提高整体性能，特别是在低资源语言如爱尔兰语翻译方面。

### 背景

微调被广泛用于定制大语言模型执行特定任务，但大型多语言模型的微调计算成本高，为研究低资源领域的研究人员设置了障碍。

### 目的

解决低资源语言领域中高质量领域适应和微调的可访问性问题，使研究人员更容易进行这些工作。

### 方法

介绍SemiAdapt和SemiLoRA作为半监督推理高效方法；利用参数高效微调和低秩适应技术；评估按数据集进行领域微调；开发基于嵌入的推理方法。

### 主要发现

SemiAdapt可以优于全领域微调；SemiLoRA可以使参数高效微调方法匹配甚至超过全模型微调的性能；基于嵌入的推理方法在更大和更嘈杂的语料库上表现特别好。

### 结论

这些方法使高质量领域适应和微调更容易被低资源语言研究人员获取，所有爱尔兰语翻译模型都作为开放资源发布。

### 翻译

本研究特别关注爱尔兰语翻译，开发的爱尔兰语翻译模型已作为开放资源发布，旨在促进低资源语言的研究。


### 论文摘要

Fine-tuning is widely used to tailor large language models for specific tasks such as neural machine translation (NMT). However, leveraging transfer learning is computationally expensive when fine-tuning large multilingual models with billions of parameters, thus creating a barrier to entry for researchers working on low-resource domains such as Irish translation. Parameter-efficient fine-tuning (PEFT) bridges this gap by training on a fraction of the original model parameters, with the Low-Rank Adaptation (LoRA) approach introducing small, trainable adapter layers. We introduce SemiAdapt and SemiLoRA as semi-supervised inference-efficient approaches that strengthen domain adaptation and lead to improved overall performance in NMT. We demonstrate that SemiAdapt can outperform full-domain fine-tuning, while most notably, SemiLoRA can propel PEFT methods to match or even outperform full-model fine-tuning. We further evaluate domain-by-dataset fine-tuning and demonstrate that our embedding-based inference methods perform especially well on larger and noisier corpora. All Irish translation models developed in this work are released as open resources. These methods aim to make high-quality domain adaptation and fine-tuning more accessible to researchers working with low-resource languages.

---

## 55. Automated Wicket-Taking Delivery Segmentation and Weakness Detection in Cricket Videos Using OCR-Guided YOLOv8 and Trajectory Modeling

**论文链接:** [http://arxiv.org/abs/2510.18405v1](http://arxiv.org/abs/2510.18405v1)

**作者:** Mst Jannatun Ferdous, Masum Billah, Joy Karmoker, Mohd Ruhul Ameen, Akif Islam, Md. Omar Faruqe

**发布时间:** 2025-10-21

**备注:** 6 figures, 5 tables, submitted to the 11th IEEE International Women  in Engineering (WIE) Conference on Electrical and Computer Engineering 2025

### GPT解析

### 总结

这篇论文提出了一种用于板球视频分析的自动化系统，该系统利用深度学习技术来提取导致球门被取的投球、检测板球并建立球轨迹模型。系统使用YOLOv8架构进行场地和球检测，结合光学字符识别技术提取记分卡信息来识别球门被取的时刻。通过全面的图像预处理，系统实现了从视频帧中稳健的文本提取。场地检测模型达到高精确度，球检测模型使用迁移学习也表现出色。该系统可以在检测到的场地上进行轨迹建模，为识别击球弱点提供数据驱动的洞察。

### 背景

板球比赛产生大量视频数据，人工分析这些数据以提取战术信息是一个耗时且复杂的过程。随着深度学习技术的发展，自动分析板球视频以提取关键战术信息成为可能，这可以为教练团队和战略决策提供数据支持。

### 目的

开发一个自动化系统，用于分析板球视频，提取关键信息（如导致球门被取的投球、球的检测和轨迹建模），从而为教练团队提供数据驱动的战术洞察，帮助识别击球弱点和改进比赛策略。

### 方法

使用YOLOv8架构进行场地和球检测；结合光学字符识别技术提取记分卡信息；应用图像预处理技术，包括灰度转换、幂变换和形态学操作；使用迁移学习技术改进球检测模型；在检测到的场地上进行轨迹建模；在多个板球比赛视频上进行实验验证。

### 主要发现

场地检测模型达到99.5%的平均精度均值和0.999的精确度；球检测模型使用迁移学习达到99.18%的平均精度均值、0.968的精确度和0.978的召回率；系统能够有效识别导致球门被取的时刻；轨迹建模能够提供识别击球弱点的数据驱动洞察；该系统在多个板球比赛视频上表现出有效性。

### 结论

该自动化板球视频分析系统利用深度学习技术实现了高精度的场地和球检测，能够有效提取关键战术信息。该系统不仅能够识别导致球门被取的时刻，还能通过轨迹建模提供数据驱动的战术洞察，为教练团队和战略决策提供重要支持，具有广阔的应用前景。

### 翻译

本文提出了一种用于板球视频分析的自动化系统，该系统利用深度学习技术来提取导致球门被取的投球、检测板球并建立球轨迹模型。系统采用YOLOv8架构进行场地和球检测，结合光学字符识别技术提取记分卡信息以识别球门被取的时刻。通过全面的图像预处理，包括灰度转换、幂变换和形态学操作，系统实现了从视频帧中稳健的文本提取。场地检测模型达到99.5%的平均精度均值，精确度为0.999；而使用迁移学习的球检测模型达到99.18%的平均精度均值，精确度为0.968，召回率为0.978。该系统可在检测到的场地上进行轨迹建模，为识别击球弱点提供数据驱动的洞察。在多个板球比赛视频上的实验结果证明了这种方法在自动化板球分析中的有效性，为教练和战略决策提供了巨大潜力。


### 论文摘要

This paper presents an automated system for cricket video analysis that leverages deep learning techniques to extract wicket-taking deliveries, detect cricket balls, and model ball trajectories. The system employs the YOLOv8 architecture for pitch and ball detection, combined with optical character recognition (OCR) for scorecard extraction to identify wicket-taking moments. Through comprehensive image preprocessing, including grayscale transformation, power transformation, and morphological operations, the system achieves robust text extraction from video frames. The pitch detection model achieved 99.5% mean Average Precision at 50% IoU (mAP50) with a precision of 0.999, while the ball detection model using transfer learning attained 99.18% mAP50 with 0.968 precision and 0.978 recall. The system enables trajectory modeling on detected pitches, providing data-driven insights for identifying batting weaknesses. Experimental results on multiple cricket match videos demonstrate the effectiveness of this approach for automated cricket analytics, offering significant potential for coaching and strategic decision-making.

---

