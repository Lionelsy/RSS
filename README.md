# 今日论文推荐 - 2025-11-06

共 49 篇论文

---

## 1. Model order reduction via Lie groups

**论文链接:** [http://arxiv.org/abs/2511.03520v1](http://arxiv.org/abs/2511.03520v1)

**作者:** Yannik P. Wotte, Patrick Buchfink, Silke Glas, Federico Califano, Stefano Stramigioli

**发布时间:** 2025-11-05

**备注:** 22 pages, 21 figures

### GPT解析

### 总结

本文提出了一种名为MORLie的新型基于李群的模型降阶框架，该框架能够将流形上的高维动力系统近似为李群上的低维动力系统，特别适用于处理实际应用中常见的非等变动力学问题。

### 背景

李群及其作用在物理系统的描述中无处不在，模型降阶(MOR)是处理高维系统的重要方法。

### 目的

探索李群在模型降阶中的影响，并提出一种新的基于李群的模型降阶框架。

### 方法

提出MORLie框架，在流形上高维动力系统与李群上低维动力系统之间建立近似关系，提供基于几何公式的新非侵入式MOR方法，能够处理非等变动力学。

### 主要发现

MORLie的误差边界低于Kolmogorov N-宽度，限制了线性子空间方法；在三个应用案例中表现优异：1)变形体建模中优于POD方法；2)肝脏运动重建接近最先进水平且训练时间大幅减少；3)解析例证显示方法的通用性。

### 结论

MORLi是一种有效的模型降阶方法，能够处理非等变动力学，在多个实际应用中展现出色性能，包括变形体建模和医学图像处理等。

### 翻译

李群及其作用在物理系统的描述中无处不在，我们探索了在模型降阶(MOR)设置中的影响。我们提出了一个名为MORLie的基于李群的新型模型降阶框架，其中流形上的高维动力系统被李群上的低维动力系统近似。与其他李群方法相比，我们能够处理实际应用中常见的非等变动力学，并基于提出的几何公式提供了新的非侵入式MOR方法。我们还通过数值计算强调，MORLie的误差边界低于限制线性子空间方法的Kolmogorov N-宽度。该方法应用于各种示例：1. 对遵循剪切运动的噪声点云数据建模的变形体简化模型，其中MORLie在准确性和降维方面优于简单的POD方法；2. 通过超声扫描边缘检测数据重建呼吸期间的肝脏运动，MORLi的性能接近最先进水平，同时将训练时间从计算集群上的几小时减少到移动工作站上的几分钟；3. 一个解析例子，显示冻结方法作为特例被解析恢复，表明了几何框架的通用性。


### 论文摘要

Lie groups and their actions are ubiquitous in the description of physical systems, and we explore implications in the setting of model order reduction (MOR). We present a novel framework of MOR via Lie groups, called MORLie, in which high-dimensional dynamical systems on manifolds are approximated by low-dimensional dynamical systems on Lie groups. In comparison to other Lie group methods we are able to attack non-equivariant dynamics, which are frequent in practical applications, and we provide new non-intrusive MOR methods based on the presented geometric formulation. We also highlight numerically that MORLie has a lower error bound than the Kolmogorov $N$-width, which limits linear-subspace methods. The method is applied to various examples: 1. MOR of a simplified deforming body modeled by a noisy point cloud data following a sheering motion, where MORLie outperforms a naive POD approach in terms of accuracy and dimensionality reduction. 2. Reconstructing liver motion during respiration with data from edge detection in ultrasound scans, where MORLie reaches performance approaching the state of the art, while reducing the training time from hours on a computing cluster to minutes on a mobile workstation. 3. An analytic example showing that the method of freezing is analytically recovered as a special case, showing the generality of the geometric framework.

---

## 2. IEC3D-AD: A 3D Dataset of Industrial Equipment Components for Unsupervised Point Cloud Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.03267v1](http://arxiv.org/abs/2511.03267v1)

**作者:** Bingyang Guo, Hongjie Li, Ruiyun Yu, Hanzhe Liang, Jinbao Wang

**发布时间:** 2025-11-05

### GPT解析

### 总结

该研究开发了一个针对真实工业场景的点云异常检测数据集(IEC3D-AD)和新的3D异常检测范式(GMANet)，解决了现有数据集无法捕捉真实工业环境复杂性的问题。

### 背景

3D异常检测在工业制造中至关重要，特别是对核心设备组件的可靠性和安全。现有数据集如Real3D-AD和MVTec3D-AD无法捕捉真实工业环境中的复杂性和细微缺陷，限制了工业设备组件(如轴承、环和螺栓)的精确异常检测研究。

### 目的

开发一个针对真实工业场景的点云异常检测数据集，直接从实际生产线收集，确保高保真度和相关性，以支持更严格的异常检测任务。

### 方法

创建IEC3D-AD数据集，具有改进的点云分辨率和缺陷注释粒度；引入GMANet范式，基于几何形态分析生成合成点云样本，通过空间差异优化减少正常和异常点级特征之间的边界并增加重叠。

### 主要发现

大量实验证明，所提出的方法在IEC3D-AD和其他数据集上均表现出有效性。

### 结论

新开发的数据集和方法有效解决了工业3D异常检测中的挑战，提高了异常检测的精确度。

### 翻译

三维异常检测在工业制造中发挥着关键作用，特别是在确保核心设备组件的可靠性和安全性方面。尽管现有的三维数据集如Real3D-AD和MVTec3D-AD提供了广泛的应用支持，但它们无法捕捉真实工业环境中存在的复杂性和细微缺陷。这一限制阻碍了精确异常检测研究，特别是对于轴承、环和螺栓等工业设备组件。为了应对这一挑战，我们开发了一个针对真实工业场景的点云异常检测数据集(IEC3D-AD)。该数据集直接从实际生产线收集，确保了高保真度和相关性。与现有数据集相比，IEC3D-AD具有显著改进的点云分辨率和缺陷注释粒度，支持更严格的异常检测任务。此外，受生成式二维异常检测方法的启发，我们在IEC3D-AD上引入了一种新的三维异常检测范式(GMANet)。该范式基于几何形态分析生成合成点云样本，然后通过空间差异优化减少正常和异常点级特征之间的边界并增加重叠。大量实验证明了我们的方法在IEC3D-AD和其他数据集上的有效性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有3D异常检测数据集缺乏真实工业场景样本的问题，以及无法同时平衡空间覆盖率和数据密度的局限性。这个问题在现实中非常重要，因为工业设备组件是工业基础和产业链现代化的重要连接点，确保其可靠性和安全对工业生产至关重要；在研究中，现有数据集多来自模具和玩具而非真实工业环境，导致算法在实际应用中性能下降。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别到现有数据集缺乏真实工业场景数据，因此构建了专门针对真实工业场景的数据集IEC3D-AD。方法设计上，作者受生成式2D异常检测方法的启发，引入了基于几何形态分析的合成点云生成(SPCG)模块，借鉴了教师-学生网络的思想使用专家域和学徒域两个编码器，并参考了焦点损失来设计权重优化特征分布差异。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过几何形态分析生成合成异常样本，然后利用空间差异优化减少正常和异常特征边界，增加特征重叠。整体流程分为训练和测试两个阶段：训练阶段使用正常样本，通过SPCG生成合成异常样本，双编码器提取特征，计算差异并优化；测试阶段输入真实样本，使用训练好的编码器提取特征并计算异常分数识别异常点。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创建了IEC3D-AD数据集，直接从工业生产线收集，实现360度全覆盖和高点云密度；2)提出了GMANet方法，包含SPCG和SDO两个创新模块；3)提供了全面的基准测试。相比之前工作，不同之处在于：数据来源更真实(工业生产线vs模具玩具)，同时实现了高覆盖率和密度，缺陷比例更低(0.78%-2.28%)，且包含功能拓扑结构和微观几何畸变等真实工业特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一个专门针对工业设备组件的高质量3D点云异常检测数据集IEC3D-AD，以及一种基于几何形态分析和空间差异优化的无监督异常检测方法GMANet，显著提升了在真实工业场景中检测微小缺陷的能力。'}


### 论文摘要

3D anomaly detection (3D-AD) plays a critical role in industrial manufacturing, particularly in ensuring the reliability and safety of core equipment components. Although existing 3D datasets like Real3D-AD and MVTec 3D-AD offer broad application support, they fall short in capturing the complexities and subtle defects found in real industrial environments. This limitation hampers precise anomaly detection research, especially for industrial equipment components (IEC) such as bearings, rings, and bolts. To address this challenge, we have developed a point cloud anomaly detection dataset (IEC3D-AD) specific to real industrial scenarios. This dataset is directly collected from actual production lines, ensuring high fidelity and relevance. Compared to existing datasets, IEC3D-AD features significantly improved point cloud resolution and defect annotation granularity, facilitating more demanding anomaly detection tasks. Furthermore, inspired by generative 2D-AD methods, we introduce a novel 3D-AD paradigm (GMANet) on IEC3D-AD. This paradigm generates synthetic point cloud samples based on geometric morphological analysis, then reduces the margin and increases the overlap between normal and abnormal point-level features through spatial discrepancy optimization. Extensive experiments demonstrate the effectiveness of our method on both IEC3D-AD and other datasets.

---

## 3. Scheduling the Off-Diagonal Weingarten Loss of Neural SDFs for CAD Models

**论文链接:** [http://arxiv.org/abs/2511.03147v1](http://arxiv.org/abs/2511.03147v1)

**作者:** Haotian Yin, Przemyslaw Musialski

**发布时间:** 2025-11-05

**备注:** Lecture Notes in Computer Science (LNCS), 20th International  Symposium on Visual Computing 2025, 12 pages, 4 figures, preprint

### GPT解析

### 总结

本文提出了一种改进神经符号距离函数(SDFs)在CAD表面重建中的方法，通过引入时变调度策略优化非对角魏恩加滕(ODW)损失，显著提升了重建质量。

### 背景

神经符号距离函数已成为从点云进行几何重建的强大表示方法，但通常需要基于梯度和曲率的正则化来抑制伪影并保持结构保真度。FlatCAD虽然引入了高效的ODW损失作为二阶先验，但使用固定权重存在局限性。

### 目的

开发ODW损失的调度策略，在高初始权重稳定优化的同时，逐渐衰减权重以允许后期细尺度细节恢复，从而提升CAD重建质量。

### 方法

研究并实现了多种调度策略，包括常数、线性、五次和步长插值调度，以及增加的预热变体，并在ABC CAD数据集上进行了实验验证。

### 主要发现

时变调度策略显著优于固定权重方法，与FlatCAD基线相比，在Chamfer距离上实现了高达35%的性能提升。

### 结论

将调度作为曲率正则化的简单而有效的扩展，能够显著提升神经符号距离函数在CAD重建中的鲁棒性和质量。

### 翻译

神经符号距离函数(SDFs)已成为从点云进行几何重建的强大表示方法，但它们通常需要基于梯度和曲率的正则化来抑制伪影并保持结构保真度。FlatCAD引入了非对角魏恩加滕(ODW)损失作为CAD表面的高效二阶先验，以大约一半的计算成本近似完整Hessian正则化。然而，FlatCAD在整个训练过程中应用固定的ODW权重，这是次优的：强正则化稳定了早期优化，但在后期阶段抑制了细节恢复。我们提出了ODW损失的调度策略，分配高初始权重以稳定优化，并逐渐衰减它以允许细尺度细化。我们研究了常数、线性、五次和步长插值调度，以及一个增加的预热变体。在ABC CAD数据集上的实验表明，时变调度始终优于固定权重。我们的方法在Chamfer距离上比FlatCAD基线实现了高达35%的改进，确立了调度作为曲率正则化的简单而有效的扩展，用于鲁棒的CAD重建。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的问题是神经符号距离函数在重建CAD模型时使用固定的Off-Diagonal Weingarten损失权重导致的问题。固定权重在训练早期能稳定优化但会抑制后期细节恢复。这个问题重要是因为CAD模型通常由简单几何形状组成，需要平衡结构稳定性和细节精度，而固定权重无法满足训练不同阶段的需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认识到固定权重无法平衡训练早期的稳定性和后期的细节恢复，因此借鉴了课程学习和多任务学习中的动态权重调整思想。他们设计了多种调度策略，包括常量、线性、五次多项式和阶跃插值，并进行了系统比较。他们还受到Neuralangelo的粗到细策略启发，但将其应用于曲率正则化领域。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是采用'强开始-衰减'策略：训练初期使用高权重ODW损失稳定优化并抑制伪影，随着训练进行逐渐降低权重，允许网络恢复几何细节。实现流程是：定义一组关键点指定训练进度和对应权重；根据当前训练进度确定所在区间；在区间内根据选择的调度策略计算当前权重；使用动态权重更新总损失函数进行网络优化。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：首次系统研究ODW损失调度策略；提出多种调度方法；引入'强开始-衰减'训练范式；对比衰减和预热策略。相比之前工作，本文解决了固定权重无法适应训练不同阶段需求的问题，首次系统研究了ODW权重调度，证明了动态权重比固定权重效果更好，最多提升35%的Chamfer距离。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种针对神经符号距离函数中Off-Diagonal Weingarten损失的动态权重调度策略，通过初期高权重稳定优化、后期降低权重允许细节恢复，显著提高了CAD模型的重建质量。'}


### 论文摘要

Neural signed distance functions (SDFs) have become a powerful representation for geometric reconstruction from point clouds, yet they often require both gradient- and curvature-based regularization to suppress spurious warp and preserve structural fidelity. FlatCAD introduced the Off-Diagonal Weingarten (ODW) loss as an efficient second-order prior for CAD surfaces, approximating full-Hessian regularization at roughly half the computational cost. However, FlatCAD applies a fixed ODW weight throughout training, which is suboptimal: strong regularization stabilizes early optimization but suppresses detail recovery in later stages. We present scheduling strategies for the ODW loss that assign a high initial weight to stabilize optimization and progressively decay it to permit fine-scale refinement. We investigate constant, linear, quintic, and step interpolation schedules, as well as an increasing warm-up variant. Experiments on the ABC CAD dataset demonstrate that time-varying schedules consistently outperform fixed weights. Our method achieves up to a 35% improvement in Chamfer Distance over the FlatCAD baseline, establishing scheduling as a simple yet effective extension of curvature regularization for robust CAD reconstruction.

---

## 4. DentalSplat: Dental Occlusion Novel View Synthesis from Sparse Intra-Oral Photographs

**论文链接:** [http://arxiv.org/abs/2511.03099v1](http://arxiv.org/abs/2511.03099v1)

**作者:** Yiyi Miao, Taoyu Wu, Tong Chen, Sihao Li, Ji Jiang, Youpeng Yang, Angelos Stefanidis, Limin Yu, Jionglong Su

**发布时间:** 2025-11-05

### GPT解析

### 总结

本文提出了一种名为DentalSplat的框架，用于从稀疏正畸图像进行3D重建，解决了传统3D高斯溅射技术在正畸远程医疗应用中的局限性。

### 背景

在正畸治疗特别是远程医疗背景下，从多角度观察患者咬合情况有助于及时临床决策。传统3D高斯溅射技术需要密集多视角输入和精确相机姿态，但正畸病例通常只有三张稀疏图像（正面和双侧颊视图），使重建极具挑战性。

### 目的

开发一个有效框架，能够从稀疏正畸图像进行高质量3D重建，支持远程正畸诊断。

### 方法

DentalSplat框架利用先验引导的密集立体重建模型初始化点云，采用尺度自适应剪枝策略提高3DGS训练效率和重建质量，在极度稀疏视角情况下结合光流作为几何约束和梯度正则化来提高渲染保真度。

### 主要发现

该方法能有效处理稀疏输入场景，在咬合可视化方面实现卓越的新视角合成质量，优于现有最先进技术。

### 结论

DentalSplat成功解决了正畸治疗中从稀疏图像进行3D重建的挑战，为远程正畸诊断提供了有效工具。

### 翻译

在正畸治疗中，特别是在远程医疗背景下，从多角度观察患者的咬合情况有助于及时的临床决策。最近的3D高斯溅射（3DGS）技术在3D重建和新视角合成方面显示出强大潜力。然而，传统的3DGS流程通常依赖于密集捕获的多视角输入和精确初始化的相机姿态，限制了其实用性。正畸病例通常只有三张稀疏图像，即正面视图和双侧颊视图，使重建任务特别具有挑战性。输入视图的极度稀疏会严重降低重建质量，而相机姿态信息的缺失进一步复杂化了这一过程。为了克服这些限制，我们提出了DentalSplat，一个从稀疏正畸图像进行3D重建的有效框架。我们的方法利用先验引导的密集立体重建模型初始化点云，随后采用尺度自适应剪枝策略提高3DGS的训练效率和重建质量。在极度稀疏视角的情况下，我们进一步结合光流作为几何约束，并结合梯度正则化来提高渲染保真度。我们在一个包含950个临床病例的大型数据集上验证了我们的方法，以及一个额外的基于视频的测试集，包含195个病例，用于模拟现实世界远程正畸成像条件。实验结果表明，我们的方法能有效处理稀疏输入场景，并在咬合可视化方面实现卓越的新视角合成质量，优于最先进的技术。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从稀疏口腔内照片中合成新视角以观察牙齿咬合的问题。在正畸治疗中，特别是远程医疗场景下，这有助于及时临床决策。传统方法如CBCT和IOS需要专业设备，限制远程使用；而现有AI系统主要依赖单视图图像，无法全面评估咬合关系。此问题对提高远程正畸治疗效果、降低专业设备依赖具有重要意义。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：传统3DGS需要密集多视角输入和精确相机姿态；其他方法如MVSplat、Nope-NeRF假设有重叠视图，不适合真正稀疏场景；DUSt3R虽解决稀疏输入问题但在正畸应用中面临设备差异、牙齿反射、运动模糊等挑战。作者借鉴了3DGS和DUSt3R技术，并针对正畸场景特点设计了专门改进：尺度自适应修剪策略处理密集点云，集成光学流约束和梯度正则化提高渲染质量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合DUSt3R和3DGS优势，解决正畸场景中稀疏输入和未知相机姿态挑战。流程包括：1)用DUSt3R生成初始点云和相机姿态；2)应用尺度自适应修剪策略处理点云；3)在3DGS优化中结合光学流约束确保几何一致性；4)使用梯度约束增强密集化过程；5)通过联合优化相机姿态和3D高斯原语实现高质量渲染。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)尺度自适应修剪(SAP)策略，减少点云大小同时保持质量；2)增强的差分高斯光栅化模块，集成光学流约束和梯度权重计算；3)专门针对正畸场景的优化，处理反射、模糊等问题。不同之处：专为三张稀疏图像设计，而其他方法假设更多输入；解决DUSt3R密集点云导致的计算效率问题；结合多种约束提高复杂牙齿结构渲染质量；无需相机姿态信息也能工作。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'DentalSplat首次实现了从稀疏、无姿态的口腔内图像中快速（一分钟内）生成高质量牙齿咬合3D重建和新视角合成的方法，显著优于现有技术，为远程正畸治疗提供了新的解决方案。'}


### 论文摘要

In orthodontic treatment, particularly within telemedicine contexts, observing patients' dental occlusion from multiple viewpoints facilitates timely clinical decision-making. Recent advances in 3D Gaussian Splatting (3DGS) have shown strong potential in 3D reconstruction and novel view synthesis. However, conventional 3DGS pipelines typically rely on densely captured multi-view inputs and precisely initialized camera poses, limiting their practicality. Orthodontic cases, in contrast, often comprise only three sparse images, specifically, the anterior view and bilateral buccal views, rendering the reconstruction task especially challenging. The extreme sparsity of input views severely degrades reconstruction quality, while the absence of camera pose information further complicates the process. To overcome these limitations, we propose DentalSplat, an effective framework for 3D reconstruction from sparse orthodontic imagery. Our method leverages a prior-guided dense stereo reconstruction model to initialize the point cloud, followed by a scale-adaptive pruning strategy to improve the training efficiency and reconstruction quality of 3DGS. In scenarios with extremely sparse viewpoints, we further incorporate optical flow as a geometric constraint, coupled with gradient regularization, to enhance rendering fidelity. We validate our approach on a large-scale dataset comprising 950 clinical cases and an additional video-based test set of 195 cases designed to simulate real-world remote orthodontic imaging conditions. Experimental results demonstrate that our method effectively handles sparse input scenarios and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art techniques.

---

## 5. From Propagation to Prediction: Point-level Uncertainty Evaluation of MLS Point Clouds under Limited Ground Truth

**论文链接:** [http://arxiv.org/abs/2511.03053v1](http://arxiv.org/abs/2511.03053v1)

**作者:** Ziyang Xu, Olaf Wysocki, Christoph Holst

**发布时间:** 2025-11-04

### GPT解析

### 总结

该研究提出了一种基于学习的MLS点云不确定性评估框架，结合最优邻域估计和几何特征提取，实验证明其可行且高效。

### 背景

移动激光扫描点云在许多高精度应用（如扫描到建筑信息模型、变形分析和三维建模）中的可靠使用依赖于不确定性评估。然而，在许多实际应用中，获取用于评估的地面真实值通常成本高昂且不可行。

### 目的

减少不确定性评估研究中对地面真实值的长期依赖，提出一种基于学习的框架用于MLS点云的不确定性评估。

### 方法

提出了一种基于学习的框架，结合最优邻域估计和几何特征提取，使用XGBoost模型进行预测。

### 主要发现

提出的框架可行；XGBoost模型与随机森林相比具有完全相当的准确性，同时效率更高（快约3倍）；几何特征可用于预测由点到点距离量化的点级不确定性。

### 结论

MLS点云的不确定性是可以学习的，为不确定性评估研究提供了新的基于学习的视角。

### 翻译

评估不确定性对于移动激光扫描点云在许多高精度应用（如扫描到建筑信息模型、变形分析和三维建模）中的可靠使用至关重要。然而，在许多实际应用中，获取用于评估的地面真实值通常成本高昂且不可行。为了减少不确定性评估研究中对地面真实值的长期依赖，本研究提出了一个MLS点云的基于学习框架，结合了最优邻域估计和几何特征提取。在真实世界数据集上的实验表明，所提出的框架是可行的，XGBoost模型与随机森林相比具有完全相当的准确性，同时实现了更高的效率（快约3倍），初步证明了几何特征可用于预测由点到点距离量化的点级不确定性。总之，这项研究表明MLS点云的不确定性是可以学习的，为不确定性评估研究提供了新的基于学习的视角。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决MLS点云不确定性评估中过度依赖地面真实值(GT)的问题。这个问题很重要，因为在高精度应用如Scan-to-BIM、变形分析和3D建模中，不仅需要准确几何信息，还需要可靠的不确定性估计；而传统方法要么难以全面建模所有误差源，要么获取GT成本过高，限制了MLS技术的广泛应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统前向和后向建模方法的局限性，特别是后向建模对GT的依赖问题。然后转向学习-based方法，将不确定性评估转化为监督学习问题，学习点的误差与局部几何特征间的关系。方法设计借鉴了TLS测量中使用集成方法的研究成果，采用了点云分类中表现良好的几何特征提取方法，以及最优邻域估计策略，但将其首次应用于真实世界的MLS数据集。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是点级不确定性可以通过局部几何特征学习和预测，无需为每个新场景都获取GT。流程包括：1)数据准备(使用C2C距离量化不确定性，筛选C2C<80mm的点)；2)特征工程(估计最优邻域，提取26种几何特征)；3)模型训练(Random Forest和XGBoost两种集成学习模型)；4)模型测试(使用多种评估指标和可视化)；5)结果分析(比较模型性能，进行特征重要性分析)。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个针对MLS点云的学习-based不确定性评估框架；2)集成最优邻域估计与几何特征提取；3)证明XGBoost在保持与RF相当精度的同时效率更高；4)通过SHAP和排列重要性分析揭示关键几何特征。相比之前工作，本文从TLS扩展到复杂MLS场景，从实验室条件扩展到真实世界，使用C2C距离提供更敏感的局部误差表征，并减少了GT需求，提高了实用性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文首次提出了一种基于机器学习的MLS点云点级不确定性评估框架，通过几何特征预测不确定性，减少了对地面真实值的依赖，同时证明了XGBoost模型在保持与随机森林相当精度的同时具有更高的计算效率。'}


### 论文摘要

Evaluating uncertainty is critical for reliable use of Mobile Laser Scanning (MLS) point clouds in many high-precision applications such as Scan-to-BIM, deformation analysis, and 3D modeling. However, obtaining the ground truth (GT) for evaluation is often costly and infeasible in many real-world applications. To reduce this long-standing reliance on GT in uncertainty evaluation research, this study presents a learning-based framework for MLS point clouds that integrates optimal neighborhood estimation with geometric feature extraction. Experiments on a real-world dataset show that the proposed framework is feasible and the XGBoost model delivers fully comparable accuracy to Random Forest while achieving substantially higher efficiency (about 3 times faster), providing initial evidence that geometric features can be used to predict point-level uncertainty quantified by the C2C distance. In summary, this study shows that MLS point clouds' uncertainty is learnable, offering a novel learning-based viewpoint towards uncertainty evaluation research.

---

## 6. Curvature of high-dimensional data

**论文链接:** [http://arxiv.org/abs/2511.02873v1](http://arxiv.org/abs/2511.02873v1)

**作者:** Jiayi Chen, Mohammad Javad Latifi Jebelli, Daniel N. Rockmore

**发布时间:** 2025-11-04

### GPT解析

### 总结

这篇论文研究了从带噪声的样本数据中估计曲率的问题，探讨了高维情况下曲率估计的偏差问题，并提出了一种改进的概率框架来构建更准确的曲率估计器。

### 背景

对于维度大于一的流形，存在多种局部曲率的定义，每种定义对应不同的估计过程。最近的研究证明了'局部点云曲率'估计会随着点云密度趋近于无限而收敛到相关的局部曲率光滑概念。

### 目的

研究收敛定理的实际局限性，分析曲率估计中偏差的显著影响，特别是在高维情况下，并提出构建更准确曲率估计器的方法。

### 方法

提出一个概率框架，为任意噪声模型构建更准确的曲率估计器，并在高达十二维的球体上进行实验验证。

### 主要发现

偏差在高维情况下会急剧增加，在高维情况下，朴素曲率估计落在真实曲率附近小区间内的概率可能接近于零；提出的概率框架能够构建更准确的曲率估计器。

### 结论

在高维流形中，曲率估计面临显著的偏差挑战，但通过提出的概率框架可以有效提高估计的准确性。

### 翻译

我们考虑估计曲率的问题，其中数据可以被视为来自基础流形的噪声样本。对于维度大于一的流形，存在多种局部曲率的定义，每种定义对给定数据集提出了不同的估计过程。最近，在证明'局部点云曲率'估计随着点云密度趋近于无限而收敛到相关的局部曲率光滑概念方面取得了进展。在此，我们研究了这些收敛定理的实际局限性，并讨论了最近文献中报道的此类估计中偏差的显著影响。我们提供了理论论证，证明偏差在高维情况下急剧增加，以至于在高维情况下，朴素曲率估计落在真实曲率附近小区间内的概率可能接近于零。我们提出了一个概率框架，能够为任意噪声模型构建更准确的曲率估计器。我们在高达十二维的球体上的实验支持了我们技术的有效性。


### 论文摘要

We consider the problem of estimating curvature where the data can be viewed as a noisy sample from an underlying manifold. For manifolds of dimension greater than one there are multiple definitions of local curvature, each suggesting a different estimation process for a given data set. Recently, there has been progress in proving that estimates of ``local point cloud curvature" converge to the related smooth notion of local curvature as the density of the point cloud approaches infinity. Herein we investigate practical limitations of such convergence theorems and discuss the significant impact of bias in such estimates as reported in recent literature. We provide theoretical arguments for the fact that bias increases drastically in higher dimensions, so much so that in high dimensions, the probability that a naive curvature estimate lies in a small interval near the true curvature could be near zero. We present a probabilistic framework that enables the construction of more accurate estimators of curvature for arbitrary noise models. The efficacy of our technique is supported with experiments on spheres of dimension as large as twelve.

---

## 7. POEMS: Product of Experts for Interpretable Multi-omic Integration using Sparse Decoding

**论文链接:** [http://arxiv.org/abs/2511.03464v1](http://arxiv.org/abs/2511.03464v1)

**作者:** Mihriban Kocak Balik, Pekka Marttinen, Negar Safinianaini

**发布时间:** 2025-11-05

### GPT解析

### 总结

POEMS是一种新的无监督概率框架，用于整合多组学数据，通过稀疏解码和专家乘积模型，在保持预测性能的同时提供可解释性，实现了生物标志物发现和跨组学关联。

### 背景

整合不同分子层（即多组学数据）对于理解疾病复杂性至关重要；然而，大多数深度生成模型要么优先考虑预测性能而牺牲可解释性，要么通过线性化解码器来强制可解释性，从而削弱了网络的非线性表达能力。

### 目的

克服预测性能和可解释性之间的权衡，开发一种能够同时保持预测性能并提供可解释性的无监督概率框架。

### 方法

引入POEMS框架，通过以下方式提供可解释性而不需要线性化网络的任何部分：1)使用稀疏连接将特征映射到潜在因子，直接转化为生物标志物发现；2)使用专家乘积模型通过共享潜在空间实现跨组学关联；3)通过门控网络报告每个组学的贡献，该网络自适应地计算它们在表示学习中的影响；此外，还提出了一种高效的稀疏解码器。

### 主要发现

在癌症亚型分型案例研究中，POEMS实现了具有竞争力的聚类和分类性能，同时提供了一套新的解释方法，证明基于生物标志物的洞察力和预测准确性可以在多组学表示学习中共存。

### 结论

POEMS框架成功整合了预测性能和可解释性，证明在多组学表示学习中，生物标志物洞察力和预测准确性可以共存。

### 翻译

整合不同的分子层，即多组学数据，对于揭示疾病复杂性至关重要；然而，大多数深度生成模型要么优先考虑预测性能而牺牲可解释性，要么通过线性化解码器来强制可解释性，从而削弱了网络的非线性表达能力。为了克服这种权衡，我们引入了POEMS：使用稀疏解码的可解释多组学集成的专家乘积模型，这是一个无监督概率框架，在提供可解释性的同时保持预测性能。POEMS通过以下方式在不线性化网络任何部分的情况下提供可解释性：1)使用稀疏连接将特征映射到潜在因子，直接转化为生物标志物发现；2)通过专家乘积模型使用共享潜在空间实现跨组学关联；3)通过门控网络报告每个组学的贡献，该网络自适应地计算它们在表示学习中的影响。此外，我们还提出了一种高效的稀疏解码器。在癌症亚型分型案例研究中，POEMS实现了具有竞争力的聚类和分类性能，同时提供了一套新的解释方法，证明了基于生物标志物的洞察力和预测准确性可以在多组学表示学习中共存。


### 论文摘要

Integrating different molecular layers, i.e., multiomics data, is crucial for unraveling the complexity of diseases; yet, most deep generative models either prioritize predictive performance at the expense of interpretability or enforce interpretability by linearizing the decoder, thereby weakening the network's nonlinear expressiveness. To overcome this tradeoff, we introduce POEMS: Product Of Experts for Interpretable Multiomics Integration using Sparse Decoding, an unsupervised probabilistic framework that preserves predictive performance while providing interpretability. POEMS provides interpretability without linearizing any part of the network by 1) mapping features to latent factors using sparse connections, which directly translates to biomarker discovery, 2) allowing for cross-omic associations through a shared latent space using product of experts model, and 3) reporting contributions of each omic by a gating network that adaptively computes their influence in the representation learning. Additionally, we present an efficient sparse decoder. In a cancer subtyping case study, POEMS achieves competitive clustering and classification performance while offering our novel set of interpretations, demonstrating that biomarker based insight and predictive accuracy can coexist in multiomics representation learning.

---

## 8. ProM3E: Probabilistic Masked MultiModal Embedding Model for Ecology

**论文链接:** [http://arxiv.org/abs/2511.02946v1](http://arxiv.org/abs/2511.02946v1)

**作者:** Srikumar Sastry, Subash Khanal, Aayush Dhakal, Jiayu Lin, Dan Cher, Phoenix Jarosz, Nathan Jacobs

**发布时间:** 2025-11-04

**备注:** 21 pages, 16 figures

### GPT解析

### 总结

这项研究介绍了一个名为ProM3E的概率掩码多模态嵌入模型，用于生态学中的多模态表示生成。该模型基于嵌入空间中的掩码模态重建，支持模态反转，能够分析模态融合的可行性，并提出了新的跨模态检索方法，展示了优越的表示学习能力。

### 背景

生态学研究需要处理多种模态的数据，但现有方法可能无法有效处理任意模态间的转换和融合。需要一种能够处理多模态数据并支持任意模态间转换的模型。

### 目的

开发一个能够支持生态学多模态表示任意生成的模型，学习推断缺失的模态，分析模态融合的可行性，并提高跨模态检索性能。

### 方法

提出ProM3E模型，基于嵌入空间中的掩码模态重建；支持模态反转功能；利用概率性质分析模态融合的可行性；提出结合模态间和模态内相似性的跨模态检索方法；使用隐藏表示进行线性探测任务。

### 主要发现

所提出的跨模态检索方法在所有检索任务中实现了优越的性能；模型展示了卓越的表示学习能力；能够有效分析不同模态融合的可行性。

### 结论

ProM3E模型为生态学多模态数据的处理提供了有效解决方案，支持任意模态间的转换和融合，并在跨模态检索和表示学习任务中表现出色。研究团队将公开代码、数据集和模型以促进进一步研究。

### 翻译

我们介绍了ProM3E，一个用于生态学多模态表示的任意生成概率掩码多模态嵌入模型。ProM3E基于嵌入空间中的掩码模态重建，学习在给定少量上下文模态的情况下推断缺失的模态。根据设计，我们的模型支持嵌入空间中的模态反转。我们模型的概率性质使我们能够分析融合各种模态以用于给定下游任务的可行性，本质上学习融合什么。利用我们模型的这些特性，我们提出了一种新的跨模态检索方法，该方法结合了模态间和模态内相似性，以在所有检索任务中实现卓越的性能。我们进一步利用我们模型的隐藏表示来执行线性探测任务，并展示了我们模型卓越的表示学习能力。我们所有的代码、数据集和模型将在https://vishu26.github.io/prom3e上发布。


### 论文摘要

We introduce ProM3E, a probabilistic masked multimodal embedding model for any-to-any generation of multimodal representations for ecology. ProM3E is based on masked modality reconstruction in the embedding space, learning to infer missing modalities given a few context modalities. By design, our model supports modality inversion in the embedding space. The probabilistic nature of our model allows us to analyse the feasibility of fusing various modalities for given downstream tasks, essentially learning what to fuse. Using these features of our model, we propose a novel cross-modal retrieval approach that mixes inter-modal and intra-modal similarities to achieve superior performance across all retrieval tasks. We further leverage the hidden representation from our model to perform linear probing tasks and demonstrate the superior representation learning capability of our model. All our code, datasets and model will be released at https://vishu26.github.io/prom3e.

---

## 9. XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations

**论文链接:** [http://arxiv.org/abs/2511.02776v1](http://arxiv.org/abs/2511.02776v1)

**作者:** Shichao Fan, Kun Wu, Zhengping Che, Xinhua Wang, Di Wu, Fei Liao, Ning Liu, Yixue Zhang, Zhen Zhao, Zhiyuan Xu, Meng Li, Qingjie Liu, Shanghang Zhang, Min Wan, Jian Tang

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了XR-1框架，通过统一视觉运动编码(UVMC)和三阶段训练范式解决了视觉语言动作模型面临的两个基本挑战：从高维观测中产生精确低级动作和弥合跨异构数据源的领域差距。

### 背景

大规模机器人数据集和视觉语言模型的发展推动了VLA模型研究，但现有模型面临两个挑战：从高维观测产生精确低级动作，以及弥合不同机器人形态和人类演示数据之间的领域差距。现有方法未能充分利用大规模异构数据集中的互补多模态知识。

### 目的

开发一个多功能可扩展的VLA学习框架(XR-1)，能够在多样化机器人、任务和环境上有效工作。

### 方法

引入统一视觉运动编码(UVMC)，一种通过双分支VQ-VAE学习的离散潜在表示，同时编码视觉动力学和机器人运动。采用三阶段训练范式：自监督的UVMC学习、UVMC引导的大规模跨形态机器人数据集预训练、以及任务特定的后训练。

### 主要发现

在六种不同机器人形态上进行了超过14,000次滚动的真实世界实验，涵盖120多种操作任务。XR-1持续优于π₀.₅、π₀、RDT、UniVLA和GR00T-N1.5等基线方法，并对新物体、背景变化、干扰物和光照变化展现出强大泛化能力。

### 结论

XR-1通过UVMC和三阶段训练范式成功解决了VLA模型的关键挑战，在多样化场景中实现了优越的性能和泛化能力。

### 翻译

最近大规模机器人数据集和视觉语言模型(VLMs)的进展推动了视觉语言动作(VLA)模型的研究。然而，现有VLA模型仍面临两个基本挑战：(i)从高维观测中产生精确的低级动作，(ii)弥合跨异构数据源的领域差距，包括多样化的机器人形态和人类演示。现有方法通常从视觉动力学或机器人动作中编码潜在变量来指导策略学习，但它们未能充分利用大规模、异构数据集中存在的互补多模态知识。在这项工作中，我们提出了X机器人模型1(XR-1)，一个适用于多样化机器人、任务和环境的多功能可扩展VLA学习框架。XR-1引入了统一视觉运动编码(UVMC)，一种通过双分支VQ-VAE学习的离散潜在表示，可同时编码视觉动力学和机器人运动。UVMC通过(i)作为观测和动作之间的中间表示，和(ii)对齐来自异构数据源的多模态动态信息以捕获互补知识来解决这些挑战。为了有效利用UVMC，我们提出了三阶段训练范式：(i)自监督的UVMC学习，(ii)在大型跨形态机器人数据集上进行UVMC引导的预训练，和(iii)任务特定的后训练。我们通过在六种不同机器人形态上进行超过14,000次滚动的广泛真实世界实验验证了XR-1，涵盖120多种不同的操作任务。XR-1在性能上持续优于最先进的基线方法，如π₀.₅、π₀、RDT、UniVLA和GR00T-N1.5，同时展现出对新物体、背景变化、干扰物和光照变化的强大泛化能力。我们的项目网址是https://xr-1-vla.github.io/。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决两个问题：1) 从高维观测生成精确的低级行动困难；2) 跨形态数据集利用受到形态异质性阻碍。这些问题在现实中很重要，因为精确的低级行动对机器人执行实际任务至关重要，特别是在需要精确操作的场景；而有效利用跨形态数据集可以提高机器人学习的数据效率和泛化能力，推动机器人向更通用、适应性强的方向发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有方法要么只编码视觉动态，要么只编码机器人动作，没有充分利用多模态知识。受人类认知启发——人类自然将异构感官输入整合成跨模态代码，作者设计了双分支VQ-VAE架构，将视觉动态和机器人运动编码到共享的离散潜在空间中，并添加了对齐损失强制视觉代码与运动代码保持一致。该方法借鉴了VQ-VAE架构、大规模预训练数据集和视觉-语言模型，但创新性地将它们整合用于统一的视觉-运动表示学习。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过学习统一的视觉-运动表示(UVMC)实现跨模态对齐，作为观测和动作之间的中间表示。整体实现流程分为三阶段：1) 第一阶段学习UVMC，使用双分支VQ-VAE分别编码视觉动态和机器人运动到共享潜在空间，并添加跨模态对齐损失；2) 第二阶段UVMC指导预训练，将UVMC作为监督信号注入VLM；3) 第三阶段任务特定后训练，微调VLA策略提高特定任务性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 统一的视觉-运动表示(UVMC)同时编码视觉和运动信息；2) 三阶段训练范式增加UVMC学习阶段；3) 跨模态对齐损失强制视觉与运动代码一致；4) 模型无关设计可灵活应用于不同VLA架构。相比之前工作，XR-1不仅利用视觉信息还整合机器人运动信息，不需要大量标记的机器人动作数据，能从人类演示视频中学习，并通过增加UVMC学习阶段更有效利用异构数据源。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'XR-1通过学习统一的视觉-运动表示，实现了跨数据利用、跨模态对齐和跨形态控制的通用视觉-语言-动作模型，显著提高了机器人在多样化任务和环境中的性能和泛化能力。'}


### 论文摘要

Recent progress in large-scale robotic datasets and vision-language models (VLMs) has advanced research on vision-language-action (VLA) models. However, existing VLA models still face two fundamental challenges: (i) producing precise low-level actions from high-dimensional observations, (ii) bridging domain gaps across heterogeneous data sources, including diverse robot embodiments and human demonstrations. Existing methods often encode latent variables from either visual dynamics or robotic actions to guide policy learning, but they fail to fully exploit the complementary multi-modal knowledge present in large-scale, heterogeneous datasets. In this work, we present X Robotic Model 1 (XR-1), a novel framework for versatile and scalable VLA learning across diverse robots, tasks, and environments. XR-1 introduces the \emph{Unified Vision-Motion Codes (UVMC)}, a discrete latent representation learned via a dual-branch VQ-VAE that jointly encodes visual dynamics and robotic motion. UVMC addresses these challenges by (i) serving as an intermediate representation between the observations and actions, and (ii) aligning multimodal dynamic information from heterogeneous data sources to capture complementary knowledge. To effectively exploit UVMC, we propose a three-stage training paradigm: (i) self-supervised UVMC learning, (ii) UVMC-guided pretraining on large-scale cross-embodiment robotic datasets, and (iii) task-specific post-training. We validate XR-1 through extensive real-world experiments with more than 14,000 rollouts on six different robot embodiments, spanning over 120 diverse manipulation tasks. XR-1 consistently outperforms state-of-the-art baselines such as $\pi_{0.5}$, $\pi_0$, RDT, UniVLA, and GR00T-N1.5 while demonstrating strong generalization to novel objects, background variations, distractors, and illumination changes. Our project is at https://xr-1-vla.github.io/.

---

## 10. DANIEL: A Distributed and Scalable Approach for Global Representation Learning with EHR Applications

**论文链接:** [http://arxiv.org/abs/2511.02754v1](http://arxiv.org/abs/2511.02754v1)

**作者:** Zebin Wang, Ziming Gan, Weijing Tang, Zongqi Xia, Tianrun Cai, Tianxi Cai, Junwei Lu

**发布时间:** 2025-11-04

### GPT解析

### 总结

本研究重新审视了Ising模型，开发了一个分布式框架，使具有内在低秩结构的大规模二元数据的可扩展和隐私保护表示学习成为可能。通过双因子梯度下降优化非凸替代损失函数，在多机构电子健康记录数据集上验证了算法的优越性，突显了在高维环境中进行统计推断的潜力。

### 背景

传统的概率图模型在现代数据环境中面临基本挑战，这些数据环境具有高维度、源异构性和严格的数据共享限制。

### 目的

重新审视Ising模型，并开发一个分布式框架，使具有内在低秩结构的大规模二元数据的可扩展和隐私保护表示学习成为可能。

### 方法

通过双因子梯度下降优化非凸替代损失函数，与传统的凸方法相比，提供了显著的计算和通信优势。

### 主要发现

在匹兹堡大学医学中心和马萨诸塞州总医院的58,248名患者的多机构电子健康记录数据集上评估了算法，在全局表示学习和下游临床任务（包括关系检测、患者表型和患者聚类）中表现出优越性能。

### 结论

这些结果突显了在联邦、高维环境中进行统计推断的更广泛潜力，同时解决了数据复杂性和多机构整合的实际挑战。

### 翻译

传统的概率图模型在现代数据环境中面临基本挑战，这些数据环境具有高维度、源异构性和严格的数据共享限制。在本工作中，我们重新审视了Ising模型，它是马尔可夫随机场家族中一个成熟的成员，并开发了一个分布式框架，使具有内在低秩结构的大规模二元数据的可扩展和隐私保护表示学习成为可能。我们的方法通过双因子梯度下降优化非凸替代损失函数，与传统的凸方法相比，提供了显著的计算和通信优势。我们在匹兹堡大学医学中心和马萨诸塞州总医院的58,248名患者的多机构电子健康记录数据集上评估了我们的算法，在全局表示学习和下游临床任务（包括关系检测、患者表型和患者聚类）中表现出优越性能。这些结果突显了在联邦、高维环境中进行统计推断的更广泛潜力，同时解决了数据复杂性和多机构整合的实际挑战。


### 论文摘要

Classical probabilistic graphical models face fundamental challenges in modern data environments, which are characterized by high dimensionality, source heterogeneity, and stringent data-sharing constraints. In this work, we revisit the Ising model, a well-established member of the Markov Random Field (MRF) family, and develop a distributed framework that enables scalable and privacy-preserving representation learning from large-scale binary data with inherent low-rank structure. Our approach optimizes a non-convex surrogate loss function via bi-factored gradient descent, offering substantial computational and communication advantages over conventional convex approaches. We evaluate our algorithm on multi-institutional electronic health record (EHR) datasets from 58,248 patients across the University of Pittsburgh Medical Center (UPMC) and Mass General Brigham (MGB), demonstrating superior performance in global representation learning and downstream clinical tasks, including relationship detection, patient phenotyping, and patient clustering. These results highlight a broader potential for statistical inference in federated, high-dimensional settings while addressing the practical challenges of data complexity and multi-institutional integration.

---

## 11. Modality-Transition Representation Learning for Visible-Infrared Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.02685v1](http://arxiv.org/abs/2511.02685v1)

**作者:** Chao Yuan, Zanwu Liu, Guiwei Zhang, Haoxuan Xu, Yujian Zhao, Guanglin Niu, Bo Li

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了一种名为模态转换表示学习（MTRL）的新型VI-ReID框架，通过中间生成的图像作为模态转换的桥梁，有效对齐可见光和红外模态的特征，无需额外参数即可提升性能。

### 背景

可见光-红外行人重识别（VI-ReID）技术能够在背景光照变化场景中关联不同模态的行人图像，但可见光和红外模态间存在本质差距。现有方法主要依赖中间表示来对齐跨模态特征，但这些方法要么通过生成中间图像（数据增强），要么融合中间特征（参数多，可解释性差），且未能充分利用中间特征。

### 目的

解决可见光和红外模态间的差距问题，改进现有的VI-ReID方法，使其能够更有效地对齐跨模态特征，同时保持推理速度不增加额外参数。

### 方法

提出模态转换表示学习（MTRL）框架，使用中间生成的图像作为从可见光到红外模态的传输器，这些图像与原始可见光图像完全对齐且与红外模态相似。采用模态转换对比损失和模态查询正则化损失进行训练，实现更有效的跨模态特征对齐。

### 主要发现

所提出的MTRL框架不需要额外参数，保持与骨干网络相同的推理速度，同时在VI-ReID任务上性能得到提升。在三个典型VI-ReID数据集上的实验结果表明，该方法显著且一致地优于现有最先进方法。

### 结论

通过模态转换表示学习框架，可以有效解决可见光和红外模态间的差距问题，提升VI-ReID性能，且不增加计算负担，为跨模态行人重识别提供了新的有效解决方案。

### 翻译

可见光-红外行人重识别（VI-ReID）技术能够在实际场景中关联可见光和红外模态的行人图像，特别是在背景光照变化的情况下。然而，这两种模态之间本质上存在显著差距。此外，现有方法主要依靠中间表示来对齐同一人的跨模态特征。这些中间特征表示通常是通过生成中间图像（一种数据增强）或融合中间特征（参数更多，可解释性差）来创建的，并且它们没有很好地利用中间特征。因此，我们提出了一种通过模态转换表示学习（MTRL）的新型VI-ReID框架，使用中间生成的图像作为从可见光到红外模态的传输器，这些图像与原始可见光图像完全对齐，并且与红外模态相似。之后，使用模态转换对比损失和模态查询正则化损失进行训练，可以更有效地对齐跨模态特征。值得注意的是，我们提出的框架不需要任何额外参数，在保持与骨干网络相同推理速度的同时，提高了其在VI-ReID任务上的性能。大量实验结果表明，在三个典型的VI-ReID数据集上，我们的模型显著且一致地优于现有的最先进方法。


### 论文摘要

Visible-infrared person re-identification (VI-ReID) technique could associate the pedestrian images across visible and infrared modalities in the practical scenarios of background illumination changes. However, a substantial gap inherently exists between these two modalities. Besides, existing methods primarily rely on intermediate representations to align cross-modal features of the same person. The intermediate feature representations are usually create by generating intermediate images (kind of data enhancement), or fusing intermediate features (more parameters, lack of interpretability), and they do not make good use of the intermediate features. Thus, we propose a novel VI-ReID framework via Modality-Transition Representation Learning (MTRL) with a middle generated image as a transmitter from visible to infrared modals, which are fully aligned with the original visible images and similar to the infrared modality. After that, using a modality-transition contrastive loss and a modality-query regularization loss for training, which could align the cross-modal features more effectively. Notably, our proposed framework does not need any additional parameters, which achieves the same inference speed to the backbone while improving its performance on VI-ReID task. Extensive experimental results illustrate that our model significantly and consistently outperforms existing SOTAs on three typical VI-ReID datasets.

---

## 12. Using Deep Learning for Robust Classification of Fast Radio Bursts

**论文链接:** [http://arxiv.org/abs/2511.02634v1](http://arxiv.org/abs/2511.02634v1)

**作者:** Rohan Arni, Carlos Blanco, Anirudh Prabhu

**发布时间:** 2025-11-04

**备注:** 15 pages, 4 figures, 9 tables. Comments are welcome

### GPT解析

### 总结

本研究使用深度学习方法对快速射电暴进行分类并分析其潜在结构模式，通过监督变分自编码器模型实现了高分类准确率，并揭示了重复发射器与非重复发射器之间的特征差异。

### 背景

快速射电暴的本质仍然未知，但群体层面的分析可以阐明这些信号中的潜在结构。研究使用了第一个CHIME目录中的数据。

### 目的

使用深度学习方法对FRBs进行分类，并分析从CHIME目录中学习的潜在空间中的结构模式。

### 方法

采用监督变分自编码器(sVAE)架构，结合变分自编码器的表示学习能力和监督分类任务，构建学习到的潜在空间并进行进一步降维以寻找数据中的潜在结构。

### 主要发现

sVAE模型对FRB重复发射器实现了高分类准确率，揭示了重复发射器和非重复发射器群体之间的分离。色散测量过剩、光谱指数和光谱运行是区分重复发射器和非重复发射器的主导特征。研究还确定了四个非重复FRBs作为重复发射器候选者，其中两个在先前研究中已被独立标记。

### 结论

深度学习方法可以有效地对FRBs进行分类并揭示其潜在结构，重复发射器和非重复发射器之间存在可区分的特征差异。

### 翻译

尽管快速射电暴的性质仍然未知，但群体层面的分析可以阐明这些信号中的潜在结构。在本研究中，我们采用深度学习方法来对FRBs进行分类，并分析从第一个CHIME目录中学习的潜在空间中的结构模式。我们采用监督变分自编码器架构，该架构结合了变分自编码器的表示学习能力和监督分类任务，从而提高了分类性能和潜在空间的可解释性。我们在构建的潜在空间中执行进一步的降维，以寻找数据中的潜在结构。我们的结果表明，sVAE模型对FRB重复发射器实现了高分类准确率，并揭示了重复发射器与非重复发射器群体之间的分离。通过对潜在空间的进一步分析，我们观察到色散测量过剩、光谱指数和光谱运行是区分重复发射器与非重复发射器的主导特征。我们还确定了四个非重复FRBs作为重复发射器候选者，其中两个在先前研究中已被独立标记。


### 论文摘要

While the nature of fast radio bursts (FRBs) remains unknown, population-level analyses can elucidate underlying structure in these signals. In this study, we employ deep learning methods to both classify FRBs and analyze structural patterns in the latent space learned from the first CHIME catalog. We adopt a Supervised Variational Autoencoder (sVAE) architecture which combines the representational learning capabilities of Variational Autoencoders (VAEs) with a supervised classification task, thereby improving both classification performance and the interpretability of the latent space. We construct a learned latent space in which we perform further dimensionality reduction to find underlying structure in the data. Our results demonstrate that the sVAE model achieves high classification accuracy for FRB repeaters and reveals separation between repeater and non-repeater populations. Upon further analysis of the latent space, we observe that dispersion measure excess, spectral index, and spectral running are the dominant features distinguishing repeaters from non-repeaters. We also identify four non-repeating FRBs as repeater candidates, two of which have been independently flagged in previous studies.

---

## 13. SKGE: Spherical Knowledge Graph Embedding with Geometric Regularization

**论文链接:** [http://arxiv.org/abs/2511.02460v1](http://arxiv.org/abs/2511.02460v1)

**作者:** Xuan-Truong Quan, Xuan-Son Quan, Duc Do Minh, Vinh Nguyen Van

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了球面知识图谱嵌入(SKGE)模型，通过将实体表示限制在超球面上，克服了传统欧几里得空间KGE模型的局限性。实验证明SKGE在多个基准测试上优于TransE模型，特别是在大规模数据集上表现出色。研究表明几何约束作为正则化器可以提高性能，且球面几何创造了更好的负采样环境。

### 背景

知识图谱嵌入(KGE)已成为多关系数据表示学习的基本技术。许多经典模型(如TransE)在无界欧几里得空间中操作，这种方法在建模复杂关系时存在固有局限性，可能导致训练效率低下。

### 目的

提出一种新的模型来挑战现有范式，通过将实体表示限制在紧凑流形(超球面)上，解决现有方法的局限性。

### 方法

提出了球面知识图谱嵌入(SKGE)模型，使用可学习的非线性球化层将实体映射到球面上，并将关系解释为混合平移-投影变换。

### 主要发现

在三个基准数据集(FB15k-237, CoDEx-S, CoDEx-M)上的广泛实验表明，SKGE一致且显著优于TransE模型，特别是在大型基准测试上表现更佳。球面几何先验的有效性得到验证，几何约束作为一种强大的正则化器，导致所有关系类型的性能全面提升。球面几何创造了'内在困难负采样'环境，自然消除平凡负样本，迫使模型学习更强大和语义一致的表示。

### 结论

流形的选择不仅仅是实现细节，而是基本设计原则。倡议将几何先验作为设计下一代强大且稳定的KGE模型的基础。

### 翻译

知识图谱嵌入(KGE)已成为多关系数据表示学习的基本技术。许多开创性模型，如TransE，在无界欧几里得空间中运行，这在建模复杂关系时存在固有局限性，并可能导致训练效率低下。在本文中，我们提出了球面知识图谱嵌入(SKGE)模型，通过将实体表示限制在紧凑流形(超球面)上，挑战了这一范式。SKGE使用可学习的非线性球化层将实体映射到球面上，并将关系解释为混合平移-投影变换。通过对三个基准数据集FB15k-237、CoDEx-S和CoDEx-M的广泛实验，我们证明SKGE一致且显著地优于其强大的欧几里得对应模型TransE，特别是在FB15k-237和CoDEx-M等大规模基准测试上，证明了球面几何先验的有效性。我们提供了深入分析以揭示这种优势的来源，表明这种几何约束作为一种强大的正则化器，导致所有关系类型的全面性能提升。更根本的是，我们证明了球面几何创造了'内在困难负采样'环境，自然消除平凡负样本，迫使模型学习更强大和语义一致的表示。我们的发现有力地证明了流形的选择不仅仅是实现细节，而是基本设计原则，倡导将几何先验作为设计下一代强大且稳定的KGE模型的基础。


### 论文摘要

Knowledge graph embedding (KGE) has become a fundamental technique for representation learning on multi-relational data. Many seminal models, such as TransE, operate in an unbounded Euclidean space, which presents inherent limitations in modeling complex relations and can lead to inefficient training. In this paper, we propose Spherical Knowledge Graph Embedding (SKGE), a model that challenges this paradigm by constraining entity representations to a compact manifold: a hypersphere. SKGE employs a learnable, non-linear Spherization Layer to map entities onto the sphere and interprets relations as a hybrid translate-then-project transformation. Through extensive experiments on three benchmark datasets, FB15k-237, CoDEx-S, and CoDEx-M, we demonstrate that SKGE consistently and significantly outperforms its strong Euclidean counterpart, TransE, particularly on large-scale benchmarks such as FB15k-237 and CoDEx-M, demonstrating the efficacy of the spherical geometric prior. We provide an in-depth analysis to reveal the sources of this advantage, showing that this geometric constraint acts as a powerful regularizer, leading to comprehensive performance gains across all relation types. More fundamentally, we prove that the spherical geometry creates an "inherently hard negative sampling" environment, naturally eliminating trivial negatives and forcing the model to learn more robust and semantically coherent representations. Our findings compellingly demonstrate that the choice of manifold is not merely an implementation detail but a fundamental design principle, advocating for geometric priors as a cornerstone for designing the next generation of powerful and stable KGE models.

---

## 14. Measuring the Intrinsic Dimension of Earth Representations

**论文链接:** [http://arxiv.org/abs/2511.02101v2](http://arxiv.org/abs/2511.02101v2)

**作者:** Arjun Rao, Marc Rußwurm, Konstantin Klemmer, Esther Rolf

**发布时间:** 2025-11-03

**备注:** Pre-print. 27 pages, 11 figures, 6 tables

### GPT解析

### 总结

本研究首次探讨了地理隐式神经表示(INRs)的内在维度特性，发现其内在维度在2到10之间，且与下游任务性能相关，为无监督评估模型提供了新方法。

### 背景

地理隐式神经表示(INRs)在地球观测学习中将低维位置输入(经度、纬度)嵌入到高维表示中，基于地理参考的卫星、图像或文本数据训练，但缺乏对其信息含量和分布的理解。

### 目的

探究地理INRs的内在维度，了解这些表示中包含多少信息以及信息集中在哪里，为模型评估提供新方法。

### 方法

分析内在维度在256到512之间的INRs，研究内在维度如何捕获数据局部变化，评估内在维度与下游任务性能的关系。

### 主要发现

地理INRs的内在维度大致在2到10之间；对INR预训练过程中变化的空间分辨率和输入模态敏感；与下游任务性能相关；能够捕获空间伪影，促进模型评估和诊断。

### 结论

提供了一种与架构无关、无标签的信息内容度量方法，可以在INRs之间实现无监督评估、模型选择和预训练设计。

### 翻译

在地球观测表征学习的背景下，地理隐式神经表示(INRs)将低维位置输入(经度、纬度)嵌入到高维嵌入中，这些模型基于地理参考的卫星、图像或文本数据训练。尽管地理INRs的共同目标是将地球数据提炼成紧凑、易于学习的表示，但我们缺乏对这些地球表示中包含多少信息以及这些信息集中在哪里理解。数据集的内在维度衡量了捕获其局部变化所需的自由度数量，无论它嵌入的高维空间如何。这项工作提供了对地理INRs内在维度的首次研究。分析ambent维度在256到512之间的INRs，我们发现它们的内在维度大致在2到10之间，并且对INR预训练过程中变化的空间分辨率和输入模态敏感。此外，我们表明地理INRs的内在维度与下游任务性能相关，并且可以捕获空间伪影，促进模型评估和诊断。更广泛地说，我们的工作提供了一种与架构无关、无标签的信息内容度量方法，可以在INRs之间实现无监督评估、模型选择和预训练设计。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何量化地理隐式神经表示(Geographic INRs)的信息含量及信息集中位置的问题。这个问题很重要，因为目前地球表示模型的质量主要通过特定下游任务的有监督性能评估，缺乏对模型基本表示能力的理解，而理解地球表示的信息含量对于评估和改进地理表示学习模型至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从地理INRs将低维位置输入映射到高维嵌入但实际信息含量可能远低于环境维度这一观察出发，借鉴了内在维度(ID)测量这一已有概念，设计了全局ID和局部ID两种测量方法。作者借鉴了基于距离的估计器(如MLE、TwoNN)和基于角度的估计器(如FisherS)等现有ID估计方法，以及SatCLIP、GeoCLIP等地理INRs的现有工作，将其应用于地球表示这一特定领域。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过测量地理INRs嵌入的内在维度(ID)来量化地球表示的信息丰富度和信息集中位置。整体流程包括：1)使用预训练位置编码器生成地理嵌入；2)通过角度估计器计算全局ID，通过基于距离的估计器计算局部ID生成ID地图；3)在下游任务激活空间中测量ID评估任务对齐；4)分析ID与下游性能的关系及分辨率和输入模态的影响；5)解释ID如何反映模型表示能力和任务对齐。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点包括：首次研究地理INRs的内在维度；提出ID作为无监督评估指标；区分表示性和任务对齐；揭示空间伪影；建立ID与下游性能的关系。相比之前工作，不同之处在于：评估方法从有监督转向无监督；评估深度从'学习友好性'深入到信息含量和表示能力；应用范围从特定任务扩展到一般INRs；增加了诊断模型空间异质性和伪影的能力。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文首次系统地测量了地理隐式神经表示的内在维度，发现其远低于环境维度但与下游任务性能相关，为地球表示学习提供了一种新的无监督评估方法。'}


### 论文摘要

Within the context of representation learning for Earth observation, geographic Implicit Neural Representations (INRs) embed low-dimensional location inputs (longitude, latitude) into high-dimensional embeddings, through models trained on geo-referenced satellite, image or text data. Despite the common aim of geographic INRs to distill Earth's data into compact, learning-friendly representations, we lack an understanding of how much information is contained in these Earth representations, and where that information is concentrated. The intrinsic dimension of a dataset measures the number of degrees of freedom required to capture its local variability, regardless of the ambient high-dimensional space in which it is embedded. This work provides the first study of the intrinsic dimensionality of geographic INRs. Analyzing INRs with ambient dimension between 256 and 512, we find that their intrinsic dimensions fall roughly between 2 and 10 and are sensitive to changing spatial resolution and input modalities during INR pre-training. Furthermore, we show that the intrinsic dimension of a geographic INR correlates with downstream task performance and can capture spatial artifacts, facilitating model evaluation and diagnostics. More broadly, our work offers an architecture-agnostic, label-free metric of information content that can enable unsupervised evaluation, model selection, and pre-training design across INRs.

---

## 15. Bridging Lifelong and Multi-Task Representation Learning via Algorithm and Complexity Measure

**论文链接:** [http://arxiv.org/abs/2511.01847v1](http://arxiv.org/abs/2511.01847v1)

**作者:** Zhi Wang, Chicheng Zhang, Ramya Korlakai Vinayak

**发布时间:** 2025-11-03

### GPT解析

### 总结

本文研究终身学习中的表示学习问题，提出了一种基于多任务经验风险最小化的算法，并基于任务回避维度建立了样本复杂度界限。

### 背景

终身学习中，学习者面临一系列具有共享结构的任务，需要识别并利用这些结构来加速学习。与多任务学习不同，终身学习要求学习者利用现有知识，同时在线方式持续收集部分信息。

### 目的

研究一个广义的终身表示学习框架，开发一种简单算法来处理在线方式下的持续学习，并建立样本复杂度界限。

### 方法

提出一种使用多任务经验风险最小化作为子程序的算法，并基于引入的任务回避维度概念建立样本复杂度界限。

### 主要发现

基于任务回避维度建立了样本复杂度界限，该结果适用于涉及一般函数类的广泛学习问题，并在噪声下的分类和回归任务中得到了具体应用。

### 结论

所提出的算法和理论框架为终身学习提供了有效的表示学习方法，适用于多种学习场景。

### 翻译

在终身学习中，学习者面临一系列具有共享结构的任务，旨在识别并利用这些结构来加速学习。我们研究了一种结构通过数据的共同表示来捕捉的场景。与多任务学习或学习-to-learn不同（在这些方法中任务一开始就可用以学习表示），终身学习要求学习者利用现有知识，同时在线方式持续收集部分信息。在本文中，我们考虑了一个广义的终身表示学习框架。我们提出了一种使用多任务经验风险最小化作为子程序的简单算法，并基于我们引入的新概念——任务回避维度，建立了样本复杂度界限。我们的结果适用于涉及一般函数类的广泛学习问题。作为具体例子，我们在噪声下的分类和回归任务中实例化了我们的结果。


### 论文摘要

In lifelong learning, a learner faces a sequence of tasks with shared structure and aims to identify and leverage it to accelerate learning. We study the setting where such structure is captured by a common representation of data. Unlike multi-task learning or learning-to-learn, where tasks are available upfront to learn the representation, lifelong learning requires the learner to make use of its existing knowledge while continually gathering partial information in an online fashion. In this paper, we consider a generalized framework of lifelong representation learning. We propose a simple algorithm that uses multi-task empirical risk minimization as a subroutine and establish a sample complexity bound based on a new notion we introduce--the task-eluder dimension. Our result applies to a wide range of learning problems involving general function classes. As concrete examples, we instantiate our result on classification and regression tasks under noise.

---

## 16. From Pixels to Cooperation Multi Agent Reinforcement Learning based on Multimodal World Models

**论文链接:** [http://arxiv.org/abs/2511.01310v1](http://arxiv.org/abs/2511.01310v1)

**作者:** Sureyya Akin, Kavita Srivastava, Prateek B. Kapoor, Pradeep G. Sethi, Sunita Q. Patel, Rahu Srivastava

**发布时间:** 2025-11-03

### GPT解析

### 总结

本文提出了一种基于共享生成多模态世界模型（MWM）的新框架，用于解决从高维多模态感官输入学习合作多智能体策略的样本效率问题。

### 背景

从高维、多模态感官输入（如像素和音频）直接学习合作多智能体策略存在样本效率低的问题。无模型多智能体强化学习算法面临表示学习、部分可观察性和信用分配的联合挑战。

### 目的

提出一种基于共享生成多模态世界模型（MWM）的新框架，提高合作多智能体强化学习的样本效率，解决表示学习、部分可观察性和信用分配的联合挑战。

### 方法

提出多模态世界模型（MWM），使用基于注意力的机制融合所有智能体的分布式多模态观测，学习环境动力学的压缩潜在表示；利用MWM作为快速'想象'模拟器，在潜在空间内训练合作MARL策略（如MAPPO），将表示学习与策略学习解耦；引入基于3D物理模拟器的多模态、多智能体基准测试集。

### 主要发现

MWM-MARL框架与最先进的无模型MARL基线相比实现了数量级更高的样本效率；在感觉不对称环境中，所提出的多模态融合对任务成功至关重要；架构对传感器脱落具有更好的鲁棒性，这对实际部署至关重要。

### 结论

基于共享生成多模态世界模型的框架能够有效解决从高维多模态输入学习合作多智能体策略的样本效率问题，并在感觉不对称环境和传感器脱落情况下表现出优越的鲁棒性。

### 翻译

从高维、多模态感官输入（如像素和音频）直接学习合作多智能体策略存在众所周知的样本效率低下问题。无模型多智能体强化学习算法难以应对表示学习、部分可观察性和信用分配的联合挑战。为此，我们提出了一种基于共享生成式多模态世界模型（MWM）的新框架。我们的MWM通过使用可扩展的基于注意力的机制融合所有智能体的分布式多模态观测，学习环境动力学的压缩潜在表示。随后，我们利用这个学习的MWM作为快速的'想象'模拟器，在其潜在空间内完全训练合作MARL策略（如MAPPO），将表示学习与策略学习解耦。我们引入了一组基于3D物理模拟器的新挑战性多模态、多智能体基准测试。我们的实验证明，与最先进的无模型MARL基线相比，我们的MWM-MARL框架实现了数量级更高的样本效率。我们进一步表明，在感觉不对称的环境中，我们提出的多模态融合对任务成功至关重要，并且我们的架构对传感器脱落提供了更强的鲁棒性，这是实际部署的关键特性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从高维多模态感官输入(如像素和音频)直接学习合作多智能体策略时样本效率低下的问题。这个问题在现实中很重要，因为它关系到开发能在物理世界协作的智能体系统(如机器人团队、自动驾驶车辆等)；在研究中也很重要，因为现有的模型无关MARL算法难以同时处理高维感官输入、部分可观察环境和复杂社会推理的联合挑战，限制了智能体系统在复杂环境中的实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先认识到模型无关MARL算法在高维多模态输入上的样本效率低下问题，然后借鉴了单智能体'World Models'范式的成功经验，意识到需要将其扩展到多智能体多模态环境。作者还考虑了计算效率和可扩展性挑战。该方法借鉴了多个领域的工作：单智能体世界模型(如Dreamer系列)、MARL的价值分解方法和CTDE范式、多模态学习(如CLIP、Vision Transformer)以及高效大规模系统(如MoE架构、联邦学习原则)。作者设计了一个基于共享多模态世界模型(MWM)的框架，使用注意力机制融合多智能体信息，并在潜在空间中训练策略。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是：1)学习一个共享的多模态世界模型(MWM)来统一表示环境；2)将表示学习与策略学习解耦，先学习环境模型再在潜在空间中训练策略；3)使用分层注意力机制动态融合多智能体多模态信息；4)利用MWM作为'想象'模拟器在潜在空间中训练策略。整体流程分为两个阶段：第一阶段学习MWM，包括多模态观察编码、分层融合、循环潜在动力学建模和训练；第二阶段在潜在空间中训练合作策略，包括使用MWM生成'梦想'轨迹、计算优势并更新策略、最终实现分散执行。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将单智能体世界模型扩展到多智能体多模态环境；2)提出基于注意力的分层多模态融合机制，实现动态信息路由；3)在潜在空间中完全训练合作MARL策略，解耦表示学习与策略学习；4)基于RSSM的概率性质提高对传感器故障的鲁棒性；5)引入新的多智能体多模态基准测试。相比之前工作，本文不仅扩展了世界模型到多智能体环境，还引入了多模态融合机制；与传统MARL不同，不是直接从原始数据学习而是先学习环境模型；相比多模态学习，不仅学习表示还学习环境动力学；相比联邦学习，将高效通信原则应用于多智能体协作场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种基于多模态世界模型的多智能体强化学习框架，通过解耦表示学习与策略学习，实现了从高维多模态感官输入直接学习合作策略的样本效率大幅提升，并显著提高了对传感器故障的鲁棒性。'}


### 论文摘要

Learning cooperative multi-agent policies directly from high-dimensional, multimodal sensory inputs like pixels and audio (from pixels) is notoriously sample-inefficient. Model-free Multi-Agent Reinforcement Learning (MARL) algorithms struggle with the joint challenge of representation learning, partial observability, and credit assignment. To address this, we propose a novel framework based on a shared, generative Multimodal World Model (MWM). Our MWM is trained to learn a compressed latent representation of the environment's dynamics by fusing distributed, multimodal observations from all agents using a scalable attention-based mechanism. Subsequently, we leverage this learned MWM as a fast, "imagined" simulator to train cooperative MARL policies (e.g., MAPPO) entirely within its latent space, decoupling representation learning from policy learning. We introduce a new set of challenging multimodal, multi-agent benchmarks built on a 3D physics simulator. Our experiments demonstrate that our MWM-MARL framework achieves orders-of-magnitude greater sample efficiency compared to state-of-the-art model-free MARL baselines. We further show that our proposed multimodal fusion is essential for task success in environments with sensory asymmetry and that our architecture provides superior robustness to sensor-dropout, a critical feature for real-world deployment.

---

## 17. Influence-aware Causal Autoencoder Network for Node Importance Ranking in Complex Networks

**论文链接:** [http://arxiv.org/abs/2511.01228v1](http://arxiv.org/abs/2511.01228v1)

**作者:** Jiahui Gao, Kuang Zhou, Yuchen Zhu

**发布时间:** 2025-11-03

### GPT解析

### 总结

本文提出了一种名为ICAN（影响感知因果自编码网络）的新框架，通过因果表示学习获取稳健、不变的节点嵌入，用于跨网络排序任务。ICAN在合成网络上训练，但能有效应用于真实网络，解决了现有方法依赖目标网络拓扑结构导致的隐私问题和泛化能力差的问题。

### 背景

节点重要性排序是图数据分析中的基础问题。现有方法通常依赖于传统中心性度量或先进图表示学习方法，这些方法直接依赖目标网络的拓扑结构，引发隐私问题，且在不同网络间泛化能力差。

### 目的

设计一个仅在合成网络上训练，但能有效应用于真实网络的节点重要性排序模型，消除对目标网络拓扑的依赖，提高实用性和泛化能力。

### 方法

提出影响感知因果自编码网络（ICAN），在自编码器架构中引入影响感知因果表示学习模块，提取与节点重要性因果相关的节点嵌入；引入因果排序损失，设计统一优化框架，联合优化重建和排序目标，实现节点表示学习和排序优化的相互强化。

### 主要发现

ICAN在合成网络上训练，能有效泛化到多样化的真实图；在多个基准数据集上的实验表明，ICAN在排序准确性和泛化能力方面持续优于最先进的基线方法。

### 结论

ICAN成功解决了在合成网络上训练并应用于真实网络的问题，通过因果表示学习提高了模型的泛化能力和实用性。

### 翻译

节点重要性排序是图数据分析中的一个基础问题。现有方法通常依赖于从传统中心性度量或先进的图表示学习方法中推导出的节点特征，这些方法直接依赖于目标网络的拓扑结构。然而，这种对结构信息的依赖引发了隐私问题，并且通常在不同网络间的泛化能力较差。在这项工作中，我们解决了一个关键问题：我们能否设计一个仅在合成网络上训练的节点重要性排序模型，并有效应用于真实网络，消除对目标网络拓扑的依赖，同时提高实用性和泛化能力？我们通过提出影响感知因果自编码网络（ICAN）对此问题给予了肯定的回答，这是一个利用因果表示学习获取稳健、不变的节点嵌入的新框架，用于跨网络排序任务。首先，ICAN在自编码器架构中引入了影响感知因果表示学习模块，提取与节点重要性因果相关的节点嵌入。此外，我们引入了因果排序损失，并设计了一个统一优化框架，联合优化重建和排序目标，使节点表示学习和排序优化能够相互强化。这种设计使得ICAN在合成网络上训练后，能够有效泛化到多样化的真实图。在多个基准数据集上的广泛实验表明，ICAN在排序准确性和泛化能力方面持续优于最先进的基线方法。


### 论文摘要

Node importance ranking is a fundamental problem in graph data analysis. Existing approaches typically rely on node features derived from either traditional centrality measures or advanced graph representation learning methods, which depend directly on the target network's topology. However, this reliance on structural information raises privacy concerns and often leads to poor generalization across different networks. In this work, we address a key question: Can we design a node importance ranking model trained exclusively on synthetic networks that is effectively appliable to real-world networks, eliminating the need to rely on the topology of target networks and improving both practicality and generalizability? We answer this question affirmatively by proposing the Influence-aware Causal Autoencoder Network (ICAN), a novel framework that leverages causal representation learning to get robust, invariant node embeddings for cross-network ranking tasks. Firstly, ICAN introduces an influence-aware causal representation learning module within an autoencoder architecture to extract node embeddings that are causally related to node importance. Moreover, we introduce a causal ranking loss and design a unified optimization framework that jointly optimizes the reconstruction and ranking objectives, enabling mutual reinforcement between node representation learning and ranking optimization. This design allows ICAN, trained on synthetic networks, to generalize effectively across diverse real-world graphs. Extensive experiments on multiple benchmark datasets demonstrate that ICAN consistently outperforms state-of-the-art baselines in terms of both ranking accuracy and generalization capability.

---

## 18. Anatomically Constrained Transformers for Echocardiogram Analysis

**论文链接:** [http://arxiv.org/abs/2511.01109v1](http://arxiv.org/abs/2511.01109v1)

**作者:** Alexander Thorley, Agis Chartsias, Jordan Strom, Jeremy Slivnick, Dipak Kotecha, Alberto Gomez, Jinming Duan

**发布时间:** 2025-11-02

### GPT解析

### 总结

本文提出了Video Anatomically Constrained Transformer (ViACT)，一种将解剖先验直接整合到变换器架构中的新框架，用于超声心动图分析。

### 背景

视频变换器在超声心动图分析中显示出强大潜力，但与其他视频模型一样，它们容易从非诊断区域（如图像背景）学习到虚假相关性。

### 目的

克服现有视频变换器模型的局限性，通过整合解剖先验来提高模型在医学图像分析中的表现和可解释性。

### 方法

ViACT将变形的解剖结构表示为点集，将其空间几何和相应的图像块编码为变换器token；在预训练过程中采用掩码自编码策略，仅掩码和重建解剖区域；预训练模型可针对该区域的任务进行微调；专注于心肌应用，并在左心室射血分数回归和心脏淀粉样变性检测等任务上展示框架。

### 主要发现

解剖约束将变换器的注意力集中在心肌区域，产生与已知病理区域对齐的可解释注意力图；ViACT能够推广到心肌点跟踪，无需专门跟踪网络中的特定任务组件。

### 结论

ViACT通过整合解剖先验，解决了视频变换器在医学图像分析中学习非相关区域的问题，提高了模型在超声心动图分析任务中的性能和可解释性。

### 翻译

视频变换器最近在超声心动图分析中显示出强大的潜力，利用自监督预训练和灵活适应不同任务的能力。然而，与其他视频模型一样，它们容易从非诊断区域（如图像背景）学习到虚假相关性。为克服这一局限，我们提出了视频解剖约束变换器（ViACT），一种将解剖先验直接整合到变换器架构的新框架。ViACT将变形的解剖结构表示为点集，并将其空间几何和相应的图像块编码为变换器token。在预训练过程中，ViACT遵循掩码自编码策略，仅掩码和重建解剖区域，强制表示学习集中在解剖区域。预训练模型随后可以针对该区域的任务进行微调。本文中我们专注于心肌，在左心室射血分数回归和心脏淀粉样变性检测等超声分析任务上展示了该框架。解剖约束将变换器的注意力集中在心肌区域，产生与已知CA病理区域对齐的可解释注意力图。此外，ViACT能够推广到心肌点跟踪，而无需专门跟踪网络中使用的特定任务组件，如相关体积。


### 论文摘要

Video transformers have recently demonstrated strong potential for echocardiogram (echo) analysis, leveraging self-supervised pre-training and flexible adaptation across diverse tasks. However, like other models operating on videos, they are prone to learning spurious correlations from non-diagnostic regions such as image backgrounds. To overcome this limitation, we propose the Video Anatomically Constrained Transformer (ViACT), a novel framework that integrates anatomical priors directly into the transformer architecture. ViACT represents a deforming anatomical structure as a point set and encodes both its spatial geometry and corresponding image patches into transformer tokens. During pre-training, ViACT follows a masked autoencoding strategy that masks and reconstructs only anatomical patches, enforcing that representation learning is focused on the anatomical region. The pre-trained model can then be fine-tuned for tasks localized to this region. In this work we focus on the myocardium, demonstrating the framework on echo analysis tasks such as left ventricular ejection fraction (EF) regression and cardiac amyloidosis (CA) detection. The anatomical constraint focuses transformer attention within the myocardium, yielding interpretable attention maps aligned with regions of known CA pathology. Moreover, ViACT generalizes to myocardium point tracking without requiring task-specific components such as correlation volumes used in specialized tracking networks.

---

## 19. The Geometry of Grokking: Norm Minimization on the Zero-Loss Manifold

**论文链接:** [http://arxiv.org/abs/2511.01938v1](http://arxiv.org/abs/2511.01938v1)

**作者:** Tiberiu Musat

**发布时间:** 2025-11-02

### GPT解析

### 总结

论文研究了神经网络中的'Grokking'现象，即神经网络在完全记忆训练数据后，经过显著延迟才出现完全泛化的现象。论文提出通过约束优化视角理解记忆后的学习过程，证明梯度下降实际上是在零损失流形上最小化权重范数。

### 背景

先前研究将延迟泛化归因于权重衰减驱动的表示学习，但精确的潜在动态机制仍然不清楚。

### 目的

通过约束优化的视角理解神经网络记忆后的学习过程。

### 方法

论文在无限小学习率和权重衰减系数的极限下形式化证明梯度下降在零损失流形上最小化权重范数；引入近似方法解耦参数学习动态；推导两层网络第一层记忆后动力学的闭式表达式。

### 主要发现

梯度下降在零损失流形上最小化权重范数；通过近似方法可解耦参数学习动态；推导出两层网络第一层记忆后动力学的闭式表达式。

### 结论

实验证实使用预测梯度模拟训练过程能重现 grokking 的延迟泛化和表示学习特征。

### 翻译

Grokking 是神经网络中的一种令人费解的现象，其中完全泛化仅在完全记忆训练数据后经过显著延迟才发生。先前的研究将这种延迟泛化与权重衰减驱动的表示学习联系起来，但潜在的精确动态机制仍然不清楚。在本文中，我们认为记忆后的学习可以通过约束优化的视角来理解：梯度下降有效地在零损失流形上最小化权重范数。我们在学习率和权重衰减系数无限小的极限情况下形式化证明了这一点。为了进一步剖析这一机制，我们引入了一个近似方法，将网络中一部分参数的学习动态与其他参数解耦。应用这一框架，我们推导出两层网络第一层记忆后动力学的闭式表达式。实验证实，使用我们预测的梯度模拟训练过程能够重现 grokking 的延迟泛化和表示学习特征。


### 论文摘要

Grokking is a puzzling phenomenon in neural networks where full generalization occurs only after a substantial delay following the complete memorization of the training data. Previous research has linked this delayed generalization to representation learning driven by weight decay, but the precise underlying dynamics remain elusive. In this paper, we argue that post-memorization learning can be understood through the lens of constrained optimization: gradient descent effectively minimizes the weight norm on the zero-loss manifold. We formally prove this in the limit of infinitesimally small learning rates and weight decay coefficients. To further dissect this regime, we introduce an approximation that decouples the learning dynamics of a subset of parameters from the rest of the network. Applying this framework, we derive a closed-form expression for the post-memorization dynamics of the first layer in a two-layer network. Experiments confirm that simulating the training process using our predicted gradients reproduces both the delayed generalization and representation learning characteristic of grokking.

---

## 20. TRISKELION-1: Unified Descriptive-Predictive-Generative AI

**论文链接:** [http://arxiv.org/abs/2511.00711v1](http://arxiv.org/abs/2511.00711v1)

**作者:** Nardeep Kumar, Arun Kanwar

**发布时间:** 2025-11-01

**备注:** 12 pages, 18 figures, submitted to arXiv (2025)

### GPT解析

### 总结

TRISKELION-1是一种统一架构，在一个编码器-解码器框架中集成了描述性、预测性和生成性功能，能够同时实现描述性重建、预测分类和生成采样。

### 背景

当前人工智能领域需要能够同时处理描述性、预测性和生成性任务的统一架构。

### 目的

开发一个能够联合优化描述性表示学习、预测推理和生成合成的统一模型框架。

### 方法

TRISKELION-1架构在一个编码器-解码器框架中集成了统计、机制和生成推理，使用变分目标进行联合优化。

### 主要发现

在MNIST数据集上的实验表明，描述性重建、预测分类和生成采样可以在一个模型中稳定共存。

### 结论

该框架为连接可解释性、准确性和创造力的通用智能架构提供了蓝图。

### 翻译

TRISKELION-1是一种统一的描述性-预测性-生成性架构，它在单个编码器-解码器框架中集成了统计、机制和生成推理。该模型展示了如何使用变分目标联合优化描述性表示学习、预测推理和生成合成。在MNIST上的实验验证了描述性重建、预测分类和生成采样可以在一个模型中稳定共存。该框架为连接可解释性、准确性和创造力的通用智能架构提供了蓝图。


### 论文摘要

TRISKELION-1 is a unified descriptive-predictive-generative architecture that integrates statistical, mechanistic, and generative reasoning within a single encoder-decoder framework. The model demonstrates how descriptive representation learning, predictive inference, and generative synthesis can be jointly optimized using variational objectives. Experiments on MNIST validate that descriptive reconstruction, predictive classification, and generative sampling can coexist stably within one model. The framework provides a blueprint toward universal intelligence architectures that connect interpretability, accuracy, and creativity.

---

## 21. Improving Robustness to Out-of-Distribution States in Imitation Learning via Deep Koopman-Boosted Diffusion Policy

**论文链接:** [http://arxiv.org/abs/2511.00555v1](http://arxiv.org/abs/2511.00555v1)

**作者:** Dianye Huang, Nassir Navab, Zhongliang Jiang

**发布时间:** 2025-11-01

**备注:** Accepted by IEEE T-RO

### GPT解析

### 总结

本文提出了一种名为DeepKoopman增强的双分支扩散策略(D3P)的算法，用于解决机器人操作模仿学习中现有扩散模型难以捕捉多步骤间强时间依赖性的问题，特别是在整合本体感受输入时。D3P通过双分支架构解耦不同感官模态的作用，并利用Deep Koopman Operator模块增强视觉表示学习，显著提高了任务执行效果。

### 背景

将生成模型与动作块结合在机器人操作模仿学习中显示出巨大潜力，但现有的基于扩散的方法往往难以捕捉多步骤间的强时间依赖性，特别是在整合本体感受输入时，这会导致任务失败，策略过度适应本体感受线索而忽略视觉特征。

### 目的

克服现有扩散模型在捕捉多步骤间强时间依赖性方面的局限性，特别是当整合本体感受输入时，防止策略过度适应本体感受线索而忽略视觉特征，从而提高机器人操作任务的执行效果。

### 方法

提出D3P算法，采用双分支架构解耦不同感官模态组合的作用：视觉分支编码视觉观察以指示任务进展，融合分支整合视觉和本体感受输入实现精确操作。当机器人无法完成中间目标时，策略可动态切换到视觉分支生成的动作块。同时，集成Deep Koopman Operator模块捕捉视觉输入中的结构化时间动态，并利用测试时损失作为置信信号引导预测动作块的聚合。

### 主要发现

在六个RLBench桌面任务的模拟实验中，D3P比最先进的扩散策略平均高出14.6%。在三个真实世界机器人操作任务中，实现了15.0%的改进。代码已公开在GitHub上。

### 结论

D3P算法通过双分支架构和Deep Koopman Operator模块有效解决了现有扩散模型在捕捉时间依赖性和整合多感官输入方面的局限性，显著提高了机器人操作任务的执行效果和可靠性。

### 翻译

将生成模型与动作块整合在机器人操作模仿学习中显示出巨大潜力。然而，现有的基于扩散的范式往往难以捕捉多步骤间的强时间依赖性，特别是在整合本体感受输入时。这种局限性会导致任务失败，策略过度适应本体感受线索而牺牲对任务中视觉派生特征的捕捉。为克服这一挑战，我们提出了DeepKoopman增强的双分支扩散策略(D3P)算法。D3P引入双分支架构来解耦不同感官模态组合的作用。视觉分支编码视觉观察以指示任务进展，而融合分支整合视觉和本体感受输入以实现精确操作。在此架构中，当机器人无法完成中间目标（如抓取抽屉把手）时，策略可动态切换到执行由视觉分支生成的动作块，允许恢复到先前观察的状态并促进任务重试。为进一步增强视觉表示学习，我们集成了Deep Koopman Operator模块，从视觉输入中捕捉结构化时间动态。在推理过程中，我们使用生成模型的测试时损失作为置信信号来指导时间重叠预测动作块的聚合，从而提高策略执行的可靠性。在六个RLBench桌面任务的模拟实验中，D3P比最先进的扩散策略平均高出14.6%。在三个真实世界机器人操作任务中，实现了15.0%的改进。代码：https://github.com/dianyeHuang/D3P。


### 论文摘要

Integrating generative models with action chunking has shown significant promise in imitation learning for robotic manipulation. However, the existing diffusion-based paradigm often struggles to capture strong temporal dependencies across multiple steps, particularly when incorporating proprioceptive input. This limitation can lead to task failures, where the policy overfits to proprioceptive cues at the expense of capturing the visually derived features of the task. To overcome this challenge, we propose the Deep Koopman-boosted Dual-branch Diffusion Policy (D3P) algorithm. D3P introduces a dual-branch architecture to decouple the roles of different sensory modality combinations. The visual branch encodes the visual observations to indicate task progression, while the fused branch integrates both visual and proprioceptive inputs for precise manipulation. Within this architecture, when the robot fails to accomplish intermediate goals, such as grasping a drawer handle, the policy can dynamically switch to execute action chunks generated by the visual branch, allowing recovery to previously observed states and facilitating retrial of the task. To further enhance visual representation learning, we incorporate a Deep Koopman Operator module that captures structured temporal dynamics from visual inputs. During inference, we use the test-time loss of the generative model as a confidence signal to guide the aggregation of the temporally overlapping predicted action chunks, thereby enhancing the reliability of policy execution. In simulation experiments across six RLBench tabletop tasks, D3P outperforms the state-of-the-art diffusion policy by an average of 14.6\%. On three real-world robotic manipulation tasks, it achieves a 15.0\% improvement. Code: https://github.com/dianyeHuang/D3P.

---

## 22. FedMGP: Personalized Federated Learning with Multi-Group Text-Visual Prompts

**论文链接:** [http://arxiv.org/abs/2511.00480v1](http://arxiv.org/abs/2511.00480v1)

**作者:** Weihao Bo, Yanpeng Sun, Yu Wang, Xinyu Zhang, Zechao Li

**发布时间:** 2025-11-01

### GPT解析

### 总结

FedMGP是一种用于视觉语言模型中个性化联邦提示学习的新范式，通过多组配对的文本和视觉提示捕捉多样化语义，采用动态提示聚合策略平衡全局与本地特征，实现高效且高性能的联邦学习。

### 背景

在联邦学习环境中，视觉语言模型需要同时考虑全局知识的共享和客户端个性化特征的保留，而传统的联邦提示学习方法在平衡这两方面存在挑战。

### 目的

开发一种新的联邦提示学习范式，能够有效捕捉多样化、细粒度的语义和实例级线索，平衡通用知识保留与客户端特定特征，同时保持参数效率。

### 方法

FedMGP为每个客户端配备多组配对的文本和视觉提示，引入多样性损失促使各组提示专注于不同语义方面，采用基于相似度引导概率采样的动态提示聚合策略进行通信，通过softmax加权分布选择最相关的提示组。

### 主要发现

动态聚合策略通过强化共享语义同时抑制客户端特定噪声，促进鲁棒的全局表征学习；FedMGP在所有联邦提示学习方法中实现了最低的通信参数，同时达到最先进的性能。

### 结论

FedMGP在多样化的联邦视觉语言基准测试中，在个性化和领域泛化方面均始终优于先前的方法，为联邦视觉语言模型提供了一种高效且有效的个性化学习框架。

### 翻译

本文介绍了FedMGP，一种用于视觉语言模型中个性化联邦提示学习的新范式。FedMGP为每个客户端配备多组配对的文本和视觉提示，使模型能够捕捉多样化、细粒度的语义和实例级线索。引入了多样性损失促使每组提示专注于不同且互补的语义方面，确保各组共同覆盖更广泛的本地特征。在通信过程中，FedMGP采用基于相似度引导概率采样的动态提示聚合策略：每个客户端计算其提示组与上一轮全局提示之间的余弦相似度，然后通过softmax加权分布采样s个组。这种软选择机制优先聚合语义对齐的知识，同时有效探索代表性不足的模式，平衡了通用知识的保留与客户端特定特征。值得注意的是，FedMGP通过在多个组之间重新分配固定的提示容量，在所有联邦提示学习方法中实现了最低的通信参数，同时达到最先进的性能。理论分析表明，我们的动态聚合策略通过强化共享语义同时抑制客户端特定噪声，促进鲁棒的全局表征学习。大量实验证明，在多样化的联邦视觉语言基准测试中，FedMGP在个性化和领域泛化方面均始终优于先前的方法。代码将在https://github.com/weihao-bo/FedMGP.git上发布。


### 论文摘要

In this paper, we introduce FedMGP, a new paradigm for personalized federated prompt learning in vision-language models. FedMGP equips each client with multiple groups of paired textual and visual prompts, enabling the model to capture diverse, fine-grained semantic and instance-level cues. A diversity loss is introduced to drive each prompt group to specialize in distinct and complementary semantic aspects, ensuring that the groups collectively cover a broader range of local characteristics. During communication, FedMGP employs a dynamic prompt aggregation strategy based on similarity-guided probabilistic sampling: each client computes the cosine similarity between its prompt groups and the global prompts from the previous round, then samples s groups via a softmax-weighted distribution. This soft selection mechanism preferentially aggregates semantically aligned knowledge while still enabling exploration of underrepresented patterns effectively balancing the preservation of common knowledge with client-specific features. Notably, FedMGP maintains parameter efficiency by redistributing a fixed prompt capacity across multiple groups, achieving state-of-the-art performance with the lowest communication parameters among all federated prompt learning methods. Theoretical analysis shows that our dynamic aggregation strategy promotes robust global representation learning by reinforcing shared semantics while suppressing client-specific noise. Extensive experiments demonstrate that FedMGP consistently outperforms prior approaches in both personalization and domain generalization across diverse federated vision-language benchmarks. The code will be released on https://github.com/weihao-bo/FedMGP.git.

---

## 23. SHIELD: Securing Healthcare IoT with Efficient Machine Learning Techniques for Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.03661v1](http://arxiv.org/abs/2511.03661v1)

**作者:** Mahek Desai, Apoorva Rumale, Marjan Asadinia

**发布时间:** 2025-11-05

**DOI:** 10.1109/AIIoT65859.2025.11105287

### GPT解析

### 总结

该研究提出了一种机器学习驱动的框架，用于检测物联网医疗环境中的恶意网络攻击和设备异常，通过评估多种机器学习模型，确定了最优解决方案并提升了医疗设备安全性。

### 背景

物联网设备在医疗领域的整合引入了显著的安全性和可靠性挑战，增加了系统对网络威胁和操作异常的易感性。

### 目的

开发一种机器学习框架，用于检测恶意网络攻击和识别故障设备异常，从而提高物联网医疗环境的安全性和可靠性。

### 方法

利用包含20万条记录的数据集，评估了八种机器学习模型，包括监督学习(XGBoost、K-NN)、半监督学习(GAN、VAE)和无监督学习(One-Class SVM、Isolation Forest、GNN、LSTM自编码器)方法，通过F1分数、精确率、召回率、准确率、ROC-AUC和计算效率等指标进行综合评估。

### 主要发现

异常检测方面，XGBoost表现最优(99%准确率，0.04秒计算时间)，隔离森林有效平衡了精确率和召回率，LSTM自编码器表现较差；攻击检测方面，KNN实现接近完美的指标且计算成本最低(0.05秒)，VAE准确率达97%，GAN计算成本最高但准确率和ROC-AUC最低。

### 结论

该框架通过有效的异常检测策略增强了物联网医疗安全性，能够早期检测网络威胁和设备故障，防止数据泄露，最小化系统停机时间，确保医疗设备持续安全运行，最终保护患者健康和对物联网医疗解决方案的信任。

### 翻译

物联网设备在医疗领域的整合引入了显著的安全性和可靠性挑战，增加了对网络威胁和操作异常的易感性。本研究提出了一种机器学习驱动的框架，用于(1)检测恶意网络攻击和(2)识别故障设备异常，利用包含20万条记录的数据集。评估了八种机器学习模型，涵盖三种学习方法：监督学习(XGBoost、K-近邻)、半监督学习(生成对抗网络、变分自编码器)和无监督学习(一类支持向量机、隔离森林、图神经网络和长短期记忆自编码器)。通过F1分数、精确率、召回率、准确率、ROC-AUC和计算效率等多个指标进行综合评估。XGBoost以99%的准确率和最小的计算开销(0.04秒)实现了异常检测，隔离森林有效地平衡了精确率和召回率。LSTM自编码器表现较差，准确率较低且延迟较高。在攻击检测方面，KNN以最低的计算成本(0.05秒)实现了接近完美的精确率、召回率和F1分数，其次是VAE，准确率为97%。GAN显示出最高的计算成本，但准确率和ROC-AUC最低。这些发现通过有效的异常检测策略增强了物联网医疗安全性。通过提高网络威胁和设备故障的早期检测，该框架有可能防止数据泄露，最小化系统停机时间，并确保医疗设备的持续安全运行，最终保护患者健康和对物联网驱动医疗解决方案的信任。


### 论文摘要

The integration of IoT devices in healthcare introduces significant security and reliability challenges, increasing susceptibility to cyber threats and operational anomalies. This study proposes a machine learning-driven framework for (1) detecting malicious cyberattacks and (2) identifying faulty device anomalies, leveraging a dataset of 200,000 records. Eight machine learning models are evaluated across three learning approaches: supervised learning (XGBoost, K-Nearest Neighbors (K- NN)), semi-supervised learning (Generative Adversarial Networks (GAN), Variational Autoencoders (VAE)), and unsupervised learning (One-Class Support Vector Machine (SVM), Isolation Forest, Graph Neural Networks (GNN), and Long Short-Term Memory (LSTM) Autoencoders). The comprehensive evaluation was conducted across multiple metrics like F1-score, precision, recall, accuracy, ROC-AUC, computational efficiency. XGBoost achieved 99\% accuracy with minimal computational overhead (0.04s) for anomaly detection, while Isolation Forest balanced precision and recall effectively. LSTM Autoencoders underperformed with lower accuracy and higher latency. For attack detection, KNN achieved near-perfect precision, recall, and F1-score with the lowest computational cost (0.05s), followed by VAE at 97% accuracy. GAN showed the highest computational cost with lowest accuracy and ROC-AUC. These findings enhance IoT-enabled healthcare security through effective anomaly detection strategies. By improving early detection of cyber threats and device failures, this framework has the potential to prevent data breaches, minimize system downtime, and ensure the continuous and safe operation of medical devices, ultimately safeguarding patient health and trust in IoT-driven healthcare solutions.

---

## 24. NABench: Large-Scale Benchmarks of Nucleotide Foundation Models for Fitness Prediction

**论文链接:** [http://arxiv.org/abs/2511.02888v1](http://arxiv.org/abs/2511.02888v1)

**作者:** Zhongmin Li, Runze Ma, Jiahao Tan, Chengzi Tan, Shuangjia Zheng

**发布时间:** 2025-11-04

### GPT解析

### 总结

NABench是一个用于核酸适应性预测的大规模、系统化基准测试平台，汇集了162个高通量测定和260万个突变序列，涵盖多种DNA和RNA家族。研究团队评估了29个代表性基础模型在不同场景下的表现，建立了强大且可重现的基准线，并公开了代码以促进核酸建模和相关应用。

### 背景

核苷酸序列变异可能导致功能适应性的显著变化。最近的核苷酸基础模型可以直接从序列预测这些适应性效应，但数据集的不一致性和预处理的不统一使得难以公平地比较不同DNA和RNA家族的方法。

### 目的

介绍NABench，一个用于核酸适应性预测的大规模、系统化基准测试平台，以解决现有方法比较困难的问题。

### 方法

NABench汇集了162个高通量测定，整理了260万个突变序列，涵盖了多种DNA和RNA家族，并具有标准化的分割和丰富的元数据。研究者在统一的评估套件下，严格评估了29个代表性基础模型，包括零样本预测、少样本预测、迁移学习和监督设置等场景。

### 主要发现

NABench在规模、多样性和数据质量上超越了先前的核苷酸适应性基准测试；评估结果量化了不同任务和核酸类型之间的性能异质性；展示了不同建模选择的明显优势和失败模式；建立了强大且可重现的基准线。

### 结论

研究团队发布了NABench以促进核酸建模，支持RNA/DNA设计、合成生物学和生物化学等下游应用。代码已在GitHub上提供。

### 翻译

核苷酸序列变异可能引起功能适应性的显著变化。最近的核苷酸基础模型有望直接从序列预测这些适应性效应，然而异构数据集和不一致的预处理使得难以在DNA和RNA家族间公平地比较方法。在此，我们引入NABench，一个用于核酸适应性预测的大规模、系统化基准。NABench汇集了162个高通量测定，整理了260万个突变序列，涵盖多样化的DNA和RNA家族，具有标准化的分割和丰富的元数据。我们证明NABench在规模、多样性和数据质量上超越了先前的核苷酸适应性基准测试。在统一的评估套件下，我们严格评估了29个代表性基础模型，包括零样本预测、少样本预测、迁移学习和监督设置。结果量化了不同任务和核酸类型之间的性能异质性，展示了不同建模选择的明显优势和失败模式，并建立了强大、可重现的基准线。我们发布NABench以推进核酸建模，支持RNA/DNA设计、合成生物学和生物化学的下游应用。我们的代码可在https://github.com/mrzzmrzz/NABench获取。


### 论文摘要

Nucleotide sequence variation can induce significant shifts in functional fitness. Recent nucleotide foundation models promise to predict such fitness effects directly from sequence, yet heterogeneous datasets and inconsistent preprocessing make it difficult to compare methods fairly across DNA and RNA families. Here we introduce NABench, a large-scale, systematic benchmark for nucleic acid fitness prediction. NABench aggregates 162 high-throughput assays and curates 2.6 million mutated sequences spanning diverse DNA and RNA families, with standardized splits and rich metadata. We show that NABench surpasses prior nucleotide fitness benchmarks in scale, diversity, and data quality. Under a unified evaluation suite, we rigorously assess 29 representative foundation models across zero-shot, few-shot prediction, transfer learning, and supervised settings. The results quantify performance heterogeneity across tasks and nucleic-acid types, demonstrating clear strengths and failure modes for different modeling choices and establishing strong, reproducible baselines. We release NABench to advance nucleic acid modeling, supporting downstream applications in RNA/DNA design, synthetic biology, and biochemistry. Our code is available at https://github.com/mrzzmrzz/NABench.

---

## 25. Graph Neural AI with Temporal Dynamics for Comprehensive Anomaly Detection in Microservices

**论文链接:** [http://arxiv.org/abs/2511.03285v1](http://arxiv.org/abs/2511.03285v1)

**作者:** Qingyuan Zhang, Ning Lyu, Le Liu, Yuxi Wang, Ziyu Cheng, Cancan Hua

**发布时间:** 2025-11-05

### GPT解析

### 总结

本研究提出了一种结合图神经网络与时间建模的统一框架，用于微服务架构中的异常检测和根因追踪。该框架将微服务调用链抽象为有向图，通过图卷积聚合特征并建模依赖关系，同时使用门控循环单元建模时间演化，最终实现从局部异常检测到全局调用链追踪的统一建模。

### 背景

微服务架构中的异常检测和根因追踪问题

### 目的

提出一个统一框架来解决微服务架构中的异常检测和根因追踪问题

### 方法

结合图神经网络与时间建模的统一框架：将微服务调用链抽象为有向图，使用节点和边的多维特征构建服务拓扑表示，应用图卷积聚合节点特征并建模依赖关系，引入门控循环单元(GRU)建模调用链的时间演化，使用多层堆叠和连接操作联合获取结构和时间表示，定义节点和路径级别的异常评分函数

### 主要发现

所提出的框架在AUC、ACC、Recall和F1-Score等关键指标上优于基线方法，在动态拓扑和复杂环境下保持高准确性和稳定性

### 结论

该研究不仅为微服务异常检测提供了新的技术路径，也为分布式系统的智能运维奠定了方法论基础

### 翻译

本研究解决了微服务架构中的异常检测和根因追踪问题，并提出了一种结合图神经网络与时间建模的统一框架。微服务调用链被抽象为有向图，其中节点和边的多维特征用于构建服务拓扑表示，并应用图卷积来聚合节点特征并建模依赖关系，捕捉服务间的复杂结构关系。在此基础上，引入门控循环单元来建模调用链的时间演化，并使用多层堆叠和连接操作联合获取结构和时间表示，提高识别异常模式的能力。此外，定义了节点和路径级别的异常评分函数，实现从局部异常检测到全局调用链追踪的统一建模，从而能够识别异常服务节点并重建潜在的异常传播路径。随后从超参数、环境干扰和数据分布等多个维度设计了敏感性实验来评估该框架，结果表明其在AUC、ACC、Recall和F1-Score等关键指标上优于基线方法，在动态拓扑和复杂环境下保持高准确性和稳定性。这项研究不仅为微服务异常检测提供了新的技术路径，也为分布式系统的智能运维奠定了方法论基础。


### 论文摘要

This study addresses the problem of anomaly detection and root cause tracing in microservice architectures and proposes a unified framework that combines graph neural networks with temporal modeling. The microservice call chain is abstracted as a directed graph, where multidimensional features of nodes and edges are used to construct a service topology representation, and graph convolution is applied to aggregate features across nodes and model dependencies, capturing complex structural relationships among services. On this basis, gated recurrent units are introduced to model the temporal evolution of call chains, and multi-layer stacking and concatenation operations are used to jointly obtain structural and temporal representations, improving the ability to identify anomaly patterns. Furthermore, anomaly scoring functions at both the node and path levels are defined to achieve unified modeling from local anomaly detection to global call chain tracing, which enables the identification of abnormal service nodes and the reconstruction of potential anomaly propagation paths. Sensitivity experiments are then designed from multiple dimensions, including hyperparameters, environmental disturbances, and data distribution, to evaluate the framework, and results show that it outperforms baseline methods in key metrics such as AUC, ACC, Recall, and F1-Score, maintaining high accuracy and stability under dynamic topologies and complex environments. This research not only provides a new technical path for anomaly detection in microservices but also lays a methodological foundation for intelligent operations in distributed systems.

---

## 26. SurgAnt-ViVQA: Learning to Anticipate Surgical Events through GRU-Driven Temporal Cross-Attention

**论文链接:** [http://arxiv.org/abs/2511.03178v1](http://arxiv.org/abs/2511.03178v1)

**作者:** Shreyas C. Dhake, Jiayuan Huang, Runlong He, Danyal Z. Khan, Evangelos B. Mazomenos, Sophia Bano, Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarak I. Hoque

**发布时间:** 2025-11-05

**备注:** 12 pages

### GPT解析

### 总结

本研究提出了PitVQA-Anticipation数据集和SurgAnt-ViVQA模型，用于解决经鼻蝶垂体手术中的前瞻性手术推理问题。通过结合时间感知编码器和细粒度门控交叉注意力，系统能够从回顾性描述转向主动预测，为手术实时辅助提供支持。

### 背景

在经鼻蝶垂体手术中，视野有限且工作流程变化迅速，预测即将发生的手术事件对实时辅助至关重要。现有视觉问答系统基于孤立帧进行静态推理，对预测下一步或器械需求支持有限，且现有数据集关注当前场景而非近期未来。

### 目的

创建第一个用于前瞻性手术推理的VQA数据集，并提出一种能够预测手术未来阶段、步骤、器械需求和剩余时间的视频语言模型。

### 方法

构建了包含33.5小时手术视频和734,769个问答对的PitVQA-Anticipation数据集，涵盖四个预测任务。提出SurgAnt-ViVQA模型，使用双向GRU编码帧动态，通过自适应门将视觉上下文注入语言流，并采用参数高效微调定制语言主干。

### 主要发现

SurgAnt-ViVQA在PitVQA-Anticipation和EndoVis数据集上超越了现有基线。消融研究表明时间循环和门控 fusion带来主要性能提升。帧预算研究显示8帧最大化流畅性，32帧略微降低BLEU但改进数值时间估计。

### 结论

SurgAnt-ViVQA将手术VQA从回顾性描述推进到主动预测，PitVQA-Anticipation为此提供了全面基准，强调了针对时间建模对可靠、未来感知型手术辅助的重要性。

### 翻译

预测即将发生的手术事件对于经鼻蝶垂体手术中的实时辅助至关重要，在这些手术中视野有限且工作流程变化迅速。大多数视觉问答系统基于孤立帧进行推理，使用静态视觉语言对齐，对预测下一步或器械需求提供很少支持。现有的手术VQA数据集同样关注当前场景而非近期未来。我们引入了PitVQA-Anticipation，这是第一个用于前瞻性手术推理的VQA数据集。它包含33.5小时手术视频和734,769个问答对，这些问答对基于时间分组剪辑和专家注释构建，涵盖四个任务：预测未来阶段、下一步、即将使用的器械和剩余时间。我们进一步提出了SurgAnt-ViVQA，这是一个视频语言模型，使用GRU门控时间交叉注意力模块来适应大语言模型。双向GRU编码帧到帧的动态，同时自适应门将视觉上下文注入到令牌级别的语言流中。参数高效微调将语言主干定制到手术领域。SurgAnt-ViVQA在PitVQA-Anticipation和EndoVis数据集上测试，超越了强大的基于图像和视频的基线。消融研究表明，时间循环和门控融合带来了大部分性能提升。帧预算研究表明存在权衡：8帧最大化流畅性，而32帧略微降低BLEU但改进了数值时间估计。通过将时间感知编码器与细粒度门控交叉注意力相结合，SurgAnt-ViVQA推动手术VQA从回顾性描述发展到主动预测。PitVQA-Anticipation为此设置提供了全面的基准，并强调了针对时间建模对可靠、未来感知型手术辅助的重要性。


### 论文摘要

Anticipating forthcoming surgical events is vital for real-time assistance in endonasal transsphenoidal pituitary surgery, where visibility is limited and workflow changes rapidly. Most visual question answering (VQA) systems reason on isolated frames with static vision language alignment, providing little support for forecasting next steps or instrument needs. Existing surgical VQA datasets likewise center on the current scene rather than the near future. We introduce PitVQA-Anticipation, the first VQA dataset designed for forward looking surgical reasoning. It comprises 33.5 hours of operative video and 734,769 question answer pairs built from temporally grouped clips and expert annotations across four tasks: predicting the future phase, next step, upcoming instrument, and remaining duration. We further propose SurgAnt-ViVQA, a video language model that adapts a large language model using a GRU Gated Temporal Cross-Attention module. A bidirectional GRU encodes frame to frame dynamics, while an adaptive gate injects visual context into the language stream at the token level. Parameter efficient fine tuning customizes the language backbone to the surgical domain. SurgAnt-ViVQA tested upon on PitVQA-Anticipation and EndoVis datasets, surpassing strong image and video based baselines. Ablations show that temporal recurrence and gated fusion drive most of the gains. A frame budget study indicates a trade-off: 8 frames maximize fluency, whereas 32 frames slightly reduce BLEU but improve numeric time estimation. By pairing a temporally aware encoder with fine grained gated cross-attention, SurgAnt-ViVQA advances surgical VQA from retrospective description to proactive anticipation. PitVQA-Anticipation offers a comprehensive benchmark for this setting and highlights the importance of targeted temporal modeling for reliable, future aware surgical assistance.

---

## 27. Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.26241v2](http://arxiv.org/abs/2510.26241v2)

**作者:** Shiho Matta, Lis Kanashiro Pereira, Peitao Han, Fei Cheng, Shigeru Kitazawa

**发布时间:** 2025-10-30

**备注:** 10 pages

### GPT解析

### 总结

研究评估了视觉-语言模型(VLMs)对视频中时间信息的理解能力，发现当前模型在判断时间方向方面表现接近随机水平，远低于人类准确度。

### 背景

现代视觉-语言模型在许多多模态任务中表现出色，但它们对视频中时间信息的掌握仍然薄弱，且这一点尚未得到充分评估。

### 目的

通过判断时间方向(AoT)的挑战来探测VLMs在时间理解方面的差距，即判断短片是正向播放还是反向播放。

### 方法

引入AoT-PsyPhyBENCH基准测试，这是一个经过心理物理学验证的评估工具，使用与人类相同的刺激和行为基线来测试VLMs推断自然视频中时间方向的能力。

### 主要发现

对各类VLMs的全面评估显示，大多数模型表现接近随机水平，即使在物理不可逆过程和因果手动操作上，表现最好的模型也远低于人类准确度，而人类能几乎立即识别这些过程。

### 结论

这些结果突显了当前多模态系统的基本差距：虽然模型能捕捉丰富的视觉-语义相关性，但缺乏时间连续性和因果理解所需的归纳偏差。

### 翻译

现代视觉-语言模型在许多多模态任务中表现出色，但它们对视频中时间信息的掌握仍然薄弱，并且这一点尚未得到充分评估。我们通过一个看似简单但具有揭示性的挑战来探测这一差距：判断时间方向(AoT)——即判断短片是正向播放还是反向播放。我们引入了AoT-PsyPhyBENCH，这是一个经过心理物理学验证的基准测试，用于测试VLMs是否能推断自然视频中的时间方向，使用与人类相同的刺激和行为基线。对开源和专有、推理和非推理VLMs的全面评估表明，大多数模型的表现接近随机水平，即使在物理不可逆过程(如自由落体、扩散/爆炸)和因果手动操作(分割/添加)上，表现最好的模型也远低于人类的准确度，而这些过程人类几乎可以立即识别。这些结果突显了当前多模态系统中的一个基本差距：虽然它们捕捉了丰富的视觉-语义相关性，但缺乏时间连续性和因果理解所需的归纳偏差。我们发布了AoT-PsyPhyBENCH的代码和数据，以鼓励VLMs在物理和时间推理能力方面的进一步进展。


### 论文摘要

Modern vision-language models (VLMs) excel at many multimodal tasks, yet their grasp of temporal information in video remains weak and, crucially, under-evaluated. We probe this gap with a deceptively simple but revealing challenge: judging the arrow of time (AoT)-whether a short clip is played forward or backward. We introduce AoT-PsyPhyBENCH, a psychophysically validated benchmark that tests whether VLMs can infer temporal direction in natural videos using the same stimuli and behavioral baselines established for humans. Our comprehensive evaluation of open-weight and proprietary, reasoning and non-reasoning VLMs reveals that most models perform near chance, and even the best lag far behind human accuracy on physically irreversible processes (e.g., free fall, diffusion/explosion) and causal manual actions (division/addition) that humans recognize almost instantly. These results highlight a fundamental gap in current multimodal systems: while they capture rich visual-semantic correlations, they lack the inductive biases required for temporal continuity and causal understanding. We release the code and data for AoT-PsyPhyBENCH to encourage further progress in the physical and temporal reasoning capabilities of VLMs.

---

## 28. SVG Decomposition for Enhancing Large Multimodal Models Visualization Comprehension: A Study with Floor Plans

**论文链接:** [http://arxiv.org/abs/2511.03478v1](http://arxiv.org/abs/2511.03478v1)

**作者:** Jeongah Lee, Ali Sarvghad

**发布时间:** 2025-11-05

**备注:** 10 pages, 2 figures

### GPT解析

### 总结

本研究探讨了使用可缩放矢量图形(SVG)作为分解策略，以提高大型多模态模型(LMMs)对平面图理解能力的效果。研究分析了三种LMM模型在75个平面图上的表现，发现SVG与光栅输入结合可提高空间理解任务表现，但可能阻碍空间推理能力。

### 背景

大型多模态模型(LMMs)虽然越来越能够解释可视化，但在空间推理方面仍然存在困难。平面图是一个有价值的测试平台，因为它们结合了几何、拓扑和语义，且其可靠理解对盲人和低视力人士的无障碍服务有实际应用价值。

### 目的

研究可缩放矢量图形(SVG)作为一种分解策略，以提高LMMs对平面图理解的效果。

### 方法

进行了一项探索性研究，使用了三个LMMs（GPT-4o、Claude 3.7 Sonnet和Llama 3.2 11B Vision Instruct），分析了75个平面图的表现。

### 主要发现

将SVG与光栅输入结合(SVG+PNG)可以提高空间理解任务的表现，但往往阻碍空间推理，特别是在路径查找方面。

### 结论

这些发现突显了分解策略在推进空间可视化理解方面的潜力和局限性。

### 翻译

大型多模态模型(LMMs)越来越能够解释可视化，但在空间推理方面仍然存在困难。一种提出的策略是分解，将复杂的可视化分解为结构化组件。在这项工作中，我们研究了可缩放矢量图形(SVG)作为一种分解策略的有效性，以提高LMMs对平面图理解的能力。平面图是一个有价值的测试平台，因为它们结合了几何、拓扑和语义，且其可靠理解有实际应用价值，例如为盲人和低视力人士提供无障碍服务。我们对三种LMMs（GPT-4o、Claude 3.7 Sonnet和Llama 3.2 11B Vision Instruct）进行了探索性研究，分析了75个平面图。结果表明，将SVG与光栅输入结合(SVG+PNG)可以提高空间理解任务的表现，但往往阻碍空间推理，特别是在路径查找方面。这些发现突显了分解策略在推进空间可视化理解方面的潜力和局限性。


### 论文摘要

Large multimodal models (LMMs) are increasingly capable of interpreting visualizations, yet they continue to struggle with spatial reasoning. One proposed strategy is decomposition, which breaks down complex visualizations into structured components. In this work, we examine the efficacy of scalable vector graphics (SVGs) as a decomposition strategy for improving LMMs' performance on floor plans comprehension. Floor plans serve as a valuable testbed because they combine geometry, topology, and semantics, and their reliable comprehension has real-world applications, such as accessibility for blind and low-vision individuals. We conducted an exploratory study with three LMMs (GPT-4o, Claude 3.7 Sonnet, and Llama 3.2 11B Vision Instruct) across 75 floor plans. Results show that combining SVG with raster input (SVG+PNG) improves performance on spatial understanding tasks but often hinders spatial reasoning, particularly in pathfinding. These findings highlight both the promise and limitations of decomposition as a strategy for advancing spatial visualization comprehension.

---

## 29. SurgViVQA: Temporally-Grounded Video Question Answering for Surgical Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.03325v1](http://arxiv.org/abs/2511.03325v1)

**作者:** Mauro Orazio Drago, Luca Carlini, Pelinsu Celebi Balyemez, Dennis Pierantozzi, Chiara Lena, Cesare Hassan, Danail Stoyanov, Elena De Momi, Sophia Bano, Mobarak I. Hoque

**发布时间:** 2025-11-05

### GPT解析

### 总结

该论文提出了SurgViVQA，一个专门用于手术领域的视频问答模型，能够捕捉时间连贯的事件而非孤立图像。该模型使用掩码视频-文本编码器融合视频和问题特征，捕捉运动和工具-组织交互等时间线索，由微调的大语言模型解码为连贯答案。作者还创建了REAL-Colon-VQA数据集进行评估，实验表明该模型在关键词准确性上显著优于现有方法。

### 背景

当前手术领域的视频问答方法局限于静态图像特征，可用的数据集通常缺乏时间标注，忽略了准确解读手术程序所需的关键动态信息。手术视频问答旨在通过AI模型对时间连贯事件进行推理来增强手术中的理解。

### 目的

开发一个能够从静态图像扩展到动态手术场景的视频问答模型，捕捉运动和工具-组织交互等时间线索，提高AI模型对动态手术程序的解读能力。

### 方法

提出SurgViVQA模型，使用掩码视频-文本编码器融合视频和问题特征，捕捉时间线索；创建REAL-Colon-VQA数据集，包括与运动相关的问题和诊断属性，以及重新表述或语义改变的问题形式来评估模型鲁棒性；在REAL-Colon-VQA和EndoVis18-VQA数据集上进行实验验证。

### 主要发现

实验表明SurgViVQA在关键词准确性上显著优于现有基于图像的VQA基准模型，在REAL-Colon-VQA上比PitVQA提高11%，在EndoVis18-VQA上提高9%。对问题的扰动研究证实了模型对问题表述变化的泛化能力和鲁棒性得到改善。

### 结论

SurgViVQA和REAL-Colon-VQA数据集为手术视频问答中的时间感知理解提供了框架，使AI模型能够更有效地解读动态手术程序上下文。

### 翻译

手术领域的视频问答旨在通过使AI模型能够对时间上连贯的事件进行推理，而不是孤立的帧，来增强手术中的理解。当前的方法局限于静态图像特征，可用的数据集通常缺乏时间标注，忽略了准确解读手术程序所需的关键动态信息。我们提出了SurgViVQA，一个手术视频问答模型，它将视觉推理从静态图像扩展到动态手术场景。它使用掩码视频-文本编码器来融合视频和问题特征，捕捉运动和工具-组织交互等时间线索，然后由微调的大语言模型解码为连贯的答案。为了评估其性能，我们创建了REAL-Colon-VQA，一个结肠镜视频数据集，包括与运动相关的问题和诊断属性，以及重新表述或语义改变的问题形式来评估模型的鲁棒性。在REAL-Colon-VQA和公共的EndoVis18-VQA数据集上的实验验证表明，SurgViVQA优于现有的基于图像的VQA基准模型，特别是在关键词准确性方面，在REAL-Colon-VQA上比PitVQA提高了11%，在EndoVis18-VQA上提高了9%。对问题的扰动研究进一步证实了模型对问题表述变化的泛化能力和鲁棒性得到了改善。SurgViVQA和REAL-Colon-VQA数据集为手术视频问答中的时间感知理解提供了框架，使AI模型能够更有效地解读动态手术程序上下文。代码和数据集可在https://github.com/madratak/SurgViVQA获取。


### 论文摘要

Video Question Answering (VideoQA) in the surgical domain aims to enhance intraoperative understanding by enabling AI models to reason over temporally coherent events rather than isolated frames. Current approaches are limited to static image features, and available datasets often lack temporal annotations, ignoring the dynamics critical for accurate procedural interpretation. We propose SurgViVQA, a surgical VideoQA model that extends visual reasoning from static images to dynamic surgical scenes. It uses a Masked Video--Text Encoder to fuse video and question features, capturing temporal cues such as motion and tool--tissue interactions, which a fine-tuned large language model (LLM) then decodes into coherent answers. To evaluate its performance, we curated REAL-Colon-VQA, a colonoscopic video dataset that includes motion-related questions and diagnostic attributes, as well as out-of-template questions with rephrased or semantically altered formulations to assess model robustness. Experimental validation on REAL-Colon-VQA and the public EndoVis18-VQA dataset shows that SurgViVQA outperforms existing image-based VQA benchmark models, particularly in keyword accuracy, improving over PitVQA by +11\% on REAL-Colon-VQA and +9\% on EndoVis18-VQA. A perturbation study on the questions further confirms improved generalizability and robustness to variations in question phrasing. SurgViVQA and the REAL-Colon-VQA dataset provide a framework for temporally-aware understanding in surgical VideoQA, enabling AI models to interpret dynamic procedural contexts more effectively. Code and dataset available at https://github.com/madratak/SurgViVQA.

---

## 30. MME-CC: A Challenging Multi-Modal Evaluation Benchmark of Cognitive Capacity

**论文链接:** [http://arxiv.org/abs/2511.03146v1](http://arxiv.org/abs/2511.03146v1)

**作者:** Kaiyuan Zhang, Chenghao Yang, Zhoufutu Wen, Sihang Yuan, Qiuyue Wang, Chaoyi Huang, Guosheng Zhu, He Wang, Huawenyu Lu, Jianing Wen, Jianpeng Jiao, Lishu Luo, Longxiang Liu, Sijin Wu, Xiaolei Zhu, Xuanliang Zhang, Ge Zhang, Yi Lin, Guang Shi, Chaoyou Fu, Wenhao Huang

**发布时间:** 2025-11-05

### GPT解析

### 总结

本文提出了MME-CC基准测试，用于评估多模态大语言模型在视觉信息处理方面的认知能力，通过16个模型的实验发现闭源模型整体领先，但空间和几何推理能力普遍较弱，并确定了常见错误模式。

### 背景

随着推理模型的迅速扩展，多模态在人类认知中的重要作用日益凸显，但现有多模态基准测试要么过度强调文本推理，要么未能系统地捕捉以视觉为中心的认知行为，导致对MLLMs的认知能力评估不足。

### 目的

解决现有多模态基准测试的局限性，引入MME-CC基准测试，系统评估MLLMs在视觉信息处理方面的认知能力。

### 方法

创建MME-CC基准测试，将11个代表性推理任务组织为空间、几何和基于知识的三个基本类别，并对16个代表性MLLMs进行广泛实验。

### 主要发现

闭源模型目前整体领先（如Gemini-2.5-Pro得分为42.66，GLM-4.5V为30.45）；空间和几何推理能力普遍较弱（≤30%）；常见错误包括方向错误、脆弱的跨视图身份持久性和对反事实指令的遵循不良；思维链通常遵循提取→推理→验证的三阶段过程，且严重依赖视觉提取。

### 结论

希望这项工作能促使将MLLMs的认知能力作为评估和模型设计的中心。

### 翻译

随着推理模型的迅速扩展，多模态在人类认知中的重要作用日益凸显，促使人们需要探索以视觉为中心的认知行为。然而，现有的多模态基准测试要么过度强调文本推理，要么未能系统地捕捉以视觉为中心的认知行为，导致对MLLMs的认知能力评估不足。为了解决这一局限性，我们引入了MME-CC（认知能力多模态评估基准），这是一个以视觉为基础的基准测试，将11个代表性的推理任务组织为三个基本类别：空间、几何和基于知识的推理，并提供了MLLMs在这些维度上认知能力的细粒度分析。基于MME-CC，我们对16个代表性的MLLMs进行了广泛的实验。我们的研究表明，闭源模型目前整体领先（例如，Gemini-2.5-Pro得分为42.66，而GLM-4.5V得分为30.45），而空间和几何推理能力普遍较弱（小于或等于30%）。我们进一步确定了常见的错误模式，包括方向错误、脆弱的跨视图身份持久性以及未能很好地遵循反事实指令，并观察到思维链通常遵循三阶段过程（提取→推理→验证），且严重依赖视觉提取。我们希望这项工作能够促使将MLLMs的认知能力作为评估和模型设计的中心。


### 论文摘要

As reasoning models scale rapidly, the essential role of multimodality in human cognition has come into sharp relief, driving a growing need to probe vision-centric cognitive behaviors. Yet, existing multimodal benchmarks either overemphasize textual reasoning or fall short of systematically capturing vision-centric cognitive behaviors, leaving the cognitive capacity of MLLMs insufficiently assessed. To address this limitation, we introduce MME-CC (Multi-Modal Evaluation benchmark of Cognitive Capacity), a vision-grounded benchmark that organizes 11 representative reasoning tasks into three fundamental categories of visual information: spatial, geometric, and knowledge-based reasoning, and provides fine-grained analyses of MLLMs' cognitive capacity across these dimensions. Based on MME-CC, we conduct extensive experiments over 16 representative MLLMs. Our study reveals that closed-source models currently lead overall (e.g., 42.66 for Gemini-2.5-Pro vs. 30.45 for GLM-4.5V), while spatial and geometric reasoning remain broadly weak (less than or equal to 30%). We further identify common error patterns, including orientation mistakes, fragile cross-view identity persistence, and poor adherence to counterfactual instructions, and observe that Chain-of-Thought typically follows a three-stage process (extract -> reason -> verify) with heavy reliance on visual extraction. We hope this work catalyzes a shift toward treating the cognitive capacity of MLLMs as central to both evaluation and model design.

---

## 31. nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN

**论文链接:** [http://arxiv.org/abs/2511.03634v1](http://arxiv.org/abs/2511.03634v1)

**作者:** Alexander Pfefferle, Johannes Hog, Lennart Purucker, Frank Hutter

**发布时间:** 2025-11-05

### GPT解析

### 总结

nanoTabPFN是一个简化的TabPFN v2架构实现，使表格基础模型对学生和研究人员更易访问，在小数据场景下性能与传统机器学习基线相当，且预训练速度快160,000倍。

### 背景

表格基础模型如TabPFN已革新表格数据的预测机器学习，但现有开源实现复杂（超过10,000行代码），缺乏架构文档和代码质量，难以理解和适应新实验。

### 目的

解决现有表格基础模型实现复杂、难以理解、对初学者不友好的问题，使表格基础模型更易于教育和研究使用。

### 方法

引入nanoTabPFN，作为TabPFN v2架构的简化轻量级实现，并使用预生成的训练数据实现相应的训练循环。

### 主要发现

在小数据场景下，nanoTabPFN在单个GPU上进行一分钟预训练后性能与传统机器学习基线相当，预训练速度比TabPFN v2快160,000倍，消除了对大型计算资源的需求。

### 结论

nanoTabPFN使表格基础模型更易于访问，其代码已在GitHub上提供（https://github.com/automl/nanoTabPFN）。

### 翻译

表格基础模型如TabPFN已经革新了表格数据的预测机器学习。同时，这一革命的驱动因素难以理解。现有的开源表格基础模型在复杂的管道中实现，拥有超过10,000行代码，缺乏架构文档或代码质量。简而言之，这些实现难以理解，对初学者不友好，且难以适应新实验。我们引入了nanoTabPFN，这是TabPFN v2架构的简化和轻量级实现，以及使用预生成训练数据的相应训练循环。nanoTabPFN使表格基础模型对学生和研究人员都更加易于访问。例如，限制在小数据场景下，它在单个GPU上进行一分钟预训练后，实现了与传统机器学习基线相当的性能（比TabPFN v2预训练快160,000倍）。这种对大型计算资源需求的消除使表格基础模型的预训练可用于教育目的。我们的代码可在https://github.com/automl/nanoTabPFN获取。


### 论文摘要

Tabular foundation models such as TabPFN have revolutionized predictive machine learning for tabular data. At the same time, the driving factors of this revolution are hard to understand. Existing open-source tabular foundation models are implemented in complicated pipelines boasting over 10,000 lines of code, lack architecture documentation or code quality. In short, the implementations are hard to understand, not beginner-friendly, and complicated to adapt for new experiments. We introduce nanoTabPFN, a simplified and lightweight implementation of the TabPFN v2 architecture and a corresponding training loop that uses pre-generated training data. nanoTabPFN makes tabular foundation models more accessible to students and researchers alike. For example, restricted to a small data setting it achieves a performance comparable to traditional machine learning baselines within one minute of pre-training on a single GPU (160,000x faster than TabPFN v2 pretraining). This eliminated requirement of large computational resources makes pre-training tabular foundation models accessible for educational purposes. Our code is available at https://github.com/automl/nanoTabPFN.

---

## 32. GUIDES: Guidance Using Instructor-Distilled Embeddings for Pre-trained Robot Policy Enhancement

**论文链接:** [http://arxiv.org/abs/2511.03400v1](http://arxiv.org/abs/2511.03400v1)

**作者:** Minquan Gao, Xinyi Li, Qing Yan, Xiaojian Sun, Xiaopan Zhang, Chien-Ming Huang, Jiachen Li

**发布时间:** 2025-11-05

**备注:** 8 pages, 4 figures, Accepted by IEEE IROS 2025 Workshop WIR-M

### GPT解析

### 总结

GUIDES是一个轻量级框架，通过注入基础模型的语义指导来增强预训练的机器人策略，无需重新设计架构。该方法结合了微调的视觉语言模型生成上下文指令，并通过辅助模块编码为引导嵌入，注入到策略的潜在空间中。此外，基于大语言模型的Reflector模块在推理时监控指导模型的置信度，并在置信度低时启动推理循环以增强鲁棒性。实验表明GUIDES能显著提高任务成功率并增强运动精度。

### 背景

预训练的机器人策略是许多已验证的机器人系统的基础，包含大量具身知识。然而，它们通常缺乏基础模型特有的语义感知能力，而完全替换这些策略在许多情况下是不切实际的，因为成本高昂且会损失积累的知识。

### 目的

解决预训练机器人策略缺乏语义感知能力的问题，同时避免完全替换这些策略带来的高成本和知识损失，提供一种实用且资源高效的升级途径。

### 方法

GUIDES框架通过以下步骤实现：1)使用微调的视觉语言模型(Instructor)生成上下文指令；2)通过辅助模块将指令编码为引导嵌入；3)将嵌入注入到策略的潜在空间中；4)通过短暂微调使传统模型适应新语义输入；5)使用基于大语言模型的Reflector模块监控置信度；6)当置信度低时启动推理循环分析历史并优化后续行动。

### 主要发现

在RoboCasa仿真环境中对多种策略架构进行的广泛验证显示，GUIDES在任务成功率方面有一致且显著的提高。在UR5机器人上的实际部署进一步表明，GUIDES增强了抓取等关键子任务的运动精度。

### 结论

GUIDES提供了一种实用且资源高效的途径来升级而非替换已验证的机器人策略，使传统策略能够获得基础模型的语义感知能力，同时保留其积累的知识和经验。

### 翻译

预训练的机器人策略是许多已验证的机器人系统的基础，它们封装了大量具身知识。然而，它们通常缺乏基础模型特有的语义感知能力，并且在许多情况下完全替换它们是不切实际的，因为成本高昂且会损失积累的知识。为了解决这一差距，我们引入了GUIDES，这是一个轻量级框架，通过基础模型的语义指导增强预训练策略，而无需重新设计架构。GUIDES使用微调的视觉语言模型(Instructor)生成上下文指令，这些指令由辅助模块编码为引导嵌入。这些嵌入被注入到策略的潜在空间中，使传统模型能够通过短暂、有针对性的微调来适应这种新的语义输入。为了提高推理时的鲁棒性，基于大语言模型的Reflector模块监控Instructor的置信度，当置信度低时，启动一个推理循环，分析执行历史，检索相关示例，并增强VLM的上下文以细化后续行动。在RoboCasa仿真环境中对多种策略架构进行的广泛验证显示，任务成功率有一致且显著的提高。在UR5机器人上的实际部署进一步表明，GUIDES增强了抓取等关键子任务的运动精度。总体而言，GUIDES提供了一种实用且资源高效的途径来升级而非替换已验证的机器人策略。


### 论文摘要

Pre-trained robot policies serve as the foundation of many validated robotic systems, which encapsulate extensive embodied knowledge. However, they often lack the semantic awareness characteristic of foundation models, and replacing them entirely is impractical in many situations due to high costs and the loss of accumulated knowledge. To address this gap, we introduce GUIDES, a lightweight framework that augments pre-trained policies with semantic guidance from foundation models without requiring architectural redesign. GUIDES employs a fine-tuned vision-language model (Instructor) to generate contextual instructions, which are encoded by an auxiliary module into guidance embeddings. These embeddings are injected into the policy's latent space, allowing the legacy model to adapt to this new semantic input through brief, targeted fine-tuning. For inference-time robustness, a large language model-based Reflector monitors the Instructor's confidence and, when confidence is low, initiates a reasoning loop that analyzes execution history, retrieves relevant examples, and augments the VLM's context to refine subsequent actions. Extensive validation in the RoboCasa simulation environment across diverse policy architectures shows consistent and substantial improvements in task success rates. Real-world deployment on a UR5 robot further demonstrates that GUIDES enhances motion precision for critical sub-tasks such as grasping. Overall, GUIDES offers a practical and resource-efficient pathway to upgrade, rather than replace, validated robot policies.

---

## 33. GMoPE:A Prompt-Expert Mixture Framework for Graph Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.03251v1](http://arxiv.org/abs/2511.03251v1)

**作者:** Zhibin Wang, Zhixing Zhang, Shuqi Wang, Xuanting Xie, Zhao Kang

**发布时间:** 2025-11-05

### GPT解析

### 总结

论文提出GMoPE框架，结合专家混合架构与基于提示的图学习，提高图神经网络跨领域泛化能力并降低适应成本

### 背景

图神经网络在特定任务上表现优异，但跨领域和任务泛化能力有限，现有方法存在负迁移、可扩展性问题和高适应成本

### 目的

解决现有方法的局限性，提高图神经网络的泛化能力和效率

### 方法

提出GMoPE框架，利用专家特定提示向量和结构感知的MoE路由，引入提示向量间软正交约束促进专家多样性，采用仅提示微调策略降低时空复杂度

### 主要发现

实验表明GMoPE优于最先进基线，性能接近完整参数微调，但适应开销显著降低

### 结论

GMoPE为推进可泛化和高效的图基础模型提供了有原则且可扩展的框架

### 翻译

图神经网络在特定任务基准上表现出色，但它们跨不同领域和任务的泛化能力仍然有限。现有方法通常难以处理负迁移、可扩展性问题和高适应成本。为应对这些挑战，我们提出了GMoPE（图专家混合提示），一种将专家混合架构与基于提示的图学习无缝集成的新框架。GMoPE利用专家特定的提示向量和结构感知的MoE路由，使每个专家能够专注于不同的子域，并动态贡献预测。为了促进多样性和防止专家坍塌，我们在提示向量之间引入了软正交约束，鼓励专家专业化并促进更平衡的专家利用。此外，我们采用仅提示微调策略，显著减少了迁移过程中的时空复杂度。我们通过各种预训练策略和多个下游任务的广泛实验验证了GMoPE。结果表明，GMoPE始终优于最先进的基线方法，并且实现了与完整参数微调相当的性能，同时只需要一小部分的适应开销。我们的工作为推进可泛化和高效的图基础模型提供了一个有原则且可扩展的框架。


### 论文摘要

Graph Neural Networks (GNNs) have demonstrated impressive performance on task-specific benchmarks, yet their ability to generalize across diverse domains and tasks remains limited. Existing approaches often struggle with negative transfer, scalability issues, and high adaptation costs. To address these challenges, we propose GMoPE (Graph Mixture of Prompt-Experts), a novel framework that seamlessly integrates the Mixture-of-Experts (MoE) architecture with prompt-based learning for graphs. GMoPE leverages expert-specific prompt vectors and structure-aware MoE routing to enable each expert to specialize in distinct subdomains and dynamically contribute to predictions. To promote diversity and prevent expert collapse, we introduce a soft orthogonality constraint across prompt vectors, encouraging expert specialization and facilitating a more balanced expert utilization. Additionally, we adopt a prompt-only fine-tuning strategy that significantly reduces spatiotemporal complexity during transfer. We validate GMoPE through extensive experiments under various pretraining strategies and multiple downstream tasks. Results show that GMoPE consistently outperforms state-of-the-art baselines and achieves performance comparable to full parameter fine-tuning-while requiring only a fraction of the adaptation overhead. Our work provides a principled and scalable framework for advancing generalizable and efficient graph foundation models.

---

## 34. SENT Map - Semantically Enhanced Topological Maps with Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.03165v1](http://arxiv.org/abs/2511.03165v1)

**作者:** Raj Surya Rajendran Kathirvel, Zach A Chavis, Stephen J. Guy, Karthik Desingh

**发布时间:** 2025-11-05

**备注:** Accepted at ICRA 2025 Workshop on Foundation Models and  Neuro-Symbolic AI for Robotics

### GPT解析

### 总结

SENT-Map是一种语义增强的拓扑地图，用于表示室内环境，通过基础模型支持机器人的自主导航和操作。

### 背景

室内环境的表示对机器人自主导航和操作至关重要，基础模型的发展为环境表示和规划提供了新的可能性。

### 目的

开发一种能够结合语义信息的基础模型友好的室内环境表示方法，使机器人能够在规划过程中避免不可行状态。

### 方法

采用两阶段方法：首先使用视觉基础模型与操作员一起映射环境；然后使用SENT-Map表示和自然语言查询在基础模型中进行规划。SENT-Map以JSON文本格式表示环境，使人类和基础模型都能理解并编辑语义信息。

### 主要发现

语义增强使即使是小型本地部署的基础模型也能够成功规划室内环境。

### 结论

SENT-Map通过JSON文本格式表示环境，支持语义信息的添加和编辑，同时帮助机器人在规划过程中避免不可行状态。

### 翻译

我们引入SENT-Map，一种用于表示室内环境的语义增强拓扑地图，旨在通过利用基础模型的进步来支持自主导航和操作。通过以JSON文本格式表示环境，我们能够以人类和基础模型都能理解的方式添加和编辑语义信息，同时在规划过程中将机器人与现有节点绑定，以避免部署过程中的不可行状态。我们提出的框架采用两阶段方法，首先使用视觉基础模型与操作员一起映射环境，然后使用SENT-Map表示和自然语言查询在基础模型中进行规划。我们的实验结果表明，语义增强使即使是小型本地部署的基础模型也能够成功规划室内环境。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何让机器人在复杂人类环境中进行自主导航和操作的问题，同时利用基础模型的能力但避免其幻觉和虚假置信度等风险。这个问题很重要，因为基础模型虽提供了强大的语义理解和规划能力，但在实际应用中存在可靠性问题，而将基础模型与现实世界位置相结合并通过拓扑地图表示，可以让机器人更安全、可靠地执行任务。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过认识到基础模型的潜力与风险，设计了一个两阶段框架：映射阶段和规划/执行阶段。在映射阶段，人类引导机器人构建环境地图并标记语义节点；在规划阶段，利用基础模型生成任务计划。作者借鉴了现有工作如SLAM、视觉语言模型和对象中心映射方法，但解决了这些方法的高计算需求、缺乏操作推理、3D重建困难以及地图不可编辑等问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一种语义增强的拓扑地图(SENT-Map)，结合基础模型能力与人类可编辑的JSON表示，使机器人能理解环境语义并可靠执行任务。实现流程分两阶段：1)映射阶段：人类引导机器人探索环境，机器人拍摄关键位置图像，基础模型生成JSON格式的语义节点，人类可编辑完善；2)规划阶段：基础模型根据SENT-Map、技能描述和自然语言命令生成可执行计划，机器人执行这些计划完成导航和操作任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)SENT-Map：一种人类可编辑的JSON格式语义增强拓扑地图；2)使用基础模型构建和规划SENT-Map的框架；3)实验证明SENT-Map能提高基础模型规划成功率。相比之前工作，SENT-Map解决了高计算需求、缺乏操作保证、3D重建困难、地图不可验证等问题，通过JSON格式使地图可由人类验证和编辑，同时支持开放集语义增强和自然语言任务规范。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SENT-Map通过结合人类引导的基础模型和可编辑的JSON表示，使机器人能够在复杂环境中更可靠地理解和执行自然语言指令的导航和操作任务。'}


### 论文摘要

We introduce SENT-Map, a semantically enhanced topological map for representing indoor environments, designed to support autonomous navigation and manipulation by leveraging advancements in foundational models (FMs). Through representing the environment in a JSON text format, we enable semantic information to be added and edited in a format that both humans and FMs understand, while grounding the robot to existing nodes during planning to avoid infeasible states during deployment. Our proposed framework employs a two stage approach, first mapping the environment alongside an operator with a Vision-FM, then using the SENT-Map representation alongside a natural-language query within an FM for planning. Our experimental results show that semantic-enhancement enables even small locally-deployable FMs to successfully plan over indoor environments.

---

## 35. Subsampled Randomized Fourier GaLore for Adapting Foundation Models in Depth-Driven Liver Landmark Segmentation

**论文链接:** [http://arxiv.org/abs/2511.03163v1](http://arxiv.org/abs/2511.03163v1)

**作者:** Yun-Chen Lin, Jiayuan Huang, Hanyuan Zhang, Sergi Kavtaradze, Matthew J. Clarkson, Mobarak I. Hoque

**发布时间:** 2025-11-05

**备注:** 12 pages

### GPT解析

### 总结

本文提出了一种深度引导的肝脏地标分割框架，通过视觉基础编码器整合语义和几何线索，并引入SRFT-GaLore方法高效适应大型视觉模型，在腹腔镜肝脏手术中实现了精确的解剖结构检测和分割。

### 背景

医学影像中解剖结构的精确检测和描绘对计算机辅助手术至关重要，尤其在腹腔镜肝脏手术中，2D视频流限制了深度感知，使地标定位复杂化。现有研究利用单目深度线索增强地标检测，但在融合RGB和深度特征以及高效调整大规模视觉模型方面仍有挑战。

### 目的

开发一种深度引导的肝脏地标分割框架，整合语义和几何线索，高效适应大型视觉模型，并评估其跨数据集泛化能力。

### 方法

使用Segment Anything Model V2 (SAM2)编码器提取RGB特征，Depth Anything V2 (DA2)编码器提取深度感知特征；引入SRFT-GaLore低秩梯度投影方法替代计算昂贵的SVD；实现交叉注意力融合模块整合RGB和深度线索；构建新的腹腔镜肝脏手术数据集(LLSD)作为外部验证基准。

### 主要发现

在公共L3D数据集上，Dice相似系数提高4.85%，平均对称表面距离减少11.78个百分点；在LLSD数据集上，模型保持竞争性性能，显著优于基于SAM的基线，展示了强大的跨数据集鲁棒性和对未见手术环境的适应性。

### 结论

SRFT-GaLore增强的双编码器框架能够在实时、深度受限的手术设置中实现可扩展和精确的分割。

### 翻译

医学影像中解剖结构的精确检测和描绘对计算机辅助手术至关重要，特别是在腹腔镜肝脏手术中，2D视频流限制了深度感知并使地标定位复杂化。虽然最近的工作已利用单目深度线索增强地标检测，但在融合RGB和深度特征以及高效调整大规模视觉模型以适应手术领域方面仍存在挑战。我们提出了一种深度引导的肝脏地标分割框架，通过视觉基础编码器整合语义和几何线索。我们采用Segment Anything Model V2 (SAM2)编码器提取RGB特征，Depth Anything V2 (DA2)编码器提取深度感知特征。为高效适应SAM2，我们引入了SRFT-GaLore，一种新颖的低秩梯度投影方法，用子采样随机傅里叶变换(SRFT)替代计算昂贵的SVD。这能够在不牺牲表征能力的情况下高效微调高维注意力层。交叉注意力融合模块进一步整合RGB和深度线索。为评估跨数据集泛化能力，我们还构建了一个新的腹腔镜肝脏手术数据集(LLSD)作为外部验证基准。在公共L3D数据集上，我们的方法比D2GPLand提高了4.85%的Dice相似系数，并将平均对称表面距离降低了11.78个百分点。为进一步评估泛化能力，我们在LLSD数据集上评估了我们的模型。我们的模型保持竞争性性能，并显著优于基于SAM的基线，展示了强大的跨数据集鲁棒性和对未见手术环境的适应性。这些结果表明，我们的SRFT-GaLore增强的双编码器框架能够在实时、深度受限的手术设置中实现可扩展和精确的分割。


### 论文摘要

Accurate detection and delineation of anatomical structures in medical imaging are critical for computer-assisted interventions, particularly in laparoscopic liver surgery where 2D video streams limit depth perception and complicate landmark localization. While recent works have leveraged monocular depth cues for enhanced landmark detection, challenges remain in fusing RGB and depth features and in efficiently adapting large-scale vision models to surgical domains. We propose a depth-guided liver landmark segmentation framework integrating semantic and geometric cues via vision foundation encoders. We employ Segment Anything Model V2 (SAM2) encoder to extract RGB features and Depth Anything V2 (DA2) encoder to extract depth-aware features. To efficiently adapt SAM2, we introduce SRFT-GaLore, a novel low-rank gradient projection method that replaces the computationally expensive SVD with a Subsampled Randomized Fourier Transform (SRFT). This enables efficient fine-tuning of high-dimensional attention layers without sacrificing representational power. A cross-attention fusion module further integrates RGB and depth cues. To assess cross-dataset generalization, we also construct a new Laparoscopic Liver Surgical Dataset (LLSD) as an external validation benchmark. On the public L3D dataset, our method achieves a 4.85% improvement in Dice Similarity Coefficient and a 11.78-point reduction in Average Symmetric Surface Distance compared to the D2GPLand. To further assess generalization capability, we evaluate our model on LLSD dataset. Our model maintains competitive performance and significantly outperforms SAM-based baselines, demonstrating strong cross-dataset robustness and adaptability to unseen surgical environments. These results demonstrate that our SRFT-GaLore-enhanced dual-encoder framework enables scalable and precise segmentation under real-time, depth-constrained surgical settings.

---

## 36. Forecast2Anomaly (F2A): Adapting Multivariate Time Series Foundation Models for Anomaly Prediction

**论文链接:** [http://arxiv.org/abs/2511.03149v1](http://arxiv.org/abs/2511.03149v1)

**作者:** Atif Hassan, Tarun Kumar, Ashish Mishra, Sergey Serebryakov, Satish Kumar Mopur, Phanidhar Koganti, Murthy Chelankuri, Ramanagopal Vogety, Suparna Bhattacharya, Martin Foltin

**发布时间:** 2025-11-05

### GPT解析

### 总结

论文提出了一种名为Forecast2Anomaly (F2A)的新框架，使时间序列基础模型(TSFMs)具备异常预测能力，通过联合预测-异常损失和检索增强生成(RAG)模块实现，能够在不更新模型的情况下跟踪演变的异常。

### 背景

在不同现实世界、动态和复杂系统中预测多变量时间序列中的异常对于预防关键故障至关重要。现有方法仅适用于特定系统，无法随时间演变泛化到异常模式。尽管预训练的时间序列基础模型展示了强大的泛化和零样本预测能力，但其异常预测潜力尚未被开发。

### 目的

开发一个新框架，使预训练的时间序列基础模型(TSFMs)具备异常预测能力，弥合TSFM零样本预测和零样本异常预测之间的差距。

### 方法

F2A框架包含两个关键创新：1) 提出联合预测-异常损失，微调TSFMs以准确预测异常时间点的未来信号；2) 引入检索增强生成(RAG)模块，检索历史上相关的范围并基于它们进行条件预测，使模型在推理时能动态适应分布变化。

### 主要发现

在16个不同数据集和多个TSFM骨干网络上的广泛实验表明，F2A持续优于最先进的方法，提供了可扩展的零样本异常预测解决方案。

### 结论

通过结合目标微调和动态检索，F2A成功将TSFM的零样本预测能力扩展到零样本异常预测，为实际应用提供了一种有效的异常预测解决方案。

### 翻译

摘要翻译：来自不同现实世界、动态和复杂系统的多变量时间序列中的异常预测（异常预测）对于预防关键故障至关重要，能够显著降低运营成本和人工劳动。然而，现有方法仅适用于特定系统，无法随时间演变泛化到异常模式。相比之下，预训练的时间序列基础模型(TSFMs)最近展示了强大的泛化和零样本预测能力。然而，它们在异常预测方面的潜力尚未被开发，因为异常预测与预测正常行为的任务根本不同。因此，我们提出了Forecast2Anomaly (F2A)，一个新颖的框架，通过两个关键创新使TSFMs具备异常预测能力。首先，我们提出了一种联合预测-异常损失，微调TSFMs以准确预测异常时间点的未来信号。其次，我们引入了检索增强生成(RAG)模块，检索历史上相关的范围并基于它们进行条件预测。该组件在推理时动态适应分布变化，使F2A能够在不更新模型的情况下跟踪演变的异常。通过结合目标微调和动态检索，F2A弥合了健壮的TSFM零样本预测和零样本异常预测之间的差距。在16个不同数据集和多个TSFM骨干网络上的广泛实验表明，F2A持续优于最先进的方法，为实际应用提供了可扩展的零样本异常预测解决方案。


### 论文摘要

Forecasting anomalies (anomaly prediction) in multivariate time series from different real-world, dynamic, and complex systems is vital for preempting critical failures, leading to a substantial minimization in operational costs and human labor. Yet, existing methods are limited to specific systems while failing to generalize to evolving anomaly patterns over time. In contrast, pretrained Time Series Foundation Models (TSFMs) have recently demonstrated strong generalization and zero-shot forecasting capabilities. However, their potential remains untapped for anomaly prediction, a task fundamentally different from forecasting normal behavior. Thus, we present Forecast2Anomaly (F2A), a novel framework that empowers TSFMs with anomaly prediction abilities through two key innovations. First, we propose a joint forecast-anomaly loss that fine-tunes TSFMs to accurately forecast future signals even at anomalous time points. Second, we introduce a Retrieval-Augmented Generation (RAG) module that retrieves historically relevant horizons and conditions predictions on them. This component dynamically adapts to distributional shifts at inference time, enabling F2A to track evolving anomalies without requiring model updates. By combining targeted fine-tuning with dynamic retrieval, F2A bridges the gap between robust TSFM zero-shot forecasting and zero-shot anomaly prediction. Extensive experiments across 16 diverse datasets and multiple TSFM backbones show that F2A consistently outperforms state-of-the-art methods, offering a scalable, zero-shot anomaly prediction solution for real-world applications.

---

## 37. A Foundation Model for Brain MRI with Dynamic Modality Integration

**论文链接:** [http://arxiv.org/abs/2511.03014v1](http://arxiv.org/abs/2511.03014v1)

**作者:** Minh Sao Khue Luu, Bair N. Tuchinov

**发布时间:** 2025-11-04

**备注:** Preliminary work; results ongoing

### GPT解析

### 总结

本文提出了一种脑部MRI基础模型，能够处理不同成像序列的组合，使用单一编码器架构，通过可学习模态嵌入和条件层归一化技术，实现了对缺失模态的自适应处理，无需为每种模态单独建模。

### 背景

传统脑部MRI分析方法通常需要为不同成像序列分别训练模型，当某些模态缺失或未见时，模型性能会显著下降，限制了临床应用的实际价值。

### 目的

开发一个能够灵活处理不同成像序列组合的脑部MRI基础模型，使其能够在模态缺失或未见的情况下保持良好性能，提高模型的实用性和适应性。

### 方法

使用单个编码器架构，配备可学习模态嵌入和条件层归一化技术；采用考虑缺失模态的掩码自编码目标函数；应用方差-协方差正则化器稳定特征学习；在约60,000多中心MRI数据上通过自监督重建和模态插值进行训练；利用可学习模态嵌入引导特征提取。

### 主要发现

模型能够有效处理不同模态组合；在序列缺失或未见时能够自适应调整；方差-协方差正则化器有助于稳定特征学习并提高表示多样性；初步结果显示该方法可行。

### 结论

所提出的脑部MRI基础模型能够灵活处理不同成像序列组合，无需为每种模态单独建模，在模态缺失情况下仍能保持良好性能，代码和预训练模型已公开可用。

### 翻译

我们提出了一种脑部MRI基础模型，它可以处理不同成像序列的组合。该模型使用一个带有可学习模态嵌入的编码器、条件层归一化，以及一个考虑缺失模态的掩码自编码目标函数。应用了方差-协方差正则化器来稳定特征学习并提高表示多样性。这种设计消除了为每种模态单独建模的需要，并允许网络在某些序列缺失或未见时进行自适应调整。模型在约60,000多中心MRI上通过自监督重建和模态插值进行训练，以学习灵活的表示。可学习的模态嵌入引导特征提取，使编码器能够适应不同的输入。我们描述了计划在脑肿瘤和多发性硬化症分割以及病变分类方面的评估，在各种模态设置下进行。初步结果显示该方法可行，并计划进行更多实验以更详细地研究其性能。所有代码和预训练模型可在https://github.com/BrainFM/brainfm获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决脑部MRI分析中不同成像序列组合不一致的问题。现实中，不同医院使用的MRI协议不同，导致可用的成像序列（如T1、T2、FLAIR等）存在差异，现有方法通常需要为每种模态组合训练单独的模型，效率低下且难以处理缺失或未见过的新模态组合。这个问题限制了医学影像分析的泛化能力和实际应用价值。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了多个现有工作：自然语言处理和计算机视觉中的掩码自编码器、医学影像的自监督学习方法（如Models Genesis）以及多模态MRI处理技术（如AMAES、M4oE、MoME和mmFormer）。作者首先分析了现有方法的局限性，然后设计了一个单一编码器结合可学习模态嵌入的架构，通过条件层归一化和掩码自编码目标使模型能够适应不同模态组合，并应用方差-协方差正则化来稳定特征学习。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用单一编码器处理不同模态组合的MRI数据，通过可学习的模态嵌入和条件层归一化使模型能够自适应不同输入。整体流程包括：1)数据准备与预处理，将MRI调整为统一大小并应用数据增强；2)模型架构，将MRI分割为3D块并添加模态和位置嵌入；3)使用条件层归一化的Transformer编码器处理可见块，轻量级解码器重建被掩盖块；4)训练目标结合掩码重建和方差-协方差正则化；5)下游任务适配时丢弃解码器，保留编码器。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)动态模态集成，可处理任意MRI序列组合；2)单一编码器架构，无需为每种模态组合创建单独模型；3)可学习模态嵌入，能泛化到未见过序列；4)条件层归一化，使编码器自适应不同模态；5)模态感知掩码重建，提高对缺失数据的鲁棒性；6)方差-协方差正则化，稳定特征学习。相比之前工作，它超越了AMAES的单模态限制，避免了MoME和M4oE需要专家网络的复杂性，比mmFormer更专注于广泛下游应用，且能处理未见过的模态组合。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了BrainFM-MRI，一个能够动态适应不同MRI模态组合的基础模型，通过单一编码器和可学习模态嵌入解决了医疗成像中模态不一致和缺失的问题，为脑部MRI分析提供了更加灵活和鲁棒的解决方案。'}


### 论文摘要

We present a foundation model for brain MRI that can work with different combinations of imaging sequences. The model uses one encoder with learnable modality embeddings, conditional layer normalization, and a masked autoencoding objective that accounts for missing modalities. A variance-covariance regularizer is applied to stabilize feature learning and improve representation diversity. This design removes the need for separate models for each modality and allows the network to adapt when some sequences are missing or unseen. It is trained on about 60,000 multi-center MRIs using self-supervised reconstruction and modality imputation to learn flexible representations. A learnable modality embedding guides feature extraction so the encoder can adjust to different inputs. We describe our planned evaluation on brain tumor and multiple sclerosis segmentation, as well as lesion classification, under various modality settings. Preliminary results show that the method works feasibly, and further experiments are planned to study its performance in more detail. All code and pretrained models are available at https://github.com/BrainFM/brainfm

---

## 38. Zero-shot data citation function classification using transformer-based large language models (LLMs)

**论文链接:** [http://arxiv.org/abs/2511.02936v1](http://arxiv.org/abs/2511.02936v1)

**作者:** Neil Byers, Ali Zaidi, Valerie Skye, Chris Beecroft, Kjiersten Fagnan

**发布时间:** 2025-11-04

### GPT解析

### 总结

本研究应用开源大型语言模型Llama 3.1-405B为引用特定基因组数据集的出版物生成结构化的数据使用案例标签，并引入新的评估框架验证方法有效性。结果显示模型在零样本分类任务上达到0.674的F1分数，但面临数据可用性、提示过拟合等挑战。

### 背景

近年来，确定特定数据集与包含这些数据集的科学文献之间关联的努力有所增加。当已知某出版物引用了某数据集后，探索该数据如何或为何被使用成为下一步逻辑。基于预训练转换器的大型语言模型进步为扩展文献中数据使用案例描述提供了新途径，避免了传统机器学习系统需要昂贵的手动标注和训练数据集开发。

### 目的

应用开源LLM Llama 3.1-405B为已知包含特定基因组数据集的出版物生成结构化数据使用案例标签，并引入新的评估框架确定方法有效性。

### 方法

使用开源大型语言模型Llama 3.1-405B生成结构化数据使用案例标签，针对引用特定基因组数据集的出版物。同时引入创新评估框架验证方法效果。

### 主要发现

基础模型在没有预先定义类别的零样本数据引用分类任务上达到0.674的F1分数，表明使用大型语言模型描述数据使用案例具有良好前景。

### 结论

尽管结果很有希望，但研究受到数据可用性、提示过拟合、计算基础设施和负责任性能评估成本等因素的限制，这些挑战需要在实际应用中加以解决。

### 翻译

近年来，确定特定数据集与包含这些数据集的科学文献之间关联的努力有所增加。已知某出版物引用了某数据集后，下一步逻辑就是探索该数据是如何或为何被使用的。近年来基于预训练、转换器的大型语言模型（LLMs）的进步，为扩展已发表文献中数据使用案例的描述提供了潜在手段。这避免了传统机器学习系统需要昂贵的手动标记和训练数据集开发。在本研究中，我们应用开源LLM Llama 3.1-405B，为已知包含特定基因组数据集的出版物生成结构化的数据使用案例标签。我们还引入了一种新的评估框架来确定我们方法的有效性。我们的结果表明，基础模型在没有预先定义类别的零样本数据引用分类任务上可以达到0.674的F1分数。虽然结果很有前景，但我们的结果受到与数据可用性、提示过拟合、计算基础设施和进行负责任的性能评估所需成本相关的限制。


### 论文摘要

Efforts have increased in recent years to identify associations between specific datasets and the scientific literature that incorporates them. Knowing that a given publication cites a given dataset, the next logical step is to explore how or why that data was used. Advances in recent years with pretrained, transformer-based large language models (LLMs) offer potential means for scaling the description of data use cases in the published literature. This avoids expensive manual labeling and the development of training datasets for classical machine-learning (ML) systems. In this work we apply an open-source LLM, Llama 3.1-405B, to generate structured data use case labels for publications known to incorporate specific genomic datasets. We also introduce a novel evaluation framework for determining the efficacy of our methods. Our results demonstrate that the stock model can achieve an F1 score of .674 on a zero-shot data citation classification task with no previously defined categories. While promising, our results are qualified by barriers related to data availability, prompt overfitting, computational infrastructure, and the expense required to conduct responsible performance evaluation.

---

## 39. Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything

**论文链接:** [http://arxiv.org/abs/2511.02834v2](http://arxiv.org/abs/2511.02834v2)

**作者:** Huawei Lin, Yunzhi Shi, Tong Geng, Weijie Zhao, Wei Wang, Ravender Pal Singh

**发布时间:** 2025-11-04

**备注:** 16 pages, 7 figures, 14 tables. Under Review

### GPT解析

### 总结

Agent-Omni框架通过主代理系统协调现有基础模型，实现灵活的多模态推理而无需重新训练，在多种模态和跨模态推理任务上取得最先进性能。

### 背景

多模态大语言模型(MLLMs)目前只能处理固定模态对，需要大量对齐数据进行昂贵的微调，构建完全全能的模型(能整合文本、图像、音频和视频)仍然不切实际且缺乏强大的推理支持。

### 目的

开发一种能够灵活处理多种模态推理的框架，无需重新训练模型，并能整合现有基础模型的能力。

### 方法

提出Agent-Omni框架，通过主代理系统协调现有基础模型，主代理解释用户意图，将子任务委托给特定模态的代理，并将它们的输出整合成连贯的响应。

### 主要发现

Agent-Omni在文本、图像、音频、视频和全能基准测试中持续取得最先进的性能，特别是在需要复杂跨模态推理的任务上表现出色；基于代理的设计实现了专业基础模型的无缝集成，确保对不同输入的适应性，同时保持透明性和可解释性。

### 结论

Agent-Omni框架是模块化和可扩展的，允许随着更强大模型的可用性进行未来改进，为多模态推理提供了灵活且高效的解决方案。

### 翻译

多模态大语言模型(MLLMs)已展现出强大的能力，但仍局限于固定的模态对，并且需要使用大型对齐数据集进行昂贵的微调。构建能够整合文本、图像、音频和视频的完全全能模型仍然不切实际，且缺乏强大的推理支持。在本文中，我们提出了一个Agent-Omni框架，通过主代理系统协调现有基础模型，实现灵活的多模态推理而无需重新训练。主代理解释用户意图，将子任务委托给特定模态的代理，并将它们的输出整合成连贯的响应。在文本、图像、音频、视频和全能基准测试中的大量实验表明，Agent-Omni持续取得最先进的性能，特别是在需要复杂跨模态推理的任务上。其基于代理的设计实现了专业基础模型的无缝集成，确保对不同输入的适应性，同时保持透明性和可解释性。此外，该框架是模块化和可扩展的，允许随着更强模型的可用性进行未来改进。


### 论文摘要

Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available.

---

## 40. PLUTO-4: Frontier Pathology Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.02826v2](http://arxiv.org/abs/2511.02826v2)

**作者:** Harshith Padigela, Shima Nofallah, Atchuth Naveen Chilaparasetti, Ryun Han, Andrew Walker, Judy Shen, Chintan Shah, Blake Martin, Aashish Sood, Elliot Miller, Ben Glass, Andy Beck, Harsha Pokkalla, Syed Ashar Javed

**发布时间:** 2025-11-04

### GPT解析

### 总结

PLUTO-4是病理学基础模型的下一代版本，包含PLUTO-4S和PLUTO-4G两种互补架构，在多种病理学任务上实现了最先进的性能，具有实际应用潜力。

### 背景

基于大规模病理图像语料库训练的基础模型已显示出在多样组织病理学任务中的强大迁移能力，PLUTO是病理学通用转换器模型。

### 目的

介绍PLUTO-4，扩展PLUTO到前沿规模，并提供两种互补的视觉转换器架构。

### 方法

PLUTO-4S使用FlexiViT设置进行多尺度部署，采用2D-RoPE嵌入；PLUTO-4G使用单一补丁大小训练；两者均使用基于DINOv2的自监督目标在包含551,164个WSI的多机构语料库上预训练；在公共和内部基准上评估性能。

### 主要发现

PLUTO-4在补丁级分类、分割和幻灯片级诊断等任务上实现最先进性能；PLUTO-4S提供高吞吐量和稳健性能；PLUTO-4G在多个病理学基准上建立新性能前沿，皮肤病理学诊断提高11%。

### 结论

PLUTO-4的多样化改进强调了其作为转化研究和诊断用例主干，改变现实世界应用的潜力。

### 翻译

在大型病理图像语料库上训练的基础模型已在多种组织病理学任务中展现出强大的迁移能力。在此基础上，我们介绍了PLUTO-4，这是我们的病理学基础模型下一代版本，将病理学通用转换器(PLUTO)扩展到前沿规模。我们在PLUTO-4家族中分享了两种互补的视觉转换器架构：紧凑高效的PLUTO-4S模型，使用FlexiViT设置进行多尺度部署，采用2D-RoPE嵌入；以及前沿规模的PLUTO-4G模型，使用单一补丁大小训练以最大化表示能力和稳定性。两种模型都使用从DINOv2衍生的自监督目标进行预训练，在包含来自137,144名患者的551,164个WSI的多机构语料库上训练，涵盖50多个机构、60多种疾病类型和100多种染色方法。在公共和内部基准上的全面评估表明，PLUTO-4在需要不同空间和生物学背景的任务上实现了最先进的性能，包括补丁级分类、分割和幻灯片级诊断。紧凑的PLUTO-4S为实际部署提供高吞吐量和稳健性能，而PLUTO-4G在多个病理学基准上建立了新的性能前沿，包括在皮肤病理学诊断上提高11%。这些多样化的改进强调了PLUTO-4作为转化研究和诊断用例主干，改变现实世界应用的潜力。


### 论文摘要

Foundation models trained on large-scale pathology image corpora have demonstrated strong transfer capabilities across diverse histopathology tasks. Building on this progress, we introduce PLUTO-4, our next generation of pathology foundation models that extend the Pathology-Universal Transformer (PLUTO) to frontier scale. We share two complementary Vision Transformer architectures in the PLUTO-4 family: a compact and efficient PLUTO-4S model optimized for multi-scale deployment using a FlexiViT setup with 2D-RoPE embeddings, and a frontier-scale PLUTO-4G model trained with a single patch size to maximize representation capacity and stability. Both models are pretrained using a self-supervised objective derived from DINOv2 on a large multi-institutional corpus containing 551,164 WSIs from 137,144 patients across over 50 institutions, spanning over 60 disease types and over 100 stains. Comprehensive evaluation across public and internal benchmarks demonstrates that PLUTO-4 achieves state-of-the-art performance on tasks requiring varying spatial and biological context, including patch-level classification, segmentation, and slide-level diagnosis. The compact PLUTO-4S provides high-throughput and robust performance for practical deployment, while PLUTO-4G establishes new performance frontiers across multiple pathology benchmarks, including an 11% improvement in dermatopathology diagnosis. These diverse improvements underscore PLUTO-4's potential to transform real-world applications as a backbone for translational research and diagnostic use cases.

---

## 41. TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.02802v2](http://arxiv.org/abs/2511.02802v2)

**作者:** Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Utsav Avaiya, Vinay Kumar Sankarapu

**发布时间:** 2025-11-04

**备注:** The library is open source and available at  https://github.com/Lexsi-Labs/TabTune

### GPT解析

### 总结

TabTune是一个统一库，旨在标准化表格基础模型的完整工作流程，通过单一接口提供一致的服务，解决了表格基础模型采用中的主要障碍，包括异构预处理管道、碎片化API、不一致微调程序和缺乏标准化评估等问题。

### 背景

表格基础模型是结构化数据学习中的一个新兴范式，将大规模预训练的好处扩展到表格领域，但其采用仍然有限。

### 目的

提出一个统一的库来标准化表格基础模型的完整工作流程，通过单一接口提供一致的服务。

### 方法

提出TabTune库，提供对七种最先进模型的一致访问，支持零样本推理、元学习、监督微调和参数高效微调等多种适应策略；框架自动化模型感知预处理，内部管理架构异构性，并集成性能、校准和公平性的评估模块。

### 主要发现

表格基础模型的采用受到异构预处理管道、碎片化API、不一致微调程序和缺乏针对校准和公平性等部署导向指标的标准化评估的限制。

### 结论

TabTune通过提供可扩展且可重现的框架，能够对表格基础模型的适应策略进行一致的基准测试，解决了表格基础模型采用中的主要障碍。

### 翻译

表格基础模型代表了结构化数据学习中的一个不断增长的范式，将大规模预训练的好处扩展到表格领域。然而，由于异构预处理管道、碎片化API、不一致的微调程序以及缺乏针对校准和公平性等部署导向指标的标准化评估，它们的采用仍然有限。我们提出了TabTune，这是一个通过单一界面标准化表格基础模型完整工作流程的统一库。TabTune提供对七种最先进模型的一致访问，支持多种适应策略，包括零样本推理、元学习、监督微调(SFT)和参数高效微调(PEFT)。该框架自动化模型感知预处理，内部管理架构异构性，并集成性能、校准和公平性的评估模块。TabTune设计为可扩展和可重现，能够对表格基础模型的适应策略进行一致的基准测试。


### 论文摘要

Tabular foundation models represent a growing paradigm in structured data learning, extending the benefits of large-scale pretraining to tabular domains. However, their adoption remains limited due to heterogeneous preprocessing pipelines, fragmented APIs, inconsistent fine-tuning procedures, and the absence of standardized evaluation for deployment-oriented metrics such as calibration and fairness. We present TabTune, a unified library that standardizes the complete workflow for tabular foundation models through a single interface. TabTune provides consistent access to seven state-of-the-art models supporting multiple adaptation strategies, including zero-shot inference, meta-learning, supervised fine-tuning (SFT), and parameter-efficient fine-tuning (PEFT). The framework automates model-aware preprocessing, manages architectural heterogeneity internally, and integrates evaluation modules for performance, calibration, and fairness. Designed for extensibility and reproducibility, TabTune enables consistent benchmarking of adaptation strategies of tabular foundation models.

---

## 42. Discourse-Aware Scientific Paper Recommendation via QA-Style Summarization and Multi-Level Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2511.03330v1](http://arxiv.org/abs/2511.03330v1)

**作者:** Shenghua Wang, Zhen Yin

**发布时间:** 2025-11-05

### GPT解析

### 总结

这项研究提出了OMRC-MR框架，一种结合问答式OMRC摘要、多级对比学习和结构感知重排序的层次化方法，用于科学论文推荐，显著提高了推荐的精确度和召回率。

### 背景

开放获取出版物的快速增长使识别相关科学论文更具挑战性。由于隐私限制和用户交互数据有限访问，研究转向基于内容的推荐方法，但这些方法通常将论文视为非结构化文本，忽略了话语组织结构，限制了语义完整性和可解释性。

### 目的

开发能够捕捉论文结构化信息的内容推荐方法，提高科学论文推荐的精确度和召回率，同时增强推荐结果的可解释性。

### 方法

OMRC-MR框架包含三个主要模块：问答式OMRC摘要模块将原始论文转换为结构化表示；多级对比学习目标在元数据、章节和文档级别对齐语义表示；结构感知重排序阶段通过上下文相似度校准优化检索精确度。

### 主要发现

在DBLP、S2ORC和Sci-OMRC数据集上的实验表明，OMRC-MR在Precision@10和Recall@10上分别实现了高达7.2%和3.8%的改进。问答式摘要产生了更连贯和事实完整的表示。

### 结论

OMRC-MR为科学论文推荐提供了一个统一且可解释的内容范式，推进了可信和隐私感知的学术信息检索。

### 翻译

开放获取出版物的快速增长加剧了识别相关科学论文的挑战。由于隐私限制和用户交互数据的有限访问，近期努力转向了基于内容的推荐，这完全依赖于文本信息。然而，现有模型通常将论文视为非结构化文本，忽略了它们的话语组织，从而限制了语义完整性和可解释性。为了解决这些限制，我们提出了OMRC-MR，这是一个层次化框架，集成了问答式OMRC摘要、多级对比学习和结构感知重排序用于学术推荐。问答式摘要模块将原始论文转换为结构化和话语一致的表示，而多级对比目标在元数据、章节和文档级别对齐语义表示。最终的重排序阶段通过上下文相似度校准进一步优化检索精确度。在DBLP、S2ORC和新构建的Sci-OMRC数据集上的实验表明，OMRC-MR始终超越最先进的基线，在Precision@10和Recall@10上分别实现了高达7.2%和3.8%的改进。额外的评估确认问答式摘要产生了更连贯和事实完整的表示。总体而言，OMRC-MR为科学论文推荐提供了一个统一且可解释的内容范式，推进了可信和隐私感知的学术信息检索。


### 论文摘要

The rapid growth of open-access (OA) publications has intensified the challenge of identifying relevant scientific papers. Due to privacy constraints and limited access to user interaction data, recent efforts have shifted toward content-based recommendation, which relies solely on textual information. However, existing models typically treat papers as unstructured text, neglecting their discourse organization and thereby limiting semantic completeness and interpretability. To address these limitations, we propose OMRC-MR, a hierarchical framework that integrates QA-style OMRC (Objective, Method, Result, Conclusion) summarization, multi-level contrastive learning, and structure-aware re-ranking for scholarly recommendation. The QA-style summarization module converts raw papers into structured and discourse-consistent representations, while multi-level contrastive objectives align semantic representations across metadata, section, and document levels. The final re-ranking stage further refines retrieval precision through contextual similarity calibration. Experiments on DBLP, S2ORC, and the newly constructed Sci-OMRC dataset demonstrate that OMRC-MR consistently surpasses state-of-the-art baselines, achieving up to 7.2% and 3.8% improvements in Precision@10 and Recall@10, respectively. Additional evaluations confirm that QA-style summarization produces more coherent and factually complete representations. Overall, OMRC-MR provides a unified and interpretable content-based paradigm for scientific paper recommendation, advancing trustworthy and privacy-aware scholarly information retrieval.

---

## 43. An Augmentation Overlap Theory of Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2511.03114v1](http://arxiv.org/abs/2511.03114v1)

**作者:** Qi Zhang, Yifei Wang, Yisen Wang

**发布时间:** 2025-11-05

### GPT解析

### 总结

本文研究了自监督对比学习的底层工作机制，提出了基于增强重叠的理论框架，并开发了与下游性能高度一致的无监督评估指标。

### 背景

自监督对比学习在各种任务上取得了巨大成功，但其底层工作机制尚不清楚。

### 目的

探索对比学习的底层工作机制，特别是放松条件独立假设，研究更实际的增强重叠假设对下游性能的影响。

### 方法

基于条件独立假设提供最紧的界限；将条件独立假设放松到增强重叠假设，推导下游性能的渐近闭合界限；提出增强重叠理论；开发用于对比学习表示评估的无监督指标。

### 主要发现

不同类内样本的支撑集在激进数据增强下会变得更加重叠，简单对齐正样本可使对比学习将类内样本聚集在一起。

### 结论

所提出的增强重叠理论解释了对比学习的机制，开发的无监督评估指标与下游性能高度一致，几乎不需要额外模块。

### 翻译

最近，自监督对比学习在各种任务上取得了巨大成功。然而，其底层工作机制尚不清楚。在本文中，我们首先基于广泛采用的条件独立假设提供最紧的界限。进一步，我们将条件独立假设放松到更实际的增强重叠假设，并推导出下游性能的渐近闭合界限。我们提出的增强重叠理论基于这样的洞察：不同类内样本的支撑集在激进的数据增强下会变得更加重叠，因此简单地对齐正样本（同一样本的增强视图）可以使对比学习将类内样本聚集在一起。此外，从新推导的增强重叠角度，我们开发了一种用于对比学习表示评估的无监督指标，它与下游性能几乎完美一致，且几乎不需要依赖额外模块。代码可在https://github.com/PKU-ML/GARC获取。


### 论文摘要

Recently, self-supervised contrastive learning has achieved great success on various tasks. However, its underlying working mechanism is yet unclear. In this paper, we first provide the tightest bounds based on the widely adopted assumption of conditional independence. Further, we relax the conditional independence assumption to a more practical assumption of augmentation overlap and derive the asymptotically closed bounds for the downstream performance. Our proposed augmentation overlap theory hinges on the insight that the support of different intra-class samples will become more overlapped under aggressive data augmentations, thus simply aligning the positive samples (augmented views of the same sample) could make contrastive learning cluster intra-class samples together. Moreover, from the newly derived augmentation overlap perspective, we develop an unsupervised metric for the representation evaluation of contrastive learning, which aligns well with the downstream performance almost without relying on additional modules. Code is available at https://github.com/PKU-ML/GARC.

---

## 44. Stochastic Deep Graph Clustering for Practical Group Formation

**论文链接:** [http://arxiv.org/abs/2511.02879v1](http://arxiv.org/abs/2511.02879v1)

**作者:** Junhyung Park, Hyungjin Kim, Seokho Ahn, Young-Duk Seo

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了DeepForm框架，解决群组推荐系统中的动态群组形成问题，满足高阶用户信息整合、实时群组形成和动态群组数量调整三个关键需求。

### 背景

现有群组推荐系统研究主要关注提高推荐准确性，但多数方法假设群组静态或预定义，不适合动态、真实场景。

### 目的

重新将群组形成视为群组推荐系统的核心挑战，提出满足三个关键操作要求的框架：整合高阶用户信息、实时群组形成和动态调整群组数量。

### 方法

DeepForm采用轻量级GCN架构捕获高阶结构信号，通过随机聚类学习实现无需重新训练的自适应群组重新配置，利用对比学习在动态条件下优化群组。

### 主要发现

多个数据集实验表明，DeepForm在群组形成质量、效率和推荐准确性方面均优于各种基线方法。

### 结论

DeepForm为动态场景下的群组形成提供了有效解决方案，同时保持了高质量的推荐性能。

### 翻译

虽然先前关于群组推荐系统(GRSs)的研究主要集中在提高推荐准确性上，但大多数方法假设群组是静态或预定义的，使它们不适合动态、真实的场景。我们将群组形成重新定义为GRSs中的核心挑战，并提出了DeepForm（用于实际群组形成的随机深度图聚类），这是一个旨在满足三个关键操作要求的框架：(1)整合高阶用户信息，(2)实时群组形成，(3)动态调整群组数量。DeepForm采用轻量级GCN架构，有效捕获高阶结构信号。随机聚类学习使群组能够自适应重新配置而无需重新训练，同时对比学习在动态条件下优化群组。在多个数据集上的实验表明，与各种基线相比，DeepForm在群组形成质量、效率和推荐准确性方面都表现出色。


### 论文摘要

While prior work on group recommender systems (GRSs) has primarily focused on improving recommendation accuracy, most approaches assume static or predefined groups, making them unsuitable for dynamic, real-world scenarios. We reframe group formation as a core challenge in GRSs and propose DeepForm (Stochastic Deep Graph Clustering for Practical Group Formation), a framework designed to meet three key operational requirements: (1) the incorporation of high-order user information, (2) real-time group formation, and (3) dynamic adjustment of the number of groups. DeepForm employs a lightweight GCN architecture that effectively captures high-order structural signals. Stochastic cluster learning enables adaptive group reconfiguration without retraining, while contrastive learning refines groups under dynamic conditions. Experiments on multiple datasets demonstrate that DeepForm achieves superior group formation quality, efficiency, and recommendation accuracy compared with various baselines.

---

## 45. Probabilistic Graph Cuts

**论文链接:** [http://arxiv.org/abs/2511.02272v2](http://arxiv.org/abs/2511.02272v2)

**作者:** Ayoub Ghriss

**发布时间:** 2025-11-04

**备注:** 23 pages

### GPT解析

### 总结

作者提出了一种统一的概率框架，作为谱聚类的可微分替代方法，涵盖广泛的图切类型，包括Normalized Cut。该框架提供了紧密的解析上界，具有闭合形式的前向和反向传播，为可扩展的、可微分的图分割提供了严格且数值稳定的基础。

### 背景

概率松弛的图切作为谱聚类的可微分替代方法，可以在不进行特征分解的情况下实现端到端和在线学习。然而，先前的工作主要集中在RatioCut上，缺乏通用保证和有原则的梯度。

### 目的

提出一个统一的概率框架，涵盖广泛的图切类型，包括Normalized Cut，并提供紧密的解析上界和具有闭合形式的前向和反向传播。

### 方法

构建统一的概率框架，通过积分表示和具有闭合形式前向和反向传播的高斯超几何函数，为期望离散切提供紧密的解析上界。

### 主要发现

该框架提供了紧密的解析上界，具有闭合形式的前向和反向传播，使得算法在数值上更加稳定。

### 结论

这些结果为可扩展的、可微分的图分割提供了严格且数值稳定的基础，涵盖了广泛的聚类和对比学习目标。

### 翻译

概率松弛的图切为谱聚类提供了可微分的替代方案，无需特征分解即可实现端到端和在线学习，但先前的工作主要集中在RatioCut上，缺乏通用保证和有原则的梯度。我们提出了一个统一的概率框架，涵盖了广泛的图切类型，包括Normalized Cut。我们的框架通过积分表示和具有闭合形式前向和反向传播的高斯超几何函数，为期望离散切提供了紧密的解析上界。这些结果共同为可扩展的、可微分的图分割提供了严格且数值稳定的基础，涵盖了广泛的聚类和对比学习目标。


### 论文摘要

Probabilistic relaxations of graph cuts offer a differentiable alternative to spectral clustering, enabling end-to-end and online learning without eigendecompositions, yet prior work centered on RatioCut and lacked general guarantees and principled gradients. We present a unified probabilistic framework that covers a wide class of cuts, including Normalized Cut. Our framework provides tight analytic upper bounds on expected discrete cuts via integral representations and Gauss hypergeometric functions with closed-form forward and backward. Together, these results deliver a rigorous, numerically stable foundation for scalable, differentiable graph partitioning covering a wide range of clustering and contrastive learning objectives.

---

## 46. Enhancing composition-based materials property prediction by cross-modal knowledge transfer

**论文链接:** [http://arxiv.org/abs/2511.03371v1](http://arxiv.org/abs/2511.03371v1)

**作者:** Ivan Rubtsov, Ivan Dudakov, Yuri Kuratov, Vadim Korolev

**发布时间:** 2025-11-05

**备注:** 7 pages, 2 figures, 1 table

### GPT解析

### 总结

晶体图神经网络在建模实验合成的化合物和具有未知合成可能性的假设材料方面具有广泛应用。相比之下，结构不可知预测算法可以探索化学空间中以前无法访问的领域。本研究提出了一种通过跨模态知识转移来增强基于成分的材料属性预测的通用方法。

### 背景

晶体图神经网络广泛用于建模实验合成的化合物和未知合成可能性的假设材料。结构不可知预测算法则允许探索化学空间中以前无法访问的领域。

### 目的

开发一种通用方法，通过跨模态知识转移来增强基于成分的材料属性预测。

### 方法

提出了两种公式：隐式转移涉及在多模态嵌入上预训练化学语言模型，而显式转移建议生成晶体结构并实现结构感知预测器。这些方法在LLM4Mat-Bench和MatBench任务上进行了基准测试。

### 主要发现

提出的方法在32个案例中的25个案例中达到了最先进的性能。此外，展示了化学语言模型的另一个建模方面——可解释性——如何通过应用博弈论方法受益，该方法能够纳入高阶特征交互。

### 结论

通过跨模态知识转移的方法可以显著提高基于成分的材料属性预测性能，化学语言模型的可解释性可以通过博弈论方法得到增强。

### 翻译

晶体图神经网络在建模实验合成的化合物和具有未知合成可能性的假设材料方面具有广泛应用。相比之下，结构不可知预测算法可以探索化学空间中以前无法访问的领域。在此，我们提出了一种通过跨模态知识转移来增强基于成分的材料属性预测的通用方法。提出了两种公式：隐式转移涉及在多模态嵌入上预训练化学语言模型，而显式转移建议生成晶体结构并实现结构感知预测器。所提出的方法在LLM4Mat-Bench和MatBench任务上进行了基准测试，在32个案例中的25个案例中取得了最先进的性能。此外，我们展示了化学语言模型的另一个建模方面——可解释性——如何通过应用博弈论方法受益，该方法能够纳入高阶特征交互。


### 论文摘要

Crystal graph neural networks are widely applicable in modeling experimentally synthesized compounds and hypothetical materials with unknown synthesizability. In contrast, structure-agnostic predictive algorithms allow exploring previously inaccessible domains of chemical space. Here we present a universal approach for enhancing composition-based materials property prediction by means of cross-modal knowledge transfer. Two formulations are proposed: implicit transfer involves pretraining chemical language models on multimodal embeddings, whereas explicit transfer suggests generating crystal structures and implementing structure-aware predictors. The proposed approaches were benchmarked on LLM4Mat-Bench and MatBench tasks, achieving state-of-the-art performance in 25 out of 32 cases. In addition, we demonstrated how another modeling aspect of chemical language models - interpretability - benefits from applying a game-theoretic approach, which is able to incorporate high-order feature interactions.

---

## 47. GraphCliff: Short-Long Range Gating for Subtle Differences but Critical Changes

**论文链接:** [http://arxiv.org/abs/2511.03170v1](http://arxiv.org/abs/2511.03170v1)

**作者:** Hajung Kim, Jueon Park, Junseok Choe, Sheunheun Baek, Hyeon Hwang, Jaewoo Kang

**发布时间:** 2025-11-05

### GPT解析

### 总结

本文提出了一种名为GraphCliff的新模型，通过门控机制整合短程和长程信息，解决了分子图嵌入在区分结构相似但功能不同分子时表现不佳的问题，从而提高了在活性悬崖和非悬崖化合物上的预测性能。

### 背景

定量构效关系(QSAR)假设分子结构与生物活性之间存在平滑关系，但活性悬崖(结构相似但效力差异大的化合物对)会破坏这种连续性。最近的基准测试表明，具有扩展连接性指纹的经典机器学习模型在处理活性悬崖时优于图神经网络。

### 目的

开发一种能够保留分子图结构表示并有效区分结构相似但功能不同分子的模型，以提高在活性悬崖和非悬崖化合物上的预测性能。

### 方法

提出名为GraphCliff的新模型，通过门控机制整合短程和长程信息，保留分子作为图的结构表示。

### 主要发现

GraphCliff在非悬崖和悬崖化合物上都一致提高了性能；分层节点嵌入分析显示与强基线图模型相比，减少了过平滑并增强了判别能力。

### 结论

GraphCliff成功解决了分子图嵌入在区分结构相似但功能不同分子时表现不佳的问题，同时保留了分子图结构的表达力，为处理活性悬崖问题提供了有效解决方案。

### 翻译

定量构效关系假设分子结构和生物活性之间存在平滑关系。然而，活性悬崖被定义为结构相似的化合物对，其效力差异很大，这破坏了这种连续性。最近针对活性悬崖的基准测试表明，具有扩展连接性指纹的经典机器学习模型优于图神经网络。我们的分析表明，图嵌入无法在嵌入空间中充分分离结构相似的分子，这使得难以区分结构相似但功能不同的分子。尽管存在这一限制，分子图结构本质上具有表达力且吸引人，因为它们保留了分子拓扑结构。为了保留分子作为图的结构表示，我们提出了一个新模型GraphCliff，它通过门控机制整合短程和长程信息。实验结果表明，GraphCliff在非悬崖和悬崖化合物上都一致提高了性能。此外，分层节点嵌入分析显示与强基线图模型相比，减少了过平滑并增强了判别能力。


### 论文摘要

Quantitative structure-activity relationship assumes a smooth relationship between molecular structure and biological activity. However, activity cliffs defined as pairs of structurally similar compounds with large potency differences break this continuity. Recent benchmarks targeting activity cliffs have revealed that classical machine learning models with extended connectivity fingerprints outperform graph neural networks. Our analysis shows that graph embeddings fail to adequately separate structurally similar molecules in the embedding space, making it difficult to distinguish between structurally similar but functionally different molecules. Despite this limitation, molecular graph structures are inherently expressive and attractive, as they preserve molecular topology. To preserve the structural representation of molecules as graphs, we propose a new model, GraphCliff, which integrates short- and long-range information through a gating mechanism. Experimental results demonstrate that GraphCliff consistently improves performance on both non-cliff and cliff compounds. Furthermore, layer-wise node embedding analyses reveal reduced over-smoothing and enhanced discriminative power relative to strong baseline graph models.

---

## 48. Homomorphism distortion: A metric to distinguish them all and in the latent space bind them

**论文链接:** [http://arxiv.org/abs/2511.03068v1](http://arxiv.org/abs/2511.03068v1)

**作者:** Martin Carrasco, Olga Zaghen, Erik Bekkers, Bastian Rieck

**发布时间:** 2025-11-04

### GPT解析

### 总结

本文提出了一种新的图相似性度量方法——图同态失真(graph homomorphism distortion)，它能够完全表征图，是一种完整的图嵌入。通过采样方法可以有效计算这一度量，并从中获得一个度量标准。实证表明，该方法能区分传统方法无法区分的图，并在特定数据集上优于现有方法。

### 背景

长期以来，图神经网络的表达能力仅通过组合性质来衡量，缺乏对图相似性的原则性测量方法。

### 目的

提供一种测量顶点属性图相似性的原则性方法，并开发一种新的图相似性度量。

### 方法

引入图同态失真(graph homomorphism distortion)作为相似性度量，证明它可以完全表征图，因此也是一种完整的图嵌入。为解决图规范化问题，通过采样方法有效计算这一度量，并从中获得一个度量标准。

### 主要发现

图同态失真能够：(1)完全区分BREC数据集中的图，包括那些通过4-WL无法区分的图；(2)在ZINC-12k数据集上，优于之前受同态启发的方法。

### 结论

这些理论和实证结果为未来图的表征铺平了道路，将图论传统扩展到新的前沿领域。

### 翻译

长期以来，图神经网络的表达能力仅通过组合性质来衡量。本文打破了这一传统，提供了一种测量顶点属性图相似性的原则性方法，我们将其称为图同态失真(graph homomorphism distortion)。我们证明它可以完全表征图，因此也是一种完整的图嵌入。然而，在研究过程中，我们遇到了图规范化问题。为克服这一障碍，我们设计了通过采样来有效计算这一度量的方法，期望上保证了完整性。此外，我们还发现可以从这一度量中获得一个度量标准。我们通过实证验证了我们的主张，发现图同态失真：(1)能完全区分BREC数据集中的图，包括那些通过4-WL无法区分的图；(2)在ZINC-12k数据集上，优于之前受同态启发的方法。这些理论结果（及其实证验证）为未来图的表征铺平了道路，将图论传统扩展到新的前沿领域。


### 论文摘要

For far too long, expressivity of graph neural networks has been measured \emph{only} in terms of combinatorial properties. In this work we stray away from this tradition and provide a principled way to measure similarity between vertex attributed graphs. We denote this measure as the \emph{graph homomorphism distortion}. We show it can \emph{completely characterize} graphs and thus is also a \emph{complete graph embedding}. However, somewhere along the road, we run into the graph canonization problem. To circumvent this obstacle, we devise to efficiently compute this measure via sampling, which in expectation ensures \emph{completeness}. Additionally, we also discovered that we can obtain a metric from this measure. We validate our claims empirically and find that the \emph{graph homomorphism distortion}: (1.) fully distinguishes the \texttt{BREC} dataset with up to $4$-WL non-distinguishable graphs, and (2.) \emph{outperforms} previous methods inspired in homomorphisms under the \texttt{ZINC-12k} dataset.   These theoretical results, (and their empirical validation), pave the way for future characterization of graphs, extending the graph theoretic tradition to new frontiers.

---

## 49. Digital Twin-Driven Pavement Health Monitoring and Maintenance Optimization Using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.02957v1](http://arxiv.org/abs/2511.02957v1)

**作者:** Mohsin Mahmud Topu, Mahfuz Ahmed Anik, Azmine Toushik Wasi, Md Manjurul Ahsan

**发布时间:** 2025-11-04

### GPT解析

### 总结

该研究提出了一种结合数字孪生和图神经网络的创新方法，用于路面基础设施的智能监测和维护，解决了传统路面管理系统被动响应的问题，实现了主动维护和预测性规划。

### 背景

路面基础设施监测面临复杂的空间依赖性、变化的环境条件和道路网络上的非线性退化等挑战。传统的路面管理系统(PMS)主要是被动的，缺乏故障预防和最优维护规划的实时智能。

### 目的

提出一个统一的数字孪生(DT)和图神经网络(GNN)框架，用于可扩展、数据驱动的路面健康监测和预测性维护。

### 方法

将路段和空间关系建模为图的节点和边，实时无人机、传感器和LiDAR数据流入数字孪生系统。归纳式GNN从图结构输入中学习退化模式以预测损坏。开发了交互式仪表板和强化学习模块用于模拟、可视化和自适应维护规划。

### 主要发现

模型实现了0.3798的R²值，优于基线回归器，并有效捕获了非线性退化模式。

### 结论

DT-GNN集成提高了预测精度并建立了持续改进的闭环反馈系统，为主动、智能和可持续的路面管理奠定了基础，未来将向实际部署、多智能体协调和智慧城市集成扩展。

### 翻译

路面基础设施监测面临复杂的空间依赖性、变化的环境条件和道路网络上的非线性退化等挑战。传统的路面管理系统(PMS)主要是被动的，缺乏故障预防和最优维护规划的实时智能。为解决这一问题，我们提出了一个统一的数字孪生(DT)和图神经网络(GNN)框架，用于可扩展、数据驱动的路面健康监测和预测性维护。路段和空间关系被建模为图的节点和边，而实时无人机、传感器和LiDAR数据流入数字孪生系统。归纳式GNN从图结构输入中学习退化模式，以预测损坏并实现主动干预。在具有路段属性和动态连接的真实世界启发的数据集上训练，我们的模型实现了0.3798的R²值，优于基线回归器，并有效捕获了非线性退化。我们还开发了交互式仪表板和强化学习模块，用于模拟、可视化和自适应维护规划。这种DT-GNN集成提高了预测精度，并建立了持续改进的闭环反馈系统，将该方法定位为主动、智能和可持续路面管理的基础，未来将向实际部署、多智能体协调和智慧城市集成扩展。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决传统路面管理系统(PMS)的被动维护问题，这些系统只在路面出现故障时才进行干预，缺乏实时智能来预防故障和优化维护计划。这个问题很重要，因为路面是现代交通系统的支柱，恶化路面会导致旅行时间延长、燃料消耗增加、车辆运营成本提高和交通事故风险增加。传统固定时间表的维护方法可能与实际退化轨迹不匹配，造成过早的结构性故障或过高的维护成本。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统路面管理系统的局限性，然后注意到数字孪生(DT)技术可以提供物理资产的详细实时虚拟表示，图神经网络(GNN)能够处理复杂的时空数据。作者借鉴了DT技术在多个领域的应用经验，以及GNN在分析相互连接系统方面的能力，将两者结合创建了一个统一框架。他们整合了现有的数据采集技术，如实时UAV、传感器和LiDAR数据，并利用了已有的路面监测和预测方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将数字孪生(DT)和图神经网络(GNN)集成到一个统一框架中，使用图结构表示路面段和空间关系，通过实时数据更新DT，并利用GNN学习退化模式。整体实现流程包括：1)数据合成与集成层，收集和预处理多源数据；2)图构建模块，将数据转换为动态图表示；3)模拟和分析引擎，使用有限元建模、无人机和LiDAR评估以及GNN预测分析；4)交互式维护和可视化系统，提供决策支持和可视化界面。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)集成数字孪生框架实现实时动态监测；2)开发基于GNN的预测模型捕获路面段间的时空依赖关系；3)DT-GNN集成支持交互式模拟和假设场景分析；4)全面的比较评估证明系统性能优于传统方法。相比之前工作，不同之处在于：之前研究主要关注孤立的DT应用或独立的GNN模型，而本文将两者统一集成，实现了从被动到主动维护的范式转变，建立了闭环反馈系统，提高了预测精度并促进了连续监测。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过集成数字孪生和图神经网络技术，建立了一个实时、数据驱动的路面健康监测和维护优化框架，实现了从被动到主动维护的范式转变，提高了预测准确性并延长了路面使用寿命。'}


### 论文摘要

Pavement infrastructure monitoring is challenged by complex spatial dependencies, changing environmental conditions, and non-linear deterioration across road networks. Traditional Pavement Management Systems (PMS) remain largely reactive, lacking real-time intelligence for failure prevention and optimal maintenance planning. To address this, we propose a unified Digital Twin (DT) and Graph Neural Network (GNN) framework for scalable, data-driven pavement health monitoring and predictive maintenance. Pavement segments and spatial relations are modeled as graph nodes and edges, while real-time UAV, sensor, and LiDAR data stream into the DT. The inductive GNN learns deterioration patterns from graph-structured inputs to forecast distress and enable proactive interventions. Trained on a real-world-inspired dataset with segment attributes and dynamic connectivity, our model achieves an R2 of 0.3798, outperforming baseline regressors and effectively capturing non-linear degradation. We also develop an interactive dashboard and reinforcement learning module for simulation, visualization, and adaptive maintenance planning. This DT-GNN integration enhances forecasting precision and establishes a closed feedback loop for continuous improvement, positioning the approach as a foundation for proactive, intelligent, and sustainable pavement management, with future extensions toward real-world deployment, multi-agent coordination, and smart-city integration.

---

