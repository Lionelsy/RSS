# 今日论文推荐 - 2025-10-21

共 115 篇论文

---

## 1. 论文ID: 2510.17772v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.17772v1.json'

---

## 2. Flow-Aware Ellipsoidal Filtration for Persistent Homology of Recurrent Signals

**论文链接:** [http://arxiv.org/abs/2510.17735v1](http://arxiv.org/abs/2510.17735v1)

**作者:** Omer Bahadir Eryilmaz, Cihan Katar, Max A. Little

**发布时间:** 2025-10-20

**备注:** 23 pages, 13 figures. Extended version of the paper presented at  NOLTA 2025; prepared for journal submission

### GPT解析

### 总结

本文提出了一种名为椭圆滤波的新型滤波方法，用于从动态光滑流中采样的点云分析，改进了循环信号的去噪和循环时间的估计。

### 背景

持久同调通常用于探索点云的形状，这些点云假设是从几何对象中采样得到的。传统方法使用各向同性球体在不断增加的尺度上创建拓扑结构。

### 目的

提出一种假设点云从动态光滑流中采样的新型滤波方法，以改进信号处理和特征估计。

### 方法

椭圆滤波基于局部流方差在点周围创建椭圆体，随着尺度增加来近似流的流形，而非使用传统的各向同性球体方法。

### 主要发现

构建椭圆邻域能够改进循环信号的去噪和循环时间的估计，特别是在数据包含瓶颈的情况下效果更佳。

### 结论

根据H1类的最大持久性选择椭圆体，可以为去噪和循环时间估计提供数据驱动的阈值。

### 翻译

持久同调的一个常见用途是探索点云的形状，其中点假设是从几何对象中采样的。我们提出了一种新型滤波，称为椭圆滤波，它假设点云是从动态光滑流中采样的。椭圆滤波不是使用各向同性球体（例如Vietoris-Rips滤波）在不断增加的尺度上从点云创建拓扑结构，而是基于局部流方差在点周围创建椭圆体，随着尺度的增加来近似流的流形。我们表明，构建椭圆邻域可以改进循环信号的去噪和循环时间的估计，特别是当数据包含瓶颈时。根据H1类的最大持久性选择椭圆体，可以为去噪和循环时间估计提供数据驱动的阈值。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决传统持久同调方法（如Vietoris-Rips过滤）在处理重复信号时的局限性，特别是在信号含有瓶颈结构和不同采样密度的情况下表现不佳的问题。这个问题在现实中很重要，因为重复信号在自然系统中很常见（如生理信号、气候数据、机械振动等），准确分析其拓扑结构对于理解系统动态特性、去噪和估计返回时间等任务至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别到传统拓扑数据分析方法在处理时间序列数据时的局限性，特别是在处理瓶颈结构和不同采样密度时表现不佳。他们借鉴了持久同调的基本概念，特别是H1类的持久性分析；参考了Fernández等人的Fermat距离方法，该方法考虑了采样密度；受Kališnik等人的椭球过滤工作启发，但他们的工作针对的是静态点云而非时间序列数据。作者结合时空邻域概念，通过局部协方差估计和主成分分析构建自适应椭球体，使其能够根据局部流动方向和速度调整形状，从而更好地捕捉信号的动态特性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点云视为从动态光滑流中采样的样本，而非静态几何对象，通过使用自适应的椭球体邻域替代传统的各向同性球体邻域来更好地捕捉信号的局部流动特性。整体实现流程包括：1)局部协方差估计，结合时间邻域和空间邻域；2)基于PCA构建自适应椭球体；3)检测椭球体相交情况；4)构建椭球复形；5)进行持久同调分析；6)应用在信号去噪和返回时间估计上。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)时空邻域设计，同时考虑短期时间演化和空间重复性；2)自适应椭球体过滤，根据局部流动方向和速度调整形状；3)基于H1类最大持久性的数据驱动尺度选择方法；4)将拓扑分析方法应用于返回时间估计。相比之前的工作，本文方法超越了传统Vietoris-Rips过滤的各向同性限制，解决了Fermat距离方法未整合时间信息的问题，并改进了Kališnik等人针对静态点云的椭球过滤方法，使其能够处理时间序列数据的时空特性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于流动感知的自适应椭球过滤方法，通过结合时空邻域信息和局部流动几何，显著提高了持久同调在处理重复信号（特别是含有瓶颈结构和不同采样密度的信号）时的拓扑分析、去噪和返回时间估计性能。'}


### 论文摘要

One common use of persistent homology is to explore the shape of point clouds, where points are assumed to be sampled from a geometric object. We propose a novel filtration, called ellipsoidal filtration, which assumes that point clouds are sampled from a dynamic smooth flow. Instead of creating topologies from point clouds at increasing scales using isotropic balls (for example, Vietoris-Rips filtration), ellipsoidal filtration creates ellipsoids around points based on local flow variances, approximating the flow's manifold as the scale increases. We show that constructing ellipsoidal neighbourhoods improves the denoising of recurrent signals and the estimation of recurrence times, especially when the data contain bottlenecks. Choosing ellipsoids according to the maximum persistence of the H1 class provides a data-driven threshold for both denoising and recurrence-time estimation.

---

## 3. Raindrop GS: A Benchmark for 3D Gaussian Splatting under Raindrop Conditions

**论文链接:** [http://arxiv.org/abs/2510.17719v1](http://arxiv.org/abs/2510.17719v1)

**作者:** Zhiqiang Teng, Beibei Lin, Tingting Chen, Zifeng Yuan, Xuanyi Li, Xuanyu Zhang, Shunli Zhang

**发布时间:** 2025-10-20

### GPT解析

### 总结

本研究针对雨滴条件下3D高斯散射(3DGS)的重建质量问题，提出了一个名为RaindropGS的综合基准测试，用于评估从受雨滴影响的非约束图像到清晰3D重建的完整流程。

### 背景

3DGS在雨滴条件下因镜头上的雨滴污染而面临严重的遮挡和光学失真，导致重建质量显著下降。现有基准测试通常使用已知相机姿态的合成雨滴图像评估3DGS，假设理想条件。然而，在真实场景中，雨滴干扰相机姿态估计和点云初始化，且合成与真实雨滴间的域差距损害了泛化能力。

### 目的

解决雨滴条件下3DGS评估的局限性，提出一个全面基准测试，评估从受雨滴影响的非约束图像到清晰3DGS重建的完整流程。

### 方法

RaindropGS基准测试包含三部分：数据准备、数据处理和雨滴感知的3DGS评估。研究收集了真实世界雨滴重建数据集，每个场景包含三个对齐图像集：雨滴聚焦、背景聚焦和无雨滴真实地面，用于全面评估不同聚焦条件下的重建质量。

### 主要发现

现有3DGS方法在非约束雨滴图像上存在性能限制；相机聚焦位置显著影响3DGS重建性能；不准确的姿态估计和点云初始化对重建造成干扰。

### 结论

研究为开发雨滴条件下更强大的3DGS方法提供了明确方向，通过全面评估揭示了不同流水线组件的影响和局限性。

### 翻译

在雨滴条件下的3D高斯散射(3DGS)因相机镜头上雨滴污染造成的严重遮挡和光学失真而遭受显著重建质量下降。现有基准测试通常使用具有已知相机姿态的合成雨滴图像来评估3DGS，假设理想条件。然而，在真实场景中，雨滴常常干扰准确的相机姿态估计和点云初始化。此外，合成雨滴和真实雨滴之间的显著域差距进一步损害了泛化能力。为了解决这些问题，我们引入了RaindropGS，这是一个全面的基准测试，旨在评估从受雨滴影响的非约束图像到清晰的3DGS重建的完整3DGS流程。具体而言，整个基准测试流程包括三个部分：数据准备、数据处理和雨滴感知的3DGS评估，包括雨滴干扰类型、相机姿态估计和点云初始化、单图像雨滴去除比较以及3D高斯训练比较。首先，我们收集了一个真实世界的雨滴重建数据集，其中每个场景包含三个对齐的图像集：雨滴聚焦、背景聚焦和无雨滴真实地面，从而能够全面评估不同聚焦条件下的重建质量。通过全面的实验和分析，我们揭示了现有3DGS方法在非约束雨滴图像上的性能限制以及不同流水线组件的 varying 影响：相机聚焦位置对3DGS重建性能的影响，以及不准确姿态和点云初始化对重建造成的干扰。这些见解为在雨滴条件下开发更强大的3DGS方法指明了明确的方向。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在雨滴条件下进行3D高斯泼溅(3DGS)重建时，因相机镜头上雨滴造成的遮挡和光学失真导致的重建质量下降问题。这个问题在现实中很重要，因为雨滴是常见的环境干扰因素，会影响自动驾驶、增强现实等户外视觉应用；在研究中，现有方法在合成数据上表现良好但在真实场景中效果不佳，且缺乏针对真实雨滴条件的全面评估基准，无法准确衡量方法在实际应用中的性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别出现有研究的局限性：主要在合成数据上评估，假设了理想条件；认识到雨滴影响3DGS流程的多个阶段；注意到真实雨滴和合成雨滴间的差异；强调相机聚焦位置的重要性。作者设计了RaindropGS基准，包括数据准备(收集真实世界数据集)、数据处理(评估相机姿态估计、点云初始化和雨滴去除)和雨滴感知的3DGS评估。该方法借鉴了现有3DGS方法、COLMAP和VGGT进行姿态估计、以及Uformer、Restormer和IDT等雨滴去除模型。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个全面基准，评估从受雨滴影响的未约束图像到清晰3DGS重建的完整流程，关注每个关键步骤(相机姿态估计、点云初始化、雨滴去除)对最终质量的影响。整体流程：1)数据收集-11个真实场景，每场景有雨滴聚焦、背景聚焦和无雨滴真实图像；2)数据处理-使用COLMAP和VGGT进行姿态估计和点云初始化，使用Uformer、Restormer和IDT进行雨滴去除；3)3DGS重建-使用原始3DGS、WeatherGS、GS-W和3DGS-MCMC四种方法；4)评估-使用PSNR、SSIM等指标比较不同聚焦条件和预处理策略的影响。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个全面的3DGS雨滴重建基准，覆盖完整流程；2)首个真实世界3DGS雨滴重建数据集，包含三种对齐图像集和随机变化的雨滴；3)对现有方法的全面评估和见解。相比之前工作：1)评估从合成数据转向真实世界数据；2)关注点从仅关注3D高斯拟合扩展到整个流程；3)数据集从合成转向真实，考虑了雨滴在不同视角的变化；4)明确考虑了雨滴聚焦和背景聚焦两种条件，使评估更全面。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'RaindropGS提供了一个全面的基准和真实世界数据集，用于评估在真实雨滴条件下3D高斯泼溅技术的完整重建流程，揭示了现有方法的局限性并指明了未来研究方向。'}


### 论文摘要

3D Gaussian Splatting (3DGS) under raindrop conditions suffers from severe occlusions and optical distortions caused by raindrop contamination on the camera lens, substantially degrading reconstruction quality. Existing benchmarks typically evaluate 3DGS using synthetic raindrop images with known camera poses (constrained images), assuming ideal conditions. However, in real-world scenarios, raindrops often interfere with accurate camera pose estimation and point cloud initialization. Moreover, a significant domain gap between synthetic and real raindrops further impairs generalization. To tackle these issues, we introduce RaindropGS, a comprehensive benchmark designed to evaluate the full 3DGS pipeline-from unconstrained, raindrop-corrupted images to clear 3DGS reconstructions. Specifically, the whole benchmark pipeline consists of three parts: data preparation, data processing, and raindrop-aware 3DGS evaluation, including types of raindrop interference, camera pose estimation and point cloud initialization, single image rain removal comparison, and 3D Gaussian training comparison. First, we collect a real-world raindrop reconstruction dataset, in which each scene contains three aligned image sets: raindrop-focused, background-focused, and rain-free ground truth, enabling a comprehensive evaluation of reconstruction quality under different focus conditions. Through comprehensive experiments and analyses, we reveal critical insights into the performance limitations of existing 3DGS methods on unconstrained raindrop images and the varying impact of different pipeline components: the impact of camera focus position on 3DGS reconstruction performance, and the interference caused by inaccurate pose and point cloud initialization on reconstruction. These insights establish clear directions for developing more robust 3DGS methods under raindrop conditions.

---

## 4. Towards 3D Objectness Learning in an Open World

**论文链接:** [http://arxiv.org/abs/2510.17686v1](http://arxiv.org/abs/2510.17686v1)

**作者:** Taichi Liu, Zhenyu Wang, Ruofeng Liu, Guang Wang, Desheng Zhang

**发布时间:** 2025-10-20

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

本文提出OP3Det，一种无类别开放世界无提示3D检测器，能够检测3D场景中的任何物体，包括训练中未见过的物体，显著提升了开放世界3D物体检测性能。

### 背景

3D物体检测和新类别检测虽有进展，但学习泛化3D物体性的研究仍不足；传统封闭集3D检测器难以泛化到开放世界场景，而直接引入3D开放词汇模型面临词汇扩展和语义重叠的挑战。

### 目的

研究开放世界3D物体性，实现能够检测3D场景中所有物体（包括训练中未见过的物体）的泛化3D物体发现。

### 方法

提出OP3Det检测器，引入2D基础模型的强泛化和零样本能力，利用2D语义先验和3D几何先验进行无类别提案，通过跨模态专家混合集成点云和RGB图像的互补信息，动态路由单模态和多模态特征以学习泛化的3D物体性。

### 主要发现

OP3Det表现卓越，与现有开放世界3D检测器相比在AR指标上显著提高最多16.0%，与封闭世界3D检测器相比实现了13.5%的改进。

### 结论

OP3Det成功实现了开放世界3D物体检测的目标，能够检测3D场景中的任何物体而不依赖手工制作的文本提示。

### 翻译

最近的3D物体检测和新类别检测研究取得了显著进展，然而关于学习泛化3D物体性的研究仍然不足。在本文中，我们深入研究开放世界3D物体性学习，专注于检测3D场景中的所有物体，包括训练中未见过的新物体。传统的封闭集3D检测器难以泛化到开放世界场景，而直接将3D开放词汇模型整合以获得开放世界能力又面临词汇扩展和语义重叠的挑战。为实现泛化的3D物体发现，我们提出了OP3Det，一种无类别开放世界无提示3D检测器，可以检测3D场景中的任何物体，而不依赖手工制作的文本提示。我们引入了2D基础模型的强泛化和零样本能力，利用2D语义先验和3D几何先验进行无类别提案，以扩展3D物体发现。然后，通过在跨模态专家混合中集成点云和RGB图像的互补信息，OP3Det动态路由单模态和多模态特征以学习泛化的3D物体性。大量实验证明了OP3Det的卓越性能，在AR指标上显著超越现有开放世界3D检测器最多16.0%，相比封闭世界3D检测器实现了13.5%的改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决开放世界中的3D物体检测问题，特别是学习通用的3D物体性，使模型能够检测3D场景中的所有物体，包括训练过程中未见过的物体类别。这个问题在现实世界中非常重要，因为自动驾驶、机器人等应用场景中物体类别可能会动态变化，传统封闭集3D检测器无法泛化到这些开放世界场景，而直接使用开放词汇模型又面临词汇扩展和语义重叠的问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先注意到现实世界环境中物体类别可能动态变化，导致需要开放世界3D检测能力。观察到2D领域在未知物体识别方面已有较多探索，但3D领域有限。意识到3D点云数据规模和标注类别有限，而2D领域有丰富的基础模型和训练数据。因此，作者将2D预训练模型的强大零样本能力转移到3D领域。借鉴了SAM模型用于提取类别无关物体掩码，以及多模态融合方法，但提出了创新的跨模态专家混合模块来解决现有方法的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用2D基础模型（特别是SAM）的强大泛化能力帮助3D物体发现，并通过多尺度点采样策略优化SAM产生的掩码，再通过跨模态专家混合模块动态融合单模态和多模态特征学习通用3D物体性。整体流程分为：1)3D物体发现阶段，使用SAM和多尺度点采样策略处理RGB图像，投影到3D空间；2)训练阶段，提取点云和图像特征，使用跨模态MoE融合特征；3)推理阶段，直接在点云-图像对上进行检测，完全无提示地进行类别无关的3D物体性推理。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次定义并解决3D领域的类别无关开放世界物体检测问题；2)设计多尺度点采样策略增强2D-3D关联，解决SAM碎片化掩码问题；3)提出跨模态专家混合模块动态选择单模态和多模态路径。相比传统封闭集3D检测器，它能检测未见过的物体类别；相比开放词汇3D检测器，它不需要预定义词汇表；相比现有多模态融合方法，它保留模态特定信息并动态选择最相关特征。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了OP3Det，首个类别无关的开放世界3D检测器，通过利用2D语义知识和创新的跨模态专家混合机制，实现了在开放世界中检测所有物体的能力，包括训练过程中未见过的物体类别。'}


### 论文摘要

Recent advancements in 3D object detection and novel category detection have made significant progress, yet research on learning generalized 3D objectness remains insufficient. In this paper, we delve into learning open-world 3D objectness, which focuses on detecting all objects in a 3D scene, including novel objects unseen during training. Traditional closed-set 3D detectors struggle to generalize to open-world scenarios, while directly incorporating 3D open-vocabulary models for open-world ability struggles with vocabulary expansion and semantic overlap. To achieve generalized 3D object discovery, We propose OP3Det, a class-agnostic Open-World Prompt-free 3D Detector to detect any objects within 3D scenes without relying on hand-crafted text prompts. We introduce the strong generalization and zero-shot capabilities of 2D foundation models, utilizing both 2D semantic priors and 3D geometric priors for class-agnostic proposals to broaden 3D object discovery. Then, by integrating complementary information from point cloud and RGB image in the cross-modal mixture of experts, OP3Det dynamically routes uni-modal and multi-modal features to learn generalized 3D objectness. Extensive experiments demonstrate the extraordinary performance of OP3Det, which significantly surpasses existing open-world 3D detectors by up to 16.0% in AR and achieves a 13.5% improvement compared to closed-world 3D detectors.

---

## 5. Integrating BIM and UAV-based photogrammetry for Automated 3D Structure Model Segmentation

**论文链接:** [http://arxiv.org/abs/2510.17609v1](http://arxiv.org/abs/2510.17609v1)

**作者:** Siqi Chen, Shanyue Guan

**发布时间:** 2025-10-20

### GPT解析

### 总结

该研究提出了一种基于机器学习的框架，用于自动分割3D点云，特别是在基础设施健康监测中的应用，结合了无人机扫描的真实数据和从BIM生成的合成数据。

### 背景

无人机技术进步实现了高效、非接触式的结构健康监测，结合摄影测量可捕获高分辨率扫描并重建基础设施的详细3D模型，但从这些模型中分割特定结构组件仍面临挑战，传统方法依赖耗时且容易出错的手动标注。

### 目的

解决从3D模型中分割特定结构组件的挑战，开发一种自动化分割方法替代耗时且易错的手动标注流程。

### 方法

提出一种基于机器学习的框架用于3D点云自动分割，利用真实无人机扫描点云和从建筑信息建模(BIM)生成的合成数据的互补优势，结合BIM数据克服手动标注的局限性。

### 主要发现

在铁路轨道数据集验证中，该方法在识别和分割主要组件如铁轨和轨枕方面表现高准确性；通过使用较小的数据集并补充BIM数据，框架显著减少了训练时间同时保持了合理的分割准确性。

### 结论

这种自动化方法提高了3D基础设施模型分割的精度和效率，推进了无人机和BIM技术在结构健康监测和基础设施管理中的集成应用。

### 翻译

无人机技术的进步实现了高效、非接触式的结构健康监测。结合摄影测量技术，无人机可以捕获高分辨率扫描并重建基础设施的详细3D模型。然而，从这些模型中分割特定结构组件仍然是一个关键挑战，这一过程传统上依赖于耗时且容易出错的手动标注。为解决这一问题，我们提出了一种基于机器学习的框架，用于3D点云的自动分割。我们的方法利用真实无人机扫描点云和从建筑信息建模(BIM)生成的合成数据的互补优势，克服了与手动标注相关的局限性。在铁路轨道数据集上的验证显示，在识别和分割铁轨和轨枕等主要组件方面具有高准确性。此外，通过使用较小的数据集并补充BIM数据，该框架显著减少了训练时间，同时保持了合理的分割准确性。这种自动化方法提高了3D基础设施模型分割的精度和效率，并推进了无人机和BIM技术在结构健康监测和基础设施管理中的集成。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从3D点云中自动分割铁路基础设施的关键结构组件（如钢轨和轨枕）的问题。传统方法依赖耗时且易错的手动标注，这在大型数据集上尤其不切实际。这个问题很重要，因为铁路是社会经济支柱，组件损坏可能导致严重事故；人工检查既耗时又危险；而无人机虽高效安全，但点云分割仍需手动标注，限制了自动化监测的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到铁路基础设施健康监测的重要性及传统人工检查的局限性。他们借鉴了无人机技术用于基础设施检查的研究、摄影测量3D重建方法、以及PointNet++等深度学习网络用于点云分割的工作。在此基础上，他们创新性地提出结合真实世界无人机扫描点云和从建筑信息模型生成的合成数据，设计了一个机器学习框架，通过创建不同规模的训练数据集并测试不同旋转策略，实现了高效准确的自动分割。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是结合真实世界无人机扫描点云和从BIM生成的合成数据，减少对手动标注数据的依赖，实现铁路基础设施3D点云的自动分割。实现流程包括：1)使用无人机沿规划路径采集铁路图像并重建3D点云；2)对真实点云进行手动标注，同时创建BIM模型并自动生成标注点云；3)对数据进行下采样并创建不同规模的数据集；4)使用PointNet++架构训练分割模型；5)评估模型性能并测试在不同材料轨枕上的泛化能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)创新性地融合真实点云与BIM合成数据；2)使用较小规模训练数据集结合BIM数据实现高效分割；3)系统研究数据增强策略对模型性能的影响；4)展示模型在不同材料轨枕上的泛化能力。相比之前工作，本文不仅减少了手动标注需求，还显著缩短了训练时间，实现了小规模数据集的高效利用，并证明了方法在不同材料铁路结构上的适用性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种创新的机器学习框架，通过结合无人机摄影测量的真实点云和BIM生成的合成数据，实现了铁路基础设施3D点云的高效自动分割，为铁路健康监测提供了一种可扩展且实用的解决方案。'}


### 论文摘要

The advancement of UAV technology has enabled efficient, non-contact structural health monitoring. Combined with photogrammetry, UAVs can capture high-resolution scans and reconstruct detailed 3D models of infrastructure. However, a key challenge remains in segmenting specific structural components from these models-a process traditionally reliant on time-consuming and error-prone manual labeling. To address this issue, we propose a machine learning-based framework for automated segmentation of 3D point clouds. Our approach uses the complementary strengths of real-world UAV-scanned point clouds and synthetic data generated from Building Information Modeling (BIM) to overcome the limitations associated with manual labeling. Validation on a railroad track dataset demonstrated high accuracy in identifying and segmenting major components such as rails and crossties. Moreover, by using smaller-scale datasets supplemented with BIM data, the framework significantly reduced training time while maintaining reasonable segmentation accuracy. This automated approach improves the precision and efficiency of 3D infrastructure model segmentation and advances the integration of UAV and BIM technologies in structural health monitoring and infrastructure management.

---

## 6. PAGE-4D: Disentangled Pose and Geometry Estimation for 4D Perception

**论文链接:** [http://arxiv.org/abs/2510.17568v1](http://arxiv.org/abs/2510.17568v1)

**作者:** Kaichen Zhou, Yuhan Wang, Grace Chen, Xinhai Chang, Gaspard Beaudouin, Fangneng Zhan, Paul Pu Liang, Mengyu Wang

**发布时间:** 2025-10-20

### GPT解析

### 总结

本文提出了一种名为PAGE-4D的前馈模型，能够处理包含动态元素的3D场景，实现相机姿态估计、深度预测和点云重建，无需后处理处理。

### 背景

现有的3D前馈模型（如VGGT）在静态场景的3D属性推断上表现良好，但由于主要在静态数据集上训练，在处理包含移动人类或可变形物体等复杂动态元素的真实世界场景时效果不佳。

### 目的

引入PAGE-4D模型，将VGGT扩展到动态场景，实现相机姿态估计、深度预测和点云重建的多功能4D重建系统。

### 方法

提出了一种动态感知聚合器，通过预测动态感知掩码来解耦静态和动态信息，在相机姿态估计时抑制运动线索，在几何重建时增强这些线索，解决了多任务4D重建中任务间的固有冲突。

### 主要发现

PAGE-4D在动态场景中始终优于原始VGGT，在相机姿态估计、单目和视频深度估计以及密集点图重建方面取得了更好的性能表现。

### 结论

PAGE-4D成功解决了多任务4D重建中任务间的固有冲突，通过动态感知聚合器有效处理了静态和动态场景中的各种任务，无需后处理即可实现高质量的4D重建。

### 翻译

最近的3D前馈模型，如视觉几何基础变换器（VGGT），在推断静态场景的3D属性方面表现出强大能力。然而，由于它们通常在静态数据集上训练，这些模型在涉及复杂动态元素的真实世界场景中往往表现不佳，例如移动的人类或像雨伞这样的可变形物体。为了解决这一局限性，我们引入了PAGE-4D，一种将VGGT扩展到动态场景的前馈模型，能够实现相机姿态估计、深度预测和点云重建——所有这些都无需后处理。多任务4D重建的一个核心挑战是任务之间的固有冲突：准确的相机姿态估计需要抑制动态区域，而几何重建则需要建模这些区域。为了解决这种张力，我们提出了一种动态感知聚合器，通过预测动态感知掩码来解耦静态和动态信息——为姿态估计抑制运动线索，同时为几何重建增强这些线索。大量实验表明，PAGE-4D在动态场景中始终优于原始VGGT，在相机姿态估计、单目和视频深度估计以及密集点图重建方面取得了优越的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决将静态3D场景理解模型扩展到动态场景的问题。这个问题在现实中非常重要，因为动态场景（如包含移动人或物体的环境）在自动驾驶、机器人导航、增强现实等应用中非常普遍，准确理解和重建这些场景对现实应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先观察到VGGT在静态场景中表现良好但在动态场景中性能下降，然后分析发现核心冲突：相机姿态估计需要抑制动态区域，而几何重建则需要利用动态信息。他们借鉴了VGGT的基础架构，引入了动态感知聚合器和注意力机制，并采用了针对性的微调策略，只调整对动态最敏感的中间层，而不是重新设计整个网络。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是解耦动态信息在不同任务中的作用：对相机姿态估计抑制动态区域，对几何重建则利用动态信息。整体流程包括：输入RGB图像序列；使用预训练编码器提取特征；通过三阶段动态感知聚合器处理特征；最后通过解码器输出深度图、3D点云和相机姿态估计结果。其中动态感知聚合器预测动态掩码，并根据任务类型有选择地应用这个掩码。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 动态感知聚合器，解耦动态和静态信息；2) 针对性微调策略，只调整关键层；3) 动态掩码预测，自适应学习动态区域；4) 任务特定的注意力应用。相比之前的工作，PAGE-4D不需要大的架构改变，而是通过微调和动态感知注意力机制实现，在保持高效的同时提高了动态场景的性能，超越了VGGT和其他动态场景模型。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PAGE-4D通过解耦动态信息在不同任务中的作用，有效地将静态3D场景理解模型扩展到动态场景，实现了高效的相机姿态估计和几何重建。'}


### 论文摘要

Recent 3D feed-forward models, such as the Visual Geometry Grounded Transformer (VGGT), have shown strong capability in inferring 3D attributes of static scenes. However, since they are typically trained on static datasets, these models often struggle in real-world scenarios involving complex dynamic elements, such as moving humans or deformable objects like umbrellas. To address this limitation, we introduce PAGE-4D, a feedforward model that extends VGGT to dynamic scenes, enabling camera pose estimation, depth prediction, and point cloud reconstruction -- all without post-processing. A central challenge in multi-task 4D reconstruction is the inherent conflict between tasks: accurate camera pose estimation requires suppressing dynamic regions, while geometry reconstruction requires modeling them. To resolve this tension, we propose a dynamics-aware aggregator that disentangles static and dynamic information by predicting a dynamics-aware mask -- suppressing motion cues for pose estimation while amplifying them for geometry reconstruction. Extensive experiments show that PAGE-4D consistently outperforms the original VGGT in dynamic scenarios, achieving superior results in camera pose estimation, monocular and video depth estimation, and dense point map reconstruction.

---

## 7. Initialize to Generalize: A Stronger Initialization Pipeline for Sparse-View 3DGS

**论文链接:** [http://arxiv.org/abs/2510.17479v1](http://arxiv.org/abs/2510.17479v1)

**作者:** Feng Zhou, Wenkai Guo, Pu Cao, Zhicheng Zhang, Jianqin Yin

**发布时间:** 2025-10-20

**备注:** A preprint paper

### GPT解析

### 总结

这项研究针对稀疏视图3D高斯溅射(3DGS)对训练视图过拟合导致的新视图渲染伪影问题，提出了一种基于改进初始化的解决方案。研究发现初始化是决定性能的关键因素，而非训练时约束。研究团队设计了频率感知SfM、3DGS自初始化和点云正则化三种方法，显著提升了稀疏视图设置下的渲染质量，并在LLFF和Mip-NeRF360数据集上验证了其有效性。

### 背景

稀疏视图3D高斯溅射(3DGS)技术通常对训练视图过拟合，导致在新视图渲染时出现模糊等伪影。先前的研究主要通过增强初始化（即改进来自运动结构SfM的点云）或添加训练时约束（正则化）来解决这个问题。

### 目的

本研究旨在解决稀疏视图3DGS的过拟合问题，通过改进初始化策略而非依赖训练时约束，提升新视图渲染质量，消除伪影。

### 方法

研究团队设计了三种方法来改进初始化：(i) 频率感知SfM：通过低频视图增强和放宽的多视图对应关系提高低纹理区域的覆盖率；(ii) 3DGS自初始化：将光度监督提升为附加点，用学习的高斯中心补偿SfM未能充分覆盖的区域；(iii) 点云正则化：通过简单的几何/可见性先验强制执行多视图一致性和均匀空间覆盖，产生干净可靠的点云。

### 主要发现

通过对照实验发现，初始化是决定稀疏视图3DGS性能的关键因素，它决定了可达到的性能范围，而训练时约束只能在该范围内带来适度改进，且需要额外计算成本。

### 结论

本研究提出的三种初始化改进方法在稀疏视图设置下展示了一致的性能提升，确立了一种更强初始化策略的有效性。代码已在GitHub开源，可供进一步研究和应用。

### 翻译

稀疏视图3D高斯溅射(3DGS)通常对训练视图过拟合，导致新视图渲染时出现模糊等伪影。先前的研究通过增强初始化（即来自运动结构SfM的点云）或添加训练时约束（正则化）来解决3DGS优化问题。然而，我们的对照实验揭示初始化是决定性因素：它决定了稀疏视图3DGS可达到的性能范围，而训练时约束只能在该范围内带来适度改进且需要额外成本。鉴于初始化的首要地位，我们将设计重点放在那里。尽管SfM由于其依赖特征匹配在稀疏视图下表现不佳，但它仍能提供可靠的种子点。因此，我们在SfM基础上努力尽可能全面地补充它未能覆盖的区域。具体而言，我们设计了：(i)频率感知SfM，通过低频视图增强和放宽的多视图对应关系提高低纹理覆盖率；(ii)3DGS自初始化，将光度监督提升为附加点，用学习的高斯中心补偿SfM稀疏区域；(iii)点云正则化，通过简单的几何/可见性先验强制执行多视图一致性和均匀空间覆盖，产生干净可靠的点云。我们在LLFF和Mip-NeRF360上的实验展示了在稀疏视图设置中的一致性改进，确立了我们的方法作为一种更强的初始化策略。代码可在https://github.com/zss171999645/ItG-GS获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决稀疏视图3D高斯泼溅（3DGS）对训练视图过拟合的问题，导致在渲染新视图时出现模糊等伪影。这个问题在现实中非常重要，因为虚拟现实、增强现实、自由视点视频和数字内容创作等应用都需要从有限视点生成逼真新视图的能力。在实际场景中，由于硬件限制、成本或捕获条件，我们通常只能获取有限数量的视点图像，因此解决稀疏视图下的过拟合问题能显著提升这些应用的质量和实用性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先通过受控实验发现初始化质量是决定稀疏视图3DGS性能的关键因素，而训练时正则化只能提供有限的改进。基于这一发现，他们将设计重点放在初始化阶段。他们借鉴了现有的SfM算法，尽管它在稀疏视图下表现不佳但仍能提供可靠的种子点。具体来说，他们参考了EAP-GS的工作，通过将最小轨道匹配要求从三个视图降低到两个视图来增加初始点集密度。此外，他们还利用了3DGS自身的学习信号，设计了自初始化方法，将像素级的光度约束转化为额外的3D点。最后，他们引入了点云正则化技术，结合了几何和可见性先验的概念来优化点云质量。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过增强初始化点云的质量和覆盖范围来解决稀疏视图3DGS的过拟合问题。整体实现流程分为三个阶段：1）低频感知SfM：通过掩码高频图像区域，在增强的双图像集上执行SfM，从而改善低纹理区域的覆盖；2）3DGS自初始化：在输入视图上训练一个轻量级3DGS模型，并将所有原始高斯中心作为新的点云重用，这能补偿图像特征不足的区域；3）点云正则化：通过单视图点过滤（去除深度模糊的点）、聚类去噪（减少不稳定点和重复点）以及基于法线的一致性过滤（去除几何不一致的点），最终产生一个干净可靠的点云作为3DGS的初始化输入。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有三个：1）频率感知SfM：通过掩码高频区域和放松多视图对应要求，显著改善低纹理区域的覆盖；2）3DGS自初始化：创新性地利用3DGS的学习信号，将像素级光度约束提升为额外的3D点，弥补了传统SfM在弱纹理区域的不足；3）点云正则化：结合单视图过滤、聚类去噪和基于法线的一致性过滤，产生干净可靠的点云。相比之前的工作，本文的不同之处在于：首先，它首次系统性地证明了初始化质量比训练时正则化对稀疏视图3DGS的性能影响更大；其次，它不是简单地改进单一组件，而是设计了一个完整的三阶段初始化管道；最后，它能够与现有的正则化方法（如DropGS）结合使用，进一步提升性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种三阶段初始化管道，通过显著增强稀疏视图3DGS的初始点云质量和覆盖范围，有效解决了过拟合问题，大幅提升了新视图合成的质量和泛化能力。'}


### 论文摘要

Sparse-view 3D Gaussian Splatting (3DGS) often overfits to the training views, leading to artifacts like blurring in novel view rendering. Prior work addresses it either by enhancing the initialization (\emph{i.e.}, the point cloud from Structure-from-Motion (SfM)) or by adding training-time constraints (regularization) to the 3DGS optimization. Yet our controlled ablations reveal that initialization is the decisive factor: it determines the attainable performance band in sparse-view 3DGS, while training-time constraints yield only modest within-band improvements at extra cost. Given initialization's primacy, we focus our design there. Although SfM performs poorly under sparse views due to its reliance on feature matching, it still provides reliable seed points. Thus, building on SfM, our effort aims to supplement the regions it fails to cover as comprehensively as possible. Specifically, we design: (i) frequency-aware SfM that improves low-texture coverage via low-frequency view augmentation and relaxed multi-view correspondences; (ii) 3DGS self-initialization that lifts photometric supervision into additional points, compensating SfM-sparse regions with learned Gaussian centers; and (iii) point-cloud regularization that enforces multi-view consistency and uniform spatial coverage through simple geometric/visibility priors, yielding a clean and reliable point cloud. Our experiments on LLFF and Mip-NeRF360 demonstrate consistent gains in sparse-view settings, establishing our approach as a stronger initialization strategy. Code is available at https://github.com/zss171999645/ItG-GS.

---

## 8. 论文ID: 2510.17237v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.17237v1.json'

---

## 9. ProDAT: Progressive Density-Aware Tail-Drop for Point Cloud Coding

**论文链接:** [http://arxiv.org/abs/2510.17068v1](http://arxiv.org/abs/2510.17068v1)

**作者:** Zhe Luo, Wenjing Jia, Stuart Perry

**发布时间:** 2025-10-20

### GPT解析

### 总结

本文提出了一种名为ProDAT的新型渐进式点云编码方法，通过密度感知尾部丢弃机制实现多比特率下的渐进式解码，并在编码效率上显著优于现有方法。

### 背景

三维点云在自动驾驶、增强现实和沉浸式通信等应用中日益重要，需要实时处理和低延迟。但大数据量和带宽限制在资源有限环境中阻碍了高质量服务的部署。

### 目的

解决现有基于学习的点云几何编码方法固定潜在表示不支持渐进式解码的问题，实现高效的渐进式点云编码。

### 方法

提出ProDAT，一种密度感知尾部丢弃机制，通过利用密度信息作为指导信号，使潜在特征和坐标根据其重要性进行自适应解码，从而使用单个模型在多个比特率下实现渐进式解码。

### 主要发现

在基准数据集上的实验结果表明，ProDAT不仅能够实现渐进式编码，而且与最先进的基于学习的编码技术相比，实现了更高的编码效率。在SemanticKITTI上，PSNR-D2的BD-rate改进超过28.6%，在ShapeNet上超过18.15%。

### 结论

ProDAT成功填补了现有方法不支持渐进式解码的空白，同时显著提高了点云编码的效率。

### 翻译

三维（3D）点云在自动驾驶、增强现实和沉浸式通信等应用中变得越来越重要，要求实时处理和低延迟。然而，其大数据量和带宽限制阻碍了在资源有限环境中部署高质量服务。渐进式编码允许在不同细节级别解码，通过允许初始部分解码随后进行细化提供了一种替代方案。尽管最近基于学习的点云几何编码方法取得了显著成功，但其固定的潜在表示不支持渐进式解码。为了填补这一空白，我们提出了ProDAT，一种用于渐进式点云编码的新型密度感知尾部丢弃机制。通过利用密度信息作为指导信号，潜在特征和坐标根据其重要性进行自适应解码，从而使用单个模型在多个比特率下实现渐进式解码。在基准数据集上的实验结果表明，所提出的ProDAT不仅能够实现渐进式编码，而且与最先进的基于学习的编码技术相比，实现了更高的编码效率，在SemanticKITTI上PSNR-D2的BD-rate改进超过28.6%，在ShapeNet上超过18.15%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云编码中不支持渐进式解码的问题。这个问题在现实中很重要，因为点云数据在自动驾驶、增强现实和沉浸式通信等应用中需求日益增长，但这些应用需要实时处理和低延迟。现有的学习点云编码方法产生单一比特流，必须完全解码才能重建，无法满足带宽受限环境下的渐进式质量提升需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了点云编码的现状，发现传统方法在大规模数据集上效率低，而学习-based方法虽性能好但不支持渐进式解码。他们借鉴了2D图像和视频领域的渐进式编码技术（如JPEG、JPEG2000和H.264 SVC），并参考了密度保留点云压缩方法和图像编码中的尾部丢弃技术。通过结合这些思想，作者设计了密度感知的尾部丢弃机制，利用点云密度信息指导特征选择，实现渐进式解码。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用密度信息作为指导信号，高密度区域通常表示复杂几何结构，稀疏区域对应简单结构，通过密度感知的尾部丢弃机制优先保留重要特征。整体流程包括：1)编码阶段提取特征并应用密度感知的尾部丢弃；2)根据渐进比例选择保留特征；3)解码器重建点云；4)训练时使用随机丢弃比例使模型适应不同级别的特征完整性，损失函数结合几何质量、密度保持和比特率约束。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)单一训练阶段的渐进式点云编码；2)密度感知的尾部丢弃机制；3)联合特征和坐标丢弃策略；4)动态密度归一化方法；5)结合全局方差和局部梯度的通道重要性计算。相比之前的工作，ProDAT不仅支持渐进式解码，还通过密度信息优化特征选择，在保持高质量重建的同时提高了编码效率，特别适合大规模点云数据，如SemanticKITTI和ShapeNet数据集。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ProDAT通过引入密度感知的尾部丢弃机制，实现了单一训练阶段的渐进式点云编码，在保持高质量重建的同时显著提高了编码效率和灵活性，特别适合资源受限的实时应用场景。'}


### 论文摘要

Three-dimensional (3D) point clouds are becoming increasingly vital in applications such as autonomous driving, augmented reality, and immersive communication, demanding real-time processing and low latency. However, their large data volumes and bandwidth constraints hinder the deployment of high-quality services in resource-limited environments. Progres- sive coding, which allows for decoding at varying levels of detail, provides an alternative by allowing initial partial decoding with subsequent refinement. Although recent learning-based point cloud geometry coding methods have achieved notable success, their fixed latent representation does not support progressive decoding. To bridge this gap, we propose ProDAT, a novel density-aware tail-drop mechanism for progressive point cloud coding. By leveraging density information as a guidance signal, latent features and coordinates are decoded adaptively based on their significance, therefore achieving progressive decoding at multiple bitrates using one single model. Experimental results on benchmark datasets show that the proposed ProDAT not only enables progressive coding but also achieves superior coding efficiency compared to state-of-the-art learning-based coding techniques, with over 28.6% BD-rate improvement for PSNR- D2 on SemanticKITTI and over 18.15% for ShapeNet

---

## 10. Registration is a Powerful Rotation-Invariance Learner for 3D Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2510.16865v1](http://arxiv.org/abs/2510.16865v1)

**作者:** Yuyang Yu, Zhengwei Chen, Xuemiao Xu, Lei Zhang, Haoxin Yang, Yongwei Nie, Shengfeng He

**发布时间:** 2025-10-19

### GPT解析

### 总结

本研究提出了一种基于配准的旋转不变性特征提取框架，用于点云数据中的3D异常检测，解决了现有方法在特征转换一致性和判别能力方面的局限性。

### 背景

3D点云异常检测对工业质量控制至关重要，但当前基于内存库的方法常面临特征转换不一致和判别能力有限的问题，特别是在捕捉局部几何细节和实现旋转不变性方面。当配准失败时，这些问题更加突出，导致不可靠的检测结果。

### 目的

开发一种能够同时解决点云配准和异常检测问题的框架，通过整合这两个任务的目标，实现旋转不变且局部判别性强的特征提取，提高异常检测的可靠性和有效性。

### 方法

提出了一种配准诱导的旋转不变性特征提取框架，将点云配准和基于内存的异常检测目标整合在一起。通过将特征提取嵌入到配准学习过程中，框架联合优化对齐和表示学习，使网络获得对旋转鲁棒且对异常检测高效的特征。

### 主要发现

点云配准不仅在几何结构对齐中起关键作用，还引导特征提取实现旋转不变性和局部判别性表示。两个任务都依赖于建模局部几何结构和利用样本间的特征相似性。在Anomaly-ShapeNet和Real3D-AD数据集上的实验表明，该方法在有效性和泛化能力上一致优于现有方法。

### 结论

通过将特征提取与配准学习过程相结合，所提出的框架能够获得既对旋转具有鲁棒性又对异常检测高效的特征，显著提高了点云异常检测的性能和可靠性。

### 翻译

点云数据中的3D异常检测对工业质量控制至关重要，旨在以高可靠性识别结构缺陷。然而，当前基于内存库的方法常常遭受特征转换不一致和判别能力有限的困扰，特别是在捕捉局部几何细节和实现旋转不变性方面。当配准失败时，这些局限性变得更加明显，导致不可靠的检测结果。我们认为点云配准不仅在几何结构对齐中起着关键作用，还引导特征提取朝向旋转不变和局部判别性表示方向发展。为此，我们提出了一种基于配准的旋转不变性特征提取框架，整合了点云配准和基于内存的异常检测目标。我们的关键见解是，这两个任务都依赖于建模局部几何结构和利用样本间的特征相似性。通过将特征提取嵌入到配准学习过程中，我们的框架联合优化了对齐和表示学习。这种整合使网络能够获得对旋转具有鲁棒性且对异常检测非常有效的特征。在Anomaly-ShapeNet和Real3D-AD数据集上的大量实验表明，我们的方法在有效性和泛化能力上一致优于现有方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D点云异常检测中的两个关键问题：现有基于内存库的方法在特征变换方面存在不一致性，以及在捕捉局部几何细节和实现旋转不变性方面能力有限。这个问题在现实中非常重要，因为3D异常检测对工业质量控制至关重要，可以可靠地识别结构缺陷。当物体以不同方向呈现时，旋转不变性变得尤为重要，而当前方法在注册失败时会产生不可靠的检测结果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析点云注册和基于内存库的异常检测之间的内在联系，发现这两个任务都依赖于对局部几何结构的建模和跨样本的特征相似性利用。他们提出将注册作为特征学习过程的一部分，而不是独立的预处理模块。作者借鉴了RIConv++、KPConv-FPN和Geometric Transformer等现有组件，应用了粗到细的点云注册策略、最优传输算法、RANSAC技术和内存库采样技术，但将它们整合到一个新的统一框架中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点云注册过程整合到异常检测的特征学习中，通过注册任务强制源点云和目标点云之间的几何对齐和多尺度特征一致性，使网络能够获得对旋转具有鲁棒性且对异常检测有效的特征。整体流程分为两个阶段：1)注册诱导的特征学习阶段：生成变换点云对、多尺度下采样、构建补丁匹配、提取多尺度特征，并通过三个损失函数联合优化；2)注册诱导的异常检测阶段：点对齐、特征归一化和内存库构建、特征过滤和异常分数计算。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：将点云注册作为特征学习的一部分而非独立预处理步骤；提出统一的Reg2Inv框架用于注册诱导的旋转不变特征提取；联合优化原型-样本对齐和基于鲁棒局部几何特征的异常评分；通过注册任务同时学习旋转不变和局部判别性表示。相比之前工作，本文方法将注册整合到特征学习中而非作为预处理；保留了细粒度局部几何而非仅关注全局语义；在注册不完美时仍保持鲁棒性；同时优化空间对齐和特征学习而非分开处理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种将点云注册整合到特征学习过程中的统一框架Reg2Inv，通过联合优化几何对齐和多尺度特征一致性，实现了对旋转具有鲁棒性且对异常检测高度有效的3D点云特征提取，显著提升了工业质量控制中的异常检测性能。'}


### 论文摘要

3D anomaly detection in point-cloud data is critical for industrial quality control, aiming to identify structural defects with high reliability. However, current memory bank-based methods often suffer from inconsistent feature transformations and limited discriminative capacity, particularly in capturing local geometric details and achieving rotation invariance. These limitations become more pronounced when registration fails, leading to unreliable detection results. We argue that point-cloud registration plays an essential role not only in aligning geometric structures but also in guiding feature extraction toward rotation-invariant and locally discriminative representations. To this end, we propose a registration-induced, rotation-invariant feature extraction framework that integrates the objectives of point-cloud registration and memory-based anomaly detection. Our key insight is that both tasks rely on modeling local geometric structures and leveraging feature similarity across samples. By embedding feature extraction into the registration learning process, our framework jointly optimizes alignment and representation learning. This integration enables the network to acquire features that are both robust to rotations and highly effective for anomaly detection. Extensive experiments on the Anomaly-ShapeNet and Real3D-AD datasets demonstrate that our method consistently outperforms existing approaches in effectiveness and generalizability.

---

## 11. Rotation, Scale, and Translation Resilient Black-box Fingerprinting for Intellectual Property Protection of EaaS Models

**论文链接:** [http://arxiv.org/abs/2510.16706v1](http://arxiv.org/abs/2510.16706v1)

**作者:** Hongjie Zhang, Zhiqi Zhao, Hanzhou Wu, Zhihua Xia, Athanasios V. Vasilakos

**发布时间:** 2025-10-19

### GPT解析

### 总结

该研究提出了一种用于EaaS（嵌入即服务）模型的指纹框架，通过分析嵌入空间拓扑结构的几何特性来验证模型所有权，而非传统水印技术。该方法能有效抵抗旋转、缩放和平移（RST）攻击，并在视觉和文本嵌入任务中验证了其优越性。

### 背景

特征嵌入已成为处理高维复杂数据的核心技术，导致EaaS模型在云环境中广泛部署。现有水印技术通过修改训练样本或网络参数注入后门触发器来保护知识产权，但这些方法易受语义分析和几何变换攻击的影响。

### 目的

解决现有EaaS模型水印技术易受语义分析和几何变换攻击的问题，开发一种更鲁棒的模型所有权验证方法。

### 方法

提出一种指纹框架，将受害者模型和可疑模型的嵌入建模为点云，执行鲁棒的空间对齐和相似度测量，通过分析嵌入空间拓扑结构的几何特性建立EaaS模型所有权，而非依赖修改的训练样本或触发器。

### 主要发现

将嵌入建模为点云的方法实现了鲁棒的空间对齐和相似度测量；该方法能有效抵抗RST攻击；在视觉和文本嵌入任务中验证了该方法的优越性和适用性。

### 结论

该研究揭示了EaaS模型的固有特性，并为黑盒场景下EaaS模型的所有权验证提供了一种有前景的解决方案，通过几何分析而非传统水印技术提供更鲁棒的验证方法。

### 翻译

特征嵌入已成为处理高维和复杂数据的核心技术，这使得嵌入即服务（EaaS）模型已在云环境中广泛部署。为了保护EaaS模型的知识产权，现有方法应用数字水印技术，通过修改训练样本或网络参数向EaaS模型注入特定的后门触发器。然而，这些方法不可避免地会产生可通过语义分析检测到的模式，并且容易受到几何变换（包括旋转、缩放和平移，RST）的影响。为了解决这个问题，我们提出了一种用于EaaS模型的指纹框架，而非仅仅改进现有的水印技术。与水印技术不同，所提出的方法通过分析嵌入空间拓扑结构的几何特性来建立EaaS模型所有权，而不是依赖修改的训练样本或触发器。关键创新在于将受害者模型和可疑模型的嵌入建模为点云，使我们能够执行鲁棒的空间对齐和相似度测量，这 inherently 抵抗RST攻击。在视觉和文本嵌入任务上评估的实验结果验证了其优越性和适用性。这项研究揭示了EaaS模型的固有特性，并为黑盒场景下EaaS模型的所有权验证提供了一种有前景的解决方案。


### 论文摘要

Feature embedding has become a cornerstone technology for processing high-dimensional and complex data, which results in that Embedding as a Service (EaaS) models have been widely deployed in the cloud. To protect the intellectual property of EaaS models, existing methods apply digital watermarking to inject specific backdoor triggers into EaaS models by modifying training samples or network parameters. However, these methods inevitably produce detectable patterns through semantic analysis and exhibit susceptibility to geometric transformations including rotation, scaling, and translation (RST). To address this problem, we propose a fingerprinting framework for EaaS models, rather than merely refining existing watermarking techniques. Different from watermarking techniques, the proposed method establishes EaaS model ownership through geometric analysis of embedding space's topological structure, rather than relying on the modified training samples or triggers. The key innovation lies in modeling the victim and suspicious embeddings as point clouds, allowing us to perform robust spatial alignment and similarity measurement, which inherently resists RST attacks. Experimental results evaluated on visual and textual embedding tasks verify the superiority and applicability. This research reveals inherent characteristics of EaaS models and provides a promising solution for ownership verification of EaaS models under the black-box scenario.

---

## 12. A Bayesian Framework for Symmetry Inference in Chaotic Attractors

**论文链接:** [http://arxiv.org/abs/2510.16509v1](http://arxiv.org/abs/2510.16509v1)

**作者:** Ziad Ghanem, Chang Hyunwoong, Preskella Mrad

**发布时间:** 2025-10-18

### GPT解析

### 总结

本文提出了一种贝叶斯框架，用于从动力系统轨迹数据中检测对称性，解决了现有方法缺乏不确定性量化的问题，提高了对噪声的鲁棒性，并能够处理层次对称结构。

### 背景

对称性检测是信号分析中的基本问题，能揭示底层结构和约束。当数据表现为动力系统轨迹时，对称性编码了动力系统的结构特性，实现模型简化、条件间比较和状态变化检测。现有最优传输方法依赖确定性阈值且缺乏不确定性量化，限制了鲁棒性和层次对称结构的解决能力。

### 目的

开发一个贝叶斯框架，将对称性检测构建为在候选子群格上的概率模型选择，使用基于Wasserstein距离的Gibbs后验，以解决不确定性量化问题，提高对噪声的鲁棒性，并处理层次对称结构。

### 方法

提出贝叶斯框架，将对称性检测表述为候选子群格上的概率模型选择，使用基于Wasserstein距离的Gibbs后验。通过Metropolis-Hastings采样进行后验推断，建立在三个理论保证上：贝叶斯奥卡姆剃刀原理、共轭等变性和扰动下的稳定性界限。

### 主要发现

建立了三个理论保证：(i)贝叶斯奥卡姆剃刀原理，倾向于与数据一致的最小对称性；(ii)共轭等变性确保框架独立性；(iii)扰动下的稳定性界限确保对噪声的鲁棒性。数值实验展示了高噪声和小样本量下准确的对称性恢复，应用人类步行动力学揭示了机械约束引起的对称性变化。

### 结论

该贝叶斯框架有效解决了对称性检测中的不确定性量化问题，提高了对噪声的鲁棒性，并能够处理层次对称结构，在生物力学和动力系统的统计推断中具有实用价值。

### 翻译

从数据中检测对称性是信号分析中的一个基本问题，它为底层结构和约束提供了见解。当数据表现为动力系统的轨迹时，对称性编码了动力系统的结构特性，这些特性能够实现模型简化、条件间的有原则比较以及检测状态变化。虽然最近的最优传输方法为这种情况下的数据驱动对称性检测提供了实用工具，但它们依赖于确定性阈值且缺乏不确定性量化，限制了它们对噪声的鲁棒性以及解决层次对称结构的能力。我们提出了一个贝叶斯框架，将对称性检测构建为在候选子群格上的概率模型选择，使用基于观测数据与群变换副本之间Wasserstein距离构建的Gibbs后验。我们建立了三个理论保证：(i)贝叶斯奥卡姆剃刀原理，倾向于与数据一致的最小对称性；(ii)共轭等变性确保了框架独立性；(iii)扰动下的稳定性界限确保了对噪声的鲁棒性。后验推断通过Metropolis-Hastings采样进行，在等变动力系统和合成点云上的数值实验展示了在高噪声和小样本量下准确的对称性恢复。在人类步行动力学的应用中揭示了机械约束引起的对称性变化，证明了该框架在生物力学和动力系统统计推断中的实用性。


### 论文摘要

Detecting symmetry from data is a fundamental problem in signal analysis, providing insight into underlying structure and constraints. When data emerge as trajectories of dynamical systems, symmetries encode structural properties of the dynamics that enable model reduction, principled comparison across conditions, and detection of regime changes. While recent optimal transport methods provide practical tools for data-driven symmetry detection in this setting, they rely on deterministic thresholds and lack uncertainty quantification, limiting robustness to noise and ability to resolve hierarchical symmetry structures. We present a Bayesian framework that formulates symmetry detection as probabilistic model selection over a lattice of candidate subgroups, using a Gibbs posterior constructed from Wasserstein distances between observed data and group-transformed copies. We establish three theoretical guarantees: $(i)$ a Bayesian Occam's razor favoring minimal symmetry consistent with data, $(ii)$ conjugation equivariance ensuring frame-independence, and $(iii)$ stability bounds under perturbations for robustness to noise. Posterior inference is performed via Metropolis-Hastings sampling and numerical experiments on equivariant dynamical systems and synthetic point clouds demonstrate accurate symmetry recovery under high noise and small sample sizes. An application to human gait dynamics reveals symmetry changes induced by mechanical constraints, demonstrating the framework's utility for statistical inference in biomechanical and dynamical systems.

---

## 13. MNO: Multiscale Neural Operator for Computational Fluid Dynamics with 3D Point Cloud Data

**论文链接:** [http://arxiv.org/abs/2510.16071v1](http://arxiv.org/abs/2510.16071v1)

**作者:** Qinxuan Wang, Chuang Wang, Mingyu Zhang, Jingwei Sun, Peipei Yang, Shuo Tang, Shiming Xiang

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了一种多尺度神经网络算子(MNO)架构，用于三维非结构化点云上的计算流体动力学，通过分解三个尺度的信息来提高精度和可扩展性，在多个基准测试中表现优异。

### 背景

现有神经网络算子在求解偏微分方程时虽然比传统求解器快几个数量级，但在精度和可扩展性方面仍存在局限，特别是在不规则域上具有丰富多尺度结构的流体流动问题中。

### 目的

引入多尺度神经网络算子(MNO)架构，解决三维非结构化点云上的计算流体动力学问题，提高预测精度和可扩展性。

### 方法

MNO明确分解三个尺度的信息：全局维度缩减注意力模块处理长程依赖关系，局部图注意力模块处理邻域级相互作用，微观逐点注意力模块处理细粒度细节，这种设计保留了多尺度归纳偏置同时保持计算效率。

### 主要发现

在四个涵盖稳态和非稳态流动场景的基准测试上(最多30万个点)，MNO始终优于最先进的基线方法，减少5%到40%的预测误差，并在具有挑战性的三维CFD问题中表现出更强的鲁棒性。

### 结论

明确的多尺度设计对神经网络算子至关重要，MNO可作为不规则域上学习复杂流体动力学的可扩展框架。

### 翻译

神经网络算子已成为求解偏微分方程的强大数据驱动范式，比传统求解器快几个数量级。然而，现有方法在精度和可扩展性方面仍然有限，特别是在流体流动表现出丰富多尺度结构的不规则域上。在这项工作中，我们引入了多尺度神经网络算子(MNO)，这是一种用于三维非结构化点云上计算流体动力学(CFD)的新架构。MNO明确分解三个尺度的信息：全局维度缩减注意力模块用于长程依赖关系，局部图注意力模块用于邻域级相互作用，微观逐点注意力模块用于细粒度细节。这种设计保留了多尺度归纳偏置，同时保持计算效率。我们在四个不同的基准测试上评估了MNO，涵盖了最多30万个点的稳态和非稳态流动场景。在所有任务中，MNO始终优于最先进的基线方法，减少5%到40%的预测误差，并在具有挑战性的三维CFD问题中表现出改进的鲁棒性。我们的结果强调了明确的多尺度设计对神经网络算子的重要性，并将MNO确立为不规则域上学习复杂流体动力学的可扩展框架。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有神经算子在计算流体动力学(CFD)中处理不规则域时的局限，特别是对流体流动中丰富多尺度结构的建模不足问题。这个问题很重要，因为CFD在工程设计、气象预测等领域有广泛应用，而传统CFD计算成本高，难以实现实时计算；神经算子虽能提供更快速度，但在精度上仍落后于传统方法，特别是在处理复杂几何形状和动态域时。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到物理量在流场中表现出强烈的多尺度效应（大尺度全局趋势、局部相互作用和细粒度点变化），而现有方法要么过于依赖全局建模牺牲局部细节，要么细粒度注意力机制计算成本过高。因此设计了MNO，包含三个互补的并行模块分别处理不同尺度信息。该方法借鉴了Transolver和LNO的低秩投影策略（全局模块）、Point Transformer的思想（局部模块），以及Encoder-MNO-Decoder的常见架构，但针对点云数据和多尺度特性进行了专门设计。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是显式地将信息分解为全局、局部和微观三个尺度，通过三个并行注意力模块分别捕获不同尺度的特征，并在每个块后融合这些特征。整体流程包括：1)输入处理接收3D点云数据；2)编码器将输入嵌入到潜在标记空间；3)MNO块处理（包含全局维度收缩注意力、局部图注意力和微观点级注意力三个并行模块）；4)特征融合三个模块的输出；5)解码器将处理后的特征映射回目标物理量；6)输出预测流场中的关键物理量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个针对3D无结构点云的多尺度神经算子；2)显式的多尺度分解设计，包含三个互补的注意力模块；3)直接在点云上处理，避免网格约束；4)统一框架同时提取全局、局部和细粒度流场表示。相比之前工作的不同：与规则域方法相比，MNO可直接处理不规则域；与不规则域方法相比，MNO明确考虑多尺度特性并提供更平衡的表示能力；与多尺度方法相比，MNO直接在点云上工作且不使用重复下采样/上采样，避免了信息丢失。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MNO通过显式的多尺度注意力机制设计，首次实现了在3D点云数据上高效准确地捕获流体流动的全局趋势、局部相互作用和细粒度细节，显著提高了不规则域上计算流体动力学任务的预测精度和鲁棒性。'}


### 论文摘要

Neural operators have emerged as a powerful data-driven paradigm for solving Partial Differential Equations (PDEs), offering orders-of-magnitude acceleration over traditional solvers. However, existing approaches still suffer from limited accuracy and scalability, particularly on irregular domains where fluid flows exhibit rich multiscale structures. In this work, we introduce the Multiscale Neural Operator (MNO), a new architecture for Computational Fluid Dynamics (CFD) on three-dimensional (3D) unstructured point clouds. MNO explicitly decomposes information across three scales: a global dimension-shrinkage attention module for long-range dependencies, a local graph attention module for neighborhood-level interactions, and a micro point-wise attention module for fine-grained details. This design preserves multiscale inductive biases while remaining computationally efficient. We evaluate MNO on four diverse benchmarks, covering both steady-state and unsteady flow scenarios with up to 300K points. Across all tasks, MNO consistently outperforms state-of-the-art baselines, reducing prediction errors by 5% to 40% and demonstrating improved robustness in challenging 3D CFD problems. Our results highlight the importance of explicit multiscale design for neural operators and establish MNO as a scalable framework for learning complex fluid dynamics on irregular domains.

---

## 14. M2H: Multi-Task Learning with Efficient Window-Based Cross-Task Attention for Monocular Spatial Perception

**论文链接:** [http://arxiv.org/abs/2510.17363v1](http://arxiv.org/abs/2510.17363v1)

**作者:** U. V. B. L Udugama, George Vosselman, Francesco Nex

**发布时间:** 2025-10-20

**备注:** Accepted to the IEEE/RSJ International Conference on Intelligent  Robots and Systems (IROS 2025). 8 pages, 7 figures

### GPT解析

### 总结

论文提出了一种名为Multi-Mono-Hydra (M2H)的新型多任务学习框架，用于从单目图像中进行语义分割和深度、边缘及表面法线估计，在边缘设备上实现高效实时空间感知。

### 背景

在边缘设备上部署实时空间感知需要高效的多任务模型，这些模型需要能够利用互补任务信息同时最小化计算开销。

### 目的

开发一种优化的多任务学习框架，支持从单目图像中同时进行多种空间感知任务，并能在边缘设备上实时运行，为动态环境中的3D场景图构建提供基础。

### 方法

M2H框架采用基于窗口的跨任务注意模块实现结构化特征交换，同时保留任务特定细节，提高预测一致性。框架基于轻量级ViT-based DINOv2主干网络构建，优化了实时部署性能。

### 主要发现

M2H在NYUDv2数据集上超越最先进的多任务模型；在Hypersim上超越单任务深度和语义基线；在Cityscapes数据集上表现优异；同时保持笔记本电脑硬件上的计算效率；在真实世界数据验证中展现出实用性。

### 结论

M2H通过创新的跨任务注意机制有效利用了任务间的互补信息，在多个基准测试和实际应用中表现出色，是一种适用于边缘设备的高效多任务空间感知解决方案。

### 翻译

在边缘设备上部署实时空间感知需要高效的多任务模型，这些模型能够利用互补任务信息同时最小化计算开销。本文介绍了Multi-Mono-Hydra (M2H)，一种新颖的多任务学习框架，专为从单目图像进行语义分割和深度、边缘及表面法线估计而设计。与传统依赖独立单任务模型或共享编码器-解码器架构的方法不同，M2H引入了基于窗口的跨任务注意模块，能够在保留任务特定细节的同时实现结构化特征交换，提高跨任务预测一致性。基于轻量级ViT-based DINOv2主干网络构建，M2H针对实时部署进行优化，并作为支持动态环境中3D场景图构建的单目空间感知系统的基础。全面评估显示，M2H在NYUDv2上超越最先进的多任务模型，在Hypersim上超越单任务深度和语义基线，在Cityscapes数据集上实现卓越性能，同时在笔记本电脑硬件上保持计算效率。除了基准测试外，M2H还在真实世界数据上得到验证，展示了其在空间感知任务中的实用性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何在边缘设备上实现高效的多任务空间感知问题，特别是从单目图像同时进行语义分割和深度、边缘、表面法线估计。这个问题在现实中非常重要，因为自动驾驶系统、增强现实和机器人感知等应用需要实时理解环境，而边缘设备资源有限，需要高效的多任务模型来利用任务间的互补信息，同时最小化计算开销。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者基于Taskonomy的研究，认识到任务间关系可以减少监督需求并提高泛化能力。他们发现现有方法存在局限：局部交互保留了效率但限制了协同效应，全局注意力提供了丰富上下文但计算开销高。因此设计了一种平衡效率与协同效应的框架。作者借鉴了DINOv2的轻量级ViT主干网络、Swin Transformer的窗口注意力机制，并参考了注意力机制在其他多任务学习方法中的应用，如PAD-Net和MTI-Net。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是引入基于窗口的跨任务注意力模块实现结构化特征交换，同时保留任务特定细节，并使用轻量级ViT主干网络优化实时部署。整体流程：1) DINOv2编码器提取多尺度令牌表示；2) 通过MSTR块重组为空间特征图；3) MSF块生成任务特定特征；4) 双路径细化：WMCA捕获局部跨任务交互，GGFM聚合全局上下文；5) 融合局部和全局特征；6) 专用解码器头生成最终预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 窗口多任务交叉注意力(WMCA)模块，在局部窗口内高效交换信息；2) 全局门控特征合并(GGFM)模块，使用门控机制聚合全局上下文；3) 双路径细化策略，平衡特征交换深度与计算效率；4) 使用动态权重平均(DWA)平衡跨任务学习。相比之前工作，M2H在保持计算效率的同时实现了更好的性能，超越了多种最先进方法，特别适合边缘设备上的实时应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'M2H通过创新的窗口化跨任务注意力和双路径特征融合机制，实现了在边缘设备上高效运行的多任务空间感知系统，在保持计算效率的同时超越了现有最先进方法在多个任务和数据集上的性能。'}


### 论文摘要

Deploying real-time spatial perception on edge devices requires efficient multi-task models that leverage complementary task information while minimizing computational overhead. This paper introduces Multi-Mono-Hydra (M2H), a novel multi-task learning framework designed for semantic segmentation and depth, edge, and surface normal estimation from a single monocular image. Unlike conventional approaches that rely on independent single-task models or shared encoder-decoder architectures, M2H introduces a Window-Based Cross-Task Attention Module that enables structured feature exchange while preserving task-specific details, improving prediction consistency across tasks. Built on a lightweight ViT-based DINOv2 backbone, M2H is optimized for real-time deployment and serves as the foundation for monocular spatial perception systems supporting 3D scene graph construction in dynamic environments. Comprehensive evaluations show that M2H outperforms state-of-the-art multi-task models on NYUDv2, surpasses single-task depth and semantic baselines on Hypersim, and achieves superior performance on the Cityscapes dataset, all while maintaining computational efficiency on laptop hardware. Beyond benchmarks, M2H is validated on real-world data, demonstrating its practicality in spatial perception tasks.

---

## 15. Enhanced Motion Forecasting with Plug-and-Play Multimodal Large Language Models

**论文链接:** [http://arxiv.org/abs/2510.17274v1](http://arxiv.org/abs/2510.17274v1)

**作者:** Katie Luo, Jingwei Ji, Tong He, Runsheng Xu, Yichen Xie, Dragomir Anguelov, Mingxing Tan

**发布时间:** 2025-10-20

**备注:** In proceedings of IROS 2025

### GPT解析

### 总结

本文提出了一种名为'即插即预测'(PnF)的即插即用方法，通过多模态大语言模型增强现有运动预测模型，提高自动驾驶系统在多样化实际场景中的泛化能力。

### 背景

当前自动驾驶系统依赖专用模型进行感知和运动预测，在标准条件下表现可靠，但在多样化实际场景中泛化且经济高效地适应仍然是一个重大挑战。

### 目的

提出一种新方法解决自动驾驶系统在多样化实际场景中的泛化问题。

### 方法

PnF方法利用自然语言描述复杂场景的优势，设计提示从多模态大语言模型中提取结构化场景理解，并将这些信息蒸馏为可学习的嵌入，增强现有行为预测模型，无需微调即可实现性能提升。

### 主要发现

该方法利用多模态大语言模型的零样本推理能力，显著提高了运动预测性能，同时不需要微调，使其易于实际采用。

### 结论

在Waymo OpenMotion数据集和nuScenes数据集上使用两种最先进的运动预测模型验证了该方法，证明了在两个基准测试中都有一致的性能改进。

### 翻译

当前的自动驾驶系统依赖于用于感知和预测运动的专用模型，这些模型在标准条件下表现出可靠的性能。然而，泛化到多样化的实际场景并保持成本效益仍然是一个重大挑战。为此，我们提出了即插即预测(PnF)，一种即插即用的方法，通过多模态大语言模型(MLLMs)增强现有的运动预测模型。PnF基于自然语言能更有效地描述和处理复杂场景的见解，实现了针对特定行为的快速适应。我们设计了提示从MLLMs中提取结构化的场景理解，并将这些信息蒸馏为可学习的嵌入，以增强现有的行为预测模型。我们的方法利用MLLMs的零样本推理能力，显著提高了运动预测性能，同时不需要微调，使其易于实际采用。我们在Waymo OpenMotion数据集和nuScenes数据集上使用两种最先进的运动预测模型验证了我们的方法，证明了在两个基准测试中都有一致的性能改进。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决自动驾驶系统在多样化现实场景中运动预测模型的泛化能力问题。这个问题很重要，因为自动驾驶车辆不可避免会遇到训练数据中未涵盖的罕见情况(长尾案例)，而持续收集大量数据和重新训练系统的成本过高。提高系统在复杂场景下的泛化能力对确保安全性和实用性至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从多模态大语言模型(MLLM)的进展获得灵感，认识到语言能更好地描述复杂场景。他们设计提示从MLLM中提取结构化场景理解，并将其转化为可学习的嵌入来增强现有预测模型。借鉴了现有运动预测模型(如Wayformer和MotionLM)的架构，以及MLLM在其他自动驾驶应用中的使用，但创新点在于采用零样本推理而非微调方式，保留了MLLM的通用能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用MLLM的零样本推理能力增强现有运动预测模型，通过自然语言描述更有效地处理复杂场景。整体流程包括：1)视觉语义分析器(VSA)提取代理特定语义；2)场景分类器(SC)提供场景级理解；3)通过学习的信息增益机制将MLLM提取的特征选择性整合到预测模型中。整个过程作为即插即用组件，无需对MLLM进行微调。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)即插即用框架PnF；2)利用MLLM零样本推理能力；3)多模态提示设计；4)选择性信息整合机制；5)双重分析组件(代理级和场景级)。相比之前工作，PnF不需要对MLLM进行微调，保留了其通用能力；不同于直接映射原始数据的EMMA模型，PnF专注于增强运动预测任务；相比其他需要修改架构的方法，PnF可作为轻量级组件集成到现有系统。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出的Plug-and-Forecast方法通过即插即用多模态大语言模型的零样本推理能力，显著提升了自动驾驶系统中运动预测模型的性能，特别是在处理复杂和罕见场景时无需对语言模型进行微调。'}


### 论文摘要

Current autonomous driving systems rely on specialized models for perceiving and predicting motion, which demonstrate reliable performance in standard conditions. However, generalizing cost-effectively to diverse real-world scenarios remains a significant challenge. To address this, we propose Plug-and-Forecast (PnF), a plug-and-play approach that augments existing motion forecasting models with multimodal large language models (MLLMs). PnF builds on the insight that natural language provides a more effective way to describe and handle complex scenarios, enabling quick adaptation to targeted behaviors. We design prompts to extract structured scene understanding from MLLMs and distill this information into learnable embeddings to augment existing behavior prediction models. Our method leverages the zero-shot reasoning capabilities of MLLMs to achieve significant improvements in motion prediction performance, while requiring no fine-tuning -- making it practical to adopt. We validate our approach on two state-of-the-art motion forecasting models using the Waymo Open Motion Dataset and the nuScenes Dataset, demonstrating consistent performance improvements across both benchmarks.

---

## 16. Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes

**论文链接:** [http://arxiv.org/abs/2510.16714v1](http://arxiv.org/abs/2510.16714v1)

**作者:** Xiongkun Linghu, Jiangyong Huang, Ziyu Zhu, Baoxiong Jia, Siyuan Huang

**发布时间:** 2025-10-19

**备注:** Project page: https://scenecot.github.io/

### GPT解析

### 总结

本文提出了一种新颖的SCENECOT框架，首次将思维链推理应用于3D场景理解，解决了现有3D大型语言模型在实现基于场景的问答方面的困难。

### 背景

现有关于3D大型语言模型的研究在实现基于场景的问答方面仍然存在困难，主要原因是对类人场景-对象基础推理机制的探索不足。

### 目的

通过提出一个新颖的框架来填补3D场景理解中类人推理机制的空白。

### 方法

作者在3D场景中引入了基础的思维链推理方法（SCENECOT），将复杂推理任务解耦为更简单的问题，并基于多模态专家模块构建视觉线索。同时开发了SCENECOT-185K数据集，包含185K个高质量实例。

### 主要发现

在各种复杂的3D场景推理基准上的广泛实验表明，新框架在保持高基础问答连贯性的同时实现了强大的性能。

### 结论

思维链推理首次成功应用于3D场景理解，实现了类人的逐步推理，并显示出扩展到更广泛的3D场景理解场景的潜力。

### 翻译

现有关于3D大型语言模型的研究在实现基于场景的问答方面仍然存在困难，主要原因是对类人场景-对象基础推理机制的探索不足。本文通过提出一个新颖的框架来填补这一空白。我们首先在3D场景中引入了一种基础的思维链推理方法（SCENECOT），将复杂的推理任务解耦为更简单、更易管理的问题，并基于多模态专家模块构建相应的视觉线索。为此，我们开发了SCENECOT-185K，这是第一个大规模的基础思维链推理数据集，包含185K个高质量实例。在各种复杂的3D场景推理基准上的广泛实验表明，我们的新框架在保持高基础问答连贯性的同时实现了强大的性能。据我们所知，这是思维链推理首次成功应用于3D场景理解，实现了类人的逐步推理，并显示出扩展到更广泛的3D场景理解场景的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D大语言模型难以实现接地式问答的问题，即模型生成的回答与3D场景中的实际对象和空间关系缺乏明确联系。这个问题很重要，因为3D场景理解是构建人类级别具身智能体的基础能力，而现有模型往往产生看似合理但未与场景关联的答案，导致接地-问答连贯性差，阻碍了AI系统在需要精确空间理解的应用中的表现。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过识别3D推理的挑战（导航大空间、解释复杂空间关系、处理部分可观察性），借鉴了语言领域的思维链（CoT）推理方法，将复杂问题分解为可管理的子问题。他们还参考了2D视觉-语言模型中的CoT应用，将其扩展到3D场景理解。此外，作者结合了多模态大语言模型、专门的3D-VL和2D-VL模型以及符号引擎等现有技术，构建了SCENECOT框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将复杂的3D场景推理任务分解为四个明确的阶段，每个阶段都引入明确的接地信号，确保逐步推理并提高接地-问答连贯性。整体流程包括：1）任务识别和分析，识别问题类型和初始分析；2）任务相关区域定位，缩小推理空间；3）实体和属性接地，使用多模态专家模块关联目标对象；4）接地推理，集成中间结果生成最终答案。整个过程使用特殊标记表示不同推理阶段，结合多种模型共同完成。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）SCENECOT框架，首次将思维链推理应用于3D场景理解；2）SCENECOT-185K数据集，首个大规模接地CoT推理数据集；3）四阶段推理流程，明确分解复杂任务；4）显著提高接地-问答连贯性。相比之前工作，SCENECOT采用逐步推理而非单步任务，在每个阶段都引入明确的接地信号，专门设计了针对3D场景的推理数据集，并在接地-问答连贯性方面表现更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SCENECOT通过引入首个专门为3D场景设计的接地式思维链推理框架和数据集，显著提升了AI系统在复杂3D环境中的类人推理能力和答案与场景的连贯性。'}


### 论文摘要

Existing research on 3D Large Language Models (LLMs) still struggles to achieve grounded question-answering, primarily due to the under-exploration of the mech- anism of human-like scene-object grounded reasoning. This paper bridges the gap by presenting a novel framework. We first introduce a grounded Chain-of- Thought reasoning method in 3D scenes (SCENECOT), decoupling a complex reasoning task into simpler and manageable problems, and building corresponding visual clues based on multimodal expert modules. To enable such a method, we develop SCENECOT-185K, the first large-scale grounded CoT reasoning dataset, consisting of 185K high-quality instances. Extensive experiments across various complex 3D scene reasoning benchmarks demonstrate that our new framework achieves strong performance with high grounding-QA coherence. To the best of our knowledge, this is the first successful application of CoT reasoning to 3D scene understanding, enabling step-by-step human-like reasoning and showing potential for extension to broader 3D scene understanding scenarios.

---

## 17. Structured Interfaces for Automated Reasoning with 3D Scene Graphs

**论文链接:** [http://arxiv.org/abs/2510.16643v1](http://arxiv.org/abs/2510.16643v1)

**作者:** Aaron Ray, Jacob Arkin, Harel Biggie, Chuchu Fan, Luca Carlone, Nicholas Roy

**发布时间:** 2025-10-18

**备注:** 25 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种结合大型语言模型与三维场景图的新方法，通过检索增强生成技术选择相关场景图子集，并使用图数据库和Cypher查询语言接口作为工具，提高自然语言与机器人世界表示之间的连接效率。

### 背景

为了使机器人能够理解和响应用户自然语言输入，需要将自然语言与机器人对世界的基础表示连接起来。目前，大型语言模型和三维场景图已成为热门选择，但现有方法将场景图编码为LLM上下文窗口内的序列化文本，无法有效处理大型或丰富的场景图。

### 目的

解决使用大型语言模型与三维场景图连接自然语言时的可扩展性问题，特别是处理大型或丰富场景图的挑战。

### 方法

采用检索增强生成技术选择与任务相关的三维场景图子集，将场景图编码到图数据库中，并提供Cypher查询语言接口作为大型语言模型的工具，使其能够检索与语言连接相关的数据。

### 主要发现

在指令跟随和场景问答任务上的评估表明，使用Cypher作为三维场景图的接口能更好地扩展到大型、丰富的图形，显著提高语言连接任务性能，同时大幅减少场景图内容的token数量。

### 结论

通过将三维场景图存储在图数据库并提供Cypher接口作为工具，可有效解决大型场景图的表示问题，提高自然语言与机器人世界表示之间的连接效率，减少计算资源消耗。

### 翻译

为了使机器人具备理解和响应用户自然语言输入的能力，自然语言必须与机器人对世界的基础表示相连接。最近，大型语言模型和三维场景图已成为连接自然语言和表示世界的流行选择。在这项工作中，我们解决了使用大型语言模型与三维场景图连接自然语言的挑战。现有方法将场景图编码为大型语言模型上下文窗口内的序列化文本，但这种编码无法扩展到大型或丰富的三维场景图。相反，我们提议使用一种检索增强生成形式来选择与任务相关的三维场景图子集。我们将三维场景图编码在图数据库中，并提供Cypher查询语言接口作为大型语言模型的工具，使其能够检索与语言连接相关的数据。我们在指令跟随和场景问答任务上评估了我们的方法，并与基线上下文窗口和代码生成方法进行了比较。我们的结果表明，使用Cypher作为三维场景图的接口，在本地和基于云的模型上都能显著更好地扩展到大型、丰富的图形。这大大提高了语言连接任务的性能，同时大幅减少了场景图内容的token数量。视频补充材料可在https://www.youtube.com/watch?v=zY_YI9giZSA获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何让大型语言模型（LLMs）有效地利用3D场景图（3DSGs）来理解并处理自然语言指令的问题。这个问题非常重要，因为它关系到机器人能否理解并执行人类的自然语言指令，对于人机交互至关重要。传统方法将3D场景图序列化为文本放入LLMs上下文窗口，但这种方法无法扩展到大型场景图、难以处理空间关系推理、且依赖LLMs进行定量推理（而这并非其强项）。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性，特别是序列化方法无法处理大型场景图的问题。他们考虑了两种替代方法：使用LLMs过滤节点和基于向量的RAG，但发现这些方法各有不足。作者借鉴了现有的检索增强生成（RAG）技术和图数据库技术，结合'代理AI'概念，设计出使用Cypher查询语言作为LLMs和3D场景图之间接口的新方法。他们特别受到GraphRAG技术的启发，将其专门应用于3D场景图和LLMs的结合。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用图检索增强生成（GraphRAG）方法，通过Cypher查询语言作为大型语言模型（LLMs）和3D场景图（3DSGs）之间的接口。整体流程包括：1)将3D场景图编码到图数据库中；2)向LLMs提供Cypher查询作为工具；3)当用户输入自然语言时，LLMs决定是否需要查询场景图；4)如果需要，LLMs生成一个或多个Cypher查询；5)执行查询并获取结果；6)使用查询结果作为上下文生成最终响应（对于指令跟随任务转换为PDDL目标，对于问答任务直接回答问题）。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)使用Cypher查询语言作为结构化接口；2)将RAG技术扩展到图数据结构（GraphRAG）；3)采用代理AI方法允许LLMs主动决定如何与场景图交互；4)利用图数据库的几何空间索引功能处理定量推理；5)有效处理大型、丰富的3D场景图。相比之前的工作，本文不再将整个场景图序列化为文本，而是按需查询；不依赖向量检索，而是使用结构化的图查询语言；不依赖LLMs进行定量推理，而是利用图数据库的专门功能；并采用更灵活的代理交互方式。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种使用Cypher查询语言作为大型语言模型和3D场景图之间结构化接口的新方法，通过图检索增强生成技术，使机器人系统能够有效地理解和执行自然语言指令，同时解决了传统方法在处理大型场景图和定量推理方面的局限性。'}


### 论文摘要

In order to provide a robot with the ability to understand and react to a user's natural language inputs, the natural language must be connected to the robot's underlying representations of the world. Recently, large language models (LLMs) and 3D scene graphs (3DSGs) have become a popular choice for grounding natural language and representing the world. In this work, we address the challenge of using LLMs with 3DSGs to ground natural language. Existing methods encode the scene graph as serialized text within the LLM's context window, but this encoding does not scale to large or rich 3DSGs. Instead, we propose to use a form of Retrieval Augmented Generation to select a subset of the 3DSG relevant to the task. We encode a 3DSG in a graph database and provide a query language interface (Cypher) as a tool to the LLM with which it can retrieve relevant data for language grounding. We evaluate our approach on instruction following and scene question-answering tasks and compare against baseline context window and code generation methods. Our results show that using Cypher as an interface to 3D scene graphs scales significantly better to large, rich graphs on both local and cloud-based models. This leads to large performance improvements in grounded language tasks while also substantially reducing the token count of the scene graph content. A video supplement is available at https://www.youtube.com/watch?v=zY_YI9giZSA.

---

## 18. REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2510.16410v1](http://arxiv.org/abs/2510.16410v1)

**作者:** Changyue Shi, Minghao Chen, Yiping Mao, Chuxiao Yang, Xinyuan Hu, Jiajun Ding, Zhou Yu

**发布时间:** 2025-10-18

### GPT解析

### 总结

本文提出了REALM，一个创新的MLLM-agent框架，实现了开放世界的基于推理的3D物体分割，无需大量3D特定后训练。

### 背景

在视觉和机器人领域，弥合复杂人类指令与精确3D物体定位之间的差距仍然是一个重大挑战。现有的3D分割方法难以解释模糊的、基于推理的指令，而擅长此类推理的2D视觉-语言模型则缺乏内在的3D空间理解。

### 目的

开发一种能够理解复杂人类指令并进行精确3D物体分割的方法，无需大量3D特定后训练。

### 方法

直接在3D高斯溅射表示上进行分割，利用其渲染逼真新视图的能力。提出全局到局部空间定位策略：先并行输入多个全局视图到MLLM代理进行粗略定位，聚合响应识别目标物体；然后合成物体的特写新视图进行细粒度局部分割，获得准确且一致的3D掩码。

### 主要发现

REALM在解释LERF、3D-OVS和REALM3D基准测试中的显式和隐式指令方面取得了显著性能，并能支持一系列3D交互任务。

### 结论

REALM代理框架具有实用性和多功能性，能够无缝支持物体移除、替换和风格转换等多种3D交互任务。

### 翻译

弥合复杂人类指令与精确3D物体定位之间的差距在视觉和机器人领域仍然是一个重大挑战。现有的3D分割方法往往难以解释模糊的、基于推理的指令，而擅长此类推理的2D视觉-语言模型则缺乏内在的3D空间理解。在本文中，我们介绍了REALM，一个创新的MLLM-agent框架，能够实现开放世界的基于推理的分割，而无需大量的3D特定后训练。我们直接在3D高斯溅射表示上进行分割，利用其能够渲染高度适合MLLM理解的逼真新视图的能力。由于直接将一个或多个渲染视图输入到MLLM可能导致对视角选择的高度敏感性，我们提出了一种新颖的全局到局部空间定位策略。具体来说，多个全局视图首先并行输入到MLLM代理中进行粗略定位，聚合响应以稳健地识别目标物体。然后，合成物体的几个特写新视图以执行细粒度的局部分割，从而获得准确且一致的3D掩码。大量实验表明，REALM在解释LERF、3D-OVS和我们新引入的REALM3D基准测试中的显式和隐式指令方面取得了显著性能。此外，我们的代理框架无缝支持一系列3D交互任务，包括物体移除、替换和风格转换，展示了其实用性和多功能性。项目页面：https://ChangyueShi.github.io/REALM。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何弥合复杂人类指令和精确3D物体定位之间的差距。这个问题重要是因为AI系统需要理解并能够通过自然语言与3D世界交互，这对未来机器人和人机协作至关重要。目前，3D分割方法难以解释模糊的推理指令，而2D视觉语言模型又缺乏3D空间理解能力，这限制了AI在现实场景中的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有方法各有局限：3D分割方法擅长直接查询但缺乏推理能力，而2D视觉语言模型能推理但缺乏3D空间意识。作者借鉴了3D高斯溅射(3DGS)作为3D世界的高保真表示，结合SAM进行实例分割，并利用多模态大语言模型(MLLM)进行推理。为解决视角选择敏感性问题，作者设计了全局到局部空间定位策略，通过多视角聚合提高分割准确性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用MLLM的推理能力和3DGS的高保真表示，实现开放世界的3D推理分割和编辑。整体流程包括：1)构建3D特征场为每个高斯基元分配身份特征；2)使用MLLM基础视觉分割器(LMSeg)进行图像级推理；3)通过全局到局部空间定位(GLSpaG)策略聚合多视图结果，先从全局视角粗略定位目标，再从局部视角细粒度分割；4)基于分割结果执行各种3D编辑任务如移除、替换和风格转换。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)REALM框架实现无需大量3D特定后训练的开放世界3D推理分割；2)MLLM基础实例分割器结合MLLM和SAM的能力；3)全局到局部空间定位策略提高分割准确性；4)REALM3D基准数据集促进研究。相比之前工作，REALM能处理需要推理空间关系、语义属性或常识的模糊指令，将2D推理能力提升到3D领域，支持多种3D交互任务，解决了视角选择敏感性问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'REALM通过将多模态大语言模型的推理能力与3D高斯溅射的高保真表示相结合，实现了开放世界中的3D推理分割和编辑，解决了现有方法在处理模糊、基于推理的3D指令时的局限性。'}


### 论文摘要

Bridging the gap between complex human instructions and precise 3D object grounding remains a significant challenge in vision and robotics. Existing 3D segmentation methods often struggle to interpret ambiguous, reasoning-based instructions, while 2D vision-language models that excel at such reasoning lack intrinsic 3D spatial understanding. In this paper, we introduce REALM, an innovative MLLM-agent framework that enables open-world reasoning-based segmentation without requiring extensive 3D-specific post-training. We perform segmentation directly on 3D Gaussian Splatting representations, capitalizing on their ability to render photorealistic novel views that are highly suitable for MLLM comprehension. As directly feeding one or more rendered views to the MLLM can lead to high sensitivity to viewpoint selection, we propose a novel Global-to-Local Spatial Grounding strategy. Specifically, multiple global views are first fed into the MLLM agent in parallel for coarse-level localization, aggregating responses to robustly identify the target object. Then, several close-up novel views of the object are synthesized to perform fine-grained local segmentation, yielding accurate and consistent 3D masks. Extensive experiments show that REALM achieves remarkable performance in interpreting both explicit and implicit instructions across LERF, 3D-OVS, and our newly introduced REALM3D benchmarks. Furthermore, our agent framework seamlessly supports a range of 3D interaction tasks, including object removal, replacement, and style transfer, demonstrating its practical utility and versatility. Project page: https://ChangyueShi.github.io/REALM.

---

## 19. SWIR-LightFusion: Multi-spectral Semantic Fusion of Synthetic SWIR with Thermal IR (LWIR/MWIR) and RGB

**论文链接:** [http://arxiv.org/abs/2510.13404v2](http://arxiv.org/abs/2510.13404v2)

**作者:** Muhammad Ishfaq Hussain, Ma Van Linh, Zubia Naz, Unse Fatima, Yeongmin Ko, Moongu Jeon

**发布时间:** 2025-10-15

### GPT解析

### 总结

该研究提出了一种通过合成短波红外（SWIR）图像来增强多模态融合框架的方法，以改善在能见度不良条件下的场景理解，同时保持实时性能。

### 背景

在能见度不良条件下进行场景理解是监控和自主导航系统面临的重大挑战。传统成像模式（如RGB和热红外）在融合时往往无法提供全面的场景信息，特别是在大气干扰或照明不足的条件下。SWIR成像虽然能够穿透大气干扰并提供更清晰的材料区分能力，但其广泛应用面临主要障碍是缺乏公开可用的SWIR数据集。

### 目的

解决SWIR数据集稀缺的问题，通过从现有LWIR数据合成生成类似SWIR的结构/对比度提示图像，并开发一种多模态融合框架，整合合成的SWIR、LWIR和RGB模式，以提高在不良能见度条件下的场景理解能力。

### 方法

研究提出了一种从现有LWIR数据合成生成类似SWIR的结构/对比度提示图像的方法，然后提出了一种多模态融合框架，整合合成的SWIR、LWIR和RGB模式。该框架采用优化的编码器-解码器神经网络架构，具有模态特定的编码器和softmax门控融合头。

### 主要发现

在公共RGB-LWIR基准测试集和额外的私有真实RGB-MWIR-SWIR数据集上的综合实验表明，合成SWIR增强融合框架提高了融合图像质量（对比度、边缘定义、结构保真度），同时保持实时性能。研究还添加了公平的三模态基线和级联三模态变体。

### 结论

该研究提出的合成SWIR增强融合框架在监控和自主系统等实际应用中具有巨大潜力，能够有效改善在能见度不良条件下的场景理解能力。

### 翻译

在能见度不良条件下增强场景理解仍然是监控和自主导航系统的关键挑战。传统成像模式，如RGB和热红外（MWIR/LWIR），在融合时往往无法提供全面的场景信息，特别是在大气干扰或照明不足的条件下。为了解决这些局限性，短波红外（SWIR）成像已成为一种有前景的模式，因为它能够穿透大气干扰并提供更清晰的材料区分能力。然而，基于SWIR系统的进步和广泛应用面临重大障碍，主要是由于公开可用的SWIR数据集稀缺。为应对这一挑战，我们的研究提出了一种使用先进的对比度增强技术从现有LWIR数据合成生成类似SWIR的结构/对比度提示（不声称光谱再现）图像的方法。随后，我们提出了一种多模态融合框架，整合合成的SWIR、LWIR和RGB模式，采用优化的编码器-解码器神经网络架构，具有模态特定的编码器和softmax门控融合头。在公共RGB-LWIR基准（M3FD、TNO、CAMEL、MSRS、RoadScene）和额外的私有真实RGB-MWIR-SWIR数据集上的综合实验表明，我们的合成SWIR增强融合框架提高了融合图像质量（对比度、边缘定义、结构保真度），同时保持实时性能。我们还添加了公平的三模态基线（LP、LatLRR、GFF）和U2Fusion/SwinFusion的级联三模态变体，采用统一协议。结果突显了在监控和自主系统中实际应用的巨大潜力。


### 论文摘要

Enhancing scene understanding in adverse visibility conditions remains a critical challenge for surveillance and autonomous navigation systems. Conventional imaging modalities, such as RGB and thermal infrared (MWIR / LWIR), when fused, often struggle to deliver comprehensive scene information, particularly under conditions of atmospheric interference or inadequate illumination. To address these limitations, Short-Wave Infrared (SWIR) imaging has emerged as a promising modality due to its ability to penetrate atmospheric disturbances and differentiate materials with improved clarity. However, the advancement and widespread implementation of SWIR-based systems face significant hurdles, primarily due to the scarcity of publicly accessible SWIR datasets. In response to this challenge, our research introduces an approach to synthetically generate SWIR-like structural/contrast cues (without claiming spectral reproduction) images from existing LWIR data using advanced contrast enhancement techniques. We then propose a multimodal fusion framework integrating synthetic SWIR, LWIR, and RGB modalities, employing an optimized encoder-decoder neural network architecture with modality-specific encoders and a softmax-gated fusion head. Comprehensive experiments on public RGB-LWIR benchmarks (M3FD, TNO, CAMEL, MSRS, RoadScene) and an additional private real RGB-MWIR-SWIR dataset demonstrate that our synthetic-SWIR-enhanced fusion framework improves fused-image quality (contrast, edge definition, structural fidelity) while maintaining real-time performance. We also add fair trimodal baselines (LP, LatLRR, GFF) and cascaded trimodal variants of U2Fusion/SwinFusion under a unified protocol. The outcomes highlight substantial potential for real-world applications in surveillance and autonomous systems.

---

## 20. Urban-R1: Reinforced MLLMs Mitigate Geospatial Biases for Urban General Intelligence

**论文链接:** [http://arxiv.org/abs/2510.16555v1](http://arxiv.org/abs/2510.16555v1)

**作者:** Qiongyan Wang, Xingchen Zou, Yutian Jiang, Haomin Wen, Jiaheng Wei, Qingsong Wen, Yuxuan Liang

**发布时间:** 2025-10-18

### GPT解析

### 总结

本文提出Urban-R1框架，一种基于强化学习的后训练方法，用于解决城市基础模型中的地域偏见问题，有效提升了跨区域泛化能力。

### 背景

快速城市化加剧了对城市通用智能(UGI)的需求，即能够理解和推理复杂城市环境的AI系统。现有使用监督微调(SFT)构建的城市基础模型存在持续的地域偏见，产生区域倾斜的预测和有限的泛化能力。

### 目的

提出Urban-R1框架，使多模态大型语言模型(MLLMs)与UGI目标保持一致，缓解地域偏见并提高跨区域泛化能力。

### 方法

Urban-R1采用组相对策略优化(GRPO)来优化跨地理群体的推理，并使用城市区域画像作为代理任务，从多模态城市数据提供可衡量的奖励。

### 主要发现

在不同地区和任务的广泛实验中，Urban-R1有效缓解了地域偏见，改善了跨区域泛化能力，性能优于监督微调训练和闭源模型。

### 结论

强化学习对齐是迈向公平和可信城市智能的有前景的途径。

### 翻译

快速城市化加剧了对城市通用智能(UGI)的需求，UGI指的是能够理解和推理复杂城市环境的AI系统。最近的研究使用监督微调(SFT)构建了城市基础模型，但这些模型存在持续的地域偏见，产生区域倾斜的预测和有限的泛化能力。为此，我们提出Urban-R1，一个基于强化学习的后训练框架，使MLLMs与UGI目标保持一致。Urban-R1采用组相对策略优化(GRPO)来优化跨地理群体的推理，并使用城市区域画像作为代理任务，从多模态城市数据提供可衡量的奖励。在不同地区和任务的广泛实验表明，Urban-R1有效缓解了地域偏见并改善了跨区域泛化能力，性能优于监督微调训练和闭源模型。我们的结果强调了强化学习对齐作为迈向公平和可信城市智能的有前景途径。


### 论文摘要

Rapid urbanization intensifies the demand for Urban General Intelligence (UGI), referring to AI systems that can understand and reason about complex urban environments. Recent studies have built urban foundation models using supervised fine-tuning (SFT) of LLMs and MLLMs, yet these models exhibit persistent geospatial bias, producing regionally skewed predictions and limited generalization. To this end, we propose Urban-R1, a reinforcement learning-based post-training framework that aligns MLLMs with the objectives of UGI. Urban-R1 adopts Group Relative Policy Optimization (GRPO) to optimize reasoning across geographic groups and employs urban region profiling as a proxy task to provide measurable rewards from multimodal urban data. Extensive experiments across diverse regions and tasks show that Urban-R1 effectively mitigates geo-bias and improves cross-region generalization, outperforming both SFT-trained and closed-source models. Our results highlight reinforcement learning alignment as a promising pathway toward equitable and trustworthy urban intelligence.

---

## 21. Lingua Custodi's participation at the WMT 2025 Terminology shared task

**论文链接:** [http://arxiv.org/abs/2510.17504v1](http://arxiv.org/abs/2510.17504v1)

**作者:** Jingshu Liu, Raheel Qader, Gaëtan Caillaut, Mariam Nakhlé

**发布时间:** 2025-10-20

### GPT解析

### 总结

研究BERT-based跨语言句子嵌入方法，结合单语和跨语言表征学习技术，显著减少所需训练数据量，并在多语言任务上取得优异性能。

### 背景

BERT在单语句子嵌入学习中表现有效，但BERT-based跨语言句子嵌入尚未被充分探索。

### 目的

系统研究学习多语言句子嵌入的方法，结合单语和跨语言表征的最佳方法。

### 方法

结合掩码语言建模(MLM)、翻译语言建模(TLM)、双编码器翻译排序和加性边际softmax等方法。

### 主要发现

引入预训练多语言语言模型可将实现良好性能所需的并行训练数据量减少80%；组合方法产生的模型在112种语言上达到83.7%的双语检索准确率，高于LASER的65.5%；在单语迁移学习基准测试中仍具竞争力；使用该方法挖掘的并行数据可训练出具有竞争力的NMT模型。

### 结论

公开发布了109+语言的最佳多语言句子嵌入模型。

### 翻译

虽然BERT是学习单语句子嵌入用于语义相似性和基于嵌入的迁移学习的有效方法，但基于BERT的跨语言句子嵌入尚未被探索。我们通过结合学习单语和跨语言表征的最佳方法，系统研究了学习多语言句子嵌入的方法，包括：掩码语言建模(MLM)、翻译语言建模(TLM)、双编码器翻译排序和加性边际softmax。我们证明，引入预训练多语言语言模型可将实现良好性能所需的并行训练数据量减少80%。组合这些最佳方法产生的模型在Tatoeba的112种语言上达到83.7%的双语检索准确率，远高于LASER的65.5%，同时在单语迁移学习基准测试中仍保持竞争力。使用我们的最佳模型从CommonCrawl挖掘的并行数据被证明可以训练出具有竞争力的NMT模型用于en-zh和en-de。我们在https://tfhub.dev/google/LaBSE上公开了我们的最佳多语言句子嵌入模型，适用于109+种语言。


### 论文摘要

While BERT is an effective method for learning monolingual sentence embeddings for semantic similarity and embedding based transfer learning BERT based cross-lingual sentence embeddings have yet to be explored. We systematically investigate methods for learning multilingual sentence embeddings by combining the best methods for learning monolingual and cross-lingual representations including: masked language modeling (MLM), translation language modeling (TLM), dual encoder translation ranking, and additive margin softmax. We show that introducing a pre-trained multilingual language model dramatically reduces the amount of parallel training data required to achieve good performance by 80%. Composing the best of these methods produces a model that achieves 83.7% bi-text retrieval accuracy over 112 languages on Tatoeba, well above the 65.5 achieved by LASER, while still performing competitively on monolingual transfer learning benchmarks. Parallel data mined from CommonCrawl using our best model is shown to train competitive NMT models for en-zh and en-de. We publicly release our best multilingual sentence embedding model for 109+ languages at https://tfhub.dev/google/LaBSE.

---

## 22. Integrating Trustworthy Artificial Intelligence with Energy-Efficient Robotic Arms for Waste Sorting

**论文链接:** [http://arxiv.org/abs/2510.17408v1](http://arxiv.org/abs/2510.17408v1)

**作者:** Halima I. Kure, Jishna Retnakumari, Augustine O. Nwajana, Umar M. Ismail, Bilyaminu A. Romo, Ehigiator Egho-Promise

**发布时间:** 2025-10-20

**备注:** 5 pages, 2 figures

### GPT解析

### 总结

本文提出了一种新颖的方法，将可信人工智能与节能机械臂相结合，用于智能垃圾分类和分拣。该系统通过使用MobileNetV2迁移学习增强的卷积神经网络，准确地将废物分为六类：塑料、玻璃、金属、纸张、纸板和垃圾。

### 背景

城市废物管理需要智能化的解决方案来提高分类效率和可持续性。

### 目的

开发一个结合可信人工智能和节能机械臂的系统，用于智能垃圾分类和分拣，提高废物管理的效率和可靠性。

### 方法

使用MobileNetV2迁移学习增强的卷积神经网络进行废物分类；实现机械臂模拟器进行虚拟分拣；使用欧几里得距离计算每个动作的能耗；融入可信人工智能的要素：透明度、鲁棒性、公平性和安全性。

### 主要发现

模型实现了99.8%的训练准确率和80.5%的验证准确率；系统能够准确将废物分为六类；通过能耗计算优化了机械臂的移动路径，提高了能源效率。

### 结论

该框架是一个可靠且可扩展的解决方案，适用于城市环境中的智能废物管理系统，结合了可信人工智能和节能机械臂的优势。

### 翻译

本文提出了一种新颖的方法，将可信人工智能与节能机械臂相结合，用于智能垃圾分类和分拣。通过利用通过MobileNetV2迁移学习增强的卷积神经网络，系统准确地将废物分为六类：塑料、玻璃、金属、纸张、纸板和垃圾。该模型实现了99.8%的高训练准确率和80.5%的验证准确率，展示了强大的学习和泛化能力。实现了机械臂模拟器进行虚拟分拣，使用欧几里得距离计算每个动作的能耗，确保最佳和高效的移动。该框架融入了可信人工智能的关键要素，如透明度、鲁棒性、公平性和安全性，使其成为城市环境中智能废物管理系统的可靠且可扩展的解决方案。


### 论文摘要

This paper presents a novel methodology that integrates trustworthy artificial intelligence (AI) with an energy-efficient robotic arm for intelligent waste classification and sorting. By utilizing a convolutional neural network (CNN) enhanced through transfer learning with MobileNetV2, the system accurately classifies waste into six categories: plastic, glass, metal, paper, cardboard, and trash. The model achieved a high training accuracy of 99.8% and a validation accuracy of 80.5%, demonstrating strong learning and generalization. A robotic arm simulator is implemented to perform virtual sorting, calculating the energy cost for each action using Euclidean distance to ensure optimal and efficient movement. The framework incorporates key elements of trustworthy AI, such as transparency, robustness, fairness, and safety, making it a reliable and scalable solution for smart waste management systems in urban settings.

---

## 23. Dictionary-Based Deblurring for Unpaired Data

**论文链接:** [http://arxiv.org/abs/2510.16428v1](http://arxiv.org/abs/2510.16428v1)

**作者:** Alok Panigrahi, Jayaprakash Katual, Satish Mulleti

**发布时间:** 2025-10-18

**备注:** 10 pages

### GPT解析

### 总结

本文提出了一种基于字典学习的图像去模糊方法，能够在不同数据监督条件下实现鲁棒的去模糊效果，解决了传统方法对大量配对数据的依赖问题。

### 背景

有效的图像去模糊通常依赖于大量完全配对的模糊和清晰图像数据集，但在现实世界中获取这种准确对齐的数据存在困难，限制了现有方法的有效性和泛化能力。

### 目的

解决数据稀缺依赖问题，提出一种基于字典学习的去模糊方法，用于联合估计结构化的模糊矩阵和高分辨率图像字典。

### 方法

提出了一种新颖的基于字典学习的去模糊方法，能够在不同程度的监督下实现鲁棒的图像去模糊，并在三种不同实验设置下进行了评估：完全监督、部分监督和无监督学习。

### 主要发现

在CMU-Cornell iCoseg数据集和FocusPath数据集的合成模糊子集上的实验表明，所提出的框架相比传统耦合字典学习方法具有优越的性能。

### 结论

该方法通过准确的模糊建模和自适应字典表示，能够使用更少的训练样本，为数据受限场景下的图像去模糊提供了一种高效且鲁棒的解决方案。

### 翻译

有效的图像去模糊通常依赖于大量完全配对的模糊和对应清晰图像数据集。然而，在现实世界中获取这种准确对齐的数据存在许多困难，限制了现有去模糊方法的有效性和泛化能力。为解决这种数据稀缺依赖问题，我们提出了一种新颖的基于字典学习的去模糊方法，用于联合估计结构化的模糊矩阵和高分辨率图像字典。该框架能够在不同程度的监督下实现鲁棒的图像去模糊。我们在三种不同的实验设置下对方法进行了全面评估：(i) 涉及具有明确对应关系的配对数据的完全监督；(ii) 使用具有隐含关系的未配对数据的部分监督；(iii) 使用不存在直接配对的非对应数据的无监督学习。在CMU-Cornell iCoseg数据集和真实世界FocusPath数据集的合成模糊子集上进行的大量实验验证一致表明，与传统的耦合字典学习方法相比，所提出的框架具有优越的性能。结果验证了我们的方法通过准确的模糊建模和自适应字典表示，能够使用显著更少的训练样本，为数据受限场景下的图像去模糊提供了一种高效且鲁棒的解决方案。


### 论文摘要

Effective image deblurring typically relies on large and fully paired datasets of blurred and corresponding sharp images. However, obtaining such accurately aligned data in the real world poses a number of difficulties, limiting the effectiveness and generalizability of existing deblurring methods. To address this scarcity of data dependency, we present a novel dictionary learning based deblurring approach for jointly estimating a structured blur matrix and a high resolution image dictionary. This framework enables robust image deblurring across different degrees of data supervision. Our method is thoroughly evaluated across three distinct experimental settings: (i) full supervision involving paired data with explicit correspondence, (ii) partial supervision employing unpaired data with implicit relationships, and (iii) unsupervised learning using non-correspondence data where direct pairings are absent. Extensive experimental validation, performed on synthetically blurred subsets of the CMU-Cornell iCoseg dataset and the real-world FocusPath dataset, consistently shows that the proposed framework has superior performance compared to conventional coupled dictionary learning approaches. The results validate that our approach provides an efficient and robust solution for image deblurring in data-constrained scenarios by enabling accurate blur modeling and adaptive dictionary representation with a notably smaller number of training samples.

---

## 24. A Semiparametric Gaussian Mixture Model with Spatial Dependence and Its Application to Whole-Slide Image Clustering Analysis

**论文链接:** [http://arxiv.org/abs/2510.16421v1](http://arxiv.org/abs/2510.16421v1)

**作者:** Baichen Yu, Jin Liu, Hansheng Wang

**发布时间:** 2025-10-18

### GPT解析

### 总结

该研究提出了一种半参数高斯混合模型(SGMM)，用于考虑空间信息进行无监督学习，该模型比传统GMM更灵活，能够实现同类实例的空间聚类。

### 背景

无监督学习中通常需要考虑空间信息，但传统高斯混合模型在这方面存在局限性。

### 目的

开发一种能够考虑空间信息的半参数高斯混合模型，提高无监督学习的聚类性能。

### 方法

提出半参数高斯混合模型(SGMM)，为每个实例假设随机位置，并基于此假设特征向量服从标准GMM；开发新的EM算法估计SGMM并建立渐近理论；进行数值模拟验证性能；将方法应用于CAMELYON16数据集进行乳腺癌检测。

### 主要发现

SGMM比传统GMM更灵活，能使同类实例在空间上聚集；在数值模拟和实际应用中表现出色，尤其在乳腺癌检测任务中展现了卓越的聚类性能。

### 结论

SGMM是一种有效的无监督学习方法，通过考虑空间信息提高了聚类性能，在理论和实际应用中都表现出色。

### 翻译

我们在这里开发了一种半参数高斯混合模型(SGMM)，用于考虑有价值空间信息的无监督学习。具体来说，我们为每个实例假设一个随机位置。然后，基于这个随机位置，我们为特征向量假设一个标准高斯混合模型(GMM)。所提出的SGMM允许混合概率与空间位置非参数相关。与传统GMM相比，SGMM更加灵活，并允许同一类的实例在空间上聚集。为了估计SGMM，开发了新的EM算法并建立了严格的渐近理论。进行了大量的数值模拟来证明我们的有限样本性能。对于实际应用，我们将SGMM方法应用于CAMELYON16数据集的全幻灯片图像(WSIs)进行乳腺癌检测。SGMM方法表现出卓越的聚类性能。


### 论文摘要

We develop here a semiparametric Gaussian mixture model (SGMM) for unsupervised learning with valuable spatial information taken into consideration. Specifically, we assume for each instance a random location. Then, conditional on this random location, we assume for the feature vector a standard Gaussian mixture model (GMM). The proposed SGMM allows the mixing probability to be nonparametrically related to the spatial location. Compared with a classical GMM, SGMM is considerably more flexible and allows the instances from the same class to be spatially clustered. To estimate the SGMM, novel EM algorithms are developed and rigorous asymptotic theories are established. Extensive numerical simulations are conducted to demonstrate our finite sample performance. For a real application, we apply our SGMM method to the CAMELYON16 dataset of whole-slide images (WSIs) for breast cancer detection. The SGMM method demonstrates outstanding clustering performance.

---

## 25. Adversarially Robust Quantum Transfer Learning

**论文链接:** [http://arxiv.org/abs/2510.16301v1](http://arxiv.org/abs/2510.16301v1)

**作者:** Amena Khatun, Muhammad Usman

**发布时间:** 2025-10-18

**备注:** This Book Chapter will publish in "Quantum Robustness in Artificial  Intelligence" Book by Springer and is currently in production. More  information about the Book is at:  https://link.springer.com/book/9783032111524?srsltid=AfmBOood7vZYc5xJYtLrQWND4pjedgfWAfAFFocjvnNS1lrNpVBwvJcO#accessibility-information

### GPT解析

### 总结

本文提出了一种量子迁移学习（QTL）模型，结合量子计算与迁移学习技术，用于高分辨率图像分类，在多个数据集上显示出优于传统和量子模型的性能，并通过对抗训练提高了模型鲁棒性。

### 背景

量子机器学习（QML）作为利用量子计算原理增强经典机器学习系统性能的有前景领域，受限于当前硬件约束（如量子比特数量有限和量子噪声），其实际部署仍然有限。

### 目的

开发一种混合量子-经典架构，结合量子计算优势与迁移学习技术，解决高分辨率图像分类问题，并提高模型在实际应用中的鲁棒性。

### 方法

提出量子迁移学习（QTL）模型，集成经典卷积特征提取与量子变分电路，并在Ants & Bees、CIFAR-10和道路标志检测等数据集上进行模拟实验，同时研究模型对抗攻击的脆弱性并加入对抗训练提高鲁棒性。

### 主要发现

QTL在多个数据集上实现了比传统模型和未使用迁移学习的量子模型更优的分类性能；对抗训练显著增强了QTL的鲁棒性，提高了其在安全敏感应用中部署的潜力。

### 结论

量子迁移学习模型结合了量子计算与迁移学习的优势，有效解决了量子机器学习在实际部署中的局限性，为高分辨率图像分类提供了一种有前景的解决方案。

### 翻译

量子机器学习（QML）已成为一个有前景的研究领域，通过利用量子计算原理来增强经典机器学习系统的性能。然而，由于当前硬件限制（如量子比特数量有限和量子噪声），QML的实际部署仍然有限。本章介绍了一种混合量子-经典架构，结合量子计算的优势和迁移学习技术来解决高分辨率图像分类问题。具体而言，我们提出了一种量子迁移学习（QTL）模型，集成了经典卷积特征提取和量子变分电路。通过在Ants & Bees、CIFAR-10和道路标志检测等多样化数据集上进行广泛模拟，我们证明QTL比传统模型和未使用迁移学习的量子模型实现了更好的分类性能。此外，我们还研究了模型对抗攻击的脆弱性，并证明加入对抗训练显著提高了QTL的鲁棒性，增强了其在安全敏感应用中部署的潜力。


### 论文摘要

Quantum machine learning (QML) has emerged as a promising area of research for enhancing the performance of classical machine learning systems by leveraging quantum computational principles. However, practical deployment of QML remains limited due to current hardware constraints such as limited number of qubits and quantum noise. This chapter introduces a hybrid quantum-classical architecture that combines the advantages of quantum computing with transfer learning techniques to address high-resolution image classification. Specifically, we propose a Quantum Transfer Learning (QTL) model that integrates classical convolutional feature extraction with quantum variational circuits. Through extensive simulations on diverse datasets including Ants \& Bees, CIFAR-10, and Road Sign Detection, we demonstrate that QTL achieves superior classification performance compared to both conventional and quantum models trained without transfer learning. Additionally, we also investigate the model's vulnerability to adversarial attacks and demonstrate that incorporating adversarial training significantly boosts the robustness of QTL, enhancing its potential for deployment in security sensitive applications.

---

## 26. Transfer Orthology Networks

**论文链接:** [http://arxiv.org/abs/2510.15837v1](http://arxiv.org/abs/2510.15837v1)

**作者:** Vikash Singh

**发布时间:** 2025-10-17

**备注:** 4 pages

### GPT解析

### 总结

本研究提出了TRON（Transfer Orthology Networks），一种用于跨物种迁移学习的新型神经网络架构，利用直系同源关系来指导知识迁移，实现了从源物种到目标物种的高效知识传递。

### 背景

跨物种迁移学习在生物信息学领域具有重要意义，特别是如何有效利用一个物种的知识来帮助理解另一个物种。现有的方法可能缺乏生物学解释性，且难以充分利用物种间的进化关系。

### 目的

开发一种能够利用物种间直系同源关系进行知识迁移的神经网络架构，提高跨物种预测的准确性和可解释性，并更有效地利用现有的转录组数据。

### 方法

设计了TRON架构，通过在预训练的前馈神经网络前添加一个学习的物种转换层来实现知识迁移。该转换层的权重被物种间二分图的二元邻接矩阵掩码，学习一个线性变换将源物种的基因表达映射到目标物种的基因空间。

### 主要发现

转换层的学习权重可以解释功能直系同源，提供不同物种基因如何对特定表型做出贡献的见解。这种方法为跨物种迁移学习提供了生物学基础和可解释的途径。

### 结论

TRON为跨物种迁移学习提供了一种生物学基础扎实且可解释的方法，为更有效地利用现有转录组数据铺平了道路。研究团队正在收集跨物种转录组/表型数据，以获得对TRON架构的实验验证。

### 翻译

我们提出了TRON（Transfer Orthology Networks），一种用于跨物种迁移学习的新型神经网络架构。TRON利用直系同源关系（以物种之间的二分图表示）来指导知识迁移。具体来说，我们在一个预训练的前馈神经网络前添加一个学习的物种转换层，该层的权重被这个二分图的二元邻接矩阵掩码，该神经网络用于从源物种的基因表达数据预测表型。这通过学习一个线性变换来实现知识向目标物种的高效迁移，该变换将源物种的基因表达映射到目标物种的基因空间。这个转换层的学习权重为解释功能直系同源提供了潜在途径，提供了关于不同物种基因如何对感兴趣的表型做出贡献的见解。TRON为跨物种迁移学习提供了一种生物学基础扎实且可解释的方法，为更有效地利用现有转录组数据铺平了道路。我们正在收集跨物种转录组/表型数据，以获得对TRON架构的实验验证。


### 论文摘要

We present Transfer Orthology Networks (TRON), a novel neural network architecture designed for cross-species transfer learning. TRON leverages orthologous relationships, represented as a bipartite graph between species, to guide knowledge transfer. Specifically, we prepend a learned species conversion layer, whose weights are masked by the biadjacency matrix of this bipartite graph, to a pre-trained feedforward neural network that predicts a phenotype from gene expression data in a source species. This allows for efficient transfer of knowledge to a target species by learning a linear transformation that maps gene expression from the source to the target species' gene space. The learned weights of this conversion layer offer a potential avenue for interpreting functional orthology, providing insights into how genes across species contribute to the phenotype of interest. TRON offers a biologically grounded and interpretable approach to cross-species transfer learning, paving the way for more effective utilization of available transcriptomic data. We are in the process of collecting cross-species transcriptomic/phenotypic data to gain experimental validation of the TRON architecture.

---

## 27. Towards Label-Free Brain Tumor Segmentation: Unsupervised Learning with Multimodal MRI

**论文链接:** [http://arxiv.org/abs/2510.15684v1](http://arxiv.org/abs/2510.15684v1)

**作者:** Gerard Comas-Quiles, Carles Garcia-Cabrera, Julia Dietlmeier, Noel E. O'Connor, Ferran Marques

**发布时间:** 2025-10-17

**备注:** 10 pages, 5 figures, BraTS GoAT 2025 challenge

### GPT解析

### 总结

本研究提出了一种新型的多模态视觉变换器自编码器(MViT-AE)，用于在磁共振成像中进行脑肿瘤分割的无监督异常检测，无需依赖手动标注数据。

### 背景

无监督异常检测(UAD)是脑肿瘤分割的一种替代方法，特别在标注数据有限、昂贵或不一致的情况下，可解决神经影像工作流程中的可扩展性瓶颈。

### 目的

开发一种不依赖手动标注的脑肿瘤分割方法，解决传统监督学习方法在数据获取方面的限制。

### 方法

提出MViT-AE模型，仅在健康脑部MRI图像上训练，通过基于重建的误差图检测和定位肿瘤；采用多模态早期-晚期融合策略整合多种MRI序列信息；引入后处理流程整合分割任何模型(SAM)优化肿瘤轮廓预测。

### 主要发现

在BraTS-GoAT 2025 Lighthouse数据集(包含胶质瘤、脑膜瘤和儿童脑肿瘤等)上评估，测试集上病变级别Dice相似系数：全肿瘤0.437，肿瘤核心0.316，增强肿瘤0.350；验证集上异常检测率为89.4%。

### 结论

基于变换器的无监督模型有潜力成为神经肿瘤成像中可扩展、标签高效的工具，尽管在检测小或非增强病变方面仍存在挑战。

### 翻译

无监督异常检测(UAD)为磁共振成像(MRI)中的脑肿瘤分割提供了监督学习的替代方案，特别是在标注数据有限、昂贵或不一致的情况下。本研究提出了一种新型的多模态视觉变换器自编码器(MViT-AE)，仅在健康脑部MRI上进行训练，通过基于重建的误差图检测和定位肿瘤。这种无监督范式实现了不依赖手动标签的分割，解决了神经影像工作流程中的一个关键可扩展性瓶颈。我们的方法在BraTS-GoAT 2025 Lighthouse数据集上进行了评估，该数据集包含各种类型的肿瘤，如胶质瘤、脑膜瘤和儿童脑肿瘤。为提高性能，我们引入了多模态早期-晚期融合策略，利用多种MRI序列的互补信息，以及一个整合分割任何模型(SAM)的后处理流程，以优化预测的肿瘤轮廓。尽管UAD存在已知挑战，特别是在检测小或非增强病变方面，我们的方法仍实现了具有临床意义的肿瘤定位，测试集上的病变级别Dice相似系数为0.437(全肿瘤)、0.316(肿瘤核心)和0.350(增强肿瘤)，验证集上的异常检测率为89.4%。这些发现强调了基于变换器的无监督模型作为神经肿瘤成像中可扩展、标签高效工具的潜力。


### 论文摘要

Unsupervised anomaly detection (UAD) presents a complementary alternative to supervised learning for brain tumor segmentation in magnetic resonance imaging (MRI), particularly when annotated datasets are limited, costly, or inconsistent. In this work, we propose a novel Multimodal Vision Transformer Autoencoder (MViT-AE) trained exclusively on healthy brain MRIs to detect and localize tumors via reconstruction-based error maps. This unsupervised paradigm enables segmentation without reliance on manual labels, addressing a key scalability bottleneck in neuroimaging workflows. Our method is evaluated in the BraTS-GoAT 2025 Lighthouse dataset, which includes various types of tumors such as gliomas, meningiomas, and pediatric brain tumors. To enhance performance, we introduce a multimodal early-late fusion strategy that leverages complementary information across multiple MRI sequences, and a post-processing pipeline that integrates the Segment Anything Model (SAM) to refine predicted tumor contours. Despite the known challenges of UAD, particularly in detecting small or non-enhancing lesions, our method achieves clinically meaningful tumor localization, with lesion-wise Dice Similarity Coefficient of 0.437 (Whole Tumor), 0.316 (Tumor Core), and 0.350 (Enhancing Tumor) on the test set, and an anomaly Detection Rate of 89.4% on the validation set. These findings highlight the potential of transformer-based unsupervised models to serve as scalable, label-efficient tools for neuro-oncological imaging.

---

## 28. Adaptive transfer learning for surgical tool presence detection in laparoscopic videos through gradual freezing fine-tuning

**论文链接:** [http://arxiv.org/abs/2510.15372v1](http://arxiv.org/abs/2510.15372v1)

**作者:** Ana Davila, Jacinto Colan, Yasuhisa Hasegawa

**发布时间:** 2025-10-17

**DOI:** 10.1002/ima.70218

### GPT解析

### 总结

本文提出了一种分阶段自适应微调方法，用于解决微创手术中自动化工具检测面临的标注数据有限问题。该方法通过线性探测和渐进式冻结两个阶段，有效提高了手术工具检测的性能，在胆囊切除和眼科手术数据集上均表现出色。

### 背景

微创手术可以从自动化手术工具检测中获益，但手术环境中标注数据的有限性对训练强健的深度学习模型构成了挑战。

### 目的

引入一种新的分阶段自适应微调方法，提高手术工具检测的性能和效率。

### 方法

提出两步法微调策略：1)线性探测阶段，在预训练CNN架构上添加额外分类层；2)渐进式冻结阶段，动态减少可微调层数。该方法降低了网络复杂性，提高效率，只需单次训练循环。使用在ImageNet上预训练的ResNet-50和DenseNet-121架构，在Cholec80数据集上检测胆囊切除内窥镜视频中的手术工具。

### 主要发现

该方法比现有方法和既定微调技术提高了检测性能，平均精度均值(mAP)达到96.4%。在CATARACTS数据集(眼科手术)上验证了该方法的可推广性。

### 结论

渐进式冻结微调是提高不同手术过程中工具存在检测的一种有前途的技术，可能在一般图像分类任务中有更广泛的应用。

### 翻译

微创手术可以从自动化手术工具检测中显著获益，实现高级分析和辅助。然而，手术环境中标注数据的有限性对训练强健的深度学习模型构成了挑战。本文引入了一种新颖的分阶段自适应微调方法，包含两个步骤：线性探测阶段，在预训练的基于CNN的架构上添加额外的分类层；渐进式冻结阶段，动态减少可微调层数，旨在调节对手术领域的适应。这种策略降低了网络复杂性并提高了效率，只需要单个训练循环，消除了多次迭代的必要性。我们在Cholec80数据集上验证了我们的方法，使用在ImageNet上预训练的CNN架构(ResNet-50和DenseNet-121)来检测胆囊切除内窥镜视频中的手术工具。我们的结果表明，与现有方法和既定的微调技术相比，我们的方法提高了检测性能，平均精度均值(mAP)达到96.4%。为了评估其更广泛的适用性，在CATARACTS数据集(一个不同的微创眼科手术领域)上进一步确认了微调策略的可推广性。这些发现表明，渐进式冻结微调是提高不同手术过程中工具存在检测的一种有前途的技术，可能在一般的图像分类任务中有更广泛的应用。


### 论文摘要

Minimally invasive surgery can benefit significantly from automated surgical tool detection, enabling advanced analysis and assistance. However, the limited availability of annotated data in surgical settings poses a challenge for training robust deep learning models. This paper introduces a novel staged adaptive fine-tuning approach consisting of two steps: a linear probing stage to condition additional classification layers on a pre-trained CNN-based architecture and a gradual freezing stage to dynamically reduce the fine-tunable layers, aiming to regulate adaptation to the surgical domain. This strategy reduces network complexity and improves efficiency, requiring only a single training loop and eliminating the need for multiple iterations. We validated our method on the Cholec80 dataset, employing CNN architectures (ResNet-50 and DenseNet-121) pre-trained on ImageNet for detecting surgical tools in cholecystectomy endoscopic videos. Our results demonstrate that our method improves detection performance compared to existing approaches and established fine-tuning techniques, achieving a mean average precision (mAP) of 96.4%. To assess its broader applicability, the generalizability of the fine-tuning strategy was further confirmed on the CATARACTS dataset, a distinct domain of minimally invasive ophthalmic surgery. These findings suggest that gradual freezing fine-tuning is a promising technique for improving tool presence detection in diverse surgical procedures and may have broader applications in general image classification tasks.

---

## 29. 论文ID: 2510.15337v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.15337v1.json'

---

## 30. Policy Transfer Ensures Fast Learning for Continuous-Time LQR with Entropy Regularization

**论文链接:** [http://arxiv.org/abs/2510.15165v1](http://arxiv.org/abs/2510.15165v1)

**作者:** Xin Guo, Zijiu Lyu

**发布时间:** 2025-10-16

### GPT解析

### 总结

该研究探讨了在连续时间线性二次调节器(LQRs)中使用策略迁移方法来提高强化学习效率，提供了连续时间RL策略迁移的第一个理论证明，并提出了新型策略学习算法。

### 背景

强化学习使智能体能够通过与环境的交互学习最优决策策略，但在复杂任务上从零开始训练效率低下。迁移学习在大语言模型中非常成功，为提高强化学习效率提供了有前景的方向。

### 目的

研究策略迁移方法，即在带熵正则化的连续时间线性二次调节器(LQRs)背景下，使用相关源任务的策略初始化目标RL任务的学习，并提供连续时间RL策略迁移的理论证明。

### 方法

采用策略迁移方法，使用相关源任务的策略初始化目标RL任务的学习；提出针对连续时间LQRs的新型策略学习算法；分析连续时间LQRs与基于分数的扩散模型之间的联系。

### 主要发现

证明了一个最优于一个LQR的策略可以作为紧密相关LQRs的近似最优初始化，同时保持原始算法的收敛速率；提出的策略学习算法实现了全局线性和局部超线性收敛；通过分析推导出一类基于分数的连续时间扩散模型的稳定性。

### 结论

展示了迁移学习在连续时间RL中的理论保证和算法优势，弥补了现有文献中的空白，将先前的工作从离散时间扩展到连续时间设置。

### 翻译

强化学习使智能体能够通过与环境的交互学习最优决策策略，但在复杂任务上从零开始训练可能效率极低。在大语言模型中广泛成功的迁移学习为利用预训练模型提高强化学习效率提供了有前景的方向。本文研究了策略迁移，这是一种迁移学习方法，在带熵正则化的连续时间线性二次调节器(LQRs)背景下，使用相关源任务的策略初始化目标RL任务的学习。我们首次提供了连续时间RL策略迁移的理论证明，证明了一个最优于一个LQR的策略可以作为紧密相关LQRs的近似最优初始化，同时保持原始算法的收敛速率。此外，我们提出了针对连续时间LQRs的新型策略学习算法，实现了全局线性和局部超线性收敛。我们的结果展示了连续时间RL中迁移学习的理论保证和算法优势，解决了现有文献中的空白，并将先前的工作从离散时间扩展到连续时间设置。作为我们分析的副产品，我们通过LQRs与基于分数的连续时间扩散模型之间的联系，推导出一类连续时间扩散模型的稳定性。


### 论文摘要

Reinforcement Learning (RL) enables agents to learn optimal decision-making strategies through interaction with an environment, yet training from scratch on complex tasks can be highly inefficient. Transfer learning (TL), widely successful in large language models (LLMs), offers a promising direction for enhancing RL efficiency by leveraging pre-trained models.   This paper investigates policy transfer, a TL approach that initializes learning in a target RL task using a policy from a related source task, in the context of continuous-time linear quadratic regulators (LQRs) with entropy regularization. We provide the first theoretical proof of policy transfer for continuous-time RL, proving that a policy optimal for one LQR serves as a near-optimal initialization for closely related LQRs, while preserving the original algorithm's convergence rate. Furthermore, we introduce a novel policy learning algorithm for continuous-time LQRs that achieves global linear and local super-linear convergence. Our results demonstrate both theoretical guarantees and algorithmic benefits of transfer learning in continuous-time RL, addressing a gap in existing literature and extending prior work from discrete to continuous time settings.   As a byproduct of our analysis, we derive the stability of a class of continuous-time score-based diffusion models via their connection with LQRs.

---

## 31. Transfer learning strategies for accelerating reinforcement-learning-based flow control

**论文链接:** [http://arxiv.org/abs/2510.16016v1](http://arxiv.org/abs/2510.16016v1)

**作者:** Saeed Salehi

**发布时间:** 2025-10-15

### GPT解析

### 总结

本研究探讨了迁移学习策略以加速深度强化学习在多保真度混沌流体流动控制中的应用，首次将渐进神经网络应用于基于DRL的流动控制，并评估了传统微调策略的性能。

### 背景

在深度强化学习应用于混沌流体流动控制时，如何有效利用低保真度环境训练的知识到高保真度环境是一个挑战，传统微调方法存在局限性。

### 目的

研究迁移学习策略，特别是渐进神经网络，来加速深度强化学习在多保真度混沌流体流动控制中的应用，并评估这些策略的性能、收敛行为和保留已转移知识的能力。

### 方法

1)首次将渐进神经网络应用于基于DRL的流动控制；2)对传统微调策略进行全面基准测试；3)使用Kuramoto-Sivashinsky系统作为基准，研究知识转移；4)进行逐层敏感性分析，研究PNNs如何重用中间表示。

### 主要发现

1)微调虽可加速收敛但对预训练时长敏感且易发生灾难性遗忘；2)PNNs通过保留先验知识实现稳定高效迁移；3)PNNs能动态重用源策略中间表示并逐步适应新任务；4)即使环境差异大，PNNs仍有效，而微调策略往往失败。

### 结论

新型迁移学习框架在稳健、可扩展和计算高效的流动控制方面具有潜力，可应用于更复杂的流动配置。

### 翻译

本研究探讨了迁移学习策略以加速深度强化学习在多保真度混沌流体流动控制中的应用。渐进神经网络是一种模块化架构，旨在跨任务保留和重用知识，首次被应用于基于DRL的流动控制背景下。此外，还对传统的微调策略进行了全面的基准测试，评估了它们的性能、收敛行为以及保留已转移知识的能力。Kuramoto-Sivashinsky系统被用作基准，研究如何在低保真度环境中训练的控制策略知识有效地转移到高保真度设置中。系统评估表明，虽然微调可以加速收敛，但它对预训练时长非常敏感，且容易发生灾难性遗忘。相比之下，渐进神经网络通过保留先验知识实现稳定高效的迁移，提供一致的性能提升，并且在预训练阶段对过拟合具有显著的鲁棒性。逐层敏感性分析进一步揭示了渐进神经网络如何动态重用来自源策略的中间表示，同时逐步使更深层次层适应目标任务。此外，即使在源环境和目标环境差异很大的情况下，如物理机制不匹配或控制目标不同的情况下，渐进神经网络仍然有效，而微调策略往往导致次优适应或知识转移完全失败。这些结果突显了新型迁移学习框架在稳健、可扩展和计算高效的流动控制方面的潜力，可以应用于更复杂的流动配置。


### 论文摘要

This work investigates transfer learning strategies to accelerate deep reinforcement learning (DRL) for multifidelity control of chaotic fluid flows. Progressive neural networks (PNNs), a modular architecture designed to preserve and reuse knowledge across tasks, are employed for the first time in the context of DRL-based flow control. In addition, a comprehensive benchmarking of conventional fine-tuning strategies is conducted, evaluating their performance, convergence behavior, and ability to retain transferred knowledge. The Kuramoto-Sivashinsky (KS) system is employed as a benchmark to examine how knowledge encoded in control policies, trained in low-fidelity environments, can be effectively transferred to high-fidelity settings. Systematic evaluations show that while fine-tuning can accelerate convergence, it is highly sensitive to pretraining duration and prone to catastrophic forgetting. In contrast, PNNs enable stable and efficient transfer by preserving prior knowledge and providing consistent performance gains, and are notably robust to overfitting during the pretraining phase. Layer-wise sensitivity analysis further reveals how PNNs dynamically reuse intermediate representations from the source policy while progressively adapting deeper layers to the target task. Moreover, PNNs remain effective even when the source and target environments differ substantially, such as in cases with mismatched physical regimes or control objectives, where fine-tuning strategies often result in suboptimal adaptation or complete failure of knowledge transfer. The results highlight the potential of novel transfer learning frameworks for robust, scalable, and computationally efficient flow control that can potentially be applied to more complex flow configurations.

---

## 32. Machine Learning-Based Ultrasonic Weld Characterization Using Hierarchical Wave Modeling and Diffusion-Driven Distribution Alignment

**论文链接:** [http://arxiv.org/abs/2510.13023v2](http://arxiv.org/abs/2510.13023v2)

**作者:** Joshua R. Tempelman, Adam J. Wachtor, Eric B. Flynn

**发布时间:** 2025-10-14

**备注:** 26 pages, 6 page appendix

### GPT解析

### 总结

本研究提出了一种端到端的机器学习工作流程，用于解决工业环境中超声波焊接检测面临的训练数据有限和环境波动导致的信号损坏问题。

### 背景

自动化超声波焊接检测在无损评估领域仍然是一个重大挑战，主要由于训练数据有限（实验标本整理或高保真模拟的复杂性）和工业环境的环境波动性导致的实时测量数据损坏。

### 目的

开发一种端到端的机器学习工作流程，用于真实工业环境中的声学焊接检测，解决数据整理和信号损坏的挑战。

### 方法

提出的工作流程包括降阶建模方案、基于扩散的分布对齐和基于U-Net的分割与反演；使用基于Lamb波理论的降阶Helmholtz模型生成数据集；通过迁移学习使用有限的全3D弹性动力学模拟完善模型；使用引导扩散处理分布外真实世界测量数据。

### 主要发现

低阶解决方案为反演模型提供了强大的训练数据集；引导扩散能有效处理具有不同且不可预测噪声分布的真实世界测量数据。

### 结论

这种集成框架为真实数据上的自动化焊接检测提供了端到端解决方案。

### 翻译

自动化超声波焊接检测由于训练数据有限（由于实验标本整理或高保真模拟的复杂性）和许多工业环境的环境波动性（导致实时测量数据损坏）等因素，在无损评估社区仍然是一个重大挑战。因此，用于真实（即工业）环境中声学焊接检测的端到端机器学习工作流程一直是一个难以实现的目标。这项工作通过提出包括降阶建模方案、基于扩散的分布对齐以及基于U-Net的分割和反演的工作流程，解决了数据整理和信号损坏的挑战。使用基于Lamb波理论的降阶Helmholtz模型生成涵盖不同焊接异质性和裂纹缺陷的综合数据集。相对廉价低阶解决方案为反演模型提供了强大的训练数据集，并通过使用有限的全3D弹性动力学模拟的迁移学习阶段进行完善。为了处理具有不同且不可预测噪声分布的分布外真实世界测量，即激光多普勒测振扫描，引导扩散生成OOD实验LDV扫描的分布内表示，随后由反演模型处理。这种集成框架为真实数据上的自动化焊接检测提供了端到端解决方案。


### 论文摘要

Automated ultrasonic weld inspection remains a significant challenge in the nondestructive evaluation (NDE) community to factors such as limited training data (due to the complexity of curating experimental specimens or high-fidelity simulations) and environmental volatility of many industrial settings (resulting in the corruption of on-the-fly measurements). Thus, an end-to-end machine learning (ML) workflow for acoustic weld inspection in realistic (i.e., industrial) settings has remained an elusive goal. This work addresses the challenges of data curation and signal corruption by proposing workflow consisting of a reduced-order modeling scheme, diffusion based distribution alignment, and U-Net-based segmentation and inversion. A reduced-order Helmholtz model based on Lamb wave theory is used to generate a comprehensive dataset over varying weld heterogeneity and crack defects. The relatively inexpensive low-order solutions provide a robust training dateset for inversion models which are refined through a transfer learning stage using a limited set of full 3D elastodynamic simulations. To handle out-of-distribution (OOD) real-world measurements with varying and unpredictable noise distributions, i.e., Laser Doppler Vibrometry scans, guided diffusion produces in-distribution representations of OOD experimental LDV scans which are subsequently processed by the inversion models. This integrated framework provides an end-to-end solution for automated weld inspection on real data.

---

## 33. MT-Video-Bench: A Holistic Video Understanding Benchmark for Evaluating Multimodal LLMs in Multi-Turn Dialogues

**论文链接:** [http://arxiv.org/abs/2510.17722v1](http://arxiv.org/abs/2510.17722v1)

**作者:** Yaning Pan, Zekun Wang, Qianqian Xie, Yongqian Wen, Yuanxing Zhang, Guohui Zhang, Haoxuan Hu, Zhiyu Pan, Yibing Huang, Zhidong Gan, Yonghong Lin, An Ping, Tianhao Peng, Jiaheng Liu

**发布时间:** 2025-10-20

**备注:** Project Website: https://github.com/NJU-LINK/MT-Video-Bench

### GPT解析

### 总结

本文提出了MT-Video-Bench，一个专门用于评估多模态大语言模型在多轮对话中视频理解能力的基准测试。

### 背景

多模态大语言模型(MLLMs)的近期发展显著提升了AI理解视觉模态的能力，但现有评估基准仅限于单轮问答，忽视了现实场景中多轮对话的复杂性。

### 目的

弥补现有评估基准的不足，引入一个专门用于评估MLLMs在多轮对话中表现的视频理解基准。

### 方法

MT-Video-Bench主要评估六种关注感知性和互动性的核心能力，包含来自不同领域的987个精心策划的多轮对话，这些能力与现实应用如交互式体育分析和多轮视频智能辅导紧密对齐。

### 主要发现

通过评估各种最先进的开源和闭源MLLMs，揭示了它们在处理多轮视频对话时的显著性能差异和局限性。

### 结论

该基准将公开可用，以促进未来研究。

### 翻译

多模态大语言模型(MLLMs)的近期发展显著提升了AI理解视觉模态的能力。然而，现有评估基准仍仅限于单轮问答，忽视了现实场景中多轮对话的复杂性。为弥补这一差距，我们引入了MT-Video-Bench，一个用于评估MLLMs在多轮对话中表现的整体视频理解基准。具体而言，我们的MT-Video-Bench主要评估六种关注感知性和互动性的核心能力，包含来自不同领域的987个精心策划的多轮对话。这些能力与现实应用（如交互式体育分析和多轮视频智能辅导）紧密对齐。通过MT-Video-Bench，我们广泛评估了各种最先进的开源和闭源MLLMs，揭示了它们在处理多轮视频对话时的显著性能差异和局限性。该基准将公开可用以促进未来研究。


### 论文摘要

The recent development of Multimodal Large Language Models (MLLMs) has significantly advanced AI's ability to understand visual modalities. However, existing evaluation benchmarks remain limited to single-turn question answering, overlooking the complexity of multi-turn dialogues in real-world scenarios. To bridge this gap, we introduce MT-Video-Bench, a holistic video understanding benchmark for evaluating MLLMs in multi-turn dialogues. Specifically, our MT-Video-Bench mainly assesses six core competencies that focus on perceptivity and interactivity, encompassing 987 meticulously curated multi-turn dialogues from diverse domains. These capabilities are rigorously aligned with real-world applications, such as interactive sports analysis and multi-turn video-based intelligent tutoring. With MT-Video-Bench, we extensively evaluate various state-of-the-art open-source and closed-source MLLMs, revealing their significant performance discrepancies and limitations in handling multi-turn video dialogues. The benchmark will be publicly available to foster future research.

---

## 34. A Mimamsa Inspired Framework For Instruction Sequencing In AI Agents

**论文链接:** [http://arxiv.org/abs/2510.17691v1](http://arxiv.org/abs/2510.17691v1)

**作者:** Bama Srinivasan

**发布时间:** 2025-10-20

**备注:** 16 pages

### GPT解析

### 总结

论文介绍了一种基于印度Mimamsa哲学体系的AI智能体指令排序正式框架

### 背景

灵感来源于印度哲学体系Mimamsa

### 目的

建立可靠的指令排序机制，影响AI应用如任务规划和机器人技术

### 方法

通过动作-对象对以三种方式形式化排序机制：直接断言(Srutikrama)用于时间先后顺序，目的驱动排序(Arthakrama)用于功能依赖关系，迭代过程(Pravrittikrama)用于区分重复任务中的并行和顺序执行。引入动作对象命令式逻辑的语法和语义，扩展MIRA形式化体系

### 主要发现

建立了指令排序的正确性定理，基于连续指令间对象依赖关系，并证明了可靠性和完备性

### 结论

形式化验证实现可靠的指令排序，解决时间推理和依赖建模问题

### 翻译

这篇论文提出了一个用于AI智能体指令排序的正式框架，灵感来源于印度哲学体系Mimamsa。该框架通过动作-对象对以三种不同方式形式化排序机制：直接断言(Srutikrama)用于时间先后顺序，目的驱动排序(Arthakrama)用于功能依赖关系，以及迭代过程(Pravrittikrama)用于区分重复任务中的并行和顺序执行。它引入了动作对象命令式逻辑的语法和语义，扩展了MIRA形式化体系，并添加了明确的排序演绎规则。指令排序的正确性通过一个验证定理建立，该定理基于连续指令间的对象依赖关系。这得到了可靠性和完备性的证明支持。这种形式化验证实现了可靠的指令排序，通过解决时间推理和依赖建模问题，影响了任务规划和机器人等AI应用领域。


### 论文摘要

This paper presents a formal framework for sequencing instructions in AI agents, inspired by the Indian philosophical system of Mimamsa. The framework formalizes sequencing mechanisms through action object pairs in three distinct ways: direct assertion (Srutikrama) for temporal precedence, purpose driven sequencing (Arthakrama) for functional dependencies, and iterative procedures (Pravrittikrama) for distinguishing between parallel and sequential execution in repetitive tasks. It introduces the syntax and semantics of an action object imperative logic, extending the MIRA formalism (Srinivasan and Parthasarathi, 2021) with explicit deduction rules for sequencing. The correctness of instruction sequencing is established through a validated theorem, which is based on object dependencies across successive instructions. This is further supported by proofs of soundness and completeness. This formal verification enables reliable instruction sequencing, impacting AI applications across areas like task planning and robotics by addressing temporal reasoning and dependency modeling.

---

## 35. LongInsightBench: A Comprehensive Benchmark for Evaluating Omni-Modal Models on Human-Centric Long-Video Understanding

**论文链接:** [http://arxiv.org/abs/2510.17305v1](http://arxiv.org/abs/2510.17305v1)

**作者:** ZhaoYang Han, Qihan Lin, Hao Liang, Bowen Chen, Zhou Liu, Wentao Zhang

**发布时间:** 2025-10-20

**备注:** Submitted to ARR Rolling Review

### GPT解析

### 总结

LongInsightBench是首个专门评估模型理解长视频能力的基准测试，整合视觉、音频和文本模态，包含长信息密集视频、多样化任务场景和严格质量保证流程。

### 背景

目前缺乏专门评估模型理解长视频能力的基准测试，尤其关注人类语言、视角、动作等上下文元素。

### 目的

开发一个基准测试来评估模型对长视频的理解能力，特别关注多模态整合和复杂推理任务。

### 方法

构建包含约1000个长信息密集视频的基准测试，设计六种挑战性任务场景，开发三步半自动数据质量保证流程，并进行一系列实验评估模型性能。

### 主要发现

全模态模型在需要精确时间定位和长距离因果推理的任务中面临挑战，多模态融合中存在信息损失和处理偏差。

### 结论

LongInsightBench为评估长视频理解能力提供了有效工具，揭示了当前模型在特定任务上的局限性。

### 翻译

我们引入了LongInsightBench，这是第一个专门评估模型理解长视频能力的基准测试，重点关注人类语言、视角、动作和其他上下文元素，同时整合视觉、音频和文本模态。我们的基准测试在三个关键方面表现出色：a) 长时长、信息密集的视频：我们从开源数据集FineVideo中根据时长限制和视觉及音频模态的信息密度精心选择了约1000个视频，重点关注包含丰富语言元素的内容，如讲座、访谈和vlog。b) 多样且具有挑战性的任务场景：我们设计了六种具有挑战性的任务场景，包括事件内部和事件之间的任务。c) 严格且全面的质量保证流程：我们开发了一个三步半自动数据质量保证流程，以确保合成问题和答案选项的难度和有效性。基于LongInsightBench，我们设计了一系列实验。实验结果表明，全模态模型在需要精确时间定位和长距离因果推理的任务中仍然面临挑战。扩展实验揭示了全模态模型多模态融合中的信息损失和处理偏差。我们的数据集和代码可在提供的链接获取。


### 论文摘要

We introduce \textbf{LongInsightBench}, the first benchmark designed to assess models' ability to understand long videos, with a focus on human language, viewpoints, actions, and other contextual elements, while integrating \textbf{visual, audio, and text} modalities. Our benchmark excels in three key areas: \textbf{a) Long-Duration, Information-Dense Videos:} We carefully select approximately 1,000 videos from open-source datasets FineVideo based on duration limit and the information density of both visual and audio modalities, focusing on content like lectures, interviews, and vlogs, which contain rich language elements. \textbf{b) Diverse and Challenging Task Scenarios:} We have designed six challenging task scenarios, including both Intra-Event and Inter-Event Tasks. \textbf{c) Rigorous and Comprehensive Quality Assurance Pipelines:} We have developed a three-step, semi-automated data quality assurance pipeline to ensure the difficulty and validity of the synthesized questions and answer options. Based on LongInsightBench, we designed a series of experiments. Experimental results shows that Omni-modal models(OLMs) still face challenge in tasks requiring precise temporal localization (T-Loc) and long-range causal inference (CE-Caus). Extended experiments reveal the information loss and processing bias in multi-modal fusion of OLMs. Our dataset and code is available at https://anonymous.4open.science/r/LongInsightBench-910F/.

---

## 36. Fair and Interpretable Deepfake Detection in Videos

**论文链接:** [http://arxiv.org/abs/2510.17264v1](http://arxiv.org/abs/2510.17264v1)

**作者:** Akihito Yoshii, Ryosuke Sonoda, Ramya Srinivasan

**发布时间:** 2025-10-20

**备注:** 10 pages (including References)

### GPT解析

### 总结

提出了一种公平感知的深度伪造检测框架，整合时间特征学习和人口感知数据增强，提高检测的公平性和可解释性。

### 背景

现有深度伪造检测方法存在偏见、缺乏透明度，无法捕捉时间信息，导致对不同人口统计群体做出有偏见的决策和不可靠结果。

### 目的

开发一个公平感知的深度伪造检测框架，整合时间特征学习和人口感知数据增强，以提高公平性和可解释性。

### 方法

使用基于序列的聚类进行深度伪造视频的时间建模，利用概念提取提高检测可靠性并为非专业用户提供可解释决策；引入人口感知的数据增强方法，平衡代表性不足的群体，应用频域变换保留深度伪造伪影，减轻偏见并提高泛化能力。

### 主要发现

在FaceForensics++、DFD、Celeb-DF和DFDC数据集上的实验表明，所提出的方法在公平性和准确性之间取得了最佳平衡，优于现有最先进的方法。

### 结论

所提出的公平感知深度伪造检测框架通过整合时间特征学习和人口感知数据增强，有效提高了检测的公平性和可解释性，同时保持了高准确性。

### 翻译

现有的深度伪造检测方法往往存在偏见、缺乏透明度，并且无法捕捉时间信息，导致对不同人口群体做出有偏见的决策和不可靠的结果。在本文中，我们提出了一个公平感知的深度伪造检测框架，整合时间特征学习和人口感知的数据增强，以提高公平性和可解释性。我们的方法利用基于序列的聚类对深度伪造视频进行时间建模，并通过概念提取提高检测可靠性，同时也为非专业用户提供可解释的决策。此外，我们引入了一种人口感知的数据增强方法，平衡代表性不足的群体，并应用频域变换来保留深度伪造伪影，从而减轻偏见并提高泛化能力。在FaceForensics++、DFD、Celeb-DF和DFDC数据集上使用最先进的架构（Xception、ResNet）进行的广泛实验证明了所提出方法在获得公平性和准确性之间最佳平衡方面的有效性，优于现有最先进的方法。


### 论文摘要

Existing deepfake detection methods often exhibit bias, lack transparency, and fail to capture temporal information, leading to biased decisions and unreliable results across different demographic groups. In this paper, we propose a fairness-aware deepfake detection framework that integrates temporal feature learning and demographic-aware data augmentation to enhance fairness and interpretability. Our method leverages sequence-based clustering for temporal modeling of deepfake videos and concept extraction to improve detection reliability while also facilitating interpretable decisions for non-expert users. Additionally, we introduce a demography-aware data augmentation method that balances underrepresented groups and applies frequency-domain transformations to preserve deepfake artifacts, thereby mitigating bias and improving generalization. Extensive experiments on FaceForensics++, DFD, Celeb-DF, and DFDC datasets using state-of-the-art (SoTA) architectures (Xception, ResNet) demonstrate the efficacy of the proposed method in obtaining the best tradeoff between fairness and accuracy when compared to SoTA.

---

## 37. An empirical study of the effect of video encoders on Temporal Video Grounding

**论文链接:** [http://arxiv.org/abs/2510.17007v1](http://arxiv.org/abs/2510.17007v1)

**作者:** Ignacio M. De la Jara, Cristian Rodriguez-Opazo, Edison Marrese-Taylor, Felipe Bravo-Marquez

**发布时间:** 2025-10-19

### GPT解析

### 总结

该研究调查了不同视频特征对时序视频定位任务中经典架构性能的影响，发现仅通过更改视频编码器就能显著改变模型性能，并揭示了特征间可能存在的互补性。

### 背景

时序视频定位是计算机视觉的基础任务，旨在长视频中定位自然语言查询。由于每天产生大量视频，该任务在科学界具有重要地位。

### 目的

解决当前研究仅集中在少数视频表示上可能导致架构过拟合的问题，通过实证研究调查不同视频特征对经典架构的影响。

### 方法

在三个基准测试（Charades-STA、ActivityNet-Captions和YouCookII）上提取特征，使用基于CNN、时序推理和transformers的视频编码器进行实验。

### 主要发现

仅通过更改视频编码器，模型的性能就显示出显著差异，同时揭示了使用某些特征产生的明显模式和错误。

### 结论

不同视频特征对模型性能有显著影响，特征之间可能存在互补性，这为未来研究提供了新的方向。

### 翻译

时序视频定位是计算机视觉中的一个基础任务，旨在定位长视频中未经修剪的自然语言查询。由于每天产生大量视频，它在科学界起着关键作用。尽管我们发现该领域有大量工作，但我们注意到研究仍然集中在少数几种视频表示上，长期来看可能导致架构过拟合。为了解决这个问题，我们提出了一项实证研究来调查不同视频特征对经典架构的影响。我们使用基于CNN、时序推理和transformers的视频编码器，为三个知名基准测试Charades-STA、ActivityNet-Captions和YouCookII提取特征。我们的结果表明，仅通过更改视频编码器，我们模型的性能就显示出显著差异，同时揭示了使用某些特征产生的明显模式和错误，最终表明特征间可能存在互补性。


### 论文摘要

Temporal video grounding is a fundamental task in computer vision, aiming to localize a natural language query in a long, untrimmed video. It has a key role in the scientific community, in part due to the large amount of video generated every day. Although we find extensive work in this task, we note that research remains focused on a small selection of video representations, which may lead to architectural overfitting in the long run. To address this issue, we propose an empirical study to investigate the impact of different video features on a classical architecture. We extract features for three well-known benchmarks, Charades-STA, ActivityNet-Captions and YouCookII, using video encoders based on CNNs, temporal reasoning and transformers. Our results show significant differences in the performance of our model by simply changing the video encoder, while also revealing clear patterns and errors derived from the use of certain features, ultimately indicating potential feature complementarity.

---

## 38. From Mannequin to Human: A Pose-Aware and Identity-Preserving Video Generation Framework for Lifelike Clothing Display

**论文链接:** [http://arxiv.org/abs/2510.16833v1](http://arxiv.org/abs/2510.16833v1)

**作者:** Xiangyu Mu, Dongliang Zhou, Jie Hou, Haijun Zhang, Weili Guan

**发布时间:** 2025-10-19

### GPT解析

### 总结

本文提出了一种名为M2HVideo的视频生成框架，能够从人体模特影像生成身份可控、照片级真实的人体视频，解决了头部与身体运动不匹配以及时间建模导致的身份漂移问题。

### 背景

人体模特展示是线上服装展示的经济有效替代方案，但缺乏真实感和表现细节，限制了其在时尚展示中的应用效果。

### 目的

引入人体模特到人体(M2H)视频生成任务，旨在从人体模特影像中合成身份可控、照片级真实的人体视频，提高线上服装展示的真实感和表现力。

### 方法

提出M2HVideo框架，包含动态姿态感知头部编码器融合面部语义和身体姿态，通过基于DDIM的一步去噪在像素空间应用镜像损失解决面部细节丢失问题，以及设计分布感知适配器对齐身份和服装特征统计分布增强时间一致性。

### 主要发现

在UBC时尚数据集、自建的ASOS数据集和新收集的MannequinVideos数据集上的实验表明，M2HVideo在服装一致性、身份保持和视频保真度方面优于现有最先进方法。

### 结论

M2HVideo框架有效解决了人体模特到人体视频生成的关键挑战，能够生成高质量、高保真度的服装展示视频，为线上时尚展示提供了新的解决方案。

### 翻译

人体模特展示为线上服装展示提供了比真人展示更具成本效益的替代方案，但缺乏真实感和表现细节。为克服这一局限，我们引入了一种称为人体模特到人体(M2H)视频生成的新任务，旨在从人体模特影像中合成身份可控、照片级真实的人体视频。我们提出M2HVideo，一个姿态感知和身份保持的视频生成框架，解决了两个关键挑战：头部和身体运动不匹配，以及时间建模导致的身份漂移。特别是，M2HVideo集成了动态姿态感知头部编码器，融合面部语义和身体姿态，产生跨帧一致的身份嵌入。为解决潜在空间压缩导致的面部细节丢失问题，我们引入了通过基于DDIM的一步去噪在像素空间应用的镜像损失。此外，我们设计了一个分布感知适配器，对齐身份和服装特征的统计分布，以增强时间一致性。在UBC时尚数据集、我们自建的ASOS数据集以及在现场收集的新MannequinVideos数据集上的大量实验表明，M2HVideo在服装一致性、身份保持和视频保真度方面优于最先进方法。


### 论文摘要

Mannequin-based clothing displays offer a cost-effective alternative to real-model showcases for online fashion presentation, but lack realism and expressive detail. To overcome this limitation, we introduce a new task called mannequin-to-human (M2H) video generation, which aims to synthesize identity-controllable, photorealistic human videos from footage of mannequins. We propose M2HVideo, a pose-aware and identity-preserving video generation framework that addresses two key challenges: the misalignment between head and body motion, and identity drift caused by temporal modeling. In particular, M2HVideo incorporates a dynamic pose-aware head encoder that fuses facial semantics with body pose to produce consistent identity embeddings across frames. To address the loss of fine facial details due to latent space compression, we introduce a mirror loss applied in pixel space through a denoising diffusion implicit model (DDIM)-based one-step denoising. Additionally, we design a distribution-aware adapter that aligns statistical distributions of identity and clothing features to enhance temporal coherence. Extensive experiments on the UBC fashion dataset, our self-constructed ASOS dataset, and the newly collected MannequinVideos dataset captured on-site demonstrate that M2HVideo achieves superior performance in terms of clothing consistency, identity preservation, and video fidelity in comparison to state-of-the-art methods.

---

## 39. Xiaoice: Training-Free Video Understanding via Self-Supervised Spatio-Temporal Clustering of Semantic Features

**论文链接:** [http://arxiv.org/abs/2510.16781v1](http://arxiv.org/abs/2510.16781v1)

**作者:** Shihao Ji, Zihui Song

**发布时间:** 2025-10-19

### GPT解析

### 总结

本文提出了一种无需训练的视频理解框架，通过结合预训练视觉语言模型(VLMs)和经典机器学习算法，实现了视频内容的零样本、自动化结构分析。

### 背景

大规模视觉语言模型在静态图像上表现出显著的零样本推理能力，但这种能力尚未完全转移到视频领域。传统视频理解模型依赖于大量标注数据集的特定任务训练，过程昂贵且难以扩展。

### 目的

开发一种新颖的、无需训练的视频理解框架，避免端到端训练，协同结合预训练VLM的语义先验与经典机器学习算法的模式发现能力。

### 方法

将视频理解重构为高维语义特征空间中的自监督时空聚类问题：使用预训练VLM的冻结视觉编码器将视频转换为语义特征轨迹；应用核时间分割(KTS)技术将特征流分割为语义连贯事件段；通过无监督密度聚类识别重复出现的宏观场景和主题；从每个聚类中选择代表性关键帧并利用VLM生成文本描述，最终形成结构化多模态摘要。

### 主要发现

该方法为视频内容的零样本、自动化结构分析提供了有效、可解释且与模型无关的途径。

### 结论

该框架无需训练即可实现视频理解，结合了预训练VLM和经典机器学习算法的优势，能够生成结构化的视频摘要。

### 翻译

大规模视觉语言模型(VLMs)在静态图像上显著的零样本推理能力尚未完全转移到视频领域。传统视频理解模型通常依赖于在标注数据集上进行大量特定任务的训练，这一过程既昂贵又难以扩展。本文提出了一种新颖的、无需训练的视频理解框架，通过协同结合预训练VLM的丰富语义先验与经典机器学习算法的模式发现能力，避免了端到端训练。我们的核心思想是将视频理解重新构建为高维语义特征空间中的自监督时空聚类问题。所提出的管道首先使用预训练VLM的冻结视觉编码器将视频流转换为语义特征轨迹。随后，我们采用核时间分割(KTS)这一稳健的机器学习技术，将连续特征流分割为离散的语义连贯事件段。这些段随后接受无监督密度聚类，以识别视频中重复出现的宏观场景和主题。通过从每个发现的聚类中选择代表性关键帧，并利用VLM的生成能力进行文本描述，我们的框架自动生成视频内容的结构化、多模态摘要。这种方法为视频内容的零样本、自动化结构分析提供了有效、可解释且与模型无关的途径。


### 论文摘要

The remarkable zero-shot reasoning capabilities of large-scale Visual Language Models (VLMs) on static images have yet to be fully translated to the video domain. Conventional video understanding models often rely on extensive, task-specific training on annotated datasets, a process that is both costly and limited in scalability. This paper introduces a novel, training-free framework for video understanding that circumvents end-to-end training by synergistically combining the rich semantic priors of pre-trained VLMs with classic machine learning algorithms for pattern discovery. Our core idea is to reframe video understanding as a self-supervised spatio-temporal clustering problem within a high-dimensional semantic feature space. The proposed pipeline first transforms a video stream into a semantic feature trajectory using the frozen visual encoder of a pre-trained VLM. Subsequently, we employ Kernel Temporal Segmentation (KTS), a robust machine learning technique, to partition the continuous feature stream into discrete, semantically coherent event segments. These segments are then subjected to unsupervised density-based clustering to identify recurring macroscopic scenes and themes throughout the video. By selecting representative keyframes from each discovered cluster and leveraging the VLM's generative capabilities for textual description, our framework automatically produces a structured, multi-modal summary of the video content. This approach provides an effective, interpretable, and model-agnostic pathway for zero-shot, automated structural analysis of video content.

---

## 40. A Comprehensive Survey on World Models for Embodied AI

**论文链接:** [http://arxiv.org/abs/2510.16732v1](http://arxiv.org/abs/2510.16732v1)

**作者:** Xinqing Li, Xin He, Le Zhang, Yun Liu

**发布时间:** 2025-10-19

**备注:** https://github.com/Li-Zn-H/AwesomeWorldModels

### GPT解析

### 总结

该论文综述了具身AI中的世界模型，提出了一个统一的框架，包括三轴分类法、系统化的数据资源和评估指标，并对最先进模型进行了定量比较。

### 背景

具身AI需要能够感知、行动并预测行动如何重塑未来世界状态的智能体。世界模型作为内部模拟器捕捉环境动态，支持感知、预测和决策。

### 目的

提出一个统一的具身AI中世界模型的框架，正式化问题设定和学习目标，并系统化相关资源和评估方法。

### 方法

提出一个三轴分类法：(1)功能性：决策耦合型vs通用型；(2)时间建模：顺序模拟与推理vs全局差异预测；(3)空间表示：全局潜在向量、标记特征序列、空间潜在网格和分解渲染表示。系统化机器人、自动驾驶和一般视频环境的数据资源和评估指标，对最先进模型进行定量比较，并总结关键挑战。

### 主要发现

世界模型在具身AI中具有统一框架，可通过三轴分类法进行系统化分类；当前研究面临统一数据集稀缺、评估指标需要更多关注物理一致性而非像素保真度、模型性能与计算效率之间的权衡，以及实现长时间一致性同时减轻错误累积等挑战。

### 结论

世界模型研究仍面临多个开放挑战，包括需要统一的评估指标、平衡性能与计算效率、实现长时间一致性等。论文提供了一个精选的参考文献库供进一步研究。

### 翻译

具身AI需要能够感知、行动并预测行动如何重塑未来世界状态的智能体。世界模型作为内部模拟器捕捉环境动态，支持前向和反事实展开以支持感知、预测和决策。本综述提出了具身AI中世界模型的统一框架。具体而言，我们正式化了问题设定和学习目标，并提出了一个三轴分类法，包括：(1)功能性：决策耦合型vs通用型；(2)时间建模：顺序模拟与推理vs全局差异预测；(3)空间表示：全局潜在向量、标记特征序列、空间潜在网格和分解渲染表示。我们系统化了机器人、自动驾驶和一般视频环境的数据资源和评估指标，涵盖了像素预测质量、状态级理解和任务性能。此外，我们对最先进模型进行了定量比较，并总结了关键开放挑战，包括统一数据集的稀缺性、需要评估物理一致性而非像素保真度的评估指标、模型性能与实时控制所需计算效率之间的权衡，以及实现长时间一致性同时减轻错误累积的核心建模难度。最后，我们在https://github.com/Li-Zn-H/AwesomeWorldModels上维护了一个精选的参考文献库。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决具身AI领域中世界模型缺乏统一分类框架的问题。这个问题很重要，因为具身AI需要智能体能够感知环境、行动并预测行动如何改变未来状态，而世界模型作为内部模拟器是支持这些能力的关键组件。缺乏统一分类导致研究分散、术语不一致，难以进行有效比较和知识整合，阻碍了领域的发展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到具身AI中世界模型的重要性及当前研究的分类混乱问题。他们从认知科学中人类构建内部世界模型的方式获得启发，分析了世界模型的核心概念（模拟与规划、时间演化、空间表示）。作者借鉴了早期基于模型强化学习的研究、Ha和Schmidhuber的开创性工作、Dreamer系列模型以及Sora和V-JEPA 2等大型生成模型的研究成果，但发现现有综述采用功能导向或应用驱动的方法缺乏统一框架，因此提出了自己的三轴分类系统。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提出一个统一的三轴分类框架来系统化组织具身AI中的世界模型研究。这三个轴分别是：功能（决策耦合vs通用目的）、时间建模（顺序模拟与推理vs全局差异预测）和空间表示（全局潜在向量、标记特征序列、空间潜在网格和分解渲染表示）。整体流程包括：介绍核心概念和理论基础、提出分类框架并映射方法、调查数据资源和评估指标、提供模型定量比较、讨论挑战和趋势、总结综述。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出统一的三轴分类框架；区分决策耦合和通用目的两种功能类型；区分顺序模拟与推理和全局差异预测两种时间建模策略；涵盖多种空间表示方法；系统整理跨领域数据资源和评估指标；提供最先进模型的定量比较；确定关键开放挑战。相比之前工作，本文提供了一个更全面、系统的分类框架，超越了之前功能导向或应用驱动的局限，覆盖更广泛的应用场景，并提供更全面的数据资源、评估指标和未来研究方向。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过提出一个统一的三轴分类框架，系统化地组织和分析了具身AI中的世界模型研究，为该领域提供了全面的知识图谱和未来研究方向。'}


### 论文摘要

Embodied AI requires agents that perceive, act, and anticipate how actions reshape future world states. World models serve as internal simulators that capture environment dynamics, enabling forward and counterfactual rollouts to support perception, prediction, and decision making. This survey presents a unified framework for world models in embodied AI. Specifically, we formalize the problem setting and learning objectives, and propose a three-axis taxonomy encompassing: (1) Functionality, Decision-Coupled vs. General-Purpose; (2) Temporal Modeling, Sequential Simulation and Inference vs. Global Difference Prediction; (3) Spatial Representation, Global Latent Vector, Token Feature Sequence, Spatial Latent Grid, and Decomposed Rendering Representation. We systematize data resources and metrics across robotics, autonomous driving, and general video settings, covering pixel prediction quality, state-level understanding, and task performance. Furthermore, we offer a quantitative comparison of state-of-the-art models and distill key open challenges, including the scarcity of unified datasets and the need for evaluation metrics that assess physical consistency over pixel fidelity, the trade-off between model performance and the computational efficiency required for real-time control, and the core modeling difficulty of achieving long-horizon temporal consistency while mitigating error accumulation. Finally, we maintain a curated bibliography at https://github.com/Li-Zn-H/AwesomeWorldModels.

---

## 41. Temporal Understanding under Deictic Frame of Reference

**论文链接:** [http://arxiv.org/abs/2510.16685v1](http://arxiv.org/abs/2510.16685v1)

**作者:** Damin Zhang, Julia Rayz

**发布时间:** 2025-10-19

**备注:** Under review

### GPT解析

### 总结

本研究引入了TUuD框架，评估大型语言模型(LLMs)在时间参考框架(t-FoR)下如何解释时间关系，发现LLMs表现出部分类人时间认知，但推理能力仍受参考框架转换和时间距离影响。

### 背景

理解时间是人类认知的基础，时间经验常通过空间隐喻概念化。人类依赖参考框架(FoR)解释意义，而时间参考框架(t-FoR)定义了时间关系如何相对于'现在'被感知。尽管LLMs在自然语言理解上进展显著，但其时间解释和推理能力有限。

### 目的

引入TUuD(Deictic t-FoR下的时间理解)框架，评估当'现在'的参考点沿时间线动态移动时，LLMs如何解释时间-事件和事件-事件关系。

### 方法

提示LLMs对当前时刻和目标事件之间的相似性进行评分(0.00-1.00)，其中相似性量化了两个点之间的感知时间对齐。

### 主要发现

四个评估的LLMs表现出对指示性t-FoR的可测量适应，相似性评分在当前时刻达到峰值并向过去和未来事件递减。然而，这种适应在近期语境之外会减弱。

### 结论

虽然LLMs显示出部分类人时间认知，但它们的时间推理仍然对参考框架的转换和时间距离敏感。

### 翻译

理解时间是人类认知的基础，其中时间经验通常通过基于感官-运动经验的空间隐喻来概念化。例如，'夏天即将来临'与'我们正在接近夏天'是平行的表达。在这些表达中，人类依赖参考框架(FoR)来解释相对于特定视点的意义。将这一概念扩展到时间，时间参考框架(t-FoR)定义了时间关系如何相对于体验者的'现在'时刻被感知。虽然大型语言模型(LLMs)在自然语言理解方面显示出显著进展，但它们解释和推理时间的能力仍然有限。在这项工作中，我们引入了TUuD(Deictic t-FoR下的时间理解)框架，评估当'现在'的参考点沿时间线动态移动时，LLMs如何解释时间-事件和事件-事件关系。遵循最近关于时间认知的工作，提示LLMs对当前时刻和目标事件之间的相似性进行评分，从0.00(完全不同)到1.00(高度相似)，其中相似性量化了两个点之间的感知时间对齐。我们的结果表明，四个评估的LLMs表现出对指示性t-FoR的可测量适应，相似性评分在当前时刻达到峰值并向过去和未来事件递减。然而，这种适应在近期语境之外会减弱，这表明虽然LLMs显示出部分类人时间认知，但它们的时间推理仍然对参考框架的转换和时间距离敏感。


### 论文摘要

Understanding time is fundamental to human cognition, where temporal experience is often conceptualized through spatial metaphors grounded in sensory-motor experience. For example, "summer is approaching" parallels "We are approaching the summer". In such expressions, humans rely on a frame of reference (FoR) to interpret meaning relative to a particular viewpoint. Extending this concept to time, a temporal frame of reference (t-FoR) defines how temporal relations are perceived relative to an experiencer's moment of "now". While Large Language Models (LLMs) have shown remarkable advances in natural language understanding, their ability to interpret and reason about time remains limited. In this work, we introduce TUuD (Temporal Understanding under Deictic t-FoR), a framework that evaluates how LLMs interpret time-event and event-event relations when the reference point of "now" dynamically shifts along a timeline. Following recent work on temporal cognition \cite{li2025other}, LLMs are prompted to rate the similarity between the current moment and a target event from 0.00 (completely dissimilar) to 1.00 (highly similar), where similarity quantifies perceived temporal alignment between the two points. Our results show that four evaluated LLMs exhibit measurable adaptation to a deictic t-FoR, with similarity ratings peaking around the present and decreasing toward past and future events. The adaptation, however, weakens beyond near-term contexts, suggesting that while LLMs display partial human-like temporal cognition, their temporal reasoning remains sensitive to reference-frame shifts and temporal distance.

---

## 42. Watch Where You Move: Region-aware Dynamic Aggregation and Excitation for Gait Recognition

**论文链接:** [http://arxiv.org/abs/2510.16541v1](http://arxiv.org/abs/2510.16541v1)

**作者:** Binyuan Huang, Yongdong Luo, Xianda Guo, Xiawu Zheng, Zheng Zhu, Jiahui Pan, Chengju Zhou

**发布时间:** 2025-10-18

**DOI:** 10.1109/TMM.2025.3613158

### GPT解析

### 总结

本文提出了一种名为GaitRDAE的区域感知动态聚合和激励框架，用于解决步态识别中动态建模运动区域的问题

### 背景

深度学习步态识别在各种应用中取得了巨大成功，准确步态识别的关键在于考虑不同运动区域的独特和多样化的行为模式，特别是当协变量影响视觉外观时

### 目的

解决现有方法使用预定义区域进行时间建模，为不同类型区域分配固定或等效时间尺度，难以处理随时间动态变化的运动区域的问题

### 方法

提出GaitRDAE框架，包括两个核心模块：区域感知动态聚合（RDA）模块，为每个区域动态搜索最佳时间感受野；区域感知动态激励（RDE）模块，强调学习稳定行为模式的运动区域，抑制对易受协变量影响的静态区域的注意力

### 主要发现

实验结果表明，GaitRDAE在几个基准数据集上取得了最先进的性能

### 结论

GaitRDAE框架能够有效处理步态识别中动态变化的运动区域，提高了识别准确率

### 翻译

基于深度学习的步态识别在各种应用中取得了巨大成功。准确步态识别的关键在于考虑不同运动区域中独特且多样化的行为模式，特别是当协变量影响视觉外观时。然而，现有方法通常使用预定义区域进行时间建模，为不同类型的区域分配固定或等效的时间尺度，这使得难以建模随时间动态变化的运动区域并适应其特定模式。为解决这个问题，我们引入了一个区域感知动态聚合和激励框架（GaitRDAE），该框架自动搜索运动区域，分配自适应时间尺度并应用相应的注意力机制。具体而言，该框架包括两个核心模块：区域感知动态聚合（RDA）模块，为每个区域动态搜索最佳时间感受野；区域感知动态激励（RDE）模块，强调学习包含更稳定行为模式的运动区域，同时抑制对更易受协变量影响的静态区域的注意力。实验结果表明，GaitRDAE在几个基准数据集上取得了最先进的性能。


### 论文摘要

Deep learning-based gait recognition has achieved great success in various applications. The key to accurate gait recognition lies in considering the unique and diverse behavior patterns in different motion regions, especially when covariates affect visual appearance. However, existing methods typically use predefined regions for temporal modeling, with fixed or equivalent temporal scales assigned to different types of regions, which makes it difficult to model motion regions that change dynamically over time and adapt to their specific patterns. To tackle this problem, we introduce a Region-aware Dynamic Aggregation and Excitation framework (GaitRDAE) that automatically searches for motion regions, assigns adaptive temporal scales and applies corresponding attention. Specifically, the framework includes two core modules: the Region-aware Dynamic Aggregation (RDA) module, which dynamically searches the optimal temporal receptive field for each region, and the Region-aware Dynamic Excitation (RDE) module, which emphasizes the learning of motion regions containing more stable behavior patterns while suppressing attention to static regions that are more susceptible to covariates. Experimental results show that GaitRDAE achieves state-of-the-art performance on several benchmark datasets.

---

## 43. StretchySnake: Flexible SSM Training Unlocks Action Recognition Across Spatio-Temporal Scales

**论文链接:** [http://arxiv.org/abs/2510.16209v1](http://arxiv.org/abs/2510.16209v1)

**作者:** Nyle Siddiqui, Rohit Gupta, Sirnam Swetha, Mubarak Shah

**发布时间:** 2025-10-17

### GPT解析

### 总结

该研究提出了一种名为StretchySnake的灵活状态空间模型训练方法，解决了视频理解中时空不灵活性问题，使模型能够更好地处理不同分辨率和长度的视频，在多种动作识别基准测试中表现优异。

### 背景

状态空间模型(SSMs)已成为transformers的有力替代方案，具有线性计算复杂度和隐藏状态递归特性，特别适合建模长序列。然而，当前视频理解训练方法针对transformers设计，未能充分利用SSMs的独特属性，导致模型在训练中未见过的空间和时间分辨率视频上性能下降。

### 目的

提出一种灵活的训练方法，利用并改进SSMs的固有适应性，使模型能够无缝处理从短时精细片段到长时间复杂活动的各种视频，解决时空不灵活性限制模型在短视频和长视频间保持性能的问题。

### 方法

在训练过程中对视频进行不同时间和空间分辨率的采样，并动态插值模型权重以适应任何时空尺度。研究提出了五种不同的灵活训练变体，并确定了视频SSMs最有效的策略，创建了名为StretchySnake的模型。

### 主要发现

StretchySnake在短动作(UCF-101, HMDB-51)和长动作(COIN, Breakfast)基准测试中，超越了transformer和SSM基线，性能提升高达28%，同时展现出对精细动作(SSV2, Diving-48)的强大适应能力。

### 结论

该方法提供了一种简单的即插即用训练方案，使视频SSMs在各种动作识别场景中更加健壮、分辨率无关且高效，解决了视频模型在时空变化下的性能退化问题。

### 翻译

状态空间模型(SSMs)已成为各种任务中transformers的有竞争力的替代方案。它们的线性复杂度和隐藏状态递归特性使它们特别适合建模长序列，而注意力机制则变得二次方昂贵。然而，当前视频理解的训练方法是为transformers量身定制的，未能充分利用SSMs的独特属性。例如，视频模型通常在固定分辨率和视频长度下训练，以平衡注意力成本的二次方扩展与性能。因此，当在训练中未见过的空间和时间分辨率的视频上评估时，这些模型性能会下降；我们称这种特性为时空不灵活性。在动作识别的背景下，这严重限制了模型在短视频和长视频之间保持性能的能力。因此，我们提出了一种灵活的训练方法，利用并改进SSMs的固有适应性。我们的方法在训练过程中对视频进行不同时间和空间分辨率的采样，并动态插值模型权重以适应任何时空尺度。这使我们的SSM（我们称之为StretchySnake）具有时空灵活性，能够无缝处理从短时精细片段到长时间复杂活动的各种视频。我们介绍并比较了五种不同的灵活训练变体，确定了视频SSMs最有效的策略。在短动作(UCF-101, HMDB-51)和长动作(COIN, Breakfast)基准测试中，StretchySnake超越了transformer和SSM基线，性能提升高达28%，对精细动作(SSV2, Diving-48)具有强大的适应能力。因此，我们的方法提供了一种简单的即插即用训练方案，使视频SSMs在各种动作识别场景中更加健壮、分辨率无关且高效。


### 论文摘要

State space models (SSMs) have emerged as a competitive alternative to transformers in various tasks. Their linear complexity and hidden-state recurrence make them particularly attractive for modeling long sequences, whereas attention becomes quadratically expensive. However, current training methods for video understanding are tailored towards transformers and fail to fully leverage the unique attributes of SSMs. For example, video models are often trained at a fixed resolution and video length to balance the quadratic scaling of attention cost against performance. Consequently, these models suffer from degraded performance when evaluated on videos with spatial and temporal resolutions unseen during training; a property we call spatio-temporal inflexibility. In the context of action recognition, this severely limits a model's ability to retain performance across both short- and long-form videos. Therefore, we propose a flexible training method that leverages and improves the inherent adaptability of SSMs. Our method samples videos at varying temporal and spatial resolutions during training and dynamically interpolates model weights to accommodate any spatio-temporal scale. This instills our SSM, which we call StretchySnake, with spatio-temporal flexibility and enables it to seamlessly handle videos ranging from short, fine-grained clips to long, complex activities. We introduce and compare five different variants of flexible training, and identify the most effective strategy for video SSMs. On short-action (UCF-101, HMDB-51) and long-action (COIN, Breakfast) benchmarks, StretchySnake outperforms transformer and SSM baselines alike by up to 28%, with strong adaptability to fine-grained actions (SSV2, Diving-48). Therefore, our method provides a simple drop-in training recipe that makes video SSMs more robust, resolution-agnostic, and efficient across diverse action recognition scenarios.

---

## 44. Temporal Referential Consistency: Do LLMs Favor Sequences Over Absolute Time References?

**论文链接:** [http://arxiv.org/abs/2510.15513v1](http://arxiv.org/abs/2510.15513v1)

**作者:** Ashutosh Bajpai, Tanmoy Chakraborty

**发布时间:** 2025-10-17

**备注:** EMNLP Main Long Paper 2025

### GPT解析

### 总结

本文针对大型语言模型(LLMs)在时间参考一致性方面的不足，提出了一个新的基准测试TEMP-ReCon和一种基于推理路径对齐的模型UnTRaP，以增强LLMs在时间敏感领域的时间推理能力。

### 背景

大型语言模型(LLMs)正越来越多地被作为知识来源的替代品，在法律、医疗保健和金融等时间敏感领域尤为显著。LLMs需要具备事实准确性和时间维度上的一致性，但目前确保LLMs时间一致性的努力仍然稀缺。

### 目的

引入一个名为'temporal referential consistency'的新基准，并开发资源TEMP-ReCon，用于评估各种开源和闭源LLMs在不同语言环境(包括英语、法语和罗马尼亚语)中的时间参考一致性。

### 方法

提出UnTRaP模型，这是一种基于推理路径对齐的模型，旨在提高LLMs的时间参考一致性。通过实验验证其与基线模型相比的有效性。

### 主要发现

研究发现大型语言模型确实表现出不足的时间参考一致性。通过引入新基准和资源，以及提出UnTRaP模型，可以有效解决这一问题。

### 结论

UnTRaP模型相比几个基线模型更为有效，能够增强LLMs的时间参考一致性，为LLMs在时间敏感领域的应用提供了更好的解决方案。

### 翻译

大型语言模型(LLMs)作为知识来源替代品的日益普及标志着各领域的重要范式转变，包括法律、医疗保健和金融等时间敏感领域。为了满足这一扩展角色，LLMs不仅需要事实准确，还需要在时间维度上表现出一致性，这需要强大的时间推理能力。尽管有这一关键需求，确保LLMs时间一致性的努力仍然很少，包括在时间敏感查询中评估或增强LLMs时间参考方面的明显缺失。在本文中，我们通过引入一个名为'temporal referential consistency'的新基准以及资源TEMP-ReCon来填补这一空白，该资源用于评估各种开源和闭源LLMs在不同资源丰富度的语言环境(包括英语、法语和罗马尼亚语)中的表现。研究结果强调LLMs确实表现出不足的时间参考一致性。为此，我们提出了UnTRaP，一种基于推理路径对齐的模型，旨在提高LLMs的时间参考一致性。我们的实证实验证明了UnTRaP相比几个基线模型的有效性。


### 论文摘要

The increasing acceptance of large language models (LLMs) as an alternative to knowledge sources marks a significant paradigm shift across various domains, including time-sensitive fields such as law, healthcare, and finance. To fulfill this expanded role, LLMs must not only be factually accurate but also demonstrate consistency across temporal dimensions, necessitating robust temporal reasoning capabilities. Despite this critical requirement, efforts to ensure temporal consistency in LLMs remain scarce including noticeable absence of endeavors aimed at evaluating or augmenting LLMs across temporal references in time-sensitive inquiries. In this paper, we seek to address this gap by introducing a novel benchmark entitled temporal referential consistency, accompanied by a resource TEMP-ReCon designed to benchmark a wide range of both open-source and closed-source LLMs with various linguistic contexts characterized by differing resource richness (including English, French, and Romanian). The findings emphasis that LLMs do exhibit insufficient temporal referent consistency. To address this, we propose \newmodel, a reasoning path alignment-based model that aims to enhance the temporal referential consistency of LLMs. Our empirical experiments substantiate the efficacy of UnTRaP compared to several baseline models.

---

## 45. XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models

**论文链接:** [http://arxiv.org/abs/2510.15148v1](http://arxiv.org/abs/2510.15148v1)

**作者:** Xingrui Wang, Jiang Liu, Chao Huang, Xiaodong Yu, Ze Wang, Ximeng Sun, Jialian Wu, Alan Yuille, Emad Barsoum, Zicheng Liu

**发布时间:** 2025-10-16

### GPT解析

### 总结

该研究引入了XModBench，一个用于评估全模态大语言模型(OLLMs)跨模态一致性的基准测试。研究发现，即使是目前最强的模型如Gemini 2.5 Pro，在空间和时间推理方面表现不佳，存在模态差异和方向性不平衡问题，表明当前OLLMs距离真正的模态不变推理还有很大差距。

### 背景

全模态大语言模型(OLLMs)旨在统一音频、视觉和文本理解于单一框架。现有基准主要评估通用跨模态问答能力，但不清楚OLLMs是否实现了模态不变推理或存在模态特定偏差。

### 目的

引入XModBench，一个大规模的三模态基准，专门用于测量跨模态一致性，评估OLLMs的模态不变推理能力、模态差异和方向性不平衡。

### 方法

XModBench包含60,828个多选题，涵盖五个任务家族，系统覆盖了问答对中的所有六种模态组合，能够对OLLM的模态不变推理、模态差异和方向性不平衡进行细粒度诊断。

### 主要发现

即使是目前最强的模型Gemini 2.5 Pro，(i)在空间和时间推理方面表现不佳，准确率低于60%，(ii)存在持续的模态差异，当相同语义内容通过音频而非文本传达时性能显著下降，(iii)表现出系统性的方向性不平衡，当视觉作为上下文时一致性低于文本。

### 结论

当前OLLMs距离真正的模态不变推理还有很长的路要走，XModBench可作为评估和改进跨模态能力的基本诊断工具。

### 翻译

全模态大语言模型(OLLMs)旨在在单一框架内统一音频、视觉和文本理解。虽然现有基准主要评估通用的跨模态问答能力，但尚不清楚OLLMs是否实现了模态不变的推理或表现出模态特定的偏差。我们引入了XModBench，一个大规模的三模态基准，专门设计用于测量跨模态一致性。XModBench包含60,828个多选题，涵盖五个任务家族，并系统性地覆盖了问答对中的所有六种模态组合，能够对OLLM的模态不变推理、模态差异和方向性不平衡进行细粒度诊断。实验表明，即使是最强的模型Gemini 2.5 Pro，(i)在空间和时间推理方面表现不佳，准确率低于60%，(ii)表现出持续的模态差异，当相同语义内容通过音频而非文本传达时，性能显著下降，(iii)显示出系统性的方向性不平衡，当视觉作为上下文时，一致性低于文本。这些发现表明，当前的OLLMs距离真正的模态不变推理还有很长的路要走，并将XModBench定位为评估和改进跨模态能力的基本诊断工具。所有数据和评估工具将在https://xingruiwang.github.io/projects/XModBench/提供。


### 论文摘要

Omni-modal large language models (OLLMs) aim to unify audio, vision, and text understanding within a single framework. While existing benchmarks primarily evaluate general cross-modal question-answering ability, it remains unclear whether OLLMs achieve modality-invariant reasoning or exhibit modality-specific biases. We introduce XModBench, a large-scale tri-modal benchmark explicitly designed to measure cross-modal consistency. XModBench comprises 60,828 multiple-choice questions spanning five task families and systematically covers all six modality compositions in question-answer pairs, enabling fine-grained diagnosis of an OLLM's modality-invariant reasoning, modality disparity, and directional imbalance. Experiments show that even the strongest model, Gemini 2.5 Pro, (i) struggles with spatial and temporal reasoning, achieving less than 60% accuracy, (ii) reveals persistent modality disparities, with performance dropping substantially when the same semantic content is conveyed through audio rather than text, and (iii) shows systematic directional imbalance, exhibiting lower consistency when vision serves as context compared to text. These findings indicate that current OLLMs remain far from truly modality-invariant reasoning and position XModBench as a fundamental diagnostic tool for evaluating and improving cross-modal competence. All data and evaluation tools will be available at https://xingruiwang.github.io/projects/XModBench/.

---

## 46. FUSE-Traffic: Fusion of Unstructured and Structured Data for Event-aware Traffic Forecasting

**论文链接:** [http://arxiv.org/abs/2510.16053v1](http://arxiv.org/abs/2510.16053v1)

**作者:** Chenyang Yu, Xinpeng Xie, Yan Huang, Chenxi Qiu

**发布时间:** 2025-10-16

**DOI:** 10.1145/3748636.3762776

### GPT解析

### 总结

交通预测是智能交通系统的核心技术，图神经网络已成为该领域的主流方法，但在处理事件信息方面仍面临挑战。

### 背景

随着城市化进程加快，交通拥堵问题加剧，需要可靠且响应迅速的预测模型来改善城市资源分配和出行体验。

### 目的

开发能够有效捕捉交通网络空间依赖关系和时间演化模式的预测模型，提高对复杂交通状况的响应能力。

### 方法

采用图神经网络(GNNs)作为主要技术路线，结合图卷积结构和时间建模机制，包括STGCN、GraphWaveNet、STWave和D2STGNN等模型，并探索融入事件信息的方法。

### 主要发现

GNNs在捕捉周期性交通模式方面特别有效；早期基于人工特征的方法虽能提升对特定事件的响应，但严重依赖领域专家先验知识，难以泛化到复杂未知事件，且低维人工特征会导致语义细节丢失。

### 结论

需要减少对人工特征的依赖，开发更有效的方法来处理交通预测中的事件信息，提高模型对未知事件的泛化能力。

### 翻译

准确的交通预测是构建智能交通系统的核心技术，能够更好地进行城市资源分配和改善出行体验。随着城市化的发展，交通拥堵加剧，凸显了对可靠且响应迅速的预测模型的需求。近年来，深度学习，特别是图神经网络(GNNs)，已成为交通预测的主流范式。GNNs能够有效捕捉道路网络拓扑中的复杂空间依赖关系和交通流量数据中的动态时间演化模式。诸如STGCN和GraphWaveNet等基础模型，以及包括STWave和D2STGNN在内的最新发展，在标准交通数据集上取得了令人印象深刻的性能。这些方法结合了复杂的图卷积结构和时间建模机制，在捕捉和预测具有周期性规律的交通模式方面表现出特别的有效性。为了应对这一挑战，研究人员探索了多种融入事件信息的方式。早期尝试主要依赖人工设计的特征。例如，一些方法引入了人工定义的事件影响分数，或为不同事件引起的交通状况构建特定的子图。虽然这些方法在某种程度上增强了对特定事件的响应能力，但其主要缺点在于严重依赖领域专家的先验知识，使得对多样且复杂的未知事件的泛化变得困难，而低维人工特征往往导致丰富语义细节的丢失。


### 论文摘要

Accurate traffic forecasting is a core technology for building Intelligent Transportation Systems (ITS), enabling better urban resource allocation and improved travel experiences. With growing urbanization, traffic congestion has intensified, highlighting the need for reliable and responsive forecasting models. In recent years, deep learning, particularly Graph Neural Networks (GNNs), has emerged as the mainstream paradigm in traffic forecasting. GNNs can effectively capture complex spatial dependencies in road network topology and dynamic temporal evolution patterns in traffic flow data. Foundational models such as STGCN and GraphWaveNet, along with more recent developments including STWave and D2STGNN, have achieved impressive performance on standard traffic datasets. These approaches incorporate sophisticated graph convolutional structures and temporal modeling mechanisms, demonstrating particular effectiveness in capturing and forecasting traffic patterns characterized by periodic regularities. To address this challenge, researchers have explored various ways to incorporate event information. Early attempts primarily relied on manually engineered event features. For instance, some approaches introduced manually defined incident effect scores or constructed specific subgraphs for different event-induced traffic conditions. While these methods somewhat enhance responsiveness to specific events, their core drawback lies in a heavy reliance on domain experts' prior knowledge, making generalization to diverse and complex unknown events difficult, and low-dimensional manual features often lead to the loss of rich semantic details.

---

## 47. Intelligent Communication Mixture-of-Experts Boosted-Medical Image Segmentation Foundation Model

**论文链接:** [http://arxiv.org/abs/2510.17684v1](http://arxiv.org/abs/2510.17684v1)

**作者:** Xinwei Zhang, Hu Chen, Zhe Yuan, Sukun Tian, Peng Feng

**发布时间:** 2025-10-20

### GPT解析

### 总结

本文提出了IC-MoE模型，一种智能通信混合专家增强的医学图像分割基础模型，解决了现有微调方法中高级特征表示不足和预训练权重结构完整性受损的问题。

### 背景

基础模型在医学图像分割领域已取得显著性能，自适应微调自然图像分割基础模型对医学图像分割任务至关重要。然而，现有微调方法存在两个局限性：高级特征表示不足和微调过程破坏预训练权重的结构完整性。

### 目的

解决现有微调方法的局限性，提出一个能够增强高级特征表示同时保持预训练权重结构完整性的医学图像分割基础模型。

### 方法

提出IC-MoE模型，包含两个核心创新：1) 构建基础专家、语义专家和自适应专家，实现像素概率自适应投票策略，通过标签一致性和负载平衡进行专家选择和融合；2) 提出语义引导的对比学习方法，解决对比学习中的弱监督问题。

### 主要发现

在三个公共医学图像分割数据集上的大量实验表明，IC-MoE优于其他最先进模型。IC-MoE有效地为基础医学图像分割模型补充了高级特征和预训练结构完整性，并在多样化医学图像分割场景中展现出优越的泛化能力。

### 结论

IC-MoE模型成功解决了现有微调方法的局限性，能够在增强高级特征表示的同时保持预训练权重的结构完整性，为医学图像分割任务提供了有效的解决方案。

### 翻译

医学图像分割的基础模型已取得显著性能。自然图像分割基础模型的自适应微调对医学图像分割任务至关重要。然而，现有微调方法存在一些局限性：1) 高级特征表示不足；2) 微调过程破坏了预训练权重的结构完整性。受这些关键问题的启发，我们提出了一个智能通信混合专家增强的医学图像分割基础模型，名为IC-MoE，包含两个核心想法：1) 我们构建基础专家、语义专家和自适应专家。此外，我们实现了像素概率自适应投票策略，通过标签一致性和负载平衡实现专家选择和融合。这种方法初步增强了高级特征的表示能力，同时保留了预训练权重的结构完整性。2) 我们提出了一种语义引导的对比学习方法，解决了对比学习中弱监督的问题。这种方法进一步增强了高级特征的表示能力，同时保留了预训练权重的结构完整性。在三个公共医学图像分割数据集上的大量实验表明，IC-MoE优于其他最先进的模型。因此，所提出的IC-MoE有效地为基础医学图像分割模型补充了高级特征和预训练结构完整性。我们还验证了IC-MoE在多样化医学图像分割场景中的优越泛化能力。


### 论文摘要

Foundation models for medical image segmentation have achieved remarkable performance. Adaptive fine-tuning of natural image segmentation foundation models is crucial for medical image segmentation tasks. However, some limitations exist in existing fine-tuning methods: 1) insufficient representation of high-level features and 2) the fine-tuning process disrupts the structural integrity of pretrained weights. Inspired by these critical problems, we propose an intelligent communication mixture-of-experts boosted-medical image segmentation foundation model, named IC-MoE, with twofold ideas: 1) We construct basic experts, semantic experts, and adaptive experts. Moreover, we implement a pixel probability adaptive voting strategy, which enables expert selection and fusion through label consistency and load balancing. This approach preliminarily enhances the representation capability of high-level features while preserving the structural integrity of pretrained weights. 2) We propose a semantic-guided contrastive learning method to address the issue of weak supervision in contrastive learning. This method further enhances the representation capability of high-level features while preserving the structural integrity of pretrained weights. Extensive experiments across three public medical image segmentation datasets demonstrate that the IC-MoE outperforms other SOTA models. Consequently, the proposed IC-MoE effectively supplements foundational medical image segmentation models with high-level features and pretrained structural integrity. We also validate the superior generalizability of the IC-MoE across diverse medical image segmentation scenarios.

---

## 48. Curiosity-driven RL for symbolic equation solving

**论文链接:** [http://arxiv.org/abs/2510.17022v1](http://arxiv.org/abs/2510.17022v1)

**作者:** Kevin P. O Keeffe

**发布时间:** 2025-10-19

**备注:** Accepted at the NeurIPS 2025 MATH-AI Workshop

### GPT解析

### 总结

研究展示了增强的PPO算法在解决符号数学问题上的有效性，特别是能够处理涉及根式、指数和三角函数的非线性方程。

### 背景

先前的研究表明对比学习可以解决单变量线性方程，但强化学习在符号数学领域的应用尚未充分探索。

### 目的

探索强化学习是否可以有效地应用于符号数学问题。

### 方法

使用无模型PPO算法，并加入基于好奇心的探索机制和基于图的动作表示。

### 主要发现

所提出的方法能够解决涉及根式、指数和三角函数的非线性方程，而不仅仅是简单的线性方程。

### 结论

基于好奇心的探索机制可能对解决一般符号推理任务具有实用价值。

### 翻译

我们探索强化学习是否可以用于符号数学。先前的工作表明对比学习可以解决单变量线性方程。我们展示了无模型PPO结合基于好奇心的探索和基于图的动作可以解决非线性方程，如涉及根式、指数和三角函数的方程。我们的研究表明基于好奇心的探索可能对一般符号推理任务有用。


### 论文摘要

We explore if RL can be useful for symbolic mathematics. Previous work showed contrastive learning can solve linear equations in one variable. We show model-free PPO \cite{schulman2017proximal} augmented with curiosity-based exploration and graph-based actions can solve nonlinear equations such as those involving radicals, exponentials, and trig functions. Our work suggests curiosity-based exploration may be useful for general symbolic reasoning tasks.

---

## 49. MOSAIC: Masked Objective with Selective Adaptation for In-domain Contrastive Learning

**论文链接:** [http://arxiv.org/abs/2510.16797v1](http://arxiv.org/abs/2510.16797v1)

**作者:** Vera Pavlova, Mohammed Makhlouf

**发布时间:** 2025-10-19

### GPT解析

### 总结

MOSAIC是一种多阶段框架，用于句子嵌入模型的领域自适应，结合了领域特定的掩码监督。

### 背景

大规模通用领域句子嵌入模型适应到专业领域面临的挑战。

### 目的

有效学习领域相关表示，同时保持原始模型的强语义区分特性。

### 方法

通过在统一训练流程中联合优化掩码语言模型(MLM)和对比目标。

### 主要发现

在高资源和低资源领域都取得了显著提升，NDCG@10指标比基线提高最多13.4%。

### 结论

平衡的联合监督和阶段适应对有效领域自适应至关重要。

### 翻译

我们介绍了MOSAIC（具有选择性适应的掩码目标用于领域内对比学习），这是一种用于句子嵌入模型领域自适应的多阶段框架，它结合了联合领域特定的掩码监督。我们的方法解决了将大规模通用领域句子嵌入模型适应到专业领域的挑战。通过在统一的训练流程中联合优化掩码语言模型(MLM)和对比目标，我们的方法能够有效地学习领域相关表示，同时保持原始模型的强语义区分特性。我们在高资源和低资源领域上都经验性地验证了我们的方法，在NDCG@10（标准化折损累积增益）上比强大的通用领域基线提高了最多13.4%。全面的消融研究进一步证明了每个组件的有效性，强调了平衡联合监督和阶段适应的重要性。


### 论文摘要

We introduce MOSAIC (Masked Objective with Selective Adaptation for In-domain Contrastive learning), a multi-stage framework for domain adaptation of sentence embedding models that incorporates joint domain-specific masked supervision. Our approach addresses the challenges of adapting large-scale general-domain sentence embedding models to specialized domains. By jointly optimizing masked language modeling (MLM) and contrastive objectives within a unified training pipeline, our method enables effective learning of domain-relevant representations while preserving the robust semantic discrimination properties of the original model. We empirically validate our approach on both high-resource and low-resource domains, achieving improvements up to 13.4% in NDCG@10 (Normalized Discounted Cumulative Gain) over strong general-domain baselines. Comprehensive ablation studies further demonstrate the effectiveness of each component, highlighting the importance of balanced joint supervision and staged adaptation.

---

## 50. Connecting Domains and Contrasting Samples: A Ladder for Domain Generalization

**论文链接:** [http://arxiv.org/abs/2510.16704v1](http://arxiv.org/abs/2510.16704v1)

**作者:** Tianxin Wei, Yifan Chen, Xinrui He, Wenxuan Bao, Jingrui He

**发布时间:** 2025-10-19

**DOI:** 10.1145/3690624.3709280

**备注:** Accepted by KDD 2025

### GPT解析

### 总结

该论文提出了一种新的领域连接对比学习(DCCL)方法，用于解决领域泛化(DG)中的问题。研究发现直接应用对比学习(CL)会降低DG性能，原因是缺乏类内连接。DCCL通过改进数据增强和跨域正样本，以及提出模型锚定和生成变换损失来增强类内连接，从而提升DG性能。实验表明，该方法在五个标准DG基准上优于最先进的基线方法，且不需要领域监督。

### 背景

训练和测试样本之间的分布偏移在实践中经常发生，阻碍了模型的泛化性能。这促使了对领域泛化(DG)的研究，DG旨在仅使用源域数据来预测未见过的目标域数据的标签。

### 目的

解决直接应用对比学习(CL)降低领域泛化(DG)性能的问题，增强跨领域的概念连接，获得适用于DG的可泛化表示。

### 方法

提出领域连接对比学习(DCCL)范式。在数据方面，引入更积极的数据增强和跨域正样本以改善类内连接；在模型方面，提出模型锚定来利用预训练表示中的类内连接，并通过生成变换损失补充锚定。

### 主要发现

在DG设置中缺乏类内连接是导致直接应用CL降低性能的原因；提出的DCCL方法通过增强类内连接，在五个标准DG基准上优于最先进的基线方法，且不需要领域监督。

### 结论

领域连接对比学习(DCCL)是解决DG中分布偏移问题的有效方法，通过增强类内连接和跨领域概念连接，能够获得更好的可泛化表示。

### 翻译

训练和测试样本之间的分布偏移在实践中经常发生，并阻碍了模型的泛化性能。这一关键挑战促使了对领域泛化(DG)的研究，DG旨在仅使用源域数据来预测未见过的目标域数据的标签。直观上，对比学习(CL)中学到的类分离表示应该能够改善DG，但实际情况恰恰相反：直接应用CL会降低性能。作者通过CL理论的见解分析了这一现象，发现在DG设置中缺乏类内连接导致了这一缺陷。因此，作者提出了一个新的范式——领域连接对比学习(DCCL)，以增强跨领域的概念连接并获得适用于DG的可泛化表示。在数据方面，引入更积极的数据增强和跨域正样本以改善类内连接；在模型方面，为更好地嵌入未见过的测试域，作者提出模型锚定来利用预训练表示中的类内连接，并通过生成变换损失补充锚定。在五个标准的DG基准上进行了大量实验，结果验证了DCCL优于最先进的基线方法，甚至不需要领域监督。详细的模型实现和代码可通过https://github.com/weitianxin/DCCL获取。


### 论文摘要

Distribution shifts between training and testing samples frequently occur in practice and impede model generalization performance. This crucial challenge thereby motivates studies on domain generalization (DG), which aim to predict the label on unseen target domain data by solely using data from source domains. It is intuitive to conceive the class-separated representations learned in contrastive learning (CL) are able to improve DG, while the reality is quite the opposite: users observe directly applying CL deteriorates the performance. We analyze the phenomenon with the insights from CL theory and discover lack of intra-class connectivity in the DG setting causes the deficiency. We thus propose a new paradigm, domain-connecting contrastive learning (DCCL), to enhance the conceptual connectivity across domains and obtain generalizable representations for DG. On the data side, more aggressive data augmentation and cross-domain positive samples are introduced to improve intra-class connectivity. On the model side, to better embed the unseen test domains, we propose model anchoring to exploit the intra-class connectivity in pre-trained representations and complement the anchoring with generative transformation loss. Extensive experiments on five standard DG benchmarks are performed. The results verify that DCCL outperforms state-of-the-art baselines even without domain supervision. The detailed model implementation and the code are provided through https://github.com/weitianxin/DCCL

---

## 51. Enhancing Compositional Reasoning in CLIP via Reconstruction and Alignment of Text Descriptions

**论文链接:** [http://arxiv.org/abs/2510.16540v1](http://arxiv.org/abs/2510.16540v1)

**作者:** Jihoon Kwon, Kyle Min, Jy-yong Sohn

**发布时间:** 2025-10-18

**备注:** Accepted at NeurIPS 2025 (poster). This is the camera-ready version

### GPT解析

### 总结

本文提出READ方法，通过添加两个辅助目标到对比学习中，增强视觉-语言模型的组合推理能力。READ-CLIP在五个主要组合推理基准测试中取得最先进性能，比传统微调基线高4.1%。

### 背景

尽管近期有所进展，但使用标准对比目标训练的视觉-语言模型在组合推理（理解视觉和语言元素间结构关系）方面仍存在困难。

### 目的

开发一种微调方法，增强视觉-语言模型的组合推理能力，解决文本编码器关注单词而非单词间关系的问题。

### 方法

引入READ方法，添加两个辅助目标到对比学习中：(1)令牌级重建目标，使用冻结预训练解码器重建替代标题；(2)句子级对齐目标，在嵌入空间中对齐释义句子。

### 主要发现

READ-CLIP在五个组合推理基准测试中达到最先进性能，比最强基线高4.1%；READ应用于其他CLIP变体也提高了性能；重建和对齐目标提供互补好处。

### 结论

READ方法有效增强了视觉-语言模型的组合推理能力，重建目标促进编码器捕获单词间关系，对齐目标确保不同措辞表达的释义具有一致表示。

### 翻译

尽管近期有所进展，但使用标准对比目标训练的视觉-语言模型仍然在组合推理——即理解视觉和语言元素之间结构关系的能力——方面存在困难。这一缺点主要是由于文本编码器倾向于关注单个单词而非它们之间的关系，这种局限性通过主要将单词与视觉对象对齐的对比训练得到了强化。在本文中，我们引入了READ（文本描述的重建和对齐），这是一种微调方法，通过向对比学习中添加两个辅助目标来增强组合推理能力：(1)令牌级重建目标，其中冻结的预训练解码器基于原始标题的嵌入重建替代标题；(2)句子级对齐目标，在嵌入空间中明确对齐释义句子。我们表明，通过将READ方法应用于预训练的CLIP模型得到的READ-CLIP模型在五个主要的组合推理基准测试中取得了最先进的性能，比最强的传统微调基线高出最多4.1%。此外，将READ应用于现有的CLIP变体（包括NegCLIP和FSC-CLIP）也提高了这些基准测试的性能。定量和定性分析表明，我们提出的目标——重建和对齐——提供了互补的好处：前者鼓励编码器捕获标题中单词之间的关系，而后者确保用不同措辞表达的释义具有一致的表示。


### 论文摘要

Despite recent advances, vision-language models trained with standard contrastive objectives still struggle with compositional reasoning -- the ability to understand structured relationships between visual and linguistic elements. This shortcoming is largely due to the tendency of the text encoder to focus on individual words rather than their relations, a limitation reinforced by contrastive training that primarily aligns words with visual objects. In this paper, we introduce REconstruction and Alignment of text Descriptions (READ), a fine-tuning method designed to enhance compositional reasoning by adding two auxiliary objectives to the contrastive learning: (1) a token-level reconstruction objective, where a frozen pre-trained decoder reconstructs alternative captions based on the embedding of the original caption; and (2) a sentence-level alignment objective, which explicitly aligns paraphrased sentences in the embedding space. We show that READ-CLIP, a model derived by applying the READ method to the pre-trained CLIP model, achieves the state-of-the-art performance across five major compositional reasoning benchmarks, outperforming the strongest conventional fine-tuning baseline by up to 4.1%. Furthermore, applying the READ to existing CLIP variants (including NegCLIP and FSC-CLIP) also improves performance on these benchmarks. Quantitative and qualitative analyses reveal that our proposed objectives -- reconstruction and alignment -- offer complementary benefits: the former encourages the encoder to capture relationships between words within a caption, while the latter ensures consistent representations for paraphrases expressed with different wording.

---

## 52. Instance-Aware Pseudo-Labeling and Class-Focused Contrastive Learning for Weakly Supervised Domain Adaptive Segmentation of Electron Microscopy

**论文链接:** [http://arxiv.org/abs/2510.16450v1](http://arxiv.org/abs/2510.16450v1)

**作者:** Shan Xiong, Jiabao Chen, Ye Wang, Jialin Peng

**发布时间:** 2025-10-18

### GPT解析

### 总结

本研究提出了一种弱监督域适应方法，用于电子显微镜图像中线粒体的高效分割，通过多任务学习框架和实例感知的伪标签选择策略，显著提高了分割性能。

### 背景

从各种电子显微镜图像中分割大量线粒体实例对生物和神经科学研究具有重要价值。无监督域适应方法虽然可以缓解域偏移并降低标注成本，但在实际应用中性能较低。

### 目的

研究弱监督域适应(WDA)方法，利用目标域上的稀疏点标签，这些标签需要最少的标注工作和专业知识，以实现高效准确的线粒体分割。

### 方法

引入一个多任务学习框架，同时进行分割和中心检测，采用新颖的交叉教学机制和面向类的跨域对比学习。提出分割自训练，使用实例感知的伪标签(IPL)选择策略，帮助选择语义上可靠和多样的伪标签。

### 主要发现

在具有挑战性的数据集上验证，该方法优于现有的UDA和WDA方法，显著缩小了与监督上限的性能差距。在UDA设置下，也显著优于其他UDA技术。

### 结论

所提出的弱监督域适应方法通过有效利用稀疏点标注和实例感知的伪标签策略，实现了电子显微镜图像中线粒体的高效分割，为生物和神经科学研究提供了有力工具。

### 翻译

从各种电子显微镜图像中高效分割大量线粒体实例对生物和神经科学研究非常有价值。尽管无监督域适应方法可以帮助缓解域偏移并降低每个域的标注成本，但它们在实际应用中通常具有相对较低的性能。因此，我们研究了弱监督域适应(WDA)，它利用目标域上的额外稀疏点标签，这些标签需要最少的标注工作和最少的专家知识。为了充分利用不完整和不精确的点标注，我们引入了一个多任务学习框架，通过新颖的交叉教学机制和面向类的跨域对比学习共同进行分割和中心检测。虽然利用未标记的图像区域至关重要，我们引入了分割自训练，采用新颖的实例感知的伪标签(IPL)选择策略。与通常依赖像素级伪标签过滤的现有方法不同，IPL在检测任务的帮助下，在语义上选择可靠和多样的伪标签。在具有挑战性的数据集上进行的全面验证和比较表明，我们的方法优于现有的UDA和WDA方法，显著缩小了与监督上限的性能差距。此外，在UDA设置下，我们的方法也实现了对其他UDA技术的显著改进。


### 论文摘要

Annotation-efficient segmentation of the numerous mitochondria instances from various electron microscopy (EM) images is highly valuable for biological and neuroscience research. Although unsupervised domain adaptation (UDA) methods can help mitigate domain shifts and reduce the high costs of annotating each domain, they typically have relatively low performance in practical applications. Thus, we investigate weakly supervised domain adaptation (WDA) that utilizes additional sparse point labels on the target domain, which require minimal annotation effort and minimal expert knowledge. To take full use of the incomplete and imprecise point annotations, we introduce a multitask learning framework that jointly conducts segmentation and center detection with a novel cross-teaching mechanism and class-focused cross-domain contrastive learning. While leveraging unlabeled image regions is essential, we introduce segmentation self-training with a novel instance-aware pseudo-label (IPL) selection strategy. Unlike existing methods that typically rely on pixel-wise pseudo-label filtering, the IPL semantically selects reliable and diverse pseudo-labels with the help of the detection task. Comprehensive validations and comparisons on challenging datasets demonstrate that our method outperforms existing UDA and WDA methods, significantly narrowing the performance gap with the supervised upper bound. Furthermore, under the UDA setting, our method also achieves substantial improvements over other UDA techniques.

---

## 53. Toward General Digraph Contrastive Learning: A Dual Spatial Perspective

**论文链接:** [http://arxiv.org/abs/2510.16311v1](http://arxiv.org/abs/2510.16311v1)

**作者:** Daohan Su, Yang Zhang, Xunkai Li, Rong-Hua Li, Guoren Wang

**发布时间:** 2025-10-18

### GPT解析

### 总结

S2-DiGCL是一种针对有向图对比学习的新型框架，通过结合复域和实域的空间视角，构建高质量的正负样本，实现更通用和鲁棒的有向图对比学习。

### 背景

现有图对比学习方法主要关注无向图，忽略了现实网络(如社交网络和推荐系统)中基本且不可或缺的方向信息。

### 目的

开发一种能够从复杂和真实领域角度强调空间洞察力的有向图对比学习框架，以捕获方向信息。

### 方法

S2-DiGCL从复域角度将个性化扰动引入磁拉普拉斯矩阵以自适应调整边相位和方向语义；从实域角度采用基于路径的子图增强策略来捕获细粒度的局部非对称性和拓扑依赖。

### 主要发现

在7个真实有向图数据集上的广泛实验表明，S2-DiGCL方法具有优越性，在监督和非监督设置下，节点分类和链路预测任务均达到了SOTA性能，分别提高了4.41%和4.34%。

### 结论

通过联合利用互补的空间视角，S2-DiGCL能够构建高质量的正负样本，实现更通用和鲁棒的有向图对比学习。

### 翻译

图对比学习(GCL)已成为从图中提取一致表示的强大工具，独立于标记信息。然而，现有方法主要关注无向图，忽略了现实网络(如社交网络和推荐系统)中基本且不可或缺的方向信息。本文介绍了S2-DiGCL，一种新型框架，从复杂和真实领域角度强调有向图(有向图)对比学习的空间洞察力。从复域角度，S2-DiGCL将个性化扰动引入磁拉普拉斯矩阵，以自适应调整边相位和方向语义。从实域角度，它采用基于路径的子图增强策略来捕获细粒度的局部非对称性和拓扑依赖。通过联合利用这两个互补的空间视角，S2-DiGCL构建高质量的正负样本，实现更通用和鲁棒的有向图对比学习。在7个真实有向图数据集上的广泛实验证明了我们方法的优越性，在监督和非监督设置下，节点分类和链路预测任务均达到了SOTA性能，分别提高了4.41%和4.34%。


### 论文摘要

Graph Contrastive Learning (GCL) has emerged as a powerful tool for extracting consistent representations from graphs, independent of labeled information. However, existing methods predominantly focus on undirected graphs, disregarding the pivotal directional information that is fundamental and indispensable in real-world networks (e.g., social networks and recommendations).In this paper, we introduce S2-DiGCL, a novel framework that emphasizes spatial insights from complex and real domain perspectives for directed graph (digraph) contrastive learning. From the complex-domain perspective, S2-DiGCL introduces personalized perturbations into the magnetic Laplacian to adaptively modulate edge phases and directional semantics. From the real-domain perspective, it employs a path-based subgraph augmentation strategy to capture fine-grained local asymmetries and topological dependencies. By jointly leveraging these two complementary spatial views, S2-DiGCL constructs high-quality positive and negative samples, leading to more general and robust digraph contrastive learning. Extensive experiments on 7 real-world digraph datasets demonstrate the superiority of our approach, achieving SOTA performance with 4.41% improvement in node classification and 4.34% in link prediction under both supervised and unsupervised settings.

---

## 54. SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection

**论文链接:** [http://arxiv.org/abs/2510.16219v1](http://arxiv.org/abs/2510.16219v1)

**作者:** Yang Feng, Xudong Pan

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出SentinelNet，首个用于多智能体系统中主动检测和减轻恶意行为的去中心化框架，通过基于信誉的检测器和动态邻居排名实现高效防御。

### 背景

恶意智能体对基于大型语言模型的多智能体系统的可靠性和决策能力构成重大威胁，现有防御措施因反应式设计或集中式架构存在单点故障问题而效果不佳。

### 目的

开发一个去中心化框架，能够主动检测并减轻多智能体协作中的恶意行为，提高系统安全性。

### 方法

为每个智能体配备基于信誉的检测器，通过对比学习在增强的对抗辩论轨迹上进行训练，实现消息可信度自主评估和动态邻居排名，并通过生成对抗轨迹解决攻击数据稀缺问题。

### 主要发现

SentinelNet在多智能体系统基准测试中实现了接近100%的恶意智能体检测率（两轮内），能从受损系统中恢复95%的准确性，并展现出跨领域和攻击模式的强泛化能力。

### 结论

SentinelNet为保护协作多智能体系统建立了新的防御范式，有效解决了现有防御机制的局限性。

### 翻译

恶意智能体对由大型语言模型驱动的多智能体系统的可靠性和决策能力构成重大威胁。现有防御往往因反应式设计或集中式架构而不足，这些架构可能引入单点故障。为解决这些挑战，我们提出SentinelNet，首个用于主动检测和减轻多智能体协作中恶意行为的去中心化框架。SentinelNet为每个智能体配备基于信誉的检测器，通过在增强的对抗辩论轨迹上进行对比学习训练，使智能体能够自主评估消息可信度并通过bottom-k消除进行动态邻居排名，以抑制恶意通信。为克服攻击数据稀缺问题，它生成模拟各种威胁的对抗轨迹，确保稳健训练。在多智能体系统基准测试中，SentinelNet实现了对恶意智能体的近乎完美检测，在两轮辩论内接近100%，并从受损的基线系统中恢复95%的准确性。通过在不同领域和攻击模式中展现强大的泛化能力，SentinelNet为保护协作多智能体系统建立了新的范式。


### 论文摘要

Malicious agents pose significant threats to the reliability and decision-making capabilities of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs). Existing defenses often fall short due to reactive designs or centralized architectures which may introduce single points of failure. To address these challenges, we propose SentinelNet, the first decentralized framework for proactively detecting and mitigating malicious behaviors in multi-agent collaboration. SentinelNet equips each agent with a credit-based detector trained via contrastive learning on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking via bottom-k elimination to suppress malicious communications. To overcome the scarcity of attack data, it generates adversarial trajectories simulating diverse threats, ensuring robust training. Experiments on MAS benchmarks show SentinelNet achieves near-perfect detection of malicious agents, close to 100% within two debate rounds, and recovers 95% of system accuracy from compromised baselines. By exhibiting strong generalizability across domains and attack patterns, SentinelNet establishes a novel paradigm for safeguarding collaborative MAS.

---

## 55. UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action

**论文链接:** [http://arxiv.org/abs/2510.17790v1](http://arxiv.org/abs/2510.17790v1)

**作者:** Yuhao Yang, Zhen Yang, Zi-Yi Dou, Anh Nguyen, Keen You, Omar Attia, Andrew Szot, Michael Feng, Ram Ramrakhya, Alexander Toshev, Chao Huang, Yinfei Yang, Zhe Gan

**发布时间:** 2025-10-20

### GPT解析

### 总结

该研究提出了UltraCUA，一种基础模型，通过混合动作机制无缝集成GUI基本操作与高级程序化工具调用，解决了传统计算机使用代理(CUAs)仅依赖基本操作导致的级联故障和性能瓶颈问题。研究包含四个关键组件：自动化工具扩展管道、合成数据引擎、混合动作轨迹收集和两阶段训练流程。实验证明UltraCUA在多个基准测试上显著优于现有代理。

### 背景

当前计算机使用多模态代理(CUAs)完全依赖基本操作(点击、输入、滚动)，这些操作需要准确的视觉定位和冗长的执行链，导致级联故障和性能瓶颈。与其他利用丰富程序化接口(API、MCP服务器、工具)的代理不同，CUAs仍然与这些能力隔离。

### 目的

开发一种基础模型，弥合CUAs与其他代理之间的差距，通过混合动作无缝集成GUI基本操作与高级程序化工具调用，提高CUAs的性能和效率。

### 方法

研究方法包括四个关键组件：(1)自动化管道，从软件文档、开源代码库和代码生成扩展程序化工具；(2)合成数据引擎，生成超过17,000个可验证的任务，涵盖真实世界计算机使用场景；(3)大规模高质量混合动作轨迹收集，同时包含低级GUI动作和高级程序化工具调用；(4)两阶段训练流程，结合监督微调与在线强化学习，实现低级和高级动作之间的战略性交替。

### 主要发现

1) 在OSWorld基准测试中，UltraCUA模型比基础模型平均实现22%的相对改进，并且在步骤上快11%；2) 在WindowsAgentArena上的跨域评估显示，模型达到21.7%的成功率，优于在Windows数据上训练的基线模型；3) 混合动作机制被证明是关键的，它减少了错误传播，同时保持了执行效率。

### 结论

UltraCUA成功解决了传统CUAs的局限性，通过混合动作机制将GUI基本操作与高级程序化工具调用相结合，显著提高了性能和效率。这种创新方法不仅减少了错误传播，还保持了执行效率，为计算机使用代理领域带来了重大进步。

### 翻译

多模态计算机使用代理完全依赖基本操作(点击、输入、滚动)，这些操作需要准确的视觉定位和冗长的执行链，导致级联故障和性能瓶颈。而其他代理则利用丰富的程序化接口(API、MCP服务器、工具)，计算机使用代理(CUAs)仍然与这些能力隔离。我们提出了UltraCUA，一种基础模型，通过混合动作弥合这一差距——无缝集成GUI基本操作与高级程序化工具调用。为实现这一目标，我们的方法包含四个关键组件：(1)自动化管道，从软件文档、开源代码库和代码生成扩展程序化工具；(2)合成数据引擎，生成超过17,000个可验证的任务，涵盖真实世界计算机使用场景；(3)大规模高质量混合动作轨迹收集，同时包含低级GUI动作和高级程序化工具调用；(4)两阶段训练流程，结合监督微调与在线强化学习，实现低级和高级动作之间的战略性交替。我们的7B和32B模型的实验表明，比最先进的代理有显著改进。在OSWorld上，UltraCUA模型比基础模型平均实现22%的相对改进，并且在步骤上快11%。在WindowsAgentArena上的跨域评估显示，我们的模型达到21.7%的成功率，优于在Windows数据上训练的基线模型。混合动作机制被证明是关键的，它减少了错误传播，同时保持了执行效率。


### 论文摘要

Multimodal agents for computer use rely exclusively on primitive actions (click, type, scroll) that require accurate visual grounding and lengthy execution chains, leading to cascading failures and performance bottlenecks. While other agents leverage rich programmatic interfaces (APIs, MCP servers, tools), computer-use agents (CUAs) remain isolated from these capabilities. We present UltraCUA, a foundation model that bridges this gap through hybrid action -- seamlessly integrating GUI primitives with high-level programmatic tool calls. To achieve this, our approach comprises four key components: (1) an automated pipeline that scales programmatic tools from software documentation, open-source repositories, and code generation; (2) a synthetic data engine producing over 17,000 verifiable tasks spanning real-world computer-use scenarios; (3) a large-scale high-quality hybrid action trajectory collection with both low-level GUI actions and high-level programmatic tool calls; and (4) a two-stage training pipeline combining supervised fine-tuning with online reinforcement learning, enabling strategic alternation between low-level and high-level actions. Experiments with our 7B and 32B models demonstrate substantial improvements over state-of-the-art agents. On OSWorld, UltraCUA models achieve an average 22% relative improvement over base models, while being 11% faster in terms of steps. Out-of-domain evaluation on WindowsAgentArena shows our model reaches 21.7% success rate, outperforming baselines trained on Windows data. The hybrid action mechanism proves critical, reducing error propagation while maintaining execution efficiency.

---

## 56. Elastic ViTs from Pretrained Models without Retraining

**论文链接:** [http://arxiv.org/abs/2510.17700v1](http://arxiv.org/abs/2510.17700v1)

**作者:** Walter Simoncini, Michael Dorkenwald, Tijmen Blankevoort, Cees G. M. Snoek, Yuki M. Asano

**发布时间:** 2025-10-20

**备注:** Accepted at NeurIPS 2025

### GPT解析

### 总结

本文提出SnapViT，一种用于Vision Transformers的单次网络近似方法，通过结构化剪枝实现弹性推理，无需重新训练或标签数据，可适应各种计算预算。

### 背景

现有视觉基础模型仅在有限的预定义尺寸中可用，导致在现实约束下无法做出最优部署选择。

### 目的

开发一种新的预训练后结构化剪枝方法，使模型能够在连续的计算预算范围内进行弹性推理。

### 方法

SnapViT结合梯度信息和跨网络结构相关性，通过进化算法近似，无需标记数据，适用于无分类头的模型，且无需重新训练。

### 主要发现

在DINO、SigLIPv2、DeIT和AugReg模型上的实验表明，该方法在各种稀疏度下优于最先进方法，在单个A100 GPU上仅需不到五分钟即可生成可调整到任何计算预算的弹性模型。

### 结论

SnapViT贡献包括：预训练Vision Transformers的有效剪枝策略、Hessian非对角结构的新进化近似方法，以及自监督重要性评分机制，无需重新训练或标签即可保持强性能。

### 翻译

视觉基础模型取得了显著的性能，但仅在有限的预定义尺寸中可用，这迫使在现实约束下做出次优的部署选择。我们介绍了SnapViT：用于剪枝Vision Transformers的单次网络近似，这是一种新的预训练后结构化剪枝方法，能够在连续的计算预算范围内实现弹性推理。我们的方法高效地结合了梯度信息和跨网络结构相关性，通过进化算法近似，不需要标记数据，适用于没有分类头的模型，且无需重新训练。在DINO、SigLIPv2、DeIT和AugReg模型上的实验表明，在各种稀疏度下优于最先进的方法，在单个A100 GPU上需要不到五分钟生成可调整到任何计算预算的弹性模型。我们的主要贡献包括：预训练Vision Transformers的有效剪策策略，Hessian非对角结构的新进化近似方法，以及无需重新训练或标签的自监督重要性评分机制。代码和剪枝模型可在以下网址获取：https://elastic.ashita.nl/


### 论文摘要

Vision foundation models achieve remarkable performance but are only available in a limited set of pre-determined sizes, forcing sub-optimal deployment choices under real-world constraints. We introduce SnapViT: Single-shot network approximation for pruned Vision Transformers, a new post-pretraining structured pruning method that enables elastic inference across a continuum of compute budgets. Our approach efficiently combines gradient information with cross-network structure correlations, approximated via an evolutionary algorithm, does not require labeled data, generalizes to models without a classification head, and is retraining-free. Experiments on DINO, SigLIPv2, DeIT, and AugReg models demonstrate superior performance over state-of-the-art methods across various sparsities, requiring less than five minutes on a single A100 GPU to generate elastic models that can be adjusted to any computational budget. Our key contributions include an efficient pruning strategy for pretrained Vision Transformers, a novel evolutionary approximation of Hessian off-diagonal structures, and a self-supervised importance scoring mechanism that maintains strong performance without requiring retraining or labels. Code and pruned models are available at: https://elastic.ashita.nl/

---

## 57. On-the-Fly OVD Adaptation with FLAME: Few-shot Localization via Active Marginal-Samples Exploration

**论文链接:** [http://arxiv.org/abs/2510.17670v1](http://arxiv.org/abs/2510.17670v1)

**作者:** Yehonathan Refael, Amit Aides, Aviad Barzilai, George Leifman, Genady Beryozkin, Vered Silverman, Bolous Jaber, Tomer Shekel

**发布时间:** 2025-10-20

### GPT解析

### 总结

本文提出了一种级联方法，结合大型预训练开放词汇目标检测模型与轻量级少样本分类器，解决了遥感等专业领域中开放词汇目标检测模型的零样本性能问题，显著提高了对细粒度类别的区分能力，并大幅降低了遥感图像标注成本。

### 背景

开放词汇目标检测(OVD)模型能够通过任意文本查询检测物体，具有显著灵活性，但在遥感等专业领域的零样本性能常受自然语言固有歧义的影响，限制了关键下游应用，例如难以区分'渔船'和'游艇'等细粒度类别。

### 目的

解决OVD模型在专业领域如遥感中的零样本性能问题，提高模型对细粒度类别的区分能力，降低遥感图像标注的高成本，实现即时适应特定用户需求。

### 方法

提出一种级联方法，首先使用零样本模型生成高召回率的目标提案，然后通过仅用少量用户标注示例实时训练的紧凑分类器进行高精度精炼；引入FLAME作为框架核心，这是一种一步主动学习策略，使用密度识别决策边界附近的不确定边际候选样本，并通过聚类确保样本多样性。

### 主要发现

该方法无需昂贵的全模型微调即可实现高精度；能在不到一分钟内实现即时适应，比最先进的替代方案快得多；在遥感基准测试中一致超越最先进的性能。

### 结论

建立了一个实用且资源高效的框架，使基础模型能够适应特定用户需求，为开放词汇目标检测在专业领域的应用提供了新思路。

### 翻译

开放词汇目标检测(OVD)模型能够通过任意文本查询检测物体，提供显著灵活性。然而，它们在遥感等专业领域的零样本性能常因自然语言的固有歧义而受到影响，限制了关键的下游应用。例如，OVD模型可能难以区分'渔船'和'游艇'等细粒度类别，因为它们的嵌入相似且常常不可分割。这可能会通过产生不相关的检测来阻碍特定的用户目标，如监控非法捕鱼。为解决此问题，我们提出了一种级联方法，将大型预训练OVD模型的广泛泛化能力与轻量级少样本分类器相结合。我们的方法首先使用零样本模型生成高召回率的目标提案。然后，这些提案通过仅用少量用户标注示例实时训练的紧凑分类器进行高精度精炼，大幅降低了遥感图像标注的高成本。我们框架的核心是FLAME，一种一步主动学习策略，用于选择信息量最大的样本进行训练。FLAME实时识别决策边界附近的不确定边际候选样本，然后进行聚类以确保样本多样性。这种高效的采样技术无需昂贵的全模型微调即可实现高精度，并能在不到一分钟内实现即时适应，比最先进的替代方案快得多。我们的方法在遥感基准测试中一致超越最先进的性能，为将基础模型适应特定用户需求建立了实用且资源高效的框架。


### 论文摘要

Open-vocabulary object detection (OVD) models offer remarkable flexibility by detecting objects from arbitrary text queries. However, their zero-shot performance in specialized domains like Remote Sensing (RS) is often compromised by the inherent ambiguity of natural language, limiting critical downstream applications. For instance, an OVD model may struggle to distinguish between fine-grained classes such as "fishing boat" and "yacht" since their embeddings are similar and often inseparable. This can hamper specific user goals, such as monitoring illegal fishing, by producing irrelevant detections. To address this, we propose a cascaded approach that couples the broad generalization of a large pre-trained OVD model with a lightweight few-shot classifier. Our method first employs the zero-shot model to generate high-recall object proposals. These proposals are then refined for high precision by a compact classifier trained in real-time on only a handful of user-annotated examples - drastically reducing the high costs of RS imagery annotation.The core of our framework is FLAME, a one-step active learning strategy that selects the most informative samples for training. FLAME identifies, on the fly, uncertain marginal candidates near the decision boundary using density estimation, followed by clustering to ensure sample diversity. This efficient sampling technique achieves high accuracy without costly full-model fine-tuning and enables instant adaptation, within less then a minute, which is significantly faster than state-of-the-art alternatives.Our method consistently surpasses state-of-the-art performance on RS benchmarks, establishing a practical and resource-efficient framework for adapting foundation models to specific user needs.

---

## 58. DELULU: Discriminative Embedding Learning Using Latent Units for Speaker-Aware Self-Supervised Speech Foundational Model

**论文链接:** [http://arxiv.org/abs/2510.17662v1](http://arxiv.org/abs/2510.17662v1)

**作者:** Massa Baali, Rita Singh, Bhiksha Raj

**发布时间:** 2025-10-20

### GPT解析

### 总结

DELULU是一个说话人感知的自监督语音基础模型，通过在外部监督集成到伪标签生成过程中，显著提升了在说话人相关任务上的性能。

### 背景

自监督语音模型在内容驱动任务上表现优异，但在捕捉说话人区分性特征方面有限，这对验证、说话人分割和档案应用至关重要。

### 目的

开发一个能够捕捉说话人区分性特征的自监督语音模型，解决现有模型在说话人相关任务上的局限性。

### 方法

DELULU利用ReDimNet的帧级嵌入指导k-means聚类，引入说话人区分性归纳偏差；使用掩码预测和去噪的双重目标进行训练，增强鲁棒性和泛化能力。

### 主要发现

DELULU在说话人验证任务上实现高达62%的相对EER改进，在零样本档案任务（性别、年龄、口音、说话人计数）上取得一致提升。

### 结论

DELULU是说话人感知语音处理的强大通用编码器，无需任务特定微调即可实现卓越性能。

### 翻译

自监督语音模型在内容驱动任务上取得了显著成功，但在捕捉对验证、说话人分割和档案应用至关重要的说话人区分性特征方面仍然有限。我们引入了DELULU，一个说话人感知的自监督基础模型，通过在外部监督集成到伪标签生成过程中来解决这一局限性。DELULU利用来自ReDimNet（最先进的说话人验证模型）的帧级嵌入来指导预训练期间的k-means聚类步骤，引入了强大的说话人区分性归纳偏差，使表示学习与说话人身份保持一致。该模型使用结合掩码预测和去噪的双重目标进行训练，进一步增强了鲁棒性和泛化能力。DELULU在一系列以说话人为中心的任务上显著优于先前的自监督学习模型，在说话人验证的等错误率上实现了高达62%的相对改进，并在零样本档案任务（如性别、年龄、口音和说话人计数）上取得了一致的提升。我们的研究结果表明，DELULU是说话人感知语音处理的强大通用编码器，即使在没有任务特定微调的情况下也能实现卓越的性能。


### 论文摘要

Self-supervised speech models have achieved remarkable success on content-driven tasks, yet they remain limited in capturing speaker-discriminative features critical for verification, diarization, and profiling applications. We introduce DELULU, a speaker-aware self-supervised foundational model that addresses this limitation by integrating external supervision into the pseudo-label generation process. DELULU leverages frame-level embeddings from ReDimNet, a state-of-the-art speaker verification model, to guide the k-means clustering step during pre-training, introducing a strong speaker-discriminative inductive bias that aligns representation learning with speaker identity. The model is trained using a dual objective that combines masked prediction and denoising, further enhancing robustness and generalization. DELULU significantly outperforms prior self-supervised learning (SSL) models across a range of speaker-centric tasks, achieving up to 62% relative improvement in equal error rate (EER) for speaker verification and consistent gains on zero-shot profiling tasks such as gender, age, accent, and speaker counting. Our findings demonstrate that DELULU is a strong universal encoder for speaker-aware speech processing, enabling superior performance even without task-specific fine-tuning.

---

## 59. Deeper with Riemannian Geometry: Overcoming Oversmoothing and Oversquashing for Graph Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.17457v1](http://arxiv.org/abs/2510.17457v1)

**作者:** Li Sun, Zhenhao Huang, Ming Zhang, Philip S. Yu

**发布时间:** 2025-10-20

**备注:** Accept by NeurIPS 25

### GPT解析

### 总结

本文提出了一种名为GBN的局部方法，通过自适应调整基于局部结构的消息传递来解决MPNNs的过平滑和过挤压问题。

### 背景

MPNNs是图基础模型的构建块，但存在过平滑和过挤压问题。现有解决方案主要采用全局方法，导致表达能力不足。

### 目的

开发一种局部方法，能够自适应地调整消息传递，同时解决过平滑和过挤压问题，提高MPNNs的表达能力。

### 方法

作者将局部黎曼几何与MPNNs连接，建立了新的非齐次边界条件，并设计了具有局部瓶颈调整的GBN网络，基于Robin条件构建。

### 主要发现

谱间隙的增加会导致梯度消失，削弱消息传递效果；GBN在同类同质和异类异质图上表现出强大的表达能力，且在网络深度超过256层时仍保持性能。

### 结论

局部方法比全局方法更有效地解决MPNNs的过平滑和过挤压问题，GBN网络提供了理论保证和优异的实验性能。

### 翻译

消息传递神经网络是图基础模型的构建块，但 fundamentally suffer from 过平滑和过挤压问题。最近有很多研究试图解决这两个问题。现有工作主要采用全局方法，在某些区域可能有益，但在其他区域可能有害，最终导致表达能力不足。本文通过全局度量谱间隙重新审视过挤压问题，并证明谱间隙的增加会导致相对于输入特征的梯度消失，从而削弱消息传递的有效性。基于这些理论见解，我们提出了一种局部方法，根据局部结构自适应调整消息传递。为此，我们将局部黎曼几何与MPNNs连接，并建立了新的非齐次边界条件来解决过挤压和过平滑问题。基于Robin条件，我们设计了具有局部瓶颈调整的GBN网络，并提供了理论保证。在同类同质和异类异质图上的广泛实验表明GBN的表达能力。此外，即使网络深度超过256层，GBN也不会表现出性能下降。


### 论文摘要

Message Passing Neural Networks (MPNNs) is the building block of graph foundation models, but fundamentally suffer from oversmoothing and oversquashing. There has recently been a surge of interest in fixing both issues. Existing efforts primarily adopt global approaches, which may be beneficial in some regions but detrimental in others, ultimately leading to the suboptimal expressiveness. In this paper, we begin by revisiting oversquashing through a global measure -- spectral gap $\lambda$ -- and prove that the increase of $\lambda$ leads to gradient vanishing with respect to the input features, thereby undermining the effectiveness of message passing. Motivated by such theoretical insights, we propose a \textbf{local} approach that adaptively adjusts message passing based on local structures. To achieve this, we connect local Riemannian geometry with MPNNs, and establish a novel nonhomogeneous boundary condition to address both oversquashing and oversmoothing. Building on the Robin condition, we design a GBN network with local bottleneck adjustment, coupled with theoretical guarantees. Extensive experiments on homophilic and heterophilic graphs show the expressiveness of GBN. Furthermore, GBN does not exhibit performance degradation even when the network depth exceeds $256$ layers.

---

## 60. From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors

**论文链接:** [http://arxiv.org/abs/2510.17439v1](http://arxiv.org/abs/2510.17439v1)

**作者:** Zhengshen Zhang, Hao Li, Yalun Dai, Zhengbang Zhu, Lei Zhou, Chenchen Liu, Dong Wang, Francis E. H. Tay, Sijin Chen, Ziwei Liu, Yuxiao Liu, Xinghang Li, Pan Zhou

**发布时间:** 2025-10-20

**备注:** Project page: https://falcon-vla.github.io/

### GPT解析

### 总结

本文提出了一种名为FALCON的新型视觉-语言-行动模型，通过将3D空间令牌注入行动头，解决了现有VLA模型的空间推理差距问题，实现了在模拟和现实场景中的最先进性能。

### 背景

现有的视觉-语言-行动模型在3D真实世界中运行，但通常基于2D编码器构建，导致空间推理差距，限制了泛化能力和适应性。近期VLA的3D集成技术要么需要专用传感器且跨模态迁移性差，要么注入缺乏几何信息的弱提示，导致视觉-语言对齐质量下降。

### 目的

解决现有VLA模型在空间表示、模态迁移性和对齐方面的局限性，提升模型在复杂环境中的表现和鲁棒性。

### 方法

引入FALCON（From Spatial to Action）范式，将丰富的3D空间令牌注入到行动头中；利用空间基础模型仅从RGB提供强大的几何先验；包含一个可选择性融合深度或姿态的具身空间模型，无需重新训练或架构改变；为保留语言推理能力，空间令牌被空间增强行动头处理，而非连接到视觉-语言主干。

### 主要发现

FALCON在三个模拟基准测试和十一个现实世界任务的综合评估中实现了最先进性能，一致超越竞争基线，并且在杂乱环境、空间提示条件和物体高度变化等情况下保持鲁棒性。

### 结论

FALCON通过创新的空间令牌注入机制和灵活的多模态融合能力，有效解决了VLA模型在空间表示、模态迁移性和对齐方面的局限性，为3D环境中的智能行动提供了新的解决方案。

### 翻译

现有的视觉-语言-行动模型在3D真实世界中运行，但通常基于2D编码器构建，留下了限制泛化和适应性的空间推理差距。近期VLA的3D集成技术要么需要专用传感器且跨模态迁移性差，要么注入缺乏几何信息的弱提示，导致视觉-语言对齐质量下降。在这项工作中，我们引入FALCON（From Spatial to Action），一种将丰富的3D空间令牌注入行动头的新颖范式。FALCON利用空间基础模型仅从RGB提供强大的几何先验，并包含一个可选择性融合深度或姿态的具身空间模型，当可用时提供更高保真度，无需重新训练或架构改变。为保留语言推理能力，空间令牌被空间增强行动头消耗，而非连接到视觉-语言主干。这些设计使FALCON能够解决空间表示、模态迁移性和对齐方面的局限性。在三个模拟基准测试和十一个现实世界任务的综合评估中，我们提出的FALCON实现了最先进性能，一致超越竞争基线，并且在杂乱环境、空间提示条件和物体高度变化等情况下保持鲁棒性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决现有视觉-语言-动作（VLA）模型的空间推理差距问题。这些模型虽然能在3D真实世界运行，但通常基于2D编码器构建，导致缺乏可靠的3D空间理解，限制了机器人在新场景、背景变化或物体变化时的泛化能力和适应性。这个问题非常重要，因为机器人需要与3D物理世界交互，而缺乏明确的3D意识使它们难以处理需要几何推理、深度感知或空间关系理解的任务，这已成为开发可靠通用机器人政策的主要瓶颈。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先深入分析了现有VLA模型的局限性，注意到它们基于2D编码器但需要在3D世界运行，存在空间推理差距。作者借鉴了空间基础模型（如VGGT、DUSt3R）的思路，这些模型能将场景编码为令牌序列进行3D重建；同时受到大脑分工的启发，将VLM比作处理高级推理的大脑，动作头比作管理精细运动的小脑。作者还参考了现有VLA架构（如RT-2、OpenVLA），但改进了空间信息集成方式，并利用了现有的深度估计和相机姿态编码技术。最终设计了具身空间模型(ESM)和空间增强动作头，实现了空间与语义的有效融合。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过将丰富的3D空间令牌注入到VLA模型的动作头中来增强空间理解能力，同时保持语言推理能力。整体流程包括：1)双路径处理 - VLM路径处理视觉和语言输入提取语义表示，ESM路径处理图像和可选几何输入提取空间令牌；2)ESM通过令牌化、空间编码和可选的深度/姿态注入来生成空间令牌；3)通过最大池化和MLP适配器将空间特征投影到VLM特征空间；4)使用元素级加法融合空间特征与语义动作令牌；5)融合后的特征输入动作预测器（MLP或LSTM）生成机器人动作序列；6)采用两阶段后训练方法确保训练稳定性和特征对齐。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)空间令牌注入新范式 - 将3D空间令牌注入动作头而非连接文本令牌；2)具身空间模型(ESM) - 可选择性整合深度和姿态等3D模态；3)空间增强动作头 - 直接将空间令牌整合到动作决策中；4)随机条件策略 - 确保模型在不同输入条件下都能有效工作。相比之前工作，FALCON不依赖特定3D传感器（区别于PointVLA、GeoVLA），不会破坏预训练的视觉-语言对齐（区别于3D-VLA、SpatialVLA），提供了显式3D理解（区别于传统2D VLA），并采用高效的两阶段训练方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'FALCON通过将空间基础模型提供的丰富3D空间令牌注入到专门设计的空间增强动作头中，解决了现有视觉-语言-动作模型在3D空间理解上的局限，实现了强大的模态转移能力和在复杂空间任务中的最先进性能。'}


### 论文摘要

Existing vision-language-action (VLA) models act in 3D real-world but are typically built on 2D encoders, leaving a spatial reasoning gap that limits generalization and adaptability. Recent 3D integration techniques for VLAs either require specialized sensors and transfer poorly across modalities, or inject weak cues that lack geometry and degrade vision-language alignment. In this work, we introduce FALCON (From Spatial to Action), a novel paradigm that injects rich 3D spatial tokens into the action head. FALCON leverages spatial foundation models to deliver strong geometric priors from RGB alone, and includes an Embodied Spatial Model that can optionally fuse depth, or pose for higher fidelity when available, without retraining or architectural changes. To preserve language reasoning, spatial tokens are consumed by a Spatial-Enhanced Action Head rather than being concatenated into the vision-language backbone. These designs enable FALCON to address limitations in spatial representation, modality transferability, and alignment. In comprehensive evaluations across three simulation benchmarks and eleven real-world tasks, our proposed FALCON achieves state-of-the-art performance, consistently surpasses competitive baselines, and remains robust under clutter, spatial-prompt conditioning, and variations in object scale and height.

---

## 61. Diffusion Models as Dataset Distillation Priors

**论文链接:** [http://arxiv.org/abs/2510.17421v1](http://arxiv.org/abs/2510.17421v1)

**作者:** Duo Su, Huyu Wu, Huanran Chen, Yiming Shi, Yuzhu Wang, Xi Ye, Jun Zhu

**发布时间:** 2025-10-20

### GPT解析

### 总结

本研究提出了Diffusion As Priors (DAP)方法，通过利用扩散模型中的代表性先验，解决了数据集蒸馏中同时实现多样性、泛化能力和代表性的挑战。DAP在特征空间中使用Mercer核量化合成数据与真实数据的相似性，并将此先验作为指导引导反向扩散过程，无需重新训练即可提高蒸馏数据集质量。实验证明，DAP在ImageNet-1K等大型数据集上优于现有方法，实现了更好的跨架构泛化能力。

### 背景

数据集蒸馏旨在从大型数据集中合成紧凑而信息丰富的数据集。该领域的一个重大挑战是在单个蒸馏数据集中同时实现多样性、泛化能力和代表性。虽然最近的生成式数据集蒸馏方法采用了强大的扩散模型作为基础模型，但这些方法忽略了扩散模型中固有的代表性先验，因此需要集成外部约束来提高数据质量。

### 目的

解决数据集蒸馏中同时实现多样性、泛化能力和代表性的挑战，通过利用扩散模型中固有的代表性先验，提出一种无需重新训练即可提高蒸馏数据集质量的方法。

### 方法

作者提出了Diffusion As Priors (DAP)方法，该方法通过以下步骤实现：在特征空间中使用Mercer核量化合成数据与真实数据之间的相似性，将代表性形式化；将此先验作为指导来引导反向扩散过程；增强蒸馏样本的代表性，无需任何重新训练。

### 主要发现

在ImageNet-1K及其子集等大规模数据集上的大量实验表明，DAP在生成高保真度数据集方面优于最先进的方法；DAP实现了更好的跨架构泛化能力；该研究在扩散先验与数据集蒸馏目标之间建立了理论联系。

### 结论

Diffusion As Priors (DAP)方法为提高数据集蒸馏质量提供了一个实用的、无需训练的框架。该研究不仅在扩散先验与数据集蒸馏目标之间建立了理论联系，还为解决数据集蒸馏中的代表性挑战提供了有效的方法，同时实现了多样性和泛化能力的平衡。

### 翻译

数据集蒸馏旨在从大型数据集中合成紧凑而信息丰富的数据集。该领域的一个重大挑战是在单个蒸馏数据集中同时实现多样性、泛化能力和代表性。虽然最近的生成式数据集蒸馏方法采用了强大的扩散模型作为基础模型，但这些方法忽略了扩散模型中固有的代表性先验，因此需要集成外部约束来提高数据质量。为此，我们提出了Diffusion As Priors (DAP)，该方法通过在特征空间中使用Mercer核量化合成数据与真实数据之间的相似性，将代表性形式化。然后，我们将此先验作为指导来引导反向扩散过程，无需任何重新训练即可增强蒸馏样本的代表性。在ImageNet-1K及其子集等大规模数据集上的大量实验表明，DAP在生成高保真度数据集方面优于最先进的方法，同时实现了更好的跨架构泛化能力。我们的研究不仅在扩散先验与数据集蒸馏目标之间建立了理论联系，还为提高蒸馏数据集质量提供了一个实用的、无需训练的框架。


### 论文摘要

Dataset distillation aims to synthesize compact yet informative datasets from large ones. A significant challenge in this field is achieving a trifecta of diversity, generalization, and representativeness in a single distilled dataset. Although recent generative dataset distillation methods adopt powerful diffusion models as their foundation models, the inherent representativeness prior in diffusion models is overlooked. Consequently, these approaches often necessitate the integration of external constraints to enhance data quality. To address this, we propose Diffusion As Priors (DAP), which formalizes representativeness by quantifying the similarity between synthetic and real data in feature space using a Mercer kernel. We then introduce this prior as guidance to steer the reverse diffusion process, enhancing the representativeness of distilled samples without any retraining. Extensive experiments on large-scale datasets, such as ImageNet-1K and its subsets, demonstrate that DAP outperforms state-of-the-art methods in generating high-fidelity datasets while achieving superior cross-architecture generalization. Our work not only establishes a theoretical connection between diffusion priors and the objectives of dataset distillation but also provides a practical, training-free framework for improving the quality of the distilled dataset.

---

## 62. Monitoring Horses in Stalls: From Object to Event Detection

**论文链接:** [http://arxiv.org/abs/2510.17409v1](http://arxiv.org/abs/2510.17409v1)

**作者:** Dmitrii Galimzianov, Viacheslav Vyshegorodtsev, Ivan Nezhivykh

**发布时间:** 2025-10-20

**备注:** 12 pages, 4 figures, 4 tables

### GPT解析

### 总结

该研究开发了一种基于视觉的监控系统，可自动化检测和跟踪马厩中的马匹和人，用于早期发现健康和福利问题，减少人工监控的劳动强度。

### 背景

监控拴马的行为对于早期发现健康和福利问题至关重要，但目前的监控方法仍然劳动密集且耗时。

### 目的

开发一个基于视觉的原型监控系统，自动化检测和跟踪马厩内的马匹和人，实现实时行为监控。

### 方法

使用目标检测和多目标跟踪技术，系统利用YOLOv11和BoT-SORT进行检测和跟踪，基于物体轨迹和空间关系推断事件状态，构建了使用CLIP和GroundingDINO标注的自定义数据集，系统能区分五种事件类型并考虑相机盲点。

### 主要发现

定性评估表明系统在马匹相关事件检测方面表现可靠，但由于数据稀缺，在检测人方面存在局限性。

### 结论

这项工作为马匹设施的实时行为监控提供了基础，对动物福利和马厩管理有重要意义。

### 翻译

监控拴马的行为对于早期发现健康和福利问题至关重要，但仍然劳动密集且耗时。在本研究中，我们提出了一个基于视觉的原型监控系统，使用目标检测和多目标跟踪技术自动化检测和跟踪马厩内的马匹和人。系统利用YOLOv11和BoT-SORT进行检测和跟踪，同时基于马厩内物体的轨迹和空间关系推断事件状态。为支持开发，我们构建了一个使用基础模型CLIP和GroundingDINO协助标注的自定义数据集。系统能区分五种事件类型并考虑相机的盲点。定性评估表明系统在马匹相关事件检测方面表现可靠，同时指出由于数据稀缺，在检测人方面存在局限性。这项工作为马匹设施的实时行为监控提供了基础，对动物福利和马厩管理有重要意义。


### 论文摘要

Monitoring the behavior of stalled horses is essential for early detection of health and welfare issues but remains labor-intensive and time-consuming. In this study, we present a prototype vision-based monitoring system that automates the detection and tracking of horses and people inside stables using object detection and multi-object tracking techniques. The system leverages YOLOv11 and BoT-SORT for detection and tracking, while event states are inferred based on object trajectories and spatial relations within the stall. To support development, we constructed a custom dataset annotated with assistance from foundation models CLIP and GroundingDINO. The system distinguishes between five event types and accounts for the camera's blind spots. Qualitative evaluation demonstrated reliable performance for horse-related events, while highlighting limitations in detecting people due to data scarcity. This work provides a foundation for real-time behavioral monitoring in equine facilities, with implications for animal welfare and stable management.

---

## 63. Leveraging Group Relative Policy Optimization to Advance Large Language Models in Traditional Chinese Medicine

**论文链接:** [http://arxiv.org/abs/2510.17402v1](http://arxiv.org/abs/2510.17402v1)

**作者:** Jiacheng Xie, Shuai Zeng, Yang Yu, Xiaoting Tang, Guanghui An, Dong Xu

**发布时间:** 2025-10-20

### GPT解析

### 总结

研究团队开发了Ladder-base，首个使用组相对策略优化(GRPO)训练的中医领域大语言模型，在多个推理指标上表现优于通用大语言模型和特定中医模型。

### 背景

中医拥有丰富且结构独特的知识体系，这对常规大语言模型的应用提出了挑战。虽然之前的中医特定LLM通过监督微调取得进展，但它们在对齐、数据质量和评估一致性方面存在局限性。

### 目的

开发一个针对中医领域的大语言模型，解决现有模型在一致性、数据质量和评估一致性方面的局限性，提高模型在中医领域的推理能力和事实一致性。

### 方法

使用组相对策略优化(GRPO)强化学习方法训练Ladder-base模型，该方法通过基于组内比较优化响应选择来提高推理和事实一致性。模型基于Qwen2.5-7B-Instruct构建，在中医阶梯基准的文本子集上训练，使用80%数据训练，剩余20%平均分为验证和测试集。

### 主要发现

通过标准化评估，Ladder-base在多个推理指标上表现优于GPT-4、Gemini 2.5、Claude 3、Qwen3等通用大语言模型，以及BenTsao、HuatuoGPT2、Zhongjing等特定中医模型。

### 结论

GRPO为将大语言模型与中医领域专家级推理对齐提供了有效且高效的策略，支持开发可信且临床基础的中医人工智能系统。

### 翻译

传统中医呈现了一个丰富且结构独特的知识体系，这对常规大语言模型的应用提出了挑战。尽管之前的中医特定LLM通过监督微调已经显示出进展，但它们常常在对齐、数据质量和评估一致性方面面临局限性。在本研究中，我们引入了Ladder-base，这是第一个使用组相对策略优化训练的中医领域LLM，这是一种通过基于组内比较优化响应选择来提高推理和事实一致性的强化学习方法。Ladder-base基于Qwen2.5-7B-Instruct基础模型构建，并仅在中医阶梯基准的文本子集上进行训练，使用80%的数据进行训练，剩余的20%平均分为验证集和测试集。通过标准化评估，Ladder-base在多个推理指标上表现出优于最先进的通用LLM和特定中医模型的性能。这些发现表明，GRPO为将LLM与中医领域专家级推理对齐提供了一种有效且高效的策略，支持开发可信且临床基础的中医人工智能系统。


### 论文摘要

Traditional Chinese Medicine (TCM) presents a rich and structurally unique knowledge system that challenges conventional applications of large language models (LLMs). Although previous TCM-specific LLMs have shown progress through supervised fine-tuning, they often face limitations in alignment, data quality, and evaluation consistency. In this study, we introduce Ladder-base, the first TCM-focused LLM trained with Group Relative Policy Optimization (GRPO), a reinforcement learning method that improves reasoning and factual consistency by optimizing response selection based on intra-group comparisons. Ladder-base is built upon the Qwen2.5-7B-Instruct foundation model and trained exclusively on the textual subset of the TCM-Ladder benchmark, using 80 percent of the data for training and the remaining 20 percent split evenly between validation and test sets. Through standardized evaluation, Ladder-base demonstrates superior performance across multiple reasoning metrics when compared to both state-of-the-art general-purpose LLMs such as GPT-4, Gemini 2.5, Claude 3, and Qwen3 and domain-specific TCM models including BenTsao, HuatuoGPT2, and Zhongjing. These findings suggest that GRPO provides an effective and efficient strategy for aligning LLMs with expert-level reasoning in traditional medical domains and supports the development of trustworthy and clinically grounded TCM artificial intelligence systems.

---

## 64. Exploring The Missing Semantics In Event Modality

**论文链接:** [http://arxiv.org/abs/2510.17347v1](http://arxiv.org/abs/2510.17347v1)

**作者:** Jingqian Wu, Shengpeng Xu, Yunbo Jia, Edmund Y. Lam

**发布时间:** 2025-10-20

### GPT解析

### 总结

本文提出了Semantic-E2VID框架，通过探索事件模态中缺失的视觉语义知识并利用其增强事件到视频重建，解决了事件相机无法捕捉静态物体和背景导致的语义信息缺失问题。

### 背景

事件相机具有低延迟、高动态范围和高效运动捕捉等优势，但事件到视频重建任务面临重建和恢复语义信息的挑战。事件相机只捕捉强度变化，忽略静态物体和背景，导致捕获的事件模态中缺乏语义信息。

### 目的

提出Semantic-E2VID框架，探索事件模态中缺失的视觉语义知识并利用其增强事件到视频重建。

### 方法

引入跨模态特征对齐(CFA)模块将SAM模型的鲁棒视觉语义传输到事件编码器；提出语义感知特征融合(SFF)块整合学习到的语义信息；提出语义感知E2V监督利用SAM生成的类别标签帮助重建语义细节。

### 主要发现

Semantic-E2VID显著提高了帧质量，在多个基准测试中优于最先进的E2V方法。

### 结论

Semantic-E2VID有效解决了事件模态中语义信息缺失的问题，通过跨模态特征对齐和语义感知特征融合提升了事件到视频重建的质量。

### 翻译

事件相机提供低延迟、高动态范围和高效运动捕捉等独特优势。然而，作为基础事件视觉任务的事件到视频重建(E2V)仍然具有挑战性，特别是在重建和恢复语义信息方面。这主要源于事件相机的本质，因为它只捕捉强度变化，忽略静态物体和背景，导致捕获的事件模态中缺乏语义信息。此外，语义信息在视频和帧重建中起着关键作用，但现有的E2V方法常常忽略了这一点。为了弥合这一差距，我们提出了Semantic-E2VID，这是一个E2V框架，探索事件模态中缺失的视觉语义知识，并利用它来增强事件到视频的重建。具体来说，Semantic-E2VID引入了跨模态特征对齐(CFA)模块，将基于帧的视觉基础模型(Segment Anything Model, SAM)的鲁棒视觉语义传输到事件编码器，同时对齐来自不同模态的高级特征。为了更好地利用学习到的语义特征，我们进一步提出了一个语义感知特征融合(SFF)块，将学习到的帧模态语义整合到事件表示中，形成具有丰富语义的事件表示，可被事件解码器解码。此外，为了促进语义信息的重建，我们提出了一种新颖的语义感知E2V监督，它通过利用SAM生成的类别标签帮助模型重建语义细节。大量实验证明，Semantic-E2VID显著提高了帧质量，在多个基准测试中优于最先进的E2V方法。示例代码包含在补充材料中。


### 论文摘要

Event cameras offer distinct advantages such as low latency, high dynamic range, and efficient motion capture. However, event-to-video reconstruction (E2V), a fundamental event-based vision task, remains challenging, particularly for reconstructing and recovering semantic information. This is primarily due to the nature of the event camera, as it only captures intensity changes, ignoring static objects and backgrounds, resulting in a lack of semantic information in captured event modality. Further, semantic information plays a crucial role in video and frame reconstruction, yet is often overlooked by existing E2V approaches. To bridge this gap, we propose Semantic-E2VID, an E2V framework that explores the missing visual semantic knowledge in event modality and leverages it to enhance event-to-video reconstruction. Specifically, Semantic-E2VID introduces a cross-modal feature alignment (CFA) module to transfer the robust visual semantics from a frame-based vision foundation model, the Segment Anything Model (SAM), to the event encoder, while aligning the high-level features from distinct modalities. To better utilize the learned semantic feature, we further propose a semantic-aware feature fusion (SFF) block to integrate learned semantics in frame modality to form event representations with rich semantics that can be decoded by the event decoder. Further, to facilitate the reconstruction of semantic information, we propose a novel Semantic Perceptual E2V Supervision that helps the model to reconstruct semantic details by leveraging SAM-generated categorical labels. Extensive experiments demonstrate that Semantic-E2VID significantly enhances frame quality, outperforming state-of-the-art E2V methods across multiple benchmarks. The sample code is included in the supplementary material.

---

## 65. 论文ID: 2510.17172v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.17172v1.json'

---

## 66. Trading with the Devil: Risk and Return in Foundation Model Strategies

**论文链接:** [http://arxiv.org/abs/2510.17165v1](http://arxiv.org/abs/2510.17165v1)

**作者:** Jinrui Zhang

**发布时间:** 2025-10-20

### GPT解析

### 总结

本文提出了一种扩展的资本资产定价模型(CAPM)，用于分离基础模型引入的系统风险和特定微调带来的特定风险，帮助金融从业者更好地理解和评估基于基础模型的交易策略风险状况。

### 背景

基础模型已在自然语言处理等领域产生变革性影响，现开始应用于金融时间序列任务。这些预训练架构虽能提供多样化预测信号，但如何影响交易策略风险状况尚不明确，导致实践者不愿投入大量资本。

### 目的

扩展资本资产定价模型，分离基础模型引入的系统风险（可能产生alpha）与特定微调带来的特定风险（通常不积累系统性溢价），并开发实用方法估计这些风险。

### 方法

将风险分解与不确定性解耦概念对齐，将系统性风险视为本体不确定性，特定风险视为偶然不确定性。在偶然崩溃假设下，使用蒙特卡洛dropout等方法直接测量本体风险，将交易策略映射到更透明的风险-回报平面。

### 主要发现

分离不同风险因素可更深入了解基于基础模型策略的性能限制、模型随时间的退化情况，以及有针对性的改进途径。

### 结论

研究结果突出了在竞争性金融市场部署大型预训练模型的希望和陷阱，为金融从业者提供了更全面的风险评估框架。

### 翻译

基础模型-已在自然语言处理等领域产生变革性影响-现正开始出现在金融时间序列任务中。虽然这些预训练架构承诺提供多样化的预测信号，但人们对其如何塑造构建于其上的交易策略的风险状况知之甚少，导致实践者不愿投入大量资本。在本文中，我们提出对资本资产定价模型(CAPM)的扩展，该模型分离了共享基础模型引入的系统风险-如果底层模型真正具有预测能力，则可能产生alpha-以及归因于自定义微调的特定风险，后者通常不积累系统性溢价。为了能够实际估计这些独立风险，我们将这种分解与不确定性解耦的概念对齐，将系统性风险视为本体不确定性（源于预训练模型），将特定风险视为偶然不确定性（在自定义适应过程中引入）。在偶然崩溃假设下，我们说明了如何使用蒙特卡洛dropout-以及其他不确定性量化工具包中的方法-直接测量本体风险，从而将交易策略映射到更透明的风险-回报平面。我们的实验表明，分离这些不同的风险因素可以更深入地了解基于基础模型的策略的性能限制、其随时间的模型退化情况，以及有针对性的改进途径。总的来说，我们的结果突出了在竞争性金融市场部署大型预训练模型的希望和陷阱。


### 论文摘要

Foundation models - already transformative in domains such as natural language processing - are now starting to emerge for time-series tasks in finance. While these pretrained architectures promise versatile predictive signals, little is known about how they shape the risk profiles of the trading strategies built atop them, leaving practitioners reluctant to commit serious capital. In this paper, we propose an extension to the Capital Asset Pricing Model (CAPM) that disentangles the systematic risk introduced by a shared foundation model - potentially capable of generating alpha if the underlying model is genuinely predictive - from the idiosyncratic risk attributable to custom fine-tuning, which typically accrues no systematic premium. To enable a practical estimation of these separate risks, we align this decomposition with the concepts of uncertainty disentanglement, casting systematic risk as epistemic uncertainty (rooted in the pretrained model) and idiosyncratic risk as aleatory uncertainty (introduced during custom adaptations). Under the Aleatory Collapse Assumption, we illustrate how Monte Carlo dropout - among other methods in the uncertainty-quantization toolkit - can directly measure the epistemic risk, thereby mapping trading strategies to a more transparent risk-return plane. Our experiments show that isolating these distinct risk factors yields deeper insights into the performance limits of foundation-model-based strategies, their model degradation over time, and potential avenues for targeted refinements. Taken together, our results highlight both the promise and the pitfalls of deploying large pretrained models in competitive financial markets.

---

## 67. TREAT: A Code LLMs Trustworthiness / Reliability Evaluation and Testing Framework

**论文链接:** [http://arxiv.org/abs/2510.17163v1](http://arxiv.org/abs/2510.17163v1)

**作者:** Shuzheng Gao, Eric John Li, Man Ho Lam, Jingyu Xiao, Yuxuan Wan, Chaozheng Wang, Ng Man Tik, Michael R. Lyu

**发布时间:** 2025-10-20

### GPT解析

### 总结

大型基础模型正在改变软件工程领域，但缺乏全面的可信度评估方法。研究团队提出了TREAT评估框架，通过多任务、多语言多模态、鲁棒性和严格评估方法四个改进点，对26个先进模型进行了评估，发现了模型在编程任务上的性能差异和多模态模型在UI代码生成方面的局限性。

### 背景

大型基础模型正在从根本上改变软件工程领域，在代码生成、调试和测试等任务上表现出色。然而，如何全面评估这些模型在真实软件工程场景中的可信度仍存在显著差距。现有基准测试存在任务范围有限、未包含模型鲁棒性和可靠性等关键评估方面的问题。

### 目的

填补现有评估方法的不足，提供一个全面的模型性能评估框架，以评估大型基础模型在软件工程任务中的可信度和可靠性。

### 方法

提出名为TREAT（Code LLMs Trustworthiness/Reliability Evaluation And Testing）的评估框架，包含四个主要改进：1) 多任务全面评估，涵盖多样化的软件工程活动；2) 多语言和多模态评估，包含多模态编码任务；3) 鲁棒性评估，评估模型在语义保持代码转换下的可靠性；4) 严格的评估方法，通过多样化的评估提示和自适应解决方案提取提高评估结果的可信度。基于此框架评估了26个最先进的模型。

### 主要发现

1) 当前模型在编程任务上表现出显著的性能差异；2) 多模态语言模型在UI代码生成和编辑方面表现出特定的性能局限性。

### 结论

TREAT框架为评估大型基础模型在软件工程任务中的可信度和可靠性提供了更全面的方法，有助于识别模型的优势和局限性，指导未来的模型改进方向。

### 翻译

大型基础模型正在从根本上改变软件工程领域，在代码生成、调试和测试等多样化任务上表现出色。尽管进展迅速，但在如何全面评估这些模型在真实软件工程场景中的可信度方面仍存在显著差距。现有基准测试存在任务范围有限，未能纳入模型的鲁棒性和可靠性等关键评估方面的问题。为填补这一差距，我们提出了一个名为TREAT（Code LLMs Trustworthiness/Reliability Evaluation And Testing）的评估框架，该框架提供对模型在代码智能任务中性能的全面评估。我们的评估框架通过四个主要改进解决了现有方法的关键局限性：(1) 多任务全面评估，涵盖多样化的软件工程活动，而非有限的编码任务；(2) 多语言和多模态评估，超越传统的单语言、纯文本基准，包含多模态编码任务；(3) 鲁棒性评估，评估模型在语义保持代码转换下的可靠性；(4) 严格的评估方法，通过多样化的评估提示和自适应解决方案提取提高评估结果的可信度。基于此评估框架，我们评估了26个最先进的模型，发现了它们的优势和局限性，得出了几个关键见解：(1) 当前模型在编程任务上表现出显著的性能差异；(2) 多模态语言模型在UI代码生成和编辑方面表现出特定的性能局限性。


### 论文摘要

Large foundation models are fundamentally transforming the software engineering landscape, demonstrating exceptional capabilities across diverse tasks such as code generation, debugging, and testing. Despite this rapid progress, a significant gap remains in how to comprehensively evaluate these models' trustworthiness in real-world software engineering scenarios. Existing benchmarks suffer from limited task scope and fail to incorporate critical evaluation aspects such as the robustness and reliability of models. To bridge this gap, we present an evaluation framework called TREAT (Code LLMs Trustworthiness / Reliability Evaluation And Testing) that provides a holistic assessment of model performance in code intelligence tasks. Our evaluation framework addresses key limitations in existing approaches with four main improvements: (1) Multi-Task Holistic Evaluation that spans diverse software engineering activities rather than limited coding tasks; (2) Multi-Language and Multi-Modality Assessment that extends beyond traditional single-language, text-only benchmarks to include multi-modality coding tasks; (3) Robustness Assessment that evaluates model reliability under semantically-preserving code transformations; and (4) Rigorous Evaluation Methodology that enhances the trustworthiness of evaluation results through diverse evaluation prompts and adaptive solution extraction. Based on this evaluation framework, we assess 26 state-of-the-art models and uncover both their strengths and limitations, yielding several key insights:(1) Current models show substantial performance variation across programming tasks; (2) Multi-modal language models demonstrate specific performance limitations in UI code generation and edit;

---

## 68. Do Satellite Tasks Need Special Pretraining?

**论文链接:** [http://arxiv.org/abs/2510.17014v1](http://arxiv.org/abs/2510.17014v1)

**作者:** Ani Vanyan, Alvard Barseghyan, Hakob Tamazyan, Tigran Galstyan, Vahan Huroyan, Naira Hovakimyan, Hrant Khachatrian

**发布时间:** 2025-10-19

### GPT解析

### 总结

该研究挑战了特定遥感基础模型比通用视觉基础模型更有用的观点，特别是在小规模应用中。作者设计了一个评估模型对低分辨率图像泛化能力的基准，并在卫星图像数据集上训练了iBOT模型，但发现没有预训练模型能比通用基线带来一致改进。

### 背景

基础模型已在多种模态中推动了机器学习的发展，最近多个团队训练了专门用于遥感应用的基础模型。这一研究方向受到遥感图像的独特特性、特定应用以及对卫星图像分析有用的鲁棒性类型的驱动。

### 目的

系统性地挑战特定基础模型比通用视觉基础模型更有用的观点，至少在小规模情况下。

### 方法

设计了一个简单的基准来衡量遥感模型对较低分辨率图像的泛化能力；在MillionAID（ImageNet规模的卫星图像数据集）上训练了iBOT（自监督视觉编码器），并进行了针对遥感的若干修改。

### 主要发现

在ViT-B规模下，没有一个预训练模型能比通用基线带来一致的改进。

### 结论

特定基础模型在小规模应用中并不比通用视觉基础模型更有优势。

### 翻译

基础模型已在各种模态中推动了机器学习的发展，包括图像。最近，多个团队训练了专门用于遥感应用的基础模型。这一研究方向受到遥感图像的独特特性、特定应用以及对卫星图像分析有用的鲁棒性类型的驱动。在这项工作中，我们系统地挑战了特定基础模型比通用视觉基础模型更有用的观点，至少在小规模情况下。首先，我们设计了一个简单的基准，用于衡量遥感模型在两个下游任务中对较低分辨率图像的泛化能力。其次，我们在MillionAID（一个ImageNet规模的卫星图像数据集）上训练了iBOT（一种自监督视觉编码器），并进行了针对遥感的若干修改。我们表明，在ViT-B规模下，这些预训练模型中没有哪一个比通用基线带来一致的改进。


### 论文摘要

Foundation models have advanced machine learning across various modalities, including images. Recently multiple teams trained foundation models specialized for remote sensing applications. This line of research is motivated by the distinct characteristics of remote sensing imagery, specific applications and types of robustness useful for satellite image analysis. In this work we systematically challenge the idea that specific foundation models are more useful than general-purpose vision foundation models, at least in the small scale. First, we design a simple benchmark that measures generalization of remote sensing models towards images with lower resolution for two downstream tasks. Second, we train iBOT, a self-supervised vision encoder, on MillionAID, an ImageNet-scale satellite imagery dataset, with several modifications specific to remote sensing. We show that none of those pretrained models bring consistent improvements upon general-purpose baselines at the ViT-B scale.

---

## 69. Graph4MM: Weaving Multimodal Learning with Structural Information

**论文链接:** [http://arxiv.org/abs/2510.16990v1](http://arxiv.org/abs/2510.16990v1)

**作者:** Xuying Ning, Dongqi Fu, Tianxin Wei, Wujiang Xu, Jingrui He

**发布时间:** 2025-10-19

**备注:** ICML 2025

### GPT解析

### 总结

该研究提出了Graph4MM，一个基于图的多模态学习框架，通过Hop-Diffused Attention和MM-QFormer解决了多模态学习中的两个关键挑战：整合多跳邻居结构信息和融合模态特定信息。实验表明该方法显著优于现有模型。

### 背景

现实世界多模态数据具有复杂结构关系，跨模态实体通过上下文依赖和共指关系形成多样连接。图为建模模态内和模态间关系提供强大结构信息，但先前工作未能区分多跳邻居并将图视为独立模态，导致理解碎片化。

### 目的

解决多模态学习中的两个关键挑战：(1)将多跳邻居的结构信息整合到基础模型中，(2)以原则性的方式融合模态特定信息。重新审视图在基础模型时代多模态学习中的作用。

### 方法

提出Graph4MM框架，包含Hop-Diffused Attention（通过因果掩蔽和跳扩散将多跳结构信息整合到自注意力中）和MM-QFormer（用于跨模态融合的多映射查询transformer）。

### 主要发现

利用结构整合模态内和模态间交互比将它们视为独立模态能更好地提升多模态理解。在生成性和判别性任务上，Graph4MM优于更大的VLMs、LLMs和多模态图基线，平均实现6.93%的改进。

### 结论

Graph4MM框架有效解决了多模态学习中的关键挑战，通过整合多跳结构信息和跨模态融合，显著提升了多模态理解能力。

### 翻译

现实世界多模态数据通常表现出超越传统图像-标题对等一对一映射的复杂结构关系。跨模态的实体以复杂的方式交互，图像和文本通过上下文依赖和共指关系形成多样的相互连接。图为建模模态内和模态间关系提供了强大的结构信息。然而，先前的工作未能区分多跳邻居，而是将图视为独立模态，这碎片化了整体理解。这一局限性给多模态学习带来了两个关键挑战：(1)将多跳邻居的结构信息整合到基础模型中，(2)以原则性的方式融合模态特定信息。为应对这些挑战，我们重新审视了基础模型时代图在多模态学习中的作用，并提出了Graph4MM，一个基于图的多模态学习框架。具体而言，我们引入了Hop-Diffused Attention，通过因果掩蔽和跳扩散将多跳结构信息整合到自注意力中。此外，我们设计了MM-QFormer，一个用于跨模态融合的多映射查询transformer。通过理论和经验分析，我们表明利用结构整合模态内和模态间交互，比将它们视为独立模态能更好地提升多模态理解。在生成性和判别性任务上的实验表明，Graph4MM优于更大的VLMs、LLMs和多模态图基线，实现了6.93%的平均改进。


### 论文摘要

Real-world multimodal data usually exhibit complex structural relationships beyond traditional one-to-one mappings like image-caption pairs. Entities across modalities interact in intricate ways, with images and text forming diverse interconnections through contextual dependencies and co-references. Graphs provide powerful structural information for modeling intra-modal and inter-modal relationships. However, previous works fail to distinguish multi-hop neighbors and treat the graph as a standalone modality, which fragments the overall understanding. This limitation presents two key challenges in multimodal learning: (1) integrating structural information from multi-hop neighbors into foundational models, and (2) fusing modality-specific information in a principled manner. To address these challenges, we revisit the role of graphs in multimodal learning within the era of foundation models and propose Graph4MM, a graph-based multimodal learning framework. To be specific, we introduce Hop-Diffused Attention, which integrates multi-hop structural information into self-attention through causal masking and hop diffusion. Furthermore, we design MM-QFormer, a multi-mapping querying transformer for cross-modal fusion. Through theoretical and empirical analysis, we show that leveraging structures to integrate both intra- and inter-modal interactions improves multimodal understanding beyond treating them as a standalone modality. Experiments on both generative and discriminative tasks show that Graph4MM outperforms larger VLMs, LLMs, and multimodal graph baselines, achieving a 6.93% average improvement.

---

## 70. Foundation Models in Medical Image Analysis: A Systematic Review and Meta-Analysis

**论文链接:** [http://arxiv.org/abs/2510.16973v1](http://arxiv.org/abs/2510.16973v1)

**作者:** Praveenbalaji Rajendran, Mojtaba Safari, Wenfeng He, Mingzhe Hu, Shansong Wang, Jun Zhou, Xiaofeng Yang

**发布时间:** 2025-10-19

### GPT解析

### 总结

这篇综述文章对医学图像分析中的基础模型(FMs)进行了全面和结构化的分析，系统性地分类了研究进展并评估了其临床应用价值。

### 背景

人工智能特别是基础模型的最新进展彻底改变了医学图像分析，在多种医学影像任务中表现出强大的零样本和少样本性能。与传统特定任务AI模型不同，基础模型利用大量标记和非标记的多模态数据集学习通用表示，可通过微调适应各种下游临床应用。

### 目的

弥补医学影像领域基础模型研究的碎片化现状，提供一个统一的综合分析，系统性地映射不同模态下架构、训练范式和临床应用的演变。

### 方法

将研究按架构基础、训练策略和下游临床任务分为纯视觉基础模型和视觉语言基础模型；进行定量元分析，描述数据集利用和应用领域的时间趋势；批判性地讨论持续存在的挑战和新出现的解决方案。

### 主要发现

基础模型可通过微调适应各种临床应用；持续存在的挑战包括领域适应、高效微调、计算限制和可解释性；新兴解决方案包括联邦学习、知识蒸馏和高级提示技术。

### 结论

需要加强基础模型的鲁棒性、可解释性和临床集成研究，以加速这些模型转化为实际医疗实践。

### 翻译

人工智能(AI)特别是基础模型(FMs)的最新进展彻底改变了医学图像分析，在从分割到报告生成的多种医学影像任务中表现出强大的零样本和少样本性能。与传统的特定任务AI模型不同，基础模型利用大量标记和非标记的多模态数据集学习通用表示，这些通用表示可以通过微调适应各种下游临床应用。然而，尽管医学影像中基础模型研究迅速增长，该领域仍然碎片化，缺乏一个统一的综合分析来系统性地映射不同模态下架构、训练范式和临床应用的演变。为解决这一差距，这篇综述文章对医学图像分析中的基础模型提供了全面和结构化的分析。我们根据架构基础、训练策略和下游临床任务将研究系统性地分为纯视觉基础模型和视觉语言基础模型。此外，还对研究进行了定量元分析，以描述数据集利用和应用领域的时间趋势。我们还批判性地讨论了持续存在的挑战，包括领域适应、高效微调、计算限制和可解释性，以及新兴的解决方案，如联邦学习、知识蒸馏和高级提示技术。最后，我们确定了旨在增强基础模型的鲁棒性、可解释性和临床集成的关键未来研究方向，从而加速它们转化为实际医疗实践。


### 论文摘要

Recent advancements in artificial intelligence (AI), particularly foundation models (FMs), have revolutionized medical image analysis, demonstrating strong zero- and few-shot performance across diverse medical imaging tasks, from segmentation to report generation. Unlike traditional task-specific AI models, FMs leverage large corpora of labeled and unlabeled multimodal datasets to learn generalized representations that can be adapted to various downstream clinical applications with minimal fine-tuning. However, despite the rapid proliferation of FM research in medical imaging, the field remains fragmented, lacking a unified synthesis that systematically maps the evolution of architectures, training paradigms, and clinical applications across modalities. To address this gap, this review article provides a comprehensive and structured analysis of FMs in medical image analysis. We systematically categorize studies into vision-only and vision-language FMs based on their architectural foundations, training strategies, and downstream clinical tasks. Additionally, a quantitative meta-analysis of the studies was conducted to characterize temporal trends in dataset utilization and application domains. We also critically discuss persistent challenges, including domain adaptation, efficient fine-tuning, computational constraints, and interpretability along with emerging solutions such as federated learning, knowledge distillation, and advanced prompting. Finally, we identify key future research directions aimed at enhancing the robustness, explainability, and clinical integration of FMs, thereby accelerating their translation into real-world medical practice.

---

## 71. Chem-R: Learning to Reason as a Chemist

**论文链接:** [http://arxiv.org/abs/2510.16880v1](http://arxiv.org/abs/2510.16880v1)

**作者:** Weida Wang, Benteng Chen, Di Zhang, Wanhao Liu, Shuchen Pu, Ben Gao, Jin Zeng, Lei Bai, Wanli Ouyang, Xiaoyong Wei, Tianshu Yu, Tianfan Fu, Shuzhou Sun, Jiatong Li, Zifu Wang, Yuqiang Li, Shufei Zhang

**发布时间:** 2025-10-19

**备注:** 9 pages, 5 figures, 14 tables

### GPT解析

### 总结

Chem-R是一个通用的化学推理模型，通过三阶段训练框架实现先进化学推理能力，在综合基准测试上取得最先进性能，超越现有模型。

### 背景

当前大语言模型在化学发现方面缺乏核心化学知识，推理轨迹不可靠，且在各类化学任务中表现不佳。

### 目的

解决现有大语言模型在化学领域的局限性，开发一个能模拟化学家深思熟虑过程的通用化学推理模型。

### 方法

通过三阶段框架训练：1)化学基础训练建立核心知识；2)化学推理协议蒸馏融入结构化专家推理轨迹；3)多任务组相对策略优化模型在分子和反应级任务上的平衡性能。

### 主要发现

Chem-R在综合基准测试上取得最先进性能，超越Gemini-2.5-Pro和DeepSeek-R1等领先模型，在分子任务上领先最多46%，在反应任务上领先最多66%，且在分子和反应级任务上都优于现有化学基础模型。

### 结论

Chem-R具有强大的泛化能力和可解释性，有望成为下一代AI驱动化学发现的基础。

### 翻译

尽管大型语言模型在化学发现方面具有巨大潜力，但当前模型缺乏核心化学知识，产生不可靠的推理轨迹，并在各种化学任务中表现不佳。为解决这些挑战，我们提出了Chem-R，一个通用的化学推理模型，旨在模拟化学家的深思熟虑过程。Chem-R通过三阶段框架进行训练，逐步构建高级推理能力：1)化学基础训练，建立核心化学知识；2)化学推理协议蒸馏，融入结构化、专家式的推理轨迹，引导系统化和可靠的问题解决；3)多任务组相对策略优化，优化模型在多样化的分子级和反应级任务上的平衡性能。这种结构化管道使Chem-R能够在综合基准测试上实现最先进的性能，超越包括Gemini-2.5-Pro和DeepSeek-R1在内的领先大型语言模型，在分子任务上领先高达46%，在反应任务上领先高达66%。同时，Chem-R在分子和反应级任务上也 consistently 优于现有的化学基础模型。这些结果突显了Chem-R的强大泛化能力、可解释性以及作为下一代AI驱动化学发现基础的潜力。


### 论文摘要

Although large language models (LLMs) have significant potential to advance chemical discovery, current LLMs lack core chemical knowledge, produce unreliable reasoning trajectories, and exhibit suboptimal performance across diverse chemical tasks. To address these challenges, we propose Chem-R, a generalizable Chemical Reasoning model designed to emulate the deliberative processes of chemists. Chem-R is trained through a three-phase framework that progressively builds advanced reasoning capabilities, including: 1) Chemical Foundation Training, which establishes core chemical knowledge. 2) Chemical Reasoning Protocol Distillation, incorporating structured, expert-like reasoning traces to guide systematic and reliable problem solving. 3) Multi-task Group Relative Policy Optimization that optimizes the model for balanced performance across diverse molecular- and reaction-level tasks. This structured pipeline enables Chem-R to achieve state-of-the-art performance on comprehensive benchmarks, surpassing leading large language models, including Gemini-2.5-Pro and DeepSeek-R1, by up to 46% on molecular tasks and 66% on reaction tasks. Meanwhile, Chem-R also consistently outperforms the existing chemical foundation models across both molecular and reaction level tasks. These results highlight Chem-R's robust generalization, interpretability, and potential as a foundation for next-generation AI-driven chemical discovery.

---

## 72. EMRRG: Efficient Fine-Tuning Pre-trained X-ray Mamba Networks for Radiology Report Generation

**论文链接:** [http://arxiv.org/abs/2510.16776v1](http://arxiv.org/abs/2510.16776v1)

**作者:** Mingzheng Zhang, Jinfeng Gao, Dan Xu, Jiangrui Yu, Yuhan Qiao, Lan Chen, Jin Tang, Xiao Wang

**发布时间:** 2025-10-19

### GPT解析

### 总结

本文提出了一种名为EMRRG的新型X光报告生成框架，该框架使用参数高效方法微调预训练的Mamba网络，结合具有混合解码器的LLM生成医学报告，在基准数据集上取得了良好效果。

### 背景

X光图像医学报告生成(MRG)是人工智能的重要领域，可以显著减轻临床医生的诊断负担和患者等待时间。现有MRG模型主要依赖大型语言模型，对预训练视觉基础模型或高级微调技术的探索有限，且非Transformer架构在医学报告生成中研究不足。

### 目的

提出一种新的X光报告生成框架EMRRG，使用参数高效方法微调预训练的Mamba网络，探索非Transformer架构在医学报告生成中的应用潜力。

### 方法

将X光图像分割成块并进行标记化处理，通过基于SSM的视觉主干网络进行特征提取，使用部分LoRA技术获得最佳性能。采用具有混合解码器的LLM生成医学报告，实现端到端训练。

### 主要发现

在三个广泛使用的基准数据集上的大量实验验证了所提出策略在X光医学报告生成中的有效性。

### 结论

EMRRG框架是一种有效的X光医学报告生成方法，结合了Mamba网络和参数高效微调技术，为医学报告生成领域提供了新的研究方向。

### 翻译

X-ray image-based medical report generation (MRG) is a pivotal area in artificial intelligence that can significantly reduce diagnostic burdens for clinicians and patient wait times. Existing MRG models predominantly rely on Large Language Models (LLMs) to improve report generation, with limited exploration of pre-trained vision foundation models or advanced fine-tuning techniques. Mainstream frameworks either avoid fine-tuning or utilize simplistic methods like LoRA, often neglecting the potential of enhancing cross-attention mechanisms. Additionally, while Transformer-based models dominate vision-language tasks, non-Transformer architectures, such as the Mamba network, remain underexplored for medical report generation, presenting a promising avenue for future research. In this paper, we propose EMRRG, a novel X-ray report generation framework that fine-tunes pre-trained Mamba networks using parameter-efficient methods. Specifically, X-ray images are divided into patches, tokenized, and processed by an SSM-based vision backbone for feature extraction, with Partial LoRA yielding optimal performance. An LLM with a hybrid decoder generates the medical report, enabling end-to-end training and achieving strong results on benchmark datasets. Extensive experiments on three widely used benchmark datasets fully validated the effectiveness of our proposed strategies for the X-ray MRG. The source code of this paper will be released on https://github.com/Event-AHU/Medical_Image_Analysis.


### 论文摘要

X-ray image-based medical report generation (MRG) is a pivotal area in artificial intelligence that can significantly reduce diagnostic burdens for clinicians and patient wait times. Existing MRG models predominantly rely on Large Language Models (LLMs) to improve report generation, with limited exploration of pre-trained vision foundation models or advanced fine-tuning techniques. Mainstream frameworks either avoid fine-tuning or utilize simplistic methods like LoRA, often neglecting the potential of enhancing cross-attention mechanisms. Additionally, while Transformer-based models dominate vision-language tasks, non-Transformer architectures, such as the Mamba network, remain underexplored for medical report generation, presenting a promising avenue for future research. In this paper, we propose EMRRG, a novel X-ray report generation framework that fine-tunes pre-trained Mamba networks using parameter-efficient methods. Specifically, X-ray images are divided into patches, tokenized, and processed by an SSM-based vision backbone for feature extraction, with Partial LoRA yielding optimal performance. An LLM with a hybrid decoder generates the medical report, enabling end-to-end training and achieving strong results on benchmark datasets. Extensive experiments on three widely used benchmark datasets fully validated the effectiveness of our proposed strategies for the X-ray MRG. The source code of this paper will be released on https://github.com/Event-AHU/Medical_Image_Analysis.

---

## 73. DistilLock: Safeguarding LLMs from Unauthorized Knowledge Distillation on the Edge

**论文链接:** [http://arxiv.org/abs/2510.16716v1](http://arxiv.org/abs/2510.16716v1)

**作者:** Asmita Mohanty, Gezheng Kang, Lei Gao, Murali Annavaram

**发布时间:** 2025-10-19

### GPT解析

### 总结

DistilLock是一个TEE辅助的微调框架，能够在边缘设备上实现隐私保护的知识蒸馏，同时保护数据隐私和模型知识产权。

### 背景

大型语言模型在各种任务上表现出色，但微调通常依赖于基于云的集中式基础设施，需要数据所有者上传敏感数据，引发隐私问题；而在边缘设备上直接微调则存在模型知识产权泄露风险。

### 目的

解决在保护数据隐私和模型知识产权的同时，在边缘设备上高效微调大型语言模型的困境。

### 方法

提出DistilLock框架，在TEE中执行专有基础模型作为安全黑盒教师，并采用模型模糊化机制将模糊化权重卸载到不可信加速器上进行知识蒸馏。

### 主要发现

DistilLock能够防止未授权的知识蒸馏过程和模型窃取攻击，同时保持高计算效率。

### 结论

DistilLock为基于边缘的LLM个性化提供了一种安全且实用的解决方案。

### 翻译

大型语言模型已在各种任务上展现出强大的性能，但对其进行微调通常依赖于基于云的集中式基础设施。这需要数据所有者将可能敏感的数据上传到外部服务器，引发严重的隐私问题。另一种替代方法是在边缘设备上使用本地数据直接微调LLMs；然而，这带来了新的挑战：模型所有者必须将专有模型传输到边缘设备，这存在知识产权泄露的风险。为了解决这一困境，我们提出了DistilLock，一个TEE辅助的微调框架，能够在边缘上实现隐私保护的知识蒸馏。在DistilLock中，专有基础模型在数据所有者设备上的可信执行环境enclave中执行，充当安全的黑盒教师。这种设置通过防止直接访问模型内部，既保护了数据隐私又保护了模型IP。此外，DistilLock采用模型模糊化机制，将模糊化的权重卸载到不可信的加速器上，以实现高效的知识蒸馏而不损害安全性。我们证明，DistilLock能够防止未授权的知识蒸馏过程和模型窃取攻击，同时保持高计算效率，为基于边缘的LLM个性化提供了一种安全且实用的解决方案。


### 论文摘要

Large Language Models (LLMs) have demonstrated strong performance across diverse tasks, but fine-tuning them typically relies on cloud-based, centralized infrastructures. This requires data owners to upload potentially sensitive data to external servers, raising serious privacy concerns. An alternative approach is to fine-tune LLMs directly on edge devices using local data; however, this introduces a new challenge: the model owner must transfer proprietary models to the edge, which risks intellectual property (IP) leakage. To address this dilemma, we propose DistilLock, a TEE-assisted fine-tuning framework that enables privacy-preserving knowledge distillation on the edge. In DistilLock, a proprietary foundation model is executed within a trusted execution environment (TEE) enclave on the data owner's device, acting as a secure black-box teacher. This setup preserves both data privacy and model IP by preventing direct access to model internals. Furthermore, DistilLock employs a model obfuscation mechanism to offload obfuscated weights to untrusted accelerators for efficient knowledge distillation without compromising security. We demonstrate that DistilLock prevents unauthorized knowledge distillation processes and model-stealing attacks while maintaining high computational efficiency, but offering a secure and practical solution for edge-based LLM personalization.

---

## 74. Universal and Transferable Attacks on Pathology Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.16660v1](http://arxiv.org/abs/2510.16660v1)

**作者:** Yuntian Wang, Xilin Yang, Che-Yung Shen, Nir Pillar, Aydogan Ozcan

**发布时间:** 2025-10-18

**备注:** 38 Pages, 8 Figures

### GPT解析

### 总结

研究团队提出了UTAP（通用可迁移对抗扰动）方法，揭示病理学基础模型的关键漏洞，该扰动能系统性地破坏多个模型的特征表示能力，导致下游任务性能下降。

### 背景

病理学基础模型在医学诊断中应用广泛，但存在安全漏洞和鲁棒性问题，需要评估和防御机制。

### 目的

开发一种通用的对抗扰动方法，评估病理学基础模型的鲁棒性，并推动防御机制的发展。

### 方法

使用深度学习优化UTAP，创建一种固定且微弱的噪声模式，添加到病理图像中以破坏基础模型的特征表示能力。

### 主要发现

UTAP导致下游任务性能下降，具有通用性（可应用于不同视野范围，与开发数据集无关）和可迁移性（能降低各种黑盒病理学基础模型的性能），构成对多种病理学基础模型的广泛威胁。

### 结论

UTAP为模型鲁棒性评估设定了高标准基准，突显了推进防御机制的必要性，可能为对抗训练提供资源，确保AI在病理学中的安全可靠部署。

### 翻译

我们引入了针对病理学基础模型的通用可迁移对抗扰动（UTAP），揭示了其能力中的关键漏洞。使用深度学习优化，UTAP由固定且微弱的噪声模式组成，当添加到病理图像中时，会系统性地破坏多个病理学基础模型的特征表示能力。因此，UTAP会导致利用基础模型的下游任务性能下降，包括在广泛未见过的数据分布上的错误分类。除了损害模型性能外，我们还证明了UTAP的两个关键特性：(1) 通用性：其扰动可以应用于不同的视野范围，且与开发UTAP的数据集无关；(2) 可迁移性：其扰动可以成功降低各种外部、黑盒病理学基础模型的性能——这些模型以前从未见过。这两个特性表明，UTAP不是与特定基础模型或图像数据集相关的专门攻击，而是对各种新兴病理学基础模型及其应用构成广泛威胁。我们在多个数据集上的各种最先进病理学基础模型上系统评估了UTAP，使用固定噪声模式对输入图像进行几乎不可见的修改，导致了其性能显著下降。这些强大攻击的建立为模型鲁棒性评估设定了关键的高标准基准，突显了推进防御机制的必要性，并可能为对抗训练提供必要资源，以确保AI在病理学中的安全可靠部署。


### 论文摘要

We introduce Universal and Transferable Adversarial Perturbations (UTAP) for pathology foundation models that reveal critical vulnerabilities in their capabilities. Optimized using deep learning, UTAP comprises a fixed and weak noise pattern that, when added to a pathology image, systematically disrupts the feature representation capabilities of multiple pathology foundation models. Therefore, UTAP induces performance drops in downstream tasks that utilize foundation models, including misclassification across a wide range of unseen data distributions. In addition to compromising the model performance, we demonstrate two key features of UTAP: (1) universality: its perturbation can be applied across diverse field-of-views independent of the dataset that UTAP was developed on, and (2) transferability: its perturbation can successfully degrade the performance of various external, black-box pathology foundation models - never seen before. These two features indicate that UTAP is not a dedicated attack associated with a specific foundation model or image dataset, but rather constitutes a broad threat to various emerging pathology foundation models and their applications. We systematically evaluated UTAP across various state-of-the-art pathology foundation models on multiple datasets, causing a significant drop in their performance with visually imperceptible modifications to the input images using a fixed noise pattern. The development of these potent attacks establishes a critical, high-standard benchmark for model robustness evaluation, highlighting a need for advancing defense mechanisms and potentially providing the necessary assets for adversarial training to ensure the safe and reliable deployment of AI in pathology.

---

## 75. Hallucination Benchmark for Speech Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.16567v1](http://arxiv.org/abs/2510.16567v1)

**作者:** Alkis Koudounas, Moreno La Quatra, Manuel Giollo, Sabato Marco Siniscalchi, Elena Baralis

**发布时间:** 2025-10-18

**备注:** Under Review

### GPT解析

### 总结

本文提出了SHALLOW，第一个系统分类和量化语音识别系统幻觉现象的基准框架，解决了传统评估指标无法区分幻觉与其他类型错误的问题。

### 背景

自动语音识别系统中的幻觉现象指的是神经模型产生的流畅连贯的转录，但这些转录与底层声学输入完全无关。这些幻觉虽然类似于传统解码错误，但由于保留了句法和语义上合理的结构，可能更具危害性，特别是在医疗和法律等关键领域。传统评估指标主要基于错误指标，无法区分语音不准确和幻觉。

### 目的

开发新的评估框架，能够有效识别和评估产生幻觉内容倾向更高的模型，并提供更细粒度的错误分析。

### 方法

提出SHALLOW框架，系统性地沿着四个互补轴对ASR中的幻觉现象进行分类和量化：词汇、语音、形态和语义。在每个类别中定义有针对性的指标，以产生可解释的模型行为特征。

### 主要发现

通过评估各种架构和语音领域，发现当识别质量高（即低词错误率WER）时，SHALLOW指标与WER高度相关；但随着WER的增加，这种相关性显著减弱。SHALLOW能够捕获在降级和挑战性条件下WER无法区分的细粒度错误模式。

### 结论

SHALLOW框架支持对模型弱点的具体诊断，并提供比总体错误率所能提供的更详细的模型改进反馈，有助于提高语音识别系统在关键领域的可靠性。

### 翻译

自动语音识别系统中的幻觉现象指的是神经ASR模型产生的流畅连贯的转录，这些转录与底层声学输入（即语音信号）完全无关。虽然幻觉可能类似于传统解码错误，在可能降低转录对下游应用的可用性方面，但幻觉由于其保留了句法和语义上合理的结构，可能更具危害性。这种明显的连贯性可能误导后续处理阶段并引入严重风险，特别是在医疗和法律等关键领域。传统评估指标主要基于错误指标，无法区分语音不准确和幻觉。因此，迫切需要新的评估框架，能够有效识别和评估产生幻觉内容倾向更高的模型。为此，我们引入了SHALLOW，这是第一个基准框架，系统性地沿着四个互补轴对ASR中的幻觉现象进行分类和量化：词汇、语音、形态和语义。我们在每个类别中定义有针对性的指标，以产生可解释的模型行为特征。通过评估各种架构和语音领域，我们发现当识别质量高（即低WER）时，SHALLOW指标与词错误率（WER）高度相关。然而，随着WER的增加，这种相关性显著减弱。因此，SHALLOW捕获了在降级和挑战性条件下WER无法区分的细粒度错误模式。我们的框架支持对模型弱点的具体诊断，并提供比总体错误率所能提供的更详细的模型改进反馈。


### 论文摘要

Hallucinations in automatic speech recognition (ASR) systems refer to fluent and coherent transcriptions produced by neural ASR models that are completely unrelated to the underlying acoustic input (i.e., the speech signal). While similar to conventional decoding errors in potentially compromising the usability of transcriptions for downstream applications, hallucinations can be more detrimental due to their preservation of syntactically and semantically plausible structure. This apparent coherence can mislead subsequent processing stages and introduce serious risks, particularly in critical domains such as healthcare and law. Conventional evaluation metrics are primarily centered on error-based metrics and fail to distinguish between phonetic inaccuracies and hallucinations. Consequently, there is a critical need for new evaluation frameworks that can effectively identify and assess models with a heightened propensity for generating hallucinated content. To this end, we introduce SHALLOW, the first benchmark framework that systematically categorizes and quantifies hallucination phenomena in ASR along four complementary axes: lexical, phonetic, morphological, and semantic. We define targeted metrics within each category to produce interpretable profiles of model behavior. Through evaluation across various architectures and speech domains, we have found that SHALLOW metrics correlate strongly with word error rate (WER) when recognition quality is high (i.e., low WER). Still, this correlation weakens substantially as WER increases. SHALLOW, therefore, captures fine-grained error patterns that WER fails to distinguish under degraded and challenging conditions. Our framework supports specific diagnosis of model weaknesses and provides feedback for model improvement beyond what aggregate error rates can offer.

---

## 76. NeurIPT: Foundation Model for Neural Interfaces

**论文链接:** [http://arxiv.org/abs/2510.16548v1](http://arxiv.org/abs/2510.16548v1)

**作者:** Zitao Fang, Chenxuan Li, Hongting Zhou, Shuyang Yu, Guodong Du, Ashwaq Qasem, Yang Lu, Jing Li, Junsong Zhang, Sim Kuan Goh

**发布时间:** 2025-10-18

**备注:** Accepted by The Thirty-Ninth Annual Conference on Neural Information  Processing Systems (NeurIPS 2025). Project Page:  https://ZzzitaoFang.github.io/projects/NeurIPT/

### GPT解析

### 总结

本文提出了NeurIPT，一种专为多样化EEG神经接口设计的基础模型，通过结合基于幅度的掩码预训练和渐进式专家混合架构，有效捕捉EEG信号中的时空特征，在多个BCI数据集上取得了最先进的性能。

### 背景

脑电图(EEG)在临床诊断和脑机接口中有广泛应用，随着EEG数据量和多样性的增加，建立基础模型来扩展和泛化神经解码成为研究热点。然而，将基础模型应用于EEG仍面临受试者间、任务间和条件间变异性，以及不同电极配置带来的挑战。

### 目的

开发一种能够处理EEG数据多样性和变异性的基础模型，以提高神经解码的性能和泛化能力。

### 方法

提出NeurIPT基础模型，采用预训练Transformer架构；时间维度引入基于信号幅度的掩码预训练(AAMP)和渐进式专家混合(PMoE)架构；空间维度利用电极的三维物理坐标实现跨设置的嵌入迁移，并开发脑叶内-间池化(IILP)以利用区域脑特征。

### 主要发现

在八个下游BCI数据集上的实证评估表明，NeurIPT通过微调始终取得了最先进的性能，展示了其广泛的适用性和鲁棒的泛化能力。

### 结论

这项工作推动了EEG基础模型的前沿发展，并为可扩展和可泛化的神经信息处理系统提供了见解。

### 翻译

脑电图(EEG)有广泛的应用，从临床诊断到脑机接口(BCIs)。随着EEG数据的数量和多样性的增加，人们越来越有兴趣建立基础模型(FMs)来扩展和泛化神经解码。尽管显示出早期潜力，但由于显著的受试者间、任务间和条件间变异性，以及不同记录设置中的多样化电极配置，将基础模型应用于EEG仍然具有挑战性。为了解决这些开放挑战，我们提出了NeurIPT，这是一种为多样化EEG神经接口开发的基础模型，通过捕捉EEG信号中固有的同质和异质时空特征，采用预训练Transformer架构。在时间维度上，我们引入了基于幅度的掩码预训练(AAMP)，基于信号幅度而非随机间隔进行掩码，以学习跨越不同信号强度的鲁棒表示，而不仅仅是局部插值。此外，这种时间表示通过渐进式专家混合(PMoE)架构得到增强，在更深层次逐步引入专门的专家子网络，有效适应EEG信号的多样化时间特征。在空间维度上，NeurIPT利用电极的三维物理坐标，实现不同EEG设置间的嵌入有效迁移，并在微调过程中开发脑叶内-间池化(IILP)，以高效利用区域脑特征。在八个下游BCI数据集上的实证评估，通过微调，证明了NeurIPT始终取得了最先进的性能，突显了其广泛的适用性和鲁棒的泛化能力。我们的工作推动了EEG基础模型的前沿发展，并为可扩展和可泛化的神经信息处理系统提供了见解。


### 论文摘要

Electroencephalography (EEG) has wide-ranging applications, from clinical diagnosis to brain-computer interfaces (BCIs). With the increasing volume and variety of EEG data, there has been growing interest in establishing foundation models (FMs) to scale up and generalize neural decoding. Despite showing early potential, applying FMs to EEG remains challenging due to substantial inter-subject, inter-task, and inter-condition variability, as well as diverse electrode configurations across recording setups. To tackle these open challenges, we propose NeurIPT, a foundation model developed for diverse EEG-based Neural Interfaces with a Pre-trained Transformer by capturing both homogeneous and heterogeneous spatio-temporal characteristics inherent in EEG signals. Temporally, we introduce Amplitude-Aware Masked Pretraining (AAMP), masking based on signal amplitude rather than random intervals, to learn robust representations across varying signal intensities beyond local interpolation. Moreover, this temporal representation is enhanced by a Progressive Mixture-of-Experts (PMoE) architecture, where specialized expert subnetworks are progressively introduced at deeper layers, adapting effectively to the diverse temporal characteristics of EEG signals. Spatially, NeurIPT leverages the 3D physical coordinates of electrodes, enabling effective transfer of embedding across varying EEG settings, and develops Intra-Inter Lobe Pooling (IILP) during fine-tuning to efficiently exploit regional brain features. Empirical evaluations across eight downstream BCI datasets, via fine-tuning, demonstrated NeurIPT consistently achieved state-of-the-art performance, highlighting its broad applicability and robust generalization. Our work pushes forward the state of FMs in EEG and offers insights into scalable and generalizable neural information processing systems.

---

## 77. VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion

**论文链接:** [http://arxiv.org/abs/2510.16446v1](http://arxiv.org/abs/2510.16446v1)

**作者:** Jaekyun Park, Hye Won Chung

**发布时间:** 2025-10-18

**备注:** NeurIPS 2025

### GPT解析

### 总结

VIPAMIN是一种视觉提示初始化策略，通过将提示与嵌入空间中的语义信息丰富区域对齐，并向预训练子空间注入新的表示方向，增强自监督模型的适应性，在各种任务和数据集大小上一致提高性能。

### 背景

在大规模基础模型时代，为每个下游任务完全微调预训练网络通常需要大量资源。

### 目的

解决现有视觉提示调整方法在专门化提示或丰富表示空间方面的局限性，特别是在自监督主干网络应用于具有挑战性任务和数据稀缺环境时。

### 方法

提出VIPAMIN，一种视觉提示初始化策略，通过两种方式增强自监督模型的适应性：(1)将提示与嵌入空间中的语义信息丰富的区域对齐，(2)向预训练子空间注入新的表示方向。该方法仅需一次前向传播和轻量级操作。

### 主要发现

VIPAMIN在各种任务和数据集大小上一致提高了性能，在视觉提示调整方面树立了新的最先进水平，特别是在具有挑战性的任务和数据稀缺环境中表现突出。

### 结论

VIPAMIN是一种简单而有效的视觉提示初始化策略，能够显著提升自监督模型在下游任务中的适应性和性能。

### 翻译

在大规模基础模型时代，为每个下游任务完全微调预训练网络通常需要大量资源。提示调整通过引入可调整提示同时保持主干网络冻结，提供了一种轻量级替代方案。然而，现有的视觉提示调整方法通常无法专门化提示或丰富表示空间——特别是当应用于自监督主干网络时。我们表明，这些限制在具有挑战性的任务和数据稀缺环境中变得尤为明显，而有效适应在这些环境中最为关键。在这项工作中，我们介绍了VIPAMIN，一种视觉提示初始化策略，通过(1)将提示与嵌入空间中的语义信息丰富区域对齐，和(2)向预训练子空间注入新的表示方向，来增强自监督模型的适应性。尽管简单——只需要一次前向传播和轻量级操作——VIPAMIN在不同任务和数据集大小上一致提高了性能，在视觉提示调整方面树立了新的最先进水平。我们的代码可在https://github.com/iamjaekyun/vipamin获取。


### 论文摘要

In the era of large-scale foundation models, fully fine-tuning pretrained networks for each downstream task is often prohibitively resource-intensive. Prompt tuning offers a lightweight alternative by introducing tunable prompts while keeping the backbone frozen. However, existing visual prompt tuning methods often fail to specialize the prompts or enrich the representation space--especially when applied to self-supervised backbones. We show that these limitations become especially pronounced in challenging tasks and data-scarce settings, where effective adaptation is most critical. In this work, we introduce VIPAMIN, a visual prompt initialization strategy that enhances adaptation of self-supervised models by (1) aligning prompts with semantically informative regions in the embedding space, and (2) injecting novel representational directions beyond the pretrained subspace. Despite its simplicity--requiring only a single forward pass and lightweight operations--VIPAMIN consistently improves performance across diverse tasks and dataset sizes, setting a new state of the art in visual prompt tuning. Our code is available at https://github.com/iamjaekyun/vipamin.

---

## 78. Probing the Hidden Talent of ASR Foundation Models for L2 English Oral Assessment

**论文链接:** [http://arxiv.org/abs/2510.16387v1](http://arxiv.org/abs/2510.16387v1)

**作者:** Fu-An Chao, Bi-Cheng Yan, Berlin Chen

**发布时间:** 2025-10-18

### GPT解析

### 总结

这篇论文探讨了Whisper模型在第二语言口语评估中的应用潜力，通过提取声学和语言特征，实现了超越现有方法的性能，并揭示了模型内在编码语言能力的特点。

### 背景

Whisper是一个成熟的自动语音识别基础模型，先前研究主要分析其生成的转录文本，而对其潜在能力的探索不足。

### 目的

探索Whisper模型在第二语言口语评估任务中的潜在能力，分析其内在编码的语言能力特点。

### 方法

从Whisper的隐藏表示中提取声学和语言特征，在Whisper的中间和最终输出之上训练轻量级分类器，并融入图像和文本提示信息作为辅助线索。

### 主要发现

Whisper模型在GEPT图片描述数据集上超越了现有最先进基线；融入图像和文本提示信息可进一步提升性能；即使没有任务特定微调，Whisper也能内在编码语言熟练程度的顺序模式和语音的语义方面。

### 结论

Whisper模型具有作为第二语言口语评估和其他口语理解任务强大基础的潜力。

### 翻译

在本文中，我们探讨了Whisper这一成熟的自动语音识别基础模型在第二语言口语评估背景下的未开发潜力。与先前研究仅外在分析Whisper生成的转录文本不同，我们的方法更进一步，通过从隐藏表示中提取声学和语言特征来探测其潜在能力。仅通过在Whisper的中间和最终输出之上训练一个轻量级分类器，我们的方法在GEPT图片描述数据集上实现了强大的性能，超越了现有的最先进基线，包括一种多模态方法。此外，通过将图像和文本提示信息作为辅助相关性线索纳入，我们展示了额外的性能提升。最后，我们对Whisper的嵌入进行了深入分析，揭示出即使没有任务特定的微调，该模型也内在地编码了熟练程度的顺序模式和语音的语义方面，突显了其作为第二语言口语评估和其他口语理解任务强大基础的潜力。


### 论文摘要

In this paper, we explore the untapped potential of Whisper, a well-established automatic speech recognition (ASR) foundation model, in the context of L2 spoken language assessment (SLA). Unlike prior studies that extrinsically analyze transcriptions produced by Whisper, our approach goes a step further to probe its latent capabilities by extracting acoustic and linguistic features from hidden representations. With only a lightweight classifier being trained on top of Whisper's intermediate and final outputs, our method achieves strong performance on the GEPT picture-description dataset, outperforming existing cutting-edge baselines, including a multimodal approach. Furthermore, by incorporating image and text-prompt information as auxiliary relevance cues, we demonstrate additional performance gains. Finally, we conduct an in-depth analysis of Whisper's embeddings, which reveals that, even without task-specific fine-tuning, the model intrinsically encodes both ordinal proficiency patterns and semantic aspects of speech, highlighting its potential as a powerful foundation for SLA and other spoken language understanding tasks.

---

## 79. Cosmos-Surg-dVRK: World Foundation Model-based Automated Online Evaluation of Surgical Robot Policy Learning

**论文链接:** [http://arxiv.org/abs/2510.16240v1](http://arxiv.org/abs/2510.16240v1)

**作者:** Lukas Zbinden, Nigel Nelson, Juo-Tung Chen, Xinhao Chen, Ji Woong, Kim, Mahdi Azizian, Axel Krieger, Sean Huver

**发布时间:** 2025-10-17

### GPT解析

### 总结

本研究引入了Cosmos-Surg-dVRK，一种基于Cosmos世界基础模型的外科微调版本，结合视频分类器，实现了手术策略的全自动在线评估和基准测试，解决了在真实机器人平台上评估的高成本、时间和可重复性问题。

### 背景

手术机器人和视觉-语言-动作模型的兴起推动了自主手术策略的发展，但在物理机器人平台（如dVRK）上直接评估这些策略面临高成本、时间需求、可重复性挑战和执行变异性问题。

### 目的

开发一种高保真度的模拟方法，用于评估复杂的现实世界手术任务，并提供全自动化的在线评估和基准测试平台。

### 方法

引入Cosmos-Surg-dVRK（Cosmos世界基础模型的外科微调版本），结合训练好的视频分类器，使用两个不同的外科数据集进行评估。

### 主要发现

在桌面缝合垫任务上，Cosmos-Surg-dVRK中的在线运行与真实dVRK平台上的策略结果具有强相关性；人类标注者与V-JEPA2派生的视频分类器达成良好一致；离体猪胆囊切除术任务的初步实验显示与现实世界评估的良好一致性。

### 结论

Cosmos-Surg-dVRK平台在更复杂的手术程序评估方面具有潜力，为手术策略的自动化评估提供了有效解决方案。

### 翻译

手术机器人和视觉-语言-动作模型的兴起加速了自主手术策略和高效评估策略的发展。然而，在da Vinci研究套件(dVRK)等物理机器人平台上直接评估这些策略仍然受到高成本、时间需求、可重复性挑战和执行变异性的阻碍。物理AI的世界基础模型(WFM)为模拟复杂的现实世界手术任务（如软组织变形）提供了具有高保真度的变革性方法。这项工作介绍了Cosmos-Surg-dVRK，这是Cosmos WFM的外科微调版本，它与训练好的视频分类器一起，实现了手术策略的完全自动化在线评估和基准测试。我们使用两个不同的外科数据集评估了Cosmos-Surg-dVRK。在桌面缝合垫任务上，自动化流程在Cosmos-Surg-dVRK中的在线运行与真实dVRK Si平台上的策略结果之间实现了强相关性，并且人类标注者与V-JEPA2派生的视频分类器之间达成良好一致。此外，在Cosmos-Surg-dVRK中使用离体猪胆囊切除术任务的初步实验显示出与现实世界评估的良好一致性，突显了该平台在更复杂手术程序方面的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决外科机器人策略评估的问题。传统上，直接在物理机器人平台（如dVRK）上评估策略存在成本高、耗时长、可重复性差和执行变异大等挑战。此外，现有模拟器难以准确模拟外科手术中的软组织变形等复杂物理现象。这个问题在现实中很重要，因为随着外科机器人和视觉-语言-动作模型的发展，自主外科策略的开发日益增多，但缺乏高效、可靠的评估方法严重制约了这一领域的进步。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了世界基础模型（WFM）的概念，特别是Cosmos WFM这一视频生成模型，将其作为可扩展的通用学习模拟器。他们针对外科手术领域对Cosmos WFM进行了微调，创建了Cosmos-Surg-dVRK。此外，他们还使用了V-JEPA 2视频分类器来自动评估策略执行的成功率。作者参考了早期世界模型工作（如Ha & Schmidhuber, 2018）和基于扩散过程的大规模多模态世界基础模型，以及现有的视觉-语言-动作模型在外科机器人中的应用，但针对外科手术的特殊需求进行了创新性改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用微调后的Cosmos世界基础模型（Cosmos-Surg-dVRK）作为专门针对外科手术的学习模拟器，结合视频分类器实现完全自动化的策略评估。整体流程包括：1) 使用真实机器人记录的初始状态初始化策略；2) 策略和Cosmos-Surg-dVRK自回归地生成未来帧；3) 将生成的帧组合成视频；4) 使用V-JEPA 2视频分类器自动评估视频中的任务成功或失败；5) 根据评估结果选择最有前途的策略进行真实机器人测试。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 针对外科手术领域微调的Cosmos-Surg-dVRK学习模拟器；2) 使用V-JEPA 2视频分类器实现完全自动化的策略评估；3) 在真实外科任务（包括桌面缝合垫任务和离体猪胆囊切除术）上验证了方法有效性；4) 通过消融研究强调了失败数据在训练中的重要性。相比之前的工作，Cosmos-Surg-dVRK直接从外科数据中学习软组织动力学，不需要显式指定材料属性；使用标准的相对笛卡尔动作空间，实现即插即用的接口；专注于外科手术这一特定领域，而大多数现有方法专注于其他领域如视频游戏或通用机器人。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了Cosmos-Surg-dVRK，一个基于世界基础模型的自动化外科机器人策略评估系统，通过在模拟环境中生成逼真的外科手术视频并自动评估策略性能，有效解决了传统评估中成本高、耗时长和难以模拟软组织变形等问题。'}


### 论文摘要

The rise of surgical robots and vision-language-action models has accelerated the development of autonomous surgical policies and efficient assessment strategies. However, evaluating these policies directly on physical robotic platforms such as the da Vinci Research Kit (dVRK) remains hindered by high costs, time demands, reproducibility challenges, and variability in execution. World foundation models (WFM) for physical AI offer a transformative approach to simulate complex real-world surgical tasks, such as soft tissue deformation, with high fidelity. This work introduces Cosmos-Surg-dVRK, a surgical finetune of the Cosmos WFM, which, together with a trained video classifier, enables fully automated online evaluation and benchmarking of surgical policies. We evaluate Cosmos-Surg-dVRK using two distinct surgical datasets. On tabletop suture pad tasks, the automated pipeline achieves strong correlation between online rollouts in Cosmos-Surg-dVRK and policy outcomes on the real dVRK Si platform, as well as good agreement between human labelers and the V-JEPA 2-derived video classifier. Additionally, preliminary experiments with ex-vivo porcine cholecystectomy tasks in Cosmos-Surg-dVRK demonstrate promising alignment with real-world evaluations, highlighting the platform's potential for more complex surgical procedures.

---

## 80. Probing the Higgs Portal to a Strongly-Interacting Dark Sector at the FCC-ee

**论文链接:** [http://arxiv.org/abs/2510.17675v1](http://arxiv.org/abs/2510.17675v1)

**作者:** Cesare Cazzaniga, Annapaola de Cosa, Felix Kahlhoefer, Andrea S. Maria, Roberto Seidita, Emre Sitti

**发布时间:** 2025-10-20

**备注:** 13 pages, 9 figure, to be submitted to EPJC

### GPT解析

### 总结

本研究探讨了在未来的圆形对撞机e+e-碰撞模式下可能出现的来自禁闭暗区的奇异信号，研究希格斯玻色子作为媒介产生的暗夸克及其导致的半可见喷注终态，并提出使用图神经网络喷注标记器提高信号探测灵敏度。

### 背景

在未来的圆形对撞机中，e+e-碰撞模式下可能产生来自禁闭暗区的奇异信号，希格斯玻色子可能作为标准模型和暗区之间相互作用的媒介。

### 目的

研究希格斯玻色子诱导的半可见喷注的探测方法，提高信号与背景的区分度，增强对希格斯玻色子到暗夸克稀有分支比的探测能力。

### 方法

研究不同不可见状态比例的半可见喷注特性；当不可见成分较大时，基于运动学特征（如缺失能量）进行选择；当不可见成分较小时，采用图神经网络喷注标记器利用喷注亚结构差异进行信号识别。

### 主要发现

对于不可见成分较大的情况，基于缺失能量的选择能提供良好的信背比；对于不可见成分较小的情况，图神经网络喷注标记器能有效提高信号探测灵敏度；所提策略可探测广泛参数空间，将希格斯玻色子到暗夸克的稀有分支比限制在千分之一水平。

### 结论

提出的机器学习策略能有效探测希格斯玻色子诱导的半可见喷注，增强在未来的圆形对撞机上的发现前景，并限制希格斯玻色子的稀有分支比。

### 翻译

本研究探讨了在未来的圆形对撞机e+e-碰撞模式下可能出现的来自禁闭暗区的奇异信号。假设希格斯玻色子介导标准模型与暗区之间的相互作用，暗夸克可以在e+e-碰撞中产生。随后的强动力学可能导致包含可见和不可见粒子的半可见喷注终态。我们研究了不同不可见状态比例的半可见喷注，以及富含轻子和光子的喷注。当不可见成分较大时，基于运动学特征（如事件中的缺失能量）的选择已能提供良好的信背比。对于较小的不可见比例，缺失能量的减少使这些信号更类似于标准模型事件，因此我们采用利用喷注亚结构差异的图神经网络喷注标记器。这种机器学习策略提高了灵敏度，增强了在未来的圆形对撞机上发现希格斯玻色子诱导的半可见喷注的前景。我们的结果表明，所提出的策略可以有效探测所考虑模型的广泛参数空间和各种信号，将希格斯玻色子到暗夸克的稀有分支比限制在千分之一水平。


### 论文摘要

This work explores exotic signatures from confining dark sectors that may arise in the e+e- collision mode at the Future Circular Collider. Assuming the Higgs boson mediates the interaction between the Standard Model and the dark sector, dark quarks can be produced in e+e- collisions. The ensuing strong dynamics may lead to semi-visible jet final states, containing both visible and invisible particles. We investigate semi-visible jets with different fractions of invisible states, and enriched in leptons and photons. When the invisible component is large, selections based on kinematic features, such as the missing energy in the event, already provide good signal-to-background discrimination. For smaller invisible fractions, the reduced missing energy makes these signals more similar to Standard Model events, and we therefore employ a graph neural network jet tagger exploiting differences in jet substructure. This machine learning strategy improves sensitivity and enhances the discovery prospects of Higgs boson-induced semi-visible jets at the Future Circular Collider. Our results show that the proposed strategy can effectively probe a wide parameter space for the models considered, and a variety of signatures, constraining the Higgs boson exotic branching ratios into dark quarks at the permille-level.

---

## 81. Model Metamers Reveal Invariances in Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2510.17378v1](http://arxiv.org/abs/2510.17378v1)

**作者:** Wei Xu, Xiaoyi Jiang, Lixiang Xu, Dechao Tang

**发布时间:** 2025-10-20

### GPT解析

### 总结

本研究通过元形态生成技术揭示了图神经网络(GNNs)中的过度不变性问题，并提出了改进方向和评估基准。

### 背景

深度神经网络在感知系统中被广泛使用以学习具有不变性的表示，模仿人脑机制。然而，视觉和听觉领域的研究证实，人工神经网络的不变性属性与人脑之间仍存在显著差距。

### 目的

研究图神经网络(GNNs)中的不变性行为，探索其与人脑不变性机制的差异。

### 方法

引入元形态生成技术，通过优化输入图使内部节点激活与参考图匹配，获得在表示空间中等效但在结构和特征上显著不同的图。理论研究聚焦于单个节点的局部元形态维度和元形态流形的激活诱导体积变化。

### 主要发现

多种经典GNN架构表现出极端水平的表示不变性。虽然修改模型架构和训练策略可部分减轻这种过度不变性，但无法从根本上达到人脑水平的不变性。

### 结论

量化元形态图与原始图之间的偏差，揭示了当前GNNs的独特失效模式，为模型评估提供了补充基准。

### 翻译

近年来，深度神经网络已被广泛应用于感知系统，以学习具有不变性的表示，旨在模仿人脑中观察到的不变性机制。然而，视觉和听觉领域的研究证实，人工神经网络的不变性与人类之间仍存在显著差距。为了研究图神经网络(GNNs)中的不变性行为，我们引入了一种模型'元形态'生成技术。通过优化输入图使其内部节点激活与参考图相匹配，我们获得了在模型表示空间中等效但在结构和节点特征上显著不同的图。我们的理论分析聚焦于两个方面：单个节点的局部元形态维度和元形态流形的激活诱导体积变化。利用这种方法，我们在几种经典的GNN架构中发现了极端水平的表示不变性。虽然对模型架构和训练策略的有针对性的修改可以部分减轻这种过度不变性，但它们无法从根本上弥合与人脑相似的不变性之间的差距。最后，我们量化了元形态图与其原始对应物之间的偏差，揭示了当前GNNs的独特失效模式，并为模型评估提供了补充基准。


### 论文摘要

In recent years, deep neural networks have been extensively employed in perceptual systems to learn representations endowed with invariances, aiming to emulate the invariance mechanisms observed in the human brain. However, studies in the visual and auditory domains have confirmed that significant gaps remain between the invariance properties of artificial neural networks and those of humans. To investigate the invariance behavior within graph neural networks (GNNs), we introduce a model ``metamers'' generation technique. By optimizing input graphs such that their internal node activations match those of a reference graph, we obtain graphs that are equivalent in the model's representation space, yet differ significantly in both structure and node features. Our theoretical analysis focuses on two aspects: the local metamer dimension for a single node and the activation-induced volume change of the metamer manifold. Utilizing this approach, we uncover extreme levels of representational invariance across several classic GNN architectures. Although targeted modifications to model architecture and training strategies can partially mitigate this excessive invariance, they fail to fundamentally bridge the gap to human-like invariance. Finally, we quantify the deviation between metamer graphs and their original counterparts, revealing unique failure modes of current GNNs and providing a complementary benchmark for model evaluation.

---

## 82. Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses

**论文链接:** [http://arxiv.org/abs/2510.17185v1](http://arxiv.org/abs/2510.17185v1)

**作者:** Runlin Lei, Lu Yi, Mingguo He, Pengyu Qiu, Zhewei Wei, Yongchao Liu, Chuntao Hong

**发布时间:** 2025-10-20

### GPT解析

### 总结

论文提出了一种统一的框架来评估图神经网络和大语言模型在文本属性图学习中的鲁棒性，发现了模型在文本和结构之间存在固有的鲁棒性权衡，并提出了SFT-auto框架来克服这些权衡。

### 背景

图神经网络(GNNs)和大语言模型(LLMs)是学习文本属性图(TAGs)的强大方法，但目前对其鲁棒性的理解还不全面。现有的评估是零散的，未能系统地研究不同模型和攻击场景下文本和结构扰动的不同影响。

### 目的

为了解决这些局限性，作者引入了一个统一的框架来评估TAG学习的鲁棒性，旨在系统地研究不同类型的扰动对各种模型的影响，并提出解决方案来克服发现的权衡。

### 方法

作者提出了一个统一的框架，评估了经典GNNs、鲁棒GNNs(RGNNs)和GraphLLMs在四个领域的十个数据集上的性能，测试了基于文本、基于结构和混合扰动在投毒和规避场景下的影响。为了克服发现的权衡，他们引入了SFT-auto框架。

### 主要发现

1) 模型在文本和结构之间存在固有的鲁棒性权衡；2) GNNs和RGNNs的性能在很大程度上取决于文本编码器和攻击类型；3) GraphLLMs特别容易受到训练数据损坏的影响。

### 结论

该研究为未来的TAG安全研究奠定了基础，并为对抗环境中的鲁棒TAG学习提供了实用的解决方案。作者公开了他们的代码。

### 翻译

尽管图神经网络(GNNs)和大语言模型(LLMs)是学习文本属性图(TAGs)的强大方法，但对其鲁棒性的全面理解仍然模糊。目前的评估是零散的，未能系统地研究不同模型和攻击场景下文本和结构扰动的不同影响。为了解决这些局限性，我们引入了一个统一的综合框架来评估TAG学习中的鲁棒性。我们的框架在四个领域的十个数据集上评估了经典GNNs、鲁棒GNNs(RGNNs)和GraphLLMs，在投毒和规避场景下，应对了多种基于文本、基于结构和混合的扰动。我们的广泛分析揭示了多个发现，其中三个特别值得注意：1) 模型在文本和结构之间存在固有的鲁棒性权衡；2) GNNs和RGNNs的性能在很大程度上取决于文本编码器和攻击类型；3) GraphLLMs特别容易受到训练数据损坏的影响。为了克服识别出的权衡，我们引入了SFT-auto，这是一个新颖的框架，在单一模型内提供针对文本和结构攻击的优越且平衡的鲁棒性。我们的研究为未来的TAG安全研究奠定了基础，并为对抗环境中的鲁棒TAG学习提供了实用的解决方案。我们的代码可在以下网址获取：https://github.com/Leirunlin/TGRB。


### 论文摘要

While Graph Neural Networks (GNNs) and Large Language Models (LLMs) are powerful approaches for learning on Text-Attributed Graphs (TAGs), a comprehensive understanding of their robustness remains elusive. Current evaluations are fragmented, failing to systematically investigate the distinct effects of textual and structural perturbations across diverse models and attack scenarios. To address these limitations, we introduce a unified and comprehensive framework to evaluate robustness in TAG learning. Our framework evaluates classical GNNs, robust GNNs (RGNNs), and GraphLLMs across ten datasets from four domains, under diverse text-based, structure-based, and hybrid perturbations in both poisoning and evasion scenarios. Our extensive analysis reveals multiple findings, among which three are particularly noteworthy: 1) models have inherent robustness trade-offs between text and structure, 2) the performance of GNNs and RGNNs depends heavily on the text encoder and attack type, and 3) GraphLLMs are particularly vulnerable to training data corruption. To overcome the identified trade-offs, we introduce SFT-auto, a novel framework that delivers superior and balanced robustness against both textual and structural attacks within a single model. Our work establishes a foundation for future research on TAG security and offers practical solutions for robust TAG learning in adversarial environments. Our code is available at: https://github.com/Leirunlin/TGRB.

---

## 83. Deep Learning-Based Extraction of Promising Material Groups and Common Features from High-Dimensional Data: A Case of Optical Spectra of Inorganic Crystals

**论文链接:** [http://arxiv.org/abs/2510.17123v1](http://arxiv.org/abs/2510.17123v1)

**作者:** Akira Takahashi, Yu Kumagai, Arata Takamatsu, Fumiyasu Oba

**发布时间:** 2025-10-20

### GPT解析

### 总结

该研究提出了一种深度学习模型的解释方法，用于处理材料科学中的高维光谱数据，并通过特征提取和聚类分析对材料进行分类，最终成功应用于光学吸收光谱预测模型的解释。

### 背景

材料科学研究中需要处理高维光谱数据，而深度学习模型在处理这类数据时存在解释性挑战。传统方法难以同时考虑光谱数据和化学特性（如元素组成和原子排列）的相似性。

### 目的

开发一种能够处理高维光谱数据的深度学习模型解释方法，并根据光谱数据和化学特性的相似性对材料进行分类。

### 方法

使用特征提取和聚类分析技术，结合光谱数据和化学特性（元素组成和原子排列）对材料进行分类。将该方法应用于原子线图神经网络(ALIGNN)模型，该模型使用2,681种金属氧化物、硫属化物及相关化合物的第一性原理计算数据进行训练，用于预测光学吸收光谱。

### 主要发现

分析揭示了影响光学吸收起始特性的关键元素种类及其配位环境，这些因素对材料的光学特性有重要影响。

### 结论

提出的方法适用于各种光谱数据的分类和解释，不仅限于无机晶体的光学吸收光谱，为材料科学研究提供了一种新的分析工具。

### 翻译

我们报道了一种深度学习模型的解释方法，使我们能够处理材料科学中的高维光谱数据。所提出的方法使用特征提取和聚类分析，根据光谱数据以及化学特性（如元素组成和原子排列）的相似性将材料分类。作为演示，我们将此方法应用于原子线图神经网络(ALIGNN)模型，该模型使用2,681种金属氧化物、硫属化物及相关化合物的第一性原理计算数据进行训练，用于预测光学吸收光谱。我们的分析揭示了影响光学吸收起始特性的关键元素种类及其配位环境。本文提出的方法适用于各种光谱数据的分类和解释，超越了无机晶体的光学吸收光谱范围。


### 论文摘要

We report an interpretation method for deep learning models that allows us to handle high-dimensional spectral data in materials science. The proposed method uses feature extraction and clustering analysis to categorize materials into classes based on similarities in both spectral data and chemical characteristics such as elemental composition and atomic arrangement. As a demonstration, we apply this method to an atomistic line graph neural network (ALIGNN) model trained on first-principles calculation data of 2,681 metal oxides, chalcogenides, and related compounds for optical absorption spectrum prediction. Our analysis reveals key elemental species and their coordination environments that influence optical absorption onset characteristics. The method proposed herein is broadly applicable to the classification and interpretation of diverse spectral data, extending beyond the optical absorption spectra of inorganic crystals.

---

## 84. UniGTE: Unified Graph-Text Encoding for Zero-Shot Generalization across Graph Tasks and Domains

**论文链接:** [http://arxiv.org/abs/2510.16885v1](http://arxiv.org/abs/2510.16885v1)

**作者:** Duo Wang, Yuan Zuo, Guangyue Lu, Junjie Wu

**发布时间:** 2025-10-19

### GPT解析

### 总结

UniGTE是一种指令调优的编码器-解码器框架，通过整合图结构与大型语言模型语义，实现了无需任务特定监督的通用图推理能力，在多种图任务上达到最先进的零样本性能。

### 背景

传统图神经网络通常与固定标签空间绑定，而大型语言模型难以捕捉图结构，使得在没有任务特定监督的情况下推广到未见过的图任务具有挑战性。

### 目的

开发一个统一的框架，整合结构和语义推理能力，以解决图神经网络与大型语言模型各自的局限性，实现跨任务和跨域的通用图推理。

### 方法

UniGTE采用增强预训练自回归大型语言模型的编码器，通过可学习对齐令牌和结构感知的图-文本注意力机制，使模型能同时处理标记化的图和自然语言任务提示，同时保持节点排列不变性；编码器生成紧凑的任务感知图表示，基于这些表示，冻结的大型语言模型解码器预测任务答案并重新表述输入图，重建目标正则化编码器保留结构线索。

### 主要发现

UniGTE在五个涵盖不同领域节点级、边级和图级任务的数据集上进行指令调优后，推理时无需微调，在跨任务和跨域设置下的节点分类、链接预测、图分类和图回归任务上实现了新的最先进零样本结果。

### 结论

图结构与大型语言模型语义的紧密集成能够实现强大且可迁移的图推理能力，为无需任务特定监督的通用图学习提供了新方向。

### 翻译

在没有任务特定监督的情况下推广到未见过的图任务具有挑战性：传统图神经网络通常与固定标签空间绑定，而大型语言模型难以捕捉图结构。我们引入UniGTE，一种统一的指令调优编码器-解码器框架，整合了结构和语义推理能力。编码器通过可学习对齐令牌和结构感知的图-文本注意力机制增强预训练的自回归大型语言模型，使其能够同时处理标记化的图和自然语言任务提示，同时保持对节点排列的不变性。这产生了紧凑的、任务感知的图表示。仅基于这些表示，冻结的大型语言模型解码器进行预测和重建：输出任务答案并同时用自然语言重新表述输入图。重建目标正则化编码器以保留结构线索。UniGTE在五个涵盖不同领域节点级、边级和图级任务的数据集上进行指令调优，但推理时不需要微调。在跨任务和跨域设置下的节点分类、链接预测、图分类和图回归任务上，它实现了新的最先进零样本结果，表明图结构与大型语言模型语义的紧密集成能够实现强大且可迁移的图推理能力。


### 论文摘要

Generalizing to unseen graph tasks without task-specific supervision is challenging: conventional graph neural networks are typically tied to a fixed label space, while large language models (LLMs) struggle to capture graph structure. We introduce UniGTE, an instruction-tuned encoder-decoder framework that unifies structural and semantic reasoning. The encoder augments a pretrained autoregressive LLM with learnable alignment tokens and a structure-aware graph-text attention mechanism, enabling it to attend jointly to a tokenized graph and a natural-language task prompt while remaining permutation-invariant to node order. This yields compact, task-aware graph representations. Conditioned solely on these representations, a frozen LLM decoder predicts and reconstructs: it outputs the task answer and simultaneously paraphrases the input graph in natural language. The reconstruction objective regularizes the encoder to preserve structural cues. UniGTE is instruction-tuned on five datasets spanning node-level, edge-level, and graph-level tasks across diverse domains, yet requires no fine-tuning at inference. It achieves new state-of-the-art zero-shot results on node classification, link prediction, graph classification, and graph regression under cross-task and cross-domain settings, demonstrating that tight integration of graph structure with LLM semantics enables robust, transferable graph reasoning.

---

## 85. Deep Learning Accelerated First-Principles Quantum Transport Simulations at Nonequilibrium State

**论文链接:** [http://arxiv.org/abs/2510.16878v1](http://arxiv.org/abs/2510.16878v1)

**作者:** Zili Tang, Xiaoxin Xie, Guanwen Yao, Ligong Zhang, Xiaoyan Liu, Xing Zhang, Liu Fei

**发布时间:** 2025-10-19

**备注:** 32 pages, 5 figures

### GPT解析

### 总结

DeepQT是一种深度学习框架，结合图神经网络和transformer架构，实现了电子结构和输运的多属性预测，无需手动特征工程，同时保持了第一性原理精度并大幅降低计算成本。

### 背景

非平衡格林函数方法结合密度泛函理论(NEGF-DFT)为纳米尺度电子输运模拟提供了严格框架，但其计算成本随系统规模急剧增加。现有的人工智能方法在加速此类模拟时存在局限性，包括缺乏原子分辨率、难以外推到更大系统以及无法同时预测多个属性。

### 目的

开发一种能够加速纳米尺度电子输运模拟的深度学习方法，解决现有AI方法的局限性，实现多属性预测，并能够从小系统外推到更大系统。

### 方法

引入DeepQT框架，结合图神经网络和transformer架构，通过学习NEGF-DFT的关键中间量(平衡哈密顿量和非平衡总势差)来重建哈密顿量，利用电子近视原理实现从小训练系统到更大系统的推广。

### 主要发现

在石墨烯、MoS2和硅二极管(具有不同缺陷和掺杂剂)的基准测试中，DeepQT达到了第一性原理精度，同时将计算成本降低了几个数量级。

### 结论

DeepQT是一个可扩展、可转移的框架，推进了AI辅助的量子输运研究，为下一代纳米电子器件设计提供了强大工具。

### 翻译

非平衡格林函数方法结合密度泛函理论(NEGF-DFT)为纳米尺度电子输运模拟提供了严格的框架，但其计算成本随系统规模急剧增加。最近的人工智能方法试图加速此类模拟，但大多数依赖传统机器学习，缺乏原子分辨率，难以外推到更大系统，并且无法同时预测多个属性。在此我们引入DeepQT，一个深度学习框架，集成了图神经网络和transformer架构，实现了电子结构和输运的多属性预测，无需手动特征工程。通过学习NEGF-DFT的关键中间量——平衡哈密顿量和非平衡总势差，DeepQT重建了平衡和偏置条件下的哈密顿量，从而获得准确的输运预测。利用电子近视原理，DeepQT能够从小训练系统高保真地推广到更大系统。在石墨烯、MoS2和硅二极管(具有不同缺陷和掺杂剂)上的基准测试表明，DeepQT实现了第一性原理精度，同时将计算成本降低了几个数量级。这种可扩展、可转移的框架推动了AI辅助的量子输运发展，为下一代纳米电子器件设计提供了强大工具。


### 论文摘要

The non-equilibrium Green's function method combined with density functional theory (NEGF-DFT) provides a rigorous framework for simulating nanoscale electronic transport, but its computational cost scales steeply with system size. Recent artificial intelligence (AI) approaches have sought to accelerate such simulations, yet most rely on conventional machine learning, lack atomic resolution, struggle to extrapolate to larger systems, and cannot predict multiple properties simultaneously. Here we introduce DeepQT, a deep-learning framework that integrates graph neural networks with transformer architectures to enable multi-property predictions of electronic structure and transport without manual feature engineering. By learning key intermediate quantities of NEGF-DFT, the equilibrium Hamiltonian and the non-equilibrium total potential difference, DeepQT reconstructs Hamiltonians under both equilibrium and bias conditions, yielding accurate transport predictions. Leveraging the principle of electronic nearsightedness, DeepQT generalizes from small training systems to much larger ones with high fidelity. Benchmarks on graphene, MoS2, and silicon diodes with varied defects and dopants show that DeepQT achieves first-principles accuracy while reducing computational cost by orders of magnitude. This scalable, transferable framework advances AI-assisted quantum transport, offering a powerful tool for next-generation nanoelectronic device design.

---

## 86. ProtoMol: Enhancing Molecular Property Prediction via Prototype-Guided Multimodal Learning

**论文链接:** [http://arxiv.org/abs/2510.16824v1](http://arxiv.org/abs/2510.16824v1)

**作者:** Yingxu Wang, Kunyu Zhang, Jiaxin Huang, Nan Yin, Siwei Liu, Eran Segal

**发布时间:** 2025-10-19

### GPT解析

### 总结

ProtoMol是一种原型引导的多模态框架，通过双分支层次编码器和层次双向跨模态注意力机制，实现分子图和文本描述之间的细粒度集成和一致语义对齐，解决了现有方法在跨模态交互和原型空间方面的局限性。

### 背景

多模态分子表示学习通过联合建模分子图和文本描述，整合结构和语义信息，提高药物毒性、生物活性和理化性质的预测准确性和可解释性。

### 目的

解决现有多模态方法的两个关键局限性：层次语义依赖被忽略和缺乏统一的原型空间，实现更稳健的跨模态对齐。

### 方法

ProtoMol采用双分支层次编码器（图神经网络处理分子图，Transformer编码文本），引入层次双向跨模态注意力机制逐层对齐语义特征，并构建共享原型空间引导模态向一致且具有区分性的表示发展。

### 主要发现

在多个基准数据集上，ProtoMol在各种分子性质预测任务中持续优于最先进的基线方法。

### 结论

ProtoMol通过细粒度的跨模态集成和一致的语义对齐，有效提升了分子性质预测的准确性和可靠性。

### 翻译

多模态分子表示学习通过联合建模分子图和它们的文本描述，通过整合结构和语义信息，能够更稳健可靠地预测药物毒性、生物活性和理化性质，从而提高预测准确性和可解释性。然而，现有的多模态方法存在两个关键局限性：(1)它们通常只在最终编码器层执行跨模态交互，从而忽略了层次语义依赖；(2)它们缺乏统一的原型空间来实现模态间的稳健对齐。为了解决这些局限性，我们提出了ProtoMol，一种原型引导的多模态框架，能够实现分子图和文本描述之间的细粒度集成和一致语义对齐。ProtoMol采用双分支层次编码器，利用图神经网络处理结构化分子图，使用Transformer编码非结构化文本，从而生成全面的层次化表示。然后，ProtoMol引入了一种层次双向跨模态注意力机制，逐层对齐跨层的语义特征。此外，还构建了一个共享原型空间，包含可学习的、类别特定的锚点，引导两种模态向连贯且具有区分性的表示发展。在多个基准数据集上的广泛实验表明，在各种分子性质预测任务中，ProtoMol持续优于最先进的基线方法。


### 论文摘要

Multimodal molecular representation learning, which jointly models molecular graphs and their textual descriptions, enhances predictive accuracy and interpretability by enabling more robust and reliable predictions of drug toxicity, bioactivity, and physicochemical properties through the integration of structural and semantic information. However, existing multimodal methods suffer from two key limitations: (1) they typically perform cross-modal interaction only at the final encoder layer, thus overlooking hierarchical semantic dependencies; (2) they lack a unified prototype space for robust alignment between modalities. To address these limitations, we propose ProtoMol, a prototype-guided multimodal framework that enables fine-grained integration and consistent semantic alignment between molecular graphs and textual descriptions. ProtoMol incorporates dual-branch hierarchical encoders, utilizing Graph Neural Networks to process structured molecular graphs and Transformers to encode unstructured texts, resulting in comprehensive layer-wise representations. Then, ProtoMol introduces a layer-wise bidirectional cross-modal attention mechanism that progressively aligns semantic features across layers. Furthermore, a shared prototype space with learnable, class-specific anchors is constructed to guide both modalities toward coherent and discriminative representations. Extensive experiments on multiple benchmark datasets demonstrate that ProtoMol consistently outperforms state-of-the-art baselines across a variety of molecular property prediction tasks.

---

## 87. Graph Neural Network for Unified Electronic and Interatomic Potentials: Strain-tunable Electronic Structures in 2D Materials

**论文链接:** [http://arxiv.org/abs/2510.16605v1](http://arxiv.org/abs/2510.16605v1)

**作者:** Moon-ki Choi, Daniel Palmer, Harley T. Johnson

**发布时间:** 2025-10-18

### GPT解析

### 总结

本文介绍了UEIPNet，一种等变图神经网络，用于预测原子结构的原子间势和紧束缚哈密顿量，实现物理上一致的机械-电子响应耦合建模，具有接近DFT的精度。

### 背景

在原子结构模拟中，需要能够同时准确描述机械和电子响应的方法，传统DFT计算准确但计算成本高，而经典力场方法无法准确描述电子效应。

### 目的

开发一种能够同时预测原子间势和紧束缚哈密顿量的神经网络模型，实现物理一致的机械-电子响应耦合建模，并具有接近DFT的精度。

### 方法

UEIPNet是一种等变图神经网络，使用密度泛函理论计算并结合Wannier投影进行训练，预测节点级别的能量和力作为目标，以及Wannier投影的TB矩阵作为边级别目标，在双层石墨烯和单层MoS2的DFT数据上进行训练。

### 主要发现

在扭曲双层石墨烯中，UEIPNet揭示了层间间距、面内应变和平面外起伏如何驱动孤立平带的形成，并显示调制基底相互作用强度可以在非魔角处产生平带；对于单层MoS2，UEIPNet准确重现了声子色散、应变相关的带隙演化以及非均匀应变下的局部态密度调制。

### 结论

UEIPNet提供了一个通用、高效且可扩展的框架，用于研究大规模原子系统中的变形-电子耦合，桥接了经典原子模拟和电子结构计算。

### 翻译

我们引入了UEIPNet，一种等变图神经网络，专为预测原子结构的原子间势和紧束缚哈密顿量而设计。UEIPNet使用密度泛函理论计算结合Wannier投影进行训练，预测能量和力作为节点级别目标，Wannier投影的TB矩阵作为边级别目标。这实现了物理上一致的机械-电子响应耦合建模，具有接近DFT的精度。在双层石墨烯和单层MoS2的DFT数据上训练后，UEIPNet捕捉了关键的变形-电子效应：在扭曲双层石墨烯中，它揭示了层间间距、面内应变和平面外起伏如何驱动孤立平带的形成，并进一步表明调制基底相互作用强度可以在非魔角处产生平带。对于单层MoS2，UEIPNet准确重现了声子色散、应变相关的带隙演化以及非均匀应变下的局部态密度调制。UEIPNet为研究大规模原子系统中的变形-电子耦合提供了一个通用、高效且可扩展的框架，桥接了经典原子模拟和电子结构计算。


### 论文摘要

We introduce UEIPNet, an equivariant graph neural network designed to predict both interatomic potentials and tight-binding (TB) Hamiltonians for an atomic structure. The UEIPNet is trained using density functional theory calculations followed by Wannier projection to predict energies and forces as node-level targets and Wannier-projected TB matrices as edge-level targets. This enables physically consistent modeling of coupled mechanical electronic responses with near-DFT accuracy. Trained on bilayer graphene and monolayer MoS2 DFT data, UEIPNet captures key deformation-electronic effects: in twisted bilayer graphene, it reveals how interlayer spacing, in-plane strain, and out-of-plane corrugation drive isolated flat-band formation, and further shows that modulating substrate interaction strength can generate flat bands even away from the magic angle. For monolayer MoS2, the UEIPNet accurately reproduces phonon dispersions, strain-dependent band-gap evolution, and local density of states modulations under non-uniform strain. The UEIPNet offers a generalized, efficient, and scalable framework for studying deformation-electronic coupling in large-scale atomistic systems, bridging classical atomistic simulations and electronic-structure calculations.

---

## 88. Symmetry and Generalisation in Neural Approximations of Renormalisation Transformations

**论文链接:** [http://arxiv.org/abs/2510.16591v1](http://arxiv.org/abs/2510.16591v1)

**作者:** Cassidy Ashworth, Pietro Liò, Francesco Caso

**发布时间:** 2025-10-18

### GPT解析

### 总结

本研究探讨了深度学习模型中参数对称性和网络表达能力对泛化行为的影响，特别是在学习实空间重正化群变换时。研究发现对称性约束和表达能力之间存在竞争，过度复杂或受限的模型泛化能力较差。

### 背景

深度学习模型通过多层表示学习结构化数据特征非常成功。将物理对称性编码到模型中可提高困难任务性能，且参数对称性破坏和恢复原则被视为其分层学习动力学的统一机制。

### 目的

评估参数对称性和网络表达能力在神经网络学习实空间重正化群(RG)变换时对泛化行为的作用，使用中心极限定理(CLT)作为测试案例映射。

### 方法

研究简单多层感知器(MLPs)和图神经网络(GNNs)，在不同架构中改变权重对称性和激活函数。通过将CLT重新表述为累积量递归关系并利用既定框架分析MLPs的泛化行为，并验证该框架从MLPs到GNNs的扩展。

### 主要发现

对称性约束和表达能力之间存在竞争，过于复杂或过度受限的模型泛化能力较差。分析揭示了这些复杂模型执行的信息处理过程。

### 结论

这些发现为对称网络的学习动态提供了新见解，揭示了它们在建模结构化物理转换方面的局限性。

### 翻译

深度学习模型已被证明在使用多层表示学习结构化数据的相关特征方面极为成功。将这些模型中的物理对称性编码可以提高困难任务上的性能，最近的工作提出了参数对称性破坏和恢复的原则，作为其分层学习动力学的统一机制。我们评估了参数对称性和网络表达能力在神经网络学习实空间重正化群(RG)变换时的泛化行为中的作用，使用中心极限定理(CLT)作为测试案例映射。我们考虑了简单的多层感知器(MLPs)和图神经网络(GNNs)，并在不同架构中改变权重对称性和激活函数。我们的结果表明对称性约束和表达能力之间存在竞争，过于复杂或过度受限的模型泛化能力较差。我们通过将CLT重新表述为累积量递归关系，并利用既定框架通过MLPs传播累积量，分析性地证明了某些受限MLP架构的这种 poor generalisation 行为。我们还经验性地验证了该框架从MLPs到GNNs的扩展，阐明了这些更复杂模型执行的信息处理过程。这些发现为对称网络的学习动态提供了新的见解，以及它们在建模结构化物理转换方面的局限性。


### 论文摘要

Deep learning models have proven enormously successful at using multiple layers of representation to learn relevant features of structured data. Encoding physical symmetries into these models can improve performance on difficult tasks, and recent work has motivated the principle of parameter symmetry breaking and restoration as a unifying mechanism underlying their hierarchical learning dynamics. We evaluate the role of parameter symmetry and network expressivity in the generalisation behaviour of neural networks when learning a real-space renormalisation group (RG) transformation, using the central limit theorem (CLT) as a test case map. We consider simple multilayer perceptrons (MLPs) and graph neural networks (GNNs), and vary weight symmetries and activation functions across architectures. Our results reveal a competition between symmetry constraints and expressivity, with overly complex or overconstrained models generalising poorly. We analytically demonstrate this poor generalisation behaviour for certain constrained MLP architectures by recasting the CLT as a cumulant recursion relation and making use of an established framework to propagate cumulants through MLPs. We also empirically validate an extension of this framework from MLPs to GNNs, elucidating the internal information processing performed by these more complex models. These findings offer new insight into the learning dynamics of symmetric networks and their limitations in modelling structured physical transformations.

---

## 89. LightGlueStick: a Fast and Robust Glue for Joint Point-Line Matching

**论文链接:** [http://arxiv.org/abs/2510.16438v1](http://arxiv.org/abs/2510.16438v1)

**作者:** Aidyn Ubingazhibov, Rémi Pautrat, Iago Suárez, Shaohui Liu, Marc Pollefeys, Viktor Larsson

**发布时间:** 2025-10-18

**备注:** Accepted at ICCVW 2025

### GPT解析

### 总结

本文提出了一种轻量级的点和线段匹配器LightGlueStick，通过注意力线条消息传递(ALMP)实现高效通信，在基准测试中达到最先进水平。

### 背景

线条和点是互补的局部特征，在SLAM和运动恢复结构等应用中有效。传统方法将点和线匹配视为独立任务，而GlueStick虽通过联合匹配降低计算复杂度，但架构过于复杂难以实时应用。

### 目的

开发一种轻量级的点和线段匹配器，适用于实时应用和边缘设备部署。

### 方法

提出LightGlueStick，引入注意力线条消息传递(ALMP)组件，明确向网络暴露线条连接性，实现节点间高效通信。

### 主要发现

LightGlueStick在不同基准测试中建立了新的最先进水平，同时保持了轻量级架构。

### 结论

LightGlueStick实现了高效且轻量的点和线段匹配，适合实时应用和边缘设备部署。

### 翻译

线条和点是互补的局部特征，它们的组合已被证明在SLAM和运动恢复结构等应用中有效。这些流程的核心是局部特征匹配器，用于在图像之间建立对应关系。传统上，点和线匹配被视为独立任务。最近，GlueStick提出了一种基于GNN的网络，同时处理点和线以建立匹配。虽然单一联合运行降低了整体计算复杂度，但复杂的架构阻碍了实时应用或边缘设备的部署。受点匹配最新进展的启发，我们提出了LightGlueStick，一种用于点和线段的轻量级匹配器。我们架构中的关键新颖组件是注意力线条消息传递(ALMP)，它明确地向网络暴露线条的连接性，允许节点之间进行高效通信。在彻底的实验中，我们表明LightGlueStick在不同基准测试中建立了新的最先进水平。代码可在https://github.com/aubingazhib/LightGlueStick获取。


### 论文摘要

Lines and points are complementary local features, whose combination has proven effective for applications such as SLAM and Structure-from-Motion. The backbone of these pipelines are the local feature matchers, establishing correspondences across images. Traditionally, point and line matching have been treated as independent tasks. Recently, GlueStick proposed a GNN-based network that simultaneously operates on points and lines to establish matches. While running a single joint matching reduced the overall computational complexity, the heavy architecture prevented real-time applications or deployment to edge devices.   Inspired by recent progress in point matching, we propose LightGlueStick, a lightweight matcher for points and line segments. The key novel component in our architecture is the Attentional Line Message Passing (ALMP), which explicitly exposes the connectivity of the lines to the network, allowing for efficient communication between nodes. In thorough experiments we show that LightGlueStick establishes a new state-of-the-art across different benchmarks. The code is available at https://github.com/aubingazhib/LightGlueStick.

---

## 90. PassREfinder-FL: Privacy-Preserving Credential Stuffing Risk Prediction via Graph-Based Federated Learning for Representing Password Reuse between Websites

**论文链接:** [http://arxiv.org/abs/2510.16083v1](http://arxiv.org/abs/2510.16083v1)

**作者:** Jaehan Kim, Minkyoo Song, Minjae Seo, Youngjin Jin, Seungwon Shin, Jinwoo Kim

**发布时间:** 2025-10-17

**备注:** Accepted by Elsevier Expert Systems with Applications

### GPT解析

### 总结

本文提出了一种名为PassREfinder-FL的新框架，用于预测跨网站的凭证填充风险，解决了现有方法在可用性和实际部署方面的问题。

### 背景

凭证填充攻击对经常在多个网站重复使用密码的在线用户造成了重大伤害。先前的研究方法虽然试图检测重复使用密码的用户或识别恶意登录尝试，但通常通过限制密码创建或网站访问影响可用性，且依赖复杂的账户共享机制，阻碍了实际部署。

### 目的

提出PassREfinder-FL框架，预测跨网站的凭证填充风险，解决现有方法的局限性，同时保护用户隐私并提高可用性。

### 方法

引入密码重用关系的概念，将其表示为网站图中的边；使用图神经网络(GNNs)执行链接预测任务，评估网站间的凭证重用风险；整合公共网站信息使方法可扩展到大量网站；采用联邦学习(FL)方法保护用户隐私，避免共享敏感信息。

### 主要发现

在包含22,378个网站的3.6亿个泄露账户的真实数据集上评估，PassREfinder-FL在FL设置中实现了0.9153的F1分数；基于FL的GNN比其他最先进的GNN模型性能提升4-11%；预测结果可用于量化密码重用可能性，作为可操作的风险分数。

### 结论

PassREfinder-FL是一个有效的框架，能够在保护用户隐私的同时准确预测跨网站的凭证填充风险，其预测结果可作为风险分数使用，帮助量化密码重用风险。

### 翻译

凭证填充攻击对经常在多个网站重复使用密码的在线用户造成了重大伤害。虽然先前的研究试图检测有重复使用密码的用户或识别恶意登录尝试，但现有方法通常通过限制密码创建或网站访问来影响可用性，且它们依赖复杂的账户共享机制，阻碍了实际部署。为解决这些局限性，我们提出了PassREfinder-FL，一个预测跨网站凭证填充风险的新框架。我们引入了密码重用关系的概念——定义为用户在网站间重用密码的可能性——并将其表示为网站图中的边。使用图神经网络(GNNs)，我们执行链接预测任务来评估网站之间的凭证重用风险。我们的方法通过整合公共网站信息并将新观察到的网站作为图中的节点链接，可扩展到大量任意网站。为了保护用户隐私，我们使用联邦学习(FL)方法扩展了PassREfinder-FL，消除了跨管理员共享用户敏感信息的需求。在包含22,378个网站的3.6亿个泄露账户的真实世界数据集上的评估显示，PassREfinder-FL在FL设置中实现了0.9153的F1分数。我们进一步通过消融研究验证，基于FL的GNN比其他最先进的GNN模型实现了4-11%的性能提升。最后，我们证明预测结果可用于量化密码重用可能性，作为可操作的风险分数。


### 论文摘要

Credential stuffing attacks have caused significant harm to online users who frequently reuse passwords across multiple websites. While prior research has attempted to detect users with reused passwords or identify malicious login attempts, existing methods often compromise usability by restricting password creation or website access, and their reliance on complex account-sharing mechanisms hinders real-world deployment. To address these limitations, we propose PassREfinder-FL, a novel framework that predicts credential stuffing risks across websites. We introduce the concept of password reuse relations -- defined as the likelihood of users reusing passwords between websites -- and represent them as edges in a website graph. Using graph neural networks (GNNs), we perform a link prediction task to assess credential reuse risk between sites. Our approach scales to a large number of arbitrary websites by incorporating public website information and linking newly observed websites as nodes in the graph. To preserve user privacy, we extend PassREfinder-FL with a federated learning (FL) approach that eliminates the need to share user sensitive information across administrators. Evaluation on a real-world dataset of 360 million breached accounts from 22,378 websites shows that PassREfinder-FL achieves an F1-score of 0.9153 in the FL setting. We further validate that our FL-based GNN achieves a 4-11% performance improvement over other state-of-the-art GNN models through an ablation study. Finally, we demonstrate that the predicted results can be used to quantify password reuse likelihood as actionable risk scores.

---

## 91. Residual Correction Models for AC Optimal Power Flow Using DC Optimal Power Flow Solutions

**论文链接:** [http://arxiv.org/abs/2510.16064v1](http://arxiv.org/abs/2510.16064v1)

**作者:** Muhy Eddin Za'ter, Bri-Mathias Hodge, Kyri Baker

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了一种基于残差学习的电力系统优化方法，通过结合快速直流最优潮流解和图神经网络，解决了交流最优潮流计算效率低下的问题，实现了比传统方法更快的计算速度和更高的准确性。

### 背景

解决非线性交流最优潮流(AC OPF)问题是实时电网运行中的主要计算瓶颈，传统AC OPF求解器计算复杂度高，难以满足实时决策需求。

### 目的

开发一种高效且可扩展的方法，能够快速提供交流可行的最优潮流解，以支持近实时电网运行决策。

### 方法

提出残差学习范式，使用直流最优潮流解作为基线，通过拓扑感知图神经网络结合局部注意力和两级直流特征集成，学习非线性修正项，并采用物理信息损失函数强制执行交流潮流可行性和运行限制。

### 主要发现

在57、118和2000总线系统上的评估表明，与传统AC OPF求解器相比，均方误差降低约25%，可行性误差减少高达3倍，运行时间加速高达13倍。模型在N-1 contingencies情况下保持准确性，并能高效扩展到大型网络。

### 结论

残差学习是线性近似和交流可行最优潮流之间的实用且可扩展的桥梁，能够实现近实时运行决策。

### 翻译

解决非线性交流最优潮流(AC OPF)问题仍然是实时电网运行的主要计算瓶颈。在本文中，我们提出了一种残差学习范式，使用快速的直流最优潮流(DC OPF)解作为基线，并学习仅提供完整AC-OPF解决方案所需的非线性修正。该方法利用了具有局部注意力和两级直流特征集成的拓扑感知图神经网络，使用强制执行交流潮流可行性和运行限制的物理信息损失函数进行训练。在57、118和2000总线系统上的OPFData评估显示，与传统AC OPF求解器相比，均方误差(MSE)降低约25%，可行性误差减少高达3倍，运行时间加速高达13倍。该模型在N-1 contingencies情况下保持准确性，并能高效扩展到大型网络。这些结果表明，残差学习是线性近似和交流可行最优潮流之间实用且可扩展的桥梁，能够实现近实时运行决策。


### 论文摘要

Solving the nonlinear AC optimal power flow (AC OPF) problem remains a major computational bottleneck for real-time grid operations. In this paper, we propose a residual learning paradigm that uses fast DC optimal power flow (DC OPF) solutions as a baseline, and learns only the nonlinear corrections required to provide the full AC-OPF solution. The method utilizes a topology-aware Graph Neural Network with local attention and two-level DC feature integration, trained using a physics-informed loss that enforces AC power-flow feasibility and operational limits. Evaluations on OPFData for 57-, 118-, and 2000-bus systems show around 25% lower MSE, up to 3X reduction in feasibility error, and up to 13X runtime speedup compared to conventional AC OPF solvers. The model maintains accuracy under N-1 contingencies and scales efficiently to large networks. These results demonstrate that residual learning is a practical and scalable bridge between linear approximations and AC-feasible OPF, enabling near real-time operational decision making.

---

## 92. Learning a Generalized Model for Substation Level Voltage Estimation in Distribution Networks

**论文链接:** [http://arxiv.org/abs/2510.16063v1](http://arxiv.org/abs/2510.16063v1)

**作者:** Muhy Eddin Za'ter, Bri-Mathias Hodge

**发布时间:** 2025-10-17

### GPT解析

### 总结

本文提出了一种用于变电站级电压估计的分层图神经网络方法，能够处理配电网中常见的低可观测性水平，并在实验中展现出比其他数据驱动模型更优的性能。

### 背景

配电网中准确的电压估计对实时监控和提高电网可靠性至关重要。随着分布式能源渗透率和配电层电压变化性的增加，稳健的配电网状态估计(DSSE)对保持安全高效运行更加重要。然而，传统DSSE技术在处理稀疏测量和现代馈线规模方面存在困难，限制了它们在大规模网络中的可扩展性。

### 目的

开发一种能够利用电气拓扑和物理特征的分层图神经网络，用于变电站级电压估计，同时对现实配电网中常见的低可观测性水平保持稳健。

### 方法

利用公开的SMART-DS数据集，在多个变电站和DER渗透场景的数千个总线上的模型进行训练和评估。提出了一种分层图神经网络方法，该方法能够有效处理配电网的拓扑结构和物理特性。

### 主要发现

所提出的方法比其他数据驱动模型的RMSE低达2倍，即使在只有1%的测量覆盖率的情况下，也能保持高精度。

### 结论

研究结果突出了图神经网络在实现可扩展、可重现和数据驱动的配电系统电压监控方面的潜力。

### 翻译

配电网中的准确电压估计对实时监控和提高电网可靠性至关重要。随着分布式能源渗透率和配电层电压变化性的增加，稳健的配电网状态估计(DSSE)对保持安全高效运行变得更加必要。然而，传统的DSSE技术在处理稀疏测量和现代馈线规模方面存在困难，限制了它们在大规模网络中的可扩展性。本文提出了一种用于变电站级电压估计的分层图神经网络，它利用电气拓扑和物理特征，同时对现实配电网中常见的低可观测性水平保持稳健。利用公开的SMART-DS数据集，该模型在多个变电站和DER渗透场景的数千个总线上进行了训练和评估。全面的实验表明，所提出的方法比其他数据驱动模型的RMSE低达2倍，并且在只有1%的测量覆盖率的情况下仍能保持高精度。这些结果突出了图神经网络在实现可扩展、可重现和数据驱动的配电系统电压监控方面的潜力。


### 论文摘要

Accurate voltage estimation in distribution networks is critical for real-time monitoring and increasing the reliability of the grid. As DER penetration and distribution level voltage variability increase, robust distribution system state estimation (DSSE) has become more essential to maintain safe and efficient operations. Traditional DSSE techniques, however, struggle with sparse measurements and the scale of modern feeders, limiting their scalability to large networks. This paper presents a hierarchical graph neural network for substation-level voltage estimation that exploits both electrical topology and physical features, while remaining robust to the low observability levels common to real-world distribution networks. Leveraging the public SMART-DS datasets, the model is trained and evaluated on thousands of buses across multiple substations and DER penetration scenarios. Comprehensive experiments demonstrate that the proposed method achieves up to 2 times lower RMSE than alternative data-driven models, and maintains high accuracy with as little as 1\% measurement coverage. The results highlight the potential of GNNs to enable scalable, reproducible, and data-driven voltage monitoring for distribution systems.

---

## 93. OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs

**论文链接:** [http://arxiv.org/abs/2510.15188v2](http://arxiv.org/abs/2510.15188v2)

**作者:** Ahmed Aly, Essam Mansour, Amr Youssef

**发布时间:** 2025-10-16

**备注:** This is the authors' extended version of the paper accepted for  publication at the ACM SIGSAC Conference on Computer and Communications  Security (CCS 2025). The final published version is available at  https://doi.org/10.1145/3719027.3765219

### GPT解析

### 总结

本研究提出了OCR-APT系统，用于高级持续性威胁(APTs)的检测和攻击故事重建，通过结合图神经网络和大型语言模型，实现了更准确的异常检测和更具可解释性的攻击报告。

### 背景

高级持续性威胁(APTs)是一种隐蔽的网络攻击，通常能逃避系统级审计日志中的检测。现有的图异常检测系统存在高误报率和粗粒度警报问题，且依赖于文件路径或IP地址等节点属性，导致虚假关联，降低了检测的稳健性和可靠性。

### 目的

开发一个能够生成准确、类人叙述整个攻击的系统，帮助安全分析师完全理解攻击的进展和影响，提供更可靠、更具可解释性的APT检测方案。

### 方法

OCR-APT系统采用图神经网络(GNNs)进行子图异常检测，学习节点周围的行为模式而非脆弱的属性。然后使用大型语言模型(LLMs)迭代检测到的子图，重建多阶段攻击故事，并在每个阶段进行验证以减少幻觉并确保可解释的最终报告。

### 主要发现

在DARPA TC3、OpTC和NODLINK数据集上的评估表明，OCR-APT在检测准确性和警报可解释性方面优于最先进的系统。OCR-APT能够重建类人报告，全面捕获攻击故事。

### 结论

OCR-APT通过结合图神经网络和大型语言模型，解决了现有APT检测系统的局限性，提供了更准确、更可靠且更具可解释性的攻击检测和报告方案。

### 翻译

高级持续性威胁(APTs)是一种隐蔽的网络攻击，通常能逃避系统级审计日志中的检测。来源图将这些日志建模为连接的实体和事件，揭示了线性日志表示中遗漏的关系。现有系统对这些图应用异常检测，但常常存在高误报率和粗粒度警报的问题。它们对文件路径或IP等节点属性的依赖导致虚假关联，降低了检测的稳健性和可靠性。为了完全理解攻击的进展和影响，安全分析师需要能够生成整个攻击的准确、类人叙述的系统。为了解决这些挑战，我们介绍了OCR-APT，一个用于APT检测和重建类人攻击故事的系统。OCR-APT使用图神经网络(GNNs)进行子图异常检测，学习节点周围的行为模式而非文件路径或IP等脆弱属性。这种方法带来了更稳健的异常检测。然后使用大型语言模型(LLMs)迭代检测到的子图，重建多阶段攻击故事。每个阶段在继续之前都经过验证，减少了幻觉并确保了可解释的最终报告。我们在DARPA TC3、OpTC和NODLINK数据集上的评估表明，OCR-APT在检测准确性和警报可解释性方面优于最先进的系统。此外，OCR-APT重建的类人报告能够全面捕获攻击故事。


### 论文摘要

Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.

---

## 94. RoBCtrl: Attacking GNN-Based Social Bot Detectors via Reinforced Manipulation of Bots Control Interaction

**论文链接:** [http://arxiv.org/abs/2510.16035v1](http://arxiv.org/abs/2510.16035v1)

**作者:** Yingguang Yang, Xianghua Zeng, Qi Wu, Hao Peng, Yutong Xia, Hao Liu, Bin Chong, Philip S. Yu

**发布时间:** 2025-10-16

**备注:** 27 pages, 10 figures

### GPT解析

### 总结

该论文提出了首个针对基于图神经网络(GNN)的社交机器人检测器的对抗性多智能体强化学习框架(RoBCtrl)，通过扩散模型生成高保真机器人账户和多智能体强化学习优化攻击策略，实验证明该框架能有效削弱基于GNN的检测器性能。

### 背景

社交网络已成为个人获取实时信息的关键来源。社交机器人在这些平台上的影响引起了研究者的广泛关注，导致了许多检测技术的发展。然而，这些检测方法的脆弱性和鲁棒性仍未得到充分探索。由于对社交代理的控制有限、机器人检测器的黑盒特性以及机器人的异构性问题，现有的基于图神经网络(GNN)的方法无法直接应用。

### 目的

为应对现有GNN-based社交机器人检测方法面临的挑战，提出首个针对基于GNN的社交机器人检测器的对抗性多智能体强化学习框架，用于社交机器人控制攻击(RoBCtrl)。

### 方法

使用扩散模型通过微小修改重构现有账户数据来生成高保真机器人账户；采用多智能体强化学习(MARL)方法模拟机器人的对抗行为；根据影响力和预算对社交账户进行分类；使用不同的智能体控制各类别的机器人账户，通过强化学习优化附着策略；设计基于结构熵的分层状态抽象以加速强化学习过程。

### 主要发现

据我们所知，这是首次应用扩散模型有效模拟 evolving social bots 的行为；在社交机器人检测数据集上的大量实验表明，该框架可以有效削弱基于GNN的检测器的性能。

### 结论

提出的RoBCtrl框架能够有效对抗基于GNN的社交机器人检测器，通过生成高保真机器人账户和智能化的攻击策略实现了这一目标。

### 翻译

社交网络已成为个人获取实时信息的关键来源。社交机器人在这些平台上的影响引起了研究者的广泛关注，导致了许多检测技术的发展。然而，这些检测方法的脆弱性和鲁棒性仍未得到充分探索。由于对社交代理的控制有限、机器人检测器的黑盒特性以及机器人的异构性问题，现有的基于图神经网络(GNN)的方法无法直接应用。为应对这些挑战，本文提出了首个针对基于GNN的社交机器人检测器的对抗性多智能体强化学习框架，用于社交机器人控制攻击(RoBCtrl)。具体而言，我们使用扩散模型通过微小修改重构现有账户数据来生成高保真机器人账户，从而逃避社交平台的检测。据我们所知，这是首次应用扩散模型有效模拟 evolving social bots 的行为。随后，我们采用多智能体强化学习(MARL)方法模拟机器人的对抗行为。我们根据影响力和预算对社交账户进行分类，然后使用不同的智能体控制各类别的机器人账户，通过强化学习优化附着策略。此外，我们还设计了一种基于结构熵的分层状态抽象以加速强化学习。在社交机器人检测数据集上的大量实验表明，我们的框架可以有效削弱基于GNN的检测器的性能。


### 论文摘要

Social networks have become a crucial source of real-time information for individuals. The influence of social bots within these platforms has garnered considerable attention from researchers, leading to the development of numerous detection technologies. However, the vulnerability and robustness of these detection methods is still underexplored. Existing Graph Neural Network (GNN)-based methods cannot be directly applied due to the issues of limited control over social agents, the black-box nature of bot detectors, and the heterogeneity of bots. To address these challenges, this paper proposes the first adversarial multi-agent Reinforcement learning framework for social Bot control attacks (RoBCtrl) targeting GNN-based social bot detectors. Specifically, we use a diffusion model to generate high-fidelity bot accounts by reconstructing existing account data with minor modifications, thereby evading detection on social platforms. To the best of our knowledge, this is the first application of diffusion models to mimic the behavior of evolving social bots effectively. We then employ a Multi-Agent Reinforcement Learning (MARL) method to simulate bots adversarial behavior. We categorize social accounts based on their influence and budget. Different agents are then employed to control bot accounts across various categories, optimizing the attachment strategy through reinforcement learning. Additionally, a hierarchical state abstraction based on structural entropy is designed to accelerate the reinforcement learning. Extensive experiments on social bot detection datasets demonstrate that our framework can effectively undermine the performance of GNN-based detectors.

---

## 95. Plasma Shape Control via Zero-shot Generative Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.17531v1](http://arxiv.org/abs/2510.17531v1)

**作者:** Niannian Wu, Rongpeng Li, Zongyu Yang, Yong Xiao, Ning Wei, Yihang Chen, Bo Li, Zhifeng Zhao, Wulyu Zhong

**发布时间:** 2025-10-20

### GPT解析

### 总结

本文提出了一种结合生成对抗模仿学习与希尔伯特空间表征学习的新框架，从历史PID控制数据中开发通用的零样本控制策略，用于等离子体形状控制，无需任务特定微调即可在各种场景下精确稳定地跟踪参考轨迹。

### 背景

传统PID控制器在等离子体形状控制方面的适应性有限，而特定任务的强化学习方法存在泛化能力不足和需要重复重新训练的问题。

### 目的

提出一个新框架，从大规模历史PID控制放电数据中开发通用的零样本控制策略，克服传统方法和特定任务强化学习方法的局限性。

### 方法

结合生成对抗模仿学习(GAIL)与希尔伯特空间表征学习，实现双重目标：模仿PID数据的稳定操作风格，构建几何结构的潜在空间以实现高效的目标导向控制。

### 主要发现

基础策略可以零样本方式部署，无需任务特定的微调；在HL-3托卡马克模拟器上的评估表明，该策略能够精确且稳定地跟踪各种等离子体场景下关键形状参数的参考轨迹。

### 结论

这项工作为未来聚变反应堆开发高度灵活和数据高效的智能控制系统提供了可行途径。

### 翻译

传统PID控制器在等离子体形状控制方面的适应性有限，而特定任务的强化学习方法存在泛化能力有限和需要重复重新训练的问题。为克服这些挑战，本文提出了一种新框架，用于从大规模历史PID控制放电数据中开发通用的零样本控制策略。我们的方法将生成对抗模仿学习(GAIL)与希尔伯特空间表征学习相结合，以实现双重目标：模仿PID数据的稳定操作风格，构建几何结构的潜在空间以实现高效的目标导向控制。所得基础策略可以零样本方式部署，用于各种轨迹跟踪任务，无需任务特定的微调。在HL-3托卡马克模拟器上的评估表明，该策略在多种等离子体场景下能够精确且稳定地跟踪关键形状参数的参考轨迹。这项工作为未来聚变反应堆开发高度灵活和数据高效的智能控制系统提供了可行途径。


### 论文摘要

Traditional PID controllers have limited adaptability for plasma shape control, and task-specific reinforcement learning (RL) methods suffer from limited generalization and the need for repetitive retraining. To overcome these challenges, this paper proposes a novel framework for developing a versatile, zero-shot control policy from a large-scale offline dataset of historical PID-controlled discharges. Our approach synergistically combines Generative Adversarial Imitation Learning (GAIL) with Hilbert space representation learning to achieve dual objectives: mimicking the stable operational style of the PID data and constructing a geometrically structured latent space for efficient, goal-directed control. The resulting foundation policy can be deployed for diverse trajectory tracking tasks in a zero-shot manner without any task-specific fine-tuning. Evaluations on the HL-3 tokamak simulator demonstrate that the policy excels at precisely and stably tracking reference trajectories for key shape parameters across a range of plasma scenarios. This work presents a viable pathway toward developing highly flexible and data-efficient intelligent control systems for future fusion reactors.

---

## 96. DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.17489v1](http://arxiv.org/abs/2510.17489v1)

**作者:** Yongxin He, Shan Zhang, Yixuan Cao, Lei Ma, Ping Luo

**发布时间:** 2025-10-20

**备注:** To appear in NeurIPS 2025

### GPT解析

### 总结

DETree是一种新型AI参与文本检测方法，通过层次亲和树结构建模不同文本生成过程间的关系，并引入专门损失函数对齐文本表示。研究团队开发了RealBench基准数据集，显著提升了混合文本检测性能和分布外场景的鲁棒性。

### 背景

检测AI参与的文本对打击错误信息、剽窃和学术不端行为至关重要。AI文本生成涉及多种协作过程（如AI生成文本由人类编辑、人类文本由AI编辑、AI文本由其他AI优化），不同过程生成的文本具有复杂特征，给检测带来巨大挑战。

### 目的

开发更有效的AI参与文本检测方法，准确识别不同人-AI协作过程生成的文本，提高检测性能和鲁棒性。

### 方法

提出DETree方法，将不同文本生成过程间的关系建模为层次亲和树结构，并引入专门损失函数使文本表示与该树结构对齐。同时开发了RealBench基准数据集，自动整合各种人-AI协作过程产生的混合文本。

### 主要发现

不同过程生成的文本表示表现出内在聚类关系；DETree方法在混合文本检测任务中提高了性能；显著增强了在分布外场景中的鲁棒性和泛化能力，特别是在少样本学习条件下。

### 结论

基于训练的方法在分布外设置中具有潜力，DETree为AI参与文本检测提供了有效解决方案。

### 翻译

检测AI参与的文本对于打击错误信息、剽窃和学术不端行为至关重要。然而，AI文本生成包括多样的协作过程（AI生成文本由人类编辑、人类编写文本由AI编辑、AI生成文本由其他AI优化），其中可能涉及各种甚至新的LLM。这些不同过程生成的文本表现出复杂特征，给检测带来巨大挑战。当前方法对这些过程建模过于简单，主要采用二元分类（纯人类vs AI参与）或多分类（将人-AI协作视为新类别）。我们观察到，通过不同过程生成的文本表示表现出内在的聚类关系。因此，我们提出了DETree，一种新方法，将不同过程之间的关系建模为层次亲和树结构，并引入专门的损失函数使文本表示与此树对齐。为此，我们开发了RealBench，一个全面的基准数据集，自动整合通过各种人-AI协作过程产生的混合文本。我们的方法在混合文本检测任务中提高了性能，显著增强了在分布外场景中的鲁棒性和泛化能力，特别是在少样本学习条件下，进一步证明了基于训练的方法在OOD设置中的潜力。我们的代码和数据集可在https://github.com/heyongxin233/DETree获取。


### 论文摘要

Detecting AI-involved text is essential for combating misinformation, plagiarism, and academic misconduct. However, AI text generation includes diverse collaborative processes (AI-written text edited by humans, human-written text edited by AI, and AI-generated text refined by other AI), where various or even new LLMs could be involved. Texts generated through these varied processes exhibit complex characteristics, presenting significant challenges for detection. Current methods model these processes rather crudely, primarily employing binary classification (purely human vs. AI-involved) or multi-classification (treating human-AI collaboration as a new class). We observe that representations of texts generated through different processes exhibit inherent clustering relationships. Therefore, we propose DETree, a novel approach that models the relationships among different processes as a Hierarchical Affinity Tree structure, and introduces a specialized loss function that aligns text representations with this tree. To facilitate this learning, we developed RealBench, a comprehensive benchmark dataset that automatically incorporates a wide spectrum of hybrid texts produced through various human-AI collaboration processes. Our method improves performance in hybrid text detection tasks and significantly enhances robustness and generalization in out-of-distribution scenarios, particularly in few-shot learning conditions, further demonstrating the promise of training-based approaches in OOD settings. Our code and dataset are available at https://github.com/heyongxin233/DETree.

---

## 97. Nearest-Class Mean and Logits Agreement for Wildlife Open-Set Recognition

**论文链接:** [http://arxiv.org/abs/2510.17338v1](http://arxiv.org/abs/2510.17338v1)

**作者:** Jiahao Huo, Mufhumudzi Muthivhi, Terence L. van Zyl, Fredrik Gustafsson

**发布时间:** 2025-10-20

### GPT解析

### 总结

本研究提出了一种后处理开放集识别方法，通过测量模型特征与预测logit之间的一致性来识别未知类别，无需重新训练预训练模型，在两个数据集上都取得了优异的性能。

### 背景

当前最先进的野生动物分类模型在封闭世界设置下训练，遇到未知类别时仍过于自信预测。

### 目的

开发一种开放集识别方法，能够分类已知类别同时拒绝未知样本，无需重新训练预训练模型。

### 方法

提出基于输入到最近类别均值(NCM)距离的概率分布，并与logit空间的softmax概率比较，测量NCM与分类头之间的一致性。

### 主要发现

该方法在两个评估数据集上排名前三，性能一致；在非洲和瑞典动物数据集上分别实现了93.41和95.35的AUROC。

### 结论

该方法作为后处理技术应用，无需重新训练预训练模型，在开放集识别任务中表现优异。

### 翻译

当前最先进的野生动物分类模型是在封闭世界设置下训练的。当遇到未知类别时，它们对自己的预测仍然过于自信。开放集识别(OSR)旨在分类已知类别同时拒绝未知样本。本研究提出了一种后处理OSR方法，用于测量模型特征和预测logit之间的一致性。我们提出了一种基于输入到其最近类别均值(NCM)距离的概率分布。然后将基于NCM的分布与logit空间中的softmax概率进行比较，以测量NCM和分类头之间的一致性。所提出的策略在两个评估数据集中排名前三，且在两个数据集上表现一致。相比之下，当前最先进的方法在单个数据集上表现出色。我们在非洲和瑞典动物数据集上分别实现了93.41和95.35的AUROC。代码可在https://github.com/Applied-Representation-Learning-Lab/OSR找到。


### 论文摘要

Current state-of-the-art Wildlife classification models are trained under the closed world setting. When exposed to unknown classes, they remain overconfident in their predictions. Open-set Recognition (OSR) aims to classify known classes while rejecting unknown samples. Several OSR methods have been proposed to model the closed-set distribution by observing the feature, logit, or softmax probability space. A significant drawback of many existing approaches is the requirement to retrain the pre-trained classification model with the OSR-specific strategy. This study contributes a post-processing OSR method that measures the agreement between the models' features and predicted logits. We propose a probability distribution based on an input's distance to its Nearest Class Mean (NCM). The NCM-based distribution is then compared with the softmax probabilities from the logit space to measure agreement between the NCM and the classification head. Our proposed strategy ranks within the top three on two evaluated datasets, showing consistent performance across the two datasets. In contrast, current state-of-the-art methods excel on a single dataset. We achieve an AUROC of 93.41 and 95.35 for African and Swedish animals. The code can be found https://github.com/Applied-Representation-Learning-Lab/OSR.

---

## 98. 论文ID: 2510.17289v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.17289v1.json'

---

## 99. HIDISC: A Hyperbolic Framework for Domain Generalization with Generalized Category Discovery

**论文链接:** [http://arxiv.org/abs/2510.17188v1](http://arxiv.org/abs/2510.17188v1)

**作者:** Vaibhav Rathore, Divyam Gupta, Biplab Banerjee

**发布时间:** 2025-10-20

**备注:** Accpeted at NeurIPS (2025) Main Conference

### GPT解析

### 总结

HIDISC是一种双曲表示学习框架，用于解决领域泛化广义类别发现问题，无需片段模拟即可实现领域和类别级别的泛化。

### 背景

广义类别发现(GCD)旨在对测试样本进行分类，分为训练期间可见的类别或新类别，无需标签监督。现有GCD方法假设训练期间可以同时访问有标签和无标签数据，且来自同一领域，限制了其在涉及分布偏移的开放世界场景中的应用。DG-GCD要求模型泛化到包含新类别的未见领域，且在训练期间不访问目标领域数据。

### 目的

开发一种高效的方法，使模型能够泛化到未见领域中的新类别，同时避免现有方法的高计算成本和错误累积问题。

### 方法

HIDISC框架使用GPT引导的扩散增强源领域，引入Tangent CutMix在切线空间中合成伪新样本，采用统一损失函数(惩罚Busemann对齐、混合双曲对比正则化和自适应异常值排斥)，并使用可学习的曲率参数使几何结构适应数据集复杂性。

### 主要发现

HIDISC在PACS、Office-Home和DomainNet数据集上实现了最先进的结果，一致性地优于现有的欧几里得和双曲(DG)-GCD基线。

### 结论

HIDISC通过双曲表示学习框架有效解决了领域泛化广义类别发现问题，无需片段模拟即可实现领域和类别级别的泛化，同时保持了高效率。

### 翻译

广义类别发现(GCD)旨在对测试样本进行分类，分为训练期间可见的类别或新类别，无需依赖标签监督。大多数现有GCD方法假设在训练期间可以同时访问有标签和无标签数据，并且来自同一领域，这限制了其在涉及分布偏移的开放世界场景中的应用。带有GCD的领域泛化(DG-GCD)通过要求模型泛化到包含新类别的未见领域，且在训练期间不访问目标领域数据，从而消除了这一限制。唯一的现有DG-GCD方法DG2CD-Net依赖于多合成领域的片段训练和任务向量聚合，导致高计算成本和错误累积。我们提出了HIDISC，一种双曲表示学习框架，无需片段模拟即可实现领域和类别级别的泛化。为了使模型接触到最小但多样化的领域变化，我们使用GPT引导的扩散增强源领域，避免过拟合并保持效率。为了构建表示空间，我们引入了Tangent CutMix，这是一种曲率感知的插值方法，在切线空间中合成伪新样本，保持流形一致性。统一的损失函数(结合惩罚Busemann对齐、混合双曲对比正则化和自适应异常值排斥)促进了紧凑、语义结构化的嵌入。可学习的曲率参数进一步使几何结构适应数据集的复杂性。HIDISC在PACS、Office-Home和DomainNet上实现了最先进的结果，一致性地优于现有的欧几里得和双曲(DG)-GCD基线。


### 论文摘要

Generalized Category Discovery (GCD) aims to classify test-time samples into either seen categories** -- available during training -- or novel ones, without relying on label supervision. Most existing GCD methods assume simultaneous access to labeled and unlabeled data during training and arising from the same domain, limiting applicability in open-world scenarios involving distribution shifts. Domain Generalization with GCD (DG-GCD) lifts this constraint by requiring models to generalize to unseen domains containing novel categories, without accessing targetdomain data during training. The only prior DG-GCD method, DG2CD-Net, relies on episodic training with multiple synthetic domains and task vector aggregation, incurring high computational cost and error accumulation. We propose HIDISC, a hyperbolic representation learning framework that achieves domain and category-level generalization without episodic simulation. To expose the model to minimal but diverse domain variations, we augment the source domain using GPT-guided diffusion, avoiding overfitting while maintaining efficiency. To structure the representation space, we introduce Tangent CutMix, a curvature-aware interpolation that synthesizes pseudo-novel samples in tangent space, preserving manifold consistency. A unified loss -- combining penalized Busemann alignment, hybrid hyperbolic contrastive regularization, and adaptive outlier repulsion -- **facilitates compact, semantically structured embeddings. A learnable curvature parameter further adapts the geometry to dataset complexity. HIDISC achieves state-of-the-art results on PACS , Office-Home , and DomainNet, consistently outperforming the existing Euclidean and hyperbolic (DG)-GCD baselines.

---

## 100. DFNN: A Deep Fréchet Neural Network Framework for Learning Metric-Space-Valued Responses

**论文链接:** [http://arxiv.org/abs/2510.17072v1](http://arxiv.org/abs/2510.17072v1)

**作者:** Kyum Kim, Yaqing Chen, Paromita Dubey

**发布时间:** 2025-10-20

### GPT解析

### 总结

论文提出了深度Fréchet神经网络（DFNNs）框架，用于从欧几里得预测变量预测非欧几里得响应变量，并通过理论证明和实证研究展示了其优越性。

### 背景

非欧几里得响应变量（如概率分布、网络、对称正定矩阵和组合）的回归在现代应用中变得越来越重要。

### 目的

提出一种端到端的深度学习框架，用于预测被视为度量空间中随机对象的非欧几里得响应变量。

### 方法

深度Fréchet神经网络（DFNNs），利用深度神经网络的表示学习能力来近似给定预测变量的响应的条件Fréchet均值，通过最小化Fréchet风险实现。

### 主要发现

建立了DFNNs的通用近似定理，将神经网络近似理论推进到一般度量空间值响应，无需模型假设或局部平滑；在多种应用场景中，DFNNs始终优于现有方法。

### 结论

DFNNs是一种高度灵活的框架，能够适应不同的度量和高维预测变量，为非欧几里得响应变量的回归提供了有效解决方案。

### 翻译

回归非欧几里得响应变量——例如概率分布、网络、对称正定矩阵和组合——在现代应用中已变得越来越重要。在本文中，我们提出了深度Fréchet神经网络（DFNNs），一个用于从欧几里得预测变量预测非欧几里得响应变量的端到端深度学习框架——这些响应被视为度量空间中的随机对象。我们的方法利用深度神经网络的表示学习能力，通过最小化Fréchet风险来近似给定预测变量的响应的条件Fréchet均值——这是条件期望的度量空间类比。该框架非常灵活，能够适应不同的度量和高维预测变量。我们建立了DFNNs的通用近似定理，将神经网络近似理论的最先进水平推进到一般度量空间值响应，无需做出模型假设或依赖局部平滑。在合成分布和网络值响应以及预测就业职业构成的真实世界应用中的实证研究表明，DFNNs始终优于现有方法。


### 论文摘要

Regression with non-Euclidean responses -- e.g., probability distributions, networks, symmetric positive-definite matrices, and compositions -- has become increasingly important in modern applications. In this paper, we propose deep Fr\'echet neural networks (DFNNs), an end-to-end deep learning framework for predicting non-Euclidean responses -- which are considered as random objects in a metric space -- from Euclidean predictors. Our method leverages the representation-learning power of deep neural networks (DNNs) to the task of approximating conditional Fr\'echet means of the response given the predictors, the metric-space analogue of conditional expectations, by minimizing a Fr\'echet risk. The framework is highly flexible, accommodating diverse metrics and high-dimensional predictors. We establish a universal approximation theorem for DFNNs, advancing the state-of-the-art of neural network approximation theory to general metric-space-valued responses without making model assumptions or relying on local smoothing. Empirical studies on synthetic distributional and network-valued responses, as well as a real-world application to predicting employment occupational compositions, demonstrate that DFNNs consistently outperform existing methods.

---

## 101. Diverse Influence Component Analysis: A Geometric Approach to Nonlinear Mixture Identifiability

**论文链接:** [http://arxiv.org/abs/2510.17040v1](http://arxiv.org/abs/2510.17040v1)

**作者:** Hoang-Son Nguyen, Xiao Fu

**发布时间:** 2025-10-19

**备注:** 30 pages, 3 figures

### GPT解析

### 总结

该论文提出了Diverse Influence Component Analysis (DICA)框架，利用混合函数Jacobian的凸几何特性，通过Jacobian Volume Maximization (J-VolMax)准则实现潜在成分识别，无需依赖辅助信息、潜在成分独立性或Jacobian稀疏假设。

### 背景

从未知非线性混合中识别潜在成分是机器学习的基础挑战，应用于解纠缠表示学习和因果推断等领域。先前工作表明辅助信号可支持条件独立潜在成分的可识别性，而更新的方法则通过结构假设（如混合函数Jacobian的稀疏性）来放宽要求。

### 目的

引入DICA框架，利用混合函数Jacobian的凸几何特性，开发一种新的潜在成分识别方法。

### 方法

提出Jacobian Volume Maximization (J-VolMax)准则，通过鼓励潜在成分对观察变量的影响多样性来实现潜在成分识别。

### 主要发现

在合理条件下，DICA方法无需依赖辅助信息、潜在成分独立性或Jacobian稀疏假设即可实现潜在成分的可识别性。

### 结论

这些结果扩展了可识别性分析的范围，为现有方法提供了互补的视角。

### 翻译

从未知非线性混合中识别潜在成分是机器学习中的一个基础性挑战，应用于解纠缠表示学习和因果推断等任务。先前在非线性独立成分分析方面的工作表明，辅助信号（如弱监督）可以支持条件独立潜在成分的可识别性。更新的方法探索结构假设，例如混合函数的Jacobian稀疏性，以放宽这些要求。在这项工作中，我们引入了Diverse Influence Component Analysis (DICA)框架，利用混合函数Jacobian的凸几何特性。我们提出了Jacobian Volume Maximization (J-VolMax)准则，通过鼓励潜在成分对观察变量的影响多样性来实现潜在成分识别。在合理条件下，这种方法无需依赖辅助信息、潜在成分独立性或Jacobian稀疏假设即可实现可识别性。这些结果扩展了可识别性分析的范围，为现有方法提供了互补的视角。


### 论文摘要

Latent component identification from unknown nonlinear mixtures is a foundational challenge in machine learning, with applications in tasks such as disentangled representation learning and causal inference. Prior work in nonlinear independent component analysis (nICA) has shown that auxiliary signals -- such as weak supervision -- can support identifiability of conditionally independent latent components. More recent approaches explore structural assumptions, e.g., sparsity in the Jacobian of the mixing function, to relax such requirements. In this work, we introduce Diverse Influence Component Analysis (DICA), a framework that exploits the convex geometry of the mixing function's Jacobian. We propose a Jacobian Volume Maximization (J-VolMax) criterion, which enables latent component identification by encouraging diversity in their influence on the observed variables. Under reasonable conditions, this approach achieves identifiability without relying on auxiliary information, latent component independence, or Jacobian sparsity assumptions. These results extend the scope of identifiability analysis and offer a complementary perspective to existing methods.

---

## 102. 论文ID: 2510.17034v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2510.17034v1.json'

---

## 103. CARE: Contrastive Alignment for ADL Recognition from Event-Triggered Sensor Streams

**论文链接:** [http://arxiv.org/abs/2510.16988v1](http://arxiv.org/abs/2510.16988v1)

**作者:** Junhao Zhao, Zishuai Liu, Ruili Fang, Jin Lu, Linghan Zhang, Fei Dou

**发布时间:** 2025-10-19

### GPT解析

### 总结

本文提出了CARE框架，通过序列-图像对比对齐方法解决了日常生活活动识别中的表征局限性，实现了高性能和鲁棒性。

### 背景

从事件触发式环境传感器识别日常生活活动(ADLs)是环境辅助生活的关键任务，但现有方法存在表征层面的局限性。基于序列的方法对噪声敏感且缺乏空间感知，而基于图像的方法压缩了时间动态并扭曲了传感器布局。简单融合无法充分利用这两种方法的互补优势。

### 目的

开发一个端到端框架，通过联合优化表征学习和分类，确保跨表征对齐和任务特定判别性，从而提高ADL识别的准确性和鲁棒性。

### 方法

提出CARE(从事件触发式传感器流进行ADL识别的对比对齐)框架，集成时间感知、噪声鲁棒的序列编码与空间感知、频率敏感的图像表征，并采用联合对比-分类目标进行端到端学习。

### 主要发现

在三个CASAS数据集上评估，CARE实现了最先进的性能：Milan上89.8%，Cairo上88.9%，Kyoto7上73.3%。同时，该方法展示了对传感器故障和布局变化的鲁棒性。

### 结论

CARE框架在智能家居环境中可靠的ADL识别具有显著潜力，其性能和鲁棒性证明了该方法的有效性。

### 翻译

从事件触发式环境传感器识别日常生活活动(ADLs)是环境辅助生活(AAL)中的关键任务，然而现有方法仍受表征层面限制。基于序列的方法保留了传感器激活的时间顺序，但对噪声敏感且缺乏空间感知，而基于图像的方法捕捉全局模式和隐含的空间相关性，但压缩了细粒度时间动态并扭曲了传感器布局。简单融合(如特征连接)无法强制序列和图像表征视图之间的对齐，未能充分利用它们的互补优势。我们提出了CARE(从事件触发式传感器流进行ADL识别的对比对齐)，一个通过序列-图像对比对齐(SICA)和交叉熵联合优化表征学习的端到端框架，确保跨表征对齐和任务特定判别性。CARE集成(i)时间感知、噪声鲁棒的序列编码，(ii)空间感知和频率敏感的图像表征，并采用(iii)联合对比-分类目标进行对齐且具有判别性的嵌入的端到端学习。在三个CASAS数据集上评估，CARE实现了最先进的性能(Milan上89.8%，Cairo上88.9%，Kyoto7上73.3%)，并展示了对传感器故障和布局变化的鲁棒性，突显了其在智能家居中可靠ADL识别的潜力。


### 论文摘要

The recognition of Activities of Daily Living (ADLs) from event-triggered ambient sensors is an essential task in Ambient Assisted Living, yet existing methods remain constrained by representation-level limitations. Sequence-based approaches preserve temporal order of sensor activations but are sensitive to noise and lack spatial awareness, while image-based approaches capture global patterns and implicit spatial correlations but compress fine-grained temporal dynamics and distort sensor layouts. Naive fusion (e.g., feature concatenation) fail to enforce alignment between sequence- and image-based representation views, underutilizing their complementary strengths. We propose Contrastive Alignment for ADL Recognition from Event-Triggered Sensor Streams (CARE), an end-to-end framework that jointly optimizes representation learning via Sequence-Image Contrastive Alignment (SICA) and classification via cross-entropy, ensuring both cross-representation alignment and task-specific discriminability. CARE integrates (i) time-aware, noise-resilient sequence encoding with (ii) spatially-informed and frequency-sensitive image representations, and employs (iii) a joint contrastive-classification objective for end-to-end learning of aligned and discriminative embeddings. Evaluated on three CASAS datasets, CARE achieves state-of-the-art performance (89.8% on Milan, 88.9% on Cairo, and 73.3% on Kyoto7) and demonstrates robustness to sensor malfunctions and layout variability, highlighting its potential for reliable ADL recognition in smart homes.

---

## 104. Domain Generalizable Continual Learning

**论文链接:** [http://arxiv.org/abs/2510.16914v1](http://arxiv.org/abs/2510.16914v1)

**作者:** Hongwei Yan, Guanglong Sun, Zhiqi Kang, Yi Zhong, Liyuan Wang

**发布时间:** 2025-10-19

**备注:** 25 pages

### GPT解析

### 总结

本文提出了一种名为域可推广持续学习(DGCL)的新设置，以及自适应域变换(DoT)方法来解决智能系统在动态环境中学习新技能并推广到多样化场景的挑战。

### 背景

智能系统需要不断获取新技能并将其推广到多样化、未见过的场景。现有持续学习方法假设每个任务的训练和测试域相同，在域变化场景下表现不佳。

### 目的

提出DGCL设置，使模型能够学习序列任务，每个任务涉及单一域，目标是模型在所有遇到的任务和域中表现良好。解决获取、保留和利用语义及域相关信息的挑战。

### 方法

提出自适应域变换(DoT)方法，基于预训练模型，受人类大脑分布式加枢纽理论启发，在表示学习中解耦语义和域相关信息，自适应转换跨域任务表示以实现输出对齐，确保平衡和泛化的预测。

### 主要发现

DoT作为即插即用策略显著提升了最先进CL基线在DGCL下的性能，能够积累域可推广知识，具有轻量级实现确保资源效率，在全参数调整和参数高效调整范式下均有效。

### 结论

DoT解决了DGCL中的独特挑战，通过解耦语义和域相关信息实现更好的泛化能力，使智能系统能够有效适应动态现实环境。

### 翻译

为了有效适应动态现实环境，智能系统必须不断获取新技能，同时将其推广到多样化、未见过的场景。在此，我们引入一种名为域可推广持续学习(DGCL)的新颖且现实的设置：模型学习序列任务，每个任务涉及单一域，旨在在所有遇到的任务和域中表现良好。这种设置在获取、保留和利用语义和域相关信息以实现稳健泛化方面提出了独特挑战。尽管最先进的持续学习方法采用预训练模型来增强任务特定泛化，但它们通常假设每个任务的训练和测试域相同，因此在DGCL中表现不佳。为此，我们提出了自适应域变换(DoT)，这是一种专为DGCL设计的创新预训练模型方法。受人类大脑分布式加枢纽理论的启发，DoT在表示学习中解耦语义和域相关信息，并自适应地跨不同域转换任务表示以实现输出对齐，确保平衡和泛化的预测。DoT作为一种即插即用策略，在DGCL下极大地促进了最先进CL基线在全参数调整和参数高效调整范式中的性能，并通过大量实验得到验证。此外，DoT被证明能够从DGCL中积累域可推广知识，并通过轻量级实现确保资源效率。


### 论文摘要

To adapt effectively to dynamic real-world environments, intelligent systems must continually acquire new skills while generalizing them to diverse, unseen scenarios. Here, we introduce a novel and realistic setting named domain generalizable continual learning (DGCL): a model learns sequential tasks with each involving a single domain, aiming to perform well across all encountered tasks and domains. This setting poses unique challenges in acquiring, retaining, and leveraging both semantic- and domain-relevant information for robust generalization. Although state-of-the-art continual learning (CL) methods have employed pre-trained models (PTMs) to enhance task-specific generalization, they typically assume identical training and testing domains for each task and therefore perform poorly in DGCL. To this end, we propose adaptive Domain Transformation (DoT), an innovative PTMs-based approach tailored to DGCL. Inspired by the distributed-plus-hub theory of the human brain, DoT disentangles semantic- and domain-relevant information in representation learning, and adaptively transforms task representations across various domains for output alignment, ensuring balanced and generalized predictions. DoT serves as a plug-in strategy that greatly facilitates state-of-the-art CL baselines under both full parameter tuning and parameter-efficient tuning paradigms in DGCL, validated by extensive experiments. Also, DoT is shown to accumulate domain-generalizable knowledge from DGCL, and ensure resource efficiency with a lightweight implementation.

---

## 105. Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.16877v1](http://arxiv.org/abs/2510.16877v1)

**作者:** Heming Zou, Yunliang Zang, Wutong Xu, Xiangyang Ji

**发布时间:** 2025-10-19

### GPT解析

### 总结

Fly-CL是一种受果蝇嗅觉回路启发的生物启发框架，用于持续表征学习，解决了直接利用预训练特征时的多重共线性问题，同时显著减少了训练时间，性能达到或超过当前最先进方法。

### 背景

持续表征学习范式将参数更新重新构建为相似度匹配问题以减轻灾难性遗忘，但直接利用预训练特征进行下游任务通常在相似度匹配阶段存在多重共线性问题，且更高级的方法可能对实时、低延迟应用计算成本过高。

### 目的

解决直接利用预训练特征进行下游任务时存在的多重共线性问题，并提出一种计算效率高的方法，适用于实时、低延迟应用。

### 方法

受果蝇嗅觉回路的启发，提出了Fly-CL框架，与各种预训练骨干网络兼容。从理论上展示了Fly-CL如何逐步解决多重共线性问题，实现更有效的相似度匹配，同时具有低时间复杂度。

### 主要发现

Fly-CL显著减少了训练时间，同时实现了与当前最先进方法相当或更好的性能。通过生物启发设计有效解决了多重共线性挑战。

### 结论

Fly-CL是一种生物启发框架，与多种预训练骨干网络兼容。在不同网络架构和数据集上的广泛模拟实验验证了其有效性。代码已公开在GitHub上。

### 翻译

使用几乎冻结的预训练模型，持续表征学习范式将参数更新重新构建为相似度匹配问题，以减轻灾难性遗忘。然而，直接利用预训练特征进行下游任务通常在相似度匹配阶段存在多重共线性问题，更高级的方法可能对实时、低延迟应用来说计算成本过高。受果蝇嗅觉回路的启发，我们提出了Fly-CL，这是一种与多种预训练骨干网络兼容的生物启发框架。Fly-CL显著减少了训练时间，同时实现了与当前最先进方法相当或更好的性能。我们从理论上展示了Fly-CL如何逐步解决多重共线性问题，实现更有效的相似度匹配，同时具有低时间复杂度。在各种网络架构和数据集上的广泛模拟实验验证了Fly-CL通过生物启发设计解决这一挑战的有效性。代码可在https://github.com/gfyddha/Fly-CL获取。


### 论文摘要

Using a nearly-frozen pretrained model, the continual representation learning paradigm reframes parameter updates as a similarity-matching problem to mitigate catastrophic forgetting. However, directly leveraging pretrained features for downstream tasks often suffers from multicollinearity in the similarity-matching stage, and more advanced methods can be computationally prohibitive for real-time, low-latency applications. Inspired by the fly olfactory circuit, we propose Fly-CL, a bio-inspired framework compatible with a wide range of pretrained backbones. Fly-CL substantially reduces training time while achieving performance comparable to or exceeding that of current state-of-the-art methods. We theoretically show how Fly-CL progressively resolves multicollinearity, enabling more effective similarity matching with low time complexity. Extensive simulation experiments across diverse network architectures and data regimes validate Fly-CL's effectiveness in addressing this challenge through a biologically inspired design. Code is available at https://github.com/gfyddha/Fly-CL.

---

## 106. 3D-GSRD: 3D Molecular Graph Auto-Encoder with Selective Re-mask Decoding

**论文链接:** [http://arxiv.org/abs/2510.16780v1](http://arxiv.org/abs/2510.16780v1)

**作者:** Chang Wu, Zhiyuan Liu, Wen Shu, Liang Wang, Yanchen Luo, Wenqiang Lei, Yatao Bian, Junfeng Fang, Xiang Wang

**发布时间:** 2025-10-19

### GPT解析

### 总结

本文提出了一种名为3D-GSRD的新型3D分子图自编码器，通过选择性重新掩码解码技术解决了将2D MGM成功扩展到3D MGM时面临的两个相互冲突的挑战。

### 背景

掩码图模型(MGM)是分子表示学习(MRL)的一种有前景的方法，但将2D重新掩码解码的成功经验扩展到3D MGM面临两个相互冲突的挑战：避免2D结构信息泄漏到解码器，同时为重新掩码的原子重构提供足够的2D上下文。

### 目的

开发一种能够有效处理3D分子数据并解决2D结构信息泄漏与上下文提供之间矛盾的分子表示学习方法。

### 方法

提出3D-GSRD，其核心创新是选择性重新掩码解码(SRD)，该技术仅从编码器表示中重新掩码3D相关信息，同时保留2D图结构。SRD与3D关系转换器(3D-ReTrans)编码器和结构无关的解码器协同集成。

### 主要发现

SRD与结构无关的解码器增强了编码器在分子表示学习中的作用。在MD17分子性质预测基准测试中，3D-GSRD在8个目标中的7个上达到了最新的最优性能。

### 结论

3D-GSRD成功解决了将2D MGM扩展到3D MGM时面临的挑战，为分子表示学习提供了新的有效方法。

### 翻译

掩码图建模(MGM)是分子表示学习(MRL)的一种有前景的方法。然而，将重新掩码解码的成功从2D扩展到3D MGM并非易事，主要由于两个相互冲突的挑战：避免将2D结构信息泄漏到解码器，同时仍为重新掩码的原子重构提供足够的2D上下文。为解决这些挑战，我们提出了3D-GSRD：一种具有选择性重新掩码解码的3D分子图自编码器。3D-GSRD的核心创新在于其选择性重新掩码解码(SRD)，它仅从编码器表示中重新掩码3D相关信息，同时保留2D图结构。SRD与3D关系转换器(3D-ReTrans)编码器和结构无关的解码器协同集成。我们分析指出，SRD结合结构无关的解码器增强了编码器在MRL中的作用。大量实验表明，3D-GSRD实现了强大的下游性能，在广泛使用的MD17分子性质预测基准的8个目标中，有7个达到了最新的最优状态。代码已发布在https://github.com/WuChang0124/3D-GSRD。


### 论文摘要

Masked graph modeling (MGM) is a promising approach for molecular representation learning (MRL).However, extending the success of re-mask decoding from 2D to 3D MGM is non-trivial, primarily due to two conflicting challenges: avoiding 2D structure leakage to the decoder, while still providing sufficient 2D context for reconstructing re-masked atoms.To address these challenges, we propose 3D-GSRD: a 3D Molecular Graph Auto-Encoder with Selective Re-mask Decoding. The core innovation of 3D-GSRD lies in its Selective Re-mask Decoding(SRD), which re-masks only 3D-relevant information from encoder representations while preserving the 2D graph structures.This SRD is synergistically integrated with a 3D Relational-Transformer(3D-ReTrans) encoder alongside a structure-independent decoder. We analyze that SRD, combined with the structure-independent decoder, enhances the encoder's role in MRL. Extensive experiments show that 3D-GSRD achieves strong downstream performance, setting a new state-of-the-art on 7 out of 8 targets in the widely used MD17 molecular property prediction benchmark. The code is released at https://github.com/WuChang0124/3D-GSRD.

---

## 107. SCALAR: Self-Calibrating Adaptive Latent Attention Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.16474v1](http://arxiv.org/abs/2510.16474v1)

**作者:** Farwa Abbas, Hussain Ahmad, Claudia Szabo

**发布时间:** 2025-10-18

### GPT解析

### 总结

本文提出了一种基于自适应核注意力机制的新预测建模方法，解决了传统方法在高维、异构数据处理中的局限性，实验证明该方法优于现有技术。

### 背景

高维、异构数据及其复杂特征交互对传统预测建模方法构成挑战。传统方法如投影到潜在结构(PLS)难以建模复杂非线性关系，特别是在具有高维相关结构的多变量系统中。多尺度上的同时交互使局部处理无法捕获跨组依赖，而静态特征加权限制了模型对上下文变化的适应性。

### 目的

提出一种新方法，通过新颖的架构创新来增强预测性能，解决传统方法在高维、异构数据处理中的局限性。

### 方法

提出了一种新颖的架构，引入基于自适应核的注意力机制。该机制分别处理不同的特征组，然后在集成之前进行整合，从而能够捕获局部模式同时保留全局关系。

### 主要发现

实验结果表明，与最先进的方法相比，作者提出的方法在各种数据集上的性能指标都有显著改进。

### 结论

基于自适应核的注意力机制架构能够有效处理高维、异构数据中的复杂特征交互和多尺度交互问题，提高了预测性能。

### 翻译

高维、异构数据与复杂特征交互对传统预测建模方法构成了重大挑战。虽然投影到潜在结构(PLS)仍然是一种流行技术，但它难以建模复杂的非线性关系，特别是在具有高维相关结构的多变量系统中。多尺度上的同时交互进一步加剧了这一挑战，使得局部处理无法捕获跨组依赖关系。此外，静态特征加权限制了适应上下文变化的能力，因为它忽略了样本特定的相关性。为解决这些局限性，我们提出了一种通过新颖架构创新来增强预测性能的新方法。我们的架构引入了一个基于自适应核的注意力机制，它分别处理不同的特征组，然后在集成之前进行整合，从而能够捕获局部模式同时保留全局关系。实验结果表明，与最先进的方法相比，在各种数据集上的性能指标都有显著改进。


### 论文摘要

High-dimensional, heterogeneous data with complex feature interactions pose significant challenges for traditional predictive modeling approaches. While Projection to Latent Structures (PLS) remains a popular technique, it struggles to model complex non-linear relationships, especially in multivariate systems with high-dimensional correlation structures. This challenge is further compounded by simultaneous interactions across multiple scales, where local processing fails to capture crossgroup dependencies. Additionally, static feature weighting limits adaptability to contextual variations, as it ignores sample-specific relevance. To address these limitations, we propose a novel method that enhances predictive performance through novel architectural innovations. Our architecture introduces an adaptive kernel-based attention mechanism that processes distinct feature groups separately before integration, enabling capture of local patterns while preserving global relationships. Experimental results show substantial improvements in performance metrics, compared to the state-of-the-art methods across diverse datasets.

---

## 108. Humanoid-inspired Causal Representation Learning for Domain Generalization

**论文链接:** [http://arxiv.org/abs/2510.16382v1](http://arxiv.org/abs/2510.16382v1)

**作者:** Ze Tao, Jian Zhang, Haowei Li, Xianshuai Li, Yifei Peng, Xiyao Liu, Senzhang Wang, Chao Liu, Sheng Ren, Shichao Zhang

**发布时间:** 2025-10-18

### GPT解析

### 总结

本文提出了一种受人类智能启发的结构化因果模型HSCM，通过模仿人类视觉系统的分层处理和多级学习机制，专注于建模细粒度因果关系，从而提升模型在不同领域间的泛化能力和稳健性。

### 背景

传统领域泛化模型存在局限性，它们通常依赖统计数据来捕获数据标签依赖和学习扭曲不变表示，无法充分捕捉人类视觉系统的层次化处理机制。

### 目的

克服传统领域泛化模型的局限性，开发一种受人类智能启发的因果框架，提升模型在不同领域间的泛化能力，确保模型的稳健性和可解释性。

### 方法

提出Humanoid-inspired Structural Causal Model (HSCM)，模仿人类视觉系统的分层处理和多级学习，通过解耦和重新加权关键图像属性（如颜色、纹理和形状）来建模细粒度因果机制。

### 主要发现

HSCM通过理论和实证评估证明优于现有领域泛化模型，提供了更规范的方法来捕获因果关系并提高模型稳健性，在动态复杂环境中实现更有效的迁移学习。

### 结论

HSCM作为一种受人类智能启发的因果框架，能够有效提升模型跨领域的泛化能力，确保稳健性和可解释性，为领域泛化问题提供了新的解决思路。

### 翻译

本文提出了一种受人类智能启发的结构化因果模型（HSCM），这是一种新颖的因果框架，旨在克服传统领域泛化模型的局限性。与依赖统计数据捕获数据标签依赖和学习扭曲不变表示的方法不同，HSCM模仿人类视觉系统的分层处理和多级学习，专注于建模细粒度因果机制。通过解耦和重新加权关键图像属性（如颜色、纹理和形状），HSCM增强了跨不同领域的泛化能力，确保了模型的稳健性和可解释性。利用人类智能的灵活性和适应性，我们的方法使模型在动态复杂环境中能够实现更有效的迁移和学习。通过理论和实证评估，我们证明了HSCM优于现有领域泛化模型，为捕获因果关系和提高模型稳健性提供了更规范的方法。代码可在https://github.com/lambett/HSCM获取。


### 论文摘要

This paper proposes the Humanoid-inspired Structural Causal Model (HSCM), a novel causal framework inspired by human intelligence, designed to overcome the limitations of conventional domain generalization models. Unlike approaches that rely on statistics to capture data-label dependencies and learn distortion-invariant representations, HSCM replicates the hierarchical processing and multi-level learning of human vision systems, focusing on modeling fine-grained causal mechanisms. By disentangling and reweighting key image attributes such as color, texture, and shape, HSCM enhances generalization across diverse domains, ensuring robust performance and interpretability. Leveraging the flexibility and adaptability of human intelligence, our approach enables more effective transfer and learning in dynamic, complex environments. Through both theoretical and empirical evaluations, we demonstrate that HSCM outperforms existing domain generalization models, providing a more principled method for capturing causal relationships and improving model robustness. The code is available at https://github.com/lambett/HSCM.

---

## 109. MLCPD: A Unified Multi-Language Code Parsing Dataset with Universal AST Schema

**论文链接:** [http://arxiv.org/abs/2510.16357v1](http://arxiv.org/abs/2510.16357v1)

**作者:** Jugal Gajjar, Kamalasankari Subramaniakuppusamy

**发布时间:** 2025-10-18

**备注:** 12 pages, 7 figures, 4 tables, 2 algorithms, and 34 references.  HuggingFace:  https://huggingface.co/datasets/jugalgajjar/MultiLang-Code-Parser-Dataset  GitHub: https://github.com/JugalGajjar/MultiLang-Code-Parser-Dataset

### GPT解析

### 总结

本文介绍了多语言代码解析器数据集(MLCPD)，一个统一十种主要编程语言语法结构的大规模、语言无关数据集，包含超过七百万个标准化解析源文件，支持跨语言推理、结构学习和多语言软件分析。

### 背景

现有代码语料库主要关注标记级代码或孤立解析器，缺乏跨语言的统一表示，需要一种能够统一不同编程语言语法结构的数据集。

### 目的

创建一个大规模、语言无关的数据集，统一十种主要编程语言的语法和结构表示，为跨语言表示学习和程序分析提供开放、可重现的基础。

### 方法

提出通用抽象语法树(AST)模式标准化解析源文件，为每个文件提供分层树表示和丰富元数据，以Parquet格式存储，进行跨语言结构分析，并开发数据集复现、语法编译和可视化工具。

### 主要发现

跨语言代码结构存在强大的规律性，差异很大的编程语言(如Python、Java和Go)的语法图可以在共享模式下对齐，提出的统一AST模式能够无损地表示不同语言的语法结构。

### 结论

MLCPD为跨语言表示学习和程序分析提供了开放、可重现的基础，通过统一的数据表示和丰富的工具支持，促进了多语言软件分析和跨语言推理研究。

### 翻译

我们引入了多语言代码解析器数据集(MLCPD)，这是一个大规模、语言无关的数据集，统一了十种主要编程语言的代码语法和结构表示。MLCPD包含超过七百万个在我们提出的通用抽象语法树(AST)模式下标准化的解析源文件，实现了跨语言推理、结构学习和多语言软件分析的一致性。与仅关注标记级代码或孤立解析器的现有语料库不同，MLCPD为每个文件提供了分层树表示和丰富的元数据，确保无损的语法覆盖和结构一致性。每个条目包括标准化的模式、语言级元数据和抽象节点语义，以Parquet格式存储以便可扩展检索。经验分析揭示了强大的跨语言结构规律性，证明像Python、Java和Go这样差异很大的语言的语法图可以在共享模式下对齐。我们在Hugging Face上公开发布了该数据集，并在GitHub上提供了配套代码库，包括数据集复现、语法编译和探索跨语言统一AST的可视化工具。这些资源共同确立了MLCPD作为跨语言表示学习和程序分析未来研究的开放、可重现基础。


### 论文摘要

We introduce the MultiLang Code Parser Dataset (MLCPD), a large-scale, language-agnostic dataset unifying syntactic and structural representations of code across ten major programming languages. MLCPD contains over seven million parsed source files normalized under our proposed universal Abstract Syntax Tree (AST) schema, enabling consistent cross-language reasoning, structural learning, and multilingual software analysis. Unlike existing corpora that focus purely on token-level code or isolated parsers, MLCPD provides both hierarchical tree representations and rich metadata for every file, ensuring lossless syntactic coverage and structural uniformity. Each entry includes a normalized schema, language-level metadata, and abstracted node semantics stored in Parquet format for scalable retrieval. Empirical analyses reveal strong cross-language structural regularities-demonstrating that syntactic graphs from languages as diverse as Python, Java, and Go can be aligned under a shared schema. We release the dataset publicly on Hugging Face and the accompanying codebase on GitHub, which includes complete pipelines for dataset reproduction, grammar compilation, and a visualization tool for exploring the unified AST across languages. Together, these resources establish MLCPD as an open, reproducible foundation for future research in cross-language representation learning and program analysis.

---

## 110. Disentangling Hyperedges through the Lens of Category Theory

**论文链接:** [http://arxiv.org/abs/2510.16289v1](http://arxiv.org/abs/2510.16289v1)

**作者:** Yoonho Lee, Junseok Lee, Sangwoo Seo, Sungwon Kim, Yeongmin Kim, Chanyoung Park

**发布时间:** 2025-10-18

**备注:** Accepted to NeurIPS 2025

### GPT解析

### 总结

本研究探索了超图结构数据中的超边解缠问题，从范畴论角度提出了一种新的解缠准则，并通过基因功能关系分析验证了其有效性。

### 背景

尽管解缠表示学习在图结构数据分析中取得了进展，但针对超图结构数据的解缠研究较少，存在研究空白。

### 目的

将超边解缠整合到超图神经网络中，使模型能够利用与标签相关的隐藏超边语义，如节点间未注释的关系。

### 方法

从范畴论角度分析超边解缠，提出基于自然性条件的新解缠准则，并构建概念验证模型进行实验。

### 主要发现

概念验证模型成功捕获了基因通路中基因的功能关系，证明了所提准则的潜力。

### 结论

基于自然性条件的解缠准则在超图结构数据中有效，特别是在分析基因功能关系方面表现出应用潜力。

### 翻译

尽管解缠表示学习在发现图结构数据中的潜在模式方面取得了有希望的结果，但很少有研究探索超图结构数据的解缠。将超边解缠整合到超图神经网络中，使模型能够利用与标签相关的隐藏超边语义，例如节点之间未注释的关系。本文从范畴论的角度对超边解缠进行了分析，并提出了一种从自然性条件推导出的新解缠准则。我们的概念验证模型通过成功捕获基因通路（超边）中基因（节点）的功能关系，实验性地展示了所提出准则的潜力。


### 论文摘要

Despite the promising results of disentangled representation learning in discovering latent patterns in graph-structured data, few studies have explored disentanglement for hypergraph-structured data. Integrating hyperedge disentanglement into hypergraph neural networks enables models to leverage hidden hyperedge semantics, such as unannotated relations between nodes, that are associated with labels. This paper presents an analysis of hyperedge disentanglement from a category-theoretical perspective and proposes a novel criterion for disentanglement derived from the naturality condition. Our proof-of-concept model experimentally showed the potential of the proposed criterion by successfully capturing functional relations of genes (nodes) in genetic pathways (hyperedges).

---

## 111. MuseTok: Symbolic Music Tokenization for Generation and Semantic Understanding

**论文链接:** [http://arxiv.org/abs/2510.16273v1](http://arxiv.org/abs/2510.16273v1)

**作者:** Jingyue Huang, Zachary Novack, Phillip Long, Yupeng Hou, Ke Chen, Taylor Berg-Kirkpatrick, Julian McAuley

**发布时间:** 2025-10-18

### GPT解析

### 总结

MuseTok是一种创新的离散表示学习方法，专门针对符号音乐设计，结合了RQ-VAE和Transformer架构，在音乐生成和语义理解任务中均取得了优异的性能。

### 背景

离散表示学习在图像、语音和语言的生成和理解领域已显示出有前景的结果，这些进展启发了作者对音乐符号表示的研究。

### 目的

提出MuseTok，一种用于符号音乐的标记化方法，研究其在音乐生成和理解任务中的有效性。

### 方法

MuseTok采用基于残差向量量化-变分自编码器（RQ-VAE）的方法，在基于Transformer的编码器-解码器框架中，对小节音乐段进行处理，生成能够实现高保真音乐重建和准确音乐理论理解的音乐代码。

### 主要发现

在音乐生成和语义理解任务的综合评估中，使用MuseTok的模型在语义理解方面优于先前的表示学习基线，在内容生成方面保持可比的性能；对MuseTok代码的定性分析表明，它能够从大型音乐集中有效捕捉潜在的音乐概念。

### 结论

MuseTok是一种有效的符号音乐标记化方法，在音乐生成和理解任务中表现良好。

### 翻译

离散表示学习在图像、语音和语言的生成和理解等多个领域已显示出有前景的结果。受这些进展的启发，我们提出了MuseTok，一种用于符号音乐的标记化方法，并研究了其在音乐生成和理解任务中的有效性。MuseTok在基于Transformer的编码器-解码器框架中，对小节音乐段应用残差向量量化-变分自编码器（RQ-VAE），生成能够实现高保真音乐重建和准确音乐理论理解的音乐代码。为了进行全面评估，我们将MuseTok应用于音乐生成和语义理解任务，包括旋律提取、和弦识别和情感识别。采用MuseTok的模型在语义理解方面优于先前的表示学习基线，同时在内容生成方面保持可比的性能。此外，使用真实类别和合成数据集对MuseTok代码进行的定性分析表明，MuseTok能够有效从大型音乐集中捕捉潜在的音乐概念。


### 论文摘要

Discrete representation learning has shown promising results across various domains, including generation and understanding in image, speech and language. Inspired by these advances, we propose MuseTok, a tokenization method for symbolic music, and investigate its effectiveness in both music generation and understanding tasks. MuseTok employs the residual vector quantized-variational autoencoder (RQ-VAE) on bar-wise music segments within a Transformer-based encoder-decoder framework, producing music codes that achieve high-fidelity music reconstruction and accurate understanding of music theory. For comprehensive evaluation, we apply MuseTok to music generation and semantic understanding tasks, including melody extraction, chord recognition, and emotion recognition. Models incorporating MuseTok outperform previous representation learning baselines in semantic understanding while maintaining comparable performance in content generation. Furthermore, qualitative analyses on MuseTok codes, using ground-truth categories and synthetic datasets, reveal that MuseTok effectively captures underlying musical concepts from large music collections.

---

## 112. FSRF: Factorization-guided Semantic Recovery for Incomplete Multimodal Sentiment Analysis

**论文链接:** [http://arxiv.org/abs/2510.16086v1](http://arxiv.org/abs/2510.16086v1)

**作者:** Ziyang Liu, Pengjunfei Chu, Shuming Dong, Chen Zhang, Mingcheng Li, Jin Wang

**发布时间:** 2025-10-17

**备注:** 6 pages,3 figures

### GPT解析

### 总结

本文提出了一种因子引导的语义恢复框架(FSRF)，用于解决多模态情感分析中的模态缺失问题，通过去冗余的同质异质因子分解模块和分布对齐的自蒸馏模块，有效恢复了缺失模态的语义信息，实验证明该方法在不确定缺失模态的情况下具有显著的性能优势。

### 背景

多模态情感分析(MSA)已成为研究热点，旨在利用多模态数据进行人类情感理解。以往研究主要关注完整多模态数据的交互和融合，忽略了实际应用中因遮挡、个人隐私限制和设备故障导致的模态缺失问题，导致泛化能力低。

### 目的

提出一个因子引导的语义恢复框架(FSRF)，以缓解MSA任务中的模态缺失问题。

### 方法

提出了一种去冗余的同质异质因子分解模块，将模态分解为模态同质、模态异质和噪声表示，并设计了表示学习的精细约束范式；设计了一种分布对齐的自蒸馏模块，通过利用双向知识转移完全恢复缺失的语义。

### 主要发现

在两个数据集上的综合实验表明，与之前的方法相比，FSRF在不确定缺失模态的情况下具有显著的性能优势。

### 结论

FSRF框架有效解决了多模态情感分析中的模态缺失问题，提高了模型在真实应用场景中的泛化能力。

### 翻译

近年来，多模态情感分析(MSA)已成为一个研究热点，旨在利用多模态数据进行人类情感理解。以往的MSA研究主要集中在完整多模态数据的交互和融合上，忽略了实际应用中因遮挡、个人隐私限制和设备故障导致的模态缺失问题，导致泛化能力低。为此，我们提出了一种因子引导的语义恢复框架(FSRF)，以缓解MSA任务中的模态缺失问题。具体而言，我们提出了一种去冗余的同质异质因子分解模块，将模态分解为模态同质、模态异质和噪声表示，并设计了表示学习的精细约束范式。此外，我们设计了一种分布对齐的自蒸馏模块，通过利用双向知识转移完全恢复缺失的语义。在两个数据集上的综合实验表明，与之前的方法相比，FSRF在不确定缺失模态的情况下具有显著的性能优势。


### 论文摘要

In recent years, Multimodal Sentiment Analysis (MSA) has become a research hotspot that aims to utilize multimodal data for human sentiment understanding. Previous MSA studies have mainly focused on performing interaction and fusion on complete multimodal data, ignoring the problem of missing modalities in real-world applications due to occlusion, personal privacy constraints, and device malfunctions, resulting in low generalizability.   To this end, we propose a Factorization-guided Semantic Recovery Framework (FSRF) to mitigate the modality missing problem in the MSA task.   Specifically, we propose a de-redundant homo-heterogeneous factorization module that factorizes modality into modality-homogeneous, modality-heterogeneous, and noisy representations and design elaborate constraint paradigms for representation learning.   Furthermore, we design a distribution-aligned self-distillation module that fully recovers the missing semantics by utilizing bidirectional knowledge transfer.   Comprehensive experiments on two datasets indicate that FSRF has a significant performance advantage over previous methods with uncertain missing modalities.

---

## 113. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.15430v2](http://arxiv.org/abs/2510.15430v2)

**作者:** Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**发布时间:** 2025-10-17

**备注:** Withdrawn due to an accidental duplicate submission. This paper  (arXiv:2510.15430) was unintentionally submitted as a new entry instead of a  new version of our previous work (arXiv:2508.09201)

### GPT解析

### 总结

本文提出了一种名为Learning to Detect (LoD)的通用框架，用于检测大视觉语言模型中的未知越狱攻击。

### 背景

尽管进行了广泛的对齐努力，大视觉语言模型(LVLMs)仍然容易受到越狱攻击，带来严重的安全风险。

### 目的

解决现有检测方法的局限性，这些方法要么学习特定攻击参数（难以泛化到新攻击），要么依赖启发式原理（限制准确性和效率）。

### 方法

提出Learning to Detect (LoD)框架，通过从攻击特定学习转向任务特定学习来检测未知越狱攻击。框架包括：1)多模态安全概念激活向量模块，用于安全导向的表征学习；2)安全模式自动编码器模块，用于无监督攻击分类。

### 主要发现

广泛实验表明，该方法在各种未知攻击上实现了更高的一致性检测AUROC，同时提高了效率。

### 结论

Learning to Detect框架有效解决了现有检测方法的局限性，能够准确检测未知越狱攻击并提高效率。

### 翻译

尽管进行了广泛的对齐努力，大视觉语言模型(LVLMs)仍然容易受到越狱攻击，带来了严重的安全风险。为了解决这个问题，现有的检测方法要么学习特定攻击的参数，这阻碍了对未见过攻击的泛化能力，要么依赖启发式原理，这限制了准确性和效率。为了克服这些限制，我们提出了Learning to Detect (LoD)框架，通过将重点从攻击特定学习转向任务特定学习，准确检测未知的越狱攻击。该框架包括一个用于安全导向表征学习的多模态安全概念激活向量模块和一个用于无监督攻击分类的安全模式自动编码器模块。广泛的实验表明，我们的方法在各种未知攻击上实现了更高的一致性检测AUROC，同时提高了效率。代码可在https://anonymous.4open.science/r/Learning-to-Detect-51CB获取。


### 论文摘要

Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.

---

## 114. Large-scale User Game Lifecycle Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.15412v2](http://arxiv.org/abs/2510.15412v2)

**作者:** Yanjie Gou, Jiangming Liu, Kouying Xue, Yi Hu

**发布时间:** 2025-10-17

### GPT解析

### 总结

该研究针对在线游戏平台广告和推荐系统的需求，提出了一种称为用户游戏生命周期(UGL)的表示学习方法，以解决游戏数据稀疏和游戏不平衡问题，并通过实验验证了该方法的有效性。

### 背景

随着视频游戏的快速扩张，在线游戏平台需要有效的广告和推荐系统。现有的表示学习方法是为处理推荐系统中的数十亿个项目而设计的，但不适用于游戏广告和推荐，主要原因是游戏稀疏性（仅有数百个游戏不足以进行大规模用户表示学习）和游戏不平衡性（用户行为被少数热门游戏主导）。

### 目的

解决游戏稀疏性和游戏不平衡性对游戏广告和推荐系统的影响，提高用户兴趣捕捉的准确性。

### 方法

1. 引入用户游戏生命周期(UGL)来丰富用户在游戏中的行为，解决稀疏性问题；2. 提出两种创新策略来操纵用户行为，更有效地提取短期和长期兴趣；3. 提出逆概率掩码策略用于UGL表示学习，解决游戏不平衡挑战。

### 主要发现

UGL表示显著提升了模型性能：对于游戏广告，平均AUC离线增加1.83%，CVR在线平均增加21.67%；对于游戏内物品推荐，平均AUC离线增加0.5%，ARPU在线平均增加0.82%。

### 结论

UGL表示学习方法能够有效解决游戏稀疏性和不平衡性问题，显著提升游戏广告和推荐系统的性能。

### 翻译

随着视频游戏生产的快速扩张，有必要为在线游戏平台开发有效的广告和推荐系统。向用户推荐和宣传游戏取决于捕捉他们对游戏的兴趣。然而，为处理推荐系统中的数十亿个项目而设计的现有表示学习方法不适用于游戏广告和推荐。这主要是由于游戏稀疏性，其中仅有的数百个游戏不足以进行大规模用户表示学习，以及游戏不平衡性，其中用户行为被少数热门游戏主导。为了解决稀疏性问题，我们引入了用户游戏生命周期(UGL)，旨在丰富用户在游戏中的行为。此外，我们提出了两种创新策略，旨在操纵用户行为以更有效地提取短期和长期兴趣。为了应对游戏不平衡挑战，我们提出了用于UGL表示学习的逆概率掩码策略。离线和在线实验结果表明，UGL表示显著增强了模型性能，在游戏广告方面平均实现1.83%的AUC离线增长和21.67%的CVR在线增长，在游戏内物品推荐方面平均实现0.5%的AUC离线增长和0.82%的ARPU在线增长。


### 论文摘要

The rapid expansion of video game production necessitates the development of effective advertising and recommendation systems for online game platforms. Recommending and advertising games to users hinges on capturing their interest in games. However, existing representation learning methods crafted for handling billions of items in recommendation systems are unsuitable for game advertising and recommendation. This is primarily due to game sparsity, where the mere hundreds of games fall short for large-scale user representation learning, and game imbalance, where user behaviors are overwhelmingly dominated by a handful of popular games. To address the sparsity issue, we introduce the User Game Lifecycle (UGL), designed to enrich user behaviors in games. Additionally, we propose two innovative strategies aimed at manipulating user behaviors to more effectively extract both short and long-term interests. To tackle the game imbalance challenge, we present an Inverse Probability Masking strategy for UGL representation learning. The offline and online experimental results demonstrate that the UGL representations significantly enhance model by achieving a 1.83% AUC offline increase on average and a 21.67% CVR online increase on average for game advertising and a 0.5% AUC offline increase and a 0.82% ARPU online increase for in-game item recommendation.

---

## 115. HumanCM: One Step Human Motion Prediction

**论文链接:** [http://arxiv.org/abs/2510.16709v1](http://arxiv.org/abs/2510.16709v1)

**作者:** Liu Haojie, Gao Suixiang

**发布时间:** 2025-10-19

**备注:** 6 pages, 2 figures, 2 tables

### GPT解析

### 总结

本文提出了HumanCM，一个基于一致性模型的一步式人体运动预测框架，能够高效地单步生成人体运动。

### 背景

现有的基于扩散模型的人体运动预测方法依赖多步去噪过程，计算效率较低。

### 目的

开发一种能够高效、准确地预测人体运动的单步生成方法，减少计算负担。

### 方法

HumanCM采用基于Transformer的时空架构，通过学习嘈杂和清洁运动状态之间的自一致映射来实现单步生成，并使用时间嵌入来建模长程依赖关系和保持运动连贯性。

### 主要发现

在Human3.6M和HumanEva-I数据集上的实验表明，HumanCM实现了与最先进的扩散模型相当或更好的准确性，同时将推理步骤减少了高达两个数量级。

### 结论

HumanCM是一种高效的人体运动预测方法，能够在保持高准确性的同时显著减少计算负担。

### 翻译

我们提出了HumanCM，这是一个基于一致性模型的一步式人体运动预测框架。与依赖多步去噪的基于扩散的方法不同，HumanCM通过学习嘈杂和清洁运动状态之间的自一致映射来执行高效的单步生成。该框架采用基于Transformer的时空架构，并使用时间嵌入来建模长程依赖关系并保持运动连贯性。在Human3.6M和HumanEva-I上的实验表明，HumanCM实现了与最先进的扩散模型相当或更好的准确性，同时将推理步骤减少了高达两个数量级。


### 论文摘要

We present HumanCM, a one-step human motion prediction framework built upon consistency models. Instead of relying on multi-step denoising as in diffusion-based methods, HumanCM performs efficient single-step generation by learning a self-consistent mapping between noisy and clean motion states. The framework adopts a Transformer-based spatiotemporal architecture with temporal embeddings to model long-range dependencies and preserve motion coherence. Experiments on Human3.6M and HumanEva-I demonstrate that HumanCM achieves comparable or superior accuracy to state-of-the-art diffusion models while reducing inference steps by up to two orders of magnitude.

---

