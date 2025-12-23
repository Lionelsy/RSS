# 今日论文推荐 - 2025-12-23

共 108 篇论文

---

## 1. From Pixels to Predicates Structuring urban perception with scene graphs

**论文链接:** [http://arxiv.org/abs/2512.19221v1](http://arxiv.org/abs/2512.19221v1)

**作者:** Yunlong Liu, Shuyang Li, Pengyuan Liu, Yu Zhang, Rudi Stouffs

**发布时间:** 2025-12-22

**备注:** 10 pages, CAADRIA2026 presentation forthcoming

### GPT解析

### 总结

本研究提出了一种基于图的三阶段管道，将街道视图图像转换为结构化表示，用于预测城市感知指标，显著提高了预测准确性和跨城市泛化能力。

### 背景

感知研究越来越多地使用街道景观建模，但许多方法仍依赖像素特征或物体共现统计，忽略了塑造人类感知的显式关系。

### 目的

开发一种能捕捉城市环境中显式关系的结构化表示方法，用于准确预测城市感知指标。

### 方法

三阶段管道：1)使用开放集全景场景图模型(OpenPSG)提取图像中的对象-谓词-对象三元组；2)通过异构图自编码器(GraphMAE)学习场景级嵌入；3)使用神经网络从嵌入预测感知分数。

### 主要发现

1)该方法比基线模型平均提高26%的感知预测准确性；2)在跨城市预测任务中保持强泛化性能；3)结构化表示揭示了导致低感知分数的关系模式，如墙上的涂鸦和停在人行道上的汽车。

### 结论

基于图的结构为建模城市感知提供了表达性强、可泛化和可解释的信号，推进了以人为中心和上下文感知的城市分析。

### 翻译

感知研究越来越多地使用街道景观建模，但许多方法仍然依赖像素特征或物体共现统计，忽略了塑造人类感知的显式关系。本研究提出一个三阶段管道，将街道视图图像(SVI)转换为结构化表示，用于预测六个感知指标。第一阶段使用开放集全景场景图模型(OpenPSG)解析每张图像，提取对象-谓词-对象三元组。第二阶段通过异构图自编码器(GraphMAE)学习紧凑的场景级嵌入。第三阶段使用神经网络从这些嵌入预测感知分数。我们针对准确性、精确度和跨城市泛化，将该方法与仅基于图像的基线模型进行比较。结果表明：(i)我们的方法比基线模型平均提高26%的感知预测准确性，(ii)在跨城市预测任务中保持强泛化性能。此外，结构化表示阐明了哪些关系模式导致城市场景中感知分数降低，例如墙上的涂鸦和停在人行道上的汽车。总体而言，本研究表明基于图的结构为建模城市感知提供了表达性强、可泛化和可解释的信号，推进了以人为中心和上下文感知的城市分析。


### 论文摘要

Perception research is increasingly modelled using streetscapes, yet many approaches still rely on pixel features or object co-occurrence statistics, overlooking the explicit relations that shape human perception. This study proposes a three stage pipeline that transforms street view imagery (SVI) into structured representations for predicting six perceptual indicators. In the first stage, each image is parsed using an open-set Panoptic Scene Graph model (OpenPSG) to extract object predicate object triplets. In the second stage, compact scene-level embeddings are learned through a heterogeneous graph autoencoder (GraphMAE). In the third stage, a neural network predicts perception scores from these embeddings. We evaluate the proposed approach against image-only baselines in terms of accuracy, precision, and cross-city generalization. Results indicate that (i) our approach improves perception prediction accuracy by an average of 26% over baseline models, and (ii) maintains strong generalization performance in cross-city prediction tasks. Additionally, the structured representation clarifies which relational patterns contribute to lower perception scores in urban scenes, such as graffiti on wall and car parked on sidewalk. Overall, this study demonstrates that graph-based structure provides expressive, generalizable, and interpretable signals for modelling urban perception, advancing human-centric and context-aware urban analytics.

---

## 2. PRISM-Loc: a Lightweight Long-range LiDAR Localization in Urban Environments with Topological Maps

**论文链接:** [http://arxiv.org/abs/2506.15849v2](http://arxiv.org/abs/2506.15849v2)

**作者:** Kirill Muravyev, Artem Kobozev, Vasily Yuryev, Alexander Melekhin, Oleg Bulichev, Dmitry Yudin, Konstantin Yakovlev

**发布时间:** 2025-06-18

**备注:** This version was submitted to ICRA 2026 conference

### GPT解析

### 总结

提出PRISM-Loc，一种用于大型户外环境定位的轻量级且稳健的方法，结合紧凑拓扑表示和基于原始激光雷达扫描的扫描匹配与路沿检测模块。

### 背景

在资源受限平台上进行大型户外环境定位面临挑战，需要考虑实时性能和对常见城市感知挑战的鲁棒性。

### 目的

设计适用于资源受限平台的定位方法，强调实时性能和对常见城市感知挑战的鲁棒性。

### 方法

结合紧凑拓扑表示和新型扫描匹配与路沿检测模块，直接在原始激光雷达扫描上操作，使用全局位置识别和原始扫描匹配技术在紧凑拓扑地图中提供准确定位。

### 主要发现

在标准基准测试和嵌入式平台上实验证明方法有效；在ITLP-Campus数据集上实现99%成功率；每次定位运行时间为150毫秒；使用20MB地图进行定位。

### 结论

PRISM-Loc是一种有效的大型户外环境定位解决方案，特别适合资源受限平台，具有高准确率和实时性能。

### 翻译

我们提出了PRISM-Loc - 一种用于大型户外环境定位的轻量级且稳健的方法，它结合了紧凑的拓扑表示和一种新颖的扫描匹配及路沿检测模块，直接在原始激光雷达扫描上操作。该方法专为资源受限平台设计，强调实时性能和对常见城市感知挑战的鲁棒性。它使用全局位置识别和原始扫描匹配技术在紧凑的拓扑地图中提供准确的定位。在标准基准测试和嵌入式平台上的实验证明了我们方法的有效性。我们的方法在大型ITLP-Campus数据集上实现了99%的成功率，同时每次定位运行时间为150毫秒，使用20MB的地图进行定位。我们强调了三个主要贡献：(1) 城市规模定位的紧凑表示；(2) 直接在原始激光雷达点上操作的新型路沿检测和扫描匹配流程；(3) 对我们方法的彻底评估和性能分析。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决资源受限平台上的城市环境中远程激光雷达定位问题。传统方法依赖密集的全局激光雷达地图，但随着环境范围扩大，这些地图计算成本高昂且内存需求大，限制了实际应用。在城市环境中，GPS信号可能被遮挡，准确高效的定位对自动驾驶汽车和移动机器人至关重要，因此需要一种轻量级、实时且鲁棒的定位方法。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了传统方法如RTAB-Map在长距离和嘈杂定位源下的局限性，包括内存消耗大和地图不一致问题。他们借鉴了拓扑SLAM的基本概念，将环境表示为图结构（节点为关键位置，边为可导航路径）。方法分为两阶段：先通过位置识别确定大致位置，再用扫描匹配细化估计。作者使用了MinkLoc3D进行位置识别，并改进了之前PRISM-TopoMap的扫描匹配方法，增加了专门针对原始激光雷达点云的路沿检测算法，以提高城市环境中的定位准确性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用紧凑的拓扑地图表示大型城市环境，结合位置识别和扫描匹配技术实现轻量级、准确的定位。整体流程包括：1) 地图创建：构建图结构，每个节点存储位置描述符和2D网格；2) 定位跟踪：维护当前在图中的位置和相对姿态，根据重叠度决定是否移动到相邻节点；3) 全局定位：使用MinkLoc3D进行位置识别，找到候选位置后通过扫描匹配（结合ORB特征、改进RANSAC和路沿检测）精确估计姿态；4) 路沿检测：通过平面拟合分割地面，检测边缘作为路沿特征，提高城市环境中的匹配准确性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 紧凑的城市规模定位表示，拓扑地图每个位置仅需几KB存储；2) 新颖的激光雷达扫描匹配算法，直接在原始扫描上操作；3) 原始的路沿检测算法，专门针对激光雷达点云设计。相比之前工作，PRISM-Loc显著减少了内存需求（传统方法可能需8GB，而此方法仅需20MB），提高了计算效率（定位时间0.3秒），增强了城市环境中的鲁棒性（通过路沿检测），并且相比其他方法如BEVPlace++在多个指标上表现更优。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'PRISM-Loc提出了一种轻量级、高效的定位方法，通过结合紧凑的拓扑地图表示、新颖的扫描匹配算法和原始的路沿检测技术，实现了在资源受限平台上的城市环境中高精度、实时的机器人定位。'}


### 论文摘要

We propose PRISM-Loc - a lightweight and robust approach for localization in large outdoor environments that combines a compact topological representation with a novel scan-matching and curb-detection module operating on raw LiDAR scans. The method is designed for resource-constrained platforms and emphasizes real-time performance and resilience to common urban sensing challenges. It provides accurate localization in compact topological maps using global place recognition and an original scan matching technique. Experiments on standard benchmarks and on an embedded platform demonstrate the effectiveness of our approach. Our method achieves a 99\% success rate on the large-scale ITLP-Campus dataset while running at 150 ms per localization and using a 20 MB map for localization. We highlight three main contributions: (1) a compact representation for city-scale localization; (2) a novel curb detection and scan matching pipeline operating directly on raw LiDAR points; (3) a thorough evaluation of our method with performance analysis.

---

## 3. 论文ID: 2512.19684v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19684v1.json'

---

## 4. Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface

**论文链接:** [http://arxiv.org/abs/2512.19402v1](http://arxiv.org/abs/2512.19402v1)

**作者:** Yujie Zhao, Hongwei Fan, Di Chen, Shengcong Chen, Liliang Chen, Xiaoqi Li, Guanghui Ren, Hao Dong

**发布时间:** 2025-12-22

### GPT解析

### 总结

Real2Edit2Real框架通过结合3D可编辑性与2D视觉数据生成新的操作演示，显著提高了数据效率，减少了实际数据收集的需求。

### 背景

机器人学习的最新进展由大规模数据集和强大的视觉运动策略架构推动，但策略的稳健性仍然受到收集多样化演示数据的显著成本限制，特别是在操作任务中的空间泛化方面。

### 目的

减少重复的数据收集，提出一个框架通过3D可编辑性与2D视觉数据之间的桥梁来生成新的演示。

### 方法

使用多视图RGB观察重建场景几何；在点云上进行深度可靠的3D编辑生成新的操作轨迹；几何校正机器人姿态恢复物理一致的深度；提出以深度为主要控制信号的多条件视频生成模型，合成空间增强的多视图操作视频。

### 主要发现

在四个实际操作任务上的实验表明，仅使用1-5个源演示生成的数据训练的策略可以匹配或优于使用50个实际演示训练的策略，数据效率提高了10-50倍；在高度和纹理编辑上的实验结果证明了该框架的灵活性和可扩展性。

### 结论

该框架有潜力作为一个统一的数据生成框架，有效解决机器人学习中数据收集成本高的问题。

### 翻译

机器人学习的最新进展由大规模数据集和强大的视觉运动策略架构推动，然而策略的稳健性仍然受到收集多样化演示数据的显著成本限制，特别是在操作任务中的空间泛化方面。为了减少重复的数据收集，我们提出了Real2Edit2Real，一个通过3D可编辑性与2D视觉数据之间的桥梁来生成新演示的框架。我们的方法首先使用具有度量尺度3D重建模型的多视图RGB观察重建场景几何。基于重建的几何，我们在点云上进行深度可靠的3D编辑，以生成新的操作轨迹，同时几何校正机器人姿态以恢复物理一致的深度，这作为合成新演示的可靠条件。最后，我们提出了一个以深度为主要控制信号的多条件视频生成模型，结合动作、边缘和射线图，来合成空间增强的多视图操作视频。在四个实际操作任务上的实验表明，仅使用1-5个源演示生成的数据训练的策略可以匹配或优于使用50个实际演示训练的策略，将数据效率提高了10-50倍。此外，在高度和纹理编辑上的实验结果证明了该框架的灵活性和可扩展性，表明其有潜力作为一个统一的数据生成框架。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人学习中的数据收集成本高、效率低的问题。具体来说，机器人学习需要大量多样化的演示数据，特别是对于空间泛化任务（操作任务中物体随机排列的场景），收集这些数据的成本非常高。这个问题在现实中很重要，因为它限制了机器人学习策略的鲁棒性和可扩展性，使得在实际应用中难以快速部署高性能的机器人系统。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者认为3D数据比2D图像具有更大的编辑灵活性，因此首先对源演示进行几何重建。为了提高重建在机器人场景中的准确性，他们提出结合真实和模拟数据的混合训练范式。方法设计包含三个主要组件：几何重建、空间编辑和视频生成。作者确实借鉴了现有工作，如使用VGGT作为基础模型重建几何，受DemoGen启发分解点云演示为运动段和技能段，以及参考GE-Sim等使用Transformer架构进行视频生成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过3D可编辑性连接2D视觉数据，实现可扩展的演示生成，并将深度图作为3D模态和2D观察之间的自然界面。整体流程分为三步：1)尺度感知的几何重建：使用混合训练范式从多视图RGB观测重建场景几何；2)深度可靠的空间编辑：基于点云编辑和运动规划合成新轨迹，同时校正机器人姿态以获得物理一致的深度图；3)3D控制视频生成：以深度为主要控制信号，结合边缘、动作和射线图生成多视图一致的操作视频。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：不依赖仿真引擎直接从RGB生成数据；尺度感知的几何重建；深度可靠的空间编辑；3D控制视频生成；以及10-50倍的数据效率提升。相比之前工作的不同之处在于：与MimicGen家族不同，避免了Sim2Real差距；与DemoGen不同，兼容多视图RGB相机设置；与Real2Render2Real和RoboSplat不同，不需要密集图像捕获；与基于视频生成的工作不同，不仅增强视觉方面，还增加了物体空间分布和机器人轨迹的多样性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Real2Edit2Real通过结合3D可编辑性和2D视觉数据，实现了高效、真实的多视图机器人演示生成，将数据效率提高了10-50倍，同时保持了视觉真实性和正确的物理交互。'}


### 论文摘要

Recent progress in robot learning has been driven by large-scale datasets and powerful visuomotor policy architectures, yet policy robustness remains limited by the substantial cost of collecting diverse demonstrations, particularly for spatial generalization in manipulation tasks. To reduce repetitive data collection, we present Real2Edit2Real, a framework that generates new demonstrations by bridging 3D editability with 2D visual data through a 3D control interface. Our approach first reconstructs scene geometry from multi-view RGB observations with a metric-scale 3D reconstruction model. Based on the reconstructed geometry, we perform depth-reliable 3D editing on point clouds to generate new manipulation trajectories while geometrically correcting the robot poses to recover physically consistent depth, which serves as a reliable condition for synthesizing new demonstrations. Finally, we propose a multi-conditional video generation model guided by depth as the primary control signal, together with action, edge, and ray maps, to synthesize spatially augmented multi-view manipulation videos. Experiments on four real-world manipulation tasks demonstrate that policies trained on data generated from only 1-5 source demonstrations can match or outperform those trained on 50 real-world demonstrations, improving data efficiency by up to 10-50x. Moreover, experimental results on height and texture editing demonstrate the framework's flexibility and extensibility, indicating its potential to serve as a unified data generation framework.

---

## 5. Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation

**论文链接:** [http://arxiv.org/abs/2512.19088v1](http://arxiv.org/abs/2512.19088v1)

**作者:** Khanh Nguyen, Dasith de Silva Edirimuni, Ghulam Mubashar Hassan, Ajmal Mian

**发布时间:** 2025-12-22

**备注:** Accepted to AAAI 2026 Workshop on New Frontiers in Information Retrieval

### GPT解析

### 总结

论文提出了一种改进的3D物体定位和检索方法，解决了现有方法的计算效率问题和对稀有物体的泛化问题，通过结合2D开放词汇检测器与3D处理，实现了高效准确的结果。

### 背景

从场景级点云中定位和检索物体是一个具有挑战性的问题，在机器人和增强现实领域有广泛应用。现有方法依赖SAM和CLIP从点云伴随的图像生成和分类3D实例掩码，导致大量计算开销和处理速度慢，限制了在现实场景中的部署。

### 目的

解决现有方法的计算效率问题，提高对不常见物体类别的泛化能力，实现从开放文本查询中快速准确地检索稀有实例。

### 方法

提出一种从RGB图像生成3D实例掩码的方法，使用2D开放词汇检测器指导生成过程，继承2D检测器识别新物体的能力，同时保持高效的分类性能。

### 主要发现

Open-YOLO 3D通过使用实时2D检测器减轻计算负担，直接从点云生成类别无关的掩码，消除了对SAM和CLIP的需求，显著减少了推理时间；然而，该方法通常无法泛化到3D训练数据中很少出现的物体类别。

### 结论

提出的方法能够处理新颖物体，同时保持高效的分类性能，实现了从开放文本查询中快速准确地检索稀有实例，代码将在https://github.com/ndkhanh360/BoxOVIS上提供。

### 翻译

在场景级点云中定位和检索物体是一个具有挑战性的问题，在机器人和增强现实领域有广泛应用。此任务通常被表述为开放词汇3D实例分割。尽管最近的方法表现出强大的性能，但它们严重依赖SAM和CLIP从点云伴随的图像生成和分类3D实例掩码，导致大量计算开销和缓慢的处理速度，限制了它们在现实环境中的部署。Open-YOLO 3D通过使用实时2D检测器来分类由预训练3D分割器直接从点云生成的类别无关掩码，减轻了这一问题，消除了对SAM和CLIP的需求，并显著减少了推理时间。然而，Open-YOLO 3D通常无法泛化到在3D训练数据中很少出现的物体类别。在本文中，我们提出了一种方法，通过2D开放词汇检测器指导，从RGB图像生成新物体的3D实例掩码。我们的方法继承了2D检测器识别新物体的能力，同时保持高效的分类，能够从开放文本查询中快速准确地检索稀有实例。我们的代码将在https://github.com/ndkhanh360/BoxOVIS上提供。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决从3D场景中高效且准确地检索对象的问题，特别是针对在训练数据中不常见的罕见对象。这个问题在机器人和增强现实等领域非常重要，因为现有方法要么计算开销大（依赖SAM和CLIP导致处理缓慢），要么对罕见对象识别能力差，限制了它们在实际应用中的部署。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有开放词汇3D实例分割方法的局限性：计算开销大和对罕见类别泛化能力差。他们借鉴了Open-YOLO 3D的效率思路和Mask3D的点基础掩码生成方法，同时引入YOLO-World作为2D检测器。作者的创新在于利用2D检测器的预测来引导3D点云中罕见对象的掩码生成，避免了计算密集型的SAM模型，同时保留了2D检测器对罕见类别的识别能力。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用2D开放词汇检测器的预测来引导3D点云中罕见对象的掩码生成，同时保持高效的分类能力。整体流程包括：1) 使用基于图的分割方法处理点云获取超点；2) 使用Mask3D生成点基础掩码；3) 使用YOLO-World在RGB图像上生成边界框；4) 将2D边界框提升到3D并提取超点形成新掩码；5) 通过投影3D实例到2D标签图进行分类，匹配输入查询。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 框引导的RGBD基础掩码生成，利用2D检测器预测来形成罕见对象掩码；2) 高效的分类方法，避免使用计算密集型的CLIP特征提取；3) 对罕见类别的改进性能。相比Open-YOLO 3D等之前的工作，本文方法不完全依赖3D分割器生成候选对象，而是利用2D检测器来检测和形成罕见对象的掩码，显著提高了对罕见类别的泛化能力，同时保持了较高的处理效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种结合2D检测器识别能力和3D点云几何信息的开放词汇3D实例分割方法，实现了高效且能够检测罕见对象的3D场景检索系统。'}


### 论文摘要

Locating and retrieving objects from scene-level point clouds is a challenging problem with broad applications in robotics and augmented reality. This task is commonly formulated as open-vocabulary 3D instance segmentation. Although recent methods demonstrate strong performance, they depend heavily on SAM and CLIP to generate and classify 3D instance masks from images accompanying the point cloud, leading to substantial computational overhead and slow processing that limit their deployment in real-world settings. Open-YOLO 3D alleviates this issue by using a real-time 2D detector to classify class-agnostic masks produced directly from the point cloud by a pretrained 3D segmenter, eliminating the need for SAM and CLIP and significantly reducing inference time. However, Open-YOLO 3D often fails to generalize to object categories that appear infrequently in the 3D training data. In this paper, we propose a method that generates 3D instance masks for novel objects from RGB images guided by a 2D open-vocabulary detector. Our approach inherits the 2D detector's ability to recognize novel objects while maintaining efficient classification, enabling fast and accurate retrieval of rare instances from open-ended text queries. Our code will be made available at https://github.com/ndkhanh360/BoxOVIS.

---

## 6. ICP-4D: Bridging Iterative Closest Point and LiDAR Panoptic Segmentation

**论文链接:** [http://arxiv.org/abs/2512.18991v1](http://arxiv.org/abs/2512.18991v1)

**作者:** Gyeongrok Oh, Youngdong Jang, Jonghyun Choi, Suk-Ju Kang, Guang Lin, Sangpil Kim

**发布时间:** 2025-12-22

### GPT解析

### 总结

论文提出了ICP-4D框架，通过几何关系统一空间和时间推理，无需训练即可实现4D LiDAR全景分割，使用ICP算法和Sinkhorn软匹配实现实例关联，在多个数据集上优于现有方法。

### 背景

现有4D LiDAR全景分割方法通常需要大型叠加点云训练深度神经网络或设计专门的实例关联模块，这些方法执行冗余的点处理导致计算成本高昂，同时忽略了原始点云中的几何先验。

### 目的

开发一个简单有效的无需训练的框架，通过实例级点集之间的几何关系统一空间和时间推理，避免冗余计算并利用点云中的几何先验。

### 方法

应用迭代最近点(ICP)算法通过估计变换对齐源和目标点集关联时间上一致的实例；引入基于Sinkhorn的软匹配稳定嘈杂预测下的关联；设计考虑静态、动态和缺失三种实例类型的管道，提供计算效率和遮挡感知匹配能力。

### 主要发现

在SemanticKITTI和nuScenes数据集上的广泛实验表明，该方法一致性地优于最先进方法，无需额外训练或额外的点云输入即可实现优异性能。

### 结论

ICP-4D框架通过利用原始点云中的几何先验关系，解决了现有4D LiDAR全景分割方法中的冗余计算问题，提供了一个简单、高效且无需训练的解决方案。

### 翻译

4D LiDAR全景分割的主导范式通常需要使用大型叠加点云训练深度神经网络或设计专门的实例关联模块。然而，这些方法执行冗余的点处理，导致计算成本高昂，同时仍然忽略了原始点云中固有的丰富几何先验。为此，我们引入了ICP-4D，一个简单而有效的无需训练的框架，它通过实例级点集之间的几何关系统一了空间和时间推理。具体来说，我们应用迭代最近点(ICP)算法，通过估计的变换直接对齐源和目标点集，从而关联时间上一致的实例。为了在嘈杂的实例预测下稳定关联，我们引入了基于Sinkhorn的软匹配。这利用了底层实例分布来获得准确的点对应关系，从而实现鲁棒的几何对齐。此外，我们精心设计的管道考虑了三种实例类型（静态、动态和缺失），提供了计算效率和遮挡感知匹配能力。我们在SemanticKITTI和nuScenes上的广泛实验表明，我们的方法一致性地优于最先进的方法，即使没有额外的训练或额外的点云输入。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决4D LiDAR全景分割中的实例关联问题，确保不同时间帧中相同物体能保持一致的ID标识。这个问题在自动驾驶领域至关重要，因为系统需要精确感知动态3D环境，而LiDAR分割是实现全面点理解的关键技术，确保时间连续性对于可靠的环境感知和物体跟踪至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者分析了现有三种主要方法（IoU关联、查询传播关联、检测与跟踪关联）的局限性，发现它们存在计算开销大、误差累积和依赖训练等问题。作者思考能否仅利用3D全景网络实现高效实例关联，最终借鉴了ICP点云配准算法和Sinkhorn最优传输理论，设计了无需训练的ICP-4D框架。该方法将ICP首次应用于4D LiDAR全景分割，同时引入了软匹配机制处理噪声预测问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过实例级点集之间的几何关系统一空间和时间推理，利用ICP算法直接对齐连续扫描中的实例点集，并结合基于Sinkhorn的软匹配增强鲁棒性。整体流程包括：1)使用预训练3D分割模型获取点云的实例和语义预测；2)将实例分为静态、动态和缺失三类处理；3)静态实例基于中心点和协方差进行快速匹配；4)动态实例使用ICP和Sinkhorn软匹配进行精确对齐；5)缺失实例通过内存银行进行恢复；6)最后通过匈牙利算法优化匹配结果。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个无需训练的4D LiDAR全景分割框架；2)基于Sinkhorn的软匹配策略增强点对应的鲁棒性；3)针对静态、动态和缺失实例的三种状态条件解决方案；4)内存银行机制处理遮挡问题。相比之前的工作，ICP-4D不依赖大规模点云叠加或专门训练模块，直接利用点云几何先验信息，在单扫描设置下实现了更优性能，同时显著降低了计算复杂度（内存需求减少60.7%-79.4%，运行时间提高26.1%-45.6%）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ICP-4D通过结合迭代最近点算法和基于Sinkhorn的软匹配策略，实现了无需训练的高效4D LiDAR全景分割，显著提升了时间连续性并平衡了计算效率与性能。'}


### 论文摘要

Dominant paradigms for 4D LiDAR panoptic segmentation are usually required to train deep neural networks with large superimposed point clouds or design dedicated modules for instance association. However, these approaches perform redundant point processing and consequently become computationally expensive, yet still overlook the rich geometric priors inherently provided by raw point clouds. To this end, we introduce ICP-4D, a simple yet effective training-free framework that unifies spatial and temporal reasoning through geometric relations among instance-level point sets. Specifically, we apply the Iterative Closest Point (ICP) algorithm to directly associate temporally consistent instances by aligning the source and target point sets through the estimated transformation. To stabilize association under noisy instance predictions, we introduce a Sinkhorn-based soft matching. This exploits the underlying instance distribution to obtain accurate point-wise correspondences, resulting in robust geometric alignment. Furthermore, our carefully designed pipeline, which considers three instance types-static, dynamic, and missing-offers computational efficiency and occlusion-aware matching. Our extensive experiments across both SemanticKITTI and panoptic nuScenes demonstrate that our method consistently outperforms state-of-the-art approaches, even without additional training or extra point cloud inputs.

---

## 7. E-RGB-D: Real-Time Event-Based Perception with Structured Light

**论文链接:** [http://arxiv.org/abs/2512.18429v1](http://arxiv.org/abs/2512.18429v1)

**作者:** Seyed Ehsan Marjani Bajestani, Giovanni Beltrame

**发布时间:** 2025-12-20

### GPT解析

### 总结

本文提出了一种结合数字光处理投影仪和事件相机的新型RGB-D传感方法，实现了高速彩色深度感知，解决了传统单色事件相机无法检测静态或缓慢移动物体以及缺乏颜色信息的问题。

### 背景

事件相机作为受生物启发的传感器，能异步报告像素亮度变化，具有高动态范围、高时间分辨率、低功耗和计算简单等优点。然而，传统单色事件相机在检测静态或缓慢移动物体方面存在局限性，且缺乏某些应用所需的颜色信息。

### 目的

为了解决传统单色事件相机的局限性，作者提出集成数字光处理投影仪，形成主动结构光用于RGB-D传感，以实现颜色和深度的同时检测。

### 方法

作者结合事件相机和基于投影技术的优势，通过动态投影调整优化带宽，确保选择性颜色数据获取，产生不牺牲空间分辨率的彩色点云。这种集成通过商业TI LightCrafter 4500投影仪和单目单色事件相机实现。

### 主要发现

通过该方法，实现了相当于1400 fps的颜色检测速度和4 kHz的像素深度检测速度，显著推进了从机器人技术到3D重建方法等不同领域的计算机视觉发展。

### 结论

这种集成不仅实现了无帧RGB-D传感应用，还取得了显著的性能里程碑，为计算机视觉领域带来了重大进步。

### 翻译

事件相机作为一种受生物启发的传感器，能够异步报告像素亮度变化，在视觉感知方面提供了无与伦比的速度和效率。尽管它们具有高动态范围、高时间分辨率、低功耗和计算简单等优点，传统的单色事件相机在检测静态或缓慢移动物体方面存在局限性，并且缺乏某些应用必需的颜色信息。为了应对这些挑战，我们提出了一种新颖的方法，集成数字光处理投影仪，形成主动结构光用于RGB-D传感。通过结合事件相机和基于投影技术的优势，我们的方法能够分别检测每个像素的颜色和深度。动态投影调整优化带宽，确保选择性获取颜色数据，产生不牺牲空间分辨率的彩色点云。这种集成通过商业TI LightCrafter 4500投影仪和单目单色事件相机实现，不仅实现了无帧RGB-D传感应用，还取得了显著的性能里程碑。通过我们的方法，我们实现了相当于1400 fps的颜色检测速度和4 kHz的像素深度检测速度，显著推进了从机器人技术到3D重建方法等不同领域的计算机视觉领域。我们的代码已公开：https://github.com/MISTLab/event_based_rgbd_ros


### 论文摘要

Event-based cameras (ECs) have emerged as bio-inspired sensors that report pixel brightness changes asynchronously, offering unmatched speed and efficiency in vision sensing. Despite their high dynamic range, temporal resolution, low power consumption, and computational simplicity, traditional monochrome ECs face limitations in detecting static or slowly moving objects and lack color information essential for certain applications. To address these challenges, we present a novel approach that integrates a Digital Light Processing (DLP) projector, forming Active Structured Light (ASL) for RGB-D sensing. By combining the benefits of ECs and projection-based techniques, our method enables the detection of color and the depth of each pixel separately. Dynamic projection adjustments optimize bandwidth, ensuring selective color data acquisition and yielding colorful point clouds without sacrificing spatial resolution. This integration, facilitated by a commercial TI LightCrafter 4500 projector and a monocular monochrome EC, not only enables frameless RGB-D sensing applications but also achieves remarkable performance milestones. With our approach, we achieved a color detection speed equivalent to 1400 fps and 4 kHz of pixel depth detection, significantly advancing the realm of computer vision across diverse fields from robotics to 3D reconstruction methods. Our code is publicly available: https://github.com/MISTLab/event_based_rgbd_ros

---

## 8. Chorus: Multi-Teacher Pretraining for Holistic 3D Gaussian Scene Encoding

**论文链接:** [http://arxiv.org/abs/2512.17817v2](http://arxiv.org/abs/2512.17817v2)

**作者:** Yue Li, Qi Ma, Runyi Yang, Mengjiao Ma, Bin Ren, Nikola Popovic, Nicu Sebe, Theo Gevers, Luc Van Gool, Danda Pani Paudel, Martin R. Oswald

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文提出了Chorus，一个多教师预训练框架，用于学习3D高斯溅射场景编码器，通过从2D基础模型中提炼互补信号，实现了从高保真场景表示中提取丰富通用特征的能力。

### 背景

3D高斯溅射(3DGS)已成为一种高保真场景表示方法，但直接从其基本元素编码丰富、通用特征的研究仍然不足。

### 目的

引入Chorus框架，学习一个整体的3D高斯溅射场景编码器，通过从2D基础模型中提炼互补信号来填补这一研究空白。

### 方法

Chorus采用共享的3D编码器和教师特定的投影器，从语言对齐、通用和对象感知的教师中学习，鼓励捕获从高级语义到细粒度结构的共享嵌入空间。

### 主要发现

Chorus在多种任务上表现优异，包括开放词汇语义和实例分割、线性和解码器探测以及数据高效监督；在仅支持点云的基准测试上，使用39.9倍更少的训练场景就优于点云基线；提出的渲染和提炼适应方法促进了域外微调。

### 结论

Chorus是一个有效的多教师预训练框架，能够从3DGS中提取丰富的特征，在各种任务上表现出色，并且在点云处理上也显示出显著优势，代码和模型将在发表后发布。

### 翻译

虽然3DGS已成为一种高保真场景表示方法，但直接从其基本元素编码丰富、通用特征的研究仍然不足。我们通过引入Chorus解决了这一差距，这是一个多教师预训练框架，通过从2D基础模型中提炼互补信号来学习一个整体的3D高斯溅射场景编码器。Chorus采用共享的3D编码器和教师特定的投影器，从语言对齐、通用和对象感知的教师中学习，鼓励捕获从高级语义到细粒度结构的共享嵌入空间。我们在广泛的任务上评估Chorus：开放词汇语义和实例分割、线性和解码器探测，以及数据高效监督。除了3DGS外，我们还通过仅使用高斯中心、颜色和估计法线作为输入预训练一个变体，在几个仅支持点云的基准测试上测试了Chorus。有趣的是，该编码器显示出强大的迁移能力，使用39.9倍更少的训练场景就优于点云基线。最后，我们提出了一种渲染和提炼的适应方法，促进域外微调。我们的代码和模型将在发表后发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从3D高斯溅射(3DGS)表示中直接提取丰富、通用的特征问题。这很重要，因为虽然3DGS能提供高保真场景表示，但直接从中提取可转移的通用特征仍处于探索阶段，而这类通用特征对于实现多样化的3D场景理解任务(如语义分割、实例分割、视觉问答等)至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者借鉴了'lift-then-align'范式(如SceneSplat工作)，将2D特征提升到3D空间。同时融合了自监督学习和知识蒸馏的思想，特别是多教师知识蒸馏在2D领域的成功应用。作者注意到现有方法(如SceneSplat)主要关注语义信息，忽略了更广泛的场景理解能力，因此设计了多教师框架，整合语言对齐(SigLIP)、通用视觉特征(DINO)和对象感知(PE-Spatial)三种互补信号，以捕捉从高级语义到精细空间结构的多种信息。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过多教师预训练框架，学习一个通用的前馈3DGS场景编码器，将互补的2D基础模型信号整合到共享的3D嵌入空间。整体流程包括：1)使用3DGS场景数据并从多角度渲染；2)将2D教师特征提升到3D高斯空间并标准化；3)训练共享3D编码器和每个教师特定的轻量投影头，应用匹配损失和对比损失；4)采用分阶段预训练策略；5)设计渲染和蒸馏适应方法实现新域适应；6)应用3DGS感知的增强技术提高鲁棒性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次将多教师知识蒸馏应用于3DGS场景编码；2)设计统一的3D场景编码器，实现高度结构化和可转移的特征表示；3)提出渲染和蒸馏适应策略，简化新域适应流程；4)开发3DGS感知的数据增强方法。相比之前工作，Chorus超越了单教师方法(如SceneSplat)的语义局限性，实现了更广泛的场景理解能力；相比点云预训练方法，Chorus使用更少训练数据(8.32-39.9倍)实现了更强性能；相比传统适应方法，Chorus避免了昂贵的3D伪标签预处理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Chorus通过多教师预训练框架，首次实现了从3D高斯溅射中直接提取通用、可转移的3D场景特征，在各种3D场景理解任务上达到最先进性能，同时展示卓越的数据效率和适应性。'}


### 论文摘要

While 3DGS has emerged as a high-fidelity scene representation, encoding rich, general-purpose features directly from its primitives remains under-explored. We address this gap by introducing Chorus, a multi-teacher pretraining framework that learns a holistic feed-forward 3D Gaussian Splatting (3DGS) scene encoder by distilling complementary signals from 2D foundation models. Chorus employs a shared 3D encoder and teacher-specific projectors to learn from language-aligned, generalist, and object-aware teachers, encouraging a shared embedding space that captures signals from high-level semantics to fine-grained structure.   We evaluate Chorus on a wide range of tasks: open-vocabulary semantic and instance segmentation, linear and decoder probing, as well as data-efficient supervision. Besides 3DGS, we also test Chorus on several benchmarks that only support point clouds by pretraining a variant using only Gaussians' centers, colors, estimated normals as inputs. Interestingly, this encoder shows strong transfer and outperforms the point clouds baseline while using 39.9 times fewer training scenes. Finally, we propose a render-and-distill adaptation that facilitates out-of-domain finetuning. Our code and model will be released upon publication.

---

## 9. CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives

**论文链接:** [http://arxiv.org/abs/2512.14696v2](http://arxiv.org/abs/2512.14696v2)

**作者:** Zihan Wang, Jiashun Wang, Jeff Tan, Yiwen Zhao, Jessica Hodgins, Shubham Tulsiani, Deva Ramanan

**发布时间:** 2025-12-16

**备注:** Project page: https://crisp-real2sim.github.io/CRISP-Real2Sim/

### GPT解析

### 总结

CRISP是一种从单目视频中恢复可模拟人体运动和场景几何的方法，通过平面基元拟合和人体-场景接触建模，显著提高了运动跟踪的成功率并加速了模拟过程。

### 背景

先前的人体-场景联合重建工作要么依赖数据驱动的先验和无物理循环的联合优化，要么恢复带有噪声的几何形状，导致涉及场景交互的运动跟踪策略失败。

### 目的

开发一种能够从单目视频中恢复可模拟人体运动和场景几何的方法，解决现有方法中的噪声和物理一致性问题。

### 方法

通过基于深度、法向量和流的简单聚类流程，将平面基元拟合到场景的点云重建中；利用人体-场景接触建模重建可能被交互遮挡的场景几何；通过强化学习使用重建结果驱动人形控制器以确保物理合理性。

### 主要发现

在以人为中心的视频基准测试上，将运动跟踪失败率从55.2%降低到6.9%；RL模拟吞吐量提高43%；在多种野外视频上验证了方法的有效性，包括 casually-captured视频、互联网视频和Sora生成的视频。

### 结论

CRISP能够大规模生成物理有效的人体运动和交互环境，大大推进了机器人和AR/VR领域的现实到模拟应用。

### 翻译

我们介绍了CRISP，一种从单目视频中恢复可模拟人体运动和场景几何的方法。先前关于人体-场景联合重建的工作依赖于数据驱动的先验和无物理循环的联合优化，或者恢复带有噪声的几何形状，导致涉及场景交互的运动跟踪策略失败。相比之下，我们的关键洞见是通过将平面基元拟合到场景的点云重建中，恢复凸起、干净且可模拟的几何，这通过基于深度、法向量和流的简单聚类流程实现。为了重建交互过程中可能被遮挡的场景几何，我们利用人体-场景接触建模（例如，我们使用人体姿势重建被遮挡的椅子座位）。最后，我们通过强化学习使用重建的人体和场景驱动人形控制器，确保人体和场景重建在物理上是合理的。我们的方法在以人为中心的视频基准测试(EMDB, PROX)上，将运动跟踪失败率从55.2%降低到6.9%，同时使RL模拟吞吐量提高43%。我们在各种野外视频上进一步验证了该方法，包括 casually-captured视频、互联网视频，甚至是Sora生成的视频。这证明了CRISP能够大规模生成物理有效的人体运动和交互环境，大大推进了机器人和AR/VR领域的现实到模拟应用。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从单目视频中恢复可模拟的3D人体运动和场景几何的问题，特别是在人体与场景交互的情况下。这个问题很重要，因为真正理解视频中的行为需要物理层面的理解，人体与环境的交互是日常生活中的常见场景（如坐在椅子上、爬楼梯等），而现有的重建方法在处理视差和遮挡时表现不佳，重建结果可能包含噪声和伪影，导致物理模拟失败。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有方法依赖数据驱动的先验和联合优化但没有物理循环，重建的几何结构有噪声导致交互失败。他们的设计思路是通过平面基元拟合点云来获得可模拟的几何，利用人体-场景接触建模重建被遮挡的场景，最后用强化学习确保物理合理性。他们借鉴了MegaSAM进行相机姿态估计，MoGe改进几何质量，GVHMR估计人体姿态，InteractVLM预测接触，MaskedMimic设计观察行动模型，以及使用Isaac Gym和PPO进行模拟和训练。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将场景分解为一小组（约50个）凸平面基元来解决噪声问题，利用人体-场景接触建模重建被遮挡的场景，并用强化学习确保物理合理性。整体流程：1)初始化：使用MegaSAM和MoGe推断相机参数和场景点云，GVHMR估计人体姿态；2)平面基元拟合：通过聚类算法从点云中提取平面基元；3)接触引导场景完成：使用InteractVLM预测接触并完成被遮挡场景；4)物理运动跟踪：用强化学习训练控制策略在模拟环境中跟踪重建的人体运动。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)平面基元表示：将场景分解为少量凸平面基元生成轻量级可模拟重建；2)接触引导场景完成：利用人体-场景接触建模重建被遮挡几何；3)物理验证：用强化学习确保重建的物理合理性。相比之前工作的不同：与VideoMimic相比，CRISP提供更准确的人体、场景和接触建模，RL成功率从44.8%提高到93.1%，模拟吞吐量从16K提高到23K FPS；与其他几何重建方法相比，避免了过平滑和重复结构；与其他人体运动恢复方法相比，在EMDB数据集上实现了更低的姿态误差。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CRISP提出了一种从单目视频中重建可模拟3D人体运动和场景几何的创新方法，通过平面基元表示、接触引导的场景完成和物理验证，显著提高了人体-场景交互的重建质量和模拟稳定性。'}


### 论文摘要

We introduce CRISP, a method that recovers simulatable human motion and scene geometry from monocular video. Prior work on joint human-scene reconstruction relies on data-driven priors and joint optimization with no physics in the loop, or recovers noisy geometry with artifacts that cause motion tracking policies with scene interactions to fail. In contrast, our key insight is to recover convex, clean, and simulation-ready geometry by fitting planar primitives to a point cloud reconstruction of the scene, via a simple clustering pipeline over depth, normals, and flow. To reconstruct scene geometry that might be occluded during interactions, we make use of human-scene contact modeling (e.g., we use human posture to reconstruct the occluded seat of a chair). Finally, we ensure that human and scene reconstructions are physically-plausible by using them to drive a humanoid controller via reinforcement learning. Our approach reduces motion tracking failure rates from 55.2\% to 6.9\% on human-centric video benchmarks (EMDB, PROX), while delivering a 43\% faster RL simulation throughput. We further validate it on in-the-wild videos including casually-captured videos, Internet videos, and even Sora-generated videos. This demonstrates CRISP's ability to generate physically-valid human motion and interaction environments at scale, greatly advancing real-to-sim applications for robotics and AR/VR.

---

## 10. ALIGN: Advanced Query Initialization with LiDAR-Image Guidance for Occlusion-Robust 3D Object Detection

**论文链接:** [http://arxiv.org/abs/2512.18187v1](http://arxiv.org/abs/2512.18187v1)

**作者:** Janghyun Baek, Mincheol Chang, Seokha Moon, Seung Joon Lee, Jinkyu Kim

**发布时间:** 2025-12-20

**备注:** 12 pages, 6 figures

### GPT解析

### 总结

这篇论文提出了ALIGN方法，一种创新的3D目标检测查询初始化策略，通过三个关键组件解决了遮挡和拥挤物体检测的问题，并在nuScenes基准上取得了显著的性能提升。

### 背景

基于查询的3D目标检测方法使用相机和LiDAR输入已显示出强大性能，但现有的查询初始化策略（如随机采样或基于BEV热图的采样）往往导致查询使用效率低下和准确性降低，特别是在被遮挡或拥挤物体的检测中。

### 目的

解决现有查询初始化策略的局限性，提出一种对遮挡具有鲁棒性且具有物体感知能力的查询初始化方法。

### 方法

提出了ALIGN（Advanced query initialization with LiDAR and Image GuidaNce）方法，包含三个关键组件：(i)遮挡感知中心估计(OCE)：集成LiDAR几何和图像语义准确估计物体中心；(ii)自适应邻域采样(ANS)：从LiDAR聚类生成物体候选，并通过在其周围采样空间和语义对齐的点来补充每个物体；(iii)动态查询平衡(DQB)：自适应地平衡前景和背景区域之间的查询。

### 主要发现

在nuScenes基准上的广泛实验表明，ALIGN持续提高了多个最先进检测器的性能，最多可实现+0.9 mAP和+1.2 NDS的增益，特别是在具有遮挡或密集人群的挑战性场景中表现更好。

### 结论

ALIGN方法能有效解决现有查询初始化策略的问题，代码将在发表后公开。

### 翻译

最近的基于查询的3D目标检测方法使用相机和LiDAR输入已显示出强大的性能，但现有的查询初始化策略，如随机采样或基于BEV热图的采样，往往导致查询使用效率低下和准确性降低，特别是对于被遮挡或拥挤的物体。为了解决这一限制，我们提出了ALIGN（使用LiDAR和图像引导的高级查询初始化），一种新颖的对遮挡具有鲁棒性、具有物体感知能力的查询初始化方法。我们的模型包含三个关键组件：(i)遮挡感知中心估计(OCE)，它集成LiDAR几何和图像语义来准确估计物体中心；(ii)自适应邻域采样(ANS)，它从LiDAR聚类生成物体候选，并通过在其周围采样空间和语义对齐的点来补充每个物体；(iii)动态查询平衡(DQB)，它自适应地平衡前景和背景区域之间的查询。我们在nuScenes基准上的广泛实验表明，ALIGN持续提高了多个最先进检测器的性能，最多实现+0.9 mAP和+1.2 NDS的增益，特别是在具有遮挡或密集人群的挑战性场景中。我们的代码将在发表后公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决基于查询的3D目标检测中查询初始化策略的不足问题。现有方法（随机采样或BEV热图采样）在处理被遮挡、拥挤或小型物体时表现不佳。这个问题在现实中非常重要，因为3D目标检测是自动驾驶和机器人的核心能力，准确识别被遮挡物体对于确保安全导航至关重要，而这些场景在复杂的城市环境中非常常见。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有查询初始化方法的局限性，指出它们要么完全随机分布查询不考虑物体相关性，要么仅依赖热图信息而未充分利用激光雷达几何和图像语义信息。作者设计了一种物体感知的查询初始化策略，从估计的物体中心附近采样查询。该方法借鉴了DETR等查询检测框架，结合了图像分割和激光雷达点云投影技术（如OCE模块中的单应性估计），以及Deformable DETR的关键点采样思想（如ANS模块中的邻域采样），同时创新地引入了类特定的深度补偿和动态查询平衡机制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过结合激光雷达的几何信息和图像的语义信息，实现一种物体感知的查询初始化策略，特别是在处理被遮挡物体时更加鲁棒。整体流程包含三个主要组件：(1)遮挡感知中心估计(OCE)：将激光雷达点投影到图像，利用分割掩码和单应性变换估计物体中心，并应用类特定的深度补偿；(2)自适应邻域采样(ANS)：通过DBSCAN聚类识别潜在物体，并在每个聚类核心周围采样语义对齐的点；(3)动态查询平衡(DQB)：根据场景复杂度自适应平衡前景和背景查询分配。最终将生成的查询编码并传递给多模态Transformer解码器进行目标检测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：(1)提出物体感知查询初始化策略，从估计的物体中心附近采样查询；(2)设计三个关键模块：OCE结合激光雷达几何和图像语义估计物体中心，ANS通过聚类和邻域采样补充被遮挡物体，DQB平衡前景和背景查询；(3)在查询初始化阶段而非特征融合阶段有效利用多模态信息。相比之前工作，不同之处在于：与随机初始化相比，提供了关键区域的密集覆盖和更高的查询效率；与基于热图的初始化相比，不依赖可能漂移或消失的热图峰值，即使在复杂场景中也能提供更可靠的定位；与其他多模态方法相比，在查询初始化阶段就结合了几何和语义线索。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'ALIGN提出了一种结合激光雷达几何和图像语义的物体感知查询初始化框架，通过三个关键模块显著提高了被遮挡和密集场景下的3D目标检测性能，同时保持了与现有检测器的兼容性。'}


### 论文摘要

Recent query-based 3D object detection methods using camera and LiDAR inputs have shown strong performance, but existing query initialization strategies,such as random sampling or BEV heatmap-based sampling, often result in inefficient query usage and reduced accuracy, particularly for occluded or crowded objects. To address this limitation, we propose ALIGN (Advanced query initialization with LiDAR and Image GuidaNce), a novel approach for occlusion-robust, object-aware query initialization. Our model consists of three key components: (i) Occlusion-aware Center Estimation (OCE), which integrates LiDAR geometry and image semantics to estimate object centers accurately (ii) Adaptive Neighbor Sampling (ANS), which generates object candidates from LiDAR clustering and supplements each object by sampling spatially and semantically aligned points around it and (iii) Dynamic Query Balancing (DQB), which adaptively balances queries between foreground and background regions. Our extensive experiments on the nuScenes benchmark demonstrate that ALIGN consistently improves performance across multiple state-of-the-art detectors, achieving gains of up to +0.9 mAP and +1.2 NDS, particularly in challenging scenes with occlusions or dense crowds. Our code will be publicly available upon publication.

---

## 11. 论文ID: 2512.19687v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19687v1.json'

---

## 12. AMap: Distilling Future Priors for Ahead-Aware Online HD Map Construction

**论文链接:** [http://arxiv.org/abs/2512.19150v1](http://arxiv.org/abs/2512.19150v1)

**作者:** Ruikai Li, Xinrun Li, Mengwei Xie, Hao Shan, Shoumeng Qiu, Xinyuan Chang, Yizhe Fan, Feng Xiong, Han Jiang, Yilong Ren, Haiyang Yu, Mu Xu, Yang Long, Varun Ojha, Zhiyong Cui

**发布时间:** 2025-12-22

**备注:** 19 pages, 11 figures

### GPT解析

### 总结

本研究提出了一种名为AMap的新型在线高清地图构建框架，解决了现有方法'空间向后看'的安全缺陷，通过'从未来蒸馏'范式使模型具备前瞻性感知能力，在不增加推理时间成本的情况下显著提升了前方区域的感知准确性。

### 背景

在线高清地图构建对自动驾驶至关重要。现有方法主要利用历史时间融合来提高性能，但这些方法存在一个关键的安全缺陷：它们本质上是'空间向后看'的，主要增强已通过区域的地图重建，对前方未见的道路改进有限。分析显示，向后感知错误通常可以容忍，但前方区域的不准确性会直接导致危险的驾驶操作。

### 目的

本研究旨在解决现有在线高清地图方法的安全缺陷，特别关注前方区域的感知准确性，以提高自动驾驶系统的安全性。

### 方法

研究提出了AMap框架，采用'从未来蒸馏'范式，让拥有未来上下文访问权限的教师模型指导仅限于当前帧的轻量级学生模型。技术上引入了多级BEV蒸馏策略（具有空间掩码）和不对称查询适应模块，有效将未来感知表示转移到学生的静态查询中。

### 主要发现

在nuScenes和Argoverse 2基准测试上的大量实验表明，AMap显著增强了当前帧的感知能力。值得注意的是，它在关键的前方区域超越了最先进的时序模型，同时保持了单当前帧推理的效率。

### 结论

AMap通过前瞻性感知能力的引入，有效弥补了现有在线高清地图方法的安全差距，在不增加计算成本的情况下提高了前方区域的感知准确性，为自动驾驶提供了更安全的地图构建解决方案。

### 翻译

在线高清地图构建对自动驾驶至关重要。虽然最近的方法利用历史时间融合来提高性能，但我们确定这一范式中的一个关键安全缺陷：它本质上'空间向后看'。这些方法主要增强已通过区域的地图重建，对前方未见的道路改进有限。重要的是，我们对下游规划任务的分析揭示了严重的非对称性：虽然向后感知错误通常可以容忍，但前方区域的不准确性会直接导致危险的驾驶操作。为了弥补这一安全差距，我们提出了AMap，一种面向前方的在线高清地图构建的新型框架。我们开创了'从未来蒸馏'范式，其中拥有未来上下文访问权限的教师模型指导受限于当前帧的轻量级学生模型。这个过程将前瞻性知识隐式压缩到学生模型中，使其以零推理时间成本获得'前瞻'能力。技术上，我们引入了具有空间掩码的多级BEV蒸馏策略和不对称查询适应模块，以有效将未来感知表示转移到学生的静态查询中。在nuScenes和Argoverse 2基准测试上的大量实验表明，AMap显著增强了当前帧的感知能力。最值得注意的是，它在关键的前方区域超越了最先进的时序模型，同时保持了单当前帧推理的效率。


### 论文摘要

Online High-Definition (HD) map construction is pivotal for autonomous driving. While recent approaches leverage historical temporal fusion to improve performance, we identify a critical safety flaw in this paradigm: it is inherently ``spatially backward-looking." These methods predominantly enhance map reconstruction in traversed areas, offering minimal improvement for the unseen road ahead. Crucially, our analysis of downstream planning tasks reveals a severe asymmetry: while rearward perception errors are often tolerable, inaccuracies in the forward region directly precipitate hazardous driving maneuvers. To bridge this safety gap, we propose AMap, a novel framework for Ahead-aware online HD Mapping. We pioneer a ``distill-from-future" paradigm, where a teacher model with privileged access to future temporal contexts guides a lightweight student model restricted to the current frame. This process implicitly compresses prospective knowledge into the student model, endowing it with ``look-ahead" capabilities at zero inference-time cost. Technically, we introduce a Multi-Level BEV Distillation strategy with spatial masking and an Asymmetric Query Adaptation module to effectively transfer future-aware representations to the student's static queries. Extensive experiments on the nuScenes and Argoverse 2 benchmark demonstrate that AMap significantly enhances current-frame perception. Most notably, it outperforms state-of-the-art temporal models in critical forward regions while maintaining the efficiency of single current frame inference.

---

## 13. D$^{2}$Stream: Decoupled Dual-Stream Temporal-Speaker Interaction for Audio-Visual Speaker Detection

**论文链接:** [http://arxiv.org/abs/2512.19130v1](http://arxiv.org/abs/2512.19130v1)

**作者:** Junhao Xiao, Shun Feng, Zhiyu Wu, Jianjun Li, Zhiyuan Ma, Yi Chen

**发布时间:** 2025-12-22

### GPT解析

### 总结

研究提出了一种名为D²Stream的去耦双流框架，用于视频中的活跃说话人检测，分离了跨帧时间建模和帧内说话人判别，在保持高性能的同时显著提高了计算效率。

### 背景

音频-视觉说话人检测旨在通过利用互补的音频和视觉线索来识别视频中的活跃说话人。现有方法通常由于时间建模和说话人交互的联合建模而导致计算效率低下或性能不佳。

### 目的

解决现有方法在计算效率和性能上的不足，提出一种能够有效分离时间建模和说话人判别的框架，以提高说话人检测的准确性和效率。

### 方法

提出D²Stream去耦双流框架，通过跨模态注意力对齐音频和视觉特征，输入两个轻量级流：时间交互流捕获长程时间依赖，说话人交互流建模每帧间的人际关系。两个流提取的特征通过交叉注意力相互作用，并引入轻量级语音门模块减轻非言语面部运动的假阳性。

### 主要发现

在AVA-ActiveSpeaker上达到95.6% mAP的最新技术水平，相比基于GNN的模型计算量减少80%，参数比基于注意力的替代方案减少30%，同时在Columbia ASD数据集上表现出良好的泛化能力。

### 结论

D²Stream框架通过分离时间建模和说话人判别，有效解决了现有方法的计算效率和性能问题，实现了更好的性能和效率平衡，具有良好的泛化能力和实用价值。

### 翻译

音频-视觉说话人检测旨在通过利用互补的音频和视觉线索来识别视频中的活跃说话人。现有方法通常由于时间建模和说话人交互的联合建模而导致计算效率低下或性能不佳。我们提出了D²Stream，一种去耦双流框架，将跨帧时间建模与帧内说话人判别分离。音频和视觉特征首先通过跨模态注意力进行对齐，然后输入两个轻量级流：时间交互流捕获长程时间依赖，而说话人交互流建模每帧间的人际关系。两个流提取的时间和关系特征通过交叉注意力相互作用以丰富表示。轻量级语音门模块进一步减轻了来自非言语面部运动的假阳性。在AVA-ActiveSpeaker上，D²Stream以95.6% mAP取得了最新的技术水平，计算量比基于GNN的模型减少80%，参数比基于注意力的替代方案减少30%，同时在Columbia ASD上也具有良好的泛化能力。源代码可在https://anonymous.4open.science/r/D2STREAM获取。


### 论文摘要

Audio-visual speaker detection aims to identify the active speaker in videos by leveraging complementary audio and visual cues. Existing methods often suffer from computational inefficiency or suboptimal performance due to joint modeling of temporal and speaker interactions. We propose D$^{2}$Stream, a decoupled dual-stream framework that separates cross-frame temporal modeling from within-frame speaker discrimination. Audio and visual features are first aligned via cross-modal attention, then fed into two lightweight streams: a Temporal Interaction Stream captures long-range temporal dependencies, while a Speaker Interaction Stream models per-frame inter-person relationships. The temporal and relational features extracted by the two streams interact via cross-attention to enrich representations. A lightweight Voice Gate module further mitigates false positives from non-speech facial movements. On AVA-ActiveSpeaker, D$^{2}$Stream achieves a new state-of-the-art at 95.6% mAP, with 80% reduction in computation compared to GNN-based models and 30% fewer parameters than attention-based alternatives, while also generalizing well on Columbia ASD. Source code is available at https://anonymous.4open.science/r/D2STREAM.

---

## 14. 论文ID: 2512.19107v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19107v1.json'

---

## 15. Distinguishing Visually Similar Actions: Prompt-Guided Semantic Prototype Modulation for Few-Shot Action Recognition

**论文链接:** [http://arxiv.org/abs/2512.19036v1](http://arxiv.org/abs/2512.19036v1)

**作者:** Xiaoyang Li, Mingming Lu, Ruiqi Wang, Hao Li, Zewei Le

**发布时间:** 2025-12-22

**备注:** 19 pages, 7 figures. Preprint under review for journal submission

### GPT解析

### 总结

本文提出CLIP-SPM框架，解决少样本动作识别中的三个核心挑战：时间建模、视觉相似性和模态差距，包含HSMR模块、SPM策略和PADM方法三个组件。

### 背景

少样本动作识别旨在使模型从有限标记样本中快速学习新动作类别，解决现实应用中数据稀缺问题。当前研究面临三大挑战：时间建模易受静态背景干扰、视觉相似动作难以区分、视觉-文本模态差距导致对齐困难。

### 目的

解决少样本动作识别中的时间建模、视觉相似性和模态差距三大核心挑战。

### 方法

提出CLIP-SPM框架，包含三个组件：1)层次协同运动精炼(HSMR)模块，对齐深度和浅层运动特征减少背景干扰；2)语义原型调制(SPM)策略，生成查询相关文本提示弥合模态差距；3)原型锚点双调制(PADM)方法，精炼支持原型并提升查询特征一致性。

### 主要发现

在Kinetics、SSv2-Full、SSv2-Small、UCF101和HMDB51等基准测试上，CLIP-SPM在1-shot、3-shot和5-shot设置下取得竞争性性能，消融研究验证了各组件有效性。

### 结论

CLIP-SPM框架通过三个关键组件有效解决少样本动作识别核心挑战，源代码和模型已在GitHub公开。

### 翻译

少样本动作识别旨在使模型能够从有限的标记样本中快速学习新的动作类别，解决现实应用中数据稀缺的挑战。当前研究主要解决三个核心挑战：(1)时间建模，模型容易受到无关静态背景信息的干扰，难以捕捉动态动作特征的精髓；(2)视觉相似性，具有细微视觉差异的类别难以区分；(3)视觉-文本支持原型与仅视觉查询之间的模态差距，这使得在共享嵌入空间中对齐变得复杂。为解决这些挑战，本文提出CLIP-SPM框架，包含三个组件：(1)层次协同运动精炼(HSMR)模块，对齐深度和浅层运动特征，通过减少静态背景干扰来改善时间建模；(2)语义原型调制(SPM)策略，生成查询相关的文本提示，弥合模态差距，并将其与视觉特征集成，增强相似动作之间的区分性；(3)原型锚点双调制(PADM)方法，精炼支持原型并将查询特征与全局语义锚点对齐，提高支持样本和查询样本之间的一致性。在Kinetics、SSv2-Full、SSv2-Small、UCF101和HMDB51等标准基准上进行的综合实验表明，我们的CLIP-SPM在1-shot、3-shot和5-shot设置下取得了具有竞争力的性能。大量的消融研究和可视化分析进一步验证了每个组件的有效性及其对解决核心挑战的贡献。源代码和模型已在GitHub上公开。


### 论文摘要

Few-shot action recognition aims to enable models to quickly learn new action categories from limited labeled samples, addressing the challenge of data scarcity in real-world applications. Current research primarily addresses three core challenges: (1) temporal modeling, where models are prone to interference from irrelevant static background information and struggle to capture the essence of dynamic action features; (2) visual similarity, where categories with subtle visual differences are difficult to distinguish; and (3) the modality gap between visual-textual support prototypes and visual-only queries, which complicates alignment within a shared embedding space. To address these challenges, this paper proposes a CLIP-SPM framework, which includes three components: (1) the Hierarchical Synergistic Motion Refinement (HSMR) module, which aligns deep and shallow motion features to improve temporal modeling by reducing static background interference; (2) the Semantic Prototype Modulation (SPM) strategy, which generates query-relevant text prompts to bridge the modality gap and integrates them with visual features, enhancing the discriminability between similar actions; and (3) the Prototype-Anchor Dual Modulation (PADM) method, which refines support prototypes and aligns query features with a global semantic anchor, improving consistency across support and query samples. Comprehensive experiments across standard benchmarks, including Kinetics, SSv2-Full, SSv2-Small, UCF101, and HMDB51, demonstrate that our CLIP-SPM achieves competitive performance under 1-shot, 3-shot, and 5-shot settings. Extensive ablation studies and visual analyses further validate the effectiveness of each component and its contributions to addressing the core challenges. The source code and models are publicly available at GitHub.

---

## 16. CrashChat: A Multimodal Large Language Model for Multitask Traffic Crash Video Analysis

**论文链接:** [http://arxiv.org/abs/2512.18878v1](http://arxiv.org/abs/2512.18878v1)

**作者:** Kaidi Liang, Ke Li, Xianbiao Hu, Ruwen Qin

**发布时间:** 2025-12-21

### GPT解析

### 总结

该研究提出了CrashChat，一个基于VideoLLaMA3构建的多模态大语言模型，用于自动化交通事故视频分析。CrashChat通过指令微调获取领域知识，并采用基于任务解耦和分组的新型多任务学习策略，在事故识别、定位和描述推理任务上取得了最先进的性能。

### 背景

自动化交通事故视频分析对于交通安全研究和自动驾驶责任归因至关重要。由于视频数据中事故事件的复杂时空动态和多样化分析需求，交通事故视频分析是一个具有挑战性的多任务问题，需要涵盖事故识别、时间定位和高级视频理解能力。

### 目的

填补现有模型无法在统一框架内执行所有这些任务的空白，并探索此类模型的有效训练策略。

### 方法

提出了CrashChat，一个基于VideoLLaMA3构建的多模态大语言模型(MLLM)，通过指令微调获取领域特定知识，并采用基于任务解耦和分组的新型多任务学习策略，最大化组内和跨组联合学习的收益，同时减轻负迁移。

### 主要发现

在合并的公共数据集上，CrashChat在各种模型规模和传统基于视觉的方法上都胜过现有的MLLM，达到最先进性能。事故识别达到接近完美准确性，事故定位能力提高176%，事故前定位提高40%。与通用MLLM相比，在事故描述和推理任务中，BLEU分数提高0.18-0.41，ROUGE分数提高0.18-0.42。

### 结论

CrashChat是一个方便的端到端分析工具，已准备好实际实施。相关数据集和实现代码已在GitHub开源。

### 翻译

自动化交通事故视频分析对于利用不断增长的驾驶视频数据进行交通安全研究和自动驾驶责任归因至关重要。由于视频数据中事故事件的复杂时空动态和多样化的分析需求，交通事故视频分析是一个具有挑战性的多任务问题。它需要涵盖事故识别、时间定位和高级视频理解的能力。然而，现有模型无法在统一框架内执行所有这些任务，此类模型的有效训练策略仍有待探索。为填补这些空白，本文提出了CrashChat，一个基于VideoLLaMA3构建的多模态大语言模型(MLLM)，用于多任务交通事故分析。CrashChat通过指令微调获取领域特定知识，并采用一种基于任务解耦和分组的新型多任务学习策略，最大化组内和跨组联合学习的收益，同时减轻负迁移。在合并的公共数据集上的数值实验表明，CrashChat在各种模型规模和传统基于视觉的方法上都胜过现有的MLLM，达到最先进性能。在事故识别方面达到接近完美的准确性，事故定位能力提高176%，在更具挑战性的事故前定位方面提高40%。与通用MLLM相比，它在事故描述和推理任务中显著提高了文本准确性和内容覆盖率，BLEU分数提高0.18-0.41，ROUGE分数提高0.18-0.42。除了强大的性能外，CrashChat是一个方便的端到端分析工具，已准备好实际实施。CrashChat的数据集和实现代码可在https://github.com/Liangkd/CrashChat获取。


### 论文摘要

Automating crash video analysis is essential to leverage the growing availability of driving video data for traffic safety research and accountability attribution in autonomous driving. Crash video analysis is a challenging multitask problem due to the complex spatiotemporal dynamics of crash events in video data and the diverse analytical requirements involved. It requires capabilities spanning crash recognition, temporal grounding, and high-level video understanding. Existing models, however, cannot perform all these tasks within a unified framework, and effective training strategies for such models remain underexplored. To fill these gaps, this paper proposes CrashChat, a multimodal large language model (MLLM) for multitask traffic crash analysis, built upon VideoLLaMA3. CrashChat acquires domain-specific knowledge through instruction fine-tuning and employs a novel multitask learning strategy based on task decoupling and grouping, which maximizes the benefit of joint learning within and across task groups while mitigating negative transfer. Numerical experiments on consolidated public datasets demonstrate that CrashChat consistently outperforms existing MLLMs across model scales and traditional vision-based methods, achieving state-of-the-art performance. It reaches near-perfect accuracy in crash recognition, a 176\% improvement in crash localization, and a 40\% improvement in the more challenging pre-crash localization. Compared to general MLLMs, it substantially enhances textual accuracy and content coverage in crash description and reasoning tasks, with 0.18-0.41 increases in BLEU scores and 0.18-0.42 increases in ROUGE scores. Beyond its strong performance, CrashChat is a convenient, end-to-end analytical tool ready for practical implementation. The dataset and implementation code for CrashChat are available at https://github.com/Liangkd/CrashChat.

---

## 17. Context-Aware Network Based on Multi-scale Spatio-temporal Attention for Action Recognition in Videos

**论文链接:** [http://arxiv.org/abs/2512.18750v1](http://arxiv.org/abs/2512.18750v1)

**作者:** Xiaoyang Li, Wenzhu Yang, Kanglin Wang, Tiebiao Wang, Qingsong Fei

**发布时间:** 2025-12-21

**备注:** 21 pages, 4 figures. Preprint under review for journal submission

### GPT解析

### 总结

本文提出了一种上下文感知网络(CAN)用于动作识别，通过多尺度时间线索模块和分组空间线索模块分别捕捉多尺度的时空特征，在五个基准数据集上取得了有竞争力的性能。

### 背景

动作识别是视频理解中的关键任务，需要全面捕捉各种尺度的时空线索。

### 目的

解决现有方法忽略动作多粒度性质的问题，提高动作识别的准确性。

### 方法

提出上下文感知网络(CAN)，包含两个核心模块：多尺度时间线索模块(MTCM)用于提取多尺度时间线索，捕捉快速变化的运动细节和整体动作流程；分组空间线索模块(GSCM)通过分组特征图并对每组应用专门的提取方法来提取不同尺度的空间线索。

### 主要发现

在五个基准数据集(Something-Something V1和V2、Diving48、Kinetics-400和UCF101)上的实验表明，CAN取得了有竞争力的性能，准确率分别为50.4%、63.9%、88.4%、74.9%和86.9%，优于大多数主流方法。

### 结论

捕捉多尺度时空线索对鲁棒的动作识别非常重要，所提出的CAN方法有效解决了现有方法的局限性。

### 翻译

动作识别是视频理解中的关键任务，需要全面捕捉各种尺度的时空线索。然而，现有方法常常忽略动作的多粒度性质。为解决这一局限，我们引入了上下文感知网络(CAN)。CAN包含两个核心模块：多尺度时间线索模块(MTCM)和分组空间线索模块(GSCM)。MTCM有效提取多尺度时间线索，同时捕捉快速变化的运动细节和整体动作流程。另一方面，GSCM通过分组特征图并对每组应用专门的提取方法来提取不同尺度的空间线索。在五个基准数据集(Something-Something V1和V2、Diving48、Kinetics-400和UCF101)上进行的实验证明了CAN的有效性。我们的方法取得了有竞争力的性能，优于大多数主流方法，在Something-Something V1上的准确率为50.4%，Something-Something V2为63.9%，Diving48为88.4%，Kinetics-400为74.9%，UCF101为86.9%。这些结果强调了捕捉多尺度时空线索对鲁棒动作识别的重要性。


### 论文摘要

Action recognition is a critical task in video understanding, requiring the comprehensive capture of spatio-temporal cues across various scales. However, existing methods often overlook the multi-granularity nature of actions. To address this limitation, we introduce the Context-Aware Network (CAN). CAN consists of two core modules: the Multi-scale Temporal Cue Module (MTCM) and the Group Spatial Cue Module (GSCM). MTCM effectively extracts temporal cues at multiple scales, capturing both fast-changing motion details and overall action flow. GSCM, on the other hand, extracts spatial cues at different scales by grouping feature maps and applying specialized extraction methods to each group. Experiments conducted on five benchmark datasets (Something-Something V1 and V2, Diving48, Kinetics-400, and UCF101) demonstrate the effectiveness of CAN. Our approach achieves competitive performance, outperforming most mainstream methods, with accuracies of 50.4% on Something-Something V1, 63.9% on Something-Something V2, 88.4% on Diving48, 74.9% on Kinetics-400, and 86.9% on UCF101. These results highlight the importance of capturing multi-scale spatio-temporal cues for robust action recognition.

---

## 18. SmartSight: Mitigating Hallucination in Video-LLMs Without Compromising Video Understanding via Temporal Attention Collapse

**论文链接:** [http://arxiv.org/abs/2512.18671v1](http://arxiv.org/abs/2512.18671v1)

**作者:** Yiming Sun, Mi Zhang, Feifei Li, Geng Hong, Min Yang

**发布时间:** 2025-12-21

**备注:** AAAI26 accepted

### GPT解析

### 总结

SmartSight是一种无需训练的视频大语言模型幻觉缓解方法，通过生成多个候选响应并利用时间注意力崩溃分数评估幻觉程度，同时识别视觉注意力消失点提高效率，显著降低了幻觉风险并增强了视频理解能力。

### 背景

视频大语言模型近年来快速发展，但知觉幻觉构成重大安全风险，严重限制了其现实世界应用。现有幻觉缓解方法往往损害模型的理解和推理能力。

### 目的

提出一种不损害视频理解和推理能力的幻觉缓解方法，采用无需训练的方式解决视频大语言模型中的幻觉问题。

### 方法

SmartSight通过生成多个候选响应来发现低幻觉输出，使用时间注意力崩溃分数评估幻觉程度，识别视觉注意力消失点以提高效率和估计准确性，实现提前终止幻觉响应并降低解码成本。

### 主要发现

SmartSight在VRIPT-HAL数据集上将Qwen2.5-VL-7B的幻觉降低了10.59%，同时在VideoMMMU上将性能提高了高达8.86%，增强了视频理解和推理能力。

### 结论

SmartSight在提高开源视频大语言模型的可靠性方面是有效的，解决了幻觉问题而不损害模型性能。

### 翻译

尽管视频大语言模型近年来迅速发展，但知觉幻觉构成了重大安全风险，严重限制了它们的现实适用性。虽然已经提出了几种幻觉缓解方法，但它们往往损害了模型对视频的理解和推理能力。在这项工作中，我们提出了SmartSight，这是通过利用模型自身的内省能力以无需训练的方式解决此问题的开创性步骤。具体来说，SmartSight生成多个候选响应以揭示通常被标准贪婪解码掩盖的低幻觉输出。它使用时间注意力崩溃分数评估每个响应的幻觉程度，该分数衡量模型在生成响应时是否过度关注输入视频中琐碎的时间区域。为了提高效率，SmartSight识别视觉注意力消失点，使更准确的幻觉估计和提前终止幻觉响应成为可能，从而显著降低了解码成本。实验表明，SmartSight在VRIPT-HAL上将Qwen2.5-VL-7B的幻觉大幅降低了10.59%，同时增强了视频理解和推理能力，在VideoMMMU上将性能提高了高达8.86%。这些结果突显了SmartSight在提高开源视频大语言模型可靠性方面的有效性。


### 论文摘要

Despite Video Large Language Models having rapidly advanced in recent years, perceptual hallucinations pose a substantial safety risk, which severely restricts their real-world applicability. While several methods for hallucination mitigation have been proposed, they often compromise the model's capacity for video understanding and reasoning. In this work, we propose SmartSight, a pioneering step to address this issue in a training-free manner by leveraging the model's own introspective capabilities. Specifically, SmartSight generates multiple candidate responses to uncover low-hallucinated outputs that are often obscured by standard greedy decoding. It assesses the hallucination of each response using the Temporal Attention Collapse score, which measures whether the model over-focuses on trivial temporal regions of the input video when generating the response. To improve efficiency, SmartSight identifies the Visual Attention Vanishing point, enabling more accurate hallucination estimation and early termination of hallucinated responses, leading to a substantial reduction in decoding cost. Experiments show that SmartSight substantially lowers hallucinations for Qwen2.5-VL-7B by 10.59% on VRIPT-HAL, while simultaneously enhancing video understanding and reasoning, boosting performance on VideoMMMU by up to 8.86%. These results highlight SmartSight's effectiveness in improving the reliability of open-source Video-LLMs.

---

## 19. Insider Threat Detection Using GCN and Bi-LSTM with Explicit and Implicit Graph Representations

**论文链接:** [http://arxiv.org/abs/2512.18483v1](http://arxiv.org/abs/2512.18483v1)

**作者:** Rahul Yumlembam, Biju Issac, Seibu Mary Jacob, Longzhi Yang, Deepa Krishnan

**发布时间:** 2025-12-20

**DOI:** 10.1109/TAI.2025.3647418

**备注:** 12 pages, IEEE Transactions on Artificial Intelligence (2025)

### GPT解析

### 总结

本文提出了一种结合显式和隐式图表示与时间建模的事后内部威胁检测框架，通过图卷积网络和双向长短期记忆网络捕获复杂用户行为模式，实验证明该方法在CERT数据集上优于现有方法。

### 背景

内部威胁检测具有挑战性，因为恶意活动通常由可信用户执行，且行为微妙且隐蔽，难以检测。

### 目的

开发一种能够捕获复杂用户行为模式的有效内部威胁检测框架，通过结合图表示和时间建模提高检测准确性。

### 方法

构建显式图建模直接关系，使用Gumbel-Softmax技巧学习隐式图发现潜在关系，分别通过图卷积网络处理两种图生成节点嵌入，通过注意力机制强调威胁特征，最后使用双向长短期记忆网络捕获时间依赖性，低于阈值的活动被标记为异常。

### 主要发现

在CERT r5.2数据集上达到AUC为98.62，检测率100%，误报率0.05；在更具挑战性的r6.2数据集上获得AUC为88.48，检测率80.15%，误报率0.15，显著优于现有方法。

### 结论

结合基于图和时间表示的方法对于强大的内部威胁检测是有效的，能够准确识别可疑活动同时保持低误报率。

### 翻译

内部威胁检测具有挑战性，因为由可信用户执行的恶意活动具有微妙和隐蔽的特性。本文提出了一种事后内部威胁检测框架，结合显式和隐式图表示与时间建模，以捕获复杂的用户行为模式。使用预定义的组织规则构建显式图，建模用户活动之间的直接关系。为了减轻这种手工制作结构中的噪声和限制，使用Gumbel-Softmax技巧从特征相似性中学习隐式图， enabling发现潜在的行为关系。单独的图卷积网络处理显式和隐式图以生成节点嵌入，这些嵌入通过注意力机制连接和细化，以强调威胁相关特征。然后将细化的表示传递到双向长短期记忆网络，以捕获用户行为中的时间依赖性。当活动概率分数低于预定义阈值时，它们被标记为异常。在CERT r5.2和r6.2数据集上的广泛实验表明，所提出的框架优于最先进的方法。在r5.2上，模型达到AUC为98.62，检测率为100%，误报率为0.05。在更具挑战性的r6.2数据集上，它获得AUC为88.48，检测率为80.15%，误报率为0.15，突显了结合基于图和时间表示对于强大的内部威胁检测的有效性。


### 论文摘要

Insider threat detection (ITD) is challenging due to the subtle and concealed nature of malicious activities performed by trusted users. This paper proposes a post-hoc ITD framework that integrates explicit and implicit graph representations with temporal modelling to capture complex user behaviour patterns. An explicit graph is constructed using predefined organisational rules to model direct relationships among user activities. To mitigate noise and limitations in this hand-crafted structure, an implicit graph is learned from feature similarities using the Gumbel-Softmax trick, enabling the discovery of latent behavioural relationships. Separate Graph Convolutional Networks (GCNs) process the explicit and implicit graphs to generate node embeddings, which are concatenated and refined through an attention mechanism to emphasise threat-relevant features. The refined representations are then passed to a bidirectional Long Short-Term Memory (Bi-LSTM) network to capture temporal dependencies in user behaviour. Activities are flagged as anomalous when their probability scores fall below a predefined threshold. Extensive experiments on CERT r5.2 and r6.2 datasets demonstrate that the proposed framework outperforms state-of-the-art methods. On r5.2, the model achieves an AUC of 98.62, a detection rate of 100%, and a false positive rate of 0.05. On the more challenging r6.2 dataset, it attains an AUC of 88.48, a detection rate of 80.15%, and a false positive rate of 0.15, highlighting the effectiveness of combining graph-based and temporal representations for robust ITD.

---

## 20. 论文ID: 2512.18477v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18477v1.json'

---

## 21. Embodied4C: Measuring What Matters for Embodied Vision-Language Navigation

**论文链接:** [http://arxiv.org/abs/2512.18028v1](http://arxiv.org/abs/2512.18028v1)

**作者:** Tin Stribor Sohn, Maximilian Dillitzer, Jason J. Corso, Eric Sax

**发布时间:** 2025-12-19

### GPT解析

### 总结

该研究引入了Embodied4C基准测试，用于评估视觉-语言模型在不同具身平台上的核心具身能力，通过三个异构具身平台(自动驾驶汽车、空中无人机和机械臂)评估模型的语义、空间、时间和物理推理能力。

### 背景

视觉-语言导航需要代理在具身约束下进行推理和行动。虽然视觉-语言模型展现出强大的泛化能力，但当前基准测试对具身因素(物理平台选择、传感器配置和模态对齐)如何影响感知、推理和控制的理解有限。

### 目的

创建一个闭环基准测试(Embodied4C)作为具身推理的图灵测试，评估视觉-语言模型在不同具身环境中的核心能力，并了解具身因素对模型性能的影响。

### 方法

Embodied4C通过三个异构具身平台评估VLMs，包含约1.1K个一次性推理问题和58个目标导向的导航任务。这些任务评估语义、空间、时间和物理推理四个维度。每个具身平台提供动态传感器配置和环境变化，并整合远域查询以防止具身过拟合。

### 主要发现

对十个先进VLMs和四个具身控制基线的评估表明，跨模态对齐和指令调优比模型规模更重要，而空间和时间推理是可靠具身能力的主要瓶颈。

### 结论

具身因素对视觉-语言模型性能有显著影响，Embodied4C为评估和改进VLMs的具身推理能力提供了重要工具。

### 翻译

视觉-语言导航需要代理在具身约束下进行推理和行动。虽然视觉-语言模型展现出强大的泛化能力，但当前基准测试对具身因素如何影响感知、推理和控制的理解有限。我们引入了Embodied4C，一个闭环基准测试，设计为具身推理的图灵测试。该基准测试通过三个异构具身平台评估VLMs的核心具身能力，包含约1.1K个一次性推理问题和58个目标导向的导航任务。这些任务共同评估四个基础维度：语义、空间、时间和物理推理。每个具身平台提供动态传感器配置和环境变化，以测试超越平台特定适应的泛化能力。为防止具身过拟合，Embodied4C整合了针对抽象和跨上下文推理的远域查询。对十个先进VLMs和四个具身控制基线的全面评估表明，跨模态对齐和指令调优比模型规模更重要，而空间和时间推理仍然是可靠具身能力的主要瓶颈。


### 论文摘要

Vision-language navigation requires agents to reason and act under constraints of embodiment. While vision-language models (VLMs) demonstrate strong generalization, current benchmarks provide limited understanding of how embodiment -- i.e., the choice of physical platform, sensor configuration, and modality alignment -- influences perception, reasoning, and control. We introduce Embodied4C, a closed-loop benchmark designed as a Turing test for embodied reasoning. The benchmark evaluates the core embodied capabilities of VLMs across three heterogeneous embodiments -- autonomous vehicles, aerial drones, and robotic manipulators -- through approximately 1.1K one-shot reasoning questions and 58 goal-directed navigation tasks. These tasks jointly assess four foundational dimensions: semantic, spatial, temporal, and physical reasoning. Each embodiment presents dynamic sensor configurations and environment variations to probe generalization beyond platform-specific adaptation. To prevent embodiment overfitting, Embodied4C integrates domain-far queries targeting abstract and cross-context reasoning. Comprehensive evaluation across ten state-of-the-art VLMs and four embodied control baselines shows that cross-modal alignment and instruction tuning matter more than scale, while spatial and temporal reasoning remains the primary bottleneck for reliable embodied competence.

---

## 22. MauBERT: Universal Phonetic Inductive Biases for Few-Shot Acoustic Units Discovery

**论文链接:** [http://arxiv.org/abs/2512.19612v1](http://arxiv.org/abs/2512.19612v1)

**作者:** Angelo Ortiz Tandazo, Manel Khentout, Youssef Benchekroun, Thomas Hueber, Emmanuel Dupoux

**发布时间:** 2025-12-22

### GPT解析

### 总结

本文介绍了MauBERT，这是HuBERT的一个多语言扩展版本，利用发音特征进行稳健的跨语言语音表征学习。

### 背景

现有自监督语音模型在多语言场景下的表征能力有限，需要更有效的跨语言学习方法。

### 目的

开发一种能够学习语言无关表征的多语言语音模型，捕捉多语言语音特性，并适应未见语言。

### 方法

基于55种语言的语音到发音特征映射监督继续进行HuBERT预训练，让模型从多语言数据中预测发音特征或音素。

### 主要发现

MauBERT模型比最先进的多语言自监督学习模型产生更多上下文不变的表征，能够有效适应未见语言和日常语音，仅需10小时语音的自监督微调。

### 结论

为在自监督语音模型中植入语言归纳偏置提供了一种有效方法，提高了模型的跨语言适应性和语音表征能力。

### 翻译

本文介绍了MauBERT，这是HuBERT的一个多语言扩展版本，利用发音特征进行稳健的跨语言语音表征学习。我们基于55种语言的语音到发音特征映射监督继续进行HuBERT预训练。我们的模型从多语言数据中学习预测发音特征或音素，从而捕捉多语言语音特性的语言无关表征。通过全面的ABX可辨别性测试，我们表明MauBERT模型比最先进的多语言自监督学习模型产生更多上下文不变的表征。此外，模型能够有效适应未见语言和日常语音，只需最少的自监督微调（10小时语音）。这为在自监督语音模型中植入语言归纳偏置建立了一种有效方法。


### 论文摘要

This paper introduces MauBERT, a multilingual extension of HuBERT that leverages articulatory features for robust cross-lingual phonetic representation learning. We continue HuBERT pre-training with supervision based on a phonetic-to-articulatory feature mapping in 55 languages. Our models learn from multilingual data to predict articulatory features or phones, resulting in language-independent representations that capture multilingual phonetic properties. Through comprehensive ABX discriminability testing, we show MauBERT models produce more context-invariant representations than state-of-the-art multilingual self-supervised learning models. Additionally, the models effectively adapt to unseen languages and casual speech with minimal self-supervised fine-tuning (10 hours of speech). This establishes an effective approach for instilling linguistic inductive biases in self-supervised speech models.

---

## 23. KerJEPA: Kernel Discrepancies for Euclidean Self-Supervised Learning

**论文链接:** [http://arxiv.org/abs/2512.19605v1](http://arxiv.org/abs/2512.19605v1)

**作者:** Eric Zimmermann, Harley Wiltzer, Justin Szeto, David Alvarez-Melis, Lester Mackey

**发布时间:** 2025-12-22

### GPT解析

### 总结

本文介绍了一种新的基于核正则器的自监督学习算法家族KerJEPAs，通过扩展可用核和先验的类别，提高了训练稳定性和设计灵活性。

### 背景

自监督的联合嵌入预测架构(JEPAs)的最新研究表明，将欧几里得表示正则化为各向同性高斯先验可以在训练稳定性和下游泛化方面带来可证明的提升。

### 目的

引入一种新的、灵活的KerJEPAs家族，作为自监督学习算法，提供基于核正则器的解决方案。

### 方法

扩展可用的核和先验类别，计算切片最大均值差异的高维闭式极限，开发具有改进特性的替代KerJEPAs算法。

### 主要发现

新开发的KerJEPAs算法具有改进的训练稳定性和设计灵活性，其中一个实例对应于最近引入的LeJEPA Epps-Pulley正则器，它使用高斯先验和高斯核来近似切片最大均值差异。

### 结论

KerJEPAs为自监督学习提供了一种灵活的新方法，通过多样化的核和先验选择，能够提高模型训练稳定性和设计灵活性。

### 翻译

自监督联合嵌入预测架构的最新突破已经证明，将欧几里得表示正则化为各向同性高斯先验可以为训练稳定性和下游泛化带来可证明的提升。我们引入了一个新的、灵活的KerJEPAs家族，这是具有基于核的正则器的自监督学习算法。该家族的一个实例对应于最近引入的LeJEPA Epps-Pulley正则器，它使用高斯先验和高斯核来近似切片最大均值差异。通过扩展可用核和先验的类别，并计算切片MMD的高维闭式极限，我们开发了具有多种有利特性的替代KerJEPAs，包括改进的训练稳定性和设计灵活性。


### 论文摘要

Recent breakthroughs in self-supervised Joint-Embedding Predictive Architectures (JEPAs) have established that regularizing Euclidean representations toward isotropic Gaussian priors yields provable gains in training stability and downstream generalization. We introduce a new, flexible family of KerJEPAs, self-supervised learning algorithms with kernel-based regularizers. One instance of this family corresponds to the recently-introduced LeJEPA Epps-Pulley regularizer which approximates a sliced maximum mean discrepancy (MMD) with a Gaussian prior and Gaussian kernel. By expanding the class of viable kernels and priors and computing the closed-form high-dimensional limit of sliced MMDs, we develop alternative KerJEPAs with a number of favorable properties including improved training stability and design flexibility.

---

## 24. 论文ID: 2512.19213v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19213v1.json'

---

## 25. WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.19133v1](http://arxiv.org/abs/2512.19133v1)

**作者:** Pengxuan Yang, Ben Lu, Zhongpu Xia, Chao Han, Yinfeng Gao, Teng Zhang, Kun Zhan, XianPeng Lang, Yupeng Zheng, Qichao Zhang

**发布时间:** 2025-12-22

**备注:** AAAI 2026, first version

### GPT解析

### 总结

论文提出了WorldRFT，一种面向规划的潜在世界模型框架，通过分层规划分解和局部感知交互细化机制将场景表示学习与规划对齐，并通过强化学习微调提高安全关键策略性能，在nuScenes和NavSim基准测试上取得最先进性能。

### 背景

潜在世界模型通过时间自监督学习增强场景表示，为端到端自动驾驶提供无需感知标注的范式。然而，以重建为导向的表示学习将感知与规划任务纠缠，导致规划优化次优。

### 目的

解决以重建为导向的表示学习与规划任务之间的冲突，提出面向规划的潜在世界模型框架，使场景表示学习与规划目标对齐，提高安全关键策略性能。

### 方法

提出WorldRFT框架，集成视觉几何基础模型提高3D空间感知能力，采用分层规划任务分解指导表示优化，利用局部感知迭代细化推导面向规划的驾驶策略，并引入组相对策略优化(GRPO)应用轨迹高斯化和碰撞感知奖励微调驾驶策略。

### 主要发现

在nuScenes上将碰撞率降低83%（0.30%降至0.05%）；在NavSim上仅使用摄像头传感器输入，达到与基于LiDAR的最先进方法DiffusionDrive相当的性能（87.8对88.1 PDMS）。

### 结论

WorldRFT通过将场景表示学习与规划目标对齐，显著提高自动驾驶安全性和性能，在开放循环nuScenes和封闭循环NavSim基准测试上都取得最先进性能。

### 翻译

潜在世界模型通过时间自监督学习增强场景表示，为端到端自动驾驶呈现了一种无需感知标注的范式。然而，以重建为导向的表示学习将感知与规划任务纠缠在一起，导致规划优化次优。为了解决这一挑战，我们提出了WorldRFT，这是一种面向规划的潜在世界模型框架，通过分层规划分解和局部感知交互细化机制将场景表示学习与规划对齐，并通过强化学习微调增强安全关键策略性能。具体而言，WorldRFT集成了视觉几何基础模型以提高3D空间感知能力，采用分层规划任务分解来指导表示优化，并利用局部感知迭代细化来推导面向规划的驾驶策略。此外，我们引入了组相对策略优化，该优化应用轨迹高斯化和碰撞感知奖励来微调驾驶策略，从而在安全性方面产生系统性改进。WorldRFT在开放循环nuScenes和封闭循环NavSim基准测试上都取得了最先进的性能。


### 论文摘要

Latent World Models enhance scene representation through temporal self-supervised learning, presenting a perception annotation-free paradigm for end-to-end autonomous driving. However, the reconstruction-oriented representation learning tangles perception with planning tasks, leading to suboptimal optimization for planning. To address this challenge, we propose WorldRFT, a planning-oriented latent world model framework that aligns scene representation learning with planning via a hierarchical planning decomposition and local-aware interactive refinement mechanism, augmented by reinforcement learning fine-tuning (RFT) to enhance safety-critical policy performance. Specifically, WorldRFT integrates a vision-geometry foundation model to improve 3D spatial awareness, employs hierarchical planning task decomposition to guide representation optimization, and utilizes local-aware iterative refinement to derive a planning-oriented driving policy. Furthermore, we introduce Group Relative Policy Optimization (GRPO), which applies trajectory Gaussianization and collision-aware rewards to fine-tune the driving policy, yielding systematic improvements in safety. WorldRFT achieves state-of-the-art (SOTA) performance on both open-loop nuScenes and closed-loop NavSim benchmarks. On nuScenes, it reduces collision rates by 83% (0.30% -> 0.05%). On NavSim, using camera-only sensors input, it attains competitive performance with the LiDAR-based SOTA method DiffusionDrive (87.8 vs. 88.1 PDMS).

---

## 26. DTCCL: Disengagement-Triggered Contrastive Continual Learning for Autonomous Bus Planners

**论文链接:** [http://arxiv.org/abs/2512.18988v1](http://arxiv.org/abs/2512.18988v1)

**作者:** Yanding Yang, Weitao Zhou, Jinhai Wang, Xiaomin Guo, Junze Wen, Xiaolong Liu, Lang Ding, Zheng Fu, Jinyu Miao, Kun Jiang, Diange Yang

**发布时间:** 2025-12-22

### GPT解析

### 总结

本文提出了一种脱离触发对比持续学习(DTCCL)框架，通过真实世界运营改进自动驾驶公交车的规划策略，无需人工监督即可显著提升性能。

### 背景

自动驾驶公交车在固定路线上运行，但面临开放、动态的城市环境。脱离事件通常地理上集中在高度互动区域，传统模仿学习方法难以纠正这些策略层面的失败，因为容易对稀疏的脱离数据过拟合。

### 目的

开发一种能够通过自动驾驶公交车的真实世界运营来持续改进其规划策略的框架，解决传统学习方法在处理稀疏脱离数据时的局限性。

### 方法

提出DTCCL框架，每次脱离事件触发基于云的数据增强，通过扰动周围代理同时保留路线上下文生成正负样本；利用对比学习优化策略表示以更好区分安全与不安全行为；在云-边缘循环中应用持续更新，无需人工监督。

### 主要发现

在城市公交路线上的实验表明，与直接重新训练相比，DTCCL将整体规划性能提高了48.6%

### 结论

DTCCL验证了其在自动驾驶公共交通中可扩展的闭环策略改进的有效性，为自动驾驶公交车的安全运营提供了新的解决方案

### 翻译

自动驾驶公交车在固定路线上运行，但必须在开放、动态的城市环境中运行。这些路线上的脱离事件通常地理上集中，并且通常发生在高度互动区域。这些策略层面的失败很难通过传统的模仿学习来纠正，因为传统模仿学习容易对稀疏的脱离数据过拟合。为解决这一问题，本文提出了脱离触发对比持续学习(DTCCL)框架，使自动驾驶公交车能够通过真实世界运营改进规划策略。每次脱离事件触发基于云的数据增强，通过扰动周围代理同时保留路线上下文来生成正负样本。对比学习优化策略表示以更好地区分安全和不安全行为，并在云-边缘循环中应用持续更新，无需人工监督。城市公交路线上的实验表明，与直接重新训练相比，DTCCL将整体规划性能提高了48.6%，验证了其在自动驾驶公共交通中可扩展的闭环策略改进的有效性。


### 论文摘要

Autonomous buses run on fixed routes but must operate in open, dynamic urban environments. Disengagement events on these routes are often geographically concentrated and typically arise from planner failures in highly interactive regions. Such policy-level failures are difficult to correct using conventional imitation learning, which easily overfits to sparse disengagement data. To address this issue, this paper presents a Disengagement-Triggered Contrastive Continual Learning (DTCCL) framework that enables autonomous buses to improve planning policies through real-world operation. Each disengagement triggers cloud-based data augmentation that generates positive and negative samples by perturbing surrounding agents while preserving route context. Contrastive learning refines policy representations to better distinguish safe and unsafe behaviors, and continual updates are applied in a cloud-edge loop without human supervision. Experiments on urban bus routes demonstrate that DTCCL improves overall planning performance by 48.6 percent compared with direct retraining, validating its effectiveness for scalable, closed-loop policy improvement in autonomous public transport.

---

## 27. 论文ID: 2512.18951v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18951v1.json'

---

## 28. KeenKT: Knowledge Mastery-State Disambiguation for Knowledge Tracing

**论文链接:** [http://arxiv.org/abs/2512.18709v1](http://arxiv.org/abs/2512.18709v1)

**作者:** Zhifei Li, Lifan Chen, Jiali Yi, Xiaoju Hou, Yue Zhao, Wenxin Huang, Miao Zhang, Kui Xiao, Bing Yang

**发布时间:** 2025-12-21

**备注:** Accepted by the Association for the Advancement of Artificial Intelligence 2026(AAAI2026)

### GPT解析

### 总结

论文提出了一种名为KeenKT的知识追踪模型，通过使用正态逆高斯分布表示学生知识状态，解决了传统单点估计方法无法区分真实能力与偶然表现的问题，显著提高了预测准确性和对行为波动的敏感性。

### 背景

知识追踪(KT)旨在根据学生历史学习交互动态建模学生对知识概念的掌握程度。目前大多数方法依赖于单点估计，无法区分真实能力与爆发性表现或粗心，从而在判断掌握程度时产生歧义。

### 目的

解决当前KT方法中单点估计导致的歧义问题，更准确地捕捉学生学习行为的波动性，提高知识追踪的准确性。

### 方法

提出知识掌握状态消歧知识追踪(KeenKT)模型，使用正态逆高斯(NIG)分布表示学生知识状态，设计基于NIG距离的注意力机制建模知识状态动态演化，并引入基于扩散的去噪重建损失和分布对比学习损失增强模型鲁棒性。

### 主要发现

在六个公共数据集上进行的实验表明，KeenKT在预测准确性和对行为波动的敏感性方面优于最先进的KT模型，最大AUC改进为5.85%，最大ACC改进为6.89%。

### 结论

KeenKT模型通过分布表示而非单点估计，能够更好地捕捉学生知识掌握状态的变化，在预测准确性和鲁棒性方面都有显著提升，为知识追踪领域提供了新的有效方法。

### 翻译

知识追踪(KT)旨在根据学生历史学习交互动态建模学生对知识概念的掌握程度。大多数当前方法依赖于单点估计，无法区分真实能力与爆发性表现或粗心，从而在判断掌握程度时产生歧义。为解决这一问题，我们提出了知识掌握状态消歧知识追踪模型(KeenKT)，该模型使用正态逆高斯(NIG)分布表示学生在每次交互时的知识状态，从而捕捉学习行为的波动。此外，我们设计了一种基于NIG距离的注意力机制来建模知识状态的动态演化。同时，我们引入了一种基于扩散的去噪重建损失和分布对比学习损失，以增强模型的鲁棒性。在六个公共数据集上的大量实验表明，KeenKT在预测准确性和对行为波动的敏感性方面优于最先进的KT模型。所提出的方法最大AUC改进为5.85%，最大ACC改进为6.89%。


### 论文摘要

Knowledge Tracing (KT) aims to dynamically model a student's mastery of knowledge concepts based on their historical learning interactions. Most current methods rely on single-point estimates, which cannot distinguish true ability from outburst or carelessness, creating ambiguity in judging mastery. To address this issue, we propose a Knowledge Mastery-State Disambiguation for Knowledge Tracing model (KeenKT), which represents a student's knowledge state at each interaction using a Normal-Inverse-Gaussian (NIG) distribution, thereby capturing the fluctuations in student learning behaviors. Furthermore, we design an NIG-distance-based attention mechanism to model the dynamic evolution of the knowledge state. In addition, we introduce a diffusion-based denoising reconstruction loss and a distributional contrastive learning loss to enhance the model's robustness. Extensive experiments on six public datasets demonstrate that KeenKT outperforms SOTA KT models in terms of prediction accuracy and sensitivity to behavioral fluctuations. The proposed method yields the maximum AUC improvement of 5.85% and the maximum ACC improvement of 6.89%.

---

## 29. Modality-Dependent Memory Mechanisms in Cross-Modal Neuromorphic Computing

**论文链接:** [http://arxiv.org/abs/2512.18575v1](http://arxiv.org/abs/2512.18575v1)

**作者:** Effiong Blessing, Chiung-Yi Tseng, Somshubhra Roy, Junaid Rehman, Isaac Nkrumah

**发布时间:** 2025-12-21

### GPT解析

### 总结

该研究首次全面评估了记忆增强型脉冲神经网络(SNNs)的跨模态泛化能力，发现不同记忆机制在视觉和听觉任务上表现各异，Hopfield网络模态特异性强，监督对比学习更平衡，HGRN多模态联合训练表现优异。

### 背景

记忆增强型脉冲神经网络(SNNs)有望实现能效高效的神经形态计算，但其在不同感官模态间的泛化能力尚未被探索。

### 目的

研究SNN中记忆机制在视觉和听觉神经形态数据集上的跨模态表现，评估不同记忆机制的适用性和效率。

### 方法

对三种记忆机制(Hopfield网络、分层门控循环网络HGRNs和监督对比学习SCL)进行跨模态消融研究，在视觉(N-MNIST)和听觉(SHD)神经形态数据集上评估五种架构的性能。

### 主要发现

Hopfield网络在视觉任务上准确率达97.68%，但听觉任务仅76.15%，显示严重的模态特异性；监督对比学习表现出更平衡的跨模态性能(视觉96.72%，听觉82.16%)；HGRN多模态联合训练达到94.41%视觉和79.37%听觉准确率；记忆机制表现出任务特定优势而非普遍适用性；定量印迹分析证实弱跨模态对齐(0.038相似度)，验证了并行架构设计的合理性。

### 结论

该研究提供了神经形态系统中模态特定记忆优化的首个实证证据，实现了比传统神经网络高603倍的能效。

### 翻译

记忆增强型脉冲神经网络(SNNs)有望实现能效高效的神经形态计算，但它们在感官模态间的泛化能力仍未被探索。我们首次对SNN中的记忆机制进行了全面的跨模态消融研究，评估了在视觉(N-MNIST)和听觉(SHD)神经形态数据集上的Hopfield网络、分层门控循环网络(HGRNs)和监督对比学习(SCL)的表现。我们对五种架构的系统评估揭示了显著的模态依赖性性能模式：Hopfield网络在视觉任务上达到97.68%的准确率，但在听觉任务上仅为76.15%(相差21.53个百分点)，显示出严重的模态特异性专业化，而监督对比学习表现出更平衡的跨模态性能(视觉96.72%，听觉82.16%，相差14.56个百分点)。这些发现确立了记忆机制表现出特定任务的优势，而非普遍适用性。使用HGRN进行联合多模态训练实现了94.41%的视觉和79.37%的听觉准确率(平均88.78%)，通过统一部署与并行HGRN性能相当。定量印迹分析证实了弱跨模态对齐(0.038相似度)，验证了我们的并行架构设计。我们的工作为神经形态系统中的模态特定记忆优化提供了首个实证证据，实现了比传统神经网络高603倍的能效。


### 论文摘要

Memory-augmented spiking neural networks (SNNs) promise energy-efficient neuromorphic computing, yet their generalization across sensory modalities remains unexplored. We present the first comprehensive cross-modal ablation study of memory mechanisms in SNNs, evaluating Hopfield networks, Hierarchical Gated Recurrent Networks (HGRNs), and supervised contrastive learning (SCL) across visual (N-MNIST) and auditory (SHD) neuromorphic datasets. Our systematic evaluation of five architectures reveals striking modality-dependent performance patterns: Hopfield networks achieve 97.68% accuracy on visual tasks but only 76.15% on auditory tasks (21.53 point gap), revealing severe modality-specific specialization, while SCL demonstrates more balanced cross-modal performance (96.72% visual, 82.16% audio, 14.56 point gap). These findings establish that memory mechanisms exhibit task-specific benefits rather than universal applicability. Joint multi-modal training with HGRN achieves 94.41% visual and 79.37% audio accuracy (88.78% average), matching parallel HGRN performance through unified deployment. Quantitative engram analysis confirms weak cross-modal alignment (0.038 similarity), validating our parallel architecture design. Our work provides the first empirical evidence for modality-specific memory optimization in neuromorphic systems, achieving 603x energy efficiency over traditional neural networks.

---

## 30. 论文ID: 2512.18368v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18368v1.json'

---

## 31. 论文ID: 2512.18133v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18133v1.json'

---

## 32. Parameter-Efficient Fine-Tuning for HAR: Integrating LoRA and QLoRA into Transformer Models

**论文链接:** [http://arxiv.org/abs/2512.17983v1](http://arxiv.org/abs/2512.17983v1)

**作者:** Irina Seregina, Philippe Lalanda, German Vega

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文研究了参数高效的微调技术在人类活动识别中的应用，特别是低秩适应(LoRA)和量化LoRA(Quantized LoRA)，作为全模型微调的可扩展替代方案。

### 背景

人类活动识别是普适计算的基础任务。尽管自监督学习和基于Transformer的架构最近显著提高了HAR性能，但将大型预训练模型适应到新领域仍面临挑战，主要受限于目标设备的计算资源。

### 目的

研究参数高效的微调技术，特别是LoRA和QLoRA，探索它们作为全模型微调替代方案的可行性和效果，以解决资源受限环境中的模型适应问题。

### 方法

提出一个基于掩码自编码器骨干的适应框架，并在五个开放HAR数据集上使用留一数据集验证协议(Leave-One-Dataset-Out)进行性能评估。

### 主要发现

LoRA和QLoRA能够达到与全微调相当的识别性能，同时显著减少可训练参数数量、内存使用和训练时间。LoRA在有限监督下仍保持稳健性能，适配器秩提供了准确性和效率间的可控权衡。QLoRA通过量化减少冻结权重内存占用，对分类质量影响最小。

### 结论

参数高效的微调技术(LoRA和QLoRA)是全模型微调的有效替代方案，能够在保持高性能的同时显著减少资源需求，特别适合资源受限的HAR应用场景。

### 翻译

人类活动识别是普适计算的基础任务。尽管最近自监督学习和基于Transformer架构的进展显著提高了HAR性能，但由于目标设备上计算资源有限，将大型预训练模型适应到新领域仍然是一个实际挑战。本文研究了参数高效的微调技术，特别是低秩适应(LoRA)和量化LoRA，作为全模型微调的可扩展替代方案用于HAR。我们提出了一个基于掩码自编码器骨干的适应框架，并在五个开放HAR数据集上使用留一数据集验证协议评估其性能。实验证明，LoRA和QLoRA都能匹配全微调的识别性能，同时显著减少可训练参数数量、内存使用和训练时间。进一步分析显示，LoRA即使在有限监督下也能保持稳健性能，且适配器秩提供了准确性和效率之间可控的权衡。QLoRA通过量化减少冻结权重的内存占用，同时最小化对分类质量的影响。


### 论文摘要

Human Activity Recognition is a foundational task in pervasive computing. While recent advances in self-supervised learning and transformer-based architectures have significantly improved HAR performance, adapting large pretrained models to new domains remains a practical challenge due to limited computational resources on target devices. This papers investigates parameter-efficient fine-tuning techniques, specifically Low-Rank Adaptation (LoRA) and Quantized LoRA, as scalable alternatives to full model fine-tuning for HAR. We propose an adaptation framework built upon a Masked Autoencoder backbone and evaluate its performance under a Leave-One-Dataset-Out validation protocol across five open HAR datasets. Our experiments demonstrate that both LoRA and QLoRA can match the recognition performance of full fine-tuning while significantly reducing the number of trainable parameters, memory usage, and training time. Further analyses reveal that LoRA maintains robust performance even under limited supervision and that the adapter rank provides a controllable trade-off between accuracy and efficiency. QLoRA extends these benefits by reducing the memory footprint of frozen weights through quantization, with minimal impact on classification quality.

---

## 33. Skeleton-Snippet Contrastive Learning with Multiscale Feature Fusion for Action Localization

**论文链接:** [http://arxiv.org/abs/2512.16504v2](http://arxiv.org/abs/2512.16504v2)

**作者:** Qiushuo Cheng, Jingjing Liu, Catherine Morgan, Alan Whone, Majid Mirmehdi

**发布时间:** 2025-12-18

### GPT解析

### 总结

本文提出了一种基于片段判别的自监督预训练方法，用于解决基于骨架的时间动作定位问题，通过对比学习提升特征区分能力，并结合U形模块增强特征分辨率，在多个数据集上取得了优异的性能。

### 背景

自监督预训练范式在基于骨架的动作识别中取得了很大成功，但在基于骨架的时间动作定位方面仍然具有挑战性且研究不足。

### 目的

开发一种能够学习有效时间敏感特征的自监督方法，用于基于骨架的时间动作定位，以捕捉动作边界处相邻帧之间的细微差别。

### 方法

提出片段判别自监督预训练任务，将骨架序列密集投影到不重叠片段，通过对比学习促进跨视频片段特征区分；结合U形模块融合中间特征，增强帧级定位的特征分辨率。

### 主要发现

所提方法在BABEL数据集上对现有基于骨架的对比学习方法进行了改进，在多个子集和评估协议上表现一致；在NTU RGB+D和BABEL上预训练后，在PKUMMD上实现了最先进的迁移学习性能。

### 结论

基于片段判别的自监督预训练方法结合特征增强技术能够有效提升基于骨架的时间动作定位性能，为该领域提供了新的研究方向。

### 翻译

自监督预训练范式在通过对比学习学习基于骨架的动作识别的3D动作表示方面取得了巨大成功。然而，学习用于基于骨架的时间动作定位的有效表示仍然具有挑战性且研究不足。与视频级动作识别不同，检测动作边界需要时间敏感的特征，这些特征能够捕捉标签变化处相邻帧之间的细微差别。为此，我们提出了一个用于自监督预训练的片段判别预训练任务，该方法将骨架序列密集投影到不重叠的片段中，并通过对比学习促进跨视频的片段特征区分。此外，我们通过融合中间特征和U形模块构建了基于骨架的动作识别模型的强大骨干网络，以提高帧级定位的特征分辨率。我们的方法在BABEL上对现有的基于骨架的对比学习方法进行了改进，在多个子集和评估协议上表现一致。我们还在NTU RGB+D和BABEL上进行预训练后，在PKUMMD上实现了最先进的迁移学习性能。


### 论文摘要

The self-supervised pretraining paradigm has achieved great success in learning 3D action representations for skeleton-based action recognition using contrastive learning. However, learning effective representations for skeleton-based temporal action localization remains challenging and underexplored. Unlike video-level {action} recognition, detecting action boundaries requires temporally sensitive features that capture subtle differences between adjacent frames where labels change. To this end, we formulate a snippet discrimination pretext task for self-supervised pretraining, which densely projects skeleton sequences into non-overlapping segments and promotes features that distinguish them across videos via contrastive learning. Additionally, we build on strong backbones of skeleton-based action recognition models by fusing intermediate features with a U-shaped module to enhance feature resolution for frame-level localization. Our approach consistently improves existing skeleton-based contrastive learning methods for action localization on BABEL across diverse subsets and evaluation protocols. We also achieve state-of-the-art transfer learning performance on PKUMMD with pretraining on NTU RGB+D and BABEL.

---

## 34. SCS-SupCon: Sigmoid-based Common and Style Supervised Contrastive Learning with Adaptive Decision Boundaries

**论文链接:** [http://arxiv.org/abs/2512.17954v1](http://arxiv.org/abs/2512.17954v1)

**作者:** Bin Wang, Fadi Dornaika

**发布时间:** 2025-12-17

### GPT解析

### 总结

本文提出了一种基于Sigmoid的通用和风格监督对比学习方法（SCS-SupCon），解决了图像分类中的类间差异小和类内变化大问题，通过引入基于Sigmoid的成对对比损失和风格距离约束，显著提升了细粒度识别任务的性能。

### 背景

图像分类受到类间差异小和类内变化大的阻碍，限制了现有对比学习方法的有效性。基于InfoNCE损失的有监督对比方法存在负样本稀释问题，且缺乏自适应决策边界，降低了细粒度识别任务的判别能力。

### 目的

解决现有对比学习方法的局限性，提出一种能够自适应决策边界、减轻负样本稀释并有效利用监督的新型对比学习方法。

### 方法

提出SCS-SupCon框架，引入基于Sigmoid的成对对比损失，具有可学习的温度和偏置参数以实现自适应决策边界，强调困难负样本并减轻负样本稀释；同时添加明确的风格距离约束，解耦风格和内容表示，实现更稳健的特征学习。

### 主要发现

在六个基准数据集上的实验表明，SCS-SupCon在CNN和Transformer骨干网络上均达到最先进性能。在CIFAR-100上使用ResNet-50，比SupCon提高约3.9个百分点top-1准确率，比CS-SupCon提高约1.7个百分点；在细粒度数据集上，比CS-SupCon高出0.4-3.0个百分点。消融研究和统计分析确认了方法的稳健性和泛化能力。

### 结论

Friedman检验和Nemenyi事后评估验证了所观察到的改进具有统计显著性，证明了SCS-SupCon框架的稳定性和有效性。

### 翻译

图像分类受到类间细微差异和类内显著变化的阻碍，这限制了现有对比学习方法的有效性。基于InfoNCE损失的有监督对比方法遭受负样本稀释问题，且缺乏自适应决策边界，从而降低了细粒度识别任务的判别能力。为解决这些限制，我们提出了基于Sigmoid的通用和风格监督对比学习（SCS-SupCon）。我们的框架引入了基于Sigmoid的成对对比损失，具有可学习的温度和偏置参数，以实现自适应决策边界。这种公式强调困难负样本，减轻负样本稀释，并更有效地利用监督。此外，明确的风格距离约束进一步解耦了风格和内容表示，导致更稳健的特征学习。在包括CUB200-2011和Stanford Dogs在内的六个基准数据集上的综合实验表明，SCS-SupCon在CNN和Transformer骨干网络上均实现了最先进的性能。在CIFAR-100上使用ResNet-50，SCS-SupCon在五折交叉验证下比SupCon提高约3.9个百分点的top-1准确率，比CS-SupCon提高约1.7个百分点。在细粒度数据集上，它比CS-SupCon高出0.4-3.0个百分点。大量的消融研究和统计分析进一步确认了所提出框架的稳健性和泛化能力，Friedman检验和Nemenyi事后评估验证了所观察到的改进的稳定性。


### 论文摘要

Image classification is hindered by subtle inter-class differences and substantial intra-class variations, which limit the effectiveness of existing contrastive learning methods. Supervised contrastive approaches based on the InfoNCE loss suffer from negative-sample dilution and lack adaptive decision boundaries, thereby reducing discriminative power in fine-grained recognition tasks. To address these limitations, we propose Sigmoid-based Common and Style Supervised Contrastive Learning (SCS-SupCon). Our framework introduces a sigmoid-based pairwise contrastive loss with learnable temperature and bias parameters to enable adaptive decision boundaries. This formulation emphasizes hard negatives, mitigates negative-sample dilution, and more effectively exploits supervision. In addition, an explicit style-distance constraint further disentangles style and content representations, leading to more robust feature learning. Comprehensive experiments on six benchmark datasets, including CUB200-2011 and Stanford Dogs, demonstrate that SCS-SupCon achieves state-of-the-art performance across both CNN and Transformer backbones. On CIFAR-100 with ResNet-50, SCS-SupCon improves top-1 accuracy over SupCon by approximately 3.9 percentage points and over CS-SupCon by approximately 1.7 points under five-fold cross-validation. On fine-grained datasets, it outperforms CS-SupCon by 0.4--3.0 points. Extensive ablation studies and statistical analyses further confirm the robustness and generalization of the proposed framework, with Friedman tests and Nemenyi post-hoc evaluations validating the stability of the observed improvements.

---

## 35. 论文ID: 2512.19360v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19360v1.json'

---

## 36. DIVER-1 : Deep Integration of Vast Electrophysiological Recordings at Scale

**论文链接:** [http://arxiv.org/abs/2512.19097v1](http://arxiv.org/abs/2512.19097v1)

**作者:** Danny Dongyeop Han, Yonghyeon Gwon, Ahhyun Lucy Lee, Taeyang Lee, Seong Jin Lee, Jubin Choi, Sebin Lee, Jihyun Bang, Seungju Lee, David Keetae Park, Shinjae Yoo, Chun Kee Chung, Jiook Cha

**发布时间:** 2025-12-22

**备注:** 47 pages, 13 figures, 26 tables

### GPT解析

### 总结

该研究介绍了DIVER-1，一种基于最大规模多样化脑电生理信号数据集训练的基础模型家族，实现了在EEG和iEEG任务上的最先进性能，并首次揭示了该领域的扩展规律。

### 背景

脑电图(EEG)和颅内脑电图(iEEG)信号在神经科学、脑机接口和临床应用中至关重要，但现有的基础模型规模有限，尽管有明确证据表明扩大规模可以提高性能。

### 目的

开发大规模EEG和iEEG基础模型，研究该领域的扩展规律，并提供高效扩展和资源分配的具体指导。

### 方法

训练了迄今为止最大和最多样化的语料库(5.3k小时iEEG和54k小时EEG，来自17.7k+受试者)，将模型规模扩大到18.2亿参数，设计了任意变量注意力、滑动时态条件位置编码和多域重建等架构创新。

### 主要发现

首次对脑电生理学领域进行系统扩展规律分析，发现该领域遵循数据约束的扩展规律：对于给定数据和计算量，训练时间更长的小型模型优于训练时间较短的大型模型，这与之前强调模型规模而非训练时间的模型形成对比。

### 结论

DIVER-1的iEEG和EEG模型分别在各自基准测试中实现最先进性能，为脑电生理学基础模型开发中的高效扩展和资源分配提供了具体指导。

### 翻译

脑电图信号如EEG和iEEG是神经科学、脑机接口和临床应用的核心，然而尽管有明确证据表明扩大规模可以提高性能，现有的基础模型仍然规模有限。我们介绍了DIVER-1，一种迄今为止在最大和最多样化语料库上训练的EEG和iEEG基础模型家族-5.3k小时的iEEG和54k小时的EEG(来自超过17.7k受试者的160万通道小时)-并将规模扩大到18.2亿参数。我们首次对该领域进行了系统的扩展规律分析，表明它们遵循数据约束的扩展规律：对于给定的数据和计算量，训练时间更长的小型模型始终优于训练时间较短的大型模型。这种行为与之前强调模型规模而非训练时间的脑电生理学基础模型形成对比。为实现强性能，我们还设计了架构创新，包括任意变量注意力、滑动时态条件位置编码和多域重建。DIVER-1的iEEG和EEG模型分别在各自基准测试中实现了最先进的性能，为脑电生理学基础模型开发中的高效扩展和资源分配提供了具体指导。


### 论文摘要

Electrophysiology signals such as EEG and iEEG are central to neuroscience, brain-computer interfaces, and clinical applications, yet existing foundation models remain limited in scale despite clear evidence that scaling improves performance. We introduce DIVER-1, a family of EEG and iEEG foundation models trained on the largest and most diverse corpus to date-5.3k hours of iEEG and 54k hours of EEG (1.6M channel-hours from over 17.7k subjects)-and scaled up to 1.82B parameters. We present the first systematic scaling law analysis for this domain, showing that they follow data-constrained scaling laws: for a given amount of data and compute, smaller models trained for extended epochs consistently outperform larger models trained briefly. This behavior contrasts with prior electrophysiology foundation models that emphasized model size over training duration. To achieve strong performance, we also design architectural innovations including any-variate attention, sliding temporal conditional positional encoding, and multi-domain reconstruction. DIVER-1 iEEG and EEG models each achieve state-of-the-art performance on their respective benchmarks, establishing a concrete guidelines for efficient scaling and resource allocation in electrophysiology foundation model development.

---

## 37. 论文ID: 2512.19090v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19090v1.json'

---

## 38. CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models

**论文链接:** [http://arxiv.org/abs/2512.19083v1](http://arxiv.org/abs/2512.19083v1)

**作者:** Pengyu Chen, Tao Ouyang, Ke Luo, Weijie Hong, Xu Chen

**发布时间:** 2025-12-22

**备注:** This paper is accepted by the IEEE Internet of Things Journal (IoT-J) for publication in the Special Issue on "Augmented Edge Sensing Intelligence for Low-Altitude IoT Systems"

### GPT解析

### 总结

CoDrone是一种创新的云-边缘-端协同计算框架，通过集成基础模型解决无人机自主导航中的计算资源限制问题，显著提升了性能。

### 背景

无人机自主导航面临机载计算资源有限的挑战，限制了深度神经网络的复杂度；将任务卸载到远程边缘服务器会引入高延迟，造成系统设计中的固有权衡。

### 目的

解决资源受限无人机平台的自主导航问题，通过集成基础模型提升性能，减少机载计算和数据传输开销，实现高效的环境感知和导航决策。

### 方法

采用灰度图像减少计算开销；使用边缘辅助的Depth Anything V2模型进行深度估计；引入基于一维占用网格的导航方法；设计基于深度强化学习的神经调度器；开发无人机特定的视觉语言交互模块，实现云端基础模型与无人机的有效交互。

### 主要发现

实验表明CoDrone在不同飞行速度和网络条件下优于基线方法，实现平均飞行距离增加40%，平均导航质量提高5%。

### 结论

CoDrone成功解决了无人机自主导航中的计算资源限制问题，通过云-边缘-端协同计算框架和基础模型的集成，实现了高效且适应性强的自主导航系统。

### 翻译

无人机自主导航面临来自机载计算资源有限的关键挑战，这限制了部署的深度神经网络只能使用浅层架构，无法处理复杂环境。将任务卸载到远程边缘服务器会引入高延迟，造成系统设计中的固有权衡。为解决这些限制，我们提出了CoDrone——首个将基础模型集成到无人机自主巡航场景的云-边缘-端协同计算框架——有效利用基础模型提升资源受限无人机平台的性能。为减少机载计算和数据传输开销，CoDrone使用灰度图像进行导航模型。当需要增强环境感知时，CoDrone利用边缘辅助的基础模型Depth Anything V2进行深度估计，并引入基于一维占用网格的新型导航方法——在提高自主导航效率和表示简单性的同时实现细粒度场景理解。CoDrone的关键组件是一个基于深度强化学习的神经调度器，无缝集成深度估计与自主导航决策，实现实时适应动态环境。此外，该框架引入了无人机特定的视觉语言交互模块，包含领域定制的低级飞行原语，实现云端基础模型与无人机之间的有效交互。VLM的引入增强了在复杂未见场景中的开放集推理能力。实验结果表明，CoDrone在不同飞行速度和网络条件下优于基线方法，实现了平均飞行距离增加40%和平均导航质量提高5%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决无人机自主导航中的计算资源限制和网络延迟问题。由于无人机上的计算资源有限，通常只能部署浅层神经网络，无法处理复杂环境；而将任务卸载到远程边缘服务器又会引入高延迟。这个问题很重要，因为低空经济正在成为全球数字和工业转型的重要驱动力，预计到2030年市场价值将达到4000亿美元，而无人机作为低空物联网系统的主要平台，在精准农业、智能制造、无人机配送等领域应用广泛，自主导航对它们完成任务至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了无人机导航面临的三大挑战：传统DNN方法在复杂环境中的局限性、网络条件波动带来的性能问题、以及VLMs计算需求高的部署障碍。针对这些问题，作者设计了三层解决方案：使用灰度图像减少计算负担，在边缘部署深度估计模型增强环境感知；引入基于DRL的神经调度器优化任务分配；设计无人机特定的视觉语言交互框架实现云端智能调用。作者借鉴了现有工作如Depth Anything V2深度估计模型、Qwen-VL-Max视觉语言模型、A2C调度算法等，但进行了创新性整合和应用。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建云-边-端协同计算框架，将基础模型集成到无人机自主巡航中，通过结合灰度图像处理、深度估计、一维占用网格导航和神经调度器，实现资源受限条件下的高效自主导航。整体流程：1)无人机采集灰度图像；2)神经调度器决定处理位置和压缩比；3)导航模块处理图像推断转向角和碰撞概率；4)边缘服务器使用深度估计模型生成占用网格地图；5)控制算法调整飞行命令；6)调用策略模块决定是否激活云端视觉语言模型；7)飞行控制器执行最终命令控制无人机。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首创云-边-端协同自主无人机导航框架；2)使用灰度图像和轻量级DNN优化计算；3)引入一维占用网格导航方法增强环境感知；4)设计无人机特定的视觉语言交互模块。相比之前工作：与传统DNN方法相比，增强了复杂环境中的环境感知能力；与adaDrone等边缘辅助框架相比，更好地平衡了计算卸载和本地处理；与早期DRL路径规划相比，提供了更强的环境理解能力；与直接部署VLM的方法相比，通过云端部署和智能调用策略解决了计算需求问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CoDrone通过创新性地整合云-边-端计算架构和基础模型，解决了无人机自主导航中的计算资源限制和环境感知挑战，实现了在复杂环境中的高效、安全自主飞行。'}


### 论文摘要

Autonomous navigation for Unmanned Aerial Vehicles faces key challenges from limited onboard computational resources, which restrict deployed deep neural networks to shallow architectures incapable of handling complex environments. Offloading tasks to remote edge servers introduces high latency, creating an inherent trade-off in system design. To address these limitations, we propose CoDrone - the first cloud-edge-end collaborative computing framework integrating foundation models into autonomous UAV cruising scenarios - effectively leveraging foundation models to enhance performance of resource-constrained unmanned aerial vehicle platforms. To reduce onboard computation and data transmission overhead, CoDrone employs grayscale imagery for the navigation model. When enhanced environmental perception is required, CoDrone leverages the edge-assisted foundation model Depth Anything V2 for depth estimation and introduces a novel one-dimensional occupancy grid-based navigation method - enabling fine-grained scene understanding while advancing efficiency and representational simplicity of autonomous navigation. A key component of CoDrone is a Deep Reinforcement Learning-based neural scheduler that seamlessly integrates depth estimation with autonomous navigation decisions, enabling real-time adaptation to dynamic environments. Furthermore, the framework introduces a UAV-specific vision language interaction module incorporating domain-tailored low-level flight primitives to enable effective interaction between the cloud foundation model and the UAV. The introduction of VLM enhances open-set reasoning capabilities in complex unseen scenarios. Experimental results show CoDrone outperforms baseline methods under varying flight speeds and network conditions, achieving a 40% increase in average flight distance and a 5% improvement in average Quality of Navigation.

---

## 39. CETCAM: Camera-Controllable Video Generation via Consistent and Extensible Tokenization

**论文链接:** [http://arxiv.org/abs/2512.19020v1](http://arxiv.org/abs/2512.19020v1)

**作者:** Zelin Zhao, Xinyu Gong, Bangya Liu, Ziyang Song, Jun Zhang, Suhui Wu, Yongxin Chen, Hao Zhang

**发布时间:** 2025-12-22

### GPT解析

### 总结

论文介绍了一种名为CETCAM的相机可控视频生成框架，通过一致的标记化方案消除对相机标注的需求，利用几何基础模型估计深度和相机参数，并采用渐进式训练策略实现高质量视频生成。

### 背景

在视频生成中实现精确的相机控制具有挑战性，现有方法依赖相机姿态标注，这些标注难以扩展到大型动态数据集，且常与深度估计不一致，导致训练-测试差异。

### 目的

开发一种不需要相机标注的相机可控视频生成框架，解决现有方法在可扩展性和一致性方面的问题。

### 方法

引入CETCAM框架，利用几何基础模型估计深度和相机参数并转换为几何感知标记，通过轻量级上下文块集成到预训练的视频扩散网络中，采用两阶段渐进式训练：先从原始视频数据学习相机可控性，再用高保真数据集细化视觉质量。

### 主要发现

CETCAM在多个基准测试中实现了最先进的几何一致性、时间稳定性和视觉真实性，同时对修复和布局控制等额外控制模态表现出强大的适应性。

### 结论

CETCAM成功解决了视频生成中精确相机控制的挑战，通过创新的标记化方法和渐进式训练策略实现了无需相机标注的高质量视频生成，具有良好的可扩展性和适应性。

### 翻译

在视频生成中实现精确的相机控制仍然具有挑战性，因为现有方法通常依赖于相机姿态标注，这些标注难以扩展到大型和动态数据集，并且经常与深度估计不一致，导致训练-测试差异。我们引入了CETCAM，一种相机可控的视频生成框架，它通过一致的、可扩展的标记化方案消除了对相机标注的需求。CETCAM利用几何基础模型（如VGGT）的最新进展来估计深度和相机参数，并将它们转换为统一的、几何感知的标记。这些标记通过轻量级上下文块无缝集成到预训练的视频扩散主干网络中。CETCAM采用两个渐进阶段进行训练：首先从多样化的原始视频数据中学习强大的相机可控性，然后使用精选的高保真数据集细化细粒度的视觉质量。在多个基准测试上的广泛实验证明了最先进的几何一致性、时间稳定性和视觉真实性。此外，CETCAM对额外的控制模态（包括修复和布局控制）表现出强大的适应性，突显了其在相机控制之外的灵活性。项目页面可在https://sjtuytc.github.io/CETCam_project_page.github.io/获取。


### 论文摘要

Achieving precise camera control in video generation remains challenging, as existing methods often rely on camera pose annotations that are difficult to scale to large and dynamic datasets and are frequently inconsistent with depth estimation, leading to train-test discrepancies. We introduce CETCAM, a camera-controllable video generation framework that eliminates the need for camera annotations through a consistent and extensible tokenization scheme. CETCAM leverages recent advances in geometry foundation models, such as VGGT, to estimate depth and camera parameters and converts them into unified, geometry-aware tokens. These tokens are seamlessly integrated into a pretrained video diffusion backbone via lightweight context blocks. Trained in two progressive stages, CETCAM first learns robust camera controllability from diverse raw video data and then refines fine-grained visual quality using curated high-fidelity datasets. Extensive experiments across multiple benchmarks demonstrate state-of-the-art geometric consistency, temporal stability, and visual realism. Moreover, CETCAM exhibits strong adaptability to additional control modalities, including inpainting and layout control, highlighting its flexibility beyond camera control. The project page is available at https://sjtuytc.github.io/CETCam_project_page.github.io/.

---

## 40. ORPR: An OR-Guided Pretrain-then-Reinforce Learning Model for Inventory Management

**论文链接:** [http://arxiv.org/abs/2512.19001v1](http://arxiv.org/abs/2512.19001v1)

**作者:** Lingjie Zhao, Xue Yu, Yongzhi Qi, Hao Hu, Jianshen Zhang, Yingzheng Ma, Shuyu Han, Wei Qi, Zuo-Jun Max Shen

**发布时间:** 2025-12-22

### GPT解析

### 总结

本文提出了一种创新的OR引导的'预训练-强化'框架，用于弥合AI的自适应感知能力与OR的结构化严谨性之间的差距。通过模拟增强的OR模型生成高质量参考决策，设计领域信息深度学习基础模型，并利用强化学习进行微调，实现了显著的供应链管理改进。

### 背景

随着人工智能(AI)与运筹学(OR)在处理复杂库存系统方面的协同追求日益增强，一个关键挑战依然存在：如何有效调和AI的自适应感知能力与OR的结构化严谨性。

### 目的

为了弥合AI与OR之间的差距，提出一种OR引导的'预训练-强化'框架，结合AI的自适应能力和OR的结构化严谨性，以优化复杂库存系统的管理。

### 方法

1. 提出模拟增强的OR模型生成高质量参考决策；2. 使用OR衍生的决策作为训练标签，设计领域信息深度学习基础模型；3. 通过强化学习微调，使AI代理内化OR的最优性原则；4. 利用探索改进通用策略，允许专家指导进行场景特定适应；5. 通过数值实验和京东实地部署验证效果。

### 主要发现

1. 模型显著优于现有工业实践；2. 库存周转天数减少5.27天；3. 库存率提高2.29%；4. 持有成本降低29.95%；5. 轻量级、领域信息模型在结构化OR逻辑引导下可实现先进性能和强大可转移性。

### 结论

这种方法为智能供应链管理提供了一种可扩展且具有成本效益的范式，突显了深度对齐AI与OR的价值。与单纯依靠模型规模扩大的趋势不同，结合领域知识和结构化逻辑的方法能够取得更好的效果。

### 翻译

随着人工智能(AI)与运筹学(OR)在处理复杂库存系统方面的协同追求日益增强，一个关键挑战依然存在：如何有效调和AI的自适应感知能力与OR的结构化严谨性。为了弥合这一差距，我们提出了一种创新的OR引导的'预训练-强化'框架。为了提供结构化指导，我们提出了一种模拟增强的OR模型，生成高质量的参考决策，隐式地捕捉复杂的业务约束和管理偏好。利用这些OR衍生的决策作为基础训练标签，我们设计了一个领域信息深度学习基础模型，建立基础决策能力，然后是强化学习(RL)微调阶段。独特的是，我们将RL定位为深度对齐机制，使AI代理能够内化OR的最优性原则，同时利用探索进行通用策略改进，并允许专家指导进行场景特定适应（如促销活动）。通过大量数值实验和京东公司的实地部署（辅以双重差分(DiD)分析）验证，我们的模型显著优于现有的工业实践，实现了现实世界的收益：库存周转天数减少5.27天，库存率提高2.29%，同时持有成本降低29.95%。与当前流行的蛮力模型扩展趋势相反，我们的研究表明，当由结构化的OR逻辑引导时，轻量级、领域信息模型可以实现最先进的性能和强大的可转移性。这种方法为智能供应链管理提供了一种可扩展且具有成本效益的范式，突显了深度对齐AI与OR的价值。


### 论文摘要

As the pursuit of synergy between Artificial Intelligence (AI) and Operations Research (OR) gains momentum in handling complex inventory systems, a critical challenge persists: how to effectively reconcile AI's adaptive perception with OR's structural rigor. To bridge this gap, we propose a novel OR-Guided "Pretrain-then-Reinforce" framework. To provide structured guidance, we propose a simulation-augmented OR model that generates high-quality reference decisions, implicitly capturing complex business constraints and managerial preferences. Leveraging these OR-derived decisions as foundational training labels, we design a domain-informed deep learning foundation model to establish foundational decision-making capabilities, followed by a reinforcement learning (RL) fine-tuning stage. Uniquely, we position RL as a deep alignment mechanism that enables the AI agent to internalize the optimality principles of OR, while simultaneously leveraging exploration for general policy refinement and allowing expert guidance for scenario-specific adaptation (e.g., promotional events). Validated through extensive numerical experiments and a field deployment at JD.com augmented by a Difference-in-Differences (DiD) analysis, our model significantly outperforms incumbent industrial practices, delivering real-world gains of a 5.27-day reduction in turnover and a 2.29% increase in in-stock rates, alongside a 29.95% decrease in holding costs. Contrary to the prevailing trend of brute-force model scaling, our study demonstrates that a lightweight, domain-informed model can deliver state-of-the-art performance and robust transferability when guided by structured OR logic. This approach offers a scalable and cost-effective paradigm for intelligent supply chain management, highlighting the value of deeply aligning AI with OR.

---

## 41. Towards AI-Guided Open-World Ecological Taxonomic Classification

**论文链接:** [http://arxiv.org/abs/2512.18994v1](http://arxiv.org/abs/2512.18994v1)

**作者:** Cheng Yaw Low, Heejoon Koo, Jaewoo Park, Kaleb Mesfin Asfaw, Meeyoung Cha

**发布时间:** 2025-12-22

**备注:** 4 figures, 11 tables, and 15 pages

### GPT解析

### 总结

该研究提出了TaxoNet框架，解决AI指导生态分类中的多重挑战，包括类别不平衡、细粒度变异、时空域偏移和封闭集限制，显著提升了稀有分类单元的分类性能。

### 背景

AI指导的生态分类对全球可持续发展如生物多样性监测、保护规划和政策制定至关重要，但面临长尾分类分布、细粒度分类变异、测试时空域偏移和封闭集假设等挑战。

### 目的

引入开放世界生态分类学分类框架，捕捉现实生态环境中这些挑战的共存情况，并提出TaxoNet方法应对这些相互关联的挑战。

### 方法

提出TaxoNet，一种基于嵌入的编码器，采用双边缘惩罚损失函数，加强对稀有代表性分类单元的学习信号，减轻代表性过强分类单元的主导地位。

### 主要发现

模型在Google Auto-Arborist、iNat-Plantae和NAFlora-Mini等多样化生态领域评估中始终优于基线方法，尤其对稀有分类单元表现更好，同时发现通用多模态基础模型在植物领域应用中受限。

### 结论

TaxoNet框架有效解决了生态分类中的多种挑战，为开放世界植物分类监测提供了强大基础，特别提升了稀有分类单元的分类性能。

### 翻译

AI指导的生态科、属、种分类支持全球可持续发展努力，如生物多样性监测、保护规划和政策制定。实现这一目标的进展受到长尾分类分布（由类别不平衡导致）、细粒度分类变异、测试时空域偏移以及只能识别已见过分类单元的封闭集假设的阻碍。我们引入了开放世界生态分类学分类，这是一个统一框架，捕捉了现实生态环境中这些挑战的共存情况。为应对这些挑战，我们提出了TaxoNet，一种基于嵌入的编码器，采用双边缘惩罚损失函数，加强来自稀有代表性分类单元的学习信号，同时减轻代表性过强分类单元的主导地位，直接应对相互关联的挑战。我们在多样化的生态领域评估了我们的方法：Google Auto-Arborist（城市树木）、iNat-Plantae（来自iNaturalist-2019各种生态系统的植物观察）和NAFlora-Mini（精选的标本集收藏）。我们的模型始终优于基线方法，特别是对稀有分类单元，为开放世界植物分类监测奠定了坚实基础。我们的研究进一步表明，通用多模态基础模型在植物领域应用中仍然受限。


### 论文摘要

AI-guided classification of ecological families, genera, and species underpins global sustainability efforts such as biodiversity monitoring, conservation planning, and policy-making. Progress toward this goal is hindered by long-tailed taxonomic distributions from class imbalance, along with fine-grained taxonomic variations, test-time spatiotemporal domain shifts, and closed-set assumptions that can only recognize previously seen taxa. We introduce the Open-World Ecological Taxonomy Classification, a unified framework that captures the co-occurrence of these challenges in realistic ecological settings. To address them, we propose TaxoNet, an embedding-based encoder with a dual-margin penalization loss that strengthens learning signals from rare underrepresented taxa while mitigating the dominance of overrepresented ones, directly confronting interrelated challenges. We evaluate our method on diverse ecological domains: Google Auto-Arborist (urban trees), iNat-Plantae (Plantae observations from various ecosystems in iNaturalist-2019), and NAFlora-Mini (a curated herbarium collection). Our model consistently outperforms baselines, particularly for rare taxa, establishing a strong foundation for open-world plant taxonomic monitoring. Our findings further show that general-purpose multimodal foundation models remain constrained in plant-domain applications.

---

## 42. Foundation Model for Unified Characterization of Optical Quantum States

**论文链接:** [http://arxiv.org/abs/2512.18801v1](http://arxiv.org/abs/2512.18801v1)

**作者:** Xiaoting Gao, Yan Zhu, Feng-Xiao Sun, Ya-Dong Wu, Qiongyi He

**发布时间:** 2025-12-21

### GPT解析

### 总结

该研究提出了第一个用于表征复杂度广泛的光学量子态的基础模型，能够在不进行完整层析的情况下预测广泛的光学量子态特性，特别是多模非高斯态。

### 背景

机器学习方法已被用于推断有限族光学量子态的特定性质，但目前仍缺乏一个统一的模型可以在不进行完整层析的情况下预测广泛的光学量子态特性，特别是多模非高斯态。

### 目的

引入第一个用于表征复杂度广泛的光学量子态的基础模型，该模型由三个关键因素定义：非高斯性、模数和压缩程度。

### 方法

使用在低复杂度态上预训练的单个模型，可以直接应用于表征高复杂度态；通过有限的微调，模型可以适应下游任务，如预测保真度和Wigner负性。

### 主要发现

模型可以表征实验相关态的广泛类别，包括强非高斯Schrödinger猫态、多达十个模的多模系统以及压缩程度高达10.4dB的高度压缩态。

### 结论

建立了一个从有限测量数据表征光学量子态的统一框架，能够有效认证与光学量子信息计算、通信和计量学相关的量子态。

### 翻译

机器学习方法已被用于推断有限族光学量子态的特定性质，但一个能够在不进行完整层析的情况下预测广泛的光学量子态特性（特别是多模非高斯态）的统一模型仍然缺乏。本文我们引入了第一个用于表征广泛复杂度的光学量子态的基础模型，该模型由三个关键因素定义：非高斯性、模数和压缩程度。我们展示了在低复杂度态上预训练的单个模型可以直接应用于表征高复杂度态。通过有限的微调，模型可以适应下游任务，如预测广泛实验相关态类的量子保真度和Wigner负性，包括强非高斯Schrödinger猫态、多达十个模的多模系统以及压缩程度高达10.4dB的高度压缩态。我们的结果建立了一个从有限测量数据表征光学量子态的统一框架，能够有效认证与光学量子信息计算、通信和计量学相关的量子态。


### 论文摘要

Machine learning methods have been used to infer specific properties of limited families of optical quantum states, but a unified model that predicts a broad range of properties for practically relevant-especially multimode non-Gaussian-states without full tomography is still lacking. Here we introduce the first foundation model for the characterization of optical quantum states across a wide range of complexity, defined by three key factors: non-Gaussianity, number of modes, and degree of squeezing. We show that a single model pretrained on low-complexity states can be directly applied to characterize states of higher complexity. With limited fine-tuning, the model adapts to downstream tasks such as predicting quantum fidelity and Wigner negativity over a broad class of experimentally relevant states, including strongly non-Gaussian Schrödinger cat states, multimode systems with up to ten modes, and highly squeezed states with squeezing levels up to 10.4dB. Our results establish a unified framework for characterizing optical quantum states from limited measurement data, enabling efficient certification of quantum states relevant to optical quantum information computation, communication and metrology.

---

## 43. In-Context Audio Control of Video Diffusion Transformers

**论文链接:** [http://arxiv.org/abs/2512.18772v1](http://arxiv.org/abs/2512.18772v1)

**作者:** Wenze Liu, Weicai Ye, Minghong Cai, Quande Liu, Xintao Wang, Xiangyu Yue

**发布时间:** 2025-12-21

### GPT解析

### 总结

本文提出了ICAC框架，研究在视频生成中集成音频信号，通过掩码3D注意力机制实现稳定的训练和卓越的性能，达到良好的口型同步和视频质量。

### 背景

视频生成领域正转向基于Transformer的统一基础模型，但这些模型主要关注文本、图像和深度图等模态，而音频等严格时间同步信号研究不足。

### 目的

引入ICAC框架，研究在统一的完全注意力架构中集成音频信号，用于语音驱动的视频生成。

### 方法

系统探索了三种注入音频条件的机制：标准交叉注意力、2D自注意力和统一的3D自注意力；并提出掩码3D注意力机制来限制注意力模式，强制时间对齐。

### 主要发现

3D注意力在捕捉时空音频视觉相关性方面具有最高潜力，但存在显著的训练挑战；掩码3D注意力机制能够实现稳定训练和卓越性能。

### 结论

基于音频流和参考图像的条件，该方法实现了良好的口型同步和视频质量。

### 翻译

最近的视频生成进展已经转向统一的、基于Transformer的基础模型，这些模型可以在上下文中处理多种条件输入。然而，这些模型主要关注文本、图像和深度图等模态，而音频等严格时间同步信号研究不足。本文介绍了视频扩散变压器的上下文音频控制（ICAC），这是一个研究在统一的完全注意力架构中集成音频信号用于语音驱动视频生成的框架。我们系统探索了三种不同的音频条件注入机制：标准交叉注意力、2D自注意力和统一的3D自注意力。我们的研究揭示，虽然3D注意力在捕捉时空音频视觉相关性方面具有最高潜力，但它带来了显著的训练挑战。为克服这一问题，我们提出了掩码3D注意力机制，通过限制注意力模式来强制时间对齐，从而实现稳定训练和卓越性能。我们的实验证明，这种方法在音频流和参考图像的条件下实现了良好的口型同步和视频质量。


### 论文摘要

Recent advancements in video generation have seen a shift towards unified, transformer-based foundation models that can handle multiple conditional inputs in-context. However, these models have primarily focused on modalities like text, images, and depth maps, while strictly time-synchronous signals like audio have been underexplored. This paper introduces In-Context Audio Control of video diffusion transformers (ICAC), a framework that investigates the integration of audio signals for speech-driven video generation within a unified full-attention architecture, akin to FullDiT. We systematically explore three distinct mechanisms for injecting audio conditions: standard cross-attention, 2D self-attention, and unified 3D self-attention. Our findings reveal that while 3D attention offers the highest potential for capturing spatio-temporal audio-visual correlations, it presents significant training challenges. To overcome this, we propose a Masked 3D Attention mechanism that constrains the attention pattern to enforce temporal alignment, enabling stable training and superior performance. Our experiments demonstrate that this approach achieves strong lip synchronization and video quality, conditioned on an audio stream and reference images.

---

## 44. 论文ID: 2512.18745v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18745v1.json'

---

## 45. A Study of Finetuning Video Transformers for Multi-view Geometry Tasks

**论文链接:** [http://arxiv.org/abs/2512.18684v1](http://arxiv.org/abs/2512.18684v1)

**作者:** Huimin Wu, Kwang-Ting Cheng, Stephen Lin, Zhirong Wu

**发布时间:** 2025-12-21

**备注:** AAAI 20206, Project website: geovit-aaai26.github.io

### GPT解析

### 总结

本研究探讨了使用视觉Transformer学习多视图几何任务（如光流估计），通过微调视频基础模型。研究发现通用视频预训练模型可简单迁移到多视图问题，只需少量调整。通过在Transformer骨干上添加线性解码器和迭代细化，实现了最先进的性能，并在多种几何视觉任务中展示了强大的多功能性。

### 背景

先前的方法涉及自定义架构设计和特定任务预训练，而本研究采用了一种更简单的方法，利用通用视频预训练模型。这种新方法避免了复杂的定制设计，而是利用了模型在预训练过程中已经学习到的时间和空间信息。

### 目的

研究旨在验证通用视频预训练模型在多视图几何任务（如光流估计、3D深度估计和立体匹配）中的有效性，并探索通过简单调整实现最先进性能的可能性。

### 方法

通过微调视频基础模型，在Transformer骨干网络上添加线性解码器，并采用迭代细化技术来提升性能。这种方法利用了补丁之间通用注意力机制学习的时间和空间信息，用于几何推理。

### 主要发现

1. 通用视频预训练模型可以很容易地迁移到多视图问题，只需少量调整
2. 在Transformer骨干网络上添加线性解码器可以产生满意的结果
3. 迭代细化可以将性能提升到最先进的水平
4. 在光流估计任务上实现了跨数据集的最先进泛化结果
5. 在3D深度估计和立体匹配中也显示出强大性能

### 结论

视频预训练模型在解决几何视觉任务中具有多功能性和有效性。这种概念上简单的方法通过利用预训练模型中已经存在的时间和空间信息，避免了复杂的定制设计，实现了最先进的性能，为多视图几何任务提供了一种新的有效解决方案。

### 翻译

本文研究了通过微调视频基础模型，将视觉Transformer学习应用于多视图几何任务（如光流估计）。与涉及自定义架构设计和特定任务预训练的先前方法不同，我们的研究发现通用视频预训练模型可以很容易地迁移到多视图问题，只需少量调整。核心见解是补丁之间的通用注意力机制学习了用于几何推理的时间和空间信息。我们证明，在Transformer骨干网络上添加线性解码器可以产生满意的结果，而迭代细化可以将性能提升到最先进的水平。这种概念上简单的方法在光流估计上实现了跨数据集的最先进泛化结果，在Sintel clean、Sintel final和KITTI数据集上分别达到了0.69、1.78和3.15的端点误差(EPE)。我们的方法还在在线测试基准上创造了新记录，EPE值分别为0.79和1.88，F1值为3.79。在3D深度估计和立体匹配中的应用也显示出强大性能，说明了视频预训练模型在解决几何视觉任务中的多功能性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要研究如何通过微调视频基础模型来解决多视图几何任务（如光流估计、立体匹配和深度估计）。这个问题在研究中很重要，因为目前大多数几何任务需要复杂的手工设计和任务特定的预训练策略；在现实中，这些任务是计算机视觉的基础，广泛应用于自动驾驶、机器人导航、增强现实和三维重建等领域。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者发现视频transformer中的跨帧注意力机制能学习时空信息，可用于几何推理。他们首先尝试在预训练视频transformer上添加简单线性解码器，然后引入迭代细化机制进一步提高性能。作者借鉴了视频基础模型的预训练表示、RAFT的迭代细化思想，但创新性地用图像扭曲代替了成本体积查询。他们还调整了预训练3D ViT的位置编码，使其能处理两帧任务。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是视频预训练transformer通过自注意力学习到的特征包含丰富时空信息，可有效迁移到多视图几何任务。实现流程包括：1)调整预训练视频模型的位置编码使其适应两帧输入；2)添加线性解码器直接回归几何属性；3)可选的迭代细化机制，通过图像扭曲和残差预测逐步改进结果。整个过程无需复杂架构设计，只需微调预训练模型。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出简单通用框架，只需微调预训练视频transformer；2)创新性地调整位置编码适应两帧任务；3)设计基于图像扭曲而非成本体积的迭代细化机制；4)统一框架处理多种几何任务；5)展示强大的跨数据集泛化能力。相比之前工作，本文方法无需复杂架构设计和任务特定预训练，比CroCo等更通用，比RAFT等更适合全局上下文建模，比FlowFormer++等性能更优但预训练更简单。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文证明了通过简单微调通用视频预训练transformer模型，可以有效地解决多种多视图几何任务并取得最先进性能，为视频预训练模型在几何视觉任务中的应用开辟了新途径。'}


### 论文摘要

This paper presents an investigation of vision transformer learning for multi-view geometry tasks, such as optical flow estimation, by fine-tuning video foundation models. Unlike previous methods that involve custom architectural designs and task-specific pretraining, our research finds that general-purpose models pretrained on videos can be readily transferred to multi-view problems with minimal adaptation. The core insight is that general-purpose attention between patches learns temporal and spatial information for geometric reasoning. We demonstrate that appending a linear decoder to the Transformer backbone produces satisfactory results, and iterative refinement can further elevate performance to stateof-the-art levels. This conceptually simple approach achieves top cross-dataset generalization results for optical flow estimation with end-point error (EPE) of 0.69, 1.78, and 3.15 on the Sintel clean, Sintel final, and KITTI datasets, respectively. Our method additionally establishes a new record on the online test benchmark with EPE values of 0.79, 1.88, and F1 value of 3.79. Applications to 3D depth estimation and stereo matching also show strong performance, illustrating the versatility of video-pretrained models in addressing geometric vision tasks.

---

## 46. brat: Aligned Multi-View Embeddings for Brain MRI Analysis

**论文链接:** [http://arxiv.org/abs/2512.18679v1](http://arxiv.org/abs/2512.18679v1)

**作者:** Maxime Kayser, Maksim Gridnev, Wanting Wang, Max Bain, Aneesh Rangnekar, Avijit Chatterjee, Aleksandr Petrov, Harini Veeraraghavan, Nathaniel C. Swinburne

**发布时间:** 2025-12-21

**备注:** First round accept at WACV 2026

### GPT解析

### 总结

brat是一个针对脑部MRI的多视角表示学习框架，使用与临床报告配对的MRI数据进行训练

### 背景

脑部MRI面临独特挑战，因为存在大量、高度多样化且通常微妙的异常，这些异常往往仅分布在3D体积的少数切片中

### 目的

解决脑部MRI分析中的挑战，开发一个能够有效处理这些异常的多视角表示学习框架

### 方法

引入了一个比现有数据集大10倍的脑部MRI数据集(约80,000个3D扫描及其相应的放射学报告)，提出受文档检索启发的多视角预训练方法，开发隐式查询-特征匹配机制，采用质量多样性概念获得与临床特征对齐的MRI多视角嵌入

### 主要发现

在多个视觉语言和视觉任务上评估，显示出显著的性能提升

### 结论

brat基础模型已成功开发并公开发布，能够有效处理脑部MRI分析中的挑战

### 翻译

我们提出了brat(脑部报告对齐变换器)，这是一个针对脑部磁共振成像(MRI)的多视角表示学习框架，使用与临床报告配对的MRI数据进行训练。由于脑部MRI中存在大量、高度多样化且通常微妙的异常，这些异常往往仅分布在3D体积的少数切片中，因此脑部MRI面临独特挑战。为应对这些挑战，我们引入了一个比现有数据集大10倍的脑部MRI数据集，包含约80,000个3D扫描及其相应的放射学报告，并提出了一种受文档检索进展启发的多视角预训练方法。我们开发了一种隐式查询-特征匹配机制，并采用质量多样性概念，获得与报告句子提供的临床特征对齐的MRI多视角嵌入。我们在多个视觉语言和视觉任务上评估了我们的方法，显示出显著的性能提升。brat基础模型已公开发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决脑部MRI分析中的挑战：3D脑部MRI图像中存在众多、高度变化且通常微妙的异常，这些异常往往仅限于3D体积中的几个切片。这个问题很重要，因为脑部MRI是诊断脑部疾病的标准方法，但开发有效的AI模型面临数据稀缺和现有方法无法充分利用临床报告中丰富信息的问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者借鉴了文档检索中的多向量检索方法(如ColBERT)，将脑部MRI视为'文档'，报告句子视为'查询'。他们还参考了BLIP-2中的Q-Former架构，使用可学习的查询令牌作为潜在变量。结合这些思想，作者设计了brat框架，通过成对视图对齐(PVA)算法将多视图嵌入与临床特征对齐，并使用行列式点过程(DPPs)增强嵌入多样性。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用多视图嵌入表示脑部MRI中的多样化信息，将报告分解为句子嵌入，通过对比学习将这些子单元与MRI对齐。流程包括：1)处理MSKBrain数据集；2)从MRI和报告中提取特征；3)生成多视图嵌入；4)使用PVA算法对齐嵌入；5)通过DPPs优化多样性；6)计算对比损失和DPP损失进行训练；7)将模型应用于下游任务如报告生成和肿瘤分割。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)引入MSKBrain数据集(比现有数据集大10倍)；2)提出多视图嵌入方法，专门针对3D医学图像和长临床报告；3)开发成对视图对齐(PVA)算法；4)首次将行列式点过程(DPPs)应用于多视图表示学习。相比之前工作，brat专门处理3D医学图像和长报告，使用多视图而非单一嵌入，结合了对齐和多样性优化，并在大规模数据集上预训练并公开模型权重。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过引入大规模脑部MRI数据集、多视图表示学习框架和质量-多样性优化方法，显著提升了脑部MRI分析在图像-文本检索、报告生成和疾病分类等任务上的性能，为医学影像分析提供了新的基础模型。'}


### 论文摘要

We present brat (brain report alignment transformer), a multi-view representation learning framework for brain magnetic resonance imaging (MRI) trained on MRIs paired with clinical reports. Brain MRIs present unique challenges due to the presence of numerous, highly varied, and often subtle abnormalities that are localized to a few slices within a 3D volume. To address these challenges, we introduce a brain MRI dataset $10\times$ larger than existing ones, containing approximately 80,000 3D scans with corresponding radiology reports, and propose a multi-view pre-training approach inspired by advances in document retrieval. We develop an implicit query-feature matching mechanism and adopt concepts from quality-diversity to obtain multi-view embeddings of MRIs that are aligned with the clinical features given by report sentences. We evaluate our approach across multiple vision-language and vision tasks, demonstrating substantial performance improvements. The brat foundation models are publicly released.

---

## 47. 论文ID: 2512.18634v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18634v1.json'

---

## 48. Automated Mosaic Tesserae Segmentation via Deep Learning Techniques

**论文链接:** [http://arxiv.org/abs/2512.18406v1](http://arxiv.org/abs/2512.18406v1)

**作者:** Charilaos Kapelonis, Marios Antonakakis, Konstantinos Politof, Aristomenis Antoniadis, Michalis Zervakis

**发布时间:** 2025-12-20

**DOI:** 10.1109/IST66504.2025.11268445

### GPT解析

### 总结

该论文提出了一种基于SAM 2模型的马赛克图像分割方法，通过微调模型和创建专门的数据集，显著提高了马赛克图像中小块的分割准确度，为文化遗产的数字化保存提供了新方法。

### 背景

艺术被广泛视为文明的反映，马赛克作为文化遗产的重要组成部分，是由小块通过粘合剂在表面排列组成的古老艺术形式。由于其年代久远和易碎性，马赛克容易受损，因此需要进行数字化保存。

### 目的

解决马赛克数字化过程中的图像分割问题，即将马赛克中的小块从背景中分离出来，实现文化遗产的有效数字化保存。

### 方法

采用Meta AI的Segment Anything Model 2基础模型，并创建了一个专门标注的马赛克图像数据集用于微调和评估模型。

### 主要发现

与基线SAM 2模型相比，微调后的模型性能显著提升：交并比从89.00%提高到91.02%，召回率从92.12%提高到95.89%。在先前方法提出的基准测试上，模型的F-measure比先前方法高3%，预测与实际小块之间的绝对误差从0.20降低到0.02。

### 结论

微调后的SAM 2模型与新创建的标注数据集相结合，为马赛克图像的实时分割铺平了道路，对文化遗产的数字化保存具有重要意义。

### 翻译

艺术被广泛视为文明的反映，马赛克代表文化遗产的重要组成部分。马赛克是一种古老的艺术形式，通过使用粘合剂将称为小块的小块在表面上排列而创造。由于其年代久远和易碎性，它们容易受损，突显了数字化保存的必要性。本文通过在计算机视觉的图像分割领域中分割小块以将它们与背景分离，解决了马赛克数字化的问题。我们提出了一种利用Meta AI的Segment Anything Model 2的方法，这是一个基础模型，其性能优于大多数传统分割模型，可以自动分割马赛克。由于该领域开放数据集有限，我们还创建了一个标注的马赛克图像数据集，用于微调和评估模型。在我们的测试数据集上的定量评估显示，与基线SAM 2模型相比有显著改进，交并比从89.00%提高到91.02%，召回率从92.12%提高到95.89%。此外，在先前方法提出的基准测试上，我们的模型的F-measure比先前方法高3%，并将预测小块与实际小块之间的绝对误差从0.20降低到仅0.02。微调后的SAM 2模型的显著性能与新创建的标注数据集一起，可以为马赛克图像的实时分割铺平道路。


### 论文摘要

Art is widely recognized as a reflection of civilization and mosaics represent an important part of cultural heritage. Mosaics are an ancient art form created by arranging small pieces, called tesserae, on a surface using adhesive. Due to their age and fragility, they are prone to damage, highlighting the need for digital preservation. This paper addresses the problem of digitizing mosaics by segmenting the tesserae to separate them from the background within the broader field of Image Segmentation in Computer Vision. We propose a method leveraging Segment Anything Model 2 (SAM 2) by Meta AI, a foundation model that outperforms most conventional segmentation models, to automatically segment mosaics. Due to the limited open datasets in the field, we also create an annotated dataset of mosaic images to fine-tune and evaluate the model. Quantitative evaluation on our testing dataset shows notable improvements compared to the baseline SAM 2 model, with Intersection over Union increasing from 89.00% to 91.02% and Recall from 92.12% to 95.89%. Additionally, on a benchmark proposed by a prior approach, our model achieves an F-measure 3% higher than previous methods and reduces the error in the absolute difference between predicted and actual tesserae from 0.20 to just 0.02. The notable performance of the fine-tuned SAM 2 model together with the newly annotated dataset can pave the way for real-time segmentation of mosaic images.

---

## 49. Towards Guided Descent: Optimization Algorithms for Training Neural Networks At Scale

**论文链接:** [http://arxiv.org/abs/2512.18373v1](http://arxiv.org/abs/2512.18373v1)

**作者:** Ansh Nagwekar

**发布时间:** 2025-12-20

**备注:** Master's Thesis at the University of Pennsylvania

### GPT解析

### 总结

该论文探讨了神经网络优化在现代AI研究中的重要性，研究了从经典一阶方法到现代高阶技术的优化算法演变，展示了如何通过有原则的算法设计来阐明训练过程，并提供了将这些方法整合到现代深度学习工作流中的实用建议。

### 背景

神经网络优化是现代AI研究中最重要但理解最不充分的挑战之一。尽管随机梯度下降(SGD)及其变体已成为训练深度网络的标准方法，但在过度参数化情况下，它们的成功似乎更多是经验性的而非基于原则的。

### 目的

研究SGD等传统优化方法在训练深度网络时表现出的'经验性成功多于原则性'这一悖论，通过追踪优化算法的演变，揭示如何通过有原则的算法设计来阐明训练过程，并弥合理论理解与实际部署之间的差距。

### 方法

从基本原理开始分析SGD和自适应梯度方法，逐步揭示这些传统方法在面对真实世界数据各向异性时的局限性，进而探索基于曲率信息的复杂替代方法，如二阶近似技术、逐层预调节和自适应学习率等，并研究这些优化算法与神经网络训练工具包中其他元素的相互作用。

### 主要发现

传统优化方法在面对真实世界数据的各向异性时存在局限性；基于曲率信息的复杂替代方法(如二阶近似技术、逐层预调节、自适应学习率等)可能更有效；优化算法与神经网络训练工具包中其他元素(如最大更新参数化、学习率计划和指数移动平均等)的相互作用对实证成功同样重要。

### 结论

通过有原则的算法设计可以更好地理解和改进神经网络训练过程；将高级优化技术与训练工具包中的其他方法结合使用可以显著提高模型性能；提供实用的建议和实施策略有助于弥合理论理解与实际部署之间的差距。

### 翻译

神经网络优化仍然是现代AI研究中最重要但理解最不充分的挑战之一，其中训练算法的改进可以导致基础模型中特征学习的增强，训练时间的数量级减少，以及对网络学习方式的更好解释。虽然随机梯度下降(SGD)及其变体已成为训练深度网络的事实标准，但它们在这些过度参数化情况下的成功往往看起来更多是经验性的而非基于原则的。本文通过追踪优化算法从经典一阶方法到现代高阶技术的演变来研究这一明显悖论，展示如何通过有原则的算法设计来阐明训练过程。从SGD和自适应梯度方法的基本原理开始，分析逐步揭示了这些传统方法在面对代表真实世界数据的各向异性时的局限性。这些局限性促使探索基于曲率信息的复杂替代方法：二阶近似技术、逐层预调节、自适应学习率等。接下来，这些优化算法与更广泛的神经网络训练工具包之间的相互作用(包括最大更新参数化、学习率计划和指数移动平均等近期发展)被证明对实证成功同样重要。为了弥合理论理解与实际部署之间的差距，本文提供了将这些方法整合到现代深度学习工作流中的实用建议和实施策略。


### 论文摘要

Neural network optimization remains one of the most consequential yet poorly understood challenges in modern AI research, where improvements in training algorithms can lead to enhanced feature learning in foundation models, order-of-magnitude reductions in training time, and improved interpretability into how networks learn. While stochastic gradient descent (SGD) and its variants have become the de facto standard for training deep networks, their success in these over-parameterized regimes often appears more empirical than principled. This thesis investigates this apparent paradox by tracing the evolution of optimization algorithms from classical first-order methods to modern higher-order techniques, revealing how principled algorithmic design can demystify the training process. Starting from first principles with SGD and adaptive gradient methods, the analysis progressively uncovers the limitations of these conventional approaches when confronted with anisotropy that is representative of real-world data. These breakdowns motivate the exploration of sophisticated alternatives rooted in curvature information: second-order approximation techniques, layer-wise preconditioning, adaptive learning rates, and more. Next, the interplay between these optimization algorithms and the broader neural network training toolkit, which includes prior and recent developments such as maximal update parametrization, learning rate schedules, and exponential moving averages, emerges as equally essential to empirical success. To bridge the gap between theoretical understanding and practical deployment, this paper offers practical prescriptions and implementation strategies for integrating these methods into modern deep learning workflows.

---

## 50. TICL+: A Case Study On Speech In-Context Learning for Children's Speech Recognition

**论文链接:** [http://arxiv.org/abs/2512.18263v1](http://arxiv.org/abs/2512.18263v1)

**作者:** Haolong Zheng, Yekaterina Yegorova, Mark Hasegawa-Johnson

**发布时间:** 2025-12-20

**备注:** Published at IEEE ASRU 2025 Satellite Workshop-AI for Children's Speech and Language

### GPT解析

### 总结

本研究提出了一种改进的儿童语音识别方法TICL+，通过结合语义和声学信息，显著提高了儿童语音识别的准确性。

### 背景

儿童语音识别面临显著挑战，包括声学和语言变异性大、标记数据有限，以及与成人语音存在显著差异。

### 目的

提高语音基础模型在上下文学习中的效果，特别是优化上下文示例的选择方式，以提升儿童语音识别性能。

### 方法

扩展了现有的基于检索的TICL方法，引入声学重排序步骤创建TICL+，优先选择在语义和声学上都与测试输入对齐的示例。

### 主要发现

在四个儿童语音语料库实验中，TICL+相比零样本性能实现了高达53.3%的相对词错误率降低，相比基线TICL实现了37.6%的相对词错误率降低。

### 结论

结合语义和声学信息对于开发鲁棒、可扩展的儿童语音自动识别系统具有重要价值。

### 翻译

儿童语音识别由于显著的声学和语言变异性、有限的标记数据以及与成人语音的显著差异而仍然具有挑战性。语音基础模型可以通过语音上下文学习（SICL）解决这些挑战，允许无需微调即可适应新领域。然而，SICL的有效性取决于如何选择上下文示例。我们扩展了一个现有的基于检索的方法，用于SICL的文本嵌入KNN（TICL），引入了一个声学重排序步骤来创建TICL+。这种扩展优先选择在语义和声学上都与测试输入对齐的示例。在四个儿童语音语料库上的实验表明，TICL+相比零样本性能实现了高达53.3%的相对词错误率降低，相比基线TICL实现了37.6%的降低，突显了结合语义和声学信息对于儿童语音鲁棒、可扩展ASR的价值。


### 论文摘要

Children's speech recognition remains challenging due to substantial acoustic and linguistic variability, limited labeled data, and significant differences from adult speech. Speech foundation models can address these challenges through Speech In-Context Learning (SICL), allowing adaptation to new domains without fine-tuning. However, the effectiveness of SICL depends on how in-context examples are selected. We extend an existing retrieval-based method, Text-Embedding KNN for SICL (TICL), introducing an acoustic reranking step to create TICL+. This extension prioritizes examples that are both semantically and acoustically aligned with the test input. Experiments on four children's speech corpora show that TICL+ achieves up to a 53.3% relative word error rate reduction over zero-shot performance and 37.6% over baseline TICL, highlighting the value of combining semantic and acoustic information for robust, scalable ASR in children's speech.

---

## 51. 论文ID: 2512.18176v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18176v1.json'

---

## 52. SAM Audio: Segment Anything in Audio

**论文链接:** [http://arxiv.org/abs/2512.18099v1](http://arxiv.org/abs/2512.18099v1)

**作者:** Bowen Shi, Andros Tjandra, John Hoffman, Helin Wang, Yi-Chiao Wu, Luya Gao, Julius Richter, Matt Le, Apoorv Vyas, Sanyuan Chen, Christoph Feichtenhofer, Piotr Dollár, Wei-Ning Hsu, Ann Lee

**发布时间:** 2025-12-19

### GPT解析

### 总结

SAM Audio是一个基础模型，用于通用音频分离，统一了文本、视觉和时间跨度提示，在多种音频分离任务中取得最先进性能。

### 背景

通用音频源分离是能够感知和推理声音的多模态AI系统的关键能力。尽管近年来取得进展，但现有分离模型要么是领域特定的（针对语音或音乐等固定类别），要么可控性有限（仅支持文本提示等单一提示模态）。

### 目的

提出一个名为SAM Audio的基础模型，用于通用音频分离，该模型在单一框架内统一了文本、视觉和时间跨度提示。

### 方法

SAM Audio基于扩散Transformer架构，使用流匹配技术在大规模音频数据（涵盖语音、音乐和一般声音）上进行训练，可以灵活地分离由语言、视觉掩码或时间跨度描述的目标源。

### 主要发现

该模型在多样化的基准测试中取得了最先进的性能，包括在自然和专业制作的音频中进行一般声音、语音、音乐和乐器分离，显著优于之前的通用和专业系统。此外，研究者还引入了一个具有人工标注多模态提示的新现实世界分离基准，以及一个与人类判断高度相关的无参考评估模型。

### 结论

SAM Audio通过统一多种提示模态，实现了更灵活和强大的音频分离能力，在多个领域都表现出色，为多模态AI系统提供了重要的音频感知能力。

### 翻译

通用音频源分离是能够感知和推理声音的多模态AI系统的关键能力。尽管近年来取得了重大进展，但现有的分离模型要么是领域特定的，设计用于语音或音乐等固定类别，要么可控性有限，仅支持文本等单一提示模态。在这项工作中，我们提出了SAM Audio，这是一个用于通用音频分离的基础模型，在单一框架内统一了文本、视觉和时间跨度提示。SAM Audio基于扩散Transformer架构，使用流匹配技术在涵盖语音、音乐和一般声音的大规模音频数据上进行训练，可以灵活地分离由语言、视觉掩码或时间跨度描述的目标源。该模型在多样化的基准测试中取得了最先进的性能，包括在自然和专业制作的音频中进行一般声音、语音、音乐和乐器分离，显著优于之前的通用和专业系统。此外，我们还引入了一个具有人工标注多模态提示的新现实世界分离基准，以及一个与人类判断高度相关的无参考评估模型。


### 论文摘要

General audio source separation is a key capability for multimodal AI systems that can perceive and reason about sound. Despite substantial progress in recent years, existing separation models are either domain-specific, designed for fixed categories such as speech or music, or limited in controllability, supporting only a single prompting modality such as text. In this work, we present SAM Audio, a foundation model for general audio separation that unifies text, visual, and temporal span prompting within a single framework. Built on a diffusion transformer architecture, SAM Audio is trained with flow matching on large-scale audio data spanning speech, music, and general sounds, and can flexibly separate target sources described by language, visual masks, or temporal spans. The model achieves state-of-the-art performance across a diverse suite of benchmarks, including general sound, speech, music, and musical instrument separation in both in-the-wild and professionally produced audios, substantially outperforming prior general-purpose and specialized systems. Furthermore, we introduce a new real-world separation benchmark with human-labeled multimodal prompts and a reference-free evaluation model that correlates strongly with human judgment.

---

## 53. FPBench: A Comprehensive Benchmark of Multimodal Large Language Models for Fingerprint Analysis

**论文链接:** [http://arxiv.org/abs/2512.18073v1](http://arxiv.org/abs/2512.18073v1)

**作者:** Ekta Balkrishna Gavas, Sudipta Banerjee, Chinmay Hegde, Nasir Memon

**发布时间:** 2025-12-19

### GPT解析

### 总结

该研究建立了首个用于评估多模态大语言模型在指纹理解领域能力的全面基准测试FPBench。

### 背景

多模态大语言模型已在复杂数据分析、视觉问答、生成和推理方面取得显著进展，并被用于分析虹膜和人脸图像的生物测量效用，但在指纹理解方面的能力尚未被探索。

### 目的

设计一个全面的基准测试FPBench，用于评估多模态大语言模型在指纹理解方面的性能。

### 方法

创建了名为FPBench的基准测试，评估了20个开源和专有的多模态大语言模型，在7个真实和合成数据集上执行8个生物识别和法医任务，采用零样本和思维链提示策略。

### 主要发现

从性能和可解释性角度讨论了研究发现，并分享了关于挑战和局限性的见解。

### 结论

FPBench作为首个用于指纹领域理解的多模态大语言模型全面基准，为指纹基础模型的发展铺平了道路。

### 翻译

多模态大语言模型在复杂数据分析、视觉问答、生成和推理方面已获得显著关注。最近，它们被用于分析虹膜和人脸图像的生物测量效用。然而，它们在指纹理解方面的能力尚未被探索。在这项工作中，我们设计了一个名为FPBench的全面基准，该基准使用零样本和思维链提示策略，在7个真实和合成数据集上，通过8个生物识别和法医任务，评估了20个多模态大语言模型（开源和专有）的性能。我们从性能和可解释性角度讨论了研究结果，并分享了关于挑战和局限性的见解。我们建立了FPBench作为首个用于指纹领域理解的多模态大语言模型全面基准，为指纹基础模型铺平了道路。


### 论文摘要

Multimodal LLMs (MLLMs) have gained significant traction in complex data analysis, visual question answering, generation, and reasoning. Recently, they have been used for analyzing the biometric utility of iris and face images. However, their capabilities in fingerprint understanding are yet unexplored. In this work, we design a comprehensive benchmark, \textsc{FPBench} that evaluates the performance of 20 MLLMs (open-source and proprietary) across 7 real and synthetic datasets on 8 biometric and forensic tasks using zero-shot and chain-of-thought prompting strategies. We discuss our findings in terms of performance, explainability and share our insights into the challenges and limitations. We establish \textsc{FPBench} as the first comprehensive benchmark for fingerprint domain understanding with MLLMs paving the path for foundation models for fingerprints.

---

## 54. A Dataset and Benchmarks for Atrial Fibrillation Detection from Electrocardiograms of Intensive Care Unit Patients

**论文链接:** [http://arxiv.org/abs/2512.18031v1](http://arxiv.org/abs/2512.18031v1)

**作者:** Sarah Nassar, Nooshin Maghsoodi, Sophia Mannina, Shamel Addas, Stephanie Sibley, Gabor Fichtinger, David Pichora, David Maslove, Purang Abolmaesumi, Parvin Mousavi

**发布时间:** 2025-12-19

**备注:** 10 pages, 3 figures, 6 tables

### GPT解析

### 总结

本研究比较了三种AI方法在心房颤动检测中的表现，结果表明ECG基础模型表现最佳，为ICU患者心房颤动的自动监测提供了新思路。

### 背景

心房颤动是重症监护室患者最常见的心律失常，可能导致不良健康影响。目前缺乏针对ICU患者心房颤动检测的专门数据集和性能基准。

### 目的

发布一个标记的ICU数据集和心房颤动检测的基准，比较机器学习模型在三种基于数据的人工智能方法中的表现，确定最适合心房颤动检测的AI方法。

### 方法

比较三种AI方法：基于特征的分类器、深度学习和心电图基础模型。使用加拿大ICU的心电图和2021年PhysioNet/计算心脏病学挑战赛的数据进行实验，测试多种训练配置，从零样本推理到迁移学习。

### 主要发现

平均而言，在两个数据集上，ECG基础模型表现最好，其次是深度学习，最后是基于特征的分类器。通过迁移学习策略的ECG-FM模型在ICU测试集上获得了最高的F1分数(0.89)。

### 结论

这项研究展示了使用人工智能构建自动患者监测系统的巨大潜力，通过发布标记的ICU数据集和性能基准，为研究社区继续推进IC中心房颤动检测的最新技术提供了基础。

### 翻译

目的：心房颤动是重症监护室患者最常见的心律失常，可能导致不良健康影响。在本研究中，我们发布了一个标记的ICU数据集和心房颤动检测的基准。方法：我们比较了三种基于数据的人工智能方法的机器学习模型：基于特征的分类器、深度学习和心电图基础模型。这种比较解决了文献中的一个关键空白，旨在确定哪种AI方法最适合心房颤动检测。使用加拿大ICU的心电图和2021年PhysioNet/计算心脏病学挑战赛的数据进行实验。测试了多种训练配置，从零样本推理到迁移学习。结果：平均而言，在两个数据集上，ECG基础模型表现最好，其次是深度学习，然后是基于特征的分类器。在我们的ICU测试集上获得最高F1分数的模型是通过迁移学习策略的ECG-FM(F1=0.89)。结论：这项研究展示了使用人工智能构建自动患者监测系统的巨大潜力。意义：通过发布我们的标记ICU数据集(链接待添加)和性能基准，这项工作使研究社区能够继续推进IC中心房颤动检测的最新技术。


### 论文摘要

Objective: Atrial fibrillation (AF) is the most common cardiac arrhythmia experienced by intensive care unit (ICU) patients and can cause adverse health effects. In this study, we publish a labelled ICU dataset and benchmarks for AF detection. Methods: We compared machine learning models across three data-driven artificial intelligence (AI) approaches: feature-based classifiers, deep learning (DL), and ECG foundation models (FMs). This comparison addresses a critical gap in the literature and aims to pinpoint which AI approach is best for accurate AF detection. Electrocardiograms (ECGs) from a Canadian ICU and the 2021 PhysioNet/Computing in Cardiology Challenge were used to conduct the experiments. Multiple training configurations were tested, ranging from zero-shot inference to transfer learning. Results: On average and across both datasets, ECG FMs performed best, followed by DL, then feature-based classifiers. The model that achieved the top F1 score on our ICU test set was ECG-FM through a transfer learning strategy (F1=0.89). Conclusion: This study demonstrates promising potential for using AI to build an automatic patient monitoring system. Significance: By publishing our labelled ICU dataset (LinkToBeAdded) and performance benchmarks, this work enables the research community to continue advancing the state-of-the-art in AF detection in the ICU.

---

## 55. 论文ID: 2512.17992v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.17992v1.json'

---

## 56. 论文ID: 2512.19246v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19246v1.json'

---

## 57. 论文ID: 2512.19037v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19037v1.json'

---

## 58. 论文ID: 2512.18661v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18661v1.json'

---

## 59. Few-Shot Learning of a Graph-Based Neural Network Model Without Backpropagation

**论文链接:** [http://arxiv.org/abs/2512.18412v1](http://arxiv.org/abs/2512.18412v1)

**作者:** Mykyta Lapin, Kostiantyn Bokhan, Yurii Parzhyn

**发布时间:** 2025-12-20

**备注:** 9 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种基于结构图的方法，用于在小样本情况下分类轮廓图像，无需使用反向传播。核心思想是将结构作为解释的载体：图像被编码为属性图，关键点和线作为带有几何属性的节点，通过形成概念吸引子实现泛化。

### 背景

在小样本学习领域，需要一种能够从少量样本中学习并做出透明决策的方法，同时避免使用反向传播。

### 目的

设计和实验验证一种架构，其中类概念通过结构和参数缩减从少量示例（每类5-6个）形成，提供透明的决策并消除反向传播的需要。

### 方法

轮廓矢量化后构建二部图（点/线作为节点），具有标准化的几何属性，如坐标、长度、角度和方向；缩减包括消除不稳定子结构或噪声以及对齐关键点之间的路径。概念通过样本的迭代组合形成，分类通过选择最佳图到概念匹配（使用近似GED）执行。

### 主要发现

在MNIST子集上（每类5-6个基础示例，单轮训练），获得约82%的一致准确率，且决策具有完全可追溯性：误分类可以通过明确的结构相似性来解释。提供了与SVM、MLP、CNN以及度量和元学习基线的比较。

### 结论

带有概念吸引子的结构图方案能够在没有反向传播的情况下实现小样本学习，并通过显式图结构提供内置解释。

### 翻译

我们提出了一种结构图方法，用于在小样本情况下分类轮廓图像，无需使用反向传播。核心思想是使结构成为解释的载体：图像被编码为属性图（关键点和线表示为带有几何属性的节点），并通过形成概念吸引子（类级概念图）来实现泛化。目的：设计和实验验证一种架构，其中类概念通过结构和参数缩减从少量示例（每类5-6个）形成，提供透明的决策并消除反向传播。方法：轮廓矢量化后构建二部图（点/线作为节点），具有标准化的几何属性，如坐标、长度、角度和方向；缩减包括消除不稳定子结构或噪声以及对齐关键点之间的路径。概念通过样本的迭代组合形成，分类通过选择最佳图到概念匹配（使用近似GED）执行。结果：在MNIST子集上（每类5-6个基础示例，单轮训练），获得约82%的一致准确率，且决策具有完全可追溯性：误分类可以通过明确的结构相似性来解释。提供了与SVM、MLP、CNN以及度量和元学习基线的比较。带有概念吸引子的结构图方案能够在没有反向传播的情况下实现小样本学习，并通过显式图结构提供内置解释。局限性涉及GED的计算成本和骨架化的质量；有前途的方向包括分类算法优化、静态场景工作和联想识别。


### 论文摘要

We propose a structural-graph approach to classifying contour images in a few-shot regime without using backpropagation. The core idea is to make structure the carrier of explanations: an image is encoded as an attributed graph (critical points and lines represented as nodes with geometric attributes), and generalization is achieved via the formation of concept attractors (class-level concept graphs). Purpose. To design and experimentally validate an architecture in which class concepts are formed from a handful of examples (5 - 6 per class) through structural and parametric reductions, providing transparent decisions and eliminating backpropagation. Methods. Contour vectorization is followed by constructing a bipartite graph (Point/Line as nodes) with normalized geometric attributes such as coordinates, length, angle, and direction; reductions include the elimination of unstable substructures or noise and the alignment of paths between critical points. Concepts are formed by iterative composition of samples, and classification is performed by selecting the best graph-to-concept match (using approximated GED). Results. On an MNIST subset with 5 - 6 base examples per class (single epoch), we obtain a consistent accuracy of around 82% with full traceability of decisions: misclassifications can be explained by explicit structural similarities. An indicative comparison with SVM, MLP, CNN, as well as metric and meta-learning baselines, is provided. The structural-graph scheme with concept attractors enables few-shot learning without backpropagation and offers built-in explanations through the explicit graph structure. Limitations concern the computational cost of GED and the quality of skeletonization; promising directions include classification-algorithm optimization, work with static scenes, and associative recognition.

---

## 60. Convolutional-neural-operator-based transfer learning for solving PDEs

**论文链接:** [http://arxiv.org/abs/2512.17969v1](http://arxiv.org/abs/2512.17969v1)

**作者:** Peng Fan, Guofei Pang

**发布时间:** 2025-12-19

**备注:** 12 pages, 4 figures, 2 tables

### GPT解析

### 总结

本文提出了一种将卷积神经算子扩展到少样本学习场景的方法，通过预训练和参数调整策略，发现神经元线性变换在求解PDE时具有最高的代理精度。

### 背景

卷积神经算子是一种基于CNN的架构，用于强制保持连续-离散等价性，并能够真实、无混叠地学习偏微分方程的解算子。该算子在某些情况下超越了DeepONet、傅里叶神经算子和Galerkin变换器等基线模型，但在少样本学习方面尚未得到验证。

### 目的

将卷积神经算子扩展到少样本学习场景，通过先在源数据集上预训练，然后仅使用小的目标数据集调整已训练神经算子的参数。

### 方法

研究三种调整已训练神经算子参数的策略：微调、低秩适应和神经元线性变换，并在Kuramoto-Sivashinsky方程、Brusselator扩散反应系统和Navier-Stokes方程等PDE上进行测试。

### 主要发现

在求解PDE时，神经元线性变换策略具有最高的代理精度，优于微调和低秩适应策略。

### 结论

神经元线性变换策略在少样本学习场景下表现最佳，能够有效提高卷积神经算子在求解PDE时的代理精度。

### 翻译

卷积神经算子是一种基于CNN的架构，最近被提出用于强制保持连续-离散等价性，并能够真实、无混叠地学习偏微分方程的解算子。该神经算子在某些情况下被证明在代理精度方面优于一些基线模型，如DeepONet、傅里叶神经算子和Galerkin变换器。然而，卷积神经算子似乎尚未在少样本学习中得到验证。我们通过首先使用源数据集预训练卷积神经算子，然后仅使用小的目标数据集调整已训练神经算子的参数，将模型扩展到少样本学习场景。我们研究了三种调整已训练神经算子参数的策略，包括微调、低秩适应和神经元线性变换，并发现神经元线性变换策略在求解Kuramoto-Sivashinsky方程、Brusselator扩散反应系统和Navier-Stokes方程等PDE时具有最高的代理精度。


### 论文摘要

Convolutional neural operator is a CNN-based architecture recently proposed to enforce structure-preserving continuous-discrete equivalence and enable the genuine, alias-free learning of solution operators of PDEs. This neural operator was demonstrated to outperform for certain cases some baseline models such as DeepONet, Fourier neural operator, and Galerkin transformer in terms of surrogate accuracy. The convolutional neural operator, however, seems not to be validated for few-shot learning. We extend the model to few-shot learning scenarios by first pre-training a convolutional neural operator using a source dataset and then adjusting the parameters of the trained neural operator using only a small target dataset. We investigate three strategies for adjusting the parameters of a trained neural operator, including fine-tuning, low-rank adaption, and neuron linear transformation, and find that the neuron linear transformation strategy enjoys the highest surrogate accuracy in solving PDEs such as Kuramoto-Sivashinsky equation, Brusselator diffusion-reaction system, and Navier-Stokes equations.

---

## 61. Efficient Beamforming Optimization for STAR-RIS-Assisted Communications: A Gradient-Based Meta Learning Approach

**论文链接:** [http://arxiv.org/abs/2512.17928v1](http://arxiv.org/abs/2512.17928v1)

**作者:** Dongdong Yang, Bin Li, Jiguang He, Yicheng Yan, Xiaoyu Zhang, Chongwen Huang

**发布时间:** 2025-12-09

### GPT解析

### 总结

本文提出了一种基于梯度的元学习(GML)框架，用于解决STAR-RIS辅助通信系统中的联合优化问题。该框架通过将优化梯度直接输入轻量级神经网络，避免了预训练需求，实现了快速适应。针对独立相位和耦合相位STAR-RIS模型设计了专门的GML方案，在保持接近基准AO方法性能的同时，显著降低了计算复杂度，提高了可扩展性。

### 背景

STAR-RIS作为一种新兴技术，有望在下一代无线网络中实现全空间覆盖并提高频谱效率。然而，基站预编码矩阵与STAR-RIS传输和反射系数矩阵的联合设计导致了一个高维度、强非凸且NP难的优化问题。传统的交替优化(AO)方案通常涉及重复的大规模矩阵求逆运算，计算复杂度高且可扩展性差，而现有的深度学习方法往往依赖昂贵的预训练和大型网络模型。

### 目的

解决STAR-RIS辅助通信系统中的联合优化问题，降低计算复杂度，提高方法的可扩展性，同时保持接近基准AO方法的性能，避免现有方法中预训练和大型网络模型的依赖。

### 方法

开发了一种基于梯度的元学习(GML)框架，直接将优化梯度输入轻量级神经网络，去除预训练需求并实现快速适应。针对独立相位和耦合相位STAR-RIS模型设计了专门的GML方案，有效处理各自的幅度和相位约束，实现接近AO基准的加权和速率性能。

### 主要发现

1. 提出的GML方法在两种相位模型下都显著降低了计算开销；2. 计算复杂度随基站天线数和STAR-RIS单元数的增长呈近似线性增长；3. 相比AO方法实现了高达10倍的运行时间加速；4. 在保持接近AO方法性能的同时，验证了所提GML方法在大规模STAR-RIS辅助通信中的可扩展性和实用性。

### 结论

基于梯度的元学习框架为STAR-RIS辅助通信系统提供了一种高效、可扩展的解决方案，能够有效处理复杂的联合优化问题，同时保持接近最优方法的性能，为大规模STAR-RIS技术的实际应用提供了可行途径。

### 翻译

同时传输和反射可重构智能表面(STAR-RIS)已成为一项有前景的技术，可在下一代无线网络中实现全空间覆盖并提高频谱效率。然而，基站预编码矩阵以及STAR-RIS传输和反射系数矩阵的联合设计导致了一个高维度、强非凸且NP难的优化问题。传统的交替优化(AO)方案通常涉及重复的大规模矩阵求逆运算，导致高计算复杂性和 poor 可扩展性，而现有的深度学习方法往往依赖昂贵的预训练和大型网络模型。在本文中，我们开发了一种基于梯度的元学习(GML)框架，直接将优化梯度输入轻量级神经网络，从而无需预训练并实现快速适应。具体而言，我们为独立相位和耦合相位STAR-RIS模型设计了专门的GML方案，有效处理各自的幅度和相位约束，同时实现非常接近基于AO的基准的加权和速率性能。大量仿真表明，对于两种相位模型，所提出的方法显著降低了计算开销，当基站天线和STAR-RIS单元数量增加时，复杂度呈近似线性增长，相比AO方法实现了高达10倍的运行时间加速，这证实了所提GML方法在大规模STAR-RIS辅助通信中的可扩展性和实用性。


### 论文摘要

Simultaneously transmitting and reflecting reconfigurable intelligent surface (STAR-RIS) has emerged as a promising technology to realize full-space coverage and boost spectral efficiency in next-generation wireless networks. Yet, the joint design of the base station precoding matrix as well as the STAR-RIS transmission and reflection coefficient matrices leads to a high-dimensional, strongly nonconvex, and NP-hard optimization problem. Conventional alternating optimization (AO) schemes typically involve repeated large-scale matrix inversion operations, resulting in high computational complexity and poor scalability, while existing deep learning approaches often rely on expensive pre-training and large network models. In this paper, we develop a gradient-based meta learning (GML) framework that directly feeds optimization gradients into lightweight neural networks, thereby removing the need for pre-training and enabling fast adaptation. Specifically, we design dedicated GML-based schemes for both independent-phase and coupled-phase STAR-RIS models, effectively handling their respective amplitude and phase constraints while achieving weighted sum-rate performance very close to that of AO-based benchmarks. Extensive simulations demonstrate that, for both phase models, the proposed methods substantially reduce computational overhead, with complexity growing nearly linearly when the number of BS antennas and STAR-RIS elements grows, and yielding up to 10 times runtime speedup over AO, which confirms the scalability and practicality of the proposed GML method for large-scale STAR-RIS-assisted communications.

---

## 62. ShibuyaSocial: Multi-scale Model of Pedestrian Flows in Scramble Crossing

**论文链接:** [http://arxiv.org/abs/2512.18550v1](http://arxiv.org/abs/2512.18550v1)

**作者:** Akihiro Sakurai, Naoya Kajio, Ko Yamamoto

**发布时间:** 2025-12-21

### GPT解析

### 总结

这篇论文提出了一种基于学习的行人流模型，整合了多尺度行为（如全局路线选择和局部碰撞避免），特别关注涩谷十字路口的行人移动。

### 背景

行人流过度拥挤可能导致严重事故，因此数学建模和预测行人行为对于预防事故和提供安全舒适的环境至关重要。

### 目的

开发一个能够同时考虑全局路线选择和局部碰撞避免的行人行为模型，以更准确地预测行人在城市空间中的行为。

### 方法

提出一个整合局部行为和全局路线选择的模型，使用注意力机制确保全局和局部行为预测的一致性。使用涩谷十字路口的行人行走轨迹数据训练该模型。

### 主要发现

通过基于训练模型的行人行为模拟，定性和定量验证了所提出的模型能够适当预测行人行为。

### 结论

所提出的模型成功整合了行人的多尺度行为，能够更准确地预测行人在复杂环境中的行为，有助于提高行人环境的安全性。

### 翻译

本文提出了一种基于学习的行人流模型，整合了城市空间中的多尺度行为，如全局路线选择和局部碰撞避免，特别关注涩谷十字路口的行人移动。由于行人流过度拥挤可能导致严重事故，数学建模和预测行人行为对于预防此类事故和提供安全舒适的环境非常重要。尽管许多研究已经调查了基于学习的建模方法，但它们大多只关注行人的局部行为，如与邻居和环境物体的碰撞避免。在实际环境中，行人行为涉及更复杂的决策，包括全局路线选择。此外，交通灯处从停止到行走的转变状态应同时考虑。在本研究中，所提出的模型使用注意力机制整合局部行为和全局路线选择，以确保全局和局部行为预测的一致性。我们录制了涩谷十字路口行人的视频数据，并使用从视频中获得的行人行走轨迹数据训练了所提出的模型。基于训练模型的行人行为模拟定性和定量地验证了所提出的模型能够适当预测行人行为。


### 论文摘要

This paper presents a learning-based model of pedestrian flows that integrates multi scale behaviors such as global route selection and local collision avoidance in urban spaces, particularly focusing on pedestrian movements at Shibuya scramble crossing. Since too much congestion of pedestrian flows can cause serious accidents, mathematically modeling and predicting pedestrian behaviors is important for preventing such accidents and providing a safe and comfortable environment. Although numerous studies have investigated learning-based modeling methods, most of them focus only on the local behavior of pedestrians, such as collision avoidance with neighbors and environmental objects. In an actual environment, pedestrian behavior involves more complicated decision making including global route selection. Moreover, a state transition from stopping to walking at a traffic light should be considered simultaneously. In this study, the proposed model integrates local behaviors with global route selection, using an Attention mechanism to ensure consistent global and local behavior predictions. We recorded video data of pedestrians at Shibuya scramble crossing and trained the proposed model using pedestrian walking trajectory data obtained from the video. Simulations of pedestrian behaviors based on the trained model qualitatively and quantitatively validated that the proposed model can appropriately predict pedestrian behaviors.

---

## 63. LLaViDA: A Large Language Vision Driving Assistant for Explicit Reasoning and Enhanced Trajectory Planning

**论文链接:** [http://arxiv.org/abs/2512.18211v1](http://arxiv.org/abs/2512.18211v1)

**作者:** Yudong Liu, Spencer Hallyburton, Jiwoo Kim, Yueqian Lin, Yiming Li, Qinsi Wang, Hui Ye, Jingwei Sun, Miroslav Pajic, Yiran Chen, Hai Li

**发布时间:** 2025-12-20

### GPT解析

### 总结

LLaViDA是一种基于视觉语言模型的自动驾驶轨迹规划方法，通过两阶段训练流程实现了对复杂场景的更好理解和更准确的轨迹规划，在基准测试中表现优异。

### 背景

轨迹规划是自动驾驶中的一个基础但具有挑战性的组成部分。端到端规划器在恶劣天气、不可预测的人类行为或复杂的道路布局下经常失败，主要是因为它们缺乏在训练数据之外的强泛化能力或少样本学习能力。

### 目的

提出一种能够更好地处理各种驾驶场景的轨迹规划方法，提高系统在复杂环境下的鲁棒性和泛化能力。

### 方法

提出了LLaViDA（大型语言视觉驾驶助手），利用视觉语言模型（VLM）进行目标运动预测、语义定位和思维链推理。采用两阶段训练流程：监督微调后接轨迹偏好优化（TPO），通过注入基于回归的监督增强场景理解和轨迹规划。

### 主要发现

在NuScenes基准测试中，LLaViDA在开环轨迹规划任务上超越了最先进的端到端和其他最新的VLM/LLM基线，在NuScenes测试集上实现了平均L2轨迹误差为0.31米，碰撞率为0.10%。

### 结论

LLaViDA是一种强大的'VLM自动驾驶轨迹规划器'，能够有效处理各种复杂的驾驶场景，显著提高了自动驾驶系统的安全性和可靠性。

### 翻译

轨迹规划是自动驾驶中的一个基础但具有挑战性的组成部分。端到端规划器在恶劣天气、不可预测的人类行为或复杂的道路布局下经常失败，主要是因为它们缺乏在训练数据之外的强泛化能力或少样本学习能力。我们提出了LLaViDA，这是一种大型语言视觉驾驶助手，它利用视觉语言模型（VLM）进行目标运动预测、语义定位和思维链推理，用于自动驾驶的轨迹规划。一个两阶段训练流程——监督微调后接轨迹偏好优化（TPO）——通过注入基于回归的监督，增强了场景理解和轨迹规划，产生了一个强大的'VLM自动驾驶轨迹规划器'。在NuScenes基准测试中，LLaViDA在开环轨迹规划任务上超越了最先进的端到端和其他最新的VLM/LLM基线，在NuScenes测试集上实现了平均L2轨迹误差为0.31米，碰撞率为0.10%。本文的代码可在GitHub上获取。


### 论文摘要

Trajectory planning is a fundamental yet challenging component of autonomous driving. End-to-end planners frequently falter under adverse weather, unpredictable human behavior, or complex road layouts, primarily because they lack strong generalization or few-shot capabilities beyond their training data. We propose LLaViDA, a Large Language Vision Driving Assistant that leverages a Vision-Language Model (VLM) for object motion prediction, semantic grounding, and chain-of-thought reasoning for trajectory planning in autonomous driving. A two-stage training pipeline--supervised fine-tuning followed by Trajectory Preference Optimization (TPO)--enhances scene understanding and trajectory planning by injecting regression-based supervision, produces a powerful "VLM Trajectory Planner for Autonomous Driving." On the NuScenes benchmark, LLaViDA surpasses state-of-the-art end-to-end and other recent VLM/LLM-based baselines in open-loop trajectory planning task, achieving an average L2 trajectory error of 0.31 m and a collision rate of 0.10% on the NuScenes test set. The code for this paper is available at GitHub.

---

## 64. 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation

**论文链接:** [http://arxiv.org/abs/2512.17012v2](http://arxiv.org/abs/2512.17012v2)

**作者:** Chiao-An Yang, Ryo Hachiuma, Sifei Liu, Subhashree Radhakrishnan, Raymond A. Yeh, Yu-Chiang Frank Wang, Min-Hung Chen

**发布时间:** 2025-12-18

**备注:** Project page: https://ca-joe-yang.github.io/resource/projects/4D_RGPT

### GPT解析

### 总结

该研究针对多模态大语言模型在3D结构和时间动态推理方面的局限性，提出了4D-RGPT模型、P4D训练框架和R4D-Bench基准测试，显著提升了模型在4D视频问答任务上的表现。

### 背景

尽管多模态大语言模型取得了进展，但它们对3D结构和时间动态的推理能力仍然有限，受限于弱的4D感知和时间理解能力。现有的3D和4D视频问答基准测试也强调静态场景，缺乏区域级提示。

### 目的

解决多模态大语言模型在4D感知和时间理解方面的局限性，以及现有基准测试中缺乏区域级提示的问题。

### 方法

引入4D-RGPT，一种专门设计的多模态大语言模型，能够从视频输入中捕获4D表示并增强时间感知；提出感知4D蒸馏训练框架，将冻结专家模型的4D表示转移到4D-RGPT中；构建R4D-Bench基准测试，这是一个具有区域级提示的深度感知动态场景基准测试，通过混合自动化和人工验证的流程构建。

### 主要发现

4D-RGPT在现有4D视频问答基准测试和提出的R4D-Bench基准测试上都取得了显著的改进。

### 结论

通过4D-RGPT、P4D训练框架和R4D-Bench基准测试，有效解决了多模态大语言模型在4D感知和时间理解方面的局限性，提升了模型在动态场景理解和推理任务上的表现。

### 翻译

尽管多模态大语言模型取得了进展，但它们对3D结构和时间动态的推理能力仍然有限，受限于弱的4D感知和时间理解能力。现有的3D和4D视频问答基准测试也强调静态场景，缺乏区域级提示。我们通过引入以下内容解决这些问题：(a) 4D-RGPT，一种专门设计的多模态大语言模型，能够从视频输入中捕获4D表示并增强时间感知；(b) 感知4D蒸馏，一种训练框架，将冻结专家模型的4D表示转移到4D-RGPT中，实现全面的4D感知；(c) R4D-Bench，一个具有区域级提示的深度感知动态场景基准测试，通过混合自动化和人工验证的流程构建。我们的4D-RGPT在现有4D视频问答基准测试和提出的R4D-Bench基准测试上都取得了显著的改进。


### 论文摘要

Despite advances in Multimodal LLMs (MLLMs), their ability to reason over 3D structures and temporal dynamics remains limited, constrained by weak 4D perception and temporal understanding. Existing 3D and 4D Video Question Answering (VQA) benchmarks also emphasize static scenes and lack region-level prompting. We tackle these issues by introducing: (a) 4D-RGPT, a specialized MLLM designed to capture 4D representations from video inputs with enhanced temporal perception; (b) Perceptual 4D Distillation (P4D), a training framework that transfers 4D representations from a frozen expert model into 4D-RGPT for comprehensive 4D perception; and (c) R4D-Bench, a benchmark for depth-aware dynamic scenes with region-level prompting, built via a hybrid automated and human-verified pipeline. Our 4D-RGPT achieves notable improvements on both existing 4D VQA benchmarks and the proposed R4D-Bench benchmark.

---

## 65. On Network-Aware Semantic Communication and Edge-Cloud Collaborative Intelligence Systems

**论文链接:** [http://arxiv.org/abs/2512.19563v1](http://arxiv.org/abs/2512.19563v1)

**作者:** Murdadha Nasif, Ahmed Refaey Hussein

**发布时间:** 2025-12-22

### GPT解析

### 总结

这篇综述全面总结了语义通信和边缘-云协作智能作为下一代智能服务基础支撑技术的最新进展，强调语义通信通过传输任务相关语义表示而非完美比特传输，实现了通信效率与性能之间的自适应权衡。

### 背景

下一代智能服务在严格的带宽、延迟和资源约束下运行，需要新的通信范式来满足需求，传统的比特完美传输方式已无法满足这些要求。

### 目的

提供对边缘-云接口处语义通信最新进展的全面和系统性的综合，包括架构模型、表示学习技术、编码策略和优化机制，并探讨其实际应用及未来研究方向。

### 方法

采用系统级综合方法，涵盖协作智能的架构模型、表示学习和语义抽象技术、网络感知和资源自适应的语义编码策略，以及学习驱动的优化和编排机制。

### 主要发现

语义通信能够实现通信开销、推理准确性、计算负载和端到端延迟之间的自适应权衡，并在安全、信任、弹性和可扩展性等方面具有实际运营价值，与零信任网络和物理层安全等新兴范式相关联。

### 结论

语义通信是构建AI原生网络和6G就绪智能系统的关键模块，具有广阔的应用前景和研究价值，但仍面临开放挑战需要进一步研究。

### 翻译

语义通信和边缘-云协作智能正逐渐被认为是下一代智能服务的基础支撑技术，这些服务在严格的带宽、延迟和资源约束下运行。通过将通信目标从完美的比特传输转向传输任务相关的语义表示，语义通信实现了通信开销、推理准确性、计算负载和端到端延迟之间的自适应权衡。这篇综述全面总结了边缘-云接口处语义通信的最新进展，包括协作智能的架构模型、表示学习和语义抽象技术、网络感知和资源自适应的语义编码策略，以及学习驱动的优化和编排机制。除了效率考虑外，该综述还将语义通信置于实际的运营环境中，包括安全、信任、弹性和可扩展性，并联系到零信任网络、物理层安全和新兴的边缘-云控制范式。最后，确定了开放挑战和研究方向，强调了语义通信作为AI原生网络和6G就绪智能系统的关键构建模块的作用。


### 论文摘要

Semantic communication and edge-cloud collaborative intelligence are increasingly recognized as foundational enablers for next-generation intelligent services operating under stringent bandwidth, latency, and resource constraints. By shifting the communication objective from bit-perfect delivery toward the transmission of task-relevant semantic representations, semantic communication enables adaptive tradeoffs among communication overhead, inference accuracy, computational load, and end-to-end latency. This survey provides a comprehensive and system-level synthesis of recent advances in semantic communication at the edge-cloud interface, encompassing architectural models for collaborative intelligence, representation learning and semantic abstraction techniques, network-aware and resource-adaptive semantic encoding strategies, and learning-driven optimization and orchestration mechanisms. Beyond efficiency considerations, the survey situates semantic communication within practical operational contexts, including security, trust, resilience, and scalability, drawing connections to zero-trust networking, physical-layer security, and emerging edge-cloud control paradigms. Finally, open challenges and research directions are identified, highlighting the role of semantic communication as a key building block for AI-native networking and 6G-ready intelligent systems.

---

## 66. Learning Continuous Solvent Effects from Transient Flow Data: A Graph Neural Network Benchmark on Catechol Rearrangement

**论文链接:** [http://arxiv.org/abs/2512.19530v1](http://arxiv.org/abs/2512.19530v1)

**作者:** Hongsheng Xing, Qiuxin Si

**发布时间:** 2025-12-22

**备注:** 13 pages, 6 figures

### GPT解析

### 总结

该研究提出了一个混合GNN架构，用于预测连续溶剂组成范围内的反应结果，显著提高了预测精度。

### 背景

预测连续溶剂组成范围内的反应结果是有机合成和工艺化学中的关键挑战。传统机器学习方法将溶剂视为离散变量，无法进行系统内推和外推。

### 目的

开发能够处理连续溶剂组成范围的预测方法，并创建基准数据集评估不同方法在连续溶剂空间中的性能。

### 方法

引入'Catechol Benchmark'数据集，包含1227个实验产率测量值，涵盖24种纯溶剂及其二元混合物；评估多种架构；提出混合GNN架构，结合图注意力网络、差分反应指纹和混合感知溶剂编码。

### 主要发现

经典表格方法和大型语言模型在定量精度上表现不佳(MSE分别为0.099和0.129)；提出的混合GNN架构实现MSE为0.0039，比基线减少60%错误，比表格集成提高25倍以上；分子图消息传递和连续混合编码对稳健泛化至关重要。

### 结论

完整数据集、评估协议和参考实现已发布，将促进数据高效的反应预测和连续溶剂表示学习。

### 翻译

在有机合成和工艺化学中，预测连续溶剂组成范围内的反应结果仍然是一个关键挑战。传统的机器学习方法通常将溶剂身份视为离散的分类变量，这阻碍了在溶剂空间中的系统内插和外推。这项工作引入了'Catechol Benchmark'，一个包含1227个实验产率测量值的高通量瞬态流动化学数据集，涉及在24种纯溶剂及其二元混合物中取代烯丙基的儿茶酚的重排，通过连续体积分数(%B)参数化。我们在严格的留一溶剂和留一混合物协议下评估各种架构，以测试对未见化学环境的泛化能力。我们的结果表明，经典表格方法(如梯度提升决策树)和大语言模型嵌入(如Qwen-7B)在定量精度上表现不佳，分别产生0.099和0.129的平均平方误差(MSE)。相比之下，我们提出了一种基于混合GNN的架构，集成了图注意力网络(GATs)、差分反应指纹(DRFP)和学习的混合感知溶剂编码。该方法实现了0.0039的MSE(±0.0003)，比竞争基线减少了60%的错误，比表格集成提高了25倍以上。消融研究证实，明确的分子图消息传递和连续混合编码对于稳健泛化是必不可少的。完整的数据集、评估协议和参考实现已发布，以促进数据高效的反应预测和连续溶剂表示学习。


### 论文摘要

Predicting reaction outcomes across continuous solvent composition ranges remains a critical challenge in organic synthesis and process chemistry. Traditional machine learning approaches often treat solvent identity as a discrete categorical variable, which prevents systematic interpolation and extrapolation across the solvent space. This work introduces the \textbf{Catechol Benchmark}, a high-throughput transient flow chemistry dataset comprising 1,227 experimental yield measurements for the rearrangement of allyl-substituted catechol in 24 pure solvents and their binary mixtures, parameterized by continuous volume fractions ($\% B$). We evaluate various architectures under rigorous leave-one-solvent-out and leave-one-mixture-out protocols to test generalization to unseen chemical environments.   Our results demonstrate that classical tabular methods (e.g., Gradient-Boosted Decision Trees) and large language model embeddings (e.g., Qwen-7B) struggle with quantitative precision, yielding Mean Squared Errors (MSE) of 0.099 and 0.129, respectively. In contrast, we propose a hybrid GNN-based architecture that integrates Graph Attention Networks (GATs) with Differential Reaction Fingerprints (DRFP) and learned mixture-aware solvent encodings. This approach achieves an \textbf{MSE of 0.0039} ($\pm$ 0.0003), representing a 60\% error reduction over competitive baselines and a $>25\times$ improvement over tabular ensembles. Ablation studies confirm that explicit molecular graph message-passing and continuous mixture encoding are essential for robust generalization. The complete dataset, evaluation protocols, and reference implementations are released to facilitate data-efficient reaction prediction and continuous solvent representation learning.

---

## 67. 论文ID: 2512.19510v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19510v1.json'

---

## 68. 论文ID: 2512.19504v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19504v1.json'

---

## 69. MT-Mark: Rethinking Image Watermarking via Mutual-Teacher Collaboration with Adaptive Feature Modulation

**论文链接:** [http://arxiv.org/abs/2512.19438v1](http://arxiv.org/abs/2512.19438v1)

**作者:** Fei Ge, Ying Huang, Jie Liu, Guixuan Zhang, Zhi Zeng, Shuwu Zhang, Hu Guan

**发布时间:** 2025-12-22

### GPT解析

### 总结

该研究重新思考了深度图像水印方法，引入协作交互机制(CIM)和自适应特征调制模块(AFMM)，通过建立嵌入器和提取器之间的直接双向通信实现协作优化，显著提升了水印提取准确性和鲁棒性。

### 背景

现有深度图像水印方法遵循固定嵌入-失真-提取流水线，嵌入器和提取器通过最终损失函数弱耦合且独立优化，缺乏明确协作机制，没有结构化方式让嵌入器整合解码感知线索或让提取器指导嵌入过程。

### 目的

解决传统架构限制，将嵌入和提取重新设计为明确协作组件，引入协作交互机制建立双向通信，并提出自适应特征调制模块支持有效交互。

### 方法

引入协作交互机制(CIM)建立嵌入器和提取器间直接双向通信，采用相互教师训练范式；提出自适应特征调制模块(AFMM)通过解耦调制结构和强度实现内容感知特征调节，引导水印嵌入向稳定图像特征同时抑制宿主干扰；两侧AFMM形成闭环协作使嵌入行为与提取目标一致。

### 主要发现

架构级重新设计改变了水印系统中鲁棒性的学习方式，鲁棒性不再依赖详尽失真模拟，而是从嵌入和提取间的协调表示学习中涌现；实验证明该方法在真实世界和AI生成数据集上均优于最先进方法，同时保持高感知质量。

### 结论

所提出的CIM和AFMM架构成功解决了传统深度图像水印方法的架构限制，通过明确协作机制实现了更好的水印提取准确性和感知质量，在各种数据集上表现出强大的鲁棒性和泛化能力。

### 翻译

现有的深度图像水印方法遵循固定的嵌入-失真-提取流水线，其中嵌入器和提取器通过最终损失函数弱耦合，并且独立优化。这种设计缺乏明确的协作，没有结构化的机制让嵌入器整合解码感知线索，也没有让提取器在训练过程中指导嵌入。为了解决这一架构限制，我们通过将嵌入和提取重新明确设计为协作组件来重新思考深度图像水印。为了实现这一重新设计，我们引入了协作交互机制(CIM)，该机制在嵌入器和提取器之间建立直接的双向通信，实现了相互教师训练范式和协调优化。基于这种明确协作的架构，我们进一步提出了自适应特征调制模块(AFMM)来支持有效交互。AFMM通过解耦调制结构和强度实现内容感知特征调节，引导水印嵌入向稳定的图像特征，同时在提取过程中抑制宿主干扰。在CIM下，两侧的AFMM形成闭环协作，使嵌入行为与提取目标保持一致。这种架构级的重新设计改变了水印系统中鲁棒性的学习方式。鲁棒性不再依赖于详尽的失真模拟，而是从嵌入和提取之间的协调表示学习中涌现。在真实世界和AI生成的数据集上的实验表明，所提出的方法在水印提取准确性方面始终优于最先进的方法，同时保持高感知质量，显示出强大的鲁棒性和泛化能力。


### 论文摘要

Existing deep image watermarking methods follow a fixed embedding-distortion-extraction pipeline, where the embedder and extractor are weakly coupled through a final loss and optimized in isolation. This design lacks explicit collaboration, leaving no structured mechanism for the embedder to incorporate decoding-aware cues or for the extractor to guide embedding during training. To address this architectural limitation, we rethink deep image watermarking by reformulating embedding and extraction as explicitly collaborative components. To realize this reformulation, we introduce a Collaborative Interaction Mechanism (CIM) that establishes direct, bidirectional communication between the embedder and extractor, enabling a mutual-teacher training paradigm and coordinated optimization. Built upon this explicitly collaborative architecture, we further propose an Adaptive Feature Modulation Module (AFMM) to support effective interaction. AFMM enables content-aware feature regulation by decoupling modulation structure and strength, guiding watermark embedding toward stable image features while suppressing host interference during extraction. Under CIM, the AFMMs on both sides form a closed-loop collaboration that aligns embedding behavior with extraction objectives. This architecture-level redesign changes how robustness is learned in watermarking systems. Rather than relying on exhaustive distortion simulation, robustness emerges from coordinated representation learning between embedding and extraction. Experiments on real-world and AI-generated datasets demonstrate that the proposed method consistently outperforms state-of-the-art approaches in watermark extraction accuracy while maintaining high perceptual quality, showing strong robustness and generalization.

---

## 70. Cluster-Based Generalized Additive Models Informed by Random Fourier Features

**论文链接:** [http://arxiv.org/abs/2512.19373v1](http://arxiv.org/abs/2512.19373v1)

**作者:** Xin Huang, Jia Li, Jun Yu

**发布时间:** 2025-12-22

**备注:** 25 pages, 13 figures, 4 tables

### GPT解析

### 总结

这篇论文提出了一种基于随机傅里叶特征和广义加性模型混合体的新方法，用于实现高预测性能同时保持模型可解释性。

### 背景

可解释机器学习旨在平衡预测准确性和模型透明性，特别是在黑盒预测模型（如深度神经网络或基于核的方法）表现出色但难以解释的情况下。

### 目的

提出一种结合表示学习和透明统计建模的方法，通过广义加性模型混合体来揭示数据中的局部自适应结构。

### 方法

利用随机傅里叶特征表示来发现数据中的局部自适应结构，首先学习基于RFF的嵌入，然后通过主成分分析进行压缩，使用高斯混合模型对数据进行软聚类，构建混合GAM框架，其中每个局部GAM通过可解释的单变量平滑函数捕获非线性效应。

### 主要发现

在加州住房、NASA翼型自噪声和自行车共享等真实世界回归基准数据集上的数值实验表明，该方法相对于经典可解释模型具有改进的预测性能。

### 结论

这种构建为将表示学习与透明统计建模相结合提供了原则性的方法。

### 翻译

可解释机器学习旨在平衡预测准确性和模型透明性，特别是在黑盒预测模型（如深度神经网络或基于核的方法）取得强大经验性能但仍然难以解释的情况下。这项工作引入了广义加性模型混合体，其中利用随机傅里叶特征表示来揭示数据中的局部自适应结构。在所提出的方法中，首先学习基于RFF的嵌入，然后通过主成分分析进行压缩。所得的低维表示通过高斯混合模型执行数据的软聚类。然后应用这些聚类分配来构建混合GAM框架，其中每个局部GAM通过可解释的单变量平滑函数捕获非线性效应。在加州住房、NASA翼型自噪声和自行车共享等真实世界回归基准数据集上的数值实验表明，相对于经典可解释模型具有改进的预测性能。总体而言，这种构建为将表示学习与透明统计建模相结合提供了原则性的方法。


### 论文摘要

Explainable machine learning aims to strike a balance between prediction accuracy and model transparency, particularly in settings where black-box predictive models, such as deep neural networks or kernel-based methods, achieve strong empirical performance but remain difficult to interpret. This work introduces a mixture of generalized additive models (GAMs) in which random Fourier feature (RFF) representations are leveraged to uncover locally adaptive structure in the data. In the proposed method, an RFF-based embedding is first learned and then compressed via principal component analysis. The resulting low-dimensional representations are used to perform soft clustering of the data through a Gaussian mixture model. These cluster assignments are then applied to construct a mixture-of-GAMs framework, where each local GAM captures nonlinear effects through interpretable univariate smooth functions. Numerical experiments on real-world regression benchmarks, including the California Housing, NASA Airfoil Self-Noise, and Bike Sharing datasets, demonstrate improved predictive performance relative to classical interpretable models. Overall, this construction provides a principled approach for integrating representation learning with transparent statistical modeling.

---

## 71. Efficient Spike-driven Transformer for High-performance Drone-View Geo-Localization

**论文链接:** [http://arxiv.org/abs/2512.19365v1](http://arxiv.org/abs/2512.19365v1)

**作者:** Zhongwei Chen, Hai-Jun Rong, Zhao-Xu Yang, Guoqi Li

**发布时间:** 2025-12-22

### GPT解析

### 总结

该研究提出了SpikeViMFormer，这是第一个专为无人机视图地理定位(DVGL)设计的脉冲神经网络(SNN)框架，解决了传统人工神经网络(ANN)高功耗问题以及SNN在表征学习中的信息丢失和长程依赖学习困难问题。

### 背景

传统基于ANN的DVGL方法虽表现优异，但依赖密集计算导致高功耗；SNN虽有低功耗优势，但在DVGL领域的潜力尚未充分探索，且其脉冲驱动计算的固有稀疏性会导致关键信息丢失和学习长程依赖的困难。

### 目的

开发一种SNN框架，解决DVGL应用中的功耗问题，同时克服SNN在表征学习中的信息丢失和长程依赖学习困难。

### 方法

提出SpikeViMFormer框架，采用轻量级脉冲驱动Transformer主干网络提取特征；设计脉冲驱动选择性注意力(SSA)块通过门控机制实现选择性特征增强；引入脉冲驱动混合状态空间(SHS)块学习长程依赖；推理阶段仅使用主干网络降低计算成本；提出分层重排序对齐学习(HRAL)策略优化主干网络。

### 主要发现

SpikeViMFormer性能优于最先进的SNN，与先进的ANN相比也具有竞争力，同时保持了SNN的低功耗优势。

### 结论

SpikeViMFormer是首个专为DVGL设计的SNN框架，有效解决了SNN在该应用中的关键问题，在保持低功耗的同时实现了与先进ANN相当的性能。

### 翻译

传统的基于人工神经网络(ANN)的无人机视图地理定位(DVGL)方法已取得了显著性能。然而，ANN依赖密集计算，导致高功耗。相比之下，受益于脉冲驱动计算的脉冲神经网络(SNN) inherently 提供低功耗。遗憾的是，SNN在DVGL方面的潜力尚未得到充分研究。同时，脉冲驱动计算在表征学习场景中的固有稀疏性也导致关键信息丢失和对齐异构视觉数据源时学习长程依赖的困难。为解决这些问题，我们提出了SpikeViMFormer，这是第一个为DVGL设计的SNN框架。在该框架中，采用轻量级脉冲驱动Transformer主干网络提取粗粒度特征。为减轻关键信息丢失，设计了脉冲驱动选择性注意力(SSA)块，使用脉冲驱动门控机制实现选择性特征增强，突出判别性区域。此外，引入了脉冲驱动混合状态空间(SHS)块，使用混合状态空间学习长程依赖。而且，在推理阶段仅使用主干网络以减少计算成本。为确保主干网络有效性，提出了一种新的分层重排序对齐学习(HRAL)策略。它通过邻域重排序细化特征，保持跨批次一致性以直接优化主干网络。实验结果表明，SpikeViMFormer优于最先进的SNN。与先进的ANN相比，它也取得了具有竞争力的性能。我们的代码可在https://github.com/ISChenawei/SpikeViMFormer获取。


### 论文摘要

Traditional drone-view geo-localization (DVGL) methods based on artificial neural networks (ANNs) have achieved remarkable performance. However, ANNs rely on dense computation, which results in high power consumption. In contrast, spiking neural networks (SNNs), which benefit from spike-driven computation, inherently provide low power consumption. Regrettably, the potential of SNNs for DVGL has yet to be thoroughly investigated. Meanwhile, the inherent sparsity of spike-driven computation for representation learning scenarios also results in loss of critical information and difficulties in learning long-range dependencies when aligning heterogeneous visual data sources. To address these, we propose SpikeViMFormer, the first SNN framework designed for DVGL. In this framework, a lightweight spike-driven transformer backbone is adopted to extract coarse-grained features. To mitigate the loss of critical information, the spike-driven selective attention (SSA) block is designed, which uses a spike-driven gating mechanism to achieve selective feature enhancement and highlight discriminative regions. Furthermore, a spike-driven hybrid state space (SHS) block is introduced to learn long-range dependencies using a hybrid state space. Moreover, only the backbone is utilized during the inference stage to reduce computational cost. To ensure backbone effectiveness, a novel hierarchical re-ranking alignment learning (HRAL) strategy is proposed. It refines features via neighborhood re-ranking and maintains cross-batch consistency to directly optimize the backbone. Experimental results demonstrate that SpikeViMFormer outperforms state-of-the-art SNNs. Compared with advanced ANNs, it also achieves competitive performance.Our code is available at https://github.com/ISChenawei/SpikeViMFormer

---

## 72. 论文ID: 2512.19194v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19194v1.json'

---

## 73. Fraud Detection Through Large-Scale Graph Clustering with Heterogeneous Link Transformation

**论文链接:** [http://arxiv.org/abs/2512.19061v1](http://arxiv.org/abs/2512.19061v1)

**作者:** Chi Liu

**发布时间:** 2025-12-22

**备注:** 13 pages, 6 figures

### GPT解析

### 总结

论文提出了一种创新的基于图的欺诈检测方法，通过区分硬链接和软链接，结合图转换技术和LINE与HDBSCAN算法，有效解决了大规模异构图聚类问题，显著提高了欺诈检测的覆盖范围和效率。

### 背景

协作欺诈问题日益严重，多个欺诈账户协调利用在线支付系统形成复杂网络结构。传统检测方法存在局限性：仅依赖高置信度身份链接的方法覆盖范围有限，而使用所有可用链接的方法会导致图形碎片化，降低聚类效果。

### 目的

提出一种新的基于图的欺诈检测框架，解决大规模异构图聚类挑战，通过有原则的链接转换方法提高欺诈检测的覆盖范围和效率。

### 方法

区分硬链接（高置信度身份关系，如电话号码、信用卡、国家ID）和软链接（行为关联，包括设备指纹、cookies和IP地址）。采用图转换技术：通过硬链接识别连接组件，合并为超节点，重建加权软链接图。使用LINE进行表示学习，使用HDBSCAN进行基于密度的聚类发现。

### 主要发现

在真实支付平台数据集上实验表明：实现显著的图形规模缩减（从2500万节点减少到770万节点）；与仅使用硬链接的基线方法相比，检测覆盖范围提高一倍；在识别的欺诈聚类中保持高精度。

### 结论

该框架为工业规模的欺诈检测系统提供了可扩展且实用的解决方案。

### 翻译

协作欺诈是指多个欺诈账户协调利用在线支付系统，由于形成复杂的网络结构而带来重大挑战。仅依赖高置信度身份链接的传统检测方法存在覆盖范围有限的缺陷，而使用所有可用链接的方法往往导致碎片化的图形，降低聚类效果。在本文中，我们提出了一种新颖的基于图的欺诈检测框架，通过有原则的链接转换方法解决大规模异构图聚类挑战。我们的方法区分硬链接（高置信度身份关系，如电话号码、信用卡和国家ID）和软链接（行为关联，包括设备指纹、cookies和IP地址）。我们引入了一种图转换技术，首先通过硬链接识别连接组件，将它们合并为超节点，然后重建一个适合高效嵌入和聚类的加权软链接图。使用LINE（大规模信息网络嵌入）处理变换后的图进行表示学习，接着使用HDBSCAN（基于密度的空间聚类应用与噪声）进行基于密度的聚类发现。在真实支付平台数据集上的实验表明，我们的方法实现了显著的图形规模缩减（从2500万减少到770万节点），比仅使用硬链接的基线方法提高了一倍的检测覆盖率，并在识别的欺诈聚类中保持高精度。我们的框架为工业规模的欺诈检测系统提供了可扩展且实用的解决方案。


### 论文摘要

Collaborative fraud, where multiple fraudulent accounts coordinate to exploit online payment systems, poses significant challenges due to the formation of complex network structures. Traditional detection methods that rely solely on high-confidence identity links suffer from limited coverage, while approaches using all available linkages often result in fragmented graphs with reduced clustering effectiveness. In this paper, we propose a novel graph-based fraud detection framework that addresses the challenge of large-scale heterogeneous graph clustering through a principled link transformation approach. Our method distinguishes between \emph{hard links} (high-confidence identity relationships such as phone numbers, credit cards, and national IDs) and \emph{soft links} (behavioral associations including device fingerprints, cookies, and IP addresses). We introduce a graph transformation technique that first identifies connected components via hard links, merges them into super-nodes, and then reconstructs a weighted soft-link graph amenable to efficient embedding and clustering. The transformed graph is processed using LINE (Large-scale Information Network Embedding) for representation learning, followed by HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) for density-based cluster discovery. Experiments on a real-world payment platform dataset demonstrate that our approach achieves significant graph size reduction (from 25 million to 7.7 million nodes), doubles the detection coverage compared to hard-link-only baselines, and maintains high precision across identified fraud clusters. Our framework provides a scalable and practical solution for industrial-scale fraud detection systems.

---

## 74. 论文ID: 2512.18554v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18554v1.json'

---

## 75. FairExpand: Individual Fairness on Graphs with Partial Similarity Information

**论文链接:** [http://arxiv.org/abs/2512.18180v1](http://arxiv.org/abs/2512.18180v1)

**作者:** Rebecca Salganik, Yibin Wang, Guillaume Salha-Galvan, Jian Kang

**发布时间:** 2025-12-20

### GPT解析

### 总结

本文提出了FairExpand框架，用于在只有部分节点对相似性信息的情况下实现图表示学习中的个体公平性，解决了现有方法需要所有节点对预定义相似性信息的不现实假设问题。

### 背景

个体公平性已成为公平机器学习的核心原则，在图表示学习中受到关注，因为它在高风险的Web领域（如用户建模、推荐系统和搜索）具有实际重要性。

### 目的

研究旨在解决现有方法需要所有节点对预定义相似性信息的不现实假设，提出一种在只有部分相似性信息的情况下促进个体公平性的实用框架。

### 方法

FairExpand采用两步流程，交替使用骨干模型（如图神经网络）细化节点表示和逐渐传播相似性信息，使公平性能够有效扩展到整个图。

### 主要发现

大量实验表明，FairExpand在保持模型性能的同时持续增强个体公平性，使其成为在具有部分相似性信息的现实应用中实现基于图的个体公平性的有效解决方案。

### 结论

FairExpand为在现实世界应用中实现图表示学习的个体公平性提供了实用解决方案，克服了现有方法对完整相似性信息依赖的限制。

### 翻译

个体公平性要求算法系统对相似个体给予相似对待，已成为公平机器学习的中心原则。个体公平性在图表示学习中受到关注，因为它在高风险的Web领域（如用户建模、推荐系统和搜索）具有实际重要性。然而，现有方法假设所有节点对都存在预定义的相似性信息，这是一个通常不现实的要求，阻碍了它们的实际应用。本文假设相似性信息仅适用于有限的节点对子集，并提出了FairExpand框架，促进在更现实的局部信息场景下的个体公平性。FairExpand遵循两步流程，交替使用骨干模型（如图神经网络）细化节点表示和逐渐传播相似性信息，使公平性能够有效扩展到整个图。大量实验表明，FairExpand在保持性能的同时持续增强个体公平性，使其成为在具有部分相似性信息的现实应用中实现基于图的个体公平性的实用解决方案。


### 论文摘要

Individual fairness, which requires that similar individuals should be treated similarly by algorithmic systems, has become a central principle in fair machine learning. Individual fairness has garnered traction in graph representation learning due to its practical importance in high-stakes Web areas such as user modeling, recommender systems, and search. However, existing methods assume the existence of predefined similarity information over all node pairs, an often unrealistic requirement that prevents their operationalization in practice. In this paper, we assume the similarity information is only available for a limited subset of node pairs and introduce FairExpand, a flexible framework that promotes individual fairness in this more realistic partial information scenario. FairExpand follows a two-step pipeline that alternates between refining node representations using a backbone model (e.g., a graph neural network) and gradually propagating similarity information, which allows fairness enforcement to effectively expand to the entire graph. Extensive experiments show that FairExpand consistently enhances individual fairness while preserving performance, making it a practical solution for enabling graph-based individual fairness in real-world applications with partial similarity information.

---

## 76. Unifying Causal Reinforcement Learning: Survey, Taxonomy, Algorithms and Applications

**论文链接:** [http://arxiv.org/abs/2512.18135v1](http://arxiv.org/abs/2512.18135v1)

**作者:** Cristiano da Costa Cunha, Wei Liu, Tim French, Ajmal Mian

**发布时间:** 2025-12-19

**备注:** 26 pages, 14 figures, 5 algorithms

### GPT解析

### 总结

因果推断与强化学习的集成已成为解决传统强化学习局限性的有力范式，通过明确建模因果关系提高系统的可解释性、鲁棒性和泛化能力。

### 背景

传统强化学习技术依赖相关性驱动的决策，在面对分布偏移、混杂变量和动态环境时表现不佳，存在可解释性低、鲁棒性差和泛化能力不足等局限性。

### 目的

系统性地回顾因果推断与强化学习交叉领域的最新进展，识别现有方法的挑战，强调实际应用中的成功经验，讨论开放性问题，并提供未来研究方向。

### 方法

将现有方法分类为因果表示学习、反事实策略优化、离线因果强化学习、因果迁移学习和因果可解释性，并通过结构化分析进行系统综述。

### 主要发现

因果强化学习通过利用因果推断的基本原理，明确建模因果关系，为传统强化学习面临的挑战提供了有希望的解决方案，并在实际应用中取得了成功。

### 结论

因果强化学习在开发鲁棒、可泛化和可解释的人工智能系统方面具有巨大潜力，为未来人工智能研究提供了重要方向。

### 翻译

将因果推断(CI)与强化学习(RL)集成已成为解决传统RL关键局限性的有力范式，包括低可解释性、缺乏鲁棒性和泛化失败。传统RL技术通常依赖相关性驱动的决策，在面对分布偏移、混杂变量和动态环境时表现不佳。因果强化学习(CRL)利用因果推断的基本原理，通过明确建模因果关系，为这些挑战提供了有希望的解决方案。在本综述中，我们系统地回顾了因果推断与RL交叉领域的最新进展。我们将现有方法分为因果表示学习、反事实策略优化、离线因果RL、因果迁移学习和因果可解释性。通过这种结构化分析，我们确定了当前面临的挑战，强调了实际应用中的成功经验，并讨论了开放性问题。最后，我们提供了未来研究方向，强调了CRL在开发鲁棒、可泛化和可解释的人工智能系统方面的潜力。


### 论文摘要

Integrating causal inference (CI) with reinforcement learning (RL) has emerged as a powerful paradigm to address critical limitations in classical RL, including low explainability, lack of robustness and generalization failures. Traditional RL techniques, which typically rely on correlation-driven decision-making, struggle when faced with distribution shifts, confounding variables, and dynamic environments. Causal reinforcement learning (CRL), leveraging the foundational principles of causal inference, offers promising solutions to these challenges by explicitly modeling cause-and-effect relationships. In this survey, we systematically review recent advancements at the intersection of causal inference and RL. We categorize existing approaches into causal representation learning, counterfactual policy optimization, offline causal RL, causal transfer learning, and causal explainability. Through this structured analysis, we identify prevailing challenges, highlight empirical successes in practical applications, and discuss open problems. Finally, we provide future research directions, underscoring the potential of CRL for developing robust, generalizable, and interpretable artificial intelligence systems.

---

## 77. Factorized Transport Alignment for Multimodal and Multiview E-commerce Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.18117v1](http://arxiv.org/abs/2512.18117v1)

**作者:** Xiwen Chen, Yen-Chieh Lien, Susan Liu, María Castaños, Abolfazl Razi, Xiaoting Zhao, Congzhe Su

**发布时间:** 2025-12-19

**备注:** Accepted by WSDM'26

### GPT解析

### 总结

该论文提出了一种基于因子传输的多视图学习框架，解决了电子商务平台中现有视觉语言模型仅关注主图像而忽略辅助图像和文本信息的问题。该方法在训练时优化主视图并随机采样辅助视图，在推理时将所有视图融合为单个缓存嵌入，在大型电商数据集上显著提升了检索性能。

### 背景

电子商务的快速增长需要能够捕捉用户生成列表中多样化信号的鲁棒多模态表示。现有的视觉语言模型通常只将标题与主图像对齐，忽略了在开放市场平台中提供关键语义的非主图像和辅助文本视图。

### 目的

提出一种统一多模态和多视图学习的框架，通过因子传输方法（最优传输的轻量级近似）解决现有VLMs的局限性，提高可扩展性和部署效率。

### 方法

1) 提出基于因子传输的框架统一多模态和多视图学习；2) 训练时强调主视图并随机采样辅助视图，将训练成本从二次降低到常数级；3) 推理时将所有视图融合为单个缓存嵌入，保留双塔检索效率；4) 在包含100万产品列表和30万次交互的工业数据集上进行测试。

### 主要发现

在工业数据集上，该方法在跨视图和查询到项目检索方面取得持续改进，相比强大的多模态基线，Recall@500提升了高达7.9%。

### 结论

该框架将可扩展性与基于最优传输的学习相结合，使多视图预训练在大规模电子商务搜索中变得实用。

### 翻译

电子商务的快速增长需要能够捕捉用户生成列表中多样化信号的鲁棒多模态表示。现有的视觉语言模型通常将标题与主图像对齐，即单视图方法，但忽略了在Etsy或Poshmark等开放市场中提供关键语义的非主图像和辅助文本视图。为此，我们提出了一种通过因子传输统一多模态和多视图学习的框架，这是最优传输的轻量级近似，专为可扩展性和部署效率而设计。在训练期间，该方法强调主视图，同时随机采样辅助视图，将训练成本从与视图数量的二次关系降低到每项目常数级。在推理时，所有视图被融合为单个缓存嵌入，保留了双塔检索的效率，没有额外的在线开销。在包含100万个产品列表和30万次交互的工业数据集上，我们的方法在跨视图和查询到项目检索方面取得了持续改进，相比强大的多模态基线，Recall@500提升了高达7.9%。总体而言，我们的框架将可扩展性与基于最优传输的学习相结合，使多视图预训练在大规模电子商务搜索中变得实用。


### 论文摘要

The rapid growth of e-commerce requires robust multimodal representations that capture diverse signals from user-generated listings. Existing vision-language models (VLMs) typically align titles with primary images, i.e., single-view, but overlook non-primary images and auxiliary textual views that provide critical semantics in open marketplaces such as Etsy or Poshmark. To this end, we propose a framework that unifies multimodal and multi-view learning through Factorized Transport, a lightweight approximation of optimal transport, designed for scalability and deployment efficiency. During training, the method emphasizes primary views while stochastically sampling auxiliary ones, reducing training cost from quadratic in the number of views to constant per item. At inference, all views are fused into a single cached embedding, preserving the efficiency of two-tower retrieval with no additional online overhead. On an industrial dataset of 1M product listings and 0.3M interactions, our approach delivers consistent improvements in cross-view and query-to-item retrieval, achieving up to +7.9% Recall@500 over strong multimodal baselines. Overall, our framework bridges scalability with optimal transport-based learning, making multi-view pretraining practical for large-scale e-commerce search.

---

## 78. Greater than the Sum of Its Parts: Building Substructure into Protein Encoding Models

**论文链接:** [http://arxiv.org/abs/2512.18114v1](http://arxiv.org/abs/2512.18114v1)

**作者:** Robert Calef, Arthur Liang, Manolis Kellis, Marinka Zitnik

**发布时间:** 2025-12-19

### GPT解析

### 总结

本文介绍了Magneton，一个用于开发亚结构感知蛋白质模型的环境，提供了大规模亚结构注释数据集、训练框架和多层次基准任务。通过亚结构微调方法，成功将亚结构知识整合到预训练蛋白质模型中，改善了功能预测并产生了更一致的表示。

### 背景

蛋白质表示学习随着序列和结构监督的规模扩大而快速发展，但大多数模型仍将蛋白质编码为残基序列或全局嵌入，忽略了蛋白质是由重复的、进化上保守的亚结构组成这一关键特性。

### 目的

开发一个能够识别和利用蛋白质亚结构的环境，以改进蛋白质表示学习，并探索亚结构信息如何补充全局结构信息。

### 方法

创建了Magneton环境，包含(1)530,601个蛋白质的亚结构注释数据集，(2)亚结构整合训练框架，(3)多层次基准任务套件。并开发了亚结构微调方法，将亚结构知识蒸馏到预训练蛋白质模型中。

### 主要发现

亚结构微调改善了功能预测，产生了对未观察过的亚结构类型更一致的表示，表明亚结构监督提供了与全局结构输入互补的信息。

### 结论

亚结构感知的蛋白质模型能够更好地捕捉蛋白质的生物学特性，Magneton环境为开发此类模型提供了完整工具链。

### 翻译

蛋白质表示学习随着序列和结构监督的规模扩大而迅速发展，但大多数模型仍然将蛋白质编码为每个残基的标记序列或单个全局嵌入。这忽略了蛋白质组织的定义特性：蛋白质是由重复的、进化上保守的亚结构组成，这些亚结构集中了生化活性并介导核心分子功能。尽管结构域和功能位点等亚结构已被系统性地分类，但它们很少被用作蛋白质模型中的训练信号或表示单位。我们引入了Magneton，这是一个用于开发亚结构感知蛋白质模型的环境。Magneton提供(1)一个包含530,601个蛋白质的数据集，标注了超过170万个亚结构，涵盖13,075种类型，(2)一个将亚结构整合到现有蛋白质模型中的训练框架，(3)一个包含13个任务的基准套件，探测残基、亚结构和蛋白质水平的表示。使用Magneton，我们开发了亚结构微调，这是一种监督微调方法，可以将亚结构知识预训练到蛋白质模型中。在最先进的序列和结构模型中，亚结构微调改善了功能预测，产生了对在微调过程中从未观察到的亚结构类型更一致的表示，并表明亚结构监督提供了与全局结构输入互补的信息。Magneton环境、数据集和亚结构微调模型都是公开可用的。


### 论文摘要

Protein representation learning has advanced rapidly with the scale-up of sequence and structure supervision, but most models still encode proteins either as per-residue token sequences or as single global embeddings. This overlooks a defining property of protein organization: proteins are built from recurrent, evolutionarily conserved substructures that concentrate biochemical activity and mediate core molecular functions. Although substructures such as domains and functional sites are systematically cataloged, they are rarely used as training signals or representation units in protein models. We introduce Magneton, an environment for developing substructure-aware protein models. Magneton provides (1) a dataset of 530,601 proteins annotated with over 1.7 million substructures spanning 13,075 types, (2) a training framework for incorporating substructures into existing protein models, and (3) a benchmark suite of 13 tasks probing representations at the residue, substructural, and protein levels. Using Magneton, we develop substructure-tuning, a supervised fine-tuning method that distills substructural knowledge into pretrained protein models. Across state-of-the-art sequence- and structure-based models, substructure-tuning improves function prediction, yields more consistent representations of substructure types never observed during tuning, and shows that substructural supervision provides information that is complementary to global structure inputs. The Magneton environment, datasets, and substructure-tuned models are all openly available (https://github.com/rcalef/magneton/).

---

## 79. 论文ID: 2512.18056v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18056v1.json'

---

## 80. 论文ID: 2512.19494v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19494v1.json'

---

## 81. Learning General Policies with Policy Gradient Methods

**论文链接:** [http://arxiv.org/abs/2512.19366v1](http://arxiv.org/abs/2512.19366v1)

**作者:** Simon Ståhlberg, Blai Bonet, Hector Geffner

**发布时间:** 2025-12-22

**备注:** In Proceedings of the 20th International Conference on Principles of Knowledge Representation and Reasoning (KR 2023)

### GPT解析

### 总结

该研究结合强化学习与组合方法，探索如何使深度强化学习策略优化方法学习到类似组合方法的泛化能力。通过将策略建模为状态转换分类器并使用图神经网络表示值函数，实现了与组合方法几乎相当的泛化效果，同时避免了组合方法的可扩展性瓶颈。

### 背景

强化学习在多个领域已取得显著成果，但泛化能力（即可靠系统地产生泛化策略）仍是挑战。经典规划中已通过组合方法解决了泛化问题，学习了在给定领域所有实例上具有可证明正确性的泛化策略。

### 目的

将强化学习与组合方法两个研究线索结合，阐明深度强化学习方法，特别是策略优化方法，在何种条件下可学习到像组合方法那样泛化的策略。

### 方法

1. 将策略建模为状态转换分类器，因为基础动作不具有泛化性；2. 使用适应关系结构的图神经网络表示规划状态上的值函数和策略；3. 应用actor-critic方法学习泛化策略；4. 通过添加派生谓词和替代成本结构解决GNNs表达限制及最优性与泛化权衡问题。

### 主要发现

1. actor-critic方法可学习到与组合方法几乎一样好的泛化策略，同时避免可扩展性瓶颈和特征池使用；2. DRL方法限制源于GNNs表达限制及最优性与泛化权衡，而非深度学习或强化学习算法本身；3. 通过添加派生谓词和替代成本结构可在不改变基本DRL方法的情况下解决这些限制。

### 结论

通过结合组合方法和深度强化学习的优势，提出的方法使DRL能够学习到与组合方法几乎一样好的泛化策略，同时避免组合方法的可扩展性瓶颈。通过解决GNNs表达限制和最优性与泛化权衡问题，进一步提高了方法的有效性。

### 翻译

虽然强化学习方法在许多场景中已经取得了显著成果，但泛化能力，即可靠且系统地产生能泛化的策略的能力，仍然是一个挑战。在经典规划中，泛化问题已经得到了正式解决，使用组合方法学习了在给定领域所有实例上具有可证明正确性的泛化策略。本工作的目标是将这两个研究线索结合起来，阐明深度强化学习方法，特别是策略优化方法，在何种条件下可以学习到像组合方法那样泛化的策略。我们从之前组合方法和深度学习方法中汲取经验，并以方便的方式扩展它们。从前者的经验中，我们将策略建模为状态转换分类器，因为基础动作不具有泛化性，在不同实例间会变化。从后者的经验中，我们使用适应关系结构的图神经网络来表示规划状态上的值函数，在我们的案例中是策略。有了这些组件，我们发现actor-critic方法可以用来学习泛化策略，其效果几乎与组合方法获得的策略一样好，同时避免了可扩展性瓶颈和特征池的使用。此外，DRL方法在所考虑基准测试中的限制与深度学习或强化学习算法关系不大，而是源于图神经网络众所周知的表达限制，以及最优性和泛化之间的权衡（在某些领域中，泛化策略不能是最优的）。通过添加派生谓词和替代成本结构来优化，可以在不改变基本DRL方法的情况下解决这两个限制。


### 论文摘要

While reinforcement learning methods have delivered remarkable results in a number of settings, generalization, i.e., the ability to produce policies that generalize in a reliable and systematic way, has remained a challenge. The problem of generalization has been addressed formally in classical planning where provable correct policies that generalize over all instances of a given domain have been learned using combinatorial methods. The aim of this work is to bring these two research threads together to illuminate the conditions under which (deep) reinforcement learning approaches, and in particular, policy optimization methods, can be used to learn policies that generalize like combinatorial methods do. We draw on lessons learned from previous combinatorial and deep learning approaches, and extend them in a convenient way. From the former, we model policies as state transition classifiers, as (ground) actions are not general and change from instance to instance. From the latter, we use graph neural networks (GNNs) adapted to deal with relational structures for representing value functions over planning states, and in our case, policies. With these ingredients in place, we find that actor-critic methods can be used to learn policies that generalize almost as well as those obtained using combinatorial approaches while avoiding the scalability bottleneck and the use of feature pools. Moreover, the limitations of the DRL methods on the benchmarks considered have little to do with deep learning or reinforcement learning algorithms, and result from the well-understood expressive limitations of GNNs, and the tradeoff between optimality and generalization (general policies cannot be optimal in some domains). Both of these limitations are addressed without changing the basic DRL methods by adding derived predicates and an alternative cost structure to optimize.

---

## 82. 论文ID: 2512.19332v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19332v1.json'

---

## 83. 论文ID: 2512.19182v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19182v1.json'

---

## 84. Scaling up Stability: Reinforcement Learning for Distributed Control of Networked Systems in the Space of Stabilizing Policies

**论文链接:** [http://arxiv.org/abs/2512.18540v1](http://arxiv.org/abs/2512.18540v1)

**作者:** John Cao, Luca Furieri

**发布时间:** 2025-12-20

### GPT解析

### 总结

本研究提出了一种基于强化学习的网络化系统分布式控制方法，通过结合图神经网络和类似Youla的参数化方法，实现了可扩展、表达能力强且稳定的控制策略。

### 背景

在网络化系统中进行分布式控制时，神经策略需要同时满足可扩展性、表达能力和稳定性，这是一个具有挑战性的问题。

### 目的

设计一种能够同时满足可扩展性、表达能力和稳定性的分布式控制策略，并通过强化学习方法进行训练，确保网络级闭环稳定性。

### 方法

引入一种将图神经网络嵌入到类似Youla的幅值-方向参数化中的策略参数化方法，其中幅值部分由GNN作用于扰动反馈实现为稳定算子，方向部分由GNN作用于局部观测，并与近端策略优化集成。

### 主要发现

训练在小网络上的策略可以直接扩展到更大的网络和未见过的网络拓扑；与最先进的MARL基线相比，所提策略实现了更高的回报和更低的方差，同时保持稳定性。

### 结论

所提出的策略参数化方法有效地解决了网络化系统中的分布式控制问题，具有良好的泛化能力和鲁棒性，能够在保持稳定性的同时实现高性能。

### 翻译

我们通过强化学习研究网络化系统的分布式控制，其中神经策略必须同时具备可扩展性、表达能力和稳定性。我们引入了一种将图神经网络（GNNs）嵌入到类似Youla的幅值-方向参数化中的策略参数化方法，从而产生分布式随机控制器，通过设计保证网络级闭环稳定性。幅值部分实现为稳定算子，由GNN作用于扰动反馈，而方向部分为由GNN作用于局部观测。我们证明了闭环系统对图拓扑和模型参数扰动的鲁棒性，并展示了如何将我们的参数化方法与近端策略优化集成。在多智能体导航任务上的实验表明，训练在小网络上的策略可以直接扩展到更大的网络和未见过的网络拓扑，与最先进的MARL基线相比，实现了更高的回报和更低的方差，同时保持了稳定性。


### 论文摘要

We study distributed control of networked systems through reinforcement learning, where neural policies must be simultaneously scalable, expressive and stabilizing. We introduce a policy parameterization that embeds Graph Neural Networks (GNNs) into a Youla-like magnitude-direction parameterization, yielding distributed stochastic controllers that guarantee network-level closed-loop stability by design. The magnitude is implemented as a stable operator consisting of a GNN acting on disturbance feedback, while the direction is a GNN acting on local observations. We prove robustness of the closed loop to perturbations in both the graph topology and model parameters, and show how to integrate our parameterization with Proximal Policy Optimization. Experiments on a multi-agent navigation task show that policies trained on small networks transfer directly to larger ones and unseen network topologies, achieve higher returns and lower variance than a state-of-the-art MARL baseline while preserving stability.

---

## 85. Feature-Enhanced Graph Neural Networks for Classification of Synthetic Graph Generative Models: A Benchmarking Study

**论文链接:** [http://arxiv.org/abs/2512.18524v1](http://arxiv.org/abs/2512.18524v1)

**作者:** Janek Dyer, Jagdeep Ahluwalia, Javad Zarrin

**发布时间:** 2025-12-20

**备注:** This is a preprint version of a manuscript currently under review at The Journal of Supercomputing (Springer)

### GPT解析

### 总结

本研究探讨了结合图神经网络(GNNs)和可解释图论特征进行合成图分类的混合方法，发现GraphSAGE和GTN架构表现最佳，达到98.5%的分类准确率。

### 背景

区分生成图模型对于理解合成图及其模拟的现实世界复杂结构模式至关重要。虽然图神经网络在图分类任务中效果显著，但很少有研究探索它们与可解释图论特征的结合。

### 目的

研究一种混合方法，将图神经网络与工程化的图论特征相结合，用于合成图家族的分类任务。

### 方法

生成包含五个代表性生成家族(Erdos-Renyi、Watts-Strogatz、Barab'asi-Albert、Holme-Kim和Stochastic Block Model)的大型结构多样合成数据集；提取并修剪节点和图级别特征；将特征整合到六种GNN架构(GCN、GAT、GATv2、GIN、GraphSAGE和GTN)中；使用Optuna进行超参数优化；与基于手工特征的支持向量机基线模型进行比较。

### 主要发现

GraphSAGE和GTN实现了最高的分类性能，准确率达98.5%，t-SNE和UMAP可视化显示良好的类别分离；GCN和GIN表现良好；基于GAT的模型表现较差，因其捕获全局结构能力有限；SVM基线证实了消息传递功能对性能提升和有意义类别分离的重要性。

### 结论

结合图神经网络和工程化图论特征的混合方法在合成图分类任务中表现优异，特别是GraphSAGE和GTN架构。

### 翻译

区分生成图模型的能力对于理解合成图以及它们所模拟的现实世界复杂结构模式至关重要。虽然图神经网络(GNNs)在图分类任务中已见广泛应用并取得显著效果，但很少有研究探索它们与可解释图论特征的结合。本文研究了使用混合方法分类合成图家族，该方法结合了GNNs和工程化的图论特征。我们生成了一个大型结构多样的合成数据集，包含五个代表性生成家族的图：Erdos-Renyi、Watts-Strogatz、Barab'asi-Albert、Holme-Kim和Stochastic Block Model。这些图的大小最多达到1x10^4个节点，包含最多1.1x10^5条边。为每个图提取了全面的节点和图级别特征，并使用基于随机森林的特征选择流程进行了修剪。将这些特征整合到六种GNN架构中：GCN、GAT、GATv2、GIN、GraphSAGE和GTN。使用Optuna进行超参数选择优化。最后，将模型与仅基于手工制作特征训练的基线支持向量机(SVM)进行比较。我们的评估表明，GraphSAGE和GTN实现了最高的分类性能，准确率达98.5%，t-SNE和UMAP可视化显示了良好的类别分离。GCN和GIN也表现良好，而基于GAT的模型表现滞后，因为它们捕获全局结构的能力有限。SVM基线确认了消息传递功能对性能提升和有意义类别分离的重要性。


### 论文摘要

The ability to discriminate between generative graph models is critical to understanding complex structural patterns in both synthetic graphs and the real-world structures that they emulate. While Graph Neural Networks (GNNs) have seen increasing use to great effect in graph classification tasks, few studies explore their integration with interpretable graph theoretic features. This paper investigates the classification of synthetic graph families using a hybrid approach that combines GNNs with engineered graph-theoretic features. We generate a large and structurally diverse synthetic dataset comprising graphs from five representative generative families, Erdos-Renyi, Watts-Strogatz, Barab'asi-Albert, Holme-Kim, and Stochastic Block Model. These graphs range in size up to 1x10^4 nodes, containing up to 1.1x10^5 edges. A comprehensive range of node and graph level features is extracted for each graph and pruned using a Random Forest based feature selection pipeline. The features are integrated into six GNN architectures: GCN, GAT, GATv2, GIN, GraphSAGE and GTN. Each architecture is optimised for hyperparameter selection using Optuna. Finally, models were compared against a baseline Support Vector Machine (SVM) trained solely on the handcrafted features. Our evaluation demonstrates that GraphSAGE and GTN achieve the highest classification performance, with 98.5% accuracy, and strong class separation evidenced by t-SNE and UMAP visualisations. GCN and GIN also performed well, while GAT-based models lagged due to limitations in their ability to capture global structures. The SVM baseline confirmed the importance of the message passing functionality for performance gains and meaningful class separation.

---

## 86. APC-GNN++: An Adaptive Patient-Centric GNN with Context-Aware Attention and Mini-Graph Explainability for Diabetes Classification

**论文链接:** [http://arxiv.org/abs/2512.18473v1](http://arxiv.org/abs/2512.18473v1)

**作者:** Khaled Berkani

**发布时间:** 2025-12-20

**备注:** 17 pages, 2 figures, 5 tables

### GPT解析

### 总结

提出APC-GNN++，一种自适应以患者为中心的图神经网络，用于糖尿病分类，通过整合上下文感知边注意力、置信度引导的特征混合和邻域一致性正则化来捕捉患者间的临床关系。

### 背景

糖尿病分类需要捕捉患者间的临床关系，传统方法难以有效处理。

### 目的

开发一种能够处理未见患者并提供可解释预测的图神经网络模型，用于糖尿病分类。

### 方法

设计APC-GNN++模型，集成上下文感知边注意力、置信度引导的节点特征和图表示混合、邻域一致性正则化；引入小图方法处理新患者；在阿尔及利亚地区医院收集的真实数据集上评估性能。

### 主要发现

APC-GNN++优于传统机器学习模型和普通GCN，获得更高的测试准确率和宏F1分数；节点级置信度分析显示模型在不同患者群体中平衡自我信息和基于图的证据。

### 结论

APC-GNN++能有效捕捉患者间的临床关系，提供实时可解释预测，并已嵌入GUI供医疗专业人员使用。

### 翻译

我们提出APC-GNN++，一种自适应以患者为中心的图神经网络，用于糖尿病分类。我们的模型整合了上下文感知的边注意力、置信度引导的节点特征和图表示混合，以及邻域一致性正则化，以更好地捕捉患者之间的临床有意义的关系。为处理未见患者，我们引入了一种小图方法，利用新患者的最近邻，无需重新训练全局模型即可实现实时可解释预测。我们在阿尔及利亚地区医院收集的真实世界糖尿病数据集上评估了APC-GNN++，结果表明它优于传统机器学习模型（MLP、随机森林、XGBoost）和普通GCN，获得了更高的测试准确率和宏F1分数。节点级置信度分数的分析进一步揭示了模型如何在不同患者群体中平衡自我信息和基于图的证据，提供了可解释的以患者为中心的见解。该系统还嵌入了一个基于Tkinter的图形用户界面（GUI），供医疗专业人员交互使用。


### 论文摘要

We propose APC-GNN++, an adaptive patient-centric Graph Neural Network for diabetes classification. Our model integrates context-aware edge attention, confidence-guided blending of node features and graph representations, and neighborhood consistency regularization to better capture clinically meaningful relationships between patients. To handle unseen patients, we introduce a mini-graph approach that leverages the nearest neighbors of the new patient, enabling real-time explainable predictions without retraining the global model. We evaluate APC-GNN++ on a real-world diabetes dataset collected from a regional hospital in Algeria and show that it outperforms traditional machine learning models (MLP, Random Forest, XGBoost) and a vanilla GCN, achieving higher test accuracy and macro F1- score. The analysis of node-level confidence scores further reveals how the model balances self-information and graph-based evidence across different patient groups, providing interpretable patient-centric insights. The system is also embedded in a Tkinter-based graphical user interface (GUI) for interactive use by healthcare professionals .

---

## 87. A Distributed Hierarchical Spatio-Temporal Edge-Enhanced Graph Neural Network for City-Scale Dynamic Logistics Routing

**论文链接:** [http://arxiv.org/abs/2512.18441v1](http://arxiv.org/abs/2512.18441v1)

**作者:** Zihan Han, Lingran Meng, Jingwei Zhang

**发布时间:** 2025-12-20

### GPT解析

### 总结

本文提出了一种分布式分层时空边缘增强图神经网络（HSTE-GNN），用于解决超大型城市道路网络中的动态物流路由问题，通过区域并行处理和跨区域协调，显著提高了路由效率和实时适应性。

### 背景

随着都市道路网络增长到数千万条边，在高流动性需求下交通状况快速变化，传统集中式路由算法和整体图神经网络模型在可扩展性、延迟和实时适应性方面存在局限，难以有效处理大型城市物流系统。

### 目的

提出一种分布式分层时空边缘增强图神经网络（HSTE-GNN），用于超大型道路网络的动态路由，解决传统方法的可扩展性、延迟和实时适应性问题。

### 方法

将城市规模图划分为区域子图在分布式计算节点上并行处理；在每个区域内使用边缘增强时空模块建模节点状态、动态边属性和短期时间依赖；通过分层协调层和异步参数服务器机制聚合跨区域表示，确保高频交通更新下的全局路由一致性。

### 主要发现

分布式分层设计平衡了局部响应性和全局一致性；在北京和纽约的真实大规模交通数据集上，HSTE-GNN优于ST-GRAPH等基线模型，实现了34.9%更低的路由延迟，14.7%更低的MAPE，11.8%更低的RMSE，全局路由一致性提高了7.3%。

### 结论

所提出的HSTE-GNN框架为下一代智能交通系统和大规模物流平台提供了可扩展、自适应且高效的解决方案。

### 翻译

城市规模的物流路由随着都市道路网络增长到数千万条边以及在高流动性需求下交通状况快速变化而变得越来越具有挑战性。传统的集中式路由算法和整体图神经网络模型在可扩展性、高延迟和实时适应性方面存在局限，这限制了它们在大型城市物流系统中的有效性。为解决这些挑战，本文提出了一种用于超大型道路网络动态路由的分布式分层时空边缘增强图神经网络（HSTE-GNN）。该框架将城市规模图划分为区域子图，在分布式计算节点上并行处理，实现局部交通动力学的高效学习。在每个区域内，边缘增强时空模块联合建模节点状态、动态边属性和短期时间依赖。分层协调层通过异步参数服务器机制进一步聚合跨区域表示，确保在高频交通更新下的全局路由一致性。这种分布式分层设计平衡了局部响应性与全局一致性，显著提高了可扩展性和推理效率。在北京和纽约真实世界大规模交通数据集上的实验表明，HSTE-GNN优于ST-GRAPH等强时空基线模型，实现了34.9%更低的路由延迟，14.7%更低的MAPE，11.8%更低的RMSE，同时将全局路由一致性提高了7.3%。这些结果证实，所提出的框架为下一代智能交通系统和大规模物流平台提供了可扩展、自适应且高效的解决方案。


### 论文摘要

City-scale logistics routing has become increasingly challenging as metropolitan road networks grow to tens of millions of edges and traffic conditions evolve rapidly under high-volume mobility demands. Conventional centralized routing algorithms and monolithic graph neural network (GNN) models suffer from limited scalability, high latency, and poor real-time adaptability, which restricts their effectiveness in large urban logistics systems. To address these challenges, this paper proposes a Distributed Hierarchical Spatio-Temporal Edge-Enhanced Graph Neural Network (HSTE-GNN) for dynamic routing over ultra-large road networks. The framework partitions the city-scale graph into regional subgraphs processed in parallel across distributed computing nodes, enabling efficient learning of localized traffic dynamics. Within each region, an edge-enhanced spatio-temporal module jointly models node states, dynamic edge attributes, and short-term temporal dependencies. A hierarchical coordination layer further aggregates cross-region representations through an asynchronous parameter-server mechanism, ensuring global routing coherence under high-frequency traffic updates. This distributed hierarchical design balances local responsiveness with global consistency, significantly improving scalability and inference efficiency. Experiments on real-world large-scale traffic datasets from Beijing and New York demonstrate that HSTE-GNN outperforms strong spatio-temporal baselines such as ST-GRAPH, achieving 34.9% lower routing delay, 14.7% lower MAPE, and 11.8% lower RMSE, while improving global route consistency by 7.3%. These results confirm that the proposed framework provides a scalable, adaptive, and efficient solution for next-generation intelligent transportation systems and large-scale logistics platforms.

---

## 88. Through the PRISm: Importance-Aware Scene Graphs for Image Retrieval

**论文链接:** [http://arxiv.org/abs/2512.18407v1](http://arxiv.org/abs/2512.18407v1)

**作者:** Dimitrios Georgoulopoulos, Nikolaos Chaidos, Angeliki Dimitriou, Giorgos Stamou

**发布时间:** 2025-12-20

**备注:** 10 pages, 5 figures

### GPT解析

### 总结

PRISm是一个基于语义图重要性预测的剪枝图像检索多模态框架，通过两个新组件改进图像到图像的检索：重要性预测模块和边缘感知图神经网络。该模型能够明确建模对象及其交互的语义重要性，实现与人类感知一致的图像检索。

### 背景

在计算机视觉中，准确检索语义相似的图像仍然是一个基本挑战，因为传统方法往往无法捕捉场景的关系和上下文细微差别。

### 目的

开发一种能够更好地捕捉图像中对象及其关系语义重要性的图像检索方法，使检索结果更接近人类感知。

### 方法

重要性预测模块识别并保留图像中最关键的对象和关系三元组，同时修剪无关元素；边缘感知图神经网络明确编码关系结构并整合全局视觉特征，以生成语义感知的图像嵌入；结合关系推理与视觉表示，实现语义基础的检索。

### 主要发现

在基准和真实世界数据集上的广泛实验展示了持续优越的顶级排名性能；定性分析表明PRISm准确捕捉了关键对象和交互，产生了可解释和语义上有意义的结果。

### 结论

PRISm通过明确建模对象及其交互的语义重要性，实现了与人类感知紧密对齐的图像检索，其架构有效地结合了关系推理与视觉表示，实现了语义基础的检索。

### 翻译

准确检索语义相似的图像在计算机视觉中仍然是一个基本挑战，因为传统方法往往无法捕捉场景的关系和上下文细微差别。我们介绍了PRISm（基于语义图重要性预测的剪枝图像检索），这是一个多模态框架，通过两个新颖组件推进图像到图像的检索。首先，重要性预测模块识别并保留图像中最关键的对象和关系三元组，同时修剪无关元素。其次，边缘感知图神经网络明确编码关系结构并整合全局视觉特征，以生成语义感知的图像嵌入。PRISm通过明确建模对象及其交互的语义重要性，实现了与人类感知紧密对齐的图像检索，这是先前方法中 largely 缺乏的能力。其架构有效地结合了关系推理与视觉表示，实现了语义基础的检索。在基准和真实世界数据集上的广泛实验展示了持续优越的顶级排名性能，同时定性分析表明PRISm准确捕捉了关键对象和交互，产生了可解释和语义上有意义的结果。


### 论文摘要

Accurately retrieving images that are semantically similar remains a fundamental challenge in computer vision, as traditional methods often fail to capture the relational and contextual nuances of a scene. We introduce PRISm (Pruning-based Image Retrieval via Importance Prediction on Semantic Graphs), a multimodal framework that advances image-to-image retrieval through two novel components. First, the Importance Prediction Module identifies and retains the most critical objects and relational triplets within an image while pruning irrelevant elements. Second, the Edge-Aware Graph Neural Network explicitly encodes relational structure and integrates global visual features to produce semantically informed image embeddings. PRISm achieves image retrieval that closely aligns with human perception by explicitly modeling the semantic importance of objects and their interactions, capabilities largely absent in prior approaches. Its architecture effectively combines relational reasoning with visual representation, enabling semantically grounded retrieval. Extensive experiments on benchmark and real-world datasets demonstrate consistently superior top-ranked performance, while qualitative analyses show that PRISm accurately captures key objects and interactions, producing interpretable and semantically meaningful results.

---

## 89. AL-GNN: Privacy-Preserving and Replay-Free Continual Graph Learning via Analytic Learning

**论文链接:** [http://arxiv.org/abs/2512.18295v1](http://arxiv.org/abs/2512.18295v1)

**作者:** Xuling Zhang, Jindong Li, Yifei Zhang, Menglin Yang

**发布时间:** 2025-12-20

### GPT解析

### 总结

本文提出了一种名为AL GNN的新型持续图学习框架，通过解析学习理论原理，将学习转化为递归最小二乘优化过程，无需反向传播和回放缓冲区，有效解决了现有方法的隐私和效率问题。

### 背景

持续图学习旨在使图神经网络能够从流式图数据中增量学习而不遗忘先前知识。现有基于经验回放的方法需要存储和重新访问过去的图数据，但存在隐私问题和效率低下等显著局限性。

### 目的

开发一种新的持续图学习框架，消除对反向传播和回放缓冲区的需求，同时解决隐私问题和效率问题。

### 方法

AL GNN框架利用解析学习理论原理，将学习公式化为递归最小二乘优化过程。通过封闭形式的分类器更新和正则化特征自相关矩阵来分析和更新模型知识，实现每个任务的高效单次训练，并避免存储历史样本以保护数据隐私。

### 主要发现

在多个动态图分类基准上，AL GNN与现有方法相比具有竞争力或更优的性能：在CoraFull上平均性能提高10%，在Reddit上减少超过30%的遗忘，同时由于无反向传播设计，训练时间减少近50%。

### 结论

AL GNN是一种有效的持续图学习框架，能够解决现有方法的局限性，同时保持或提高性能，并在保护数据隐私方面具有优势。

### 翻译

持续图学习(CGL)旨在使图神经网络能够从流式图结构数据中增量学习，同时不忘记之前获得的知识。现有方法，特别是基于经验回放的方法，通常存储和重新访问过去的图数据来减轻灾难性遗忘。然而，这些方法存在显著局限性，包括隐私问题和效率低下。在本工作中，我们提出了AL GNN，一种用于持续图学习的新型框架，消除了对反向传播和回放缓冲区的需求。相反，AL GNN利用解析学习理论原理，将学习公式化为递归最小二乘优化过程。它通过封闭形式的分类器更新和正则化特征自相关矩阵来分析和更新模型知识。这种设计使得每个任务只需高效的单次训练，并通过避免存储历史样本来本质上保护数据隐私。在多个动态图分类基准上的广泛实验表明，与现有方法相比，AL GNN实现了竞争性或更优的性能。例如，它在CoraFull上平均性能提高了10%，在Reddit上减少了超过30%的遗忘，同时由于其无反向传播的设计，训练时间减少了近50%。


### 论文摘要

Continual graph learning (CGL) aims to enable graph neural networks to incrementally learn from a stream of graph structured data without forgetting previously acquired knowledge. Existing methods particularly those based on experience replay typically store and revisit past graph data to mitigate catastrophic forgetting. However, these approaches pose significant limitations, including privacy concerns, inefficiency. In this work, we propose AL GNN, a novel framework for continual graph learning that eliminates the need for backpropagation and replay buffers. Instead, AL GNN leverages principles from analytic learning theory to formulate learning as a recursive least squares optimization process. It maintains and updates model knowledge analytically through closed form classifier updates and a regularized feature autocorrelation matrix. This design enables efficient one pass training for each task, and inherently preserves data privacy by avoiding historical sample storage. Extensive experiments on multiple dynamic graph classification benchmarks demonstrate that AL GNN achieves competitive or superior performance compared to existing methods. For instance, it improves average performance by 10% on CoraFull and reduces forgetting by over 30% on Reddit, while also reducing training time by nearly 50% due to its backpropagation free design.

---

## 90. AutoSchA: Automatic Hierarchical Music Representations via Multi-Relational Node Isolation

**论文链接:** [http://arxiv.org/abs/2512.18232v1](http://arxiv.org/abs/2512.18232v1)

**作者:** Stephen Ni-Hahn, Rico Zhu, Jerry Yin, Yue Jiang, Cynthia Rudin, Simon Mak

**发布时间:** 2025-12-20

### GPT解析

### 总结

本文提出了一种名为AutoSchA的新方法，利用图神经网络实现自动分层音乐分析，该方法在分析巴洛克赋格主题时表现与人类专家相当。

### 背景

分层表示为分析多种音乐流派提供了强大且系统的方法，如Schenkerian分析。然而，分层音乐分析成本高昂，需要专家投入大量时间和精力，且将其表示为计算机可读格式存在挑战。

### 目的

利用分层深度学习和计算机可读数据量增加的最新进展，建立一个自动的分层音乐表示框架。

### 方法

提出AutoSchA方法，扩展了图神经网络在分层音乐分析中的应用。该方法包含三个关键贡献：新的分层音乐表示图学习框架；基于节点隔离的新图池化机制；以及集成这些发展的最先进架构。

### 主要发现

在一系列实验中，AutoSchA在分析巴洛克赋格主题时表现可与人类专家相媲美。

### 结论

AutoSchA是一种有前景的自动分层音乐分析方法，能够达到专家级别的性能。

### 翻译

分层表示为分析许多音乐流派提供了强大而系统的方法。这种表示方法已经在音乐理论中得到广泛研究，例如通过Schenkerian分析(SchA)。然而，分层音乐分析成本很高；分析一首音乐需要专家投入大量的时间和精力。将分层分析表示为计算机可读格式是另一个挑战。鉴于最近在分层深度学习和计算机可读数据量增加方面的进展，将此类工作扩展到自动分层表示框架非常有前景。因此，本文引入了一种新方法AutoSchA，它扩展了图神经网络(GNNs)在分层音乐分析中的最新进展。AutoSchA具有三个关键贡献：1)一种用于分层音乐表示的新图学习框架；2)一种基于节点隔离的新图池化机制，直接优化学习的池化分配；3)一种集成这些发展用于自动分层音乐分析的最先进架构。我们在一系列实验中表明，当分析巴洛克赋格主题时，AutoSchA的表现可与人类专家相媲美。


### 论文摘要

Hierarchical representations provide powerful and principled approaches for analyzing many musical genres. Such representations have been broadly studied in music theory, for instance via Schenkerian analysis (SchA). Hierarchical music analyses, however, are highly cost-intensive; the analysis of a single piece of music requires a great deal of time and effort from trained experts. The representation of hierarchical analyses in a computer-readable format is a further challenge. Given recent developments in hierarchical deep learning and increasing quantities of computer-readable data, there is great promise in extending such work for an automatic hierarchical representation framework. This paper thus introduces a novel approach, AutoSchA, which extends recent developments in graph neural networks (GNNs) for hierarchical music analysis. AutoSchA features three key contributions: 1) a new graph learning framework for hierarchical music representation, 2) a new graph pooling mechanism based on node isolation that directly optimizes learned pooling assignments, and 3) a state-of-the-art architecture that integrates such developments for automatic hierarchical music analysis. We show, in a suite of experiments, that AutoSchA performs comparably to human experts when analyzing Baroque fugue subjects.

---

## 91. Toward Efficient Testing of Graph Neural Networks via Test Input Prioritization

**论文链接:** [http://arxiv.org/abs/2512.18228v1](http://arxiv.org/abs/2512.18228v1)

**作者:** Lichen Yang, Qiang Wang, Zhonghao Yang, Daojing He, Yu Li

**发布时间:** 2025-12-20

**DOI:** 10.1007/s10515-025-00554-0

**备注:** This is the author-accepted manuscript of a paper published in Automated Software Engineering Journal

### GPT解析

### 总结

本文提出了一种名为GraphRank的新型测试输入优先排序框架，用于图神经网络(GNNs)的测试，通过结合模型感知属性和模型无关属性，并利用图结构信息提高测试效率。

### 背景

图神经网络在处理图结构数据方面表现出色，但部署后会出现故障，可能导致严重后果。全面测试需要大量手动标注数据，增加了成本。

### 目的

降低标注成本，有策略地优先选择高质量无标签输入进行测试，在有限预算下发现更多模型故障。

### 方法

GraphRank框架引入模型无关属性弥补模型感知属性的局限性，利用图结构信息聚合相邻节点属性增强特征，结合二元分类器作为排序模型，并通过迭代训练提高性能。

### 主要发现

大量实验证明GraphRank在测试输入优先排序方面优于现有技术。

### 结论

GraphRank有效解决了现有测试输入优先排序技术的局限性，提高了GNNs测试的效率和可靠性。

### 翻译

图神经网络(GNNs)在处理图结构数据方面表现出色；然而，它们在部署后会出现故障，可能导致严重后果。因此，部署前进行全面测试对于确保GNNs的可靠性变得至关重要。然而，全面测试需要大量手动标注的测试数据。为了降低标注成本，有策略地优先选择和标注高质量的无标签输入用于测试变得至关重要，这有助于在有限的标注预算下发现更多的模型故障。不幸的是，现有的测试输入优先排序技术要么忽略了图结构中包含的有价值信息，要么过度依赖于从目标模型提取的属性（即模型感知属性），而这些属性的质量可能差异很大。为了解决这些问题，我们提出了一个名为GraphRank的新型测试输入优先排序框架，专门用于GNNs。GraphRank引入了模型无关属性来弥补模型感知属性的局限性。它还利用图结构信息来聚合来自相邻节点的属性，从而增强模型感知和模型无关属性。此外，GraphRank将上述属性与二元分类器结合，使用它作为排序模型来优先处理输入。该分类器经过迭代训练，使其能够从每轮的反馈中学习并相应地提高其性能。大量实验证明了GraphRank相对于现有技术的优越性。


### 论文摘要

Graph Neural Networks (GNNs) have demonstrated remarkable efficacy in handling graph-structured data; however, they exhibit failures after deployment, which can cause severe consequences. Hence, conducting thorough testing before deployment becomes imperative to ensure the reliability of GNNs. However, thorough testing requires numerous manually annotated test data. To mitigate the annotation cost, strategically prioritizing and labeling high-quality unlabeled inputs for testing becomes crucial, which facilitates uncovering more model failures with a limited labeling budget. Unfortunately, existing test input prioritization techniques either overlook the valuable information contained in graph structures or are overly reliant on attributes extracted from the target model, i.e., model-aware attributes, whose quality can vary significantly. To address these issues, we propose a novel test input prioritization framework, named GraphRank, for GNNs. GraphRank introduces model-agnostic attributes to compensate for the limitations of the model-aware ones. It also leverages the graph structure information to aggregate attributes from neighboring nodes, thereby enhancing the model-aware and model-agnostic attributes. Furthermore, GraphRank combines the above attributes with a binary classifier, using it as a ranking model to prioritize inputs. This classifier undergoes iterative training, which enables it to learn from each round's feedback and improve its performance accordingly. Extensive experiments demonstrate GraphRank's superiority over existing techniques.

---

## 92. PROVEX: Enhancing SOC Analyst Trust with Explainable Provenance-Based IDS

**论文链接:** [http://arxiv.org/abs/2512.18199v1](http://arxiv.org/abs/2512.18199v1)

**作者:** Devang Dhanuka, Nidhi Rastogi

**发布时间:** 2025-12-20

### GPT解析

### 总结

该研究提出了一种全面的XAI框架，用于增强基于图神经网络的入侵检测系统的可解释性，使安全分析师能够理解系统为何发出警报，从而提高信任度和分类速度。

### 背景

现代入侵检测系统利用图神经网络检测系统来源数据中的恶意活动，但这些系统的决策过程对分析师来说往往是一个黑箱，缺乏透明度。

### 目的

开发一个可解释人工智能框架，通过使基于图的检测透明化，弥合安全运营中心中的信任差距，使分析师能够理解模型决策的原因。

### 方法

在KAIROS（一种先进的基于时间图的入侵检测系统）上实现了该框架，并集成了三种GNN解释方法（GraphMask、GNNExplainer和VA-TGExplainer），使其适用于时间来源上下文，通过事后解释突出显示触发警报的关键因果子图和事件。

### 主要发现

该框架能够生成人类可解释的异常行为表示，包括重要边和不确定性估计；在DARPA CADETS数据集上测试显示，解释器以高保真度保留了原始模型的决策，突出了恶意文件交互和异常网络流量等关键特征；平均解释开销为每个事件3-5秒。

### 结论

通过提供模型推理的洞察，该框架有效提高了分析师对入侵检测系统的信任度和事件分类速度，同时保持了检测性能。

### 翻译

现代入侵检测系统利用图神经网络来检测系统来源数据中的恶意活动，但它们的决策对分析师来说往往是一个黑箱。本文提出了一个全面的XAI框架，旨在通过使基于图的检测透明化，弥合安全运营中心中的信任差距。我们在KAIROS（一种最先进的基于时间图的IDS）上实现了这个框架，尽管我们的设计可以以最小的适应应用于任何基于时间图的检测器。完整代码库可在https://github.com/devang1304/provex.git获取。我们通过事后解释增强了检测管道，突出了触发警报的原因，识别关键的因果子图和事件。我们使三种GNN解释方法适应时间来源上下文：GraphMask、GNNExplainer和变分时间GNN解释器（VA-TGExplainer）。这些工具输出异常行为的人类可解释表示，包括重要的边和不确定性估计。我们的贡献专注于这些解释器的实际集成，解决了内存管理和可重复性方面的挑战。我们在DARPA CADETS Engagement 3数据集上展示了我们的框架，并证明它为检测到的攻击生成简洁的窗口级解释。我们的评估揭示了这些解释器以高保真度保留了TGNN的决策，突出了关键的边，如恶意文件交互和异常网络流量。平均解释开销为每个事件3-5秒。通过提供模型推理的洞察，我们的框架旨在提高分析师的信任度和分类速度。


### 论文摘要

Modern intrusion detection systems (IDS) leverage graph neural networks (GNNs) to detect malicious activity in system provenance data, but their decisions often remain a black box to analysts. This paper presents a comprehensive XAI framework designed to bridge the trust gap in Security Operations Centers (SOCs) by making graph-based detection transparent. We implement this framework on top of KAIROS, a state-of-the-art temporal graph-based IDS, though our design is applicable to any temporal graph-based detector with minimal adaptation. The complete codebase is available at https://github.com/devang1304/provex.git. We augment the detection pipeline with post-hoc explanations that highlight why an alert was triggered, identifying key causal subgraphs and events. We adapt three GNN explanation methods - GraphMask, GNNExplainer, and a variational temporal GNN explainer (VA-TGExplainer) - to the temporal provenance context. These tools output human-interpretable representations of anomalous behavior, including important edges and uncertainty estimates. Our contributions focus on the practical integration of these explainers, addressing challenges in memory management and reproducibility. We demonstrate our framework on the DARPA CADETS Engagement 3 dataset and show that it produces concise window-level explanations for detected attacks. Our evaluation reveals that the explainers preserve the TGNN's decisions with high fidelity, surfacing critical edges such as malicious file interactions and anomalous netflows. The average explanation overhead is 3-5 seconds per event. By providing insight into the model's reasoning, our framework aims to improve analyst trust and triage speed.

---

## 93. On Swarm Leader Identification using Probing Policies

**论文链接:** [http://arxiv.org/abs/2512.18146v1](http://arxiv.org/abs/2512.18146v1)

**作者:** Stergios E. Bachoumas, Panagiotis Artemiadis

**发布时间:** 2025-12-20

**备注:** 13 pages, journal

### GPT解析

### 总结

该研究提出了一种交互式群体领导者识别（iSLI）方法，通过对抗性探测代理与机器人群体成员的物理交互来识别领导者。研究将问题建模为POMDP，并采用深度强化学习训练探测策略，创新的神经网络架构结合了时间图关系层（TGR）和简化结构状态空间序列（S5）模型，在模拟和真实机器人实验中均表现出色。

### 背景

在机器人群体中识别领导者至关重要，特别是在对抗环境中，领导者隐藏对于任务成功是必要的。传统的领导者识别方法可能难以应对动态和对抗性的环境挑战。

### 目的

开发一种新颖的交互式群体领导者识别方法，通过对抗性探测代理与群体成员的物理交互来识别领导者，并验证该方法在不同条件下的有效性和泛化能力。

### 方法

将iSLI问题建模为部分可观察马尔可夫决策过程（POMDP），使用深度强化学习中的近端策略优化（PPO）算法训练探测代理，设计创新的神经网络架构结合时间图关系层（TGR）和简化结构状态空间序列（S5）模型，TGR层处理基于图的群体观测并捕获时间依赖性，进行广泛的模拟实验和真实机器人实验验证。

### 主要发现

基于TGR的模型优于基准图神经网络架构，展现出显著的零样本泛化能力；训练后的探测代理能高精度识别领导者，即使在训练分布之外的场景中也能保持性能；预测表现出适当的置信度；真实机器人实验证实了模拟到现实的迁移能力；方法对动态变化（如意外代理断开连接）具有鲁棒性。

### 结论

所提出的iSLI方法结合创新的神经网络架构，能够在对抗环境中有效识别机器人群体中的领导者，具有良好的泛化能力和鲁棒性，适用于真实世界的应用场景。

### 翻译

识别机器人群体中的领导者至关重要，特别是在需要隐藏领导者以确保任务成功的对抗环境中。这项工作引入了交互式群体领导者识别（iSLI）问题，这是一种新颖的方法，其中对抗性探测代理通过与群体成员的物理交互来识别群体的领导者。我们将iSLI问题表述为部分可观察马尔可夫决策过程（POMDP），并采用深度强化学习，特别是近端策略优化（PPO），来训练探测代理的策略。提出的方法利用了一种新颖的神经网络架构，具有时间图关系层（TGR）和简化结构状态空间序列（S5）模型。TGR层有效地处理基于图的群体观测，捕获时间依赖性并使用学习的门控机制融合关系信息，为策略学习生成信息丰富的表示。广泛的模拟表明，我们的基于TGR的模型优于基准图神经网络架构，并在不同于训练所使用的不同群体大小和速度上展现出显著的零样本泛化能力。训练后的探测代理在识别领导者方面具有高准确性，即使在训练分布之外的场景中也能保持性能，并在其预测中表现出适当的置信度水平。使用物理机器人的真实世界实验进一步验证了该方法，证实了成功的模拟到现实迁移以及对动态变化（如意外代理断开连接）的鲁棒性。


### 论文摘要

Identifying the leader within a robotic swarm is crucial, especially in adversarial contexts where leader concealment is necessary for mission success. This work introduces the interactive Swarm Leader Identification (iSLI) problem, a novel approach where an adversarial probing agent identifies a swarm's leader by physically interacting with its members. We formulate the iSLI problem as a Partially Observable Markov Decision Process (POMDP) and employ Deep Reinforcement Learning, specifically Proximal Policy Optimization (PPO), to train the prober's policy. The proposed approach utilizes a novel neural network architecture featuring a Timed Graph Relationformer (TGR) layer combined with a Simplified Structured State Space Sequence (S5) model. The TGR layer effectively processes graph-based observations of the swarm, capturing temporal dependencies and fusing relational information using a learned gating mechanism to generate informative representations for policy learning. Extensive simulations demonstrate that our TGR-based model outperforms baseline graph neural network architectures and exhibits significant zero-shot generalization capabilities across varying swarm sizes and speeds different from those used during training. The trained prober achieves high accuracy in identifying the leader, maintaining performance even in out-of-training distribution scenarios, and showing appropriate confidence levels in its predictions. Real-world experiments with physical robots further validate the approach, confirming successful sim-to-real transfer and robustness to dynamic changes, such as unexpected agent disconnections.

---

## 94. A Hybrid Inductive-Transductive Network for Traffic Flow Imputation on Unsampled Locations

**论文链接:** [http://arxiv.org/abs/2512.17984v1](http://arxiv.org/abs/2512.17984v1)

**作者:** Mohammadmahdi Rahimiasl, Ynte Vanderhoydonc, Siegfried Mercelis

**发布时间:** 2025-12-19

**备注:** 10 pages, 8 figures, 3 tables

### GPT解析

### 总结

本文提出了一种混合归纳-转导网络(HINT)和相应的训练策略，用于解决未监测位置的交通流量预测问题，通过结合归纳式流量预测与转导速度、交通模拟和外部地理空间信息，显著提高了预测准确性。

### 背景

准确预测未监测位置的交通流量存在挑战：环形检测器提供精确但稀疏的测量数据；探测车辆的速度数据广泛可用但与流量相关性弱；附近路段通常表现出强烈的流量规模异质性(如匝道与主线)，这打破了标准图神经网络(GNN)的假设。

### 目的

开发一种混合归纳-转导网络(HINT)和相应的INDU-TRANSDUCTIVE训练策略，将速度视为转导的、网络范围的信号，同时归纳地学习流量以推广到未见过位置，提高交通流量预测的准确性。

### 方法

HINT结合了三个组件：(i)归纳式空间变换器，学习相似度驱动的长程交互；(ii)基于FiLM条件化的扩散GCN，利用丰富的静态上下文；(iii)节点级校准层，纠正尺度偏差。训练使用掩码重建、节点采样、困难节点挖掘和噪声注入，图结构基于驾驶距离构建。

### 主要发现

在三个真实世界数据集(MOW、UTD19-Torino和UTD19-Essen)上，HINT始终优于最先进的归纳基线。相比KITS基线，HINT在MOW上将MAE减少了约42%(基本模拟)和50%(校准模拟)，在Torino上减少约22%，在Essen上减少约12%。即使没有模拟，HINT在MOW和Torino上仍表现更好，而在Essen上模拟至关重要。

### 结论

将归纳流量预测与转导速度、交通模拟和外部地理空间信息相结合，可以显著提高未监测位置交通流量预测的准确性，为解决交通监测数据稀疏性问题提供了有效方法。

### 翻译

准确预测未监测位置的交通流量很困难：环形检测器提供精确但稀疏的测量，探测车辆的速度数据广泛可用但与流量相关性弱，且附近路段通常表现出强烈的流量规模异质性(例如匝道与主线)，这打破了标准GNN的假设。我们提出了HINT，一种混合归纳-转导网络，以及一种INDU-TRANSDUCTIVE训练策略，将速度视为转导的、网络范围的信号，同时归纳地学习流量以推广到未见过位置。HINT结合了(i)一个归纳式空间变换器，从节点特征学习相似度驱动的长程交互；(ii)一个基于FiLM条件化的扩散GCN，利用丰富的静态上下文(从OSM派生的属性和交通模拟)；以及(iii)一个节点级校准层，纠正每段的尺度偏差。训练使用掩码重建，按节点采样，困难节点挖掘以强调困难传感器，并在可见流量上注入噪声以防止恒等映射，同时图结构基于驾驶距离构建。在三个真实世界数据集上，MOW(比利时安特卫普)、UTD19-Torino和UTD19-Essen，HINT始终超越最先进的归纳基线。相对于KITS，HINT在MOW上将MAE减少了约42%(基本模拟)和约50%(校准模拟)，在Torino上减少了约22%，在Essen上减少了约12%。即使没有模拟，HINT在MOW和Torino上仍然表现更好，而在Essen上模拟至关重要。这些结果表明，将归纳流量预测与转导速度、交通模拟和外部地理空间相结合可以提高上述任务的准确性。


### 论文摘要

Accurately imputing traffic flow at unsensed locations is difficult: loop detectors provide precise but sparse measurements, speed from probe vehicles is widely available yet only weakly correlated with flow, and nearby links often exhibit strong heterophily in the scale of traffic flow (e.g., ramps vs. mainline), which breaks standard GNN assumptions. We propose HINT, a Hybrid INductive-Transductive Network, and an INDU-TRANSDUCTIVE training strategy that treats speed as a transductive, network-wide signal while learning flow inductively to generalize to unseen locations. HINT couples (i) an inductive spatial transformer that learns similarity-driven, long-range interactions from node features with (ii) a diffusion GCN conditioned by FiLM on rich static context (OSM-derived attributes and traffic simulation), and (iii) a node-wise calibration layer that corrects scale biases per segment. Training uses masked reconstruction with epoch-wise node sampling, hard-node mining to emphasize difficult sensors, and noise injection on visible flows to prevent identity mapping, while graph structure is built from driving distances.   Across three real-world datasets, MOW (Antwerp, Belgium), UTD19-Torino, and UTD19-Essen, HINT consistently surpasses state-of-the-art inductive baselines. Relative to KITS, HINT reduces MAE on MOW by $\approx42$% with basic simulation and $\approx50$% with calibrated simulation; on Torino by $\approx22$%, and on Essen by $\approx12$%. Even without simulation, HINT remains superior on MOW and Torino, while simulation is crucial on Essen. These results show that combining inductive flow imputation with transductive speed, traffic simulations and external geospatial improves accuracy for the task described above.

---

## 95. Deep learning directed synthesis of fluid ferroelectric materials

**论文链接:** [http://arxiv.org/abs/2512.16671v2](http://arxiv.org/abs/2512.16671v2)

**作者:** Charles Parton-Barr, Stuart R. Berrow, Calum J. Gibb, Jordan Hobbs, Wanhe Jiang, Caitlin O'Brien, Will C. Ogle, Helen F. Gleeson, Richard J. Mandle

**发布时间:** 2025-12-18

**备注:** 104 pages, 76 figures

### GPT解析

### 总结

论文介绍了一种基于深度学习的从数据到分子的流水线，用于有目标地设计和合成新型有机流体铁电体，实现了对可合成流体铁电体的闭环发现方法。

### 背景

流体铁电体是一类新发现的液晶材料，具有可切换的长程极化有序性，在超快电光技术、响应性软物质和下一代能量材料方面有应用前景。然而，该领域的进展几乎完全依赖于直觉和偶然发现，限制了领域发展。

### 目的

开发并实验验证一种深度学习数据到分子流水线，实现对新型有机流体铁电体的有目标设计和合成。

### 方法

1. 收集所有已知纵向极化液晶材料数据集；2. 训练图神经网络预测铁电行为；3. 使用图变分自编码器生成全新分子结构；4. 通过分类器和回归器过滤候选物；5. 结合计算逆合成引擎和数字化化学库存缩小设计空间；6. 合成并表征11个候选物；7. 比较实验结果与神经网络预测。

### 主要发现

1. 开发了准确率高达95%的图神经网络预测铁电行为；2. 成功生成了具有预测铁电液晶行为的全新分子结构；3. 实验验证了新型材料的存在；4. 实验验证的材料增强了数据集质量，有助于未来研究。

### 结论

这些结果展示了一种实用的、闭环的发现可合成流体铁电体的方法，标志着向自主设计功能性软材料迈出了一步。

### 翻译

流体铁电体是一类最近发现的液晶，它们表现出可切换的长程极化有序性，为超快电光技术、响应性软物质和下一代能量材料提供了机会。然而，它们的发现几乎完全依赖于直觉和偶然，限制了该领域的进展。在这里，我们开发和实验验证了一种深度学习数据到分子的流水线，使新型有机流体铁电体的有目标设计和合成成为可能。我们整理了所有已知的纵向极化液晶材料的综合数据集，并训练了图神经网络，这些网络能够以高达95%的准确率预测铁电行为，并将转变温度的均方根误差低至11K。图变分自编码器生成全新的分子结构，使用高性能分类器和回归器组合进行过滤，以识别具有预测铁电液晶行为和可及转变温度的候选分子。与计算逆合成引擎和数字化化学库存的进一步整合将设计空间缩小到合成就绪的长名单。通过既定的基于混合物的外推方法合成并表征了11个候选物。从中，外推的铁电液晶转变与神经网络预测进行了比较。新型材料的实验验证通过质量反馈数据增强了原始数据集，从而有助于未来研究。


### 论文摘要

Fluid ferroelectrics, a recently discovered class of liquid crystals that exhibit switchable, long-range polar order, offer opportunities in ultrafast electro-optic technologies, responsive soft matter, and next-generation energy materials. Yet their discovery has relied almost entirely on intuition and chance, limiting progress in the field. Here we develop and experimentally validate a deep-learning data-to-molecule pipeline that enables the targeted design and synthesis of new organic fluid ferroelectrics. We curate a comprehensive dataset of all known longitudinally polar liquid-crystal materials and train graph neural networks that predict ferroelectric behaviour with up to 95% accuracy and achieve root mean square errors as low as 11 K for transition temperatures. A graph variational autoencoder generates de novo molecular structures which are filtered using an ensemble of high-performing classifiers and regressors to identify candidates with predicted ferroelectric nematic behaviour and accessible transition temperatures. Integration with a computational retrosynthesis engine and a digitised chemical inventory further narrows the design space to a synthesis-ready longlist. 11 candidates were synthesised and characterized through established mixture-based extrapolation methods. From which extrapolated ferroelectric nematic transitions were compared against neural network predictions. The experimental verification of novel materials augments the original dataset with quality feedback data thus aiding future research. These results demonstrate a practical, closed-loop approach to discovering synthesizable fluid ferroelectrics, marking a step toward autonomous design of functional soft materials.

---

## 96. Microsoft Academic Graph Information Retrieval for Research Recommendation and Assistance

**论文链接:** [http://arxiv.org/abs/2512.16661v2](http://arxiv.org/abs/2512.16661v2)

**作者:** Shikshya Shiwakoti, Samuel Goldsmith, Ujjwal Pandit

**发布时间:** 2025-12-18

**备注:** 5 pages, 3 figures

### GPT解析

### 总结

本文提出了一种基于注意力的子图检索器，这是一种GNN-as-retriever模型，应用基于注意力的剪枝技术提取精细的子图，然后将子图传递给大型语言模型进行高级知识推理。

### 背景

在当今信息驱动的世界，获取科学出版物变得越来越容易，但同时从大量可用研究中筛选出有价值的信息变得比以往更具挑战性。

### 目的

提出一种基于注意力的子图检索器模型，用于提高科学文献检索和知识推理的效率。

### 方法

提出一种GNN-as-retriever模型，应用基于注意力的剪枝技术提取精细的子图，然后将提取的子图传递给大型语言模型进行高级知识推理。

### 主要发现

图神经网络和图注意力机制在搜索大规模信息数据库方面显示出强大的有效性，当与现代大型语言模型结合时，效果更佳。

### 结论

该模型旨在通过结合GNN和大型语言模型的能力，提高科学文献检索和知识推理的效率。

### 翻译

在当今信息驱动的世界中，获取科学出版物变得越来越容易。与此同时，从大量可用研究中进行筛选比以往任何时候都更具挑战性。图神经网络和图注意力机制在搜索大规模信息数据库方面显示出强大的有效性，特别是与现代大型语言模型结合时。在本文中，我们提出了一种基于注意力的子图检索器，这是一种GNN-as-retriever模型，应用基于注意力的剪枝技术提取精细的子图，然后将其传递给大型语言模型进行高级知识推理。


### 论文摘要

In today's information-driven world, access to scientific publications has become increasingly easy. At the same time, filtering through the massive volume of available research has become more challenging than ever. Graph Neural Networks (GNNs) and graph attention mechanisms have shown strong effectiveness in searching large-scale information databases, particularly when combined with modern large language models. In this paper, we propose an Attention-Based Subgraph Retriever, a GNN-as-retriever model that applies attention-based pruning to extract a refined subgraph, which is then passed to a large language model for advanced knowledge reasoning.

---

## 97. 论文ID: 2512.19575v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19575v1.json'

---

## 98. 论文ID: 2512.19540v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.19540v1.json'

---

## 99. Beyond Language Boundaries: Uncovering Programming Language Families for Code Language Models

**论文链接:** [http://arxiv.org/abs/2512.19509v1](http://arxiv.org/abs/2512.19509v1)

**作者:** Shangbo Yun, Xiaodong Gu, Jianghong Huang, Beijun Shen

**发布时间:** 2025-12-22

**备注:** Accepted by FSE 2026

### GPT解析

### 总结

该研究探索了编程语言之间的深层关系，并提出了一种基于嵌入的框架来揭示编程语言的潜在家族，进而优化多语言代码大语言模型的训练和推理。

### 背景

多种编程语言的快速增长为开发多语言代码大语言模型带来机遇和挑战，但现有技术通常仅通过简单聚合多语言代码数据来训练，很少探索编程语言间的深层关系及其对模型训练的优化作用。

### 目的

研究两个基本问题：1) 编程语言之间有哪些深层语言关系？2) 如何利用这些关系改进多语言代码大语言模型？

### 方法

提出基于嵌入的框架，定义21种编程语言主要特征，使用大语言模型生成多语言特征对齐代码样本，通过嵌入19种语言的语义平行代码片段构建相似性矩阵，执行层次聚类发现语言关系。

### 主要发现

编程语言间存在清晰层次结构，关系密切的语言形成明确定义的簇(如C、C++、Java和Swift)，而Go作为中心语言具有最高的跨语言相似性。

### 结论

基于发现的语言家族，提出三种增强多语言大语言模型训练的策略：语言相关间的迁移学习、语言接近度引导的课程学习和基于质心的中间代码翻译，实验证明这些方法显著提升了模型性能。

### 翻译

多种编程语言的快速增长为开发多语言代码大语言模型带来了机遇和挑战。虽然现有技术通常通过简单地聚合多语言代码数据来训练代码大语言模型，但很少探索编程语言之间的深层关系以及如何利用这些关系来优化代码大语言模型训练和推理。在本工作中，我们研究了两个基本问题：1) 编程语言之间的深层语言关系是什么？以及2) 如何利用这些关系来改进多语言代码大语言模型？我们提出了一种基于嵌入的框架来揭示编程语言的潜在家族。我们的方法首先定义了21种编程语言的主要语言特征，如变量定义、控制结构和方法声明，然后使用大语言模型在多种语言中生成特征对齐的代码样本。通过嵌入来自19种语言的语义平行代码片段，我们构建了一个相似性矩阵并进行层次聚类以发现固有的语言关系。我们的分析揭示了编程语言之间的清晰层次结构。关系密切的语言形成明确定义的簇(例如，C、C++、Java和Swift组合在一起)，而Go表现出作为中心语言的特点，具有最高的跨语言相似性。基于发现的语言家族，我们提出了三种策略来增强多语言大语言模型训练：在语言相关的语言之间进行迁移学习、语言接近度引导的课程学习以及基于质心的中间代码翻译。在4个代码智能任务上的实验证明，我们的方法显著提高了多语言大语言模型的性能。这项工作为编程语言提供了通用视角，并推进了更有效的多语言代码大语言模型训练策略。


### 论文摘要

The rapid proliferation of diverse programming languages presents both opportunities and challenges for developing multilingual code LLMs. While existing techniques often train code LLMs by simply aggregating multilingual code data, few explore the deeper relationships between programming languages(PLs) and how such relationships can be utilized to optimize the training and inference of code LLMs. In this work, we investigate 2 fundamental questions: 1) What are the deep linguistic relationships among PLs? and 2) How can these relationships be leveraged to improve multilingual code LLMs? We propose an embedding-based framework to uncover the latent families of PLs. Our approach begins by defining 21 primary linguistic features of programming languages, such as variable definition, control structures, and method declarations, and then employs LLMs to generate feature-aligned code samples across multiple languages. By embedding these semantically parallel code snippets from 19 languages, we construct a similarity matrix and perform hierarchical clustering to uncover inherent language relationships. Our analysis reveals clear hierarchical structures among programming languages. Closely related languages form well-defined clusters (e.g., C, C++, Java, and Swift group together), while Go exhibits as a central language with the highest cross-language similarity. Building on the uncovered language families, we propose three strategies to enhance multilingual LLM training: transfer learning across linguistically related languages, linguistic proximity-guided curriculum learning, and centroid-based intermediary code translation. Experiments on 4 code intelligence tasks demonstrate that our methods significantly improve multilingual LLM performance. This work offers a universal perspective on programming languages and advances more effective strategies for multilingual code LLM training.

---

## 100. 论文ID: 2512.18947v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18947v1.json'

---

## 101. Controllable Probabilistic Forecasting with Stochastic Decomposition Layers

**论文链接:** [http://arxiv.org/abs/2512.18815v1](http://arxiv.org/abs/2512.18815v1)

**作者:** John S. Schreck, William E. Chapman, Charlie Becker, David John Gagne, Dhamma Kimpara, Nihanth Cherukuru, Judith Berner, Kirsten J. Mayer, Negin Sobhani

**发布时间:** 2025-12-21

### GPT解析

### 总结

本文提出了一种名为随机分解层(Stochastic Decomposition Layers, SDL)的新方法，用于将确定性机器学习天气模型转换为概率性集合预测系统。SDL基于StyleGAN的分层噪声注入，通过潜在驱动的调制、逐像素噪声和通道缩放在三个解码器尺度上应用扰动。该方法计算成本低，仅需不到2%的基线模型训练成本，且每个集合成员由紧凑的潜在张量(5 MB)生成，可实现完美可重复性和后推理扩散调整。

### 背景

基于潜在噪声注入并通过连续排名概率分数(CRPS)优化的AI天气预测集合已经产生了准确且校准良好的预测，与基于扩散的方法相比计算成本大大降低。然而，当前的CRPS集合方法在训练策略和噪声注入机制上存在差异，大多数通过条件归一化在整个网络中全局注入噪声，这种结构增加了训练成本，并限制了随机扰动的物理解释性。

### 目的

引入随机分解层(Stochastic Decomposition Layers)，用于将确定性机器学习天气模型转换为概率性集合系统，解决现有方法训练成本高和物理解释性差的问题。

### 方法

1. 基于StyleGAN的分层噪声注入机制设计SDL；2. 在三个解码器尺度上应用学习到的扰动：潜在驱动的调制、逐像素噪声和通道缩放；3. 通过迁移学习将SDL应用于WXFormer模型；4. 使用紧凑的潜在张量(5 MB)生成每个集合成员；5. 通过潜在重缩放实现后推理扩散调整；6. 在2022年ERA5再分析数据集上评估模型性能。

### 主要发现

1. SDL仅需不到2%的基线模型训练计算成本；2. 集合的扩散-技能比接近 unity，排名直方图在中等范围预报中逐渐趋于均匀；3. 校准性能可与业务IFS-ENS相竞争；4. 多尺度实验揭示了分层不确定性：粗层调节天气尺度模式，细层控制中尺度变异性；5. 显式的潜在参数化为业务预报和气候应用提供了可解释的不确定性量化。

### 结论

SDL提供了一种高效且可解释的方法来生成天气预测集合，它不仅计算成本低，而且能够提供物理上有意义的不确定性表示，对业务预报和气候应用具有重要价值。

### 翻译

AI天气预测集合通过潜在噪声注入并用连续排名概率分数(CRPS)优化，与基于扩散的方法相比，产生了既准确又校准良好的预测，且计算成本大大降低。然而，当前的CRPS集合方法在训练策略和噪声注入机制上各不相同，大多数通过条件归一化在整个网络中全局注入噪声。这种结构增加了训练成本，并限制了随机扰动的物理解释性。我们引入了随机分解层(Stochastic Decomposition Layers, SDL)，用于将确定性机器学习天气模型转换为概率性集合系统。SDL借鉴了StyleGAN的分层噪声注入，通过潜在驱动的调制、逐像素噪声和通道缩放，在三个解码器尺度上应用学习到的扰动。当通过迁移学习应用于WXFormer时，SDL需要的计算成本不到基线模型训练的2%。每个集合成员由紧凑的潜在张量(5 MB)生成，实现了完美可重复性，并通过潜在重缩放实现后推理扩散调整。在2022年ERA5再分析数据集上的评估显示，集合的扩散-技能比接近 unity，排名直方图在中等范围预报中逐渐趋于均匀，校准性能可与业务IFS-ENS相媲美。多尺度实验揭示了分层不确定性：粗层调节天气尺度模式，而细层控制中尺度变异性。显式的潜在参数化为业务预报和气候应用提供了可解释的不确定性量化。


### 论文摘要

AI weather prediction ensembles with latent noise injection and optimized with the continuous ranked probability score (CRPS) have produced both accurate and well-calibrated predictions with far less computational cost compared with diffusion-based methods. However, current CRPS ensemble approaches vary in their training strategies and noise injection mechanisms, with most injecting noise globally throughout the network via conditional normalization. This structure increases training expense and limits the physical interpretability of the stochastic perturbations. We introduce Stochastic Decomposition Layers (SDL) for converting deterministic machine learning weather models into probabilistic ensemble systems. Adapted from StyleGAN's hierarchical noise injection, SDL applies learned perturbations at three decoder scales through latent-driven modulation, per-pixel noise, and channel scaling. When applied to WXFormer via transfer learning, SDL requires less than 2\% of the computational cost needed to train the baseline model. Each ensemble member is generated from a compact latent tensor (5 MB), enabling perfect reproducibility and post-inference spread adjustment through latent rescaling. Evaluation on 2022 ERA5 reanalysis shows ensembles with spread-skill ratios approaching unity and rank histograms that progressively flatten toward uniformity through medium-range forecasts, achieving calibration competitive with operational IFS-ENS. Multi-scale experiments reveal hierarchical uncertainty: coarse layers modulate synoptic patterns while fine layers control mesoscale variability. The explicit latent parameterization provides interpretable uncertainty quantification for operational forecasting and climate applications.

---

## 102. Building UI/UX Dataset for Dark Pattern Detection and YOLOv12x-based Real-Time Object Recognition Detection System

**论文链接:** [http://arxiv.org/abs/2512.18269v1](http://arxiv.org/abs/2512.18269v1)

**作者:** Se-Young Jang, Su-Yeon Yoon, Jae-Woong Jung, Dong-Hun Lee, Seong-Hun Choi, Soo-Kyung Jun, Yu-Bin Kim, Young-Seon Ju, Kyounggon Kim

**发布时间:** 2025-12-20

**备注:** 7page

### GPT解析

### 总结

本研究提出了一种视觉暗模式检测框架，通过构建专有数据集和应用YOLOv12x模型，实现了高准确性和实时性能的暗模式检测。

### 背景

数字化转型加速和在线平台普及导致暗模式问题日益突出，企业平台设计策略日益复杂，而监管机构主要采用被动方法，需要主动和实时检测技术。

### 目的

提出一种视觉暗模式检测框架，提高检测准确性和实时性能。

### 方法

构建包含4,066个UI/UX截图的专有数据集，标注五种暗模式相关UI组件，采用YOLOv12x目标检测模型和应用迁移学习优化性能。

### 主要发现

该方法在检测准确率方面达到92.8%，同时保持40.5帧每秒的实时推理速度，确认了在实际在线环境中部署的有效性。

### 结论

构建的数据集已公开发布，支持暗模式检测领域的进一步研究和开发，数据集可在GitHub上获取。

### 翻译

随着数字化转型步伐的加快和在线平台的广泛采用，关于暗模式（即削弱用户做出明智和理性选择能力的用户界面设计）的社会和技术问题日益突出。随着企业在线平台在设计策略上变得更加复杂，监管机构主要采用的被动方法之外，迫切需要主动和实时检测技术。在本文中，我们提出了一种视觉暗模式检测框架，提高了检测准确性和实时性能。为此，我们通过手动收集来自韩国和国外六个主要行业的194个网站的4,066个包含暗模式的UI/UX截图，构建了一个专有的视觉目标检测数据集。收集的图像被标注了五种与暗模式相关的代表性UI组件：按钮、复选框、输入字段、弹出窗口和二维码。该数据集已公开发布，以支持该领域的进一步研究和开发。为实现实时检测，本研究采用了YOLOv12x目标检测模型，并应用迁移学习优化其性能以实现视觉暗模式识别。实验结果表明，所提出的方法在检测准确率方面达到了92.8%，同时保持40.5帧每秒的实时推理速度，证实了其在在线环境中实际部署的有效性。此外，为促进未来研究并为技术进步做出贡献，本研究构建的数据集已在GitHub上公开发布。


### 论文摘要

With the accelerating pace of digital transformation and the widespread adoption of online platforms, both social and technical concerns regarding dark patterns-user interface designs that undermine users' ability to make informed and rational choices-have become increasingly prominent. As corporate online platforms grow more sophisticated in their design strategies, there is a pressing need for proactive and real-time detection technologies that go beyond the predominantly reactive approaches employed by regulatory authorities. In this paper, we propose a visual dark pattern detection framework that improves both detection accuracy and real-time performance. To this end, we constructed a proprietary visual object detection dataset by manually collecting 4,066 UI/UX screenshots containing dark patterns from 194 websites across six major industrial sectors in South Korea and abroad. The collected images were annotated with five representative UI components commonly associated with dark patterns: Button, Checkbox, Input Field, Pop-up, and QR Code. This dataset has been publicly released to support further research and development in the field. To enable real-time detection, this study adopted the YOLOv12x object detection model and applied transfer learning to optimize its performance for visual dark pattern recognition. Experimental results demonstrate that the proposed approach achieves a high detection accuracy of 92.8% in terms of mAP@50, while maintaining a real-time inference speed of 40.5 frames per second (FPS), confirming its effectiveness for practical deployment in online environments. Furthermore, to facilitate future research and contribute to technological advancements, the dataset constructed in this study has been made publicly available at https://github.com/B4E2/B4E2-DarkPattern-YOLO-DataSet.

---

## 103. Unsupervised Anomaly Detection with an Enhanced Teacher for Student-Teacher Feature Pyramid Matching

**论文链接:** [http://arxiv.org/abs/2512.18219v1](http://arxiv.org/abs/2512.18219v1)

**作者:** Mohammad Zolfaghari, Hedieh Sajedi

**发布时间:** 2025-12-20

**DOI:** 10.1109/CSICC55295.2022.9780522

### GPT解析

### 总结

本文提出了一种增强教师网络的学生-教师框架用于异常检测，在图像级别和像素级别上都取得了优异的性能。

### 背景

异常检测或离群点检测是无监督学习中的一个挑战性课题。

### 目的

开发一种高性能的学生-教师框架用于异常检测，通过增强教师网络来提高检测性能。

### 方法

首先在ImageNet上预训练ResNet-18网络，然后在MVTech-AD数据集上进行微调，提出名为ET-STPM的模型。

### 主要发现

在图像级别和像素级别上的实验结果表明，该方法比先前方法取得了更好的指标，ET-STPM模型在图像级别达到0.971的平均准确率，在像素级别达到0.977的平均准确率。

### 结论

增强教师网络的学生-教师框架在异常检测任务中表现优异，具有很高的实用价值。

### 翻译

异常检测或离群点是无监督学习中的一个挑战性课题。本文提出了一种用于异常检测的学生-教师框架，其教师网络被增强以实现高性能指标。为此，我们首先在ImageNet上预训练ResNet-18网络，然后在MVTech-AD数据集上进行微调。图像级别和像素级别的实验结果表明，这一想法比先前方法取得了更好的指标。我们的模型，增强教师学生-教师特征金字塔(ET-STPM)，在图像级别异常检测中达到0.971的平均准确率，在像素级别达到0.977的平均准确率。


### 论文摘要

Anomaly detection or outlier is one of the challenging subjects in unsupervised learning . This paper is introduced a student-teacher framework for anomaly detection that its teacher network is enhanced for achieving high-performance metrics . For this purpose , we first pre-train the ResNet-18 network on the ImageNet and then fine-tune it on the MVTech-AD dataset . Experiment results on the image-level and pixel-level demonstrate that this idea has achieved better metrics than the previous methods . Our model , Enhanced Teacher for Student-Teacher Feature Pyramid (ET-STPM), achieved 0.971 mean accuracy on the image-level and 0.977 mean accuracy on the pixel-level for anomaly detection.

---

## 104. 论文ID: 2512.18173v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.18173v1.json'

---

## 105. cardinalR: Generating Interesting High-Dimensional Data Structures

**论文链接:** [http://arxiv.org/abs/2512.18172v1](http://arxiv.org/abs/2512.18172v1)

**作者:** Jayani P. Gamage, Dianne Cook, Paul Harrison, Michael Lydeamore, Thiyanga S. Talagala

**发布时间:** 2025-12-20

### GPT解析

### 总结

本文介绍了一个名为cardinalR的R包，该包提供了生成多种高维数据结构的新方法，用于测试、验证和改进降维、监督学习和无监督学习算法。

### 背景

高维数据具有多个相互依赖或关联的变量，这些关联可能是线性的、非线性的、聚类形式或异常形式。这类数据对于算法测试和验证非常重要。

### 目的

提供生成各种高维结构的新方法，帮助研究人员更好地理解和改进不同的分析方法，特别关注非线性降维方法。

### 方法

使用数学函数和统计分布来生成高维结构，并将其组织到R包cardinalR中，同时提供了多个示例数据集。

### 主要发现

成功开发了多种生成高维数据的方法和相关示例数据集，这些工具可以帮助研究人员评估和改进算法性能。

### 结论

cardinalR包丰富了用于评估算法的基准数据集集合，为高维数据分析提供了新的工具。

### 翻译

模拟的高维数据对于测试、验证和改进降维、监督学习和无监督学习中使用的算法非常有用。高维数据的特点是具有多个以某种方式相互依赖或关联的变量，如线性、非线性、聚类或异常。本文我们提供了使用数学函数和统计分布生成各种高维结构的新方法，并将其组织到R包cardinalR中。同时提供了几个示例数据集。这些将有助于研究人员更好地理解和改进不同的分析方法，特别关注非线性降维方法。这个包丰富了用于评估算法的基准数据集集合。


### 论文摘要

Simulated high-dimensional data is useful for testing, validating, and improving algorithms used in dimension reduction, supervised and unsupervised learning. High-dimensional data is characterized by multiple variables that are dependent or associated in some way, such as linear, nonlinear, clustering or anomalies. Here we provide new methods for generating a variety of high-dimensional structures using mathematical functions and statistical distributions organized into the R package cardinalR. Several example data sets are also provided. These will be useful for researchers to better understand how different analytical methods work and can be improved, with a special focus on nonlinear dimension reduction methods. This package enriches the existing toolset of benchmark datasets for evaluating algorithms.

---

## 106. Pretrained Battery Transformer (PBT): A battery life prediction foundation model

**论文链接:** [http://arxiv.org/abs/2512.16334v2](http://arxiv.org/abs/2512.16334v2)

**作者:** Ruifeng Tan, Weixiang Hong, Jia Li, Jiaqiang Huang, Tong-Yi Zhang

**发布时间:** 2025-12-18

**备注:** 5 figures in the main content

### GPT解析

### 总结

本研究提出了预训练电池Transformer(PBT)，这是首个用于电池寿命预测的基础模型，通过迁移学习在多样化数据集上实现了最先进的性能。

### 背景

电池循环寿命的早期预测对加速电池研究、制造和部署至关重要，但机器学习方法因数据稀缺性和异质性而受限。虽然基础模型在其他领域通过迁移学习实现了广泛泛化，但尚未应用于电池寿命预测。

### 目的

开发首个用于电池寿命预测的基础模型，解决数据稀缺性和异质性问题，提高预测准确性。

### 方法

提出预训练电池Transformer(PBT)，通过领域知识编码的专家混合层开发，并在最大的公开电池寿命数据库上验证，从13个锂离子电池数据集中学习可迁移表示。

### 主要发现

PBT比现有模型平均性能提高19.8%，通过迁移学习在15个包含各种操作条件、形成协议和化学成分的多样化数据集上实现了最先进性能。

### 结论

该研究为电池寿命预测建立了基础模型路径，为开发通用电池寿命预测系统奠定了基础。

### 翻译

电池循环寿命的早期预测对于加速电池研究、制造和部署至关重要。尽管机器学习方法已经显示出令人鼓舞的结果，但由于不同老化条件导致的数据稀缺性和异质性，进展受到阻碍。在其他领域，通过迁移学习在多样化数据集上训练的基础模型已经实现了广泛的泛化能力，但尚未有关于电池循环寿命预测的基础模型的报道。在此，我们提出了预训练电池Transformer(PBT)，这是首个用于电池寿命预测的基础模型，通过领域知识编码的专家混合层开发。在最大的公开电池寿命数据库上验证，PBT从13个锂离子电池数据集中学习可迁移表示，比现有模型平均性能提高19.8%。通过迁移学习，PBT在15个包含各种操作条件、形成协议和化学成分的多样化数据集上实现了最先进性能。这项工作为电池寿命预测建立了基础模型路径，为通用电池寿命预测系统铺平了道路。


### 论文摘要

Early prediction of battery cycle life is essential for accelerating battery research, manufacturing, and deployment. Although machine learning methods have shown encouraging results, progress is hindered by data scarcity and heterogeneity arising from diverse aging conditions. In other fields, foundation models (FMs) trained on diverse datasets have achieved broad generalization through transfer learning, but no FMs have been reported for battery cycle life prediction yet. Here we present the Pretrained Battery Transformer (PBT), the first FM for battery life prediction, developed through domain-knowledge-encoded mixture-of-expert layers. Validated on the largest public battery life database, PBT learns transferable representations from 13 lithium-ion battery (LIB) datasets, outperforming existing models by an average of 19.8%. With transfer learning, PBT achieves state-of-the-art performance across 15 diverse datasets encompassing various operating conditions, formation protocols, and chemistries. This work establishes a foundation model pathway for battery lifetime prediction, paving the way toward universal battery lifetime prediction systems.

---

## 107. MapTrace: Scalable Data Generation for Route Tracing on Maps

**论文链接:** [http://arxiv.org/abs/2512.19609v1](http://arxiv.org/abs/2512.19609v1)

**作者:** Artemis Panagopoulou, Aveek Purohit, Achin Kulshrestha, Soroosh Yazdani, Mohit Goyal

**发布时间:** 2025-12-22

### GPT解析

### 总结

本文提出了一种可扩展的合成数据生成方法，用于改善多模态大语言模型在细粒度空间理解（特别是地图路线追踪）方面的能力，通过构建23k路径样本的数据集进行微调，显著提高了模型性能。

### 背景

多模态大语言模型在视觉和文本推理任务上已达到类人性能，但在细粒度空间理解（如地图路线追踪）方面表现有限。与人类能快速解析和导航地图不同，当前模型往往无法遵守基本路径约束，部分原因是收集大规模像素级精确路径标注的成本和难度过高。

### 目的

解决多模态大语言模型在细粒度空间理解（特别是地图路线追踪）方面的局限性，通过创建精确的标注数据集来提升模型的类人空间能力。

### 方法

引入了一个可扩展的合成数据生成管道，利用合成地图图像和像素级解析自动生成具有精确标注的挑战性任务数据。使用该管道构建了包含4k张地图上23k个路径样本的微调数据集，并使用该数据集对开源和专有的多模态大语言模型进行微调。

### 主要发现

在MapBench上的结果显示，微调显著提高了模型的鲁棒性，成功率提高了最多6.4个百分点，同时减少了路径追踪错误（NDTW）。这些增益表明，预训练模型中缺乏的细粒度空间推理能力可以通过合成监督明确地教授给模型。

### 结论

细粒度空间推理能力，尽管在预训练模型中不存在，但可以通过合成监督明确地教授给模型，从而改善多模态大语言模型在空间理解任务上的表现。

### 翻译

虽然多模态大语言模型在许多视觉和文本推理任务上已达到类人性能，但它们在细粒度空间理解（如地图路线追踪）方面的能力仍然有限。与人类能够快速学习解析和导航地图不同，当前模型往往无法遵守基本的路径约束，部分原因是收集大规模像素级精确路径标注的成本和难度过高。为此，我们引入了一个可扩展的合成数据生成管道，利用合成地图图像和像素级解析自动为这一具有挑战性的任务生成精确标注。使用该管道，我们构建了一个包含4k张地图上23k个路径样本的微调数据集，使模型能够获得更类人的空间能力。使用该数据集，我们对开源和专有的多模态大语言模型进行了微调。MapBench上的结果显示，微调显著提高了鲁棒性，将成功率提高了最多6.4个百分点，同时减少了路径追踪错误（NDTW）。这些增益表明，预训练模型中缺乏的细粒度空间推理能力可以通过合成监督明确地教授。


### 论文摘要

While Multimodal Large Language Models have achieved human-like performance on many visual and textual reasoning tasks, their proficiency in fine-grained spatial understanding, such as route tracing on maps remains limited. Unlike humans, who can quickly learn to parse and navigate maps, current models often fail to respect fundamental path constraints, in part due to the prohibitive cost and difficulty of collecting large-scale, pixel-accurate path annotations. To address this, we introduce a scalable synthetic data generation pipeline that leverages synthetic map images and pixel-level parsing to automatically produce precise annotations for this challenging task. Using this pipeline, we construct a fine-tuning dataset of 23k path samples across 4k maps, enabling models to acquire more human-like spatial capabilities. Using this dataset, we fine-tune both open-source and proprietary MLLMs. Results on MapBench show that finetuning substantially improves robustness, raising success rates by up to 6.4 points, while also reducing path-tracing error (NDTW). These gains highlight that fine-grained spatial reasoning, absent in pretrained models, can be explicitly taught with synthetic supervision.

---

## 108. VOIC: Visible-Occluded Decoupling for Monocular 3D Semantic Scene Completion

**论文链接:** [http://arxiv.org/abs/2512.18954v1](http://arxiv.org/abs/2512.18954v1)

**作者:** Zaidao Han, Risa Higashita, Jiang Liu

**发布时间:** 2025-12-22

### GPT解析

### 总结

论文提出了一种名为VOIC的新型双解码器框架，通过引入可见区域标签提取策略，将3D语义场景完成任务解耦为可见区域语义感知和遮挡区域场景完成两个子任务，有效解决了单图像输入导致的特征稀释和错误传播问题。

### 背景

基于相机的3D语义场景完成是自动驾驶和机器人场景理解的关键任务，旨在从单张图像推断完整的3D体积表示（包括语义和几何）。现有方法忽略了单图像输入导致的可见区域与遮挡区域之间的干扰问题。

### 目的

解决现有单图像3D语义场景完成方法中可见区域感知与遮挡区域推理之间的干扰，提高几何完成和语义分割的准确性。

### 方法

引入离线可见区域标签提取策略分离可见和遮挡区域的监督；提出VOIC双解码器框架，将SSC解耦为可见区域语义感知和遮挡区域场景完成；通过融合图像特征和深度信息构建基础3D体素表示；可见解码器生成几何和语义先验，遮挡解码器利用先验和跨模态交互进行全局场景推理。

### 主要发现

在SemanticKITTI和SSCBench-KITTI360基准上的实验表明，VOIC在几何完成和语义分割准确性方面均优于现有单目SSC方法，达到最先进性能。

### 结论

通过将SSC任务明确解耦并利用VRLE策略净化监督空间，有效提高了3D语义场景完成的性能，解决了单图像输入的局限性。

### 翻译

基于相机的3D语义场景完成是自动驾驶和机器人场景理解的关键任务。它旨在从单张图像推断出包含语义和几何信息的完整3D体积表示。现有方法通常专注于端到端的2D到3D特征提升和体素完成。然而，它们常常忽略了单图像输入导致的高置信度可见区域感知与低置信度遮挡区域推理之间的干扰，这可能导致特征稀释和错误传播。为解决这些挑战，我们引入了一种离线可见区域标签提取策略，该策略明确分离并提取来自密集3D真实可见区域的体素级监督。这一策略为两个互补的子任务净化了监督空间：可见区域感知和遮挡区域推理。基于这一理念，我们提出了可见-遮挡交互完成网络，一种新型双解码器框架，明确将SSC解耦为可见区域语义感知和遮挡区域场景完成。VOIC首先通过融合图像特征和深度推导的占用率构建基础3D体素表示。可见解码器专注于生成高保真度的几何和语义先验，而遮挡解码器则利用这些先验以及跨模态交互执行连贯的全局场景推理。在SemanticKITTI和SSCBench-KITTI360基准上的大量实验表明，VOIC在几何完成和语义分割准确性方面均优于现有的单目SSC方法，达到了最先进的性能。


### 论文摘要

Camera-based 3D Semantic Scene Completion (SSC) is a critical task for autonomous driving and robotic scene understanding. It aims to infer a complete 3D volumetric representation of both semantics and geometry from a single image. Existing methods typically focus on end-to-end 2D-to-3D feature lifting and voxel completion. However, they often overlook the interference between high-confidence visible-region perception and low-confidence occluded-region reasoning caused by single-image input, which can lead to feature dilution and error propagation.   To address these challenges, we introduce an offline Visible Region Label Extraction (VRLE) strategy that explicitly separates and extracts voxel-level supervision for visible regions from dense 3D ground truth. This strategy purifies the supervisory space for two complementary sub-tasks: visible-region perception and occluded-region reasoning. Building on this idea, we propose the Visible-Occluded Interactive Completion Network (VOIC), a novel dual-decoder framework that explicitly decouples SSC into visible-region semantic perception and occluded-region scene completion. VOIC first constructs a base 3D voxel representation by fusing image features with depth-derived occupancy. The visible decoder focuses on generating high-fidelity geometric and semantic priors, while the occlusion decoder leverages these priors together with cross-modal interaction to perform coherent global scene reasoning.   Extensive experiments on the SemanticKITTI and SSCBench-KITTI360 benchmarks demonstrate that VOIC outperforms existing monocular SSC methods in both geometric completion and semantic segmentation accuracy, achieving state-of-the-art performance.

---

