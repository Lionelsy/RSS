# 今日论文推荐 - 2025-11-21

共 152 篇论文

---

## 1. GeoPTH: A Lightweight Approach to Category-Based Trajectory Retrieval via Geometric Prototype Trajectory Hashing

**论文链接:** [http://arxiv.org/abs/2511.16258v1](http://arxiv.org/abs/2511.16258v1)

**作者:** Yang Xu, Zuliang Yang, Kai Ming Ting

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种名为GeoPTH的新型轻量级非学习框架，用于高效基于类别的轨迹检索，通过几何原型解决了传统方法和学习方法在计算效率和稳定性方面的局限性。

### 背景

轨迹相似性检索是时空数据挖掘的重要组成部分，但现有方法存在局限性：传统计算方法计算成本高，而基于学习的方法训练成本高且可能不稳定。

### 目的

解决现有方法的局限性，提出一种高效、轻量级的基于类别的轨迹检索框架。

### 方法

提出GeoPTH（几何原型轨迹哈希）框架，使用代表性轨迹原型（保留几何特征的小点集）作为锚点构建数据相关的哈希函数，通过鲁棒的Hausdorff度量将新轨迹映射到最近的原型，实现高效的哈希过程。

### 主要发现

GeoPTH的检索准确性与传统方法和最先进的学习方法具有竞争力，显著优于通过简单二值化学习嵌入生成的二进制代码，在效率方面一致优于所有竞争方法。

### 结论

轻量级、以原型为中心的方法是一种实用且强大的替代方案，实现了卓越的检索性能和计算效率。

### 翻译

轨迹相似性检索是时空数据挖掘的重要组成部分，然而现有方法存在以下局限性：传统度量计算成本高，而基于学习的方法则存在大量训练成本和潜在的不稳定性。本文通过提出GeoPTH（几何原型轨迹哈希）来解决这些问题，这是一种新颖的轻量级非学习框架，用于高效的基于类别的轨迹检索。GeoPTH使用代表性轨迹原型（即保留几何特征的小点集）作为锚点，构建数据相关的哈希函数。哈希过程高效，涉及通过鲁棒的Hausdorff度量将新轨迹映射到其最近的原型。大量实验表明，GeoPTH的检索准确性与传统方法和最先进的学习方法高度竞争，并且显著优于通过简单二值化学习嵌入生成的二进制代码。关键的是，GeoPTH在效率方面始终优于所有竞争方法。我们的工作表明，轻量级、以原型为中心的方法提供了一种实用且强大的替代方案，实现了卓越的检索性能和计算效率。


### 论文摘要

Trajectory similarity retrieval is an important part of spatiotemporal data mining, however, existing methods have the following limitations: traditional metrics are computationally expensive, while learning-based methods suffer from substantial training costs and potential instability. This paper addresses these problems by proposing \textbf{Geo}metric \textbf{P}rototype \textbf{T}rajectory \textbf{H}ashing (GeoPTH), a novel, lightweight, and non-learning framework for efficient category-based trajectory retrieval. GeoPTH constructs data-dependent hash functions by using representative trajectory prototypes, i.e., small point sets preserving geometric characteristics, as anchors. The hashing process is efficient, which involves mapping a new trajectory to its closest prototype via a robust, \textit{Hausdorff} metric. Extensive experiments show that GeoPTH's retrieval accuracy is highly competitive with both traditional metrics and state-of-the-art learning methods, and it significantly outperforms binary codes generated through simple binarization of the learned embeddings. Critically, GeoPTH consistently outperforms all competitors in terms of efficiency. Our work demonstrates that a lightweight, prototype-centric approach offers a practical and powerful alternative, achieving an exceptional retrieval performance and computational efficiency.

---

## 2. Physics-Guided Inductive Spatiotemporal Kriging for PM2.5 with Satellite Gradient Constraints

**论文链接:** [http://arxiv.org/abs/2511.16013v1](http://arxiv.org/abs/2511.16013v1)

**作者:** Shuo Wang, Mengfan Teng, Yun Cheng, Lothar Thiele, Olga Saukh, Shuangshuang He, Yuanting Zhang, Jiang Zhang, Gangfeng Zhang, Xingyuan Yuan, Jingfang Fan

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究提出了时空物理引导推理网络(SPIN)，一种用于PM2.5高分辨率映射的新型框架，克服了传统卫星数据方法的局限性，实现了在未监测区域生成连续且物理合理的污染场。

### 背景

高分辨率细颗粒物(PM2.5)映射是可持续城市发展的基石，但地面监测网络的稀疏性严重阻碍了这一工作。传统数据驱动方法使用卫星气溶胶光学厚度(AOD)尝试填补空白，但面临严重非随机数据缺失和反演偏差问题。

### 目的

克服传统数据驱动方法的局限性，开发一种能够有效处理数据空缺且物理合理的PM2.5高分辨率映射方法。

### 方法

提出时空物理引导推理网络(SPIN)，通过并行图核显式建模物理平流和扩散过程将领域知识深度学习相结合；创新性地将有误差的AOD作为损失函数中的空间梯度约束而非直接输入，使模型能从卫星数据学习污染结构模式同时对数据空缺保持鲁棒性。

### 主要发现

在京津冀及周边地区(BTHSA)验证中，SPIN达到新最先进水平，平均绝对误差(MAE)为9.52微克/立方米，能在未监测区域生成连续且物理合理的污染场。

### 结论

SPIN为精细化环境管理提供了一种稳健的、低成本的、全天候的解决方案。

### 翻译

高分辨率细颗粒物(PM2.5)映射是可持续城市主义的基石，但仍然受到地面监测网络空间稀疏性的严重阻碍。虽然传统数据驱动方法尝试使用卫星气溶胶光学厚度(AOD)来弥合这一差距，但它们常常遭受严重的非随机数据缺失(例如由于云覆盖或夜间)和反演偏差。为了克服这些限制，本研究提出了时空物理引导推理网络(SPIN)，一种专为归纳时空克里金法设计的新型框架。与传统方法不同，SPIN通过并行图核显式建模物理平流和扩散过程，将领域知识深度学习协同整合。关键的是，我们引入了一种变革性的训练策略：不是将有误差的AOD作为直接输入，而是将其重新用作损失函数中的空间梯度约束。这使得模型能够从卫星数据中学习污染结构模式，同时对数据空缺保持鲁棒性。在高度污染的京津冀及周边地区(BTHSA)验证中，SPIN实现了新的最先进水平，平均绝对误差(MAE)为9.52微克/立方米，即使在未监测区域也能有效生成连续的、物理上合理的污染场。这项工作为精细化环境管理提供了一种稳健的、低成本的和全天候的解决方案。


### 论文摘要

High-resolution mapping of fine particulate matter (PM2.5) is a cornerstone of sustainable urbanism but remains critically hindered by the spatial sparsity of ground monitoring networks. While traditional data-driven methods attempt to bridge this gap using satellite Aerosol Optical Depth (AOD), they often suffer from severe, non-random data missingness (e.g., due to cloud cover or nighttime) and inversion biases. To overcome these limitations, this study proposes the Spatiotemporal Physics-Guided Inference Network (SPIN), a novel framework designed for inductive spatiotemporal kriging. Unlike conventional approaches, SPIN synergistically integrates domain knowledge into deep learning by explicitly modeling physical advection and diffusion processes via parallel graph kernels. Crucially, we introduce a paradigm-shifting training strategy: rather than using error-prone AOD as a direct input, we repurpose it as a spatial gradient constraint within the loss function. This allows the model to learn structural pollution patterns from satellite data while remaining robust to data voids. Validated in the highly polluted Beijing-Tianjin-Hebei and Surrounding Areas (BTHSA), SPIN achieves a new state-of-the-art with a Mean Absolute Error (MAE) of 9.52 ug/m^3, effectively generating continuous, physically plausible pollution fields even in unmonitored areas. This work provides a robust, low-cost, and all-weather solution for fine-grained environmental management.

---

## 3. Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models

**论文链接:** [http://arxiv.org/abs/2511.15743v1](http://arxiv.org/abs/2511.15743v1)

**作者:** Linnea M. Wolniewicz, Halil S. Kelebek, Simone Mestici, Michael D. Vergalla, Giacomo Acciarini, Bala Poduval, Olga Verkhoglyadova, Madhulika Guhathakurta, Thomas E. Berger, Atılım Güneş Baydin, Frank Soboczenski

**发布时间:** 2025-11-18

**备注:** 8 pages, 2 figures, 2 tables. Accepted as a poster presentation in the Machine Learning for the Physical Sciences workshop at NeurIPS 2025

### GPT解析

### 总结

该研究整合了多种电离层和日球层测量数据，创建了一个开放获取的数据集，用于支持下一代电离层预测模型，并通过机器学习方法预测总电子含量。

### 背景

电离层业务预测面临观测稀疏、地理层间复杂耦合等挑战，同时GNSS、通信、航空安全和卫星运行等领域对及时准确的预测需求日益增长。

### 目的

作为2025年NASA Heliolab的一部分，创建一个整合多样化电离层和日球层测量的开放获取数据集，支持下一代预测模型并解决当前业务框架中的差距。

### 方法

整合多种数据源包括太阳动力学观测站数据、太阳辐照度指数、太阳风参数、地磁活动指数和NASA JPL的电离层图，以及来自全球GNSS接收机网络和智能手机的稀疏数据。将这些时空对齐的异构数据集用于训练和评估时空机器学习架构，预测垂直总电子含量。

### 主要发现

通过该数据集训练和评估了多个机器学习模型，用于在平静和地磁活动条件下预测垂直总电子含量。

### 结论

该研究提供的数据集和建模管道支持探索电离层动力学和太阳-地球相互作用，有助于科学研究和业务预测工作。

### 翻译

电离层的业务预测仍然是空间天气的重大挑战，原因在于观测稀疏、地理层间复杂耦合，以及对支持全球导航卫星系统(GNSS)、通信、航空安全以及卫星运行的及时准确预测的日益增长的需求。作为2025年NASA Heliolab的一部分，我们提出了一个精心策划的开放获取数据集，将多样化的电离层和日球层测量整合到一个连贯的、机器学习就绪的结构中，专门用于支持下一代预测模型并解决当前业务框架中的差距。我们的工作流程整合了大量数据源，包括太阳动力学观测站数据、太阳辐照度指数(F10.7)、太阳风参数(速度和行星际磁场)、地磁活动指数(Kp, AE, SYM-H)以及NASA JPL的总电子含量全球电离层图(GIM-TEC)。我们还实现了地理上稀疏的数据，如来自全球GNSS接收机网络和众包Android智能手机测量的TEC。这个新颖的异构数据集在时间和空间上对齐到单个模块化数据结构中，支持物理和数据驱动建模。利用这个数据集，我们训练和评估了几个时空机器学习架构，用于在平静和地磁活动条件下预测垂直TEC。这项工作提出了一个广泛的数据集和建模管道，使能够探索电离层动力学以及更广泛的太阳-地球相互作用，支持科学研究和业务预测工作。


### 论文摘要

Operational forecasting of the ionosphere remains a critical space weather challenge due to sparse observations, complex coupling across geospatial layers, and a growing need for timely, accurate predictions that support Global Navigation Satellite System (GNSS), communications, aviation safety, as well as satellite operations. As part of the 2025 NASA Heliolab, we present a curated, open-access dataset that integrates diverse ionospheric and heliospheric measurements into a coherent, machine learning-ready structure, designed specifically to support next-generation forecasting models and address gaps in current operational frameworks. Our workflow integrates a large selection of data sources comprising Solar Dynamic Observatory data, solar irradiance indices (F10.7), solar wind parameters (velocity and interplanetary magnetic field), geomagnetic activity indices (Kp, AE, SYM-H), and NASA JPL's Global Ionospheric Maps of Total Electron Content (GIM-TEC). We also implement geospatially sparse data such as the TEC derived from the World-Wide GNSS Receiver Network and crowdsourced Android smartphone measurements. This novel heterogeneous dataset is temporally and spatially aligned into a single, modular data structure that supports both physical and data-driven modeling. Leveraging this dataset, we train and benchmark several spatiotemporal machine learning architectures for forecasting vertical TEC under both quiet and geomagnetically active conditions. This work presents an extensive dataset and modeling pipeline that enables exploration of not only ionospheric dynamics but also broader Sun-Earth interactions, supporting both scientific inquiry and operational forecasting efforts.

---

## 4. Learning from Dense Events: Towards Fast Spiking Neural Networks Training via Event Dataset Distillation

**论文链接:** [http://arxiv.org/abs/2511.12095v2](http://arxiv.org/abs/2511.12095v2)

**作者:** Shuhan Ye, Yi Yu, Qixin Zhang, Chenqi Kong, Qiangqiang Wu, Kun Wang, Xudong Jiang

**发布时间:** 2025-11-15

### GPT解析

### 总结

PACE是首个针对SNNs和事件视觉的数据集蒸馏框架，通过将大型训练数据集提炼为紧凑合成数据集，实现快速SNN训练，显著降低训练时间和存储成本，使SNNs能够在边缘设备上高效部署。

### 背景

事件相机通过感知亮度变化输出二进制异步事件流，其生物启发的动态特性与脉冲神经网络高度契合，作为传统视觉系统有希望的节能替代方案，但SNNs由于时间编码导致训练成本高，限制了实际部署。

### 目的

降低SNNs的训练成本，开发一种数据集蒸馏框架，使SNNs能够在资源受限环境中高效训练和部署。

### 方法

PACE框架通过两个核心模块实现：ST-DSM使用剩余膜电位增加基于尖峰的特征密度并进行细粒度时空匹配；PEQ-N提供即插即用的直通概率整数量化器，兼容标准事件帧管道，将大型训练数据集提炼为紧凑合成数据集。

### 主要发现

在DVS-Gesture、CIFAR10-DVS和N-MNIST数据集上，PACE优于现有基线，在动态事件流和低或中等IPC方面表现特别强；在N-MNIST上达到84.4%准确率，接近完整训练集性能的85%，同时将训练时间减少50倍以上，存储成本减少6000倍。

### 结论

PACE为SNNs提供了高效的数据集蒸馏方法，显著降低训练时间和存储需求，使SNNs能够在资源受限的边缘设备上实现分钟级训练和高效部署。

### 翻译

事件相机感知亮度变化并输出二进制异步事件流，日益受到关注。它们的生物启发的动态特性与脉冲神经网络高度契合，为传统视觉系统提供了有希望的节能替代方案。然而，由于时间编码，SNNs的训练成本仍然很高，限制了它们的实际部署。为了减轻SNNs的高训练成本，我们引入了PACE(事件相位对齐凝聚)，这是首个针对SNNs和事件视觉的数据集蒸馏框架。PACE将大型训练数据集提炼为紧凑的合成数据集，实现快速SNN训练，这通过两个核心模块实现：ST-DSM和PEQ-N。ST-DSM使用剩余膜电位来增加基于尖峰的特征密度，并进行幅度和相位的细粒度时空匹配，而PEQ-N提供兼容标准事件帧管道的即插即用直通概率整数量化器。在DVS-Gesture、CIFAR10-DVS和N-MNIST数据集上，PACE优于现有的核心集选择和数据集蒸馏基线，在动态事件流和低或中等IPC方面表现出特别强的增益。具体来说，在N-MNIST上，它达到84.4%的准确率，约为完整训练集性能的85%，同时将训练时间减少50倍以上，存储成本减少6000倍，产生紧凑的替代模型，实现分钟级的SNN训练和高效的边缘部署。


### 论文摘要

Event cameras sense brightness changes and output binary asynchronous event streams, attracting increasing attention. Their bio-inspired dynamics align well with spiking neural networks (SNNs), offering a promising energy-efficient alternative to conventional vision systems. However, SNNs remain costly to train due to temporal coding, which limits their practical deployment. To alleviate the high training cost of SNNs, we introduce \textbf{PACE} (Phase-Aligned Condensation for Events), the first dataset distillation framework to SNNs and event-based vision. PACE distills a large training dataset into a compact synthetic one that enables fast SNN training, which is achieved by two core modules: \textbf{ST-DSM} and \textbf{PEQ-N}. ST-DSM uses residual membrane potentials to densify spike-based features (SDR) and to perform fine-grained spatiotemporal matching of amplitude and phase (ST-SM), while PEQ-N provides a plug-and-play straight through probabilistic integer quantizer compatible with standard event-frame pipelines. Across DVS-Gesture, CIFAR10-DVS, and N-MNIST datasets, PACE outperforms existing coreset selection and dataset distillation baselines, with particularly strong gains on dynamic event streams and at low or moderate IPC. Specifically, on N-MNIST, it achieves \(84.4\%\) accuracy, about \(85\%\) of the full training set performance, while reducing training time by more than \(50\times\) and storage cost by \(6000\times\), yielding compact surrogates that enable minute-scale SNN training and efficient edge deployment.

---

## 5. Mesh-based Super-resolution of Detonation Flows with Multiscale Graph Transformers

**论文链接:** [http://arxiv.org/abs/2511.12041v2](http://arxiv.org/abs/2511.12041v2)

**作者:** Shivam Barwey, Pinaki Pal

**发布时间:** 2025-11-15

### GPT解析

### 总结

本文提出了一种首创的多尺度图变换器方法(SR-GT)用于基于网格的超分辨率反应流重建，该方法利用基于图的流场表示和变换器主干结构捕获长程依赖关系，在复杂反应流场测试中表现出比传统插值方法更高的超分辨率精度。

### 背景

超分辨率流场重建对多种应用有价值，如亚网格/亚过滤器闭合建模、加速时空预测、数据压缩和稀疏实验测量的上采样工具。

### 目的

开发一种首创的多尺度图变换器方法用于基于网格的超分辨率(SR-GT)处理反应流。

### 方法

提出了一种新的数据驱动建模范式，利用基于图的流场表示兼容复杂几何和非均匀/非结构化网格。变换器主干捕获低分辨率流场不同部分之间的长程依赖关系，识别重要特征，然后生成保留这些特征的超分辨率流场。SR-GT框架利用独特的元素局部(+邻域)图表示处理粗输入，然后在进行变换器组件处理之前进行标记化，以产生精细输出。

### 主要发现

在预混氢气-空气混合物中2D爆震传播的具有挑战性的测试问题上，SR-GT提供了反应流场特征的高超分辨率精度，并且比传统的基于插值的超分辨率方案具有优越性能。

### 结论

SR-GT方法在处理复杂反应流场的超分辨率重建方面表现出色，是一种创新且有效的方法。

### 翻译

使用最先进的数据驱动技术进行超分辨率流场重建对多种应用有价值，例如亚网格/亚过滤器闭合建模、加速时空预测、数据压缩以及作为稀疏实验测量的上采样工具。在当前工作中，开发了一种首创的多尺度图变换器方法用于基于网格的超分辨率(SR-GT)处理反应流。这种新的数据驱动建模范式利用了基于图的流场表示，兼容复杂几何和非均匀/非结构化网格。此外，变换器主干捕获低分辨率流场不同部分之间的长程依赖关系，识别重要特征，然后生成保留这些特征的超分辨率流场。在预混氢气-空气中2D爆震传播的具有挑战性的测试问题上，SR-GT的性能在谱元离散化网格背景下得到了展示，该问题表现出高度复杂的多尺度反应流行为。SR-GT框架利用独特的元素局部(+邻域)图表示处理粗输入，然后在进行变换器组件处理之前进行标记化，以产生精细输出。研究表明，SR-GT为反应流场特征提供了高超分辨率精度，并且比传统的基于插值的超分辨率方案具有优越性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决高保真计算流体动力学(CFD)模拟湍流反应流动的计算瓶颈问题。具体而言，是通过超分辨率技术从低分辨率流场重建高分辨率流场。这个问题很重要，因为直接数值模拟(DNS)计算成本过高，而粗网格模拟方法需要准确的亚网格模型。超分辨率方法可用于子网格建模、加速时空预测、数据压缩和处理稀疏实验数据，对于理解和优化燃烧设备如内燃机、燃气轮机等至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有超分辨率方法的局限性，特别是传统插值方法在复杂流动场景中的不足。他们认识到图神经网络(GNN)能处理复杂几何结构，而Transformer能捕获长程依赖关系，因此将两者结合。作者借鉴了Barwey等人的图神经网络超分辨率方法(SR-GNN)和Xu等人的Transformer超分辨率框架，以及Bode等人的物理信息增强生成对抗网络(PIERSGAN)。在这些现有工作的基础上，作者创新性地设计了多尺度图Transformer架构(SR-GT)，专门针对反应流动的超分辨率重建。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用图表示兼容复杂几何和非结构化网格，同时利用Transformer主干网络捕获低分辨率流场中的长程依赖关系，识别重要特征，并在更高分辨率下重建保留这些特征的流场。实现流程包括：1)从高保真模拟生成训练数据集；2)使用K近邻算法构建局部邻域图；3)将查询元素及其邻居转换为token；4)通过8层Transformer处理token序列，每层包含多头自注意力和MLP；5)训练时使用AdamW优化器和自定义学习率调度；6)评估时比较与KNN插值的性能，并检查物理约束如质量守恒的满足情况。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首创的多尺度图Transformer方法用于反应流场超分辨率；2)结合图表示与Transformer的优势，同时处理复杂几何结构和捕捉长程依赖；3)使用元素局部图表示处理粗输入；4)在爆轰流动这一复杂多尺度反应流动问题上验证方法有效性。相比之前工作，SR-GT避免了GAN的训练不稳定问题，相比SR-GNN能更好捕获长程依赖，相比CNN等方法天然兼容复杂几何，相比传统插值方法在反应流场特征超分辨率上表现更优，特别是在爆轰前沿附近。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种创新的多尺度图Transformer方法，能够从低分辨率爆轰流场重建高分辨率流场，在保持物理约束的同时显著提高了复杂反应流动的超分辨率精度，为计算流体动力学中的计算效率与精度平衡提供了新的解决方案。'}


### 论文摘要

Super-resolution flow reconstruction using state-of-the-art data-driven techniques is valuable for a variety of applications, such as subgrid/subfilter closure modeling, accelerating spatiotemporal forecasting, data compression, and serving as an upscaling tool for sparse experimental measurements. In the present work, a first-of-its-kind multiscale graph transformer approach is developed for mesh-based super-resolution (SR-GT) of reacting flows. The novel data-driven modeling paradigm leverages a graph-based flow-field representation compatible with complex geometries and non-uniform/unstructured grids. Further, the transformer backbone captures long-range dependencies between different parts of the low-resolution flow-field, identifies important features, and then generates the super-resolved flow-field that preserves those features at a higher resolution. The performance of SR-GT is demonstrated in the context of spectral-element-discretized meshes for a challenging test problem of 2D detonation propagation within a premixed hydrogen-air mixture exhibiting highly complex multiscale reacting flow behavior. The SR-GT framework utilizes a unique element-local (+ neighborhood) graph representation for the coarse input, which is then tokenized before being processed by the transformer component to produce the fine output. It is demonstrated that SR-GT provides high super-resolution accuracy for reacting flow-field features and superior performance compared to traditional interpolation-based SR schemes.

---

## 6. Late-decoupled 3D Hierarchical Semantic Segmentation with Semantic Prototype Discrimination based Bi-branch Supervision

**论文链接:** [http://arxiv.org/abs/2511.16650v1](http://arxiv.org/abs/2511.16650v1)

**作者:** Shuyu Cao, Chongshou Li, Jie Xu, Tianrui Li, Na Zhao

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种新颖的3D分层语义分割框架，解决了多层级冲突和类别不平衡问题，通过晚期解耦架构和双分支监督机制实现了最先进的性能。

### 背景

3D分层语义分割(3DHS)对于需要多粒度和多层级理解3D场景的具身智能应用至关重要。

### 目的

解决3DHS中的两个主要挑战：多层级冲突和跨层级的类别不平衡问题。

### 方法

提出包含主3DHS分支和辅助判别分支的新框架，采用晚期解耦3DHS架构使用多个解码器实现从粗到细的层级指导和一致性，并引入基于语义原型的双分支监督机制进行相互监督。

### 主要发现

在多个数据集和主干网络上的实验表明，该方法实现了最先进的3DHS性能，其核心组件可作为即插即用的增强改进现有方法。

### 结论

所提出的框架有效缓解了多层级冲突和类别不平衡问题，显著提升了3DHS性能。

### 翻译

3D分层语义分割(3DHS)对于需要多粒度和多层级理解3D场景的具身智能应用至关重要。尽管有进展，但之前的3DHS方法忽略了两个挑战：I)使用参数共享模型进行多标签学习可能导致跨层级优化中的多层级冲突；II)3D场景的多个层级中不可避免地存在类别不平衡问题，这使模型性能被主要类别主导。为解决这些问题，我们提出了一个包含主3DHS分支和辅助判别分支的新框架。具体而言，为了缓解多层级冲突，我们提出了一个晚期解耦的3DHS框架，该框架使用具有从粗到细层级指导和一致性的多个解码器。晚期解耦架构可以减轻多个层级之间的欠拟合和过拟合冲突，并可以约束每个单独层级的类别不平衡问题。此外，我们引入了一种面向3DHS的基于语义原型的双分支监督机制，该机制额外学习类判别性点云特征，并在辅助分支和3DHS分支之间执行相互监督，以增强类别不平衡分割。在多个数据集和主干网络上的广泛实验表明，我们的方法实现了最先进的3DHS性能，并且其核心组件可以作为即插即用的增强来改进之前的方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D层次语义分割中的两个关键挑战：1) 多层次冲突问题 - 共享参数模型在跨层次优化时会导致不同层次间的学习目标冲突；2) 类别不平衡问题 - 多层次场景中多数类主导模型性能，少数类分割效果差。这些问题在现实世界中非常重要，因为自动驾驶、机器人导航和增强现实等应用需要多粒度、多层次的场景理解，传统单层次分割无法满足这些需求。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将3DHS任务类比为多标签分类问题，认识到共享参数会导致某些层次过拟合而其他层次欠拟合。因此设计晚期解耦架构，每个层次使用独立解码器同时共享编码器。还借鉴了对比学习思想引入辅助判别分支，学习类别判别性特征。方法融合了多任务学习、对比学习和原型引导等现有技术，创新性地将它们组合解决3DHS特有的挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1) 晚期解耦架构 - 使用共享编码器和多个独立解码器避免多层次冲突；2) 粗到细层次指导 - 融合高层语义预测与低层特征实现信息流动；3) 语义原型引导的双分支监督 - 辅助分支学习判别性特征并与主分支相互监督。流程包括：输入点云→共享编码器提取特征→主分支进行层次分割并应用粗到细指导→辅助分支学习类别判别特征并构建语义原型→双分支相互监督→联合优化训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) 晚期解耦3DHS框架解决多层次冲突；2) 语义原型引导的双分支监督机制处理类别不平衡；3) 粗到细层次指导增强层次间信息流动。相比之前工作(如MTHS和DHL)，本文使用独立解码器而非共享解码器，引入专门处理类别不平衡的辅助分支，并设计了更有效的层次间信息流动机制。此外，本文方法可作为即插即用模块增强现有方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种晚期解耦的3D层次语义分割框架和语义原型引导的双分支监督机制，有效解决了多层次冲突和类别不平衡问题，显著提升了3D层次语义分割的性能，并可即插即用地增强现有方法。'}


### 论文摘要

3D hierarchical semantic segmentation (3DHS) is crucial for embodied intelligence applications that demand a multi-grained and multi-hierarchy understanding of 3D scenes. Despite the progress, previous 3DHS methods have overlooked following two challenges: I) multi-label learning with a parameter-sharing model can lead to multi-hierarchy conflicts in cross-hierarchy optimization, and II) the class imbalance issue is inevitable across multiple hierarchies of 3D scenes, which makes the model performance become dominated by major classes. To address these issues, we propose a novel framework with a primary 3DHS branch and an auxiliary discrimination branch. Specifically, to alleviate the multi-hierarchy conflicts, we propose a late-decoupled 3DHS framework which employs multiple decoders with the coarse-to-fine hierarchical guidance and consistency. The late-decoupled architecture can mitigate the underfitting and overfitting conflicts among multiple hierarchies and can also constrain the class imbalance problem in each individual hierarchy. Moreover, we introduce a 3DHS-oriented semantic prototype based bi-branch supervision mechanism, which additionally learns class-wise discriminative point cloud features and performs mutual supervision between the auxiliary and 3DHS branches, to enhance the class-imbalance segmentation. Extensive experiments on multiple datasets and backbones demonstrate that our approach achieves state-of-the-art 3DHS performance, and its core components can also be used as a plug-and-play enhancement to improve previous methods.

---

## 7. NaTex: Seamless Texture Generation as Latent Color Diffusion

**论文链接:** [http://arxiv.org/abs/2511.16317v1](http://arxiv.org/abs/2511.16317v1)

**作者:** Zeqiang Lai, Yunfei Zhao, Zibo Zhao, Xin Yang, Xin Huang, Jingwei Huang, Xiangyu Yue, Chunchao Guo

**发布时间:** 2025-11-20

**备注:** Technical Report

### GPT解析

### 总结

NaTex是一个原生纹理生成框架，直接在3D空间中预测纹理颜色，避免了传统多视图扩散模型(MVDs)的局限性，通过将纹理视为密集颜色点云的新范式实现了更好的纹理连贯性和对齐效果。

### 背景

传统纹理生成方法依赖将几何条件多视图扩散模型合成的2D多视图图像进行烘焙，存在处理遮挡区域困难、难以实现精确网格纹理对齐、难以保持跨视图一致性和连贯性等问题。

### 目的

开发一种能够直接在3D空间中生成纹理的方法，解决传统方法中的固有局限性，实现更好的纹理质量、对齐效果和跨视图一致性。

### 方法

提出潜在颜色扩散方法，包括几何感知的颜色点云VAE和多控制扩散变换器(DiT)，完全使用3D数据从头训练；引入原生几何控制，通过位置嵌入和几何潜在条件将DiT直接条件化于3D空间信息；共同设计VAE-DiT架构，通过专用几何分支与颜色VAE紧密耦合提供精细表面指导。

### 主要发现

NaTex在纹理连贯性和对齐方面显著优于先前方法；展示了强大的泛化能力，无需训练或通过简单调适即可用于各种下游应用，如材料生成、纹理细化以及部分分割和纹理化。

### 结论

NaTex通过将纹理视为密集颜色点云的新范式，成功解决了传统纹理生成方法的多个局限性，为3D纹理生成提供了更有效、更高质量的解决方案。

### 翻译

我们提出了NaTex，一个原生纹理生成框架，可以直接在3D空间中预测纹理颜色。与之前依赖将几何条件多视图扩散模型(MVDs)合成的2D多视图图像进行烘焙的方法不同，NaTex避免了MVD流程的几个固有局限性。这些包括处理需要修复的遮挡区域的困难、实现沿边界的精确网格纹理对齐、以及保持内容和颜色强度在跨视图间的一致性和连贯性。NaTex采用新范式，将纹理视为密集颜色点云，解决了上述问题。基于这一思想，我们提出了潜在颜色扩散，包括几何感知的颜色点云VAE和多控制扩散变换器(DiT)，完全使用3D数据从头训练，用于纹理重建和生成。为实现精确对齐，我们引入了原生几何控制，通过位置嵌入和几何潜在将DiT条件化于直接3D空间信息。我们共同设计了VAE-DiT架构，其中几何潜在通过专用几何分支提取，与颜色VAE紧密耦合，提供保持与纹理强对应关系的精细表面指导。凭借这些设计，NaTex展示了强大的性能，在纹理连贯性和对齐方面显著优于先前方法。此外，NaTex还表现出强大的泛化能力，无需训练或通过简单调适即可用于各种下游应用，例如材料生成、纹理细化以及部分分割和纹理化。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D纹理生成中的三个关键问题：处理遮挡区域、实现纹理与几何边界的精确对齐、以及保持多视角间的一致性和连贯性。这些问题在现实中非常重要，因为手动创建纹理是计算机图形学中的一个瓶颈过程，既耗时又需要专业知识，而高质量的纹理直接影响从电影视觉效果到沉浸式虚拟世界的一切视觉保真度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有多视角纹理生成方法的根本局限性，认识到这些问题源于3D到2D模态转换带来的固有错误。他们提出将3D纹理作为'一等公民'直接在3D空间中生成，避免中间表示带来的问题。设计上借鉴了扩散模型在图像、视频和3D形状生成中的成功应用，参考了3DShape2VecSet的VAE架构但针对颜色点云进行了修改，同时采用了类似DiT的架构但进行了多控制条件的适配。核心创新在于协同设计了VAE-DiT架构，使几何分支与颜色VAE紧密耦合，提供细粒度的表面指导。", '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D纹理表示为3D空间中的密集颜色点云，形成连续的颜色场，通过潜在颜色扩散直接在3D空间中预测纹理颜色，避免多视角投影和烘焙过程。整体流程包括：1)使用几何感知的颜色VAE对颜色点云进行编码和解码，包含并行的几何分支和颜色分支；2)设计多控制颜色DiT，整合图像、几何和颜色条件；3)通过流匹配损失进行训练；4)应用时根据不同任务调整控制条件，实现纹理生成、材质生成、纹理细化等多种功能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)原生3D纹理生成，直接在3D空间生成纹理而非依赖2D多视角烘焙；2)几何感知的颜色VAE，提供细粒度表面指导；3)多控制颜色DiT，灵活整合多种控制信号；4)原生几何控制，实现精确纹理-几何对齐。相比之前工作，NaTex避免了MVD方法的遮挡处理难题，实现了更精确的边界对齐和更好的多视角一致性；与其他原生3D方法相比，它采用了潜在扩散模型范式，提供更强的几何指导，支持更广泛的下游应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'NaTex通过直接在3D空间中生成纹理作为密集颜色点云的潜在扩散模型，解决了传统多视角纹理生成中的核心挑战，实现了高质量、高效率的3D纹理生成，并展现出强大的泛化能力。'}


### 论文摘要

We present NaTex, a native texture generation framework that predicts texture color directly in 3D space. In contrast to previous approaches that rely on baking 2D multi-view images synthesized by geometry-conditioned Multi-View Diffusion models (MVDs), NaTex avoids several inherent limitations of the MVD pipeline. These include difficulties in handling occluded regions that require inpainting, achieving precise mesh-texture alignment along boundaries, and maintaining cross-view consistency and coherence in both content and color intensity. NaTex features a novel paradigm that addresses the aforementioned issues by viewing texture as a dense color point cloud. Driven by this idea, we propose latent color diffusion, which comprises a geometry-awared color point cloud VAE and a multi-control diffusion transformer (DiT), entirely trained from scratch using 3D data, for texture reconstruction and generation. To enable precise alignment, we introduce native geometry control that conditions the DiT on direct 3D spatial information via positional embeddings and geometry latents. We co-design the VAE-DiT architecture, where the geometry latents are extracted via a dedicated geometry branch tightly coupled with the color VAE, providing fine-grained surface guidance that maintains strong correspondence with the texture. With these designs, NaTex demonstrates strong performance, significantly outperforming previous methods in texture coherence and alignment. Moreover, NaTex also exhibits strong generalization capabilities, either training-free or with simple tuning, for various downstream applications, e.g., material generation, texture refinement, and part segmentation and texturing.

---

## 8. Dataset Distillation for Pre-Trained Self-Supervised Vision Models

**论文链接:** [http://arxiv.org/abs/2511.16674v1](http://arxiv.org/abs/2511.16674v1)

**作者:** George Cazenavette, Antonio Torralba, Vincent Sitzmann

**发布时间:** 2025-11-20

**备注:** Accepted at NeurIPS 2025. Project page: https://linear-gradient-matching.github.io/ Code: https://github.com/GeorgeCazenavette/linear-gradient-matching

### GPT解析

### 总结

数据集蒸馏任务旨在找到一组合成图像，使得在这些合成图像上训练的模型能够与在更大的真实数据集上训练的模型达到相同的性能。本文研究如何蒸馏数据集以优化在大型预训练视觉模型上训练线性探测器的性能。

### 背景

现有数据集蒸馏方法主要关注合成能够训练随机初始化模型的数据集，而最先进的视觉方法越来越多地构建在大型预训练自监督模型上，而不是从头开始训练。

### 目的

研究如何蒸馏数据集，以便在大型预训练视觉模型上最优地训练线性探测器，目标是合成能够在预训练特征提取器上产生与真实数据相似梯度的合成图像。

### 方法

提出了一种名为'线性梯度匹配'(Linear Gradient Matching)的数据集蒸馏方法，优化合成图像使得当它们通过预训练的特征提取器时，它们在线性分类器中产生的梯度与真实数据产生的梯度相似。

### 主要发现

该方法产生的合成数据优于所有真实图像基线；合成数据能够跨预训练视觉模型泛化；蒸馏的数据集在细粒度分类方面特别有效；为模型可解释性提供了有价值的工具。

### 结论

线性梯度匹配方法为预训练视觉模型提供了有效的数据集蒸馏解决方案，不仅优于现有方法，还具有跨模型泛化能力和多种应用场景，包括细粒度分类和模型可解释性。

### 翻译

数据集蒸馏任务旨在找到一组合成图像，使得在这些合成图像上训练的模型能够与在更大的真实数据集上训练的模型达到相同的性能。现有的蒸馏方法主要关注合成能够训练随机初始化模型的数据集。相比之下，最先进的视觉方法越来越多地构建在大型预训练自监督模型上，而不是从头开始训练。在本文中，我们研究了如何蒸馏数据集，以便在大型预训练视觉模型上最优地训练线性探测器。我们为此任务引入了一种名为线性梯度匹配的数据集蒸馏方法，优化合成图像使得当它们通过预训练的特征提取器时，它们在线性分类器中产生的梯度与真实数据产生的梯度相似。我们的方法产生的合成数据优于所有真实图像基线，并且显著地能够跨预训练视觉模型泛化，例如使我们能够使用通过DINO主干网络蒸馏的数据集训练具有竞争力的线性CLIP探测器。此外，我们表明我们的蒸馏数据集在细粒度分类方面特别有效，并为模型可解释性提供了有价值的工具，可以预测在柏拉图表示假设下两个模型的嵌入空间有多相似，或者模型是否对对抗数据集中的虚假相关性敏感。


### 论文摘要

The task of dataset distillation aims to find a small set of synthetic images such that training a model on them reproduces the performance of the same model trained on a much larger dataset of real samples. Existing distillation methods focus on synthesizing datasets that enable training randomly initialized models. In contrast, state-of-the-art vision approaches are increasingly building on large, pre-trained self-supervised models rather than training from scratch. In this paper, we investigate the problem of distilling datasets that enable us to optimally train linear probes on top of such large, pre-trained vision models. We introduce a method of dataset distillation for this task called Linear Gradient Matching that optimizes the synthetic images such that, when passed through a pre-trained feature extractor, they induce gradients in the linear classifier similar to those produced by the real data. Our method yields synthetic data that outperform all real-image baselines and, remarkably, generalize across pre-trained vision models, enabling us, for instance, to train a linear CLIP probe that performs competitively using a dataset distilled via a DINO backbone. Further, we show that our distilled datasets are exceptionally effective for fine-grained classification and provide a valuable tool for model interpretability, predicting, among other things, how similar two models' embedding spaces are under the platonic representation hypothesis or whether a model is sensitive to spurious correlations in adversarial datasets.

---

## 9. EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards

**论文链接:** [http://arxiv.org/abs/2511.16672v1](http://arxiv.org/abs/2511.16672v1)

**作者:** Omkat Thawakar, Shravan Venkatraman, Ritesh Thawkar, Abdelrahman Shaker, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Fahad Khan

**发布时间:** 2025-11-20

**备注:** 9 Pages, 6 Figures, 4 Tables

### GPT解析

### 总结

本文提出了一种名为EvoLMM的自进化框架，通过两个合作代理在完全无监督的方式下提升大型多模态模型的推理能力，无需人工注释数据或奖励模型。

### 背景

大型多模态模型(LMMs)的最新进展使其具有令人印象深刻的推理和感知能力，但大多数现有的训练流程仍然依赖人工整理的数据或外部验证的奖励模型，限制了它们的自主性和可扩展性。

### 目的

以完全无监督的方式提高LMM的推理能力，不使用任何注释数据或奖励蒸馏。

### 方法

提出EvoLMM框架，从单一骨干模型实例化两个合作代理：Proposer生成多样化、基于图像的问题，Solver通过内部一致性解决问题，学习通过持续的自奖励过程进行。

### 主要发现

当使用Qwen2.5-VL作为基础模型时，仅使用原始训练图像，EvoLMM在ChartQA、MathVista和MathVision等多模态数学推理基准测试上取得了高达约3%的一致性提升。

### 结论

这个简单而有效的方法将成为未来在完全无监督方式下研究自我改进LMM的坚实基础。

### 翻译

大型多模态模型(LMMs)的最新进展使其具有令人印象深刻的推理和感知能力，但大多数现有的训练流程仍然依赖人工整理的数据或外部验证的奖励模型，限制了它们的自主性和可扩展性。在这项工作中，我们努力以完全无监督的方式提高LMM的推理能力（不使用任何注释数据或奖励蒸馏）。为此，我们提出了一个名为EvoLMM的自进化框架，该框架从单一骨干模型实例化两个合作代理：Proposer生成多样化、基于图像的问题，Solver通过内部一致性解决问题，学习通过持续的自奖励过程进行。这种动态反馈鼓励生成信息丰富的查询和结构化推理的改进，而不依赖于真实标签或人工判断。当使用流行的Qwen2.5-VL作为基础模型时，我们的EvoLMM仅使用原始训练图像，在ChartQA、MathVista和MathVision等多模态数学推理基准测试上取得了高达约3%的一致性提升。我们希望这个简单而有效的方法将成为未来在完全无监督方式下研究自我改进LMM的坚实基础。我们的代码和模型可在https://github.com/mbzuai-oryx/EvoLMM获取。


### 论文摘要

Recent advances in large multimodal models (LMMs) have enabled impressive reasoning and perception abilities, yet most existing training pipelines still depend on human-curated data or externally verified reward models, limiting their autonomy and scalability. In this work, we strive to improve LMM reasoning capabilities in a purely unsupervised fashion (without any annotated data or reward distillation). To this end, we propose a self-evolving framework, named EvoLMM, that instantiates two cooperative agents from a single backbone model: a Proposer, which generates diverse, image-grounded questions, and a Solver, which solves them through internal consistency, where learning proceeds through a continuous self-rewarding process. This dynamic feedback encourages both the generation of informative queries and the refinement of structured reasoning without relying on ground-truth or human judgments. When using the popular Qwen2.5-VL as the base model, our EvoLMM yields consistent gains upto $\sim$3\% on multimodal math-reasoning benchmarks, including ChartQA, MathVista, and MathVision, using only raw training images. We hope our simple yet effective approach will serve as a solid baseline easing future research in self-improving LMMs in a fully-unsupervised fashion. Our code and models are available at https://github.com/mbzuai-oryx/EvoLMM.

---

## 10. Video-as-Answer: Predict and Generate Next Video Event with Joint-GRPO

**论文链接:** [http://arxiv.org/abs/2511.16669v1](http://arxiv.org/abs/2511.16669v1)

**作者:** Junhao Cheng, Liang Hou, Xin Tao, Jing Liao

**发布时间:** 2025-11-20

**备注:** Project page: https://video-as-answer.github.io/

### GPT解析

### 总结

该论文提出了VNEP任务，将视频作为一种新的回答模态扩展到下一事件预测中，并开发了VANS模型通过强化学习对齐视觉语言模型与视频扩散模型，实现了在视频事件预测和可视化方面的最先进性能。

### 背景

语言模型在许多实际应用中具有重要影响，但视频生成主要局限于娱乐领域。视频能够展示难以仅通过语言传达的物理世界信息，例如仅通过文本教授打领带这样的程序性任务。

### 目的

将视频作为一种新的回答模态扩展到下一事件预测(NEP)任务中，形式化为视频下一事件预测(VNEP)。从传统的'讲述'转向'展示'，为程序学习和创造性探索提供更直观和定制的答案。

### 方法

引入VANS模型，利用强化学习将视觉语言模型(VLM)与视频扩散模型(VDM)对齐。提出Joint-GRPO方法协调VLM和VDM作为一个单元运作，通过共享奖励优化VLM生成准确且易于可视化的标题，同时指导VDM生成忠实于这些标题和输入视觉上下文的视频。构建了VANS-Data-100K专用数据集。

### 主要发现

在程序性和预测性基准实验中，VANS模型在视频事件预测和可视化方面取得了最先进的性能，有效解决了多模态输入理解、基于指令的推理以及视觉语义一致性视频生成等挑战。

### 结论

VANS模型成功将视频作为回答模态引入下一事件预测任务，通过协调视觉语言模型与视频扩散模型，实现了更直观和定制化的程序学习与创造性探索答案。

### 翻译

虽然语言模型在许多实际应用中具有重要影响，但视频生成主要仍局限于娱乐领域。受视频能够展示难以仅通过语言单独传达的物理世界信息的启发(例如，仅通过文本想象教人打领带)，我们确定了将视频扩展为下一事件预测(NEP)新回答模态的未充分利用机会，形式化为视频下一事件预测(VNEP)。虽然既定的NEP任务将包含程序性或预测性问题的视频作为输入，以文本形式预测下一事件，但VNEP需要动态视频响应。这种从讲述到展示的转变，为程序学习和创造性探索解锁了更直观和定制的答案。然而，现有模型在此任务上仍面临挑战，因为它需要理解多模态输入、基于指令的推理，以及生成具有视觉和语义一致性的视频。为此，我们引入了VANS模型，该模型利用强化学习将视觉语言模型(VLM)与视频扩散模型(VDM)对齐，用于VNEP。VANS的核心是我们提出的Joint-GRPO，它协调VLM和VDM作为一个单元运作。通过在各自输出上的共享奖励驱动，它优化VLM生成既准确又易于可视化的标题，同时指导VDM生成忠实于这些标题和输入视觉上下文的视频。为实现这种学习，我们专门为VNEP任务构建了VANS-Data-100K数据集。在程序性和预测性基准上的实验表明，VANS在视频事件预测和可视化方面均取得了最先进的性能。代码已在https://github.com/KlingTeam/VANS发布。


### 论文摘要

While language models have become impactful in many real-world applications, video generation remains largely confined to entertainment. Motivated by video's inherent capacity to demonstrate physical-world information that is difficult to convey through language alone (e.g., imagine teaching someone to tie a tie using only text), we identify an underutilized opportunity to extend video as a new answer modality for Next-Event Prediction (NEP), formalized as Video-Next-Event Prediction (VNEP). While the established NEP task takes a video with a procedural or predictive question as input to predict the next event in text, VNEP requires dynamic video responses. This shift from telling to showing unlocks more intuitive and customized answers for procedural learning and creative exploration. However, this task remains challenging for existing models, as it demands an understanding of multimodal input, instruction-conditioned reasoning, and the generation of video with visual and semantic consistency. To address this, we introduce VANS, a model that leverages reinforcement learning to align a Vision-Language Model (VLM) with a Video Diffusion Model (VDM) for VNEP. The core of VANS is our proposed Joint-GRPO that orchestrates the VLM and VDM to function as a unit. Driven by a shared reward on their respective output, it optimizes the VLM to produce captions that are both accurate and friendly to visualize, while guiding the VDM to generate videos that are faithful to these captions and the input visual context. To enable this learning, we craft VANS-Data-100K, a dedicated dataset for the VNEP task. Experiments on procedural and predictive benchmarks demonstrate that VANS achieves state-of-the-art performance in both video event prediction and visualization. Codes are released in https://github.com/KlingTeam/VANS.

---

## 11. Cognitive Foundations for Reasoning and Their Manifestation in LLMs

**论文链接:** [http://arxiv.org/abs/2511.16660v1](http://arxiv.org/abs/2511.16660v1)

**作者:** Priyanka Kargupta, Shuyue Stella Li, Haocheng Wang, Jinu Lee, Shan Chen, Orevaoghene Ahia, Dean Light, Thomas L. Griffiths, Max Kleiman-Weiner, Jiawei Han, Asli Celikyilmaz, Yulia Tsvetkov

**发布时间:** 2025-11-20

**备注:** 40 pages, 4 tables, 6 figures

### GPT解析

### 总结

本研究通过认知科学视角分析了大型语言模型与人类推理的差异，提出了一个包含28个认知元素的分类法，并开发了一种细粒度认知评估框架，发现模型与人类在推理结构上存在系统性差异，基于这些差异开发了测试时推理指导方法，显著提升了模型性能。

### 背景

大型语言模型能够解决复杂问题，但在处理更简单的变体时却失败，这表明它们通过 fundamentally不同于人类推理的机制来获得正确输出。

### 目的

将认知科学研究综合到一个包含28个认知元素的分类法中，涵盖计算约束、元认知控制、知识表示和转换操作，并分析它们在推理轨迹中的行为表现，为开发更接近人类推理机制的模型奠定基础。

### 方法

提出细粒度认知评估框架；分析17个跨文本、视觉和音频模态的170K条模型轨迹及54条人类思维出声轨迹；对1598篇LLM推理论文进行元分析；基于发现的模式开发测试时推理指导方法。

### 主要发现

人类使用分层嵌套和元认知监控，而模型依赖浅层前向链接，这种差异在非结构化问题上最明显；研究社区关注易于量化的行为（顺序组织：55%，分解：60%），而忽视与成功相关的元认知控制（自我意识：16%，评估：8%）；模型拥有与成功相关的行为库但无法自发部署。

### 结论

通过弥合认知科学与LLM研究，为开发通过原则性认知机制而非虚假推理捷径或记忆来推理的模型奠定基础，为改进模型能力和大规模测试人类认知理论开辟新方向。

### 翻译

大型语言模型能够解决复杂问题，但在处理更简单的变体时却失败，这表明它们通过 fundamentally不同于人类推理的机制来获得正确输出。我们将认知科学研究综合到一个包含28个认知元素的分类法中，这些元素涵盖计算约束、元认知控制、知识表示和转换操作，然后分析它们在推理轨迹中的行为表现。我们提出了一个细粒度的认知评估框架，并对来自17个跨文本、视觉和音频模态的170K条轨迹以及54条人类思维出声轨迹进行了首次大规模分析，这些数据我们已公开。我们的分析揭示了系统性结构差异：人类采用分层嵌套和元认知监控，而模型依赖浅层前向链接，这种差异在非结构化问题上最为明显。对1598篇LLM推理论文的元分析显示，研究社区集中在易于量化的行为（顺序组织：55%，分解：60%），而忽视了与成功相关的元认知控制（自我意识：16%，评估：8%）。模型拥有与成功相关的行为库，但无法自发部署它们。利用这些模式，我们开发了测试时推理指导，自动构建成功结构，将复杂问题上的性能提高了高达60%。通过弥合认知科学与LLM研究，我们为开发通过原则性认知机制而非脆弱的虚假推理捷径或记忆来推理的模型奠定了基础，为改进模型能力和大规模测试人类认知理论开辟了新方向。


### 论文摘要

Large language models solve complex problems yet fail on simpler variants, suggesting they achieve correct outputs through mechanisms fundamentally different from human reasoning. We synthesize cognitive science research into a taxonomy of 28 cognitive elements spanning computational constraints, meta-cognitive controls, knowledge representations, and transformation operations, then analyze their behavioral manifestations in reasoning traces. We propose a fine-grained cognitive evaluation framework and conduct the first large-scale analysis of 170K traces from 17 models across text, vision, and audio modalities, alongside 54 human think-aloud traces, which we make publicly available. Our analysis reveals systematic structural differences: humans employ hierarchical nesting and meta-cognitive monitoring while models rely on shallow forward chaining, with divergence most pronounced on ill-structured problems. Meta-analysis of 1,598 LLM reasoning papers reveals the research community concentrates on easily quantifiable behaviors (sequential organization: 55%, decomposition: 60%) while neglecting meta-cognitive controls (self-awareness: 16%, evaluation: 8%) that correlate with success. Models possess behavioral repertoires associated with success but fail to deploy them spontaneously. Leveraging these patterns, we develop test-time reasoning guidance that automatically scaffold successful structures, improving performance by up to 60% on complex problems. By bridging cognitive science and LLM research, we establish a foundation for developing models that reason through principled cognitive mechanisms rather than brittle spurious reasoning shortcuts or memorization, opening new directions for both improving model capabilities and testing theories of human cognition at scale.

---

## 12. InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy

**论文链接:** [http://arxiv.org/abs/2511.16651v1](http://arxiv.org/abs/2511.16651v1)

**作者:** Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, Zherui Qiu, Feng Yuan, Yaping Li, Ping Wang, Junhao Cai, Jia Zeng, Hao Dong, Jiangmiao Pang

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究证明仅使用大规模合成数据可以匹敌最强真实数据集在预训练视觉-语言-行动模型方面的性能，并展现出令人惊讶的零样本模拟到现实迁移能力。

### 背景

近期研究探索真实数据和合成数据对视觉-语言-行动模型泛化的贡献。当前模型已显示大规模真实机器人预训练的有效性，但合成数据此前未能在同等规模展示可比性能。

### 目的

提供证据证明合成数据单独可匹敌最强π数据集在预训练视觉-语言-行动模型方面的性能，揭示大规模模拟的价值。

### 方法

创建包含630k+轨迹和7,433小时数据的InternData-A1数据集，涵盖4种具身、18种技能、70项任务和227个场景，支持刚性、关节式、可变形和流体物体操作。通过高度自主、完全解耦和组合式模拟管道生成，支持长时程技能组合、灵活任务组装和异构具身。使用与π0相同架构，完全在合成数据上预训练模型。

### 主要发现

仅使用合成数据预训练的模型匹敌官方π0模型在49个模拟任务、5个真实世界任务和4个长时程灵巧任务上的性能，并在多个挑战性任务上表现出令人惊讶的零样本模拟到现实迁移能力。

### 结论

合成数据单独可匹敌最强真实数据集性能，揭示大规模模拟的显著价值。研究将发布数据集并开源生成管道，扩大对大规模机器人数据的访问，降低具身AI研究中可扩展数据创建的门槛。

### 翻译

近期的研究探索了真实数据和合成数据如何贡献于视觉-语言-行动模型的泛化能力。虽然当前的视觉-语言-行动模型已经显示出大规模真实机器人预训练的强大有效性，但合成数据此前尚未在同等规模上展示出可比的性能。本文首次提供了证据，证明仅使用合成数据可以匹敌最强π数据集在预训练视觉-语言-行动模型方面的性能，揭示了大规模模拟的显著价值。 resulting模型还在几个具有挑战性的任务上表现出令人惊讶的零样本模拟到现实迁移能力。我们的合成数据集InternData-A1包含超过63万个轨迹和7,433小时的数据，涵盖4种具身、18种技能、70项任务和227个场景，包括对刚性、关节式、可变形和流体物体的操作。它通过高度自主、完全解耦和组合式的模拟管道生成，支持长时程技能组合、灵活任务组装和异构具身，只需最少的手动调整。使用与π0相同的架构，我们在InternData-A1上完全预训练了一个模型，发现它在49个模拟任务、5个真实世界任务和4个长时程灵巧任务上与官方π0性能相当。我们将发布该数据集，并将开源生成管道，以扩大对大规模机器人数据的访问渠道，降低具身AI研究中可扩展数据创建的门槛。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的问题是：能否仅使用大规模高保真合成数据来训练视觉-语言-动作模型，使其性能媲美使用真实机器人数据训练的模型。这个问题在现实中非常重要，因为收集大规模真实机器人数据成本高昂，需要专业操作员、特殊硬件和大量人力劳动，大多数研究团队难以获取足够规模的多样化数据集。此外，现有模拟数据集覆盖的技能集狭窄，主要关注刚体对象，需要大量人工操作，严重限制了具身AI研究的进展。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到真实机器人数据收集的高成本和局限性，发现现有模拟数据集在技能多样性、物体类型覆盖和自动化程度方面的不足。基于这些观察，他们提出假设：高保真合成数据在足够大的规模下可以匹配最强真实世界数据用于VLA预训练。他们设计了一个高度自主、完全解耦和组合式的模拟管道。该方法借鉴了现有模拟数据集的经验，如MimicGen、ManiSkill2等，采用了模块化的原子技能设计，并优化了数据生成流程，如使用CuRobo运动规划器等技术。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过大规模、高保真、多样化的合成数据来训练VLA模型，使其性能媲美使用真实机器人数据训练的模型。整体实现流程包括：1) 环境构建：从资产库中检索机器人、场景和对象；2) 技能组合：从技能库中选择原子技能组合成完整任务；3) 域随机化：增强视觉和轨迹多样性；4) 生成与存储：使用CuRobo运动规划器插值动作并记录数据；5) 框架优化：通过阶段解耦、动态资源调度等提高效率。整个流程实现了高度自动化的数据生成，支持长距离技能组合、灵活任务组装和异构embodiment。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 规模和多样性：包含630k轨迹和7,433小时数据，覆盖4种embodiment、18种技能、70种任务和227种场景，涵盖刚体、关节、可变形和流体物体的操作；2) 高度自动化的数据生成管道：完全解耦、组合式的模拟管道，最小化手动调整；3) 性能表现：首次证明仅使用合成数据训练的VLA模型可以匹敌使用最强真实数据集训练的模型；4) 数据效率：在sim-to-real转移中，少于1,600个模拟样本可以匹配200个真实样本的性能。相比之前的工作，InternData-A1提供了更广泛的任务多样性、异构embodiment、照片级真实感渲染和长距离多技能轨迹，同时实现了更高的自动化程度和更低的成本。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文贡献了一种大规模、高保真合成数据集及其完全自动化的生成管道，首次证明仅使用合成数据训练的VLA模型可以达到与使用真实机器人数据训练的最先进模型相媲美的性能，大大降低了机器人学习的门槛并促进了具身AI研究的发展。'}


### 论文摘要

Recent works explore how real and synthetic data contribute to Vision-Language-Action (VLA) models' generalization. While current VLA models have shown the strong effectiveness of large-scale real-robot pre-training, synthetic data has not previously demonstrated comparable capability at scale. This paper provides the first evidence that synthetic data alone can match the performance of the strongest $π$-dataset in pre-training a VLA model, revealing the substantial value of large-scale simulation. The resulting model also exhibits surprisingly zero-shot sim-to-real transfer on several challenging tasks. Our synthetic dataset, InternData-A1, contains over 630k trajectories and 7,433 hours across 4 embodiments, 18 skills, 70 tasks, and 227 scenes, covering rigid, articulated, deformable, and fluid-object manipulation. It is generated through a highly autonomous, fully decoupled, and compositional simulation pipeline that enables long-horizon skill composition, flexible task assembly, and heterogeneous embodiments with minimal manual tuning. Using the same architecture as $π_0$, we pre-train a model entirely on InternData-A1 and find that it matches the official $π_0$ across 49 simulation tasks, 5 real-world tasks, and 4 long-horizon dexterous tasks. We release the dataset and will open-source the generation pipeline to broaden access to large-scale robotic data and to lower the barrier to scalable data creation for embodied AI research.

---

## 13. SAM2S: Segment Anything in Surgical Videos via Semantic Long-term Tracking

**论文链接:** [http://arxiv.org/abs/2511.16618v1](http://arxiv.org/abs/2511.16618v1)

**作者:** Haofeng Liu, Ziyue Wang, Sudhanshu Mishra, Mingqi Gao, Guanyi Qin, Chang Han Low, Alex Y. W. Kong, Yueming Jin

**发布时间:** 2025-11-20

**备注:** 11 pages, 4 figures

### GPT解析

### 总结

本研究提出了SA-SV基准数据集和SAM2S模型，解决了手术视频分割中长期跟踪和零样本泛化的挑战，显著提升了性能并保持实时处理能力。

### 背景

手术视频分割对计算机辅助手术至关重要，能够实现器械和组织的精确定位和跟踪。交互式视频目标分割模型如SAM2提供了基于提示的灵活性，但在手术场景中面临领域差距和有限长期跟踪的挑战。

### 目的

构建一个全面的基准数据集来促进手术视频分割中长期跟踪和零样本泛化的发展与评估，并提出一种增强模型来解决现有方法的局限性。

### 方法

构建SA-SV最大的手术iVOS基准数据集，包含61k帧和1.6k masklets的实例级时空标注；提出SAM2S模型，通过DiveMem记忆机制、时序语义学习和抗歧义学习增强SAM2性能。

### 主要发现

在SA-SV上微调使SAM2性能提升12.99平均J&F；SAM2S达到80.42平均J&F，比原始和微调SAM2分别高17.10和4.11点，同时保持68 FPS实时推理和强零样本泛化能力。

### 结论

SA-SV基准和SAM2S模型有效解决了手术视频分割中的关键挑战，为计算机辅助手术提供了更精确的器械和组织定位跟踪能力。

### 翻译

手术视频分割对计算机辅助手术至关重要，能够实现器械和组织的精确定位和跟踪。交互式视频目标分割模型如SAM2提供了基于提示的灵活性，超越了具有预定义类别的方法，但在手术场景中面临领域差距和有限长期跟踪的挑战。为此，我们构建了SA-SV，这是最大的手术iVOS基准数据集，包含实例级时空标注，涵盖八种手术类型，为长期跟踪和零样本泛化的发展与评估提供了全面基础。基于SA-SV，我们提出了SAM2S模型，通过DiveMem记忆机制、时序语义学习和抗歧义学习增强SAM2性能。实验证明，在SA-SV上微调显著提升了性能，SAM2S进一步提高了性能指标，同时保持实时处理能力和零样本泛化能力。代码和数据集将公开发布。


### 论文摘要

Surgical video segmentation is crucial for computer-assisted surgery, enabling precise localization and tracking of instruments and tissues. Interactive Video Object Segmentation (iVOS) models such as Segment Anything Model 2 (SAM2) provide prompt-based flexibility beyond methods with predefined categories, but face challenges in surgical scenarios due to the domain gap and limited long-term tracking. To address these limitations, we construct SA-SV, the largest surgical iVOS benchmark with instance-level spatio-temporal annotations (masklets) spanning eight procedure types (61k frames, 1.6k masklets), enabling comprehensive development and evaluation for long-term tracking and zero-shot generalization. Building on SA-SV, we propose SAM2S, a foundation model enhancing \textbf{SAM2} for \textbf{S}urgical iVOS through: (1) DiveMem, a trainable diverse memory mechanism for robust long-term tracking; (2) temporal semantic learning for instrument understanding; and (3) ambiguity-resilient learning to mitigate annotation inconsistencies across multi-source datasets. Extensive experiments demonstrate that fine-tuning on SA-SV enables substantial performance gains, with SAM2 improving by 12.99 average $\mathcal{J}$\&$\mathcal{F}$ over vanilla SAM2. SAM2S further advances performance to 80.42 average $\mathcal{J}$\&$\mathcal{F}$, surpassing vanilla and fine-tuned SAM2 by 17.10 and 4.11 points respectively, while maintaining 68 FPS real-time inference and strong zero-shot generalization. Code and dataset will be released at https://jinlab-imvr.github.io/SAM2S.

---

## 14. Almost Sure Convergence Analysis of Differentially Private Stochastic Gradient Methods

**论文链接:** [http://arxiv.org/abs/2511.16587v1](http://arxiv.org/abs/2511.16587v1)

**作者:** Amartya Mukherjee, Jun Liu

**发布时间:** 2025-11-20

**备注:** 6 pages

### GPT解析

### 总结

本研究证明了差分隐私随机梯度下降(DP-SGD)在标准平滑假设下几乎必然收敛，适用于非凸和强凸情况，并扩展到了动量变体如随机重球(DP-SHB)和Nesterov加速梯度(DP-NAG)。

### 背景

差分隐私随机梯度下降(DP-SGD)已成为训练具有严格隐私保证的机器学习模型的标准算法，尽管被广泛使用，但对DP-SGD长期行为的理论理解仍然有限，现有分析通常只建立期望收敛或高概率收敛，而不解决单个轨迹的几乎必然收敛问题。

### 目的

证明DP-SGD在标准平滑假设下几乎必然收敛，分析适用于非凸和强凸情况，并将分析扩展到动量变体。

### 方法

在步长满足标准衰减条件下，证明DP-SGD几乎必然收敛，为动量变体构建仔细的能量构造以获得类似保证。

### 主要发现

DP-SGD在标准平滑假设下几乎必然收敛，这一结果适用于非凸和强凸情况，动量变体如DP-SHB和DP-NAG也具有类似的保证。

### 结论

这些结果为差分隐私优化提供了更强的理论基础，表明尽管存在隐私引起的失真，但该算法在凸和非凸情况下仍保持路径稳定性。

### 翻译

差分隐私随机梯度下降(DP-SGD)已成为训练具有严格隐私保证的机器学习模型的标准算法。尽管被广泛使用，但对其长期行为的理论理解仍然有限：现有分析通常建立期望收敛或高概率收敛，但不解决单个轨迹的几乎必然收敛问题。在这项工作中，我们证明在标准平滑假设下，DP-SGD几乎必然收敛，无论是在非凸还是强凸情况下，只要步长满足某些标准衰减条件。我们的分析扩展到动量变体，如随机重球(DP-SHB)和Nesterov加速梯度(DP-NAG)，我们展示了仔细的能量构造可以产生类似的保证。这些结果为差分隐私优化提供了更强的理论基础，并表明尽管存在隐私引起的失真，但该算法在凸和非凸情况下仍保持路径稳定性。


### 论文摘要

Differentially private stochastic gradient descent (DP-SGD) has become the standard algorithm for training machine learning models with rigorous privacy guarantees. Despite its widespread use, the theoretical understanding of its long-run behavior remains limited: existing analyses typically establish convergence in expectation or with high probability, but do not address the almost sure convergence of single trajectories. In this work, we prove that DP-SGD converges almost surely under standard smoothness assumptions, both in nonconvex and strongly convex settings, provided the step sizes satisfy some standard decaying conditions. Our analysis extends to momentum variants such as the stochastic heavy ball (DP-SHB) and Nesterov's accelerated gradient (DP-NAG), where we show that careful energy constructions yield similar guarantees. These results provide stronger theoretical foundations for differentially private optimization and suggest that, despite privacy-induced distortions, the algorithm remains pathwise stable in both convex and nonconvex regimes.

---

## 15. Li-P-S Electrolyte Materials as a Benchmark for Machine-Learned Interatomic Potentials

**论文链接:** [http://arxiv.org/abs/2511.16569v1](http://arxiv.org/abs/2511.16569v1)

**作者:** Natascia L. Fragapane, Volker L. Deringer

**发布时间:** 2025-11-20

### GPT解析

### 总结

研究团队开发了LiPS-25基准数据集及配套性能测试方法，用于评估机器学习原子势模型在固态电解质材料中的表现，该框架可扩展应用于其他材料系统。

### 背景

随着机器学习原子势(MLIP)模型在材料模拟中的可用性日益增加，对稳健、自动化且具有化学洞察力的基准测试方法的需求不断增长。

### 目的

引入LiPS-25基准数据集，开发全面的性能测试套件，评估基于图的MLIP架构在固态电解质材料中的表现，并研究超参数影响和预训练模型的微调行为。

### 方法

构建包含晶体和非晶体构型的Li2S-P2S5伪二元成分线固态电解质材料基准数据集；提出从传统数值误差指标到物理驱动评估任务的性能测试套件；针对基于图的MLIP架构进行数值实验，评估超参数影响和预训练模型的微调行为。

### 主要发现

研究展示了超参数选择对MLIP模型性能的影响，以及预训练模型在固态电解质材料中的微调行为，但没有在摘要中明确列出具体发现数据。

### 结论

LiPS-25基准测试框架及其代码实现不仅适用于Li-P-S固态电解质，还可以很容易地扩展和适应到其他材料系统中。

### 翻译

随着机器学习原子势(MLIP)模型在材料模拟中的可用性日益增加，对稳健、自动化且具有化学洞察力的基准测试方法的需求不断增长。为此，我们在此介绍了LiPS-25，这是一个针对Li2S-P2S5伪二元成分线上典型固态电解质材料的精选基准数据集，包括晶体和非晶体构型。与数据集一起，我们提出了一系列性能测试，范围从传统的数值误差指标到物理驱动的评估任务。重点关注基于图的MLIP架构，我们进行了数值实验，评估了(i)超参数的影响和(ii)选定的预训练('基础')MLIP模型的微调行为。除了Li-P-S固态电解质外，我们预计这样的基准测试及其代码实现可以很容易地适应到其他材料系统中。


### 论文摘要

With the growing availability of machine-learned interatomic potential (MLIP) models for materials simulations, there is an increasing demand for robust, automated, and chemically insightful benchmarking methodologies. In response, we here introduce LiPS-25, a curated benchmark dataset for a canonical series of solid-state electrolyte materials from the Li2S-P2S5 pseudo-binary compositional line, including crystalline and amorphous configurations. Together with the dataset, we present a suite of performance tests that range from conventional numerical error metrics to physically motivated evaluation tasks. With a focus on graph-based MLIP architectures, we run numerical experiments that assess (i) the effect of hyperparameters and (ii) the fine-tuning behavior of selected pre-trained ("foundational") MLIP models. Beyond the Li-P-S solid-state electrolytes, we expect that such benchmarks and their code implementations can be readily adapted to other material systems.

---

## 16. POMA-3D: The Point Map Way to 3D Scene Understanding

**论文链接:** [http://arxiv.org/abs/2511.16567v1](http://arxiv.org/abs/2511.16567v1)

**作者:** Ye Mao, Weixun Luo, Ranran Huang, Junpeng Jing, Krystian Mikolajczyk

**发布时间:** 2025-11-20

**备注:** 11 pages, 6 tables, 5 figures

### GPT解析

### 总结

本文介绍了POMA-3D，这是第一个从点地图(point maps)中学习的自监督3D表示模型。点地图在结构化2D网格上编码显式3D坐标，保留全局3D几何形状，同时兼容2D基础模型输入格式。研究设计了视图到场景对齐策略转移2D先验知识，引入POMA-JEPA架构确保多视图几何一致性，并构建了包含6.5K房间级RGB-D场景和1M 2D图像场景的ScenePoint数据集。实验证明POMA-3D是3D理解任务的强大骨干，可应用于3D问答、具身导航等多种任务，仅使用几何输入即可实现。

### 背景

3D表示学习领域面临预训练先验稀缺和数据有限的挑战，现有方法难以充分利用2D基础模型中已学习的丰富先验知识。

### 目的

开发一种新的3D表示模型，能够从点地图中学习自监督的3D表示，保留全局3D几何形状，同时与2D基础模型兼容，并利用2D先验知识增强3D理解能力。

### 方法

1. 提出POMA-3D，第一个从点地图学习的自监督3D表示模型；2. 设计视图到场景对齐策略转移2D先验知识；3. 引入POMA-JEPA联合嵌入-预测架构确保多视图几何一致性；4. 构建ScenePoint大规模数据集；5. 仅使用几何输入进行多种3D理解任务。

### 主要发现

1. POMA-3D可作为专业和通用3D理解的强大骨干网络；2. 该模型能应用于3D问答、具身导航、场景检索和具身定位等多种任务；3. 所有任务仅使用几何输入即可实现；4. 点地图方法有效解决了3D表示学习中预训练先验稀缺和数据有限的问题。

### 结论

POMA-3D探索了一种创新的点地图方法进行3D场景理解，成功将2D先验知识转移到3D领域，解决了3D表示学习中的数据稀缺和先验知识不足的问题，为3D理解任务提供了强大的基础模型。

### 翻译

在本文中，我们介绍了POMA-3D，这是第一个从点地图中学习的自监督3D表示模型。点地图在结构化的2D网格上编码显式的3D坐标，保留了全局3D几何形状，同时保持与2D基础模型输入格式的兼容性。为了将丰富的2D先验知识转移到POMA-3D中，设计了视图到场景的对齐策略。此外，由于点地图相对于规范空间是视图依赖的，我们引入了POMA-JEPA，这是一种联合嵌入-预测架构，强制跨多个视图的几何一致的点地图特征。此外，我们引入了ScenePoint，这是一个由6.5K个房间级RGB-D场景和1M个2D图像场景构建的点地图数据集，用于促进大规模POMA-3D预训练。实验表明，POMA-3D可以作为专业和通用3D理解的强大骨干网络。它受益于多种任务，包括3D问答、具身导航、场景检索和具身定位，所有这些仅使用几何输入（即3D坐标）即可实现。总体而言，我们的POMA-3D探索了一种点地图方式来进行3D场景理解，解决了3D表示学习中预训练先验稀缺和数据有限的问题。项目页面：https://matchlab-imperial.github.io/poma3d/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决3D场景理解中的表示学习问题，特别是缺乏大规模预训练模型和有限的3D数据问题。这个问题在现实中很重要，因为3D场景理解是增强现实系统和智能体感知物理世界的基础；在研究中重要是因为目前3D表示学习面临数据稀缺挑战，缺乏像2D领域那样强大的基础模型。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考认为需要一种既能保留丰富3D信息又能与2D知识对齐的中间表示，因此选择点图(point maps)作为桥梁。他们借鉴了CLIP的对比学习方法和JEPA架构思想，但进行了改进以适应点图特性。方法设计包括构建ScenePoint数据集、设计视图到场景的视觉-语言对齐策略和POMA-JEPA架构，并采用两阶段预训练策略。作者利用了现有的2D视觉-语言数据集和3D场景数据集，以及VGGT模型将图像转换为点图。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用点图作为3D和2D之间的桥梁，通过视图到场景的视觉-语言对齐将2D基础模型知识转移到3D领域，并使用POMA-JEPA确保多视图几何一致性。整体流程：1)从RGB-D视频和2D图像构建点图并生成文本描述；2)两阶段预训练(预热阶段在单视图点图上进行视觉-语言对齐，主要阶段在多视图点图上进行联合优化)；3)将预训练的点图编码器作为骨干网络，在3D问答、具身导航等任务上进行微调。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首次使用点图进行自监督3D表示学习；2)构建了包含6.5K房间级场景和1M图像场景的ScenePoint数据集；3)设计视图到场景的视觉-语言对齐策略；4)提出POMA-JEPA架构确保多视图几何一致性。不同之处：不同于点云、深度图或3D高斯溅射方法，使用点图作为中间表示；不仅使用3D场景数据，还利用2D图像数据扩展预训练规模；对齐点图视图和场景表示而非对象或场景级对齐；通过POMA-JEPA确保几何一致性；在多个任务上超越现有方法且无需颜色信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'POMA-3D首次通过点图表示和两阶段预训练策略，成功将2D基础模型的丰富知识迁移到3D场景理解任务中，解决了3D表示学习中的数据稀缺和预训练先验不足的问题。'}


### 论文摘要

In this paper, we introduce POMA-3D, the first self-supervised 3D representation model learned from point maps. Point maps encode explicit 3D coordinates on a structured 2D grid, preserving global 3D geometry while remaining compatible with the input format of 2D foundation models. To transfer rich 2D priors into POMA-3D, a view-to-scene alignment strategy is designed. Moreover, as point maps are view-dependent with respect to a canonical space, we introduce POMA-JEPA, a joint embedding-predictive architecture that enforces geometrically consistent point map features across multiple views. Additionally, we introduce ScenePoint, a point map dataset constructed from 6.5K room-level RGB-D scenes and 1M 2D image scenes to facilitate large-scale POMA-3D pretraining. Experiments show that POMA-3D serves as a strong backbone for both specialist and generalist 3D understanding. It benefits diverse tasks, including 3D question answering, embodied navigation, scene retrieval, and embodied localization, all achieved using only geometric inputs (i.e., 3D coordinates). Overall, our POMA-3D explores a point map way to 3D scene understanding, addressing the scarcity of pretrained priors and limited data in 3D representation learning. Project Page: https://matchlab-imperial.github.io/poma3d/

---

## 17. Broad stochastic configuration residual learning system for norm-convergent universal approximation

**论文链接:** [http://arxiv.org/abs/2511.16550v1](http://arxiv.org/abs/2511.16550v1)

**作者:** Han Su, Zhongyan Li, Wanquan Liu

**发布时间:** 2025-11-20

### GPT解析

### 总结

这篇论文提出了一种改进的广义随机配置残差学习系统(BSCRLS)算法，解决了原BRLS算法在通用近似性质上的局限性，并通过实验验证了其有效性和优越性。

### 背景

通用近似是神经网络学习算法的基础，但一些网络通过证明迭代误差在概率测度下收敛而非更严格的范数收敛来建立其通用近似性质，这使得随机学习网络的通用近似性质对随机参数选择高度敏感。广义残差学习系统(BRLS)作为随机学习模型的成员也面临这一问题。

### 目的

理论证明BRLS通用近似性质的局限性，并提出改进算法解决该问题。

### 方法

提出了广义随机配置残差学习系统(BSCRLS)算法，在BRLS框架基础上引入新的监督机制自适应约束随机参数的范围设置。提出了三种增量BSCRLS算法以满足不同网络更新的应用需求。在公开数据集上进行了太阳能电池板灰尘检测实验，并与13种深度和广度学习算法进行比较。

### 主要发现

如果随机参数选择不当且收敛速率满足某些条件，BRLS的迭代误差不满足范数收敛。BSCRLS基于更严格的范数收敛证明了通用近似定理，实验结果揭示了BSCRLS算法的有效性和优越性。

### 结论

BSCRLS算法解决了BRLS在通用近似性质方面的问题，实验验证了BSCRLS算法的有效性和优越性。

### 翻译

通用近似作为神经网络学习算法的基础。然而，一些网络通过证明迭代误差在概率测度下收敛而非更严格的范数收敛来建立其通用近似性质，这使得随机学习网络的通用近似性质对随机参数选择高度敏感。作为随机学习模型成员的广义残差学习系统(BRLS)也遇到这一问题。我们从理论上证明了其通用近似性质的局限性，即如果随机参数选择不当且收敛速率满足某些条件，迭代误差不满足范数收敛。为解决这一问题，我们提出了广义随机配置残差学习系统(BSCRLS)算法，该算法在BRLS框架基础上具有一种新颖的监督机制，能够自适应约束随机参数的范围设置。此外，我们基于更严格的范数收敛证明了BSCRLS的通用近似定理。提出了三种增量BSCRLS算法以满足各种网络更新的应用需求。在公开数据集上进行了太阳能电池板灰尘检测实验，并与13种深度和广度学习算法进行了比较。实验结果揭示了BSCRLS算法的有效性和优越性。


### 论文摘要

Universal approximation serves as the foundation of neural network learning algorithms. However, some networks establish their universal approximation property by demonstrating that the iterative errors converge in probability measure rather than the more rigorous norm convergence, which makes the universal approximation property of randomized learning networks highly sensitive to random parameter selection, Broad residual learning system (BRLS), as a member of randomized learning models, also encounters this issue. We theoretically demonstrate the limitation of its universal approximation property, that is, the iterative errors do not satisfy norm convergence if the selection of random parameters is inappropriate and the convergence rate meets certain conditions. To address this issue, we propose the broad stochastic configuration residual learning system (BSCRLS) algorithm, which features a novel supervisory mechanism adaptively constraining the range settings of random parameters on the basis of BRLS framework, Furthermore, we prove the universal approximation theorem of BSCRLS based on the more stringent norm convergence. Three versions of incremental BSCRLS algorithms are presented to satisfy the application requirements of various network updates. Solar panels dust detection experiments are performed on publicly available dataset and compared with 13 deep and broad learning algorithms. Experimental results reveal the effectiveness and superiority of BSCRLS algorithms.

---

## 18. Saving Foundation Flow-Matching Priors for Inverse Problems

**论文链接:** [http://arxiv.org/abs/2511.16520v1](http://arxiv.org/abs/2511.16520v1)

**作者:** Yuxiang Wan, Ryan Devera, Wenjie Zhang, Ju Sun

**发布时间:** 2025-11-20

### GPT解析

### 总结

基础流匹配(FM)模型有望成为解决逆问题(IP)的通用先验，但目前它们的表现不如领域特定甚至未经训练的先验。

### 背景

基础流匹配(FM)模型虽然承诺能作为解决逆问题(IP)的通用先验，但实际应用中它们的表现落后于领域特定甚至未经训练的先验方法。

### 目的

释放基础流匹配模型的潜力，使其成为解决逆问题的有效工具。

### 方法

作者提出了FMPlug框架，该框架结合了实例引导的时间相关热启动策略和尖锐高斯正则化，在保持高斯结构的同时添加问题特定的指导。

### 主要发现

FMPlug框架在图像恢复和科学逆问题上带来了显著的性能提升。

### 结论

研究结果为使基础流匹配模型成为逆问题求解中实用、可重用的先验指明了一条可行的道路。

### 翻译

基础流匹配(FM)模型有望成为解决逆问题(IP)的通用先验，但目前它们落后于领域特定甚至未经训练的先验。我们如何释放它们的潜力？我们引入了FMPlug，一个重新定义基础流匹配模型在逆问题中使用的插件框架。FMPlug结合了实例引导的时间相关热启动策略和尖锐高斯正则化，在保持高斯结构的同时添加了问题特定的指导。这导致在图像恢复和科学逆问题上的显著性能提升。我们的结果为使基础流匹配模型成为逆问题求解中实用、可重用的先验指明了一条道路。


### 论文摘要

Foundation flow-matching (FM) models promise a universal prior for solving inverse problems (IPs), yet today they trail behind domain-specific or even untrained priors. How can we unlock their potential? We introduce FMPlug, a plug-in framework that redefines how foundation FMs are used in IPs. FMPlug combines an instance-guided, time-dependent warm-start strategy with a sharp Gaussianity regularization, adding problem-specific guidance while preserving the Gaussian structures. This leads to a significant performance boost across image restoration and scientific IPs. Our results point to a path for making foundation FM models practical, reusable priors for IP solving.

---

## 19. MiMo-Embodied: X-Embodied Foundation Model Technical Report

**论文链接:** [http://arxiv.org/abs/2511.16518v1](http://arxiv.org/abs/2511.16518v1)

**作者:** Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, Yuchen Zhang, Jing Wu, Jinghui Lu, Chenxu Dang, Jiayi Guan, Jianhua Wu, Zhiyi Hou, Hanbing Li, Shumeng Xia, Mingliang Zhou, Yinan Zheng, Zihao Yue, Shuhao Gu, Hao Tian, Yuannan Shen, Jianwei Cui, Wen Zhang, Shaoqing Xu, Bing Wang, Haiyang Sun, Zeyu Zhu, Yuncheng Jiang, Zibin Guo, Chuhong Gong, Chaofan Zhang, Wenbo Ding, Kun Ma, Guang Chen, Rui Cai, Diyun Xiang, Heng Qu, Fuli Luo, Hangjun Ye, Long Chen

**发布时间:** 2025-11-20

**备注:** Code: https://github.com/XiaomiMiMo/MiMo-Embodied Model: https://huggingface.co/XiaomiMiMo/MiMo-Embodied-7B

### GPT解析

### 总结

MiMo-Embodied是一个开源的跨具身基础模型，首次成功整合并实现了自动驾驶和具身AI领域的最先进性能。

### 背景

自动驾驶和具身AI是两个相关但各自发展的领域，缺乏能够同时在这两个领域表现良好的统一模型。

### 目的

开发一个能够同时处理自动驾驶和具身AI任务的基础模型，实现两个领域知识和能力的有效迁移与整合。

### 方法

采用多阶段学习策略、精心策划的数据构建方法，以及CoT/RL微调技术来训练模型。

### 主要发现

在17个具身AI基准测试和12个自动驾驶基准测试中创造了新记录，显著优于现有开源、闭源和专用基线；两个领域通过所提方法表现出强烈的正迁移和相互强化效应。

### 结论

自动驾驶和具身AI领域可以通过适当的方法实现知识和能力的有效迁移与相互强化，为构建更通用的人工智能系统提供了新思路。

### 翻译

我们开源了MiMo-Embodied，这是首个成功整合并在自动驾驶和具身AI领域都实现最先进性能的跨具身基础模型。MiMo-Embodied在任务规划、可预测性和空间理解的17个具身AI基准测试中创造了新记录，同时在环境感知、状态预测和驾驶规划的12个自动驾驶基准测试中也表现出色。在这些任务中，MiMo-Embodied显著优于现有的开源、闭源和专用基线。我们的研究结果表明，通过多阶段学习、精心策划的数据构建以及CoT/RL微调，这两个领域表现出强烈的正迁移效应并相互强化。我们提供了详细的模型设计和训练方法分析，以促进进一步研究。代码和模型可在https://github.com/XiaomiMiMo/MiMo-Embodied获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决缺乏统一的具身AI和自动驾驶基础模型的问题。现有模型专注于单一领域（具身AI专注于室内任务，自动驾驶专注于室外道路），导致领域差距，限制了跨场景泛化能力和在动态环境中与物理世界的有效交互。这个问题阻碍了空间理解和推理能力在多样化室内外场景中的泛化，也缺乏综合评估框架来衡量模型的整体性能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有专用具身VLM的局限性，特别是领域差距和碎片化问题，提出了构建统一VLM的思路。他们借鉴了MiMo-VL的架构，继承其视觉语言对齐能力，并整合了多种现有数据集（如PixMo-Points、RoboAfford、RoboRefIt等具身AI数据集，以及CODA-LM、DriveLM、MME-RealWorld等自动驾驶数据集）。设计上采用了渐进式四阶段训练策略，从基础能力逐步提升到高级推理，并融入了链式思维和强化学习方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是构建一个统一的跨具身基础模型，同时处理具身AI和自动驾驶任务，实现两个领域间的知识迁移和相互强化。整体流程包括：1) 架构设计：使用Vision Transformer编码视觉输入，通过投影器映射到与LLM对齐的潜在空间，由LLM进行文本理解和推理；2) 数据构建：整合通用数据集、具身AI数据集和自动驾驶数据集；3) 四阶段训练：首先进行具身AI监督微调，然后是自动驾驶监督微调，接着是链式思维微调，最后是强化学习微调，逐步提升模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首个开源的跨具身基础模型，成功整合自动驾驶和具身AI领域；2) 在17个具身AI基准和12个自动驾驶基准上创纪录；3) 通过多阶段学习实现领域间强正迁移；4) 提供详细的模型设计和训练方法分析。相比之前工作，MiMo-Embodied打破了领域壁垒，实现了统一框架下的高性能表现，超越了现有开源、闭源和专用模型，并建立了全面的跨具身能力评估框架。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'MiMo-Embodied是首个统一自动驾驶和具身AI领域的开源基础模型，通过多阶段学习策略实现了两个领域间的知识迁移和相互强化，在29个基准测试上取得了最先进的性能。'}


### 论文摘要

We open-source MiMo-Embodied, the first cross-embodied foundation model to successfully integrate and achieve state-of-the-art performance in both Autonomous Driving and Embodied AI. MiMo-Embodied sets new records across 17 embodied AI benchmarks in Task Planning, Affordance Prediction and Spatial Understanding, while also excelling in 12 autonomous driving benchmarks across Environmental Perception, Status Prediction, and Driving Planning. Across these tasks, MiMo-Embodied significantly outperforms existing open-source, closed-source, and specialized baselines. Our results indicate that through multi-stage learning, curated data construction, and CoT/RL fine-tuning, these two domains exhibit strong positive transfer and mutually reinforce one another. We provide a detailed analysis of our model design and training methodologies to facilitate further research. Code and models are available at https://github.com/XiaomiMiMo/MiMo-Embodied.

---

## 20. ODE-ViT: Plug & Play Attention Layer from the Generalization of the ViT as an Ordinary Differential Equation

**论文链接:** [http://arxiv.org/abs/2511.16501v1](http://arxiv.org/abs/2511.16501v1)

**作者:** Carlos Boned Riera, David Romero Sanchez, Oriol Ramos Terrades

**发布时间:** 2025-11-20

### GPT解析

### 总结

研究提出了一种将Vision Transformer重新表述为常微分方程系统的方法，显著减少了模型参数量，同时保持了性能并提高了可解释性。

### 背景

近年来，大型模型在计算机视觉任务中表现出色，但它们需要大量计算资源和存储空间，且模型日益复杂限制了人们对决策过程的理解。大多数这些架构依赖于基于Transformer设计的注意力机制。

### 目的

开发一种参数量更少、更稳定且更易理解的模型架构，同时保持与原始模型相当的性能。

### 方法

基于残差神经网络与常微分方程之间的联系，引入ODE-ViT，将Vision Transformer重新表述为满足适定且稳定动力学条件的ODE系统。此外，提出了即插即用的教师-学生框架，其中离散ViT通过将教师的中间表示视为ODE的解来指导ODE-ViT的连续轨迹。

### 主要发现

在CIFAR-10和CIFAR-100上的实验表明，ODE-ViT实现了稳定、可解释且具有竞争力的性能，参数量减少了一个数量级，在分类任务中超越了先前的基于ODE的Transformer方法。教师-学生策略相比从头训练自由ODE-ViT提高了10%以上的性能。

### 结论

将Vision Transformer重新表述为ODE系统能够有效减少参数量，同时保持性能并提高可解释性和稳定性。教师-学生框架能够有效指导ODE模型的训练，进一步提升性能。

### 翻译

近年来，越来越大的模型在计算机视觉任务中取得了卓越的性能。然而，这些模型需要大量的计算资源和存储空间，并且它们日益增长的复杂性限制了我们对它们如何做决策的理解。这些架构中的大多数依赖于基于Transformer设计的注意力机制。基于残差神经网络与常微分方程之间的联系，我们引入了ODE-ViT，这是一种被重新表述为ODE系统的Vision Transformer，该系统满足适定且稳定动力学的条件。在CIFAR-10和CIFAR-100上的实验表明，ODE-ViT实现了稳定、可解释且具有竞争力的性能，参数量减少了一个数量级，在分类任务中超越了先前的基于ODE的Transformer方法。我们进一步提出了一个即插即用的教师-学生框架，其中离散ViT通过将教师的中间表示视为ODE的解来指导ODE-ViT的连续轨迹。与从头开始训练自由ODE-ViT相比，这种策略将性能提高了10%以上。


### 论文摘要

In recent years, increasingly large models have achieved outstanding performance across CV tasks. However, these models demand substantial computational resources and storage, and their growing complexity limits our understanding of how they make decisions. Most of these architectures rely on the attention mechanism within Transformer-based designs. Building upon the connection between residual neural networks and ordinary differential equations (ODEs), we introduce ODE-ViT, a Vision Transformer reformulated as an ODE system that satisfies the conditions for well-posed and stable dynamics. Experiments on CIFAR-10 and CIFAR-100 demonstrate that ODE-ViT achieves stable, interpretable, and competitive performance with up to one order of magnitude fewer parameters, surpassing prior ODE-based Transformer approaches in classification tasks. We further propose a plug-and-play teacher-student framework in which a discrete ViT guides the continuous trajectory of ODE-ViT by treating the intermediate representations of the teacher as solutions of the ODE. This strategy improves performance by more than 10% compared to training a free ODE-ViT from scratch.

---

## 21. Acquisition Time-Informed Breast Tumor Segmentation from Dynamic Contrast-Enhanced MRI

**论文链接:** [http://arxiv.org/abs/2511.16498v1](http://arxiv.org/abs/2511.16498v1)

**作者:** Rui Wang, Yuexi Du, John Lewin, R. Todd Constable, Nicha C. Dvornek

**发布时间:** 2025-11-20

**备注:** 5 pages, 3 figures

### GPT解析

### 总结

该研究提出了一种利用图像采集时间知识的肿瘤分割方法，通过特征线性调制(FiLM)层整合采集时间信息，提高了动态对比增强磁共振成像(DCE-MRI)中乳腺癌肿瘤分割的性能和模型泛化能力。

### 背景

DCE-MRI在乳腺癌筛查、肿瘤评估和治疗计划及监测中发挥重要作用。不同组织中的动态对比变化有助于在对比后图像中突出显示肿瘤，但不同的采集协议和个体因素导致组织外观的变异很大，即使是同一阶段的图像，这使得自动肿瘤分割具有挑战性。

### 目的

提出一种利用图像采集时间知识的肿瘤分割方法，根据特定的采集序列调节模型特征，以提高肿瘤分割的准确性和模型的泛化能力。

### 方法

使用特征线性调制(FiLM)层整合采集时间信息，这是一种轻量级方法可以整合时间信息，同时能够利用每个成像研究获取的全部可变数量的图像。在大型公共多中心乳腺DCE-MRI数据集上训练基线和不同配置的时间调制模型，使用不同的骨干架构。

### 主要发现

在领域内图像和公共领域外数据集上的评估显示，整合阶段采集时间知识提高了肿瘤分割性能和模型泛化能力。

### 结论

利用采集时间信息可以改善DCE-MRI图像中的肿瘤分割效果，并提高模型的泛化能力。

### 翻译

动态对比增强磁共振成像(DCE-MRI)在乳腺癌筛查、肿瘤评估和治疗计划及监测中发挥重要作用。不同组织中的动态对比变化有助于在对比后图像中突出显示肿瘤。然而，不同的采集协议和个体因素导致组织外观的变异很大，即使是同一阶段的图像(如首次对比后阶段)，这使得自动肿瘤分割具有挑战性。在此，我们提出了一种利用图像采集时间知识的肿瘤分割方法，根据特定的采集序列调节模型特征。我们使用特征线性调制(FiLM)层整合采集时间信息，这是一种轻量级方法，可以整合时间信息，同时能够利用每个成像研究获取的全部可变数量的图像。我们在大型公共多中心乳腺DCE-MRI数据集上训练基线和不同配置的时间调制模型，使用不同的骨干架构。在领域内图像和公共领域外数据集上的评估显示，整合阶段采集时间知识提高了肿瘤分割性能和模型泛化能力。


### 论文摘要

Dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) plays an important role in breast cancer screening, tumor assessment, and treatment planning and monitoring. The dynamic changes in contrast in different tissues help to highlight the tumor in post-contrast images. However, varying acquisition protocols and individual factors result in large variation in the appearance of tissues, even for images acquired in the same phase (e.g., first post-contrast phase), making automated tumor segmentation challenging. Here, we propose a tumor segmentation method that leverages knowledge of the image acquisition time to modulate model features according to the specific acquisition sequence. We incorporate the acquisition times using feature-wise linear modulation (FiLM) layers, a lightweight method for incorporating temporal information that also allows for capitalizing on the full, variables number of images acquired per imaging study. We trained baseline and different configurations for the time-modulated models with varying backbone architectures on a large public multisite breast DCE-MRI dataset. Evaluation on in-domain images and a public out-of-domain dataset showed that incorporating knowledge of phase acquisition time improved tumor segmentation performance and model generalization.

---

## 22. Mesoscale tissue properties and electric fields in brain stimulation - bridging the macroscopic and microscopic scales

**论文链接:** [http://arxiv.org/abs/2511.16465v1](http://arxiv.org/abs/2511.16465v1)

**作者:** Boshuo Wang, Torge Worbs, Minhaj A. Hussain, Aman S. Aberra, Axel Thielscher, Warren M. Grill, Angel V. Peterchev

**发布时间:** 2025-11-20

**备注:** 16 pages, 1 main figure, 6 appendix figures and 4 appendix tables

### GPT解析

### 总结

这篇论文探讨了脑刺激中电场模拟的准确性问题，重点关注如何通过连接宏观和微观尺度来推导中等尺度的导电性分布，以建立更准确的多尺度模型。

### 背景

准确的脑刺激电场模拟依赖于将宏观假设与微观组织结构联系起来的组织导电性表示。中等尺度的导电性变化对电场和神经激活阈值有重要影响，但在标准宏观模型中被忽视。微观模型虽能提供局部电场扰动的信息，但其有效性受到组织变形和细胞外空间重建不完整的限制。

### 目的

提出连接宏观和微观尺度的方法，推导一致的中等尺度导电性分布，为脑刺激中电场和神经激活的准确多尺度模型提供基础。

### 方法

作者概述了连接宏观和微观尺度的方法，以推导一致的中等尺度导电性分布。具体方法在摘要中未详细描述，但提到了需要克服微观模型中的固定相关组织变形和不完全细胞外空间重建问题。

### 主要发现

中等尺度的导电性变化可以对电场和神经激活阈值产生有意义的影响；微观模型表明存在显著的局部电场扰动，可以提供中等尺度导电性的信息。

### 结论

通过连接宏观和微观尺度推导一致的中等尺度导电性分布，可以为脑刺激中电场和神经激活的准确多尺度模型提供基础。

### 翻译

脑刺激中电场的准确模拟取决于将宏观假设与潜在微观组织结构联系起来的组织导电性表示。中等尺度的导电性变化可以对电场和神经激活阈值产生有意义的变化，但在标准宏观模型中基本缺失。最近的微观模型表明存在显著的局部电场扰动，原则上可以提供中等尺度的导电性信息。然而，微观模型的定量有效性受到固定相关组织变形和不完全细胞外空间重建的限制。我们概述了连接宏观和微观尺度的方法，以推导一致的中等尺度导电性分布，为脑刺激中电场和神经激活的准确多尺度模型提供了基础。


### 论文摘要

Accurate simulations of electric fields (E-fields) in brain stimulation depend on tissue conductivity representations that link macroscopic assumptions with underlying microscopic tissue structure. Mesoscale conductivity variations can produce meaningful changes in E-fields and neural activation thresholds but remain largely absent from standard macroscopic models. Recent microscopic models have suggested substantial local E-field perturbations and could, in principle, inform mesoscale conductivity. However, the quantitative validity of microscopic models is limited by fixation-related tissue distortion and incomplete extracellular-space reconstruction. We outline approaches that bridge macro- and microscales to derive consistent mesoscale conductivity distributions, providing a foundation for accurate multiscale models of E-fields and neural activation in brain stimulation.

---

## 23. Beyond Visual Cues: Leveraging General Semantics as Support for Few-Shot Segmentation

**论文链接:** [http://arxiv.org/abs/2511.16435v1](http://arxiv.org/abs/2511.16435v1)

**作者:** Jin Wang, Bingfeng Zhang, Jian Pang, Mengyu Liu, Honglong Chen, Weifeng Liu

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种语言驱动的属性泛化(LDAG)架构，用于解决少样本分割任务中支持图像参考信息不准确的问题。通过利用大型语言模型生成目标类别的属性描述，并设计多属性增强和多模态属性对齐模块，实现了对已训练和未训练类别的无偏元指导。

### 背景

现有少样本分割方法主要从支持图像中挖掘参考信息作为元学习指导，但由于视觉表示的类内变化，这些元信息无法为未训练类别提供准确的分割指导。

### 目的

解决现有方法中支持图像参考信息不准确的问题，为少样本分割任务提供更有效的元指导策略。

### 方法

提出语言驱动的属性泛化(LDAG)架构，包含：(1)多属性增强(MaE)模块，通过大型语言模型生成目标类别的多个详细属性描述，并利用多模态匹配构建精细的视觉-文本先验指导；(2)多模态属性对齐(MaA)模块，实现属性文本与视觉特征之间的跨模态交互，解决文本-视觉模态转换问题。

### 主要发现

支持图像中的参考可能不是必需的，支持的关键在于为已训练和未训练类别提供无偏的元指导；利用语言描述和大型语言模型可以构建更有效的少样本分割策略。

### 结论

提出的方法在实验中明显优于现有方法，达到了最新的最先进性能，代码将公开发布。

### 翻译

少样本分割(FSS)旨在通过元学习范式，在有限支持样本的指导下分割新类别。现有方法主要从支持图像中挖掘参考信息作为元指导。然而，由于视觉表示中的类内变化，从支持图像中提取的元信息无法为未训练类别提供准确的分割指导。在本文中，我们认为支持图像中的参考可能不是必需的，支持的关键在于为已训练和未训练类别提供无偏的元指导。随后，我们引入了语言驱动的属性泛化(LDAG)架构，利用固有的目标属性语言描述来构建健壮的支持策略。具体而言，为了获得无偏的支持表示，我们设计了一个多属性增强(MaE)模块，该模块通过大型语言模型生成目标类别的多个详细属性描述，然后利用多模态匹配构建精细的视觉-文本先验指导。同时，由于文本-视觉模态转换问题，属性文本难以促进视觉特征表示，我们设计了一个多模态属性对齐(MaA)模块，以实现属性文本与视觉特征之间的跨模态交互。实验表明，我们提出的方法以明显优势优于现有方法，并实现了最新的最先进性能。代码将公开发布。


### 论文摘要

Few-shot segmentation (FSS) aims to segment novel classes under the guidance of limited support samples by a meta-learning paradigm. Existing methods mainly mine references from support images as meta guidance. However, due to intra-class variations among visual representations, the meta information extracted from support images cannot produce accurate guidance to segment untrained classes. In this paper, we argue that the references from support images may not be essential, the key to the support role is to provide unbiased meta guidance for both trained and untrained classes. We then introduce a Language-Driven Attribute Generalization (LDAG) architecture to utilize inherent target property language descriptions to build robust support strategy. Specifically, to obtain an unbiased support representation, we design a Multi-attribute Enhancement (MaE) module, which produces multiple detailed attribute descriptions of the target class through Large Language Models (LLMs), and then builds refined visual-text prior guidance utilizing multi-modal matching. Meanwhile, due to text-vision modal shift, attribute text struggles to promote visual feature representation, we design a Multi-modal Attribute Alignment (MaA) to achieve cross-modal interaction between attribute texts and visual feature. Experiments show that our proposed method outperforms existing approaches by a clear margin and achieves the new state-of-the art performance. The code will be released.

---

## 24. Pharos-ESG: A Framework for Multimodal Parsing, Contextual Narration, and Hierarchical Labeling of ESG Report

**论文链接:** [http://arxiv.org/abs/2511.16417v1](http://arxiv.org/abs/2511.16417v1)

**作者:** Yan Chen, Yu Zou, Jialei Zeng, Haoran You, Xiaorui Zhou, Aixi Zhong

**发布时间:** 2025-11-20

**备注:** Accepted to AAAI 26:main technical track Oral

### GPT解析

### 总结

Pharos-ESG是一个统一框架，通过多模态解析、上下文叙述和分层标签化将ESG报告转化为结构化表示，解决了ESG报告因布局混乱和结构松散导致的大规模理解难题，并发布了首个大规模ESG报告公共数据集Aurora-ESG。

### 背景

ESG原则正在重塑全球金融治理基础，改变资本分配架构、监管框架和系统性风险协调机制。然而，作为评估企业ESG表现的核心媒介，ESG报告因幻灯片式不规则布局导致的阅读顺序混乱，以及内容冗长且结构松散导致的隐式层次结构，给大规模理解带来了重大挑战。

### 目的

解决ESG报告难以理解的问题，提出一个统一框架将ESG报告转化为结构化表示，以支持ESG在金融治理和决策中的整合。

### 方法

提出Pharos-ESG框架，通过以下方式实现：1)基于布局流的阅读顺序建模模块；2)由目录锚点引导的层次感知分割模块；3)将视觉元素上下文地转换为连贯自然语言的多模态聚合管道；4)通过ESG、GRI和情感标签丰富输出，产生与金融研究分析需求一致的标注。

### 主要发现

在标注基准上的大量实验表明，Pharos-ESG持续优于专门的文档解析系统和通用多模态模型。同时发布了Aurora-ESG数据集，首个覆盖中国大陆、香港和美国市场的大规模ESG报告公共数据集，具有多模态内容的统一结构化表示和细粒度的布局和语义标注。

### 结论

Pharos-ESG框架有效地解决了ESG报告理解中的挑战，通过结构化表示使ESG信息更易于分析，有助于ESG在金融治理和决策中的整合。

### 翻译

环境、社会和治理(ESG)原则正在重塑全球金融治理的基础，转变资本分配架构、监管框架和系统性风险协调机制。然而，作为评估企业ESG表现的核心媒介，ESG报告由于幻灯片式的不规则布局导致的阅读顺序混乱，以及内容冗长且结构松散导致的隐式层次结构，给大规模理解带来了重大挑战。为应对这些挑战，我们提出了Pharos-ESG，一个统一框架，通过多模态解析、上下文叙述和分层标签化将ESG报告转化为结构化表示。它集成了一个基于布局流的阅读顺序建模模块、一个由目录锚点引导的层次感知分割模块，以及一个将视觉元素上下文地转换为连贯自然语言的多模态聚合管道。该框架还通过ESG、GRI和情感标签丰富了其输出，产生与金融研究分析需求一致的标注。在标注基准上的大量实验表明，Pharos-ESG持续优于专门的文档解析系统和通用多模态模型。此外，我们发布了Aurora-ESG，这是首个大规模的ESG报告公共数据集，覆盖中国大陆、香港和美国市场，具有多模态内容的统一结构化表示，并配有细粒度的布局和语义标注，以更好地支持ESG在金融治理和决策中的整合。


### 论文摘要

Environmental, Social, and Governance (ESG) principles are reshaping the foundations of global financial gover- nance, transforming capital allocation architectures, regu- latory frameworks, and systemic risk coordination mecha- nisms. However, as the core medium for assessing corpo- rate ESG performance, the ESG reports present significant challenges for large-scale understanding, due to chaotic read- ing order from slide-like irregular layouts and implicit hier- archies arising from lengthy, weakly structured content. To address these challenges, we propose Pharos-ESG, a uni- fied framework that transforms ESG reports into structured representations through multimodal parsing, contextual nar- ration, and hierarchical labeling. It integrates a reading-order modeling module based on layout flow, hierarchy-aware seg- mentation guided by table-of-contents anchors, and a multi- modal aggregation pipeline that contextually transforms vi- sual elements into coherent natural language. The framework further enriches its outputs with ESG, GRI, and sentiment labels, yielding annotations aligned with the analytical de- mands of financial research. Extensive experiments on anno- tated benchmarks demonstrate that Pharos-ESG consistently outperforms both dedicated document parsing systems and general-purpose multimodal models. In addition, we release Aurora-ESG, the first large-scale public dataset of ESG re- ports, spanning Mainland China, Hong Kong, and U.S. mar- kets, featuring unified structured representations of multi- modal content, enriched with fine-grained layout and seman- tic annotations to better support ESG integration in financial governance and decision-making.

---

## 25. LAOF: Robust Latent Action Learning with Optical Flow Constraints

**论文链接:** [http://arxiv.org/abs/2511.16407v1](http://arxiv.org/abs/2511.16407v1)

**作者:** Xizhou Bu, Jiexi Lyu, Fulei Sun, Ruichen Yang, Zhiqiang Ma, Wei Li

**发布时间:** 2025-11-20

**备注:** Code can be found at https://github.com/XizoB/LAOF

### GPT解析

### 总结

本文提出了一种名为LAOF的鲁棒潜在动作学习方法，利用光流作为动作驱动信号，学习对干扰具有鲁棒性的潜在动作表示。实验表明，即使在极度标签稀缺条件下，LAOF也能稳定训练并提高表示质量，下游任务表现优于现有方法。

### 背景

从大规模视频中学习潜在动作对于可扩展具身基础模型的预训练至关重要，但现有方法往往难以处理与动作无关的干扰因素。

### 目的

开发一种能够在标签稀缺条件下学习鲁棒潜在动作表示的方法，减轻与动作无关的干扰因素。

### 方法

提出LAOF（使用光流约束的鲁棒潜在动作学习），这是一个伪监督框架，利用智能体的光流作为动作驱动信号来学习对干扰具有鲁棒性的潜在动作表示。光流自然抑制背景元素并强调移动物体。

### 主要发现

1) LAOF学习的潜在表示在下游模仿学习和强化学习任务上优于现有方法；2) 光流约束显著稳定了训练，提高标签稀缺条件下的表示质量；3) 即使没有动作监督，LAOF也能匹配或超越使用1%动作标签训练的动作监督方法；4) LAOF在动作标签比例增加到10%时仍然有效。

### 结论

LAOF通过利用光流作为动作驱动信号，成功解决了标签稀缺条件下学习鲁棒潜在动作表示的挑战，为具身基础模型的预训练提供了有效解决方案。

### 翻译

从大规模视频中学习潜在动作对于可扩展具身基础模型的预训练至关重要，然而现有方法往往难以处理与动作无关的干扰因素。虽然加入动作监督可以减轻这些干扰，但其效果受限于可用动作标签的稀缺性。光流表示连续帧之间的像素级运动，自然地抑制背景元素并强调移动物体。受此启发，我们提出了LAOF（使用光流约束的鲁棒潜在动作学习），这是一个伪监督框架，利用智能体的光流作为动作驱动信号来学习对干扰具有鲁棒性的潜在动作表示。实验结果表明，LAOF学习的潜在表示在下游模仿学习和强化学习任务上优于现有方法。这种优越性能源于光流约束，它显著稳定了训练，并在标签极度稀缺的条件下提高了潜在表示的质量，同时在动作标签比例增加到10%时仍然有效。重要的是，即使没有动作监督，LAOF也能匹配或超越使用1%动作标签训练的动作监督方法。


### 论文摘要

Learning latent actions from large-scale videos is crucial for the pre-training of scalable embodied foundation models, yet existing methods often struggle with action-irrelevant distractors. Although incorporating action supervision can alleviate these distractions, its effectiveness is restricted by the scarcity of available action labels. Optical flow represents pixel-level motion between consecutive frames, naturally suppressing background elements and emphasizing moving objects. Motivated by this, we propose robust Latent Action learning with Optical Flow constraints, called LAOF, a pseudo-supervised framework that leverages the agent's optical flow as an action-driven signal to learn latent action representations robust to distractors. Experimental results show that the latent representations learned by LAOF outperform existing methods on downstream imitation learning and reinforcement learning tasks. This superior performance arises from optical flow constraints, which substantially stabilize training and improve the quality of latent representations under extremely label-scarce conditions, while remaining effective as the proportion of action labels increases to 10 percent. Importantly, even without action supervision, LAOF matches or surpasses action-supervised methods trained with 1 percent of action labels.

---

## 26. Are Foundation Models Useful for Bankruptcy Prediction?

**论文链接:** [http://arxiv.org/abs/2511.16375v1](http://arxiv.org/abs/2511.16375v1)

**作者:** Marcin Kostrzewa, Oleksii Furman, Roman Furman, Sebastian Tomczak, Maciej Zięba

**发布时间:** 2025-11-20

**备注:** NeurIPS 2025 Workshop: Generative AI in Finance

### GPT解析

### 总结

该研究首次系统性比较了基础模型与经典机器学习方法在公司破产预测任务上的表现，结果表明专业化的机器学习方法在预测准确性和可靠性方面仍然优于基础模型。

### 背景

基础模型在各种金融应用中显示出潜力，但它们在公司破产预测方面的有效性尚未与已建立的方法进行系统性评估。

### 目的

研究使用Llama-3.3-70B-Instruct和TabPFN进行破产预测，在包含超过一百万条维谢格拉德集团公司记录的大型高度不平衡数据集上进行评估，并提供基础模型与经典机器学习基线方法在此任务上的首次系统性比较。

### 方法

使用Llama-3.3-70B-Instruct和TabPFN基础模型，在大型高度不平衡数据集上测试，并将这些基础模型与XGBoost和CatBoost等经典机器学习方法进行比较，评估不同预测时间范围内的表现。

### 主要发现

XGBoost和CatBoost等模型在所有预测时间范围内都持续优于基础模型；基于大语言模型的方法在概率估计方面不可靠，削弱了它们在风险敏感的金融环境中的使用；TabPFN虽然能与简单的基线方法竞争，但需要大量计算资源，而其性能提升并不足以证明这些成本的合理性。

### 结论

尽管基础模型具有通用性，但当前的基础模型在破产预测方面仍然不如专门开发的方法有效。

### 翻译

基础模型在各种金融应用中显示出潜力，但它们在公司破产预测方面的有效性尚未与已建立的方法进行系统性评估。我们研究了使用Llama-3.3-70B-Instruct和TabPFN进行破产预测，在包含超过一百万条维谢格拉德集团公司记录的大型高度不平衡数据集上进行了评估。我们首次提供了基础模型与经典机器学习基线方法在此任务上的系统性比较。我们的结果显示，XGBoost和CatBoost等模型在所有预测时间范围内都持续优于基础模型。基于大语言模型的方法在概率估计方面不可靠，削弱了它们在风险敏感的金融环境中的使用。TabPFN虽然能与简单的基线方法竞争，但需要大量计算资源，而其性能提升并不足以证明这些成本的合理性。这些发现表明，尽管基础模型具有通用性，但当前的基础模型在破产预测方面仍然不如专门开发的方法有效。


### 论文摘要

Foundation models have shown promise across various financial applications, yet their effectiveness for corporate bankruptcy prediction remains systematically unevaluated against established methods. We study bankruptcy forecasting using Llama-3.3-70B-Instruct and TabPFN, evaluated on large, highly imbalanced datasets of over one million company records from the Visegrád Group. We provide the first systematic comparison of foundation models against classical machine learning baselines for this task. Our results show that models such as XGBoost and CatBoost consistently outperform foundation models across all prediction horizons. LLM-based approaches suffer from unreliable probability estimates, undermining their use in risk-sensitive financial settings. TabPFN, while competitive with simpler baselines, requires substantial computational resources with costs not justified by performance gains. These findings suggest that, despite their generality, current foundation models remain less effective than specialized methods for bankruptcy forecasting.

---

## 27. Reasoning Meets Representation: Envisioning Neuro-Symbolic Wireless Foundation Models

**论文链接:** [http://arxiv.org/abs/2511.16369v1](http://arxiv.org/abs/2511.16369v1)

**作者:** Jaron Fontaine, Mohammad Cheraghinia, John Strassner, Adnan Shahid, Eli De Poorter

**发布时间:** 2025-11-20

**备注:** Accepted at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: AI and ML for Next-Generation Wireless Communications and Networking (AI4NextG)

### GPT解析

### 总结

本文提出神经符号范式作为解决无线物理层基础模型局限性的方法，以实现可信、通用且高效的无线AI，满足未来6G网络需求。

### 背景

无线物理层基础模型代表了射频表示的新范式，但这些模型继承了深度学习的局限性，包括缺乏可解释性、鲁棒性、适应性和可验证的合规性。AI原生6G网络需要深度嵌入系统的智能且可信的AI。

### 目的

通过神经符号范式弥合现有模型与未来网络需求之间的差距，构建可信、通用且高效的无线AI系统。

### 方法

提出神经符号框架，整合通用RF嵌入、符号知识图谱和可微分逻辑层，结合数据驱动的神经网络与基于规则和逻辑的符号推理。

### 主要发现

神经符号方法使模型能够从大型数据集中学习，同时利用显式领域知识进行推理，实现可信、通用且高效的无线AI。

### 结论

神经符号范式对于构建满足未来网络需求的可信、通用且高效的无线AI系统至关重要。

### 翻译

无线物理层基础模型的最新进展代表了通用射频表示的新范式。然而，这些模型继承了深度学习的关键局限性，如缺乏可解释性、鲁棒性、适应性和可验证的物理及监管合规性。此外，AI原生6G网络的愿景需要一种深度嵌入系统且可信的智能水平。在这篇愿景论文中，我们认为神经符号范式（整合数据驱动的神经网络与基于规则和逻辑的符号推理）对于弥合这一差距至关重要。我们设想了一种新的神经符号框架，整合通用RF嵌入、符号知识图谱和可微分逻辑层。这种混合方法使模型能够从大型数据集中学习，同时对显式领域知识进行推理，从而实现满足未来网络需求的可信、通用且高效的无线AI。


### 论文摘要

Recent advances in Wireless Physical Layer Foundation Models (WPFMs) promise a new paradigm of universal Radio Frequency (RF) representations. However, these models inherit critical limitations found in deep learning such as the lack of explainability, robustness, adaptability, and verifiable compliance with physical and regulatory constraints. In addition, the vision for an AI-native 6G network demands a level of intelligence that is deeply embedded into the systems and is trustworthy. In this vision paper, we argue that the neuro-symbolic paradigm, which integrates data-driven neural networks with rule- and logic-based symbolic reasoning, is essential for bridging this gap. We envision a novel Neuro-Symbolic framework that integrates universal RF embeddings with symbolic knowledge graphs and differentiable logic layers. This hybrid approach enables models to learn from large datasets while reasoning over explicit domain knowledge, enabling trustworthy, generalizable, and efficient wireless AI that can meet the demands of future networks.

---

## 28. OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe

**论文链接:** [http://arxiv.org/abs/2511.16334v1](http://arxiv.org/abs/2511.16334v1)

**作者:** Kaichen Zhang, Keming Wu, Zuhao Yang, Kairui Hu, Bin Wang, Ziwei Liu, Xingxuan Li, Lidong Bing

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究提出了OpenMMReasoner，一个完全透明的多模态推理两阶段训练配方，通过监督微调和强化学习相结合的方法，显著提升了多模态推理能力。

### 背景

大型推理模型的发展激发了人们对将这些能力扩展到多模态领域的兴趣。然而，尽管视觉推理取得了显著进展，但缺乏透明且可复现的数据构建和训练策略仍然是可扩展研究的主要障碍。

### 目的

介绍OpenMMReasoner，一个完全透明的多模态推理两阶段配方，包括监督微调(SFT)和强化学习(RL)。

### 方法

在SFT阶段，构建了一个包含874K样本的冷启动数据集，具有严格的逐步验证；在RL阶段，利用跨不同领域的74K样本数据集进一步磨练和稳定推理能力。

### 主要发现

广泛的评估表明，该训练配方不仅超越了强大的基线，还突显了数据质量和训练设计在塑造多模态推理性能中的关键作用。在九大多模态推理基准测试上比Qwen2.5-VL-7B-Instruct基线提高了11.6%。

### 结论

为未来大规模多模态推理研究奠定了坚实的经验基础，所有代码、流水线和数据已开源。

### 翻译

最近大型推理模型的进步激发了人们对将这些能力扩展到多模态领域的兴趣。然而，尽管视觉推理取得了显著进展，但缺乏透明且可复现的数据构建和训练策略仍然是可扩展研究的主要障碍。在这项工作中，我们介绍了OpenMMReasoner，一个完全透明的多模态推理两阶段配方，包括监督微调(SFT)和强化学习(RL)。在SFT阶段，我们构建了一个包含874K样本的冷启动数据集，具有严格的逐步验证，为推理能力提供了坚实的基础。随后的RL阶段利用跨不同领域的74K样本数据集进一步磨练和稳定这些能力，实现更稳健和高效的学习过程。广泛的评估表明，我们的训练配方不仅超越了强大的基线，还突显了数据质量和训练设计在塑造多模态推理性能中的关键作用。值得注意的是，我们的方法在九大多模态推理基准测试上比Qwen2.5-VL-7B-Instruct基线提高了11.6%，为未来大规模多模态推理研究奠定了坚实的经验基础。我们在https://github.com/EvolvingLMMs-Lab/OpenMMReasoner开源了所有代码、流水线和数据。


### 论文摘要

Recent advancements in large reasoning models have fueled growing interest in extending such capabilities to multimodal domains. However, despite notable progress in visual reasoning, the lack of transparent and reproducible data curation and training strategies remains a major barrier to scalable research. In this work, we introduce OpenMMReasoner, a fully transparent two-stage recipe for multimodal reasoning spanning supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct an 874K-sample cold-start dataset with rigorous step-by-step validation, providing a strong foundation for reasoning capabilities. The subsequent RL stage leverages a 74K-sample dataset across diverse domains to further sharpen and stabilize these abilities, resulting in a more robust and efficient learning process. Extensive evaluations demonstrate that our training recipe not only surpasses strong baselines but also highlights the critical role of data quality and training design in shaping multimodal reasoning performance. Notably, our method achieves a 11.6% improvement over the Qwen2.5-VL-7B-Instruct baseline across nine multimodal reasoning benchmarks, establishing a solid empirical foundation for future large-scale multimodal reasoning research. We open-sourced all our codes, pipeline, and data at https://github.com/EvolvingLMMs-Lab/OpenMMReasoner.

---

## 29. Beyond Generative AI: World Models for Clinical Prediction, Counterfactuals, and Planning

**论文链接:** [http://arxiv.org/abs/2511.16333v1](http://arxiv.org/abs/2511.16333v1)

**作者:** Mohammad Areeb Qazi, Maryam Nadeem, Mohammad Yaqub

**发布时间:** 2025-11-20

**备注:** 2 Figures, 1 Table

### GPT解析

### 总结

这篇论文回顾了医疗保健系统中使用世界模型进行预测性动态学习的研究，涵盖了医学影像诊断、疾病进展建模和机器人手术三个领域，并提出了一个能力标准框架和临床可靠世界模型的研究议程。

### 背景

医疗保健需要具有预测性、可靠性和数据效率的AI，但当前的生成模型缺乏物理基础和时间推理能力。随着语言模型扩展在临床推理方面收益递减，世界模型因其能够学习反映护理物理和因果结构的表示而受到关注。

### 目的

回顾医疗保健系统中学习预测性动态以实现多步展开、反事实评估和规划的世界模型，并确定研究差距和未来方向。

### 方法

调查三个领域的工作：医学影像和诊断（如纵向肿瘤模拟、投影转换建模和联合嵌入预测架构）、电子健康记录的疾病进展建模（大规模生成事件预测）以及机器人手术和手术规划（行动条件引导和控制）。引入了一个四级能力标准框架来评估这些系统。

### 主要发现

大多数 reviewed 系统达到L1-L2能力级别，较少实现L3，罕见达到L4。确定的主要研究差距包括：未明确的行动空间和安全约束、薄弱的干预验证、不完整的多模态状态构建以及有限的轨迹级别不确定性校准。

### 结论

需要开发临床稳健的预测优先世界模型，将生成主干（如transformers、diffusion、VAE）与因果/机械基础相结合，以实现医疗保健中的安全决策支持。

### 翻译

医疗保健需要具有预测性、可靠性和数据效率的AI。然而，最近的生成模型缺乏临床决策支持所需的物理基础和时间推理能力。随着扩展语言模型在基础临床推理方面显示出收益递减，世界模型正在获得关注，因为它们学习多模态、时间连贯和行动条件化的表示，反映护理的物理和因果结构。本文回顾了医疗保健系统的世界模型，这些模型学习预测性动态以实现多步展开、反事实评估和规划。我们调查了三个领域的最新工作：（i）医学影像和诊断（如纵向肿瘤模拟、投影转换建模和联合嵌入预测架构即JEPA风格预测表示学习），（ii）电子健康记录的疾病进展建模（大规模生成事件预测），以及（iii）机器人手术和手术规划（行动条件引导和控制）。我们还引入了一个能力标准：L1时间预测、L2行动条件预测、L3用于决策支持的反事实展开，以及L4规划/控制。大多数 reviewed 系统达到L1-L2，较少实现L3，罕见达到L4。我们确定了限制临床可靠性的跨领域差距：未明确的行动空间和安全约束、薄弱的干预验证、不完整的多模态状态构建以及有限的轨迹级别不确定性校准。本综述概述了临床稳健的预测优先世界模型的研究议程，这些模型将生成主干（transformers、diffusion、VAE）与因果/机械基础相结合，以实现医疗保健中的安全决策支持。


### 论文摘要

Healthcare requires AI that is predictive, reliable, and data-efficient. However, recent generative models lack physical foundation and temporal reasoning required for clinical decision support. As scaling language models show diminishing returns for grounded clinical reasoning, world models are gaining traction because they learn multimodal, temporally coherent, and action-conditioned representations that reflect the physical and causal structure of care. This paper reviews World Models for healthcare systems that learn predictive dynamics to enable multistep rollouts, counterfactual evaluation and planning. We survey recent work across three domains: (i) medical imaging and diagnostics (e.g., longitudinal tumor simulation, projection-transition modeling, and Joint Embedding Predictive Architecture i.e., JEPA-style predictive representation learning), (ii) disease progression modeling from electronic health records (generative event forecasting at scale), and (iii) robotic surgery and surgical planning (action-conditioned guidance and control). We also introduce a capability rubric: L1 temporal prediction, L2 action-conditioned prediction, L3 counterfactual rollouts for decision support, and L4 planning/control. Most reviewed systems achieve L1--L2, with fewer instances of L3 and rare L4. We identify cross-cutting gaps that limit clinical reliability; under-specified action spaces and safety constraints, weak interventional validation, incomplete multimodal state construction, and limited trajectory-level uncertainty calibration. This review outlines a research agenda for clinically robust prediction-first world models that integrate generative backbones (transformers, diffusion, VAE) with causal/mechanical foundation for safe decision support in healthcare.

---

## 30. SDA: Steering-Driven Distribution Alignment for Open LLMs without Fine-Tuning

**论文链接:** [http://arxiv.org/abs/2511.16324v1](http://arxiv.org/abs/2511.16324v1)

**作者:** Wei Xia, Zhi-Hong Deng

**发布时间:** 2025-11-20

### GPT解析

### 总结

论文提出SDA（Steering-Driven Distribution Alignment）框架，一种无需训练且与模型无关的对齐方法，通过动态重新分配模型输出概率增强LLMs与人类意图的一致性，无需微调且资源高效。

### 背景

随着大型语言模型（LLMs）快速发展，其在现实世界应用中部署日益广泛，但确保LLMs产生与人类意图一致的响应仍面临基础性挑战，特别是在推理过程中有效且高效地对齐模型行为。

### 目的

解决LLMs在推理过程中无需昂贵重新训练或广泛监督的情况下，有效且高效地对齐模型行为与人类意图的挑战。

### 方法

SDA框架根据用户定义的对齐指令动态重新分配模型输出概率，增强模型行为与人类意图一致性，方法轻量级、资源高效，兼容多种开源LLM，支持个性化偏好对齐，可独立运行或与基于训练的对齐策略集成。

### 主要发现

实验表明SDA在8个不同规模和来源的开源LLM上，在有用性、无害性和诚实性三个维度上均显著提升对齐性能，平均实现64.4%的有用性提升、30%的诚实性提升和11.5%的无害性提升。

### 结论

SDA是一种有效且通用的对齐框架，能显著提高开源LLMs与人类意图的一致性，无需昂贵的重新训练或广泛的监督，具有广泛兼容性和良好泛化能力。

### 翻译

随着大型语言模型（LLMs）的快速发展，它们在现实世界应用中的部署变得越来越广泛。LLMs被期望能够在多样化的任务、用户偏好和实际场景中提供强大的性能。然而，随着需求的增长，确保LLMs产生与人类意图一致的响应仍然是一个基础性挑战。特别是在推理过程中有效且高效地对齐模型行为，而无需昂贵的重新训练或广泛的监督，既是一项关键要求，也是一个非平凡的技术难题。为应对这一挑战，我们提出了SDA（Steering-Driven Distribution Alignment），一种专为开源LLMs设计的无需训练且与模型无关的对齐框架。SDA根据用户定义的对齐指令动态重新分配模型输出概率，无需微调即可增强模型行为与人类意图之间的一致性。该方法轻量级、资源高效，与多种开源LLM兼容。它可以在推理过程中独立运行，也可以与基于训练的对齐策略集成。此外，SDA支持个性化偏好对齐，能够灵活控制模型响应行为。实验结果表明，SDA在8个不同规模和来源的开源LLM上，在三个关键对齐维度（有用性、无害性和诚实性）上均一致提高了对齐性能。


### 论文摘要

With the rapid advancement of large language models (LLMs), their deployment in real-world applications has become increasingly widespread. LLMs are expected to deliver robust performance across diverse tasks, user preferences, and practical scenarios. However, as demands grow, ensuring that LLMs produce responses aligned with human intent remains a foundational challenge. In particular, aligning model behavior effectively and efficiently during inference, without costly retraining or extensive supervision, is both a critical requirement and a non-trivial technical endeavor. To address the challenge, we propose SDA (Steering-Driven Distribution Alignment), a training-free and model-agnostic alignment framework designed for open-source LLMs. SDA dynamically redistributes model output probabilities based on user-defined alignment instructions, enhancing alignment between model behavior and human intents without fine-tuning. The method is lightweight, resource-efficient, and compatible with a wide range of open-source LLMs. It can function independently during inference or be integrated with training-based alignment strategies. Moreover, SDA supports personalized preference alignment, enabling flexible control over the model response behavior. Empirical results demonstrate that SDA consistently improves alignment performance across 8 open-source LLMs with varying scales and diverse origins, evaluated on three key alignment dimensions, helpfulness, harmlessness, and honesty (3H). Specifically, SDA achieves average gains of 64.4% in helpfulness, 30% in honesty and 11.5% in harmlessness across the tested models, indicating its effectiveness and generalization across diverse models and application scenarios.

---

## 31. Quantitative Geometric Market Structuralism: A Framework for Detecting Structural Endpoints in Financial Markets. :

**论文链接:** [http://arxiv.org/abs/2511.16319v1](http://arxiv.org/abs/2511.16319v1)

**作者:** Amir Kavoosi

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究引入了定量几何市场结构主义(QGMS)框架，结合几何模式识别与定量数学建模，用于识别大规模市场走势的终端区域。该框架通过盲测验证保护知识产权，并在多次金融危机中验证了其预测能力，结果显示系统能提前识别市场逆转的结构端点。

### 背景

传统计量经济学或基于信号的模型无法充分捕捉市场动力学的复杂性。市场动态可被概念化为受价格形成自组织原则 governed 的演化几何结构。

### 目的

开发一种新方法识别大规模市场走势的终端区域，建立既能保护知识产权又能保持学术可测试性的框架，探索几何定量市场解释作为预测工具的潜力。

### 方法

定量几何市场结构主义(QGMS)框架，结合几何模式识别与定量数学建模。采用盲测验证过程，在分析中隐藏价格、符号和时间标识符，确保客观验证而不暴露算法核心。在多次金融危机中测试其预测稳健性。

### 主要发现

QGMS框架在2008年全球金融危机、2015年瑞郎事件、2016年英国脱欧公投和2020年COVID-19市场崩盘等多次金融危机中得到验证。在每种情况下，系统都能在主要市场逆转前一致地识别出结构端点。

### 结论

几何定量市场解释可能为数学形式主义和经验价格行为之间提供新的预测工具类别。QGMS框架通过结合学术可测试性和知识产权保护，为机构评估和非线性结构预测模型的进一步研究奠定了基础。

### 翻译

本研究引入了定量几何市场结构主义(QGMS)框架，这是一种混合分析方法，将几何模式识别与定量数学建模相结合，以识别大规模市场走势的终端区域。与传统计量经济学或基于信号的模型不同，QGMS框架将市场动力学概念化为受价格形成自组织原则 governed 的演化几何结构。为了保护其内部数学架构的专有性质，该方法采用盲测验证过程，在分析过程中隐藏价格、符号和时间标识符。这种设计确保了客观验证而无需揭示底层算法核心。该框架的预测稳健性已在多次金融危机中得到经验检验，包括2008年全球金融危机、2015年EUR CHF SNB事件、2016年英国脱欧公投和2020年COVID-19市场崩盘。在每种情况下，系统都能在主要市场逆转前一致地识别出结构端点。研究结果表明，几何定量市场解释可能提供一类新的预测工具，弥合数学形式主义与经验价格行为之间的差距。通过结合学术可测试性与知识产权保护，QGMS框架为机构评估和非线性结构预测模型的进一步研究建立了可行的基础。


### 论文摘要

This study introduces the Quantitative Geometric Market Structuralist (QGMS) framework a hybrid analytical methodology integrating geometric pattern recognition with quantitative mathematical modeling to identify terminal zones of large-scale market movements. Unlike conventional econometric or signal-based models, the QGMS framework conceptualizes market dynamics as evolving geometric structures governed by self-organizing principles of price formation.   To preserve the proprietary nature of its internal mathematical architecture, the methodology employs a blind-testing validation process, wherein price, symbol, and temporal identifiers are concealed during analysis. This design ensures objective verification without revealing the underlying algorithmic core. The frameworks predictive robustness has been empirically examined across multiple financial crises, including the 2008 Global Financial Collapse, the 2015 EUR CHF SNB event, the 2016 Brexit referendum, and the 2020 COVID-19 market crash. In each case, the system consistently identified structural endpoints preceding major market reversals.   The findings suggest that geometric quantitative market interpretation may offer a new class of predictive tools bridging the gap between mathematical formalism and empirical price behavior. By combining academic testability with intellectual property protection, the QGMS framework establishes a viable foundation for institutional evaluation and further research into nonlinear structural forecasting models.

---

## 32. Theoretical Analysis of Chirped Pulse Effects on Plasma Formation in Water Liquid Jet

**论文链接:** [http://arxiv.org/abs/2511.16310v1](http://arxiv.org/abs/2511.16310v1)

**作者:** Shireen Hilal, Azat O. Ismagilov, Anton N. Tsypkin, Maksim V. Melnik

**发布时间:** 2025-11-20

### GPT解析

### 总结

这是一项关于线性啁啾如何控制水射流中等离子体密度的理论研究，采用两阶段框架分离光谱相位效应与其他因素，为实验提供可测试预测。

### 背景

研究聚焦于水射流中等离子体密度的控制问题，需要理解啁啾参数对等离子体形成的影响机制。

### 目的

分离光谱相位效应与带宽和强度的关系，为水射流实验提供可测试的预测，并为未来实验和自洽传播模型奠定基础。

### 方法

采用两阶段框架：第一阶段在单一点求解载波布居数和电流方程，由啁啾超高斯脉冲驱动；第二阶段通过角谱法在水介质中传播场，并在整个空间应用相同方程。通过固定带宽和归一化强度，隔离啁啾对等离子体密度的单独响应。

### 主要发现

第一阶段：啁啾单独响应使等离子体密度超过1，负啁啾比正啁啾具有一致优势；第二阶段：正常色散下趋势逆转，啁啾单独响应的等离子体密度随啁啾增加而减小，负啁啾仍然危害较小，较长的FTL脉冲（如80 fs）抑制效果最强，由于色散引起的时域扩展和时空失同步。

### 结论

该研究成功分离了光谱相位效应与带宽和强度的关系，为水射流中等离子体密度控制提供了理论基础和实验指导。

### 翻译

我们提出了一项理论研究，探讨如何使用线性啁啾通过两阶段框架控制水射流中的等离子体密度。第一阶段在单一点求解载波布居数和电流方程，由啁啾超高斯脉冲驱动。通过固定带宽和归一化强度，我们隔离了等离子体密度的纯啁啾响应，该响应超过1且显示负啁啾相对于正啁啾具有一致优势。第二阶段通过角谱法在水介质中传播场，并在整个空间应用相同方程。正常色散逆转了这一趋势：纯啁啾等离子体密度随啁啾增加而减小，负啁啾仍然危害较小，且对于较长的FTL脉冲（如80 fs），抑制效果最强，这是由于色散引起的时域扩展和时空失同步。这项研究分离了光谱相位效应与带宽和强度，为水射流实验提供了可测试的预测，并为未来实验和自洽传播模型奠定了基础。


### 论文摘要

We present a theoretical study of how linear chirp controls plasma density in a water jet using a two-stage framework. Stage I solves carrier-population and current equations at a single point, driven by a chirped super-Gaussian pulse. By fixing bandwidth and normalizing for intensity, we isolate a chirp-only response of plasma density, which exceeds unity and shows a consistent advantage for negative over positive chirp. Stage II propagates the field in water via the angular-spectrum method and applies the same equations across space. Normal dispersion reverses the trend: the chirp-only plasma density decreases as chirp grows, negative chirp remains less detrimental, and suppression is strongest for longer FTL pulses (e.g., 80 fs) due to dispersion-induced temporal spreading and spatio-temporal desynchronization. This study separates spectral-phase effects from bandwidth and intensity, yields testable predictions for water jets, and provides a foundation for future experiments and self-consistent propagation models.

---

## 33. Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling

**论文链接:** [http://arxiv.org/abs/2511.16301v1](http://arxiv.org/abs/2511.16301v1)

**作者:** Minseok Seo, Mark Hamilton, Changick Kim

**发布时间:** 2025-11-20

**备注:** 15 pages, 12 figures

### GPT解析

### 总结

研究提出了'Upsample Anything'，一个轻量级的测试时优化框架，能够将低分辨率特征恢复为高分辨率像素级输出，无需训练，在多种任务上达到最先进性能。

### 背景

视觉基础模型虽然在各种下游任务中表现出强大的泛化能力，但其表示通常被大幅下采样(14x/16x)，限制了它们在像素级应用中的直接使用。现有特征上采样方法依赖于数据集特定的重新训练或沉重的隐式优化，可扩展性和泛化能力有限。

### 目的

开发一种无需训练且可扩展的特征上采样方法，使视觉基础模型能够直接应用于像素级任务。

### 方法

通过简单的每图像优化，学习一个结合空间和范围线索的各向异性高斯核，桥接高斯飞溅和联合双边上采样技术，创建一个通用的、边缘感知的算子。

### 主要发现

学习到的核可以无缝地跨架构和模态转移，实现特征、深度或概率图的高精度高分辨率重建。处理一张224x224图像仅需约0.419秒。

### 结论

'Upsample Anything'成功解决了视觉基础模型在像素级应用中的限制，提供了一种高效、通用的上采样解决方案，在语义分割、深度估计以及深度和概率图上采样任务上达到了最先进的性能。

### 翻译

我们提出了'Upsample Anything'，一个轻量级的测试时优化(TTO)框架，它将低分辨率特征恢复为高分辨率的像素级输出，无需任何训练。尽管视觉基础模型在各种下游任务中表现出强大的泛化能力，但它们的表示通常被下采样14x/16x(如ViT)，这限制了它们在像素级应用中的直接使用。现有的特征上采样方法依赖于数据集特定的重新训练或沉重的隐式优化，限制了可扩展性和泛化能力。'Upsample Anything'通过简单的每图像优化解决了这些问题，学习了一个结合空间和范围线索的各向异性高斯核，有效地桥接了高斯飞溅和联合双边上采样。学习到的核作为一个通用的、边缘感知的算子，可以无缝地跨架构和模态转移，实现特征、深度或概率图的高精度高分辨率重建。它处理一张224x224图像仅需约0.419秒，并在语义分割、深度估计以及深度和概率图上采样任务上达到了最先进的性能。


### 论文摘要

We present \textbf{Upsample Anything}, a lightweight test-time optimization (TTO) framework that restores low-resolution features to high-resolution, pixel-wise outputs without any training. Although Vision Foundation Models demonstrate strong generalization across diverse downstream tasks, their representations are typically downsampled by 14x/16x (e.g., ViT), which limits their direct use in pixel-level applications. Existing feature upsampling approaches depend on dataset-specific retraining or heavy implicit optimization, restricting scalability and generalization. Upsample Anything addresses these issues through a simple per-image optimization that learns an anisotropic Gaussian kernel combining spatial and range cues, effectively bridging Gaussian Splatting and Joint Bilateral Upsampling. The learned kernel acts as a universal, edge-aware operator that transfers seamlessly across architectures and modalities, enabling precise high-resolution reconstruction of features, depth, or probability maps. It runs in only $\approx0.419 \text{s}$ per 224x224 image and achieves state-of-the-art performance on semantic segmentation, depth estimation, and both depth and probability map upsampling.

---

## 34. How Robot Dogs See the Unseeable

**论文链接:** [http://arxiv.org/abs/2511.16262v1](http://arxiv.org/abs/2511.16262v1)

**作者:** Oliver Bimber, Karl Dietrich von Ellenrieder, Michael Haller, Rakesh John Amala Arokia Nathan, Gianni Lunardi, Marco Camurri, Mohamed Youssef, Santos Miguel Orozco Soto, Jeremy E. Niven

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究展示了动物Peering行为在机器人视觉中的应用，通过执行侧向运动形成合成孔径，有效解决了传统相机在部分遮挡场景下的局限性，实现了对遮挡场景的高效感知和理解。

### 背景

传统机器人相机的小光圈和大景深导致前景障碍物和背景物体都清晰聚焦，造成遮挡问题，限制了机器人对场景的理解能力。

### 目的

建立动物Peering行为与合成孔径(SA)感光之间的正式联系，克服机器人视觉中的部分遮挡问题，实现复杂环境中的高级场景理解。

### 方法

让机器人执行Peering运动，使相机形成一个宽合成孔径，通过计算机整合捕获的图像，合成具有极浅景深的图像，从而模糊遮挡元素同时使背景清晰对焦。

### 主要发现

这种高效、波长无关的技术可实现各种光谱波段上的实时、高分辨率感知；不仅能恢复基本场景理解，还能使大型多模态模型在遮挡图像中进行高级视觉推理；与依赖特征的多视图3D视觉方法或LiDAR等主动传感器相比，对遮挡具有鲁棒性且计算效率高。

### 结论

Peering运动用于合成孔径感光是理解复杂、杂乱环境中高级场景的关键，这种方法计算效率高，可立即在任何移动机器人上部署。

### 翻译

Peering是一种动物用来通过运动视差估计距离的侧向运动，为克服机器人视觉中的一个基本限制——部分遮挡，提供了强大的生物启发策略。传统机器人相机的小光圈和大景深使前景障碍物和背景物体都清晰聚焦，导致遮挡物遮挡了关键的场景信息。本研究建立了动物Peering与光学成像中的合成孔径(SA)感光之间的正式联系。通过让机器人执行Peering运动，其相机描述了一个宽合成孔径。对捕获图像的计算合成产生了一个具有极浅景深的图像，有效地模糊了遮挡元素，同时使背景清晰对焦。这种高效、波长无关的技术能够在各种光谱波段上实现实时、高分辨率感知。我们证明这种方法不仅能恢复基本场景理解，还能使大型多模态模型在传统遮挡图像中失效的情况下进行高级视觉推理。与依赖特征的多视图3D视觉方法或LiDAR等主动传感器不同，通过Peering的SA感光对遮挡具有鲁棒性，计算效率高，并可立即在任何移动机器人上部署。这项研究将动物行为与机器人技术联系起来，表明用于合成孔径感光的Peering运动是理解复杂、杂乱环境中高级场景的关键。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决机器人视觉系统中的部分遮挡问题，即前景物体遮挡背景物体导致场景理解困难。这个问题在现实应用中非常重要，因为它限制了机器人在监视、地形探索、检查、搜救等场景中的效能，而现有的3D视觉方法和主动传感器在处理遮挡场景时存在各种局限性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受到自然界昆虫（如蝗虫、螳螂等）窥视行为的启发，这些昆虫通过左右摆动身体利用运动视差增强视觉感知。作者将这种生物行为与合成孔径传感技术相结合，设计出让机器人通过侧向移动来创建合成孔径的方法。该方法借鉴了昆虫视觉研究、合成孔径传感技术、多模态视觉推理和植被指数技术等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是模仿昆虫的窥视行为，让机器人进行侧向移动创建合成孔径，捕获多张图像后通过计算合成为具有极浅景深的图像，通过调整合成焦平面使机器人能够聚焦于被前景遮挡的背景物体。整体流程包括：机器人进行窥视运动并捕获多张图像；记录每张图像的位姿数据；将图像投影到可调整的合成焦平面上；对投影图像进行平均处理生成合成孔径积分图像；调整焦平面参数实现对不同距离场景的聚焦；可选地使用植被指数识别遮挡物提高图像质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：生物启发的窥视行为应用；合成孔径与运动视差的结合；实时处理能力；波长无关性（适用于各种光谱带）；与多模态模型的结合提高遮挡场景理解能力。相比之前的工作，该方法不需要重建深度图而是直接在2D图像中抑制遮挡；相比主动传感器提供了更高分辨率和更丰富的多光谱数据；相比静态多相机阵列允许自适应采样；相比传统合成孔径成像特别针对遮挡场景优化且可实时处理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': "通过模仿昆虫的窥视行为并结合合成孔径传感技术，本文提出了一种使机器人能够实时'看穿'遮挡物的高效视觉感知方法。"}


### 论文摘要

Peering, a side-to-side motion used by animals to estimate distance through motion parallax, offers a powerful bio-inspired strategy to overcome a fundamental limitation in robotic vision: partial occlusion. Conventional robot cameras, with their small apertures and large depth of field, render both foreground obstacles and background objects in sharp focus, causing occluders to obscure critical scene information. This work establishes a formal connection between animal peering and synthetic aperture (SA) sensing from optical imaging. By having a robot execute a peering motion, its camera describes a wide synthetic aperture. Computational integration of the captured images synthesizes an image with an extremely shallow depth of field, effectively blurring out occluding elements while bringing the background into sharp focus. This efficient, wavelength-independent technique enables real-time, high-resolution perception across various spectral bands. We demonstrate that this approach not only restores basic scene understanding but also empowers advanced visual reasoning in large multimodal models, which fail with conventionally occluded imagery. Unlike feature-dependent multi-view 3D vision methods or active sensors like LiDAR, SA sensing via peering is robust to occlusion, computationally efficient, and immediately deployable on any mobile robot. This research bridges animal behavior and robotics, suggesting that peering motions for synthetic aperture sensing are a key to advanced scene understanding in complex, cluttered environments.

---

## 35. Difficulty-Controlled Simplification of Piano Scores with Synthetic Data for Inclusive Music Education

**论文链接:** [http://arxiv.org/abs/2511.16228v1](http://arxiv.org/abs/2511.16228v1)

**作者:** Pedro Ramoneda, Emilia Parada-Cabaleiro, Dasaem Jeong, Xavier Serra

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究提出了一种基于Transformer的MusicXML钢琴乐谱难度调整方法，通过合成数据集和预训练模型实现了对音乐难度的精确控制，并公开所有资源促进开源创新。

### 背景

AI在音乐教育领域有潜力，但受专有系统限制，技术民主化受阻。AI驱动的音乐难度调整可让音乐教育更包容，但现有工作依赖专有数据集且多使用缺乏可读性的MIDI格式，限制了实际应用。

### 目的

开发一种基于Transformer的方法，用于调整MusicXML钢琴乐谱的难度，克服现有方法的局限性。

### 方法

创建合成数据集，包含按难度排序的钢琴乐谱对，每对为同一曲目的不同难度版本。通过基于相同旋律和和谐条件生成变体，并利用预训练模型评估难度和风格确保适当配对。

### 主要发现

实验结果证明该方法有效，能准确控制可演奏性和目标难度，通过定性和定量评估得到验证。

### 结论

与以往工作不同，本研究公开所有资源（代码、数据集和模型），确保研究可复现性，同时促进开源创新，有助于弥合数字鸿沟。

### 翻译

尽管AI在音乐教育领域具有潜力，但专有系统阻碍了该领域技术的民主化。特别是，AI驱动的音乐难度调整特别有前景，因为简化复杂作品可以使音乐教育对各个年龄段和背景的学习者更加包容和可及。然而，近期的工作依赖于专有数据集，这阻碍了研究社区对当前最先进技术的复现、比较或扩展。此外，虽然这些生成方法具有巨大潜力，但大多数使用MIDI格式，与其他格式（如MusicXML）不同，MIDI缺乏可读性和布局信息，从而限制了它们对人类演奏者的实际应用。本研究介绍了一种基于Transformer的MusicXML钢琴乐谱难度调整方法。与依赖标注数据集的先前方法不同，我们提出了一种由按估计难度排序的钢琴乐谱对组成的合成数据集，每对包含同一曲目的更具挑战性和更简单的编排。我们通过创建基于相同旋律和和谐条件的变体来生成这些对，并利用预训练模型评估难度和风格，确保适当配对。实验结果证明了所提出方法的有效性，显示了可演奏性和目标难度的准确控制，这通过定性和定量评估得到强调。与以往工作相比，我们公开释放所有资源（代码、数据集和模型），确保可复现性的同时促进开源创新，帮助弥合数字鸿沟。


### 论文摘要

Despite its potential, AI advances in music education are hindered by proprietary systems that limit the democratization of technology in this domain. In particular, AI-driven music difficulty adjustment is especially promising, as simplifying complex pieces can make music education more inclusive and accessible to learners of all ages and contexts. Nevertheless, recent efforts have relied on proprietary datasets, which prevents the research community from reproducing, comparing, or extending the current state of the art. In addition, while these generative methods offer great potential, most of them use the MIDI format, which, unlike others, such as MusicXML, lacks readability and layout information, thereby limiting their practical use for human performers. This work introduces a transformer-based method for adjusting the difficulty of MusicXML piano scores. Unlike previous methods, which rely on annotated datasets, we propose a synthetic dataset composed of pairs of piano scores ordered by estimated difficulty, with each pair comprising a more challenging and easier arrangement of the same piece. We generate these pairs by creating variations conditioned on the same melody and harmony and leverage pretrained models to assess difficulty and style, ensuring appropriate pairing. The experimental results illustrate the validity of the proposed approach, showing accurate control of playability and target difficulty, as highlighted through qualitative and quantitative evaluations. In contrast to previous work, we openly release all resources (code, dataset, and models), ensuring reproducibility while fostering open-source innovation to help bridge the digital divide.

---

## 36. Can MLLMs Read the Room? A Multimodal Benchmark for Assessing Deception in Multi-Party Social Interactions

**论文链接:** [http://arxiv.org/abs/2511.16221v1](http://arxiv.org/abs/2511.16221v1)

**作者:** Caixin Kang, Yifei Huang, Liangyang Ouyang, Mingfang Zhang, Ruicong Liu, Yoichi Sato

**发布时间:** 2025-11-20

### GPT解析

### 总结

这篇论文介绍了多模态大语言模型在评估社交互动中的欺骗行为方面的局限性，提出了一个新的评估任务和数据集，分析了现有模型的失败模式，并提出了一个改进框架。

### 背景

最先进的多模态大语言模型(MLLMs)缺乏人类智能的核心能力——'察言观色'并评估复杂社交互动中欺骗的能力。

### 目的

量化MLLMs在社交欺骗评估方面的失败，并提出改进方法，使MLLMs具备更接近人类的社交推理能力。

### 方法

引入多模态互动欺骗评估(MIDA)任务和数据集，评估12个最先进的MLLMs，分析失败模式，并设计社交思维链(SoCoT)推理流程和动态社会认知记忆(DSEM)模块。

### 主要发现

即使是最强大的模型如GPT-4o也难以可靠地区分真伪；这些模型无法有效将语言与多模态社交线索联系起来，缺乏对他人认知状态建模的能力。

### 结论

构建更具感知力和可信度的AI系统需要新的方法；所提出的SoCoT和DSEM框架在欺骗评估任务上取得了性能提升，为构建具有真正类人社交推理能力的MLLMs提供了有希望的新路径。

### 翻译

尽管最先进的多模态大语言模型(MLLMs)具有先进的推理能力，但它们明显缺乏人类智能的一个核心组成部分：在复杂社交互动中'察言观色'和评估欺骗的能力。为了严格量化这种失败，我们引入了一个新任务——多模态互动欺骗评估(MIDA)，并提出了一个新颖的多模态数据集，提供同步的视频和文本，并且每个陈述都有可验证的真实标签。我们建立了一个全面的基准，评估了12个最先进的开源和闭源MLLMs，揭示了显著的性能差距：即使是像GPT-4o这样的强大模型也难以可靠地区分真伪。我们对失败模式的分析表明，这些模型无法有效地将语言与多模态社交线索联系起来，并且缺乏对他人所知、所信或所意图的能力建模，这凸显了构建更具感知力和可信度AI系统的迫切需要。为了取得进展，我们设计了一个社交思维链(SoCoT)推理流程和一个动态社会认知记忆(DSEM)模块。我们的框架在这一具有挑战性的任务上取得了性能提升，展示了构建具有真正类人社交推理能力的MLLMs的有希望的新路径。


### 论文摘要

Despite their advanced reasoning capabilities, state-of-the-art Multimodal Large Language Models (MLLMs) demonstrably lack a core component of human intelligence: the ability to `read the room' and assess deception in complex social interactions. To rigorously quantify this failure, we introduce a new task, Multimodal Interactive Deception Assessment (MIDA), and present a novel multimodal dataset providing synchronized video and text with verifiable ground-truth labels for every statement. We establish a comprehensive benchmark evaluating 12 state-of-the-art open- and closed-source MLLMs, revealing a significant performance gap: even powerful models like GPT-4o struggle to distinguish truth from falsehood reliably. Our analysis of failure modes indicates that these models fail to effectively ground language in multimodal social cues and lack the ability to model what others know, believe, or intend, highlighting the urgent need for novel approaches to building more perceptive and trustworthy AI systems. To take a step forward, we design a Social Chain-of-Thought (SoCoT) reasoning pipeline and a Dynamic Social Epistemic Memory (DSEM) module. Our framework yields performance improvement on this challenging task, demonstrating a promising new path toward building MLLMs capable of genuine human-like social reasoning.

---

## 37. FlipVQA-Miner: Cross-Page Visual Question-Answer Mining from Textbooks

**论文链接:** [http://arxiv.org/abs/2511.16216v1](http://arxiv.org/abs/2511.16216v1)

**作者:** Zhen Hao Wong, Jingwen Deng, Hao Liang, Runming He, Chengyu Shen, Wentao Zhang

**发布时间:** 2025-11-20

### GPT解析

### 总结

论文提出了一种从教育文档中自动提取高质量问答和视觉问答对的方法，解决了大型语言模型训练中高质量监督数据获取的难题。

### 背景

大型语言模型的发展越来越依赖于高质量的监督数据，但现有的指令微调和强化学习数据集成本高昂，且常依赖会产生幻觉且多样性有限的合成样本。教科书和练习材料包含丰富的高质量人类问答内容，但由于难以将原始PDF转换为AI可用的监督数据而未被充分利用。

### 目的

提出一种自动化流程，从教育文档中提取结构良好的问答（QA）和视觉问答（VQA）对，利用这些真实教育内容来改进大型语言模型的训练。

### 方法

提出一个自动化流程，结合了布局感知的OCR（光学字符识别）和基于LLM的语义解析，从教育文档中提取问答和视觉问答对。

### 主要发现

在多种文档类型上的实验表明，该方法能够产生准确、对齐且低噪声的问答/视觉问答对；这种方法能够实现真实教育内容的可扩展使用；为提高推理导向型大型语言模型训练提供了合成数据生成的实用替代方案。

### 结论

提出的方法有效地利用了教育文档中的高质量内容；所有代码和数据处理流程已在GitHub开源。

### 翻译

大型语言模型（LLMs）的发展越来越依赖于高质量的监督数据，然而现有的指令微调和强化学习数据集整理成本高昂，且常常依赖引入幻觉和有限多样性的合成样本。同时，教科书和练习材料包含丰富的高质量人类问答内容，但由于难以将原始PDF转换为AI可用的监督数据，这些内容仍未被充分利用。尽管现代OCR和视觉语言模型可以准确解析文档结构，但其输出缺乏训练所需的语义对齐。我们提出了一种自动化流程，通过结合布局感知的OCR和基于LLM的语义解析，从教育文档中提取结构良好的问答和视觉问答对。在多种文档类型上的实验表明，该方法能够产生准确、对齐且低噪声的问答/视觉问答对。这种方法能够实现真实教育内容的可扩展使用，并为改进推理导向型大型语言模型训练提供了合成数据生成的实用替代方案。所有代码和数据处理流程已在https://github.com/OpenDCAI/DataFlow开源。


### 论文摘要

The development of Large Language Models (LLMs) increasingly depends on high-quality supervised data, yet existing instruction-tuning and RL datasets remain costly to curate and often rely on synthetic samples that introduce hallucination and limited diversity. At the same time, textbooks and exercise materials contain abundant, high-quality human-authored Question-Answer(QA) content that remains underexploited due to the difficulty of transforming raw PDFs into AI-ready supervision. Although modern OCR and vision-language models can accurately parse document structure, their outputs lack the semantic alignment required for training. We propose an automated pipeline that extracts well-formed QA and visual-QA (VQA) pairs from educational documents by combining layout-aware OCR with LLM-based semantic parsing. Experiments across diverse document types show that the method produces accurate, aligned, and low-noise QA/VQA pairs. This approach enables scalable use of real-world educational content and provides a practical alternative to synthetic data generation for improving reasoning-oriented LLM training. All code and data-processing pipelines are open-sourced at https://github.com/OpenDCAI/DataFlow.

---

## 38. Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs

**论文链接:** [http://arxiv.org/abs/2511.16639v1](http://arxiv.org/abs/2511.16639v1)

**作者:** Wei-Cheng Tseng, David Harwath

**发布时间:** 2025-11-20

**备注:** To be presented at ASRU 2025

### GPT解析

### 总结

这篇摘要介绍了一种名为Codec2Vec的新型语音表征学习框架，该框架完全基于离散的音频编解码单元，具有高效存储、快速训练和数据隐私保护等优势。

### 背景

神经音频编解码技术的最新进展不仅实现了卓越的音频压缩，还改进了语音合成技术。研究人员正在探索其作为更广泛语音处理任务的通用声学特征提取器的潜力。

### 目的

基于这一趋势，作者引入了Codec2Vec，这是第一个完全基于离散音频编解码单元的语音表征学习框架。

### 方法

作者探索了掩码预测和各种训练目标推导策略，以全面了解该框架的有效性。

### 主要发现

在SUPERB基准测试中，Codec2Vec与连续输入模型相比具有竞争力，同时将存储需求减少了16.5倍，训练时间减少了2.3倍。

### 结论

Codec2Vec展示了其可扩展性和效率，证明了基于离散音频编解码单元的语音表征学习框架的有效性。

### 翻译

神经音频编解码技术的最新进展不仅实现了卓越的音频压缩，还改进了语音合成技术。研究人员正在探索其作为更广泛语音处理任务的通用声学特征提取器的潜力。基于这一趋势，我们引入了Codec2Vec，这是第一个完全基于离散音频编解码单元的语音表征学习框架。这种方法具有多种优势，包括提高数据存储和传输效率、加快训练速度以及增强数据隐私。我们探索了掩码预测和各种训练目标推导策略，以全面了解该框架的有效性。在SUPERB基准测试中，Codec2Vec与连续输入模型相比具有竞争力，同时将存储需求减少了16.5倍，训练时间减少了2.3倍，展示了其可扩展性和效率。


### 论文摘要

Recent advancements in neural audio codecs have not only enabled superior audio compression but also enhanced speech synthesis techniques. Researchers are now exploring their potential as universal acoustic feature extractors for a broader range of speech processing tasks. Building on this trend, we introduce Codec2Vec, the first speech representation learning framework that relies exclusively on discrete audio codec units. This approach offers several advantages, including improved data storage and transmission efficiency, faster training, and enhanced data privacy. We explore masked prediction with various training target derivation strategies to thoroughly understand the effectiveness of this framework. Evaluated on the SUPERB benchmark, Codec2Vec achieves competitive performance compared to continuous-input models while reducing storage requirements by up to 16.5x and training time by 2.3x, showcasing its scalability and efficiency.

---

## 39. Toward Artificial Palpation: Representation Learning of Touch on Soft Bodies

**论文链接:** [http://arxiv.org/abs/2511.16596v1](http://arxiv.org/abs/2511.16596v1)

**作者:** Zohar Rimon, Elisei Shafer, Tal Tepper, Efrat Shimron, Aviv Tamar

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究提出了一种基于自监督学习的人工触诊方法，通过编码器-解码器框架从触觉测量序列中学习表示，可用于触觉成像和变化检测任务。

### 背景

触诊作为医疗检查中的触觉诊断方法，目前几乎完全依赖人类执行，缺乏自动化解决方案。

### 目的

探索一种基于自监督学习的人工触诊方法的可行性，开发能够从触觉数据中提取有用信息的系统。

### 方法

开发模拟环境并收集包含软物体及其磁共振成像(MRI)图像的真实数据集；使用配备触觉传感器的机器人收集触诊序列；训练模型预测物体不同位置的感官读数；研究所学表示及其在成像和变化检测中的应用。

### 主要发现

编码器-解码器框架能够从触觉测量序列中学习包含被触诊对象所有相关信息的表示；这种表示超越了简单的力图映射，能够捕捉复杂的触觉测量模式；该表示可用于触觉成像和变化检测等下游任务。

### 结论

基于自监督学习的人工触诊方法具有可行性，能够从触觉数据中学习有意义的表示，为触诊的自动化提供了新途径。

### 翻译

触诊，即在医疗检查中使用触觉的方法，几乎完全由人类执行。我们研究了一种基于自监督学习的人工触诊方法的可行性概念。我们的核心思想是，编码器-解码器框架可以从一系列触觉测量中学习一种表示，这种表示包含被触诊对象的所有相关信息。我们推测，这种表示可用于下游任务，如触觉成像和变化检测。有足够的训练数据后，它应该能够捕捉触觉测量中超越简单力图映射（当前最先进技术）的复杂模式。为了验证我们的方法，我们既开发了模拟环境，也收集了包含软物体及其磁共振成像(MRI)获得相应真实图像的真实数据集。我们使用配备触觉传感器的机器人收集触诊序列，并训练一个模型来预测物体不同位置的感官读数。我们研究了在此过程中学到的表示，并展示了其在成像和变化检测中的应用。


### 论文摘要

Palpation, the use of touch in medical examination, is almost exclusively performed by humans. We investigate a proof of concept for an artificial palpation method based on self-supervised learning. Our key idea is that an encoder-decoder framework can learn a $\textit{representation}$ from a sequence of tactile measurements that contains all the relevant information about the palpated object. We conjecture that such a representation can be used for downstream tasks such as tactile imaging and change detection. With enough training data, it should capture intricate patterns in the tactile measurements that go beyond a simple map of forces -- the current state of the art. To validate our approach, we both develop a simulation environment and collect a real-world dataset of soft objects and corresponding ground truth images obtained by magnetic resonance imaging (MRI). We collect palpation sequences using a robot equipped with a tactile sensor, and train a model that predicts sensory readings at different positions on the object. We investigate the representation learned in this process, and demonstrate its use in imaging and change detection.

---

## 40. Reinforcement learning of quantum circuit architectures for molecular potential energy curves

**论文链接:** [http://arxiv.org/abs/2511.16559v1](http://arxiv.org/abs/2511.16559v1)

**作者:** Maureen Krumtünger, Alissa Wilms, Paul K. Faehrmann, Jens Eisert, Jakob Kottmann, Paolo Andrea Erdman, Sumeet Khatri

**发布时间:** 2025-11-20

**备注:** 16+14 pages, 21 figures. Comments welcome!

### GPT解析

### 总结

介绍了一种使用强化学习方法学习问题相关的量子电路映射，该方法能够为参数化哈密顿量族的基态输出量子电路，并在多个分子系统上展示了有效性和物理可解释性。

### 背景

量子化学和优化是量子计算机的两个最突出的应用领域，变分量子算法已被提议用于解决这些问题，但量子电路ansatz的设计仍是一个挑战。

### 目的

开发一种方法，可以为任何给定的问题实例生成电路，而不仅仅是为特定问题实例定制的电路，具体是提出强化学习方法学习问题相关的量子电路映射。

### 方法

提出了一种强化学习(RL)方法，输入分子和离散键距离集合，输出依赖于键距离的量子电路，本质是非贪心的，与现有的贪心方法形成对比。

### 主要发现

该方法在四个量子比特和六个量子比特的氢化锂分子以及八个量子比特的H4链上展示了有效性，学习到的电路在物理意义上具有可解释性。

### 结论

强化学习方法为开发大规模分子系统基态的新型量子电路铺平了道路，学习到的电路具有物理意义上的可解释性。

### 翻译

量子化学和优化是量子计算机两个最突出的应用领域。变分量子算法已被提议用于解决这些领域中的问题。然而，量子电路ansatz的设计仍然是一个挑战。特别令人感兴趣的是开发一种方法，可以为任何给定的问题实例生成电路，而不仅仅是为特定问题实例定制的电路。为此，我们提出了一种强化学习方法来学习问题相关的量子电路映射，该方法能够为给定参数化哈密顿量族中的哈密顿量基态输出电路。对于量子化学，我们的强化学习框架输入一个分子和一组离散的键距离，并输出一个依赖于键距离的量子电路，可用于任意键距离下的势能曲线。我们的强化学习方法的本质非贪心方法与现有的自适应、问题定制电路构建的贪心方法形成对比。我们在四个量子比特和六个量子比特的氢化锂分子以及八个量子比特的H4链上证明了其有效性。我们学习到的电路在物理意义上具有可解释性，从而为将强化学习应用于开发大规模分子系统基态的新型量子电路铺平了道路。


### 论文摘要

Quantum chemistry and optimization are two of the most prominent applications of quantum computers. Variational quantum algorithms have been proposed for solving problems in these domains. However, the design of the quantum circuit ansatz remains a challenge. Of particular interest is developing a method to generate circuits for any given instance of a problem, not merely a circuit tailored to a specific instance of the problem. To this end, we present a reinforcement learning (RL) approach to learning a problem-dependent quantum circuit mapping, which outputs a circuit for the ground state of a Hamiltonian from a given family of parameterized Hamiltonians. For quantum chemistry, our RL framework takes as input a molecule and a discrete set of bond distances, and it outputs a bond-distance-dependent quantum circuit for arbitrary bond distances along the potential energy curve. The inherently non-greedy approach of our RL method contrasts with existing greedy approaches to adaptive, problem-tailored circuit constructions. We demonstrate its effectiveness for the four-qubit and six-qubit lithium hydride molecules, as well as an eight-qubit H$_4$ chain. Our learned circuits are interpretable in a physically meaningful manner, thus paving the way for applying RL to the development of novel quantum circuits for the ground states of large-scale molecular systems.

---

## 41. Supervised Contrastive Learning for Few-Shot AI-Generated Image Detection and Attribution

**论文链接:** [http://arxiv.org/abs/2511.16541v1](http://arxiv.org/abs/2511.16541v1)

**作者:** Jaime Álvarez Urueña, David Camacho, Javier Huertas Tato

**发布时间:** 2025-11-20

**备注:** 17 pages, 6 figures, 6 tables

### GPT解析

### 总结

本文提出了一种新的两阶段检测框架，用于解决生成式AI创建的合成图像检测中的泛化挑战。该框架通过监督对比学习和少样本学习实现了高准确率的检测和来源归属。

### 背景

生成式人工智能的快速发展使得合成图像越来越难以与真实内容区分，对数字媒体完整性构成重大挑战。新型生成模型的加速发布周期使得传统依赖定期重新训练的检测方法在计算上不可行且操作上不实际。

### 目的

提出一种新的两阶段检测框架，解决合成图像检测中的泛化挑战，使系统能够适应不断发展的生成式AI景观而无需 exhaustive 重新训练。

### 方法

第一阶段使用通过监督对比学习训练的视觉深度学习模型提取图像判别性嵌入，在战略划分的生成器子集上训练并保留特定架构以测试跨生成器泛化能力；第二阶段使用k-NN分类器在学习的嵌入空间上操作，通过少样本学习范式训练，包含来自未见过的生成器的有限样本。

### 主要发现

在少样本学习模式下，每个类别仅使用150张图像时，框架实现了91.3%的平均检测准确率，比现有方法提高5.2个百分点；对于来源归属任务，在开放集分类背景下，AUC和OSCR分别提高了14.70%和4.27%。

### 结论

该方法代表了向健壮、可扩展的取证归属系统的重要进展，能够适应不断发展的生成式AI景观，无需 exhaustive 重新训练协议。

### 翻译

生成式人工智能的快速发展使得创建的合成图像越来越难以与真实内容区分，对数字媒体完整性构成重大挑战。新型生成模型的加速发布周期使得传统依赖定期重新训练的检测方法在计算上不可行且操作上不实际。本文提出了一种新的两阶段检测框架，旨在解决合成图像检测中固有的泛化挑战。第一阶段采用通过监督对比学习训练的视觉深度学习模型，从输入图像中提取判别性嵌入。关键的是，该模型在可用的生成器的战略划分子集上训练，特定架构被保留在训练之外，以严格消除跨生成器泛化能力。第二阶段利用在学习的嵌入空间上操作的k最近邻(k-NN)分类器，在少样本学习范式中训练，包含来自以前未见过的测试生成器的有限样本。在少样本学习模式下，每个类别仅使用150张图像（可从当前生成模型轻松获得），所提出的框架实现了91.3%的平均检测准确率，比现有方法提高了5.2个百分点。对于来源归属任务，在开放集分类背景下，所提出的方法在AUC和OSCR上分别提高了14.70%和4.27%，标志着向健壮、可扩展的取证归属系统的重要进展，这些系统能够适应不断发展的生成式AI景观，而无需 exhaustive 重新训练协议。


### 论文摘要

The rapid advancement of generative artificial intelligence has enabled the creation of synthetic images that are increasingly indistinguishable from authentic content, posing significant challenges for digital media integrity. This problem is compounded by the accelerated release cycle of novel generative models, which renders traditional detection approaches (reliant on periodic retraining) computationally infeasible and operationally impractical.   This work proposes a novel two-stage detection framework designed to address the generalization challenge inherent in synthetic image detection. The first stage employs a vision deep learning model trained via supervised contrastive learning to extract discriminative embeddings from input imagery. Critically, this model was trained on a strategically partitioned subset of available generators, with specific architectures withheld from training to rigorously ablate cross-generator generalization capabilities. The second stage utilizes a k-nearest neighbors (k-NN) classifier operating on the learned embedding space, trained in a few-shot learning paradigm incorporating limited samples from previously unseen test generators.   With merely 150 images per class in the few-shot learning regime, which are easily obtainable from current generation models, the proposed framework achieves an average detection accuracy of 91.3\%, representing a 5.2 percentage point improvement over existing approaches . For the source attribution task, the proposed approach obtains improvements of of 14.70\% and 4.27\% in AUC and OSCR respectively on an open set classification context, marking a significant advancement toward robust, scalable forensic attribution systems capable of adapting to the evolving generative AI landscape without requiring exhaustive retraining protocols.

---

## 42. Contrastive vision-language learning with paraphrasing and negation

**论文链接:** [http://arxiv.org/abs/2511.16527v1](http://arxiv.org/abs/2511.16527v1)

**作者:** Kwun Ho Ngan, Saman Sadeghi Afgeh, Joe Townsend, Artur d'Avila Garcez

**发布时间:** 2025-11-20

### GPT解析

### 总结

该论文提出了一种称为SemCLIP的新方法，通过结合文本改写和否定来改进视觉语言模型的表现，显著提高了模型对语义变换的鲁棒性。

### 背景

对比视觉语言模型（如CLIP）是图像和文本检索的主导方法，但它们在处理否定或改写文本时表现不佳，因为否定会以最小词汇变化根本改变含义，而改写可能以不同文本表达相同意思。

### 目的

解决视觉语言模型在评估结果和对齐方面的挑战，评估改写和否定的组合，提出新的CLIP对比损失函数，并应用LLM生成的训练三元组。

### 方法

提出SemCLIP方法，使用新的对比损失函数同时考虑改写和否定，应用包含原始、改写和否定文本标题的LLM生成的训练三元组进行训练，将改写标题推向原始图像嵌入，同时将否定标题在嵌入空间中推得更远。

### 主要发现

SemCLIP在保持CLIP性能的同时，显著增加与否定标题的距离；在CC-Neg基准测试上，准确度从68.1%提高到78.1%；在Sugarcrepe++基准测试上表现优于使用否定标题训练的模型；在下游零样本分类任务上也表现优于CLIP。

### 结论

SemCLIP能够实现对语义变换的显著鲁棒性，改进了视觉语言模型对否定和改写文本的处理能力。

### 翻译

对比视觉语言模型继续成为图像和文本检索的主导方法。对比语言-图像预训练（CLIP）通过对比方式训练两个神经网络，将它们的图像和文本嵌入对齐到共享的潜在空间。最近评估CLIP在否定或改写文本上的结果显示表现不一，因为否定会以最小的词汇变化根本性地改变含义，而改写可能会以不同的文本表达相同的意思。这为改进视觉语言模型的评估结果和对齐带来了重大挑战。为应对这一挑战，本文评估了改写和否定的组合，提出了一种考虑改写和否定的新的CLIP对比损失函数，并将包含原始、改写和否定文本标题的LLM生成的训练三元组应用于类CLIP训练模型。这种方法称为SemCLIP，能够将改写标题推向原始图像嵌入，同时在嵌入空间中将否定标题推得更远。实验证明，SemCLIP能够在保持CLIP性能的同时，显著增加与否定标题的距离。在使用原始否定图像检索准确度指标的CC-Neg基准测试上，SemCLIP将准确度从68.1%提高到78.1%。尽管与CLIP在Sugarcrepe++基准测试上的结果混合，但SemCLIP的性能通常优于使用否定标题训练的模型。这种对否定的鲁棒性扩展到了下游的零样本分类任务，在Sugarcrepe++上预训练的SemCLIP在所有测试的下游任务上表现优于CLIP。这些结果表明SemCLIP能够实现对语义变换的显著鲁棒性。


### 论文摘要

Contrastive vision-language models continue to be the dominant approach for image and text retrieval. Contrastive Language-Image Pre-training (CLIP) trains two neural networks in contrastive manner to align their image and text embeddings in a shared latent space. Recent results evaluating CLIP on negated or paraphrased text have shown mixed performance because negation changes meaning radically with minimal lexical changes, while paraphrasing can create very different textual expressions with the same intended meaning. This poses a significant challenge for improving the evaluation results and alignment of vision-language models. To address this challenge, this paper evaluates the combination of paraphrasing and negation, proposes a new CLIP contrastive loss function accounting for both paraphrasing and negation, and applies LLM-generated training triples consisting of original, paraphrased and negated textual captions to CLIP-like training models. The approach, called SemCLIP, is shown to move paraphrased captions towards the original image embeddings while pushing negated captions further away in embedding space. Empirically, SemCLIP is shown to be capable of preserving CLIP's performance while increasing considerably the distances to negated captions. On the CC-Neg benchmark using an original over negation image-retrieval accuracy metric, SemCLIP improves accuracy from 68.1% to 78.1%. Although results are mixed when compared with CLIP on the Sugarcrepe++ benchmark, SemCLIP's performance is generally better than the models trained with negated captions. This robustness to negation extends to downstream zero-shot classification tasks where SemCLIP pre-trained on Sugarcrepe++ performs better than CLIP on all tested downstream tasks. These results indicate that SemCLIP can achieve significant robustness to semantic transformations.

---

## 43. Limitations of Scalarisation in MORL: A Comparative Study in Discrete Environments

**论文链接:** [http://arxiv.org/abs/2511.16476v1](http://arxiv.org/abs/2511.16476v1)

**作者:** Muhammad Sa'ood Shah, Asad Jeewa

**发布时间:** 2025-11-20

**备注:** 15 pages, 4 figures, published in the Proceedings of the 46th Annual Conference of the South African Institute of Computer Scientists and Information Technologists (SAICSIT 2025)

### GPT解析

### 总结

本研究探讨了多目标强化学习算法中的标量化函数方法及其局限性，发现内循环多策略算法可能更适合复杂不确定环境下的智能决策。

### 背景

标量化函数被广泛用于多目标强化学习算法以实现智能决策，但这些函数在复杂、不确定的环境中往往难以准确逼近帕累托前沿。

### 目的

研究者在离散动作和观察空间的多目标强化学习环境中，研究标量化方法在多目标决策中的局限性。

### 方法

使用外循环多策略方法评估使用线性标量化和切比雪夫标量化函数实现的MO Q-Learning算法，并探索内循环多策略算法Pareto Q-Learning作为替代方案。

### 主要发现

标量化函数的性能高度依赖于环境和帕累托前沿的形状；这些函数往往无法保留学习过程中发现的解决方案，倾向于在解空间的某些区域寻找解决方案；找到适当的权重配置以采样整个帕累托前沿很复杂，限制了它们在不确定环境中的适用性。

### 结论

内循环多策略算法可能提供更可持续和通用的方法，并可能促进在动态和不确定环境中的智能决策。

### 翻译

标量化函数被广泛应用于多目标强化学习算法中，以实现智能决策。然而，这些函数往往难以准确逼近帕累托前沿，在复杂、不确定的环境中表现不理想。本研究考察了在离散动作和观察空间的多目标强化学习环境中的选定多目标强化学习算法。我们旨在进一步研究标量化方法在多目标决策中的局限性。具体而言，我们使用外循环多策略方法评估了一种经典单策略多目标强化学习算法（MO Q-Learning），该算法使用线性标量化和切比雪夫标量化函数实现。此外，我们还探索了一种创新的内循环多策略算法——帕累托Q学习，它提供了更强大的替代方案。我们的研究结果表明，标量化函数的性能高度依赖于环境和帕累托前沿的形状。这些函数往往无法保留学习过程中发现的解决方案，倾向于在解空间的某些区域寻找解决方案。此外，找到适当的权重配置以采样整个帕累托前沿很复杂，限制了它们在不确定环境中的适用性。相比之下，内循环多策略算法可能提供更可持续和通用的方法，并可能促进在动态和不确定环境中的智能决策。


### 论文摘要

Scalarisation functions are widely employed in MORL algorithms to enable intelligent decision-making. However, these functions often struggle to approximate the Pareto front accurately, rendering them unideal in complex, uncertain environments. This study examines selected Multi-Objective Reinforcement Learning (MORL) algorithms across MORL environments with discrete action and observation spaces. We aim to investigate further the limitations associated with scalarisation approaches for decision-making in multi-objective settings. Specifically, we use an outer-loop multi-policy methodology to assess the performance of a seminal single-policy MORL algorithm, MO Q-Learning implemented with linear scalarisation and Chebyshev scalarisation functions. In addition, we explore a pioneering inner-loop multi-policy algorithm, Pareto Q-Learning, which offers a more robust alternative. Our findings reveal that the performance of the scalarisation functions is highly dependent on the environment and the shape of the Pareto front. These functions often fail to retain the solutions uncovered during learning and favour finding solutions in certain regions of the solution space. Moreover, finding the appropriate weight configurations to sample the entire Pareto front is complex, limiting their applicability in uncertain settings. In contrast, inner-loop multi-policy algorithms may provide a more sustainable and generalizable approach and potentially facilitate intelligent decision-making in dynamic and uncertain environments.

---

## 44. A Comparison Between Decision Transformers and Traditional Offline Reinforcement Learning Algorithms

**论文链接:** [http://arxiv.org/abs/2511.16475v1](http://arxiv.org/abs/2511.16475v1)

**作者:** Ali Murtaza Caunhye, Asad Jeewa

**发布时间:** 2025-11-20

**备注:** 15 pages, 4 figures, published in the Proceedings of the 46th Annual conference of the South African Institute of Computer Scientists and Information Technologists (SIACSIT 2025)

### GPT解析

### 总结

本研究比较了传统离线强化学习算法(如CQL和IQL)与决策Transformer(DT)在不同奖励密度环境下的性能表现，发现DT对奖励密度变化不太敏感，在稀疏奖励场景中表现优异，而传统方法在密集奖励场景和高质量数据中表现更好。

### 背景

离线强化学习旨在从预收集数据集中推导有效策略而无需主动与环境交互。传统方法在平衡探索与利用方面面临挑战，特别是在奖励密度不同的环境中。决策Transformer将离线RL重新表述为序列建模问题，在多种基准测试中表现优异。

### 目的

评估DT与传统离线RL算法在ANT连续控制环境中密集和稀疏奖励设置下的性能表现，研究这些算法面对不同奖励结构时的表现及泛化能力。

### 方法

进行了一项比较研究，在ANT环境中评估DT与传统离线RL算法在不同奖励密度设置下的性能，通过实证分析比较这些算法的表现。

### 主要发现

DT对变化的奖励密度敏感性较低，在稀疏奖励场景的中等质量数据集上表现特别出色；IQL在密集奖励设置和高质量数据中表现更好；CQL在不同数据质量下提供平衡性能；DT表现出较低性能方差但需要更多计算资源。

### 结论

序列建模方法更适合具有不确定奖励结构或混合质量数据的场景，而基于价值的方法在具有密集奖励和高质量演示的设置中仍然具有竞争力。

### 翻译

离线强化学习(Offline RL)领域旨在从预收集的数据集中推导出有效的策略，而无需主动与环境交互。虽然传统的离线RL算法如保守Q学习(CQL)和隐式Q学习(IQL)已经显示出前景，但它们在平衡探索和利用方面常常面临挑战，特别是在具有不同奖励密度的环境中。最近提出的决策Transformer(DT)方法，将离线RL重新表述为序列建模问题，已在各种基准测试中展示了令人印象深刻的结果。本文提出了一个比较研究，评估了DT在ANT连续控制环境中密集和稀疏奖励设置下与传统离线RL算法的性能。我们的研究调查了这些算法在面对不同奖励结构时的表现，考察了它们学习有效策略和在变化的反馈水平上泛化的能力。通过在ANT环境中的实证分析，我们发现与其它方法相比，DT对变化的奖励密度敏感性较低，并且在稀疏奖励场景中的中等专家数据集上表现特别出色。相比之下，传统的基于价值的方法如IQL在密集奖励设置和高质量数据中表现出更好的性能，而CQL则在不同数据质量下提供了平衡的性能。此外，DT表现出较低的性能方差，但与传统方法相比需要更多的计算资源。这些发现表明，序列建模方法可能更适合具有不确定奖励结构或混合质量数据的场景，而基于价值的方法在具有密集奖励和高质量演示的设置中仍然具有竞争力。


### 论文摘要

The field of Offline Reinforcement Learning (RL) aims to derive effective policies from pre-collected datasets without active environment interaction. While traditional offline RL algorithms like Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) have shown promise, they often face challenges in balancing exploration and exploitation, especially in environments with varying reward densities. The recently proposed Decision Transformer (DT) approach, which reframes offline RL as a sequence modelling problem, has demonstrated impressive results across various benchmarks. This paper presents a comparative study evaluating the performance of DT against traditional offline RL algorithms in dense and sparse reward settings for the ANT continous control environment. Our research investigates how these algorithms perform when faced with different reward structures, examining their ability to learn effective policies and generalize across varying levels of feedback. Through empirical analysis in the ANT environment, we found that DTs showed less sensitivity to varying reward density compared to other methods and particularly excelled with medium-expert datasets in sparse reward scenarios. In contrast, traditional value-based methods like IQL showed improved performance in dense reward settings with high-quality data, while CQL offered balanced performance across different data qualities. Additionally, DTs exhibited lower variance in performance but required significantly more computational resources compared to traditional approaches. These findings suggest that sequence modelling approaches may be more suitable for scenarios with uncertain reward structures or mixed-quality data, while value-based methods remain competitive in settings with dense rewards and high-quality demonstrations.

---

## 45. CylinderDepth: Cylindrical Spatial Attention for Multi-View Consistent Self-Supervised Surround Depth Estimation

**论文链接:** [http://arxiv.org/abs/2511.16428v1](http://arxiv.org/abs/2511.16428v1)

**作者:** Samer Abualhanud, Christian Grannemann, Max Mehltretter

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种几何引导的自监督全景深度估计方法，解决了多图像间深度估计不一致的问题，实现了密集、度量和跨视图一致的深度预测。

### 背景

自监督全景深度估计可以从多个最小重叠图像实现密集、低成本的360度视野3D感知，但现有方法存在不同图像间深度估计不一致的问题。

### 目的

提出一种新的几何引导方法，用于校准的、时间同步的多相机系统，预测密集、度量且跨视图一致的深度。

### 方法

首先为每个图像预测深度图，然后将所有图像的3D点投影到共享的单位圆柱上建立跨图像邻域关系，生成二维位置图，最后基于位置图应用显式空间注意力跨图像聚合特征，预测每个图像的最终深度图。

### 主要发现

在DDAD和nuScenes数据集上的评估表明，该方法提高了图像间深度估计的一致性和整体深度质量，优于现有最先进方法。

### 结论

所提出的几何引导方法有效解决了多图像间深度估计不一致的问题，实现了更准确的360度3D感知。

### 翻译

自监督全景深度估计能够从多个最小重叠图像实现密集、低成本的360度视野3D感知。然而，大多数现有方法存在不同重叠图像间深度估计不一致的问题。针对这一局限，我们提出了一种针对校准的、时间同步的多相机系统的几何引导新方法，可预测密集、度量且跨视图一致的深度。给定内参和相对方向参数，首先为每个图像预测一个深度图，并将所有图像推导出的3D点投影到共享的单位圆柱上，建立不同图像间的邻域关系。这为每个图像生成一个二维位置图，其中每个像素被分配其在圆柱上的投影位置。基于这些位置图，我们应用显式的、非学习的空间注意力，根据像素在圆柱上的距离跨图像聚合特征，预测每个图像的最终深度图。在DDAD和nuScenes数据集上的评估表明，与最先进的方法相比，我们的方法提高了图像间深度估计的一致性和整体深度质量。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自监督环绕视图深度估计中不同图像之间的深度估计不一致的问题。这个问题在自动驾驶和机器人领域至关重要，因为它直接影响3D场景理解、定位、避障和运动规划的准确性。不一致的深度估计会导致错误的3D重建，影响后续的感知和决策，而环绕相机系统提供360°场景覆盖，是现代自动驾驶车辆的关键传感器配置。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到现有方法只在训练过程中隐式地强制多视图一致性，无法保证推理时的一致性。他们设计了一种几何引导的方法，利用已知的相机参数将所有图像的3D点投影到共享的圆柱空间，建立跨图像的邻域关系。在此基础上，他们应用显式的非学习空间注意力机制聚合特征。该方法借鉴了自监督深度估计中的光度一致性原理、多视图立体几何以及注意力机制，但创新性地将这些元素应用于圆柱空间并使其成为几何引导的。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将所有输入图像的像素投影到一个共享的圆柱表示中，在这个圆柱空间中应用基于几何距离的空间注意力机制，从而强制多视图一致性。整体流程包括：1)使用编码器-解码器架构初步预测每个图像的深度；2)基于初步深度和相机参数将所有图像的特征投影到共享圆柱上，生成位置图；3)在圆柱空间应用基于测地距离的空间注意力，并结合特征相似性进行调制；4)将注意力聚合后的特征解码为最终深度图；5)使用光度一致性损失和辅助损失进行自监督训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)圆柱空间表示，将多视图点映射到统一坐标系；2)显式的非学习几何引导空间注意力机制；3)多视图一致性强制机制；4)特征相似性调制空间注意力。相比之前的工作，不同之处在于：与纯学习方法不同，作者的方法利用已知相机几何提供显式一致性保证；与3D方法相比，在二维圆柱面上操作效率更高；与仅通过损失函数强制一致性的方法相比，能在推理时保证多视图一致性；与球面投影相比避免了极点失真。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CylinderDepth通过将多视图深度估计映射到圆柱空间并应用几何引导的显式空间注意力，实现了前所未有的多视图一致性和深度准确性，为自动驾驶和机器人应用提供了更可靠的3D感知能力。'}


### 论文摘要

Self-supervised surround-view depth estimation enables dense, low-cost 3D perception with a 360° field of view from multiple minimally overlapping images. Yet, most existing methods suffer from depth estimates that are inconsistent between overlapping images. Addressing this limitation, we propose a novel geometry-guided method for calibrated, time-synchronized multi-camera rigs that predicts dense, metric, and cross-view-consistent depth. Given the intrinsic and relative orientation parameters, a first depth map is predicted per image and the so-derived 3D points from all images are projected onto a shared unit cylinder, establishing neighborhood relations across different images. This produces a 2D position map for every image, where each pixel is assigned its projected position on the cylinder. Based on these position maps, we apply an explicit, non-learned spatial attention that aggregates features among pixels across images according to their distances on the cylinder, to predict a final depth map per image. Evaluated on the DDAD and nuScenes datasets, our approach improves the consistency of depth estimates across images and the overall depth compared to state-of-the-art methods.

---

## 46. CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement

**论文链接:** [http://arxiv.org/abs/2511.16378v1](http://arxiv.org/abs/2511.16378v1)

**作者:** Pan Yang, Cheng Deng, Jing Yang, Han Zhao, Yun Liu, Yuling Chen, Xiaoli Ruan, Yanping Chen

**发布时间:** 2025-11-20

### GPT解析

### 总结

这篇论文提出了CAMS方法，用于组合零样本学习，通过从视觉特征中提取语义特征并在多维空间中进行语义解耦，提高对未见过的属性-对象组合的泛化能力。

### 背景

组合零样本学习旨在学习已见过的属性-对象组合中的概念并识别未见过的组合。现有的基于CLIP的方法依赖于全局语义表示来解耦属性和对象，但这种表示能力有限，无法完全解耦两者。

### 目的

提出一种能够从视觉特征中提取语义特征并在多维空间中进行语义解耦的方法，以改善对未见过的属性-对象组合的泛化能力。

### 方法

CAMS方法包含两个主要部分：1）门控交叉注意力机制，通过潜在单元从CLIP的高级图像编码块中捕获细粒度语义特征，同时抑制背景和无关信息；2）多空间解耦，实现属性和对象语义的解耦。

### 主要发现

在三个基准数据集（MIT-States、UT-Zappos和C-GQA）上的实验表明，CAMS在闭世界和开世界设置下都取得了最先进的性能。

### 结论

CAMS方法通过多维空间中的语义解耦有效提高了对未见过的属性-对象组合的识别能力，在多个基准测试上超越了现有方法。

### 翻译

组合零样本学习（CZSL）旨在学习已见过的组合中的属性和对象概念，并识别它们的未见过的组合。大多数基于对比语言-图像预训练（CLIP）的CZSL方法专注于利用从图像编码器获得的全局语义表示来解耦属性和对象。然而，这种表示的表示能力有限，且不允许两者完全解耦。为此，我们提出了CAMS，旨在从视觉特征中提取语义特征，并在多维空间中进行语义解耦，从而提高对未见过的属性-对象组合的泛化能力。具体来说，CAMS设计了一个门控交叉注意力，通过一组潜在单元从CLIP的高级图像编码块中捕获细粒度语义特征，同时自适应地抑制背景和其他无关信息。随后，它进行多空间解耦以实现属性和对象语义的解耦。在三个流行基准（MIT-States、UT-Zappos和C-GQA）上的实验表明，CAMS在闭世界和开世界设置下都取得了最先进的性能。代码可在https://github.com/ybyangjing/CAMS获取。


### 论文摘要

Compositional zero-shot learning (CZSL) aims to learn the concepts of attributes and objects in seen compositions and to recognize their unseen compositions. Most Contrastive Language-Image Pre-training (CLIP)-based CZSL methods focus on disentangling attributes and objects by leveraging the global semantic representation obtained from the image encoder. However, this representation has limited representational capacity and do not allow for complete disentanglement of the two. To this end, we propose CAMS, which aims to extract semantic features from visual features and perform semantic disentanglement in multidimensional spaces, thereby improving generalization over unseen attribute-object compositions. Specifically, CAMS designs a Gated Cross-Attention that captures fine-grained semantic features from the high-level image encoding blocks of CLIP through a set of latent units, while adaptively suppressing background and other irrelevant information. Subsequently, it conducts Multi-Space Disentanglement to achieve disentanglement of attribute and object semantics. Experiments on three popular benchmarks (MIT-States, UT-Zappos, and C-GQA) demonstrate that CAMS achieves state-of-the-art performance in both closed-world and open-world settings. The code is available at https://github.com/ybyangjing/CAMS.

---

## 47. Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement

**论文链接:** [http://arxiv.org/abs/2511.16331v1](http://arxiv.org/abs/2511.16331v1)

**作者:** Jiashu Yao, Heyan Huang, Shuang Zeng, Chuwei Luo, WangJie You, Jie Tang, Qingsong Liu, Yuhang Guo, Yangyang Kang

**发布时间:** 2025-11-20

**备注:** Accepted to AAAI 2026

### GPT解析

### 总结

研究提出了一种自我重写框架，通过让大型推理模型重写自己的推理文本并从中学习，显著提高了内部推理质量，同时提高了准确率(+0.6)并减少了推理长度(-46%)，成功缓解了过度思考、思考不足、冗余思考和无序思考等问题。

### 背景

大型推理模型通过强化学习与结果正确性奖励相结合，在复杂推理任务上取得了显著成功，但仅关注最终正确性的单向奖励限制了模型对内部推理过程的详细监督能力。

### 目的

引入自我重写框架，让模型重写自己的推理文本，然后从重写的推理中学习，以提高内部思维过程质量。

### 方法

提出选择性重写方法，仅重写模型一致正确的'简单'样本，保留所有原始奖励信号；在实际实现中，将重写和普通生成编译在一个批次中，保持RL算法的可扩展性，仅引入约10%的开销。

### 主要发现

在不同模型大小的多种任务上的实验验证了自我重写的有效性；在准确率-长度权衡方面实现了更高准确率和更短推理长度；在内部推理质量方面获得了显著更高的分数(+7.2)。

### 结论

自我重写方法能够有效提高大型推理模型的内部推理质量，同时提高准确率和减少推理长度。

### 翻译

通过带有结果正确性奖励的强化学习，具有扩展推理计算的大型推理模型在复杂推理任务上展示了显著成功。然而，仅关注最终正确性的单向奖励限制了其对内部推理过程提供详细监督的能力。这种缺陷导致内部推理质量不佳，表现为过度思考、思考不足、冗余思考和无序思考等问题。受LRM自我奖励进展的启发，我们引入了自我重写框架，模型重写自己的推理文本，然后从重写的推理中学习，以提高内部思维过程质量。在算法设计上，我们提出选择性重写方法，仅重写模型一致正确的'简单'样本，从而保留GRPO的所有原始奖励信号。在实际实现中，我们将重写和普通生成编译在一个批次中，保持RL算法的可扩展性，仅引入约10%的开销。在不同模型大小的多种任务上的广泛实验验证了自我重写的有效性。在准确率-长度权衡方面，自我重写方法实现了更高的准确率(+0.6)和显著更短的推理长度(-46%)，即使在重写提示中没有明确指示减少推理长度的情况下，也优于现有的强基线。在内部推理质量方面，自我重写在LLM作为评判者的指标上获得了显著更高的分数(+7.2)，成功缓解了内部推理缺陷。


### 论文摘要

Through reinforcement learning (RL) with outcome correctness rewards, large reasoning models (LRMs) with scaled inference computation have demonstrated substantial success on complex reasoning tasks. However, the one-sided reward, focused solely on final correctness, limits its ability to provide detailed supervision over internal reasoning process. This deficiency leads to suboptimal internal reasoning quality, manifesting as issues like over-thinking, under-thinking, redundant-thinking, and disordered-thinking. Inspired by the recent progress in LRM self-rewarding, we introduce self-rewriting framework, where a model rewrites its own reasoning texts, and subsequently learns from the rewritten reasoning to improve the internal thought process quality. For algorithm design, we propose a selective rewriting approach wherein only "simple" samples, defined by the model's consistent correctness, are rewritten, thereby preserving all original reward signals of GRPO. For practical implementation, we compile rewriting and vanilla generation within one single batch, maintaining the scalability of the RL algorithm and introducing only ~10% overhead. Extensive experiments on diverse tasks with different model sizes validate the effectiveness of self-rewriting. In terms of the accuracy-length tradeoff, the self-rewriting approach achieves improved accuracy (+0.6) with substantially shorter reasoning (-46%) even without explicit instructions in rewriting prompts to reduce reasoning length, outperforming existing strong baselines. In terms of internal reasoning quality, self-rewriting achieves significantly higher scores (+7.2) under the LLM-as-a-judge metric, successfully mitigating internal reasoning flaws.

---

## 48. ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning

**论文链接:** [http://arxiv.org/abs/2511.16326v1](http://arxiv.org/abs/2511.16326v1)

**作者:** Jiawei Zhou, Hang Ding, Haiyun Jiang

**发布时间:** 2025-11-20

**备注:** Under Review in ARR

### GPT解析

### 总结

这篇论文提出了一种新的微调框架，用于优化检索增强生成（RAG）系统中的检索器，使其能够更好地识别与答案相关的关键证据，从而提高长上下文场景下的性能。

### 背景

检索增强生成（RAG）已成为知识密集型任务的有力框架，但在长上下文场景中，其效果常受限于检索器无法区分稀疏但关键的证据。标准检索器针对查询-文档相似度进行优化，但往往无法与生成精确答案的下游目标保持一致。

### 目的

为了解决检索器与下游生成目标之间的差距，作者提出了一个新的微调框架，优化检索器使其能够更好地与答案对齐，提高长上下文RAG系统的性能。

### 方法

作者首先通过评估生成正确答案的充分性来识别高质量的正向块，然后采用基于课程的对比学习方案来微调检索器。该方法利用大语言模型构建的知识图谱生成增强查询，进而挖掘具有渐进挑战性的困难负例，训练检索器区分答案充分的正向块和这些微妙的干扰项。

### 主要发现

在Ultradomain和LongBench基准的10个数据集上的大量实验表明，微调后的检索器实现了最先进的性能，比基础模型提高了14.5%，而无需对架构进行重大修改，同时保持了长上下文RAG的强效效率。

### 结论

该工作提出了一种健壮且有效的方法，用于构建真正以答案为中心的检索器，解决了RAG在长上下文场景中的关键瓶颈问题。

### 翻译

检索增强生成（RAG）已成为知识密集型任务的有力框架，然而在长上下文场景中，其效果常受限于检索器无法区分稀疏但关键的证据。标准检索器针对查询-文档相似度进行优化，经常无法与生成精确答案的下游目标保持一致。为了弥合这一差距，我们提出了一种新颖的微调框架，优化检索器以实现答案对齐。具体而言，我们首先通过评估生成正确答案的充分性来识别高质量的正向块。然后，我们采用基于课程的对比学习方案来微调检索器。该课程利用大语言模型构建的知识图谱生成增强查询，进而挖掘具有渐进挑战性的困难负例。这一过程训练检索器区分答案充分的正向块和这些微妙的干扰项，增强其泛化能力。在Ultradomain和LongBench基准的10个数据集上的大量实验表明，我们的微调检索器实现了最先进的性能，比基础模型提高了14.5%，而无需对架构进行重大修改，同时保持了长上下文RAG的强效效率。我们的工作为构建真正以答案为中心的检索器提供了一种健壮且有效的方法论。


### 论文摘要

Retrieval-Augmented Generation (RAG) has emerged as a powerful framework for knowledge-intensive tasks, yet its effectiveness in long-context scenarios is often bottlenecked by the retriever's inability to distinguish sparse yet crucial evidence. Standard retrievers, optimized for query-document similarity, frequently fail to align with the downstream goal of generating a precise answer. To bridge this gap, we propose a novel fine-tuning framework that optimizes the retriever for Answer Alignment. Specifically, we first identify high-quality positive chunks by evaluating their sufficiency to generate the correct answer. We then employ a curriculum-based contrastive learning scheme to fine-tune the retriever. This curriculum leverages LLM-constructed Knowledge Graphs (KGs) to generate augmented queries, which in turn mine progressively challenging hard negatives. This process trains the retriever to distinguish the answer-sufficient positive chunks from these nuanced distractors, enhancing its generalization. Extensive experiments on 10 datasets from the Ultradomain and LongBench benchmarks demonstrate that our fine-tuned retriever achieves state-of-the-art performance, improving 14.5% over the base model without substantial architectural modifications and maintaining strong efficiency for long-context RAG. Our work presents a robust and effective methodology for building truly answer-centric retrievers.

---

## 49. Explainable AI for Diabetic Retinopathy Detection Using Deep Learning with Attention Mechanisms and Fuzzy Logic-Based Interpretability

**论文链接:** [http://arxiv.org/abs/2511.16294v1](http://arxiv.org/abs/2511.16294v1)

**作者:** Abishek Karthik, Pandiyaraju V, Sreya Mynampati

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种混合深度学习框架用于杂草检测，结合了CNNs、ViTs和GNNs，通过GAN数据增强和自监督对比预训练方法，在多个基准数据集上实现了99.33%的高准确率、精确率、召回率和F1分数。

### 背景

杂草检测是精准农业的重要组成部分，准确的物种识别使农民能够有选择性地应用除草剂，符合可持续农业的作物管理理念。

### 目的

开发一种能够在多种田间条件下保持鲁棒性的杂草检测混合深度学习框架。

### 方法

结合卷积神经网络(CNNs)、视觉变换器(ViTs)和图神经网络(GNNs)构建混合框架；采用基于生成对抗网络(GAN)的数据增强方法平衡类别分布并提高模型泛化能力；使用自监督对比预训练方法从有限的标注数据中学习更多特征。

### 主要发现

所提出的模型在多个基准数据集上达到了99.33%的准确率、精确率、召回率和F1分数；模型架构能够实现局部、全局和关系特征表示，并提供高可解释性和适应性。

### 结论

该框架允许在边缘设备上实时、高效地部署自动化杂草检测系统，减少对除草剂的过度依赖，并提供可扩展的可持续精准农业选择。

### 翻译

杂草检测任务是精准农业的重要组成部分，因为准确的物种识别允许农民有选择性地使用除草剂，并符合可持续农业的作物管理。本文提出了一种用于杂草检测的混合深度学习框架，利用卷积神经网络(CNNs)、视觉变换器(ViTs)和图神经网络(GNNs)构建对多种田间条件的鲁棒性。采用基于生成对抗网络(GAN)的数据增强方法来平衡类别分布并提高模型泛化能力。此外，自监督对比预训练方法有助于从有限的标注数据中学习更多特征。实验结果在多个基准数据集上取得了99.33%的准确率、精确率、召回率和F1分数的优越结果。所提出的模型架构能够实现局部、全局和关系特征表示，并提供高可解释性和适应性。实际上，该框架允许在边缘设备上实时、高效地部署自动化杂草检测，减少对除草剂的过度依赖，并提供可扩展的可持续精准农业选择。


### 论文摘要

The task of weed detection is an essential element of precision agriculture since accurate species identification allows a farmer to selectively apply herbicides and fits into sustainable agriculture crop management. This paper proposes a hybrid deep learning framework recipe for weed detection that utilizes Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and Graph Neural Networks (GNNs) to build robustness to multiple field conditions. A Generative Adversarial Network (GAN)-based augmentation method was imposed to balance class distributions and better generalize the model. Further, a self-supervised contrastive pre-training method helps to learn more features from limited annotated data. Experimental results yield superior results with 99.33% accuracy, precision, recall, and F1-score on multi-benchmark datasets. The proposed model architecture enables local, global, and relational feature representations and offers high interpretability and adaptability. Practically, the framework allows real-time, efficient deployment of edge devices for automated weed detecting, reducing over-reliance on herbicides and providing scalable, sustainable precision-farming options.

---

## 50. EvoVLA: Self-Evolving Vision-Language-Action Model

**论文链接:** [http://arxiv.org/abs/2511.16166v1](http://arxiv.org/abs/2511.16166v1)

**作者:** Zeting Liu, Zida Yang, Zeyu Zhang, Hao Tang

**发布时间:** 2025-11-20

### GPT解析

### 总结

EvoVLA是一个自监督视觉语言行动框架，通过三个互补组件解决长视距机器人操作中的阶段幻觉问题，在基准测试和实际部署中表现出色。

### 背景

尽管在零样本泛化和仿真到现实迁移方面取得了进展，但长视距机器人操作对视觉语言行动模型仍然具有挑战性。当前VLA模型存在阶段幻觉问题，即代理利用粗略评估信号绕过多步骤任务，报告高进度而未真正完成任务。

### 目的

提出EvoVLA自监督VLA框架，解决长视距机器人操作中的阶段幻觉问题。

### 方法

EvoVLA包含三个互补组件：1)阶段对齐奖励(SAR)：使用三元对比学习和Gemini生成的硬负样本来防止视觉捷径；2)基于姿态的对象探索(POE)：将好奇心锚定在相对对象-夹持器姿态上；3)长视距记忆：使用选择性上下文保留和门控融合来稳定扩展运行过程中的内在塑形。

### 主要发现

在Discoverse-L基准测试上，EvoVLA比最强基线(OpenVLA-OFT)平均提高10.2个百分点的任务成功率(达69.2%)，实现1.5倍的更好样本效率，并将阶段幻觉从38.5%减少到14.8%。在物理机器人上的实际部署在四个任务上平均成功率达54.6%，比OpenVLA-OFT高11个百分点。

### 结论

EvoVLA有效解决了长视距机器人操作中的阶段幻觉问题，并在多个基准测试和实际部署中表现出色，证明了有效的仿真到现实迁移和强大的泛化能力。

### 翻译

尽管在零样本泛化和仿真到现实世界的迁移方面最近取得了进展，但长视距机器人操作对视觉语言行动(VLA)模型仍然具有挑战性。当前的VLA模型存在阶段幻觉问题，即代理利用粗略的评估信号来绕过多步骤任务，报告高进度而没有真正完成它们。我们提出了EvoVLA，一个自监督的VLA框架，通过三个互补组件解决这个问题：阶段对齐奖励(SAR)，它使用三元对比学习和Gemini生成的硬负样本来防止视觉捷径；基于姿态的对象探索(POE)，它将好奇心锚定在相对对象-夹持器姿态而不是原始像素上；以及长视距记忆，它使用选择性上下文保留和门控融合来稳定扩展运行过程中的内在塑形。在Discoverse-L(一个包含三个多阶段任务的长视距操作基准)上的广泛评估显示，EvoVLA比最强基线(OpenVLA-OFT)平均提高了10.2个百分点的任务成功率，达到69.2%。EvoVLA还实现了1.5倍的更好的样本效率，并将阶段幻觉从38.5%减少到14.8%。在物理机器人上的实际部署在四个操作任务上平均成功率达到54.6%，比OpenVLA-OFT高出11个百分点，证明了有效的仿真到现实迁移和强大的泛化能力。代码：https://github.com/AIGeeksGroup/EvoVLA。网站：https://aigeeksgroup.github.io/EvoVLA。


### 论文摘要

Long-horizon robotic manipulation remains challenging for Vision-Language-Action (VLA) models despite recent progress in zero-shot generalization and simulation-to-real-world transfer. Current VLA models suffer from stage hallucination, where agents exploit coarse evaluation signals to shortcut multi-step tasks, reporting high progress without truly completing them. We present EvoVLA, a self-supervised VLA framework that addresses this issue through three complementary components: Stage-Aligned Reward (SAR), which uses triplet contrastive learning with Gemini-generated hard negatives to prevent visual shortcuts; Pose-Based Object Exploration (POE), which grounds curiosity in relative object-gripper pose instead of raw pixels; and Long-Horizon Memory, which uses selective context retention and gated fusion to stabilize intrinsic shaping during extended rollouts. Extensive evaluations on Discoverse-L, a long-horizon manipulation benchmark with three multi-stage tasks, show that EvoVLA improves average task success by 10.2 percentage points over the strongest baseline (OpenVLA-OFT), reaching 69.2 percent. EvoVLA also achieves one-and-a-half times better sample efficiency and reduces stage hallucination from 38.5 percent to 14.8 percent. Real-world deployment on physical robots reaches an average success rate of 54.6 percent across four manipulation tasks, outperforming OpenVLA-OFT by 11 points, demonstrating effective sim-to-real transfer and strong generalization. Code: https://github.com/AIGeeksGroup/EvoVLA. Website: https://aigeeksgroup.github.io/EvoVLA.

---

## 51. CoSP: Reconfigurable Multi-State Metamaterial Inverse Design via Contrastive Pretrained Large Language Model

**论文链接:** [http://arxiv.org/abs/2511.16135v1](http://arxiv.org/abs/2511.16135v1)

**作者:** Shujie Yang, Xuzhe Zhao, Yuqi Zhang, Yansong Tang, Kaichen Dong

**发布时间:** 2025-11-20

**备注:** 5 pages, 6 figures

### GPT解析

### 总结

本文提出了一种名为CoSP的智能逆向设计方法，基于对比预训练大语言模型，解决了可重构多态超材料设计中多态切换的挑战，能够为任意多态、多波段光学响应设计相应的薄膜超材料结构。

### 背景

超材料能够亚波长尺度操控光，但因其复杂结构面临重大设计挑战。深度学习已成为简化超材料设计过程的有力工具。可重构多态超材料具有可调参数，可通过外部刺激在不同状态间切换光学特性。

### 目的

解决现有基于深度学习的逆向设计方法未能充分考虑多态切换可重构性的问题，提出一种能够处理多态可重构性的智能逆向设计方法。

### 方法

提出CoSP方法，通过对多态光谱进行对比预训练获得能够理解的光谱编码器，该编码器与预训练的大语言模型交互，使模型能够保留语言能力同时理解麦克斯韦方程，用自然语言描述具有目标光学特性的材料结构。

### 主要发现

实验证明CoSP能够为任意多态、多波段光学响应设计相应的薄膜超材料结构。

### 结论

CoSP在可重构多态超材料的智能设计方面具有巨大潜力，可应用于多种场景。

### 翻译

Metamaterials, known for their ability to manipulate light at subwavelength scales, face significant design challenges due to their complex and sophisticated structures. Consequently, deep learning has emerged as a powerful tool to streamline their design process. Reconfigurable multi-state metamaterials (RMMs) with adjustable parameters can switch their optical characteristics between different states upon external stimulation, leading to numerous applications. However, existing deep learning-based inverse design methods fall short in considering reconfigurability with multi-state switching. To address this challenge, we propose CoSP, an intelligent inverse design method based on contrastive pretrained large language model (LLM). By performing contrastive pretraining on multi-state spectrum, a well-trained spectrum encoder capable of understanding the spectrum is obtained, and it subsequently interacts with a pretrained LLM. This approach allows the model to preserve its linguistic capabilities while also comprehending Maxwell's Equations, enabling it to describe material structures with target optical properties in natural language. Our experiments demonstrate that CoSP can design corresponding thin-film metamaterial structures for arbitrary multi-state, multi-band optical responses, showing great potentials in the intelligent design of RMMs for versatile applications.


### 论文摘要

Metamaterials, known for their ability to manipulate light at subwavelength scales, face significant design challenges due to their complex and sophisticated structures. Consequently, deep learning has emerged as a powerful tool to streamline their design process. Reconfigurable multi-state metamaterials (RMMs) with adjustable parameters can switch their optical characteristics between different states upon external stimulation, leading to numerous applications. However, existing deep learning-based inverse design methods fall short in considering reconfigurability with multi-state switching. To address this challenge, we propose CoSP, an intelligent inverse design method based on contrastive pretrained large language model (LLM). By performing contrastive pretraining on multi-state spectrum, a well-trained spectrum encoder capable of understanding the spectrum is obtained, and it subsequently interacts with a pretrained LLM. This approach allows the model to preserve its linguistic capabilities while also comprehending Maxwell's Equations, enabling it to describe material structures with target optical properties in natural language. Our experiments demonstrate that CoSP can design corresponding thin-film metamaterial structures for arbitrary multi-state, multi-band optical responses, showing great potentials in the intelligent design of RMMs for versatile applications.

---

## 52. Crossmodal learning for Crop Canopy Trait Estimation

**论文链接:** [http://arxiv.org/abs/2511.16031v1](http://arxiv.org/abs/2511.16031v1)

**作者:** Timilehin T. Ayanlade, Anirudha Powadi, Talukder Z. Jubery, Baskar Ganapathysubramanian, Soumik Sarkar

**发布时间:** 2025-11-20

**备注:** 18 pages, 7 figures

### GPT解析

### 总结

本研究提出了一种跨模态学习策略，通过将无人机级别的视觉细节融入高分辨率卫星图像，用于作物冠层性状估计。研究使用美国玉米带五个地点84个杂交玉米品种的卫星-无人机图像对数据集，训练模型学习不同传感模态间的细粒度光谱-空间对应关系。结果表明，从卫星输入生成的类无人机表示在产量和氮预测等下游任务中优于真实卫星图像，证明了跨模态学习在弥合卫星和无人机农业监测差距方面的潜力。

### 背景

植物表型研究的最新进展推动了多传感器平台在收集作物冠层反射率数据方面的广泛应用。无人机(UAV)因在作物监测、预测任务中的高性能而得到显著使用，卫星任务也被证明对农业相关任务有效。然而，卫星受限于空间分辨率，影响了其在微区管理现代农业系统中的应用效果。

### 目的

提出一种跨模态学习策略，通过添加无人机级别的视觉细节来丰富高分辨率卫星图像，用于作物冠层性状估计，以弥合卫星和无人机在农业监测中的差距。

### 方法

使用从美国玉米带五个不同地点的84个杂交玉米品种的重复地块收集的大致配准的卫星-无人机图像对数据集，训练一个模型，学习不同传感模态之间的细粒度光谱-空间对应关系。

### 主要发现

从卫星输入生成的类无人机表示在多个下游任务中持续优于真实卫星图像，包括产量和氮预测任务，证明了跨模态对应学习在弥合卫星和无人机农业监测差距方面的潜力。

### 结论

跨模态对应学习有潜力弥合卫星和无人机农业监测之间的差距，为现代农业监测提供更有效的解决方案。

### 翻译

植物表型研究的最新进展推动了多传感器平台在收集作物冠层反射率数据方面的广泛采用。这包括跨多个平台收集异构数据，其中无人机(UAV)因在作物监测、预测和预测任务中的高性能而得到显著使用。同样，卫星任务也被证明对农业相关任务有效。与无人机相比，此类任务受空间分辨率的限制，这阻碍了它们专注于微区管理的现代农业系统的有效性。在本工作中，我们提出了一种跨模态学习策略，通过添加无人机级别的视觉细节来丰富高分辨率卫星图像，用于作物冠层性状估计。使用从美国玉米带五个不同地点的84个杂交玉米品种的重复地块收集的大致配准的卫星-无人机图像对数据集，我们训练了一个学习不同传感模态之间细粒度光谱-空间对应关系的模型。结果表明，从卫星输入生成的类无人机表示在多个下游任务中持续优于真实卫星图像，包括产量和氮预测，证明了跨模态对应学习在弥合卫星和无人机农业监测差距方面的潜力。


### 论文摘要

Recent advances in plant phenotyping have driven widespread adoption of multi sensor platforms for collecting crop canopy reflectance data. This includes the collection of heterogeneous data across multiple platforms, with Unmanned Aerial Vehicles (UAV) seeing significant usage due to their high performance in crop monitoring, forecasting, and prediction tasks. Similarly, satellite missions have been shown to be effective for agriculturally relevant tasks. In contrast to UAVs, such missions are bound to the limitation of spatial resolution, which hinders their effectiveness for modern farming systems focused on micro-plot management. In this work, we propose a cross modal learning strategy that enriches high-resolution satellite imagery with UAV level visual detail for crop canopy trait estimation. Using a dataset of approximately co registered satellite UAV image pairs collected from replicated plots of 84 hybrid maize varieties across five distinct locations in the U.S. Corn Belt, we train a model that learns fine grained spectral spatial correspondences between sensing modalities. Results show that the generated UAV-like representations from satellite inputs consistently outperform real satellite imagery on multiple downstream tasks, including yield and nitrogen prediction, demonstrating the potential of cross-modal correspondence learning to bridge the gap between satellite and UAV sensing in agricultural monitoring.

---

## 53. Self-supervised and Multi-fidelity Learning for Extended Predictive Soil Spectroscopy

**论文链接:** [http://arxiv.org/abs/2511.15965v1](http://arxiv.org/abs/2511.15965v1)

**作者:** Luning Sun, José L. Safanelli, Jonathan Sanderman, Katerina Georgiou, Colby Brungard, Kanchan Grover, Bryan G. Hopkins, Shusen Liu, Timo Bremer

**发布时间:** 2025-11-20

**备注:** 49 pages, 9 figures, submitted to Geoderma

### GPT解析

### 总结

提出了一种基于潜在空间嵌入的自监督机器学习框架，用于多保真度学习和扩展预测土壤光谱学，通过NIR与MIR光谱间的映射实现大型MIR光谱库的预测能力。

### 背景

土壤光谱分析领域需要利用大型MIR光谱数据库的丰富预测能力，同时保持低成本便携NIR扫描器的实用性。

### 目的

开发一种自监督机器学习框架，通过潜在空间嵌入实现多保真度学习，并扩展预测土壤光谱学的能力，特别是利用大型MIR光谱库的预测能力。

### 方法

使用大型MIR光谱库和变分自编码器算法进行自监督表示预训练获取压缩的潜在空间；冻结训练好的MIR解码器并将其插入NIR编码器中学习光谱间映射；使用KSSL库中较小的一对NIR和MIR光谱子集；训练下游机器学习模型映射原始光谱、预测光谱和九种土壤属性的潜在空间嵌入。

### 主要发现

提出的SSML框架在所有土壤属性预测任务中实现了与基线模型相似或更好的准确性；光谱转换任务的预测性能不如原始MIR光谱，但与仅使用NIR的模型相似或更好；统一的光谱潜在空间可以有效地利用更大更多样化的MIR数据集来预测当前NIR库中代表性不足的土壤属性。

### 结论

自监督机器学习框架能够有效利用大型MIR光谱数据库的预测能力，同时保持低成本便携NIR扫描器的实用性，为土壤属性预测提供了一种有效方法。

### 翻译

我们提出了一种基于潜在空间嵌入的自监督机器学习框架，用于多保真度学习和扩展预测土壤光谱学。使用大型MIR光谱库和变分自编码器算法进行自监督表示预训练，以获取压缩的潜在空间用于生成光谱嵌入。在此阶段仅使用未标记的光谱数据，使我们能够利用完整的光谱数据库和扫描重复数据用于增强训练。我们还利用并冻结了训练好的MIR解码器用于光谱转换任务，将其插入NIR编码器中学习NIR与MIR光谱之间的映射，试图利用大型MIR库中包含的预测能力，同时使用低成本便携的NIR扫描器。这是通过使用KSSL库中较小的一对NIR和MIR光谱子集实现的。然后训练下游机器学习模型，用于映射原始光谱、预测光谱和九种土壤属性的潜在空间嵌入。使用金标准测试集独立评估性能，不包括KSSL训练数据，同时使用回归拟合优度指标。与基线模型相比，提出的SSML及其嵌入在所有土壤属性预测任务中实现了相似或更好的准确性。来自光谱转换（NIR到MIR）任务的预测性能不如原始MIR光谱，但与仅使用NIR的模型相似或更好，这表明统一的光谱潜在空间可以有效地利用更大更多样化的MIR数据集来预测当前NIR库中代表性不足的土壤属性。


### 论文摘要

We propose a self-supervised machine learning (SSML) framework for multi-fidelity learning and extended predictive soil spectroscopy based on latent space embeddings. A self-supervised representation was pretrained with the large MIR spectral library and the Variational Autoencoder algorithm to obtain a compressed latent space for generating spectral embeddings. At this stage, only unlabeled spectral data were used, allowing us to leverage the full spectral database and the availability of scan repeats for augmented training. We also leveraged and froze the trained MIR decoder for a spectrum conversion task by plugging it into a NIR encoder to learn the mapping between NIR and MIR spectra in an attempt to leverage the predictive capabilities contained in the large MIR library with a low cost portable NIR scanner. This was achieved by using a smaller subset of the KSSL library with paired NIR and MIR spectra. Downstream machine learning models were then trained to map between original spectra, predicted spectra, and latent space embeddings for nine soil properties. The performance of was evaluated independently of the KSSL training data using a gold-standard test set, along with regression goodness-of-fit metrics. Compared to baseline models, the proposed SSML and its embeddings yielded similar or better accuracy in all soil properties prediction tasks. Predictions derived from the spectrum conversion (NIR to MIR) task did not match the performance of the original MIR spectra but were similar or superior to predictive performance of NIR-only models, suggesting the unified spectral latent space can effectively leverage the larger and more diverse MIR dataset for prediction of soil properties not well represented in current NIR libraries.

---

## 54. Boosting Medical Visual Understanding From Multi-Granular Language Learning

**论文链接:** [http://arxiv.org/abs/2511.15943v1](http://arxiv.org/abs/2511.15943v1)

**作者:** Zihan Li, Yiqing Wang, Sina Farsiu, Paul Kinahan

**发布时间:** 2025-11-20

**备注:** Preprint. 40 pages

### GPT解析

### 总结

本文提出了一种名为多粒度语言学习（MGLL）的对比学习框架，旨在解决现有图像-文本预训练方法在处理多标签和跨粒度对齐时的局限性，特别是在医学影像等复杂领域。

### 背景

图像-文本预训练的最新进展显著增强了视觉理解能力，通过将视觉和文本表示对齐。对比语言-图像预训练（CLIP）在多模态学习中发挥了关键作用，但其专注于单标签、单一粒度对齐的限制在复杂领域如医学影像中表现不佳。

### 目的

设计一个能够同时改进多标签和跨粒度对齐的对比学习框架，以应对医学影像等复杂领域中图像对应多个高级标签和不同注释粒度的挑战。

### 方法

MGLL利用结构化多标签监督，整合不同粒度的文本描述，并引入带点约束的软标签监督来增强对齐。使用平滑的Kullback-Leibler散度确保跨粒度一致性，同时保持计算效率，作为视觉-语言模型的即插即用模块。

### 主要发现

在构建的大规模多粒度数据集上预训练，并在多个数据集上评估，MGLL在下游任务中优于其他最先进的方法。

### 结论

MGLL是一种有效的多模态学习框架，能够处理多标签和跨粒度对齐问题，特别适用于医学影像等复杂领域。

### 翻译

图像-文本预训练的最新进展通过将视觉和文本表示对齐，显著增强了视觉理解能力。对比语言-图像预训练（CLIP）在多模态学习中发挥了关键作用。然而，其专注于单标签、单一粒度对齐的特性限制了其在医学影像等复杂领域的有效性，这些领域的图像通常对应多个高级标签（如疾病类别）以及不同注释粒度（如诊断描述、临床解释）。为解决这一问题，我们提出了多粒度语言学习（MGLL），这是一种对比学习框架，旨在改进多标签和跨粒度对齐。MGLL利用结构化多标签监督，整合不同粒度的文本描述，并引入带点约束的软标签监督来增强对齐。MGLL使用平滑的Kullback-Leibler散度确保跨粒度一致性，同时保持计算效率，作为视觉-语言模型的即插即用模块。在我们构建的大规模多粒度数据集上预训练，并在多个数据集上评估，MGLL在下游任务中优于其他最先进的方法。代码可在https://github.com/HUANGLIZI/MGLL获取。


### 论文摘要

Recent advances in image-text pretraining have significantly enhanced visual understanding by aligning visual and textual representations. Contrastive Language-Image Pretraining (CLIP) has played a pivotal role in multimodal learning. However, its focus on single-label, single-granularity alignment limits its effectiveness in complex domains such as medical imaging, where images often correspond to multiple high-level labels (e.g., disease categories) across different annotation granularities (e.g., diagnostic description, clinical explanation). To address this, we propose Multi-Granular Language Learning (MGLL), a contrastive learning framework designed to improve both multi-label and cross-granularity alignment. MGLL leverages structured multi-label supervision, integrates textual descriptions across granularities, and introduces soft-label supervision with point-wise constraints to enhance alignment. MGLL employs smooth Kullback-Leibler (KL) divergence to ensure cross-granularity consistency while maintaining computational efficiency as a plug-and-play module for vision-language models. Pretrained on our constructed large-scale multi-granular datasets and evaluated across multiple datasets, MGLL outperforms other state-of-the-art methods in downstream tasks. The code is available at \href{https://github.com/HUANGLIZI/MGLL}{https://github.com/HUANGLIZI/MGLL}.

---

## 55. SURFing to the Fundamental Limit of Jet Tagging

**论文链接:** [http://arxiv.org/abs/2511.15779v1](http://arxiv.org/abs/2511.15779v1)

**作者:** Ian Pang, Darius A. Faroughy, David Shih, Ranit Das, Gregor Kasieczka

**发布时间:** 2025-11-19

**备注:** 15 pages, 10 figures, 2 tables

### GPT解析

### 总结

这项研究介绍了SURF方法，用于验证生成模型在喷注分类任务上的性能上限，发现现代喷注分类器已接近真正的统计极限，而某些生成模型可能高估了这一极限。

### 背景

除了通过改进喷注标记算法提高搜索和测量灵敏度的实际目标外，还存在一个更深层次的问题：这些算法的性能上限是什么？具有学习似然函数的生成代理模型为解决这个问题提供了新方法，前提是代理模型能正确捕获底层数据分布。

### 目的

研究喷注标记算法的性能上限，并开发一种新方法来验证生成模型在捕捉这种上限方面的有效性。

### 方法

作者引入了SURF方法，通过从另一个可处理的代理模型（该代理模型本身在真实数据上训练）中采样来训练目标模型，从而实现精确的Neyman-Pearson检验。

### 主要发现

1. EPiC-FM生成模型是JetClass喷注的有效代理参考；2. 现代喷注分类器可能已经接近真正的统计极限；3. 自回归GPT模型在顶夸克与QCD分离能力方面存在物理上不合理的夸大，可能给出误导性的基本极限描述。

### 结论

SURF方法为评估生成模型在粒子物理喷注分类任务中的性能提供了新工具，表明现代喷注分类器已经接近性能上限，而某些生成模型可能高估了这一极限。

### 翻译

除了通过改进喷注标记算法来提高搜索和测量灵敏度的实际目标外，还存在一个更深层次的问题：它们的性能上限是什么？具有学习似然函数的生成代理模型为解决这个问题提供了新方法，前提是代理模型能正确捕获底层数据分布。在这项工作中，我们介绍了SURF方法，这是一种验证生成模型的新方法。该框架通过从另一个可处理的代理模型（该代理模型本身在真实数据上训练）中采样来训练目标模型，从而实现精确的Neyman-Pearson检验。我们认为EPiC-FM生成模型是JetClass喷注的有效代理参考，并应用SURF方法表明现代喷注分类器可能已经接近真正的统计极限。相比之下，我们发现自回归GPT模型在代理参考中编码的顶夸克与QCD分离能力方面存在物理上不合理的夸大，这意味着它们对基本极限给出了误导性的描述。


### 论文摘要

Beyond the practical goal of improving search and measurement sensitivity through better jet tagging algorithms, there is a deeper question: what are their upper performance limits? Generative surrogate models with learned likelihood functions offer a new approach to this problem, provided the surrogate correctly captures the underlying data distribution. In this work, we introduce the SUrrogate ReFerence (SURF) method, a new approach to validating generative models. This framework enables exact Neyman-Pearson tests by training the target model on samples from another tractable surrogate, which is itself trained on real data. We argue that the EPiC-FM generative model is a valid surrogate reference for JetClass jets and apply SURF to show that modern jet taggers may already be operating close to the true statistical limit. By contrast, we find that autoregressive GPT models unphysically exaggerate top vs. QCD separation power encoded in the surrogate reference, implying that they are giving a misleading picture of the fundamental limit.

---

## 56. Bayesian polarization calibration and imaging in very long baseline interferometry

**论文链接:** [http://arxiv.org/abs/2511.16556v1](http://arxiv.org/abs/2511.16556v1)

**作者:** Jong-Seo Kim, Jakob Roth, Jongho Park, Jack D. Livingston, Philipp Arras, Torsten A. Enßlin, Michael Janssen, J. Anton Zensus, Andrei P. Lobanov

**发布时间:** 2025-11-20

**备注:** submitted to A&A

### GPT解析

### 总结

该研究提出了一种贝叶斯偏振校准和成像方法，用于从VLBI数据中提取偏振信息，相比传统CLEAN方法能提供更高分辨率和更准确的图像，并自动处理校准不确定性。

### 背景

从VLBI数据中提取偏振信息具有挑战性但对理解同步辐射过程和天体（如活动星系核）的磁场至关重要。传统基于CLEAN的校准和成像方法分辨率次优，无法提供校准解的不确定性估计，且需要经验丰富的用户手动操作。

### 目的

开发一种自动化的贝叶斯偏振校准和成像方法，能够从VLBI数据中获取高保真度偏振图像，并提供校准解的不确定性估计。

### 方法

使用贝叶斯成像软件resolve，从预校准数据中联合探索基于天线的增益、偏振泄漏和偏振图像的后验分布。通过15GHz频率的VLBA对类星体3C273和86GHz频率的GMVA+ALMA对耀变体OJ287的观测进行验证。

### 主要发现

与传统CLEAN方法相比，贝叶斯方法提供物理上真实的图像，满足通量和偏振约束的正性，能够重建由各种空间尺度组成的复杂源结构，系统性地考虑校准不确定性，并提供Stokes图像和校准解的不确定性。

### 结论

自动化的贝叶斯校准和成像方法能够利用下一代射电阵列的高质量数据获取高保真度偏振图像，所开发的流程已公开可用。

### 翻译

从甚长基线干涉测量数据中提取偏振信息具有挑战性但对理解同步辐射过程和天体（如活动星系核）的磁场至关重要。然而，传统的基于CLEAN的校准和成像方法分辨率次优，无法提供校准解的不确定性估计，且需要经验丰富的用户手动操作。我们提出了一种使用贝叶斯成像软件resolve的贝叶斯偏振校准和成像方法，用于VLBI数据集，从预校准数据中联合探索基于天线的增益、偏振泄漏和偏振图像的后验分布。我们使用15GHz频率的VLBA对类星体3C273和86GHz频率的GMVA+ALMA对耀变体OJ287的观测来验证我们的校准和成像方法。与CLEAN方法相比，我们的方法提供物理上真实的图像，满足通量和偏振约束的正性，并能重建由各种空间尺度组成的复杂源结构。我们的方法系统性地考虑了最终图像中的校准不确定性，并提供Stokes图像和校准解的不确定性。用于此工作的自动化贝叶斯校准和成像方法将能够利用下一代射电阵列的高质量数据获取高保真度偏振图像。为此开发的流程已公开可用。


### 论文摘要

Extracting polarimetric information from very long baseline interferometry (VLBI) data is demanding but vital for understanding the synchrotron radiation process and the magnetic fields of celestial objects, such as active galactic nuclei (AGNs). However, conventional CLEAN-based calibration and imaging methods provide suboptimal resolution without uncertainty estimation of calibration solutions, while requiring manual steering from an experienced user. We present a Bayesian polarization calibration and imaging method using Bayesian imaging software resolve for VLBI data sets, that explores the posterior distribution of antenna-based gains, polarization leakages, and polarimetric images jointly from pre-calibrated data. We demonstrate our calibration and imaging method with observations of the quasar 3C273 with the VLBA at 15 GHz and the blazar OJ287 with the GMVA+ALMA at 86 GHz. Compared to the CLEAN method, our approach provides physically realistic images that satisfy positivity of flux and polarization constraints and can reconstruct complex source structures composed of various spatial scales. Our method systematically accounts for calibration uncertainties in the final images and provides uncertainties of Stokes images and calibration solutions. The automated Bayesian approach for calibration and imaging will be able to obtain high-fidelity polarimetric images using high-quality data from next-generation radio arrays. The pipeline developed for this work is publicly available.

---

## 57. LLaVA$^3$: Representing 3D Scenes like a Cubist Painter to Boost 3D Scene Understanding of VLMs

**论文链接:** [http://arxiv.org/abs/2511.16454v1](http://arxiv.org/abs/2511.16454v1)

**作者:** Doriand Petit, Steve Bourgeois, Vincent Gay-Bellile, Florian Chabot, Loïc Barthe

**发布时间:** 2025-11-20

**备注:** Accepted at AAAI'26

### GPT解析

### 总结

本文介绍了一种名为LLaVA³的新方法，通过仅使用多视角2D图像且无需微调，来增强视觉语言模型对3D场景的理解能力，解决了3D训练数据有限的问题。

### 背景

开发能够理解3D场景的多模态语言模型具有挑战性，主要是因为与用于视觉语言模型的丰富2D数据集相比，3D训练数据有限。

### 目的

提出一种新方法，利用多视角2D图像增强视觉语言模型的3D场景理解能力，而无需使用3D训练数据或进行微调。

### 方法

受到立体派画家的启发，研究人员提出通过每个物体的全方位视觉表示来描述3D场景，这些表示来自场景的中间多视角3D重建，无需直接使用3D训练数据或进行微调。

### 主要发现

在3D视觉问答和3D语言定位的广泛实验表明，该方法优于之前的基于2D的视觉语言模型解决方案。

### 结论

通过多视角2D图像和中间多视角3D重建，可以有效地增强视觉语言模型对3D场景的理解能力，而无需直接使用3D训练数据或进行微调。

### 翻译

由于3D训练数据有限，与用于视觉语言模型的丰富2D数据集相比，开发能够理解3D场景的多模态语言模型仍然具有挑战性。作为替代方案，我们引入了LLaVA³，这是一种新颖的方法，仅使用多视角2D图像且无需任何微调即可提高视觉语言模型的3D场景理解能力。受立体派画家的启发，他们在单一画作中表现3D物体的多个视角，我们提出通过每个物体的全方位视觉表示来描述视觉语言模型的3D场景。这些表示来自场景的中间多视角3D重建。在3D视觉问答和3D语言定位的广泛实验表明，我们的方法优于之前的基于2D的视觉语言模型解决方案。


### 论文摘要

Developing a multi-modal language model capable of understanding 3D scenes remains challenging due to the limited availability of 3D training data, in contrast to the abundance of 2D datasets used for vision-language models (VLM). As an alternative, we introduce LLaVA$^3$ (pronounced LLaVA-Cube), a novel method that improves the 3D scene understanding capabilities of VLM using only multi-view 2D images and without any fine-tuning. Inspired by Cubist painters, who represented multiple viewpoints of a 3D object within a single picture, we propose to describe the 3D scene for the VLM through omnidirectional visual representations of each object. These representations are derived from an intermediate multi-view 3D reconstruction of the scene. Extensive experiments on 3D VQA and 3D language grounding show that our approach outperforms previous 2D-based VLM solutions.

---

## 58. StreetView-Waste: A Multi-Task Dataset for Urban Waste Management

**论文链接:** [http://arxiv.org/abs/2511.16440v1](http://arxiv.org/abs/2511.16440v1)

**作者:** Diogo J. Paulo, João Martins, Hugo Proença, João C. Neves

**发布时间:** 2025-11-20

**备注:** Accepted at WACV 2026

### GPT解析

### 总结

这篇论文介绍了StreetView-Waste数据集，用于解决城市垃圾管理中的垃圾箱溢出监控问题，提供垃圾容器检测、跟踪和溢出分割三个任务的评估基准。

### 背景

城市垃圾管理是智能城市发展面临的关键挑战。尽管垃圾检测数据集数量不断增加，但监控垃圾箱溢出问题（特别是通过垃圾车拍摄图像）很少受到关注。现有数据集通常缺乏特定容器跟踪的标注或在静态、脱离实际环境中采集，限制了它们在现实物流中的应用价值。

### 目的

为了解决这一研究空白，作者提出了StreetView-Waste数据集，这是一个包含城市场景中垃圾和垃圾箱的综合数据集，支持垃圾容器检测、跟踪和溢出分割三个关键评估任务。

### 方法

作者提供了每个任务的基线，通过基准测试了最先进的检测、跟踪和分割模型。此外，作者提出了两种互补策略：一种基于启发式的方法用于改进垃圾容器跟踪，以及一种与模型无关的框架，利用几何先验来优化垃圾分割。

### 主要发现

经过微调的目标检测器在检测垃圾箱方面表现良好，但基线跟踪方法在准确估计垃圾箱数量方面表现不佳；提出的启发式方法将平均绝对计数误差降低了79.6%。分割无定形垃圾具有挑战性，但几何感知策略在轻量级模型上将分割mAP@0.5提高了27%，展示了多模态输入的价值。

### 结论

StreetView-Waste提供了一个具有挑战性的基准，以鼓励对城市垃圾管理的现实感知系统进行研究。

### 翻译

城市垃圾管理仍然是智能城市发展面临的严峻挑战。尽管垃圾检测数据集的数量不断增加，但监控垃圾箱溢出问题，特别是通过垃圾车拍摄图像的问题，很少受到关注。虽然现有数据集很有价值，但它们通常缺乏特定容器跟踪的标注或在静态、脱离实际的环境中采集，限制了它们在现实物流中的应用价值。为了解决这一研究空白，我们提出了StreetView-Waste，这是一个包含城市场景中垃圾和垃圾箱的综合数据集。该数据集支持三个关键评估任务：(1) 垃圾容器检测，(2) 垃圾容器跟踪，(3) 垃圾溢出分割。除了数据集外，我们还通过在目标检测、跟踪和分割领域对最先进的模型进行基准测试，为每个任务提供了基线。此外，我们提出了两种互补策略来增强基线性能：一种基于启发式的方法用于改进垃圾容器跟踪，以及一种与模型无关的框架，利用几何先验来优化垃圾分割。我们的实验结果表明，虽然经过微调的目标检测器在检测垃圾箱方面取得了合理的性能，但基线跟踪方法在准确估计垃圾箱数量方面表现不佳；然而，我们提出的启发式方法将平均绝对计数误差降低了79.6%。同样，虽然分割无定形垃圾具有挑战性，但我们的几何感知策略在轻量级模型上将分割mAP@0.5提高了27%，展示了多模态输入对该任务的价值。最终，StreetView-Waste提供了一个具有挑战性的基准，以鼓励对城市垃圾管理的现实感知系统进行研究。


### 论文摘要

Urban waste management remains a critical challenge for the development of smart cities. Despite the growing number of litter detection datasets, the problem of monitoring overflowing waste containers, particularly from images captured by garbage trucks, has received little attention. While existing datasets are valuable, they often lack annotations for specific container tracking or are captured in static, decontextualized environments, limiting their utility for real-world logistics. To address this gap, we present StreetView-Waste, a comprehensive dataset of urban scenes featuring litter and waste containers. The dataset supports three key evaluation tasks: (1) waste container detection, (2) waste container tracking, and (3) waste overflow segmentation. Alongside the dataset, we provide baselines for each task by benchmarking state-of-the-art models in object detection, tracking, and segmentation. Additionally, we enhance baseline performance by proposing two complementary strategies: a heuristic-based method for improved waste container tracking and a model-agnostic framework that leverages geometric priors to refine litter segmentation. Our experimental results show that while fine-tuned object detectors achieve reasonable performance in detecting waste containers, baseline tracking methods struggle to accurately estimate their number; however, our proposed heuristics reduce the mean absolute counting error by 79.6%. Similarly, while segmenting amorphous litter is challenging, our geometry-aware strategy improves segmentation mAP@0.5 by 27% on lightweight models, demonstrating the value of multimodal inputs for this task. Ultimately, StreetView-Waste provides a challenging benchmark to encourage research into real-world perception systems for urban waste management.

---

## 59. Graph Neural Networks for Surgical Scene Segmentation

**论文链接:** [http://arxiv.org/abs/2511.16430v1](http://arxiv.org/abs/2511.16430v1)

**作者:** Yihan Li, Nikhil Churamani, Maria Robu, Imanol Luengo, Danail Stoyanov

**发布时间:** 2025-11-20

**备注:** 12 pages, 4 figures, 3 tables

### GPT解析

### 总结

本研究提出了一种基于图的分割方法，用于腹腔镜胆囊手术中肝囊肿解剖结构的准确识别，解决了深度学习模型在处理遮挡、长距离依赖和稀有结构精细几何形状方面的困难。

### 背景

准确识别肝囊肿解剖结构对预防腹腔镜胆囊切除术中的手术并发症至关重要，但深度学习模型在处理遮挡、长距离依赖关系和捕捉稀有结构的精细几何形状方面存在挑战。

### 目的

开发基于图的分割方法，增强手术场景分析中的空间和语义理解，提高解剖结构识别的准确性。

### 方法

提出两种结合Vision Transformer (ViT)特征编码器和图神经网络(GNNs)的分割模型：(1)静态k近邻(k-NN)图与具有初始残差和恒等映射的图卷积网络(GCNII)；(2)动态可微分图生成器(DGG)与图注意力网络(GAT)。在Endoscopes-Seg50和CholecSeg8k基准上进行评估。

### 主要发现

与最先进的基线相比，所提出的方法在平均交并比(mIoU)上提高7-8%，在平均Dice分数(mDice)上提高6%，特别是在薄、稀有和关键安全结构的预测上产生了解剖上一致的预测结果。

### 结论

基于图的分割方法提高了手术场景分割的性能和解剖一致性，通过结合基于ViT的全局上下文和基于图的推理，提高了模型的可解释性和可靠性，为更安全的腹腔镜和机器人辅助手术铺平了道路。

### 翻译

目的：准确识别肝囊肿解剖结构对预防腹腔镜胆囊切除术中的手术并发症至关重要。深度学习模型通常难以处理遮挡、长距离依赖关系和捕捉稀有结构的精细几何形状。本研究通过引入基于图的分割方法来解决这些挑战，该方法在手术场景分析中增强了空间和语义理解。方法：我们提出两种分割模型，将Vision Transformer (ViT)特征编码器与图神经网络(GNNs)相结合，以明确建模解剖区域之间的空间关系。(1)静态k近邻(k-NN)图与具有初始残差和恒等映射的图卷积网络(GCNII)可实现稳定的长距离信息传播。(2)动态可微分图生成器(DGG)与图注意力网络(GAT)支持自适应拓扑学习。两种模型均在Endoscopes-Seg50和CholecSeg8k基准上进行了评估。结果：所提出的方法与最先进的基线相比，在平均交并比(mIoU)上提高了7-8%，在平均Dice分数(mDice)上提高了6%。它在解剖上一致的预测方面表现良好，特别是在薄、稀有和关键安全结构上。结论：所提出的基于图的分割方法提高了手术场景分割的性能和解剖一致性。通过结合基于ViT的全局上下文和基于图的推理，这些模型提高了可解释性和可靠性，通过精确识别关键解剖特征，为更安全的腹腔镜和机器人辅助手术铺平了道路。


### 论文摘要

Purpose: Accurate identification of hepatocystic anatomy is critical to preventing surgical complications during laparoscopic cholecystectomy. Deep learning models often struggle with occlusions, long-range dependencies, and capturing the fine-scale geometry of rare structures. This work addresses these challenges by introducing graph-based segmentation approaches that enhance spatial and semantic understanding in surgical scene analyses.   Methods: We propose two segmentation models integrating Vision Transformer (ViT) feature encoders with Graph Neural Networks (GNNs) to explicitly model spatial relationships between anatomical regions. (1) A static k Nearest Neighbours (k-NN) graph with a Graph Convolutional Network with Initial Residual and Identity Mapping (GCNII) enables stable long-range information propagation. (2) A dynamic Differentiable Graph Generator (DGG) with a Graph Attention Network (GAT) supports adaptive topology learning. Both models are evaluated on the Endoscapes-Seg50 and CholecSeg8k benchmarks.   Results: The proposed approaches achieve up to 7-8% improvement in Mean Intersection over Union (mIoU) and 6% improvement in Mean Dice (mDice) scores over state-of-the-art baselines. It produces anatomically coherent predictions, particularly on thin, rare and safety-critical structures.   Conclusion: The proposed graph-based segmentation methods enhance both performance and anatomical consistency in surgical scene segmentation. By combining ViT-based global context with graph-based relational reasoning, the models improve interpretability and reliability, paving the way for safer laparoscopic and robot-assisted surgery through a precise identification of critical anatomical features.

---

## 60. Quantifying Phase Transformations in Alloying Anodes via In-Situ Liquid Cell Hard X-ray Spectroscopy and Cryogenic Microscopy

**论文链接:** [http://arxiv.org/abs/2511.16382v1](http://arxiv.org/abs/2511.16382v1)

**作者:** Neil Mulcahy, Syeda Ramin Jannat, Yaqi Li, Tigran Simonian, Mariana Palos, James O. Douglas, Jessica M. Walker, Baptiste Gault, Mary P. Ryan, Michele Shelly Conroy

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究通过多尺度表征方法揭示了铂基合金阳极在锂化过程中的行为机制，包括Li2Pt向LiPt金属间化合物的转变、固体电解质界面的组成变化以及合金内部的纳米尺度组成梯度。

### 背景

理解复杂液-固界面的电化学现象需要将实时结构动力学与原子尺度的界面化学联系起来。

### 目的

提供对Pt基合金阳极跨尺度的机理理解，揭示合金形成和电化学稳定性的控制因素。

### 方法

整合原位同步辐射X射线荧光和衍射、高分辨率冷冻电子和离子多模态显微镜、冷冻扫描透射电子显微镜、电子能量损失谱以及冷冻原子探针层析技术等多种表征手段。

### 主要发现

1) 初始锂化驱动的Li2Pt形成及其通过固溶体反应机制演变为稳定的LiPt金属间化合物相；2) 固体电解质界面从富含碳酸盐转变为以LiF为主的稳定组成；3) 合金阳极内存在空间上不同的组成区域，包括锂通量限制区、异质界面区和扩散控制的均匀LiPt合金体相；4) 纳米尺度组成梯度解释了固溶体反应机制，表明动力学限制和界面动力学控制合金形成和电化学稳定性。

### 结论

研究成果展示了一个广泛适用的关联框架，将原位结构动力学与近原子分辨率的界面化学联系起来，为下一代储能装置中耐久合金电极的理性设计提供了基础。

### 翻译

理解复杂液-固界面的电化学现象需要将实时结构动力学与原子尺度的界面化学联系起来。本研究整合了原位同步辐射X射线荧光和衍射、高分辨率冷冻电子和离子多模态显微镜，为跨尺度的Pt基合金阳极提供了机理理解。我们直接观察到初始锂化驱动的Li2Pt形成及其在循环过程中通过固溶体反应机制演变为稳定的LiPt金属间化合物相。同时，固体电解质界面从不稳定的富含碳酸盐转变为稳定的以LiF为主的组成，这一发现通过冷冻扫描透射电子显微镜和电子能量损失谱得到确认。关键的是，冷冻原子探针层析技术揭示了合金阳极内空间上不同的组成区域，包括锂通量限制区、异质界面区和扩散控制的均匀LiPt合金体相。这种纳米尺度的组成梯度解释了新兴的固溶体反应机制，并强调了动力学限制和界面动力学如何控制合金形成和电化学稳定性。我们的发现展示了一个广泛适用的关联框架，将原位结构动力学与近原子分辨率的界面化学联系起来，推动了下一代储能装置中耐久合金电极的理性设计。


### 论文摘要

Understanding electrochemical phenomena at complex liquid solid interfaces requires linking real time structural dynamics with atomic scale interfacial chemistry. Here, we integrate operando synchrotron X-ray fluorescence and diffraction with high resolution cryogenic electron and ion multi model microscopy to provide a mechanistic understanding of Pt based alloying anodes across length scales. We directly observe the initial lithiation driven formation of Li2Pt and its evolution to a stable LiPt intermetallic phase during extended cycling via a solid solution type reaction mechanism. Simultaneously, the solid electrolyte interphase transitions from an unstable carbonate rich to a stable LiF dominated composition, confirmed by cryogenic scanning transmission electron microscopy and electron energy loss spectroscopy. Crucially, cryogenic atom probe tomography reveals spatially distinct compositional regimes within the alloy anode, including lithium flux limited, heterogeneous interfacial zone and a diffusion controlled, homogeneous LiPt alloy bulk. This nanoscale compositional gradient rationalises the emergent solid solution reaction mechanism and highlights how kinetic limitations and interface dynamics govern alloy formation and electrochemical stability. Our findings demonstrate a broadly applicable correlative framework bridging operando structural dynamics with near atomic resolution interfacial chemistry, advancing the rational design of durable alloy electrodes for next generation energy storage.

---

## 61. Building temporally coherent 3D maps with VGGT for memory-efficient Semantic SLAM

**论文链接:** [http://arxiv.org/abs/2511.16282v1](http://arxiv.org/abs/2511.16282v1)

**作者:** Gergely Dinya, Péter Halász, András Lőrincz, Kristóf Karacs, Anna Gelencsér-Horváth

**发布时间:** 2025-11-20

### GPT解析

### 总结

这篇论文提出了一种基于视觉门控生成Transformer的快速时空场景理解框架，实现了高效、接近实时的场景理解，支持辅助导航等应用。

### 背景

场景理解对于辅助导航等应用至关重要，但现有的方法可能面临计算效率低或内存需求大的问题。

### 目的

开发一个高效的时空场景理解框架，实现接近实时的性能，同时克服高内存需求的问题，并支持环境变化检测。

### 方法

基于视觉门控生成Transformer构建框架，使用滑动窗口处理图像流并对齐子图，利用VGGT跟踪头将2D语义实例掩码聚合为3D对象，并存储时间戳和实例级身份以实现时间一致性。

### 主要发现

该方法在知名基准测试和辅助导航专用数据集上表现良好，证明了其在真实场景中的适用性。

### 结论

所提出的框架能够有效实现高效的时空场景理解，适用于真实世界的辅助导航等应用场景。

### 翻译

我们提出了一种基于视觉门控生成Transformer的快速时空场景理解框架。所提出的管道旨在实现高效、接近实时的性能，支持包括辅助导航在内的应用。为了实现3D场景表示的连续更新，我们使用滑动窗口处理图像流，对齐子图，从而克服VGGT的高内存需求。我们利用VGGT跟踪头将2D语义实例掩码聚合为3D对象。为了实现时间一致性和更丰富的上下文推理，系统存储时间戳和实例级身份，从而能够检测环境变化。我们在知名基准测试和专门为辅助导航场景设计的自定义数据集上评估了该方法。结果表明该框架适用于真实场景。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何构建内存高效且时间上一致的3D语义地图的问题，特别是在处理长视频序列时的内存限制和时间一致性问题。这个问题在现实中很重要，因为辅助导航（如为视障人士提供导航）需要在杂乱、不熟悉且不断变化的室内环境中提供稳定的3D地图，系统能够适应视角变化、部分遮挡和场景变化，同时需要实时处理连续视频流而非依赖离线处理。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析VGGT模型在长序列处理中的局限性（内存需求增长、缺乏时间依赖建模）来设计方法。他们借鉴了多项现有工作：以VGGT为基础模型，参考了VGGT Long的分块处理思想，受FastVGGT和StreamVGGT的加速和流式处理启发，改进了VGGT-SLAM的子图对齐方法。作者设计了一个分块处理视频流并对齐子图的系统，结合VGGT的跟踪能力和语义分割，实现了时间一致的3D重建和变化检测。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过分块处理视频流并对齐子图实现内存高效的3D重建，利用VGGT跟踪能力实现时间一致的语义映射，并通过带时间戳的物体轨迹实现变化检测。整体流程包括：1)全局对齐：将视频流分块，选择关键帧，预测深度和位姿，计算比例因子和相似变换，平滑轨迹；2)3D分割：使用YOLO进行2D分割，用VGGT跟踪掩码，通过投票聚合成3D物体，实现重识别；3)变化检测：分配时间戳，维护物体状态，基于可见性更新置信度；4)自定位和物体位置：计算用户与物体间的距离，使用点云中值点估计物体位置。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)内存高效的分块VGGT管道，通过子图对齐克服内存限制；2)时间一致的语义映射策略，融合2D分割与VGGT跟踪；3)基于轨迹和可见性的轻量级变化检测机制；4)支持实时操作的在线SLAM框架。相比之前工作：与VGGT Long不同，支持流式输入并能处理动态物体；与StreamVGGT不同，内存使用不随序列长度增长；与VGGT-SLAM不同，更适合在线场景并提供更一致的时间语义信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于VGGT的内存高效、时间一致的语义SLAM框架，通过分块处理和子图对齐实现了长序列视频流的实时3D重建和语义理解，特别适用于需要实时处理和变化检测的辅助导航应用场景。'}


### 论文摘要

We present a fast, spatio-temporal scene understanding framework based on Vision Gated Generative Transformers (VGGT). The proposed pipeline is designed to enable efficient, close to real-time performance, supporting applications including assistive navigation. To achieve continuous updates of the 3D scene representation, we process the image flow with a sliding window, aligning submaps, thereby overcoming VGGT's high memory demands. We exploit the VGGT tracking head to aggregate 2D semantic instance masks into 3D objects. To allow for temporal consistency and richer contextual reasoning the system stores timestamps and instance-level identities, thereby enabling the detection of changes in the environment. We evaluate the approach on well-known benchmarks and custom datasets specifically designed for assistive navigation scenarios. The results demonstrate the applicability of the framework to real-world scenarios.

---

## 62. Weakly Supervised Segmentation and Classification of Alpha-Synuclein Aggregates in Brightfield Midbrain Images

**论文链接:** [http://arxiv.org/abs/2511.16268v1](http://arxiv.org/abs/2511.16268v1)

**作者:** Erwan Dereure, Robin Louiset, Laura Parkkinen, David A Menassa, David Holcman

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究开发了一个基于深度学习的自动化图像处理流程，用于分割和分类帕金森病和偶发性路易体病病例中脑组织全切片图像中的α-突触核蛋白聚集物。

### 背景

帕金森病是一种神经退行性疾病，与错误折叠的α-突触核蛋白聚集物的积累有关，这些聚集物形成路易体和神经突起，用于病理诊断。利用深度学习对免疫组织化学组织病理学图像进行自动分析，为更好地理解这些聚集物的空间组织提供了有希望的工具。

### 目的

开发一个自动化图像处理流程，用于分割和分类帕金森病和偶发性路易体病病例中脑组织全切片图像中的聚集物，并能区分主要的聚集物形态，包括路易体和神经突起。

### 方法

研究基于弱监督分割方法开发了一个自动化图像处理流程，对免疫组织化学标记的变异性具有鲁棒性，并使用了ResNet50分类器。该方法能够处理全切片图像。

### 主要发现

该方法能够区分主要的聚集物形态，包括路易体和神经突起，平衡准确率达到80%。该框架为大尺度表征α-突触核蛋白聚集物在明场免疫组织化学组织中的空间分布和异质性铺平了道路，并可用于研究与周围细胞(如小胶质细胞和星形胶质细胞)的关系。

### 结论

开发的自动化图像处理流程为帕金森病和相关疾病中α-突触核蛋白聚集物的分析提供了有效工具，有助于更好地理解这些聚集物的空间组织和与周围细胞的关系。

### 翻译

帕金森病是一种神经退行性疾病，与错误折叠的α-突触核蛋白聚集物的积累有关，这些聚集物形成路易体和神经突起，用于病理诊断。利用深度学习对免疫组织化学组织病理学图像进行自动分析，为更好地理解这些聚集物的空间组织提供了有希望的工具。在本研究中，我们开发了一个自动化图像处理流程，基于弱监督分割方法，对免疫组织化学标记的变异性具有鲁棒性，使用ResNet50分类器，用于分割和分类帕金森病和偶发性路易体病病例中脑组织全切片图像中的这些聚集物。我们的方法能够区分主要的聚集物形态，包括路易体和神经突起，平衡准确率达到80%。这一框架为大尺度表征α-突触核蛋白聚集物在明场免疫组织化学组织中的空间分布和异质性铺平了道路，并可用于研究与它们周围细胞(如小胶质细胞和星形胶质细胞)的关系。


### 论文摘要

Parkinson's disease (PD) is a neurodegenerative disorder associated with the accumulation of misfolded alpha-synuclein aggregates, forming Lewy bodies and neuritic shape used for pathology diagnostics. Automatic analysis of immunohistochemistry histopathological images with Deep Learning provides a promising tool for better understanding the spatial organization of these aggregates. In this study, we develop an automated image processing pipeline to segment and classify these aggregates in whole-slide images (WSIs) of midbrain tissue from PD and incidental Lewy Body Disease (iLBD) cases based on weakly supervised segmentation, robust to immunohistochemical labelling variability, with a ResNet50 classifier. Our approach allows to differentiate between major aggregate morphologies, including Lewy bodies and neurites with a balanced accuracy of $80\%$. This framework paves the way for large-scale characterization of the spatial distribution and heterogeneity of alpha-synuclein aggregates in brightfield immunohistochemical tissue, and for investigating their poorly understood relationships with surrounding cells such as microglia and astrocytes.

---

## 63. GazeInterpreter: Parsing Eye Gaze to Generate Eye-Body-Coordinated Narrations

**论文链接:** [http://arxiv.org/abs/2511.16245v1](http://arxiv.org/abs/2511.16245v1)

**作者:** Qing Chang, Zhiming Hu

**发布时间:** 2025-11-20

**备注:** Accepted to AAAI 2026. 9 pages, 4 figures

### GPT解析

### 总结

论文提出了GazeInterpreter，一种基于大型语言模型的新方法，通过解析眼神注视数据并生成眼神-身体协调的叙述来全面解释人类行为，克服了之前工作中忽视眼神注视及其与身体运动协同作用的局限性。

### 背景

全面解释人类行为是人类感知人工智能的核心挑战。然而，先前的工作通常只关注身体行为，忽视了眼神注视及其与身体运动协同的关键作用。

### 目的

提出一种新的基于大型语言模型的方法，用于解析眼神注视数据并生成眼神-身体协调的叙述，以全面解释人类行为。

### 方法

GazeInterpreter包含三个主要特点：1) 符号化注视解析器，将原始注视信号转换为符号化注视事件；2) 分层结构，首先使用大型语言模型在语义层面生成注视叙述，然后在同一观察窗口内将注视与身体运动整合以产生综合叙述；3) 自纠正循环，迭代改进综合叙述的模态匹配、时间连贯性和完整性。

### 主要发现

在大规模Nymeria基准测试中的文本驱动运动生成任务上验证了眼神-身体协调叙述的有效性；在动作预测和行为总结的下游任务中报告了显著的性能改进。

### 结论

解析眼神注视对于解释人类行为具有巨大潜力，为人类行为理解开辟了新方向。

### 翻译

全面解释人类行为是人类感知人工智能的核心挑战。然而，先前的工作通常只关注身体行为，忽视了眼神注视及其与身体运动协同的关键作用。我们提出了GazeInterpreter——一种新颖的基于大型语言模型的方法，用于解析眼神注视数据并生成眼神-身体协调的叙述。具体来说，我们的方法具有三个特点：1) 一个符号化注视解析器，将原始注视信号转换为符号化注视事件；2) 一个分层结构，首先使用大型语言模型在语义层面生成注视叙述，然后在同一观察窗口内将注视与身体运动整合以产生综合叙述；3) 一个自纠正循环，迭代改进综合叙述的模态匹配、时间连贯性和完整性。这种分层和迭代处理可以有效对齐时空域中的物理值和语义文本。我们在大规模Nymeria基准测试中的文本驱动运动生成任务上验证了我们眼神-身体协调叙述的有效性。此外，我们报告了在动作预测和行为总结的下游任务中有显著的性能提升。总之，这些结果揭示了解析眼神注视以解释人类行为的巨大潜力，为人类行为理解开辟了新方向。


### 论文摘要

Comprehensively interpreting human behavior is a core challenge in human-aware artificial intelligence. However, prior works typically focused on body behavior, neglecting the crucial role of eye gaze and its synergy with body motion. We present GazeInterpreter - a novel large language model-based (LLM-based) approach that parses eye gaze data to generate eye-body-coordinated narrations. Specifically, our method features 1) a symbolic gaze parser that translates raw gaze signals into symbolic gaze events; 2) a hierarchical structure that first uses an LLM to generate eye gaze narration at semantic level and then integrates gaze with body motion within the same observation window to produce integrated narration; and 3) a self-correcting loop that iteratively refines the modality match, temporal coherence, and completeness of the integrated narration. This hierarchical and iterative processing can effectively align physical values and semantic text in the temporal and spatial domains. We validated the effectiveness of our eye-body-coordinated narrations on the text-driven motion generation task in the large-scale Nymeria benchmark. Moreover, we report significant performance improvements for the sample downstream tasks of action anticipation and behavior summarization. Taken together, these results reveal the significant potential of parsing eye gaze to interpret human behavior and open up a new direction for human behavior understanding.

---

## 64. Video2Layout: Recall and Reconstruct Metric-Grounded Cognitive Map for Spatial Reasoning

**论文链接:** [http://arxiv.org/abs/2511.16160v1](http://arxiv.org/abs/2511.16160v1)

**作者:** Yibin Huang, Wang Xu, Wanyue Zhang, Helu Zhi, Jingjing Huang, Yangbin Xu, Yangang Sun, Conghui Zhu, Tiejun Zhao

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了Video2Layout框架，通过使用连续物体边界坐标而非离散网格表示，提升了多模态大语言模型的空间推理能力，在QVS-Bench和主流空间推理基准上实现了4.92%的平均性能提升。

### 背景

空间智能是多模态大语言模型理解物理世界的关键能力。现有研究通过基于网格的认知地图构建空间理解，但这些方法依赖离散光栅表示，限制了模型的细粒度空间推理能力。

### 目的

克服基于网格方法的局限性，开发一种能够从视频中重建基于度量的空间布局的框架，提升模型的空间推理能力。

### 方法

Video2Layout框架使用连续物体边界坐标量化物体间物理距离和大小，赋予模型定量空间计算能力。方法包含两个阶段：1)监督微调阶段，利用AI2THOR模拟器构建数据集学习视觉输入到边界坐标的映射；2)强化微调阶段增强现实世界泛化能力。同时引入QVS-Bench基准评估图像数量与空间推理准确性的关系。

### 主要发现

在QVS-Bench和主流空间推理基准上评估的V2LO-7B模型，比在网格地图上训练的模型平均提高4.92%，验证了所提出方法的优越性。

### 结论

Video2Layout框架通过采用连续物体边界坐标而非离散网格表示，有效提升了多模态大语言模型的空间推理能力和自然语言空间描述的准确性。

### 翻译

空间智能是多模态大语言模型的关键前沿，使它们能够理解物理世界。受人类感知机制启发，现有研究试图通过基于网格的认知地图从多帧视觉输入构建连贯的空间理解。然而，当前基于网格的地图方法依赖于离散的光栅表示，限制了模型在细粒度空间推理方面的能力。为克服这一局限，我们提出了Video2Layout，一个用于从视频中重建基于度量的空间布局的框架。该框架采用连续的物体边界坐标来量化物体间的物理距离和物体大小。这使模型具有定量空间计算能力，有效缓解了在自然语言中描述空间关系时的固有模糊性。具体而言，我们的方法包含两个核心阶段。首先，在监督微调阶段，我们从AI2THOR模拟器构建了一个高质量数据集，使模型能够学习从视觉输入到精确边界坐标的映射。随后，强化微调阶段进一步增强了模型的现实世界泛化能力。为了系统评估认知地图准确性与图像数量之间的相关性，以及图像输入数量如何影响空间推理准确性，我们引入了QVS-Bench，一个旨在分析相关机制的诊断基准。在QVS-Bench和主流空间推理基准上评估，我们的V2LO-7B模型比在网格地图上训练的模型平均提高了4.92%，验证了我们方法的优越性。我们的代码可在https://github.com/ybrrraway/Video2Layout获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态大语言模型在空间智能方面的局限性，特别是基于网格的认知地图方法依赖离散表示，限制了模型在细粒度空间推理中的能力。这个问题很重要，因为空间智能是模型理解物理世界的关键，是具身智能的核心环节，而当前模型在空间感知和推理方面存在显著不足，难以有效聚合多帧视觉信息形成连贯的空间表示。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者受人类感知机制和认知神经科学的启发，认识到现有网格地图方法的局限性后，设计出Video2Layout框架。他们借鉴了认知地图的基本思想，但改进为使用连续坐标；借鉴了结构化思维链方法，引入几何约束；同时采用监督微调和强化微调的两阶段训练范式，但针对空间推理任务进行了专门定制。整体上是在现有方法基础上进行创新和改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用连续的物体边界坐标构建度量认知地图，为模型提供定量空间计算能力，并通过结构化思维链将空间推理转化为数学计算问题。整体流程分为三个阶段：1)数据准备，构建包含模拟和真实场景的V2LO-28K数据集；2)监督微调，在模拟数据上训练模型生成度量地图和结构化推理；3)强化微调，使用GRPO算法在真实数据上优化，提高模型泛化能力。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出Video2Layout框架从视频中重建度量空间布局；2)使用连续坐标而非离散网格提供精确空间信息；3)构建QVS-Bench基准评估视觉输入数量与推理性能关系；4)采用SFT到RL的两阶段训练范式。相比之前工作，本文在空间表示上从离散网格变为连续坐标，在数据上从依赖真实数据变为先模拟后真实两阶段，在推理上从纯文本描述变为数学计算，在应用上从单帧扩展为连续视频处理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Video2Layout通过引入基于连续坐标的度量认知地图和两阶段训练范式，显著提升了多模态大语言模型在细粒度空间推理任务中的性能，为具身智能的发展提供了新的技术路径。'}


### 论文摘要

Spatial intelligence is a critical frontier for Multimodal Large Language Models (MLLMs), empowering them to comprehend the physical world. Drawing inspiration from human perception mechanisms, existing studies attempt to construct a coherent spatial understanding via grid-based cognitive maps from multi-frame visual inputs. However, current grid-based map methods rely on discretized raster representations, which limit the model's ability in fine-grained spatial reasoning. To overcome this limitation, we propose Video2Layout, a framework for reconstructing metric-grounded spatial layouts from video. The framework employs continuous object boundary coordinates to quantify inter-object physical distances and object size. This empowers the model with quantitative spatial computation capabilities, effectively alleviating the inherent ambiguity when describing spatial relationships in natural language. Specifically, our method comprises two core stages. First, in supervised fine-tuning stage, we construct a high-quality dataset from the AI2THOR simulator, which enables the model to learn the mapping from visual inputs to precise boundary coordinates. Subsequently, a reinforcement fine-tuning stage further enhances the model's real-world generalization capabilities. To systematically evaluate the correlation between cognitive map accuracy and image quantity, as well as how the quantity of image inputs affects spatial reasoning accuracy, we introduce QVS-Bench, a diagnostic benchmark designed to analyze the relevant mechanisms. Evaluated on QVS-Bench and mainstream spatial reasoning benchmarks, our model, V2LO-7B achieves an average improvement of 4.92% over the model trained on grid maps, validating the superiority of our method. Our code is available at https://github.com/ybrrraway/Video2Layout.

---

## 65. LEGO-SLAM: Language-Embedded Gaussian Optimization SLAM

**论文链接:** [http://arxiv.org/abs/2511.16144v1](http://arxiv.org/abs/2511.16144v1)

**作者:** Sibaek Lee, Seongbo Ha, Kyeongsu Kang, Joonyeol Choi, Seungjun Tak, Hyeonwoo Yu

**发布时间:** 2025-11-20

**备注:** 18 pages

### GPT解析

### 总结

LEGO-SLAM是一种创新的3DGS-based SLAM框架，首次实现了实时开放词汇映射，通过场景自适应编码器-解码器将高维语言特征压缩为16维紧凑表示，解决了内存和渲染开销问题。

### 背景

3D高斯飞溅技术使SLAM系统能构建逼真地图，但这些地图缺乏高级机器人交互所需的开放词汇语义理解。将语言特征整合到SLAM面临高维特征存储导致内存和渲染开销过大，以及现有静态模型对新环境适应性差等挑战。

### 目的

开发一种能在3DGS-based SLAM系统中实现实时开放词汇映射的框架，克服现有方法的内存和性能限制。

### 方法

提出LEGO-SLAM，核心是场景自适应编码器-解码器，将高维语言嵌入压缩为16维特征空间；实现语言引导的剪枝策略减少60%以上高斯数量；引入基于语言的循环检测方法重用映射特征，无需单独检测模型。

### 主要发现

LEGO-SLAM在保持渲染质量的同时显著减少内存需求；系统具有开放词汇能力，运行速度达15 FPS；实现了与现有方法相媲美的映射质量和跟踪精度；编码器能在线适应未见场景，增强系统适应性。

### 结论

LEGO-SLAM成功解决了3DGS地图缺乏语义理解的问题，通过紧凑特征表示和自适应编码实现了实时性能，语言引导的剪枝和循环检测进一步优化了系统，为机器人交互提供了更强大的语义理解能力。

### 翻译

最近的3D高斯飞溅(3DGS)进展使SLAM系统能够构建照片级真实的地图。然而，这些地图缺乏高级机器人交互所需的开放词汇语义理解。将语言特征整合到SLAM仍然是一个重大挑战，因为存储高维特征需要过多的内存和渲染开销，而现有静态模型方法对新环境缺乏适应性。为解决这些局限，我们提出了LEGO-SLAM(语言嵌入高斯优化SLAM)，这是首个在基于3DGS的SLAM系统中实现实时开放词汇映射的框架。我们方法的核心是一个场景自适应编码器-解码器，将高维语言嵌入压缩为紧凑的16维特征空间。这种设计减少了每个高斯的内存并加速渲染，实现实时性能。与静态方法不同，我们的编码器能在线适应未见场景。这些紧凑特征还实现了语言引导的剪枝策略，识别语义冗余，在保持渲染质量的同时将地图高斯数量减少60%以上。此外，我们引入了基于语言的循环检测方法，重用这些映射特征，无需单独的检测模型。大量实验证明，LEGO-SLAM实现了具有竞争力的映射质量和跟踪精度，同时提供15 FPS的开放词汇能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文解决的是现有3D高斯溅射(3DGS)SLAM系统缺乏开放词汇语义理解能力的问题。这个问题很重要，因为随着机器人技术的发展，仅仅构建几何地图已不够，机器人需要理解环境的语义信息才能执行高级任务，如物体操作、场景理解和人机交互。现有方法要么因存储高维语言特征导致内存和渲染开销过大，要么使用静态模型缺乏对新环境的适应性。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有3DGS-SLAM的局限性：缺乏语义理解、高维特征存储开销大、静态模型适应性差。然后设计了一个场景自适应编码器-解码器，将高维语言特征压缩到16维空间。方法借鉴了3DGS框架、视觉-语言模型(如CLIP)、G-ICP位估计算法以及SLAM系统的跟踪-映射架构。关键创新在于解耦优化策略和预训练先验的使用，确保实时性能。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将开放词汇语义理解整合到实时3D高斯溅射SLAM中，通过紧凑特征表示平衡语义表达和计算效率。流程包括：1)跟踪模块用G-ICP估计相机位姿并选择关键帧；2)映射模块初始化新高斯，优化地图时联合最小化RGB、深度和特征损失；3)语言引导剪枝基于距离和特征相似性移除冗余高斯；4)循环检测重用语言特征进行位置识别和几何验证。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新：1)首个实时开放词汇3DGS-SLAM系统，15FPS下提供语义能力；2)场景自适应编码器压缩语言特征到16维，支持在线环境适应；3)语言引导剪枝策略减少60%高斯而不损失质量；4)重用语言特征的循环检测，无需额外模型。相比之前工作，它解决了高维特征存储的开销问题，摆脱了静态模型的限制限制，并首次将开放词汇语义理解整合到实时SLAM系统中。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LEGO-SLAM通过紧凑的语言特征表示和场景自适应编码器，首次实现了实时开放的词汇3D高斯溅射SLAM系统，在保持高精度地图构建和跟踪的同时，提供了强大的语义理解和物体定位能力。'}


### 论文摘要

Recent advances in 3D Gaussian Splatting (3DGS) have enabled Simultaneous Localization and Mapping (SLAM) systems to build photorealistic maps. However, these maps lack the open-vocabulary semantic understanding required for advanced robotic interaction. Integrating language features into SLAM remains a significant challenge, as storing high-dimensional features demands excessive memory and rendering overhead, while existing methods with static models lack adaptability for novel environments. To address these limitations, we propose LEGO-SLAM (Language-Embedded Gaussian Optimization SLAM), the first framework to achieve real-time, open-vocabulary mapping within a 3DGS-based SLAM system. At the core of our method is a scene-adaptive encoder-decoder that distills high-dimensional language embeddings into a compact 16-dimensional feature space. This design reduces the memory per Gaussian and accelerates rendering, enabling real-time performance. Unlike static approaches, our encoder adapts online to unseen scenes. These compact features also enable a language-guided pruning strategy that identifies semantic redundancy, reducing the map's Gaussian count by over 60\% while maintaining rendering quality. Furthermore, we introduce a language-based loop detection approach that reuses these mapping features, eliminating the need for a separate detection model. Extensive experiments demonstrate that LEGO-SLAM achieves competitive mapping quality and tracking accuracy, all while providing open-vocabulary capabilities at 15 FPS.

---

## 66. Real-Time 3D Object Detection with Inference-Aligned Learning

**论文链接:** [http://arxiv.org/abs/2511.16140v1](http://arxiv.org/abs/2511.16140v1)

**作者:** Chenyu Zhao, Xianwei Zheng, Zimin Xia, Linwei Yue, Nan Xue

**发布时间:** 2025-11-20

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

论文提出了SR3D框架，用于室内点云的实时3D目标检测，通过空间优先的最优传输分配和排序感知的自适应自蒸馏方案，解决了训练与推理之间的差距问题。

### 背景

实时3D目标检测对于增强现实、机器人和导航等应用中的动态场景理解至关重要。现有的检测方法在训练和评估之间存在差距，影响了模型学习与推理行为一致的表示能力。

### 目的

弥合3D目标检测器训练与评估之间的差距，提高模型在推理时的表现，同时保持实时检测速度。

### 方法

提出SR3D框架，包含两个针对点云空间特性的组件：1) 空间优先的最优传输分配，动态强调定位良好且空间可靠的样本；2) 排序感知的自适应自蒸馏方案，通过自蒸馏范式自适应地注入排序感知能力。

### 主要发现

在ScanNet V2和SUN RGB-D数据集上的广泛实验表明，SR3D有效地弥合了训练-推理差距，在保持实时速度的同时显著优于先前方法的准确性。

### 结论

SR3D框架通过解决训练与推理之间的不一致性问题，提高了3D目标检测的性能，适用于需要实时动态场景理解的应用领域。

### 翻译

从点云进行实时3D目标检测对于增强现实、机器人和导航等应用中的动态场景理解至关重要。我们引入了一种用于室内点云的新型空间优先和排序感知3D目标检测（SR3D）框架，以弥合检测器训练方式与评估方式之间的差距。这种差距源于训练过程中缺乏空间可靠性和排序意识，与推理中使用的基于排序的预测选择相冲突。这种训练-推理差距阻碍了模型学习与推理时行为一致的表示能力。为解决这一局限性，SR3D包含两个针对点云空间特性定制的组件：一种新型空间优先的最优传输分配，动态强调定位良好且空间可靠的样本；以及一种排序感知的自适应自蒸馏方案，通过自蒸馏范式自适应地注入排序感知能力。在ScanNet V2和SUN RGB-D上的广泛实验表明，SR3D有效地弥合了训练-推理差距，并在保持实时速度的同时显著优于先前方法的准确性。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D目标检测中的训练-推理不一致性问题。当前检测器在训练时使用固定的标签分配策略，不考虑预测的空间可靠性和排序，而推理时却基于排序选择预测，这种不一致阻碍了模型学习与推理行为一致的表示。这个问题在增强现实、机器人和导航等需要实时处理点云数据的应用中至关重要，解决它可以提高检测精度同时保持实时性能，使3D目标检测在实际应用中更加可靠有效。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了3D检测器的两个主要问题：缺乏空间可靠性和排序意识。他们通过案例研究发现使用真实IoU分数代替分类分数能显著提高性能，表明排序意识缺失是主要瓶颈。设计上，作者借鉴了最优传输分配(OTA)的思想，但针对3D检测特点进行了改进；同时借鉴了自知识蒸馏方法，设计了排序感知的自蒸馏方案。整体上，作者采用密集检测框架而非稀疏框架，因为更适合实时应用，并通过引入归一化顶点距离和空间优先策略解决了OTA在3D场景中的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：1)空间优先：在3D检测中几何信息比语义信息更重要，应优先考虑形状、边缘等几何线索；2)排序感知：训练过程应考虑预测排序，因为评估指标如AP本质是排序敏感的。整体流程：输入点云→稀疏卷积骨干网络+特征金字塔提取特征→两个任务特定头生成密集预测→训练时使用SPOTA进行空间优先标签分配，同时用RAS进行排序感知自蒸馏推理时→应用NMS去除冗余预测→输出检测结果。SPOTA通过归一化顶点距离和中心先验选择高质量样本，RAS则通过自适应权重平衡分类损失和蒸馏损失。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点有两个：1)空间优先最优传输分配(SPOTA)：引入归一化顶点距离作为更精确的几何测量，完全移除成本矩阵中的分类损失，仅依赖几何线索；2)排序感知自适应自蒸馏(RAS)：设计排序感知的自蒸馏损失和自适应权重机制，动态调整监督贡献。相比之前工作，SPOTA解决了OTA在3D场景中的多目标冲突问题，RAS则比质量感知损失方法(如QFL、VFL)在低IoU条件下更稳定；SR3D整体框架在保持实时性能的同时提高了精度，且不增加计算开销，比DLLA等方法更高效。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SR3D通过引入空间优先最优传输分配和排序感知自适应自蒸馏两个创新组件，有效解决了3D目标检测中的训练-推理不一致性问题，在保持实时性能的同时显著提高了检测精度。'}


### 论文摘要

Real-time 3D object detection from point clouds is essential for dynamic scene understanding in applications such as augmented reality, robotics and navigation. We introduce a novel Spatial-prioritized and Rank-aware 3D object detection (SR3D) framework for indoor point clouds, to bridge the gap between how detectors are trained and how they are evaluated. This gap stems from the lack of spatial reliability and ranking awareness during training, which conflicts with the ranking-based prediction selection used as inference. Such a training-inference gap hampers the model's ability to learn representations aligned with inference-time behavior. To address the limitation, SR3D consists of two components tailored to the spatial nature of point clouds during training: a novel spatial-prioritized optimal transport assignment that dynamically emphasizes well-located and spatially reliable samples, and a rank-aware adaptive self-distillation scheme that adaptively injects ranking perception via a self-distillation paradigm. Extensive experiments on ScanNet V2 and SUN RGB-D show that SR3D effectively bridges the training-inference gap and significantly outperforms prior methods in accuracy while maintaining real-time speed.

---

## 67. 论文ID: 2511.16139v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.16139v1.json'

---

## 68. Local and global bifurcations to large-scale oblique patterns in inclined layer convection

**论文链接:** [http://arxiv.org/abs/2511.15979v1](http://arxiv.org/abs/2511.15979v1)

**作者:** Zheng Zheng, Sajjad Azimi, Florian Reetz, Tobias M. Schneider

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究探讨了倾斜层对流系统中切换菱形面板模式的形成机制和动力学特性，通过分析非线性不变解及其分叉行为，揭示了该模式远离对流起始点的出现机制。

### 背景

研究的是倾斜层对流系统，这是一个相对于重力倾斜放置的Rayleigh-Bénard单元，其中流体受到浮力和剪切力的竞争影响。对于不同的倾斜角度和Rayleigh数，系统会表现出多种时空模式。

### 目的

研究在特定参数(γ≈100°, Ra≈10000)下观察到的切换菱形面板模式，这是Prandtl数Pr=1.07时的五种复杂三级模式之一，具有大尺度斜向特征。

### 方法

分析二级状态横向对流卷的线性不稳定性及其五个分叉分支；研究一个捕获调制 rolls 大尺度结构和小尺度缺陷的周期轨道；通过参数连续性揭示全局同宿分叉；特征化两个动力学相关周期轨道的吸引域边界并分析边界上的额外解；使用非线性不变解及其分叉来理解SDP模式。

### 主要发现

SDP模式表现出大尺度菱形幅度调制；发现了一个能够完整描述调制 rolls 特征的周期轨道；识别了周期轨道吸引域边界上的额外稳态和周期性解；揭示了SDP模式远离对流起始点的出现机制。

### 结论

通过非线性不变解及其分叉分析，成功理解了远离对流起始点的SDP模式的出现和动力学特性，而线性方法在这一区域未能成功应用。

### 翻译

在倾斜层对流系统中，这是一个相对于重力倾斜放置的Rayleigh-Bénard单元中的热对流，流动受到浮力和剪切力的竞争影响。对于变化的倾斜角度(γ)和Rayleigh数(Ra)，观察到多种时空模式。我们研究在(γ, Ra)≈(100°, 10000)时观察到的切换菱形面板模式，该模式具有大尺度斜向特征，是Prandtl数Pr=1.07时的五种复杂三级模式之一。首先，我们研究二级状态横向对流卷的线性不稳定性以及五个分支（包括两个行进波和三个周期轨道）从其同时分叉的情况。这些非通用分叉源于横向 rolls 的空间D4和O(2)对称性的破坏，由此得到的分叉解表现出大尺度菱形幅度调制。其次，我们探索了一个捕获调制 rolls 大尺度结构和小尺度缺陷的周期轨道。在Ra参数上的连续性揭示了这个周期轨道出现的全局同宿分叉。第三，我们特征化了两个动力学相关周期轨道的吸引域边界。具体来说，在吸引域边界上识别出了额外的稳态和周期性解，并分析了它们的分叉结构。总之，使用非线性不变解及其分叉，我们向理解远离对流起始点的SDP的出现和动力学迈出了进一步的一步，在那里线性方法未能成功应用。


### 论文摘要

In the inclined layer convection system, thermal convection in a Rayleigh--Bénard cell tilted against gravity, the flow is subject to competing buoyancy and shear forces. For varying inclination angle ($γ$) and Rayleigh number ($Ra$), a variety of spatio-temporal patterns is observed. We investigate the switching diamond panes (SDP) pattern, observed at $(γ, Ra)\simeq(100\degree,10000)$, which exhibits large-scale oblique features and is one of the five complex tertiary patterns at Prandtl number $Pr=1.07$. First, we study the linear instability of the secondary-state transverse convection rolls and the five branches including two travelling waves and three periodic orbits, bifurcating simultaneously from it. These non-generic bifurcations arise from the breaking of the spatial $D_4$ and $O(2)$ symmetries of transverse rolls, and the resulting bifurcated solutions show large-scale diamond-shaped amplitude modulations. Second, we explore a periodic orbit that captures both the large-scale structure and small-scale defects of modulated rolls. Parametric continuation in $Ra$ reveals the global homoclinic bifurcation via which this periodic orbit emerges. Third, the boundary of the basins of attraction of two dynamically relevant periodic orbits has been characterised. Specifically, additional steady and time-periodic solutions are identified on the basin boundary and their bifurcation structures are analysed. Together, using non-linear invariant solutions and their bifurcations, we take a further step toward understanding the emergence and dynamics of SDP far from the onset of convection, where linear methods have not been applied successfully.

---

## 69. The Role of Consequential and Functional Sound in Human-Robot Interaction: Toward Audio Augmented Reality Interfaces

**论文链接:** [http://arxiv.org/abs/2511.15956v1](http://arxiv.org/abs/2511.15956v1)

**作者:** Aliyah Smith, Monroe Kennedy

**发布时间:** 2025-11-20

**备注:** 9 pages, 6 figures

### GPT解析

### 总结

本研究探讨了声音作为机器人与人类交互的重要渠道，分析了结果性声音和功能性声音对人类感知和行为的影响，特别是在空间声音方面的创新应用，发现声音设计能够有效提升人机协作体验。

### 背景

随着机器人越来越多地融入日常环境，理解机器人如何与人类交流变得至关重要。声音作为一种强大的交互渠道，既包含操作噪音，也包含有意设计的听觉提示。

### 目的

研究旨在考察结果性声音和功能性声音对人类感知和行为的影响，并通过定位和交接任务对空间声音进行创新性探索，以优化人机交互的声音设计。

### 方法

通过实验研究Kinova Gen3机械臂的声音效果，包括评估声音对感知的影响、空间声音的定位准确性测试，以及声音在交接任务中的作用。

### 主要发现

Kinova Gen3机械臂的结果性声音没有负面地影响人类感知；空间定位在横向提示上高度准确但在正面提示上准确性下降；空间声音能够同时传达任务相关信息，同时促进温暖感并减少不适感。

### 结论

功能和变革性的听觉设计具有增强人机协作的潜力，可以为未来的基于声音的交互策略提供指导。

### 翻译

随着机器人越来越多地融入日常环境，理解它们如何与人类交流变得至关重要。声音为交互提供了强大的渠道，既包含操作噪音，也包含有意设计的听觉提示。在本研究中，我们考察了结果性声音和功能性声音对人类感知和行为的影响，包括通过定位和交接任务对空间声音的创新探索。结果表明，Kinova Gen3机械臂的结果性声音没有负面地影响感知，空间定位在横向提示上高度准确但在正面提示上准确性下降，空间声音可以同时传达任务相关信息，同时促进温暖感并减少不适感。这些发现突显了功能和变革性听觉设计在增强人机协作方面的潜力，并为未来的基于声音的交互策略提供了信息。


### 论文摘要

As robots become increasingly integrated into everyday environments, understanding how they communicate with humans is critical. Sound offers a powerful channel for interaction, encompassing both operational noises and intentionally designed auditory cues. In this study, we examined the effects of consequential and functional sounds on human perception and behavior, including a novel exploration of spatial sound through localization and handover tasks. Results show that consequential sounds of the Kinova Gen3 manipulator did not negatively affect perceptions, spatial localization is highly accurate for lateral cues but declines for frontal cues, and spatial sounds can simultaneously convey task-relevant information while promoting warmth and reducing discomfort. These findings highlight the potential of functional and transformative auditory design to enhance human-robot collaboration and inform future sound-based interaction strategies.

---

## 70. Click2Graph: Interactive Panoptic Video Scene Graphs from a Single Click

**论文链接:** [http://arxiv.org/abs/2511.15948v1](http://arxiv.org/abs/2511.15948v1)

**作者:** Raphael Ruschel, Hardikkumar Prajapati, Awsafur Rahman, B. S. Manjunath

**发布时间:** 2025-11-20

### GPT解析

### 总结

Click2Graph是首个交互式全景视频场景图生成框架，结合了人类提示和视觉理解能力，通过动态交互发现和语义分类，实现了从简单用户输入到复杂场景图的转换。

### 背景

最先进的视频场景图生成系统提供结构化视觉理解但无法融入人类指导，而可提示分割模型能实现精确用户交互却缺乏语义或关系推理能力。

### 目的

引入Click2Graph，一个将视觉提示与空间、时间和语义理解相结合的交互式全景视频场景图生成框架。

### 方法

从单个用户提示(如点击或边界框)开始，Click2Graph分割并跟踪主体随时间变化，自主发现相互作用的物体，预测三元组形成时间一致场景图。框架包含动态交互发现模块和语义分类头两个关键组件。

### 主要发现

在OpenPVSG基准测试上的实验表明，Click2Graph为用户引导的PVSG奠定了坚实基础，展示了如何将人类提示与全景定位和关系推理相结合，实现可控和可解释的视频场景理解。

### 结论

Click2Graph成功地将人类交互与视频场景图生成相结合，解决了现有系统的局限性，为视频场景理解提供了新的交互式方法，增强了系统的可控性和可解释性。

### 翻译

最先进的视频场景图生成系统提供结构化的视觉理解，但作为封闭的前馈管道运行，无法融入人类指导。相比之下，可提示的分割模型(如SAM2)能够实现精确的用户交互，但缺乏语义或关系推理能力。我们引入Click2Graph，这是第一个用于全景视频场景图生成的交互式框架，将视觉提示与空间、时间和语义理解相结合。从单个用户提示(如点击或边界框)开始，Click2Graph能够分割并跟踪主体随时间变化，自主发现相互作用的物体，并预测三元组以形成时间一致的场景图。我们的框架引入两个关键组件：动态交互发现模块(生成基于主体的对象提示)和语义分类头(执行联合实体和谓词推理)。在OpenPVSG基准测试上的实验表明，Click2Graph为用户引导的PVSG奠定了坚实基础，展示了如何将人类提示与全景定位和关系推理相结合，以实现可控和可解释的视频场景理解。


### 论文摘要

State-of-the-art Video Scene Graph Generation (VSGG) systems provide structured visual understanding but operate as closed, feed-forward pipelines with no ability to incorporate human guidance. In contrast, promptable segmentation models such as SAM2 enable precise user interaction but lack semantic or relational reasoning. We introduce Click2Graph, the first interactive framework for Panoptic Video Scene Graph Generation (PVSG) that unifies visual prompting with spatial, temporal, and semantic understanding. From a single user cue, such as a click or bounding box, Click2Graph segments and tracks the subject across time, autonomously discovers interacting objects, and predicts <subject, object, predicate> triplets to form a temporally consistent scene graph. Our framework introduces two key components: a Dynamic Interaction Discovery Module that generates subject-conditioned object prompts, and a Semantic Classification Head that performs joint entity and predicate reasoning. Experiments on the OpenPVSG benchmark demonstrate that Click2Graph establishes a strong foundation for user-guided PVSG, showing how human prompting can be combined with panoptic grounding and relational inference to enable controllable and interpretable video scene understanding.

---

## 71. WALDO: Where Unseen Model-based 6D Pose Estimation Meets Occlusion

**论文链接:** [http://arxiv.org/abs/2511.15874v1](http://arxiv.org/abs/2511.15874v1)

**作者:** Sajjad Pakdamansavoji, Yintao Ma, Amir Rasouli, Tongtong Cao

**发布时间:** 2025-11-19

### GPT解析

### 总结

该论文提出了一种改进的6D物体姿态估计方法，特别针对遮挡情况下的挑战，通过四种创新技术提高了准确性和鲁棒性，同时提高了推理速度。

### 背景

准确的6D物体姿态估计对机器人、增强现实和场景理解至关重要。对于已知物体，通过针对每个物体的微调通常可以实现高精度，但推广到未知物体仍然是一个挑战。现有的方法通常假设在测试时可以访问CAD模型，并遵循多阶段流程来估计姿态：检测和分割物体、提出初始姿态，然后进行优化。然而，在遮挡情况下，这种管道的早期阶段容易出错，这些错误会通过顺序处理传播，从而降低性能。

### 目的

解决现有6D物体姿态估计方法在遮挡情况下的局限性，提高准确性和鲁棒性，同时加速推理过程。

### 方法

作者提出了四种基于模型的6D姿态估计方法的创新扩展：(i) 动态非均匀密集采样策略，专注于可见区域，减少遮挡引起的错误；(ii) 多假设推理机制，保留多个置信度排序的姿态候选，减轻脆弱的单路径故障；(iii) 迭代优化，逐步提高姿态准确性；(iv) 一系列针对遮挡的训练增强，增强鲁棒性和泛化能力。此外，作者还提出了一种新的基于可见性加权的评估指标，用于在遮挡情况下评估，以减少现有协议中的偏差。

### 主要发现

通过大量实证评估，作者提出的方法在ICBIN数据集上准确率提高了5%以上，在BOP数据集基准上提高了2%以上，同时推理速度提高了约3倍。

### 结论

作者提出的创新方法有效解决了遮挡情况下的6D物体姿态估计挑战，显著提高了准确性和鲁棒性，同时加速了推理过程。

### 翻译

准确的6D物体姿态估计对机器人、增强现实和场景理解至关重要。对于已知物体，通过针对每个物体的微调通常可以实现高精度，但推广到未知物体仍然是一个挑战。为了解决这个问题，现有方法假设在测试时可以访问CAD模型，并通常遵循多阶段流程来估计姿态：检测和分割物体，提出初始姿态，然后进行优化。然而，在遮挡情况下，这种管道的早期阶段容易出错，这些错误会通过顺序处理传播，从而降低性能。为了弥补这一不足，我们提出了基于模型的6D姿态估计方法的四种创新扩展：(i) 动态非均匀密集采样策略，专注于可见区域，减少遮挡引起的错误；(ii) 多假设推理机制，保留多个置信度排序的姿态候选，减轻脆弱的单路径故障；(iii) 迭代优化，逐步提高姿态准确性；(iv) 一系列针对遮挡的训练增强，增强鲁棒性和泛化能力。此外，我们还提出了一种新的基于可见性加权的评估指标，用于在遮挡情况下评估，以减少现有协议中的偏差。通过大量实证评估，我们表明提出的方法在ICBIN数据集上准确率提高了5%以上，在BOP数据集基准上提高了2%以上，同时推理速度提高了约3倍。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在物体部分被遮挡的情况下，对未见过的物体进行准确的6D姿态估计问题。这个问题在机器人抓取、增强现实叠加和场景理解等领域至关重要，因为在现实场景中物体经常被其他物体部分遮挡，而现有方法在遮挡情况下容易产生误差，尤其是对于训练过程中没有见过的物体。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法在遮挡情况下的局限性，如早期阶段错误传播、间接处理遮挡、计算成本高以及对环境有强假设限制等问题，设计了结合推理时策略和训练时增强的方法。作者借鉴了MUSE进行3D感知目标检测、Grounding DINO和SAM2进行边界框提议和分割、DINOv2-L进行特征编码、几何Transformer进行特征表示和交互、SVD解决姿态估计以及InfoNCE目标函数进行对应监督等现有工作。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想包括：1)动态非均匀密集采样策略，将计算资源集中在可见区域；2)多假设推理机制，保留多个姿态候选；3)迭代优化，逐步提高姿态准确性；4)遮挡焦点训练增强，增强对部分可见性的鲁棒性；5)遮挡感知评估指标，提供无偏性能估计。整体流程为：3D感知目标检测→特征提取→粗略点匹配生成初始假设→动态非均匀密集采样→密集点匹配→迭代姿态优化→应用遮挡焦点训练增强→使用无偏指标评估。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)动态非均匀密集采样策略，根据遮挡概率分配计算资源；2)多假设推理机制，保留多个姿态候选；3)迭代优化，逐步提高姿态准确性；4)专门设计的遮挡焦点训练增强；5)遮挡感知的无偏评估指标。相比之前的工作，WALDO系统地解决了遮挡问题，从采样策略、推理机制、优化过程、训练方法和评估指标等多个方面进行了创新，同时实现了约3倍的推理速度提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'WALDO通过动态非均匀采样、多假设推理、迭代优化和专门设计的遮挡增强方法，显著提高了在部分遮挡情况下对未见物体的6D姿态估计准确性，同时提出了无偏评估指标来更公平地衡量模型性能。'}


### 论文摘要

Accurate 6D object pose estimation is vital for robotics, augmented reality, and scene understanding. For seen objects, high accuracy is often attainable via per-object fine-tuning but generalizing to unseen objects remains a challenge. To address this problem, past arts assume access to CAD models at test time and typically follow a multi-stage pipeline to estimate poses: detect and segment the object, propose an initial pose, and then refine it. Under occlusion, however, the early-stage of such pipelines are prone to errors, which can propagate through the sequential processing, and consequently degrade the performance. To remedy this shortcoming, we propose four novel extensions to model-based 6D pose estimation methods: (i) a dynamic non-uniform dense sampling strategy that focuses computation on visible regions, reducing occlusion-induced errors; (ii) a multi-hypothesis inference mechanism that retains several confidence-ranked pose candidates, mitigating brittle single-path failures; (iii) iterative refinement to progressively improve pose accuracy; and (iv) series of occlusion-focused training augmentations that strengthen robustness and generalization. Furthermore, we propose a new weighted by visibility metric for evaluation under occlusion to minimize the bias in the existing protocols. Via extensive empirical evaluations, we show that our proposed approach achieves more than 5% improvement in accuracy on ICBIN and more than 2% on BOP dataset benchmarks, while achieving approximately 3 times faster inference.

---

## 72. A first look at a complete view of spatially resolved star formation at 1<z<1.8 with JWST NGDEEP+FRESCO slitless spectroscopy

**论文链接:** [http://arxiv.org/abs/2511.15792v1](http://arxiv.org/abs/2511.15792v1)

**作者:** Jasleen Matharu, Lu Shen, Irene Shivaei, Pascal A. Oesch, Casey Papovich, Gabriel Brammer, Naveen A. Reddy, Yingjie Cheng, Pieter van Dokkum, Steven L. Finkelstein, Nimish P. Hathi, Jeyhan S. Kartaltepe, Anton M. Koekemoer, Jorryt Matthee, Nor Pirzkal, Stephen M. Wilkins, Michael A. Wozniak, Mengyuan Xiao

**发布时间:** 2025-11-19

**备注:** 7 pages, 7 figures, to be submitted to A&A

### GPT解析

### 总结

研究利用JWST首次实现了对遥远星系中恒星形成指示器Paα的空间分辨，比较了被尘埃遮蔽与未被遮蔽的恒星形成在主序列星系中的空间分布，发现星系通过由内向外生长的模式普遍存在，但中心集中的尘埃衰减并非如此。

### 背景

以前无法获取的恒星形成指示器Paα现在可以通过JWST NIRCam无狭缝光谱学在遥远的星系中（直到宇宙中午）进行空间分辨。

### 目的

提供被尘埃遮蔽（由Paα追踪）与未被遮蔽（由Hα追踪）的恒星形成在主序列上的首次直接空间分辨比较。

### 方法

对31个在1<z<1.8的星系（按恒星质量分为三组）的Paα和Hα发射线图以及两个波长处的恒星连续谱图像进行堆叠，测量表面亮度剖面并计算等效宽度（EW）剖面。

### 主要发现

1) 在所有探测的恒星质量范围内，Paα和Hα的EW剖面随银河中心半径增加，首次提供了通过被尘埃遮蔽和未被遮蔽的恒星形成实现星系由内向外生长的直接证据；2) 对于主序列星系，不同质量范围内的星系表现出不同的Paα/Hα线剖面随半径变化的模式；3) 具有高sSFR的低质量星系显示出负相关的Paα/Hα线剖面梯度。

### 结论

尽管在宇宙中午之后主序列上的星系通过恒星形成实现由内向外生长是普遍存在的，但中心集中的尘埃衰减并非如此。研究结果激励未来对主序列上大量宇宙中午个体星系的空间分辨恒星形成剖面进行研究，以理解空间分辨恒星形成的内在离散性。

### 翻译

以前无法获取的恒星形成指示器Paα现在可以通过JWST NIRCam无狭缝光谱学在遥远的星系中（直到宇宙中午）进行空间分辨。在这项首次研究中，我们结合了JWST NGDEEP NIRISS和FRESCO NIRCam无狭缝光谱学，首次提供了被尘埃遮蔽（由Paα追踪）与未被遮蔽（由Hα追踪）的恒星形成在主序列上的直接空间分辨比较。我们对31个在1<z<1.8的星系（按恒星质量分为三组）的Paα和Hα发射线图以及两个波长处的恒星连续谱图像进行堆叠。测量表面亮度剖面并计算等效宽度（EW）剖面。在所有探测的恒星质量范围内，Paα和Hα的EW剖面随银河中心半径增加，这首次提供了通过被尘埃遮蔽和未被遮蔽的恒星形成实现星系由内向外生长的直接证据。对于主序列星系，在中等质量范围内发现微弱正相关的Paα/Hα线剖面随半径变化，在高质量范围内发现负相关的Paα/Hα线剖面随半径变化。具有高特定恒星形成率的低质量星系显示出负相关的Paα/Hα线剖面梯度。我们的结果表明，尽管在宇宙中午之后主序列上的星系通过恒星形成实现由内向外生长是普遍存在的，但中心集中的尘埃衰减并非如此。结合文献中的其他近期工作，我们的发现激励未来对主序列上大量宇宙中午个体星系的空间分辨恒星形成剖面进行研究，以理解空间分辨恒星形成的内在离散性。


### 论文摘要

[abridged] The previously inaccessible star formation tracer Pa$α$ can now be spatially resolved by JWST NIRCam slitless spectroscopy in distant galaxies up to cosmic noon. In the first study of its kind, we combine JWST NGDEEP NIRISS and FRESCO NIRCam slitless spectroscopy to provide the first direct comparison of spatially resolved dust-obscured (traced by Pa$α$) versus unobscured (traced by H$α$) star formation across the main sequence. We stack Pa$α$ and H$α$ emission-line maps, along with stellar continuum images at both wavelengths of 31 galaxies at 1<z<1.8 in three bins of stellar mass. Surface brightness profiles are measured and equivalent width (EW) profiles computed. Increasing Pa$α$ and H$α$ EW profiles with galactocentric radius across all stellar masses probed provide direct evidence for the inside-out growth of galaxies both via dust-obscured and unobscured star formation for the first time. For galaxies on the main sequence, a weakly positive ($0.1\pm0.1$) Pa$α$/H$α$ line profile as a function of radius is found at $8.8\leqslant\mathrm{log}(M_{*}/\mathrm{M}_{\odot})<9.9$ with a negative ($-0.4\pm0.1$) Pa$α$/H$α$ line profile found at $9.9\leqslant\mathrm{log}(M_{*}/\mathrm{M}_{\odot})<11.0$. Low mass galaxies ($7.7\leqslant\mathrm{log}(M_{*}/\mathrm{M}_{\odot})<8.8$) with high sSFRs are found to have a negative ($-0.5\pm0.1$) Pa$α$/H$α$ line profile gradient. Our results demonstrate that while inside-out growth via star formation is ubiquitous across the main sequence just after cosmic noon, centrally concentrated dust attenuation is not. Along with other recent work in the literature, our findings motivate future studies of resolved SFR profiles in large samples of individual cosmic noon galaxies across the main sequence, to understand the intrinsic scatter in spatially resolved star formation.

---

## 73. Revealing Phonon Bridge Effect for Amorphous vs Crystalline Metal-Silicide Layers at Si/Ti Interfaces by a Machine Learning Potential

**论文链接:** [http://arxiv.org/abs/2511.16646v1](http://arxiv.org/abs/2511.16646v1)

**作者:** Mayur Singh, Lokanath Patra, Chengyang Zhang, Greg MacDougall, Suman Datta, David Cahill, Satish Kumar

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究开发了一种用于Si-Ti系统的神经进化势，能够准确模拟金属-半导体界面的热传输特性，并通过实验验证了其准确性，发现了界面层厚度和晶相对热传输的影响规律。

### 背景

金属-半导体界面在微纳米电子器件中起着核心作用，因为这些界面上的热散逸或温度下降会显著影响器件性能。预测这些界面的精确热边界电阻(TBR)具有挑战性。

### 目的

开发一种统一的方法来预测金属-半导体界面的热边界电阻，考虑实际结构及其与底层热传输的相关性。

### 方法

开发了一种用于Si-Ti系统的统一神经进化势(NEP)，能够准确重现体Si、Ti和TiSi2的能量、力和声子特性，并自然扩展到界面环境以分析界面传输。这种方法能够模拟金属-半导体界面上的复杂结构，并允许进行大规模非平衡分子动力学模拟。

### 主要发现

模拟的TBR与时间域热反射(TDTR)测量结果有极好的一致性；当厚度小于1.5nm时，非晶TiSi2界面层有助于高效界面传输，但当厚度超过1.5nm时，这一趋势会反转；比较不同TiSi2晶相在Si/TiSi2界面的TBR表明，C54相的TBR低于C49相，这与它们与Si的声子态密度(PDOS)重叠差异相关。

### 结论

这些结果提供了关于晶态与非晶态硅化物在界面热传输中作用的原子级见解，并展示了一种可转移的机器学习势，用于研究先进半导体器件中的热散逸。

### 翻译

金属-半导体界面在微纳米电子器件中起着核心作用，因为这些界面上的热散逸或温度下降会显著影响器件性能。预测这些界面的精确热边界电阻(TBR)，考虑实际结构及其与底层热传输的相关性，仍然具有挑战性。在本工作中，我们为Si-Ti系统开发了一种统一的神经进化势(NEP)，能够准确重现体Si、Ti和TiSi2的能量、力和声子特性，并自然扩展到界面环境以分析界面传输。与当前机器学习原子间势相比，一个重要发展是能够模拟金属-半导体界面上的复杂结构，因为NEP允许对外延Si/Ti界面进行大规模非平衡分子动力学模拟，以阐明非晶或晶态硅化物界面层的影响。模拟的TBR与我们时间域热反射(TDTR)测量结果有极好的一致性。光谱分析显示，当厚度小于1.5nm时，非晶TiSi2界面层有助于高效界面传输，而晶态TiSi2层则相反，但当界面层厚度增加到超过1.5nm时，这一趋势会反转。比较不同TiSi2晶相在Si/TiSi2界面的TBR表明，与C49相相比，C54相的TBR降低，这与它们与Si的声子态密度(PDOS)重叠差异相关。这些结果提供了关于晶态与非晶态硅化物在界面热传输中作用的原子级见解，并展示了一种可转移的机器学习势，用于研究先进半导体器件中的热散逸。


### 论文摘要

Metal-semiconductor interfaces play a central role in micro and nano-electronic devices as heat dissipation or temperature drop across these interfaces can significantly affect device performance. Prediction of accurate thermal boundary resistance (TBR) across these interfaces, considering realistic structures and their correlation with underlying thermal transport, remains challenging. In this work we develop a unified Neuroevolution Potential (NEP) for the Si-Ti system that accurately reproduces energies, forces, and phonon properties of bulk Si, Ti, and TiSi2 and extends naturally to interfacial environments to analyze interfacial transport. An important development over current machine-learned interatomic potentials is the capability to model complex structures at metal-semiconductor interfaces, as the NEP enables large scale non-equilibrium molecular dynamics simulations of epitaxial Si/Ti interfaces to elucidate the effect of amorphous or crystalline silicide interfacial layers. Simulated TBRs show excellent agreement with our time-domain thermoreflectance (TDTR) measurements. Spectral analyses reveal that amorphous TiSi2 interfacial layer helps in efficient interfacial transport when the thickness is less than 1.5 nm compared to the crystalline TiSi2 layer, but this trend reverses when the interfacial layer thickness increases beyond 1.5 nm. Comparison of TBRs at Si/TiSi2 interface for different crystalline phases of TiSi2 establishes that C54 phase has reduced TBR compared to C49 phase, which is correlated with the difference in their phonon density of states (PDOS) overlap with Si. These results provide atomistic insight into the role of crystalline versus amorphous silicides in interfacial heat transport and demonstrate a transferable machine-learned potential for studying heat dissipation in advanced semiconductor devices.

---

## 74. Deep Learning Framework for Enhanced Neutrino Reconstruction of Single-line Events in the ANTARES Telescope

**论文链接:** [http://arxiv.org/abs/2511.16614v1](http://arxiv.org/abs/2511.16614v1)

**作者:** A. Albert, S. Alves, M. André, M. Ardid, S. Ardid, J. -J. Aubert, J. Aublin, B. Baret, S. Basa, Y. Becherini, B. Belhorma, F. Benfenati, V. Bertin, S. Biagi, J. Boumaaza, M. Bouta, M. C. Bouwhuis, H. Brânzaş, R. Bruijn, J. Brunner, J. Busto, B. Caiffi, D. Calvo, S. Campion, A. Capone, F. Carenini, J. Carr, V. Carretero, T. Cartraud, S. Celli, L. Cerisy, M. Chabab, R. Cherkaoui El Moursli, T. Chiarusi, M. Circella, J. A. B. Coelho, A. Coleiro, R. Coniglione, P. Coyle, A. Creusot, A. F. Díaz, B. De Martino, C. Distefano, I. Di Palma, C. Donzaud, D. Dornic, D. Drouhin, T. Eberl, A. Eddymaoui, T. van Eeden, D. van Eijk, S. El Hedri, N. El Khayati, A. Enzenhöfer, P. Fermani, G. Ferrara, F. Filippini, L. Fusco, S. Gagliardini, J. García-Méndez, C. Gatius Oliver, P. Gay, N. Geißelbrecht, H. Glotin, R. Gozzini, R. Gracia Ruiz, K. Graf, C. Guidi, L. Haegel, H. van Haren, A. J. Heijboer, Y. Hello, L. Hennig, J. J. Hernández-Rey, J. Hößl, F. Huang, G. Illuminati, B. Jisse-Jung, M. de Jong, P. de Jong, M. Kadler, O. Kalekin, U. Katz, A. Kouchner, I. Kreykenbohm, V. Kulikovskiy, R. Lahmann, M. Lamoureux, A. Lazo, D. Lefèvre, E. Leonora, G. Levi, S. Le Stum, S. Loucatos, J. Manczak, M. Marcelin, A. Margiotta, A. Marinelli, J. A. Martínez-Mora, P. Migliozzi, A. Moussa, R. Muller, S. Navas, E. Nezri, B. Ó Fearraigh, E. Oukacha, A. M. Păun, G. E. Păvălaş, S. Peña-Martínez, M. Perrin-Terrin, P. Piattelli, C. Poiré, V. Popa, T. Pradier, N. Randazzo, D. Real, G. Riccobene, A. Romanov, A. Sánchez Losa, A. Saina, F. Salesa Greus, D. F. E. Samtleben, M. Sanguineti, P. Sapienza, F. Schüssler, J. Seneca, M. Spurio, Th. Stolarczyk, M. Taiuti, Y. Tayalati, B. Vallage, G. Vannoye, V. Van Elewyck, S. Viola, D. Vivolo, J. Wilms, S. Zavatarelli, A. Zegarelli, J. D. Zornoza, J. Zúñiga

**发布时间:** 2025-11-20

### GPT解析

### 总结

N-fit算法是一种基于深度学习的神经网络模型，用于改进ANTARES水下望远镜探测到的低能中微子事件的重建，显著提高了方向和能量估计的准确性。

### 背景

ANTARES水下望远镜通常探测到低能中微子事件（约100 GeV），传统方法如χ²拟合方法在重建这些事件时存在局限性，特别是在方向和能量估计方面。

### 目的

开发一种先进的算法来提高单线中微子事件的重建质量，包括方向（天顶角和方位角）和能量的准确估计，以及事件拓扑分类。

### 方法

N-Fit是一个神经网络模型，结合了深度卷积层、混合密度输出层和迁移学习技术。该框架将重建过程分为两个专用分支（针对轨迹和簇射事件拓扑），包含空间估计和能量推断的子模型，并通过迁移学习整合关键特征。

### 主要发现

N-Fit显著优化了单线事件的天顶角估计，提供了可靠的方位角预测（传统方法无法实现），并通过迁移学习有效改进了能量估计。所有重建参数的平均和中位绝对误差在蒙特卡洛模拟和数据测试中均显著减少。

### 结论

N-Fit算法的改进展示了其在推进多信使天体物理学方面的潜力，并增强了我们使用ANTARES数据的单线事件探测超出标准模型基本物理的能力。

### 翻译

我们提出了N-fit算法，旨在改进ANTARES水下望远镜单线探测到的中微子事件的重建，这些事件通常与低能中微子事件（约100 GeV）相关。N-Fit是一个依赖深度学习的神经网络模型，结合了机器学习中的多种先进技术——深度卷积层、混合密度输出层和迁移学习。该框架将重建过程分为针对每种中微子事件拓扑结构（轨迹和簇射）的两个专用分支，包含空间估计（方向和位置）和能量推断的子模型，随后组合进行事件分类。关于单线事件的方向，N-Fit算法显著优化了天顶角的估计，并提供了可靠的方位角预测，这是传统χ²拟合方法无法实现的。改进单线事件的能量估计是一项艰巨任务；N-Fit通过迁移学习有效整合了关键特征，如事件到探测器的最近距离估计。N-Fit还通过冻结预训练分支的卷积层，在事件拓扑分类中利用迁移学习。蒙特卡洛模拟和数据的测试表明，所有重建参数的平均和中位绝对误差均显著减少。N-Fit实现的改进凸显了其在推进多信使天体物理学方面的潜力，并增强了我们使用ANTARES数据单线事件探测超出标准模型基本物理的能力。


### 论文摘要

We present the $N$-fit algorithm designed to improve the reconstruction of neutrino events detected by a single line of the ANTARES underwater telescope, usually associated with low energy neutrino events ($\sim$ 100 GeV). $N$-Fit is a neural network model that relies on deep learning and combines several advanced techniques in machine learning --deep convolutional layers, mixture density output layers, and transfer learning. This framework divides the reconstruction process into two dedicated branches for each neutrino event topology --tracks and showers-- composed of sub-models for spatial estimation --direction and position-- and energy inference, which later on are combined for event classification. Regarding the direction of single-line events, the $N$-Fit algorithm significantly refines the estimation of the zenithal angle, and delivers reliable azimuthal angle predictions that were previously unattainable with traditional $χ^2$-fit methods. Improving on energy estimation of single-line events is a tall order; $N$-Fit benefits from transfer learning to efficiently integrate key characteristics, such as the estimation of the closest distance from the event to the detector. $N$-Fit also takes advantage from transfer learning in event topology classification by freezing convolutional layers of the pretrained branches. Tests on Monte Carlo simulations and data demonstrate a significant reduction in mean and median absolute errors across all reconstructed parameters. The improvements achieved by $N$-Fit highlight its potential for advancing multimessenger astrophysics and enhancing our ability to probe fundamental physics beyond the Standard Model using single-line events from ANTARES data.

---

## 75. Correlation-Aware Feature Attribution Based Explainable AI

**论文链接:** [http://arxiv.org/abs/2511.16482v1](http://arxiv.org/abs/2511.16482v1)

**作者:** Poushali Sengupta, Yan Zhang, Frank Eliassen, Sabita Maharjan

**发布时间:** 2025-11-20

**备注:** Accepted, 2026 International Conference on Advances in Artificial Intelligence and Machine Learning (AAIML 2026)

### GPT解析

### 总结

ExCIR是一种相关性感知的归因分数，配备轻量级传输协议，只需使用一小部分数据即可重现完整模型排名。它通过稳健中心化量化特征与模型输出间的符号对齐共同运动，并提供了BlockCIR扩展来处理相关特征组。ExCIR在多种数据集上表现出与基线方法的一致性，提供更稳定的排名，且计算效率高。

### 背景

随着现代模型变得日益复杂，高风险应用对透明度、信任和监管合规性的需求使得可解释AI(XAI)变得越来越重要。

### 目的

解决现有全局归因方法计算成本高、在相关输入下缺乏稳定性、无法有效扩展到大型或异构数据集的问题。

### 方法

提出ExCIR(通过相关性影响比率的可解释性)，一种相关性感知的归因分数，配备轻量级传输协议；通过稳健中心化量化特征与模型输出间的符号对齐共同运动；引入BlockCIR作为ExCIR的分组扩展，将相关特征组作为一个单元评分，减轻共线性簇中的重复计数问题。

### 主要发现

在多样化的文本、表格、信号和图像数据集上，ExCIR与既定的全局基线和完整模型显示出可信的一致性；在不同设置下提供一致的top-k排名；通过在行子集上进行轻量级评估减少运行时间；当存在强依赖性时产生更平滑、更稳定的排名。

### 结论

ExCIR为实际部署提供了计算高效、一致且可扩展的可解释性解决方案。

### 翻译

可解释AI(XAI)随着现代模型变得越来越复杂以及高风险应用要求透明度、信任和监管合规性而变得越来越重要。现有的全局归因方法通常计算成本高，在相关输入下缺乏稳定性，并且无法有效地扩展到大型或异构数据集。我们通过ExCIR(通过相关性影响比率的可解释性)解决了这些差距，这是一种配备轻量级传输协议的相关性感知归因分数，只需使用一小部分数据即可重现完整模型的排名。ExCIR在稳健中心化(从特征和输出中减去稳健的位置估计，如中位数或中位数均值)后量化特征与模型输出之间的符号对齐的共同运动。我们进一步介绍了BlockCIR，这是ExCIR的分组扩展，将一组相关特征作为一个单元评分。通过在预定义或数据驱动的组上聚合相同的符号共同运动分子和幅度，BlockCIR减轻了共线性簇(如同义词或重复传感器)中的重复计数，并在存在强依赖性时产生更平滑、更稳定的排名。在多样化的文本、表格、信号和图像数据集上，ExCIR显示出与既定的全局基线和完整模型可信的一致性，在不同设置下提供一致的top-k排名，并通过在行子集上进行轻量级评估减少运行时间。总的来说，ExCIR为实际部署提供了计算高效、一致且可扩展的可解释性。


### 论文摘要

Explainable AI (XAI) is increasingly essential as modern models become more complex and high-stakes applications demand transparency, trust, and regulatory compliance. Existing global attribution methods often incur high computational costs, lack stability under correlated inputs, and fail to scale efficiently to large or heterogeneous datasets. We address these gaps with \emph{ExCIR} (Explainability through Correlation Impact Ratio), a correlation-aware attribution score equipped with a lightweight transfer protocol that reproduces full-model rankings using only a fraction of the data. ExCIR quantifies sign-aligned co-movement between features and model outputs after \emph{robust centering} (subtracting a robust location estimate, e.g., median or mid-mean, from features and outputs). We further introduce \textsc{BlockCIR}, a \emph{groupwise} extension of ExCIR that scores \emph{sets} of correlated features as a single unit. By aggregating the same signed-co-movement numerators and magnitudes over predefined or data-driven groups, \textsc{BlockCIR} mitigates double-counting in collinear clusters (e.g., synonyms or duplicated sensors) and yields smoother, more stable rankings when strong dependencies are present. Across diverse text, tabular, signal, and image datasets, ExCIR shows trustworthy agreement with established global baselines and the full model, delivers consistent top-$k$ rankings across settings, and reduces runtime via lightweight evaluation on a subset of rows. Overall, ExCIR provides \emph{computationally efficient}, \emph{consistent}, and \emph{scalable} explainability for real-world deployment.

---

## 76. PersonaDrift: A Benchmark for Temporal Anomaly Detection in Language-Based Dementia Monitoring

**论文链接:** [http://arxiv.org/abs/2511.16445v1](http://arxiv.org/abs/2511.16445v1)

**作者:** Joy Lai, Alex Mihailidis

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文介绍了PersonaDrift，一个用于评估机器学习和统计方法检测痴呆症患者日常沟通行为渐进性变化的合成基准工具。

### 背景

痴呆症患者(PLwD)的沟通方式通常逐渐变化，表现为表达减少、重复增加或话题偏离，而现有计算工具大多无法跟踪此类行为漂移。

### 目的

开发一个基准工具来评估检测痴呆症患者与数字提醒系统交互时沟通行为随时间变化的方法。

### 方法

PersonaDrift基于对护理人员的访谈，模拟60天合成用户交互日志，关注情感平淡和语义漂移两种变化，并评估了多种方法包括统计模型、序列模型和分类器。

### 主要发现

情感平淡变化在基线变异性低的用户中可通过简单统计模型检测，而语义漂移检测需要时间建模和个人化基线；个性化分类器在两项任务中均优于通用分类器。

### 结论

PersonaDrift为评估检测痴呆症患者沟通行为变化提供了有效工具，个性化方法在行为变化检测中表现更佳。

### 翻译

痴呆症患者(PLwD)通常表现出沟通方式的逐渐转变，表达变得不那么丰富，更加重复，或者以微妙的方式偏离主题。虽然护理人员可能会非正式地注意到这些变化，但大多数计算工具并非设计用来随时间跟踪此类行为漂移。本文介绍了PersonaDrift，这是一个合成基准，旨在评估机器学习和统计方法检测日常交流中渐进性变化的能力，特别关注用户对数字提醒系统的响应。PersonaDrift基于对护理人员的访谈，模拟了类似于真实PLwD的合成用户的60天交互日志。这些基于护理人员信息的角色在语调、方式和沟通习惯上各不相同，实现了行为的真实多样性。该基准重点关注护理人员强调的两种纵向变化，这些变化尤为显著：情感平淡（减少情感语调和冗长性）和离题回复（语义漂移）。这些变化以不同速率渐进注入，以模拟自然认知轨迹，该框架设计为可在未来用例中扩展到其他行为。


### 论文摘要

People living with dementia (PLwD) often show gradual shifts in how they communicate, becoming less expressive, more repetitive, or drifting off-topic in subtle ways. While caregivers may notice these changes informally, most computational tools are not designed to track such behavioral drift over time. This paper introduces PersonaDrift, a synthetic benchmark designed to evaluate machine learning and statistical methods for detecting progressive changes in daily communication, focusing on user responses to a digital reminder system. PersonaDrift simulates 60-day interaction logs for synthetic users modeled after real PLwD, based on interviews with caregivers. These caregiver-informed personas vary in tone, modality, and communication habits, enabling realistic diversity in behavior. The benchmark focuses on two forms of longitudinal change that caregivers highlighted as particularly salient: flattened sentiment (reduced emotional tone and verbosity) and off-topic replies (semantic drift). These changes are injected progressively at different rates to emulate naturalistic cognitive trajectories, and the framework is designed to be extensible to additional behaviors in future use cases. To explore this novel application space, we evaluate several anomaly detection approaches, unsupervised statistical methods (CUSUM, EWMA, One-Class SVM), sequence models using contextual embeddings (GRU + BERT), and supervised classifiers in both generalized and personalized settings. Preliminary results show that flattened sentiment can often be detected with simple statistical models in users with low baseline variability, while detecting semantic drift requires temporal modeling and personalized baselines. Across both tasks, personalized classifiers consistently outperform generalized ones, highlighting the importance of individual behavioral context.

---

## 77. Unsupervised Graph Neural Network Framework for Balanced Multipatterning in Advanced Electronic Design Automation Layouts

**论文链接:** [http://arxiv.org/abs/2511.16374v1](http://arxiv.org/abs/2511.16374v1)

**作者:** Abdelrahman Helaly, Nourhan Sakr, Kareem Madkour, Ilhami Torunoglu

**发布时间:** 2025-11-20

**备注:** manuscript under review

### GPT解析

### 总结

本研究提出了一种基于GNN的混合工作流程，用于解决电子设计自动化中的多图案分解问题，通过结合图着色方法和细化策略，实现了无冲突分解和颜色平衡。

### 背景

多图案分解是电子设计自动化(EDA)中一种必要的分解策略，用于克服在打印密集电路布局时的光刻限制。现有的基于启发式回溯和SAT求解器的方法虽然可以应对这些挑战，但往往难以同时处理复杂约束和次要目标。

### 目的

提出一种混合工作流程，将多图案分解视为约束图着色问题的变体，主要目标是最小化特征违规，次要目标是平衡每个掩模上的特征数量。

### 方法

该流程集成了两个主要组件：(1)一种基于GNN的代理，以无监督方式训练生成初始颜色预测；(2)通过基于GNN的启发式和模拟退火等细化策略进行改进，共同提高解决方案质量和平衡性。

### 主要发现

在专有数据集和公开可用的开源布局上的实验评估表明，该方法能够实现完全无冲突的分解和一致的颜色平衡。

### 结论

所提出的框架为EDA工作流中可扩展的布局分解提供了一个可复现、数据高效且可部署的基准。

### 翻译

多图案分解是电子设计自动化(EDA)中一种必要的分解策略，它能够克服在打印密集电路布局时的光刻限制。尽管基于启发式回溯和SAT求解器的方法可以解决这些挑战，但它们通常难以同时处理复杂约束和次要目标。在本研究中，我们提出了一种混合工作流程，将多图案分解视为约束图着色问题的一种变体，主要目标是最小化特征违规，次要目标是平衡每个掩模上的特征数量。我们的流程集成了两个主要组件：(1)一种基于GNN的代理，以无监督方式训练生成初始颜色预测，这些预测通过(2)细化策略(基于GNN的启发式和模拟退火)进行改进，共同提高解决方案质量和平衡性。在专有数据集和公开可用的开源布局上的实验评估表明，该方法能够实现完全无冲突的分解和一致的颜色平衡。所提出的框架为EDA工作流中可扩展的布局分解提供了一个可复现、数据高效且可部署的基准。


### 论文摘要

Multipatterning is an essential decomposition strategy in electronic design automation (EDA) that overcomes lithographic limitations when printing dense circuit layouts. Although heuristic-based backtracking and SAT solvers can address these challenges, they often struggle to simultaneously handle both complex constraints and secondary objectives. In this study, we present a hybrid workflow that casts multipatterning as a variant of a constrained graph coloring problem with the primary objective of minimizing feature violations and a secondary objective of balancing the number of features on each mask. Our pipeline integrates two main components: (1) A GNN-based agent, trained in an unsupervised manner to generate initial color predictions, which are refined by (2) refinement strategies (a GNN-based heuristic and simulated annealing) that together enhance solution quality and balance. Experimental evaluation in both proprietary data sets and publicly available open source layouts demonstrate complete conflict-free decomposition and consistent color balancing. The proposed framework provides a reproducible, data-efficient and deployable baseline for scalable layout decomposition in EDA workflows.

---

## 78. ChangeDINO: DINOv3-Driven Building Change Detection in Optical Remote Sensing Imagery

**论文链接:** [http://arxiv.org/abs/2511.16322v1](http://arxiv.org/abs/2511.16322v1)

**作者:** Ching-Heng Cheng, Chih-Chung Hsu

**发布时间:** 2025-11-20

### GPT解析

### 总结

论文提出了ChangeDINO，一个用于光学建筑变化检测的端到端多尺度Siamese框架，能够有效利用语义信息并提高变化检测的鲁棒性。

### 背景

遥感变化检测(RSCD)旨在从配准的双时相图像中识别地表变化。然而，许多基于深度学习的RSCD方法仅依赖变化图标注，未充分利用非变化区域的语义信息，这导致在光照变化、非垂直视角和标签稀缺的情况下鲁棒性不足。

### 目的

开发一个能够更好利用语义信息的变化检测方法，提高在复杂条件下的鲁棒性，实现更精确的建筑变化检测。

### 方法

提出了ChangeDINO，一个端到端多尺度Siamese框架，融合轻量级骨干流与从冻结的DINOv3转移的特征；使用空间-光谱差分变换器解码器，利用多尺度绝对差异作为变化先验；包含一个可学习的形态学模块，用于优化上采样逻辑以恢复清晰边界。

### 主要发现

在四个公共基准测试中，ChangeDINO在IoU和F1指标上一致优于最新的先进方法，消融研究证实了每个组件的有效性。

### 结论

ChangeDINO通过充分利用语义信息和创新的架构设计，显著提高了建筑变化检测的性能，即使在数据集较小的情况下也能生成语义丰富和上下文丰富的特征金字塔。

### 翻译

遥感变化检测旨在从配准的双时相图像中识别地表变化。然而，许多基于深度学习的遥感变化检测方法仅依赖变化图标注，且未充分利用非变化区域的语义信息，这限制了在光照变化、非垂直视角和标签稀缺情况下的鲁棒性。本文介绍了ChangeDINO，一个用于光学建筑变化检测的端到端多尺度Siamese框架。该模型将轻量级骨干流与从冻结DINOv3转移的特征相融合，即使在小型数据集上也能生成语义丰富和上下文丰富的特征金字塔。随后，空间-光谱差分变换器解码器利用多尺度绝对差异作为变化先验，以突出真实建筑变化并抑制无关响应。最后，一个可学习的形态学模块优化上采样逻辑以恢复清晰边界。在四个公共基准上的实验表明，ChangeDINO在IoU和F1指标上一致优于最近的先进方法，消融研究证实了每个组件的有效性。源代码可在https://github.com/chingheng0808/ChangeDINO获取。


### 论文摘要

Remote sensing change detection (RSCD) aims to identify surface changes from co-registered bi-temporal images. However, many deep learning-based RSCD methods rely solely on change-map annotations and underuse the semantic information in non-changing regions, which limits robustness under illumination variation, off-nadir views, and scarce labels. This article introduces ChangeDINO, an end-to-end multiscale Siamese framework for optical building change detection. The model fuses a lightweight backbone stream with features transferred from a frozen DINOv3, yielding semantic- and context-rich pyramids even on small datasets. A spatial-spectral differential transformer decoder then exploits multi-scale absolute differences as change priors to highlight true building changes and suppress irrelevant responses. Finally, a learnable morphology module refines the upsampled logits to recover clean boundaries. Experiments on four public benchmarks show that ChangeDINO consistently outperforms recent state-of-the-art methods in IoU and F1, and ablation studies confirm the effectiveness of each component. The source code is available at https://github.com/chingheng0808/ChangeDINO.

---

## 79. Optimized User Experience for Labeling Systems for Predictive Maintenance Applications (Extended)

**论文链接:** [http://arxiv.org/abs/2511.16266v1](http://arxiv.org/abs/2511.16266v1)

**作者:** Michelle Hallmann, Michael Stern, Juliane Henning, Ute Franke, Thomas Ostertag, Joao Paulo Javidi da Costa, Jan-Niklas Voigt-Antons

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究介绍了一个名为DigiOnTrack的经济有效预测性维护系统，结合结构噪声测量与监督学习，为德国农村地区的铁路车辆和基础设施提供监控维护建议。系统整合了无线传感器网络、分布式账本技术和Docker化容器架构，由火车司机和车间主管进行故障标注。可用性评估显示火车司机界面获优秀评级，车间主管界面为良好，系统有潜力融入日常工作流程，但明晰性等数据密集场景领域需进一步优化。

### 背景

铁路车辆和基础设施的维护对减少延误、防止故障和确保铁路运输公司经济效率至关重要。预测性维护系统依赖高质量标注数据，需要以用户为中心的标注界面来满足可用性和用户体验需求。

### 目的

开发一个经济有效的预测性维护系统，结合结构噪声测量和监督学习，为农村德国的铁路车辆和基础设施提供监控和维护建议，并评估其可用性和用户体验。

### 方法

开发DigiOnTrack系统，整合无线传感器网络、分布式账本技术和Docker化容器基础设施用于托管标注界面和仪表板。由火车司机和车间主管对基础设施和车辆上的故障进行标注，以提供准确的维护建议。

### 主要发现

火车司机的界面获得'优秀可用性'评级，车间主管的界面被评为'良好'。系统有潜力集成到日常工作流程中，特别是在标注效率方面。然而，在数据密集型场景中，明晰性等领域需要进一步优化。

### 结论

研究结果为预测性维护系统和标注界面的设计提供了见解，为工业4.0应用（特别是铁路运输）的未来指南奠定了基础，展示了系统在实际工作环境中的实用性和改进空间。

### 翻译

铁路车辆和基础设施的维护在减少延误、防止故障和确保铁路运输公司的经济效率方面发挥着关键作用。由监督机器学习驱动的预测性维护系统提供了一种有前景的方法，可以在故障发生前检测故障，减少非计划停机时间，提高运营效率。然而，此类系统的成功取决于高质量的有标签数据，需要以用户为中心的标注界面，以满足标注者对可用性和用户体验的需求。本研究介绍了在联邦资助项目DigiOnTrack中开发的经济有效的预测性维护系统，该系统结合了结构噪声测量和监督学习，为德国农村地区的铁路车辆和基础设施提供监控和维护建议。该系统整合了无线传感器网络、用于安全数据传输的分布式账本技术，以及托管标注界面和仪表板的Docker化容器基础设施。火车司机和车间主管对基础设施和车辆上的故障进行标注，以确保建议的准确性。可用性和用户体验评估显示，火车司机界面获得了优秀可用性，而车间主管界面被评为良好。这些结果突显了系统整合到日常工作流程中的潜力，特别是在标注效率方面。然而，明晰性等领域需要进一步优化，以适应更密集的数据场景。研究结果为预测性维护系统和标注界面的设计提供了见解，为工业4.0应用（特别是铁路运输）的未来指南奠定了基础。


### 论文摘要

The maintenance of rail vehicles and infrastructure plays a critical role in reducing delays, preventing malfunctions, and ensuring the economic efficiency of rail transportation companies. Predictive maintenance systems powered by supervised machine learning offer a promising approach by detecting failures before they occur, reducing unscheduled downtime, and improving operational efficiency. However, the success of such systems depends on high quality labeled data, necessitating user centered labeling interfaces tailored to annotators needs for Usability and User Experience. This study introduces a cost effective predictive maintenance system developed in the federally funded project DigiOnTrack, which combines structure borne noise measurement with supervised learning to provide monitoring and maintenance recommendations for rail vehicles and infrastructure in rural Germany. The system integrates wireless sensor networks, distributed ledger technology for secure data transfer, and a dockerized container infrastructure hosting the labeling interface and dashboard. Train drivers and workshop foremen labeled faults on infrastructure and vehicles to ensure accurate recommendations. The Usability and User Experience evaluation showed that the locomotive drivers interface achieved Excellent Usability, while the workshop foremans interface was rated as Good. These results highlight the systems potential for integration into daily workflows, particularly in labeling efficiency. However, areas such as Perspicuity require further optimization for more data intensive scenarios. The findings offer insights into the design of predictive maintenance systems and labeling interfaces, providing a foundation for future guidelines in Industry 4.0 applications, particularly in rail transportation.

---

## 80. Optimizing Predictive Maintenance: Enhanced AI and Backend Integration

**论文链接:** [http://arxiv.org/abs/2511.16239v1](http://arxiv.org/abs/2511.16239v1)

**作者:** Michael Stern, Michelle Hallmann, Francesco Vona, Ute Franke, Thomas Ostertag, Benjamin Schlueter, Jan-Niklas Voigt-Antons

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究提出了一种基于传感器和机器学习的无线监控系统，用于提高铁路运输的维护效率和可靠性，特别是在资源有限的农村地区。

### 背景

铁路运输的成功依赖于高效的维护以避免延误和故障，农村地区由于资源有限面临更大挑战。

### 目的

开发一种具有成本效益的无线监控系统，整合传感器和机器学习技术来解决铁路维护中的挑战。

### 方法

开发安全数据管理系统，在车厢和铁路路段安装传感器收集结构和环境数据，实施预测性维护，建立强大的后端基础设施进行数据处理，与利益相关者合作定制系统设计。

### 主要发现

系统设计包括传感器选择、数据处理协议和机器学习模型的合理选择，以及系统架构（网络拓扑和数据处理工作流程）的提出。

### 结论

通过先进技术集成，该系统有望提高铁路运输的可靠性和效率。

### 翻译

铁路运输的成功取决于高效的维护，以避免延误和故障，特别是在资源有限的农村地区。我们提出了一种具有成本效益的无线监控系统，整合传感器和机器学习来解决这些挑战。我们开发了一个安全的数据管理系统，在车厢和铁路路段安装传感器以收集结构和环境数据。这些数据支持预测性维护，可在问题导致故障之前识别潜在问题。实施此系统需要强大的后端基础设施，用于安全的数据传输、存储和分析。我们与铁路公司和项目合作伙伴等利益相关者合作设计，定制系统以满足特定要求，同时确保数据完整性和安全。本文讨论了我们设计选择的理由，包括传感器选择、数据处理协议和机器学习模型。我们提出了一个用于实施解决方案的系统架构，涵盖了网络拓扑和数据处理工作流程等方面。我们的目标是通过先进的技术集成提高铁路运输的可靠性和效率。


### 论文摘要

Rail transportation success depends on efficient maintenance to avoid delays and malfunctions, particularly in rural areas with limited resources. We propose a cost-effective wireless monitoring system that integrates sensors and machine learning to address these challenges. We developed a secure data management system, equipping train cars and rail sections with sensors to collect structural and environmental data. This data supports Predictive Maintenance by identifying potential issues before they lead to failures. Implementing this system requires a robust backend infrastructure for secure data transfer, storage, and analysis. Designed collaboratively with stakeholders, including the railroad company and project partners, our system is tailored to meet specific requirements while ensuring data integrity and security. This article discusses the reasoning behind our design choices, including the selection of sensors, data handling protocols, and Machine Learning models. We propose a system architecture for implementing the solution, covering aspects such as network topology and data processing workflows. Our approach aims to enhance the reliability and efficiency of rail transportation through advanced technological integration.

---

## 81. Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security

**论文链接:** [http://arxiv.org/abs/2511.16229v1](http://arxiv.org/abs/2511.16229v1)

**作者:** Wei Zhao, Zhe Li, Yige Li, Jun Sun

**发布时间:** 2025-11-20

**备注:** Accepted by NDSS 2026

### GPT解析

### 总结

本文提出Q-MLLM架构，通过两级向量量化解决多模态大语言模型对视觉输入的对抗性攻击脆弱性问题，实现了高效的防御同时保持模型效用。

### 背景

多模态大语言模型在跨模态理解方面表现出色，但对视觉输入的对抗性攻击仍然脆弱，尽管文本安全机制较为健壮。

### 目的

解决多模态大语言模型对视觉输入的对抗性攻击脆弱性问题，弥合文本安全机制向视觉内容转移的差距。

### 方法

提出Q-MLLM架构，集成两级向量量化在像素块和语义级别上离散化视觉表示，阻断攻击路径并弥合跨模态安全对齐差距，采用两阶段训练方法确保稳健学习同时保持模型效用。

### 主要发现

Q-MLLM对越狱攻击和有毒图像攻击的防御成功率显著高于现有方法，对越狱攻击实现完美的防御成功率(100%)，除一个可争论的情况外，同时在多个效用基准测试上保持有竞争力的性能，且推理开销最小。

### 结论

向量量化是安全多模态AI系统的有效防御机制，不需要昂贵的特定安全微调或检测开销。

### 翻译

多模态大语言模型在跨模态理解方面展现出了令人印象深刻的能力，但尽管具有强大的文本安全机制，仍然容易受到通过视觉输入的对抗性攻击。这些脆弱性源于两个核心弱点：视觉表示的连续性，这允许基于梯度的攻击；以及文本安全机制向视觉内容的不充分转移。我们引入Q-MLLM，一种新颖的架构，它集成两级向量量化以创建对抗性攻击的离散瓶颈，同时保持多模态推理能力。通过在像素块和语义级别上离散化视觉表示，Q-MLLM阻断了攻击路径并弥合了跨模态安全对齐的差距。我们的两阶段训练方法确保了稳健学习同时保持模型效用。实验证明，Q-MLLM对越狱攻击和有毒图像攻击的防御成功率显著优于现有方法。值得注意的是，除一个可争论的情况外，Q-MLLM对越狱攻击实现了完美的防御成功率(100%)，同时在多个效用基准测试上保持有竞争力的性能，且推理开销最小。这项工作确立了向量量化作为安全多模态AI系统的有效防御机制，不需要昂贵的特定安全微调或检测开销。代码可在https://github.com/Amadeuszhao/QMLLM获取。


### 论文摘要

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM.

---

## 82. Unsupervised Image Classification with Adaptive Nearest Neighbor Selection and Cluster Ensembles

**论文链接:** [http://arxiv.org/abs/2511.16213v1](http://arxiv.org/abs/2511.16213v1)

**作者:** Melih Baydar, Emre Akbas

**发布时间:** 2025-11-20

### GPT解析

### 总结

这篇论文提出了一种名为ICCE的无监督图像分类方法，通过聚类集成策略实现了最先进的性能，特别是在ImageNet数据集上首次突破了70%的准确率。

### 背景

无监督图像分类旨在将未标记的图像分组为语义上有意义的类别。早期方法在迭代框架中整合了表示学习和聚类，而基础模型的兴起最近将焦点仅转向聚类，绕过了表示学习步骤。

### 目的

提高无监督图像聚类的性能，缩小与监督方法的性能差距。

### 方法

提出名为'Image Clustering through Cluster Ensembles'（ICCE）的方法，包含三个阶段：1）聚类阶段：在冻结的主干网络上训练多个聚类头，产生多样化的图像聚类；2）聚类集成阶段：使用聚类集成技术将可能冲突的结果统一为共识聚类；3）训练阶段：使用共识聚类结果作为伪标签训练图像分类器。

### 主要发现

ICCE在十个图像分类基准测试上取得了最先进的性能，在CIFAR10上达到99.3%的准确率，在CIFAR100上达到89%，在ImageNet数据集上达到70.4%，据作者所知是第一个在ImageNet上准确率超过70%的完全无监督图像分类方法。

### 结论

ICCE方法通过结合自适应最近邻选择和聚类集成策略，显著提高了无监督图像分类的性能，缩小了与监督方法的性能差距。

### 翻译

无监督图像分类，或图像聚类，旨在将未标记的图像分组为语义上有意义的类别。早期方法在迭代框架中整合了表示学习和聚类。然而，基础模型的兴起最近将焦点仅转向聚类，绕过了表示学习步骤。在这项工作中，我们通过引入自适应最近邻选择和聚类集成策略，基于一种最近的多头聚类方法来提高聚类性能。我们的方法'通过聚类集成的图像聚类'（ICCE），始于一个聚类阶段，我们在冻结的主干网络上训练多个聚类头，产生多样化的图像聚类。然后我们采用聚类集成技术将这些可能冲突的结果统一为共识聚类。最后，我们使用共识聚类结果作为伪标签训练图像分类器。ICCE在十个图像分类基准测试上取得了最先进的性能，在CIFAR10上达到99.3%的准确率，在CIFAR100上达到89%，在ImageNet数据集上达到70.4%，缩小了与监督方法的性能差距。据我们所知，ICCE是第一个在ImageNet上准确率超过70%的完全无监督图像分类方法。


### 论文摘要

Unsupervised image classification, or image clustering, aims to group unlabeled images into semantically meaningful categories. Early methods integrated representation learning and clustering within an iterative framework. However, the rise of foundational models have recently shifted focus solely to clustering, bypassing the representation learning step. In this work, we build upon a recent multi-head clustering approach by introducing adaptive nearest neighbor selection and cluster ensembling strategies to improve clustering performance. Our method, "Image Clustering through Cluster Ensembles" (ICCE), begins with a clustering stage, where we train multiple clustering heads on a frozen backbone, producing diverse image clusterings. We then employ a cluster ensembling technique to consolidate these potentially conflicting results into a unified consensus clustering. Finally, we train an image classifier using the consensus clustering result as pseudo-labels. ICCE achieves state-of-the-art performance on ten image classification benchmarks, achieving 99.3% accuracy on CIFAR10, 89% on CIFAR100, and 70.4% on ImageNet datasets, narrowing the performance gap with supervised methods. To the best of our knowledge, ICCE is the first fully unsupervised image classification method to exceed 70% accuracy on ImageNet.

---

## 83. Domain-Shared Learning and Gradual Alignment for Unsupervised Domain Adaptation Visible-Infrared Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2511.16184v1](http://arxiv.org/abs/2511.16184v1)

**作者:** Nianchang Huang, Yi Xu, Ruida Xi, Ruida Xi, Qiang Zhang

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究提出了一种无监督域适应可见光-红外行人再识别方法，通过两阶段模型处理域间和域内模态差异，显著提升了模型在实际应用中的性能。

### 背景

可见光-红外行人再识别在公共数据集上表现优异，但由于公共数据集与真实世界数据存在差异，现有算法在实际应用中效果不佳。

### 目的

研究无监督域适应可见光-红外行人再识别，旨在将公共数据中学到的知识转移到真实世界数据中，同时保持准确性且无需对新样本进行标注。

### 方法

分析UDA-VI-ReID中的域间模态差异和域内模态挑战；设计DSLGA两阶段模型，第一阶段采用域共享学习策略(DSLS)减轻域间模态差异导致的无效预训练，第二阶段采用渐进对齐策略(GAS)处理域内模态差异；构建CMDA-XD测试方法用于训练和测试不同UDA-VI-ReID模型。

### 主要发现

大量实验表明，该方法在各种设置下显著优于现有域适应方法，甚至优于一些监督学习方法。

### 结论

提出的DSLGA模型和CMDA-XD测试方法有效解决了UDA-VI-ReID中的挑战，在实际应用中表现出色。

### 翻译

最近，可见光-红外行人再识别(VI-ReID)在公共数据集上取得了显著性能。然而，由于公共数据集与真实世界数据之间的差异，大多数现有VI-ReID算法在实际应用中表现不佳。为解决这一问题，我们率先研究无监督域适应可见光-红外行人再识别(UDA-VI-ReID)，旨在在不降低准确性的情况下，将公共数据中学到的知识转移到真实世界数据中，且无需对新样本进行标注。具体而言，我们首先分析了UDA-VI-ReID中的两个基本挑战，即域间模态差异和域内模态差异。然后，我们设计了一个新颖的两阶段模型，即域共享学习和渐进对齐(DSLGA)，以处理这些差异。在第一阶段预训练中，DSLGA引入域共享学习策略(DSLS)，通过利用源域和目标域之间的共享信息来减轻域间模态差异导致的无效预训练。而在第二阶段微调中，DSLGA设计渐进对齐策略(GAS)，通过聚类到整体的对齐方式处理由大的域内模态差异引起的可见光和红外数据之间的跨模态对齐挑战。最后，构建了一种新的UDA-VI-ReID测试方法，即CMDA-XD，用于训练和测试不同的UDA-VI-ReID模型。大量实验证明，在各种设置下，我们的方法显著优于现有的VI-ReID域适应方法，甚至优于一些监督学习方法。


### 论文摘要

Recently, Visible-Infrared person Re-Identification (VI-ReID) has achieved remarkable performance on public datasets. However, due to the discrepancies between public datasets and real-world data, most existing VI-ReID algorithms struggle in real-life applications. To address this, we take the initiative to investigate Unsupervised Domain Adaptation Visible-Infrared person Re-Identification (UDA-VI-ReID), aiming to transfer the knowledge learned from the public data to real-world data without compromising accuracy and requiring the annotation of new samples. Specifically, we first analyze two basic challenges in UDA-VI-ReID, i.e., inter-domain modality discrepancies and intra-domain modality discrepancies. Then, we design a novel two-stage model, i.e., Domain-Shared Learning and Gradual Alignment (DSLGA), to handle these discrepancies. In the first pre-training stage, DSLGA introduces a Domain-Shared Learning Strategy (DSLS) to mitigate ineffective pre-training caused by inter-domain modality discrepancies via exploiting shared information between the source and target domains. While, in the second fine-tuning stage, DSLGA designs a Gradual Alignment Strategy (GAS) to handle the cross-modality alignment challenges between visible and infrared data caused by the large intra-domain modality discrepancies through a cluster-to-holistic alignment way. Finally, a new UDA-VI-ReID testing method i.e., CMDA-XD, is constructed for training and testing different UDA-VI-ReID models. A large amount of experiments demonstrate that our method significantly outperforms existing domain adaptation methods for VI-ReID and even some supervised methods under various settings.

---

## 84. Labels Matter More Than Models: Quantifying the Benefit of Supervised Time Series Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2511.16145v1](http://arxiv.org/abs/2511.16145v1)

**作者:** Zhijie Zhong, Zhiwen Yu, Kaixiang Yang, C. L. Philip Chen

**发布时间:** 2025-11-20

**备注:** 16 pages, 14 figures, 7 tables. Under review

### GPT解析

### 总结

该研究挑战了时间序列异常检测(TSAD)领域中追求复杂架构的趋势，表明有限的监督标签比复杂的无监督方法更有效。研究提出了STAND这一简化的监督基线方法，并在五个公共数据集上进行了实验验证。

### 背景

时间序列异常检测(TSAD)是一个重要的数据挖掘任务，但通常受限于标签稀缺。当前研究主要集中在无监督时间序列异常检测(UTAD)上，依赖复杂架构来建模正常数据分布，忽视了实际场景中有限的异常标签所能带来的性能提升。

### 目的

挑战'架构复杂性是TSAD最优路径'的前提；进行监督与无监督范式之间的首次系统性比较；引入STAND，一个简化的监督基线方法。

### 方法

提出STAND方法，这是一个简化的监督基线；在五个公共数据集上进行广泛实验；比较监督与无监督方法的性能差异。

### 主要发现

标签比模型更重要：在有限的标签预算下，简单的监督模型显著优于复杂的最先进无监督方法；监督带来更高回报：最小监督带来的性能提升远超过架构创新带来的提升；实用性：与无监督方法相比，STAND表现出更好的预测一致性和异常定位能力。

### 结论

倡导TSAD研究向数据为中心转变，强调标签利用而非纯粹的算法复杂性；代码已在GitHub上公开：https://github.com/EmorZz1G/STAND。

### 翻译

时间序列异常检测(TSAD)是一个重要的数据挖掘任务，通常受限于标签稀缺。因此，当前研究主要集中在无监督时间序列异常检测(UTAD)上，依赖复杂架构来建模正常数据分布。然而，这种方法忽视了在实际场景中有限的异常标签所能带来的显著性能提升。本文挑战了'架构复杂性是TSAD最优路径'的前提。我们进行了监督与无监督范式之间的首次系统性比较，并引入了STAND，一个简化的监督基线。在五个公共数据集上的广泛实验表明：(1)标签比模型更重要：在有限的标签预算下，简单的监督模型显著优于复杂的最先进无监督方法；(2)监督带来更高回报：最小监督带来的性能提升远超过架构创新带来的提升；(3)实用性：与无监督方法相比，STAND表现出更好的预测一致性和异常定位能力。这些发现倡导TSAD研究向数据为中心转变，强调标签利用而非纯粹的算法复杂性。代码已在https://github.com/EmorZz1G/STAND公开。


### 论文摘要

Time series anomaly detection (TSAD) is a critical data mining task often constrained by label scarcity. Consequently, current research predominantly focuses on Unsupervised Time-series Anomaly Detection (UTAD), relying on complex architectures to model normal data distributions. However, this approach often overlooks the significant performance gains available from limited anomaly labels achievable in practical scenarios. This paper challenges the premise that architectural complexity is the optimal path for TSAD. We conduct the first methodical comparison between supervised and unsupervised paradigms and introduce STAND, a streamlined supervised baseline. Extensive experiments on five public datasets demonstrate that: (1) Labels matter more than models: under a limited labeling budget, simple supervised models significantly outperform complex state-of-the-art unsupervised methods; (2) Supervision yields higher returns: the performance gain from minimal supervision far exceeds that from architectural innovations; and (3) Practicality: STAND exhibits superior prediction consistency and anomaly localization compared to unsupervised counterparts. These findings advocate for a data-centric shift in TSAD research, emphasizing label utilization over purely algorithmic complexity. The code is publicly available at https://github.com/EmorZz1G/STAND.

---

## 85. A Primer on Quantum Machine Learning

**论文链接:** [http://arxiv.org/abs/2511.15969v1](http://arxiv.org/abs/2511.15969v1)

**作者:** Su Yeon Chang, M. Cerezo

**发布时间:** 2025-11-20

**备注:** 29+16 pages, 5 figures, 15 boxes. Chapter for Comprehensive Quantum Physics. Comments welcomed!

### GPT解析

### 总结

量子机器学习(QML)是一种应用量子力学资源解决学习问题的计算范式，旨在利用量子处理器比经典模型更高效地处理各类学习任务。本文提供了QML的高级概述，重点关注量子设备作为主要学习单元的场景，并分析了该领域的张力关系和证据状况。

### 背景

量子机器学习作为一种新兴的计算范式，试图将量子力学原理应用于机器学习领域，以解决传统计算方法难以处理的问题。

### 目的

提供QML领域的高级概述，帮助读者理解量子方法在什么条件下以及基于什么假设可能提供真正的优势。

### 方法

通过概述QML在优化、监督学习、无监督学习、强化学习和生成建模等任务中的应用，分析该领域的张力关系和证据状况。

### 主要发现

1. QML领域存在实用性与保证、访问模型与加速、经典基线与量子优势之间的张力关系；2. 某些方面的证据充分，某些方面的证据有条件或缺乏；3. 仍存在开放性问题需要解决。

### 结论

通过阐明QML领域的细微差别和争议，为读者提供QML领域的友好路线图，帮助判断量子方法可能提供真正益处的条件和假设。

### 翻译

量子机器学习(QML)是一种寻求将量子力学资源应用于解决学习问题的计算范式。因此，该框架的目标是利用量子处理器比经典模型更高效地处理优化、监督学习、无监督学习和强化学习以及生成建模等任务。在本文中，我们提供了QML的高级概述，重点关注量子设备作为主要学习或数据生成单元的场景。我们概述了该领域在实用性与保证、访问模型与加速、经典基线与声称的量子优势之间的张力关系，指出了哪些地方证据充分，哪些地方证据有条件或仍然缺乏，以及哪些地方仍有开放性问题。通过阐明这些细微差别和争议，我们旨在为读者提供QML领域的友好路线图，使读者能够判断在什么条件下以及基于什么假设量子方法可能提供真正的益处。


### 论文摘要

Quantum machine learning (QML) is a computational paradigm that seeks to apply quantum-mechanical resources to solve learning problems. As such, the goal of this framework is to leverage quantum processors to tackle optimization, supervised, unsupervised and reinforcement learning, and generative modeling-among other tasks-more efficiently than classical models. Here we offer a high level overview of QML, focusing on settings where the quantum device is the primary learning or data generating unit. We outline the field's tensions between practicality and guarantees, access models and speedups, and classical baselines and claimed quantum advantages-flagging where evidence is strong, where it is conditional or still lacking, and where open questions remain. By shedding light on these nuances and debates, we aim to provide a friendly map of the QML landscape so that the reader can judge when-and under what assumptions-quantum approaches may offer real benefits.

---

## 86. InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer

**论文链接:** [http://arxiv.org/abs/2511.15967v1](http://arxiv.org/abs/2511.15967v1)

**作者:** Muyao Yuan, Yuanhong Zhang, Weizhan Zhang, Lan Ma, Yuan Gao, Jiangyong Ying, Yudeng Xin

**发布时间:** 2025-11-20

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了InfoCLIP方法，通过信息论视角将预训练CLIP的对齐知识转移到分割任务中，解决了在有限类别上微调CLIP进行语义分割时的过拟合和对齐能力下降问题。

### 背景

CLIP具有强大的泛化能力，促进了使用任意文本标记像素的开放词汇语义分割。然而，现有方法在有限类别上微调CLIP进行分割时容易过拟合，并降低预训练的视觉-语言对齐能力。

### 目的

在微调过程中稳定模态对齐，解决过拟合和对齐能力下降的问题，提高开放词汇语义分割的性能。

### 方法

提出InfoCLIP方法，基于互信息的两个新目标：(1)压缩来自预训练CLIP的像素-文本模态对齐，减少噪声；(2)最大化预训练CLIP对齐知识与微调模型之间的互信息，转移适合分割任务的紧凑局部语义关系。

### 主要发现

InfoCLIP在增强CLIP微调用于开放词汇语义分割方面有效，在各种基准测试中验证了其有效性，展示了其在不对称转移中的适应性和优越性。

### 结论

InfoCLIP能够有效解决CLIP在有限类别上微调时的过拟合问题，同时保持预训练模型的视觉-语言对齐能力，提高了开放词汇语义分割的性能。

### 翻译

最近，CLIP的强大泛化能力促进了开放词汇语义分割的发展，该方法可以使用任意文本来标记像素。然而，现有方法在有限类别上微调CLIP进行分割时往往导致过拟合并降低预训练的视觉-语言对齐能力。为了在微调过程中稳定模态对齐，我们提出了InfoCLIP，它利用信息论视角将预训练CLIP的对齐知识转移到分割任务中。具体而言，这种转移由基于互信息的两个新目标指导。首先，我们压缩来自预训练CLIP的像素-文本模态对齐，以减少其在图像-文本监督下学习到的粗粒度局部语义表示所产生的噪声。其次，我们最大化预训练CLIP的对齐知识与微调模型之间的互信息，以转移适合分割任务的紧凑局部语义关系。在各种基准上的广泛评估验证了InfoCLIP在增强CLIP微调用于开放词汇语义分割方面的有效性，展示了其在不对称转移中的适应性和优越性。


### 论文摘要

Recently, the strong generalization ability of CLIP has facilitated open-vocabulary semantic segmentation, which labels pixels using arbitrary text. However, existing methods that fine-tune CLIP for segmentation on limited seen categories often lead to overfitting and degrade the pretrained vision-language alignment. To stabilize modality alignment during fine-tuning, we propose InfoCLIP, which leverages an information-theoretic perspective to transfer alignment knowledge from pretrained CLIP to the segmentation task. Specifically, this transfer is guided by two novel objectives grounded in mutual information. First, we compress the pixel-text modality alignment from pretrained CLIP to reduce noise arising from its coarse-grained local semantic representations learned under image-text supervision. Second, we maximize the mutual information between the alignment knowledge of pretrained CLIP and the fine-tuned model to transfer compact local semantic relations suited for the segmentation task. Extensive evaluations across various benchmarks validate the effectiveness of InfoCLIP in enhancing CLIP fine-tuning for open-vocabulary semantic segmentation, demonstrating its adaptability and superiority in asymmetric transfer.

---

## 87. iLTM: Integrated Large Tabular Model

**论文链接:** [http://arxiv.org/abs/2511.15941v1](http://arxiv.org/abs/2511.15941v1)

**作者:** David Bonet, Marçal Comajoan Cara, Alvaro Calafell, Daniel Mas Montserrat, Alexander G. Ioannidis

**发布时间:** 2025-11-20

### GPT解析

### 总结

iLTM是一种集成的大型表格模型，统一了多种技术架构，在表格数据处理任务中表现出色，优于传统方法。

### 背景

表格数据支撑科学、工业和公共服务领域的决策。尽管深度学习进展迅速，但这些进步未完全应用于表格领域，梯度提升决策树(GBDTs)在实践中仍是默认选择。

### 目的

提出iLTM模型，解决表格领域深度学习的局限性，提供一种统一的表格数据处理框架。

### 方法

iLTM集成了树派生嵌入、维度无关表示、元训练的超网络、多层感知器和检索功能，在超过1,800个异构分类数据集上进行了预训练。

### 主要发现

iLTM在表格分类和回归任务上表现出一致优越的性能，从小数据集到大型高维任务均有优势；经过轻微微调后，元训练的超网络可迁移到回归目标；大量实验表明iLTM优于精心调整的GBDT和领先的深度表格模型，同时需要较少的任务特定调整。

### 结论

iLTM弥合了基于树的方法和神经方法之间的差距，为表格基础模型提供了新框架，实现了稳健、可适应和可扩展的表格学习。

### 翻译

表格数据支撑着科学、工业和公共服务领域的决策。尽管进展迅速，深度学习的进步尚未完全应用于表格领域，其中梯度提升决策树（GBDT）在实践中仍然是默认选择。我们提出了iLTM，一种集成的大型表格模型，它将树派生嵌入、维度无关表示、元训练的超网络、多层感知器（MLP）和检索功能统一在单一架构中。在超过1,800个异构分类数据集上预训练后，iLTM在表格分类和回归任务上实现了持续优越的性能，从小数据集到大型高维任务均适用。经过轻微微调后，元训练的超网络可迁移到回归目标，匹配或超越强基线。大量实验表明，iLTM优于精心调整的GBDT和领先的深度表格模型，同时需要较少的任务特定调整。通过弥合基于树和神经方法之间的差距，iLTM为表格基础模型提供了新框架，实现了稳健、可适应和可扩展的表格学习。


### 论文摘要

Tabular data underpins decisions across science, industry, and public services. Despite rapid progress, advances in deep learning have not fully carried over to the tabular domain, where gradient-boosted decision trees (GBDTs) remain a default choice in practice. We present iLTM, an integrated Large Tabular Model that unifies tree-derived embeddings, dimensionality-agnostic representations, a meta-trained hypernetwork, multilayer perceptrons (MLPs), and retrieval within a single architecture. Pretrained on more than 1,800 heterogeneous classification datasets, iLTM achieves consistently superior performance across tabular classification and regression tasks, from small datasets to large and high-dimensional tasks. After light fine-tuning, the meta-trained hypernetwork transfers to regression targets, matching or surpassing strong baselines. Extensive experiments show that iLTM outperforms well-tuned GBDTs and leading deep tabular models while requiring less task-specific tuning. By bridging the gap between tree-based and neural methods, iLTM offers a new framework for tabular foundation models for robust, adaptable, and scalable tabular learning.

---

## 88. Box6D : Zero-shot Category-level 6D Pose Estimation of Warehouse Boxes

**论文链接:** [http://arxiv.org/abs/2511.15884v1](http://arxiv.org/abs/2511.15884v1)

**作者:** Yintao Ma, Sajjad Pakdamansavoji, Amir Rasouli, Tongtong Cao

**发布时间:** 2025-11-19

### GPT解析

### 总结

Box6d是一种针对仓库环境中存储箱的类别级6D姿态估计方法，通过RGB-D观察推断箱子尺寸并使用CAD模板估计姿态，实现了高精度和低计算成本。

### 背景

准确高效的6D姿态估计对机器人操作在仓库自动化、拣选、物流和电商履行中至关重要。现有方法包括：基于模型的方法需要高分辨率网格且泛化能力差；无模型方法灵活性高但在挑战条件下表现不佳；类别级方法过于通用，忽略了环境和对象先验知识。

### 目的

提出一种专门针对仓库环境中存储箱的类别级6D姿态估计方法，解决现有方法的局限性。

### 方法

从单个RGB-D观察推断箱子尺寸使用快速二分搜索；使用类别CAD模板而非特定实例模型估计姿态；采用基于深度的合理性过滤器和早期停止策略来拒绝不合理假设，降低计算成本。

### 主要发现

在真实世界存储场景和公共基准测试中，Box6d方法提供了具有竞争力的或更优的6D姿态精度，同时将推理时间减少了约76%。

### 结论

Box6d是一种高效的类别级6D姿态估计方法，特别适用于仓库环境中的存储箱，在保持高精度的同时显著提高了计算效率。

### 翻译

准确高效的6D姿态估计对于在仓库自动化、拣选、物流和电商履行中的机器人操作至关重要。该领域有三种主要方法：基于模型的方法假设推理时有精确的CAD模型，但需要高分辨率网格，并且在新环境中泛化能力差；依赖少量参考图像或视频的无模型方法更灵活，然而在挑战条件下经常失败；类别级方法旨在平衡灵活性和准确性，但许多方法过于通用，忽略了环境和对象先验知识，限制了它们在工业环境中的实用性。为此，我们提出了Box6d，一种针对仓库环境中存储箱定制的类别级6D姿态估计方法。从单个RGB-D观察中，Box6d通过快速二分搜索推断箱子的尺寸，并使用类别CAD模板而非特定实例模型来估计姿态。使用基于深度的合理性过滤器和早期停止策略，Box6d拒绝不合理假设，降低计算成本。我们在真实世界存储场景和公共基准上进行了评估，表明我们的方法提供了具有竞争力或更优的6D姿态精度，同时将推理时间减少了约76%。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决在仓库环境中对新的（未见过的）物体进行准确高效的6D姿态估计问题。这个问题在仓库自动化、拣选、物流和电子商务履行中至关重要，因为机器人需要准确估计物体的位置和方向才能执行抓取、放置和运输等操作。现有方法要么需要高分辨率的CAD模型难以扩展到新环境，要么在杂乱和遮挡条件下表现不佳，要么过于通用忽略了工业环境的特定约束。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有三种主要方法的优缺点：基于模型的方法准确但需要实例级CAD模型，无模型方法灵活但表现不稳定，类别级别方法试图平衡两者但过于通用。针对仓库环境中箱子的特点（对称性、弱纹理、几何模糊性），作者借鉴了SAM6D的框架，但添加了维度估计模块来推断箱子尺寸，引入基于深度的一致性过滤器处理对称性问题，并采用早期停止策略降低计算成本。作者的方法结合了基于模型和无模型方法的优势，同时针对仓库场景进行了专门优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用一个类别级别的CAD模板代替实例特定的模型，通过快速二进制搜索推断箱子尺寸，使用基于深度的一致性过滤器拒绝不合理假设，并采用早期停止策略降低计算成本。整体流程包括：1)物体检测：使用SAM生成掩码提案；2)姿态估计：通过3D-3D对应关系生成姿态假设；3)深度一致性过滤：拒绝深度不一致的假设；4)维度估计：通过二进制搜索估计每个轴上的尺寸；5)早期停止：当旋转稳定后切换到比例更新，减少计算时间。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)零样本类别级别的6D姿态估计方法，专门针对仓库箱子设计；2)基于深度的一致性过滤器，处理对称性和几何模糊性；3)高效的维度估计模块，使用快速二进制搜索；4)早期停止策略，减少计算时间约76%而不牺牲精度。相比之前的工作，该方法不需要实例级CAD模型，比无模型方法更稳定，比通用类别级别方法更利用环境和物体先验，并扩展了SAM6D的功能使其能够处理类别级别估计。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Box6D提出了一种针对仓库箱子的零样本类别级别6D姿态估计方法，通过结合基于深度的一致性过滤和早期停止策略，在保持高精度的同时显著提高了计算效率。'}


### 论文摘要

Accurate and efficient 6D pose estimation of novel objects under clutter and occlusion is critical for robotic manipulation across warehouse automation, bin picking, logistics, and e-commerce fulfillment. There are three main approaches in this domain; Model-based methods assume an exact CAD model at inference but require high-resolution meshes and transfer poorly to new environments; Model-free methods that rely on a few reference images or videos are more flexible, however often fail under challenging conditions; Category-level approaches aim to balance flexibility and accuracy but many are overly general and ignore environment and object priors, limiting their practicality in industrial settings.   To this end, we propose Box6d, a category-level 6D pose estimation method tailored for storage boxes in the warehouse context. From a single RGB-D observation, Box6D infers the dimensions of the boxes via a fast binary search and estimates poses using a category CAD template rather than instance-specific models. Suing a depth-based plausibility filter and early-stopping strategy, Box6D then rejects implausible hypotheses, lowering computational cost. We conduct evaluations on real-world storage scenarios and public benchmarks, and show that our approach delivers competitive or superior 6D pose precision while reducing inference time by approximately 76%.

---

## 89. Automatic Uncertainty-Aware Synthetic Data Bootstrapping for Historical Map Segmentation

**论文链接:** [http://arxiv.org/abs/2511.15875v1](http://arxiv.org/abs/2511.15875v1)

**作者:** Lukas Arzoumanidis, Julius Knechtel, Jan-Henrik Haunert, Youness Dehbi

**发布时间:** 2025-11-19

### GPT解析

### 总结

本文提出了一种通过将原始历史地图集的制图风格转移到矢量数据上，生成大量合成历史地图的方法，解决了历史地图标注数据稀缺的问题。

### 背景

深度学习在历史文档（特别是地图）的自动分析中取得了显著进展，但大多数方法依赖于大量标注的训练数据，而历史地图（尤其是特定同质制图领域的地图）通常缺乏这类数据。

### 目的

解决历史地图数据稀缺问题，生成具有足够真实性和多样性的合成训练数据，以提高深度学习模型在历史地图分析中的性能。

### 方法

通过将原始历史地图集的制图风格转移到矢量数据上生成合成历史地图，提出自动深度生成方法和手动随机退化技术，以模拟历史地图扫描中的视觉不确定性和噪声。

### 主要发现

生成的训练数据可用于域自适应语义分割，通过自构建图卷积网络在同质地图集上评估，表明数据引导方法能有效提高模型性能。

### 结论

该方法能够有效解决历史地图数据稀缺的问题，通过生成大量高质量合成数据，提高了深度学习模型在历史地图分析中的适用性。

### 翻译

历史文档（特别是地图）的自动分析已从深度学习及其在各种计算机视觉应用中的成功中显著受益。然而，大多数基于深度学习的方法严重依赖大量标注的训练数据，而这些数据通常不可用于历史地图，尤其是属于特定同质制图领域（也称为语料库）的历史地图。创建适合机器学习的高质量训练数据通常需要大量时间和大量人工工作。虽然合成训练数据可以缓解真实样本的稀缺性，但它通常缺乏有效学习所需的亲和力（真实性）和多样性（变化性）。通过将原始历史地图集的制图风格转移到矢量数据上，我们引导生成了适合于同质历史地图集（如土地覆盖解释）的合成历史地图。我们提出了一种自动深度生成方法和一种替代的手动随机退化技术，以模拟历史地图扫描中常见的视觉不确定性和噪声，也称为数据相关不确定性。为了定量评估我们方法的有效性和适用性，生成的训练数据被用于在同质地图集上使用自构建图卷积网络进行域自适应语义分割，从而能够全面评估我们数据引导方法的影响。


### 论文摘要

The automated analysis of historical documents, particularly maps, has drastically benefited from advances in deep learning and its success across various computer vision applications. However, most deep learning-based methods heavily rely on large amounts of annotated training data, which are typically unavailable for historical maps, especially for those belonging to specific, homogeneous cartographic domains, also known as corpora. Creating high-quality training data suitable for machine learning often takes a significant amount of time and involves extensive manual effort. While synthetic training data can alleviate the scarcity of real-world samples, it often lacks the affinity (realism) and diversity (variation) necessary for effective learning. By transferring the cartographic style of an original historical map corpus onto vector data, we bootstrap an effectively unlimited number of synthetic historical maps suitable for tasks such as land-cover interpretation of a homogeneous historical map corpus. We propose an automatic deep generative approach and a alternative manual stochastic degradation technique to emulate the visual uncertainty and noise, also known as data-dependent uncertainty, commonly observed in historical map scans. To quantitatively evaluate the effectiveness and applicability of our approach, the generated training datasets were employed for domain-adaptive semantic segmentation on a homogeneous map corpus using a Self-Constructing Graph Convolutional Network, enabling a comprehensive assessment of the impact of our data bootstrapping methods.

---

## 90. Step-Audio-R1 Technical Report

**论文链接:** [http://arxiv.org/abs/2511.15848v1](http://arxiv.org/abs/2511.15848v1)

**作者:** Fei Tian, Xiangyu Tony Zhang, Yuxin Zhang, Haoyang Zhang, Yuxin Li, Daijiao Liu, Yayue Deng, Donghang Wu, Jun Chen, Liang Zhao, Chengyuan Yao, Hexin Liu, Eng Siong Chng, Xuerui Yang, Xiangyu Zhang, Daxin Jiang, Gang Yu

**发布时间:** 2025-11-19

**备注:** 15 pages, 5 figures. Technical Report

### GPT解析

### 总结

Step-Audio-R1是首个成功解锁音频领域推理能力的模型，通过多模态基础推理蒸馏框架，实现了基于声学特征的音频推理链，性能超越Gemini 2.5 Pro并接近Gemini 3 Pro。

### 背景

推理模型在文本和视觉领域通过扩展思考链取得显著成功，但音频语言模型却表现出相反现象：最少或无推理时表现更好，引发音频智能是否能从深思熟虑中受益的疑问。

### 目的

开发首个能够进行音频推理的模型，证明音频智能可以从深思熟虑中获益，并探索推理能力跨模态转移的可能性。

### 方法

提出多模态基础推理蒸馏(MGRD)框架，使模型学习生成真正基于声学特征的音频推理链，避免产生不相关的幻觉推理。

### 主要发现

Step-Audio-R1展现出强大的音频推理能力，在涵盖语音、环境声音和音乐的全面基准测试中，超越了Gemini 2.5 Pro，性能与最先进的Gemini 3 Pro相当。

### 结论

推理能力在适当锚定时可以跨模态转移，将扩展思考从负债转变为音频智能的资产，为构建跨所有感官模态深度思考的多模态推理系统开辟新途径。

### 翻译

最近推理模型的进展在文本和视觉领域通过扩展思考链的深思熟虑展示了显著成功。然而，音频语言模型中持续存在一个令人费解的现象：它们在最少或没有推理的情况下表现更好，引发了一个基本问题——音频智能真的能从深思熟虑中受益吗？我们引入了Step-Audio-R1，这是第一个成功解锁音频领域推理能力的音频推理模型。通过我们提出的多模态基础推理蒸馏(MGRD)框架，Step-Audio-R1学习生成真正基于声学特征而非产生不相关幻觉的音频相关推理链。我们的模型展现出强大的音频推理能力，在涵盖语音、环境声音和音乐的全面音频理解和推理基准测试中，超越了Gemini 2.5 Pro，并取得了与最先进的Gemini 3 Pro相当的性能。这些结果表明，推理能力在适当锚定时是可以跨模态转移的，将扩展思考从负债转变为音频智能的强大资产。通过建立第一个成功的音频推理模型，Step-Audio-R1为构建真正跨所有感官模态深度思考的多模态推理系统开辟了新途径。


### 论文摘要

Recent advances in reasoning models have demonstrated remarkable success in text and vision domains through extended chain-of-thought deliberation. However, a perplexing phenomenon persists in audio language models: they consistently perform better with minimal or no reasoning, raising a fundamental question - can audio intelligence truly benefit from deliberate thinking? We introduce Step-Audio-R1, the first audio reasoning model that successfully unlocks reasoning capabilities in the audio domain. Through our proposed Modality-Grounded Reasoning Distillation (MGRD) framework, Step-Audio-R1 learns to generate audio-relevant reasoning chains that genuinely ground themselves in acoustic features rather than hallucinating disconnected deliberations. Our model exhibits strong audio reasoning capabilities, surpassing Gemini 2.5 Pro and achieving performance comparable to the state-of-the-art Gemini 3 Pro across comprehensive audio understanding and reasoning benchmarks spanning speech, environmental sounds, and music. These results demonstrate that reasoning is a transferable capability across modalities when appropriately anchored, transforming extended deliberation from a liability into a powerful asset for audio intelligence. By establishing the first successful audio reasoning model, Step-Audio-R1 opens new pathways toward building truly multimodal reasoning systems that think deeply across all sensory modalities.

---

## 91. Flow-Aided Flight Through Dynamic Clutters From Point To Motion

**论文链接:** [http://arxiv.org/abs/2511.16372v1](http://arxiv.org/abs/2511.16372v1)

**作者:** Bowen Xu, Zexuan Yan, Minghao Lu, Xiyu Fan, Yi Luo, Youshen Lin, Zhiqiang Chen, Yeke Chen, Qiyuan Qiao, Peng Lu

**发布时间:** 2025-11-20

**备注:** Accepted to IEEE Robotics and Automation Letters (RA-L), November, 2025

### GPT解析

### 总结

该研究提出了一种基于强化学习的动态障碍物穿越方法，通过单线激光雷达感知和变化感知表示实现自主飞行，无需复杂的物体检测、跟踪和预测过程。

### 背景

动态障碍物穿越的主要挑战在于有效感知环境动态并生成规避行为。现有方法通过明确建模动态障碍物运动来避免障碍，但在高度动态且存在遮挡的场景中既耗时又不可靠。

### 目的

开发一种无需物体检测、跟踪和预测的系统，仅使用单线激光雷达感知，实现从原始点到运动决策的直接自主飞行。

### 方法

从原始点云编码固定形状、低分辨率的深度感知距离图；采用环境变化感知点流作为运动特征；将两者整合为轻量级环境表示；通过变化感知表示隐式驱动规避行为；使用相对调制的距离场进行策略优化；采用部署友好的感知模拟和无动力学模型的加速控制。

### 主要发现

所提系统相比替代方案具有更高的成功率和适应性；从模拟器导出的策略可有效驱动现实世界中的四旋翼无人机进行安全机动。

### 结论

通过简化感知过程并利用强化学习，实现了在动态环境中的有效导航，为动态障碍物穿越提供了新思路。

### 翻译

穿越动态障碍物的主要挑战在于有效感知环境动态并生成考虑障碍物运动的规避行为。之前的解决方案在明确建模动态障碍物运动以避免碰撞方面取得了进展，但在高度动态且存在遮挡的场景中，这种决策依赖既耗时又不可靠。相反，无需引入物体检测、跟踪和预测，我们通过单线激光雷达感知使强化学习能够直接从点到运动实现自主飞行系统。对于外部感知，从原始点云编码成固定形状、低分辨率、细节安全的深度感知距离图，并采用环境变化感知点流作为从多帧观测中提取的运动特征。这两者被整合为复杂动态环境的轻量级且易于学习的表示。对于动作生成，规避动态威胁的行为由提出的变化感知表示隐式驱动，其中策略优化由相对调制的距离场指示。通过部署友好的感知模拟和无动力学模型的加速控制，所提出的系统相比替代方案表现出更高的成功率和适应性，且从模拟器导出的策略可以驱动现实世界中的四旋翼无人机进行安全机动。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决无人机在动态杂乱环境中高效飞行的问题，具体是如何高效感知环境动态变化并考虑障碍物运动生成规避行为。这个问题在现实世界中非常重要，因为无人机需要在复杂动态环境中安全导航，如城市配送、搜索救援、监控等应用场景，而传统方法在高度动态且存在遮挡的环境中既耗时又不可靠。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，指出传统方法需要显式建模动态障碍物运动，在高度动态和遮挡场景中不可靠且计算量大。作者设计了一个不引入对象检测、跟踪和预测的系统，而是使用强化学习直接从原始点云到运动控制。他们借鉴了强化学习技术（PPO算法）、深度地图编码思想以及预训练的光流估计器（NeuFlowV2），同时创新性地设计了固定形状低分辨率但保留细节安全的深度感知距离图，以及环境变化感知的点流作为运动特征。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是不需要显式检测、跟踪和预测动态障碍物，而是结合深度感知（距离图）和环境变化感知（点流）来理解动态环境，使用强化学习直接从原始点云生成规避行为。整体流程包括：1)从原始点云编码低分辨率距离图并提取环境变化点流；2)将距离图和点流结合形成环境表示；3)结合指向目标的方向、速度和最后动作形成完整状态；4)通过CNN编码器和MLP融合模块决策输出加速度命令；5)使用PPO算法和包含多种奖励的函数进行训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)结合深度感知和环境变化感知的高效激光雷达表示；2)通过隐式环境运动特征学习提前规避行为的通用强化学习训练；3)无需对象检测、跟踪和预测的端到端系统。相比之前工作，这种方法避免了显式区分动态对象和环境的计算负担，减少了模块间复合误差，在高度动态和遮挡场景中更可靠；同时整合了多帧感知并使用预训练光流估计器，提供了低噪声且与环境变化高度相关的表示。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种结合激光雷达深度感知和环境变化感知的强化学习方法，实现了无人机在动态杂乱环境中无需显式对象检测和预测的安全自主飞行。'}


### 论文摘要

Challenges in traversing dynamic clutters lie mainly in the efficient perception of the environmental dynamics and the generation of evasive behaviors considering obstacle movement. Previous solutions have made progress in explicitly modeling the dynamic obstacle motion for avoidance, but this key dependency of decision-making is time-consuming and unreliable in highly dynamic scenarios with occlusions. On the contrary, without introducing object detection, tracking, and prediction, we empower the reinforcement learning (RL) with single LiDAR sensing to realize an autonomous flight system directly from point to motion. For exteroception, a depth sensing distance map achieving fixed-shape, low-resolution, and detail-safe is encoded from raw point clouds, and an environment change sensing point flow is adopted as motion features extracted from multi-frame observations. These two are integrated into a lightweight and easy-to-learn representation of complex dynamic environments. For action generation, the behavior of avoiding dynamic threats in advance is implicitly driven by the proposed change-aware sensing representation, where the policy optimization is indicated by the relative motion modulated distance field. With the deployment-friendly sensing simulation and dynamics model-free acceleration control, the proposed system shows a superior success rate and adaptability to alternatives, and the policy derived from the simulator can drive a real-world quadrotor with safe maneuvers.

---

## 92. A Spatial Semantics and Continuity Perception Attention for Remote Sensing Water Body Change Detection

**论文链接:** [http://arxiv.org/abs/2511.16143v1](http://arxiv.org/abs/2511.16143v1)

**作者:** Quanqing Ma, Jiaen Chen, Peng Wang, Yao Zheng, Qingzhan Zhao, Yuchen Zheng

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究提出了一种高空间分辨率水体变化检测数据集HSRW-CD和一种空间语义与连续性感知(SSCP)注意力模块，解决了WBCD中数据稀缺和特征利用不充分的问题，显著提高了水体检测的准确性。

### 背景

水体变化检测(WBCD)旨在从同一地理区域的双时相图像中检测水体表面的变化。目前缺乏高空间分辨率数据集限制了WBCD在城市和农村地区的应用，而这些地区需要更精确的定位。之前的基于深度学习方法未能充分利用变化检测网络中深层次特征的空间语义和结构信息。

### 目的

解决高空间分辨率数据集稀缺的问题，以及深度学习方法未能充分利用空间语义和结构信息的问题。

### 方法

提出了一种名为HSRW-CD的新数据集，空间分辨率高于3米，包含大量图像对，广泛覆盖各种水体类型；设计了一种空间语义和连续性感知(SSCP)注意力模块，包含三个组件：多语义空间注意力(MSA)、结构关系感知全局注意力(SRGA)和通道自注意力(CSA)。SSCP作为即插即用模块，可集成到现有的WBCD模型中。

### 主要发现

提出的HSRW-CD数据集为WBCD提供了高空间分辨率的数据支持；SSCP注意力模块能够有效利用深层次特征的空间语义和结构信息；SSCP模块提高了水体检测的辨别能力；在HSRW-CD和Water-CD数据集上的大量实验验证了SSCP的有效性和泛化能力。

### 结论

提出的HSRW-CD数据集和SSCP注意力模块有效解决了WBCD中的关键问题；SSCP模块可以作为即插即用组件集成到现有模型中；代码和数据集将在https://github.com/QingMa1/SSCP上公开。

### 翻译

遥感水体变化检测(WBCD)旨在从同一地理区域的双时相图像中检测水体表面的变化。目前，高空间分辨率数据集的稀缺限制了WBCD在城市和农村地区的应用，而这些地区需要更精确的定位。同时，之前的基于深度学习方法未能充分利用变化检测网络中深层次特征的空间语义和结构信息。为解决这些问题，我们首先提出了一个名为HSRW-CD的新数据集，其空间分辨率高于3米，专门用于WBCD。具体而言，它包含大量图像对，广泛覆盖各种水体类型。此外，我们还设计了一种空间语义和连续性感知(SSCP)注意力模块，充分利用WBCD网络中深层次特征的空间语义和结构，显著提高了水体的辨别能力。所提出的SSCP包含三个组件：多语义空间注意力(MSA)、结构关系感知全局注意力(SRGA)和通道自注意力(CSA)。MSA增强水体特征的空间语义，并为CSA提供精确的空间语义先验。然后，SRGA进一步提取空间结构，学习水体的空间连续性。最后，CSA利用MSA和SRGA提供的空间语义和结构先验计算通道间的相似性。作为专为水体深层次特征设计的即插即用模块，所提出的SSCP可以集成到现有的WBCD模型中。在提出的HSRW-CD和Water-CD数据集上进行的大量实验验证了SSCP的有效性和泛化能力。本工作的代码和HSRW-CD数据集可在https://github.com/QingMa1/SSCP获取。


### 论文摘要

Remote sensing Water Body Change Detection (WBCD) aims to detect water body surface changes from bi-temporal images of the same geographic area. Recently, the scarcity of high spatial resolution datasets for WBCD restricts its application in urban and rural regions, which require more accurate positioning. Meanwhile, previous deep learning-based methods fail to comprehensively exploit the spatial semantic and structural information in deep features in the change detection networks. To resolve these concerns, we first propose a new dataset, HSRW-CD, with a spatial resolution higher than 3 meters for WBCD. Specifically, it contains a large number of image pairs, widely covering various water body types. Besides, a Spatial Semantics and Continuity Perception (SSCP) attention module is designed to fully leverage both the spatial semantics and structure of deep features in the WBCD networks, significantly improving the discrimination capability for water body. The proposed SSCP has three components: the Multi-Semantic spatial Attention (MSA), the Structural Relation-aware Global Attention (SRGA), and the Channel-wise Self-Attention (CSA). The MSA enhances the spatial semantics of water body features and provides precise spatial semantic priors for the CSA. Then, the SRGA further extracts spatial structure to learn the spatial continuity of the water body. Finally, the CSA utilizes the spatial semantic and structural priors from the MSA and SRGA to compute the similarity across channels. Specifically designed as a plug-and-play module for water body deep features, the proposed SSCP allows integration into existing WBCD models. Numerous experiments conducted on the proposed HSRW-CD and Water-CD datasets validate the effectiveness and generalization of the SSCP. The code of this work and the HSRW-CD dataset will be accessed at https://github.com/QingMa1/SSCP.

---

## 93. AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture

**论文链接:** [http://arxiv.org/abs/2511.15870v1](http://arxiv.org/abs/2511.15870v1)

**作者:** Qiming Guo, Bishal Khatri, Wenbo Sun, Jinwen Tang, Hua Zhang, Wenlu Wang

**发布时间:** 2025-11-19

**备注:** 7 pages, 1 figure, 2 tables, Accepted to the 40th AAAI Conference on Artificial Intelligence (AAAI 2026), IAAI Deployed Applications Track

### GPT解析

### 总结

本文提出AquaSentinel，一种用于城市地下供水管网实时异常检测的新型物理信息AI系统，通过四个关键创新实现高效、准确的泄漏检测和定位。

### 背景

地下管道泄漏和渗漏对水资源安全和环境安全构成重大威胁。传统人工检查方法覆盖范围有限且响应延迟，常常错过关键异常。

### 目的

开发一种创新的AI系统，实现城市地下供水管网中的实时异常检测，提高检测精度并降低成本。

### 方法

系统包含四个关键创新：(1)在高中心度节点进行战略性稀疏传感器部署，结合基于物理的状态增强；(2)RTCA检测算法，采用双阈值监测和自适应统计；(3)专家混合时空图神经网络集成；(4)基于因果流的泄漏定位方法。系统通过物理建模将测量传播到未监测节点，创建虚拟传感器增强网络数据可用性。

### 主要发现

实验评估使用110种泄漏场景，AquaSentinel实现了100%的检测精度。物理信息稀疏传感可以在成本远低于密集部署的情况下匹配其性能。

### 结论

该系统为老化城市基础设施提供了实用的管道监测解决方案，证明稀疏传感策略能够以较低成本实现与密集部署相当的检测性能。

### 翻译

地下管道泄漏和渗漏对水资源安全和环境安全构成重大威胁。传统的人工检查方法覆盖范围有限且响应延迟，常常错过关键异常。本文提出了AquaSentinel，一种用于城市地下供水管网实时异常检测的新型物理信息AI系统。我们介绍了四个关键创新：(1)在高中心度节点进行战略性稀疏传感器部署，结合基于物理的状态增强，以实现从最少基础设施的网络级可观测性；(2)RTCA（实时累积异常）检测算法，采用双阈值监测和自适应统计来区分瞬时波动与真实异常；(3)专家混合（MoE）时空图神经网络集成，通过动态加权模型贡献提供稳健预测；(4)基于因果流的泄漏定位，向上游追踪异常以识别源节点和受影响的管道段。我们的系统在关键网络节点战略性地部署传感器，并利用基于物理的建模将测量传播到未监测的节点，创建虚拟传感器，增强整个网络的数据可用性。使用110种泄漏场景进行的实验评估表明，AquaSentinel实现了100%的检测精度。这项工作通过证明物理信息稀疏传感可以在成本远低于密集部署的情况下匹配其性能，推进了管道监测，为老化城市基础设施提供了实用解决方案。


### 论文摘要

Underground pipeline leaks and infiltrations pose significant threats to water security and environmental safety. Traditional manual inspection methods provide limited coverage and delayed response, often missing critical anomalies. This paper proposes AquaSentinel, a novel physics-informed AI system for real-time anomaly detection in urban underground water pipeline networks. We introduce four key innovations: (1) strategic sparse sensor deployment at high-centrality nodes combined with physics-based state augmentation to achieve network-wide observability from minimal infrastructure; (2) the RTCA (Real-Time Cumulative Anomaly) detection algorithm, which employs dual-threshold monitoring with adaptive statistics to distinguish transient fluctuations from genuine anomalies; (3) a Mixture of Experts (MoE) ensemble of spatiotemporal graph neural networks that provides robust predictions by dynamically weighting model contributions; (4) causal flow-based leak localization that traces anomalies upstream to identify source nodes and affected pipe segments. Our system strategically deploys sensors at critical network junctions and leverages physics-based modeling to propagate measurements to unmonitored nodes, creating virtual sensors that enhance data availability across the entire network. Experimental evaluation using 110 leak scenarios demonstrates that AquaSentinel achieves 100% detection accuracy. This work advances pipeline monitoring by demonstrating that physics-informed sparse sensing can match the performance of dense deployments at a fraction of the cost, providing a practical solution for aging urban infrastructure.

---

## 94. Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion

**论文链接:** [http://arxiv.org/abs/2511.16161v1](http://arxiv.org/abs/2511.16161v1)

**作者:** Lirui Zhang, Zhengkai Zhao, Zhi Zuo, Pan Gao, Jie Qin

**发布时间:** 2025-11-20

**备注:** Accepted for publication at the 40th AAAI Conference on Artificial Intelligence (AAAI-26)

### GPT解析

### 总结

Simba是一种新颖的点云补全框架，通过将点状变换回归转化为分布学习问题，结合对称先验和扩散模型，解决了现有方法在细节保留和全局结构完整性方面的挑战，并在多个基准测试上实现了最先进的性能。

### 背景

点云补全是3D视觉中的基础任务。该领域的一个持续挑战是同时保留输入中的细粒度细节并确保完成形状的全局结构完整性。

### 目的

解决现有基于回归方法的两个主要限制：(1)容易过拟合，倾向于记忆特定实例的变换而不是学习可泛化的几何先验；(2)对点状变换回归的依赖导致对输入噪声高度敏感，严重降低了鲁棒性和泛化能力。

### 方法

提出名为Simba的新框架，将点状变换重新表述为分布学习问题，结合对称先验和扩散模型的强大生成能力，并引入基于Mamba的分层架构实现高保真上采样。

### 主要发现

在PCN、ShapeNet和KITTI基准测试上的广泛实验验证了该方法的最先进(SOTA)性能。

### 结论

Simba方法避免了实例特定的记忆，同时捕获了鲁棒的几何结构，有效解决了点云补全任务中的细节保留和全局结构完整性挑战。

### 翻译

点云补全是3D视觉中的一个基础任务。该领域的一个持续挑战是同时保留输入中的细粒度细节并确保完成形状的全局结构完整性。虽然最近利用局部对称变换通过直接回归的工作显著提高了几何结构细节的保留，但这些方法存在两个主要限制：(1)这些基于回归的方法容易过拟合，倾向于记忆特定实例的变换而不是学习可泛化的几何先验。(2)它们对点状变换回归的依赖导致对输入噪声高度敏感，严重降低了它们的鲁棒性和泛化能力。为了应对这些挑战，我们引入了Simba，一个新颖的框架，将点状变换重新表述为分布学习问题。我们的方法将对称先验与扩散模型的强大生成能力相结合，避免了实例特定的记忆，同时捕获了鲁棒的几何结构。此外，我们引入了一种基于Mamba的分层架构来实现高保真上采样。在PCN、ShapeNet和KITTI基准测试上的广泛实验验证了我们方法的最先进(SOTA)性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决点云补全中同时保留细粒度细节和确保全局结构完整性的挑战。这个问题在现实中非常重要，因为点云是自动驾驶、机器人和增强现实等领域的基础3D表示，而现实环境中的点云常因遮挡和传感器限制而不完整，影响后续应用效果。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者借鉴了现有工作中的对称性先验和扩散模型思想。他们观察到早期对称性方法受限于全局对称假设，而SymmCompletion等方法虽学习局部对称变换但存在过拟合和噪声敏感问题。作者创新性地将扩散模型与几何变换结合，设计了两阶段框架：第一阶段预训练SymmGT生成目标变换场，第二阶段使用Symmetry-Diffusion扩散模型学习变换分布，并引入基于Mamba的高效细化网络。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将点对点变换回归重新表述为分布学习问题，利用扩散模型学习几何变换的分布而非固定值，避免过拟合并提高鲁棒性。整体流程分为两阶段：第一阶段预训练SymmGT网络从部分输入和完整真实点云生成目标变换场；第二阶段使用Sym-Diffuser扩散模型生成变换场构建粗略补全，再通过MBA-Refiner级联网络逐步细化和上采样为高保真输出。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次将点云补全表述为几何变换场的条件生成任务；2) 创新性地采用扩散模型学习仿射变换分布，确保几何一致性；3) 设计基于Mamba的细化网络实现高效高保真上采样。相比之前工作，Simba不再直接回归变换矩阵，而是学习其分布，提高了鲁棒性和泛化能力，尤其在真实世界数据上表现出强大的跨域适应能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Simba通过将点云补全重新表述为几何变换的扩散学习问题，结合对称性先验与扩散模型的生成能力，实现了高保真且几何一致性的点云补全，在多个基准测试上达到最先进性能并展现出强大的跨域泛化能力。'}


### 论文摘要

Point cloud completion is a fundamental task in 3D vision. A persistent challenge in this field is simultaneously preserving fine-grained details present in the input while ensuring the global structural integrity of the completed shape. While recent works leveraging local symmetry transformations via direct regression have significantly improved the preservation of geometric structure details, these methods suffer from two major limitations: (1) These regression-based methods are prone to overfitting which tend to memorize instant-specific transformations instead of learning a generalizable geometric prior. (2) Their reliance on point-wise transformation regression lead to high sensitivity to input noise, severely degrading their robustness and generalization. To address these challenges, we introduce Simba, a novel framework that reformulates point-wise transformation regression as a distribution learning problem. Our approach integrates symmetry priors with the powerful generative capabilities of diffusion models, avoiding instance-specific memorization while capturing robust geometric structures. Additionally, we introduce a hierarchical Mamba-based architecture to achieve high-fidelity upsampling. Extensive experiments across the PCN, ShapeNet, and KITTI benchmarks validate our method's state-of-the-art (SOTA) performance.

---

## 95. The MeerKAT Fornax Survey VI. The collapse of the galaxy HI Mass Function in Fornax

**论文链接:** [http://arxiv.org/abs/2511.15795v1](http://arxiv.org/abs/2511.15795v1)

**作者:** D. Kleiner, P. Serra, A. Loni, S. H. A. Rajohnson, F. M. Maccagni, W. J. G. de Blok, P. Kamphuis, R. C. Kraan-Korteweg, M. A. W. Verheijen

**发布时间:** 2025-11-19

**备注:** Accepted in Astronomy & Astrophysics. 12 pages, 6 figures

### GPT解析

### 总结

研究团队测量了本星系群之外最深度的HI质量函数，发现了HI质量函数在低质量端的坍缩现象，表明在特定质量范围内缺乏星系。

### 背景

研究使用了MeerKAT Fornax Survey的数据，覆盖了4x4平方度的区域，对应于约Rvir范围，探测到了35个有光学对应体的星系和44个无光学对应体的HI云。

### 目的

测量HI质量函数并探测低质量HI天体，研究星系群环境中的HI分布特征和演化规律。

### 方法

使用MeerKAT望远镜进行观测，结合深度光学图像分析，通过信噪比分析和Rauzy统计确保数据完整性，并使用修改的最大似然方法拟合Schechter函数。

### 主要发现

在log(MHI/Msun) = 7处HI探测星系的数密度急剧下降，6 < log(MHI/Msun) < 7范围内几乎没有星系；低质量斜率为α = -1.31 ± 0.13，与场星系相似；大多数无光学对应体的HI云与NGC 1365和NGC 1427A两个星系相关；HI质量函数在低质量端出现坍缩，需要比观测到的多约6倍的星系才能维持幂律分布。

### 结论

Fornax HIMF的坍缩很可能是由于低质量星系中HI的快速去除所致，这反映了星系群环境中气体动力学过程对低质量星系演化的影响。

### 翻译

我们呈现了有史以来测量到的本星系群之外最深的HI质量函数。观测是MeerKAT Fornax Survey的一部分，覆盖了4x4平方度的区域，对应于约Rvir。对于50 km/s宽的点源，3σ探测限为log(MHI/Msun) = 5.7。我们在35个星系和44个没有光学对应体的云中探测到HI。使用Fornax Deep Survey的深度光学图像，我们表明这些云是一个不同的种群，与最暗的HI探测星系之间存在四个星等的差距。大多数（44个中的33个）云与星系团中HI最多的两个星系——NGC 1365和NGC 1427A相关，尽管云对总MHI预算的贡献可忽略不计。通过对HI探测进行信噪比分析和计算Rauzy统计，我们证明了我们的目录在log(MHI/Msun) = 6以下完整，因此我们能够探测到这一水平的HI质量函数。我们发现HI探测星系的数密度在log(MHI/Msun) = 7处急剧下降，表明在6 < log(MHI/Msun) < 7范围内明显缺乏星系。我们使用修改的最大似然方法将Schechter函数拟合到log(MHI/Msun) > 7的范围，即HI质量函数遵循幂律的范围。测量的低质量斜率为α = -1.31 ± 0.13，特征膝点质量为log(M*/Msun) = 10.52 ± 1.89。低质量斜率与场中的斜率相匹配，而膝点由单个星系定义且未受约束。在log(MHI/Msun) = 7以下，与Schechter函数有显著偏离，我们报告了首次对HI质量函数坍缩的稳健测量。为了使log(MHI/Msun) = 7以下的HI质量函数遵循幂律，需要数十个星系——比观测到的多约6倍。Fornax HI质量函数的坍缩可能是由于低质量星系中HI的快速去除。


### 论文摘要

We present the deepest HI mass Function (HIMF) ever measured, outside the Local Group. The observations are part of the MeerKAT Fornax Survey and cover a 4 x 4 deg^2 field, corresponding to ~ Rvir. The 3$σ$ detection limit is log(MHI/Msun) = 5.7 for a 50 km/s-wide point source. We detect HI in 35 galaxies and 44 clouds with no optical counterparts. Using deep optical images from the Fornax Deep Survey, we show that the clouds are a distinct population, separated by a four magnitude gap from the faintest HI-detected galaxies. The majority (33 out of 44) of the clouds are associated with the two galaxies with the most HI in the cluster -- NGC 1365 and NGC 1427A, although the clouds contribute a negligible amount to the total MHI budget. By performing a SNR analysis and computing the Rauzy statistic on the HI detections, we demonstrate that our catalogue is complete down log(MHI/Msun) = 6, and we are therefore able to probe the HIMF down to this level. We find an abrupt drop of the number density of HI-detected galaxies at log(MHI/Msun) = 7, signifying a clear absence of galaxies between 6 < log(MHI/Msun) < 7. We use the modified maximum likelihood method to fit a Schechter function down to log(MHI/Msun) > 7, the range where the HIMF follows a power-law. The measured low-mass slope is $α$ = -1.31 $\pm$ 0.13, with a characteristic knee mass of log(M*/Msun) = 10.52 $\pm$ 1.89. The low-mass slope matches the slope in the field, while the knee is defined by a single galaxy and is unconstrained. Below log(MHI/Msun) = 7, there is a sharp departure from a Schechter function, and we report the first robust measurement of the collapse of a HIMF. For the HIMF below log(MHI/Msun) = 7 to follow a power-law, tens of galaxies are needed -- a factor ~ six higher than what is observed. The collapse of the Fornax HIMF is likely due to the rapid removal of HI from low-mass galaxies.

---

## 96. CRISTAL: Real-time Camera Registration in Static LiDAR Scans using Neural Rendering

**论文链接:** [http://arxiv.org/abs/2511.16349v1](http://arxiv.org/abs/2511.16349v1)

**作者:** Joni Vanherck, Steven Moonen, Brent Zoomers, Kobe Werner, Jeroen Put, Lode Jorissen, Nick Michiels

**发布时间:** 2025-11-20

### GPT解析

### 总结

该论文提出了一种基于高精度彩色LiDAR点云的实时相机定位方法，通过合成视图渲染和神经渲染技术实现无漂移、正确尺度的相机跟踪，解决了传统视觉方法的漂移和尺度模糊问题。

### 背景

准确的相机定位对机器人和扩展现实(XR)至关重要，但现有视觉方法常受漂移、尺度模糊问题困扰，且依赖标记或回环闭合。

### 目的

开发一种在预先捕获的高精度彩色LiDAR点云中实时定位相机的方法，避免传统视觉方法的局限性。

### 方法

从LiDAR点云渲染合成视图建立2D-3D对应关系，应用神经渲染技术缩小合成与真实图像的域差距，减少遮挡和背景伪影提高特征匹配，提出在线渲染匹配和预构建定位两种实时变体。

### 主要发现

方法实现了无漂移的相机跟踪，获得正确的全局LiDAR坐标系度量比例，在ScanNet++数据集上优于现有SLAM管道。

### 结论

结合LiDAR点云、合成视图渲染和神经渲染技术，为机器人和XR应用提供了更可靠的相机定位解决方案。

### 翻译

准确的相机定位对于机器人和扩展现实(XR)至关重要，它能够实现可靠的导航以及虚拟和现实内容的对齐。现有的视觉方法通常存在漂移、尺度模糊问题，并且依赖于标记或回环闭合。这项工作介绍了一种在预先捕获的高精度彩色LiDAR点云中实时定位相机的方法。通过从该点云渲染合成视图，建立了实时帧与点云之间的2D-3D对应关系。神经渲染技术缩小了合成图像与真实图像之间的域差距，减少遮挡和背景伪影，从而改善特征匹配。结果是在全局LiDAR坐标系中实现了无漂移的相机跟踪和正确的度量比例。文中提出了两种实时变体：在线渲染与匹配，以及预构建与定位。我们在ScanNet++数据集上展示了改进的结果，并优于现有的SLAM管道。


### 论文摘要

Accurate camera localization is crucial for robotics and Extended Reality (XR), enabling reliable navigation and alignment of virtual and real content. Existing visual methods often suffer from drift, scale ambiguity, and depend on fiducials or loop closure. This work introduces a real-time method for localizing a camera within a pre-captured, highly accurate colored LiDAR point cloud. By rendering synthetic views from this cloud, 2D-3D correspondences are established between live frames and the point cloud. A neural rendering technique narrows the domain gap between synthetic and real images, reducing occlusion and background artifacts to improve feature matching. The result is drift-free camera tracking with correct metric scale in the global LiDAR coordinate system. Two real-time variants are presented: Online Render and Match, and Prebuild and Localize. We demonstrate improved results on the ScanNet++ dataset and outperform existing SLAM pipelines.

---

## 97. SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation

**论文链接:** [http://arxiv.org/abs/2511.16666v1](http://arxiv.org/abs/2511.16666v1)

**作者:** Zhenyuan Qin, Xincheng Shuai, Henghui Ding

**发布时间:** 2025-11-20

**备注:** NeurIPS 2025 (Spotlight), Project Page: https://henghuiding.com/SceneDesigner/

### GPT解析

### 总结

SceneDesigner是一种用于精确和灵活的多对象9D姿态操控的方法，通过引入CNOCS map表示和两阶段训练策略，解决了现有方法在多对象9D姿态控制中的可控性和质量问题。

### 背景

可控图像生成近年来受到越来越多的关注，使用户能够操控视觉内容如身份和风格。然而，同时控制多个对象的9D姿态（位置、大小和方向）仍然是一个开放的挑战，现有方法通常存在可控性有限和质量下降的问题。

### 目的

开发一种能够准确和灵活地操控多对象9D姿态的方法，以解决现有方法在全面多对象9D姿态控制方面的不足。

### 方法

SceneDesigner将分支网络集成到预训练基础模型中，并利用CNOCS map这一新的表示方法，从相机视角编码9D姿态信息。研究者构建了新数据集ObjectPose9D，并采用两阶段训练策略结合强化学习来处理数据不平衡问题。在推理时，提出了解耦对象采样技术，并支持用户特定的个性化权重。

### 主要发现

CNOCS map表示具有强大的几何解释特性，使训练更加高效和稳定。两阶段训练策略结合强化学习有效解决了数据不平衡问题，特别是在低频姿态上的性能下降。解耦对象采样技术减轻了在复杂多对象场景中对象生成不足和概念混淆的问题。

### 结论

SceneDesigner在可控性和质量方面显著优于现有方法，为多对象9D姿态控制提供了有效的解决方案。

### 翻译

可控图像生成近年来吸引了越来越多的关注，使用户能够操控视觉内容如身份和风格。然而，同时控制多个对象的9D姿态（位置、大小和方向）仍然是一个开放的挑战。尽管最近取得了进展，但现有方法通常存在可控性有限和质量下降的问题，无法实现全面的多对象9D姿态控制。为解决这些限制，我们提出了SceneDesigner，一种用于精确和灵活的多对象9D姿态操控的方法。SceneDesigner将分支网络集成到预训练基础模型中，并利用一种新的表示方法CNOCS map，该方法从相机视角编码9D姿态信息。这种表示具有强大的几何解释特性，使训练更加高效和稳定。为支持训练，我们构建了一个新数据集ObjectPose9D，该数据集聚合了来自不同来源的图像和9D姿态标注。为进一步解决数据不平衡问题，特别是在低频姿态上的性能下降，我们引入了一种结合强化学习的两阶段训练策略，其中第二阶段在重新平衡的数据上使用基于奖励的目标对模型进行微调。在推理时，我们提出了解耦对象采样技术，减轻了在复杂多对象场景中对象生成不足和概念混淆的问题。此外，通过集成用户特定的个性化权重，SceneDesigner能够为参考对象实现定制化的姿态控制。大量的定性和定量实验表明，SceneDesigner在可控性和质量方面显著优于现有方法。代码已在https://github.com/FudanCVL/SceneDesigner公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在图像生成中实现对多个物体的9D位姿（位置、大小和方向）同时精确控制的问题。这个问题在现实中非常重要，例如设计师需要安排房间中不同大小和方向的家具，或用户希望生成宠物狗背对相机望向风景的图像等场景。当前图像生成模型缺乏这种细粒度的3D空间控制能力，限制了其在虚拟现实、产品设计等领域的应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：LOOSECONTROL缺乏方向控制，Zero-1-to-3等方法在复杂场景中控制能力有限，ORIGEN兼容性差。他们借鉴了ControlNet-like架构、NOCS表示方法和RLHF强化学习技术，并通过创新设计解决了这些问题：提出CNOCS地图简化NOCS方法，构建ObjectPose9D数据集，采用两阶段训练策略，以及设计解耦物体采样技术。这些创新使模型能够实现多物体9D位姿的精确控制。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用CNOCS地图编码物体9D位姿信息，通过分支网络集成到预训练模型中，采用两阶段训练解决数据不平衡问题，并使用解耦物体采样技术确保多物体场景中的正确对应。整体流程包括：1)构建ObjectPose9D数据集；2)创建CNOCS地图表示位姿；3)两阶段训练（第一阶段学习基本控制，第二阶段强化学习改善低频位姿）；4)使用解耦物体采样生成图像，支持个性化控制。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)CNOCS地图表示，简化NOCS方法使用立方体形状；2)ObjectPose9D数据集，整合多种来源图像和9D位姿标注；3)两阶段训练策略，结合强化学习改善低频位姿；4)解耦物体采样技术，解决多物体场景中的概念混淆。相比之前工作，SceneDesigner首次实现多物体9D位姿控制，同时控制位置、大小和方向；使用真实世界数据而非合成数据提高泛化能力；在保持控制精度的同时提高图像质量；支持用户个性化定制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'SceneDesigner通过创新的CNOCS地图表示、两阶段训练策略和解耦物体采样技术，首次实现了对图像中多个物体的9D位姿（位置、大小和方向）的精确控制，显著提升了图像生成在3D空间中的可控性和质量。'}


### 论文摘要

Controllable image generation has attracted increasing attention in recent years, enabling users to manipulate visual content such as identity and style. However, achieving simultaneous control over the 9D poses (location, size, and orientation) of multiple objects remains an open challenge. Despite recent progress, existing methods often suffer from limited controllability and degraded quality, falling short of comprehensive multi-object 9D pose control. To address these limitations, we propose SceneDesigner, a method for accurate and flexible multi-object 9-DoF pose manipulation. SceneDesigner incorporates a branched network to the pre-trained base model and leverages a new representation, CNOCS map, which encodes 9D pose information from the camera view. This representation exhibits strong geometric interpretation properties, leading to more efficient and stable training. To support training, we construct a new dataset, ObjectPose9D, which aggregates images from diverse sources along with 9D pose annotations. To further address data imbalance issues, particularly performance degradation on low-frequency poses, we introduce a two-stage training strategy with reinforcement learning, where the second stage fine-tunes the model using a reward-based objective on rebalanced data. At inference time, we propose Disentangled Object Sampling, a technique that mitigates insufficient object generation and concept confusion in complex multi-object scenes. Moreover, by integrating user-specific personalization weights, SceneDesigner enables customized pose control for reference subjects. Extensive qualitative and quantitative experiments demonstrate that SceneDesigner significantly outperforms existing approaches in both controllability and quality. Code is publicly available at https://github.com/FudanCVL/SceneDesigner.

---

## 98. Formal Abductive Latent Explanations for Prototype-Based Networks

**论文链接:** [http://arxiv.org/abs/2511.16588v1](http://arxiv.org/abs/2511.16588v1)

**作者:** Jules Soria, Zakaria Chihani, Julien Girard-Satabin, Alban Grastien, Romain Xu-Darme, Daniela Cancila

**发布时间:** 2025-11-20

**备注:** Accepted at AAAI-26

### GPT解析

### 总结

该论文介绍了案例推理网络及其解释机制，指出现有解释方法的局限性，并提出了一种新的解释形式化方法——溯因潜在解释（ALEs），该方法结合了案例推理的可解释性和形式化XAI的保证，并通过实验验证了其有效性。

### 背景

案例推理网络是基于机器学习的模型，通过比较输入与训练样本的原型之间的相似性进行预测，并能通过指出贡献最大的原型来解释决策，因此被设计为'可解释的'。

### 目的

解决现有案例推理网络解释方法有时会产生误导的问题，特别是在安全关键环境中，提出一种更可靠、更形式化的解释方法。

### 方法

受形式化可解释人工智能（FXAI）领域的启发，提出溯因潜在解释（ALEs）形式化表达，用于表达实例的中间表示上足以推断预测的充分条件。设计了一个基于三种不同范式的无求解器且可扩展的生成ALEs的算法。

### 主要发现

现有案例推理网络的解释方法有时会产生误导，几个不同的实例可能导致不同的预测，但却有相同的解释。提出的ALEs方法能够提供更可靠、更形式化的解释。

### 结论

ALEs方法结合了案例推理模型的内在可解释性和形式化XAI提供的保证，在各种数据集上的标准图像分类和细粒度图像分类任务上表现出可行性，为安全关键环境中的解释提供了更可靠的解决方案。

### 翻译

案例推理网络是机器学习模型，它们基于输入与训练样本的原型部分之间的相似性进行预测。这类模型能够通过指出对最终结果贡献最大的原型来解释每个决策。由于解释是预测的核心部分，它们通常被设计为'可解释的'。尽管有前景，我们表明这类解释有时会产生误导，这限制了它们在安全关键环境中的有用性。特别是，几个实例可能导致不同的预测，但却有相同的解释。受形式化可解释人工智能（FXAI）领域的启发，我们提出了溯因潜在解释（ALEs），这是一种形式化表达，用于表达实例的中间（潜在）表示上足以推断预测的充分条件。我们的方法结合了案例推理模型的内在可解释性和形式化XAI提供的保证。我们提出了一个基于三种不同范式的无求解器且可扩展的生成ALEs的算法，比较了这些算法，并在各种数据集上展示了我们的方法在标准图像分类和细粒度图像分类任务上的可行性。相关代码可在https://github.com/julsoria/ale找到。


### 论文摘要

Case-based reasoning networks are machine-learning models that make predictions based on similarity between the input and prototypical parts of training samples, called prototypes. Such models are able to explain each decision by pointing to the prototypes that contributed the most to the final outcome. As the explanation is a core part of the prediction, they are often qualified as ``interpretable by design". While promising, we show that such explanations are sometimes misleading, which hampers their usefulness in safety-critical contexts. In particular, several instances may lead to different predictions and yet have the same explanation. Drawing inspiration from the field of formal eXplainable AI (FXAI), we propose Abductive Latent Explanations (ALEs), a formalism to express sufficient conditions on the intermediate (latent) representation of the instance that imply the prediction. Our approach combines the inherent interpretability of case-based reasoning models and the guarantees provided by formal XAI. We propose a solver-free and scalable algorithm for generating ALEs based on three distinct paradigms, compare them, and present the feasibility of our approach on diverse datasets for both standard and fine-grained image classification. The associated code can be found at https://github.com/julsoria/ale

---

## 99. TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2511.16423v1](http://arxiv.org/abs/2511.16423v1)

**作者:** Li Zhang, Zhongxuan Han, XiaoHua Feng, Jiaming Zhang, Yuyuan Li, Linbo Jiang, Jianan Lin, Chaochao Chen

**发布时间:** 2025-11-20

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

论文提出了一种名为TOFA的训练免费单轮联邦适应框架，用于解决预训练视觉语言模型在联邦学习环境下的适应问题，有效降低了通信成本并提高了安全性。

### 背景

联邦学习中预训练视觉语言模型(VLMs)的高效轻量适应是一个新兴研究课题。现有适应算法通常需要迭代训练，导致显著的通信成本和潜在攻击风险增加。单轮联邦训练技术可以减少客户端-服务器交换，但当前方法面临多模态信息利用不足、缺乏处理数据异质性的策略以及需要额外训练资源等挑战。

### 目的

开发一种轻量级单轮联邦VLM适应方法，解决现有方法的通信成本和安全问题，同时克服当前单轮方法面临的三个主要挑战：充分利用VLMs中的多模态信息、处理严重数据异质性、避免额外训练资源需求。

### 方法

提出TOFA(Training-free One-shot Federated Adaptation)框架，采用双管道策略：视觉管道使用分层贝叶斯模型学习个性化的类别特定原型分布；文本管道评估和对齐生成的本地文本提示以增强鲁棒性；并通过自适应权重校准机制结合两种模态的预测，平衡个性化与鲁棒性以处理数据异质性。整个方法无需训练，不依赖额外的客户端或服务器训练资源。

### 主要发现

TOFA方法能够充分利用预训练VLMs中可泛化的多模态特征，通过双管道提取任务相关表示。在多种联邦设置下的9个数据集上的广泛实验证明了该方法的有效性，成功解决了现有单轮方法面临的三个主要挑战。

### 结论

TOFA框架实现了真正的训练免费联邦VLM适应，在通信效率、安全性和处理数据异质性方面表现出色，为联邦学习环境下的视觉语言模型适应提供了新思路。

### 翻译

通过本地客户端与中央服务器之间的协作交互，高效轻量地使预训练视觉语言模型(VLMs)适应下游任务是联邦学习中一个快速发展的研究课题。现有的适应算法通常进行迭代训练，这会产生显著的通信成本并增加对潜在攻击的敏感性。受单轮联邦训练技术的启发，开发一种轻量级单轮联邦VLM适应方法来解决这些问题特别有吸引力。然而，当前的单轮方法在联邦环境中适应VLMs时面临某些挑战：(1)未能充分利用VLMs固有的丰富多模态信息；(2)缺乏专门的处理严重数据异质性的适应策略；(3)需要额外的客户端或服务器训练资源。为解决这些差距，我们提出了一种新型的VLM训练免费单轮联邦适应框架，名为TOFA。为了充分利用预训练VLMs中可泛化的多模态特征，TOFA采用视觉和文本两种管道来提取任务相关的表示。在视觉管道中，分层贝叶斯模型学习个性化的、类别特定的原型分布。对于文本管道，TOFA评估并全局对齐生成的本地文本提示以提高鲁棒性。还引入了自适应权重校准机制来结合两种模态的预测，平衡个性化与鲁棒性以处理数据异质性。我们的方法是训练免费的，不依赖于客户端或服务器端的额外训练资源。在各种联邦设置下的9个数据集上的广泛实验证明了所提出的TOFA方法的有效性。


### 论文摘要

Efficient and lightweight adaptation of pre-trained Vision-Language Models (VLMs) to downstream tasks through collaborative interactions between local clients and a central server is a rapidly emerging research topic in federated learning. Existing adaptation algorithms are typically trained iteratively, which incur significant communication costs and increase the susceptibility to potential attacks. Motivated by the one-shot federated training techniques that reduce client-server exchanges to a single round, developing a lightweight one-shot federated VLM adaptation method to alleviate these issues is particularly attractive. However, current one-shot approaches face certain challenges in adapting VLMs within federated settings: (1) insufficient exploitation of the rich multimodal information inherent in VLMs; (2) lack of specialized adaptation strategies to systematically handle the severe data heterogeneity; and (3) requiring additional training resource of clients or server. To bridge these gaps, we propose a novel Training-free One-shot Federated Adaptation framework for VLMs, named TOFA. To fully leverage the generalizable multimodal features in pre-trained VLMs, TOFA employs both visual and textual pipelines to extract task-relevant representations. In the visual pipeline, a hierarchical Bayesian model learns personalized, class-specific prototype distributions. For the textual pipeline, TOFA evaluates and globally aligns the generated local text prompts for robustness. An adaptive weight calibration mechanism is also introduced to combine predictions from both modalities, balancing personalization and robustness to handle data heterogeneity. Our method is training-free, not relying on additional training resources on either the client or server side. Extensive experiments across 9 datasets in various federated settings demonstrate the effectiveness of the proposed TOFA method.

---

## 100. Mem-MLP: Real-Time 3D Human Motion Generation from Sparse Inputs

**论文链接:** [http://arxiv.org/abs/2511.16264v1](http://arxiv.org/abs/2511.16264v1)

**作者:** Sinan Mutlu, Georgios F. Angelis, Savas Ozkan, Paul Wisbey, Anastasios Drosou, Mete Ozay

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究提出了一种基于增强型多层感知器的新型全身追踪方法，通过记忆块组件和多任务学习框架，从稀疏传感器输入生成流畅的全身动作，显著提高了AR/VR应用中的全身追踪质量。

### 背景

真实且流畅的全身追踪对沉浸式AR/VR应用至关重要。现有系统主要通过头戴式显示器（HMD）和控制器追踪头部和手部，导致3D全身重建不完整。

### 目的

开发一种从有限传感器收集的稀疏输入生成全身动作的方法，使用神经网络模型解决3D全身重建不完整的问题。

### 方法

提出了一种基于多层感知器（MLP）主干网络的新方法，增强残差连接和名为'记忆块'的新型神经网络组件。记忆块使用可训练的代码向量表示缺失的传感器数据，并与先前时间点的稀疏信号结合以提高时间一致性。将解决方案制定为多任务学习问题，使MLP主干网络能够学习强大的表示，提高准确性。

### 主要发现

实验表明，该方法通过显著减少预测误差，优于最先进的基线方法。在移动HMD上达到72帧每秒，最终提高了准确性与运行时间的权衡。

### 结论

提出的全身追踪方法能够有效利用有限传感器数据生成流畅的全身动作，在移动设备上运行效率高，适合实际AR/VR应用。

### 翻译

真实且流畅的全身追踪对沉浸式AR/VR应用至关重要。现有系统主要通过头戴式显示器（HMD）和控制器追踪头部和手部，导致3D全身重建不完整。一种潜在的方法是使用神经网络模型从有限传感器收集的稀疏输入生成全身动作。在本文中，我们提出了一种基于多层感知器（MLP）主干网络的新方法，该方法增强了残差连接和一种名为记忆块的新型神经网络组件。特别是，记忆块使用可训练的代码向量表示缺失的传感器数据，这些代码向量与前一时间点的稀疏信号结合，以提高时间一致性。此外，我们将解决方案制定为多任务学习问题，使我们的MLP主干网络能够学习强大的表示，提高准确性。我们的实验表明，我们的方法通过显著减少预测误差，优于最先进的基线方法。此外，它在移动HMD上达到72帧每秒，最终提高了准确性与运行时间的权衡。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决如何从稀疏输入（仅头部和双手的传感器数据）实时生成完整的3D人体全身运动。这个问题在AR/VR应用中至关重要，因为现有的系统只能跟踪头部和手部，导致虚拟环境中的用户身体不完整，无法提供真正的沉浸式体验。解决这一问题可以仅用标准设备（头显和手控制器）实现全身运动跟踪，无需额外传感器，使AR/VR体验更加自然和逼真。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有AR/VR系统的局限性，即只能跟踪头部和手部，而添加更多传感器不切实际。他们基于AGRoL [13]的MLP架构，引入了创新的Memory-Block组件来处理缺失的传感器数据，并借鉴了VQ-VAE模型[15]来生成可训练的code-vectors。与扩散模型方法不同，作者选择单步生成运动序列以提高效率。他们还采用多任务学习框架，同时估计关节旋转和位置，并使用同方差不确定性进行损失加权，解决不同类型损失函数之间的冲突。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用Memory-Block组件表示缺失的传感器输入，并将其与先前时间点的稀疏信号结合以提高时间一致性；同时采用多任务学习框架，同时估计关节旋转和位置。整体流程包括：1) 输入处理：接收时间窗口内的头部、左手和右手的位置、旋转、线速度和角速度数据；2) MLP主干处理：通过MLP块和Memory-块处理输入，Memory-块结合稀疏特征、运动数据和code-vectors生成记忆特征；3) 多任务预测：两个分支分别生成关节旋转和位置；4) 损失计算：使用多个损失函数并通过自动加权机制平衡；5) 推理：在移动设备上以72 FPS速度实时生成全身运动。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) Memory-Block组件，使用可训练code-vectors表示缺失数据并提高时间一致性；2) 多任务学习框架，同时估计旋转和位置，互相增强；3) 高效的损失加权机制，解决角度损失和距离损失冲突；4) 轻量级实时架构，在移动设备上达到72 FPS。相比之前的工作，Mem-MLP比AvatarPoser更准确且更高效；比AGRoL-Diffusion快得多（72 FPS vs 3.5 FPS）；仅需三个传感器输入（头部和双手）就能达到与需要更多输入的方法相当的准确性；避免了扩散模型的多步生成过程，更适合实时应用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Mem-MLP通过创新的Memory-Block组件和多任务学习框架，实现了从仅头部和双手的稀疏传感器输入中实时、高精度地生成完整3D人体全身运动，显著优于现有方法并达到了移动设备上的实时性能要求。'}


### 论文摘要

Realistic and smooth full-body tracking is crucial for immersive AR/VR applications. Existing systems primarily track head and hands via Head Mounted Devices (HMDs) and controllers, making the 3D full-body reconstruction in-complete. One potential approach is to generate the full-body motions from sparse inputs collected from limited sensors using a Neural Network (NN) model. In this paper, we propose a novel method based on a multi-layer perceptron (MLP) backbone that is enhanced with residual connections and a novel NN-component called Memory-Block. In particular, Memory-Block represents missing sensor data with trainable code-vectors, which are combined with the sparse signals from previous time instances to improve the temporal consistency. Furthermore, we formulate our solution as a multi-task learning problem, allowing our MLP-backbone to learn robust representations that boost accuracy. Our experiments show that our method outperforms state-of-the-art baselines by substantially reducing prediction errors. Moreover, it achieves 72 FPS on mobile HMDs that ultimately improves the accuracy-running time tradeoff.

---

## 101. CausalMamba: Interpretable State Space Modeling for Temporal Rumor Causality

**论文链接:** [http://arxiv.org/abs/2511.16191v1](http://arxiv.org/abs/2511.16191v1)

**作者:** Xiaotong Zhan, Xi Cheng

**发布时间:** 2025-11-20

**备注:** Preprint. 9 pages, 3 figures, 2 tables. Code and implementation details available at: https://github.com/XiaotongZhan/Causal_Mamba

### GPT解析

### 总结

CausalMamba是一个新型框架，整合了Mamba序列建模、图卷积网络和因果发现技术，用于社交媒体谣言检测，不仅能进行分类，还能提供反事实干预分析和可解释的见解。

### 背景

社交媒体上的谣言检测因复杂的传播动态和现有模型的有限可解释性而具有挑战性。现有神经网络架构虽然能捕捉内容和结构特征，但往往无法揭示错误信息传播的潜在因果机制。

### 目的

开发一个能够揭示谣言传播潜在因果机制的谣言检测框架，提供更可解释和可操作的错误信息检测系统。

### 方法

提出CausalMamba框架，整合基于Mamba的序列建模、图卷积网络(GCNs)和通过NOTEARS进行的不同iable因果发现。该框架学习推文时间序列和回复结构的联合表示，同时揭示潜在因果图以识别每个传播链中的有影响力节点。

### 主要发现

在Twitter15数据集上，CausalMamba与强基线相比具有竞争力的分类性能，并能实现反事实干预分析。定性结果表明，移除排名靠前的因果节点会显著改变图连接性，提供对谣言动态的可解释见解。

### 结论

CausalMamba为谣言分类和影响分析提供了统一的方法，为更可解释和可操作的错误信息检测系统铺平了道路。

### 翻译

社交媒体上的谣言检测仍然是一个具有挑战性的任务，由于复杂的传播动态和现有模型的有限可解释性。虽然最近的神经网络架构能够捕捉内容和结构特征，但它们往往无法揭示错误信息传播的潜在因果机制。我们提出了CausalMamba，一个新颖的框架，整合了基于Mamba的序列建模、图卷积网络(GCNs)和通过NOTEARS进行的不同iable因果发现。CausalMamba学习推文时间序列和回复结构的联合表示，同时揭示潜在因果图以识别每个传播链中的有影响力的节点。在Twitter15数据集上的实验表明，与强基线相比，我们的模型具有竞争力的分类性能，并唯一地实现了反事实干预分析。定性结果表明，移除排名靠前的因果节点会显著改变图连接性，为谣言动态提供了可解释的见解。我们的框架为谣言分类和影响分析提供了统一的方法，为更可解释和可操作的错误信息检测系统铺平了道路。


### 论文摘要

Rumor detection on social media remains a challenging task due to the complex propagation dynamics and the limited interpretability of existing models. While recent neural architectures capture content and structural features, they often fail to reveal the underlying causal mechanisms of misinformation spread. We propose CausalMamba, a novel framework that integrates Mamba-based sequence modeling, graph convolutional networks (GCNs), and differentiable causal discovery via NOTEARS. CausalMamba learns joint representations of temporal tweet sequences and reply structures, while uncovering latent causal graphs to identify influential nodes within each propagation chain. Experiments on the Twitter15 dataset show that our model achieves competitive classification performance compared to strong baselines, and uniquely enables counterfactual intervention analysis. Qualitative results demonstrate that removing top-ranked causal nodes significantly alters graph connectivity, offering interpretable insights into rumor dynamics. Our framework provides a unified approach for rumor classification and influence analysis, paving the way for more explainable and actionable misinformation detection systems.

---

## 102. Layer-wise Noise Guided Selective Wavelet Reconstruction for Robust Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2511.16162v1](http://arxiv.org/abs/2511.16162v1)

**作者:** Yuting Lu, Ziliang Wang, Weixin Xu, Wei Zhang, Yongqiang Zhao, Yang Yu, Xiaohong Zhang

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种名为分层噪声引导选择性小波重建(LNG-SWR)的新方法，用于提高医学图像分割模型在分布偏移和扰动下的稳定性，解决了传统对抗训练(AT)带来的清洁-鲁棒性权衡和高训练成本问题。

### 背景

临床应用中的分割模型需要在分布偏移和扰动下保持稳定。主流解决方案是对抗训练(AT)来提高鲁棒性，但AT通常带来清洁-鲁棒性权衡和高训练/调整成本，限制了医学影像的可扩展性和可维护性。

### 目的

提出一种新的方法来解决AT的局限性，提高医学图像分割的鲁棒性，同时避免传统AT方法的高训练成本和清洁-鲁棒性权衡问题。

### 方法

提出分层噪声引导选择性小波重建(LNG-SWR)。在训练过程中，在多个层注入小的、零均值噪声，学习频率偏差先验，使表示远离噪声敏感方向。然后在输入/特征分支上应用先验引导的选择性小波重建，实现频率适应：抑制噪声敏感带，增强方向结构和形状线索，稳定边界响应，同时保持频谱一致性。该框架与骨干网络无关，增加了较低的额外推理开销。

### 主要发现

在CT和超声数据集上，使用统一的PGD-L∞/L2和SSAH协议，LNG-SWR在清洁Dice/IoU上提供了一致的增益，并在强攻击下显著减少了性能下降；将LNG-SWR与AT结合使用会产生累加增益；与对抗训练结合时，鲁棒性进一步提高，而不牺牲清洁准确度。

### 结论

LNG-SWR为对抗和标准训练中的鲁棒医学图像分割提供了一条简单、有效且工程友好的路径，可以作为AT的插件增强，也可以在没有AT的情况下提高鲁棒性。

### 翻译

临床应用要求分割模型在分布偏移和扰动下保持稳定。主流解决方案是通过对抗训练(AT)来提高鲁棒性；然而，AT常常带来清洁-鲁棒性权衡和高训练/调整成本，这限制了医学影像中的可扩展性和可维护性。我们提出了分层噪声引导选择性小波重建(LNG-SWR)。在训练期间，我们在多个层注入小的、零均值噪声，学习频率偏差先验，使表示远离噪声敏感方向。然后我们在输入/特征分支上应用先验引导的选择性小波重建来实现频率适应：抑制噪声敏感带，增强方向结构和形状线索，稳定边界响应，同时保持频谱一致性。该框架与骨干网络无关，增加了较低的额外推理开销。它可以作为AT的插件增强，也可以在没有AT的情况下提高鲁棒性。在CT和超声数据集上，使用统一的PGD-L∞/L2和SSAH协议，LNG-SWR在清洁Dice/IoU上提供了一致的增益，并在强攻击下显著减少了性能下降；将LNG-SWR与AT结合使用会产生累加增益。当与对抗训练结合时，鲁棒性进一步提高而不牺牲清洁准确度，表明这是一种工程友好且可扩展的鲁棒分割路径。这些结果表明，LNG-SWR在对抗和标准训练中都为鲁棒医学图像分割提供了一条简单、有效且工程友好的路径。


### 论文摘要

Clinical deployment requires segmentation models to stay stable under distribution shifts and perturbations. The mainstream solution is adversarial training (AT) to improve robustness; however, AT often brings a clean--robustness trade-off and high training/tuning cost, which limits scalability and maintainability in medical imaging. We propose \emph{Layer-wise Noise-Guided Selective Wavelet Reconstruction (LNG-SWR)}. During training, we inject small, zero-mean noise at multiple layers to learn a frequency-bias prior that steers representations away from noise-sensitive directions. We then apply prior-guided selective wavelet reconstruction on the input/feature branch to achieve frequency adaptation: suppress noise-sensitive bands, enhance directional structures and shape cues, and stabilize boundary responses while maintaining spectral consistency. The framework is backbone-agnostic and adds low additional inference overhead. It can serve as a plug-in enhancement to AT and also improves robustness without AT. On CT and ultrasound datasets, under a unified protocol with PGD-$L_{\infty}/L_{2}$ and SSAH, LNG-SWR delivers consistent gains on clean Dice/IoU and significantly reduces the performance drop under strong attacks; combining LNG-SWR with AT yields additive gains. When combined with adversarial training, robustness improves further without sacrificing clean accuracy, indicating an engineering-friendly and scalable path to robust segmentation. These results indicate that LNG-SWR provides a simple, effective, and engineering-friendly path to robust medical image segmentation in both adversarial and standard training regimes.

---

## 103. Degradation-Aware Hierarchical Termination for Blind Quality Enhancement of Compressed Video

**论文链接:** [http://arxiv.org/abs/2511.16137v1](http://arxiv.org/abs/2511.16137v1)

**作者:** Li Yu, Yingbo Zhao, Shiyu Wu, Siyue Yu, Moncef Gabbouj, Qingshan Liu

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种针对压缩视频质量增强的盲方法，通过预训练的退化表征学习模块和分层终止机制，解决了现有方法在处理未知量化参数时的局限性，显著提高了视频质量并减少了计算时间。

### 背景

现有视频压缩质量增强研究主要依赖已知的量化参数(QPs)，采用非盲方法，每个QP设置使用不同增强模型。但在实际转码或传输场景中，QP可能部分或完全未知，限制了这些方法的适用性。

### 目的

开发盲QECV技术，解决当前盲方法仅捕获全局退化信息、缺乏空间细节，以及未考虑不同压缩级别计算需求差异的问题。

### 方法

提出预训练的退化表征学习(DRL)模块，解耦并提取高维、多尺度退化表征指导伪影去除；引入分层终止机制，根据压缩级别动态调整伪影减少阶段数量。

### 主要发现

在QP = 22时，与竞争的最先进盲方法相比，PSNR提高了110%(从0.31 dB提高到0.65 dB)；分层终止机制在QP = 22时的平均推理时间比QP = 42减少了一半。

### 结论

所提出的DRL模块和分层终止机制有效解决了现有盲QECV方法的局限性，在性能和计算效率上都有显著提升。

### 翻译

现有的压缩视频质量增强研究主要依赖于已知的量化参数(QPs)，采用每个QP设置使用不同增强模型的非盲方法。然而，在实际转码或传输场景中，QP可能部分或完全未知，限制了这些方法的适用性，促使了盲QECV技术的发展。当前盲方法通过使用带交叉熵损失的分类模型生成退化向量，作为通道注意力来指导伪影去除。但这些向量仅捕获全局退化信息，缺乏空间细节，难以适应不同空间位置变化的伪影模式。为解决这些限制，我们提出了一种预训练的退化表征学习(DRL)模块，解耦并从视频内容中提取高维、多尺度的退化表征，用于指导伪影去除。此外，盲方法和非盲方法通常在所有QP上采用统一架构，忽略了不同压缩级别固有的不同计算需求。因此，我们引入了分层终止机制，根据压缩级别动态调整伪影减少阶段的数量。实验结果表明，所提出的方法显著提高了性能，在QP = 22时，与竞争的最先进盲方法相比，PSNR提高了110%(从0.31 dB提高到0.65 dB)。此外，提出的分层终止机制在QP = 22时的平均推理时间比QP = 42减少了一半。


### 论文摘要

Existing studies on Quality Enhancement for Compressed Video (QECV) predominantly rely on known Quantization Parameters (QPs), employing distinct enhancement models per QP setting, termed non-blind methods. However, in real-world scenarios involving transcoding or transmission, QPs may be partially or entirely unknown, limiting the applicability of such approaches and motivating the development of blind QECV techniques. Current blind methods generate degradation vectors via classification models with cross-entropy loss, using them as channel attention to guide artifact removal. However, these vectors capture only global degradation information and lack spatial details, hindering adaptation to varying artifact patterns at different spatial positions. To address these limitations, we propose a pretrained Degradation Representation Learning (DRL) module that decouples and extracts high-dimensional, multiscale degradation representations from video content to guide the artifact removal. Additionally, both blind and non-blind methods typically employ uniform architectures across QPs, hence, overlooking the varying computational demands inherent to different compression levels. We thus introduce a hierarchical termination mechanism that dynamically adjusts the number of artifact reduction stages based on the compression level. Experimental results demonstrate that the proposed approach significantly enhances performance, achieving a PSNR improvement of 110% (from 0.31 dB to 0.65 dB) over a competing state-of-the-art blind method at QP = 22. Furthermore, the proposed hierarchical termination mechanism reduces the average inference time at QP = 22 by half compared to QP = 42.

---

## 104. Pathlet Variational Auto-Encoder for Robust Trajectory Generation

**论文链接:** [http://arxiv.org/abs/2511.16105v1](http://arxiv.org/abs/2511.16105v1)

**作者:** Yuanbo Tang, Yan Tang, Zixuan Zhang, Zihui Zhao, Yang Li

**发布时间:** 2025-11-20

### GPT解析

### 总结

轨迹生成在隐私保护的城市移动研究中日益重要，现有深度学习方法虽有效但缺乏鲁棒性和可解释性。本文提出基于路径表示的深度生成模型，能有效学习数据分布，在真实数据集上表现优异，且能用于多种下游任务，同时具有显著的效率优势。

### 背景

轨迹生成在隐私保护的城市移动研究和基于位置的服务应用中受到越来越多的关注。许多研究使用深度学习或生成式AI方法对轨迹建模并取得有希望的结果，但这些模型的鲁棒性和可解释性很大程度上尚未被探索，限制了其在嘈杂真实世界数据和下游任务中的应用。

### 目的

解决现有轨迹生成模型在鲁棒性和可解释性方面的不足，提高其在嘈杂数据上的表现和下游任务中的可信度，同时提高计算效率。

### 方法

利用城市轨迹中的规则结构，提出了一种基于路径表示的深度生成模型。该模型使用与学习的轨迹段字典相关联的二进制向量对轨迹进行编码，引入概率图模型描述轨迹生成过程，包括变分自编码器组件和线性解码器组件。模型能同时学习路径表示的潜在嵌入和捕捉移动模式的路径字典，并能根据时间和空间约束生成定制轨迹。

### 主要发现

即使使用嘈杂数据，模型也能有效学习数据分布，在两个真实世界轨迹数据集上比强基线方法分别实现了35.4%和26.3%的相对改进。生成的轨迹可以方便地用于多个下游任务，包括轨迹预测和数据去噪。框架设计提供了显著的效率优势，与之前的方法相比节省了64.8%的时间和56.5%的GPU内存。

### 结论

所提出的基于路径表示的深度生成模型在轨迹生成任务中表现出色，具有高鲁棒性、可解释性和计算效率，能有效应用于嘈杂数据并支持多种下游任务，为隐私保护的城市移动研究和基于位置的服务应用提供了新的解决方案。

### 翻译

轨迹生成最近在隐私保护的城市移动研究和基于位置的服务应用中引起了越来越多的关注。尽管许多研究使用深度学习或生成式AI方法对轨迹进行建模并取得了有希望的结果，但这些模型的鲁棒性和可解释性在很大程度上尚未被探索。这限制了轨迹生成算法在嘈杂的真实世界数据上的应用及其在下游任务中的可信度。为了解决这个问题，我们利用城市轨迹中的规则结构，提出了一种基于路径表示的深度生成模型，该模型使用与学习的轨迹段字典相关联的二进制向量对轨迹进行编码。具体来说，我们引入了一个概率图模型来描述轨迹生成过程，其中包括一个变分自编码器组件和一个线性解码器组件。在训练过程中，模型可以同时学习路径表示的潜在嵌入和捕捉轨迹数据集中移动模式的路径字典。我们模型的条件版本也可用于根据时间和空间约束生成定制轨迹。即使使用嘈杂数据，我们的模型也能有效学习数据分布，在两个真实世界轨迹数据集上比强基线方法分别实现了35.4%和26.3%的相对改进。此外，生成的轨迹可以方便地用于多个下游任务，包括轨迹预测和数据去噪。最后，框架设计提供了显著的效率优势，与之前的方法相比节省了64.8%的时间和56.5%的GPU内存。


### 论文摘要

Trajectory generation has recently drawn growing interest in privacy-preserving urban mobility studies and location-based service applications. Although many studies have used deep learning or generative AI methods to model trajectories and have achieved promising results, the robustness and interpretability of such models are largely unexplored. This limits the application of trajectory generation algorithms on noisy real-world data and their trustworthiness in downstream tasks. To address this issue, we exploit the regular structure in urban trajectories and propose a deep generative model based on the pathlet representation, which encode trajectories with binary vectors associated with a learned dictionary of trajectory segments. Specifically, we introduce a probabilistic graphical model to describe the trajectory generation process, which includes a Variational Autoencoder (VAE) component and a linear decoder component. During training, the model can simultaneously learn the latent embedding of pathlet representations and the pathlet dictionary that captures mobility patterns in the trajectory dataset. The conditional version of our model can also be used to generate customized trajectories based on temporal and spatial constraints.   Our model can effectively learn data distribution even using noisy data, achieving relative improvements of $35.4\%$ and $26.3\%$ over strong baselines on two real-world trajectory datasets. Moreover, the generated trajectories can be conveniently utilized for multiple downstream tasks, including trajectory prediction and data denoising. Lastly, the framework design offers a significant efficiency advantage, saving $64.8\%$ of the time and $56.5\%$ of GPU memory compared to previous approaches.

---

## 105. Mitigating Estimation Bias with Representation Learning in TD Error-Driven Regularization

**论文链接:** [http://arxiv.org/abs/2511.16090v1](http://arxiv.org/abs/2511.16090v1)

**作者:** Haohui Chen, Zhiyong Chen, Aoxiang Liu, Wentuo Fang

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种基于时间差误差驱动正则化的双行动者-批评者框架，通过凸组合策略平衡悲观估计和乐观探索，实现灵活的偏差控制，并将增强的状态和动作表征整合到网络中，显著提升了连续控制任务中的性能。

### 背景

确定性策略梯度算法在连续控制中存在价值估计偏差问题，虽然双批评者可以减少这种偏差，但双行动者的探索潜力尚未被充分探索。

### 目的

基于时间差误差驱动的正则化框架，提出改进方法，实现灵活的偏差控制和更强的表征学习能力。

### 方法

提出三种凸组合策略（对称和非对称）平衡悲观估计缓解高估问题；通过双行动者进行乐观探索缓解低估问题；使用单一超参数实现偏差谱上的可调控制；将增强的状态和动作表征整合到行动者和批评者网络中。

### 主要发现

该方法在实验中始终优于基准，展示了可调偏差的价值，并揭示了高估和低估可以根据环境不同而被不同利用。

### 结论

通过结合双行动者和双批评者的优势，实现了更好的性能和更灵活的偏差控制，证明了在连续控制任务中有效利用不同类型偏差的重要性。

### 翻译

确定性策略梯度算法在连续控制中存在价值估计偏差，降低了性能。虽然双批评者可以减少此类偏差，但双行动者的探索潜力尚未被充分探索。基于时间差误差驱动的正则化（TDDR）这一双行动者-批评者框架，本文引入了增强方法以实现灵活的偏差控制和更强的表征学习。我们提出了三种凸组合策略，对称和非对称，平衡悲观估计以缓解高估，并通过双行动者进行乐观探索以缓解低估。单一超参数控制这一机制，实现偏差谱上的可调控制。为进一步提升性能，我们将增强的状态和动作表征整合到行动者和批评者网络中。大量实验表明，我们的方法始终优于基准，证明了可调偏差的价值，并揭示了高估和低估可以根据环境不同而被不同利用。


### 论文摘要

Deterministic policy gradient algorithms for continuous control suffer from value estimation biases that degrade performance. While double critics reduce such biases, the exploration potential of double actors remains underexplored. Building on temporal-difference error-driven regularization (TDDR), a double actor-critic framework, this work introduces enhanced methods to achieve flexible bias control and stronger representation learning. We propose three convex combination strategies, symmetric and asymmetric, that balance pessimistic estimates to mitigate overestimation and optimistic exploration via double actors to alleviate underestimation. A single hyperparameter governs this mechanism, enabling tunable control across the bias spectrum. To further improve performance, we integrate augmented state and action representations into the actor and critic networks. Extensive experiments show that our approach consistently outperforms benchmarks, demonstrating the value of tunable bias and revealing that both overestimation and underestimation can be exploited differently depending on the environment.

---

## 106. Change-of-Basis Pruning via Rotational Invariance

**论文链接:** [http://arxiv.org/abs/2511.16061v1](http://arxiv.org/abs/2511.16061v1)

**作者:** Alex Ning, Vainateya Rangaraju

**发布时间:** 2025-11-20

**备注:** 14 pages, 5 figures

### GPT解析

### 总结

论文提出TSRAs激活函数族，实现与基变换剪枝(CoB)兼容的神经网络架构，通过旋转不变性设计提高结构化剪枝有效性

### 背景

结构化剪枝效果取决于重要性在表示空间中的分布，CoB剪枝通过正交线性变换集中重要性，但标准深度学习架构对这类变换不具有内在不变性

### 目的

提出TSRAs激活函数，使其对正交线性变换具有不变性，从而实现与CoB剪枝的兼容

### 方法

引入TSRAs激活函数，在VGG-16上应用CoB+TSRA框架，使用激活幅度重要性评估方法，在CIFAR-10数据集上测试固定比例和基于阈值的剪枝策略

### 主要发现

1) TSRAs比ReLU精度下降4.52%；2) CoB在固定比例剪枝下将可靠剪枝前沿从30%扩展到70%；3) 基于阈值剪枝下可剪除90-96%参数，微调后仅1-6%精度下降

### 结论

旋转不变的架构为基变换剪枝提供了一条有前途的路径，TSRAs作为概念验证表明这种设计原则可行

### 翻译

结构化剪枝会移除整个神经元或通道，但其有效性取决于重要性在表示空间中的分布方式。基变换(CoB)剪枝通过应用正交线性变换来解决这个问题，这些变换将重要性集中在某些维度内。然而，许多标准的深度学习架构对这类变换不具有内在不变性。为了实现兼容性，我们引入了双子空间径向激活函数(TSRAs)：一种激活函数族，在其两个激活子空间内对独立应用的正交线性变换具有不变性。这种不变性允许CoB变换与周围权重合并而不会产生额外参数。我们将这项工作定位为概念验证，表明旋转不变性设计可能为基变换剪枝提供一种原则性的方法。我们没有分析多种TSRA候选，也没有探索任何TSRA的权重初始化。这些局限性，加上我们为允许旋转不变性所做的其他必要修改，导致与基于ReLU的对照组相比精度下降了4.52%。然而，使用激活幅度重要性，实现我们CoB+TSRA框架的VGG-16在CIFAR-10上显示出令人鼓舞的结果。在固定比例的结构化剪枝下，CoB在所有剪枝比例下都优于TSRA基线，并将可靠的剪枝前沿从约30%扩展到70%的参数，无需剪枝后微调。在基于阈值的剪枝策略下，CoB可以剪除90-96%的参数，同时在微调后保持1-6%的精度下降。这些结果表明，旋转不变的架构可能为基变换剪枝提供了一条有希望的路径。


### 论文摘要

Structured pruning removes entire neurons or channels, but its effectiveness depends on how importance is distributed across the representation space. Change-of-basis (CoB) pruning addresses this challenge by applying orthogonal linear transformations that concentrate importance within certain dimensions. However, many standard deep learning architectures are not inherently invariant to such transformations. To enable compatibility, we introduce two-subspace radial activations (TSRAs): an activation family that is invariant to orthogonal linear transformations applied independently within its two activation subspaces. This invariance allows CoB transformations to be merged into surrounding weights without incurring extra parameters. We position this work as a proof-of-concept that a rotationally invariant design may offer a principled approach towards change-of-basis pruning. We do not provide an analysis of multiple TSRA candidates nor do we explore weight initialization for any TSRAs. These limitations, combined with other necessary modifications we make to permit rotational invariance, result in a slight accuracy drop of $4.52\%$ compared to a ReLU-based control. However, using activation-magnitude importance, VGG-16 implementing our CoB+TSRA framework shows encouraging results on CIFAR-10. Under fixed-ratio structured pruning, CoB improves accuracy over a TSRA baseline at all pruning ratios and extends reliable pruning frontier from roughly $30\%$ to $70\%$ of parameters without post-prune fine tuning. Under threshold-based pruning strategies, CoB prunes $90-96\%$ of parameters while maintaining $1-6\%$ accuracy drop after fine-tuning. Together, these results indicate that rotationally invariant architectures may offer a promising path towards CoB pruning.

---

## 107. Learning Tractable Distributions Of Language Model Continuations

**论文链接:** [http://arxiv.org/abs/2511.16054v1](http://arxiv.org/abs/2511.16054v1)

**作者:** Gwen Yidou-Weng, Ian Li, Anji Liu, Oliver Broadrick, Guy Van den Broeck, Benjie Wang

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究提出了Learning to Look Ahead (LTLA)方法，用于解决受控语言生成中的条件约束问题，通过结合语言模型和隐马尔可夫模型提高约束满足度，同时保持推理效率。

### 背景

受控语言生成需要在序列级别上对文本进行条件约束（如语法、风格或安全性），这些约束可能依赖于未来的标记，使得直接条件化自回归语言模型变得不可处理。

### 目的

开发一种能够处理未来标记依赖的约束条件，同时保持计算效率的受控语言生成方法。

### 方法

LTLA是一种混合方法，将相同的基线语言模型用于丰富的前缀编码，与固定的可处理替代模型配对计算精确连续概率。通过批处理HMM更新和保持替代解码器固定，实现计算重用。

### 主要发现

LTLA比无条件HMM获得更高的条件似然，能够为视觉语言模型近似连续分布，在受控生成任务上提高约束满足度，同时保持相当的流畅性，且推理开销最小。

### 结论

LTLA有效地解决了受控语言生成中处理未来标记依赖的约束条件问题，通过结合语言模型和隐马尔可夫模型的优势，实现了高质量的约束满足和高效的计算。

### 翻译

受控语言生成在序列级别上对文本施加约束（例如语法、风格或安全性）。这些约束可能依赖于未来标记，这使得直接条件化自回归语言模型通常是不可处理的。先前工作使用可处理的替代模型（如隐马尔可夫模型HMMs）来近似连续分布，并在解码时调整模型的下一个标记的logits。然而，我们发现这些替代模型通常是弱上下文感知的，降低了查询质量。我们提出了Learning to Look Ahead (LTLA)，一种混合方法，将相同的基线语言模型用于丰富的前缀编码，与一个固定的可处理替代模型配对，该模型计算精确的连续概率。添加神经上下文时会出现两个效率问题：(i) 为每个候选下一个标记重新评分前缀需要在每一步对整个词汇表进行扫描；(ii) 为每个前缀预测新的替代参数虽然在单步上是可处理的，但迫使为每个新前缀重新计算未来概率，消除了重用。LTLA通过使用单个批处理HMM更新一次考虑所有下一个标记候选，并且仅将替代的潜在状态先验条件化为LM的隐藏表示，同时保持替代解码器固定，从而可以在前缀之间重用计算。实验表明，LTLA比无条件HMM获得更高的条件似然，可以为视觉语言模型近似连续分布（独立HMM无法编码视觉上下文），并在受控生成任务上提高约束满足度，同时保持相当的流畅性，且推理开销最小。


### 论文摘要

Controlled language generation conditions text on sequence-level constraints (for example, syntax, style, or safety). These constraints may depend on future tokens, which makes directly conditioning an autoregressive language model (LM) generally intractable. Prior work uses tractable surrogates such as hidden Markov models (HMMs) to approximate the distribution over continuations and adjust the model's next-token logits at decoding time. However, we find that these surrogates are often weakly context aware, which reduces query quality. We propose Learning to Look Ahead (LTLA), a hybrid approach that pairs the same base language model for rich prefix encoding with a fixed tractable surrogate model that computes exact continuation probabilities. Two efficiency pitfalls arise when adding neural context: (i) naively rescoring the prefix with every candidate next token requires a sweep over the entire vocabulary at each step, and (ii) predicting fresh surrogate parameters for each prefix, although tractable at a single step, forces recomputation of future probabilities for every new prefix and eliminates reuse. LTLA avoids both by using a single batched HMM update to account for all next-token candidates at once, and by conditioning only the surrogate's latent state prior on the LM's hidden representations while keeping the surrogate decoder fixed, so computations can be reused across prefixes. Empirically, LTLA attains higher conditional likelihood than an unconditional HMM, approximates continuation distributions for vision-language models where a standalone HMM cannot encode visual context, and improves constraint satisfaction at comparable fluency on controlled-generation tasks, with minimal inference overhead.

---

## 108. Bi-AQUA: Bilateral Control-Based Imitation Learning for Underwater Robot Arms via Lighting-Aware Action Chunking with Transformers

**论文链接:** [http://arxiv.org/abs/2511.16050v1](http://arxiv.org/abs/2511.16050v1)

**作者:** Takeru Tsunoori, Masato Kobayashi, Yuki Uranishi

**发布时间:** 2025-11-20

### GPT解析

### 总结

Bi-AQUA是一个基于水下双边控制的模仿学习框架，整合了光照感知的视觉处理，通过三级光照适应机制解决水下机器人操作面临的极端光照变化、颜色失真和能见度降低的挑战。

### 背景

水下机器人操作面临极端光照变化、颜色失真和能见度降低的挑战。

### 目的

引入Bi-AQUA，第一个基于水下双边控制的模仿学习框架，整合光照感知的视觉处理，提高水下机器人操作的性能。

### 方法

Bi-AQUA采用分层的三级光照适应机制：1) 光照编码器从RGB图像提取光照表示，无需人工标注；2) FiLM调制对视觉主干特征进行调制，实现自适应特征提取；3) 显式光照标记添加到transformer编码器输入中，实现任务感知的条件设置。

### 主要发现

在真实水下抓取放置任务上的实验表明，Bi-AQUA在多种静态和动态光照条件下实现了鲁棒性能，显著优于没有光照建模的双边基线。消融研究确认了所有三个光照感知组件都是关键的。

### 结论

这项工作连接了基于双边控制的模仿学习和水下操作，使机器人能够在具有挑战性的海洋环境中进行力敏感的自主操作。

### 翻译

水下机器人操作从根本上受到极端光照变化、颜色失真和能见度降低的挑战。我们引入了Bi-AQUA，这是第一个基于水下双边控制的模仿学习框架，集成了用于水下机械臂的光照感知视觉处理。Bi-AQUA采用分层的三级光照适应机制：一个光照编码器，无需人工标注即可从RGB图像中提取光照表示，并由模仿目标隐式监督；视觉主干特征的FiLM调制，用于自适应、光照感知的特征提取；以及一个显式的光照标记，添加到transformer编码器输入中，用于任务感知的条件设置。在多种静态和动态光照条件下的真实世界水下抓取放置任务上的实验表明，Bi-AQUA实现了鲁棒性能，并显著优于没有光照建模的双边基线。消融研究进一步确认了所有三个光照感知组件都是关键的。这项工作连接了基于双边控制的模仿学习和水下操作，使机器人能够在具有挑战性的海洋环境中进行力敏感的自主操作。更多材料请访问：https://mertcookimg.github.io/bi-aqua


### 论文摘要

Underwater robotic manipulation is fundamentally challenged by extreme lighting variations, color distortion, and reduced visibility. We introduce Bi-AQUA, the first underwater bilateral control-based imitation learning framework that integrates lighting-aware visual processing for underwater robot arms. Bi-AQUA employs a hierarchical three-level lighting adaptation mechanism: a Lighting Encoder that extracts lighting representations from RGB images without manual annotation and is implicitly supervised by the imitation objective, FiLM modulation of visual backbone features for adaptive, lighting-aware feature extraction, and an explicit lighting token added to the transformer encoder input for task-aware conditioning. Experiments on a real-world underwater pick-and-place task under diverse static and dynamic lighting conditions show that Bi-AQUA achieves robust performance and substantially outperforms a bilateral baseline without lighting modeling. Ablation studies further confirm that all three lighting-aware components are critical. This work bridges terrestrial bilateral control-based imitation learning and underwater manipulation, enabling force-sensitive autonomous operation in challenging marine environments. For additional material, please check: https://mertcookimg.github.io/bi-aqua

---

## 109. LiSTAR: Ray-Centric World Models for 4D LiDAR Sequences in Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2511.16049v1](http://arxiv.org/abs/2511.16049v1)

**作者:** Pei Liu, Songtao Wang, Lang Zhang, Xingyue Peng, Yuandong Lyu, Jiaxin Deng, Songxin Lu, Weiliang Ma, Xueyang Zhang, Yifei Zhan, XianPeng Lang, Jun Ma

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文介绍了LiSTAR，一种新颖的生成世界模型，用于合成高保真度和可控的4D LiDAR数据，以创建可扩展的自动驾驶模拟环境。

### 背景

合成高保真度和可控的4D LiDAR数据对于创建可扩展的自动驾驶模拟环境至关重要，但这一任务具有挑战性，因为传感器独特的球形几何结构、点云的时间稀疏性和动态场景的复杂性。

### 目的

解决LiDAR数据合成中的挑战，开发一种能够在传感器原生几何结构上操作的生成世界模型。

### 方法

提出了LiSTAR模型，引入了混合圆柱-球形（HCS）表示法减轻量化伪影，使用时空注意力和以射线为中心的转换器（START）捕获复杂动态，提出4D点云对齐的体素布局用于条件控制，并开发了掩码生成START（MaskSTART）框架学习场景的紧凑标记化表示。

### 主要发现

LiSTAR在4D LiDAR重建、预测和条件生成方面达到最先进性能，生成MMD降低76%，重建IoU提高32%，预测L1 Med降低50%。

### 结论

LiSTAR的性能水平为创建真实和可控的自动驾驶系统模拟提供了强大的新基础。

### 翻译

合成高保真度和可控的4D LiDAR数据对于创建可扩展的自动驾驶模拟环境至关重要。由于传感器独特的球形几何结构、点云的时间稀疏性和动态场景的复杂性，这一任务本质上具有挑战性。为解决这些挑战，我们提出了LiSTAR，一种新颖的生成世界模型，直接在传感器的原生几何结构上操作。LiSTAR引入了混合圆柱-球形（HCS）表示法，通过减轻笛卡尔网格中常见的量化伪影来保持数据保真度。为了从稀疏时间数据捕获复杂动态，它使用了时空注意力和以射线为中心的转换器（START），明确建模沿单个传感器射线的特征演化，以实现稳健的时间一致性。此外，为了实现可控合成，我们提出了一个用于条件控制的4D点云对齐的体素布局，以及相应的掩码生成START（MaskSTART）框架，该框架学习场景的紧凑、标记化表示，实现高效、高分辨率和布局引导的合成生成。全面的实验验证了LiSTAR在4D LiDAR重建、预测和条件生成方面的最先进性能，并有显著的定量提升：将生成MMD大幅降低76%，重建IoU提高32%，预测L1 Med降低50%。这种性能水平为创建真实和可控的自动驾驶系统模拟提供了强大的新基础。项目链接：https://ocean-luna.github.io/LiSTAR.gitub.io。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何生成高质量、可控的4D LiDAR数据（包含3D空间信息和时间维度的点云序列）的问题。这个问题在自动驾驶领域非常重要，因为LiDAR是自动驾驶的关键传感器，但真实世界数据采集成本高、覆盖场景有限。现有方法在处理LiDAR数据的独特几何特性（球形采样、稀疏性、不规则性）时存在局限，传统笛卡尔坐标网格化会引入量化失真，难以保持时间一致性，且缺乏精确控制生成内容的能力，限制了自动驾驶算法的训练和测试。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先深入理解了LiDAR数据的独特性（射线几何、稀疏分布、不规则采样），并识别出现有方法的局限性（忽略射线几何、时间不一致、控制能力有限）。他们借鉴了VQ-VAE的思想用于离散表示学习，受Transformer启发设计了专门针对LiDAR的注意力机制，扩展了Swin Transformer的移位窗口思想处理球面坐标周期性，并结合了扩散模型和掩码生成策略。系统设计上，他们创建了混合柱面-球面坐标系统(HCS)保留原始几何特性，开发了射线中心时空注意力(START)模块建模时空依赖，提出了掩码生成START(MaskSTART)框架实现可控合成。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是尊重传感器原生几何、时空联合建模、离散表示学习和可控合成。整体流程包括：1)将原始点云转换为HCS坐标并进行体素化；2)使用基于START模块的层次编码器映射到离散潜在空间；3)使用对称解码器重建原始序列；4)通过MaskSTART模块实现预测或生成任务；5)采用迭代掩码策略和分类器自由指导提高生成质量。训练过程使用组合损失函数优化模型参数。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)混合柱面-球面坐标系统(HCS)保留射线几何和距离分辨率；2)射线中心时空注意力(START)模块结合空间射线中心注意力和循环移位时间因果注意力；3)掩码生成START(MaskSTART)框架支持4D点云对齐的体素布局作为条件；4)端到端世界模型框架统一多个组件。相比之前工作，HCS更好地保留了LiDAR特性，START专门针对射线结构设计，联合建模时空关系，使用3D体素布局而非2D鸟瞰图提供更精细控制，在多个指标上显著超越现有方法。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiSTAR通过创新的混合柱面-球面坐标表示和射线中心时空注意力机制，实现了高质量、可控的4D LiDAR序列生成，为自动驾驶仿真提供了强大的新基础。'}


### 论文摘要

Synthesizing high-fidelity and controllable 4D LiDAR data is crucial for creating scalable simulation environments for autonomous driving. This task is inherently challenging due to the sensor's unique spherical geometry, the temporal sparsity of point clouds, and the complexity of dynamic scenes. To address these challenges, we present LiSTAR, a novel generative world model that operates directly on the sensor's native geometry. LiSTAR introduces a Hybrid-Cylindrical-Spherical (HCS) representation to preserve data fidelity by mitigating quantization artifacts common in Cartesian grids. To capture complex dynamics from sparse temporal data, it utilizes a Spatio-Temporal Attention with Ray-Centric Transformer (START) that explicitly models feature evolution along individual sensor rays for robust temporal coherence. Furthermore, for controllable synthesis, we propose a novel 4D point cloud-aligned voxel layout for conditioning and a corresponding discrete Masked Generative START (MaskSTART) framework, which learns a compact, tokenized representation of the scene, enabling efficient, high-resolution, and layout-guided compositional generation. Comprehensive experiments validate LiSTAR's state-of-the-art performance across 4D LiDAR reconstruction, prediction, and conditional generation, with substantial quantitative gains: reducing generation MMD by a massive 76%, improving reconstruction IoU by 32%, and lowering prediction L1 Med by 50%. This level of performance provides a powerful new foundation for creating realistic and controllable autonomous systems simulations. Project link: https://ocean-luna.github.io/LiSTAR.gitub.io.

---

## 110. Stabilization of nonautonomous linear parabolic equations with inputs subject to time-delay

**论文链接:** [http://arxiv.org/abs/2511.16616v1](http://arxiv.org/abs/2511.16616v1)

**作者:** Karl Kunisch, Sérgio S. Rodrigues

**发布时间:** 2025-11-20

**备注:** 8 figures

### GPT解析

### 总结

研究通过反馈输入调整有限执行器来实现非自治抛物方程的稳定性控制，并考虑输入时间延迟的影响。

### 背景

非自治抛物方程的稳定性控制面临输入时间延迟的挑战，这种延迟可能导致系统不稳定。

### 目的

克服时间延迟对系统稳定性的破坏性影响，设计有效的控制策略。

### 方法

基于对未来时间状态的预测来设计输入，预测依赖于当前时间的状态估计，状态估计由Luenberger观测器提供，观测器使用有限数量传感器的测量输出进行设计。

### 主要发现

研究了所得耦合系统的渐进行为，并通过数值模拟验证了理论发现，包括对传感器测量误差的响应测试。

### 结论

通过预测控制方法可以有效克服时间延迟对非自治抛物方程稳定性的不利影响。

### 翻译

通过调整有限数量执行器的反馈输入来实现非自治抛物方程的稳定性，其中假设输入存在时间延迟。为了克服时间延迟的破坏性影响，输入基于对未来时间状态的预测。该预测依赖于当前时间的状态估计，而状态估计由Luenberger观测器提供。观测器使用有限数量传感器的测量输出进行设计。研究了所得耦合系统的渐进行为。通过数值模拟验证了理论发现，包括展示对传感器测量误差响应的测试。


### 论文摘要

The stabilization of nonautonomous parabolic equations is achieved by feedback inputs tuning a finite number of actuators, where it is assumed that the input is subject to a time delay. To overcome destabilizing effects of the time delay, the input is based on a prediction of the state at a future time. This prediction is computed depending on a state-estimate at the current time, which in turn is provided by a Luenberger observer. The observer is designed using the output of measurements performed by a finite number of sensors. The asymptotic behavior of the resulting coupled system is investigated. Numerical simulations are presented validating the theoretical findings, including tests showing the response against sensor measurement errors.

---

## 111. Flow and Depth Assisted Video Prediction with Latent Transformer

**论文链接:** [http://arxiv.org/abs/2511.16484v1](http://arxiv.org/abs/2511.16484v1)

**作者:** Eliyas Suleyman, Paul Henderson, Eksan Firkat, Nicolas Pugeault

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究针对视频预测中的遮挡问题，提出通过整合点流和深度信息来改善模型性能，进行了首个系统性遮挡视频预测研究，发现整合这些信息的模型在遮挡场景和背景运动预测上表现更优。

### 背景

视频预测是机器人技术和世界建模等多种下游应用的基本任务。尽管通用视频预测模型在标准场景中取得了显著性能，但遮挡仍然是视频预测中一个固有的挑战。

### 目的

验证通过提供运动信息（点流）和几何结构信息（深度图）是否能够改善视频预测模型在遮挡和背景运动场景中的表现。

### 方法

使用标准的多对象潜在变换器架构预测未来帧，并修改该架构以整合深度和点流信息。在受控环境中对合成和真实世界数据集进行评估，同时使用基于外观的指标和对象掩码上的Wasserstein距离进行测量。

### 主要发现

当预测模型得到点流和深度信息的辅助时，在遮挡场景下表现更好，能够预测出更准确的背景运动，相比没有这些模态帮助的模型有显著提升。

### 结论

整合点流和深度信息可以显著改善视频预测模型在遮挡场景和背景运动预测中的性能，这对需要处理遮挡场景的实际应用具有重要意义。

### 翻译

视频预测是机器人技术和世界建模等多种下游应用的基本任务。尽管通用视频预测模型在标准场景中取得了显著性能，但遮挡仍然是视频预测中一个固有的挑战。我们假设通过明确提供运动信息（通过点流）和几何结构信息（通过深度图），可以使视频预测模型在遮挡和背景运动的情况下表现更好。为了研究这一点，我们进行了首个针对遮挡视频预测的系统性研究。我们使用标准的多对象潜在变换器架构来预测未来帧，但修改了该架构以整合来自深度和点流的信息。我们在受控环境中对合成和真实世界数据集评估了该模型，不仅使用了基于外观的指标，还使用了对象掩码上的Wasserstein距离，这可以有效预测运动分布。我们发现，当预测模型得到点流和深度信息的辅助时，在遮挡场景下表现更好，并且与没有这些模态帮助的模型相比，能预测出更准确的背景运动。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决视频预测中的遮挡问题。这个问题在现实中非常重要，因为遮挡是常见现象（如物体相互遮挡），准确预测遮挡情况下的视频对机器人导航、自动驾驶等应用至关重要，同时背景运动的准确预测对自移动物体的建模也很重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有视频预测方法在处理遮挡时的局限性，特别是光学流方法的误差累积和完全遮挡时信息丢失问题。他们设计了一个整合点流和深度信息的视频预测模型，借鉴了SCAT作为基础架构，使用Cotracker提取点流信息，DepthAnything-V2提取深度图，SAM2进行实例分割。作者设计了多种模型变体来测试不同模态组合的效果，并在合成和真实世界数据集上进行实验。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过整合点流和深度信息作为额外模态，增强视频预测模型在遮挡场景中的表现，使模型能够结合视觉信息、运动信息和空间信息。整体流程包括：1)数据预处理，使用Cotracker跟踪点计算点流，DepthAnything-V2估计深度图，SAM2进行实例分割；2)两阶段模型训练，第一阶段训练对象感知自编码器(OAAE)将多模态信息编码到潜在空间，第二阶段使用SCAT在潜在空间中预测未来帧；3)使用多种指标评估预测结果，包括外观指标和运动指标。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首次系统研究遮挡视频预测；2)设计能整合点流和深度的多模态视频预测模型；3)引入基于物体掩码Wasserstein距离的新评估方法。相比之前工作，本文使用点流而非光学流（更稳健不累积误差），不仅依赖RGB信息而是整合运动和几何信息，在潜在空间中处理多模态信息，并专门设计实验测试遮挡场景。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文通过整合点流和深度信息到潜在变换器架构中，显著提高了视频预测模型在遮挡场景和背景运动情况下的准确性，特别是在预测被遮挡物体的重新出现方面表现优异。'}


### 论文摘要

Video prediction is a fundamental task for various downstream applications, including robotics and world modeling. Although general video prediction models have achieved remarkable performance in standard scenarios, occlusion is still an inherent challenge in video prediction. We hypothesize that providing explicit information about motion (via point-flow) and geometric structure (via depth-maps) will enable video prediction models to perform better in situations with occlusion and the background motion. To investigate this, we present the first systematic study dedicated to occluded video prediction. We use a standard multi-object latent transformer architecture to predict future frames, but modify this to incorporate information from depth and point-flow. We evaluate this model in a controlled setting on both synthetic and real-world datasets with not only appearance-based metrics but also Wasserstein distances on object masks, which can effectively measure the motion distribution of the prediction. We find that when the prediction model is assisted with point flow and depth, it performs better in occluded scenarios and predicts more accurate background motion compared to models without the help of these modalities.

---

## 112. From percolation transition to Anderson localization in one-dimensional speckle potentials

**论文链接:** [http://arxiv.org/abs/2511.16460v1](http://arxiv.org/abs/2511.16460v1)

**作者:** Margaux Vrech, Jan Major, Dominique Delande, Marcel Filoche, Nicolas Cherroret

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究探讨了经典粒子与量子输运之间的半经典交叉区域，揭示了一维红色斑纹势中粒子传播行为如何从经典逾渗相变连续过渡到量子安德森局域化，并发现了标准DMPK描述失效及双模态传输分布等新现象。

### 背景

经典粒子在随机势中通常经历逾渗相变，被困在平均大小为χ的团簇中，在逾渗阈值处χ呈现代数发散。相比之下，量子输运由安德森局域化长度控制，在经典临界点处没有明显特征。标准DMPK描述适用于无关联的无序系统。

### 目的

通过研究一维红色斑纹势中粒子传播行为，对经典与量子这两个区域之间的半经典交叉区域进行全面的理论分析，特别关注势能的关联和非高斯性质如何影响输运行为。

### 方法

采用半经典方法研究一维红色斑纹势中的粒子传播，该势在其上界处存在逾渗相变。通过数值模拟和理论分析相结合的方式研究系统偏离经典极限时的行为。

### 主要发现

1. 随着系统偏离经典极限，χ的代数发散连续地连接到局域化长度的平滑但非解析增长；2. 在交叉区域，斑纹势的关联和非高斯性质导致标准DMPK描述失效；3. 预测会出现双模态传输分布，这种在通常一维系统中不存在的行为被半经典分析所捕捉；4. 在量子区域深处，DMPK框架得以恢复，安德森局域化的普适特征重新出现。

### 结论

经典与量子输运之间存在一个半经典交叉区域，其中关联和非高斯势能的性质改变了输运行为，导致出现新的现象。随着系统进一步进入量子区域，标准描述和安德森局域化的普适特征重新出现。

### 翻译

随机势中的经典粒子通常经历逾渗相变，被困在平均大小为χ的团簇中，在逾渗阈值处χ呈现代数发散。相比之下，随机势中的量子输运由安德森局域化长度控制，在经典临界点处没有明显特征。本文通过研究一维红色斑纹势中粒子的传播行为，对这两个区域之间的半经典交叉区域进行了全面的理论分析，该势在其上界处存在逾渗相变。随着系统偏离经典极限，我们发现χ的代数发散连续地连接到局域化长度的平滑但非解析增长。我们使用半经典方法在数值和理论上表征了这种行为。在此交叉区域，斑纹势的关联和非高斯性质变得至关重要，导致标准DMPK描述（适用于无关联无序）失效。相反，我们预测会出现双模态传输分布，这种行为在通常一维系统中不存在，我们在半经典分析中捕捉到了这一行为。在量子区域深处，DMPK框架得以恢复，安德森局域化的普适特征重新出现。


### 论文摘要

Classical particles in random potentials typically experience a percolation phase transition, being trapped in clusters of mean size $χ$ that diverges algebraically at a percolation threshold. In contrast, quantum transport in random potentials is controlled by the Anderson localization length, which shows no distinct feature at this classical critical point. Here, we present a comprehensive theoretical analysis of the semi-classical crossover between these two regimes by studying particle propagation in a one-dimensional, red speckle potential, which hosts a percolation transition at its upper bound. As the system deviates from the classical limit, we find that the algebraic divergence of $χ$ continuously connects to a smooth yet non-analytic increase of the localization length. We characterize this behavior both numerically and theoretically using a semi-classical approach. In this crossover regime, the correlated and non-Gaussian nature of the speckle potential becomes essential, causing the standard DMPK description for uncorrelated disorder to break down. Instead, we predict the emergence of a bimodal transmission distribution, a behavior normally absent in one dimension, which we capture within our semi-classical analysis. Deep in the quantum regime, the DMPK framework is recovered and the universal features of Anderson localization reappear.

---

## 113. Generative Modeling of Clinical Time Series via Latent Stochastic Differential Equations

**论文链接:** [http://arxiv.org/abs/2511.16427v1](http://arxiv.org/abs/2511.16427v1)

**作者:** Muhammad Aslanimoghanloo, Ahmed ElGazzar, Marcel van Gerven

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究提出了一种基于潜在神经随机微分方程的生成建模框架，用于处理临床时间序列数据中的不规则采样、复杂生理机制和不确定性问题，并在治疗效果估计和生理信号预测任务中表现出色。

### 背景

临床时间序列数据来自电子健康记录和医疗登记，提供了理解患者轨迹和指导医疗决策的空前机会。但利用这些数据面临显著挑战，包括不规则采样、复杂的潜在生理机制以及测量和疾病进展中的固有不确定性。

### 目的

开发一种能够处理临床时间序列数据中不规则采样、复杂生理机制和不确定性的建模框架，以支持临床决策。

### 方法

提出基于潜在神经随机微分方程(SDEs)的生成建模框架，将临床时间序列视为底层受控随机动力系统的离散时间部分观测。通过具有模态相关发射模型的神经SDEs建模潜在动态，并通过变分推断执行状态估计和参数学习。

### 主要发现

在两个互补任务上验证了框架：(i)使用模拟的肺癌药代动力学-药效学模型进行个体治疗效果估计；(ii)使用来自12,000名患者的真实世界ICU数据进行生理信号的概率预测。结果表明，该框架在准确性和不确定性估计方面优于常微分方程和长短期记忆基线模型。

### 结论

该框架能够自然处理不规则采样观测，学习复杂非线性相互作用，并在统一可扩展的概率框架内捕获疾病进展和测量噪声的随机性，具有支持临床决策的潜力。

### 翻译

来自电子健康记录和医疗登记的临床时间序列数据为理解患者轨迹和指导医疗决策提供了前所未有的机会。然而，由于不规则采样、复杂的潜在生理机制以及测量和疾病进展中的固有不确定性，利用此类数据带来了重大挑战。为解决这些挑战，我们提出了一种基于潜在神经随机微分方程(SDEs)的生成建模框架，该方法将临床时间序列视为底层受控随机动力系统的离散时间部分观测。我们的方法通过具有模态相关发射模型的神经SDEs对潜在动态进行建模，同时通过变分推断执行状态估计和参数学习。这种表述自然处理了不规则采样观测，学习了复杂的非线性相互作用，并在统一可扩展的概率框架内捕获了疾病进展和测量噪声的随机性。我们在两个互补任务上验证了该框架：(i)使用模拟的肺癌药代动力学-药效学(PKPD)模型进行个体治疗效果估计；(ii)使用来自12,000名患者的真实世界重症监护室(ICU)数据进行生理信号的概率预测。结果表明，我们的框架在准确性和不确定性估计方面优于常微分方程和长短期记忆基线模型。这些结果突显了该框架在支持临床决策方面实现精确、不确定性感知预测的潜力。


### 论文摘要

Clinical time series data from electronic health records and medical registries offer unprecedented opportunities to understand patient trajectories and inform medical decision-making. However, leveraging such data presents significant challenges due to irregular sampling, complex latent physiology, and inherent uncertainties in both measurements and disease progression. To address these challenges, we propose a generative modeling framework based on latent neural stochastic differential equations (SDEs) that views clinical time series as discrete-time partial observations of an underlying controlled stochastic dynamical system. Our approach models latent dynamics via neural SDEs with modality-dependent emission models, while performing state estimation and parameter learning through variational inference. This formulation naturally handles irregularly sampled observations, learns complex non-linear interactions, and captures the stochasticity of disease progression and measurement noise within a unified scalable probabilistic framework. We validate the framework on two complementary tasks: (i) individual treatment effect estimation using a simulated pharmacokinetic-pharmacodynamic (PKPD) model of lung cancer, and (ii) probabilistic forecasting of physiological signals using real-world intensive care unit (ICU) data from 12,000 patients. Results show that our framework outperforms ordinary differential equation and long short-term memory baseline models in accuracy and uncertainty estimation. These results highlight its potential for enabling precise, uncertainty-aware predictions to support clinical decision-making.

---

## 114. Tube-Based Model Predictive Control with Random Fourier Features for Nonlinear Systems

**论文链接:** [http://arxiv.org/abs/2511.16425v1](http://arxiv.org/abs/2511.16425v1)

**作者:** Ákos M. Bokor, Tamás Dózsa, Felix Biertümpfel, Ádám Szabó

**发布时间:** 2025-11-20

**备注:** Submitted to IEEE IV 2026, The IEEE Intelligent Vehicles Symposium

### GPT解析

### 总结

本文提出了一种结合随机傅里叶特征与基于管(tube)的模型预测控制的计算高效方法，用于非线性系统的鲁棒控制。

### 背景

基于管的模型预测控制能够在由近似误差和外部干扰引起的有界模型不确定性下提供鲁棒的约束满足。随机傅里叶特征方法通过解决数值上可处理的最小二乘问题来近似非线性系统动力学，从而减少近似误差。

### 目的

开发将基于随机傅里叶特征的残差学习与管模型预测控制集成的算法，并展示其在自主车辆路径跟踪问题中的应用。

### 方法

结合随机傅里叶特征(RFF)与基于管的模型预测控制方法，其中RFF用于近似非线性系统动力学，管MPC提供鲁棒性保证。

### 主要发现

与线性基准相比，所提出的方法将管尺寸减少了约50%，导致行为保守性降低，并在测试场景中产生约70%更小的误差。此外，该方法实现了实时性能，同时保持了可证明的鲁棒性保证。

### 结论

该方法在保持计算效率的同时，提高了非线性系统控制的鲁棒性和性能。

### 翻译

本文通过结合随机傅里叶特征与基于管的模型预测控制，提出了一种用于非线性系统的计算高效鲁棒控制方法。基于管的模型预测控制能够在由近似误差和外部干扰引起的有界模型不确定性下提供鲁棒的约束满足。随机傅里叶特征方法通过解决一个数值上可处理的二乘问题来近似非线性系统动力学，从而减少近似误差。我们开发了基于随机傅里叶特征的残差学习与管模型预测控制的集成方法，并展示了其在使用非线性自行车模型的自主车辆路径跟踪问题中的应用。与线性基准相比，所提出的方法将管尺寸减少了约50%，导致行为保守性降低，并在测试场景中产生约70%更小的误差。此外，所提出的方法实现了实时性能，同时保持了可证明的鲁棒性保证。


### 论文摘要

This paper presents a computationally efficient approach for robust Model Predictive Control of nonlinear systems by combining Random Fourier Features with tube-based MPC. Tube-based Model Predictive Control provides robust constraint satisfaction under bounded model uncertainties arising from approximation errors and external disturbances. The Random Fourier Features method approximates nonlinear system dynamics by solving a numerically tractable least-squares problem, thereby reducing the approximation error. We develop the integration of RFF-based residual learning with tube MPC and demonstrate its application to an autonomous vehicle path-tracking problem using a nonlinear bicycle model. Compared to the linear baseline, the proposed method reduces the tube size by approximately 50%, leading to less conservative behavior and resulting in around 70% smaller errors in the test scenario. Furthermore, the proposed method achieves real-time performance while maintaining provable robustness guarantees.

---

## 115. Prediction of atomic H adsorption energies in metalloid doped MSSe (M = Mo/W) Janus layers: A combined DFT and machine learning study

**论文链接:** [http://arxiv.org/abs/2511.16263v1](http://arxiv.org/abs/2511.16263v1)

**作者:** G. Tejaswini, Anjana E Sudheer, Amrendra Kumar, M. Vallinayagam, Pavan Kumar Perepu, Attila Cangi, Mani Lokamani, M. Posselt, M. Zschornak, C. Kamal, D. Amaranatha Reddy, D. Murali

**发布时间:** 2025-11-20

**备注:** 19 pages, 11 figures

### GPT解析

### 总结

研究掺杂B、Si、Ge金属loid的MSSe Janus材料中氢原子吸附行为，通过DFT计算和机器学习模型预测吸附能，发现掺杂可改善光催化性能。

### 背景

MSSe Janus材料作为2H MX2的衍生物已在实验中实现，并探索了其在光催化、光伏和光电器件中的应用，本研究特别关注其光催化性质。

### 目的

研究在B、Si、Ge金属loid掺杂剂存在下，原子氢在MSSe层上的吸附情况，以改善材料的光催化性能。

### 方法

使用密度泛函理论(DFT)计算氢原子在不同位点的吸附能；选择原子和间隙位点进行掺杂剂替代；开发监督机器学习模型，利用23个元素特征预测氢吸附能；应用主成分分析(PCA)降维；实现多层感知器回归器模型；使用数据增强技术扩大数据集。

### 主要发现

原始MSSe层具有正吸附能，表明是吸热过程；掺杂改变了局部对称性、键合特性和电荷分布，增加了活性位点并降低了吸附能；原子位点掺杂使吸附过程自发性增强，间隙位点导致吸热行为；机器学习模型提高了测试数据准确率0.90%。

### 结论

掺杂可以改善MSSe Janus材料的光催化性能，特别是通过改变氢吸附特性；机器学习模型能有效预测氢吸附能，减少对DFT计算的依赖。

### 翻译

摘要已翻译为中文，内容涵盖了MSSe Janus材料在掺杂条件下的氢吸附行为研究，包括DFT计算结果和机器学习模型的开发与应用。


### 论文摘要

Janus derivatives of 2H MX2 (M = Mo/W; X = S/Se), namely MSSe, have already been experimentally realized and explored for applications in photocatalysis, photovoltaics, and optoelectronics. Focusing on the photocatalytic properties of these layers, we investigate the adsorption of atomic hydrogen on the MSSe layers in the presence of metalloid dopants B, Si, and Ge. The layers in their pristine form exhibit positive adsorption energies, indicating an endothermic nature. Substitution of a dopant in the pristine MSSe layers alters the local symmetry, bonding character, and charge distribution, thereby increasing the number of active sites for hosting H adsorption and reducing the adsorption energy. We select distinct sites, both atomic and interstitial, for the substitution of dopants. The energetics of the H atom at various sites is studied to find the most favorable active site on the MSSe Janus layers. Our results based on density functional theory calculations show that the adsorption process becomes spontaneous and less attractive in the presence of atomic site dopant substitution, whereas the interstitial site results in an endothermic behavior. Moreover, having the data from DFT, we develop a supervised machine learning model for predicting the hydrogen adsorption energy. For this purpose, we utilize 23 elemental features of the atoms involved in the structure, thereby eliminating the need for DFT calculations in feature design. The dimensionality reduction technique, principal component analysis, is employed to reduce the dimensionality of the feature space, yielding independent features that are mutually orthogonal. The model is implemented as a multi-layer perceptron regressor with two hidden layers. The data augmentation technique is employed to artificially expand the dataset size, thereby enhancing the accuracy of the neural network model by 0.90% on the testing data.

---

## 116. FOOTPASS: A Multi-Modal Multi-Agent Tactical Context Dataset for Play-by-Play Action Spotting in Soccer Broadcast Videos

**论文链接:** [http://arxiv.org/abs/2511.16183v1](http://arxiv.org/abs/2511.16183v1)

**作者:** Jeremie Ochin, Raphael Chekroun, Bogdan Stanciulescu, Sotiris Manitsaris

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了FOOTPASS数据集，这是第一个针对整个足球比赛的逐场动作定位基准，结合了计算机视觉和足球战术知识，用于实现更自动化、可靠的逐场数据提取。

### 背景

足球视频理解推动了用于时序动作定位、时空动作检测和多目标跟踪等任务的数据集创建。足球分析中使用的结构化事件序列(谁在何时何地做了什么)的标注需要整合STAD和MOT的整体方法，但当前动作识别方法不足以构建可靠的逐场数据。

### 目的

利用战术知识作为先验来支持基于计算机视觉的预测，实现更自动化、可靠的逐场数据提取。

### 方法

引入FOOTPASS数据集，支持以球员为中心的动作定位方法，利用计算机视觉任务(如跟踪、识别)的输出和足球的先验知识，包括长期时间范围内的战术规律，生成可靠的逐场数据流。

### 主要发现

当前动作识别方法不足以构建可靠的逐场数据，通常用于辅助而非完全自动化标注。

### 结论

生成的逐场数据流形成数据驱动体育分析的关键输入。

### 翻译

足球视频理解推动了用于时序动作定位、时空动作检测(STAD)或多目标跟踪(MOT)等任务的数据集创建。用于足球分析的结构化事件序列(谁在何时何地做了什么)的标注需要整合STAD和MOT的整体方法。然而，当前动作识别方法仍不足以构建可靠的逐场数据，通常用于辅助而非完全自动化标注。平行研究推进了战术建模、轨迹预测和性能分析，这些都基于比赛状态和逐场数据。这促使利用战术知识作为先验来支持基于计算机视觉的预测，实现更自动化、可靠的逐场数据提取。我们引入了FOOTPASS数据集，这是第一个针对整个足球比赛在多模态、多智能体战术背景下的逐场动作定位基准。它支持以球员为中心的动作定位方法，利用计算机视觉任务(如跟踪、识别)的输出和足球的先验知识(包括长期时间范围内的战术规律)，生成可靠的逐场数据流。这些数据流形成数据驱动体育分析的关键输入。


### 论文摘要

Soccer video understanding has motivated the creation of datasets for tasks such as temporal action localization, spatiotemporal action detection (STAD), or multiobject tracking (MOT). The annotation of structured sequences of events (who does what, when, and where) used for soccer analytics requires a holistic approach that integrates both STAD and MOT. However, current action recognition methods remain insufficient for constructing reliable play-by-play data and are typically used to assist rather than fully automate annotation. Parallel research has advanced tactical modeling, trajectory forecasting, and performance analysis, all grounded in game-state and play-by-play data. This motivates leveraging tactical knowledge as a prior to support computer-vision-based predictions, enabling more automated and reliable extraction of play-by-play data. We introduce Footovision Play-by-Play Action Spotting in Soccer Dataset (FOOTPASS), the first benchmark for play-by-play action spotting over entire soccer matches in a multi-modal, multi-agent tactical context. It enables the development of methods for player-centric action spotting that exploit both outputs from computer-vision tasks (e.g., tracking, identification) and prior knowledge of soccer, including its tactical regularities over long time horizons, to generate reliable play-by-play data streams. These streams form an essential input for data-driven sports analytics.

---

## 117. Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective

**论文链接:** [http://arxiv.org/abs/2511.16170v1](http://arxiv.org/abs/2511.16170v1)

**作者:** Jiahao Li, Yang Lu, Yachao Zhang, Yong Xie, Fangyong Wang, Yuan Xie, Yanyun Qu

**发布时间:** 2025-11-20

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

这项研究针对开放词汇语义分割中的像素级多模态对齐挑战，发现CLIP模型在密集预测时会出现类似人类注意力分散的现象，通过提出ReFocusing CLIP方法重新引导注意力，实现了更精准的多模态对齐。

### 背景

开放词汇语义分割采用像素级的视觉-语言对齐技术，将类别提示与对应像素关联。虽然现有方法利用CLIP的视觉-语言对齐取得了不错效果，但很少从可解释性角度研究CLIP在密集预测中的性能边界。

### 目的

系统研究CLIP的内部机制，识别其在密集预测中的注意力分散现象，并提出方法增强CLIP的多模态对齐粒度，提高密集预测能力。

### 方法

提出ReFocusing CLIP (RF-CLIP)，一种无需训练的方法，模拟人类分散注意力-重新聚焦的行为，通过过滤导致注意力分散的维度特异性过度激活令牌，将注意力从无关令牌重新引导回目标区域。

### 主要发现

1) CLIP在密集预测时会分散注意力到无关令牌，类似人类注意力分散；2) 这些分散注意力的令牌源于维度特异性的过度激活；3) 过滤这些令牌可提升CLIP的密集预测性能；4) RF-CLIP方法能有效重新引导注意力，提高多模态对齐粒度。

### 结论

ReFocusing CLIP方法通过模拟人类的注意力重新聚焦行为，解决了CLIP在密集预测中的注意力分散问题，在八个基准测试上实现了最先进性能，同时保持了较高的推理效率。

### 翻译

开放词汇语义分割采用像素级的视觉-语言对齐，将类别相关提示与相应像素关联起来。一个关键挑战是增强多模态密集预测能力，特别是这种像素级的多模态对齐。尽管现有方法通过利用CLIP的视觉-语言对齐取得了有希望的结果，但很少从可解释性机制的角度研究CLIP在密集预测中的性能边界。在这项工作中，我们系统研究了CLIP的内部机制，并确定了一个关键现象：类似于人类的注意力分散，CLIP会将大量注意力资源从目标区域分散到无关令牌上。我们的分析表明，这些令牌源于维度特异性的过度激活；过滤它们可以增强CLIP的密集预测性能。因此，我们提出了ReFocusing CLIP (RF-CLIP)，一种无需训练的方法，模拟人类分散注意力-重新聚焦的行为，将注意力从分散令牌重新引导回目标区域，从而完善CLIP的多模态对齐粒度。我们的方法在八个基准测试上实现了最先进的性能，同时保持了较高的推理效率。


### 论文摘要

Open-vocabulary semantic segmentation (OVSS) employs pixel-level vision-language alignment to associate category-related prompts with corresponding pixels. A key challenge is enhancing the multimodal dense prediction capability, specifically this pixel-level multimodal alignment. Although existing methods achieve promising results by leveraging CLIP's vision-language alignment, they rarely investigate the performance boundaries of CLIP for dense prediction from an interpretability mechanisms perspective. In this work, we systematically investigate CLIP's internal mechanisms and identify a critical phenomenon: analogous to human distraction, CLIP diverts significant attention resources from target regions to irrelevant tokens. Our analysis reveals that these tokens arise from dimension-specific over-activation; filtering them enhances CLIP's dense prediction performance. Consequently, we propose ReFocusing CLIP (RF-CLIP), a training-free approach that emulates human distraction-refocusing behavior to redirect attention from distraction tokens back to target regions, thereby refining CLIP's multimodal alignment granularity. Our method achieves SOTA performance on eight benchmarks while maintaining high inference efficiency.

---

## 118. VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation

**论文链接:** [http://arxiv.org/abs/2511.16124v1](http://arxiv.org/abs/2511.16124v1)

**作者:** Chenyang Wu, Jiayi Fu, Chun-Le Guo, Shuhao Han, Chongyi Li

**发布时间:** 2025-11-20

**备注:** Accepted by AAAI 2026

### GPT解析

### 总结

本文提出了一种名为VTinker的新型视频帧插值(VFI)流水线，解决了高分辨率光流估计中的模糊、马赛克、鬼影和不连续性问题。

### 背景

高分辨率帧的运动估计具有挑战性，因为像素移动大且计算成本高。大多数基于光流的VFI方法先在低分辨率预测光流再上采样，但这种方法会导致光流边缘模糊或马赛克，且无法充分捕捉高分辨率下的精细像素运动。

### 目的

开发一种新的VFI方法，解决低分辨率光流上采样导致的模糊和细节丢失问题，以及像素级鬼影和不连续性问题。

### 方法

VTinker包含两个核心组件：引导光流上采样(GFU)和纹理映射。GFU利用输入帧作为指导减轻双线性上采样的模糊；纹理映射生成中间代理帧，并从输入帧选择清晰纹理块映射到代理上，通过重建模块生成最终插值帧。

### 主要发现

实验证明VTinker在VFI任务中取得了最先进的表现，有效解决了高分辨率光流估计中的关键问题。

### 结论

VTinker通过创新的光流上采样和纹理映射方法，显著提高了视频帧插值的质量，特别是在处理高分辨率视频时表现优异。

### 翻译

由于像素移动量大和计算成本高，估计高分辨率帧的运动具有挑战性。因此，大多数基于光流的视频帧插值(VFI)方法首先在低分辨率预测双向光流，然后使用高倍放大(如双线性)获得高分辨率光流。然而，这种上采样策略可能导致光流边缘模糊或马赛克。此外，低分辨率运动估计无法充分捕捉高分辨率下精细像素的运动，导致任务导向光流不对齐。使用这些不准确的光流，输入帧被逐像素扭曲和组合，在插值帧中产生鬼影和不连续性。在本研究中，我们提出了一种新颖的VFI流水线VTinker，包含两个核心组件：引导光流上采样(GFU)和纹理映射。在低分辨率运动估计后，GFU引入输入帧作为指导，减轻双线性上采样光流中的模糊细节，使光流边缘更清晰。随后，为了避免像素级鬼影和不连续性，纹理映射生成一个初始插值帧，称为中间代理。代理作为从输入帧中选择清晰纹理块的线索，然后将这些纹理块映射到代理上，通过重建模块生成最终的插值帧。大量实验证明VTinker在VFI方面取得了最先进的表现。代码可在以下网址获取：https://github.com/Wucy0519/VTinker。


### 论文摘要

Due to large pixel movement and high computational cost, estimating the motion of high-resolution frames is challenging. Thus, most flow-based Video Frame Interpolation (VFI) methods first predict bidirectional flows at low resolution and then use high-magnification upsampling (e.g., bilinear) to obtain the high-resolution ones. However, this kind of upsampling strategy may cause blur or mosaic at the flows' edges. Additionally, the motion of fine pixels at high resolution cannot be adequately captured in motion estimation at low resolution, which leads to the misalignment of task-oriented flows. With such inaccurate flows, input frames are warped and combined pixel-by-pixel, resulting in ghosting and discontinuities in the interpolated frame. In this study, we propose a novel VFI pipeline, VTinker, which consists of two core components: guided flow upsampling (GFU) and Texture Mapping. After motion estimation at low resolution, GFU introduces input frames as guidance to alleviate the blurring details in bilinear upsampling flows, which makes flows' edges clearer. Subsequently, to avoid pixel-level ghosting and discontinuities, Texture Mapping generates an initial interpolated frame, referred to as the intermediate proxy. The proxy serves as a cue for selecting clear texture blocks from the input frames, which are then mapped onto the proxy to facilitate producing the final interpolated frame via a reconstruction module. Extensive experiments demonstrate that VTinker achieves state-of-the-art performance in VFI. Codes are available at: https://github.com/Wucy0519/VTinker.

---

## 119. Future-Back Threat Modeling: A Foresight-Driven Security Framework

**论文链接:** [http://arxiv.org/abs/2511.16088v1](http://arxiv.org/abs/2511.16088v1)

**作者:** Vu Van Than

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了前向-后向威胁建模(FBTM)理论和方法，从未来威胁状态反向识别当前防御架构的漏洞，帮助应对新兴威胁。

### 背景

传统威胁建模专注于已知的TTPs和过去事件数据，而威胁预测框架与操作或架构工件脱节，导致无法有效应对来自未来的新兴威胁。

### 目的

解决传统威胁建模的局限性，通过前向-后向方法识别当前防御架构中的假设、差距、盲点和漏洞，提高对未来威胁的预测能力。

### 方法

前向-后向威胁建模(FBTM)，从设想的未来威胁状态开始，反向分析当前防御架构，揭示已知未知和未知未知，包括新兴、预期和合理的战术、技术和程序。

### 主要发现

最严重的网络威胁往往来自未知、被忽视或尚未构想的事物，常来自AI、信息战和供应链攻击等未来领域；FBTM方法能增强对对手行为的可预测性，特别是在未来不确定性情况下。

### 结论

通过FBTM方法，安全领导者可以做出明智决策，为未来构建更有弹性的安全态势，主动应对而非被动反应新兴威胁。

### 翻译

传统威胁建模仍然是反应性的，专注于已知的TTPs和过去的事件数据，而威胁预测和预测框架通常与操作或架构工件脱节。这造成了一个根本弱点：最严重的网络威胁往往不是来自已知的事物，而是来自被假设、忽视或尚未构想的事物，并且经常来自未来，如人工智能、信息战和供应链攻击，在这些领域中，对手不断开发新的漏洞利用方法，可以绕过基于当前知识构建的防御。为了解决这一思维差距，本文引入了前向-后向威胁建模(FBTM)的理论和方法论。这种预测方法从设想的未来威胁状态开始，反向识别当前防御架构中的假设、差距、盲点和漏洞，提供对即将到来的威胁更清晰、更准确的看法，这样我们就可以预见其出现并通过现在采取的行动塑造我们想要的未来。提出的方法还旨在揭示已知未知和未知未知，包括新兴、预期和合理的战术、技术和程序。这增强了对手行为的可预测性，特别是在未来不确定性方面，帮助安全领导者做出明智决策，为未来塑造更有弹性的安全态势。


### 论文摘要

Traditional threat modeling remains reactive-focused on known TTPs and past incident data, while threat prediction and forecasting frameworks are often disconnected from operational or architectural artifacts. This creates a fundamental weakness: the most serious cyber threats often do not arise from what is known, but from what is assumed, overlooked, or not yet conceived, and frequently originate from the future, such as artificial intelligence, information warfare, and supply chain attacks, where adversaries continuously develop new exploits that can bypass defenses built on current knowledge. To address this mental gap, this paper introduces the theory and methodology of Future-Back Threat Modeling (FBTM). This predictive approach begins with envisioned future threat states and works backward to identify assumptions, gaps, blind spots, and vulnerabilities in the current defense architecture, providing a clearer and more accurate view of impending threats so that we can anticipate their emergence and shape the future we want through actions taken now. The proposed methodology further aims to reveal known unknowns and unknown unknowns, including tactics, techniques, and procedures that are emerging, anticipated, and plausible. This enhances the predictability of adversary behavior, particularly under future uncertainty, helping security leaders make informed decisions today that shape more resilient security postures for the future.

---

## 120. Nature of 2D XY antiferromagnetism in a van der Waals monolayer

**论文链接:** [http://arxiv.org/abs/2511.16065v1](http://arxiv.org/abs/2511.16065v1)

**作者:** Cheol-Yeon Cheon, Volodymyr Multian, Kenji Watanabe, Takashi Taniguchi, Alberto F. Morpurgo, Dmitry Lebedev

**发布时间:** 2025-11-20

**DOI:** 10.1038/s41467-025-66672-1

**备注:** 18 pages, 4 figures in main text; 6 pages, 11 figures in Supplementary Information. Accepted in Nature Communications (2025)

### GPT解析

### 总结

本研究通过磁输运测量研究了原子级薄的范德华磁体NiPS3中的反铁磁性，发现单层NiPS3与较厚样品表现出不同的磁行为，单层经历两个磁相变，而双层和多层只有一个相变。

### 背景

二维反铁磁性在凝聚态物理领域长期受到关注，但由于范德华反铁磁单层的分离，实验探索直到最近才变得可行。然而，探索这些单层的磁相图仍具挑战性，因为既有的实验技术通常缺乏所需的灵敏度。

### 目的

研究原子级薄的范德华磁体NiPS3中的反铁磁性，特别是通过磁输运测量来探索其磁相图。

### 方法

在场效应晶体管器件中使用磁输运测量，分析温度依赖电导和磁阻数据来研究NiPS3的磁性行为。

### 主要发现

单层NiPS3与较厚样品表现出明显不同的磁行为；双层和多层NiPS3表现出单一磁相变，进入由单轴各向异性驱动的之字形反铁磁态；单层NiPS3经历两个磁相变，低温相由面内六角磁各向异性控制；实验构建的单层NiPS3相图与包含六角各向异性的六态时钟和2D-XY模型的预测相符。

### 结论

通过磁输运测量成功表征了单层NiPS3的磁性相变行为，证实了单层与较厚样品在磁性行为上的显著差异，实验结果与理论预测相符。

### 翻译

二维反铁磁性长期以来在凝聚态物理的许多领域引起了广泛关注，但由于范德华反铁磁单层的成功分离，实验探索直到最近才变得可行。然而，探索这些单层的磁相图仍然具有挑战性，因为既有的实验技术通常缺乏所需的灵敏度。在这里，我们使用场效应晶体管器件中的磁输运测量研究了原子级薄的范德华磁体NiPS3中的反铁磁性。温度依赖电导和磁阻数据揭示了与较厚样品相比，单层具有明显的不同磁行为。虽然双层和多层NiPS3表现出进入由单轴各向异性驱动的之字形反铁磁态的单一磁相变，但单层NiPS3经历两个磁相变，低温相由面内六角磁各向异性控制。为单层NiPS3构建的实验相图与包含六角各向异性的六态时钟和2D-XY模型的预测相符。


### 论文摘要

Two-dimensional antiferromagnetism has long attracted significant interest in many areas of condensed matter physics, but only recently has experimental exploration become feasible due to the isolation of van der Waals antiferromagnetic monolayers. Probing the magnetic phase diagram of these monolayers remains however challenging because established experimental techniques often lack the required sensitivity. Here, we investigate antiferromagnetism in atomically thin van der Waals magnet NiPS3 using magnetotransport measurements in field-effect transistor devices. Temperature-dependent conductance and magnetoresistance data reveal a distinct magnetic behavior in monolayers as compared to thicker samples. While bilayer and multilayer NiPS3 exhibit a single magnetic phase transition into a zig-zag antiferromagnetic state driven by uniaxial anisotropy, monolayer NiPS3 undergoes two magnetic transitions, with a low-temperature phase governed by in-plane hexagonal magnetic anisotropy. The experimentally constructed phase diagram for monolayer NiPS3 matches theoretical predictions from the six-state clock and 2D-XY models incorporating hexagonal anisotropy.

---

## 121. Hiding in the AI Traffic: Abusing MCP for LLM-Powered Agentic Red Teaming

**论文链接:** [http://arxiv.org/abs/2511.15998v1](http://arxiv.org/abs/2511.15998v1)

**作者:** Strahinja Janjuesvic, Anna Baron Garcia, Sohrob Kazerounian

**发布时间:** 2025-11-20

**备注:** 23 pages, 9 figures, 3 tables. Submitted as a full paper for review

### GPT解析

### 总结

该研究介绍了一种利用模型上下文协议(MCP)的新型命令与控制(C2)架构，用于协调分布式、自适应的侦察代理，在网络安全领域实现隐蔽操作。该架构显著改善了系统的目标导向行为，并消除了可检测命令与控制行为的关键主机和网络工件。

### 背景

生成式AI正在改变进攻性网络安全，使自主红队能够在渗透测试中规划、执行和适应。然而，现有方法在通用性和专业性之间面临权衡，实际部署存在幻觉、上下文限制和伦理问题等挑战。

### 目的

开发一种基于MCP的C2架构，以协调跨网络的分布式、自适应侦察代理，实现隐蔽操作，并克服现有方法的局限性。

### 方法

对最先进的生成式红队方法进行全面回顾，分析从微调专业模型到模块化或智能体框架的自动化能力与任务特定准确性。基于MCP的C2架构实现异步、并行操作和实时情报共享，无需周期性信标，并探索其高级对抗能力和检测规避技术。

### 主要发现

该架构不仅改善了整个系统的目标导向行为，还消除了可用于检测和防止命令与控制行为的主机和网络关键工件。实验显示与传统C2相比，手动工作和检测足迹大幅减少。

### 结论

所提出的MCP支持的C2框架向现实、AI驱动的红队行动迈出了重要一步，能够模拟高级持续性威胁，同时为下一代防御系统的开发提供信息。未来方向包括整合自主利用、防御性LLM智能体、预测规避机动和多智能体集群。

### 翻译

生成式AI正在通过启用能够规划、执行和适应渗透测试的自主红队代理来重塑进攻性网络安全。然而，现有方法在通用性和专业性之间面临权衡，实际部署显示存在幻觉、上下文限制和伦理问题等挑战。在这项工作中，我们引入了一种利用模型上下文协议(MCP)的新型命令与控制(C2)架构，用于协调跨网络的分布式、自适应侦察代理进行隐蔽操作。值得注意的是，我们发现我们的架构不仅改善了整个系统的目标导向行为，还消除了可用于检测和防止命令与控制行为的关键主机和网络工件。我们从微调专业模型到模块化或智能体框架，对最先进的生成式红队方法进行全面回顾，分析其针对任务特定准确性的自动化能力。然后，我们详细阐述了基于MCP的C2如何通过实现异步、并行操作和实时情报共享而不需要周期性信标来克服当前限制。此外，我们探索了该架构的高级对抗能力、其检测规避技术，并解决了双重使用伦理问题，提出了防御措施和实验室环境中的受控评估。与传统C2的实验比较显示，手动工作和检测足迹大幅减少。我们最后总结了未来方向，包括整合自主利用、防御性LLM智能体、预测规避机动和多智能体集群。所提出的MCP支持的C2框架向现实、AI驱动的红队行动迈出了重要一步，可以模拟高级持续性威胁，同时为下一代防御系统的开发提供信息。


### 论文摘要

Generative AI is reshaping offensive cybersecurity by enabling autonomous red team agents that can plan, execute, and adapt during penetration tests. However, existing approaches face trade-offs between generality and specialization, and practical deployments reveal challenges such as hallucinations, context limitations, and ethical concerns. In this work, we introduce a novel command & control (C2) architecture leveraging the Model Context Protocol (MCP) to coordinate distributed, adaptive reconnaissance agents covertly across networks. Notably, we find that our architecture not only improves goal-directed behavior of the system as whole, but also eliminates key host and network artifacts that can be used to detect and prevent command & control behavior altogether. We begin with a comprehensive review of state-of-the-art generative red teaming methods, from fine-tuned specialist models to modular or agentic frameworks, analyzing their automation capabilities against task-specific accuracy. We then detail how our MCP-based C2 can overcome current limitations by enabling asynchronous, parallel operations and real-time intelligence sharing without periodic beaconing. We furthermore explore advanced adversarial capabilities of this architecture, its detection-evasion techniques, and address dual-use ethical implications, proposing defensive measures and controlled evaluation in lab settings. Experimental comparisons with traditional C2 show drastic reductions in manual effort and detection footprint. We conclude with future directions for integrating autonomous exploitation, defensive LLM agents, predictive evasive maneuvers, and multi-agent swarms. The proposed MCP-enabled C2 framework demonstrates a significant step toward realistic, AI-driven red team operations that can simulate advanced persistent threats while informing the development of next-generation defensive systems.

---

## 122. Quantum field theory approach to neutrino oscillations in dark matter and implications at JUNO

**论文链接:** [http://arxiv.org/abs/2511.15494v2](http://arxiv.org/abs/2511.15494v2)

**作者:** Wei Chao

**发布时间:** 2025-11-19

**备注:** 12 pages, 1 figure

### GPT解析

### 总结

本文研究了标量型超轻暗物质中质量中微子的物质效应，使用量子场论方法计算中微子振荡概率，并讨论了朱诺实验的相关预测。

### 背景

中微子振荡是一个重要的物理过程，值得深入研究。

### 目的

研究标量型超轻暗物质中质量中微子的物质效应，并计算中微子振荡概率。

### 方法

使用量子场论方法进行计算。

### 主要发现

量子场论方法推导的中微子振荡概率没有额外的时间依赖性，这是与量子力学方法获得的中微子振荡结果最显著的区别。

### 结论

这项研究扩展了对中微子与暗物质相互作用的理解，值得进一步探索。

### 翻译

中微子振荡是一个值得深入探索的重要物理过程。本文研究了标量型超轻暗物质中质量中微子的物质效应，并使用量子场论方法计算了中微子振荡概率。结果表明，通过量子场论方法推导的中微子振荡概率没有表现出额外的时间依赖性，这是与通过量子力学方法获得的中微子振荡结果最显著的区别。此外，我们还讨论了朱诺实验关于标量型超轻暗物质中中微子振荡行为的预测。这项研究扩展了对中微子与暗物质相互作用的理解，值得进一步探索。


### 论文摘要

Neutrino oscillation is a significant physical process worthy of in-depth exploration. In this paper, we investigate the matter effect of massive neutrinos in a scalar-type ultra-light dark matter and calculate the neutrino oscillation probability using the quantum field theory method. The result reveals that the neutrino oscillation probability derived from the quantum field theory approach exhibits no additional time dependence, which marks the most significant distinction from the oscillation result obtained through the quantum mechanics method. Furthermore, we discuss predictions of the Juno experiment regarding neutrino oscillation behavior in scalar-type ultra-light dark matter. This study extends the understanding of the interaction between neutrinos and dark matter, which warrants further exploration.

---

## 123. Prediction of Retention Time in Larger Antisense Oligonucleotide Datasets using Machine Learning

**论文链接:** [http://arxiv.org/abs/2511.15753v1](http://arxiv.org/abs/2511.15753v1)

**作者:** Manal Rahal, Bestoun S. Ahmed, Christoph A. Bauer, Johan Ulander, Jorgen Samuelsson

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究应用机器学习方法预测反义寡核苷酸在离子对液相色谱中的保留时间，解决ASOs生产纯化过程中的挑战。研究评估了四种机器学习模型，发现梯度提升模型在准确性和效率方面表现优异，同时确定了新的特征变量能提高预测能力。

### 背景

反义寡核苷酸(ASOs)是具有变革性治疗潜力的核酸分子，尤其对传统药物无法治疗的疾病。然而，ASOs的生产和纯化因存在不需要的杂质而具有挑战性。离子对液相色谱(IPC)是分离ASO化合物与杂质的关键工具，每个化合物通过其在IPC中的保留时间(tR)来识别。由于ASOs的复杂序列依赖性行为和色谱条件的变异性，准确预测tR是一项困难任务。

### 目的

应用机器学习(ML)基于ASOs的序列特征来预测tR，以解决色谱条件变化和序列依赖性行为带来的预测挑战。

### 方法

评估了四种机器学习模型：梯度提升、随机森林、决策树和支持向量回归。研究在三个具有不同梯度时间的大型ASO数据集上进行测试。通过特征工程和网格搜索优化，确定了关键预测因子，并使用均方根误差、决定系数R平方和运行时间比较了模型准确性。

### 主要发现

梯度提升模型在三个数据集中的两个中与支持向量机性能相当，但调整速度快3.94倍；新提出的代表硫计数和序列第一和最后位置的核苷酸的特征提高了模型的预测能力；基于机器学习的tR预测在大规模应用中具有优势。

### 结论

该研究证明了基于机器学习的tR预测在大规模应用中的优势，并为在色谱应用中可解释和高效地利用机器学习提供了见解。

### 翻译

反义寡核苷酸(ASOs)是具有变革性治疗潜力的核酸分子，特别是对传统药物无法治疗的疾病。然而，由于存在不需要的杂质，ASOs的生产和纯化仍然具有挑战性。离子对液相色谱(IPC)是成功用于将ASO化合物与杂质分离的工具，它是分离中的关键步骤，其中每个化合物通过其在IPC中的保留时间(tR)来识别。由于ASOs的复杂序列依赖性行为和色谱条件的变异性，准确预测tR是一项困难任务。本研究通过应用机器学习(ML)基于ASOs的序列特征来预测tR，解决了这一挑战。在三个具有不同梯度时间的大型ASO数据集上评估了四种机器学习模型：梯度提升、随机森林、决策树和支持向量回归。通过特征工程和网格搜索优化，确定了关键预测因子，并使用均方根误差、决定系数R平方和运行时间比较了模型准确性。结果表明，梯度提升性能在三个数据集中的两个中与支持向量机相当，但调整速度快3.94倍。此外，新提出的代表硫计数和序列第一和最后位置的核苷酸的特征被发现提高了模型的预测能力。该研究证明了基于机器学习的tR预测在大规模应用中的优势，并为在色谱应用中可解释和高效地利用机器学习提供了见解。


### 论文摘要

Antisense oligonucleotides (ASOs) are nucleic acid molecules with transformative therapeutic potential, especially for diseases that are untreatable by traditional drugs. However, the production and purification of ASOs remain challenging due to the presence of unwanted impurities. One tool successfully used to separate an ASO compound from the impurities is ion pair liquid chromatography (IPC). It is a critical step in separation, where each compound is identified by its retention time (tR) in the IPC. Due to the complex sequence-dependent behavior of ASOs and variability in chromatographic conditions, the accurate prediction of tR is a difficult task. This study addresses this challenge by applying machine learning (ML) to predict tR based on the sequence characteristics of ASOs. Four ML models Gradient Boosting, Random Forest, Decision Tree, and Support Vector Regression were evaluated on three large ASO datasets with different gradient times. Through feature engineering and grid search optimization, key predictors were identified and compared for model accuracy using root mean square error, coefficient of determination R-squared, and run time. The results showed that Gradient Boost performance competes with the Support Vector Machine in two of the three datasets, but is 3.94 times faster to tune. Additionally, newly proposed features representing the sulfur count and the nucleotides residing at the first and last positions of a sequence were found to improve the predictive power of the models. This study demonstrates the advantages of ML-based tR prediction at scale and provides insights into interpretable and efficient utilization of ML in chromatographic applications.

---

## 124. Angular Graph Fractional Fourier Transform: Theory and Application

**论文链接:** [http://arxiv.org/abs/2511.16111v1](http://arxiv.org/abs/2511.16111v1)

**作者:** Feiyue Zhao, Yangfan He, Zhichao Zhang

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究提出了一种角度GFRFT(AGFRFT)框架，结合了分数阶和角度谱分析，解决了现有GFRFT缺乏角度调节和AGFT在零角度时无法正确退化为GFT的问题。AGFRFT提供了统一的谱分析工具，在多个去噪任务中表现出优越性能。

### 背景

图谱表示在图信号处理中是基础性的，为分析和处理图结构数据提供了严格的框架。图分数傅里叶变换(GFRFT)通过分数阶参数扩展了经典图傅里叶变换(GFT)，实现了灵活的谱分析，同时保持了数学一致性。角度图傅里叶变换(AGFT)通过GFT特征向量旋转引入角度控制；然而，现有构造在零角度时无法退化为GFT，这是一个关键缺陷，损害了理论一致性和可解释性。

### 目的

解决GFRFT缺乏角度调节和AGFT有缺陷的退化问题，提出一个统一的框架，将分数阶和角度谱分析结合起来，并确保在零角度时精确退化为GFT。

### 方法

提出角度GFRFT(AGFRFT)，设计了一个退化友好的旋转矩阵族，确保在零角度时精确退化为GFT。定义了两种AGFRFT变体(I-AGFRFT和II-AGFRFT)，并进行了严格的理论分析，确认了它们的酉性、可逆性和平滑参数依赖性。该方法支持角度和分数阶的联合参数化学习，实现适应性的图信号谱处理。

### 主要发现

AGFRFT在谱集中度、重建质量和可控谱操作方面优于GFRFT和AGFT。实验在真实数据去噪、图像去噪和点云去噪任务中验证了其优越性能。AGFRFT为图信号处理中的集成角度分数谱分析提供了强大而灵活的工具。

### 结论

AGFRFT成功解决了现有方法的局限性，提供了一个统一且理论上严谨的框架，用于图信号处理中的角度分数谱分析，具有实际应用价值和理论意义。

### 翻译

图谱表示在图信号处理中是基础性的，为分析和处理图结构数据提供了严格的框架。图分数傅里叶变换(GFRFT)通过分数阶参数扩展了经典图傅里叶变换(GFT)，实现了灵活的谱分析，同时保持了数学一致性。角度图傅里叶变换(AGFT)通过GFT特征向量旋转引入角度控制；然而，现有构造在零角度时无法退化为GFT，这是一个关键缺陷，损害了理论一致性和可解释性。为了解决这些互补的局限性——GFRFT缺乏角度调节和AGFT有缺陷的退化——本研究提出了角度GFRFT(AGFRFT)，一个将分数阶和角度谱分析与理论严谨性相结合的统一框架。一个退化友好的旋转矩阵族确保在零角度时精确退化为GFT，并据此定义了两种AGFRFT变体(I-AGFRFT和II-AGFRFT)。严格的理论分析证实了它们的酉性、可逆性和平滑参数依赖性。两者都支持角度和分数阶的可联合参数化学习，使各种图信号能够进行自适应谱处理。在真实数据去噪、图像去噪和点云去噪方面的广泛实验表明，AGFRFT在谱集中度、重建质量和可控谱操作方面优于GFRFT和AGFT，为图信号处理中的集成角度分数谱分析建立了强大而灵活的工具。


### 论文摘要

Graph spectral representations are fundamental in graph signal processing, offering a rigorous framework for analyzing and processing graph-structured data. The graph fractional Fourier transform (GFRFT) extends the classical graph Fourier transform (GFT) with a fractional-order parameter, enabling flexible spectral analysis while preserving mathematical consistency. The angular graph Fourier transform (AGFT) introduces angular control via GFT eigenvector rotation; however, existing constructions fail to degenerate to the GFT at zero angle, which is a critical flaw that undermines theoretical consistency and interpretability. To resolve these complementary limitations - GFRFT's lack of angular regulation and AGFT's defective degeneracy - this study proposes an angular GFRFT (AGFRFT), a unified framework that integrates fractional-order and angular spectral analyses with theoretical rigor. A degeneracy-friendly rotation matrix family ensures exact GFT degeneration at zero angle, with two AGFRFT variants (I-AGFRFT and II-AGFRFT) defined accordingly. Rigorous theoretical analyses confirm their unitarity, invertibility, and smooth parameter dependence. Both support learnable joint parameterization of the angle and fractional order, enabling adaptive spectral processing for diverse graph signals. Extensive experiments on real-world data denoising, image denoising, and point cloud denoising demonstrate that AGFRFT outperforms GFRFT and AGFT in terms of spectral concentration, reconstruction quality, and controllable spectral manipulation, establishing a robust and flexible tool for integrated angular fractional spectral analysis in graph signal processing.

---

## 125. Rad-GS: Radar-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments

**论文链接:** [http://arxiv.org/abs/2511.16091v1](http://arxiv.org/abs/2511.16091v1)

**作者:** Renxiang Xiao, Wei Liu, Yuanfan Zhang, Yushuai Chen, Jinming Chen, Zilu Wang, Liang Hu

**发布时间:** 2025-11-20

**DOI:** 10.1109/LRA.2025.3630875

### GPT解析

### 总结

本文提出了一种名为Rad-GS的4D雷达-相机SLAM系统，专为公里级户外环境设计，利用3D高斯作为可微分空间表示，结合雷达多普勒信息和几何数据提高定位精度和渲染质量，同时减少内存消耗。

### 背景

大规模户外环境中的SLAM系统面临动态物体处理和噪声抑制的挑战，传统方法通常依赖相机或激光雷达，而4D毫米波雷达的优势尚未在大规模场景中充分利用。

### 目的

开发一种利用4D雷达数据的SLAM系统，实现公里级户外环境的高精度定位和场景重建，同时优化内存使用和渲染质量。

### 方法

结合原始雷达点云的多普勒信息和几何增强点云指导动态物体掩码；利用非同步图像帧全局优化3D高斯表示；采用全局八叉树结构和针对性高斯基元管理策略抑制噪声并减少内存消耗。

### 主要发现

Rad-GS减轻了渲染伪影并提高了定位精度；增强了纹理一致性和新视图合成保真度；性能与传统基于相机或激光雷达的3D高斯方法相当；证明了4D毫米波雷达在大规模户外场景中的可行性。

### 结论

Rad-GS是一种有效的4D雷达-相机SLAM系统，能够实现公里级户外环境的高精度定位和场景重建，为大规模场景重建提供了新方法。

### 翻译

我们提出Rad-GS，一种专为公里级户外环境设计的4D雷达-相机SLAM系统，利用3D高斯作为可微分空间表示。Rad-GS结合原始雷达点云的多普勒信息和几何增强点云的优势，引导同步图像中的动态物体掩码，从而减轻渲染伪影并提高定位精度。此外，利用非同步图像帧全局优化3D高斯表示，增强纹理一致性和新视图合成保真度。全局八叉树结构结合针对性高斯基元管理策略进一步抑制噪声，显著减少大规模环境中的内存消耗。大量实验和消融研究表明，Rad-GS性能与传统基于相机或激光雷达输入的3D高斯方法相当，突显了使用4D毫米波雷达进行稳健户外地图绘制的可行性。公里级真实世界重建验证了Rad-GS在大规模场景重建中的潜力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决在大型户外环境中如何利用4D毫米波雷达和相机进行高保真地图构建和同步定位与地图创建（SLAM）的问题。这个问题很重要，因为4D雷达能提供全天候感知性能，而现有方法受限于雷达信号的噪声、稀疏性以及动态物体处理不当等问题，无法实现高质量的大规模场景重建，这对自动驾驶、机器人导航和增强现实等应用至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有4D雷达-相机融合SLAM方法的局限性，然后选择3D高斯溅射作为空间表示方法。他们借鉴了3DGS的基本框架、CMDF的雷达点云增强方法、LiV-GS的多传感器融合思想以及Doppler-based ego-motion模型。在此基础上，他们创新性地设计了多普勒引导的动态物体移除模块、全局八叉树结构管理策略和粗糙度感知的损失函数，形成了完整的Rad-GS系统。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将4D雷达的多普勒信息和视觉数据融合到3D高斯表示框架中，实现大型户外环境的高保真动态场景重建。整体流程包括：1)数据增强和多普勒引导的动态物体移除；2)前端跟踪，通过增强雷达点云与高斯原语匹配优化姿态；3)后端优化，利用未同步图像提高渲染质量；4)自适应高斯八叉树管理，实现内存高效的增量式映射；5)全局优化，持续更新整个地图表示。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个统一的4D雷达-相机SLAM框架，使用3D高斯表示实现动态场景重建；2)单帧动态物体移除方法，利用多普勒信息和几何点云生成掩码；3)全局八叉树维护策略，显著减少内存消耗；4)粗糙度感知的损失函数，适应不同表面特性；5)多普勒信息与视觉数据的深度融合。相比之前的工作，Rad-GS更专注于雷达-视觉融合，能更好地处理动态物体，并在大规模环境中实现更高效的内存管理。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Rad-GS首次将4D雷达的多普勒信息与视觉数据融合到3D高斯表示框架中，实现了大型户外环境的高保真动态场景重建，同时通过创新的八叉树管理策略显著降低了内存消耗。'}


### 论文摘要

We present Rad-GS, a 4D radar-camera SLAM system designed for kilometer-scale outdoor environments, utilizing 3D Gaussian as a differentiable spatial representation. Rad-GS combines the advantages of raw radar point cloud with Doppler information and geometrically enhanced point cloud to guide dynamic object masking in synchronized images, thereby alleviating rendering artifacts and improving localization accuracy. Additionally, unsynchronized image frames are leveraged to globally refine the 3D Gaussian representation, enhancing texture consistency and novel view synthesis fidelity. Furthermore, the global octree structure coupled with a targeted Gaussian primitive management strategy further suppresses noise and significantly reduces memory consumption in large-scale environments. Extensive experiments and ablation studies demonstrate that Rad-GS achieves performance comparable to traditional 3D Gaussian methods based on camera or LiDAR inputs, highlighting the feasibility of robust outdoor mapping using 4D mmWave radar. Real-world reconstruction at kilometer scale validates the potential of Rad-GS for large-scale scene reconstruction.

---

## 126. Methane on the temperate exo-Saturn TOI-199b

**论文链接:** [http://arxiv.org/abs/2511.15835v1](http://arxiv.org/abs/2511.15835v1)

**作者:** Aaron Bello-Arufe, Renyu Hu, Mantas Zilinskas, Jeehyun Yang, Armen Tokadjian, Luis Welbanks, Guangwei Fu, Michael Greklek-McKeon, Mario Damiano, Jonathan Gomez Barrientos, Heather A. Knutson, David K. Sing, Xi Zhang

**发布时间:** 2025-11-19

**备注:** 23 pages, 16 figures, submitted to AAS Journals

### GPT解析

### 总结

研究团队使用JWST观测了土星质量系外行星TOI-199 b的透射光谱，检测到甲烷的存在，并分析了其大气成分、烟雾模型和轨道特性。

### 背景

温和温度（平衡温度低于400开尔文）的气体巨行星是系外行星大气光谱学中一个未被探索的前沿领域。TOI-199 b是一颗土星质量的系外行星，围绕G型恒星运行，周期约100天，平衡温度为350开尔文，是大气研究的理想低温气体巨行星之一。

### 目的

展示TOI-199 b的透射光谱，研究其大气成分和特性。

### 方法

使用JWST的NIRSpec G395M模式对TOI-199 b的单次凌日进行观测，应用贝叶斯检索分析数据，测试多种烟雾模型，并分析行星凌日时间变化（TTV）。

### 主要发现

检测到CH4（在多云大气中，贝叶斯因子约700），对应碳氢金属licity为太阳值的13倍，上下误差范围分别为78倍和12倍；未检测到CO和CO2，不支持金属licity为太阳值50倍以上的情况；对烟雾模型的偏好较弱（相对于清晰情况，贝叶斯因子约2）；在3微米附近凌日深度增加，可能是NH3或HCN导致；TOI-199系统由于外部不凌日的巨行星而表现出强烈的凌日时间变化；对于行星c，TTV分析将其质量不确定性降低了50%，并倾向于比先前研究更长的轨道周期（仍在保守宜居区内）和更高的离心率。

### 结论

TOI-199 b是研究温和气体巨行星中云和烟雾的第一个数据点。甲烷的检测支持新兴趋势：温和低分子量大气在透射中显示光谱特征。

### 翻译

TOI-199 b是一颗土星质量的系外行星，围绕G型恒星运行，周期约100天，平衡温度为350开尔文。研究团队使用JWST观测了其透射光谱，检测到甲烷的存在，对应碳氢金属licity为太阳值的13倍，上下误差范围分别为78倍和12倍。未检测到CO和CO2，不支持金属licity为太阳值50倍以上的情况。光谱显示3微米附近凌日深度增加，可能是NH3或HCN导致。TOI-199系统表现出强烈的凌日时间变化，对行星c的分析表明其质量不确定性降低50%，轨道周期更长且离心率更高。TOI-199 b是研究温和气体巨行星云和烟雾的第一个数据点，甲烷检测支持温和低分子量大气在透射中显示光谱特征的趋势。


### 论文摘要

Temperate ($T_{\rm eq}<400$ K) gas giants represent an unexplored frontier in exoplanet atmospheric spectroscopy. Orbiting a G-type star every ~100 days, the Saturn-mass exoplanet TOI-199 b ($T_{\rm eq}=350$ K) is one of the most favorable low-temperature gas giants for atmospheric study. Here, we present its transmission spectrum from a single transit observed with JWST's NIRSpec G395M mode. Despite lower-than-nominal precision due to a pointing misalignment, Bayesian retrievals reveal the presence of CH$_4$ (Bayes factor of $\sim$700 in a cloudy atmosphere), corresponding to a metallicity of $\rm{C/H}=13^{+78}_{-12}\times$ solar, although the absence of detectable CO and CO$_2$ disfavors metallicities $\gtrsim50\times$ solar. We also tested several haze prescriptions (Titan-like tholin, soot, and water-rich tholin), but the preference for these models is weak (Bayes factors of $\sim 2$ relative to the clear case). The spectrum also shows an increase in transit depth near 3 $μ$m, which our self-consistent models attribute to either NH$_3$ or, less likely, HCN. Follow-up observations will distinguish between these species, helping determine the planet's vertical mixing regime. The TOI-199 system exhibits strong transit timing variations (TTVs) due to an outer non-transiting giant planet. For planet c, our TTV analysis reduces its mass uncertainty by 50% and prefers a slightly longer orbital period (still within the conservative habitable zone) and higher eccentricity relative to previous studies. TOI-199 b serves as the first data point for studying clouds and hazes in temperate gas giants. The detection of methane supports the emerging trend that temperate low-molecular-weight atmospheres display spectral features in transmission.

---

## 127. Atlas Gaussian processes on restricted domains and point clouds

**论文链接:** [http://arxiv.org/abs/2511.15822v1](http://arxiv.org/abs/2511.15822v1)

**作者:** Mu Niu, Yue Zhang, Ke Ye, Pokman Cheung, Yizhu Wang, Xiaochen Yang

**发布时间:** 2025-11-19

### GPT解析

### 总结

该研究提出了一种处理具有未知几何结构和非平凡拓扑结构的高维点云数据的新方法，通过建立图集布朗运动框架和黎曼修正核，改进了传统高斯过程在复杂数据集上的性能。

### 背景

现实世界中的数据通常存在于具有未知边界的受限域中，或者是位于低维、复杂、未知流形上的高维点云。传统高斯过程难以捕捉这类数据中的潜在几何结构。

### 目的

解决传统高斯过程在处理具有未知几何结构的点云数据时的局限性，特别是在数据稀疏或不规则采样的情况下。

### 方法

建立了图集布朗运动(Atlas BM)框架用于估计点云上的热核，并构建了黎曼修正核，通过结合全局热核与局部RBF核，形成了黎曼修正的图集高斯过程(RC-AGPs)。

### 主要发现

RC-AGPs在热核估计和回归准确性方面均优于现有方法，能够有效弥合复杂高维观测与基于流形的推理之间的差距，改善统计推断效果。

### 结论

所提出的方法成功解决了传统高斯过程在处理具有未知几何结构和非平凡拓扑结构的高维点云数据时的挑战，为复杂数据的分析提供了更有效的工具。

### 翻译

在实际应用中，数据通常存在于具有未知边界的受限域中，或者是位于低维、复杂、未知流形上的高维点云。传统高斯过程(GPs)难以在这样的设置中捕捉潜在的几何结构。一些现有方法假设点云嵌入在平坦空间中，可以用单一潜在图表(潜在空间)表示，而另一些方法在点云稀疏或不规则采样时性能较弱。本工作的目标是解决这些挑战。主要贡献有两方面：(1)我们建立了图集布朗运动(BM)框架，用于估计具有未知几何结构和非平凡拓扑结构的点云上的热核；(2)我们没有直接使用热核估计，而是通过结合全局热核与局部RBF核构建了一个黎曼修正核，从而形成了黎曼修正的图集高斯过程(RC-AGPs)。将得到的RC-AGPs应用于合成和真实世界数据集的回归任务。这些示例表明，我们的方法在热核估计和回归准确性方面都优于现有方法。通过有效弥合复杂高维观测与基于流形的推理之间的差距，它改善了统计推断。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决传统高斯过程在处理具有未知边界或作为高维点云存在于低维、非平凡、未知流形上的受限域数据时，难以捕捉潜在几何结构的问题。这个问题在现实中很重要，因为许多实际应用的数据（如湖泊内的污染数据、图像中物体的旋转运动）并不遵循简单的欧几里得几何结构，传统方法无法有效处理这些复杂数据，限制了我们在复杂空间中进行统计推断的能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到传统高斯过程在处理复杂几何结构数据时的局限性，发现现有方法要么假设平坦空间嵌入点云（可由单个潜在图表表示），要么在点云稀疏或不规则采样时表现不佳。作者借鉴了多种现有工作：Mapper工具用于将复杂点云划分为局部简单子集；图拉普拉斯高斯过程用于近似流形但存在局限性；单图表布朗运动方法局限于平坦拓扑；高斯过程潜在变量模型和自编码器用于学习映射函数。基于这些工作，作者设计了新的解决方案，通过构建概率图谱和使用布朗运动在图谱上模拟路径来估计热核。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用多个局部坐标系统（图表）覆盖具有非平凡拓扑结构的未知流形，通过在图谱上模拟布朗运动路径来估计热核捕捉全局几何结构，并将热核与局部RBF核结合创建黎曼修正核，平衡全局几何和局部平滑性。整体流程包括：1）使用Mapper将点云划分为子集，用GPLVM或自编码器学习每个子集的局部图表和映射函数；2）在每个图表上定义随机微分方程模拟布朗运动，实现图表间的坐标转换；3）将估计的热核矩阵扩展并与RBF核结合，构建黎曼修正核；4）使用黎曼修正核构建高斯过程进行回归任务。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）Atlas Brownian Motion框架，首次在具有未知几何和非平凡拓扑结构的点云上估计热核；2）Riemannian修正核，结合全局热核与局部RBF核；3）Riemannian修正图谱高斯过程（RC-AGPs），在保持计算效率的同时提高回归准确性。相比之前工作的不同：相比单图表方法，能处理非平凡拓扑结构；相比图拉普拉斯方法，在点云稀疏时更鲁棒且对超参数不那么敏感；相比外在框架方法，不依赖流形嵌入；计算效率更高，将模拟路径数量从n×Nbm减少到nv×Nbm。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种创新的Riemannian修正图谱高斯过程方法，通过在多个局部图表上模拟布朗运动来估计热核，结合全局几何与局部平滑性，有效解决了具有未知边界和复杂几何结构的高维点云上的回归问题，显著提高了预测精度和计算效率。'}


### 论文摘要

In real-world applications, data often reside in restricted domains with unknown boundaries, or as high-dimensional point clouds lying on a lower-dimensional, nontrivial, unknown manifold. Traditional Gaussian Processes (GPs) struggle to capture the underlying geometry in such settings. Some existing methods assume a flat space embedded in a point cloud, which can be represented by a single latent chart (latent space), while others exhibit weak performance when the point cloud is sparse or irregularly sampled. The goal of this work is to address these challenges. The main contributions are twofold: (1) We establish the Atlas Brownian Motion (BM) framework for estimating the heat kernel on point clouds with unknown geometries and nontrivial topological structures; (2) Instead of directly using the heat kernel estimates, we construct a Riemannian corrected kernel by combining the global heat kernel with local RBF kernel and leading to the formulation of Riemannian-corrected Atlas Gaussian Processes (RC-AGPs). The resulting RC-AGPs are applied to regression tasks across synthetic and real-world datasets. These examples demonstrate that our method outperforms existing approaches in both heat kernel estimation and regression accuracy. It improves statistical inference by effectively bridging the gap between complex, high-dimensional observations and manifold-based inferences.

---

## 128. CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking

**论文链接:** [http://arxiv.org/abs/2511.15580v2](http://arxiv.org/abs/2511.15580v2)

**作者:** Sifan Zhou, Yichao Cao, Jiahao Nie, Yuqian Fu, Ziyu Zhao, Xiaobo Lu, Shuo Wang

**发布时间:** 2025-11-19

**备注:** Accepted by AAAI 2026 (Oral)

### GPT解析

### 总结

这篇论文提出了一种名为CompTrack的新型端到端框架，用于解决点云中的双重冗余问题，实现了高效且准确的3D单目标跟踪。

### 背景

3D单目标跟踪在计算机视觉和自动驾驶中是一项关键任务，尽管取得了很大成功，但点云的固有稀疏性带来了双重冗余挑战，限制了现有跟踪器的性能。

### 目的

解决点云中的两种冗余问题：空间冗余（背景噪声）和信息冗余（前景内部），以提高跟踪器的准确性和效率。

### 方法

CompTrack框架包含两个主要模块：1) 空间前景预测器(SFP)：基于信息熵过滤掉不相关的背景噪声，解决空间冗余问题；2) 信息瓶颈引导的动态令牌压缩(IB-DTC)：基于低秩近似理论，利用在线SVD分析自适应地将冗余前景压缩为紧凑且信息丰富的代理令牌集合，解决信息冗余问题。

### 主要发现

在KITTI、nuScenes和Waymo数据集上的大量实验表明，CompTrack实现了顶级的跟踪性能，同时具有卓越的效率，在单张RTX 3090 GPU上能够以实时90 FPS的速度运行。

### 结论

CompTrack通过系统性地消除点云中的两种冗余形式，实现了高效且准确的3D单目标跟踪，为自动驾驶等领域提供了有效的解决方案。

### 翻译

3D单目标跟踪在激光雷达点云中是计算机视觉和自动驾驶的关键任务。尽管取得了巨大成功，但点云的固有稀疏性引入了双重冗余挑战，限制了现有跟踪器：(1)来自背景噪声的大量空间冗余损害了准确性，(2)前景内部的信息冗余阻碍了效率。为解决这些问题，我们提出了CompTrack，一种新型端到端框架，系统性地消除了点云中的两种冗余。首先，CompTrack集成了空间前景预测器(SFP)模块，基于信息熵过滤掉不相关的背景噪声，解决空间冗余问题。随后，其核心是信息瓶颈引导的动态令牌压缩(IB-DTC)模块，消除了前景内部的信息冗余。基于低秩近似的理论，该模块利用在线SVD分析自适应地将冗余前景压缩为紧凑且信息丰富的代理令牌集合。在KITTI、nuScenes和Waymo数据集上的大量实验表明，CompTrack实现了顶级的跟踪性能和卓越的效率，在单张RTX 3090 GPU上以实时90 FPS的速度运行。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决LiDAR点云数据中的3D单目标跟踪问题，具体处理点云固有的稀疏性带来的双重冗余挑战：(1)空间冗余：来自背景噪声的大量不相关点影响准确性；(2)信息冗余：前景中重复几何结构带来的信息冗余影响效率。这个问题在自动驾驶和机器人领域非常重要，因为LiDAR点云是这些系统的主要感知数据源，准确且高效地跟踪3D目标对于自动驾驶的安全决策和实时响应至关重要。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性，指出它们主要关注空间冗余而忽视了信息冗余。作者设计了一个名为CompTrack的框架，包含两个主要组件：空间前景预测器(SFP)基于信息熵理论过滤背景噪声；信息瓶颈引导的动态令牌压缩(IB-DTC)基于低秩近似理论消除前景信息冗余。作者借鉴了信息瓶颈理论和Eckart-Young定理，同时参考了现有的点云处理方法如Pillar编码器和鸟瞰图(BEV)表示，将这些理论创新性地应用到点云跟踪任务中。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过双重冗余消除策略提高点云跟踪的效率和准确性：首先过滤空间冗余，然后压缩信息冗余。整体流程为：(1)将原始点云转换为BEV特征图；(2)使用SFP生成空间注意力图过滤背景噪声；(3)使用IB-DTC模块对前景进行动态令牌压缩，包括在线秩估计(通过SVD确定有效秩K)、动态查询学习(基于奇异值选择查询)和引导交叉注意力(生成压缩后的代理令牌)；(4)使用压缩后的令牌进行目标预测。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：(1)首次同时处理点云中的空间冗余和信息冗余；(2)提出SFP模块，基于信息熵理论过滤背景噪声；(3)提出IB-DTC模块，结合信息瓶颈理论和低秩近似，实现动态令牌压缩；(4)在多个数据集上实现最先进性能的同时保持高效率(90 FPS)。相比之前的工作，CompTrack不仅关注空间冗余，还解决了之前方法忽视的信息冗余问题，理论基础更加扎实，结合了信息论和线性代数的原理，实现了精度和效率的更好平衡。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CompTrack通过信息瓶颈引导的低秩动态令牌压缩方法，首次同时解决了点云跟踪中的空间和信息冗余问题，实现了高精度和实时性的平衡。'}


### 论文摘要

3D single object tracking (SOT) in LiDAR point clouds is a critical task in computer vision and autonomous driving. Despite great success having been achieved, the inherent sparsity of point clouds introduces a dual-redundancy challenge that limits existing trackers: (1) vast spatial redundancy from background noise impairs accuracy, and (2) informational redundancy within the foreground hinders efficiency. To tackle these issues, we propose CompTrack, a novel end-to-end framework that systematically eliminates both forms of redundancy in point clouds. First, CompTrack incorporates a Spatial Foreground Predictor (SFP) module to filter out irrelevant background noise based on information entropy, addressing spatial redundancy. Subsequently, its core is an Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module that eliminates the informational redundancy within the foreground. Theoretically grounded in low-rank approximation, this module leverages an online SVD analysis to adaptively compress the redundant foreground into a compact and highly informative set of proxy tokens. Extensive experiments on KITTI, nuScenes and Waymo datasets demonstrate that CompTrack achieves top-performing tracking performance with superior efficiency, running at a real-time 90 FPS on a single RTX 3090 GPU.

---

## 129. Optimizing Quantum Key Distribution Network Performance using Graph Neural Networks

**论文链接:** [http://arxiv.org/abs/2511.16468v1](http://arxiv.org/abs/2511.16468v1)

**作者:** Akshit Pramod Anchan, Ameiy Acharya, Leki Chom Thungon

**发布时间:** 2025-11-20

**备注:** 11 pages, 4 figures, and 2 tables

### GPT解析

### 总结

本文提出了一种使用图神经网络(GNN)框架优化量子密钥分发(QKD)网络的方法，解决了QKD网络在动态条件适应、多参数优化和资源利用方面的困难。

### 背景

量子计算机的发展威胁了经典密码系统的安全性，而QKD网络虽然设计用于保护秘密通信，但面临多个操作困难。

### 目的

克服QKD网络面临的操作困难，提高网络性能和安全性。

### 方法

提出基于GNN的框架，将QKD网络建模为动态图，从网络结构中提取可利用的特征，图中包含拓扑信息和与量子通信相关的特定特征。

### 主要发现

GNN优化的QKD网络显著提高了总密钥速率(从27.1 Kbits/s到470 Kbits/s)，降低了平均量子误码率(从6.6%到6.0%)，保持路径完整性同时略微减少平均传输距离(从7.13 km到6.42 km)，在不同规模网络中改进了链路预测精度和密钥生成速率。

### 结论

这项工作为QKD网络引入了新的操作模式，通过自适应和可扩展的量子通信系统转变网络优化范式，提高了安全性和性能。

### 翻译

本文提出了一种使用图神经网络(GNN)框架优化量子密钥分发(QKD)网络的方法。如今，量子计算机的发展威胁了经典密码系统的安全性。此外，由于QKD网络旨在保护秘密通信，它们面临多个操作困难：适应动态条件、多参数优化和有效资源利用。为了克服这些障碍，我们提出了一种基于GNN的框架，可将QKD网络建模为动态图，并从这些网络的结构中提取可利用的特征。图中不仅包含拓扑信息，还包含与量子通信相关的特定特征(如节点之间的边数等)。实验结果表明，GNN优化的QKD网络实现了总密钥速率的显著提高(从27.1 Kbits/s到470 Kbits/s)，降低了平均量子误码率(从6.6%到6.0%)，并保持路径完整性，同时平均传输距离略有减少(从7.13 km到6.42 km)。此外，我们分析了不同规模(10到250个节点)的网络性能，显示出改进的链路预测精度和中型网络中增强的密钥生成速率。这项工作为QKD网络引入了一种新的操作模式，通过自适应和可扩展的量子通信系统转变网络优化范式，提高了安全性和性能。


### 论文摘要

This paper proposes an optimization of Quantum Key Distribution (QKD) Networks using Graph Neural Networks (GNN) framework. Today, the development of quantum computers threatens the security systems of classical cryptography. Moreover, as QKD networks are designed for protecting secret communication, they suffer from multiple operational difficulties: adaptive to dynamic conditions, optimization for multiple parameters and effective resource utilization. In order to overcome these obstacles, we propose a GNN-based framework which can model QKD networks as dynamic graphs and extracts exploitable characteristics from these networks' structure. The graph contains not only topological information but also specific characteristics associated with quantum communication (the number of edges between nodes, etc). Experimental results demonstrate that the GNN-optimized QKD network achieves a substantial increase in total key rate (from 27.1 Kbits/s to 470 Kbits/s), a reduced average QBER (from 6.6% to 6.0%), and maintains path integrity with a slight reduction in average transmission distance (from 7.13 km to 6.42 km). Furthermore, we analyze network performance across varying scales (10 to 250 nodes), showing improved link prediction accuracy and enhanced key generation rate in medium-sized networks. This work introduces a novel operation mode for QKD networks, shifting the paradigm of network optimization through adaptive and scalable quantum communication systems that enhance security and performance.

---

## 130. Dynamic Multiple-Parameter Joint Time-Vertex Fractional Fourier Transform and its Intelligent Filtering Methods

**论文链接:** [http://arxiv.org/abs/2511.16277v1](http://arxiv.org/abs/2511.16277v1)

**作者:** Manjun Cui, Ziqi Yan, Yangfan He, Zhichao Zhang

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种动态多参数联合时-顶点分数傅里叶变换（DMPJFRFT）框架，通过引入时变分数参数实现动态图结构的自适应谱建模，有效捕捉时间拓扑变化，并在去噪和去模糊任务中表现优异。

### 背景

动态图信号处理是分析不规则图域上时变数据的原则性框架，但现有联合时-顶点变换只能为空间域和时间域分配单一分数阶，限制了建模图信号复杂连续变化动态的能力。

### 目的

解决现有变换的限制，提出能够动态灵活表示时空信号演化的新变换框架。

### 方法

提出DMPJFRFT框架，为每个时间步分配不同分数阶，在联合时-顶点谱域中动态表示时空信号演化；分析DMPJFRFT理论性质；开发基于梯度下降和神经网络的两种滤波方法用于动态信号恢复。

### 主要发现

实验表明DMPJFRFT能有效捕捉时间拓扑变化，在去噪和去模糊任务中优于现有基于图的变换和神经网络。

### 结论

DMPJFRFT通过引入时变分数参数克服了现有变换的局限，为动态图信号处理提供了更强大的工具。

### 翻译

动态图信号处理为定义在不规则图域上的时变数据分析提供了原则性框架。然而，现有的联合时-顶点变换（如联合时-顶点分数傅里叶变换）只为空间域分配一个分数阶，为时间域分配另一个分数阶，从而限制了它们建模图信号复杂且连续变化动态的能力。为解决这一限制，我们提出了一种新颖的动态多参数联合时-顶点分数傅里叶变换（DMPJFRFT）框架，该框架引入时变分数参数以实现动态图结构的自适应谱建模。通过为每个时间步分配不同的分数阶，所提出的变换能够在联合时-顶点谱域中动态且灵活地表示时空信号演化。DMPJFRFT的理论性质得到了系统分析，并开发了两种滤波方法：基于梯度下降的方法和基于神经网络的方法，用于动态信号恢复。在动态图和视频数据集上的实验结果表明，所提出的框架能够有效捕捉时间拓扑变化，并且在去噪和去模糊任务中与一些最先进的基于图的变换和神经网络相比实现了卓越的性能。


### 论文摘要

Dynamic graph signal processing provides a principled framework for analyzing time-varying data defined on irregular graph domains. However, existing joint time-vertex transforms such as the joint time-vertex fractional Fourier transform assign only one fractional order to the spatial domain and another one to the temporal domain, thereby restricting their capacity to model the complex and continuously varying dynamics of graph signals. To address this limitation, we propose a novel dynamic multiple-parameter joint time-vertex fractional Fourier transform (DMPJFRFT) framework, which introduces time-varying fractional parameters to achieve adaptive spectral modeling of dynamic graph structures. By assigning distinct fractional orders to each time step, the proposed transform enables dynamic and flexible representation of spatio-temporal signal evolution in the joint time-vertex spectral domain. Theoretical properties of the DMPJFRFT are systematically analyzed, and two filtering approaches: a gradient descent-based method and a neural network-based method, are developed for dynamic signal restoration. Experimental results on dynamic graph and video datasets demonstrate that the proposed framework effectively captures temporal topology variations and achieves superior performance in denoising and deblurring tasks compared with some state-of-the-art graph-based transforms and neural networks.

---

## 131. 论文ID: 2511.16101v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2511.16101v1.json'

---

## 132. L-JacobiNet and S-JacobiNet: An Analysis of Adaptive Generalization, Stabilization, and Spectral Domain Trade-offs in GNNs

**论文链接:** [http://arxiv.org/abs/2511.16081v1](http://arxiv.org/abs/2511.16081v1)

**作者:** Huseyin Goksu

**发布时间:** 2025-11-20

### GPT解析

### 总结

该研究探讨了Spectral GNNs在异质性和过度平滑方面的局限性，提出并比较了不同域中的自适应正交多项式滤波器模型，发现稳定性而非静态性质是ChebyNet的主要问题，并提出S-JacobiNet作为更有效的解决方案。

### 背景

Spectral GNNs（如ChebyNet）由于其静态低通滤波器设计，受到异质性和过度平滑的限制。

### 目的

研究'自适应正交多项式滤波器'(AOPF)类作为解决方案，并探索不同域中AOPFs的性能差异。

### 方法

引入两个在[-1, 1]域运行的模型：1) L-JacobiNet（ChebyNet的自适应推广，具有可学习的alpha、beta形状参数）和2) S-JacobiNet（LayerNorm稳定的静态ChebyNet），并将其与[0, ∞)域中的AOPFs（如LaguerreNet）进行比较分析。

### 主要发现

[0, ∞)域在建模异质性方面表现更优；[-1, 1]域在高K值时提供更好的数值稳定性；ChebyNet的主要缺陷是稳定性问题而非静态特性；静态的S-JacobiNet在5个基准数据集中的4个上优于自适应的L-JacobiNet。

### 结论

S-JacobiNet是一个强大但被忽视的基线模型；在[-1, 1]域中的自适应可能导致过拟合。

### 翻译

Spectral GNNs（如ChebyNet）由于其静态低通滤波器设计，受到异质性和过度平滑的限制。本研究将'自适应正交多项式滤波器'(AOPF)类作为解决方案。我们引入了两个在[-1, 1]域运行的模型：1) L-JacobiNet，即ChebyNet的自适应推广，具有可学习的alpha、beta形状参数；2) S-JacobiNet，一个新颖的基线，代表LayerNorm稳定的静态ChebyNet。我们的分析将这些模型与[0, ∞)域中的AOPFs（如LaguerreNet）进行比较，揭示了关键且先前未知的权衡。我们发现[0, ∞)域在建模异质性方面更优越，而[-1, 1]域（Jacobi）在高K值（K>20）时提供更好的数值稳定性。最重要的是，我们发现ChebyNet的主要缺陷是稳定性问题，而非其静态性质。我们的静态S-JacobiNet（ChebyNet+LayerNorm）在5个基准数据集中的4个上优于自适应的L-JacobiNet，这表明S-JacobiNet是一个强大但被忽视的基线，并提示在[-1, 1]域中的自适应可能导致过拟合。


### 论文摘要

Spectral GNNs, like ChebyNet, are limited by heterophily and over-smoothing due to their static, low-pass filter design. This work investigates the "Adaptive Orthogonal Polynomial Filter" (AOPF) class as a solution. We introduce two models operating in the [-1, 1] domain: 1) `L-JacobiNet`, the adaptive generalization of `ChebyNet` with learnable alpha, beta shape parameters, and 2) `S-JacobiNet`, a novel baseline representing a LayerNorm-stabilized static `ChebyNet`. Our analysis, comparing these models against AOPFs in the [0, infty) domain (e.g., `LaguerreNet`), reveals critical, previously unknown trade-offs. We find that the [0, infty) domain is superior for modeling heterophily, while the [-1, 1] domain (Jacobi) provides superior numerical stability at high K (K>20). Most significantly, we discover that `ChebyNet`'s main flaw is stabilization, not its static nature. Our static `S-JacobiNet` (ChebyNet+LayerNorm) outperforms the adaptive `L-JacobiNet` on 4 out of 5 benchmark datasets, identifying `S-JacobiNet` as a powerful, overlooked baseline and suggesting that adaptation in the [-1, 1] domain can lead to overfitting.

---

## 133. Bellman Memory Units: A neuromorphic framework for synaptic reinforcement learning with an evolving network topology

**论文链接:** [http://arxiv.org/abs/2511.16066v1](http://arxiv.org/abs/2511.16066v1)

**作者:** Shreyan Banerjee, Aasifa Rounak, Vikram Pakrashi

**发布时间:** 2025-11-20

**备注:** 11 pages, submitted to IEEE Transactions on Automatic Control

### GPT解析

### 总结

这篇论文提出了一种基于神经形态边缘设备的控制方法，通过突触Q-learning算法和神经形态Bellman记忆单元(BMU)解决了传统方法在梯度-free在线学习和硬件可扩展性方面的限制，实现了网络拓扑的动态演化，减少了资源利用，并增强了系统对新控制场景的适应能力。

### 背景

神经形态边缘设备在控制应用中受到梯度-free在线学习约束和硬件可扩展性限制的制约。

### 目的

开发一种能够在神经形态硬件上有效运行的强化学习方法，解决控制问题，同时优化资源利用并增强适应性。

### 方法

提出突触Q-learning算法，将Bellman方程整合到突触层面，实现网络拓扑结构的迭代演化；开发神经形态Bellman记忆单元(BMU)，使用神经工程框架在Intel Loihi神经形态芯片上实现；结合拓扑演化和混合信号计算优化神经元和突触数量。

### 主要发现

所提出的架构可以减少板上资源利用，有助于制造紧凑的专用神经形态IC；芯片上学习使系统能够适应未见过的控制场景。

### 结论

通过突触级Bellman方程整合和拓扑演化，神经形态边缘设备在控制应用中的性能和可扩展性得到了显著提升，为设计基于脉冲的强化学习加速器提供了新途径。

### 翻译

神经形态边缘设备在控制应用中的使用受到梯度-free在线学习约束和硬件在控制问题上可扩展性的限制。本文介绍了用于经典Cartpole控制的突触Q-learning算法，其中Bellman方程被整合到突触级别。这种公式使得网络拓扑结构（表示为有向图）在整个训练过程中能够迭代演化。随后采用了一种类似的方法，称为神经形态Bellman记忆单元(BMU)，使用神经工程框架在Intel的Loihi神经形态芯片上实现。拓扑演化与混合信号计算相结合，可以利用神经元和突观数量的优化来设计基于脉冲的强化学习加速器。所提出的架构可以潜在减少板上资源利用，有助于制造紧凑的专用神经形态IC。此外，本工作引入的芯片上学习并在神经形态芯片上实现，可以使系统适应未见过的控制场景。


### 论文摘要

Application of neuromorphic edge devices for control is limited by the constraints on gradient-free online learning and scalability of the hardware across control problems. This paper introduces a synaptic Q-learning algorithm for the control of the classical Cartpole, where the Bellman equations are incorporated at the synaptic level. This formulation enables the iterative evolution of the network topology, represented as a directed graph, throughout the training process. This is followed by a similar approach called neuromorphic Bellman Memory Units (BMU(s)), which are implemented with the Neural Engineering Framework on Intel's Loihi neuromorphic chip. Topology evolution, in conjunction with mixed-signal computation, leverages the optimization of the number of neurons and synapses that could be used to design spike-based reinforcement learning accelerators. The proposed architecture can potentially reduce resource utilization on board, aiding the manufacturing of compact application-specific neuromorphic ICs. Moreover, the on-chip learning introduced in this work and implemented on a neuromorphic chip can enable adaptation to unseen control scenarios.

---

## 134. Gauge-Equivariant Graph Networks via Self-Interference Cancellation

**论文链接:** [http://arxiv.org/abs/2511.16062v1](http://arxiv.org/abs/2511.16062v1)

**作者:** Yoonhyuk Choi, Chong-Kwon Kim

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种称为GESC的新型图神经网络架构，通过基于投影的干扰机制替代传统加性聚合，解决了GNN在异类图上的性能问题，并在多种基准测试中超越了现有最先进模型。

### 背景

图神经网络(GNNs)在同类图上表现出色，但在异类图上往往表现不佳，这是由于自强化和相位不一致的信号导致的。

### 目的

开发一种能够有效处理异类图的图神经网络架构，克服传统GNN方法的局限性。

### 方法

提出Gauge-Equivariant Graph Network with Self-Interference Cancellation (GESC)，引入U(1)相位连接和秩1投影来减弱自平行分量，并使用符号和相位感知的门控机制调节邻居影响，在低频模式下充当局部陷波滤波器。

### 主要发现

GESC在多种图基准测试中持续优于最新的最先进模型，同时提供了一种统一的、干扰感知的消息传递视角。

### 结论

GESC方法通过创新的干扰处理机制有效解决了GNN在异类图上的性能问题，为图神经网络研究提供了新思路。

### 翻译

图神经网络(GNNs)在同类图上表现出色，但由于自强化和相位不一致的信号，在异类图上常常失败。我们提出了一种具有自干扰消除的规范等变图网络(GESC)，它用基于投影的干扰机制替代了加性聚合。与之前专注于频谱滤波中相位处理的磁规范或规范等变GNN不同，GESC引入了U(1)相位连接，然后在注意力机制前进行秩1投影以减弱自平行分量。符号和相位感知的门控进一步调节邻居影响，减弱与当前节点状态对齐的分量，并在低频模式下充当局部陷波滤波器。在各种图基准测试中，我们的方法持续优于最新的最先进模型，同时提供了一种统一的、干扰感知的消息传递视角。我们的代码可在https://anonymous.4open.science/r/GESC-1B22获取。


### 论文摘要

Graph Neural Networks (GNNs) excel on homophilous graphs but often fail under heterophily due to self-reinforcing and phase-inconsistent signals. We propose a Gauge-Equivariant Graph Network with Self-Interference Cancellation (GESC), which replaces additive aggregation with a projection-based interference mechanism. Unlike prior magnetic or gauge-equivariant GNNs that typically focus on phase handling in spectral filtering while largely relying on scalar weighting, GESC introduces a $\mathrm{U}(1)$ phase connection followed by a rank-1 projection that attenuates self-parallel components before attention. A sign- and phase-aware gate further regulates neighbor influence, attenuating components aligned with current node states and acting as a local notch on low-frequency modes. Across diverse graph benchmarks, our method consistently outperforms recent state-of-the-art models while offering a unified, interference-aware view of message passing. Our code is available at \href{here}{https://anonymous.4open.science/r/GESC-1B22}.

---

## 135. Exploiting Inter-Sample Information for Long-tailed Out-of-Distribution Detection

**论文链接:** [http://arxiv.org/abs/2511.16015v1](http://arxiv.org/abs/2511.16015v1)

**作者:** Nimeshika Udayangani, Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**发布时间:** 2025-11-20

**DOI:** 10.1109/WACV61041.2025.00828

### GPT解析

### 总结

这篇论文提出了一种基于图表示的方法，利用样本间关系来改善长尾视觉数据集的分布外(OOD)检测性能，有效降低了误报率并提高了尾部类别的分类准确率。

### 背景

在深度神经网络的安全部署中，检测分布外(OOD)数据至关重要。然而，当存在长尾分布的内部数据集时，这一问题变得尤为具有挑战性，常常导致高误报率(FPR)和低尾部类别ID分类准确率。

### 目的

研究如何利用样本间关系，通过基于图的表示方法来显著提高视觉数据集长尾识别中的OOD检测性能，同时解决现有OOD检测方法在ID尾部类别中观察到的性能不佳问题。

### 方法

使用预训练模型的特征空间初始化图结构，考虑预训练与训练数据之间的激活层分布差异，主动引入高斯化来缓解预训练模型激活层对标准正态分布的偏差，然后使用图卷积网络(GCNs)精炼初始图表示，以获得适合长尾OOD检测的特征空间。

### 主要发现

在三个基准数据集(CIFAR10-LT、CIFAR100-LT和ImageNet-LT)上的实验表明，该方法在误报率和尾部类别ID分类准确率方面显著优于最先进的方法。

### 结论

基于图的表示方法可以有效地利用样本间关系，显著改善长尾视觉数据集的OOD检测性能，同时提高尾部类别的分类准确率。

### 翻译

检测分布外数据对于深度神经网络的安全部署至关重要。当存在长尾分布的内部数据集时，这一问题变得特别具有挑战性，常常导致高误报率和低尾部类别ID分类准确率。在本文中，我们证明使用基于图的表示利用样本间关系可以显著提高视觉数据集长尾识别中的OOD检测。为此，我们使用预训练模型的特征空间来初始化我们的图结构。我们考虑了预训练与训练数据激活层分布之间的差异，并主动引入高斯化来缓解预训练模型激活层对标准正态分布的任何偏差。然后，我们使用图卷积网络精炼这个初始图表示，以获得适合长尾OOD检测的特征空间。这使我们能够解决现有OOD检测方法在ID尾部类别中观察到的性能不佳问题。在三个基准CIFAR10-LT、CIFAR100-LT和ImageNet-LT上的实验表明，我们的方法在误报率和尾部类别ID分类准确率方面以较大优势优于最先进的方法。


### 论文摘要

Detecting out-of-distribution (OOD) data is essential for safe deployment of deep neural networks (DNNs). This problem becomes particularly challenging in the presence of long-tailed in-distribution (ID) datasets, often leading to high false positive rates (FPR) and low tail-class ID classification accuracy. In this paper, we demonstrate that exploiting inter-sample relationships using a graph-based representation can significantly improve OOD detection in long-tailed recognition of vision datasets. To this end, we use the feature space of a pre-trained model to initialize our graph structure. We account for the differences between the activation layer distribution of the pre-training vs. training data, and actively introduce Gaussianization to alleviate any deviations from a standard normal distribution in the activation layers of the pre-trained model. We then refine this initial graph representation using graph convolutional networks (GCNs) to arrive at a feature space suitable for long-tailed OOD detection. This leads us to address the inferior performance observed in ID tail-classes within existing OOD detection methods. Experiments over three benchmarks CIFAR10-LT, CIFAR100-LT, and ImageNet-LT demonstrate that our method outperforms the state-of-the-art approaches by a large margin in terms of FPR and tail-class ID classification accuracy.

---

## 136. TriDiff-4D: Fast 4D Generation through Diffusion-based Triplane Re-posing

**论文链接:** [http://arxiv.org/abs/2511.16662v1](http://arxiv.org/abs/2511.16662v1)

**作者:** Eddie Pokming Sheung, Qihao Liu, Wufei Ma, Prakhar Kaushik, Jianwen Xie, Alan Yuille

**发布时间:** 2025-11-20

**备注:** 8 pages, 10 figures, Under review at a conference

### GPT解析

### 总结

TriDiff-4D是一种创新的4D生成管道，能够从文本描述生成高保真、时间连贯的4D头像，具有更好的时间一致性、运动准确性、计算效率和视觉保真度。

### 背景

随着3D动画需求的增加，从文本描述生成高保真、可控制的4D头像仍然是一个重大挑战。现有的4D生成方法存在基本局限性，包括时间与几何不一致性、感知伪影、运动不规则性、高计算成本以及对动态控制的有限能力。

### 目的

提出TriDiff-4D，一个新的4D生成管道，使用基于扩散的三重平面重定位来生成高质量、时间连贯的4D头像。

### 方法

采用自回归策略生成任意长度的4D序列，通过单个扩散过程合成每个3D帧。从大规模3D和运动数据集中明确学习3D结构和运动先验知识。首先从文本提示生成规范化的3D头像和相应的运动序列，然后使用第二个扩散模型根据运动序列使头像动起来，支持任意长的4D生成。

### 主要发现

实验结果表明TriDiff-4D显著优于现有方法。通过消除优化过程，将生成时间从小时减少到秒。大大提高了复杂运动的生成质量，具有高保真外观和准确的3D几何结构。

### 结论

TriDiff-4D通过扩散模型和三重平面重定位技术，解决了现有4D头像生成方法的局限性，实现了高效、高质量的4D头像生成。

### 翻译

随着3D动画需求的增加，从文本描述生成高保真、可控制的4D头像仍然是一个重大挑战。尽管在4D生成建模方面做出了显著努力，但现有方法表现出基本局限性，阻碍了它们的更广泛应用，包括时间与几何不一致性、感知伪影、运动不规则性、高计算成本以及对动态控制的有限能力。为了应对这些挑战，我们提出了TriDiff-4D，一种新颖的4D生成管道，采用基于扩散的三重平面重定位来产生高质量、时间连贯的4D头像。我们的模型采用自回归策略生成任意长度的4D序列，通过单个扩散过程合成每个3D帧。通过从大规模3D和运动数据集中明确学习3D结构和运动先验知识，TriDiff-4D实现了骨架驱动的4D生成，在时间一致性、运动准确性、计算效率和视觉保真度方面表现出色。具体来说，TriDiff-4D首先从文本提示生成规范化的3D头像和相应的运动序列，然后使用第二个扩散模型根据运动序列使头像动起来，支持任意长的4D生成。实验结果表明，TriDiff-4D显著优于现有方法，通过消除优化过程将生成时间从小时减少到秒，同时大大提高了复杂运动的生成质量，具有高保真外观和准确的3D几何结构。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决生成高保真、可控的4D虚拟人从文本描述的挑战，以及现有方法存在的时间不一致性、几何不一致性、感知伪影、运动不规则性、高计算成本等问题。这些问题在游戏、VR/AR等领域尤为重要，因为随着这些领域对逼真、表现力和可控3D虚拟人的需求快速增长，创建结合高视觉保真度和精细运动控制的数字虚拟人变得非常困难，而现有方法常受'果冻效应'和'雅努斯问题'困扰，严重影响动画质量和实际应用性。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考通过明确分离3D结构建模和运动控制，再通过基于扩散的重定位重新组合来实现高效4D生成。他们采用三阶段方法：生成静态3D虚拟人、编码目标运动、通过扩散重定位机制统一起来。该方法借鉴了Direct-3D的三平面表示和MoMask在运动生成方面的先进工作，但进行了改进以更好地与扩散模型兼容，特别是将3D骨骼转换为适合扩散模型的2D三平面骨骼编码。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过基于扩散的三平面重定位实现高效4D虚拟人生成，明确分离3D结构建模和运动控制，使用三平面表示保持几何一致性避免'雅努斯问题'，并用骨骼引导条件确保时间一致性和解剖准确性。整体流程：1)使用基于三平面的扩散模型从文本生成静态3D虚拟人；2)将3D骨骼转换为2D三平面骨骼编码；3)使用条件扩散模型根据初始特征和骨骼表示生成重定位特征序列；4)将重定位特征解码为动态、时间平滑和视角一致的4D虚拟人。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': "关键创新点：1)提出完整的文本到4D生成流水线，几秒内生成高质量4D虚拟人；2)引入基于扩散的3D虚拟人重定位模型；3)直接从大规模资产学习3D结构和运动先验。相比之前工作，不依赖计算昂贵的优化循环和2D先验，使用三平面直接编码3D信息将时间从小时减到秒，结合三平面表示和扩散重定位确保时间一致性和几何正确性，直接在三平面特征空间条件化于3D骨骼而非依赖网格变形，消除了'果冻效应'和视角不一致性问题。", '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TriDiff-4D通过基于扩散的三平面重定位技术，实现了从文本到高质量4D虚拟人的快速生成，解决了现有方法中的时间不一致性、几何不一致性和计算效率低等问题。'}


### 论文摘要

With the increasing demand for 3D animation, generating high-fidelity, controllable 4D avatars from textual descriptions remains a significant challenge. Despite notable efforts in 4D generative modeling, existing methods exhibit fundamental limitations that impede their broader applicability, including temporal and geometric inconsistencies, perceptual artifacts, motion irregularities, high computational costs, and limited control over dynamics. To address these challenges, we propose TriDiff-4D, a novel 4D generative pipeline that employs diffusion-based triplane re-posing to produce high-quality, temporally coherent 4D avatars. Our model adopts an auto-regressive strategy to generate 4D sequences of arbitrary length, synthesizing each 3D frame with a single diffusion process. By explicitly learning 3D structure and motion priors from large-scale 3D and motion datasets, TriDiff-4D enables skeleton-driven 4D generation that excels in temporal consistency, motion accuracy, computational efficiency, and visual fidelity. Specifically, TriDiff-4D first generates a canonical 3D avatar and a corresponding motion sequence from a text prompt, then uses a second diffusion model to animate the avatar according to the motion sequence, supporting arbitrarily long 4D generation. Experimental results demonstrate that TriDiff-4D significantly outperforms existing methods, reducing generation time from hours to seconds by eliminating the optimization process, while substantially improving the generation of complex motions with high-fidelity appearance and accurate 3D geometry.

---

## 137. Solving Spatial Supersensing Without Spatial Supersensing

**论文链接:** [http://arxiv.org/abs/2511.16655v1](http://arxiv.org/abs/2511.16655v1)

**作者:** Vishaal Udandarao, Shyamgopal Karthik, Surabhi S. Nath, Andreas Hochlehnert, Matthias Bethge, Ameya Prabhu

**发布时间:** 2025-11-20

**备注:** Tech Report

### GPT解析

### 总结

本文对Cambrian-S进行了批判性分析，发现其基准测试和推理方法存在严重缺陷，无法可靠评估空间超感能力。

### 背景

Cambrian-S旨在通过空间超感改进视频世界模型，引入了两个基准测试(VSI-Super-Recall和VSI-Super-Counting)以及针对每个基准的定制预测感知推理策略。

### 目的

对Cambrian-S在基准测试设计和推理策略两个方面的有效性进行评估和批判性分析。

### 方法

1) 引入NoSense基线方法，丢弃时间结构仅使用词袋SigLIP模型；2) 设计VSC-Repeat测试，将视频与自身连接1-5次而不改变独特对象数量。

### 主要发现

1) NoSense基线在VSR基准上达到95%准确率，表明该基准无需空间认知即可几乎被解决；2) VSC-Repeat测试导致Cambrian-S准确率从42%降至0%，表明其推理方法依赖捷径而非真正的空间超感。

### 结论

当前VSI-Super基准测试尚未可靠测量空间超感；Cambrian-S的性能提升源于无意中利用基准测试中的捷径，而非稳健的空间超感能力。

### 翻译

Cambrian-S旨在通过空间超感改进视频世界模型迈出第一步，通过引入(i)两个基准测试VSI-Super-Recall(VSR)和VSI-Super-Counting(VSC)，以及(ii)针对每个基准定制的预测感知推理策略。在本工作中，我们对Cambrian-S在这两个方面的表现进行了批判性分析。首先，我们引入了一个简单的基线NoSense，它丢弃了几乎所有时间结构，仅使用词袋SigLIP模型，却近乎完美地解决了VSR，即使在4小时长的视频上也达到95%的准确率。这表明像VSR这样的基准测试可以在没有空间认知、世界建模或空间超感的情况下几乎被解决。其次，我们假设Cambrian-S提出的定制推理方法很可能利用了基准测试中的捷径。我们通过在VSC基准测试上进行一个简单的健全性检查来说明这一点，称为VSC-Repeat：我们将每个视频与自身连接1-5次，这不会改变独特对象的数目。然而，这种简单的扰动完全使Cambrian-S的平均相对准确率从42%下降到0%。一个执行空间超感并整合跨经验信息的系统应该能够识别同一场景的视图并保持对象计数预测不变；相反，Cambrian-S推理算法在很大程度上依赖于VSC基准测试中的一个捷径，即房间永远不会被重访。综上所述，我们的发现表明(i)当前的VSI-Super基准测试尚未可靠地测量空间超感，以及(ii)Cambrian-S使用的预测感知推理方法是通过无意中利用捷径而非稳健的空间超感来提高性能。我们在附录A中包含了Cambrian-S作者的回应，以提供与我们的主张平衡的观点。我们在https://github.com/bethgelab/supersanity发布了我们的代码。


### 论文摘要

Cambrian-S aims to take the first steps towards improving video world models with spatial supersensing by introducing (i) two benchmarks, VSI-Super-Recall (VSR) and VSI-Super-Counting (VSC), and (ii) bespoke predictive sensing inference strategies tailored to each benchmark. In this work, we conduct a critical analysis of Cambrian-S across both these fronts. First, we introduce a simple baseline, NoSense, which discards almost all temporal structure and uses only a bag-of-words SigLIP model, yet near-perfectly solves VSR, achieving 95% accuracy even on 4-hour videos. This shows benchmarks like VSR can be nearly solved without spatial cognition, world modeling or spatial supersensing. Second, we hypothesize that the tailored inference methods proposed by Cambrian-S likely exploit shortcut heuristics in the benchmark. We illustrate this with a simple sanity check on the VSC benchmark, called VSC-Repeat: We concatenate each video with itself 1-5 times, which does not change the number of unique objects. However, this simple perturbation entirely collapses the mean relative accuracy of Cambrian-S from 42% to 0%. A system that performs spatial supersensing and integrates information across experiences should recognize views of the same scene and keep object-count predictions unchanged; instead, Cambrian-S inference algorithm relies largely on a shortcut in the VSC benchmark that rooms are never revisited. Taken together, our findings suggest that (i) current VSI-Super benchmarks do not yet reliably measure spatial supersensing, and (ii) predictive-sensing inference recipes used by Cambrian-S improve performance by inadvertently exploiting shortcuts rather than from robust spatial supersensing. We include the response from the Cambrian-S authors (in Appendix A) to provide a balanced perspective alongside our claims. We release our code at: https://github.com/bethgelab/supersanity

---

## 138. TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming

**论文链接:** [http://arxiv.org/abs/2511.16642v1](http://arxiv.org/abs/2511.16642v1)

**作者:** Zeyuan Yin, Xiaoming Liu

**发布时间:** 2025-11-20

**备注:** NeurIPS 2025

### GPT解析

### 总结

本文提出了一种名为TRIM的3D高斯扩散模型加速方法，通过轨迹缩减和实例掩码去噪策略提高效率，同时保持输出质量。

### 背景

当前3D高斯扩散模型由于高斯基元数量庞大，导致去噪和后处理耗时，生成速度慢，沿采样轨迹的可扩展性有限。

### 目的

提高3D扩散模型的效率，在不影响输出质量的同时加速推理，并支持推理时的扩展。

### 方法

提出TRIM（轨迹缩减和实例掩码去噪），一种训练后方法，结合时间和空间修剪策略。开发轻量级选择器模型评估潜在高斯基元实现早期轨迹缩减；引入实例掩码去噪通过过滤冗余背景区域减少计算量。

### 主要发现

大量实验和分析表明TRIM显著提高了3D生成的效率和质量。

### 结论

TRIM是一种有效的后训练方法，可以在不牺牲输出质量的情况下提高3D扩散模型的效率。

### 翻译

最近的3D高斯扩散模型进展因高斯基元数量庞大而面临耗时的去噪和后处理问题，导致生成速度慢且沿采样轨迹的可扩展性有限。为提高3D扩散模型的效率，我们提出TRIM（轨迹缩减和实例掩码去噪），一种结合时间和空间修剪策略的训练后方法，可在不妥协输出质量的同时加速推理，并支持3D高斯扩散模型的推理时扩展。我们开发了一个轻量级选择器模型来评估从多个采样噪声中推导出的潜在高斯基元，通过选择具有高质量潜力的候选者实现早期轨迹缩减。此外，我们引入实例掩码去噪，通过过滤冗余背景区域来修剪可学习的高斯基元，减少每个去噪步骤的推理计算。大量实验和分析证明TRIM显著提高了3D生成的效率和质量。源代码可在提供的链接获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D高斯扩散模型在推理过程中的低效性问题。由于需要处理大量高斯基元，导致去噪和后处理过程耗时过长，限制了生成速度和模型的可扩展性。这个问题在现实中很重要，因为3D生成技术在电影制作、游戏设计和虚拟现实等领域有广泛应用，而当前模型的效率不足阻碍了这些技术的实际应用和普及。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有3D扩散模型中的两个主要低效点来设计方法：1)在轨迹层面，增加采样轨迹提高质量但计算成本高；2)在标记层面，不必要地优化透明背景区域导致效率低下。作者借鉴了2D扩散模型中的推理时扩展概念、大型语言模型的轨迹选择策略、DINO的块级注意力机制以及图像生成中的扩散蒸馏技术，设计了时间修剪和空间修剪两种策略来解决这些问题。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是通过'轨迹减少和实例掩码去噪'同时处理时间轴和空间轴上的冗余。时间轴上提前选择高质量轨迹减少不必要的去噪过程；空间轴上识别并去除背景区域的高斯基元。整体流程分为三阶段：1)轨迹减少阶段：采样多种噪声生成多个轨迹，用选择器评估并选择最有潜力的轨迹；2)实例掩码去噪阶段：检测实例掩码，渐进式扩展掩码区域，合并背景标记；3)去噪后修正阶段：用掩码修正高斯基元参数，消除背景伪影。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出轨迹级别和标记级别的修剪框架；2)开发轻量级选择器模型早期识别高质量轨迹；3)设计训练自由的实例掩码去噪机制；4)提出渐进式掩码扩展调度器确保平滑过渡；5)结合时空修剪同时提高效率。相比之前工作，TRIM通过早期选择减少计算量而非端到端扩展；是训练自由的即插即用方法而非需架构更新；专门针对3D高斯扩散模型优化；同时提高质量和效率而非仅关注其中一方面。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'TRIM通过时间轴上的轨迹减少和空间轴上的实例掩码去噪，显著提高了3D高斯扩散模型的推理效率，同时增强了生成质量，为3D内容创作提供了更实用的工具。'}


### 论文摘要

Recent advances in 3D Gaussian diffusion models suffer from time-intensive denoising and post-denoising processing due to the massive number of Gaussian primitives, resulting in slow generation and limited scalability along sampling trajectories. To improve the efficiency of 3D diffusion models, we propose $\textbf{TRIM}$ ($\textbf{T}$rajectory $\textbf{R}$eduction and $\textbf{I}$nstance $\textbf{M}$ask denoising), a post-training approach that incorporates both temporal and spatial trimming strategies, to accelerate inference without compromising output quality while supporting the inference-time scaling for Gaussian diffusion models. Instead of scaling denoising trajectories in a costly end-to-end manner, we develop a lightweight selector model to evaluate latent Gaussian primitives derived from multiple sampled noises, enabling early trajectory reduction by selecting candidates with high-quality potential. Furthermore, we introduce instance mask denoising to prune learnable Gaussian primitives by filtering out redundant background regions, reducing inference computation at each denoising step. Extensive experiments and analysis demonstrate that TRIM significantly improves both the efficiency and quality of 3D generation. Source code is available at $\href{https://github.com/zeyuanyin/TRIM}{link}$.

---

## 139. A Core-Collapse Supernova Neutrino Parameterization with Enhanced Physical Interpretability

**论文链接:** [http://arxiv.org/abs/2511.16631v1](http://arxiv.org/abs/2511.16631v1)

**作者:** Haihao Shi, Zhenyang Huang, Junda Zhou, Guoliang Lü, Xuefei Chen

**发布时间:** 2025-11-20

**备注:** 48 pages, 31 figures, It has been accepted by APJS

### GPT解析

### 总结

研究团队引入了一种具有明确物理动机的超新星中微子能量谱新型参数化方法，核心参数τ(t)量化了爆炸过程中的特征热扩散区域。该方法对SN1987A数据提供了统计显著的拟合，并约束了未观测到的低能部分谱。在3D核心坍缩超新星模拟中，τ(t)的时间演化能够区分成功与失败的爆炸。研究还估计SN 1987A的前身星质量约为19个太阳质量，并发现τ(t)与引力波应变幅度之间存在强烈的协同共同演化，表明爆炸的热力学状态同时记录在中微子通量和能量谱形状中。

### 背景

超新星中微子能量谱的研究对于理解超新星爆炸机制至关重要，尤其是对于SN1987A这样的历史事件，以及通过3D模拟研究核心坍缩过程。

### 目的

开发一种具有明确物理动机的超新星中微子能量谱参数化方法，约束未观测到的低能部分谱，并应用于区分不同类型的超新星爆炸。

### 方法

引入以τ(t)为核心参数的新型参数化方法，量化爆炸过程中的特征热扩散区域；应用于SN1987A数据；在3D核心坍缩超新星模拟中测试模型；使用平滑保序回归估计前身星质量。

### 主要发现

1) 参数化方法对SN1987A数据提供统计显著拟合；2) τ(t)时间演化能区分成功与失败爆炸；3) SN 1987A前身星质量约19个太阳质量；4) τ(t)与引力波应变幅度有强烈协同共同演化；5) 爆炸热力学状态同时记录在中微子通量和能量谱形状中。

### 结论

该框架为解码未来银河系超新星的详细核心动力学和多信使过程提供了有价值的工具。

### 翻译

我们引入了一种具有明确物理动机的超新星中微子能量谱的新型参数化方法。其核心参数τ(t)量化了爆炸过程中的特征热扩散区域。当应用于历史性的SN1987A数据时，这种参数化方法产生了统计显著的拟合，并为未观测到的低能部分谱提供了稳健的约束。除了这一特定应用外，我们在一系列3D核心坍缩超新星模拟中展示了该模型的威力，发现τ(t)的时间演化明显地区分了成功与失败的爆炸。此外，通过应用平滑保序回归，我们将SN 1987A的前身星质量约束在约19个太阳质量，同时注意到这一估计对观测不确定性的敏感性。此外，在这些模拟中，τ(t)和引力波应变幅度显示出强烈的、协同的共同演化，直接将引擎的能量学演化与其几何不对称性联系起来。这意味着爆炸的热力学状态不仅体现在逃逸的中微子通量中，也记录在能量谱的形状中。因此，我们的框架为解码未来银河系超新星的详细核心动力学和多信使过程提供了有价值的工具。


### 论文摘要

We introduce a novel parameterization of supernova neutrino energy spectra with a clear physical motivation. Its central parameter, $τ(t)$, quantifies the characteristic thermal-diffusion area during the explosion. When applied to the historic SN1987A data, this parameterization yields statistically significant fits and provides robust constraints on the unobserved low-energy portion of the spectrum. Beyond this specific application, we demonstrate the model's power on a suite of 3D core-collapse supernova simulations, finding that the temporal evolution of $τ(t)$ distinctly separates successful from failed explosions. Furthermore, we constrain the progenitor mass of SN 1987A to approximately 19 solar masses by applying Smoothed Isotonic Regression, while noting the sensitivity of this estimate to observational uncertainties. Moreover, in these simulations, $τ(t)$ and the gravitational-wave strain amplitude display a strong, synergistic co-evolution, directly linking the engine's energetic evolution to its geometric asymmetry. This implies that the thermodynamic state of the explosion is imprinted not only on the escaping neutrino flux, but also recorded in the shape of the energy spectrum. Our framework therefore offers a valuable tool for decoding the detailed core dynamics and multi-messenger processes of future galactic supernovae.

---

## 140. TFCDiff: Robust ECG Denoising via Time-Frequency Complementary Diffusion

**论文链接:** [http://arxiv.org/abs/2511.16627v1](http://arxiv.org/abs/2511.16627v1)

**作者:** Pengxin Li, Yimin Zhou, Jie Min, Yirong Wang, Wei Liang, Wang Li

**发布时间:** 2025-11-20

### GPT解析

### 总结

TFCDiff是一种创新的ECG去噪方法，通过在DCT域操作并结合时序特征增强机制，有效去除多种噪声并保持信号保真度，在多个评估指标上表现优异，具有良好的泛化能力和实用性。

### 背景

动态心电图(ECG)读数容易受到物理活动产生的混合噪声干扰，包括基线漂移(BW)、肌肉伪影(MA)和电极运动伪影(EM)。

### 目的

开发一种方法来去除这种复杂噪声并重建高保真信号，提高诊断准确性。然而，多拍心电图段的去噪研究不足且存在技术挑战。

### 方法

提出了一种名为时频互补扩散(TFCDiff)的新方法，该方法在离散余弦变换(DCT)域中操作，并使用噪声信号的DCT系数作为条件输入。为了细化波形细节，他们引入了时序特征增强机制(TFEM)来增强时序表示并保留关键生理信息。

### 主要发现

在合成数据集上的比较实验表明，TFCDiff在五个评估指标上达到了最先进的性能。此外，TFCDiff在未见过的SimEMG数据库上表现出优异的泛化能力，超越了所有基准模型。TFCDiff处理原始10秒序列，并在灵活的随机混合噪声下保持鲁棒性，使其能够在高运动场景的可穿戴ECG监视器中即插即用部署。

### 结论

TFCDiff是一种有效的ECG去噪方法，能够处理复杂噪声并保留生理信息，适用于实际临床应用。

### 翻译

动态心电图(ECG)读数容易受到物理活动产生的混合噪声干扰，包括基线漂移(BW)、肌肉伪影(MA)和电极运动伪影(EM)。开发一种方法来去除这种复杂噪声并重建高保真信号对提高诊断准确性具有临床价值。然而，多拍心电图段的去噪研究仍然不足且存在技术挑战。为了解决这个问题，我们提出了时频互补扩散(TFCDiff)，一种在离散余弦变换(DCT)域中操作的新方法，使用噪声信号的DCT系数作为条件输入。为了细化波形细节，我们纳入了时序特征增强机制(TFEM)来增强时序表示并保留关键生理信息。在合成数据集上的比较实验表明，TFCDiff在五个评估指标上达到了最先进的性能。此外，TFCDiff在未见过的SimEMG数据库上表现出优异的泛化能力，超越了所有基准模型。值得注意的是，TFCDiff处理原始10秒序列，并在灵活的随机混合噪声下保持鲁棒性，使其能够在高运动场景的可穿戴ECG监视器中即插即用部署。源代码可在https://github.com/Miroircivil/TFCDiff获取。


### 论文摘要

Ambulatory electrocardiogram (ECG) readings are prone to mixed noise from physical activities, including baseline wander (BW), muscle artifact (MA), and electrode motion artifact (EM). Developing a method to remove such complex noise and reconstruct high-fidelity signals is clinically valuable for diagnostic accuracy. However, denoising of multi-beat ECG segments remains understudied and poses technical challenges. To address this, we propose Time-Frequency Complementary Diffusion (TFCDiff), a novel approach that operates in the Discrete Cosine Transform (DCT) domain and uses the DCT coefficients of noisy signals as conditioning input. To refine waveform details, we incorporate Temporal Feature Enhancement Mechanism (TFEM) to reinforce temporal representations and preserve key physiological information. Comparative experiments on a synthesized dataset demonstrate that TFCDiff achieves state-of-the-art performance across five evaluation metrics. Furthermore, TFCDiff shows superior generalization on the unseen SimEMG Database, outperforming all benchmark models. Notably, TFCDiff processes raw 10-second sequences and maintains robustness under flexible random mixed noise (fRMN), enabling plug-and-play deployment in wearable ECG monitors for high-motion scenarios. Source code is available at https://github.com/Miroircivil/TFCDiff.

---

## 141. TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding

**论文链接:** [http://arxiv.org/abs/2511.16595v1](http://arxiv.org/abs/2511.16595v1)

**作者:** Boshen Xu, Zihan Xiao, Jiaze Li, Jianzhong Ju, Zhenbo Luo, Jian Luan, Qin Jin

**发布时间:** 2025-11-20

**备注:** Project page: https://xuboshen.github.io/TimeViper

### GPT解析

### 总结

TimeViper是一种混合视觉-语言模型，用于解决长视频理解挑战，通过结合Mamba和Transformer架构，并引入TransV模块压缩视觉令牌，实现了对超长视频的高效处理。

### 背景

长视频理解需要高效的模型架构和有效处理扩展时间上下文的机制，传统方法在处理超长视频时面临挑战。

### 目的

开发一种能够处理超长视频的混合视觉-语言模型，解决视频理解中的效率和上下文处理问题。

### 方法

采用混合Mamba-Transformer主干架构，结合状态空间模型的效率和注意力机制的表达能力；提出TransV令牌信息传输模块，将视觉令牌压缩到指令令牌中，同时保持多模态理解能力。

### 主要发现

揭示了视觉到文本的信息聚合现象，随着LLM深度增加，信息从视觉令牌流向文本令牌，导致视觉令牌冗余；分析了Mamba和Transformer层的注意力行为，为混合模型可解释性提供新见解。

### 结论

TimeViper能够处理超过10,000帧的长达一小时的视频，在多个基准测试中与最先进模型竞争，是开发、解释和压缩混合Mamba-Transformer架构的初步步骤。

### 翻译

我们介绍了TimeViper，一种为解决长视频理解挑战而设计的混合视觉-语言模型。处理长视频需要高效的模型架构和有效处理扩展时间上下文的机制。为此，TimeViper采用了混合Mamba-Transformer主干，结合了状态空间模型的效率和注意力机制的表达能力。通过这种混合设计，我们揭示了视觉到文本的信息聚合现象，信息随着LLM深度增加从视觉令牌流向文本令牌，导致严重的视觉令牌冗余。受此观察启发，我们提出了TransV，一个令牌信息传输模块，将视觉令牌传输并压缩到指令令牌中，同时保持多模态理解能力。这种设计使TimeViper能够处理超过10,000帧的长达一小时的视频。在多个基准测试中的广泛实验表明，TimeViper在增加帧数的同时能与最先进的模型竞争。我们进一步分析了Mamba和Transformer层的注意力行为，为混合模型可解释性提供了新见解。这项工作是开发、解释和压缩混合Mamba-Transformer架构的初步步骤。


### 论文摘要

We introduce TimeViper, a hybrid vision-language model designed to tackle challenges of long video understanding. Processing long videos demands both an efficient model architecture and an effective mechanism for handling extended temporal contexts. To this end, TimeViper adopts a hybrid Mamba-Transformer backbone that combines the efficiency of state-space models with the expressivity of attention mechanisms. Through this hybrid design, we reveal the vision-to-text information aggregation phenomenon, where information progressively flows from vision tokens to text tokens across increasing LLM depth, resulting in severe vision token redundancy. Motivated by this observation, we propose TransV, a token information transfer module that transfers and compresses vision tokens into instruction tokens while maintaining multimodal understanding capabilities. This design enables TimeViper to process hour-long videos exceeding 10,000 frames. Extensive experiments across multiple benchmarks demonstrate that TimeViper competes with state-of-the-art models while extending frame numbers. We further analyze attention behaviors of both Mamba and Transformer layers, offering new insights into hybrid model interpretability. This work represents an initial step towards developing, interpreting, and compressing hybrid Mamba-Transformer architectures.

---

## 142. Adiabatic charge transport through non-Bloch bands

**论文链接:** [http://arxiv.org/abs/2511.16480v1](http://arxiv.org/abs/2511.16480v1)

**作者:** Dharana Joshi, Tanay Nag

**发布时间:** 2025-11-20

**备注:** Main text: 8 pages, 4 figures, SM: 13 pages, 9 figures

### GPT解析

### 总结

该研究探讨了扩展的Su-Schrieffer-Heeger模型中由非互易胞内跳跃介导的非厄米拓扑相，该模型包含次近邻跳跃。研究使用非布洛赫动量分析相边界，通过特征方程和规范自由度推导非布洛赫能带，并验证了体边界对应关系。研究还考察了绝热动力学，促进了对非厄米情景下绝热电荷输运的理解，并统一了静态和驱动情况下非布洛赫能带的概念。

### 背景

研究扩展的Su-Schrieffer-Heeger模型中的非厄米拓扑相，该模型包含次近邻跳跃，由非互易胞内跳跃介导。

### 目的

探索非厄米拓扑相的相边界，分析非布洛赫能带特性，研究绝热动力学和电荷输运，并统一静态和驱动情况下非布洛赫能带的概念。

### 方法

使用非布洛赫动量微观分析相边界，通过特征方程和规范自由度推导非布洛赫能带，研究开放边界条件下的边界对应关系，并考察绝热动力学。

### 主要发现

非布洛赫动量准确反映了体边界对应关系，解释了开放边界条件下的绕数分布；绝热动力学促进了非厄米情景下的绝热电荷输运概念；非布洛赫能带在时间演化过程中不经历能隙关闭时，保持了量化流；研究系统地统一了静态和驱动情况下非布洛赫能带的概念。

### 结论

研究系统地统一了静态和驱动情况下非布洛赫能带的概念，为理解非厄米拓扑相提供了新的视角。

### 翻译

我们探索了包含次近邻跳跃的扩展Su-Schrieffer-Heeger模型中由非互易胞内跳跃介导的非厄米拓扑相。我们使用非布洛赫动量微观分析了相边界，而非临界（临界）相与非布洛赫能带的带隙（无带隙）特性直接相关，这些能带是通过特征方程和规范自由度推导得出的。非布洛赫动量准确反映了体边界对应关系，解释了开放边界条件下的绕数分布。我们考察了绝热动力学，促进了非厄米情景下绝热电荷输运的概念，证明了时空Bott指数和非布洛赫陈数中的体边界对应关系。当非布洛赫能带在时间演化过程中不经历（经历）能隙关闭时，量化流得以保持（破坏）。我们的研究系统地统一了静态和驱动情况下非布洛赫能带的概念。


### 论文摘要

We explore the non-reciprocal intracell hopping mediated non-Hermitian topological phases of an extended Su-Schrieffer-Heeger model hosting second-nearest-neighbour hopping. We microscopically analyze the phase boundaries using the non-Bloch momentum while the off-critical (critical) phases are directly associated with the gapped (gapless) nature of the non-Bloch bands that we derive from the characteristic equation using the gauge freedom. The non-Bloch momentum accurately reflects the bulk boundary correspondence (BBC) explaining the winding number profile under open boundary conditions. We examine the adiabatic dynamics to promote the concept of adiabatic charge transport in a non-Hermitian scenario justifying the BBC in spatio-temporal Bott index and non-Bloch Chern number. Once the non-Bloch bands experience no (a) gap-closing during the evolution of time, quantized flow of is preserved (broken). Our study systematically unifies the concept of non-Bloch bands for both static and driven situations.

---

## 143. VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference

**论文链接:** [http://arxiv.org/abs/2511.16449v1](http://arxiv.org/abs/2511.16449v1)

**作者:** Ziyan Liu, Yeqiu Chen, Hongyi Cai, Tao Lin, Shuo Yang, Zheng Liu, Bo Zhao

**发布时间:** 2025-11-20

### GPT解析

### 总结

VLA-Pruner是一种创新的token剪枝方法，专门针对VLA模型的双重系统特性设计，通过同时考虑语义级和动作级的重要性，实现了高效且高性能的视觉token处理。

### 背景

Vision-Language-Action (VLA)模型在具身AI方面显示出巨大潜力，但处理连续视觉流的计算成本高限制了实时部署。现有的VLM特定的token pruning方法仅基于语义显著性指标选择token，忽略了VLA的双重系统特性，导致偏向语义线索保留、丢弃动作生成关键信息。

### 目的

解决现有方法偏向语义线索保留、丢弃动作生成关键信息的问题，提出一种适应VLA模型双重系统特性的token剪枝方法。

### 方法

提出VLA-Pruner，一种通用的即插即用VLA特定token剪枝方法，利用机器人操作中的时间连续性，采用双重级别的重要性标准（视觉-语言预填充注意力和动作解码注意力），并提出双重级别token选择策略，在给定计算预算下自适应保留紧凑、信息丰富的视觉token集。

### 主要发现

VLA-Pruner在多种VLA架构和多样化机器人任务上实现了最先进的性能。

### 结论

VLA-Pruner成功解决了现有token pruning方法在VLA模型上的局限性，通过考虑语义理解和动作执行的双重需求，提高了VLA模型的效率。

### 翻译

Vision-Language-Action (VLA)模型在具身AI方面显示出巨大潜力，但处理连续视觉流的沉重计算成本严重限制了它们的实时部署。标记剪枝（保留重要的视觉token并丢弃冗余的）已成为加速Vision-Language Models (VLMs)的有效方法，为高效的VLA提供了解决方案。然而，这些VLM特定的标记剪枝方法仅基于语义显著性指标（如预填充注意力）选择token，而忽视了VLA固有的高级语义理解和低级动作执行的双重系统特性。因此，这些方法偏向于保留语义线索的token，丢弃了动作生成的关键信息，并显著降低了VLA性能。为了弥合这一差距，我们提出了VLA-Pruner，一种通用的即插即用VLA特定标记剪枝方法，该方法符合VLA模型的双重系统特性，并利用机器人操作中的时间连续性。具体来说，VLA-Pruner采用双重级别的重要性标准来保留视觉token：视觉-语言预填充注意力用于语义级相关性，通过时间平滑估计的动作解码注意力用于动作级重要性。基于这一标准，VLA-Pruner提出了一种新颖的双重级别token选择策略，在给定计算预算下自适应保留用于语义理解和动作执行的紧凑、信息丰富的视觉token集。实验表明，VLA-Pruner在多种VLA架构和多样化的机器人任务上实现了最先进的性能。


### 论文摘要

Vision-Language-Action (VLA) models have shown great promise for embodied AI, yet the heavy computational cost of processing continuous visual streams severely limits their real-time deployment. Token pruning (keeping salient visual tokens and dropping redundant ones) has emerged as an effective approach for accelerating Vision-Language Models (VLMs), offering a solution for efficient VLA. However, these VLM-specific token pruning methods select tokens based solely on semantic salience metrics (e.g., prefill attention), while overlooking the VLA's intrinsic dual-system nature of high-level semantic understanding and low-level action execution. Consequently, these methods bias token retention toward semantic cues, discard critical information for action generation, and significantly degrade VLA performance. To bridge this gap, we propose VLA-Pruner, a versatile plug-and-play VLA-specific token prune method that aligns with the dual-system nature of VLA models and exploits the temporal continuity in robot manipulation. Specifically, VLA-Pruner adopts a dual-level importance criterion for visual token retention: vision-language prefill attention for semantic-level relevance and action decode attention, estimated via temporal smoothing, for action-level importance. Based on this criterion, VLA-Pruner proposes a novel dual-level token selection strategy that adaptively preserves a compact, informative set of visual tokens for both semantic understanding and action execution under given compute budget. Experiments show that VLA-Pruner achieves state-of-the-art performance across multiple VLA architectures and diverse robotic tasks.

---

## 144. FreqFlow: Long-term forecasting using lightweight flow matching

**论文链接:** [http://arxiv.org/abs/2511.16426v1](http://arxiv.org/abs/2511.16426v1)

**作者:** Seyed Mohamad Moghadas, Bruno Cornelis, Adrian Munteanu

**发布时间:** 2025-11-20

**备注:** Accepted at EurIPS, 2025

### GPT解析

### 总结

FreqFlow是一种基于频域条件流匹配的新型多变量时间序列预测框架，通过在频域中操作而非时域，实现了高效、准确的预测，模型参数量仅为89k，比现有扩散模型小一个数量级，同时预测性能提升7%。

### 背景

多变量时间序列预测在城市交通、资源管理和气候建模等领域有广泛应用。现有基于去噪扩散的生成模型虽然能捕捉复杂数据分布，但因迭代随机采样过程导致计算开销大，限制了实时部署能力，且在处理高维、非平稳和多尺度周期性模式时表现脆弱。

### 目的

开发一种轻量级、高效的多变量时间序列预测方法，解决现有模型的计算效率和性能问题，实现更准确的预测，特别是在处理长期预测任务时。

### 方法

FreqFlow将预测问题转换到频域，通过单个复值线性层学习建模幅度和相位偏移，利用复数乘法捕捉时间动态。该方法将MTS信号分解为趋势、季节性和残差分量，流匹配机制专门为残差学习设计，通过常微分方程积分实现单通道确定性采样。

### 主要发现

在真实世界的交通速度、流量和流量数据集上，FreqFlow实现了最先进的预测性能，平均RMSE改进7%，同时比现有方法更快且参数效率更高。模型仅包含89k参数，比竞争的基于扩散的模型小一个数量级。

### 结论

FreqFlow通过频域条件流匹配方法，成功解决了现有MTS预测模型的计算效率和性能问题，提供了一个轻量级、高效的预测框架，特别适合处理高维、非平稳和多尺度周期性的时间序列数据。

### 翻译

多变量时间序列预测是从城市交通和资源管理到气候建模等应用的基础。虽然最近基于去噪扩散的生成模型在捕获复杂数据分布方面取得了最先进性能，但由于迭代随机采样过程导致显著的计算开销，限制了实时部署。此外，这些模型在处理真实传感器网络特有的高维、非平稳和多尺度周期性模式时可能表现脆弱。我们引入了FreqFlow，这是一个新颖的框架，利用频域条件流匹配进行确定性多变量时间序列预测。与传统在时域操作的方法不同，FreqFlow将预测问题转换到频域，通过单个复值线性层学习建模幅度和相位偏移。这种频域公式使模型能够通过复数乘法有效地捕获时间动态，对应缩放和时间平移。 resulting架构异常轻量，仅89k参数-比竞争的基于扩散的模型小一个数量级-同时通过常微分方程积分实现单通道确定性采样。我们的方法将多变量时间序列信号分解为趋势、季节性和残差分量，流匹配机制专门为残差学习设计，以提高长期预测准确性。在真实世界的交通速度、流量和流量数据集上的大量实验表明，FreqFlow实现了最先进的预测性能，平均RMSE改进7%，同时比现有方法更快且参数效率更高。


### 论文摘要

Multivariate time-series (MTS) forecasting is fundamental to applications ranging from urban mobility and resource management to climate modeling. While recent generative models based on denoising diffusion have advanced state-of-the-art performance in capturing complex data distributions, they suffer from significant computational overhead due to iterative stochastic sampling procedures that limit real-time deployment. Moreover, these models can be brittle when handling high-dimensional, non-stationary, and multi-scale periodic patterns characteristic of real-world sensor networks. We introduce FreqFlow, a novel framework that leverages conditional flow matching in the frequency domain for deterministic MTS forecasting. Unlike conventional approaches that operate in the time domain, FreqFlow transforms the forecasting problem into the spectral domain, where it learns to model amplitude and phase shifts through a single complex-valued linear layer. This frequency-domain formulation enables the model to efficiently capture temporal dynamics via complex multiplication, corresponding to scaling and temporal translations. The resulting architecture is exceptionally lightweight with only 89k parameters - an order of magnitude smaller than competing diffusion-based models-while enabling single-pass deterministic sampling through ordinary differential equation (ODE) integration. Our approach decomposes MTS signals into trend, seasonal, and residual components, with the flow matching mechanism specifically designed for residual learning to enhance long-term forecasting accuracy. Extensive experiments on real-world traffic speed, volume, and flow datasets demonstrate that FreqFlow achieves state-of-the-art forecasting performance, on average 7\% RMSE improvements, while being significantly faster and more parameter-efficient than existing methods

---

## 145. Grain growth in protoplanetary disks in the Upper Scorpius revealed by millimeter-wave spectral indices

**论文链接:** [http://arxiv.org/abs/2511.16405v1](http://arxiv.org/abs/2511.16405v1)

**作者:** Tau Bito, Akimasa Kataoka, Takahiro Ueda, Luca Ricci, Tilman Birnstiel, John Carpenter

**发布时间:** 2025-11-20

**备注:** Accepted for publication in PASJ. This version corresponds to the accepted manuscript

### GPT解析

### 总结

通过测量毫米波光谱中的尘埃大小，本研究分析了上天蝎座区域23个原行星盘的数据，发现其平均光谱指数为2.09±0.10，与其他年轻区域相当或略低。通过盘演化模型解释，表明即使在后期演化阶段，大量尘埃质量仍保留在低温外盘区域。

### 背景

毫米波光谱测量尘埃大小可直接限制原行星盘中的颗粒生长。在金牛座、蛇夫座和狼蛛座等年轻恒星形成区域（年龄1-3百万年）测量的光谱指数低至2-3，表明盘中的颗粒比星际介质中的颗粒大得多。

### 目的

分析上天蝎座区域原行星盘的毫米波光谱数据，研究其尘埃颗粒大小和演化特征，并与其他年轻区域进行比较。

### 方法

使用ALMA存档数据观测上天蝎座区域23个盘，波长为2.9毫米（波段3），角分辨率为3.3角秒×2.1角秒，均方根噪声低于0.075 mJy/束。结合文献中同一目标波段7的通量值，计算光谱指数。构建简单的盘演化模型来解释观测结果。

### 主要发现

上天蝎座区域盘的平均光谱指数为2.09±0.10，与其他较年轻区域的光谱指数相等或略小。观测结果通过盘内半径增加的模型得到最佳重现，表明即使在后期演化阶段，大量尘埃质量仍保留在尘埃温度低于20K的外盘区域。

### 结论

这些发现为颗粒生长和原行星盘的时间演化提供了关键见解，表明即使在演化晚期，外盘中仍存在大量尘埃质量。

### 翻译

通过测量毫米波光谱中的尘埃大小，可以直接限制原行星盘中的颗粒生长。在多个年轻恒星形成区域（如金牛座、蛇夫座和狼蛛座，年龄为1-3百万年）测量的0.88毫米和2.9毫米之间的光谱指数低至2-3，表明盘中的颗粒比星际介质中的颗粒大得多。在本研究中，我们分析了上天蝎座区域23个盘的ALMA存档数据。观测波长为2.9毫米（波段3），角分辨率为3.3角秒×2.1角秒，不足以分辨目标，几乎所有源的均方根噪声低于0.075 mJy/束。结合文献中同一目标波段7的通量值，我们发现上天蝎座区域盘的平均光谱指数为2.09±0.10，与其他较年轻区域的光谱指数相等或略小。为解释金牛座、蛇夫座、狼蛛座和上天蝎座区域盘通量和光谱指数之间的关系，我们构建了简单的盘演化模型。观测结果通过盘内半径增加的模型得到最佳重现，这表明即使在后期演化阶段，大量尘埃质量必须保留在尘埃温度低于20K的外盘区域。这些发现为颗粒生长和原行星盘的时间演化提供了关键见解。


### 论文摘要

The measurement of dust size from millimeter-wavelength spectra provides direct constraints on grain growth in protoplanetary disks. The spectral indices between 0.88 mm and 2.9 mm have been measured in multiple young star-forming regions, such as Taurus, Ophiuchus, and Lupus, which have ages of 1-3 Myr. These spectral indices are as low as 2-3, suggesting that grains in disks are much larger than those in the interstellar medium. In this study, we analyze the ALMA archival data of 23 disks in the Upper Scorpius region. The observed wavelength is 2.9 mm in Band 3, the angular resolution is 3.3 arcsec x 2.1 arcsec, which is not high enough to resolve the targets, and the rms noise is below 0.075 mJy beam$^{-1}$ for almost all sources. Together with the literature values of the Band 7 fluxes of the same targets, we find that the average spectral index of the disks in the Upper Scorpius region is $α_\mathrm{mm}=2.09 \pm 0.10$, which is equal to or slightly smaller than those at the other younger regions. To explain the relationship between the fluxes and spectral indices of the disks in the Taurus, Ophiuchus, Lupus, and Upper Scorpius regions, we construct simple disk evolution models. The observations are best reproduced by models in which the inner radius of the disk increases. This suggests that a substantial amount of dust mass must persist in the outer disk regions where the dust temperature is lower than 20 K even at late evolutionary stages. These findings offer key insights into the grain growth and the temporal evolution of protoplanetary disks.

---

## 146. SwiTrack: Tri-State Switch for Cross-Modal Object Tracking

**论文链接:** [http://arxiv.org/abs/2511.16227v1](http://arxiv.org/abs/2511.16227v1)

**作者:** Boyue Xu, Ruichao Hou, Tongwei Ren, Dongming Zhou, Gangshan Wu, Jinde Cao

**发布时间:** 2025-11-20

### GPT解析

### 总结

本文提出了一种名为SwiTrack的新型状态切换框架，用于解决跨模态目标跟踪中的特征提取不全面和目标漂移问题。该框架通过三个专用流处理不同模态的数据，实现了更高的精确率和成功率，同时保持实时性能。

### 背景

跨模态目标跟踪(CMOT)是一个新兴任务，在视频流在不同模态间切换时保持目标一致性，每帧只有一种模态可用，主要关注RGB-近红外(RGB-NIR)跟踪。现有方法将并行的RGB和NIR分支连接到共享主干网络，限制了独特模态特定特征的全面提取，且无法有效处理目标漂移问题，特别是在输入不可靠的情况下。

### 目的

提出一种新的状态切换框架来重新定义CMOT，解决现有方法中特征提取不全面和目标漂移的问题，提高跨模态目标跟踪的性能和鲁棒性。

### 方法

部署三个专用流：RGB帧由视觉编码器处理；NIR帧通过NIR门控适配器与视觉编码器耦合，逐步校准共享潜在空间特征；对于无效模态，使用一致性轨迹预测模块利用时空线索估计目标移动；采用动态模板重建迭代更新模板特征；使用相似性对齐损失强化特征一致性。

### 主要发现

在最新基准测试上取得了最先进的性能，精确率提高7.2%，成功率提高4.3%，同时保持每秒65帧的实时跟踪速度。

### 结论

SwiTrack框架有效解决了跨模态目标跟踪中的关键挑战，通过专用流和状态切换机制提高了跟踪性能，能够处理不可靠输入并减少目标漂移。

### 翻译

跨模态目标跟踪(CMOT)是一个新兴任务，在视频流在不同模态间切换时保持目标一致性，每帧只有一种模态可用，主要集中在RGB-近红外(RGB-NIR)跟踪上。现有方法通常将并行的RGB和NIR分支连接到共享主干网络，这限制了独特模态特定特征的全面提取，并且无法处理目标漂移问题，特别是在输入不可靠的情况下。在本文中，我们提出了SwiTrack，一种新的状态切换框架，通过部署三个专用流重新定义了CMOT。具体来说，RGB帧由视觉编码器处理，而NIR帧通过NIR门控适配器进行处理，该适配器与视觉编码器耦合，逐步校准共享的潜在空间特征，从而产生更鲁棒的跨模态表示。对于无效模态，一致性轨迹预测模块利用时空线索来估计目标移动，确保鲁棒跟踪并减轻漂移。此外，我们采用动态模板重建来迭代更新模板特征，并使用相似性对齐损失来强化特征一致性。在最新基准上的实验结果表明，我们的跟踪器实现了最先进的性能，精确率和成功率分别提高了7.2%和4.3%，同时保持每秒65帧的实时跟踪速度。代码和模型可在https://github.com/xuboyue1999/SwiTrack.git获取。


### 论文摘要

Cross-modal object tracking (CMOT) is an emerging task that maintains target consistency while the video stream switches between different modalities, with only one modality available in each frame, mostly focusing on RGB-Near Infrared (RGB-NIR) tracking. Existing methods typically connect parallel RGB and NIR branches to a shared backbone, which limits the comprehensive extraction of distinctive modality-specific features and fails to address the issue of object drift, especially in the presence of unreliable inputs. In this paper, we propose SwiTrack, a novel state-switching framework that redefines CMOT through the deployment of three specialized streams. Specifically, RGB frames are processed by the visual encoder, while NIR frames undergo refinement via a NIR gated adapter coupled with the visual encoder to progressively calibrate shared latent space features, thereby yielding more robust cross-modal representations. For invalid modalities, a consistency trajectory prediction module leverages spatio-temporal cues to estimate target movement, ensuring robust tracking and mitigating drift. Additionally, we incorporate dynamic template reconstruction to iteratively update template features and employ a similarity alignment loss to reinforce feature consistency. Experimental results on the latest benchmarks demonstrate that our tracker achieves state-of-the-art performance, boosting precision rate and success rate gains by 7.2\% and 4.3\%, respectively, while maintaining real-time tracking at 65 frames per second. Code and models are available at https://github.com/xuboyue1999/SwiTrack.git.

---

## 147. ART: A Graph-based Framework for Investigating Illicit Activity in Monero via Address-Ring-Transaction Structures

**论文链接:** [http://arxiv.org/abs/2511.16192v1](http://arxiv.org/abs/2511.16192v1)

**作者:** Andrea Venturi, Imanol Jerico-Yoldi, Francesco Zola, Raul Orduna

**发布时间:** 2025-11-20

**备注:** Paper accepted @ BLOCKCHAIN & CRYPTOCURRENCY CONFERENCE (B2C'2025)

### GPT解析

### 总结

这篇论文提出了一种基于图的方法，用于分析Monero加密货币交易中的犯罪行为模式，以帮助执法机构在隐私保护区块链环境中进行调查。

### 背景

随着执法机构在加密货币取证方面取得进展，犯罪分子越来越多地使用' mixin'服务或基于隐私的加密货币（如Monero）来隐藏非法资金流动。Monero因其强大的隐私保护和不可追踪性而成为犯罪分子的首选，使得传统的区块链分析方法失效。

### 目的

理解犯罪分子在Monero中的行为和运作模式，以支持未来的调查策略并破坏非法活动。

### 方法

研究人员通过案例研究，利用一种新颖的基于图的方法，从与已发现的犯罪活动相关的Monero交易中提取结构和时间模式。他们通过从标记的交易构建地址-环-交易图，提取结构和时间特征，并使用这些特征训练机器学习模型来检测类似的行为模式。

### 主要发现

通过构建地址-环-交易图并提取结构和时间特征，可以训练机器学习模型来检测可能突出犯罪作案手法的行为模式。

### 结论

这种方法代表了开发支持隐私保护区块链生态系统中调查工作的分析工具的部分第一步，有助于执法机构在Monero等隐私保护加密货币中识别犯罪活动。

### 翻译

随着执法机构在加密货币取证方面的进步，旨在隐藏非法资金流动的犯罪分子越来越多地转向' mixin'服务或基于隐私的加密货币。Monero因其强大的隐私保护和不可追踪性而成为首选，使得传统的区块链分析无效。因此，理解犯罪分子在Monero中的行为和运作模式具有挑战性，但对支持未来调查策略和破坏非法活动至关重要。在这项工作中，我们提出了一个案例研究，利用一种新颖的基于图的方法，从与已发现的犯罪活动相关的Monero交易中提取结构和时间模式。通过从标记的交易构建地址-环-交易图，我们提取结构和时间特征，并使用它们训练能够检测可能突出犯罪作案手法的类似行为模式的机器学习模型。这代表了开发支持隐私保护区块链生态系统中调查工作的分析工具的部分第一步。


### 论文摘要

As Law Enforcement Agencies advance in cryptocurrency forensics, criminal actors aiming to conceal illicit fund movements increasingly turn to "mixin" services or privacy-based cryptocurrencies. Monero stands out as a leading choice due to its strong privacy preserving and untraceability properties, making conventional blockchain analysis ineffective. Understanding the behavior and operational patterns of criminal actors within Monero is therefore challenging and it is essential to support future investigative strategies and disrupt illicit activities. In this work, we propose a case study in which we leverage a novel graph-based methodology to extract structural and temporal patterns from Monero transactions linked to already discovered criminal activities. By building Address-Ring-Transaction graphs from flagged transactions, we extract structural and temporal features and use them to train Machine Learning models capable of detecting similar behavioral patterns that could highlight criminal modus operandi. This represents a first partial step toward developing analytical tools that support investigative efforts in privacy-preserving blockchain ecosystems

---

## 148. Achieving Skilled and Reliable Daily Probabilistic Forecasts of Wind Power at Subseasonal-to-Seasonal Timescales over France

**论文链接:** [http://arxiv.org/abs/2511.16164v1](http://arxiv.org/abs/2511.16164v1)

**作者:** Eloi Lindas, Yannig Goude, Philippe Ciais

**发布时间:** 2025-11-20

### GPT解析

### 总结

本研究提出了一种将ECMWF亚季节到季节性天气预报转换为风能预测的方法，预测时间范围从1天到46天，分辨率为每天，并通过后处理解决了天气预报中的偏差和离散度问题。该方法在预测性能和校准方面均表现出色。

### 背景

准确可靠的风能预测对电网稳定性、供需平衡和市场风险管理至关重要。短期天气预报已被广泛用于短期可再生能源预测，但更长预测时间范围的预测仍需研究。尽管亚季节到季节性天气概率预测有所进展，但通常需要时间和空间聚合才能获得合理预测技能。

### 目的

开发一种预测流程，将ECMWF的亚季节到季节性天气预报转换为风能预测，并对结果进行后处理以提高预测质量。

### 方法

提出了一种预测流程，将ECMWF的亚季节到季节性天气预报转换为风能预测，并对生成的功率集合进行后处理，以考虑天气预报的偏差和离散度不足的问题。

### 主要发现

该方法在连续排序概率技能评分和集合均方误差方面比气候学基线提高了50%，并且在15到46天的预测时间范围内提供了近乎完美的预测校准。

### 结论

所提出的方法能够有效地将亚季节到季节性天气预报转换为高质量的风能预测，适用于较长的预测时间范围。

### 翻译

准确可靠的风能预测对电网稳定性、平衡供需和市场风险管理至关重要。尽管短期天气预报已被广泛用于提供短期可再生能源预测，但涉及更长预测时间范围的预测仍需研究。尽管最近在亚季节到季节性天气概率预测方面取得了进展，但它们通常需要时间和空间聚合才能获得合理的预测技能。在本研究中，我们提出了一种预测流程，能够将ECMWF的亚季节到季节性天气预报转换为风能预测，预测时间范围从1天到46天，分辨率为每天。该框架还包括对生成的功率集合进行后处理，以考虑天气预报的偏差和离散度不足的问题。我们证明，在连续排序概率技能评分和集合均方误差方面，我们的方法比气候学基线提高了50%，同时在15至46天的预测时间范围内提供了近乎完美的预测校准。


### 论文摘要

Accurate and reliable wind power forecasts are crucial for grid stability, balancing supply and demand, and market risk management. Even though short-term weather forecasts have been thoroughly used to provide short-term renewable power predictions, forecasts involving longer prediction horizons still need investigations. Despite the recent progress in subseasonal-to-seasonal weather probabilistic forecasting, their use for wind power prediction usually involves both temporal and spatial aggregation achieve reasonable skill. In this study, we present a forecasting pipeline enabling to transform ECMWF subseasonal-to-seasonal weather forecasts into wind power forecasts for lead times ranging from 1 day to 46 days at daily resolution. This framework also include post-processing of the resulting power ensembles to account for the biases and lack of dispersion of the weather forecasts. We show that our method is able to outperform a climatological baseline by 50 % in terms of both Continuous Ranked Probability Skill Score and Ensemble Mean Squared Error while also providing near perfect calibration of the forecasts for lead times ranging from 15 to 46 days.

---

## 149. Thinking-while-Generating: Interleaving Textual Reasoning throughout Visual Generation

**论文链接:** [http://arxiv.org/abs/2511.16671v1](http://arxiv.org/abs/2511.16671v1)

**作者:** Ziyu Guo, Renrui Zhang, Hongyu Li, Manyuan Zhang, Xinyan Chen, Sifan Wang, Yan Feng, Peng Pei, Pheng-Ann Heng

**发布时间:** 2025-11-20

**备注:** Project Page: https://think-while-gen.github.io Code: https://github.com/ZiyuGuo99/Thinking-while-Generating

### GPT解析

### 总结

本文提出了Thinking-while-Generating (TwiG)框架，这是第一个在视觉生成过程中实现文本推理共同演进的框架，通过动态交互产生更具上下文感知力和语义丰富的视觉输出。

### 背景

视觉生成领域的最新进展越来越多地探索推理能力的整合，但现有方法仅在生成前或生成后进行文本推理，缺乏生成过程中的实时多模态交互。

### 目的

引入TwiG框架，实现在视觉生成过程中交错进行文本推理，以指导即将生成的内容并反思已生成的内容。

### 方法

TwiG框架允许在视觉内容逐步生成时交错进行文本推理，并研究了三种策略：零样本提示、在TwiG-50K数据集上进行监督微调、以及通过TwiG-GRPO策略进行强化学习。

### 主要发现

这种文本推理与视觉生成的动态交互能够产生更具上下文感知力和语义丰富的视觉输出。

### 结论

希望这项工作能够启发进一步研究，探索交错文本推理以增强视觉生成。

### 翻译

视觉生成的最新进展越来越多地探索推理能力的整合。它们将文本推理（即思考）整合在生成过程之前（作为预规划）或之后（作为后优化），但在生成过程中缺乏实时的多模态交互。在这项初步研究中，我们引入了Thinking-while-Generating (TwiG)，这是第一个在视觉生成过程中实现文本推理共同演进的框架。随着视觉内容的逐步生成，文本推理被交错进行，以指导即将生成的局部区域并反思已合成的区域。这种动态交互产生更具上下文感知力和语义丰富的视觉输出。为了揭示这一框架的潜力，我们研究了三种候选策略：零样本提示、在我们策划的TwiG-50K数据集上进行监督微调(SFT)、以及通过定制的TwiG-GRPO策略进行强化学习(RL)，每种策略都为交错推理的动态特性提供了独特的见解。我们希望这项工作能够启发进一步研究交错文本推理以增强视觉生成。代码将在以下地址发布：https://github.com/ZiyuGuo99/Thinking-while-Generating。


### 论文摘要

Recent advances in visual generation have increasingly explored the integration of reasoning capabilities. They incorporate textual reasoning, i.e., think, either before (as pre-planning) or after (as post-refinement) the generation process, yet they lack on-the-fly multimodal interaction during the generation itself. In this preliminary study, we introduce Thinking-while-Generating (TwiG), the first interleaved framework that enables co-evolving textual reasoning throughout the visual generation process. As visual content is progressively generating, textual reasoning is interleaved to both guide upcoming local regions and reflect on previously synthesized ones. This dynamic interplay produces more context-aware and semantically rich visual outputs. To unveil the potential of this framework, we investigate three candidate strategies, zero-shot prompting, supervised fine-tuning (SFT) on our curated TwiG-50K dataset, and reinforcement learning (RL) via a customized TwiG-GRPO strategy, each offering unique insights into the dynamics of interleaved reasoning. We hope this work inspires further research into interleaving textual reasoning for enhanced visual generation. Code will be released at: https://github.com/ZiyuGuo99/Thinking-while-Generating.

---

## 150. Mind the Gap: Bridging Prior Shift in Realistic Few-Shot Crop-Type Classification

**论文链接:** [http://arxiv.org/abs/2511.16218v1](http://arxiv.org/abs/2511.16218v1)

**作者:** Joana Reuss, Ekaterina Gikalo, Marco Körner

**发布时间:** 2025-11-20

**备注:** 7 pages, 4 figures

### GPT解析

### 总结

本文提出了一种名为Dirichlet Prior Augmentation (DirPA)的新方法，用于解决农业数据集中的类别不平衡问题，通过模拟目标域的未知标签分布偏移，提高模型在真实世界条件下的泛化能力。

### 背景

现实世界中的农业数据分布通常存在严重的类别不平衡问题，遵循长尾分布；作物类型分类的标记数据本质稀缺且获取成本高；在有限数据情况下，训练集常被构建为人为平衡的（特别是在小样本学习中），无法反映真实世界条件；这种训练与测试标签分布的不匹配会降低真实世界的泛化能力。

### 目的

提出一种新方法来模拟目标域的未知标签分布偏移，从而提高模型在真实世界条件下的泛化能力。

### 方法

提出Dirichlet Prior Augmentation (DirPA)方法，在模型训练过程中主动模拟目标域的未知标签分布偏移；将现实世界分布建模为Dirichlet分布的随机变量，在小样本学习期间执行先验增强。

### 主要发现

实验表明，DirPA通过作为动态特征正则化器，成功转移决策边界并稳定训练过程。

### 结论

DirPA方法有效解决了农业数据集中类别不平衡导致的泛化能力下降问题，通过模拟真实世界的标签分布偏移，提高了模型在真实环境中的表现。

### 翻译

现实世界中的农业分布通常遭受严重的类别不平衡问题，通常遵循长尾分布。作物类型分类的标记数据本质上是稀缺的，且获取成本高昂。在使用此类有限数据时，训练集通常被构建为人为平衡的——特别是在小样本学习的情况下——这无法反映真实世界的条件。这种不匹配导致训练和测试标签分布之间的偏移，降低了真实世界的泛化能力。为了解决这个问题，我们提出了Dirichlet Prior Augmentation (DirPA)，一种新颖的方法，在模型训练过程中主动模拟目标域的未知标签分布偏移。具体来说，我们将现实世界分布建模为Dirichlet分布的随机变量，在小样本学习期间有效地执行先验增强。我们的实验表明，DirPA通过作为动态特征正则化器，成功转移决策边界并稳定训练过程。


### 论文摘要

Real-world agricultural distributions often suffer from severe class imbalance, typically following a long-tailed distribution. Labeled datasets for crop-type classification are inherently scarce and remain costly to obtain. When working with such limited data, training sets are frequently constructed to be artificially balanced -- in particular in the case of few-shot learning -- failing to reflect real-world conditions. This mismatch induces a shift between training and test label distributions, degrading real-world generalization. To address this, we propose Dirichlet Prior Augmentation (DirPA), a novel method that simulates an unknown label distribution skew of the target domain proactively during model training. Specifically, we model the real-world distribution as Dirichlet-distributed random variables, effectively performing a prior augmentation during few-shot learning. Our experiments show that DirPA successfully shifts the decision boundary and stabilizes the training process by acting as a dynamic feature regularizer.

---

## 151. Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight

**论文链接:** [http://arxiv.org/abs/2511.16175v1](http://arxiv.org/abs/2511.16175v1)

**作者:** Yi Yang, Xueqi Li, Yiyang Chen, Jin Song, Yihan Wang, Zipeng Xiao, Jiadi Su, You Qiaoben, Pengfei Liu, Zhijie Deng

**发布时间:** 2025-11-20

### GPT解析

### 总结

Mantis是一种新型框架，通过解耦视觉预测与主干网络，结合元查询和扩散Transformer(DiT)头部，解决了VLA模型中视觉预测与语言监督之间的平衡问题，提高了模型在机器人任务中的表现。

### 背景

视觉-语言-动作(VLA)模型可以利用视觉信号补充稀疏动作监督，但直接预测高维视觉状态会分散模型能力并导致高昂训练成本；压缩视觉状态为更紧凑监督信号会导致信息瓶颈；现有方法常因忽视语言监督而表现较差的理解和推理能力。

### 目的

开发一种新型框架，解决VLA模型中视觉预测与语言监督之间的平衡问题，提高模型在机器人任务中的表现，同时保持对语言的理解和推理能力。

### 方法

提出Mantis框架，通过解耦视觉预测与主干网络，结合元查询和扩散Transformer(DiT)头部。通过残差连接将当前视觉状态提供给DiT，使元查询能够自动捕捉定义视觉轨迹的潜在动作，从而增强显式动作的学习，减轻VLA主干网络的负担。

### 主要发现

在人类操作视频、机器人演示和图像文本对上进行预训练后，Mantis在LIBERO基准测试上实现了96.7%的成功率，超过了强大的基线模型，并且表现出高收敛速度。真实世界评估显示，Mantis在指令遵循能力、对未见指令的泛化能力和推理能力方面优于领先的开放源码VLA模型π_{0.5}。

### 结论

Mantis框架成功解决了VLA模型中视觉预测与语言监督之间的平衡问题，通过解耦视觉预测与主干网络，提高了模型在机器人任务中的表现，同时保持了对语言的理解和推理能力。代码和权重的发布将进一步促进相关研究的发展。

### 翻译

最近的视觉-语言-动作(VLA)模型进展表明，视觉信号可以有效补充稀疏的动作监督。然而，让VLA直接预测高维视觉状态会分散模型能力并导致高昂的训练成本，而将视觉状态压缩为更紧凑的监督信号则不可避免地会导致信息瓶颈。此外，由于忽视语言监督，现有方法通常存在理解和推理能力差的问题。本文介绍了Mantis，这是一种具有解耦视觉预测(DVF)的新型框架，用于解决这些问题。具体而言，Mantis通过元查询和扩散Transformer(DiT)头部组合，将视觉预测与主干网络解耦。通过残差连接将当前视觉状态提供给DiT，简单的下一状态预测目标使元查询能够自动捕捉定义视觉轨迹的潜在动作，从而增强显式动作的学习。这种解耦减轻了VLA主干网络的负担，使其能够通过语言监督保持理解和推理能力。实验表明，在人类操作视频、机器人演示和图像文本对上进行预训练后，Mantis在微调后在LIBERO基准测试上实现了96.7%的成功率，超过了强大的基线模型，同时表现出高收敛速度。真实世界的评估显示，Mantis在指令遵循能力、对未见指令的泛化能力和推理能力方面优于领先的开放源码VLA模型π_{0.5}。代码和权重已发布以支持开源社区。


### 论文摘要

Recent advances in Vision-Language-Action (VLA) models demonstrate that visual signals can effectively complement sparse action supervisions. However, letting VLA directly predict high-dimensional visual states can distribute model capacity and incur prohibitive training cost, while compressing visual states into more compact supervisory signals inevitably incurs information bottlenecks. Moreover, existing methods often suffer from poor comprehension and reasoning capabilities due to the neglect of language supervision. This paper introduces Mantis, a novel framework featuring a Disentangled Visual Foresight (DVF) to tackle these issues. Specifically, Mantis decouples visual foresight prediction from the backbone with the combination of meta queries and a diffusion Transformer (DiT) head. With the current visual state provided to the DiT via a residual connection, a simple next-state prediction objective enables the meta queries to automatically capture the latent actions that delineate the visual trajectory, and hence boost the learning of explicit actions. The disentanglement reduces the burden of the VLA backbone, enabling it to maintain comprehension and reasoning capabilities through language supervision. Empirically, pretrained on human manipulation videos, robot demonstrations, and image-text pairs, Mantis achieves a 96.7% success rate on LIBERO benchmark after fine-tuning, surpassing powerful baselines while exhibiting high convergence speed. Real-world evaluations show that Mantis outperforms $π_{0.5}$, a leading open-source VLA model, particularly in instruction-following capability, generalization to unseen instructions, and reasoning ability. Code and weights are released to support the open-source community.

---

## 152. Integrated 4D/5D Digital-Twin Framework for Cost Estimation and Probabilistic Schedule Control: A Texas Mid-Rise Case Study

**论文链接:** [http://arxiv.org/abs/2511.15711v1](http://arxiv.org/abs/2511.15711v1)

**作者:** Atena Khoshkonesh, Mohsen Mohammadagha, Navid Ebrahimi

**发布时间:** 2025-11-04

### GPT解析

### 总结

该研究提出了一种创新的4D/5D数字孪生框架，整合多种先进技术解决传统建筑项目管理方法的局限性，通过自动化和智能分析显著提高了成本和进度控制的准确性和效率。

### 背景

美国建筑项目中持续存在成本和进度超支问题，传统基于文档的估算和确定性关键路径法(CPM)调度方法在不确定性下缺乏灵活性，难以适应动态的现场条件。

### 目的

开发一个集成的4D/5D数字孪生框架，统一建筑信息建模(BIM)、自然语言处理(NLP)、实景捕获、计算机视觉、贝叶斯风险建模和深度强化学习(DRL)技术，用于建筑成本和进度控制。

### 方法

系统通过六个方面实现项目控制功能自动化：(a)使用基于Transformer的NLP将合同文档映射到标准化成本项；(b)将摄影测量和LiDAR数据与BIM对齐计算挣值；(c)从现场图像推导实时活动完成情况；(d)通过贝叶斯推断和蒙特卡洛模拟更新概率性CPM预测；(e)使用DRL进行自适应资源分配；(f)提供4D/5D决策沙盒进行预测分析。案例研究使用RSMeans城市成本指数和劳工统计局工资数据进行本地化成本调整。

### 主要发现

估算劳动力减少43%，加班时间减少6%(91小时)，项目完成时间与128天的P50概率预测相匹配。

### 结论

该数字孪生框架确认提高了估算准确性和响应能力，有效解决了传统建筑项目管理方法的局限性。

### 翻译

美国建筑项目中持续的成本和进度超支暴露了传统基于文档的估算和确定性关键路径法(CPM)调度的局限性，这些方法在不确定性下缺乏灵活性，且落后于动态的现场条件。本研究提出了一种集成的4D/5D数字孪生框架，统一了建筑信息建模(BIM)、自然语言处理(NLP)、实景捕获、计算机视觉、贝叶斯风险建模和深度强化学习(DRL)技术，用于建筑成本和进度控制。该系统通过以下方式自动化项目控制功能：(a)使用基于Transformer的NLP将合同文档映射到标准化成本项(加权F1得分为0.883)；(b)将摄影测量和LiDAR数据与BIM对齐以计算挣值；(c)从现场图像推导实时活动完成情况(微准确率为0.891)；(d)通过贝叶斯推断和蒙特卡洛模拟更新概率性CPM预测；(e)使用DRL进行自适应资源分配(采用率为75%)；(f)提供4D/5D决策沙盒进行预测分析。德克萨斯州中层建筑的案例研究展示了使用RSMeans城市成本指数和劳工统计局工资数据进行本地化成本调整。结果显示估算劳动力减少43%，加班时间减少6%(91小时)，项目完成时间与128天的P50概率预测相匹配，证实提高了估算准确性和响应能力。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决美国建筑项目中普遍存在的成本和进度超支问题，以及传统基于文档的估算方法和确定性关键路径法(CPM)调度在面对不确定性和动态现场条件时的局限性。这个问题在现实中非常重要，因为它直接影响项目效益和行业可持续发展；在研究中也很重要，因为现有方法在集成先进传感、分析和决策框架方面存在不足，数字孪生技术与实际项目控制的连接仍有困难，成本管理也面临验证、可追溯性和合同集成的碎片化问题。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者在现有技术发展基础上设计方法，首先识别了研究中的差距和机会，特别是在集成、自动化和现场部署方面的不足。他们设计了一个统一的框架，将估算、进度监控和调度整合为闭环控制系统。该方法借鉴了多项现有工作：4D BIM的可视化和协调能力、数字孪生和基于视觉的进度监控技术、Scan-to-BIM的客观进度量化能力、5D BIM的成本估算功能、AI在估算和调度中的应用，以及贝叶斯和蒙特卡洛方法的风险预测能力。作者不是简单堆砌这些技术，而是将它们有机整合，解决各自孤立应用时的局限性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个集成的4D/5D数字孪生框架，统一多种技术实现建筑项目的成本估算和概率进度控制，通过自动化和智能分析连接设计意图与现场表现，形成闭环控制系统，并保持人类专业知识与AI能力的结合。整体实现流程包括：1)使用NLP将项目规范和图纸自动映射到成本项目；2)通过摄影测量和LiDAR数据与BIM模型对齐计算挣值；3)利用计算机视觉从现场图像实时获取活动完成情况；4)通过贝叶斯推断和蒙特卡洛模拟更新CPM预测；5)应用深度强化学习进行自适应资源分配；6)提供4D/5D决策沙盒进行预测分析。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)集成框架统一了4D/5D BIM、数字孪生、NLP、计算机视觉等多种技术；2)实现了从设计到现场执行的闭环控制系统；3)建立了自动化传感到调度的集成；4)提供概率进度控制而非确定性预测；5)保持人类参与的AI应用以提高实用性；6)使用本地化成本数据提高估算准确性。相比之前工作的不同：传统4D BIM缺乏与实际项目控制的集成；现有数字孪生技术与进度成本控制连接不足；传统成本估算依赖手动工作；传统CPM是确定性的而非概率性的；现有AI方法在建筑领域的实际应用有限。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过集成多种先进技术创建了一个4D/5D数字孪生框架，显著提高了建筑项目成本估算的准确性、进度预测的可靠性以及资源分配的效率，实现了从设计到现场执行的闭环智能控制。'}


### 论文摘要

Persistent cost and schedule overruns in U.S. building projects expose limitations of conventional, document-based estimating and deterministic Critical Path Method (CPM) scheduling, which remain inflexible under uncertainty and lag dynamic field conditions. This study presents an integrated 4D/5D digital-twin framework unifying Building Information Modeling (BIM), natural language processing (NLP), reality capture, computer vision, Bayesian risk modeling, and deep reinforcement learning (DRL) for construction cost and schedule control. The system automates project-control functions by: (a) mapping contract documents to standardized cost items using transformer-based NLP (0.883 weighted F1 score); (b) aligning photogrammetry and LiDAR data with BIM to compute earned value; (c) deriving real-time activity completion from site imagery (0.891 micro accuracy); (d) updating probabilistic CPM forecasts via Bayesian inference and Monte Carlo simulation; (e) using DRL for adaptive resource allocation (75% adoption rate); and (f) providing 4D/5D decision sandbox for predictive analysis. A Texas mid-rise case study demonstrates localized cost adjustment using RSMeans City Cost Index and Bureau of Labor Statistics wage data. Results show 43% reduction in estimating labor, 6% overtime reduction (91 hours), and project completion matching P50 probabilistic forecast of 128 days, confirming improved estimation accuracy and responsiveness.

---

