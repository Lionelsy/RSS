# 今日论文推荐 - 2025-12-05

共 138 篇论文

---

## 1. Light-X: Generative 4D Video Rendering with Camera and Illumination Control

**论文链接:** [http://arxiv.org/abs/2512.05115v1](http://arxiv.org/abs/2512.05115v1)

**作者:** Tianqi Liu, Zhaoxi Chen, Zihao Huang, Shaocong Xu, Saining Zhang, Chongjie Ye, Bohan Li, Zhiguo Cao, Wei Li, Hao Zhao, Ziwei Liu

**发布时间:** 2025-12-04

**备注:** Project Page: https://lightx-ai.github.io/

### GPT解析

### 总结

本文提出了Light-X，一个视频生成框架，能够从单目视频中实现可控渲染，同时控制视角和光照。该框架通过解耦几何和光照信号，以及合成训练数据的方法，解决了现有方法在光照保真度和时间一致性之间的权衡问题。

### 背景

光照控制方面的最新进展将基于图像的方法扩展到了视频领域，但仍面临光照保真度和时间一致性之间的权衡。超越简单重照明的关键步骤是对相机轨迹和光照的联合控制，因为视觉动态本质上由几何和光照共同塑造。

### 目的

开发一个视频生成框架，实现从单目视频中同时控制视角和光照的渲染。

### 方法

1) 提出解耦设计，将几何和光照信号分离：通过沿用户定义的相机轨迹投影动态点云来捕获几何和运动，通过将重照明帧一致地投影到相同几何中来提供光照线索。2) 引入Light-Syn，一种基于退化且具有逆映射的管道，从野外单目素材中合成训练对，生成涵盖静态、动态和AI生成场景的数据集。

### 主要发现

Light-X在联合相机-光照控制方面优于基线方法，并且在文本和背景条件设置下，超越了先前的视频重照明方法。

### 结论

Light-X框架通过解耦几何和光照信号，以及合成多样化的训练数据，有效解决了视频生成中视角和光照联合控制的挑战。

### 翻译

最近在光照控制方面的进展将基于图像的方法扩展到了视频领域，但仍面临光照保真度和时间一致性之间的权衡。超越简单重照明的关键步骤是对相机轨迹和光照的联合控制，因为视觉动态本质上由几何和光照共同塑造。为此，我们提出了Light-X，一个视频生成框架，能够从单目视频中实现可控渲染，同时控制视角和光照。1) 我们提出了一种解耦设计，将几何和光照信号分离：通过沿用户定义的相机轨迹投影动态点云来捕获几何和运动，而通过将重照明帧一致地投影到相同几何中来提供光照线索。这些明确、细粒度的线索实现了有效的解耦并指导高质量的光照。2) 为了解决缺乏多视角和多光照视频配对的问题，我们引入了Light-Syn，一种基于退化且具有逆映射的管道，从野外单目素材中合成训练对。这种策略生成了一个涵盖静态、动态和AI生成场景的数据集，确保了鲁棒的训练。大量实验表明，Light-X在联合相机-光照控制方面优于基线方法，并且在文本和背景条件设置下，超越了先前的视频重照明方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从单目视频生成具有可控相机轨迹和光照条件的4D视频的问题。这个问题很重要，因为真实世界的场景是丰富、动态且高维的，由几何、运动和光照共同塑造，而单目视频只记录了这种复杂性的2D投影。实现可控的视频生成可以让我们从新视角和多样化光照下重温素材，支持沉浸式AR/VR体验和灵活的电影制作流程，是向真实世界场景生成建模迈出的关键一步。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将问题分解为相机控制和光照控制两个独立但相关的部分，采用解耦设计思路。相机控制方面借鉴了使用动态点云作为几何先验的工作；光照控制方面使用了IC-Light进行图像重光照；架构上基于扩散模型进行视频生成。由于缺乏配对的多视角多光照视频数据，作者设计了Light-Syn数据合成流程。总的来说，作者整合了多个现有技术的思想，但将它们巧妙结合，形成了统一的框架。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过解耦相机控制和光照控制，使用动态点云作为几何先验，重投影帧作为光照线索，并基于退化的数据合成解决数据稀缺问题。整体流程：1)输入源视频；2)使用IC-Light对一帧重光照；3)估计深度构建动态点云和重光照点云；4)将点云沿指定相机轨迹投影；5)提取光照令牌；6)将线索与噪声输入DiT块进行条件去噪；7)通过VAE解码器重建符合目标轨迹和光照的高保真视频。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个实现相机轨迹和光照联合控制的视频生成框架；2)解耦的条件设计分离几何/运动与光照；3)Light-Syn数据合成管道解决训练数据稀缺；4)全局光照控制模块确保一致性；5)支持多种光照条件。相比之前工作，Light-X不仅支持光照编辑还支持相机轨迹控制，平衡了光照保真度和时间一致性，通过解耦设计更好地分离几何/运动和光照因素，提高了泛化能力。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Light-X首次实现了从单目视频生成具有可控相机轨迹和光照条件的4D视频，通过解耦的条件设计和基于退化的数据合成，解决了联合控制相机和光照的关键挑战，为沉浸式AR/VR体验和灵活的电影制作提供了新工具。'}


### 论文摘要

Recent advances in illumination control extend image-based methods to video, yet still facing a trade-off between lighting fidelity and temporal consistency. Moving beyond relighting, a key step toward generative modeling of real-world scenes is the joint control of camera trajectory and illumination, since visual dynamics are inherently shaped by both geometry and lighting. To this end, we present Light-X, a video generation framework that enables controllable rendering from monocular videos with both viewpoint and illumination control. 1) We propose a disentangled design that decouples geometry and lighting signals: geometry and motion are captured via dynamic point clouds projected along user-defined camera trajectories, while illumination cues are provided by a relit frame consistently projected into the same geometry. These explicit, fine-grained cues enable effective disentanglement and guide high-quality illumination. 2) To address the lack of paired multi-view and multi-illumination videos, we introduce Light-Syn, a degradation-based pipeline with inverse-mapping that synthesizes training pairs from in-the-wild monocular footage. This strategy yields a dataset covering static, dynamic, and AI-generated scenes, ensuring robust training. Extensive experiments show that Light-X outperforms baseline methods in joint camera-illumination control and surpasses prior video relighting methods under both text- and background-conditioned settings.

---

## 2. Geometric Data Science

**论文链接:** [http://arxiv.org/abs/2512.05040v1](http://arxiv.org/abs/2512.05040v1)

**作者:** Olga D Anosova, Vitaliy A Kurlin

**发布时间:** 2025-12-04

**备注:** Questions and comments are welcome at vitaliy.kurlin@gmail.com. The latest version is at http://kurlin.org/Geometric-Data-Science-book.pdf

### GPT解析

### 总结

本书介绍了几何数据科学这一新研究领域，通过对有限点集和周期点集的分类，扩展了门捷列夫表到完整的晶体宇宙。

### 背景

几何数据科学是一个新兴研究领域，其中数据可以通过几何测量来表示任何真实物体。

### 目的

介绍几何数据科学，并提供对有限点集和周期点集的分类方法。

### 方法

开发算法对点集进行分类，处理排列复杂性和噪声问题，构建模空间。

### 主要发现

1) 对任意欧几里得空间中无序点集在刚体运动下的完整连续分类；2) 周期点集的层次化不变量；3) 晶体等距原理的验证；4) 所得模空间包含所有已知和尚未发现的周期性晶体。

### 结论

几何数据科学为理解和分类基于几何测量的真实物体提供了新框架，特别是对点集和晶体的分类。

### 翻译

本书介绍了几何数据科学这一新的研究领域，其中数据可以通过几何测量来表示任何真实物体。本书第一部分专注于有限点集。最重要的成果是对任意欧几里得空间中无序点集在刚体运动下的完整连续分类。关键挑战是避免给定无序点排列导致的指数复杂性。对于固定的欧几里得空间维度，所有算法的时间复杂度与点数成多项式关系。本书第二部分在更困难的周期点集情况下进行类似分类，这些点集模拟原子尺度的所有周期性晶体。最重要的成果是从超快到完整的层次化不变量。关键挑战是解决晶体表示在几乎任何噪声下都会失效的不连续性问题。在所有主要材料数据库上的实验验证确认了晶体等距原理：任何真实的周期性晶体在刚体运动下所有周期结构共同模空间中都有唯一位置。所得模空间包含所有已知和尚未发现的周期性晶体，因此将门捷列夫表连续扩展到完整的晶体宇宙。


### 论文摘要

This book introduces the new research area of Geometric Data Science, where data can represent any real objects through geometric measurements.   The first part of the book focuses on finite point sets. The most important result is a complete and continuous classification of all finite clouds of unordered points under rigid motion in any Euclidean space. The key challenge was to avoid the exponential complexity arising from permutations of the given unordered points. For a fixed dimension of the ambient Euclidean space, the times of all algorithms for the resulting invariants and distance metrics depend polynomially on the number of points.   The second part of the book advances a similar classification in the much more difficult case of periodic point sets, which model all periodic crystals at the atomic scale. The most significant result is the hierarchy of invariants from the ultra-fast to complete ones. The key challenge was to resolve the discontinuity of crystal representations that break down under almost any noise. Experimental validation on all major materials databases confirmed the Crystal Isometry Principle: any real periodic crystal has a unique location in a common moduli space of all periodic structures under rigid motion. The resulting moduli space contains all known and not yet discovered periodic crystals and hence continuously extends Mendeleev's table to the full crystal universe.

---

## 3. A dynamic memory assignment strategy for dilation-based ICP algorithm on embedded GPUs

**论文链接:** [http://arxiv.org/abs/2512.04996v1](http://arxiv.org/abs/2512.04996v1)

**作者:** Qiong Chang, Weimin Wang, Junpei Zhong, Jun Miyazaki

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种针对高性能点云配准算法VANICP的内存优化策略，使其能够在资源受限的嵌入式GPU上轻量级执行。通过动态内存分配策略，优化了膨胀操作的内存使用，在保持原始性能的同时实现了超过97%的内存消耗减少。

### 背景

VANICP是一种最近发表的加速框架，显著提高了基于点云的应用的计算效率。它通过基于膨胀的信息传播机制将全局最近邻搜索转换为局部过程，大大降低了NNS的计算复杂度。然而，其原始实现需要大量内存，限制了在资源受限环境中的部署。

### 目的

解决VANICP原始实现内存消耗过大的问题，使其能够在资源受限的环境（如嵌入式系统）中部署，同时保持算法的高性能。

### 方法

提出了一种面向GPU的动态内存分配策略，优化了膨胀操作的内存使用，并基于此策略构建了VANICP框架的增强版本。

### 主要发现

通过所提出的方法，实现了超过97%的内存消耗减少，同时保持了原始性能。

### 结论

所提出的内存优化策略成功解决了VANICP在资源受限环境中的部署问题，使其能够在嵌入式GPU上轻量级执行，同时保持高性能。

### 翻译

本文提出了一种针对高性能点云配准算法VANICP的内存高效优化策略，使其能够在资源受限的嵌入式GPU上进行轻量级执行。VANICP是一种最近发表的加速框架，显著提高了基于点云的应用的计算效率。通过基于膨胀的信息传播机制将全局最近邻搜索转换为局部过程，VANICP大大降低了NNS的计算复杂度。然而，其原始实现需要大量内存，限制了其在资源受限环境（如嵌入式系统）中的部署。为解决此问题，我们提出了一种面向GPU的动态内存分配策略，优化了膨胀操作的内存使用。此外，基于此策略，我们构建了VANICP框架的增强版本，在保持原始性能的同时实现了超过97%的内存消耗减少。源代码已发布在：https://github.com/changqiong/VANICP4Em.git。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决VANICP算法在嵌入式GPU上运行时内存消耗过大的问题。原始VANICP需要预分配约2GB的固定内存空间来支持体素膨胀操作，这使得它无法在资源受限的嵌入式系统中部署。这个问题在现实中很重要，因为嵌入式系统通常有严格的计算资源、能量预算和移动性限制，而大内存消耗限制了点云注册技术在移动设备、机器人等边缘应用中的使用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了VANICP内存效率低下的根本原因：每个体素需要独立内存块但稀疏度差异大。他们思考理想的内存分配应动态适应每个体素的实际点数。设计上借鉴了VANICP的核心膨胀机制，将全局最近邻搜索转为局部过程；同时利用嵌入式GPU的统一内存架构，结合GPU并行处理能力（构建体素占用直方图）和CPU串行处理能力（计算内存偏移量），实现异构计算优化。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是动态内存分配策略，根据每个体素中实际包含的点数动态分配内存，而非预分配固定大小的内存块，同时采用间接寻址替代直接寻址。整体流程：1)体素化：将点云转为3D网格；2)并行直方图构建：在GPU上统计各体素点数；3)序列偏移计算：在CPU上计算各体素内存起始地址；4)动态内存分配：基于偏移量分配内存；5)膨胀操作：在动态内存上执行膨胀；6)最近邻搜索：在动态分配的内存上进行局部搜索。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)针对嵌入式GPU优化的动态内存分配策略；2)结合GPU并行和CPU串行的异构计算方法；3)基于内存偏移的间接寻址机制；4)实现超过97%的内存使用减少同时保持原始性能。相比之前工作，与原始VANICP相比从固定内存分配改为动态分配；与传统ICP相比通过膨胀机制提高效率；与其他加速方法相比专门针对嵌入式系统资源限制优化；与其他内存优化方法相比专门针对点云处理中的体素膨胀操作设计。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种针对嵌入式GPU的动态内存分配策略，将VANICP算法的内存消耗降低97%以上同时保持其高性能计算能力，使该算法能够在资源受限的嵌入式系统上有效运行。'}


### 论文摘要

This paper proposes a memory-efficient optimization strategy for the high-performance point cloud registration algorithm VANICP, enabling lightweight execution on embedded GPUs with constrained hardware resources. VANICP is a recently published acceleration framework that significantly improves the computational efficiency of point-cloud-based applications. By transforming the global nearest neighbor search into a localized process through a dilation-based information propagation mechanism, VANICP greatly reduces the computational complexity of the NNS. However, its original implementation demands a considerable amount of memory, which restricts its deployment in resource-constrained environments such as embedded systems. To address this issue, we propose a GPU-oriented dynamic memory assignment strategy that optimizes the memory usage of the dilation operation. Furthermore, based on this strategy, we construct an enhanced version of the VANICP framework that achieves over 97% reduction in memory consumption while preserving the original performance. Source code is published on: https://github.com/changqiong/VANICP4Em.git.

---

## 4. 论文ID: 2512.04966v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04966v1.json'

---

## 5. Hardware-aware Neural Architecture Search of Early Exiting Networks on Edge Accelerators

**论文链接:** [http://arxiv.org/abs/2512.04705v1](http://arxiv.org/abs/2512.04705v1)

**作者:** Alaa Zniber, Arne Symons, Ouassim Karrakchou, Marian Verhelst, Mounir Ghogho

**发布时间:** 2025-12-04

**备注:** Submitted to IEEE Transactions on Emerging Topics in Computing

### GPT解析

### 总结

本文提出了一种硬件感知的神经架构搜索框架，用于优化早期退出神经网络的设计，使其更适合资源受限的边缘环境。

### 背景

高性能计算和云技术促进了复杂深度学习模型的发展，但边缘设备对嵌入式智能的需求增长带来了严格的计算和能源限制，使大规模模型难以部署。

### 目的

开发一种硬件感知的神经架构搜索框架，系统整合量化和硬件资源分配的影响，优化网络骨干中早期退出点的位置。

### 方法

提出硬件感知的神经架构搜索框架，系统整合量化和硬件资源分配的影响，优化网络中早期退出点的位置。

### 主要发现

在CIFAR-10数据集上的实验表明，该框架可以发现相比传统静态网络减少50%以上计算成本的架构，更适合资源受限的边缘环境。

### 结论

所提出的NAS框架能有效优化早期退出神经网络设计，提高其在边缘计算环境中的适用性。

### 翻译

高性能计算和云技术的进步使得复杂的深度学习模型得以发展。然而，边缘设备对嵌入式智能日益增长的需求施加了严格的计算和能源限制，挑战了这些大规模模型的部署。早期退出神经网络已成为一种有前景的解决方案，允许根据输入复杂度动态终止推理以提高效率。尽管有潜力，EENN的性能受到边缘加速器异质性和量化约束的严重影响，这影响了准确性、能源效率和延迟。然而，关于针对边缘硬件自动优化EENN设计的研究仍然有限。为了弥补这一差距，我们提出了一种硬件感知的神经架构搜索框架，系统地整合量化和硬件资源分配的影响，以优化网络骨干中早期退出点的位置。在CIFAR-10数据集上的实验结果表明，我们的NAS框架可以发现架构，相比传统静态网络实现超过50%的计算成本减少，使它们更适合在资源受限的边缘环境中部署。


### 论文摘要

Advancements in high-performance computing and cloud technologies have enabled the development of increasingly sophisticated Deep Learning (DL) models. However, the growing demand for embedded intelligence at the edge imposes stringent computational and energy constraints, challenging the deployment of these large-scale models. Early Exiting Neural Networks (EENN) have emerged as a promising solution, allowing dynamic termination of inference based on input complexity to enhance efficiency. Despite their potential, EENN performance is highly influenced by the heterogeneity of edge accelerators and the constraints imposed by quantization, affecting accuracy, energy efficiency, and latency. Yet, research on the automatic optimization of EENN design for edge hardware remains limited. To bridge this gap, we propose a hardware-aware Neural Architecture Search (NAS) framework that systematically integrates the effects of quantization and hardware resource allocation to optimize the placement of early exit points within a network backbone. Experimental results on the CIFAR-10 dataset demonstrate that our NAS framework can discover architectures that achieve over a 50\% reduction in computational costs compared to conventional static networks, making them more suitable for deployment in resource-constrained edge environments.

---

## 6. STeP-Diff: Spatio-Temporal Physics-Informed Diffusion Models for Mobile Fine-Grained Pollution Forecasting

**论文链接:** [http://arxiv.org/abs/2512.04385v1](http://arxiv.org/abs/2512.04385v1)

**作者:** Nan Zhou, Weijie Hong, Huandong Wang, Jianfeng Zheng, Qiuhua Wang, Yali Song, Xiao-Ping Zhang, Yong Li, Xinlei Chen

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了一种创新的时空物理信息扩散模型（STeP-Diff），用于细粒度空气污染预测，通过结合物理规律和实际测量数据，显著提高了预测准确性。

### 背景

细粒度空气污染预测对城市管理和健康建筑发展至关重要。在移动平台上部署便携式传感器提供低成本、易维护且覆盖面广的数据收集方案，但非专用移动平台的随机移动模式导致数据不完整且时间不一致。

### 目的

开发一种能够从不完整和时间变化的数据中预测空气污染时空场的模型，确保预测既基于实际测量又符合污染扩散的基本物理规律。

### 方法

提出时空物理信息扩散模型（STeP-Diff），利用DeepONet建模测量空间序列，结合PDE信息扩散模型预测时空场，并通过PDE约束的正则化框架使去噪过程渐近收敛到对流扩散动力学。

### 主要发现

在两个城市部署59个自设计便携式传感设备运行14天收集数据，相比第二表现最好的算法，STeP-Diff在MAE上提升89.12%，RMSE上提升82.30%，MAPE上提升25.00%，有效捕捉了空气污染场的时空依赖性。

### 结论

STeP-Diff模型能够有效预测空气污染，通过结合物理规律和实际测量数据，显著提高了预测准确性，为城市空气质量管理提供了可靠工具。

### 翻译

细粒度空气污染预测对城市管理和健康建筑的发展至关重要。在汽车和公交车等移动平台上部署便携式传感器，提供了一种低成本、易于维护且覆盖面广的数据收集解决方案。然而，由于这些非专用移动平台的随机和不可控移动模式， resulting传感器数据往往不完整且时间不一致。通过探索扩散模型反向过程中的潜在训练模式，我们提出了时空物理信息扩散模型（STeP-Diff）。STeP-Diff利用DeepONet建模测量的空间序列，并结合PDE信息扩散模型来预测从和不完整且时间变化的数据中的时空场。通过PDE约束的正则化框架，去噪过程渐近收敛到对流扩散动力学，确保预测既基于实际测量又符合污染扩散的基本物理规律。为评估系统性能，我们在两个城市部署了59个自设计的便携式传感设备，运行14天收集空气污染数据。与第二表现最好的算法相比，我们的模型在MAE上实现了高达89.12%的改进，在RMSE上实现了82.30%的改进，在MAPE上实现了25.00%的改进，大量评估表明STeP-Diff有效捕捉了空气污染场中的时空依赖性。


### 论文摘要

Fine-grained air pollution forecasting is crucial for urban management and the development of healthy buildings. Deploying portable sensors on mobile platforms such as cars and buses offers a low-cost, easy-to-maintain, and wide-coverage data collection solution. However, due to the random and uncontrollable movement patterns of these non-dedicated mobile platforms, the resulting sensor data are often incomplete and temporally inconsistent. By exploring potential training patterns in the reverse process of diffusion models, we propose Spatio-Temporal Physics-Informed Diffusion Models (STeP-Diff). STeP-Diff leverages DeepONet to model the spatial sequence of measurements along with a PDE-informed diffusion model to forecast the spatio-temporal field from incomplete and time-varying data. Through a PDE-constrained regularization framework, the denoising process asymptotically converges to the convection-diffusion dynamics, ensuring that predictions are both grounded in real-world measurements and aligned with the fundamental physics governing pollution dispersion. To assess the performance of the system, we deployed 59 self-designed portable sensing devices in two cities, operating for 14 days to collect air pollution data. Compared to the second-best performing algorithm, our model achieved improvements of up to 89.12% in MAE, 82.30% in RMSE, and 25.00% in MAPE, with extensive evaluations demonstrating that STeP-Diff effectively captures the spatio-temporal dependencies in air pollution fields.

---

## 7. CNN on `Top': In Search of Scalable & Lightweight Image-based Jet Taggers

**论文链接:** [http://arxiv.org/abs/2512.05031v1](http://arxiv.org/abs/2512.05031v1)

**作者:** Rajneil Baruah, Subhadeep Mondal, Sunando Kumar Patra, Satyajit Roy

**发布时间:** 2025-12-04

**备注:** 12 pages, 3 figures, 2 tables

### GPT解析

### 总结

研究提出了一种基于EfficientNet架构的轻量级、可扩展方法，结合喷流全局特征，用于喷流分类，在保持竞争力的同时显著降低了计算成本。

### 背景

基于Transformer的标准图神经网络(GNNs)在分类不同类型喷流方面表现最佳，但需要大量计算资源。

### 目的

探索使用EfficientNet架构的轻量级和可扩展版本，结合喷流的全球特征，开发一种计算效率高但性能有竞争力的喷流分类方法。

### 方法

采用EfficientNet架构的轻量级版本，结合喷流的全局特征进行喷流分类。

### 主要发现

所提出的网络计算成本低，但具有与现有方法相竞争的性能表现。

### 结论

展示了该网络在标记顶夸克喷流（从其他轻夸克和胶子喷流中识别）方面的有效性。

### 翻译

虽然基于Transformer和标准的图神经网络(GNNs)已被证明是分类不同类型喷流的最佳方法，但它们需要大量的计算能力。我们探索了使用EfficientNet架构的轻量级和可扩展版本，结合喷流的全局特征的可能性。最终产品计算成本低廉，但能够实现有竞争力的性能。我们展示了我们的网络在标记顶夸克喷流（在其他轻夸克和胶子喷流的海洋中）方面的有效性。


### 论文摘要

While Transformer-based and standard Graph Neural Networks (GNNs) have proven to be the best performers in classifying different types of jets, they require substantial computational power. We explore the scope of using a lightweight and scalable version of the EfficientNet architecture, along with global features of the jet. The end product is computationally inexpensive but is capable of competitive performance. We showcase the efficacy of our network for tagging top-quark jets in a sea of other light-quark and gluon jets.

---

## 8. PVLS: A Learning-based Parameter Prediction Technique for Variational Quantum Linear Solvers

**论文链接:** [http://arxiv.org/abs/2512.04909v1](http://arxiv.org/abs/2512.04909v1)

**作者:** Youla Yang

**发布时间:** 2025-12-04

### GPT解析

### 总结

论文提出了一种名为PVLS的新方法，通过使用图神经网络预测高质量的初始参数，显著改善了变分量子线性求解器的性能，解决了贫瘠高原和低效参数初始化的问题，在保持精度的同时大幅提升了优化速度。

### 背景

变分量子线性求解器(VQLS)是在近期量子设备上求解线性系统的有前景的方法，但其性能常受贫瘠高原和低效参数初始化的限制，随着系统规模增大，这些限制会严重影响可训练性。

### 目的

引入PVLS，一个基于学习的参数预测框架，旨在为VQLS电路生成高质量初始参数，改善收敛性并减少优化难度。

### 方法

PVLS使用图神经网络(GNNs)来为VQLS电路生成高质量初始参数，通过利用系数矩阵的结构信息，预测具有表达能力和可扩展性的初始化参数。

### 主要发现

在矩阵大小从16到1024的广泛实验中，PVLS提供了高达2.6倍的优化加速，需要更少的迭代次数，同时保持相当的解决方案准确性。

### 结论

机器学习引导的初始化策略在提高NISQ时代混合量子-经典算法的实际应用潜力方面具有巨大潜力。

### 翻译

变分量子线性求解器(VQLS)是在近期量子设备上求解线性系统的有前景方法。然而，它们的性能常受贫瘠高原和低效参数初始化的限制，随着系统规模增大，这些限制会严重影响可训练性。在这项工作中，我们引入了PVLS，一个基于学习的参数预测框架，它使用图神经网络(GNNs)为VQLS电路生成高质量初始参数。通过利用系数矩阵的结构信息，PVLS预测具有表达能力和可扩展性的初始化参数，从而改善收敛性并减少优化难度。在矩阵大小从16到1024的广泛实验中，PVLS提供了高达2.6倍的优化加速，需要更少的迭代次数，同时保持相当的解决方案准确性。这些结果表明，机器学习引导的初始化策略在提高NISQ时代混合量子-经典算法的实际应用潜力方面具有巨大潜力。


### 论文摘要

Variational Quantum Linear Solvers (VQLS) are a promising method for solving linear systems on near-term quantum devices. However, their performance is often limited by barren plateaus and inefficient parameter initialization, which significantly hinder trainability as the system size increases. In this work, we introduce PVLS, a learning-based parameter prediction framework that uses Graph Neural Networks (GNNs) to generate high-quality initial parameters for VQLS circuits. By leveraging structural information from the coefficient matrix, PVLS predicts expressive and scalable initializations that improve convergence and reduce optimization difficulty. Extensive experiments on matrix sizes ranging from 16 to 1024 show that PVLS provides up to a 2.6x speedup in optimization and requires fewer iterations while maintaining comparable solution accuracy. These results demonstrate the potential of machine-learning-guided initialization strategies for improving the practicality of hybrid quantum-classical algorithms in the NISQ era.

---

## 9. QoSDiff: An Implicit Topological Embedding Learning Framework Leveraging Denoising Diffusion and Adversarial Attention for Robust QoS Prediction

**论文链接:** [http://arxiv.org/abs/2512.04596v1](http://arxiv.org/abs/2512.04596v1)

**作者:** Guanchen Du, Jianlong Xu, Wei Wei

**发布时间:** 2025-12-04

**备注:** Preprint submitted to IEEE Transactions on Services Computing

### GPT解析

### 总结

QoSDiff是一种新颖的嵌入学习框架，用于服务质量预测，绕过了显式图构建的需求，通过去噪扩散概率模型和对抗交互模块有效处理噪声和稀疏数据，显著提升了预测性能。

### 背景

准确的QoS预测对服务计算至关重要，但现有方法（特别是图神经网络GNNs）严重依赖构建显式的用户-服务交互图，这种依赖性带来了严重的可扩展性瓶颈，并在显式连接稀疏或被噪声污染时限制了性能。

### 目的

解决现有方法对显式图构建的依赖问题，提高在稀疏或噪声数据下的性能，并提升模型的泛化能力和鲁棒性。

### 方法

提出QoSDiff框架，使用去噪扩散概率模型从噪声初始化中恢复潜在结构，设计对抗交互模块整合双向混合注意力机制，采用对抗范式区分信息模式与噪声，实现用户-服务关联的双视角建模。

### 主要发现

在两个大规模真实数据集上的实验表明，QoSDiff显著优于最先进的基线方法，具有出色的跨数据集泛化能力和对数据稀疏性与观测噪声的卓越鲁棒性。

### 结论

QoSDiff成功解决了传统方法对显式图构建的依赖问题，通过去噪扩散和对抗交互机制有效处理噪声和稀疏数据，为服务质量预测提供了更有效、更鲁棒的解决方案。

### 翻译

准确的服务质量(QoS)预测是服务计算的基础，为服务选择提供必要的数据驱动指导，确保卓越的用户体验。然而，主流方法，特别是图神经网络(GNNs)，严重依赖构建显式的用户-服务交互图。这种依赖引入了严重的可扩展性瓶颈，并在显式连接稀疏或被噪声污染时限制了性能。为解决这些挑战，本文引入了QoSDiff，一种新颖的嵌入学习框架，绕过了显式图构建的先决条件。具体而言，它利用去噪扩散概率模型从噪声初始化中恢复内在的潜在结构。为进一步捕获高阶交互，我们提出了一个对抗交互模块，整合了双向混合注意力机制。这种对抗范式能够动态区分信息模式与噪声，实现了复杂用户-服务关联的双视角建模。在两个大规模真实数据集上的广泛实验表明，QoSDiff显著优于最先进的基线方法。值得注意的是，结果突显了该框架卓越的跨数据集泛化能力和对数据稀疏性和观测噪声的出色鲁棒性。


### 论文摘要

Accurate Quality of Service (QoS) prediction is fundamental to service computing, providing essential data-driven guidance for service selection and ensuring superior user experiences. However, prevalent approaches, particularly Graph Neural Networks (GNNs), heavily rely on constructing explicit user--service interaction graphs. This dependency introduces severe scalability bottlenecks and limits performance when explicit connections are sparse or corrupted by noise. To address these challenges, this paper introduces \emph{QoSDiff}, a novel embedding learning framework that bypasses the prerequisite of explicit graph construction. Specifically, it leverages a denoising diffusion probabilistic model to recover intrinsic latent structures from noisy initializations. To further capture high-order interactions, we propose an adversarial interaction module that integrates a bidirectional hybrid attention mechanism. This adversarial paradigm dynamically distinguishes informative patterns from noise, enabling a dual-perspective modeling of intricate user--service associations. Extensive experiments on two large-scale real-world datasets demonstrate that QoSDiff significantly outperforms state-of-the-art baselines. Notably, the results highlight the framework's superior cross-dataset generalization capability and exceptional robustness against data sparsity and observational noise.

---

## 10. Tensor Neyman-Pearson Classification: Theory, Algorithms, and Error Control

**论文链接:** [http://arxiv.org/abs/2512.04583v1](http://arxiv.org/abs/2512.04583v1)

**作者:** Lingchong Liu, Elynn Chen, Yuefeng Han, Lucy Xia

**发布时间:** 2025-12-04

**备注:** 59 pages, 5 figures and 12 tables (including Supplementary Material)

### GPT解析

### 总结

本研究开发了首个张量Neyman-Pearson分类框架，能够在有限样本控制第一类错误的同时利用张量数据的多模态结构，为生物化学中的不对称风险评估提供了可靠工具。

### 背景

生物化学发现越来越多地依赖于在错误后果高度不对称的情况下对分子结构进行分类。在致突变性和致癌性研究中，将有害化合物误分类为良性会带来重大科学、监管和健康风险，而误报主要增加实验室工作量。

### 目的

开发一种能够控制第一类错误同时利用张量数据多模态结构的分类框架，解决现有张量分类器在第一类错误控制方面的不足。

### 方法

开发了第一个张量Neyman-Pearson（Tensor-NP）分类框架，在张量正态混合模型下推导了oracle NP判别式，并描述了其Tucker低秩流形几何结构。提出了判别张量迭代投影估计器和结合深度学习与Tensor-NP伞形校准的张量-NP神经网络分类器。

### 主要发现

在四个生物化学数据集上，Tensor-NP分类器能够将第一类错误维持在预定水平，同时提供有竞争力的第二类错误性能，证明了该方法在实际应用中的有效性。

### 结论

Tensor-NP为具有复杂分子张量的不对称风险决策提供了可靠工具，能够在保持第一类错误可控的同时实现良好的分类性能。

### 翻译

生物化学发现越来越多地依赖于在错误后果高度不对称的情况下对分子结构进行分类。在致突变性和致癌性研究中，将有害化合物误分类为良性会引发重大的科学、监管和健康风险，而误报主要增加实验室工作量。现代表示方法将分子图转换为保留多尺度几何和拓扑结构的持久图像张量，然而现有的张量分类器和深度张量神经网络无法提供关于第一类错误的有限样本保证，在实践中经常表现出严重的错误膨胀。我们开发了首个张量Neyman-Pearson（Tensor-NP）分类框架，能够在有限样本控制第一类错误的同时利用张量数据的多模态结构。在张量正态混合模型下，我们推导了oracle NP判别式，描述了其Tucker低秩流形几何结构，并建立了特定的张量边界和条件检测条件，从而能够以高概率获得关于第二类误差过剩的界限。我们进一步提出了判别张量迭代投影估计器和结合深度学习与Tensor-NP伞形校准的张量-NP神经网络分类器，为多向数据提供了首个分布自由的NP有效方法。在四个生物化学数据集上，Tensor-NP分类器将第一类错误维持在预定水平，同时提供有竞争力的第二类错误性能，为具有复杂分子张量的不对称风险决策提供了可靠工具。


### 论文摘要

Biochemical discovery increasingly relies on classifying molecular structures when the consequences of different errors are highly asymmetric. In mutagenicity and carcinogenicity, misclassifying a harmful compound as benign can trigger substantial scientific, regulatory, and health risks, whereas false alarms primarily increase laboratory workload. Modern representations transform molecular graphs into persistence image tensors that preserve multiscale geometric and topological structure, yet existing tensor classifiers and deep tensor neural networks provide no finite-sample guarantees on type I error and often exhibit severe error inflation in practice.   We develop the first Tensor Neyman-Pearson (Tensor-NP) classification framework that achieves finite-sample control of type I error while exploiting the multi-mode structure of tensor data. Under a tensor-normal mixture model, we derive the oracle NP discriminant, characterize its Tucker low-rank manifold geometry, and establish tensor-specific margin and conditional detection conditions enabling high-probability bounds on excess type II error. We further propose a Discriminant Tensor Iterative Projection estimator and a Tensor-NP Neural Classifier combining deep learning with Tensor-NP umbrella calibration, yielding the first distribution-free NP-valid methods for multiway data. Across four biochemical datasets, Tensor-NP classifiers maintain type I errors at prespecified levels while delivering competitive type II error performance, providing reliable tools for asymmetric-risk decisions with complex molecular tensors.

---

## 11. Explainable Graph Representation Learning via Graph Pattern Analysis

**论文链接:** [http://arxiv.org/abs/2512.04530v1](http://arxiv.org/abs/2512.04530v1)

**作者:** Xudong Wang, Ziheng Sun, Chris Ding, Jicong Fan

**发布时间:** 2025-12-04

**DOI:** 10.24963/ijcai.2025/381

**备注:** Full version with appendix of the paper published in the Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25), Main Track

### GPT解析

### 总结

该论文提出了一种基于图模式分析的可解释图表示学习框架PXGL-GNN，通过采样和学习不同模式的图子结构，并使用加权求和组合这些模式的表示，从而实现图表示的可解释性。

### 背景

可解释人工智能(XAI)是AI社区的重要领域，可解释性对于构建稳健和可信的AI模型至关重要。先前的研究已探索了模型级和实例级的可解释图学习，但对可解释图表示学习的研究有限。

### 目的

专注于表示级可解释图学习，探讨图表示中捕获了关于图的哪些特定信息这一基本问题。

### 方法

受图核启发，引入PXGL-GNN框架，首先采样各种模式的图子结构，然后学习这些模式的表示，使用加权求和组合它们，权重表示每个图模式贡献的重要性。同时提供方法的理论分析，包括稳健性和泛化能力。

### 主要发现

在实验中展示了如何使用模式分析学习和解释现实数据的图表示，与多个基线方法在监督和无监督学习任务中比较后，证明了所提方法的有效性。

### 结论

图模式分析框架(PXGL-GNN)能够有效地学习和解释图表示，解决了图表示可解释性的挑战。

### 翻译

可解释人工智能(XAI)是AI社区的重要领域，可解释性对于构建稳健和可信的AI模型至关重要。虽然先前的研究已经探索了模型级和实例级的可解释图学习，但对可解释图表示学习的研究有限。在本文中，我们专注于表示级可解释图学习，并提出了一个基本问题：图表示中捕获了关于图的哪些特定信息？我们的方法受图核启发，图核通过计算特定图模式中的子结构来评估图相似度。尽管模式计数向量可以作为可解释表示，但它存在忽略节点特征和高维度的局限性。为解决这些局限性，我们引入了一个框架(PXGL-GNN)，通过图模式分析来学习和解释图表示。我们首先采样各种模式的图子结构，然后学习这些模式的表示，并使用加权求和组合它们，其中权重表示每个图模式贡献的重要性。我们还提供了方法的理论分析，包括稳健性和泛化能力。在我们的实验中，我们展示了如何使用模式分析学习和解释现实数据的图表示。此外，我们在监督和无监督学习任务中将我们的方法与多个基线方法进行比较，以证明其有效性。


### 论文摘要

Explainable artificial intelligence (XAI) is an important area in the AI community, and interpretability is crucial for building robust and trustworthy AI models. While previous work has explored model-level and instance-level explainable graph learning, there has been limited investigation into explainable graph representation learning. In this paper, we focus on representation-level explainable graph learning and ask a fundamental question: What specific information about a graph is captured in graph representations? Our approach is inspired by graph kernels, which evaluate graph similarities by counting substructures within specific graph patterns. Although the pattern counting vector can serve as an explainable representation, it has limitations such as ignoring node features and being high-dimensional. To address these limitations, we introduce a framework (PXGL-GNN) for learning and explaining graph representations through graph pattern analysis. We start by sampling graph substructures of various patterns. Then, we learn the representations of these patterns and combine them using a weighted sum, where the weights indicate the importance of each graph pattern's contribution. We also provide theoretical analyses of our methods, including robustness and generalization. In our experiments, we show how to learn and explain graph representations for real-world data using pattern analysis. Additionally, we compare our method against multiple baselines in both supervised and unsupervised learning tasks to demonstrate its effectiveness.

---

## 12. GraphBench: Next-generation graph learning benchmarking

**论文链接:** [http://arxiv.org/abs/2512.04475v1](http://arxiv.org/abs/2512.04475v1)

**作者:** Timo Stoll, Chendi Qian, Ben Finkelshtein, Ali Parviz, Darius Weber, Fabrizio Frasca, Hadar Shavit, Antoine Siraudin, Arman Mielke, Marie Anastacio, Erik Müller, Maya Bechler-Speicher, Michael Bronstein, Mikhail Galkin, Holger Hoos, Mathias Niepert, Bryan Perozzi, Jan Tönshoff, Christopher Morris

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文介绍了GraphBench，一个全面的图机器学习基准测试套件，旨在解决当前基准测试实践碎片化的问题，提供标准化的评估框架和参考性能基线。

### 背景

图机器学习在分子属性预测和芯片设计等领域取得了显著进展，但基准测试实践仍然分散，通常依赖于狭窄的、特定任务的数据集和不一致的评估协议，这阻碍了可复现性和更广泛的进步。

### 目的

引入GraphBench基准测试套件，解决当前图机器学习基准测试碎片化问题，提供更全面、标准化的评估框架。

### 方法

GraphBench跨越多个领域和预测任务，包括节点级、边级、图级和生成设置；提供标准化的评估协议，包括一致的数据集分割和考虑分布外泛化的性能指标；提供统一的超参数调整框架；使用消息传递神经网络和图Transformer模型进行基准测试。

### 主要发现

通过GraphBench建立了图机器学习模型的基准性能和参考标准，为未来研究提供了比较基础。

### 结论

GraphBench为图机器学习领域提供了一个全面、标准化的基准测试框架，有助于促进研究的可复现性和更广泛的进步。

### 翻译

图机器学习在分子属性预测和芯片设计等各个领域最近取得了令人印象深刻的进展。然而，基准测试实践仍然分散，通常依赖于狭窄的、特定任务的数据集和不一致的评估协议，这阻碍了可复现性和更广泛的进步。为了解决这个问题，我们引入了GraphBench，一个全面的基准测试套件，跨越多个领域和预测任务，包括节点级、边级、图级和生成设置。GraphBench提供了标准化的评估协议——包括一致的数据集分割和考虑分布外泛化的性能指标——以及统一的超参数调整框架。此外，我们使用消息传递神经网络和图Transformer模型对GraphBench进行了基准测试，提供了有原则的基线并建立了参考性能。更多详情请访问www.graphbench.io。


### 论文摘要

Machine learning on graphs has recently achieved impressive progress in various domains, including molecular property prediction and chip design. However, benchmarking practices remain fragmented, often relying on narrow, task-specific datasets and inconsistent evaluation protocols, which hampers reproducibility and broader progress. To address this, we introduce GraphBench, a comprehensive benchmarking suite that spans diverse domains and prediction tasks, including node-level, edge-level, graph-level, and generative settings. GraphBench provides standardized evaluation protocols -- with consistent dataset splits and performance metrics that account for out-of-distribution generalization -- as well as a unified hyperparameter tuning framework. Additionally, we benchmark GraphBench using message-passing neural networks and graph transformer models, providing principled baselines and establishing a reference performance. See www.graphbench.io for further details.

---

## 13. Learning Beamforming for Pinching Antenna System-Enabled ISAC in Low-Altitude Wireless Networks

**论文链接:** [http://arxiv.org/abs/2512.04293v1](http://arxiv.org/abs/2512.04293v1)

**作者:** Jia Guo, Yuanwei Liu, Arumugam Nallanathan

**发布时间:** 2025-12-03

**备注:** 13 pages, 6 figures

### GPT解析

### 总结

本研究针对低空无线网络中的集成感知与通信(ISAC)系统，提出了一种基于挤压天线(PA)和图神经网络(GNN)的联合学习方法，用于优化天线位置和发射波束形成。

### 背景

在低空无线网络中，传统的ISAC系统面临路径损耗问题，限制了感知和通信无人驾驶飞行器(UAV)在大范围内的性能表现。

### 目的

研究如何通过自由部署天线位置和优化波束形成，同时最大化多个目标的感知性能并满足多个用户的通信性能要求。

### 方法

采用分段波导支持的挤压天线(SWAN)系统减轻波导内衰减；开发基于SWAN的ISAC替代优化算法(SWISAC-AO)；提出SWISAC-GNN图神经网络来联合学习PA位置和发射波束形成，其更新过程受SWISAC-AO算法启发。

### 主要发现

GNN方法实现了与优化算法相当或更好的感知性能，同时更好地满足通信要求；SWISAC-GNN具有更低的实现复杂度，能够实现实时部署。

### 结论

SWISAC-GNN方法在保持良好感知性能的同时满足通信要求，且计算复杂度低，适合实际应用和实时部署。

### 翻译

本研究探讨了低空无线网络中挤压天线辅助的集成感知与通信(ISAC)系统的天线位置和发射波束形成的联合学习。通过沿波导自由部署天线位置，挤压天线系统有效减轻了路径损耗的影响，从而提高了在大范围内飞行的感知和通信无人驾驶飞行器(UAV)的能力。我们首先建立了最大化多个目标感知性能同时满足多个用户通信性能要求的问题模型，其中目标和用户均为UAV。为减轻波导内衰减并提高感知性能，采用了分段波导支持的挤压天线(SWAN)系统。进一步，开发了基于SWAN的ISAC的替代优化算法(SWISAC-AO)，推导了发射波束形成解决方案的最优结构。随后，提出了称为SWISAC-GNN的图神经网络，用于联合学习PA位置和发射波束形成，其替代更新过程受SWISAC-AO算法启发。数值结果表明，GNN实现了与AO算法相当或更好的感知性能，同时更好地满足通信要求。此外，SWISAC-GNN具有更低的实现复杂度，能够实现实时部署。


### 论文摘要

This work investigates the joint learning of pinching antenna (PA) positions and transmit beamforming for PA-aided integrated sensing and communication (ISAC) in the low-altitude wireless networks. By freely deploying antenna positions along waveguides, the pinching antenna system effectively mitigates the impact of path loss and thus enhances the capacities of sensing and communicating unmanned aerial vehicles (UAVs) that fly over a large range. We first model the problem of maximizing the sensing performance of multiple targets while satisfying the communication performance requirements of multiple users, where both the targets and users are UAVs. For mitigating in-waveguide attenuation and improving sensing performance, the segmented waveguide-enabled pinching antenna (SWAN) system is adopted. Furthermore, an alternative optimization (AO) algorithm for SWAN-based ISAC (SWISAC-AO) is developed, where the optimal structure of the transmit beamforming solution is derived. A graph neural network (GNN), termed SWISAC-GNN, is then proposed to jointly learn PA positions and transmit beamforming, with its alternative update procedure inspired by the SWISAC-AO algorithm. Numerical results show that the GNN achieves sensing performance comparable to or better than the AO algorithm while better satisfying communication requirements. Moreover, the SWISAC-GNN is with much lower implementation complexity, enabling real-time deployment.

---

## 14. From Generated Human Videos to Physically Plausible Robot Trajectories

**论文链接:** [http://arxiv.org/abs/2512.05094v1](http://arxiv.org/abs/2512.05094v1)

**作者:** James Ni, Zekai Wang, Wei Lin, Amir Bar, Yann LeCun, Trevor Darrell, Jitendra Malik, Roei Herzig

**发布时间:** 2025-12-04

**备注:** For project website, see https://genmimic.github.io

### GPT解析

### 总结

该研究提出了一种两阶段流程，使机器人能够零样本模仿生成视频中的人类动作，并创建了新的基准数据集，证明了该方法在模拟和实际机器人上的有效性。

### 背景

视频生成模型正在快速改进，能够在新情境中合成人类动作，这使它们有可能成为情境机器人控制的高级规划者。

### 目的

解决人形机器人如何能以零样本方式执行生成视频中的人类动作这一关键研究问题，克服生成视频噪声和形态扭曲导致的直接模仿困难。

### 方法

引入两阶段流程：首先将视频像素提升到4D人类表示并重新定位到人形形态；其次提出GenMimic——基于3D关键点的物理感知强化学习策略，使用对称正则化和关键点加权的跟踪奖励进行训练。

### 主要发现

GenMimic能模仿来自嘈杂生成视频的人类动作；创建了GenMimicBench合成人类运动数据集；实验显示与强大基线相比有改进；在Unitree G1机器人上实现了连贯、物理稳定的运动跟踪，无需微调。

### 结论

该工作为实现视频生成模型作为机器人控制高级策略的潜力提供了一条有希望的路径。

### 翻译

视频生成模型正在快速改进其在新情境中合成人类动作的能力，有潜力作为情境机器人控制的高级规划者。为实现这一潜力，一个关键研究问题仍然悬而未决：人形机器人如何能以零样本方式执行生成视频中的人类动作？这一挑战的出现是因为生成的视频通常存在噪声并表现出形态扭曲，使得直接模仿比真实视频更困难。为解决这一问题，我们引入了一个两阶段流程。首先，我们将视频像素提升到4D人类表示，然后重新定位到人形形态。其次，我们提出了GenMimic——一个基于3D关键点的物理感知强化学习策略，使用对称正则化和关键点加权的跟踪奖励进行训练。因此，GenMimic可以模仿来自嘈杂生成视频的人类动作。我们整理了GenMimicBench，这是一个使用两种视频生成模型创建的跨多种动作和情境的合成人类运动数据集，建立了用于评估零样本泛化和策略鲁棒性的基准。大量实验表明与强大的基线相比有所改进，并在无需微调的情况下确认了Unitree G1人形机器人上连贯、物理稳定的运动跟踪。这项工作为实现视频生成模型作为机器人控制高级策略的潜力提供了一条有希望的路径。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决的是如何让人形机器人能够零样本地模仿视频中生成的人类动作。这个问题很重要，因为视频生成模型可以创建各种新颖场景中的人类动作，但目前这些生成视频往往包含噪声和形态学不准确，难以直接让机器人模仿。解决这个问题将使机器人能够执行更广泛的任务，无需针对每个任务进行专门训练，大大提高机器人的适应性和通用性，为机器人利用生成式AI进行高层规划开辟了新途径。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者采用两阶段方法来解决这个问题。首先，他们使用现有的4D人体重建模型（如TRAM）从生成视频中提取人体运动轨迹，然后通过重定向技术（如PHC）将其适配到机器人形态。其次，他们设计了GenMimic策略，这是一个基于3D关键点的强化学习模型，通过加权跟踪奖励和对称性正则化来提高对噪声的鲁棒性。作者借鉴了现有工作，包括使用标准的强化学习框架（PPO）、DAgger算法进行知识蒸馏，以及受对称性学习启发设计的对称损失函数，但将这些技术组合成一种新颖的方式来解决生成视频模仿的特定挑战。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是：由于生成视频中的人类动作往往包含噪声和形态学不准确，直接模仿困难，因此通过关注关键点（特别是末端执行器）并利用人体对称性，可以提高对噪声的鲁棒性。具体来说，使用加权跟踪奖励使机器人优先关注最重要的关键点（如手和头），而对称性损失则允许机器人从一侧动作推断另一侧动作，从而纠正噪声。整体流程是：首先从生成视频中提取4D人体表示并重定向到机器人形态得到3D关键点；然后GenMimic策略接收这些关键点和机器人本体感受信息，输出期望关节角度；最后PD控制器将关节角度转换为力矩驱动机器人执行动作。训练过程中使用加权跟踪奖励和对称性损失，并通过域随机化提高鲁棒性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出第一个通用框架使人形机器人能执行视频生成模型生成的动作；2）GenMimic策略使用加权关键点跟踪奖励和对称性正则化提高对噪声的鲁棒性；3）创建GenMimicBench数据集，包含428个使用Wan2.1和Cosmos-Predict2生成的多样化人体动作视频。相比之前的工作，GenMimic不依赖于真实、高质量的动作捕捉数据，而是能在仅使用AMASS数据训练的情况下零样本泛化到生成视频。实验表明，它在生成视频上的跟踪成功率显著高于GMT和TWIST等基线方法，且相比BeyondMimic，在处理噪声和形态变化方面更加鲁棒。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'GenMimic通过结合4D人体重建、形态重定向和具有对称性感知的强化学习策略，首次实现了人形机器人零样本模仿生成视频中的人类动作，为视频生成模型作为机器人高层规划提供了可行路径。'}


### 论文摘要

Video generation models are rapidly improving in their ability to synthesize human actions in novel contexts, holding the potential to serve as high-level planners for contextual robot control. To realize this potential, a key research question remains open: how can a humanoid execute the human actions from generated videos in a zero-shot manner? This challenge arises because generated videos are often noisy and exhibit morphological distortions that make direct imitation difficult compared to real video. To address this, we introduce a two-stage pipeline. First, we lift video pixels into a 4D human representation and then retarget to the humanoid morphology. Second, we propose GenMimic-a physics-aware reinforcement learning policy conditioned on 3D keypoints, and trained with symmetry regularization and keypoint-weighted tracking rewards. As a result, GenMimic can mimic human actions from noisy, generated videos. We curate GenMimicBench, a synthetic human-motion dataset generated using two video generation models across a spectrum of actions and contexts, establishing a benchmark for assessing zero-shot generalization and policy robustness. Extensive experiments demonstrate improvements over strong baselines in simulation and confirm coherent, physically stable motion tracking on a Unitree G1 humanoid robot without fine-tuning. This work offers a promising path to realizing the potential of video generation models as high-level policies for robot control.

---

## 15. The Geometry of Intelligence: Deterministic Functional Topology as a Foundation for Real-World Perception

**论文链接:** [http://arxiv.org/abs/2512.05089v1](http://arxiv.org/abs/2512.05089v1)

**作者:** Eduardo Di Santi

**发布时间:** 2025-12-04

**备注:** 35 pages, 6 figures. This preprint develops a deterministic functional-topological framework showing that physical systems generate compact perceptual manifolds with finite radius. We provide theory, Monte-Carlo estimators, and validation across PM, battery, and ECG domains, unifying biological perception and self-supervised AI

### GPT解析

### 总结

该研究开发了一种确定性功能拓扑框架，证明物理现象的有效实现形成一个具有稳定不变量和有限Hausdorff半径的紧凑感知流形，可以通过自监督方式发现边界，为感知和世界模型构建提供了统一数学基础。

### 背景

现实世界物理过程不会产生任意变异性，它们的信号集中在功能空间的紧凑和低变异性子集上。这种几何结构使得生物和人工系统能够从少量例子中快速泛化。

### 目的

开发一个确定性功能拓扑框架，其中物理现象的有效实现形成一个具有稳定不变量和有限Hausdorff半径的紧凑感知流形。

### 方法

展示即使系统的控制方程未知，也可以通过蒙特卡洛采样以完全自监督的方式发现这种流形的边界。提供理论保证、知识边界的实际估计器，以及在机电道岔、电化学电池放电曲线和生理ECG信号三个领域的实证验证。

### 主要发现

确定性功能拓扑为感知、表征和世界模型构建提供了统一的数学基础，解释了为什么生物学习者和自监督AI模型可以从有限的观察中泛化。

### 结论

确定性功能拓扑框架能够有效捕捉物理现象的几何结构，并解释了从有限数据中泛化的能力，为理解生物和人工系统的学习机制提供了理论基础。

### 翻译

现实世界的物理过程不会产生任意变异性：它们的信号集中在功能空间的紧凑和低变异性子集上。这种几何结构使得生物和人工系统能够从少量例子中快速泛化。本研究开发了一个确定性功能拓扑框架，其中物理现象的有效实现形成一个具有稳定不变量和有限Hausdorff半径的紧凑感知流形。我们表明，即使系统的控制方程未知，也可以通过蒙特卡洛采样以完全自监督的方式发现这种流形的边界。我们在三个领域提供了理论保证、知识边界的实际估计器和实证验证：机电道岔、电化学电池放电曲线和生理ECG信号。我们的研究结果表明，确定性功能拓扑为感知、表征和世界模型构建提供了统一的数学基础，解释了为什么生物学习者和自监督AI模型可以从有限的观察中泛化。


### 论文摘要

Real-world physical processes do not generate arbitrary variability: their signals concentrate on compact and low-variability subsets of functional space. This geometric structure enables rapid generalization from a few examples in both biological and artificial systems.   This work develops a deterministic functional-topological framework in which the set of valid realizations of a physical phenomenon forms a compact perceptual manifold with stable invariants and a finite Hausdorff radius. We show that the boundaries of this manifold can be discovered in a fully self-supervised manner through Monte Carlo sampling, even when the governing equations of the system are unknown.   We provide theoretical guarantees, practical estimators of knowledge boundaries, and empirical validations across three domains: electromechanical railway point machines, electrochemical battery discharge curves, and physiological ECG signals.   Our results demonstrate that deterministic functional topology offers a unified mathematical foundation for perception, representation, and world-model construction, explaining why biological learners and self-supervised AI models can generalize from limited observations.

---

## 16. QKAN-LSTM: Quantum-inspired Kolmogorov-Arnold Long Short-term Memory

**论文链接:** [http://arxiv.org/abs/2512.05049v1](http://arxiv.org/abs/2512.05049v1)

**作者:** Yu-Chao Hsu, Jiun-Cheng Jiang, Chun-Hua Lin, Kuo-Chung Peng, Nan-Yow Chen, Samuel Yen-Chi Chen, En-Jui Kuo, Hsi-Sheng Goan

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种名为QKAN-LSTM的新型神经网络模型，通过整合数据重新上传激活模块到LSTM门控结构中，实现了在经典硬件上执行的量子级表达能力，并在预测任务中表现出优越的性能。

### 背景

长短期记忆（LSTM）模型是循环神经网络的一种特殊类型，在城市电信预测等领域的序列建模任务中至关重要，这些领域的时间相关性和非线性依赖性占主导地位。然而，传统LSTM模型存在参数冗余度高和非线性表达能力有限的问题。

### 目的

解决传统LSTM模型的高参数冗余和有限非线性表达能力问题，开发一种能够在经典硬件上执行但具有量子级表达能力的序列模型。

### 方法

提出QKAN-LSTM模型，将DARUAN模块整合到LSTM的门控结构中，每个DARUAN作为量子变分激活函数增强频率适应性；将框架扩展到JHCG Net，将KAN推广到编码器-解码器结构，使用QKAN实现潜在KAN，创建用于分层表示学习的混合QKAN（HQKAN）。

### 主要发现

在阻尼简谐运动、贝塞尔函数和城市电信三个数据集上的经验评估表明，与经典LSTM相比，QKAN-LSTM实现了优越的预测准确性和泛化能力，同时可训练参数减少了79%。

### 结论

提出的HQKAN-LSTM为真实数据环境中的量子启发序列建模提供了可扩展且可解释的途径。

### 翻译

长短期记忆（LSTM）模型是循环神经网络（RNN）的一种特殊类型，在城市电信预测等领域序列建模任务中起着核心作用，这些领域的时间相关性和非线性依赖性占主导地位。然而，传统LSTM模型存在高参数冗余和有限非线性表达能力的问题。在这项工作中，我们提出了量子启发Kolmogorov-Arnold长短期记忆（QKAN-LSTM），将数据重新上传激活（DARUAN）模块整合到LSTM的门控结构中。每个DARUAN作为量子变分激活函数（QVAF），增强频率适应性，实现无需多量子比特纠缠的指数级丰富的频谱表示。所得架构在保持量子级表达能力的同时，完全可以在经典硬件上执行。在阻尼简谐运动、贝塞尔函数和城市电信三个数据集上的经验评估表明，与经典LSTM相比，QKAN-LSTM实现了优越的预测准确性和泛化能力，同时可训练参数减少了79%。我们将框架扩展到江-黄-陈-Goan网络（JHCG Net），该网络将KAN推广到编码器-解码器结构，然后进一步使用QKAN实现潜在KAN，从而创建用于分层表示学习的混合QKAN（HQKAN）。因此，提出的HQKAN-LSTM为真实数据环境中的量子启发序列建模提供了可扩展且可解释的途径。


### 论文摘要

Long short-term memory (LSTM) models are a particular type of recurrent neural networks (RNNs) that are central to sequential modeling tasks in domains such as urban telecommunication forecasting, where temporal correlations and nonlinear dependencies dominate. However, conventional LSTMs suffer from high parameter redundancy and limited nonlinear expressivity. In this work, we propose the Quantum-inspired Kolmogorov-Arnold Long Short-Term Memory (QKAN-LSTM), which integrates Data Re-Uploading Activation (DARUAN) modules into the gating structure of LSTMs. Each DARUAN acts as a quantum variational activation function (QVAF), enhancing frequency adaptability and enabling an exponentially enriched spectral representation without multi-qubit entanglement. The resulting architecture preserves quantum-level expressivity while remaining fully executable on classical hardware. Empirical evaluations on three datasets, Damped Simple Harmonic Motion, Bessel Function, and Urban Telecommunication, demonstrate that QKAN-LSTM achieves superior predictive accuracy and generalization with a 79% reduction in trainable parameters compared to classical LSTMs. We extend the framework to the Jiang-Huang-Chen-Goan Network (JHCG Net), which generalizes KAN to encoder-decoder structures, and then further use QKAN to realize the latent KAN, thereby creating a Hybrid QKAN (HQKAN) for hierarchical representation learning. The proposed HQKAN-LSTM thus provides a scalable and interpretable pathway toward quantum-inspired sequential modeling in real-world data environments.

---

## 17. RAMEN: Resolution-Adjustable Multimodal Encoder for Earth Observation

**论文链接:** [http://arxiv.org/abs/2512.05025v1](http://arxiv.org/abs/2512.05025v1)

**作者:** Nicolas Houdré, Diego Marcos, Hugo Riffaud de Turckheim, Dino Ienco, Laurent Wendling, Camille Kurtz, Sylvain Lobry

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文介绍了RAMEN，一种分辨率可调整的多模态编码器，能够以完全传感器无关的方式学习跨地球观测数据的共享视觉表示，解决了现有基础模型在处理不同分辨率数据时的局限性。

### 背景

地球观测数据具有广泛的时空和光谱分辨率，从高分辨率光学图像到低分辨率多光谱产品或雷达时间序列。现有的基础模型虽然改进了多模态集成，但通常期望固定的输入分辨率或基于特定传感器的编码器，限制了在异构EO模态间的泛化能力。

### 目的

克服现有模型的局限性，引入RAMEN作为一种分辨率可调整的多模态编码器，以完全传感器无关的方式学习跨EO数据的共享视觉表示，并允许用户直接控制推理所需的细节级别。

### 方法

RAMEN将模态、空间和时间分辨率视为关键输入数据特征，在统一的潜在空间内实现跨模态的连贯分析。主要方法论贡献是将空间分辨率定义为可控的输出参数。使用单个统一的transformer编码器重构来自不同来源的掩码多模态EO数据，确保在传感器和分辨率间的泛化能力。

### 主要发现

RAMEN在预训练后能够有效迁移到已知和未见的传感器配置。在包含各种多传感器和多分辨率下游任务的社区标准PANGAEA基准测试中，RAMEN优于更大的最先进模型。

### 结论

RAMEN提供了一种处理不同分辨率地球观测数据的新方法，允许用户直接控制推理所需的细节级别，并在空间精度和计算成本之间进行明确的权衡。

### 翻译

地球观测数据跨越广泛的时空和光谱分辨率，从高分辨率光学图像到低分辨率多光谱产品或雷达时间序列。虽然最近的基础模型改进了多模态集成以学习有意义的表示，但它们通常期望固定的输入分辨率或基于特定传感器的编码器，限制了在异构EO模态间的泛化能力。为了克服这些局限性，我们引入了RAMEN，一种分辨率可调整的多模态编码器，以完全传感器无关的方式学习跨地球观测数据的共享视觉表示。RAMEN将模态、空间和时间分辨率视为关键输入数据特征， enabling在统一的潜在空间内跨模态的连贯分析。其主要方法论贡献是将空间分辨率定义为可控的输出参数，使用户能够直接控制推理时所需的细节级别，并允许在空间精度和计算成本之间进行明确的权衡。我们训练一个单一的统一transformer编码器，重构来自不同来源的掩码多模态地球观测数据，确保在传感器和分辨率间的泛化能力。一旦预训练完成，RAMEN能够有效迁移到已知和未见的传感器配置，并在包含各种多传感器和多分辨率下游任务的社区标准PANGAEA基准测试中，优于更大的最先进模型。我们的代码和预训练模型可在https://github.com/nicolashoudre/RAMEN获取。


### 论文摘要

Earth observation (EO) data spans a wide range of spatial, spectral, and temporal resolutions, from high-resolution optical imagery to low resolution multispectral products or radar time series. While recent foundation models have improved multimodal integration for learning meaningful representations, they often expect fixed input resolutions or are based on sensor-specific encoders limiting generalization across heterogeneous EO modalities. To overcome these limitations we introduce RAMEN, a resolution-adjustable multimodal encoder that learns a shared visual representation across EO data in a fully sensor-agnostic manner. RAMEN treats the modality and spatial and temporal resolutions as key input data features, enabling coherent analysis across modalities within a unified latent space. Its main methodological contribution is to define spatial resolution as a controllable output parameter, giving users direct control over the desired level of detail at inference and allowing explicit trade-offs between spatial precision and computational cost. We train a single, unified transformer encoder reconstructing masked multimodal EO data drawn from diverse sources, ensuring generalization across sensors and resolutions. Once pretrained, RAMEN transfers effectively to both known and unseen sensor configurations and outperforms larger state-of-the-art models on the community-standard PANGAEA benchmark, containing various multi-sensor and multi-resolution downstream tasks. Our code and pretrained model are available at https://github.com/nicolashoudre/RAMEN.

---

## 18. Learning Causality for Longitudinal Data

**论文链接:** [http://arxiv.org/abs/2512.04980v1](http://arxiv.org/abs/2512.04980v1)

**作者:** Mouad EL Bouchattaoui

**发布时间:** 2025-12-04

**备注:** PhD thesis manuscript

### GPT解析

### 总结

这篇论文开发了三种方法用于高维、时变数据中的因果推断和因果表示学习，包括CDVAE模型、基于增强RNN的长期反事实回归框架，以及基于Jacobian几何的因果表示学习方法。

### 背景

研究高维、时变数据中的因果推断和因果表示学习问题，需要处理未观察到的异质性、时变混杂条件和长程依赖关系。

### 目的

开发有效的方法来估计个体治疗效果、进行长期反事实推理，并揭示潜在原因在观察变量中的表现方式。

### 方法

提出CDVAE模型、基于CPC和InfoMax增强的RNN框架，以及基于解码器Jacobian几何的可解释性层和稀疏自表达先验。

### 主要发现

CDVAE在估计ITE方面优于基线方法；增强RNN框架在避免transformers计算成本的同时实现了最先进结果；基于Jacobian的方法能够恢复有意义的潜在到观察结构，无需锚定特征或单亲假设。

### 结论

所提出的方法为高维、时变数据中的因果推断和因果表示学习提供了有效解决方案，并在理论和实验上都证明了其优越性。

### 翻译

这篇论文开发了高维、时变数据中因果推断和因果表示学习的方法。第一个贡献引入了因果动态变分自编码器，这是一个通过捕捉仅影响结果的潜在风险因素导致的未观察到的异质性来估计个体治疗效果的模型。该模型对有效潜在调整和治疗效果误差提供了理论保证。在合成和真实数据集上的实验表明，该模型优于基线方法，并且最先进的模型通过增加其潜在替代品得到显著改进，无需访问真实调整变量即可接近最佳性能。第二个贡献提出了一个基于增强了对比预测编码和InfoMax的循环神经网络的长期反事实回归高效框架。它在时变混杂条件下捕获长程依赖关系，同时避免了transformers的计算成本，实现了最先进的结果，并将对比预测编码引入因果推断。第三个贡献通过解决潜在原因如何在观察变量中表现的问题推进了因果表示学习。我们引入了一个基于解码器Jacobian几何的模型无关可解释性层。稀疏自表达先验诱导模块化、可能重叠的观察特征组，与共享的潜在影响对齐。我们在不同设置中提供了恢复保证，并表明无需锚定特征或单亲假设即可恢复有意义的潜在到观察结构。还开发了可扩展的基于Jacobian的正则化技术。


### 论文摘要

This thesis develops methods for causal inference and causal representation learning (CRL) in high-dimensional, time-varying data.   The first contribution introduces the Causal Dynamic Variational Autoencoder (CDVAE), a model for estimating Individual Treatment Effects (ITEs) by capturing unobserved heterogeneity in treatment response driven by latent risk factors that affect only outcomes. CDVAE comes with theoretical guarantees on valid latent adjustment and generalization bounds for ITE error. Experiments on synthetic and real datasets show that CDVAE outperforms baselines, and that state-of-the-art models greatly improve when augmented with its latent substitutes, approaching oracle performance without access to true adjustment variables.   The second contribution proposes an efficient framework for long-term counterfactual regression based on RNNs enhanced with Contrastive Predictive Coding (CPC) and InfoMax. It captures long-range dependencies under time-varying confounding while avoiding the computational cost of transformers, achieving state-of-the-art results and introducing CPC into causal inference.   The third contribution advances CRL by addressing how latent causes manifest in observed variables. We introduce a model-agnostic interpretability layer based on the geometry of the decoder Jacobian. A sparse self-expression prior induces modular, possibly overlapping groups of observed features aligned with shared latent influences. We provide recovery guarantees in both disjoint and overlapping settings and show that meaningful latent-to-observed structure can be recovered without anchor features or single-parent assumptions. Scalable Jacobian-based regularization techniques are also developed.

---

## 19. Efficient Generative Transformer Operators For Million-Point PDEs

**论文链接:** [http://arxiv.org/abs/2512.04974v1](http://arxiv.org/abs/2512.04974v1)

**作者:** Armand Kassaï Koupaï, Lise Le Boudec, Patrick Gallinari

**发布时间:** 2025-12-04

### GPT解析

### 总结

ECHO是一种创新的transformer-operator框架，专为大规模PDE轨迹生成而设计，通过分层压缩、稀疏输入适应和轨迹级学习，克服了现有神经算子在密集网格扩展性、动态展开误差累积和任务特定设计方面的局限性，在多种复杂PDE系统中实现了最先进的性能。

### 背景

现有的神经算子在求解偏微分方程方面显示出潜力，但它们在实际应用中存在局限性，包括在密集网格上扩展性差、在动态展开过程中误差累积以及专门针对特定任务的设计。

### 目的

介绍ECHO框架，解决现有神经算子在处理大规模PDE模拟时的局限性，实现高效准确的百万点PDE轨迹生成。

### 方法

ECHO采用三个关键创新：(1)分层卷积编码-解码架构实现100倍时空压缩同时保持保真度；(2)训练和适应策略支持从稀疏输入生成高分辨率解；(3)生成式建模范式学习完整轨迹段减轻长期误差漂移。训练策略将表示学习与任务监督解耦，支持多种任务处理，包括轨迹生成、正逆问题和插值，并支持条件和无条件生成。

### 主要发现

ECHO在具有复杂几何形状、高频动态和长期跨度的多样化PDE系统上展示了最先进的性能，能够处理百万点模拟。

### 结论

ECHO框架通过分层架构、稀疏到密集的生成策略和轨迹级学习，成功解决了现有神经算子在处理大规模PDE模拟时的主要限制，实现了高效且准确的PDE求解。

### 翻译

我们引入ECHO，一个用于生成百万点PDE轨迹的transformer-operator框架。虽然现有的神经算子在求解偏微分方程方面显示出潜力，但由于在密集网格上扩展性差、动态展开过程中误差累积以及任务特定设计，它们在实际应用中仍然存在局限性。ECHO通过三个关键创新解决了这些挑战。(i)它采用分层卷积编码-解码架构，在保持网格点保真度的同时实现100倍的时空压缩。(ii)它结合训练和适应策略，能够从稀疏输入网格生成高分辨率PDE解。(iii)它采用生成式建模范式，学习完整的轨迹段，减轻长期误差漂移。训练策略将表示学习与下游任务监督解耦，使模型能够处理多种任务，如轨迹生成、正逆问题和插值。生成模型进一步支持条件和无条件生成。我们在具有复杂几何形状、高频动态和长期跨度的多样化PDE系统上，展示了百万点模拟的最先进性能。


### 论文摘要

We introduce ECHO, a transformer-operator framework for generating million-point PDE trajectories. While existing neural operators (NOs) have shown promise for solving partial differential equations, they remain limited in practice due to poor scalability on dense grids, error accumulation during dynamic unrolling, and task-specific design. ECHO addresses these challenges through three key innovations. (i) It employs a hierarchical convolutional encode-decode architecture that achieves a 100 $\times$ spatio-temporal compression while preserving fidelity on mesh points. (ii) It incorporates a training and adaptation strategy that enables high-resolution PDE solution generation from sparse input grids. (iii) It adopts a generative modeling paradigm that learns complete trajectory segments, mitigating long-horizon error drift. The training strategy decouples representation learning from downstream task supervision, allowing the model to tackle multiple tasks such as trajectory generation, forward and inverse problems, and interpolation. The generative model further supports both conditional and unconditional generation. We demonstrate state-of-the-art performance on million-point simulations across diverse PDE systems featuring complex geometries, high-frequency dynamics, and long-term horizons.

---

## 20. Stable Single-Pixel Contrastive Learning for Semantic and Geometric Tasks

**论文链接:** [http://arxiv.org/abs/2512.04970v1](http://arxiv.org/abs/2512.04970v1)

**作者:** Leonid Pogorelyuk, Niels Bracher, Aaron Verkleeren, Lars Kühmichel, Stefan T. Radev

**发布时间:** 2025-12-04

**备注:** UniReps Workshop 2025, 12 pages, 8 figures

### GPT解析

### 总结

本研究提出了一种稳定的对比损失函数家族，用于学习同时包含语义和几何信息的像素级表示，能够在图像间建立精确点对应关系，无需动量教师-学生训练。

### 背景

在计算机视觉和表示学习中，同时捕获语义和几何信息的像素级表示对于许多任务至关重要，但现有方法通常需要复杂的训练机制。

### 目的

开发一种能够学习像素级表示的方法，这些表示同时包含语义和几何信息，且能够在不同视图间建立精确对应关系。

### 方法

提出一种稳定的对比损失函数家族，将图像中的每个像素映射到一个过完备描述符，该描述符既具有视图不变性又具有语义意义，实现图像间的精确点对应。

### 主要发现

在合成的2D和3D环境中的实验表明，所提出的损失函数能够产生具有所需特性的过完备表示，能够有效捕获语义和几何信息。

### 结论

所提出的稳定对比损失函数家族为学习像素级表示提供了一种有效方法，无需复杂的动量教师-学生训练机制即可实现精确的点对应关系。

### 翻译

我们试点了一系列稳定的对比损失函数，用于学习像素级表示，这些表示同时捕获语义和几何信息。我们的方法将图像中的每个像素映射到一个过完备描述符，该描述符既具有视图不变性又具有语义意义。它能够在图像之间实现精确的点对应，而不需要基于动量的教师-学生训练。在合成的2D和3D环境中的两个实验证明了我们的损失函数的性质以及由此产生的过完备表示。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决几何任务和语义任务之间的鸿沟问题。几何任务需要像素级精确特征但不关注语义，而语义任务提供上下文信息但精度不足。这个问题很重要，因为它试图创建一种能同时捕捉图像语义和几何信息的像素级表示，弥合两种任务之间的差距，提高计算机视觉系统在空间定位和对象理解方面的综合能力。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有视觉模型的局限性：基于transformer的模型提供块级特征而非像素级精度；监督方法需要标签；自监督方法虽能实现语义密度但精度不足。作者借鉴了Xie等人的几何对比学习和UCN的像素级描述符思想，但创新性地避免了基于动量的教师-学生训练方案，设计了一种新的损失函数，能够在像素级同时编码语义和几何信息，适用于2D和3D环境。', '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是使用一种新的对比损失函数训练特征提取器，生成'过完备描述符'，每个像素的特征维度大于3，同时包含语义和几何信息。流程包括：1)生成同一图像的两个视图；2)使用U-Net作为特征提取器；3)应用几何和光度变换；4)计算图像内对比损失（匹配像素与随机像素）；5)计算图像间对比损失（不同视图间）；6)组合损失函数进行训练；7)最终得到能同时捕捉语义和几何信息的像素级表示。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)新的稳定对比损失函数家族，无需动量更新；2)像素级语义和几何信息的联合表示；3)扩展到3D环境。相比之前工作，本文避免了基于动量的训练方案，同时关注语义和几何信息（而非仅几何），在像素级别而非块级别操作，并提供2D和3D环境的统一解决方案。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种稳定的单像素对比学习方法，能够在不依赖动量更新的情况下，学习到同时包含语义和几何信息的像素级过完备表示，适用于2D和3D环境。'}


### 论文摘要

We pilot a family of stable contrastive losses for learning pixel-level representations that jointly capture semantic and geometric information. Our approach maps each pixel of an image to an overcomplete descriptor that is both view-invariant and semantically meaningful. It enables precise point-correspondence across images without requiring momentum-based teacher-student training. Two experiments in synthetic 2D and 3D environments demonstrate the properties of our loss and the resulting overcomplete representations.

---

## 21. Crack detection by holomorphic neural networks and transfer-learning-enhanced genetic optimization

**论文链接:** [http://arxiv.org/abs/2512.04947v1](http://arxiv.org/abs/2512.04947v1)

**作者:** Jonas Hund, Nicolas Cuenca, Tito Andriollo

**发布时间:** 2025-12-04

### GPT解析

### 总结

论文提出了一种基于应变数据检测二维固体中裂缝的新策略，结合了遗传优化、全纯神经网络和迁移学习技术，实现了比现有方法更高的效率。

### 背景

裂缝检测被表述为一个逆问题，传统方法如XFEM计算模型响应较慢，需要开发更高效的检测策略。

### 目的

开发一种比现有裂缝检测方法效率更高的新策略，通过结合遗传优化与全纯神经网络技术实现。

### 方法

将裂缝检测表述为逆问题并使用遗传优化求解；使用全纯势表达平面弹性问题的解，通过训练两个全纯神经网络确定势函数；势函数预先满足裂缝面上的平衡和自由牵引条件，仅基于边界信息进行快速训练；将遗传搜索分为长程和短程阶段，在后阶段应用迁移学习提高效率。

### 主要发现

在三个基准问题上测试了新策略，发现存在一个最佳训练周期数量可提供最佳整体性能；与使用XFEM的方法相比，在相同应力场表示精度假设下，所提出的方法快7到23倍；该方法虽针对单内部裂缝的简化情况提出，但具有推广可行性。

### 结论

将遗传优化与全纯神经网络和迁移学习相结合，为开发比现有方法效率更高的裂缝检测策略提供了有前景的途径。

### 翻译

介绍了一种基于应变数据检测二维固体中裂缝的新策略。裂缝检测被表述为一个逆问题，并使用遗传优化求解。新颖性在于评估每代的模型响应。具体来说，对应平面弹性问题的解通过全纯势表达，这些势函数由训练两个全纯神经网络确定。由于势函数预先满足裂缝面上的平衡和自由牵引条件，训练仅基于边界信息快速进行。通过将遗传搜索分为长程和短程阶段，在后阶段使用迁移学习，进一步提高了训练效率。新策略在三个基准问题上进行了测试，表明存在一个最佳训练周期数量，可提供最佳整体性能。还与一种使用XFEM计算模型响应的流行裂缝检测方法进行了比较。在假设应力场表示精度相同的情况下，发现所提出的方法比基于XFEM的方法快7到23倍。虽然该策略在此针对单内部裂缝的简化情况提出，但推广是可行的。总体而言，研究结果表明，将遗传优化与全纯神经网络和迁移学习相结合，为开发比现有方法效率更高的裂缝检测策略提供了有前景的途径。


### 论文摘要

A new strategy for detecting cracks in 2D solids based on strain data is introduced. Crack detection is formulated as an inverse problem and solved using genetic optimization. The novelty lies in the evaluation of the model response at each generation. Specifically, the solution to the corresponding plane elasticity problem is expressed via holomorphic potentials, which are determined by training two holomorphic neural networks. As the potentials satisfy equilibrium and traction-free conditions along the crack faces a priori, the training proceeds quickly based solely on boundary information. Training efficiency is further improved by splitting the genetic search into long-range and short-range stages, enabling the use of transfer learning in the latter. The new strategy is tested on three benchmark problems, showing that an optimal number of training epochs exists that provides the best overall performance. A comparison is also made with a popular crack detection approach that uses XFEM to compute the model response. Under the assumption of identical stress-field representation accuracy, the proposed method is found to be between 7 and 23 times faster than the XFEM-based approach. While the strategy is presented here for the simplified case of a single internal crack, generalization is feasible. Overall, the present findings demonstrate that combining genetic optimization with holomorphic neural networks and transfer learning offers a promising avenue for developing crack detection strategies with higher efficiency than those currently available.

---

## 22. Towards Adaptive Fusion of Multimodal Deep Networks for Human Action Recognition

**论文链接:** [http://arxiv.org/abs/2512.04943v1](http://arxiv.org/abs/2512.04943v1)

**作者:** Novanto Yudistira

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究提出了一种创新的人类动作识别方法，通过深度神经网络技术和多模态自适应融合策略，结合RGB、光流、音频和深度信息，利用门控机制实现多模态融合，超越了传统单模态识别方法的局限性。

### 背景

传统单模态识别方法存在局限性，需要探索多模态融合的新方法来提高动作识别的准确性和鲁棒性。

### 目的

开发一种基于门控机制的多模态自适应融合方法，实现从不同模态中选择性整合相关信息，提升动作识别性能。

### 方法

采用深度神经网络技术，结合RGB、光流、音频和深度信息等多种模态，通过门控机制和自适应加权融合架构进行多模态信息整合。

### 主要发现

门控机制能够提取关键特征，形成更全面的动作表示，显著提高识别性能；该方法在人类动作识别、暴力动作检测和多个自监督学习任务上均展现出优于传统单模态方法的性能。

### 结论

多模态信息融合技术有潜力革新各领域的动作识别系统，在监控和人机交互等应用中具有广阔前景，特别是在主动辅助生活相关环境中。

### 翻译

本研究通过利用深度神经网络技术和多模态自适应融合策略，包括RGB、光流、音频和深度信息，引入了一种开创性的人类动作识别方法。采用门控机制进行多模态融合，我们旨在超越传统单模态识别方法固有的局限性，同时探索多样化应用的新可能性。通过对门控机制和基于自适应加权的融合架构进行全面研究，我们的方法能够从各种模态中选择性地整合相关信息，从而增强动作识别任务的准确性和鲁棒性。我们仔细检查了各种门控融合策略，以找出多模态动作识别的最有效方法，展示了其优于传统单模态方法的优越性。门控机制有助于提取关键特征，形成更全面的动作表示，并显著提高识别性能。我们在人类动作识别、暴力动作检测和多个自监督学习任务中对基准数据集的评估显示了准确性的显著进步。本研究的意义在于其有潜力革新各领域的动作识别系统。多模态信息的融合承诺在监控和人机交互等复杂应用中发挥作用，特别是在与主动辅助生活相关的环境中。


### 论文摘要

This study introduces a pioneering methodology for human action recognition by harnessing deep neural network techniques and adaptive fusion strategies across multiple modalities, including RGB, optical flows, audio, and depth information. Employing gating mechanisms for multimodal fusion, we aim to surpass limitations inherent in traditional unimodal recognition methods while exploring novel possibilities for diverse applications. Through an exhaustive investigation of gating mechanisms and adaptive weighting-based fusion architectures, our methodology enables the selective integration of relevant information from various modalities, thereby bolstering both accuracy and robustness in action recognition tasks. We meticulously examine various gated fusion strategies to pinpoint the most effective approach for multimodal action recognition, showcasing its superiority over conventional unimodal methods. Gating mechanisms facilitate the extraction of pivotal features, resulting in a more holistic representation of actions and substantial enhancements in recognition performance. Our evaluations across human action recognition, violence action detection, and multiple self-supervised learning tasks on benchmark datasets demonstrate promising advancements in accuracy. The significance of this research lies in its potential to revolutionize action recognition systems across diverse fields. The fusion of multimodal information promises sophisticated applications in surveillance and human-computer interaction, especially in contexts related to active assisted living.

---

## 23. Channel-Aware Multi-Domain Feature Extraction for Automatic Modulation Recognition in MIMO Systems

**论文链接:** [http://arxiv.org/abs/2512.04899v1](http://arxiv.org/abs/2512.04899v1)

**作者:** Yunpeng Qu, Yazhou Sun, Bingyu Hui, Jintao Wang, Jian Wang

**发布时间:** 2025-12-04

**备注:** 5 pages, 3 figures

### GPT解析

### 总结

本文提出了一种用于多输入多输出(MIMO)系统自动调制识别(AMR)的信道感知多域特征提取(CAMD)框架，通过信道补偿和多域特征提取解决了多天线信道干扰问题，在复杂移动信道环境下展现出优越性能。

### 背景

自动调制识别(AMR)是非协作通信系统中的关键技术，深度学习方法因优异性能受到广泛关注，但现有研究主要集中在单输入单输出(SISO)系统，对多输入多输出(MIMO)系统的探索有限。

### 目的

提出一种用于MIMO系统AMR的信道感知多域特征提取(CAMD)框架，解决多天线信道干扰导致的调制识别挑战。

### 方法

通过高效的信道补偿模块重建传输信号，提取并整合多域特征，包括天线内时间相关性和天线间信道相关性，提高对信道干扰的鲁棒性表示能力。

### 主要发现

在MIMOSig-Ref数据集和复杂移动信道环境下的实验表明，CAMD方法比先前最先进方法具有性能优势。

### 结论

CAMD框架有效解决了MIMO系统中多天线信道干扰导致的调制识别挑战，为非协作通信系统中的AMR提供了新的解决方案。

### 翻译

自动调制识别(AMR)是非协作通信系统中的关键技术，旨在没有先验信息的情况下从信号中识别调制方案。基于深度学习(DL)的方法因其优异性能而受到广泛关注，但研究主要集中在单输入单输出(SISO)系统，对多输入多输出(MIMO)系统的探索有限。多天线的信道混淆效应会干扰MIMO信号的统计特性，使得识别特别具有挑战性。为了克服这些限制，我们提出了一种用于MIMO系统AMR的信道感知多域特征提取(CAMD)框架。我们的CAMD框架通过高效的信道补偿模块重建传输信号，并通过提取和整合多域特征（包括天线内时间相关性和天线间信道相关性）实现对信道干扰的更鲁棒表示能力。我们在广泛使用的数据集MIMOSig-Ref上验证了我们的方法，该数据集具有复杂的移动信道环境。大量实验证实了CAMD比先前最先进方法的性能优势。


### 论文摘要

Automatic modulation recognition (AMR) is a key technology in non-cooperative communication systems, aiming to identify the modulation scheme from signals without prior information. Deep learning (DL)-based methods have gained wide attention due to their excellent performance, but research mainly focuses on single-input single-output (SISO) systems, with limited exploration for multiple-input multiple-output (MIMO) systems. The confounding effects of multi-antenna channels can interfere with the statistical properties of MIMO signals, making identification particularly challenging. To overcome these limitations, we propose a Channel-Aware Multi-Domain feature extraction (CAMD) framework for AMR in MIMO systems. Our CAMD framework reconstructs the transmitted signal through an efficient channel compensation module and achieves a more robust representation capability against channel interference by extracting and integrating multi-domain features, including intra-antenna temporal correlations and inter-antenna channel correlations. We have verified our method on the widely-used dataset, MIMOSig-Ref, with complex mobile channel environments. Extensive experiments confirm the performance advantages of CAMD over previous state-of-the-art methods.

---

## 24. Series of quasi-uniform scatterings with fast search, root systems and neural network classifications

**论文链接:** [http://arxiv.org/abs/2512.04865v1](http://arxiv.org/abs/2512.04865v1)

**作者:** Igor V. Netay

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种在预定义维度空间中构建大型可扩展向量集合的方法，用于神经网络潜在空间配置和训练，特别适用于具有大量或未知类别数量的分类问题。

### 背景

神经网络在处理具有大量或未知类别数量的分类问题时面临挑战，传统方法通常需要分类层且难以灵活扩展类别数量。

### 目的

开发一种能够构建大型、间隔良好的向量集合的方法，使分类器无需分类层即可工作，并能灵活扩展类别数量而无需从头重新训练网络。

### 方法

基于半单李群不可约表示的组合数学和几何学，利用最高权重理论构造向量集合。

### 主要发现

该方法可以在最小可能维度的空间中创建大型且间隔良好的向量集合，具有规则对称结构，可简化潜在空间中的搜索问题，并支持在不重新训练网络的情况下扩展类别数量。

### 结论

所提出的向量集合构造方法为神经网络潜在空间配置和训练提供了灵活解决方案，特别适用于类别数量大或不确定的分类问题。

### 翻译

在本文中，我们描述了一种在给定维度的预定义空间中构建大型可扩展向量集合的方法。这些集合对神经网络潜在空间配置和训练很有用。对于具有大量或未知类别数量的分类问题，这允许构建没有分类层的分类器，并在不从头开始重新训练网络的情况下扩展类别数量。该构造方法可以在最小可能维度的空间中创建大型且间隔良好的向量集合。如果类别数量已知或可大致预测，可以选择足够大的向量集合大小。如果需要显著扩展类别数量，可以在相同的潜在空间中扩展集合，或将集合合并到更高维度的集合中，同时保持向量之间的相同间隔。此外，构造的向量集合的规则对称结构可以显著简化在潜在空间中搜索最近聚类中心或嵌入的问题。向量集合的构造基于半单李群不可约表示的组合数学和几何学，具有最高权重。


### 论文摘要

In this paper we describe an approach to construct large extendable collections of vectors in predefined spaces of given dimensions. These collections are useful for neural network latent space configuration and training. For classification problem with large or unknown number of classes this allows to construct classifiers without classification layer and extend the number of classes without retraining of network from the very beginning. The construction allows to create large well-spaced vector collections in spaces of minimal possible dimension. If the number of classes is known or approximately predictable, one can choose sufficient enough vector collection size. If one needs to significantly extend the number of classes, one can extend the collection in the same latent space, or to incorporate the collection into collection of higher dimensions with same spacing between vectors. Also, regular symmetric structure of constructed vector collections can significantly simplify problems of search for nearest cluster centers or embeddings in the latent space. Construction of vector collections is based on combinatorics and geometry of semi-simple Lie groups irreducible representations with highest weight.

---

## 25. Language Models as Semantic Teachers: Post-Training Alignment for Medical Audio Understanding

**论文链接:** [http://arxiv.org/abs/2512.04847v1](http://arxiv.org/abs/2512.04847v1)

**作者:** Tsai-Ning Wang, Lin-Lin Chen, Neil Zeghidour, Aaqib Saeed

**发布时间:** 2025-12-04

### GPT解析

### 总结

这篇论文介绍了一个名为AcuLa的轻量级后训练框架，通过将音频编码器与医疗语言模型对齐，赋予音频模型临床语义理解能力，从而提高其在诊断任务中的表现。

### 背景

预训练的音频模型在检测听诊声音中的声学模式方面表现出色，但往往难以理解其临床意义，限制了它们在诊断任务中的使用和性能。

### 目的

为了解决音频模型缺乏临床理解的问题，作者提出了AcuLa框架，旨在通过音频-语言对齐将纯声学模型转变为具有临床意识的诊断工具。

### 方法

1. 构建大规模数据集：利用现有的大型语言模型将音频记录的元数据翻译成临床报告；2. 对齐策略：结合表示级对比目标和自监督建模，确保模型学习临床语义的同时保留精细的时间线索。

### 主要发现

1. AcuLa在10个不同数据集的18个心肺任务上取得最先进结果；2. 分类基准测试的平均AUROC从0.68提高到0.79；3. COVID-19咳嗽检测任务的AUROC从0.55提高到0.89。

### 结论

音频-语言对齐技术将纯声学模型转变为具有临床意识的诊断工具，为基于音频的健康监测中增强生理理解建立了新的范式。

### 翻译

预训练的音频模型在检测听诊声音中的声学模式方面表现出色，但往往难以理解其临床意义，限制了它们在诊断任务中的使用和性能。为了弥合这一差距，我们引入了AcuLa（通过语言对齐实现音频临床理解），这是一个轻量级后训练框架，通过将任何音频编码器与医疗语言模型对齐，为其注入语义理解，该语言模型充当'语义教师'。为了实现大规模对齐，我们利用现成的大型语言模型构建了一个大型数据集，将现有音频记录的丰富结构化元数据翻译成连贯的临床报告。我们的对齐策略结合了表示级对比目标和自监督建模，确保模型在学习临床语义的同时保留精细的时间线索。AcuLa在来自10个不同数据集的18个多样化心肺任务上取得了最先进的结果，将分类基准测试的平均AUROC从0.68提高到0.79，在最具挑战性的COVID-19咳嗽检测任务中，将AUROC从0.55提高到0.89。我们的工作表明，这种音频-语言对齐将纯声学模型转变为具有临床意识的诊断工具，为基于音频的健康监测中增强生理理解建立了一种新范式。


### 论文摘要

Pre-trained audio models excel at detecting acoustic patterns in auscultation sounds but often fail to grasp their clinical significance, limiting their use and performance in diagnostic tasks. To bridge this gap, we introduce AcuLa (Audio-Clinical Understanding via Language Alignment), a lightweight post-training framework that instills semantic understanding into any audio encoder by aligning it with a medical language model, which acts as a "semantic teacher." To enable alignment at scale, we construct a large-scale dataset by leveraging off-the-shelf large language models to translate the rich, structured metadata accompanying existing audio recordings into coherent clinical reports. Our alignment strategy combines a representation-level contrastive objective with a self-supervised modeling, ensuring that the model learns clinical semantics while preserving fine-grained temporal cues. AcuLa achieves state-of-the-art results across 18 diverse cardio-respiratory tasks from 10 different datasets, improving the mean AUROC on classification benchmarks from 0.68 to 0.79 and, on the most challenging COVID-19 cough detection task, boosting the AUROC from 0.55 to 0.89. Our work demonstrates that this audio-language alignment transforms purely acoustic models into clinically-aware diagnostic tools, establishing a novel paradigm for enhancing physiological understanding in audio-based health monitoring.

---

## 26. Optimal Transport Event Representation for Anomaly Detection

**论文链接:** [http://arxiv.org/abs/2512.04839v1](http://arxiv.org/abs/2512.04839v1)

**作者:** Aditya Bhargava, Tianji Cai, Benjamin Nachman

**发布时间:** 2025-12-04

**备注:** 8 pages, 5 figures

### GPT解析

### 总结

本研究提出使用最优传输(OT)作为基于物理的中间事件表示方法，用于弱监督异常检测。

### 背景

在大型强子对撞机(LHC)奥运会基准数据集中，仅有0.5%的共振信号被注入。

### 目的

探索最优传输方法在弱监督异常检测中的应用效果。

### 方法

采用OT增强的特征集进行异常检测，并与标准高级可观测量和端到端深度学习方法进行比较。

### 主要发现

OT增强的特征集实现了比标准高级可观测量几乎两倍的显著性提升；在低信号区域，基于低水平四动量的端到端深度学习方法表现不佳；这些增益在不同信号类型和分类器中持续存在。

### 结论

结构化表示在机器学习异常检测中具有重要价值。

### 翻译

我们引入最优传输(OT)作为基于物理的中间事件表示，用于弱监督异常检测。在LHC奥运会基准数据集中仅注入0.5%的共振信号的情况下，OT增强的特征集实现了比标准高级可观测量几乎两倍的显著性提升，而在低水平四动量上进行端到端深度学习在低信号区域表现不佳。这些增益在信号类型和分类器中持续存在，强调了结构化表示在机器学习异常检测中的价值。


### 论文摘要

We introduce optimal transport (OT) as a physics-based intermediate event representation for weakly supervised anomaly detection. With only $0.5\%$ injection of resonant signals in the LHC Olympics benchmark datasets, the OT-augmented feature set achieves nearly twice the significance improvement of standard high-level observables, while end-to-end deep learning on low-level four-momenta struggles in the low-signal regime. The gains persist across signal types and classifiers, underscoring the value of structured representations in machine learning for anomaly detection.

---

## 27. Tokenizing Buildings: A Transformer for Layout Synthesis

**论文链接:** [http://arxiv.org/abs/2512.04832v1](http://arxiv.org/abs/2512.04832v1)

**作者:** Manuel Ladron de Guevara, Jinmo Rhee, Ardavan Bidgoli, Vaidas Razgaitis, Michael Bergin

**发布时间:** 2025-12-04

**备注:** 8 pages, 1 page References, 4 figures

### GPT解析

### 总结

本文介绍了小型建筑模型(SBM)，这是一种基于Transformer的架构，用于建筑信息模型(BIM)场景中的布局合成。该模型通过统一建筑元素的异构特征集为序列来解决建筑标记问题，同时保持组合结构。实验表明，SBM能够学习紧凑的房间嵌入，实现按类型和拓扑的可靠聚类，并支持强大的语义检索。在DDEP模式下，SBM能生成功能合理的布局，减少碰撞和边界违规，提高可导航性。

### 背景

建筑信息模型(BIM)场景中的布局合成是一个重要但具有挑战性的任务。建筑元素具有异构特征集，如何将这些特征有效整合并保持其组合结构是一个关键问题。

### 目的

开发一种能够有效处理建筑信息模型场景中布局合成的模型，解决如何将建筑元素进行标记的问题，同时保持其组合结构，并实现高质量的房间嵌入和布局生成。

### 方法

提出名为小型建筑模型(SBM)的基于Transformer的架构，包括：(1)将建筑元素的异构特征集表示为稀疏属性-特征矩阵，捕捉房间属性；(2)设计统一嵌入模块，学习分类和可能相关的连续特征组的联合表示；(3)使用单个Transformer主干网络在两种模式下训练：仅编码器路径生成高保真房间嵌入，以及编码器-解码器管道用于房间实体的自回归预测(数据驱动实体预测DDEP)。

### 主要发现

SBM能够学习紧凑的房间嵌入，这些嵌入可以按类型和拓扑可靠地聚类，从而实现强大的语义检索。在DDEP模式下，SBM生成的布局功能合理，碰撞和边界违规更少，可导航性得到改善。

### 结论

小型建筑模型(SBM)是一种有效的基于Transformer的架构，能够处理建筑信息模型场景中的布局合成问题。通过统一异构特征集并保持组合结构，SBM能够生成高质量的房间嵌入和功能合理的建筑布局，为建筑设计和分析提供了有价值的工具。

### 翻译

我们介绍了小型建筑模型(SBM)，这是一种基于Transformer的架构，用于建筑信息模型(BIM)场景中的布局合成。我们解决了如何通过将建筑元素的异构特征集统一为序列来对建筑进行标记的问题，同时保持组合结构。这些特征集被表示为捕获房间属性的稀疏属性-特征矩阵。然后，我们设计了一个统一嵌入模块，学习分类和可能相关的连续特征组的联合表示。最后，我们在两种模式下训练单个Transformer主干网络：仅编码器路径产生高保真房间嵌入，以及用于房间实体自回归预测的编码器-解码器管道，称为数据驱动实体预测(DDEP)。在检索和生成式布局合成的实验表明，SBM学习紧凑的房间嵌入，能够按类型和拓扑可靠聚类，实现强大的语义检索。在DDEP模式下，SBM生成功能合理的布局，碰撞和边界违规更少，可导航性得到改善。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决建筑信息建模(BIM)场景中的布局合成问题，即如何生成功能正确且语义连贯的建筑房间布局。这个问题在现实中很重要，因为建筑设计是一个高度约束但重复性强的过程，设计师需要布置墙壁、门、橱柜和通道等元素以满足规范和功能需求；在研究中也很重要，因为现有的3D生成模型通常将场景视为非结构化几何，产生视觉合理但难以编辑的输出，经常违反基本的有效性规则。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者首先分析了现有方法的局限性，包括规则系统的约束性、深度学习模型对训练数据的依赖性、以及LLM/VLM方法缺乏几何基础等问题。他们借鉴了基于图的生成模型、图像和特征神经模型、语言模型和多模态模型等现有工作，但创新性地提出将建筑元素转换为'标记'序列，设计混合类型嵌入模块处理不同特征，并创建支持两种操作模式的统一Transformer骨干网络，从而解决了现有方法的不足。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将建筑元素(房间、墙壁、门、窗、家具等)转换为'标记'(tokens)，使它们能像文本序列一样被Transformer处理，并使用混合类型嵌入模块处理不同特征。整体流程包括：1)将建筑数据提取并转换为BIM-Token包；2)使用混合类型嵌入模块将稀疏特征矩阵转换为密集标记嵌入；3)通过Transformer骨干网络的两种模式处理：仅编码器用于房间嵌入和检索，编码器-解码器用于布局生成；4)使用覆盖率、可导航性和重叠/间隙等指标评估结果。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)规范化的分层标记化方法，将建筑元素异构特征转换为序列；2)混合类型嵌入模块，处理分类和连续特征；3)统一的Transformer骨干网络支持两种操作模式。相比之前工作，它不依赖显式规则，提供结构化标记表示而非非结构化几何，专门针对建筑布局优化，能处理连续几何属性，同时支持生成和检索任务，而非单一功能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文提出了一种基于Transformer的小型建筑模型(SBM)，通过将建筑元素标记化并使用混合类型嵌入模块，实现了功能正确且语义连贯的建筑房间布局生成，同时支持房间嵌入和检索任务。'}


### 论文摘要

We introduce Small Building Model (SBM), a Transformer-based architecture for layout synthesis in Building Information Modeling (BIM) scenes. We address the question of how to tokenize buildings by unifying heterogeneous feature sets of architectural elements into sequences while preserving compositional structure. Such feature sets are represented as a sparse attribute-feature matrix that captures room properties. We then design a unified embedding module that learns joint representations of categorical and possibly correlated continuous feature groups. Lastly, we train a single Transformer backbone in two modes: an encoder-only pathway that yields high-fidelity room embeddings, and an encoder-decoder pipeline for autoregressive prediction of room entities, referred to as Data-Driven Entity Prediction (DDEP). Experiments across retrieval and generative layout synthesis show that SBM learns compact room embeddings that reliably cluster by type and topology, enabling strong semantic retrieval. In DDEP mode, SBM produces functionally sound layouts, with fewer collisions and boundary violations and improved navigability.

---

## 28. LatentFM: A Latent Flow Matching Approach for Generative Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2512.04821v1](http://arxiv.org/abs/2512.04821v1)

**作者:** Huynh Trinh Ngoc, Hoang Anh Nguyen Kim, Toan Nguyen Hai, Long Tran Quoc

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了LatentFM，一种在潜在空间中操作的基于流的医学图像分割模型，通过条件速度场引导流，实现高精度且不确定性感知的预测，并生成置信度图提供更丰富的临床信息。

### 背景

生成模型在流匹配(FM)的出现下取得了显著进展。流匹配作为一种无需模拟的基于流的框架，能够学习精确的数据密度，展现了强大的生成能力并引起了广泛关注。

### 目的

提出LatentFM，一种在潜在空间中操作的基于流的模型，用于医学图像分割。

### 方法

设计两个变分自编码器(VAEs)将医学图像及其对应掩码编码到低维潜在空间；估计条件速度场基于输入图像引导流；通过采样多个潜在表示合成多样化的分割输出；生成置信度图量化模型确定性。

### 主要发现

方法实现高精度且不确定性感知的预测；为临床医生提供更丰富的信息；在ISIC-2018和CVC-Clinic数据集上实验；与多种确定性和生成性基线方法比较；定性和定量结果均显示优越的分割准确性。

### 结论

LatentFM在潜在空间中实现了高效的医学图像分割，具有优越的分割准确性。

### 翻译

随着流匹配(FM)的出现，生成模型取得了显著进展。它展示了强大的生成能力，并作为一种无需模拟的基于流的框架，能够学习精确的数据密度，引起了广泛关注。受这些进展的启发，我们提出了LatentFM，一种在潜在空间中操作的基于流的模型，用于医学图像分割。为了建模数据分布，我们首先设计了两个变分自编码器(VAEs)，将医学图像及其对应的掩码编码到低维潜在空间。然后，我们估计一个条件速度场，基于输入图像引导流。通过采样多个潜在表示，我们的方法合成了多样化的分割输出，其像素级方差可靠地捕获了底层数据分布，实现了高精度且具有不确定性感知能力的预测。此外，我们生成置信度图来量化模型确定性，为临床医生提供更丰富的信息进行深入分析。我们在两个数据集ISIC-2018和CVC-Clinic上进行了实验，并将我们的方法与几种先前的基线方法进行了比较，包括确定性和生成性方法。通过全面的评估，定性和定量结果均表明，我们的方法在保持潜在空间高效性的同时，实现了优越的分割准确性。


### 论文摘要

Generative models have achieved remarkable progress with the emergence of flow matching (FM). It has demonstrated strong generative capabilities and attracted significant attention as a simulation-free flow-based framework capable of learning exact data densities. Motivated by these advances, we propose LatentFM, a flow-based model operating in the latent space for medical image segmentation. To model the data distribution, we first design two variational autoencoders (VAEs) to encode both medical images and their corresponding masks into a lower-dimensional latent space. We then estimate a conditional velocity field that guides the flow based on the input image. By sampling multiple latent representations, our method synthesizes diverse segmentation outputs whose pixel-wise variance reliably captures the underlying data distribution, enabling both highly accurate and uncertainty-aware predictions. Furthermore, we generate confidence maps that quantify the model certainty, providing clinicians with richer information for deeper analysis. We conduct experiments on two datasets, ISIC-2018 and CVC-Clinic, and compare our method with several prior baselines, including both deterministic and generative approach models. Through comprehensive evaluations, both qualitative and quantitative results show that our approach achieves superior segmentation accuracy while remaining highly efficient in the latent space.

---

## 29. LaFiTe: A Generative Latent Field for 3D Native Texturing

**论文链接:** [http://arxiv.org/abs/2512.04786v1](http://arxiv.org/abs/2512.04786v1)

**作者:** Chia-Hao Chen, Zi-Xin Zou, Yan-Pei Cao, Ze Yuan, Guan Luo, Xiaojuan Qi, Ding Liang, Song-Hai Zhang, Yuan-Chen Guo

**发布时间:** 2025-12-04

**备注:** Project page: https://vast-ai-research.github.io/LaFiTe/

### GPT解析

### 总结

本文提出了一种名为LaFiTe的框架，用于在3D表面上生成高保真度的无缝纹理，解决了3D原生纹理生成中的表示差距问题。

### 背景

3D原生纹理生成(3D-native texturing)是一个基础性开放挑战，有望克服基于UV和多视图投影方法的长期局限性，但现有方法因缺乏强大且通用的潜在表示而受到限制。

### 目的

解决3D表面纹理生成中的表示差距问题，开发一种能够生成高保真度、无缝纹理的新方法。

### 方法

LaFiTe框架通过学习生成3D生成稀疏潜在彩色场，核心是使用变分自编码器(VAE)将复杂表面外观编码为稀疏、结构化的潜在空间，再解码为连续彩色场，并在此基础上使用条件修正流模型合成高质量纹理。

### 主要发现

LaFiTe实现的表示在重建上超过最先进方法10 dB以上PSNR，通过有效分离纹理外观与网格拓扑和UV参数化实现了前所未有的保真度。

### 结论

LaFiTe不仅为3D原生纹理生成设定了新基准，还支持材料合成和纹理超分辨率等灵活的下游应用，为下一代3D内容创建工作流程铺平了道路。

### 翻译

直接在3D表面上生成高保真度的无缝纹理，我们称之为3D原生纹理生成，仍然是一个基础性的开放挑战，有可能克服基于UV和多视图投影方法的长期局限性。然而，现有的原生方法因缺乏强大且通用的潜在表示而受到限制，这严重限制了其生成纹理的保真度和通用性。我们将这一表示差距确定为进一步进步的主要障碍。我们引入了LaFiTe框架，通过学习生成3D生成稀疏潜在彩色场来解决这一挑战。其核心是LaFiTe采用变分自编码器(VAE)将复杂表面外观编码为稀疏、结构化的潜在空间，随后解码为连续彩色场。这种表示通过有效分离纹理外观与网格拓扑和UV参数化，实现了前所未有的保真度，在重建上超过最先进方法10 dB以上PSNR。基于这一强大的表示，条件修正流模型能够跨不同风格和几何形状合成高质量、连贯的纹理。大量实验证明，LaFiTe不仅为3D原生纹理生成设定了新基准，还支持材料合成和纹理超分辨率等灵活的下游应用，为下一代3D内容创建工作流程铺平了道路。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决3D原生纹理生成问题，即如何直接在3D表面上生成高质量、无缝的纹理。这个问题很重要，因为纹理对3D资产的视觉丰富度和真实感贡献往往比几何本身更大，而现有的多视角投影和UV展开方法存在固有缺陷：多视角投影会产生接缝和伪影，UV展开则依赖不唯一且常扭曲的参数化，导致纹理变形。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者识别出3D原生纹理生成的主要瓶颈是缺乏强大且通用的潜在表示，因此提出将纹理建模为3D生成稀疏潜在颜色场。他们设计了一个新型变分自编码器(VAE)来学习这种表示，并发现同一编码器能同时处理纹理和几何信息。该方法借鉴了现有工作如SparseFlex的稀疏结构、TRELLIS的VAE架构，以及点体素注意力机制等，但进行了创新改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将3D纹理表示为稀疏潜在颜色场，这种表示是局部和连续的，能集中在物体表面附近建模，并将纹理与网格拓扑解耦。整体流程包括：1)纹理编码：从网格采样彩色点云并压缩为潜在特征；2)几何编码：用单色点云提取几何潜在表示；3)纹理解码：从潜在表示重建连续颜色场；4)条件纹理合成：使用修正流模型生成纹理潜在表示；5)应用接口：将3D颜色场烘焙为2D纹理图或进行局部修改。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)高保真潜在颜色场表示，实现与网格拓扑解耦的纹理建模；2)统一的纹理生成框架，通过重用纹理编码器处理几何信息；3)最先进的原生纹理生成质量。相比之前工作，LaFiTe直接使用3D彩色点云而非2D特征投影，无需单独的几何编码器，在3D空间直接监督而非基于渲染损失，并采用分层生成流程(几何→albedo→PBR)。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LaFiTe通过引入新颖的稀疏潜在颜色场表示和统一生成框架，首次实现了高质量、无缝的3D原生纹理生成，显著超越了现有方法并支持多种下游应用。'}


### 论文摘要

Generating high-fidelity, seamless textures directly on 3D surfaces, what we term 3D-native texturing, remains a fundamental open challenge, with the potential to overcome long-standing limitations of UV-based and multi-view projection methods. However, existing native approaches are constrained by the absence of a powerful and versatile latent representation, which severely limits the fidelity and generality of their generated textures. We identify this representation gap as the principal barrier to further progress. We introduce LaFiTe, a framework that addresses this challenge by learning to generate textures as a 3D generative sparse latent color field. At its core, LaFiTe employs a variational autoencoder (VAE) to encode complex surface appearance into a sparse, structured latent space, which is subsequently decoded into a continuous color field. This representation achieves unprecedented fidelity, exceeding state-of-the-art methods by >10 dB PSNR in reconstruction, by effectively disentangling texture appearance from mesh topology and UV parameterization. Building upon this strong representation, a conditional rectified-flow model synthesizes high-quality, coherent textures across diverse styles and geometries. Extensive experiments demonstrate that LaFiTe not only sets a new benchmark for 3D-native texturing but also enables flexible downstream applications such as material synthesis and texture super-resolution, paving the way for the next generation of 3D content creation workflows.

---

## 30. YingMusic-Singer: Zero-shot Singing Voice Synthesis and Editing with Annotation-free Melody Guidance

**论文链接:** [http://arxiv.org/abs/2512.04779v1](http://arxiv.org/abs/2512.04779v1)

**作者:** Junjie Zheng, Chunbo Hao, Guobin Ma, Xiaoyu Zhang, Gongyu Chen, Chaofan Ding, Zihao Chen, Lei Xie

**发布时间:** 2025-12-04

**备注:** 13 pages, 3 figures

### GPT解析

### 总结

本文提出了一种旋律驱动的歌唱语音合成框架，无需音素级对齐即可合成任意歌词，解决了传统SVS方法资源密集和可扩展性差的问题。

### 背景

传统歌唱语音合成(SVS)在实际部署中受到限制，因为它严重依赖准确的音素级对齐和手动标注的旋律轮廓，这些要求资源密集且阻碍了可扩展性。

### 目的

克服传统SVS方法的局限性，提出一个能够合成任意歌词并遵循任何参考旋律的SVS框架，无需依赖音素级对齐。

### 方法

基于Diffusion Transformer (DiT)架构，增加了专门的旋律提取模块直接从参考音频派生旋律表示；使用教师模型指导旋律提取器优化；采用隐式对齐机制强制执行相似性分布约束；使用弱标注歌曲数据改进持续时间建模；引入Flow-GRPO强化学习策略与多目标奖励函数。

### 主要发现

实验表明，该模型在客观指标和主观听力测试中均优于现有方法，特别是在零样本和歌词适应设置中表现优异，同时保持高质量音频质量无需手动标注。

### 结论

这项工作为推进数据高效的歌唱语音合成提供了实用且可扩展的解决方案，作者已发布推理代码和模型检查点以支持可重复性。

### 翻译

歌唱语音合成(SVS)由于其严重依赖准确的音素级对齐和手动标注的旋律轮廓，在实际部署中仍然受到限制，这些要求资源密集并阻碍可扩展性。为克服这些限制，我们提出了一个旋律驱动的SVS框架，能够合成任意歌词并遵循任何参考旋律，而不依赖音素级对齐。我们的方法基于Diffusion Transformer (DiT)架构，增强了专门的旋律提取模块，直接从参考音频中派生旋律表示。为确保稳健的旋律编码，我们使用教师模型指导旋律提取器的优化，同时采用隐式对齐机制强制执行相似性分布约束，以提高旋律稳定性和连贯性。此外，我们使用弱标注歌曲数据改进持续时间建模，并引入具有多目标奖励函数的Flow-GRPO强化学习策略，共同提高发音清晰度和旋律保真度。实验表明，我们的模型在客观指标和主观听力测试中均优于现有方法，特别是在零样本和歌词适应设置中表现优异，同时保持高质量音频质量无需手动标注。这项工作为推进数据高效的歌唱语音合成提供了实用且可扩展的解决方案。为支持可重复性，我们发布了推理代码和模型检查点。


### 论文摘要

Singing Voice Synthesis (SVS) remains constrained in practical deployment due to its strong dependence on accurate phoneme-level alignment and manually annotated melody contours, requirements that are resource-intensive and hinder scalability. To overcome these limitations, we propose a melody-driven SVS framework capable of synthesizing arbitrary lyrics following any reference melody, without relying on phoneme-level alignment. Our method builds on a Diffusion Transformer (DiT) architecture, enhanced with a dedicated melody extraction module that derives melody representations directly from reference audio. To ensure robust melody encoding, we employ a teacher model to guide the optimization of the melody extractor, alongside an implicit alignment mechanism that enforces similarity distribution constraints for improved melodic stability and coherence. Additionally, we refine duration modeling using weakly annotated song data and introduce a Flow-GRPO reinforcement learning strategy with a multi-objective reward function to jointly enhance pronunciation clarity and melodic fidelity. Experiments show that our model achieves superior performance over existing approaches in both objective measures and subjective listening tests, especially in zero-shot and lyric adaptation settings, while maintaining high audio quality without manual annotation. This work offers a practical and scalable solution for advancing data-efficient singing voice synthesis. To support reproducibility, we release our inference code and model checkpoints.

---

## 31. Complementary Characterization of Agent-Based Models via Computational Mechanics and Diffusion Models

**论文链接:** [http://arxiv.org/abs/2512.04771v1](http://arxiv.org/abs/2512.04771v1)

**作者:** Roberto Garrone

**发布时间:** 2025-12-04

**备注:** 11 pages. Methods paper introducing a dual-domain framework for analyzing ABM dynamics. Companion temporal-analysis preprint: arXiv:2510.12729

### GPT解析

### 总结

本文引入扩散模型作为ε-机器的正交和互补工具，用于表征基于智能体模型(ABMs)的输出。ε-机器捕捉ABM生成的时间序列的预测时间结构和内在计算，而扩散模型表征高维横截面分布，学习底层数据流形，并 enable 合成生成群体水平的结果。这两种方法在不同的数学域上运行：过程vs分布，它们的组合提供了基于时间组织和分布几何的ABM行为两轴表示。

### 背景

本文是对预印本'Characterizing Agent-Based Model Dynamics via ε-Machines and Kolmogorov-Style Complexity'的扩展，旨在引入扩散模型作为分析基于智能体模型(ABMs)输出的新工具。

### 目的

结合ε-机器和扩散模型来分析基于智能体的模型，建立将计算力学与基于分数的生成建模相结合的框架，用于ABM输出的结构分析，并将ABM表征置于现代机器学习方法中密度估计和内在计算的更广阔背景下。

### 方法

ε-机器捕捉ABM生成的时间序列的预测时间结构和内在计算；扩散模型表征高维横截面分布，学习底层数据流形，并 enable 合成生成群体水平的结果；提供形式分析证明两种方法在不同的数学域上运行；使用老年人护理ABM数据集验证框架；提供精确的定义和命题形式化ε-机器和扩散模型之间的数学互补性。

### 主要发现

ε-机器和扩散模型在不同的数学域上运行：过程vs分布；它们的组合产生了基于时间组织和分布几何的ABM行为两轴表示；这是第一个将计算力学与基于分数的生成建模相结合用于ABM输出结构分析的框架。

### 结论

建立了一个原则性的方法，用于联合分析复杂模拟模型中的时间可预测性和高维分布结构。

### 翻译

本文通过引入扩散模型作为正交和互补工具来表征基于智能体模型(ABMs)的输出，从而扩展了预印本'通过ε-机器和Kolmogorov风格复杂度表征基于智能体模型动态'。其中ε-机器捕捉ABM生成的时间序列的预测时间结构和内在计算，而扩散模型表征高维横截面分布，学习底层数据流形，并 enable 合成生成群体水平的结果。我们提供了形式分析，证明两种方法在不同的数学域上运行—过程vs分布—并表明它们的组合产生了基于时间组织和分布几何的ABM行为两轴表示。据我们所知，这是第一个将计算力学与基于分数的生成建模相结合用于ABM输出结构分析的框架，从而使ABM表征处于现代机器学习方法中密度估计和内在计算的更广阔背景下。该框架使用配套论文中介绍的相同老年人护理ABM数据集进行验证，我们提供了精确的定义和命题，形式化了ε-机器和扩散模型之间的数学互补性。这为联合分析复杂模拟模型中的时间可预测性和高维分布结构建立了原则性的方法论。


### 论文摘要

This article extends the preprint "Characterizing Agent-Based Model Dynamics via $ε$-Machines and Kolmogorov-Style Complexity" by introducing diffusion models as orthogonal and complementary tools for characterizing the output of agent-based models (ABMs). Where $ε$-machines capture the predictive temporal structure and intrinsic computation of ABM-generated time series, diffusion models characterize high-dimensional cross-sectional distributions, learn underlying data manifolds, and enable synthetic generation of plausible population-level outcomes. We provide a formal analysis demonstrating that the two approaches operate on distinct mathematical domains -processes vs.\ distributions- and show that their combination yields a two-axis representation of ABM behavior based on temporal organization and distributional geometry. To our knowledge, this is the first framework to integrate computational mechanics with score-based generative modeling for the structural analysis of ABM outputs, thereby situating ABM characterization within the broader landscape of modern machine-learning methods for density estimation and intrinsic computation. The framework is validated using the same elder-caregiver ABM dataset introduced in the companion paper, and we provide precise definitions and propositions formalizing the mathematical complementarity between $ε$-machines and diffusion models. This establishes a principled methodology for jointly analyzing temporal predictability and high-dimensional distributional structure in complex simulation models.

---

## 32. Bridging Simulation and Reality: Cross-Domain Transfer with Semantic 2D Gaussian Splatting

**论文链接:** [http://arxiv.org/abs/2512.04731v1](http://arxiv.org/abs/2512.04731v1)

**作者:** Jian Tang, Pu Pang, Haowen Sun, Chengzhong Ma, Xingyu Chen, Hua Huang, Xuguang Lan

**发布时间:** 2025-12-04

### GPT解析

### 总结

提出了一种名为语义2D高斯泼溅（S2GS）的新方法，用于解决机器人操作中的跨域迁移问题，通过提取域不变特征有效弥合模拟与真实环境之间的差距。

### 背景

跨域迁移在机器人操作中长期存在挑战，模拟与真实环境间的显著域差距导致现有方法（如域随机化、适应和模拟-真实校准）需要大量调参或无法推广到新场景。

### 目的

开发一种能够提取域不变特征的方法，使策略在模拟中训练后能够在真实世界中有效部署，提高泛化能力。

### 方法

提出语义2D高斯泼溅（S2GS）表示方法，构建多视图2D语义场并通过特征级高斯泼溅投影到统一3D空间，使用语义过滤机制移除不相关背景内容。

### 主要发现

S2GS显著提高了模拟到真实的迁移能力，在真实场景中保持高且稳定的任务性能。

### 结论

通过在策略训练中使用域不变特征并在部署时提供相同特征作为输入，可以有效弥合域差距，显著改善策略泛化。

### 翻译

跨域迁移在机器人操作中由于模拟和真实环境之间的显著域差距而一直是一个长期存在的挑战。现有的域随机化、适应和模拟-真实校准等方法通常需要大量调参或无法推广到未见过的场景。为解决这个问题，我们观察到如果在模拟中的策略训练期间使用域不变特征，并且在真实世界部署期间能够提取并提供相同的特征作为策略输入，则可以有效弥合域差距，显著提高策略泛化能力。因此，我们提出了语义2D高斯泼溅（S2GS），一种提取以对象为中心的域不变空间特征的新型表示方法。S2GS构建多视图2D语义场，并通过特征级高斯泼溅将它们投影到统一的3D空间中。语义过滤机制移除不相关的背景内容，确保策略学习的干净一致输入。为评估S2GS的有效性，我们采用扩散策略作为下游学习算法，在ManiSkill模拟环境中进行实验，随后在真实世界进行部署。结果表明，S2GS显著提高了模拟到真实的迁移能力，在真实场景中保持高且稳定的任务性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决机器人操作中的跨域迁移问题，即如何让在模拟环境中训练的机器人策略有效应用到现实世界中。这个问题非常重要，因为机器人学习通常在模拟环境中进行，但模拟和现实世界存在视觉外观、物体多样性、环境复杂性等差异，导致模拟训练的策略往往无法直接迁移到现实环境中，限制了机器人在实际应用中的效能。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者从认知神经科学获得灵感，观察到人类能在不熟悉环境中执行已学习技能，因为大脑能提取稳定不变的视觉表示。基于此，作者思考如果在模拟训练中使用域不变特征，并在现实部署时提取相同特征，可有效缩小域差距。该方法借鉴了3D高斯泼溅技术但改进为2D版本以解决多视角一致性问题；利用CLIP和SAM等视觉基础模型提取语义特征；结合扩散策略作为下游学习算法。整体设计围绕提取物体中心的域不变空间特征这一核心思想展开。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提取域不变的空间特征作为机器人策略输入，这些特征对视觉变化不敏感但对物体几何结构敏感。整体流程分为三部分：1)初始化阶段，使用多视角图像构建语义2D高斯泼溅场，提取层次化语义特征并优化模型；2)执行阶段，通过语义检索过滤任务相关物体，提取空间特征作为扩散策略输入；3)动态更新，通过优化SE(3)变换跟踪物体运动，保持场景表示的实时准确性。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出语义2D高斯泼溅(S2GS)提取物体中心的域不变空间特征；2)语义过滤机制移除无关背景干扰；3)提供实时性能满足在线机器人控制需求；4)显著提高跨域泛化能力。相比之前工作，S2GS与传统视觉输入方法不同在于其对视觉变化不敏感；与传统3D表示方法不同在于解决了多视角渲染的深度不一致问题；与传统域适应方法不同在于减少了对大量域随机化的需求；与实时性要求的方法不同在于提供了满足实时控制的高效性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出的语义2D高斯泼溅方法通过提取物体中心的域不变空间特征，有效缩小了模拟与现实环境之间的差距，显著提高了机器人操作策略的跨域迁移能力，同时保持实时性能和任务相关性。'}


### 论文摘要

Cross-domain transfer in robotic manipulation remains a longstanding challenge due to the significant domain gap between simulated and real-world environments. Existing methods such as domain randomization, adaptation, and sim-real calibration often require extensive tuning or fail to generalize to unseen scenarios. To address this issue, we observe that if domain-invariant features are utilized during policy training in simulation, and the same features can be extracted and provided as the input to policy during real-world deployment, the domain gap can be effectively bridged, leading to significantly improved policy generalization. Accordingly, we propose Semantic 2D Gaussian Splatting (S2GS), a novel representation method that extracts object-centric, domain-invariant spatial features. S2GS constructs multi-view 2D semantic fields and projects them into a unified 3D space via feature-level Gaussian splatting. A semantic filtering mechanism removes irrelevant background content, ensuring clean and consistent inputs for policy learning. To evaluate the effectiveness of S2GS, we adopt Diffusion Policy as the downstream learning algorithm and conduct experiments in the ManiSkill simulation environment, followed by real-world deployment. Results demonstrate that S2GS significantly improves sim-to-real transferability, maintaining high and stable task performance in real-world scenarios.

---

## 33. CIG-MAE: Cross-Modal Information-Guided Masked Autoencoder for Self-Supervised WiFi Sensing

**论文链接:** [http://arxiv.org/abs/2512.04723v1](http://arxiv.org/abs/2512.04723v1)

**作者:** Gang Liu, Yanling Hao, Yixuan Zou

**发布时间:** 2025-12-04

**备注:** 11 pages, 7 figures

### GPT解析

### 总结

本文提出了一种名为CIG-MAE的跨模态信息引导掩码自编码器，用于解决WiFi CSI人体动作识别中的标注挑战，通过自适应信息引导掩码策略和Barlow Twins正则化器实现了优异的性能。

### 背景

基于WiFi信道状态信息(CSI)的人体动作识别因其普遍性、设备无关性和隐私保护能力成为视觉方法的替代方案，但手动标注成本高和公开数据集规模有限限制了监督方法的性能。

### 目的

克服现有对比学习方法与CSI物理语义冲突且需要大批量训练的问题，开发适合CSI数据的自监督学习方法。

### 方法

提出CIG-MAE，采用对称双流架构重建CSI幅度和相位，结合高掩码率；设计自适应信息引导掩码策略动态关注高信息密度区域；集成Barlow Twins正则化器实现无负样本的跨模态表示对齐。

### 主要发现

在三个公共数据集上，CIG-MAE持续优于最先进自监督方法，甚至超过完全监督基线，展示了优越的数据效率、鲁棒性和表示泛化能力。

### 结论

CIG-MAE有效解决了CSI数据标注挑战，在动作识别任务中表现出色，特别是在数据有限情况下。

### 翻译

基于WiFi信道状态信息(CSI)的人体动作识别已成为一种有吸引力的替代方案，优于基于视觉的方法，因为它具有普遍性、设备无关性和固有的隐私保护能力。然而，手动标注的高成本和公开可用的CSI数据集的有限规模限制了监督方法的性能。自监督学习提供了一条有希望的途径，但现有的对比范式依赖于与无线电信号物理语义冲突的数据增强，并且需要大批量训练，使它们不太适合CSI。为了克服这些挑战，我们引入了CIG-MAE——一种跨模态信息引导的掩码自编码器——它使用具有高掩码率的对称双流架构重建CSI的幅度和相位。具体来说，我们提出了一种自适应信息引导掩码策略，动态关注高信息密度的时间-频率区域以提高学习效率，并整合了Barlow Twins正则化器以在没有负样本的情况下对齐跨模态表示。在三个公共数据集上的实验表明，CIG-MAE持续优于最先进的自监督方法，甚至超过了完全监督的基线，展示了优越的数据效率、鲁棒性和表示泛化能力。


### 论文摘要

Human Action Recognition using WiFi Channel State Information (CSI) has emerged as an attractive alternative to vision-based methods due to its ubiquity, device-agnostic nature, and inherent privacy-preserving capabilities. However, the high cost of manual annotation and the limited scale of publicly available CSI datasets restrict the performance of supervised approaches. Self-supervised learning (SSL) offers a promising avenue, but existing contrastive paradigms rely on data augmentations that conflict with the physical semantics of radio signals and require large-batch training, making them poorly suited for CSI. To overcome these challenges, we introduce CIG-MAE -- a Cross-modal Information-Guided Masked Autoencoder -- that reconstructs both the amplitude and phase of CSI using a symmetric dual-stream architecture with a high masking ratio. Specifically, we propose an Adaptive Information-Guided Masking strategy that dynamically allocates attention to time-frequency regions with high information density to improve learning efficiency, and incorporate a Barlow Twins regularizer to align cross-modal representations without negative samples. Experiments on three public datasets show that CIG-MAE consistently outperforms SOTA SSL methods and even surpasses a fully supervised baseline, demonstrating superior data efficiency, robustness, and representation generalization.

---

## 34. TRINITY: An Evolved LLM Coordinator

**论文链接:** [http://arxiv.org/abs/2512.04695v1](http://arxiv.org/abs/2512.04695v1)

**作者:** Jinglue Xu, Qi Sun, Peter Schwendeman, Stefan Nielsen, Edoardo Cetin, Yujin Tang

**发布时间:** 2025-12-04

### GPT解析

### 总结

Trinity是一种通过轻量级协调器来协调多个大型语言模型协作的新方法，解决了基础模型合并时架构不匹配和API封闭的限制问题。

### 背景

结合多样化的基础模型是有前景的，但权重合并受到架构不匹配和封闭API的限制。

### 目的

开发一种能够协调多个大型语言模型(LLMs)协作的轻量级协调器，克服架构不匹配和API封闭的问题。

### 方法

Trinity包含一个紧凑语言模型(约0.6B参数)和一个轻量级头部(约10K参数)，使用进化策略进行优化，实现高效和自适应的任务分配。系统处理多轮查询，在每轮中协调器为选定的LLM分配三种角色之一(思考者、工作者或验证者)，有效将复杂技能获取从协调器本身转移出去。

### 主要发现

Trinity在编程、数学、推理和领域知识任务中持续优于单个模型和现有方法；能够稳健地推广到分布外任务；在标准基准测试中达到最先进的结果，包括在LiveCodeBench上获得86.2%的分数；性能优势的两个主要因素：(1)协调器的隐藏状态表示提供了丰富的输入上下文化；(2)在高维度和严格预算约束下，可分离协方差矩阵自适应进化策略比强化学习、模仿学习和随机搜索具有优势。

### 结论

Trinity通过轻量级协调器有效解决了基础模型合并的局限性，实现了多模型协作的优势，并在多个任务中展示了卓越的性能和泛化能力。

### 翻译

结合多样化的基础模型是有前景的，但权重合并受到架构不匹配和封闭API的限制。Trinity通过一个轻量级协调器解决了这一问题，该协调器协调大型语言模型(LLMs)之间的协作。该协调器由一个紧凑的语言模型(约0.6B参数)和一个轻量级头部(约10K参数)组成，使用进化策略进行优化，以实现高效和自适应的任务分配。Trinity处理多轮查询，在每轮中协调器为选定的LLM分配三种角色之一(思考者、工作者或验证者)，有效将复杂技能获取从协调器本身转移出去。实验表明，Trinity在编程、数学、推理和领域知识任务中持续优于单个模型和现有方法，并能稳健地推广到分布外任务。在标准基准测试中，Trinity取得了最先进的结果，包括在LiveCodeBench上获得86.2%的分数。理论和实证分析确定了这一性能背后的两个主要因素：(1)协调器的隐藏状态表示提供了丰富的输入上下文化；(2)在高维度和严格预算约束下，可分离协方差矩阵自适应进化策略通过利用潜在的块epsilon可分离性，比强化学习、模仿学习和随机搜索具有优势。


### 论文摘要

Combining diverse foundation models is promising, but weight-merging is limited by mismatched architectures and closed APIs. Trinity addresses this with a lightweight coordinator that orchestrates collaboration among large language models (LLMs). The coordinator, comprising a compact language model (approximately $0.6$B parameters) and a lightweight head (approximately $10$K parameters), is optimized with an evolutionary strategy for efficient and adaptive delegation. Trinity processes queries over multiple turns, where at each turn the coordinator assigns one of three roles (Thinker, Worker, or Verifier) to a selected LLM, effectively offloading complex skill acquisition from the coordinator itself. Experiments show that Trinity consistently outperforms individual models and existing methods across coding, math, reasoning, and domain knowledge tasks, and generalizes robustly to out-of-distribution tasks. On standard benchmarks, Trinity achieves state-of-the-art results, including a score of 86.2% on LiveCodeBench. Theoretical and empirical analyses identify two main factors behind this performance: (1) the coordinator's hidden-state representations provide rich contextualization of inputs, and (2) under high dimensionality and strict budget constraints, the separable Covariance Matrix Adaptation Evolution Strategy offers advantages over reinforcement learning, imitation learning, and random search by exploiting potential block-epsilon-separability.

---

## 35. Neural Decoding of Overt Speech from ECoG Using Vision Transformers and Contrastive Representation Learning

**论文链接:** [http://arxiv.org/abs/2512.04618v1](http://arxiv.org/abs/2512.04618v1)

**作者:** Mohamed Baha Ben Ticha, Xingchen Ran, Guillaume Saldanha, Gaël Le Godais, Philémon Roussel, Marc Aubert, Amina Fontanell, Thomas Costecalde, Lucas Struber, Serpil Karakas, Shaomin Zhang, Philippe Kahane, Guillaume Charvet, Stéphan Chabardès, Blaise Yvert

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文介绍了一种基于编码器-解码器深度神经架构的离线语音解码流程，整合了视觉变换器和对比学习技术，用于直接从皮层脑电图信号回归语音。

### 背景

语音脑机接口为严重瘫痪无法沟通的人群提供了有前景的解决方案。最近研究表明可通过预测音素或单词并使用语言模型重建可理解语音，但直接将皮层信号回归为语音声学特征的流式模式重建仍面临挑战，特别是在表面ECoG记录方面。

### 目的

开发一种能够直接从表面ECoG信号回归语音的解码方法，优化神经解码器以实现与皮层内数据相当的结果，并探索完全可植入无线硬膜外记录系统在语音解码中的应用。

### 方法

基于编码器-解码器深度神经网络架构，整合视觉变换器和对比学习技术，构建离线语音解码流程。使用两个数据集进行评估：一个来自癫痫患者的临床硬膜下电极记录，另一个来自运动BCI试验参与者的完全可植入WIMAGINE硬膜外系统记录。

### 主要发现

成功从表面ECoG信号直接回归语音，首次实现从完全可植入和无线硬膜外记录系统解码语音，为长期应用提供了可能性。

### 结论

通过优化神经解码器和整合先进的深度学习技术，可以在表面ECoG记录中实现有效的语音重建，为临床应用开辟了新途径。

### 翻译

语音脑机接口为严重瘫痪无法沟通的人提供了有前景的解决方案。最近的研究已经证明，通过预测一系列音素或单词并使用下游语言模型来获得有意义的句子，可以从皮层脑电图或皮层内记录中重建可理解的语音。当前挑战是通过直接将皮层信号回归为语音声学特征，以流式模式重建语音。虽然最近使用皮层内数据已实现这一目标，但还需要更多工作来在表面ECoG记录中获得 comparable 的结果。在这种情况下，优化神经解码器变得至关重要。我们在此提出了一种基于编码器-解码器深度神经架构的离线语音解码流程，整合了视觉变换器和对比学习，以增强从ECoG信号直接回归语音。该方法在两个数据集上进行了评估：一个来自癫痫患者使用临床硬膜下电极获得，另一个来自运动BCI试验参与者使用完全可植入的WIMAGINE硬膜外系统获得。据我们所知，这是首次尝试从完全可植入和无线硬膜外记录系统解码语音，为长期使用提供了前景。


### 论文摘要

Speech Brain Computer Interfaces (BCIs) offer promising solutions to people with severe paralysis unable to communicate. A number of recent studies have demonstrated convincing reconstruction of intelligible speech from surface electrocorticographic (ECoG) or intracortical recordings by predicting a series of phonemes or words and using downstream language models to obtain meaningful sentences. A current challenge is to reconstruct speech in a streaming mode by directly regressing cortical signals into acoustic speech. While this has been achieved recently using intracortical data, further work is needed to obtain comparable results with surface ECoG recordings. In particular, optimizing neural decoders becomes critical in this case. Here we present an offline speech decoding pipeline based on an encoder-decoder deep neural architecture, integrating Vision Transformers and contrastive learning to enhance the direct regression of speech from ECoG signals. The approach is evaluated on two datasets, one obtained with clinical subdural electrodes in an epileptic patient, and another obtained with the fully implantable WIMAGINE epidural system in a participant of a motor BCI trial. To our knowledge this presents a first attempt to decode speech from a fully implantable and wireless epidural recording system offering perspectives for long-term use.

---

## 36. TARDis: Time Attenuated Representation Disentanglement for Incomplete Multi-Modal Tumor Segmentation and Classification

**论文链接:** [http://arxiv.org/abs/2512.04576v1](http://arxiv.org/abs/2512.04576v1)

**作者:** Zishuo Wan, Qinqin Kang, Yi Huang, Yun Bian, Dawei Ding, Ke Yan

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种名为时间衰减表示解缠(TARDis)的新型物理感知框架，用于解决对比增强CT中缺失模态的问题。该方法通过将缺失阶段视为连续时间-衰减曲线上的缺失样本点，并解缠特征空间为静态解剖成分和动态灌注成分，显著提高了在有限扫描数据下的肿瘤分割和诊断性能。

### 背景

肿瘤分割和诊断在对比增强CT中高度依赖造影剂的生理动力学，但由于辐射暴露和扫描限制，获取完整的多阶段扫描序列在临床上往往不可行，导致'缺失模态'问题。

### 目的

开发一种能够有效处理不完整CT扫描数据的方法，在不增加辐射暴露的情况下保持肿瘤分割和诊断的准确性。

### 方法

提出TARDis框架，采用双路径架构：1)基于量化的路径使用可学习嵌入字典提取一致的解剖结构；2)基于概率的路径使用条件变分自编码器建模基于估计扫描时间的动态增强。该方法将缺失模态视为连续时间-衰减曲线上的缺失样本点，并解缠特征空间为静态解剖成分和时间依赖的动态灌注成分。

### 主要发现

在大型私人腹部CT数据集(2,282个病例)和两个公共数据集上的实验表明，TARDis显著优于现有最先进的不完整模态处理方法。即使在极端数据稀疏情况下，该方法也能保持稳健的诊断性能。

### 结论

TARDis框架有潜力在保持诊断精度的同时减少患者辐射暴露，为临床实践中更安全、更高效的肿瘤诊断提供了新思路。

### 翻译

本文提出了一种名为时间衰减表示解缠(TARDis)的新型物理感知框架，用于解决对比增强CT中缺失模态的问题。该方法通过将缺失阶段视为连续时间-衰减曲线上的缺失样本点，并解缠特征空间为静态解剖成分和动态灌注成分，显著提高了在有限扫描数据下的肿瘤分割和诊断性能。


### 论文摘要

Tumor segmentation and diagnosis in contrast-enhanced Computed Tomography (CT) rely heavily on the physiological dynamics of contrast agents. However, obtaining a complete multi-phase series is often clinically unfeasible due to radiation concerns or scanning limitations, leading to the "missing modality" problem. Existing deep learning approaches typically treat missing phases as absent independent channels, ignoring the inherent temporal continuity of hemodynamics. In this work, we propose Time Attenuated Representation Disentanglement (TARDis), a novel physics-aware framework that redefines missing modalities as missing sample points on a continuous Time-Attenuation Curve. TARDis explicitly disentangles the latent feature space into a time-invariant static component (anatomy) and a time-dependent dynamic component (perfusion). We achieve this via a dual-path architecture: a quantization-based path using a learnable embedding dictionary to extract consistent anatomical structures, and a probabilistic path using a Conditional Variational Autoencoder to model dynamic enhancement conditioned on the estimated scan time. This design allows the network to hallucinate missing hemodynamic features by sampling from the learned latent distribution. Extensive experiments on a large-scale private abdominal CT dataset (2,282 cases) and two public datasets demonstrate that TARDis significantly outperforms state-of-the-art incomplete modality frameworks. Notably, our method maintains robust diagnostic performance even in extreme data-sparsity scenarios, highlighting its potential for reducing radiation exposure while maintaining diagnostic precision.

---

## 37. Identity Clue Refinement and Enhancement for Visible-Infrared Person Re-Identification

**论文链接:** [http://arxiv.org/abs/2512.04522v1](http://arxiv.org/abs/2512.04522v1)

**作者:** Guoqing Zhang, Zhun Wang, Hairui Wang, Zhonglin Ye, Yuhui Zheng

**发布时间:** 2025-12-04

**备注:** 14 pages, 7 figures

### GPT解析

### 总结

该研究针对可见光-红外人员重识别中的模态差异问题，提出了一种新的身份线索精炼与增强网络，通过多感知特征精炼、语义蒸馏级联增强和身份线索引导损失三个组件，有效利用了模态特定属性中的判别知识，实验证明该方法优于现有最先进方法。

### 背景

可见光-红外人员重识别(VI-ReID)是一个具有挑战性的跨模态匹配任务，因为存在显著的模态差异。当前方法主要关注通过统一嵌入空间学习模态不变特征，但往往只关注跨模态间的共同判别语义，而忽略了模态特定身份感知知识在判别特征学习中的关键作用。

### 目的

为了弥补现有方法的不足，提出一种能够挖掘和利用模态特定属性中隐含判别知识的新方法，以提高跨模态人员重识别的准确性。

### 方法

提出了一种身份线索精炼与增强(ICRE)网络，包含三个主要组件：1)多感知特征精炼(MPFR)模块，聚合共享分支的浅层特征以捕捉模态特定属性；2)语义蒸馏级联增强(SDCE)模块，从聚合特征中蒸馏身份感知知识并引导模态不变特征学习；3)身份线索引导(ICG)损失，减轻模态差异并促进多样化表示空间学习。

### 主要发现

在多个公共数据集上的大量实验表明，所提出的ICRE方法显著优于现有的最先进(SOTA)方法，证明了利用模态特定属性中判别知识的有效性。

### 结论

通过有效挖掘和利用模态特定属性中的判别知识，ICRE网络成功解决了VI-ReID任务中的模态差异问题，提高了跨模态人员重识别的性能。

### 翻译

可见光-红外人员重识别(VI-ReID)是一个具有挑战性的跨模态匹配任务，因为存在显著的模态差异。虽然当前方法主要关注通过统一嵌入空间学习模态不变特征，但它们往往只关注跨模态间的共同判别语义，而忽略了模态特定身份感知知识在判别特征学习中的关键作用。为了弥补这一差距，我们提出了一种新颖的身份线索精炼与增强(ICRE)网络，用于挖掘和利用模态特定属性中固有的隐含判别知识。首先，我们设计了一个多感知特征精炼(MPFR)模块，聚合共享分支的浅层特征，旨在捕捉容易被忽略的模态特定属性。然后，我们提出了一个语义蒸馏级联增强(SDCE)模块，从聚合的浅层特征中蒸馏身份感知知识，并引导模态不变特征的学习。最后，提出了一个身份线索引导(ICG)损失，以减轻增强特征内的模态差异，促进多样化表示空间的学习。在多个公共数据集上的大量实验清楚地表明，我们提出的ICRE方法优于现有的最先进方法。


### 论文摘要

Visible-Infrared Person Re-Identification (VI-ReID) is a challenging cross-modal matching task due to significant modality discrepancies. While current methods mainly focus on learning modality-invariant features through unified embedding spaces, they often focus solely on the common discriminative semantics across modalities while disregarding the critical role of modality-specific identity-aware knowledge in discriminative feature learning. To bridge this gap, we propose a novel Identity Clue Refinement and Enhancement (ICRE) network to mine and utilize the implicit discriminative knowledge inherent in modality-specific attributes. Initially, we design a Multi-Perception Feature Refinement (MPFR) module that aggregates shallow features from shared branches, aiming to capture modality-specific attributes that are easily overlooked. Then, we propose a Semantic Distillation Cascade Enhancement (SDCE) module, which distills identity-aware knowledge from the aggregated shallow features and guide the learning of modality-invariant features. Finally, an Identity Clues Guided (ICG) Loss is proposed to alleviate the modality discrepancies within the enhanced features and promote the learning of a diverse representation space. Extensive experiments across multiple public datasets clearly show that our proposed ICRE outperforms existing SOTA methods.

---

## 38. Boundary-Aware Test-Time Adaptation for Zero-Shot Medical Image Segmentation

**论文链接:** [http://arxiv.org/abs/2512.04520v1](http://arxiv.org/abs/2512.04520v1)

**作者:** Chenlin Xu, Lei Zhang, Lituan Wang, Xinyu Pu, Pengfei Ma, Guangwu Qian, Zizhou Wang, Yan Wang

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出BA-TTA-SAM框架，通过测试时适应显著增强SAM在医学图像分割中的零样本性能，无需源域训练数据。

### 背景

医学图像分割面临标注数据稀缺和模型计算成本高的挑战，传统调优方法和当前预训练模型适应方法仍严重依赖下游任务上的特定训练，而SAM在医学数据集上因领域迁移问题存在明显局限。

### 目的

解决SAM在医学数据集上的领域迁移问题，实现高效的零样本分割增强。

### 方法

提出BA-TTA-SAM，一个与任务无关的测试时适应框架，整合两个关键机制：(1)编码器级高斯提示注入，将基于高斯的提示直接嵌入图像编码器；(2)跨层边界感知注意力对齐，利用ViT骨干网络中的分层特征交互，将深层语义响应与浅层边界线索对齐。

### 主要发现

在ISIC、Kvasir、BUSI和REFUGE四个数据集上的实验表明，与SAM的零样本分割性能相比，DICE分数平均提高了12.4%，该方法在医学图像分割中始终优于最先进的模型。

### 结论

该框架显著增强了SAM的泛化能力，无需任何源域训练数据，公开可用医学数据集上的大量实验充分证明了该框架的优越性。

### 翻译

由于标注数据稀缺和模型计算成本高，医学图像分割中的传统调优方法面临严峻挑战。当前适应预训练模型的方法，包括全参数和参数高效微调，仍然严重依赖下游任务上的特定训练。因此，零样本分割日益受到关注，特别是像SAM这样的基础模型展现出有前景的泛化能力。然而，SAM在医学数据集上仍因领域迁移问题而面临显著局限，使得高效的零样本增强成为紧迫的研究目标。为解决这些挑战，我们提出了BA-TTA-SAM，一个与任务无关的测试时适应框架，通过测试时适应显著增强了SAM的零样本分割性能。该框架整合了两个关键机制：(1)编码器级高斯提示注入，将基于高斯的提示直接嵌入图像编码器，为初始表示学习提供明确指导。(2)跨层边界感知注意力对齐，利用ViT骨干网络中的分层特征交互，将深层语义响应与浅层边界线索对齐。在ISIC、Kvasir、BUSI和REFUGE四个数据集上的实验表明，与SAM的零样本分割性能相比，DICE分数平均提高了12.4%。结果证明，我们的方法在医学图像分割中始终优于最先进的模型。我们的框架显著增强了SAM的泛化能力，无需任何源域训练数据。公开可用医学数据集上的大量实验充分证明了我们框架的优越性。我们的代码可在https://github.com/Emilychenlin/BA-TTA-SAM获取。


### 论文摘要

Due to the scarcity of annotated data and the substantial computational costs of model, conventional tuning methods in medical image segmentation face critical challenges. Current approaches to adapting pretrained models, including full-parameter and parameter-efficient fine-tuning, still rely heavily on task-specific training on downstream tasks. Therefore, zero-shot segmentation has gained increasing attention, especially with foundation models such as SAM demonstrating promising generalization capabilities. However, SAM still faces notable limitations on medical datasets due to domain shifts, making efficient zero-shot enhancement an urgent research goal. To address these challenges, we propose BA-TTA-SAM, a task-agnostic test-time adaptation framework that significantly enhances the zero-shot segmentation performance of SAM via test-time adaptation. This framework integrates two key mechanisms: (1) The encoder-level Gaussian prompt injection embeds Gaussian-based prompts directly into the image encoder, providing explicit guidance for initial representation learning. (2) The cross-layer boundary-aware attention alignment exploits the hierarchical feature interactions within the ViT backbone, aligning deep semantic responses with shallow boundary cues. Experiments on four datasets, including ISIC, Kvasir, BUSI, and REFUGE, show an average improvement of 12.4\% in the DICE score compared with SAM's zero-shot segmentation performance. The results demonstrate that our method consistently outperforms state-of-the-art models in medical image segmentation. Our framework significantly enhances the generalization ability of SAM, without requiring any source-domain training data. Extensive experiments on publicly available medical datasets strongly demonstrate the superiority of our framework. Our code is available at https://github.com/Emilychenlin/BA-TTA-SAM.

---

## 39. BiTAgent: A Task-Aware Modular Framework for Bidirectional Coupling between Multimodal Large Language Models and World Models

**论文链接:** [http://arxiv.org/abs/2512.04513v1](http://arxiv.org/abs/2512.04513v1)

**作者:** Yu-Wei Zhan, Xin Wang, Pengzhe Mao, Tongtong Feng, Ren Wang, Wenwu Zhu

**发布时间:** 2025-12-04

### GPT解析

### 总结

该论文提出了BiTAgent框架，通过多模态大语言模型(MLLMs)与世界模型(WMs)的双向耦合，解决了构建通用具身智能体的关键挑战，实现了语义推理和动态预测的和谐统一。

### 背景

构建通用具身智能体需要能够解释多模态目标、建模环境动态并在多样化现实世界任务中执行可靠动作的统一系统。MLLMs提供语义先验和跨模态泛化，WMs提供可操作的潜在动态，但它们的结合面临两个关键挑战。

### 目的

解决MLLMs和WMs结合时的两个关键挑战：建立语义意图与动态状态表示的紧密耦合，以及实现支持多任务学习和跨环境泛化的任务感知适应性。

### 方法

提出BiTAgent框架，实现MLLMs和WMs之间的双向耦合。建立前向路径(将MLLM表示注入WM潜在空间)和后向路径(通过WM生成的反馈 refine MLLMs语义空间)，通过任务感知动态联合学习、任务感知行为学习和MLLM-WM联合优化三个协同组件实现。

### 主要发现

在多任务和跨环境设置下的实验表明，BiTAgent比最先进基线方法具有更好的稳定性和泛化能力，标志着向开放式具身学习迈进了一步。

### 结论

BiTAgent成功解决了MLLMs和WMs结合时的关键挑战，通过双向耦合机制实现了语义推理和动态预测的和谐统一，为构建通用具身智能体提供了有效方法。

### 翻译

构建通用具身智能体需要一个统一系统，能够解释多模态目标、建模环境动态并在多样化的现实世界任务中执行可靠动作。多模态大语言模型(MLLMs)提供强大的语义先验和跨模态泛化能力，而世界模型(WMs)提供可操作的潜在动态用于预测和控制。它们的组合为开放式具身智能提供了前景，但也带来了两个关键挑战：(1)建立MLLMs的语义意图与WM潜在空间中的动态状态表示之间的紧密耦合；(2)实现任务感知的适应性，支持多任务学习和跨环境泛化。为了解决这些局限性，我们提出了BiTAgent，一个任务感知的动态联合框架，实现了MLLMs和WMs之间的双向耦合。BiTAgent建立了两条互补路径：前向路径将MLLM表示注入WM的潜在空间，用于语义引导的想象；后向路径让WM生成的反馈通过密集文本条件奖励来 refine MLLMs的语义空间。这种双向交互通过三个协同组件实现：任务感知动态联合学习、任务感知行为学习和MLLM-WM联合优化，它们共同协调语义推理和动态预测。在多任务和跨环境设置下的广泛实验表明，BiTAgent比最先进的基线方法具有更好的稳定性和泛化能力，标志着向开放式具身学习迈进了一步。


### 论文摘要

Building generalist embodied agents requires a unified system that can interpret multimodal goals, model environment dynamics, and execute reliable actions across diverse real-world tasks. Multimodal large language models (MLLMs) offer strong semantic priors and cross-modal generalization, while world models (WMs) provide actionable latent dynamics for prediction and control. Their combination holds promise for open-ended embodied intelligence, yet introduces two key challenges: (1) establishing a tight coupling between the semantic intent from MLLMs and the dynamic state representations within the WM's latent space, and (2) achieving task-aware adaptability that supports multi-task learning and cross-environment generalization. To address these limitations, we propose BiTAgent, a task-aware dynamic joint framework that enables bidirectional coupling between MLLMs and WMs. BiTAgent establishes two complementary pathways: a forward path that injects MLLM representations into the WM's latent space for semantically guided imagination, and a backward path where WM-generated feedback refines the MLLM's semantic space via dense text-conditioned rewards. This bidirectional interaction is realized through three synergistic components: Task-Aware Dynamic Joint Learning, Task-Aware Behavior Learning, and MLLM-WM Joint Optimization, which together harmonize semantic reasoning and dynamic prediction. Extensive experiments across multi-task and cross-environment settings demonstrate superior stability and generalization over state-of-the-art baselines, marking a step toward open-ended embodied learning.

---

## 40. DeRA: Decoupled Representation Alignment for Video Tokenization

**论文链接:** [http://arxiv.org/abs/2512.04483v1](http://arxiv.org/abs/2512.04483v1)

**作者:** Pengbo Guo, Junke Wang, Zhen Xing, Chengxu Liu, Daoguo Dong, Xueming Qian, Zuxuan Wu

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出DeRA，一种新颖的一维视频标记器，通过解耦空间-时间表征学习提高视频处理的效率和性能。

### 背景

现有视频标记方法在处理视频内容时面临训练效率不高和性能有限的问题，特别是在同时捕捉空间语义和时间动态方面。

### 目的

开发一种更高效的视频标记方法，能够分别捕获视频的空间语义和时间动态特性，同时解决异质监督带来的梯度冲突问题。

### 方法

DeRA保持紧凑的一维潜在空间，将视频编码分解为外观和运动流，分别与预训练的视觉基础模型对齐，以捕获空间语义和动态特性。同时提出对称对齐-冲突投影(SACP)模块，通过抑制冲突方向上的梯度成分来解决异质监督带来的梯度冲突问题。

### 主要发现

1. DeRA在UCF-101数据集上以rFVD指标相比之前最先进的视频标记器LARP提高了25%；2. 使用DeRA进行自回归视频生成，在UCF-101类条件生成和K600帧预测任务上达到新的最先进结果。

### 结论

DeRA通过解耦空间-时间表征学习和有效解决梯度冲突问题，显著提升了视频标记和生成的性能，为视频处理领域提供了新的有效方法。

### 翻译

本文提出DeRA，一种新颖的一维视频标记器，它在视频标记化过程中解耦空间-时间表征学习，以实现更好的训练效率和性能。具体来说，DeRA保持紧凑的一维潜在空间，同时将视频编码分解为外观和运动流，这些流与预训练的视觉基础模型对齐，分别捕获视频中的空间语义和动态特性。为了解决异质监督引入的梯度冲突，作者提出了对称对齐-冲突投影(SACP)模块，通过抑制冲突方向上的成分来主动重构梯度。大量实验表明，DeRA在UCF-101数据集上以rFVD指标相比之前最先进的视频标记器LARP提高了25%。此外，使用DeRA进行自回归视频生成，在UCF-101类条件生成和K600帧预测任务上也达到了新的最先进结果。


### 论文摘要

This paper presents DeRA, a novel 1D video tokenizer that decouples the spatial-temporal representation learning in video tokenization to achieve better training efficiency and performance. Specifically, DeRA maintains a compact 1D latent space while factorizing video encoding into appearance and motion streams, which are aligned with pretrained vision foundation models to capture the spatial semantics and temporal dynamics in videos separately. To address the gradient conflicts introduced by the heterogeneous supervision, we further propose the Symmetric Alignment-Conflict Projection (SACP) module that proactively reformulates gradients by suppressing the components along conflicting directions. Extensive experiments demonstrate that DeRA outperforms LARP, the previous state-of-the-art video tokenizer by 25% on UCF-101 in terms of rFVD. Moreover, using DeRA for autoregressive video generation, we also achieve new state-of-the-art results on both UCF-101 class-conditional generation and K600 frame prediction.

---

## 41. Explainable Parkinsons Disease Gait Recognition Using Multimodal RGB-D Fusion and Large Language Models

**论文链接:** [http://arxiv.org/abs/2512.04425v1](http://arxiv.org/abs/2512.04425v1)

**作者:** Manar Alnaasan, Md Selim Sarowar, Sungho Kim

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种可解释的多模态框架，整合RGB和深度数据用于帕金森病步态识别，结合双YOLOv11编码器、多尺度特征提取和跨空间融合机制，并使用大型语言模型提供临床解释。实验表明该方法在准确性和鲁棒性上优于单输入基线方法。

### 背景

准确且可解释的步态分析在帕金森病早期检测中起关键作用，但现有方法受限于单模态输入、低鲁棒性和临床透明度不足。

### 目的

开发一个整合RGB和深度数据的可解释多模态框架，用于在现实条件下识别帕金森病步态模式并提供临床解释。

### 方法

使用双YOLOv11编码器进行模态特定特征提取，采用多尺度局部-全局提取模块和跨空间颈部融合机制增强时空表示，并集成冻结的大型语言模型将视觉特征转换为临床文本解释。

### 主要发现

提出的RGB-D融合框架实现了更高的识别准确率，对环境变化具有更强的鲁棒性，并提供清晰的视觉-语言推理能力。

### 结论

通过结合多模态特征学习和基于语言的可解释性，该研究弥合了视觉识别与临床理解之间的差距，为帕金森病步态分析提供了一种新颖的视觉语言范式。

### 翻译

准确且可解释的步态分析在帕金森病的早期检测中起着至关重要的作用，然而大多数现有方法仍受限于单模态输入、低鲁棒性和临床透明度不足。本文提出了一种可解释的多模态框架，整合RGB和深度数据，以在现实条件下识别帕金森病步态模式。所提出的系统采用基于双YOLOv11的编码器进行模态特定特征提取，随后是多尺度局部-全局提取模块和跨空间颈部融合机制，以增强时空表示。


### 论文摘要

Accurate and interpretable gait analysis plays a crucial role in the early detection of Parkinsons disease (PD),yet most existing approaches remain limited by single-modality inputs, low robustness, and a lack of clinical transparency. This paper presents an explainable multimodal framework that integrates RGB and Depth (RGB-D) data to recognize Parkinsonian gait patterns under realistic conditions. The proposed system employs dual YOLOv11-based encoders for modality-specific feature extraction, followed by a Multi-Scale Local-Global Extraction (MLGE) module and a Cross-Spatial Neck Fusion mechanism to enhance spatial-temporal representation. This design captures both fine-grained limb motion (e.g., reduced arm swing) and overall gait dynamics (e.g., short stride or turning difficulty), even in challenging scenarios such as low lighting or occlusion caused by clothing. To ensure interpretability, a frozen Large Language Model (LLM) is incorporated to translate fused visual embeddings and structured metadata into clinically meaningful textual explanations. Experimental evaluations on multimodal gait datasets demonstrate that the proposed RGB-D fusion framework achieves higher recognition accuracy, improved robustness to environmental variations, and clear visual-linguistic reasoning compared with single-input baselines. By combining multimodal feature learning with language-based interpretability, this study bridges the gap between visual recognition and clinical understanding, offering a novel vision-language paradigm for reliable and explainable Parkinsons disease gait analysis. Code:https://github.com/manaralnaasan/RGB-D_parkinson-LLM

---

## 42. 4DLangVGGT: 4D Language-Visual Geometry Grounded Transformer

**论文链接:** [http://arxiv.org/abs/2512.05060v1](http://arxiv.org/abs/2512.05060v1)

**作者:** Xianfeng Wu, Yajing Bai, Minghan Li, Xianzu Wu, Xueqi Zhao, Zhongyuan Lai, Wenyu Liu, Xinggang Wang

**发布时间:** 2025-12-04

**备注:** Code: https://github.com/hustvl/4DLangVGGT, Webpage: https://hustvl.github.io/4DLangVGGT

### GPT解析

### 总结

本文提出了4DLangVGGT，第一个基于Transformer的前向统一框架，用于4D语言定位，解决了现有方法需要每场景优化、泛化能力有限和难以扩展的问题。

### 背景

构建4D语言场对具身AI、增强/虚拟现实和4D场景理解至关重要，但现有方法主要依赖场景特定的高斯飞溅技术，需要每场景优化，泛化能力有限且难以扩展到实际应用。

### 目的

解决现有4D语义场构建方法的局限性，提出一个不需要每场景优化的统一框架，提高泛化能力和部署效率。

### 方法

提出4DLangVGGT，包含两个关键组件：1) 4D视觉几何变换器StreamVGGT，用于捕获动态场景的时空几何表示；2) 语义桥接解码器(SBD)，将几何感知特征投影到语言对齐的语义空间。该方法可跨多个动态场景联合训练，推理时直接应用。

### 主要发现

在HyperNeRF和Neu3D数据集上的实验表明，该方法能有效泛化并达到最先进性能，在单场景训练下提高2%，在多场景训练下提高1%。

### 结论

4DLangVGGT显著提高了大规模部署的实用性，为开放词汇的4D场景理解建立了新范式。

### 翻译

构建4D语言场对具身AI、增强/虚拟现实和4D场景理解至关重要，因为它们为动态环境提供了丰富的语义表示，并在复杂场景中实现开放词汇查询。然而，现有的4D语义场构建方法主要依赖于场景特定的高斯飞溅技术，这需要每场景优化，表现出有限的泛化能力，并且难以扩展到实际应用。为解决这些限制，我们提出了4DLangVGGT，这是第一个基于Transformer的前向统一框架，用于4D语言定位，它在单一架构中集成了几何感知和语言对齐。4DLangVGGT有两个关键组件：4D视觉几何变换器StreamVGGT，它捕获动态场景的时空几何表示；以及语义桥接解码器(SBD)，它将几何感知特征投影到语言对齐的语义空间，从而增强语义可解释性同时保持结构保真度。与依赖昂贵每场景优化的先前方法不同，4DLangVGGT可以跨多个动态场景联合训练并在推理时直接应用，实现部署效率和强泛化性。这种设计显著提高了大规模部署的实用性，并为开放词汇的4D场景理解建立了新范式。在HyperNeRF和Neu3D数据集上的实验表明，我们的方法不仅能有效泛化，而且达到了最先进的性能，在每场景训练下提高2%，在多场景训练下提高1%。我们的代码已在https://github.com/hustvl/4DLangVGGT发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决4D语言场构建中的效率与泛化问题。现有方法依赖场景特定的高斯泼溅技术，需要每个场景单独优化，计算成本高且难以扩展。这个问题很重要，因为4D语言场对具身人工智能、增强/虚拟现实和4D场景理解至关重要，能提供动态环境的丰富语义表示，支持复杂场景中的开放词汇查询。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析现有高斯泼溅方法的局限性，然后转向前馈4D几何重建范式如StreamVGGT，但发现这些方法只关注几何而缺乏语义对齐。因此，作者设计了一个统一框架结合几何重建和语言对齐。借鉴了StreamVGGT作为几何编码器基础，VGGT作为架构基础，CLIP用于语义对齐，以及多模态大语言模型处理时间敏感语义监督。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个统一的Transformer框架，将动态几何重建与视觉语言对齐结合，使用语义桥接解码器连接几何感知和语义空间。整体流程：1)基于StreamVGGT的几何编码器捕捉时空几何表示；2)语义桥接解码器将几何标记转换为上下文特征，再通过双头解码器投影到语义和视觉子空间；3)多目标训练策略结合语义监督(时间不敏感和时间敏感)和重建监督，联合优化语义对齐和视觉保真度。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出首个基于Transformer的前馈统一框架4DLangVGGT用于4D语言接地；2)引入语义桥接解码器(SBD)映射几何特征到语言对齐语义空间；3)实现跨多个动态场景联合训练和直接应用，无需场景特定优化。相比之前工作，不依赖高斯泼溅技术，避免了每个场景优化，提高了部署效率和泛化能力，在保持结构保真度的同时增强了语义可解释性，在多个数据集上实现最先进性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '4DLangVGGT通过统一的Transformer框架将动态几何重建与视觉语言对齐相结合，实现了高效的4D语义场构建，无需场景特定优化，显著提高了动态场景理解的实用性和泛化能力。'}


### 论文摘要

Constructing 4D language fields is crucial for embodied AI, augmented/virtual reality, and 4D scene understanding, as they provide enriched semantic representations of dynamic environments and enable open-vocabulary querying in complex scenarios. However, existing approaches to 4D semantic field construction primarily rely on scene-specific Gaussian splatting, which requires per-scene optimization, exhibits limited generalization, and is difficult to scale to real-world applications. To address these limitations, we propose 4DLangVGGT, the first Transformer-based feed-forward unified framework for 4D language grounding, that jointly integrates geometric perception and language alignment within a single architecture. 4DLangVGGT has two key components: the 4D Visual Geometry Transformer, StreamVGGT, which captures spatio-temporal geometric representations of dynamic scenes; and the Semantic Bridging Decoder (SBD), which projects geometry-aware features into a language-aligned semantic space, thereby enhancing semantic interpretability while preserving structural fidelity. Unlike prior methods that depend on costly per-scene optimization, 4DLangVGGT can be jointly trained across multiple dynamic scenes and directly applied during inference, achieving both deployment efficiency and strong generalization. This design significantly improves the practicality of large-scale deployment and establishes a new paradigm for open-vocabulary 4D scene understanding. Experiments on HyperNeRF and Neu3D datasets demonstrate that our approach not only generalizes effectively but also achieves state-of-the-art performance, achieving up to 2% gains under per-scene training and 1% improvements under multi-scene training. Our code released in https://github.com/hustvl/4DLangVGGT

---

## 43. GeoPE:A Unified Geometric Positional Embedding for Structured Tensors

**论文链接:** [http://arxiv.org/abs/2512.04963v1](http://arxiv.org/abs/2512.04963v1)

**作者:** Yupu Yao, Bowen Yang

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究提出了一种名为几何位置编码(GeoPE)的新框架，通过将旋转扩展到使用四元数的3D欧几里得空间，解决了标准视觉Transformer将2D图像展平为1D序列时破坏自然空间拓扑的问题。GeoPE通过在李代数中计算几何均值来构建统一的旋转算子，克服了非交换性问题并确保了对称性，创建了能有效分离空间维度的几何耦合编码。实验表明，GeoPE在图像分类、目标检测和3D语义分割任务中始终胜过现有的2D RoPE变体，并显著增强了形状偏差，证实了其捕捉真实几何结构的能力。

### 背景

标准视觉Transformer将2D图像展平为1D序列，破坏了自然的空间拓扑。虽然旋转位置编码(RoPE)在1D序列中表现出色，但它继承了这一局限性，经常将空间上距离较远的块(例如行边缘的块)视为序列邻居。现有的2D方法通常独立处理空间轴，无法将这种错误的序列邻近性与真实空间距离解耦。

### 目的

为了恢复2D空间流形，本研究引入了几何位置编码(GeoPE)框架，旨在解决视觉Transformer中2D图像被展平为1D序列导致的空间拓扑破坏问题，以及现有2D RoPE变体无法正确处理空间距离的问题。

### 方法

GeoPE是一种通过四元数将旋转扩展到3D欧几里得空间的框架。为了克服非交换性问题并确保对称性，GeoPE通过在李代数中计算几何均值来构建统一的旋转算子，创建了能有效分离空间维度的几何耦合编码。

### 主要发现

在图像分类、目标检测和3D语义分割的广泛实验表明，GeoPE始终胜过现有的2D RoPE变体，并显著增强了形状偏差，证实了其捕捉真实几何结构的能力。

### 结论

GeoPE框架成功地恢复了2D空间流形，通过几何耦合编码有效分离了空间维度，并在多种视觉任务中表现出优越性能，证明了其在捕捉真实几何结构方面的有效性。

### 翻译

标准视觉Transformer将2D图像展平为1D序列，破坏了自然的空间拓扑。虽然旋转位置编码(RoPE)在1D序列中表现出色，但它继承了这一局限性，经常将空间上距离较远的块(例如行边缘的块)视为序列邻居。现有的2D方法通常独立处理空间轴，无法将这种错误的序列邻近性与真实空间距离解耦。为了恢复2D空间流形，我们引入了几何位置编码(GeoPE)，这是一个通过四元数将旋转扩展到3D欧几里得空间的框架。为了克服非交换性问题并确保对称性，GeoPE通过在李代数中计算几何均值来构建统一的旋转算子。这创建了能有效分离空间维度的几何耦合编码。在图像分类、目标检测和3D语义分割的广泛实验表明，GeoPE始终胜过现有的2D RoPE变体，并显著增强了形状偏差，证实了其捕捉真实几何结构的能力。


### 论文摘要

Standard Vision Transformers flatten 2D images into 1D sequences, disrupting the natural spatial topology. While Rotary Positional Embedding (RoPE) excels in 1D, it inherits this limitation, often treating spatially distant patches (e.g., at row edges) as sequence neighbors. Existing 2D approaches typically treat spatial axes independently, failing to decouple this false sequential proximity from true spatial distance. To restore the 2D spatial manifold, we introduce Geometric Positional Embedding (GeoPE), a framework that extends rotations to 3D Euclidean space using quaternions. To overcome non-commutativity and ensure symmetry, GeoPE constructs a unified rotational operator by computing the geometric mean in the Lie algebra. This creates a geometrically coupled encoding that effectively separates spatial dimensions. Extensive experiments on image classification, object detection, and 3D semantic segmentation demonstrate that GeoPE consistently outperforms existing 2D RoPE variants and significantly enhances shape bias, confirming its ability to capture true geometric structure.

---

## 44. Hybrid Quantum-Classical Autoencoders for Unsupervised Network Intrusion Detection

**论文链接:** [http://arxiv.org/abs/2512.05069v1](http://arxiv.org/abs/2512.05069v1)

**作者:** Mohammad Arif Rasyidi, Omar Alhussein, Sami Muhaidat, Ernesto Damiani

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究首次对量子-经典混合（HQC）自动编码器进行大规模评估，用于无监督异常入侵检测任务。通过统一的实验框架，研究团队迭代了关键的量子设计选择，并在三个基准NIDS数据集上进行了实验。结果表明，最佳配置下的HQC自动编码器可以匹配或超越经典性能，且对架构决策更敏感。在零日评估中，配置良好的HQC模型展现出比经典和监督基线更强且更稳定的泛化能力。然而，模拟门噪声实验显示早期性能下降，表明需要噪声感知的HQC设计。

### 背景

无监督异常入侵检测需要能够泛化到训练过程中未观察到的攻击模式的模型。传统方法在面对新型攻击时可能表现不佳，因此需要探索新的技术方案。

### 目的

评估量子-经典混合（HQC）自动编码器在无监督异常入侵检测任务中的性能，并研究其设计选择对模型表现的影响。

### 方法

构建统一的实验框架，迭代关键量子设计选择（量子层放置、测量方法、变分和非变分公式、潜在空间正则化），并在三个基准NIDS数据集上进行实验。同时进行了模拟门噪声实验以评估量子噪声对模型性能的影响。

### 主要发现

1) 最佳配置下，HQC自动编码器可以匹配或超越经典性能；2) HQC模型对架构决策表现出更高的敏感性；3) 在零日评估中，配置良好的HQC模型提供更强且更稳定的泛化能力；4) 模拟门噪声实验显示早期性能下降，表明需要噪声感知的HQC设计。

### 结论

这些结果提供了HQC自动编码器用于网络入侵检测的第一个数据驱动特性描述，并概述了影响其实际可行性的关键因素。所有实验代码和配置已在GitHub上公开。

### 翻译

无监督异常入侵检测需要能够泛化到训练过程中未观察到的攻击模式的模型。本研究首次对量子-经典混合（HQC）自动编码器进行此任务的大规模评估。我们构建了一个统一的实验框架，迭代关键量子设计选择，包括量子层放置、测量方法、变分和非变分公式以及潜在空间正则化。在三个基准NIDS数据集上的实验表明，HQC自动编码器在其最佳配置下可以匹配或超越经典性能，尽管它们对架构决策表现出更高的敏感性。在零日评估下，配置良好的HQC模型比经典和监督基线提供更强且更稳定的泛化能力。模拟门噪声实验显示早期性能下降，表明需要噪声感知的HQC设计。这些结果提供了HQC自动编码器用于网络入侵检测的第一个数据驱动特性描述，并概述了影响其实际可行性的关键因素。所有实验代码和配置可在https://github.com/arasyi/hqcae-network-intrusion-detection获取。


### 论文摘要

Unsupervised anomaly-based intrusion detection requires models that can generalize to attack patterns not observed during training. This work presents the first large-scale evaluation of hybrid quantum-classical (HQC) autoencoders for this task. We construct a unified experimental framework that iterates over key quantum design choices, including quantum-layer placement, measurement approach, variational and non-variational formulations, and latent-space regularization. Experiments across three benchmark NIDS datasets show that HQC autoencoders can match or exceed classical performance in their best configurations, although they exhibit higher sensitivity to architectural decisions. Under zero-day evaluation, well-configured HQC models provide stronger and more stable generalization than classical and supervised baselines. Simulated gate-noise experiments reveal early performance degradation, indicating the need for noise-aware HQC designs. These results provide the first data-driven characterization of HQC autoencoder behavior for network intrusion detection and outline key factors that govern their practical viability. All experiment code and configurations are available at https://github.com/arasyi/hqcae-network-intrusion-detection.

---

## 45. Meta-Learning for Quantum Optimization via Quantum Sequence Model

**论文链接:** [http://arxiv.org/abs/2512.05058v1](http://arxiv.org/abs/2512.05058v1)

**作者:** Yu-Cheng Lin, Yu-Chao Hsu, Samuel Yen-Chi Chen

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了一种量子元学习框架，使用量子序列模型优化QAOA参数初始化，显著提高了算法性能。

### 背景

量子近似优化算法(QAOA)是用于近期量子处理器解决组合优化问题的领先方法，但寻找好的变分参数存在挑战，导致收敛慢和解决方案质量差。

### 目的

开发一种方法来改进QAOA的参数初始化，提高收敛速度和解决方案质量。

### 方法

提出量子元学习框架，训练量子序列模型(包括QK-LSTM)作为学习优化器，在'学习学习'范式中生成有效的参数初始化策略。

### 主要发现

QK-LSTM优化器在Max-Cut问题上表现最佳，获得最高近似比和最快收敛速度；实现了完美的参数可转移性，即使推广到更大问题也能保持加速；仅用43个参数就优于经典LSTM和其他量子序列模型。

### 结论

QK-LSTM为NISQ时代变分量子算法的高效参数初始化提供了稳健途径，其紧凑性和表现力使其成为有前途的解决方案。

### 翻译

量子近似优化算法(QAOA)是在近期量子处理器上解决组合优化问题的领先方法。然而，由于能量景观的非凸性，寻找好的变分参数仍然是一个重大挑战，通常导致收敛速度慢和解决方案质量差。在这项工作中，我们提出了一种量子元学习框架，该框架训练先进的量子序列模型来生成有效的参数初始化策略。我们研究了四种经典或量子序列模型，包括基于量子的长短期记忆(QK-LSTM)，作为'学习学习'范式中的学习优化器。我们在Max-Cut问题上的数值实验表明，QK-LSTM优化器实现了卓越的性能，在所有测试的问题规模(n=10到13)上获得了最高的近似比和最快的收敛速度。重要的是，QK-LSTM模型通过合成一组单一的、固定的近优参数，实现了完美的参数可转移性，即使在推广到更大问题时也能带来显著的持续收敛加速。这种能力由量子内核架构的紧凑性和表现力所实现，突显了其有效性。QK-LSTM只有43个可训练参数，其性能明显优于经典LSTM(56个参数)和其他量子序列模型，为NISQ时代变分量子算法的高效参数初始化建立了一条稳健的途径。


### 论文摘要

The Quantum Approximate Optimization Algorithm (QAOA) is a leading approach for solving combinatorial optimization problems on near-term quantum processors. However, finding good variational parameters remains a significant challenge due to the non-convex energy landscape, often resulting in slow convergence and poor solution quality. In this work, we propose a quantum meta-learning framework that trains advanced quantum sequence models to generate effective parameter initialization policies. We investigate four classical or quantum sequence models, including the Quantum Kernel-based Long Short-Term Memory (QK-LSTM), as learned optimizers in a "learning to learn" paradigm. Our numerical experiments on the Max-Cut problem demonstrate that the QK-LSTM optimizer achieves superior performance, obtaining the highest approximation ratios and exhibiting the fastest convergence rate across all tested problem sizes (n=10 to 13). Crucially, the QK-LSTM model achieves perfect parameter transferability by synthesizing a single, fixed set of near-optimal parameters, leading to a remarkable sustained acceleration of convergence even when generalizing to larger problems. This capability, enabled by the compact and expressive power of the quantum kernel architecture, underscores its effectiveness. The QK-LSTM, with only 43 trainable parameters, substantially outperforms the classical LSTM (56 parameters) and other quantum sequence models, establishing a robust pathway toward highly efficient parameter initialization for variational quantum algorithms in the NISQ era.

---

## 46. Hybrid-Diffusion Models: Combining Open-loop Routines with Visuomotor Diffusion Policies

**论文链接:** [http://arxiv.org/abs/2512.04960v1](http://arxiv.org/abs/2512.04960v1)

**作者:** Jonne Van Haastregt, Bastian Orthmann, Michael C. Welle, Yuchong Zhang, Danica Kragic

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种结合开环程序和视觉运动扩散政策的混合扩散模型，通过远程操作增强基元(TAPs)提高操作任务的精度和速度。

### 背景

基于视觉运动的政策通过模仿学习在复杂操作任务中表现出色，但通常难以达到与传统控制方法相同的精度和速度。

### 目的

提出一种混合扩散模型，结合开环程序和视觉运动扩散政策，以提高操作任务的精度和速度。

### 方法

开发了远程操作增强基元(TAPs)，允许操作者在演示过程中无缝执行预定义程序，如锁定特定轴、移动到停驻点或触发特定任务的程序。混合扩散方法学习在推理过程中触发这些TAPs。

### 主要发现

该方法在具有挑战性的现实世界任务中得到了验证，包括瓶吸、开放容器液体转移和容器拧松。

### 结论

混合扩散模型结合了模仿学习的优势和传统控制方法的精度和速度。

### 翻译

尽管通过模仿学习获得的基于视觉运动的政策在复杂操作任务中表现出良好的性能，但它们通常难以达到与传统控制方法相同的精度和速度。在这项工作中，我们介绍了结合开环程序和视觉运动扩散政策的混合扩散模型。我们开发了远程操作增强基元(TAPs)，允许操作者在演示过程中无缝执行预定义程序，如锁定特定轴、移动到停驻点或触发特定任务的程序。我们的混合扩散方法学习在推理过程中触发这些TAPs。我们在具有挑战性的现实世界任务中验证了该方法：瓶吸、开放容器液体转移和容器拧松。所有实验视频都可以在项目网站上获取：https://hybriddiffusion.github.io/


### 论文摘要

Despite the fact that visuomotor-based policies obtained via imitation learning demonstrate good performances in complex manipulation tasks, they usually struggle to achieve the same accuracy and speed as traditional control based methods. In this work, we introduce Hybrid-Diffusion models that combine open-loop routines with visuomotor diffusion policies. We develop Teleoperation Augmentation Primitives (TAPs) that allow the operator to perform predefined routines, such as locking specific axes, moving to perching waypoints, or triggering task-specific routines seamlessly during demonstrations. Our Hybrid-Diffusion method learns to trigger such TAPs during inference. We validate the method on challenging real-world tasks: Vial Aspiration, Open-Container Liquid Transfer, and container unscrewing. All experimental videos are available on the project's website: https://hybriddiffusion.github.io/

---

## 47. Semi Centralized Training Decentralized Execution Architecture for Multi Agent Deep Reinforcement Learning in Traffic Signal Control

**论文链接:** [http://arxiv.org/abs/2512.04653v1](http://arxiv.org/abs/2512.04653v1)

**作者:** Pouria Yazdani, Arash Rezaali, Monireh Abdoos

**发布时间:** 2025-12-04

**备注:** Co-first authors: Pouria Yazdani and Arash Rezaali

### GPT解析

### 总结

该论文提出了一种用于多交叉口自适应交通信号控制的半集中式训练、分布式执行(SEMI-CTDE)架构，解决了现有集中式和分布式方法的局限性，在多种交通条件下表现出优越性能。

### 背景

多智能体强化学习已成为多交叉口自适应交通信号控制的有前景范式，但现有方法要么采用完全集中式设计（受维度诅咒和单点故障影响），要么采用完全分布式设计（面临部分可观测和缺乏协调的问题）。

### 目的

开发一种区域多智能体强化学习方法，将网络划分为小型紧密耦合交叉口形成的区域，围绕这些区域组织训练，以克服现有方法的局限性。

### 方法

提出SEMI-CTDE架构，在每个区域内进行集中式训练并共享区域参数，使用复合状态和奖励公式联合编码局部和区域信息，该架构可移植于不同策略骨干和状态-奖励实例化，并实现了两个具有不同设计目标的模型。

### 主要发现

多视角实验分析表明，基于SEMI-CTDE的模型在架构核心元素消融研究中表现优越，相比基于规则和完全分布式的基线，实现了持续更优的性能，且在广泛的交通密度和分布下保持有效性。

### 结论

SEMI-CTDE架构成功克服了集中式和分布式方法的局限性，在多交叉口交通信号控制中表现优越且具有鲁棒性，为实际应用提供了有效解决方案。

### 翻译

多智能体强化学习(MARL)已成为多交叉口自适应交通信号控制(ATSC)的一种有前景的范式。现有方法通常遵循完全集中式或完全分布式设计。完全集中式方法受维度诅咒和对单一学习服务器的依赖影响，而纯分布式方法在严重部分可观测条件下运行，缺乏明确协调导致次优性能。这些局限性促使了基于区域的MARL发展，其中网络被划分为形成区域的小型紧密耦合交叉口，训练围绕这些区域组织。本文为多交叉口ATSC引入了一种半集中式训练、分布式执行(SEMI-CTDE)架构。在每个区域内，SEMI-CTDE执行具有区域参数共享的集中式训练，并使用复合状态和奖励公式联合编码局部和区域信息。该架构在不同策略骨干和状态-奖励实例化之间具有高度可移植性。基于此架构，我们实现了两个具有不同设计目标的模型。对两个基于SEMI-CTDE的模型进行了多视角实验分析，包括对架构核心元素的消融研究以及基于规则和完全分布式的基线，表明它们实现了持续优越的性能，并在广泛的交通密度和分布下保持有效。


### 论文摘要

Multi-agent reinforcement learning (MARL) has emerged as a promising paradigm for adaptive traffic signal control (ATSC) of multiple intersections. Existing approaches typically follow either a fully centralized or a fully decentralized design. Fully centralized approaches suffer from the curse of dimensionality, and reliance on a single learning server, whereas purely decentralized approaches operate under severe partial observability and lack explicit coordination resulting in suboptimal performance. These limitations motivate region-based MARL, where the network is partitioned into smaller, tightly coupled intersections that form regions, and training is organized around these regions. This paper introduces a Semi-Centralized Training, Decentralized Execution (SEMI-CTDE) architecture for multi intersection ATSC. Within each region, SEMI-CTDE performs centralized training with regional parameter sharing and employs composite state and reward formulations that jointly encode local and regional information. The architecture is highly transferable across different policy backbones and state-reward instantiations. Building on this architecture, we implement two models with distinct design objectives. A multi-perspective experimental analysis of the two implemented SEMI-CTDE-based models covering ablations of the architecture's core elements including rule based and fully decentralized baselines shows that they achieve consistently superior performance and remain effective across a wide range of traffic densities and distributions.

---

## 48. Standard audiogram classification from loudness scaling data using unsupervised, supervised, and explainable machine learning techniques

**论文链接:** [http://arxiv.org/abs/2512.04616v1](http://arxiv.org/abs/2512.04616v1)

**作者:** Chen Xu, Lena Schell-Majoor, Birger Kollmeier

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究探讨了使用不依赖校准的自适应分类响度标定数据通过机器学习预测标准Bisgaard听力图类型的可行性，以解决远程听力图评估的校准和程序性挑战。

### 背景

远程听力图评估在康复听力学中存在校准和程序性挑战，需要寻找替代方法。

### 目的

研究是否可以使用不依赖校准的响度标定数据，通过机器学习将听者分类到标准Bisgaard听力图类型，从而近似个体听力图。

### 方法

评估了三类机器学习方法(无监督、监督和可解释方法)，执行主成分分析提取前两个主成分，训练和比较了七个监督多类分类器，使用包含响度标定数据的大型听觉参考数据库进行模型开发和评估。

### 主要发现

主成分分析因子图显示听者之间存在大量重叠，表明仅根据响度模式清晰分类具有挑战性；然而模型展示了合理分类性能，逻辑回归在监督方法中实现了最高准确率。

### 结论

机器学习模型可以在一定限制内从不依赖校准的响度感知数据预测标准Bisgaard听力图类型，支持在远程或资源有限环境中应用，无需传统听力图。

### 翻译

为解决康复听力学中远程听力图评估固有的校准和程序性挑战，本研究探讨了是否可以使用不依赖校准的自适应分类响度标定数据，通过机器学习将听者分类到标准Bisgaard听力图类型来近似个体听力图。评估了三种机器学习方法类别 - 无监督、监督和可解释方法。执行了主成分分析提取前两个主成分，它们共同解释了超过50%的方差。训练和比较了七个监督多类分类器，以及无监督和可解释方法。模型开发和评估使用了包含响度标定数据的大型听觉参考数据库。主成分分析因子图显示听者之间存在大量重叠，表明仅根据响度模式将参与者清晰地分为六种Bisgaard类具有挑战性。然而，模型展示了合理的分类性能，其中逻辑回归在监督方法中实现了最高准确率。这些发现表明，机器学习模型可以在一定限制内从不依赖校准的响度感知数据预测标准Bisgaard听力图类型，支持在远程或资源有限环境中应用，无需传统听力图。


### 论文摘要

To address the calibration and procedural challenges inherent in remote audiogram assessment for rehabilitative audiology, this study investigated whether calibration-independent adaptive categorical loudness scaling (ACALOS) data can be used to approximate individual audiograms by classifying listeners into standard Bisgaard audiogram types using machine learning. Three classes of machine learning approaches - unsupervised, supervised, and explainable - were evaluated. Principal component analysis (PCA) was performed to extract the first two principal components, which together explained more than 50 percent of the variance. Seven supervised multi-class classifiers were trained and compared, alongside unsupervised and explainable methods. Model development and evaluation used a large auditory reference database containing ACALOS data (N = 847). The PCA factor map showed substantial overlap between listeners, indicating that cleanly separating participants into six Bisgaard classes based solely on their loudness patterns is challenging. Nevertheless, the models demonstrated reasonable classification performance, with logistic regression achieving the highest accuracy among supervised approaches. These findings demonstrate that machine learning models can predict standard Bisgaard audiogram types, within certain limits, from calibration-independent loudness perception data, supporting potential applications in remote or resource-limited settings without requiring a traditional audiogram.

---

## 49. Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval

**论文链接:** [http://arxiv.org/abs/2512.04524v1](http://arxiv.org/abs/2512.04524v1)

**作者:** Tianle Hu, Weijun Lv, Na Han, Xiaozhao Fang, Jie Wen, Jiaxing Li, Guoxu Zhou

**发布时间:** 2025-12-04

**备注:** This paper was accepted by AAAI2026 main tech track not long ago. This is an expanded version with an appendix

### GPT解析

### 总结

本文提出了一种名为PSCA的两阶段框架，用于解决领域自适应检索中的问题，通过原型建立类级别语义连接和特征重建来提高哈希编码质量。

### 背景

领域自适应检索旨在将知识从标记的源领域转移到未标记的目标领域，实现有效检索同时减轻领域差异。

### 目的

解决现有领域自适应检索方法中存在的三个主要问题：忽略类级别语义对齐、缺乏伪标签可靠性考虑或几何指导、直接量化受领域偏移影响的原始特征。

### 方法

PSCA是一个两阶段框架：第一阶段使用正交原型建立类级别语义连接，通过几何接近性提供伪标签可靠性的语义一致性对齐；第二阶段使用领域特定量化函数处理重建特征，生成跨领域的统一二进制哈希码。

### 主要发现

通过原型学习实现类间最大可分离性和类内样本聚集，特征重建提高了哈希编码质量，相互逼近约束下的量化生成了统一的跨领域哈希码。

### 结论

大量实验在多个数据集上验证了PSCA的优越性能。

### 翻译

领域自适应检索旨在将知识从标记的源领域转移到未标记的目标领域，实现有效检索同时减轻领域差异。然而，现有方法存在几个基本局限性：1) 忽略类级别语义对齐，过度追求成对样本对齐；2) 缺乏伪标签可靠性考虑或几何指导来评估标签正确性；3) 直接量化受领域偏移影响的原始特征，损害了学习到的哈希码质量。针对这些局限性，我们提出了基于原型的语义一致性对齐(PSCA)，这是一个用于有效领域自适应检索的两阶段框架。在第一阶段，一组正交原型直接建立类级别语义连接，最大化类间可分离性同时聚集类内样本。在原型学习过程中，几何接近性通过自适应加权伪标签置信度，为语义一致性对齐提供可靠性指标。得到的成员矩阵和原型有助于特征重建，确保在重建特征而非原始特征上进行量化，从而提高后续哈希编码质量并无缝连接两个阶段。在第二阶段，领域特定量化函数在相互逼近约束下处理重建特征，生成跨领域的统一二进制哈希码。大量实验验证了PSCA在多个数据集上的优越性能。


### 论文摘要

Domain adaptive retrieval aims to transfer knowledge from a labeled source domain to an unlabeled target domain, enabling effective retrieval while mitigating domain discrepancies. However, existing methods encounter several fundamental limitations: 1) neglecting class-level semantic alignment and excessively pursuing pair-wise sample alignment; 2) lacking either pseudo-label reliability consideration or geometric guidance for assessing label correctness; 3) directly quantizing original features affected by domain shift, undermining the quality of learned hash codes. In view of these limitations, we propose Prototype-Based Semantic Consistency Alignment (PSCA), a two-stage framework for effective domain adaptive retrieval. In the first stage, a set of orthogonal prototypes directly establishes class-level semantic connections, maximizing inter-class separability while gathering intra-class samples. During the prototype learning, geometric proximity provides a reliability indicator for semantic consistency alignment through adaptive weighting of pseudo-label confidences. The resulting membership matrix and prototypes facilitate feature reconstruction, ensuring quantization on reconstructed rather than original features, thereby improving subsequent hash coding quality and seamlessly connecting both stages. In the second stage, domain-specific quantization functions process the reconstructed features under mutual approximation constraints, generating unified binary hash codes across domains. Extensive experiments validate PSCA's superior performance across multiple datasets.

---

## 50. Multi-source Learning for Target Population by High-dimensional Calibration

**论文链接:** [http://arxiv.org/abs/2512.04412v1](http://arxiv.org/abs/2512.04412v1)

**作者:** Haoxiang Zhan, Jae Kwang Kim, Yumou Qiu

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了一种高维度去校准(HDC)方法和多源HDC(MHDC)估计量，用于多源学习中的参数估计。该方法通过高维协变量平衡实现Neyman正交性，避免了增强逆概率加权公式，提供了更简单的优化算法。MHDC估计器整合多源数据，支持密度比和结果回归模型的灵活规范，具有多重稳健性。研究表明，MHDC估计器比单源HDC估计量的线性组合更高效，能有效适应多个源和多个工作模型，并且在模拟研究中表现优于现有的双重稳健估计器。气象数据集的实证分析验证了该方法在实际应用中的有效性。

### 背景

多源学习是统计学中一个新兴的研究领域，它结合了具有异质分布的多个数据集的信息，来估计目标总体中感兴趣参数，而无需观测响应。

### 目的

提出一种高维度去校准(HDC)方法和多源HDC(MHDC)估计量，用于一般估计方程，以提高多源学习的效率和稳健性。

### 方法

研究提出的高维度去校准(HDC)方法通过在扩展的协变量集上进行高维协变量平衡，为目标参数实现Neyman正交性。多源HDC(MHDC)估计器整合多源数据，支持密度比和结果回归模型的灵活规范，具有多重稳健性。该方法避免了增强逆概率加权公式，为估计方程和M估计中的目标参数提供了更简单的优化算法。

### 主要发现

1. MHDC估计器通过联合利用所有数据源，比单源HDC估计量的线性组合更高效；2. MHDC估计器能有效适应多个源和多个工作模型；3. 在模拟研究中，MHDC估计器比现有的多源学习双重稳健估计器表现更好；4. 建立了MHDC估计器的渐近正态性；5. 提出了一个规格测试来检查多源数据的可转移性条件。

### 结论

所提出的高维度去校准(HDC)方法和多源HDC(MHDC)估计器在多源学习中表现优异，能有效整合多源数据，支持灵活的模型规范，具有多重稳健性，并且在效率和性能上优于现有方法。气象数据集的实证分析进一步验证了该方法在实际应用中的有效性。

### 翻译

多源学习是统计学中一个新兴的研究领域，其中结合了具有异质分布的多个数据集的信息，来估计目标总体中感兴趣的参数，而无需观测响应。我们提出了一种高维度去校准(HDC)方法和一种用于一般估计方程的多源HDC(MHDC)估计量。HDC方法使用一种新颖的方法，通过在扩展的协变量集上进行高维协变量平衡，为目标参数实现Neyman正交性。它避免了增强逆概率加权公式，并为估计方程和M估计中的目标参数提供了更简单的优化算法。所提出的MHDC估计器整合多源数据，同时支持密度比和结果回归模型的灵活规范，实现对模型误规格化的多重稳健性。建立了其渐近正态性，并提出了一个规格测试来检查多源数据的可转移性条件。与单源HDC估计量的线性组合相比，MHDC估计器通过联合利用所有数据源提高了效率。通过模拟研究，我们表明MHDC估计器能有效适应多个源和多个工作模型，并且在多源学习中比现有的双重稳健估计器表现更好。对气象数据集的实证分析证明了所提出方法在实际中的实用性。


### 论文摘要

Multi-source learning is an emerging area of research in statistics, where information from multiple datasets with heterogeneous distributions is combined to estimate the parameter of interest for a target population without observed responses. We propose a high-dimensional debiased calibration (HDC) method and a multi-source HDC (MHDC) estimator for general estimating equations. The HDC method uses a novel approach to achieve Neyman orthogonality for the target parameter via high-dimensional covariate balancing on an augmented set of covariates. It avoids the augmented inverse probability weighting formulation and leads to an easier optimization algorithm for the target parameter in estimating equations and M-estimation. The proposed MHDC estimator integrates multi-source data while supporting flexible specifications for both density ratio and outcome regression models, achieving multiple robustness against model misspecification. Its asymptotic normality is established, and a specification test is proposed to examine the transferability condition for the multi-source data. Compared to the linear combination of single-source HDC estimators, the MHDC estimator improves efficiency by jointly utilizing all data sources. Through simulation studies, we show that the MHDC estimator accommodates multiple sources and multiple working models effectively and performs better than the existing doubly robust estimators for multi-source learning. An empirical analysis of a meteorological dataset demonstrates the utility of the proposed method in practice.

---

## 51. Performance Evaluation of Transfer Learning Based Medical Image Classification Techniques for Disease Detection

**论文链接:** [http://arxiv.org/abs/2512.04397v1](http://arxiv.org/abs/2512.04397v1)

**作者:** Zeeshan Ahmad, Shudi Bao, Meng Chen

**发布时间:** 2025-12-04

**DOI:** 10.1109/EMBC58623.2025.11253609

### GPT解析

### 总结

该研究全面分析了迁移学习技术在医学图像分类中的应用，通过评估多种预训练模型发现InceptionV3性能最佳，ResNet系列模型性能随深度增加而提高，并验证了迁移学习在数据有限情况下的有效性，同时为模型选择提供了指导。

### 背景

医学图像分类在识别各种疾病方面扮演着越来越重要的角色，通过对X光片、MRI和CT扫描等医学图像进行分类，可以将其归入不同类别。近年来，深度学习技术在医学图像分类领域引起了广泛关注。

### 目的

由于从头训练整个大型深度学习模型通常不可行，该研究旨在通过迁移学习（TL）技术来解决这一问题，即重用预训练模型来完成新任务，并对基于深度卷积神经网络的医学图像分类中的TL技术进行全面分析。

### 方法

研究者在自定义的胸部X光数据集上评估了六种预训练模型（AlexNet、VGG16、ResNet18、ResNet34、ResNet50和InceptionV3）用于疾病检测。此外，还进行了不确定性分析和运行时间比较，以评估这些模型的鲁棒性和计算效率。

### 主要发现

InceptionV3在所有标准指标上均优于其他模型；ResNet系列模型随着深度的增加表现出更好的性能；VGG16和AlexNet表现良好但准确率较低；迁移学习在大多数情况下是有益的，尤其是在数据有限的情况下，但改进程度取决于多种因素，如模型架构、数据集大小以及源任务和目标任务之间的领域相似性；经过良好训练的特征提取器只需要一个轻量级的前馈模型就足以提供高效的预测。

### 结论

该研究有助于理解医学图像分类中的迁移学习，并为根据特定需求选择适当的模型提供了见解。

### 翻译

医学图像分类通过将医学图像（如X光片、MRI和CT扫描）根据其特征分类到不同类别，在识别各种疾病方面发挥着越来越重要的作用。近年来，深度学习技术在医学图像分类领域引起了广泛关注。然而，从头训练整个大型深度学习模型通常不可行。为解决这个问题，迁移学习（TL）技术是一种解决方案，其中预训练模型被重用于新任务。在本文中，我们对基于深度卷积神经网络的医学图像分类中的TL技术进行了全面分析。我们在自定义的胸部X光数据集上评估了六种预训练模型（AlexNet、VGG16、ResNet18、ResNet34、ResNet50和InceptionV3）用于疾病检测。实验结果表明，InceptionV3在所有标准指标上均优于其他模型。ResNet系列模型随着深度的增加表现出更好的性能，而VGG16和AlexNet表现良好但准确率较低。此外，我们还进行了不确定性分析和运行时间比较，以评估这些模型的鲁棒性和计算效率。我们的研究结果表明，迁移学习在大多数情况下是有益的，尤其是在数据有限的情况下，但改进程度取决于多种因素，如模型架构、数据集大小以及源任务和目标任务之间的领域相似性。此外，我们证明经过良好训练的特征提取器只需要一个轻量级的前馈模型就足以提供高效的预测。因此，这项研究有助于理解医学图像分类中的迁移学习，并为根据特定需求选择适当的模型提供了见解。


### 论文摘要

Medical image classification plays an increasingly vital role in identifying various diseases by classifying medical images, such as X-rays, MRIs and CT scans, into different categories based on their features. In recent years, deep learning techniques have attracted significant attention in medical image classification. However, it is usually infeasible to train an entire large deep learning model from scratch. To address this issue, one of the solutions is the transfer learning (TL) technique, where a pre-trained model is reused for a new task. In this paper, we present a comprehensive analysis of TL techniques for medical image classification using deep convolutional neural networks. We evaluate six pre-trained models (AlexNet, VGG16, ResNet18, ResNet34, ResNet50, and InceptionV3) on a custom chest X-ray dataset for disease detection. The experimental results demonstrate that InceptionV3 consistently outperforms other models across all the standard metrics. The ResNet family shows progressively better performance with increasing depth, whereas VGG16 and AlexNet perform reasonably well but with lower accuracy. In addition, we also conduct uncertainty analysis and runtime comparison to assess the robustness and computational efficiency of these models. Our findings reveal that TL is beneficial in most cases, especially with limited data, but the extent of improvement depends on several factors such as model architecture, dataset size, and domain similarity between source and target tasks. Moreover, we demonstrate that with a well-trained feature extractor, only a lightweight feedforward model is enough to provide efficient prediction. As such, this study contributes to the understanding of TL in medical image classification, and provides insights for selecting appropriate models based on specific requirements.

---

## 52. Inference-time Stochastic Refinement of GRU-Normalizing Flow for Real-time Video Motion Transfer

**论文链接:** [http://arxiv.org/abs/2512.04282v1](http://arxiv.org/abs/2512.04282v1)

**作者:** Tasmiah Haque, Srinjoy Das

**发布时间:** 2025-12-03

### GPT解析

### 总结

本文提出了一种名为GRU-SNF的新型推理时细化技术，通过在GRU-NF模型中引入马尔可夫链蒙特卡洛步骤，提高了视频运动预测的多样性，同时保持准确性，无需重新训练模型。

### 背景

实时视频运动传输应用（如沉浸式游戏和基于视觉的异常检测）需要准确且多样化的未来预测，以支持在不确定性下的真实合成和稳健的下游决策。

### 目的

提高顺序预测的多样性，解决GRU-NF的确定性变换结构限制表达能力的问题，使模型能够更好地近似真实数据分布。

### 方法

结合GRU-NF与随机采样方法，受随机归一化流启发，在GRU-NF推理过程中引入马尔可夫链蒙特卡洛步骤，使模型能够探索更丰富的输出空间。

### 主要发现

在关键点视频运动传输管道中验证，GRU-SNF在生成多样化输出方面优于GRU-NF，且不牺牲准确性，即使在更长预测范围内也是如此；通过注入随机性更有效地捕获多模态行为。

### 结论

将随机动力学与基于流的序列模型相结合，在生成时间序列预测方面具有巨大潜力。

### 翻译

实时视频运动传输应用（如沉浸式游戏和基于视觉的异常检测）需要准确且多样化的未来预测，以支持在不确定性下的真实合成和稳健的下游决策。为了提高此类顺序预测的多样性，我们提出了一种新颖的推理时细化技术，结合了门控循环单元-归一化流和随机采样方法。虽然GRU-NF可以通过在时间预测框架中集成归一化流来捕获多模态分布，但其确定性变换结构可能限制表达能力。为解决这一问题，受随机归一化流启发，我们在GRU-NF推理过程中引入马尔可夫链蒙特卡洛步骤，使模型能够探索更丰富的输出空间，更好地近似真实数据分布，无需重新训练。我们在关键点视频运动传输管道中验证了我们的方法，其中捕获时间连贯且感知多样化的未来轨迹对于真实样本和低带宽通信至关重要。实验表明，我们的推理框架——门控循环单元-随机归一化流在生成多样化输出方面优于GRU-NF，且不牺牲准确性，即使在更长预测范围内也是如此。通过在推理过程中注入随机性，我们的方法更有效地捕获多模态行为。这些结果突显了将随机动力学与基于流的序列模型相结合用于生成时间序列预测的潜力。


### 论文摘要

Real-time video motion transfer applications such as immersive gaming and vision-based anomaly detection require accurate yet diverse future predictions to support realistic synthesis and robust downstream decision making under uncertainty. To improve the diversity of such sequential forecasts we propose a novel inference-time refinement technique that combines Gated Recurrent Unit-Normalizing Flows (GRU-NF) with stochastic sampling methods. While GRU-NF can capture multimodal distributions through its integration of normalizing flows within a temporal forecasting framework, its deterministic transformation structure can limit expressivity. To address this, inspired by Stochastic Normalizing Flows (SNF), we introduce Markov Chain Monte Carlo (MCMC) steps during GRU-NF inference, enabling the model to explore a richer output space and better approximate the true data distribution without retraining. We validate our approach in a keypoint-based video motion transfer pipeline, where capturing temporally coherent and perceptually diverse future trajectories is essential for realistic samples and low bandwidth communication. Experiments show that our inference framework, Gated Recurrent Unit- Stochastic Normalizing Flows (GRU-SNF) outperforms GRU-NF in generating diverse outputs without sacrificing accuracy, even under longer prediction horizons. By injecting stochasticity during inference, our approach captures multimodal behavior more effectively. These results highlight the potential of integrating stochastic dynamics with flow-based sequence models for generative time series forecasting.

---

## 53. Enhancing next token prediction based pre-training for jet foundation models

**论文链接:** [http://arxiv.org/abs/2512.04149v1](http://arxiv.org/abs/2512.04149v1)

**作者:** Joschka Birk, Anna Hallin, Gregor Kasieczka, Nikol Madzharova, Ian Pang, David Shih

**发布时间:** 2025-12-03

### GPT解析

### 总结

该研究基于OmniJet-α的初步工作，对next token预测进行了多项改进，包括采用混合输入策略和结合预训练方法，显著提升了下游分类任务性能同时保持生成能力。

### 背景

Next token prediction是一种吸引人的jet foundation模型预训练任务，它无需模拟且具有出色的跨数据集迁移生成能力。

### 目的

研究对next token预测方法的多种改进，以提升模型性能。

### 方法

1) 采用混合设置，使用连续特征向量作为模型输入，同时在预测目标中仅使用token-ID；2) 探索结合masked particle modeling和生成学习目标的预训练策略。

### 主要发现

所提出的改进方法显著提高了下游分类任务性能，同时没有任何生成性能的损失。

### 结论

通过混合输入策略和结合预训练方法，可以在不牺牲生成能力的情况下提升分类性能。

### 翻译

下一个token预测是jet foundation模型的一个有吸引力的预训练任务，因为它不需要模拟，并且能够实现出色的跨数据集迁移生成能力。在这里，我们研究了对next token预测的多种改进，基于OmniJet-α的初步工作。我们没有将粒子标记化后仅使用token-ID作为生成和分类任务的模型输入，而是采用混合设置，这允许我们使用连续特征向量作为模型输入，同时仅在下一个token预测目标中使用token-ID。其次，我们探索了结合masked particle modeling和生成学习目标的组合预训练策略。总而言之，这些改进大大提高了下游分类任务的性能，同时没有任何生成性能的损失。


### 论文摘要

Next token prediction is an attractive pre-training task for jet foundation models, in that it is simulation free and enables excellent generative capabilities that can transfer across datasets. Here we study multiple improvements to next token prediction, building on the initial work of OmniJet-$α$. Instead of tokenizing particles and subsequently only using the token-ID as the model input for both the generative and the classification task, we adopt a hybrid setup, which allows us to use continuous feature vectors as model input while only using token-IDs in the next token prediction target. Secondly, we explore a combined pre-training strategy that combines masked particle modeling and generative learning objectives. Taken together, these changes greatly improve the performance in downstream classification tasks without any loss in generative performance.

---

## 54. Lévy sources in UrQMD in Ar+Sc collisions at SPS energies

**论文链接:** [http://arxiv.org/abs/2512.05019v1](http://arxiv.org/abs/2512.05019v1)

**作者:** Barnabas Porfy, Mate Csanad

**发布时间:** 2025-12-04

**备注:** 8 pages, 5 figures

### GPT解析

### 总结

本研究探讨了SPS能量下氩-45钪中心碰撞产生的三维双π对源分布，使用Lévy稳定分布进行拟合，并分析了描述源特性的Lévy参数。

### 背景

飞秒镜领域的发展由实验测量和理论计算之间的相互作用驱动。实验为理论提供数据，理论预测指导新的测量。过去十年中，多项实验证实双粒子π发射源可用Lévy alpha稳定分布很好地描述。

### 目的

研究SPS能量下氩-45钪中心碰撞产生的三维双π对源分布，并通过Lévy稳定分布拟合提取描述源特性的参数。

### 方法

使用超相对论量子分子动力学蒙特卡洛事件生成器生成碰撞数据，用Lévy稳定分布拟合对源，并分析提取的Lévy参数。

### 主要发现

通过拟合得到了描述源空间尺度、形状和强度的Lévy参数，这些参数有助于理解双π发射源的特性。

### 结论

Lévy稳定分布能够很好地描述双π对源分布，提取的参数提供了关于源特性的重要信息。

### 翻译

在过去的几十年中，飞秒镜的进展是由实验测量和理论计算之间的相互作用驱动的。测量为理论提供数据以理解它，而理论预测则指导新的测量。在最近十年中，几项实验证实双粒子π发射源可以用Lévy alpha稳定分布很好地描述。为了进行理论解释，已在RHIC和LHC能量下进行了现象学模拟，使用了各种可用的重离子碰撞模型。在本文中，我们研究了使用超相对论量子分子动力学蒙特卡洛事件生成器生成的SPS能量下氩-45钪中心碰撞的三维双π对源分布。我们使用Lévy稳定分布拟合对源，并讨论了提取的描述源空间尺度、形状和强度的Lévy参数。


### 论文摘要

Over the past few decades, progress in femtoscopy has been driven by the interplay between experimental measurements and theoretical calculations. Measurements provide data for the theory to understand it, while theoretical predictions guide new measurements. In the recent decade, several experiments have confirmed that the two-particle pion emitting source is well described by Lévy alpha-stable distributions. To enable theoretical interpretation, phenomenological simulations have been done at RHIC and LHC energies, using various available heavy-ion collision models. In this paper, we investigate three-dimensional two-pion pair source distributions from $^{40}$Ar+$^{45}$Sc central collisions at SPS energies, generated with the Ultra-Relativistic Quantum Molecular Dynamics Monte-Carlo event generator. We fit the pair source with Lévy-stable distributions, and discuss the extracted Lévy parameters describing the spatial scale, shape and strength of the source.

---

## 55. 论文ID: 2512.05009v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05009v1.json'

---

## 56. 论文ID: 2512.04790v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04790v1.json'

---

## 57. MT-Depth: Multi-task Instance feature analysis for the Depth Completion

**论文链接:** [http://arxiv.org/abs/2512.04734v1](http://arxiv.org/abs/2512.04734v1)

**作者:** Abdul Haseeb Nizamani, Dandi Zhou, Xinhai Sun

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了一种实例感知的深度补全框架，通过集成二进制实例掩码作为空间先验来优化深度预测，显著提高了3D感知系统中深度补全的准确性。

### 背景

深度补全在3D感知系统中至关重要，特别是在自动驾驶、机器人和增强现实等需要将稀疏深度数据稠化的场景中。现有方法通常依赖语义分割指导深度补全，但忽略了对象级理解的优势。

### 目的

引入一个实例感知的深度补全框架，显式集成二进制实例掩码作为空间先验来优化深度预测，提高深度补全的准确性，特别是在对象边界、遮挡和薄结构区域。

### 方法

模型结合四个主要组件：冻结的YOLO V11实例分割分支、基于U-Net的深度补全主干网络、交叉注意力融合模块和注意力引导的预测头。实例分割分支生成前景掩码，通过交叉注意力引导深度分支，使网络在细化过程中专注于以对象为中心的区域。

### 主要发现

在Virtual KITTI 2数据集上验证，与仅使用U-Net的基线方法和之前的语义引导方法相比，实现了更低的RMSE，同时保持了有竞争力的MAE。定性和定量结果表明，提出的模型有效提高了对象边界、遮挡和薄结构附近的深度准确性。

### 结论

将实例感知线索纳入深度补全过程是一个有前景的方向，可以在不依赖密集语义标签的情况下提高深度补全性能。

### 翻译

深度补全在3D感知系统中起着至关重要的作用，特别是在需要将稀疏深度数据稠化的场景中，如自动驾驶、机器人和增强现实。虽然许多现有方法依赖语义分割来指导深度补全，但它们常常忽略了对象级理解的好处。在这项工作中，我们引入了一个实例感知的深度补全框架，显式集成二进制实例掩码作为空间先验来优化深度预测。我们的模型结合了四个主要组件：一个冻结的YOLO V11实例分割分支、一个基于U-Net的深度补全主干网络、一个交叉注意力融合模块和一个注意力引导的预测头。实例分割分支生成每张图像的前景掩码，通过交叉注意力引导深度分支，使网络在细化过程中专注于以对象为中心的区域。我们在Virtual KITTI 2数据集上验证了我们的方法，结果表明与仅使用U-Net的基线方法和之前的语义引导方法相比，它实现了更低的RMSE，同时保持了有竞争力的MAE。定性和定量结果表明，所提出的模型有效提高了对象边界、遮挡和薄结构附近的深度准确性。我们的研究结果表明，纳入实例感知线索是改善深度补全的一个有前景的方向，无需依赖密集的语义标签。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决深度完成（Depth Completion）技术中的精度问题，特别是在物体边界、遮挡区域和薄结构等挑战场景下的表现不足。深度完成在自动驾驶、机器人和增强现实等3D感知系统中至关重要，它需要将稀疏传感器数据（如LiDAR点云）转换为密集的深度图。准确完成这一任务对于实现可靠的3D场景理解、物体检测和导航决策至关重要，因为这些应用需要精确的距离和形状信息来避免碰撞、进行交互和重建环境。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有深度完成方法的局限性，特别是它们在处理复杂场景时的不足。他们借鉴了语义分割在深度完成中的应用（如SemSegDepth和PanDepth模型），以及多任务学习框架中共享表示的理念。然而，作者注意到大多数先前工作依赖于像素级语义标签，忽略了实例级理解的价值。基于这一观察，作者设计了一个多任务架构，利用YOLO V11实例分割模型提取对象级信息，并通过交叉注意力机制将这些实例特征整合到深度完成过程中，从而在不增加大量计算复杂度的情况下提高深度预测的准确性。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是利用实例级特征作为空间先验来引导深度完成过程，使网络能够关注物体中心区域并保持深度预测与物体边界的一致性。整体流程包括四个主要步骤：1）使用冻结的YOLO V11实例分割模型从RGB图像中提取并合并二进制实例掩码；2）基于U-Net的深度完成分支处理RGB图像、稀疏深度图和有效掩码的拼接输入，生成初始深度预测；3）通过交叉注意力机制融合实例特征（作为查询）和深度特征（作为键和值），增强对物体区域的关注；4）使用注意力引导的预测头（包含通道注意力机制）融合特征并生成最终的密集深度图。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）提出了一种实例感知的深度完成框架，明确整合二进制实例掩码作为空间先验；2）设计了交叉注意力机制，使实例特征能够引导深度特征的关注区域；3）采用四组件架构（YOLO V11分支、U-Net分支、交叉注意力模块和预测头）；4）仅使用二进制实例掩码而不需要像素级语义标签。与之前工作相比，本文专注于实例级而非语义级特征，使用预训练分割模型而非联合训练降低复杂度，并在挑战性场景（如遮挡和边界区域）实现了更好的性能，同时保持了较低的RMSE误差。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种创新的实例感知深度完成框架，通过整合二进制实例掩码和交叉注意力机制，显著提高了在物体边界、遮挡区域和薄结构等挑战场景中的深度预测准确性，同时无需依赖密集的语义标签。'}


### 论文摘要

Depth completion plays a vital role in 3D perception systems, especially in scenarios where sparse depth data must be densified for tasks such as autonomous driving, robotics, and augmented reality. While many existing approaches rely on semantic segmentation to guide depth completion, they often overlook the benefits of object-level understanding. In this work, we introduce an instance-aware depth completion framework that explicitly integrates binary instance masks as spatial priors to refine depth predictions. Our model combines four main components: a frozen YOLO V11 instance segmentation branch, a U-Net-based depth completion backbone, a cross-attention fusion module, and an attention-guided prediction head. The instance segmentation branch generates per-image foreground masks that guide the depth branch via cross-attention, allowing the network to focus on object-centric regions during refinement. We validate our method on the Virtual KITTI 2 dataset, showing that it achieves lower RMSE compared to both a U-Net-only baseline and previous semantic-guided methods, while maintaining competitive MAE. Qualitative and quantitative results demonstrate that the proposed model effectively enhances depth accuracy near object boundaries, occlusions, and thin structures. Our findings suggest that incorporating instance-aware cues offers a promising direction for improving depth completion without relying on dense semantic labels.

---

## 58. E3AD: An Emotion-Aware Vision-Language-Action Model for Human-Centric End-to-End Autonomous Driving

**论文链接:** [http://arxiv.org/abs/2512.04733v1](http://arxiv.org/abs/2512.04733v1)

**作者:** Yihong Tang, Haicheng Liao, Tong Nie, Junlin He, Ao Qu, Kehua Chen, Wei Ma, Zhenning Li, Lijun Sun, Chengzhong Xu

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种情感感知的视觉-语言-行动框架E3AD，用于端到端自动驾驶系统，通过考虑乘客情感状态来提高驾驶舒适度和接受度。

### 背景

端到端自动驾驶系统越来越多地采用视觉-语言-行动模型，但通常忽略乘客的情感状态，而情感状态对舒适度和自动驾驶接受度至关重要。

### 目的

引入开放领域端到端自动驾驶，使自动驾驶车辆能够解释自由形式的自然语言命令，推断情感，并规划物理可行的轨迹。

### 方法

提出E3AD框架，包含两个认知启发的组件：连续的Valence-Arousal-Dominance情感模型捕捉语言中的语调和紧迫性；双通路空间推理模块融合以自我为中心和以世界为中心的视图实现类人空间认知。采用一致性导向的训练方案，结合模态预训练和基于偏好的对齐。

### 主要发现

在真实世界数据集上，E3AD提高了视觉定位和路径点规划能力，并在情感估计方面取得了最先进的VAD相关性。

### 结论

将情感注入视觉-语言-行动风格的驾驶中，可以产生更符合人类期望的定位、规划和以人为中心的反馈。

### 翻译

端到端自动驾驶系统日益采用视觉-语言-行动模型，但通常忽略乘客的情感状态，这对舒适度和自动驾驶接受度至关重要。我们引入开放领域端到端自动驾驶，其中自动驾驶车辆必须解释自由形式的自然语言命令，推断情感，并规划物理可行的轨迹。我们提出E3AD，一个情感感知的视觉-语言-行动框架，通过两个认知启发的组件增强语义理解：捕捉语言中语调和紧迫性的连续Valence-Arousal-Dominance情感模型，以及融合以自我为中心和以世界为中心视图实现类人空间认知的双通路空间推理模块。结合模态预训练和基于偏好的对齐的一致性导向训练方案，进一步强化情感意图和驾驶行动之间的一致性。在真实世界数据集上，E3AD提高了视觉定位和路径点规划能力，并在情感估计方面取得了最先进的VAD相关性。这些结果表明，将情感注入视觉-语言-行动风格的驾驶中，可以产生更符合人类期望的定位、规划和以人为中心的反馈。


### 论文摘要

End-to-end autonomous driving (AD) systems increasingly adopt vision-language-action (VLA) models, yet they typically ignore the passenger's emotional state, which is central to comfort and AD acceptance. We introduce Open-Domain End-to-End (OD-E2E) autonomous driving, where an autonomous vehicle (AV) must interpret free-form natural-language commands, infer the emotion, and plan a physically feasible trajectory. We propose E3AD, an emotion-aware VLA framework that augments semantic understanding with two cognitively inspired components: a continuous Valenc-Arousal-Dominance (VAD) emotion model that captures tone and urgency from language, and a dual-pathway spatial reasoning module that fuses egocentric and allocentric views for human-like spatial cognition. A consistency-oriented training scheme, combining modality pretraining with preference-based alignment, further enforces coherence between emotional intent and driving actions. Across real-world datasets, E3AD improves visual grounding and waypoint planning and achieves state-of-the-art (SOTA) VAD correlation for emotion estimation. These results show that injecting emotion into VLA-style driving yields more human-aligned grounding, planning, and human-centric feedback.

---

## 59. Towards Cross-View Point Correspondence in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.04686v1](http://arxiv.org/abs/2512.04686v1)

**作者:** Yipu Wang, Yuheng Ji, Yuyang Liu, Enshen Zhou, Ziqiang Yang, Yuxuan Tian, Ziheng Qin, Yue Liu, Huajie Tan, Cheng Chi, Zhiyuan Ma, Daniel Dajun Zeng, Xiaolong Zheng

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了跨视图点对应(CVPC)任务和CrossPoint-Bench基准测试，构建了CrossPoint-378K数据集和CroPond模型，解决了视觉语言模型中精确点级对应能力不足的问题，显著提升了跨视图对应的性能。

### 背景

跨视图对应是空间理解和具身AI的基本能力，但在视觉语言模型(VLMs)中仍未实现，特别是在精确的点级对应方面，这对精确的交互能力至关重要。

### 目的

提出跨视图点对应任务和综合基准测试，解决视觉语言模型中精确点级对应能力不足的问题，为精确的交互能力提供基础。

### 方法

构建了CrossPoint-Bench基准测试，采用分层设计，灵感来自人类'感知'、'推理'和'对应'的认知过程；创建了CrossPoint-378K数据集，包含37.8万个问答对，覆盖900个场景，专注于可操作区域；提出了CroPond模型，在CrossPoint-378K数据集上进行训练。

### 主要发现

最先进的模型(如Gemini-2.5-Pro)在整体准确率上仍远低于人类，差距超过54.65%，暴露了从粗粒度判断到细粒度坐标预测转变的挑战；CroPond在CrossPoint-Bench上取得了最先进的性能，比Gemini-2.5-Pro高出39.7%的准确率。

### 结论

该研究为推进跨视图对应方面的未来工作提供了基础，基准测试、数据集和模型已在GitHub上公开。

### 翻译

跨视图对应是空间理解和具身AI的基本能力。然而，在视觉语言模型(VLMs)中仍未实现，特别是在实现精确的点级对应方面，这对精确的交互能力至关重要。因此，我们提出了跨视图点对应(CVPC)任务和CrossPoint-Bench，这是一个受人类'感知'、'推理'和'对应'认知过程启发的综合基准测试，采用分层设计。我们的评估显示，最先进的模型(如Gemini-2.5-Pro)仍远落后于人类，整体准确率差距超过54.65%，暴露了从粗粒度判断到细粒度坐标预测转变的挑战。为解决这个问题，我们构建了CrossPoint-378K数据集，包含900个场景中的37.8万个问答对，专注于能更好反映现实世界操作和交互场景的可操作区域。此外，我们提出了在CrossPoint-378K数据集上训练的CroPond模型。我们的CroPond在CrossPoint-Bench上取得了最先进的性能，比Gemini-2.5-Pro高出39.7%的准确率，为推进未来跨视图对应工作提供了基础。该基准测试、数据集和模型已在https://github.com/WangYipu2002/CrossPoint公开。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决视觉语言模型(VLMs)在跨视角点对应(Cross-View Point Correspondence, CVPC)方面的能力不足问题。这个问题非常重要，因为人类具有出色的跨视角空间对应能力，可以从不同视角可靠地推断同一物理点的位置，这种能力对于导航、抓取、多智能体协作等具身AI任务至关重要。然而，当前VLMs的理解局限于静态单视角，无法预测机器人运动所需的精确点，导致在现实应用中存在'现实差距'。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': "作者受到人类跨视角推理认知过程的启发，将其分解为'感知'、'推理'和'对应'三个阶段，并基于此定义了CVPC任务。他们分析了现有基准的局限性(任务粒度有限、缺乏语义相关性、评估维度不足)，设计了CrossPoint-Bench基准。数据集构建借鉴了现有图像分割技术(如SAM2.1)和物体检测技术(如Grounding-DINO)，并利用了现有的VLM(如Qwen2.5-VL)作为基础模型进行微调。作者还参考了现有的空间理解数据集来扩充训练数据，采用多源数据联合训练策略。", '这个方法的核心思想是什么？整体实现流程是怎样的？': "核心思想是将跨视角点对应作为多阶段认知任务，围绕'可操作区域'构建数据，采用层次化基准设计，并通过多源数据联合训练提升模型性能。整体流程包括：1)数据集构建：从3D视频中采样图像，使用多阶段技术分割可操作区域，进行跨视角配对，生成问答对；2)模型开发：基于Qwen2.5-VL使用监督微调训练CroPond模型；3)评估：在CrossPoint-Bench上系统评估模型性能，多项选择任务使用准确率，指向任务使用掩码内命中率。", '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)系统定义CVPC任务并分解为三个认知阶段；2)提出首个综合基准CrossPoint-Bench，采用层次化设计；3)构建CrossPoint-378K大规模数据集，专注于可操作区域；4)开发CroPond模型实现最先进性能。相比之前的工作，本文专注于点级定位而非选择级任务，所有任务围绕功能上重要的可操作区域，系统评估模型对尺度变化和遮挡的鲁棒性，数据集专门针对几何一致性和细粒度可操作性需求构建，模型性能显著提升。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过系统定义跨视角点对应任务、提出首个综合基准、构建大规模数据集和开发高性能模型，显著提升了视觉语言模型在跨视角空间理解和精确点级对应方面的能力，为具身AI和多智能体协作奠定了基础。'}


### 论文摘要

Cross-view correspondence is a fundamental capability for spatial understanding and embodied AI. However, it is still far from being realized in Vision-Language Models (VLMs), especially in achieving precise point-level correspondence, which is crucial for precise affordance interaction. So we propose the Cross-View Point Correspondence (CVPC) task and CrossPoint-Bench, a comprehensive benchmark with hierarchical design, inspired by the human cognitive process of "perceive", "reason", and "correspond". Our evaluation shows the state-of-the-art models (e.g., Gemini-2.5-Pro) still fall far behind humans, with a gap of over 54.65% in overall accuracy, exposing a challenge in transitioning from coarse-grained judgement to fine-grained coordinate prediction. To address this problem, we construct CrossPoint-378K, a dataset with 378K question-answering pairs across 900 scenes, focused on actionable affordance regions that better reflect real-world manipulation and interaction scenarios. Furthermore, we propose CroPond that trained on the CrossPoint-378K dataset. Our CroPond achieves state-of-the-art performance on CrossPoint-Bench, surpassing Gemini-2.5-Pro by 39.7% accuracy, which offers a foundation for advancing future work on cross-view correspondence. The benchmark, dataset, and model are publicly available at https://github.com/WangYipu2002/CrossPoint.

---

## 60. SEASON: Mitigating Temporal Hallucination in Video Large Language Models via Self-Diagnostic Contrastive Decoding

**论文链接:** [http://arxiv.org/abs/2512.04643v1](http://arxiv.org/abs/2512.04643v1)

**作者:** Chang-Hsun Wu, Kai-Po Chang, Yu-Yang Sheng, Hung-Kai Chung, Kuei-Chun Wang, Yu-Chiang Frank Wang

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种名为SEASON（Self-Diagnostic Contrastive Decoding）的无需训练方法，用于解决视频大语言模型在视频理解中存在的时间幻觉问题，通过自适应增强输出的时间和空间一致性来减轻幻觉。

### 背景

视频大语言模型在视频理解方面已取得显著进展，但这些模型在感知和利用视频中的丰富时间信息时仍存在困难，导致生成时间不一致或因果关系不合理的事件描述，造成严重幻觉问题。时间推理在视频理解中相对未被充分探索。

### 目的

解决VideoLLMs在时间感知方面的不足，提出一种能够有效提高模型输出的时间和空间一致性的方法，减轻视频理解中的幻觉问题。

### 方法

提出SEASON（Self-Diagnostic Contrastive Decoding），一种无需训练的方法。通过动态诊断每个输出标记的幻觉倾向，并针对相应的时间和空间负面样本应用自适应对比解码，从而自适应地增强输出的时间和空间一致性。

### 主要发现

大量实验表明，SEASON在三个幻觉检查基准测试中优于所有现有的无需训练的幻觉缓解方法，同时在四个通用视频理解基准测试中进一步提高了VideoLLMs的性能。

### 结论

SEASON方法有效解决了VideoLLMs在时间理解和幻觉方面的问题，无需额外训练就能显著提高模型的表现，代码将在论文接受后发布。

### 翻译

视频大语言模型（VideoLLMs）在视频理解方面已经取得了显著进展。然而，这些模型在响应用户查询时，仍然难以有效感知和利用视频中的丰富时间信息。因此，它们经常生成时间不一致或因果关系不合理的事件描述，导致严重的幻觉问题。虽然大多数先前的研究集中在空间幻觉（例如对象不匹配）上，但视频理解中的时间推理仍然相对未被充分探索。为了解决这个问题，我们提出了SEASON（Self-Diagnostic Contrastive Decoding），一种无需训练的方法，能够自适应地增强每个输出标记的时间和空间一致性。它通过动态诊断每个标记的幻觉倾向，并针对相应的时间和空间负面样本应用自适应对比解码来实现这一点。大量的实验表明，SEASON在三个幻觉检查基准测试中优于所有现有的无需训练的幻觉缓解方法，同时在四个通用视频理解基准测试中进一步提高了VideoLLMs的性能。代码将在论文接受后发布。


### 论文摘要

Video Large Language Models (VideoLLMs) have shown remarkable progress in video understanding. However, these models still struggle to effectively perceive and exploit rich temporal information in videos when responding to user queries. Therefore, they often generate descriptions of events that are temporal inconsistent or causally implausible, causing severe hallucination issues. While most prior studies have focused on spatial hallucinations (e.g. object mismatches), temporal reasoning in video understanding remains relatively underexplored. To address this issue, we propose Self-Diagnostic Contrastive Decoding (SEASON), a training-free method that adaptively enhances temporal and spatial faithfulness for each output token. It achieves this by dynamically diagnosing each token's hallucination tendency and applying adaptive contrastive decoding against its corresponding temporal and spatial negatives. Extensive experiments demonstrate that SEASON outperforms all existing training-free hallucination mitigation approaches on three hallucination examination benchmarks, while further improves VideoLLMs across four general video understanding benchmarks. The code will be released upon acceptance.

---

## 61. SAM3-I: Segment Anything with Instructions

**论文链接:** [http://arxiv.org/abs/2512.04585v1](http://arxiv.org/abs/2512.04585v1)

**作者:** Jingjing Li, Yue Feng, Yuchen Guo, Jincai Huang, Yongri Piao, Qi Bi, Miao Zhang, Xiaoqi Zhao, Qiang Chen, Shihao Zou, Wei Ji, Huchuan Lu, Li Cheng

**发布时间:** 2025-12-04

**备注:** Preliminary results; work in progress

### GPT解析

### 总结

本文提出了一种名为SAM3-I的增强框架，统一了概念级理解和指令级推理，使模型能够直接遵循自然语言指令进行分割，同时保留原有的概念驱动能力。

### 背景

SAM3模型通过可提示的概念分割实现了开放词汇分割，但现实世界的使用通常需要更丰富的表达方式，包括属性、空间关系、功能、动作、状态等。目前SAM3依赖于外部多模态代理将复杂指令转换为名词短语(NP)，但这些NP级概念过于粗糙，无法精确表示特定实例。

### 目的

开发一个增强框架，使SAM3能够直接遵循自然语言指令进行分割，同时不牺牲其原有的概念驱动能力。

### 方法

提出指令感知的级联适应机制，逐步将表达性指令语义与SAM3现有的视觉语言表示对齐；设计跨越概念级、简单级和复杂级的结构化指令分类法；开发可扩展的数据引擎构建多样化指令-掩码对数据集。

### 主要发现

实验表明SAM3-I具有吸引人的性能，证明SAM3可以有效地扩展为遵循自然语言指令，同时保持其强大的概念基础能力。

### 结论

SAM3-I成功扩展了SAM3的功能，使其能够遵循自然语言指令进行分割，同时保留概念驱动能力。研究团队开源了SAM3-I并提供实用的微调工作流程，使其能够适应特定领域的应用。

### 翻译

Segment Anything Model 3 (SAM3)通过可提示的概念分割推动了开放词汇分割的发展，使用户能够分割与给定概念对应的所有实例，通常用简短名词短语(NP)提示指定。虽然这标志着语言级概念首次在SAM家族中的集成，但现实世界的使用通常需要更丰富的表达，包括属性、空间关系、功能、动作、状态，甚至对实例的隐式推理。目前，SAM3依赖于外部多模态代理将复杂指令转换为NP，然后进行迭代掩码过滤。然而，这些NP级概念仍然过于粗糙，通常无法精确表示特定实例。在这项工作中，我们提出了SAM3-I，这是一个增强框架，统一了SAM家族内的概念级理解和指令级推理。SAM3-I引入了一种指令感知的级联适应机制，逐步将表达性指令语义与SAM3现有的视觉语言表示对齐，使其能够直接遵循指令进行分割，而不牺牲其原有的概念驱动能力。此外，我们设计了一个跨越概念级、简单级和复杂级的结构化指令分类法，并开发了一个可扩展的数据引擎来构建具有多样化指令-掩码对的数据集。实验表明SAM3-I具有吸引人的性能，证明SAM3可以有效地扩展为遵循自然语言指令，同时保持其强大的概念基础能力。我们开源了SAM3-I并提供实用的微调工作流程，使研究人员能够将其适应于特定领域的应用。源代码可在此处获取。


### 论文摘要

Segment Anything Model 3 (SAM3) has advanced open-vocabulary segmentation through promptable concept segmentation, allowing users to segment all instances corresponding to a given concept, typically specified with short noun-phrase (NP) prompts. While this marks the first integration of language-level concepts within the SAM family, real-world usage typically requires far richer expressions that include attributes, spatial relations, functionalities, actions, states, and even implicit reasoning over instances. Currently, SAM3 relies on external multi-modal agents to convert complex instructions into NPs and then conduct iterative mask filtering. However, these NP-level concepts remain overly coarse, often failing to precisely represent a specific instance. In this work, we present SAM3-I, an enhanced framework that unifies concept-level understanding and instruction-level reasoning within the SAM family. SAM3-I introduces an instruction-aware cascaded adaptation mechanism that progressively aligns expressive instruction semantics with SAM3's existing vision-language representations, enabling direct instruction-following segmentation without sacrificing its original concept-driven capabilities. Furthermore, we design a structured instruction taxonomy spanning concept, simple, and complex levels, and develop a scalable data engine to construct a dataset with diverse instruction-mask pairs. Experiments show that SAM3-I delivers appealing performance, demonstrating that SAM3 can be effectively extended to follow natural-language instructions while preserving its strong concept grounding. We open-source SAM3-I and provide practical fine-tuning workflows, enabling researchers to adapt it to domain-specific applications. The source code is available here.

---

## 62. COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence

**论文链接:** [http://arxiv.org/abs/2512.04563v1](http://arxiv.org/abs/2512.04563v1)

**作者:** Zefeng Zhang, Xiangzhao Hao, Hengzhu Tang, Zhenyu Zhang, Jiawei Sheng, Xiaodong Li, Zhenyang Li, Li Gao, Daiting Shi, Dawei Yin, Tingwen Liu

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种名为COOPER的统一多模态大语言模型(MLLM)，通过深度和分割作为辅助模态，并在两个阶段进行训练，以增强空间感知和推理能力。实验表明，该方法在空间推理任务上平均提高了百分之六点九一，同时保持了整体性能。仅训练辅助模态生成的变体在距离和大小估计上获得了百分之七点九二的提升。

### 背景

视觉空间推理对多模态大语言模型(MLLMs)理解物体属性和空间关系至关重要，但当前模型在三维感知推理方面仍有困难。现有方法通常将感知增强和推理增强孤立处理，要么通过添加深度和分割等辅助模态来增强感知能力，要么通过在空间VQA数据集上训练并应用强化学习来增强推理能力。

### 目的

研究统一的MLLM是否能够发展出增强空间感知的内在能力，并通过自适应交错推理实现更强的空间智能。

### 方法

提出COOPER，一个统一的MLLM，利用深度和分割作为辅助模态，并通过两个阶段的训练来获得辅助模态生成和自适应交错推理能力。

### 主要发现

COOPER在空间推理上平均提高了百分之六点九一，同时保持了整体性能。仅训练辅助模态生成的变体在距离和大小估计上也获得了百分之七点九二的提升，这表明学习生成辅助模态有助于内部化空间知识并加强空间理解。

### 结论

统一的MLLM可以通过辅助模态生成和自适应交错推理来增强空间感知和推理能力，实现更强的空间智能。

### 翻译

视觉空间推理对于使多模态大语言模型(MLLMs)能够理解物体属性和空间关系至关重要，然而当前模型在三维感知推理方面仍然存在困难。现有方法通常通过增强RGB输入并添加深度和分割等辅助模态来提升感知能力，或通过在空间VQA数据集上训练并应用强化学习来增强推理能力，因此将这两个方面孤立处理。在本工作中，我们研究统一的MLLM是否能够发展出增强空间感知的内在能力，并通过自适应交错推理实现更强的空间智能。我们提出COOPER，一个统一的MLLM，利用深度和分割作为辅助模态，并通过两个阶段的训练来获得辅助模态生成和自适应交错推理能力。COOPER在空间推理上平均实现了百分之六点九一的提升，同时保持了整体性能。此外，即使仅训练辅助模态生成的变体，在距离和大小估计上也获得了百分之七点九二的提升，这表明学习生成辅助模态有助于内部化空间知识并加强空间理解。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态大语言模型在视觉空间推理方面的不足，特别是在3D感知和空间关系理解方面的困难。这个问题在现实中非常重要，因为视觉空间推理是机器人、自动驾驶和AR/VR等应用的核心基础，这些领域需要模型准确理解物体间的空间关系和3D结构。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有方法的局限性：感知增强方法和推理增强方法通常被独立处理，缺乏有机结合。他们借鉴了统一多模态大语言模型(如BAGEL)的架构，并设计了一个两阶段训练方法：第一阶段让模型生成辅助模态(深度图和分割图)，第二阶段训练模型自适应地决定何时使用这些模态进行推理。这种方法将感知和推理在统一框架下协同工作，而不是将它们视为独立部分。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是在统一模型中协同感知和推理，让模型既能生成辅助模态增强感知，又能自适应地选择何时使用这些模态进行推理。整体实现分为两个阶段：第一阶段将深度图和分割图转换为RGB伪图像，训练模型生成这些模态；第二阶段使用监督微调强化模型的自适应选择能力，再通过协作感知-推理奖励(CPR奖励)进行强化学习，优化模型的空间推理行为，使其能够根据任务需求灵活切换感知和推理模式。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出交错推理范式，协同统一感知和推理；2)设计两阶段训练流程，先学习生成辅助模态，再优化自适应推理能力；3)引入协作感知-推理奖励机制。相比之前工作，COOPER将感知和推理在统一模型中协同工作，而非孤立处理，让模型能够根据任务需求自适应选择何时使用感知增强或推理增强，这是之前方法不具备的。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'COOPER通过在统一模型中协同感知和推理，使多模态大语言模型能够自适应地生成和利用辅助视觉模态，显著提升了空间推理能力。'}


### 论文摘要

Visual Spatial Reasoning is crucial for enabling Multimodal Large Language Models (MLLMs) to understand object properties and spatial relationships, yet current models still struggle with 3D-aware reasoning. Existing approaches typically enhance either perception, by augmenting RGB inputs with auxiliary modalities such as depth and segmentation, or reasoning, by training on spatial VQA datasets and applying reinforcement learning, and thus treat these two aspects in isolation. In this work, we investigate whether a unified MLLM can develop an intrinsic ability to enhance spatial perception and, through adaptive interleaved reasoning, achieve stronger spatial intelligence. We propose \textbf{COOPER}, a unified MLLM that leverages depth and segmentation as auxiliary modalities and is trained in two stages to acquire auxiliary modality generation and adaptive, interleaved reasoning capabilities. COOPER achieves an average \textbf{6.91\%} improvement in spatial reasoning while maintaining general performance. Moreover, even a variant trained only for auxiliary modality generation attains a \textbf{7.92\%} gain on distance and size estimation, suggesting that learning to generate auxiliary modalities helps internalize spatial knowledge and strengthen spatial understanding.

---

## 63. The Geometry of Benchmarks: A New Path Toward AGI

**论文链接:** [http://arxiv.org/abs/2512.04276v1](http://arxiv.org/abs/2512.04276v1)

**作者:** Przemyslaw Chojecki

**发布时间:** 2025-12-03

### GPT解析

### 总结

该论文引入了一个几何框架，将AI代理的心理测量测试视为模空间中的点，通过能力泛函描述代理性能，并提出了自主AI量表、测试套件模空间和生成器-验证器-更新器(GVU)算子，以评估和推动AI向通用人工智能的发展。

### 背景

基准测试是评估人工智能进展的主要工具，但当前实践仅在孤立的测试套件上评估模型，很少提供关于AI系统通用性或自主自我改进能力的指导。

### 目的

引入一个几何框架，将所有AI代理的心理测量测试视为结构模空间中的点，通过该空间上的能力泛函描述代理性能，以更好地理解和评估AI系统的自主性和自我改进能力。

### 方法

定义自主AI(AAI)量表作为基于可测量性能的自主性层次结构；构建测试套件的模空间，识别在代理排序和能力推断层面不可区分的基准测试等价类；引入生成器-验证器-更新器(GVU)算子，涵盖强化学习、自我博弈等方法；定义自我改进系数κ作为能力泛函沿诱导流的李导数。

### 主要发现

几何框架提供了确定性结果：密集的测试套件足以证明在任务空间整个区域上的性能；生成和验证的联合噪声的方差不等式提供了自我改进系数κ>0的充分条件。

### 结论

向通用人工智能(AGI)的进展最好理解为基准测试模上的流动，这种流动由GVU动力学驱动，而非由单个排行榜的分数驱动。

### 翻译

基准测试是评估人工智能进展的主要工具，然而当前实践是在孤立的测试套件上评估模型，并且对于推理通用性或自主自我改进方面提供的指导很少。在此，我们引入了一个几何框架，其中所有AI代理的心理测量测试都被视为结构模空间中的点，代理性能通过该空间上的能力泛函来描述。首先，我们定义了一个自主AI(AAI)量表，这是一个基于在跨越任务家族(例如推理、规划、工具使用和长时程控制)的测试套件上的可测量性能的卡尔达舍夫式自主性层次结构。其次，我们构建了测试套件的模空间，识别出在代理排序和能力推断层面不可区分的基准测试的等价类。这种几何结构提供了确定性结果：密集的测试套件足以证明在任务空间整个区域上的性能。第三，我们引入了一个通用的生成器-验证器-更新器(GVU)算子，它概括了强化学习、自我博弈、辩论和基于验证器的微调作为特殊情况，并且我们将自我改进系数κ定义为能力泛函沿诱导流的李导数。关于生成和验证的联合噪声的方差不等式提供了κ>0的充分条件。我们的结果表明，向通用人工智能(AGI)的进展最好理解为基准测试模上的流动，由GVU动力学驱动，而非由单个排行榜的分数驱动。


### 论文摘要

Benchmarks are the primary tool for assessing progress in artificial intelligence (AI), yet current practice evaluates models on isolated test suites and provides little guidance for reasoning about generality or autonomous self-improvement. Here we introduce a geometric framework in which all psychometric batteries for AI agents are treated as points in a structured moduli space, and agent performance is described by capability functionals over this space. First, we define an Autonomous AI (AAI) Scale, a Kardashev-style hierarchy of autonomy grounded in measurable performance on batteries spanning families of tasks (for example reasoning, planning, tool use and long-horizon control). Second, we construct a moduli space of batteries, identifying equivalence classes of benchmarks that are indistinguishable at the level of agent orderings and capability inferences. This geometry yields determinacy results: dense families of batteries suffice to certify performance on entire regions of task space. Third, we introduce a general Generator-Verifier-Updater (GVU) operator that subsumes reinforcement learning, self-play, debate and verifier-based fine-tuning as special cases, and we define a self-improvement coefficient $κ$ as the Lie derivative of a capability functional along the induced flow. A variance inequality on the combined noise of generation and verification provides sufficient conditions for $κ> 0$. Our results suggest that progress toward artificial general intelligence (AGI) is best understood as a flow on moduli of benchmarks, driven by GVU dynamics rather than by scores on individual leaderboards.

---

## 64. CRAFT-E: A Neuro-Symbolic Framework for Embodied Affordance Grounding

**论文链接:** [http://arxiv.org/abs/2512.04231v1](http://arxiv.org/abs/2512.04231v1)

**作者:** Zhou Chen, Joe Lin, Carson Bulgin, Sathyanarayanan N. Aakur

**发布时间:** 2025-12-03

**备注:** 20 pages. 3 figures, 4 tables. Under Review

### GPT解析

### 总结

CRAFT-E是一种模块化神经符号框架，结合了结构化知识图谱、视觉语言对齐和基于能量的抓取推理，实现了可解释的物体选择功能，在辅助机器人系统中取得了有竞争力的性能。

### 背景

辅助机器人在非结构化环境中运行需要理解物体功能，现有方法依赖黑盒模型或固定功能标签，限制了透明度、可控性和可靠性。

### 目的

开发一个能够将语言动作查询与具有所需功能且可物理抓取的物体相关联的系统，提高辅助机器人决策的可解释性和可靠性。

### 方法

引入CRAFT-E框架，整合结构化的动词-属性-物体知识图谱、视觉语言对齐和基于能量的抓取推理，生成可解释的定位路径，并将抓取可行性作为功能推断的组成部分。

### 主要发现

构建了统一注释的基准数据集，在静态场景、基于ImageNet的功能检索和涉及20个动词39个物体的真实世界试验中取得有竞争力性能，系统在感知噪声下保持稳健并提供组件级诊断。

### 结论

通过结合符号推理与具身感知，CRAFT-E为端到端模型提供了可解释和可替代的方案，支持辅助机器人系统中的可信决策。

### 翻译

在非结构化环境中运行的辅助机器人不仅需要理解物体是什么，还需要理解它们的用途。这需要将基于语言的动作查询与既具有所需功能又能被物理抓取的物体相关联。现有方法通常依赖黑盒模型或固定的功能标签，限制了面向人类应用的透明度、可控性和可靠性。我们引入CRAFT-E，这是一个模块化神经符号框架，它结合了结构化的动词-属性-物体知识图谱、视觉语言对齐和基于能量的抓取推理。该系统生成可解释的定位路径，揭示影响物体选择的因素，并将抓取可行性作为功能推断的组成部分。我们进一步构建了一个包含动词-物体兼容性、分割和抓取候选者的统一注释的基准数据集，并在物理机器人上部署了完整流程。CRAFT-E在静态场景、基于ImageNet的功能检索以及涉及20个动词和39个物体的真实世界试验中取得了有竞争力的性能。该框架在感知噪声下保持稳健，并提供透明、组件级别的诊断。通过将符号推理与具身感知相结合，CRAFT-E为基于功能定位的物体选择提供了端到端模型的可解释和可定制替代方案，支持辅助机器人系统中的可信决策。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "论文主要解决辅助机器人在非结构化环境中如何理解物体功能属性（即'可供性'）的问题，而不仅仅是识别物体是什么。这个问题很重要，因为辅助机器人需要在家庭、诊所等开放环境中工作，这些环境中的物体和任务无法完全预先指定。机器人必须理解物体能用来做什么，才能正确响应用户请求（如'给我写字的东西'）。当前方法依赖黑盒模型或固定标签，限制了透明度和可靠性，而辅助机器人的决策必须让用户能理解、可定制且可信，这对安全可靠的人机交互至关重要。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者思考的关键是机器人需要超越简单识别物体类别，而要推理物体的功能如何支持用户目标。他们注意到现有的大型语言模型和视觉-语言模型虽能开放词汇推理，但决策过程不透明且难以解释。作者借鉴了神经符号模型工作，将大型语言模型作为外部知识源提供结构化常识推理；参考了机器人抓取和操作研究，特别是几何可供性和基于部件的方法；整合了CLIP等视觉-语言模型进行视觉-语言对齐；并研究了知识图谱如ConceptNet，但发现其对于可供性推理存在局限性，因此设计了专门针对可供性的知识结构。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是创建一个模块化的神经符号框架，结合结构化知识图谱与视觉-语言对齐，使机器人能根据物体功能而非仅类别进行选择。整体流程：1)输入场景处理，预测可抓取区域和物体分割；2)使用大型语言模型构建可供性知识库，生成与动词相关的属性并评估物体匹配度；3)构建从动词到物体的两跳知识图谱；4)通过CLIP等模型将符号假设与视觉证据对齐；5)评估每个物体的抓取可行性；6)通过能量公式平衡功能合理性、视觉匹配和抓取可行性，选择最佳物体。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)提出CRAFT-E神经符号框架，整合知识图谱、视觉对齐和抓取推理；2)创建新的功能定位基准数据集，包含统一注释；3)开发真实世界机器人测试平台，评估完整感知-动作管道；4)提供可解释推理模块，构建从动词到物体的定位路径。不同之处：相比仅依赖视觉上下文或固定类别的方法，CRAFT-E构建专门针对可供性的知识库；区别于传统几何可供性方法，专注于功能而非操作方式；不同于端到端系统，采用模块化架构提供透明决策；区别于通用知识图谱，强调可操作且视觉验证的属性。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'CRAFT-E通过整合神经感知与符号推理，为辅助机器人提供了一种透明、可解释且可靠的框架，使其能够根据功能需求而非仅物体类别来识别和抓取适合的物体。'}


### 论文摘要

Assistive robots operating in unstructured environments must understand not only what objects are, but what they can be used for. This requires grounding language-based action queries to objects that both afford the requested function and can be physically retrieved. Existing approaches often rely on black-box models or fixed affordance labels, limiting transparency, controllability, and reliability for human-facing applications. We introduce CRAFT-E, a modular neuro-symbolic framework that composes a structured verb-property-object knowledge graph with visual-language alignment and energy-based grasp reasoning. The system generates interpretable grounding paths that expose the factors influencing object selection and incorporates grasp feasibility as an integral part of affordance inference. We further construct a benchmark dataset with unified annotations for verb-object compatibility, segmentation, and grasp candidates, and deploy the full pipeline on a physical robot. CRAFT-E achieves competitive performance in static scenes, ImageNet-based functional retrieval, and real-world trials involving 20 verbs and 39 objects. The framework remains robust under perceptual noise and provides transparent, component-level diagnostics. By coupling symbolic reasoning with embodied perception, CRAFT-E offers an interpretable and customizable alternative to end-to-end models for affordance-grounded object selection, supporting trustworthy decision-making in assistive robotic systems.

---

## 65. Look Around and Pay Attention: Multi-camera Point Tracking Reimagined with Transformers

**论文链接:** [http://arxiv.org/abs/2512.04213v1](http://arxiv.org/abs/2512.04213v1)

**作者:** Bishoy Galoaa, Xiangyu Bai, Shayda Moezzi, Utsav Nandi, Sai Siddhartha Vivek Dhir Rangoju, Somaieh Amraee, Sarah Ostadabbas

**发布时间:** 2025-12-03

### GPT解析

### 总结

论文提出了LAPA（环顾四周并关注注意力），一种基于Transformer的端到端多摄像头点跟踪架构，结合了基于外观的匹配与几何约束，在具有挑战性的场景中显著优于现有方法。

### 背景

传统方法将检测、关联和跟踪解耦，导致在具有挑战性的场景中出现错误传播和时间不一致性。多摄像头点跟踪面临复杂运动和遮挡等挑战。

### 目的

解决传统方法中的错误传播和时间一致性问题，提高多摄像头点跟踪的准确性，特别是在复杂运动和遮挡场景下。

### 方法

使用注意力机制跨视图和时间进行联合推理；通过结合几何先验的跨视图注意力机制建立软对应关系；不依赖经典三角测量，而是通过注意力加权的聚合构建3D点表示；使用Transformer解码器保持时间一致性，建模长程依赖关系；通过扩展遮挡保持身份。

### 主要发现

在TAPVid-3D-MC数据集上达到37.5%的APD；在PointOdyssey-MC数据集上达到90.3%的APD；在具有复杂运动和遮挡的场景中表现特别出色。

### 结论

统一方法显著优于现有方法，代码已公开在GitHub上。

### 翻译

本文提出了LAPA（环顾四周并关注注意力），一种新颖的基于Transformer的端到端多摄像头点跟踪架构，将基于外观的匹配与几何约束相结合。传统流程将检测、关联和跟踪解耦，导致在具有挑战性的场景中出现错误传播和时间不一致。LAPA通过利用注意力机制跨视图和时间进行联合推理，通过结合几何先验的跨视图注意力机制建立软对应关系来解决这些限制。我们不依赖经典三角测量，而是通过注意力加权的聚合构建3D点表示， inherently适应不确定性和部分观测。通过建模长程依赖关系的Transformer解码器进一步保持时间一致性，通过扩展遮挡保持身份。在具有挑战性的数据集上的大量实验，包括我们新创建的多摄像头（MC）版本的TAPVid-3D全景和PointOdyssey，证明我们的统一方法显著优于现有方法，在TAPVid-3D-MC上达到37.5%的APD，在PointOdyssey-MC上达到90.3%的APD，特别在具有复杂运动和遮挡的场景中表现出色。代码可在https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-获取。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多摄像头点跟踪问题，即如何从多个同步摄像头视图中跟踪特定点的3D轨迹，保持点身份一致性，即使在部分遮挡或某些摄像头视野外的情况下。这个问题很重要，因为单一摄像头方法在目标被遮挡或离开视野时完全失效，而许多应用如机器人操作、监控和体育分析需要精确、时间一致的多视角点跟踪，传统方法要么只能处理单一摄像头，要么只能处理物体级别跟踪，缺乏点级精度。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别多摄像头点跟踪是被忽视的研究领域，受transformer在单视角点跟踪中成功的启发，但发现它们无法处理遮挡。意识到多摄像头互补视角可解决遮挡问题，核心洞察是3D空间中的空间邻近性比外观匹配更可靠的关联线索。借鉴了CoTracker等单摄像头点跟踪方法用于初始2D跟踪，借鉴了多视角几何原理如极线约束和SfM约束，借鉴了transformer注意力机制但创新性地应用于3D空间，借鉴了EpiTransformer等方法的几何约束思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是使用基于距离的体积注意力机制结合几何先验建立跨视图软对应关系，通过注意力加权聚合构建3D点表示处理不确定性，使用transformer解码器建模长程依赖保持时间一致性。整体流程：1)使用CoTracker和ViT进行2D点跟踪和特征提取；2)创建3D体积网格，投影到各视图，计算基于距离的几何注意力，用视图特定特征填充3D体积；3)通过轨道查询机制保持时间一致性，整合多种特征，用神经网络预测3D坐标；4)使用结合重建损失、投影损失和注意力损失的多目标损失函数训练。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)首个端到端transformer架构专门用于多摄像头点跟踪；2)基于距离的体积注意力机制整合几何约束；3)使用软对应关系表示不确定性；4)直接在3D空间推理空间关系；5)创建首个多摄像头点跟踪评估框架。不同之处：单摄像头方法无法处理遮挡，多摄像头物体跟踪缺乏点级精度，多视角对应方法没有时间跟踪，传统方法分离检测、关联和跟踪导致误差传播，LAPA统一处理所有步骤并实现端到端优化。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LAPA提出了首个端到端的transformer架构，通过基于距离的体积注意力机制整合几何约束与外观信息，实现了在复杂遮挡和多视角场景下的鲁棒3D点跟踪，显著超越了现有方法的性能，并为多摄像头点跟踪这一新兴研究领域建立了基准和评估框架。'}


### 论文摘要

This paper presents LAPA (Look Around and Pay Attention), a novel end-to-end transformer-based architecture for multi-camera point tracking that integrates appearance-based matching with geometric constraints. Traditional pipelines decouple detection, association, and tracking, leading to error propagation and temporal inconsistency in challenging scenarios. LAPA addresses these limitations by leveraging attention mechanisms to jointly reason across views and time, establishing soft correspondences through a cross-view attention mechanism enhanced with geometric priors. Instead of relying on classical triangulation, we construct 3D point representations via attention-weighted aggregation, inherently accommodating uncertainty and partial observations. Temporal consistency is further maintained through a transformer decoder that models long-range dependencies, preserving identities through extended occlusions. Extensive experiments on challenging datasets, including our newly created multi-camera (MC) versions of TAPVid-3D panoptic and PointOdyssey, demonstrate that our unified approach significantly outperforms existing methods, achieving 37.5% APD on TAPVid-3D-MC and 90.3% APD on PointOdyssey-MC, particularly excelling in scenarios with complex motions and occlusions. Code is available at https://github.com/ostadabbas/Look-Around-and-Pay-Attention-LAPA-

---

## 66. Resolving the terrestrial planet-forming region of HD 172555 with ALMA: I. Post-impact dust distribution

**论文链接:** [http://arxiv.org/abs/2512.04154v1](http://arxiv.org/abs/2512.04154v1)

**作者:** Zoe Roumeliotis, Luca Matrà, Grant M. Kennedy, Sebastian Marino, Kate Y. L. Su, David J. Wilner, Mark C. Wyatt, Alan P. Jackson

**发布时间:** 2025-12-03

**备注:** 14 pages, 6 figures, accepted by A&A

### GPT解析

### 总结

行星胚胎间的巨大撞击会在恒星类地区域形成碎片盘，研究使用ALMA观测揭示了HD 172555系统中毫米级尘埃颗粒的分布特征。

### 背景

巨大撞击是类地行星形成过程中的自然步骤，预计会产生气体和尘埃碎片，理解这些碎片对于行星碰撞模型至关重要。

### 目的

首次揭示HD 172555巨大撞击碎片盘中毫米级颗粒的分布，并研究其与其他波长观测的尘埃和气体分布的关系。

### 方法

使用ALMA 0.87毫米观测，分辨率约80毫角秒；通过建模干涉测量可见度获取磁盘空间特性；与不同波长的观测数据进行比较；进行辐射转移建模和SED建模。

### 主要发现

探测到倾斜磁盘的恒星和尘埃发射，延伸至约9天文单位；尘埃分布没有显著不对称性；毫米级颗粒的磁盘面密度分布最可能在约5天文单位处达到峰值；小颗粒与毫米级颗粒之间存在径向向外偏移；毫米级颗粒的大小分布斜率比微米级颗粒更平缓，表明颗粒大小分布存在断点

### 结论

HD 172555系统的巨大撞击碎片盘中的尘埃分布特征支持碰撞演变模型，小颗粒的偏移可能是气体拖曳和辐射压力共同作用的结果。

### 翻译

行星胚胎之间的巨大撞击是类地行星形成过程中的自然步骤，预计会在恒星的类地区域形成温暖的碎片盘。理解巨大撞击中产生的气体和尘埃碎片对于理解和限制行星碰撞模型至关重要。我们使用新的ALMA 0.87毫米观测，首次揭示了HD 172555巨大撞击碎片盘中毫米级颗粒的分布，分辨率约为80毫角秒。我们通过建模干涉测量可见度来获取磁盘的基本空间特性，并将其与其他波长的磁盘尘埃和气体分布进行比较。我们探测到了来自倾斜磁盘的恒星和尘埃发射，延伸到距离中心恒星约9天文单位，并在天空平面上低至2.3天文单位，尘埃分布没有显著的不对称性。可见度的辐射转移建模表明，毫米级颗粒的磁盘面密度分布最可能在约5天文单位处达到峰值，而推断的宽度在当前信噪比下仍依赖于模型。我们强调了散射光观测所追踪的小颗粒与毫米级颗粒之间的径向向外偏移，这可能足够大的气体密度下气体拖曳和辐射压力的共同效应所解释。此外，SED建模表明，毫米级颗粒的大小分布斜率与碰撞演变的期望一致，并且比推断的微米级颗粒更平缓，这表明颗粒大小分布存在断点，并确认了小颗粒的过剩。


### 论文摘要

Giant impacts between planetary embryos are a natural step in the terrestrial planet formation process and are expected to create disks of warm debris in the terrestrial regions of their stars. Understanding the gas and dust debris produced in giant impacts is vital for comprehending and constraining models of planetary collisions. We reveal the distribution of millimeter grains in the giant impact debris disk of HD 172555 for the first time, using new ALMA 0.87 mm observations at $\sim$80 mas (2.3 au) resolution. We modeled the interferometric visibilities to obtain basic spatial properties of the disk, and compared it to the disk's dust and gas distributions at other wavelengths. We detect the star and dust emission from an inclined disk out to $\sim$9 au and down to 2.3 au (on-sky) from the central star, with no significant asymmetry in the dust distribution. Radiative transfer modeling of the visibilities indicates the disk surface density distribution of millimeter grains most likely peaks around $\sim$5 au, while the width inferred remains model-dependent at the S/N of the data. We highlight an outward radial offset of the small grains traced by scattered light observations compared to the millimeter grains, which could be explained by the combined effect of gas drag and radiation pressure in the presence of large enough gas densities. Furthermore, SED modeling implies a size distribution slope for the millimeter grains consistent with the expectation of collisional evolution and flatter than inferred for the micron-sized grains, implying a break in the grain size distribution and confirming an overabundance of small grains.

---

## 67. The Universal Weight Subspace Hypothesis

**论文链接:** [http://arxiv.org/abs/2512.05117v1](http://arxiv.org/abs/2512.05117v1)

**作者:** Prakhar Kaushik, Shravan Chaudhari, Ankit Vaidya, Rama Chellappa, Alan Yuille

**发布时间:** 2025-12-04

**备注:** 37 pages

### GPT解析

### 总结

该研究揭示了深度神经网络在不同任务训练后表现出相似的低维参数子空间，无论初始化方式、任务或领域如何，神经网络都会系统地收敛到共享的谱子空间。研究通过大规模实证分析发现了这些通用子空间，并对模型重用、多任务学习等领域有重要意义。

### 背景

深度神经网络在训练过程中可能存在一些尚未被充分理解的内在结构特性。随着模型规模的扩大，理解这些特性对于提高模型效率和减少计算资源变得尤为重要。

### 目的

探究深度神经网络在不同任务训练后是否表现出共同的参数子空间结构，以及这些结构是否与初始化方式、任务或领域无关，从而揭示深度网络内部信息的组织方式。

### 方法

研究对1100多个模型（包括500个Mistral-7B LoRAs、500个视觉Transformer和50个LLaMA-8B模型）进行了模态谱分析，并对各种架构在广泛任务和数据集上训练的权重矩阵应用了谱分解技术，以识别稀疏的联合子空间。

### 主要发现

1. 深度神经网络在不同任务训练后表现出相似的低维参数子空间；2. 无论初始化方式、任务或领域如何，神经网络都会系统地收敛到共享的谱子空间；3. 只需几个主方向就能捕捉到大部分方差的通用子空间；4. 存在稀疏的联合子空间在不同任务和数据集的共享架构中被一致利用。

### 结论

深度神经网络内部存在一种通用的、任务无关的参数子空间结构，这一发现为理解深度网络的信息组织提供了新见解，并可能对模型重用、多任务学习、模型融合以及开发高效算法产生重要影响，同时有助于减少大规模神经模型的碳足迹。

### 翻译

我们表明，在多样化任务上训练的深度神经网络展现出显著相似的低维参数子空间。我们首次提供了大规模实证证据，证明神经网络无论初始化方式、任务或领域如何，都会系统地收敛到共享的谱子空间。通过对1100多个模型（包括500个Mistral-7B LoRAs、500个视觉Transformer和50个LLaMA-8B模型）进行模态谱分析，我们识别出通用子空间，这些子空间只需几个主方向就能捕捉大部分方差。通过对各种架构在广泛任务和数据集上训练的权重矩阵应用谱分解技术，我们识别出稀疏的联合子空间，这些子空间在不同任务和数据集的共享架构中被一致利用。我们的发现为深度网络内部信息的内在组织提供了新见解，并提出了关于无需大量数据和计算资源就能发现这些通用子空间的重要问题。此外，这种内在结构对模型重用、多任务学习、模型融合以及开发和训练推理高效算法有重要意义，可能减少大规模神经模型的碳足迹。


### 论文摘要

We show that deep neural networks trained across diverse tasks exhibit remarkably similar low-dimensional parametric subspaces. We provide the first large-scale empirical evidence that demonstrates that neural networks systematically converge to shared spectral subspaces regardless of initialization, task, or domain. Through mode-wise spectral analysis of over 1100 models - including 500 Mistral-7B LoRAs, 500 Vision Transformers, and 50 LLaMA-8B models - we identify universal subspaces capturing majority variance in just a few principal directions. By applying spectral decomposition techniques to the weight matrices of various architectures trained on a wide range of tasks and datasets, we identify sparse, joint subspaces that are consistently exploited, within shared architectures across diverse tasks and datasets. Our findings offer new insights into the intrinsic organization of information within deep networks and raise important questions about the possibility of discovering these universal subspaces without the need for extensive data and computational resources. Furthermore, this inherent structure has significant implications for model reusability, multi-task learning, model merging, and the development of training and inference-efficient algorithms, potentially reducing the carbon footprint of large-scale neural models.

---

## 68. DraCo: Draft as CoT for Text-to-Image Preview and Rare Concept Generation

**论文链接:** [http://arxiv.org/abs/2512.05112v1](http://arxiv.org/abs/2512.05112v1)

**作者:** Dongzhi Jiang, Renrui Zhang, Haodong Li, Zhuofan Zong, Ziyu Guo, Jun He, Claire Guo, Junyan Ye, Rongyao Fang, Weijia Li, Rui Liu, Hongsheng Li

**发布时间:** 2025-12-04

**备注:** Project Page: https://github.com/CaraJ7/DraCo

### GPT解析

### 总结

本文提出了Draft-as-CoT (DraCo)方法，一种新颖的交错推理范式，通过同时利用文本和视觉内容改进图像生成的规划和验证过程。

### 背景

现有的统一多模态大语言模型在文本到图像生成方面展示了能力，但现有方法存在局限性，要么将模型仅作为独立生成器，要么依赖抽象文本规划。

### 目的

开发一种能够同时利用文本和视觉内容进行规划和验证的方法，解决文本规划的粗粒度性质和生成罕见属性组合的困难。

### 方法

DraCo首先生成低分辨率草稿图像作为预览，提供具体和结构化的视觉规划；然后利用模型的理解能力验证草稿与输入提示的语义对齐，并通过选择性校正和超分辨率进行细化；同时构建了DraCo-240K数据集和DraCo-CFG策略支持训练。

### 主要发现

DraCo在GenEval上提升+8%，在Imagine-Bench上提升+0.91，在GenEval++上提升+3%，显著优于直接生成和其他由CoT增强的生成方法。

### 结论

DraCo通过交错推理范式有效解决了现有多模态大语言模型在图像生成中的局限性，特别是在规划和验证方面。

### 翻译

最近的统一多模态大语言模型(MLLMs)展示了令人印象深刻的能力，融入了思维链(CoT)推理以增强文本到图像生成。然而，现有方法仍然有限，要么将模型仅作为独立生成器，要么依赖抽象文本规划。为此，我们提出了Draft-as-CoT (DraCo)，一种新颖的交错推理范式，充分利用CoT中的文本和视觉内容以实现更好的规划和验证。我们的方法首先生成低分辨率草稿图像作为预览，提供更具体和结构化的视觉规划和指导。然后，我们利用模型固有的理解能力验证草稿与输入提示之间的潜在语义错位，并通过选择性校正和超分辨率进行细化。通过这种方式，我们的方法解决了两个基本挑战：文本规划的粗粒度性质以及生成罕见属性组合的困难。为支持训练，我们构建了DraCo-240K，旨在增强三个原子能力，涵盖一般校正、实例操作和布局重组。在DraCo-CFG的支持下，这是一种专门用于交错推理的分类器无引导(CFG)策略，DraCo在GenEval上实现了显著提升(+8%)，在Imagine-Bench上提升了+0.91，在GenEval++上提升了+3%，显著优于直接生成和其他由CoT增强的生成方法。


### 论文摘要

Recent unified multimodal large language models (MLLMs) have shown impressive capabilities, incorporating chain-of-thought (CoT) reasoning for enhanced text-to-image generation. However, existing approaches remain limited, either treating the model merely as a standalone generator or relying on abstract textual planning. To this end, we propose Draft-as-CoT (DraCo), a novel interleaved reasoning paradigm that fully leverages both textual and visual contents in CoT for better planning and verification. Our method first generates a low-resolution draft image as preview, providing more concrete and structural visual planning and guidance. Then, we employ the model's inherent understanding capability to verify potential semantic misalignments between the draft and input prompt, and performs refinement through selective corrections with super-resolution. In this way, our approach addresses two fundamental challenges: the coarse-grained nature of textual planning and the difficulty in generating rare attribute combinations. To support training, we curate DraCo-240K, aiming to enhance three atomic capabilities spanning general correction, instance manipulation, and layout reorganization. Supported by DraCo-CFG, a specialized classifier-free guidance (CFG) strategy for interleaved reasoning, DraCo achieves a tremendous increase on GenEval (+8%), Imagine-Bench (+0.91), and GenEval++ (+3%), significantly outperforming direct generation and other generation methods empowered by CoT.

---

## 69. STARE-VLA: Progressive Stage-Aware Reinforcement for Fine-Tuning Vision-Language-Action Models

**论文链接:** [http://arxiv.org/abs/2512.05107v1](http://arxiv.org/abs/2512.05107v1)

**作者:** Feng Xu, Guangyao Zhai, Xin Kong, Tingzhong Fu, Daniel F. N. Gordon, Xueli An, Benjamin Busam

**发布时间:** 2025-12-04

### GPT解析

### 总结

论文提出了STARE模块及其衍生方法STA-TPO和STA-PPO，以及IPI微调流程，用于改进VLA模型在机器人操作任务中的表现。

### 背景

现有VLA模型方法将长时程动作视为语言序列并使用轨迹级别优化方法，导致粗略信用分配和不稳定训练。动作轨迹通过因果链式阶段进展，各阶段学习难度不同，与语言特性不同。

### 目的

解决现有方法中的粗略信用分配和不稳定训练问题，通过阶段感知优化提高VLA模型在机器人操作中的性能。

### 方法

提出STARE模块，将长时程动作轨迹分解为语义上有意义的阶段并提供密集、可解释的强化信号；集成STARE到TPO和PPO形成STA-TPO和STA-PPO；提出IPI微调流程，包括监督微调作为初始化。

### 主要发现

STARE能有效分解长时程动作轨迹；STA-TPO和STA-PPO在机器人操作任务中表现优异；IPI微调流程提高VLA模型动作准确性；在SimplerEnv和ManiSkill3上分别达到98.0%和96.4%的最先进成功率。

### 结论

阶段感知的优化方法显著提高了VLA模型在机器人操作任务中的性能，为解决长时程动作学习中的信用分配问题提供了新思路。

### 翻译

由大型语言模型和基于强化学习的微调驱动的视觉-语言-动作（VLA）模型的最新进展，在机器人操作方面显示出显著进展。现有方法通常将长时程动作视为语言序列，并应用轨迹级别的优化方法，如轨迹级偏好优化（TPO）或近端策略优化（PPO），这导致粗略的信用分配和不稳定的训练。然而，与语言不同（尽管句子顺序灵活，但保持统一的语义意义），动作轨迹通过具有不同学习难度的因果链式阶段进展。这促使了渐进式阶段优化。因此，我们提出了STARE（阶段感知强化）模块，它将长时程动作轨迹分解为语义上有意义的阶段，并提供密集、可解释且与阶段对齐的强化信号。将STARE集成到TPO和PPO中，我们分别获得了用于离线阶段偏好和在线阶段内交互的STA-TPO和STA-PPO。进一步基于监督微调作为初始化，我们提出了IPI（模仿->偏好->交互）顺序微调流程，以提高VLA模型中的动作准确性。在SimplerEnv和ManiSkill3上的实验证明了显著的提升，在SimplerEnv任务上达到了98.0%的最先进成功率，在ManiSkill3任务上达到了96.4%的最先进成功率。


### 论文摘要

Recent advances in Vision-Language-Action (VLA) models, powered by large language models and reinforcement learning-based fine-tuning, have shown remarkable progress in robotic manipulation. Existing methods often treat long-horizon actions as linguistic sequences and apply trajectory-level optimization methods such as Trajectory-wise Preference Optimization (TPO) or Proximal Policy Optimization (PPO), leading to coarse credit assignment and unstable training. However, unlike language, where a unified semantic meaning is preserved despite flexible sentence order, action trajectories progress through causally chained stages with different learning difficulties. This motivates progressive stage optimization. Thereby, we present Stage-Aware Reinforcement (STARE), a module that decomposes a long-horizon action trajectory into semantically meaningful stages and provides dense, interpretable, and stage-aligned reinforcement signals. Integrating STARE into TPO and PPO, we yield Stage-Aware TPO (STA-TPO) and Stage-Aware PPO (STA-PPO) for offline stage-wise preference and online intra-stage interaction, respectively. Further building on supervised fine-tuning as initialization, we propose the Imitation -> Preference -> Interaction (IPI), a serial fine-tuning pipeline for improving action accuracy in VLA models. Experiments on SimplerEnv and ManiSkill3 demonstrate substantial gains, achieving state-of-the-art success rates of 98.0 percent on SimplerEnv and 96.4 percent on ManiSkill3 tasks.

---

## 70. Foundations of Diffusion Models in General State Spaces: A Self-Contained Introduction

**论文链接:** [http://arxiv.org/abs/2512.05092v1](http://arxiv.org/abs/2512.05092v1)

**作者:** Vincent Pauline, Tobias Höppe, Kirill Neklyudov, Alexander Tong, Stefan Bauer, Andrea Dittadi

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提供了一个关于一般状态空间上扩散模型的自包含入门指南，统一了连续域和离散/分类结构，建立了离散时间视角和连续时间极限之间的联系，推导了相关方程，并为不同层次的受众提供了分层呈现方式。

### 背景

扩散模型目前在生成建模中占据中心地位，但传统介绍通常假设欧几里得数据，很少阐明它们与离散状态类比的联系。

### 目的

为一般状态空间上的扩散提供统一的理论框架，将连续域和离散/分类结构统一在一个视角下，建立离散时间视角和连续时间极限之间的联系，并推导相关方程。

### 方法

开发了离散时间视角（通过马尔可夫核的前向加噪和学习到的反向动力学）及其连续时间极限（在R^d中的随机微分方程和有限字母表上的连续时间马尔可夫链），推导了相关的Fokker-Planck方程和主方程，并提供了共同的变分处理以得到支撑标准训练损失的ELBO。

### 主要发现

明确了前向损坏选择（连续空间中的高斯过程和离散空间中的结构化分类转移核）如何塑造反向动力学和ELBO，提供了适用于三种不同受众的分层呈现方式，并得出了现代扩散方法学的统一路线图。

### 结论

该研究提供了跨越连续域和离散序列的现代扩散方法的统一路线图，突出了紧凑的可重用证明、恒等式和核心理论原则集合。

### 翻译

虽然扩散模型目前在生成建模中占据中心地位，但入门教程通常假设欧几里得数据，很少阐明它们与离散状态类比的联系。本文是关于一般状态空间上扩散的自包含入门指南，将连续域和离散/分类结构统一在一个视角下。我们开发了离散时间视角（通过马尔可夫核的前向加噪和学习到的反向动力学）及其连续时间极限——在R^d中的随机微分方程和有限字母表上的连续时间马尔可夫链，并推导了相关的Fokker-Planck方程和主方程。共同的变分处理产生了支撑标准训练损失的ELBO。我们明确指出了前向损坏选择（连续空间中的高斯过程和离散空间中的结构化分类转移核（均匀、掩码/吸收等））如何塑造反向动力学和ELBO。呈现方式分为三层，面向三种受众：寻求自包含直观介绍的新手；希望获得全局理论综合的扩散实践者；以及寻找类比路径进入离散扩散的连续扩散专家。结果是跨越连续域和离散序列的现代扩散方法的统一路线图，突出了紧凑的可重用证明、恒等式和核心理论原则集合。


### 论文摘要

Although diffusion models now occupy a central place in generative modeling, introductory treatments commonly assume Euclidean data and seldom clarify their connection to discrete-state analogues. This article is a self-contained primer on diffusion over general state spaces, unifying continuous domains and discrete/categorical structures under one lens. We develop the discrete-time view (forward noising via Markov kernels and learned reverse dynamics) alongside its continuous-time limits -- stochastic differential equations (SDEs) in $\mathbb{R}^d$ and continuous-time Markov chains (CTMCs) on finite alphabets -- and derive the associated Fokker--Planck and master equations. A common variational treatment yields the ELBO that underpins standard training losses. We make explicit how forward corruption choices -- Gaussian processes in continuous spaces and structured categorical transition kernels (uniform, masking/absorbing and more) in discrete spaces -- shape reverse dynamics and the ELBO. The presentation is layered for three audiences: newcomers seeking a self-contained intuitive introduction; diffusion practitioners wanting a global theoretical synthesis; and continuous-diffusion experts looking for an analogy-first path into discrete diffusion. The result is a unified roadmap to modern diffusion methodology across continuous domains and discrete sequences, highlighting a compact set of reusable proofs, identities, and core theoretical principles.

---

## 71. Visual Reasoning Tracer: Object-Level Grounded Reasoning Benchmark

**论文链接:** [http://arxiv.org/abs/2512.05091v1](http://arxiv.org/abs/2512.05091v1)

**作者:** Haobo Yuan, Yueyi Sun, Yanwei Li, Tao Zhang, Xueqing Deng, Henghui Ding, Lu Qi, Anran Wang, Xiangtai Li, Ming-Hsuan Yang

**发布时间:** 2025-12-04

**备注:** Technical Report; Project Page: https://harboryuan.github.io/visual-reasoning-tracer

### GPT解析

### 总结

多模态大语言模型在视觉任务上表现优异，但推理过程不透明。作者提出视觉推理追踪任务及相关数据集和评估指标，发现现有模型难以追踪中间推理步骤，而专门训练的模型在此方面表现更好。

### 背景

多模态大语言模型在视觉定位和视觉问答等任务上取得了显著进展，但它们的推理过程不透明，只输出最终预测而不显示中间推理步骤。

### 目的

解决多模态大语言模型推理过程不透明的问题，引入视觉推理追踪任务，使模型能够明确展示中间推理步骤。

### 方法

提出视觉推理追踪(VRT)任务，要求模型定位目标对象并预测构成推理路径的中间对象。同时创建VRT-Bench基准、新的评估指标和VRT-80k训练数据集。

### 主要发现

现有模型虽然能产生正确的最终输出，但在追踪中间推理方面存在困难；而在VRT-80k上训练的模型在追踪推理路径方面取得了显著改进。

### 结论

通过引入视觉推理追踪任务及相关资源，可以提高多模态大语言模型的推理透明度，使其更接近人类智能的推理方式。

### 翻译

多模态大语言模型(MLLMs)的最新进展显著提高了视觉定位和视觉问答等任务上的性能。然而，这些模型的推理过程仍然不透明；它们通常只输出最终预测，而不显示导致结果的中间步骤或细粒度证据(如像素、位置)。这与人类智能形成对比，人类智能通过视觉推理链自然运作。为了解决这一限制，我们引入了视觉推理追踪(VRT)任务，该任务要求模型不仅要定位目标对象，还要明确预测构成推理路径的中间对象。为推进该领域研究，我们贡献了：(1)VRT-Bench，一个人工标注的用于评估视觉推理的基准；(2)一种用于评估推理轨迹质量的新指标；以及(3)VRT-80k，一个用于推理模型训练的大规模数据集。我们的实验表明，虽然现有模型通常能产生正确的最终输出，但在追踪中间推理方面存在困难。相比之下，在VRT-80k上训练的模型在追踪推理路径方面取得了显著改进。


### 论文摘要

Recent advances in Multimodal Large Language Models (MLLMs) have significantly improved performance on tasks such as visual grounding and visual question answering. However, the reasoning processes of these models remain largely opaque; they typically output only final predictions without revealing the intermediate steps or fine-grained evidence (e.g., pixels, locations) that lead to the result. This contrasts with human intelligence, which naturally operates through a chain of visual reasoning. To address this limitation, we introduce the Visual Reasoning Tracer (VRT) task, which requires models to not only localize the target object but also explicitly predict the intermediate objects that form the reasoning path. To advance research in this area, we contribute: (1) VRT-Bench, a human-annotated benchmark for evaluating visual reasoning; (2) a new metric for assessing the quality of reasoning traces; and (3) VRT-80k, a large-scale dataset for reasoning model training. Our experiments reveal that while existing models often produce the correct final output, they struggle to ground their intermediate reasoning. In contrast, models trained on VRT-80k achieve substantial improvements in tracing the reasoning path.

---

## 72. David vs. Goliath: Can Small Models Win Big with Agentic AI in Hardware Design?

**论文链接:** [http://arxiv.org/abs/2512.05073v1](http://arxiv.org/abs/2512.05073v1)

**作者:** Shashwat Shankar, Subhranshu Pandey, Innocent Dengkhw Mochahari, Bhabesh Mali, Animesh Basak Chowdhury, Sukanta Bhattacharjee, Chandan Karfa

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究探讨了在硬件设计中是否大型语言模型总是更好的选择，发现小型语言模型结合智能体AI框架可以以较低成本实现接近大型语言模型的性能，并为复杂设计任务提供高效自适应解决方案。

### 背景

大型语言模型(LLM)推理需要大量计算和能源，使特定领域任务变得昂贵且不可持续。随着基础模型不断扩展，作者质疑在硬件设计中是否'更大总是更好'。

### 目的

测试在硬件设计中，小型语言模型与智能体AI框架结合是否能替代大型语言模型，以降低成本同时保持性能。

### 方法

评估小型语言模型与精心设计的智能体AI框架在NVIDIA综合Verilog设计问题(CVDP)基准测试上的表现。智能体工作流程通过任务分解、迭代反馈和修正来工作。

### 主要发现

智能体工作流程能够以极低的成本实现接近大型语言模型的性能，同时为智能体创造学习机会。

### 结论

在复杂设计任务中，小型语言模型结合智能体AI框架可以提供高效、自适应的解决方案，挑战了'更大总是更好'的观点。

### 翻译

大型语言模型(LLM)推理需要大量计算和能源，使得特定领域任务变得昂贵且不可持续。随着基础模型不断扩展，我们质疑：在硬件设计中，更大总是更好吗？我们的工作通过评估小型语言模型与精心设计的智能体AI框架在NVIDIA综合Verilog设计问题(CVDP)基准测试上的表现来测试这一点。结果表明，智能体工作流程：通过任务分解、迭代反馈和修正-不仅能够以极低的成本解锁接近LLM的性能，而且为智能体创造学习机会，为复杂设计任务中的高效、自适应解决方案铺平了道路。


### 论文摘要

Large Language Model(LLM) inference demands massive compute and energy, making domain-specific tasks expensive and unsustainable. As foundation models keep scaling, we ask: Is bigger always better for hardware design? Our work tests this by evaluating Small Language Models coupled with a curated agentic AI framework on NVIDIA's Comprehensive Verilog Design Problems(CVDP) benchmark. Results show that agentic workflows: through task decomposition, iterative feedback, and correction - not only unlock near-LLM performance at a fraction of the cost but also create learning opportunities for agents, paving the way for efficient, adaptive solutions in complex design tasks.

---

## 73. Semantic-Guided Two-Stage GAN for Face Inpainting with Hybrid Perceptual Encoding

**论文链接:** [http://arxiv.org/abs/2512.05039v1](http://arxiv.org/abs/2512.05039v1)

**作者:** Abhigyan Bhattacharya, Hiranmoy Roy

**发布时间:** 2025-12-04

**备注:** Submitted for review CVPR-2025

### GPT解析

### 总结

这篇论文提出了一种新型的人脸图像修复方法，通过语义引导的层次化合成技术解决了现有方法在大不规则掩码下面临的问题，如边缘模糊、语义不一致和不真实的人脸结构。

### 背景

人脸图像修复旨在恢复人脸图像中缺失或损坏的区域，同时保持身份特征、结构一致性和真实感图像质量，这一任务专门为照片修复而设计。尽管深度生成模型最近取得了很大进展，但现有方法在大不规则掩码情况下存在诸多问题。

### 目的

提出一种新型架构，通过语义引导的层次化合成解决现有方法在大不规则掩码下面临的边缘模糊、语义不一致和不真实人脸结构等问题。

### 方法

论文提出的方法分为两个阶段：1) 结合使用CNN处理局部特征和Vision Transformers处理全局特征的技术，创建清晰详细的语义布局；2) 使用多模态纹理生成器，通过整合不同尺度的信息来细化这些布局，确保整体外观连贯一致。该架构通过动态注意力自然处理任意掩码配置，无需针对特定掩码进行训练。

### 主要发现

在CelebA-HQ和FFHQ两个数据集上的实验表明，该模型优于其他最先进的方法，在LPIPS、PSNR和SSIM等指标上有所改进。特别是在具有挑战性的大面积修复情况下，该方法能产生视觉上令人印象深刻的结果，并更好地保留语义信息。

### 结论

该论文提出的新型架构通过语义引导的层次化合成方法，有效解决了现有人脸图像修复方法在大不规则掩码下面临的问题，能够产生更好的视觉质量和语义一致性结果。

### 翻译

人脸图像修复旨在恢复人脸图像中缺失或损坏的区域，同时保持身份特征、结构一致性和真实感图像质量，这是一个专门为照片修复而设计的任务。尽管最近深度生成模型取得了许多进展，但现有方法在大不规则掩码情况下存在一些问题，通常在掩码区域边缘产生模糊纹理、语义不一致，或由于直接像素级合成方法和有限的面部先验利用而导致不真实的人脸结构。在本文中，我们提出了一种新型架构，通过语义引导的层次合成解决上述挑战。我们的方法首先采用一种基于意义组织和合成信息的方法，然后细化纹理。


### 论文摘要

Facial Image inpainting aim is to restore the missing or corrupted regions in face images while preserving identity, structural consistency and photorealistic image quality, a task specifically created for photo restoration. Though there are recent lot of advances in deep generative models, existing methods face problems with large irregular masks, often producing blurry textures on the edges of the masked region, semantic inconsistencies, or unconvincing facial structures due to direct pixel level synthesis approach and limited exploitation of facial priors. In this paper we propose a novel architecture, which address these above challenges through semantic-guided hierarchical synthesis. Our approach starts with a method that organizes and synthesizes information based on meaning, followed by refining the texture. This process gives clear insights into the facial structure before we move on to creating detailed images. In the first stage, we blend two techniques: one that focuses on local features with CNNs and global features with Vision Transformers. This helped us create clear and detailed semantic layouts. In the second stage, we use a Multi-Modal Texture Generator to refine these layouts by pulling in information from different scales, ensuring everything looks cohesive and consistent. The architecture naturally handles arbitrary mask configurations through dynamic attention without maskspecific training. Experiment on two datasets CelebA-HQ and FFHQ shows that our model outperforms other state-of-the-art methods, showing improvements in metrics like LPIPS, PSNR, and SSIM. It produces visually striking results with better semantic preservation, in challenging large-area inpainting situations.

---

## 74. 论文ID: 2512.05016v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05016v1.json'

---

## 75. Self-Supervised Learning for Transparent Object Depth Completion Using Depth from Non-Transparent Objects

**论文链接:** [http://arxiv.org/abs/2512.05006v1](http://arxiv.org/abs/2512.05006v1)

**作者:** Xianghui Fan, Zhaoyu Chen, Mengyang Pan, Anping Deng, Hang Yang

**发布时间:** 2025-12-04

**备注:** conference

### GPT解析

### 总结

本研究提出了一种新的自监督方法用于训练深度补全网络，解决了透明物体深度感知中标注数据成本高的问题。

### 背景

透明物体的感知是计算机视觉中的一个著名挑战。传统深度传感器由于光的折射和反射难以感知透明物体的深度。之前的研究通常使用神经网络来补全传感器获取的深度数据，但这种方法依赖于大量标注数据进行监督，而深度图的标注成本很高。

### 目的

开发一种自监督方法来训练深度补全网络，减少对大量标注数据的依赖，同时保持与监督方法相当的性能。

### 方法

提出一种新的自监督训练方法，在非透明区域模拟透明物体的深度缺陷，并利用原始深度图作为真实值进行监督。

### 主要发现

实验表明，该方法可以达到与监督方法相当的性能，并且在训练样本较少时，使用该方法进行预训练可以提高模型性能。

### 结论

该自监督方法有效解决了透明物体深度感知中标注数据成本高的问题，为实际应用提供了更高效、经济的解决方案。

### 翻译

透明物体的感知是计算机视觉中一个著名的挑战。传统深度传感器由于光的折射和反射难以感知透明物体的深度。先前的研究通常训练神经网络来补全传感器获取的深度，这种方法可以快速准确地获取透明物体的精确深度图。然而，之前的训练依赖于大量标注数据进行监督，而深度图的标注成本很高。为了应对这一挑战，我们提出了一种新的自监督方法来训练深度补全网络。我们的方法在非透明区域模拟透明物体的深度缺陷，并利用原始深度图作为真实值进行监督。实验证明，我们的方法达到了与监督方法相当的性能，并且当训练样本较少时，使用我们的方法进行预训练可以提高模型性能。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决透明物体的深度感知问题。传统深度传感器因光线折射和反射难以准确获取透明物体的深度信息。这个问题在现实中很重要，因为透明物体在日常生活和工业环境中很常见，而深度信息对计算机理解场景至关重要。此外，现有方法需要大量标注数据进行监督学习，但透明物体的深度图标注成本高且难以获取。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有监督学习方法的局限性，即需要大量标注数据。然后借鉴了自监督学习的思想，但发现现有自监督方法（如MAE）模拟的是全局深度缺失，而透明物体的深度缺失是局部化的。作者使用SAM模型获取透明和非透明物体的分割掩码，并在非透明物体区域模拟透明物体的深度缺失模式，通过形态学腐蚀操作保留边缘深度，更好地模拟真实情况。这种方法利用了原始深度图作为地面真实值，避免了对透明物体完整深度图的依赖。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过在非透明物体区域模拟透明物体的深度缺失模式，创建一个类似透明物体深度完成的任务，使模型能够学习到适用于透明物体深度完成的特征表示，而无需透明物体的完整深度图标注。整体流程包括：1)使用SAM模型获取分割掩码；2)对非透明物体掩码进行形态学腐蚀；3)生成掩码后的RGB和深度图作为输入；4)使用原始深度图作为地面真实值进行自监督训练；5)最后进行监督微调，专注于透明区域的深度恢复。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)首个专门针对透明物体深度完成任务的自监督方法；2)创新的掩码策略，通过形态学腐蚀操作保留边缘深度，更好地模拟透明物体的深度缺失模式；3)无需透明物体完整深度图标注，大大降低数据获取成本。相比之前的工作，我们的方法不依赖大量标注数据，而其他自监督方法使用全局随机掩码，与透明物体的局部深度缺失模式不符。此外，我们的方法在数据有限的情况下也能取得良好效果，预训练可以显著提升模型性能。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了一种自监督学习方法，通过在非透明物体区域模拟透明物体的深度缺失模式，使模型能够在无需透明物体完整深度图标注的情况下学习到有效的深度完成特征，显著降低了数据获取成本并提高了模型在小数据集上的性能。'}


### 论文摘要

The perception of transparent objects is one of the well-known challenges in computer vision. Conventional depth sensors have difficulty in sensing the depth of transparent objects due to refraction and reflection of light. Previous research has typically train a neural network to complete the depth acquired by the sensor, and this method can quickly and accurately acquire accurate depth maps of transparent objects. However, previous training relies on a large amount of annotation data for supervision, and the labeling of depth maps is costly. To tackle this challenge, we propose a new self-supervised method for training depth completion networks. Our method simulates the depth deficits of transparent objects within non-transparent regions and utilizes the original depth map as ground truth for supervision. Experiments demonstrate that our method achieves performance comparable to supervised approach, and pre-training with our method can improve the model performance when the training samples are small.

---

## 76. Reflection Removal through Efficient Adaptation of Diffusion Transformers

**论文链接:** [http://arxiv.org/abs/2512.05000v1](http://arxiv.org/abs/2512.05000v1)

**作者:** Daniyar Zakarin, Thiemo Wandel, Anton Obukhov, Dengxin Dai

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究提出了一种基于扩散变换器（DiT）的框架，用于单图像反射去除，通过利用基础扩散模型在恢复任务中的泛化能力，实现了高效且高质量的反射去除效果。

### 背景

现有的反射去除任务通常依赖于特定任务架构，而基础扩散模型在恢复任务中展现出强大的泛化能力。然而，缺乏多样性和真实感的数据限制了现有方法的性能。

### 目的

研究旨在探索如何利用预训练的扩散变换器基础模型，结合基于物理的合成数据，解决单图像反射去除问题，并提高方法的可扩展性和保真度。

### 方法

研究通过以下方法实现目标：1.重新利用预训练的DiT基础模型，通过条件化反射污染输入并引导生成干净的传输层；2.系统分析现有反射去除数据源的多样性、可扩展性和真实感；3.构建基于物理的渲染流水线合成真实玻璃材料和反射效果；4.使用基于LoRA的高效适配方法结合合成数据进行模型训练。

### 主要发现

基于LoRA的高效基础模型适配，结合提出的合成数据，在领域内和零样本基准测试中达到了最先进的性能，证明了该方法的有效性。

### 结论

预训练的扩散变换器与基于物理的数据合成和高效适配相结合，为反射去除提供了可扩展和高保真的解决方案，展示了基础模型在特定视觉任务中的潜力。

### 翻译

我们提出了一种用于单图像反射去除的扩散变换器（DiT）框架，该框架利用了基础扩散模型在恢复任务中的泛化能力。我们不是依赖于特定任务架构，而是通过将预训练的基于DiT的基础模型条件化于反射污染输入，并引导它生成干净的传输层，从而重新利用该模型。我们系统地分析了现有的反射去除数据源，评估了其多样性、可扩展性和真实感。为解决合适数据短缺的问题，我们在Blender中构建了基于物理的渲染（PBR）流水线，该流水线围绕Principled BSDF构建，用于合成真实的玻璃材料和反射效果。基础模型基于LoRA的高效适配，结合提出的合成数据，在领域内和零样本基准测试中取得了最先进的性能。这些结果表明，预训练的扩散变换器与基于物理的数据合成和高效适配相结合，为反射去除提供了可扩展和高保真的解决方案。项目页面：https://hf.co/spaces/huawei-bayerlab/windowseat-reflection-removal-web


### 论文摘要

We introduce a diffusion-transformer (DiT) framework for single-image reflection removal that leverages the generalization strengths of foundation diffusion models in the restoration setting. Rather than relying on task-specific architectures, we repurpose a pre-trained DiT-based foundation model by conditioning it on reflection-contaminated inputs and guiding it toward clean transmission layers. We systematically analyze existing reflection removal data sources for diversity, scalability, and photorealism. To address the shortage of suitable data, we construct a physically based rendering (PBR) pipeline in Blender, built around the Principled BSDF, to synthesize realistic glass materials and reflection effects. Efficient LoRA-based adaptation of the foundation model, combined with the proposed synthetic data, achieves state-of-the-art performance on in-domain and zero-shot benchmarks. These results demonstrate that pretrained diffusion transformers, when paired with physically grounded data synthesis and efficient adaptation, offer a scalable and high-fidelity solution for reflection removal. Project page: https://hf.co/spaces/huawei-bayerlab/windowseat-reflection-removal-web

---

## 77. Strategic Self-Improvement for Competitive Agents in AI Labour Markets

**论文链接:** [http://arxiv.org/abs/2512.04988v1](http://arxiv.org/abs/2512.04988v1)

**作者:** Christopher Chiu, Simpson Zhang, Mihaela van der Schaar

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一个新框架，首次捕捉塑造AI代理劳动力市场的现实经济力量，包括逆向选择、道德风险和声誉动态。该框架强调AI代理需要具备元认知、竞争意识和长期战略规划三大核心能力。

### 背景

随着人工智能代理在经济各领域的部署，理解它们的战略行为和市场影响变得至关重要。现有研究缺乏对AI代理劳动力市场经济力量的系统分析。

### 目的

开发一个能够捕捉塑造AI代理劳动力市场现实经济力量的框架，并研究AI代理的战略行为及其市场影响。

### 方法

通过构建一个可模拟的零工经济场景，让代理的大型语言模型(LLM)竞争工作、发展技能并在竞争压力下调整策略。使用模拟和受控实验来研究AI代理的行为和市场影响。

### 主要发现

1. 具有推理能力的LLM代理能够学习战略性地自我改进；2. 这些代理表现出对变化市场条件的卓越适应性；3. 模拟重现了人类劳动力市场中的经典宏观经济现象；4. 受控实验揭示了AI可能带来的经济趋势，如快速垄断和系统性价格紧缩。

### 结论

该工作为进一步探索AI驱动劳动力市场的经济特性提供了基础，并为研究新兴经济中竞争AI代理的战略推理能力提供了概念框架。

### 翻译

随着人工智能(AI)代理在各经济领域的部署，理解它们的战略行为和市场层面的影响变得至关重要。本文提出了一个开创性的新框架，首次捕捉塑造代理劳动力市场的现实经济力量：逆向选择、道德风险和声誉动态。我们的框架包含成功的LLM代理所需的三种核心能力：元认知（准确评估自身技能）、竞争意识（建模竞争对手和市场动态）和长期战略规划。我们通过一个可模拟的零工经济场景来说明我们的框架，其中代理的大型语言模型(LLM)竞争工作、发展技能并在竞争压力下调整策略。我们的模拟展示了具有推理能力的LLM代理如何学习战略性地自我改进，并表现出对变化市场条件的卓越适应性。在市场层面，我们的模拟重现了人类劳动力市场中的经典宏观经济现象，而受控实验则揭示了潜在的AI驱动的经济趋势，如快速垄断和系统性价格紧缩。这项工作为进一步探索AI驱动劳动力市场的经济特性提供了基础，并为研究新兴经济中竞争代理的战略推理能力提供了概念框架。


### 论文摘要

As artificial intelligence (AI) agents are deployed across economic domains, understanding their strategic behavior and market-level impact becomes critical. This paper puts forward a groundbreaking new framework that is the first to capture the real-world economic forces that shape agentic labor markets: adverse selection, moral hazard, and reputation dynamics. Our framework encapsulates three core capabilities that successful LLM-agents will need: \textbf{metacognition} (accurate self-assessment of skills), \textbf{competitive awareness} (modeling rivals and market dynamics), and \textbf{long-horizon strategic planning}. We illustrate our framework through a tractable simulated gig economy where agentic Large Language Models (LLMs) compete for jobs, develop skills, and adapt their strategies under competitive pressure. Our simulations illustrate how LLM agents explicitly prompted with reasoning capabilities learn to strategically self-improve and demonstrate superior adaptability to changing market conditions. At the market level, our simulations reproduce classic macroeconomic phenomena found in human labor markets, while controlled experiments reveal potential AI-driven economic trends, such as rapid monopolization and systemic price deflation. This work provides a foundation to further explore the economic properties of AI-driven labour markets, and a conceptual framework to study the strategic reasoning capabilities in agents competing in the emerging economy.

---

## 78. Operator Formalism for Laser-Plasma Wakefield Acceleration

**论文链接:** [http://arxiv.org/abs/2512.04982v1](http://arxiv.org/abs/2512.04982v1)

**作者:** Mostafa Behtouei, Carlos Salgado Lopez, Giancarlo Gatti

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文开发了一种基于算子的框架，用于描述毛细管放电中的激光等离子体尾波加速，提供了激光场和等离子体响应耦合动力学的紧凑且系统的描述。

### 背景

激光等离子体尾波加速是下一代加速器实验中高强度激光-等离子体相互作用的重要研究领域。

### 目的

开发一个算子框架，为LPWA提供紧凑和系统的描述，建立LPWA与希尔伯特空间算子理论的联系，并整合神经算子方法用于降阶建模和预测控制。

### 方法

采用关键算子，包括横向模态算子、非线性等离子体算子、等离子体振荡算子和动压源算子，描述模态耦合、等离子体振荡和由动压力引起的非线性反馈。该方法与神经算子方法集成，允许对非线性算子和动压源算子进行高效近似。

### 主要发现

在线性情况下，系统表现出与稳定模态结构相关的不变子空间特性；非线性相互作用会打破这些不变性，导致模态混合和复杂动力学。该方法建立了LPWA与希尔伯特空间算子理论之间的直接联系，为能量传输和尾波形成提供了正式的数学解释。

### 结论

这种混合物理-人工智能框架为下一代加速器实验中的高强度激光-等离子体相互作用的建模、分析和优化提供了坚实的基础。

### 翻译

在本文中，我们开发了一种基于算子的框架，用于描述毛细管放电中的激光等离子体尾波加速，为激光场和等离子体响应的耦合动力学提供了紧凑且系统的描述。该形式主义采用了关键算子：横向模态算子、非线性等离子体算子、等离子体振荡算子和动压源算子，这些算子共同描述了模态耦合、等离子体振荡和由动压力引起的非线性反馈。在线性情况下，系统以与稳定模态结构相关的不变子空间为特征，而非线性相互作用则打破这些不变性，导致模态混合和复杂动力学。该方法建立了LPWA与希尔伯特空间算子理论之间的直接联系，为能量传输和尾波形成提供了正式的数学解释。此外，算子形式主义与神经算子方法集成，允许对非线性算子和动压源算子进行高效近似，用于降阶建模和预测控制。这种混合物理-人工智能框架为下一代加速器实验中的高强度激光-等离子体相互作用的建模、分析和优化提供了坚实的基础。


### 论文摘要

In this paper, we develop an operator-based framework for laser--plasma wakefield acceleration (LPWA) in capillary discharges, providing a compact and systematic description of the coupled dynamics of laser fields and plasma response. The formalism employs key operators: the transverse modal operator $\hat{K}$, the nonlinear plasma operator $\hat{N}[Ψ]$, the plasma oscillation operator $\hatΩ_p^{\,2}$, and the ponderomotive source operator $\hatα$, which together describe mode coupling, plasma oscillations, and nonlinear feedback induced by the ponderomotive force. In the linear regime, the system is characterized by invariant subspaces associated with stable modal structures, while nonlinear interactions break these invariances, leading to mode mixing and complex dynamics. The approach establishes a direct connection between LPWA and Hilbert-space operator theory, including the invariant subspace, providing a formal mathematical interpretation of energy transfer and wakefield formation. Furthermore, the operator formalism integrates with neural operator methods, allowing efficient approximation of $\hat{N}$ and $\hatα$ for reduced-order modeling and predictive control. This hybrid physics--AI framework offers a robust foundation for modeling, analysis, and optimization of high-intensity laser--plasma interactions in next-generation accelerator experiments.

---

## 79. Aligned but Stereotypical? The Hidden Influence of System Prompts on Social Bias in LVLM-Based Text-to-Image Models

**论文链接:** [http://arxiv.org/abs/2512.04981v1](http://arxiv.org/abs/2512.04981v1)

**作者:** NaHyeon Park, Namin An, Kunhee Kim, Soyeon Yoon, Jiahao Huo, Hyunjung Shim

**发布时间:** 2025-12-04

**备注:** Project page: https://fairpro-t2i.github.io

### GPT解析

### 总结

本研究探讨了大型视觉-语言模型（LVLM）在图像生成中是否放大社会偏见的问题，发现LVLM模型比非LVLM模型产生更多社会偏见的图像，并提出了FairPro框架来减少这种偏见。

### 背景

大型视觉-语言模型（LVLM）已成为图像生成的主导范式，但其是否放大社会偏见尚不完全清楚。

### 目的

研究LVLM模型在图像生成过程中是否放大社会偏见，并探索减少这种偏见的方法。

### 方法

引入一个包含1024个提示的基准，涵盖四个层次的语言复杂性；系统性地评估多个属性中的人口统计偏见；通过解码中间表示、标记概率诊断和嵌入关联分析，揭示系统提示如何编码人口统计先验并传播到图像合成中；提出FairPro，一种无需训练的元提示框架，使LVLM能够在测试时进行自我审计并构建公平感知的系统提示。

### 主要发现

LVLM模型比非LVLM模型产生明显更多社会偏见的图像；系统提示（指导LVLM的预定义指令）是偏见行为的主要驱动因素；系统提示编码的人口统计先验会传播到图像合成过程中。

### 结论

FairPro框架在SANA和Qwen-Image两个LVLM模型上显著减少了人口统计偏见，同时保持了文本-图像的一致性。研究结果提供了对系统提示在偏见传播中核心作用的更深入见解，并为构建更负责任的社会T2I系统提供了实用、可部署的方法。

### 翻译

大型视觉-语言模型（LVLM）基于文本到图像（T2I）系统已成为图像生成的主导范式，然而它们是否放大社会偏见仍然未被充分理解。在本文中，我们表明基于LVLM的模型比非LVLM模型产生明显更多社会偏见的图像。我们引入了一个包含1024个提示的基准，涵盖四个层次的语言复杂性，并以系统的方式评估多个属性中的人口统计偏见。我们的分析确定系统提示（指导LVLM的预定义指令）是偏见行为的主要驱动因素。通过解码中间表示、标记概率诊断和嵌入关联分析，我们揭示了系统提示如何编码传播到图像合成中的人口统计先验。为此，我们提出了FairPro，一种无需训练的元提示框架，使LVLM能够在测试时进行自我审计并构建公平感知的系统提示。在SANA和Qwen-Image两个基于LVLM的T2I模型上的实验表明，FairPro显著减少了人口统计偏见，同时保持了文本-图像的一致性。我们相信我们的发现提供了对系统提示在偏见传播中核心作用的更深入见解，并为构建更负责任的社会T2I系统提供了实用、可部署的方法。


### 论文摘要

Large vision-language model (LVLM) based text-to-image (T2I) systems have become the dominant paradigm in image generation, yet whether they amplify social biases remains insufficiently understood. In this paper, we show that LVLM-based models produce markedly more socially biased images than non-LVLM-based models. We introduce a 1,024 prompt benchmark spanning four levels of linguistic complexity and evaluate demographic bias across multiple attributes in a systematic manner. Our analysis identifies system prompts, the predefined instructions guiding LVLMs, as a primary driver of biased behavior. Through decoded intermediate representations, token-probability diagnostics, and embedding-association analyses, we reveal how system prompts encode demographic priors that propagate into image synthesis. To this end, we propose FairPro, a training-free meta-prompting framework that enables LVLMs to self-audit and construct fairness-aware system prompts at test time. Experiments on two LVLM-based T2I models, SANA and Qwen-Image, show that FairPro substantially reduces demographic bias while preserving text-image alignment. We believe our findings provide deeper insight into the central role of system prompts in bias propagation and offer a practical, deployable approach for building more socially responsible T2I systems.

---

## 80. Balanced Few-Shot Episodic Learning for Accurate Retinal Disease Diagnosis

**论文链接:** [http://arxiv.org/abs/2512.04967v1](http://arxiv.org/abs/2512.04967v1)

**作者:** Jasmaine Khale, Ravi Prakash Srivastava

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究提出了一种平衡的少样本情节学习框架，用于解决视网膜疾病诊断中数据稀缺和类别不平衡的问题，通过平衡采样、数据增强和预训练模型相结合的方法，显著提高了诊断准确率并减少了对多数类的偏见。

### 背景

随着糖尿病视网膜病变和黄斑变性等视网膜疾病发病率的上升，自动化视网膜疾病诊断变得至关重要。然而传统深度学习方法需要大量标注数据，而这些数据成本高且类别不平衡，限制了实际应用中的可靠性。

### 目的

开发一种能够在数据受限条件下有效工作的视网膜疾病诊断方法，解决数据稀缺和类别不平衡问题，提高对少数类疾病的诊断准确性。

### 方法

提出一种针对视网膜眼底多疾病图像数据集(RFMiD)的平衡少样本情节学习框架，包含三个关键组件：(1)平衡的情节采样，确保每个5类5样本情节中所有类别平等参与；(2)有针对性的数据增强，包括CLAHE和颜色/几何变换，提高少数类多样性；(3)使用在ImageNet上预训练的ResNet-50编码器捕获细粒度视网膜特征。在嵌入空间计算原型并使用余弦相似度进行分类。

### 主要发现

框架在100个情节上训练，1000个测试情节上评估，取得了显著的准确率提升，减少了对多数类的偏见，对代表性不足的疾病有显著改善。结果表明，数据感知的少样本流程结合平衡采样和CLAHE增强预处理，可以在数据受限条件下提供更稳健和临床公平的视网膜疾病诊断。

### 结论

数据感知的少样本诊断流程，结合平衡采样和CLAHE增强预处理，能够在数据受限条件下提供更稳健和临床公平的视网膜疾病诊断，特别适用于少数类疾病的识别。

### 翻译

鉴于糖尿病视网膜病变和黄斑变性等疾病的发病率不断上升，自动化视网膜疾病诊断至关重要。传统深度学习方法需要大量标注数据，这些数据不仅成本高昂，而且在疾病类别间常常不平衡，限制了它们在实际应用中的可靠性。少样本学习通过使模型能够从每个类别仅有的少量标记样本中泛化来解决这一挑战。在本研究中，我们提出了一个针对视网膜眼底多疾病图像数据集的平衡少样本情节学习框架。关注最具代表性的十个类别，这些类别在多数疾病和少数疾病之间仍然存在显著不平衡，我们的方法整合了三个关键组件：平衡的情节采样，确保每个情节中所有类别平等参与；有针对性的增强，包括对比限制自适应直方图均衡化和颜色/几何变换，以提高少数类的多样性；以及在ImageNet上预训练的ResNet-50编码器。原型在嵌入空间中计算，并使用余弦相似度进行分类以提高稳定性。在100个情节上训练，在1000个测试情节上评估，我们的框架取得了显著的准确率提升，减少了对多数类的偏见，对代表性不足的疾病有显著改善。这些结果表明，数据感知的少样本流程，结合平衡采样和CLAHE增强预处理，可以在数据受限条件下提供更稳健和临床公平的视网膜疾病诊断。


### 论文摘要

Automated retinal disease diagnosis is vital given the rising prevalence of conditions such as diabetic retinopathy and macular degeneration. Conventional deep learning approaches require large annotated datasets, which are costly and often imbalanced across disease categories, limiting their reliability in practice. Few-shot learning (FSL) addresses this challenge by enabling models to generalize from only a few labeled samples per class. In this study,we propose a balanced few-shot episodic learning framework tailored to the Retinal Fundus Multi-Disease Image Dataset (RFMiD). Focusing on the ten most represented classes, which still show substantial imbalance between majority diseases (e.g., Diabetic Retinopathy, Macular Hole) and minority ones (e.g., Optic Disc Edema, Branch Retinal Vein Occlusion), our method integrates three key components: (i) balanced episodic sampling, ensuring equal participation of all classes in each 5-way 5-shot episode; (ii) targeted augmentation, including Contrast Limited Adaptive Histogram Equalization (CLAHE) and color/geometry transformations, to improve minority-class di- versity; and (iii) a ResNet-50 encoder pretrained on ImageNet, selected for its superior ability to capture fine-grained retinal features. Prototypes are computed in the embedding space and classification is performed with cosine similarity for improved stability. Trained on 100 episodes and evaluated on 1,000 test episodes, our framework achieves substantial accuracy gains and reduces bias toward majority classes, with notable improvements for underrepresented diseases. These results demonstrate that dataset-aware few-shot pipelines, combined with balanced sampling and CLAHE-enhanced preprocessing, can deliver more robust and clinically fair retinal disease diagnosis under data-constrained conditions.

---

## 81. LiteVGGT: Boosting Vanilla VGGT via Geometry-aware Cached Token Merging

**论文链接:** [http://arxiv.org/abs/2512.04939v1](http://arxiv.org/abs/2512.04939v1)

**作者:** Zhijian Shu, Cheng Lin, Tao Xie, Wei Yin, Ben Li, Zhiyuan Pu, Weize Li, Yao Yao, Xun Cao, Xiaoyang Guo, Xiao-Xiao Long

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了LiteVGGT，一种高效处理大规模3D视觉场景的基础模型，实现了最高10倍的加速和显著的内存减少，同时保持核心性能。

### 背景

3D视觉基础模型如视觉几何基础Transformer(VGGT)在几何感知方面取得了重大进展，但对于长序列处理存在计算效率低和内存消耗大的问题，限制了其在数百张图像以上的大规模场景中的应用。

### 目的

解决VGGT模型在处理长序列时的计算效率和内存消耗问题，使其能够高效处理包含1000张图像的大规模场景。

### 方法

提出几何感知的缓存令牌合并策略，通过分析令牌的几何重要性优化锚点令牌选择，并缓存跨层合并索引以减少计算冗余。

### 主要发现

局部图像区域的令牌具有固有的几何相关性，导致高相似性和计算冗余；相邻网络层之间的令牌相似性保持稳定，允许重用合并决策。

### 结论

LiteVGGT在保持VGGT核心性能的同时，实现了显著的加速和内存减少，支持高效的微调和FP8量化，实验验证了其有效性、可扩展性和鲁棒性。

### 翻译

3D视觉基础模型如视觉几何基础Transformer(VGGT)在几何感知方面已取得巨大进展。然而，对于长序列处理，这些模型耗时且内存密集，限制了其在数百张图像以上大规模场景中的应用。为解决这一问题，我们提出了LiteVGGT，实现了最高10倍的加速和显著的内存减少，使其能够高效处理包含1000张图像的场景。我们推导出两个用于3D重建的关键见解：(1)局部图像区域的令牌具有固有的几何相关性，导致高相似性和计算冗余；(2)相邻网络层之间的令牌相似性保持稳定，允许重用合并决策。基于这些见解，我们设计了一种简单而高效的策略，称为几何感知的缓存令牌合并。我们分析每个令牌的几何重要性，优化锚点令牌选择以更好地保留重建的关键信息。我们还缓存并跨层重用合并索引，显著降低延迟且对精度影响最小。该策略保留了VGGT的核心性能，支持高效的微调和FP8量化以获得进一步的性能提升。大量实验验证了LiteVGGT的有效性、可扩展性和鲁棒性。项目页面：https://garlicba.github.io/LiteVGGT/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决VGGT模型在处理长序列图像时的效率和内存消耗问题。VGGT的帧全局注意力机制导致计算和内存复杂度呈二次方增长，使其难以处理大规模场景（数百张图像）。这个问题在现实中很重要，因为3D视觉基础模型对自动驾驶、增强现实和机器人导航等关键应用至关重要，而大规模场景重建是这些应用的核心需求，但现有方法效率不足限制了实际应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析VGGT的瓶颈和token特性，发现了两个关键洞察：1)来自局部图像区域的token具有高几何相关性和计算冗余；2)相邻网络层间token相似性保持稳定。作者借鉴了token merging技术，但指出现有方法(如ToMe)是为语义token设计的，忽略了VGGT token与3D几何信息的紧密耦合。因此，作者设计了专门针对3D重建的几何感知缓存token合并策略，包括构建几何感知特征图、三类token分区和缓存合并索引等创新组件。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过几何感知的token合并策略在保留关键几何信息的同时减少计算冗余，并利用token相似性在相邻网络层中保持稳定的特性缓存和重用合并索引。整体流程包括：1)构建几何感知特征图(融合像素梯度和token方差)；2)将token分为三类(GA tokens、dst tokens和src tokens)；3)通过余弦相似度合并src tokens到dst tokens；4)缓存合并索引减少重复计算；5)在预测前恢复原始token序列；6)结合微调和FP8量化进一步提高效率。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)几何感知缓存token合并策略，专门针对3D重建设计；2)三类token分区方法，保留关键几何信息；3)缓存合并索引，利用层间相似性稳定性；4)综合优化方案结合微调和FP8量化。相比之前工作，LiteVGGT不同于通用token合并方法(如ToMe)的随机采样，考虑几何约束；区别于FastVGGT的通用策略，更好地保留几何细节；也不同于StreamVGGT(牺牲端到端能力)和QuantVGGT(需要跨场景校准)，保留VGGT核心性能的同时大幅提升效率。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'LiteVGGT通过引入几何感知的缓存token合并策略，在保持VGGT高质量3D重建能力的同时，实现了高达10倍的加速和显著的内存减少，使大规模场景重建变得高效可行。'}


### 论文摘要

3D vision foundation models like Visual Geometry Grounded Transformer (VGGT) have advanced greatly in geometric perception. However, it is time-consuming and memory-intensive for long sequences, limiting application to large-scale scenes beyond hundreds of images. To address this, we propose LiteVGGT, achieving up to 10x speedup and substantial memory reduction, enabling efficient processing of 1000-image scenes. We derive two key insights for 3D reconstruction: (1) tokens from local image regions have inherent geometric correlations, leading to high similarity and computational redundancy; (2) token similarity across adjacent network layers remains stable, allowing for reusable merge decisions. Guided by these, we design a simple yet efficient strategy, dubbed geometry-aware cached token merging. We analyze each token's geometric importance, optimizing anchor token selection to better preserve key information for reconstruction. We also cache and reuse merge indices across layers, substantially reducing latency with minimal accuracy impact. This strategy retains VGGT's core performance, enabling efficient fine-tuning and FP8 quantization for further gains. Extensive experiments validate LiteVGGT's effectiveness, scalability, and robustness. Project page: https://garlicba.github.io/LiteVGGT/

---

## 82. A Systemic Pathological Network Model and Combinatorial Intervention Strategies for Alzheimer's Disease

**论文链接:** [http://arxiv.org/abs/2512.04937v1](http://arxiv.org/abs/2512.04937v1)

**作者:** She Xutong

**发布时间:** 2025-12-04

**备注:** 23 pages

### GPT解析

### 总结

这篇综述探讨了阿尔茨海默病(AD)的病理机制、诊断方法和治疗策略的演变，从传统的淀粉样级联假说转向多因素交互作用的病理网络模型，强调了早期生物标志物检测和多靶点联合治疗的重要性。

### 背景

阿尔茨海默病是神经研究中的重大挑战，其病理特征包括β-淀粉样蛋白斑块和由过度磷酸化tau蛋白组成的神经原纤维缠结。传统观点认为AD遵循线性淀粉样级联假说。

### 目的

综合理解AD发病机制的演变，探索超越单靶点治疗的多靶点联合疗法策略，并强调新兴技术平台在推进AD精准医学中的应用。

### 方法

采用回顾性综述方法，分析基于生物标志物的诊断策略(如AT(N)框架)，探索同时靶向Aβ病理、异常tau和神经炎症的联合治疗方法。

### 主要发现

AD病理生理学理解正转变为阶段适应性病理网络模型；基于生物标志物的诊断策略可实现早期疾病检测；抗Aβ单克隆抗体作为首个疾病修饰疗法效果有限，表明单靶点方法存在局限性；同时靶向多种病理过程的联合治疗具有理论依据。

### 结论

早期生物标志物检测、多靶点治疗策略和AI驱动的患者分层为改变AD进程提供了有希望的路线图；AD管理的未来将由预防性、生物标志物引导的个性化联合干预措施定义。

### 翻译

阿尔茨海默病(AD)持续作为神经研究中的重大挑战，其病理特征为β-淀粉样蛋白(Aβ)斑块和由过度磷酸化tau蛋白组成的神经原纤维缠结。本综述综合了AD发病机制的演变理解，超越线性淀粉样级联假说，将疾病概念化为相互作用的病理学交叉对话，包括Aβ、tau和神经炎症作为阶段适应性病理网络模型的基础。这种不断发展的病理生理学理解与诊断范式的转变相平行，其中基于生物标志物的策略(如AT(N)框架)能够在临床前或前驱阶段实现疾病的早期检测。在这一新背景下，虽然抗Aβ单克隆抗体(如lecanemab、donanemab)作为首个疾病修饰疗法取得突破，但其有限的效果突显了单靶点方法的局限性。因此，我探讨了同时靶向Aβ病理、异常tau和神经炎症的联合治疗的令人信服的理论依据。展望未来，我强调了基因编辑和生物物理神经调控等新兴技术平台在推进精准医学中的作用。最终，早期生物标志物检测、多靶点治疗策略和AI驱动的患者分层为改变AD进程绘制了一条有希望的路线图。AD管理的未来将由预防性、生物标志物引导的个性化联合干预措施定义。


### 论文摘要

Alzheimer's disease (AD) persists as a paramount challenge in neurological research, characterized by the pathological hallmarks of amyloid-$β$ (A$β$) plaques and neurofibrillary tangles composed of hyperphosphorylated tau. This review synthesizes the evolving understanding of AD pathogenesis, moving beyond the linear amyloid cascade hypothesis to conceptualize the disease as a cross-talk of intricately interacting pathologies, encompassing A$β$, tau, and neuroinflammation as the foundation of phase-adapted pathological network model. This evolving pathophysiological understanding parallels a transformation in diagnostic paradigms, where biomarker-based strategies such as the AT(N) framework enable early disease detection during preclinical or prodromal stages. Within this new landscape, while anti-A$β$ monoclonal antibodies (e.g., lecanemab, donanemab), represent a breakthrough as the first disease-modifying therapies, their modest efficacy underscores the limitation of single-target approaches. Therefore, I explore the compelling rationale for combination therapies that simultaneously target A$β$ pathology, aberrant tau, and neuroinflammation. Looking forward, I emphasize emerging technological platforms such as gene editing and biophysical neuromodulation in advancing precision medicine. Ultimately, the integration of early biomarker detection, multi-target therapeutic strategies, and AI-driven patient stratification charts a promising roadmap toward fundamentally altering the trajectory of AD. The future of AD management will be defined by preemptive, biomarker-guided, and personalized combination interventions. Keywords: Alzheimer's disease, amyloid-$β$, tau pathology, neuroinflammation, combination therapy, multi-target therapy, precision medicine, biomarkers

---

## 83. Algorithmic Thinking Theory

**论文链接:** [http://arxiv.org/abs/2512.04923v1](http://arxiv.org/abs/2512.04923v1)

**作者:** MohammadHossein Bateni, Vincent Cohen-Addad, Yuzhou Gu, Silvio Lattanzi, Simon Meierhans, Christopher Mohri

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究引入了一个理论框架，用于分析大型语言模型在复杂推理任务中的迭代改进能力，特别是通过生成和组合多种解决方案来提升推理效果的方法。

### 背景

大型语言模型已被证明能够有效解决复杂推理任务，但通过迭代改进先前生成的解决方案可以进一步提升其能力。这种推理方法可以被视为使用概率推理器的算法。

### 目的

开发一个理论框架来分析这类推理算法，形式化迭代改进和答案聚合等技术的原理，为设计更强大的推理方法提供基础。

### 方法

作者提出了一个基于实验证据的理论框架，不依赖于模型的特定架构细节，而是从实验结果中归纳出通用原理。

### 主要发现

通过迭代改进和组合多种解决方案，可以显著提升大型语言模型的推理能力；所提出的理论框架能够形式化这些技术的原理，并提供一个通用的分析视角。

### 结论

该理论框架为理解大型语言模型的推理过程提供了新视角，不依赖于特定架构，而是基于实验证据，因此可能适用于当前和未来的各种推理系统。

### 翻译

大型语言模型已被证明非常擅长解决复杂推理任务。令人惊讶的是，通过迭代改进先前生成的解决方案，它们的能力常常可以得到提升。在这种情况下，用于生成和组合一组解决方案的推理计划可以被视为使用概率推理器的推理算法。我们引入了一个理论框架来分析这类推理算法。该框架形式化了迭代改进和答案聚合等流行技术的基本原理，为设计新一代更强大的推理方法奠定了基础。与依赖特定架构细节的模型理解方法不同，我们的模型基于实验证据。因此，它提供了一个可能适用于当前和未来各种推理器的通用视角。


### 论文摘要

Large language models (LLMs) have proven to be highly effective for solving complex reasoning tasks. Surprisingly, their capabilities can often be improved by iterating on previously generated solutions. In this context, a reasoning plan for generating and combining a set of solutions can be thought of as an algorithm for reasoning using a probabilistic oracle.   We introduce a theoretical framework for analyzing such reasoning algorithms. This framework formalizes the principles underlying popular techniques for iterative improvement and answer aggregation, providing a foundation for designing a new generation of more powerful reasoning methods. Unlike approaches for understanding models that rely on architectural specifics, our model is grounded in experimental evidence. As a result, it offers a general perspective that may extend to a wide range of current and future reasoning oracles.

---

## 84. Chameleon: Adaptive Adversarial Agents for Scaling-Based Visual Prompt Injection in Multimodal AI Systems

**论文链接:** [http://arxiv.org/abs/2512.04895v1](http://arxiv.org/abs/2512.04895v1)

**作者:** M Zeeshan, Saud Satti

**发布时间:** 2025-12-04

**备注:** 5 pages, 2 figures, IEEE Transactions on Dependable and Secure Computing

### GPT解析

### 总结

本研究揭示并解决了多模态人工智能系统，特别是视觉语言模型(VLMs)中一个被忽视的安全漏洞。研究者提出了名为'Chameleon'的自适应对抗框架，能够有效利用图像预处理操作中的缩放漏洞，实验证明其攻击成功率显著高于传统方法，同时提出了相应的防御建议。

### 背景

多模态人工智能系统，特别是视觉语言模型(VLMs)，已成为从自主决策到自动化文档处理等关键应用的重要组成部分。随着系统规模扩大，它们严重依赖预处理管道来高效处理多样化输入，特别是图像缩放操作，这创造了一个显著但常被忽视的安全漏洞。

### 目的

揭示多模态AI系统中由预处理操作引起的安全漏洞，提出新型自适应对抗框架来暴露和利用这些漏洞，评估框架有效性，并探讨相应的防御机制。

### 方法

研究者提出了名为'Chameleon'的新型自适应对抗框架，采用迭代、基于代理的优化机制，根据目标模型的实时反馈动态优化图像扰动，制作能够承受标准缩放操作的对抗性示例。研究团队在Gemini 2.5 Flash模型上评估了该框架的性能。

### 主要发现

Chameleon在不同缩放因子下的攻击成功率(ASR)达到84.5%，显著优于静态基线攻击的平均32.1%。这些攻击能够有效破坏代理管道，在多步骤任务中将决策准确率降低45%以上，证明了预处理阶段的安全漏洞对多模态AI系统构成的严重威胁。

### 结论

研究揭示了多模态AI系统中一个被忽视的安全漏洞，并提出了有效的攻击和防御方法。Chameleon框架的成功表明，现代AI系统中的预处理操作可能成为安全风险的来源。研究建议采用多尺度一致性检查作为必要的防御机制，以应对这类新型攻击。

### 翻译

多模态人工智能（AI）系统，特别是视觉语言模型（VLMs），已成为从自主决策到自动化文档处理等关键应用中不可或缺的组成部分。随着这些系统规模的扩大，它们严重依赖预处理管道来高效处理多样化输入。然而，这种对标准预处理操作（特别是图像缩放）的依赖，创造了一个显著但常被忽视的安全漏洞。虽然缩放算法旨在计算优化，但可以被利用来隐藏恶意视觉提示，这些提示对人类观察者不可见，但在经过模型处理后会成为活跃的语义指令。当前的对抗策略大多保持静态，未能考虑现代代理工作流的动态特性。为解决这一差距，我们提出了Chameleon，一种新型自适应对抗框架，旨在暴露和利用生产环境VLMs中的缩放漏洞。与传统的静态攻击不同，Chameleon采用迭代、基于代理的优化机制，根据目标模型的实时反馈动态优化图像扰动。这使得框架能够制作出能够承受标准缩放操作的强大对抗性示例，以劫持下游执行。我们在Gemini 2.5 Flash模型上评估了Chameleon。实验表明，Chameleon在不同缩放因子下的攻击成功率（ASR）达到84.5%，显著优于平均只有32.1%的静态基线攻击。此外，我们证明这些攻击能够有效破坏代理管道，在多步骤任务中将决策准确率降低45%以上。最后，我们讨论了这些漏洞的影响，并提出了多尺度一致性检查作为必要的防御机制。


### 论文摘要

Multimodal Artificial Intelligence (AI) systems, particularly Vision-Language Models (VLMs), have become integral to critical applications ranging from autonomous decision-making to automated document processing. As these systems scale, they rely heavily on preprocessing pipelines to handle diverse inputs efficiently. However, this dependency on standard preprocessing operations, specifically image downscaling, creates a significant yet often overlooked security vulnerability. While intended for computational optimization, scaling algorithms can be exploited to conceal malicious visual prompts that are invisible to human observers but become active semantic instructions once processed by the model. Current adversarial strategies remain largely static, failing to account for the dynamic nature of modern agentic workflows. To address this gap, we propose Chameleon, a novel, adaptive adversarial framework designed to expose and exploit scaling vulnerabilities in production VLMs. Unlike traditional static attacks, Chameleon employs an iterative, agent-based optimization mechanism that dynamically refines image perturbations based on the target model's real-time feedback. This allows the framework to craft highly robust adversarial examples that survive standard downscaling operations to hijack downstream execution. We evaluate Chameleon against Gemini 2.5 Flash model. Our experiments demonstrate that Chameleon achieves an Attack Success Rate (ASR) of 84.5% across varying scaling factors, significantly outperforming static baseline attacks which average only 32.1%. Furthermore, we show that these attacks effectively compromise agentic pipelines, reducing decision-making accuracy by over 45% in multi-step tasks. Finally, we discuss the implications of these vulnerabilities and propose multi-scale consistency checks as a necessary defense mechanism.

---

## 85. SoK: a Comprehensive Causality Analysis Framework for Large Language Model Security

**论文链接:** [http://arxiv.org/abs/2512.04841v1](http://arxiv.org/abs/2512.04841v1)

**作者:** Wei Zhao, Zhe Li, Jun Sun

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文介绍了一个统一的因果分析框架，用于系统性地研究大型语言模型中的因果因素，从标记级、神经元级、层级干预到表示级分析。该框架支持对基于因果的攻击和防御方法进行一致实验和比较，并在多个开源模型和安全关键基准上进行了实证评估。

### 背景

大型语言模型展现出显著能力，但容易受到对抗性操纵如越狱攻击，其中精心设计的提示可绕过安全机制。理解此类漏洞背后的因果因素对构建可靠防御至关重要。

### 目的

引入一个统一的因果分析框架，系统性地支持LLMs中所有级别的因果调查，并为基于因果的攻击和防御方法提供一致的实验和比较基础。

### 方法

提出支持标记级、神经元级、层级干预到表示级分析的所有级别因果调查的框架；提供首个因果驱动的越狱研究全面调查；在多个开源模型和安全关键基准(包括越狱、幻觉检测、后门识别和公平性评估)上实证评估框架。

### 主要发现

1) 对因果关键组件的目标干预可可靠修改安全行为；2) 安全相关机制高度局部化(集中在早期到中层，仅1-2%神经元表现出因果影响)；3) 框架提取的因果特征在多种威胁类型上实现超过95%检测准确率。

### 结论

通过连接理论因果分析和实际模型安全，该框架为基于因果的攻击研究、可解释性和鲁棒攻击检测与缓解建立了可复现基础，代码已在GitHub上公开。

### 翻译

大型语言模型展现出显著能力，但仍然容易受到对抗性操纵，如越狱攻击，其中精心设计的提示可以绕过安全机制。理解此类漏洞背后的因果因素对于构建可靠的防御至关重要。在本工作中，我们引入了一个统一的因果分析框架，系统性地支持LLMs中所有级别的因果调查，范围从标记级、神经元级和层级干预到表示级分析。该框架能够在各种基于因果的攻击和防御方法之间进行一致的实验和比较。伴随这一实现，我们提供了首个关于因果驱动的越狱研究的全面调查，并在多个开源模型和安全关键基准(包括越狱、幻觉检测、后门识别和公平性评估)上实证评估了该框架。我们的结果表明：(1)对因果关键组件的目标干预可以可靠地修改安全行为；(2)与安全相关的机制高度局部化(即集中在早期到中层，只有1-2%的神经元表现出因果影响)；(3)从我们的框架中提取的因果特征在多种威胁类型上实现了超过95%的检测准确率。通过连接理论因果分析和实际模型安全，我们的框架为基于因果的攻击研究、可解释性和鲁棒攻击检测与缓解建立了可复现的基础。代码可在https://github.com/Amadeuszhao/SOK_Casuality获取。


### 论文摘要

Large Language Models (LLMs) exhibit remarkable capabilities but remain vulnerable to adversarial manipulations such as jailbreaking, where crafted prompts bypass safety mechanisms. Understanding the causal factors behind such vulnerabilities is essential for building reliable defenses.   In this work, we introduce a unified causality analysis framework that systematically supports all levels of causal investigation in LLMs, ranging from token-level, neuron-level, and layer-level interventions to representation-level analysis. The framework enables consistent experimentation and comparison across diverse causality-based attack and defense methods. Accompanying this implementation, we provide the first comprehensive survey of causality-driven jailbreak studies and empirically evaluate the framework on multiple open-weight models and safety-critical benchmarks including jailbreaks, hallucination detection, backdoor identification, and fairness evaluation. Our results reveal that: (1) targeted interventions on causally critical components can reliably modify safety behavior; (2) safety-related mechanisms are highly localized (i.e., concentrated in early-to-middle layers with only 1--2\% of neurons exhibiting causal influence); and (3) causal features extracted from our framework achieve over 95\% detection accuracy across multiple threat types.   By bridging theoretical causality analysis and practical model safety, our framework establishes a reproducible foundation for research on causality-based attacks, interpretability, and robust attack detection and mitigation in LLMs. Code is available at https://github.com/Amadeuszhao/SOK_Casuality.

---

## 86. EMMA: Efficient Multimodal Understanding, Generation, and Editing with a Unified Architecture

**论文链接:** [http://arxiv.org/abs/2512.04810v1](http://arxiv.org/abs/2512.04810v1)

**作者:** Xin He, Longhui Wei, Jianbo Ouyang, Lingxi Xie, Qi Tian

**发布时间:** 2025-12-04

**备注:** Project Page: https://emma-umm.github.io/emma/

### GPT解析

### 总结

该研究提出了EMMA，一种用于多模态理解、生成和编辑的高效统一架构，包含四个主要创新组件，在效率和性能上均优于现有方法。

### 背景

多模态学习领域需要能够同时处理理解、生成和编辑任务的统一架构，但现有方法在效率和性能上存在局限。

### 目的

设计一个高效且统一的多模态架构，能够在理解、生成和编辑任务上表现出色，同时保持高效率。

### 方法

EMMA架构包含四个核心组件：1) 32:1压缩比的高效自动编码器；2) 视觉标记间的通道级连接而非标记级连接；3) 共享且解耦的网络；4) 视觉理解编码器中的专家混合机制。

### 主要发现

EMMA-4B在效率和性能上都显著优于最先进的统一多模态方法（如BAGEL-7B），并且与专门的多模态理解和生成专家（如Qwen3-VL和Qwen-Image）相比具有竞争力。

### 结论

EMMA为统一多模态架构的未来发展奠定了坚实基础，证明了高效统一架构在多模态任务上的潜力。

### 翻译

我们提出了EMMA，一种用于多模态理解、生成和编辑的高效统一架构。具体来说，EMMA主要包含：1) 一个具有32倍压缩率的高效自动编码器，显著减少了生成所需的标记数量，通过对图像应用相同的压缩率确保了理解和生成任务之间的训练平衡；2) 视觉理解和生成标记之间的通道级连接而非标记级连接，进一步减少了统一架构中的视觉标记；3) 一个共享且解耦的网络，能够在满足任务特定建模需求的同时实现任务间的相互改进；4) 为视觉理解编码器采用的专家混合机制，在参数少量增加的情况下显著提高了感知能力。大量实验表明，EMMA-4B在效率和性能上都显著优于最先进的统一多模态方法（如BAGEL-7B），同时与最近的多模态理解和生成专家（如Qwen3-VL和Qwen-Image）相比也取得了具有竞争力的结果。我们相信EMMA为统一多模态架构的未来发展奠定了坚实基础。


### 论文摘要

We propose EMMA, an efficient and unified architecture for multimodal understanding, generation and editing. Specifically, EMMA primarily consists of 1) An efficient autoencoder with a 32x compression ratio, which significantly reduces the number of tokens required for generation. This also ensures the training balance between understanding and generation tasks by applying the same compression ratio to images. 2) Channel-wise concatenation instead of token-wise concatenation among visual understanding and generation tokens, which further reduces the visual tokens in unified architectures. 3) A shared-and-decoupled network that enables mutual improvements across tasks while meeting the task-specific modeling requirements. 4) A mixture-of-experts mechanism adopted for visual understanding encoder, which substantially improves perceptual capabilities with a few parameters increase. Extensive experiments have shown that EMMA-4B can significantly outperform state-of-the-art unified multimodal approaches (e.g., BAGEL-7B) in both efficiency and performance, while also achieving competitive results compared to recent multimodal understanding and generation experts (e.g., Qwen3-VL and Qwen-Image). We believe that EMMA lays a solid foundation for the future development of unified multimodal architectures.

---

## 87. 论文ID: 2512.04797v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04797v1.json'

---

## 88. Teaching a Transformer to Think Like a Chemist: Predicting Nanocluster Stability

**论文链接:** [http://arxiv.org/abs/2512.04794v1](http://arxiv.org/abs/2512.04794v1)

**作者:** João Marcos T. Palheta, Octavio Rodrigues Filho, Mohammad Soleymanibrojeni, Alexandre Cavalheiro Dias, Diego Guedes-Sobrinho, Wolfgang Wenzel, Roland Aydin, Celso R. C. Rêgo, Maurício Jeomar Piotrowski

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究结合密度泛函理论和基于物理原理的预测性人工智能，系统研究了13个原子二十面体双金属纳米团簇的构型景观，预测了团簇形成能和结构偏好，并识别出关键描述符，为催化和能量转换应用提供了可重现、可解释的纳米团簇筛选方法。

### 背景

原子精确的金属纳米团架桥了分子和体相区域，但设计具有目标稳定性和反应性的双金属基序仍然具有挑战性。

### 目的

研究13个原子二十面体纳米团簇X₁₂TM的构型景观，其中宿主X为(Ti, Zr, Hf)，以及Fe和跨越3d-5d系列的单个过渡金属掺杂剂。

### 方法

结合密度泛函理论(DFT)计算和预测性人工智能方法，对240个双金属团簇进行自旋极化DFT计算，揭示能量和结构趋势；在Quantum Cluster Database中的2968个单元团簇上预训练transformer架构，然后在双金属数据上微调；使用注意力模式和Shapley归因分析关键描述符。

### 主要发现

揭示了控制核壳和表面分离排列之间竞争的能量和结构趋势；预测形成能和结构偏好的平均绝对误差约为0.6-0.7 eV；模型能快速适应未见过的Fe-宿主区域；尺寸不匹配、d电子数和配位环境是关键描述符。

### 结论

所有数据、代码和工作流遵循FAIR/TRUE原则，能够对催化和能量转换中未探索的纳米团簇化学进行可重现、可解释的高效筛选。

### 翻译

原子精确的金属纳米团架桥了分子和体相区域，但设计具有目标稳定性和反应性的双金属基序仍然具有挑战性。我们结合密度泛函理论和基于物理原理的预测性人工智能，绘制了13个原子二十面体纳米团簇X₁₂TM的构型景观，其中宿主X = (Ti, Zr, Hf)，以及Fe和跨越3d-5d系列的单个过渡金属掺杂剂。对240个双金属团簇的自旋极化DFT计算揭示了结合能、形成能、畸变能、有效配位数、d带中心和HOMO-LUMO间隙中的系统性趋势，这些趋势控制着核壳(内部)和表面分离(外部)排列之间的竞争。我们在Quantum Cluster Database中的2968个单元团簇上预训练transformer架构，然后在双金属数据上微调以预测形成能和内部/外部偏好，实现了约0.6-0.7 eV的平均绝对误差和校准的不确定性区间。所得模型仅用少量标记示例即可快速适应未见过的Fe-宿主区域。同时，注意力模式和Shapley归因突出了尺寸不匹配、d电子数和配位环境作为关键描述符。所有数据、代码和工作流遵循FAIR/TRUE原则，能够对催化和能量转换中未探索的纳米团簇化学进行可重现、可解释的筛选。


### 论文摘要

Atomically precise metal nanoclusters bridge the molecular and bulk regimes, but designing bimetallic motifs with targeted stability and reactivity remains challenging. Here we combine density functional theory (DFT) and physics-grounded predictive artificial intelligence to map the configurational landscape of 13-atom icosahedral nanoclusters X$_{12}$TM, with hosts X = (Ti, Zr, Hf), and Fe and a single transition--metal dopant spanning the 3$d$-5$d$ series. Spin-polarized DFT calculations on 240 bimetallic clusters reveal systematic trends in binding and formation energies, distortion penalties, effective coordination number, d-band centre, and HOMO-LUMO gap that govern the competition between core-shell (in) and surface-segregated (out) arrangements. We then pretrain a transformer architecture on a curated set of 2968 unary clusters from the Quantum Cluster Database and fine-tune it on bimetallic data to predict formation energies and in/out preference, achieving mean absolute errors of about $0.6-0.7$eV and calibrated uncertainty intervals. The resulting model rapidly adapts to an unseen Fe-host domain with only a handful of labelled examples. At the same time, attention patterns and Shapley attributions highlight size mismatch, $d$-electron count, and coordination environment as key descriptors. All data, code, and workflows follow FAIR/TRUE principles, enabling reproducible, interpretable screening of unexplored nanocluster chemistries for catalysis and energy conversion.

---

## 89. ASTRIDE: A Security Threat Modeling Platform for Agentic-AI Applications

**论文链接:** [http://arxiv.org/abs/2512.04785v1](http://arxiv.org/abs/2512.04785v1)

**作者:** Eranga Bandara, Amin Hass, Ross Gore, Sachin Shetty, Ravi Mukkamala, Safdar H. Bouk, Xueping Liang, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文介绍了ASTRIDE，一个专为AI代理系统设计的自动化威胁建模平台，解决了传统框架无法有效捕捉的新型安全挑战。

### 背景

AI代理系统在现代软件架构中日益重要，通过大型语言模型实现自主决策、动态任务执行和多模态交互，但同时也带来了提示注入攻击、上下文污染、模型操纵等新型安全威胁，这些威胁无法被传统威胁建模框架有效捕捉。

### 目的

开发ASTRIDE平台，为AI代理系统提供专门的威胁建模解决方案。

### 方法

ASTRIDE扩展了传统STRIDE框架，新增AI代理特定攻击(A)威胁类别；结合微调的视觉语言模型(VLMs)和OpenAI-gpt-oss推理LLM，实现从视觉代理架构图(如数据流图)的端到端分析；通过LLM代理协调VLM联盟和推理LLM之间的交互，自动化整个威胁建模过程。

### 主要发现

ASTRIDE能够为下一代智能系统提供准确、可扩展和可解释的威胁建模。

### 结论

ASTRIDE是首个扩展STRIDE以包含AI特定威胁并集成微调VLMs与推理LLM的框架，能够完全自动化基于图表的威胁建模，适用于AI代理应用程序。

### 翻译

AI代理系统正日益成为现代软件架构中不可或缺的组成部分，通过大型语言模型实现自主决策、动态任务执行和多模态交互。然而，这些系统引入了新颖且不断发展的安全挑战，包括提示注入攻击、上下文污染、模型操纵和不透明的代理间通信，这些挑战无法被传统的威胁建模框架有效捕捉。在本文中，我们介绍了ASTRIDE，一个专为AI代理系统设计的自动化威胁建模平台。ASTRIDE通过引入一个新的威胁类别A(AI代理特定攻击)来扩展传统的STRIDE框架，该类别包含了代理应用程序特有的新兴漏洞，如提示注入、不安全的工具调用和推理颠覆。为了实现自动化威胁建模，ASTRIDE结合了一组微调的视觉语言模型(VLMs)和OpenAI-gpt-oss推理LLM，直接从视觉代理架构图(如数据流图DFDs)执行端到端分析。LLM代理通过协调VLM联盟和推理LLM之间的交互来编排端到端的威胁建模自动化过程。我们的评估表明，ASTRIDE为下一代智能系统提供了准确、可扩展和可解释的威胁建模。据我们所知，ASTRIDE是第一个扩展STRIDE以包含AI特定威胁并集成微调VLMs与推理LLM的框架，能够完全自动化基于图表的威胁建模，适用于AI代理应用程序。


### 论文摘要

AI agent-based systems are becoming increasingly integral to modern software architectures, enabling autonomous decision-making, dynamic task execution, and multimodal interactions through large language models (LLMs). However, these systems introduce novel and evolving security challenges, including prompt injection attacks, context poisoning, model manipulation, and opaque agent-to-agent communication, that are not effectively captured by traditional threat modeling frameworks. In this paper, we introduce ASTRIDE, an automated threat modeling platform purpose-built for AI agent-based systems. ASTRIDE extends the classical STRIDE framework by introducing a new threat category, A for AI Agent-Specific Attacks, which encompasses emerging vulnerabilities such as prompt injection, unsafe tool invocation, and reasoning subversion, unique to agent-based applications. To automate threat modeling, ASTRIDE combines a consortium of fine-tuned vision-language models (VLMs) with the OpenAI-gpt-oss reasoning LLM to perform end-to-end analysis directly from visual agent architecture diagrams, such as data flow diagrams(DFDs). LLM agents orchestrate the end-to-end threat modeling automation process by coordinating interactions between the VLM consortium and the reasoning LLM. Our evaluations demonstrate that ASTRIDE provides accurate, scalable, and explainable threat modeling for next-generation intelligent systems. To the best of our knowledge, ASTRIDE is the first framework to both extend STRIDE with AI-specific threats and integrate fine-tuned VLMs with a reasoning LLM to fully automate diagram-driven threat modeling in AI agent-based applications.

---

## 90. MemLoRA: Distilling Expert Adapters for On-Device Memory Systems

**论文链接:** [http://arxiv.org/abs/2512.04763v1](http://arxiv.org/abs/2512.04763v1)

**作者:** Massimo Bini, Ondrej Bohdal, Umberto Michieli, Zeynep Akata, Mete Ozay, Taha Ceritli

**发布时间:** 2025-12-04

### GPT解析

### 总结

这篇论文介绍了一种名为MemLoRA的新型记忆系统，它通过为小型语言模型配备专门的记忆适配器，实现了在本地设备上的高效部署和记忆操作。同时，作者还提出了MemLoRA-V，这是MemLoRA的视觉扩展，集成了小型视觉-语言模型，实现了原生视觉理解能力。

### 背景

记忆增强的大型语言模型在长时间对话中表现出色，但它们通常过于庞大，不适合本地设备部署。虽然小型语言模型更适合设备端推理，但性能不足，且缺乏原生视觉能力，限制了它们在多模态环境中的应用。

### 目的

开发一种能够在本地设备上高效运行的记忆增强系统，同时保持高性能，并扩展到视觉理解能力。

### 方法

作者提出了MemLoRA系统，通过为小型语言模型配备专门针对记忆操作（知识提取、记忆更新和记忆增强生成）训练的记忆适配器。遵循知识蒸馏原则，每个适配器单独训练。此外，还提出了MemLoRA-V，集成了小型视觉-语言模型实现视觉理解能力。

### 主要发现

在纯文本操作中，MemLoRA比大10倍的基线模型表现更好，在LoCoMo基准测试上达到与大60倍的模型相当的性能。在视觉问答任务中，MemLoRA-V的准确率达到81.3，远高于基于字幕方法的23.7，同时在文本任务中保持强大性能。

### 结论

MemLoRA和MemLoRA-V系统成功实现了在本地设备上的高效记忆操作和视觉理解，无需云依赖，在多模态环境中表现出色。

### 翻译

记忆增强型大型语言模型通过存储相关记忆并将其作为上下文，在长时间对话中表现出显著的一致性。这种基于记忆的个性化在允许用户保持对话和数据隐私的本地设备设置中也很关键。然而，基于记忆的系统通常依赖大型语言模型，这些模型对于本地设备部署来说成本太高。尽管小型语言模型比大型语言模型更适合设备端推理，但它们无法达到足够的性能。此外，这些基于大型语言模型的系统缺乏原生视觉能力，限制了它们在多模态环境中的应用。在本文中，我们介绍了(i)MemLoRA，一种新型记忆系统，通过为小型语言模型配备专门的记忆适配器实现本地部署，以及(ii)其视觉扩展MemLoRA-V，它将小型视觉-语言模型集成到记忆系统中，实现原生视觉理解。遵循知识蒸馏原则，每个适配器针对特定记忆操作（知识提取、记忆更新和记忆增强生成）单独训练。配备记忆适配器的小型模型能够在不依赖云的情况下实现准确的设备端记忆操作。在纯文本操作中，MemLoRA比大10倍的基线模型表现更好，在LoCoMo基准测试上达到与大60倍的模型相当的性能。为了评估视觉理解操作，我们扩展了LoCoMo，增加了需要直接视觉推理的具有挑战性的视觉问答任务。在这方面，我们集成了视觉-语言模型的MemLoRA-V比基于字幕的方法显示出巨大改进（准确率81.3对比23.7），同时在文本任务中保持强大性能，证明了我们的方法在多模态环境中的有效性。


### 论文摘要

Memory-augmented Large Language Models (LLMs) have demonstrated remarkable consistency during prolonged dialogues by storing relevant memories and incorporating them as context. Such memory-based personalization is also key in on-device settings that allow users to keep their conversations and data private. However, memory-augmented systems typically rely on LLMs that are too costly for local on-device deployment. Even though Small Language Models (SLMs) are more suitable for on-device inference than LLMs, they cannot achieve sufficient performance. Additionally, these LLM-based systems lack native visual capabilities, limiting their applicability in multimodal contexts. In this paper, we introduce (i) MemLoRA, a novel memory system that enables local deployment by equipping SLMs with specialized memory adapters, and (ii) its vision extension MemLoRA-V, which integrates small Vision-Language Models (SVLMs) to memory systems, enabling native visual understanding. Following knowledge distillation principles, each adapter is trained separately for specific memory operations$\unicode{x2013}$knowledge extraction, memory update, and memory-augmented generation. Equipped with memory adapters, small models enable accurate on-device memory operations without cloud dependency. On text-only operations, MemLoRA outperforms 10$\times$ larger baseline models (e.g., Gemma2-27B) and achieves performance comparable to 60$\times$ larger models (e.g., GPT-OSS-120B) on the LoCoMo benchmark. To evaluate visual understanding operations instead, we extend LoCoMo with challenging Visual Question Answering tasks that require direct visual reasoning. On this, our VLM-integrated MemLoRA-V shows massive improvements over caption-based approaches (81.3 vs. 23.7 accuracy) while keeping strong performance in text-based tasks, demonstrating the efficacy of our method in multimodal contexts.

---

## 91. A Tutorial on Regression Analysis: From Linear Models to Deep Learning -- Lecture Notes on Artificial Intelligence

**论文链接:** [http://arxiv.org/abs/2512.04747v1](http://arxiv.org/abs/2512.04747v1)

**作者:** Jingyuan Wang, Jiahao Ji

**发布时间:** 2025-12-04

### GPT解析

### 总结

这是一篇面向具备基础数学知识学生的回归分析讲义，属于智能计算课程集群的一部分，提供全面且自包含的回归分析理解。

### 背景

作为智能计算课程集群（包括人工智能、数据挖掘、机器学习和模式识别课程）的回归分析讲义，面向仅具备基础大学数学知识（微积分、线性代数和概率论先修课程）的学生。

### 目的

为学生提供全面且自包含的回归分析理解，无需额外参考资料，系统介绍回归分析的基本概念、建模组件和理论基础。

### 方法

系统介绍线性回归、逻辑回归、多项逻辑回归、多项式回归、基函数模型、基于核的方法以及基于神经网络的非线性回归等，涵盖损失函数设计、参数估计原理、普通最小二乘法、基于梯度的优化算法及其变体，以及Ridge和LASSO回归等正则化技术。

### 主要发现

通过详细的数学推导、示例说明和直观的可视化解释，帮助学生理解回归模型的构建和优化过程，以及特征与响应变量之间潜在关系的揭示。

### 结论

通过连接经典统计建模和现代机器学习实践，为学生提供进一步学习高级人工智能模型的坚实概念和技术基础。

### 翻译

本文作为智能计算课程集群（包括人工智能、数据挖掘、机器学习和模式识别课程）的回归分析讲义。旨在为仅具备基础大学数学知识（即已修读微积分、线性代数和概率论先修课程）的学生提供全面且自包含的回归分析理解，无需任何额外参考资料。讲义系统介绍了回归分析的基本概念、建模组件和理论基础，涵盖线性回归、逻辑回归、多项逻辑回归、多项式回归、基函数模型、基于核的方法以及基于神经网络的非线性回归。核心方法主题包括损失函数设计、参数估计原理、普通最小二乘法、基于梯度的优化算法及其变体，以及Ridge和LASSO回归等正则化技术。通过详细的数学推导、示例说明和直观的可视化解释，材料帮助学生不仅理解回归模型的构建和优化过程，还理解它们如何揭示特征与响应变量之间的潜在关系。通过连接经典统计建模和现代机器学习实践，这些讲义旨在为学生提供进一步学习高级人工智能模型的坚实概念和技术基础。


### 论文摘要

This article serves as the regression analysis lecture notes in the Intelligent Computing course cluster (including the courses of Artificial Intelligence, Data Mining, Machine Learning, and Pattern Recognition). It aims to provide students -- who are assumed to possess only basic university-level mathematics (i.e., with prerequisite courses in calculus, linear algebra, and probability theory) -- with a comprehensive and self-contained understanding of regression analysis without requiring any additional references. The lecture notes systematically introduce the fundamental concepts, modeling components, and theoretical foundations of regression analysis, covering linear regression, logistic regression, multinomial logistic regression, polynomial regression, basis-function models, kernel-based methods, and neural-network-based nonlinear regression. Core methodological topics include loss-function design, parameter-estimation principles, ordinary least squares, gradient-based optimization algorithms and their variants, as well as regularization techniques such as Ridge and LASSO regression. Through detailed mathematical derivations, illustrative examples, and intuitive visual explanations, the materials help students understand not only how regression models are constructed and optimized, but also how they reveal the underlying relationships between features and response variables. By bridging classical statistical modeling and modern machine-learning practice, these lecture notes aim to equip students with a solid conceptual and technical foundation for further study in advanced artificial intelligence models.

---

## 92. 论文ID: 2512.04728v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04728v1.json'

---

## 93. Large Speech Model Enabled Semantic Communication

**论文链接:** [http://arxiv.org/abs/2512.04711v1](http://arxiv.org/abs/2512.04711v1)

**作者:** Yun Tian, Zhijin Qin, Guocheng Lv, Ye Jin, Kaibin Huang, Zhu Han

**发布时间:** 2025-12-04

**备注:** 15 pages, 9 figures

### GPT解析

### 总结

该研究提出了一种基于大型语音模型的语义通信系统(LargeSC)，通过自适应控制器和低秩适应技术，实现了在有损信道上的高效语音传输。

### 背景

现有基于联合信源信道编码的语音语义通信系统受限于特定任务和数据集的模型结构，而生成式大模型在大量数据预训练后能在多种下游任务上表现出色。

### 目的

利用大模型中嵌入的丰富语义知识，实现在有损信道上的自适应传输。

### 方法

提出LargeSC系统，使用Mimi作为语音编解码器将语音转换为离散token，引入自适应控制器实现自适应传输和带内不等错误保护，并采用低秩适应微调Moshi基础模型以恢复丢失的语音token。

### 主要发现

系统支持550 bps至2.06 kbps带宽，在高丢包率下语音质量优于传统方法，端到端延迟约460毫秒。

### 结论

该系统展示了实时部署的潜力。

### 翻译

现有的语音语义通信系统主要基于联合信源信道编码架构，已展现出令人印象深刻的性能，但它们的有效性仍受限于为特定任务和数据集专门设计的模型结构。最近的进展表明，在大量数据集上预训练的生成式大模型可以在各种下游任务上实现出色性能，只需少量微调。为了利用大模型中嵌入的丰富语义知识并实现在有损信道上的自适应传输，我们提出了一种基于大型语音模型的语义通信系统。同时实现有损信道上的自适应压缩和鲁棒传输仍然具有挑战性，需要在压缩效率、语音质量和延迟之间进行权衡。在本工作中，我们使用Mimi作为语音编解码器，将语音转换为与现有网络架构兼容的离散token。我们提出了一个自适应控制器模块，实现自适应传输和带内不等错误保护，根据语音内容和丢包概率在带宽限制下动态调整。此外，我们采用低秩适应技术对Moshi基础模型进行微调，用于生成恢复丢失的语音token。仿真结果表明，所提出的系统支持从550 bps到2.06 kbps的带宽，在高丢包率下语音质量优于传统基线方法，并实现了约460毫秒的端到端延迟，从而展示了其实时部署的潜力。


### 论文摘要

Existing speech semantic communication systems mainly based on Joint Source-Channel Coding (JSCC) architectures have demonstrated impressive performance, but their effectiveness remains limited by model structures specifically designed for particular tasks and datasets. Recent advances indicate that generative large models pre-trained on massive datasets, can achieve outstanding performance arexhibit exceptional performance across diverse downstream tasks with minimal fine-tuning. To exploit the rich semantic knowledge embedded in large models and enable adaptive transmission over lossy channels, we propose a Large Speech Model enabled Semantic Communication (LargeSC) system. Simultaneously achieving adaptive compression and robust transmission over lossy channels remains challenging, requiring trade-offs among compression efficiency, speech quality, and latency. In this work, we employ the Mimi as a speech codec, converting speech into discrete tokens compatible with existing network architectures. We propose an adaptive controller module that enables adaptive transmission and in-band Unequal Error Protection (UEP), dynamically adjusting to both speech content and packet loss probability under bandwidth constraints. Additionally, we employ Low-Rank Adaptation (LoRA) to finetune the Moshi foundation model for generative recovery of lost speech tokens. Simulation results show that the proposed system supports bandwidths ranging from 550 bps to 2.06 kbps, outperforms conventional baselines in speech quality under high packet loss rates and achieves an end-to-end latency of approximately 460 ms, thereby demonstrating its potential for real-time deployment.

---

## 94. 论文ID: 2512.05113v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05113v1.json'

---

## 95. BulletTime: Decoupled Control of Time and Camera Pose for Video Generation

**论文链接:** [http://arxiv.org/abs/2512.05076v1](http://arxiv.org/abs/2512.05076v1)

**作者:** Yiming Wang, Qihang Zhang, Shengqu Cai, Tong Wu, Jan Ackermann, Zhengfei Kuang, Yang Zheng, Frano Rajič, Siyu Tang, Gordon Wetzstein

**发布时间:** 2025-12-04

**备注:** Project Page: https://19reborn.github.io/Bullet4D/

### GPT解析

### 总结

该研究提出了一种4D可控的视频扩散框架，通过解耦场景动态与相机运动，实现了对视频生成过程中时间和空间维度的精确控制，实验表明该方法在可控性和生成质量方面均优于现有方法。

### 背景

新兴的视频扩散模型实现了高视觉保真度，但基本将场景动态与相机运动耦合在一起，限制了它们提供精确空间和时间控制的能力。

### 目的

引入一个4D可控的视频扩散框架，明确地将场景动态与相机姿态解耦，从而能够对场景动态和相机视点进行细粒度操控。

### 方法

框架接受连续的世界时间序列和相机轨迹作为条件输入，通过注意力层中的4D位置编码和自适应归一化进行特征调制，将这些输入注入视频扩散模型。训练时使用了一个独特的数据集，其中时间和相机变化被独立参数化，该数据集将公开提供。

### 主要发现

实验表明，该模型能够在各种时间模式和相机轨迹上实现稳健的现实世界4D控制，同时保持高生成质量，并且在可控性方面优于先前的工作。

### 结论

该方法成功实现了场景动态和相机运动的解耦，提供对视频生成过程中时间和空间维度的精确控制，优于现有的视频扩散模型在可控性方面的表现。

### 翻译

新兴的视频扩散模型实现了高视觉保真度，但基本将场景动态与相机运动耦合在一起，限制了它们提供精确空间和时间控制的能力。我们引入了一个4D可控的视频扩散框架，明确地将场景动态与相机姿态解耦，从而能够对场景动态和相机视点进行细粒度操控。我们的框架接受连续的世界时间序列和相机轨迹作为条件输入，通过注意力层中的4D位置编码和自适应归一化进行特征调制，将这些输入注入视频扩散模型。为了训练该模型，我们整理了一个独特的数据集，其中时间和相机变化被独立参数化；该数据集将公开提供。实验表明，我们的模型能够在各种时间模式和相机轨迹上实现稳健的现实世界4D控制，同时保持高生成质量，并且在可控性方面优于先前的工作。有关视频结果，请访问我们的网站：https://19reborn.github.io/Bullet4D/

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决视频生成中场景动态与相机运动耦合的问题，即无法独立控制视频中的世界时间和相机视角。这个问题很重要，因为它限制了视频生成在电影特效（如子弹时间效果）、游戏和XR场景中的应用，使得用户无法自由控制何时和从何处观察场景中的事件。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了现有视频扩散模型的局限性，特别是它们将视频时间与世界时间耦合的问题。他们借鉴了现有的相机控制视频生成方法，但意识到这些方法仍然将视频时间与世界时间耦合。作者还参考了多视图视频扩散方法，但发现它们需要额外的重建步骤。基于这些分析，作者设计了统一的时间-相机控制模块，包括时间感知的位置编码和基于时间的自适应层归一化，以及相应的相机控制模块。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将视频时间分解为连续的世界时间和相机视角，作为两个独立的条件信号注入到视频扩散模型中。实现流程包括：1) 构建一个4D可控的视频扩散框架，包含连续世界时间控制机制和统一的4D时间-相机条件模块；2) 使用时间感知的位置编码将世界时间信息注入注意力机制；3) 使用时间条件自适应层归一化提供细粒度的时间调制；4) 使用4D位置编码和相机条件自适应归一化实现相机控制；5) 在专门构建的4D控制数据集上训练模型，该数据集中时间和相机因素独立变化。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1) 首次实现视频生成中世界时间和相机姿态的完全解耦控制；2) 提出了时间感知的位置编码和基于时间的自适应层归一化来实现连续时间控制；3) 设计了统一的4D位置编码结合相机条件自适应归一化实现4D控制；4) 构建了专门的4D控制合成数据集，其中时间和相机因素独立变化；5) 与之前工作相比，这种方法不需要额外的4D重建步骤，可以直接生成高质量的视频，同时提供更精确的时间和相机控制。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文通过解耦世界时间和相机姿态，实现了在视频生成中独立控制时间和视角的4D可控框架，显著提升了视频生成在时间和空间上的精确控制能力。'}


### 论文摘要

Emerging video diffusion models achieve high visual fidelity but fundamentally couple scene dynamics with camera motion, limiting their ability to provide precise spatial and temporal control. We introduce a 4D-controllable video diffusion framework that explicitly decouples scene dynamics from camera pose, enabling fine-grained manipulation of both scene dynamics and camera viewpoint. Our framework takes continuous world-time sequences and camera trajectories as conditioning inputs, injecting them into the video diffusion model through a 4D positional encoding in the attention layer and adaptive normalizations for feature modulation. To train this model, we curate a unique dataset in which temporal and camera variations are independently parameterized; this dataset will be made public. Experiments show that our model achieves robust real-world 4D control across diverse timing patterns and camera trajectories, while preserving high generation quality and outperforming prior work in controllability. See our website for video results: https://19reborn.github.io/Bullet4D/

---

## 96. 论文ID: 2512.05030v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.05030v1.json'

---

## 97. HTR-ConvText: Leveraging Convolution and Textual Information for Handwritten Text Recognition

**论文链接:** [http://arxiv.org/abs/2512.05021v1](http://arxiv.org/abs/2512.05021v1)

**作者:** Pham Thach Thanh Truc, Dang Hoai Nam, Huynh Tong Dang Khoa, Vo Nguyen Le Duy

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了一种名为HTR-ConvText的手写文本识别模型，能够捕获细粒度的笔画级局部特征同时保持全局上下文依赖关系，在多个数据集上展现出优于现有方法的性能和泛化能力。

### 背景

手写文本识别面临数据有限、书写风格差异大以及复杂变音符号脚本等挑战，现有方法通常需要大量合成数据才能有效泛化。

### 目的

设计一个能够捕获细粒度、笔画级局部特征同时保留全局上下文依赖关系的手写文本识别模型，以提高识别性能和泛化能力。

### 方法

在特征提取阶段集成残差卷积神经网络骨干网络与带有位置编码的MobileViT块；引入ConvText编码器作为混合架构，在层次结构中结合全局上下文和局部特征，减少序列长度提高效率；添加辅助模块注入文本上下文，缓解连接主义时间分类的弱点。

### 主要发现

在IAM、READ2016、LAM和HANDS-VNOnDB数据集上的评估表明，该方法在性能和泛化能力上优于现有方法，特别是在训练样本有限和手写多样性高的场景中表现更佳。

### 结论

HTR-ConvText模型通过结合结构模式识别和细微书写细节学习，有效解决了手写文本识别中的挑战，特别是在数据受限和书写风格多样的情况下。

### 翻译

手写文本识别由于数据有限、书写风格差异大以及具有复杂变音符号的脚本而仍然具有挑战性。现有方法虽然部分解决了这些问题，但通常在没有大量合成数据的情况下难以泛化。为应对这些挑战，我们提出了HTR-ConvText模型，该模型旨在捕获细粒度的、笔画级的局部特征，同时保持全局上下文依赖关系。在特征提取阶段，我们将残差卷积神经网络骨干网络与带有位置编码的MobileViT块相结合。这使得模型能够捕获结构模式并学习细微的书写细节。随后，我们引入了ConvText编码器，这是一种混合架构，在层次结构中结合了全局上下文和局部特征，减少了序列长度以提高效率。此外，一个辅助模块注入文本上下文，以缓解连接主义时间分类的弱点。在IAM、READ2016、LAM和HANDS-VNOnDB上的评估表明，与现有方法相比，我们的方法实现了更好的性能和泛化能力，特别是在训练样本有限和手写多样性高的场景中。


### 论文摘要

Handwritten Text Recognition remains challenging due to the limited data, high writing style variance, and scripts with complex diacritics. Existing approaches, though partially address these issues, often struggle to generalize without massive synthetic data. To address these challenges, we propose HTR-ConvText, a model designed to capture fine-grained, stroke-level local features while preserving global contextual dependencies. In the feature extraction stage, we integrate a residual Convolutional Neural Network backbone with a MobileViT with Positional Encoding block. This enables the model to both capture structural patterns and learn subtle writing details. We then introduce the ConvText encoder, a hybrid architecture combining global context and local features within a hierarchical structure that reduces sequence length for improved efficiency. Additionally, an auxiliary module injects textual context to mitigate the weakness of Connectionist Temporal Classification. Evaluations on IAM, READ2016, LAM and HANDS-VNOnDB demonstrate that our approach achieves improved performance and better generalization compared to existing methods, especially in scenarios with limited training samples and high handwriting diversity.

---

## 98. Plug-and-Play Homeostatic Spark: Zero-Cost Acceleration for SNN Training Across Paradigms

**论文链接:** [http://arxiv.org/abs/2512.05015v1](http://arxiv.org/abs/2512.05015v1)

**作者:** Rui Chen, Xingyu Chen, Yaoqing Hu, Shihan Kong, Zhiheng Wu, Junzhi Yu

**发布时间:** 2025-12-04

**备注:** 12 pages, 4 figures

### GPT解析

### 总结

本文提出了一种名为自适应稳态脉冲活动调节(AHSAR)的简单方法，用于稳定脉冲神经网络训练并加速收敛。

### 背景

脉冲神经网络具有事件驱动计算、稀疏激活和硬件效率等优点，但训练往往收敛缓慢且缺乏稳定性。

### 目的

开发一种即插即用且与训练范式无关的方法，能够稳定优化并加速收敛，同时不改变模型架构、损失函数或梯度。

### 方法

AHSAR在前向传播中维持每层的稳态状态，通过有界非线性将居中的发放率偏差映射到阈值尺度，使用轻量级跨层扩散避免尖锐不平衡，并应用缓慢的跨周期全局增益，结合验证进展和活动能量调整工作点。该方法不引入可训练参数，计算成本可忽略不计。

### 主要发现

在各种训练方法、不同深度宽度和时间步长的SNN架构，以及RGB和DVS数据集上，AHSAR始终改进了强大的基线模型，并增强了分布外鲁棒性。

### 结论

保持层活动在适度范围内是可扩展和高效SNN训练的一个简单而有效的原则。

### 翻译

脉冲神经网络提供事件驱动计算、稀疏激活和硬件效率，但训练往往收敛缓慢且缺乏稳定性。我们提出了自适应稳态脉冲活动调节(AHSAR)，一种极其简单即插即用且与训练范式无关的方法，可以在不改变模型架构、损失或梯度的情况下稳定优化并加速收敛。AHSAR不引入可训练参数。它在前向传播过程中维持每层的稳态状态，通过有界非线性将居中的发放率偏差映射到阈值尺度，使用轻量级跨层扩散来避免尖锐不平衡，并应用缓慢的跨周期全局增益，结合验证进展和活动能量来调整工作点。计算成本可以忽略不计。在各种训练方法、不同深度、宽度和时间步长的SNN架构，以及RGB和DVS数据集上，AHSAR始终改进了强大的基线并增强了分布外鲁棒性。这些结果表明，保持层活动在适度范围内是可扩展和高效SNN训练的一个简单而有效的原则。


### 论文摘要

Spiking neural networks offer event driven computation, sparse activation, and hardware efficiency, yet training often converges slowly and lacks stability. We present Adaptive Homeostatic Spiking Activity Regulation (AHSAR), an extremely simple plug in and training paradigm agnostic method that stabilizes optimization and accelerates convergence without changing the model architecture, loss, or gradients. AHSAR introduces no trainable parameters. It maintains a per layer homeostatic state during the forward pass, maps centered firing rate deviations to threshold scales through a bounded nonlinearity, uses lightweight cross layer diffusion to avoid sharp imbalance, and applies a slow across epoch global gain that combines validation progress with activity energy to tune the operating point. The computational cost is negligible. Across diverse training methods, SNN architectures of different depths, widths, and temporal steps, and both RGB and DVS datasets, AHSAR consistently improves strong baselines and enhances out of distribution robustness. These results indicate that keeping layer activity within a moderate band is a simple and effective principle for scalable and efficient SNN training.

---

## 99. Detecting Perspective Shifts in Multi-agent Systems

**论文链接:** [http://arxiv.org/abs/2512.05013v1](http://arxiv.org/abs/2512.05013v1)

**作者:** Eric Bridgeford, Hayden Helm

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了时序数据核视角空间(TDKPS)框架，用于联合嵌入跨时间点的智能体，并提出了新颖的假设检验方法来检测黑盒多智能体系统中智能体和群体层面的行为变化。

### 背景

增强外部工具和更新机制的生成模型（智能体）已展示出超越基础模型智能提示的能力。随着智能体使用普及，动态多智能体系统自然涌现。最近的研究调查了基于单时间点查询响应的低维智能体表示的理论和实证特性。

### 目的

开发能够联合嵌入跨时间智能体的框架，并提出新颖的假设检验方法，用于检测黑盒多智能体系统中智能体和群体层面的行为变化。

### 方法

引入时序数据核视角空间(TDKPS)框架联合嵌入不同时间点的智能体，提出多种假设检验方法。通过受 evolving digital personas 多智能体系统启发的模拟来表征所提出检验的实证特性，包括它们对关键超参数的敏感性。最后通过自然实验验证方法的有效性。

### 主要发现

TDKPS框架能有效联合嵌入跨时间点的智能体；提出的假设检验方法能检测智能体和群体层面的行为变化；这些检验方法对关键超参数具有敏感性；通过自然实验验证，所提出的方法能检测与真实外生事件敏感、特异且显著相关的变化。

### 结论

据作者所知，TDKPS是第一个用于监控黑盒多智能体系统行为动态的系统性框架，随着生成智能体部署的持续扩展，这一能力至关重要。

### 翻译

增强外部工具和更新机制的生成模型（或称智能体）已经展示出超越基础模型智能提示的能力。随着智能体使用的普及，动态多智能体系统自然涌现。最近的研究调查了基于单时间点查询响应的低维智能体表示的理论和实证特性。本文引入了时序数据核视角空间(TDKPS)，该框架能够联合嵌入不同时间点的智能体，并提出了几种新颖的假设检验方法，用于检测黑盒多智能体系统中智能体和群体层面的行为变化。我们在受 evolving digital personas 多智能体系统启发的模拟中表征了所提出检验的实证特性，包括它们对关键超参数的敏感性。最后，我们通过自然实验证明，所提出的检验能够检测与真实外生事件敏感、特异且显著相关的变化。据我们所知，TDKPS是第一个用于监控黑盒多智能体系统行为动态的系统性框架——随着生成智能体部署的持续扩展，这一能力至关重要。


### 论文摘要

Generative models augmented with external tools and update mechanisms (or \textit{agents}) have demonstrated capabilities beyond intelligent prompting of base models. As agent use proliferates, dynamic multi-agent systems have naturally emerged. Recent work has investigated the theoretical and empirical properties of low-dimensional representations of agents based on query responses at a single time point. This paper introduces the Temporal Data Kernel Perspective Space (TDKPS), which jointly embeds agents across time, and proposes several novel hypothesis tests for detecting behavioral change at the agent- and group-level in black-box multi-agent systems. We characterize the empirical properties of our proposed tests, including their sensitivity to key hyperparameters, in simulations motivated by a multi-agent system of evolving digital personas. Finally, we demonstrate via natural experiment that our proposed tests detect changes that correlate sensitively, specifically, and significantly with a real exogenous event. As far as we are aware, TDKPS is the first principled framework for monitoring behavioral dynamics in black-box multi-agent systems -- a critical capability as generative agent deployment continues to scale.

---

## 100. Influence of Object Affordance on Action Language Understanding: Evidence from Dynamic Causal Modeling Analysis

**论文链接:** [http://arxiv.org/abs/2512.04989v1](http://arxiv.org/abs/2512.04989v1)

**作者:** Supriya Bordoloi, Cota Navin Gupta, Shyamanta M. Hazarika

**发布时间:** 2025-12-04

**备注:** This work has been submitted to the IEEE Transactions for possible publication

### GPT解析

### 总结

本研究探讨了动作表征通过因果神经动力学影响动作语言理解的过程，发现前脑区域的动作处理通过激活下游的顶叶和颞叶区域来驱动动作语言理解。

### 背景

研究关注动作表征如何影响动作语言理解，以及感觉运动信息如何贡献于语言理解的机制，与具身认知理论相关。

### 目的

调查动作表征影响动作语言理解的因果神经机制，提供感觉运动信息如何贡献于语言理解的机制性解释。

### 方法

18名参与者观察两种刺激条件（纯文本和视频+文本），记录32通道EEG数据，分析事件相关电位和源定位，使用LORETA识别四个左侧半球感兴趣区域，构建动态因果模型并进行多种连接配置测试，应用贝叶斯模型选择和平均方法进行分析。

### 主要发现

贝叶斯模型选择显示主导模型为腹侧前运动皮层(PMv)因果影响下顶小叶(IPL)和后 superior 颞回(pSTG)，反映了从与动作相关的运动区域到语义枢纽的前馈架构；贝叶斯模型平均证实了从外侧枕叶皮层(LOC)到PMv和IPL的强内源性连接，以及从PMv到IPL的显著调制作用。

### 结论

这些发现提供了直接证据，证明前脑区域的动作处理通过激活下游的顶叶和颞叶区域来驱动动作语言理解，结果支持具身认知理论，并提供了感觉运动信息如何贡献于语言理解的机制性解释。

### 翻译

本研究探讨了动作表征通过因果神经动力学影响动作语言理解的机制。在本研究中，18名参与者在实验期间观察两种条件下的刺激：纯文本（如'用锤子击打'）和视频+文本（匹配短语的视觉片段）。从32个通道记录EEG数据，并使用LORETA分析事件相关电位和源定位，确定了四个左侧半球感兴趣区域：外侧枕叶皮层、后 superior 颞回、腹侧前运动皮层和下顶小叶。构建了具有向LOC和pSTG驱动输入的动态因果模型空间，并测试了多种连接配置。贝叶斯模型选择显示了一个主导模型，其中PMv因果影响IPL和pSTG，反映了从与动作相关的运动区域到语义枢纽的前馈架构。贝叶斯模型平均进一步证实了从LOC到PMv和IPL的强内源性连接，以及从PMv到IPL的显著调制。这些发现提供了直接证据，证明前脑区域的动作处理通过激活下游的顶叶和颞叶区域来驱动动作语言理解。结果支持了具身认知理论，并提供了感觉运动信息如何贡献于语言理解的机制性解释。


### 论文摘要

This study investigates the causal neural dynamics by which affordance representations influence action language comprehension. In this study, 18 participants observed stimuli displayed in two conditions during the experiment: text-only (e.g., `Hit with a hammer') and video+text (visual clips with matching phrases). EEG data were recorded from 32 channels and analyzed for event-related potentials and source localization using LORETA, which identified four left-hemisphere regions of interest: the Lateral Occipital Cortex (LOC), Posterior Superior Temporal Gyrus (pSTG), Ventral Premotor Cortex (PMv), and Inferior Parietal Lobule (IPL). A space of dynamic causal modeling (DCM) was constructed with driving inputs to LOC and pSTG, and multiple connectivity configurations were tested. Bayesian Model Selection revealed a dominant model in which PMv causally influenced IPL and pSTG, reflecting a feedforward architecture from affordance-related motor regions to semantic hubs. Bayesian Model Averaging further confirmed strong endogenous connections from LOC to PMv and IPL, and significant modulation from PMv to IPL. These findings provide direct evidence that affordance processing in premotor regions drives action language understanding by engaging downstream parietal and temporal areas. The results support grounded cognition theories and offer a mechanistic account of how sensorimotor information contributes to linguistic comprehension.

---

## 101. 论文ID: 2512.04952v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04952v1.json'

---

## 102. Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion

**论文链接:** [http://arxiv.org/abs/2512.04926v1](http://arxiv.org/abs/2512.04926v1)

**作者:** Yueming Pan, Ruoyu Feng, Qi Dai, Yuqi Wang, Wenfeng Lin, Mingyu Guo, Chong Luo, Nanning Zheng

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了Semantic-First Diffusion (SFD)，一种潜在扩散模型的新范式，通过异步处理语义和纹理，实现从粗到细的图像生成过程，显著提高了生成质量和收敛速度。

### 背景

潜在扩散模型(LDMs)本质上遵循从粗到细的生成过程，高级语义结构比精细纹理生成得早。最近的进展集成了预训练视觉编码器的语义先验，但仍然同步去噪语义和纹理，忽略了这种顺序关系。

### 目的

提出一种明确优先语义形成的潜在扩散范式，利用语义先验为纹理生成提供更好的指导。

### 方法

SFD首先结合从预训练视觉编码器通过专门语义VAE提取的紧凑语义潜力和纹理潜力构建复合潜力。核心是使用独立噪声计划异步去噪语义和纹理，语义比纹理提前一个时间偏移量，为纹理细化提供更清晰的高级指导。

### 主要发现

在ImageNet 256x256上，SFD实现了FID 1.06 (LightningDiT-XL)和FID 1.04 (1.0B LightningDiT-XXL)，比原始DiT快多达100倍的收敛速度，同时改进了ReDi和VA-VAE等现有方法。

### 结论

异步、语义主导的建模方法能有效提升图像生成质量，实现自然的从粗到细生成过程，为扩散模型的发展提供了新思路。

### 翻译

潜在扩散模型(LDMs)本质上遵循从粗到细的生成过程，其中高级语义结构比精细纹理生成得稍早。这表明先前的语义可能通过提供语义锚点来帮助纹理生成。最近的进展集成了来自预训练视觉编码器的语义先验来增强LDMs，但它们仍然同步去噪语义和VAE编码的纹理，忽略了这种顺序。观察到这些，我们提出了Semantic-First Diffusion (SFD)，一种明确优先语义形成的潜在扩散范式。SFD首先通过组合紧凑的语义潜力和纹理潜力来构建复合潜力，语义潜力是通过专门的语义VAE从预训练的视觉编码器中提取的。SFD的核心是使用独立的噪声计划异步去噪语义和纹理潜力：语义比纹理提前一个时间偏移量，为纹理细化提供更清晰的高级指导，并实现自然的从粗到细的生成。在ImageNet 256x256上使用指导，SFD实现了FID 1.06 (LightningDiT-XL)和FID 1.04 (1.0B LightningDiT-XXL)，同时比原始DiT快多达100倍的收敛速度。SFD还改进了现有方法如ReDi和VA-VAE，证明了异步、语义主导建模的有效性。项目页面和代码：https://yuemingpan.github.io/SFD.github.io/


### 论文摘要

Latent Diffusion Models (LDMs) inherently follow a coarse-to-fine generation process, where high-level semantic structure is generated slightly earlier than fine-grained texture. This indicates the preceding semantics potentially benefit texture generation by providing a semantic anchor. Recent advances have integrated semantic priors from pretrained visual encoders to further enhance LDMs, yet they still denoise semantic and VAE-encoded texture synchronously, neglecting such ordering. Observing these, we propose Semantic-First Diffusion (SFD), a latent diffusion paradigm that explicitly prioritizes semantic formation. SFD first constructs composite latents by combining a compact semantic latent, which is extracted from a pretrained visual encoder via a dedicated Semantic VAE, with the texture latent. The core of SFD is to denoise the semantic and texture latents asynchronously using separate noise schedules: semantics precede textures by a temporal offset, providing clearer high-level guidance for texture refinement and enabling natural coarse-to-fine generation. On ImageNet 256x256 with guidance, SFD achieves FID 1.06 (LightningDiT-XL) and FID 1.04 (1.0B LightningDiT-XXL), while achieving up to 100x faster convergence than the original DiT. SFD also improves existing methods like ReDi and VA-VAE, demonstrating the effectiveness of asynchronous, semantics-led modeling. Project page and code: https://yuemingpan.github.io/SFD.github.io/.

---

## 103. Hoi! - A Multimodal Dataset for Force-Grounded, Cross-View Articulated Manipulation

**论文链接:** [http://arxiv.org/abs/2512.04884v1](http://arxiv.org/abs/2512.04884v1)

**作者:** Tim Engelbracht, René Zurbrügg, Matteo Wohlrapp, Martin Büchner, Abhinav Valada, Marc Pollefeys, Hermann Blum, Zuria Bauer

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究介绍了一个用于基于力、跨视图关节操作的数据集，结合视觉、动作和触觉感知，模拟真实人类交互。

### 背景

现有研究缺乏将视觉、动作和触觉感知结合的数据集，特别是在人类和机器人交互视角之间。

### 目的

创建一个包含多种形态和感知模态的数据集，以促进交互理解研究，特别是在人类和机器人视角迁移方面。

### 方法

收集381个关节对象在38个环境中的3048个序列，涵盖四种操作形态（人类手、佩戴手腕相机的人类手、手持UMI夹持器、自定义Hoi!夹持器），并提供同步的末端执行器力和触觉感知数据。

### 主要发现

该数据集提供了从视频中理解交互的整体视角，使研究人员能够评估方法在人类和机器人视角之间的迁移能力。

### 结论

该数据集为研究交互理解提供了新资源，特别是探索力感知和预测等未被充分研究的模态。

### 翻译

我们提出了一个用于基于力、跨视图关节操作的数据集，它在真实人类交互中将视觉、动作和触觉感知结合起来。该数据集包含38个环境中381个关节对象的3048个序列。每个对象在四种形态下操作：(i)人类手，(ii)佩戴手腕相机的人类手，(iii)手持UMI夹持器，(iv)自定义Hoi!夹持器，其中工具形态提供同步的末端执行器力和触觉感知。我们的数据集从视频中提供了交互理解的整体视角，使研究人员能够评估方法在人类和机器人视角之间的迁移能力，同时探索力感知和预测等未被充分研究的模态。


### 论文摘要

We present a dataset for force-grounded, cross-view articulated manipulation that couples what is seen with what is done and what is felt during real human interaction. The dataset contains 3048 sequences across 381 articulated objects in 38 environments. Each object is operated under four embodiments - (i) human hand, (ii) human hand with a wrist-mounted camera, (iii) handheld UMI gripper, and (iv) a custom Hoi! gripper - where the tool embodiment provides synchronized end-effector forces and tactile sensing. Our dataset offers a holistic view of interaction understanding from video, enabling researchers to evaluate how well methods transfer between human and robotic viewpoints, but also investigate underexplored modalities such as force sensing and prediction.

---

## 104. STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions

**论文链接:** [http://arxiv.org/abs/2512.04871v1](http://arxiv.org/abs/2512.04871v1)

**作者:** Junjie Fan, Hongye Zhao, Linduo Wei, Jiayu Rao, Guijia Li, Jiaxin Yuan, Wenqi Xu, Yong Qi

**发布时间:** 2025-12-04

**备注:** This work has been submitted to the IEEE for possible publication

### GPT解析

### 总结

本文提出STELLA框架，通过动态语义抽象机制将时间序列解耦为趋势、季节性和残差组件，并将这些组件的特征翻译为分层语义锚点，有效提升了大语言模型在时间序列预测中的性能。

### 背景

当前大语言模型在时间序列预测中的应用通常无法有效增强原始序列信息，现有提示策略依赖静态相关性而非动态行为的生成式解释，缺乏关键的全局和实例特定上下文。

### 目的

解决现有LLM在时间序列预测中的局限性，通过系统挖掘和注入结构化的补充和互补信息，充分利用LLM的推理能力。

### 方法

提出STELLA(语义-时间与语言抽象的语义-时间对齐)框架，采用动态语义抽象机制将输入序列解耦为趋势、季节性和残差组件，并将其内在行为特征翻译为分层语义锚点：语料级语义先验(CSP)提供全局上下文，细粒度行为提示(FBP)提供实例级模式，使用这些锚点作为前缀提示指导LLM建模。

### 主要发现

在八个基准数据集上的实验表明，STELLA在长期和短期预测中优于最先进方法，在零样本和少样本设置中表现出更好的泛化能力，消融研究验证了动态生成的语义锚点的有效性。

### 结论

STELLA框架通过动态语义抽象和分层语义锚点，有效解决了现有LLM在时间序列预测中的问题，显著提升了预测性能。

### 翻译

近期针对时间序列预测的大语言模型(LLMs)调整通常无法有效增强原始序列信息，导致LLM推理能力未被充分利用。现有的提示策略依赖静态相关性而非动态行为的生成式解释，缺乏关键的全局和实例特定上下文。为解决这一问题，我们提出了STELLA(语义-时间与语言抽象的语义-时间对齐)框架，该框架系统性地挖掘和注入结构化的补充和互补信息。STELLA采用动态语义抽象机制，将输入序列解耦为趋势、季节性和残差组件。然后将这些组件的内在行为特征翻译为分层语义锚点：用于全局上下文的语料级语义先验(CSP)和用于实例级模式的细粒度行为提示(FBP)。使用这些锚点作为前缀提示，STELLA指导LLM建模内在动态。在八个基准数据集上的实验表明，STELLA在长期和短期预测中优于最先进方法，在零样本和少样本设置中显示出更好的泛化能力。消融研究进一步验证了我们动态生成的语义锚点的有效性。


### 论文摘要

Recent adaptations of Large Language Models (LLMs) for time series forecasting often fail to effectively enhance information for raw series, leaving LLM reasoning capabilities underutilized. Existing prompting strategies rely on static correlations rather than generative interpretations of dynamic behavior, lacking critical global and instance-specific context. To address this, we propose STELLA (Semantic-Temporal Alignment with Language Abstractions), a framework that systematically mines and injects structured supplementary and complementary information. STELLA employs a dynamic semantic abstraction mechanism that decouples input series into trend, seasonality, and residual components. It then translates intrinsic behavioral features of these components into Hierarchical Semantic Anchors: a Corpus-level Semantic Prior (CSP) for global context and a Fine-grained Behavioral Prompt (FBP) for instance-level patterns. Using these anchors as prefix-prompts, STELLA guides the LLM to model intrinsic dynamics. Experiments on eight benchmark datasets demonstrate that STELLA outperforms state-of-the-art methods in long- and short-term forecasting, showing superior generalization in zero-shot and few-shot settings. Ablation studies further validate the effectiveness of our dynamically generated semantic anchors.

---

## 105. PENCO: A Physics-Energy-Numerical-Consistent Operator for 3D Phase Field Modeling

**论文链接:** [http://arxiv.org/abs/2512.04863v1](http://arxiv.org/abs/2512.04863v1)

**作者:** Mostafa Bamdad, Mohammad Sadegh Eshaghi, Cosmin Anitescu, Navid Valizadeh, Timon Rabczuk

**发布时间:** 2025-12-04

### GPT解析

### 总结

本文提出了PENCO(物理-能量-数值一致性算子)，一种混合算子学习框架，用于解决空间-时间偏微分方程，特别是在材料科学和流体力学中的界面动力学和微观结构演化问题。

### 背景

空间-时间偏微分方程的准确高效解对理解材料科学和流体力学中的界面动力学和微观结构演化至关重要。神经算子作为传统求解器的数据驱动替代方案已出现，但现有架构存在累积时间误差、难以在长期模拟中泛化以及需要大型训练数据集等局限性。

### 目的

克服现有神经算子的局限性，开发一种能够保持物理一致性并减少长期误差增长的混合算子学习框架。

### 方法

提出PENCO框架，整合物理定律和数值结构，包含：1)围绕时间中点的增强L^2 Gauss-Lobatto配置残差；2)傅里叶空间数值一致性项；3)能量耗散约束；以及4)低频频谱锚定和教师一致性机制。

### 主要发现

PENCO在三维相场基准测试中(包括相分离、结晶、外延生长和复杂模式形成)表现出优越的准确性、稳定性和数据效率，与MHNO和FNO-4D等最先进神经算子相比具有优势，同时保持物理一致的演化。

### 结论

PENCO的混合设计能够在保持控制物理的同时减轻长期误差增长，为解决空间-时间PDEs提供了有效方法。

### 翻译

准确高效地求解空间-时间偏微分方程(如相场模型)对于理解材料科学和流体力学中的界面动力学和微观结构演化至关重要。神经算子(NOs)最近已出现作为传统求解器的强大数据驱动替代方案；然而，现有架构通常累积时间误差，难以在长期模拟中泛化，并且需要大型训练数据集。为克服这些局限性，我们提出了PENCO(物理-能量-数值一致性算子)，一种混合算子学习框架，将物理定律和数值结构整合在数据驱动架构中。该公式引入了围绕时间中点的增强L^2 Gauss-Lobatto配置残差，稳健地强制执行控制动力学并显著提高准确性；傅里叶空间数值一致性项，捕捉半隐式离散化的平衡行为；以及能量耗散约束，确保热力学一致性。额外的低频频谱锚定和教师一致性机制进一步稳定学习并抑制长期误差增长。这种混合设计使PENCO能够在保持控制物理的同时减轻长期误差增长。通过涵盖相分离、结晶、外延生长和复杂模式形成的三维相场广泛基准测试，PENCO与包括多头神经算子(MHNO)和傅里叶神经算子(FNO-4D)在内的最先进神经算子相比，表现出优越的准确性、稳定性和数据效率，同时保持物理一致的演化。相关数据集和实现可在github.com/MBamdad/PENCO获取。


### 论文摘要

Accurate and efficient solutions of spatio-temporal partial differential equations (PDEs), such as phase-field models, are fundamental for understanding interfacial dynamics and microstructural evolution in materials science and fluid mechanics. Neural Operators (NOs) have recently emerged as powerful data-driven alternatives to traditional solvers; however, existing architectures often accumulate temporal errors, struggle to generalize in long-horizon simulations, and require large training datasets. To overcome these limitations, we propose PENCO (Physics-Energy-Numerical-Consistent Operator), a hybrid operator-learning framework that integrates physical laws and numerical structure within a data-driven architecture. The formulation introduces an enhanced L^2 Gauss-Lobatto collocation residual around the temporal midpoint that robustly enforces the governing dynamics and significantly improves accuracy, a Fourier-space numerical consistency term that captures the balanced behavior of semi-implicit discretizations, and an energy-dissipation constraint that ensures thermodynamic consistency. Additional low-frequency spectral anchoring and teacher-consistency mechanisms further stabilize learning and suppress long-term error growth. This hybrid design enables PENCO to preserve governing physics while mitigating long-term error growth. Through extensive three-dimensional phase-field benchmarks covering phase ordering, crystallization, epitaxial growth, and complex pattern formation, PENCO demonstrates superior accuracy, stability, and data efficiency compared to state-of-the-art neural operators, including Multi-Head Neural Operator (MHNO) and Fourier Neural Operator (FNO-4D), while maintaining physically consistent evolution. The associated dataset and implementation are available at github.com/MBamdad/PENCO.

---

## 106. The Stagnant Persistence Paradox: Survival Analysis and Temporal Efficiency in Exact Sciences and Engineering Education

**论文链接:** [http://arxiv.org/abs/2512.04828v1](http://arxiv.org/abs/2512.04828v1)

**作者:** H. R. Paz

**发布时间:** 2025-12-04

**备注:** 18 pages , 5 figures, 3 tables

### GPT解析

### 总结

本研究探讨了高等教育中学生学术进展的垂直结果和水平流动，特别关注了工程教育中学生的辍学和专业转换模式，揭示了系统性低效率问题。

### 背景

传统研究聚焦于学生坚持和辍学的垂直结果，简化了复杂的学术历史；水平流动虽被认识为核心特征，但其时间成本和效率尚未被充分量化。

### 目的

量化高等教育中水平流动的时间成本和效率，分析工程教育中的最终辍学和首次专业转换现象，提出改进机构评估指标的建议。

### 方法

使用阿根廷一所工程学院40年的行政记录(N=24,016)，采用双结果生存分析框架，重建学术轨迹并应用非参数Kaplan-Meier估计器分析时间事件数据。

### 主要发现

学生辍学前中位生存时间为4.33年，存在明显的延长入学长尾，呈现'停滞坚持'现象；专业转换则集中在第一学年，转换者中位时间为1.0年。

### 结论

工程教育中的学术失败是长尾过程而非突然事件，产生高机会成本；机构评估应从静态保留指标转向基于时间事件分析的课程速度指标。

### 翻译

高等教育中学生进展的研究传统上集中在垂直结果上，如坚持和辍学，通常将复杂的学术历史简化为二元指标。虽然水平流动（专业转换、计划变更、重新入学）的结构组成部分最近被认识到是当代大学系统的核心特征，但这些途径的时间成本和效率仍然 largely 未被量化。使用阿根廷一所大型工程学院和理学院40年的行政记录（N = 24,016），本研究将双结果生存分析框架应用于两个关键结果：最终辍学和首次专业转换。我们将学术轨迹重建为入学阶段的序列和CAPIRE协议下的类型化转换，然后部署非参数Kaplan-Meier估计量来模拟右删失情况下的时间事件。结果揭示了关键的系统性低效率：最终辍学前全球中位生存时间为4.33年，存在明显的延长入学的长尾模式。这种模式揭示了一种停滞坚持的现象，学生长期正式注册，但课程进展不成比例。相比之下，专业转换遵循早期事件模式，转换者中位时间为1.0年，大多数转换集中在第一学年。我们认为，僵化工程课程中的学术失败不是一个突然的结果，而是一个长尾过程，产生高机会成本，机构指标应从静态的保留指标转向基于时间事件分析的课程速度指标。


### 论文摘要

Research on student progression in higher education has traditionally focused on vertical outcomes such as persistence and dropout, often reducing complex academic histories to binary indicators. While the structural component of horizontal mobility (major switching, plan changes, re-entries) has recently been recognised as a core feature of contemporary university systems, the temporal cost and efficiency of these pathways remain largely unquantified. Using forty years of administrative records from a large faculty of engineering and exact sciences in Argentina (N = 24,016), this study applies a dual-outcome survival analysis framework to two key outcomes: definitive dropout and first major switch. We reconstruct academic trajectories as sequences of enrolment spells and typed transitions under the CAPIRE protocol, and then deploy non-parametric Kaplan-Meier estimators to model time-to-event under right-censoring. Results uncover a critical systemic inefficiency: a global median survival time of 4.33 years prior to definitive dropout, with a pronounced long tail of extended enrolment. This pattern reveals a phenomenon of stagnant persistence, where students remain formally enrolled for long periods without commensurate curricular progression. In contrast, major switching follows an early-event regime, with a median time of 1.0 year among switchers and most switches concentrated within the first academic year. We argue that academic failure in rigid engineering curricula is not a sudden outcome but a long-tail process that generates high opportunity costs, and that institutional indicators should shift from static retention metrics towards measures of curricular velocity based on time-to-event analysis.

---

## 107. 论文ID: 2512.04823v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04823v1.json'

---

## 108. Order Matters: 3D Shape Generation from Sequential VR Sketches

**论文链接:** [http://arxiv.org/abs/2512.04761v1](http://arxiv.org/abs/2512.04761v1)

**作者:** Yizi Chen, Sidi Wu, Tianyi Xiao, Nina Wiedemann, Loic Landrieu

**发布时间:** 2025-12-04

### GPT解析

### 总结

VRSketch2Shape是首个从连续VR素描生成3D形状的框架和多类别数据集，考虑了笔画的时序顺序，提高了几何保真度并有效泛化到真实素描。

### 背景

VR素描让用户能在3D环境中直接探索和迭代想法，比传统CAD工具更快直观，但现有素描到形状模型忽略了笔画时序顺序，丢失了结构和设计意图信息。

### 目的

引入VRSketch2Shape框架和数据集，解决现有模型忽略笔画时序顺序的问题，实现从连续VR素描生成3D形状。

### 方法

三方面贡献：(1)从任意形状生成连续VR素描的自动化流程；(2)包含四个类别的超过20k个合成和900个手绘素描-形状对的数据集；(3)考虑顺序的素描编码器与基于扩散的3D生成器相结合。

### 主要发现

该方法比先前工作产生更高的几何保真度，能有效从合成数据泛化到真实素描且只需少量监督，在部分素描上表现良好。

### 结论

所有数据和模型将在https://chenyizi086.github.io/VRSketch2Shape_website开源发布。

### 翻译

VR素描让用户能够在3D中直接探索和迭代想法，提供了比传统CAD工具更快、更直观的替代方案。然而，现有的素描到形状模型忽略了笔画的时序顺序，丢弃了关于结构和设计意图的重要线索。我们介绍了VRSketch2Shape，这是第一个从连续VR素描生成3D形状的框架和多类别数据集。我们的贡献有三个方面：(i) 一个从任意形状生成连续VR素描的自动化流程，(ii) 一个包含四个类别的超过20k个合成和900个手绘素描-形状对的数据集，(iii) 一个考虑顺序的素描编码器与基于扩散的3D生成器相结合。我们的方法比先前的工作产生更高的几何保真度，能够有效地从合成数据泛化到真实素描且只需少量监督，即使在部分素描上也能表现良好。所有数据和模型将在https://chenyizi086.github.io/VRSketch2Shape_website开源发布。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决现有VR草图到3D形状生成模型忽略笔画时序顺序的问题。这个问题很重要，因为传统CAD工具学习曲线陡峭，不适合快速创意设计，而自然语言又难以精确描述复杂几何形状。VR草图提供了直观的3D设计体验，但现有方法将其视为无序点云，丢失了关于连接性、结构和设计意图的重要线索。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先识别了VR草图到3D形状生成的三大挑战：数据稀缺、几何不对齐和时间信息丢失。针对这些问题，作者设计了三个主要贡献：合成草图生成流水线、真实草图收集工具和顺序感知的形状生成模型。作者借鉴了现有工作，如使用BERT架构编码笔画序列和SDFUSION进行扩散模型生成，但创新性地将VR草图表示为有序的笔画序列，并开发了特殊的标记化和嵌入方法来保留时序信息。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是将VR草图视为有序的笔画序列，每个笔画本身是有序的3D点序列，利用绘制顺序信息来理解结构和设计意图。整体流程包括：1)数据准备阶段，通过自动化流水线生成合成草图和收集真实草图；2)草图编码阶段，将草图标记化为有序序列，使用空间和序列嵌入，并通过BERT风格编码器处理；3)3D形状生成阶段，使用SDFUSION扩散模型根据草图编码生成高保真3D形状。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)顺序感知的草图表示，保留笔画和点的时序信息；2)自动化合成草图生成流水线，无需人工标注；3)首个多类别VR草图数据集；4)顺序感知的草图编码器结合扩散模型。相比之前工作，本文明确建模笔画顺序，提供多类别数据集，实现了从合成到真实的有效泛化，并能处理部分草图输入，而之前方法将草图视为无序点云，无法充分利用时序信息。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本文提出了首个考虑笔画顺序的VR草图到3D形状生成框架VRSKETCH2SHAPE，通过顺序感知的草图编码器和基于扩散的生成器，实现了从顺序VR草图生成高保真3D形状，并发布了首个多类别VR草图数据集。'}


### 论文摘要

VR sketching lets users explore and iterate on ideas directly in 3D, offering a faster and more intuitive alternative to conventional CAD tools. However, existing sketch-to-shape models ignore the temporal ordering of strokes, discarding crucial cues about structure and design intent. We introduce VRSketch2Shape, the first framework and multi-category dataset for generating 3D shapes from sequential VR sketches. Our contributions are threefold: (i) an automated pipeline that generates sequential VR sketches from arbitrary shapes, (ii) a dataset of over 20k synthetic and 900 hand-drawn sketch-shape pairs across four categories, and (iii) an order-aware sketch encoder coupled with a diffusion-based 3D generator. Our approach yields higher geometric fidelity than prior work, generalizes effectively from synthetic to real sketches with minimal supervision, and performs well even on partial sketches. All data and models will be released open-source at https://chenyizi086.github.io/VRSketch2Shape_website.

---

## 109. Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length

**论文链接:** [http://arxiv.org/abs/2512.04677v1](http://arxiv.org/abs/2512.04677v1)

**作者:** Yubo Huang, Hailong Guo, Fangtai Wu, Shifeng Zhang, Shijie Huang, Qijun Gan, Lin Liu, Sirui Zhao, Enhong Chen, Jiaming Liu, Steven Hoi

**发布时间:** 2025-12-04

### GPT解析

### 总结

Live Avatar是一个算法-系统协同设计框架，使用140亿参数的扩散模型实现高效、高保真和无限长度的人脸生成，解决了现有扩散模型在视频生成中的顺序计算和长程不一致性问题。

### 背景

现有的基于扩散的视频生成方法受到顺序计算和长程不一致性的根本限制，阻碍了它们在实时、流式音频驱动的人脸合成中的实际应用。

### 目的

提出一个名为Live Avatar的算法-系统协同设计框架，实现高效、高保真和无限长度的人脸生成，使用140亿参数的扩散模型。

### 方法

引入Timestep-forcing Pipeline Parallelism (TPP)分布式推理范式，在多个GPU上流水线式去噪步骤；提出Rolling Sink Frame Mechanism (RSFM)通过动态校准外观维持序列保真度；利用Self-Forcing Distribution Matching Distillation实现大规模模型的因果、可流式适配。

### 主要发现

Live Avatar展示了最先进的性能，在5个H800 GPU上达到20 FPS的端到端生成速度，是首个在此规模上实现实用、实时、高保真人脸生成的系统。

### 结论

为在工业长格式视频合成应用中部署先进的扩散模型建立了新范式。

### 翻译

现有的基于扩散的视频生成方法从根本上受到顺序计算和长程不一致性的限制，限制了它们在实时、流式音频驱动的人脸合成中的实际应用。我们提出了Live Avatar，一个算法-系统协同设计的框架，使用140亿参数的扩散模型实现高效、高保真和无限长度的人脸生成。我们的方法引入了Timestep-forcing Pipeline Parallelism (TPP)，一种分布式推理范式，在多个GPU上流水线式处理去噪步骤，有效打破了自回归瓶颈，确保稳定、低延迟的实时流式处理。为了进一步增强时间一致性和减轻身份漂移和颜色伪影，我们提出了Rolling Sink Frame Mechanism (RSFM)，通过使用缓存的参考图像动态校准外观来维持序列保真度。此外，我们利用Self-Forcing Distribution Matching Distillation促进大规模模型的因果、可流式适配，同时不牺牲视觉质量。Live Avatar展示了最先进的性能，在5个H800 GPU上达到20 FPS的端到端生成速度，据我们所知，是首个在此规模上实现实用、实时、高保真人脸生成的系统。我们的工作为在工业长格式视频合成应用中部署先进的扩散模型建立了新范式。


### 论文摘要

Existing diffusion-based video generation methods are fundamentally constrained by sequential computation and long-horizon inconsistency, limiting their practical adoption in real-time, streaming audio-driven avatar synthesis. We present Live Avatar, an algorithm-system co-designed framework that enables efficient, high-fidelity, and infinite-length avatar generation using a 14-billion-parameter diffusion model. Our approach introduces Timestep-forcing Pipeline Parallelism (TPP), a distributed inference paradigm that pipelines denoising steps across multiple GPUs, effectively breaking the autoregressive bottleneck and ensuring stable, low-latency real-time streaming. To further enhance temporal consistency and mitigate identity drift and color artifacts, we propose the Rolling Sink Frame Mechanism (RSFM), which maintains sequence fidelity by dynamically recalibrating appearance using a cached reference image. Additionally, we leverage Self-Forcing Distribution Matching Distillation to facilitate causal, streamable adaptation of large-scale models without sacrificing visual quality. Live Avatar demonstrates state-of-the-art performance, reaching 20 FPS end-to-end generation on 5 H800 GPUs, and, to the best of our knowledge, is the first to achieve practical, real-time, high-fidelity avatar generation at this scale. Our work establishes a new paradigm for deploying advanced diffusion models in industrial long-form video synthesis applications.

---

## 110. Infinity of solutions to initial-boundary value problems for linear constant-coefficient evolution PDEs on semi-infinite intervals

**论文链接:** [http://arxiv.org/abs/2512.04670v1](http://arxiv.org/abs/2512.04670v1)

**作者:** Andreas Chatziafratis, Spyridon Kamvissis

**发布时间:** 2025-12-04

### GPT解析

### 总结

该论文提出了一种算法程序，用于构建一大类线性演化偏微分方程在四分之一平面中的初边值问题经典解的非唯一性反例，并通过热方程和线性KdV方程的Dirichlet数据应用案例展示了该技术。

### 背景

研究关注的是线性演化偏微分方程在四分之一平面中的初边值问题，这类问题在数学物理中有重要应用。

### 目的

开发一种系统化的方法来构造这类方程解的非唯一性反例，从而揭示某些条件下解的唯一性问题。

### 方法

研究基于对通过复分析技术和Fokas统一变换方法导出的闭式积分表示公式在时空域边界附近的正则性和渐近性质的分析，并严格实施现代PDE技术Fokas统一变换方法。

### 主要发现

1. 提出了一种算法程序，可构造任意阶数和具有常数系数的线性演化偏微分方程的非唯一性反例；2. 通过热方程和线性KdV方程的Dirichlet数据应用案例展示了该技术；3. 为这两个模型提出了新的唯一性定理。

### 结论

该研究提供了一种新的方法来分析线性演化偏微分方程解的唯一性问题，并通过具体案例验证了该方法的有效性。

### 翻译

在这篇简短通讯中，我们宣布了一种算法程序，用于构建一大类线性演化偏微分方程（具有任意阶数和常数系数，在四分之一平面中表述的初边值问题）的经典解的非唯一性反例。我们的方法依赖于通过复分析技术和严格实施被称为Fokas统一变换方法的现代PDE技术导出的闭式积分表示公式在时空域边界附近的正则性和渐近性质分析。为了阐明新思想并以自包含方式展示所提出的技术，我们明确将其应用于两个具体例子，即具有Dirichlet数据的热方程和线性KdV方程。本文还介绍了这两个模型的新唯一性定理。


### 论文摘要

In this short communication, we announce an algorithmic procedure for constructing non-uniqueness counter-examples of classical solutions to initial-boundary-value problems for a wide class of linear evolution partial differential equations, of any order and with constant coefficients, formulated in a quarter-plane. Our approach relies on analysis of regularity and asymptotic properties, near the boundary of the spatio-temporal domain, of closed-form integral-representation formulae derived via complex-analytic techniques and rigorous implementation of the modern PDE technique known as Fokas unified transform method. In order to elucidate the novel idea and demonstrate the proposed technique in a self-contained fashion, we explicitly present its application to two concrete examples, namely the heat equation and the linear KdV equation with Dirichlet data. New uniqueness theorems for these two models are also presented herein.

---

## 111. Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs

**论文链接:** [http://arxiv.org/abs/2512.04668v1](http://arxiv.org/abs/2512.04668v1)

**作者:** Jinbo Liu, Defu Cao, Yifei Wei, Tianyao Su, Yuan Liang, Yushun Dong, Yue Zhao, Xiyang Hu

**发布时间:** 2025-12-04

**备注:** Under review at ACL Rolling Review

### GPT解析

### 总结

本研究引入MAMA框架测量网络结构如何影响多智能体大语言模型系统中的记忆泄露，系统评估六种网络拓扑结构，发现完全连接图泄露最大而链式保护最强，提供架构选择到隐私风险的系统映射和可行指导。

### 背景

图拓扑结构是多智能体大语言模型系统中记忆泄露的基本决定因素，但其影响尚未得到很好的量化。

### 目的

引入MAMA（多智能体内存攻击）框架，测量网络结构如何塑造泄露情况。

### 方法

MAMA在包含标记的个人信息实体的合成文档上运行，执行两阶段协议：Engram（植入私人信息）和Resonance（多轮互动尝试提取），通过精确匹配量化泄露程度。系统评估六种网络拓扑结构，变化智能体数量、攻击者-目标位置和基础模型。

### 主要发现

完全连接图表现最大泄露而链式提供最强保护；较短攻击者-目标距离和较高目标中心性增加脆弱性；泄露在早期轮次急剧上升后趋于平稳；模型选择改变绝对泄露率但保持拓扑排名；时间/位置个人信息属性比身份凭证更易泄露。

### 结论

结果首次提供架构选择到隐私风险的系统映射，建议优先选择稀疏或分层连接，最大化攻击者-目标分离，限制节点度和网络半径，避免绕过枢纽的捷径，实施拓扑感知访问控制。

### 翻译

图拓扑结构是多智能体大语言模型系统中记忆泄露的基本决定因素，但其影响仍然难以量化。我们引入MAMA（多智能体内存攻击）框架，用于测量网络结构如何塑造泄露情况。MAMA在包含标记的个人信息实体的合成文档上运行，从中生成清理过的任务指令。我们执行两阶段协议：Engram（将私人信息植入目标智能体的记忆）和Resonance（多轮互动，攻击者尝试提取信息）。在多达10轮互动中，我们通过精确匹配量化泄露程度，即从攻击者输出中恢复的真实个人信息比例。我们系统评估六种常见网络拓扑结构（完全连接、环形、链式、二叉树、星形和星环形），变化智能体数量、攻击者-目标位置和基础模型。我们的发现显示了一致的模式：完全连接图表现出最大泄露而链式提供最强保护；较短的攻击者-目标图距离和较高的目标中心性显著增加脆弱性；泄露在早期轮次中急剧上升然后趋于平稳；模型选择改变绝对泄露率但保持拓扑排名；时间/位置个人信息属性比身份凭证或受监管标识符更容易泄露。这些结果首次从架构选择到可测量的隐私风险提供了系统映射，提供可行的指导：优先选择稀疏或分层连接，最大化攻击者-目标分离，限制节点度和网络半径，避免绕过枢纽的捷径，并实施拓扑感知的访问控制。


### 论文摘要

Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over up to 10 interaction rounds, we quantify leakage as the fraction of ground-truth PII recovered from attacking agent outputs via exact matching. We systematically evaluate six common network topologies (fully connected, ring, chain, binary tree, star, and star-ring), varying agent counts $n\in\{4,5,6\}$, attacker-target placements, and base models. Our findings reveal consistent patterns: fully connected graphs exhibit maximum leakage while chains provide strongest protection; shorter attacker-target graph distance and higher target centrality significantly increase vulnerability; leakage rises sharply in early rounds before plateauing; model choice shifts absolute leakage rates but preserves topology rankings; temporal/locational PII attributes leak more readily than identity credentials or regulated identifiers. These results provide the first systematic mapping from architectural choices to measurable privacy risk, yielding actionable guidance: prefer sparse or hierarchical connectivity, maximize attacker-target separation, limit node degree and network radius, avoid shortcuts bypassing hubs, and implement topology-aware access controls.

---

## 112. Temporal and Spatial Decomposition for Prospective Studies in Energy Systems under Uncertainty

**论文链接:** [http://arxiv.org/abs/2512.04622v1](http://arxiv.org/abs/2512.04622v1)

**作者:** Camila Martinez Parra, Michel de Lara, Jean-Philippe Chancelier, Pierre Carpentier, Jean-Marc Janin

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究针对可再生能源渗透率增加背景下储能资源价值评估问题，提出了一种基于时空分解和双近似动态规划(DADP)的方法，用于计算欧洲互联电力系统中各市场区域的储能使用价值，并通过数值研究验证了该方法的有效性。

### 背景

随着可再生能源渗透率提高，系统间歇性增加，需要更多使用储能资源来管理这种间歇性。评估储能的机会成本或使用价值变得日益重要，这些价值可通过解决多阶段随机优化问题来推导，随机性来源于净需求、可调度发电可用性以及水坝储能设施的流入量。

### 目的

计算欧洲互联电力系统中每个市场区域的储能使用价值，特别是在法国输电系统运营商RTE进行的 prospective 研究背景下，同时处理大型能源系统中的时间、空间和随机三种主要复杂性。

### 方法

将能源系统建模为有向图（节点代表市场区域，弧代表互联链接），采用时空分解方案处理多节点多阶段随机优化问题，应用双近似动态规划(DADP)方法在时间和空间维度上进行可分解处理，计算仅依赖于各节点局部状态的使用价值。

### 主要发现

在由三十个节点组成的模拟欧洲部分区域的现实系统上进行的数值研究表明，DADP方法与传统方法如随机双动态规划(SDDP)相比能够获得具有竞争力的结果。

### 结论

双近似动态规划(DADP)是处理大规模能源系统时空复杂性的有效方法，能够准确计算储能使用价值且计算效率与传统方法相当。

### 翻译

可再生能源渗透率的增加需要更多地使用储能资源来管理系统的间歇性。因此，评估储能的机会成本或使用价值越来越受到关注，这些价值可以通过解决多阶段随机优化问题来推导。随机性来源于净需求（需求和不可调度发电的聚合）、可调度发电的可用性，以及当所考虑的储能设施是水坝时的流入量。我们的目标是在法国输电系统运营商RTE目前进行的 prospective 研究背景下，计算欧洲互联电力系统中每个市场区域的使用价值。能源系统在数学上被建模为有向图，其中节点代表市场区域，弧代表互联链接。在大型能源系统中，空间复杂性（系统中的三十个节点，每个节点最多有一个聚合储能单元）加剧了时间复杂性（使用两个时间尺度建模的一年期时间范围：带有小时时间步长的周子问题）。这项工作解决了三种主要的复杂性来源：时间、空间和随机性。我们通过结合时空分解方案来解决多节点多阶段随机优化问题。为了高效计算使用价值，我们应用了双近似动态规划(DADP)，该方法可以在时间和空间上进行可分解的处理。这种方法得到了仅依赖于每个节点局部状态的节点使用值，与其他节点无关。我们在一个由三十个节点组成的现实系统（模拟欧洲部分地区）上进行了数值研究，证明与传统方法如随机双动态规划(SDDP)相比，DADP能够获得具有竞争力的结果。


### 论文摘要

The increasing penetration of renewable energy requires greater use of storage resources to manage system intermittency. As a result, there is growing interest in evaluating the opportunity cost of stored energy, or usage values, which can be derived by solving a multistage stochastic optimization problem. Stochasticity arises from net demand (the aggregation of demand and non-dispatchable generation), the availability of dispatchable generation, and inflows when the storage facilities considered are hydroelectric dams. We aim to compute these usage values for each market zone of the interconnected European electricity system, in the context of prospective studies currently conducted by RTE, the French TSO. The energy system is mathematically modelled as a directed graph, where nodes represent market zones and arcs represent interconnection links. In large energy systems, spatial complexity (thirty nodes in the system, each with at most one aggregated storage unit) compounds temporal complexity (a one-year horizon modelled with two timescales: weekly subproblems with hourly time steps). This work addresses three main sources of complexity: temporal, spatial, and stochastic. We tackle the multinode multistage stochastic optimisation problem by incorporating a spatio-temporal decomposition scheme. To efficiently compute usage values, we apply Dual Approximate Dynamic Programming (DADP), which enables tractable decomposition across both time and space. This approach yields nodal usage values that depend solely on the local state of each node, independently of the others. We conduct numerical studies on a realistic system composed of thirty nodes (modelling part of Europe) and show that DADP obtains competitive results when comparing with traditional methods like Stochastic Dual Dynamic Programming (SDDP).

---

## 113. Denoise to Track: Harnessing Video Diffusion Priors for Robust Correspondence

**论文链接:** [http://arxiv.org/abs/2512.04619v1](http://arxiv.org/abs/2512.04619v1)

**作者:** Tianyu Yuan, Yuanbo Yang, Lin-Zhuo Chen, Yao Yao, Zhuzhong Qian

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了HeFT（Head-Frequency Tracker）零样本点跟踪框架，利用预训练视频扩散模型的视觉先验，通过分析视频扩散变换器内部表示，提出头部和频率感知的特征选择策略，实现了接近监督方法的跟踪性能。

### 背景

预训练视频扩散模型具有强大的视觉先验能力，但如何有效利用其编码的时空信息进行点跟踪仍是一个挑战。

### 目的

分析视频扩散变换器（VDiT）内部表示，理解其如何编码时空信息，并基于此开发零样本点跟踪框架。

### 方法

提出HeFT框架，通过单步去噪提取判别性特征，应用基于注意力头和频率成分的特征选择策略，使用软argmax定位与前向-后向一致性检查进行对应估计。

### 主要发现

注意力头作为最小功能单元具有不同专业化功能；低频分量对建立对应关系至关重要，高频分量会引入噪声。

### 结论

HeFT在TAP-Vid基准上实现了最先进的零样本跟踪性能，接近监督方法准确性，无需注释训练数据，展示了视频扩散模型作为基础模型的潜力。

### 翻译

在这项工作中，我们介绍了HeFT（Head-Frequency Tracker），一个利用预训练视频扩散模型视觉先验的零样本点跟踪框架。为了更好地理解它们如何编码时空信息，我们分析了视频扩散变换器（VDiT）的内部表示。我们的分析表明，注意力头作为具有匹配、语义理解和位置编码不同专业化的最小功能单元。此外，我们发现VDiT特征中的低频分量对于建立对应关系至关重要，而高频分量往往会引入噪声。基于这些见解，我们提出了一种头部和频率感知的特征选择策略，联合选择最具信息量的注意力头和低频组件来提高跟踪性能。具体来说，我们的方法通过单步去噪提取判别性特征，应用特征选择，并采用前向-后向一致性检查的软argmax定位进行对应估计。在TAP-Vid基准上的大量实验表明，HeFT实现了最先进的零样本跟踪性能，接近监督方法的准确性，同时消除了对注释训练数据的需求。我们的工作进一步强调了视频扩散模型作为强大基础模型的潜力，为统一视觉基础模型铺平了道路。


### 论文摘要

In this work, we introduce HeFT (Head-Frequency Tracker), a zero-shot point tracking framework that leverages the visual priors of pretrained video diffusion models. To better understand how they encode spatiotemporal information, we analyze the internal representations of Video Diffusion Transformer (VDiT). Our analysis reveals that attention heads act as minimal functional units with distinct specializations for matching, semantic understanding, and positional encoding. Additionally, we find that the low-frequency components in VDiT features are crucial for establishing correspondences, whereas the high-frequency components tend to introduce noise. Building on these insights, we propose a head- and frequency-aware feature selection strategy that jointly selects the most informative attention head and low-frequency components to enhance tracking performance. Specifically, our method extracts discriminative features through single-step denoising, applies feature selection, and employs soft-argmax localization with forward-backward consistency checks for correspondence estimation. Extensive experiments on TAP-Vid benchmarks demonstrate that HeFT achieves state-of-the-art zero-shot tracking performance, approaching the accuracy of supervised methods while eliminating the need for annotated training data. Our work further underscores the promise of video diffusion models as powerful foundation models for a wide range of downstream tasks, paving the way toward unified visual foundation models.

---

## 114. Score Matching for Estimating Finite Point Processes

**论文链接:** [http://arxiv.org/abs/2512.04617v1](http://arxiv.org/abs/2512.04617v1)

**作者:** Haoqun Cao, Yixuan Zhang, Feng Zhou

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究解决了现有score matching估计器在有限点过程中的局限性，提出了基于Janossy测度的形式化框架和加权score matching估计器，并针对非参数模型提出了生存分类增强方法，实验表明该方法能准确恢复强度且效率优于MLE。

### 背景

score matching估计器近年来受到广泛关注，因为它不需要计算归一化常数，从而减轻了最大似然估计的计算挑战。然而，现有针对点过程的score matching估计器存在局限性，主要源于缺乏对score matching在有限点过程中行为的数学严谨分析。

### 目的

开发有限点过程上的score matching形式化框架，解决现有方法的局限性，并提出适用于非参数模型的完整训练目标。

### 方法

基于Janossy测度开发了有限点过程的score matching形式化框架，引入了(自回归)加权score matching估计器，并针对非参数模型提出了生存分类增强方法。

### 主要发现

在经典参数设置中分析了加权score matching估计器的统计性质；对于非参数点过程模型，单独使用score matching不能唯一识别真实分布，但通过生存分类增强可以解决。

### 结论

提出的score matching方法在合成和真实世界的时间及时空数据集上能准确恢复强度，实现与MLE相当的性能，且具有更好的效率。

### 翻译

Score matching估计器近年来受到广泛关注，因为它们消除了计算归一化常数的需求，从而减轻了与最大似然估计相关的计算挑战。虽然已有研究提出了针对点过程的score matching估计器，但本文指出了这些现有方法的局限性，主要源于缺乏对score matching在有限点过程中行为的数学严谨分析——在有限点过程中，许多score matching的常规假设和性质不再成立。为此，我们通过Janossy测度开发了有限点过程上score matching的形式化框架，并在该框架内引入了一种(自回归)加权score matching估计器，我们在经典参数设置中分析了其统计性质。对于一般非参数(如深度)点过程模型，我们表明单独使用score matching不能唯一识别真实分布，由于微妙的归一化问题，我们提出了一个简单的生存分类增强方法，为任何基于强度的时空点过程模型提供了完整的、无需积分的训练目标。在合成和真实世界的时间及时空数据集上的实验表明，我们的方法能准确恢复强度，实现了与MLE相当的性能且效率更高。


### 论文摘要

Score matching estimators have garnered significant attention in recent years because they eliminate the need to compute normalizing constants, thereby mitigating the computational challenges associated with maximum likelihood estimation (MLE).While several studies have proposed score matching estimators for point processes, this work highlights the limitations of these existing methods, which stem primarily from the lack of a mathematically rigorous analysis of how score matching behaves on finite point processes -- special random configurations on bounded spaces where many of the usual assumptions and properties of score matching no longer hold. To this end, we develop a formal framework for score matching on finite point processes via Janossy measures and, within this framework, introduce an (autoregressive) weighted score-matching estimator, whose statistical properties we analyze in classical parametric settings. For general nonparametric (e.g., deep) point process models, we show that score matching alone does not uniquely identify the ground-truth distribution due to subtle normalization issues, and we propose a simple survival-classification augmentation that yields a complete, integration-free training objective for any intensity-based point process model for spatio-temporal case. Experiments on synthetic and real-world temporal and spatio-temporal datasets, demonstrate that our method accurately recovers intensities and achieves performance comparable to MLE with better efficiency.

---

## 115. Factuality and Transparency Are All RAG Needs! Self-Explaining Contrastive Evidence Re-ranking

**论文链接:** [http://arxiv.org/abs/2512.05012v1](http://arxiv.org/abs/2512.05012v1)

**作者:** Francielle Vargas, Daniel Pedronette

**发布时间:** 2025-12-04

**备注:** This work was presented as a poster at the Applied Social Media Lab during the 2025 Synthesizer & Open Showcase at the Berkman Klein Center for Internet & Society at Harvard University

### GPT解析

### 总结

这篇摘要介绍了一种名为'自我解释对比证据重排序'(CER)的新方法，该方法通过对比学习微调嵌入，并为每个检索到的段落生成标记级别的归因理由，从而围绕事实证据重构检索过程。

### 背景

研究背景涉及检索增强生成(RAG)系统，特别是在需要高可靠性的安全关键领域。

### 目的

提高检索准确性，减少RAG系统中的幻觉现象，并提供透明、基于证据的检索，增强安全关键领域的可靠性。

### 方法

使用基于主观性的标准自动选择困难负样本，通过对比学习微调嵌入，为每个检索到的段落生成标记级别的归因理由，使模型将事实理由拉近，同时将主观或误导性解释推远。

### 主要发现

该方法创建了与证据推理明确对齐的嵌入空间，在临床试验报告上的初步实验结果显示CER提高了检索准确性，减轻了RAG系统中的潜在幻觉问题，提供了透明、基于证据的检索。

### 结论

CER方法通过重构检索过程，围绕事实证据进行优化，提高了检索系统的准确性和可靠性，特别是在安全关键领域。

### 翻译

这篇扩展摘要介绍了自我解释对比证据重排序(CER)，一种新颖的方法，它通过对比学习微调嵌入，并为每个检索到的段落生成标记级别的归因理由，从而围绕事实证据重构检索过程。使用基于主观性的标准自动选择困难负样本，强制模型将事实理由拉近，同时将主观或误导性解释推远。因此，该方法创建了一个与证据推理明确对齐的嵌入空间。我们在临床试验报告上评估了我们的方法，初步实验结果表明，CER提高了检索准确性，减轻了RAG系统中的潜在幻觉，并提供了透明、基于证据的检索，增强了可靠性，特别是在安全关键领域。


### 论文摘要

This extended abstract introduces Self-Explaining Contrastive Evidence Re-Ranking (CER), a novel method that restructures retrieval around factual evidence by fine-tuning embeddings with contrastive learning and generating token-level attribution rationales for each retrieved passage. Hard negatives are automatically selected using a subjectivity-based criterion, forcing the model to pull factual rationales closer while pushing subjective or misleading explanations apart. As a result, the method creates an embedding space explicitly aligned with evidential reasoning. We evaluated our method on clinical trial reports, and initial experimental results show that CER improves retrieval accuracy, mitigates the potential for hallucinations in RAG systems, and provides transparent, evidence-based retrieval that enhances reliability, especially in safety-critical domains.

---

## 116. HiPPO: Exploring A Novel Hierarchical Pronunciation Assessment Approach for Spoken Languages

**论文链接:** [http://arxiv.org/abs/2512.04964v1](http://arxiv.org/abs/2512.04964v1)

**作者:** Bi-Cheng Yan, Hsin-Wei Wang, Fu-An Chao, Tien-Hong Lo, Yung-Chang Hsu, Berlin Chen

**发布时间:** 2025-12-04

**备注:** Accepted and to appear in AACL-IJCNLP2025

### GPT解析

### 总结

该研究提出了一种名为HiPPO的分层发音评估模型，专门用于评估第二语言学习者的口语熟练度，特别是在无脚本演讲场景中。研究引入了对比序数正则化器和课程学习策略来提高评估准确性，并在基准数据集上验证了其优越性。

### 背景

大多数自动发音评估研究主要集中在高度受限的朗读任务上，而对无脚本演讲中的发音质量评估相对较少探索。

### 目的

提出一种分层发音评估模型，专门针对口语设计，可以仅基于学习者所说的语音，在多个语言层次上评估第二语言学习者的口语熟练度。

### 方法

引入对比序数正则化器和课程学习策略进行模型训练。前者利用回归目标的序数性质生成具有评分区分性的特征；后者逐步提高训练复杂度，以促进以无脚本语音作为输入的评估任务。

### 主要发现

在Speechocean762基准数据集上进行的实验验证了该方法相对于几种最先进基线的可行性和优越性。

### 结论

HiPPO模型能够在无脚本演讲场景中有效评估第二语言学习者的发音熟练度，为自动发音评估提供了新的研究方向。

### 翻译

自动发音评估旨在通过提供及时和细粒度的诊断反馈，来量化第二语言学习者在目标语言中的发音熟练度。大多数现有的自动发音评估工作主要集中在高度受限的朗读任务上；然而，评估无脚本演讲中的发音质量仍然相对未被充分探索。鉴于此，我们首先提出了HiPPO，一种专为口语设计的分层发音评估模型，该模型仅基于学习者所说的语音，在多个语言层次上评估第二语言学习者的口语熟练度。为了提高整体评估准确性，引入了对比序数正则化器和课程学习策略进行模型训练。前者旨在通过利用回归目标的序数性质来生成具有评分区分性的特征，而后者则逐步提高训练复杂度，以促进以无脚本语音作为输入的评估任务。在Speechocean762基准数据集上进行的实验验证了我们的方法相对于几种最先进基线的可行性和优越性。


### 论文摘要

Automatic pronunciation assessment (APA) seeks to quantify a second language (L2) learner's pronunciation proficiency in a target language by offering timely and fine-grained diagnostic feedback. Most existing efforts on APA have predominantly concentrated on highly constrained reading-aloud tasks (where learners are prompted to read a reference text aloud); however, assessing pronunciation quality in unscripted speech (or free-speaking scenarios) remains relatively underexplored. In light of this, we first propose HiPPO, a hierarchical pronunciation assessment model tailored for spoken languages, which evaluates an L2 learner's oral proficiency at multiple linguistic levels based solely on the speech uttered by the learner. To improve the overall accuracy of assessment, a contrastive ordinal regularizer and a curriculum learning strategy are introduced for model training. The former aims to generate score-discriminative features by exploiting the ordinal nature of regression targets, while the latter gradually ramps up the training complexity to facilitate the assessment task that takes unscripted speech as input. Experiments conducted on the Speechocean762 benchmark dataset validates the feasibility and superiority of our method in relation to several cutting-edge baselines.

---

## 117. 论文ID: 2512.04827v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04827v1.json'

---

## 118. Multi-Loss Learning for Speech Emotion Recognition with Energy-Adaptive Mixup and Frame-Level Attention

**论文链接:** [http://arxiv.org/abs/2512.04551v1](http://arxiv.org/abs/2512.04551v1)

**作者:** Cong Wang, Yizhong Geng, Yuhua Wen, Qifei Li, Yingming Gao, Ruimin Wang, Chunfeng Wang, Hao Li, Ya Li, Wei Chen

**发布时间:** 2025-12-04

**备注:** Submitted to ICASSP 2026. Copyright 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works

### GPT解析

### 总结

提出了一种结合能量自适应混合(EAM)方法和帧级注意力模块(FLAM)的多损失学习(MLL)框架，用于解决语音情感识别中的挑战，并在多个数据集上实现了最先进的性能。

### 背景

语音情感识别(SER)是人机交互中的重要技术，但由于情感复杂性和标注数据稀少，实现高性能具有挑战性。

### 目的

解决情感复杂性和标注数据稀少带来的挑战，提高语音情感识别的性能。

### 方法

提出多损失学习(MLL)框架，集成能量自适应混合(EAM)方法和帧级注意力模块(FLAM)；EAM方法利用基于信噪比的增强技术生成多样化语音样本捕捉细微情感变化；FLAM增强帧级特征提取用于多帧情感线索；MLL策略结合Kullback-Leibler散度、focal、center和监督对比损失优化学习，解决类别不平衡问题并提高特征可分性。

### 主要发现

在四个广泛使用的SER数据集(IEMOCAP、MSP-IMPROV、RAVDESS和SAVEE)上评估，结果表明该方法达到了最先进的性能。

### 结论

提出的多损失学习框架有效且稳健，能够解决语音情感识别中的挑战并实现高性能。

### 翻译

语音情感识别(SER)是人机交互中的重要技术。然而，由于情感复杂性和标注数据稀少，实现高性能具有挑战性。为应对这些挑战，我们提出了一种多损失学习(MLL)框架，集成了能量自适应混合(EAM)方法和帧级注意力模块(FLAM)。EAM方法利用基于信噪比的增强技术生成多样化的语音样本，捕捉细微的情感变化。FLAM增强帧级特征提取，用于多帧情感线索。我们的MLL策略结合Kullback-Leibler散度、focal、center和监督对比损失，优化学习，解决类别不平衡问题，并提高特征可分性。我们在四个广泛使用的SER数据集上评估了我们的方法：IEMOCAP、MSP-IMPROV、RAVDESS和SAVEE。结果表明我们的方法达到了最先进的性能，证明了其有效性和稳健性。


### 论文摘要

Speech emotion recognition (SER) is an important technology in human-computer interaction. However, achieving high performance is challenging due to emotional complexity and scarce annotated data. To tackle these challenges, we propose a multi-loss learning (MLL) framework integrating an energy-adaptive mixup (EAM) method and a frame-level attention module (FLAM). The EAM method leverages SNR-based augmentation to generate diverse speech samples capturing subtle emotional variations. FLAM enhances frame-level feature extraction for multi-frame emotional cues. Our MLL strategy combines Kullback-Leibler divergence, focal, center, and supervised contrastive loss to optimize learning, address class imbalance, and improve feature separability. We evaluate our method on four widely used SER datasets: IEMOCAP, MSP-IMPROV, RAVDESS, and SAVEE. The results demonstrate our method achieves state-of-the-art performance, suggesting its effectiveness and robustness.

---

## 119. Relative Wavefront Error Correction Over a 2.4 km Free-Space Optical Link via Machine Learning

**论文链接:** [http://arxiv.org/abs/2512.04460v1](http://arxiv.org/abs/2512.04460v1)

**作者:** Nathan K. Long, Benjamin P. Dix-Matthews, Alex Frost, John Wallis, Ziqing Wang, Kenneth J. Grant, Robert Malaney

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究针对湍流大气信道中的相干光通信，实验发现了偏振复用参考信标与信号之间的相对波前误差，并开发了基于机器学习的波前校正算法来补偿这些误差，显著提高了系统性能，特别是在连续变量量子密钥分发应用中具有潜在的安全密钥速率提升。

### 背景

在湍流大气信道中进行相干光通信时，通常将参考信标与信息编码信号在传输过程中复用。传统假设认为这两种信号的波前失真是等价的，这一假设在实际应用中被广泛接受。

### 目的

本研究旨在实验验证偏振复用参考信标与信号在大气信道传输后的波前失真是否真的等价，并开发有效的方法来补偿可能存在的相对波前误差，以提高系统性能。

### 方法

研究团队通过2.4公里大气链路实验，测量了偏振复用参考信标与信号之间的相对波前误差。基于这些发现，开发了机器学习波前校正算法，通过相位检索技术来补偿观察到的波前误差。同时，在连续变量量子密钥分发(CV-QKD)的框架下分析了相对波前误差带来的额外噪声贡献。

### 主要发现

1. 实验证明，通过2.4公里大气链路后，偏振复用参考信标和信号之间存在显著的相对波前误差(WFEs)
2. 开发的机器学习波前校正算法能够有效补偿这些误差，使相对相位误差方差减少了多达2/3
3. 相对波前误差在连续变量量子密钥分发系统中引入了额外的噪声

### 结论

研究表明，传统假设中参考信标与信号波前失真等价的情况并不成立。采用基于机器学习的波前校正算法可以有效补偿相对波前误差，显著提高系统性能。特别是在连续变量量子密钥分发应用中，此类算法可能使安全密钥速率提高一个数量级，具有重要的实际应用价值。

### 翻译

在湍流大气信道的相干光通信中，参考信标可以在传输过程中与信息编码信号复用。在这种情况下，通常假设两者的波前失真是等价的。与此假设相反，我们提供了实验证据，显示通过2.4公里大气链路后，偏振复用的参考信标和信号之间存在相对波前误差(WFEs)。我们开发了基于机器学习的波前校正算法，通过相位检索来补偿观察到的WFEs，使相对相位误差方差减少了多达2/3。此外，我们在连续变量量子密钥分发(CV-QKD)的背景下分析了相对WFEs带来的额外噪声贡献，我们的研究结果表明，如果未来的CV-QKD实现采用类似于此处报道的波前校正算法，安全密钥速率可能会提高一个数量级。


### 论文摘要

In coherent optical communication across turbulent atmospheric channels, reference beacons can be multiplexed with information-encoded signals during transmission. In this case, it is commonly assumed that the wavefront distortion of the two is equivalent. In contrast to this assumption, we present experimental evidence of relative wavefront errors (WFEs) between polarization-multiplexed reference beacons and signals, after passing through a 2.4 km atmospheric link. We develop machine learning-based wavefront correction algorithms to compensate for observed WFEs, via phase retrieval, resulting in up to a 2/3 reduction in the relative phase error variance. Further, we analyze the excess noise contributions from relative WFEs in the context of continuous-variable quantum key distribution (CV-QKD), where our findings suggest that if future CV-QKD implementations employ wavefront correction algorithms similar to those reported here, an order of magnitude increase in secure key rates may be forthcoming.

---

## 120. 论文ID: 2512.04356v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04356v1.json'

---

## 121. Gamma-from-Mono: Road-Relative, Metric, Self-Supervised Monocular Geometry for Vehicular Applications

**论文链接:** [http://arxiv.org/abs/2512.04303v1](http://arxiv.org/abs/2512.04303v1)

**作者:** Gasser Elazab, Maximilian Jansen, Michael Unterreiner, Olaf Hellwich

**发布时间:** 2025-12-03

**备注:** Accepted in 3DV 2026

### GPT解析

### 总结

本文提出了一种名为Gamma-from-Mono (GfM)的轻量级单目几何估计方法，能够准确捕捉车辆周围3D环境中的精细道路几何特征，如凸起、坡度和表面不规则性，解决了传统单目深度估计过度平滑这些特征的问题。

### 背景

车辆对3D环境的精确感知对安全和舒适的车辆控制至关重要，但传统单目深度估计往往会过度平滑道路的精细特征，丢失运动规划和稳定性所需的关键信息。

### 目的

开发一种轻量级单目几何估计方法，解决单目重建中的投影模糊问题，准确捕捉精细的道路几何特征。

### 方法

GfM通过解耦全局和局部结构来预测主导道路表面平面和由gamma表示的残差变化（点到平面的垂直偏差与点到相机距离的无量纲度量）。仅需相机高度信息，该方法可通过闭式确定性地恢复度量深度，避免完全的外部标定，并自然地优先考虑近道路细节。其可物理解释的公式使其适合自监督学习，无需大型标注数据集。

### 主要发现

在KITTI和道路表面重建数据集(RSRD)上评估，GfM在深度和gamma估计方面实现了最先进的近场精度，同时保持了有竞争力的全局深度性能。其8.88M参数的轻量级模型能够适应各种相机设置，是第一个在RSRD上评估的自监督单目方法。

### 结论

GfM是一种有效的单目几何估计方法，能够准确捕捉道路的精细几何特征，适用于各种相机设置，并且是自监督的，无需大型标注数据集。

### 翻译

准确感知车辆的3D周围环境，包括精细的道路几何，如凸起、坡度和表面不规则性，对于安全和舒适的车辆控制至关重要。然而，传统的单目深度估计往往会过度平滑这些特征，丢失运动规划和稳定性的关键信息。为此，我们引入了Gamma-from-Mono (GfM)，一种轻量级单目几何估计方法，通过解耦全局和局部结构解决单目重建中的投影模糊问题。GfM预测主导道路表面平面以及由gamma表示的残差变化，gamma是一个从平面垂直偏差的无量纲度量，定义为点高于平面的高度与点距相机距离的比值，基于已建立的平面视差几何学。仅需相机高度信息，这种表示可通过闭式确定性地恢复度量深度，避免完全的外部标定，并自然地优先考虑近道路细节。其可物理解释的公式使其非常适合自监督学习，消除了对大型标注数据集的需求。在KITTI和道路表面重建数据集(RSRD)上评估，GfM在深度和gamma估计方面实现了最先进的近场精度，同时保持了有竞争力的全局深度性能。我们的轻量级8.88M参数模型能够适应各种相机设置，据我们所知，它是第一个在RSRD上评估的自监督单目方法。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决传统单目深度估计方法过度平滑道路几何特征的问题，导致对车辆运动规划和稳定性至关重要的精细尺度道路几何信息丢失。这个问题在现实中很重要，因为准确感知车辆周围3D环境，包括路面颠簸、坡度和表面不规则性，对于自动驾驶的安全和舒适控制至关重要，这些细微的高度变化可能区分可行驶区域和危险区域。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先认识到单目深度估计存在尺度模糊性问题，发现现有方法要么过度关注深度而忽略道路几何，要么依赖外部传感器或地面平面假设。受地面平面先验启发，作者设计了一种对高度敏感的单目几何估计框架。他们借鉴了平面视差几何的概念，特别是γ参数在多视图重建中的应用，以及自监督学习技术如新颖视图合成。但与现有工作不同，作者将γ作为主要预测目标，而不是仅作为训练线索。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是解耦全局和局部结构，通过预测主导道路平面和残差变化来解决单目重建中的投影模糊性。方法引入γ参数（高度与深度的比率）作为无量纲的垂直偏离平面度量，使用γ和道路法线向量表示场景，仅利用相机高度作为已知参数通过闭合形式恢复度量深度。整体流程包括：单帧图像输入→预测γ和道路法线→后处理恢复深度和高度→通过自监督训练（利用光度损失、单应性对齐损失等）→输出精确的近场道路几何和3D重建。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)提出直接预测道路相对的γ表示而非传统深度图；2)首次实现单帧、自监督方法直接回归γ；3)将场景重建分解为全局道路平面和局部每像素结构；4)仅使用相机高度避免完全外部标定；5)轻量级模型(8.88M参数)适应不同相机设置。相比之前工作，这种方法不仅关注深度，还显式建模道路相对量；不依赖多帧光流；仅需相机高度信息；模型更小更适合实时部署；采用自监督学习无需密集标注。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'Gamma-from-Mono提出了一种轻量级、自监督的单目几何估计方法，通过预测道路相对的高度-深度比率和全局道路法线，解决了传统方法中道路几何细节丢失问题，仅需相机高度信息即可恢复精确的近场道路几何，提升自动驾驶的安全性和舒适性。'}


### 论文摘要

Accurate perception of the vehicle's 3D surroundings, including fine-scale road geometry, such as bumps, slopes, and surface irregularities, is essential for safe and comfortable vehicle control. However, conventional monocular depth estimation often oversmooths these features, losing critical information for motion planning and stability. To address this, we introduce Gamma-from-Mono (GfM), a lightweight monocular geometry estimation method that resolves the projective ambiguity in single-camera reconstruction by decoupling global and local structure. GfM predicts a dominant road surface plane together with residual variations expressed by gamma, a dimensionless measure of vertical deviation from the plane, defined as the ratio of a point's height above it to its depth from the camera, and grounded in established planar parallax geometry. With only the camera's height above ground, this representation deterministically recovers metric depth via a closed form, avoiding full extrinsic calibration and naturally prioritizing near-road detail. Its physically interpretable formulation makes it well suited for self-supervised learning, eliminating the need for large annotated datasets. Evaluated on KITTI and the Road Surface Reconstruction Dataset (RSRD), GfM achieves state-of-the-art near-field accuracy in both depth and gamma estimation while maintaining competitive global depth performance. Our lightweight 8.88M-parameter model adapts robustly across diverse camera setups and, to our knowledge, is the first self-supervised monocular approach evaluated on RSRD.

---

## 122. Towards better dense rewards in Reinforcement Learning Applications

**论文链接:** [http://arxiv.org/abs/2512.04302v1](http://arxiv.org/abs/2512.04302v1)

**作者:** Shuyuan Zhang

**发布时间:** 2025-12-03

**备注:** arXiv admin note: substantial text overlap with arXiv:2505.20417

### GPT解析

### 总结

本文探讨了强化学习中密集奖励函数的重要性、挑战及解决方法，旨在提高密集奖励构建的有效性和可靠性。

### 背景

在传统强化学习中，代理通过与环境交互并由奖励信号引导来学习最优策略。然而，当奖励信号稀疏、延迟或与任务目标不一致时，代理学习效果不佳。密集奖励函数提供每一步或状态转换的反馈，可加速学习，但设计不当会导致意外行为、奖励破解或低效探索，特别是在复杂或高维环境中。

### 目的

解决密集奖励构建中的未解决问题，提高不同强化学习应用中密集奖励构建的有效性和可靠性。

### 方法

探索多种方法，包括逆强化学习、基于人类偏好的奖励建模以及内在奖励的自监督学习。

### 主要发现

尽管这些方法有前景，但它们通常在通用性、可扩展性与人类意图一致性之间存在权衡。

### 结论

需要探索多种方法来处理这些未解决的问题，并提高密集奖励构建在不同强化学习应用中的有效性和可靠性。

### 翻译

寻找有意义且准确的密集奖励是强化学习领域的一项基本任务，它使代理能够更有效地探索环境。在传统强化学习设置中，代理通过与环境交互并在奖励信号的引导下学习最优策略。然而，当这些信号稀疏、延迟或与预期任务目标不一致时，代理往往难以有效学习。密集奖励函数在每一步或状态转换中提供信息丰富的反馈，通过塑造代理行为和加速学习提供了一种潜在的解决方案。尽管有这些好处，设计不当的奖励函数可能导致意外行为、奖励破解或低效探索。在复杂或高维环境中，手工设计的奖励难以指定和验证，这个问题尤为突出。为解决这一问题，最近的研究探索了各种方法，包括逆强化学习、基于人类偏好的奖励建模和内在奖励的自监督学习。虽然这些方法提供了有前景的方向，但它们通常在通用性、可扩展性和与人类意图的一致性之间存在权衡。本提案探讨了几种处理这些未解决问题的方法，并提高不同强化学习应用中密集奖励构建的有效性和可靠性。


### 论文摘要

Finding meaningful and accurate dense rewards is a fundamental task in the field of reinforcement learning (RL) that enables agents to explore environments more efficiently. In traditional RL settings, agents learn optimal policies through interactions with an environment guided by reward signals. However, when these signals are sparse, delayed, or poorly aligned with the intended task objectives, agents often struggle to learn effectively. Dense reward functions, which provide informative feedback at every step or state transition, offer a potential solution by shaping agent behavior and accelerating learning. Despite their benefits, poorly crafted reward functions can lead to unintended behaviors, reward hacking, or inefficient exploration. This problem is particularly acute in complex or high-dimensional environments where handcrafted rewards are difficult to specify and validate. To address this, recent research has explored a variety of approaches, including inverse reinforcement learning, reward modeling from human preferences, and self-supervised learning of intrinsic rewards. While these methods offer promising directions, they often involve trade-offs between generality, scalability, and alignment with human intent. This proposal explores several approaches to dealing with these unsolved problems and enhancing the effectiveness and reliability of dense reward construction in different RL applications.

---

## 123. POLARIS: Is Multi-Agentic Reasoning the Next Wave in Engineering Self-Adaptive Systems?

**论文链接:** [http://arxiv.org/abs/2512.04702v1](http://arxiv.org/abs/2512.04702v1)

**作者:** Divyansh Pandey, Vyakhya Gupta, Prakhar Singhal, Karthik Vaidhyanathan

**发布时间:** 2025-12-04

**备注:** Accepted as a short paper at SEAMS 2026

### GPT解析

### 总结

研究介绍了一个名为POLARIS的三层多智能体自适应性框架，该框架整合了低延迟适配器层、透明推理层和元学习层，通过共享知识和预测模型处理不确定性，从过去行动中学习并发展策略，在两个自适应示例上的评估显示其性能优于最先进的基线方法。

### 背景

现代软件生态系统的规模、复杂性、互连性和自主性不断增长，引入了前所未有的不确定性，挑战了传统自适应的基础。现有方法（通常是规则驱动控制器或孤立的学习组件）难以推广到新上下文或协调分布式子系统的响应，无法应对突发的未知情况。自适应2.0的最近讨论强调AI和自适应系统之间的平等伙伴关系。

### 目的

开发一个能够处理不确定性、从过去经验中学习并随时间发展策略的自适应框架，创建能够预测变化并保持有弹性、目标导向行为的系统，推进自适应用户到3.0时代，类似于软件3.0范式。

### 方法

提出POLARIS，一个三层多智能体自适应性框架，包括：1)低延迟适配器层，用于监控和安全执行；2)透明推理层，使用工具感知、可解释的智能体生成和验证计划；3)元层，记录经验并随时间元学习改进的自适应策略。通过共享知识和预测模型实现。

### 主要发现

POLARIS在两个自适应示例(SWIM和SWITCH)上的初步评估显示，其性能始终优于最先进的基线方法，系统能够预测变化并保持有弹性、目标导向的行为。

### 结论

POLARIS标志着向自适应3.0的转变，类似于软件3.0范式，在这一范式中，系统不仅从环境中学习，还推理和演化自己的自适应过程，能够持续改进以应对新的挑战。

### 翻译

现代软件生态系统的不断增长的规模、复杂性、互连性和自主性引入了前所未有的不确定性，挑战了传统自适应的基础。现有方法，通常是规则驱动控制器或孤立的学习组件，难以推广到新上下文或协调分布式子系统的响应，使它们无法应对突发的未知情况。最近关于自适应2.0的讨论强调AI和自适应系统之间的平等伙伴关系，将学习驱动的智能与自适应控制相结合，以实现预测性和主动性行为。基于这一基础，我们引入了POLARIS，一个三层多智能体自适应框架，超越了反应式适应。POLARIS整合：1)一个用于监控和安全执行的低延迟适配器层；2)一个透明推理层，使用工具感知、可解释的智能体生成和验证计划；3)一个元层，记录经验并随时间元学习改进的自适应策略。通过共享知识和预测模型，POLARIS处理不确定性，从过去行动中学习，并发展其策略，使系统能够预测变化并保持有弹性、目标导向的行为。在两个自适应示例SWIM和SWITCH上的初步评估显示，POLARIS始终优于最先进的基线。我们认为这标志着向自适应3.0的转变，类似于软件3.0：一个范式，其中系统不仅从环境中学习，还推理和演化自己的自适应过程，持续改进以应对新的挑战。


### 论文摘要

The growing scale, complexity, interconnectivity, and autonomy of modern software ecosystems introduce unprecedented uncertainty, challenging the foundations of traditional self-adaptation. Existing approaches, typically rule-driven controllers or isolated learning components, struggle to generalize to novel contexts or coordinate responses across distributed subsystems, leaving them ill-equipped for emergent unknown unknowns. Recent discussions on Self-Adaptation 2.0 emphasize an equal partnership between AI and adaptive systems, merging learning-driven intelligence with adaptive control for predictive and proactive behavior. Building on this foundation, we introduce POLARIS, a three-layer multi-agentic self-adaptation framework that advances beyond reactive adaptation. POLARIS integrates: (1) a low-latency Adapter layer for monitoring and safe execution, (2) a transparent Reasoning layer that generates and verifies plans using tool-aware, explainable agents, and (3) a Meta layer that records experiences and meta-learns improved adaptation policies over time. Through shared knowledge and predictive models, POLARIS handles uncertainty, learns from past actions, and evolves its strategies, enabling systems that anticipate change and maintain resilient, goal-directed behavior. Preliminary evaluation on two self-adaptive exemplars, SWIM and SWITCH, shows that POLARIS consistently outperforms state-of-the-art baselines. We argue this marks a shift toward Self-Adaptation 3.0, akin to Software 3.0: a paradigm where systems not only learn from their environment but also reason about and evolve their own adaptation processes, continuously improving to meet novel challenges.

---

## 124. ADAPT: Learning Task Mixtures for Budget-Constrained Instruction Tuning

**论文链接:** [http://arxiv.org/abs/2512.04555v1](http://arxiv.org/abs/2512.04555v1)

**作者:** Pritam Kadasi, Abhishek Upperwal, Mayank SIngh

**发布时间:** 2025-12-04

**备注:** Under Review

### GPT解析

### 总结

ADAPT是一种元学习算法，在多任务指令调整中学习任务采样比例，在有限的token预算下实现自适应课程分配。

### 背景

在多任务指令调整中，通常需要手工固定任务权重，这种方法可能不是最优的。

### 目的

开发一种能够自适应分配token到不同任务的算法，避免手工固定任务权重的局限性。

### 方法

ADAPT维护一个任务上的连续分布，通过平滑的最坏情况验证目标的元梯度来更新分布，产生自适应课程，将更多token分配给有用任务，同时避免崩溃。

### 主要发现

ADAPT与最佳静态混合相比，匹配或略微提高了平均下游性能，同时使用更少的有效训练token，并将预算重新分配给更难、与基准对齐的任务。

### 结论

ADAPT是一种有效的元学习方法，可以在有限的token预算下自适应地分配任务资源，提高训练效率。

### 翻译

我们提出了ADAPT，一种元学习算法，在明确的token预算下，为多任务指令调整学习任务采样比例。ADAPT不是手工固定任务权重，而是维护一个任务上的连续分布，并通过平滑的最坏情况验证目标的元梯度来更新它，从而产生一个自适应课程，将更多token分配给有用任务，同时避免崩溃。我们在三个约10亿参数的开源LLMs（Gemma-3-1B、LLaMA-3.2-1B、Qwen-0.6B）上实例化ADAPT，在20种自然指令任务类型上训练，预算分别为可用监督token的1%、5%和10%，并与具有均匀和按比例混合的强监督微调基线进行比较。我们在11个跨领域基准测试上进行了评估，涵盖推理、阅读理解、代码生成和指令遵循，我们发现ADAPT与最佳静态混合相比，匹配或略微提高了平均下游性能，同时使用了更少的有效训练token，并将预算重新分配给更难、与基准对齐的任务。


### 论文摘要

We propose ADAPT, a meta-learning algorithm that \emph{learns} task sampling proportions under an explicit token budget for multi-task instruction tuning. Instead of fixing task weights by hand, \adapt{} maintains a continuous distribution over tasks and updates it via meta-gradients of a smooth worst-case validation objective, inducing an adaptive curriculum that allocates more tokens to useful tasks while avoiding collapse. We instantiate ADAPT on three $\sim$1B-parameter open-weight LLMs (Gemma-3-1B, LLaMA-3.2-1B, Qwen-0.6B), training on 20 Natural Instructions task types under budgets of $1\%$, $5\%$, and $10\%$ of the available supervised tokens, and compare against strong supervised fine-tuning baselines with uniform and size-proportional mixing. We conduct evaluations on 11 out-of-domain benchmarks spanning reasoning, reading comprehension, code generation, and instruction following, we find that ADAPT matches or slightly improves average downstream performance relative to the best static mixture, while using fewer effective training tokens and reallocating budget toward harder, benchmark-aligned tasks.

---

## 125. Fourier-Attentive Representation Learning: A Fourier-Guided Framework for Few-Shot Generalization in Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2512.04395v1](http://arxiv.org/abs/2512.04395v1)

**作者:** Hieu Dinh Trung Pham, Huy Minh Nhat Nguyen, Cuong Tuan Nguyen

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究提出了Fourier-Attentive Representation Learning (FARL)框架，通过傅里叶分析明确解耦视觉表征中的结构和风格特征，以提高视觉-语言模型的小样本学习能力和泛化性能。

### 背景

大规模预训练视觉-语言模型（VLMs）已展现出强大小样本学习能力，但这些方法通常学习整体表征，其中图像的域不变结构与域特定风格是隐式纠缠在一起的，限制了模型的泛化能力。

### 目的

通过解耦视觉表征中的域不变结构和域特定风格，进一步增强视觉-语言模型的小样本学习能力和泛化性能。

### 方法

提出Fourier-Attentive Representation Learning (FARL)框架，核心是双交叉注意力机制，其中可学习的表征令牌分别查询图像的结构特征（来自相位谱）和风格特征（来自振幅谱），然后将这些丰富、解耦的令牌注入到VLM编码器深处以引导适应，包括非对称注入策略，迫使模型学习更强大的视觉-语言对齐。

### 主要发现

在15个数据集上的大量实验证明了该方法的有效性，表明通过解耦视觉表征可以显著提高视觉-语言模型的小样本学习能力和泛化性能。

### 结论

通过明确解耦视觉表征中的结构和风格特征，FARL框架能够有效增强视觉-语言模型的小样本学习能力，提高模型的泛化性能，为视觉-语言模型的进一步发展提供了新思路。

### 翻译

大规模预训练视觉-语言模型（VLMs）已展现出强大的小样本学习能力。然而，这些方法通常学习整体表征，其中图像的域不变结构与域特定风格是隐式纠缠在一起的。这为通过解耦这些视觉线索来进一步增强泛化能力提供了机会。在本文中，我们提出了Fourier-Attentive Representation Learning (FARL)，这是一个新框架，通过使用傅里叶分析明确解耦视觉表征来解决这个问题。我们方法的核心是一个双交叉注意力机制，其中可学习的表征令牌分别查询图像的结构特征（来自相位谱）和风格特征（来自振幅谱）。这一过程产生丰富、解耦的令牌，然后被深度注入到VLM编码器中以引导适应。我们的设计包括非对称注入策略，迫使模型学习更强大的视觉-语言对齐。在15个数据集上的大量实验证明了我们方法的有效性。


### 论文摘要

Large-scale pre-trained Vision-Language Models (VLMs) have demonstrated strong few-shot learning capabilities. However, these methods typically learn holistic representations where an image's domain-invariant structure is implicitly entangled with its domain-specific style. This presents an opportunity to further enhance generalization by disentangling these visual cues. In this paper, we propose Fourier-Attentive Representation Learning (FARL), a novel framework that addresses this by explicitly disentangling visual representations using Fourier analysis. The core of our method is a dual cross-attention mechanism, where learnable representation tokens separately query an image's structural features (from the phase spectrum) and stylistic features (from the amplitude spectrum). This process yields enriched, disentangled tokens that are then injected deep into the VLM encoders to guide adaptation. Our design, which includes an asymmetric injection strategy, forces the model to learn a more robust vision-language alignment. Extensive experiments on 15 datasets demonstrate the effectiveness of our approach.

---

## 126. Contact-Implicit Modeling and Simulation of a Snake Robot on Compliant and Granular Terrain

**论文链接:** [http://arxiv.org/abs/2512.05008v1](http://arxiv.org/abs/2512.05008v1)

**作者:** Haroon Hublikar

**发布时间:** 2025-12-04

### GPT解析

### 总结

该研究提出了一个统一的建模和仿真框架，用于分析COBRA蛇机器人在刚性、柔性和颗粒状地形上的侧向移动和翻滚运动。

### 背景

蛇机器人在不同地形上的运动表现受到多种物理因素影响，需要综合考虑摩擦相互作用、地形变形和颗粒状环境等因素。

### 目的

建立一个统一的建模和仿真框架，准确预测蛇机器人在多种地形上的运动表现，从实时控制到高保真物理仿真。

### 方法

使用接触隐式公式建模侧向移动时的分布式摩擦相互作用，通过MATLAB Simscape仿真和物理实验验证；集成Chrono的土壤接触模型模拟地形变形效应；使用Chrono DEM引擎模拟颗粒状环境中的高能量滚动运动。

### 主要发现

刚性地面模型能提供准确的短期运动预测，而连续介质和基于颗粒的地形建模对于软性和高度动态环境中的可靠移动性分析是必要的；在颗粒状环境中，土壤破坏、间歇性离地和能量耗散机制是重要因素。

### 结论

该研究建立了一个分层仿真管道，提高了机器人在具有挑战性的非结构化环境中的稳健性和地形感知运动能力。

### 翻译

本论文提出了一个统一的建模和仿真框架，用于分析COBRA蛇机器人在刚性、柔性和颗粒状地形上的侧向移动和翻滚运动。使用接触隐式公式建模侧向移动过程中的分布式摩擦相互作用，并通过MATLAB Simscape仿真和刚性地面及松散沙地上的物理实验进行了验证。为了捕捉地形变形效应，将Chrono的土壤接触模型(SCM)与多体动力学相结合，能够预测可变形基底上的滑动、下沉和负载重新分布，这些因素会降低步态效率。对于陡坡上的高能量滚动运动，使用Chrono DEM引擎模拟颗粒分辨率的颗粒相互作用，揭示了刚性模型无法捕捉的土壤破坏、间歇性离地和能量耗散机制。这些方法共同涵盖了实时控制导向仿真和高保真颗粒物理仿真。结果表明，刚性地面模型能提供准确的短期运动预测，而连续介质和基于颗粒的地形建模对于软性和高度动态环境中的可靠移动性分析是必要的。这项工作建立了一个分层仿真管道，提高了机器人在具有挑战性的非结构化环境中运行的稳健性和地形感知运动能力。


### 论文摘要

This thesis presents a unified modeling and simulation framework for analyzing sidewinding and tumbling locomotion of the COBRA snake robot across rigid, compliant, and granular terrains. A contact-implicit formulation is used to model distributed frictional interactions during sidewinding, and validated through MATLAB Simscape simulations and physical experiments on rigid ground and loose sand. To capture terrain deformation effects, Project Chrono's Soil Contact Model (SCM) is integrated with the articulated multibody dynamics, enabling prediction of slip, sinkage, and load redistribution that reduce stride efficiency on deformable substrates. For high-energy rolling locomotion on steep slopes, the Chrono DEM Engine is used to simulate particle-resolved granular interactions, revealing soil failure, intermittent lift-off, and energy dissipation mechanisms not captured by rigid models. Together, these methods span real-time control-oriented simulation and high-fidelity granular physics. Results demonstrate that rigid-ground models provide accurate short-horizon motion prediction, while continuum and particle-based terrain modeling becomes necessary for reliable mobility analysis in soft and highly dynamic environments. This work establishes a hierarchical simulation pipeline that advances robust, terrain-aware locomotion for robots operating in challenging unstructured settings.

---

## 127. Isostructural phase transition and equation of state of type-I and type-VIII metallic sodium borosilicide clathrates

**论文链接:** [http://arxiv.org/abs/2512.04878v1](http://arxiv.org/abs/2512.04878v1)

**作者:** M. Demoucron, S. Pandolfi, Y. Guarnelli, B. Baptiste, P. Chevignon, N. Guignot, D. Portehault, T. A. Strobel, W. A. Crichton, Y. Le Godec, A. Courac

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究探讨了硅基笼状化合物的高压行为，发现I型钠硼硅笼状化合物在13 GPa压力下发生等结构相变，而VIII型则表现出常规弹性压缩，并通过反射光谱证实了这些材料具有金属性质。

### 背景

硅基笼状化合物的电子性质可以通过将硼掺入硅笼网络中进行调控。与其他碱金属半导体Zintl硼硅化物相比，钠硼硅笼状化合物因其非寻常的化学计量比和预期的金属性质而脱颖而出。

### 目的

研究I型和VIII型钠硼硅笼状化合物的高压行为，特别关注压力诱导的相变现象。

### 方法

通过实验研究I型和VIII型钠硼硅笼状化合物的高压行为，并使用反射光谱法在宽能量范围内分析其金属性质。

### 主要发现

I型钠硼硅笼状化合物在13 GPa压力下发生等结构相变，特征是体积突然塌陷，归因于压力诱导的硅原子扩散到阳离子位点。这种机制为理论预测提供了首次实验验证。等结构相变仅在I型硼硅化物中观察到，而VIII型硼硅化物相表现出常规弹性压缩。

### 结论

I型和VIII型钠硼硅笼状化合物在高压下表现出不同的相变行为，这些发现为理解这类材料的高压性质提供了重要见解，并验证了理论预测。

### 翻译

硅基笼状化合物的电子性质可以通过将硼掺入硅笼网络中进行调控。与其他碱金属半导体Zintl硼硅化物相比，钠硼硅笼状化合物因其非寻常的化学计量比和预期的金属性质而脱颖而出。在本研究中，我们报道了对I型和VIII型钠硼硅笼状化合物高压行为的实验研究。仅在I型钠硼硅笼状化合物中观察到等结构相变，特征是在13 GPa下的体积突然塌陷。这种相变归因于压力诱导的硅原子扩散到阳离子位点。这种机制为这类材料理论上预测的相变提供了首次实验验证。等结构相变仅在I型硼硅化物中观察到。相比之下，VIII型硼硅化物相表现出常规弹性压缩。使用宽能量范围内的反射光谱法确立了金属性质，与硼含量的晶体学数据相符。


### 论文摘要

Electronic properties of silicon-based clathrates can be tuned by boron incorporation into the silicon cage network. Sodium borosilicides clathrate outstands with uncommon stoichiometry and expected metallic properties, in contrast to other alcali metal semiconductive Zintl borosilicides. In this study, we report an experimental investigation of the high-pressure behavior of type-I and type-VIII sodium borosilicide clathrates. An isostructural phase transition, marked by an abrupt volume collapse at 13 GPa, is observed exclusively in type-I sodium borosilicide clathrates. This transition is attributed to the pressure-induced diffusion of silicon atoms into cationic sites. This mechanism provides the first experimental validation of a transition predicted theoretically for this class of materials. Isostructural phase transitions were only observed in type-I borosilicide. In contrast, the type-VIII borosilicide phase exhibits conventional elastic compression. The metallic character was established using reflectance spectroscopy over a wide energy range, in good agreement with crystallographic data on the boron content.

---

## 128. Movable Antenna Assisted Flexible Beamforming for Integrated Sensing and Communication in Vehicular Networks

**论文链接:** [http://arxiv.org/abs/2512.04802v1](http://arxiv.org/abs/2512.04802v1)

**作者:** Luyang Sun, Zhiqing Wei, Haotian Liu, Kan Yu, Zhendong Li, Zhiyong Feng

**发布时间:** 2025-12-04

### GPT解析

### 总结

本研究探讨了可移动天线(MA)技术在集成感知与通信(ISAC)系统中的应用，特别是在车对基础设施(V2I)网络中的性能优化。作者提出了两种算法来优化发射天线位置、波束成形和功率分配，以提高通信速率和感知性能。

### 背景

集成感知与通信(ISAC)已被认为是第六代无线网络的关键技术，而可移动天线(MA)技术能够提供额外的空间自由度，显著提升ISAC系统性能。在动态的车对基础设施(V2I)网络中，需要优化天线位置和通信参数以适应环境变化。

### 目的

研究ISAC辅助的V2I网络中，如何通过结合扩展卡尔曼滤波预测和实时优化，联合优化发射天线位置、波束成形和功率分配向量，以提高系统性能。

### 方法

作者提出了两种算法：1)预处理-舒尔补-投影梯度上升算法，适用于没有感知服务质量(QoS)约束的场景，探索感知性能的潜在范围，为后续约束优化提供参考和热启动；2)启发式反射投影动态粒子群优化算法，适用于感知QoS约束场景，在非凸约束条件下以少量迭代实现显著性能提升。

### 主要发现

仿真结果表明，这些方法能够同时提高通信总速率和运动参数估计的克拉美罗下界(CRLB)的下限，验证了可移动天线辅助波束成形在动态V2I ISAC网络中的有效性。

### 结论

可移动天线技术与集成感知与通信系统的结合，特别是在动态V2I网络中，通过优化天线位置和通信参数，能够有效提升系统性能，为第六代无线网络提供了可行的技术方案。

### 翻译

集成感知与通信(ISAC)已被认为是第六代无线网络的关键技术，而可移动天线(MA)技术获得的额外空间自由度可以显著提高ISAC系统性能。本文考虑了一个ISAC辅助的车对基础设施(V2I)网络，其中基于扩展卡尔曼滤波的预测与实时优化相结合，以在动态环境中联合优化发射天线位置、波束成形和功率分配向量。我们提出了两种算法：一种是无感知服务质量(QoS)约束场景下的预处理-舒尔补-投影梯度上升算法，它探索感知性能的潜在范围，为后续约束优化提供参考和热启动；另一种是有感知QoS约束场景下的启发式反射投影动态粒子群优化算法，它在非凸约束条件下以少量迭代实现显著性能提升。仿真结果表明，这些方法提高了通信总速率和运动参数估计的克拉美罗下界(CRLB)的下限，验证了可移动天线辅助波束成形在动态V2I ISAC网络中的有效性。


### 论文摘要

Integrated sensing and communication (ISAC) has been recognized as a key technology in sixth-generation wireless networks, and the additional spatial degrees of freedom obtained by movable antenna (MA) technology can significantly improve the performance of ISAC systems. This paper considers an ISAC-assisted vehicle-to-infrastructure (V2I) network, where extended kalman filter-based prediction is combined with real-time optimization to jointly optimize transmit antenna positions and beamforming and power allocation vectors in dynamic environments. We propose two algorithms: a preprocessing-schur complement-projected gradient ascent algorithm for scenarios without sensing quality of service (QoS) constraints, which explores the potential range of sensing performance to provide reference and warm-starting for subsequent constrained optimization; and a heuristic reflective projected dynamic particle swarm optimization algorithm for sensing QoS-constrained scenarios, which achieves substantial performance gains under non-convex constraints with a small number of iterations. Simulation results demonstrate that these approaches enhance both the communication sum-rate and the lower of the Cramér-Rao lower bound of motion parameter estimation, validating the effectiveness of MA-assisted beamforming in dynamic V2I ISAC networks.

---

## 129. Back to Basics: Motion Representation Matters for Human Motion Generation Using Diffusion Model

**论文链接:** [http://arxiv.org/abs/2512.04499v1](http://arxiv.org/abs/2512.04499v1)

**作者:** Yuduo Jin, Brandon Haworth

**发布时间:** 2025-12-04

### GPT解析

### 总结

该论文通过受控研究探讨了运动表示和损失函数的基本问题，使用代理运动扩散模型进行了实证研究，评估了不同运动表示的性能，比较了训练时间，并在大型数据集上进行了分析，研究结果揭示了不同运动表示在不同数据集中的性能差异以及配置对模型训练的影响。

### 背景

扩散模型已成为人类运动合成中被广泛使用且成功的方法，任务导向的扩散模型显著促进了动作到运动、文本到运动和音频到运动的应用。

### 目的

研究运动表示和损失函数的基本问题，列出生成运动扩散模型工作流程中各种决策的影响，增强对潜在数据分布的理解，为改进条件运动扩散模型提供基础。

### 方法

基于代理运动扩散模型（MDM）进行实证研究，使用v损失作为MDM（vMDM）的预测目标，其中v是运动数据和噪声的加权和；评估文献中六种常见的运动表示，比较其在质量和多样性指标方面的性能；比较各种配置下的训练时间；在大型运动数据集上进行评估分析。

### 主要发现

不同数据集中的运动表示存在明显的性能差异；不同配置对模型训练有影响；这些决策对运动扩散模型的结果具有重要性和有效性。

### 结论

研究结果提供了对运动扩散模型中各种决策影响的理解，这些发现有助于改进条件运动扩散模型。

### 翻译

扩散模型已成为人类运动合成中被广泛使用且成功的方法论。任务导向的扩散模型显著促进了动作到运动、文本到运动和音频到运动的应用。在本文中，我们通过受控研究探讨了关于运动表示和损失函数的基本问题，并列举了生成运动扩散模型工作流程中各种决策的影响。为了回答这些问题，我们基于代理运动扩散模型（MDM）进行了实证研究。我们在MDM上应用v损失作为预测目标（vMDM），其中v是运动数据和噪声的加权和。我们的目标是增强对潜在数据分布的理解，并为改进条件运动扩散模型的状态提供基础。首先，我们评估了文献中六种常见的运动表示，并比较了它们在质量和多样性指标方面的性能。其次，我们比较了各种配置下的训练时间，以阐明如何加速运动扩散模型的训练过程。最后，我们还在大型运动数据集上进行了评估分析。我们的实验结果表明，不同数据集中的运动表示存在明显的性能差异。我们的结果还展示了不同配置对模型训练的影响，并表明这些决策对运动扩散模型结果的重要性和有效性。


### 论文摘要

Diffusion models have emerged as a widely utilized and successful methodology in human motion synthesis. Task-oriented diffusion models have significantly advanced action-to-motion, text-to-motion, and audio-to-motion applications. In this paper, we investigate fundamental questions regarding motion representations and loss functions in a controlled study, and we enumerate the impacts of various decisions in the workflow of the generative motion diffusion model. To answer these questions, we conduct empirical studies based on a proxy motion diffusion model (MDM). We apply v loss as the prediction objective on MDM (vMDM), where v is the weighted sum of motion data and noise. We aim to enhance the understanding of latent data distributions and provide a foundation for improving the state of conditional motion diffusion models. First, we evaluate the six common motion representations in the literature and compare their performance in terms of quality and diversity metrics. Second, we compare the training time under various configurations to shed light on how to speed up the training process of motion diffusion models. Finally, we also conduct evaluation analysis on a large motion dataset. The results of our experiments indicate clear performance differences across motion representations in diverse datasets. Our results also demonstrate the impacts of distinct configurations on model training and suggest the importance and effectiveness of these decisions on the outcomes of motion diffusion models.

---

## 130. 论文ID: 2512.04480v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04480v1.json'

---

## 131. FMA-Net++: Motion- and Exposure-Aware Real-World Joint Video Super-Resolution and Deblurring

**论文链接:** [http://arxiv.org/abs/2512.04390v1](http://arxiv.org/abs/2512.04390v1)

**作者:** Geunhyuk Youk, Jihyong Oh, Munchurl Kim

**发布时间:** 2025-12-04

**备注:** 20 pages, 15 figures. Project Page: https://kaist-viclab.github.io/fmanetpp_site/

### GPT解析

### 总结

本文提出FMA-Net++框架，用于解决现实世界中运动与动态变化曝光耦合导致的复杂视频退化问题，实现了联合视频超分辨率和去模糊功能。

### 背景

现实世界的视频恢复受到运动与动态变化曝光的复杂退化的困扰，这是一个先前工作中很大程度上被忽视的关键挑战，也是自动曝光或低光捕捉的常见伪影。

### 目的

开发一个框架，明确建模运动和动态变化曝光的耦合效应，实现高质量的联合视频超分辨率和去模糊。

### 方法

提出FMA-Net++框架，采用基于层次细化和双向传播块构建的序列级架构，支持并行、长程时序建模。包含曝光时间感知调制层和曝光感知的流引导动态滤波模块，将退化学习与恢复解耦，提高准确性和效率。

### 主要发现

引入REDS-ME和REDS-RE基准来评估真实捕捉条件下的性能。仅在合成数据上训练的FMA-Net++在新的基准和GoPro上实现了最先进的准确性和时间一致性，在恢复质量和推理速度上都优于最近的方法，并能很好地推广到具有挑战性的真实世界视频。

### 结论

FMA-Net++是处理运动与动态变化曝光耦合退化的有效框架，在视频超分辨率和去模糊任务中表现出色，具有实际应用价值。

### 翻译

现实世界的视频恢复受到运动与动态变化曝光的复杂退化的困扰——这是一个先前工作中很大程度上被忽视的关键挑战，也是自动曝光或低光捕捉的常见伪影。我们提出了FMA-Net++，一个用于联合视频超分辨率和去模糊的框架，明确建模了运动和动态变化曝光的这种耦合效应。FMA-Net++采用基于层次细化和双向传播块构建的序列级架构，支持并行、长程时序建模。在每个块中，一个曝光时间感知调制层根据每帧曝光调节特征，进而驱动一个曝光感知的流引导动态滤波模块来推断运动和曝光感知的退化核。FMA-Net++将退化学习与恢复解耦：前者预测曝光和运动感知先验来指导后者，提高准确性和效率。为了在真实捕捉条件下进行评估，我们引入了REDS-ME（多曝光）和REDS-RE（随机曝光）基准。仅在合成数据上训练的FMA-Net++在我们的新基准和GoPro上实现了最先进的准确性和时间一致性，在恢复质量和推理速度上都优于最近的方法，并能很好地推广到具有挑战性的真实世界视频。


### 论文摘要

Real-world video restoration is plagued by complex degradations from motion coupled with dynamically varying exposure - a key challenge largely overlooked by prior works and a common artifact of auto-exposure or low-light capture. We present FMA-Net++, a framework for joint video super-resolution and deblurring that explicitly models this coupled effect of motion and dynamically varying exposure. FMA-Net++ adopts a sequence-level architecture built from Hierarchical Refinement with Bidirectional Propagation blocks, enabling parallel, long-range temporal modeling. Within each block, an Exposure Time-aware Modulation layer conditions features on per-frame exposure, which in turn drives an exposure-aware Flow-Guided Dynamic Filtering module to infer motion- and exposure-aware degradation kernels. FMA-Net++ decouples degradation learning from restoration: the former predicts exposure- and motion-aware priors to guide the latter, improving both accuracy and efficiency. To evaluate under realistic capture conditions, we introduce REDS-ME (multi-exposure) and REDS-RE (random-exposure) benchmarks. Trained solely on synthetic data, FMA-Net++ achieves state-of-the-art accuracy and temporal consistency on our new benchmarks and GoPro, outperforming recent methods in both restoration quality and inference speed, and generalizes well to challenging real-world videos.

---

## 132. 论文ID: 2512.04355v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04355v1.json'

---

## 133. SmartAlert: Implementing Machine Learning-Driven Clinical Decision Support for Inpatient Lab Utilization Reduction

**论文链接:** [http://arxiv.org/abs/2512.04354v1](http://arxiv.org/abs/2512.04354v1)

**作者:** April S. Liang, Fatemeh Amrollahi, Yixing Jiang, Conor K. Corbin, Grace Y. E. Kim, David Mui, Trevor Crowell, Aakash Acharya, Sreedevi Mony, Soumya Punnathanam, Jack McKeown, Margaret Smith, Steven Lin, Arnold Milstein, Kevin Schulman, Jason Hom, Michael A. Pfeffer, Tho D. Pham, David Svec, Weihan Chu, Lisa Shieh, Christopher Sharp, Stephen P. Ma, Jonathan H. Chen

**发布时间:** 2025-12-04

**备注:** 22 pages, 5 figures

### GPT解析

### 总结

该研究介绍并评估了SmartAlert系统，一个机器学习驱动的临床决策支持系统，用于预测稳定的实验室结果以减少不必要的重复测试。在两家医院的八个急性护理单元进行的随机对照试点中，该系统成功减少了全血细胞计数测试，且不影响患者安全。

### 背景

重复进行不太可能产生有用临床信息的实验室测试是一种常见做法，给患者带来负担并增加医疗成本。教育和反馈干预效果有限，而一般的测试限制和电子警报又会阻碍适当的临床护理。

### 目的

引入并评估SmartAlert系统，一个集成到电子健康记录中的机器学习驱动的临床决策支持系统，用于预测稳定的实验室结果，减少不必要的重复测试。

### 方法

这是一项案例研究，描述了在两家医院的八个急性护理单元针对9270次入院进行的随机对照试点实施过程、挑战和经验教训，时间从2024年8月15日到2025年3月15日，目标是减少全血细胞计数(CBC)的使用。

### 主要发现

结果显示，在SmartAlert显示后的52小时内，CBC结果数量显著减少(1.54 vs 1.82)，且对次要安全结果没有不良影响，代表重复测试相对减少了15%。实施经验包括：在临床背景下解释概率模型预测、与利益相关者合作定义可接受的模型行为、在临床环境中部署复杂模型的治理流程、用户界面设计考虑、与临床运营优先事项的对齐，以及从终端用户获取定性反馈的价值。

### 结论

由深思熟虑的实施和治理过程支持的机器学习驱动的临床决策支持系统，可以为住院实验室测试提供精准指导，从而安全地减少不必要的重复测试。

### 翻译

重复进行不太可能产生临床有用信息的实验室测试是一种常见做法，给患者带来负担并增加医疗成本。教育和反馈干预的成功有限，而一般的测试限制和电子警报会阻碍适当的临床护理。我们引入并评估了SmartAlert，这是一个机器学习驱动的临床决策支持系统，集成到电子健康记录中，用于预测稳定的实验室结果以减少不必要的重复测试。本案例研究描述了在两家医院八个急性护理单元针对9270次入院进行的随机对照试点中的实施过程、挑战和经验教训，时间从2024年8月15日到2025年3月15日，目标是减少全血细胞计数(CBC)的使用。结果显示，在SmartAlert显示后的52小时内，CBC结果数量显著减少(1.54 vs 1.82, p<0.01)，且对次要安全结果没有不良影响，代表重复测试相对减少了15%。实施经验教训包括：在临床背景下解释概率模型预测、与利益相关者合作定义可接受的模型行为、在临床环境中部署复杂模型的治理流程、用户界面设计考虑、与临床运营优先事项的对齐，以及从终端用户获取定性反馈的价值。总之，由深思熟虑的实施和治理过程支持的机器学习驱动的临床决策支持系统，可以为住院实验室测试提供精准指导，从而安全地减少不必要的重复测试。


### 论文摘要

Repetitive laboratory testing unlikely to yield clinically useful information is a common practice that burdens patients and increases healthcare costs. Education and feedback interventions have limited success, while general test ordering restrictions and electronic alerts impede appropriate clinical care. We introduce and evaluate SmartAlert, a machine learning (ML)-driven clinical decision support (CDS) system integrated into the electronic health record that predicts stable laboratory results to reduce unnecessary repeat testing. This case study describes the implementation process, challenges, and lessons learned from deploying SmartAlert targeting complete blood count (CBC) utilization in a randomized controlled pilot across 9270 admissions in eight acute care units across two hospitals between August 15, 2024, and March 15, 2025. Results show significant decrease in number of CBC results within 52 hours of SmartAlert display (1.54 vs 1.82, p <0.01) without adverse effect on secondary safety outcomes, representing a 15% relative reduction in repetitive testing. Implementation lessons learned include interpretation of probabilistic model predictions in clinical contexts, stakeholder engagement to define acceptable model behavior, governance processes for deploying a complex model in a clinical environment, user interface design considerations, alignment with clinical operational priorities, and the value of qualitative feedback from end users. In conclusion, a machine learning-driven CDS system backed by a deliberate implementation and governance process can provide precision guidance on inpatient laboratory testing to safely reduce unnecessary repetitive testing.

---

## 134. tritonBLAS: Triton-based Analytical Approach for GEMM Kernel Parameter Selection

**论文链接:** [http://arxiv.org/abs/2512.04226v1](http://arxiv.org/abs/2512.04226v1)

**作者:** Ryan Swann, Muhammad Osama, Xiaohu Guo, Bryant Nelson, Lixun Zhang, Alex Brown, Yen Ong, Ali Yazdani, Sean Siddens, Ganesh Dasika, Alex Underwood

**发布时间:** 2025-12-03

### GPT解析

### 总结

tritonBLAS是一个快速且确定性分析模型，利用架构参数如缓存层次结构和相对代码和数据放置来生成高性能的GPU GEMM内核。

### 背景

GPU计算在高性能计算和机器学习工作负载中广泛应用，而GEMM是其中的关键操作。现有解决方案通常依赖运行时自动调优，这需要时间且可能不够确定性。

### 目的

开发一个能够预测接近最优配置的模型，无需运行时自动调优，从而提高GPU GEMM性能并减少调优时间。

### 方法

tritonBLAS通过明确建模架构拓扑、矩阵形状和算法分块行为之间的关系来预测接近最优的配置。基于此模型，开发并实现了一个完全在Triton内的轻量级GEMM框架。

### 主要发现

tritonBLAS在现代GPU上针对各种GEMM问题尺寸进行评估时，实现了超过自动调优解决方案95%的性能，同时将自动调优时间减少到零。

### 结论

tritonBLAS可以作为生产HPC和ML工作负载中经验调优的实际替代方案。

### 翻译

我们提出了tritonBLAS，这是一个快速且确定性分析模型，它使用缓存层次结构等架构参数以及相对代码和数据放置来生成高性能的GPU GEMM内核。tritonBLAS明确建模了架构拓扑、矩阵形状和算法分块行为之间的关系，无需运行时自动调优即可预测接近最优的配置。基于此模型，我们在Triton中开发并实现了一个轻量级GEMM框架。我们在现代GPU上针对各种GEMM问题尺寸评估了tritonBLAS的性能。tritonBLAS实现了超过自动调优解决方案95%的性能，同时将自动调优时间减少到零。这使得tritonBLAS成为生产HPC和ML工作负载中经验调优的实际替代方案。


### 论文摘要

We present tritonBLAS, a fast and deterministic analytical model that uses architectural parameters like the cache hierarchy, and relative code and data placement to generate performant GPU GEMM kernels. tritonBLAS explicitly models the relationship between architectural topology, matrix shapes, and algorithmic blocking behavior to predict near-optimal configurations without runtime autotuning. Based on this model, we developed and implemented a lightweight GEMM framework entirely within Triton. We evaluate the performance of tritonBLAS across a diverse set of GEMM problem sizes on modern GPUs. tritonBLAS achieves over 95% of the performance of autotuning solutions, while reducing autotuning time to zero. This makes tritonBLAS a practical drop-in replacement for empirical tuning in production HPC and ML workloads.

---

## 135. 论文ID: 2512.04212v1

**错误:** 无法读取论文信息 - [Errno 2] No such file or directory: 'storage/result/2512.04212v1.json'

---

## 136. 2D Helical Twist Controls Tricritical Point in an Interacting Majorana Chain

**论文链接:** [http://arxiv.org/abs/2512.04180v1](http://arxiv.org/abs/2512.04180v1)

**作者:** Hekai Zhao, Philip Phillips

**发布时间:** 2025-12-03

### GPT解析

### 总结

研究具有有限范围配对相互作用的Majorana费米子链，发现存在一个三临界点分隔Ising临界相与超对称间隙相

### 背景

相互作用的Majorana费米子链在耦合强度g下表现出奇偶不对称性，取决于相互作用范围δ

### 目的

分析相互作用Majorana费米子链的相变行为，理解1D系统与2D系统之间的关系

### 方法

引入旋转几何扭曲，将1D的δ范围链映射为2D的δ/2宽度模型，并构建具有螺旋边界条件的新2D模型

### 主要发现

1) 存在三临界点分隔不同相；2) 偶数δ情况下存在竞争序难以数值处理；3) 奇数δ情况下在g=-0.5处有完全可解点且纠缠熵为零；4) 1D系统的相变可视为2D系统的有限尺寸相变

### 结论

1D系统中g_c-δ行为由δ→∞极限下的2D三临界普适类控制，可通过有限尺寸标度理论预测

### 翻译

我们分析了一系列具有有限范围配对相互作用的相互作用的Majorana费米子链，耦合强度为g，这些链都表现出一个三临界点，该点将Ising临界相与超对称间隙相分开。我们首先注意到，相互作用模型表现出奇偶不对称性，这取决于相互作用的范围δ。偶数情况下表现出竞争序，从而使其难以数值处理，而奇数情况下在g=-0.5处表现出一个完全可解的点，此时纠缠熵为零。通过引入旋转几何扭曲，我们将1D的δ范围链映射为一系列2D的δ/2宽度模型。我们的新2D模型具有独特的螺旋边界条件，由一个链的末端连接到另一个链的起点构成。我们提出，1D系统中的相变可以理解为2D系统中的有限系统尺寸相变。也就是说，g_c-δ行为受δ→∞极限下的2D三临界普适类控制，并由有限尺寸标度理论预测。


### 论文摘要

We analyze a series of interacting Majorana Fermion chains with finite range pair interactions with coupling strength $g$ that all exhibit a tri-critical point that separates an Ising critical phase from a supersymmetric gapped phase. We first notice that the interacting models exhibit an even-odd asymmetry depending on the number of sites, $δ$, over which the interaction ranges. The even case exhibits competing order, thereby making it numerically untractable while the odd case exhibits an exactly solvable point at $g=-0.5$ where the entanglement entropy vanishes. By introducing a swirling geometrical twist, we map our 1D $δ$-range chains to a series of 2D $δ/2$-width models. Our new 2D models possess a unique helical boundary condition, constructed from 1D chains with the end of one connected to the start of another. We propose that the phase transition in the 1D system can be understood as a finite-system size transition in 2D. That is, the $g_c-δ$ behavior is controlled by a 2D tri-critical universality class at $δ\to\infty$ limit and is predicted by finite-size scaling theory.

---

## 137. Highly Anisotropic Charge Dynamics and Spectral Weight Redistribution in the Trilayer Nickelate La$_{4}$Ni$_{3}$O$_{10}$

**论文链接:** [http://arxiv.org/abs/2512.03806v2](http://arxiv.org/abs/2512.03806v2)

**作者:** Zhe Liu, Jie Li, Deyuan Hu, Bingke Ji, Haoran Zhang, Jiahao Hao, Yaomin Dai, Qing Li, Mengjun Ou, Bing Xu, Yi Lu, Meng Wang, Hai-Hu Wen

**发布时间:** 2025-12-03

**备注:** 8 pages, 3 figures. Comments are welcome and appreciated

### GPT解析

### 总结

本研究使用光学光谱学研究了La₄Ni₃O₁₀材料的ab面和c轴电荷动力学，发现其表现出强烈的各向异性，ab面呈现金属行为而c轴呈现半导体行为。

### 背景

La₄Ni₃O₁₀是一种具有特殊电荷动力学的材料，研究其不同方向的电学性质对于理解其电子结构具有重要意义。

### 目的

探究La₄Ni₃O₁₀材料在ab面和c轴方向上的电荷动力学特性，以及其电子关联效应。

### 方法

采用光学光谱学方法测量了La₄Ni₃O₁₀材料的ab面和c轴光学电导率，并分析了其谱权重转移和能带间跃迁特性。

### 主要发现

1) ab面光学电导率表现出金属响应，而c轴表现出半导体行为；2) 300K时电阻率各向异性ρ_c/ρ_ab约为366，介于铁基超导体和高温铜氧化物超导体之间；3) 能带间响应具有高度各向异性和轨道选择性；4) 能带间跃迁峰能量低于理论预测，表明存在显著电子关联；5) pristine相中库仑关联对电荷动力学有显著影响，密度波态中能隙打开且涉及Ni-d_{z²}轨道。

### 结论

La₄Ni₃O₁₀材料表现出强烈的电荷动力学各向异性，其电子结构受强关联效应影响，这种材料可能具有独特的电子特性，值得进一步研究。

### 翻译

我们使用光学光谱学研究了La₄Ni₃O₁₀的ab面和c轴电荷动力学。虽然在ab面光学电导率中观察到明显的德鲁德轮廓（即金属响应），但c轴光学光谱表现出半导体行为。光学电导率的零频率外推给出了La₄Ni₃O₁₀在300K时的电阻率各向异性约为366，这个值比铁基超导体大，但与高温铜氧化物超导体相当。能带间响应也具有高度各向异性，显示出对ab面偏振光和c轴偏振光明显的轨道选择性。中的能带间跃迁峰比密度泛函理论预测的能量低，表明存在显著的电子关联。通过研究谱权重转移，我们发现p pristine相中库仑关联对电荷动力学有显著影响，而在密度波态中，能隙打开且涉及Ni-d_{z²}轨道。


### 论文摘要

We study the $ab$-plane and $c$-axis charge dynamics of La$_{4}$Ni$_{3}$O$_{10}$ using optical spectroscopy. While a pronounced Drude profile, i.e. metallic response, is observed in the $ab$-plane optical conductivity $σ_{1}^{ab}(ω)$, the $c$-axis optical spectra $σ_{1}^{c}(ω)$ exhibit semiconducting behavior. The zero-frequency extrapolation of the optical conductivity $σ_{1}(ω\rightarrow 0) \equiv 1/ρ_{\text{dc}}$ gives a resistivity anisotropy of $ρ_{c}/ρ_{ab} \simeq 366$ at 300~K for La$_{4}$Ni$_{3}$O$_{10}$, which is much larger than the values in iron-based superconductors but comparable to those in high-$T_{c}$ cuprates. The interband response is also highly anisotropic, showing salient orbital selectivity for light polarized in the $ab$ plane and along the $c$ axis. The interband-transition peaks in both $σ_{1}^{ab}(ω)$ and $σ_{1}^{c}(ω)$ are located at lower energies compared to density-functional-theory predictions, signifying considerable electronic correlations. By investigating the spectral weight transfer, we find that in the pristine phase, Coulomb correlations have a marked impact on the charge dynamics of \LNO, whereas in the density-wave state, a gap opens with the Ni-$d_{z^{2}}$ orbital being involved.

---

## 138. Vision and Causal Learning Based Channel Estimation for THz Communications

**论文链接:** [http://arxiv.org/abs/2512.04380v1](http://arxiv.org/abs/2512.04380v1)

**作者:** Kitae Kim, Yan Kyaw Tun, Md. Shirajum Munir, Chirsto Kurisummoottil Thomas, Walid Saad, Choong Seon Hong

**发布时间:** 2025-12-04

**备注:** Submitted to IEEE Transactions on Mobile Computing on Mar. 20, 2025 (18 pages, 9 figures)

### GPT解析

### 总结

本文提出了一种创新的基于视觉的信道估计技术，将因果推理整合到城市太赫兹通信系统中，通过结合计算机视觉算法与变分因果动力学分析城市环境实时图像，显著提高了信道估计的准确性，特别是在非视距场景中表现优异。

### 背景

太赫兹通信与大规模多输入多输出系统在6G中可提供高速率低延迟通信，但太赫兹频率下的精确信道估计面临高传播损耗、环境障碍物敏感性和强大气吸收等挑战，这些挑战在城市环境中尤为明显，传统方法在复杂非视距场景中难以提供可靠结果。

### 目的

引入一种创新的基于视觉的信道估计技术，将因果推理整合到城市太赫兹通信系统中，以提高信道估计的准确性，特别是在复杂城市环境中的非视距场景。

### 方法

提出的方法结合计算机视觉算法与变分因果动力学，分析城市环境的实时图像，深入了解影响太赫兹信号传播的物理因素，捕捉物理物体与传输信号之间复杂、动态的相互作用，从而预测信道状态。

### 主要发现

该模型将信道预测准确性提高到传统方法的两倍，提高了估计精度并展示了卓越的泛化性能，即使在未见过的城市环境中也能提供可靠预测；在非视距条件下特别有效，能考虑间接信号路径如反射和衍射；模拟结果证实该方法在准确性和鲁棒性上超越了传统人工智能估计技术，在各种动态城市场景中显示出显著改进。

### 结论

该基于视觉和因果推理的方法为太赫兹通信在城市环境中的信道估计提供了创新解决方案，克服了传统方法的局限性，特别是在复杂非视距场景中表现优异。

### 翻译

太赫兹通信与大规模多输入多输出系统在6G中的应用可以潜在地提供高速率和低延迟通信。然而，在太赫兹频率下进行精确的信道估计由于高传播损耗、对环境障碍物的敏感性以及强大气吸收等因素而面临重大挑战。这些挑战在城市环境中尤为明显，传统信道估计方法通常无法提供可靠的结果，特别是在复杂的非视距场景中。本文引入了一种创新的基于视觉的信道估计技术，将因果推理整合到城市太赫兹通信系统中。所提出的方法结合了计算机视觉算法与变分因果动力学，分析城市环境的实时图像，深入了解影响太赫兹信号传播的物理因素。通过捕捉物理物体（如建筑物、树木和车辆）与传输信号之间复杂、动态的相互作用，该模型可以将信道预测的准确性提高到传统方法的两倍。该模型提高了估计精度并展示了卓越的泛化性能，因此即使在以前未见过的城市环境中也能提供可靠的预测。所提出方法的有效性在非视距条件下尤为明显，它显著优于传统方法，能够考虑间接信号路径，如反射和衍射。模拟结果证实，所提出的基于视觉的方法在准确性和鲁棒性上超越了传统的人工智能估计技术，在各种动态城市场景中显示出显著的改进。


### 论文摘要

The use of terahertz (THz) communications with massive multiple input multiple output (MIMO) systems in 6G can potentially provide high data rates and low latency communications. However, accurate channel estimation in THz frequencies presents significant challenges due to factors such as high propagation losses, sensitivity to environmental obstructions, and strong atmospheric absorption. These challenges are par- ticularly pronounced in urban environments, where traditional channel estimation methods often fail to deliver reliable results, particularly in complex non-line-of-sight (NLoS) scenarios. This paper introduces a novel vision-based channel estimation tech- nique that integrates causal reasoning into urban THz communi- cation systems. The proposed method combines computer vision algorithms with variational causal dynamics (VCD) to analyze real-time images of the urban environment, allowing for a deeper understanding of the physical factors that influence THz signal propagation. By capturing the complex, dynamic interactions between physical objects (such as buildings, trees, and vehicles) and the transmitted signals, the model can predict the channel with up to twice the accuracy of conventional methods. This model improves estimation accuracy and demonstrates supe- rior generalization performance. Hence, it can provide reliable predictions even in previously unseen urban environments. The effectiveness of the proposed method is particularly evident in NLoS conditions, where it significantly outperforms traditional methods such as by accounting for indirect signal paths, such as reflections and diffractions. Simulation results confirm that the proposed vision-based approach surpasses conventional artificial intelligence (AI)-based estimation techniques in accuracy and robustness, showing a substantial improvement across various dynamic urban scenarios.

---

