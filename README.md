# 今日论文推荐 - 2025-10-30

共 39 篇论文

---

## 1. 4-Doodle: Text to 3D Sketches that Move!

**论文链接:** [http://arxiv.org/abs/2510.25319v1](http://arxiv.org/abs/2510.25319v1)

**作者:** Hao Chen, Jiaqi Wang, Yonggang Qi, Ke Li, Kaiyue Pang, Yi-Zhe Song

**发布时间:** 2025-10-29

### GPT解析

### 总结

本文提出了4-Doodle框架，一个从文本描述生成动态3D草图动画的无需训练方法，通过双空间蒸馏方案解决文本到3D草图动画任务中的关键挑战。

### 背景

现有3D内容生成方法主要关注逼真内容，忽视了稀疏、风格化的3D矢量草图这一轻量级媒介。该任务面临三大挑战：缺乏配对数据集、结构抽象难以建模、动画需要时间一致性和多视角一致性。

### 目的

开发一个无需训练的框架，从文本生成动态、时间一致且多视角一致的3D矢量草图动画，实现结构稳定的动画效果，包括翻转、旋转和关节运动等。

### 方法

4-Doodle采用双空间蒸馏方案：一个空间使用可微贝塞尔曲线捕获多视角一致的几何形状；另一个通过时间感知先验编码运动动态。采用多视图优化确保结构对齐，并引入结构感知的运动模块将保持形状的轨迹与变形感知的变化分开。

### 主要发现

实验表明，4-Doodle能生成时间逼真且结构稳定的3D草图动画，在保真度和可控性方面优于现有基线。多视图优化确保结构对齐，结构感知运动模块实现富有表现力的动画效果。

### 结论

4-Doodle为文本到动态3D草图动画提供了有效解决方案，使4D内容创作更加直观和易于访问，填补了相关研究空白，为视觉交流和原型设计提供新可能。

### 翻译

我们提出了一个新任务：文本到3D草图动画，旨在让自由形式的草图在动态3D空间中'活起来'。与专注于生成逼真内容的前期工作不同，我们目标是稀疏的、风格化的和视角一致的3D矢量草图，这是一种轻量级且可解释的媒介，非常适合视觉交流和原型设计。然而，这项任务非常具有挑战性：(i) 没有文本和3D（或4D）草图的配对数据集；(ii) 草图需要结构抽象，难以用传统的3D表示建模；(iii) 为这样的草图添加动画需要时间一致性和多视角一致性，而当前的处理流程无法解决这个问题。因此，我们提出了4-Doodle，这是第一个从文本生成动态3D草图的无需训练的框架。它通过双空间蒸馏方案利用预训练的图像和视频扩散模型：一个空间使用可微的贝塞尔曲线捕获多视角一致的几何形状，而另一个通过时间感知先验编码运动动态。与之前的工作不同，后者每步从单一视图进行优化，而我们的多视图优化确保了结构对齐并避免了视图模糊性，这对稀疏草图至关重要。此外，我们引入了一个结构感知的运动模块，该模块将保持形状的轨迹与变形感知的变化分开，实现翻转、旋转和关节运动等富有表现力的动作。大量实验表明，我们的方法能够生成时间上逼真且结构稳定的3D草图动画，在保真度和可控性方面都优于现有的基线。我们希望这项工作能推动更加直观和易于访问的4D内容创作发展。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决从文本描述生成动态3D矢量草图的问题。这个问题很重要，因为随着空间计算平台（如Apple Vision Pro和Meta Quest）的兴起，创建和动画3D草图成为沉浸式内容创建的基础。传统的3D内容生成方法主要关注照片真实感内容，而3D草图作为一种轻量级、可解释的媒介，非常适合设计原型、视觉叙事和空间用户界面等应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者将问题分解为两个互补阶段：构建一致的3D草图结构和添加动画。他们借鉴了现有工作，如DreamFusion中的Score Distillation Sampling技术、3Doodle中的贝塞尔曲线表示、LiveSketch中的运动先验蒸馏，以及MVDream等多视角扩散模型。作者的核心创新在于设计了一个双空间知识蒸馏框架，利用预训练的图像和视频扩散模型，通过多视角优化和基于贝塞尔曲线的表示来生成动态3D草图，避免了需要成对的文本-4D草图训练数据。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是双空间知识蒸馏框架，利用预训练的图像和视频扩散模型将3D结构和运动动力学的知识转移过来，无需特定训练数据。整体流程分为两个阶段：第一阶段是多视角3D草图生成，通过随机初始化贝塞尔曲线，从多个视角（前、后、左、右）渲染并使用Score Distillation Sampling优化曲线参数；第二阶段是运动场学习，将3D场景投影到前视图和侧视图，利用视频扩散模型预测位移序列，然后重建3D位移向量，最后添加时间平滑确保动画流畅。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1）首个文本到3D草图动画框架；2）基于可微分贝塞尔曲线的双空间架构；3）多视角优化策略减少歧义并确保结构对齐；4）结构感知的运动生成模块；5）投影-重建策略使视频扩散模型能在3D空间中合成运动。相比之前工作，4-Doodle不仅处理静态3D草图（如SketchDream、Sketch2NeRF），还能生成动态内容；不需要手动绘制运动轨迹（如Sketch2Anim）；专注于结构抽象和草图感知（如Animate3D、3DTopia）；支持动态草图抽象和跨视图时间一致性（如CLAY）。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '4-Doodle首次实现了从文本描述直接生成动态、空间一致且富有表现力的3D矢量草图动画，通过双空间知识蒸馏框架和基于贝塞尔曲线的可微表示，解决了草图动画中的结构一致性和时间连贯性挑战。'}


### 论文摘要

We present a novel task: text-to-3D sketch animation, which aims to bring freeform sketches to life in dynamic 3D space. Unlike prior works focused on photorealistic content generation, we target sparse, stylized, and view-consistent 3D vector sketches, a lightweight and interpretable medium well-suited for visual communication and prototyping. However, this task is very challenging: (i) no paired dataset exists for text and 3D (or 4D) sketches; (ii) sketches require structural abstraction that is difficult to model with conventional 3D representations like NeRFs or point clouds; and (iii) animating such sketches demands temporal coherence and multi-view consistency, which current pipelines do not address. Therefore, we propose 4-Doodle, the first training-free framework for generating dynamic 3D sketches from text. It leverages pretrained image and video diffusion models through a dual-space distillation scheme: one space captures multi-view-consistent geometry using differentiable B\'ezier curves, while the other encodes motion dynamics via temporally-aware priors. Unlike prior work (e.g., DreamFusion), which optimizes from a single view per step, our multi-view optimization ensures structural alignment and avoids view ambiguity, critical for sparse sketches. Furthermore, we introduce a structure-aware motion module that separates shape-preserving trajectories from deformation-aware changes, enabling expressive motion such as flipping, rotation, and articulated movement. Extensive experiments show that our method produces temporally realistic and structurally stable 3D sketch animations, outperforming existing baselines in both fidelity and controllability. We hope this work serves as a step toward more intuitive and accessible 4D content creation.

---

## 2. SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation

**论文链接:** [http://arxiv.org/abs/2510.25268v1](http://arxiv.org/abs/2510.25268v1)

**作者:** Wang zhi, Yuyan Liu, Liu Liu, Li Zhang, Ruixuan Lu, Dan Guo

**发布时间:** 2025-10-29

### GPT解析

### 总结

本文提出了SynHLMA框架，用于生成关节物体的手部语言操作序列，实现了HAOI生成、预测和插值三种任务，在HAOI-lang数据集上展示了优越性能，并可用于机器人抓取应用。

### 背景

生成手部抓取动作是具身AI和VR/AR应用中的广泛研究课题。当涉及关节物体交互时，手部抓取合成需要同时考虑物体功能性和物体变形过程中的长期操作序列。

### 目的

提出一种新的HAOI序列生成框架SynHLMA，用于合成关节物体的手部语言操作。

### 方法

给定关节物体的完整点云，使用离散的HAOI表示建模每个手部物体交互帧；结合自然语言嵌入，通过HAOI操作语言模型训练这些表示，在共享表示空间中对齐抓取过程与语言描述；采用关节感知损失确保手部抓取遵循关节物体的动态变化。

### 主要发现

SynHLMA实现了关节物体的三种典型手部操作任务：HAOI生成、HAOI预测和HAOI插值；在HAOI-lang数据集上的评估结果显示，与最先进方法相比具有优越的手部抓取序列生成性能；通过使用SynHLMA提供的操作序列，机器人可以实现灵巧抓取的模仿学习。

### 结论

SynHLMA框架在关节物体的手部抓取序列生成方面表现优越，代码和数据集将公开可用，为具身AI和VR/AR应用提供了新的可能性。

### 翻译

通过语言指令生成手部抓取是一个广泛研究的课题，受益于具身AI和VR/AR应用。当转化为关节物体交互时，手部抓取合成不仅需要物体功能性，还需要考虑物体变形过程中的长期操作序列。本文提出了一个新的HAOI序列生成框架SynHLMA，用于合成关节物体的手部语言操作。给定关节物体的完整点云，我们使用离散的HAOI表示来建模每个手部物体交互帧。结合自然语言嵌入，通过HAOI操作语言模型训练这些表示，在共享表示空间中对齐抓取过程与语言描述。采用关节感知损失来确保手部抓取遵循关节物体的动态变化。通过这种方式，我们的SynHLMA实现了关节物体的三种典型手部操作任务：HAOI生成、HAOI预测和HAOI插值。我们在构建的HAOI-lang数据集上评估SynHLMA，实验结果展示了与最先进方法相比的优越手部抓取序列生成性能。我们还展示了机器抓取应用，通过使用SynHLMA提供的操作序列，使模仿学习能够执行灵巧抓取。我们的代码和数据集将公开可用。


### 论文摘要

Generating hand grasps with language instructions is a widely studied topic that benefits from embodied AI and VR/AR applications. While transferring into hand articulatied object interaction (HAOI), the hand grasps synthesis requires not only object functionality but also long-term manipulation sequence along the object deformation. This paper proposes a novel HAOI sequence generation framework SynHLMA, to synthesize hand language manipulation for articulated objects. Given a complete point cloud of an articulated object, we utilize a discrete HAOI representation to model each hand object interaction frame. Along with the natural language embeddings, the representations are trained by an HAOI manipulation language model to align the grasping process with its language description in a shared representation space. A joint-aware loss is employed to ensure hand grasps follow the dynamic variations of articulated object joints. In this way, our SynHLMA achieves three typical hand manipulation tasks for articulated objects of HAOI generation, HAOI prediction and HAOI interpolation. We evaluate SynHLMA on our built HAOI-lang dataset and experimental results demonstrate the superior hand grasp sequence generation performance comparing with state-of-the-art. We also show a robotics grasp application that enables dexterous grasps execution from imitation learning using the manipulation sequence provided by our SynHLMA. Our codes and datasets will be made publicly available.

---

## 3. U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching

**论文链接:** [http://arxiv.org/abs/2510.25210v1](http://arxiv.org/abs/2510.25210v1)

**作者:** Junsheng Zhou, Xingyu Shi, Haichuan Song, Yi Fang, Yu-Shen Liu, Zhizhong Han

**发布时间:** 2025-10-29

**备注:** Accepted by NeurIPS 2025. Project page:  https://gloriasze.github.io/U-CAN/

### GPT解析

### 总结

本文提出了一种名为U-CAN的无监督点云去噪框架，采用一致性感知的Noise2Noise匹配方法，通过神经网络推断多步去噪路径，并引入几何一致性约束，无需大量人工标注数据即可达到与监督方法相当的去噪效果。

### 背景

扫描传感器捕获的点云数据通常受到噪声干扰，这对下游任务（如表面重建和形状理解）有负面影响。先前的工作大多使用含噪-清洁点云对训练神经网络来学习去噪先验，这需要大量的人工努力。

### 目的

开发一种无需人工标注数据的无监督点云去噪方法，以减少对大量含噪-清洁点云对的依赖。

### 方法

U-CAN框架利用神经网络推断每个点或场景的多步去噪路径，通过噪声到噪声匹配方案实现。通过一种新的损失函数，使模型能够对多个含噪点云观测进行统计推理。引入一种去噪后几何一致性约束，以学习一致性感知的去噪模式。该约束不仅限于3D领域，还可以贡献于2D图像去噪领域。

### 主要发现

在点云去噪、上采样和图像去噪的广泛基准测试中，U-CAN比最先进的无监督方法有显著改进，并且产生的结果与监督方法相当。

### 结论

U-CAN是一种有效的无监督点云去噪方法，不需要大量人工标注的数据，同时能够达到与监督方法相当的性能，为点云去噪领域提供了一种新的无监督解决方案。

### 翻译

扫描传感器捕获的点云数据常常受到噪声干扰，这对下游任务（例如表面重建和形状理解）有严重的负面影响。先前的工作主要集中在使用含噪-清洁点云对训练神经网络来学习去噪先验，这需要大量的人工努力。在本工作中，我们引入了U-CAN，一种基于一致性感知的Noise2Noise匹配的无监督点云去噪框架。具体来说，我们利用神经网络推断形状或场景中每个点的多步去噪路径，采用噪声到噪声匹配方案。我们通过一种新的损失函数实现这一点，该损失函数能够在多个含噪点云观测上进行统计推理。我们进一步引入了一种对去噪后几何一致性的新约束，以学习一致性感知的去噪模式。我们证明所提出的约束是一个通用术语，不仅限于3D领域，还可以贡献于2D图像去噪领域。在点云去噪、上采样和图像去噪的广泛基准测试中，我们的评估显示比最先进的无监督方法有显著改进，其中U-CAN也产生了与监督方法相当的结果。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': "这篇论文主要解决无监督点云去噪问题。点云数据（如激光雷达扫描获取的三维点数据）通常包含噪声，影响下游任务如表面重建和形状理解。现有方法需要成对的'带噪-干净'点云数据训练，需要大量人工标注，成本高。在现实中，自动驾驶汽车、手机等设备每天都在产生大量带噪点云，而干净数据获取困难。解决这个问题能减少对人工标注的依赖，使去噪技术更容易应用于实际场景，提升自动驾驶、增强现实和机器人等领域的性能。", '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到尽管干净点云有限，但带噪点云数据每天都在快速增长。借鉴了2D图像的Noise2Noise方法，但发现点云无序不规则，没有像素间的一对一对应关系，不能直接应用。现有无监督方法如TotalDenoising只使用全局约束，难以保持局部几何结构。因此，作者设计多步去噪框架，通过神经网络为每个点推断去噪路径；提出点对点噪声到噪声匹配，使用地球移动距离建立点间对应关系；引入一致性感知约束确保不同噪声观测的去噪预测保持一致。借鉴了PointNet、PointNet++等点云处理架构和TotalDenoising的全局约束思想。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是噪声到噪声匹配和一致性感知约束。通过学习从一个带噪点云到另一个带噪点云的映射，利用统计推理从多个带噪观测中揭示干净结构；同时确保不同噪声观测的去噪预测间保持几何一致性，解决缺乏真实表面位置信息导致的收敛不稳定问题。整体流程：输入带噪点云→多步去噪框架（每步包含特征提取和路径预测）→噪声到噪声匹配（使用地球移动距离建立点间对应）→一致性感知约束（最小化不同去噪预测间的几何差异）→输出去噪后点云。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1) U-CAN无监督框架，利用噪声到噪声匹配和一致性感知约束；2) 点对点噪声匹配方案，使用地球移动距离建立点云对应关系；3) 去噪几何一致性约束，确保不同噪声观测的去噪预测一致；4) 证明该约束不仅限于3D领域，也可用于2D图像去噪；5) 可用于无监督点云上采样任务。不同之处：相比监督方法，无需干净数据；相比TotalDenoising等，不仅用全局约束，还引入局部约束；相比直接应用Noise2Noise，解决了点云对应关系缺失问题；相比其他无监督方法，引入一致性约束解决收敛不稳定问题。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'U-CAN通过创新的噪声到噪声匹配和一致性感知约束，实现了仅使用带噪点云数据就能达到与监督方法相当的去噪效果，无需人工标注的干净数据。'}


### 论文摘要

Point clouds captured by scanning sensors are often perturbed by noise, which have a highly negative impact on downstream tasks (e.g. surface reconstruction and shape understanding). Previous works mostly focus on training neural networks with noisy-clean point cloud pairs for learning denoising priors, which requires extensively manual efforts. In this work, we introduce U-CAN, an Unsupervised framework for point cloud denoising with Consistency-Aware Noise2Noise matching. Specifically, we leverage a neural network to infer a multi-step denoising path for each point of a shape or scene with a noise to noise matching scheme. We achieve this by a novel loss which enables statistical reasoning on multiple noisy point cloud observations. We further introduce a novel constraint on the denoised geometry consistency for learning consistency-aware denoising patterns. We justify that the proposed constraint is a general term which is not limited to 3D domain and can also contribute to the area of 2D image denoising. Our evaluations under the widely used benchmarks in point cloud denoising, upsampling and image denoising show significant improvement over the state-of-the-art unsupervised methods, where U-CAN also produces comparable results with the supervised methods.

---

## 4. $D^2GS$: Dense Depth Regularization for LiDAR-free Urban Scene Reconstruction

**论文链接:** [http://arxiv.org/abs/2510.25173v1](http://arxiv.org/abs/2510.25173v1)

**作者:** Kejing Xia, Jidong Jia, Ke Jin, Yucai Bai, Li Sun, Dacheng Tao, Youjian Zhang

**发布时间:** 2025-10-29

### GPT解析

### 总结

D²GS是一种无LiDAR的城市场景重建框架，通过多视图深度预测和优化技术，实现了比使用LiDAR的方法更准确的几何重建。

### 背景

高斯溅射(GS)在自动驾驶城市场景重建中显示出潜力，但现有方法依赖多模态传感器(如LiDAR和图像)，而LiDAR数据获取存在时空校准困难和空间不对齐问题。

### 目的

开发一种无需LiDAR数据的城市场景重建方法，避免获取准确LiDAR深度的困难，同时保持或提高重建质量。

### 方法

1) 通过反向投影多视图度量深度预测初始化密集点云，并用渐进修剪策略优化；2) 利用深度基础模型的扩散先验增强高斯渲染的深度图，提供更强几何约束；3) 约束道路区域内高斯的形状和法线属性提高地面几何准确性。

### 主要发现

在Waymo数据集上的实验表明，D²GS方法始终优于最先进方法，即使与使用真实LiDAR数据的方法相比，也能产生更准确的几何。

### 结论

D²GS框架成功实现了无LiDAR的城市场景重建，获得了比LiDAR方法更密集、更准确的几何先验，证明了深度先验和优化策略的有效性。

### 翻译

最近，高斯溅射(GS)在自动驾驶领域的城市场景重建中显示出巨大潜力。然而，当前的城市场景重建方法通常依赖于多模态传感器作为输入，即LiDAR和图像。虽然LiDAR点云提供的几何先验可以大大减轻重建中的不适定性，但在实践中获取准确的LiDAR数据仍然具有挑战性：i)需要LiDAR和其他传感器之间精确的时空校准，因为它们可能不会同时捕获数据；ii)当LiDAR和相机安装在不同位置时，空间不对齐会导致重投影误差。为了避免获取准确LiDAR深度的困难，我们提出了D²GS，一个无LiDAR的城市场景重建框架。在这项工作中，我们获得了与LiDAR一样有效但更密集、更准确的几何先验。首先，我们通过反向投影多视图度量深度预测来初始化密集点云。然后通过渐进修剪策略优化该点云以提高全局一致性。其次，我们通过深度增强器联合优化高斯几何和预测的密集度量深度。具体来说，我们利用来自深度基础模型的扩散先验来增强由高斯渲染的深度图。反过来，增强的深度在高斯训练期间提供更强的几何约束。最后，我们通过约束道路区域内高斯的形状和法线属性来提高地面几何的准确性。在Waymo数据集上的大量实验表明，我们的方法始终优于最先进的方法，即使与使用真实LiDAR数据的方法相比，也能产生更准确的几何。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决自动驾驶领域城市场景重建中对LiDAR（激光雷达）数据的依赖问题。这个问题很重要，因为获取准确的LiDAR数据在实际应用中面临诸多挑战：需要专业设备和车辆进行数据收集、传感器间需要精确的时空校准、LiDAR与相机安装在不同位置会导致重投影误差，此外LiDAR数据成本高昂且难以扩展，限制了大规模应用。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过分析现有方法对LiDAR依赖的问题，探索了替代方案。他们借鉴了多视图深度估计网络、3D Gaussian Splatting框架、扩散先验模型（如Marigold）、场景图表示方法以及道路几何先验知识。在此基础上，设计了三个关键组件：利用渐进式剪枝策略管理密集点云、通过深度增强模块迭代优化深度和高斯表示、在场景图中引入专门的道路节点约束。这些设计既吸收了现有工作的优点，又针对LiDAR-free场景进行了创新改进。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过有效利用从图像中推导的几何先验，消除对LiDAR数据的依赖，创建一个仅使用相机输入的动态城市街道场景重建框架。整体流程分为：1)初始化阶段：使用多视图深度估计预测深度图，反投影得到点云，通过渐进式剪枝获得代表性点集；2)优化阶段：创建道路节点约束，实施联合优化策略，使用深度增强模块利用扩散先验细化深度；3)训练阶段：迭代更新高斯参数和深度估计，利用置信度图指导深度增强；4)评估阶段：在Waymo数据集上评估性能，与使用LiDAR的方法进行比较。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)LiDAR-free框架，消除对LiDAR数据的需求和校准误差；2)渐进式剪枝策略，有效管理密集点云；3)基于扩散的深度增强联合优化策略，提供密集度量深度监督；4)道路节点约束，利用地面平面先验提高道路重建精度。相比之前工作，不同之处在于：不需要LiDAR数据避免校准问题；不依赖单目深度估计避免尺度模糊；能处理动态场景而多视图深度估计不能；将生成深度先验直接集成到3DGS优化循环中；提供比LiDAR更密集、更准确的几何先验。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'D²GS提出了一种无需LiDAR的城市场景重建框架，通过渐进式剪枝、深度增强和道路节点约束，仅使用相机输入就能实现比使用LiDAR数据更准确的动态城市街道场景重建。'}


### 论文摘要

Recently, Gaussian Splatting (GS) has shown great potential for urban scene reconstruction in the field of autonomous driving. However, current urban scene reconstruction methods often depend on multimodal sensors as inputs, \textit{i.e.} LiDAR and images. Though the geometry prior provided by LiDAR point clouds can largely mitigate ill-posedness in reconstruction, acquiring such accurate LiDAR data is still challenging in practice: i) precise spatiotemporal calibration between LiDAR and other sensors is required, as they may not capture data simultaneously; ii) reprojection errors arise from spatial misalignment when LiDAR and cameras are mounted at different locations. To avoid the difficulty of acquiring accurate LiDAR depth, we propose $D^2GS$, a LiDAR-free urban scene reconstruction framework. In this work, we obtain geometry priors that are as effective as LiDAR while being denser and more accurate. $\textbf{First}$, we initialize a dense point cloud by back-projecting multi-view metric depth predictions. This point cloud is then optimized by a Progressive Pruning strategy to improve the global consistency. $\textbf{Second}$, we jointly refine Gaussian geometry and predicted dense metric depth via a Depth Enhancer. Specifically, we leverage diffusion priors from a depth foundation model to enhance the depth maps rendered by Gaussians. In turn, the enhanced depths provide stronger geometric constraints during Gaussian training. $\textbf{Finally}$, we improve the accuracy of ground geometry by constraining the shape and normal attributes of Gaussians within road regions. Extensive experiments on the Waymo dataset demonstrate that our method consistently outperforms state-of-the-art methods, producing more accurate geometry even when compared with those using ground-truth LiDAR data.

---

## 5. Point-level Uncertainty Evaluation of Mobile Laser Scanning Point Clouds

**论文链接:** [http://arxiv.org/abs/2510.24773v1](http://arxiv.org/abs/2510.24773v1)

**作者:** Ziyang Xu, Olaf Wysocki, Christoph Holst

**发布时间:** 2025-10-24

### GPT解析

### 总结

本研究提出了一种基于机器学习的框架，用于评估移动激光扫描点云的点级别不确定性，无需依赖高精度参考数据。

### 背景

移动激光扫描点云中的不确定性可靠量化对3D制图、建模和变化分析等下游应用的准确性和可信度至关重要，而传统方法高度依赖难以获取的高精度参考数据。

### 目的

解决传统不确定性建模方法依赖高精度参考数据的问题，开发一种不依赖此类数据的点级别不确定性评估方法。

### 方法

提出基于机器学习的框架，学习局部几何特征与点级别误差之间的关系，使用随机森林和XGBoost两种集成学习模型，并在空间分区化的真实世界数据集上训练验证以避免数据泄露。

### 主要发现

两种模型能有效捕捉几何特征与不确定性间的非线性关系，平均ROC-AUC值超过0.87；描述高程变化、点密度和局部结构复杂性的几何特征在预测不确定性中起主导作用。

### 结论

该框架为不确定性评估提供了数据驱动的方法，为大规模点云的质量控制和误差分析提供了可扩展且适应性强的基础。

### 翻译

移动激光扫描点云中不确定性的可靠量化对于确保3D制图、建模和变化分析等下游应用的准确性和可信度至关重要。传统的不确定性建模方法高度依赖于高精度参考数据，而这些数据在大规模情况下通常成本高昂或难以获取。为解决这一问题，本研究提出了一种基于机器学习的点级别不确定性评估框架，学习局部几何特征与点级别误差之间的关系。该框架使用随机森林和XGBoost两种集成学习模型实现，在空间分区化的真实世界数据集上进行训练和验证以避免数据泄露。实验结果表明，两种模型都能有效捕捉几何特征与不确定性之间的非线性关系，平均ROC-AUC值超过0.87。分析进一步表明，描述高程变化、点密度和局部结构复杂性的几何特征在预测不确定性中起主导作用。所提出的框架为不确定性评估提供了数据驱动的方法，为未来大规模点云的质量控制和误差分析提供了可扩展且适应性强的基础。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '论文主要解决移动激光扫描点云的点级不确定性评估问题，特别是减少对高精度参考数据的依赖。这个问题很重要，因为可靠的不确定性量化对3D建模、变化分析等下游应用的准确性和可信度至关重要，而不充分评估点云质量会影响高精度应用如导航和变化分析，不仅降低可靠性，还会浪费时间和资源。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者首先分析了MLS系统中的不确定性来源，评估了现有方法（前向建模和后向建模）的局限性，特别是后向建模对参考数据的依赖和高成本问题。然后提出用机器学习替代方案，建立点云特征与不确定性关系。该方法借鉴了现有工作，如使用C2C距离作为不确定性度量、基于KNN的邻域定义策略，以及采用随机森林和XGBoost等集成学习方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是通过机器学习学习局部几何特征与点级误差之间的关系，将不确定性评估转化为二分类问题。整体流程包括：1)使用C2C距离定义不确定性度量；2)基于KNN提取每个点的局部几何特征；3)采用随机森林和XGBoost模型进行二分类训练；4)使用多种指标和空间分区5折交叉验证评估模型性能。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：提出基于机器学习的框架减少对参考数据依赖；将不确定性评估转化为二分类问题；使用局部几何特征预测不确定性；采用互补的集成学习方法验证；通过特征重要性分析提供误差源新见解。相比传统方法，本研究采用数据驱动方式，训练后不再需要参考数据，提供了更可扩展和适应性强的解决方案。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '本研究提出了一种基于机器学习的框架，能够通过学习点云的局部几何特征与点级误差之间的关系，实现对移动激光扫描点云的点级不确定性预测，减少了对高精度参考数据的依赖，为大规模点云质量评估提供了新的数据驱动方法。'}


### 论文摘要

Reliable quantification of uncertainty in Mobile Laser Scanning (MLS) point clouds is essential for ensuring the accuracy and credibility of downstream applications such as 3D mapping, modeling, and change analysis. Traditional backward uncertainty modeling heavily rely on high-precision reference data, which are often costly or infeasible to obtain at large scales. To address this issue, this study proposes a machine learning-based framework for point-level uncertainty evaluation that learns the relationship between local geometric features and point-level errors. The framework is implemented using two ensemble learning models, Random Forest (RF) and XGBoost, which are trained and validated on a spatially partitioned real-world dataset to avoid data leakage. Experimental results demonstrate that both models can effectively capture the nonlinear relationships between geometric characteristics and uncertainty, achieving mean ROC-AUC values above 0.87. The analysis further reveals that geometric features describing elevation variation, point density, and local structural complexity play a dominant role in predicting uncertainty. The proposed framework offers a data-driven perspective of uncertainty evaluation, providing a scalable and adaptable foundation for future quality control and error analysis of large-scale point clouds.

---

## 6. Controlling Contrastive Self-Supervised Learning with Knowledge-Driven Multiple Hypothesis: Application to Beat Tracking

**论文链接:** [http://arxiv.org/abs/2510.25560v1](http://arxiv.org/abs/2510.25560v1)

**作者:** Antonin Gagnere, Slim Essid, Geoffroy Peeters

**发布时间:** 2025-10-29

### GPT解析

### 总结

论文提出了一种对比自监督预训练方法，利用多种可能的正样本假设来解决数据模糊性问题，在音乐节拍跟踪任务上取得了优于现有方法的性能。

### 背景

数据中的模糊性和问题约束的多样性会导致机器学习任务产生多种同样合理的不同结果。例如在节拍和强拍跟踪中，不同听众可能采用各种节奏解释，这些解释都不一定是错误的。

### 目的

开发一种能够处理数据模糊性的方法，通过利用多种可能的正样本假设来提高机器学习模型的性能，特别是在音乐表示学习领域。

### 方法

提出一种对比自监督预训练方法，模型被训练为学习与不同假设兼容的表示，这些假设通过基于知识的评分函数选择，以保留最合理的假设。

### 主要发现

在有标签数据上进行微调时，该方法在标准基准测试上优于现有方法，证明了将领域知识与多假设选择相结合的有效性。

### 结论

将领域知识与多假设选择相结合在音乐表示学习中具有显著优势，能够有效处理数据中的模糊性问题并提高模型性能。

### 翻译

数据中的模糊性和问题约束可能导致机器学习任务产生多种同样合理的不同结果。例如在节拍和强拍跟踪中，不同听众可能采用各种节奏解释，这些解释都不一定是错误的。为此，我们提出了一种对比自监督预训练方法，利用数据中可能的正样本的多种假设。我们的模型被训练为学习与不同假设兼容的表示，这些假设通过基于知识的评分函数选择，以保留最合理的假设。在有标签数据上进行微调时，我们的模型在标准基准测试上优于现有方法，展示了将领域知识与多假设选择相结合在音乐表示学习中的优势。


### 论文摘要

Ambiguities in data and problem constraints can lead to diverse, equally plausible outcomes for a machine learning task. In beat and downbeat tracking, for instance, different listeners may adopt various rhythmic interpretations, none of which would necessarily be incorrect. To address this, we propose a contrastive self-supervised pre-training approach that leverages multiple hypotheses about possible positive samples in the data. Our model is trained to learn representations compatible with different such hypotheses, which are selected with a knowledge-based scoring function to retain the most plausible ones. When fine-tuned on labeled data, our model outperforms existing methods on standard benchmarks, showcasing the advantages of integrating domain knowledge with multi-hypothesis selection in music representation learning in particular.

---

## 7. IBNorm: Information-Bottleneck Inspired Normalization for Representation Learning

**论文链接:** [http://arxiv.org/abs/2510.25262v1](http://arxiv.org/abs/2510.25262v1)

**作者:** Xiandong Zou, Pan Zhou

**发布时间:** 2025-10-29

### GPT解析

### 总结

论文提出了一种基于信息瓶颈原理的新归一化方法IBNorm，通过有界压缩操作鼓励嵌入保留预测信息同时抑制无用变异性，在大规模语言模型和视觉模型上均优于传统归一化方法。

### 背景

归一化是深度学习的基础，但现有方法如BatchNorm、LayerNorm和RMSNorm都是方差中心的，通过强制零均值和单位方差稳定训练，但没有控制表示如何捕获任务相关信息。

### 目的

提出一种新的归一化方法，能够鼓励表示保留预测信息同时抑制无用变异性，从而产生更具信息量的表示。

### 方法

提出IB-Inspired Normalization (IBNorm)，一种基于信息瓶颈原理的简单而强大的方法系列，引入有界压缩操作来优化信息表示。

### 主要发现

理论上证明IBNorm比方差中心方法获得更高的IB值和更紧的泛化边界；实验上在大型语言模型和视觉模型上一致优于传统归一化方法，互信息分析证实了其优越的信息瓶颈行为。

### 结论

IBNorm能够产生更具信息量的表示，同时保持标准归一化的稳定性和兼容性，是一种优于传统归一化方法的新方法。

### 翻译

归一化是深度学习的基础，但现有的方法如BatchNorm、LayerNorm和RMSNorm都是方差中心的，通过强制零均值和单位方差来稳定训练，而没有控制表示如何捕获任务相关信息。我们提出了受信息瓶颈原理启发的归一化方法（IBNorm），这是一种简单而强大的方法系列。IBNorm引入了有界压缩操作，鼓励嵌入保留预测信息同时抑制无用变异性，从而产生更具信息量的表示，同时保持标准归一化的稳定性和兼容性。理论上，我们证明IBNorm比方差中心方法获得更高的IB值和更紧的泛化边界。实验上，IBNorm在大型语言模型（LLaMA、GPT-2）和视觉模型（ResNet、ViT）上一致优于BatchNorm、LayerNorm和RMSNorm，互信息分析证实了其优越的信息瓶颈行为。代码将公开发布。


### 论文摘要

Normalization is fundamental to deep learning, but existing approaches such as BatchNorm, LayerNorm, and RMSNorm are variance-centric by enforcing zero mean and unit variance, stabilizing training without controlling how representations capture task-relevant information. We propose IB-Inspired Normalization (IBNorm), a simple yet powerful family of methods grounded in the Information Bottleneck principle. IBNorm introduces bounded compression operations that encourage embeddings to preserve predictive information while suppressing nuisance variability, yielding more informative representations while retaining the stability and compatibility of standard normalization. Theoretically, we prove that IBNorm achieves a higher IB value and tighter generalization bounds than variance-centric methods. Empirically, IBNorm consistently outperforms BatchNorm, LayerNorm, and RMSNorm across large-scale language models (LLaMA, GPT-2) and vision models (ResNet, ViT), with mutual information analysis confirming superior information bottleneck behavior. Code will be released publicly.

---

## 8. Improving time series estimation and prediction via transfer learning

**论文链接:** [http://arxiv.org/abs/2510.25236v1](http://arxiv.org/abs/2510.25236v1)

**作者:** Yuchang Lin, Qianqian Zhu, Guodong Li

**发布时间:** 2025-10-29

### GPT解析

### 总结

本文提出了一种基于表示的迁移学习框架，用于解决高维度但样本量有限的时间序列数据集的估计和预测问题，通过利用相关源数据集的丰富观测信息提高估计效率。

### 背景

许多时间序列数据集（如宏观经济变量）具有高维度但样本量有限，仅使用这些数据集本身几乎无法获得有效的估计和准确的预测。

### 目的

引入一种基于表示的迁移学习框架用于向量自回归模型，利用相关源数据集的丰富观测信息，通过表示学习提高估计效率。

### 方法

提出一种具有良好非渐近性质的两阶段正则化估计程序，并建议使用交替更新算法来寻找估计值。该框架能够处理具有不同样本量和异步开始/结束时间点的时间序列，灵活整合来自不同数据集的信息。

### 主要发现

通过模拟实验评估了所提出方法的有限样本性能，并通过对日本和其他20个宏观经济变量的实证分析证明了该方法的有效性和实用性。

### 结论

该迁移学习框架解决了高维度但样本量有限的时间序列分析问题，通过利用相关源数据集的信息提高了估计效率和预测准确性。

### 翻译

现有文献中存在许多高维度但样本量有限的时间序列，如宏观经济变量，仅使用相应的数据集本身几乎不可能获得有效的估计和准确的预测。本文通过引入一种基于表示的迁移学习框架来填补这一空白，该框架用于向量自回归模型，可以通过表示学习利用来自相关源数据集的丰富观测信息来提高估计效率。提出了一种具有良好建立的非渐近性质的两阶段正则化估计程序，并建议使用交替更新算法来寻找估计值。我们的迁移学习框架可以处理具有不同样本量和异步开始/结束时间点的时间序列，从而在整合来自不同数据集的信息方面提供了显著的灵活性。进行了模拟实验来评估所提出方法的有限样本性能，并通过分析日本和其他20个宏观经济变量的实证分析证明了其有用性。


### 论文摘要

There are many time series in the literature with high dimension yet limited sample sizes, such as macroeconomic variables, and it is almost impossible to obtain efficient estimation and accurate prediction by using the corresponding datasets themselves. This paper fills the gap by introducing a novel representation-based transfer learning framework for vector autoregressive models, and information from related source datasets with rich observations can be leveraged to enhance estimation efficiency through representation learning. A two-stage regularized estimation procedure is proposed with well established non-asymptotic properties, and algorithms with alternating updates are suggested to search for the estimates. Our transfer learning framework can handle time series with varying sample sizes and asynchronous starting and/or ending time points, thereby offering remarkable flexibility in integrating information from diverse datasets. Simulation experiments are conducted to evaluate the finite-sample performance of the proposed methodology, and its usefulness is demonstrated by an empirical analysis on 20 macroeconomic variables from Japan and another nine countries.

---

## 9. Learning Fair Graph Representations with Multi-view Information Bottleneck

**论文链接:** [http://arxiv.org/abs/2510.25096v1](http://arxiv.org/abs/2510.25096v1)

**作者:** Chuxun Liu, Debo Cheng, Qingfeng Chen, Jiangzhang Gan, Jiuyong Li, Lin Liu

**发布时间:** 2025-10-29

### GPT解析

### 总结

FairMIB是一种多视图信息瓶颈框架，通过分解图为特征、结构和扩散视图，结合对比学习和逆概率加权邻域校正，有效减轻图神经网络中的偏见传播，在保持高任务效用的同时提高公平性。

### 背景

图神经网络在处理关系数据时表现优秀，但会放大训练数据中的偏见，将歧视性属性和结构不平衡传播到不公平的结果中。现有公平性方法将偏见视为单一来源，忽略了不同的属性和结构效应，导致公平性和实用性之间的次优权衡。

### 目的

开发一种能够同时考虑属性和结构效应的框架，以减轻图神经网络中的偏见传播，实现更好的公平性和实用性权衡。

### 方法

FairMIB是一种多视图信息瓶颈框架，将图分解为特征、结构和扩散三个视图。它使用对比学习最大化跨视图互信息实现无偏见表示学习，整合多视角条件信息瓶颈目标平衡任务效用和公平性，并在扩散视图中引入逆概率加权邻域校正减少偏见传播。

### 主要发现

FairMIB能够有效分解并处理不同类型的偏见，通过多视图方法实现了比现有方法更好的公平性和实用性权衡。实验表明它在五个真实世界基准数据集上达到了最先进的性能。

### 结论

FairMIB通过多视图信息瓶颈框架和创新的偏见缓解技术，成功解决了图神经网络中的偏见问题，在不牺牲任务效用的前提下显著提高了公平性。

### 翻译

图神经网络通过在节点特征和结构上传递消息，在关系数据上表现出色，但它们会放大训练数据中的偏见，将歧视性属性和结构不平衡传播到不公平的结果中。许多公平性方法将偏见视为单一来源，忽略了不同的属性和结构效应，导致公平性和实用性之间的次优权衡。为了克服这一挑战，我们提出了FairMIB，一种多视图信息瓶颈框架，旨在将图分解为特征、结构和扩散视图，以减轻图神经网络中的复杂度偏见。特别是，所提出的FairMIB采用对比学习来最大化跨视图互信息，实现无偏见的表示学习。它进一步整合多视角条件信息瓶颈目标，通过最小化与敏感属性的互信息来平衡任务效用和公平性。此外，FairMIB在扩散视图中引入了逆概率加权邻域校正，减少了消息传递过程中偏见的传播。在五个真实世界基准数据集上的实验表明，FairMIB在效用和公平性指标上都达到了最先进的性能。


### 论文摘要

Graph neural networks (GNNs) excel on relational data by passing messages over node features and structure, but they can amplify training data biases, propagating discriminatory attributes and structural imbalances into unfair outcomes. Many fairness methods treat bias as a single source, ignoring distinct attribute and structure effects and leading to suboptimal fairness and utility trade-offs. To overcome this challenge, we propose FairMIB, a multi-view information bottleneck framework designed to decompose graphs into feature, structural, and diffusion views for mitigating complexity biases in GNNs. Especially, the proposed FairMIB employs contrastive learning to maximize cross-view mutual information for bias-free representation learning. It further integrates multi-perspective conditional information bottleneck objectives to balance task utility and fairness by minimizing mutual information with sensitive attributes. Additionally, FairMIB introduces an inverse probability-weighted (IPW) adjacency correction in the diffusion view, which reduces the spread of bias propagation during message passing. Experiments on five real-world benchmark datasets demonstrate that FairMIB achieves state-of-the-art performance across both utility and fairness metrics.

---

## 10. Topic Analysis with Side Information: A Neural-Augmented LDA Approach

**论文链接:** [http://arxiv.org/abs/2510.24918v1](http://arxiv.org/abs/2510.24918v1)

**作者:** Biyi Fang, Kripa Rajshekhar, Truong Vo, Diego Klabjan

**发布时间:** 2025-10-28

### GPT解析

### 总结

nnLDA是一种创新的神经增强概率主题模型，通过神经先验机制动态整合辅助信息，解决了传统主题模型难以融入元数据、用户属性或文档标签等辅助信息的局限性，在多个基准数据集上表现出色。

### 背景

传统主题模型如LDA被广泛用于揭示文本语料库中的潜在结构，但这些模型往往难以整合辅助信息如元数据、用户属性或文档标签，限制了它们的表现力、个性化和可解释性。

### 目的

提出nnLDA，一种神经增强的概率主题模型，通过神经先验机制动态整合辅助信息，以克服传统主题模型的局限性。

### 方法

nnLDA将每个文档建模为潜在主题的混合，其中主题比例的先验由基于辅助特征的神经网络生成。这种设计使模型能够捕获辅助信息和主题分布之间复杂的非线性交互，并开发了随机变分期望最大化算法来联合优化神经和概率组件。

### 主要发现

在多个基准数据集上，nnLDA在主题一致性、困惑度和下游分类方面持续优于LDA和Dirichlet-Multinomial Regression。

### 结论

当辅助信息可用时，结合神经表示学习和概率主题建模能够带来显著优势，nnLDA证明了这种混合方法的有效性。

### 翻译

传统的主题模型如潜在狄利克雷分配（LDA）已被广泛用于揭示文本语料库中的潜在结构，但它们往往难以整合辅助信息，如元数据、用户属性或文档标签。这些局限性限制了它们的表现力、个性化和可解释性。为此，我们提出了nnLDA，一种神经增强的概率主题模型，通过神经先验机制动态整合辅助信息。nnLDA将每个文档建模为潜在主题的混合，其中主题比例的先验由基于辅助特征的神经网络生成。这种设计使模型能够捕获辅助信息和主题分布之间复杂的非线性交互，这是静态狄利克雷先验无法表示的。我们开发了一种随机变分期望最大化算法来联合优化神经和概率组件。在多个基准数据集上，nnLDA在主题一致性、困惑度和下游分类方面持续优于LDA和狄利克雷-多项式回归。这些结果强调了在辅助信息可用的情况下，结合神经表示学习和概率主题建模的好处。


### 论文摘要

Traditional topic models such as Latent Dirichlet Allocation (LDA) have been widely used to uncover latent structures in text corpora, but they often struggle to integrate auxiliary information such as metadata, user attributes, or document labels. These limitations restrict their expressiveness, personalization, and interpretability. To address this, we propose nnLDA, a neural-augmented probabilistic topic model that dynamically incorporates side information through a neural prior mechanism. nnLDA models each document as a mixture of latent topics, where the prior over topic proportions is generated by a neural network conditioned on auxiliary features. This design allows the model to capture complex nonlinear interactions between side information and topic distributions that static Dirichlet priors cannot represent. We develop a stochastic variational Expectation-Maximization algorithm to jointly optimize the neural and probabilistic components. Across multiple benchmark datasets, nnLDA consistently outperforms LDA and Dirichlet-Multinomial Regression in topic coherence, perplexity, and downstream classification. These results highlight the benefits of combining neural representation learning with probabilistic topic modeling in settings where side information is available.

---

## 11. Transformers from Compressed Representations

**论文链接:** [http://arxiv.org/abs/2510.23665v2](http://arxiv.org/abs/2510.23665v2)

**作者:** Juan C. Leon Alcazar, Mattia Soldan, Mohammad Saatialsoruji, Alejandro Pardo, Hani Itani, Juan Camilo Perez, Bernard Ghanem

**发布时间:** 2025-10-26

### GPT解析

### 总结

TEMPEST是一种利用压缩文件字节流结构进行表示学习的方法，通过紧凑编码实现高效语义表示，同时保持与最先进方法相当的准确性。

### 背景

压缩文件格式是高效数据存储和传输的基石，但其在表示学习方面的潜力尚未被充分探索。

### 目的

引入一种能够利用压缩文件固有字节流结构进行有效标记化和编码策略的方法，直接从压缩数据流中学习语义表示。

### 方法

TEMPEST（TransformErs froM comPressed rEpreSenTations）利用压缩文件的固有字节流结构设计标记化和编码策略，使标准transformer可以直接从压缩数据流中学习语义表示，绕过原始字节级处理或完整媒体解码的需要。

### 主要发现

TEMPEST显著减少了语义分类所需的标记数量，降低了计算复杂性和内存使用；在多个数据集、编码方案和模态的实验中，实现了与最先进方法相当的准确性，同时在内存和计算方面提高了效率。

### 结论

TEMPEST是一种有效的方法，可以从压缩数据中学习语义表示，在保持准确性的同时提高了效率。

### 翻译

压缩文件格式是高效数据存储和传输的基石，但其在表示学习方面的潜力仍未被充分探索。我们引入了TEMPEST（一种基于压缩表示的transformer方法），它利用压缩文件的固有字节流结构来设计有效的标记化和编码策略。通过利用这种紧凑编码，标准的transformer可以直接从压缩数据流中学习语义表示，绕过了原始字节级处理或完整媒体解码的需要。我们的提议显著减少了语义分类所需的标记数量，从而降低了计算复杂性和内存使用。通过在不同数据集、编码方案和模态上的广泛实验，我们表明TEMPEST实现了与最先进方法相当的准确性，同时在内存和计算方面带来了效率提升。


### 论文摘要

Compressed file formats are the corner stone of efficient data storage and transmission, yet their potential for representation learning remains largely underexplored. We introduce TEMPEST (TransformErs froM comPressed rEpreSenTations), a method that exploits the inherent byte-stream structure of compressed files to design an effective tokenization and encoding strategy. By leveraging this compact encoding, a standard transformer can directly learn semantic representations from compressed data streams, bypassing the need for raw byte-level processing or full media decoding. Our proposal substantially reduces the number of tokens required for semantic classification, thereby lowering both computational complexity and memory usage. Through extensive experiments across diverse datasets, coding schemes, and modalities, we show that TEMPEST achieves accuracy competitive wit the state-of-the-art while delivering efficiency gains in memory and compute.

---

## 12. Cross-Enhanced Multimodal Fusion of Eye-Tracking and Facial Features for Alzheimer's Disease Diagnosis

**论文链接:** [http://arxiv.org/abs/2510.24777v1](http://arxiv.org/abs/2510.24777v1)

**作者:** Yujie Nie, Jianzhang Ni, Yonglong Ye, Yuan-Ting Zhang, Yun Kwok Wing, Xiangqing Xu, Xin Ma, Lizhou Fan

**发布时间:** 2025-10-25

**备注:** 35 pages, 8 figures, and 7 tables

### GPT解析

### 总结

本研究提出了一种多模态交叉增强融合框架，通过整合眼动追踪和面部特征进行阿尔茨海默病诊断，并在包含25名AD患者和25名健康对照者的数据集上实现了95.11%的分类准确率。

### 背景

准确诊断阿尔茨海默病对及时干预和减缓疾病进展至关重要。多模态诊断方法通过整合行为和感知领域的互补信息显示出巨大潜力，而眼动追踪和面部特征是认知功能的重要指标，但很少有研究探索它们的联合集成用于辅助AD诊断。

### 目的

开发一种能够协同利用眼动追踪和面部特征进行AD检测的多模态交叉增强融合框架，提高诊断性能的鲁棒性和准确性。

### 方法

提出一个包含两个关键模块的多模态框架：(a)交叉增强融合注意力模块(CEFAM)，通过交叉注意力和全局增强建模模态间交互；(b)方向感知卷积模块(DACM)，通过水平-垂直感受野捕获细粒度方向性面部特征。同时构建了一个同步多模态数据集，包括AD患者和健康对照者在视觉记忆搜索范式中的面部视频和眼动追踪序列。

### 主要发现

在构建的数据集上，该框架优于传统的后期融合和特征连接方法，在区分AD和健康对照者方面实现了95.11%的分类准确率，通过明确建模模态间依赖性和模态特定贡献，显示出卓越的鲁棒性和诊断性能。

### 结论

多模态交叉增强融合框架通过协同整合眼动追踪和面部特征，能够有效提高阿尔茨海默病的诊断准确性和鲁棒性，为AD的辅助诊断提供了新的方法。

### 翻译

阿尔茨海默病(AD)的准确诊断对于实现及时干预和减缓疾病进展至关重要。多模态诊断方法通过整合行为和感知领域的互补信息显示出巨大潜力。特别是，眼动追踪和面部特征是认知功能的重要指标，反映了注意力分布和神经认知状态。然而，很少有研究探索它们的联合集成用于辅助AD诊断。在本研究中，我们提出了一种多模态交叉增强融合框架，通过协同利用眼动追踪和面部特征进行AD检测。该框架包含两个关键模块：(a)交叉增强融合注意力模块(CEFAM)，通过交叉注意力和全局增强建模模态间交互；(b)方向感知卷积模块(DACM)，通过水平-垂直感受野捕获细粒度方向性面部特征。这些模块共同实现了自适应和判别性多模态表示学习。为支持这项工作，我们构建了一个同步多模态数据集，包括25名AD患者和25名健康对照者(HC)，通过在视觉记忆搜索范式期间记录对齐的面部视频和眼动追踪序列，为评估集成策略提供了生态有效资源。在该数据集上的大量实验表明，我们的框架优于传统的后期融合和特征连接方法，在区分AD和HC方面实现了95.11%的分类准确率，通过明确建模模态间依赖性和模态特定贡献，突显了卓越的鲁棒性和诊断性能。


### 论文摘要

Accurate diagnosis of Alzheimer's disease (AD) is essential for enabling timely intervention and slowing disease progression. Multimodal diagnostic approaches offer considerable promise by integrating complementary information across behavioral and perceptual domains. Eye-tracking and facial features, in particular, are important indicators of cognitive function, reflecting attentional distribution and neurocognitive state. However, few studies have explored their joint integration for auxiliary AD diagnosis. In this study, we propose a multimodal cross-enhanced fusion framework that synergistically leverages eye-tracking and facial features for AD detection. The framework incorporates two key modules: (a) a Cross-Enhanced Fusion Attention Module (CEFAM), which models inter-modal interactions through cross-attention and global enhancement, and (b) a Direction-Aware Convolution Module (DACM), which captures fine-grained directional facial features via horizontal-vertical receptive fields. Together, these modules enable adaptive and discriminative multimodal representation learning. To support this work, we constructed a synchronized multimodal dataset, including 25 patients with AD and 25 healthy controls (HC), by recording aligned facial video and eye-tracking sequences during a visual memory-search paradigm, providing an ecologically valid resource for evaluating integration strategies. Extensive experiments on this dataset demonstrate that our framework outperforms traditional late fusion and feature concatenation methods, achieving a classification accuracy of 95.11% in distinguishing AD from HC, highlighting superior robustness and diagnostic performance by explicitly modeling inter-modal dependencies and modality-specific contributions.

---

## 13. WBT-BGRL: A Non-Contrastive Weighted Bipartite Link Prediction Model for Inductive Learning

**论文链接:** [http://arxiv.org/abs/2510.24927v1](http://arxiv.org/abs/2510.24927v1)

**作者:** Joel Frank Huarayo Quispe, Lilian Berton, Didier Vega-Oliveros

**发布时间:** 2025-10-28

**备注:** 5 pages, submitted to the 12th International Conference on Soft  Computing and Machine Intelligence (ISCMI 2025)

### GPT解析

### 总结

本文提出了WBT-BGRL框架，一种用于二分图链接预测的加权非对比学习方法，通过三元损失中的加权机制增强自举学习，在真实数据集上展示了有竞争力的性能。

### 背景

二分图链接预测对推荐系统和故障检测等应用至关重要，但研究较少；对比方法在负采样上效率低且偏差大，非对比方法仅依赖正样本；现有模型在直推式设置中表现良好，但在归纳式、加权和二分场景中效果未验证。

### 目的

解决现有二分图链接预测方法的局限性，特别是在归纳、加权和二分场景中的有效性问题。

### 方法

提出加权二分图三元自举图潜在表示(WBT-BGRL)，采用非对比框架，通过三元损失中的新加权机制增强自举学习；使用具有双GCN编码器的二分架构；与适配的最先进模型(T-BGRL, BGRL, GBT, CCA-SSG)进行比较评估。

### 主要发现

在工业和电子商务真实数据集上，WBT-BGRL展现出有竞争力的性能，特别是在预训练过程中应用加权时效果更佳，突显了加权的非对比学习在二分图归纳链接预测中的价值。

### 结论

加权的非对比学习对于二分图中的归纳链接预测具有重要价值，WBT-BGRL框架为此提供了有效的解决方案。

### 翻译

二分图中的链接预测对于推荐系统和故障检测等应用至关重要，但相比单分图的研究较少。对比方法在负采样方面效率低下且存在偏差，而非对比方法仅依赖正样本。现有模型在直推式设置中表现良好，但在归纳式、加权和二分场景中的有效性尚未得到验证。为解决这一问题，我们提出了加权二分图三元自举图潜在表示(WBT-BGRL)，这是一种非对比框架，通过三元损失中的新加权机制增强自举学习。使用具有双GCN编码器的二分架构，将WBT-BGRL与适配的最先进模型(T-BGRL, BGRL, GBT, CCA-SSG)进行比较评估。在工业和电子商务真实世界数据集上的结果显示了具有竞争力的性能，特别是在预训练过程中应用加权时，突显了加权的非对比学习在二分图归纳链接预测中的价值。


### 论文摘要

Link prediction in bipartite graphs is crucial for applications like recommendation systems and failure detection, yet it is less studied than in monopartite graphs. Contrastive methods struggle with inefficient and biased negative sampling, while non-contrastive approaches rely solely on positive samples. Existing models perform well in transductive settings, but their effectiveness in inductive, weighted, and bipartite scenarios remains untested. To address this, we propose Weighted Bipartite Triplet-Bootstrapped Graph Latents (WBT-BGRL), a non-contrastive framework that enhances bootstrapped learning with a novel weighting mechanism in the triplet loss. Using a bipartite architecture with dual GCN encoders, WBT-BGRL is evaluated against adapted state-of-the-art models (T-BGRL, BGRL, GBT, CCA-SSG). Results on real-world datasets (Industry and E-commerce) show competitive performance, especially when weighting is applied during pretraining-highlighting the value of weighted, non-contrastive learning for inductive link prediction in bipartite graphs.

---

## 14. Improving Temporal Consistency and Fidelity at Inference-time in Perceptual Video Restoration by Zero-shot Image-based Diffusion Models

**论文链接:** [http://arxiv.org/abs/2510.25420v1](http://arxiv.org/abs/2510.25420v1)

**作者:** Nasrin Rahimi, A. Murat Tekalp

**发布时间:** 2025-10-29

### GPT解析

### 总结

本研究提出两种无需训练的推理时策略，以改善基于零样本图像扩散模型的时间一致性视频修复，通过感知直线引导(PSG)和多路径集成采样(MPES)技术，实现时间稳定的高保真感知视频修复。

### 背景

扩散模型已成为单图像修复的强大先验，但由于采样的随机性和整合显式时间建模的复杂性，将其应用于零样本视频修复时存在时间一致性问题。

### 目的

在不重新训练或修改预训练扩散模型架构的情况下，提高视频修复中的时间一致性，实现时间稳定的高保真感知视频修复。

### 方法

提出两种互补的推理时策略：(1)感知直线引导(PSG)：基于神经科学启发的感知直线假设，通过在感知空间中引入曲率惩罚，引导扩散去噪过程向更平滑的时间演化发展；(2)多路径集成采样(MPES)：通过集成多个扩散轨迹来减少随机变化，提高保真度分数而不牺牲清晰度。

### 主要发现

PSG增强了时间自然性，特别是在时间模糊的情况下；MPES在所有任务中一致提高了保真度和时空感知-失真权衡。这两种策略无需重新训练或修改模型架构即可实现显著改进。

### 结论

这些无需训练的技术为使用大型预训练扩散模型实现时间稳定的高保真感知视频修复提供了实用路径，通过结合PSG和MPES可以同时改善时间自然性和保真度。

### 翻译

扩散模型已成为单图像修复的强大先验，但将其应用于零样本视频修复时，由于采样的随机性和整合显式时间建模的复杂性，存在时间一致性问题。在本工作中，我们解决了在不重新训练或修改其架构的情况下，使用零样本基于图像的扩散模型提高视频修复时间一致性的挑战。我们提出了两种互补的推理时策略：(1)基于神经科学启发的感知直线假设的感知直线引导(PSG)，通过在感知空间中引入曲率惩罚来引导扩散去噪过程向更平滑的时间演化发展，以改善时间感知分数，如Fréchet视频距离(FVD)和感知直线度；(2)多路径集成采样(MPES)，旨在通过集成多个扩散轨迹来减少随机变化，提高保真度(失真)分数，如PSNR和SSIM，而不牺牲清晰度。这些无需训练的技术共同为使用大型预训练扩散模型实现时间稳定的高保真感知视频修复提供了实用路径。我们在多个数据集和退化类型上进行了广泛实验，系统评估了每种策略以了解其优势和局限性。我们的结果表明，虽然PSG增强了时间自然性，特别是在时间模糊的情况下，但MPES在所有任务中一致提高了保真度和时空感知-失真权衡。


### 论文摘要

Diffusion models have emerged as powerful priors for single-image restoration, but their application to zero-shot video restoration suffers from temporal inconsistencies due to the stochastic nature of sampling and complexity of incorporating explicit temporal modeling. In this work, we address the challenge of improving temporal coherence in video restoration using zero-shot image-based diffusion models without retraining or modifying their architecture. We propose two complementary inference-time strategies: (1) Perceptual Straightening Guidance (PSG) based on the neuroscience-inspired perceptual straightening hypothesis, which steers the diffusion denoising process towards smoother temporal evolution by incorporating a curvature penalty in a perceptual space to improve temporal perceptual scores, such as Fr\'echet Video Distance (FVD) and perceptual straightness; and (2) Multi-Path Ensemble Sampling (MPES), which aims at reducing stochastic variation by ensembling multiple diffusion trajectories to improve fidelity (distortion) scores, such as PSNR and SSIM, without sacrificing sharpness. Together, these training-free techniques provide a practical path toward temporally stable high-fidelity perceptual video restoration using large pretrained diffusion models. We performed extensive experiments over multiple datasets and degradation types, systematically evaluating each strategy to understand their strengths and limitations. Our results show that while PSG enhances temporal naturalness, particularly in case of temporal blur, MPES consistently improves fidelity and spatio-temporal perception--distortion trade-off across all tasks.

---

## 15. StreamingCoT: A Dataset for Temporal Dynamics and Multimodal Chain-of-Thought Reasoning in Streaming VideoQA

**论文链接:** [http://arxiv.org/abs/2510.25332v1](http://arxiv.org/abs/2510.25332v1)

**作者:** Yuhang Hu, Zhenyu Yang, Shihan Wang, Shengsheng Qian, Bin Wen, Fan Yang, Tingting Gao, Changsheng Xu

**发布时间:** 2025-10-29

**DOI:** 10.1145/3746027.3758311

### GPT解析

### 总结

StreamingCoT是首个专为流媒体视频问答中的时间演化和多模态思维链任务设计的数据集，解决了现有VideoQA数据集无法捕捉时间动态和缺少明确推理过程标注的问题。

### 背景

流媒体视频应用的快速增长需要具有增强时间动态理解和复杂推理能力多模态模型，但当前VideoQA数据集存在静态标注机制无法捕捉视频流中答案的演变性质，以及缺少明确推理过程标注两大关键限制。

### 目的

解决现有VideoQA数据集的两个关键限制：1)静态标注机制无法捕捉时间视频流中答案的演变性质；2)缺少明确的推理过程标注，限制了模型的可解释性和逻辑推理能力。

### 方法

建立动态分层标注架构，生成每秒密集描述并通过相似性融合构建时间依赖的语义段，加入受时间演化模式约束的问题-答案集；提出明确推理链生成范式，通过关键帧语义提取时空对象，使用大语言模型基于对象状态转换推导推理路径，并通过人工验证确保逻辑一致性。

### 主要发现

通过StreamingCoT数据集的构建，为推进流媒体视频理解、复杂时间推理和多模态推理研究奠定了基础。

### 结论

StreamingCoT数据集及其构建工具包为解决流媒体视频理解中的时间动态和复杂推理问题提供了有效支持，相关资源已在GitHub上公开。

### 翻译

流媒体视频应用的快速增长需要具有增强时间动态理解和复杂推理能力多模态模型。然而，当前视频问答(VideoQA)数据集存在两个关键限制：1)静态标注机制无法捕捉时间视频流中答案的演变性质；2)缺少明确的推理过程标注，限制了模型的可解释性和逻辑推理能力。为解决这些挑战，我们引入了StreamingCoT，这是首个专为流媒体视频问答中的时间演化推理和多模态思维链(CoT)任务设计的数据集。我们的框架首先建立了动态分层标注架构，生成每秒密集描述并通过相似性融合构建时间依赖的语义段，同时加入受时间演化模式约束的问题-答案集。我们进一步提出了明确的推理链生成范式，通过关键帧语义对齐提取时空对象，使用大语言模型基于对象状态转换推导推理路径，并通过人工验证确保逻辑一致性。该数据集为推进流媒体视频理解、复杂时间推理和多模态推理研究奠定了基础。我们的StreamingCoT及其构建工具包可在https://github.com/Fleeting-hyh/StreamingCoT获取。


### 论文摘要

The rapid growth of streaming video applications demands multimodal models with enhanced capabilities for temporal dynamics understanding and complex reasoning. However, current Video Question Answering (VideoQA) datasets suffer from two critical limitations: 1) Static annotation mechanisms fail to capture the evolving nature of answers in temporal video streams, and 2) The absence of explicit reasoning process annotations restricts model interpretability and logical deduction capabilities. To address these challenges, We introduce StreamingCoT, the first dataset explicitly designed for temporally evolving reasoning in streaming VideoQA and multimodal Chain-of-Thought (CoT) tasks. Our framework first establishes a dynamic hierarchical annotation architecture that generates per-second dense descriptions and constructs temporally-dependent semantic segments through similarity fusion, paired with question-answer sets constrained by temporal evolution patterns. We further propose an explicit reasoning chain generation paradigm that extracts spatiotemporal objects via keyframe semantic alignment, derives object state transition-based reasoning paths using large language models, and ensures logical coherence through human-verified validation. This dataset establishes a foundation for advancing research in streaming video understanding, complex temporal reasoning, and multimodal inference. Our StreamingCoT and its construction toolkit can be accessed at https://github.com/Fleeting-hyh/StreamingCoT.

---

## 16. Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks

**论文链接:** [http://arxiv.org/abs/2510.25760v1](http://arxiv.org/abs/2510.25760v1)

**作者:** Xu Zheng, Zihao Dongfang, Lutao Jiang, Boyuan Zheng, Yulong Guo, Zhenquan Zhang, Giuliano Albanese, Runyi Yang, Mengjiao Ma, Zixin Zhang, Chenfei Liao, Dingcheng Zhen, Yuanhuiyi Lyu, Yuqian Fu, Bin Ren, Linfeng Zhang, Danda Pani Paudel, Nicu Sebe, Luc Van Gool, Xuming Hu

**发布时间:** 2025-10-29

### GPT解析

### 总结

这篇综述文章全面回顾了大型多模态空间推理模型在各类任务中的进展，分类了多模态大语言模型的最新研究，并引入了开放基准进行评估。

### 背景

人类具有空间推理能力，能够通过视觉和声音等多模态观察来理解空间。大型多模态推理模型扩展了这些能力，在各种空间任务中展现出有希望的性能，但系统性综述和公开可用的基准仍然有限。

### 目的

提供对大型多模态空间推理任务的全面回顾，分类多模态大语言模型的进展，并引入用于评估的开放基准。

### 方法

文章首先概述通用空间推理，重点关注训练后技术、可解释性和架构。研究内容包括空间关系推理、场景和布局理解、3D空间中的视觉问答和定位，以及具身AI（如视觉语言导航和动作模型）。此外还探讨了音频和第一人称视频等新兴模态对空间理解的贡献。

### 主要发现

多模态空间推理模型在2D和3D空间理解、视觉问答、具身AI等任务中展现出有希望的性能。音频和第一人称视频等新兴模态通过新传感器为空间理解提供了新的视角。

### 结论

这篇综述为多模态空间推理这一不断发展的领域奠定了坚实的基础，并提供了有价值的见解。

### 翻译

人类拥有空间推理能力，使他们能够通过多模态观察（如视觉和声音）来理解空间。大型多模态推理模型通过学习感知和推理扩展了这些能力，在各种空间任务中展现出有希望的性能。然而，对这些模型的系统性综述和公开可用的基准仍然有限。在本综述中，我们对大型多模态空间推理任务进行了全面回顾，分类了多模态大语言模型的最新进展，并引入了用于评估的开放基准。我们首先概述了通用的空间推理，重点关注训练后技术、可解释性和架构。除了传统的2D任务外，我们还研究了空间关系推理、场景和布局理解，以及3D空间中的视觉问答和定位。我们还回顾了具身AI的进展，包括视觉语言导航和动作模型。此外，我们还考虑了音频和第一人称视频等新兴模态，这些模态通过新传感器为空间理解做出贡献。我们相信本综述为多模态空间推理这一不断发展的领域奠定了坚实的基础，并提供了有价值的见解。关于本综述的更新信息、开放基准的代码和实现可以在https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning找到。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决多模态空间推理在大型模型时代缺乏系统综述和公开基准测试的问题。这个问题很重要，因为人类通过视觉、声音等多模态输入理解空间的能力是基础性的，而大型语言模型虽然文本处理能力强，但空间推理能力有限。整合多模态信息增强空间推理对机器人导航、增强现实、自动驾驶等现实应用至关重要，同时缺乏系统评估阻碍了该领域的标准化发展和比较研究。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者通过系统性回顾和分析现有文献构建了这篇综述。他们首先定义多模态空间推理，然后分类各类空间任务（从2D到3D，从静态到动态），分析技术进展（测试时扩展、后训练方法、架构修改等），最后引入评估基准。作者确实借鉴了大量现有工作，如Wang等人的小型推理模型研究、Ke等人的推理扩展分析、Zha等人的3D能力研究等，但指出这些工作要么未深入多模态空间推理，要么缺乏系统评估框架，因此他们的综述填补了这一空白。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是提供首个多模态空间推理在大型模型时代的全面综述，建立系统分类框架，并引入开放基准。整体流程：1)定义多模态空间推理并概述评估维度；2)分析一般多模态空间推理技术（测试时扩展、后训练、架构修改等）；3)探讨3D空间中的核心任务（视觉定位、场景推理、3D生成）；4)讨论具身AI中的空间推理；5)考虑音频和第一人称视频等新兴模态；6)提供开放基准和评估框架。作者还通过GitHub仓库提供代码和最新信息，方便研究实践。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点：1)构建首个专门针对多模态空间推理的系统综述框架；2)建立详细任务分类体系，涵盖从2D到3D、静态到动态、视觉到其他模态的广泛任务；3)引入开放基准标准化评估；4)整合音频和第一人称视频等新兴模态；5)提供跨领域视角连接传统2D理解与3D推理、具身AI等。相比之前工作，本文专注多模态空间推理而非一般推理，提供系统性评估框架而非单一任务分析，引入开放基准而非仅文献回顾，并通过GitHub提供实用资源，更全面且实用。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': '这篇论文为多模态空间推理在大型模型时代提供了首个全面的综述框架，系统性地分类了各类空间任务，引入了开放评估基准，并通过整合新兴模态为该领域的研究和实践奠定了坚实基础。'}


### 论文摘要

Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on post-training techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.

---

## 17. EA3D: Online Open-World 3D Object Extraction from Streaming Videos

**论文链接:** [http://arxiv.org/abs/2510.25146v1](http://arxiv.org/abs/2510.25146v1)

**作者:** Xiaoyu Zhou, Jingqi Wang, Yuang Jia, Yongtao Wang, Deqing Sun, Ming-Hsuan Yang

**发布时间:** 2025-10-29

**备注:** The Thirty-Ninth Annual Conference on Neural Information Processing  Systems(NeurIPS 2025)

### GPT解析

### 总结

本文提出了ExtractAnything3D (EA3D)，一个统一的在线开放世界3D物体提取框架，能够同时实现几何重建和整体场景理解。

### 背景

当前3D场景理解方法受限于离线收集的多视图数据或预构建的3D几何形状。

### 目的

开发一个统一的在线框架，用于开放世界的3D物体提取，同时实现几何重建和整体场景理解。

### 方法

EA3D通过视觉语言和2D视觉基础编码器动态解释每个视频帧提取物体级知识，使用前馈在线更新策略将知识集成到高斯特征图中，从历史帧迭代估计视觉里程计并增量更新高斯特征，通过循环联合优化模块引导模型关注感兴趣区域。

### 主要发现

EA3D在多样化基准测试和任务中表现出色，包括照片级真实感渲染、语义和实例分割、3D边界框和语义占用估计以及3D网格生成。

### 结论

EA3D建立了统一的、高效的框架，用于联合在线3D重建和整体场景理解，能够支持广泛的下游任务。

### 翻译

当前的3D场景理解方法受限于离线收集的多视图数据或预构建的3D几何形状。在本文中，我们提出了ExtractAnything3D (EA3D)，一个统一的在线开放世界3D物体提取框架，能够同时实现几何重建和整体场景理解。给定流式视频，EA3D使用视觉语言和2D视觉基础编码器动态解释每个帧，提取物体级知识。这些知识通过前馈在线更新策略被集成并嵌入到高斯特征图中。然后我们从历史帧迭代估计视觉里程计，并用新观察增量更新在线高斯特征。循环联合优化模块引导模型关注感兴趣区域，同时增强几何重建和语义理解。在多样化的基准测试和任务中的大量实验，包括照片级真实感渲染、语义和实例分割、3D边界框和语义占用估计以及3D网格生成，证明了EA3D的有效性。我们的方法为联合在线3D重建和整体场景理解建立了统一且高效的框架，能够支持广泛的下游任务。

### 深入解读

{'这篇论文主要想解决什么问题？这个问题在现实或研究中为什么重要？': '这篇论文主要解决如何从流式视频中实时提取和理解开放世界中的3D物体问题。这个问题在现实中非常重要，因为自主智能体（如机器人）需要在陌生环境中实时理解和重建周围环境，而现实世界中的场景是开放的，物体种类和数量未知，需要同时处理流式视频输入并理解物体的几何结构和语义信息。', '作者是如何思考并设计出这个方法的？是否有借鉴现有工作？': '作者观察到现有的视觉语言模型在2D开放世界理解上表现出色，但在3D领域存在视角不一致和几何错位问题。他们发现直接将2D模型提升到3D的方法需要预构建的几何和标注数据，而现有的可微分渲染框架又需要完整的多视图图像。受人类感知启发，作者设计了EA3D，使其能像人类一样进入环境时立即开始处理视觉输入。该方法借鉴了视觉语言模型进行开放世界解释，利用视觉基础模型提取特征，基于高斯泼溅构建在线表示，并参考了在线视觉里程计和高斯更新的方法。', '这个方法的核心思想是什么？整体实现流程是怎样的？': '核心思想是建立一个统一的在线开放世界3D物体提取框架，能同时进行几何重建和场景理解，无需几何或姿态先验。整体流程包括：1)知识提取与集成：使用VLMs识别物体，维护语义缓存，利用VFMs提取特征并嵌入高斯表示；2)在线3D物体提取：通过在线视觉里程计估计相机姿态，利用在线高斯更新重建几何和理解语义；3)循环联合优化：设计语义感知正则化，联合优化高斯特征和相机姿态，结合多种损失函数提升重建和理解质量。', '论文的关键创新点有哪些？相比之前的工作，有什么不同？': '关键创新点包括：1)统一的在线开放世界3D物体提取框架，能同时进行重建和理解；2)在线特征高斯表示，结合在线视觉里程计和高斯更新；3)循环联合优化策略，动态引导模型注意力；4)支持多种下游任务。相比之前的工作，EA3D的不同之处在于：它能在线处理流式视频而非依赖完整多视图；能处理开放世界中的未知物体类别；无需预构建几何或姿态先验；提供支持多种任务的统一框架而非专注于单一任务。', '如果要用一句话总结这篇论文的贡献，你会怎么说？': 'EA3D提出了一种统一的在线框架，能够从流式视频中实时提取开放世界中的3D物体，同时进行几何重建和语义理解，无需任何几何或姿态先验，支持多种下游3D感知任务。'}


### 论文摘要

Current 3D scene understanding methods are limited by offline-collected multi-view data or pre-constructed 3D geometry. In this paper, we present ExtractAnything3D (EA3D), a unified online framework for open-world 3D object extraction that enables simultaneous geometric reconstruction and holistic scene understanding. Given a streaming video, EA3D dynamically interprets each frame using vision-language and 2D vision foundation encoders to extract object-level knowledge. This knowledge is integrated and embedded into a Gaussian feature map via a feed-forward online update strategy. We then iteratively estimate visual odometry from historical frames and incrementally update online Gaussian features with new observations. A recurrent joint optimization module directs the model's attention to regions of interest, simultaneously enhancing both geometric reconstruction and semantic understanding. Extensive experiments across diverse benchmarks and tasks, including photo-realistic rendering, semantic and instance segmentation, 3D bounding box and semantic occupancy estimation, and 3D mesh generation, demonstrate the effectiveness of EA3D. Our method establishes a unified and efficient framework for joint online 3D reconstruction and holistic scene understanding, enabling a broad range of downstream tasks.

---

## 18. Vision-Language Integration for Zero-Shot Scene Understanding in Real-World Environments

**论文链接:** [http://arxiv.org/abs/2510.25070v1](http://arxiv.org/abs/2510.25070v1)

**作者:** Manjunath Prasad Holenarasipura Rajiv, B. M. Vidyavathi

**发布时间:** 2025-10-29

**备注:** Preprint under review at IEEE Transactions on Pattern Analysis and  Machine Intelligence (TPAMI), 2025

### GPT解析

### 总结

该研究提出了一种视觉-语言集成框架，通过统一预训练视觉编码器和大语言模型，实现零样本场景理解，在多个数据集上取得了显著性能提升。

### 背景

真实世界环境中的零样本场景理解面临重大挑战，由于自然场景的复杂性和可变性，模型必须在没有先前标记示例的情况下识别新对象、动作和上下文。

### 目的

实现稳健的零样本场景理解，利用自然语言作为桥梁，推广到未见过的类别和上下文。

### 方法

提出视觉-语言集成框架，统一预训练视觉编码器（如CLIP、ViT）和大语言模型（如基于GPT的架构），开发将视觉输入和文本提示嵌入共享空间的统一模型，并使用多模态融合和推理层进行上下文解释。

### 主要发现

在Visual Genome、COCO、ADE20K和自定义真实世界数据集上的实验表明，与最先进的零样本模型相比，在对象识别、活动检测和场景字幕生成方面有显著提升，top-1准确率提高高达18%，语义一致性指标也有显著提升。

### 结论

跨模态对齐和语言锚定在增强真实世界场景理解的泛化能力方面非常有效。

### 翻译

真实世界环境中的零样本场景理解由于自然场景的复杂性和可变性而面临重大挑战，模型必须在没有先前标记示例的情况下识别新对象、动作和上下文。这项工作提出了一种视觉-语言集成框架，统一了预训练的视觉编码器（如CLIP、ViT）和大语言模型（如基于GPT的架构），以实现视觉和文本模态之间的语义对齐。目标是利用自然语言作为桥梁，推广到未见过的类别和上下文，实现稳健的零样本场景理解。我们的方法开发了一个统一模型，将视觉输入和文本提示嵌入到共享空间，然后使用多模态融合和推理层进行上下文解释。在Visual Genome、COCO、ADE20K和自定义真实世界数据集上的实验表明，与最先进的零样本模型相比，在对象识别、活动检测和场景字幕生成方面有显著提升。提出的系统在top-1准确率上提高了高达18%，在语义一致性指标方面也有显著提升，突显了跨模态对齐和语言锚定在增强真实世界场景理解泛化能力方面的有效性。


### 论文摘要

Zero-shot scene understanding in real-world settings presents major challenges due to the complexity and variability of natural scenes, where models must recognize new objects, actions, and contexts without prior labeled examples. This work proposes a vision-language integration framework that unifies pre-trained visual encoders (e.g., CLIP, ViT) and large language models (e.g., GPT-based architectures) to achieve semantic alignment between visual and textual modalities. The goal is to enable robust zero-shot comprehension of scenes by leveraging natural language as a bridge to generalize over unseen categories and contexts. Our approach develops a unified model that embeds visual inputs and textual prompts into a shared space, followed by multimodal fusion and reasoning layers for contextual interpretation. Experiments on Visual Genome, COCO, ADE20K, and custom real-world datasets demonstrate significant gains over state-of-the-art zero-shot models in object recognition, activity detection, and scene captioning. The proposed system achieves up to 18% improvement in top-1 accuracy and notable gains in semantic coherence metrics, highlighting the effectiveness of cross-modal alignment and language grounding in enhancing generalization for real-world scene understanding.

---

## 19. PISA-Bench: The PISA Index as a Multilingual and Multimodal Metric for the Evaluation of Vision-Language Models

**论文链接:** [http://arxiv.org/abs/2510.24792v1](http://arxiv.org/abs/2510.24792v1)

**作者:** Patrick Haller, Fabio Barth, Jonas Golde, Georg Rehm, Alan Akbik

**发布时间:** 2025-10-27

**备注:** 8 pages, 11 tables and figures

### GPT解析

### 总结

PISA-Bench是一个基于PISA测试的多语言多模态推理基准，包含六种语言的平行数据集，用于评估视觉语言模型在不同语言和推理任务上的表现。

### 背景

视觉语言模型在多模态推理方面取得显著进展，但现有基准测试在高质量、人工验证的例子方面有限，且大多数数据集仅限于英语，翻译样本的质量保证既耗时又昂贵。

### 目的

填补高质量多语言多模态推理基准的空白，提供一个包含多种语言的人工验证数据集。

### 方法

从专家创建的PISA测试英语例子中衍生出PISA-Bench，包含人工提取的指令、问题、答案选项和图像，并增加问题类型分类；将这些内容从英语翻译成西班牙语、德语、中文、法语和意大利语，形成完全平行的六语言语料库。

### 主要发现

小型视觉语言模型(<20B参数)在PISA-Bench上无法获得高分；模型在非英语部分的性能显著下降；当处理空间和几何推理任务时，模型错误率高。

### 结论

通过发布PISA-Bench数据集和评估框架，为推进多模态多语言推理研究提供了重要资源。

### 翻译

原文摘要为英文，上述内容已将其核心信息翻译并结构化为中文。


### 论文摘要

Vision-language models (VLMs) have demonstrated remarkable progress in multimodal reasoning. However, existing benchmarks remain limited in terms of high-quality, human-verified examples. Many current datasets rely on synthetically generated content by large language models (LLMs). Furthermore, most datasets are limited to English, as manual quality assurance of translated samples is time-consuming and costly. To fill this gap, we introduce PISA-Bench, a multilingual benchmark derived from English examples of the expert-created PISA tests, a unified framework for the assessment of student competencies in over eighty countries. Each example consists of human-extracted instructions, questions, answer options, and images, enriched with question type categories, and has been translated from English into five additional languages (Spanish, German, Chinese, French, and Italian), resulting in a fully parallel corpus covering six languages. We evaluate state-of-the-art vision-language models on PISA-Bench and find that especially small models (<20B parameters) fail to achieve high test scores. We further find substantial performance degradation on non-English splits as well as high error-rates when models are tasked with spatial and geometric reasoning. By releasing the dataset and evaluation framework, we provide a resource for advancing research on multilingual multimodal reasoning.

---

## 20. How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs

**论文链接:** [http://arxiv.org/abs/2510.25753v1](http://arxiv.org/abs/2510.25753v1)

**作者:** Samet Demir, Zafer Dogan

**发布时间:** 2025-10-29

**备注:** NeurIPS 2025, 24 pages, 6 figures

### GPT解析

### 总结

这篇论文研究了预训练Transformer模型中的上下文学习（ICL）能力，特别是在具有非线性MLP头部的模型上，从多个异构数据源学习的非线性任务。作者通过理论分析和实验验证，证明了非线性MLP相比线性基线能显著提升ICL性能，特别是在非线性任务上，并确定了高质量数据源的关键特性和特征学习的条件。

### 背景

预训练Transformer模型展现出显著的上下文学习能力，使其能够在无需参数更新的情况下从演示中适应新任务。然而，理论研究通常依赖于简化的架构（如省略MLP）、数据模型（如具有各向同性输入的线性回归）和单源训练，这限制了它们与现实设置的相关性。

### 目的

研究具有非线性MLP头部的预训练Transformer在从多个具有异构输入、任务和噪声分布的数据源获取的非线性任务上的ICL能力，分析数据混合效应，并提供关于架构和数据在ICL中作用的可操作见解。

### 方法

分析一个包含两层的MLP模型，其中第一层通过单次梯度步骤训练，第二层完全优化。在高维渐近条件下，利用高斯普适性和正交多项式理论，证明这类模型的ICL误差等价于结构化多项式预测器。在各种激活函数、模型大小和数据分布上进行经验验证，并在多语言情感分析的真实场景中进行实验。

### 主要发现

非线性MLP能显著提升ICL性能，特别是在非线性任务上；高质量数据源具有低噪声和结构化协方差的关键特性；只有当任务协方差具有足够结构时，特征学习才会出现；这些发现在各种激活函数、模型大小和数据分布上得到了经验验证；多语言情感分析实验表明这些发现可以扩展到真实世界案例。

### 结论

这项工作推进了Transformer中ICL的理论基础，并提供了关于架构和数据在ICL中作用的可操作见解，特别是在非线性任务和异构数据源设置下。

### 翻译

预训练的Transformer模型展现出显著的上下文学习（ICL）能力，使它们能够在无需参数更新的情况下从演示中适应新任务。然而，理论研究通常依赖于简化的架构（例如，省略MLP）、数据模型（例如，具有各向同性输入的线性回归）和单源训练，限制了它们与现实设置的相关性。在这项工作中，我们研究了具有非线性MLP头部的预训练Transformer的ICL能力，这些模型在从多个具有异构输入、任务和噪声分布的数据源中获取的非线性任务上表现。我们分析了一个模型，其中MLP包含两层，第一层通过单次梯度步骤训练，第二层完全优化。在高维渐近条件下，我们证明这类模型的ICL误差等价于结构化多项式预测器，利用了高斯普适性和正交多项式理论的结果。这种等价性表明非线性MLP相比线性基线能显著提升ICL性能，特别是在非线性任务上。它还使数据分析混合效应的精确分析成为可能：我们确定了高质量数据源的关键特性（低噪声、结构化协方差），并表明只有当任务协方差具有足够结构时，特征学习才会出现。这些发现在各种激活函数、模型大小和数据分布上得到了经验验证。最后，我们在一个涉及多语言情感分析的真实场景中进行了实验，每种语言被视为不同的数据源。这个案例的实验结果说明了我们的发现如何扩展到真实世界案例。总体而言，我们的工作推进了Transformer中ICL的理论基础，并提供了关于架构和数据在ICL中作用的可操作见解。


### 论文摘要

Pretrained Transformers demonstrate remarkable in-context learning (ICL) capabilities, enabling them to adapt to new tasks from demonstrations without parameter updates. However, theoretical studies often rely on simplified architectures (e.g., omitting MLPs), data models (e.g., linear regression with isotropic inputs), and single-source training, limiting their relevance to realistic settings. In this work, we study ICL in pretrained Transformers with nonlinear MLP heads on nonlinear tasks drawn from multiple data sources with heterogeneous input, task, and noise distributions. We analyze a model where the MLP comprises two layers, with the first layer trained via a single gradient step and the second layer fully optimized. Under high-dimensional asymptotics, we prove that such models are equivalent in ICL error to structured polynomial predictors, leveraging results from the theory of Gaussian universality and orthogonal polynomials. This equivalence reveals that nonlinear MLPs meaningfully enhance ICL performance, particularly on nonlinear tasks, compared to linear baselines. It also enables a precise analysis of data mixing effects: we identify key properties of high-quality data sources (low noise, structured covariances) and show that feature learning emerges only when the task covariance exhibits sufficient structure. These results are validated empirically across various activation functions, model sizes, and data distributions. Finally, we experiment with a real-world scenario involving multilingual sentiment analysis where each language is treated as a different source. Our experimental results for this case exemplify how our findings extend to real-world cases. Overall, our work advances the theoretical foundations of ICL in Transformers and provides actionable insight into the role of architecture and data in ICL.

---

## 21. Lost in Phonation: Voice Quality Variation as an Evaluation Dimension for Speech Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.25577v1](http://arxiv.org/abs/2510.25577v1)

**作者:** Harm Lameris, Shree Harsha Bokkahalli Satish, Joakim Gustafson, Éva Székely

**发布时间:** 2025-10-29

**备注:** 8 pages, 3 figures, 4 tables, submitted to LREC 2026

### GPT解析

### 总结

本研究探讨了语音基础模型(SFMs)如何处理语音中的副语言变化，特别是音质(如嘶哑和气声)对模型行为的影响。作者通过开放式生成任务和语音情感识别评估模型对不同音质输入的一致性反应，并引入了新的平行数据集来评估SFMs对音质的敏感性。

### 背景

语音基础模型(SFMs)可直接处理原始音频中的口语，绕过文本表示，因此能接触到语音信号中的副语言变化。音质是副语言变化中未被充分探索的维度，包括嘶哑和气声等发声类型，这些类型影响听众对情感状态、立场和社会意义的推断。现有语音理解基准主要依赖多项选择题格式，容易失败，难以捕捉副语言特征对模型行为的微妙影响。

### 目的

通过开放式生成任务和语音情感识别探测SFMs，评估模型行为在不同音质输入下是否一致；引入包含音质合成修改的平行数据集，评估SFMs对嘶哑和气声的反应；提供对SFMs对这些特定语音感知非词汇方面敏感性的首次检验。

### 方法

使用开放式生成任务探测SFMs；通过语音情感识别评估模型反应；引入包含音质合成修改的平行数据集；评估SFMs对嘶哑和气声的反应。

### 主要发现

摘要中未明确提及具体实验结果，主要介绍了研究方法和数据集的构建。

### 结论

摘要中未明确提及具体结论，主要介绍了研究的创新点和贡献，即首次检验了SFMs对语音中特定非词汇方面的敏感性。

### 翻译

语音基础模型(SFMs)的最新进展使得可以直接处理原始音频中的口语，绕过中间的文本表示。这种能力使SFMs能够接触到输入语音信号中嵌入的丰富副语言变化，并可能对这些变化做出响应。副语言变化的一个未被充分探索的维度是音质，包括嘶哑和气声等发声类型。这些发声类型已知会影响听众如何推断语音中的情感状态、立场和社会意义。现有的语音理解基准测试主要依赖多项选择题问答(MCQA)格式，这些格式容易失败，因此在捕捉副语言特征如何微妙影响模型行为方面并不可靠。在本文中，我们通过开放式生成任务和语音情感识别来探测SFMs，评估模型行为在不同音质输入下是否一致。我们引入了一个新的平行数据集，其中包含对音质的合成修改，旨在评估SFMs对嘶哑和气声的反应。我们的工作首次检验了SFMs对这些特定语音感知非词汇方面的敏感性。


### 论文摘要

Recent advances in speech foundation models (SFMs) have enabled the direct processing of spoken language from raw audio, bypassing intermediate textual representations. This capability allows SFMs to be exposed to, and potentially respond to, rich paralinguistic variations embedded in the input speech signal. One under-explored dimension of paralinguistic variation is voice quality, encompassing phonation types such as creaky and breathy voice. These phonation types are known to influence how listeners infer affective state, stance and social meaning in speech. Existing benchmarks for speech understanding largely rely on multiple-choice question answering (MCQA) formats, which are prone to failure and therefore unreliable in capturing the nuanced ways paralinguistic features influence model behaviour. In this paper, we probe SFMs through open-ended generation tasks and speech emotion recognition, evaluating whether model behaviours are consistent across different phonation inputs. We introduce a new parallel dataset featuring synthesized modifications to voice quality, designed to evaluate SFM responses to creaky and breathy voice. Our work provides the first examination of SFM sensitivity to these particular non-lexical aspects of speech perception.

---

## 22. Leveraging an Atmospheric Foundational Model for Subregional Sea Surface Temperature Forecasting

**论文链接:** [http://arxiv.org/abs/2510.25563v1](http://arxiv.org/abs/2510.25563v1)

**作者:** Víctor Medina, Giovanny A. Cuervo-Londoño, Javier Sánchez

**发布时间:** 2025-10-29

**备注:** 18 pages, 9 figures

### GPT解析

### 总结

本研究将大气预报深度学习模型Aurora适应化用于海洋预测，通过微调实现了高精度的海表温度预测，同时降低了计算成本，为数据驱动的海洋预报提供了新方法。

### 背景

准确的海洋变量预测对理解气候变化、管理海洋资源和优化海洋活动至关重要。传统海洋预报依赖数值模型，但面临计算成本高和可扩展性有限的问题。

### 目的

将Aurora深度学习模型（原为大气预报设计）适应化，用于预测加那利上升流系统的海表温度(SST)，探索深度学习在海洋预报中的应用潜力。

### 方法

使用高分辨率海洋再分析数据对模型进行分阶段微调，结合纬度加权误差指标，优化超参数以实现高效学习，减少计算需求。

### 主要发现

模型实现了0.119K的低均方根误差，异常相关系数高达0.997，成功重现大尺度SST结构，但在捕捉沿海地区精细细节方面存在挑战。

### 结论

研究证明使用在不同领域预训练的深度学习模型进行海洋应用具有可行性，未来改进方向包括整合更多海洋变量、提高空间分辨率和探索物理信息神经网络。

### 翻译

准确的海洋变量预测对于理解气候变化、管理海洋资源和优化海洋活动至关重要。传统的海洋预报依赖于数值模型；然而，这些方法在计算成本和可扩展性方面存在局限性。在本研究中，我们将Aurora（一种最初为大气预报设计的基础深度学习模型）适应化，用于预测加那利上升流系统的海表温度(SST)。通过使用高分辨率的海洋再分析数据对模型进行微调，我们展示了其捕捉复杂时空模式的能力，同时减少了计算需求。我们的方法包括分阶段微调过程，结合纬度加权误差指标，并优化超参数以实现高效学习。实验结果显示，模型实现了0.119K的低均方根误差，并保持高的异常相关系数(ACC≈0.997)。模型成功重现了大尺度SST结构，但在捕捉沿海地区更精细的细节方面面临挑战。这项工作通过证明使用在不同领域预训练的深度学习模型进行海洋应用的可行性，为数据驱动的海洋预报领域做出了贡献。未来改进包括整合额外的海洋变量、提高空间分辨率，以及探索物理信息神经网络以增强可解释性和理解。这些进步可以改善气候建模和海洋预测精度，支持环境和经济部门的决策制定。


### 论文摘要

The accurate prediction of oceanographic variables is crucial for understanding climate change, managing marine resources, and optimizing maritime activities. Traditional ocean forecasting relies on numerical models; however, these approaches face limitations in terms of computational cost and scalability. In this study, we adapt Aurora, a foundational deep learning model originally designed for atmospheric forecasting, to predict sea surface temperature (SST) in the Canary Upwelling System. By fine-tuning this model with high-resolution oceanographic reanalysis data, we demonstrate its ability to capture complex spatiotemporal patterns while reducing computational demands. Our methodology involves a staged fine-tuning process, incorporating latitude-weighted error metrics and optimizing hyperparameters for efficient learning. The experimental results show that the model achieves a low RMSE of 0.119K, maintaining high anomaly correlation coefficients (ACC $\approx 0.997$). The model successfully reproduces large-scale SST structures but faces challenges in capturing finer details in coastal regions. This work contributes to the field of data-driven ocean forecasting by demonstrating the feasibility of using deep learning models pre-trained in different domains for oceanic applications. Future improvements include integrating additional oceanographic variables, increasing spatial resolution, and exploring physics-informed neural networks to enhance interpretability and understanding. These advancements can improve climate modeling and ocean prediction accuracy, supporting decision-making in environmental and economic sectors.

---

## 23. FaCT: Faithful Concept Traces for Explaining Neural Network Decisions

**论文链接:** [http://arxiv.org/abs/2510.25512v1](http://arxiv.org/abs/2510.25512v1)

**作者:** Amin Parchami-Araghi, Sukrut Rao, Jonas Fischer, Bernt Schiele

**发布时间:** 2025-10-29

**备注:** Accepted to NeurIPS 2025; Code is available at  https://github.com/m-parchami/FaCT

### GPT解析

### 总结

该研究提出了一种具有模型内在机制概念解释的新模型，强调概念化解释的忠实性，并引入了概念一致性度量C²-Score来评估概念化方法。

### 背景

深度网络在各种任务上表现出色，但要全面理解其工作机制仍然是一个挑战。现有的后验概念化方法在解释模型时并不总是忠实于模型本身，且对模型学习的概念做出了严格的假设。

### 目的

强调概念化解释的忠实性，提出一种具有模型内在机制概念解释的新模型，并开发一种新的概念一致性度量标准来评估概念化方法。

### 方法

提出的新模型的概念跨类别共享，可以从任何层追踪其对logit的贡献和输入可视化。利用基础模型提出了一种新的概念一致性度量标准C²-Score，用于评估概念化方法。

### 主要发现

与先前的工作相比，提出的模型在定量上更加一致，用户发现其概念更具可解释性，同时保持了有竞争力的ImageNet性能。

### 结论

通过强调概念化解释的忠实性和提出新的度量标准，该研究为理解深度网络的工作机制提供了更有效的方法。

### 翻译

深度网络在广泛任务中表现出色，但要全面理解其工作机制仍然是一个关键挑战。许多后验概念化方法被引入以解释其工作原理，但它们并不总是忠实于模型。此外，它们对模型学习的概念做出了严格的假设，如类别特异性、小的空间范围或符合人类预期。在本工作中，我们强调此类概念化解释的忠实性，并提出了一种具有模型内在机制概念解释的新模型。我们的概念跨类别共享，并且可以从任何层追踪其对logit的贡献及其输入可视化。我们还利用基础模型提出了一个新的概念一致性度量标准C²-Score，可用于评估概念化方法。我们表明，与先前的工作相比，我们的概念在定量上更加一致，用户发现我们的概念更具可解释性，同时保持了有竞争力的ImageNet性能。


### 论文摘要

Deep networks have shown remarkable performance across a wide range of tasks, yet getting a global concept-level understanding of how they function remains a key challenge. Many post-hoc concept-based approaches have been introduced to understand their workings, yet they are not always faithful to the model. Further, they make restrictive assumptions on the concepts a model learns, such as class-specificity, small spatial extent, or alignment to human expectations. In this work, we put emphasis on the faithfulness of such concept-based explanations and propose a new model with model-inherent mechanistic concept-explanations. Our concepts are shared across classes and, from any layer, their contribution to the logit and their input-visualization can be faithfully traced. We also leverage foundation models to propose a new concept-consistency metric, C$^2$-Score, that can be used to evaluate concept-based methods. We show that, compared to prior work, our concepts are quantitatively more consistent and users find our concepts to be more interpretable, all while retaining competitive ImageNet performance.

---

## 24. TempoPFN: Synthetic Pre-training of Linear RNNs for Zero-shot Time Series Forecasting

**论文链接:** [http://arxiv.org/abs/2510.25502v1](http://arxiv.org/abs/2510.25502v1)

**作者:** Vladyslav Moroshan, Julien Siems, Arber Zela, Timur Carstensen, Frank Hutter

**发布时间:** 2025-10-29

**备注:** 30 pages, 18 figures, 13 tables

### GPT解析

### 总结

本文提出了TempoPFN，一种基于线性循环神经网络的单变量时间序列基础模型，仅在合成数据上预训练，解决了零样本时间序列预测中的长期预测效率和可重现性问题。

### 背景

零样本时间序列预测的基础模型面临长期预测效率低和可重现性差的挑战，现有的仅使用合成数据的方法在具有挑战性的基准测试中表现不佳。

### 目的

开发一种高效且可重现的时间序列基础模型，用于零样本预测，超越现有仅使用合成数据的方法的性能。

### 方法

TempoPFN采用GatedDeltaProduct架构和状态编织技术，实现跨序列长度的完全并行化训练，消除对窗口化或摘要技术的需求。综合合成数据管道统一了随机微分方程、高斯过程和音频合成等多种生成器，并引入新颖的数据增强技术。

### 主要发现

在Gift-Eval基准测试中，TempoPFN达到顶尖竞争性能，超越所有现有的仅使用合成数据的方法，并超过绝大多数在真实数据上训练的模型。同时，通过完全并行化的训练和推理，比现有基线更高效。

### 结论

开源完整的数据生成管道和训练代码，为未来研究提供可重现的基础，推动零样本时间序列预测领域的发展。

### 翻译

零样本时间序列预测的基础模型面临长期预测效率低和可重现性差的挑战，现有的仅使用合成数据的方法在具有挑战性的基准测试中表现不佳。本文提出了TempoPFN，一种基于线性循环神经网络的单变量时间序列基础模型，该模型仅在合成数据上进行预训练。该模型采用GatedDeltaProduct架构和状态编织技术，实现跨序列长度的完全并行化训练，消除对窗口化或摘要技术的需求，同时保持强大的时间状态跟踪能力。我们的综合合成数据管道统一了多种生成器，包括随机微分方程、高斯过程和音频合成，并引入了新颖的数据增强技术。在Gift-Eval基准的零样本评估中，TempoPFN达到了顶尖的竞争性能，超越了所有现有的仅使用合成数据的方法，并超过了绝大多数在真实数据上训练的模型，同时通过利用完全并行化的训练和推理，比现有基线更高效。我们开源了完整的数据生成管道和训练代码，为未来研究提供可重现的基础。


### 论文摘要

Foundation models for zero-shot time series forecasting face challenges in efficient long-horizon prediction and reproducibility, with existing synthetic-only approaches underperforming on challenging benchmarks. This paper presents TempoPFN, a univariate time series foundation model based on linear Recurrent Neural Networks (RNNs) pre-trained exclusively on synthetic data. The model uses a GatedDeltaProduct architecture with state-weaving for fully parallelizable training across sequence lengths, eliminating the need for windowing or summarization techniques while maintaining robust temporal state-tracking. Our comprehensive synthetic data pipeline unifies diverse generators, including stochastic differential equations, Gaussian processes, and audio synthesis, with novel augmentations. In zero-shot evaluations on the Gift-Eval benchmark, TempoPFN achieves top-tier competitive performance, outperforming all existing synthetic-only approaches and surpassing the vast majority of models trained on real-world data, while being more efficient than existing baselines by leveraging fully parallelizable training and inference. We open-source our complete data generation pipeline and training code, providing a reproducible foundation for future research.

---

## 25. Position: Biology is the Challenge Physics-Informed ML Needs to Evolve

**论文链接:** [http://arxiv.org/abs/2510.25368v1](http://arxiv.org/abs/2510.25368v1)

**作者:** Julien Martinelli

**发布时间:** 2025-10-29

### GPT解析

### 总结

本文提出将物理信息机器学习(PIML)扩展到生物学领域，称为生物学信息机器学习(BIML)，以应对生物建模的独特挑战。

### 背景

物理信息机器学习已成功将机理理解整合到机器学习中，特别是在受已知物理定律支配的领域，这一成功促使人们尝试将其应用于生物学领域。

### 目的

将PIML的原则性方法扩展到生物学，创建BIML框架，使其能够适应生物学的实际现实，而非视为障碍。

### 方法

重新调整PIML的方法，使其能够在更软性、概率形式的先验知识下运行，提出四大基础支柱：不确定性量化、上下文化、受限潜在结构推断和可扩展性。

### 主要发现

生物建模面临的挑战（多方面且不确定的先验知识、异构且嘈杂的数据、部分可观察性以及复杂的高维网络）不应被视为PIML的障碍，而应被视为其进化的催化剂。

### 结论

基础模型和大语言模型将成为关键推动因素，将人类专业知识与计算建模结合，构建BIML生态系统，并将PIML启发的创新引向具有高度科学和社会相关性的挑战。

### 翻译

物理信息机器学习(PIML)已成功将机理理解整合到机器学习中，特别是在受已知物理定律支配的领域。这一成功促使人们尝试将PIML应用于生物学，这是一个充满动态系统但受不同约束塑造的领域。然而，生物建模面临独特挑战：多方面且不确定的先验知识、异构且嘈杂的数据、部分可观察性以及复杂的高维网络。在这篇立场论文中，我们认为这些挑战不应被视为PIML的障碍，而应是其进化的催化剂。我们提出了生物学信息机器学习(BIML)：PIML的原则性扩展，保留了其结构基础，同时适应生物学的实际现实。BIML不是取代PIML，而是重新调整其方法，使其能够在更软性、概率形式的先验知识下运行。我们概述了四个基础支柱作为这一转变的路线图：不确定性量化、上下文化、受限潜在结构推断和可扩展性。基础模型和大语言模型将成为关键推动因素，将人类专业知识与计算建模结合起来。最后，我们提出具体建议，以构建BIML生态系统，并将PIML启发的创新引向具有高度科学和社会相关性的挑战。


### 论文摘要

Physics-Informed Machine Learning (PIML) has successfully integrated mechanistic understanding into machine learning, particularly in domains governed by well-known physical laws. This success has motivated efforts to apply PIML to biology, a field rich in dynamical systems but shaped by different constraints. Biological modeling, however, presents unique challenges: multi-faceted and uncertain prior knowledge, heterogeneous and noisy data, partial observability, and complex, high-dimensional networks. In this position paper, we argue that these challenges should not be seen as obstacles to PIML, but as catalysts for its evolution. We propose Biology-Informed Machine Learning (BIML): a principled extension of PIML that retains its structural grounding while adapting to the practical realities of biology. Rather than replacing PIML, BIML retools its methods to operate under softer, probabilistic forms of prior knowledge. We outline four foundational pillars as a roadmap for this transition: uncertainty quantification, contextualization, constrained latent structure inference, and scalability. Foundation Models and Large Language Models will be key enablers, bridging human expertise with computational modeling. We conclude with concrete recommendations to build the BIML ecosystem and channel PIML-inspired innovation toward challenges of high scientific and societal relevance.

---

## 26. 3D CT-Based Coronary Calcium Assessment: A Feature-Driven Machine Learning Framework

**论文链接:** [http://arxiv.org/abs/2510.25347v1](http://arxiv.org/abs/2510.25347v1)

**作者:** Ayman Abaid, Gianpiero Guidone, Sara Alsubai, Foziyah Alquahtani, Talha Iqbal, Ruth Sharif, Hesham Elzomor, Emiliano Bianchini, Naeif Almagal, Michael G. Madden, Faisal Sharif, Ihsan Ullah

**发布时间:** 2025-10-29

**备注:** 11 pages, 2 Figures, MICCAI AMAI 2025 workshop, to be published in  Volume 16206 of the Lecture Notes in Computer Science series

### GPT解析

### 总结

本研究提出了一种基于放射组学的流程，利用伪标记生成训练标签，解决了非对比冠状动脉计算机断层血管造影(CCTA)扫描中冠状动脉钙化(CAC)评分标记数据有限的问题。

### 背景

冠状动脉钙化(CAC)评分在冠状动脉疾病(CAD)的早期检测和风险分层中起着关键作用。非对比冠状动脉计算机断层血管造影(CCTA)扫描在临床中常用于早期钙化检测。

### 目的

开发一种基于放射组学的流程，利用伪标记生成训练标签，避免需要专家定义的分割，并探索预训练基础模型在特征提取中的应用。

### 方法

提出基于放射组学的流程，利用伪标记生成训练标签；探索使用预训练基础模型(CT-FM和RadImageNet)提取图像特征并与传统分类器结合；比较深度学习特征与放射组学特征性能；在包含182名患者的临床CCTA数据集上评估，将患者分为零钙化评分组和非零钙化评分组；研究在非对比数据集与对比+非对比数据集上训练的影响。

### 主要发现

基于放射组学的模型显著优于来自基础模型的CNN嵌入，达到84%的准确率(p<0.05)，尽管没有专家标注可用。

### 结论

基于放射组学的方法在冠状动脉钙化检测中表现出色，即使在没有专家标注的情况下也能达到高准确率。

### 翻译

冠状动脉钙化(CAC)评分在冠状动脉疾病(CAD)的早期检测和风险分层中起着关键作用。在本研究中，我们关注非对比冠状动脉计算机断层血管造影(CCTA)扫描，这些扫描在临床中常用于早期钙化检测。为解决标记数据有限这一挑战，我们提出了一种基于放射组学的流程，利用伪标记生成训练标签，从而消除对专家定义分割的需求。此外，我们探索了使用预训练基础模型(特别是CT-FM和RadImageNet)提取图像特征，然后与传统分类器一起使用。我们将这些深度学习特征与放射组学特征的性能进行了比较。评估在包含182名患者的临床CCTA数据集上进行，个体被分为两组：零钙化评分组与非零钙化评分组。我们进一步研究了在非对比数据集与对比+非对比数据集上训练的影响，测试仅在非对比扫描上进行。结果表明，尽管没有专家标注可用，但基于放射组学的模型显著优于来自基础模型的CNN嵌入(达到84%的准确率和p<0.05)。


### 论文摘要

Coronary artery calcium (CAC) scoring plays a crucial role in the early detection and risk stratification of coronary artery disease (CAD). In this study, we focus on non-contrast coronary computed tomography angiography (CCTA) scans, which are commonly used for early calcification detection in clinical settings. To address the challenge of limited annotated data, we propose a radiomics-based pipeline that leverages pseudo-labeling to generate training labels, thereby eliminating the need for expert-defined segmentations. Additionally, we explore the use of pretrained foundation models, specifically CT-FM and RadImageNet, to extract image features, which are then used with traditional classifiers. We compare the performance of these deep learning features with that of radiomics features. Evaluation is conducted on a clinical CCTA dataset comprising 182 patients, where individuals are classified into two groups: zero versus non-zero calcium scores. We further investigate the impact of training on non-contrast datasets versus combined contrast and non-contrast datasets, with testing performed only on non contrast scans. Results show that radiomics-based models significantly outperform CNN-derived embeddings from foundation models (achieving 84% accuracy and p<0.05), despite the unavailability of expert annotations.

---

## 27. GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning

**论文链接:** [http://arxiv.org/abs/2510.25320v1](http://arxiv.org/abs/2510.25320v1)

**作者:** Jiaqi Wu, Qinlao Zhao, Zefeng Chen, Kai Qin, Yifei Zhao, Xueqian Wang, Yuhang Yao

**发布时间:** 2025-10-29

### GPT解析

### 总结

该论文提出了基于图的代理规划(GAP)框架，通过建模任务间依赖关系实现工具的并行和顺序执行，解决了现有顺序推理范式的效率问题。

### 背景

大型语言模型驱动的自主代理在工具操作方面显示出强大能力，但现有范式(如ReAct)依赖顺序推理和执行，无法利用独立子任务之间的内在并行性。

### 目的

解决顺序推理瓶颈导致的工具利用效率低下和多步推理场景中表现不佳的问题。

### 方法

提出基于图的代理规划(GAP)框架，训练基础模型将复杂任务分解为依赖感知的子任务图，自主确定工具的并行或顺序执行方式；采用两阶段训练策略：监督微调(SFT)和强化学习(RL)。

### 主要发现

GAP在MHQA数据集上显著优于传统ReAct基线，特别是在多步检索任务上，同时通过智能并行化实现了工具调用效率的显著提升。

### 结论

依赖感知的任务编排在执行效率和任务准确性方面都取得了实质性改进。

### 翻译

由大型语言模型(LLM)驱动的自主代理在工具操作方面展现出解决复杂任务的强大能力。然而，现有的ReAct等范式依赖顺序推理和执行，无法利用独立子任务之间的内在并行性。这种顺序瓶颈导致工具利用效率低下，以及在多步推理场景中表现不佳。我们引入了基于图的代理规划(GAP)，这是一个新框架，通过基于图的规划明确建模任务间依赖关系，实现自适应并行和顺序工具执行。我们的方法训练基础模型将复杂任务分解为依赖感知的子任务图，自主确定哪些工具可以并行执行，哪些必须遵循顺序依赖。这种依赖感知的编排在执行效率和任务准确性方面都取得了实质性改进。为了训练GAP，我们从多跳问答(MHQA)基准构建了高质量的基于图的规划轨迹数据集。我们采用两阶段训练策略：首先在整理的数据集上进行监督微调(SFT)，然后在基于正确性奖励函数的强化学习(RL)阶段，在战略采样的查询上进行训练。在MHQA数据集上的实验结果表明，GAP显著优于传统的ReAct基线，特别是在多步检索任务上，同时通过智能并行化实现了工具调用效率的显著提升。项目页面可在以下网址访问：https://github.com/WJQ7777/Graph-Agent-Planning。


### 论文摘要

Autonomous agents powered by large language models (LLMs) have shown impressive capabilities in tool manipulation for complex task-solving. However, existing paradigms such as ReAct rely on sequential reasoning and execution, failing to exploit the inherent parallelism among independent sub-tasks. This sequential bottleneck leads to inefficient tool utilization and suboptimal performance in multi-step reasoning scenarios. We introduce Graph-based Agent Planning (GAP), a novel framework that explicitly models inter-task dependencies through graph-based planning to enable adaptive parallel and serial tool execution. Our approach trains agent foundation models to decompose complex tasks into dependency-aware sub-task graphs, autonomously determining which tools can be executed in parallel and which must follow sequential dependencies. This dependency-aware orchestration achieves substantial improvements in both execution efficiency and task accuracy. To train GAP, we construct a high-quality dataset of graph-based planning traces derived from the Multi-Hop Question Answering (MHQA) benchmark. We employ a two-stage training strategy: supervised fine-tuning (SFT) on the curated dataset, followed by reinforcement learning (RL) with a correctness-based reward function on strategically sampled queries where tool-based reasoning provides maximum value. Experimental results on MHQA datasets demonstrate that GAP significantly outperforms traditional ReAct baselines, particularly on multi-step retrieval tasks, while achieving dramatic improvements in tool invocation efficiency through intelligent parallelization. The project page is available at: https://github.com/WJQ7777/Graph-Agent-Planning.

---

## 28. RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models

**论文链接:** [http://arxiv.org/abs/2510.25257v1](http://arxiv.org/abs/2510.25257v1)

**作者:** Zijun Liao, Yian Zhao, Xin Shan, Yu Yan, Chang Liu, Lei Lu, Xiangyang Ji, Jie Chen

**发布时间:** 2025-10-29

### GPT解析

### 总结

本文提出了一种成本效益高且适应性强的蒸馏框架，利用视觉基础模型(VFMs)增强轻量级目标检测器，解决了高速度推理与特征表示能力之间的矛盾。

### 背景

实时目标检测通过精心设计的架构和优化策略取得了显著进展，但轻量级网络设计追求高速推理往往导致特征表示能力下降，阻碍了性能提升和实际设备部署。

### 目的

提出一种利用视觉基础模型(VFMs)能力增强轻量级目标检测器的成本效益高且适应性强的蒸馏框架，解决VFM与资源受限检测器之间架构和学习目标差异导致的语义传输挑战。

### 方法

引入深度语义注入器(DSI)模块促进VFM高级表示与检测器深层集成；设计基于梯度的自适应调制(GAM)策略，根据梯度范数比率动态调整语义传输强度。

### 主要发现

该方法在不增加部署和推理开销的情况下，为各种基于DETR的模型带来显著且一致的性能提升；新模型RT-DETRv4在COCO上取得最先进结果，在273/169/124/78 FPS速度下分别达到49.7/53.5/55.4/57.0的AP分数。

### 结论

该方法强调了其在实时检测中的实际效用，为实时目标检测提供了有效解决方案。

### 翻译

实时目标检测通过精心设计的架构和优化策略取得了实质性进展。然而，通过轻量级网络设计追求高速推理通常会导致特征表示能力下降，这阻碍了性能的进一步改进和实际设备部署。在本文中，我们提出了一种具有成本效益且高度适应性的蒸馏框架，利用视觉基础模型(VFMs)的快速发展能力来增强轻量级目标检测器。鉴于VFM与资源受限检测器之间存在显著的架构和学习目标差异，实现稳定且任务对齐的语义传输具有挑战性。为此，一方面，我们引入了深度语义注入器(DSI)模块，促进VFM的高级表示与检测器深层层的集成；另一方面，我们设计了基于梯度的自适应调制(GAM)策略，根据梯度范数比率动态调整语义传输强度。在不增加部署和推理开销的情况下，我们的方法在各种基于DETR的模型上轻松实现了显著且一致的性能提升，凸显了其在实时检测中的实际效用。我们的新模型系列RT-DETRv4在COCO上取得了最先进的结果，在相应速度为273/169/124/78 FPS时分别达到49.7/53.5/55.4/57.0的AP分数。


### 论文摘要

Real-time object detection has achieved substantial progress through meticulously designed architectures and optimization strategies. However, the pursuit of high-speed inference via lightweight network designs often leads to degraded feature representation, which hinders further performance improvements and practical on-device deployment. In this paper, we propose a cost-effective and highly adaptable distillation framework that harnesses the rapidly evolving capabilities of Vision Foundation Models (VFMs) to enhance lightweight object detectors. Given the significant architectural and learning objective disparities between VFMs and resource-constrained detectors, achieving stable and task-aligned semantic transfer is challenging. To address this, on one hand, we introduce a Deep Semantic Injector (DSI) module that facilitates the integration of high-level representations from VFMs into the deep layers of the detector. On the other hand, we devise a Gradient-guided Adaptive Modulation (GAM) strategy, which dynamically adjusts the intensity of semantic transfer based on gradient norm ratios. Without increasing deployment and inference overhead, our approach painlessly delivers striking and consistent performance gains across diverse DETR-based models, underscoring its practical utility for real-time detection. Our new model family, RT-DETRv4, achieves state-of-the-art results on COCO, attaining AP scores of 49.7/53.5/55.4/57.0 at corresponding speeds of 273/169/124/78 FPS.

---

## 29. Test-Time Adaptive Object Detection with Foundation Model

**论文链接:** [http://arxiv.org/abs/2510.25175v1](http://arxiv.org/abs/2510.25175v1)

**作者:** Yingjie Gao, Yanan Zhang, Zhi Cai, Di Huang

**发布时间:** 2025-10-29

**备注:** Accepted by NeurIPS 2025

### GPT解析

### 总结

该研究提出了一种基于基础模型的测试时自适应目标检测方法，消除了对源数据的完全需求，克服了传统封闭集限制，实现了高效的跨域和跨类别适应。

### 背景

测试时自适应目标检测近年来受到越来越多的关注，因为它在在线领域适应方面具有独特优势，更接近实际应用场景。然而，现有方法严重依赖于源域统计特征，并假设源域和目标域共享相同的类别空间。

### 目的

提出第一个基于基础模型的测试时自适应目标检测方法，消除对源数据的完全需求，克服传统封闭集限制，实现任意跨域和跨类别的目标数据适应。

### 方法

设计了一个多模态提示的Mean-Teacher框架，结合文本和视觉提示调整，以参数高效的方式适应语言和视觉表示空间；提出了针对视觉提示的测试时预热策略；维护实例动态内存模块存储高质量伪标签；并提出了内存增强和内存幻觉两种新策略。

### 主要发现

在跨损坏和跨数据集基准上的广泛实验表明，该方法持续优于之前的最先进方法，能够适应任意跨域和跨类别的目标数据。

### 结论

基于基础模型的方法在测试时自适应目标检测方面表现出色，代码已在GitHub上公开。

### 翻译

近年来，测试时自适应目标检测因其在线领域适应中的独特优势而受到越来越多的关注，这更接近实际应用场景。然而，现有方法严重依赖于源域统计特征，同时做出源域和目标域共享相同类别空间的强假设。本文提出了第一个基于基础模型的测试时自适应目标检测方法，完全消除了对源数据的需求并克服了传统封闭集限制。具体而言，我们设计了一个多模态提示的Mean-Teacher框架，用于视觉-语言检测器驱动的测试时适应，结合文本和视觉提示调整，以参数高效的方式在测试数据上适应语言和视觉表示空间。相应地，我们提出了针对视觉提示定制的测试时预热策略，以有效保持视觉分支的表示能力。此外，为保证每个测试批次的高质量伪标签，我们维护了一个存储来自先前测试样本的高质量伪标签的实例动态内存模块，并提出了两种新策略——内存增强和内存幻觉，分别利用IDM的高质量实例来增强原始预测和对没有可用伪标签的图像进行幻觉处理。在跨损坏和跨数据集基准上的广泛实验表明，我们的方法持续优于之前的最先进方法，并能适应任意跨域和跨类别的目标数据。代码可在https://github.com/gaoyingjay/ttaod_foundation获取。


### 论文摘要

In recent years, test-time adaptive object detection has attracted increasing attention due to its unique advantages in online domain adaptation, which aligns more closely with real-world application scenarios. However, existing approaches heavily rely on source-derived statistical characteristics while making the strong assumption that the source and target domains share an identical category space. In this paper, we propose the first foundation model-powered test-time adaptive object detection method that eliminates the need for source data entirely and overcomes traditional closed-set limitations. Specifically, we design a Multi-modal Prompt-based Mean-Teacher framework for vision-language detector-driven test-time adaptation, which incorporates text and visual prompt tuning to adapt both language and vision representation spaces on the test data in a parameter-efficient manner. Correspondingly, we propose a Test-time Warm-start strategy tailored for the visual prompts to effectively preserve the representation capability of the vision branch. Furthermore, to guarantee high-quality pseudo-labels in every test batch, we maintain an Instance Dynamic Memory (IDM) module that stores high-quality pseudo-labels from previous test samples, and propose two novel strategies-Memory Enhancement and Memory Hallucination-to leverage IDM's high-quality instances for enhancing original predictions and hallucinating images without available pseudo-labels, respectively. Extensive experiments on cross-corruption and cross-dataset benchmarks demonstrate that our method consistently outperforms previous state-of-the-art methods, and can adapt to arbitrary cross-domain and cross-category target data. Code is available at https://github.com/gaoyingjay/ttaod_foundation.

---

## 30. TabMGP: Martingale Posterior with TabPFN

**论文链接:** [http://arxiv.org/abs/2510.25154v1](http://arxiv.org/abs/2510.25154v1)

**作者:** Kenyon Ng, Edwin Fong, David T. Frazier, Jeremias Knoblauch, Susan Wei

**发布时间:** 2025-10-29

**备注:** 11 pages (+3 reference, +22 appendix). Extra plots in  https://drive.google.com/drive/folders/1ct_effOoTEGpiWUf0_1xI3VqLWHtJY16

### GPT解析

### 总结

本研究提出了一种基于基础转换器的鞅后验方法（TabMGP），用于解决贝叶斯推断中的不确定性量化问题。

### 背景

贝叶斯推断在不确定性量化方面具有优势，但面临先验设定、似然误设和计算负担等挑战。

### 目的

开发一种有效的预测规则，用于构建高质量的鞅后验方法，提高不确定性量化的准确性。

### 方法

利用基础转换器（特别是TabPFN，一种表格数据领域的最先进模型）构建鞅后验（TabMGP），通过自回归生成模拟前向数据生成过程。

### 主要发现

TabMGP产生的可信集具有接近名义覆盖范围的性能，并且通常优于现有的MGP结构和标准贝叶斯方法。

### 结论

基于基础转换器的鞅后验方法（TabMGP）在表格数据的不确定性量化方面表现优异，为贝叶斯推断提供了有效替代方案。

### 翻译

贝叶斯推断提供了有原则的不确定性量化，但常常受到先验设定、似然误设和计算负担的限制。鞅后验（MGP，Fong等人，2023年）提供了一种替代方案，用预测规则（即一步前向预测分布序列）替代先验-似然设定，用于前向数据生成。MGP的有效性取决于预测规则的选择，但文献中很少有令人信服的例子。基础转换器在这里非常适合，因为它们的自回归生成模拟了这种前向模拟，并且它们的通用设计能够实现丰富的预测建模。我们介绍了TabMGP，这是一种基于TabPFN构建的MGP，TabPFN是表格数据当前最先进的基础模型。TabMGP产生的可信集具有接近名义覆盖范围的性能，并且通常优于现有的MGP结构和标准贝叶斯方法。


### 论文摘要

Bayesian inference provides principled uncertainty quantification but is often limited by challenges of prior elicitation, likelihood misspecification, and computational burden. The martingale posterior (MGP, Fong et al., 2023) offers an alternative, replacing prior-likelihood elicitation with a predictive rule - namely, a sequence of one-step-ahead predictive distributions - for forward data generation. The utility of MGPs depends on the choice of predictive rule, yet the literature has offered few compelling examples. Foundation transformers are well-suited here, as their autoregressive generation mirrors this forward simulation and their general-purpose design enables rich predictive modeling. We introduce TabMGP, an MGP built on TabPFN, a transformer foundation model that is currently state-of-the-art for tabular data. TabMGP produces credible sets with near-nominal coverage and often outperforms both existing MGP constructions and standard Bayes.

---

## 31. POWSM: A Phonetic Open Whisper-Style Speech Foundation Model

**论文链接:** [http://arxiv.org/abs/2510.24992v1](http://arxiv.org/abs/2510.24992v1)

**作者:** Chin-Jou Li, Kalvin Chang, Shikhar Bharadwaj, Eunjung Yeo, Kwanghee Choi, Jian Zhu, David Mortensen, Shinji Watanabe

**发布时间:** 2025-10-28

**备注:** 14 pages, under review

### GPT解析

### 总结

POWSM是一个统一的语音处理框架，能够同时执行多个音位相关任务，性能与专业模型相当，且支持多种转换功能。

### 背景

语音处理领域在自动语音识别(ASR)、音位识别(PR)、字形到音位转换(G2P)和音位到字形转换(P2G)等音位任务上取得了显著进展，但这些任务大多是孤立研究的，每个任务都依赖于特定的架构和数据集。

### 目的

引入POWSM（Phonetic Open Whisper-style Speech Model），创建第一个能够同时执行多个与音位相关任务的统一框架。

### 方法

POWSM模型实现了音频、文本（字形）和音位之间的无缝转换，为通用和低资源语音处理开辟了新的可能性。

### 主要发现

POWSM在性能上与类似大小的专业PR模型（Wav2Vec2Phoneme和ZIPA）相当或更优，同时支持G2P、P2G和ASR。

### 结论

训练数据、代码和模型已公开发布，以促进开放科学。

### 翻译

最近语音处理方面的进展已在语音识别、音位识别、字形到音位转换和音位到字形转换等音位任务中取得实质性进展。尽管这些任务在概念上相似，但它们大多被孤立研究，每个任务都依赖于特定的架构和数据集。在本文中，我们引入了POWSM（Phonetic Open Whisper-style Speech Model），这是第一个能够同时执行多个音位相关任务的统一框架。POWSM实现了音频、文本（字形）和音位之间的无缝转换，为通用和低资源语音处理开辟了新的可能性。我们的模型在性能上与类似大小的专业PR模型（Wav2Vec2Phoneme和ZIPA）相当或更优，同时支持G2P、P2G和ASR。我们的训练数据、代码和模型已公开发布，以促进开放科学。


### 论文摘要

Recent advances in spoken language processing have led to substantial progress in phonetic tasks such as automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G). Despite their conceptual similarity, these tasks have largely been studied in isolation, each relying on task-specific architectures and datasets. In this paper, we introduce POWSM (Phonetic Open Whisper-style Speech Model), the first unified framework capable of jointly performing multiple phone-related tasks. POWSM enables seamless conversion between audio, text (graphemes), and phones, opening up new possibilities for universal and low-resource speech processing. Our model outperforms or matches specialized PR models of similar size (Wav2Vec2Phoneme and ZIPA) while jointly supporting G2P, P2G, and ASR. Our training data, code and models are released to foster open science.

---

## 32. Pearl: A Foundation Model for Placing Every Atom in the Right Location

**论文链接:** [http://arxiv.org/abs/2510.24670v2](http://arxiv.org/abs/2510.24670v2)

**作者:** Genesis Research Team, Alejandro Dobles, Nina Jovic, Kenneth Leidal, Pranav Murugan, David C. Williams, Drausin Wulsin, Nate Gruver, Christina X. Ji, Korrawat Pruegsanusak, Gianluca Scarpellini, Ansh Sharma, Wojciech Swiderski, Andrea Bootsma, Richard Strong Bowen, Charlotte Chen, Jamin Chen, Marc André Dämgen, Benjamin DiFrancesco, J. D. Fishman, Alla Ivanova, Zach Kagin, David Li-Bland, Zuli Liu, Igor Morozov, Jeffrey Ouyang-Zhang, Frank C. Pickard IV, Kushal S. Shah, Ben Shor, Gabriel Monteiro da Silva, Roy Tal, Maxx Tessmer, Carl Tilbury, Cyr Vetcher, Daniel Zeng, Maruan Al-Shedivat, Aleksandra Faust, Evan N. Feinberg, Michael V. LeVine, Matteus Pan

**发布时间:** 2025-10-28

**备注:** technical report

### GPT解析

### 总结

Pearl是一个用于蛋白质-配体共同折叠的基础模型，通过创新的训练方法、架构设计和推理控制，显著提高了蛋白质-配体复合物结构预测的准确性。

### 背景

准确预测蛋白质-配体复合物的三维结构是计算药物发现中的基本挑战，限制了治疗设计的速度和成功。深度学习方法虽有潜力，但受限于实验数据稀缺、架构效率低下、物理无效构象以及辅助信息利用能力有限等因素。

### 目的

开发一个能够克服当前蛋白质-配体结构预测局限性的基础模型，提高预测的准确性和可靠性。

### 方法

Pearl通过三个关键创新解决挑战：(1)包含大规模合成数据的训练配方，克服数据稀缺；(2)融合SO(3)-等变扩散模块的架构，尊重3D旋转对称性，提高泛化能力；(3)可控推理系统，支持蛋白质和非聚合物组分以及无条件/条件模式。

### 主要发现

Pearl在蛋白质-配体共同折叠方面建立了新性能标准，在公共基准测试中超越AlphaFold3和其他开源模型14%以上；在口袋条件共同折叠模式下，对真实世界药物目标实现了3.6倍的改进；模型性能与训练中使用的合成数据集大小直接相关。

### 结论

Pearl通过创新的训练方法、架构设计和推理控制，显著提高了蛋白质-配体复合物结构预测的准确性，为药物设计提供了更强大的工具。

### 翻译

准确预测蛋白质-配体复合物的三维结构仍然是计算药物发现中的一个基本挑战，它限制了治疗设计的速度和成功率。深度学习方法最近作为结构预测工具显示出强大的潜力，在各种生物分子系统中取得了有希望的准确性。然而，它们的性能和实用性受到实验数据稀缺、架构效率低下、物理无效构象以及在推理过程中利用辅助信息能力有限等因素的制约。为解决这些问题，我们引入了Pearl（Placing Every Atom in the Right Location），一个用于大规模蛋白质-配体共同折叠的基础模型。Pearl通过三个关键创新来解决这些挑战：(1)包含大规模合成数据的训练配方，以克服数据稀缺；(2)融合SO(3)-等变扩散模块的架构， inherently尊重3D旋转对称性，提高泛化能力和样本效率；(3)可控推理，包括支持蛋白质和非聚合物组分以及无条件/条件模式的双链通用模板系统。Pearl在蛋白质-配体共同折叠方面建立了新的最先进性能。在生成准确（RMSD < 2Å）和物理有效构象的关键指标上，Pearl在公共Runs N' Poses和PoseBusters基准测试中超越了AlphaFold3和其他开源基线，比次优模型分别提高了14.5%和14.2%。在口袋条件共同折叠模式下，Pearl在一个具有挑战性的真实世界药物目标专有集上，在更严格的RMSD < 1Å阈值下实现了3.6倍的改进。最后，我们证明了模型性能与训练中使用的合成数据集大小直接相关。


### 论文摘要

Accurately predicting the three-dimensional structures of protein-ligand complexes remains a fundamental challenge in computational drug discovery that limits the pace and success of therapeutic design. Deep learning methods have recently shown strong potential as structural prediction tools, achieving promising accuracy across diverse biomolecular systems. However, their performance and utility are constrained by scarce experimental data, inefficient architectures, physically invalid poses, and the limited ability to exploit auxiliary information available at inference. To address these issues, we introduce Pearl (Placing Every Atom in the Right Location), a foundation model for protein-ligand cofolding at scale. Pearl addresses these challenges with three key innovations: (1) training recipes that include large-scale synthetic data to overcome data scarcity; (2) architectures that incorporate an SO(3)-equivariant diffusion module to inherently respect 3D rotational symmetries, improving generalization and sample efficiency, and (3) controllable inference, including a generalized multi-chain templating system supporting both protein and non-polymeric components as well as dual unconditional/conditional modes. Pearl establishes a new state-of-the-art performance in protein-ligand cofolding. On the key metric of generating accurate (RMSD < 2 \r{A}) and physically valid poses, Pearl surpasses AlphaFold 3 and other open source baselines on the public Runs N' Poses and PoseBusters benchmarks, delivering 14.5% and 14.2% improvements, respectively, over the next best model. In the pocket-conditional cofolding regime, Pearl delivers $3.6\times$ improvement on a proprietary set of challenging, real-world drug targets at the more rigorous RMSD < 1 \r{A} threshold. Finally, we demonstrate that model performance correlates directly with synthetic dataset size used in training.

---

## 33. Compiler.next: A Search-Based Compiler to Power the AI-Native Future of Software Engineering

**论文链接:** [http://arxiv.org/abs/2510.24799v1](http://arxiv.org/abs/2510.24799v1)

**作者:** Filipe R. Cogo, Gustavo A. Oliva, Ahmed E. Hassan

**发布时间:** 2025-10-27

**备注:** 31 pages, 5 figures, submitted to ACM Transactions on Software  Engineering and Methodology

### GPT解析

### 总结

本文提出Compiler.next，一种基于搜索的新型编译器，旨在解决AI辅助软件工程中的认知过载、工具集成效率低和AI副驾驶功能有限等问题，实现AI原生软件系统的无缝演进。

### 背景

AI辅助软件工程快速发展，但现有工具和范式受认知过载、工具集成效率低下和AI副驾驶功能有限等因素的限制。

### 目的

提出Compiler.next作为软件工程3.0时代的一部分，实现AI原生软件系统的无缝演进，降低非专家技术门槛，实现可扩展、适应性强和可靠的AI驱动软件。

### 方法

Compiler.next接受人类编写的意图，通过搜索最优解决方案自动生成工作软件，涉及认知架构及其组成部分的动态优化，在准确性、成本和延迟等多个目标间找到最佳平衡。

### 主要发现

Compiler.next的架构设计使其能够作为降低非专家技术门槛、实现可扩展、适应性强和可靠的AI驱动软件的基石。

### 结论

Compiler.next为完全自动化、搜索驱动的软件开发奠定了基础，促进了更快创新和更高效的AI驱动系统，解决了意图编译的核心挑战。

### 翻译

AI辅助软件工程的快速发展为软件工程领域带来了变革潜力，但现有工具和范式仍然受到认知过载、工具集成效率低下以及AI副驾驶功能有限等因素的限制。为此，我们提出了Compiler.next，一种基于搜索的新型编译器，作为新兴软件工程3.0时代的一部分，旨在实现AI原生软件系统的无缝演进。与传统的静态编译器不同，Compiler.next接受人类编写的意图，并通过搜索最优解决方案来自动生成工作软件。这一过程涉及认知架构及其组成部分（如提示、基础模型配置和系统参数）的动态优化，同时找到准确性、成本和延迟等多个目标之间的最佳权衡。本文概述了Compiler.next的架构，并将其定位为降低非专家技术门槛、实现可扩展、适应性强和可靠的AI驱动软件的基石。我们提出了一个路线图来解决意图编译中的核心挑战，包括开发高质量编程构造、有效搜索启发式方法、可复现性以及编译器之间的互操作性。我们的愿景为完全自动化、搜索驱动的软件开发奠定了基础，促进了更快创新和更高效的AI驱动系统。


### 论文摘要

The rapid advancement of AI-assisted software engineering has brought transformative potential to the field of software engineering, but existing tools and paradigms remain limited by cognitive overload, inefficient tool integration, and the narrow capabilities of AI copilots. In response, we propose Compiler.next, a novel search-based compiler designed to enable the seamless evolution of AI-native software systems as part of the emerging Software Engineering 3.0 era. Unlike traditional static compilers, Compiler.next takes human-written intents and automatically generates working software by searching for an optimal solution. This process involves dynamic optimization of cognitive architectures and their constituents (e.g., prompts, foundation model configurations, and system parameters) while finding the optimal trade-off between several objectives, such as accuracy, cost, and latency. This paper outlines the architecture of Compiler.next and positions it as a cornerstone in democratizing software development by lowering the technical barrier for non-experts, enabling scalable, adaptable, and reliable AI-powered software. We present a roadmap to address the core challenges in intent compilation, including developing quality programming constructs, effective search heuristics, reproducibility, and interoperability between compilers. Our vision lays the groundwork for fully automated, search-driven software development, fostering faster innovation and more efficient AI-driven systems.

---

## 34. Why Foundation Models in Pathology Are Failing

**论文链接:** [http://arxiv.org/abs/2510.23807v2](http://arxiv.org/abs/2510.23807v2)

**作者:** Hamid R. Tizhoosh

**发布时间:** 2025-10-27

### GPT解析

### 总结

本文探讨了基础模型在计算病理学应用中的不足，指出其存在诊断准确率低、鲁棒性差等问题，并分析了这些问题背后的七个相互关联原因，认为当前病理学基础模型在概念上与组织形态学本质不匹配，需要范式重构。

### 背景

在非医疗领域，基础模型通过大规模自监督和多模态学习彻底改变了计算机视觉和语言处理，人们预期其在计算病理学领域也能取得类似突破。

### 目的

检查基础模型在计算病理学应用中的不足，分析这些缺点背后的根本原因。

### 方法

通过系统评估方法检查基础模型的缺点，并分析这些缺点背后的原因。

### 主要发现

基础模型存在诊断准确率低、鲁棒性差、几何不稳定性、计算需求量大以及安全漏洞等问题；这些问题源于七个相互关联的原因：生物复杂性、无效的自监督、过度泛化、过度的架构复杂性、缺乏领域特定创新、数据不足以及与组织块大小相关的基本设计缺陷。

### 结论

当前病理学基础模型在概念上与组织形态学本质不匹配，需要对范式本身进行根本性的重新思考。

### 翻译

在非医疗领域，基础模型通过大规模自监督和多模态学习彻底改变了计算机视觉和语言处理。因此，计算病理学领域对这类模型的快速应用预期能在癌症诊断、预后判断和多模态检索方面带来类似的突破。然而，最近的系统评估揭示了根本性弱点：低诊断准确率、差鲁棒性、几何不稳定性、高计算需求以及令人担忧的安全漏洞。这篇短文检查了这些不足，并认为它们源于主流人工智能中通用基础建模的基本假设与人体组织内在复杂性之间的更深层次概念不匹配。确定了七个相互关联的原因：生物复杂性、无效的自监督、过度泛化、过度的架构复杂性、缺乏领域特定创新、数据不足以及与组织块大小相关的基本设计缺陷。这些发现表明，当前的病理学基础模型在概念上仍与组织形态学的本质不匹配，需要对范式本身进行根本性的重新思考。


### 论文摘要

In non-medical domains, foundation models (FMs) have revolutionized computer vision and language processing through large-scale self-supervised and multimodal learning. Consequently, their rapid adoption in computational pathology was expected to deliver comparable breakthroughs in cancer diagnosis, prognostication, and multimodal retrieval. However, recent systematic evaluations reveal fundamental weaknesses: low diagnostic accuracy, poor robustness, geometric instability, heavy computational demands, and concerning safety vulnerabilities. This short paper examines these shortcomings and argues that they stem from deeper conceptual mismatches between the assumptions underlying generic foundation modeling in mainstream AI and the intrinsic complexity of human tissue. Seven interrelated causes are identified: biological complexity, ineffective self-supervision, overgeneralization, excessive architectural complexity, lack of domain-specific innovation, insufficient data, and a fundamental design flaw related to tissue patch size. These findings suggest that current pathology foundation models remain conceptually misaligned with the nature of tissue morphology and call for a fundamental rethinking of the paradigm itself.

---

## 35. DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans

**论文链接:** [http://arxiv.org/abs/2510.14205v3](http://arxiv.org/abs/2510.14205v3)

**作者:** Bingsheng Yao, Bo Sun, Yuanzhe Dong, Yuxuan Lu, Dakuo Wang

**发布时间:** 2025-10-16

**备注:** In Submission

### GPT解析

### 总结

本研究提出动态人格完善框架(DPRF)，用于提高大型语言模型角色扮演代理的行为与目标个体的一致性，通过迭代识别认知分歧并优化个人资料实现。

### 背景

大型语言模型角色扮演代理旨在模拟个体人类行为，但手动创建的个人资料(如精心挑选的信息和个性特征)未经验证与目标个体的对齐度，导致人格保真度受损。

### 目的

优化LLM RPAs的行为与目标个体行为的一致性，提高角色扮演的准确性和可靠性。

### 方法

DPRF通过迭代识别生成行为与人类真实认知之间的分歧(无论是自由形式还是基于理论的结构化分析)，并完善个人资料以减轻这些分歧。

### 主要发现

在五个大型语言模型和四种多样的行为预测场景(正式辩论、涉及心理健康问题的社交媒体帖子、公开采访和电影评论)中，DPRF能够显著提高行为一致性，并且跨模型和场景具有通用性。

### 结论

该工作为创建高保真度个人资料和增强下游应用(如用户模拟、社会研究和个性化AI)的有效性提供了稳健的方法论。

### 翻译

新兴的大型语言模型角色扮演代理旨在模拟个体人类行为，但人格保真度常因手动创建的个人资料(如精心挑选的信息和个性特征)未经验证与目标个体的对齐度而受损。为解决这一局限，我们的工作引入了动态人格完善框架(DPRF)。DPRF旨在通过迭代识别生成行为与人类真实认知之间的分歧(无论是自由形式还是基于理论的结构化分析)，并完善个人资料以减轻这些分歧，从而优化LLM RPAs的行为与目标个体行为的一致性。我们在五个大型语言模型和四种多样的行为预测场景(正式辩论、涉及心理健康问题的社交媒体帖子、公开采访和电影评论)中评估了DPRF。DPRF能够显著提高行为一致性，并且跨模型和场景具有通用性。我们的工作为创建高保真度个人资料和增强下游应用(如用户模拟、社会研究和个性化AI)的有效性提供了稳健的方法论。


### 论文摘要

The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF). DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences. We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews. DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios. Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.

---

## 36. Graph Network-based Structural Simulator: Graph Neural Networks for Structural Dynamics

**论文链接:** [http://arxiv.org/abs/2510.25683v1](http://arxiv.org/abs/2510.25683v1)

**作者:** Alessandro Lucchetti, Francesco Cadini, Marco Giglio, Luca Lomazzi

**发布时间:** 2025-10-29

**备注:** 16 pages, 14 figures

### GPT解析

### 总结

论文介绍了一种名为GNSS的图神经网络框架，用于动态结构问题的代理建模，通过三个关键特征解决了现有方法的局限性，在案例研究中表现出色，实现了比传统方法更快的推理速度。

### 背景

图神经网络作为数值模拟的代理模型已在计算流体动力学领域有所研究，但在结构问题特别是动态情况中的应用相对较少，存在研究空白。

### 目的

为了填补动态结构问题中图神经网络应用的空白，作者开发了GNSS框架，专门用于动态结构问题的代理建模。

### 方法

GNSS遵循编码-处理-解码范式，具有三个关键特征：在节点固定的局部框架中表达节点运动学；采用符号感知回归损失减少相位误差；使用波长感知的连接半径优化图结构构建。

### 主要发现

GNSS在50kHz汉宁调制脉冲激励梁的案例研究中，能够准确复现物理特性并泛化到未见过的加载条件，而现有GNN方法无法收敛。与有限元方法相比，GNSS实现了显著的推理加速同时保持空间和时间保真度。

### 结论

具有物理一致性更新规则且保持局部性的图神经网络是动态、波主导结构模拟的有竞争力的替代方案。

### 翻译

图神经网络最近被探索作为数值模拟的代理模型。虽然它们在计算流体动力学中的应用已被研究，但很少被应用于结构问题，特别是动态情况。为了解决这一研究空白，我们引入了基于图网络的结构模拟器，这是一个用于动态结构问题代理建模的图神经网络框架。GNSS遵循基于GNN的机器学习模型的典型编码-处理-解码范式，其设计使其特别适合动态模拟，这得益于三个关键特征：在节点固定的局部框架中表达节点运动学，避免有限差分速度中的灾难性取消；采用符号感知回归损失，减少长期rollout中的相位误差；使用波长感知的连接半径，优化图结构构建。我们在一个涉及由50kHz汉宁调制脉冲激励的梁的案例研究中评估了GNSS。结果表明GNSS能够在数百个时间步长内准确复现问题的物理特性，并能泛化到未见过的加载条件，而现有的GNN方法无法收敛或提供有意义的预测。与显式有限元基线方法相比，GNSS在保持空间和时间保真度的同时实现了显著的推理加速。这些发现表明，具有物理一致性更新规则且保持局部性的GNN是动态、波主导结构模拟的有竞争力的替代方案。


### 论文摘要

Graph Neural Networks (GNNs) have recently been explored as surrogate models for numerical simulations. While their applications in computational fluid dynamics have been investigated, little attention has been given to structural problems, especially for dynamic cases. To address this gap, we introduce the Graph Network-based Structural Simulator (GNSS), a GNN framework for surrogate modeling of dynamic structural problems.   GNSS follows the encode-process-decode paradigm typical of GNN-based machine learning models, and its design makes it particularly suited for dynamic simulations thanks to three key features: (i) expressing node kinematics in node-fixed local frames, which avoids catastrophic cancellation in finite-difference velocities; (ii) employing a sign-aware regression loss, which reduces phase errors in long rollouts; and (iii) using a wavelength-informed connectivity radius, which optimizes graph construction.   We evaluate GNSS on a case study involving a beam excited by a 50kHz Hanning-modulated pulse. The results show that GNSS accurately reproduces the physics of the problem over hundreds of timesteps and generalizes to unseen loading conditions, where existing GNNs fail to converge or deliver meaningful predictions.   Compared with explicit finite element baselines, GNSS achieves substantial inference speedups while preserving spatial and temporal fidelity. These findings demonstrate that locality-preserving GNNs with physics-consistent update rules are a competitive alternative for dynamic, wave-dominated structural simulations.

---

## 37. Bridging the Divide: End-to-End Sequence-Graph Learning

**论文链接:** [http://arxiv.org/abs/2510.25126v1](http://arxiv.org/abs/2510.25126v1)

**作者:** Yuen Chen, Yulun Wu, Samuel Sharpe, Igor Melnyk, Nam H. Nguyen, Furong Huang, C. Bayan Bruss, Rizal Fathony

**发布时间:** 2025-10-29

### GPT解析

### 总结

BRIDGE是一种统一的端到端架构，能够联合学习序列和图信息，在友谊预测和欺诈检测任务上表现优于现有方法。

### 背景

现实世界的数据集通常是序列性和关系性的，每个节点携带事件序列，边编码交互。现有方法往往只考虑一种模态而忽略另一种。

### 目的

作者认为序列和图是同一数据集的互补方面，应该联合学习，而不是作为独立问题处理。

### 方法

BRIDGE将序列编码器与图神经网络(GNN)耦合在单一目标下，允许梯度在两个模块间流动。添加了TOKENXATTN标记级交叉注意力层，实现邻居间细粒度的标记级消息传递。

### 主要发现

在友谊预测(Brightkite)和欺诈检测(Amazon)两种场景下，BRIDGE在排序和分类指标上始终优于静态GNN、时图方法和仅基于序列的基线。

### 结论

BRIDGE通过联合学习序列和图信息，能够学习任务对齐的表示，在各种任务上表现优异。

### 翻译

许多现实世界的数据集既是序列性的又是关系性的：每个节点携带事件序列，而边则编码交互。现有的序列建模和图建模方法通常忽略了一种或另一种模态。我们认为序列和图不是独立的问题，而是同一数据集的互补方面，应该联合学习。我们引入了BRIDGE，一个统一的端到端架构，将序列编码器与GNN在单一目标下耦合，允许梯度在两个模块间流动，并学习任务对齐的表示。为了实现邻居间细粒度的标记级消息传递，我们添加了TOKENXATTN，一个标记级交叉注意力层，用于在相邻序列的事件之间传递消息。在友谊预测(Brightkite)和欺诈检测(Amazon)两种设置下，BRIDGE在排序和分类指标上始终优于静态GNN、时图方法和仅基于序列的基线。


### 论文摘要

Many real-world datasets are both sequential and relational: each node carries an event sequence while edges encode interactions. Existing methods in sequence modeling and graph modeling often neglect one modality or the other. We argue that sequences and graphs are not separate problems but complementary facets of the same dataset, and should be learned jointly. We introduce BRIDGE, a unified end-to-end architecture that couples a sequence encoder with a GNN under a single objective, allowing gradients to flow across both modules and learning task-aligned representations. To enable fine-grained token-level message passing among neighbors, we add TOKENXATTN, a token-level cross-attention layer that passes messages between events in neighboring sequences. Across two settings, friendship prediction (Brightkite) and fraud detection (Amazon), BRIDGE consistently outperforms static GNNs, temporal graph methods, and sequence-only baselines on ranking and classification metrics.

---

## 38. The Underappreciated Power of Vision Models for Graph Structural Understanding

**论文链接:** [http://arxiv.org/abs/2510.24788v1](http://arxiv.org/abs/2510.24788v1)

**作者:** Xinjian Zhao, Wei Pang, Zhongkai Xue, Xiangru Jian, Lei Zhang, Yaoyao Xu, Xiaozhuang Song, Shu Wu, Tianshu Yu

**发布时间:** 2025-10-27

**备注:** NeurIPS 2025

### GPT解析

### 总结

论文研究了图神经网络与人类视觉感知的差异，发现视觉模型在图理解任务中具有与GNNs相当的性能但展现出不同的学习模式。作者提出了新的基准测试GraphAbstract，评估模型对全局图属性的理解能力，结果表明视觉模型在需要整体结构理解的任务上优于GNNs。

### 背景

图神经网络通过自下而上的消息传递机制工作，这与人类视觉感知先捕捉全局结构的直觉方式有根本不同。现有基准测试将领域特征与拓扑理解混为一谈，无法有效评估模型对图全局结构的理解能力。

### 目的

研究视觉模型在图理解方面的潜力，评估它们与GNNs在性能和学习模式上的差异，并开发新的基准测试来评估模型对全局图属性的理解能力。

### 方法

提出了名为GraphAbstract的新基准测试，评估模型识别组织原型、检测对称性、感知连接强度和识别关键元素的能力，这些能力与人类对图的全局理解方式相似。

### 主要发现

视觉模型在需要整体结构理解的任务上显著优于GNNs；视觉模型在不同图规模上保持泛化能力；GNNs在全局模式抽象方面存在困难，且随着图规模增大性能下降；视觉模型具有显著的但未被充分利用的图结构理解能力。

### 结论

视觉模型在需要全局拓扑意识和尺度不变推理的问题上具有显著的能力，这些发现为开发更有效的图基础模型开辟了新途径，特别是对于那些由整体模式识别主导的任务。

### 翻译

图神经网络通过自下而上的消息传递运行，这与人类视觉感知有根本不同，人类视觉直觉上首先捕捉全局结构。我们研究了视觉模型在图理解方面的未被充分认识到的潜力，发现它们在既定基准上实现了与GNNs相当的性能，同时表现出明显不同的学习模式。这些不同的行为，加上现有基准将领域特征与拓扑理解混为一谈的限制，促使我们引入GraphAbstract。这个基准评估模型像人类一样感知全局图属性的能力：识别组织原型、检测对称性、感知连接强度和识别关键元素。我们的结果显示，在需要整体结构理解的任务上，视觉模型显著优于GNNs，并且在不同图规模上保持泛化能力，而GNNs在全局模式抽象方面存在困难，且随着图规模增大性能下降。这项工作表明，视觉模型具有显著的但未被充分利用的图结构理解能力，特别是对于需要全局拓扑意识和尺度不变推理的问题。这些发现为利用这种未被充分认识到的潜力开发更有效的图基础模型开辟了新途径，这些任务主要由整体模式识别主导。


### 论文摘要

Graph Neural Networks operate through bottom-up message-passing, fundamentally differing from human visual perception, which intuitively captures global structures first. We investigate the underappreciated potential of vision models for graph understanding, finding they achieve performance comparable to GNNs on established benchmarks while exhibiting distinctly different learning patterns. These divergent behaviors, combined with limitations of existing benchmarks that conflate domain features with topological understanding, motivate our introduction of GraphAbstract. This benchmark evaluates models' ability to perceive global graph properties as humans do: recognizing organizational archetypes, detecting symmetry, sensing connectivity strength, and identifying critical elements. Our results reveal that vision models significantly outperform GNNs on tasks requiring holistic structural understanding and maintain generalizability across varying graph scales, while GNNs struggle with global pattern abstraction and degrade with increasing graph size. This work demonstrates that vision models possess remarkable yet underutilized capabilities for graph structural understanding, particularly for problems requiring global topological awareness and scale-invariant reasoning. These findings open new avenues to leverage this underappreciated potential for developing more effective graph foundation models for tasks dominated by holistic pattern recognition.

---

## 39. FastJAM: a Fast Joint Alignment Model for Images

**论文链接:** [http://arxiv.org/abs/2510.22842v2](http://arxiv.org/abs/2510.22842v2)

**作者:** Omri Hirsch, Ron Shapira Weber, Shira Ifergane, Oren Freifeld

**发布时间:** 2025-10-26

**备注:** Accepted to NeurIPS 2025. Pages 1-10 are the Main Paper. Pages 23-31  are Supplemental Material. FastJAM website -  https://bgu-cs-vil.github.io/FastJAM/

### GPT解析

### 总结

本研究提出了一种名为FastJAM的快速图像联合对齐方法，显著降低计算复杂度，实现高质量对齐效果。

### 背景

现有图像联合对齐方法通常需要长时间训练、大容量模型和大量超参数调整，计算效率低下。

### 目的

开发一种快速、高效的图像联合对齐方法，减少计算时间并提高对齐质量。

### 方法

FastJAM基于图方法，利用现成图像匹配器计算pairwise匹配，结合快速非参数聚类构建表示图像内和图像间关键点关系的图。通过图神经网络传播和聚合对应关系，利用图像级池化预测单应性参数。采用逆组合损失消除正则化项需求，避免超参数调整。

### 主要发现

实验结果表明，FastJAM在多个基准测试上实现了优于现有现代JA方法的对齐质量，同时将计算时间从小时或分钟级减少到几秒钟。

### 结论

FastJAM通过创新的图神经网络和逆组合损失方法，实现了快速、高效的图像联合对齐，为图像处理领域提供了新的解决方案。

### 翻译

图像联合对齐（JA）旨在将一组图像对齐到统一的坐标系中，使语义相似的特征出现在对应的空间位置。大多数现有方法通常需要长时间训练、大容量模型和大量超参数调整。我们引入了FastJAM，一种快速的基于图的方法，显著降低了联合对齐任务的计算复杂度。FastJAM利用现成的图像匹配器计算的pairwise匹配，结合快速非参数聚类，构建表示图像内和图像间关键点关系的图。图神经网络传播和聚合这些对应关系，通过图像级池化有效预测每个图像的单应性参数。利用逆组合损失，消除了对预测变换的正则化项的需求（因此也避免了与这些项相关的超参数调整），FastJAM能够快速有效地执行图像JA。在几个基准测试上的实验结果表明，FastJAM在对齐质量方面优于现有的现代JA方法，同时将计算时间从小时或分钟减少到几秒钟。我们的代码可在项目网页获取：https://bgu-cs-vil.github.io/FastJAM/


### 论文摘要

Joint Alignment (JA) of images aims to align a collection of images into a unified coordinate frame, such that semantically-similar features appear at corresponding spatial locations. Most existing approaches often require long training times, large-capacity models, and extensive hyperparameter tuning. We introduce FastJAM, a rapid, graph-based method that drastically reduces the computational complexity of joint alignment tasks. FastJAM leverages pairwise matches computed by an off-the-shelf image matcher, together with a rapid nonparametric clustering, to construct a graph representing intra- and inter-image keypoint relations. A graph neural network propagates and aggregates these correspondences, efficiently predicting per-image homography parameters via image-level pooling. Utilizing an inverse-compositional loss, that eliminates the need for a regularization term over the predicted transformations (and thus also obviates the hyperparameter tuning associated with such terms), FastJAM performs image JA quickly and effectively. Experimental results on several benchmarks demonstrate that FastJAM achieves results better than existing modern JA methods in terms of alignment quality, while reducing computation time from hours or minutes to mere seconds. Our code is available at our project webpage, https://bgu-cs-vil.github.io/FastJAM/

---

